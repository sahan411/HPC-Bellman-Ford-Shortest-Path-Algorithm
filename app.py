from flask import Flask, jsonify, render_template, request
import csv
import glob
import multiprocessing
import os
import re
import subprocess
import tempfile

try:
    import psutil
except ImportError:
    psutil = None


app = Flask(__name__)

BIN_DIR = "bin"
GRAPHS_DIR = "graphs"
RESULTS_DIR = "results"
MPIEXEC = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"


def abs_path(*parts):
    return os.path.abspath(os.path.join(*parts))


UI_RUN_ROOT = abs_path(".ui_runs")


def query_gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"available": False, "error": result.stderr.strip() or "nvidia-smi failed"}

        gpus = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mb": parts[2],
                    "memory_used_mb": parts[3],
                    "utilization_percent": parts[4],
                })
        return {"available": bool(gpus), "gpus": gpus}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/system-resources", methods=["GET"])
def get_system_resources():
    try:
        cpu_count = multiprocessing.cpu_count()
        memory_info = {}

        if psutil is not None:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_info = {
                "memory_total_gb": round(memory.total / (1024 ** 3), 2),
                "memory_available_gb": round(memory.available / (1024 ** 3), 2),
                "memory_percent": memory.percent,
            }
        else:
            cpu_percent = None
            memory_info = {
                "memory_total_gb": None,
                "memory_available_gb": None,
                "memory_percent": None,
            }

        return jsonify({
            "success": True,
            "cpu_cores": cpu_count,
            "cpu_percent": cpu_percent,
            "gpu": query_gpu(),
            **memory_info,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/graphs", methods=["GET"])
def get_graphs():
    try:
        graphs = [
            os.path.basename(path)
            for path in glob.glob(os.path.join(GRAPHS_DIR, "*.txt"))
        ]
        return jsonify({"success": True, "graphs": graphs})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/graph_data", methods=["GET"])
def get_graph_data():
    graph_file = request.args.get("file")
    if not graph_file:
        return jsonify({"success": False, "error": "No file specified"})

    graph_path = os.path.join(GRAPHS_DIR, graph_file)
    if not os.path.exists(graph_path):
        return jsonify({"success": False, "error": "File not found"})

    nodes = []
    edges = []
    truncated = False
    max_edges = 500

    try:
        with open(graph_path, "r", encoding="utf-8") as file:
            header = file.readline().strip().split()
            if len(header) < 2:
                raise ValueError("Invalid graph header")

            vertices = int(header[0])
            total_edges = int(header[1])
            seen_nodes = set()

            for count, line in enumerate(file):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                src, dest, weight = map(int, parts[:3])
                seen_nodes.add(src)
                seen_nodes.add(dest)
                edges.append({
                    "from": src,
                    "to": dest,
                    "label": str(weight),
                    "arrows": "to",
                })

                if count + 1 >= max_edges:
                    truncated = True
                    break

            for node in sorted(seen_nodes):
                nodes.append({"id": node, "label": str(node)})

        return jsonify({
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "truncated": truncated,
            "totalV": vertices,
            "totalE": total_edges,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


def build_command(algorithm, graph_path, source, threads, processes):
    exe_map = {
        "serial": "bellman_ford_serial.exe",
        "openmp": "bellman_ford_openmp.exe",
        "mpi": "bellman_ford_mpi.exe",
        "hybrid": "bellman_ford_hybrid.exe",
        "mpi_cuda": "bellman_ford_mpi_cuda.exe",
    }

    if algorithm not in exe_map:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    exe_path = abs_path(BIN_DIR, exe_map[algorithm])
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"Implementation not found: {exe_map[algorithm]}")

    if algorithm == "openmp":
        return [exe_path, graph_path, str(source), str(threads)]
    if algorithm == "mpi":
        return [MPIEXEC, "-n", str(processes), exe_path, graph_path, str(source)]
    if algorithm == "hybrid":
        return [MPIEXEC, "-n", str(processes), exe_path, graph_path, str(source), str(threads)]
    if algorithm == "mpi_cuda":
        return [MPIEXEC, "-n", str(processes), exe_path, graph_path, str(source)]
    return [exe_path, graph_path, str(source)]


@app.route("/api/run", methods=["POST"])
def run_algorithm():
    data = request.json or {}
    algorithm = data.get("algorithm")
    graph_file = data.get("graph_file")
    source = int(data.get("source", 0))
    threads = int(data.get("threads", 4))
    processes = int(data.get("processes", 4))
    timeout_sec = int(data.get("timeout", 300))

    if not algorithm or not graph_file:
        return jsonify({"success": False, "error": "Missing algorithm or graph_file"})

    timeout_sec = max(5, min(timeout_sec, 3600))
    cpu_count = multiprocessing.cpu_count()
    warnings = []

    if threads > cpu_count:
        warnings.append(
            f"Requested {threads} threads but system has {cpu_count} logical CPUs. Capping to {cpu_count}."
        )
        threads = cpu_count

    if processes > cpu_count:
        warnings.append(
            f"Requested {processes} processes but system has {cpu_count} logical CPUs. Capping to {cpu_count}."
        )
        processes = cpu_count

    graph_path = abs_path(GRAPHS_DIR, graph_file)
    if not os.path.exists(graph_path):
        return jsonify({"success": False, "error": f"Graph file {graph_file} not found."})

    try:
        command = build_command(algorithm, graph_path, source, threads, processes)
        env = os.environ.copy()
        if algorithm in {"openmp", "hybrid"}:
            env["OMP_NUM_THREADS"] = str(threads)

        os.makedirs(UI_RUN_ROOT, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hpc_ui_run_", dir=UI_RUN_ROOT) as run_dir:
            os.makedirs(os.path.join(run_dir, RESULTS_DIR), exist_ok=True)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=run_dir,
                env=env,
            )

            try:
                stdout_data, stderr_data = process.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout_data, stderr_data = process.communicate()
                return jsonify({
                    "success": False,
                    "error": f"Execution timed out after {timeout_sec}s",
                    "stdout": stdout_data,
                    "stderr": stderr_data,
                    "command": command,
                })

        match = re.search(r"Execution time\s*:\s*([\d.]+)\s*seconds", stdout_data)
        time_elapsed = match.group(1) if match else None

        if process.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"Execution failed with code {process.returncode}",
                "stdout": stdout_data,
                "stderr": stderr_data,
                "command": command,
            })

        return jsonify({
            "success": True,
            "stdout": stdout_data,
            "stderr": stderr_data,
            "time": time_elapsed,
            "warnings": warnings or None,
            "command": command,
        })
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "traceback": traceback.format_exc()})


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    if not os.path.exists(csv_path):
        return jsonify({"success": False, "error": "Benchmark results not found."})

    try:
        results = []
        with open(csv_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                results.append(row)
        return jsonify({"success": True, "data": results})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


if __name__ == "__main__":
    if not os.path.exists(BIN_DIR):
        print(f"Warning: '{BIN_DIR}' directory not found. Compile the project first.")
    if not os.path.exists(GRAPHS_DIR):
        print(f"Warning: '{GRAPHS_DIR}' directory not found. Generate graphs first.")

    app.run(debug=False, host="127.0.0.1", port=5000)
