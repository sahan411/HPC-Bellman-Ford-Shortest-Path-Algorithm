from flask import Flask, render_template, request, jsonify
import os
import subprocess
import glob

app = Flask(__name__)

# Ensure paths are correct based on existing project structure
BIN_DIR = "bin"
GRAPHS_DIR = "graphs"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/graphs', methods=['GET'])
def get_graphs():
    """Returns a list of all .txt files in the graphs/ directory"""
    try:
        graphs = []
        for file in glob.glob(os.path.join(GRAPHS_DIR, "*.txt")):
            graphs.append(os.path.basename(file))
        return jsonify({"success": True, "graphs": graphs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/run', methods=['POST'])
def run_algorithm():
    data = request.json
    algorithm = data.get('algorithm')
    graph_file = data.get('graph_file')
    source = data.get('source', 0)
    
    # Optional parameters
    threads = data.get('threads', 4)
    processes = data.get('processes', 4)

    if not algorithm or not graph_file:
        return jsonify({"success": False, "error": "Missing algorithm or graph_file"})

    graph_path = os.path.join(GRAPHS_DIR, graph_file)
    if not os.path.exists(graph_path):
        return jsonify({"success": False, "error": f"Graph file {graph_file} not found."})

    executable = ""
    command = []
    env = os.environ.copy()

    try:
        # Detect correct WSL distribution (avoid docker-desktop)
        distro_arg = []
        try:
            wsl_list = subprocess.check_output(["wsl", "-l", "-q"], text=True, encoding='utf-16le', errors='ignore')
            for line in wsl_list.splitlines():
                distro = line.strip('\x00').strip()
                if distro and "docker" not in distro.lower():
                    distro_arg = ["-d", distro]
                    break
        except Exception:
            pass

        # Convert Windows paths to WSL format (e.g., F:\HPC\... to /mnt/f/HPC/...)
        def wsl_path(win_path):
            if not win_path: return win_path
            path = win_path.replace('\\', '/')
            if ':' in path:
                drive, rest = path.split(':', 1)
                path = f"/mnt/{drive.lower()}{rest}"
            return path

        wsl_graph_path = wsl_path(os.path.abspath(graph_path))
        wsl_bin_dir = wsl_path(os.path.abspath(BIN_DIR))

        if algorithm == 'serial':
            executable = f"{wsl_bin_dir}/bellman_ford_serial"
            command = ["wsl"] + distro_arg + [executable, wsl_graph_path, str(source)]
        
        elif algorithm == 'openmp':
            executable = f"{wsl_bin_dir}/bellman_ford_openmp"
            command = ["wsl"] + distro_arg + ["bash", "-c", f"export OMP_NUM_THREADS={threads} && {executable} {wsl_graph_path} {source}"]
            
        elif algorithm == 'mpi':
            executable = f"{wsl_bin_dir}/bellman_ford_mpi"
            command = ["wsl"] + distro_arg + ["mpiexec", "-np", str(processes), executable, wsl_graph_path, str(source)]
            
        elif algorithm == 'hybrid':
            executable = f"{wsl_bin_dir}/bellman_ford_hybrid"
            command = ["wsl"] + distro_arg + ["bash", "-c", f"export OMP_NUM_THREADS={threads} && mpiexec -np {processes} {executable} {wsl_graph_path} {source}"]
            
        elif algorithm == 'cuda':
            executable = f"{wsl_bin_dir}/bellman_ford_cuda"
            command = ["wsl"] + distro_arg + [executable, wsl_graph_path, str(source)]
            
        else:
            return jsonify({"success": False, "error": f"Unknown algorithm: {algorithm}"})

        # Remove direct OS binary checks as Windows won't natively see Linux binaries without extension correctly
        # We rely on WSL returning an error if the binary is missing

        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout_data, stderr_data = process.communicate(timeout=60) # 60 second timeout
        
        if process.returncode != 0 and stderr_data:
             return jsonify({
                "success": False, 
                "error": f"Execution failed with code {process.returncode}",
                "stderr": stderr_data,
                "stdout": stdout_data
            })

        # Basic parsing to extract execution time out of standard output
        time_elapsed = None
        for line in stdout_data.split('\n'):
            if "Execution time" in line:
                try:
                    time_elapsed = line.split(':')[1].strip().split(' ')[0]
                except:
                    pass
        
        return jsonify({
            "success": True,
            "stdout": stdout_data,
            "time": time_elapsed
        })

    except subprocess.TimeoutExpired:
        process.kill()
        return jsonify({"success": False, "error": "Execution timed out (60s limit)"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Ensure graphs and bin directories exist, or show warning
    if not os.path.exists(BIN_DIR):
        print(f"Warning: '{BIN_DIR}' directory not found. Compile the project first.")
    if not os.path.exists(GRAPHS_DIR):
        print(f"Warning: '{GRAPHS_DIR}' directory not found. Generate graphs first.")
        
    app.run(debug=True, host='127.0.0.1', port=5000)
