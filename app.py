from flask import Flask, render_template, request, jsonify
import os
import subprocess
import glob
<<<<<<< HEAD
import csv
=======
import psutil
import multiprocessing
>>>>>>> 199a8666ee01a721a3109eece8ecad271655d22b

app = Flask(__name__)

# Ensure paths are correct based on existing project structure
BIN_DIR = "bin"
GRAPHS_DIR = "graphs"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/system-resources', methods=['GET'])
def get_system_resources():
    """Returns system resource information"""
    try:
        cpu_count = multiprocessing.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return jsonify({
            "success": True,
            "cpu_cores": cpu_count,
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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

@app.route('/api/graph_data', methods=['GET'])
def get_graph_data():
    """Reads a graph file and returns its nodes/edges for visualization"""
    graph_file = request.args.get('file')
    if not graph_file:
         return jsonify({"success": False, "error": "No file specified"})

    graph_path = os.path.join(GRAPHS_DIR, graph_file)
    if not os.path.exists(graph_path):
         return jsonify({"success": False, "error": "File not found"})

    nodes = []
    edges = []
    truncated = False
    MAX_EDGES = 500

    try:
        with open(graph_path, 'r') as f:
            lines = f.readlines()
            
            if not lines:
                raise ValueError("Empty file")

            # First line is V E
            header = lines[0].strip().split()
            V = int(header[0])
            E = int(header[1])

            # Add nodes
            for i in range(V):
                # Don't add a million nodes to UI. Limit to nodes that appear in the first MAX_EDGES
                if i > MAX_EDGES * 2: 
                    break
                nodes.append({"id": i, "label": str(i)})

            # Add edges
            count = 0
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    u, v, w = map(int, parts[:3])
                    edges.append({
                        "from": u,
                        "to": v,
                        "label": str(w),
                        "arrows": "to" # Directed graph
                    })
                    count += 1
                    
                    if count >= MAX_EDGES:
                        truncated = True
                        break

            return jsonify({
                "success": True, 
                "nodes": nodes, 
                "edges": edges, 
                "truncated": truncated,
                "totalV": V,
                "totalE": E
            })

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
    
    # Validate against system resources
    cpu_count = multiprocessing.cpu_count()
    warnings = []
    
    if threads > cpu_count:
        warnings.append(f"Warning: Requesting {threads} threads but system has {cpu_count} CPU cores. Performance may be suboptimal.")
        threads = cpu_count
    
    if processes > cpu_count:
        warnings.append(f"Warning: Requesting {processes} processes but system has {cpu_count} CPU cores. Performance may be suboptimal.")
        processes = cpu_count

    if not algorithm or not graph_file:
        return jsonify({"success": False, "error": "Missing algorithm or graph_file"})

    graph_path = os.path.abspath(os.path.join(GRAPHS_DIR, graph_file))
    if not os.path.exists(graph_path):
        return jsonify({"success": False, "error": f"Graph file {graph_file} not found."})

    process = None
    try:
        import sys
        python_exe = sys.executable
        
        # Map algorithm to implementation
        script_map = {
            'serial': 'bellman_ford_serial.py',
            'openmp': 'bellman_ford_openmp.py',
            'mpi': 'bellman_ford_mpi.py',
            'hybrid': 'bellman_ford_hybrid.py',
            'cuda': 'bellman_ford_cuda.py'
        }
        
        if algorithm not in script_map:
            return jsonify({"success": False, "error": f"Unknown algorithm: {algorithm}"})
        
        script_name = script_map[algorithm]
        script_path = os.path.abspath(os.path.join(BIN_DIR, script_name))
        
        if not os.path.exists(script_path):
            return jsonify({"success": False, "error": f"Implementation not found: {script_name}"})
        
        # Build command based on algorithm
        if algorithm == 'openmp':
            command = [python_exe, script_path, graph_path, str(source), str(threads)]
        else:
            command = [python_exe, script_path, graph_path, str(source)]
        
        # Run the algorithm
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.abspath('.')
        )
        
        try:
            stdout_data, stderr_data = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            return jsonify({"success": False, "error": "Execution timed out (60s limit)"})
        
        # Parse execution time
        time_elapsed = None
        for line in stdout_data.split('\n'):
            if "Execution time:" in line:
                try:
                    time_elapsed = line.split(':')[1].strip().split()[0]
                except:
                    pass
        
        if process.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"Execution failed with code {process.returncode}",
                "stdout": stdout_data,
                "stderr": stderr_data
            })
        
        return jsonify({
            "success": True,
            "stdout": stdout_data,
            "time": time_elapsed,
            "warnings": warnings if warnings else None
        })

    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    """Reads the benchmark results CSV and returns it as JSON"""
    csv_path = os.path.join("results", "benchmark_results.csv")
    if not os.path.exists(csv_path):
        return jsonify({"success": False, "error": "Benchmark results not found."})

    try:
        results = []
        with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        return jsonify({"success": True, "data": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Ensure graphs and bin directories exist, or show warning
    if not os.path.exists(BIN_DIR):
        print(f"Warning: '{BIN_DIR}' directory not found. Compile the project first.")
    if not os.path.exists(GRAPHS_DIR):
        print(f"Warning: '{GRAPHS_DIR}' directory not found. Generate graphs first.")
        
    app.run(debug=False, host='127.0.0.1', port=5000)
