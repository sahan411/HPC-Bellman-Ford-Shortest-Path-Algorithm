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
        if algorithm == 'serial':
            executable = os.path.join(BIN_DIR, "bellman_ford_serial").replace('\\', '/')
            command = ["wsl", "./" + executable, graph_path, str(source)]
        
        elif algorithm == 'openmp':
            executable = os.path.join(BIN_DIR, "bellman_ford_openmp").replace('\\', '/')
            env['OMP_NUM_THREADS'] = str(threads)
            command = ["wsl", "./" + executable, graph_path, str(source)]
            
        elif algorithm == 'mpi':
            executable = os.path.join(BIN_DIR, "bellman_ford_mpi").replace('\\', '/')
            command = ["wsl", "mpiexec", "-np", str(processes), "./" + executable, graph_path, str(source)]
            
        elif algorithm == 'hybrid':
            executable = os.path.join(BIN_DIR, "bellman_ford_hybrid").replace('\\', '/')
            env['OMP_NUM_THREADS'] = str(threads)
            command = ["wsl", "mpiexec", "-np", str(processes), "./" + executable, graph_path, str(source)]
            
        elif algorithm == 'cuda':
            executable = os.path.join(BIN_DIR, "bellman_ford_cuda").replace('\\', '/')
            command = ["wsl", "./" + executable, graph_path, str(source)]
            
        else:
            return jsonify({"success": False, "error": f"Unknown algorithm: {algorithm}"})

        # Remove direct OS binary checks as Windows won't natively see Linux binaries without extension correctly
        # We rely on WSL returning an error if the binary is missing

        # Run the command
        # Note: environment variables (like OMP_NUM_THREADS) need to be passed into WSL
        # To do this safely, we export the variable before running the command
        if algorithm in ['openmp', 'hybrid']:
            wsl_cmd_string = f"export OMP_NUM_THREADS={threads} && " + " ".join(command[1:])
            command = ["wsl", "bash", "-c", wsl_cmd_string]
        
        # When sending paths to WSL, they need to use forward slashes. 
        # Python's replace('\\', '/') handles Windows paths.
        graph_path = graph_path.replace('\\', '/')
        if algorithm not in ['openmp', 'hybrid']:
            for i, arg in enumerate(command):
                command[i] = command[i].replace('\\', '/')

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
