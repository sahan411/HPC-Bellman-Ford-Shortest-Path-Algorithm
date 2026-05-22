import subprocess
import os

graph_path = os.path.abspath("graphs/small.txt")
bin_dir = os.path.abspath("bin")

def get_wsl_path(win_path):
    proc = subprocess.Popen(["wsl", "wslpath", "-a", win_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if proc.returncode == 0:
        return out.strip()
    return None

wsl_graph_path = get_wsl_path(graph_path)
wsl_bin_dir = get_wsl_path(bin_dir)

print("wsl_graph_path:", wsl_graph_path)
print("wsl_bin_dir:", wsl_bin_dir)

command = ["wsl", f"{wsl_bin_dir}/bellman_ford_serial", wsl_graph_path, "0"]
print(f"Running command: {' '.join(command)}")

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
stdout, stderr = process.communicate()
print("RETURN CODE:", process.returncode)
print("STDOUT:", stdout)
print("STDERR:", stderr)
