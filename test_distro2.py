import subprocess

# Let's test execution by cd-ing directly into the directory using full bash command
cmd = [
    "wsl", "-d", "Ubuntu-24.04", "bash", "-c", 
    "cd /mnt/host/f/HPC/HPC-Bellman-Ford-Shortest-Path-Algorithm && ls -la bin && ./bin/bellman_ford_serial graphs/small.txt 0"
]

print("Running command:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print("RETURN CODE:", result.returncode)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

cmd2 = [
    "wsl", "-d", "Ubuntu-24.04", "bash", "-c", 
    "cd /mnt/host/f/HPC/HPC-Bellman-Ford-Shortest-Path-Algorithm && ./bin/bellman_ford_mpi graphs/small.txt 0"
]
print("Running command2:", " ".join(cmd2))
result2 = subprocess.run(cmd2, capture_output=True, text=True)
print("RETURN CODE:", result2.returncode)
print("STDOUT:\n", result2.stdout)
print("STDERR:\n", result2.stderr)
