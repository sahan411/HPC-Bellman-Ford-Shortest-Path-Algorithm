import subprocess

try:
    # wsl -l -v output is utf-16le on Windows
    wsl_list = subprocess.check_output(["wsl", "-l", "-q"], text=True, encoding='utf-16le')
    for line in wsl_list.splitlines():
        distro = line.strip('\x00').strip()
        if distro and "docker" not in distro.lower():
            print(f"Found WSL distro: {distro}")
            break
except Exception as e:
    print("Error:", e)
