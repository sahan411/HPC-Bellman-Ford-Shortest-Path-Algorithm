# How to Run the Project - Quick Guide
# =====================================
#
# This guide shows how to set up, build, and run every version
# of the Bellman-Ford parallel implementation on your machine.

---

## Requirements

You need the following tools installed:

| Tool | Used For | Check if installed |
|------|----------|--------------------|
| GCC  | Compiling C code | `gcc --version` |
| Make | Build system | `make --version` |
| MPI  | MPI version | `mpicc --version` |
| CUDA | GPU version | `nvcc --version` |

**On this project we use MSYS2 (already installed):**
- GCC is at `C:\msys64\ucrt64\bin\gcc.exe`
- Make is at `C:\msys64\usr\bin\make.exe`

**For MPI (needed later - Person 3's task):**
- Open MSYS2 terminal and run:
  ```
  pacman -S mingw-w64-ucrt-x86_64-msmpi
  ```

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/sahan411/HPC-Bellman-Ford-Shortest-Path-Algorithm.git
cd HPC-Bellman-Ford-Shortest-Path-Algorithm
```

---

## Step 2: Build the Programs

Make sure you are in the project root folder (`HPC-Bellman-Ford-Shortest-Path-Algorithm/`).

### Build graph generator + serial version only:
```bash
gcc -O2 -Wall -o bin/gen_graph.exe graph_generator/gen_graph.c

gcc -O2 -Wall -o bin/bellman_ford_serial.exe \
    src/serial/bellman_ford_serial.c \
    src/common/graph.c \
    src/common/utils.c \
    -Isrc/common
```

### Build OpenMP version (when ready):
```bash
gcc -O2 -Wall -fopenmp -o bin/bellman_ford_openmp.exe \
    src/openmp/bellman_ford_openmp.c \
    src/common/graph.c \
    src/common/utils.c \
    -Isrc/common
```

### Build MPI version (when ready):
```bash
mpicc -O2 -Wall -o bin/bellman_ford_mpi.exe \
    src/mpi/bellman_ford_mpi.c \
    src/common/graph.c \
    src/common/utils.c \
    -Isrc/common
```

### Build Hybrid version (when ready):
```bash
mpicc -O2 -Wall -fopenmp -o bin/bellman_ford_hybrid.exe \
    src/hybrid/bellman_ford_hybrid.c \
    src/common/graph.c \
    src/common/utils.c \
    -Isrc/common
```

### Build CUDA version (when ready):
```bash
nvcc -O2 -o bin/bellman_ford_cuda.exe \
    src/cuda/bellman_ford_cuda.cu \
    src/common/graph.c \
    src/common/utils.c \
    -Isrc/common
```

---

## Step 3: Generate Test Graphs

Before running the algorithm, you need to generate test graphs.

### Syntax:
```bash
./bin/gen_graph.exe <vertices> <edges> <output_file> <seed>
```

### Generate all standard test graphs:
```bash
./bin/gen_graph.exe 6 10 graphs/tiny.txt 42
./bin/gen_graph.exe 1000 10000 graphs/small.txt 42
./bin/gen_graph.exe 10000 100000 graphs/medium.txt 42
./bin/gen_graph.exe 100000 1000000 graphs/large.txt 42
```

> The seed (42) makes the graph reproducible - same seed = same graph every time.

---

## Step 4: Run the Programs

### Syntax for all versions:
```bash
./bin/<program_name> <graph_file> <source_vertex>
```

`source_vertex` is the starting point (usually 0).

---

### Run Serial Version:
```bash
./bin/bellman_ford_serial.exe graphs/small.txt 0
./bin/bellman_ford_serial.exe graphs/large.txt 0
```

**Example output:**
```
Graph loaded: 1000 vertices, 10000 edges
Running Bellman-Ford serial from source vertex 0...
  Early termination at iteration 9 (no changes)
  No negative-weight cycles detected.

  Source vertex   : 0
  Execution time  : 0.000450 seconds

Distances saved to 'results/serial_distances.txt'
```

---

### Run OpenMP Version:
```bash
# Set number of threads BEFORE running
set OMP_NUM_THREADS=4          # Windows
export OMP_NUM_THREADS=4       # Linux/Mac

./bin/bellman_ford_openmp.exe graphs/large.txt 0
```

Try with different thread counts to see speedup:
```bash
set OMP_NUM_THREADS=2 && ./bin/bellman_ford_openmp.exe graphs/large.txt 0
set OMP_NUM_THREADS=4 && ./bin/bellman_ford_openmp.exe graphs/large.txt 0
set OMP_NUM_THREADS=8 && ./bin/bellman_ford_openmp.exe graphs/large.txt 0
```

---

### Run MPI Version:
```bash
# -np = number of processes
mpirun -np 2 ./bin/bellman_ford_mpi.exe graphs/large.txt 0
mpirun -np 4 ./bin/bellman_ford_mpi.exe graphs/large.txt 0
mpirun -np 8 ./bin/bellman_ford_mpi.exe graphs/large.txt 0
```

---

### Run Hybrid Version (MPI + OpenMP):
```bash
# 2 MPI processes, each using 4 OpenMP threads (= 8 total workers)
set OMP_NUM_THREADS=4
mpirun -np 2 ./bin/bellman_ford_hybrid.exe graphs/large.txt 0
```

---

### Run CUDA Version:
```bash
# Just run it - it automatically uses your NVIDIA GPU
./bin/bellman_ford_cuda.exe graphs/large.txt 0
```

---

## Step 5: Verify Correctness

After running any parallel version, compare its output with the serial version:

```
The serial version saves distances to: results/serial_distances.txt
Each parallel version saves distances to: results/<version>_distances.txt

If all distances match the serial version --> CORRECT!
```

Each parallel program automatically prints a verification result like:
```
VERIFICATION PASSED: All 100000 distances match.
```

or if something went wrong:
```
VERIFICATION FAILED: 5 out of 100000 distances mismatch.
```

---

## Step 6: Record Your Results

After each run, note down the execution time printed in the output.
Record into the table in `docs/PROJECT_STEPS.md`.

**Results table format:**
| Version | Graph | Threads/Procs | Time (s) | Speedup |
|---------|-------|---------------|----------|---------|
| Serial  | Large | 1             | 0.022    | 1.0     |
| OpenMP  | Large | 4             | ?        | ?       |
| MPI     | Large | 4             | ?        | ?       |
| Hybrid  | Large | 2x4           | ?        | ?       |
| CUDA    | Large | GPU           | ?        | ?       |

Speedup = Serial_time / Parallel_time

---

## Common Problems & Fixes

| Problem | Possible Cause | Fix |
|---------|---------------|-----|
| `gcc: command not found` | GCC not in PATH | Add `C:\msys64\ucrt64\bin` to PATH |
| `Cannot open graph file` | Wrong path | Make sure you run from project root folder |
| `bin/` folder missing | Not created yet | Run `mkdir bin` first |
| Negative cycle detected | Wrong graph | Regenerate graph with different seed |
| Distances don't match | Bug in parallel code | Check race conditions and atomic operations |
| MPI not found | Not installed | `pacman -S mingw-w64-ucrt-x86_64-msmpi` |

---

## Useful Folder Map

```
Root folder:           C:\Users\...\HPC-Bellman-Ford-Shortest-Path-Algorithm\
Compiled programs:     bin\
Source code:           src\
Generated graphs:      graphs\
Results & distances:   results\
Documentation:         docs\
```

Always run programs FROM the root folder so file paths resolve correctly.
