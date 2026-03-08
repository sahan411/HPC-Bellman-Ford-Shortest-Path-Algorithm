# Bellman-Ford HPC Project - Step-by-Step Progress Tracker
# ========================================================
#
# Team: 3 members
# Language: C
# Technologies: OpenMP, MPI, MPI+OpenMP (Hybrid), CUDA
#
# Use this document to track what's done and assign tasks to team members.
# Update the checkboxes as each task is completed.

---

## STEP 1: Project Setup & Serial Implementation
**Assigned to:** Person 1 (Lead)
**Status:** COMPLETED

### What was done:
- [x] Created GitHub repository and cloned locally
- [x] Set up full project directory structure
- [x] Created common headers: `graph.h`, `timer.h`, `utils.h`
- [x] Implemented `graph.c` (load, save, create, free, print graph)
- [x] Implemented `utils.c` (save, load, verify, print distances)
- [x] Created graph generator `gen_graph.c` (random directed weighted graphs)
- [x] Implemented serial Bellman-Ford `bellman_ford_serial.c`
- [x] Created Makefile with build targets for all versions
- [x] Created this project tracker document

### Files created:
```
src/common/graph.h          - Graph data structures (Edge, Graph)
src/common/graph.c          - Graph I/O operations
src/common/timer.h          - Cross-platform timer (OpenMP / Windows / Linux)
src/common/utils.h          - Utility function declarations
src/common/utils.c          - Distance save/load/verify/print
src/serial/bellman_ford_serial.c  - Serial baseline implementation
graph_generator/gen_graph.c       - Random graph generator
Makefile                          - Build system
```

---

## STEP 2: Build, Test & Validate Serial Version
**Assigned to:** Person 1 (Lead)
**Status:** COMPLETED

### Tasks:
- [x] Build tools found: MSYS2 GCC 13.1.0, GNU Make 4.4.1 (already on machine)
- [x] Build graph generator: compiled with zero errors/warnings
- [x] Generate test graphs of different sizes (tiny, small, medium, large)
- [x] Build serial version: compiled with zero errors/warnings
- [x] Run serial on tiny graph (6V, 10E), manually verified all distances correct
- [x] Run serial on small/medium/large graphs, recorded execution times
- [x] Fixed graph generator: Johnson's reweighting technique guarantees no negative cycles
- [x] Save serial distance results (used as ground truth for all parallel versions)

### Bug Fixed:
- Original generator could produce negative-weight cycles (crashed on small graph)
- Fixed with Johnson's Reweighting: assign vertex potentials h[v], edge weight = base + h[src] - h[dest]
- Cycle weights always positive (h terms telescope), but individual edges can be negative (~12%)

### Test Graph Sizes:
| Label      | Vertices | Edges      | File                    | Neg. Edges |
|------------|----------|------------|-------------------------|------------|
| Tiny       | 6        | 10         | graphs/tiny.txt         | 3 (30.0%)  |
| Small      | 1,000    | 10,000     | graphs/small.txt        | 1188 (11.9%) |
| Medium     | 10,000   | 100,000    | graphs/medium.txt       | 11846 (11.8%) |
| Large      | 100,000  | 1,000,000  | graphs/large.txt        | 117096 (11.7%) |

### Serial Baseline Timing Results:
| Graph   | Vertices | Edges      | Time (s)  | Iterations | Early Stop? |
|---------|----------|------------|-----------|------------|-------------|
| Tiny    | 6        | 10         | 0.000111  | 2 of 5     | Yes         |
| Small   | 1,000    | 10,000     | 0.000450  | 9 of 999   | Yes         |
| Medium  | 10,000   | 100,000    | 0.001852  | 10 of 9999 | Yes         |
| Large   | 100,000  | 1,000,000  | 0.022036  | 13 of 99999| Yes         |

> Note: Early termination works well due to Johnson's reweighting - graph structure 
> allows shortest paths to converge in very few iterations.

### How to run:
```bash
# Generate graphs
./bin/gen_graph 1000 10000 graphs/small.txt 42
./bin/gen_graph 10000 100000 graphs/medium.txt 42
./bin/gen_graph 100000 1000000 graphs/large.txt 42

# Run serial version
./bin/bellman_ford_serial graphs/small.txt 0
./bin/bellman_ford_serial graphs/large.txt 0
```

---

## STEP 3: OpenMP Implementation (Shared Memory)
**Assigned to:** Person 2
**Status:** COMPLETED

### What was done:
- [x] Implemented `src/openmp/bellman_ford_openmp.c`
- [x] Parallelized edge relaxation with `#pragma omp parallel for schedule(dynamic, 1024)`
- [x] Used OpenMP reduction for early termination flag (`reduction(|:updated)`)
- [x] Relaxed race conditions on dist[] — acceptable for Bellman-Ford convergence
- [x] Parallel negative cycle detection with reduction
- [x] Accepts thread count as 3rd CLI argument
- [x] Auto-verifies against serial results

### Build & Run:
```bash
gcc -O2 -Wall -fopenmp -o bin/bellman_ford_openmp.exe src/openmp/bellman_ford_openmp.c src/common/graph.c src/common/utils.c -Isrc/common
.\bin\bellman_ford_openmp.exe graphs/large.txt 0 8
```

### Correctness Verified:
- small (1K V): VERIFICATION PASSED: All 1000 distances match
- medium (10K V): VERIFICATION PASSED: All 10000 distances match
- large (100K V): VERIFICATION PASSED: All 100000 distances match

---

## STEP 4: MPI Implementation (Distributed Memory)
**Assigned to:** Person 3
**Status:** COMPLETED

### What was done:
- [x] Installed MS-MPI 10.1 runtime via winget + MSYS2 msmpi package
- [x] Implemented `src/mpi/bellman_ford_mpi.c`
- [x] Edges divided evenly across ranks (with remainder handling)
- [x] Each rank relaxes its edges locally, then MPI_Allreduce with MPI_MIN syncs distances
- [x] Early termination using MPI_Allreduce with MPI_LOR on updated flag
- [x] Negative cycle detection distributed across ranks
- [x] Only rank 0 saves results and verifies

### Build & Run:
```bash
C:\msys64\ucrt64\bin\mpicc.exe -O2 -Wall -o bin/bellman_ford_mpi.exe src/mpi/bellman_ford_mpi.c src/common/graph.c src/common/utils.c -Isrc/common
&"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 .\bin\bellman_ford_mpi.exe graphs/large.txt 0
```

### Correctness Verified:
- small (1K V), 2 processes: VERIFICATION PASSED
- small (1K V), 4 processes: VERIFICATION PASSED
- large (100K V), 4 processes: VERIFICATION PASSED: All 100000 distances match

---

## STEP 5: Hybrid Implementation (MPI + OpenMP)
**Assigned to:** All 3 members together
**Status:** COMPLETED

### What was done:
- [x] Implemented `src/hybrid/bellman_ford_hybrid.c`
- [x] Used `MPI_Init_thread` with `MPI_THREAD_FUNNELED`
- [x] MPI divides edges across processes (same as pure MPI)
- [x] OpenMP further parallelizes edge relaxation WITHIN each process
- [x] MPI_Allreduce syncs distances across processes after each iteration
- [x] All thread updates combined via OpenMP reduction before MPI sync

### Build & Run:
```bash
C:\msys64\ucrt64\bin\mpicc.exe -O2 -Wall -fopenmp -o bin/bellman_ford_hybrid.exe src/hybrid/bellman_ford_hybrid.c src/common/graph.c src/common/utils.c -Isrc/common
&"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 2 .\bin\bellman_ford_hybrid.exe graphs/large.txt 0 4
```

### Correctness Verified:
- small (1K V), 2 procs x 4 threads: VERIFICATION PASSED
- large (100K V), 2 procs x 4 threads: VERIFICATION PASSED: All 100000 distances match

---

## STEP 6: CUDA Implementation (GPU)
**Assigned to:** Person 1 or Person 2 (whoever has NVIDIA GPU ready first)
**Status:** COMPLETED (code written, compilation needs Windows SDK headers)

### What was done:
- [x] Implemented `src/cuda/bellman_ford_cuda.cu`
- [x] Graph converted to struct-of-arrays (SOA) for better GPU memory coalescing
- [x] One CUDA thread per edge — massive parallelism
- [x] `atomicMin()` for race-condition-safe dist[] updates on GPU
- [x] Grid: `(E + 255) / 256` blocks × 256 threads/block
- [x] `cudaDeviceSynchronize()` between iterations for correctness
- [x] Host↔Device memory copy with CUDA_CHECK macro
- [x] Parallel negative cycle detection kernel

### Build command (requires Windows SDK headers):
```bash
# In a VS Developer Command Prompt:
nvcc -O2 -Wno-deprecated-gpu-targets -o bin/bellman_ford_cuda.exe \
    src/cuda/bellman_ford_cuda.cu src/common/graph.c src/common/utils.c -Isrc/common
```

### Note on compilation:
CUDA compilation on Windows requires Visual Studio (cl.exe) AND the Windows
10/11 SDK headers. Install the SDK via the VS Installer → "Desktop development
with C++" → "Windows 10 SDK (10.0.19041.0)" or later.

### Key CUDA features used:
- `__global__` kernel — executes on GPU
- `atomicMin()` — thread-safe minimum update
- `cudaMalloc` / `cudaMemcpy` — GPU memory management
- `cudaDeviceSynchronize()` — wait for all GPU threads
- `struct cudaDeviceProp` — GPU information query

---

## STEP 7: Performance Benchmarking
**Assigned to:** All 3 members
**Status:** COMPLETED

### What was done:
- [x] Created `scripts/run_benchmarks.py` — runs all versions, all graph sizes
- [x] Created `scripts/plot_results.py` — generates chart images
- [x] Full benchmark run on Windows desktop (GCC 13.1, 8-core CPU)
- [x] Results saved to `results/benchmark_results.csv`
- [x] Charts saved to `results/charts/`

### Actual Benchmark Results (Large Graph, 100K vertices, 1M edges):
| Version       | Config         | Time (s)  | Speedup |
|---------------|----------------|-----------|---------|
| Serial        | 1 thread       | 0.029020  | 1.00x   |
| OpenMP        | 2 threads      | 0.019000  | 1.53x   |
| OpenMP        | 4 threads      | 0.016000  | 1.81x   |
| OpenMP        | 8 threads      | 0.018000  | 1.61x   |
| MPI           | 4 processes    | 0.023815  | 1.22x   |
| Hybrid        | 1 proc × 8 OMP | 0.013666  | **2.12x** |
| Hybrid        | 4 proc × 2 OMP | 0.021890  | 1.33x   |

> **Best speedup: Hybrid (1 × 8) = 2.12x** — uses all cores within one process  
> **Note:** Small graphs show slowdown (parallel overhead > compute time).  
> On larger graphs / a proper cluster, MPI speedup would be more significant.

### Timer resolution note:
- Serial uses Windows QueryPerformanceCounter (nano-second precision)
- OpenMP uses `omp_get_wtime()` (millisecond precision on Windows)
- MPI uses `MPI_Wtime()` (microsecond precision)
- This explains the "rounded" timing for OpenMP on small graphs

### How to run benchmarks:
```bash
python scripts/run_benchmarks.py   # run all versions
python scripts/plot_results.py     # generate charts
```

---

## STEP 8: Performance Charts & Visualization
**Assigned to:** Person 1 (Lead)
**Status:** COMPLETED

### What was done:
- [x] Created `scripts/plot_results.py` with matplotlib
- [x] Execution time bar charts (4 subplots per graph size)
- [x] Speedup charts with serial=1x reference line
- [x] Charts saved as `results/charts/execution_time.png` and `results/charts/speedup.png`

### To regenerate:
```bash
python scripts/run_benchmarks.py   # re-run benchmarks
python scripts/plot_results.py     # re-generate charts
```

---

## STEP 9: Final Report
**Assigned to:** All 3 members
**Status:** ✅ COMPLETED

### Report Sections:
- [x] Introduction & Problem Statement
- [x] Literature Review (Bellman-Ford algorithm background)
- [x] Methodology (serial, OpenMP, MPI, Hybrid, CUDA approaches)
- [x] Implementation Details (code walkthrough, key design decisions)
- [x] Experimental Setup (hardware, graph sizes, configurations)
- [x] Results & Analysis (tables, charts, speedup, efficiency)
- [x] Challenges & Solutions (race conditions, load balancing, etc.)
- [x] Conclusion & Future Work
- [x] References

### Deliverable:
```
docs/REPORT.md   ← comprehensive analysis and final report
```

### Key Findings Summary:
- Best parallel speedup on large graph (100K V, 1M E): **Hybrid 1×8 at 2.12x**
- OpenMP scales well for large graphs (4 threads → 1.81x)
- MPI overhead dominates for small graphs; benefits appear only at large scale
- Parallel overhead outweighs benefit for tiny/small graphs (normal behaviour)
- CUDA code implemented; compilation requires Windows 10 SDK (available on Linux)

---

## Quick Reference: Build & Run Commands

```bash
# Build
make serial       # serial + graph generator
make openmp       # OpenMP version
make mpi          # MPI version
make hybrid       # Hybrid (MPI+OpenMP) version
make cuda         # CUDA version
make all          # everything

# Generate test graphs
./bin/gen_graph 1000 10000 graphs/small.txt 42
./bin/gen_graph 100000 1000000 graphs/large.txt 42

# Run
./bin/bellman_ford_serial graphs/small.txt 0

export OMP_NUM_THREADS=4
./bin/bellman_ford_openmp graphs/large.txt 0

mpirun -np 4 ./bin/bellman_ford_mpi graphs/large.txt 0

export OMP_NUM_THREADS=4
mpirun -np 2 ./bin/bellman_ford_hybrid graphs/large.txt 0

./bin/bellman_ford_cuda graphs/large.txt 0
```
