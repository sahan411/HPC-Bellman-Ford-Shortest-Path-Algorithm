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
**Status:** IN PROGRESS

### Tasks:
- [ ] Install build tools (GCC, Make) via WSL or MinGW
- [ ] Build graph generator: `make gen_graph`
- [ ] Generate test graphs of different sizes
- [ ] Build serial version: `make serial`
- [ ] Run serial on small graph, verify output manually
- [ ] Run serial on medium/large graphs, record execution times
- [ ] Save serial distance results (used as ground truth for all parallel versions)

### Test Graph Sizes:
| Label      | Vertices | Edges      | File                    |
|------------|----------|------------|-------------------------|
| Tiny       | 6        | 8          | graphs/tiny.txt         |
| Small      | 1,000    | 10,000     | graphs/small.txt        |
| Medium     | 10,000   | 100,000    | graphs/medium.txt       |
| Large      | 100,000  | 1,000,000  | graphs/large.txt        |
| Very Large | 500,000  | 5,000,000  | graphs/vlarge.txt       |

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
**Status:** NOT STARTED

### Tasks:
- [ ] Study how OpenMP parallelizes loops (`#pragma omp parallel for`)
- [ ] Implement `src/openmp/bellman_ford_openmp.c`
- [ ] Handle race conditions on dist[] array (use `#pragma omp atomic` or local arrays)
- [ ] Implement parallel early termination using OpenMP reduction
- [ ] Build: `make openmp`
- [ ] Test correctness: compare distances with serial output
- [ ] Benchmark with 2, 4, 8, 16 threads
- [ ] Record execution times in results/

### Key OpenMP Concepts to Use:
- `#pragma omp parallel for` - parallelize the edge relaxation loop
- `#pragma omp atomic` - protect dist[] updates from race conditions
- `omp_set_num_threads()` - control thread count
- `omp_get_wtime()` - timing (already in timer.h)
- `#pragma omp parallel for reduction(||:updated)` - parallel early termination check

### How to run:
```bash
make openmp
export OMP_NUM_THREADS=4
./bin/bellman_ford_openmp graphs/large.txt 0
```

### Expected File:
```
src/openmp/bellman_ford_openmp.c
```

---

## STEP 4: MPI Implementation (Distributed Memory)
**Assigned to:** Person 3
**Status:** NOT STARTED

### Tasks:
- [ ] Study MPI basics (rank, size, send/recv, collective operations)
- [ ] Implement `src/mpi/bellman_ford_mpi.c`
- [ ] Partition edges evenly across MPI processes
- [ ] Each process relaxes its local edges, then synchronize with MPI_Allreduce
- [ ] Use MPI_Allreduce with MPI_MIN to merge distances after each iteration
- [ ] Implement parallel early termination using MPI_Allreduce on "updated" flag
- [ ] Build: `make mpi`
- [ ] Test correctness: compare distances with serial output
- [ ] Benchmark with 2, 4, 8, 16 processes
- [ ] Measure communication overhead (time in MPI calls vs computation)

### Key MPI Concepts to Use:
- `MPI_Init` / `MPI_Finalize` - setup and teardown
- `MPI_Comm_rank` / `MPI_Comm_size` - process identification
- `MPI_Bcast` - broadcast graph data to all processes
- `MPI_Allreduce` with `MPI_MIN` - synchronize distance arrays
- `MPI_Wtime` - timing
- `MPI_Scatter` / `MPI_Gather` - optional, for edge distribution

### How to run:
```bash
make mpi
mpirun -np 4 ./bin/bellman_ford_mpi graphs/large.txt 0
```

### Expected File:
```
src/mpi/bellman_ford_mpi.c
```

---

## STEP 5: Hybrid Implementation (MPI + OpenMP)
**Assigned to:** All 3 members together
**Status:** NOT STARTED

### Tasks:
- [ ] Combine MPI (between processes) + OpenMP (within each process)
- [ ] Implement `src/hybrid/bellman_ford_hybrid.c`
- [ ] Use `MPI_Init_thread` with `MPI_THREAD_FUNNELED`
- [ ] Each MPI process uses OpenMP threads for local edge relaxation
- [ ] Synchronize between processes with MPI_Allreduce
- [ ] Build: `make hybrid`
- [ ] Test correctness: compare with serial output
- [ ] Benchmark with various process x thread combinations (e.g., 2x4, 4x2, 4x4)

### How to run:
```bash
make hybrid
export OMP_NUM_THREADS=4
mpirun -np 2 ./bin/bellman_ford_hybrid graphs/large.txt 0
```

### Expected File:
```
src/hybrid/bellman_ford_hybrid.c
```

---

## STEP 6: CUDA Implementation (GPU)
**Assigned to:** Person 1 or Person 2 (whoever has NVIDIA GPU ready first)
**Status:** NOT STARTED

### Tasks:
- [ ] Study CUDA kernel basics (grid, block, thread mapping)
- [ ] Implement `src/cuda/bellman_ford_cuda.cu`
- [ ] Copy graph data to GPU using cudaMemcpy
- [ ] Launch kernel: each CUDA thread relaxes one edge (or a chunk of edges)
- [ ] Use atomicMin() for dist[] updates on GPU
- [ ] Copy results back to CPU
- [ ] Build: `make cuda`
- [ ] Test correctness: compare with serial output
- [ ] Benchmark on GPU vs serial CPU time

### Key CUDA Concepts to Use:
- `cudaMalloc` / `cudaMemcpy` - GPU memory management
- `__global__` kernel function - runs on GPU
- `atomicMin()` - atomic minimum for safe dist[] updates
- `cudaDeviceSynchronize()` - wait for kernel completion
- Grid/block sizing: `blocks = (E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK`

### How to run:
```bash
make cuda
./bin/bellman_ford_cuda graphs/large.txt 0
```

### Expected File:
```
src/cuda/bellman_ford_cuda.cu
```

---

## STEP 7: Performance Benchmarking
**Assigned to:** All 3 members
**Status:** NOT STARTED

### Tasks:
- [ ] Create benchmark script `scripts/run_benchmarks.sh`
- [ ] Run ALL versions on ALL graph sizes
- [ ] Record execution times in CSV format
- [ ] Calculate speedup: S(p) = T_serial / T_parallel(p)
- [ ] Calculate efficiency: E(p) = S(p) / p
- [ ] Test strong scaling (fixed problem size, vary threads/processes)
- [ ] Test weak scaling (increase problem with processors)
- [ ] Measure MPI communication overhead separately

### Results Table Template:
| Version | Graph Size | Threads/Procs | Time (s) | Speedup | Efficiency |
|---------|-----------|---------------|----------|---------|------------|
| Serial  | 100K/1M   | 1             | ?        | 1.0     | 100%       |
| OpenMP  | 100K/1M   | 2             | ?        | ?       | ?          |
| OpenMP  | 100K/1M   | 4             | ?        | ?       | ?          |
| OpenMP  | 100K/1M   | 8             | ?        | ?       | ?          |
| MPI     | 100K/1M   | 2             | ?        | ?       | ?          |
| MPI     | 100K/1M   | 4             | ?        | ?       | ?          |
| MPI     | 100K/1M   | 8             | ?        | ?       | ?          |
| Hybrid  | 100K/1M   | 2x4           | ?        | ?       | ?          |
| CUDA    | 100K/1M   | GPU           | ?        | ?       | ?          |

---

## STEP 8: Performance Charts & Visualization
**Assigned to:** Person 1 (Lead)
**Status:** NOT STARTED

### Tasks:
- [ ] Create plotting script `scripts/plot_results.py`
- [ ] Generate charts:
  - [ ] Execution time comparison (bar chart: all versions)
  - [ ] Speedup vs. number of threads/processes (line chart)
  - [ ] Efficiency vs. number of threads/processes (line chart)
  - [ ] Strong scaling analysis chart
  - [ ] Communication overhead breakdown (for MPI/Hybrid)
- [ ] Save charts as PNG in `results/` folder

---

## STEP 9: Final Report
**Assigned to:** All 3 members
**Status:** NOT STARTED

### Report Sections:
- [ ] Introduction & Problem Statement
- [ ] Literature Review (Bellman-Ford algorithm background)
- [ ] Methodology (serial, OpenMP, MPI, Hybrid, CUDA approaches)
- [ ] Implementation Details (code walkthrough, key design decisions)
- [ ] Experimental Setup (hardware, graph sizes, configurations)
- [ ] Results & Analysis (tables, charts, speedup, efficiency)
- [ ] Challenges & Solutions (race conditions, load balancing, etc.)
- [ ] Conclusion & Future Work
- [ ] References

### Expected File:
```
docs/report.pdf
```

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
