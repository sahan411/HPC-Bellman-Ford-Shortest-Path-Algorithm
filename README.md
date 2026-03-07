# Bellman-Ford Shortest Path Algorithm - HPC Parallel Implementation

## Project Title
**Parallel Implementation and Performance Evaluation of the Bellman-Ford Shortest Path Algorithm Using OpenMP, MPI, and Hybrid Programming Models**

## Overview
This project implements the Bellman-Ford shortest path algorithm in **5 versions** and compares their performance:

| Version | Model | Technology | Description |
|---------|-------|------------|-------------|
| Serial | Baseline | C (single thread) | Standard Bellman-Ford, used as ground truth |
| OpenMP | Shared Memory | C + OpenMP | Multi-threaded on a single machine |
| MPI | Distributed Memory | C + MPI | Multi-process across cores/machines |
| Hybrid | MPI + OpenMP | C + MPI + OpenMP | Distributed + shared memory combined |
| CUDA | GPU | C + CUDA | Massively parallel on NVIDIA GPU |

## Why Bellman-Ford?
- Solves **single-source shortest path** for weighted directed graphs
- Handles **negative edge weights** (unlike Dijkstra's algorithm)
- Can detect **negative-weight cycles**
- Inner loop (edge relaxation) is **naturally parallelizable**
- Used in GPS navigation, network routing, transportation planning

## Project Structure
```
HPC-Bellman-Ford-Shortest-Path-Algorithm/
├── README.md                   # This file
├── Makefile                    # Build system for all versions
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code
│   ├── common/                 # Shared code (used by all versions)
│   │   ├── graph.h             # Graph data structures (Edge, Graph)
│   │   ├── graph.c             # Graph load/save/create/print
│   │   ├── timer.h             # Cross-platform high-res timer
│   │   ├── utils.h             # Utility function declarations
│   │   └── utils.c             # Distance save/load/verify/print
│   ├── serial/                 # Serial (baseline) implementation
│   │   └── bellman_ford_serial.c
│   ├── openmp/                 # OpenMP (shared memory) implementation
│   │   └── bellman_ford_openmp.c
│   ├── mpi/                    # MPI (distributed memory) implementation
│   │   └── bellman_ford_mpi.c
│   ├── hybrid/                 # Hybrid (MPI + OpenMP) implementation
│   │   └── bellman_ford_hybrid.c
│   └── cuda/                   # CUDA (GPU) implementation
│       └── bellman_ford_cuda.cu
│
├── graph_generator/            # Tool to generate random test graphs
│   └── gen_graph.c
│
├── scripts/                    # Automation scripts
│   ├── run_benchmarks.sh       # Run all versions & collect times
│   └── plot_results.py         # Generate performance charts
│
├── results/                    # Output: CSVs, distance files, charts
├── graphs/                     # Generated test graph files
└── docs/                       # Documentation
    ├── PROJECT_STEPS.md        # Step-by-step progress tracker
    └── report.pdf              # Final project report
```

## Prerequisites
- **GCC** compiler (with OpenMP support)
- **MPI** library (OpenMPI or MPICH)
- **NVIDIA CUDA Toolkit** (for GPU version)
- **Make** build tool
- **Python 3 + matplotlib** (for plotting, optional)

### Install on Ubuntu/WSL:
```bash
sudo apt update
sudo apt install gcc make openmpi-bin libopenmpi-dev
# For CUDA: install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
```

## Quick Start

### Build
```bash
make serial       # Build serial version + graph generator
make openmp       # Build OpenMP version
make mpi          # Build MPI version
make hybrid       # Build Hybrid (MPI+OpenMP) version
make cuda         # Build CUDA (GPU) version
make all          # Build everything
make clean        # Remove compiled files
```

### Generate Test Graphs
```bash
# Small graph (1K vertices, 10K edges)
./bin/gen_graph 1000 10000 graphs/small.txt 42

# Large graph (100K vertices, 1M edges)
./bin/gen_graph 100000 1000000 graphs/large.txt 42
```

### Run
```bash
# Serial (baseline)
./bin/bellman_ford_serial graphs/small.txt 0

# OpenMP (4 threads)
export OMP_NUM_THREADS=4
./bin/bellman_ford_openmp graphs/large.txt 0

# MPI (4 processes)
mpirun -np 4 ./bin/bellman_ford_mpi graphs/large.txt 0

# Hybrid (2 processes x 4 threads each)
export OMP_NUM_THREADS=4
mpirun -np 2 ./bin/bellman_ford_hybrid graphs/large.txt 0

# CUDA (GPU)
./bin/bellman_ford_cuda graphs/large.txt 0
```

## Performance Metrics
- **Execution Time** - wall-clock time for each version
- **Speedup** - S(p) = T_serial / T_parallel(p)
- **Efficiency** - E(p) = S(p) / p
- **Strong Scaling** - fixed problem size, increasing parallelism
- **Weak Scaling** - problem size grows with parallelism
- **Communication Overhead** - time spent in MPI calls vs computation

## Algorithm Complexity
| Metric | Serial | Parallel (p workers) |
|--------|--------|---------------------|
| Time | O(V × E) | O(V × E/p) + communication |
| Space | O(V) | O(V) per worker + O(E/p) edges |

## Team
- Person 1 (Lead): Serial implementation, project setup, CUDA, benchmarking
- Person 2: OpenMP implementation, OpenMP benchmarking
- Person 3: MPI implementation, MPI benchmarking
- All: Hybrid implementation, final report

## Progress
See [docs/PROJECT_STEPS.md](docs/PROJECT_STEPS.md) for detailed step-by-step progress.

## License
Academic project for HPC module.
