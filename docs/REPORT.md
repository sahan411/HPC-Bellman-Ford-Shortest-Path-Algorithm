# Parallel Implementation and Performance Evaluation of the Bellman-Ford Algorithm

**Course:** High-Performance Computing  
**Date:** 22 May 2026  
**Implementations compared:** Serial, POSIX pthreads, OpenMP, MPI, Hybrid MPI+OpenMP, MPI+CUDA  

---

## 1. Introduction

The single-source shortest path problem finds the shortest distance from one source vertex to every other vertex in a weighted directed graph. Bellman-Ford is used because it supports negative edge weights and can detect negative-weight cycles. Its main cost is high: in the worst case it relaxes every edge up to `V - 1` times, giving `O(V * E)` time complexity.

The goal of this project is to compare parallel implementations against the serial baseline:

```text
speedup = serial_time / parallel_time
```

A speedup above `1.00x` means the version is faster than serial. A speedup below `1.00x` means overhead is larger than the benefit.

Final implementations tested:

| Version | Programming model | Parallelism used |
|---|---|---|
| Serial | Plain C | Baseline |
| POSIX pthreads | Shared memory | Manual CPU threads using pthreads |
| OpenMP | Shared memory | CPU threads on one machine |
| MPI | Distributed memory | CPU processes with message passing |
| Hybrid | MPI + OpenMP | MPI processes, each using OpenMP threads |
| MPI+CUDA | MPI + GPU | MPI partitions edges; each rank relaxes its partition on CUDA |

Standalone CUDA is not included in the final comparison. CUDA is used only inside the MPI+CUDA version.

---

## 2. Methodology

All versions compute shortest paths from source vertex `0`. The serial version writes `results/serial_distances.txt`, and every parallel result is verified against that serial distance array. A failed verification is treated as a failed benchmark.

Each configuration was run 3 times, and the minimum execution time was reported. The measured time is algorithm computation time, not graph file loading time.

### 2.1 Serial Baseline

The serial implementation scans all edges repeatedly and stops early when a full pass produces no updates. It is the correctness and speedup baseline.

### 2.2 POSIX pthreads

The POSIX version was added as an extra shared-memory CPU comparison because it was straightforward to implement with the existing edge-list structure. It manually creates pthread workers and splits the edge list across threads. To keep correctness clear, it uses double-buffered distance arrays and per-vertex mutexes for updates.

Tested thread counts: `1`, `2`, `4`, and `8`.

### 2.3 OpenMP

OpenMP parallelizes the edge-relaxation loop:

```c
#pragma omp parallel for schedule(dynamic, 1024) reduction(|:updated)
for (int e = 0; e < graph->E; e++) {
    ...
}
```

Tested thread counts: `1`, `2`, `4`, and `8`.

### 2.4 MPI

MPI divides the edge array across processes. After each local relaxation pass, all processes combine distances with:

```c
MPI_Allreduce(dist, new_dist, V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
```

Tested process counts: `1`, `2`, `4`, and `8`.

### 2.5 Hybrid MPI+OpenMP

Hybrid combines the two CPU models:

- MPI splits the edge list across processes.
- OpenMP splits each process's edge partition across threads.
- MPI synchronizes global distances after each iteration.

Tested hybrid configurations:

| Configuration | Total workers |
|---|---:|
| 1 proc x 8 threads | 8 |
| 2 procs x 4 threads | 8 |
| 4 procs x 2 threads | 8 |
| 8 procs x 1 thread | 8 |
| 2 procs x 8 threads | 16 |
| 4 procs x 4 threads | 16 |

The 16-worker cases oversubscribe the 8-logical-processor laptop, but they were tested to check whether more hybrid workers helped.

### 2.6 MPI+CUDA

MPI+CUDA was added for the final run because CUDA should be used to improve the MPI-based implementation rather than run alone.

Implementation strategy:

- Each MPI rank gets a partition of the edge list.
- Each rank copies only its local edges to CUDA memory.
- A CUDA kernel relaxes local edges in parallel.
- The rank copies its distance array back to host memory.
- `MPI_Allreduce(..., MPI_MIN, ...)` combines distances across ranks.
- The reduced distance array is copied back to the GPU for the next iteration.

Tested MPI+CUDA configurations:

| Configuration | Notes |
|---|---|
| 2 MPI processes + CUDA | Lower MPI rank count |
| 4 MPI processes + CUDA | More edge partitioning, more GPU sharing |

This machine has one visible CUDA GPU: NVIDIA GeForce RTX 3050 Laptop GPU. Therefore all MPI ranks share the same GPU. On a multi-GPU system, each rank could use a different GPU, which is the correct environment for MPI+CUDA.

---

## 3. Datasets

| Dataset | Vertices | Edges | Purpose |
|---|---:|---:|---|
| `tiny.txt` | 6 | 10 | Smoke test |
| `small.txt` | 1,000 | 10,000 | Small overhead-dominated case |
| `medium.txt` | 10,000 | 100,000 | Medium crossover case |
| `large.txt` | 100,000 | 1,000,000 | Main mixed-weight CPU scaling case |
| `xlarge_pos.txt` | 500,000 | 5,000,000 | Extended positive-weight scaling case |
| `xxlarge_pos.txt` | 1,000,000 | 10,000,000 | Largest final benchmark |

The graph generator uses Johnson-style vertex potentials so mixed-weight graphs can contain negative edges without negative-weight cycles. The `_pos` datasets use positive-only weights and a random-tree backbone, which gives a smaller hop-count diameter and is more suitable for GPU-based testing.

---

## 4. Experimental Setup

| Component | Value |
|---|---|
| OS | Microsoft Windows 10.0.26200.7462 |
| CPU workers available | 8 logical processors |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB |
| GPU driver | 595.79 |
| MPI runtime | Microsoft MPI 10.1 |
| GCC | 13.2.0, MinGW-W64 UCRT |
| CUDA toolkit | 12.9 |
| Python | 3.10.4 |

Raw results and charts:

- `results/benchmark_results.csv`
- `results/charts/execution_time.png` - CPU-only chart, so MPI+CUDA does not distort the scale
- `results/charts/speedup.png` - CPU-only speedup chart
- `results/charts/mpi_cuda_execution_time.png` - MPI+CUDA timing chart
- `results/charts/mpi_cuda_speedup.png` - MPI+CUDA speedup chart

---

## 5. Results

All times are in seconds. Speedup is relative to serial on the same dataset.

### 5.1 Tiny Dataset

| Version | Config | Time | Speedup |
|---|---|---:|---:|
| Serial | 1 thread | 0.000001 | 1.00x |
| POSIX pthreads | 1 thread | 0.001521 | 0.00x |
| POSIX pthreads | 2 threads | 0.001667 | 0.00x |
| POSIX pthreads | 4 threads | 0.002074 | 0.00x |
| POSIX pthreads | 8 threads | 0.003187 | 0.00x |
| OpenMP | 1 thread | N/A | N/A |
| OpenMP | 2 threads | 0.001000 | 0.00x |
| OpenMP | 4 threads | 0.001000 | 0.00x |
| OpenMP | 8 threads | 0.002000 | 0.00x |
| MPI | 1 proc | 0.000015 | 0.07x |
| MPI | 2 procs | 0.000023 | 0.04x |
| MPI | 4 procs | 0.000141 | 0.01x |
| MPI | 8 procs | 0.000201 | 0.00x |
| Hybrid | 1 proc x 8 threads | 0.000784 | 0.00x |
| Hybrid | 2 procs x 4 threads | 0.000916 | 0.00x |
| Hybrid | 4 procs x 2 threads | 0.000555 | 0.00x |
| Hybrid | 8 procs x 1 thread | 0.000405 | 0.00x |
| Hybrid | 2 procs x 8 threads | 0.001204 | 0.00x |
| Hybrid | 4 procs x 4 threads | 0.000781 | 0.00x |
| MPI+CUDA | 2 procs + CUDA | 0.149356 | 0.00x |
| MPI+CUDA | 4 procs + CUDA | 0.181315 | 0.00x |

### 5.2 Small Dataset

| Version | Config | Time | Speedup |
|---|---|---:|---:|
| Serial | 1 thread | 0.000133 | 1.00x |
| POSIX pthreads | 1 thread | 0.002090 | 0.06x |
| POSIX pthreads | 2 threads | 0.003640 | 0.04x |
| POSIX pthreads | 4 threads | 0.003818 | 0.03x |
| POSIX pthreads | 8 threads | 0.006087 | 0.02x |
| OpenMP | 1 thread | 0.001000 | 0.13x |
| OpenMP | 2 threads | 0.001000 | 0.13x |
| OpenMP | 4 threads | 0.001000 | 0.13x |
| OpenMP | 8 threads | 0.003000 | 0.04x |
| MPI | 1 proc | 0.000133 | 1.00x |
| MPI | 2 procs | 0.000229 | 0.58x |
| MPI | 4 procs | 0.000292 | 0.46x |
| MPI | 8 procs | 0.000503 | 0.26x |
| Hybrid | 1 proc x 8 threads | 0.001228 | 0.11x |
| Hybrid | 2 procs x 4 threads | 0.002086 | 0.06x |
| Hybrid | 4 procs x 2 threads | 0.001890 | 0.07x |
| Hybrid | 8 procs x 1 thread | 0.021196 | 0.01x |
| Hybrid | 2 procs x 8 threads | 0.002086 | 0.06x |
| Hybrid | 4 procs x 4 threads | 0.001620 | 0.08x |
| MPI+CUDA | 2 procs + CUDA | 0.143974 | 0.00x |
| MPI+CUDA | 4 procs + CUDA | 0.143246 | 0.00x |

### 5.3 Medium Dataset

| Version | Config | Time | Speedup |
|---|---|---:|---:|
| Serial | 1 thread | 0.001542 | 1.00x |
| POSIX pthreads | 1 thread | 0.009067 | 0.17x |
| POSIX pthreads | 2 threads | 0.008018 | 0.19x |
| POSIX pthreads | 4 threads | 0.008474 | 0.18x |
| POSIX pthreads | 8 threads | 0.012822 | 0.12x |
| OpenMP | 1 thread | 0.001000 | 1.54x |
| OpenMP | 2 threads | 0.002000 | 0.77x |
| OpenMP | 4 threads | 0.002000 | 0.77x |
| OpenMP | 8 threads | 0.003000 | 0.51x |
| MPI | 1 proc | 0.001810 | 0.85x |
| MPI | 2 procs | 0.001421 | 1.09x |
| MPI | 4 procs | 0.001182 | 1.30x |
| MPI | 8 procs | 0.036539 | 0.04x |
| Hybrid | 1 proc x 8 threads | 0.001903 | 0.81x |
| Hybrid | 2 procs x 4 threads | 0.002316 | 0.67x |
| Hybrid | 4 procs x 2 threads | 0.002888 | 0.53x |
| Hybrid | 8 procs x 1 thread | 0.002337 | 0.66x |
| Hybrid | 2 procs x 8 threads | 0.003175 | 0.49x |
| Hybrid | 4 procs x 4 threads | 0.003141 | 0.49x |
| MPI+CUDA | 2 procs + CUDA | 0.147406 | 0.01x |
| MPI+CUDA | 4 procs + CUDA | 0.200787 | 0.01x |

### 5.4 Large Dataset

| Version | Config | Time | Speedup |
|---|---|---:|---:|
| Serial | 1 thread | 0.018609 | 1.00x |
| POSIX pthreads | 1 thread | 0.173121 | 0.11x |
| POSIX pthreads | 2 threads | 0.125835 | 0.15x |
| POSIX pthreads | 4 threads | 0.077531 | 0.24x |
| POSIX pthreads | 8 threads | 0.081755 | 0.23x |
| OpenMP | 1 thread | 0.021000 | 0.89x |
| OpenMP | 2 threads | 0.017000 | 1.09x |
| OpenMP | 4 threads | 0.011000 | 1.69x |
| OpenMP | 8 threads | 0.008000 | 2.33x |
| MPI | 1 proc | 0.018791 | 0.99x |
| MPI | 2 procs | 0.015102 | 1.23x |
| MPI | 4 procs | 0.012714 | 1.46x |
| MPI | 8 procs | 0.019221 | 0.97x |
| Hybrid | 1 proc x 8 threads | 0.008857 | 2.10x |
| Hybrid | 2 procs x 4 threads | 0.012458 | 1.49x |
| Hybrid | 4 procs x 2 threads | 0.013413 | 1.39x |
| Hybrid | 8 procs x 1 thread | 0.110367 | 0.17x |
| Hybrid | 2 procs x 8 threads | 0.012207 | 1.52x |
| Hybrid | 4 procs x 4 threads | 0.014444 | 1.29x |
| MPI+CUDA | 2 procs + CUDA | 0.791917 | 0.02x |
| MPI+CUDA | 4 procs + CUDA | 0.837827 | 0.02x |

### 5.5 XLarge Positive Dataset

| Version | Config | Time | Speedup |
|---|---|---:|---:|
| Serial | 1 thread | 0.052995 | 1.00x |
| POSIX pthreads | 1 thread | 0.092870 | 0.57x |
| POSIX pthreads | 2 threads | 0.072072 | 0.74x |
| POSIX pthreads | 4 threads | 0.063038 | 0.84x |
| POSIX pthreads | 8 threads | 0.057266 | 0.93x |
| OpenMP | 1 thread | 0.049000 | 1.08x |
| OpenMP | 2 threads | 0.025000 | 2.12x |
| OpenMP | 4 threads | 0.019000 | 2.79x |
| OpenMP | 8 threads | 0.013000 | 4.08x |
| MPI | 1 proc | 0.051467 | 1.03x |
| MPI | 2 procs | 0.031437 | 1.69x |
| MPI | 4 procs | 0.034778 | 1.52x |
| MPI | 8 procs | 0.068073 | 0.78x |
| Hybrid | 1 proc x 8 threads | 0.017999 | 2.94x |
| Hybrid | 2 procs x 4 threads | 0.022451 | 2.36x |
| Hybrid | 4 procs x 2 threads | 0.035130 | 1.51x |
| Hybrid | 8 procs x 1 thread | 0.090061 | 0.59x |
| Hybrid | 2 procs x 8 threads | 0.020170 | 2.63x |
| Hybrid | 4 procs x 4 threads | 0.032618 | 1.62x |
| MPI+CUDA | 2 procs + CUDA | 0.721424 | 0.07x |
| MPI+CUDA | 4 procs + CUDA | 0.780160 | 0.07x |

### 5.6 XXLarge Positive Dataset

| Version | Config | Time | Speedup |
|---|---|---:|---:|
| Serial | 1 thread | 0.071039 | 1.00x |
| POSIX pthreads | 1 thread | 0.261119 | 0.27x |
| POSIX pthreads | 2 threads | 0.208561 | 0.34x |
| POSIX pthreads | 4 threads | 0.186160 | 0.38x |
| POSIX pthreads | 8 threads | 0.159591 | 0.45x |
| OpenMP | 1 thread | 0.070000 | 1.01x |
| OpenMP | 2 threads | 0.049000 | 1.45x |
| OpenMP | 4 threads | 0.035000 | 2.03x |
| OpenMP | 8 threads | 0.025000 | 2.84x |
| MPI | 1 proc | 0.071746 | 0.99x |
| MPI | 2 procs | 0.053670 | 1.32x |
| MPI | 4 procs | 0.065777 | 1.08x |
| MPI | 8 procs | 0.153479 | 0.46x |
| Hybrid | 1 proc x 8 threads | 0.036085 | 1.97x |
| Hybrid | 2 procs x 4 threads | 0.046077 | 1.54x |
| Hybrid | 4 procs x 2 threads | 0.069385 | 1.02x |
| Hybrid | 8 procs x 1 thread | 0.156903 | 0.45x |
| Hybrid | 2 procs x 8 threads | 0.042013 | 1.69x |
| Hybrid | 4 procs x 4 threads | 0.063057 | 1.13x |
| MPI+CUDA | 2 procs + CUDA | 0.757402 | 0.09x |
| MPI+CUDA | 4 procs + CUDA | 0.828005 | 0.09x |

---

## 6. Analysis

### 6.1 Best Results

| Dataset | Serial time | Best parallel result | Best speedup |
|---|---:|---|---:|
| tiny | 0.000001 | Serial | 1.00x |
| small | 0.000133 | Serial / MPI 1 proc | 1.00x |
| medium | 0.001542 | OpenMP 1 thread | 1.54x |
| large | 0.018609 | OpenMP 8 threads | 2.33x |
| xlarge_pos | 0.052995 | OpenMP 8 threads | 4.08x |
| xxlarge_pos | 0.071039 | OpenMP 8 threads | 2.84x |

The meaningful results are the large datasets. Tiny, small, and medium are heavily affected by timing granularity and framework overhead.

### 6.2 POSIX pthreads Scaling

POSIX pthreads was correct on all datasets, but it did not beat the serial baseline. The best POSIX result was `0.93x` on `xlarge_pos` with 8 threads.

| Dataset | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| large | 0.11x | 0.15x | 0.24x | 0.23x |
| xlarge_pos | 0.57x | 0.74x | 0.84x | 0.93x |
| xxlarge_pos | 0.27x | 0.34x | 0.38x | 0.45x |

The pthreads implementation uses explicit thread creation and per-vertex mutexes. That makes it a valid shared-memory implementation, but the locking and manual thread-management overhead are higher than the OpenMP runtime on this machine.

### 6.3 OpenMP Scaling

OpenMP is the strongest model on this single machine:

| Dataset | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| large | 0.89x | 1.09x | 1.69x | 2.33x |
| xlarge_pos | 1.08x | 2.12x | 2.79x | 4.08x |
| xxlarge_pos | 1.01x | 1.45x | 2.03x | 2.84x |

The best final result is **OpenMP 8 threads on `xlarge_pos`**, with **4.08x speedup**.

### 6.4 MPI Scaling

MPI improves for larger datasets, but does not keep improving at 8 processes:

| Dataset | 1 proc | 2 procs | 4 procs | 8 procs | Best MPI |
|---|---:|---:|---:|---:|---|
| large | 0.99x | 1.23x | 1.46x | 0.97x | 4 procs |
| xlarge_pos | 1.03x | 1.69x | 1.52x | 0.78x | 2 procs |
| xxlarge_pos | 0.99x | 1.32x | 1.08x | 0.46x | 2 procs |

The main limitation is `MPI_Allreduce` on the full distance array after every iteration. On a single machine, the communication and synchronization overhead grows quickly as process count increases.

### 6.5 Hybrid MPI+OpenMP

Hybrid performs best when it uses fewer MPI processes and more OpenMP threads:

| Dataset | 1x8 | 2x4 | 4x2 | 8x1 | 2x8 | 4x4 |
|---|---:|---:|---:|---:|---:|---:|
| large | 2.10x | 1.49x | 1.39x | 0.17x | 1.52x | 1.29x |
| xlarge_pos | 2.94x | 2.36x | 1.51x | 0.59x | 2.63x | 1.62x |
| xxlarge_pos | 1.97x | 1.54x | 1.02x | 0.45x | 1.69x | 1.13x |

`1 proc x 8 threads` is the best hybrid layout because it avoids most MPI communication. MPI-heavy layouts are slower.

### 6.6 MPI+CUDA Results

| Dataset | 2 procs + CUDA | 4 procs + CUDA | Best MPI+CUDA |
|---|---:|---:|---:|
| large | 0.02x | 0.02x | 0.02x |
| xlarge_pos | 0.07x | 0.07x | 0.07x |
| xxlarge_pos | 0.09x | 0.09x | 0.09x |

MPI+CUDA was correct, but it did not improve performance on this hardware. The reason is the hardware and synchronization model:

- There is only one GPU, so all MPI ranks share the same RTX 3050 Laptop GPU.
- Each MPI rank creates GPU work and competes for the same device.
- Every Bellman-Ford iteration copies the distance array from GPU to host for `MPI_Allreduce`.
- After `MPI_Allreduce`, the reduced distance array must be copied back to the GPU.
- This host-device synchronization is repeated every iteration.

Therefore the GPU work is dominated by GPU context overhead, memory copies, and MPI synchronization. MPI+CUDA is the correct direction for a multi-GPU system, but this laptop is not the correct hardware for it.

Running on Google Colab may not fix MPI+CUDA by itself. Colab usually gives one GPU to one notebook, and it is not a normal multi-node MPI environment. A better Colab GPU can reduce CUDA kernel time, but the MPI+CUDA design still needs multi-GPU hardware and GPU-aware communication to show the expected benefit.

---

## 7. Key Findings

1. **OpenMP 8 threads is the best final result.** It reaches `4.08x` on `xlarge_pos` and `2.84x` on `xxlarge_pos`.

2. **POSIX pthreads is correct but slower than OpenMP.** It is useful as a manual shared-memory comparison, but per-vertex locking makes it slower than the OpenMP implementation.

3. **MPI improves only up to 2 or 4 processes.** At 8 processes, `MPI_Allreduce` overhead dominates.

4. **Hybrid works best with fewer MPI ranks.** `1 proc x 8 threads` is consistently the best hybrid configuration.

5. **MPI+CUDA was implemented and verified, but it did not improve performance on this single-GPU laptop.** Best MPI+CUDA speedup was only `0.09x`.

6. **CUDA should improve performance only with the right hardware design.** A practical MPI+CUDA version needs multiple GPUs, preferably one GPU per MPI rank, and ideally GPU-aware MPI to avoid copying the distance array through host memory each iteration.

---

## 8. Conclusion

The project successfully implemented and benchmarked Serial, POSIX pthreads, OpenMP, MPI, Hybrid MPI+OpenMP, and MPI+CUDA versions of Bellman-Ford. Standalone CUDA was removed from the final comparison, and CUDA was tested only as part of MPI+CUDA.

The best measured implementation on this machine is **OpenMP with 8 threads**, with a peak speedup of **4.08x** on `xlarge_pos`. MPI and Hybrid show useful speedups, but only when MPI communication is limited. MPI+CUDA is correct but slower on this single-GPU laptop because all MPI ranks share one GPU and must copy distances between GPU and CPU for every `MPI_Allreduce`.

Final recommendation:

```text
Use OpenMP 8 threads as the best-performing result for this hardware.
Use POSIX pthreads as a correct manual-threading comparison, not as the best result.
Use MPI and Hybrid results to explain process-level communication overhead.
Include MPI+CUDA as implemented and verified, but explain that it needs multi-GPU hardware to improve performance.
```

---

## 9. Future Work

- Run MPI+CUDA on a multi-GPU machine with one MPI rank per GPU.
- Use GPU-aware MPI so distance arrays can be reduced directly from GPU memory.
- Use NCCL or CUDA-aware collectives for multi-GPU reductions.
- Reduce MPI+CUDA synchronization frequency by checking early termination less often.
- Profile the MPI+CUDA implementation with Nsight Systems to separate kernel time, memcpy time, and MPI time.
- Test larger graphs where GPU kernel work is large enough to outweigh fixed GPU and MPI overhead.
- Improve POSIX pthreads by reusing persistent worker threads across Bellman-Ford iterations instead of creating workers every iteration.

---

## 10. References

1. R. Bellman, "On a Routing Problem," Quarterly of Applied Mathematics, 1958.
2. L. R. Ford, "Network Flow Theory," RAND Corporation, 1956.
3. D. B. Johnson, "Efficient Algorithms for Shortest Paths in Sparse Networks," Journal of the ACM, 1977.
4. U. Meyer and P. Sanders, "Delta-stepping: A Parallelizable Shortest Path Algorithm," Journal of Algorithms, 2003.
5. P. Harish and P. J. Narayanan, "Accelerating Large Graph Algorithms on the GPU Using CUDA," HiPC, 2007.
6. OpenMP Architecture Review Board, "OpenMP Application Programming Interface."
7. MPI Forum, "MPI: A Message-Passing Interface Standard."
8. NVIDIA, "CUDA C++ Programming Guide."
