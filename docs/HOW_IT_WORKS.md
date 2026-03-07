# How This Project Works - Simple Explanation
# =============================================

## What is this project about?

We are solving the **Shortest Path Problem** using the **Bellman-Ford Algorithm**.

Imagine you have a map with cities connected by roads. Each road has a distance 
(some can even be negative, like shortcuts or discounts). You want to find the 
shortest route from one city to every other city.

**Bellman-Ford** does exactly this. It works even with negative road distances,
unlike the more famous Dijkstra's algorithm.

---

## Why do we need parallel computing?

When the map is small (few cities), a normal computer can solve it instantly.
But when you have **100,000+ cities and 1,000,000+ roads**, it takes too long
with just one CPU core.

So we use **parallel computing** - splitting the work across multiple cores,
multiple machines, or even a GPU to solve it faster.

---

## The 5 Versions We Implement

### Version 1: Serial (Baseline)
```
One worker does everything alone.
Just a normal single-threaded program.
This is our "ground truth" - we compare everything against this.
```

### Version 2: OpenMP (Shared Memory)
```
Multiple workers (threads) on the SAME computer share memory.
They split the edges among themselves.
Each iteration: all threads relax their edges in parallel.
Challenge: Two threads might update the same distance at once (race condition).
Solution: Use atomic operations to protect updates.
```

### Version 3: MPI (Distributed Memory)
```
Multiple workers (processes) that could be on DIFFERENT computers.
Each process gets a chunk of edges.
After each iteration, they ALL share their best distances (MPI_Allreduce).
Challenge: Communication takes time (sending distances between processes).
```

### Version 4: Hybrid (MPI + OpenMP)
```
Combines both approaches:
- MPI splits work between MACHINES (or process groups)
- OpenMP splits work between CORES within each machine
Best of both worlds for clusters with multi-core nodes.
```

### Version 5: CUDA (GPU)
```
Uses the NVIDIA GPU which has THOUSANDS of tiny cores.
Each core handles one or a few edges.
Massively parallel - can process millions of edges simultaneously.
Challenge: Moving data between CPU and GPU takes time.
```

---

## How does Bellman-Ford work?

### The Simple Idea:
```
1. Start: Source city = distance 0, everything else = infinity
2. Repeat (V-1) times:
     Look at every road (edge).
     If going through this road gives a shorter path, update the distance.
3. Check: If you can STILL improve, there's a negative cycle (infinite loop).
```

### Visual Example:
```
Graph: A --6--> B --(-2)--> C --3--> D

Start (source = A):
  A=0, B=INF, C=INF, D=INF

Iteration 1 (check all edges):
  Edge A->B: 0+6=6 < INF  --> B=6
  Edge B->C: 6+(-2)=4 < INF  --> C=4
  Edge C->D: 4+3=7 < INF  --> D=7

Iteration 2 (check all edges again):
  No improvements possible --> DONE (early termination)

Result: A=0, B=6, C=4, D=7
```

### Why it's parallelizable:
```
The inner loop "check all edges" can be split among workers:
  Worker 1: checks edges 1 to 250,000
  Worker 2: checks edges 250,001 to 500,000
  Worker 3: checks edges 500,001 to 750,000
  Worker 4: checks edges 750,001 to 1,000,000
They all work at the same time = faster!
```

---

## How we generate test graphs

We use **Johnson's Reweighting** technique to create random graphs:

```
1. Create a spanning tree (path through all vertices) = ensures connectivity
2. Add random extra edges = creates alternate paths
3. Use vertex potentials to set weights:
     edge_weight = base_weight + potential[source] - potential[destination]
   This GUARANTEES no negative cycles, but individual edges can be negative.
```

Why no negative cycles? For any cycle, the potential terms cancel out:
```
h[A]-h[B] + h[B]-h[C] + h[C]-h[A] = 0
So cycle weight = sum of base weights = always positive!
```

---

## How we measure performance

| Metric | Formula | What it tells us |
|--------|---------|------------------|
| Execution Time | Wall clock seconds | How fast each version runs |
| Speedup | Serial_time / Parallel_time | How many times faster |
| Efficiency | Speedup / num_workers | How well we use resources |
| Strong Scaling | Fix graph, add workers | Does more workers help? |
| Weak Scaling | Grow graph with workers | Can we handle bigger problems? |

---

## Project Files

```
src/serial/bellman_ford_serial.c     - The baseline (single thread)
src/openmp/bellman_ford_openmp.c     - OpenMP version (multi-thread)
src/mpi/bellman_ford_mpi.c           - MPI version (multi-process)
src/hybrid/bellman_ford_hybrid.c     - MPI + OpenMP combined
src/cuda/bellman_ford_cuda.cu        - GPU version
src/common/graph.h, graph.c          - Graph data structure (shared by all)
src/common/timer.h                   - Timer for benchmarking
src/common/utils.h, utils.c          - Result saving and verification
graph_generator/gen_graph.c          - Random graph generator
```

---

## How to build and run

```bash
# Build
gcc -O2 -o bin/gen_graph graph_generator/gen_graph.c
gcc -O2 -o bin/bellman_ford_serial src/serial/bellman_ford_serial.c src/common/graph.c src/common/utils.c -Isrc/common

# Generate a test graph (1000 vertices, 10000 edges)
./bin/gen_graph 1000 10000 graphs/test.txt 42

# Run serial version
./bin/bellman_ford_serial graphs/test.txt 0
```

---

## Team Task Division

| Task | Who | Status |
|------|-----|--------|
| Project setup, serial version, graph generator | Person 1 | Done |
| OpenMP implementation & benchmarking | Person 2 | Next |
| MPI implementation & benchmarking | Person 3 | Next |
| Hybrid (MPI+OpenMP) implementation | All together | After OpenMP & MPI |
| CUDA implementation | Person 1 or 2 | After Hybrid |
| Performance charts & final report | All together | Final phase |
