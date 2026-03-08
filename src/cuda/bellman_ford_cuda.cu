/*
 * bellman_ford_cuda.cu - CUDA GPU-Parallel Bellman-Ford
 * =====================================================
 *
 * This version runs Bellman-Ford on the GPU, exploiting the massive
 * parallelism of thousands of CUDA cores.
 *
 * GPU vs CPU parallelism:
 *   CPU (OpenMP):  8-16 cores, ~1 thread per core, fast per-thread
 *   GPU (CUDA):    thousands of streaming processors (SMs), each running
 *                  hundreds of threads simultaneously -- great for data-parallel
 *                  loops where ALL iterations do the same work
 *
 * CUDA Execution Model:
 *   - You write a "kernel" function that runs on the GPU
 *   - Each thread works on a different edge
 *   - Threads are grouped into "blocks" (e.g., 256 threads each)
 *   - All blocks are launched together as a "grid"
 *
 *   Example: 1,000,000 edges, 256 threads/block:
 *     --> 3907 blocks launched
 *     --> All threads run the relax loop simultaneously
 *
 * Race conditions and atomicMin:
 *   Multiple threads might try to update dist[v] for the same v.
 *   We use atomicMin() -- a hardware atomic operation that reads and
 *   writes dist[v] in one indivisible instruction. No data races.
 *
 * Memory on GPU:
 *   - CPU memory (host): dist[], edges[]  (visible to CPU)
 *   - GPU memory (device): d_dist[], d_edges[]  (visible to GPU only)
 *   - We must copy data to GPU before running, copy results back after
 *
 * Usage:
 *   ./bellman_ford_cuda <graph_file> [source_vertex]
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* Include common headers (C-compatible) */
extern "C" {
#include "../common/graph.h"
#include "../common/timer.h"
#include "../common/utils.h"
}

/*
 * THREADS_PER_BLOCK - How many GPU threads are in each block.
 * 256 is a good default: divisible by 32 (warp size), fits in most GPUs.
 */
#define THREADS_PER_BLOCK 256

/*
 * CUDA_CHECK - Macro to check CUDA API calls for errors.
 * If anything fails (out of memory, kernel crash, etc.), print and exit.
 */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/*
 * Edge struct for CUDA (plain C struct, no pointer fields).
 * We use a flat struct-of-arrays (SOA) or plain arrays on GPU.
 * Using parallel arrays (src[], dest[], weight[]) is better for GPU
 * coalesced memory access than array-of-structs (AOS).
 */

/* ================================================================
 * CUDA KERNEL - runs on GPU, one thread per edge
 * ================================================================
 *
 * Each thread:
 *   1. Figures out which edge it's responsible for (thread index)
 *   2. Reads src, dest, weight for that edge
 *   3. Checks if we can relax: dist[src] + weight < dist[dest]
 *   4. If yes, uses atomicMin to safely update dist[dest]
 *   5. Sets *updated = 1 to signal that something changed
 *
 * Parameters:
 *   d_src[]    : source vertices of all edges (GPU memory)
 *   d_dest[]   : destination vertices of all edges (GPU memory)
 *   d_weight[] : edge weights (GPU memory)
 *   d_dist[]   : current distance array (GPU memory)
 *   E          : total number of edges
 *   d_updated  : pointer to flag (GPU memory), set to 1 if any update
 */
__global__ void relax_edges_kernel(int *d_src, int *d_dest, int *d_weight,
                                   int *d_dist, int E, int *d_updated) {
    /* Global thread index: which edge does this thread handle? */
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    /* Guard against threads beyond the edge count */
    if (j >= E) return;

    int u = d_src[j];
    int v = d_dest[j];
    int w = d_weight[j];

    /* Only relax if u is reachable */
    if (d_dist[u] != INF) {
        int new_dist = d_dist[u] + w;
        if (new_dist < d_dist[v]) {
            /*
             * atomicMin(address, value):
             *   Reads *address, computes min(*address, value),
             *   writes back -- all atomically (no other thread can
             *   interfere between read and write).
             *
             * This is what makes CUDA Bellman-Ford safe without locks.
             */
            int old = atomicMin(&d_dist[v], new_dist);
            if (old > new_dist) {
                /* We actually improved something */
                *d_updated = 1;
            }
        }
    }
}

/*
 * check_negative_cycle_kernel - Check if any edge can still be relaxed
 *
 * If yes, there's a negative cycle. Runs after all (V-1) iterations.
 * One thread per edge, same pattern as relax_edges_kernel.
 */
__global__ void check_negative_cycle_kernel(int *d_src, int *d_dest,
                                             int *d_weight, int *d_dist,
                                             int E, int *d_neg_cycle) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= E) return;

    int u = d_src[j];
    int v = d_dest[j];
    int w = d_weight[j];

    if (d_dist[u] != INF && d_dist[u] + w < d_dist[v]) {
        *d_neg_cycle = 1;
    }
}

/*
 * bellman_ford_cuda - Main CUDA Bellman-Ford function (runs on CPU)
 *
 * This function:
 *   1. Converts the graph to GPU-friendly arrays
 *   2. Allocates GPU memory and copies data
 *   3. Launches GPU kernels for each Bellman-Ford iteration
 *   4. Copies results back to CPU
 *
 * @graph:  pointer to the graph (CPU memory)
 * @source: source vertex
 * @dist:   output distance array (CPU memory, caller allocates)
 *
 * Returns: 0 if successful, -1 if negative cycle detected
 */
int bellman_ford_cuda(Graph *graph, int source, int *dist) {
    int V = graph->V;
    int E = graph->E;
    int i;

    /* ================================================================
     * STEP 1: Convert edge list to parallel arrays (better GPU access)
     * ================================================================
     *
     * Array-of-structs (AOS) is bad for GPUs: when thread 0 reads
     * edge[0].src, edge[0].dest, edge[0].weight they're scattered.
     * Struct-of-arrays (SOA): all src[] contiguous, better for coalescing.
     */
    int *h_src    = (int *)malloc(E * sizeof(int));
    int *h_dest   = (int *)malloc(E * sizeof(int));
    int *h_weight = (int *)malloc(E * sizeof(int));

    if (!h_src || !h_dest || !h_weight) {
        fprintf(stderr, "Error: Failed to allocate edge arrays.\n");
        return -1;
    }

    for (i = 0; i < E; i++) {
        h_src[i]    = graph->edges[i].src;
        h_dest[i]   = graph->edges[i].dest;
        h_weight[i] = graph->edges[i].weight;
    }

    /* ================================================================
     * STEP 2: Initialize distances on CPU
     * ================================================================ */
    for (i = 0; i < V; i++) dist[i] = INF;
    dist[source] = 0;

    /* ================================================================
     * STEP 3: Allocate GPU memory and copy data to GPU
     * ================================================================ */
    int *d_src, *d_dest, *d_weight;   /* edge arrays on GPU */
    int *d_dist;                       /* distance array on GPU */
    int *d_updated;                    /* update flag on GPU */
    int *d_neg_cycle;                  /* negative cycle flag on GPU */

    CUDA_CHECK(cudaMalloc((void **)&d_src,       E * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_dest,      E * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_weight,    E * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_dist,      V * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_updated,   sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_neg_cycle, sizeof(int)));

    /* Copy edge data CPU --> GPU (once, edges don't change) */
    CUDA_CHECK(cudaMemcpy(d_src,    h_src,    E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dest,   h_dest,   E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, E * sizeof(int), cudaMemcpyHostToDevice));

    /* Copy initial distances CPU --> GPU */
    CUDA_CHECK(cudaMemcpy(d_dist, dist, V * sizeof(int), cudaMemcpyHostToDevice));

    /* ================================================================
     * STEP 4: Calculate grid dimensions
     * ================================================================
     *
     * We need enough blocks to cover all E edges.
     * Number of blocks = ceil(E / THREADS_PER_BLOCK)
     *
     * Example: 1,000,000 edges / 256 threads = 3907 blocks (rounded up)
     */
    int num_blocks = (E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Running Bellman-Ford CUDA...\n");
    printf("  %d vertices, %d edges, up to %d iterations\n", V, E, V - 1);
    printf("  Grid: %d blocks x %d threads = %d GPU threads\n",
           num_blocks, THREADS_PER_BLOCK, num_blocks * THREADS_PER_BLOCK);

    int early_stop_iter = V - 1;

    /* ================================================================
     * STEP 5: Main Bellman-Ford loop
     * ================================================================ */
    for (i = 0; i < V - 1; i++) {
        /* Reset the update flag to 0 on GPU */
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_updated, &zero, sizeof(int),
                               cudaMemcpyHostToDevice));

        /*
         * Launch the GPU kernel.
         *
         * relax_edges_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(...)
         *   - Launches num_blocks blocks, each with THREADS_PER_BLOCK threads
         *   - Total: num_blocks * THREADS_PER_BLOCK threads running in parallel
         *   - Each thread handles one edge
         */
        relax_edges_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_src, d_dest, d_weight, d_dist, E, d_updated
        );

        /*
         * cudaDeviceSynchronize(): wait for all GPU threads to finish
         * before we read d_updated. Without this, we might read the
         * flag before all threads have written to it.
         */
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Read the update flag back to CPU */
        int updated = 0;
        CUDA_CHECK(cudaMemcpy(&updated, d_updated, sizeof(int),
                               cudaMemcpyDeviceToHost));

        if (!updated) {
            early_stop_iter = i + 1;
            printf("  Early termination at iteration %d (no changes)\n",
                   i + 1);
            break;
        }
    }

    if (early_stop_iter == V - 1) {
        printf("  Completed all %d iterations\n", V - 1);
    }

    /* ================================================================
     * STEP 6: Check for negative-weight cycles
     * ================================================================ */
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_neg_cycle, &zero, sizeof(int),
                           cudaMemcpyHostToDevice));

    check_negative_cycle_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_src, d_dest, d_weight, d_dist, E, d_neg_cycle
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int neg_cycle = 0;
    CUDA_CHECK(cudaMemcpy(&neg_cycle, d_neg_cycle, sizeof(int),
                           cudaMemcpyDeviceToHost));

    /* ================================================================
     * STEP 7: Copy results back from GPU --> CPU
     * ================================================================ */
    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(int), cudaMemcpyDeviceToHost));

    /* Free GPU memory */
    cudaFree(d_src);
    cudaFree(d_dest);
    cudaFree(d_weight);
    cudaFree(d_dist);
    cudaFree(d_updated);
    cudaFree(d_neg_cycle);

    /* Free CPU temporary arrays */
    free(h_src);
    free(h_dest);
    free(h_weight);

    if (neg_cycle) {
        printf("  WARNING: Negative-weight cycle detected!\n");
        return -1;
    }

    printf("  No negative-weight cycles detected.\n");
    return 0;
}

/*
 * print_gpu_info - Print basic info about the CUDA device being used
 */
void print_gpu_info() {
    int dev;
    cudaGetDevice(&dev);

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("GPU Device: %s\n", prop.name);
    printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.1f GB\n",
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors    : %d\n", prop.multiProcessorCount);
    printf("  Max threads/block  : %d\n", prop.maxThreadsPerBlock);
    printf("\n");
}

/*
 * main - Entry point
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <graph_file> [source_vertex]\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s graphs/small.txt\n", argv[0]);
        printf("  %s graphs/large.txt 0\n", argv[0]);
        return 1;
    }

    char *graph_file = argv[1];
    int source = 0;
    if (argc >= 3) source = atoi(argv[2]);

    /* Print GPU info */
    print_gpu_info();

    /* Load graph */
    Graph *graph = load_graph(graph_file);
    if (graph == NULL) return 1;
    print_graph_info(graph);

    if (source < 0 || source >= graph->V) {
        fprintf(stderr, "Error: Source vertex %d out of range [0, %d].\n",
                source, graph->V - 1);
        free_graph(graph);
        return 1;
    }

    /* Allocate distance array on CPU */
    int *dist = (int *)malloc(graph->V * sizeof(int));
    if (dist == NULL) {
        fprintf(stderr, "Error: Failed to allocate distance array.\n");
        free_graph(graph);
        return 1;
    }

    /* Run CUDA Bellman-Ford with timing */
    printf("\n");
    double start_time = get_time();
    int result = bellman_ford_cuda(graph, source, dist);
    double end_time = get_time();
    double elapsed = end_time - start_time;

    if (result == -1) {
        printf("\nGraph contains a negative-weight cycle.\n");
        free(dist);
        free_graph(graph);
        return 1;
    }

    /* Print results */
    printf("\n");
    printf("============================================\n");
    printf("  CUDA Bellman-Ford Results\n");
    printf("============================================\n");
    printf("  Source vertex   : %d\n", source);
    printf("  Execution time  : %.6f seconds\n", elapsed);
    printf("  (Includes data transfer to/from GPU)\n");
    printf("============================================\n");

    print_distances(dist, graph->V, 20);

    int reachable = 0, i;
    for (i = 0; i < graph->V; i++) {
        if (dist[i] < INF) reachable++;
    }
    printf("Reachable vertices: %d out of %d\n\n", reachable, graph->V);

    /* Save distances */
    save_distances("results/cuda_distances.txt", dist, graph->V);

    /* Verify against serial results */
    int serial_V;
    int *serial_dist = load_distances("results/serial_distances.txt", &serial_V);
    if (serial_dist != NULL) {
        if (serial_V == graph->V) {
            printf("\nVerifying against serial results...\n");
            verify_distances(serial_dist, dist, graph->V);
        } else {
            printf("Warning: Serial results have different vertex count."
                   " Skipping verification.\n");
        }
        free(serial_dist);
    } else {
        printf("Note: Run serial version first to enable "
               "correctness verification.\n");
    }

    /* Cleanup */
    free(dist);
    free_graph(graph);

    return 0;
}
