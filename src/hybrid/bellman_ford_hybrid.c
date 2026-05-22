/*
 * bellman_ford_hybrid.c - Hybrid MPI+OpenMP Bellman-Ford
 * ======================================================
 *
 * This is the most sophisticated parallel version. It combines:
 *   - MPI for INTER-node parallelism (distributing edges across processes)
 *   - OpenMP for INTRA-node parallelism (threading within each process)
 *
 * Why hybrid?
 *   On a cluster, you might have 4 nodes with 8 cores each.
 *   - Pure MPI with 32 processes: 32 copies of dist[] in memory,
 *     32-way MPI_Allreduce overhead
 *   - Hybrid with 4 MPI ranks x 8 OpenMP threads: only 4 copies of dist[],
 *     4-way MPI_Allreduce (faster), threads share memory within each node
 *
 * How it works:
 *   1. MPI divides edges across processes (same as pure MPI version)
 *   2. Within each process, OpenMP threads parallelize edge relaxation
 *   3. After each iteration, MPI_Allreduce synchronizes across processes
 *   4. OpenMP threads within each process see the updated dist[] (shared memory)
 *
 * This gives us the best of both worlds:
 *   - MPI handles the coarse-grained distribution
 *   - OpenMP handles fine-grained parallelism cheaply (no message passing)
 *
 * Usage:
 *   export OMP_NUM_THREADS=4
 *   mpiexec -n 2 ./bellman_ford_hybrid <graph_file> [source_vertex]
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "../common/graph.h"
#include "../common/timer.h"
#include "../common/utils.h"

/*
 * bellman_ford_hybrid - Run the Hybrid MPI+OpenMP Bellman-Ford algorithm
 *
 * @graph:       pointer to the graph (loaded on all ranks)
 * @source:      source vertex (0-indexed)
 * @dist:        output distance array of size V (caller allocates)
 * @rank:        this process's MPI rank
 * @num_procs:   total number of MPI processes
 * @num_threads: number of OpenMP threads per process
 *
 * Returns: 0 if successful, -1 if a negative cycle is detected
 */
int bellman_ford_hybrid(Graph *graph, int source, int *dist,
                        int rank, int num_procs, int num_threads) {
    int V = graph->V;
    int E = graph->E;
    Edge *edges = graph->edges;
    int i, j;

    /* Set OpenMP threads for this process */
    omp_set_num_threads(num_threads);

    /*
     * Divide edges across MPI processes (same partitioning as pure MPI).
     * Within each process, OpenMP will further parallelize the loop.
     */
    int chunk = E / num_procs;
    int remainder = E % num_procs;

    int my_start, my_end;
    if (rank < remainder) {
        my_start = rank * (chunk + 1);
        my_end = my_start + chunk + 1;
    } else {
        my_start = remainder * (chunk + 1) + (rank - remainder) * chunk;
        my_end = my_start + chunk;
    }

    if (rank == 0) {
        printf("Running Bellman-Ford Hybrid (MPI+OpenMP)...\n");
        printf("  %d MPI processes x %d OpenMP threads = %d total workers\n",
               num_procs, num_threads, num_procs * num_threads);
        printf("  %d vertices, %d edges, up to %d iterations\n", V, E, V - 1);
        printf("  Edge distribution: ~%d edges per MPI process\n",
               my_end - my_start);
    }

    /* ================================================================
     * STEP 1: Initialize distances
     * ================================================================ */
    #pragma omp parallel for
    for (i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    /* Buffer for MPI_Allreduce */
    int *new_dist = (int *)malloc(V * sizeof(int));
    if (new_dist == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate buffer.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    /* ================================================================
     * STEP 2: Relax edges with hybrid parallelism
     * ================================================================
     * Outer loop: MPI iterations with synchronization
     * Inner loop: OpenMP parallel edge relaxation within each process
     */
    int early_stop_iter = V - 1;

    for (i = 0; i < V - 1; i++) {
        int local_updated = 0;

        /*
         * HYBRID PARALLELISM:
         *   - my_start..my_end = this MPI process's edges (MPI level)
         *   - #pragma omp parallel for = split those edges across threads
         *   - reduction(|:local_updated) = combine thread-level update flags
         *
         * So if this MPI rank has edges [1000, 3000) and 4 OpenMP threads:
         *   Thread 0: edges 1000-1499
         *   Thread 1: edges 1500-1999
         *   Thread 2: edges 2000-2499
         *   Thread 3: edges 2500-2999
         *
         * Same race condition approach as pure OpenMP: small races on dist[]
         * are tolerable because MPI_Allreduce with MPI_MIN corrects them.
         */
        #pragma omp parallel for schedule(dynamic, 1024) reduction(|:local_updated)
        for (j = my_start; j < my_end; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int w = edges[j].weight;

            if (dist[u] != INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                local_updated = 1;
            }
        }

        /*
         * MPI synchronization: combine dist[] from all processes.
         * MPI_Allreduce with MPI_MIN gives the correct global minimum
         * for each vertex across all ranks.
         */
        MPI_Allreduce(dist, new_dist, V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        memcpy(dist, new_dist, V * sizeof(int));

        /* Check if any rank (across all threads) made an update */
        int global_updated = 0;
        MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT,
                       MPI_LOR, MPI_COMM_WORLD);

        if (!global_updated) {
            early_stop_iter = i + 1;
            if (rank == 0) {
                printf("  Early termination at iteration %d (no changes)\n",
                       i + 1);
            }
            break;
        }
    }

    if (rank == 0 && early_stop_iter == V - 1) {
        printf("  Completed all %d iterations\n", V - 1);
    }

    /* ================================================================
     * STEP 3: Check for negative-weight cycles (hybrid parallel)
     * ================================================================ */
    int local_neg_cycle = 0;

    #pragma omp parallel for reduction(|:local_neg_cycle)
    for (j = my_start; j < my_end; j++) {
        int u = edges[j].src;
        int v = edges[j].dest;
        int w = edges[j].weight;

        if (dist[u] != INF && dist[u] + w < dist[v]) {
            local_neg_cycle = 1;
        }
    }

    int global_neg_cycle = 0;
    MPI_Allreduce(&local_neg_cycle, &global_neg_cycle, 1, MPI_INT,
                   MPI_LOR, MPI_COMM_WORLD);

    free(new_dist);

    if (global_neg_cycle) {
        if (rank == 0) {
            printf("  WARNING: Negative-weight cycle detected!\n");
        }
        return -1;
    }

    if (rank == 0) {
        printf("  No negative-weight cycles detected.\n");
    }

    return 0;
}

/*
 * main - Entry point
 */
int main(int argc, char *argv[]) {
    int rank, num_procs;

    /*
     * MPI_Init_thread: For hybrid programs, we tell MPI what level
     * of thread safety we need.
     *
     * MPI_THREAD_FUNNELED means: only the main thread makes MPI calls.
     * This is fine because our OpenMP parallel regions don't call MPI.
     * The MPI_Allreduce calls happen outside the #pragma omp sections.
     */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "Warning: MPI does not support MPI_THREAD_FUNNELED."
                    " Hybrid may not work correctly.\n");
        }
    }

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpiexec -n <procs> %s <graph_file> "
                   "[source_vertex] [threads_per_proc]\n", argv[0]);
            printf("\nExamples:\n");
            printf("  mpiexec -n 2 %s graphs/large.txt 0 4\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *graph_file = argv[1];
    int source = 0;
    int num_threads = omp_get_max_threads();

    if (argc >= 3) source = atoi(argv[2]);
    if (argc >= 4) num_threads = atoi(argv[3]);

    /* All ranks load the graph */
    Graph *graph = load_graph(graph_file);
    if (graph == NULL) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        print_graph_info(graph);
    }

    if (source < 0 || source >= graph->V) {
        if (rank == 0) {
            fprintf(stderr, "Error: Source vertex %d out of range [0, %d].\n",
                    source, graph->V - 1);
        }
        free_graph(graph);
        MPI_Finalize();
        return 1;
    }

    /* Allocate distance array */
    int *dist = (int *)malloc(graph->V * sizeof(int));
    if (dist == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Error: Failed to allocate distance array.\n");
        }
        free_graph(graph);
        MPI_Finalize();
        return 1;
    }

    /* Synchronize before timing */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Run Hybrid Bellman-Ford */
    if (rank == 0) printf("\n");
    double start_time = MPI_Wtime();
    int result = bellman_ford_hybrid(graph, source, dist, rank,
                                     num_procs, num_threads);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (result == -1) {
        if (rank == 0) {
            printf("\nGraph contains a negative-weight cycle.\n");
        }
        free(dist);
        free_graph(graph);
        MPI_Finalize();
        return 1;
    }

    /* Only rank 0 outputs results */
    if (rank == 0) {
        printf("\n");
        printf("============================================\n");
        printf("  Hybrid (MPI+OpenMP) Bellman-Ford Results\n");
        printf("============================================\n");
        printf("  Source vertex   : %d\n", source);
        printf("  MPI processes   : %d\n", num_procs);
        printf("  OpenMP threads  : %d per process\n", num_threads);
        printf("  Total workers   : %d\n", num_procs * num_threads);
        printf("  Execution time  : %.6f seconds\n", elapsed);
        printf("============================================\n");

        print_distances(dist, graph->V, 20);

        int reachable = 0, i;
        for (i = 0; i < graph->V; i++) {
            if (dist[i] < INF) reachable++;
        }
        printf("Reachable vertices: %d out of %d\n\n", reachable, graph->V);

        /* Save distances */
        save_distances("results/hybrid_distances.txt", dist, graph->V);

        /* Verify against serial results */
        int serial_V;
        int *serial_dist = load_distances("results/serial_distances.txt",
                                           &serial_V);
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
    }

    /* Cleanup */
    free(dist);
    free_graph(graph);
    MPI_Finalize();

    return 0;
}
