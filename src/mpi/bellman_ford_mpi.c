/*
 * bellman_ford_mpi.c - MPI Distributed Bellman-Ford (Message Passing)
 * ===================================================================
 *
 * This version uses MPI to distribute the Bellman-Ford algorithm
 * across multiple processes (potentially on different machines).
 *
 * Parallelization Strategy:
 *   Unlike OpenMP (shared memory), MPI processes have SEPARATE memory.
 *   They can only communicate by sending and receiving messages.
 *
 *   How we split the work:
 *     1. Rank 0 (master) loads the graph and broadcasts it to all ranks
 *     2. Each rank gets a portion of the edges to relax
 *        (rank i handles edges from i*chunk to (i+1)*chunk)
 *     3. Each rank relaxes its edges using its LOCAL copy of dist[]
 *     4. MPI_Allreduce with MPI_MIN combines all dist[] arrays
 *        (every rank gets the global minimum distance for each vertex)
 *     5. Repeat for V-1 iterations (or until no changes)
 *
 *   MPI_Allreduce(local_dist, global_dist, V, MPI_INT, MPI_MIN, ...)
 *     - Takes local_dist from each rank
 *     - Computes element-wise minimum across all ranks
 *     - Stores result in global_dist on EVERY rank
 *     - This is the key synchronization step
 *
 * Early Termination:
 *   Each rank checks if it made any updates. MPI_Allreduce with MPI_LOR
 *   (logical OR) tells all ranks whether ANY rank made an update.
 *   If no rank updated anything, we stop.
 *
 * Usage:
 *   mpiexec -n 4 ./bellman_ford_mpi <graph_file> [source_vertex]
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "../common/graph.h"
#include "../common/timer.h"
#include "../common/utils.h"

/*
 * bellman_ford_mpi - Run the MPI parallel Bellman-Ford algorithm
 *
 * Every rank calls this function. Each rank relaxes its portion of edges,
 * then all ranks synchronize using MPI_Allreduce.
 *
 * @graph:       pointer to the graph (loaded on all ranks)
 * @source:      source vertex (0-indexed)
 * @dist:        output distance array of size V (caller allocates)
 * @rank:        this process's MPI rank
 * @num_procs:   total number of MPI processes
 *
 * Returns: 0 if successful, -1 if a negative cycle is detected
 */
int bellman_ford_mpi(Graph *graph, int source, int *dist,
                     int rank, int num_procs) {
    int V = graph->V;
    int E = graph->E;
    Edge *edges = graph->edges;
    int i, j;

    /*
     * Figure out which edges this rank is responsible for.
     * We split E edges as evenly as possible across num_procs ranks.
     *
     * Example with 10 edges and 3 ranks:
     *   Rank 0: edges 0-3  (4 edges, gets the extra one)
     *   Rank 1: edges 4-6  (3 edges)
     *   Rank 2: edges 7-9  (3 edges)
     */
    int chunk = E / num_procs;
    int remainder = E % num_procs;

    int my_start, my_end;
    if (rank < remainder) {
        /* First 'remainder' ranks get one extra edge */
        my_start = rank * (chunk + 1);
        my_end = my_start + chunk + 1;
    } else {
        my_start = remainder * (chunk + 1) + (rank - remainder) * chunk;
        my_end = my_start + chunk;
    }

    if (rank == 0) {
        printf("Running Bellman-Ford MPI with %d processes...\n", num_procs);
        printf("  %d vertices, %d edges, up to %d iterations\n", V, E, V - 1);
        printf("  Edge distribution: ~%d edges per process\n", chunk);
    }

    /* ================================================================
     * STEP 1: Initialize distances (all ranks do this identically)
     * ================================================================ */
    for (i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    /*
     * Temporary buffer for MPI_Allreduce output.
     * MPI_Allreduce needs separate send and receive buffers
     * (unless using MPI_IN_PLACE, but that's less portable).
     */
    int *new_dist = (int *)malloc(V * sizeof(int));
    if (new_dist == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate new_dist buffer.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    /* ================================================================
     * STEP 2: Relax edges in parallel, synchronize after each iteration
     * ================================================================ */
    int early_stop_iter = V - 1;

    for (i = 0; i < V - 1; i++) {
        int local_updated = 0;

        /*
         * Each rank relaxes only its assigned edges.
         * All ranks read from the SAME dist[] (synchronized from last round).
         */
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
         * SYNCHRONIZATION STEP - the heart of the MPI approach
         *
         * MPI_Allreduce takes dist[] from each rank and computes
         * the element-wise MINIMUM. The result goes into new_dist[]
         * on ALL ranks.
         *
         * Why MPI_MIN? If rank 0 found dist[5] = 100 and rank 1
         * found dist[5] = 80, the correct answer is min(100, 80) = 80.
         * This is exactly what Bellman-Ford needs.
         */
        MPI_Allreduce(dist, new_dist, V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        /* Swap: new_dist becomes the current dist for next iteration */
        memcpy(dist, new_dist, V * sizeof(int));

        /*
         * Early termination: check if ANY rank made an update.
         * MPI_Allreduce with MPI_LOR = logical OR across all ranks.
         */
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
     * STEP 3: Check for negative-weight cycles (distributed check)
     * ================================================================
     * Each rank checks its edges. If any rank finds a shorter path,
     * there's a negative cycle. We combine results with MPI_LOR.
     */
    int local_neg_cycle = 0;

    for (j = my_start; j < my_end; j++) {
        int u = edges[j].src;
        int v = edges[j].dest;
        int w = edges[j].weight;

        if (dist[u] != INF && dist[u] + w < dist[v]) {
            local_neg_cycle = 1;
            break;  /* one is enough */
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
 *
 * MPI programs start by calling MPI_Init and end with MPI_Finalize.
 * Everything between those calls runs on ALL ranks simultaneously.
 * We use rank 0 for all output so messages don't get interleaved.
 */
int main(int argc, char *argv[]) {
    int rank, num_procs;

    /* Initialize MPI - must be called before any other MPI function */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpiexec -n <procs> %s <graph_file> [source_vertex]\n",
                   argv[0]);
            printf("\nExamples:\n");
            printf("  mpiexec -n 4 %s graphs/small.txt\n", argv[0]);
            printf("  mpiexec -n 8 %s graphs/large.txt 0\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *graph_file = argv[1];
    int source = 0;
    if (argc >= 3) source = atoi(argv[2]);

    /*
     * ALL ranks load the graph.
     *
     * Alternative: rank 0 loads and broadcasts with MPI_Bcast.
     * But for simplicity (and because graph files aren't huge),
     * we let every rank read the file independently.
     * This avoids complex serialization of the graph struct.
     */
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

    /* Synchronize all ranks before timing */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Run MPI Bellman-Ford with timing */
    if (rank == 0) printf("\n");
    double start_time = MPI_Wtime();  /* MPI's own high-res timer */
    int result = bellman_ford_mpi(graph, source, dist, rank, num_procs);
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

    /*
     * Only rank 0 prints results and saves distances.
     * All ranks have the same dist[] due to MPI_Allreduce,
     * so we only need one rank to output.
     */
    if (rank == 0) {
        printf("\n");
        printf("============================================\n");
        printf("  MPI Bellman-Ford Results\n");
        printf("============================================\n");
        printf("  Source vertex   : %d\n", source);
        printf("  Processes used  : %d\n", num_procs);
        printf("  Execution time  : %.6f seconds\n", elapsed);
        printf("============================================\n");

        print_distances(dist, graph->V, 20);

        /* Count reachable */
        int reachable = 0, i;
        for (i = 0; i < graph->V; i++) {
            if (dist[i] < INF) reachable++;
        }
        printf("Reachable vertices: %d out of %d\n\n", reachable, graph->V);

        /* Save distances */
        save_distances("results/mpi_distances.txt", dist, graph->V);

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
