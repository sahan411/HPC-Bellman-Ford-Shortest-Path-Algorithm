/*
 * bellman_ford_mpi.c - MPI (Distributed Memory) Bellman-Ford Algorithm
 * =======================================================================
 *
 * This implementation uses MPI to parallelize the Bellman-Ford algorithm
 * across multiple processes.
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

int main(int argc, char *argv[]) {
    int rank, size;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Parse command line ---- */
    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s <graph_file> [source_vertex]\n", argv[0]);
            printf("\nExamples:\n");
            printf("  mpirun -np 4 %s graphs/small.txt\n", argv[0]);
            printf("  mpirun -np 4 %s graphs/large.txt 0\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *graph_file = argv[1];
    int source = 0;  /* default source vertex */
    if (argc >= 3) {
        source = atoi(argv[2]);
    }

    Graph *graph = NULL;
    int V = 0, E = 0;

    /* ---- Rank 0 loads the graph ---- */
    if (rank == 0) {
        graph = load_graph(graph_file);
        if (graph == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        V = graph->V;
        E = graph->E;
        
        /* Validate source vertex */
        if (source < 0 || source >= V) {
            fprintf(stderr, "Error: Source vertex %d is out of range [0, %d].\n",
                    source, V - 1);
            free_graph(graph);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        print_graph_info(graph);
    } else {
        /* Other ranks allocate a structure to hold the graph */
        graph = (Graph *)malloc(sizeof(Graph));
    }

    /* ---- Broadcast graph details ---- */
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        graph->V = V;
        graph->E = E;
        graph->edges = (Edge *)malloc(E * sizeof(Edge));
    }

    /* Broadcast the edge list. Edge struct contains 3 ints, so we can send as bytes. */
    MPI_Bcast(graph->edges, E * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ---- Determine edge partition for this process ---- */
    int edges_per_proc = E / size;
    int remainder = E % size;
    
    int local_start = rank * edges_per_proc + (rank < remainder ? rank : remainder);
    int local_count = edges_per_proc + (rank < remainder ? 1 : 0);
    int local_end = local_start + local_count;

    /* ---- Allocate distance arrays ---- */
    /* We need dist and next_dist to ensure correctly parallelized edge relaxations */
    int *dist = (int *)malloc(V * sizeof(int));
    int *next_dist = (int *)malloc(V * sizeof(int));
    if (dist == NULL || next_dist == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate distance arrays.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    /* ---- Run Bellman-Ford with timing ---- */
    double start_time = MPI_Wtime();

    if (rank == 0) {
        printf("\nRunning Bellman-Ford MPI from source vertex %d...\n", source);
        printf("  %d vertices, %d edges, up to %d iterations on %d processes\n", V, E, V - 1, size);
    }

    int early_stop_iter = V - 1;

    for (int i = 0; i < V - 1; i++) {
        int local_updated = 0;
        int global_updated = 0;

        /* Initialize next_dist with current dist */
        memcpy(next_dist, dist, V * sizeof(int));

        /* Relax local edges */
        for (int j = local_start; j < local_end; j++) {
            int u = graph->edges[j].src;
            int v = graph->edges[j].dest;
            int w = graph->edges[j].weight;

            if (dist[u] != INF && dist[u] + w < next_dist[v]) {
                next_dist[v] = dist[u] + w;
                local_updated = 1;
            }
        }

        /* Merge distances across all processes. 
         * MPI_MIN ensures we take the shortest path found by any process. */
        MPI_Allreduce(next_dist, dist, V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        /* Merge updated flags to determine early termination globally.
         * MPI_MAX essentially acts as a logical OR. */
        MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (!global_updated) {
            early_stop_iter = i + 1;
            if (rank == 0) {
                printf("  Early termination at iteration %d (no changes)\n", i + 1);
            }
            break;
        }
    }

    if (rank == 0 && early_stop_iter == V - 1) {
        printf("  Completed all %d iterations\n", V - 1);
    }

    /* ---- Check for negative-weight cycles ---- */
    int local_has_negative_cycle = 0;
    int global_has_negative_cycle = 0;

    for (int j = local_start; j < local_end; j++) {
        int u = graph->edges[j].src;
        int v = graph->edges[j].dest;
        int w = graph->edges[j].weight;

        if (dist[u] != INF && dist[u] + w < dist[v]) {
            local_has_negative_cycle = 1;
            break;
        }
    }

    MPI_Allreduce(&local_has_negative_cycle, &global_has_negative_cycle, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (global_has_negative_cycle) {
        if (rank == 0) {
            printf("  WARNING: Negative-weight cycle detected!\n");
            printf("\nGraph contains a negative-weight cycle. Results are invalid.\n");
        }
        free(dist);
        free(next_dist);
        free_graph(graph);
        MPI_Finalize();
        return 1;
    }

    /* ---- Process 0 Prints results and Saves ---- */
    if (rank == 0) {
        printf("  No negative-weight cycles detected.\n");
        printf("\n============================================\n");
        printf("  MPI Bellman-Ford Results\n");
        printf("============================================\n");
        printf("  Source vertex   : %d\n", source);
        printf("  Execution time  : %.6f seconds\n", elapsed);
        printf("============================================\n");

        print_distances(dist, V, 20);

        int reachable = 0;
        for (int i = 0; i < V; i++) {
            if (dist[i] < INF) reachable++;
        }
        printf("Reachable vertices: %d out of %d\n\n", reachable, V);

        save_distances("results/mpi_distances.txt", dist, V);
    }

    /* ---- Cleanup ---- */
    free(dist);
    free(next_dist);
    free_graph(graph);

    if (rank == 0) {
        printf("Done. Output saved to results/mpi_distances.txt\n");
    }

    MPI_Finalize();
    return 0;
}
