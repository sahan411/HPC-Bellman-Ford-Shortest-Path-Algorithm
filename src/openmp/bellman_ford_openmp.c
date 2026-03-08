/*
 * bellman_ford_openmp.c - OpenMP Bellman-Ford (Shared Memory)
 * ============================================================
 *
 * Parallel strategy used here:
 * - Each iteration uses the previous distance array as read-only input.
 * - Each thread relaxes all assigned edges into its own local distance array
 *   (no races during relaxation).
 * - Local arrays are merged with element-wise minimum to produce next distances.
 * - Early termination uses OpenMP logical-or reduction on an "updated" flag.
 *
 * Usage:
 *   ./bellman_ford_openmp <graph_file> [source_vertex] [num_threads]
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../common/graph.h"
#include "../common/timer.h"
#include "../common/utils.h"

/*
 * bellman_ford_openmp
 * Returns 0 on success, -1 if a negative cycle is detected.
 */
int bellman_ford_openmp(Graph *graph, int source, int *dist) {
    int V = graph->V;
    int E = graph->E;
    Edge *edges = graph->edges;

    int max_threads = omp_get_max_threads();
    int *thread_buffers = (int *)malloc((size_t)max_threads * (size_t)V * sizeof(int));
    int *curr_dist = (int *)malloc((size_t)V * sizeof(int));
    int *next_dist = (int *)malloc((size_t)V * sizeof(int));

    if (thread_buffers == NULL || curr_dist == NULL || next_dist == NULL) {
        fprintf(stderr, "Error: Failed to allocate OpenMP working buffers.\n");
        free(thread_buffers);
        free(curr_dist);
        free(next_dist);
        return -1;
    }

    for (int i = 0; i < V; i++) {
        curr_dist[i] = INF;
    }
    curr_dist[source] = 0;

    printf("Running Bellman-Ford OpenMP from source vertex %d...\n", source);
    printf("  %d vertices, %d edges, up to %d iterations\n", V, E, V - 1);
    printf("  OpenMP threads: %d\n", max_threads);

    int early_stop_iter = V - 1;

    for (int iter = 0; iter < V - 1; iter++) {
        int updated = 0;

#pragma omp parallel reduction(||:updated)
        {
            int tid = omp_get_thread_num();
            int *local = thread_buffers + ((size_t)tid * (size_t)V);

            for (int i = 0; i < V; i++) {
                local[i] = curr_dist[i];
            }

#pragma omp for schedule(static)
            for (int e = 0; e < E; e++) {
                int u = edges[e].src;
                int v = edges[e].dest;
                int w = edges[e].weight;

                if (curr_dist[u] != INF) {
                    int candidate = curr_dist[u] + w;
                    if (candidate < local[v]) {
                        local[v] = candidate;
                        updated = 1;
                    }
                }
            }
        }

#pragma omp parallel for schedule(static)
        for (int i = 0; i < V; i++) {
            int best = curr_dist[i];
            for (int t = 0; t < max_threads; t++) {
                int value = thread_buffers[(size_t)t * (size_t)V + (size_t)i];
                if (value < best) {
                    best = value;
                }
            }
            next_dist[i] = best;
        }

        int *tmp = curr_dist;
        curr_dist = next_dist;
        next_dist = tmp;

        if (!updated) {
            early_stop_iter = iter + 1;
            printf("  Early termination at iteration %d (no changes)\n", iter + 1);
            break;
        }
    }

    if (early_stop_iter == V - 1) {
        printf("  Completed all %d iterations\n", V - 1);
    }

    for (int e = 0; e < E; e++) {
        int u = edges[e].src;
        int v = edges[e].dest;
        int w = edges[e].weight;
        if (curr_dist[u] != INF && curr_dist[u] + w < curr_dist[v]) {
            printf("  WARNING: Negative-weight cycle detected!\n");
            printf("  Edge %d --> %d (weight %d) can still be relaxed.\n", u, v, w);
            free(thread_buffers);
            free(curr_dist);
            free(next_dist);
            return -1;
        }
    }

    for (int i = 0; i < V; i++) {
        dist[i] = curr_dist[i];
    }

    free(thread_buffers);
    free(curr_dist);
    free(next_dist);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <graph_file> [source_vertex] [num_threads]\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s graphs/small.txt\n", argv[0]);
        printf("  %s graphs/large.txt 0 8\n", argv[0]);
        return 1;
    }

    char *graph_file = argv[1];
    int source = 0;
    if (argc >= 3) {
        source = atoi(argv[2]);
    }

    if (argc >= 4) {
        int num_threads = atoi(argv[3]);
        if (num_threads > 0) {
            omp_set_dynamic(0);
            omp_set_num_threads(num_threads);
        }
    }

    Graph *graph = load_graph(graph_file);
    if (graph == NULL) {
        return 1;
    }
    print_graph_info(graph);

    if (source < 0 || source >= graph->V) {
        fprintf(stderr, "Error: Source vertex %d is out of range [0, %d].\n", source, graph->V - 1);
        free_graph(graph);
        return 1;
    }

    int *dist = (int *)malloc((size_t)graph->V * sizeof(int));
    if (dist == NULL) {
        fprintf(stderr, "Error: Failed to allocate distance array.\n");
        free_graph(graph);
        return 1;
    }

    printf("\n");
    double start_time = get_time();
    int result = bellman_ford_openmp(graph, source, dist);
    double end_time = get_time();
    double elapsed = end_time - start_time;

    if (result == -1) {
        printf("\nGraph contains a negative-weight cycle. Results are invalid.\n");
        free(dist);
        free_graph(graph);
        return 1;
    }

    printf("\n");
    printf("============================================\n");
    printf("  OpenMP Bellman-Ford Results\n");
    printf("============================================\n");
    printf("  Source vertex   : %d\n", source);
    printf("  OpenMP threads  : %d\n", omp_get_max_threads());
    printf("  Execution time  : %.6f seconds\n", elapsed);
    printf("============================================\n");

    print_distances(dist, graph->V, 20);

    int reachable = 0;
    for (int i = 0; i < graph->V; i++) {
        if (dist[i] < INF) {
            reachable++;
        }
    }
    printf("Reachable vertices: %d out of %d\n\n", reachable, graph->V);

    save_distances("results/openmp_distances.txt", dist, graph->V);

    int ref_v = 0;
    int *serial_dist = load_distances("results/serial_distances.txt", &ref_v);
    if (serial_dist != NULL) {
        if (ref_v == graph->V) {
            printf("Comparing with serial ground truth...\n");
            verify_distances(serial_dist, dist, graph->V);
        } else {
            printf("Skipping verification: serial result size mismatch (serial=%d, current=%d).\n",
                   ref_v, graph->V);
        }
        free(serial_dist);
    } else {
        printf("Serial baseline file not found. Run serial first to enable verification.\n");
    }

    free(dist);
    free_graph(graph);

    printf("\nDone.\n");
    return 0;
}
