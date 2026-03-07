/*
 * bellman_ford_serial.c - Serial (Single-Threaded) Bellman-Ford Algorithm
 * =======================================================================
 *
 * This is the BASELINE implementation. Every parallel version (OpenMP, MPI,
 * Hybrid, CUDA) will be compared against this for:
 *   - CORRECTNESS: parallel distances must match serial distances exactly
 *   - PERFORMANCE: speedup = serial_time / parallel_time
 *
 * The Bellman-Ford Algorithm:
 * --------------------------
 * Given a weighted directed graph G(V, E) and a source vertex s, find the
 * shortest distances from s to all other vertices.
 *
 * Unlike Dijkstra's algorithm (which fails with negative edges), Bellman-Ford
 * handles negative edge weights and can detect negative-weight cycles.
 *
 * Algorithm Steps:
 *   Step 1: Initialize distances
 *           dist[source] = 0
 *           dist[every other vertex] = INF (infinity)
 *
 *   Step 2: Relax all edges, repeated (V-1) times
 *           For each edge (u, v, w):
 *             if dist[u] + w < dist[v]:
 *               dist[v] = dist[u] + w
 *
 *   Step 3: Check for negative-weight cycles
 *           For each edge (u, v, w):
 *             if dist[u] + w < dist[v]:
 *               --> negative cycle exists!
 *
 * Why V-1 iterations?
 *   The shortest path from source to any vertex has at most V-1 edges
 *   (a path with more edges would revisit a vertex = cycle).
 *   After i iterations, we've found all shortest paths using <= i edges.
 *   So after V-1 iterations, all shortest paths are found.
 *
 * Time Complexity:  O(V * E) -- V-1 iterations, each scanning all E edges
 * Space Complexity: O(V)     -- just the distance array
 *
 * Optimization: Early termination
 *   If no distance was updated in an entire iteration, we're done early.
 *   This often saves many iterations in practice.
 *
 * Usage:
 *   ./bellman_ford_serial <graph_file> [source_vertex]
 *   Default source vertex is 0.
 *
 * Output:
 *   - Prints shortest distances (first 20 vertices for large graphs)
 *   - Saves full distance array to results/serial_distances.txt
 *   - Reports execution time
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/graph.h"
#include "../common/timer.h"
#include "../common/utils.h"

/*
 * bellman_ford_serial - Run the serial Bellman-Ford algorithm
 *
 * @graph:  pointer to the loaded graph
 * @source: source vertex (0-indexed)
 * @dist:   output array of size V (caller allocates, we fill it in)
 *
 * Returns: 0 if successful, -1 if a negative cycle is detected
 */
int bellman_ford_serial(Graph *graph, int source, int *dist) {
    int V = graph->V;
    int E = graph->E;
    Edge *edges = graph->edges;

    int i, j;

    /* ================================================================
     * STEP 1: Initialize distances
     * ================================================================
     * Set distance to source = 0, everything else = INF (unreachable).
     * This is the starting point: we know how to reach the source
     * (with distance 0), but nothing else yet.
     */
    for (i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    /* ================================================================
     * STEP 2: Relax all edges, repeat (V-1) times
     * ================================================================
     * In each iteration, we scan ALL edges and try to improve distances.
     *
     * "Relaxation" means: if we found a shorter path to vertex v
     * by going through vertex u, update dist[v].
     *
     * Why this works: After iteration i, we've correctly computed
     * the shortest paths that use at most (i+1) edges.
     * After V-1 iterations, all shortest paths are found (since the
     * longest shortest path has at most V-1 edges).
     *
     * Optimization: If no distance changes in an entire iteration,
     * all shortest paths are already found - we can stop early.
     * This is a HUGE speedup for many real-world graphs.
     */
    printf("Running Bellman-Ford serial from source vertex %d...\n", source);
    printf("  %d vertices, %d edges, up to %d iterations\n", V, E, V - 1);

    int early_stop_iter = V - 1;  /* track which iteration we stopped at */

    for (i = 0; i < V - 1; i++) {
        int updated = 0;  /* flag: was any distance updated this iteration? */

        /* Scan all edges and try to relax */
        for (j = 0; j < E; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int w = edges[j].weight;

            /*
             * Relaxation condition:
             *   1. dist[u] != INF  --> we must have actually reached u
             *   2. dist[u] + w < dist[v]  --> going through u is shorter
             *
             * We check dist[u] != INF to avoid overflow (INF + w could wrap)
             * and because if we haven't reached u, we can't reach v through u.
             */
            if (dist[u] != INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                updated = 1;
            }
        }

        /* Early termination: if nothing changed, we're done! */
        if (!updated) {
            early_stop_iter = i + 1;
            printf("  Early termination at iteration %d (no changes)\n", i + 1);
            break;
        }
    }

    if (early_stop_iter == V - 1) {
        printf("  Completed all %d iterations\n", V - 1);
    }

    /* ================================================================
     * STEP 3: Check for negative-weight cycles
     * ================================================================
     * If we can STILL relax an edge after V-1 iterations, that means
     * there's a negative-weight cycle reachable from the source.
     *
     * Why? After V-1 iterations we should have found ALL shortest paths.
     * If an edge can still be relaxed, it means distances keep decreasing
     * forever (the cycle has negative total weight).
     *
     * In the real world, negative cycles mean:
     *   - A routing loop where packets endlessly circle
     *   - An arbitrage opportunity in currency exchange
     */
    for (j = 0; j < E; j++) {
        int u = edges[j].src;
        int v = edges[j].dest;
        int w = edges[j].weight;

        if (dist[u] != INF && dist[u] + w < dist[v]) {
            printf("  WARNING: Negative-weight cycle detected!\n");
            printf("  Edge %d --> %d (weight %d) can still be relaxed.\n", u, v, w);
            return -1;  /* negative cycle found */
        }
    }

    printf("  No negative-weight cycles detected.\n");
    return 0;  /* success */
}

/*
 * main - Entry point for the serial Bellman-Ford program
 *
 * Loads a graph, runs the algorithm, prints results, saves distances.
 */
int main(int argc, char *argv[]) {
    /* ---- Parse command line ---- */
    if (argc < 2) {
        printf("Usage: %s <graph_file> [source_vertex]\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s graphs/small.txt\n", argv[0]);
        printf("  %s graphs/large.txt 0\n", argv[0]);
        return 1;
    }

    char *graph_file = argv[1];
    int source = 0;  /* default source vertex */
    if (argc >= 3) {
        source = atoi(argv[2]);
    }

    /* ---- Load the graph ---- */
    Graph *graph = load_graph(graph_file);
    if (graph == NULL) {
        return 1;
    }
    print_graph_info(graph);

    /* Validate source vertex */
    if (source < 0 || source >= graph->V) {
        fprintf(stderr, "Error: Source vertex %d is out of range [0, %d].\n",
                source, graph->V - 1);
        free_graph(graph);
        return 1;
    }

    /* ---- Allocate distance array ---- */
    int *dist = (int *)malloc(graph->V * sizeof(int));
    if (dist == NULL) {
        fprintf(stderr, "Error: Failed to allocate distance array.\n");
        free_graph(graph);
        return 1;
    }

    /* ---- Run Bellman-Ford with timing ---- */
    printf("\n");
    double start_time = get_time();
    int result = bellman_ford_serial(graph, source, dist);
    double end_time = get_time();
    double elapsed = end_time - start_time;

    if (result == -1) {
        printf("\nGraph contains a negative-weight cycle. Results are invalid.\n");
        free(dist);
        free_graph(graph);
        return 1;
    }

    /* ---- Print results ---- */
    printf("\n");
    printf("============================================\n");
    printf("  Serial Bellman-Ford Results\n");
    printf("============================================\n");
    printf("  Source vertex   : %d\n", source);
    printf("  Execution time  : %.6f seconds\n", elapsed);
    printf("============================================\n");

    /* Print distances (limit to 20 for large graphs) */
    print_distances(dist, graph->V, 20);

    /* Count reachable vertices */
    int reachable = 0;
    int i;
    for (i = 0; i < graph->V; i++) {
        if (dist[i] < INF) reachable++;
    }
    printf("Reachable vertices: %d out of %d\n\n", reachable, graph->V);

    /* ---- Save distances for later comparison with parallel versions ---- */
    save_distances("results/serial_distances.txt", dist, graph->V);

    /* ---- Cleanup ---- */
    free(dist);
    free_graph(graph);

    printf("\nDone. Use this output to verify parallel implementations.\n");
    return 0;
}
