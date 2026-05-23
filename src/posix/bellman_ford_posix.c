/*
 * bellman_ford_posix.c - POSIX Threads Bellman-Ford
 * =================================================
 *
 * This implementation uses pthreads as a manual shared-memory threading model.
 * It is included as a CPU comparison point beside OpenMP. The edge list is
 * divided across worker threads, and each worker relaxes its assigned range.
 *
 * The implementation uses a double-buffered distance array:
 *   - current[] is read-only during one Bellman-Ford iteration
 *   - next[] receives improved distances
 *
 * Per-vertex mutexes protect concurrent updates to next[v]. This keeps the
 * result deterministic and directly comparable with the serial baseline.
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/graph.h"
#include "../common/timer.h"
#include "../common/utils.h"

typedef struct {
    Graph *graph;
    int *current;
    int *next;
    pthread_mutex_t *vertex_locks;
    int start_edge;
    int end_edge;
    int updated;
} WorkerArgs;

static void *relax_edges(void *arg) {
    WorkerArgs *worker = (WorkerArgs *)arg;
    Edge *edges = worker->graph->edges;
    int *current = worker->current;
    int *next = worker->next;
    int local_updated = 0;

    for (int i = worker->start_edge; i < worker->end_edge; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;
        int w = edges[i].weight;

        if (current[u] != INF) {
            int candidate = current[u] + w;
            if (candidate < next[v]) {
                pthread_mutex_lock(&worker->vertex_locks[v]);
                if (candidate < next[v]) {
                    next[v] = candidate;
                    local_updated = 1;
                }
                pthread_mutex_unlock(&worker->vertex_locks[v]);
            }
        }
    }

    worker->updated = local_updated;
    return NULL;
}

static int run_parallel_relaxation(Graph *graph, int *current, int *next,
                                   pthread_mutex_t *vertex_locks,
                                   int num_threads) {
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    WorkerArgs *args = (WorkerArgs *)calloc(num_threads, sizeof(WorkerArgs));

    if (threads == NULL || args == NULL) {
        fprintf(stderr, "Error: Failed to allocate POSIX worker metadata.\n");
        free(threads);
        free(args);
        return -1;
    }

    int edges_per_thread = (graph->E + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int start = t * edges_per_thread;
        int end = start + edges_per_thread;
        if (end > graph->E) {
            end = graph->E;
        }

        args[t].graph = graph;
        args[t].current = current;
        args[t].next = next;
        args[t].vertex_locks = vertex_locks;
        args[t].start_edge = start;
        args[t].end_edge = end;
        args[t].updated = 0;

        if (pthread_create(&threads[t], NULL, relax_edges, &args[t]) != 0) {
            fprintf(stderr, "Error: Failed to create POSIX thread %d.\n", t);
            for (int j = 0; j < t; j++) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(args);
            return -1;
        }
    }

    int updated = 0;
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
        updated |= args[t].updated;
    }

    free(threads);
    free(args);
    return updated;
}

int bellman_ford_posix(Graph *graph, int source, int *dist, int num_threads) {
    int V = graph->V;
    int E = graph->E;

    if (num_threads < 1) {
        num_threads = 1;
    }
    if (num_threads > E && E > 0) {
        num_threads = E;
    }

    int *current = (int *)malloc(V * sizeof(int));
    int *next = (int *)malloc(V * sizeof(int));
    pthread_mutex_t *vertex_locks =
        (pthread_mutex_t *)malloc(V * sizeof(pthread_mutex_t));

    if (current == NULL || next == NULL || vertex_locks == NULL) {
        fprintf(stderr, "Error: Failed to allocate POSIX Bellman-Ford buffers.\n");
        free(current);
        free(next);
        free(vertex_locks);
        return -1;
    }

    for (int i = 0; i < V; i++) {
        current[i] = INF;
        next[i] = INF;
        pthread_mutex_init(&vertex_locks[i], NULL);
    }
    current[source] = 0;

    printf("Running Bellman-Ford POSIX pthreads with %d threads...\n", num_threads);
    printf("  %d vertices, %d edges, up to %d iterations\n", V, E, V - 1);

    int stopped_early = 0;

    for (int iter = 0; iter < V - 1; iter++) {
        memcpy(next, current, V * sizeof(int));

        int updated = run_parallel_relaxation(
            graph, current, next, vertex_locks, num_threads
        );

        if (updated < 0) {
            for (int i = 0; i < V; i++) {
                pthread_mutex_destroy(&vertex_locks[i]);
            }
            free(current);
            free(next);
            free(vertex_locks);
            return -1;
        }

        if (!updated) {
            stopped_early = 1;
            printf("  Early termination at iteration %d (no changes)\n", iter + 1);
            break;
        }

        int *tmp = current;
        current = next;
        next = tmp;
    }

    if (!stopped_early) {
        printf("  Completed all %d iterations\n", V - 1);
    }

    int has_negative_cycle = 0;
    for (int j = 0; j < E; j++) {
        int u = graph->edges[j].src;
        int v = graph->edges[j].dest;
        int w = graph->edges[j].weight;

        if (current[u] != INF && current[u] + w < current[v]) {
            has_negative_cycle = 1;
            break;
        }
    }

    memcpy(dist, current, V * sizeof(int));

    for (int i = 0; i < V; i++) {
        pthread_mutex_destroy(&vertex_locks[i]);
    }
    free(current);
    free(next);
    free(vertex_locks);

    if (has_negative_cycle) {
        printf("  WARNING: Negative-weight cycle detected!\n");
        return -1;
    }

    printf("  No negative-weight cycles detected.\n");
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <graph_file> [source_vertex] [num_threads]\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s graphs/small.txt\n", argv[0]);
        printf("  %s graphs/large.txt 0 4\n", argv[0]);
        return 1;
    }

    char *graph_file = argv[1];
    int source = 0;
    int num_threads = 4;

    if (argc >= 3) {
        source = atoi(argv[2]);
    }
    if (argc >= 4) {
        num_threads = atoi(argv[3]);
    }

    Graph *graph = load_graph(graph_file);
    if (graph == NULL) {
        return 1;
    }
    print_graph_info(graph);

    if (source < 0 || source >= graph->V) {
        fprintf(stderr, "Error: Source vertex %d out of range [0, %d].\n",
                source, graph->V - 1);
        free_graph(graph);
        return 1;
    }

    int *dist = (int *)malloc(graph->V * sizeof(int));
    if (dist == NULL) {
        fprintf(stderr, "Error: Failed to allocate distance array.\n");
        free_graph(graph);
        return 1;
    }

    printf("\n");
    double start_time = get_time();
    int result = bellman_ford_posix(graph, source, dist, num_threads);
    double end_time = get_time();
    double elapsed = end_time - start_time;

    if (result == -1) {
        printf("\nGraph contains a negative-weight cycle or POSIX execution failed.\n");
        free(dist);
        free_graph(graph);
        return 1;
    }

    printf("\n");
    printf("============================================\n");
    printf("  POSIX pthreads Bellman-Ford Results\n");
    printf("============================================\n");
    printf("  Source vertex   : %d\n", source);
    printf("  Threads used    : %d\n", num_threads);
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

    save_distances("results/posix_distances.txt", dist, graph->V);

    int serial_V;
    int *serial_dist = load_distances("results/serial_distances.txt", &serial_V);
    if (serial_dist != NULL) {
        if (serial_V == graph->V) {
            printf("\nVerifying against serial results...\n");
            verify_distances(serial_dist, dist, graph->V);
        } else {
            printf("Warning: Serial results have different vertex count. Skipping verification.\n");
        }
        free(serial_dist);
    } else {
        printf("Note: Run serial version first to enable correctness verification.\n");
    }

    free(dist);
    free_graph(graph);
    return 0;
}
