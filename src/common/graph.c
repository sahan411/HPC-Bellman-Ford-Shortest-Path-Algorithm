/*
 * graph.c - Graph Loading, Saving, and Memory Management
 * ======================================================
 *
 * Implementation of graph operations defined in graph.h.
 * These functions are used by ALL versions of the algorithm.
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#include "graph.h"
#include <string.h>

/*
 * create_graph - Allocate a new graph with V vertices and E edges
 *
 * This allocates memory for the graph structure and the edge array.
 * The edges themselves are NOT initialized - the caller must fill them in.
 *
 * Returns NULL if memory allocation fails.
 */
Graph* create_graph(int V, int E) {
    /* Allocate the graph structure itself */
    Graph *graph = (Graph *)malloc(sizeof(Graph));
    if (graph == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for graph structure.\n");
        return NULL;
    }

    graph->V = V;
    graph->E = E;

    /* Allocate the edge array */
    graph->edges = (Edge *)malloc(E * sizeof(Edge));
    if (graph->edges == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for %d edges.\n", E);
        free(graph);
        return NULL;
    }

    return graph;
}

/*
 * free_graph - Release all memory used by a graph
 *
 * Always call this when you're done with a graph to avoid memory leaks.
 * Safe to call with NULL (does nothing).
 */
void free_graph(Graph *graph) {
    if (graph != NULL) {
        if (graph->edges != NULL) {
            free(graph->edges);
        }
        free(graph);
    }
}

/*
 * load_graph - Read a graph from a text file
 *
 * Expected file format:
 *   First line:  V E
 *   Next E lines: src dest weight
 *
 * Example:
 *   4 5
 *   0 1 6
 *   0 2 7
 *   1 2 8
 *   1 3 -4
 *   2 3 9
 *
 * Returns NULL if file cannot be opened or format is invalid.
 */
Graph* load_graph(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open graph file '%s'.\n", filename);
        return NULL;
    }

    int V, E;

    /* Read the first line: number of vertices and edges */
    if (fscanf(fp, "%d %d", &V, &E) != 2) {
        fprintf(stderr, "Error: Invalid format in '%s'. Expected 'V E' on first line.\n", filename);
        fclose(fp);
        return NULL;
    }

    /* Basic validation */
    if (V <= 0 || E <= 0) {
        fprintf(stderr, "Error: Invalid graph size V=%d, E=%d. Both must be positive.\n", V, E);
        fclose(fp);
        return NULL;
    }

    /* Create the graph */
    Graph *graph = create_graph(V, E);
    if (graph == NULL) {
        fclose(fp);
        return NULL;
    }

    /* Read each edge: src dest weight */
    int i;
    for (i = 0; i < E; i++) {
        if (fscanf(fp, "%d %d %d", &graph->edges[i].src, 
                   &graph->edges[i].dest, &graph->edges[i].weight) != 3) {
            fprintf(stderr, "Error: Failed to read edge %d from '%s'.\n", i + 1, filename);
            free_graph(graph);
            fclose(fp);
            return NULL;
        }

        /* Validate vertex indices are in range */
        if (graph->edges[i].src < 0 || graph->edges[i].src >= V ||
            graph->edges[i].dest < 0 || graph->edges[i].dest >= V) {
            fprintf(stderr, "Error: Edge %d has invalid vertex index (src=%d, dest=%d, V=%d).\n",
                    i + 1, graph->edges[i].src, graph->edges[i].dest, V);
            free_graph(graph);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    printf("Graph loaded: %d vertices, %d edges from '%s'\n", V, E, filename);
    return graph;
}

/*
 * save_graph - Write a graph to a text file
 *
 * Writes in the same format that load_graph expects,
 * so you can save a generated graph and load it later.
 */
void save_graph(const Graph *graph, const char *filename) {
    if (graph == NULL) {
        fprintf(stderr, "Error: Cannot save NULL graph.\n");
        return;
    }

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot create file '%s'.\n", filename);
        return;
    }

    /* Write header: V E */
    fprintf(fp, "%d %d\n", graph->V, graph->E);

    /* Write each edge: src dest weight */
    int i;
    for (i = 0; i < graph->E; i++) {
        fprintf(fp, "%d %d %d\n", graph->edges[i].src, 
                graph->edges[i].dest, graph->edges[i].weight);
    }

    fclose(fp);
    printf("Graph saved: %d vertices, %d edges to '%s'\n", graph->V, graph->E, filename);
}

/*
 * print_graph_info - Print a summary of the graph
 *
 * Shows vertex/edge counts and the first few edges.
 * Useful for sanity-checking that a graph loaded correctly.
 */
void print_graph_info(const Graph *graph) {
    if (graph == NULL) {
        printf("Graph: NULL\n");
        return;
    }

    printf("============================================\n");
    printf("  Graph Summary\n");
    printf("============================================\n");
    printf("  Vertices : %d\n", graph->V);
    printf("  Edges    : %d\n", graph->E);
    printf("--------------------------------------------\n");

    /* Print first 10 edges (don't flood terminal for large graphs) */
    int show = graph->E < 10 ? graph->E : 10;
    printf("  First %d edges:\n", show);
    int i;
    for (i = 0; i < show; i++) {
        printf("    %d --> %d  (weight: %d)\n", 
               graph->edges[i].src, graph->edges[i].dest, graph->edges[i].weight);
    }
    if (graph->E > 10) {
        printf("    ... and %d more edges\n", graph->E - 10);
    }
    printf("============================================\n");
}
