/*
 * graph.h - Common Graph Data Structures
 * =======================================
 * 
 * This header defines the graph representation used by ALL versions
 * (serial, OpenMP, MPI, hybrid, CUDA) of the Bellman-Ford algorithm.
 *
 * We use an EDGE LIST representation because Bellman-Ford's main loop
 * iterates over ALL edges in each iteration. An edge list is the most
 * natural and cache-friendly format for this access pattern.
 *
 * Graph file format (plain text):
 *   Line 1:    V E           (number of vertices and edges)
 *   Line 2+:   src dest w    (directed edge from src to dest with weight w)
 *
 * Example file for a tiny graph:
 *   4 5
 *   0 1 6
 *   0 2 7
 *   1 2 8
 *   1 3 -4
 *   2 3 9
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* INF represents "no path found yet" - we use a large int value. 
 * We don't use INT_MAX directly in arithmetic because adding to INT_MAX overflows.
 * So we use a value that's large but safe for addition with edge weights. */
#define INF 1000000000

/*
 * Edge structure
 * Represents a single directed edge: src --> dest with a given weight.
 * Weight can be negative (that's the whole reason we use Bellman-Ford 
 * instead of Dijkstra's algorithm).
 */
typedef struct {
    int src;        /* source vertex (0-indexed) */
    int dest;       /* destination vertex (0-indexed) */
    int weight;     /* edge weight (can be negative) */
} Edge;

/*
 * Graph structure (Edge List representation)
 *
 * Why edge list and not adjacency matrix or adjacency list?
 * - Bellman-Ford iterates over ALL edges in each iteration
 * - Edge list gives us a single flat array to loop through
 * - This is cache-friendly and easy to partition for parallel processing
 * - Adjacency matrix wastes memory for sparse graphs (O(V^2))
 * - Adjacency list requires pointer chasing (bad for cache)
 */
typedef struct {
    int V;          /* Total number of vertices (numbered 0 to V-1) */
    int E;          /* Total number of edges */
    Edge *edges;    /* Dynamic array of E edges */
} Graph;


/* ===================== Function Declarations ===================== */

/*
 * create_graph - Allocate memory for a new graph
 * @V: number of vertices
 * @E: number of edges
 * Returns: pointer to new Graph (edges array allocated but not filled)
 */
Graph* create_graph(int V, int E);

/*
 * free_graph - Free all memory associated with a graph
 * @graph: pointer to graph to free
 */
void free_graph(Graph *graph);

/*
 * load_graph - Read a graph from a text file
 * @filename: path to the graph file
 * Returns: pointer to loaded Graph, or NULL on error
 */
Graph* load_graph(const char *filename);

/*
 * save_graph - Write a graph to a text file
 * @graph: pointer to graph to save
 * @filename: output file path
 */
void save_graph(const Graph *graph, const char *filename);

/*
 * print_graph_info - Print summary info about the graph
 * @graph: pointer to graph
 * Prints vertex count, edge count, and first few edges (useful for debugging)
 */
void print_graph_info(const Graph *graph);

#endif /* GRAPH_H */
