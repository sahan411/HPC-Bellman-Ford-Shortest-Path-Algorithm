/*
 * utils.h - Utility Functions for Correctness Verification
 * ========================================================
 *
 * After running any parallel version, we need to verify that
 * the computed distances match the serial version exactly.
 * This header provides functions for:
 *   - Saving distance arrays to files
 *   - Loading distance arrays from files
 *   - Comparing two distance arrays (parallel vs serial)
 *   - Pretty-printing distances
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#ifndef UTILS_H
#define UTILS_H

#include "graph.h"

/*
 * save_distances - Write distance array to a text file
 * @filename: output file path
 * @dist: array of distances (size V)
 * @V: number of vertices
 *
 * File format:
 *   V
 *   dist[0]
 *   dist[1]
 *   ...
 *   dist[V-1]
 *
 * INF values are written as "INF" for readability.
 */
void save_distances(const char *filename, int *dist, int V);

/*
 * load_distances - Read distance array from a text file
 * @filename: input file path
 * @V: pointer to int where vertex count will be stored
 * Returns: dynamically allocated distance array, or NULL on error
 * Caller must free() the returned array.
 */
int* load_distances(const char *filename, int *V);

/*
 * verify_distances - Compare two distance arrays
 * @dist1: first distance array (usually from serial version)
 * @dist2: second distance array (usually from parallel version)
 * @V: number of vertices
 * Returns: 1 if all distances match, 0 if any mismatch
 *
 * This is the KEY function for validating parallel correctness.
 * If this returns 0, it prints which vertices have mismatched distances.
 */
int verify_distances(int *dist1, int *dist2, int V);

/*
 * print_distances - Print distances to stdout
 * @dist: distance array
 * @V: number of vertices
 * @max_print: maximum number of vertices to print (-1 for all)
 *
 * For large graphs, printing all distances floods the terminal.
 * Use max_print to limit output (e.g., first 20 vertices).
 */
void print_distances(int *dist, int V, int max_print);

#endif /* UTILS_H */
