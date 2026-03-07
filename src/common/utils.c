/*
 * utils.c - Utility Functions for Result Verification and Display
 * ===============================================================
 *
 * These functions help us:
 * 1. Save results from any version to a file
 * 2. Load previously saved results
 * 3. Compare parallel results against serial (correctness check)
 * 4. Print distances in a readable format
 *
 * The verify_distances() function is CRITICAL - it's how we prove
 * that our parallel implementations are correct.
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#include "utils.h"
#include <string.h>

/*
 * save_distances - Save distance array to a file
 *
 * Format:
 *   Line 1: V (number of vertices)
 *   Line 2 to V+1: one distance per line
 *   INF values are written as "INF" for readability
 *
 * We always save results so we can compare versions later
 * without having to re-run everything.
 */
void save_distances(const char *filename, int *dist, int V) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot create output file '%s'.\n", filename);
        return;
    }

    /* Write vertex count first (so loader knows how many to read) */
    fprintf(fp, "%d\n", V);

    /* Write each distance */
    int i;
    for (i = 0; i < V; i++) {
        if (dist[i] >= INF) {
            fprintf(fp, "INF\n");     /* unreachable vertex */
        } else {
            fprintf(fp, "%d\n", dist[i]);
        }
    }

    fclose(fp);
    printf("Distances saved to '%s'\n", filename);
}

/*
 * load_distances - Load a previously saved distance array
 *
 * Reads the format written by save_distances().
 * Caller is responsible for free()-ing the returned array.
 *
 * Returns NULL on error.
 */
int* load_distances(const char *filename, int *V) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open distance file '%s'.\n", filename);
        return NULL;
    }

    /* Read vertex count */
    if (fscanf(fp, "%d", V) != 1) {
        fprintf(stderr, "Error: Cannot read vertex count from '%s'.\n", filename);
        fclose(fp);
        return NULL;
    }

    /* Allocate distance array */
    int *dist = (int *)malloc((*V) * sizeof(int));
    if (dist == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for %d distances.\n", *V);
        fclose(fp);
        return NULL;
    }

    /* Read each distance value */
    char buf[32];
    int i;
    for (i = 0; i < *V; i++) {
        if (fscanf(fp, "%s", buf) != 1) {
            fprintf(stderr, "Error: Failed to read distance for vertex %d.\n", i);
            free(dist);
            fclose(fp);
            return NULL;
        }

        /* Check if it's "INF" or a number */
        if (strcmp(buf, "INF") == 0) {
            dist[i] = INF;
        } else {
            dist[i] = atoi(buf);
        }
    }

    fclose(fp);
    return dist;
}

/*
 * verify_distances - Compare two distance arrays for correctness
 *
 * This is how we prove our parallel implementations are correct:
 *   1. Run the serial version --> save distances
 *   2. Run a parallel version --> save distances
 *   3. Call verify_distances() to compare
 *
 * If ALL distances match exactly, the parallel version is correct.
 * If any mismatch, prints which vertices differ (for debugging).
 *
 * Returns: 1 if all match (PASS), 0 if any mismatch (FAIL)
 */
int verify_distances(int *dist1, int *dist2, int V) {
    int match = 1;          /* assume everything matches */
    int mismatch_count = 0; /* count mismatches */
    int max_show = 10;      /* don't print more than 10 mismatches */

    int i;
    for (i = 0; i < V; i++) {
        if (dist1[i] != dist2[i]) {
            match = 0;
            mismatch_count++;

            /* Print first few mismatches to help with debugging */
            if (mismatch_count <= max_show) {
                printf("  MISMATCH at vertex %d: expected %d, got %d\n",
                       i,
                       dist1[i] >= INF ? -1 : dist1[i],   /* -1 means INF */
                       dist2[i] >= INF ? -1 : dist2[i]);
            }
        }
    }

    if (match) {
        printf("VERIFICATION PASSED: All %d distances match.\n", V);
    } else {
        printf("VERIFICATION FAILED: %d out of %d distances mismatch.\n", mismatch_count, V);
        if (mismatch_count > max_show) {
            printf("  (Showing first %d mismatches only)\n", max_show);
        }
    }

    return match;
}

/*
 * print_distances - Display distances in a nice table format
 *
 * @max_print: how many vertices to show (-1 = all)
 *
 * For big graphs (100k+ vertices), you definitely want to limit this.
 * Use -1 only for tiny test graphs.
 */
void print_distances(int *dist, int V, int max_print) {
    /* Figure out how many to print */
    int count;
    if (max_print < 0 || max_print > V) {
        count = V;
    } else {
        count = max_print;
    }

    printf("--------------------------------------------\n");
    printf("  Vertex | Shortest Distance from Source\n");
    printf("--------------------------------------------\n");

    int i;
    for (i = 0; i < count; i++) {
        if (dist[i] >= INF) {
            printf("  %6d | INF (unreachable)\n", i);
        } else {
            printf("  %6d | %d\n", i, dist[i]);
        }
    }

    if (count < V) {
        printf("  ... (%d more vertices not shown)\n", V - count);
    }
    printf("--------------------------------------------\n");
}
