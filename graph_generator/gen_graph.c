/*
 * gen_graph.c - Random Graph Generator for Bellman-Ford Testing
 * =============================================================
 *
 * Generates random directed weighted graphs and saves them to files.
 * The generated graphs are designed for testing Bellman-Ford:
 *   - Directed edges with integer weights
 *   - Mostly positive weights, some negative (to justify using Bellman-Ford)
 *   - No negative-weight cycles (so shortest paths are well-defined)
 *   - Connected graph (every vertex reachable from vertex 0)
 *
 * How we ensure connectivity:
 *   1. First, create a random spanning tree (V-1 edges)
 *      This guarantees every vertex is reachable from vertex 0.
 *   2. Then, add (E - V + 1) more random edges.
 *      These create alternate paths and make the graph interesting.
 *
 * How we avoid negative cycles:
 *   - Spanning tree edges use positive weights only [1, 20]
 *   - Additional edges use weights in [-5, 50]
 *   - The positive spanning tree backbone makes negative cycles very unlikely
 *     in random graphs of reasonable density
 *   - Bellman-Ford's Step 3 will detect if one occurs (safety net)
 *
 * Usage:
 *   ./gen_graph <vertices> <edges> <output_file> [random_seed]
 *
 * Examples:
 *   ./gen_graph 1000 10000 graphs/small.txt
 *   ./gen_graph 100000 1000000 graphs/large.txt 42
 *
 * Authors: Team HPC
 * Date: March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/*
 * random_int - Generate a random integer in range [min, max] (inclusive)
 *
 * We use this instead of raw rand() to get values in a specific range.
 * Note: This is fine for graph generation, not for cryptography!
 */
int random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

/*
 * shuffle_array - Fisher-Yates shuffle for creating random permutations
 *
 * We use this to create a random spanning tree:
 * shuffle vertices, then connect them in sequence.
 * This gives a random tree structure, not just a simple path.
 */
void shuffle_array(int *arr, int n) {
    int i;
    for (i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        /* Swap arr[i] and arr[j] */
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

int main(int argc, char *argv[]) {
    /* ---- Parse command line arguments ---- */
    if (argc < 4) {
        printf("Usage: %s <vertices> <edges> <output_file> [random_seed]\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s 1000 10000 graphs/small.txt\n", argv[0]);
        printf("  %s 100000 1000000 graphs/large.txt 42\n", argv[0]);
        return 1;
    }

    int V = atoi(argv[1]);      /* number of vertices */
    int E = atoi(argv[2]);      /* number of edges */
    char *filename = argv[3];   /* output file */

    /* Optional random seed (for reproducible graphs) */
    if (argc >= 5) {
        srand(atoi(argv[4]));
        printf("Using random seed: %s\n", argv[4]);
    } else {
        srand((unsigned int)time(NULL));
        printf("Using random seed based on current time\n");
    }

    /* ---- Validate inputs ---- */
    if (V < 2) {
        fprintf(stderr, "Error: Need at least 2 vertices.\n");
        return 1;
    }

    /* We need at least V-1 edges for the spanning tree (connectivity) */
    if (E < V - 1) {
        fprintf(stderr, "Error: Need at least %d edges for %d vertices (to be connected).\n",
                V - 1, V);
        return 1;
    }

    /* Maximum possible edges in a directed graph (no self-loops) = V*(V-1) */
    long long max_edges = (long long)V * (V - 1);
    if (E > max_edges) {
        fprintf(stderr, "Error: Too many edges. Max for %d vertices is %lld.\n", V, max_edges);
        return 1;
    }

    printf("Generating graph: %d vertices, %d edges\n", V, E);

    /* ---- Allocate edge arrays ---- */
    int *src  = (int *)malloc(E * sizeof(int));
    int *dest = (int *)malloc(E * sizeof(int));
    int *wt   = (int *)malloc(E * sizeof(int));

    if (src == NULL || dest == NULL || wt == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    int edge_count = 0;

    /* ================================================================
     * PHASE 1: Create a random spanning tree (V-1 edges)
     * ================================================================
     * This guarantees the graph is connected.
     *
     * Method: Create a random permutation of vertices [0, 1, ..., V-1].
     * Then connect perm[0]->perm[1], perm[1]->perm[2], etc.
     * This creates a random path through all vertices (a spanning tree).
     * All spanning tree edges get POSITIVE weights [1, 20].
     */
    printf("Phase 1: Creating spanning tree (%d edges)...\n", V - 1);

    int *perm = (int *)malloc(V * sizeof(int));
    if (perm == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for permutation.\n");
        return 1;
    }

    /* Initialize permutation: [0, 1, 2, ..., V-1] */
    int i;
    for (i = 0; i < V; i++) {
        perm[i] = i;
    }

    /* Shuffle to get random order, but keep vertex 0 at position 0
     * so the spanning tree is rooted at vertex 0 (our source vertex) */
    shuffle_array(perm + 1, V - 1);  /* shuffle everything except perm[0] */

    /* Connect consecutive vertices in the permutation */
    for (i = 0; i < V - 1; i++) {
        src[edge_count]  = perm[i];
        dest[edge_count] = perm[i + 1];
        wt[edge_count]   = random_int(1, 20);  /* positive weights only */
        edge_count++;
    }

    free(perm);

    /* ================================================================
     * PHASE 2: Add remaining random edges (E - V + 1 edges)
     * ================================================================
     * These create alternative paths and make the graph more interesting.
     * Some edges get negative weights to showcase Bellman-Ford's capability.
     *
     * About 15% of additional edges get negative weights [-5, -1].
     * The rest get positive weights [1, 50].
     */
    int remaining = E - edge_count;
    printf("Phase 2: Adding %d random edges...\n", remaining);

    int attempts = 0;
    int max_attempts = remaining * 10;  /* prevent infinite loop for dense graphs */

    while (edge_count < E && attempts < max_attempts) {
        int s = random_int(0, V - 1);
        int d = random_int(0, V - 1);
        attempts++;

        /* Skip self-loops */
        if (s == d) continue;

        /* For small graphs, check for duplicate edges.
         * For large graphs (>50k vertices), skip this check - 
         * duplicates are rare and checking is too slow. */
        if (V < 50000) {
            int duplicate = 0;
            int j;
            for (j = 0; j < edge_count; j++) {
                if (src[j] == s && dest[j] == d) {
                    duplicate = 1;
                    break;
                }
            }
            if (duplicate) continue;
        }

        /* Assign weight: ~15% chance of negative */
        src[edge_count]  = s;
        dest[edge_count] = d;
        if (random_int(1, 100) <= 15) {
            wt[edge_count] = random_int(-5, -1);    /* negative edge */
        } else {
            wt[edge_count] = random_int(1, 50);     /* positive edge */
        }
        edge_count++;
    }

    /* If we couldn't generate enough unique edges, adjust E */
    if (edge_count < E) {
        printf("Warning: Could only generate %d edges (requested %d).\n", edge_count, E);
        E = edge_count;
    }

    /* ================================================================
     * PHASE 3: Save to file
     * ================================================================ */
    printf("Phase 3: Saving graph to '%s'...\n", filename);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot create file '%s'.\n", filename);
        free(src); free(dest); free(wt);
        return 1;
    }

    /* Write header */
    fprintf(fp, "%d %d\n", V, E);

    /* Write all edges */
    for (i = 0; i < E; i++) {
        fprintf(fp, "%d %d %d\n", src[i], dest[i], wt[i]);
    }

    fclose(fp);

    /* ---- Print summary ---- */
    int neg_count = 0;
    for (i = 0; i < E; i++) {
        if (wt[i] < 0) neg_count++;
    }

    printf("\n============================================\n");
    printf("  Graph Generation Complete!\n");
    printf("============================================\n");
    printf("  Vertices       : %d\n", V);
    printf("  Edges          : %d\n", E);
    printf("  Negative edges : %d (%.1f%%)\n", neg_count, 100.0 * neg_count / E);
    printf("  Output file    : %s\n", filename);
    printf("============================================\n");

    /* Clean up */
    free(src);
    free(dest);
    free(wt);

    return 0;
}
