/*
 * bellman_ford_mpi_cuda.cu - MPI + CUDA Bellman-Ford
 * ==================================================
 *
 * MPI partitions the edge list across processes. Each process relaxes its
 * local edge partition on a CUDA device, then MPI_Allreduce combines the full
 * distance array across ranks.
 *
 * On a single-GPU laptop this is expected to be limited by multiple MPI ranks
 * sharing one GPU plus host/device copies around each MPI_Allreduce. It is most
 * useful as a multi-GPU or multi-node design.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <cuda_runtime.h>

extern "C" {
#include "../common/graph.h"
#include "../common/utils.h"
}

#define THREADS_PER_BLOCK 256

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)
#define MPI_INT        ((MPI_Datatype)0x4c000405)
#define MPI_MIN        ((MPI_Op)0x58000002)
#define MPI_LOR        ((MPI_Op)0x58000007)

typedef int (__stdcall *MPI_Init_fn)(int *, char ***);
typedef int (__stdcall *MPI_Finalize_fn)(void);
typedef int (__stdcall *MPI_Comm_rank_fn)(MPI_Comm, int *);
typedef int (__stdcall *MPI_Comm_size_fn)(MPI_Comm, int *);
typedef int (__stdcall *MPI_Barrier_fn)(MPI_Comm);
typedef int (__stdcall *MPI_Allreduce_fn)(const void *, void *, int,
                                          MPI_Datatype, MPI_Op, MPI_Comm);
typedef int (__stdcall *MPI_Abort_fn)(MPI_Comm, int);
typedef double (__stdcall *MPI_Wtime_fn)(void);

static HMODULE mpi_dll = NULL;
static MPI_Init_fn pMPI_Init = NULL;
static MPI_Finalize_fn pMPI_Finalize = NULL;
static MPI_Comm_rank_fn pMPI_Comm_rank = NULL;
static MPI_Comm_size_fn pMPI_Comm_size = NULL;
static MPI_Barrier_fn pMPI_Barrier = NULL;
static MPI_Allreduce_fn pMPI_Allreduce = NULL;
static MPI_Abort_fn pMPI_Abort = NULL;
static MPI_Wtime_fn pMPI_Wtime = NULL;

static FARPROC load_mpi_symbol(const char *name) {
    FARPROC symbol = GetProcAddress(mpi_dll, name);
    if (symbol == NULL) {
        fprintf(stderr, "Error: could not load %s from msmpi.dll.\n", name);
        ExitProcess(1);
    }
    return symbol;
}

static void load_mpi_runtime(void) {
    if (mpi_dll != NULL) return;

    mpi_dll = LoadLibraryA("msmpi.dll");
    if (mpi_dll == NULL) {
        fprintf(stderr, "Error: could not load msmpi.dll.\n");
        ExitProcess(1);
    }

    pMPI_Init = (MPI_Init_fn)load_mpi_symbol("MPI_Init");
    pMPI_Finalize = (MPI_Finalize_fn)load_mpi_symbol("MPI_Finalize");
    pMPI_Comm_rank = (MPI_Comm_rank_fn)load_mpi_symbol("MPI_Comm_rank");
    pMPI_Comm_size = (MPI_Comm_size_fn)load_mpi_symbol("MPI_Comm_size");
    pMPI_Barrier = (MPI_Barrier_fn)load_mpi_symbol("MPI_Barrier");
    pMPI_Allreduce = (MPI_Allreduce_fn)load_mpi_symbol("MPI_Allreduce");
    pMPI_Abort = (MPI_Abort_fn)load_mpi_symbol("MPI_Abort");
    pMPI_Wtime = (MPI_Wtime_fn)load_mpi_symbol("MPI_Wtime");
}

static int MPI_Init(int *argc, char ***argv) {
    load_mpi_runtime();
    return pMPI_Init(argc, argv);
}

static int MPI_Finalize(void) {
    return pMPI_Finalize();
}

static int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    return pMPI_Comm_rank(comm, rank);
}

static int MPI_Comm_size(MPI_Comm comm, int *size) {
    return pMPI_Comm_size(comm, size);
}

static int MPI_Barrier(MPI_Comm comm) {
    return pMPI_Barrier(comm);
}

static int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    return pMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

static int MPI_Abort(MPI_Comm comm, int errorcode) {
    load_mpi_runtime();
    return pMPI_Abort(comm, errorcode);
}

static double MPI_Wtime(void) {
    return pMPI_Wtime();
}

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err__));        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

__global__ void relax_local_edges_kernel(const int *src, const int *dest,
                                         const int *weight, int *dist,
                                         int local_edges, int *updated) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= local_edges) return;

    int u = src[j];
    int v = dest[j];
    int w = weight[j];

    if (dist[u] != INF) {
        int new_dist = dist[u] + w;
        if (new_dist < dist[v]) {
            int old = atomicMin(&dist[v], new_dist);
            if (old > new_dist) {
                atomicOr(updated, 1);
            }
        }
    }
}

__global__ void check_local_negative_cycle_kernel(const int *src,
                                                  const int *dest,
                                                  const int *weight,
                                                  const int *dist,
                                                  int local_edges,
                                                  int *has_cycle) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= local_edges) return;

    int u = src[j];
    int v = dest[j];
    int w = weight[j];

    if (dist[u] != INF && dist[u] + w < dist[v]) {
        atomicOr(has_cycle, 1);
    }
}

static void edge_partition(int total_edges, int rank, int num_procs,
                           int *start, int *end) {
    int chunk = total_edges / num_procs;
    int remainder = total_edges % num_procs;

    if (rank < remainder) {
        *start = rank * (chunk + 1);
        *end = *start + chunk + 1;
    } else {
        *start = remainder * (chunk + 1) + (rank - remainder) * chunk;
        *end = *start + chunk;
    }
}

static int bellman_ford_mpi_cuda(Graph *graph, int source, int *dist,
                                 int rank, int num_procs) {
    int V = graph->V;
    int E = graph->E;
    int start, end;
    edge_partition(E, rank, num_procs, &start, &end);
    int local_edges = end - start;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    if (rank == 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        printf("Running Bellman-Ford MPI+CUDA with %d MPI processes...\n",
               num_procs);
        printf("  CUDA devices visible: %d\n", device_count);
        printf("  Rank 0 CUDA device: %s\n", prop.name);
        printf("  %d vertices, %d edges, up to %d iterations\n", V, E, V - 1);
    }

    int *h_src = (int *)malloc(local_edges * sizeof(int));
    int *h_dest = (int *)malloc(local_edges * sizeof(int));
    int *h_weight = (int *)malloc(local_edges * sizeof(int));
    int *reduced_dist = (int *)malloc(V * sizeof(int));
    if (!h_src || !h_dest || !h_weight || !reduced_dist) {
        fprintf(stderr, "Rank %d: host allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < local_edges; i++) {
        Edge edge = graph->edges[start + i];
        h_src[i] = edge.src;
        h_dest[i] = edge.dest;
        h_weight[i] = edge.weight;
    }

    for (int i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    int *d_src = NULL;
    int *d_dest = NULL;
    int *d_weight = NULL;
    int *d_dist = NULL;
    int *d_updated = NULL;
    int *d_has_cycle = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_src, local_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_dest, local_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_weight, local_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_dist, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_updated, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_has_cycle, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_src, h_src, local_edges * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dest, h_dest, local_edges * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, local_edges * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dist, dist, V * sizeof(int),
                          cudaMemcpyHostToDevice));

    int blocks = (local_edges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int early_stop_iter = V - 1;

    for (int iter = 0; iter < V - 1; iter++) {
        CUDA_CHECK(cudaMemset(d_updated, 0, sizeof(int)));

        relax_local_edges_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_src, d_dest, d_weight, d_dist, local_edges, d_updated);
        CUDA_CHECK(cudaDeviceSynchronize());

        int local_updated = 0;
        CUDA_CHECK(cudaMemcpy(&local_updated, d_updated, sizeof(int),
                              cudaMemcpyDeviceToHost));

        int global_updated = 0;
        MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_LOR,
                      MPI_COMM_WORLD);
        if (!global_updated) {
            early_stop_iter = iter + 1;
            if (rank == 0) {
                printf("  Early termination at iteration %d (no changes)\n",
                       iter + 1);
            }
            break;
        }

        CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(int),
                              cudaMemcpyDeviceToHost));
        MPI_Allreduce(dist, reduced_dist, V, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        memcpy(dist, reduced_dist, V * sizeof(int));
        CUDA_CHECK(cudaMemcpy(d_dist, dist, V * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    if (rank == 0 && early_stop_iter == V - 1) {
        printf("  Completed all %d iterations\n", V - 1);
    }

    CUDA_CHECK(cudaMemset(d_has_cycle, 0, sizeof(int)));
    check_local_negative_cycle_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_src, d_dest, d_weight, d_dist, local_edges, d_has_cycle);
    CUDA_CHECK(cudaDeviceSynchronize());

    int local_cycle = 0;
    CUDA_CHECK(cudaMemcpy(&local_cycle, d_has_cycle, sizeof(int),
                          cudaMemcpyDeviceToHost));
    int global_cycle = 0;
    MPI_Allreduce(&local_cycle, &global_cycle, 1, MPI_INT, MPI_LOR,
                  MPI_COMM_WORLD);

    cudaFree(d_src);
    cudaFree(d_dest);
    cudaFree(d_weight);
    cudaFree(d_dist);
    cudaFree(d_updated);
    cudaFree(d_has_cycle);
    free(h_src);
    free(h_dest);
    free(h_weight);
    free(reduced_dist);

    if (global_cycle) {
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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpiexec -n <procs> %s <graph_file> [source]\n",
                   argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *graph_file = argv[1];
    int source = 0;
    if (argc >= 3) {
        source = atoi(argv[2]);
    }

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

    int *dist = (int *)malloc(graph->V * sizeof(int));
    if (dist == NULL) {
        fprintf(stderr, "Rank %d: distance allocation failed.\n", rank);
        free_graph(graph);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    int result = bellman_ford_mpi_cuda(graph, source, dist, rank, num_procs);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (result == -1) {
        free(dist);
        free_graph(graph);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("\n");
        printf("============================================\n");
        printf("  MPI+CUDA Bellman-Ford Results\n");
        printf("============================================\n");
        printf("  Source vertex   : %d\n", source);
        printf("  MPI processes   : %d\n", num_procs);
        printf("  Execution time  : %.6f seconds\n", elapsed);
        printf("  (Includes GPU copies and MPI synchronization)\n");
        printf("============================================\n");

        print_distances(dist, graph->V, 20);
        remove("results/mpi_cuda_distances.txt");
        save_distances("results/mpi_cuda_distances.txt", dist, graph->V);

        int serial_V = 0;
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
        }
    }

    free(dist);
    free_graph(graph);
    MPI_Finalize();
    return 0;
}
