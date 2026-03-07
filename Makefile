# ==============================================================================
# Makefile - Build System for Bellman-Ford HPC Project
# ==============================================================================
#
# This Makefile compiles all versions of the Bellman-Ford algorithm.
# Run on Linux or WSL (Windows Subsystem for Linux).
#
# Quick Start:
#   make serial        - Build serial version + graph generator
#   make openmp        - Build OpenMP (shared memory) version
#   make mpi           - Build MPI (distributed memory) version
#   make hybrid        - Build Hybrid (MPI + OpenMP) version
#   make cuda          - Build CUDA (GPU) version
#   make all           - Build everything
#   make clean         - Remove all compiled files
#
# Prerequisites:
#   - GCC compiler (gcc)
#   - OpenMP support (comes with gcc, use -fopenmp flag)
#   - MPI library (install: sudo apt install openmpi-bin libopenmpi-dev)
#   - NVIDIA CUDA toolkit (for CUDA version only)
#
# Authors: Team HPC
# Date: March 2026
# ==============================================================================

# ---- Compiler Settings ----
CC       = gcc
MPICC    = mpicc
NVCC     = nvcc
CFLAGS   = -O2 -Wall
OMP_FLAG = -fopenmp

# ---- Directories ----
SERIAL_DIR   = src/serial
OPENMP_DIR   = src/openmp
MPI_DIR      = src/mpi
HYBRID_DIR   = src/hybrid
CUDA_DIR     = src/cuda
COMMON_DIR   = src/common
GEN_DIR      = graph_generator
BIN_DIR      = bin

# ---- Common Source Files (used by all versions) ----
COMMON_SRC = $(COMMON_DIR)/graph.c $(COMMON_DIR)/utils.c

# ==============================================================================
# Build Targets
# ==============================================================================

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ---- Build everything ----
all: serial openmp mpi hybrid cuda
	@echo ""
	@echo "============================================"
	@echo "  All versions built successfully!"
	@echo "  Binaries are in the '$(BIN_DIR)/' folder."
	@echo "============================================"

# ---- Graph Generator ----
gen_graph: $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/gen_graph $(GEN_DIR)/gen_graph.c
	@echo "[OK] Graph generator built: $(BIN_DIR)/gen_graph"

# ---- Serial Version (baseline) ----
serial: gen_graph $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/bellman_ford_serial \
		$(SERIAL_DIR)/bellman_ford_serial.c $(COMMON_SRC) \
		-I$(COMMON_DIR)
	@echo "[OK] Serial version built: $(BIN_DIR)/bellman_ford_serial"

# ---- OpenMP Version (shared memory parallelism) ----
openmp: $(BIN_DIR)
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $(BIN_DIR)/bellman_ford_openmp \
		$(OPENMP_DIR)/bellman_ford_openmp.c $(COMMON_SRC) \
		-I$(COMMON_DIR)
	@echo "[OK] OpenMP version built: $(BIN_DIR)/bellman_ford_openmp"

# ---- MPI Version (distributed memory parallelism) ----
mpi: $(BIN_DIR)
	$(MPICC) $(CFLAGS) -o $(BIN_DIR)/bellman_ford_mpi \
		$(MPI_DIR)/bellman_ford_mpi.c $(COMMON_SRC) \
		-I$(COMMON_DIR)
	@echo "[OK] MPI version built: $(BIN_DIR)/bellman_ford_mpi"

# ---- Hybrid Version (MPI + OpenMP) ----
hybrid: $(BIN_DIR)
	$(MPICC) $(CFLAGS) $(OMP_FLAG) -o $(BIN_DIR)/bellman_ford_hybrid \
		$(HYBRID_DIR)/bellman_ford_hybrid.c $(COMMON_SRC) \
		-I$(COMMON_DIR)
	@echo "[OK] Hybrid version built: $(BIN_DIR)/bellman_ford_hybrid"

# ---- CUDA Version (GPU parallelism) ----
cuda: $(BIN_DIR)
	$(NVCC) -O2 -o $(BIN_DIR)/bellman_ford_cuda \
		$(CUDA_DIR)/bellman_ford_cuda.cu $(COMMON_SRC) \
		-I$(COMMON_DIR)
	@echo "[OK] CUDA version built: $(BIN_DIR)/bellman_ford_cuda"

# ---- Clean up compiled files ----
clean:
	rm -rf $(BIN_DIR)
	@echo "[OK] Cleaned all compiled files."

# ---- Help ----
help:
	@echo ""
	@echo "Bellman-Ford HPC Project - Build Targets:"
	@echo "  make serial   - Build serial version + graph generator"
	@echo "  make openmp   - Build OpenMP version"
	@echo "  make mpi      - Build MPI version"
	@echo "  make hybrid   - Build Hybrid (MPI+OpenMP) version"
	@echo "  make cuda     - Build CUDA (GPU) version"
	@echo "  make all      - Build all versions"
	@echo "  make clean    - Remove compiled files"
	@echo "  make help     - Show this help message"
	@echo ""

.PHONY: all serial openmp mpi hybrid cuda gen_graph clean help
