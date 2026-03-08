#!/usr/bin/env python3
"""
run_benchmarks.py - Benchmark all Bellman-Ford implementations
==============================================================

Runs all versions on all graph sizes and records:
  - Execution time for each version/size combination
  - Speedup relative to serial baseline
  - Results saved to results/benchmark_results.csv

Usage:
    python scripts/run_benchmarks.py

Requirements:
    - All executables built in bin/
    - Graph files in graphs/
    - results/ directory exists

Authors: Team HPC
Date: March 2026
"""

import subprocess
import re
import os
import csv
import sys
import time

# ============================================================
# Configuration
# ============================================================

MPIEXEC = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"

# Graph sizes to test
GRAPHS = [
    ("tiny",   "graphs/tiny.txt"),
    ("small",  "graphs/small.txt"),
    ("medium", "graphs/medium.txt"),
    ("large",  "graphs/large.txt"),
]

# Number of repetitions per test (take the best time to reduce noise)
REPS = 3

# Source vertex for all tests
SOURCE = 0

# OpenMP thread counts to test
OMP_THREADS = [1, 2, 4, 8]

# MPI process counts to test
MPI_PROCS = [1, 2, 4]

# Hybrid: (mpi_procs, omp_threads) combinations
HYBRID_CONFIGS = [(1, 8), (2, 4), (4, 2)]


def run_cmd(cmd, env=None, timeout=300):
    """
    Run a command and return (success, elapsed_time, output).
    Extracts execution time from the program's output.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        output = result.stdout + result.stderr

        # Look for "Execution time : X.XXXXXX seconds" in output
        match = re.search(r"Execution time\s*:\s*([\d.]+)\s*seconds", output)
        if match:
            return True, float(match.group(1)), output
        else:
            return False, 0.0, output

    except subprocess.TimeoutExpired:
        return False, 0.0, "TIMEOUT"
    except Exception as e:
        return False, 0.0, str(e)


def run_serial(graph_file, source, serial_dist_file):
    """Run serial version and save distances for verification."""
    # Remove old serial distances first
    if os.path.exists(serial_dist_file):
        os.remove(serial_dist_file)

    cmd = [r".\bin\bellman_ford_serial.exe", graph_file, str(source)]
    success, elapsed, output = run_cmd(cmd)
    return success, elapsed, output


def run_openmp(graph_file, source, threads):
    """Run OpenMP version with specified thread count."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = [r".\bin\bellman_ford_openmp.exe", graph_file, str(source),
           str(threads)]
    return run_cmd(cmd, env=env)


def run_mpi(graph_file, source, procs):
    """Run MPI version with specified process count."""
    cmd = [MPIEXEC, "-n", str(procs),
           r".\bin\bellman_ford_mpi.exe", graph_file, str(source)]
    return run_cmd(cmd)


def run_hybrid(graph_file, source, procs, threads):
    """Run Hybrid version with specified MPI procs and OMP threads."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = [MPIEXEC, "-n", str(procs),
           r".\bin\bellman_ford_hybrid.exe", graph_file, str(source),
           str(threads)]
    return run_cmd(cmd, env=env)


def best_of(func, *args, reps=REPS):
    """
    Run func(*args) 'reps' times, return the best (minimum) elapsed time.
    Using minimum reduces noise from OS scheduling jitter.
    """
    times = []
    for _ in range(reps):
        success, elapsed, output = func(*args)
        if success and elapsed > 0:
            times.append(elapsed)
        elif not success:
            print(f"    WARNING: Run failed. Output: {output[:200]}")
    if times:
        return min(times)
    return None


def speedup(serial_time, parallel_time):
    """Compute speedup. Returns None if either time is invalid."""
    if serial_time and parallel_time and parallel_time > 0:
        return serial_time / parallel_time
    return None


def main():
    print("=" * 60)
    print("  HPC Bellman-Ford Benchmark Suite")
    print("=" * 60)
    print()

    # Make sure results directory exists
    os.makedirs("results", exist_ok=True)

    # CSV output
    csv_file = "results/benchmark_results.csv"
    rows = []

    for graph_name, graph_file in GRAPHS:
        print(f"\n{'=' * 60}")
        print(f"  Graph: {graph_name} ({graph_file})")
        print(f"{'=' * 60}")

        serial_dist_file = "results/serial_distances.txt"

        # ----------------------------------------------------------
        # Step 1: Serial baseline
        # ----------------------------------------------------------
        print(f"\n[Serial]")
        serial_time = best_of(run_serial, graph_file, SOURCE,
                               serial_dist_file, reps=REPS)
        if serial_time:
            print(f"  Time: {serial_time:.6f}s")
            rows.append({
                "graph": graph_name,
                "version": "serial",
                "config": "1 thread",
                "time_sec": f"{serial_time:.6f}",
                "speedup": "1.00"
            })
        else:
            print(f"  FAILED - skipping this graph size")
            continue

        # ----------------------------------------------------------
        # Step 2: OpenMP
        # ----------------------------------------------------------
        print(f"\n[OpenMP]")
        for threads in OMP_THREADS:
            t = best_of(run_openmp, graph_file, SOURCE, threads, reps=REPS)
            sp = speedup(serial_time, t)
            status = f"{t:.6f}s  speedup={sp:.2f}x" if t else "FAILED"
            print(f"  {threads:2d} threads: {status}")
            rows.append({
                "graph": graph_name,
                "version": "openmp",
                "config": f"{threads} threads",
                "time_sec": f"{t:.6f}" if t else "N/A",
                "speedup": f"{sp:.2f}" if sp else "N/A"
            })

        # ----------------------------------------------------------
        # Step 3: MPI
        # ----------------------------------------------------------
        print(f"\n[MPI]")
        for procs in MPI_PROCS:
            t = best_of(run_mpi, graph_file, SOURCE, procs, reps=REPS)
            sp = speedup(serial_time, t)
            status = f"{t:.6f}s  speedup={sp:.2f}x" if t else "FAILED"
            print(f"  {procs:2d} processes: {status}")
            rows.append({
                "graph": graph_name,
                "version": "mpi",
                "config": f"{procs} procs",
                "time_sec": f"{t:.6f}" if t else "N/A",
                "speedup": f"{sp:.2f}" if sp else "N/A"
            })

        # ----------------------------------------------------------
        # Step 4: Hybrid (MPI + OpenMP)
        # ----------------------------------------------------------
        print(f"\n[Hybrid MPI+OpenMP]")
        for procs, threads in HYBRID_CONFIGS:
            t = best_of(run_hybrid, graph_file, SOURCE, procs, threads,
                        reps=REPS)
            sp = speedup(serial_time, t)
            label = f"{procs}x{threads}"
            status = f"{t:.6f}s  speedup={sp:.2f}x" if t else "FAILED"
            print(f"  {procs} procs x {threads} threads ({label}): {status}")
            rows.append({
                "graph": graph_name,
                "version": "hybrid",
                "config": f"{procs} procs x {threads} threads",
                "time_sec": f"{t:.6f}" if t else "N/A",
                "speedup": f"{sp:.2f}" if sp else "N/A"
            })

    # ----------------------------------------------------------
    # Save results to CSV
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Saving results to {csv_file}")
    print(f"{'=' * 60}\n")

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["graph", "version", "config", "time_sec", "speedup"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {csv_file}")
    print("Run scripts/plot_results.py to generate charts.")


if __name__ == "__main__":
    main()
