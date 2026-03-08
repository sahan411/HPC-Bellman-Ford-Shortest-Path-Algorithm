#!/usr/bin/env python3
"""
plot_results.py - Generate performance charts from benchmark results
====================================================================

Reads results/benchmark_results.csv and creates:
  1. Execution time comparison (bar chart per graph size)
  2. Speedup comparison (line chart per graph size)

Output images are saved to results/charts/

Usage:
    python scripts/plot_results.py

Requirements:
    pip install matplotlib
    Run scripts/run_benchmarks.py first

Authors: Team HPC
Date: March 2026
"""

import csv
import os
from collections import defaultdict

# Try to import matplotlib, give helpful error if not installed
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (saves to file)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed.")
    print("  Install it with: pip install matplotlib")
    print("  Falling back to text-only output.\n")


# ============================================================
# Colors for each version (consistent across charts)
# ============================================================
VERSION_COLORS = {
    "serial":  "#555555",
    "openmp":  "#2196F3",  # blue
    "mpi":     "#4CAF50",  # green
    "hybrid":  "#FF9800",  # orange
    "cuda":    "#9C27B0",  # purple
}

# Order for the legend
VERSION_ORDER = ["serial", "openmp", "mpi", "hybrid", "cuda"]

GRAPH_SIZES = ["tiny", "small", "medium", "large"]


def load_csv(csv_file):
    """Load benchmark CSV and return list of row dicts."""
    rows = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_by_graph(rows):
    """
    Organize rows by graph name.
    Returns: { graph_name: [row, row, ...] }
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["graph"]].append(row)
    return groups


def plot_execution_time(groups, output_dir):
    """
    Bar chart: Execution time (seconds) for each version/config,
    grouped by graph size.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Execution Time by Version and Graph Size",
                  fontsize=14, fontweight="bold")

    axes_flat = axes.flatten()

    for idx, graph_name in enumerate(GRAPH_SIZES):
        if graph_name not in groups:
            continue

        ax = axes_flat[idx]
        rows = groups[graph_name]

        labels = []
        times = []
        colors = []

        for row in rows:
            version = row["version"]
            if row["time_sec"] == "N/A":
                continue
            try:
                t = float(row["time_sec"])
            except ValueError:
                continue

            label = f"{version}\n{row['config']}"
            labels.append(label)
            times.append(t)
            colors.append(VERSION_COLORS.get(version, "#999999"))

        if not times:
            continue

        bars = ax.bar(range(len(labels)), times, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        # Label each bar with its time
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{t:.4f}s", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{graph_name.capitalize()} Graph", fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Time (seconds)")
        ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_patches = [
        mpatches.Patch(color=VERSION_COLORS[v], label=v.upper())
        for v in VERSION_ORDER if v in VERSION_COLORS
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(output_dir, "execution_time.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_speedup(groups, output_dir):
    """
    Line chart: Speedup (relative to serial) for each version/config,
    grouped by graph size.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Speedup vs Serial Baseline",
                  fontsize=14, fontweight="bold")

    axes_flat = axes.flatten()

    for idx, graph_name in enumerate(GRAPH_SIZES):
        if graph_name not in groups:
            continue

        ax = axes_flat[idx]
        rows = groups[graph_name]

        # Group by version
        version_data = defaultdict(list)
        for row in rows:
            version = row["version"]
            if row["speedup"] == "N/A":
                continue
            try:
                sp = float(row["speedup"])
                config = row["config"]
                version_data[version].append((config, sp))
            except ValueError:
                continue

        # Plot each version as a line or bar group
        labels = []
        speedups = []
        colors = []

        for version in VERSION_ORDER:
            if version not in version_data:
                continue
            for config, sp in version_data[version]:
                labels.append(f"{version}\n{config}")
                speedups.append(sp)
                colors.append(VERSION_COLORS.get(version, "#999999"))

        if not speedups:
            continue

        bars = ax.bar(range(len(labels)), speedups, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        # Draw speedup=1 reference line
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1,
                   alpha=0.7, label="Serial baseline")

        # Label each bar
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{sp:.2f}x", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{graph_name.capitalize()} Graph", fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Speedup (higher is better)")
        ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_patches = [
        mpatches.Patch(color=VERSION_COLORS[v], label=v.upper())
        for v in VERSION_ORDER if v in VERSION_COLORS
    ] + [plt.Line2D([0], [0], color="red", linestyle="--", label="Serial (1x)")]

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(output_dir, "speedup.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def print_text_summary(rows):
    """Print a formatted text summary of benchmark results."""
    print("\n" + "=" * 65)
    print(f"  {'BENCHMARK SUMMARY':^60}")
    print("=" * 65)
    print(f"  {'Graph':<8} {'Version':<8} {'Config':<24} {'Time(s)':<12} {'Speedup'}")
    print("-" * 65)

    current_graph = None
    for row in rows:
        if row["graph"] != current_graph:
            current_graph = row["graph"]
            print(f"\n  --- {current_graph.upper()} ---")

        time_str = row["time_sec"] if row["time_sec"] != "N/A" else "  N/A   "
        sp_str = (f"{float(row['speedup']):6.2f}x" if row["speedup"] != "N/A"
                  else "  N/A")
        print(f"  {row['graph']:<8} {row['version']:<8} "
              f"{row['config']:<24} {time_str:<12} {sp_str}")

    print("=" * 65)


def main():
    csv_file = "results/benchmark_results.csv"

    if not os.path.exists(csv_file):
        print(f"ERROR: {csv_file} not found.")
        print("Run: python scripts/run_benchmarks.py")
        return

    print("=" * 60)
    print("  HPC Bellman-Ford: Generating Performance Charts")
    print("=" * 60)
    print()

    rows = load_csv(csv_file)
    groups = group_by_graph(rows)

    # Print summary table
    print_text_summary(rows)

    # Create charts
    if HAS_MATPLOTLIB:
        output_dir = "results/charts"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating charts -> {output_dir}/")
        plot_execution_time(groups, output_dir)
        plot_speedup(groups, output_dir)
        print("\nDone!")
    else:
        print("\nInstall matplotlib to generate charts:")
        print("  pip install matplotlib")


if __name__ == "__main__":
    main()
