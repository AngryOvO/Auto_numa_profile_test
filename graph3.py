#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import re
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------
# (A) Function to parse the /proc/node_pfn_stats file
# Reads the entire PFN range for each node and returns a dictionary in the form:
# {node: (start_pfn, end_pfn)}
#
# Expected file format:
#   node 0
#   start pfn 1, end pfn 5300000
#
#   node 1
#   start pfn 5300001, end pfn 10000000
# ----------------------------------------------------------------------
def parse_node_pfn_stats(filepath='/proc/node_pfn_stats'):
    node_ranges = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)

    current_node = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("node"):
            parts = line.split()
            try:
                current_node = int(parts[1])
            except Exception as e:
                print(f"Parsing node failed for line: {line}")
                current_node = None
        elif line.startswith("start pfn"):
            # Example line: "start pfn 1, end pfn 5300000" or "start pfn 5300001, 10000000"
            m = re.search(r'start pfn\s+(\d+)[,]?\s*(?:end pfn\s+)?(\d+)', line)
            if m and current_node is not None:
                start_pfn = int(m.group(1))
                end_pfn = int(m.group(2))
                node_ranges[current_node] = (start_pfn, end_pfn)
                current_node = None  # Reset the node info
    return node_ranges

# ----------------------------------------------------------------------
# (B) Main function: Execute workload, collect data, generate binned heatmaps,
# and delete the /proc/numa_folio_stats file after process completion.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="This tool collects /proc/numa_folio_stats data during the workload execution, "
                    "then creates a heatmap using binning based on the PFN ranges defined in /proc/node_pfn_stats. "
                    "The color gradually changes from navy (low migration count) to red (high migration count). "
                    "After the process is complete, the /proc/numa_folio_stats file is deleted."
    )
    parser.add_argument(
        "command", nargs="+",
        help="The workload command and arguments to execute (e.g., /usr/bin/my_workload arg1 arg2)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Snapshot collection interval in seconds (default: 0.5 seconds)"
    )
    parser.add_argument(
        "--bin-size", type=int, default=10000,
        help="PFN bin size (e.g., 10000, default: 10000)"
    )
    args = parser.parse_args()

    # Custom colormap: linear interpolation from navy to red
    custom_cmap = LinearSegmentedColormap.from_list("NavyToRed", ["navy", "red"], N=256)

    # Execute the workload using subprocess
    print("Executing workload:", args.command)
    proc = subprocess.Popen(args.command)

    # Collect data from /proc/numa_folio_stats.
    # Each record: [node, pfn, source_nid, migrate_count, snapshot]
    collected_data = []
    snapshot = 0
    numa_file = '/proc/numa_folio_stats'

    print("Starting data collection during workload execution...")
    try:
        while proc.poll() is None:
            snapshot += 1
            try:
                with open(numa_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Failed to read '{numa_file}': {e}")
                sys.exit(1)
            # Parse all lines in the current snapshot
            for line in lines:
                m = re.search(
                    r'folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)',
                    line
                )
                if m:
                    node = int(m.group(1))
                    pfn = int(m.group(2))
                    source_nid = int(m.group(3))
                    migrate_count = int(m.group(4))
                    collected_data.append([node, pfn, source_nid, migrate_count, snapshot])
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Data collection interrupted by user. Terminating workload...")
        proc.terminate()
        proc.wait()

    print("Workload terminated. Generating heatmaps from collected data...")

    # Convert collected data to a DataFrame
    df = pd.DataFrame(
        collected_data,
        columns=['node', 'pfn', 'source_nid', 'migrate_count', 'snapshot']
    )
    print("Sample of collected data:")
    print(df.head())

    # Parse the /proc/node_pfn_stats file to get PFN ranges for each node
    node_ranges = parse_node_pfn_stats('/proc/node_pfn_stats')
    print("Node PFN ranges:")
    print(node_ranges)

    # Generate binned heatmaps for each node
    all_nodes = set(list(node_ranges.keys()))
    for node in all_nodes:
        start_pfn, end_pfn = node_ranges[node]
        bin_size = args.bin_size
        # Create bin boundaries from start_pfn with a step size of bin_size
        bins = np.arange(start_pfn, end_pfn + bin_size, bin_size)
        # Use the midpoint of each bin as a label
        labels = [(bins[i] + bins[i+1]) // 2 for i in range(len(bins) - 1)]
        overall_bin_labels = labels  # Labels for reindexing

        # Select data for the current node
        node_df = df[df['node'] == node].copy()
        if node_df.empty:
            print(f"No collected data for node {node} (generating an empty heatmap).")
        # Apply binning to PFN values using pd.cut with bin midpoints as labels
        node_df['pfn_bin'] = pd.cut(
            node_df['pfn'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Create a pivot table: index is pfn_bin, columns are snapshots, values are migrate_count
        pivot_table = node_df.pivot_table(
            index='pfn_bin',
            columns='snapshot',
            values='migrate_count',
            aggfunc='first'
        )
        # Reindex to include all bins and fill missing values with 0
        pivot_table = pivot_table.reindex(overall_bin_labels).fillna(0)
    
        plt.figure(figsize=(12, 8))
        # Draw the heatmap using the custom colormap (from navy to red)
        sns.heatmap(pivot_table, cmap=custom_cmap, cbar=True)
        plt.title(f"Node {node} - Migrate Count (Binned) During Workload")
        plt.xlabel("Snapshot (Time)")
        plt.ylabel(f"PFN Bin Center (from {start_pfn} to {end_pfn}, bin size {bin_size})")
        plt.tight_layout()
        filename = f"node_{node}_workload_binned_heatmap.png"
        plt.savefig(filename)
        plt.close()
        print(f"Heatmap for node {node} saved as '{filename}'.")

    # After workload termination, delete the /proc/numa_folio_stats file (reset)
    if os.path.exists(numa_file):
        try:
            os.remove(numa_file)
            print(f"File '{numa_file}' has been deleted.")
        except Exception as e:
            print(f"Failed to delete '{numa_file}': {e}")

if __name__ == "__main__":
    main()
