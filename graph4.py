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
import numexpr as ne  # NumPy 연산 최적화를 위해 추가
from matplotlib.colors import LinearSegmentedColormap

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
            except ValueError:
                current_node = None
        elif line.startswith("start pfn"):
            m = re.search(r'start pfn\s+(\d+)[,]?\s*(?:end pfn\s+)?(\d+)', line)
            if m and current_node is not None:
                node_ranges[current_node] = (int(m.group(1)), int(m.group(2)))
                current_node = None
    return node_ranges

def main():
    parser = argparse.ArgumentParser(
        description="Collects /proc/numa_folio_stats data and generates heatmaps."
    )
    parser.add_argument("command", nargs="+", help="Workload command to execute.")
    parser.add_argument("--interval", type=float, default=0.5, help="Data collection interval (seconds).")
    args = parser.parse_args()

    custom_cmap = LinearSegmentedColormap.from_list("NavyToRed", ["navy", "red"], N=256)

    print("Executing workload:", args.command)
    proc = subprocess.Popen(args.command)

    collected_data = []
    snapshot = 0
    numa_file = '/proc/numa_folio_stats'

    print("Starting data collection...")
    try:
        while proc.poll() is None:
            snapshot += 1
            try:
                with open(numa_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Failed to read '{numa_file}': {e}")
                sys.exit(1)

            for line in lines:
                m = re.search(r'folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)', line)
                if m:
                    collected_data.append([int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), snapshot])
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Data collection interrupted. Terminating workload...")
        proc.terminate()
        proc.wait()

    print("Workload terminated. Generating heatmaps...")

    if not collected_data:
        print("No data collected. Exiting.")
        return

    df = pd.DataFrame(collected_data, columns=['node', 'pfn', 'source_nid', 'migrate_count', 'snapshot'])
    print("Collected data sample:\n", df.head())

    node_ranges = parse_node_pfn_stats('/proc/node_pfn_stats')
    print("Node PFN ranges:\n", node_ranges)

    all_nodes = set(df['node'].unique())

    for node in all_nodes:
        node_df = df[df['node'] == node]

        if node_df.empty:
            print(f"No data for node {node}. Skipping heatmap generation.")
            continue

        pivot_table = node_df.pivot_table(index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum')
        pivot_table = pivot_table.fillna(0)

        if pivot_table.empty:
            print(f"No data for node {node} after pivoting. Skipping heatmap.")
            continue

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap=custom_cmap, cbar=True)
        plt.title(f"Node {node} - Migrate Count During Workload")
        plt.xlabel("Snapshot (Time)")
        plt.ylabel("PFN")
        plt.tight_layout()
        filename = f"node_{node}_heatmap.png"
        plt.savefig(filename)
        plt.close()
        print(f"Heatmap for node {node} saved as '{filename}'.")

    if os.path.exists(numa_file):
        try:
            os.remove(numa_file)
            print(f"File '{numa_file}' has been deleted.")
        except Exception as e:
            print(f"Failed to delete '{numa_file}': {e}")

if __name__ == "__main__":
    main()
