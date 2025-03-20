#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import re
import os
import ctypes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def execute_migrate_table_reset():
    SYS_MIGRATE_TABLE_RESET = 462  # ← 시스템 콜 번호를 확인해서 바꿔야 함
    libc = ctypes.CDLL("libc.so.6")
    
    print("Executing migrate_table_reset system call...")
    ret = libc.syscall(SYS_MIGRATE_TABLE_RESET)
    
    if ret == 0:
        print("migrate_table_reset executed successfully.")
    else:
        print(f"Error: migrate_table_reset system call failed with return code {ret}.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Collects /proc/numa_folio_stats data and generates heatmaps."
    )
    parser.add_argument("command", nargs="+", help="Workload command to execute.")
    parser.add_argument("--interval", type=float, default=0.5, help="Data collection interval (seconds).")
    args = parser.parse_args()

    # 워크로드 실행 전에 migrate_table_reset 실행
    execute_migrate_table_reset()

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

        # 같은 노드에서 마이그레이션된 데이터
        same_source_df = node_df[node_df['source_nid'] == node]
        # 다른 노드에서 마이그레이션된 데이터
        diff_source_df = node_df[node_df['source_nid'] != node]

        if same_source_df.empty and diff_source_df.empty:
            print(f"No relevant data for node {node}. Skipping.")
            continue

        plt.figure(figsize=(12, 8))

        # 같은 노드에서 마이그레이션된 데이터 → 푸른색
        if not same_source_df.empty:
            pivot_same = same_source_df.pivot_table(
                index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum'
            ).fillna(0)

            sns.heatmap(pivot_same, cmap="Blues", cbar=True, alpha=0.7)

        # 다른 노드에서 마이그레이션된 데이터 → 붉은색
        if not diff_source_df.empty:
            pivot_diff = diff_source_df.pivot_table(
                index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum'
            ).fillna(0)

            sns.heatmap(pivot_diff, cmap="Reds", cbar=True, alpha=0.7)

        plt.title(f"Node {node} - Migration Heatmap")
        plt.xlabel("Snapshot (Time)")
        plt.ylabel("PFN")
        plt.tight_layout()

        filename = f"node_{node}_migration_heatmap.png"
        plt.savefig(filename)
        plt.close()
        print(f"Heatmap for node {node} saved as '{filename}'.")

    # 한 번만 삭제
    if os.path.exists(numa_file):
        try:
            os.remove(numa_file)
            print(f"File '{numa_file}' has been deleted.")
        except Exception as e:
            print(f"Failed to delete '{numa_file}': {e}")

if __name__ == "__main__":
    main()
