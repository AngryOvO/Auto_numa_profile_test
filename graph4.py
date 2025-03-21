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
    SYS_MIGRATE_TABLE_RESET = 462  # 시스템 콜 번호를 확인하여 설정
    libc = ctypes.CDLL("libc.so.6")
    
    print("Executing migrate_table_reset system call...")
    ret = libc.syscall(SYS_MIGRATE_TABLE_RESET)
    
    if ret < 0:
        print(f"Error: migrate_table_reset system call failed with return code {ret}.")
        sys.exit(1)
    print("migrate_table_reset executed successfully.")

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

    # 데이터프레임 생성 및 스냅샷 범위 확보
    df = pd.DataFrame(collected_data, columns=['node', 'pfn', 'source_nid', 'migrate_count', 'snapshot'])
    print("Collected data sample:\n", df.head())

    all_snapshots = range(df['snapshot'].min(), df['snapshot'].max() + 1)
    node_ranges = parse_node_pfn_stats('/proc/node_pfn_stats')
    print("Node PFN ranges:\n", node_ranges)

    all_nodes = set(df['node'].unique())

    for node in all_nodes:
        node_df = df[df['node'] == node]

        if node_df.empty:
            print(f"No data for node {node}. Skipping heatmap generation.")
            continue

        # 같은 노드에서 마이그레이션된 데이터 → 푸른색
        same_source_df = node_df[node_df['source_nid'] == node]
        # 다른 노드에서 마이그레이션된 데이터 → 붉은색
        diff_source_df = node_df[node_df['source_nid'] != node]

        plt.figure(figsize=(12, 8))

        # 같은 노드 데이터
        if not same_source_df.empty:
            pivot_same = same_source_df.pivot_table(
                index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum', fill_value=0
            ).fillna(0)
            pivot_same = pivot_same.reindex(columns=all_snapshots, fill_value=0)  # 모든 스냅샷 포함
            sns.heatmap(pivot_same, cmap=LinearSegmentedColormap.from_list("Thermal", ["navy", "blue", "lightblue"], N=256),
                        cbar=True)

        # 다른 노드 데이터
        if not diff_source_df.empty:
            pivot_diff = diff_source_df.pivot_table(
                index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum', fill_value=0
            ).fillna(0)
            pivot_diff = pivot_diff.reindex(columns=all_snapshots, fill_value=0)  # 모든 스냅샷 포함
            sns.heatmap(pivot_diff, cmap=LinearSegmentedColormap.from_list("Thermal", ["navy", "red", "yellow"], N=256),
                        cbar=True)

        plt.title(f"Node {node} - Migration Heatmap")
        plt.xlabel("Snapshot (Time)")
        plt.ylabel("PFN")
        plt.tight_layout()

        filename = f"node_{node}_migration_heatmap.png"
        plt.savefig(filename)
        plt.close()
        print(f"Heatmap for node {node} saved as '{filename}'.")

if __name__ == "__main__":
    main()
