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
from matplotlib.colors import ListedColormap
from concurrent.futures import ThreadPoolExecutor

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

def read_numa_folio_stats(start_pfn, end_pfn, snapshot, numa_file):
    """특정 PFN 범위에 대해 /proc/numa_folio_stats 데이터를 읽고 처리"""
    local_data = []
    try:
        with open(numa_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to read '{numa_file}': {e}")
        return local_data

    for line in lines:
        m = re.search(r'folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)', line)
        if m:
            pfn = int(m.group(2))
            if start_pfn <= pfn < end_pfn:  # PFN 범위 필터링
                local_data.append([int(m.group(1)), pfn, int(m.group(3)), int(m.group(4)), snapshot])
    return local_data

def generate_heatmap_for_node(node, node_df, all_snapshots):
    """특정 NUMA 노드에 대한 히트맵 생성"""
    if node_df.empty:
        print(f"No data for node {node}. Skipping heatmap generation.")
        return

    plt.figure(figsize=(12, 8))

    # 고유한 source_nid 추출
    unique_sources = node_df['source_nid'].unique()
    colormap = generate_colormap(unique_sources)

    # 각 source_nid에 대해 heatmap 생성
    for i, source in enumerate(unique_sources):
        source_df = node_df[node_df['source_nid'] == source]

        if source_df.empty:
            continue

        pivot = source_df.pivot_table(
            index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum', fill_value=0
        ).fillna(0)
        pivot = pivot.reindex(columns=all_snapshots, fill_value=0)  # 모든 스냅샷 포함

        sns.heatmap(pivot, cmap=colormap, cbar=True, cbar_kws={'label': f"Source Node {source}"})

    plt.title(f"Node {node} - Migration Heatmap")
    plt.xlabel("Snapshot (Time)")
    plt.ylabel("PFN")
    plt.tight_layout()

    filename = f"node_{node}_migration_heatmap.png"
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap for node {node} saved as '{filename}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Collects /proc/numa_folio_stats data and generates heatmaps."
    )
    parser.add_argument("command", nargs="+", help="Workload command to execute.")
    parser.add_argument("--interval", type=float, default=0.5, help="Data collection interval (seconds).")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing.")
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
            node_ranges = parse_node_pfn_stats('/proc/node_pfn_stats')
            print("Node PFN ranges:\n", node_ranges)

            # 스레드별로 PFN 범위를 나눔
            tasks = []
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                for node, (start_pfn, end_pfn) in node_ranges.items():
                    step = (end_pfn - start_pfn) // args.threads
                    for i in range(args.threads):
                        sub_start_pfn = start_pfn + i * step
                        sub_end_pfn = start_pfn + (i + 1) * step if i < args.threads - 1 else end_pfn
                        tasks.append(executor.submit(read_numa_folio_stats, sub_start_pfn, sub_end_pfn, snapshot, numa_file))

                # 스레드 결과 수집
                for task in tasks:
                    collected_data.extend(task.result())

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
    all_nodes = set(df['node'].unique())

    # 컬러 맵 생성 함수
    def generate_colormap(unique_sources):
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sources)))  # 고유 source_nid 개수만큼 색상 생성
        return ListedColormap(colors)

    # 히트맵 생성 작업을 멀티스레드로 처리
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        tasks = []
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

        # 모든 작업 완료 대기
        for task in tasks:
            task.result()

if __name__ == "__main__":
    main()
