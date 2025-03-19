import argparse
import subprocess
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from matplotlib.colors import LinearSegmentedColormap

# PFN 범위를 나누는 함수
def split_pfn_range(start_pfn, end_pfn, num_threads):
    step = (end_pfn - start_pfn) // num_threads
    ranges = []
    for i in range(num_threads):
        start = start_pfn + i * step
        end = start_pfn + (i + 1) * step if i < num_threads - 1 else end_pfn
        ranges.append((start, end))
    return ranges

# 범위 내 데이터를 수집하는 함수
def collect_data_for_range(start_pfn, end_pfn, snapshot, numa_file='/proc/numa_folio_stats'):
    collected_data = []
    try:
        with open(numa_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            m = re.search(r'folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)', line)
            if m:
                pfn = int(m.group(2))
                if start_pfn <= pfn <= end_pfn:
                    collected_data.append([int(m.group(1)), pfn, int(m.group(3)), int(m.group(4)), snapshot])
    except Exception as e:
        print(f"Error reading {numa_file}: {e}")
    return collected_data

# 노드별 병렬 데이터 수집 함수
def collect_data_parallel(snapshot, num_threads, numa_file='/proc/numa_folio_stats'):
    node_ranges = parse_node_pfn_stats('/proc/node_pfn_stats')
    collected_data = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for node, (start_pfn, end_pfn) in node_ranges.items():
            pfn_ranges = split_pfn_range(start_pfn, end_pfn, num_threads)
            for (start, end) in pfn_ranges:
                futures.append(executor.submit(collect_data_for_range, start, end, snapshot, numa_file))
        
        # 병렬 처리 결과 병합
        for future in futures:
            collected_data.extend(future.result())

    return collected_data

# PFN 범위를 노드별로 파싱
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

# 히트맵 생성 함수
def generate_heatmap_for_node(node, node_df, custom_cmap):
    if node_df.empty:
        print(f"No data for node {node}. Skipping heatmap generation.")
        return
    
    pivot_table = node_df.pivot_table(index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum')
    pivot_table = pivot_table.fillna(0)

    if pivot_table.empty:
        print(f"No data for node {node} after pivoting. Skipping heatmap.")
        return

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

# 병렬 히트맵 생성 함수
def generate_heatmaps_parallel(df, custom_cmap, num_threads):
    all_nodes = set(df['node'].unique())
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for node in all_nodes:
            node_df = df[df['node'] == node]
            futures.append(executor.submit(generate_heatmap_for_node, node, node_df, custom_cmap))
        
        # 결과 기다리기
        for future in futures:
            future.result()

# 메인 함수
def main():
    parser = argparse.ArgumentParser(
        description="Collects /proc/numa_folio_stats data and generates heatmaps."
    )
    # 먼저 스레드 수를 받음
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use for parallel data collection.")
    # 나머지 워크로드 명령 인자를 받음
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Workload command to execute.")
    args = parser.parse_args()

    custom_cmap = LinearSegmentedColormap.from_list("NavyToRed", ["navy", "red"], N=256)

    print(f"Number of threads: {args.threads}")
    print(f"Executing workload command: {' '.join(args.command)}")
    proc = subprocess.Popen(args.command)

    collected_data = []
    snapshot = 0
    numa_file = '/proc/numa_folio_stats'

    print("Starting data collection...")
    try:
        while proc.poll() is None:
            snapshot += 1
            # 멀티스레드로 데이터 수집
            data = collect_data_parallel(snapshot, args.threads, numa_file)
            collected_data.extend(data)

    except KeyboardInterrupt:
        print("Data collection interrupted. Terminating workload...")
        proc.terminate()
        proc.wait()

    print("Workload terminated. Generating heatmaps...")

    if not collected_data:
        print("No data collected. Exiting.")
        return

    # Pandas 데이터프레임 처리
    df = pd.DataFrame(collected_data, columns=['node', 'pfn', 'source_nid', 'migrate_count', 'snapshot'])
    print("Collected data sample:\n", df.head())

    # 노드별 히트맵 병렬 생성
    generate_heatmaps_parallel(df, custom_cmap, num_threads=args.threads)

if __name__ == "__main__":
    main()
