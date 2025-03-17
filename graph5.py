import argparse
import subprocess
import sys
import time
import re
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# PFN 범위 나누기 (예: start_pfn부터 end_pfn까지 n개 구간으로 나누기)
def split_pfn_range(start_pfn, end_pfn, num_threads):
    step = (end_pfn - start_pfn) // num_threads
    ranges = []
    for i in range(num_threads):
        start = start_pfn + i * step
        end = start_pfn + (i + 1) * step if i < num_threads - 1 else end_pfn
        ranges.append((start, end))
    return ranges

# 데이터를 수집하는 함수
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

# 데이터 수집을 병렬로 처리하는 함수
def collect_data_parallel(snapshot, num_threads, numa_file='/proc/numa_folio_stats'):
    node_ranges = parse_node_pfn_stats(numa_file)
    collected_data = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for node, (start_pfn, end_pfn) in node_ranges.items():
            pfn_ranges = split_pfn_range(start_pfn, end_pfn, num_threads)
            for (start, end) in pfn_ranges:
                futures.append(executor.submit(collect_data_for_range, start, end, snapshot, numa_file))
        
        for future in futures:
            collected_data.extend(future.result())
    
    return collected_data

# /proc/node_pfn_stats 파일을 파싱하여 노드별 pfn 범위를 얻는 함수
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
    parser = argparse.ArgumentParser(description="Collects /proc/numa_folio_stats data with parallel processing.")
    parser.add_argument("command", nargs="+", help="Workload command to execute.")
    parser.add_argument("--interval", type=float, default=0.5, help="Data collection interval (seconds).")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use for parallel data collection.")
    args = parser.parse_args()

    print("Executing workload:", args.command)
    proc = subprocess.Popen(args.command)

    collected_data = []
    snapshot = 0
    numa_file = '/proc/numa_folio_stats'

    print("Starting data collection...")
    try:
        while proc.poll() is None:
            snapshot += 1
            data = collect_data_parallel(snapshot, args.threads, numa_file)
            collected_data.extend(data)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Data collection interrupted. Terminating workload...")
        proc.terminate()
        proc.wait()

    print("Workload terminated. Generating heatmaps...")

    if not collected_data:
        print("No data collected. Exiting.")
        return

    # pandas로 수집된 데이터 처리
    df = pd.DataFrame(collected_data, columns=['node', 'pfn', 'source_nid', 'migrate_count', 'snapshot'])
    print("Collected data sample:\n", df.head())

    # 이후 heatmap 생성 로직 등을 처리

if __name__ == "__main__":
    main()
