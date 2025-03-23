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

def main():
    parser = argparse.ArgumentParser(
        description="Collects /proc/numa_folio_stats data and generates heatmaps."
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Workload command to execute."
    )
    parser.add_argument(
        "--interval", type=float, default=0.5, help="Data collection interval (seconds)."
    )
    args = parser.parse_args()

    if not args.command:
        print("Error: No workload command provided.")
        sys.exit(1)

    # 워크로드 실행 전에 migrate_table_reset 실행
    execute_migrate_table_reset()

    print("Executing workload:", " ".join(args.command))
    try:
        # 워크로드 실행
        proc = subprocess.Popen(args.command)
    except FileNotFoundError:
        print(f"Error: Command '{args.command[0]}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing command '{args.command}': {e}")
        sys.exit(1)

    collected_data = []
    snapshot = 0
    numa_file = "/proc/numa_folio_stats"

    print("Starting data collection...")
    try:
        while proc.poll() is None:  # 워크로드가 실행 중일 때
            snapshot += 1
            try:
                with open(numa_file, "r") as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Failed to read '{numa_file}': {e}")
                sys.exit(1)

            for line in lines:
                m = re.search(
                    r"folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)",
                    line,
                )
                if m:
                    collected_data.append(
                        [
                            int(m.group(1)),
                            int(m.group(2)),
                            int(m.group(3)),
                            int(m.group(4)),
                            snapshot,
                        ]
                    )
            time.sleep(args.interval)

        # 워크로드 종료 후 추가 데이터 수집
        print("Workload terminated. Collecting additional data...")
        for _ in range(5):  # 추가로 5번 데이터를 수집
            snapshot += 1
            try:
                with open(numa_file, "r") as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Failed to read '{numa_file}': {e}")
                break

            for line in lines:
                m = re.search(
                    r"folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)",
                    line,
                )
                if m:
                    collected_data.append(
                        [
                            int(m.group(1)),
                            int(m.group(2)),
                            int(m.group(3)),
                            int(m.group(4)),
                            snapshot,
                        ]
                    )
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
    df = pd.DataFrame(
        collected_data,
        columns=["node", "pfn", "source_nid", "migrate_count", "snapshot"],
    )
    print("Collected data sample:\n", df.head())

    node_ranges = parse_node_pfn_stats("/proc/node_pfn_stats")
    print("Node PFN ranges:\n", node_ranges)

    # 스냅샷 범위 설정
    if df.empty:
        print("No data collected. Using default snapshot range.")
        all_snapshots = range(0, 2)  # 기본 스냅샷 범위
    else:
        all_snapshots = range(df["snapshot"].min(), df["snapshot"].max() + 1)

    # 노드별 히트맵 생성
    for node in node_ranges.keys():  # 모든 노드에 대해 처리
        node_df = df[df["node"] == node]
        print(f"Node {node}: Filtered node_df:\n{node_df.head()}")

        # 모든 데이터를 포함
        same_source_df = node_df
        print(f"Node {node}: Filtered same_source_df:\n{same_source_df.head()}")

        if not same_source_df.empty:
            pivot_same = same_source_df.pivot_table(
                index="pfn",
                columns="snapshot",
                values="migrate_count",
                aggfunc="sum",
                fill_value=0,
            ).fillna(0)

            # 노드의 전체 PFN 범위를 포함하도록 강제 설정
            start_pfn, end_pfn = node_ranges[node]
            full_pfn_range = range(start_pfn, end_pfn + 1)
            pivot_same = pivot_same.reindex(index=full_pfn_range, fill_value=0)
        else:
            print(f"No migration data for node {node}. Generating empty heatmap.")
            start_pfn, end_pfn = node_ranges[node]
            full_pfn_range = range(start_pfn, end_pfn + 1)
            pivot_same = pd.DataFrame(0, index=full_pfn_range, columns=all_snapshots)

        # 컬러맵 범위 설정
        vmax = pivot_same.values.max() if not pivot_same.empty else 1
        sns.heatmap(
            pivot_same,
            cmap=LinearSegmentedColormap.from_list(
                "Thermal", ["navy", "red", "yellow"], N=256  # 열화상 색상
            ),
            cbar=True,
            vmin=0,  # 최소값을 0으로 고정
            vmax=vmax,  # 최대값을 데이터의 최대값으로 설정
            annot=True,  # 셀에 값 표시
            fmt="d",     # 정수 형식으로 표시
        )node} - Migration Heatmap")
        plt.xlabel("Snapshot (Time)")
        plt.title(f"Node {node} - Migration Heatmap")
        plt.xlabel("Snapshot (Time)")
        plt.ylabel("PFN")
        plt.tight_layout()
        plt.savefig(filename)
        filename = f"node_{node}_migration_heatmap.png"        plt.close()
        plt.savefig(filename)r node {node} saved as '{filename}'.")
        plt.close()
        print(f"Heatmap for node {node} saved as '{filename}'.")





    main()if __name__ == "__main__":if __name__ == "__main__":
    main()
