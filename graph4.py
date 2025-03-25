#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import mmap
import ctypes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import re  # 정규 표현식 모듈 추가

# 커널에서 사용하는 구조체 정의
class NumaFolioStat(ctypes.Structure):
    _fields_ = [
        ("source_nid", ctypes.c_int),          # 원래 NUMA 노드 ID
        ("migrate_count", ctypes.c_int),       # 마이그레이션 횟수 (atomic_t를 int로 매핑)
    ]

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

def get_total_size(filepath='/proc/node_pfn_stats'):
    """
    /proc/node_pfn_stats에서 NUMA 데이터의 총 크기를 계산합니다.
    """
    total_size = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("total size"):
                    total_size = int(line.split(":")[1].strip().split()[0])
                    break
    except Exception as e:
        print(f"Error reading total size from {filepath}: {e}")
        sys.exit(1)
    return total_size

def read_numa_data_mmap(device_path="/dev/numa_mmap", size=4096, node_ranges=None):
    """
    mmap을 사용하여 NUMA 데이터를 읽습니다.
    """
    try:
        with open(device_path, "r+b") as f:
            # 메모리 매핑
            mmapped_data = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)

            # 구조체 배열로 매핑
            data = []
            offset = 0
            for nid, (start_pfn, end_pfn) in node_ranges.items():
                node_data = []
                for pfn in range(start_pfn, end_pfn + 1):
                    # 구조체 읽기
                    stat = NumaFolioStat.from_buffer_copy(mmapped_data[offset:offset + ctypes.sizeof(NumaFolioStat)])
                    node_data.append((nid, pfn, stat.source_nid, stat.migrate_count))
                    offset += ctypes.sizeof(NumaFolioStat)
                data.extend(node_data)

            mmapped_data.close()
            return data
    except Exception as e:
        print(f"Error reading from {device_path} using mmap: {e}")
        sys.exit(1)

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
        description="Collects NUMA data using mmap and generates heatmaps."
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

    # NUMA 데이터 크기 계산
    mmap_size = get_total_size()
    if mmap_size == 0:
        print("Error: Failed to determine mmap size.")
        sys.exit(1)

    # NUMA 노드 범위 파싱
    node_ranges = parse_node_pfn_stats("/proc/node_pfn_stats")
    print("Node PFN ranges:\n", node_ranges)

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
    mmap_device = "/dev/numa_mmap"

    print("Starting data collection...")
    try:
        while proc.poll() is None:  # 워크로드가 실행 중일 때
            snapshot += 1
            # mmap을 통해 데이터 읽기
            raw_data = read_numa_data_mmap(mmap_device, mmap_size, node_ranges)
            for nid, pfn, source_nid, migrate_count in raw_data:
                collected_data.append([nid, pfn, source_nid, migrate_count, snapshot])
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Data collection interrupted. Terminating workload...")
        proc.terminate()
        proc.wait()

    print("Data collection complete. Generating heatmaps...")

    if not collected_data:
        print("No data collected. Exiting.")
        return

    # 데이터프레임 생성 및 스냅샷 범위 확보
    df = pd.DataFrame(
        collected_data,
        columns=["node", "pfn", "source_nid", "migrate_count", "snapshot"],
    )
    print("Collected data sample:\n", df.head())

    # 스냅샷 범위 설정
    if df.empty:
        print("No data collected. Using default snapshot range.")
        all_snapshots = range(0, 2)  # 기본 스냅샷 범위
    else:
        all_snapshots = range(df["snapshot"].min(), df["snapshot"].max() + 1)

    # 모든 노드에서의 최대 migrate_count 계산
    global_vmax = df["migrate_count"].max() if not df.empty else 1
    print(f"Global vmax (maximum migrate_count): {global_vmax}")

    # 노드별 히트맵 생성
    for node in node_ranges.keys():  # 모든 노드에 대해 처리
        node_df = df[df["node"] == node]

        if not node_df.empty:
            pivot = node_df.pivot_table(
                index="pfn",
                columns="snapshot",
                values="migrate_count",
                aggfunc="sum",
                fill_value=0,
            ).fillna(0)

            # 컬러맵 범위 설정 (vmin=0, vmax=global_vmax)
            sns.heatmap(
                pivot,
                cmap=LinearSegmentedColormap.from_list(
                    "Thermal", ["navy", "red", "yellow"], N=256
                ),
                cbar=True,
                vmin=0,
                vmax=global_vmax,
                annot=False,
            )

            plt.title(f"Node {node} - Migration Heatmap")
            plt.xlabel("Snapshot (Time)")
            plt.ylabel("PFN")
            plt.tight_layout()

            filename = f"node_{node}_migration_heatmap.png"
            plt.savefig(filename)
            print(f"Heatmap for node {node} saved as '{filename}'.")
            plt.close()

if __name__ == "__main__":
    main()
