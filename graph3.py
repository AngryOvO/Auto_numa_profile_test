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
# (A) /proc/node_pfn_stats 파일 파싱 함수
# 각 노드의 전체 PFN 범위를 읽어 {node: (start_pfn, end_pfn)} 형식으로 반환합니다.
# 파일 형식 예:
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
            # 라인 예: "start pfn 1, end pfn 5300000" 또는 "start pfn 5300001, 10000000"
            m = re.search(r'start pfn\s+(\d+)[,]?\s*(?:end pfn\s+)?(\d+)', line)
            if m and current_node is not None:
                start_pfn = int(m.group(1))
                end_pfn = int(m.group(2))
                node_ranges[current_node] = (start_pfn, end_pfn)
                current_node = None  # 노드 정보 초기화
    return node_ranges

# ----------------------------------------------------------------------
# (B) 메인 함수: 워크로드 실행, 데이터 수집, binning을 적용한 히트맵 생성, /proc/numa_folio_stats 삭제
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="워크로드 실행 동안 /proc/numa_folio_stats 데이터를 수집한 후, "
                    "노드별 전체 PFN 범위(/proc/node_pfn_stats)를 기준으로 binning하여 히트맵을 생성합니다. "
                    "색상은 기본 남색에서 마이그레이션 수치가 증가할수록 붉게 표현됩니다. "
                    "프로세스 종료 후 /proc/numa_folio_stats 파일은 삭제됩니다."
    )
    parser.add_argument(
        "command", nargs="+",
        help="실행할 워크로드 명령어 및 인자 (예: /usr/bin/my_workload arg1 arg2)"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="스냅샷 수집 간격 (초, 기본값: 2초)"
    )
    parser.add_argument(
        "--bin-size", type=int, default=10000,
        help="PFN bin size (예: 10000, 기본값: 10000)"
    )
    args = parser.parse_args()

    # 사용자 정의 colormap: 남색(navy)에서 붉은색(red)으로 선형 보간
    custom_cmap = LinearSegmentedColormap.from_list("NavyToRed", ["navy", "red"], N=256)

    # 워크로드 실행 (subprocess)
    print("워크로드 실행:", args.command)
    proc = subprocess.Popen(args.command)

    # /proc/numa_folio_stats 데이터를 누적합니다.
    # 각 항목: [node, pfn, source_nid, migrate_count, snapshot]
    collected_data = []
    snapshot = 0
    numa_file = '/proc/numa_folio_stats'

    print("워크로드 실행 중 데이터 수집 시작...")
    try:
        while proc.poll() is None:
            snapshot += 1
            try:
                with open(numa_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"'{numa_file}' 파일 읽기 실패: {e}")
                sys.exit(1)
            # 각 스냅샷의 모든 라인 파싱
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
            print(f"Snapshot {snapshot} 수집 완료.")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("사용자 요청으로 데이터 수집 중단. 워크로드 종료...")
        proc.terminate()
        proc.wait()

    print("워크로드 종료됨. 수집된 데이터를 기반으로 히트맵을 생성합니다.")

    # 누적된 데이터를 DataFrame으로 변환
    df = pd.DataFrame(
        collected_data,
        columns=['node', 'pfn', 'source_nid', 'migrate_count', 'snapshot']
    )
    print("수집된 데이터 예시:")
    print(df.head())

    # /proc/node_pfn_stats 파일에서 각 노드의 전체 PFN 범위를 파싱합니다.
    node_ranges = parse_node_pfn_stats('/proc/node_pfn_stats')
    print("노드 전체 PFN 범위:")
    print(node_ranges)

    # 각 노드별로 binning 기법을 적용하여 히트맵을 생성합니다.
    all_nodes = set(list(node_ranges.keys()))
    for node in all_nodes:
        start_pfn, end_pfn = node_ranges[node]
        bin_size = args.bin_size
        # 전체 PFN 범위에서 bin 경계를 생성 (start_pfn부터 bin_size 단위)
        bins = np.arange(start_pfn, end_pfn + bin_size, bin_size)
        # 각 bin의 중앙값을 라벨로 사용합니다.
        labels = [(bins[i] + bins[i+1]) // 2 for i in range(len(bins) - 1)]
        overall_bin_labels = labels  # 재인덱싱에 사용할 라벨 리스트

        # 해당 노드의 데이터 선택
        node_df = df[df['node'] == node].copy()
        if node_df.empty:
            print(f"노드 {node}에 대해 수집된 데이터가 없습니다. (빈 히트맵 생성)")
        # 'pfn' 정보를 binning: pd.cut으로 PFN을 bin에 할당 (라벨은 각 bin의 중앙값)
        node_df['pfn_bin'] = pd.cut(
            node_df['pfn'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # pivot table 생성: index는 pfn_bin, columns는 snapshot, 값은 migrate_count
        pivot_table = node_df.pivot_table(
            index='pfn_bin',
            columns='snapshot',
            values='migrate_count',
            aggfunc='first'
        )
        # 전체 bin 범위에 대해 재인덱싱, 해당 bin에 데이터가 없으면 0으로 채움
        pivot_table = pivot_table.reindex(overall_bin_labels).fillna(0)
    
        plt.figure(figsize=(12, 8))
        # custom_cmap를 사용하여 히트맵 그리기: 기본 남색에서 붉게
        sns.heatmap(pivot_table, cmap=custom_cmap, cbar=True)
        plt.title(f"Node {node} - Migrate Count (Binned) During Workload")
        plt.xlabel("Snapshot (Time)")
        plt.ylabel(f"PFN Bin Center (from {start_pfn} to {end_pfn}, bin size {bin_size})")
        plt.tight_layout()
        filename = f"node_{node}_workload_binned_heatmap.png"
        plt.savefig(filename)
        plt.close()
        print(f"노드 {node} 히트맵이 '{filename}' 로 저장되었습니다.")

    # 워크로드 종료 후 /proc/numa_folio_stats 파일 삭제 (공정하게 초기화)
    if os.path.exists(numa_file):
        try:
            os.remove(numa_file)
            print(f"'{numa_file}' 파일이 삭제되었습니다.")
        except Exception as e:
            print(f"'{numa_file}' 파일 삭제 실패: {e}")

if __name__ == "__main__":
    main()
