#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(
        description="특정 워크로드(프로세스)를 실행하면서 /proc/numa_folio_stats를 수집해 노드별 히트맵을 생성합니다."
    )
    parser.add_argument(
        "command",
        nargs="+",
        help="실행할 워크로드 명령어 및 인자 (예: /usr/bin/my_workload arg1 arg2)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="스냅샷 수집 간격 (초 단위, 기본값: 2초)"
    )
    args = parser.parse_args()

    # 워크로드 프로세스 시작
    print("워크로드 실행:", args.command)
    proc = subprocess.Popen(args.command)

    collected_data = []  # 각 원소: [node, pfn, source_nid, migrate_count, snapshot]
    snapshot = 0
    log_file_path = '/proc/numa_folio_stats'

    try:
        # 워크로드 프로세스가 살아 있는 동안 주기적으로 /proc/numa_folio_stats 읽기
        while proc.poll() is None:
            snapshot += 1
            try:
                with open(log_file_path, 'r') as f:
                    log_lines = f.readlines()
            except Exception as e:
                print(f"'{log_file_path}' 파일 읽기 실패: {e}")
                sys.exit(1)
            # 로그 라인 파싱
            for line in log_lines:
                match = re.search(
                    r'folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)',
                    line
                )
                if match:
                    node = int(match.group(1))
                    pfn = int(match.group(2))
                    source_nid = int(match.group(3))
                    migrate_count = int(match.group(4))
                    collected_data.append([node, pfn, source_nid, migrate_count, snapshot])
            time.sleep(args.interval)
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

    print("워크로드 종료. 수집된 데이터를 기반으로 히트맵을 생성합니다.")
    
    # 수집된 데이터를 DataFrame으로 변환
    df = pd.DataFrame(
        collected_data,
        columns=['node', 'pfn', 'source_nid', 'migrate_count', 'snapshot']
    )
    print("수집된 데이터 (상위 5개 항목):")
    print(df.head())

    # 노드별 히트맵 생성
    nodes = df['node'].unique()
    for node in nodes:
        node_df = df[df['node'] == node].copy()
        
        # y축: 해당 노드에서 관측된 PFN의 최소값 ~ 최대값 (연속적 PFN)
        min_pfn = node_df['pfn'].min()
        max_pfn = node_df['pfn'].max()
        all_pfns = np.arange(min_pfn, max_pfn + 1)
        
        # pivot_table을 이용해 pivot: 행은 PFN, 열은 snapshot, 값은 migrate_count
        pivot_table = node_df.pivot_table(
            index='pfn', columns='snapshot', values='migrate_count', aggfunc='first'
        )
        pivot_table = pivot_table.reindex(all_pfns).fillna(0)
        
        # 히트맵 그리기
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='YlGnBu', cbar=True)
        plt.title(f'Node {node} - Migrate Count During Workload')
        plt.xlabel('Snapshot (Time)')
        plt.ylabel(f'PFN (from {min_pfn} to {max_pfn})')
        plt.tight_layout()
        
        output_filename = f'node_{node}_workload_heatmap.png'
        plt.savefig(output_filename)
        plt.close()
        print(f"노드 {node} 히트맵이 '{output_filename}' 로 저장되었습니다.")

if __name__ == "__main__":
    main()