import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 파일 경로 지정
log_file_path = '/proc/numa_folio_stats'

# 로그 파일 읽기
with open(log_file_path, 'r') as file:
    log_lines = file.readlines()

# 로그 한 줄씩 파싱하면서 시간(스냅샷) 정보를 할당
data = []
time_counter = 0
current_snapshot_pfns = set()  # 현재 스냅샷에서 읽은 PFN을 저장

for line in log_lines:
    # 정규표현식으로 각 필드 추출
    match = re.search(r'folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)', line)
    if match:
        node = int(match.group(1))
        pfn = int(match.group(2))
        source_nid = int(match.group(3))
        migrate_count = int(match.group(4))
        
        # 만약 현재 스냅샷에서 이미 이 pfn이 등장했다면 새로운 스냅샷으로 판단
        if pfn in current_snapshot_pfns:
            time_counter += 1
            current_snapshot_pfns.clear()  # 새로운 스냅샷이 시작되었으므로 초기화
        current_snapshot_pfns.add(pfn)
        
        # 데이터에 시간 값을 포함시켜 저장
        data.append([node, pfn, source_nid, migrate_count, time_counter])

# 데이터프레임 생성
df = pd.DataFrame(data, columns=['node', 'pfn', 'source_nid', 'migrate_count', 'time'])
print(df)

# 노드별로 히트맵 생성 및 PNG 저장
nodes = df['node'].unique()
for node in nodes:
    node_df = df[df['node'] == node].copy()
    
    # 해당 노드에 대해 최소값과 최대값의 PFN 결정 (y축 범위)
    min_pfn = node_df['pfn'].min()
    max_pfn = node_df['pfn'].max()
    pfn_range = np.arange(min_pfn, max_pfn + 1)
    
    # pivot_table 생성:
    # - index: PFN
    # - columns: time (스냅샷)
    # - 값: migrate_count (해당 스냅샷에 PFN이 없는 경우 NaN → 0으로 대체)
    pivot_table = node_df.pivot_table(index='pfn', columns='time', values='migrate_count', aggfunc='first')
    pivot_table = pivot_table.reindex(pfn_range)
    pivot_table = pivot_table.fillna(0)
    
    # 히트맵 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, cmap='YlGnBu', cbar=True)
    plt.title(f'Node {node} - Migrate Count Over Time')
    plt.xlabel('Snapshot (Time)')
    plt.ylabel(f'PFN (from {min_pfn} to {max_pfn})')
    plt.tight_layout()
    
    # PNG 파일로 저장
    filename = f'node_{node}_migration_heatmap.png'
    plt.savefig(filename)
    plt.close()
    
    print(f'Heatmap for node {node} saved as {filename}')
