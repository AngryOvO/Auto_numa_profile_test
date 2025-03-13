import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# 로그 파일 경로
log_file_path = '/proc/numa_folio_stats'

# 로그 데이터를 읽어오기
with open(log_file_path, 'r') as file:
    log_lines = file.readlines()

# 데이터프레임을 생성하기 위한 리스트
data = []

# 로그 데이터를 파싱해서 리스트에 추가
for line in log_lines:
    match = re.search(r'folio node : (\d+), pfn: (\d+), source_nid: (\d+), migrate_count: (\d+)', line)
    if match:
        node = int(match.group(1))
        pfn = int(match.group(2))
        source_nid = int(match.group(3))
        migrate_count = int(match.group(4))
        data.append([node, pfn, source_nid, migrate_count])

# 데이터프레임 생성
df = pd.DataFrame(data, columns=['node', 'pfn', 'source_nid', 'migrate_count'])

# 로그 데이터에 시간 추가 (예시 데이터를 위해 임의의 시간 생성)
time_stamps = np.linspace(0, 10, len(df))  # 0부터 10 사이에 균등한 시간 생성
df['time'] = time_stamps

# 노드별로 히트맵을 생성
nodes = df['node'].unique()
for node in nodes:
    node_df = df[df['node'] == node]

    # 히트맵을 생성 및 저장
    plt.figure(figsize=(10, 6))
    pivot_df = node_df.pivot("pfn", "time", "migrate_count")
    sns.heatmap(pivot_df, cmap="YlGnBu", cbar=True)

    plt.title(f'Node {node} Migration Counts Over Time')
    plt.xlabel('Time')
    plt.ylabel('PFN')
    plt.savefig(f'node_{node}_migration_heatmap.png')  # PNG 형식으로 저장
    plt.close()  # 그래프를 닫아서 메모리 절약

    print(f'Node {node} heatmap saved as node_{node}_migration_heatmap.png')
