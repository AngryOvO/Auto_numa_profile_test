import matplotlib.pyplot as plt
import numpy as np

def parse_numa_folio_stat(file_path):
    pfns = []
    migrate_counts = []
    
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(", ")
            pfn = int(parts[1].split(": ")[1])
            migrate_count = int(parts[3].split(": ")[1])
            
            pfns.append(pfn)
            migrate_counts.append(migrate_count)
    
    return pfns, migrate_counts

def plot_numa_folio_stat(file_path):
    pfns, migrate_counts = parse_numa_folio_stat(file_path)
    times = np.arange(len(pfns))
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(times, pfns, c=migrate_counts, cmap="plasma", edgecolors="black", s=50)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Migrate Count")
    
    plt.xlabel("Time Step")
    plt.ylabel("PFN")
    plt.title("PFN Migrate Count Over Time")
    plt.gca().invert_yaxis()
    
    plt.show()

# 파일 경로를 입력하여 실행
target_file = "/proc/numa_folio_stat"  # 여기에 /proc/numa_folio_stat 경로를 지정
plot_numa_folio_stat(target_file)
