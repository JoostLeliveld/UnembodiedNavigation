import pandas as pd
import numpy as np

csv_path = "/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/aws_targets_v7b_col461/perception_targets.csv"
df = pd.read_csv(csv_path)

# Convert columns to numeric
df['x'] = pd.to_numeric(df['x'])
df['y'] = pd.to_numeric(df['y'])
df['theta'] = pd.to_numeric(df['theta'])
df['yolo_score_raw'] = pd.to_numeric(df['yolo_score_raw'])

# Group by (x, y) and compute standard deviation and range of the yolo_score_raw
stats = df.groupby(['x', 'y'])['yolo_score_raw'].agg(['std', 'min', 'max', 'count', 'mean'])
stats['range'] = stats['max'] - stats['min']

# Remove locations with fewer than 2 heading samples (to compute std dev)
stats_valid = stats[stats['count'] > 1]

mean_std = stats_valid['std'].mean()
max_std = stats_valid['std'].max()
mean_range = stats_valid['range'].mean()
max_range = stats_valid['range'].max()

print(f"Total valid (x, y) groups: {len(stats_valid)}")
print(f"Mean Standard Deviation of YOLO raw score across orientations: {mean_std:.4f}")
print(f"Max Standard Deviation of YOLO raw score across orientations: {max_std:.4f}")
print(f"Mean Range (Max - Min) of YOLO raw score across orientations: {mean_range:.4f}")
print(f"Max Range (Max - Min) of YOLO raw score across orientations: {max_range:.4f}")

# Let's count how many have std > 0.1, std > 0.2, etc.
print(f"Groups with std > 0.05: {len(stats_valid[stats_valid['std'] > 0.05])} ({len(stats_valid[stats_valid['std'] > 0.05])/len(stats_valid)*100:.1f}%)")
print(f"Groups with std > 0.10: {len(stats_valid[stats_valid['std'] > 0.10])} ({len(stats_valid[stats_valid['std'] > 0.10])/len(stats_valid)*100:.1f}%)")
print(f"Groups with std > 0.20: {len(stats_valid[stats_valid['std'] > 0.20])} ({len(stats_valid[stats_valid['std'] > 0.20])/len(stats_valid)*100:.1f}%)")
