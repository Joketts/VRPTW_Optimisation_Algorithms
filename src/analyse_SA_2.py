import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1) Read Your SA Results (CSV)
# ---------------------------------------------------------------------
df_solomon = pd.read_csv("benchmark_results_SA_solomon.csv")
# This file has columns:
#   dataset, algorithm, run, total_distance, total_cost, time_sec, violation_count, total_lateness

# ---------------------------------------------------------------------
# 2) Read Best-Known Solutions (Excel)
# ---------------------------------------------------------------------
df_best_solomon = pd.read_excel("Benchmark_best_results_solomon.xlsx")
# This file has columns:
#   dataset, Vehicles, Distance, Reference, Comment

# ---------------------------------------------------------------------
# 3) Rename 'Distance' to 'best_distance' in the Best-Known Data
#    so we can more easily merge and reference it
# ---------------------------------------------------------------------
df_best_solomon.rename(columns={"Distance": "best_distance"}, inplace=True)

# If you also want to compare the number of vehicles, you can rename it too:
# df_best_solomon.rename(columns={"Vehicles": "best_vehicles"}, inplace=True)

# ---------------------------------------------------------------------
# 4) Merge on 'dataset'
# ---------------------------------------------------------------------
df_merged = df_solomon.merge(df_best_solomon[["dataset", "best_distance"]],
                             on="dataset", how="left")

# Check we have the merged columns now
# print(df_merged.head())

# ---------------------------------------------------------------------
# 5) Compute Gap vs. Best-Known
# ---------------------------------------------------------------------
# Absolute gap: difference in total_distance from best_distance
df_merged["distance_gap"] = df_merged["total_distance"] - df_merged["best_distance"]

# Percentage gap: how many % above the best-known solution
df_merged["gap_percent"] = (df_merged["distance_gap"] / df_merged["best_distance"]) * 100

# ---------------------------------------------------------------------
# 6) Boxplot of the Percentage Gap
# ---------------------------------------------------------------------
plt.figure(figsize=(7,5))
df_merged.boxplot(column="gap_percent", grid=False)
plt.title("SA on Solomon — Gap from Best-Known Distance")
plt.ylabel("Gap (%) (Lower = Better)")
plt.show()

# ---------------------------------------------------------------------
# 7) (Optional) Per-Instance Boxplot
# ---------------------------------------------------------------------
# This shows a separate box for each dataset on the x-axis,
# displaying how your runs vary vs. the best-known solution.
"""
import seaborn as sns
plt.figure(figsize=(12,5))
sns.boxplot(x="dataset", y="gap_percent", data=df_merged)
plt.xticks(rotation=90)  # rotate dataset names if they overlap
plt.title("Gap (%) from Best-Known per Instance (Solomon, SA)")
plt.ylabel("Gap (%) (Lower = Better)")
plt.tight_layout()
plt.show()
"""
