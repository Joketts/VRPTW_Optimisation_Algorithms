import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


df_edf_solomon = pd.read_csv("benchmark_results_EDF_solomon.csv")
df_sa_solomon = pd.read_csv("benchmark_results_SA_solomon.csv")
df_sa_homberger = pd.read_csv("benchmark_results_SA_homberger.csv")
df_edf_homberger = pd.read_csv("benchmark_results_EDF_homberger.csv")

df_best_solomon = pd.read_excel("Benchmark_best_results_solomon.xlsx")
df_best_homberger = pd.read_excel("Benchmark_best_results_homberger.xlsx")


df_edf_solomon["dataset"] = df_edf_solomon["dataset"].str.lower()
df_sa_solomon["dataset"] = df_sa_solomon["dataset"].str.lower()
df_sa_homberger["dataset"] = df_sa_homberger["dataset"].str.lower()
df_edf_homberger["dataset"] = df_edf_homberger["dataset"].str.lower()

df_best_solomon.rename(columns={"Distance": "best_distance"}, inplace=True)
df_best_homberger.rename(columns={"Distance": "best_distance"}, inplace=True)


df_best_solomon["dataset"] = df_best_solomon["dataset"].str.lower()
df_best_homberger["dataset"] = df_best_homberger["dataset"].str.lower()


df_edf_sol_merged = df_edf_solomon.merge(
    df_best_solomon[["dataset", "best_distance"]],
    on="dataset",
    how="left"
)
df_edf_sol_merged["gap_percent"] = (
        100 * (df_edf_sol_merged["total_distance"] - df_edf_sol_merged["best_distance"])
        / df_edf_sol_merged["best_distance"]
)
df_edf_sol_merged["Algorithm"] = "IEDF (Solomon)"

df_sa_sol_merged = df_sa_solomon.merge(
    df_best_solomon[["dataset", "best_distance"]],
    on="dataset",
    how="left"
)
df_sa_sol_merged["gap_percent"] = (
        100 * (df_sa_sol_merged["total_distance"] - df_sa_sol_merged["best_distance"])
        / df_sa_sol_merged["best_distance"]
)
df_sa_sol_merged["Algorithm"] = "SA (Solomon)"

df_edf_hom_merged = df_edf_homberger.merge(
    df_best_homberger[["dataset", "best_distance"]],
    on="dataset",
    how="left"
)
df_edf_hom_merged["gap_percent"] = (
        100 * (df_edf_hom_merged["total_distance"] - df_edf_hom_merged["best_distance"])
        / df_edf_hom_merged["best_distance"]
)
df_edf_hom_merged["Algorithm"] = "IEDF (Homberger)"

df_sa_hom_merged = df_sa_homberger.merge(
    df_best_homberger[["dataset", "best_distance"]],
    on="dataset",
    how="left"
)
df_sa_hom_merged["gap_percent"] = (
        100 * (df_sa_hom_merged["total_distance"] - df_sa_hom_merged["best_distance"])
        / df_sa_hom_merged["best_distance"]
)
df_sa_hom_merged["Algorithm"] = "SA (Homberger)"


print("IEDF (Homberger) merged shape:", df_edf_hom_merged.shape)
print(df_edf_hom_merged.head(), "\n")

df_gap = pd.concat([
    df_edf_sol_merged[["dataset", "Algorithm", "gap_percent", "time_sec", "violation_count"]],
    df_sa_sol_merged[["dataset", "Algorithm", "gap_percent", "time_sec", "violation_count"]],
    df_edf_hom_merged[["dataset", "Algorithm", "gap_percent", "time_sec", "violation_count"]],
    df_sa_hom_merged[["dataset", "Algorithm", "gap_percent", "time_sec", "violation_count"]]
], ignore_index=True)

fig, ax = plt.subplots(figsize=(9, 5))

group_iedf_sol = df_gap[df_gap["Algorithm"] == "IEDF (Solomon)"]["gap_percent"].dropna()
group_sa_sol = df_gap[df_gap["Algorithm"] == "SA (Solomon)"]["gap_percent"].dropna()
group_iedf_hom = df_gap[df_gap["Algorithm"] == "IEDF (Homberger)"]["gap_percent"].dropna()
group_sa_hom = df_gap[df_gap["Algorithm"] == "SA (Homberger)"]["gap_percent"].dropna()

data_to_plot = [
    group_iedf_sol,   # 1
    group_sa_sol,     # 2
    group_iedf_hom,   # 3
    group_sa_hom      # 4
]

labels = [
    "IEDF (Solomon)",
    "SA (Solomon)",
    "IEDF (Homberger)",
    "SA (Homberger)"
]

ax.boxplot(data_to_plot, positions=[1,2,3,4], showmeans=True)
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(labels)
ax.set_title("Gap from Best-Known (%) by Algorithm-Benchmark")
ax.set_ylabel("Gap (%) (Lower is Better)")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
for alg in df_gap["Algorithm"].unique():
    subset = df_gap[df_gap["Algorithm"] == alg]
    plt.hist(subset["gap_percent"], bins=20, alpha=0.5, label=alg, edgecolor='black')

plt.title("Histogram of Gap (%) by Algorithm")
plt.xlabel("Gap (%) (Lower is Better)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(x="Algorithm", y="gap_percent", data=df_gap, inner="quartile")
plt.title("Violin Plot of Gap (%) by Algorithm")
plt.xlabel("Algorithm")
plt.ylabel("Gap (%) (Lower is Better)")
plt.tight_layout()
plt.show()


def plot_hist_kde(df_gap):
    plt.figure(figsize=(8, 5))
    for alg in df_gap["Algorithm"].unique():
        subset = df_gap[df_gap["Algorithm"] == alg]
        sns.histplot(subset["gap_percent"], kde=True, label=alg, alpha=0.5, edgecolor='black')
    plt.title("Histogram + KDE of gap_percent by Algorithm")
    plt.xlabel("Gap (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_hist_kde(df_gap)

def plot_qq(df_gap, algorithm):
    subset = df_gap[df_gap["Algorithm"] == algorithm]["gap_percent"].dropna()
    sm.qqplot(subset, line='s')
    plt.title(f"Q–Q Plot: Gap Percent for {algorithm}")
    plt.show()

plot_qq(df_gap, "SA (Solomon)")
plot_qq(df_gap, "IEDF (Solomon)")

def per_instance_boxplot_benchmark(df_gap, benchmark):
    df_filtered = df_gap[df_gap["Algorithm"].str.contains(benchmark)]
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="dataset", y="gap_percent", hue="Algorithm", data=df_filtered)
    plt.xticks(rotation=90)
    plt.title(f"Gap (%) by Dataset and Algorithm ({benchmark} Benchmark)")
    plt.ylabel("Gap (%) (Lower is Better)")
    plt.tight_layout()
    plt.show()

def per_instance_bar_benchmark(df_gap, benchmark):
    df_filtered = df_gap[df_gap["Algorithm"].str.contains(benchmark)]
    summary = df_filtered.groupby(["dataset", "Algorithm"])["gap_percent"].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x="dataset", y="gap_percent", hue="Algorithm", data=summary, edgecolor='black')
    plt.xticks(rotation=90)
    plt.title(f"Average Gap (%) per Instance, by Algorithm ({benchmark} Benchmark)")
    plt.ylabel("Mean Gap (%)")
    plt.tight_layout()
    plt.show()

per_instance_boxplot_benchmark(df_gap, "Solomon")
per_instance_bar_benchmark(df_gap, "Solomon")
per_instance_boxplot_benchmark(df_gap, "Homberger")
per_instance_bar_benchmark(df_gap, "Homberger")

def plot_time_vs_gap(df):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x="time_sec", y="gap_percent", hue="Algorithm", data=df)

    # Change x-axis to a logarithmic scale
    plt.xscale("log")
    plt.title("Time (sec) vs. Gap (%) (Log Scale)")
    plt.xlabel("Run Time (sec) (Log Scale)")
    plt.ylabel("Gap (%) (Lower is Better)")
    plt.tight_layout()
    plt.show()

plot_time_vs_gap(df_gap)

def violation_boxplot(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Algorithm", y="violation_count", data=df)
    plt.title("Violation Count by Algorithm & Dataset")
    plt.ylabel("Violation Count")
    plt.xlabel("Algorithm & Dataset")
    plt.tight_layout()
    plt.show()

violation_boxplot(df_gap)

def gap_heatmap(df_gap):
    pivot_data = df_gap.groupby(["dataset", "Algorithm"])["gap_percent"].mean().unstack()
    plt.figure(figsize=(8, 10))
    sns.heatmap(pivot_data, annot=True, cmap="RdYlGn_r", fmt=".1f")
    plt.title("Mean Gap (%) by Instance and Algorithm")
    plt.ylabel("Dataset")
    plt.xlabel("Algorithm")
    plt.tight_layout()
    plt.show()

gap_heatmap(df_gap)

def multi_dimension_pairplot(df):
    numeric_cols = ["gap_percent", "time_sec", "violation_count"]
    subset = df[numeric_cols + ["Algorithm"]].dropna()
    sns.pairplot(subset, hue="Algorithm", diag_kind="kde")
    plt.suptitle("Pairwise Scatter Matrix", y=1.02)
    plt.show()

multi_dimension_pairplot(df_gap)
