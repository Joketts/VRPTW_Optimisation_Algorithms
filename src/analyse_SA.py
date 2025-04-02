import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load Data
# Adjust these file paths/names if necessary
df_solomon = pd.read_csv("benchmark_results_SA_solomon.csv")
df_homberger = pd.read_csv("benchmark_results_SA_homberger.csv")
df_edf = pd.read_csv("benchmark_results_EDF_solomon.csv")

# -------------------------------------------------------------------
# 2. Basic Inspection
# -------------------------------------------------------------------
print("Solomon Data Info:")
print(df_solomon.info())
print(df_solomon.head(), "\n")

print("Homberger Data Info:")
print(df_homberger.info())
print(df_homberger.head(), "\n")

print("EDF Solomon Data Info:")
print(df_edf.info())
print(df_edf.head(), "\n")

# -------------------------------------------------------------------
# 3. Basic Descriptive Statistics
#    Make sure the column is correct (here, "total_distance").
# -------------------------------------------------------------------
print("Solomon Descriptive Stats:\n", df_solomon["total_distance"].describe(), "\n")
print("Homberger Descriptive Stats:\n", df_homberger["total_distance"].describe(), "\n")
print("EDF Solomon Descriptive Stats:\n", df_edf["total_distance"].describe(), "\n")

# -------------------------------------------------------------------
# 4. Normality Check (Shapiro–Wilk)
#    If p-value < 0.05 => data likely not normal.
#    If p-value >= 0.05 => no evidence to reject normality.
# -------------------------------------------------------------------
print("Shapiro–Wilk for Solomon total_distance:")
shapiro_solomon = stats.shapiro(df_solomon["total_distance"])
print(f"  statistic={shapiro_solomon.statistic:.4f}, pvalue={shapiro_solomon.pvalue:.8e}")

print("Shapiro–Wilk for Homberger total_distance:")
shapiro_homberger = stats.shapiro(df_homberger["total_distance"])
print(f"  statistic={shapiro_homberger.statistic:.4f}, pvalue={shapiro_homberger.pvalue:.8e}")

print("Shapiro–Wilk for EDF Solomon total_distance:")
shapiro_edf = stats.shapiro(df_edf["total_distance"])
print(f"  statistic={shapiro_edf.statistic:.4f}, pvalue={shapiro_edf.pvalue:.6f}\n")

# -------------------------------------------------------------------
# 5. Variance Check (Bartlett’s Test = an F-like test for homogeneity)
#    Example: Compare variance of SA Solomon vs. EDF Solomon
#    If p-value < 0.05 => variances differ significantly.
#    If p-value >= 0.05 => no evidence against equal variance.
# -------------------------------------------------------------------
bartlett_stat, bartlett_p = stats.bartlett(df_solomon["total_distance"],
                                           df_edf["total_distance"])
print("Bartlett’s Test (SA Solomon vs. EDF Solomon) for equal variance:")
print(f"  statistic={bartlett_stat:.4f}, pvalue={bartlett_p:.6f}")
if bartlett_p < 0.05:
    print("  => p < 0.05, variances differ significantly (cannot assume equal var).")
else:
    print("  => p >= 0.05, no evidence against equal variance.\n")

# If you also want to compare Homberger vs. EDF or other pairs, just repeat:
"""
bartlett_stat_h, bartlett_p_h = stats.bartlett(df_homberger["total_distance"],
                                              df_edf["total_distance"])
print("Bartlett’s Test (SA Homberger vs. EDF Solomon) for equal variance:")
print(f"  statistic={bartlett_stat_h:.4f}, pvalue={bartlett_p_h:.6f}")
"""

# -------------------------------------------------------------------
# 6. Boxplots (Visual Distribution)
# -------------------------------------------------------------------


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# -------------------------------------------------------------------
# (1) Subplot #1: EDF vs. SA on Solomon
# -------------------------------------------------------------------
ax1.boxplot(
    [df_edf["total_distance"], df_sa_solomon["total_distance"]],
    labels=["EDF (Solomon)", "SA (Solomon)"],
    showmeans=True
)
ax1.set_title("Solomon: EDF vs. SA")
ax1.set_ylabel("Total Distance")

# -------------------------------------------------------------------
# (2) Subplot #2: SA on Homberger & future data
# -------------------------------------------------------------------
# We'll place SA (Homberger) at x-position=1,
# and leave x-position=2 blank for when we have "EDF Homberger" (or another approach).

positions_sa = [1]  # Plot 'SA (Homberger)' at x=1
ax2.boxplot(df_sa_homberger["total_distance"],
            positions=positions_sa,
            widths=0.6,
            showmeans=True)

# We define two x-ticks: x=1 is SA (Homberger), x=2 is ??? for future data
ax2.set_xticks([1, 2])
ax2.set_xticklabels(["SA (Homberger)", "??? (Future)"])

ax2.set_title("Homberger: SA vs ???")
ax2.set_ylabel("Total Distance")

plt.tight_layout()
plt.show()
# -------------------------------------------------------------------
# 7. (Optional) Statistical Test to Compare Two Groups
# -------------------------------------------------------------------
# For example, a Mann–Whitney U test (non-parametric).
u_stat, p_val = stats.mannwhitneyu(df_solomon["total_distance"],
                                   df_homberger["total_distance"],
                                   alternative="two-sided")
print(f"Mann–Whitney U test between SA Solomon vs. SA Homberger:")
print(f"  U statistic = {u_stat:.4f}, p-value = {p_val:.6f}")
if p_val < 0.05:
    print("  => difference is statistically significant at 5% level.\n")
else:
    print("  => no significant difference at 5% level.\n")
