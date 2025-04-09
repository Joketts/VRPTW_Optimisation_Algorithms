import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as stats


df_solomon = pd.read_csv("benchmark_results_SA_solomon.csv")
df_homberger = pd.read_csv("benchmark_results_SA_homberger.csv")
df_edf = pd.read_csv("benchmark_results_EDF_solomon.csv")
df_edf_homberger = pd.read_csv("benchmark_results_EDF_homberger.csv")

print("Solomon Data Info:")
print(df_solomon.info())
print(df_solomon.head(), "\n")

print("Homberger Data Info:")
print(df_homberger.info())
print(df_homberger.head(), "\n")

print("EDF Solomon Data Info:")
print(df_edf.info())
print(df_edf.head(), "\n")

print("EDF Homberger Data Info:")
print(df_edf_homberger.info())
print(df_edf_homberger.head(), "\n")


print("Solomon Descriptive Stats:\n", df_solomon["total_distance"].describe(), "\n")
print("Homberger Descriptive Stats:\n", df_homberger["total_distance"].describe(), "\n")
print("EDF Solomon Descriptive Stats:\n", df_edf["total_distance"].describe(), "\n")
print("EDF Homberger Descriptive Stats:\n", df_edf_homberger["total_distance"].describe(), "\n")


print("Shapiro–Wilk for SA Solomon total_distance:")
shapiro_solomon = stats.shapiro(df_solomon["total_distance"])
print(f"  statistic={shapiro_solomon.statistic:.4f}, pvalue={shapiro_solomon.pvalue:.6f}")

print("Shapiro–Wilk for SA Homberger total_distance:")
shapiro_homberger = stats.shapiro(df_homberger["total_distance"])
print(f"  statistic={shapiro_homberger.statistic:.4f}, pvalue={shapiro_homberger.pvalue:.6f}")

print("Shapiro–Wilk for EDF Solomon total_distance:")
shapiro_edf = stats.shapiro(df_edf["total_distance"])
print(f"  statistic={shapiro_edf.statistic:.4f}, pvalue={shapiro_edf.pvalue:.6f}")

print("Shapiro–Wilk for EDF Homberger total_distance:")
shapiro_edf_homberger = stats.shapiro(df_edf_homberger["total_distance"])
print(f"  statistic={shapiro_edf_homberger.statistic:.4f}, pvalue={shapiro_edf_homberger.pvalue:.6f}\n")


bartlett_stat, bartlett_p = stats.bartlett(df_edf["total_distance"],
                                           df_edf_homberger["total_distance"])

print("Bartlett’s test for IEDF Solomon vs. IEDF Homberger total_distance:")
print(f"  statistic={bartlett_stat:.4f}, pvalue={bartlett_p:.6f}")

if bartlett_p < 0.05:
    print("  => p < 0.05, variances differ significantly (cannot assume equal variance).")
else:
    print("  => p >= 0.05, no evidence against equal variance (can assume equal variance).")

ax1.boxplot(
    [df_edf["total_distance"], df_solomon["total_distance"]],
    labels=["EDF (Solomon)", "SA (Solomon)"],
    showmeans=True
)
ax1.set_title("Solomon: EDF vs. SA")
ax1.set_ylabel("Total Distance")

positions_sa = [1]
ax2.boxplot(df_homberger["total_distance"],
            positions=positions_sa,
            widths=0.6,
            showmeans=True)


ax2.set_xticks([1, 2])
ax2.set_xticklabels(["SA (Homberger)", "??? (Future)"])

ax2.set_title("Homberger: SA vs ???")
ax2.set_ylabel("Total Distance")

plt.tight_layout()
plt.show()

u_stat, p_val = stats.mannwhitneyu(df_solomon["total_distance"],
                                   df_homberger["total_distance"],
                                   alternative="two-sided")
print(f"Mann–Whitney U test between SA Solomon vs. SA Homberger:")
print(f"  U statistic = {u_stat:.4f}, p-value = {p_val:.6f}")
if p_val < 0.05:
    print("  => difference is statistically significant at 5% level.\n")
else:
    print("  => no significant difference at 5% level.\n")
