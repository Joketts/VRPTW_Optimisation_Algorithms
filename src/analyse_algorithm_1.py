import pandas as pd
import scipy.stats as stats

df_edf = pd.read_csv("benchmark_results_EDF_solomon.csv")
df_sa_solomon = pd.read_csv("benchmark_results_SA_solomon.csv")
df_sa_homberger = pd.read_csv("benchmark_results_SA_homberger.csv")

# 1) EDF Solomon vs SA Solomon
u_stat_1, p_val_1 = stats.mannwhitneyu(df_edf["total_distance"],
                                       df_sa_solomon["total_distance"],
                                       alternative="two-sided")
print("Mann–Whitney U: EDF Solomon vs. SA Solomon")
print(f"  U statistic = {u_stat_1:.8e}, p-value = {p_val_1:.8e}")
if p_val_1 < 0.05:
    print("  => distributions differ significantly at the 5% level.\n")
else:
    print("  => no significant difference at the 5% level.\n")


# 2) EDF Solomon vs SA Homberger
u_stat_2, p_val_2 = stats.mannwhitneyu(df_edf["total_distance"],
                                       df_sa_homberger["total_distance"],
                                       alternative="two-sided")
print("Mann–Whitney U: EDF Solomon vs. SA Homberger")
print(f"  U statistic = {u_stat_2:.8e}, p-value = {p_val_2:.8e}")
if p_val_2 < 0.05:
    print("  => distributions differ significantly at the 5% level.\n")
else:
    print("  => no significant difference at the 5% level.\n")


# 3) SA Solomon vs SA Homberger
u_stat_3, p_val_3 = stats.mannwhitneyu(df_sa_solomon["total_distance"],
                                       df_sa_homberger["total_distance"],
                                       alternative="two-sided")
print("Mann–Whitney U: SA Solomon vs. SA Homberger")
print(f"  U statistic = {u_stat_3:.8e}, p-value = {p_val_3:.8e}")
if p_val_3 < 0.05:
    print("  => distributions differ significantly at the 5% level.\n")
else:
    print("  => no significant difference at the 5% level.\n")
