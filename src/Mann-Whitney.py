import pandas as pd
import scipy.stats as stats
import itertools

df_edf = pd.read_csv("benchmark_results_EDF_solomon.csv")
df_sa_solomon = pd.read_csv("benchmark_results_SA_solomon.csv")
df_sa_homberger = pd.read_csv("benchmark_results_SA_homberger.csv")
df_edf_homberger = pd.read_csv("benchmark_results_EDF_homberger.csv")


u_stat_1, p_val_1 = stats.mannwhitneyu(df_edf["total_distance"],
                                       df_sa_solomon["total_distance"],
                                       alternative="two-sided")
print("Mann–Whitney U: IEDF Solomon vs. SA Solomon")
print(f"  U statistic = {u_stat_1:.4f}, p-value = {p_val_1:.4f}")
if p_val_1 < 0.05:
    print("  => distributions differ significantly at the 5% level.\n")
else:
    print("  => no significant difference at the 5% level.\n")

u_stat_2, p_val_2 = stats.mannwhitneyu(df_edf["total_distance"],
                                       df_sa_homberger["total_distance"],
                                       alternative="two-sided")
print("Mann–Whitney U: IEDF Solomon vs. SA Homberger")
print(f"  U statistic = {u_stat_2:.4f}, p-value = {p_val_2:.4f}")
if p_val_2 < 0.05:
    print("  => distributions differ significantly at the 5% level.\n")
else:
    print("  => no significant difference at the 5% level.\n")

u_stat_3, p_val_3 = stats.mannwhitneyu(df_sa_solomon["total_distance"],
                                       df_sa_homberger["total_distance"],
                                       alternative="two-sided")
print("Mann–Whitney U: SA Solomon vs. SA Homberger")
print(f"  U statistic = {u_stat_3:.4f}, p-value = {p_val_3:.4f}")
if p_val_3 < 0.05:
    print("  => distributions differ significantly at the 5% level.\n")
else:
    print("  => no significant difference at the 5% level.\n")


welch_stat, welch_p = stats.ttest_ind(
    df_edf["total_distance"],
    df_edf_homberger["total_distance"],
    equal_var=False  # Welch's t-test
)
print("Welch’s t-test: IEDF Solomon vs. IEDF Homberger")
print(f"  t statistic = {welch_stat:.4f}, p-value = {welch_p:.4f}")
if welch_p < 0.05:
    print("  => distributions differ significantly at the 5% level.\n")
else:
    print("  => no significant difference at the 5% level.\n")


u_stat_4, p_val_4 = stats.mannwhitneyu(
    df_sa_homberger["total_distance"],
    df_edf_homberger["total_distance"],
    alternative="two-sided"
)
print("Mann–Whitney U: SA Homberger vs. IEDF Homberger")
print(f"  U statistic = {u_stat_4:.4f}, p-value = {p_val_4:.4f}")
if p_val_4 < 0.05:
    print("  => distributions differ significantly at the 5% level.\n")
else:
    print("  => no significant difference at the 5% level.\n")


groups_distance = {
    "SA Solomon": df_sa_solomon["total_distance"],
    "IEDF Solomon": df_edf["total_distance"],
    "SA Homberger": df_sa_homberger["total_distance"],
    "IEDF Homberger": df_edf_homberger["total_distance"]
}

kw_stat_dist, kw_p_dist = stats.kruskal(
    groups_distance["SA Solomon"],
    groups_distance["IEDF Solomon"],
    groups_distance["SA Homberger"],
    groups_distance["IEDF Homberger"]
)

print("Kruskal–Wallis Test (4 groups) on total_distance:")
print(f"  H statistic = {kw_stat_dist:.4f}, p-value = {kw_p_dist:.4f}")

if kw_p_dist < 0.05:
    print("  => At least one distribution differs significantly at the 5% level.\n")
else:
    print("  => No significant difference at the 5% level among the four groups.\n")


if kw_p_dist < 0.05:
    pairs = list(itertools.combinations(groups_distance.keys(), 2))
    alpha = 0.05
    m = len(pairs)
    bonf_alpha = alpha / m

    print(f"Pairwise Mann–Whitney U Tests on total_distance with Bonferroni-corrected alpha = {bonf_alpha:.4f}\n")

    for combo in pairs:
        group1, group2 = combo
        data1 = groups_distance[group1]
        data2 = groups_distance[group2]

        u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        significant = "SIGNIFICANT" if p_val < bonf_alpha else "not significant"

        print(f"  {group1} vs. {group2}: U={u_stat:.2f}, raw p={p_val:.4g}, "
              f"Bonf corrected alpha={bonf_alpha:.4f} => {significant}")
    print()


groups_time = {
    "SA Solomon": df_sa_solomon["time_sec"],
    "IEDF Solomon": df_edf["time_sec"],
    "SA Homberger": df_sa_homberger["time_sec"],
    "IEDF Homberger": df_edf_homberger["time_sec"]
}


kw_stat_time, kw_p_time = stats.kruskal(
    groups_time["SA Solomon"],
    groups_time["IEDF Solomon"],
    groups_time["SA Homberger"],
    groups_time["IEDF Homberger"]
)

print("Kruskal–Wallis Test (4 groups) on runtime:")
print(f"  H statistic = {kw_stat_time:.4f}, p-value = {kw_p_time:.4f}")

if kw_p_time < 0.05:
    print("  => At least one distribution differs significantly at the 5% level.\n")
else:
    print("  => No significant difference at the 5% level among the four groups.\n")


if kw_p_time < 0.05:
    pairs = list(itertools.combinations(groups_time.keys(), 2))
    alpha = 0.05
    m = len(pairs)
    bonf_alpha = alpha / m

    print(f"Pairwise Mann–Whitney U Tests on runtime with Bonferroni-corrected alpha = {bonf_alpha:.4f}\n")

    for combo in pairs:
        group1, group2 = combo
        data1 = groups_time[group1]
        data2 = groups_time[group2]

        u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        significant = "SIGNIFICANT" if p_val < bonf_alpha else "not significant"

        print(f"  {group1} vs. {group2}: U={u_stat:.2f}, raw p={p_val:.4g}, "
              f"Bonf corrected alpha={bonf_alpha:.4f} => {significant}")
    print()
