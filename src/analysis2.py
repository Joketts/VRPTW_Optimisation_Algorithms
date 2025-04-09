import pandas as pd
import scipy.stats as stats


df_solomon = pd.read_csv("benchmark_results_SA_solomon.csv")
df_homberger = pd.read_csv("benchmark_results_SA_homberger.csv")
df_edf = pd.read_csv("benchmark_results_EDF_solomon.csv")
df_edf_homberger = pd.read_csv("benchmark_results_EDF_homberger.csv")


edf_mean = df_edf["total_distance"].mean()
edf_std  = df_edf["total_distance"].std()

print("EDF Solomon (total_distance) is normal, so we use mean & std:")
print(f"  Mean = {edf_mean:.4f}")
print(f"  Std  = {edf_std:.4f}\n")


solomon_median = df_solomon["total_distance"].median()
solomon_mad = stats.median_abs_deviation(df_solomon["total_distance"], scale='normal')

print("SA Solomon (total_distance) not normal, so we use median & MAD:")
print(f"  Median = {solomon_median:.4f}")
print(f"  MAD    = {solomon_mad:.4f}\n")


homberger_median = df_homberger["total_distance"].median()
homberger_mad    = stats.median_abs_deviation(df_homberger["total_distance"], scale='normal')

print("SA Homberger (total_distance) not normal, so we use median & MAD:")
print(f"  Median = {homberger_median:.4f}")
print(f"  MAD    = {homberger_mad:.4f}\n")


edf_homberger_mean = df_edf_homberger["total_distance"].mean()
edf_homberger_std  = df_edf_homberger["total_distance"].std()

print("EDF Homberger (total_distance) is normal, so we use mean & std:")
print(f"  Mean = {edf_homberger_mean:.4f}")
print(f"  Std  = {edf_homberger_std:.4f}\n")
