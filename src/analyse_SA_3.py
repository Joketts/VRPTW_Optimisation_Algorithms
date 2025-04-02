import pandas as pd
import scipy.stats as stats
import numpy as np

# Load the data (adjust file paths if necessary)
df_solomon = pd.read_csv("benchmark_results_SA_solomon.csv")
df_homberger = pd.read_csv("benchmark_results_SA_homberger.csv")
df_edf = pd.read_csv("benchmark_results_EDF_solomon.csv")

# -------------------------------------------------------------------
# (1) EDF Solomon => Normal => mean & standard deviation
# -------------------------------------------------------------------
edf_mean = df_edf["total_distance"].mean()
edf_std  = df_edf["total_distance"].std()  # pandas std is sample standard deviation by default

print("EDF Solomon (total_distance) is normal, so we use mean & std:")
print(f"  Mean = {edf_mean:.4f}")
print(f"  Std  = {edf_std:.4f}\n")

# -------------------------------------------------------------------
# (2) SA Solomon => Not Normal => median & MAD
# -------------------------------------------------------------------
solomon_median = df_solomon["total_distance"].median()
# For median absolute deviation, we can use scipy.stats.median_abs_deviation
# but note: in older versions of SciPy, you may not have it. If so, use statsmodels.robust.mad.
solomon_mad = stats.median_abs_deviation(df_solomon["total_distance"], scale='normal')

print("SA Solomon (total_distance) not normal, so we use median & MAD:")
print(f"  Median = {solomon_median:.4f}")
print(f"  MAD    = {solomon_mad:.4f}\n")

# -------------------------------------------------------------------
# (3) SA Homberger => Not Normal => median & MAD
# -------------------------------------------------------------------
homberger_median = df_homberger["total_distance"].median()
homberger_mad    = stats.median_abs_deviation(df_homberger["total_distance"], scale='normal')

print("SA Homberger (total_distance) not normal, so we use median & MAD:")
print(f"  Median = {homberger_median:.4f}")
print(f"  MAD    = {homberger_mad:.4f}\n")
