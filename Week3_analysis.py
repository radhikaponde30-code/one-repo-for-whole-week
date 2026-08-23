"""
Week 3 Task - Statistical Analysis and Hypothesis Testing
Dataset: Car Features and MSRP (cleaned in Week 1)

Primary hypothesis (H1, two-sample test):
  H0: Mean MSRP is equal for automatic- and manual-transmission vehicles.
  H1: Mean MSRP differs between automatic- and manual-transmission vehicles.

Supporting analyses:
  H2 (one-way ANOVA): Mean MSRP differs across vehicle size classes.
  H3 (chi-square test of independence): Vehicle size class and transmission
      type are associated (not independent).
  H4 (simple linear regression / significance of slope): Engine horsepower
      is a significant linear predictor of MSRP.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
PALETTE = {"a": "#2E86AB", "b": "#D64545", "c": "#E8A33D", "d": "#4C956C"}
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

df = pd.read_csv('cars_clean.csv')
ALPHA = 0.05

# Work on a de-outliered set for the mean-based tests to avoid a handful
# of multi-million-dollar exotics dominating variance; full data is used
# for the chi-square test since that only concerns category counts.
core = df[df['msrp'] <= 300000].copy()
core['log_msrp'] = np.log10(core['msrp'])

print("=" * 70)
print("H1: Automatic vs. Manual transmission - mean MSRP")
print("=" * 70)

auto = core.loc[core.transmission_type == 'AUTOMATIC', 'msrp']
manual = core.loc[core.transmission_type == 'MANUAL', 'msrp']

print(f"n(automatic) = {len(auto)}, mean = ${auto.mean():,.0f}, sd = ${auto.std():,.0f}")
print(f"n(manual)    = {len(manual)}, mean = ${manual.mean():,.0f}, sd = ${manual.std():,.0f}")

# Levene's test for equal variances (informs which t-test variant to use)
lev_stat, lev_p = stats.levene(auto, manual)
print(f"\nLevene's test for equal variances: stat={lev_stat:.2f}, p={lev_p:.4g}")

# Welch's t-test (does not assume equal variances) - robust default choice
t_stat, t_p = stats.ttest_ind(auto, manual, equal_var=False)
print(f"Welch's t-test: t={t_stat:.3f}, p={t_p:.4g}")

# 95% CI for the difference in means (Welch-Satterthwaite)
n1, n2 = len(auto), len(manual)
m1, m2 = auto.mean(), manual.mean()
s1, s2 = auto.std(ddof=1), manual.std(ddof=1)
se = np.sqrt(s1**2/n1 + s2**2/n2)
dof = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
tcrit = stats.t.ppf(0.975, dof)
diff = m1 - m2
ci_low, ci_high = diff - tcrit*se, diff + tcrit*se
print(f"Difference in means (auto - manual): ${diff:,.0f}, 95% CI: (${ci_low:,.0f}, ${ci_high:,.0f})")

# Effect size: Cohen's d (pooled sd)
pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
cohens_d = diff / pooled_sd
print(f"Cohen's d: {cohens_d:.3f}")

# Robustness check: MSRP is right-skewed, so also run the non-parametric
# Mann-Whitney U test, which does not assume normal distributions.
u_stat, u_p = stats.mannwhitneyu(auto, manual, alternative='two-sided')
print(f"Mann-Whitney U test (robustness check): U={u_stat:.0f}, p={u_p:.4g}")

print("\n" + "=" * 70)
print("H2: One-way ANOVA - MSRP across vehicle size classes")
print("=" * 70)

groups = [core.loc[core.vehicle_size == g, 'log_msrp'] for g in ['Compact', 'Midsize', 'Large']]
f_stat, anova_p = stats.f_oneway(*groups)
print(f"One-way ANOVA (on log10 MSRP, to address right-skew): F={f_stat:.3f}, p={anova_p:.4g}")

# Manual sum-of-squares breakdown (equivalent to statsmodels anova_lm output)
grand_mean = core['log_msrp'].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
ss_total = ss_between + ss_within
df_between = len(groups) - 1
df_within = len(core) - len(groups)
eta_sq = ss_between / ss_total
print(f"SS_between={ss_between:.3f} (df={df_between}), SS_within={ss_within:.3f} (df={df_within})")
print(f"Eta-squared (effect size): {eta_sq:.3f}")

# Tukey HSD post-hoc test (pairwise group comparisons)
tukey = stats.tukey_hsd(*groups)
print("\nTukey HSD post-hoc test (Compact=0, Midsize=1, Large=2):")
print(tukey)

print("\n" + "=" * 70)
print("H3: Chi-square test - Vehicle Size x Transmission Type")
print("=" * 70)

top_trans = df['transmission_type'].value_counts().head(3).index  # sufficiently large cells
ct = pd.crosstab(df.loc[df.transmission_type.isin(top_trans), 'vehicle_size'],
                  df.loc[df.transmission_type.isin(top_trans), 'transmission_type'])
print("Contingency table (counts):\n", ct)
chi2, chi_p, dof_chi, expected = stats.chi2_contingency(ct)
print(f"\nChi-square test: chi2={chi2:.2f}, dof={dof_chi}, p={chi_p:.4g}")
n_total = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n_total * (min(ct.shape) - 1)))
print(f"Cramer's V (effect size): {cramers_v:.3f}")

print("\n" + "=" * 70)
print("H4: Simple linear regression - Engine HP predicting MSRP")
print("=" * 70)

reg_data = core.dropna(subset=['engine_hp', 'msrp'])
lin = stats.linregress(reg_data['engine_hp'], reg_data['log_msrp'])
print(f"Slope: {lin.slope:.6f}, Intercept: {lin.intercept:.4f}")
print(f"R-value: {lin.rvalue:.4f}, R-squared: {lin.rvalue**2:.4f}")
print(f"p-value (slope != 0): {lin.pvalue:.4g}")
print(f"Std error of slope: {lin.stderr:.6f}")
n_reg = len(reg_data)
tcrit_reg = stats.t.ppf(0.975, n_reg - 2)
slope_ci_low = lin.slope - tcrit_reg * lin.stderr
slope_ci_high = lin.slope + tcrit_reg * lin.stderr
print(f"95% CI for slope: ({slope_ci_low:.6f}, {slope_ci_high:.6f})")

# -----------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------

# Viz 1: Boxplot + strip comparison, Automatic vs Manual
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=core[core.transmission_type.isin(['AUTOMATIC', 'MANUAL'])],
            x='transmission_type', y='msrp', hue='transmission_type',
            palette={'AUTOMATIC': PALETTE['a'], 'MANUAL': PALETTE['b']}, legend=False, ax=ax, showfliers=False)
ax.set_title("MSRP by Transmission Type (H1)")
ax.set_xlabel("Transmission Type")
ax.set_ylabel("MSRP (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
sns.despine()
plt.tight_layout()
plt.savefig('viz1_transmission_boxplot.png', dpi=150)
plt.close()

# Viz 2: QQ plot for log(MSRP) to visually assess normality assumption
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
stats.probplot(auto, dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot: Automatic MSRP (raw)")
stats.probplot(np.log10(auto), dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: Automatic log10(MSRP)")
plt.tight_layout()
plt.savefig('viz2_qqplot.png', dpi=150)
plt.close()

# Viz 3: ANOVA - distribution by vehicle size (violin) with means marked
fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(data=core, x='vehicle_size', y='log_msrp', hue='vehicle_size',
               order=['Compact', 'Midsize', 'Large'], palette='Set2', ax=ax, cut=0, legend=False)
means = core.groupby('vehicle_size')['log_msrp'].mean().reindex(['Compact', 'Midsize', 'Large'])
ax.scatter(range(3), means.values, color='black', marker='D', s=60, zorder=5, label='Group Mean')
ax.set_title("log10(MSRP) by Vehicle Size (H2 - ANOVA)")
ax.set_xlabel("Vehicle Size")
ax.set_ylabel("log10(MSRP)")
ax.legend()
sns.despine()
plt.tight_layout()
plt.savefig('viz3_anova_violin.png', dpi=150)
plt.close()

# Viz 4: Chi-square - stacked proportion bar chart
ct_prop = ct.div(ct.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(8, 6))
ct_prop.plot(kind='bar', stacked=True, ax=ax,
             color=[PALETTE['a'], PALETTE['b'], PALETTE['c']], edgecolor='white')
ax.set_title("Transmission Type Share by Vehicle Size (H3 - Chi-square)")
ax.set_xlabel("Vehicle Size")
ax.set_ylabel("Proportion of Listings")
ax.legend(title="Transmission Type", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=0)
sns.despine()
plt.tight_layout()
plt.savefig('viz4_chisquare_stacked.png', dpi=150)
plt.close()

# Viz 5: Regression - scatter + fitted line, HP vs log(MSRP)
fig, ax = plt.subplots(figsize=(8, 6))
sample = reg_data.sample(n=min(2500, len(reg_data)), random_state=42)
ax.scatter(sample['engine_hp'], sample['log_msrp'], alpha=0.25, s=15, color=PALETTE['a'])
x_line = np.linspace(reg_data['engine_hp'].min(), reg_data['engine_hp'].max(), 100)
y_line = lin.intercept + lin.slope * x_line
ax.plot(x_line, y_line, color=PALETTE['b'], linewidth=2.5, label='OLS fit')
ax.set_title(f"Engine HP vs. log10(MSRP) — OLS Fit (H4, R² = {lin.rvalue**2:.2f})")
ax.set_xlabel("Engine Horsepower")
ax.set_ylabel("log10(MSRP)")
ax.legend()
sns.despine()
plt.tight_layout()
plt.savefig('viz5_regression_fit.png', dpi=150)
plt.close()

print("\nAll visualizations saved.")

# Save key numbers for the report
results = {
    'auto_mean': auto.mean(), 'manual_mean': manual.mean(),
    'diff': diff, 'ci_low': ci_low, 'ci_high': ci_high,
    't_stat': t_stat, 't_p': t_p, 'cohens_d': cohens_d,
    'levene_p': lev_p, 'u_stat': u_stat, 'u_p': u_p,
    'anova_f': f_stat, 'anova_p': anova_p, 'eta_sq': eta_sq,
    'chi2': chi2, 'chi_p': chi_p, 'cramers_v': cramers_v, 'chi_dof': dof_chi,
    'reg_slope': lin.slope, 'reg_intercept': lin.intercept,
    'reg_r2': lin.rvalue**2, 'reg_p': lin.pvalue,
    'reg_se': lin.stderr,
    'reg_ci_low': slope_ci_low, 'reg_ci_high': slope_ci_high,
    'ss_between': ss_between, 'ss_within': ss_within, 'df_between': df_between, 'df_within': df_within,
}
import json
with open('results.json', 'w') as f:
    json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
print(json.dumps({k: round(float(v), 4) for k, v in results.items()}, indent=2))
