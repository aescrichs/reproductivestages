"""
Whole-brain metastability comparison: women vs men across reproductive stages
Permutation test (10,000 iterations) + Cliff's Delta + FDR correction

Input:  ignitionmeta.mat, data_women.csv, metamen.mat, data_men.csv
Output: wholebrain_metastability_sex_results.csv
        wholebrain_metastability_sex.png / .pdf
"""

import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

np.random.seed(42)

sns.set_context("paper")
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.sans-serif':['Arial'],
    'pdf.fonttype':   42,
    'ps.fonttype':    42,
})

# =============================================================================
# CLIFF'S DELTA
# =============================================================================

def cliff_delta(x, y):
    diffs = np.subtract.outer(x, y)
    return np.mean(np.sign(diffs))

# =============================================================================
# 1. LOAD DATA
# =============================================================================

# Women
meta_w   = scipy.io.loadmat('ignitionmeta.mat')['meta']
data_w   = pd.read_csv('data_women.csv')
groups_w = data_w['groups'].values

# Men
meta_m   = scipy.io.loadmat('metamen.mat')['metamen']
data_m   = pd.read_csv('data_men.csv')
groups_m = data_m['reproductivematch'].values

labels = [
    'Reproductive', 'Late Reproductive', 'Perimenopause',
    'Early Postmenopause', 'Late Postmenopause'
]
female_colors = ["#9afdff", "#fa00fa", "#FF007F", "#a200ff", "#00aba9"]
male_colors   = ["#e0e0e0", "#bfbfbf", "#9f9f9f", "#7f7f7f", "#5f5f5f"]

# Group means per sex
data_groups  = [np.mean(meta_w[groups_w == g], axis=0) for g in range(1, 6)]
mdata_groups = [np.mean(meta_m[groups_m == g], axis=0) for g in range(1, 6)]

# =============================================================================
# 2. PERMUTATION TESTS + CLIFF'S DELTA
# =============================================================================

# Women vs Men (same stage)
p_values_wm  = []
cliff_wm     = []
comparisons_wm = [(i, i) for i in range(5)]

for (i, j) in comparisons_wm:
    delta = cliff_delta(data_groups[i], mdata_groups[j])
    cliff_wm.append(delta)

    observed = np.abs(np.mean(data_groups[i]) - np.mean(mdata_groups[j]))
    combined = np.concatenate([data_groups[i], mdata_groups[j]])
    n1       = len(data_groups[i])
    null_dist = []
    for _ in range(10000):
        perm = np.random.permutation(combined)
        null_dist.append(np.abs(np.mean(perm[:n1]) - np.mean(perm[n1:])))
    p_values_wm.append(np.mean(np.array(null_dist) >= observed))

_, p_fdr_wm, _, _ = multipletests(p_values_wm, alpha=0.05, method='fdr_bh')
sig_wm = [(i, i, p_fdr_wm[i]) for i in range(5) if p_fdr_wm[i] < 0.1]

# Men-only consecutive comparisons
comparisons_m = [(3, 4), (2, 3), (1, 2), (0, 1)]
p_values_m    = []
cliff_m       = []

for (i, j) in comparisons_m:
    delta = cliff_delta(mdata_groups[i], mdata_groups[j])
    cliff_m.append(delta)

    observed = np.abs(np.mean(mdata_groups[i]) - np.mean(mdata_groups[j]))
    combined = np.concatenate([mdata_groups[i], mdata_groups[j]])
    n1       = len(mdata_groups[i])
    null_dist = []
    for _ in range(10000):
        perm = np.random.permutation(combined)
        null_dist.append(np.abs(np.mean(perm[:n1]) - np.mean(perm[n1:])))
    p_values_m.append(np.mean(np.array(null_dist) >= observed))

_, p_fdr_m, _, _ = multipletests(p_values_m, alpha=0.05, method='fdr_bh')
sig_m = [(comparisons_m[i][0], comparisons_m[i][1], p_fdr_m[i])
         for i in range(len(p_fdr_m)) if p_fdr_m[i] < 0.05]

# Print results
print("\n--- Women vs Men (same stage) ---")
results = []
for idx, (i, j) in enumerate(comparisons_wm):
    print(f"{labels[i]}: p_FDR = {p_fdr_wm[idx]:.5f}, "
          f"Cliff's Delta = {cliff_wm[idx]:.3f}")
    results.append({
        'comparison': f"Women vs Men — {labels[i]}",
        'p_uncorr':   p_values_wm[idx],
        'p_fdr':      p_fdr_wm[idx],
        'cliff_delta':cliff_wm[idx]
    })

print("\n--- Men consecutive stages ---")
for idx, (i, j) in enumerate(comparisons_m):
    print(f"{labels[i]} vs {labels[j]}: p_FDR = {p_fdr_m[idx]:.5f}, "
          f"Cliff's Delta = {cliff_m[idx]:.3f}")
    results.append({
        'comparison': f"Men — {labels[i]} vs {labels[j]}",
        'p_uncorr':   p_values_m[idx],
        'p_fdr':      p_fdr_m[idx],
        'cliff_delta':cliff_m[idx]
    })

pd.DataFrame(results).to_csv(
    'wholebrain_metastability_sex_results.csv', index=False)
print("Saved: wholebrain_metastability_sex_results.csv")

# =============================================================================
# 3. FIGURE
# =============================================================================

bar_colors  = []
group_labels_plot = []
all_data    = []
for i in range(5):
    all_data.append(data_groups[i])
    group_labels_plot.append(f'{labels[i]}\nWomen')
    bar_colors.append(female_colors[i])
    all_data.append(mdata_groups[i])
    group_labels_plot.append(f'{labels[i]}\nMen')
    bar_colors.append(male_colors[i])

def add_stat_annotation(ax, x1, x2, y, p_val, offset=0.001):
    color = 'black'
    if p_val < 0.001:       text = '***'
    elif p_val < 0.01:      text = '**'
    elif p_val < 0.05:      text = '*'
    elif p_val < 0.1:       text, color = '*', 'red'
    else: return y
    ax.plot([x1, x1, x2, x2],
            [y, y + offset, y + offset, y], lw=1.2, color='black')
    ax.text((x1 + x2) / 2, y + offset, text,
            ha='center', va='bottom', fontsize=18, color=color)
    return y + offset * 4

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=all_data, palette=bar_colors, ax=ax)

ax.set_ylabel('Node-metastability', fontsize=10)
ax.set_xticks(np.arange(len(group_labels_plot)))
ax.set_xticklabels(group_labels_plot, rotation=30, fontsize=9)

for i in range(1, 5):
    ax.axvline(x=i * 2 - 0.5, linestyle='--', color='gray', alpha=0.5)

y_max = max([np.max(g) for g in all_data])

for (i, j, p_val) in sig_wm:
    y_max = add_stat_annotation(ax, i * 2, i * 2 + 1, y_max, p_val)

for (i, j, p_val) in sig_m:
    y_max = add_stat_annotation(ax, i * 2 + 1, j * 2 + 1, y_max, p_val)

sns.despine()
plt.tight_layout()
fig.savefig('wholebrain_metastability_sex.png', dpi=300, bbox_inches='tight')
fig.savefig('wholebrain_metastability_sex.pdf', dpi=300,
            bbox_inches='tight', format='pdf')
plt.close(fig)
print("Saved: wholebrain_metastability_sex.png / .pdf")
