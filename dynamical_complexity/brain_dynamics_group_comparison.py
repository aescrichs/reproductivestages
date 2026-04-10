"""
Group-level whole-brain metastability comparison across reproductive stages
Permutation test (10,000 iterations) + Cliff's Delta + FDR correction

Input:  ignitionmeta.mat, data_women.csv
Output: wholebrain_metastability_results.csv
        wholebrain_metastability.png / .pdf
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

mat_contents = scipy.io.loadmat('ignitionmeta.mat')
meta         = mat_contents['meta']
data         = pd.read_csv('data_women.csv')
groups       = data['groups'].values

group_labels = [
    'Reproductive', 'Late Reproductive', 'Perimenopause',
    'Early Postmenopause', 'Late Postmenopause'
]
palette_all  = ["#9afdff", "#fa00fa", "#FF007F", "#a200ff", "#00aba9"]

group_indices = [(groups == g) for g in range(1, 6)]
group_data    = [meta[idx] for idx in group_indices]
group_means   = [np.mean(g, axis=0) for g in group_data]

# =============================================================================
# 2. PERMUTATION TEST + CLIFF'S DELTA
# =============================================================================

comparisons  = [(i, j) for i in range(5) for j in range(i + 1, 5)]
p_values     = []
cliff_deltas = []

for (i, j) in comparisons:
    delta = cliff_delta(group_means[i], group_means[j])
    cliff_deltas.append(delta)

    observed = np.abs(np.mean(group_means[i]) - np.mean(group_means[j]))
    combined = np.concatenate([group_means[i], group_means[j]])
    n1       = len(group_means[i])
    null_dist = []
    for _ in range(10000):
        np.random.shuffle(combined)
        null_dist.append(np.abs(
            np.mean(combined[:n1]) - np.mean(combined[n1:])))
    p_values.append(np.mean(np.array(null_dist) >= observed))

_, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

significant = [
    (comparisons[i][0], comparisons[i][1], p_fdr[i])
    for i in range(len(p_fdr)) if p_fdr[i] < 0.05
]

# Print + save
print("\n--- Whole-brain metastability results ---")
results = []
for idx, (g1, g2) in enumerate(comparisons):
    print(f"{group_labels[g1]} vs {group_labels[g2]}: "
          f"p_FDR = {p_fdr[idx]:.5f}, Cliff's Delta = {cliff_deltas[idx]:.3f}")
    results.append({
        'group1':      group_labels[g1],
        'group2':      group_labels[g2],
        'p_uncorr':    p_values[idx],
        'p_fdr':       p_fdr[idx],
        'cliff_delta': cliff_deltas[idx],
        'significant': p_fdr[idx] < 0.05
    })

pd.DataFrame(results).to_csv(
    'wholebrain_metastability_results.csv', index=False)
print("Saved: wholebrain_metastability_results.csv")

# =============================================================================
# 3. FIGURE
# =============================================================================

def add_stat_annotation(ax, x1, x2, y, p_val, offset=0.001):
    if p_val < 0.001:   text = '***'
    elif p_val < 0.01:  text = '**'
    elif p_val < 0.05:  text = '*'
    else: return y
    ax.plot([x1, x1, x2, x2],
            [y, y + offset, y + offset, y], lw=1.5, color='black')
    ax.text((x1 + x2) / 2, y + offset, text,
            ha='center', va='bottom', fontsize=16)
    return y + offset * 2.5

fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(data=group_means,
            palette=sns.color_palette(palette_all), ax=ax)

ax.set_ylabel('Node-metastability', fontsize=10)
ax.set_xticks(np.arange(len(group_labels)))
ax.set_xticklabels(group_labels, rotation=20, fontsize=9)

y_max = max([np.max(g) for g in group_means]) + 0.002
for (x1, x2, p_val) in significant:
    y_max = add_stat_annotation(ax, x1, x2, y_max, p_val)

sns.despine()
plt.tight_layout()
fig.savefig('wholebrain_metastability.png', dpi=300, bbox_inches='tight')
fig.savefig('wholebrain_metastability.pdf', dpi=300,
            bbox_inches='tight', format='pdf')
plt.close(fig)
print("Saved: wholebrain_metastability.png / .pdf")
