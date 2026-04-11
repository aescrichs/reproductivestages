"""
Predictive modeling of hormonal biomarkers from brain dynamics
Brain dynamics → FSH and estradiol prediction (HCP-A women)

Pipeline: StandardScaler → SelectKBest(k=10, f_regression) → XGBoost
Cross-validation: 10-fold nested CV (n_iter=20 inner RandomizedSearchCV)
Statistics: Pearson R with FDR correction (Benjamini-Hochberg)
Figure: scatter observed vs predicted (Nature Aging style)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr, linregress
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor

# =============================================================================
# CONFIGURATION
# =============================================================================

K_FEATURES   = 10
RANDOM_STATE = 42

TARGETS = {
    'FSH':       'fsh',
    'Estradiol': 'festr',
}

COLORS = {
    'FSH':       'darkviolet',
    'Estradiol': 'magenta',
}

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        8,
    'axes.linewidth':   0.8,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'xtick.major.width':0.8,
    'ytick.major.width':0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction':  'out',
    'ytick.direction':  'out',
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("Loading data...")

mat_data = loadmat('ignitionmeta.mat')
X_all    = mat_data['meta'].astype(float)    # node-metastability (subjects x 100)

info     = pd.read_csv('data_women.csv')

print(f"Subjects: {X_all.shape[0]}, Nodes: {X_all.shape[1]}")

# =============================================================================
# 2. NESTED CV PER HORMONAL TARGET
# =============================================================================

all_results = []
predictions = {}

for target_name, col_name in TARGETS.items():
    print(f"\n  → {target_name} (column: {col_name})...")

    y_all = info[col_name].values.astype(float)

    # Valid subjects (no NaN in target or features)
    valid_mask = ~np.isnan(y_all) & ~np.isnan(X_all).any(axis=1)
    X_clean    = X_all[valid_mask]
    y_clean    = y_all[valid_mask]

    print(f"     Valid subjects: {len(y_clean)}")

    kf_outer  = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = np.zeros(len(y_clean))

    for train_idx, test_idx in kf_outer.split(X_clean):
        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
        y_train         = y_clean[train_idx]

        pipeline = Pipeline([
            ('scaler',    StandardScaler()),
            ('selector',  SelectKBest(f_regression, k=K_FEATURES)),
            ('regressor', XGBRegressor(
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1))
        ])

        param_dist = {
            'regressor__n_estimators':  [50, 100, 200],
            'regressor__max_depth':     [2, 3, 5],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__reg_alpha':     [0, 0.1],
            'regressor__reg_lambda':    [1, 5],
        }

        search = RandomizedSearchCV(
            pipeline, param_dist,
            n_iter=20, cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        search.fit(X_train, y_train)
        y_pred_cv[test_idx] = search.best_estimator_.predict(X_test)

    r_val, p_val = pearsonr(y_clean, y_pred_cv)
    all_results.append({'Target': target_name, 'R': r_val, 'p': p_val})
    predictions[target_name] = (y_clean, y_pred_cv)

    print(f"     R = {r_val:.3f}  p = {p_val:.4f}")

# =============================================================================
# 3. FDR CORRECTION + SUMMARY
# =============================================================================

results_df          = pd.DataFrame(all_results)
_, p_fdr, _, _      = multipletests(results_df['p'], alpha=0.05, method='fdr_bh')
results_df['p_fdr'] = p_fdr
results_df['sig']   = results_df['p_fdr'].apply(
    lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else '')))

print(f"\n{'='*60}")
print("  FINAL RESULTS (FDR-corrected, Benjamini-Hochberg)")
print(f"{'='*60}")
print(f"  {'Target':<12} {'R':>6}  {'p':>8}  {'p_FDR':>8}  {'':>4}")
print(f"  {'-'*45}")
for _, row in results_df.iterrows():
    print(f"  {row['Target']:<12} {row['R']:>6.3f}  {row['p']:>8.4f}  "
          f"{row['p_fdr']:>8.4f}  {row['sig']:>4}")

results_df.to_csv('hormones_prediction_results.csv', index=False)
print("\nSaved: hormones_prediction_results.csv")

# =============================================================================
# 4. FIGURE — observed vs predicted (Nature Aging style)
# =============================================================================

sig_targets = results_df[results_df['p_fdr'] < 0.05]['Target'].tolist()

if not sig_targets:
    print("\nNo significant predictions after FDR correction.")
else:
    n_plots = len(results_df)   # show all regardless of significance
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 3.54, 3.54))
    if n_plots == 1:
        axes = [axes]

    for ax_idx, (_, row) in enumerate(results_df.iterrows()):
        ax         = axes[ax_idx]
        target     = row['Target']
        color      = COLORS[target]
        y_obs, y_pred = predictions[target]

        ax.scatter(y_obs, y_pred, color=color, alpha=0.45, s=14,
                   linewidths=0, rasterized=True)

        slope, intercept, *_ = linregress(y_obs, y_pred)
        x_line = np.linspace(y_obs.min(), y_obs.max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color='#1a1a1a', linewidth=1.0, zorder=5)

        ax.set_xlabel(f'Observed {target}', fontsize=8)
        ax.set_ylabel('Predicted score', fontsize=8)

        p_fdr = row['p_fdr']
        if p_fdr < 0.001:
            p_str = 'p$_{FDR}$ < 0.001 ***'
        elif p_fdr < 0.01:
            p_str = f'p$_{{FDR}}$ = {p_fdr:.3f} **'
        elif p_fdr < 0.05:
            p_str = 'p$_{FDR}$ < 0.05 *'
        else:
            p_str = f'p$_{{FDR}}$ = {p_fdr:.3f}'

        ax.set_title(f'{target}\nr = {row["R"]:.3f}, {p_str}',
                     fontsize=8, fontweight='bold', pad=6)
        ax.tick_params(labelsize=7)

    for ax_idx, letter in enumerate(['a', 'b']):
        axes[ax_idx].text(-0.18, 1.08, letter,
                          transform=axes[ax_idx].transAxes,
                          fontsize=10, fontweight='bold',
                          va='top', ha='left')

    plt.tight_layout()
    fig.savefig('figure_hormones_prediction.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    fig.savefig('figure_hormones_prediction.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print("Saved: figure_hormones_prediction.pdf / .png")