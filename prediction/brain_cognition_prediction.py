"""
Machine learning-based predictive modeling of cognition from brain dynamics
Brain dynamics → cognitive prediction (HCP-A women)

Pipeline: StandardScaler → SelectKBest(k=10, f_regression) → XGBoost
Cross-validation: 10-fold nested CV (n_iter=20 inner RandomizedSearchCV)
Statistics: Pearson R with FDR correction (Benjamini-Hochberg)
Figure: scatter observed vs predicted for significant tests (Nature Aging style)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

test_names = [
    'Language_Vocabulary',
    'Episodic_Memory',
    'Processing_Speed',
    'Language_Reading',
    'Working_Memory',
    'Inhibition',
    'Cognitive_Flexibility'
]

# Nature-style colors
COLORS = {
    'Language_Vocabulary':  '#E87461',   # coral
    'Episodic_Memory':      '#C084C8',   # purple
    'Working_Memory':       '#5AABBB',   # teal
    'Cognitive_Flexibility':'#6BAF6B',   # green
    'Processing_Speed':     '#F4C775',   # amber
    'Language_Reading':     '#B4B2A9',   # gray
    'Inhibition':           '#85B7EB',   # blue
}

LABELS = {
    'Language_Vocabulary':  'Language vocabulary',
    'Episodic_Memory':      'Episodic memory',
    'Working_Memory':       'Working memory',
    'Cognitive_Flexibility':'Cognitive flexibility',
    'Processing_Speed':     'Processing speed',
    'Language_Reading':     'Language reading',
    'Inhibition':           'Inhibition',
}

# =============================================================================
# 1. LOAD DATA
# =============================================================================

mat_data = loadmat('dataset_matrix.mat')
dataset  = mat_data['dataset_matrix']

X_all = dataset[:, 0:100].astype(float)    # nodes 1-100
Y_all = dataset[:, 113:120].astype(float)  # cognitive tests age-adjusted 114-120

# =============================================================================
# 2. LOAD SCHAEFER REGION NAMES
# =============================================================================

schaefer_file = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
try:
    schaefer_info = pd.read_csv(schaefer_file, header=None)
    region_names  = schaefer_info.iloc[:, 1].tolist()
    print(f"Schaefer loaded: {len(region_names)} regions\n")
except FileNotFoundError:
    region_names = [f'Node_{j+1}' for j in range(100)]
    print("[WARNING] Schaefer CSV not found. Using node indices.\n")

# =============================================================================
# 3. MAIN ANALYSIS — nested CV per cognitive domain
# =============================================================================

all_results   = []
all_top_nodes = []
predictions   = {}

for i, test_name in enumerate(test_names):
    y = Y_all[:, i]

    valid_mask = ~np.isnan(y)
    X_clean    = X_all[valid_mask]
    y_clean    = y[valid_mask]

    kf_outer       = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv      = np.zeros(len(y_clean))
    node_sel_count = np.zeros(100)

    for train_idx, test_idx in kf_outer.split(X_clean):
        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
        y_train         = y_clean[train_idx]

        pipeline = Pipeline([
            ('scaler',    StandardScaler()),
            ('selector',  SelectKBest(f_regression, k=K_FEATURES)),
            ('regressor', XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1))
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

        selector       = search.best_estimator_.named_steps['selector']
        node_sel_count += selector.get_support().astype(int)

    r_val, p_val = pearsonr(y_clean, y_pred_cv)
    all_results.append({'Test': test_name, 'R': r_val, 'p': p_val})
    predictions[test_name] = (y_clean, y_pred_cv)

    # Top nodes by fold frequency
    top_idx     = np.argsort(node_sel_count)[::-1][:K_FEATURES]
    top_counts  = node_sel_count[top_idx]
    top_regions = [region_names[j] for j in top_idx]

    print(f"\n{'='*65}")
    print(f"  {test_name}  |  R = {r_val:.3f}  p = {p_val:.4f}")
    print(f"{'='*65}")
    print(f"  {'Rank':<5} {'Folds':>6}   Region")
    print(f"  {'-'*55}")
    for rank, (idx, count, name) in enumerate(zip(top_idx, top_counts, top_regions), 1):
        print(f"  {rank:<5} {int(count):>4}/10    {name}")

    for rank, (idx, count, name) in enumerate(zip(top_idx, top_counts, top_regions), 1):
        all_top_nodes.append({
            'Test':           test_name,
            'R':              round(r_val, 4),
            'p':              round(p_val, 4),
            'Rank':           rank,
            'Node_index':     int(idx) + 1,
            'Folds_selected': int(count),
            'Region':         name
        })

# =============================================================================
# 4. FDR CORRECTION + SUMMARY TABLE
# =============================================================================

results_df = pd.DataFrame(all_results)
_, p_fdr, _, _ = multipletests(results_df['p'], alpha=0.05, method='fdr_bh')
results_df['p_fdr'] = p_fdr
results_df['sig']   = results_df['p_fdr'].apply(
    lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ''))
)

print("\n\n" + "="*75)
print("  FINAL RESULTS (FDR-corrected, Benjamini-Hochberg)")
print("="*75)
print(f"  {'Test':<25} {'R':>6}  {'p':>8}  {'p_FDR':>8}  {'':>4}")
print(f"  {'-'*60}")
for _, row in results_df.iterrows():
    print(f"  {row['Test']:<25} {row['R']:>6.3f}  {row['p']:>8.4f}  "
          f"{row['p_fdr']:>8.4f}  {row['sig']:>4}")

results_df.to_csv('cognitive_prediction_results.csv', index=False)
pd.DataFrame(all_top_nodes).to_csv('cognitive_prediction_top_regions.csv', index=False)
print("\nSaved: cognitive_prediction_results.csv")
print("Saved: cognitive_prediction_top_regions.csv")

# =============================================================================
# 5. FIGURE — observed vs predicted 
#    Significant tests selected automatically, ordered by R descending
# =============================================================================

# Automatically select significant tests after FDR, ordered by R descending
results_sig = results_df[results_df['p_fdr'] < 0.05].sort_values('R', ascending=False)
PLOT_TESTS  = results_sig['Test'].tolist()

n_plots = len(PLOT_TESTS)
n_cols  = 2
n_rows  = int(np.ceil(n_plots / n_cols))

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

fig = plt.figure(figsize=(7.08, n_rows * 3.54))  # 180mm wide, adaptive height
gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.42, wspace=0.38)

for ax_idx, test_name in enumerate(PLOT_TESTS):
    ax    = fig.add_subplot(gs[ax_idx // n_cols, ax_idx % n_cols])
    color = COLORS[test_name]
    label = LABELS[test_name]

    row   = results_df[results_df['Test'] == test_name].iloc[0]
    r_val = row['R']
    p_fdr = row['p_fdr']

    y_obs, y_pred = predictions[test_name]

    # Scatter
    ax.scatter(y_obs, y_pred,
               color=color, alpha=0.45, s=14,
               linewidths=0, rasterized=True)

    # Regression line
    slope, intercept, *_ = linregress(y_obs, y_pred)
    x_line = np.linspace(y_obs.min(), y_obs.max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color='#1a1a1a', linewidth=1.0, zorder=5)

    ax.set_xlabel('Observed score', fontsize=8)
    ax.set_ylabel('Predicted score', fontsize=8)

    # p_FDR formatting with stars
    if p_fdr < 0.001:
        p_str = 'p$_{FDR}$ < 0.001 ***'
    elif p_fdr < 0.01:
        p_str = f'p$_{{FDR}}$ = {p_fdr:.3f} **'
    else:
        p_str = 'p$_{FDR}$ < 0.05 *'

    ax.set_title(f'{label}\nr = {r_val:.3f}, {p_str}',
                 fontsize=8, fontweight='bold', pad=6)

    ax.tick_params(labelsize=7)

# Panel letters
letters = 'abcdefgh'
for ax_idx in range(n_plots):
    ax = fig.axes[ax_idx]
    ax.text(-0.18, 1.08, letters[ax_idx], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

# Hide unused subplots if n_plots is odd
if n_plots % 2 != 0:
    fig.axes[-1].set_visible(False)

fig.savefig('figure3.pdf', dpi=300, bbox_inches='tight', format='pdf')
fig.savefig('figure3.png', dpi=300, bbox_inches='tight', format='png')
plt.show()

print("\nSaved: figure3.pdf")
print("Saved: figure3.png")
print(f"Figure contains {n_plots} significant tests (p_FDR < 0.05), ordered by R descending.")
