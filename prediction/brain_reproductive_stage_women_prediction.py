"""
Predictive modeling of reproductive aging stage from brain dynamics
Brain dynamics → reproductive stage prediction (HCP-A women, N=328)

Pipeline: StandardScaler → SelectKBest(k=10, f_regression) → XGBoost
Cross-validation: 10-fold nested CV (n_iter=20 inner RandomizedSearchCV)
Statistics: Pearson R
Figure: scatter observed vs predicted (Nature Aging style)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr, linregress
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

mat_data = loadmat('dataset_matrix.mat')
dataset  = mat_data['dataset_matrix']

X_all = dataset[:, 0:100].astype(float)    # nodes 1-100
y_all = dataset[:, 100].astype(float)      # column 101: reproductive group

# Schaefer region names
schaefer_file = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
try:
    schaefer_info = pd.read_csv(schaefer_file, header=None)
    region_names  = schaefer_info.iloc[:, 1].tolist()
    print(f"Schaefer loaded: {len(region_names)} regions")
except FileNotFoundError:
    region_names = [f'Node_{i+1}' for i in range(100)]
    print("[WARNING] Schaefer CSV not found. Using node indices.")

# Remove NaNs
valid_mask = ~np.isnan(y_all) & ~np.isnan(X_all).any(axis=1)
X_clean    = X_all[valid_mask]
y_clean    = y_all[valid_mask]

print(f"Valid subjects: {len(y_clean)}")

# =============================================================================
# 2. NESTED CV PIPELINE
# =============================================================================

print("\nRunning predictive model (reproductive stage ~ brain dynamics)...")

kf_outer       = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
y_pred_cv      = np.zeros(len(y_clean))
node_sel_count = np.zeros(100)

for fold, (train_idx, test_idx) in enumerate(kf_outer.split(X_clean)):
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

    selector = search.best_estimator_.named_steps['selector']
    node_sel_count += selector.get_support().astype(int)

# =============================================================================
# 3. RESULTS
# =============================================================================

r_val, p_val = pearsonr(y_clean, y_pred_cv)

print(f"\n{'='*55}")
print("  RESULTS: Reproductive stage prediction (women)")
print(f"{'='*55}")
print(f"  R = {r_val:.3f}  (p = {p_val:.4e})")
print(f"{'='*55}")

# Top nodes by fold selection frequency
print("\n  Top stable nodes (selected across folds):")
print(f"  {'-'*50}")
top_idx     = np.argsort(node_sel_count)[::-1][:K_FEATURES]
top_counts  = node_sel_count[top_idx]
top_regions = [region_names[j] for j in top_idx]

results_nodes = []
for rank, (idx, count, name) in enumerate(
        zip(top_idx, top_counts, top_regions), 1):
    print(f"  {rank:<3} {int(count):>2}/10 folds   {name}")
    results_nodes.append({
        'Rank':           rank,
        'Node_index':     int(idx) + 1,
        'Folds_selected': int(count),
        'Region':         name
    })

pd.DataFrame(results_nodes).to_csv(
    'reproductive_stage_top_nodes.csv', index=False)

# Save summary
pd.DataFrame([{'R': round(r_val, 4), 'p': round(p_val, 6)}]).to_csv(
    'reproductive_stage_results.csv', index=False)

print("\nSaved: reproductive_stage_results.csv")
print("Saved: reproductive_stage_top_nodes.csv")

# =============================================================================
# 4. FIGURE — observed vs predicted (Nature Aging style)
# =============================================================================

fig, ax = plt.subplots(figsize=(3.54, 3.54))   # 90mm x 90mm (Nature half-width)

ax.scatter(y_clean, y_pred_cv,
           color='#7F77DD', alpha=0.45, s=14,
           linewidths=0, rasterized=True)

slope, intercept, *_ = linregress(y_clean, y_pred_cv)
x_line = np.linspace(y_clean.min(), y_clean.max(), 200)
ax.plot(x_line, slope * x_line + intercept,
        color='#1a1a1a', linewidth=1.0, zorder=5)

ax.set_xlabel('Observed reproductive stage', fontsize=8)
ax.set_ylabel('Predicted score', fontsize=8)

if p_val < 0.001:
    p_str = 'p < 0.001'
else:
    p_str = f'p = {p_val:.3f}'

ax.set_title(f'Reproductive stage\nr = {r_val:.3f}, {p_str}',
             fontsize=8, fontweight='bold', pad=6)
ax.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig('figure_reproductive_stage_prediction.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
fig.savefig('figure_reproductive_stage_prediction.png',
            dpi=300, bbox_inches='tight')
plt.close(fig)

print("Saved: figure_reproductive_stage_prediction.pdf / .png")
