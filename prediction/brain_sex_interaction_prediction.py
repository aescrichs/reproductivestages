"""
Sex-specific predictive modeling of reproductive aging from brain dynamics
Brain dynamics × sex interaction → reproductive stage prediction (HCP-A)

Pipeline: StandardScaler → SelectKBest(k=10, f_regression) → XGBoost
Cross-validation: 10-fold nested CV (n_iter=20 inner RandomizedSearchCV)
Statistics: Pearson R (global, women, men)
Figure: scatter observed vs predicted, color-coded by sex (Nature Aging style)

The interaction term (node × sex) is a deterministic transformation
applied before the CV loop and does not introduce data leakage.

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless-safe
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
    'font.size':        10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("Loading data...")

# Schaefer region names
schaefer_file = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
try:
    schaefer_info = pd.read_csv(schaefer_file, header=None)
    region_names  = schaefer_info.iloc[:, 1].tolist()
    print(f"Schaefer loaded: {len(region_names)} regions")
except FileNotFoundError:
    region_names = [f'Node_{i+1}' for i in range(100)]
    print("[WARNING] Schaefer CSV not found. Using node indices.")

# Women
meta_w      = loadmat('ignitionmeta.mat')['meta']
info_w      = pd.read_csv('data_women.csv')
X_w         = meta_w
y_w         = info_w['groups'].values
sex_w       = np.ones(len(y_w))   # 1 = Women

# Men
meta_m      = loadmat('metamen.mat')['metamen']
info_m      = pd.read_csv('data_men.csv')
X_m         = meta_m
y_m         = info_m['reproductivematch'].values #reproductivematch11
sex_m       = np.zeros(len(y_m))  # 0 = Men

# Concatenate
X_nodes = np.vstack([X_w, X_m])
y_all   = np.concatenate([y_w, y_m])
sex_all = np.concatenate([sex_w, sex_m])

# Remove NaNs
valid_rows = ~np.isnan(X_nodes).any(axis=1) & ~np.isnan(y_all)
X_nodes = X_nodes[valid_rows]
y_all   = y_all[valid_rows]
sex_all = sex_all[valid_rows]

# =============================================================================
# 2. BUILD FEATURE MATRIX
# =============================================================================

# Interaction term is a deterministic transformation — no data leakage
X_interaction = X_nodes * sex_all[:, None]
X_comb        = np.hstack([X_nodes, sex_all[:, None], X_interaction])

feature_names = np.array(
    region_names + ['Sex'] + [f"{n}_x_Sex" for n in region_names])

# =============================================================================
# 3. NESTED CV PIPELINE
# =============================================================================

print("\nRunning predictive model (interaction: nodes × sex)...")

kf_outer       = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
y_pred_cv      = np.zeros(len(y_all))
node_sel_count = np.zeros(X_comb.shape[1])

for fold, (train_idx, test_idx) in enumerate(kf_outer.split(X_comb)):
    X_train, X_test = X_comb[train_idx], X_comb[test_idx]
    y_train         = y_all[train_idx]

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
# 4. RESULTS
# =============================================================================

mask_w = (sex_all == 1)
mask_m = (sex_all == 0)

r_total, p_total = pearsonr(y_all, y_pred_cv)
r_w, p_w         = pearsonr(y_all[mask_w], y_pred_cv[mask_w])
r_m, p_m         = pearsonr(y_all[mask_m], y_pred_cv[mask_m])

print(f"\n{'='*55}")
print("  RESULTS: Interaction model (nodes × sex)")
print(f"{'='*55}")
print(f"  Global : R = {r_total:.3f} (p = {p_total:.4e})")
print(f"  Women  : R = {r_w:.3f} (p = {p_w:.4e})")
print(f"  Men    : R = {r_m:.3f} (p = {p_m:.4e})")
print(f"{'='*55}")

# Top features by fold selection frequency
print("\n  Top stable features (selected across folds):")
print(f"  {'-'*55}")
top_idx = np.argsort(node_sel_count)[::-1]

interaction_results = []
for i in range(25):
    idx   = top_idx[i]
    count = node_sel_count[idx]
    if count == 0:
        break
    feat_type = ('Interaction' if '_x_Sex' in feature_names[idx]
                 else 'Covariate' if feature_names[idx] == 'Sex'
                 else 'Main Effect')
    print(f"  {int(count):>2}/10 folds | {feat_type:<12} | {feature_names[idx]}")
    interaction_results.append({
        'Feature':         feature_names[idx],
        'Type':            feat_type,
        'Folds_Selected':  int(count)
    })

pd.DataFrame(interaction_results).to_csv(
    'sex_interaction_top_features.csv', index=False)
print("\nSaved: sex_interaction_top_features.csv")

# =============================================================================
# 5. FIGURE — observed vs predicted, color-coded by sex 
# =============================================================================

fig, ax = plt.subplots(figsize=(5, 5))

# Women (coral)
ax.scatter(y_all[mask_w], y_pred_cv[mask_w],
           color='#E87461', alpha=0.6, s=20,
           label='Women', linewidths=0, rasterized=True)
slope_w, intercept_w, *_ = linregress(y_all[mask_w], y_pred_cv[mask_w])
x_line_w = np.linspace(y_all[mask_w].min(), y_all[mask_w].max(), 100)
ax.plot(x_line_w, slope_w * x_line_w + intercept_w,
        color='#E87461', linewidth=1.5)

# Men (teal)
ax.scatter(y_all[mask_m], y_pred_cv[mask_m],
           color='#5AABBB', alpha=0.6, s=20,
           label='Men', linewidths=0, rasterized=True)
slope_m, intercept_m, *_ = linregress(y_all[mask_m], y_pred_cv[mask_m])
x_line_m = np.linspace(y_all[mask_m].min(), y_all[mask_m].max(), 100)
ax.plot(x_line_m, slope_m * x_line_m + intercept_m,
        color='#5AABBB', linewidth=1.5)

ax.set_xlabel('Observed reproductive stage')
ax.set_ylabel('Predicted score')
ax.set_title('Sex-specific brain dynamics prediction',
             fontweight='bold', pad=12, fontsize=10)

textstr = (f'Global r = {r_total:.3f}\n'
           f'Women r = {r_w:.3f}\n'
           f'Men r = {r_m:.3f}')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white',
                  alpha=0.8, edgecolor='none'))

ax.legend(loc='lower right', frameon=False, fontsize=8)
plt.tight_layout()

fig.savefig('figure_sex_interaction.pdf', dpi=300,
            bbox_inches='tight', format='pdf')
fig.savefig('figure_sex_interaction.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Saved: figure_sex_interaction.pdf / .png")
