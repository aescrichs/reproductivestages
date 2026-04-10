"""
Pairwise classification of menopausal groups — Python implementation
Brain dynamics → menopausal stage classification (HCP-A women)
 
Pipeline: KNNImputer → StandardScaler → SelectKBest(k=20, f_classif)
          → [SMOTE] → XGBoost / RandomForest
Cross-validation: Stratified outer CV, 3-fold inner RandomizedSearchCV
Metrics: AUC, Accuracy, F1-macro, F1-weighted
Output: results CSV + feature importance plots (top 15 per comparison)
 
Note: All 4 model combinations (XGBoost/RF × No resampling/SMOTE) are
evaluated to select the best-performing model for the main analysis.
Final model selection is based on AUC on the held-out test folds.

"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')   # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def run_classification(df):
    group_names = {
        1: 'Reproductive', 2: 'Late Reproductive', 3: 'Perimenopause',
        4: 'Early Postmenopause', 5: 'Late Postmenopause'
    }

    key_pairs = [
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 5)
    ]

    classifiers = {
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='auc',
            objective='binary:logistic',
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=1
        )
    }

    param_grid_xgb = {
        'classifier__n_estimators':     [50, 100, 200],
        'classifier__max_depth':        [2, 3, 5],
        'classifier__learning_rate':    [0.01, 0.1],
        'classifier__min_child_weight': [1, 3],
        'classifier__reg_alpha':        [0, 0.1],
        'classifier__reg_lambda':       [1, 5]
    }

    param_grid_rf = {
        'classifier__n_estimators':     [50, 100, 200],
        'classifier__max_depth':        [None, 5],
        'classifier__min_samples_split':[2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    all_results = []

    for group1, group2 in key_pairs:
        group1_name = group_names.get(group1, f"Group {group1}")
        group2_name = group_names.get(group2, f"Group {group2}")

        print(f"\n{'='*70}")
        print(f"--- {group1_name} vs {group2_name} ---")
        print(f"{'='*70}")

        df_filtered = df[df['Group'].isin([group1, group2])].copy()
        if len(df_filtered) < 10:
            print(f"Skipping: too few samples ({len(df_filtered)}).")
            continue

        X = df_filtered.drop(columns=['Group'])
        le = LabelEncoder()
        y_encoded = le.fit_transform(df_filtered['Group'])

        n_samples_minority = pd.Series(y_encoded).value_counts().min()
        if n_samples_minority < 5:
            print(f"Skipping: minority class has {n_samples_minority} samples.")
            continue

        skf_outer = StratifiedKFold(
            n_splits=min(5, n_samples_minority),
            shuffle=True, random_state=RANDOM_STATE)

        for clf_name, clf in classifiers.items():
            print(f"\n  â†’ {clf_name}")

            current_grid = param_grid_xgb if clf_name == 'XGBoost' else param_grid_rf

            for resample_name in ['No Resampling', 'SMOTE']:
                print(f"     â†’ {resample_name}...")

                all_y_pred  = np.zeros_like(y_encoded)
                all_y_proba = np.zeros(len(y_encoded))
                feature_importances_per_fold = []
                best_params_per_fold = []

                for fold, (train_idx, test_idx) in enumerate(
                        skf_outer.split(X, y_encoded)):

                    X_train = X.iloc[train_idx]
                    X_test  = X.iloc[test_idx]
                    y_train = y_encoded[train_idx]

                    # Pipeline: feature selection BEFORE SMOTE
                    steps = [
                        ('imputer', KNNImputer(n_neighbors=5)),
                        ('scaler',  StandardScaler()),
                        ('feature_selection', SelectKBest(
                            f_classif, k=min(20, X_train.shape[1])))
                    ]

                    if resample_name == 'SMOTE':
                        n_min_train = pd.Series(y_train).value_counts().min()
                        k_neighbors = max(1, min(5, n_min_train - 1))
                        steps.append(('resampler', SMOTE(
                            random_state=RANDOM_STATE,
                            k_neighbors=k_neighbors)))

                    steps.append(('classifier', clf))
                    pipeline = ImbPipeline(steps)

                    cv_inner = StratifiedKFold(
                        n_splits=3, shuffle=True, random_state=RANDOM_STATE)

                    grid_search = RandomizedSearchCV(
                        pipeline,
                        current_grid,
                        n_iter=60,
                        cv=cv_inner,
                        n_jobs=2,
                        scoring='f1_macro',
                        random_state=RANDOM_STATE,
                        verbose=0
                    )
                    np.random.seed(RANDOM_STATE)
                    grid_search.fit(X_train, y_train)

                    all_y_pred[test_idx]  = grid_search.best_estimator_.predict(X_test)
                    all_y_proba[test_idx] = grid_search.best_estimator_.predict_proba(
                        X_test)[:, 1]

                    best_model    = grid_search.best_estimator_
                    mask          = best_model.named_steps['feature_selection'].get_support()
                    fold_features = X.columns[mask]
                    fold_imps     = best_model.named_steps['classifier'].feature_importances_

                    feature_importances_per_fold.append(pd.DataFrame({
                        'Feature':    fold_features,
                        'Importance': fold_imps,
                        'Fold':       fold + 1
                    }))
                    best_params_per_fold.append(grid_search.best_params_)

                # â”€â”€ Feature importance aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                feature_stats = defaultdict(
                    lambda: {'sum_imp': 0.0, 'count': 0, 'imp_list': []})
                for fold_df in feature_importances_per_fold:
                    for _, row in fold_df.iterrows():
                        feat = row['Feature']
                        imp  = row['Importance']
                        feature_stats[feat]['sum_imp']  += imp
                        feature_stats[feat]['count']    += 1
                        feature_stats[feat]['imp_list'].append(imp)

                fi_records = []
                n_folds = len(feature_importances_per_fold)
                for feat, stats in feature_stats.items():
                    mean_imp = stats['sum_imp'] / stats['count']
                    std_imp  = np.std(stats['imp_list']) if len(stats['imp_list']) > 1 else 0
                    freq     = stats['count'] / n_folds
                    fi_records.append({
                        'Feature':             feat,
                        'Mean_Importance':     mean_imp,
                        'Std_Importance':      std_imp,
                        'Selection_Frequency': freq
                    })

                feat_df = pd.DataFrame(fi_records).sort_values(
                    'Mean_Importance', ascending=False)
                feat_df = feat_df[feat_df['Selection_Frequency'] >= 0.8]

                # Build long-format df for stripplot (individual fold values)
                df_all_imp = pd.concat(
                    feature_importances_per_fold, ignore_index=True)

                # â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                auc    = roc_auc_score(y_encoded, all_y_proba)
                report = classification_report(
                    y_encoded, all_y_pred,
                    target_names=[group1_name, group2_name],
                    output_dict=True, zero_division=0)

                most_common_params = dict(Counter(
                    [frozenset(p.items()) for p in best_params_per_fold]
                ).most_common(1)[0][0])

                result = {
                    'group1':      group1_name,
                    'group2':      group2_name,
                    'classifier':  clf_name,
                    'resampling':  resample_name,
                    'accuracy':    report['accuracy'],
                    'auc':         auc,
                    'f1_macro':    report['macro avg']['f1-score'],
                    'f1_weighted': report['weighted avg']['f1-score'],
                    'best_params': most_common_params
                }

                print(f"     AUC = {auc:.4f} | "
                      f"Accuracy = {report['accuracy']:.4f} | "
                      f"F1-macro = {report['macro avg']['f1-score']:.4f}")

                all_results.append(result)

                # â”€â”€ Feature importance figure (Nature Aging style) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if not feat_df.empty:
                    top15_df = feat_df.sort_values(
                        'Mean_Importance', ascending=False).head(15)
                    order    = top15_df['Feature'].tolist()
                    df_plot  = df_all_imp[df_all_imp['Feature'].isin(order)]

                    fig, ax = plt.subplots(figsize=(7.2, 5))

                    # Barplot (background)
                    sns.barplot(
                        x='Mean_Importance', y='Feature',
                        data=top15_df, order=order,
                        palette='RdPu_r', errorbar=None,
                        zorder=1, ax=ax)

                    # Error bars (manual, Â±SD)
                    ax.errorbar(
                        x=top15_df['Mean_Importance'],
                        y=range(len(top15_df)),
                        xerr=top15_df['Std_Importance'],
                        fmt='none', ecolor='black',
                        capsize=3, elinewidth=1.5, zorder=2)

                    # Stripplot (individual fold values)
                    sns.stripplot(
                        data=df_plot,
                        x='Importance', y='Feature',
                        order=order,
                        color='black', alpha=0.6,
                        jitter=0.25, size=4,
                        zorder=3, ax=ax)

                    ax.set_xlim(left=0)
                    ax.set_xlabel('Mean Feature Importance (Â± SD across folds)')
                    ax.set_ylabel('')
                    ax.set_title(
                        f'{group1_name} vs {group2_name}\n'
                        f'({clf_name} with {resample_name})',
                        fontsize=9)
                    ax.grid(axis='x', linestyle='--', alpha=0.3)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()

                    fname = (f"feat_imp_{group1_name}_vs_{group2_name}_"
                             f"{clf_name}_{resample_name}.png"
                             ).replace(' ', '_')
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    plt.close(fig)   # never plt.show() inside a loop

    # â”€â”€ Save final results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    results_df = pd.DataFrame(all_results)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv    = f'classification_results_{timestamp}.csv'
    results_df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print(f"  Done. Results saved to {out_csv}")
    print(f"{'='*70}")

    # Summary: best model per pair by AUC
    print("\n  BEST MODEL PER PAIR (by AUC):")
    print(f"  {'Comparison':<40} {'Classifier':<15} {'Resampling':<16} {'AUC'}")
    print(f"  {'-'*80}")
    for _, grp in results_df.groupby(['group1', 'group2']):
        best_row   = grp.loc[grp['auc'].idxmax()]
        comparison = f"{best_row['group1']} vs {best_row['group2']}"
        print(f"  {comparison:<40} {best_row['classifier']:<15} "
              f"{best_row['resampling']:<16} {best_row['auc']:.4f}")

    return results_df, df


def load_data():
    try:
        df = pd.read_csv('data.csv')
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data.csv: {e}")
        sys.exit(1)


def main():
    df = load_data()
    results_df, _ = run_classification(df)
    return results_df


if __name__ == "__main__":
    main()
