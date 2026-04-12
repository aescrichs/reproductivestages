"""
Multiclass classification of menopausal groups — Python implementation
Brain dynamics → menopausal stage classification (HCP-A women)
 
Pipeline: KNNImputer → StandardScaler → [SMOTE] → SelectKBest(k=20, f_classif)
          → XGBoost / RandomForest
Cross-validation: Stratified outer CV, 3-fold inner RandomizedSearchCV
Metrics: Accuracy, Balanced Accuracy, F1-macro, F1-weighted
Output: results CSV + feature importance plots (top 15, Nature Aging style)
 
Note: All 4 model combinations (XGBoost/RF × No resampling/SMOTE) are
evaluated to select the best-performing model for the main analysis.
Final model selection is based on F1-macro on the held-out test folds.
SMOTE is applied inside ImbPipeline, ensuring it only affects training folds
and preventing data leakage into the test set.

"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict, Counter
from datetime import datetime

import matplotlib
matplotlib.use('Agg')   # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_classification(df):
    group_names = {
        1: 'Reproductive', 2: 'Late Reproductive', 3: 'Perimenopause',
        4: 'Early Postmenopause', 5: 'Late Postmenopause'
    }

    resampling_strategies = {
        'No Resampling': None,
        'SMOTE': True
    }

    classifiers = {
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=5,
            random_state=RANDOM_STATE,
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced'
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

    print(f"\n{'='*60}")
    print(f"  MULTICLASS: All 5 Reproductive Stages")
    print(f"{'='*60}")

    X = df.drop(columns=['Group'])
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['Group'])

    n_samples_minority = pd.Series(y_encoded).value_counts().min()
    if n_samples_minority < 5:
        print(f"Error: minority class has only {n_samples_minority} samples.")
        return pd.DataFrame(), df

    skf_outer = StratifiedKFold(
        n_splits=min(5, n_samples_minority),
        shuffle=True, random_state=RANDOM_STATE)

    for clf_name, clf in classifiers.items():
        print(f"\n  â†’ {clf_name}")
        current_grid = param_grid_xgb if clf_name == 'XGBoost' else param_grid_rf

        for resampler_name, _ in resampling_strategies.items():
            print(f"     â†’ {resampler_name}...")

            all_y_pred  = np.zeros_like(y_encoded)
            all_y_proba = np.zeros((len(y_encoded), 5))
            best_params_per_fold      = []
            feature_importances_per_fold = []

            for fold, (train_idx, test_idx) in enumerate(
                    skf_outer.split(X, y_encoded)):

                X_train = X.iloc[train_idx]
                X_test  = X.iloc[test_idx]
                y_train = y_encoded[train_idx]

                n_min_train = pd.Series(y_train).value_counts().min()

                # Pipeline: imputer â†’ scaler â†’ [SMOTE] â†’ selector â†’ classifier
                steps = [
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('scaler',  StandardScaler()),
                ]

                if resampler_name == 'SMOTE':
                    k = max(1, min(5, n_min_train - 1))
                    steps.insert(2, ('resampler', SMOTE(
                        random_state=RANDOM_STATE, k_neighbors=k)))

                k_features = min(20, X_train.shape[1])
                steps.append(('feature_selection', SelectKBest(
                    f_classif, k=k_features)))
                steps.append(('classifier', clf))

                pipeline = ImbPipeline(steps)

                n_inner_splits = min(3, int(
                    pd.Series(y_train).value_counts().min()))
                cv_inner = StratifiedKFold(
                    n_splits=n_inner_splits, shuffle=True,
                    random_state=RANDOM_STATE)

                grid_search = RandomizedSearchCV(
                    pipeline,
                    current_grid,
                    n_iter=100,
                    cv=cv_inner,
                    n_jobs=-1,
                    scoring='f1_macro',
                    random_state=RANDOM_STATE
                )
                grid_search.fit(X_train, y_train)

                all_y_pred[test_idx]  = grid_search.best_estimator_.predict(X_test)
                all_y_proba[test_idx] = grid_search.best_estimator_.predict_proba(X_test)
                best_params_per_fold.append(grid_search.best_params_)

                best_model    = grid_search.best_estimator_
                mask          = best_model.named_steps['feature_selection'].get_support()
                fold_features = X.columns[mask]
                fold_imps     = best_model.named_steps['classifier'].feature_importances_

                feature_importances_per_fold.append(pd.DataFrame({
                    'Feature':    fold_features,
                    'Importance': fold_imps,
                    'Fold':       fold + 1
                }))

            # Feature importance aggregation 
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

            print(f"\n  Top features ({clf_name}, {resampler_name}):")
            print(f"  {'-'*50}")
            print(feat_df[['Feature', 'Mean_Importance', 'Selection_Frequency']
                          ].to_string(index=False, formatters={
                'Mean_Importance':     '{:.4f}'.format,
                'Selection_Frequency': '{:.1%}'.format
            }))

            # Metrics 
            y_original      = le.inverse_transform(y_encoded)
            y_pred_original = le.inverse_transform(all_y_pred)
            target_names    = [group_names[i] for i in sorted(group_names.keys())]

            report      = classification_report(
                y_original, y_pred_original,
                target_names=target_names,
                output_dict=True, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_encoded, all_y_pred)
            most_common  = dict(Counter(
                [frozenset(p.items()) for p in best_params_per_fold]
            ).most_common(1)[0][0])

            for gid in sorted(group_names.keys()):
                gname = group_names[gid]
                print(f"  {gname}: F1={report[gname]['f1-score']:.3f} "
                      f"(n={report[gname]['support']})")

            print(f"  Accuracy = {report['accuracy']:.3f} | "
                  f"Balanced Acc = {balanced_acc:.3f} | "
                  f"F1-macro = {report['macro avg']['f1-score']:.3f} | "
                  f"F1-weighted = {report['weighted avg']['f1-score']:.3f}")

            result = {
                'classifier':        clf_name,
                'resampling':        resampler_name,
                'accuracy':          report['accuracy'],
                'balanced_accuracy': balanced_acc,
                'precision_macro':   report['macro avg']['precision'],
                'recall_macro':      report['macro avg']['recall'],
                'f1_macro':          report['macro avg']['f1-score'],
                'f1_weighted':       report['weighted avg']['f1-score'],
                'best_params':       most_common,
            }
            for gid in sorted(group_names.keys()):
                gname = group_names[gid]
                result[f'precision_{gname}'] = report[gname]['precision']
                result[f'recall_{gname}']    = report[gname]['recall']
                result[f'f1_{gname}']        = report[gname]['f1-score']
                result[f'support_{gname}']   = report[gname]['support']
            all_results.append(result)

            # Feature importance figure (Nature Aging style)
            if not feat_df.empty:
                top15_df   = feat_df.head(15)
                order      = top15_df['Feature'].tolist()
                df_all_imp = pd.concat(
                    feature_importances_per_fold, ignore_index=True)
                df_plot    = df_all_imp[df_all_imp['Feature'].isin(order)]

                fig, ax = plt.subplots(figsize=(7.2, 5))

                sns.barplot(
                    x='Mean_Importance', y='Feature',
                    data=top15_df, order=order,
                    palette='RdPu_r', errorbar=None,
                    zorder=1, ax=ax)

                ax.errorbar(
                    x=top15_df['Mean_Importance'],
                    y=range(len(top15_df)),
                    xerr=top15_df['Std_Importance'],
                    fmt='none', ecolor='black',
                    capsize=3, elinewidth=1.5, zorder=2)

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
                    f'Multiclass â€” all 5 stages\n'
                    f'{clf_name} with {resampler_name}',
                    fontsize=9)
                ax.grid(axis='x', linestyle='--', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()

                fname = (f"feat_imp_multiclass_{clf_name}_{resampler_name}"
                         ).replace(' ', '_')
                fig.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')
                fig.savefig(f'{fname}.pdf', format='pdf', bbox_inches='tight')
                plt.close(fig)

    # Save results 
    results_df = pd.DataFrame(all_results)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv    = f'multiclass_results_{timestamp}.csv'
    results_df.to_csv(out_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Done. Results saved to {out_csv}")
    print(f"{'='*60}")

    best = results_df.loc[results_df['f1_macro'].idxmax()]
    print(f"\n  Best model (F1-macro): {best['classifier']} + {best['resampling']} | "
          f"F1-macro = {best['f1_macro']:.4f} | "
          f"Balanced Acc = {best['balanced_accuracy']:.4f}")

    return results_df, df


# =============================================================================
# LOAD DATA & ENTRY POINT
# =============================================================================

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
