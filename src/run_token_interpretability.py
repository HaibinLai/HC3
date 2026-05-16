"""
Token-level probability features: interpretability analysis.
All analysis uses cached CSV features — no GPU needed.

Generates:
  figures/token_shap_{hc3,semeval,turingbench,pile}.png
  figures/token_shap_comparison.png
  figures/token_feature_distributions.png
  figures/token_feature_correlation.png
  figures/token_turingbench_permodel.png
  figures/token_shap_waterfall.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "processed"

DATASETS = ['hc3', 'semeval', 'turingbench', 'pile']


def load_dataset(name):
    """Load cached token features + reconstruct labels."""
    X_train = pd.read_csv(DATA_DIR / f"token_features_{name}_train.csv")
    X_test = pd.read_csv(DATA_DIR / f"token_features_{name}_test.csv")
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Reconstruct labels from original data
    if name == 'hc3':
        import sys; sys.path.insert(0, str(ROOT / 'src'))
        from data_splits import get_splits
        train_df, test_df = get_splits()
        n = 5000
        train_sub = pd.concat([g.sample(min(len(g), n), random_state=42)
                               for _, g in train_df.groupby('label')]).reset_index(drop=True)
        test_sub = pd.concat([g.sample(min(len(g), n//2), random_state=42)
                              for _, g in test_df.groupby('label')]).reset_index(drop=True)
        y_train = train_sub['label'].values[:len(X_train)]
        y_test = test_sub['label'].values[:len(X_test)]

    elif name == 'semeval':
        se_dir = ROOT / "data" / "external" / "semeval2024_task8" / "subtaskA_monolingual"
        se_train = pd.read_parquet(se_dir / "train-00000-of-00001.parquet")
        se_test = pd.read_parquet(se_dir / "test-00000-of-00001.parquet")
        se_train_sub = pd.concat([g.sample(min(len(g), 5000), random_state=42)
                                  for _, g in se_train.groupby('label')]).reset_index(drop=True)
        se_test_sub = pd.concat([g.sample(min(len(g), 2500), random_state=42)
                                 for _, g in se_test.groupby('label')]).reset_index(drop=True)
        y_train = se_train_sub['label'].values[:len(X_train)]
        y_test = se_test_sub['label'].values[:len(X_test)]

    elif name == 'turingbench':
        TB_DIR = ROOT / "data" / "external" / "turingbench" / "extracted" / "TuringBench"
        tb_dfs = []
        for subdir in sorted(TB_DIR.iterdir()):
            if subdir.name.startswith('.') or subdir.name == '__MACOSX':
                continue
            for split in ['train', 'test']:
                f = subdir / f'{split}.csv'
                if f.exists():
                    df = pd.read_csv(f)
                    df['split'] = split
                    df['model'] = subdir.name
                    tb_dfs.append(df)
        tb_all = pd.concat(tb_dfs, ignore_index=True).rename(columns={'Generation': 'text'})
        tb_all['label'] = (tb_all['model'] != 'AA').astype(int)
        tb_all = tb_all.dropna(subset=['text'])
        tb_all = tb_all[tb_all['text'].str.len() > 10].reset_index(drop=True)
        tb_train = tb_all[tb_all['split'] == 'train'].reset_index(drop=True)
        tb_test = tb_all[tb_all['split'] == 'test'].reset_index(drop=True)
        n_train, n_test = 10000, 5000
        tb_train_sub = pd.concat([g.sample(min(len(g), n_train // 2), random_state=42)
                                   for _, g in tb_train.groupby('label')]).reset_index(drop=True)
        tb_test_sub = pd.concat([g.sample(min(len(g), n_test // 2), random_state=42)
                                  for _, g in tb_test.groupby('label')]).reset_index(drop=True)
        y_train = tb_train_sub['label'].values[:len(X_train)]
        y_test = tb_test_sub['label'].values[:len(X_test)]
        return X_train, X_test, y_train, y_test, tb_test_sub

    elif name == 'pile':
        from sklearn.model_selection import train_test_split
        pile_dir = ROOT / "data" / "external" / "ai_text_detection_pile" / "data"
        pile_dfs = []
        for f in sorted(pile_dir.glob("*.parquet")):
            pile_dfs.append(pd.read_parquet(f, columns=['text', 'source']))
        pile_all = pd.concat(pile_dfs, ignore_index=True)
        pile_all['label'] = (pile_all['source'] == 'ai').astype(int)
        pile_all = pile_all.dropna(subset=['text'])
        pile_all = pile_all[pile_all['text'].str.len() > 10].reset_index(drop=True)
        pile_sub = pd.concat([g.sample(min(len(g), 10000), random_state=42)
                               for _, g in pile_all.groupby('label')]).reset_index(drop=True)
        pile_train, pile_test = train_test_split(pile_sub, test_size=0.33,
                                                 stratify=pile_sub['label'], random_state=42)
        y_train = pile_train['label'].values[:len(X_train)]
        y_test = pile_test['label'].values[:len(X_test)]

    return X_train, X_test, y_train, y_test, None


def train_xgb(X_train, y_train):
    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist')
    clf.fit(X_train, y_train, verbose=False)
    return clf


# ────────────────────────────────────────────────────────
# 1. SHAP summary plots per dataset
# ────────────────────────────────────────────────────────
def plot_shap_per_dataset(datasets_data):
    shap_importances = {}

    for name, (X_tr, X_te, y_tr, y_te, clf) in datasets_data.items():
        print(f"  SHAP for {name}...")
        explainer = shap.TreeExplainer(clf)
        sample_n = min(1500, len(X_te))
        X_sample = X_te.iloc[:sample_n]
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Save per-dataset SHAP summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=X_tr.columns.tolist(),
                          show=False, max_display=15)
        plt.title(f'SHAP Feature Importance: {name.upper()}', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f'token_shap_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Collect mean |SHAP| for comparison
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_importances[name] = pd.Series(mean_abs, index=X_tr.columns)

    # Comparison plot: top-10 features per dataset
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (name, imp) in zip(axes.flat, shap_importances.items()):
        top = imp.nlargest(10)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
        ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1], fontsize=10)
        ax.set_xlabel('Mean |SHAP value|')
        auc = roc_auc_score(datasets_data[name][3], datasets_data[name][4].predict_proba(
            datasets_data[name][1].iloc[:len(datasets_data[name][3])])[:, 1])
        ax.set_title(f'{name.upper()} (AUC={auc:.4f})', fontsize=13, fontweight='bold')
    plt.suptitle('Token Feature SHAP Importance Across Datasets', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'token_shap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved token_shap_comparison.png")

    return shap_importances


# ────────────────────────────────────────────────────────
# 2. Feature distributions: Human vs AI
# ────────────────────────────────────────────────────────
def plot_feature_distributions(datasets_data, shap_importances):
    # Pick top-5 features from HC3 SHAP
    top5 = shap_importances['hc3'].nlargest(5).index.tolist()

    fig, axes = plt.subplots(len(top5), 4, figsize=(20, 4 * len(top5)))

    for i, feat in enumerate(top5):
        for j, name in enumerate(DATASETS):
            ax = axes[i, j]
            X_te = datasets_data[name][1]
            y_te = datasets_data[name][3]
            n = min(len(X_te), len(y_te))
            vals_h = X_te[feat].values[:n][y_te[:n] == 0]
            vals_ai = X_te[feat].values[:n][y_te[:n] == 1]

            parts = ax.violinplot([vals_h, vals_ai], positions=[0, 1], showmedians=True)
            for pc, color in zip(parts['bodies'], ['#2E7D32', '#C62828']):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Human', 'AI'])
            if j == 0:
                ax.set_ylabel(feat, fontsize=11, fontweight='bold')
            if i == 0:
                ax.set_title(name.upper(), fontsize=12, fontweight='bold')

    plt.suptitle('Top-5 Token Features: Human vs AI Distribution', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'token_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved token_feature_distributions.png")


# ────────────────────────────────────────────────────────
# 3. Feature correlation matrix
# ────────────────────────────────────────────────────────
def plot_correlation(datasets_data):
    X = datasets_data['hc3'][0]  # use HC3 training data
    corr = X.corr()

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, annot=False,
                xticklabels=True, yticklabels=True)
    plt.title('Token Feature Correlation Matrix (HC3)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=7, rotation=45, ha='right')
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'token_feature_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved token_feature_correlation.png")


# ────────────────────────────────────────────────────────
# 4. TuringBench failure analysis: per-model feature dist
# ────────────────────────────────────────────────────────
def plot_turingbench_failure(datasets_data):
    _, X_te, _, y_te, meta = datasets_data['turingbench']
    if meta is None or 'model' not in meta.columns:
        print("  No per-model info for TuringBench, skipping")
        return

    n = min(len(X_te), len(meta))
    models = meta['model'].values[:n]

    # Key features to compare
    key_feats = ['rank_top100_frac', 'lp_mean', 'ent_mean', 'top1p_mean']
    available = [f for f in key_feats if f in X_te.columns]

    fig, axes = plt.subplots(len(available), 1, figsize=(16, 4 * len(available)))
    if len(available) == 1:
        axes = [axes]

    unique_models = sorted(set(models))
    # Reorder: human first
    if 'AA' in unique_models:
        unique_models = ['AA'] + [m for m in unique_models if m != 'AA']

    for ax, feat in zip(axes, available):
        data_by_model = []
        labels = []
        for m in unique_models:
            mask = models == m
            vals = X_te[feat].values[:n][mask]
            if len(vals) > 0:
                data_by_model.append(vals)
                labels.append(m)

        bp = ax.boxplot(data_by_model, labels=labels, patch_artist=True)
        # Color human green, AI red
        for i, patch in enumerate(bp['boxes']):
            if labels[i] == 'AA':
                patch.set_facecolor('#2E7D32')
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor('#C62828')
                patch.set_alpha(0.4)

        ax.set_ylabel(feat, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('TuringBench: Why Token Features Fail\nHuman (AA, green) vs AI Models (red) — distributions overlap',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'token_turingbench_permodel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved token_turingbench_permodel.png")


# ────────────────────────────────────────────────────────
# 5. SHAP waterfall for individual samples
# ────────────────────────────────────────────────────────
def plot_shap_waterfall(datasets_data):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for row, name in enumerate(['hc3', 'semeval']):
        X_tr, X_te, y_tr, y_te, clf = datasets_data[name][0], datasets_data[name][1], \
                                       datasets_data[name][2], datasets_data[name][3], datasets_data[name][4]
        explainer = shap.TreeExplainer(clf)
        n = min(len(X_te), len(y_te))
        y_prob = clf.predict_proba(X_te.iloc[:n])[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # Find correct and incorrect predictions
        correct = np.where((y_pred == y_te[:n]))[0]
        incorrect = np.where((y_pred != y_te[:n]))[0]

        # Pick 2 correct (1 human, 1 AI) + 1 incorrect
        samples = []
        for idx in correct:
            if y_te[idx] == 0 and len(samples) < 1:
                samples.append(('Correct\n(Human)', idx))
            elif y_te[idx] == 1 and len(samples) < 2:
                samples.append(('Correct\n(AI)', idx))
            if len(samples) >= 2:
                break
        if len(incorrect) > 0:
            samples.append(('Misclassified', incorrect[0]))

        for col, (label, idx) in enumerate(samples):
            ax = axes[row, col]
            sv = explainer.shap_values(X_te.iloc[[idx]])
            if isinstance(sv, list):
                sv = sv[1]
            sv = sv[0]

            # Manual waterfall as bar chart
            feat_names = X_te.columns.tolist()
            top_k = 10
            abs_sv = np.abs(sv)
            top_idx = np.argsort(abs_sv)[::-1][:top_k]

            colors = ['#C62828' if sv[i] > 0 else '#2E7D32' for i in top_idx]
            ax.barh(range(top_k), sv[top_idx][::-1], color=[colors[i] for i in range(top_k)][::-1])
            ax.set_yticks(range(top_k))
            ax.set_yticklabels([feat_names[i] for i in top_idx][::-1], fontsize=8)
            ax.set_xlabel('SHAP value')
            true_label = 'AI' if y_te[idx] == 1 else 'Human'
            pred_label = f'P(AI)={y_prob[idx]:.3f}'
            ax.set_title(f'{name.upper()} — {label}\nTrue={true_label}, {pred_label}', fontsize=10)

    plt.suptitle('Single-Sample SHAP Attribution (Token Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'token_shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved token_shap_waterfall.png")


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("Token Feature Interpretability Analysis")
    print("="*60)

    # Load all datasets and train XGBoost
    datasets_data = {}
    for name in DATASETS:
        print(f"\nLoading {name}...")
        result = load_dataset(name)
        X_tr, X_te, y_tr, y_te = result[0], result[1], result[2], result[3]
        meta = result[4] if len(result) > 4 else None
        clf = train_xgb(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te.iloc[:len(y_te)])[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        print(f"  {name}: AUC={auc:.4f} (train={len(X_tr)}, test={len(X_te)})")
        datasets_data[name] = (X_tr, X_te, y_tr, y_te, clf, meta)

    # 1. SHAP
    print("\n--- 1. SHAP Analysis ---")
    # Repackage for shap function (X_tr, X_te, y_tr, y_te, clf)
    shap_data = {n: (d[0], d[1], d[2], d[3], d[4]) for n, d in datasets_data.items()}
    shap_importances = plot_shap_per_dataset(shap_data)

    # 2. Feature distributions
    print("\n--- 2. Feature Distributions ---")
    plot_feature_distributions(shap_data, shap_importances)

    # 3. Correlation
    print("\n--- 3. Feature Correlation ---")
    plot_correlation(shap_data)

    # 4. TuringBench failure
    print("\n--- 4. TuringBench Failure Analysis ---")
    # Need meta info — re-extract from datasets_data
    tb_data_with_meta = {
        'turingbench': (datasets_data['turingbench'][0], datasets_data['turingbench'][1],
                        datasets_data['turingbench'][2], datasets_data['turingbench'][3],
                        datasets_data['turingbench'][5])  # meta
    }
    plot_turingbench_failure(tb_data_with_meta)

    # 5. Waterfall
    print("\n--- 5. SHAP Waterfall ---")
    plot_shap_waterfall(shap_data)

    print("\n" + "="*60)
    print("All interpretability figures generated!")
    print("="*60)
