"""
RAID — Per-Feature Deep Analysis
Analyzes each of the 90 handcrafted features individually:
  1. Single-feature AUC ranking
  2. Top-12 KDE distribution (Human vs AI)
  3. Top-12 boxplots
  4. Per-generator radar chart
  5. Feature importance vs single-feature AUC scatter
  6. Export single-feature stats CSV
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from run_raid_analysis import load_features_and_split

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


def compute_single_feature_stats(X_test, y_test):
    """Compute AUC, means, stds, Cohen's d for each feature."""
    records = []
    for feat in X_test.columns:
        vals = X_test[feat].values
        h_vals = vals[y_test == 0]
        a_vals = vals[y_test == 1]
        # AUC
        try:
            auc = roc_auc_score(y_test, vals)
            if auc < 0.5:
                auc = 1 - auc  # flip if negatively correlated
        except Exception:
            auc = 0.5
        # Cohen's d
        pooled_std = np.sqrt((h_vals.std()**2 + a_vals.std()**2) / 2)
        cohen_d = (a_vals.mean() - h_vals.mean()) / pooled_std if pooled_std > 0 else 0
        records.append({
            'feature': feat,
            'auc': auc,
            'mean_human': h_vals.mean(),
            'mean_ai': a_vals.mean(),
            'std_human': h_vals.std(),
            'std_ai': a_vals.std(),
            'cohen_d': cohen_d,
        })
    df = pd.DataFrame(records).sort_values('auc', ascending=False).reset_index(drop=True)
    return df


def plot_single_feature_auc(stats_df):
    """Bar chart of top-10 single-feature AUCs (Model-Free only, 36d)."""
    # Filter out Model-Based features (emb_pca_*, gpt2_perplexity, gpt2_log_perplexity)
    mf_df = stats_df[~stats_df['feature'].str.startswith(('emb_pca_', 'gpt2_'))].reset_index(drop=True)
    top = mf_df.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)-1, -1, -1), top['auc'].values, color=colors)
    ax.set_yticks(range(len(top)-1, -1, -1))
    ax.set_yticklabels(top['feature'].values, fontsize=11)
    ax.set_xlabel('ROC AUC (single feature)')
    ax.set_title('RAID: Model-Free Feature AUC Ranking (Top 10, 36d)')
    ax.set_xlim(0.5, 1.0)
    for i, v in enumerate(top['auc'].values):
        ax.text(v + 0.003, len(top)-1-i, f'{v:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_single_feature_auc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_single_feature_auc.png")


def plot_single_feature_dist(X_test, y_test, stats_df):
    """KDE distributions for top-12 features."""
    top12 = stats_df.head(12)['feature'].values
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for idx, (feat, ax) in enumerate(zip(top12, axes.flat)):
        h_vals = X_test[feat].values[y_test == 0]
        a_vals = X_test[feat].values[y_test == 1]
        ax.hist(h_vals, bins=50, density=True, alpha=0.5, color='#2196F3', label='Human')
        ax.hist(a_vals, bins=50, density=True, alpha=0.5, color='#F44336', label='AI')
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel('')
        if idx == 0:
            ax.legend(fontsize=8)
    plt.suptitle('RAID: Top-12 Features — Human vs AI Distribution', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_single_feature_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_single_feature_dist.png")


def plot_single_feature_boxplot(X_test, y_test, stats_df):
    """Boxplots for top-12 features."""
    top12 = stats_df.head(12)['feature'].values
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for idx, (feat, ax) in enumerate(zip(top12, axes.flat)):
        plot_data = pd.DataFrame({
            'value': X_test[feat].values,
            'label': ['Human' if y == 0 else 'AI' for y in y_test]
        })
        sns.boxplot(data=plot_data, x='label', y='value', ax=ax,
                    palette={'Human': '#2196F3', 'AI': '#F44336'}, showfliers=False)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.suptitle('RAID: Top-12 Features — Human vs AI Boxplot', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_single_feature_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_single_feature_boxplot.png")


def plot_generator_radar(test_df, X_test, stats_df):
    """Radar chart: top-8 features × generators."""
    top8 = stats_df.head(8)['feature'].values
    models = [m for m in test_df['model'].unique() if m != 'human']

    # Compute per-generator means (z-scored for comparability)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_test[top8]), columns=top8)
    X_scaled['model'] = test_df['model'].values

    means = X_scaled.groupby('model')[list(top8)].mean()
    # Keep only non-human
    means = means.drop('human', errors='ignore')

    angles = np.linspace(0, 2 * np.pi, len(top8), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(means)))
    for i, (model, row) in enumerate(means.iterrows()):
        values = row.values.tolist() + [row.values[0]]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=model, color=cmap[i], markersize=3)
        ax.fill(angles, values, alpha=0.05, color=cmap[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top8, fontsize=9)
    ax.set_title('RAID: Per-Generator Feature Profile (Top-8, z-scored)', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_generator_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_generator_radar.png")


def plot_importance_vs_auc(X_train, y_train, stats_df):
    """Scatter: XGBoost feature importance vs single-feature AUC."""
    feat_cols = [c for c in X_train.columns if not c.startswith('emb_pca_')]
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        eval_metric='logloss', random_state=42,
                        tree_method='hist', device='cuda')
    clf.fit(X_train, y_train, verbose=False)

    imp = pd.Series(clf.feature_importances_, index=X_train.columns)
    merged = stats_df.set_index('feature')
    merged['importance'] = imp

    # Only show non-embedding features for readability
    show = merged[~merged.index.str.startswith('emb_pca_')].dropna(subset=['importance'])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(show['importance'], show['auc'], alpha=0.7, s=40, c='#1976D2')
    for feat in show.index:
        ax.annotate(feat, (show.loc[feat, 'importance'], show.loc[feat, 'auc']),
                    fontsize=7, alpha=0.8, ha='left')
    ax.set_xlabel('XGBoost Feature Importance')
    ax.set_ylabel('Single-Feature AUC')
    ax.set_title('RAID: Feature Importance vs Single-Feature AUC')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_importance_vs_auc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_importance_vs_auc.png")


def main():
    print("=" * 60)
    print("RAID — Per-Feature Deep Analysis")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    train_df, test_df, X_train, X_test = load_features_and_split()
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    print(f"  Features: {X_test.shape[1]}, Test samples: {len(y_test)}")

    print("\n[2/6] Computing single-feature stats...")
    stats_df = compute_single_feature_stats(X_test, y_test)
    stats_df.to_csv(DATA_DIR / 'raid_single_feature_stats.csv', index=False)
    print(f"  Saved: raid_single_feature_stats.csv")
    print(f"\n  Top-10 features by AUC:")
    for _, row in stats_df.head(10).iterrows():
        print(f"    {row['feature']:30s} AUC={row['auc']:.4f}  Cohen_d={row['cohen_d']:+.3f}")

    print("\n[3/6] Single-feature AUC ranking plot...")
    plot_single_feature_auc(stats_df)

    print("\n[4/6] Top-12 feature distributions & boxplots...")
    plot_single_feature_dist(X_test, y_test, stats_df)
    plot_single_feature_boxplot(X_test, y_test, stats_df)

    print("\n[5/6] Per-generator radar chart...")
    plot_generator_radar(test_df, X_test, stats_df)

    print("\n[6/6] Feature importance vs AUC scatter...")
    plot_importance_vs_auc(X_train, y_train, stats_df)

    # Token features Top 10
    print("\n[Bonus] Token feature single-feature AUC...")
    from run_raid_analysis import load_token_features
    tok_tr, tok_te = load_token_features()
    n_te = min(len(tok_te), len(y_test))
    tok_stats = compute_single_feature_stats(tok_te.iloc[:n_te], y_test[:n_te])
    print(f"  Top-10 token features by AUC:")
    for _, row in tok_stats.head(10).iterrows():
        print(f"    {row['feature']:30s} AUC={row['auc']:.4f}")

    top = tok_stats.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top)))
    ax.barh(range(len(top)-1, -1, -1), top['auc'].values, color=colors)
    ax.set_yticks(range(len(top)-1, -1, -1))
    ax.set_yticklabels(top['feature'].values, fontsize=11)
    ax.set_xlabel('ROC AUC (single feature)')
    ax.set_title('RAID: Token Feature AUC Ranking (Top 10)')
    ax.set_xlim(0.5, 1.0)
    for i, v in enumerate(top['auc'].values):
        ax.text(v + 0.003, len(top)-1-i, f'{v:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_token_feature_auc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: raid_token_feature_auc.png")

    print("\n" + "=" * 60)
    print("Done! Generated 6 figures + 1 CSV.")
    print("=" * 60)


if __name__ == "__main__":
    main()
