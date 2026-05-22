"""
RAID Benchmark — Deep Analysis & Interpretability
Replicates the full HC3 analysis pipeline on RAID:
  1. 90-feature correlation, PCA, t-SNE, SHAP, ablation, LR coefficients
  2. Token-level SHAP, distributions, per-model analysis
  3. Case-level heatmaps, t-SNE, misclassification
  4. RAID-specific: adversarial feature shift, decoding strategy, model×domain heatmap
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "external" / "raid"
RESULT_DIR = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / 'src'))

from run_raid import extract_features_batch, load_raid, split_raid

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


# ════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════
def load_features_and_split():
    """Load RAID data, features (cached), and return train/test split with metadata."""
    df = load_raid(with_attacks=False)
    text_col = 'generation' if 'generation' in df.columns else 'text'
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.len() > 20].reset_index(drop=True)
    train_df, test_df = split_raid(df)

    cache_train = RESULT_DIR / "raid_features_train.csv"
    cache_test  = RESULT_DIR / "raid_features_test.csv"

    if cache_train.exists() and cache_test.exists():
        X_train = pd.read_csv(cache_train)
        X_test  = pd.read_csv(cache_test)
    else:
        raise FileNotFoundError("Run src/run_raid.py first to generate cached features.")

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    return train_df, test_df, X_train, X_test


def load_token_features():
    """Load cached token-level probability features."""
    cache_train = RESULT_DIR / "token_features_raid_train.csv"
    cache_test  = RESULT_DIR / "token_features_raid_test.csv"
    if not cache_train.exists():
        raise FileNotFoundError("Run src/run_raid.py first to generate token features.")
    Xt_train = pd.read_csv(cache_train).replace([np.inf, -np.inf], np.nan).fillna(0)
    Xt_test  = pd.read_csv(cache_test).replace([np.inf, -np.inf], np.nan).fillna(0)
    return Xt_train, Xt_test


# ════════════════════════════════════════════════════════════
# PART 1: 90-Feature Analysis
# ════════════════════════════════════════════════════════════

def plot_correlation_heatmap(X_train):
    """Feature correlation heatmap (CPU features only)."""
    cpu_cols = [c for c in X_train.columns if not c.startswith('emb_pca_')]
    corr = X_train[cpu_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.3,
                cbar_kws={'shrink': 0.7, 'label': 'Pearson r'})
    ax.set_title('RAID: Feature Correlation Matrix (CPU features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_correlation_heatmap.png")


def plot_domain_comparison(test_df, X_test):
    """Boxplots of key features by domain × label."""
    key_feats = ['type_token_ratio', 'avg_sentence_length', 'flesch_reading_ease',
                 'word_count', 'gpt2_perplexity', 'hapax_ratio']
    key_feats = [f for f in key_feats if f in X_test.columns]

    plot_df = X_test[key_feats].copy()
    plot_df['label'] = test_df['label'].values[:len(plot_df)]
    plot_df['label_name'] = plot_df['label'].map({0: 'Human', 1: 'AI'})
    if 'domain' in test_df.columns:
        plot_df['domain'] = test_df['domain'].values[:len(plot_df)]
    else:
        return

    domains = sorted(plot_df['domain'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for i, feat in enumerate(key_feats[:6]):
        ax = axes[i]
        sns.boxplot(data=plot_df, x='domain', y=feat, hue='label_name',
                    ax=ax, palette={'Human': '#2196F3', 'AI': '#F44336'},
                    fliersize=1, linewidth=0.8)
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('')
        if i > 0:
            ax.get_legend().remove()

    plt.suptitle('RAID: Key Feature Comparison by Domain (Human vs AI)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_domain_feature_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_domain_feature_comparison.png")


def plot_pca(X_test, y_test):
    """PCA 2D scatter."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    n = min(5000, len(X_2d))
    idx = np.random.RandomState(42).choice(len(X_2d), n, replace=False)

    for label, color, name in [(0, '#2196F3', 'Human'), (1, '#F44336', 'AI')]:
        mask = y_test[idx] == label
        ax.scatter(X_2d[idx[mask], 0], X_2d[idx[mask], 1],
                   c=color, alpha=0.3, s=10, label=name)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('RAID: PCA of 90 Features')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_pca.png")


def plot_tsne(X_test, y_test):
    """t-SNE 2D scatter."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    n = min(5000, len(X_test))
    idx = np.random.RandomState(42).choice(len(X_test), n, replace=False)
    scaler = StandardScaler()
    X_sub = scaler.fit_transform(X_test.iloc[idx])
    y_sub = y_test[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color, name in [(0, '#2196F3', 'Human'), (1, '#F44336', 'AI')]:
        mask = y_sub == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.3, s=10, label=name)

    ax.set_title('RAID: t-SNE of 90 Features')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_tsne.png")


def train_models_and_shap(X_train, y_train, X_test, y_test):
    """Train XGBoost & LR, compute SHAP, return models + shap values."""
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import shap

    # XGBoost
    clf_xgb = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=42, tree_method='hist', device='cuda')
    clf_xgb.fit(X_train, y_train, verbose=False)
    y_prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    print(f"  XGBoost AUC: {auc_xgb:.4f}")

    # LR
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    clf_lr = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced',
                                solver='saga', random_state=42)
    clf_lr.fit(X_tr_s, y_train)
    y_prob_lr = clf_lr.predict_proba(X_te_s)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    print(f"  LR AUC: {auc_lr:.4f}")

    # SHAP
    print("  Computing SHAP values...")
    n_shap = min(2000, len(X_test))
    idx = np.random.RandomState(42).choice(len(X_test), n_shap, replace=False)
    explainer = shap.TreeExplainer(clf_xgb)
    shap_values = explainer(X_test.iloc[idx])

    return clf_xgb, clf_lr, scaler, shap_values, idx, y_prob_xgb


def plot_shap_summary(shap_values):
    """SHAP beeswarm plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    shap_import = np.abs(shap_values.values[:, :, 1] if shap_values.values.ndim == 3
                         else shap_values.values).mean(axis=0)
    top_idx = np.argsort(shap_import)[::-1][:25]

    import shap as shap_lib
    sv = shap_values[:, top_idx]
    if sv.values.ndim == 3:
        sv = shap.Explanation(values=sv.values[:, :, 1], base_values=sv.base_values[:, 1]
                              if sv.base_values.ndim > 1 else sv.base_values,
                              data=sv.data, feature_names=sv.feature_names)
    shap_lib.plots.beeswarm(sv, show=False, max_display=25)
    plt.title('RAID: SHAP Summary (XGBoost, Top 25 Features)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_shap_summary.png")


def plot_shap_dependence(shap_values, X_sample):
    """SHAP dependence plots for top 6 features."""
    sv = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values
    shap_import = np.abs(sv).mean(axis=0)
    top6 = np.argsort(shap_import)[::-1][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, fidx in enumerate(top6):
        ax = axes[i]
        feat_name = X_sample.columns[fidx]
        feat_vals = X_sample.iloc[:, fidx].values
        shap_vals = sv[:, fidx]

        # Find best interaction feature
        best_inter = 0
        best_corr = 0
        for j in range(sv.shape[1]):
            if j == fidx:
                continue
            c = abs(np.corrcoef(sv[:, j], shap_vals)[0, 1])
            if c > best_corr:
                best_corr = c
                best_inter = j

        inter_vals = X_sample.iloc[:, best_inter].values
        inter_name = X_sample.columns[best_inter]

        sc = ax.scatter(feat_vals, shap_vals, c=inter_vals, cmap='coolwarm',
                        alpha=0.4, s=8, edgecolors='none')
        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel('SHAP value', fontsize=10)
        ax.set_title(f'{feat_name}\n(color: {inter_name})', fontsize=10)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    plt.suptitle('RAID: SHAP Dependence Plots (Top 6 Features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_shap_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_shap_dependence.png")


def plot_shap_waterfall(clf_xgb, X_test, y_test):
    """SHAP waterfall for 4 representative cases."""
    import shap

    y_prob = clf_xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    cases = {}
    # Most confident correct human
    h_mask = (y_test == 0) & (y_pred == 0)
    if h_mask.any():
        cases['Correct Human\n(most confident)'] = np.where(h_mask)[0][np.argmin(y_prob[h_mask])]
    # Most confident correct AI
    a_mask = (y_test == 1) & (y_pred == 1)
    if a_mask.any():
        cases['Correct AI\n(most confident)'] = np.where(a_mask)[0][np.argmax(y_prob[a_mask])]
    # False positive (human → AI)
    fp_mask = (y_test == 0) & (y_pred == 1)
    if fp_mask.any():
        cases['False Positive\n(Human→AI)'] = np.where(fp_mask)[0][np.argmax(y_prob[fp_mask])]
    # False negative (AI → human)
    fn_mask = (y_test == 1) & (y_pred == 0)
    if fn_mask.any():
        cases['False Negative\n(AI→Human)'] = np.where(fn_mask)[0][np.argmin(y_prob[fn_mask])]

    if len(cases) < 2:
        print("  [SKIP] Not enough case types for waterfall")
        return

    explainer = shap.TreeExplainer(clf_xgb)
    n_cases = len(cases)
    fig, axes = plt.subplots(1, n_cases, figsize=(6 * n_cases, 8))
    if n_cases == 1:
        axes = [axes]

    for ax, (title, idx) in zip(axes, cases.items()):
        sv = explainer(X_test.iloc[[idx]])
        vals = sv.values[0]
        if vals.ndim == 2:
            vals = vals[:, 1]
        base = sv.base_values[0]
        if hasattr(base, '__len__'):
            base = base[1]

        top_k = 10
        sorted_idx = np.argsort(np.abs(vals))[::-1][:top_k]
        feat_names = [X_test.columns[i] for i in sorted_idx]
        feat_vals = [vals[i] for i in sorted_idx]
        feat_data = [X_test.iloc[idx, i] for i in sorted_idx]

        # Horizontal bar
        colors = ['#F44336' if v > 0 else '#2196F3' for v in feat_vals]
        y_pos = np.arange(top_k)[::-1]
        ax.barh(y_pos, feat_vals, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{n} = {d:.2f}' for n, d in zip(feat_names, feat_data)], fontsize=9)
        ax.axvline(0, color='gray', linewidth=0.5)

        prob = 1 / (1 + np.exp(-(base + sum(vals))))
        ax.set_title(f'{title}\nP(AI) = {prob:.4f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('SHAP value')

    plt.suptitle('RAID: SHAP Waterfall — Representative Cases', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_shap_waterfall.png")


def plot_feature_ablation(X_train, y_train, X_test, y_test):
    """Train XGBoost on each feature group independently."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    groups = {
        'basic_counts': ['word_count', 'char_count', 'sentence_count', 'paragraph_count'],
        'averages': ['avg_word_length', 'avg_sentence_length', 'words_per_paragraph',
                     'sentences_per_paragraph'],
        'variability': ['word_length_std', 'sentence_length_std'],
        'lexical_richness': ['type_token_ratio', 'hapax_ratio', 'yule_k',
                             'simpson_diversity', 'brunet_w', 'unique_word_ratio', 'long_word_ratio'],
        'punctuation': [c for c in X_train.columns if c.startswith('punct_')] +
                       ['uppercase_ratio', 'digit_ratio', 'whitespace_ratio'],
        'readability': ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                        'smog_index', 'coleman_liau', 'ari', 'dale_chall'],
        'structure': ['short_sentence_ratio'],
        'embedding_pca': [c for c in X_train.columns if c.startswith('emb_pca_')],
        'perplexity': ['gpt2_perplexity', 'gpt2_log_perplexity'],
    }

    results = {}
    for gname, feats in groups.items():
        feats = [f for f in feats if f in X_train.columns]
        if not feats:
            continue
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            eval_metric='logloss', random_state=42,
                            tree_method='hist', device='cuda')
        clf.fit(X_train[feats], y_train, verbose=False)
        y_prob = clf.predict_proba(X_test[feats])[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[gname] = {'auc': auc, 'n_feats': len(feats)}
        print(f"    {gname:20s} ({len(feats):2d} feats): AUC={auc:.4f}")

    # Plot
    sorted_groups = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    names = [f"{g} ({r['n_feats']})" for g, r in sorted_groups]
    aucs = [r['auc'] for _, r in sorted_groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.barh(names[::-1], aucs[::-1], color=colors)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel('ROC AUC (group trained alone)')
    ax.set_title('RAID: Feature Group Ablation Study')
    for i, v in enumerate(aucs[::-1]):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_feature_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_feature_ablation.png")
    return results


def plot_lr_coefficients(clf_lr, scaler, feature_names):
    """LR top-20 standardized coefficients."""
    coefs = clf_lr.coef_[0]
    top_idx = np.argsort(np.abs(coefs))[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    names = [feature_names[i] for i in top_idx]
    vals = [coefs[i] for i in top_idx]
    colors = ['#F44336' if v > 0 else '#2196F3' for v in vals]

    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, vals, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Standardized Coefficient')
    ax.set_title('RAID: LR Top-20 Feature Coefficients\n(+→AI, −→Human)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_lr_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_lr_coefficients.png")


# ════════════════════════════════════════════════════════════
# PART 2: Token Feature Analysis
# ════════════════════════════════════════════════════════════

def plot_token_shap(Xt_train, y_train, Xt_test, y_test):
    """SHAP summary for token features."""
    from xgboost import XGBClassifier
    import shap

    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist', device='cuda')
    clf.fit(Xt_train, y_train, verbose=False)

    n = min(1500, len(Xt_test))
    idx = np.random.RandomState(42).choice(len(Xt_test), n, replace=False)
    explainer = shap.TreeExplainer(clf)
    sv = explainer(Xt_test.iloc[idx])

    fig, ax = plt.subplots(figsize=(12, 8))
    sv_plot = sv
    if sv.values.ndim == 3:
        sv_plot = shap.Explanation(values=sv.values[:, :, 1],
                                   base_values=sv.base_values[:, 1] if sv.base_values.ndim > 1 else sv.base_values,
                                   data=sv.data, feature_names=sv.feature_names)
    shap.plots.beeswarm(sv_plot, show=False, max_display=20)
    plt.title('RAID: Token Feature SHAP Summary (Mistral-7B)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_token_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_token_shap.png")
    return clf


def plot_token_distributions(Xt_test, y_test):
    """Violin plots of top token features: Human vs AI."""
    top_feats = ['rank_top100_frac', 'rank_top1_frac', 'lp_mean', 'ent_mean',
                 'top1p_mean', 'rank_top10_frac']
    top_feats = [f for f in top_feats if f in Xt_test.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_df = Xt_test[top_feats].copy()
    plot_df['label'] = y_test[:len(plot_df)]
    plot_df['label_name'] = plot_df['label'].map({0: 'Human', 1: 'AI'})

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        sns.violinplot(data=plot_df, x='label_name', y=feat, ax=ax,
                       palette={'Human': '#2196F3', 'AI': '#F44336'},
                       inner='quartile', cut=0)
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.set_xlabel('')

    plt.suptitle('RAID: Token Feature Distributions (Human vs AI)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_token_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_token_distributions.png")


def plot_per_model_features(test_df, Xt_test):
    """Box plots of key token features per generator model."""
    if 'model' not in test_df.columns:
        return

    top_feats = ['rank_top100_frac', 'lp_mean', 'ent_mean', 'top1p_mean']
    top_feats = [f for f in top_feats if f in Xt_test.columns]

    plot_df = Xt_test[top_feats].copy()
    plot_df['model'] = test_df['model'].values[:len(plot_df)]

    models = sorted(plot_df['model'].unique())
    colors = {m: '#2196F3' if m == 'human' else '#F44336' for m in models}

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for i, feat in enumerate(top_feats[:4]):
        ax = axes[i]
        bp = ax.boxplot([plot_df[plot_df['model'] == m][feat].values for m in models],
                        labels=models, patch_artist=True, flierprops={'markersize': 2})
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[models[j]])
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)

    plt.suptitle('RAID: Token Features per Generator Model\n(Blue=Human, Red=AI)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_per_model_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_per_model_features.png")


# ════════════════════════════════════════════════════════════
# PART 3: Case-Level Visualization
# ════════════════════════════════════════════════════════════

def plot_token_heatmap(test_df):
    """Token probability heatmaps for 6 cases (3 human + 3 AI). Requires GPU."""
    from run_token_features import load_model
    import torch

    model, tokenizer, device = load_model()

    # Select 3 human + 3 AI samples with diverse domains
    humans = test_df[test_df['label'] == 0]
    ais = test_df[test_df['label'] == 1]

    h_samples = humans.groupby('domain').first().head(3) if 'domain' in humans.columns \
        else humans.sample(3, random_state=42)
    a_samples = ais.groupby('domain').first().head(3) if 'domain' in ais.columns \
        else ais.sample(3, random_state=42)

    samples = []
    for _, row in h_samples.iterrows():
        domain = row.get('domain', '?')
        samples.append((row['generation'], f'Human ({domain})', 0))
    for _, row in a_samples.iterrows():
        domain = row.get('domain', '?')
        m = row.get('model', '?')
        samples.append((row['generation'], f'AI: {m} ({domain})', 1))

    fig, axes = plt.subplots(len(samples), 2, figsize=(22, 4 * len(samples)))

    for i, (text, title, label) in enumerate(samples):
        try:
            tokens, token_lp, entropy, ranks = _get_token_details(text, model, tokenizer, device)
            _draw_heatmap(axes[i, 0], tokens, token_lp, f'{title}\nLog Probability',
                          cmap='RdYlGn', vmin=-15, vmax=0)
            _draw_heatmap(axes[i, 1], tokens, -np.log1p(ranks), f'{title}\nToken Rank (log)',
                          cmap='RdYlGn')
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[i, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')

    plt.suptitle('RAID: Token Probability Heatmaps', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_token_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_token_heatmap.png")

    del model
    torch.cuda.empty_cache()


def _get_token_details(text, model, tokenizer, device, max_length=512):
    """Per-token log prob, entropy, rank."""
    import torch
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    input_ids = enc['input_ids']

    with torch.no_grad():
        logits = model(**enc).logits

    pred_logits = logits[0, :-1, :]
    target_ids = input_ids[0, 1:]
    T = pred_logits.shape[0]

    log_probs = torch.log_softmax(pred_logits, dim=-1)
    token_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).float().cpu().numpy()

    probs = torch.softmax(pred_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).float().cpu().numpy()

    target_logit = pred_logits.gather(1, target_ids.unsqueeze(1))
    ranks = (pred_logits > target_logit).sum(dim=-1).float().cpu().numpy()

    tokens = [tokenizer.decode([tid]) for tid in input_ids[0, 1:].cpu().tolist()]
    return tokens, token_lp, entropy, ranks


def _draw_heatmap(ax, tokens, values, title, cmap='RdYlGn', vmin=None, vmax=None, max_tokens=80):
    """Draw colored token boxes."""
    tokens = tokens[:max_tokens]
    values = values[:max_tokens]

    if vmin is None:
        vmin = np.percentile(values, 5)
    if vmax is None:
        vmax = np.percentile(values, 95)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.cm.get_cmap(cmap)

    x, y = 0.02, 0.85
    line_height = 0.18
    max_x = 0.98

    for tok, val in zip(tokens, values):
        color = colormap(norm(val))
        display = tok.replace('\n', '\\n')
        if len(display) > 12:
            display = display[:10] + '..'
        text_width = len(display) * 0.012 + 0.015
        if x + text_width > max_x:
            x = 0.02
            y -= line_height
            if y < 0.02:
                break

        r, g, b, _ = color
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        txt_color = 'white' if lum < 0.5 else 'black'

        box = FancyBboxPatch((x, y - 0.06), text_width - 0.003, 0.12,
                             boxstyle="round,pad=0.02", facecolor=color,
                             edgecolor='gray', linewidth=0.3)
        ax.add_patch(box)
        ax.text(x + text_width / 2 - 0.002, y, display,
                fontsize=7, ha='center', va='center', color=txt_color,
                fontfamily='monospace')
        x += text_width

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')


def plot_token_tsne(Xt_test, y_test):
    """t-SNE of 30-dim token features."""
    from sklearn.manifold import TSNE

    n = min(3000, len(Xt_test))
    idx = np.random.RandomState(42).choice(len(Xt_test), n, replace=False)
    X_sub = Xt_test.iloc[idx].values
    y_sub = y_test[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color, name in [(0, '#2196F3', 'Human'), (1, '#F44336', 'AI')]:
        mask = y_sub == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.3, s=10, label=name)
    ax.set_title('RAID: t-SNE of Token Features (Mistral-7B)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_token_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_token_tsne.png")


def plot_misclassification(Xt_train, y_train, Xt_test, y_test, test_df):
    """Misclassification analysis: confidence histogram + feature comparison + pie."""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist', device='cuda')
    clf.fit(Xt_train, y_train, verbose=False)
    y_prob = clf.predict_proba(Xt_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    tp = ((y_test == 1) & (y_pred == 1)).sum()
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Confidence histogram
    ax = axes[0]
    for label, name, color in [(0, 'Human (correct)', '#2196F3'), (1, 'AI (correct)', '#F44336')]:
        mask = (y_test == label) & (y_pred == label)
        ax.hist(y_prob[mask], bins=50, alpha=0.5, label=name, color=color, density=True)
    mis_mask = y_pred != y_test
    if mis_mask.any():
        ax.hist(y_prob[mis_mask], bins=30, alpha=0.7, label='Misclassified',
                color='orange', density=True, histtype='step', linewidth=2)
    ax.set_xlabel('P(AI)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()

    # 2. Feature comparison: correct vs misclassified
    ax = axes[1]
    top_feats = ['rank_top100_frac', 'lp_mean', 'ent_mean', 'top1p_mean', 'rank_top1_frac']
    top_feats = [f for f in top_feats if f in Xt_test.columns][:5]

    correct_mask = y_pred == y_test
    wrong_mask = ~correct_mask

    if wrong_mask.sum() > 0:
        x_pos = np.arange(len(top_feats))
        correct_means = [Xt_test[f].iloc[correct_mask].mean() for f in top_feats]
        wrong_means = [Xt_test[f].iloc[wrong_mask].mean() for f in top_feats]
        ax.bar(x_pos - 0.2, correct_means, 0.35, label='Correct', color='#4CAF50')
        ax.bar(x_pos + 0.2, wrong_means, 0.35, label='Misclassified', color='#FF9800')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_feats, rotation=30, ha='right', fontsize=9)
        ax.set_title('Feature Means: Correct vs Misclassified')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No misclassifications!', ha='center', va='center')

    # 3. Pie chart
    ax = axes[2]
    sizes = [tp, tn, fp, fn]
    labels_pie = [f'TP (AI→AI)\n{tp}', f'TN (H→H)\n{tn}',
                  f'FP (H→AI)\n{fp}', f'FN (AI→H)\n{fn}']
    colors_pie = ['#F44336', '#2196F3', '#FF9800', '#9C27B0']
    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Classification Breakdown\nAcc={accuracy_score(y_test, y_pred):.4f}')

    plt.suptitle('RAID: Misclassification Analysis (Token Features)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_misclassification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_misclassification.png")


# ════════════════════════════════════════════════════════════
# PART 4: RAID-Specific Analysis
# ════════════════════════════════════════════════════════════

def plot_attack_feature_shift():
    """Compare top feature distributions before vs after paraphrase attack."""
    adv_path = DATA_DIR / "train.csv"
    if not adv_path.exists():
        print("  [SKIP] train.csv not found for attack analysis")
        return

    print("  Loading adversarial data for feature shift analysis...")
    df_all = pd.read_csv(adv_path)
    df_all['label'] = (df_all['model'] != 'human').astype(int)

    # Get original (none) and paraphrased AI text
    ai_original = df_all[(df_all['attack'] == 'none') & (df_all['label'] == 1)]
    ai_paraphrase = df_all[(df_all['attack'] == 'paraphrase') & (df_all['label'] == 1)]
    ai_zero_width = df_all[(df_all['attack'] == 'zero_width_space') & (df_all['label'] == 1)]
    human = df_all[df_all['model'] == 'human']

    n = min(3000, len(ai_original), len(ai_paraphrase), len(human))
    ai_orig_sub = ai_original.sample(n, random_state=42)
    ai_para_sub = ai_paraphrase.sample(n, random_state=42)
    human_sub = human.sample(n, random_state=42)

    print("  Extracting features for shift analysis...")
    feat_orig = extract_features_batch(ai_orig_sub['generation'].tolist(), "orig")
    feat_para = extract_features_batch(ai_para_sub['generation'].tolist(), "para")
    feat_human = extract_features_batch(human_sub['generation'].tolist(), "human")

    top_feats = ['type_token_ratio', 'avg_sentence_length', 'flesch_reading_ease',
                 'hapax_ratio', 'word_count', 'word_length_std']
    top_feats = [f for f in top_feats if f in feat_orig.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        data = [human_sub[feat].values if feat in human_sub.columns else feat_human[feat].values,
                feat_orig[feat].values, feat_para[feat].values]

        # Use feature from extracted dataframes
        ax.violinplot([feat_human[feat].values, feat_orig[feat].values, feat_para[feat].values],
                      positions=[0, 1, 2], showmeans=True, showmedians=True)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Human', 'AI (original)', 'AI (paraphrased)'], fontsize=9)
        ax.set_title(feat, fontsize=11, fontweight='bold')

    plt.suptitle('RAID: Feature Distribution Shift under Paraphrase Attack\n'
                 'Paraphrase makes AI text look more like human text',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_attack_feature_shift.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_attack_feature_shift.png")


def plot_decoding_comparison(test_df, X_test):
    """Compare feature distributions across decoding strategies."""
    if 'decoding' not in test_df.columns:
        print("  [SKIP] No 'decoding' column")
        return

    ai_mask = test_df['label'].values[:len(X_test)] == 1
    plot_df = X_test[ai_mask].copy()
    plot_df['decoding'] = test_df.loc[test_df.index[:len(X_test)][ai_mask], 'decoding'].values

    # Drop NaN decoding
    plot_df = plot_df.dropna(subset=['decoding'])
    if plot_df.empty:
        print("  [SKIP] No decoding info available")
        return

    top_feats = ['type_token_ratio', 'avg_sentence_length', 'gpt2_perplexity', 'hapax_ratio']
    top_feats = [f for f in top_feats if f in plot_df.columns][:4]

    fig, axes = plt.subplots(1, len(top_feats), figsize=(5 * len(top_feats), 6))
    if len(top_feats) == 1:
        axes = [axes]

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        decodings = sorted(plot_df['decoding'].unique())
        bp = ax.boxplot([plot_df[plot_df['decoding'] == d][feat].values for d in decodings],
                        labels=[str(d)[:15] for d in decodings], patch_artist=True,
                        flierprops={'markersize': 2})
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)

    plt.suptitle('RAID: Feature Comparison across Decoding Strategies (AI text only)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_decoding_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_decoding_comparison.png")


def plot_model_domain_heatmap(test_df, X_test, clf_xgb):
    """Heatmap of AUC: models × domains."""
    from sklearn.metrics import roc_auc_score

    if 'model' not in test_df.columns or 'domain' not in test_df.columns:
        return

    y_test = test_df['label'].values[:len(X_test)]
    y_prob = clf_xgb.predict_proba(X_test)[:, 1]

    models = sorted(test_df['model'].dropna().unique())
    domains = sorted(test_df['domain'].dropna().unique())

    # For each model × domain, compute accuracy on AI detection
    acc_matrix = np.full((len(models), len(domains)), np.nan)

    for i, m in enumerate(models):
        for j, d in enumerate(domains):
            mask = (test_df['model'].values[:len(X_test)] == m) & \
                   (test_df['domain'].values[:len(X_test)] == d)
            if mask.sum() < 5:
                continue
            acc_matrix[i, j] = (y_prob[mask] > 0.5).mean() if m != 'human' \
                else (y_prob[mask] <= 0.5).mean()

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(domains)):
            if not np.isnan(acc_matrix[i, j]):
                ax.text(j, i, f'{acc_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=9, color='black' if acc_matrix[i, j] > 0.6 else 'white')

    plt.colorbar(im, ax=ax, label='Detection Accuracy', shrink=0.8)
    ax.set_title('RAID: Detection Accuracy by Model × Domain (XGBoost)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_model_domain_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: raid_model_domain_heatmap.png")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    print("="*60)
    print("RAID Deep Analysis & Interpretability")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data and features...")
    train_df, test_df, X_train, X_test = load_features_and_split()
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # ── PART 1: 90-Feature Analysis ──
    print("\n[2/4] 90-Feature Analysis...")

    print("  Correlation heatmap...")
    plot_correlation_heatmap(X_train)

    print("  Domain comparison...")
    plot_domain_comparison(test_df, X_test)

    print("  PCA...")
    plot_pca(X_test, y_test)

    print("  t-SNE...")
    plot_tsne(X_test, y_test)

    print("  Training models + SHAP...")
    clf_xgb, clf_lr, scaler, shap_values, shap_idx, y_prob_xgb = \
        train_models_and_shap(X_train, y_train, X_test, y_test)

    print("  SHAP summary...")
    plot_shap_summary(shap_values)

    print("  SHAP dependence...")
    plot_shap_dependence(shap_values, X_test.iloc[shap_idx])

    print("  SHAP waterfall...")
    plot_shap_waterfall(clf_xgb, X_test, y_test)

    print("  Feature ablation...")
    plot_feature_ablation(X_train, y_train, X_test, y_test)

    print("  LR coefficients...")
    plot_lr_coefficients(clf_lr, scaler, X_train.columns.tolist())

    # ── PART 2: Token Feature Analysis ──
    print("\n[3/4] Token Feature Analysis...")
    try:
        Xt_train, Xt_test = load_token_features()

        print("  Token SHAP...")
        plot_token_shap(Xt_train, y_train, Xt_test, y_test)

        print("  Token distributions...")
        plot_token_distributions(Xt_test, y_test)

        print("  Per-model features...")
        plot_per_model_features(test_df, Xt_test)

        print("  Token t-SNE...")
        plot_token_tsne(Xt_test, y_test)

        print("  Misclassification analysis...")
        plot_misclassification(Xt_train, y_train, Xt_test, y_test, test_df)
    except FileNotFoundError as e:
        print(f"  [SKIP] Token features not available: {e}")

    # ── PART 3: Case-Level Heatmap (GPU) ──
    print("\n  Token heatmaps (GPU)...")
    try:
        plot_token_heatmap(test_df)
    except Exception as e:
        print(f"  [SKIP] Token heatmap failed: {e}")

    # ── PART 4: RAID-Specific ──
    print("\n[4/4] RAID-Specific Analysis...")

    print("  Model × Domain heatmap...")
    plot_model_domain_heatmap(test_df, X_test, clf_xgb)

    print("  Decoding strategy comparison...")
    plot_decoding_comparison(test_df, X_test)

    print("  Adversarial feature shift (paraphrase)...")
    try:
        plot_attack_feature_shift()
    except Exception as e:
        print(f"  [SKIP] Attack feature shift: {e}")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! Generated figures in {total/60:.1f} min")
    print(f"{'='*60}")
