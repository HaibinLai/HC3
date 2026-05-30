"""
SHAP Attribution Analysis — Generate figures for PPT归因分析核心.

Generates:
  1. shap_waterfall_118.png   — Single AI sample waterfall (top-15 features)
  2. shap_group_bar.png       — Mean |SHAP| per feature group
  3. feature_human_vs_ai.png  — 6-panel KDE for key features (Human vs AI)
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
import shap

from run_raid_120 import load_combined
from auto_filter import GROUPS

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
SEED = 42

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# Token feature group (not in GROUPS)
TOKEN_GROUP = {
    'token_prob': [f'rank_top{k}_frac' for k in [1,5,10]] +
                  [f'lp_{s}' for s in ['mean','std','min','max','range']] +
                  [f'ent_{s}' for s in ['mean','std','min','max','range']] +
                  [f'rank_{s}' for s in ['mean','std','min','max','range']] +
                  ['lp_q25','lp_q75','ent_q25','ent_q75','rank_q25','rank_q75',
                   'lp_iqr','ent_iqr','rank_iqr','frac_low_prob'],
}

ALL_GROUPS = {**GROUPS, **TOKEN_GROUP}

# Layer mapping
LAYER_COLORS = {
    'basic_counts': '#2196F3',      # Model-Free
    'averages': '#2196F3',
    'variability': '#2196F3',
    'lexical_richness': '#2196F3',
    'punctuation': '#2196F3',
    'readability': '#2196F3',
    'structure': '#2196F3',
    'embedding_pca': '#FF9800',     # Model-Based
    'perplexity': '#FF9800',
    'token_prob': '#F44336',        # Model-Based (token)
}


def main():
    print("=" * 60)
    print("SHAP Attribution Analysis")
    print("=" * 60)

    # Load data and train model
    print("\n[1/4] Loading data...")
    train_df, test_df, X_tr, X_te, y_tr, y_te, *_ = load_combined()
    X_tr = X_tr.loc[:, ~X_tr.columns.duplicated()]
    X_te = X_te.loc[:, ~X_te.columns.duplicated()]
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

    print("\n[2/4] Training XGBoost...")
    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='logloss', random_state=SEED,
                        tree_method='hist', device='cuda')
    clf.fit(X_tr, y_tr)
    pred_proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, pred_proba)
    print(f"  AUC = {auc:.4f}")

    # SHAP
    print("\n[3/4] Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    n_shap = min(3000, len(X_te))
    X_shap = X_te.iloc[:n_shap]
    sv = explainer(X_shap)

    # ── Figure 1: Waterfall ──
    print("  Generating waterfall...")
    # Find a confident AI prediction
    ai_indices = np.where(y_te[:n_shap] == 1)[0]
    ai_probs = pred_proba[:n_shap][ai_indices]
    best_ai = ai_indices[np.argmax(ai_probs)]

    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(sv[best_ai], max_display=15, show=False)
    plt.title('SHAP Waterfall: AI Sample Attribution (Top 15)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shap_waterfall_118.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_waterfall_118.png")

    # Print top features for this sample
    sample_sv = sv[best_ai]
    feat_shap = list(zip(X_shap.columns, sample_sv.values))
    feat_shap.sort(key=lambda x: abs(x[1]), reverse=True)
    print("  Top-10 SHAP for this AI sample:")
    for fname, sval in feat_shap[:10]:
        print(f"    {fname:30s}  SHAP = {sval:+.4f}")

    # ── Figure 2: Group Bar ──
    print("  Generating group bar chart...")
    mean_abs_shap = np.abs(sv.values).mean(axis=0)
    feat_importance = dict(zip(X_shap.columns, mean_abs_shap))

    group_shap = {}
    for gname, gfeats in ALL_GROUPS.items():
        avail = [f for f in gfeats if f in feat_importance]
        if avail:
            group_shap[gname] = sum(feat_importance[f] for f in avail)

    # Sort by importance
    sorted_groups = sorted(group_shap.items(), key=lambda x: x[1], reverse=True)
    gnames = [g[0] for g in sorted_groups]
    gvals = [g[1] for g in sorted_groups]
    total = sum(gvals)
    gcolors = [LAYER_COLORS.get(g, '#888888') for g in gnames]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(gnames)-1, -1, -1), gvals, color=gcolors, alpha=0.85)
    ax.set_yticks(range(len(gnames)-1, -1, -1))
    ax.set_yticklabels(gnames, fontsize=11)
    ax.set_xlabel('Sum of mean |SHAP|', fontsize=12)
    ax.set_title('Feature Group Attribution (RAID 118-dim XGBoost)', fontsize=14)

    # Add percentage labels
    for i, (g, v) in enumerate(sorted_groups):
        pct = v / total * 100
        ax.text(v + total * 0.005, len(gnames)-1-i, f'{pct:.1f}%', va='center', fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Model-Free'),
        Patch(facecolor='#FF9800', label='Model-Based (embed/ppl)'),
        Patch(facecolor='#F44336', label='Model-Based (token)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shap_group_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_group_bar.png")

    # Print group contributions
    print("  Group contributions:")
    for g, v in sorted_groups:
        layer = 'MF' if LAYER_COLORS.get(g) == '#2196F3' else 'MB'
        print(f"    [{layer}] {g:20s}  |SHAP| = {v:.4f}  ({v/total*100:.1f}%)")

    # ── Figure 3: Human vs AI Feature Distributions ──
    print("  Generating feature distributions...")
    key_features = [
        ('type_token_ratio', 'Type-Token Ratio'),
        ('flesch_reading_ease', 'Flesch Reading Ease'),
        ('sentence_length_std', 'Sentence Length Std'),
        ('words_per_paragraph', 'Words per Paragraph'),
        ('avg_sentence_length', 'Avg Sentence Length'),
        ('short_sentence_ratio', 'Short Sentence Ratio'),
    ]

    # Combine train+test for visualization
    X_all = pd.concat([X_tr, X_te], axis=0).reset_index(drop=True)
    y_all = np.concatenate([y_tr, y_te])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (feat, title) in enumerate(key_features):
        ax = axes[idx]
        if feat not in X_all.columns:
            ax.set_title(f'{title}\n(not available)', fontsize=12)
            continue

        vals = X_all[feat].values
        human_vals = vals[y_all == 0]
        ai_vals = vals[y_all == 1]

        # Remove outliers for visualization
        for arr in [human_vals, ai_vals]:
            q1, q99 = np.percentile(arr, [1, 99])
            arr[(arr < q1) | (arr > q99)] = np.nan

        # KDE plot
        sns.kdeplot(human_vals[~np.isnan(human_vals)], ax=ax, label='Human',
                    color='#2196F3', fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(ai_vals[~np.isnan(ai_vals)], ax=ax, label='AI',
                    color='#F44336', fill=True, alpha=0.3, linewidth=2)

        # Compute single-feature AUC
        try:
            auc_f = roc_auc_score(y_all, X_all[feat].values)
            if auc_f < 0.5:
                auc_f = 1 - auc_f
        except:
            auc_f = 0.5
        ax.set_title(f'{title}\nAUC = {auc_f:.3f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylabel('')

    plt.suptitle('Key Features: Human vs AI Distribution (RAID)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'feature_human_vs_ai.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: feature_human_vs_ai.png")

    print("\n[4/4] Done! Generated 3 figures.")


if __name__ == '__main__':
    main()
