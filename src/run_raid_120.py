"""
RAID — 118 Combined Features Analysis
Merges 88 handcrafted + 30 token probability features, then runs:
  1. XGBoost / LR / RF comparison (120 vs 90 vs 30)
  2. SHAP summary + dependence (118 combined)
  3. Feature group ablation (11 groups now)
  4. Per-generator accuracy improvement
  5. Cross-dataset validation (HC3, SemEval, TuringBench, Pile)
  6. Adversarial robustness with 120 features
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
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

from run_raid_analysis import load_features_and_split, load_token_features

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "processed"
SEED = 42

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


def load_combined():
    """Load 88 handcrafted + 30 token features, merge, return aligned data."""
    train_df, test_df, X_tr_90, X_te_90 = load_features_and_split()
    tok_tr, tok_te = load_token_features()

    n_tr = min(len(X_tr_90), len(tok_tr), len(train_df))
    n_te = min(len(X_te_90), len(tok_te), len(test_df))

    X_tr = pd.concat([X_tr_90.iloc[:n_tr].reset_index(drop=True),
                       tok_tr.iloc[:n_tr].reset_index(drop=True)], axis=1)
    X_te = pd.concat([X_te_90.iloc[:n_te].reset_index(drop=True),
                       tok_te.iloc[:n_te].reset_index(drop=True)], axis=1)

    train_df = train_df.iloc[:n_tr].reset_index(drop=True)
    test_df = test_df.iloc[:n_te].reset_index(drop=True)
    y_tr = train_df['label'].values
    y_te = test_df['label'].values

    # Also return the sub-feature sets for ablation
    return (train_df, test_df, X_tr, X_te, y_tr, y_te,
            X_tr_90.iloc[:n_tr].reset_index(drop=True),
            X_te_90.iloc[:n_te].reset_index(drop=True),
            tok_tr.iloc[:n_tr].reset_index(drop=True),
            tok_te.iloc[:n_te].reset_index(drop=True))


def run_method_comparison(X_tr, X_te, y_tr, y_te, X_tr_90, X_te_90, tok_tr, tok_te):
    """Compare 120 vs 90 vs 30 features across 3 classifiers."""
    results = []
    configs = [
        ('XGBoost', '118 combined', X_tr, X_te),
        ('XGBoost', '88 handcrafted', X_tr_90, X_te_90),
        ('XGBoost', '30 token', tok_tr, tok_te),
        ('LR', '118 combined', X_tr, X_te),
        ('RF', '118 combined', X_tr, X_te),
    ]
    clfs = {}
    for clf_name, feat_name, Xtr, Xte in configs:
        if clf_name == 'XGBoost':
            clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric='logloss', random_state=SEED,
                                tree_method='hist', device='cuda')
            clf.fit(Xtr, y_tr, verbose=False)
        elif clf_name == 'LR':
            sc = StandardScaler()
            Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns)
            Xte = pd.DataFrame(sc.transform(Xte), columns=Xte.columns)
            clf = LogisticRegression(max_iter=2000, random_state=SEED)
            clf.fit(Xtr_s, y_tr)
            Xtr = Xtr_s
        else:
            clf = RandomForestClassifier(n_estimators=300, max_depth=15,
                                         random_state=SEED, n_jobs=-1)
            clf.fit(Xtr, y_tr)

        prob = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        auc = roc_auc_score(y_te, prob)
        acc = accuracy_score(y_te, pred)
        results.append({'Classifier': clf_name, 'Features': feat_name,
                        'AUC': auc, 'Accuracy': acc})
        key = f"{clf_name}_{feat_name}"
        clfs[key] = (clf, prob, pred)
        print(f"  {clf_name:8s} | {feat_name:16s} | AUC={auc:.4f} | Acc={acc:.4f}")

    return pd.DataFrame(results), clfs


def plot_comparison_bar(results_df):
    """Bar chart comparing all methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # AUC
    ax = axes[0]
    colors = ['#F44336' if '120' in r else '#2196F3' if '90' in r else '#4CAF50'
              for r in results_df['Features']]
    bars = ax.barh(range(len(results_df)-1, -1, -1), results_df['AUC'].values, color=colors)
    ax.set_yticks(range(len(results_df)-1, -1, -1))
    ax.set_yticklabels([f"{r['Classifier']} ({r['Features']})" for _, r in results_df.iterrows()], fontsize=9)
    ax.set_xlabel('AUC')
    ax.set_xlim(0.95, 1.001)
    ax.set_title('AUC Comparison')
    for i, v in enumerate(results_df['AUC'].values):
        ax.text(v + 0.0005, len(results_df)-1-i, f'{v:.4f}', va='center', fontsize=9)

    # Accuracy
    ax = axes[1]
    ax.barh(range(len(results_df)-1, -1, -1), results_df['Accuracy'].values, color=colors)
    ax.set_yticks(range(len(results_df)-1, -1, -1))
    ax.set_yticklabels([f"{r['Classifier']} ({r['Features']})" for _, r in results_df.iterrows()], fontsize=9)
    ax.set_xlabel('Accuracy')
    ax.set_xlim(0.93, 1.001)
    ax.set_title('Accuracy Comparison')
    for i, v in enumerate(results_df['Accuracy'].values):
        ax.text(v + 0.0005, len(results_df)-1-i, f'{v:.4f}', va='center', fontsize=9)

    plt.suptitle('RAID: 118 Combined vs 88 Handcrafted vs 30 Token Features', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_120_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: raid_120_comparison.png")


def plot_shap_120(clf, X_te):
    """SHAP summary for 118 combined features."""
    explainer = shap.TreeExplainer(clf)
    n = min(3000, len(X_te))
    X_shap = X_te.iloc[:n]
    sv = explainer.shap_values(X_shap)

    # Summary
    fig = plt.figure(figsize=(10, 12))
    shap.summary_plot(sv, X_shap, max_display=30, show=False)
    plt.title('RAID: SHAP Feature Importance (118 Combined Features, Top 30)', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_120_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: raid_120_shap_summary.png")

    # Dependence top 6
    mean_abs = np.abs(sv).mean(axis=0)
    top6 = np.argsort(mean_abs)[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, fi in zip(axes.flat, top6):
        shap.dependence_plot(fi, sv, X_shap, ax=ax, show=False)
        ax.set_xlabel(X_shap.columns[fi], fontsize=10)
        ax.set_ylabel('SHAP value', fontsize=10)
    plt.suptitle('RAID: SHAP Dependence (118 Combined, Top 6)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_120_shap_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: raid_120_shap_dependence.png")

    return sv, X_shap


def plot_ablation_11groups(X_tr, X_te, y_tr, y_te):
    """Ablation with 11 groups (9 handcrafted + token_logprob + token_rank_entropy)."""
    groups = {
        'basic_counts': ['word_count', 'char_count', 'sentence_count', 'paragraph_count'],
        'averages': ['avg_word_length', 'avg_sentence_length', 'words_per_paragraph',
                     'sentences_per_paragraph'],
        'variability': ['word_length_std', 'sentence_length_std'],
        'lexical_richness': ['type_token_ratio', 'hapax_ratio', 'yule_k',
                             'simpson_diversity', 'brunet_w', 'unique_word_ratio', 'long_word_ratio'],
        'punctuation': [c for c in X_tr.columns if c.startswith('punct_')] +
                       ['uppercase_ratio', 'digit_ratio', 'whitespace_ratio'],
        'readability': ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                        'smog_index', 'coleman_liau', 'ari', 'dale_chall'],
        'structure': ['short_sentence_ratio'],
        'embedding_pca': [c for c in X_tr.columns if c.startswith('emb_pca_')],
        'perplexity': ['gpt2_perplexity', 'gpt2_log_perplexity'],
        'token_logprob': [c for c in X_tr.columns if c.startswith('logprob_')],
        'token_rank_entropy': [c for c in X_tr.columns if c.startswith('rank_') or c.startswith('entropy_')],
    }

    results = {}
    for gname, feats in groups.items():
        feats = [f for f in feats if f in X_tr.columns]
        if not feats:
            continue
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            eval_metric='logloss', random_state=SEED,
                            tree_method='hist', device='cuda')
        clf.fit(X_tr[feats], y_tr, verbose=False)
        prob = clf.predict_proba(X_te[feats])[:, 1]
        auc = roc_auc_score(y_te, prob)
        results[gname] = {'auc': auc, 'n_feats': len(feats)}
        print(f"    {gname:22s} ({len(feats):2d} feats): AUC={auc:.4f}")

    # Plot
    sorted_g = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    names = [f"{g} ({r['n_feats']})" for g, r in sorted_g]
    aucs = [r['auc'] for _, r in sorted_g]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#F44336' if 'token' in g else '#2196F3' for g, _ in sorted_g]
    ax.barh(names[::-1], aucs[::-1], color=colors[::-1])
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel('ROC AUC (group trained alone)')
    ax.set_title('RAID: 11-Group Feature Ablation (red=token, blue=handcrafted)')
    for i, v in enumerate(aucs[::-1]):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_120_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: raid_120_ablation.png")
    return results


def plot_per_generator_improvement(test_df, y_te, pred_90, pred_120):
    """Compare per-generator accuracy: 90 vs 120 features."""
    models = [m for m in test_df['model'].unique() if m != 'human']
    records = []
    for m in sorted(models):
        mask = test_df['model'] == m
        if mask.sum() < 10:
            continue
        acc_90 = accuracy_score(y_te[mask], pred_90[mask])
        acc_120 = accuracy_score(y_te[mask], pred_120[mask])
        records.append({'Generator': m, 'Acc_90': acc_90, 'Acc_120': acc_120,
                        'Improvement': acc_120 - acc_90})

    df = pd.DataFrame(records).sort_values('Improvement', ascending=False)
    print("\n  Per-Generator Accuracy Improvement (120 vs 90):")
    for _, r in df.iterrows():
        print(f"    {r['Generator']:20s}  90={r['Acc_90']:.3f}  120={r['Acc_120']:.3f}  Δ={r['Improvement']:+.3f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df['Acc_90'], w, label='88 handcrafted', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, df['Acc_120'], w, label='118 combined', color='#F44336', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Generator'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('RAID: Per-Generator Detection — 88 vs 118 Features')
    ax.set_ylim(0.3, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_120_per_generator.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: raid_120_per_generator.png")
    return df


def run_cross_dataset_120():
    """Run 118 combined features on HC3, SemEval, TuringBench, Pile."""
    results = []

    # HC3
    print("  HC3...")
    hc3_feat = pd.read_csv(DATA_DIR / 'hc3_extended_features.csv')
    hc3_tok_tr = pd.read_csv(DATA_DIR / 'token_features_hc3_train.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
    hc3_tok_te = pd.read_csv(DATA_DIR / 'token_features_hc3_test.csv').replace([np.inf,-np.inf],np.nan).fillna(0)

    feat_cols = [c for c in hc3_feat.columns if c not in ['label', 'label_name', 'source']]
    from sklearn.model_selection import train_test_split
    hc3_tr_idx, hc3_te_idx = train_test_split(hc3_feat.index, test_size=0.2,
                                               stratify=hc3_feat['label'], random_state=SEED)

    Xtr_90 = hc3_feat.loc[hc3_tr_idx][feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0).reset_index(drop=True)
    Xte_90 = hc3_feat.loc[hc3_te_idx][feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0).reset_index(drop=True)
    ytr = hc3_feat.loc[hc3_tr_idx]['label'].values
    yte = hc3_feat.loc[hc3_te_idx]['label'].values

    n_tr = min(len(Xtr_90), len(hc3_tok_tr))
    n_te = min(len(Xte_90), len(hc3_tok_te))
    Xtr = pd.concat([Xtr_90.iloc[:n_tr], hc3_tok_tr.iloc[:n_tr].reset_index(drop=True)], axis=1)
    Xte = pd.concat([Xte_90.iloc[:n_te], hc3_tok_te.iloc[:n_te].reset_index(drop=True)], axis=1)

    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        eval_metric='logloss', random_state=SEED, tree_method='hist', device='cuda')
    clf.fit(Xtr, ytr[:n_tr], verbose=False)
    auc = roc_auc_score(yte[:n_te], clf.predict_proba(Xte)[:,1])
    acc = accuracy_score(yte[:n_te], clf.predict(Xte))
    results.append({'Dataset': 'HC3', 'AUC_90': 0.9999, 'AUC_120': auc, 'Acc_120': acc})
    print(f"    HC3: AUC={auc:.4f}")

    # SemEval
    print("  SemEval...")
    se_tr_90 = pd.read_csv(DATA_DIR / 'semeval_features_train.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
    se_te_90 = pd.read_csv(DATA_DIR / 'semeval_features_test.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
    se_tok_tr = pd.read_csv(DATA_DIR / 'token_features_semeval_train.csv').replace([np.inf,-np.inf],np.nan).fillna(0) if (DATA_DIR / 'token_features_semeval_train.csv').exists() else None

    se_dir = ROOT / 'data' / 'external' / 'semeval2024_task8' / 'subtaskA_monolingual'
    se_train = pd.read_parquet(se_dir / 'train-00000-of-00001.parquet')
    se_test = pd.read_parquet(se_dir / 'test-00000-of-00001.parquet')

    if se_tok_tr is not None and len(se_tok_tr) > 100:
        se_tok_te = pd.read_csv(DATA_DIR / 'token_features_semeval_test.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
        n_tr = min(len(se_tr_90), len(se_tok_tr), len(se_train))
        n_te = min(len(se_te_90), len(se_tok_te), len(se_test))
        ytr_se = se_train['label'].values[:n_tr]
        yte_se = se_test['label'].values[:n_te]
        # Skip if only one class present
        if len(set(ytr_se)) < 2 or len(set(yte_se)) < 2:
            auc, acc = 0, 0
        else:
            Xtr = pd.concat([se_tr_90.iloc[:n_tr].reset_index(drop=True),
                              se_tok_tr.iloc[:n_tr].reset_index(drop=True)], axis=1)
            Xte = pd.concat([se_te_90.iloc[:n_te].reset_index(drop=True),
                              se_tok_te.iloc[:n_te].reset_index(drop=True)], axis=1)
            common = [c for c in Xtr.columns if c in Xte.columns]
            Xtr, Xte = Xtr[common], Xte[common]
            clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                                eval_metric='logloss', random_state=SEED, tree_method='hist', device='cuda')
            clf.fit(Xtr, ytr_se, verbose=False)
            auc = roc_auc_score(yte_se, clf.predict_proba(Xte)[:,1])
            acc = accuracy_score(yte_se, clf.predict(Xte))
    else:
        auc, acc = 0, 0
    results.append({'Dataset': 'SemEval', 'AUC_90': 0.6872, 'AUC_120': auc, 'Acc_120': acc})
    print(f"    SemEval: AUC={auc:.4f}")

    # TuringBench
    print("  TuringBench...")
    tb_tr_90 = pd.read_csv(DATA_DIR / 'turingbench_features_train.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
    tb_te_90 = pd.read_csv(DATA_DIR / 'turingbench_features_test.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
    tb_tok_tr_path = DATA_DIR / 'token_features_turingbench_train.csv'

    from run_turingbench import load_turingbench
    tb_train, tb_test = load_turingbench()

    if tb_tok_tr_path.exists():
        tb_tok_tr = pd.read_csv(tb_tok_tr_path).replace([np.inf,-np.inf],np.nan).fillna(0)
        tb_tok_te = pd.read_csv(DATA_DIR / 'token_features_turingbench_test.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
        n_tr = min(len(tb_tr_90), len(tb_tok_tr), len(tb_train))
        n_te = min(len(tb_te_90), len(tb_tok_te), len(tb_test))
        ytr_tb = tb_train['binary_label'].values[:n_tr]
        yte_tb = tb_test['binary_label'].values[:n_te]
        if len(set(ytr_tb)) < 2 or len(set(yte_tb)) < 2:
            auc, acc = 0, 0
        else:
            Xtr = pd.concat([tb_tr_90.iloc[:n_tr].reset_index(drop=True),
                              tb_tok_tr.iloc[:n_tr].reset_index(drop=True)], axis=1)
            Xte = pd.concat([tb_te_90.iloc[:n_te].reset_index(drop=True),
                              tb_tok_te.iloc[:n_te].reset_index(drop=True)], axis=1)
            common = [c for c in Xtr.columns if c in Xte.columns]
            Xtr, Xte = Xtr[common], Xte[common]
            clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                                eval_metric='logloss', random_state=SEED, tree_method='hist', device='cuda')
            clf.fit(Xtr, ytr_tb, verbose=False)
            auc = roc_auc_score(yte_tb, clf.predict_proba(Xte)[:,1])
            acc = accuracy_score(yte_tb, clf.predict(Xte))
    else:
        auc, acc = 0, 0
    results.append({'Dataset': 'TuringBench', 'AUC_90': 0.9841, 'AUC_120': auc, 'Acc_120': acc})
    print(f"    TuringBench: AUC={auc:.4f}")

    # Pile
    print("  Pile...")
    pile_tr_90 = pd.read_csv(DATA_DIR / 'pile_features_train.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
    pile_te_90 = pd.read_csv(DATA_DIR / 'pile_features_test.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
    pile_tok_tr_path = DATA_DIR / 'token_features_pile_train.csv'

    from run_pile import load_pile
    pile_train, pile_test = load_pile(max_per_class=50000)

    if pile_tok_tr_path.exists():
        pile_tok_tr = pd.read_csv(pile_tok_tr_path).replace([np.inf,-np.inf],np.nan).fillna(0)
        pile_tok_te = pd.read_csv(DATA_DIR / 'token_features_pile_test.csv').replace([np.inf,-np.inf],np.nan).fillna(0)
        n_tr = min(len(pile_tr_90), len(pile_tok_tr), len(pile_train))
        n_te = min(len(pile_te_90), len(pile_tok_te), len(pile_test))
        ytr_p = pile_train['label'].values[:n_tr]
        yte_p = pile_test['label'].values[:n_te]
        if len(set(ytr_p)) < 2 or len(set(yte_p)) < 2:
            auc, acc = 0, 0
        else:
            Xtr = pd.concat([pile_tr_90.iloc[:n_tr].reset_index(drop=True),
                              pile_tok_tr.iloc[:n_tr].reset_index(drop=True)], axis=1)
            Xte = pd.concat([pile_te_90.iloc[:n_te].reset_index(drop=True),
                              pile_tok_te.iloc[:n_te].reset_index(drop=True)], axis=1)
            common = [c for c in Xtr.columns if c in Xte.columns]
            Xtr, Xte = Xtr[common], Xte[common]
            clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                                eval_metric='logloss', random_state=SEED, tree_method='hist', device='cuda')
            clf.fit(Xtr, ytr_p, verbose=False)
            auc = roc_auc_score(yte_p, clf.predict_proba(Xte)[:,1])
            acc = accuracy_score(yte_p, clf.predict(Xte))
    else:
        auc, acc = 0, 0
    results.append({'Dataset': 'Pile', 'AUC_90': 0.9831, 'AUC_120': auc, 'Acc_120': acc})
    print(f"    Pile: AUC={auc:.4f}")

    df = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df['AUC_90'], w, label='88 handcrafted', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, df['AUC_120'], w, label='118 combined', color='#F44336', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Dataset'], rotation=15, ha='right')
    ax.set_ylabel('AUC')
    ax.set_title('Cross-Dataset: 88 Handcrafted vs 118 Combined')
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    for i, (a90, a120) in enumerate(zip(df['AUC_90'], df['AUC_120'])):
        ax.text(i - w/2, a90 + 0.01, f'{a90:.4f}', ha='center', fontsize=8)
        ax.text(i + w/2, a120 + 0.01, f'{a120:.4f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_120_cross_dataset.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: raid_120_cross_dataset.png")
    return df


def main():
    print("=" * 60)
    print("RAID — 118 Combined Features Analysis")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    (train_df, test_df, X_tr, X_te, y_tr, y_te,
     X_tr_90, X_te_90, tok_tr, tok_te) = load_combined()
    print(f"  120 features: {X_tr.shape[1]} dims")
    print(f"  Train: {len(y_tr):,}  Test: {len(y_te):,}")

    print("\n[2/6] Method comparison...")
    results_df, clfs = run_method_comparison(X_tr, X_te, y_tr, y_te,
                                              X_tr_90, X_te_90, tok_tr, tok_te)
    plot_comparison_bar(results_df)

    print("\n[3/6] SHAP analysis (120 features)...")
    clf_120 = clfs['XGBoost_118 combined'][0]
    pred_120 = clfs['XGBoost_118 combined'][2]
    plot_shap_120(clf_120, X_te)

    print("\n[4/6] 11-group ablation...")
    ablation = plot_ablation_11groups(X_tr, X_te, y_tr, y_te)

    print("\n[5/6] Per-generator improvement...")
    # Get 90-feature predictions
    clf_90 = clfs['XGBoost_88 handcrafted'][0]
    pred_90 = clfs['XGBoost_88 handcrafted'][2]
    gen_df = plot_per_generator_improvement(test_df, y_te, pred_90, pred_120)

    print("\n[6/6] Cross-dataset validation...")
    cross_df = run_cross_dataset_120()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  RAID XGBoost (90 feat):  AUC=0.9951  Acc=0.9714")
    r120 = results_df[results_df['Features'] == '118 combined']
    r120 = r120[r120['Classifier'] == 'XGBoost'].iloc[0]
    print(f"  RAID XGBoost (120 feat): AUC={r120['AUC']:.4f}  Acc={r120['Accuracy']:.4f}")
    print(f"  Improvement:             AUC +{r120['AUC']-0.9951:.4f}  Acc +{r120['Accuracy']-0.9714:.4f}")
    print(f"\n  Generated 5 figures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
