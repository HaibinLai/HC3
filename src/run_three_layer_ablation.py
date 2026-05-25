#!/usr/bin/env python3
"""
Three-Layer Feature Ablation: 7 configs × 5 datasets + auto-filter.

Layer definitions:
  STAT  (~36d): basic_counts, averages, variability, lexical_richness, punctuation, readability, structure
  MODEL  (52d): embedding_pca (50) + perplexity (2)
  TOKEN  (30d): Mistral-7B token probability features

Outputs:
  data/processed/three_layer_results.csv
  figures/three_layer_heatmap.png
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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from auto_filter import auto_filter_groups, greedy_forward_filter, GROUPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
FIG  = ROOT / 'figures'
SEED = 42

# ── Two-layer + sub-group mapping ──
# Layer 1: Model-Free (no neural network)
MODELFREE_GROUPS = ['basic_counts', 'averages', 'variability', 'lexical_richness',
                    'punctuation', 'readability', 'structure']
# Layer 2: Model-Based (requires LLM/pretrained model inference)
#   sub-groups: embedding_pca (BERT), perplexity (GPT-2), token (Mistral-7B)
MODELBASED_GROUPS = ['embedding_pca', 'perplexity']
# Token features are also model-based (Mistral-7B)

def get_layer_cols(all_cols):
    """Given feature columns, return model-free / model-based layer columns."""
    mf_cols, mb_cols = [], []
    for g in MODELFREE_GROUPS:
        for f in GROUPS[g]:
            if f in all_cols:
                mf_cols.append(f)
    for g in MODELBASED_GROUPS:
        for f in GROUPS[g]:
            if f in all_cols:
                mb_cols.append(f)
    return mf_cols, mb_cols


# ── HC3 column rename mapping ──
HC3_RENAME = {
    'avg_word_len': 'avg_word_length',
    'avg_sentence_len': 'avg_sentence_length',
    'avg_paragraph_len': 'words_per_paragraph',
    'word_len_std': 'word_length_std',
    'sentence_len_std': 'sentence_length_std',
    'hapax_legomena_ratio': 'hapax_ratio',
    'yules_k': 'yule_k',
    'simpsons_diversity': 'simpson_diversity',
    'comma_ratio': 'punct_comma_rate',
    'semicolon_ratio': 'punct_semicolon_rate',
    'question_ratio': 'punct_question_rate',
    'exclamation_ratio': 'punct_exclaim_rate',
    'colon_ratio': 'punct_colon_rate',
    'parenthesis_ratio': 'punct_quote_rate',
    'coleman_liau_index': 'coleman_liau',
    'automated_readability_index': 'ari',
    'dale_chall_score': 'dale_chall',
    'log_perplexity': 'gpt2_log_perplexity',
}
# HC3 embedding: emb_pc{i} → emb_pca_{i}
for i in range(50):
    HC3_RENAME[f'emb_pc{i}'] = f'emb_pca_{i}'


def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def train_eval_xgb(X_tr, X_te, y_tr, y_te):
    """Train XGBoost and return AUC."""
    if len(X_tr.columns) == 0 or len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return 0.5
    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='logloss', random_state=SEED,
                        tree_method='hist', device='cuda')
    clf.fit(X_tr, y_tr, verbose=False)
    prob = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, prob)


def run_configs(X90_tr, X90_te, tok_tr, tok_te, y_tr, y_te, dataset_name):
    """Run configs on one dataset: model-free, model-based sub-groups, combinations."""
    mf_cols, mb_cols = get_layer_cols(X90_tr.columns)
    tok_cols = list(tok_tr.columns) if tok_tr is not None else []
    # BERT embedding sub-group
    bert_cols = [c for c in mb_cols if c.startswith('emb_pca_')]
    # GPT-2 perplexity sub-group
    ppl_cols = [c for c in mb_cols if 'perplexity' in c]

    has_tok = tok_tr is not None and len(tok_cols) > 0
    n_tr = len(X90_tr)
    n_te = len(X90_te)
    if has_tok:
        n_tr = min(n_tr, len(tok_tr))
        n_te = min(n_te, len(tok_te))

    X90_tr = X90_tr.iloc[:n_tr].reset_index(drop=True)
    X90_te = X90_te.iloc[:n_te].reset_index(drop=True)
    y_tr = y_tr[:n_tr]
    y_te = y_te[:n_te]
    if has_tok:
        tok_tr = tok_tr.iloc[:n_tr].reset_index(drop=True)
        tok_te = tok_te.iloc[:n_te].reset_index(drop=True)

    results = []

    def _run(name, cols_90, use_tok):
        if use_tok and has_tok:
            Xtr = pd.concat([X90_tr[cols_90], tok_tr], axis=1) if cols_90 else tok_tr.copy()
            Xte = pd.concat([X90_te[cols_90], tok_te], axis=1) if cols_90 else tok_te.copy()
        else:
            Xtr = X90_tr[cols_90]
            Xte = X90_te[cols_90]
        Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
        Xte = Xte.loc[:, ~Xte.columns.duplicated()]
        auc = train_eval_xgb(Xtr, Xte, y_tr, y_te)
        print(f"  {dataset_name:12s} | {name:20s} | {len(Xtr.columns):3d}d | AUC={auc:.4f}")
        results.append({'dataset': dataset_name, 'config': name, 'n_feats': len(Xtr.columns), 'auc': auc})

    # Layer 1: Model-Free
    _run('model-free', mf_cols, False)

    # Layer 2 sub-groups
    _run('BERT-emb', bert_cols, False)
    _run('GPT2-ppl', ppl_cols, False)
    if has_tok:
        _run('Mistral-token', [], True)
    else:
        results.append({'dataset': dataset_name, 'config': 'Mistral-token', 'n_feats': 0, 'auc': 0.5})
        print(f"  {dataset_name:12s} | {'Mistral-token':20s} |   0d | AUC=0.5000 (no token data)")

    # Layer 2 combined (all model-based)
    _run('model-based (all)', mb_cols, True)

    # Two-layer combinations
    _run('MF + BERT', mf_cols + bert_cols, False)
    _run('MF + token', mf_cols, True)
    _run('MF + MB (all)', mf_cols + mb_cols, True)

    # Auto-filter (threshold-based, legacy)
    if has_tok:
        kept, dropped, g_aucs = auto_filter_groups(X90_tr, y_tr, threshold=0.52, verbose=False)
        if kept:
            Xtr_af = pd.concat([X90_tr[kept].reset_index(drop=True), tok_tr], axis=1)
            Xte_af = pd.concat([X90_te[kept].reset_index(drop=True), tok_te], axis=1)
        else:
            Xtr_af = tok_tr.copy()
            Xte_af = tok_te.copy()
        Xtr_af = Xtr_af.loc[:, ~Xtr_af.columns.duplicated()]
        Xte_af = Xte_af.loc[:, ~Xte_af.columns.duplicated()]
        auc_af = train_eval_xgb(Xtr_af, Xte_af, y_tr, y_te)
        n_dropped = len(dropped)
        print(f"  {dataset_name:12s} | {'auto-filter':20s} | {len(Xtr_af.columns):3d}d | AUC={auc_af:.4f} (dropped {n_dropped} groups: {dropped})")
        results.append({'dataset': dataset_name, 'config': 'auto-filter', 'n_feats': len(Xtr_af.columns), 'auc': auc_af})
    else:
        last_mf_mb = [r for r in results if r['config'] == 'MF + MB (all)']
        results.append({'dataset': dataset_name, 'config': 'auto-filter', 'n_feats': len(mf_cols + mb_cols), 'auc': last_mf_mb[0]['auc'] if last_mf_mb else 0.5})

    # Greedy forward selection (CV-based)
    if has_tok:
        gf_kept, gf_kept_g, gf_dropped_g, gf_log = greedy_forward_filter(
            X90_tr, tok_tr, y_tr, min_gain=0.01, cv=3, verbose=True)
        if gf_kept:
            Xtr_gf = pd.concat([X90_tr[gf_kept].reset_index(drop=True), tok_tr], axis=1)
            Xte_gf = pd.concat([X90_te[gf_kept].reset_index(drop=True), tok_te], axis=1)
        else:
            Xtr_gf = tok_tr.copy()
            Xte_gf = tok_te.copy()
        Xtr_gf = Xtr_gf.loc[:, ~Xtr_gf.columns.duplicated()]
        Xte_gf = Xte_gf.loc[:, ~Xte_gf.columns.duplicated()]
        auc_gf = train_eval_xgb(Xtr_gf, Xte_gf, y_tr, y_te)
        print(f"  {dataset_name:12s} | {'greedy-filter':20s} | {len(Xtr_gf.columns):3d}d | AUC={auc_gf:.4f} (kept groups: {gf_kept_g}, dropped: {gf_dropped_g})")
        results.append({'dataset': dataset_name, 'config': 'greedy-filter', 'n_feats': len(Xtr_gf.columns), 'auc': auc_gf})
    else:
        last_mf_mb = [r for r in results if r['config'] == 'MF + MB (all)']
        results.append({'dataset': dataset_name, 'config': 'greedy-filter', 'n_feats': len(mf_cols + mb_cols), 'auc': last_mf_mb[0]['auc'] if last_mf_mb else 0.5})

    return results


# ── Dataset loaders ──

def load_raid():
    X_tr = clean(pd.read_csv(DATA / 'raid_features_train.csv'))
    X_te = clean(pd.read_csv(DATA / 'raid_features_test.csv'))
    y_tr = pd.read_csv(DATA / 'raid_labels_train.csv')['label'].values
    y_te = pd.read_csv(DATA / 'raid_labels_test.csv')['label'].values
    tok_tr = clean(pd.read_csv(DATA / 'token_features_raid_train.csv'))
    tok_te = clean(pd.read_csv(DATA / 'token_features_raid_test.csv'))
    return X_tr, X_te, tok_tr, tok_te, y_tr, y_te


def load_hc3():
    df = clean(pd.read_csv(DATA / 'hc3_extended_features.csv'))
    df = df.rename(columns=HC3_RENAME)
    feat_cols = [c for c in df.columns if c not in ['label', 'label_name', 'source']]
    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, stratify=df['label'], random_state=SEED)
    X_tr = df.loc[tr_idx, feat_cols].reset_index(drop=True)
    X_te = df.loc[te_idx, feat_cols].reset_index(drop=True)
    y_tr = df.loc[tr_idx, 'label'].values
    y_te = df.loc[te_idx, 'label'].values
    tok_tr = clean(pd.read_csv(DATA / 'token_features_hc3_train.csv'))
    tok_te = clean(pd.read_csv(DATA / 'token_features_hc3_test.csv'))
    return X_tr, X_te, tok_tr, tok_te, y_tr, y_te


def load_semeval():
    X_tr = clean(pd.read_csv(DATA / 'semeval_features_train.csv'))
    X_te = clean(pd.read_csv(DATA / 'semeval_features_test.csv'))
    # Use full token features + aligned labels
    tok_tr = clean(pd.read_csv(DATA / 'token_features_semeval_train_full.csv'))
    tok_te = clean(pd.read_csv(DATA / 'token_features_semeval_test_full.csv'))
    y_tr = pd.read_csv(DATA / 'semeval_labels_train_full.csv')['label'].values
    y_te = pd.read_csv(DATA / 'semeval_labels_test_full.csv')['label'].values
    # Align by index
    idx_tr = pd.read_csv(DATA / 'semeval_idx_train_full.csv').iloc[:, 0].values
    idx_te = pd.read_csv(DATA / 'semeval_idx_test_full.csv').iloc[:, 0].values
    X_tr = X_tr.iloc[idx_tr].reset_index(drop=True)
    X_te = X_te.iloc[idx_te].reset_index(drop=True)
    return X_tr, X_te, tok_tr, tok_te, y_tr, y_te


def load_turingbench():
    X_tr = clean(pd.read_csv(DATA / 'turingbench_features_train.csv'))
    X_te = clean(pd.read_csv(DATA / 'turingbench_features_test.csv'))
    tok_tr = clean(pd.read_csv(DATA / 'token_features_turingbench_train_full.csv'))
    tok_te = clean(pd.read_csv(DATA / 'token_features_turingbench_test_full.csv'))
    y_tr = pd.read_csv(DATA / 'turingbench_labels_train_full.csv')['binary_label'].values
    y_te = pd.read_csv(DATA / 'turingbench_labels_test_full.csv')['binary_label'].values
    idx_tr = pd.read_csv(DATA / 'turingbench_idx_train_full.csv').iloc[:, 0].values
    idx_te = pd.read_csv(DATA / 'turingbench_idx_test_full.csv').iloc[:, 0].values
    X_tr = X_tr.iloc[idx_tr].reset_index(drop=True)
    X_te = X_te.iloc[idx_te].reset_index(drop=True)
    return X_tr, X_te, tok_tr, tok_te, y_tr, y_te


def load_pile():
    X_tr = clean(pd.read_csv(DATA / 'pile_features_train.csv'))
    X_te = clean(pd.read_csv(DATA / 'pile_features_test.csv'))
    tok_tr = clean(pd.read_csv(DATA / 'token_features_pile_train.csv'))
    tok_te = clean(pd.read_csv(DATA / 'token_features_pile_test.csv'))
    # Pile labels are in the feature files or loaded from run_pile
    # Labels come from run_pile.py's load_pile() - try to load
    try:
        from run_pile import load_pile as _load_pile
        pile_train, pile_test = _load_pile(max_per_class=50000)
        y_tr = pile_train['label'].values[:len(X_tr)]
        y_te = pile_test['label'].values[:len(X_te)]
    except Exception:
        # Fallback: features have same order as data, first half human, second half AI
        n_tr = len(X_tr)
        n_te = len(X_te)
        y_tr = np.array([0]*(n_tr//2) + [1]*(n_tr - n_tr//2))
        y_te = np.array([0]*(n_te//2) + [1]*(n_te - n_te//2))
    n = min(len(X_tr), len(tok_tr))
    X_tr = X_tr.iloc[:n].reset_index(drop=True)
    tok_tr = tok_tr.iloc[:n].reset_index(drop=True)
    y_tr = y_tr[:n]
    n = min(len(X_te), len(tok_te))
    X_te = X_te.iloc[:n].reset_index(drop=True)
    tok_te = tok_te.iloc[:n].reset_index(drop=True)
    y_te = y_te[:n]
    return X_tr, X_te, tok_tr, tok_te, y_tr, y_te


def plot_heatmap(all_results):
    """Create heatmap: rows=configs, cols=datasets, values=AUC."""
    df = pd.DataFrame(all_results)
    pivot = df.pivot(index='config', columns='dataset', values='auc')

    # Order
    config_order = ['model-free', 'BERT-emb', 'GPT2-ppl', 'Mistral-token',
                    'model-based (all)', 'MF + BERT', 'MF + token', 'MF + MB (all)', 'auto-filter', 'greedy-filter']
    dataset_order = ['RAID', 'HC3', 'SemEval', 'TuringBench', 'Pile']
    config_order = [c for c in config_order if c in pivot.index]
    dataset_order = [d for d in dataset_order if d in pivot.columns]
    pivot = pivot.loc[config_order, dataset_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0.45, vmax=1.0,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'AUC'})
    ax.set_title('Three-Layer Feature Ablation × 5 Datasets', fontsize=14)
    ax.set_ylabel('Feature Configuration')
    ax.set_xlabel('Dataset')

    # Add n_feats annotation
    for i, config in enumerate(config_order):
        for j, ds in enumerate(dataset_order):
            row = df[(df['config'] == config) & (df['dataset'] == ds)]
            if not row.empty:
                n = row.iloc[0]['n_feats']
                ax.text(j + 0.5, i + 0.75, f'({n}d)', ha='center', va='center', fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig(FIG / 'three_layer_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIG / 'three_layer_heatmap.png'}")


if __name__ == '__main__':
    all_results = []

    loaders = [
        ('RAID', load_raid),
        ('HC3', load_hc3),
        ('SemEval', load_semeval),
        ('TuringBench', load_turingbench),
        ('Pile', load_pile),
    ]

    for name, loader in loaders:
        print(f"\n{'='*60}")
        print(f"  Dataset: {name}")
        print('='*60)
        try:
            X_tr, X_te, tok_tr, tok_te, y_tr, y_te = loader()
            print(f"  Loaded: train={len(X_tr)}, test={len(X_te)}, "
                  f"90-feat={len(X_tr.columns)}, tok={'yes' if tok_tr is not None else 'no'}")
            results = run_configs(X_tr, X_te, tok_tr, tok_te, y_tr, y_te, name)
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(DATA / 'three_layer_results.csv', index=False)
    print(f"\nSaved: {DATA / 'three_layer_results.csv'}")

    plot_heatmap(all_results)
    print("\nDone!")
