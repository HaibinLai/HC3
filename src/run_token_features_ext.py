"""
Token-level probability features on TuringBench and Pile.
Extends run_token_features.py to complete the cross-dataset matrix.
Reuses extract_token_prob_features and load_model from run_token_features.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / 'src'))
from run_token_features import extract_token_prob_features, load_model, _empty_features


def run_xgb(name, X_train, y_train, X_test, y_test, test_df=None, model_col='model'):
    from xgboost import XGBClassifier

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\n  --- XGBoost token-only ({X_train.shape[1]} features) on {name} ---")
    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist', device='cuda')
    clf.fit(X_train, y_train, verbose=False)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Token-only:  AUC={auc:.4f}, Acc={acc:.4f}")

    # Per-model
    if test_df is not None and model_col in test_df.columns:
        print(f"    Per-model:")
        for m in sorted(test_df[model_col].dropna().unique()):
            mask = (test_df[model_col] == m).values[:len(y_test)]
            if mask.sum() < 10:
                continue
            if len(set(y_test[mask])) > 1:
                m_auc = roc_auc_score(y_test[mask], y_prob[mask])
            else:
                m_auc = -1
            m_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"      {m:20s}: AUC={m_auc:.4f}, Acc={m_acc:.4f}, n={mask.sum()}")

    # Feature importance
    importances = clf.feature_importances_
    feat_names = X_train.columns
    top_idx = np.argsort(importances)[::-1][:10]
    print(f"\n  Top 10 features:")
    for idx in top_idx:
        print(f"    {feat_names[idx]:25s}: {importances[idx]:.4f}")

    return auc, acc


def extract_or_cache(name, texts, model, tokenizer, device, label):
    cache = RESULT_DIR / f"token_features_{name}.csv"
    if cache.exists():
        print(f"  Loading cached {name}...")
        return pd.read_csv(cache)
    df = extract_token_prob_features(texts, model, tokenizer, device, label=label)
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    model, tokenizer, device = load_model()
    results = {}

    # ── TuringBench ──
    print("\n" + "="*60)
    print("Loading TuringBench...")
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

    # Subsample for GPU speed
    n_train, n_test = 10000, 5000
    tb_train_sub = pd.concat([g.sample(min(len(g), n_train // 2), random_state=42)
                               for _, g in tb_train.groupby('label')]).reset_index(drop=True)
    tb_test_sub = pd.concat([g.sample(min(len(g), n_test // 2), random_state=42)
                              for _, g in tb_test.groupby('label')]).reset_index(drop=True)
    print(f"  TuringBench: train={len(tb_train_sub)}, test={len(tb_test_sub)}")

    X_tr_tb = extract_or_cache("turingbench_train", tb_train_sub['text'].tolist(),
                                model, tokenizer, device, "tb_train")
    X_te_tb = extract_or_cache("turingbench_test", tb_test_sub['text'].tolist(),
                                model, tokenizer, device, "tb_test")

    auc_tb, acc_tb = run_xgb("TuringBench", X_tr_tb, tb_train_sub['label'].values,
                              X_te_tb, tb_test_sub['label'].values,
                              test_df=tb_test_sub, model_col='model')
    results['TuringBench'] = auc_tb

    del tb_all, tb_train, tb_test, tb_train_sub, tb_test_sub
    torch.cuda.empty_cache()

    # ── Pile ──
    print("\n" + "="*60)
    print("Loading AI Text Detection Pile...")
    PILE_DIR = ROOT / "data" / "external" / "ai_text_detection_pile" / "data"
    pile_files = sorted(PILE_DIR.glob("*.parquet"))
    pile_dfs = []
    for f in pile_files:
        pile_dfs.append(pd.read_parquet(f, columns=['text', 'source']))
    pile_all = pd.concat(pile_dfs, ignore_index=True)
    pile_all['label'] = (pile_all['source'] == 'ai').astype(int)
    pile_all = pile_all.dropna(subset=['text'])
    pile_all = pile_all[pile_all['text'].str.len() > 10].reset_index(drop=True)
    print(f"  Pile total: {len(pile_all)}")

    # Subsample: 10K train, 5K test
    pile_sub = pd.concat([g.sample(min(len(g), 10000), random_state=42)
                           for _, g in pile_all.groupby('label')]).reset_index(drop=True)
    pile_train, pile_test = train_test_split(pile_sub, test_size=0.33,
                                             stratify=pile_sub['label'], random_state=42)
    pile_train = pile_train.reset_index(drop=True)
    pile_test = pile_test.reset_index(drop=True)
    print(f"  Pile subsample: train={len(pile_train)}, test={len(pile_test)}")

    X_tr_pile = extract_or_cache("pile_train", pile_train['text'].tolist(),
                                  model, tokenizer, device, "pile_train")
    X_te_pile = extract_or_cache("pile_test", pile_test['text'].tolist(),
                                  model, tokenizer, device, "pile_test")

    auc_pile, acc_pile = run_xgb("Pile", X_tr_pile, pile_train['label'].values,
                                  X_te_pile, pile_test['label'].values)
    results['Pile'] = auc_pile

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY: Token-level Features Cross-Dataset")
    print("="*60)
    prev = {
        'HC3':         {'XGBoost': 0.9999, 'Token-only': 0.9998, 'Mistral CE': 0.9933, 'Fast-DetectGPT': 0.9292},
        'SemEval':     {'XGBoost': 0.6872, 'Token-only': 0.9784, 'Mistral CE': 0.9729, 'Fast-DetectGPT': 0.8068},
        'TuringBench': {'XGBoost': 0.9841, 'Token-only': auc_tb,  'Mistral CE': 0.5895, 'Fast-DetectGPT': 0.6038},
        'Pile':        {'XGBoost': 0.9831, 'Token-only': auc_pile, 'Mistral CE': None,   'Fast-DetectGPT': 0.8889},
    }

    print(f"\n  {'Dataset':<15s} {'XGBoost':>8s} {'Token-only':>10s} {'Mistral CE':>10s} {'FastDGPT':>8s}")
    for ds, res in prev.items():
        ce = f"{res['Mistral CE']:.4f}" if res['Mistral CE'] else "   —"
        print(f"  {ds:<15s} {res['XGBoost']:>8.4f} {res['Token-only']:>10.4f} {ce:>10s} {res['Fast-DetectGPT']:>8.4f}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = ['HC3', 'SemEval', 'TuringBench', 'Pile']
    methods = ['XGBoost', 'Token-only', 'Mistral CE', 'Fast-DetectGPT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    x = np.arange(len(datasets))
    width = 0.2

    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            v = prev[ds].get(method)
            vals.append(v if v is not None else 0)
        bars = ax.bar(x + i * width, vals, width, label=method, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('AUC')
    ax.set_title('Cross-Dataset Comparison: Token Probability Features vs Other Methods')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.08)
    plt.tight_layout()
    fname = FIG_DIR / 'token_features_full_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fname}")
    print("\nDone!")
