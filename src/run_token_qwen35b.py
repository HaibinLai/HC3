"""
Token probability features with Qwen3.6-35B-A3B (MoE, 3B active) as observer model.
Compare with Mistral-7B-Instruct results.
Tests on HC3, SemEval, TuringBench, Pile (using subsamples).
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / 'src'))
from run_token_features import extract_token_prob_features, _empty_features


MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
MODEL_TAG = "qwen36_35b_a3b"


def load_model():
    device = 'cuda'
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer, device


def extract_or_cache(name, texts, model, tokenizer, device, label):
    cache = DATA_DIR / f"token_features_{MODEL_TAG}_{name}.csv"
    if cache.exists():
        print(f"  Loading cached {MODEL_TAG}_{name}...")
        return pd.read_csv(cache)
    df = extract_token_prob_features(texts, model, tokenizer, device, label=label)
    df.to_csv(cache, index=False)
    return df


def run_xgb(name, X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist', device='cuda')
    clf.fit(X_train, y_train, verbose=False)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  {name}: AUC={auc:.4f}, Acc={acc:.4f}")

    # Top 5 features
    importances = clf.feature_importances_
    feat_names = X_train.columns
    top_idx = np.argsort(importances)[::-1][:5]
    for idx in top_idx:
        print(f"    {feat_names[idx]:25s}: {importances[idx]:.4f}")

    return auc, acc


if __name__ == "__main__":
    model, tokenizer, device = load_model()
    results = {}

    # ── HC3 ──
    print("\n" + "="*60 + "\nHC3")
    from data_splits import get_splits
    train_df, test_df = get_splits()
    n = 5000
    train_sub = pd.concat([g.sample(min(len(g), n), random_state=42)
                           for _, g in train_df.groupby('label')]).reset_index(drop=True)
    test_sub = pd.concat([g.sample(min(len(g), n//2), random_state=42)
                          for _, g in test_df.groupby('label')]).reset_index(drop=True)

    X_tr = extract_or_cache("hc3_train", train_sub['text'].tolist(), model, tokenizer, device, "hc3_tr")
    X_te = extract_or_cache("hc3_test", test_sub['text'].tolist(), model, tokenizer, device, "hc3_te")
    auc, acc = run_xgb("HC3", X_tr, train_sub['label'].values, X_te, test_sub['label'].values)
    results['HC3'] = auc
    del train_df, test_df; torch.cuda.empty_cache()

    # ── SemEval ──
    print("\n" + "="*60 + "\nSemEval")
    se_dir = ROOT / "data" / "external" / "semeval2024_task8" / "subtaskA_monolingual"
    se_train = pd.read_parquet(se_dir / "train-00000-of-00001.parquet")
    se_test = pd.read_parquet(se_dir / "test-00000-of-00001.parquet")
    se_train_sub = pd.concat([g.sample(min(len(g), 5000), random_state=42)
                              for _, g in se_train.groupby('label')]).reset_index(drop=True)
    se_test_sub = pd.concat([g.sample(min(len(g), 2500), random_state=42)
                             for _, g in se_test.groupby('label')]).reset_index(drop=True)

    X_tr = extract_or_cache("semeval_train", se_train_sub['text'].tolist(), model, tokenizer, device, "se_tr")
    X_te = extract_or_cache("semeval_test", se_test_sub['text'].tolist(), model, tokenizer, device, "se_te")
    auc, acc = run_xgb("SemEval", X_tr, se_train_sub['label'].values, X_te, se_test_sub['label'].values)
    results['SemEval'] = auc
    del se_train, se_test; torch.cuda.empty_cache()

    # ── TuringBench ──
    print("\n" + "="*60 + "\nTuringBench")
    TB_DIR = ROOT / "data" / "external" / "turingbench" / "extracted" / "TuringBench"
    tb_dfs = []
    for subdir in sorted(TB_DIR.iterdir()):
        if subdir.name.startswith('.') or subdir.name == '__MACOSX':
            continue
        for split in ['train', 'test']:
            f = subdir / f'{split}.csv'
            if f.exists():
                df = pd.read_csv(f); df['split'] = split; df['model'] = subdir.name; tb_dfs.append(df)
    tb_all = pd.concat(tb_dfs, ignore_index=True).rename(columns={'Generation': 'text'})
    tb_all['label'] = (tb_all['model'] != 'AA').astype(int)
    tb_all = tb_all.dropna(subset=['text'])
    tb_all = tb_all[tb_all['text'].str.len() > 10].reset_index(drop=True)
    tb_train = tb_all[tb_all['split'] == 'train'].reset_index(drop=True)
    tb_test = tb_all[tb_all['split'] == 'test'].reset_index(drop=True)
    tb_train_sub = pd.concat([g.sample(min(len(g), 5000), random_state=42)
                               for _, g in tb_train.groupby('label')]).reset_index(drop=True)
    tb_test_sub = pd.concat([g.sample(min(len(g), 2500), random_state=42)
                              for _, g in tb_test.groupby('label')]).reset_index(drop=True)

    X_tr = extract_or_cache("tb_train", tb_train_sub['text'].tolist(), model, tokenizer, device, "tb_tr")
    X_te = extract_or_cache("tb_test", tb_test_sub['text'].tolist(), model, tokenizer, device, "tb_te")
    auc, acc = run_xgb("TuringBench", X_tr, tb_train_sub['label'].values, X_te, tb_test_sub['label'].values)
    results['TuringBench'] = auc
    del tb_all; torch.cuda.empty_cache()

    # ── Pile ──
    print("\n" + "="*60 + "\nPile")
    pile_dir = ROOT / "data" / "external" / "ai_text_detection_pile" / "data"
    pile_dfs = [pd.read_parquet(f, columns=['text', 'source']) for f in sorted(pile_dir.glob("*.parquet"))]
    pile_all = pd.concat(pile_dfs, ignore_index=True)
    pile_all['label'] = (pile_all['source'] == 'ai').astype(int)
    pile_all = pile_all.dropna(subset=['text'])
    pile_all = pile_all[pile_all['text'].str.len() > 10].reset_index(drop=True)
    pile_sub = pd.concat([g.sample(min(len(g), 10000), random_state=42)
                           for _, g in pile_all.groupby('label')]).reset_index(drop=True)
    pile_train, pile_test = train_test_split(pile_sub, test_size=0.33, stratify=pile_sub['label'], random_state=42)
    pile_train = pile_train.reset_index(drop=True)
    pile_test = pile_test.reset_index(drop=True)

    X_tr = extract_or_cache("pile_train", pile_train['text'].tolist(), model, tokenizer, device, "pile_tr")
    X_te = extract_or_cache("pile_test", pile_test['text'].tolist(), model, tokenizer, device, "pile_te")
    auc, acc = run_xgb("Pile", X_tr, pile_train['label'].values, X_te, pile_test['label'].values)
    results['Pile'] = auc

    # ── Summary ──
    print("\n" + "="*60)
    print(f"SUMMARY: {MODEL_NAME} vs Mistral-7B vs Qwen3.5-4B")
    print("="*60)
    mistral = {'HC3': 0.9998, 'SemEval': 0.9784, 'TuringBench': 0.4853, 'Pile': 0.9918}
    qwen4b = {'HC3': 0.9994, 'SemEval': 0.9844, 'TuringBench': 0.5549, 'Pile': 0.9924}
    print(f"  {'Dataset':<15s} {'Mistral-7B':>10s} {'Qwen3.5-4B':>10s} {'Qwen3.6-35B':>11s}")
    for ds in ['HC3', 'SemEval', 'TuringBench', 'Pile']:
        print(f"  {ds:<15s} {mistral[ds]:>10.4f} {qwen4b[ds]:>10.4f} {results[ds]:>11.4f}")

    # Plot comparison (3 models)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = ['HC3', 'SemEval', 'TuringBench', 'Pile']
    x = np.arange(len(datasets))
    width = 0.25

    m_vals = [mistral[d] for d in datasets]
    q4_vals = [qwen4b[d] for d in datasets]
    q35_vals = [results[d] for d in datasets]

    bars1 = ax.bar(x - width, m_vals, width, label='Mistral-7B-Instruct (7B)', color='#1f77b4', alpha=0.85)
    bars2 = ax.bar(x, q4_vals, width, label='Qwen3.5-4B (4B)', color='#ff7f0e', alpha=0.85)
    bars3 = ax.bar(x + width, q35_vals, width, label='Qwen3.6-35B-A3B (3B active)', color='#2ca02c', alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('AUC')
    ax.set_title('Token Probability Features: Observer Model Comparison (3 Models)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0.4, 1.08)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fname = FIG_DIR / 'token_observer_comparison_3models.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fname}")
    print("\nDone!")
