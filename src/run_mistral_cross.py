"""
Run Mistral-7B single-model CE (best from binoculars_7b) on SemEval & TuringBench.
Also test Binoculars ratio to confirm it doesn't help on multi-model datasets either.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


def compute_ce_per_sample(texts, model, tokenizer, device, batch_size=4, max_length=512, label=""):
    model.eval()
    ces = []
    for i in range(0, len(texts), batch_size):
        if i % 500 == 0:
            print(f"    [{label}] {i}/{len(texts)}")
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', truncation=True,
                        max_length=max_length, padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1, :]
            targets = enc['input_ids'][:, 1:]
            mask = enc['attention_mask'][:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_ce = -log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
            for j in range(token_ce.size(0)):
                valid = mask[j].sum().item()
                ces.append((token_ce[j] * mask[j]).sum().item() / valid if valid > 0 else 0.0)
    return np.array(ces)


def evaluate(scores, labels, test_df=None, model_col='model'):
    y = np.array(labels)
    auc_pos = roc_auc_score(y, scores)
    auc_neg = roc_auc_score(y, -scores)
    if auc_neg > auc_pos:
        scores_use = -scores; auc = auc_neg
    else:
        scores_use = scores; auc = auc_pos

    # Best threshold
    thresholds = np.percentile(scores_use, np.arange(5, 96, 5))
    best_f1, best_t = 0, np.median(scores_use)
    for t in thresholds:
        f = f1_score(y, (scores_use > t).astype(int))
        if f > best_f1:
            best_f1, best_t = f, t
    y_pred = (scores_use > best_t).astype(int)
    acc = accuracy_score(y, y_pred)

    print(f"    AUC={auc:.4f}, Acc={acc:.4f}")

    # Per-model
    if test_df is not None and model_col in test_df.columns:
        print(f"    Per-model:")
        for m in sorted(test_df[model_col].dropna().unique()):
            mask = (test_df[model_col] == m).values[:len(y)]
            if mask.sum() < 10:
                continue
            if len(set(y[mask])) > 1:
                m_auc = roc_auc_score(y[mask], scores_use[mask])
            else:
                m_auc = -1
            m_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"      {m:20s}: AUC={m_auc:.4f}, Acc={m_acc:.4f}, n={mask.sum()}")

    return auc, acc


def run_on_dataset(name, texts, labels, test_df, model_col='model'):
    device = 'cuda'

    print(f"\n{'='*60}")
    print(f"Mistral-7B CE on {name} ({len(texts)} samples)")
    print(f"{'='*60}")

    # Load models
    print("  Loading Mistral-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)

    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Compute CE
    ce = compute_ce_per_sample(texts, model, tokenizer, device, batch_size=4, label=name)

    print(f"\n  Mistral-7B-Instruct CE:")
    auc, acc = evaluate(ce, labels, test_df, model_col)

    # Also load base for binoculars ratio
    print("\n  Loading Mistral-7B (base) for Binoculars ratio...")
    del model; torch.cuda.empty_cache()
    model_base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    tokenizer_base = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token

    ce_base = compute_ce_per_sample(texts, model_base, tokenizer_base, device, batch_size=4, label=f"{name}_base")

    bino = np.where(ce > 0, ce_base / ce, 1.0)
    print(f"\n  Binoculars ratio (base/instruct):")
    auc_bino, acc_bino = evaluate(bino, labels, test_df, model_col)

    del model_base; torch.cuda.empty_cache()

    return {'CE_instruct': {'auc': auc, 'acc': acc},
            'binoculars': {'auc': auc_bino, 'acc': acc_bino}}


if __name__ == "__main__":
    results = {}

    # ── SemEval ──
    DATA_SE = ROOT / "data" / "external" / "semeval2024_task8" / "subtaskA_monolingual"
    se_test = pd.read_parquet(DATA_SE / "test-00000-of-00001.parquet")
    # Subsample 5K for speed
    se_sub = pd.concat([
        g.sample(min(len(g), 2500), random_state=42)
        for _, g in se_test.groupby('label')
    ]).reset_index(drop=True)

    results['SemEval'] = run_on_dataset(
        "SemEval", se_sub['text'].tolist(), se_sub['label'].tolist(), se_sub, 'model')
    del se_test, se_sub; torch.cuda.empty_cache()

    # ── TuringBench ──
    TB_DIR = ROOT / "data" / "external" / "turingbench" / "extracted" / "TuringBench"
    tb_dfs = []
    for subdir in sorted(TB_DIR.iterdir()):
        if subdir.name.startswith('.') or subdir.name == '__MACOSX':
            continue
        f = subdir / 'test.csv'
        if f.exists():
            df = pd.read_csv(f)
            df['model'] = subdir.name
            tb_dfs.append(df)
    tb_test = pd.concat(tb_dfs, ignore_index=True).rename(columns={'Generation': 'text'})
    tb_test['label'] = (tb_test['model'] != 'AA').astype(int)
    tb_test = tb_test.dropna(subset=['text'])
    tb_test = tb_test[tb_test['text'].str.len() > 10].reset_index(drop=True)
    # Subsample 5K
    tb_sub = pd.concat([
        g.sample(min(len(g), 2500), random_state=42)
        for _, g in tb_test.groupby('label')
    ]).reset_index(drop=True)

    results['TuringBench'] = run_on_dataset(
        "TuringBench", tb_sub['text'].tolist(), tb_sub['label'].tolist(), tb_sub, 'model')

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY: Mistral-7B on Multiple Datasets")
    print("="*60)
    print(f"  {'Dataset':<15s} {'CE_instruct AUC':>16s} {'Binoculars AUC':>16s}")
    for ds, res in results.items():
        print(f"  {ds:<15s} {res['CE_instruct']['auc']:>16.4f} {res['binoculars']['auc']:>16.4f}")

    print(f"\n  Comparison with previous methods:")
    print(f"  {'Dataset':<15s} {'XGBoost':>8s} {'RoBERTa':>8s} {'FastDGPT':>8s} {'Mistral CE':>10s} {'Bino 7B':>8s}")
    prev = {
        'HC3':         (0.9999, 0.9980, 0.9292, 0.9933),
        'SemEval':     (0.6872, 0.6801, 0.8068, None),
        'TuringBench': (0.9841, 0.6047, 0.6038, None),
    }
    for ds in ['SemEval', 'TuringBench']:
        p = prev[ds]
        ce = results[ds]['CE_instruct']['auc']
        bi = results[ds]['binoculars']['auc']
        print(f"  {ds:<15s} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f} {ce:>10.4f} {bi:>8.4f}")

    print("\nDone!")
