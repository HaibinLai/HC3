"""
Binoculars with large model pairs (7B scale).
Observer: base model, Performer: instruct/chat model.
Tests on HC3 dataset.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

sys.path.insert(0, str(ROOT / 'src'))
from data_splits import get_splits


def compute_cross_entropy(texts, model, tokenizer, device, batch_size=4, max_length=512, label=""):
    """Compute per-text mean cross-entropy loss."""
    model.eval()
    ces = []
    for i in range(0, len(texts), batch_size):
        if i % 200 == 0:
            print(f"    [{label}] CE: {i}/{len(texts)}")
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', truncation=True,
                        max_length=max_length, padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc['input_ids'])
            # Per-sample CE (approximate with batch loss for now)
            # For more accurate per-sample, compute manually:
            logits = out.logits[:, :-1, :]  # (B, T-1, V)
            targets = enc['input_ids'][:, 1:]  # (B, T-1)
            mask = enc['attention_mask'][:, 1:]  # (B, T-1)

            log_probs = torch.log_softmax(logits, dim=-1)
            token_ce = -log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)  # (B, T-1)
            # Masked mean per sample
            for j in range(token_ce.size(0)):
                valid = mask[j].sum().item()
                if valid > 0:
                    ces.append((token_ce[j] * mask[j]).sum().item() / valid)
                else:
                    ces.append(0.0)

    return np.array(ces)


def try_model_pair(observer_name, performer_name, texts, labels, test_df):
    device = 'cuda'
    print(f"\n{'='*60}")
    print(f"Binoculars: {observer_name} / {performer_name}")
    print(f"{'='*60}")

    # Load observer
    print(f"  Loading observer: {observer_name}")
    obs_tokenizer = AutoTokenizer.from_pretrained(observer_name, trust_remote_code=True)
    if obs_tokenizer.pad_token is None:
        obs_tokenizer.pad_token = obs_tokenizer.eos_token
    obs_model = AutoModelForCausalLM.from_pretrained(
        observer_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)

    # Load performer
    print(f"  Loading performer: {performer_name}")
    perf_tokenizer = AutoTokenizer.from_pretrained(performer_name, trust_remote_code=True)
    if perf_tokenizer.pad_token is None:
        perf_tokenizer.pad_token = perf_tokenizer.eos_token
    perf_model = AutoModelForCausalLM.from_pretrained(
        performer_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)

    print(f"  GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Compute CE
    print(f"  Computing CE for {len(texts)} texts...")
    ce_obs = compute_cross_entropy(texts, obs_model, obs_tokenizer, device, batch_size=4, label="observer")
    ce_perf = compute_cross_entropy(texts, perf_model, perf_tokenizer, device, batch_size=4, label="performer")

    # Binoculars score = CE_obs / CE_perf
    with np.errstate(divide='ignore', invalid='ignore'):
        bino_score = np.where(ce_perf > 0, ce_obs / ce_perf, 1.0)

    # Also try single-model CE
    y_test = np.array(labels)

    results = {}

    # Binoculars ratio
    for name, score in [('binoculars_ratio', bino_score), ('CE_observer', ce_obs), ('CE_performer', ce_perf)]:
        auc_pos = roc_auc_score(y_test, score)
        auc_neg = roc_auc_score(y_test, -score)
        auc = max(auc_pos, auc_neg)
        best_score = score if auc_pos >= auc_neg else -score

        # Find best threshold
        from sklearn.metrics import f1_score
        thresholds = np.percentile(best_score, np.arange(5, 96, 5))
        best_f1, best_t = 0, 0
        for t in thresholds:
            pred = (best_score > t).astype(int)
            f = f1_score(y_test, pred)
            if f > best_f1:
                best_f1, best_t = f, t

        y_pred = (best_score > best_t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'auc': auc, 'acc': acc}
        print(f"\n  {name}: AUC={auc:.4f}, Acc={acc:.4f}")

    # Per-domain if available
    if 'source' in test_df.columns:
        print(f"\n  Per-domain (binoculars ratio):")
        best_bino = bino_score if roc_auc_score(y_test, bino_score) > 0.5 else -bino_score
        for domain in sorted(test_df['source'].dropna().unique()):
            mask = (test_df['source'] == domain).values[:len(y_test)]
            if mask.sum() < 20 or len(set(y_test[mask])) < 2:
                continue
            d_auc = roc_auc_score(y_test[mask], best_bino[mask])
            print(f"    {domain:20s}: AUC={d_auc:.4f}, n={mask.sum()}")

    # Cleanup
    del obs_model, perf_model
    torch.cuda.empty_cache()

    return results, bino_score, ce_obs, ce_perf


def plot_results(bino_score, ce_obs, ce_perf, labels, pair_name):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    y = np.array(labels)

    for ax, score, title in [
        (axes[0], bino_score, 'Binoculars Ratio (CE_obs/CE_perf)'),
        (axes[1], ce_obs, 'Observer CE'),
        (axes[2], ce_perf, 'Performer CE'),
    ]:
        ax.hist(score[y==0], bins=50, alpha=0.6, label='Human', color='#2E7D32', density=True)
        ax.hist(score[y==1], bins=50, alpha=0.6, label='AI', color='#C62828', density=True)
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')

    plt.suptitle(f'Binoculars 7B: {pair_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = FIG_DIR / 'binoculars_7b_analysis.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    # Load HC3 test data
    _, test_df = get_splits()

    # Subsample for speed (7B models are slower)
    max_samples = 5000
    if len(test_df) > max_samples:
        test_sub = pd.concat([
            g.sample(min(len(g), max_samples // 2), random_state=42)
            for _, g in test_df.groupby('label')
        ]).reset_index(drop=True)
    else:
        test_sub = test_df

    texts = test_sub['text'].tolist()
    labels = test_sub['label'].tolist()
    print(f"Test samples: {len(texts)}")

    # Try model pairs - newer models with clear base/instruct split
    model_pairs = [
        ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
        ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
        ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"),
    ]

    all_results = {}
    for obs_name, perf_name in model_pairs:
        pair_label = obs_name.split("/")[-1]
        try:
            results, bino, ce_o, ce_p = try_model_pair(
                obs_name, perf_name, texts, labels, test_sub)
            all_results[pair_label] = results
            plot_results(bino, ce_o, ce_p, labels, f"{obs_name} / {perf_name}")
        except Exception as e:
            print(f"\n  ERROR with {obs_name}: {e}")
            torch.cuda.empty_cache()
            continue

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Binoculars 7B Results")
    print("="*60)
    print(f"  Previous (GPT-2 medium/large):  AUC=0.7995")
    for pair, res in all_results.items():
        for metric, vals in res.items():
            print(f"  {pair} {metric:20s}: AUC={vals['auc']:.4f}, Acc={vals['acc']:.4f}")
    print("\nDone!")
