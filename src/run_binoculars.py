"""
Binoculars: zero-shot AI text detection via dual-model perplexity ratio.

Usage:
    python src/run_binoculars.py                    # GPT-2 medium vs large
    python src/run_binoculars.py --max-rows 10000   # quick pilot
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
from data_splits import get_splits


def compute_cross_entropy(
    model, tokenizer, texts: list[str],
    batch_size: int = 16, max_len: int = 512, device: str = "cuda"
) -> np.ndarray:
    """Compute per-text mean cross-entropy under a causal LM."""
    model.eval()
    all_ce = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_len, padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            logits = outputs.logits

        # per-token cross entropy
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = enc["input_ids"][:, 1:].contiguous()
        attn = enc["attention_mask"][:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_ce = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        # mean CE per sample (masked)
        sample_ce = (token_ce * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        all_ce.extend(sample_ce.cpu().tolist())

        if (i // batch_size) % 100 == 0:
            print(f"  {i}/{len(texts)}")

    return np.array(all_ce)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--model1", default="gpt2-medium", help="Observer model")
    parser.add_argument("--model2", default="gpt2-large", help="Performer model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── data ──
    print("Loading data...")
    _, test_df = get_splits(args.max_rows)
    texts = test_df["text"].fillna("").tolist()
    labels = test_df["label"].values
    print(f"Test samples: {len(texts)}")

    # ── load models ──
    print(f"Loading observer: {args.model1}")
    tok1 = AutoTokenizer.from_pretrained(args.model1)
    tok1.pad_token = tok1.eos_token
    m1 = AutoModelForCausalLM.from_pretrained(args.model1).to(device)

    print(f"Loading performer: {args.model2}")
    tok2 = AutoTokenizer.from_pretrained(args.model2)
    tok2.pad_token = tok2.eos_token
    m2 = AutoModelForCausalLM.from_pretrained(args.model2).to(device)

    # ── compute CE ──
    print(f"\nComputing CE with {args.model1}...")
    ce1 = compute_cross_entropy(m1, tok1, texts, args.batch_size, device=device)

    print(f"\nComputing CE with {args.model2}...")
    ce2 = compute_cross_entropy(m2, tok2, texts, args.batch_size, device=device)

    # free GPU memory
    del m1, m2
    torch.cuda.empty_cache()

    # ── Binoculars score ──
    # B(x) = CE_observer / CE_performer
    binoculars_score = ce1 / np.clip(ce2, 1e-8, None)

    # Try both directions for AUC
    auc_pos = roc_auc_score(labels, binoculars_score)
    auc_neg = roc_auc_score(labels, -binoculars_score)
    if auc_pos >= auc_neg:
        auc_bino = auc_pos
        score_for_auc = binoculars_score
        ai_direction = "higher"
    else:
        auc_bino = auc_neg
        score_for_auc = -binoculars_score
        ai_direction = "lower"
    print(f"\n=== Binoculars Results ===")
    print(f"ROC AUC: {auc_bino:.4f}")

    # Also compute AUC with single-model CE (for comparison)
    auc_ce1 = roc_auc_score(labels, -ce1)  # lower CE = AI
    auc_ce2 = roc_auc_score(labels, -ce2)
    print(f"Single-model CE AUC ({args.model1}): {auc_ce1:.4f}")
    print(f"Single-model CE AUC ({args.model2}): {auc_ce2:.4f}")

    # ── threshold-based classification ──
    thresholds = np.linspace(
        np.percentile(score_for_auc, 1),
        np.percentile(score_for_auc, 99), 1000
    )
    best_acc, best_t = 0, 0
    for t in thresholds:
        preds = (score_for_auc > t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    preds = (score_for_auc > best_t).astype(int)
    print(f"Best threshold: {best_t:.4f}  Accuracy: {best_acc:.4f}")
    print(classification_report(labels, preds, target_names=["human", "chatgpt"]))

    # ── short text analysis ──
    word_counts = test_df["text"].fillna("").str.split().str.len()
    short_mask = word_counts < 100
    if short_mask.sum() > 50:
        short_auc = roc_auc_score(labels[short_mask], score_for_auc[short_mask])
        print(f"Short text (<100w, {short_mask.sum()} samples): AUC={short_auc:.4f}")

    # ── plot: score distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # distribution
    ax = axes[0]
    for lab, name, color in [(0, "human", "#4974a5"), (1, "chatgpt", "#d16f4f")]:
        mask = labels == lab
        ax.hist(binoculars_score[mask], bins=80, alpha=0.6, label=name, color=color, density=True)
    ax.set_xlabel("Binoculars score (CE ratio)")
    ax.set_ylabel("Density")
    ax.set_title("Binoculars score distribution")
    ax.axvline(best_t, color="black", linestyle="--", label=f"threshold={best_t:.3f}")
    ax.legend()

    # single-model CE comparison
    ax = axes[1]
    for lab, name, color in [(0, "human", "#4974a5"), (1, "chatgpt", "#d16f4f")]:
        mask = labels == lab
        ax.scatter(ce1[mask], ce2[mask], s=3, alpha=0.3, label=name, color=color)
    ax.set_xlabel(f"CE ({args.model1})")
    ax.set_ylabel(f"CE ({args.model2})")
    ax.set_title("Cross-entropy: observer vs performer")
    ax.plot([0, 8], [0, 8], "k--", alpha=0.3, label="y=x")
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "binoculars_analysis.png", dpi=180)
    plt.close()
    print(f"Saved {fig_dir / 'binoculars_analysis.png'}")

    # save results
    results = {
        "method": "Binoculars",
        "models": f"{args.model1} / {args.model2}",
        "auc": auc_bino,
        "accuracy": best_acc,
        "n_test": len(labels),
    }
    results_path = Path("data/processed/binoculars_results.csv")
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
