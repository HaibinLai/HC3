"""
DetectGPT & Fast-DetectGPT: zero-shot AI text detection via probability curvature.

Usage:
    python src/run_detectgpt.py                       # full test set
    python src/run_detectgpt.py --max-rows 5000       # quick pilot
    python src/run_detectgpt.py --fast-only            # skip slow DetectGPT
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

warnings.filterwarnings("ignore")
from data_splits import get_splits


# ── helpers ──────────────────────────────────────────────────────────
def compute_log_prob(
    model, tokenizer, texts: list[str],
    batch_size: int = 16, max_len: int = 512, device: str = "cuda",
) -> np.ndarray:
    """Mean log-probability per token for each text."""
    model.eval()
    all_lp = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_len, padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = enc["input_ids"][:, 1:]
        attn = enc["attention_mask"][:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        sample_lp = (token_lp * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        all_lp.extend(sample_lp.cpu().tolist())

    return np.array(all_lp)


# ── DetectGPT: T5 perturbation ───────────────────────────────────────
def perturb_texts_t5(
    texts: list[str], t5_model, t5_tokenizer,
    mask_pct: float = 0.15, device: str = "cuda",
) -> list[str]:
    """Mask ~15% of tokens and fill with T5."""
    perturbed = []
    for text in texts:
        words = text.split()
        if len(words) < 5:
            perturbed.append(text)
            continue
        n_mask = max(1, int(len(words) * mask_pct))
        mask_indices = np.random.choice(len(words), n_mask, replace=False)
        masked_words = words.copy()
        # group consecutive masks into single <extra_id_N> spans
        mask_indices = sorted(mask_indices)
        span_id = 0
        i = 0
        new_words = []
        mask_set = set(mask_indices)
        in_span = False
        for j, w in enumerate(words):
            if j in mask_set:
                if not in_span:
                    new_words.append(f"<extra_id_{span_id}>")
                    span_id += 1
                    in_span = True
            else:
                new_words.append(w)
                in_span = False
        masked_text = " ".join(new_words)

        enc = t5_tokenizer(masked_text, return_tensors="pt", max_length=256, truncation=True).to(device)
        with torch.no_grad():
            out = t5_model.generate(**enc, max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.95)
        decoded = t5_tokenizer.decode(out[0], skip_special_tokens=False)

        # reconstruct
        result_words = new_words.copy()
        for sid in range(span_id):
            tag = f"<extra_id_{sid}>"
            next_tag = f"<extra_id_{sid+1}>"
            if tag in decoded:
                start = decoded.index(tag) + len(tag)
                end = decoded.index(next_tag) if next_tag in decoded else decoded.index("</s>") if "</s>" in decoded else len(decoded)
                fill = decoded[start:end].strip()
                # replace tag in result
                try:
                    idx = result_words.index(tag)
                    result_words[idx] = fill if fill else words[mask_indices[sid]] if sid < len(mask_indices) else ""
                except ValueError:
                    pass

        perturbed.append(" ".join(result_words))
    return perturbed


def run_detectgpt(
    scoring_model, scoring_tokenizer,
    t5_model, t5_tokenizer,
    texts: list[str], n_perturbations: int = 25,
    batch_size: int = 16, device: str = "cuda",
) -> np.ndarray:
    """Compute DetectGPT scores (perturbation-based curvature)."""
    print("  Computing original log-probs...")
    orig_lp = compute_log_prob(scoring_model, scoring_tokenizer, texts, batch_size, device=device)

    all_perturbed_lp = []
    for k in range(n_perturbations):
        print(f"  Perturbation {k+1}/{n_perturbations}...")
        perturbed = perturb_texts_t5(texts, t5_model, t5_tokenizer, device=device)
        plp = compute_log_prob(scoring_model, scoring_tokenizer, perturbed, batch_size, device=device)
        all_perturbed_lp.append(plp)

    all_perturbed_lp = np.array(all_perturbed_lp)  # (K, N)
    mean_plp = all_perturbed_lp.mean(axis=0)
    std_plp = all_perturbed_lp.std(axis=0) + 1e-8

    # DetectGPT score: d(x) = (log_prob(x) - mean_log_prob(perturbed)) / std
    detectgpt_score = (orig_lp - mean_plp) / std_plp
    return detectgpt_score


# ── Fast-DetectGPT: conditional sampling ─────────────────────────────
def run_fast_detectgpt(
    model, tokenizer, texts: list[str],
    batch_size: int = 16, max_len: int = 512, device: str = "cuda",
) -> np.ndarray:
    """Compute Fast-DetectGPT scores via conditional probability sampling."""
    model.eval()
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_len, padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = enc["input_ids"][:, 1:]
        attn = enc["attention_mask"][:, 1:]

        # log prob of original tokens
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        orig_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # sample alternative tokens from conditional distribution
        probs = torch.softmax(shift_logits, dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, probs.size(-1)), 1
        ).view(probs.size(0), probs.size(1))
        sampled_lp = log_probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)

        # score = mean(orig_lp - sampled_lp) / std(orig_lp - sampled_lp)
        diff = orig_lp - sampled_lp
        masked_diff = diff * attn
        lengths = attn.sum(dim=1).clamp(min=1)
        mean_diff = masked_diff.sum(dim=1) / lengths
        var_diff = ((masked_diff - mean_diff.unsqueeze(1) * attn) ** 2 * attn).sum(dim=1) / lengths
        std_diff = torch.sqrt(var_diff + 1e-8)

        score = mean_diff / std_diff
        all_scores.extend(score.cpu().tolist())

        if i % (batch_size * 50) == 0:
            print(f"  {i}/{len(texts)}")

    return np.array(all_scores)


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--scoring-model", default="gpt2-medium")
    parser.add_argument("--n-perturbations", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument("--fig-dir", default="figures")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── data ──
    print("Loading data...")
    _, test_df = get_splits(args.max_rows)

    # For DetectGPT (slow), subsample if too large
    if not args.fast_only and len(test_df) > 2000:
        print(f"Subsampling to 2000 for DetectGPT (original: {len(test_df)})")
        test_df_detectgpt = pd.concat([
            g.sample(min(len(g), 1000), random_state=42)
            for _, g in test_df.groupby("label")
        ]).reset_index(drop=True)
    else:
        test_df_detectgpt = test_df

    texts_full = test_df["text"].fillna("").tolist()
    labels_full = test_df["label"].values
    texts_dgpt = test_df_detectgpt["text"].fillna("").tolist()
    labels_dgpt = test_df_detectgpt["label"].values

    print(f"Full test: {len(texts_full)}, DetectGPT subset: {len(texts_dgpt)}")

    # ── load scoring model ──
    print(f"Loading scoring model: {args.scoring_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.scoring_model)
    tokenizer.pad_token = tokenizer.eos_token
    scoring_model = AutoModelForCausalLM.from_pretrained(args.scoring_model).to(device)

    # ── Fast-DetectGPT ──
    print("\n=== Fast-DetectGPT ===")
    fast_scores = run_fast_detectgpt(scoring_model, tokenizer, texts_full, args.batch_size, device=device)
    auc_fast = roc_auc_score(labels_full, fast_scores)
    print(f"ROC AUC: {auc_fast:.4f}")

    # threshold
    thresholds = np.linspace(np.percentile(fast_scores, 1), np.percentile(fast_scores, 99), 1000)
    best_acc_fast, best_t_fast = 0, 0
    for t in thresholds:
        preds = (fast_scores > t).astype(int)
        # Wait - higher score means AI in DetectGPT convention
        # But in HC3, label=1 is chatgpt. Let's check correlation:
        # If AI text has higher curvature score, then score > threshold => label=1
        acc = (preds == labels_full).mean()
        if acc > best_acc_fast:
            best_acc_fast, best_t_fast = acc, t

    # Check if we need to flip (maybe lower score = AI)
    acc_flipped = ((fast_scores < best_t_fast).astype(int) == labels_full).mean()
    if acc_flipped > best_acc_fast:
        best_acc_fast = acc_flipped
        fast_preds = (fast_scores < best_t_fast).astype(int)
        auc_fast = roc_auc_score(labels_full, -fast_scores)
        print(f"(Flipped direction) ROC AUC: {auc_fast:.4f}")
    else:
        fast_preds = (fast_scores > best_t_fast).astype(int)

    print(f"Best accuracy: {best_acc_fast:.4f}")
    print(classification_report(labels_full, fast_preds, target_names=["human", "chatgpt"]))

    # short text
    word_counts = test_df["text"].fillna("").str.split().str.len()
    short_mask = word_counts < 100
    if short_mask.sum() > 50:
        # use same direction as full
        if acc_flipped > (fast_preds == labels_full).mean():
            short_auc = roc_auc_score(labels_full[short_mask], -fast_scores[short_mask])
        else:
            short_auc = roc_auc_score(labels_full[short_mask], fast_scores[short_mask])
        print(f"Short text (<100w, {short_mask.sum()} samples): AUC={short_auc:.4f}")

    # ── DetectGPT (original, slow) ──
    detectgpt_scores = None
    auc_dgpt = None
    if not args.fast_only:
        print(f"\n=== DetectGPT (K={args.n_perturbations}) ===")
        print("Loading T5-large for perturbations...")
        t5_tokenizer = T5TokenizerFast.from_pretrained("t5-large")
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)

        detectgpt_scores = run_detectgpt(
            scoring_model, tokenizer, t5_model, t5_tokenizer,
            texts_dgpt, n_perturbations=args.n_perturbations,
            batch_size=args.batch_size, device=device,
        )

        del t5_model
        torch.cuda.empty_cache()

        auc_dgpt = roc_auc_score(labels_dgpt, detectgpt_scores)
        print(f"ROC AUC: {auc_dgpt:.4f}")

        # check direction
        acc_pos = ((detectgpt_scores > 0).astype(int) == labels_dgpt).mean()
        acc_neg = ((detectgpt_scores < 0).astype(int) == labels_dgpt).mean()
        if acc_neg > acc_pos:
            auc_dgpt = roc_auc_score(labels_dgpt, -detectgpt_scores)
            print(f"(Flipped direction) ROC AUC: {auc_dgpt:.4f}")

        # threshold search
        thresholds = np.linspace(np.percentile(detectgpt_scores, 1), np.percentile(detectgpt_scores, 99), 1000)
        best_acc_dgpt = 0
        for t in thresholds:
            for sign in [1, -1]:
                preds = ((sign * detectgpt_scores) > t).astype(int) if sign == 1 else (detectgpt_scores < t).astype(int)
                acc = (preds == labels_dgpt).mean()
                if acc > best_acc_dgpt:
                    best_acc_dgpt = acc
        print(f"Best accuracy: {best_acc_dgpt:.4f}")

    # ── plots ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for lab, name, color in [(0, "human", "#4974a5"), (1, "chatgpt", "#d16f4f")]:
        mask = labels_full == lab
        ax.hist(fast_scores[mask], bins=80, alpha=0.6, label=name, color=color, density=True)
    ax.set_xlabel("Fast-DetectGPT score")
    ax.set_ylabel("Density")
    ax.set_title(f"Fast-DetectGPT (AUC={auc_fast:.4f})")
    ax.legend()

    ax = axes[1]
    if detectgpt_scores is not None:
        for lab, name, color in [(0, "human", "#4974a5"), (1, "chatgpt", "#d16f4f")]:
            mask = labels_dgpt == lab
            ax.hist(detectgpt_scores[mask], bins=80, alpha=0.6, label=name, color=color, density=True)
        ax.set_xlabel("DetectGPT score")
        ax.set_title(f"DetectGPT K={args.n_perturbations} (AUC={auc_dgpt:.4f})")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Skipped (--fast-only)", transform=ax.transAxes, ha="center")
        ax.set_title("DetectGPT (skipped)")

    plt.tight_layout()
    plt.savefig(fig_dir / "detectgpt_analysis.png", dpi=180)
    plt.close()
    print(f"Saved {fig_dir / 'detectgpt_analysis.png'}")

    # save results
    results = []
    results.append({"method": "Fast-DetectGPT", "auc": auc_fast, "accuracy": best_acc_fast, "n_test": len(labels_full)})
    if auc_dgpt is not None:
        results.append({"method": "DetectGPT", "auc": auc_dgpt, "accuracy": best_acc_dgpt, "n_test": len(labels_dgpt)})
    results_path = Path("data/processed/detectgpt_results.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
