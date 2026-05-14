"""
Extended HC3 feature extraction + multi-model comparison + analysis.

Usage:
    python src/run_extended.py                       # full run
    python src/run_extended.py --max-rows 30000      # quick pilot
    python src/run_extended.py --skip-embedding       # skip GPU features
"""
from __future__ import annotations

import argparse
import math
import re
import string
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import textstat
except ImportError:
    textstat = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import shap
except ImportError:
    shap = None

# ── regex ──────────────────────────────────────────────────────────────
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_RE = re.compile(r"\n\s*\n")

TRANSITION_PHRASES = [
    "first", "firstly", "second", "secondly", "third", "finally",
    "in conclusion", "to conclude", "overall", "therefore", "however",
    "moreover", "furthermore", "on the other hand", "as a result",
    "for example", "in addition", "additionally", "consequently",
    "nevertheless", "in summary", "to summarize", "in other words",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "he", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "with", "you", "your",
}

BULLET_RE = re.compile(r"^\s*[-•*]\s", re.MULTILINE)


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


# ── CPU feature extraction ────────────────────────────────────────────
def extract_extended_features(text: str) -> dict[str, float]:
    words = WORD_RE.findall(text)
    lower_words = [w.lower() for w in words]
    word_count = len(words)
    unique_count = len(set(lower_words))
    char_count = len(text)
    alpha_count = sum(c.isalpha() for c in text)
    upper_count = sum(c.isupper() for c in text)
    digit_count = sum(c.isdigit() for c in text)
    punct_count = sum(c in string.punctuation for c in text)

    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    sentence_count = max(1, len(sentences))
    sentence_lengths = [len(WORD_RE.findall(s)) for s in sentences]

    paragraphs = [p.strip() for p in PARAGRAPH_RE.split(text) if p.strip()]
    paragraph_count = max(1, len(paragraphs))

    word_lengths = [len(w) for w in words]
    stopword_count = sum(w in STOPWORDS for w in lower_words)
    transition_count = sum(text.lower().count(ph) for ph in TRANSITION_PHRASES)

    # frequency distribution
    freq = Counter(lower_words)
    freq_values = list(freq.values()) if freq else [0]
    hapax = sum(1 for v in freq_values if v == 1)

    f = {}

    # ── 1. basic counts ──
    f["char_count"] = char_count
    f["word_count"] = word_count
    f["sentence_count"] = sentence_count
    f["paragraph_count"] = paragraph_count

    # ── 2. averages ──
    f["avg_word_len"] = safe_divide(sum(word_lengths), word_count)
    f["avg_sentence_len"] = safe_divide(word_count, sentence_count)
    f["avg_paragraph_len"] = safe_divide(word_count, paragraph_count)

    # ── 3. variability ──
    f["word_len_std"] = float(np.std(word_lengths)) if word_lengths else 0.0
    f["sentence_len_std"] = float(np.std(sentence_lengths)) if sentence_lengths else 0.0
    f["max_sentence_len"] = max(sentence_lengths) if sentence_lengths else 0
    f["min_sentence_len"] = min(sentence_lengths) if sentence_lengths else 0

    # ── 4. lexical richness ──
    f["type_token_ratio"] = safe_divide(unique_count, word_count)
    f["hapax_legomena_ratio"] = safe_divide(hapax, word_count)
    f["long_word_ratio"] = safe_divide(sum(1 for w in words if len(w) >= 6), word_count)

    # Yule's K
    if word_count > 0:
        freq_spectrum = Counter(freq_values)
        s2 = sum(i * i * vi for i, vi in freq_spectrum.items())
        yk = 10000.0 * (s2 - word_count) / (word_count * word_count) if word_count > 1 else 0.0
        f["yules_k"] = yk
    else:
        f["yules_k"] = 0.0

    # Simpson's diversity
    n = word_count
    if n > 1:
        f["simpsons_diversity"] = 1.0 - sum(v * (v - 1) for v in freq_values) / (n * (n - 1))
    else:
        f["simpsons_diversity"] = 0.0

    # Brunet's W
    if word_count > 0 and unique_count > 0:
        f["brunet_w"] = word_count ** (unique_count ** -0.172)
    else:
        f["brunet_w"] = 0.0

    # ── 5. punctuation / formatting ──
    f["stopword_ratio"] = safe_divide(stopword_count, word_count)
    f["punct_ratio"] = safe_divide(punct_count, char_count)
    f["comma_ratio"] = safe_divide(text.count(","), char_count)
    f["semicolon_ratio"] = safe_divide(text.count(";"), char_count)
    f["question_ratio"] = safe_divide(text.count("?"), char_count)
    f["exclamation_ratio"] = safe_divide(text.count("!"), char_count)
    f["colon_ratio"] = safe_divide(text.count(":"), char_count)
    f["parenthesis_ratio"] = safe_divide(text.count("(") + text.count(")"), char_count)
    f["uppercase_ratio"] = safe_divide(upper_count, alpha_count)
    f["digit_ratio"] = safe_divide(digit_count, char_count)

    # ── 6. structure ──
    f["transition_per_100w"] = safe_divide(transition_count * 100, word_count)
    f["bullet_point_count"] = len(BULLET_RE.findall(text))
    f["number_count"] = len(re.findall(r"\d+", text))

    # repeated 3-gram ratio
    if word_count >= 3:
        trigrams = [tuple(lower_words[i : i + 3]) for i in range(word_count - 2)]
        tri_freq = Counter(trigrams)
        repeated_tri = sum(1 for v in tri_freq.values() if v > 1)
        f["repeated_3gram_ratio"] = safe_divide(repeated_tri, len(trigrams))
    else:
        f["repeated_3gram_ratio"] = 0.0

    # ── 7. readability (textstat) ──
    if textstat is not None and word_count > 0:
        f["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        f["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
        f["gunning_fog"] = textstat.gunning_fog(text)
        f["smog_index"] = textstat.smog_index(text)
        f["coleman_liau_index"] = textstat.coleman_liau_index(text)
        f["automated_readability_index"] = textstat.automated_readability_index(text)
        f["dale_chall_score"] = textstat.dale_chall_readability_score(text)
    else:
        for k in [
            "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
            "smog_index", "coleman_liau_index", "automated_readability_index",
            "dale_chall_score",
        ]:
            f[k] = 0.0

    return f


# ── GPU: BERT embedding ───────────────────────────────────────────────
def compute_embeddings(texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Return (N, 384) embeddings using a small sentence-transformer on GPU."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


# ── GPU: GPT-2 perplexity ─────────────────────────────────────────────
def compute_perplexity(texts: list[str], batch_size: int = 32, max_len: int = 512) -> np.ndarray:
    """Return per-text perplexity using GPT-2 on GPU."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = torch.device("cuda")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    ppls = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
        # per-sample loss
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = enc["input_ids"][:, 1:].contiguous()
        attn = enc["attention_mask"][:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        # mask padding
        masked_loss = (loss * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        ppls.extend(torch.exp(masked_loss).cpu().tolist())

        if (i // batch_size) % 50 == 0:
            print(f"  perplexity: {i}/{len(texts)}")

    return np.array(ppls)


# ── plotting helpers ──────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], fig_dir: Path):
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr, cmap="RdBu_r", center=0, fmt=".1f", square=True,
                linewidths=0.3, cbar_kws={"shrink": 0.7},
                xticklabels=True, yticklabels=True)
    plt.title("Feature Correlation Matrix", fontsize=14)
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "correlation_heatmap.png", dpi=180)
    plt.close()
    print(f"  saved {fig_dir / 'correlation_heatmap.png'}")


def plot_tsne(X: np.ndarray, y: np.ndarray, fig_dir: Path, sample: int = 5000):
    idx = np.random.RandomState(42).choice(len(X), min(sample, len(X)), replace=False)
    Xs = X[idx]
    ys = y[idx]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(Xs)
    plt.figure(figsize=(8, 6))
    for label, name, color in [(0, "human", "#4974a5"), (1, "chatgpt", "#d16f4f")]:
        mask = ys == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.5, label=name, c=color)
    plt.legend()
    plt.title("t-SNE of extended features")
    plt.tight_layout()
    plt.savefig(fig_dir / "tsne_extended.png", dpi=180)
    plt.close()
    print(f"  saved {fig_dir / 'tsne_extended.png'}")


def plot_pca_2d(X: np.ndarray, y: np.ndarray, fig_dir: Path, sample: int = 8000):
    idx = np.random.RandomState(42).choice(len(X), min(sample, len(X)), replace=False)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X[idx])
    ys = y[idx]
    plt.figure(figsize=(8, 6))
    for label, name, color in [(0, "human", "#4974a5"), (1, "chatgpt", "#d16f4f")]:
        mask = ys == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.5, label=name, c=color)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend()
    plt.title("PCA of extended features")
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_extended.png", dpi=180)
    plt.close()
    print(f"  saved {fig_dir / 'pca_extended.png'}")


def plot_shap_summary(clf, X_train: pd.DataFrame, feature_names: list[str], fig_dir: Path):
    if shap is None:
        print("  shap not installed, skipping")
        return
    sample_idx = np.random.RandomState(42).choice(len(X_train), min(2000, len(X_train)), replace=False)
    X_sample = X_train.iloc[sample_idx]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  saved {fig_dir / 'shap_summary.png'}")


def plot_domain_feature_comparison(df: pd.DataFrame, numeric_cols: list[str], fig_dir: Path):
    """Box plot of key features across source domains, split by label."""
    top_features = [
        "avg_sentence_len", "type_token_ratio", "transition_per_100w",
        "flesch_reading_ease", "hapax_legomena_ratio", "repeated_3gram_ratio",
    ]
    top_features = [f for f in top_features if f in numeric_cols]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    for i, feat in enumerate(top_features):
        if i >= len(axes):
            break
        ax = axes[i]
        subset = df[["source", "label_name", feat]].copy()
        sns.boxplot(data=subset, x="source", y=feat, hue="label_name",
                    palette={"human": "#4974a5", "chatgpt": "#d16f4f"}, ax=ax,
                    fliersize=1)
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=30)
        if i > 0:
            ax.get_legend().remove()
    plt.suptitle("Feature distribution by domain and label", fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "domain_feature_comparison.png", dpi=180)
    plt.close()
    print(f"  saved {fig_dir / 'domain_feature_comparison.png'}")


def plot_model_comparison(results: dict, fig_dir: Path):
    names = list(results.keys())
    aucs = [results[n]["auc"] for n in names]
    accs = [results[n]["acc"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, aucs, w, label="ROC AUC", color="#4974a5")
    ax.bar(x + w / 2, accs, w, label="Accuracy", color="#d16f4f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.85, 1.0)
    ax.legend()
    ax.set_title("Model comparison")
    for i, (a, c) in enumerate(zip(aucs, accs)):
        ax.text(i - w / 2, a + 0.003, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + w / 2, c + 0.003, f"{c:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison.png", dpi=180)
    plt.close()
    print(f"  saved {fig_dir / 'model_comparison.png'}")


def plot_ablation(ablation_results: list[dict], fig_dir: Path):
    df_abl = pd.DataFrame(ablation_results).sort_values("auc", ascending=True)
    ax = df_abl.plot.barh(x="feature_group", y="auc", figsize=(8, 5), color="#4f8f7b", legend=False)
    ax.set_xlabel("ROC AUC")
    ax.set_title("Feature group ablation (XGBoost AUC)")
    ax.set_xlim(max(0.5, df_abl["auc"].min() - 0.05), 1.0)
    for i, row in enumerate(df_abl.itertuples()):
        ax.text(row.auc + 0.002, i, f"{row.auc:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_ablation.png", dpi=180)
    plt.close()
    print(f"  saved {fig_dir / 'feature_ablation.png'}")


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/hc3_flat.csv")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--emb-pca-dim", type=int, default=50)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path("data/processed/hc3_extended_features.csv")

    if cache_path.exists() and not args.recompute:
        print(f"Loading cached features from {cache_path}...")
        feat_df = pd.read_csv(cache_path)
        print(f"  rows: {len(feat_df):,}")
    else:
        # ── load data ──
        print("Loading data...")
        df = pd.read_csv(args.input)
        if args.max_rows:
            per_class = max(1, args.max_rows // df["label"].nunique())
            df = pd.concat(
                [part.sample(min(len(part), per_class), random_state=42)
                 for _, part in df.groupby("label")],
                ignore_index=True,
            ).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  rows: {len(df):,}")

        # ── CPU features ──
        print("Extracting CPU features...")
        feat_records = []
        texts = df["text"].fillna("").tolist()
        for i, text in enumerate(texts):
            feat_records.append(extract_extended_features(text))
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(texts)}")
        feat_df = pd.DataFrame(feat_records)

        cpu_feature_cols_tmp = list(feat_df.columns)
        print(f"  CPU features: {len(cpu_feature_cols_tmp)}")

        # ── GPU: embeddings ──
        if not args.skip_embedding:
            print("Computing BERT embeddings (GPU)...")
            embeddings = compute_embeddings(texts, batch_size=256)
            pca_emb = PCA(n_components=args.emb_pca_dim, random_state=42)
            emb_pca = pca_emb.fit_transform(embeddings)
            print(f"  PCA explained variance: {pca_emb.explained_variance_ratio_.sum():.3f}")
            for i in range(args.emb_pca_dim):
                feat_df[f"emb_pc{i}"] = emb_pca[:, i]

        # ── GPU: perplexity ──
        if not args.skip_perplexity:
            print("Computing GPT-2 perplexity (GPU)...")
            ppls = compute_perplexity(texts, batch_size=48)
            ppls = np.clip(ppls, 0, 10000)
            feat_df["gpt2_perplexity"] = ppls
            feat_df["log_perplexity"] = np.log1p(ppls)

        # ── assemble ──
        feat_df.insert(0, "label", df["label"].values)
        feat_df.insert(1, "label_name", df["label_name"].values)
        feat_df.insert(2, "source", df["source"].values)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(cache_path, index=False)
        print(f"  cached to {cache_path}")

    numeric_cols = [c for c in feat_df.columns if c not in {"label", "label_name", "source"}]
    emb_cols = [c for c in numeric_cols if c.startswith("emb_pc")]
    cpu_feature_cols = [c for c in numeric_cols if not c.startswith("emb_pc")]
    all_feature_cols = numeric_cols  # for models

    # ── plots: correlation, domain, t-SNE, PCA ──
    print("\n=== Generating analysis plots ===")
    # correlation (on CPU features only for readability)
    plot_correlation_heatmap(feat_df, cpu_feature_cols, fig_dir)

    # domain comparison
    plot_domain_feature_comparison(feat_df, cpu_feature_cols, fig_dir)

    # scaling for viz
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df[all_feature_cols].values)
    y = feat_df["label"].values

    plot_pca_2d(X_scaled, y, fig_dir)
    plot_tsne(X_scaled, y, fig_dir)

    # ── train/test split ──
    X = feat_df[all_feature_cols]
    y_series = feat_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_series, test_size=0.2, stratify=y_series, random_state=42
    )

    results = {}

    # ── Model 1: Logistic Regression ──
    print("\n=== Logistic Regression ===")
    scaler_lr = StandardScaler()
    X_tr_lr = scaler_lr.fit_transform(X_train)
    X_te_lr = scaler_lr.transform(X_test)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", random_state=42)
    lr.fit(X_tr_lr, y_train)
    y_pred_lr = lr.predict(X_te_lr)
    y_prob_lr = lr.predict_proba(X_te_lr)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    acc_lr = (y_pred_lr == y_test.values).mean()
    print(f"  AUC: {auc_lr:.4f}  Acc: {acc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["human", "chatgpt"]))
    results["LR (extended)"] = {"auc": auc_lr, "acc": acc_lr}

    # ── Model 2: XGBoost ──
    if xgb is not None:
        print("=== XGBoost ===")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            tree_method="hist", device="cuda", random_state=42,
        )
        xgb_clf.fit(X_train, y_train)
        y_pred_xgb = xgb_clf.predict(X_test)
        y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)
        acc_xgb = (y_pred_xgb == y_test.values).mean()
        print(f"  AUC: {auc_xgb:.4f}  Acc: {acc_xgb:.4f}")
        print(classification_report(y_test, y_pred_xgb, target_names=["human", "chatgpt"]))
        results["XGBoost"] = {"auc": auc_xgb, "acc": acc_xgb}

        # confusion matrix
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_xgb, display_labels=["human", "chatgpt"],
            cmap="Blues", values_format="d",
        )
        plt.title("XGBoost confusion matrix (extended)")
        plt.tight_layout()
        plt.savefig(fig_dir / "confusion_matrix_xgb.png", dpi=180)
        plt.close()

        # SHAP
        print("Computing SHAP values...")
        plot_shap_summary(xgb_clf, X_train, all_feature_cols, fig_dir)

    # ── Model 3: Random Forest ──
    print("=== Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    acc_rf = (y_pred_rf == y_test.values).mean()
    print(f"  AUC: {auc_rf:.4f}  Acc: {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=["human", "chatgpt"]))
    results["RandomForest"] = {"auc": auc_rf, "acc": acc_rf}

    # ── Model comparison plot ──
    plot_model_comparison(results, fig_dir)

    # ── Feature ablation (XGBoost only) ──
    if xgb is not None:
        print("\n=== Feature Ablation ===")
        feature_groups = {
            "basic_counts": ["char_count", "word_count", "sentence_count", "paragraph_count"],
            "averages": ["avg_word_len", "avg_sentence_len", "avg_paragraph_len"],
            "variability": ["word_len_std", "sentence_len_std", "max_sentence_len", "min_sentence_len"],
            "lexical_richness": ["type_token_ratio", "hapax_legomena_ratio", "long_word_ratio",
                                  "yules_k", "simpsons_diversity", "brunet_w"],
            "punctuation": ["stopword_ratio", "punct_ratio", "comma_ratio", "semicolon_ratio",
                             "question_ratio", "exclamation_ratio", "colon_ratio",
                             "parenthesis_ratio", "uppercase_ratio", "digit_ratio"],
            "structure": ["transition_per_100w", "bullet_point_count", "number_count",
                           "repeated_3gram_ratio"],
            "readability": ["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
                             "smog_index", "coleman_liau_index", "automated_readability_index",
                             "dale_chall_score"],
        }
        if emb_cols:
            feature_groups["embedding_pca"] = emb_cols
        if "gpt2_perplexity" in all_feature_cols:
            feature_groups["perplexity"] = ["gpt2_perplexity", "log_perplexity"]

        ablation_results = []
        for group_name, cols in feature_groups.items():
            valid_cols = [c for c in cols if c in all_feature_cols]
            if not valid_cols:
                continue
            xgb_abl = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                tree_method="hist", device="cuda", random_state=42,
            )
            xgb_abl.fit(X_train[valid_cols], y_train)
            prob = xgb_abl.predict_proba(X_test[valid_cols])[:, 1]
            auc_abl = roc_auc_score(y_test, prob)
            ablation_results.append({"feature_group": group_name, "auc": auc_abl, "n_features": len(valid_cols)})
            print(f"  {group_name} ({len(valid_cols)} feats): AUC={auc_abl:.4f}")

        plot_ablation(ablation_results, fig_dir)

    # ── LR feature importance (top 20) ──
    print("\n=== LR feature importance (top 20 by |coef|) ===")
    coef_series = pd.Series(lr.coef_[0], index=all_feature_cols)
    top20 = coef_series.abs().nlargest(20).index
    imp = coef_series[top20].sort_values()
    ax = imp.plot(kind="barh", figsize=(8, 6), color="#4f8f7b")
    ax.set_title("Top-20 LR coefficients (extended features)")
    ax.set_xlabel("standardized coefficient (positive → ChatGPT)")
    plt.tight_layout()
    plt.savefig(fig_dir / "lr_top20_coefficients.png", dpi=180)
    plt.close()
    print(f"  saved {fig_dir / 'lr_top20_coefficients.png'}")

    print("\n=== Done! All figures saved to", fig_dir, "===")


if __name__ == "__main__":
    main()
