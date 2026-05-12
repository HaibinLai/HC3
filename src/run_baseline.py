from __future__ import annotations

import argparse
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import textstat
except ImportError:  # pragma: no cover
    textstat = None


TRANSITION_PHRASES = [
    "first",
    "firstly",
    "second",
    "secondly",
    "third",
    "finally",
    "in conclusion",
    "to conclude",
    "overall",
    "therefore",
    "however",
    "moreover",
    "furthermore",
    "on the other hand",
    "as a result",
    "for example",
    "in addition",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+")


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def extract_features(text: str) -> dict[str, float]:
    words = WORD_RE.findall(text)
    lower_words = [word.lower() for word in words]
    word_count = len(words)
    unique_count = len(set(lower_words))
    sentence_count = max(1, len(SENTENCE_RE.findall(text)))
    char_count = len(text)
    alpha_count = sum(ch.isalpha() for ch in text)
    upper_count = sum(ch.isupper() for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    punct_count = sum(ch in string.punctuation for ch in text)
    transition_count = sum(text.lower().count(phrase) for phrase in TRANSITION_PHRASES)
    stopword_count = sum(word in STOPWORDS for word in lower_words)

    features = {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_len": safe_divide(sum(len(word) for word in words), word_count),
        "avg_sentence_len": safe_divide(word_count, sentence_count),
        "type_token_ratio": safe_divide(unique_count, word_count),
        "stopword_ratio": safe_divide(stopword_count, word_count),
        "punct_ratio": safe_divide(punct_count, char_count),
        "comma_ratio": safe_divide(text.count(","), char_count),
        "semicolon_ratio": safe_divide(text.count(";"), char_count),
        "question_ratio": safe_divide(text.count("?"), char_count),
        "exclamation_ratio": safe_divide(text.count("!"), char_count),
        "uppercase_ratio": safe_divide(upper_count, alpha_count),
        "digit_ratio": safe_divide(digit_count, char_count),
        "transition_per_100_words": safe_divide(transition_count * 100, word_count),
    }

    if textstat is not None:
        features["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
    else:
        features["flesch_reading_ease"] = 0.0
        features["flesch_kincaid_grade"] = 0.0

    return features


def plot_class_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    counts = df["label_name"].value_counts().reindex(["human", "chatgpt"])
    ax = counts.plot(kind="bar", color=["#4974a5", "#d16f4f"], rot=0)
    ax.set_title("HC3 flattened class distribution")
    ax.set_xlabel("")
    ax.set_ylabel("documents")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=180)
    plt.close()


def plot_numeric_importance(model: Pipeline, numeric_cols: list[str], fig_dir: Path) -> None:
    clf = model.named_steps["classifier"]
    numeric_width = len(numeric_cols)
    coefs = clf.coef_[0][:numeric_width]
    importance = pd.Series(coefs, index=numeric_cols).sort_values()

    ax = importance.plot(kind="barh", figsize=(8, 6), color="#4f8f7b")
    ax.set_title("Numeric feature coefficients: positive means ChatGPT")
    ax.set_xlabel("standardized logistic regression coefficient")
    plt.tight_layout()
    plt.savefig(fig_dir / "numeric_feature_coefficients.png", dpi=180)
    plt.close()


def build_model(numeric_cols: list[str], max_features: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=3,
                    max_df=0.9,
                    ngram_range=(1, 2),
                    max_features=max_features,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
                "text",
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interpretable HC3 baseline.")
    parser.add_argument("--input", default="data/processed/hc3_flat.csv")
    parser.add_argument("--features-out", default="data/processed/hc3_features.csv")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-tfidf-features", type=int, default=30000)
    parser.add_argument(
        "--recompute-features",
        action="store_true",
        help="Recompute numeric text features even when the cached CSV exists.",
    )
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.max_rows and args.features_out == parser.get_default("features_out"):
        features_path = Path(f"data/processed/hc3_features_sample_{args.max_rows}.csv")
    else:
        features_path = Path(args.features_out)
    can_reuse_features = (
        features_path.exists() and args.max_rows is None and not args.recompute_features
    )

    if can_reuse_features:
        feature_df = pd.read_csv(features_path)
    else:
        df = pd.read_csv(args.input)
        if args.max_rows:
            per_class = max(1, args.max_rows // df["label"].nunique())
            df = pd.concat(
                [
                    part.sample(min(len(part), per_class), random_state=42)
                    for _, part in df.groupby("label")
                ],
                ignore_index=True,
            ).sample(frac=1, random_state=42).reset_index(drop=True)

        feature_df = pd.DataFrame([extract_features(text) for text in df["text"].fillna("")])
        feature_df.insert(0, "label", df["label"].to_numpy())
        feature_df.insert(1, "label_name", df["label_name"].to_numpy())
        feature_df.insert(2, "source", df["source"].to_numpy())
        feature_df.insert(3, "text", df["text"].to_numpy())
        features_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(features_path, index=False)

    plot_class_distribution(feature_df, fig_dir)

    numeric_cols = [
        col
        for col in feature_df.columns
        if col not in {"label", "label_name", "source", "text"}
    ]
    X = feature_df[["text", *numeric_cols]]
    y = feature_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model(numeric_cols, args.max_tfidf_features)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"rows used: {len(feature_df):,}")
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["human", "chatgpt"]))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["human", "chatgpt"],
        cmap="Blues",
        values_format="d",
    )
    plt.title("Baseline confusion matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    plot_numeric_importance(model, numeric_cols, fig_dir)


if __name__ == "__main__":
    main()
