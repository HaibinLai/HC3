#!/usr/bin/env python3
"""Extract full (stratified, balanced) token features for SemEval & TuringBench.

Usage:
    python src/run_token_full.py [--max_train 40000] [--max_test 10000]

Saves to data/processed/token_features_{dataset}_{split}_full.csv
"""

import sys, argparse
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"
sys.path.insert(0, str(BASE / "src"))

from run_token_features import extract_token_prob_features, load_model


def stratified_sample(df, label_col, max_n, seed=42):
    """Balanced stratified sample."""
    per_class = max_n // 2
    return pd.concat([
        g.sample(min(len(g), per_class), random_state=seed)
        for _, g in df.groupby(label_col)
    ]).reset_index(drop=True)


def run_semeval(model, tokenizer, device, max_train=40000, max_test=10000):
    semeval_dir = BASE / "data" / "external" / "semeval2024_task8" / "subtaskA_monolingual"
    train = pd.read_parquet(semeval_dir / "train-00000-of-00001.parquet")
    test = pd.read_parquet(semeval_dir / "test-00000-of-00001.parquet")

    train_sub = stratified_sample(train, "label", max_train)
    test_sub = stratified_sample(test, "label", max_test)

    print(f"\nSemEval train subset: {len(train_sub)} (labels: {train_sub['label'].value_counts().to_dict()})")
    print(f"SemEval test subset:  {len(test_sub)} (labels: {test_sub['label'].value_counts().to_dict()})")

    cache_tr = DATA / "token_features_semeval_train_full.csv"
    cache_te = DATA / "token_features_semeval_test_full.csv"

    if not cache_tr.exists():
        X_tr = extract_token_prob_features(train_sub["text"].tolist(), model, tokenizer, device, label="se_train")
        X_tr.to_csv(cache_tr, index=False)
        # Also save labels
        train_sub[["label"]].to_csv(DATA / "semeval_labels_train_full.csv", index=False)
    else:
        print(f"  Cache exists: {cache_tr}")

    if not cache_te.exists():
        X_te = extract_token_prob_features(test_sub["text"].tolist(), model, tokenizer, device, label="se_test")
        X_te.to_csv(cache_te, index=False)
        test_sub[["label"]].to_csv(DATA / "semeval_labels_test_full.csv", index=False)
    else:
        print(f"  Cache exists: {cache_te}")

    # Save the original indices so we can align 90-feat later
    # The stratified_sample reset_index, but the original indices in train/test are what we need
    # Re-do sampling to get original indices
    np.random.seed(42)
    idx_tr = []
    for lbl in sorted(train["label"].unique()):
        pool = train[train["label"] == lbl].index.tolist()
        idx_tr.extend(np.random.choice(pool, min(len(pool), max_train//2), replace=False).tolist())
    idx_te = []
    for lbl in sorted(test["label"].unique()):
        pool = test[test["label"] == lbl].index.tolist()
        idx_te.extend(np.random.choice(pool, min(len(pool), max_test//2), replace=False).tolist())
    pd.DataFrame({"orig_idx": sorted(idx_tr)}).to_csv(DATA / "semeval_idx_train_full.csv", index=False)
    pd.DataFrame({"orig_idx": sorted(idx_te)}).to_csv(DATA / "semeval_idx_test_full.csv", index=False)


def run_turingbench(model, tokenizer, device, max_train=40000, max_test=10000):
    from run_turingbench import load_turingbench
    train, test = load_turingbench()

    train_sub = stratified_sample(train, "binary_label", max_train)
    test_sub = stratified_sample(test, "binary_label", max_test)

    print(f"\nTuringBench train subset: {len(train_sub)} (labels: {train_sub['binary_label'].value_counts().to_dict()})")
    print(f"TuringBench test subset:  {len(test_sub)} (labels: {test_sub['binary_label'].value_counts().to_dict()})")

    cache_tr = DATA / "token_features_turingbench_train_full.csv"
    cache_te = DATA / "token_features_turingbench_test_full.csv"

    if not cache_tr.exists():
        X_tr = extract_token_prob_features(train_sub["text"].tolist(), model, tokenizer, device, label="tb_train")
        X_tr.to_csv(cache_tr, index=False)
        train_sub[["binary_label"]].to_csv(DATA / "turingbench_labels_train_full.csv", index=False)
    else:
        print(f"  Cache exists: {cache_tr}")

    if not cache_te.exists():
        X_te = extract_token_prob_features(test_sub["text"].tolist(), model, tokenizer, device, label="tb_test")
        X_te.to_csv(cache_te, index=False)
        test_sub[["binary_label"]].to_csv(DATA / "turingbench_labels_test_full.csv", index=False)
    else:
        print(f"  Cache exists: {cache_te}")

    np.random.seed(42)
    idx_tr = []
    for lbl in sorted(train["binary_label"].unique()):
        pool = train[train["binary_label"] == lbl].index.tolist()
        idx_tr.extend(np.random.choice(pool, min(len(pool), max_train//2), replace=False).tolist())
    idx_te = []
    for lbl in sorted(test["binary_label"].unique()):
        pool = test[test["binary_label"] == lbl].index.tolist()
        idx_te.extend(np.random.choice(pool, min(len(pool), max_test//2), replace=False).tolist())
    pd.DataFrame({"orig_idx": sorted(idx_tr)}).to_csv(DATA / "turingbench_idx_train_full.csv", index=False)
    pd.DataFrame({"orig_idx": sorted(idx_te)}).to_csv(DATA / "turingbench_idx_test_full.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train", type=int, default=40000)
    parser.add_argument("--max_test", type=int, default=10000)
    args = parser.parse_args()

    model, tokenizer, device = load_model()

    run_semeval(model, tokenizer, device, args.max_train, args.max_test)
    run_turingbench(model, tokenizer, device, args.max_train, args.max_test)

    print("\n=== Done ===")
