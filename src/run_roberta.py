"""
Fine-tune RoBERTa-base on HC3 for human vs ChatGPT classification.

Usage:
    python src/run_roberta.py                    # full run
    python src/run_roberta.py --max-rows 30000   # quick pilot
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")

from data_splits import get_splits


class HC3Dataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 512):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    auc = roc_auc_score(labels, probs)
    acc = (preds == labels).mean()
    return {"auc": auc, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--model-dir", default="models/roberta_hc3")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    print("Loading data...")
    train_df, test_df = get_splits(args.max_rows)

    # split train into train/val (90/10)
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.iloc[:val_size].reset_index(drop=True)
    train_df = train_df.iloc[val_size:].reset_index(drop=True)
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # ── tokenizer & model ──
    print("Loading RoBERTa...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    )

    # ── datasets ──
    print("Tokenizing...")
    train_texts = train_df["text"].fillna("").tolist()
    val_texts = val_df["text"].fillna("").tolist()
    test_texts = test_df["text"].fillna("").tolist()

    train_dataset = HC3Dataset(train_texts, train_df["label"].tolist(), tokenizer, args.max_len)
    val_dataset = HC3Dataset(val_texts, val_df["label"].tolist(), tokenizer, args.max_len)
    test_dataset = HC3Dataset(test_texts, test_df["label"].tolist(), tokenizer, args.max_len)

    # ── training ──
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        logging_steps=100,
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n=== Training RoBERTa ===")
    trainer.train()

    # ── evaluate on test ──
    print("\n=== Evaluating on test set ===")
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    labels = test_df["label"].values

    auc = roc_auc_score(labels, probs)
    acc = (preds == labels).mean()
    print(f"ROC AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=["human", "chatgpt"]))

    # confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        labels, preds, display_labels=["human", "chatgpt"],
        cmap="Blues", values_format="d",
    )
    plt.title(f"RoBERTa confusion matrix (AUC={auc:.4f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_roberta.png", dpi=180)
    plt.close()
    print(f"Saved {fig_dir / 'confusion_matrix_roberta.png'}")

    # ── short text analysis ──
    print("\n=== Short text analysis (<100 words) ===")
    word_counts = test_df["text"].fillna("").str.split().str.len()
    short_mask = word_counts < 100
    if short_mask.sum() > 50:
        short_auc = roc_auc_score(labels[short_mask], probs[short_mask])
        short_acc = (preds[short_mask] == labels[short_mask]).mean()
        print(f"Short text ({short_mask.sum()} samples): AUC={short_auc:.4f}  Acc={short_acc:.4f}")
    else:
        print(f"Too few short texts ({short_mask.sum()}), skipping.")

    # save results
    results = {
        "method": "RoBERTa fine-tune",
        "auc": auc,
        "accuracy": acc,
        "n_test": len(labels),
    }
    results_path = Path("data/processed/roberta_results.csv")
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
