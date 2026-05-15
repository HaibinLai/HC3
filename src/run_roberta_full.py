"""
Re-run RoBERTa with FULL training data on SemEval and TuringBench.
Previous runs used 40K subsample; now use all available training data.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import (RobertaTokenizerFast, RobertaForSequenceClassification,
                          Trainer, TrainingArguments)
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent


class TextDataset(Dataset):
    """Lazy tokenization to avoid OOM with large datasets."""
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length',
                             max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_roberta_full(name, train_texts, train_labels, test_texts, test_labels,
                     dev_texts=None, dev_labels=None, test_meta=None):
    print(f"\n{'='*60}")
    print(f"RoBERTa FULL training: {name}")
    print(f"{'='*60}")

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    if dev_texts is None:
        tr_texts, dv_texts, tr_labels, dv_labels = train_test_split(
            train_texts, train_labels, test_size=0.05, stratify=train_labels, random_state=42)
    else:
        tr_texts, dv_texts = train_texts, dev_texts
        tr_labels, dv_labels = train_labels, dev_labels

    print(f"  Train: {len(tr_texts)}, Dev: {len(dv_texts)}, Test: {len(test_texts)}")

    train_ds = TextDataset(tr_texts, tr_labels, tokenizer)
    dev_ds = TextDataset(dv_texts, dv_labels, tokenizer)

    output_dir = str(ROOT / "models" / f"roberta_{name}_full")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        logging_steps=200,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds)
    trainer.train()

    print(f"  Predicting on {len(test_texts)} test samples...")
    test_ds = TextDataset(test_texts, test_labels, tokenizer)
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_pred = np.argmax(logits, axis=1)
    y_test = np.array(test_labels)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  RoBERTa FULL Results on {name}:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Machine']))

    # Per-model breakdown if available
    if test_meta is not None and 'model' in test_meta.columns:
        print(f"\n  Per-model breakdown:")
        for m in sorted(test_meta['model'].dropna().unique()):
            mask = (test_meta['model'] == m).values
            if mask.sum() < 10:
                continue
            if len(set(y_test[mask])) > 1:
                m_auc = roc_auc_score(y_test[mask], probs[mask])
            else:
                m_auc = -1
            m_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {m:20s}: AUC={m_auc:.4f}, Acc={m_acc:.4f}, n={mask.sum()}")

    return auc, acc


if __name__ == "__main__":
    results = {}

    # ── 1. SemEval (full 120K train) ──
    print("Loading SemEval...")
    DATA_SE = ROOT / "data" / "external" / "semeval2024_task8" / "subtaskA_monolingual"
    se_train = pd.read_parquet(DATA_SE / "train-00000-of-00001.parquet")
    se_dev = pd.read_parquet(DATA_SE / "dev-00000-of-00001.parquet")
    se_test = pd.read_parquet(DATA_SE / "test-00000-of-00001.parquet")
    print(f"  SemEval: train={len(se_train)}, dev={len(se_dev)}, test={len(se_test)}")

    t0 = time.time()
    auc, acc = run_roberta_full(
        "semeval",
        se_train['text'].tolist(), se_train['label'].tolist(),
        se_test['text'].tolist(), se_test['label'].tolist(),
        dev_texts=se_dev['text'].tolist(), dev_labels=se_dev['label'].tolist(),
        test_meta=se_test,
    )
    results['SemEval'] = {'auc': auc, 'acc': acc, 'train_n': len(se_train), 'time': time.time() - t0}

    # Free memory
    del se_train, se_dev, se_test
    torch.cuda.empty_cache()

    # ── 2. TuringBench (full train) ──
    print("\nLoading TuringBench...")
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
    tb_all['binary_label'] = (tb_all['model'] != 'AA').astype(int)
    tb_all = tb_all.dropna(subset=['text'])
    tb_all = tb_all[tb_all['text'].str.len() > 10].reset_index(drop=True)
    tb_train = tb_all[tb_all['split'] == 'train'].reset_index(drop=True)
    tb_test = tb_all[tb_all['split'] == 'test'].reset_index(drop=True)
    print(f"  TuringBench: train={len(tb_train)}, test={len(tb_test)}")

    t0 = time.time()
    auc, acc = run_roberta_full(
        "turingbench",
        tb_train['text'].tolist(), tb_train['binary_label'].tolist(),
        tb_test['text'].tolist(), tb_test['binary_label'].tolist(),
        test_meta=tb_test,
    )
    results['TuringBench'] = {'auc': auc, 'acc': acc, 'train_n': len(tb_train), 'time': time.time() - t0}

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY: RoBERTa Full-Train Results")
    print("="*60)
    print(f"  {'Dataset':<15s} {'Train N':>10s} {'AUC (40K)':>10s} {'AUC (Full)':>10s} {'Acc (Full)':>10s} {'Time':>8s}")
    prev = {'SemEval': 0.6278, 'TuringBench': 0.6245}
    for name, r in results.items():
        print(f"  {name:<15s} {r['train_n']:>10d} {prev[name]:>10.4f} {r['auc']:>10.4f} {r['acc']:>10.4f} {r['time']:>7.0f}s")
    print("\nDone!")
