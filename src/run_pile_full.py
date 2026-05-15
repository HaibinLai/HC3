"""
Run FULL-SCALE experiments on AI Text Detection Pile.
Use all AI samples (364K) + balanced human samples (364K) = ~728K total.
XGBoost (90 features) + RoBERTa fine-tune only (skip Fast-DetectGPT for speed).
"""

import os, sys, time, warnings, glob
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external" / "ai_text_detection_pile" / "data"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / 'src'))
from run_semeval import extract_features_batch, compute_perplexity_batch
from multiprocessing import Pool, cpu_count


def _extract_worker(args):
    """Worker for parallel feature extraction (must be top-level for pickle)."""
    texts_chunk, chunk_label = args
    return extract_features_batch(texts_chunk, chunk_label)


def extract_features_parallel(texts, label="", n_workers=None):
    """Parallel CPU feature extraction using multiprocessing."""
    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    chunk_size = len(texts) // n_workers + 1
    chunks = [(texts[i:i+chunk_size], f"{label}_chunk{j}")
              for j, i in enumerate(range(0, len(texts), chunk_size))]

    print(f"  Parallel feature extraction: {len(texts)} texts, {n_workers} workers, {chunk_size} per chunk")

    with Pool(n_workers) as pool:
        results = pool.map(_extract_worker, chunks)

    return pd.concat(results, ignore_index=True)


def load_pile_full():
    files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    dfs = [pd.read_parquet(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.dropna(subset=['text'])
    all_df = all_df[all_df['text'].str.len() > 10].reset_index(drop=True)
    all_df['label'] = (all_df['source'] == 'ai').astype(int)

    ai_df = all_df[all_df['label'] == 1]
    human_df = all_df[all_df['label'] == 0].sample(n=len(ai_df), random_state=42)
    sampled = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(sampled, test_size=0.2, stratify=sampled['label'], random_state=42)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"Full Pile (balanced): train={len(train)}, test={len(test)}")
    print(f"Train labels: {train['label'].value_counts().to_dict()}")
    return train, test


def run_xgboost(train, test):
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    print("\n" + "="*60)
    print("EXPERIMENT 1: XGBoost (FULL Pile)")
    print("="*60)

    cache_train = RESULT_DIR / "pile_full_features_train.csv"
    cache_test  = RESULT_DIR / "pile_full_features_test.csv"

    if cache_train.exists() and cache_test.exists():
        print("Loading cached features...")
        X_train_df = pd.read_csv(cache_train)
        X_test_df  = pd.read_csv(cache_test)
    else:
        # CPU features - PARALLEL
        X_train_cpu = extract_features_parallel(train['text'].tolist(), "train")
        X_test_cpu  = extract_features_parallel(test['text'].tolist(), "test")

        # BERT embeddings
        print("  Computing BERT embeddings (train)...")
        smodel = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        emb_train = smodel.encode(train['text'].tolist(), batch_size=512,
                                  show_progress_bar=True, convert_to_numpy=True)
        pca = PCA(n_components=50, random_state=42)
        emb_train_pca = pca.fit_transform(emb_train)

        print("  Computing BERT embeddings (test)...")
        emb_test = smodel.encode(test['text'].tolist(), batch_size=512,
                                 show_progress_bar=True, convert_to_numpy=True)
        emb_test_pca = pca.transform(emb_test)

        emb_train_df = pd.DataFrame(emb_train_pca, columns=[f'emb_pca_{i}' for i in range(50)])
        emb_test_df  = pd.DataFrame(emb_test_pca,  columns=[f'emb_pca_{i}' for i in range(50)])

        # GPT-2 perplexity
        ppl_train = compute_perplexity_batch(train['text'].tolist())
        ppl_test  = compute_perplexity_batch(test['text'].tolist())

        X_train_df = pd.concat([X_train_cpu.reset_index(drop=True),
                                emb_train_df.reset_index(drop=True)], axis=1)
        X_train_df['gpt2_perplexity'] = ppl_train
        X_train_df['gpt2_log_perplexity'] = np.log1p(ppl_train)

        X_test_df = pd.concat([X_test_cpu.reset_index(drop=True),
                               emb_test_df.reset_index(drop=True)], axis=1)
        X_test_df['gpt2_perplexity'] = ppl_test
        X_test_df['gpt2_log_perplexity'] = np.log1p(ppl_test)

        X_train_df.to_csv(cache_train, index=False)
        X_test_df.to_csv(cache_test, index=False)
        print(f"  Cached to {cache_train}")

    y_train = train['label'].values
    y_test  = test['label'].values
    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_df  = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Training XGBoost... {X_train_df.shape}")
    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist', device='cuda')
    clf.fit(X_train_df, y_train, verbose=False)

    y_prob = clf.predict_proba(X_test_df)[:, 1]
    y_pred = clf.predict(X_test_df)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  XGBoost FULL Pile Results:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    return auc, acc


def run_roberta(train, test):
    import torch
    from transformers import (RobertaTokenizerFast, RobertaForSequenceClassification,
                              Trainer, TrainingArguments)
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset

    print("\n" + "="*60)
    print("EXPERIMENT 2: RoBERTa (FULL Pile)")
    print("="*60)

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=512):
            self.encodings = tokenizer(texts, truncation=True, padding=True,
                                       max_length=max_len, return_tensors='pt')
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # Use 80K for training (larger than before), full test
    n_train = min(80000, len(train))
    train_sub = train.sample(n=n_train, random_state=42) if len(train) > n_train else train
    tr, dv = train_test_split(train_sub, test_size=0.1, stratify=train_sub['label'], random_state=42)

    print(f"  Training: {len(tr)}, Dev: {len(dv)}, Test: {len(test)}")

    train_ds = TextDataset(tr['text'].tolist(), tr['label'].tolist(), tokenizer)
    dev_ds   = TextDataset(dv['text'].tolist(), dv['label'].tolist(), tokenizer)

    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "roberta_pile_full"),
        num_train_epochs=3, per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        fp16=True, logging_steps=200, report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds)
    trainer.train()

    # Predict on FULL test set (may be large, batch it)
    print(f"  Predicting on {len(test)} test samples...")
    test_ds = TextDataset(test['text'].tolist(), test['label'].tolist(), tokenizer)
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_pred = np.argmax(logits, axis=1)
    y_test = test['label'].values

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  RoBERTa FULL Pile Results:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    return auc, acc


if __name__ == "__main__":
    train, test = load_pile_full()

    results = {}

    t0 = time.time()
    auc, acc = run_xgboost(train, test)
    results['XGBoost'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    t0 = time.time()
    auc, acc = run_roberta(train, test)
    results['RoBERTa'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    print("\n" + "="*60)
    print("SUMMARY: AI Text Detection Pile FULL Results")
    print("="*60)
    print(f"  Dataset: {len(train)+len(test)} samples (balanced)")
    print(f"  Train: {len(train)}, Test: {len(test)}")
    for method, r in results.items():
        print(f"  {method:20s}: AUC={r['auc']:.4f}, Acc={r['acc']:.4f}, Time={r['time']:.0f}s")

    print("\n  Comparison with 100K subsample:")
    print(f"  {'':20s}  {'100K sample':>12s}  {'Full (~728K)':>12s}")
    print(f"  {'XGBoost':20s}  {'0.9789':>12s}  {results['XGBoost']['auc']:>12.4f}")
    print(f"  {'RoBERTa':20s}  {'0.9595':>12s}  {results['RoBERTa']['auc']:>12.4f}")
    print("\nDone!")
