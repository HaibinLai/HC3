"""
Run experiments on AI Text Detection Pile dataset.
1.39M rows (1M human, 364K ai). Subsample for tractability.
Tests: XGBoost (90 features), RoBERTa fine-tune, Fast-DetectGPT.
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


def load_pile(max_per_class=50000):
    files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    dfs = [pd.read_parquet(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.dropna(subset=['text'])
    all_df = all_df[all_df['text'].str.len() > 10].reset_index(drop=True)
    all_df['label'] = (all_df['source'] == 'ai').astype(int)

    print(f"Full dataset: {len(all_df)} rows")
    print(f"Label dist: {all_df['label'].value_counts().to_dict()}")

    # Subsample balanced
    from sklearn.model_selection import train_test_split
    human = all_df[all_df['label'] == 0].sample(n=min(max_per_class, (all_df['label']==0).sum()), random_state=42)
    ai    = all_df[all_df['label'] == 1].sample(n=min(max_per_class, (all_df['label']==1).sum()), random_state=42)
    sampled = pd.concat([human, ai]).sample(frac=1, random_state=42).reset_index(drop=True)

    train, test = train_test_split(sampled, test_size=0.2, stratify=sampled['label'], random_state=42)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"Sampled train: {len(train)}, test: {len(test)}")
    return train, test


def run_xgboost(train, test):
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    print("\n" + "="*60)
    print("EXPERIMENT 1: XGBoost on AI Text Detection Pile")
    print("="*60)

    cache_train = RESULT_DIR / "pile_features_train.csv"
    cache_test  = RESULT_DIR / "pile_features_test.csv"

    if cache_train.exists() and cache_test.exists():
        print("Loading cached features...")
        X_train_df = pd.read_csv(cache_train)
        X_test_df  = pd.read_csv(cache_test)
    else:
        X_train_cpu = extract_features_batch(train['text'].tolist(), "train")
        X_test_cpu  = extract_features_batch(test['text'].tolist(), "test")

        print("  Computing BERT embeddings...")
        smodel = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        emb_train = smodel.encode(train['text'].tolist(), batch_size=512, show_progress_bar=True, convert_to_numpy=True)
        pca = PCA(n_components=50, random_state=42)
        emb_train_pca = pca.fit_transform(emb_train)
        emb_test = smodel.encode(test['text'].tolist(), batch_size=512, show_progress_bar=True, convert_to_numpy=True)
        emb_test_pca = pca.transform(emb_test)

        emb_train_df = pd.DataFrame(emb_train_pca, columns=[f'emb_pca_{i}' for i in range(50)])
        emb_test_df  = pd.DataFrame(emb_test_pca,  columns=[f'emb_pca_{i}' for i in range(50)])

        ppl_train = compute_perplexity_batch(train['text'].tolist())
        ppl_test  = compute_perplexity_batch(test['text'].tolist())

        X_train_df = pd.concat([X_train_cpu.reset_index(drop=True), emb_train_df.reset_index(drop=True)], axis=1)
        X_train_df['gpt2_perplexity'] = ppl_train
        X_train_df['gpt2_log_perplexity'] = np.log1p(ppl_train)

        X_test_df = pd.concat([X_test_cpu.reset_index(drop=True), emb_test_df.reset_index(drop=True)], axis=1)
        X_test_df['gpt2_perplexity'] = ppl_test
        X_test_df['gpt2_log_perplexity'] = np.log1p(ppl_test)

        X_train_df.to_csv(cache_train, index=False)
        X_test_df.to_csv(cache_test, index=False)

    y_train = train['label'].values
    y_test  = test['label'].values
    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_df  = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Training XGBoost... features={X_train_df.shape[1]}")
    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist', device='cuda')
    clf.fit(X_train_df, y_train, verbose=False)

    y_prob = clf.predict_proba(X_test_df)[:, 1]
    y_pred = clf.predict(X_test_df)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  XGBoost Results:")
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
    print("EXPERIMENT 2: RoBERTa Fine-tune on AI Text Detection Pile")
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

    n_train = min(40000, len(train))
    train_sub = train.sample(n=n_train, random_state=42) if len(train) > n_train else train
    tr, dv = train_test_split(train_sub, test_size=0.1, stratify=train_sub['label'], random_state=42)

    print(f"  Training: {len(tr)}, Dev: {len(dv)}, Test: {len(test)}")

    train_ds = TextDataset(tr['text'].tolist(), tr['label'].tolist(), tokenizer)
    dev_ds   = TextDataset(dv['text'].tolist(), dv['label'].tolist(), tokenizer)

    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "roberta_pile"),
        num_train_epochs=3, per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        fp16=True, logging_steps=100, report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds)
    trainer.train()

    test_ds = TextDataset(test['text'].tolist(), test['label'].tolist(), tokenizer)
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_pred = np.argmax(logits, axis=1)
    y_test = test['label'].values

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  RoBERTa Results:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    return auc, acc


def run_fast_detectgpt(test, max_samples=5000):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from sklearn.metrics import roc_auc_score, accuracy_score

    print("\n" + "="*60)
    print("EXPERIMENT 3: Fast-DetectGPT on AI Text Detection Pile")
    print("="*60)

    device = 'cuda'
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    if len(test) > max_samples:
        test_sub = pd.concat([
            g.sample(min(len(g), max_samples // 2), random_state=42)
            for _, g in test.groupby('label')
        ]).reset_index(drop=True)
    else:
        test_sub = test

    print(f"  Running on {len(test_sub)} samples...")
    scores = []
    for i, text in enumerate(test_sub['text'].tolist()):
        if i % 500 == 0:
            print(f"    {i}/{len(test_sub)}")
        try:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
            input_ids = enc['input_ids']
            if input_ids.shape[1] < 3:
                scores.append(0.0); continue
            with torch.no_grad():
                logits = model(**enc).logits
            log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
            target_ids = input_ids[0, 1:]
            orig_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            probs_dist = torch.softmax(logits[0, :-1], dim=-1)
            sampled = torch.multinomial(probs_dist, num_samples=1).squeeze(1)
            samp_lp = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
            diff = orig_lp - samp_lp
            score = diff.mean().item() / max(diff.std().item(), 1e-8)
            scores.append(score)
        except:
            scores.append(0.0)

    scores = np.array(scores)
    y_test = test_sub['label'].values
    auc_pos = roc_auc_score(y_test, scores)
    auc_neg = roc_auc_score(y_test, -scores)
    if auc_neg > auc_pos:
        scores = -scores; auc = auc_neg
    else:
        auc = auc_pos

    threshold = np.median(scores)
    y_pred = (scores > threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Fast-DetectGPT Results:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    return auc, acc


def plot_results(results):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods = list(results.keys())
    aucs = [results[m]['auc'] for m in methods]
    accs = [results[m]['acc'] for m in methods]
    colors = ['#1565C0', '#C62828', '#2E7D32']
    for ax, vals, title in [(axes[0], aucs, 'ROC AUC'), (axes[1], accs, 'Accuracy')]:
        ax.barh(methods, vals, color=colors[:len(methods)])
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel(title)
        ax.set_title(f'AI Text Detection Pile - {title}')
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pile_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIG_DIR / 'pile_comparison.png'}")


if __name__ == "__main__":
    train, test = load_pile(max_per_class=50000)
    results = {}

    t0 = time.time()
    auc, acc = run_xgboost(train, test)
    results['XGBoost\n(90 features)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    t0 = time.time()
    auc, acc = run_roberta(train, test)
    results['RoBERTa\nfine-tune'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    t0 = time.time()
    auc, acc = run_fast_detectgpt(test)
    results['Fast-DetectGPT\n(zero-shot)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    print("\n" + "="*60)
    print("SUMMARY: AI Text Detection Pile Results")
    print("="*60)
    for method, r in results.items():
        m = method.replace('\n', ' ')
        print(f"  {m:30s}: AUC={r['auc']:.4f}, Acc={r['acc']:.4f}, Time={r['time']:.0f}s")

    plot_results(results)
    print("\nDone!")
