"""
Run experiments on TuringBench dataset.
Binary classification: human (AA) vs machine (all TT_* models).
Tests: XGBoost (90 features), RoBERTa fine-tune, Fast-DetectGPT.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
TB_DIR = ROOT / "data" / "external" / "turingbench" / "extracted" / "TuringBench"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "data" / "processed"

# ── Load data ──────────────────────────────────────────────
def load_turingbench():
    dfs = []
    for subdir in sorted(TB_DIR.iterdir()):
        if subdir.name.startswith('.') or subdir.name == '__MACOSX':
            continue
        for split in ['train', 'test']:
            f = subdir / f'{split}.csv'
            if f.exists():
                df = pd.read_csv(f)
                df['split'] = split
                df['model'] = subdir.name
                dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.rename(columns={'Generation': 'text'})

    # Binary label: AA (human) = 0, TT_* (machine) = 1
    all_df['binary_label'] = (all_df['model'] != 'AA').astype(int)

    # Drop rows with missing text
    all_df = all_df.dropna(subset=['text'])
    all_df = all_df[all_df['text'].str.len() > 10].reset_index(drop=True)

    train = all_df[all_df['split'] == 'train'].reset_index(drop=True)
    test  = all_df[all_df['split'] == 'test'].reset_index(drop=True)

    print(f"TuringBench train: {len(train)}, test: {len(test)}")
    print(f"Train label dist: {train['binary_label'].value_counts().to_dict()}")
    print(f"Models: {sorted(train['model'].unique())}")
    return train, test


# ── Reuse feature extraction from run_semeval ──────────────
sys.path.insert(0, str(ROOT / 'src'))
from run_semeval import extract_features_batch, compute_perplexity_batch


# ── XGBoost ────────────────────────────────────────────────
def run_xgboost(train, test):
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    print("\n" + "="*60)
    print("EXPERIMENT 1: XGBoost on TuringBench")
    print("="*60)

    cache_train = RESULT_DIR / "turingbench_features_train.csv"
    cache_test  = RESULT_DIR / "turingbench_features_test.csv"

    if cache_train.exists() and cache_test.exists():
        print("Loading cached features...")
        X_train_df = pd.read_csv(cache_train)
        X_test_df  = pd.read_csv(cache_test)
    else:
        X_train_cpu = extract_features_batch(train['text'].tolist(), "train")
        X_test_cpu  = extract_features_batch(test['text'].tolist(), "test")

        print("  Computing BERT embeddings...")
        smodel = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        emb_train = smodel.encode(train['text'].tolist(), batch_size=512,
                                  show_progress_bar=True, convert_to_numpy=True)
        pca = PCA(n_components=50, random_state=42)
        emb_train_pca = pca.fit_transform(emb_train)
        emb_test = smodel.encode(test['text'].tolist(), batch_size=512,
                                 show_progress_bar=True, convert_to_numpy=True)
        emb_test_pca = pca.transform(emb_test)

        emb_train_df = pd.DataFrame(emb_train_pca, columns=[f'emb_pca_{i}' for i in range(50)])
        emb_test_df  = pd.DataFrame(emb_test_pca,  columns=[f'emb_pca_{i}' for i in range(50)])

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

    y_train = train['binary_label'].values
    y_test  = test['binary_label'].values

    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_df  = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Training XGBoost... features={X_train_df.shape[1]}")
    clf = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42,
        tree_method='hist', device='cuda'
    )
    clf.fit(X_train_df, y_train, verbose=False)

    y_prob = clf.predict_proba(X_test_df)[:, 1]
    y_pred = clf.predict(X_test_df)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  XGBoost Results on TuringBench:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Machine']))

    # Per-model
    print("  Per-model breakdown:")
    for m in sorted(test['model'].unique()):
        if m == 'AA':
            continue
        mask = test['model'] == m
        if mask.sum() < 5:
            continue
        # Combine with AA for AUC
        mask_all = (test['model'] == m) | (test['model'] == 'AA')
        m_auc = roc_auc_score(y_test[mask_all], y_prob[mask_all]) if len(set(y_test[mask_all])) > 1 else -1
        m_acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"    {m:20s}: AUC={m_auc:.4f}, Acc(machine)={m_acc:.4f}, n={mask.sum()}")

    return auc, acc, y_prob, y_pred


# ── RoBERTa ────────────────────────────────────────────────
def run_roberta(train, test):
    import torch
    from transformers import (RobertaTokenizerFast, RobertaForSequenceClassification,
                              Trainer, TrainingArguments)
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from torch.utils.data import Dataset

    print("\n" + "="*60)
    print("EXPERIMENT 2: RoBERTa Fine-tune on TuringBench")
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

    # Subsample train if too large
    n_train = min(40000, len(train))
    train_sub = train.sample(n=n_train, random_state=42) if len(train) > n_train else train
    # Use 10% as dev
    from sklearn.model_selection import train_test_split
    tr, dv = train_test_split(train_sub, test_size=0.1, stratify=train_sub['binary_label'], random_state=42)

    print(f"  Training: {len(tr)}, Dev: {len(dv)}, Test: {len(test)}")

    train_ds = TextDataset(tr['text'].tolist(), tr['binary_label'].tolist(), tokenizer)
    dev_ds   = TextDataset(dv['text'].tolist(), dv['binary_label'].tolist(), tokenizer)

    output_dir = str(ROOT / "models" / "roberta_turingbench")
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
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds)
    trainer.train()

    test_ds = TextDataset(test['text'].tolist(), test['binary_label'].tolist(), tokenizer)
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_pred = np.argmax(logits, axis=1)
    y_test = test['binary_label'].values

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  RoBERTa Results on TuringBench:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Machine']))

    # Per-model
    print("  Per-model breakdown:")
    for m in sorted(test['model'].unique()):
        if m == 'AA':
            continue
        mask_all = (test['model'] == m) | (test['model'] == 'AA')
        mask_m = test['model'] == m
        m_auc = roc_auc_score(y_test[mask_all], probs[mask_all]) if len(set(y_test[mask_all])) > 1 else -1
        m_acc = accuracy_score(y_test[mask_m], y_pred[mask_m])
        print(f"    {m:20s}: AUC={m_auc:.4f}, Acc(machine)={m_acc:.4f}, n={mask_m.sum()}")

    return auc, acc


# ── Fast-DetectGPT ─────────────────────────────────────────
def run_fast_detectgpt(test, max_samples=5000):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from sklearn.metrics import roc_auc_score, accuracy_score

    print("\n" + "="*60)
    print("EXPERIMENT 3: Fast-DetectGPT on TuringBench")
    print("="*60)

    device = 'cuda'
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    if len(test) > max_samples:
        test_sub = pd.concat([
            g.sample(min(len(g), max_samples // test['binary_label'].nunique()), random_state=42)
            for _, g in test.groupby('binary_label')
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
                scores.append(0.0)
                continue
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
    y_test = test_sub['binary_label'].values

    auc_pos = roc_auc_score(y_test, scores)
    auc_neg = roc_auc_score(y_test, -scores)
    if auc_neg > auc_pos:
        scores = -scores
        auc = auc_neg
    else:
        auc = auc_pos

    threshold = np.median(scores)
    y_pred = (scores > threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Fast-DetectGPT Results on TuringBench:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    return auc, acc


# ── Visualization ──────────────────────────────────────────
def plot_results(results):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods = list(results.keys())
    aucs = [results[m]['auc'] for m in methods]
    accs = [results[m]['acc'] for m in methods]
    colors = ['#1565C0', '#C62828', '#2E7D32']

    axes[0].barh(methods, aucs, color=colors[:len(methods)])
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xlabel('ROC AUC')
    axes[0].set_title('TuringBench - ROC AUC')
    for i, v in enumerate(aucs):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')

    axes[1].barh(methods, accs, color=colors[:len(methods)])
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_xlabel('Accuracy')
    axes[1].set_title('TuringBench - Accuracy')
    for i, v in enumerate(accs):
        axes[1].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'turingbench_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIG_DIR / 'turingbench_comparison.png'}")


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    train, test = load_turingbench()
    results = {}

    t0 = time.time()
    auc, acc, _, _ = run_xgboost(train, test)
    results['XGBoost\n(90 features)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    t0 = time.time()
    auc, acc = run_roberta(train, test)
    results['RoBERTa\nfine-tune'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    t0 = time.time()
    auc, acc = run_fast_detectgpt(test)
    results['Fast-DetectGPT\n(zero-shot)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    print("\n" + "="*60)
    print("SUMMARY: TuringBench Results")
    print("="*60)
    for method, r in results.items():
        m = method.replace('\n', ' ')
        print(f"  {m:30s}: AUC={r['auc']:.4f}, Acc={r['acc']:.4f}, Time={r['time']:.0f}s")

    plot_results(results)
    print("\nDone!")
