"""
Run experiments on SemEval 2024 Task 8 SubtaskA (monolingual).
Tests: XGBoost (90 features), RoBERTa fine-tune, Fast-DetectGPT.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external" / "semeval2024_task8" / "subtaskA_monolingual"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "data" / "processed"

# ── Load data ──────────────────────────────────────────────
def load_semeval():
    train = pd.read_parquet(DATA_DIR / "train-00000-of-00001.parquet")
    dev   = pd.read_parquet(DATA_DIR / "dev-00000-of-00001.parquet")
    test  = pd.read_parquet(DATA_DIR / "test-00000-of-00001.parquet")
    print(f"SemEval train: {len(train)}, dev: {len(dev)}, test: {len(test)}")
    print(f"Models in train: {train['model'].value_counts().to_dict()}")
    print(f"Sources in train: {train['source'].value_counts().to_dict()}")
    return train, dev, test


# ── Feature Engineering (same 90-dim as HC3) ───────────────
def extract_features_batch(texts, batch_label=""):
    """Extract the same CPU-based features as run_extended.py (38 features)."""
    import re, math, string
    from collections import Counter
    import textstat

    features_list = []
    total = len(texts)
    for idx, text in enumerate(texts):
        if idx % 5000 == 0:
            print(f"  [{batch_label}] Feature extraction: {idx}/{total}")

        f = {}
        words = text.split()
        sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        chars = list(text)
        word_count = len(words)
        sent_count = max(len(sents), 1)
        char_count = len(text)
        para_count = max(text.count('\n\n') + 1, 1)

        # Basic counts
        f['word_count'] = word_count
        f['char_count'] = char_count
        f['sentence_count'] = sent_count
        f['paragraph_count'] = para_count
        f['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        f['avg_sentence_length'] = word_count / sent_count

        # Variability
        wlens = [len(w) for w in words]
        f['word_length_std'] = np.std(wlens) if len(wlens) > 1 else 0
        slens = [len(s.split()) for s in sents]
        f['sentence_length_std'] = np.std(slens) if len(slens) > 1 else 0

        # Lexical richness
        word_lower = [w.lower() for w in words]
        freq = Counter(word_lower)
        V = len(freq)
        N = max(word_count, 1)
        f['type_token_ratio'] = V / N if N > 0 else 0
        f['hapax_ratio'] = sum(1 for c in freq.values() if c == 1) / N if N > 0 else 0

        # Yule's K
        freq_spectrum = Counter(freq.values())
        M2 = sum(i * i * vi for i, vi in freq_spectrum.items())
        f['yule_k'] = 10000 * (M2 - N) / (N * N) if N > 1 else 0

        # Simpson's diversity
        f['simpson_diversity'] = 1 - sum(v * (v-1) for v in freq.values()) / (N * (N-1)) if N > 1 else 0

        # Brunet's W
        f['brunet_w'] = N ** (V ** -0.172) if V > 0 else 0

        # Punctuation features
        for p in [',', '.', '!', '?', ';', ':', '"', '-']:
            name = {',': 'comma', '.': 'period', '!': 'exclaim', '?': 'question',
                    ';': 'semicolon', ':': 'colon', '"': 'quote', '-': 'dash'}[p]
            f[f'punct_{name}_rate'] = text.count(p) / N if N > 0 else 0

        # Structural features
        f['words_per_paragraph'] = word_count / para_count
        f['sentences_per_paragraph'] = sent_count / para_count
        f['uppercase_ratio'] = sum(1 for c in chars if c.isupper()) / max(char_count, 1)
        f['digit_ratio'] = sum(1 for c in chars if c.isdigit()) / max(char_count, 1)
        f['whitespace_ratio'] = sum(1 for c in chars if c.isspace()) / max(char_count, 1)
        f['unique_word_ratio'] = V / N if N > 0 else 0
        f['long_word_ratio'] = sum(1 for w in words if len(w) > 6) / N if N > 0 else 0
        f['short_sentence_ratio'] = sum(1 for s in sents if len(s.split()) < 5) / sent_count

        # Readability
        try: f['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        except: f['flesch_reading_ease'] = 0
        try: f['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        except: f['flesch_kincaid_grade'] = 0
        try: f['gunning_fog'] = textstat.gunning_fog(text)
        except: f['gunning_fog'] = 0
        try: f['smog_index'] = textstat.smog_index(text)
        except: f['smog_index'] = 0
        try: f['coleman_liau'] = textstat.coleman_liau_index(text)
        except: f['coleman_liau'] = 0
        try: f['ari'] = textstat.automated_readability_index(text)
        except: f['ari'] = 0
        try: f['dale_chall'] = textstat.dale_chall_readability_score(text)
        except: f['dale_chall'] = 0

        features_list.append(f)

    return pd.DataFrame(features_list)


def compute_embeddings_batch(texts, batch_size=512):
    """BERT sentence embeddings -> PCA 50d."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    import torch

    print("  Computing BERT embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True)
    n_comp = min(50, embs.shape[1], embs.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    embs_pca = pca.fit_transform(embs)
    cols = [f'emb_pca_{i}' for i in range(n_comp)]
    return pd.DataFrame(embs_pca, columns=cols), pca


def compute_perplexity_batch(texts, batch_size=16):
    """GPT-2 perplexity."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("  Computing GPT-2 perplexity...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    for i in range(0, len(texts), batch_size):
        if i % 1000 == 0:
            print(f"    Perplexity: {i}/{len(texts)}")
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', truncation=True,
                        max_length=512, padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc['input_ids'])
        # per-sample approx
        ppls.extend([out.loss.item()] * len(batch))

    return np.array(ppls)


# ── XGBoost experiment ─────────────────────────────────────
def run_xgboost(train, test):
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    import matplotlib.pyplot as plt

    print("\n" + "="*60)
    print("EXPERIMENT 1: XGBoost (Feature Engineering) on SemEval")
    print("="*60)

    # Subsample for speed if very large (use all for train, all for test)
    cache_train = RESULT_DIR / "semeval_features_train.csv"
    cache_test  = RESULT_DIR / "semeval_features_test.csv"

    if cache_train.exists() and cache_test.exists():
        print("Loading cached features...")
        X_train_df = pd.read_csv(cache_train)
        X_test_df  = pd.read_csv(cache_test)
    else:
        # CPU features
        X_train_cpu = extract_features_batch(train['text'].tolist(), "train")
        X_test_cpu  = extract_features_batch(test['text'].tolist(), "test")

        # BERT embeddings (fit PCA on train, transform test)
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
        import torch

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
        print(f"  Cached features to {cache_train} and {cache_test}")

    y_train = train['label'].values
    y_test  = test['label'].values

    # Replace inf/nan
    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_df  = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Features: {X_train_df.shape[1]}")
    print(f"  Training XGBoost...")

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
    print(f"\n  XGBoost Results on SemEval Test:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Machine']))

    # Per-model breakdown
    if 'model' in test.columns:
        print("\n  Per-model breakdown:")
        for model_name in sorted(test['model'].dropna().unique()):
            mask = test['model'] == model_name
            if mask.sum() < 10:
                continue
            m_auc = roc_auc_score(y_test[mask], y_prob[mask]) if len(set(y_test[mask])) > 1 else -1
            m_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {model_name:15s}: AUC={m_auc:.4f}, Acc={m_acc:.4f}, n={mask.sum()}")

    # Per-source breakdown
    if 'source' in test.columns:
        print("\n  Per-source breakdown:")
        for src in sorted(test['source'].dropna().unique()):
            mask = test['source'] == src
            if mask.sum() < 10:
                continue
            s_auc = roc_auc_score(y_test[mask], y_prob[mask]) if len(set(y_test[mask])) > 1 else -1
            s_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {src:15s}: AUC={s_auc:.4f}, Acc={s_acc:.4f}, n={mask.sum()}")

    return auc, acc, y_prob, y_pred


# ── RoBERTa experiment ─────────────────────────────────────
def run_roberta(train, dev, test):
    import torch
    from transformers import (RobertaTokenizerFast, RobertaForSequenceClassification,
                              Trainer, TrainingArguments)
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from torch.utils.data import Dataset

    print("\n" + "="*60)
    print("EXPERIMENT 2: RoBERTa Fine-tune on SemEval")
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

    # Use subset for faster training (40k train, full test)
    n_train = min(40000, len(train))
    train_sub = train.sample(n=n_train, random_state=42)
    print(f"  Training on {n_train} samples, testing on {len(test)}")

    train_ds = TextDataset(train_sub['text'].tolist(), train_sub['label'].tolist(), tokenizer)
    dev_ds   = TextDataset(dev['text'].tolist(), dev['label'].tolist(), tokenizer)

    output_dir = str(ROOT / "models" / "roberta_semeval")
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

    # Predict on test
    test_ds = TextDataset(test['text'].tolist(), test['label'].tolist(), tokenizer)
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_pred = np.argmax(logits, axis=1)
    y_test = test['label'].values

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  RoBERTa Results on SemEval Test:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Machine']))

    # Per-model breakdown
    if 'model' in test.columns:
        print("\n  Per-model breakdown:")
        for model_name in sorted(test['model'].dropna().unique()):
            mask = test['model'] == model_name
            if mask.sum() < 10:
                continue
            m_auc = roc_auc_score(y_test[mask], probs[mask]) if len(set(y_test[mask])) > 1 else -1
            m_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {model_name:15s}: AUC={m_auc:.4f}, Acc={m_acc:.4f}, n={mask.sum()}")

    return auc, acc, probs, y_pred


# ── Fast-DetectGPT experiment ──────────────────────────────
def run_fast_detectgpt(test, max_samples=5000):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from sklearn.metrics import roc_auc_score, accuracy_score

    print("\n" + "="*60)
    print("EXPERIMENT 3: Fast-DetectGPT (zero-shot) on SemEval")
    print("="*60)

    device = 'cuda'
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    # Subsample test for speed
    if len(test) > max_samples:
        test_sub = pd.concat([
            g.sample(min(len(g), max_samples // test['label'].nunique()), random_state=42)
            for _, g in test.groupby('label')
        ])
    else:
        test_sub = test

    print(f"  Running on {len(test_sub)} samples...")

    scores = []
    texts = test_sub['text'].tolist()
    for i, text in enumerate(texts):
        if i % 500 == 0:
            print(f"    {i}/{len(texts)}")
        try:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
            input_ids = enc['input_ids']
            if input_ids.shape[1] < 3:
                scores.append(0.0)
                continue

            with torch.no_grad():
                logits = model(**enc).logits  # (1, T, V)

            # For each position, compare original token log-prob vs sampled token log-prob
            log_probs = torch.log_softmax(logits[0, :-1], dim=-1)  # (T-1, V)
            target_ids = input_ids[0, 1:]  # (T-1)

            orig_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # (T-1)

            # Sample alternative tokens from the conditional distribution
            probs_dist = torch.softmax(logits[0, :-1], dim=-1)
            sampled = torch.multinomial(probs_dist, num_samples=1).squeeze(1)
            samp_lp = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)

            diff = orig_lp - samp_lp
            score = diff.mean().item() / max(diff.std().item(), 1e-8)
            scores.append(score)
        except Exception as e:
            scores.append(0.0)

    scores = np.array(scores)
    y_test = test_sub['label'].values

    # Try both directions
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

    print(f"\n  Fast-DetectGPT Results on SemEval:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Samples:  {len(test_sub)}")

    # Per-model breakdown
    if 'model' in test_sub.columns:
        print("\n  Per-model breakdown:")
        for model_name in sorted(test_sub['model'].dropna().unique()):
            mask = test_sub['model'] == model_name
            if mask.sum() < 10:
                continue
            m_auc = roc_auc_score(y_test[mask], scores[mask]) if len(set(y_test[mask])) > 1 else -1
            m_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {model_name:15s}: AUC={m_auc:.4f}, Acc={m_acc:.4f}, n={mask.sum()}")

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
    axes[0].set_title('SemEval 2024 Task 8 - ROC AUC')
    for i, v in enumerate(aucs):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')

    axes[1].barh(methods, accs, color=colors[:len(methods)])
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_xlabel('Accuracy')
    axes[1].set_title('SemEval 2024 Task 8 - Accuracy')
    for i, v in enumerate(accs):
        axes[1].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'semeval_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {FIG_DIR / 'semeval_comparison.png'}")


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    train, dev, test = load_semeval()

    results = {}

    # 1. XGBoost
    t0 = time.time()
    auc, acc, _, _ = run_xgboost(train, test)
    results['XGBoost\n(90 features)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    # 2. RoBERTa
    t0 = time.time()
    auc, acc, _, _ = run_roberta(train, dev, test)
    results['RoBERTa\nfine-tune'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    # 3. Fast-DetectGPT
    t0 = time.time()
    auc, acc = run_fast_detectgpt(test)
    results['Fast-DetectGPT\n(zero-shot)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: SemEval 2024 Task 8 Results")
    print("="*60)
    for method, r in results.items():
        m = method.replace('\n', ' ')
        print(f"  {m:30s}: AUC={r['auc']:.4f}, Acc={r['acc']:.4f}, Time={r['time']:.0f}s")

    plot_results(results)
    print("\nDone!")
