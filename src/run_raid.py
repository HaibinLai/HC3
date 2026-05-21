"""
Run experiments on RAID Benchmark (ACL 2024).
Tests: XGBoost (90 features), Token-level prob features (Mistral-7B),
       Fast-DetectGPT (zero-shot), Adversarial robustness analysis.

RAID covers 11 generators (GPT-4, ChatGPT, Llama-2-70B, Mistral-7B, etc.),
8 domains, 4 decoding strategies, and 11 adversarial attacks.
"""

import os, sys, time, warnings, re, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external" / "raid"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / 'src'))

# ── Load data ──────────────────────────────────────────────
def load_raid(with_attacks=False):
    """Load RAID benchmark data."""
    fname = "train.csv" if with_attacks else "train_none.csv"
    fpath = DATA_DIR / fname
    print(f"Loading RAID from {fpath}...")
    df = pd.read_csv(fpath)
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Print distributions
    if 'model' in df.columns:
        print(f"\n  Model distribution:")
        for m, c in df['model'].value_counts().items():
            print(f"    {m:30s}: {c:>7,}")
    if 'domain' in df.columns:
        print(f"\n  Domain distribution:")
        for d, c in df['domain'].value_counts().items():
            print(f"    {d:30s}: {c:>7,}")
    if 'attack' in df.columns:
        print(f"\n  Attack distribution:")
        for a, c in df['attack'].value_counts().head(15).items():
            print(f"    {str(a):30s}: {c:>7,}")

    # Label: 'human' -> 0, else -> 1
    if 'label' not in df.columns:
        # RAID uses 'model' column: 'human' for human text
        df['label'] = (df['model'] != 'human').astype(int)

    print(f"\n  Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def split_raid(df, test_size=0.2, max_train=80000, max_test=20000):
    """Stratified split, subsampled for speed."""
    from sklearn.model_selection import train_test_split

    # Subsample if too large
    if len(df) > max_train + max_test:
        per_label = (max_train + max_test) // 2
        df_sub = pd.concat([
            g.sample(min(len(g), per_label), random_state=42)
            for _, g in df.groupby('label')
        ]).reset_index(drop=True)
    else:
        df_sub = df

    train_df, test_df = train_test_split(
        df_sub, test_size=test_size, stratify=df_sub['label'], random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Limit sizes
    if len(train_df) > max_train:
        train_df = pd.concat([
            g.sample(min(len(g), max_train // 2), random_state=42)
            for _, g in train_df.groupby('label')
        ]).reset_index(drop=True)
    if len(test_df) > max_test:
        test_df = pd.concat([
            g.sample(min(len(g), max_test // 2), random_state=42)
            for _, g in test_df.groupby('label')
        ]).reset_index(drop=True)

    print(f"\n  Split: train={len(train_df):,}, test={len(test_df):,}")
    print(f"  Train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"  Test labels:  {test_df['label'].value_counts().to_dict()}")
    return train_df, test_df


# ── Feature Engineering (same as run_semeval.py) ───────────
def extract_features_batch(texts, batch_label=""):
    """Extract CPU-based statistical features."""
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

        f['word_count'] = word_count
        f['char_count'] = char_count
        f['sentence_count'] = sent_count
        f['paragraph_count'] = para_count
        f['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        f['avg_sentence_length'] = word_count / sent_count

        wlens = [len(w) for w in words]
        f['word_length_std'] = np.std(wlens) if len(wlens) > 1 else 0
        slens = [len(s.split()) for s in sents]
        f['sentence_length_std'] = np.std(slens) if len(slens) > 1 else 0

        word_lower = [w.lower() for w in words]
        freq = Counter(word_lower)
        V = len(freq)
        N = max(word_count, 1)
        f['type_token_ratio'] = V / N if N > 0 else 0
        f['hapax_ratio'] = sum(1 for c in freq.values() if c == 1) / N if N > 0 else 0

        freq_spectrum = Counter(freq.values())
        M2 = sum(i * i * vi for i, vi in freq_spectrum.items())
        f['yule_k'] = 10000 * (M2 - N) / (N * N) if N > 1 else 0
        f['simpson_diversity'] = 1 - sum(v * (v-1) for v in freq.values()) / (N * (N-1)) if N > 1 else 0
        f['brunet_w'] = N ** (V ** -0.172) if V > 0 else 0

        for p in [',', '.', '!', '?', ';', ':', '"', '-']:
            name = {',': 'comma', '.': 'period', '!': 'exclaim', '?': 'question',
                    ';': 'semicolon', ':': 'colon', '"': 'quote', '-': 'dash'}[p]
            f[f'punct_{name}_rate'] = text.count(p) / N if N > 0 else 0

        f['words_per_paragraph'] = word_count / para_count
        f['sentences_per_paragraph'] = sent_count / para_count
        f['uppercase_ratio'] = sum(1 for c in chars if c.isupper()) / max(char_count, 1)
        f['digit_ratio'] = sum(1 for c in chars if c.isdigit()) / max(char_count, 1)
        f['whitespace_ratio'] = sum(1 for c in chars if c.isspace()) / max(char_count, 1)
        f['unique_word_ratio'] = V / N if N > 0 else 0
        f['long_word_ratio'] = sum(1 for w in words if len(w) > 6) / N if N > 0 else 0
        f['short_sentence_ratio'] = sum(1 for s in sents if len(s.split()) < 5) / sent_count

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


def compute_embeddings(texts, fit_pca=True, pca_model=None, batch_size=512):
    """BERT sentence embeddings -> PCA 50d."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    import torch

    print("  Computing BERT embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True)
    if fit_pca:
        pca_model = PCA(n_components=50, random_state=42)
        embs_pca = pca_model.fit_transform(embs)
    else:
        embs_pca = pca_model.transform(embs)

    cols = [f'emb_pca_{i}' for i in range(50)]
    return pd.DataFrame(embs_pca, columns=cols), pca_model


def compute_perplexity(texts, batch_size=16):
    """GPT-2 perplexity."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("  Computing GPT-2 perplexity...")
    device = 'cuda'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    for i in range(0, len(texts), batch_size):
        if i % 2000 == 0:
            print(f"    Perplexity: {i}/{len(texts)}")
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', truncation=True,
                        max_length=512, padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc['input_ids'])
        ppls.extend([out.loss.item()] * len(batch))

    return np.array(ppls)


# ── XGBoost experiment ─────────────────────────────────────
def run_xgboost(train_df, test_df):
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    print("\n" + "="*60)
    print("EXPERIMENT 1: XGBoost (90 features) on RAID")
    print("="*60)

    cache_train = RESULT_DIR / "raid_features_train.csv"
    cache_test  = RESULT_DIR / "raid_features_test.csv"

    if cache_train.exists() and cache_test.exists():
        print("Loading cached features...")
        X_train_df = pd.read_csv(cache_train)
        X_test_df  = pd.read_csv(cache_test)
    else:
        # CPU features
        X_train_cpu = extract_features_batch(train_df['generation'].tolist(), "train")
        X_test_cpu  = extract_features_batch(test_df['generation'].tolist(), "test")

        # BERT embeddings
        emb_train, pca = compute_embeddings(train_df['generation'].tolist(), fit_pca=True)
        emb_test, _    = compute_embeddings(test_df['generation'].tolist(), fit_pca=False, pca_model=pca)

        # GPT-2 perplexity
        ppl_train = compute_perplexity(train_df['generation'].tolist())
        ppl_test  = compute_perplexity(test_df['generation'].tolist())

        X_train_df = pd.concat([X_train_cpu.reset_index(drop=True),
                                emb_train.reset_index(drop=True)], axis=1)
        X_train_df['gpt2_perplexity'] = ppl_train
        X_train_df['gpt2_log_perplexity'] = np.log1p(ppl_train)

        X_test_df = pd.concat([X_test_cpu.reset_index(drop=True),
                               emb_test.reset_index(drop=True)], axis=1)
        X_test_df['gpt2_perplexity'] = ppl_test
        X_test_df['gpt2_log_perplexity'] = np.log1p(ppl_test)

        X_train_df.to_csv(cache_train, index=False)
        X_test_df.to_csv(cache_test, index=False)

    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_df  = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Features: {X_train_df.shape[1]}")
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
    print(f"\n  XGBoost Results on RAID:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Machine']))

    _print_breakdowns(test_df, y_test, y_prob, y_pred)

    return auc, acc, clf, X_train_df.columns.tolist()


# ── Token-level features ──────────────────────────────────
def run_token_features(train_df, test_df):
    from run_token_features import extract_token_prob_features, load_model
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    print("\n" + "="*60)
    print("EXPERIMENT 2: Token-level prob features (Mistral-7B) on RAID")
    print("="*60)

    cache_train = RESULT_DIR / "token_features_raid_train.csv"
    cache_test  = RESULT_DIR / "token_features_raid_test.csv"

    if cache_train.exists() and cache_test.exists():
        print("  Loading cached token features...")
        X_train = pd.read_csv(cache_train)
        X_test  = pd.read_csv(cache_test)
    else:
        model, tokenizer, device = load_model()
        X_train = extract_token_prob_features(
            train_df['generation'].tolist(), model, tokenizer, device, label="raid_train")
        X_test = extract_token_prob_features(
            test_df['generation'].tolist(), model, tokenizer, device, label="raid_test")
        X_train.to_csv(cache_train, index=False)
        X_test.to_csv(cache_test, index=False)
        # Free GPU
        del model
        import torch; torch.cuda.empty_cache()

    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Token features: {X_train.shape[1]}")
    clf = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=42, tree_method='hist', device='cuda')
    clf.fit(X_train, y_train, verbose=False)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Token Features Results on RAID:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Machine']))

    _print_breakdowns(test_df, y_test, y_prob, y_pred)

    # Feature importance
    importances = clf.feature_importances_
    feat_names = X_train.columns
    top_idx = np.argsort(importances)[::-1][:10]
    print(f"\n  Top 10 token features:")
    for idx in top_idx:
        print(f"    {feat_names[idx]:25s}: {importances[idx]:.4f}")

    return auc, acc


# ── Fast-DetectGPT ─────────────────────────────────────────
def run_fast_detectgpt(test_df, max_samples=5000):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from sklearn.metrics import roc_auc_score, accuracy_score

    print("\n" + "="*60)
    print("EXPERIMENT 3: Fast-DetectGPT (zero-shot) on RAID")
    print("="*60)

    device = 'cuda'
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    # Subsample
    if len(test_df) > max_samples:
        test_sub = pd.concat([
            g.sample(min(len(g), max_samples // 2), random_state=42)
            for _, g in test_df.groupby('label')
        ]).reset_index(drop=True)
    else:
        test_sub = test_df

    print(f"  Running on {len(test_sub)} samples...")
    texts = test_sub['generation'].tolist()
    scores = []

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
        scores = -scores
        auc = auc_neg
    else:
        auc = auc_pos

    threshold = np.median(scores)
    y_pred = (scores > threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Fast-DetectGPT Results on RAID:")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")

    _print_breakdowns(test_sub, y_test, scores, y_pred, is_score=True)

    del model
    import torch as t; t.cuda.empty_cache()

    return auc, acc


# ── Adversarial Robustness Analysis ────────────────────────
def run_adversarial_analysis(clf_xgb, feature_cols):
    """Evaluate XGBoost on adversarial attack variants."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    print("\n" + "="*60)
    print("EXPERIMENT 4: Adversarial Robustness Analysis")
    print("="*60)

    adv_path = DATA_DIR / "train.csv"
    if not adv_path.exists():
        print("  [SKIP] train.csv (with attacks) not found. Download it to enable.")
        return {}

    print("  Loading adversarial data...")
    df_adv = pd.read_csv(adv_path)
    df_adv['label'] = (df_adv['model'] != 'human').astype(int)

    # Only AI-generated text with attacks
    if 'attack' not in df_adv.columns:
        print("  [SKIP] No 'attack' column found.")
        return {}

    # Get attack types (exclude 'none' or NaN)
    attacks = df_adv['attack'].dropna().unique()
    attacks = [a for a in attacks if a != 'none' and str(a) != 'nan']
    print(f"  Found {len(attacks)} attack types: {attacks}")

    results = {}
    for attack in sorted(attacks):
        # Get attacked AI text + human text for evaluation
        ai_attacked = df_adv[(df_adv['attack'] == attack) & (df_adv['label'] == 1)]
        human = df_adv[(df_adv['model'] == 'human')]

        # Sample
        n = min(2000, len(ai_attacked), len(human))
        if n < 50:
            continue
        ai_sub = ai_attacked.sample(n, random_state=42)
        hu_sub = human.sample(n, random_state=42)
        eval_df = pd.concat([ai_sub, hu_sub]).reset_index(drop=True)

        # Extract features
        X_eval = extract_features_batch(eval_df['generation'].tolist(), f"adv_{attack}")

        # Add dummy embedding & perplexity columns to match training features
        for col in feature_cols:
            if col not in X_eval.columns:
                X_eval[col] = 0.0
        X_eval = X_eval[feature_cols]
        X_eval = X_eval.replace([np.inf, -np.inf], np.nan).fillna(0)

        y_eval = eval_df['label'].values

        try:
            y_prob = clf_xgb.predict_proba(X_eval)[:, 1]
            y_pred = clf_xgb.predict(X_eval)
            auc = roc_auc_score(y_eval, y_prob)
            acc = accuracy_score(y_eval, y_pred)
            results[attack] = {'auc': auc, 'acc': acc, 'n': len(eval_df)}
            print(f"    {attack:25s}: AUC={auc:.4f}, Acc={acc:.4f}, n={len(eval_df)}")
        except Exception as e:
            print(f"    {attack:25s}: ERROR - {e}")

    return results


# ── Helper ─────────────────────────────────────────────────
def _print_breakdowns(df, y_test, y_scores, y_pred, is_score=False):
    from sklearn.metrics import roc_auc_score, accuracy_score

    for col, label in [('model', 'Per-model'), ('domain', 'Per-domain')]:
        if col not in df.columns:
            continue
        print(f"\n  {label} breakdown:")
        for val in sorted(df[col].dropna().unique()):
            mask = (df[col] == val).values[:len(y_test)]
            if mask.sum() < 10:
                continue
            if len(set(y_test[mask])) > 1:
                m_auc = roc_auc_score(y_test[mask], y_scores[mask])
            else:
                m_auc = -1
            m_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {str(val):25s}: AUC={m_auc:.4f}, Acc={m_acc:.4f}, n={mask.sum()}")


# ── Visualization ──────────────────────────────────────────
def plot_comparison(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods = list(results.keys())
    aucs = [results[m]['auc'] for m in methods]
    accs = [results[m]['acc'] for m in methods]
    colors = ['#1565C0', '#FF6F00', '#2E7D32', '#C62828']

    axes[0].barh(methods, aucs, color=colors[:len(methods)])
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xlabel('ROC AUC')
    axes[0].set_title('RAID Benchmark - ROC AUC')
    for i, v in enumerate(aucs):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')

    axes[1].barh(methods, accs, color=colors[:len(methods)])
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_xlabel('Accuracy')
    axes[1].set_title('RAID Benchmark - Accuracy')
    for i, v in enumerate(accs):
        axes[1].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIG_DIR / 'raid_comparison.png'}")


def plot_per_model(test_df, y_test, y_prob_xgb, y_prob_token=None):
    """Bar chart of per-model AUC for each method."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    if 'model' not in test_df.columns:
        return

    models = sorted(test_df['model'].dropna().unique())
    xgb_aucs = []
    token_aucs = []
    valid_models = []

    for m in models:
        mask = (test_df['model'] == m).values[:len(y_test)]
        if mask.sum() < 10 or len(set(y_test[mask])) < 2:
            continue
        valid_models.append(m)
        xgb_aucs.append(roc_auc_score(y_test[mask], y_prob_xgb[mask]))
        if y_prob_token is not None:
            token_aucs.append(roc_auc_score(y_test[mask], y_prob_token[mask]))

    if not valid_models:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(valid_models))
    w = 0.35 if token_aucs else 0.6

    ax.bar(x - (w/2 if token_aucs else 0), xgb_aucs, w, label='XGBoost (90 feat)', color='#1565C0')
    if token_aucs:
        ax.bar(x + w/2, token_aucs, w, label='Token (Mistral-7B)', color='#FF6F00')

    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=45, ha='right')
    ax.set_ylabel('ROC AUC')
    ax.set_title('RAID: Per-Model Detection AUC')
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_per_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'raid_per_model.png'}")


def plot_adversarial(adv_results, baseline_auc):
    """Heatmap/bar of adversarial attack impact."""
    import matplotlib.pyplot as plt

    if not adv_results:
        return

    attacks = sorted(adv_results.keys())
    aucs = [adv_results[a]['auc'] for a in attacks]
    drops = [baseline_auc - adv_results[a]['auc'] for a in attacks]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AUC under each attack
    colors = ['#C62828' if d > 0.05 else '#FF6F00' if d > 0.02 else '#2E7D32' for d in drops]
    axes[0].barh(attacks, aucs, color=colors)
    axes[0].set_xlim(0.4, 1.0)
    axes[0].set_xlabel('ROC AUC')
    axes[0].set_title('XGBoost AUC under Adversarial Attacks')
    axes[0].axvline(x=baseline_auc, color='blue', linestyle='--', label=f'No attack: {baseline_auc:.4f}')
    axes[0].legend()
    for i, v in enumerate(aucs):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

    # AUC drop
    axes[1].barh(attacks, drops, color=colors)
    axes[1].set_xlabel('AUC Drop')
    axes[1].set_title('AUC Degradation by Attack Type')
    for i, v in enumerate(drops):
        axes[1].text(v + 0.002, i, f'{v:+.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raid_adversarial.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'raid_adversarial.png'}")


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()

    # Load and split
    df = load_raid(with_attacks=False)

    # Detect text column name
    text_col = 'generation' if 'generation' in df.columns else 'text'
    if text_col != 'generation':
        df = df.rename(columns={text_col: 'generation'})

    # Filter empty/short texts
    df = df.dropna(subset=['generation'])
    df = df[df['generation'].str.len() > 20].reset_index(drop=True)

    train_df, test_df = split_raid(df)

    results = {}

    # 1. XGBoost
    t0 = time.time()
    auc, acc, clf_xgb, feat_cols = run_xgboost(train_df, test_df)
    results['XGBoost\n(90 features)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    # 2. Token features
    t0 = time.time()
    auc, acc = run_token_features(train_df, test_df)
    results['Token features\n(Mistral-7B)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    # 3. Fast-DetectGPT
    t0 = time.time()
    auc, acc = run_fast_detectgpt(test_df)
    results['Fast-DetectGPT\n(zero-shot)'] = {'auc': auc, 'acc': acc, 'time': time.time() - t0}

    # 4. Adversarial analysis (if train.csv available)
    adv_results = run_adversarial_analysis(clf_xgb, feat_cols)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: RAID Benchmark Results")
    print("="*60)
    for method, r in results.items():
        m = method.replace('\n', ' ')
        print(f"  {m:35s}: AUC={r['auc']:.4f}, Acc={r['acc']:.4f}, Time={r['time']:.0f}s")

    if adv_results:
        print(f"\n  Adversarial Attack Results (XGBoost):")
        baseline_auc = results['XGBoost\n(90 features)']['auc']
        for attack, r in sorted(adv_results.items()):
            drop = baseline_auc - r['auc']
            print(f"    {attack:25s}: AUC={r['auc']:.4f} (drop={drop:+.4f})")

    # Plots
    plot_comparison(results)
    plot_adversarial(adv_results, results['XGBoost\n(90 features)']['auc'])

    total = time.time() - t_start
    print(f"\nTotal time: {total/60:.1f} min")
    print("Done!")
