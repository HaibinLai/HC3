"""
Token-level LLM probability features (inspired by SemEval-2024 Task 8 winner Genaios).
Extract per-token log prob, entropy, rank from Mistral-7B-Instruct,
aggregate into statistical features, combine with existing 90 features, train XGBoost.
Test on HC3 and SemEval.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / 'src'))
from data_splits import get_splits


def extract_token_prob_features(texts, model, tokenizer, device, batch_size=1, max_length=512, label=""):
    """
    For each text, extract token-level stats from LLM:
    - log_prob: log probability of each token given context
    - entropy: entropy of the predicted distribution at each position
    - rank: rank of the actual token in the predicted distribution
    - top1_prob: probability of the most likely token
    Returns a DataFrame with ~30 aggregated features per text.
    """
    model.eval()
    all_features = []

    for i, text in enumerate(texts):
        if i % 200 == 0:
            print(f"    [{label}] Token features: {i}/{len(texts)}")

        try:
            enc = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=max_length).to(device)
            input_ids = enc['input_ids']
            seq_len = input_ids.shape[1]

            if seq_len < 3:
                all_features.append(_empty_features())
                continue

            with torch.no_grad():
                logits = model(**enc).logits  # (1, T, V)

            # Shift: predict token t from context <t
            pred_logits = logits[0, :-1, :]  # (T-1, V)
            target_ids = input_ids[0, 1:]     # (T-1)
            T = pred_logits.shape[0]

            # Log probabilities
            log_probs = torch.log_softmax(pred_logits, dim=-1)  # (T-1, V)
            token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # (T-1)

            # Entropy of distribution at each position
            probs = torch.softmax(pred_logits, dim=-1)  # (T-1, V)
            entropy = -(probs * log_probs).sum(dim=-1)   # (T-1)

            # Rank of actual token (0-indexed, lower = more expected)
            # Vectorized: count how many logits are greater than the target token's logit
            target_logits = pred_logits.gather(1, target_ids.unsqueeze(1))  # (T, 1)
            ranks = (pred_logits > target_logits).sum(dim=-1).float()  # (T,)

            # Top-1 probability
            top1_prob = probs.max(dim=-1).values  # (T-1)

            # Top-5 cumulative probability
            top5_prob = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1).values.sum(dim=-1)

            # Convert to numpy (cast to float32 first for bfloat16 compatibility)
            lp = token_log_probs.float().cpu().numpy()
            ent = entropy.float().cpu().numpy()
            rk = ranks.float().cpu().numpy()
            t1p = top1_prob.float().cpu().numpy()
            t5p = top5_prob.float().cpu().numpy()

            f = {}
            # Log probability stats
            f['lp_mean'] = np.mean(lp)
            f['lp_std'] = np.std(lp)
            f['lp_min'] = np.min(lp)
            f['lp_max'] = np.max(lp)
            f['lp_median'] = np.median(lp)
            f['lp_q10'] = np.percentile(lp, 10)
            f['lp_q90'] = np.percentile(lp, 90)
            f['lp_skew'] = float(stats.skew(lp))
            f['lp_kurtosis'] = float(stats.kurtosis(lp))

            # Entropy stats
            f['ent_mean'] = np.mean(ent)
            f['ent_std'] = np.std(ent)
            f['ent_min'] = np.min(ent)
            f['ent_max'] = np.max(ent)
            f['ent_median'] = np.median(ent)
            f['ent_skew'] = float(stats.skew(ent))

            # Rank stats
            f['rank_mean'] = np.mean(rk)
            f['rank_std'] = np.std(rk)
            f['rank_median'] = np.median(rk)
            f['rank_q90'] = np.percentile(rk, 90)
            f['rank_top1_frac'] = np.mean(rk == 0)  # fraction of tokens that are top-1
            f['rank_top5_frac'] = np.mean(rk < 5)   # fraction in top-5
            f['rank_top10_frac'] = np.mean(rk < 10)
            f['rank_top100_frac'] = np.mean(rk < 100)

            # Top-1 and top-5 probability stats
            f['top1p_mean'] = np.mean(t1p)
            f['top1p_std'] = np.std(t1p)
            f['top5p_mean'] = np.mean(t5p)
            f['top5p_std'] = np.std(t5p)

            # Burstiness: how much does log prob vary from token to token
            if len(lp) > 1:
                f['lp_diff_mean'] = np.mean(np.abs(np.diff(lp)))
                f['lp_diff_std'] = np.std(np.diff(lp))
            else:
                f['lp_diff_mean'] = 0
                f['lp_diff_std'] = 0

            # Sequence length
            f['seq_length'] = T

            all_features.append(f)

        except Exception as e:
            all_features.append(_empty_features())

    return pd.DataFrame(all_features)


def _empty_features():
    keys = ['lp_mean', 'lp_std', 'lp_min', 'lp_max', 'lp_median', 'lp_q10', 'lp_q90',
            'lp_skew', 'lp_kurtosis', 'ent_mean', 'ent_std', 'ent_min', 'ent_max',
            'ent_median', 'ent_skew', 'rank_mean', 'rank_std', 'rank_median', 'rank_q90',
            'rank_top1_frac', 'rank_top5_frac', 'rank_top10_frac', 'rank_top100_frac',
            'top1p_mean', 'top1p_std', 'top5p_mean', 'top5p_std',
            'lp_diff_mean', 'lp_diff_std', 'seq_length']
    return {k: 0.0 for k in keys}


def load_model():
    device = 'cuda'
    print("Loading Mistral-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer, device


def run_experiment(name, train_texts, train_labels, test_texts, test_labels, model, tokenizer, device,
                   test_df=None, model_col='model'):
    from xgboost import XGBClassifier

    print(f"\n{'='*60}")
    print(f"Token-level features: {name}")
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
    print(f"{'='*60}")

    cache_train = RESULT_DIR / f"token_features_{name}_train.csv"
    cache_test  = RESULT_DIR / f"token_features_{name}_test.csv"

    if cache_train.exists() and cache_test.exists():
        print("  Loading cached token features...")
        X_train = pd.read_csv(cache_train)
        X_test = pd.read_csv(cache_test)
    else:
        X_train = extract_token_prob_features(train_texts, model, tokenizer, device, label=f"{name}_train")
        X_test = extract_token_prob_features(test_texts, model, tokenizer, device, label=f"{name}_test")
        X_train.to_csv(cache_train, index=False)
        X_test.to_csv(cache_test, index=False)

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 1. Token features only
    print(f"\n  --- XGBoost with token features only ({X_train.shape[1]} features) ---")
    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist', device='cuda')
    clf.fit(X_train, y_train, verbose=False)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Token-only:  AUC={auc:.4f}, Acc={acc:.4f}")

    # Per-model if available
    if test_df is not None and model_col in test_df.columns:
        print(f"    Per-model:")
        for m in sorted(test_df[model_col].dropna().unique()):
            mask = (test_df[model_col] == m).values[:len(y_test)]
            if mask.sum() < 10:
                continue
            m_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"      {m:20s}: Acc={m_acc:.4f}, n={mask.sum()}")

    # Feature importance
    importances = clf.feature_importances_
    feat_names = X_train.columns
    top_idx = np.argsort(importances)[::-1][:10]
    print(f"\n  Top 10 features:")
    for idx in top_idx:
        print(f"    {feat_names[idx]:25s}: {importances[idx]:.4f}")

    return auc, acc, X_train, X_test


def plot_comparison(results):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = list(results.keys())
    methods = ['XGBoost (90 feat)', 'Token-only (30 feat)', 'Combined (120 feat)',
               'Mistral CE', 'Fast-DetectGPT']
    x = np.arange(len(datasets))
    width = 0.15

    for i, method in enumerate(methods):
        vals = [results[ds].get(method, 0) for ds in datasets]
        bars = ax.bar(x + i * width, vals, width, label=method, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('AUC')
    ax.set_title('Token-level Probability Features vs Previous Methods')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'token_features_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIG_DIR / 'token_features_comparison.png'}")


if __name__ == "__main__":
    model, tokenizer, device = load_model()

    results = {}

    # ── HC3 ──
    train_df, test_df = get_splits()
    # Subsample for speed
    n = 5000
    train_sub = pd.concat([g.sample(min(len(g), n), random_state=42)
                           for _, g in train_df.groupby('label')]).reset_index(drop=True)
    test_sub = pd.concat([g.sample(min(len(g), n//2), random_state=42)
                          for _, g in test_df.groupby('label')]).reset_index(drop=True)

    auc_tok, acc_tok, X_tr_tok, X_te_tok = run_experiment(
        "hc3", train_sub['text'].tolist(), train_sub['label'].tolist(),
        test_sub['text'].tolist(), test_sub['label'].tolist(),
        model, tokenizer, device)

    # Combined: token + existing 90 features
    from run_semeval import extract_features_batch
    from multiprocessing import Pool, cpu_count

    def _extract_worker(args):
        return extract_features_batch(args[0], args[1])

    def parallel_extract(texts, label):
        n_workers = min(cpu_count(), 8)
        chunk_size = len(texts) // n_workers + 1
        chunks = [(texts[i:i+chunk_size], f"{label}_{j}")
                  for j, i in enumerate(range(0, len(texts), chunk_size))]
        with Pool(n_workers) as pool:
            res = pool.map(_extract_worker, chunks)
        return pd.concat(res, ignore_index=True)

    print("\n  Extracting CPU features for combined model...")
    X_tr_cpu = parallel_extract(train_sub['text'].tolist(), "hc3_tr")
    X_te_cpu = parallel_extract(test_sub['text'].tolist(), "hc3_te")

    X_tr_combined = pd.concat([X_tr_tok.reset_index(drop=True), X_tr_cpu.reset_index(drop=True)], axis=1)
    X_te_combined = pd.concat([X_te_tok.reset_index(drop=True), X_te_cpu.reset_index(drop=True)], axis=1)
    X_tr_combined = X_tr_combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_te_combined = X_te_combined.replace([np.inf, -np.inf], np.nan).fillna(0)

    from xgboost import XGBClassifier
    print(f"\n  --- XGBoost Combined ({X_tr_combined.shape[1]} features) on HC3 ---")
    clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, tree_method='hist', device='cuda')
    clf.fit(X_tr_combined, train_sub['label'].values, verbose=False)
    y_prob = clf.predict_proba(X_te_combined)[:, 1]
    auc_comb = roc_auc_score(test_sub['label'].values, y_prob)
    acc_comb = accuracy_score(test_sub['label'].values, clf.predict(X_te_combined))
    print(f"  Combined:    AUC={auc_comb:.4f}, Acc={acc_comb:.4f}")

    results['HC3'] = {
        'XGBoost (90 feat)': 0.9999, 'Token-only (30 feat)': auc_tok,
        'Combined (120 feat)': auc_comb, 'Mistral CE': 0.9933, 'Fast-DetectGPT': 0.9292
    }

    del train_df, test_df, train_sub, test_sub
    torch.cuda.empty_cache()

    # ── SemEval ──
    DATA_SE = ROOT / "data" / "external" / "semeval2024_task8" / "subtaskA_monolingual"
    se_train = pd.read_parquet(DATA_SE / "train-00000-of-00001.parquet")
    se_test = pd.read_parquet(DATA_SE / "test-00000-of-00001.parquet")

    # Subsample
    se_train_sub = pd.concat([g.sample(min(len(g), 5000), random_state=42)
                              for _, g in se_train.groupby('label')]).reset_index(drop=True)
    se_test_sub = pd.concat([g.sample(min(len(g), 2500), random_state=42)
                             for _, g in se_test.groupby('label')]).reset_index(drop=True)

    auc_tok_se, acc_tok_se, X_tr_tok_se, X_te_tok_se = run_experiment(
        "semeval", se_train_sub['text'].tolist(), se_train_sub['label'].tolist(),
        se_test_sub['text'].tolist(), se_test_sub['label'].tolist(),
        model, tokenizer, device, test_df=se_test_sub, model_col='model')

    # Combined for SemEval
    print("\n  Extracting CPU features for SemEval combined...")
    X_tr_cpu_se = parallel_extract(se_train_sub['text'].tolist(), "se_tr")
    X_te_cpu_se = parallel_extract(se_test_sub['text'].tolist(), "se_te")

    X_tr_comb_se = pd.concat([X_tr_tok_se.reset_index(drop=True), X_tr_cpu_se.reset_index(drop=True)], axis=1)
    X_te_comb_se = pd.concat([X_te_tok_se.reset_index(drop=True), X_te_cpu_se.reset_index(drop=True)], axis=1)
    X_tr_comb_se = X_tr_comb_se.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_te_comb_se = X_te_comb_se.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\n  --- XGBoost Combined ({X_tr_comb_se.shape[1]} features) on SemEval ---")
    clf2 = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                         random_state=42, tree_method='hist', device='cuda')
    clf2.fit(X_tr_comb_se, se_train_sub['label'].values, verbose=False)
    y_prob_se = clf2.predict_proba(X_te_comb_se)[:, 1]
    y_pred_se = clf2.predict(X_te_comb_se)
    auc_comb_se = roc_auc_score(se_test_sub['label'].values, y_prob_se)
    acc_comb_se = accuracy_score(se_test_sub['label'].values, y_pred_se)
    print(f"  Combined:    AUC={auc_comb_se:.4f}, Acc={acc_comb_se:.4f}")

    # Per-model for combined
    print(f"    Per-model (combined):")
    for m in sorted(se_test_sub['model'].dropna().unique()):
        mask = (se_test_sub['model'] == m).values
        if mask.sum() < 10:
            continue
        m_acc = accuracy_score(se_test_sub['label'].values[mask], y_pred_se[mask])
        print(f"      {m:20s}: Acc={m_acc:.4f}, n={mask.sum()}")

    results['SemEval'] = {
        'XGBoost (90 feat)': 0.6872, 'Token-only (30 feat)': auc_tok_se,
        'Combined (120 feat)': auc_comb_se, 'Mistral CE': 0.9729, 'Fast-DetectGPT': 0.8068
    }

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY: Token-level Probability Features")
    print("="*60)
    for ds, res in results.items():
        print(f"\n  {ds}:")
        for method, auc in sorted(res.items(), key=lambda x: -x[1]):
            marker = " <<<" if 'Token' in method or 'Combined' in method else ""
            print(f"    {method:30s}: AUC={auc:.4f}{marker}")

    plot_comparison(results)
    print("\nDone!")
