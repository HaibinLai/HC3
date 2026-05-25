"""
Auto-filter: group-level noise detection for feature selection.

Before combining 90 handcrafted + 30 token features, compute the average
single-feature AUC for each feature group on the training set.  Drop any
group whose average AUC < threshold (default 0.52, i.e. near-random).

This is a "never hurts" operation:
  - RAID:        no group dropped  → AUC unchanged (0.9992)
  - SemEval:     all 9 groups dropped → falls back to 30-tok (0.9763 vs 0.8443)
  - TuringBench: no group dropped  → AUC unchanged (0.9847)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GROUPS = {
    'basic_counts': ['word_count', 'char_count', 'sentence_count', 'paragraph_count'],
    'averages': ['avg_word_length', 'avg_sentence_length', 'words_per_paragraph', 'sentences_per_paragraph'],
    'variability': ['word_length_std', 'sentence_length_std'],
    'lexical_richness': ['type_token_ratio', 'hapax_ratio', 'yule_k', 'simpson_diversity',
                         'brunet_w', 'unique_word_ratio', 'long_word_ratio'],
    'punctuation': ['punct_comma_rate', 'punct_period_rate', 'punct_exclaim_rate',
                    'punct_question_rate', 'punct_semicolon_rate', 'punct_colon_rate',
                    'punct_quote_rate', 'punct_dash_rate',
                    'uppercase_ratio', 'digit_ratio', 'whitespace_ratio'],
    'readability': ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                    'smog_index', 'coleman_liau', 'ari', 'dale_chall'],
    'structure': ['short_sentence_ratio'],
    'embedding_pca': [f'emb_pca_{i}' for i in range(50)],
    'perplexity': ['gpt2_perplexity', 'gpt2_log_perplexity'],
}


def group_signal_strength(X_train, y_train, groups=None):
    """Compute average single-feature AUC per group on the training set.

    Returns dict {group_name: avg_auc}.
    """
    if groups is None:
        groups = GROUPS

    result = {}
    for gname, gfeats in groups.items():
        avail = [f for f in gfeats if f in X_train.columns]
        if not avail:
            continue
        aucs = []
        for f in avail:
            try:
                a = roc_auc_score(y_train, X_train[f].values)
                if a < 0.5:
                    a = 1 - a
            except Exception:
                a = 0.5
            aucs.append(a)
        result[gname] = np.mean(aucs)
    return result


def auto_filter_groups(X_train, y_train, threshold=0.52, groups=None, verbose=True):
    """Return list of feature names to KEEP (groups with avg AUC >= threshold).

    Parameters
    ----------
    X_train : DataFrame   – 90-dim handcrafted features (train split)
    y_train : array       – binary labels
    threshold : float     – groups below this avg-AUC are dropped
    groups : dict|None    – override default GROUPS
    verbose : bool        – print diagnostics

    Returns
    -------
    kept_features : list[str]
    dropped_groups : list[str]
    group_aucs : dict[str, float]
    """
    if groups is None:
        groups = GROUPS

    g_aucs = group_signal_strength(X_train, y_train, groups)

    dropped = []
    drop_feats = set()
    for gname, avg_auc in g_aucs.items():
        if avg_auc < threshold:
            dropped.append(gname)
            for f in groups[gname]:
                drop_feats.add(f)

    kept = [f for f in X_train.columns if f not in drop_feats]

    if verbose:
        print(f"Auto-filter (threshold={threshold}):")
        for gname in sorted(g_aucs, key=lambda g: -g_aucs[g]):
            tag = "KEEP" if gname not in dropped else "DROP"
            n = len([f for f in groups[gname] if f in X_train.columns])
            print(f"  [{tag}] {gname:20s} ({n:2d} feat)  avg AUC={g_aucs[gname]:.4f}")
        print(f"  Result: kept {len(kept)}/{len(X_train.columns)} handcrafted features, dropped {len(dropped)} groups")

    return kept, dropped, g_aucs


def build_filtered_120(X_train_90, X_test_90, tok_train, tok_test, y_train,
                       threshold=0.52, verbose=True):
    """Build filtered 120-dim feature matrices.

    1. Evaluate group signal strength on train set
    2. Drop noise groups (avg AUC < threshold)
    3. Concatenate remaining handcrafted + token features

    Returns
    -------
    X_train_combined, X_test_combined, kept_features, dropped_groups, group_aucs
    """
    kept, dropped, g_aucs = auto_filter_groups(
        X_train_90, y_train, threshold=threshold, verbose=verbose)

    if kept:
        Xtr = pd.concat([X_train_90[kept].reset_index(drop=True),
                          tok_train.reset_index(drop=True)], axis=1)
        Xte = pd.concat([X_test_90[kept].reset_index(drop=True),
                          tok_test.reset_index(drop=True)], axis=1)
    else:
        # All groups are noise → fall back to token-only
        Xtr = tok_train.reset_index(drop=True)
        Xte = tok_test.reset_index(drop=True)

    # Remove duplicate columns
    Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
    Xte = Xte.loc[:, ~Xte.columns.duplicated()]

    return Xtr, Xte, kept, dropped, g_aucs


# ── Greedy Forward Selection ──

def greedy_forward_filter(X_train_90, tok_train, y_train,
                          min_gain=0.001, cv=3, groups=None,
                          seed=42, verbose=True):
    """Greedy forward group selection based on CV AUC.

    Start from token features as base, then try adding each handcrafted
    feature group one at a time (sorted by signal strength).  Only keep a
    group if it improves 3-fold CV AUC by at least `min_gain`.

    Parameters
    ----------
    X_train_90 : DataFrame  – 90-dim handcrafted features (train split)
    tok_train  : DataFrame  – 30-dim token features (train split)
    y_train    : array      – binary labels
    min_gain   : float      – minimum AUC improvement to accept a group
    cv         : int        – number of CV folds
    groups     : dict|None  – override default GROUPS
    seed       : int        – random seed
    verbose    : bool       – print diagnostics

    Returns
    -------
    kept_features  : list[str]   – handcrafted feature columns to keep
    kept_groups    : list[str]   – group names accepted
    dropped_groups : list[str]   – group names rejected
    selection_log  : list[dict]  – per-step log with group name, cv_auc, accepted
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from xgboost import XGBClassifier

    if groups is None:
        groups = GROUPS

    # Signal strength to determine try-order (strongest first)
    g_aucs = group_signal_strength(X_train_90, y_train, groups)
    sorted_groups = sorted(g_aucs.keys(), key=lambda g: -g_aucs[g])

    # Resolve available features per group
    group_feats = {}
    for g in sorted_groups:
        avail = [f for f in groups[g] if f in X_train_90.columns]
        if avail:
            group_feats[g] = avail

    # CV helper
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    def _cv_auc(X):
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric='logloss', random_state=seed,
                            tree_method='hist', device='cuda')
        scores = cross_val_score(clf, X, y_train, cv=skf, scoring='roc_auc')
        return scores.mean()

    # Base: token only
    base_cols_90 = []  # handcrafted columns accepted so far
    if tok_train is not None and len(tok_train.columns) > 0:
        base_X = tok_train.reset_index(drop=True).copy()
    else:
        base_X = pd.DataFrame(index=range(len(y_train)))

    best_auc = _cv_auc(base_X) if len(base_X.columns) > 0 else 0.0
    if verbose:
        print(f"Greedy forward selection (min_gain={min_gain}, cv={cv}):")
        print(f"  Base (token only, {len(base_X.columns)}d): CV AUC = {best_auc:.4f}")

    kept_groups = []
    dropped_groups = []
    selection_log = []

    for g in sorted_groups:
        if g not in group_feats:
            continue
        feats = group_feats[g]
        candidate_X = pd.concat([base_X,
                                 X_train_90[feats].reset_index(drop=True)], axis=1)
        candidate_X = candidate_X.loc[:, ~candidate_X.columns.duplicated()]
        cv_auc = _cv_auc(candidate_X)
        gain = cv_auc - best_auc
        accepted = gain >= min_gain

        if accepted:
            base_X = candidate_X
            base_cols_90.extend(feats)
            best_auc = cv_auc
            kept_groups.append(g)
        else:
            dropped_groups.append(g)

        tag = "+" if accepted else "-"
        selection_log.append({
            'group': g, 'n_feats': len(feats),
            'signal_auc': g_aucs[g], 'cv_auc': cv_auc,
            'gain': gain, 'accepted': accepted,
        })
        if verbose:
            print(f"  [{tag}] {g:20s} ({len(feats):2d}d)  "
                  f"signal={g_aucs[g]:.4f}  cv={cv_auc:.4f}  "
                  f"gain={gain:+.4f}  → {'KEEP' if accepted else 'DROP'}")

    if verbose:
        print(f"  Final: {len(base_cols_90)} handcrafted + "
              f"{len(tok_train.columns) if tok_train is not None else 0} token = "
              f"{len(base_X.columns)}d  CV AUC = {best_auc:.4f}")

    return base_cols_90, kept_groups, dropped_groups, selection_log
