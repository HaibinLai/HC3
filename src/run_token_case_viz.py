"""
Case-level visualization for token probability features.
Generates:
  1. Token-level probability heatmaps (word-by-word coloring)
  2. t-SNE 2D scatter of 30-dim token features
  3. Misclassification case analysis

Requires GPU for token-level heatmaps (Mistral-7B inference on ~20 samples).
t-SNE and misclassification analysis use cached CSV features.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import textwrap

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / 'src'))


# ────────────────────────────────────────────────────────
# 1. Token-level probability heatmap
# ────────────────────────────────────────────────────────
def get_token_details(text, model, tokenizer, device, max_length=512):
    """Return per-token log prob, entropy, rank, and decoded token strings."""
    enc = tokenizer(text, return_tensors='pt', truncation=True,
                    max_length=max_length).to(device)
    input_ids = enc['input_ids']

    with torch.no_grad():
        logits = model(**enc).logits

    pred_logits = logits[0, :-1, :]
    target_ids = input_ids[0, 1:]
    T = pred_logits.shape[0]

    log_probs = torch.log_softmax(pred_logits, dim=-1)
    token_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).cpu().numpy()

    probs = torch.softmax(pred_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()

    ranks = np.zeros(T)
    sorted_idx = torch.argsort(pred_logits, dim=-1, descending=True)
    for t in range(T):
        pos = (sorted_idx[t] == target_ids[t]).nonzero(as_tuple=True)[0]
        ranks[t] = pos[0].item() if len(pos) > 0 else 50000

    # Decode individual tokens
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0, 1:].cpu().tolist()]

    return tokens, token_lp, entropy, ranks


def draw_token_heatmap(ax, tokens, values, title, cmap='RdYlGn', vmin=None, vmax=None,
                       max_tokens=80):
    """Draw tokens as colored boxes on a matplotlib axis."""
    tokens = tokens[:max_tokens]
    values = values[:max_tokens]

    if vmin is None:
        vmin = np.percentile(values, 5)
    if vmax is None:
        vmax = np.percentile(values, 95)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.cm.get_cmap(cmap)

    x, y = 0.02, 0.85
    line_height = 0.18
    max_x = 0.98

    for i, (tok, val) in enumerate(zip(tokens, values)):
        color = colormap(norm(val))
        display = tok.replace('\n', '\\n')
        if len(display) > 12:
            display = display[:10] + '..'

        text_width = len(display) * 0.012 + 0.015

        if x + text_width > max_x:
            x = 0.02
            y -= line_height
            if y < 0.02:
                break

        rect = FancyBboxPatch((x, y - 0.06), text_width, 0.12,
                               boxstyle="round,pad=0.01",
                               facecolor=color, edgecolor='#666666',
                               linewidth=0.5, transform=ax.transAxes)
        ax.add_patch(rect)
        text_color = 'white' if sum(color[:3]) < 1.5 else 'black'
        ax.text(x + text_width / 2, y, display, fontsize=6, fontfamily='monospace',
                ha='center', va='center', transform=ax.transAxes, color=text_color)
        x += text_width + 0.005

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)


def plot_token_heatmaps(model, tokenizer, device):
    """Generate heatmaps for selected Human/AI cases from HC3."""
    from data_splits import get_splits
    _, test_df = get_splits()

    # Pick diverse cases
    human_samples = test_df[test_df['label'] == 0].sample(3, random_state=42)
    ai_samples = test_df[test_df['label'] == 1].sample(3, random_state=42)
    samples = pd.concat([human_samples, ai_samples]).reset_index(drop=True)

    fig, axes = plt.subplots(6, 2, figsize=(22, 24))

    for i, (_, row) in enumerate(samples.iterrows()):
        text = row['text'][:1500]  # limit length
        label = 'HUMAN' if row['label'] == 0 else 'AI (ChatGPT)'
        source = row.get('source', '?')

        tokens, lp, ent, ranks = get_token_details(text, model, tokenizer, device)

        # Log probability heatmap (green=high prob, red=low prob)
        title_lp = f"[{label}] {source} — Log Probability (green=predictable, red=surprising)"
        draw_token_heatmap(axes[i, 0], tokens, lp, title_lp, cmap='RdYlGn')

        # Rank heatmap (green=rank 0, red=high rank)
        log_rank = np.log1p(ranks)
        title_rk = f"[{label}] {source} — Token Rank (green=top-1, red=rare)"
        draw_token_heatmap(axes[i, 1], tokens, -log_rank, title_rk, cmap='RdYlGn')

    # Add colorbars
    fig.subplots_adjust(bottom=0.04)
    cbar_ax1 = fig.add_axes([0.08, 0.01, 0.38, 0.012])
    cbar_ax2 = fig.add_axes([0.55, 0.01, 0.38, 0.012])
    sm1 = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(-12, 0))
    sm2 = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, -10))
    fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal', label='Log Probability')
    fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal', label='-Log(1+Rank)')

    plt.suptitle('Token-level Probability Heatmap: Human vs AI Text\n'
                 'Each box = one token, colored by how "predictable" it is to Mistral-7B',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fname = FIG_DIR / 'token_heatmap_cases.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


# ────────────────────────────────────────────────────────
# 2. t-SNE scatter plot
# ────────────────────────────────────────────────────────
def plot_tsne():
    """t-SNE of 30-dim token features, colored by label, paneled by dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    dataset_names = ['hc3', 'semeval', 'turingbench', 'pile']
    titles = {
        'hc3': 'HC3 (AUC=0.9998)',
        'semeval': 'SemEval (AUC=0.9784)',
        'turingbench': 'TuringBench (AUC=0.4853)',
        'pile': 'Pile (AUC=0.9918)',
    }

    for ax, name in zip(axes.flat, dataset_names):
        X_test = pd.read_csv(DATA_DIR / f"token_features_{name}_test.csv")
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Reconstruct labels
        y_test = _get_test_labels(name, len(X_test))

        # Subsample for t-SNE speed
        n = min(3000, len(X_test))
        idx = np.random.RandomState(42).choice(len(X_test), n, replace=False)
        X_sub = X_test.iloc[idx].values
        y_sub = y_test[idx]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        emb = tsne.fit_transform(X_sub)

        for lab, label_name, color, marker in [(0, 'Human', '#2E7D32', 'o'),
                                                (1, 'AI', '#C62828', '^')]:
            mask = y_sub == lab
            ax.scatter(emb[mask, 0], emb[mask, 1], c=color, alpha=0.3, s=8,
                      marker=marker, label=label_name, edgecolors='none')

        ax.set_title(titles[name], fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, markerscale=3)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.suptitle('t-SNE of 30-dim Token Probability Features\n'
                 'Clear separation → good detection; overlap → failure',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fname = FIG_DIR / 'token_tsne.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


# ────────────────────────────────────────────────────────
# 3. Misclassification analysis
# ────────────────────────────────────────────────────────
def plot_misclassification():
    """Analyze confident misclassifications on HC3 and SemEval."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for row, name in enumerate(['hc3', 'semeval']):
        X_train = pd.read_csv(DATA_DIR / f"token_features_{name}_train.csv")
        X_test = pd.read_csv(DATA_DIR / f"token_features_{name}_test.csv")
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        y_train = _get_test_labels(name, len(X_train), split='train')
        y_test = _get_test_labels(name, len(X_test), split='test')

        clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                            random_state=42, tree_method='hist')
        clf.fit(X_train, y_train, verbose=False)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # Confidence distribution
        ax = axes[row, 0]
        correct = y_pred == y_test
        ax.hist(y_prob[correct & (y_test == 0)], bins=50, alpha=0.5, label='Correct Human',
                color='#2E7D32', density=True)
        ax.hist(y_prob[correct & (y_test == 1)], bins=50, alpha=0.5, label='Correct AI',
                color='#C62828', density=True)
        ax.hist(y_prob[~correct], bins=50, alpha=0.7, label='Misclassified',
                color='#FF9800', density=True, histtype='step', linewidth=2)
        ax.set_xlabel('P(AI)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name.upper()} — Prediction Confidence', fontweight='bold')
        ax.legend(fontsize=8)
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)

        # Feature comparison: correct vs misclassified
        ax = axes[row, 1]
        top_feats = ['rank_top100_frac', 'lp_mean', 'ent_mean', 'rank_top1_frac', 'top1p_mean']
        available = [f for f in top_feats if f in X_test.columns][:4]

        mis_idx = np.where(~correct)[0]
        cor_idx = np.where(correct)[0]

        if len(mis_idx) > 0:
            x_pos = np.arange(len(available))
            width = 0.35
            cor_means = [X_test.iloc[cor_idx][f].mean() for f in available]
            mis_means = [X_test.iloc[mis_idx][f].mean() for f in available]
            ax.bar(x_pos - width/2, cor_means, width, label='Correct', color='#4CAF50', alpha=0.7)
            ax.bar(x_pos + width/2, mis_means, width, label='Misclassified', color='#FF9800', alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(available, fontsize=8, rotation=15)
            ax.legend()
        ax.set_title(f'{name.upper()} — Feature Means: Correct vs Misclassified', fontweight='bold')

        # Error type breakdown
        ax = axes[row, 2]
        fn = ((y_pred == 0) & (y_test == 1)).sum()  # AI missed as human
        fp = ((y_pred == 1) & (y_test == 0)).sum()  # Human flagged as AI
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        tp = ((y_pred == 1) & (y_test == 1)).sum()

        labels_pie = [f'True Human\n(n={tn})', f'True AI\n(n={tp})',
                      f'AI→Human\n(n={fn})', f'Human→AI\n(n={fp})']
        sizes = [tn, tp, fn, fp]
        colors_pie = ['#4CAF50', '#F44336', '#FF9800', '#FFC107']
        explode = [0, 0, 0.1, 0.1]

        wedges, texts, autotexts = ax.pie(sizes, labels=labels_pie, colors=colors_pie,
                                           explode=explode, autopct='%1.1f%%',
                                           startangle=90, pctdistance=0.8)
        for t in texts:
            t.set_fontsize(8)
        for t in autotexts:
            t.set_fontsize(7)
        ax.set_title(f'{name.upper()} — Error Breakdown', fontweight='bold')

    plt.suptitle('Misclassification Analysis: Token Probability Features',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fname = FIG_DIR / 'token_misclassification.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


# ────────────────────────────────────────────────────────
# Helper: reconstruct labels
# ────────────────────────────────────────────────────────
def _get_test_labels(name, n_samples, split='test'):
    if name == 'hc3':
        from data_splits import get_splits
        train_df, test_df = get_splits()
        ns = 5000
        if split == 'test':
            sub = pd.concat([g.sample(min(len(g), ns//2), random_state=42)
                             for _, g in test_df.groupby('label')]).reset_index(drop=True)
        else:
            sub = pd.concat([g.sample(min(len(g), ns), random_state=42)
                             for _, g in train_df.groupby('label')]).reset_index(drop=True)
        return sub['label'].values[:n_samples]

    elif name == 'semeval':
        se_dir = ROOT / "data" / "external" / "semeval2024_task8" / "subtaskA_monolingual"
        if split == 'test':
            df = pd.read_parquet(se_dir / "test-00000-of-00001.parquet")
            sub = pd.concat([g.sample(min(len(g), 2500), random_state=42)
                             for _, g in df.groupby('label')]).reset_index(drop=True)
        else:
            df = pd.read_parquet(se_dir / "train-00000-of-00001.parquet")
            sub = pd.concat([g.sample(min(len(g), 5000), random_state=42)
                             for _, g in df.groupby('label')]).reset_index(drop=True)
        return sub['label'].values[:n_samples]

    elif name == 'turingbench':
        TB_DIR = ROOT / "data" / "external" / "turingbench" / "extracted" / "TuringBench"
        dfs = []
        for subdir in sorted(TB_DIR.iterdir()):
            if subdir.name.startswith('.') or subdir.name == '__MACOSX':
                continue
            for sp in ['train', 'test']:
                f = subdir / f'{sp}.csv'
                if f.exists():
                    d = pd.read_csv(f); d['split'] = sp; d['model'] = subdir.name; dfs.append(d)
        tb = pd.concat(dfs, ignore_index=True).rename(columns={'Generation': 'text'})
        tb['label'] = (tb['model'] != 'AA').astype(int)
        tb = tb.dropna(subset=['text'])
        tb = tb[tb['text'].str.len() > 10].reset_index(drop=True)
        sp_df = tb[tb['split'] == ('test' if split == 'test' else 'train')].reset_index(drop=True)
        n_per = 5000 if split == 'test' else 10000
        sub = pd.concat([g.sample(min(len(g), n_per // 2), random_state=42)
                         for _, g in sp_df.groupby('label')]).reset_index(drop=True)
        return sub['label'].values[:n_samples]

    elif name == 'pile':
        from sklearn.model_selection import train_test_split
        pile_dir = ROOT / "data" / "external" / "ai_text_detection_pile" / "data"
        dfs = [pd.read_parquet(f, columns=['text', 'source']) for f in sorted(pile_dir.glob("*.parquet"))]
        pile = pd.concat(dfs, ignore_index=True)
        pile['label'] = (pile['source'] == 'ai').astype(int)
        pile = pile.dropna(subset=['text'])
        pile = pile[pile['text'].str.len() > 10].reset_index(drop=True)
        sub = pd.concat([g.sample(min(len(g), 10000), random_state=42)
                         for _, g in pile.groupby('label')]).reset_index(drop=True)
        ptr, pte = train_test_split(sub, test_size=0.33, stratify=sub['label'], random_state=42)
        df = pte if split == 'test' else ptr
        return df['label'].values[:n_samples]


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("Token Feature Case-level Visualization")
    print("="*60)

    # 1. Token heatmaps (requires GPU)
    print("\n--- 1. Token Probability Heatmaps ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading Mistral-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                                                  torch_dtype=torch.float16, device_map='auto',
                                                  trust_remote_code=True)
    device = 'cuda'
    plot_token_heatmaps(model, tokenizer, device)

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # 2. t-SNE (CPU)
    print("\n--- 2. t-SNE Scatter ---")
    plot_tsne()

    # 3. Misclassification (CPU)
    print("\n--- 3. Misclassification Analysis ---")
    plot_misclassification()

    print("\n" + "="*60)
    print("All case-level visualizations generated!")
    print("="*60)
