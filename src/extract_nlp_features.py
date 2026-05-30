"""
Extract NLP features: POS ratios, transition words, sentiment, clause structure.
Adds ~12 new features to RAID dataset.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import nltk
from nltk import pos_tag, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
EXT_DIR = ROOT / "data" / "external" / "raid"

# ── Word lists ──
TRANSITION_PHRASES = [
    "first", "firstly", "second", "secondly", "third", "finally",
    "in conclusion", "to conclude", "overall", "therefore",
    "however", "moreover", "furthermore", "on the other hand",
    "as a result", "for example", "in addition", "in summary",
    "to sum up", "consequently", "nevertheless", "nonetheless",
]

DISCOURSE_MARKERS = {"firstly", "secondly", "thirdly", "finally",
                     "in conclusion", "to conclude", "overall", "in summary", "to sum up"}

HEDGE_WORDS = {"however", "nevertheless", "nonetheless", "on the other hand",
               "conversely", "although", "though", "yet"}

SUBORDINATE_CONJ = {"because", "although", "though", "while", "if", "when",
                    "since", "unless", "whereas", "that", "which", "who", "whom"}

SENT_RE = re.compile(r'[.!?]+')
sia = SentimentIntensityAnalyzer()


def extract_nlp_features(text):
    """Extract 12 NLP features from a single text."""
    f = {}
    words = text.split()
    word_count = max(len(words), 1)
    word_lower = [w.lower().strip('.,!?;:"\'-()[]') for w in words]

    # ── POS ratios (4d) ──
    try:
        # Use first 200 words for speed
        sample_words = words[:200]
        tagged = pos_tag(sample_words)
        tag_counts = Counter(t for _, t in tagged)
        n_tagged = max(len(tagged), 1)
        f['noun_ratio'] = sum(tag_counts.get(t, 0) for t in ['NN','NNS','NNP','NNPS']) / n_tagged
        f['verb_ratio'] = sum(tag_counts.get(t, 0) for t in ['VB','VBD','VBG','VBN','VBP','VBZ']) / n_tagged
        f['adj_ratio'] = sum(tag_counts.get(t, 0) for t in ['JJ','JJR','JJS']) / n_tagged
        f['adv_ratio'] = sum(tag_counts.get(t, 0) for t in ['RB','RBR','RBS']) / n_tagged
    except:
        f['noun_ratio'] = f['verb_ratio'] = f['adj_ratio'] = f['adv_ratio'] = 0.0

    # ── Transition word features (3d) ──
    text_lower = text.lower()
    transition_count = sum(text_lower.count(phrase) for phrase in TRANSITION_PHRASES)
    f['transition_rate'] = transition_count / word_count

    discourse_count = sum(text_lower.count(phrase) for phrase in DISCOURSE_MARKERS)
    f['discourse_marker_rate'] = discourse_count / word_count

    hedge_count = sum(text_lower.count(phrase) for phrase in HEDGE_WORDS)
    f['hedge_rate'] = hedge_count / word_count

    # ── Sentiment features (3d) ──
    sents = [s.strip() for s in SENT_RE.split(text) if len(s.strip()) > 5]
    if sents:
        compounds = [sia.polarity_scores(s)['compound'] for s in sents[:50]]  # cap at 50 sentences
        f['sentiment_mean'] = np.mean(compounds)
        f['sentiment_std'] = np.std(compounds) if len(compounds) > 1 else 0.0
        f['sentiment_range'] = max(compounds) - min(compounds) if len(compounds) > 1 else 0.0
    else:
        f['sentiment_mean'] = f['sentiment_std'] = f['sentiment_range'] = 0.0

    # ── Clause structure (2d) ──
    sent_count = max(len(sents), 1)
    clause_markers = text.count(',') + text.count(';') + text.count(':')
    f['clause_depth_approx'] = clause_markers / sent_count

    sub_count = sum(1 for w in word_lower if w in SUBORDINATE_CONJ)
    f['subordinate_ratio'] = sub_count / word_count

    return f


def main():
    print("=" * 60)
    print("NLP Feature Extraction (POS, Transition, Sentiment, Clause)")
    print("=" * 60)

    # Load RAID data with same split as run_raid_analysis
    from run_raid_analysis import load_features_and_split
    train_df, test_df, _, _ = load_features_and_split()

    print(f"\n  Train: {len(train_df)}, Test: {len(test_df)}")

    for split_name, df in [('train', train_df), ('test', test_df)]:
        print(f"\n  Extracting {split_name} NLP features...")
        texts = df['generation'].values
        features_list = []
        total = len(texts)
        for idx, text in enumerate(texts):
            if idx % 2000 == 0:
                print(f"    [{split_name}] {idx}/{total}")
            features_list.append(extract_nlp_features(str(text)))

        nlp_df = pd.DataFrame(features_list)
        out_path = DATA_DIR / f'raid_nlp_features_{split_name}.csv'
        nlp_df.to_csv(out_path, index=False)
        print(f"    Saved: {out_path.name} ({nlp_df.shape[1]} features, {len(nlp_df)} rows)")

        # Print feature stats
        print(f"    Feature means:")
        for col in nlp_df.columns:
            print(f"      {col:25s}  mean={nlp_df[col].mean():.4f}  std={nlp_df[col].std():.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
