"""
Shared train/test split for all methods. Ensures consistent evaluation.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/hc3_flat.csv")
SEED = 42

def get_splits(max_rows: int | None = None):
    """Return (train_df, test_df) with 80/20 stratified split."""
    df = pd.read_csv(DATA_PATH)
    if max_rows:
        per_class = max(1, max_rows // df["label"].nunique())
        df = pd.concat(
            [part.sample(min(len(part), per_class), random_state=SEED)
             for _, part in df.groupby("label")],
            ignore_index=True,
        ).sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    train_df, test_df = get_splits()
    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    print(f"Train label dist:\n{train_df['label_name'].value_counts().to_string()}")
    print(f"Test label dist:\n{test_df['label_name'].value_counts().to_string()}")
