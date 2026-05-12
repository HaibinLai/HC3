from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_hc3(raw_path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    with raw_path.open(encoding="utf-8") as f:
        for row_id, line in enumerate(f):
            item = json.loads(line)
            question = item.get("question") or ""
            source = item.get("source") or "unknown"

            for answer_id, text in enumerate(item.get("human_answers") or []):
                if text and text.strip():
                    records.append(
                        {
                            "row_id": row_id,
                            "answer_id": answer_id,
                            "source": source,
                            "question": question,
                            "text": text.strip(),
                            "label": 0,
                            "label_name": "human",
                        }
                    )

            for answer_id, text in enumerate(item.get("chatgpt_answers") or []):
                if text and text.strip():
                    records.append(
                        {
                            "row_id": row_id,
                            "answer_id": answer_id,
                            "source": source,
                            "question": question,
                            "text": text.strip(),
                            "label": 1,
                            "label_name": "chatgpt",
                        }
                    )

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten HC3 JSONL into CSV.")
    parser.add_argument("--raw", default="data/raw/hc3_all.jsonl")
    parser.add_argument("--out", default="data/processed/hc3_flat.csv")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = flatten_hc3(raw_path)
    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"rows: {len(df):,}")
    print(df["label_name"].value_counts().to_string())
    print("\nsource distribution:")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
