# AI Generated Text Detection Project

方向一：AI 生成文本的特征挖掘与检测。

This project starts with the HC3 dataset and builds an interpretable CPU-friendly
baseline for human-vs-ChatGPT text classification.

## Current Plan

1. Convert HC3 JSONL into a flat binary classification table.
2. Extract interpretable linguistic and statistical features.
3. Train baseline classifiers with numeric features and TF-IDF text features.
4. Use feature importance and SHAP-style attribution for the report.

## First Baseline Result

Using full HC3:

- Rows: `85,431`
- ROC AUC: `0.9955`
- Accuracy: `0.97`
- Human F1: `0.98`
- ChatGPT F1: `0.96`

Using a balanced 30,000-row HC3 sample:

- ROC AUC: `0.9859`
- Accuracy: `0.95`
- Human F1: `0.95`
- ChatGPT F1: `0.95`

Early feature signals:

- ChatGPT answers are longer on average.
- ChatGPT answers have longer sentences and higher Flesch-Kincaid grade.
- ChatGPT answers use more transition phrases per 100 words.
- Human answers have higher type-token ratio and higher Flesch reading ease.

## Files

- `data/raw/hc3_all.jsonl`: raw HC3 download.
- `data/processed/hc3_flat.csv`: flattened text classification dataset.
- `data/processed/hc3_features.csv`: numeric feature table.
- `src/prepare_hc3.py`: HC3 preprocessing.
- `src/run_baseline.py`: feature extraction and baseline modeling.
- `figures/`: generated charts.

## Run

```bash
. .venv/bin/activate
python src/prepare_hc3.py
python src/run_baseline.py --max-rows 30000
```

Run full HC3:

```bash
python src/run_baseline.py
```

The full numeric feature table is cached at `data/processed/hc3_features.csv`.
Use `--recompute-features` if the feature extractor changes.
