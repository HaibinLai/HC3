# Direction 1 Budget and Plan

## Topic

可解释的 AI 生成文本检测：基于语言统计特征与归因分析。

## Dataset Budget

Primary dataset: HC3 from Hugging Face.

- Raw file: `data/raw/hc3_all.jsonl`
- Raw file size: about 70 MB by file size, about 80 MB on disk
- Raw question rows: 24,322
- Flattened text rows: 85,431
- Human answers: 58,546
- ChatGPT answers: 26,885

Current local storage:

- `.venv`: about 785 MB
- `data`: about 192 MB
- `figures`: about 156 KB
- Total current project footprint excluding the PDF: about 1 GB

If DAIGT V2 is added later, expect about another 100 MB download plus processed
CSV/parquet files. Reserving 2-3 GB for the whole project is comfortable.

## Compute Budget

Current baseline uses CPU only.

- No GPU required.
- No paid API required.
- Balanced 30,000-row run finishes in minutes on a laptop.
- Full HC3 run is laptop-friendly and completed locally, but SHAP should be
  computed on a sampled subset to avoid unnecessary runtime.
- The full numeric feature table is cached, so rerunning the model does not need
  to recompute readability and lexical features unless the extractor changes.

## First Baseline

Model: Logistic Regression with standardized numeric linguistic features plus
TF-IDF unigrams/bigrams.

Full HC3 result:

- Rows used: 85,431
- ROC AUC: 0.9955
- Accuracy: 0.97
- Human precision/recall/F1: 0.99 / 0.97 / 0.98
- ChatGPT precision/recall/F1: 0.94 / 0.98 / 0.96

Balanced 30,000-row result:

- ROC AUC: 0.9859
- Accuracy: 0.95
- Human precision/recall/F1: 0.96 / 0.94 / 0.95
- ChatGPT precision/recall/F1: 0.94 / 0.96 / 0.95

## Early Findings

Compared with human answers, ChatGPT answers in the sampled HC3 data tend to have:

- higher character and word counts
- longer average sentence length
- higher Flesch-Kincaid grade
- more transition phrases per 100 words
- lower type-token ratio
- lower Flesch reading ease

These signals match the course requirement well because they are interpretable
and can be explained with feature importance or SHAP.

## Next Steps

1. Run the full HC3 experiment.
2. Add a numeric-feature-only model for interpretability comparison.
3. Add XGBoost on numeric features and compute SHAP values.
4. Create the final Jupyter Notebook with EDA, model results, and figures.
5. Start the ACM/KDD-style report once the full-run results are stable.
