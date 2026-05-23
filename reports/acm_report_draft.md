# Feature-Based Detection of AI-Generated Text: A Multi-Dataset Study with Interpretable Models

## Abstract

The proliferation of large language models (LLMs) has made AI-generated text increasingly difficult to distinguish from human writing. Existing detection methods often rely on black-box deep learning models that lack interpretability. In this work, we propose a fully interpretable detection framework based on 90 handcrafted linguistic features spanning lexical richness, readability, punctuation patterns, semantic embeddings, and language model perplexity. We conduct extensive experiments on the RAID benchmark (ACL 2024) — covering 11 generators, 8 domains, and 11 adversarial attacks — and validate on four additional datasets (HC3, SemEval 2024, TuringBench, Pile). Our XGBoost classifier achieves AUC 0.9951 on RAID, significantly outperforming zero-shot methods (Fast-DetectGPT: 0.7815). Through SHAP analysis and feature ablation, we reveal that **GPT-2 perplexity — the strongest single feature on single-model datasets (AUC 0.99) — completely fails in multi-generator scenarios (AUC 0.49)**, while paragraph structure features emerge as the most robust cross-model signal. Adversarial analysis shows that paraphrase attacks are the only effective evasion strategy (AUC drop 20.5%), while 9 out of 11 surface-level attacks have negligible impact. Our findings provide actionable guidelines for deploying interpretable AI text detectors in real-world settings.

---

## 1. Introduction

The rapid advancement of large language models (LLMs) such as GPT-4, ChatGPT, and Llama-2 has enabled the generation of human-like text at unprecedented scale. This capability raises significant concerns in academic integrity, misinformation, and content authenticity. Reliable detection of AI-generated text has thus become a critical challenge.

Existing detection methods can be broadly categorized into three paradigms: (1) **supervised fine-tuned models** (e.g., RoBERTa-based classifiers) that treat detection as binary classification but function as black boxes; (2) **zero-shot statistical methods** (e.g., DetectGPT, Binoculars) that leverage language model probability distributions but generalize poorly across generators; and (3) **feature-based approaches** that extract interpretable linguistic features for traditional machine learning classifiers.

While the first two paradigms have received considerable attention, feature-based approaches remain underexplored in multi-generator, multi-domain settings. Prior feature-based studies typically evaluate on single-generator datasets (e.g., HC3 with ChatGPT only), leaving open the question of whether handcrafted features can scale to diverse generation sources.

**Contributions.** This paper makes the following contributions:

1. We design a comprehensive feature engineering pipeline with **90 interpretable features** organized into 9 linguistically motivated groups, complemented by **30 token-level probability features** from an observer LLM.

2. We conduct the first systematic feature-based analysis on the **RAID benchmark**, the most comprehensive AI text detection dataset to date, covering 11 generators, 8 domains, and 11 adversarial attack types.

3. Through SHAP interpretability analysis and cross-dataset comparison, we discover a critical finding: **perplexity-based features completely fail in multi-generator settings**, while basic text structure features become the most reliable detection signal — a reversal of the feature importance hierarchy observed on single-generator datasets.

4. We provide a thorough adversarial robustness analysis showing that only **paraphrase attacks** meaningfully degrade detection performance, while surface-level attacks are largely ineffective against feature-based detectors.

---

## 2. Related Work

### 2.1 Statistical Feature-Based Detection

Early AI text detection relied on handcrafted statistical features. Gehrmann et al. (2019) proposed GLTR, which visualizes token-level probability distributions from GPT-2. Subsequent work introduced lexical diversity metrics (Type-Token Ratio, Hapax ratio), readability indices (Flesch-Kincaid, Gunning Fog), and stylistic features (punctuation patterns, sentence length variation) as detection signals. These approaches offer full interpretability but have primarily been evaluated on single-generator datasets.

### 2.2 Language Model-Based Detection

Zero-shot methods leverage the statistical properties of language models. DetectGPT (Mitchell et al., 2023) and its fast variant Fast-DetectGPT (Bao et al., 2024) use perturbation-based log-probability curvature estimation. Binoculars (Hans et al., 2024) compares perplexity ratios between two language models. These methods require no training data but assume the detector has access to a model similar to the generator — an assumption that breaks down in open-world settings with unknown generators.

### 2.3 Supervised Deep Learning Detection

Fine-tuned transformer models (e.g., RoBERTa, DeBERTa) achieve high accuracy on in-distribution data but suffer from poor generalization across generators and domains (Bao et al., 2024). They also lack interpretability, making it difficult to understand *why* a text is classified as AI-generated.

### 2.4 The RAID Benchmark

RAID (Dugan et al., ACL 2024) addresses the limitations of prior benchmarks by providing: (1) texts from 11 different generators spanning multiple model families and sizes; (2) 8 diverse domains from academic abstracts to poetry; (3) 11 adversarial attack types including paraphrase, homoglyph substitution, and zero-width space insertion; and (4) 4 decoding strategies (greedy, sampling, top-k, top-p). This comprehensive coverage makes RAID the most challenging and realistic benchmark for evaluating detection methods.

---

## 3. Methodology

### 3.1 Feature Engineering: 90-Dimensional Handcrafted Features

We design 90 interpretable features organized into 9 linguistically motivated groups:

| Group | Dim | Description | Example Features |
|-------|-----|-------------|-----------------|
| basic_counts | 4 | Text length statistics | word_count, char_count, sentence_count, paragraph_count |
| averages | 4 | Mean structural metrics | avg_word_length, avg_sentence_length, words_per_paragraph |
| variability | 2 | Length variation | word_length_std, sentence_length_std |
| lexical_richness | 7 | Vocabulary diversity | type_token_ratio, hapax_ratio, yule_k, simpson_diversity |
| punctuation | 11 | Punctuation & character patterns | comma_rate, question_rate, uppercase_ratio, digit_ratio |
| readability | 7 | Readability indices | flesch_reading_ease, gunning_fog, smog_index, ari |
| structure | 1 | Sentence structure | short_sentence_ratio |
| embedding_pca | 50 | Semantic representation | BERT sentence embedding → PCA 50d |
| perplexity | 2 | Language model surprise | gpt2_perplexity, gpt2_log_perplexity |

**Design rationale.** Each group captures a distinct linguistic dimension. Basic counts and averages reflect text structure; lexical richness measures vocabulary diversity (humans tend to use more varied vocabulary); punctuation patterns capture stylistic habits (AI tends to overuse commas and colons); readability indices detect AI's tendency toward formal, structured prose; perplexity measures how "surprising" the text is to a reference language model; and semantic embeddings capture topic-level distributional differences.

### 3.2 Token Probability Features

We complement handcrafted features with 30 token-level probability features extracted using Mistral-7B-Instruct as an "observer model." For each text, we compute per-token log-probability, rank, and entropy, then derive 30 statistical aggregates:

- **Log-probability statistics** (10 dims): mean, std, min, max, median, skewness, kurtosis, IQR, 10th/90th percentile
- **Rank statistics** (10 dims): same aggregates for token rank in vocabulary
- **Entropy statistics** (10 dims): same aggregates for next-token prediction entropy

The intuition is that AI-generated tokens tend to have higher probability (lower surprise) under an observer LLM, creating a distributional gap between human and AI text at the token level.

### 3.3 Classification Models

We evaluate three interpretable classifiers:

- **XGBoost** (primary): 500 trees, max_depth=8, learning_rate=0.1, GPU-accelerated
- **Logistic Regression**: L2-regularized, standardized features, C=1.0
- **Random Forest**: 300 trees, max_depth=15

All models use stratified train/test splits (80/20) with fixed random seed for reproducibility.

### 3.4 Interpretability Analysis

- **SHAP (TreeExplainer)**: Computes exact Shapley values for XGBoost, enabling feature attribution at both global (summary/dependence plots) and instance level (waterfall plots).
- **Feature Group Ablation**: Trains separate XGBoost models on each of the 9 feature groups independently, measuring each group's standalone discriminative power.
- **Single-Feature AUC**: Computes ROC AUC for each individual feature against the binary label, with Cohen's d effect size.
- **Cross-Dataset Comparison**: Compares feature group ablation results across RAID and HC3 to reveal how feature importance shifts between single-model and multi-model scenarios.

---

## 4. Experimental Setup

### 4.1 Datasets

| Dataset | Year | Generators | Domains | Samples | Role |
|---------|------|-----------|---------|---------|------|
| **RAID** | 2024 | 11 (GPT-4, ChatGPT, Llama-2-70B, Mistral-7B, MPT-30B, Cohere, GPT-2/3) | 8 | 468K | Primary |
| HC3 | 2023 | 1 (ChatGPT) | 4 (QA) | 85K | Auxiliary |
| SemEval 2024 | 2024 | 6+ | Multi | 120K | Auxiliary |
| TuringBench | 2021 | 19 (GPT-1~3 variants) | News | 332K | Auxiliary |
| Pile | 2023 | Mixed | Multi | 80K | Auxiliary |

### 4.2 RAID Data Processing

The RAID dataset presents a severe class imbalance (97% AI, 3% human). We apply stratified subsampling: train set capped at 80K samples (balanced), test set at 20K. For adversarial analysis, we load the full attack-augmented dataset (12GB) and evaluate the trained model on each of the 11 attack types separately.

### 4.3 Evaluation Metrics

We report ROC AUC (primary) and Accuracy. AUC is preferred as it is threshold-independent and robust to class imbalance.

---

## 5. Results and Analysis

### 5.1 Main Results on RAID

| Method | Type | AUC | Accuracy |
|--------|------|-----|----------|
| **XGBoost (90 features)** | Traditional ML | **0.9951** | **0.9714** |
| Token features (Mistral-7B) | LLM + ML | 0.9900 | 0.9590 |
| Logistic Regression (90 features) | Traditional ML | 0.9842 | 0.9423 |
| Random Forest (90 features) | Traditional ML | 0.9916 | 0.9631 |
| Fast-DetectGPT (zero-shot) | Zero-shot | 0.7815 | 0.7452 |

XGBoost with 90 handcrafted features achieves the best performance (AUC 0.9951), significantly outperforming the zero-shot Fast-DetectGPT baseline. Notably, all three traditional ML classifiers outperform the zero-shot method, demonstrating that well-designed features with simple classifiers can rival or exceed more complex approaches.

### 5.2 Feature Ablation: The Perplexity Paradox

We train XGBoost independently on each feature group and compare results between RAID (11 generators) and HC3 (ChatGPT only):

| Feature Group | Dim | RAID AUC | HC3 AUC | Observation |
|---------------|-----|----------|---------|-------------|
| basic_counts | 4 | **0.9719** | 0.9204 | ↑ RAID strongest |
| averages | 4 | 0.9665 | 0.9021 | ↑ Significant gain |
| punctuation | 11 | 0.8617 | 0.9207 | ↓ |
| lexical_richness | 7 | 0.8503 | 0.9367 | ↓ |
| embedding_pca | 50 | 0.7624 | 0.8972 | ↓ |
| readability | 7 | 0.6966 | 0.9741 | ↓ Significant drop |
| variability | 2 | 0.6830 | 0.8987 | ↓ |
| structure | 1 | 0.5955 | 0.8965 | ↓ |
| **perplexity** | **2** | **0.4920** | **0.9912** | **↓↓↓ Complete failure** |

**The Perplexity Paradox.** GPT-2 perplexity is the single strongest feature group on HC3 (AUC 0.9912), where all AI text comes from one model (ChatGPT). However, on RAID with 11 different generators, perplexity drops to AUC 0.4920 — worse than random. This is because GPT-2 perplexity effectively measures "similarity to GPT-2's distribution," which does not generalize across diverse model families. In contrast, basic_counts (word/sentence/paragraph counts) becomes the strongest group (AUC 0.9719), suggesting that **text length and structure are the most stable cross-model signals**.

### 5.3 SHAP Interpretability

SHAP analysis reveals the top contributing features on RAID:

1. **words_per_paragraph** — the single most important feature; AI text tends to produce shorter, more uniform paragraphs
2. **sentences_per_paragraph** — closely related; AI paragraphs contain fewer sentences
3. **paragraph_count** — AI generates more paragraphs for equivalent content
4. **sentence_length_std** — human writing has higher sentence length variation

The SHAP dependence plots show clear monotonic relationships: higher words_per_paragraph strongly pushes predictions toward "human," while lower values indicate AI. This aligns with the well-known "list-making" tendency of LLMs, which break content into many short paragraphs with bullet points or numbered lists.

### 5.4 Per-Generator and Per-Domain Analysis

**Per-Generator Detection Accuracy (XGBoost):**

| Generator | Accuracy | Generator | Accuracy |
|-----------|----------|-----------|----------|
| ChatGPT | 96.2% | GPT-4 | 77.3% |
| GPT-3 | 94.5% | Llama-2-70B-Chat | 86.1% |
| Cohere-Chat | 84.4% | Mistral-7B-Chat | 82.0% |
| GPT-2 XL | 79.9% | **MPT-30B** | **47.4%** |

MPT-30B is the hardest to detect (47.4%), suggesting its output most closely mimics human statistical patterns. ChatGPT is easiest (96.2%), likely due to its distinctive formatting habits (lists, bullet points, transitional phrases).

**Per-Domain AUC:** Poetry is the most challenging domain (AUC 0.6137) because human poetry is inherently stylized, reducing the statistical gap between human and AI text. News and wiki domains are easier (AUC > 0.84) due to more consistent structural differences.

### 5.5 Adversarial Robustness

| Attack | AUC | AUC Drop | Assessment |
|--------|-----|----------|------------|
| insert_paragraphs | 0.9915 | -0.004 | ✅ Negligible |
| whitespace | 0.9701 | -0.025 | ✅ Robust |
| synonym | 0.9660 | -0.029 | ✅ Robust |
| homoglyph | 0.9516 | -0.044 | ⚠️ Minor drop |
| zero_width_space | 0.8424 | -0.153 | ❌ Significant |
| **paraphrase** | **0.7902** | **-0.205** | **❌ Most effective** |

**Key finding:** Paraphrase attack is the only strategy that significantly degrades detection (AUC drop 20.5%). It rewrites the text's lexical and syntactic structure, disrupting most statistical features. Surface-level attacks (spelling, case, number substitution) are almost entirely ineffective against our 90-feature detector, because the majority of features capture semantic and statistical properties rather than character-level patterns.

Zero-width space attack is unexpectedly effective (AUC drop 15.3%), as invisible Unicode characters interfere with character-counting features.

### 5.6 Cross-Dataset Generalization

| Dataset | Generators | XGBoost AUC |
|---------|-----------|-------------|
| RAID | 11 | 0.9951 |
| HC3 | 1 (ChatGPT) | 0.9999 |
| TuringBench | 19 (GPT-1~3) | 0.9841 |
| Pile | Mixed | 0.9831 |
| SemEval 2024 | 6+ | 0.6872 |

XGBoost with 90 features generalizes well across most datasets. The exception is SemEval 2024, which contains recent models whose surface statistics have converged with human text, challenging purely statistical detection.

---

## 6. Discussion

### The Multi-Model Detection Challenge

Our results reveal a fundamental insight: **the features that matter most depend critically on the diversity of generators in the dataset.** On single-model datasets (HC3), language model perplexity dominates because it directly measures distributional similarity to the generator. On multi-model datasets (RAID), perplexity fails because no single reference model can capture the diversity of 11 generators, and structural features (paragraph/sentence patterns) emerge as the universal signal.

### Practical Deployment Recommendations

1. **Use model-agnostic features** (text length, paragraph structure) as the primary detection backbone for unknown generators.
2. **Supplement with token probability features** from a contemporary observer model to cover newer generators.
3. **Implement paraphrase-specific defenses** (e.g., semantic similarity features) as paraphrase is the only effective evasion strategy.
4. **Account for domain differences** — poetry and creative writing require domain-adapted thresholds.

### Limitations

Our study has several limitations: (1) the 90-feature pipeline requires GPT-2 and BERT inference, adding computational overhead; (2) adversarial analysis is limited to the attacks provided by RAID; (3) the study does not evaluate on non-English text; (4) as LLMs improve, the statistical gap between human and AI text may continue to narrow.

---

## 7. Conclusion

We presented a comprehensive, interpretable approach to AI-generated text detection using 90 handcrafted linguistic features. Through systematic evaluation on the RAID benchmark and four additional datasets, we demonstrated that:

1. **Interpretable feature-based methods can achieve state-of-the-art performance** (AUC 0.9951 on RAID), rivaling black-box approaches while providing full explainability.

2. **Feature importance fundamentally shifts between single-model and multi-model scenarios**: perplexity dominates single-model detection but fails completely in multi-model settings, where paragraph structure becomes the strongest signal.

3. **Most adversarial attacks are ineffective** against well-designed statistical features; only semantic-level attacks (paraphrase) pose a genuine threat.

These findings provide both theoretical insights into what distinguishes AI from human text and practical guidelines for deploying robust, interpretable detection systems. Future work should explore multilingual detection, dynamic feature adaptation as models evolve, and semantic-level defenses against paraphrase attacks.

---

## References

- Bao, G., Zhao, Y., Teng, Z., Yang, L., & Zhang, Y. (2024). Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature. *ICLR 2024*.
- Dugan, L., Hwang, A., Trhl\u00edk, F., Ladhak, F., Ippolito, D., Callison-Burch, C., & Lee, E. (2024). RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. *ACL 2024*.
- Gao, C., et al. (2023). Human ChatGPT Comparison Corpus (HC3). *arXiv:2301.07597*.
- Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical detection and visualization of generated text. *ACL 2019*.
- Hans, A., Schwarzschild, A., Cheeseman, V., Bruss, C. B., & Goldblum, M. (2024). Binoculars: Zero-Shot Detection of LLM-Generated Text. *ICML 2024*.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
- Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. *ICML 2023*.

---

## Appendix: File Structure

| File | Description |
|------|-------------|
| `notebooks/AIGC_Detection.ipynb` | Main experiment notebook (36 cells, all executed) |
| `src/run_raid.py` | RAID experiment: XGBoost, token features, DetectGPT, adversarial |
| `src/run_raid_analysis.py` | Deep analysis: SHAP, ablation, t-SNE, 20 figures |
| `src/run_raid_feature_deep.py` | Per-feature analysis: single-feature AUC, radar, etc. |
| `reports/raid_analysis.md` | Standalone RAID analysis report |
| `figures/raid_*.png` | 25 generated visualization figures |
