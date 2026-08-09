# RAID Benchmark: Multi-Generator Detection & Adversarial Robustness Analysis

> **数据集**: RAID — Robust AI Text Detection Benchmark (Dugan et al., ACL 2024)
>
> **论文**: [arxiv 2405.07940](https://arxiv.org/abs/2405.07940)
>
> **实验脚本**: [`src/run_raid.py`](https://github.com/HaibinLai/detect-AI-generated-text/blob/main/src/run_raid.py)（方法对比） | [`src/run_raid_analysis.py`](https://github.com/HaibinLai/detect-AI-generated-text/blob/main/src/run_raid_analysis.py)（深度分析）
>
> **父项目**: [README.md](https://github.com/HaibinLai/detect-AI-generated-text/blob/main/README.md) — 完整的多数据集 AI 文本检测研究

---

## 1. 数据集概览

RAID 是目前最全面的 AI 文本检测基准，由 UPenn 团队于 ACL 2024 发布。

### 1.1 规模

| 指标 | 数值 |
|------|------|
| 总文本数 | 467,985 |
| 人类文本 | 13,371 (2.9%) |
| AI 文本 | 454,614 (97.1%) |
| AI 生成器 | 11 种 |
| 领域 | 8 个 |
| 对抗攻击 | 11 种 |
| 解码策略 | 4 种 |

### 1.2 生成器分布

| 生成器 | 数量 | 类型 | 时代 |
|--------|------|------|------|
| Llama-2-70B-Chat | 53,484 | 开源 | 2023 |
| MPT-30B | 53,484 | 开源 | 2023 |
| MPT-30B-Chat | 53,484 | 开源 | 2023 |
| GPT-2 XL | 53,484 | 开源 | 2019 |
| Mistral-7B | 53,484 | 开源 | 2023 |
| Mistral-7B-Chat | 53,484 | 开源 | 2023 |
| GPT-3 (davinci) | 26,742 | 闭源 | 2020 |
| Cohere | 26,742 | 闭源 | 2023 |
| ChatGPT | 26,742 | 闭源 | 2022 |
| GPT-4 | 26,742 | 闭源 | 2023 |
| Cohere-Chat | 26,742 | 闭源 | 2023 |

### 1.3 领域分布

| 领域 | 数量 | 描述 |
|------|------|------|
| books | 62,335 | 书籍摘要 |
| news | 62,300 | NYT 新闻 |
| reddit | 62,265 | Reddit 帖子 |
| wiki | 62,265 | Wikipedia |
| recipes | 62,020 | 食谱 |
| poetry | 61,985 | 诗歌 |
| abstracts | 61,810 | arXiv 摘要 |
| reviews | 33,005 | IMDb 影评 |

### 1.4 对抗攻击

| 攻击类型 | 描述 |
|----------|------|
| paraphrase | 用 LLM 释义重写 |
| synonym | 同义词替换 |
| homoglyph | 视觉相似字符替换（如 a→α） |
| zero_width_space | 插入零宽空格 |
| whitespace | 添加额外空白 |
| upper_lower | 大小写随机交换 |
| number | 数字替换 |
| article_deletion | 删除冠词 |
| alternative_spelling | 使用替代拼写 |
| perplexity_misspelling | 基于困惑度的拼写错误 |
| insert_paragraphs | 插入段落分隔 |

### 1.5 解码策略

| 策略 | 温度 | 重复惩罚 |
|------|------|----------|
| Greedy | T=0 | 无 |
| Sampling | T=1 | 无 |
| Greedy + RepPenalty | T=0 | θ=1.2 |
| Sampling + RepPenalty | T=1 | θ=1.2 |

---

## 2. 实验设置

### 2.1 数据划分

从 468K 无攻击数据中分层采样：

| 划分 | 样本数 | AI 比例 |
|------|--------|---------|
| 训练集 | 50,696 | 78.9% |
| 测试集 | 12,675 | 78.9% |

### 2.2 方法

| 方法 | 范式 | 是否训练 | 特征维度 |
|------|------|----------|----------|
| XGBoost (90 features) | 手工特征 + 传统 ML | 有监督 | 90 |
| Token features (Mistral-7B) | LLM 概率特征 + ML | 有监督 | 30 |
| Fast-DetectGPT | 零样本概率曲率 | 无需训练 | — |

---

## 3. 方法对比结果

### 3.1 总体性能

| 方法 | ROC AUC | Accuracy | 耗时 |
|------|---------|----------|------|
| **XGBoost (90 特征)** | **0.9951** | **0.9714** | 15 min |
| Token features (Mistral-7B) | 0.9900 | 0.9590 | 54 min |
| Fast-DetectGPT (零样本) | 0.7815 | 0.7452 | 2 min |

### 3.2 Per-Model 检测准确率

| 生成器 | Accuracy | 生成器 | Accuracy |
|--------|----------|--------|----------|
| ChatGPT | 96.2% | GPT-4 | 77.3% |
| GPT-3 | 94.5% | Llama-2-70B-Chat | 86.1% |
| Cohere-Chat | 84.4% | Mistral-7B-Chat | 82.0% |
| Cohere | 77.9% | MPT-30B | 47.4% |
| GPT-2 XL | 79.9% | MPT-30B-Chat | 64.0% |
| Mistral-7B | 59.5% | Human | 74.5% |

**分析**：
- **ChatGPT 和 GPT-3 最易检测**（>94%），因为它们的生成模式最"模板化"
- **MPT-30B 最难检测**（47.4%），其生成风格最接近人类文本
- **Mistral-7B（非 chat）也较难**（59.5%），base model 的输出更随机
- **Human 准确率 74.5%** 意味着有 25.5% 的人类文本被误判为 AI

### 3.3 Per-Domain AUC

| 领域 | AUC | 分析 |
|------|-----|------|
| wiki | 0.8441 | AI 的百科风格与 Wikipedia 差异最大 |
| reviews | 0.8412 | 影评的个人色彩使 AI 更易区分 |
| books | 0.8214 | — |
| reddit | 0.8214 | — |
| news | 0.8126 | — |
| abstracts | 0.7833 | 学术摘要的格式较固定，AI 更易伪装 |
| recipes | 0.7407 | 食谱的结构化格式对 AI 友好 |
| **poetry** | **0.6137** | 诗歌本身高度风格化，人类/AI 统计特征差异最小 |

![RAID comparison](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_comparison.png)

---

## 4. 对抗攻击鲁棒性分析

### 4.1 XGBoost 在 11 种攻击下的表现

| 攻击方式 | AUC | AUC 下降 | 评价 |
|----------|-----|----------|------|
| insert_paragraphs | 0.9915 | -0.004 | ✅ 几乎无影响 |
| whitespace | 0.9701 | -0.025 | ✅ 鲁棒 |
| alternative_spelling | 0.9686 | -0.027 | ✅ 鲁棒 |
| perplexity_misspelling | 0.9682 | -0.027 | ✅ 鲁棒 |
| upper_lower | 0.9680 | -0.027 | ✅ 鲁棒 |
| number | 0.9686 | -0.027 | ✅ 鲁棒 |
| synonym | 0.9660 | -0.029 | ✅ 鲁棒 |
| article_deletion | 0.9676 | -0.028 | ✅ 鲁棒 |
| homoglyph | 0.9516 | -0.044 | ⚠️ 轻微下降 |
| **zero_width_space** | **0.8424** | **-0.153** | ❌ 显著下降 |
| **paraphrase** | **0.7902** | **-0.205** | ❌ 最致命攻击 |

### 4.2 攻击分类

**表面攻击（AUC 下降 < 5%）**：
- 拼写、大小写、数字、冠词删除、同义词替换、段落插入
- 这些攻击只改变字符/词级别的表面形式，不影响文本的统计结构
- XGBoost 的 90 维特征中，大量是句子级和文档级统计，对字符级干扰天然鲁棒

**深层攻击（AUC 下降 > 15%）**：
- **释义（paraphrase）**：AUC 下降 20.5%，是最有效的规避手段。释义重写从根本上改变了文本的词汇选择和句法结构，破坏了词汇丰富度、可读性等核心特征
- **零宽空格（zero_width_space）**：AUC 下降 15.3%。不可见的 Unicode 字符干扰了基于字符计数的特征（char_count、punct_ratio 等），导致统计量失真

### 4.3 特征偏移分析

释义攻击使 AI 文本的特征分布向人类文本方向偏移：

![Attack feature shift](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_attack_feature_shift.png)

![RAID adversarial](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_adversarial.png)

---

## 5. 特征消融实验

### 5.1 各特征组独立判别能力

| 特征组 | 特征数 | RAID AUC | HC3 AUC | 差异分析 |
|--------|--------|----------|---------|----------|
| **basic_counts** | 4 | **0.9719** | 0.9204 | ↑ 长度差异是最稳定的跨模型信号 |
| averages | 4 | 0.9665 | 0.9021 | ↑ 句长、段落长度差异跨模型稳定 |
| punctuation | 11 | 0.8617 | 0.9207 | ↓ 不同模型标点习惯各异 |
| lexical_richness | 7 | 0.8503 | 0.9367 | ↓ 现代模型的词汇丰富度提升 |
| embedding_pca | 50 | 0.7624 | 0.8972 | ↓ 语义空间更复杂 |
| readability | 7 | 0.6966 | 0.9741 | ↓↓ 不同模型的可读性各异 |
| variability | 2 | 0.6830 | 0.8987 | ↓ |
| structure | 1 | 0.5955 | 0.8965 | ↓ |
| **perplexity** | 2 | **0.4920** | **0.9912** | ↓↓↓ 完全失效 |

### 5.2 关键发现：困惑度在多模型场景下失效

这是 RAID 分析最重要的发现之一：

**HC3 上**：GPT-2 困惑度是最强特征（AUC 0.9912），因为 HC3 只有 ChatGPT 一种生成器，AI 文本的困惑度一致地低于人类文本。

**RAID 上**：GPT-2 困惑度完全失效（AUC 0.4920，低于随机猜测 0.5），因为：
1. 11 种生成器的困惑度分布各不相同
2. 开源 base 模型（GPT-2 XL、Mistral-7B base）的困惑度可能高于人类文本
3. GPT-2 作为"观察者"，对不同生成器的评估标准不一致

**启示**：在实际部署中，如果面对的 AI 模型未知或多样化，**不应依赖困惑度作为核心特征**，而应以文本长度、句法结构等模型无关的统计量为主。

![RAID feature ablation](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_feature_ablation.png)

---

## 6. SHAP 可解释性分析

### 6.1 SHAP Summary（Top 25 特征）

SHAP beeswarm 图展示了 XGBoost 在 RAID 上最依赖的特征：

![RAID SHAP summary](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_shap_summary.png)

### 6.2 SHAP 依赖图（Top 6 特征）

展示各特征值与 SHAP 值的关系，颜色为最强交互特征：

![RAID SHAP dependence](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_shap_dependence.png)

### 6.3 SHAP Waterfall（4 个典型案例）

对最自信的正确分类和最典型的错误分类，展示 SHAP 归因：

- **Correct Human**：哪些特征使模型确信这是人类文本
- **Correct AI**：哪些特征使模型确信这是 AI 文本
- **False Positive**：人类文本为什么被误判为 AI
- **False Negative**：AI 文本为什么被误判为人类

![RAID SHAP waterfall](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_shap_waterfall.png)

### 6.4 LR 系数分析

Logistic Regression 的标准化系数提供了线性可解释性：

- 正系数 → 指向 AI
- 负系数 → 指向 Human

![RAID LR coefficients](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_lr_coefficients.png)

---

## 7. 领域与特征分布分析

### 7.1 领域特征对比

8 个领域中，关键特征在 Human/AI 间的分布差异：

![RAID domain comparison](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_domain_feature_comparison.png)

### 7.2 特征相关性

90 维特征的 Pearson 相关矩阵：

![RAID correlation](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_correlation_heatmap.png)

### 7.3 PCA 降维可视化

90 维特征 → 2D PCA：

![RAID PCA](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_pca.png)

### 7.4 t-SNE 降维可视化

90 维特征 → 2D t-SNE（5000 样本，perplexity=30）：

![RAID t-SNE](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_tsne.png)

---

## 8. Token 概率特征分析

### 8.1 方法

使用 Mistral-7B-Instruct 提取 30 维 token-level 概率特征，训练 XGBoost。

| 指标 | 值 |
|------|-----|
| Token 特征 AUC | 0.9900 |
| Token 特征 Accuracy | 0.9590 |

### 8.2 Token SHAP 归因

![RAID token SHAP](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_token_shap.png)

### 8.3 Token 特征分布（Human vs AI）

![RAID token distributions](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_token_distributions.png)

### 8.4 Per-Model Token 特征

11 个生成器 + Human 的 token 特征分布对比。可以看到 Human（蓝色）与各 AI 模型（红色）在 `rank_top100_frac`、`lp_mean` 等特征上的差异程度各不相同：

![RAID per-model features](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_per_model_features.png)

### 8.5 Token 概率热力图

每个词按 Mistral-7B 的预测概率着色（绿=可预测，红=出人意料）。直观展示 AI 文本的"流畅度均匀性" vs 人类文本的"意外用词"：

![RAID token heatmap](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_token_heatmap.png)

### 8.6 Token t-SNE

30 维 token 特征降到 2D。Human 和 AI 有清晰的分离趋势：

![RAID token t-SNE](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_token_tsne.png)

### 8.7 误分类分析

预测置信度分布、正确/错误分类的特征均值对比、TP/TN/FP/FN 饼图：

![RAID misclassification](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_misclassification.png)

---

## 9. RAID 独有分析

### 9.1 Model × Domain 检测准确率热力图

11 个模型 × 8 个领域的检测准确率矩阵。可以精确定位哪些 (模型, 领域) 组合最难检测：

![RAID model domain heatmap](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_model_domain_heatmap.png)

### 9.2 解码策略对比

不同解码策略（Greedy, Sampling, +Repetition Penalty）对 AI 文本统计特征的影响：

![RAID decoding comparison](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_decoding_comparison.png)

### 9.3 对抗攻击特征偏移（Paraphrase）

对比 Human / AI (original) / AI (paraphrased) 的特征分布，展示释义攻击如何使 AI 文本"伪装"为人类文本：

![RAID attack feature shift](https://cdn.jsdelivr.net/gh/HaibinLai/detect-AI-generated-text@main/figures/raid_attack_feature_shift.png)

---

## 10. 与其他数据集的对比分析

### 10.1 跨数据集方法对比

| 数据集 | XGBoost (90) | Token (Mistral) | Fast-DetectGPT |
|--------|:-:|:-:|:-:|
| HC3 (单模型) | **0.9999** | 0.9998 | 0.9292 |
| SemEval (6 模型) | 0.6872 | **0.9784** | 0.8068 |
| TuringBench (19 旧模型) | **0.9841** | 0.4853 | 0.6038 |
| Pile (混合) | 0.9831 | **0.9918** | 0.8889 |
| **RAID (11 模型)** | **0.9951** | 0.9900 | 0.7815 |

### 10.2 RAID 的定位

RAID 在数据集家族中的独特价值：

| 特性 | HC3 | SemEval | TuringBench | Pile | **RAID** |
|------|-----|---------|-------------|------|---------|
| 模型数量 | 1 | 6 | 19 | 混合 | **11** |
| 包含 GPT-4 | ❌ | ✅ | ❌ | ❌ | **✅** |
| 对抗攻击 | ❌ | ❌ | ❌ | ❌ | **✅ (11 种)** |
| 解码策略 | — | — | — | — | **✅ (4 种)** |
| 多领域 | 5 | 5 | 1 | 1 | **8** |

**RAID 是唯一同时覆盖多模型、多领域、多攻击、多解码策略的基准**，这使得它最适合评估检测系统的实际部署能力。

---

## 11. Key Findings

1. **XGBoost 90 特征在 RAID 上依然最强**（AUC 0.9951），证明手工特征在多模型场景下的有效性。

2. **GPT-2 困惑度在多模型场景下完全失效**（AUC 0.4920）。这是与 HC3（AUC 0.9912）最大的差异。多模型检测不能依赖单一语言模型的困惑度。

3. **基础计数特征（文本长度）是最稳定的跨模型信号**（AUC 0.9719）。不同 AI 模型生成文本的长度分布差异是最一致的判别信号。

4. **释义攻击（paraphrase）是唯一能显著降低检测的手段**（AUC 下降 20.5%），其他 10 种攻击对 XGBoost 几乎无效。

5. **诗歌领域最难检测**（AUC 0.6137），因为诗歌本身高度风格化，人类/AI 的统计特征差异最小。

6. **MPT-30B 是最难检测的生成器**（Accuracy 47.4%），其输出最接近人类文本的统计分布。

7. **实际部署建议**：面对未知 AI 模型时，应以文本长度、句法结构等模型无关特征为主，辅以 token 概率特征；同时需针对释义攻击做专门防御。

---

## 12. 文件列表

| 文件 | 说明 |
|------|------|
| `src/run_raid.py` | RAID 方法对比实验 + 对抗攻击评估 |
| `src/run_raid_analysis.py` | 深度分析与可解释性（生成 20 张图） |
| `data/external/raid/train_none.csv` | RAID 无攻击数据 (468K, 765MB) |
| `data/external/raid/train.csv` | RAID 含攻击数据 (11GB) |
| `data/processed/raid_features_*.csv` | 缓存的 90 维特征 |
| `data/processed/token_features_raid_*.csv` | 缓存的 token 概率特征 |

### 生成的图表 (20 张)

| 图表 | 说明 |
|------|------|
| `raid_comparison.png` | 方法对比柱状图 |
| `raid_adversarial.png` | 对抗攻击 AUC 下降图 |
| `raid_correlation_heatmap.png` | 特征相关性热力图 |
| `raid_domain_feature_comparison.png` | 8 领域特征对比 |
| `raid_pca.png` | PCA 2D 散点图 |
| `raid_tsne.png` | t-SNE 2D 散点图 |
| `raid_shap_summary.png` | SHAP beeswarm (top 25) |
| `raid_shap_dependence.png` | SHAP 依赖图 (top 6) |
| `raid_shap_waterfall.png` | SHAP waterfall (4 案例) |
| `raid_feature_ablation.png` | 特征消融实验 |
| `raid_lr_coefficients.png` | LR top-20 系数 |
| `raid_token_shap.png` | Token SHAP 归因 |
| `raid_token_distributions.png` | Token 特征分布 |
| `raid_per_model_features.png` | Per-model token 特征 |
| `raid_token_heatmap.png` | Token 概率热力图 |
| `raid_token_tsne.png` | Token t-SNE |
| `raid_misclassification.png` | 误分类分析 |
| `raid_model_domain_heatmap.png` | Model × Domain 热力图 |
| `raid_decoding_comparison.png` | 解码策略对比 |
| `raid_attack_feature_shift.png` | 对抗攻击特征偏移 |

---

## References

1. Dugan, L., Hwang, A., Trhlik, F., et al. (2024). *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors*. ACL 2024. [arxiv 2405.07940](https://arxiv.org/abs/2405.07940)
2. Bao, G., et al. (2024). *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature*. ICLR 2024.
3. Hans, A., et al. (2024). *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*. ICML 2024.
