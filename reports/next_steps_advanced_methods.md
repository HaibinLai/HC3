# Next Steps: Advanced AI Text Detection Methods

## Overview

当前项目已完成基于 90 个手工特征 + XGBoost 的检测（AUC 0.9999）。下一步
实现三种最新的检测方法，覆盖三个不同的技术范式，形成完整的方法对比。

| # | 方法 | 范式 | 是否需要训练 | 核心依赖 |
|---|---|---|---|---|
| 1 | DetectGPT / Fast-DetectGPT | 零样本概率扰动 | 不需要 | GPT-2 + T5 |
| 2 | Fine-tune RoBERTa | 监督微调 | 需要 | RoBERTa-base |
| 3 | Binoculars | 零样本双模型对比 | 不需要 | 两个不同 LLM |

---

## 背景知识：GLTR 和 Token 概率分布

在介绍三种方法之前，先解释一个核心直觉。

**GLTR (Giant Language Model Test Room)** 是 Gehrmann et al. (2019) 提出的
可视化工具，它揭示了 AI 文本检测的一个基本原理：

### 核心思想

给定一个语言模型（如 GPT-2），对文本中的每个词，计算它在模型预测的
词表概率分布中的排名：

$$r_t = \mathrm{rank}(x_t \mid x_1, x_2, \dots, x_{t-1})$$

即：给定前文，模型认为第 $t$ 个词是第几可能出现的词。

然后把每个词按排名分到 4 个桶中：

| 颜色 | 排名范围 | 含义 |
|---|---|---|
| 绿色 | Top-10 | 模型认为这个词非常可能出现 |
| 黄色 | Top-100 | 模型觉得比较可能 |
| 红色 | Top-1000 | 模型觉得不太可能 |
| 紫色 | > 1000 | 模型觉得很意外 |

### 关键发现

- **AI 生成文本**：大量绿色（Top-10），因为 AI 本身就是按概率从高到低采样的
- **人类文本**：更多黄/红/紫色，因为人类经常选择"出人意料"的词

这就是为什么你的实验中 `gpt2_perplexity` 是最强特征的本质原因——它是
GLTR 思想的标量压缩版。GLTR 保留了每个词的分布信息，而 perplexity 把
它压缩成了一个数。

### GLTR 的局限

GLTR 只提供可视化，没有自动判别。下面三种方法都是在这个基础上发展出来的
自动检测算法。

---

## Method 1: DetectGPT / Fast-DetectGPT

### 论文

- DetectGPT: Mitchell et al., *"DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"*, ICML 2023. (arxiv 2301.07597)
- Fast-DetectGPT: Bao et al., *"Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"*, ICLR 2024. (arxiv 2310.05130)

### 核心直觉

DetectGPT 的关键观察：

> AI 生成的文本处于语言模型对数概率函数的**局部极大值**附近。
> 对文本做微小扰动后，对数概率几乎只会**下降**。
> 而人类文本不在极大值附近，扰动后对数概率可能上升也可能下降。

用一个类比：AI 文本站在山顶，往任何方向走都是下坡；人类文本站在山腰，
往不同方向走有上有下。

### 算法步骤 (DetectGPT)

给定待检测文本 $x$ 和一个评分模型 $p_\theta$（如 GPT-2）：

**Step 1: 计算原始对数概率**

$$\ell(x) = \frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

**Step 2: 生成 $K$ 个扰动文本**

使用 T5 (mask-filling model) 对原文做局部替换，生成 $\tilde{x}_1, \tilde{x}_2, \dots, \tilde{x}_K$。

具体做法：随机 mask 掉原文中约 15% 的 token，用 T5 填充，得到语义相近但
措辞不同的变体。

**Step 3: 计算扰动后的对数概率**

$$\ell(\tilde{x}_k) = \frac{1}{T} \sum_{t=1}^{T} \log p_\theta(\tilde{x}_{k,t} \mid \tilde{x}_{k,<t})$$

**Step 4: 计算 DetectGPT 分数**

$$d(x) = \frac{\ell(x) - \frac{1}{K} \sum_{k=1}^{K} \ell(\tilde{x}_k)}{\sigma_{\tilde{x}}}$$

其中 $\sigma_{\tilde{x}}$ 是扰动文本对数概率的标准差。

**判别规则**：$d(x) > \tau$ 则判定为 AI 生成。

### Fast-DetectGPT 改进

DetectGPT 的瓶颈：需要生成 $K$（通常 100）个扰动文本再逐个计算概率，非常慢。

Fast-DetectGPT 的改进：**不做扰动**，而是利用条件概率直接估计曲率。

对每个位置 $t$，从模型的条件分布 $p_\theta(\cdot \mid x_{<t})$ 中采样
一个替代 token $\hat{x}_t$，然后比较：

$$\tilde{d}(x) = \frac{\frac{1}{T}\sum_t \log p_\theta(x_t \mid x_{<t}) - \frac{1}{T}\sum_t \log p_\theta(\hat{x}_t \mid x_{<t})}{\sigma}$$

直觉：如果原始 token 的概率远高于随机采样的 token，说明原文处于概率峰值
（更可能是 AI 生成的）。

**速度提升**：比 DetectGPT 快 340 倍，因为只需要一次 forward pass。

### 实现计划

```
src/run_detectgpt.py
```

- 评分模型：GPT-2 (medium 或 large)
- 扰动模型 (DetectGPT)：T5-large
- 扰动次数 $K = 100$
- 同时实现 DetectGPT 和 Fast-DetectGPT
- 在 HC3 测试集上评估 AUC，与 XGBoost baseline 对比
- 分析：不同文本长度下的检测准确率变化

### 预期对比

| 方法 | 需要训练数据? | 预期 AUC | 速度 |
|---|---|---|---|
| XGBoost (90 feat) | 是 | 0.9999 | 快（特征已缓存）|
| DetectGPT | 否 | ~0.95-0.98 | 慢（100 次扰动）|
| Fast-DetectGPT | 否 | ~0.95-0.98 | 中等（1 次 pass）|

零样本方法的 AUC 可能低于有监督方法，但**不需要任何训练数据**是其核心优势。

---

## Method 2: Fine-tune RoBERTa Classifier

### 背景

RoBERTa (Liu et al., 2019) 是 BERT 的优化版本，通过更大的数据量、更长的
训练和动态 masking 提升预训练效果。在文本分类任务中，RoBERTa 是最常用的
backbone 之一。

OpenAI 自己的 AI 文本检测器（2023 年发布后又下线）就是基于类似架构。

### 核心思想

直接在 HC3 数据上 fine-tune RoBERTa-base，让模型自己学习区分人类和 AI
文本的模式，不需要手工设计特征。

### 模型架构

```
输入文本
  |
  v
[RoBERTa-base encoder] (125M params)
  |
  v
[CLS] token embedding (768 维)
  |
  v
[Dropout(0.1)]
  |
  v
[Linear(768, 2)]  --> softmax --> P(human), P(chatgpt)
```

### 训练细节

| 参数 | 值 |
|---|---|
| 预训练模型 | `roberta-base` (125M params) |
| 最大序列长度 | 512 tokens |
| Batch size | 32 |
| Learning rate | 2e-5 (AdamW, linear warmup) |
| Epochs | 3 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| 数据划分 | 80% train / 10% val / 10% test |
| 硬件 | A100 80GB (FP16 混合精度) |

### 与 XGBoost 的对比意义

| 维度 | XGBoost (90 feat) | Fine-tune RoBERTa |
|---|---|---|
| 特征 | 手工设计 90 个 | 自动学习 |
| 可解释性 | 高（SHAP、系数） | 低（黑盒） |
| 参数量 | ~数千 | 125M |
| 训练数据依赖 | 需要标注数据 | 需要标注数据 |
| 跨域泛化 | 依赖特征的通用性 | 学到的模式可能更泛化 |
| 短文本性能 | 统计特征在短文本上不稳定 | Transformer 对短文本也有效 |

### 实现计划

```
src/run_roberta.py
```

- 使用 HuggingFace `transformers` + `Trainer` API
- 训练 RoBERTa-base 二分类
- 评估 AUC、Accuracy、F1
- 输出混淆矩阵
- 对比不同文本长度（短/中/长）下的准确率
- 可选：提取 [CLS] embedding 做 t-SNE 可视化，与手工特征的 t-SNE 对比

### 预期结果

| 方法 | 预期 AUC | 可解释性 |
|---|---|---|
| XGBoost (90 feat) | 0.9999 | 高 |
| RoBERTa fine-tune | ~0.999+ | 低 |

两者 AUC 接近，但 RoBERTa 在短文本上可能表现更好，因为它不依赖于
在短文本上不稳定的统计量（如 TTR、readability）。

---

## Method 3: Binoculars

### 论文

Hans et al., *"Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"*, ICML 2024. (arxiv 2401.12070)

### 核心直觉

> 如果一段文本是由某个 LLM 生成的，那么**任何能力相近的 LLM 都会觉得
> 这段文本很"正常"**（低困惑度）。但人类文本在不同模型之间的困惑度
> 差异更大。

类比：两个英语母语者（两个 LLM）读一篇 AI 写的文章，都觉得很流畅；
但读一篇人类写的文章，可能一个觉得有趣，一个觉得困惑——因为人类表达
更个性化、更不可预测。

### 算法

给定两个语言模型 $M_1$（observer）和 $M_2$（performer）：

**Step 1: 计算 cross-perplexity**

对文本 $x = (x_1, x_2, \dots, x_T)$：

$$\mathrm{PPL}_{M_1}(x) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p_{M_1}(x_t \mid x_{<t})\right)$$

$$\mathrm{PPL}_{M_2}(x) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p_{M_2}(x_t \mid x_{<t})\right)$$

**Step 2: 计算 Binoculars 分数**

$$B(x) = \frac{\mathrm{PPL}_{M_1}(x)}{\mathrm{PPL}_{M_2}(x)}$$

或者用对数形式（论文实际使用 cross-entropy 比值）：

$$B(x) = \frac{H_{M_1}(x)}{H_{M_2}(x)} = \frac{-\frac{1}{T}\sum_t \log p_{M_1}(x_t \mid x_{<t})}{-\frac{1}{T}\sum_t \log p_{M_2}(x_t \mid x_{<t})}$$

**Step 3: 判别**

$$B(x) < \tau \implies \mathrm{AI\ generated}$$

- AI 文本：两个模型的困惑度都低且接近，比值接近 1
- 人类文本：两个模型的困惑度不一致，比值偏离 1

### 为什么有效？

1. AI 文本是从某个概率分布采样的，任何相似的 LLM 对它的建模都很好
2. 人类文本包含个人风格、口语、错误、创意等，不同模型对这些的建模差异大
3. 两个模型的"共识程度"成为了 AI vs Human 的判别信号

### 与你现有 perplexity 特征的区别

| | 你的 gpt2\_perplexity | Binoculars |
|---|---|---|
| 模型数量 | 1 个 (GPT-2) | 2 个 |
| 信号 | 绝对困惑度 | 困惑度的**比值** |
| 鲁棒性 | 对文本领域敏感（专业文本困惑度天然高） | 比值消除了领域偏差 |
| 阈值 | 需要按领域调整 | 跨领域更稳定 |

### 模型选择

论文推荐使用同一系列但不同大小的模型：

| Observer ($M_1$) | Performer ($M_2$) |
|---|---|
| Falcon-7b | Falcon-7b-instruct |
| LLaMA-2-13b | LLaMA-2-13b-chat |

在我们的资源下（A100 80GB），可以选择：

- **方案 A**：GPT-2 medium + GPT-2 large（轻量，快速）
- **方案 B**：LLaMA-2-7b + LLaMA-2-7b-chat（更强，80GB 可以放下）
- **方案 C**：Mistral-7b + Mistral-7b-instruct（较新的开源模型）

### 实现计划

```
src/run_binoculars.py
```

- 实现 Binoculars 分数计算
- 在 HC3 测试集上评估 AUC
- 对比单模型 perplexity vs 双模型 Binoculars
- 分析不同 domain 下的检测稳定性

### 预期结果

| 方法 | 需要训练? | 预期 AUC | 跨域稳定性 |
|---|---|---|---|
| gpt2\_perplexity (你现有) | 否 | 0.9912 (单特征) | 中等 |
| Binoculars (GPT-2 对) | 否 | ~0.98-0.99 | 较好 |
| Binoculars (7B 对) | 否 | ~0.99+ | 好 |

---

## 三种方法对比总结

| 维度 | DetectGPT | RoBERTa Fine-tune | Binoculars |
|---|---|---|---|
| 需要训练数据 | 否 | 是 | 否 |
| 需要 GPU | 是 | 是 | 是 |
| 可解释性 | 中（概率曲率） | 低（黑盒） | 中（比值直觉） |
| 对新模型的泛化 | 好（零样本） | 差（需重训） | 好（零样本） |
| 速度 | 慢 / 中 (Fast) | 推理快 | 中等 |
| 预期 HC3 AUC | ~0.95-0.98 | ~0.999 | ~0.98-0.99 |
| 学术价值 | ICML 2023 | 标准 baseline | ICML 2024 |

## 实现顺序

建议按以下顺序实现（由易到难）：

1. **RoBERTa Fine-tune**（最标准，HuggingFace Trainer 几行代码）
2. **Binoculars**（只需两次 forward pass，逻辑简单）
3. **DetectGPT / Fast-DetectGPT**（需要扰动生成，逻辑最复杂）

## 最终对比表（目标）

实现完成后，README 中将展示：

| Method | Type | Training | AUC | Accuracy | Short-text AUC |
|---|---|---|---|---|---|
| XGBoost (90 features) | 手工特征 | 有监督 | 0.9999 | 0.9964 | ? |
| LR (90 features) | 手工特征 | 有监督 | 0.9984 | 0.9898 | ? |
| RoBERTa fine-tune | 深度学习 | 有监督 | ? | ? | ? |
| DetectGPT | 零样本 | 无 | ? | ? | ? |
| Fast-DetectGPT | 零样本 | 无 | ? | ? | ? |
| Binoculars | 零样本 | 无 | ? | ? | ? |
| gpt2\_perplexity only | 统计 | 有监督 | 0.9912 | ? | ? |

增加 "Short-text AUC" 列（<100 词的文本子集），这是区分各方法实际价值的关键指标。

## References

1. Gehrmann, S., Strobelt, H., & Rush, A. (2019). *GLTR: Statistical Detection and Visualization of Generated Text*. ACL Demo.
2. Mitchell, E., et al. (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*. ICML 2023.
3. Bao, G., et al. (2024). *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature*. ICLR 2024.
4. Hans, A., et al. (2024). *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*. ICML 2024.
5. Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*.
6. Guo, B., et al. (2023). *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection*. (HC3 dataset paper)
7. Verma, V., et al. (2024). *Ghostbuster: Detecting Text Ghostwritten by Large Language Models*.
8. Hu, X., et al. (2024). *RADAR: Robust AI-Text Detection via Adversarial Learning*.
