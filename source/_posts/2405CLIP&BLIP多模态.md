---
title: 多模态模型整理：CLIP 与 BLIP 系列（CLIP / ALBEF / BLIP / BLIP-2 / InstructBLIP / BLIP-3）
date: 2024-05-24
categories:
  - 学AI/DS
tags:
  - 多模态
  - VLM
  - CV
  - LLM
desc: 从“多模态是什么、解决什么任务”讲起，按 CLIP → ALBEF → BLIP → BLIP-2 → InstructBLIP → BLIP-3 的脉络，梳理每个模型的动机、结构、损失、代码、实验结论与优缺点，并补充 CLIP 家族改进（SigLIP/CoCa/EVA-CLIP 等）、应用生态、连接器对比、已知缺陷与多篇多模态综述的框架。
---

本文以知乎文章《BLIP系列——BLIP、BLIP-2、InstructBLIP、BLIP-3》为线索，结合各论文原文与几篇多模态综述，把 **CLIP 与 BLIP 系列**整理成一份从零讲起的笔记。目标是：**外行能看懂“它们在干什么”，内行能快速回想起“它们怎么做、为什么这样做、坑在哪里”。**

- 参考
    - <a href="https://zhuanlan.zhihu.com/p/28392731664">`知乎-糖葫芦：BLIP系列——BLIP、BLIP-2、InstructBLIP、BLIP-3`</a>

**核心论文：**
- <a href="https://arxiv.org/abs/2103.00020">CLIP: Learning Transferable Visual Models From Natural Language Supervision</a>（OpenAI，ICML 2021，<a href="https://github.com/openai/CLIP">GitHub</a>）
- <a href="https://arxiv.org/abs/2107.07651">ALBEF: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation</a>（Salesforce，NeurIPS 2021）
- <a href="https://arxiv.org/abs/2201.12086">BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation</a>（Salesforce，ICML 2022）
- <a href="https://arxiv.org/abs/2301.12597">BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models</a>（Salesforce，ICML 2023）
- <a href="https://arxiv.org/abs/2305.06500">InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning</a>（Salesforce，NeurIPS 2023）
- <a href="https://arxiv.org/abs/2408.08872">xGen-MM (BLIP-3): A Family of Open Large Multimodal Models</a>（Salesforce，2024.08）
- 代码库：<a href="https://github.com/salesforce/LAVIS">salesforce/LAVIS</a>（BLIP / BLIP-2 / InstructBLIP 官方实现）、<a href="https://github.com/mlfoundations/open_clip">OpenCLIP</a>、HuggingFace `transformers`

**综述（用来搭框架、补全视角）：**
- <a href="https://arxiv.org/abs/1705.09406">Multimodal Machine Learning: A Survey and Taxonomy</a>（Baltrušaitis et al., TPAMI 2019）——多模态五大挑战的经典分类
- <a href="https://arxiv.org/abs/2209.03430">Foundations and Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions</a>（Liang, Zadeh, Morency, ACM CSUR 2024）
- <a href="https://arxiv.org/abs/2202.10936">A Survey of Vision-Language Pre-Trained Models</a>（Du et al., IJCAI 2022）
- <a href="https://arxiv.org/abs/2210.09263">Vision-Language Pre-training: Basics, Recent Advances, and Future Trends</a>（Gan et al., 微软, 2022）
- <a href="https://arxiv.org/abs/2304.00685">Vision-Language Models for Vision Tasks: A Survey</a>（Zhang et al., TPAMI 2024）
- <a href="https://arxiv.org/abs/2306.13549">A Survey on Multimodal Large Language Models</a>（Yin et al., 2023）
- <a href="https://arxiv.org/abs/2309.10020">Multimodal Foundation Models: From Specialists to General-Purpose Assistants</a>（Li et al., 微软, 2023）
- <a href="https://arxiv.org/abs/2401.13601">MM-LLMs: Recent Advances in MultiModal Large Language Models</a>（Zhang et al., ACL Findings 2024）
- <a href="https://arxiv.org/abs/2405.17247">An Introduction to Vision-Language Modeling</a>（Bordes et al., Meta, 2024）

**补充论文（各节给链接）：** ALIGN、FILIP、LiT、FLIP、CoCa、SigLIP、EVA-CLIP、MetaCLIP、OpenCLIP Scaling Laws、Flamingo、LLaVA、MiniGPT-4、CoOp、CLIP-Adapter、Tip-Adapter、WiSE-FT、Modality Gap、ARO、MMVP 等。

<br>

## 0. 一页纸总览（先看结论）

### 0.1 一句话讲清每个模型

| 模型 | 时间 | 一句话 | 关键词 |
|---|---|---|---|
| **CLIP** | 2021.02 | 用 4 亿图文对做“图文配对”对比学习，得到一个**图像和文字在同一空间**的双塔模型，能零样本分类 | 双塔、对比学习、零样本 |
| **ALBEF** | 2021.07 | 先用对比学习把图文**对齐**，再用交叉注意力**融合**；用动量蒸馏对抗噪声数据 | Align before Fuse、ITC/ITM/MLM、动量蒸馏 |
| **BLIP** | 2022.01 | 一个模型同时会“理解”和“生成”（MED）；再用自己给网图**重写/过滤**标注（CapFilt） | MED、CapFilt、数据自举 |
| **BLIP-2** | 2023.01 | 冻结现成的视觉大模型和语言大模型，只训练一个轻量的 **Q-Former** 当“翻译官” | 冻结、Q-Former、两阶段 |
| **InstructBLIP** | 2023.05 | 在 BLIP-2 上做**指令微调**，Q-Former 也看指令，按需提取图像信息 | 指令微调、instruction-aware |
| **BLIP-3 (xGen-MM)** | 2024.08 | 扔掉复杂的 Q-Former 多任务训练，换成简单的 token sampler + 统一的 LM 损失，靠**大规模交错数据**取胜 | 简化、交错数据、DPO |

### 0.2 演化主线（三条线索）

$$\underbrace{\text{CLIP}}_{\text{双塔对齐}} \rightarrow \underbrace{\text{ALBEF}}_{\text{先对齐再融合}} \rightarrow \underbrace{\text{BLIP}}_{\text{理解+生成+洗数据}} \rightarrow \underbrace{\text{BLIP-2}}_{\text{冻结大模型+桥接}} \rightarrow \underbrace{\text{InstructBLIP}}_{\text{指令对齐}} \rightarrow \underbrace{\text{BLIP-3}}_{\text{简化+数据为王}}$$

1. **架构线**：双塔（各编各的）→ 融合编码器（互相看）→ 编解码统一 → **以 LLM 为中心**（视觉被“翻译”成 LLM 能读的 token）。
2. **数据线**：直接用噪声网图 → 动量蒸馏“容忍”噪声 → 用模型自己“清洗”数据（CapFilt）→ 合成稠密描述 + 万亿级图文交错数据。
3. **训练线**：端到端全量训练 → 冻结单模态大模型、只训桥接层 → 指令微调 → SFT + DPO 偏好对齐。

> 知乎原文的总结很到位：从 BLIP-2 到 BLIP-3，架构反而**越来越简单**——“或许正确的方法往往是简单的”。业界最终收敛到 **“视觉编码器 + 连接器（MLP / resampler）+ LLM”** 这一形态。

<br>

## 1. 预备知识：给外行的多模态入门

### 1.1 什么是“模态”与“多模态”

- **模态（modality）**：信息的一种存在形式。文字、图片、语音、视频、点云、表格……各是一种模态。
- **多模态学习**：让模型同时处理、并在多种模态之间建立联系。人看到“一只猫的照片”和读到“一只猫”时，脑中激活的是同一个概念——多模态模型想做的就是这件事。
- 本文主要讨论 **视觉-语言（Vision-Language, VL）** 这一对最常见的模态，对应的模型常叫 **VLP（Vision-Language Pre-training）模型** 或 **VLM（Vision-Language Model）**。接入大语言模型之后又叫 **MLLM / LMM（Multimodal Large Language Model）**。

### 1.2 多模态要攻克的五大挑战（Baltrušaitis 综述的分类）

| 挑战 | 白话解释 | 本文对应 |
|---|---|---|
| **表示（Representation）** | 怎么把图像和文字变成计算机能比较的向量 | CLIP 的联合嵌入空间 |
| **翻译（Translation）** | 从一种模态生成另一种模态 | BLIP 的看图说话、文生图 |
| **对齐（Alignment）** | 找到图中区域和句中词语的对应关系 | ITC / ITM、FILIP 的 token 级对齐 |
| **融合（Fusion）** | 把两种信息合起来做判断 | ALBEF/BLIP 的交叉注意力 |
| **协同学习（Co-learning）** | 用一种模态的知识帮助另一种模态 | BLIP-2 用冻结的 LLM 知识帮视觉问答 |

Liang 等人的新综述把它扩展为六个：Representation、Alignment、Reasoning、Generation、Transference、Quantification，其中 Quantification（量化理解模态间的交互、偏差）对应本文第 8 章的“缺陷分析”。

### 1.3 视觉-语言模型能做哪些任务

| 类别 | 任务 | 输入 → 输出 | 例子 | 擅长的模型 |
|---|---|---|---|---|
| 理解·检索 | 图文检索（ITR/TIR） | 文字 → 找图 / 图 → 找文字 | 搜“海边日落” | CLIP、BLIP、BLIP-2 Q-Former |
| 理解·分类 | 零样本图像分类 | 图 + 类别名 → 类别 | 不训练就认出 1000 类 | CLIP |
| 理解·匹配 | 图文匹配（ITM） | 图 + 文 → 是否匹配 | 判断标题是否党 | ALBEF、BLIP |
| 理解·问答 | 视觉问答（VQA） | 图 + 问题 → 答案 | “图里有几只猫？” | BLIP、BLIP-2、InstructBLIP |
| 理解·推理 | 视觉推理（NLVR²、VE） | 图 + 陈述 → 真/假 | “左图比右图狗多” | ALBEF、BLIP |
| 生成 | 图像描述（Captioning） | 图 → 一句话 | “一只橘猫趴在键盘上” | BLIP、BLIP-2 |
| 生成 | 文生图 | 文 → 图 | Stable Diffusion | CLIP 文本编码器做条件 |
| 定位 | 视觉定位 / 开放词汇检测 | 图 + 文 → 框 / 掩码 | “框出穿红衣服的人” | GLIP、OWL-ViT（基于 CLIP） |
| 对话 | 多模态对话 / 指令跟随 | 图 + 指令 → 自由回答 | “根据这张图写一首诗” | InstructBLIP、BLIP-3、LLaVA |

**理解（understanding）** 类任务输出“分数/类别”，适合**编码器**；**生成（generation）** 类任务输出“一段文字”，适合**解码器**。这一点是理解 BLIP 设计动机的钥匙。

### 1.4 必备积木（五分钟版）

1. **Transformer 与自注意力（Self-Attention）**：序列中每个元素“看”所有其他元素，按相关性加权汇总：

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

2. **交叉注意力（Cross-Attention）**：$Q$ 来自一种模态（如文字），$K,V$ 来自另一种模态（如图像）——“文字去图像里查资料”。这是**多模态融合的核心操作**。
3. **ViT（Vision Transformer）**：把图像切成 $16\times16$ 的小块（patch），每块当作一个“词”，加一个 `[CLS]` token 汇总全局信息。$224\times224$ 的图像在 ViT-B/16 下变成 $14\times14=196$ 个 patch + 1 个 `[CLS]`。
4. **BERT vs GPT**：BERT 是**双向**编码器（每个词能看前后文），适合理解；GPT 是**单向/因果**解码器（只能看左边），适合生成。区别只在注意力掩码（mask）上。
5. **对比学习（Contrastive Learning）与 InfoNCE**：给一批正样本对，让正对相似度高、与其他样本（负样本）相似度低。本质是一个“在一批里猜哪个是我的配对”的分类问题。
6. **温度系数 $\tau$**：softmax 前除以 $\tau$。$\tau$ 越小，分布越尖锐，越关注困难负样本。

### 1.5 VLP 模型的四种架构范式

| 范式 | 结构 | 计算方式 | 优点 | 缺点 | 代表 |
|---|---|---|---|---|---|
| **双塔 / 双编码器（Dual Encoder）** | 图、文各一个编码器，最后算点积 | 图文**不交互** | 检索极快（向量可预先算好、建索引） | 细粒度理解弱 | CLIP、ALIGN |
| **融合编码器（Fusion Encoder）** | 单模态编码后再经交叉注意力深度融合 | 图文**深度交互** | 理解任务强（VQA、NLVR） | 检索要两两过模型，慢；不会生成 | ViLBERT、UNITER、ALBEF |
| **编码器-解码器（Encoder-Decoder）** | 融合后接文本解码器 | 交互 + 生成 | 能生成 | 理解+生成难兼顾 | SimVLM、BLIP（MED）、CoCa |
| **以 LLM 为中心（LLM-centric）** | 冻结视觉编码器 + 连接器 + LLM | 视觉 token 拼进 LLM 输入 | 继承 LLM 的知识和推理能力，能对话 | 幻觉、依赖 LLM、成本高 | BLIP-2、InstructBLIP、LLaVA、BLIP-3 |

Bordes 等人的 Meta 入门综述按**训练目标**分四类：对比式（CLIP）、掩码式（FLAVA、MaskVLM）、生成式（CoCa、Chameleon）、基于预训练骨干式（Frozen、BLIP-2、LLaVA、MiniGPT-4）。两种分法互为补充。

### 1.6 时间线

```
2019  ViLBERT / LXMERT / UNITER  —— 依赖目标检测器提取区域特征（慢、贵）
2020  ViT 出现                   —— 视觉也能用 Transformer，摆脱检测器
2021.02  CLIP / ALIGN            —— 海量网图 + 对比学习，零样本时代开始
2021.07  ALBEF                   —— 先对齐再融合，动量蒸馏
2022.01  BLIP                    —— 理解+生成统一，CapFilt 洗数据
2022.04  Flamingo                —— 冻结 LLM + 交叉注意力，少样本能力
2022.11  ChatGPT                 —— LLM 爆发
2023.01  BLIP-2                  —— Q-Former 桥接冻结的 ViT 与 LLM
2023.03  SigLIP                  —— sigmoid 损失替代 softmax 对比损失
2023.04  LLaVA / MiniGPT-4       —— 线性层/MLP 连接 + 视觉指令数据
2023.05  InstructBLIP            —— 指令感知的 Q-Former
2024.08  BLIP-3 (xGen-MM)        —— 简化架构 + 交错数据 + DPO
2025.05  BLIP3-o                 —— 理解与生成统一（扩散生成 CLIP 特征）
```

<br>

## 2. CLIP：用自然语言监督学习视觉

### 2.1 动机：传统视觉模型的天花板

- 传统图像分类在 ImageNet 这样的**固定类别**上训练：1000 类就是 1000 个输出神经元。想识别第 1001 类？重新标数据、重新训练。
- 人工标注贵：ImageNet 1400 万张图花了大量众包人力。
- 而互联网上有**海量自带文字说明的图片**（alt-text、标题）。文字比类别标签信息丰富得多（“一只在雪地里奔跑的哈士奇” vs “狗”）。
- CLIP 的核心想法：**不预测类别，而是预测“哪段文字和这张图配对”**。这样类别空间就变成了“任意自然语言”，天然支持零样本。

### 2.2 为什么选“对比”而不是“生成”

论文做了一个关键实验：在同样算力下比较三种预训练目标的 ImageNet 零样本效率：
1. 用 Transformer 语言模型**逐字生成**图片的描述——最慢；
2. 预测描述的**词袋（bag-of-words）**——效率约为 1 的 3 倍；
3. **对比学习**，只判断“图文是否配对”——效率再提高约 4 倍。

原因：生成任务要求**逐字准确**地复述描述，而同一张图可以有无数种合理描述，这个目标太难、太“浪费”；对比目标只要求“比别的文字更配”，更容易学到语义。

### 2.3 数据：WIT-400M

- 自建 **WIT（WebImageText）**：4 亿图文对。
- 构造方式：用 50 万个查询词（来自维基百科高频词等）搜索，每个查询最多保留 2 万对，做了类别均衡。
- 数据集未公开，后来社区用 **LAION-400M / LAION-5B** 复现（OpenCLIP）。

### 2.4 架构：双塔

```
   图像 ──► Image Encoder (ResNet / ViT) ──► I_f ──► W_i 线性投影 ──► L2 归一化 ──► I_e ─┐
                                                                                       ├─► 余弦相似度矩阵 N×N
   文本 ──► Text Encoder (Transformer)    ──► T_f ──► W_t 线性投影 ──► L2 归一化 ──► T_e ─┘
```

- **图像编码器**：试了 5 个 ResNet（RN50、RN101、RN50x4、RN50x16、RN50x64，后者按 EfficientNet 方式等比放大）和 3 个 ViT（ViT-B/32、ViT-B/16、ViT-L/14），最好的是 **ViT-L/14@336px**（在 336 分辨率上额外训练 1 个 epoch）。ResNet 版本把全局平均池化换成了注意力池化。
- **文本编码器**：12 层、宽 512、8 头的 Transformer（约 63M 参数），BPE 词表 49,152，最大长度 76。取 `[EOS]` 位置的输出作为句向量。注意文本侧用的是**因果掩码**（为了保留以后加语言建模的可能），这一点常被忽略。
- **投影层**：只用线性投影（没用非线性 MLP），映射到共享的嵌入空间。
- 两个编码器都**从零训练**，不用 ImageNet 预训练权重。

### 2.5 损失函数：对称 InfoNCE

一个 batch 有 $N$ 对图文 $\{(x_i, t_i)\}_{i=1}^N$。记归一化后的嵌入为 $\mathbf{v}_i, \mathbf{u}_i$，相似度 $s_{ij}=\mathbf{v}_i^\top \mathbf{u}_j / \tau$。

- **图 → 文**：对第 $i$ 张图，在 $N$ 段文字中“选出”自己的配对：

$$\mathcal{L}_{i2t}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ij})}$$

- **文 → 图**：对称地：

$$\mathcal{L}_{t2i}=-\frac{1}{N}\sum_{j=1}^{N}\log\frac{\exp(s_{jj})}{\sum_{i=1}^{N}\exp(s_{ij})}$$

- **总损失**：$\mathcal{L}=\frac{1}{2}(\mathcal{L}_{i2t}+\mathcal{L}_{t2i})$

直观理解：$N\times N$ 的相似度矩阵里，**对角线是正样本（$N$ 个），其余是负样本（$N^2-N$ 个）**。训练就是让对角线变亮、其余变暗。

- **温度 $\tau$ 是可学习参数**：初始化为 0.07，用 $\log(1/\tau)$ 参数化，并截断使 logit 缩放因子不超过 100，防止训练不稳定。
- **超大 batch**：32,768。batch 越大，每个正样本面对的负样本越多，对比学习越有效——这也是 CLIP 贵的原因之一（相似度矩阵 $N^2$，需要跨 GPU 收集所有特征）。
- 训练 32 个 epoch。最大的 RN50x64 在 592 张 V100 上训练 18 天，ViT-L/14 在 256 张 V100 上训练 12 天。

### 2.6 代码：训练核心（PyTorch 版）

论文 Figure 3 给出了 numpy 风格伪代码，下面是等价的 PyTorch 写法：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, d_img, d_txt, d_emb=512):
        super().__init__()
        self.image_encoder = image_encoder          # ResNet / ViT，输出 [B, d_img]
        self.text_encoder = text_encoder            # Transformer，取 [EOS] 输出 [B, d_txt]
        self.W_i = nn.Linear(d_img, d_emb, bias=False)
        self.W_t = nn.Linear(d_txt, d_emb, bias=False)
        # 可学习温度：logit_scale = log(1/τ)，τ 初始化 0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, texts):
        I_e = F.normalize(self.W_i(self.image_encoder(images)), dim=-1)  # [B, d_emb]
        T_e = F.normalize(self.W_t(self.text_encoder(texts)), dim=-1)    # [B, d_emb]
        scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * I_e @ T_e.t()                         # [B, B]
        return logits_per_image

def clip_loss(logits_per_image):
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)       # 每行：图找文
    loss_t2i = F.cross_entropy(logits_per_image.t(), labels)   # 每列：文找图
    return (loss_i2t + loss_t2i) / 2
```

要点：对比损失**就是两个交叉熵**，标签是对角线索引 `arange(B)`。分布式训练时要用 `all_gather` 收集所有卡的特征，否则每张卡的“负样本”只有本卡的 batch。

### 2.7 零样本分类：把分类变成检索

```
类别名 ──► 套模板 "a photo of a {label}." ──► Text Encoder ──► K 个类别向量（相当于分类器权重）
测试图 ──► Image Encoder ──► 图像向量 ──► 与 K 个类别向量算余弦相似度 ──► argmax
```

```python
# HuggingFace transformers 版零样本分类
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
labels = ["cat", "dog", "car"]
prompts = [f"a photo of a {c}." for c in labels]

inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    probs = model(**inputs).logits_per_image.softmax(dim=-1)   # [1, 3]
print(dict(zip(labels, probs[0].tolist())))
```

**Prompt Engineering 与集成**：
- 为什么要模板？预训练文本几乎都是句子，单个词“crane”有歧义（鹤 / 起重机）。用 “a photo of a {label}.” 在 ImageNet 上就提升约 1.3 个点。
- 针对数据集定制：如宠物数据集用 “a photo of a {label}, a type of pet.”；卫星图用 “a satellite photo of a {label}.”。
- **Prompt Ensemble**：对每个类别用 80 个模板（“a bad photo of a …”“a sculpture of a …”等）分别编码后**在嵌入空间取平均**，ImageNet 上再提升约 3.5 个点。因为是先平均再存，推理成本不变。

### 2.8 关键实验结论

1. **零样本 ImageNet 76.2%**（ViT-L/14@336px），追平了在 128 万张标注图上训练的 ResNet-50——而 CLIP 一张 ImageNet 训练图都没用。
2. 在 27 个数据集上，零样本 CLIP 在 **16 个** 上超过“ResNet-50 特征 + 线性分类器”的全监督基线。
3. **线性探测（linear probe）**：冻结 CLIP 图像编码器只训线性层，表征质量超过当时最好的 ImageNet 模型（Noisy Student EfficientNet-L2）。
4. **分布外鲁棒性**：在 ImageNet-R（艺术画）、ImageNet-Sketch（素描）、ObjectNet、ImageNet-A（对抗样本）等分布偏移数据上，零样本 CLIP 的鲁棒性远好于 ImageNet 监督模型，“有效鲁棒性”差距缩小最多 75%。但**一旦在 ImageNet 上微调**，ImageNet 精度提升而分布外精度下降——这引出了后来的 WiSE-FT。
5. **少样本反直觉现象**：零样本 CLIP ≈ 16-shot 线性探测；但 1-shot、2-shot 线性探测反而**不如零样本**——因为少量样本学出来的分类器丢掉了文本先验。

### 2.9 CLIP 的局限（论文自己承认的 + 后续发现的）

| 局限 | 表现 | 原因分析 |
|---|---|---|
| 细粒度分类弱 | 车型、花种、飞机型号 | 网图描述很少精确到型号 |
| 抽象/系统性任务弱 | 数物体个数、判断距离 | 对比目标只学“整体语义” |
| 真正分布外仍差 | **MNIST 手写数字只有 88%**，不如像素上的逻辑回归 | 预训练数据里几乎没有手写数字 |
| 不能生成 | 只能从给定选项中挑 | 双塔没有解码器 |
| 文字攻击（Typographic Attack） | 苹果上贴一张写着 “iPod” 的纸，就被识别成 iPod | CLIP 学会了 OCR，文字信号太强 |
| 偏见 | 对某些人群的有害关联 | 网络数据自带社会偏见 |
| 组合理解差 | “马骑人”和“人骑马”难区分 | 见 8.5 节 bag-of-words 问题 |
| 训练成本高 | 超大 batch、海量数据 | 对比学习依赖大量负样本 |

### 2.10 CLIP 家族：后续改进

| 模型 | 改了什么 | 关键想法 | 效果/意义 |
|---|---|---|---|
| <a href="https://arxiv.org/abs/2102.05918">ALIGN</a>（Google, 2021） | 数据 | 18 亿**不过滤**的噪声 alt-text，EfficientNet + BERT | 证明“规模能弥补噪声” |
| <a href="https://arxiv.org/abs/2111.07783">FILIP</a>（2021） | 相似度 | **token 级后期交互**：每个 patch 找最相似的词，再平均 | 细粒度对齐更好，仍可预先算特征 |
| <a href="https://arxiv.org/abs/2111.07991">LiT</a>（Google, 2021） | 训练策略 | **锁住**预训练好的图像塔，只训文本塔 | 数据效率更高，零样本 ImageNet 84.5% |
| <a href="https://arxiv.org/abs/2212.00794">FLIP</a>（Meta, 2022） | 效率 | 训练时**随机遮掉 50%–75% 的图像 patch** | 速度提升 2–3 倍，相同时间可用更大 batch |
| <a href="https://arxiv.org/abs/2205.01917">CoCa</a>（Google, 2022） | 目标 | 对比损失 + **描述生成损失**，文本解码器前半是单模态、后半交叉注意力 | 一个模型兼顾检索与生成，ImageNet 零样本 86.3% |
| <a href="https://arxiv.org/abs/2212.07143">OpenCLIP Scaling</a>（2022） | 复现 | 在 LAION-5B 上系统研究 scaling law | 开源权重，社区主力 |
| <a href="https://arxiv.org/abs/2303.15389">EVA-CLIP</a>（BAAI, 2023） | 初始化与训练技巧 | 用 MIM 预训练的 EVA 初始化，LAMB 优化器，FLIP 式遮挡 | BLIP-2 / InstructBLIP 使用的 ViT-g 就来自它 |
| <a href="https://arxiv.org/abs/2303.15343">SigLIP</a>（Google, 2023） | 损失 | **sigmoid 二分类**替代 softmax 对比 | 不需要全局归一化，小 batch 也好用；BLIP-3、PaliGemma 等的视觉塔 |
| <a href="https://arxiv.org/abs/2309.16671">MetaCLIP</a>（Meta, 2023） | 数据 | 公开 CLIP 的数据“策展”方法：元数据匹配 + 平衡采样 | 证明数据分布比模型更关键 |
| <a href="https://arxiv.org/abs/2403.15378">Long-CLIP</a>（2024） | 文本长度 | 位置编码插值，突破 77 token 限制 | 支持长描述 |
| <a href="https://arxiv.org/abs/2211.01335">Chinese-CLIP</a>（阿里, 2022） | 语言 | 中文图文对 + 两阶段训练 | 中文场景首选之一 |

**SigLIP 损失**（值得单独记住）：把每一对 $(i,j)$ 都当成独立的二分类，正对标签 $z_{ij}=1$，负对 $z_{ij}=-1$：

$$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\log\sigma\big(z_{ij}\cdot(t\,\mathbf{v}_i^\top\mathbf{u}_j+b)\big)$$

其中 $t$ 是可学习温度、$b$ 是可学习偏置（初始化为负值，因为负对远多于正对）。不需要像 softmax 那样对整行求和归一化，所以**不需要所有卡的特征都同时在一张卡上**，可以分块计算，显存友好。

```python
def siglip_loss(img_emb, txt_emb, t, b):
    logits = t * img_emb @ txt_emb.t() + b               # [B, B]
    labels = 2 * torch.eye(logits.size(0), device=logits.device) - 1   # 对角 +1，其余 -1
    return -F.logsigmoid(labels * logits).sum() / logits.size(0)
```

### 2.11 CLIP 的应用生态：它为什么这么重要

CLIP 的真正价值不在于零样本分类本身，而在于它提供了一个**通用的“图文共同语言”**，成为大量系统的基础组件：

1. **文生图的“文字理解器”**：Stable Diffusion 1.x 用 CLIP ViT-L/14 的文本编码器做条件；SD 2.x 换成 OpenCLIP ViT-H；DALL·E 2（unCLIP，<a href="https://arxiv.org/abs/2204.06125">论文</a>）直接在 CLIP 嵌入空间里先生成图像嵌入，再解码成图。
2. **多模态大模型的“眼睛”**：LLaVA 用 CLIP ViT-L/14，BLIP-2 用 EVA-CLIP ViT-g，BLIP-3 用 SigLIP。几乎所有 MLLM 的视觉编码器都来自 CLIP 系。
3. **开放词汇检测/分割**：ViLD、<a href="https://arxiv.org/abs/2205.06230">OWL-ViT</a>、<a href="https://arxiv.org/abs/2112.03857">GLIP</a>、LSeg、CLIPSeg 用 CLIP 的文本向量替代固定类别头，实现“输入任意文字就能检测/分割”。
4. **评价指标**：<a href="https://arxiv.org/abs/2104.08718">CLIPScore</a> 用图文相似度评估描述质量或文生图的图文一致性，不需要参考答案。
5. **数据过滤**：LAION 用 CLIP 相似度阈值过滤网图；BLIP-2 用 CLIP 给合成描述排序。
6. **视频与检索**：CLIP4Clip 逐帧编码后时序聚合，做视频-文本检索；以图搜图、以文搜图的工业检索系统。
7. **高效迁移（不改主干）**：
   - <a href="https://arxiv.org/abs/2109.01134">CoOp</a>：把手写模板换成**可学习的上下文向量**（prompt learning），少样本下比手写模板强很多；CoCoOp 让上下文依赖图像以改善泛化。
   - <a href="https://arxiv.org/abs/2110.04544">CLIP-Adapter</a>：在特征后加小 MLP 残差适配器。
   - <a href="https://arxiv.org/abs/2111.03930">Tip-Adapter</a>：把少样本训练集特征做成“键值缓存”，**无需训练**。
   - <a href="https://arxiv.org/abs/2109.01903">WiSE-FT</a>：把零样本权重和微调后权重**线性插值**，同时保住分布内精度和分布外鲁棒性。

<br>

## 3. ALBEF：从“对齐”走向“融合”（BLIP 的直接前身）

> 知乎原文没有展开 ALBEF，但它是理解 BLIP 的关键：BLIP 的 ITC/ITM、难负样本挖掘、动量编码器都直接继承自 ALBEF，作者也基本是同一批人（Junnan Li 等）。

### 3.1 动机

- CLIP 双塔检索快，但**图文没有交互**，做 VQA、推理这类需要细粒度理解的任务不行。
- 之前的融合编码器（UNITER 等）依赖**目标检测器**提取区域特征：贵、需要框标注、分辨率受限；而且图像特征和文字特征**未对齐**就直接丢进融合层，融合层很难学。
- 网图噪声大：很多 alt-text 和图片根本不相关，用 one-hot 标签硬学会过拟合噪声。

**ALBEF = Align BEfore Fuse**：先用对比学习把两种特征**对齐**到同一空间，再交给交叉注意力**融合**。

### 3.2 架构

```
图像 ──► ViT-B/16（12 层）──────────────────────────────┐
                                                       │ 交叉注意力
文本 ──► BERT 前 6 层（文本编码器）──► BERT 后 6 层 + Cross-Attn（多模态编码器）──► ITM / MLM 头
              │                      │
              └──────── ITC ─────────┘（在融合之前对齐）
```

把 BERT-base 拆成两半：前 6 层只处理文本，后 6 层每层加入交叉注意力看图像。

### 3.3 三个预训练目标（BLIP 系列的“标准三件套”雏形）

1. **ITC（Image-Text Contrastive）**：与 CLIP 相同，但借鉴 MoCo 用**动量编码器 + 队列**（队列长 65,536）存储更多负样本，不再依赖超大 batch。
2. **ITM（Image-Text Matching）**：多模态编码器的 `[CLS]` 输出接二分类头，判断图文是否匹配。关键是**难负样本挖掘（hard negative mining）**：对每张图，按 ITC 相似度**采样**一个“很像但不是”的文本作负样本（相似度越高越容易被采到），否则随机负样本太容易，模型学不到东西。
3. **MLM（Masked Language Modeling）**：随机遮住 15% 的词，让模型结合图像和上下文猜出来。

### 3.4 动量蒸馏（Momentum Distillation, MoD）

- 维护一个**动量模型**：参数是主模型的指数滑动平均（EMA），$\theta_m \leftarrow m\,\theta_m + (1-m)\,\theta$，$m=0.995$。
- 动量模型对 ITC 和 MLM 产生**软的伪标签**（概率分布），主模型同时拟合真实 one-hot 标签和伪标签：

$$\mathcal{L}_{itc}^{mod}=(1-\alpha)\mathcal{L}_{itc}+\alpha\cdot\text{KL}\big(q^{m}\,\|\,p\big),\quad \alpha=0.4$$

- 直观：如果某段“负样本”文字其实也描述了这张图（网图常见），one-hot 标签会惩罚它，而动量模型给的软标签会给它一些概率，**减少对噪声的过拟合**。
- 论文还从**互信息最大化**角度解释：ITC、MLM 和 MoD 都可以看作为同一图文对生成不同“视角（view）”并最大化视角间互信息。

### 3.5 代码：难负样本挖掘

```python
# sim_i2t: [B, B] 图到文的 ITC 相似度（已除以温度）
with torch.no_grad():
    weights_i2t = F.softmax(sim_i2t, dim=1)
    weights_i2t.fill_diagonal_(0)                     # 排除正样本自己
# 对每张图，按相似度概率采样一个"像但不对"的文本
neg_idx = torch.multinomial(weights_i2t, 1).squeeze(1)
text_neg = text_embeds[neg_idx]
# 正样本对 label=1，(图, 难负文本) 和 (难负图, 文本) label=0，送入 ITM 头做二分类
```

### 3.6 小结

- 14M 图像预训练就在检索、VQA、NLVR² 上超过用更多数据的方法；推理比检测器方案快很多。
- **遗留问题**：ALBEF 是纯编码器，**不能生成文字**（不能做 captioning）；对噪声数据只是“容忍”而非“清洗”。这两点正是 BLIP 的出发点。

<br>

## 4. BLIP：理解与生成的统一 + 数据自举

### 4.1 动机：两个层面的不足

**模型层面**（知乎原文概括为“既要又要”）：
- 编码器模型（CLIP、ALBEF）擅长理解任务，但不能直接生成；
- 编码器-解码器模型（SimVLM）能生成，但在检索等理解任务上表现不佳。
- → BLIP 想用**一个预训练模型**同时做好两类任务。

**数据层面**：
- 网图图文对噪声大，扩大规模虽有收益，但**噪声是次优的监督信号**。
- → BLIP 提出 **CapFilt**：用模型自己给网图生成更好的描述，并过滤掉噪声描述，即“**自举（Bootstrapping）**”——这也是名字里 B 的来源。

### 4.2 MED：多模态混合编码器-解码器

MED（Multimodal mixture of Encoder-Decoder）可以在三种模式下工作，知乎原文称之为“大杂烩”。加上图像编码器一共四个部分：

| 模块 | 输入 | 注意力 | 训练目标 | 作用 |
|---|---|---|---|---|
| ① 图像编码器 | 图像 | ViT 双向自注意力 | 与 ②③④ 联合训练 | 提取视觉特征（ImageNet 预训练 ViT 初始化） |
| ② 单模态文本编码器 | `[CLS]` + 文本 | 双向 SA | **ITC** | 和 CLIP 一样做对齐，用于快速检索 |
| ③ 图像引导的文本编码器 | `[Encode]` + 文本 | 双向 SA + **交叉注意力（CA）** 看图 | **ITM** | 细粒度判断图文是否匹配 |
| ④ 图像引导的文本解码器 | `[Decode]` + 文本 | **因果 SA** + CA 看图 | **LM** | 看图生成描述 |

```
                     ┌──────────────── ITC ───────────────┐
图像 ──► ViT ──► 图像特征                                     │
                     │                                       │
                     │       ② Text Encoder  [CLS] + text  ──┘  (Bi-SA → FFN)
                     ├──CA─► ③ Image-grounded Encoder [Encode] + text ──► ITM (Bi-SA → CA → FFN)
                     └──CA─► ④ Image-grounded Decoder [Decode] + text ──► LM  (Causal-SA → CA → FFN)
```

**参数共享的设计**（论文消融的重点）：
- ②③④ 中，**除了自注意力（SA）层之外，所有参数共享**（嵌入层、交叉注意力层、FFN）。
- 为什么 SA 不共享？编码器用**双向**注意力，解码器用**因果**注意力，两种任务在 SA 层冲突最大；SA 层又最能体现“编码 vs 解码”的差异。
- 消融：全共享时 LM 和其他任务互相拖累；全不共享参数翻倍、训练变慢；**只分开 SA** 效果最好，且训练效率高。
- 每个图文对只需要过一次 ViT（最重的部分），三个文本前向共享这次图像计算。

### 4.3 三个损失

$$\mathcal{L}_{\text{BLIP}}=\mathcal{L}_{\text{ITC}}+\mathcal{L}_{\text{ITM}}+\mathcal{L}_{\text{LM}}$$

1. **ITC**：同 ALBEF，带动量编码器和软标签。
2. **ITM**：同 ALBEF，带难负样本挖掘。
3. **LM（语言建模）**：以图像为条件，自回归地最大化描述文本的似然，带 0.1 的标签平滑：

$$\mathcal{L}_{\text{LM}}=-\sum_{k=1}^{L}\log P_\theta(w_k\mid w_{<k},\,I)$$

与 ALBEF 的区别：**用 LM 替代了 MLM**，于是模型具备了生成能力。

### 4.4 CapFilt：用模型清洗数据

```
网络图文对 {(I_w, T_w)}  +  人工标注 {(I_h, T_h)}（COCO）
                │
  1. 用全部数据预训练一个 MED
  2. 从它初始化两个模块，分别在 COCO 上轻量微调：
       Captioner = 图像引导的文本解码器（LM 微调）
       Filter    = 图像引导的文本编码器（ITC + ITM 微调）
                │
  3. Captioner 为每张网图生成一条合成描述 T_s
  4. Filter 用 ITM 判断：原始网络描述 T_w 和合成描述 T_s，不匹配的丢掉
                │
  新数据集 = 过滤后的 {(I_w, T_w)} + 过滤后的 {(I_w, T_s)} + {(I_h, T_h)}
                │
  5. 用新数据集**从头预训练**一个新的 BLIP
```

两个关键发现：
- **生成时用核采样（nucleus sampling, p=0.9）而不是束搜索（beam search）**：束搜索生成的句子“安全但无聊”，都是高频句式；核采样更多样，虽然噪声更大（filter 过滤比例更高），但带来的**新信息**更多，下游效果更好。“多样性 > 准确性”。
- **Captioner 和 Filter 不共享参数**：否则会产生“确认偏差”——filter 倾向于放过 captioner 自己生成的描述。

Captioner 和 Filter 单独用都有提升，两者结合提升最大；而且数据集越大、模型越大，CapFilt 收益越明显。

### 4.5 训练细节

- 预训练数据：**14M** 版本（COCO + Visual Genome + CC3M + CC12M + SBU），以及加入 LAION 1.15 亿图得到的 **129M** 版本。
- 图像编码器：ViT-B/16、ViT-L/16（ImageNet 预训练初始化）；文本侧 BERT-base 初始化。
- 2 个节点 × 16 张 GPU；预训练 20 epoch；batch size 2880（ViT-B）/ 2400（ViT-L）；AdamW，峰值学习率 3e-4（ViT-B）/ 2e-4（ViT-L），余弦衰减；预训练分辨率 224，微调 384。
- 模型规模约 2.2 亿（ViT-B 版）到 5 亿级（ViT-L 版）。

### 4.6 下游任务怎么用 BLIP

| 任务 | 用哪个模块 | 做法 |
|---|---|---|
| 图文检索 | ② + ③ | 先用 ITC 相似度粗召回 top-k，再用 ITM 精排（“先快后准”） |
| 图像描述 | ④ | 以 “a picture of ” 为前缀生成 |
| VQA | ③ + ④ | 把 VQA 当**生成**任务：③ 编码图像+问题，④ 生成答案（开放式，不限定候选答案集合） |
| NLVR² | ③ 改造 | 两张图各自交叉注意力后合并 |
| 视频-文本 | 零样本迁移 | 均匀采帧，拼接帧特征当图像特征 |

### 4.7 代码：HuggingFace 推理

```python
import torch
from PIL import Image
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          BlipForImageTextRetrieval)

image = Image.open("demo.jpg").convert("RGB")

# 1) 看图说话（模块④）
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
inputs = processor(image, "a picture of", return_tensors="pt")      # 条件前缀
out = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(out[0], skip_special_tokens=True))

# 2) 图文匹配（模块③ ITM / 模块② ITC）
proc = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
itm = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
inputs = proc(image, "a woman sitting on the beach with a dog", return_tensors="pt")
with torch.no_grad():
    itm_score = itm(**inputs).itm_score.softmax(dim=-1)[:, 1]        # 匹配概率
    itc_score = itm(**inputs, use_itm_head=False).itm_score          # 余弦相似度
```

### 4.8 优缺点

**优点**：
- 一个预训练模型覆盖检索、描述、VQA、推理，开源且效果好，发布时在多项任务上 SOTA。
- CapFilt 证明了“**用模型改进数据**”的威力，这一思想后来被广泛使用（LAION-COCO、DALL·E 3 的重写描述、BLIP3-KALE）。

**缺点**（结合知乎原文的点评）：
- 本质是多个编码器/解码器的“**拼接**”，不是真正的一体化：做不同任务要取出不同模块。
- 端到端训练整个 ViT + BERT，**计算成本高**，难以扩展到更大的视觉或语言模型。
- 文本侧只是 BERT-base 规模，**没有大语言模型的知识和推理能力**，VQA 答案短、无法对话。
- CapFilt 需要“预训练 → 微调 → 生成 → 过滤 → 再预训练”，流程长。

<br>

## 5. BLIP-2：冻结大模型，只训“翻译官”

### 5.1 动机

1. **成本**：视觉和语言模型越来越大（ViT-g 十亿参数、LLM 几十上百亿），端到端联合预训练不可承受。
2. **已有资产**：已经有了很强的**冻结单模态模型**——视觉侧 EVA-CLIP，语言侧 OPT、FlanT5。能不能直接复用？
3. **难点——模态鸿沟（modality gap）**：LLM 在预训练时从未见过图像。直接把冻结 ViT 的特征丢给冻结 LLM，两者“语言不通”。而且冻结 LLM 时，只靠 LM 损失去对齐图文是不够的（论文消融证实）。
4. 背景：Flamingo（<a href="https://arxiv.org/abs/2204.14198">2022.04</a>）已经证明“冻结 LLM + 在 LLM 内部插入交叉注意力层”可行，但它要改 LLM 结构，训练参数多（800 亿参数的 Flamingo 有百亿级可训练参数）。

**BLIP-2 的方案**：在冻结的图像编码器和冻结的 LLM 之间放一个轻量的 **Q-Former（Querying Transformer，188M 参数）**，分两阶段训练：
- **阶段一：表示学习**——让 Q-Former 学会从图像中提取“和文本最相关”的视觉信息（视觉-语言对齐）；
- **阶段二：生成学习**——让 Q-Former 的输出能被 LLM “读懂”（视觉-语言生成）。

### 5.2 总体架构

```
             冻结                    可训练（188M）              冻结
图像 ──► Image Encoder ──► 图像特征 ──► Q-Former ──► Z [32×768] ──► FC ──► 软视觉提示 ──► LLM ──► 文本
       (EVA ViT-g/14 或                ▲                                  + 文本 token
        CLIP ViT-L/14)            32 个可学习 Query
```

- **图像编码器**：CLIP ViT-L/14 或 EVA-CLIP ViT-g/14，**去掉最后一层**，用倒数第二层输出（效果略好）。
- **LLM**：解码器型 OPT（2.7B / 6.7B）、编码器-解码器型 FlanT5（XL / XXL）。
- **压缩比**：ViT-L/14 输出 $257\times1024$，Q-Former 输出 $32\times768$——视觉信息被压缩成 **32 个 token**，这是一个**信息瓶颈**，迫使 Q-Former 只保留与语言最相关的信息，也大幅降低 LLM 的计算量。

### 5.3 Q-Former 的结构

Q-Former 由**两个共享自注意力层的 Transformer 子模块**组成（知乎原文记作 Q-Former-Image 和 Q-Former-Text）：

```
          32 个 Learned Queries                    文本 tokens
                  │                                    │
         ┌────────▼────────┐                 ┌─────────▼────────┐
         │  Self-Attention │◄═══ 共享参数 ═══►│  Self-Attention  │  ← 两边的 token 能否互相看见由 mask 控制
         ├─────────────────┤                 ├──────────────────┤
         │ Cross-Attention │◄── 冻结图像特征   │                  │  ← 只有 query 侧有 CA（每隔一层插入）
         ├─────────────────┤                 ├──────────────────┤
         │  Feed Forward   │                 │   Feed Forward   │
         └────────┬────────┘                 └─────────┬────────┘
             图像变换器（输出 Z）                  文本变换器（编码/解码）
```

- 用 **BERT-base** 预训练权重初始化，交叉注意力层随机初始化。
- **32 个可学习 query 向量**，每个维度 768，作为模型参数训练。
- query 通过**交叉注意力**从冻结的图像特征中“提问取信息”，通过**自注意力**和文本交互。
- query 与文本之间的交互方式，由**自注意力掩码**按任务切换——这是 Q-Former 最精巧的地方。

### 5.4 阶段一：三个任务 × 三种注意力掩码

同一个 Q-Former，同一批数据，用三种不同的掩码同时优化三个目标（这又是一次“ITC + ITM + 生成”的三件套）：

| 任务 | 掩码类型 | Query 能看 | 文本能看 | 输出怎么用 |
|---|---|---|---|---|
| **ITC** 图文对比 | **单模态掩码**（Uni-modal） | 只看 query | 只看文本 | 32 个 query 输出分别与文本 `[CLS]` 算相似度，**取最大值** |
| **ITG** 图像引导文本生成 | **多模态因果掩码**（Multi-modal Causal） | 只看 query（不看文本） | 所有 query + 自己左边的词 | 第一个 token 换成 `[DEC]`，自回归生成描述 |
| **ITM** 图文匹配 | **双向掩码**（Bi-directional） | query + 文本全部 | query + 文本全部 | 每个 query 输出接二分类头，**32 个 logit 取平均** |

用矩阵表示（行 = 谁在看，列 = 被看的；✓ 表示可见）：

```
ITC（单模态）          ITG（多模态因果）              ITM（双向）
     Q   T              Q   T1  T2  T3                Q   T
Q    ✓   ✗         Q    ✓   ✗   ✗   ✗           Q    ✓   ✓
T    ✗   ✓         T1   ✓   ✓   ✗   ✗           T    ✓   ✓
                   T2   ✓   ✓   ✓   ✗
                   T3   ✓   ✓   ✓   ✓
```

各任务的作用：
- **ITC**：粗粒度对齐，并且防止“作弊”——query 和文本互相看不见，避免信息泄露。由于图像编码器冻结，显存够用，所以直接用 batch 内负样本，不再需要 BLIP 的动量队列。
- **ITG**：**最关键的一项**。文本无法直接看图像，只能通过 query 获取视觉信息，迫使 query **提取出“能用来写描述”的所有视觉信息**——这恰好就是阶段二 LLM 需要的。
- **ITM**：细粒度对齐，同样用难负样本挖掘。

### 5.5 阶段二：接入 LLM

```
Q-Former 输出 Z [32×768] ──► 全连接层 ──► [32×d_LLM] ──► 拼在文本 embedding 前面 ──► 冻结 LLM
```

- 一个**全连接层**把 Q-Former 输出投影到 LLM 的词向量维度。这 32 个向量就是“**软视觉提示（soft visual prompts）**”，LLM 把它们当作 32 个“特殊的词”。
- 因为阶段一已经让 query 学会提取**与语言相关**的视觉信息，这里相当于给 LLM 提供了“预先消化好的”视觉信息，减轻了 LLM 学习对齐的负担，也**缓解了灾难性遗忘**。
- 两种 LLM 的损失：
  - **解码器型（OPT）**：标准 LM 损失，以视觉提示为条件生成文本。
  - **编码器-解码器型（FlanT5）**：**前缀 LM 损失**——把文本切成前缀和后缀，前缀与视觉提示一起送入编码器，后缀作为解码器的生成目标。

**消融（论文 Figure 5）**：去掉阶段一直接训练阶段二，零样本 VQA 性能大幅下降，OPT 甚至出现灾难性遗忘（训练越久越差）。这说明“先在 Q-Former 里对齐，再接 LLM”这一步是必要的。

> 知乎原文提出的质疑值得记录：论文没有说明“去掉阶段一”的消融是否也去掉了阶段一的数据；后来更简单的 LLaVA 表明，**用一个线性层直接把视觉特征接到 LLM，也能取得很好的效果**，Q-Former 并非必要。参见 8.4 节的连接器对比。

### 5.6 训练细节

- 数据：与 BLIP 相同的 **129M** 图像（COCO、VG、CC3M、CC12M、SBU、LAION400M 中的 115M）。
- 用 **CapFilt 的变体**处理网图：用 BLIP-large captioner 为每张图生成 **10 条**描述，与原始描述一起用 CLIP ViT-L/14 计算图文相似度，**保留前 2 条**，训练时每步随机选一条。
- 阶段一 250K 步，阶段二 80K 步；batch size 阶段一 2320（ViT-L）/ 1680（ViT-g），阶段二 1920（OPT）/ 1520（FlanT5）。
- 冻结的 ViT 和 LLM 用 FP16（FlanT5 用 BF16）以节约显存。
- **算力很低**：最大的模型（ViT-g + FlanT5-XXL）在一台 16 张 A100（40G）的机器上，阶段一不到 6 天，阶段二不到 3 天。

### 5.7 代码

**（1）Q-Former 简化版：理解掩码如何切换任务**

```python
import torch
import torch.nn as nn

class QFormerLayer(nn.Module):
    def __init__(self, d=768, n_heads=12, d_img=1408, has_cross=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, kdim=d_img, vdim=d_img,
                                                batch_first=True) if has_cross else None
        self.ffn_q = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))  # query 侧 FFN
        self.ffn_t = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))  # 文本侧 FFN
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, n_query, img_feats, attn_mask):
        # x = [queries; text]，共享同一个 self-attention，可见性由 attn_mask 决定
        x = self.ln1(x + self.self_attn(x, x, x, attn_mask=attn_mask)[0])
        q, t = x[:, :n_query], x[:, n_query:]
        if self.cross_attn is not None:                       # 只有 query 去看冻结的图像特征
            q = self.ln2(q + self.cross_attn(q, img_feats, img_feats)[0])
        q = self.ln3(q + self.ffn_q(q))
        t = self.ln3(t + self.ffn_t(t))
        return torch.cat([q, t], dim=1)

def build_mask(n_q, n_t, mode):
    """返回 bool 掩码，True 表示"不允许看"。"""
    n = n_q + n_t
    mask = torch.zeros(n, n, dtype=torch.bool)
    if mode == "itc":                     # 单模态：Q 与 T 互不可见
        mask[:n_q, n_q:] = True
        mask[n_q:, :n_q] = True
    elif mode == "itg":                   # 多模态因果：Q 不看 T；T 看全部 Q + 左侧文本
        mask[:n_q, n_q:] = True
        mask[n_q:, n_q:] = torch.triu(torch.ones(n_t, n_t, dtype=torch.bool), diagonal=1)
    elif mode == "itm":                   # 双向：全部可见
        pass
    return mask
```

（真实实现见 LAVIS `lavis/models/blip2_models/Qformer.py`，交叉注意力每隔一层插入，且 ITC 中文本侧单独前向。）

**（2）ITC 的“取最大值”相似度**

```python
# image_feats: [B, 32, D]（32 个 query 输出，已归一化）; text_feat: [B, D]（[CLS]，已归一化）
sim_q2t = torch.einsum("iqd,jd->ijq", image_feats, text_feat)   # [B_img, B_txt, 32]
sim_i2t = sim_q2t.max(dim=-1).values / temp                     # 每对图文取 32 个 query 的最大相似度
```

**（3）HuggingFace 推理：零样本 VQA 与描述**

```python
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")

image = Image.open("demo.jpg").convert("RGB")
prompt = "Question: how many dogs are in the picture? Answer:"     # 论文的 VQA 模板
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
out = model.generate(**inputs, max_new_tokens=20)
print(processor.batch_decode(out, skip_special_tokens=True)[0].strip())
# 不给 text 就是无条件看图说话
```

**（4）LAVIS：只用 Q-Former 抽特征做检索**

```python
from lavis.models import load_model_and_preprocess
model, vis_proc, txt_proc = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device="cuda")
sample = {"image": vis_proc["eval"](image).unsqueeze(0).cuda(),
          "text_input": [txt_proc["eval"]("a dog on the grass")]}
img_emb = model.extract_features(sample, mode="image").image_embeds_proj   # [1, 32, 256]
txt_emb = model.extract_features(sample, mode="text").text_embeds_proj     # [1, L, 256]
sim = (img_emb @ txt_emb[:, 0, :].t()).max()                                # 同 ITC：取最大
```

### 5.8 关键结果

- **零样本 VQAv2：65.0%**（ViT-g + FlanT5-XXL），比 Flamingo-80B 的 56.3% **高 8.7 个点**，而**可训练参数少 54 倍**。
- 零样本 NoCaps 描述、零样本 Flickr30K 检索均为当时 SOTA。
- 规律：**更强的视觉编码器（ViT-g > ViT-L）和更强的 LLM（大 > 小、FlanT5 > OPT）都带来提升**——验证了“模块化、可插拔、随单模态模型进步而进步”的设计。
- 涌现出**零样本的“看图 + 知识 + 推理”能力**：如看图讲故事、结合常识回答问题（得益于 LLM）。

### 5.9 局限

1. **上下文学习（in-context learning）能力弱**：给几个示例并不能提高 VQA 效果。论文归因于预训练数据**每个样本只有一个图文对**，LLM 没学过“多图多文交错”的模式（Flamingo 用了 M3W 交错数据）。→ BLIP-3 用交错数据解决。
2. **继承 LLM 的问题**：幻觉、过时知识、偏见与有害输出。
3. **不会“听话”**：只学了描述和简短问答，复杂指令下表现一般。→ InstructBLIP。
4. **32 个 token 的信息瓶颈**：对文字密集图（OCR、图表、文档）、细小物体不友好；分辨率固定 224。
5. 阶段一训练复杂，阶段一与阶段二的目标不一致（ITC/ITM/ITG vs LM）。

<br>

## 6. InstructBLIP：让多模态模型听懂指令

### 6.1 动机

- LLM 领域的经验：**指令微调（instruction tuning）** 能让模型泛化到没见过的任务（FLAN、InstructGPT）。
- 视觉-语言任务比纯文本任务更多样（不同的图像类型、不同的输出格式），之前的多任务训练（直接混合所有数据集）泛化到新任务时效果有限。
- BLIP-2 的 Q-Former 提取的视觉特征**与指令无关**：无论问“图里有几个人”还是“图上写了什么字”，给 LLM 的都是同样的 32 个 token。

### 6.2 数据：26 个数据集，11 类任务

- 收集 **26 个公开数据集**，归为 **11 类任务**：图像描述、带阅读理解的图像描述（OCR）、视觉推理、图像问答、知识问答、带阅读理解的问答、图像问题生成、视频问答、视觉对话、图像分类、LLaVA-Instruct-150K。
- 把数据集变成指令格式：每个任务手写 **10–15 个指令模板**，如 “\<Image\> Write a short description for the image.” “\<Image\> Question: {} Short answer:”。对短答案数据集，在模板中加入 “short”“briefly” 等词，防止模型过拟合到总是生成短答案。
- **13 个 held-in**（用于训练）、**13 个 held-out**（零样本评估）。held-out 中有 4 个是**整类任务都没见过**（如视觉推理、视频问答、视觉对话、图像分类），用于检验跨任务泛化。

### 6.3 核心改进：指令感知的 Q-Former（Instruction-aware Q-Former）

```
BLIP-2:        Queries ──────────────────► Q-Former ──► 与指令无关的视觉特征 ──► LLM ◄── 指令
InstructBLIP:  Queries + 指令 tokens ─────► Q-Former ──► 与指令相关的视觉特征 ──► LLM ◄── 指令
                         ↑ 两者在自注意力中交互，query 按指令去图像里找信息
```

- 指令文本既送给 LLM，**也送进 Q-Former**，在自注意力层与 query 交互，从而让交叉注意力**按指令提取**图像特征。
- 这复用了阶段一 ITG/ITM 训练出的“query 与文本交互”能力，几乎不增加参数。
- 训练时**只微调 Q-Former**（和 FC 层），图像编码器（ViT-g/14）和 LLM（FlanT5-XL/XXL、Vicuna-7B/13B）冻结；用预训练好的 BLIP-2 初始化，Vicuna 版本重新做了 BLIP-2 预训练。
- 消融：去掉指令感知，held-out 平均下降明显，在空间推理（ScienceQA）和时序推理（视频问答）等需要“按问题看图”的任务上下降最多。

### 6.4 数据平衡采样

数据集大小差异巨大，直接混合会让模型过拟合小数据集、欠拟合大数据集。采样概率按数据集大小的**平方根**：

$$p_d=\frac{\sqrt{S_d}}{\sum_{i=1}^{D}\sqrt{S_i}}$$

再手动微调部分权重（如降低 A-OKVQA 等易过拟合的多选任务、提高 OKVQA 的权重）。

### 6.5 训练与代码

- 所有模型训练最多 60K 步，学习率 1e-5，16 张 A100 大约 1.5 天。

```python
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

name = "Salesforce/instructblip-vicuna-7b"
model = InstructBlipForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
processor = InstructBlipProcessor.from_pretrained(name)

image = Image.open("demo.jpg").convert("RGB")
prompt = "What is unusual about this image? Explain in detail."
# processor 会同时生成给 LLM 的 input_ids 和给 Q-Former 的 qformer_input_ids
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
out = model.generate(**inputs, do_sample=False, num_beams=5, max_new_tokens=256)
print(processor.batch_decode(out, skip_special_tokens=True)[0].strip())
```

### 6.6 结果

- 在全部 **13 个 held-out 数据集上取得零样本 SOTA**，大幅超过 BLIP-2 和更大的 Flamingo-80B；连最小的 InstructBLIP（FlanT5-XL，4B）都超过 Flamingo-80B。
- **指令微调 vs 多任务训练**：在 held-in 上两者差不多，但在 held-out 上指令微调明显更好——说明收益来自“学会遵循指令”而非“见过更多数据”。（知乎原文也指出，即便如此，部分 held-out 任务的绝对表现依然不理想。）
- 作为下游微调的初始化也更好：ScienceQA（图像部分）微调后达到 **90.7%**。
- 定性上能做复杂视觉推理、知识关联、多轮对话；在描述细节上比 LLaVA、MiniGPT-4 更注重“基于图像”。
- 发现：FlanT5 版本在多选题上更好，Vicuna 版本在开放式生成上更好（与 LLM 的预训练方式一致）。

### 6.7 局限

- 继承 LLM 的幻觉与偏见；对“图中没有的东西”仍会编造。
- 仍然只有 32 个视觉 token、固定分辨率，OCR、细粒度任务受限。
- 只支持单图输入；指令数据主要来自“改写的学术数据集”，与真实用户的开放式请求有差距（LLaVA 则用 GPT-4 生成更自然的对话数据）。

<br>

## 7. BLIP-3（xGen-MM）：回归简单，数据为王

### 7.1 动机：BLIP-2 的三个痛点

1. **数据**：BLIP-2 的预训练数据规模和多样性有限，且是“一图一文”，缺少图文交错数据 → 没有上下文学习能力。
2. **架构**：Q-Former 结构繁琐，阶段一要用 ITC、ITM、ITG 三个损失，阶段二又换成 LM 损失，目标不一致、难以扩展。
3. **开放性**：BLIP-2 只支持单图；而且业界最强的大模型（GPT-4V、Gemini）不开源，开源社区缺少可复现的完整“数据 + 训练配方 + 模型”。

### 7.2 架构

```
图像（任意分辨率）──► 切块 + 缩略全图 ──► SigLIP ViT（冻结/微调）──► Perceiver Resampler（token sampler）──► 视觉 tokens ─┐
                                                                                                         ├─► Phi-3-mini LLM ──► 文本
文本 / 交错的图文序列 ───────────────────────────────────────────────────────────────────────► 文本 tokens ─┘
```

- **视觉编码器**：SigLIP（而不是 EVA-CLIP）。
- **连接器**：用 **Perceiver Resampler**（Flamingo 中提出的结构）作为 token sampler，替代 Q-Former。本质也是“一组可学习 query + 交叉注意力”，把可变数量的视觉 patch 压缩成**固定数量**的视觉 token——但**不再需要 ITC/ITM/ITG 预训练**。
- **LLM**：Phi-3-mini（约 3.8B），总模型约 4B，属于“小而强”的路线。
- **统一损失**：所有阶段都只用**对文本 token 的自回归 LM 损失**（视觉 token 不计损失）。

知乎原文的评价：“BLIP-3 终于变简单了，但仍保留了 token sampler”。保留它的好处是：视觉 token 数量**固定且少于** patch 数量，降低 LLM 的计算量，对多图、高分辨率尤其重要。

### 7.3 任意分辨率（Any-Resolution）视觉 token 采样

- 借鉴 LLaVA-NeXT：把高分辨率图像**切成多个小块**，每块单独过 ViT 和 token sampler；
- 另外加一张**缩小的整图**保留全局信息；
- 所有子图的视觉 token + 全图 token 一起拼接送入 LLM。
- 好处：保留细节（文字、小物体），对文档理解、OCR、图表帮助很大——直接解决了 BLIP-2 “固定 224 分辨率 + 32 token” 的瓶颈。

### 7.4 数据：BLIP-3 真正的核心贡献

| 数据集 | 规模 | 内容 | 作用 |
|---|---|---|---|
| **MINT-1T** | 约 1 万亿 token | 图文**交错**文档（HTML 网页、PDF、ArXiv） | 学会多图多文的上下文学习 |
| **BLIP3-KALE** | 大规模 | 知识增强的**稠密描述**（合成描述 + 网页原文中的事实知识融合） | 比短 alt-text 信息丰富得多 |
| **BLIP3-OCR-200M** | 约 2 亿 | 带 OCR 文字和位置的图像 | 文字识别、文档理解 |
| **BLIP3-GROUNDING-50M** | 约 5 千万 | 带物体位置的描述 | 视觉定位 |
| 其他 | — | 公开的描述数据（如 Datacomp、CC12M 重标注）等 | 基础对齐 |

可以看到 CapFilt 的思想在这里进化成了“**合成稠密描述 + 知识增强**”。

### 7.5 训练流程

1. **预训练**：在上述混合数据上做 LM 训练（约千亿级多模态 token）。
2. **SFT（有监督指令微调）**：用公开的指令数据（多模态对话、VQA、文档、数学、OCR 等）。
3. **交错多图 SFT**：进一步用多图指令数据训练，得到多图版本。
4. **对齐**：
   - **DPO（直接偏好优化）**：减少幻觉、提升有用性；
   - **安全微调**：降低有害输出。
   （知乎原文指出，论文对 DPO 部分的细节介绍不多。）

最终发布 base（预训练）、instruct（SFT）、DPO 等多个版本，同时开源权重、数据集和训练配方。

### 7.6 结果与意义

- 在同规模（~4–5B）开源模型中，单图、多图、OCR/文档类基准上表现有竞争力；base 模型具备**少样本上下文学习**能力（BLIP-2 不具备）。
- 意义：标志着 BLIP 系列与业界主流收敛到同一形态——“**视觉编码器 + 简单连接器 + LLM，统一 LM 损失，重心放在数据和后训练**”。

### 7.7 延伸：BLIP 系列的其他分支

- <a href="https://arxiv.org/abs/2305.14720">BLIP-Diffusion</a>（2023）：用 BLIP-2 的多模态编码器提取“主体”表示，作为 Stable Diffusion 的条件，实现**零样本主体驱动生成**（给一张狗的照片，生成这只狗在不同场景中的图）。
- <a href="https://arxiv.org/abs/2505.09568">BLIP3-o</a>（2025）：迈向**理解 + 生成统一**的模型，用扩散 Transformer 生成 **CLIP 图像特征**（而不是 VAE 隐变量），先训理解、再训生成（冻结理解部分）。BLIP 系列从“理解”最终走向了“理解与生成统一”。

<br>

## 8. 横向对比与深度分析

### 8.1 全系列对比总表

| | CLIP | ALBEF | BLIP | BLIP-2 | InstructBLIP | BLIP-3 |
|---|---|---|---|---|---|---|
| 时间 | 2021.02 | 2021.07 | 2022.01 | 2023.01 | 2023.05 | 2024.08 |
| 架构范式 | 双塔 | 融合编码器 | 编码器-解码器（MED） | LLM 中心 + Q-Former | 同 BLIP-2 | LLM 中心 + Resampler |
| 视觉编码器 | ResNet / ViT（从零训） | ViT-B/16 | ViT-B/L（ImageNet 初始化） | EVA ViT-g / CLIP ViT-L（**冻结**） | ViT-g（冻结） | SigLIP |
| 语言模型 | 12 层 Transformer | BERT-base | BERT-base | OPT / FlanT5（**冻结**） | FlanT5 / Vicuna（冻结） | Phi-3-mini |
| 连接/融合 | 无（点积） | 交叉注意力 | 交叉注意力 | Q-Former（32 query） | 指令感知 Q-Former | Perceiver Resampler |
| 预训练目标 | ITC | ITC + ITM + MLM | ITC + ITM + LM | 阶段一 ITC+ITM+ITG；阶段二 LM | LM（指令） | 只有 LM |
| 可训练参数 | 全部 | 全部 | 全部（~2–5 亿） | 仅 Q-Former（188M）+ FC | 仅 Q-Former | 大部分 |
| 预训练数据 | 400M | 14M | 14M / 129M | 129M | 129M + 指令数据 | 万亿 token 级交错 + 数亿合成 |
| 数据处理 | 平衡查询 | 动量蒸馏容忍噪声 | **CapFilt** | CapFilt + CLIP 排序 | 模板化 + 平方根采样 | 合成稠密描述 + 交错文档 |
| 能生成文本 | ✗ | ✗ | ✓（短） | ✓ | ✓（长、听指令） | ✓（多图、对话） |
| 上下文学习 | ✗ | ✗ | ✗ | 弱 | 弱 | ✓ |
| 最擅长 | 检索、零样本分类、做“基础组件” | 检索、VQA、推理 | 检索、描述 | 零样本 VQA、描述 | 零样本泛化、指令跟随 | 多图、OCR、对话 |

### 8.2 损失函数的演化

```
CLIP:          ITC
ALBEF:         ITC + ITM + MLM              （+ 动量蒸馏）
BLIP:          ITC + ITM + LM               （MLM → LM：获得生成能力）
BLIP-2 阶段一： ITC + ITM + ITG              （LM 换成以 query 为瓶颈的 ITG）
BLIP-2 阶段二： LM（送入冻结 LLM）
InstructBLIP:  LM（指令数据）
BLIP-3:        LM（全程统一）  + DPO
```

规律：**从“多目标拼装”走向“单一的下一个 token 预测”**。这与 LLM 的成功经验一致：足够大的模型 + 足够多样的数据 + 一个简单的目标。但要注意，**对比损失并没有消失**——它沉淀在了视觉编码器（CLIP/SigLIP）的预训练里。

### 8.3 三条演化主线背后的“为什么”

1. **为什么从端到端走向冻结？** 单模态模型越来越大，联合训练成本爆炸；冻结后可以“站在巨人肩上”，且模块可替换。代价是视觉编码器的固有缺陷（见 8.5）会一路带进下游。后来很多工作（如 LLaVA-1.5 之后的模型、BLIP-3）在后期**解冻视觉编码器**微调，取得进一步提升。
2. **为什么数据处理越来越重？** 模型能力的上限越来越由数据决定。BLIP 的 CapFilt → BLIP-2 的 CLIP 排序 → BLIP-3 的合成稠密描述，路线一致：**用模型改善数据，再用数据训练更好的模型**（数据飞轮）。
3. **为什么架构越来越简单？** 当 LLM 足够强、数据足够多时，复杂的归纳偏置（多目标、精巧掩码）带来的收益被数据规模淹没；简单结构更容易扩展和调试。

### 8.4 连接器（Connector / Projector）对比：Q-Former 到底有没有必要

| 连接器 | 代表 | 结构 | 视觉 token 数 | 优点 | 缺点 |
|---|---|---|---|---|---|
| **交叉注意力插入 LLM** | Flamingo | 在 LLM 层间插入门控交叉注意力层 | 不占 LLM 输入长度 | 多图、交错天然支持 | 要改 LLM 结构，训练参数多 |
| **Q-Former** | BLIP-2、InstructBLIP、MiniGPT-4 | query + 交叉注意力 + 三目标预训练 | 固定 32 | token 少、计算省；可按指令提取 | 预训练复杂；信息瓶颈伤细节（OCR）；收敛慢 |
| **Perceiver Resampler** | Flamingo、BLIP-3、Idefics | query + 交叉注意力，直接用 LM 损失训 | 固定（如 64/128） | 比 Q-Former 简单，token 可控 | 仍有压缩损失 |
| **线性层 / MLP** | LLaVA、LLaVA-1.5 | 每个 patch 直接投影 | = patch 数（如 576） | 最简单、保留全部信息、收敛快 | token 多，高分辨率/多图时 LLM 计算贵 |
| **卷积/池化压缩** | Honeybee（C-Abstractor）、LLaVA 的 pooling 变体 | 局部聚合后投影 | 可调 | 保留局部空间结构 | 需要设计 |

**结论（社区的共识）**：
- 在数据充足时，**MLP 简单且效果好**（<a href="https://arxiv.org/abs/2304.08485">LLaVA</a>、<a href="https://arxiv.org/abs/2310.03744">LLaVA-1.5</a> 的经验）；Q-Former 的“32 token 瓶颈”在细粒度任务上是劣势，有工作指出它会丢失空间位置信息。
- 当**视觉 token 太多**（高分辨率、多图、视频）时，resampler 类“压缩器”又重新变得有价值——BLIP-3 保留 resampler 正是这个原因。
- 所以 Q-Former 的贡献更多是**历史性**的：它在 2023 年初证明了“冻结 ViT + 冻结 LLM + 轻量桥接”的可行性。

### 8.5 CLIP 系视觉编码器的共性缺陷（会一路传导到 MLLM）

1. **模态鸿沟（Modality Gap）**（<a href="https://arxiv.org/abs/2203.02053">Liang et al., NeurIPS 2022</a>）：CLIP 的图像向量和文本向量在嵌入空间里其实分布在**两个分离的“锥形”区域**，并没有真正混在一起。原因是随机初始化时的锥效应 + 对比损失在低温度下会保持这种间隔。影响：直接用图像向量替代文本向量（或反之）的做法会出问题。
2. **词袋问题（Bag-of-Words）**（<a href="https://arxiv.org/abs/2210.01936">ARO, Yuksekgonul et al., ICLR 2023</a>）：CLIP 很难区分“马在骑人”和“人在骑马”，对词序、属性绑定、空间关系不敏感。原因：检索任务里通常不需要理解组合关系就能找到正确答案，对比学习就没有动力去学。改进：在训练中加入**打乱词序的难负样本**（NegCLIP）。
3. **“CLIP 盲点”**（<a href="https://arxiv.org/abs/2401.06209">Eyes Wide Shut / MMVP, Tong et al., CVPR 2024</a>）：找出 CLIP 认为几乎相同、但视觉上明显不同的图像对（如朝向、数量、颜色、视角差异），基于 CLIP 的 MLLM（包括 GPT-4V）在这些问题上表现接近随机。改进：混合自监督视觉特征（DINOv2）。
4. **分辨率低**：CLIP 多在 224/336 训练，OCR、小物体、文档图像吃亏 → any-resolution、高分辨率编码器。
5. **计数与空间推理弱**：对比目标只关注全局语义。
6. **文字攻击与偏见**：见 2.9 节。

### 8.6 多模态大模型（BLIP-2 之后）的共性问题

| 问题 | 表现 | 常见对策 |
|---|---|---|
| **物体幻觉** | 描述图中不存在的物体 | 更好的数据（稠密描述）、DPO/RLHF（BLIP-3、LLaVA-RLHF）、解码时对比（VCD） |
| **语言先验过强** | 不看图也能“猜”答案 | 平衡数据、反事实样本 |
| **细粒度/OCR 弱** | 读不清图中文字 | 高分辨率、任意分辨率切块、OCR 数据 |
| **多图/视频** | 上下文长度爆炸 | 交错数据、token 压缩、resampler |
| **评测不可靠** | 基准泄漏、答案格式影响 | 多维基准（MME、MMBench、SEED-Bench、MMMU） |

### 8.7 优化方向总结

1. **视觉编码器**：更好的损失（SigLIP）、更大规模（EVA-CLIP-18B）、更高分辨率、混合自监督特征（DINOv2）、解冻微调。
2. **连接器**：在“信息保留”与“token 数量”之间权衡；动态 token 数（按图像复杂度分配）。
3. **数据**：合成稠密描述、交错文档、OCR/定位数据、高质量指令数据；数据策展比堆量更重要（MetaCLIP、DFN）。
4. **训练**：统一 LM 损失；多阶段（对齐 → 指令 → 偏好）；高效微调（LoRA、只训连接器）。
5. **对齐与安全**：DPO / RLHF 抑制幻觉，安全微调。
6. **统一理解与生成**：BLIP3-o、Chameleon、Transfusion 等“一个模型既看又画”的方向。

<br>

## 9. 实践指南：我该用哪个

### 9.1 选型建议

| 需求 | 首选 | 理由 |
|---|---|---|
| 以文搜图 / 以图搜图 / 零样本分类 | CLIP / OpenCLIP / SigLIP（中文：Chinese-CLIP） | 双塔，向量可离线建索引，毫秒级检索 |
| 检索精排 | BLIP ITM 或 BLIP-2 Q-Former | “ITC 召回 + ITM 精排”两阶段 |
| 批量给图片生成描述（数据标注、alt-text） | BLIP / BLIP-2 | 轻量、速度快；需要更细节的描述用 MLLM |
| 零样本 VQA、简单问答 | BLIP-2 / InstructBLIP | 资源有限时性价比高 |
| 复杂指令、多轮对话、多图、文档 OCR | BLIP-3 或同代更强的开源 MLLM | 任意分辨率、交错数据训练 |
| 文生图的条件编码 | CLIP / OpenCLIP 文本塔 | 生态默认 |
| 少样本适配到新领域分类 | CLIP + CoOp / Tip-Adapter / 线性探测 | 不改主干，数据需求小 |

### 9.2 微调与使用技巧

- **CLIP 零样本**：一定要用 prompt 模板和 prompt ensemble；类别名尽量写成自然语言、消除歧义。
- **CLIP 微调**：全量微调容易破坏分布外鲁棒性，优先考虑 WiSE-FT（权重插值）、冻结图像塔的 LiT 式训练，或 LoRA；小 batch 训练对比损失时考虑 SigLIP 损失或特征队列。
- **BLIP-2 / InstructBLIP**：严格使用论文中的 prompt 格式（如 “Question: … Answer:”），格式不同效果可能差很多；只训 Q-Former 就能做领域适配，显存需求低。
- **显存**：BLIP-2 OPT-2.7B 用 FP16 约需十几 GB；更大的 LLM 可用 8-bit/4-bit 量化加载（`load_in_8bit`）。

### 9.3 常见坑

1. CLIP 的文本最长 77 token，超出会被截断——长描述要用 Long-CLIP 或分句。
2. 相似度数值本身没有绝对意义（与温度有关），**只适合比较排序**；做阈值过滤要在自己数据上标定。
3. 用 CLIP 特征做下游时，注意是取**投影后**的嵌入（`get_image_features`）还是投影前的特征，以及是否 L2 归一化。
4. MLLM 的 VQA 准确率对**答案格式**和**prompt**非常敏感，比较不同模型时要统一评测协议。
5. 不要相信 MLLM 对“图中是否存在某物”的回答——幻觉问题仍普遍存在。

<br>

## 10. 自测问答（复习用）

1. **CLIP 为什么能零样本分类？** 它把分类转化为“图像与哪个类别文本描述最相似”的检索问题；类别空间是自然语言，所以可以用任意新类别名。
2. **对比学习为什么需要大 batch？** 每个正样本的负样本就是 batch 内其他样本，batch 越大负样本越多越难，学到的表示越有区分度。替代方案：动量队列（ALBEF）、sigmoid 损失（SigLIP）。
3. **ITC、ITM、LM/ITG 分别解决什么？** ITC 粗粒度全局对齐（可用于快速检索）；ITM 细粒度匹配判断（用难负样本）；LM/ITG 赋予生成能力并迫使视觉表示包含足够描述信息。
4. **BLIP 中为什么只有自注意力层不共享？** 编码器需要双向注意力，解码器需要因果注意力，冲突集中在 SA 层；其余层共享可以提高训练效率和多任务协同。
5. **CapFilt 为什么用核采样？** 核采样产生更多样的描述，带来更多新信息；束搜索生成的是安全但单调的高频句式。
6. **Q-Former 的 32 个 query 是什么？** 可学习的参数向量，通过交叉注意力从冻结图像特征中提取信息，起到“信息瓶颈 + 模态翻译”的作用。
7. **Q-Former 阶段一的三种掩码？** ITC 单模态掩码（query 与文本互不可见）；ITG 多模态因果掩码（文本只能通过 query 看图、且只看左侧词）；ITM 双向掩码（全部可见）。
8. **BLIP-2 为什么需要两阶段？** 冻结 LLM 只用 LM 损失难以完成视觉-语言对齐，消融显示去掉阶段一会明显变差甚至灾难性遗忘。
9. **InstructBLIP 比 BLIP-2 多了什么？** 指令数据 + 指令感知 Q-Former（指令也送进 Q-Former）+ 平衡采样。
10. **BLIP-3 为什么能做上下文学习？** 使用了大规模图文交错数据（MINT-1T），模型学会了处理多图多文序列。
11. **Q-Former vs MLP 投影？** Q-Former token 少、计算省但信息有损、训练复杂；MLP 简单保真但 token 多。数据充足时 MLP 往往够用，token 太多时压缩器重新有价值。
12. **CLIP 的“词袋”问题是什么？怎么缓解？** CLIP 对词序与组合关系不敏感；用打乱词序/替换属性的难负样本训练（NegCLIP）。

<br>

## 11. 综述补充：几种看待多模态的视角

### 11.1 Baltrušaitis et al. (2017/2019)：五大挑战

经典的“问题分类法”：表示、翻译、对齐、融合、协同学习（见 1.2 节）。价值在于提供了**与模型无关的问题视角**——任何新模型都可以问：它主要解决了哪个挑战？例如 CLIP 解决“表示+对齐”，BLIP 的解码器解决“翻译”，BLIP-2 的冻结 LLM 体现“协同学习”。

### 11.2 Liang, Zadeh, Morency (2022/2024)：模态的三个性质与六大挑战

- 提出模态的三个核心性质：**异质性**（不同模态的结构、噪声、粒度不同）、**关联性**（模态间存在共享/互补信息）、**交互性**（组合后产生新的信息）。
- 六大挑战：表示、对齐、推理、生成、迁移、量化。新增的“量化”强调要**测量**模型到底利用了哪个模态、是否存在单模态捷径——对应本文 8.5、8.6 节的“语言先验”“词袋”问题。

### 11.3 Du et al. (IJCAI 2022)：VLP 模型综述

从五个维度梳理 VLP：**图像/文本编码方式**（区域特征 / 网格特征 / patch）、**模态交互架构**（单流 vs 双流）、**预训练任务**（MLM、MRM、ITM、ITC 等）、**下游任务**、**数据集**。其中“单流（把图文拼起来一起过 Transformer）vs 双流（各自编码再交互）”的分法有助于理解 ALBEF/BLIP 属于双流 + 交叉注意力。

### 11.4 Gan et al. (2022)：VLP 的基础与进展（微软）

按任务把 VLP 分成三大块：**图文任务**（VQA、描述、检索）、**核心视觉任务**（用 CLIP 类模型做分类、检测、分割——“开放词汇”方向）、**视频-文本任务**。它强调了 VLP 从“区域特征 + 检测器”到“端到端 patch”的转变（ALBEF、BLIP 正是这一转变的产物），并较早指出了“统一建模”与“基础模型”趋势。

### 11.5 Zhang et al. (TPAMI 2024)：VLM for Vision Tasks

专注于 **CLIP 式模型在视觉任务上的应用**，梳理三条技术线：
1. **VLM 预训练**：对比式、生成式、对齐式目标；
2. **VLM 迁移**：prompt tuning（CoOp）、特征适配器（CLIP-Adapter、Tip-Adapter）、其他微调；
3. **VLM 知识蒸馏**：把 CLIP 的开放词汇知识蒸馏到检测/分割模型（ViLD 等）。

是本文第 2.10–2.11 节最好的延伸阅读。

### 11.6 Yin et al. (2023)：MLLM 综述

把 MLLM 拆成三部分：**模态编码器**（多为 CLIP 系）、**预训练 LLM**、**模态接口（connector）**——接口又分为“token 级融合”（Q-Former、MLP，把视觉变成 token）和“特征级融合”（Flamingo 式在 LLM 内部插交叉注意力）。训练分三阶段：**预训练对齐 → 指令微调 → 对齐微调（RLHF/DPO）**。BLIP-2 → InstructBLIP → BLIP-3 恰好依次走完了这三个阶段。它还系统讨论了 M-ICL（多模态上下文学习）、M-CoT（多模态思维链）和幻觉问题。

### 11.7 Li et al. (2023)：从专家模型到通用助手（微软）

把多模态基础模型分为：
- **视觉理解模型**（CLIP 等，从“标签监督”到“语言监督”）；
- **视觉生成模型**（文生图，CLIP 作为条件编码器）；
- **统一视觉模型**（一个模型做多种视觉任务）；
- **端到端训练的 LMM**（BLIP-2、LLaVA 等）；
- **工具调用的多模态 Agent**（用 LLM 调度视觉专家模型，如 Visual ChatGPT、MM-REACT）。

这条“专家 → 通用助手”的主线，很好地解释了 BLIP 系列为什么从“多个任务头”走向“以 LLM 为核心的通用模型”。

### 11.8 Zhang et al. (2024)：MM-LLMs 综述

把 MM-LLM 统一为五个部件：**模态编码器 → 输入投影器 → LLM 骨干 → 输出投影器 → 模态生成器**。前三者对应“理解”（BLIP-2 等），后两者对应“生成”（如 NExT-GPT 输出图像/音频）。同时整理了 100 多个模型的训练配方，总结出几条经验：**更高的图像分辨率有帮助、高质量 SFT 数据比数量更重要、在 LLM 训练中混入纯文本数据可以防止语言能力退化**。

### 11.9 Bordes et al. (2024)：VLM 入门（Meta）

面向入门者，按训练范式分四类（对比、掩码、生成、基于预训练骨干），并花大量篇幅讨论**实践问题**：数据的去重与平衡、如何选择视觉编码器、如何评测（以及基准的偏差）、如何扩展到视频。其中“**数据质量 > 数据数量**”和“**评测要警惕语言先验**”两点，与本文 8.5–8.7 节的分析一致。

<br>

## 12. 总结

- **CLIP** 建立了“图像和文字共享一个语义空间”的基础设施，它的价值远超零样本分类，几乎成了整个多模态生态的“眼睛”。
- **ALBEF → BLIP** 在 CLIP 的对齐基础上加入融合与生成，用 ITC + ITM + LM 三件套统一理解与生成，并用 CapFilt 开启了“模型改进数据”的路线。
- **BLIP-2** 抓住了 LLM 时代的关键机会：冻结大模型、只训练轻量的 Q-Former，用很低的成本让 LLM “看见”图像。
- **InstructBLIP** 把 LLM 的指令微调经验搬到多模态，让模型从“会答题”变成“听懂人话”。
- **BLIP-3** 承认了“简单 + 数据”的胜利：去掉复杂的预训练任务，统一 LM 损失，把重心放到交错数据、合成描述、任意分辨率和偏好对齐上。

一句话：**多模态模型的发展，是从“设计精巧的结构”走向“简单结构 + 大模型 + 好数据”的过程；而 CLIP 式对比学习，作为视觉编码器的预训练方式，始终是这个体系的地基。** 开放问题仍然很多：文本是否应该是所有模态的核心（知乎原文的追问）、视觉编码器的盲点如何根治、幻觉如何消除、理解与生成如何真正统一。
