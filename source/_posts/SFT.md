---
title: 大模型训练三阶段总览：Pre-Training、SFT 与 RLHF 流程
date: 2026-03-03
categories:
  - 学AI/DS
tags:
  - LLM
  - SFT
  - RLHF
  - Pre-Train
desc: 梳理 Pre-Training / SFT / RLHF 概念与各自流程，并整理 PPO、GRPO、GSPO、DPO 等 RLHF 方法要点；公式与实例留待 SFT、RLHF 单独笔记展开。
---

## 一、大模型训练为何分三阶段：Pre-Training 与 Post-Training

常说的「大模型训练分三阶段」指的是：**Pre-Training（预训练）→ SFT（有监督微调）→ RLHF（人类反馈强化学习）**。从概念上可先归为两类：

| 阶段 | Bert 时期 | LLM 时期 |
| :--- | :--- | :--- |
| **Pre-Training** | Mask 训练、相似度训练等 | 用连续文本做**自监督**训练，学通用知识与生成能力；是 RLHF 流程里的**第一阶段** |
| **Post-Training** | 常叫 Fine-Tuning（微调） | 与**人类偏好对齐**：奖励模型训练、强化学习、全参/LoRA 等微调都算 Post-Training |

**Pre-Training 的两层含义**：一是预训练**方法**（如 Bert / 大模型的自监督做法），二是预训练**模型**（可直接用的基座）。在 LLM 语境下，「预训练大语言模型」常指已经过预训练（且往往还经过 RLHF）的模型；很多垂直场景是在此类基座上再做 **SFT**。

因此：**Pre-Training 打基础 → Post-Training 里先 SFT 再 RLHF**，三者关系是顺序与包含关系，不是并列三种选一。

<br>

## 二、Pre-Training 简述（为 SFT/RLHF 打基础）

**是什么**：大模型训练的**第一阶段**，用海量无标签语料做**自监督**训练，让模型学通用规律和知识，具备生成能力。

**为什么**：学通用知识、节约计算（下游不必从零训）、下游收敛更快且泛化更好。

**怎么做**：基于**下一词预测**（next-token prediction）。数据构造方式：label 与 data 错位，用「下一个 token」当标签。损失为负对数似然：

$$L = -\sum_{n=1}^{N} \log p(x_n \mid x_1,\ldots,x_{n-1}; \theta)$$

模型根据上文预测下一个 token，最小化该损失即可，无需人工标注，从而获得 generate 能力。SFT 与 RLHF 都是在此基础上的**定向优化**。

<br>

## 三、SFT 有监督微调：流程与要点

**定义与目的**：SFT 是预训练模型的**定向优化**，本质是「教模型做具体的事」。用**带标签的监督数据**把通用能力收束为**特定任务能力**，即从通用（general）到可用（usable）。

**流程要点**：

1. **数据**：人工整理「问题 + 对应答案」的**任务特定**标注数据；与预训练不同，不是连续长文本，而是 (prompt, response) 对。
2. **模型结构**：与预训练模型相同（如 Decoder-only），不改结构。
3. **损失**：一般**只对 response 部分算 loss**，不对 prompt 部分算，这样模型学的是「给定输入如何生成目标输出」。
4. **实现方式**：全参数微调，或 LoRA/QLoRA 等少量参数微调。

**与 Fine-Tuning 的区分**：Fine-Tuning 是宽泛概念——在预训练基础上用特定数据调参；SFT 是其中**一种实现**：**必须用有标注的任务数据**、**只用有监督学习**、**侧重快速适配下游任务**。半监督/无监督微调、自编码器等则属于别的 Fine-Tuning 方式，不叫 SFT。

更细的公式与数据格式示例留到 **SFT 单独笔记** 里从公式推导和例子展开。

<br>

## 四、RLHF 人类反馈强化学习：为何需要、整体定位

**是什么**：RLHF（Reinforcement Learning from Human Feedback）用**人类反馈**驱动强化学习，微调语言模型，使行为更符合人类偏好与需求。

**为什么在 SFT 之后还要 RLHF**：

- **弥补 SFT 的局限**：SFT 学的是「正确」答案，未必满足个性化与价值观；RLHF 用人类对输出的偏好训练奖励模型，让模型朝「更受人类认可」的方向优化。
- **提升综合与应变**：用户需求多样且多变，固定标注难以覆盖；RLHF 可随反馈动态调整，综合优化输出。
- **提升体验**：回答是否简洁、友好、及时等偏好，通过 RLHF 学习，而不是只靠 SFT 的规则式输出。

**整体流程概念**：收集人类对模型输出的偏好 → 训练**奖励模型（RM）** → 用强化学习（如 PPO）在「高奖励 + 不过度偏离原策略」的约束下更新策略模型。下面分方法整理几种常见做法。

<br>

## 五、几种 RLHF 方法

| 特性 | PPO | GRPO | GSPO | DPO |
| :--- | :--- | :--- | :--- | :--- |
| **奖励模型 RM** | 需要 | 需要 | 需要 | 不需要 |
| **Critic / 价值模型** | 需要 | 不需要 | 不需要 | 不需要 |
| **KL 约束 / Reference** | 需要 | 需要 | 需要 | 不需要 |
| **优化粒度** | token 级 | token 级 | 序列级 | 偏好对（非 RL） |
| **同问题多回答** | 否（单回答） | 是（G 个） | 是（G 个） | 否（偏好对） |
| **优势计算** | GAE | 组内均值/方差 | 组内 + 序列级权重 | — |
| **显存 / 计算** | 较高 | 较低（无 Critic） | 较低 | 低 |

### 5.1 PPO（Proximal Policy Optimization）

**角色**：RLHF 里最常用的策略优化算法，通过**限制策略更新步长**避免策略崩塌，实现稳定优化。

**流程（RLHF 中的 PPO）**：

1. 输入问题 $q$ → **策略模型（Policy Model）** → 生成回答 $o$。
2. $o$ 同时送入三个模块：
   - **参考模型（Reference，冻结）**：算 $o$ 与训练数据分布的 **KL 散度**，用作约束，防止策略偏离过远。
   - **奖励模型（Reward，冻结）**：给出**实际分数** $r$。
   - **价值模型（Value，更新）**：输出**预期分数** $v$。
3. 用 **GAE（广义优势估计）** 结合 $r$、$v$（以及 KL 惩罚）得到**优势** $A$。
4. 用 $A$ 更新策略模型：优势 &gt; 0 的 token 提高生成概率，&lt; 0 的降低。

**核心公式**：策略比率 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$，裁剪目标：

$$L(\theta) = \mathbb{E}_t\left[\min\big(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\right]$$

$\epsilon$ 常用 0.1 或 0.2，限制更新幅度。PPO 需要**奖励模型 + KL 约束 + Critic（价值模型）**，计算与工程成本较高。

<br>

### 5.2 GRPO（Group Relative Policy Optimization）

**思路**：对**同一问题 $q$** 用策略模型生成 **G 个回答** $o_1,\ldots,o_G$，用**组内相对分数**算优势，**不需要 Critic**，省显存。

**流程**：

1. $q$ → Policy Model → $o_1,\ldots,o_G$。
2. 每个 $o_i$ 送入：Reference（冻结）→ KL；Reward（冻结）→ 得分 $r_i$。
3. **组间计算**：用 $r_1,\ldots,r_G$ 的均值 $\mu_G$ 和标准差 $\sigma_G$ 得到**相对优势**：
   $$A_i^G = \frac{r_i - \mu_G}{\sigma_G + \varepsilon}$$
4. 优势 &gt; 0 的 token 强化，&lt; 0 的抑制，使输出更符合目标。

**特点**：用群体统计替代单独的价值模型，无需加载 Critic，更省 GPU 内存；更关注**组内相对好坏**。

<br>

### 5.3 GSPO（Grouped Sequence Policy Optimization）

**动机**：PPO/GRPO 在 **token 级别**优化，而评估往往是**整句/整段**打分，token 级容易引入噪声和奖励偏差；长文本下问题更明显。

**思路**：**序列级**优化——整句打分、整句裁剪、序列级重要性权重，与「按句给奖励」对齐。

**要点**：

- **序列级重要性权重**（对整句做长度归一化）：
  $$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{1/|y_i|} = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)$$
- **基于分组的优势**：$\hat{A}_i = \frac{r(x,y_i) - \text{mean}(\{r(x,y_i)\})}{\text{std}(\{r(x,y_i)\})}$（与 GRPO 类似，但用在序列级）。
- **整句裁剪**：对 $s_i(\theta)$ 做 clip，而不是对每个 token 的 ratio 裁剪；并可按长度做**自适应裁剪范围**。

**与 GRPO 区别**：GRPO 对**每个 token** 做重要性采样；GSPO 对 **tokens 赋予序列级**的重要性权重，用同一句的整体表现来加权，更稳、更高效。

<br>

### 5.4 DPO（Direct Preference Optimization）

**定位**：不训奖励模型、不用显式 RL，**直接从偏好数据**优化策略，实现与 RLHF 类似的对齐效果。

**数据**：偏好对 \(D = \{(x, y_w, y_l)\}\)，\(y_w\) 为优选，\(y_l\) 为劣选。

**目标**：最大化 \(y_w\) 相对 \(y_l\) 的偏好，等价于最小化损失（\(f_\theta(x,y)\) 可为对数概率或得分）：

$$\mathcal{L}_{DPO} = \mathbb{E}_{(x,y_w,y_l)\sim D}\left[-\log \frac{\exp f_\theta(x,y_w)}{\exp f_\theta(x,y_w) + \exp f_\theta(x,y_l)}\right]$$

**与 RLHF（PPO）对比**：

| 特性 | DPO | RLHF（如 PPO） |
| :--- | :--- | :--- |
| 优化方式 | 直接优化偏好目标 | 通过强化学习（如 PPO） |
| 奖励模型 | 不需要 | 需要单独训练 RM |
| KL 惩罚 | 不需要 | 通常需要（防策略漂移） |
| 计算与稳定性 | 较低、较稳 | 较高、需调参 |

DPO 把「偏好」写成分类式目标，用 SGD 等直接更新语言模型参数即可。

<br>

### 5.5 PPO/GRPO 的常见问题（简要）

- **Token 级优化与句子级评估错位**：实际按句打分，但 PPO/GRPO 按 token 更新，长文易产生噪声和奖励偏差；GSPO 通过序列级优化缓解。
- **GRPO 在 MoE 上的收敛**：MoE 每次只激活部分专家，策略更新后 **Router 可能激活不同专家**，新旧策略的「生成基础」结构不一致，重要性比率方差大，易触发剧烈 clip、梯度失真甚至崩溃；传统做法需 **Routing Replay** 等工程手段，成本高。

<br>

## 六、小结与后续笔记

- **Pre-Training**：自监督下一词预测，得到具备生成能力的基座。
- **SFT**：用 (prompt, response) 标注数据，只对 response 算 loss，把通用能力收束到具体任务。
- **RLHF**：用人类偏好训 RM，再用 PPO/GRPO/GSPO 等做策略优化，或直接用 DPO 从偏好目标优化。

本笔记侧重**流程总览**和**几种 RLHF 方法整理**，不重复不冗余。**SFT** 与 **RLHF** 各自会另有单独笔记：从公式推导、具体例子（数据格式、训练步骤）展开，并可与本文通过双链关联。
