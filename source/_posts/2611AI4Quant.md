---
title: AI for Quant 前沿研究整理（Agentic Quant / 金融基础模型 / 市场世界模型 / RL / 因果 Alpha / 微观结构）
date: 2026-11-02
categories:
  - 学AI/DS
tags:
  - LLM
  - Agent
  - Quant
  - RL
  - 时间序列
desc: 以一份 2026 年 AI for Quant 研究前沿概述为线索，精读 R&D-Agent-Quant、Quant-Arena、KTD-Fin、Agentic Quant Trading 综述、Kronos、TSFM 金融收益预测对比、TRADES、DiffLOB、MarS、ACM CSUR RL 量化综述等论文，并补充多重检验、因果/不变性学习、订单簿微观结构的经典工作。
---

本文以一份“AI for Quant：2026 年最值得关注的前沿研究”的概述为线索，逐篇核对并精读其引用的论文，再补充每个方向必须知道的经典工作。目标是：**以后只看这篇笔记，就能快速回想起这个领域的问题、方法、关键数字和坑。**

**核心论文（概述中引用，已逐篇核实）：**
- <a href="https://arxiv.org/abs/2505.15155">R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization</a>（NeurIPS 2025 Datasets & Benchmarks Track，<a href="https://github.com/microsoft/RD-Agent">GitHub</a>）
- <a href="https://arxiv.org/abs/2610.30113">Quant-Arena: Benchmarking LLM Agents for Quantitative Trading Research</a>（arXiv 2026.10，<a href="https://lab.einsia.ai/quant-arena">项目页</a>）
- <a href="https://arxiv.org/abs/2605.28359">From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets</a>（即 **KTD-Fin**，arXiv 2026.05）
- <a href="https://arxiv.org/abs/2608.31041">Agentic Quantitative Trading: A Survey of Workflows, Systems, and Evaluation</a>（综述，arXiv 2026.08）
- <a href="https://arxiv.org/abs/2508.02739">Kronos: A Foundation Model for the Language of Financial Markets</a>（AAAI 2026，<a href="https://github.com/shiyu-coder/Kronos">GitHub</a>）
- <a href="https://arxiv.org/abs/2606.27100">Pretrained Time-Series Foundation Models for Financial Return Forecasting</a>（arXiv 2026.06，即概述里“2026 年的一项比较研究”）
- <a href="https://arxiv.org/abs/2502.07071">TRADES: Generating Realistic Market Simulations with Diffusion Models</a>（ECAI 2025，<a href="https://github.com/LeonardoBerti00/DeepMarket">DeepMarket</a>）
- <a href="https://ijcai.org/proceedings/2026/730">DiffLOB: Diffusion Models for Counterfactual Generation in Limit Order Books</a>（IJCAI 2026，<a href="https://arxiv.org/abs/2602.03776">arXiv 2602.03776</a>，<a href="https://github.com/ZhuoHan1998/DiffLOB">GitHub</a>）
- <a href="https://dl.acm.org/doi/10.1145/3733714">The Evolution of Reinforcement Learning in Quantitative Finance: A Survey</a>（ACM Computing Surveys 2025，<a href="https://arxiv.org/abs/2408.10932">arXiv 2408.10932</a>，即概述里“167 篇 RL 研究”的综述）

**补充论文（为了把每个方向讲完整）：** MarS（ICLR 2025）、Re(Visiting) TSFMs in Finance、AlphaGen / Alpha-GPT / AlphaForge / AlphaAgent、Harvey-Liu-Zhu、Deflated Sharpe Ratio、PBO、White Reality Check / Hansen SPA、IRM、InvariantStock、DoubleAdapt、Gu-Kelly-Xiu、Almgren-Chriss、OPD、Avellaneda-Stoikov、Spooner、EIIE、FinRL、ORDC、ABIDES-MARL、DeepLOB、FI-2010、LOBCAST、LiT、OFI、Micro-price、Hawkes / Neural Hawkes、Deep Survival Fill Probability 等，链接在各节给出。

> **对原概述的几处更正（精读后发现）：**
> 1. “KTD-Fin”是基准名，论文标题是 *From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets*。
> 2. Quant-Arena 的“九个模型没有一个战胜等权市场”只对**每个模型的最佳得分配置**成立；38 个迭代配置中有 **1 个**（GPT-6 Sol, medium effort）以 +1.4pp 跑赢了市场。
> 3. RD-Agent(Q) 的“更少因子、更高收益”在 CSI 300 2017–2020 这一经典 Qlib 切分上取得，样本外（2024–2025）只报告了 IC/IR/MDD，没有报告 ARR。
> 4. Kronos 的“45 个交易所”在正文与附录不一致（附录写 “over 40 exchanges across more than 30 countries”）。

<br>

## 0. 全景：AI for Quant 在做什么

### 0.1 从“预测股价”到“自动化研究系统”

传统 ML for Quant 的范式很窄：给定特征 $\mathbf{x}_{i,t}$，预测股票 $i$ 下一期收益 $y_{i,t+1}$，再按预测排序选股。2025–2026 的前沿把问题扩展成一个**闭环系统**：

$$\underbrace{\text{发现信号}}_{\text{Agent / 因子挖掘}} \rightarrow \underbrace{\text{验证策略}}_{\text{回测 / 统计检验}} \rightarrow \underbrace{\text{模拟市场}}_{\text{World Model}} \rightarrow \underbrace{\text{优化执行}}_{\text{RL / 微观结构}} \rightarrow \underbrace{\text{持续适应}}_{\text{Regime / Online}}$$

这里的 Quant 指 **Quantitative Research、Systematic Trading、Portfolio Optimization、Market Making**，而不是“用 LLM 读财报”。

### 0.2 六大方向一览

| 方向 | 核心问题 | 核心技术 | 代表工作 | 研究价值 |
|---|---|---|---|---|
| 1. Agentic Quant Research | AI 能否完成“假设→因子→回测→反思”的研究闭环 | LLM Agents、代码生成、Bandit/RL、自动实验 | R&D-Agent(Q)、Quant-Arena、KTD-Fin、AlphaAgent | 极高 |
| 2. Financial Foundation Models | 能否像 LLM 学语言一样从海量 K 线中学通用表示 | Tokenization、自回归预训练、零样本 | Kronos、TimesFM/Chronos/Moirai | 极高 |
| 3. Market World Models | 学一个**对我的交易有反应**的市场 | Diffusion、自回归订单流生成、反事实 | TRADES、DiffLOB、MarS | 极高 |
| 4. RL for Trading & Execution | 在风险、成本、约束下做序列决策 | Risk-Sensitive / Offline / Model-Based / Multi-Agent RL | OPD、EIIE、Spooner、ABIDES-MARL | 高 |
| 5. Causal Alpha Discovery | 相关性 → 稳健的预测规律，避免数据挖掘 | 多重检验、不变性学习、Regime 适应 | DSR、PBO、IRM、DoubleAdapt | 极高 |
| 6. Market Microstructure | 毫秒到秒级的价格形成、成交与做市 | LOB 深度学习、点过程、生存分析 | DeepLOB、LiT、OFI、Neural Hawkes | 高 |

### 0.3 一句话总纲

读完所有论文后最强烈的感受是：**金融里最难的已经不是“产生预测”，而是“证明预测在扣除成本、杜绝泄漏、样本外之后仍有经济价值”。**

- Quant-Arena：九个前沿模型的最佳配置都没跑赢等权市场（2024 封闭年）。
- KTD-Fin：10 个 LLM 交易 Agent 中 9 个的**选股 Alpha 为负**，收益几乎全部来自市场共同项与风格暴露。
- TSFM 对比研究：预训练模型赢了 8/10 个任务，但相对 random walk 只有 **2 个**通过 Diebold–Mariano 显著性检验。

所以 Quant 是研究 **Agent Evaluation、Reward Design、Generalization** 的天然“困难模式”：reward 不能简单等于历史收益或 Sharpe。

<br>

## 预备知识：量化研究的基本语言

后面所有论文都会用到这些指标与设定，先统一记号。

### 截面预测与因子

- **因子（factor / alpha）**：对每只股票、每个时点给出一个分数 $F_{i,t}$，期望它与未来收益正相关。例如 20 日动量 $F_{i,t}=P_{i,t}/P_{i,t-20}-1$。
- **公式化因子（formulaic alpha）**：用算子（`Ref`、`Mean`、`Std`、`Corr`、`Rank`…）组合价量字段得到的表达式，如 Alpha101、Qlib 的 Alpha158 / Alpha360。
- **标签**：通常是未来收益 $y_{i,t}=(P_{i,t+\tau}-P_{i,t})/P_{i,t}$，Qlib 默认用 `Ref($close,-2)/Ref($close,-1)-1`（t+1 收盘到 t+2 收盘，避免用到 t 日收盘后才能下单的信息）。

### 信号质量指标

- **IC（Information Coefficient）**：每天在截面上计算因子与下一期收益的 Pearson 相关，再对时间取平均：

$$\text{IC}_t=\text{corr}_i(F_{i,t},\,y_{i,t}),\qquad \text{IC}=\mathbb{E}_t[\text{IC}_t]$$

- **Rank IC**：把 Pearson 换成 Spearman（秩相关），对异常值更稳健。
- **ICIR**：$\text{ICIR}=\mathbb{E}_t[\text{IC}_t]/\text{std}_t(\text{IC}_t)$，衡量信号的稳定性。
- 经验量级：日频 A 股上 IC≈0.03–0.05 已经是“不错的因子”；IC 很小但很稳定也可能非常赚钱。

### 组合与风险指标

- **ARR（年化收益）**：$\text{ARR}=\big(\prod_t(1+r_t)\big)^{252/T}-1$。
- **IR（信息比率）**：$\text{IR}=\dfrac{\text{mean}(r_t-r_b)}{\text{std}(r_t-r_b)}\sqrt{252}$，$r_b$ 为基准收益；若 $r_b=r_f$ 则即 Sharpe。
- **MDD（最大回撤）**：净值从历史高点到后续低点的最大跌幅。
- **Calmar**：$\text{ARR}/|\text{MDD}|$。
- **换手率（Turnover）**：每期调仓比例，决定交易成本。

### 回测设定中的“魔鬼细节”

- **TopK-Dropout**（Qlib 默认）：每天持有排名前 K（如 50）只，只卖出排名跌出的最差 n（如 5）只，以降低换手。
- **A 股规则**：T+1（当天买入次日才能卖）、涨跌停（主板 ±10%，创业板/科创板 ±20%，北交所 ±30%）、印花税/佣金（常见设定买 0.05%、卖 0.15%、最低 5 元）。
- **Look-ahead bias**：用了决策时点还不可得的信息（如当日收盘价下单、财报发布日错位）。
- **Survivorship bias**：只用当前还存在的股票做历史回测，遗漏退市股。
- **LLM 特有的泄漏**：模型预训练语料覆盖了测试期，“记住”了某股票后来涨了。
- **Barra 风格归因**：把组合收益拆成“市场共同项 + 风格因子暴露 × 因子收益 + 个股特异（选股 Alpha）”，用于判断收益到底是不是“真本事”。

<br>

## 1. Agentic Quant Research：让 AI 自己做量化研究

> LLM AGENTS · AUTOMATED ALPHA DISCOVERY · VERIFIABLE REWARDS

### 1.1 范式：从“研究员手工迭代”到“Agent 研究闭环”

传统 Quant Research 流程：

研究员提出假设 → 写因子 → 回测 → 分析结果 → 修改假设

Agentic Quant Research 的目标是让 Agent 完成整个闭环：

| 环节 | Agent 角色 | 关键难点 |
|---|---|---|
| Research | 阅读论文、市场数据、新闻，形成**可检验**假设 | 假设要可证伪、有经济含义 |
| Factor / Model Generation | 自动编写因子和模型代码 | 代码正确性、数据接口、防泄漏 |
| Backtesting | 滚动回测、成本模拟 | 交易规则、冲击成本、样本外隔离 |
| Evaluator + Learning Loop | 验证统计显著性、风险、成本、样本外表现，再反馈 | **多重检验偏差、回测过拟合** |

与通用 LLM Agent 的最大区别：**环境 reward 是有噪声、非平稳、可以被“刷分”的**。在代码任务里单测通过就是通过；在 Quant 里“回测 Sharpe 高”可能只是试了足够多次。

下面按“系统 → 基准 → 诊断 → 综述 → 因子挖掘谱系”的顺序精读。

<br>

### 1.2 R&D-Agent-Quant（NeurIPS 2025）：因子与模型联合优化的多智能体框架

- 论文：<a href="https://arxiv.org/abs/2505.15155">arXiv 2505.15155</a>，NeurIPS 2025 **Datasets and Benchmarks Track**
- 作者：Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, Jiang Bian（Microsoft Research Asia 等）
- 代码：<a href="https://github.com/microsoft/RD-Agent">microsoft/RD-Agent</a>（论文中称 R&D-Agent(Q)，README 中称 RD-Agent(Q)）

#### 动机

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180000123.png" style="width:90%">

图 1：量化研究五阶段流水线。Qlib 简化了 ❶ 数据处理与 ❹ 回测，R&D-Agent(Q) 进一步自动化 ❷ 因子研发、❸ 模型研发、❺ 分析反馈。

论文指出现有实践的三个问题：
1. **自动化不足**：假设、编码、调参依赖人工，迭代慢且带主观偏差；
2. **可解释性差**：很多 LLM 交易 Agent 直接从语言交互里输出买卖信号，没有落地到可执行的因子构建，容易幻觉；
3. **优化割裂**：因子研究与模型研究各自为战，缺乏跨阶段反馈。

核心卖点：**所有产出都是可执行代码**（因子表达式/模型类），因而可验证、可复现。

#### 方法：五个单元，两个阶段

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180001123.png" style="width:90%">

图 3：R&D-Agent(Q) 的五个功能单元。Research 阶段 = Specification + Synthesis；Development 阶段 = Implementation（Co-STEER）+ Validation；Analysis 单元负责反馈并用 Thompson Sampling 的双臂 Bandit 决定下一轮优化方向。

三种配置：
- **R&D-Factor**：模型固定为 LightGBM，只优化因子；
- **R&D-Model**：因子固定为 Alpha 20（Alpha158 的 20 个子集），只优化模型；
- **R&D-Agent(Q)**：两者联合优化。

**① Specification Unit（规范单元）**

把任务上下文形式化为四元组：

$$\mathcal{S}=(\mathcal{B},\mathcal{D},\mathcal{F},\mathcal{M})$$

- $\mathcal{B}$：背景假设与先验知识
- $\mathcal{D}$：市场数据接口（如 `daily_pv.h5` 包含 open/close/volume…）
- $\mathcal{F}$：期望输出格式（因子张量或收益预测，如 MultiIndex DataFrame）
- $\mathcal{M}$：外部执行环境（Qlib 回测）

任何候选 $f_\theta$ 必须满足 $\forall x\in\mathcal{D},\ f_\theta(x)\in\mathcal{F}$，且能在 $\mathcal{M}$ 中执行。本质是**用 schema 约束 LLM 输出空间**，减少无效代码。

**② Synthesis Unit（假设生成，“知识森林”）**

- 动作 $a_t\in\{\text{factor},\text{model}\}$；一次实验 $e^t=\{h^t,f^t\}$（假设，反馈）；维护 SOTA 集合。
- 生成下一个假设时，只取与当前动作相关、或已进入 SOTA 的历史：

$$\mathcal{H}_t^{(a)}=\{h_i\in\mathcal{H}_t \mid a=a_t \lor h_i\in\text{SOTA}(a)\},\quad h^{(t+1)}=G\big(\mathcal{H}_t^{(a)},\mathcal{F}_t^{(a)}\big)$$

- 若反馈显示成功，下一个假设增加复杂度或扩展范围；若失败，做结构调整或引入新变量——由此形成 Area → Idea 的“知识树/知识森林”。
- 一个因子假设可拆成多个因子子任务；一个模型假设对应一个模型任务。

**③ Implementation Unit：Co-STEER（代码生成 Agent）**

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180002123.png" style="width:80%">

图 4：Co-STEER = 调度模块（给任务排序）+ 实现模块（构建可迁移的知识库）。

- 因子任务之间有依赖，建成 DAG $\mathcal{G}=(\mathcal{V},\mathcal{E})$，按拓扑序 $\pi_S$ 实现（论文称为 guided chain-of-thought）。
- **自适应调度**：每个任务有复杂度 $\alpha_j$（初始 1），失败一次 $\alpha_j\leftarrow\alpha_j+\delta$，边权 $w_{ij}=\alpha_i/\alpha_j$，重新排拓扑序——简单任务优先。
- **知识库**：$\mathcal{K}^{(t+1)}=\mathcal{K}^{(t)}\cup\{(t_j,c_j,f_j)\}$，成功和失败的“任务-代码-反馈”三元组都存。新任务检索相似任务的代码作参考：$c_{new}=\arg\max_{c_k\in\mathcal{K}}\text{sim}(t_{new},t_k)\cdot c_k$。
- 与 Few-shot、CoT、Reflexion、Self-Debugging、Self-Planning 比，只有 Co-STEER 同时具备“调度 / 示例 / 先规划 / 自反馈 / 知识增长”五项。
- 设置：嵌入用 text-embedding-ada-002；每个任务内循环最多 10 次；实现超时 600s，验证超时 3600s。

**④ Validation Unit（验证与去重）**

新因子与 SOTA 因子库做相关性去重：

$$\text{IC}^{(n)}_{\max}=\max_m \mathbb{E}_t\big[\text{corr}(F^{(t)}_{SOTA,m},F^{(t)}_{new,n})\big],\qquad \text{IC}_{\max}\ge 0.99\Rightarrow\text{丢弃}$$

通过的因子与当前 SOTA 模型组合后在 Qlib 回测；模型侧对称（候选模型 + 当前 SOTA 因子集）。

**⑤ Analysis Unit + Bandit 调度器**

- 评估假设、任务、结果；超过 SOTA 则加入 SOTA($a_t$)；诊断失败原因写成反馈 $f_t$。论文说 Analysis 是**局部视角**，Synthesis 是**全局视角**。
- 下一轮优化“因子”还是“模型”？建模为**上下文双臂 Bandit**，用**线性 Thompson Sampling** 求解：

$$\mathbf{x}_t=[\text{IC},\text{ICIR},\text{RankIC},\text{RankICIR},\text{ARR},\text{IR},-\text{MDD},\text{SR}]^\top\in\mathbb{R}^8,\qquad r=\mathbf{w}^\top\mathbf{x}_t$$

$$\tilde\theta^{(a)}\sim\mathcal{N}\big(\mu^{(a)},(P^{(a)})^{-1}\big),\quad a_t=\arg\max_a\tilde\theta^{(a)\top}\mathbf{x}_t$$

$$P^{(a_t)}\leftarrow P^{(a_t)}+\tfrac{1}{\sigma^2}\mathbf{x}_t\mathbf{x}_t^\top,\qquad \mu^{(a_t)}\leftarrow (P^{(a_t)})^{-1}\big(P^{(a_t)}\mu^{(a_t)}+\tfrac{1}{\sigma^2}r_t\mathbf{x}_t\big)$$

- 代码细节（`rdagent/scenarios/qlib/proposal/bandit.py`）：默认权重 `(0.1,0.1,0.05,0.05,0.25,0.15,0.1,0.2)`，`prior_var=10`，`noise_var=0.5`；代码中的 “sharpe” 实际按 `arr / -mdd` 计算，即 **Calmar**；指标读取 Qlib 的 `excess_return_with_cost`。

#### 实验设置

- **主实验**：CSI 300，训练 2008–2014，验证 2015–2016，测试 2017-01 至 2020-08（经典 Qlib 切分）。
- **泛化实验**：CSI 500 与 NASDAQ 100，训练 2008–2021，验证 2022–2023，测试 2024-01 至 2025-06（晚于 GPT-4o 知识截止）。
- 交易：TopK-Dropout（top 50，drop 5），涨跌停阈值 0.095，收盘价成交，买 0.05% / 卖 0.15% / 最低 5 元；NASDAQ 100 用 top 20、单边 0.1%。
- 基线：因子库 Alpha101/158/360、AutoAlpha；ML（Linear、MLP、LightGBM、XGBoost、CatBoost、DoubleEnsemble）；DL（GRU、LSTM、ALSTM、Transformer、PatchTST、iTransformer、Mamba、TRA、MASTER、GATs）。
- 成本：每个工作流 LLM 费用 **< 10 美元**；R&D-Factor / R&D-Model 各跑 6 小时，联合版 12 小时。

#### 关键结果（CSI 300，表 1 节选）

| 方法 | IC | ICIR | RankIC | ARR | IR | MDD | Calmar |
|---|---|---|---|---|---|---|---|
| LightGBM（Alpha 20） | .0277 | .2211 | .0386 | 3.97% | .5664 | -8.55% | .4643 |
| MASTER | .0215 | .1925 | .0296 | 8.96% | 1.3406 | -8.51% | 1.0528 |
| iTransformer | .0270 | .1946 | .0340 | 9.79% | 1.2337 | -11.51% | .8506 |
| Alpha158 | .0341 | .2952 | .0450 | 5.70% | .8459 | -7.71% | .7393 |
| Alpha360 | .0420 | .3290 | .0514 | 4.38% | .6731 | -7.21% | .6074 |
| **R&D-Factor (GPT-4o)** | .0489 | .4050 | .0521 | **14.61%** | 1.6835 | -7.50% | **1.9468** |
| R&D-Model (o3-mini) | .0469 | .3688 | **.0546** | 10.09% | 1.7009 | **-6.94%** | 1.4538 |
| **R&D-Agent(Q) (o3-mini)** | **.0532** | **.4278** | .0495 | 14.21% | **1.7382** | -7.42% | 1.9150 |
| R&D-Agent(Q) (GPT-4o) | .0497 | .4069 | .0499 | 11.44% | 1.3167 | -8.11% | 1.4108 |

- 14.61% vs Alpha158 的 5.70% 约 2.6 倍；论文称用约 **22% 的因子**达到与 Alpha158/360 相当的 IC。
- **样本外（2024–2025.06，表 2）**：R&D-Agent(Q) o4-mini 在 CSI 500 上 IC .0288 / IR 2.17 / MDD -6.56%，NASDAQ 100 上 IC .0162 / IR 1.77；Alpha158 在两市的 IR 只有 0.25 / 0.03。
- **调度器消融（o3-mini）**：Random ARR 8.97%，LLM 调度 10.09%，**Bandit 14.21%**；Bandit 的有效循环和进入 SOTA 次数也最多。
- **研究行为（图 5）**：用 Sentence-BERT 计算假设两两相似度，36 次试验中 8 次进入 SOTA，覆盖 6 个簇中的 5 个，呈现“精炼—转向—复用”（refine–shift–reuse）模式。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180003123.png" style="width:60%">

图 5：假设相似度热力图，黑框为假设簇，红色编号为进入 SOTA 的试验。

#### 批判性阅读

1. **因子分支才是主力**：GPT-4o 下单独的 R&D-Factor（14.61%）**高于**联合版（11.44%）；论文自己的结论也是“因子驱动信号质量，模型是风险稳定器”。
2. **测试集参与选择的风险**：主实验测试期 2017–2020 是公开已久的 Qlib 切分；循环中 SOTA 的选择依据回测结果，而仓库 YAML 默认在 test 段回测。如果循环决策看到了测试指标，结果就带有样本内性质。
3. **LLM 先验泄漏**：GPT-4o 预训练覆盖 2017–2020 的互联网文本；作者的回应是 2024–2025 样本外实验，但那里没报 ARR。
4. **随机性**：Agent 运行单次 6/12 小时，消融未报告多种子方差；而基线跑了 5 个种子取中位数。
5. **内部不一致**：IR 定义用 $r_f$，代码读的是超额收益；标签定义（Eq.7 的 $\tau=1$）与仓库模板不同；“SR”在代码里是 Calmar。

> 结论：RD-Agent(Q) 是目前最完整、可复现的**开源 Quant 研究 Agent 框架**，最值得学的是“可执行规范 + 知识森林 + Co-STEER + Bandit 调度”这套工程设计；但它的数字更应理解为“在特定切分下的潜力展示”。

#### RD-Agent 仓库速查

<img src="https://raw.githubusercontent.com/microsoft/RD-Agent/main/docs/_static/Framework-RDAgent.png" style="width:80%">

R&D-Agent 通用框架（Research / Development 双循环），Quant 只是其中一个场景。

```bash
conda create -n rdagent python=3.10 && conda activate rdagent
pip install rdagent          # 仅支持 Linux，多数场景需要 Docker
rdagent health_check
rdagent fin_quant            # 因子+模型联合演化 = R&D-Agent(Q)
rdagent fin_factor           # 只演化因子 = R&D-Factor
rdagent fin_model            # 只演化模型 = R&D-Model
rdagent fin_factor_report --report-folder=<path>   # 从研报抽取因子
rdagent ui --port 19899 --log-dir log/             # Streamlit 可视化
```

- `.env` 配置 LLM（默认 LiteLLM 后端，如 `CHAT_MODEL=gpt-4o`）。
- 调度器可通过 `QLIB_QUANT_action_selection=bandit|llm|random` 切换，与论文消融一致。
- 数据默认 `~/.qlib/qlib_data/cn_data`，市场 csi300，基准 SH000300。
- 其他场景：`rdagent data_science --competition <name>`（Kaggle/MLE-bench）、`rdagent general_model <paper URL>`（从论文实现模型）。

<br>

### 1.3 Quant-Arena（2026.10）：LLM 能完成完整的量化研究吗？

- 论文：<a href="https://arxiv.org/abs/2610.30113">arXiv 2610.30113</a>，作者 Houde Qian 等（Navers Lab / Einsia AI）
- 项目页与排行榜：<a href="https://lab.einsia.ai/quant-arena">lab.einsia.ai/quant-arena</a>

#### 动机

现有基准只测碎片能力：写公式、写代码、做单次买卖决策。没有人测：**LLM 能否走完“假设 → 可执行信号 → 回测 → 决定保留什么”的完整循环，并产出一个在研究期之后仍然有效的策略？** 而反复回测本身就会过拟合，所以必须在**研究从未见过的数据**上评判最终提交的策略。

#### 基准设计

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180004123.png" style="width:95%">

Quant-Arena 流程：A. 在 2022–2023 上研究（≤5 次评估）并冻结一个策略；B. 在封闭的 2024 年上执行冻结代码，2024 的数据与结果**永不返回给 Agent**。

| 设计维度 | 具体做法 |
|---|---|
| 市场与数据 | 真实 A 股，3,042 只股票；保留缺失与退市股（避免幸存者偏差）；日频 + Level-2（报价、委托、成交、订单簿快照）；所有访问经过限制日期/标的/字段的接口 |
| 任务 | 只给研究目标、数据权限和接口文档；**没有预定义因子库**；最多声明 300 个命名特征；策略是 Python 选股打分程序 $s_t=f(H_t)$ |
| 时间切分 | 研究期 2022–2023（+授权预热历史），Agent 可自行划分训练/验证；封闭测试期 2024 全年 |
| 研究预算 | 每次运行最多 **5 次**候选评估；反馈只基于研究期（收益、信号质量、覆盖率、执行错误）|
| 防泄漏与冻结 | 冻结代码、环境、特征声明；在 2024 上只跑一次；所有运行全程记录，失败运行也保留，自动发布不可撤回 |
| 评估器 | 前 75 名等权多头、100 股一手、订单相关的交易成本、**基于模拟订单簿估计的市场冲击**；闲置现金收益为 0 |
| 两条赛道 | one-shot（无性能反馈）与 iterative（有研究期反馈）|

**评分公式**：用 $g_c(x)=\tfrac12\big(1+\tfrac{x}{c+|x|}\big)$ 把各指标压到 $[0,1]$（RankIC 用 $\tfrac12(1+\tanh(\text{RankIC}/0.03))$，换手用 $1/(1+TO)$）：

$$Q=0.6\,s_R(\text{超额收益})+0.2\,s_L(\text{多头 Sharpe})+0.05\,(s_{LS}+s_{IC}+s_{ST}+s_{TO})$$

$$D=Q\cdot\min(K,32)/32,\qquad S_{in}=90Q+10D,\qquad S=0.3\,S_{in}+0.7\,S_{out}$$

其中 $K$ 是合格特征簇数（奖励多样性），$S_{out}$ 维度相同但**规则不公开**（防止对着评分函数优化）。

测试模型：9 个前沿模型 × 各自支持的推理强度，共 38 个 iterative + 38 个 one-shot 配置（GPT-6 Astra、GPT-6 Sol、GPT-5.6 Terra、Claude Opus 5.5、Claude Sonnet 5.5、Gemini 3.8 Flash、DeepSeek V4.1 Flash、Kimi K3、GLM-5.3），外加不公开构造的专有参考策略 Kepler-1。

#### 关键结果（每个模型最佳 iterative 配置，2024 封闭年；等权市场 ARR = 17.1%）

| 模型（effort） | S iter | S one-shot | ARR | 超额(pp) | RankIC | Sharpe | MDD |
|---|---|---|---|---|---|---|---|
| GPT-6 Astra (xhigh) | 51.29 | 35.22 | 10.7% | −6.3 | 0.101 | 0.86 | −6.8% |
| GPT-6 Sol (xhigh) | 51.14 | 32.15 | 8.6% | −8.4 | 0.053 | 1.89 | −2.8% |
| GPT-5.6 Terra (max) | 47.58 | 42.14 | 8.4% | −8.7 | 0.026 | 0.42 | −28.8% |
| Claude Opus 5.5 (medium) | 44.28 | 42.61 | 4.4% | −12.7 | −0.002 | 0.29 | −32.7% |
| Gemini 3.8 Flash (medium) | 43.04 | 35.91 | −2.9% | −19.9 | 0.045 | 0.06 | −33.8% |
| Claude Sonnet 5.5 (max) | 40.93 | 40.24 | −12.2% | −29.2 | 0.045 | −0.23 | −36.1% |
| GLM-5.3 (max) | 38.91 | 38.02 | −7.8% | −24.9 | 0.046 | −0.18 | −27.8% |
| Kimi K3 (low) | 38.31 | 32.01 | −8.5% | −25.6 | 0.027 | −0.10 | −36.4% |
| DeepSeek V4.1 Flash (low) | 36.45 | 29.92 | −5.7% | −22.7 | −0.009 | −0.06 | −30.3% |
| Kepler-1（专有参考） | 77.56 | — | 94.0% | +76.9 | 0.085 | 2.98 | −15.9% |

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180005123.png" style="width:90%">

2024 年各模型最佳 iterative 配置的净值曲线 vs 等权市场与 Kepler-1。

**要点（按重要性）：**

1. **没有一个模型的最佳配置跑赢等权市场**；38 个 iterative 配置里只有 1 个（GPT-6 Sol, medium, ARR 18.4%, +1.4pp）跑赢；只有 13/38 的 ARR 为正。
2. **迭代提升分数，但不提升真实收益**：最佳配置平均分从 36.47（one-shot）升到 43.55（iterative），平均 ARR 只多 2.9pp；迭代让 9 个模型分数都上升，但只让 4 个模型的封闭期 ARR 上升。收益主要来自“修复一个糟糕的初版”，第二个候选后中位增益 6.5pp，随后很快平台期。
3. **过拟合信号**：iterative 配置从研究期到封闭期平均掉 12.4 分。
4. **IC 不等于可投资**：跨运行 RankIC 与 ARR 的 Spearman 相关只有 **0.09**。分组收益呈“驼峰”——组合买入的最高分组反而不如中间组。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180006123.png" style="width:75%">

按得分十分位的平均次日收益：最高分组并非最好（“hump”）。

5. **Astra 赢在择时而不是选股**：其 xhigh 策略在市场中位多日收益为负时持有现金，约 75% 交易日空仓，因此回撤小但跑不赢市场。35/38 次运行的平均仓位只有 34–48%：付了交易成本，却只吃到部分市场涨幅。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180007123.png" style="width:85%">

GPT-6 Astra (xhigh) 净值 vs 市场：靠大量空仓控制回撤。

6. **推理强度不是越高越好**：6/9 个模型的最佳 effort 不是最高档。
7. **策略高度同质却选股不同**：风格都偏小盘、低波、价值；信号两两秩相关平均 0.47，但 Top-75 持仓 Jaccard 重叠只有 6.1%。

#### 局限

只有一个封闭年、一个市场；5 次评估的预算偏小；$S_{out}$ 规则不公开；市场基准是不扣费的毛收益；harness 细节和部分交易规则（T+1、涨跌停具体处理、费率）未披露；Kepler-1 不透明。

> 启示：**评估“研究过程 + 冻结后的样本外结果”**，而不是评估单次回测；以及“研究期分数 ≠ 封闭期收益”本身就是一个可研究的 reward hacking 现象。

<br>

### 1.4 KTD-Fin（2026.05）：LLM 交易 Agent 的收益是“知道”还是“会做”？

- 论文：<a href="https://arxiv.org/abs/2605.28359">From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets</a>（arXiv 2605.28359）
- 作者：Taojie Zhu, Wentao Zhao, Rui Sun 等

#### 动机：端到端 LLM 交易回测的两个失效模式

1. **记忆泄漏（Memorization leakage）**：测试期与模型训练语料重叠，模型可以用“记住的”代码、日期、价格、新闻叙事代替推理。
2. **收益不等于能力**：累计收益混合了市场 beta、风格暴露、市场状态与真正的选股能力。

一个动机实验：在可见股票代码时，模型给出“品牌型”理由（如“贵州茅台，白酒龙头”）；同样的数字匿名化后，模型在 25/25 个格子里都选择不交易。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180008123.png" style="width:90%">

KTD-Fin 总览：四种遮蔽条件 → Agent Trader 的 Research → Submission 循环 → A 股执行引擎（T+1、涨跌停、滑点、费用）→ 评估报告（绩效、归因、风控），逐日推进。

#### 基准设计

- **股票池与输入**：CSI 300 成分股；**只给价量**（OHLCV 及衍生技术特征），不给新闻、财报、基本面。
- **评估窗口**：长窗口 2024-01-01 至 2026-04-10（548 个交易日）+ 10 个按市场状态分层的短窗口（每个约 43 日），如 W4 爆发牛市（CSI300 +24.3%）、W2 熊市（−6.5%）、W10 严格晚于锚定模型知识截止。
- **执行**：日频、每步一个 JSON 动作；100 万元起始、只做多、卖出 T+1、次日开盘成交；买 5bp / 卖 15bp / 最低 5 元；按板块的涨跌停；无效订单记录不纠正；引擎基于 Qlib。

**三种决策模式**

| 模式 | Agent 能看到什么 | 测什么 |
|---|---|---|
| memory-only | 只有股票 ID 和账户状态 | 对记忆的依赖 |
| fixed-candidate | 一张整理好的 OHLCV + 因子表 | 从干净信号中选择的能力 |
| open-research | 6 个只读工具（市场概览、筛选候选、个股快照、比较、组合状态、风控检查）| 端到端的 Agent 推理 |

**四级遮蔽（Masking）**

| 级别 | 股票代码 | 日期 |
|---|---|---|
| bright | 真实 | 真实 |
| stock-blind | 匿名化 | 真实 |
| date-blind | 真实 | 相对日序号 |
| blinded | 匿名化 | 相对日序号 |

实现：每个 episode 维护一个别名映射，prompt 中替换标识符；工具调用参数先解遮蔽再查数据，结果再遮蔽后返回；别名在 episode 内稳定、跨 episode 重洗；**数值来自真实序列，只换标识符**。遮蔽有效性用 10 个“攻击者模型”验证：top-1 代码还原率 ≤3.0%，代码 top-5 且日期 ±7 天的联合还原 ≤1.5%。

**Barra 风格归因**

每日做截面 WLS 回归：

$$r_{i,t}=f_{0,t}+\sum_k x_{i,k,t-1}\,\lambda_{k,t}+\varepsilon_{i,t}$$

$f_{0,t}$ 是单位暴露的“共同项”，$x$ 只用 $t$ 之前的信息。组合收益拆成：

$$R^{p}_t=\underbrace{f_{0,t}}_{\text{Common}}+\underbrace{\sum_k\Big(\sum_i w_{i,t-1}x_{i,k,t-1}\Big)\lambda_{k,t}}_{\text{Style}}+\underbrace{\sum_i w_{i,t-1}\varepsilon_{i,t}}_{\text{Selection }\alpha}$$

9 个风格因子（在独立校准窗口上做 VIF 筛选）：MOM_12_1、RV_60、ILLIQ、REV_ON、MOM_ID、SKEW、CORR_PV、HIGH_52W、CV_VOL。

#### 关键结果

**排行榜（blinded × open-research，长窗口；CSI 300 买入持有 +36.92%）**

| 模型 | 总收益 | 超额 | Sharpe | MDD | 换手 |
|---|---|---|---|---|---|
| qwen3.6-plus | +85.29% | +48.37% | 1.13 | −24.49% | 31.88 |
| gpt-5.5 | +61.26% | +24.34% | 0.98 | −24.63% | 14.38 |
| doubao-seed-2.0 | +61.01% | +24.09% | 1.15 | −15.62% | 14.18 |
| claude-opus-4-7 | +58.80% | +21.88% | 0.98 | −26.69% | 31.75 |
| minimax-m2.7 | +56.31% | +19.39% | 0.89 | −29.43% | 43.68 |
| step-3.5-flash | +21.86% | −15.06% | 0.49 | −28.99% | 24.66 |
| gemini-3.1-pro | +8.25% | −28.67% | 0.29 | −49.67% | 101.23 |
| deepseek-v4-pro | +1.39% | −35.53% | 0.19 | −42.35% | 73.49 |
| glm-5.1 | −3.55% | −40.47% | 0.10 | −44.88% | 100.25 |
| kimi-k2.6 | −24.23% | −61.15% | −0.34 | −45.49% | 85.26 |

同窗口 Qlib 基线：SFM +86.58%（Sharpe 2.02，MDD −7.41%）、DoubleEnsemble +73.66%、CatBoost +72.97%、LightGBM +66.22%、Transformer +60.35%、LSTM +43.92%、Linear +29.60%。**最好的 LLM 的 Sharpe（1.13）也低于很多传统 ML 基线。**

**Barra 归因（种子平均累计贡献，%）——这是全文最重要的表**

| 模型 | Common | Style | **选股 α** | 组合 |
|---|---|---|---|---|
| Claude Opus 4.7 | +35.0 | +16.6 | **+0.2** | +51.8 |
| Qwen3.6-Plus | +41.8 | +29.2 | −0.7 | +70.3 |
| Doubao-Seed-2.0 | +42.5 | +15.5 | −4.7 | +53.3 |
| MiniMax-M2.7 | +29.6 | +26.4 | −7.4 | +48.6 |
| GPT-5.5 | +40.0 | +11.8 | −14.9 | +36.9 |
| Gemini-3.1-Pro | +38.3 | +23.6 | −38.5 | +23.4 |
| Kimi-K2.6 | +32.1 | +22.0 | −77.8 | −23.7 |

- **10 个模型里 9 个选股 α 为负**，只有 Opus 4.7 约等于 0。收益几乎全部来自 Common + Style。
- 收益最高的 qwen（+85%）选股 α 是 −0.7%：**高原始收益可以掩盖零甚至负的选股能力**。
- 风格画像：LLM 组偏**高波动**（RV_60 +0.23 vs ML 组 −0.66）和**日内动量**（MOM_ID +1.06 vs −0.12）；ML 模型偏低波与趋势。
- 遮蔽后理由从“品牌叙事”变成“因子排名推理”。

#### 局限

只有 CSI 300、只有价量；完整的“遮蔽 × 模式 × 窗口”网格只在锚定模型上跑，其他 9 个模型只跑 blinded × open-research；每个模型 24–72 小时 API 时间；正文未见 α 的 t 统计量。

> 启示：评估 LLM 交易 Agent 的两条铁律——**遮蔽标识符和日期**，以及**声称 Alpha 前先做收益归因**。

<br>

### 1.5 综述：Agentic Quantitative Trading（2026.08）

- 论文：<a href="https://arxiv.org/abs/2608.31041">arXiv 2608.31041</a>，作者 Fengrui Hua, Hengyi Yang, Xinlei Hao, Haohan Zhang, Bokai Cao, Yiyan Qi, Jia Li, Jian Guo
- 范围：主要是 2025–2026 研究；排除一般的金融 LLM 应用与非 Agent 预测模型。
- “Agentic” 的定义：系统能**分解任务、使用工具、保持状态、通过反馈或反思更新**。

它围绕三个问题组织：Agent 在流水线各阶段如何分布？架构、协调、适应如何串联决策？评测设置与所声称的能力之间是什么关系？

#### 维度一：流水线五阶段（§2）

| 阶段 | 子类 | 代表系统 |
|---|---|---|
| Factor Mining | 生成式 / 协作式 / 自进化 / 工具增强 | Alpha-GPT、FAMA；FactorMAD、AlphaLogics；FactorMiner、AlphaAgentEvo、QuantaAlpha；LLM-MCTS、AlphaPROBE |
| Signal Discovery | Agent 推理信号 / 因子模型信号 / 可执行策略信号 | TradingAgents、FinCon、FinMem；RD-Agent(Q)；QuantAgents、FLAG-Trader、AlphaQuanter |
| Portfolio Construction | 资产选择 / 权重优化 | 3S-trader、MASS；RAPTOR、CEF-Agents |
| Order Execution | 交易指令标准化、执行 Agent | AlphaCrafter、AgenticAITA、PACE |
| Risk Management | 执行前控制 / 基于结果的适应 | Trade-or-not、WebCryptoAgent；FinMem、ATLAS、TrustTrade |

#### 维度二：系统设计三要素（§3）

> 注：该综述的系统设计分类图（Fig. 2）在 arXiv HTML 中的图片文件本身是损坏的，这里用表格重绘。

| 维度 | 类别 | 说明 / 例子 |
|---|---|---|
| (a) 架构 Architecture | Sequential / Iterative / Hierarchical / Orchestration | 流水线式、循环迭代、分层管理、编排者调度；可叠加人机协作（如 FinArena）|
| (b) 协调 Coordination | Direct / Aggregation / Debate / Selection / Gating | 直接传递、聚合多方输出、辩论、择优、显式门控 |
| (c) 适应 Adaptation | Memory / Reflection / Performance feedback / Regime detection | FinMem；FinAgent；FinVision、FinCon；AlphaCrafter |

#### 三个量化发现

1. **信号发现一家独大，完整流水线罕见**：20 个代表系统全部把信号发现作为核心；因子挖掘只有 4 个，执行只有 3 个；只有 2 个系统把五个阶段都当作核心。
2. **协调以“聚合”为主**：17 个多 Agent 系统中 15 个聚合输出；择优 5 个、辩论 3 个、显式门控只有 2 个。
3. **评测设置决定结果含义**：15 个基准分为策略构建（AlphaEval、QuantCode-bench、OQL）、离线交易（InvestorBench、StockAgent、StockBench）、实盘（DeepFund、AI-Trader、LiveTradeBench、Agent Market Arena 等）、可靠性（FinLake-Bench/“Profit Mirage”、AutoRedTrader、KTD-Fin）。实盘基准显示表现取决于风控、流动性与滑点；可靠性基准显示泄漏、错误信息、归因方式都会改变结论。

#### 未来方向

评估从信号到成交的**完整路径**（传统优化器和执行算法可以保留）；在相同设置下比较择优/辩论/门控，并记录决策为何被修改或拒绝；**让评估匹配声明**——控制信息时间、成本、流动性与滑点。“可执行的策略不保证盈利，盈利的回测不证明实盘能力。”

<br>

### 1.6 公式化 Alpha 挖掘谱系：从 RL 到 LLM Agent

Agentic Quant 的“因子挖掘”子方向有一条清晰的演化线，理解它有助于设计 reward：

| 工作 | 生成器 | 奖励 / 目标 | 关键贡献 |
|---|---|---|---|
| 遗传规划（GP，如 gplearn）| 表达式树的交叉变异 | 单因子 IC | 传统基线，易产生大量高相关因子 |
| <a href="https://arxiv.org/abs/2306.12964">AlphaGen</a>（KDD 2023）| PPO 生成 RPN token 序列 | **加入因子池后线性组合的 IC** | 奖励“协同性”而非单因子 IC |
| <a href="https://arxiv.org/abs/2308.00016">Alpha-GPT</a>（2023）| LLM 将研究员想法转为因子 | 人机交互 | 人机协作挖掘范式 |
| <a href="https://arxiv.org/abs/2406.18394">AlphaForge</a>（AAAI 2025）| 生成-预测网络 | IC + 多样性；**每日动态重加权** | 因子动物园 + 动态 Mega-Alpha |
| <a href="https://arxiv.org/abs/2502.16789">AlphaAgent</a>（KDD 2025）| Idea / Factor / Eval 三 Agent | IC − **原创性/复杂度/对齐正则** | 用 AST 子树相似度对抗因子拥挤与衰减 |

**AlphaGen**（Yu et al., KDD 2023，<a href="https://github.com/ICT-FinD-Lab/alphagen">GitHub</a>）：因子写成逆波兰表达式（RPN），PPO 策略逐 token 生成（action masking 保证 RPN 合法，策略和价值网络共享 LSTM）；无中间奖励，episode 结束时把新因子加入池子，以**线性组合模型的 IC** 为回报。注意这让 MDP 非平稳（池子在变）。

**AlphaForge**（Shi et al., AAAI 2025，<a href="https://github.com/DulyHao/AlphaForge">GitHub</a>）：

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180019123.png" style="width:90%">

AlphaForge：(A) 生成-预测网络挖掘因子进入 Factor Zoo；(B) 每个交易日按近期表现重新选择并加权因子，组合成 Mega-Alpha（不同于 AlphaGen 的固定权重）。

**AlphaAgent**（Tang et al., KDD 2025，<a href="https://arxiv.org/abs/2502.16789">arXiv 2502.16789</a>）——与 Agentic RL reward 设计最相关：

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180017123.png" style="width:85%">

AlphaAgent：Idea Agent、Factor Agent、Eval Agent 构成闭环，平衡金融逻辑、原创性与适应性，以对抗 alpha decay。

目标函数：

$$f^*=\arg\max_f\ \mathcal{L}(f(X),y)-\lambda\,\mathcal{R}_g(f,h),\qquad \mathcal{R}_g=\alpha_1\,SL(f)+\alpha_2\,PC(f)+\alpha_3\,ER(f,h)$$

- $SL$：符号长度；$PC$：自由参数个数（惩罚复杂度）；
- 探索正则 $ER=\beta_1 S(f)+\beta_2\mathcal{C}(h,d,f)+\beta_3\log(1+|\mathcal{F}_f|)$；
- **原创性** $S(f)=\max_{\phi\in\mathcal{Z}}s(f,\phi)$，其中 $s$ 是两个因子 AST 的**最大同构公共子树**大小——惩罚与因子动物园中已有因子“换汤不换药”；
- **对齐** $\mathcal{C}$：由 LLM 打分，检查“假设 $h$ → 描述 $d$ → 表达式 $f$”是否一致；
- 最后一项惩罚使用的原始特征数量。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180018123.png" style="width:60%">

原创性度量：新因子 AST 与因子库中 AST 的最大公共子树。

报告结果：CSI 500 年化 11.00%、S&P 500 年化 8.74%。（注意：其 GitHub 仓库当前 README 描述的是一个重写后的 CSI-1000 多因子框架，与论文实验代码不完全对应。）

<br>

### 1.7 小结：Agentic Quant 的真正研究难点

核心问题可以写成一句话：

> **How can an AI agent discover statistically valid, economically meaningful, and out-of-sample robust alpha signals?**

结合以上论文，可以细化为：

| 难点 | 现象（论文证据） | 可能的研究方向 |
|---|---|---|
| 多重检验偏差 | Quant-Arena：迭代后研究期分数涨、封闭期收益不涨（−12.4 分）| 把试验次数纳入奖励（Deflated Sharpe、PBO，见第 5 节）|
| 回测过拟合 / reward hacking | RankIC 与 ARR 相关仅 0.09；“驼峰”分组收益 | 冻结-封闭评估；隐藏评分规则；多窗口稳健性 |
| 新信号 vs 旧因子变种 | RD-Agent(Q) 用 IC≥0.99 去重；AlphaAgent 用 AST 子树 | 增量 IC、对已知因子正交化后的残差 IC |
| 收益来源是否是 Alpha | KTD-Fin：9/10 选股 α 为负 | 以归因后的选股 α 为奖励 |
| LLM 记忆泄漏 | KTD-Fin：可见代码时讲品牌故事 | 遮蔽 + 晚于知识截止的测试 |
| 从失败中学习 | RD-Agent(Q) 知识库存失败三元组 | 显式的负面实验记忆（见第 7 节课题 C）|

**Reward 不能简单等于历史收益或 Sharpe**——这是 Quant Agent 与代码/数学 Agent 最大的区别。

<br>

## 2. Financial Foundation Models：金融时间序列基础模型

> TIME-SERIES FOUNDATION MODELS · SELF-SUPERVISED LEARNING

核心问题：**能否像 LLM 学语言规律一样，让基础模型从海量金融时间序列中学习市场的通用表示？** 过去用 XGBoost、LSTM、Transformer 针对单一数据集训练；现在尝试跨市场、跨资产预训练，再零样本预测或微调。

### 2.1 通用 TSFM 三巨头

| 模型 | 机构 | 论文 | 核心思想 |
|---|---|---|---|
| TimesFM | Google | <a href="https://arxiv.org/abs/2310.10688">arXiv 2310.10688</a> | **Patched-decoder**：把序列切成 patch，decoder-only Transformer 自回归预测下一个（可以更长的）输出 patch |
| Chronos | Amazon | <a href="https://arxiv.org/abs/2403.07815">arXiv 2403.07815</a> | **缩放 + 量化**把数值变成固定词表 token，直接用 T5 语言模型架构（20M–710M）做交叉熵训练；用高斯过程合成数据（KernelSynth）增强 |
| Moirai | Salesforce | <a href="https://arxiv.org/abs/2402.02592">arXiv 2402.02592</a> | **Masked encoder**，解决跨频率、任意变量数（any-variate attention）、分布多样（混合分布输出）三个问题；在 27B+ 观测的 LOTSA 上训练 |

关键提醒：**通用 forecasting leaderboard 成绩 ≠ 交易能力**。金融收益的信噪比极低，MSE/MAE 意义上的“好预测”往往就是“预测接近 0”。

<br>

### 2.2 Kronos（AAAI 2026）：面向 K 线的基础模型

- 论文：<a href="https://arxiv.org/abs/2508.02739">arXiv 2508.02739</a>；<a href="https://ojs.aaai.org/index.php/AAAI/article/view/39730">AAAI 2026</a>
- 代码：<a href="https://github.com/shiyu-coder/Kronos">shiyu-coder/Kronos</a>（MIT），权重：<a href="https://huggingface.co/NeoQuasar">HuggingFace NeoQuasar</a>

#### 动机

通用 TSFM 在 K 线数据上经常**不如不预训练的专用模型**，而且忽略了波动率预测、合成数据生成等金融关键下游任务。Kronos 要做“统一、可扩展、专为 K 线设计”的预训练框架。

<img src="https://raw.githubusercontent.com/shiyu-coder/Kronos/master/figures/overview.png" style="width:90%">

Kronos 两阶段框架：(1) K 线 Tokenizer（BSQ 量化，粗/细两级子 token）；(2) decoder-only Transformer 自回归预训练。

#### 第一阶段：K 线 Tokenizer（BSQ + 层级子 token）

- **输入**：每个时间步是 6 维向量 OHLCVA（开、高、低、收、成交量、成交额），逐实例 z-score 归一化并裁剪到 $[-5,5]$。
- **结构**：Transformer 自编码器 $E_{enc}\to Q\to E_{dec}$（编码器/解码器各 3 层，d_model 256）。
- **量化器**：Binary Spherical Quantization（BSQ，LFQ 的变体）把连续潜变量 $\xi_t$ 投影到可学习超平面，得到 $k$ 位二值码 $b_t\in\{-1,1\}^k$，Kronos 用 $k=20$（词表等价于 $2^{20}$）。
- **层级分解**：把码拆成粗、细两半 $b_t=[b_t^c,b_t^f]$，各 10 位——一次 $2^{20}$ 分类变成两次 $2^{10}$ 分类。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180010123.png" style="width:85%">

K 线 Tokenizer：Transformer 自编码器 + BSQ 量化层，粗子 token 学主结构，细子 token 编码残差。

Tokenizer 损失：

$$\mathcal{L}_{tokenizer}=\underbrace{\mathbb{E}\big[\lVert x-E_{dec}(b^c)\rVert^2\big]}_{\mathcal{L}_{coarse}\text{：只用粗码重建}}+\underbrace{\mathbb{E}\big[\lVert x-E_{dec}(b)\rVert^2\big]}_{\mathcal{L}_{fine}\text{：完整码重建}}+\lambda\,\mathcal{L}_{quant}$$

$\lambda=1$。码本利用率：粗 97.66%，细 85.25%。有意思的是层级损失并**没有**改善重建（MAE 0.0785 vs 标准损失 0.0781），它的作用是为自回归模型制造“先粗后细”的依赖。

为什么要拆？对 Kronos-base，不拆时词表参数 1744.8M、总参数 1842.3M；拆成 2 级后总参数 102.3M，代价是推理步数 ×2。

#### 第二阶段：自回归预训练

- 骨干：因果 Transformer（RoPE、Pre-LN、RMSNorm）。
- 因式分解似然：

$$p(b_t\mid b_{<t})=p(b_t^c\mid b_{<t})\cdot p(b_t^f\mid b_{<t},b_t^c)$$

- 输入融合 $v_i=W_{fuse}([e_c(b_i^c);e_f(b_i^f)])$，再加分钟/小时/星期/日/月时间嵌入。
- 粗 head 是线性层；细 head 用交叉注意力（query 为粗 token 嵌入，key/value 为隐状态）。训练时细步以**采样的**粗预测为条件而非真值，以缓解 exposure bias。
- 目标：$\mathcal{L}_{ar}=-\mathbb{E}\big[\sum_t \log p(b_t^c\mid b_{<t})+\log p(b_t^f\mid b_{<t},b_t^c)\big]$。
- 推理：温度 $T$ + top-p 采样，可采样多条路径取平均（sample_count 越大 IC 越高，算力越大）。

#### 模型规模与数据

| 模型 | 层数 | d_model | 参数 | 上下文 | 开源 |
|---|---|---|---|---|---|
| Kronos-mini | – | – | 4.1M | 2048 | ✓（仅 README 中出现）|
| Kronos-small | 8 | 512 | 24.7M | 512 | ✓ |
| Kronos-base | 12 | 832 | 102.3M | 512 | ✓ |
| Kronos-large | 18 | 1664 | 499.2M | 512 | ✗ |

预训练数据：**超过 120 亿条 K 线**（附录：约 96,569 个资产、12.11B 观测），覆盖股票、加密货币、期货、外汇（后三者上采样），频率从分钟到周；预训练截止 2024-06，评估从 2024-07 开始。

#### 结果（零样本）

| 任务 | Kronos | 最强对比 | 说明 |
|---|---|---|---|
| 价格序列预测 RankIC | large 0.0267 | Time-MoE-B 0.0138；TimesNet 0.0143 | 即摘要中的 +93% / +87% |
| 收益预测 RankIC | large 0.0675 | Moirai ≈0.053；DLinear 0.0423 | |
| 已实现波动率 MAE | large 0.0370 | GARCH 0.0405 | 约低 9% |
| 合成 K 线 | 判别分数、TSTR、t-SNE/KDE 多样性 | DiffusionTS / TimeVAE / TimeGAN | 摘要称保真度提升 22% |
| 投资模拟（CSI300+800 平均，0.15% 成本）| large AER 0.2084 / IR 1.65 | Moirai-large 0.1577 / 1.25；Chronos-large −0.0302 | |

消融（Kronos-small）：直接自回归（Direct-AR）价格 RankIC 0.0149 → 并行粗细（Kronos-Parallel）0.0226 → **顺序粗→细 0.0254**，说明层级依赖有效；更大的词表同时提升重建与预测；模型越大各任务越好（scaling）。

#### 使用

```python
from model import Kronos, KronosTokenizer, KronosPredictor
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)
pred_df = predictor.predict(
    df=x_df,                  # open, high, low, close (+ volume, amount)
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0, top_p=0.9,         # 采样温度与 nucleus 概率
    sample_count=1            # 采样路径数，多条取平均
)
```

#### 批判性阅读

- 绝对 IC 很小（价格 RankIC≈0.027）；回测只在 A 股；消融只用 small；large 不开源；上下文只有 512。
- “45 个交易所”与附录“40+ 交易所、30+ 国家”不一致；“7 种粒度”也需谨慎引用。
- 论文没有专门的局限性章节，也明确说 K 线是否包含短期价格信息“超出本文范围”。

<br>

### 2.3 反面证据：TSFM 对金融收益预测到底有多大帮助？

#### Noguer i Alonso & Franklin（2026.06）：小而稀疏的改进

- 论文：<a href="https://arxiv.org/abs/2606.27100">Pretrained Time-Series Foundation Models for Financial Return Forecasting</a>（arXiv 2606.27100）
- 设定：5 只美股（AAPL、AMZN、GOOG、JPM、META）日频，2014-09 至 2026-02；测试期 2024-08 至 2026-02；线性与对数收益共 10 个任务；预测期 $H=20$ 个交易日，统一上下文 $L=512$，滚动 10 个窗口。
- 模型：零样本 TSFM（TimeGPT、TimesFM-2.5、Moirai-2.0、Chronos v1、Chronos-2）vs 逐股训练的神经网络（NBEATS、NHITS、PatchTST、iTransformer、KAN）+ AR(1)、EWMA。
- 指标：MAE；相对零收益 random walk 的 skill score；单侧 **Diebold–Mariano 检验**（HLN 修正）。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180011123.png" style="width:70%">

一个 20 日预测窗口内的实际值 vs 预测轨迹（示意：收益预测几乎贴近 0 附近）。

结果：
- 平均排名 Moirai-2.0（2.9）> TimesFM-2.5（3.1）> Chronos v1（3.4）> Chronos-2（3.9）> iTransformer（4.3）；**预训练模型拿下 8/10 个任务**。
- 但 5 只股票中 3 只，连获胜模型相对 random walk 的 skill 都是**负的**（如 AAPL TimesFM-2.5 −0.032）。
- DM 检验只有 **2 次拒绝原假设**：Chronos v1 on AMZN（p=0.042）、Moirai-2.0 on GOOG（p=0.042）。
- 作者结论：TSFM 是“降低低数据金融预测开发成本的实用先验”，而**不是**“统计上可靠的 alpha 生成引擎”。

#### Rahimikia, Ni & Wang（2025.11）：要用金融数据预训练

- 论文：<a href="https://arxiv.org/abs/2511.18578">Re(Visiting) Time Series Foundation Models in Finance</a>
- 设定：次日超额收益预测；CRSP + Compustat Global，1990–2023，94 个国家；样本外 2001–2023 扩展窗口。
- 结果：零样本 Chronos-large OOS $R^2=-1.37\%$，TimesFM-500M $-2.80\%$；**从头用金融数据预训练**的 Chronos-small 年化 36.84%、Sharpe 5.42，加全球数据与因子后 41.89% / 6.78；但 CatBoost（全球数据）仍有 47.25% / 6.46。
- 结论：**现成 TSFM 表现差，金融原生预训练是必要的**，但即便如此拟合优度仍落后于传统基准——这正好支持 Kronos 式“领域专用预训练”的路线，同时提醒不要高估。

<br>

### 2.4 真正前沿的问题

- 如何将金融的**非平稳性**融入预训练目标（而不是假设平稳的通用时间序列）？
- 如何学习跨股票、债券、期权、宏观数据的**统一表征**？
- 如何在 regime shift 后做 **online adaptation** 而不发生 catastrophic forgetting？
- 应该预测收益率，还是预测**未来收益分布、波动率或极端风险**？（Kronos 的波动率结果比收益结果更有说服力，这是一个线索。）
- 评估上：要用 DM 检验、相对 random walk 的 skill、扣费后的组合表现，而不是 MSE leaderboard。

<br>

## 3. Market World Models：学习一个可以交互的金融市场

> GENERATIVE WORLD MODELS · DIFFUSION · COUNTERFACTUAL SIMULATION

### 3.1 形式化：从预测模型到世界模型

普通预测模型学习：

$$P(s_{t+1}\mid s_t)$$

可交互的市场 World Model 要学习：

$$P(s_{t+1},r_t\mid s_t,a_t)$$

- $s_t$：当前市场状态（价格、买卖盘、流动性）；
- $a_t$：交易 Agent 的行动（如提交大额买单）；
- $s_{t+1}$：交易发生后市场的新状态；
- $r_t$：策略执行后的收益或执行成本。

区别在于：它不仅预测市场会怎么变化，还要**模拟我的交易行为如何改变市场**。

### 3.2 为什么需要？回测的根本缺陷

假设 Agent 要买 1000 万美元股票。普通回测（market replay）直接用历史成交价，但真实市场中大单会改变买卖价差、可用流动性和成交价。世界模型要回答：

- 现在买入会产生多少 **market impact**？
- 拆成十次买入，执行成本如何变化？
- 市场突然进入高波动，策略能否存活？
- 多个交易 Agent 同时行动，会不会产生拥挤交易甚至不稳定？

三种模拟范式的对比：

| 范式 | 能否对我的交易做出反应 | 真实性 | 问题 |
|---|---|---|---|
| Market Replay（历史回放）| ✗ | 最真实（就是历史）| 冲击只能用外生模型叠加 |
| Agent-Based（如 ABIDES 的 IABS）| ✓ | 低：交易者群体过度简化、难以校准 | 个体交易数据不公开 |
| 生成式世界模型（TRADES、MarS）| ✓ | 学自数据 | **统计真实 ≠ 因果反应正确** |

<br>

### 3.3 TRADES（ECAI 2025）：Transformer 扩散生成订单流

- 论文：<a href="https://arxiv.org/abs/2502.07071">arXiv 2502.07071</a>，Berti, Prenkaj, Velardi
- 代码：<a href="https://github.com/LeonardoBerti00/DeepMarket">DeepMarket</a>

<img src="https://raw.githubusercontent.com/LeonardoBerti00/DeepMarket/main/data/architecture.jpg" style="width:90%">

TRADES 架构：对最后一个订单加噪；订单与 LOB 快照分别经 MLP 升维后拼接，加扩散步与位置嵌入，经 Transformer 编码器输出噪声估计 $\epsilon_\theta$ 与方差插值系数 $v$。

#### 生成什么、以什么为条件

- 每个订单是 6 元组 $(p,q,d,b,\delta,c)$：价格、数量、方向、深度（与最优价的距离）、**与上一订单的时间间隔**、类型（市价/限价/撤单）。
- 条件 $s$：前 $N-1=255$ 个订单 + 前 $N$ 个 LOB 快照（10 档买卖价量）。每步生成 1 个订单，滑窗自回归。

#### 扩散公式（只对生成部分 $g$ 加噪，条件部分 $c$ 保持干净）

$$x_t^g=\sqrt{\bar\alpha_t}\,x_0^g+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)$$

$$\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\Big(x_t^g-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t^g,x_0^c,t)\Big),\qquad \Sigma_\theta=\exp\big(v\log\beta_t+(1-v)\log\tilde\beta_t\big)$$

$$\mathcal{L}=\mathbb{E}\lVert\epsilon-\epsilon_\theta\rVert^2+\lambda\,\mathcal{L}_{vlb}$$

$T=100$ 扩散步，学习方差沿用 Nichol & Dhariwal（Improved DDPM）。

#### 与 ABIDES 结合：能对 Agent 做出反应

作者在 ABIDES 中加入一个“world agent”，加载训练好的生成模型持续产生订单，其他交易 Agent 可同时运行（深度学习 + ABM 混合）。**市场冲击实验**：一个 POV 买入 Agent 每分钟醒来买入直到成交 φ 股——在 market replay 下冲击只是瞬时的、很快消失；在 TRADES 下价格出现**永久偏移**，与实证观察一致。

#### 数据与结果

- 数据：LOBSTER，NASDAQ 的 TSLA 与 INTC，2015 年 1 月约 20 个交易日；17 天训练、1 天验证、最后 2 天模拟（10:00–12:00，前 15 分钟用真实订单热启动）。
- 指标：**predictive score**（用合成数据训练中间价预测器、在真实数据上测 MAE，越低越好）、PCA 覆盖率、stylized facts（收益自相关、量-波动相关、收益分布、波动聚集等）。

| 方法 | TSLA MAE | INTC MAE |
|---|---|---|
| Market Replay（真实数据上限）| 0.923 | 0.149 |
| IABS（ABIDES：5000 噪声 + 100 价值 + 25 动量 + 1 做市）| 1.870 | 1.866 |
| CGAN（Coletta et al.）| 3.453 | 0.699 |
| **TRADES** | **1.213** | **0.307** |

相对次优方法（扣除 Market Replay 后的差距）提升 ×3.27（TSLA）和 ×3.48（INTC）；PCA 覆盖 67.04%。消融：去掉 LOB 条件、换 LSTM、换交叉注意力都明显变差；DDIM 一步采样快 100 倍但精度大降（TSLA 1.213→3.146）。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180012123.png" style="width:75%">

TSLA 2015-01-29 的 stylized facts：收益自相关、量/收益-波动相关、收益分布、波动聚集、中间价路径。

**DeepMarket 框架**：LOBSTER 预处理、PyTorch Lightning 训练、TRADES/CGAN 实现与权重、扩展版 ABIDES（`python -u ABIDES/abides.py -c world_agent_sim ...`），并发布合成数据集 **TRADES-LOB**（TSLA/INTC 两天，50 列，约 26.6 万行）。

**局限**：只有两只股票、每只单独训练一个模型；数据老（2015）；生成慢（约 6 小时算力模拟 1 小时市场）；predictive score 是代理指标；不证明任何策略盈利。

<br>

### 3.4 DiffLOB（IJCAI 2026）：订单簿的反事实生成

- 论文：<a href="https://ijcai.org/proceedings/2026/730">IJCAI 2026 AI4Tech track</a>；<a href="https://arxiv.org/abs/2602.03776">arXiv 2602.03776</a>，Zhuohan Wang, Carmine Ventre（King's College London）
- 代码：<a href="https://github.com/ZhuoHan1998/DiffLOB">ZhuoHan1998/DiffLOB</a>

#### 动机

现有 LOB 生成器是**被动的**：复现“典型”行为。想探索另一种结局，只能与交易 Agent 交互，也无法保证到达指定的未来市场状态——这限制了压力测试与情景分析。DiffLOB 的问题是：**“如果未来市场状态是 X 而不是 Y，订单簿会怎么演化？”**

#### 生成什么、怎么控制

- 输出：给定过去轨迹 $x_{1:t}$，生成未来 LOB 轨迹 $x_{t+1:t+\tau}$，每个状态是 10 档买卖价量；LOBSTER 每秒采样，$t=\tau=32$。
- 表示：价格转为中间价一阶差分 + 档间差（去掉绝对价格）；成交量截断到 99 分位后做 $\sqrt{v}/c$。
- **可干预的 Regime 变量**（由目标未来窗口计算，$m$ 为中间价，$r_t=m_{t+1}-m_t$）：

$$c^{trend}=\sum_{i=0}^{\tau-1}r_{t+i},\qquad c^{vol}=\sqrt{\tfrac1\tau\textstyle\sum r^2-\big(\tfrac1\tau\sum r\big)^2}$$

$$c^{liq}_t=\sum_k\big(v^{a,(k)}_t+v^{b,(k)}_t\big),\qquad c^{imb}_t=\frac{\sum_k(v^{a,(k)}_t-v^{b,(k)}_t)}{\sum_k(v^{a,(k)}_t+v^{b,(k)}_t)}$$

  趋势、波动是标量（全局条件）；流动性、订单流不平衡是长度 $\tau$ 的序列（局部条件）。

<img src="https://raw.githubusercontent.com/ZhuoHan1998/DiffLOB/main/docs/assets/DiffLOB_Architecture.png" style="width:90%">

DiffLOB 架构：WaveNet 式门控残差块，通过 FiLM 注入扩散步、局部与全局 regime 条件；外加 ControlNet 式控制分支。

#### 方法

- **Score-based 扩散**（VP-SDE）：$dx=-\tfrac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dw$，加权去噪 score matching 训练，100 个噪声级的 ancestral sampling。
- **Classifier-free guidance**：训练时以 0.5 概率丢弃条件；采样时 $s=(1+\omega)s_c-\omega s_\varnothing$。
- **ControlNet 式控制模块**：通过零初始化的 1×1 卷积加性注入控制信号（初始化时不影响主干）。
- **两阶段训练**：先训主干与编码器，再冻结主干只训控制模块。
- 骨干：16 个 WaveNet 门控残差块，价格档位嵌入区分深度。

#### 三段式评估

1. **可控真实性**：以观测到的 regime 为条件，用 KS / Wasserstein / KL / JS 比较价格、成交量统计，以及多期收益、价差分布、波动聚集等。
2. **反事实有效性**：强制施加高/低 regime，与真实数据中该 regime 分布前/后 20% 的轨迹比较。
3. **反事实有用性**：用 Real、Real×2、Real+CF 三种训练集训练趋势分类器和流动性回归器，在高/低 regime 子集上评估。

数据：LOBSTER 的 AMZN / AAPL / GOOG，2023 年 2 月（16 天训练）。基线：Diff-CSDI、Diff-S4、cGAN、cVAE、AR（S4 自回归）。

| AMZN（表 1）| 价格 KS | 价格 W | 成交量 KS | 成交量 W |
|---|---|---|---|---|
| **DiffLOB** | 0.0524 | 0.0284 | 0.1092 | 97.54 |
| DiffLOB w/o 控制 | 0.2124 | 0.1828 | 0.1131 | 117.37 |
| cGAN | 0.0535 | 0.0489 | 0.2885 | 321.05 |
| AR | 0.1180 | 0.0762 | 0.3972 | 444.22 |

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180013123.png" style="width:80%">

AMZN 反事实价格轨迹：在不同强制 regime 条件下生成的价格路径。

**读的时候要注意**：DiffLOB 并非处处最好——AAPL 价格 KS 上 cVAE 更好；反事实有效性中 AMZN 高趋势 KS 上 cGAN（0.0487）远好于 DiffLOB（0.444）；“有用性”提升不均匀，若干 $R^2$ 仍为负。反事实**没有 ground truth**，只能用与真实极端样本的分布相似度判断；且它生成的是 **LOB 快照轨迹而非订单流**，不能直接接入撮合引擎做交互式冲击实验（这是笔者的推断）。

<br>

### 3.5 MarS（ICLR 2025）：订单级的生成式市场基础模型

- 论文：<a href="https://arxiv.org/abs/2409.07486">arXiv 2409.07486</a>，Microsoft Research Asia
- 代码：<a href="https://github.com/microsoft/MarS">microsoft/MarS</a>，主页 <a href="https://mars-lmm.github.io/">mars-lmm.github.io</a>

<img src="https://raw.githubusercontent.com/microsoft/MarS/main/doc/img/high-level-overview.png" style="width:90%">

MarS 总览：Large Market Model（LMM）生成订单 → 模拟撮合 → LOB 反馈回模型；支持用户注入交互订单与条件控制，服务于预测、风险检测、RL 训练等下游任务。

**Large Market Model（LMM）由三部分组成：**
- **Order Model**：LLaMA2 式因果 Transformer；每个订单 = (类型∈{Ask,Bid,Cancel}, 价格, 数量, 间隔)，与之前的 LOB 状态一起 token 化（价格/数量离散为 32 档，间隔 16 档）。
- **Order-Batch Model**：把每分钟的订单批次变成 32×32×3 的“订单图像”，VQGAN 编码成 64 个 token，再自回归建模。
- **Ensemble Model**：交叉注意力融合二者。

规模：Order Model 1M → 1.02B 参数，Order-Batch Model 150M → 3B，验证了 scaling law；数据为 2017–2023 年 A 股流动性最好的 500 只股票。

**模拟机制**：生成订单在模拟交易所撮合，结果 LOB 反馈给模型；用户可注入交互订单；可控生成用“从所有可能的未来中选最好的”（采样多个候选未来，选最匹配控制信号的）。

结果：复现 stylized facts；**模拟出的市场冲击服从平方根律**；加入控制后与目标价格曲线的相关从 0.23 升到 0.47（加入交互后为 0.33）；趋势预测优于 DeepLOB；在 MarS 中训练的 RL Agent 从约 −6bp 提升到 2–6bp。开源 2M/5M/10M 的 order model（更大模型未开放）。

<br>

### 3.6 其他相关工作

| 工作 | 要点 |
|---|---|
| <a href="https://arxiv.org/abs/2110.13287">Coletta et al. CGAN</a>（ICAIF 2021）| 条件 GAN 生成订单，作为 ABIDES 中的 world agent；TRADES 的直接前身与基线 |
| <a href="https://arxiv.org/abs/2309.00638">LOBS5</a>（Nagy et al., ICAIF 2023）| 基于 S5 状态空间层的 token 级订单消息自回归模型，生成消息由 JAX-LOB 撮合；后续 <a href="https://arxiv.org/abs/2502.09172">LOB-Bench</a> |
| <a href="https://arxiv.org/abs/2309.07708">Market-GAN</a>（AAAI 2024）| 带语义上下文控制的金融时间序列生成；**注意是日频数据，不是 LOB** |

### 3.7 世界模型方向的核心陷阱与研究问题

| | TRADES | DiffLOB | MarS |
|---|---|---|---|
| 生成对象 | 单个订单（订单流）| LOB 快照轨迹 | 订单 + 分钟级订单批次 |
| 模型 | Transformer DDPM | Score-based + CFG + ControlNet | 自回归 Transformer + VQGAN |
| 可交互（接撮合引擎）| ✓（ABIDES world agent）| ✗ | ✓（模拟交易所）|
| 可控 / 反事实 | ✗ | ✓（趋势、波动、流动性、不平衡）| ✓（文本/曲线条件 + 选优）|
| 规模 | 2 只股票，单独训练 | 3 只股票 | 500 只股票，最大 3B |

最有意思的问题：

> **Can we train a trading agent inside a learned market world model and transfer its policy to previously unseen market regimes?**

但技术陷阱很大：**生成出统计上很真实的市场，不代表生成出的市场对 Agent 行动有正确的因果反应。** Stylized facts 和分布距离测的是 $P(s_{t+1}\mid s_t)$，而 RL 训练需要的是 $P(s_{t+1}\mid s_t,a_t)$ 在**分布外动作**上也正确。目前的证据只有“冲击方向/形状大致合理”（TRADES 的永久冲击、MarS 的平方根律），离“可以放心在里面训练策略”还有距离——这正是值得研究的核心。

<br>

## 4. Reinforcement Learning for Trading：不只是优化利润

> OFFLINE RL · RISK-SENSITIVE RL · MODEL-BASED RL

传统 RL Trading 的目标是最大化累计收益：

$$\max_\pi \mathbb{E}_\pi\Big[\sum_{t=0}^{T}\gamma^t r_t\Big]$$

但在 Quant 中这远远不够：还必须考虑尾部风险、市场冲击、交易成本、仓位限制和策略稳定性。

### 4.1 入口综述：ACM CSUR 2025（167 篇）

- 论文：Pippas, Ludvig, Turkay（University of Warwick），<a href="https://dl.acm.org/doi/10.1145/3733714">The Evolution of Reinforcement Learning in Quantitative Finance: A Survey</a>，ACM Computing Surveys 57(11), 2025；<a href="https://arxiv.org/abs/2408.10932">arXiv 2408.10932</a>

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180014123.png" style="width:85%">

综述图 2：把量化金融中的概念与技术映射到 RL 框架的各组件（Agent、Environment、State、Action、Reward）。

**文献统计**：167 篇，1996–2022；只有 27.5% 早于 2016（DQN 之后爆发）；排除 bandit 类与纯理论。按算法计数：DQN 37、Q-learning 36、RRL 31、DDPG 14、PPO 14、PG 13……**Model-based 只有 3 篇**，是最少的一类。

**方法分类**：
- Value-based（critic-only）：离散动作（买/持/卖）是主要限制；
- Policy-based（actor-only）：以 Moody 一系的 RRL 为主，支持连续权重，但梯度方差大；
- Actor-critic：作者认为在量化中**研究不足**；
- Model-based：样本效率高但计算贵；
- MARL：横切类别。

**应用分类**：组合管理、期权/对冲、订单执行、做市、端到端交易、资产配置、智能投顾。

**主要发现**：
- RL 相对“先预测再决策”的两段式监督学习的优势：可以**直接优化 Sharpe 等风险调整指标**，并把仓位约束和交易成本纳入同一框架。
- 市场是部分可观测的，应建模为 POMDP；LSTM/RNN 表现优于基于 Markov 假设的模型。
- 波动率作为输入特征被“出人意料地忽视”；OHLC 特征高度相关可能反而引入噪声。
- DQN 设定下基于 Sharpe 的奖励优于基于利润的奖励。
- 建议“离线预训练 + 在线微调”的混合方案。

**评测陷阱（§8，非常重要）**：
- **Sharpe 过高要警惕**：经验上 SR > 3 已经罕见；作者点名质疑了报告 SR 21.2 和 SR 217.68 的论文，认为可能来自过拟合、data snooping 或用同时刻价格决策的 look-ahead。
- 很多研究没有用 Fama-French 等因子检验所谓 alpha。
- 只在单只“成功”股票上测试（幸存者偏差）。
- 简化假设：无冲击、无风险厌恶、固定成本、忽略 bid/ask；执行方向多数假设立即成交；做市方向多数用模拟数据。
- 基准不公平（RL 与基准用不同输入）。

**开放问题**：sim-to-real gap、数据短且稀疏、非平稳 + 部分可观测、延迟的信用分配、可解释性、探索本身会产生交易成本。

<br>

### 4.2 Risk-Sensitive RL：在收益之外优化风险

目的：避免 Agent 通过承担极高尾部风险获得表面上的高期望收益。

**Differential Sharpe Ratio**（Moody & Saffell，<a href="https://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf">NeurIPS 1998</a>；<a href="https://doi.org/10.1109/72.935097">IEEE TNN 2001</a>）：Sharpe 不能按步分解，于是用指数移动平均估计一二阶矩：

$$A_t=A_{t-1}+\eta(R_t-A_{t-1}),\qquad B_t=B_{t-1}+\eta(R_t^2-B_{t-1})$$

对 Sharpe 关于 $\eta$ 做一阶展开，得到每步奖励（文献通行形式）：

$$D_t=\frac{B_{t-1}\Delta A_t-\tfrac12A_{t-1}\Delta B_t}{\big(B_{t-1}-A_{t-1}^2\big)^{3/2}}$$

这让 RRL 可以**在线优化 Sharpe**；实验中优化 DSR 比优化利润更稳定，两者都优于用 MSE 训练的预测型系统。

**CVaR 目标**：Rockafellar–Uryasev 形式（对损失 $Z$，置信水平 $\alpha$）：

$$\mathrm{CVaR}_\alpha(Z)=\min_\nu\Big\{\nu+\frac{1}{1-\alpha}\mathbb{E}\big[(Z-\nu)^+\big]\Big\}$$

方法论基础：<a href="https://arxiv.org/abs/1404.3862">Tamar et al. Optimizing the CVaR via Sampling</a>（CVaR 梯度的采样估计）；<a href="https://arxiv.org/abs/1512.01629">Chow et al. Risk-Constrained RL with Percentile Risk Criteria</a>（CVaR 约束下的拉格朗日 PG/actor-critic）。注意这两篇的实验并不在金融上，金融应用多见于组合优化论文（如同时优化方差、CVaR、EVaR 的 PPO）。

<br>

### 4.3 Offline RL：只能用历史数据学

金融不能像游戏一样任意尝试危险操作，只能利用历史交易数据学习。核心挑战：**distribution shift** 与**分布外动作价值高估**（Q 函数在数据没覆盖的动作上乱估，策略就会钻空子）。

**CQL**（<a href="https://arxiv.org/abs/2006.04779">Kumar et al. 2020</a>）：在 Bellman 误差外加一项压低分布外动作 Q 值的正则：

$$\min_Q\ \alpha\,\mathbb{E}_s\Big[\log\sum_a e^{Q(s,a)}-\mathbb{E}_{a\sim\hat\pi_\beta}Q(s,a)\Big]+\tfrac12\,\mathbb{E}\big[(Q-\hat{\mathcal{B}}^\pi\hat Q)^2\big]$$

**ORDC：交易执行中的过拟合**（Zhang et al.，<a href="https://arxiv.org/abs/2307.11685">Towards Generalizable RL for Trade Execution</a>）：

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180015123.png" style="width:65%">

ORDC（offline RL with dynamic context）示意：市场“上下文”变量不受策略影响、只能离线收集。

- 作者发现很多 RL 执行方法明显过拟合，将问题建模为 **offline RL with dynamic context**：上下文（市场变量）不受策略影响。
- 推导泛化界：过拟合来自“上下文空间大、上下文样本少”；对策是学**紧凑的上下文表示**（用先验知识，或端到端学习）。
- 搭建了基于历史 LOB 的高保真模拟器作为基准。

<br>

### 4.4 订单执行（Optimal Execution）

问题：在时间 $T$ 内卖出 $X$ 股，最小化执行成本（implementation shortfall）。

**Almgren–Chriss（2000）经典模型**：在永久冲击和临时冲击下最小化 $E(x)+\lambda V(x)$，其中 $V(x)=\sigma^2\sum_k\tau x_k^2$。线性冲击下的最优持仓轨迹：

$$x_j=\frac{\sinh\big(\kappa(T-t_j)\big)}{\sinh(\kappa T)}X,\qquad \kappa^2\approx\frac{\lambda\sigma^2}{\eta}\ (\tau\to 0)$$

- 风险厌恶 $\lambda$ 越大，$\kappa$ 越大，越“前置”卖出（怕价格波动）；
- $\lambda=0$ 时退化为 **TWAP**（匀速执行）；$1/\kappa$ 是最优交易的“半衰期”。

**RL 执行**：
- <a href="https://doi.org/10.1145/1143844.1143929">Nevmyvaka, Feng, Kearns（ICML 2006）</a>：第一篇大规模把 RL 用于执行的实证工作（1.5 年 NASDAQ 毫秒级 LOB）。
- **OPD**（Fang et al., AAAI 2021，<a href="https://arxiv.org/abs/2103.10860">arXiv 2103.10860</a>，已并入 Qlib）：**Oracle Policy Distillation**。Teacher 能看到未来信息（完美信息的 oracle），Student 只用历史信息并通过蒸馏损失模仿 Teacher，部署时只用 Student；优化器 PPO，所有股票数据合训得到通用策略。奖励（以卖出为例）：

$$R_t=\Big(\frac{p_{t+1}}{\tilde p}-1\Big)a_t-\alpha\,a_t^2$$

  $\tilde p$ 为窗口内平均价，$a_t$ 为本步成交占目标量比例，二次项是临时冲击惩罚。数据：A 股分钟级 2017–2019，CSI 800 验证/测试；基线 TWAP、AC、VWAP、DDQN、PPO。

<img src="https://raw.githubusercontent.com/microsoft/qlib/main/docs/_static/img/QlibRL_framework.png" style="width:75%">

QlibRL 架构（基于 Tianshou / Gym），OPD 的执行实验在此框架中实现（`qlib/examples/rl_order_execution`）。

<br>

### 4.5 做市（Market Making）

做市商同时挂买单和卖单赚价差，但要承担库存风险和逆向选择。

**Avellaneda–Stoikov（2008）**：最大化终端财富的指数效用，订单到达强度 $Ae^{-k\delta}$ 随报价距离指数衰减。两步解：

$$r(s,q,t)=s-q\,\gamma\sigma^2(T-t)\qquad\text{(reservation price)}$$

$$\delta^a+\delta^b=\gamma\sigma^2(T-t)+\frac{2}{\gamma}\ln\Big(1+\frac{\gamma}{k}\Big)\qquad\text{(总价差)}$$

直觉：持有多头库存 $q>0$ 时，保留价低于中间价，报价整体下移，倾向于卖出以减少库存。

**RL 做市**（Spooner et al., AAMAS 2018，<a href="https://arxiv.org/abs/1804.04216">arXiv 1804.04216</a>）：用 10 只股票的 LOB 重建模拟器；TD 学习 + 资格迹，tile coding 线性值函数；10 个动作（9 种报价宽度/偏斜 + 清仓）。关键是**非对称阻尼奖励**：

$$r_i=\Psi(t_i)-\max\big[0,\ \eta\,\mathrm{Inv}(t_i)\,\Delta m(t_i)\big]$$

$\Psi$ 为增量 PnL（两边成交收益 + 库存 × 中间价变化）。它压低“库存投机”的收益但保留投机亏损，迫使 Agent 去赚价差——训练更稳定、样本外收益更高。

<br>

### 4.6 组合管理

**EIIE**（Jiang, Xu, Liang，<a href="https://arxiv.org/abs/1706.10059">arXiv 1706.10059</a>）：各资产共享参数的独立评估器（Ensemble of Identical Independent Evaluators）+ 组合向量记忆（PVM）+ 在线随机批学习（OSBL）。奖励是平均对数收益：

$$R=\frac{1}{t_f}\sum_{t}\ln\big(\mu_t\,\mathbf{y}_t\cdot\mathbf{w}_{t-1}\big)$$

$\mu_t$ 为扣除佣金的交易剩余因子。关键假设：动作不影响价格（零冲击）。

**FinRL**（<a href="https://github.com/AI4Finance-Foundation/FinRL">GitHub</a>）：

<img src="https://raw.githubusercontent.com/AI4Finance-Foundation/FinRL/master/figs/finrl_framework.png" style="width:80%">

FinRL 三层：Market Environments / DRL Agents / Financial Applications。当前仓库定位为教学与研究原型，生产向版本迁移至 FinRL-X / FinRL-Trading。

<br>

### 4.7 Model-Based 与 Multi-Agent RL

- **Model-based**：先学市场状态转移，再用 learned simulator 训练策略——与第 3 节世界模型直接相关。代表：<a href="https://arxiv.org/abs/1901.08740">Yu et al. 2019</a>（预测模块 + GAN 数据增强 + 行为克隆），MarS 中训练 RL 执行 Agent。综述统计中只占 3/167，是明显的空白。
- **Multi-agent**：<a href="https://arxiv.org/abs/1904.12066">ABIDES</a> 是基于 Agent 的离散事件市场模拟器；<a href="https://arxiv.org/abs/2511.02016">ABIDES-MARL</a> 指出：当做市商会策略性调整报价时，“外生冲击”假设本身就是错的——执行变成有限期随机博弈，流动性由互动内生产生；在外生冲击下优化的执行策略，一旦做市商会适应就表现很差，**可预测的执行会被做市商利用**。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180016123.png" style="width:85%">

ABIDES-MARL 的通信周期：多个学习 Agent 同步决策。

<br>

### 4.8 三个分支小结

| 分支 | 解决什么 | 核心挑战 | 与其他方向的联系 |
|---|---|---|---|
| Risk-Sensitive RL | 尾部风险、回撤、风险预算 | 风险度量不可按步分解 | Reward 设计（第 1 节）|
| Offline RL | 不能在线试错 | 分布偏移、OOD 动作高估 | 历史回测即离线数据 |
| Model-Based RL | 样本效率、可做反事实 | 模型误差被策略利用 | Market World Models（第 3 节）|

<br>

## 5. Causal Alpha Discovery：从相关性转向稳健的预测规律

> CAUSAL REPRESENTATION · INVARIANT LEARNING · DISTRIBUTION SHIFT

一个典型问题：历史数据显示某种社交媒体情绪与股票未来上涨相关。但这种相关性可能来自宏观行情、市场风格、选择偏差，或者只是**大量尝试后的偶然结果**。这一节分两半：先解决“偶然”（多重检验），再解决“不稳健”（不变性/因果）。

### 5.1 多重检验：因子动物园与回测过拟合

#### Harvey, Liu & Zhu（RFS 2016）：t > 3.0

<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314">…and the Cross-Section of Expected Returns</a>：把已发表的数百个“因子”（factor zoo）视为一个整体的多重检验问题，把 Bonferroni、Holm（控制 FWER）和 BHY（控制 FDR）调整到相关且部分不可观测（未发表的检验看不到）的情形。结论：**新因子的门槛应是 t > 3.0，而不是 2.0**；金融经济学中许多声称的发现很可能是假的。

#### Deflated Sharpe Ratio（Bailey & López de Prado 2014）

<a href="https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf">DSR</a> 同时校正两件事：多次试验带来的选择偏差，以及收益的非正态性。思路：如果你试了 $N$ 个独立策略，即使它们真实 Sharpe 都是 0，最大的那个的期望也会大于 0。这个“期望最大值”就是新的拒绝门槛：

$$\widehat{SR}_0=\sqrt{V[\{\widehat{SR}_n\}]}\Big((1-\gamma)\,Z^{-1}\big[1-\tfrac1N\big]+\gamma\,Z^{-1}\big[1-\tfrac{1}{Ne}\big]\Big)$$

$$\widehat{DSR}=Z\left[\frac{(\widehat{SR}-\widehat{SR}_0)\sqrt{T-1}}{\sqrt{1-\hat\gamma_3\widehat{SR}+\frac{\hat\gamma_4-1}{4}\widehat{SR}^2}}\right]$$

$Z$ 为标准正态 CDF，$\gamma\approx0.5772$ 为 Euler–Mascheroni 常数，$\hat\gamma_3,\hat\gamma_4$ 为偏度、峰度，$V[\cdot]$ 为各次试验 Sharpe 的方差，Sharpe 不年化。（分母是 PSR 的标准形式。）

**对 Agent 的意义**：Agent 每多试一个因子，$N$ 就加一，门槛 $\widehat{SR}_0$ 随之上升——**把试验次数写进 reward**，是对抗 Quant-Arena 观察到的“越迭代越过拟合”的最直接手段。

#### Probability of Backtest Overfitting（Bailey, Borwein, López de Prado, Zhu）

<a href="https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf">PBO</a> 用组合对称交叉验证（CSCV）估计“样本内最优的策略在样本外跌到中位数以下的概率”：

1. 构造 $T\times N$ 的 PnL 矩阵（$N$ 个策略配置）；
2. 按行切成偶数 $S$ 块；
3. 取所有 $\binom{S}{S/2}$ 种组合（$S=16$ 时 12,870 种），每种组合中一半做训练集 $J$、另一半做测试集 $\bar J$；
4. 在 $J$ 上选最优策略 $n^*$，求它在 $\bar J$ 中的相对排名 $\bar\omega_c=\bar r_{n^*,c}/(N+1)$，取 logit $\lambda_c=\ln\frac{\bar\omega_c}{1-\bar\omega_c}$；
5. $\text{PBO}=\int_{-\infty}^{0}f(\lambda)\,d\lambda$。

PBO > 0.05 可以拒绝模型，也可作贝叶斯先验或组合权重。

#### White Reality Check（2000）与 Hansen SPA（2005）

- <a href="https://users.ssc.wisc.edu/~bhansen/718/White2000.pdf">White</a>：检验 $H_0:\max_k E[d_k]\le0$（$K$ 个模型中最好的那个相对基准没有优势），统计量 $\max_k\sqrt{n}\,\bar d_k$，用（平稳）bootstrap 得零分布。
- <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569">Hansen SPA</a>：学生化统计量 + 只对“不明显更差”的模型做重新中心化的样本依赖零分布——功效更高，对糟糕/无关的备选更不敏感。

<br>

### 5.2 不变性学习与 Regime 适应

#### Invariant Risk Minimization（Arjovsky et al. 2019）

<a href="https://arxiv.org/abs/1907.02893">IRM</a>：只保留在所有环境中**最优读出头都相同**的表示。

$$\min_{\Phi,w}\sum_{e\in\mathcal{E}_{tr}}R^e(w\circ\Phi)\quad\text{s.t.}\quad w\in\arg\min_{\bar w}R^e(\bar w\circ\Phi)\ \ \forall e$$

实用版 IRMv1：

$$\min_\Phi\sum_{e\in\mathcal{E}_{tr}}R^e(\Phi)+\lambda\,\big\lVert\nabla_{w\mid w=1.0}R^e(w\cdot\Phi)\big\rVert^2$$

$w=1.0$ 是固定的标量“哑分类器”，梯度范数惩罚衡量它在每个环境中离最优有多远。**在量化中**：把市场 regime / 时间段当作环境 $e$，只保留跨 regime 都有效的信号。

#### 应用到股票预测

| 工作 | 思路 |
|---|---|
| <a href="https://arxiv.org/abs/2409.00671">InvariantStock</a>（2024）| 环境感知预测模块 + 环境无关模块，以日期为环境，学跨市场环境都成立的特征；中美数据测试 |
| <a href="https://arxiv.org/abs/2306.09862">DoubleAdapt</a>（KDD 2023）| 应对概念漂移：数据适配器把输入映射到局部平稳分布，模型适配器调整参数，每次增量更新视为一个元学习任务（<a href="https://github.com/SJTU-Quant/qlib">代码</a>）|
| <a href="https://academic.oup.com/rfs/article/33/5/2223/5758276">Gu, Kelly & Xiu</a>（RFS 2020）| ML 资产定价基准：树与神经网络最好（捕捉非线性交互）；所有方法都认同主导信号是**动量、流动性、波动率** |

#### López de Prado：Causal Factor Investing（2023）

<a href="https://www.cambridge.org/core/elements/causal-factor-investing/9AFE270D7099B787B8FD4F4CBADE0C6E">Causal Factor Investing: Can Factor Investing Become Scientific?</a>（Cambridge Elements，开放获取）：因子投资停留在“关联性、描述性”阶段；没有因果理论和因果图，发现很可能是假的——原因包括回测过拟合，以及错误的规格选择（**控制了对撞因子、遗漏了混杂因子**）。

<br>

### 5.3 研究方向小结

- **Invariant Representation Learning**：学习在不同市场环境下仍有预测能力的表征。
- **Regime-Aware Adaptation**：判断市场机制何时变化、何时重新训练（DoubleAdapt 式增量学习）。
- **Causal Feature Discovery**：在明确因果假设及混杂因素的前提下研究信号机制。
- **Multiple Testing Control**：避免 Agent 大规模挖因子后只汇报碰巧有效的——DSR / PBO / SPA 就是现成的工具，应该成为 Agent 评估器的一部分。

> 注意：**稳健预测不自动等于因果发现**；真正的因果识别仍需要合理的识别假设（如工具变量、自然实验）。IRM 给出的是“跨环境不变”，这是因果的必要而非充分条件。

<br>

## 6. AI for Market Microstructure：AI 深入交易的最底层

> LIMIT ORDER BOOK · HIGH-FREQUENCY DATA · NEURAL EVENT MODELS

核心问题不再是“明天涨还是跌”，而是：未来几毫秒到几秒的价格变化？买卖盘不平衡如何影响短期价格？订单成交概率多大？何时挂单、撤单、主动成交？如何降低 adverse selection 与执行成本？

### 6.1 限价订单簿（LOB）基础

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180021123.png" style="width:80%">

限价订单簿示意：买盘（bid）与卖盘（ask）按价格档位排列，最优买价与最优卖价之间是价差（spread）。（图源：<a href="https://arxiv.org/abs/2411.06389">arXiv 2411.06389</a>）

- **中间价** $m=(P^a+P^b)/2$，**价差** $S=P^a-P^b$。
- 事件类型：限价单（加深度）、市价单（吃深度）、撤单。
- 数据源：LOBSTER（NASDAQ 重建）、FI-2010（北欧）、交易所 Level-2/Level-3。

### 6.2 经典微观结构量

**订单流不平衡 OFI**（<a href="https://arxiv.org/abs/1011.6402">Cont, Kukanov & Stoikov 2014</a>）：每个事件对最优档的贡献

$$e_n=\mathbb{1}_{\{P^B_n\ge P^B_{n-1}\}}q^B_n-\mathbb{1}_{\{P^B_n\le P^B_{n-1}\}}q^B_{n-1}-\mathbb{1}_{\{P^A_n\le P^A_{n-1}\}}q^A_n+\mathbb{1}_{\{P^A_n\ge P^A_{n-1}\}}q^A_{n-1}$$

区间内求和 $OFI_k=\sum_n e_n$，价格变化近似线性：$\Delta P_k=\beta\,OFI_k+\epsilon_k$，且 $\beta\propto 1/\text{深度}$。OFI 对价格的解释力强于成交不平衡；开盘时冲击约为收盘时 5 倍。

**Kyle's lambda**（Kyle 1985）：价格对总订单流线性 $p=\mu+\lambda(x+u)$，$\lambda$ 是单位有符号订单流的价格冲击，$1/\lambda$ 即市场深度；实证中用 $\Delta P$ 对有符号成交量回归的斜率估计。

**Micro-price**（<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694">Stoikov 2018</a>）：中间价忽略了盘口不平衡，加权中间价 $M^w=IP^a+(1-I)P^b$（$I=Q^b/(Q^b+Q^a)$）又太噪且不是鞅。Micro-price 定义为未来中间价的极限期望：

$$P^{micro}_t=\lim_{i\to\infty}E\big[M_{\tau_i}\mid\mathcal{F}_t\big]=M_t+\sum_{k\ge1}g^k(I_t,S_t)$$

$\tau_i$ 为中间价变动时刻，$g^k$ 在 (不平衡, 价差) 的离散马尔可夫链上递归计算。

### 6.3 深度学习预测中间价趋势

**FI-2010 数据集**（<a href="https://arxiv.org/abs/1705.03233">Ntakaris et al. 2018</a>）：5 只纳斯达克北欧股票、10 个交易日、约 400 万样本。标签：

$$l_i=\frac{\frac1k\sum_{j=i+1}^{i+k}m_j-m_i}{m_i},\qquad k\in\{1,2,3,5,10\}$$

阈值 0.002 分为上涨 / 平稳 / 下跌三类。

**DeepLOB**（<a href="https://arxiv.org/abs/1808.03668">Zhang, Zohren, Roberts, IEEE TSP 2019</a>，<a href="https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books">代码</a>）：

| 层 | 作用 |
|---|---|
| 输入 $X\in\mathbb{R}^{100\times40}$ | 最近 100 个 LOB 状态 × 10 档 ×（卖价、卖量、买价、买量）|
| Conv 1×2, stride 1×2 | 合并每档的 (价, 量) |
| Conv 1×2, stride 1×2 | 合并买卖两侧——相当于在每档学一个 **micro-price** |
| Conv 1×10 | 跨档整合 |
| Inception@32 | 1×1→3×1、1×1→5×1、max-pool→1×1，相当于多时间尺度的“学习型移动平均” |
| LSTM(64) → softmax | 上涨 / 平稳 / 下跌；全模型仅约 6 万参数 |

LSE 数据上作者改用平滑标签 $l_t=\frac{m_+(t)-m_-(t)}{m_-(t)}$（前后各 $k$ 步均值之比），信号更一致。

**LOBCAST 基准**（<a href="https://arxiv.org/abs/2308.01915">Prata et al., AI Review 2024</a>，<a href="https://github.com/matteoprata/LOBCAST">代码</a>）——给这个方向泼了冷水：
- 复现 15 个模型（MLP、LSTM、CNN、DeepLOB、TransLOB、BiN-CTABL、AxialLOB…）。
- FI-2010 上 BiN-CTABL 最好（F1 82.6%），但**多个模型达不到其论文报告的数字**（尤其 TransLOB、ATNBoF）。
- 换到 LOBSTER 的新数据（LOB-2021/2022，6 只纳斯达克股票），F1 跌到约 48–61%——作者归因于**对 FI-2010 的过拟合**；模型对超参数非常敏感。

**LiT: Limit Order Book Transformer**（<a href="https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full">Frontiers in AI 2025</a>）：

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20261102180020123.png" style="width:85%">

LiT 架构：把“最近 64 个快照 × 每侧 20 档”的价量网格切成**结构化 patch**（一侧盘口 × 一小段时间窗），线性投影 + 位置嵌入 → Transformer → LSTM → softmax。

无卷积；Binance Level-2 毫秒数据；300ms–1s 级别预测（如 300–1000ms 准确率 68.34%）；后续月份发生漂移时，只微调最后的全连接层可恢复约 5% 准确率。

### 6.4 点过程：对“事件何时发生”建模

**Hawkes 过程**（<a href="https://arxiv.org/abs/1502.04592">Bacry, Mastromatteo & Muzy 2015 综述</a>）：事件会“自激发”和“互激发”：

$$\lambda^i_t=\mu^i+\sum_{j=1}^D\int_{-\infty}^{t}\phi^{ij}(t-s)\,dN^j_s,\qquad \phi^{ij}(t)=\alpha_{ij}e^{-\beta_{ij}t}$$

平稳性要求核范数矩阵的谱半径 < 1；实证中市场接近临界（“内生性”）。LOB 应用：激进限价单主要由激进市价单触发；中间价变动主要由互激发主导。

**Neural Hawkes**（<a href="https://arxiv.org/abs/1612.09328">Mei & Eisner, NIPS 2017</a>）：用**连续时间 LSTM** 替代参数化核：

$$\lambda_k(t)=f_k\big(\mathbf{w}_k^\top\mathbf{h}(t)\big),\quad f_k(x)=s_k\log(1+e^{x/s_k})$$

细胞状态在事件之间指数衰减到移动目标，能表达经典 Hawkes 做不到的**非单调、抑制、延迟**效应。

### 6.5 成交概率估计

<a href="https://arxiv.org/abs/2306.05479">Arroyo, Cartea, Moreno-Pino & Zohren（Quantitative Finance 2024）</a>：把限价单的“成交时间”建模为**生存分析**——卷积-Transformer 编码器捕捉局部 LOB 模式，单调神经网络解码器保证生存函数单调，用右删失对数似然训练；在 Nasdaq LOB 数据上优于基准，并用 Shapley 值与注意力图解释。这直接服务于“挂单还是吃单”的决策。

> 这个方向与 LLM 的直接关系较弱，但它是**市场世界模型的“物理层”**：OFI、平方根冲击律、Hawkes 自激发，都是检验生成式模拟器“因果反应是否正确”的标尺。

<br>

## 7. 如果要做研究项目：三个候选课题

以下是基于上述文献的**研究设想**（不是已被证明有效的方案），适合有 LLM Agent、SFT/GRPO 和金融研究系统经验的研究者。

### A. Learning to Discover Robust Alpha（首选）

**核心问题**：能否通过可验证的实验反馈训练一个 Quant Research Agent，使其提出的策略具有更好的**样本外**表现？

- 环境：严格隔离的回测（借鉴 Quant-Arena 的冻结-封闭协议与 KTD-Fin 的遮蔽）。
- Reward 不是 Sharpe，而是组合：
  - **增量 IC**（对已有因子库正交化后的残差 IC）、
  - **Deflated Sharpe**（把试验次数 $N$ 记入门槛）、
  - **原创性**（AlphaAgent 式 AST 子树相似度惩罚）、
  - 换手与交易成本、
  - **归因后的选股 α**（KTD-Fin 式 Barra 分解）。
- 训练：Agentic RL（如 GRPO）+ verifiable rewards；对照组为 RD-Agent(Q) 式的 prompt-only 闭环。
- 关键实验：研究期 reward 与封闭期收益的相关性是否比 Quant-Arena 的基线更高（即 reward hacking 是否被抑制）。

技术组合：Agentic RL + Verifiable Rewards + Alpha Mining + Statistical Testing。

### B. World Models for Robust Trading Agents

**核心问题**：交易 Agent 能否通过生成式市场模拟，在历史上没有出现过的市场环境中学会更稳健的执行策略？

- 构建条件 Market World Model（DiffLOB 式 regime 控制 + TRADES/MarS 式订单级可交互）。
- 用不同波动、流动性和极端情景生成训练数据，训练 Offline/Model-Based RL 执行策略（OPD 作为基线）。
- 关键是**验证世界模型的因果反应**：用平方根冲击律、OFI-价格关系、Hawkes 结构做“物理一致性”检验，再测策略迁移到真实留出 regime 的效果。

技术组合：Diffusion + World Models + Offline/Model-Based RL。

### C. Self-Improving Financial Research Agents

**核心问题**：Agent 能否从过去的研究**失败**中学习，而非仅仅积累更多成功策略？

- 设计带实验记忆的研究 Agent：记录失败假设、过拟合原因、数据问题、所处市场环境（RD-Agent(Q) 的知识库已存失败三元组，但没有系统研究其作用）。
- 研究不同 memory / update 策略（失败记忆检索、反思总结、负样本微调）对研究效率与泛化能力的影响。
- 评估指标：达到同等封闭期表现所需的试验次数；重复踩坑率。

技术组合：Long-Term Memory + Continual Learning + Agent Evaluation。

### 三个课题对比

| 维度 | A：Alpha Agent | B：World Model | C：Research Memory |
|---|---|---|---|
| 与 LLM Agent 经验匹配 | 极高 | 中等 | 极高 |
| 基础 ML 深度 | 高 | 极高 | 中高 |
| Quant 专业门槛 | 中高 | 极高 | 中等 |
| 公开数据可做性 | 较高（Qlib A 股日频）| 中低（LOB 数据贵）| 较高 |
| 研究差异化空间 | 高 | 很高 | 中高 |
| 可直接复用的开源 | RD-Agent、Qlib、AlphaGen | DeepMarket、DiffLOB、MarS | RD-Agent |

**判断**：想做一篇有技术深度、又能与 LLM Agent 背景自然衔接的研究，优先 **AI Scientist for Quant + Verifiable RL（课题 A）**；想转向基础模型、环境建模或 World Models，则 **Market World Models（课题 B）** 更值得长期投入。

<br>

## 8. 速记卡片：快速回想这个领域

**一句话版本**：AI for Quant 正从“预测收益”走向“自动研究 + 可交互模拟 + 风险感知决策”，而整个领域最大的瓶颈是**评估**：泄漏、多重检验、成本、归因。

**必须记住的数字**

| 事实 | 出处 |
|---|---|
| 9 个前沿模型的最佳配置都没跑赢 2024 等权 A 股市场（17.1%）；38 个配置只有 1 个跑赢 | Quant-Arena |
| RankIC 与 ARR 跨运行相关仅 0.09；迭代后研究期→封闭期掉 12.4 分 | Quant-Arena |
| 10 个 LLM 交易 Agent 中 9 个选股 α 为负，收益来自 Common + Style | KTD-Fin |
| RD-Agent(Q) CSI300 ARR 14.21%（o3-mini），约 2.6× Alpha158，成本 <10 美元 | RD-Agent(Q) |
| Kronos：120 亿条 K 线，BSQ 20 位码拆成 2×10 位；价格 RankIC 比最强 TSFM 高 93% | Kronos |
| TSFM 赢 8/10 个任务，但相对 random walk 只有 2 个 DM 显著 | 2606.27100 |
| TRADES 比次优方法好约 3.3–3.5×，在 ABIDES 中产生永久冲击 | TRADES |
| 167 篇 RL 量化论文中 model-based 只有 3 篇；SR>3 就该怀疑 | ACM CSUR 综述 |
| 新因子门槛 t > 3.0 | Harvey-Liu-Zhu |

**必须记住的公式**

- IC：$\mathbb{E}_t[\text{corr}_i(F_{i,t},y_{i,t})]$；ICIR = 均值/标准差
- Barra 归因：Common + Σ(暴露×因子收益) + 选股 α
- 世界模型：$P(s_{t+1},r_t\mid s_t,a_t)$ vs 预测模型 $P(s_{t+1}\mid s_t)$
- Almgren-Chriss：$x_j=X\sinh(\kappa(T-t_j))/\sinh(\kappa T)$
- Avellaneda-Stoikov：$r=s-q\gamma\sigma^2(T-t)$
- Deflated Sharpe：门槛 $\widehat{SR}_0$ 随试验次数 $N$ 增长
- IRMv1：$\sum_e R^e+\lambda\lVert\nabla_{w}R^e(w\cdot\Phi)\rVert^2$
- OFI：$\Delta P=\beta\,OFI$，$\beta\propto1/\text{深度}$

**评估 Quant AI 论文的检查清单**

1. 测试期是否晚于 LLM 的知识截止？标识符是否遮蔽？
2. 试了多少次？是否报告了 Deflated Sharpe / PBO / 显著性检验？
3. 是否扣除交易成本、市场冲击，遵守 T+1 / 涨跌停？
4. 收益是否做了归因（市场、风格 vs 选股 α）？是否对比了等权/买入持有基准？
5. Sharpe 是否离谱（>3）？是否只在单只股票/单个窗口上测？
6. 是否有多种子方差？是否有冻结后的样本外？
7. 世界模型：除了统计真实性，是否检验了对动作的因果反应？
