---
title: Agent Memory综述整理 
date: 2025-12-17
categories:
  - 学AI/DS
tags:
  - LLM
  - Agent
desc: 整理自 https://arxiv.org/abs/2512.13564 《Memory in the Age of AI Agents》
---

综述地址：
- <a href="https://arxiv.org/abs/2512.13564"> Memory in the Age of AI Agents </a>
<br>



## 1.背景

### 动机

大语言模型（LLMs）虽然在单次对话中表现出色，但其本质上是**无状态的（stateless）**。给定上下文窗口 $C$，LLM 只能处理长度不超过 $|C|$ 的输入序列，其中 $|C|$ 通常限制在 4K 到 128K tokens 之间。这种限制使得 LLM 无法在**跨会话、跨任务、长期运行**的场景中维持记忆。

形式化地，设 $M_t$ 表示时刻 $t$ 的记忆状态，$X_t$ 表示时刻 $t$ 的输入，则传统 LLM 的决策函数为：

$$a_t = f_{LLM}(X_t, C_t)$$

其中 $C_t \subseteq \{X_{t-k}, X_{t-k+1}, \ldots, X_{t-1}\}$ 是上下文窗口，$k \leq |C|$。当 $t$ 很大时，早期经验 $\{X_1, X_2, \ldots, X_{t-k-1}\}$ 被完全遗忘。

智能体（Agent）在复杂环境中持续运行，必须依赖**外部记忆系统** $\mathcal{M}$ 来维持长期状态。记忆系统 $\mathcal{M}$ 提供四个核心能力：行为一致性（behavioral consistency），使得 Agent 在不同时间对相同情境产生一致响应；个性化适应（personalization），根据用户历史行为定制响应；经验复用（experience reuse），从历史成功案例中学习；协作记忆（collaborative memory），在多 Agent 系统中共享知识。

### 核心贡献

本文提出 **Forms–Functions–Dynamics 三维分析框架**，首次系统性地将智能体记忆研究纳入统一范式。该框架将记忆系统分解为三个相互关联的维度：

**Forms（形式）**：记忆的表示形式，回答"What architectural or representational forms can agent memory take"。
- 包括记忆的**编码方式**（结构化 vs 非结构化）、**粒度**（事件级 vs 知识级）、**存储介质**（向量、图、关系型数据库）等。
- 形式化地，记忆形式可以表示为 $m \in \mathcal{M}$，其中 $\mathcal{M}$ 是记忆空间，$m$ 可以是向量 $\mathbf{v} \in \mathbb{R}^d$、图节点 $n \in \mathcal{G}$、或结构化记录 $r \in \mathcal{R}$。

**Functions（功能）**：记忆的用途，回答"Why is agent memory needed, and what roles or purposes does it serve"。
- 记忆系统支持五种核心功能：
  - **检索（Retrieval）**$\mathcal{R}: \mathcal{Q} \times \mathcal{M} \rightarrow \mathcal{M}'$，从记忆库中召回相关信
  - **反思（Reflection）**$\mathcal{F}: \mathcal{M}\_{episodic} \rightarrow \mathcal{M}\_{semantic}$，从具体经验中提炼抽象原则
  - **规划（Planning）**$\mathcal{P}: \mathcal{M}\_{procedural} \times \mathcal{T} \rightarrow \Pi$，基于历史计划生成新计划
  - **个性化（Personalization）**$\mathcal{U}: \mathcal{M}\_{episodic} \rightarrow \mathcal{P}\_{user}$，构建用户画像
  - **协作（Collaboration）**$\mathcal{C}: \{\mathcal{M}_i\}\_{i=1}^n \rightarrow \mathcal{M}\_{shared}$，多 Agent 记忆融合。

**Dynamics（动态）**：记忆的演化机制，回答"How does agent memory operate, adapt, and evolve over time"。
- 记忆不是静态仓库，而是动态演化的认知器官。
- 动态机制包括：
  - 巩固（Consolidation）$\mathcal{CON}: \mathcal{M}\_{short} \rightarrow \mathcal{M}\_{long}$，将短期记忆整合为长期知识
  - 更新（Updating）$\mathcal{UPD}: \mathcal{M}\_{old} \times \mathcal{E}\_{new} \rightarrow \mathcal{M}\_{new}$，根据新证据更新记忆
  - 遗忘（Forgetting）$\mathcal{FOR}: \mathcal{M} \times \mathcal{C} \rightarrow \mathcal{M}'$，移除过时或低价值记忆。


三个维度不可割裂。例如，"反思"功能要求高抽象语义记忆（Form），并通过全局整合（Dynamics）实现。形式化地，这种耦合关系可以表示为：
$$\mathcal{F}(\mathcal{M}\_{episodic}) = \mathcal{CON}(\mathcal{M}\_{episodic}, \text{abstraction}) \subseteq \mathcal{M}\_{semantic}$$

其中抽象操作将具体事件映射到语义空间。

<br>

## 2.定义

### Agent

**Agent（智能体）**是一个以 LLM 为核心推理引擎的自主系统。形式化地，Agent 可以表示为元组 $\mathcal{A} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{M}, f_{LLM} \rangle$，其中 $\mathcal{S}$ 是状态空间，$\mathcal{A}$ 是动作空间，$\mathcal{T}$ 是工具集合，$\mathcal{M}$ 是记忆系统，$f_{LLM}: \mathcal{S} \times \mathcal{M} \rightarrow \mathcal{A}$ 是 LLM 决策函数。

Agent 具备四个核心能力。**时间持续性（Temporal Persistence）**使得 Agent 能够在时间维度上持续运行，跨越多个会话和任务。设 $T = \{t_1, t_2, \ldots, t_n\}$ 表示时间序列，Agent 的状态演化遵循 $s_{t+1} = f_{transition}(s_t, a_t, m_t)$，其中 $m_t \in \mathcal{M}$ 是从记忆系统检索的相关记忆。

**工具调用（Tool Usage）**能力使得 Agent 能够调用外部工具和 API。形式化地，工具调用可以表示为 $a_t = \text{call}(tool_i, params)$，其中 $tool_i \in \mathcal{T}$，$params$ 是参数。工具调用的结果被存储到记忆系统中，形成经验 $\mathcal{E} = \{(tool_i, params, result, timestamp)\}$。

**交互能力（Interaction）**使得 Agent 能够与人类用户或其他 Agent 进行协作和通信。在多 Agent 系统中，Agent $i$ 和 Agent $j$ 之间的交互可以建模为消息传递 $m_{i \rightarrow j} = f_{comm}(s_i, \mathcal{M}_i)$，其中 $\mathcal{M}_i$ 是 Agent $i$ 的记忆系统。

**记忆利用（Memory Utilization）**是 Agent 的核心能力。Agent 利用记忆系统存储、检索和更新历史经验，以支持决策。决策函数可以扩展为 $a_t = f_{LLM}(s_t, \mathcal{R}(q_t, \mathcal{M}))$，其中 $\mathcal{R}: \mathcal{Q} \times \mathcal{M} \rightarrow \mathcal{M}'$ 是检索函数，$q_t$ 是查询。

根据 Agent 数量，可以分为 Single-Agent 系统和 Multi-Agent 系统。Single-Agent 系统独立运行，决策函数为 $a_t = f_{LLM}(s_t, \mathcal{M})$。Multi-Agent 系统包含多个 Agent $\{\mathcal{A}_1, \mathcal{A}_2, \ldots, \mathcal{A}_n\}$，需要共享记忆 $\mathcal{M}\_{shared}$，决策函数为 $a_{i,t} = f_{LLM}(s_{i,t}, \mathcal{M}_i, \mathcal{M}\_{shared})$。

根据应用领域，可以分为 General-Purpose Agent 和 Domain-Specific Agent。General-Purpose Agent 处理通用任务，记忆系统 $\mathcal{M}$ 覆盖广泛领域。Domain-Specific Agent 专注于特定领域，记忆系统 $\mathcal{M}\_{domain}$ 包含领域特定知识。

根据记忆依赖程度，可以分为 Memory-Light Agent 和 Memory-Heavy Agent。Memory-Light Agent 主要依赖上下文窗口，记忆系统 $\mathcal{M}$ 较小或为空。Memory-Heavy Agent 严重依赖外部记忆系统，$\mathcal{M}$ 规模大且频繁使用。

<br>

### Memory System

**Memory System（记忆系统）**是一个完整的架构，由四个核心组件构成。形式化地，记忆系统可以表示为 $\mathcal{M} = \langle \mathcal{D}, \mathcal{R}, \mathcal{W}, \mathcal{I} \rangle$。

**记忆数据库（Memory Database）**$\mathcal{D}$ 是持久化存储层，支持多种数据结构。设 $\mathcal{M}\_{vector} = \{\mathbf{v}_i \in \mathbb{R}^d\}\_{i=1}^n$ 表示向量记忆库，$\mathcal{M}\_{graph} = \langle V, E \rangle$ 表示图记忆库，$\mathcal{M}\_{relational} = \{r_i\}\_{i=1}^n$ 表示关系型记忆库。记忆数据库提供存储操作 $\text{store}: \mathcal{M} \times m \rightarrow \mathcal{M}'$，其中 $m$ 是要存储的记忆项。

**读/写接口（Read/Write APIs）**提供记忆的存储、检索、更新、删除操作。写操作可以表示为 $\mathcal{M}' = \mathcal{M} \cup \{m\}$，读操作可以表示为 $m = \text{read}(\mathcal{M}, id)$，更新操作可以表示为 $\mathcal{M}' = (\mathcal{M} \setminus \{m_{old}\}) \cup \{m_{new}\}$，删除操作可以表示为 $\mathcal{M}' = \mathcal{M} \setminus \{m\}$。

**检索机制（Retrieval Mechanism）**$\mathcal{R}: \mathcal{Q} \times \mathcal{M} \rightarrow \mathcal{M}'$ 基于查询从记忆库中召回相关信息。对于向量记忆库，检索可以表示为 $\mathcal{M}' = \text{TopK}(\{\text{sim}(q, \mathbf{v}_i)\}\_{i=1}^n)$，其中 $\text{sim}$ 是相似度函数（如余弦相似度 $\cos(\mathbf{q}, \mathbf{v}) = \frac{\mathbf{q} \cdot \mathbf{v}}{||\mathbf{q}|| \cdot ||\mathbf{v}||}$）。对于图记忆库，检索可以表示为图查询 $Q = \text{match}(pattern, \mathcal{M}\_{graph})$。

**整合模块（Integration Module）**$\mathcal{I}: \mathcal{M}' \times \mathcal{C} \rightarrow \mathcal{C}'$ 将检索到的记忆整合到 Agent 的决策上下文中。整合操作可以表示为 $c' = \text{integrate}(c, \{m_1, m_2, \ldots, m_k\})$，其中 $c$ 是当前上下文，$\{m_i\}$ 是检索到的记忆。

记忆系统与 Agent 的耦合方式有两种。**紧密耦合**中，记忆操作直接嵌入 Agent 的决策循环，决策函数为 $a_t = f_{LLM}(s_t, \mathcal{I}(\mathcal{R}(q_t, \mathcal{M}), c_t))$。**松散耦合**中，记忆系统作为独立服务，通过 API 调用，决策函数为 $a_t = f_{LLM}(s_t, \text{API}(\mathcal{M}, q_t))$。

<br>

### 记忆系统的分类

#### 按时间维度

记忆系统按时间维度可以分为短期记忆和长期记忆。**短期记忆（Short-term Memory）**$\mathcal{M}\_{short}$ 存储原始、未处理、高保真的详细信息，通常存储在 LLM 上下文窗口或临时缓存中。形式化地，短期记忆可以表示为 $\mathcal{M}\_{short} = \{m_i\}\_{i=t-k}^{t-1}$，其中 $k \leq |C|$ 是上下文窗口大小。短期记忆的生命周期为单次任务或会话内，典型大小为 4K-128K tokens。

**长期记忆（Long-term Memory）**$\mathcal{M}\_{long}$ 存储抽象、结构化、可索引的压缩信息，存储在外部数据库（向量数据库、图数据库、SQL 数据库、文件系统）中。形式化地，长期记忆可以表示为 $\mathcal{M}\_{long} = \{m_j\}\_{j=1}^N$，其中 $N$ 可以非常大（受存储限制）。长期记忆的生命周期为跨任务、跨会话、持久化。

关键区别在于：短期记忆受限于 LLM 上下文长度，通常直接包含在 prompt 中，无需检索，即 $c_t = [m_{t-k}, m_{t-k+1}, \ldots, m_{t-1}]$。长期记忆需要外部存储和检索机制（如 RAG），通过相似性搜索或结构化查询获取，即 $c_t = \mathcal{I}(\mathcal{R}(q_t, \mathcal{M}\_{long}), c_{short})$。

<br>

#### 按抽象层次

记忆系统按抽象层次可以分为三个层次。**原始层（Raw）**$\mathcal{M}\_{raw}$ 存储未处理的交互日志、对话历史，形式为 $m_{raw} = \text{"User: What's the weather? Agent: It's sunny."}$。原始层保留完整信息，适合回溯，但存储效率低。

**事件层（Event-level）**$\mathcal{M}\_{event}$ 存储结构化的事件记录，包含时间戳、动作、结果。形式化地，事件可以表示为 $e = \langle t, a, o, r \rangle$，其中 $t$ 是时间戳，$a$ 是动作，$o$ 是观察，$r$ 是奖励。事件层适合个性化、行为分析，存储效率中等。

**知识层（Knowledge-level）**$\mathcal{M}\_{knowledge}$ 存储抽象的事实、规则、原则。形式化地，知识可以表示为 $k = \text{"用户偏好简洁回答"}$ 或三元组 $k = \langle subject, predicate, object \rangle$。知识层适合泛化、推理、规划，存储效率高但可能丢失细节。

抽象层次之间的关系可以表示为映射 $\phi: \mathcal{M}\_{raw} \rightarrow \mathcal{M}\_{event} \rightarrow \mathcal{M}\_{knowledge}$，其中 $\phi$ 是抽象函数。

<br>

#### 按存储结构

记忆系统按存储结构可以分为非结构化、半结构化、结构化三种。**非结构化（Unstructured）**记忆 $\mathcal{M}\_{unstructured}$ 存储原始文本，形式为 $m = \text{"raw text"}$。非结构化记忆灵活、易存储，但难以查询，检索效率低，适合原始日志、对话历史。

**半结构化（Semi-structured）**记忆 $\mathcal{M}\_{semi}$ 存储 JSON、XML 等格式的结构化数据，形式为 $m = \{\text{"key": "value"}\}$。半结构化记忆平衡灵活性和可查询性，但需要 schema 设计，适合 JSON、XML 格式的事件记录。

**结构化（Structured）**记忆 $\mathcal{M}\_{structured}$ 存储在关系型数据库或图数据库中，形式为 $m \in \mathcal{R}$ 或 $m \in \mathcal{G}$。结构化记忆高效查询、支持复杂推理，但需要预定义 schema、灵活性低，适合图数据库、关系型数据库。

<br>

## 3.记忆的形式

### 分类原则

记忆形式的分类遵循三个核心维度：表示粒度、结构化程度、模态。**表示粒度（Granularity）**描述记忆的抽象层次。事件级（Event-level）记忆记录具体的交互事件，包含完整上下文，形式化地可以表示为 $m_{event} = \langle t, s, a, o, r \rangle$，其中 $t$ 是时间戳，$s$ 是状态，$a$ 是动作，$o$ 是观察，$r$ 是奖励。知识级（Knowledge-level）记忆存储抽象的事实、规则、模式，去除具体细节，形式化地可以表示为 $m_{knowledge} = \phi(m_{event})$，其中 $\phi$ 是抽象函数。

**结构化程度（Structure）**描述记忆的组织方式。非结构化（Unstructured）记忆存储原始日志、自由文本，形式为 $m_{unstructured} = \text{"raw text"}$。半结构化（Semi-structured）记忆存储 JSON、XML 等格式的数据，形式为 $m_{semi} = \{k_i: v_i\}\_{i=1}^n$。结构化（Structured）记忆存储在图、关系型数据库中，形式为 $m_{structured} \in \mathcal{G}$ 或 $m_{structured} \in \mathcal{R}$。

**模态（Modality）**描述记忆的内容类型。单模态记忆主要处理文本，形式为 $m_{text} \in \mathcal{T}$，其中 $\mathcal{T}$ 是文本空间。多模态记忆处理文本、图像、音频、视频等多种模态，形式为 $m_{multimodal} = \{m_{text}, m_{image}, m_{audio}, m_{video}\}$，其中各模态通过嵌入函数映射到统一空间 $\mathbf{v} = f_{embed}(m_{modality}) \in \mathbb{R}^d$。

### 主要形式

#### Episodic Memory（情景记忆）

**Episodic Memory（情景记忆）**记录 Agent 与环境的**具体交互事件**，包含时间戳、动作、观察、奖励、反馈等完整信息。形式化地，episodic memory 可以表示为元组 $e = \langle t, s, a, o, r, f \rangle$，其中 $t \in \mathbb{R}^+$ 是时间戳，$s \in \mathcal{S}$ 是状态，$a \in \mathcal{A}$ 是动作，$o \in \mathcal{O}$ 是观察，$r \in \mathbb{R}$ 是奖励，$f \in \mathcal{F}$ 是反馈。Episodic memory 集合可以表示为 $\mathcal{M}\_{episodic} = \{e_i\}\_{i=1}^n$，其中 $n$ 是事件数量。

Episodic memory 的核心特征包括**高保真性**、**时间序列性**、**上下文完整性**。高保真性意味着保留原始交互细节，不进行抽象，即 $e$ 包含完整信息 $\text{info}(e) = \{t, s, a, o, r, f\}$。时间序列性意味着按时间顺序组织，支持时间查询，即 $\mathcal{M}\_{episodic}$ 可以按时间排序 $\{e_i: t_i < t_j \text{ if } i < j\}$。上下文完整性意味着包含事件发生的完整上下文，即 $e$ 包含状态 $s$ 和观察 $o$。

技术实现上，episodic memory 存储在时序数据库（如 InfluxDB）、文档数据库（如 MongoDB）、向量数据库（用于相似性检索）中。存储操作可以表示为 $\mathcal{M}\_{episodic}' = \mathcal{M}\_{episodic} \cup \{e\}$。索引包括时间索引 $I_{time}(t)$、用户 ID 索引 $I_{user}(u)$、事件类型索引 $I_{type}(\tau)$、向量嵌入索引 $I_{vector}(\mathbf{v})$，其中 $\mathbf{v} = f_{embed}(e) \in \mathbb{R}^d$ 是事件的嵌入表示。

检索操作包括时间范围查询 $\mathcal{Q}\_{time}(t_1, t_2) = \{e: t_1 \leq t(e) \leq t_2\}$、用户历史查询 $\mathcal{Q}\_{user}(u) = \{e: u(e) = u\}$、相似事件检索 $\mathcal{Q}\_{similar}(e_q, k) = \text{TopK}(\{\text{sim}(e_q, e_i)\}\_{i=1}^n)$，其中 $\text{sim}(e_q, e_i) = \cos(\mathbf{v}_q, \mathbf{v}_i)$ 是余弦相似度。

Episodic memory 的应用场景包括个性化、行为建模、错误回溯、A/B 测试。个性化应用中，基于用户历史行为定制响应，形式化地可以表示为 $a_t = f_{personalize}(s_t, \mathcal{Q}\_{user}(u_t))$。行为建模中，分析用户行为模式，形式化地可以表示为 $P(a|s, u) = \text{estimate}(\mathcal{Q}\_{user}(u))$。错误回溯中，定位问题发生的具体事件，形式化地可以表示为 $e_{error} = \arg\min_{e \in \mathcal{Q}\_{time}(t_1, t_2)} \text{error\_score}(e)$。

代表工作包括 MOOM（Chen et al., 2025d），从稀疏的 episodic memory 中构建稳定的用户角色快照（User Role Snapshot），形式化地可以表示为 $\text{snapshot}(u) = f_{MOOM}(\mathcal{Q}\_{user}(u))$。MemGPT（Packer et al., 2023）使用分层记忆系统，episodic memory 作为工作记忆。AgentBoard（Zhou et al., 2024）是长期交互基准，显式评估 episodic memory 的保留和检索。

Episodic memory 的优势是信息完整、支持细粒度查询、适合个性化，但劣势是存储开销大、检索效率相对较低、难以泛化。存储开销可以量化为 $|\mathcal{M}\_{episodic}| = O(n \cdot |e|)$，其中 $|e|$ 是单个事件的大小。

<br>

#### Semantic Memory（语义记忆）

**Semantic Memory（语义记忆）**存储**抽象的事实、规则、常识、领域知识**，去除具体事件细节，保留可泛化的语义信息。形式化地，semantic memory 可以表示为知识集合 $\mathcal{M}\_{semantic} = \{k_i\}\_{i=1}^m$，其中 $k_i$ 是知识项。知识项可以通过抽象函数从 episodic memory 中提取：$k = \phi(e)$，其中 $\phi: \mathcal{M}\_{episodic} \rightarrow \mathcal{M}\_{semantic}$ 是抽象映射。

Semantic memory 的核心特征包括**高抽象性**、**任务无关性**、**可复用性**。高抽象性意味着从具体事件中提炼通用知识，抽象过程可以表示为 $k = \arg\min_{k'} \text{loss}(\{e_i\}, k')$，其中 $\text{loss}$ 是抽象损失函数。任务无关性意味着不绑定特定任务或用户，即 $k$ 不包含任务特定信息 $\text{task\_info}(k) = \emptyset$。可复用性意味着可在多个场景中复用，形式化地可以表示为 $k$ 适用于场景集合 $\mathcal{S}_k = \{s_1, s_2, \ldots, s_n\}$。

Semantic memory 的表示格式包括三元组形式、自然语言陈述、规则形式。三元组形式可以表示为 $k = \langle s, p, o \rangle$，其中 $s \in \mathcal{E}$ 是主体（subject），$p \in \mathcal{P}$ 是谓词（predicate），$o \in \mathcal{O}$ 是客体（object）。例如，$(User_U123, prefers, direct_flights)$ 表示用户 U123 偏好直飞航班。自然语言陈述可以表示为 $k = \text{"用户偏好直飞航班，不喜欢中转"}$，通过嵌入函数映射到向量空间 $\mathbf{v}_k = f_{embed}(k) \in \mathbb{R}^d$。规则形式可以表示为 $k = \text{IF } \text{condition} \text{ THEN } \text{action}$，其中 $\text{condition}$ 是条件，$\text{action}$ 是动作。

技术实现上，semantic memory 存储在知识图谱（Neo4j、Amazon Neptune）、RDF 三元组存储、向量数据库（语义嵌入）中。知识图谱可以表示为 $\mathcal{G} = \langle V, E \rangle$，其中 $V = \{s, o\}$ 是节点集合（实体），$E = \{p\}$ 是边集合（关系）。RDF 三元组存储使用标准格式 $\text{RDF}(s, p, o)$。向量数据库存储嵌入表示 $\{\mathbf{v}_i = f_{embed}(k_i)\}\_{i=1}^m$。

构建方法包括手工构建、自动提取、混合方法。手工构建由领域专家定义，形式为 $\mathcal{M}\_{semantic} = \{\text{expert\_defined}(k_i)\}\_{i=1}^n$。自动提取从 episodic memory 中通过 LLM 提炼，形式为 $k = f_{LLM}(\{e_i\}\_{i=1}^n, \text{prompt})$，其中 $\text{prompt}$ 是提示模板。混合方法结合手工和自动，形式为 $\mathcal{M}\_{semantic} = \mathcal{M}\_{manual} \cup \mathcal{M}\_{auto}$，其中 $\mathcal{M}\_{manual}$ 是手工构建的知识，$\mathcal{M}\_{auto}$ 是自动提取的知识。

Semantic memory 的应用场景包括常识推理、任务规划、知识问答、跨任务泛化。常识推理中，利用领域知识进行推理，形式化地可以表示为 $r = f_{reason}(\mathcal{Q}\_{semantic}(q), \mathcal{M}\_{semantic})$，其中 $q$ 是查询，$\mathcal{Q}\_{semantic}(q)$ 是检索到的相关语义记忆。任务规划中，基于规则生成执行计划，形式化地可以表示为 $\pi = f_{plan}(\mathcal{M}\_{semantic}, \text{task})$。知识问答中，回答事实性问题，形式化地可以表示为 $a = f_{qa}(q, \mathcal{M}\_{semantic})$。跨任务泛化中，将知识应用到新任务，形式化地可以表示为 $\text{transfer}(k, \text{task}\_{new}) = f_{transfer}(k, \text{task}\_{new})$。

代表工作包括 Matrix（Liu et al., 2024），从 Agent 轨迹中提炼"任务无关原则"（task-agnostic principles），通过迭代反思循环优化，形式化地可以表示为 $k^{(t+1)} = f_{refine}(k^{(t)}, \mathcal{M}\_{episodic})$，其中 $t$ 是迭代次数。GraphRAG（Kim et al., 2024）构建知识图谱，支持复杂多跳推理，形式化地可以表示为 $\mathcal{G} = f_{build}(\mathcal{D})$，其中 $\mathcal{D}$ 是文档集合。Mem0（Mem0.ai）自动从对话中提取和更新语义记忆，形式化地可以表示为 $k = f_{extract}(\text{conversation})$。

Semantic memory 的优势是高度抽象、可复用、支持复杂推理、存储效率高，但劣势是可能丢失细节、需要抽象过程、可能过度泛化。抽象损失可以量化为 $\text{loss}\_{abstraction} = \sum_{e \in \mathcal{E}_k} \text{dist}(e, k)$，其中 $\mathcal{E}_k$ 是生成知识 $k$ 的事件集合，$\text{dist}$ 是距离函数。

#### Procedural Memory（程序记忆）

**Procedural Memory（程序记忆）**存储**任务执行流程、技能模板、操作序列**，记录"如何做"而非"是什么"。形式化地，procedural memory 可以表示为程序集合 $\mathcal{M}\_{procedural} = \{\pi_i\}\_{i=1}^p$，其中 $\pi_i$ 是程序或计划。程序可以表示为函数 $\pi: \mathcal{S} \times \mathcal{P} \rightarrow \mathcal{A}^*$，其中 $\mathcal{S}$ 是状态空间，$\mathcal{P}$ 是参数空间，$\mathcal{A}^*$ 是动作序列。

Procedural memory 的核心特征包括**过程性**、**可执行性**、**参数化**。过程性意味着关注执行步骤和流程，程序可以表示为步骤序列 $\pi = [a_1, a_2, \ldots, a_n]$，其中 $a_i \in \mathcal{A}$ 是动作。可执行性意味着可直接用于指导行动，执行函数可以表示为 $\text{execute}(\pi, s, p) = [a_1(s, p), a_2(s, p), \ldots, a_n(s, p)]$。参数化意味着支持不同输入参数，程序可以表示为 $\pi(p_1, p_2, \ldots, p_k)$，其中 $p_i$ 是参数。

Procedural memory 的表示格式包括代码片段、状态机、计划模板。代码片段可以表示为函数 $f: \mathcal{X} \rightarrow \mathcal{Y}$，例如 $f_{\text{book\_flight}}(user, origin, dest, date) = \text{result}$。状态机可以表示为元组 $\mathcal{SM} = \langle Q, \Sigma, \delta, q_0, F \rangle$，其中 $Q$ 是状态集合，$\Sigma$ 是输入字母表，$\delta: Q \times \Sigma \rightarrow Q$ 是转移函数，$q_0$ 是初始状态，$F$ 是接受状态集合。计划模板可以表示为元组 $\text{Plan} = \langle \text{task\_type}, \text{steps}, \text{failure\_handling} \rangle$，其中 $\text{steps} = [\text{step}_1, \text{step}_2, \ldots, \text{step}_n]$ 是步骤序列，$\text{failure\_handling}$ 是失败处理策略。

技术实现上，procedural memory 存储在代码仓库、JSON/YAML 配置文件、图数据库（流程表示为图）中。代码仓库存储可执行代码，形式为 $\mathcal{M}\_{code} = \{f_i\}\_{i=1}^n$。配置文件存储结构化计划，形式为 $\mathcal{M}\_{config} = \{\text{Plan}_i\}\_{i=1}^n$。图数据库存储流程表示为图，形式为 $\mathcal{G}\_{process} = \langle V_{step}, E_{transition} \rangle$，其中 $V_{step}$ 是步骤节点，$E_{transition}$ 是转移边。

检索操作包括基于任务类型匹配、基于输入参数相似性、基于历史成功率。任务类型匹配可以表示为 $\mathcal{Q}\_{type}(\tau) = \{\pi: \text{type}(\pi) = \tau\}$，其中 $\tau$ 是任务类型。输入参数相似性可以表示为 $\mathcal{Q}\_{similar}(p, k) = \text{TopK}(\{\text{sim}(p, p_i)\}\_{i=1}^n)$，其中 $\text{sim}$ 是相似度函数。历史成功率可以表示为 $\mathcal{Q}\_{success}(\theta) = \{\pi: \text{success\_rate}(\pi) \geq \theta\}$，其中 $\theta$ 是成功率阈值。

Procedural memory 的应用场景包括工具链复用、技能学习、错误恢复、多步骤任务。工具链复用中，在 WebArena、AgentBench 等环境中复用成功的操作序列，形式化地可以表示为 $\pi_{new} = \text{adapt}(\pi_{old}, \text{task}\_{new})$，其中 $\text{adapt}$ 是适配函数。技能学习中，从演示中学习操作技能，形式化地可以表示为 $\pi = f_{learn}(\mathcal{D}\_{demo})$，其中 $\mathcal{D}\_{demo}$ 是演示数据集。错误恢复中，存储失败处理流程，形式化地可以表示为 $\pi_{recovery} = f_{recover}(\text{error}, \pi_{failed})$。多步骤任务中，复杂任务的分解和执行，形式化地可以表示为 $\pi = \text{decompose}(\text{task}) = [\pi_1, \pi_2, \ldots, \pi_n]$。

代表工作包括 Voyager（Wang et al., 2023），Minecraft Agent，将成功技能存储为可复用的代码，形式化地可以表示为 $\pi = f_{extract}(\text{successful\_trajectory})$。ReAct（Yao et al., 2023）将工具使用模式存储为可复用的"思维-行动"模板，形式化地可以表示为 $\pi = \text{"think"} \rightarrow \text{"act"}$。WebArena（Zhou et al., 2023）评估 Agent 在网页环境中的程序记忆复用能力，形式化地可以表示为 $\text{score} = f_{eval}(\pi, \text{task})$。

Procedural memory 的优势是直接可执行、高效复用、支持复杂任务，但劣势是需要精确匹配、难以适应环境变化、可能过时。匹配精度可以量化为 $\text{precision} = \frac{|\{\pi: \text{match}(\pi, \text{task})\}|}{|\{\pi\}|}$，其中 $\text{match}$ 是匹配函数。


<br>

#### Graph-based Memory（基于图的记忆）

**Graph-based Memory（基于图的记忆）**使用**图结构**存储记忆，节点表示实体或事件，边表示关系或因果关系。形式化地，图记忆可以表示为 $\mathcal{G} = \langle V, E, \lambda_V, \lambda_E \rangle$，其中 $V$ 是节点集合，$E \subseteq V \times V$ 是边集合，$\lambda_V: V \rightarrow \mathcal{L}_V$ 是节点标签函数，$\lambda_E: E \rightarrow \mathcal{L}_E$ 是边标签函数。

Graph-based memory 的核心特征包括**关系显式化**、**多跳推理**、**灵活查询**。关系显式化意味着关系作为一等公民存储，边 $e = (u, v) \in E$ 显式表示节点 $u$ 和 $v$ 之间的关系。多跳推理支持沿着边的路径进行推理，路径可以表示为 $p = [v_1, e_1, v_2, e_2, \ldots, v_k]$，其中 $(v_i, v_{i+1}) \in E$。灵活查询支持复杂的图查询语言，查询可以表示为 $Q = \text{match}(pattern, \mathcal{G})$，其中 $pattern$ 是查询模式。

图结构设计包括实体-关系图、事件-因果图、知识图谱。实体-关系图可以表示为 $\mathcal{G}\_{ER} = \langle V_{entity}, E_{relation} \rangle$，其中 $V_{entity} = \{\text{User}, \text{Flight}, \text{Airport}\}$ 是实体节点，$E_{relation} = \{\text{prefers}, \text{booked}, \text{from}, \text{to}\}$ 是关系边。事件-因果图可以表示为 $\mathcal{G}\_{causal} = \langle V_{event}, E_{causal} \rangle$，其中 $V_{event}$ 是事件节点，$E_{causal}$ 是因果关系边，因果关系可以表示为 $e \rightarrow e'$ 表示事件 $e$ 导致事件 $e'$。知识图谱可以表示为 $\mathcal{G}\_{KG} = \langle V_{entity}, E_{knowledge} \rangle$，其中知识边可以表示为三元组 $\langle s, p, o \rangle$。

技术实现上，图记忆存储在 Neo4j、Amazon Neptune、ArangoDB 等图数据库中。Neo4j 使用 Cypher 查询语言，查询可以表示为 $\text{Cypher}(Q) = \text{MATCH } pattern \text{ RETURN } result$。SPARQL 是标准 RDF 查询语言，查询可以表示为 $\text{SPARQL}(Q) = \text{SELECT } vars \text{ WHERE } pattern$。Gremlin 是图遍历语言，查询可以表示为 $\text{Gremlin}(Q) = g.V().has(...).out(...)$。

Graph-based memory 的应用场景包括复杂关系查询、多跳推理、知识整合、因果分析。复杂关系查询可以表示为 $Q = \text{"找出所有偏好直飞且预订过国际航班的用户"}$，形式化地可以表示为 $\text{result} = \{u: u \in V \land \text{prefers}(u, \text{direct}) \land \exists f \text{ booked}(u, f) \land \text{international}(f)\}$。多跳推理可以表示为路径查询 $p = \text{find\_path}(v_1, v_k, \mathcal{G})$，例如"用户 A 的朋友 B 喜欢的餐厅"可以表示为 $\text{restaurant} = \text{find\_path}(\text{User\_A}, \text{Restaurant}, \mathcal{G})$。知识整合可以表示为图合并 $\mathcal{G}\_{merged} = \mathcal{G}_1 \cup \mathcal{G}_2$，其中 $\cup$ 是图合并操作。因果分析可以表示为因果路径查找 $\text{causal\_path} = \text{find\_causal\_path}(e_1, e_2, \mathcal{G}\_{causal})$。

代表工作包括 GraphRAG（Kim et al., 2024），从文档构建知识图谱，支持复杂查询，形式化地可以表示为 $\mathcal{G} = f_{build}(\mathcal{D})$，其中 $\mathcal{D}$ 是文档集合。G-Memory（Zhang et al., 2025c）是多 Agent 系统中的个性化图记忆，形式化地可以表示为 $\mathcal{G}_i = f_{personalize}(\mathcal{G}\_{shared}, \text{Agent}_i)$。MemGPT（Packer et al., 2023）使用图结构组织长期记忆，形式化地可以表示为 $\mathcal{M}\_{long} = \text{organize}(\mathcal{M}\_{episodic}, \mathcal{G})$。

Graph-based memory 的优势是支持复杂关系查询、多跳推理、灵活的数据模型，但劣势是查询复杂度高、需要图数据库基础设施、写入性能相对较低。查询复杂度可以量化为 $O(|V| + |E|)$ 对于简单查询，$O(|V|^k)$ 对于 $k$-跳查询。

<br>

#### Vector-based Memory（基于向量的记忆）

**Vector-based Memory（基于向量的记忆）**将文本（或其他模态）**嵌入为高维向量**，通过向量相似性搜索进行检索。形式化地，向量记忆可以表示为 $\mathcal{M}\_{vector} = \{(\mathbf{v}_i, m_i)\}\_{i=1}^n$，其中 $\mathbf{v}_i \in \mathbb{R}^d$ 是嵌入向量，$m_i$ 是原始记忆项，$d$ 是向量维度。嵌入函数可以表示为 $f_{embed}: \mathcal{M} \rightarrow \mathbb{R}^d$，将记忆项映射到向量空间。

Vector-based memory 的核心特征包括**语义相似性**、**高效检索**、**多模态支持**。语义相似性意味着相似的文本在向量空间中距离近，相似度可以表示为 $\text{sim}(\mathbf{v}_1, \mathbf{v}_2) = \cos(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{||\mathbf{v}_1|| \cdot ||\mathbf{v}_2||}$。高效检索使用近似最近邻（ANN）算法快速检索，检索可以表示为 $\mathcal{Q}\_{ANN}(q, k) = \text{TopK}(\{\text{sim}(\mathbf{q}, \mathbf{v}_i)\}\_{i=1}^n)$，其中 $\mathbf{q} = f_{embed}(q)$ 是查询向量，$k$ 是返回数量。多模态支持可扩展到图像、音频等模态，多模态嵌入可以表示为 $\mathbf{v}\_{multimodal} = f_{multimodal}(m_{text}, m_{image}, m_{audio})$。

技术流程包括嵌入、存储、检索三个阶段。嵌入阶段将文本转换为向量，形式为 $\mathbf{v} = f_{embed}(\text{"用户偏好直飞航班"}) \in \mathbb{R}^d$，其中 $d$ 通常是 1536 或 768。存储阶段将向量和元数据存储到向量数据库，形式为 $\mathcal{M}\_{vector}' = \mathcal{M}\_{vector} \cup \{(\mathbf{v}, \text{metadata})\}$，其中 $\text{metadata} = \{\text{text}, \text{timestamp}, \text{user\_id}\}$。检索阶段通过相似性搜索找到相关记忆，形式为 $\text{results} = \mathcal{Q}\_{ANN}(\mathbf{q}, k) = \text{TopK}(\{\text{sim}(\mathbf{q}, \mathbf{v}_i)\}\_{i=1}^n)$。

技术实现上，向量记忆存储在 FAISS、Pinecone、Weaviate、Milvus、Qdrant 等向量数据库中。FAISS 使用 LSH（Locality-Sensitive Hashing）或 IVF（Inverted File Index）等索引结构，检索复杂度为 $O(\log n)$。Pinecone 是托管服务，提供 REST API 接口。Weaviate 支持混合搜索，结合向量检索和关键词检索。Milvus 支持大规模部署，可以处理百万级向量。Qdrant 使用 Rust 编写，性能高。

嵌入模型包括 OpenAI Embeddings（text-embedding-ada-002、text-embedding-3）、Sentence Transformers、BGE（BAAI General Embedding）等。OpenAI Embeddings 使用 Transformer 架构，输出维度为 1536。Sentence Transformers 是开源库，支持多种预训练模型。BGE 是中文优化的嵌入模型，在中文任务上表现更好。

检索策略包括 Dense Retrieval、Hybrid Search、Reranking。Dense Retrieval 使用向量相似性，形式为 $\text{results} = \mathcal{Q}\_{dense}(q, k) = \text{TopK}(\{\text{sim}(\mathbf{q}, \mathbf{v}_i)\}\_{i=1}^n)$。Hybrid Search 结合向量检索和关键词检索（BM25），形式为 $\text{results} = \alpha \cdot \mathcal{Q}\_{dense}(q, k) + (1-\alpha) \cdot \mathcal{Q}\_{BM25}(q, k)$，其中 $\alpha$ 是权重参数。Reranking 使用交叉编码器对候选结果重排序，形式为 $\text{results}\_{reranked} = \text{rerank}(\text{results}\_{candidate}, q)$，其中 $\text{rerank}$ 是重排序函数。

Vector-based memory 的应用场景包括语义搜索、大规模记忆库、多模态检索。语义搜索中，基于语义相似性检索相关记忆，形式为 $\text{results} = \mathcal{Q}\_{semantic}(q, \mathcal{M}\_{vector})$。大规模记忆库中，支持百万级记忆的高效检索，检索复杂度为 $O(\log n)$。多模态检索中，文本查询图像记忆，形式为 $\text{results} = \mathcal{Q}\_{crossmodal}(\mathbf{q}\_{text}, \mathcal{M}\_{image})$。

代表工作包括 Mem0，使用向量数据库存储和检索用户记忆，形式为 $\mathcal{M}\_{user} = \mathcal{Q}\_{vector}(\text{user\_query}, \mathcal{M}\_{vector})$。LangChain Memory 提供向量记忆后端，形式为 $\mathcal{M}\_{langchain} = \text{VectorStore}(\mathcal{M}\_{vector})$。LlamaIndex 支持向量存储的 RAG 框架，形式为 $\text{context} = \mathcal{Q}\_{RAG}(q, \mathcal{M}\_{vector})$。

Vector-based memory 的优势是检索速度快、支持语义相似性、易于扩展、支持多模态，但劣势是丢失结构信息、难以支持精确匹配、难以进行复杂推理、可能检索到不相关结果。检索精度可以量化为 $\text{precision@k} = \frac{|\text{relevant} \cap \text{retrieved}|}{|\text{retrieved}|}$，其中 $\text{relevant}$ 是相关记忆集合，$\text{retrieved}$ 是检索到的记忆集合。

<br>

### 对比总结

**对比总结**（对应原文 Figure 2）：

| 记忆形式 | 保真度 | 抽象度 | 检索效率 | 推理能力 | 存储效率 | 主要用途 |
|---------|--------|--------|---------|---------|---------|---------|
| **Episodic** | 高 | 低 | 中 | 低 | 低 | 个性化、回溯 |
| **Semantic** | 低 | 高 | 高 | 高 | 高 | 泛化、推理 |
| **Procedural** | 中 | 中 | 中 | 中 | 中 | 任务执行 |
| **Graph** | 中 | 中 | 中 | 高 | 中 | 关系查询、推理 |
| **Vector** | 中 | 中 | 高 | 低 | 高 | 语义检索 |

<br>

**选择指南**：
- **需要个性化** → Episodic Memory
- **需要知识推理** → Semantic Memory + Graph
- **需要任务复用** → Procedural Memory
- **需要快速检索** → Vector-based Memory
- **需要复杂关系查询** → Graph-based Memory
- **混合方案**：大多数实际系统采用**混合记忆架构**，结合多种形式

<br>

## 4.记忆的功能

### 五大核心功能

记忆系统为 Agent 提供五大核心功能，每种功能对应不同的记忆形式和实现技术。形式化地，功能可以表示为映射 $F: \mathcal{M} \times \mathcal{Q} \rightarrow \mathcal{O}$，其中 $\mathcal{M}$ 是记忆空间，$\mathcal{Q}$ 是查询空间，$\mathcal{O}$ 是输出空间。

**检索（Retrieval）**$\mathcal{R}: \mathcal{Q} \times \mathcal{M} \rightarrow \mathcal{M}'$ 从记忆库中召回与当前任务相关的历史信息，所需记忆形式为 Episodic、Semantic 或 Vector，典型实现技术包括 Dense retrieval、BM25、Hybrid search，评估指标为 Recall@K、Precision@K、MRR。

**反思（Reflection）**$\mathcal{F}: \mathcal{M}\_{episodic} \rightarrow \mathcal{M}\_{semantic}$ 对历史经验进行元认知分析，提炼抽象原则，所需记忆形式为 Semantic（high-level），典型实现技术包括 LLM-based summarization、Insight extraction，评估指标为原则质量、泛化能力。

**规划（Planning）**$\mathcal{P}: \mathcal{M}\_{procedural} \times \mathcal{T} \rightarrow \Pi$ 基于历史经验指导未来行动序列，所需记忆形式为 Procedural 或 Graph，典型实现技术包括 Case-based reasoning、Plan reuse、Hierarchical planning，评估指标为任务成功率、规划效率。

**个性化（Personalization）**$\mathcal{U}: \mathcal{M}\_{episodic} \rightarrow \mathcal{P}\_{user}$ 根据用户历史行为定制 Agent 的响应和行为，所需记忆形式为 Episodic + User Profile，典型实现技术包括 Preference modeling、Role prompting、User embedding，评估指标为用户满意度、行为一致性。

**协作（Collaboration）**$\mathcal{C}: \{\mathcal{M}_i\}\_{i=1}^n \rightarrow \mathcal{M}\_{shared}$ 多 Agent 之间共享知识和经验，所需记忆形式为 Shared Semantic 或 Graph，典型实现技术包括 Memory broadcasting、Consensus protocols、Federated memory，评估指标为知识共享效率、一致性。

<br>

### 功能-形式对齐

#### Retrieval（检索）

**Retrieval（检索）**从长期记忆库中**召回与当前查询相关的历史信息**，是记忆系统最基础的功能。形式化地，检索可以表示为函数 $\mathcal{R}: \mathcal{Q} \times \mathcal{M} \rightarrow \mathcal{M}'$，其中 $\mathcal{Q}$ 是查询空间，$\mathcal{M}$ 是记忆库，$\mathcal{M}' \subseteq \mathcal{M}$ 是检索到的记忆子集。检索的目标是最大化相关性 $\text{rel}(q, m)$，同时保持检索效率。

检索的核心挑战包括相关性判断、检索效率、信息整合。相关性判断需要计算查询和记忆之间的相似度，形式化地可以表示为 $\text{rel}(q, m) = f_{similarity}(q, m)$，其中 $f_{similarity}$ 是相似度函数。检索效率需要在百万级记忆库中快速检索，检索复杂度应该为 $O(\log n)$ 或 $O(\sqrt{n})$，而不是 $O(n)$。信息整合需要将多个检索结果整合为有用的上下文，形式化地可以表示为 $c = f_{integrate}(\{m_1, m_2, \ldots, m_k\})$，其中 $c$ 是整合后的上下文。

检索策略包括 Dense Retrieval、Sparse Retrieval、Hybrid Search。Dense Retrieval 使用嵌入模型将查询和记忆都转换为向量，然后计算余弦相似度或点积。形式化地，可以表示为 $\mathbf{q} = f_{embed}(q) \in \mathbb{R}^d$，$\mathbf{m}_i = f_{embed}(m_i) \in \mathbb{R}^d$，相似度为 $\text{sim}(\mathbf{q}, \mathbf{m}_i) = \cos(\mathbf{q}, \mathbf{m}_i) = \frac{\mathbf{q} \cdot \mathbf{m}_i}{||\mathbf{q}|| \cdot ||\mathbf{m}_i||}$，检索结果为 $\mathcal{M}' = \text{TopK}(\{\text{sim}(\mathbf{q}, \mathbf{m}_i)\}\_{i=1}^n)$。

Sparse Retrieval 使用 BM25、TF-IDF 等关键词匹配方法，适合精确匹配场景。BM25 评分函数可以表示为 $\text{BM25}(q, m) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, m) \cdot (k_1 + 1)}{f(t, m) + k_1 \cdot (1 - b + b \cdot \frac{|m|}{\text{avgdl}})}$，其中 $t$ 是查询词，$f(t, m)$ 是词 $t$ 在记忆 $m$ 中的频率，$\text{IDF}(t)$ 是逆文档频率，$k_1$ 和 $b$ 是参数，$|m|$ 是记忆长度，$\text{avgdl}$ 是平均文档长度。

Hybrid Search 结合 dense 和 sparse 检索，使用 Reciprocal Rank Fusion (RRF) 合并结果。RRF 评分可以表示为 $\text{RRF}(m) = \sum_{r \in R} \frac{1}{k + r(m)}$，其中 $R$ 是排序列表集合，$r(m)$ 是记忆 $m$ 在排序列表 $r$ 中的排名，$k$ 是常数（通常为 60）。最终检索结果为 $\mathcal{M}' = \text{TopK}(\{\text{RRF}(m_i)\}\_{i=1}^n)$。

检索后处理包括 Reranking、Filtering、Deduplication。Reranking 使用交叉编码器（Cross-Encoder）对候选结果重新排序，形式化地可以表示为 $\text{score}(q, m) = f_{cross\_encoder}(q, m)$，重排序结果为 $\mathcal{M}'_{reranked} = \text{sort}(\mathcal{M}', \text{score})$。Filtering 基于元数据过滤，形式化地可以表示为 $\mathcal{M}'_{filtered} = \{m: m \in \mathcal{M}' \land \text{filter}(m, \text{metadata})\}$。Deduplication 移除重复或高度相似的检索结果，形式化地可以表示为 $\mathcal{M}'_{dedup} = \{m_i: \forall j < i, \text{sim}(m_i, m_j) < \theta\}$，其中 $\theta$ 是相似度阈值。

评估指标包括 Recall@K、Precision@K、MRR、NDCG。Recall@K 可以表示为 $\text{Recall@K} = \frac{|\mathcal{R} \cap \mathcal{M}'_{K}|}{|\mathcal{R}|}$，其中 $\mathcal{R}$ 是相关记忆集合，$\mathcal{M}'_{K}$ 是前 K 个检索结果。Precision@K 可以表示为 $\text{Precision@K} = \frac{|\mathcal{R} \cap \mathcal{M}'_{K}|}{K}$。MRR 可以表示为 $\text{MRR} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{1}{\text{rank}_q}$，其中 $\text{rank}_q$ 是第一个相关结果的排名。NDCG 可以表示为 $\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$，其中 $\text{DCG@K} = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i+1)}$，$\text{IDCG@K}$ 是理想 DCG。

检索的应用场景包括上下文增强、案例检索、知识问答。上下文增强中，为 LLM 提供相关历史信息，形式化地可以表示为 $c_{enhanced} = c \cup \mathcal{R}(q, \mathcal{M})$。案例检索中，检索相似的历史案例，形式化地可以表示为 $\text{cases} = \mathcal{R}(\text{task}, \mathcal{M}\_{episodic})$。知识问答中，从知识库中检索答案，形式化地可以表示为 $a = f_{qa}(q, \mathcal{R}(q, \mathcal{M}\_{semantic}))$。

<br>

#### Reflection（反思）

**Reflection（反思）**是 Agent 定期对历史轨迹进行**元认知分析**，从具体经验中提炼抽象的、可泛化的原则和见解的过程。形式化地，反思可以表示为函数 $\mathcal{F}: \mathcal{M}\_{episodic} \rightarrow \mathcal{M}\_{semantic}$，将 episodic memory 映射到 semantic memory。反思的目标是从具体事件 $\{e_1, e_2, \ldots, e_n\}$ 中提取抽象原则 $k$，使得 $k$ 能够解释和指导未来行为。

**核心机制**包括触发条件、反思流程、原则提取三个阶段：

- **触发条件**：形式化地可以表示为 $\text{trigger} = f_{trigger}(t, \mathcal{M}, \text{event})$，其中 $t$ 是时间，$\mathcal{M}$ 是记忆库，$\text{event}$ 是事件
  - **定期触发**：每 N 次交互后，即 $t \bmod N = 0$
  - **事件触发**：任务完成、错误发生、重要里程碑，即 $\text{event} \in \{\text{task\_complete}, \text{error}, \text{milestone}\}$
  - **容量触发**：记忆库达到一定规模，即 $|\mathcal{M}| \geq \theta_{capacity}$

- **反思流程**：包括选择相关事件、LLM 分析、验证和精炼三个阶段
  - **选择相关事件**：$\mathcal{E}\_{selected} = f_{select}(\mathcal{M}\_{episodic}, \text{criteria})$，其中 $\text{criteria}$ 可以是时间窗口 $\mathcal{E}\_{time}(t_1, t_2) = \{e: t_1 \leq t(e) \leq t_2\}$ 或主题聚类 $\mathcal{E}\_{cluster} = \text{cluster}(\mathcal{M}\_{episodic}, k)$
  - **LLM 分析**：$k^{(0)} = f_{LLM}(\mathcal{E}\_{selected}, \text{prompt})$，其中 $\text{prompt}$ 是提示模板
  - **验证和精炼**：迭代过程 $k^{(t+1)} = f_{refine}(k^{(t)}, \mathcal{E}\_{selected}, \text{feedback})$，收敛条件为 $||k^{(t+1)} - k^{(t)}|| < \epsilon$

**技术实现**包括 Prompt 设计、迭代优化、原则表示：

- **Prompt 设计**：引导 LLM 识别模式、提取原则，形式为 $\text{prompt} = \text{"Based on the following interactions, analyze and extract general principles that can guide future behavior: [Episodic Memory Records]. Please identify: 1. Patterns in user behavior, 2. Successful strategies, 3. Common failure modes, 4. Generalizable rules"}$

- **迭代优化**（如 Matrix）：通过多轮迭代提升原则质量，形式化地可以表示为 $k^{(t+1)} = f_{refine}(k^{(t)}, \mathcal{E}\_{new}, \text{validation})$，其中 $\mathcal{E}\_{new}$ 是新证据，$\text{validation}$ 是验证结果

- **原则表示**：
  - **自然语言**：$k = \text{"Always verify user's timezone before scheduling meetings"}$
  - **规则形式**：$k = \text{IF } \text{condition} \text{ THEN } \text{action}$
  - **嵌入向量**：$\mathbf{v}_k = f_{embed}(k) \in \mathbb{R}^d$，用于相似性匹配

**挑战与解决方案**：

- **过度泛化（Over-generalization）**：
  - **问题**：从少数案例中提取过于宽泛的原则，即 $\text{support}(k) < \theta_{min}$ 但 $\text{scope}(k)$ 过大
  - **解决**：要求最小支持度 $\text{support}(k) \geq \theta_{min}$，其中 $\text{support}(k) = |\{e: k \text{ applies to } e\}|$，以及交叉验证 $\text{validate}(k, \mathcal{E}\_{test})$

- **原则冲突**：
  - **问题**：不同原则可能相互矛盾，即 $\exists k_1, k_2: \text{conflict}(k_1, k_2)$
  - **解决**：引入优先级 $\text{priority}(k_1) > \text{priority}(k_2)$ 或上下文条件 $\text{context}(k_1) \neq \text{context}(k_2)$

- **计算成本**：
  - **问题**：反思过程需要调用 LLM，成本高，即 $\text{cost}(\mathcal{F}) = O(|\mathcal{E}\_{selected}| \cdot \text{cost}\_{LLM})$
  - **解决**：批量处理 $\text{batch}(\mathcal{E})$、异步执行 $\text{async}(\mathcal{F})$、缓存结果 $\text{cache}(k)$

**代表工作**：

- **Matrix**（Liu et al., 2024）：从 Agent 轨迹中提炼任务无关原则，使用迭代反思循环，形式化地可以表示为 $k^{(t+1)} = f_{Matrix}(k^{(t)}, \mathcal{M}\_{episodic})$

- **Reflexion**（Shinn et al., 2023）：通过反思改进 Agent 的错误处理能力，形式化地可以表示为 $\text{error\_handling}\_{improved} = f_{Reflexion}(\text{error\_history}, \mathcal{M}\_{episodic})$

- **Self-Refine**（Madaan et al., 2023）：Agent 自我反思和迭代改进，形式化地可以表示为 $\text{output}\_{refined} = f_{SelfRefine}(\text{output}\_{initial}, \text{feedback})$

**评估方法**：

- **原则质量**：$\text{quality}(k) = f_{quality}(k, \mathcal{E})$，其中 $f_{quality}$ 是质量评估函数，可以通过人工评估、专家评审获得

- **泛化能力**：$\text{generalization}(k) = \frac{|\{e: k \text{ applies to } e\}|}{|\mathcal{E}\_{test}|}$，其中 $\mathcal{E}\_{test}$ 是测试事件集合

- **一致性**：$\text{consistency}(\{k_i\}) = 1 - \frac{|\text{conflicts}(\{k_i\})|}{|\{k_i\}|}$，其中 $\text{conflicts}$ 是冲突集合

<br>

#### Planning（规划）

**Planning（规划）**基于历史经验（特别是成功的执行计划）**指导未来行动的序列生成**。形式化地，规划可以表示为函数 $\mathcal{P}: \mathcal{M}\_{procedural} \times \mathcal{T} \rightarrow \Pi$，其中 $\mathcal{M}\_{procedural}$ 是程序记忆库，$\mathcal{T}$ 是任务空间，$\Pi$ 是计划空间。规划的目标是生成计划 $\pi = [a_1, a_2, \ldots, a_n]$，使得执行计划能够完成任务。

**核心机制**包括案例检索、计划复用、计划生成三个阶段：

- **案例检索**：从 procedural memory 中检索相似的历史任务，形式化地可以表示为 $\pi_{similar} = \arg\max_{\pi \in \mathcal{M}\_{procedural}} \text{sim}(\text{task}, \pi)$，其中 $\text{sim}$ 是相似度函数

- **计划复用**：直接复用或调整历史计划，形式化地可以表示为 $\pi_{new} = f_{adapt}(\pi_{old}, \text{task}\_{new})$，其中 $f_{adapt}$ 是适配函数

- **计划生成**：结合当前约束生成新计划，形式化地可以表示为 $\pi = f_{generate}(\mathcal{M}\_{procedural}, \text{task}, \text{constraints})$

**技术实现**包括 Case-Based Reasoning、Hierarchical Planning、Plan Schema：

- **Case-Based Reasoning（基于案例的推理）**：
  - **案例检索**：$\pi_{cases} = \mathcal{Q}\_{similar}(\text{task}, k) = \text{TopK}(\{\text{sim}(\text{task}, \pi_i)\}\_{i=1}^n)$，其中 $k$ 是返回数量
  - **计划适配**：$\pi_{adapted} = f_{adapt}(\pi_{best}, \text{task}\_{new})$，其中 $\pi_{best} = \arg\max_{\pi \in \pi_{cases}} \text{sim}(\text{task}\_{new}, \pi)$
  - **执行计划**：$\text{result} = \text{execute}(\pi_{adapted})$

- **Hierarchical Planning（分层规划）**：
  - **高层计划**：$\pi_{high} = [\text{Search}, \text{Filter}, \text{Select}, \text{Book}]$
  - **低层动作**：$\pi_{low} = \{\pi_{Search}, \pi_{Filter}, \pi_{Select}, \pi_{Book}\}$，其中每个子计划 $\pi_i$ 包含具体动作序列
  - **计划组合**：$\pi = \text{compose}(\pi_{high}, \pi_{low})$

- **Plan Schema（计划模板）**：
  - **模板定义**：$\text{Schema} = \langle \text{task\_type}, \text{template}, \text{adaptation\_rules} \rangle$，其中 $\text{template} = [\text{step}_1, \text{step}_2, \ldots, \text{step}_n]$
  - **计划实例化**：$\pi = f_{instantiate}(\text{Schema}, \text{parameters})$，其中 $\text{parameters}$ 是任务参数
  - **适配规则**：$\text{adaptation\_rules} = \{\text{if\_no\_results}: \text{suggest\_alternative}, \text{if\_booking\_fails}: \text{retry\_backup}\}$

**失败处理**包括 Plan Repair、Fallback Plans、Learning from Failure：

- **Plan Repair**：当计划执行失败时，自动修复，形式化地可以表示为 $\pi_{repaired} = f_{repair}(\pi_{failed}, \text{error})$，其中 $\text{error}$ 是错误信息

- **Fallback Plans**：准备备用计划，形式化地可以表示为 $\pi_{fallback} = f_{fallback}(\pi_{primary}, \text{failure\_scenarios})$，其中 $\text{failure\_scenarios}$ 是失败场景集合

- **Learning from Failure**：将失败案例存储，避免重复错误，形式化地可以表示为 $\mathcal{M}\_{procedural}' = \mathcal{M}\_{procedural} \cup \{(\pi_{failed}, \text{error}, \text{avoidance\_rule})\}$

**代表工作**：

- **Voyager**（Wang et al., 2023）：Minecraft Agent，将成功技能存储为可复用的代码计划，形式化地可以表示为 $\pi = f_{extract}(\text{successful\_trajectory})$

- **ReAct**（Yao et al., 2023）：将"思考-行动"模式存储为可复用的模板，形式化地可以表示为 $\pi = \text{"think"} \rightarrow \text{"act"}$

- **WebArena**（Zhou et al., 2023）：评估 Agent 在网页环境中的规划能力，形式化地可以表示为 $\text{score} = f_{eval}(\pi, \text{task})$

**评估指标**：

- **任务成功率**：$\text{success\_rate} = \frac{|\{\text{task}: \text{completed}(\pi, \text{task})\}|}{|\{\text{task}\}|}$，使用规划后的任务完成率

- **规划效率**：$\text{efficiency} = \frac{1}{\text{time}(\text{generate}(\pi))}$，生成计划的时间

- **计划质量**：$\text{quality}(\pi) = f_{quality}(|\pi|, \text{time}(\text{execute}(\pi)), \text{resources}(\pi))$，计划的步骤数、执行时间、资源消耗

<br>

#### Personalization（个性化）

**Personalization（个性化）**根据用户的**历史行为、偏好、上下文**定制 Agent 的响应和行为，实现"千人千面"的体验。形式化地，个性化可以表示为函数 $\mathcal{U}: \mathcal{M}\_{episodic} \rightarrow \mathcal{P}\_{user}$，其中 $\mathcal{P}\_{user}$ 是用户画像空间。个性化的目标是构建用户画像 $p_u$，使得 Agent 的响应 $a_t = f_{LLM}(s_t, p_u)$ 符合用户期望。

**核心组件**包括 User Profile 和 User Role Snapshot：

- **User Profile（用户画像）**：$p_u = \langle p_{explicit}, p_{implicit}, p_{context} \rangle$
  - **显式偏好**：$p_{explicit} = \{k: \text{user\_stated}(k)\}$，用户明确表达的偏好（如"我喜欢简洁的回答"）
  - **隐式偏好**：$p_{implicit} = f_{infer}(\mathcal{Q}\_{user}(u))$，从行为中推断的偏好（如频繁选择直飞航班），其中 $f_{infer}$ 是推断函数
  - **上下文信息**：$p_{context} = \{\text{timezone}, \text{language}, \text{device}\}$，时区、语言、设备类型等

- **User Role Snapshot（用户角色快照）**：从稀疏的 episodic memory 中提取稳定的用户特征，形式化地可以表示为 $\text{snapshot}(u) = f_{MOOM}(\mathcal{Q}\_{user}(u))$，构建用户角色的高层次描述

**技术实现**包括 MOOM 方法、Preference Modeling、Role Prompting、Dynamic Adaptation：

- **MOOM 方法**（Chen et al., 2025d）：
  - **Step 1**：收集稀疏交互轨迹（Episodic Memory），$\mathcal{E}_u = \mathcal{Q}\_{user}(u) = \{e: u(e) = u\}$，其中交互轨迹稀疏且噪声大
  - **Step 2**：提取高阶特征（LLM-based extraction），$p_u = f_{LLM}(\mathcal{E}_u, \text{prompt})$，其中 $\text{prompt} = \text{"Summarize this user's preferences and characteristics"}$，输出高阶特征如"prefers concise answers"、"values efficiency over detail"、"frequently books direct flights"
  - **Step 3**：构建稳定 Prompt Prefix，$\text{prompt}\_{prefix} = f_{construct}(p_u, u)$，形式为 "You are an assistant for [User], who: [preferences]"

- **Preference Modeling（偏好建模）**：
  - **Embedding-based**：将用户偏好编码为向量，$\mathbf{v}_u = f_{embed}(p_u) \in \mathbb{R}^d$
  - **Rule-based**：使用 IF-THEN 规则，$p_u = \{r: r = \text{IF } c \text{ THEN } a\}$
  - **Probabilistic**：使用概率模型（如贝叶斯网络），$P(a|s, u) = \text{estimate}(\mathcal{Q}\_{user}(u))$

- **Role Prompting（角色提示）**：$\text{system\_prompt} = f_{construct}(u, p_u, \mathcal{E}\_{recent})$，其中包含用户名称、用户画像、用户偏好、最近交互上下文

- **Dynamic Adaptation（动态适应）**：
  - **实时更新**：$p_u^{(t+1)} = f_{update}(p_u^{(t)}, e_{new})$，实时更新用户偏好
  - **变化检测**：$\text{change\_detected} = f_{detect}(p_u^{(t)}, p_u^{(t+1)})$，检测偏好变化
  - **冲突处理**：$p_u^{(t+1)} = f_{resolve}(p_u^{(t)}, p_{conflict})$，处理偏好冲突

**应用场景**：

- **对话风格**：根据用户偏好调整回答的详细程度，形式化地可以表示为 $a_t = f_{style}(s_t, p_u)$，其中 $f_{style}$ 是风格调整函数

- **内容推荐**：基于历史行为推荐相关内容，形式化地可以表示为 $\text{recommendations} = f_{recommend}(\mathcal{Q}\_{user}(u), \text{current\_context})$

- **界面定制**：根据用户习惯调整界面布局，形式化地可以表示为 $\text{layout} = f_{layout}(p_u)$

- **功能优先级**：优先展示用户常用的功能，形式化地可以表示为 $\text{priority} = f_{priority}(\text{frequency}(\text{features}, u))$

**代表工作**：

- **MOOM**（Chen et al., 2025d）：构建稳定的用户角色快照，形式化地可以表示为 $\text{snapshot}(u) = f_{MOOM}(\mathcal{Q}\_{user}(u))$

- **MemGPT**（Packer et al., 2023）：使用分层记忆实现个性化，形式化地可以表示为 $p_u = f_{MemGPT}(\mathcal{M}\_{hierarchical}, u)$

- **Personalized ChatGPT**：基于用户历史定制响应，形式化地可以表示为 $a_t = f_{ChatGPT}(s_t, \mathcal{Q}\_{user}(u))$

**评估指标**：

- **用户满意度**：$\text{satisfaction} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{rating}(q)$，通过问卷调查、评分获得

- **行为一致性**：$\text{consistency} = \frac{|\{a: a \text{ matches } \text{expectation}(u)\}|}{|\{a\}|}$，Agent 行为与用户期望的一致性

- **偏好准确率**：$\text{accuracy} = \frac{|\{p: p \in p_u \land p \text{ matches } p_{actual}\}|}{|p_u|}$，推断的偏好与实际偏好的匹配度

- **任务完成率**：$\text{completion\_rate} = \frac{|\{\text{task}: \text{completed}(\text{task}, p_u)\}|}{|\{\text{task}\}|}$，个性化后的任务成功率

**隐私考虑**：

- **数据最小化**：只收集必要的用户数据，$\mathcal{E}\_{collected} = f_{minimize}(\mathcal{E}\_{available}, \text{necessity})$

- **用户控制**：允许用户查看和修改个人数据，$\text{control} = \{\text{view}, \text{edit}, \text{delete}\}$

- **匿名化**：去除可识别信息，$p_u^{anonymized} = f_{anonymize}(p_u)$

- **合规性**：遵守 GDPR、CCPA 等隐私法规，$\text{compliance} = f_{check}(p_u, \text{regulations})$

<br>

#### Collaboration（协作）

**Collaboration（协作）**在**多 Agent 系统**中，Agent 之间**共享知识和经验**，实现协同工作。形式化地，协作可以表示为函数 $\mathcal{C}: \{\mathcal{M}_i\}\_{i=1}^n \rightarrow \mathcal{M}\_{shared}$，其中 $\{\mathcal{M}_i\}\_{i=1}^n$ 是多个 Agent 的记忆系统，$\mathcal{M}\_{shared}$ 是共享记忆系统。协作的目标是实现知识共享和协同工作，使得多 Agent 系统能够完成单个 Agent 无法完成的复杂任务。

**核心挑战**包括知识同步、冲突解决、隐私保护：

- **知识同步**：如何保持多个 Agent 的知识一致？形式化地可以表示为 $\text{sync}(\{\mathcal{M}_i\}\_{i=1}^n) = \mathcal{M}\_{shared}$，其中 $\text{sync}$ 是同步函数

- **冲突解决**：如何处理不同 Agent 的冲突记忆？形式化地可以表示为 $\text{resolve}(\{m_i: m_i \in \mathcal{M}_i \land \text{conflict}(m_i, m_j)\}) = m_{resolved}$，其中 $\text{resolve}$ 是冲突解决函数

- **隐私保护**：如何在共享知识的同时保护隐私？形式化地可以表示为 $\mathcal{M}\_{shared}^{private} = f_{privacy}(\mathcal{M}\_{shared})$，其中 $f_{privacy}$ 是隐私保护函数

**架构模式**包括 Centralized、Decentralized、Federated：

- **Centralized（中心化）**：所有 Agent 共享一个中心记忆系统，形式化地可以表示为 $\mathcal{M}\_{shared} = \bigcup_{i=1}^n \mathcal{M}_i$，其中所有 Agent 访问同一记忆系统
  - **优势**：简单、一致性强，即 $\text{consistency}(\mathcal{M}\_{shared}) = 1$
  - **劣势**：单点故障、扩展性差，即 $\text{availability} = \text{availability}(\text{central\_server})$

- **Decentralized（去中心化）**：Agent 之间直接通信，形式化地可以表示为 $\mathcal{M}_i^{(t+1)} = f_{sync}(\mathcal{M}_i^{(t)}, \{\mathcal{M}_j^{(t)}\}\_{j \in \text{neighbors}(i)})$，其中 $\text{neighbors}(i)$ 是 Agent $i$ 的邻居集合
  - **优势**：高可用、可扩展，即 $\text{availability} = \prod_{i=1}^n \text{availability}(\text{Agent}_i)$
  - **劣势**：需要一致性协议、复杂度高，即 $\text{complexity} = O(n^2)$

- **Federated（联邦）**：本地记忆聚合为共享知识，形式化地可以表示为 $\mathcal{M}\_{shared} = f_{aggregate}(\{\mathcal{M}_i^{local}\}\_{i=1}^n)$，其中 $\mathcal{M}_i^{local}$ 是 Agent $i$ 的本地记忆
  - **优势**：隐私保护、分布式，即 $\text{privacy} = f_{privacy}(\mathcal{M}\_{shared})$
  - **劣势**：聚合质量可能下降，即 $\text{quality}(\mathcal{M}\_{shared}) \leq \text{quality}(\bigcup_{i=1}^n \mathcal{M}_i)$

**技术实现**包括 Memory Broadcasting、Consensus Protocols、Knowledge Fusion、Privacy-Preserving Sharing：

- **Memory Broadcasting（记忆广播）**：Agent 将重要记忆广播给其他 Agent，形式化地可以表示为 $\text{broadcast}(m_i, \{\text{Agent}_j\}\_{j \neq i})$，使用消息队列（如 RabbitMQ、Kafka）实现

- **Consensus Protocols（一致性协议）**：
  - **Raft**：分布式一致性算法，形式化地可以表示为 $\text{consensus}(\{m_i\}\_{i=1}^n) = m_{agreed}$，其中 $m_{agreed}$ 是达成一致的记忆
  - **Paxos**：经典一致性算法，形式化地可以表示为 $\text{paxos}(\{m_i\}\_{i=1}^n) = m_{chosen}$
  - **CRDTs**：无冲突复制数据类型，形式化地可以表示为 $\text{merge}(\{m_i\}\_{i=1}^n) = m_{merged}$，其中 $\text{merge}$ 是合并函数

- **Knowledge Fusion（知识融合）**：合并来自多个 Agent 的知识，形式化地可以表示为 $\mathcal{M}\_{fused} = f_{fusion}(\{\mathcal{M}_i\}\_{i=1}^n)$，解决冲突的方法包括时间戳 $\text{resolve}(m_1, m_2) = \arg\max_{m \in \{m_1, m_2\}} t(m)$、置信度 $\text{resolve}(m_1, m_2) = \arg\max_{m \in \{m_1, m_2\}} \text{confidence}(m)$、投票 $\text{resolve}(\{m_i\}\_{i=1}^n) = \arg\max_{m} |\{i: m_i = m\}|$

- **Privacy-Preserving Sharing（隐私保护共享）**：
  - **Differential Privacy**：添加噪声保护隐私，形式化地可以表示为 $\mathcal{M}\_{private} = \mathcal{M} + \text{Laplace}(0, \epsilon)$，其中 $\epsilon$ 是隐私参数
  - **Federated Learning**：只共享模型参数，不共享原始数据，形式化地可以表示为 $\theta_{shared} = f_{aggregate}(\{\theta_i\}\_{i=1}^n)$，其中 $\theta_i$ 是 Agent $i$ 的模型参数
  - **Homomorphic Encryption**：加密状态下的计算，形式化地可以表示为 $\text{Enc}(\mathcal{M}\_{shared}) = f_{compute}(\text{Enc}(\mathcal{M}_1), \text{Enc}(\mathcal{M}_2))$

**代表工作**：

- **G-Memory**（Zhang et al., 2025c）：多 Agent 个性化记忆系统，形式化地可以表示为 $\mathcal{G}_i = f_{personalize}(\mathcal{G}\_{shared}, \text{Agent}_i)$

- **AutoGen**（Wu et al., 2023）：多 Agent 对话框架，支持记忆共享，形式化地可以表示为 $\text{conversation} = f_{AutoGen}(\{\text{Agent}_i\}\_{i=1}^n, \mathcal{M}\_{shared})$

- **CrewAI**：多 Agent 协作框架，形式化地可以表示为 $\text{task}\_{completed} = f_{CrewAI}(\{\text{Agent}_i\}\_{i=1}^n, \text{task})$

**应用场景**：

- **团队协作**：多个 Agent 协同完成复杂任务，形式化地可以表示为 $\text{result} = f_{collaborate}(\{\text{Agent}_i\}\_{i=1}^n, \text{task})$

- **知识库构建**：多个 Agent 共同构建知识库，形式化地可以表示为 $\mathcal{M}\_{KB} = \bigcup_{i=1}^n f_{contribute}(\text{Agent}_i)$

- **分布式学习**：Agent 之间共享学习经验，形式化地可以表示为 $\theta_{learned} = f_{distributed\_learning}(\{\theta_i\}\_{i=1}^n)$

**评估指标**：

- **知识共享效率**：$\text{efficiency} = \frac{|\mathcal{M}\_{shared}|}{\text{time}(\text{sync})}$，知识传播的速度和覆盖范围

- **一致性**：$\text{consistency} = 1 - \frac{|\text{conflicts}(\{\mathcal{M}_i\}\_{i=1}^n)|}{|\mathcal{M}\_{shared}|}$，不同 Agent 对同一知识的理解一致性

- **协作效果**：$\text{collaboration\_score} = \frac{|\{\text{task}: \text{completed}(\{\text{Agent}_i\}\_{i=1}^n, \text{task})\}|}{|\{\text{task}\}|}$，多 Agent 协作的任务完成率

<br>



## 5.记忆的动态机制（Dynamics of Memory）

> **核心观点**：记忆不是静态仓库，而是**活的认知器官**，必须随经验增长而演化。记忆系统需要持续地**巩固（Consolidation）**、**更新（Updating）** 和 **遗忘（Forgetting）**，以保持其有效性和相关性。

### Consolidation（巩固）

#### 定义

**Consolidation（巩固）**是将碎片化、冗余的短期记忆整合为**连贯、高效、可泛化的长期知识**的过程。形式化地，巩固可以表示为函数 $\mathcal{CON}: \mathcal{M}\_{short} \rightarrow \mathcal{M}\_{long}$，其中 $\mathcal{M}\_{short}$ 是短期记忆集合，$\mathcal{M}\_{long}$ 是长期记忆集合。巩固的目标是压缩记忆规模 $|\mathcal{M}\_{long}| < |\mathcal{M}\_{short}|$，同时保持信息完整性 $\text{info}(\mathcal{M}\_{long}) \approx \text{info}(\mathcal{M}\_{short})$。

#### 动机

**动机**包括存储效率、检索效率、泛化能力、认知合理性：

- **存储效率**：原始 episodic memory 存储开销大，需要压缩，形式化地可以表示为 $\text{storage}(\mathcal{M}\_{long}) < \text{storage}(\mathcal{M}\_{short})$，压缩比为 $\text{ratio} = \frac{|\mathcal{M}\_{short}|}{|\mathcal{M}\_{long}|}$

- **检索效率**：抽象后的记忆更容易检索和匹配，形式化地可以表示为 $\text{time}(\mathcal{R}(\mathcal{M}\_{long})) < \text{time}(\mathcal{R}(\mathcal{M}\_{short}))$

- **泛化能力**：从具体事件中提炼可复用的知识，形式化地可以表示为 $\text{generalization}(\mathcal{M}\_{long}) > \text{generalization}(\mathcal{M}\_{short})$

- **认知合理性**：模拟人类记忆的巩固过程（如睡眠中的记忆巩固），形式化地可以表示为 $\mathcal{CON}$ 模拟人类记忆巩固机制

**关键挑战**包括信息丢失、过度泛化、时机选择：

- **信息丢失**：抽象过程可能丢失重要细节，形式化地可以表示为 $\text{loss} = \text{info}(\mathcal{M}\_{short}) - \text{info}(\mathcal{M}\_{long})$，需要最小化 $\text{loss}$

- **过度泛化**：从少数案例中提取过于宽泛的原则，形式化地可以表示为 $\text{scope}(k) > \text{support}(k)$，其中 $\text{scope}(k)$ 是原则的适用范围，$\text{support}(k)$ 是支持案例数量

- **时机选择**：何时进行巩固？太早可能信息不足，太晚可能存储爆炸，形式化地可以表示为 $\text{trigger} = f_{timing}(|\mathcal{M}\_{short}|, \text{time}, \text{events})$

<br>

#### 三级巩固机制

记忆巩固遵循**层次化架构**，从局部到全局逐步抽象。形式化地，三级巩固可以表示为：

$$\mathcal{M}\_{long} = \mathcal{CON}\_{Global}(\mathcal{CON}\_{Cluster}(\mathcal{CON}\_{Local}(\mathcal{M}\_{short})))$$

其中 $\mathcal{CON}\_{Local}$ 是局部巩固，$\mathcal{CON}\_{Cluster}$ 是聚类融合，$\mathcal{CON}\_{Global}$ 是全局整合。

**层次关系**：
- **Level 0**: Raw Episodic Memory $\mathcal{M}\_{episodic}$
- **Level 1**: Session Summaries $\mathcal{M}\_{session} = \mathcal{CON}\_{Local}(\mathcal{M}\_{episodic})$
- **Level 2**: Behavior Patterns $\mathcal{M}\_{pattern} = \mathcal{CON}\_{Cluster}(\mathcal{M}\_{session})$
- **Level 3**: Universal Principles $\mathcal{M}\_{principle} = \mathcal{CON}\_{Global}(\mathcal{M}\_{pattern})$

**三级巩固机制**：

- **局部巩固（Local Consolidation）**：
  - **输入**：同一会话内的多条记录 $\mathcal{E}\_{session} = \{e_i\}\_{i=1}^n$
  - **处理方式**：LLM 总结、聚类、压缩，形式化地可以表示为 $s = f_{summarize}(\mathcal{E}\_{session})$ 或 $s = f_{cluster}(\mathcal{E}\_{session})$
  - **输出**：会话摘要、关键事件 $\mathcal{M}\_{session} = \{s_j\}\_{j=1}^m$，其中 $m < n$
  - **时间尺度**：分钟-小时
  - **代表工作**：Context Folding (Zhang et al., 2025q)、AgentFold (Ye et al., 2025a)

- **聚类融合（Cluster Fusion）**：
  - **输入**：跨会话相似事件 $\mathcal{E}\_{cross} = \bigcup_{i=1}^k \mathcal{E}\_{session_i}$
  - **处理方式**：嵌入聚类 + LLM 描述，形式化地可以表示为 $\mathcal{C} = \text{cluster}(\{\mathbf{v}_i = f_{embed}(e_i)\}\_{i=1}^n)$，然后 $p = f_{LLM}(\mathcal{C}, \text{prompt})$
  - **输出**：行为模式模板、用户画像 $\mathcal{M}\_{pattern} = \{p_j\}\_{j=1}^l$
  - **时间尺度**：天-周
  - **代表工作**：MOOM (Chen et al., 2025d)

- **全局整合（Global Integration）**：
  - **输入**：全部历史记忆 $\mathcal{M}\_{all} = \bigcup_{t=1}^T \mathcal{M}\_{pattern}^{(t)}$
  - **处理方式**：迭代反思 + 原则提炼，形式化地可以表示为 $k^{(t+1)} = f_{refine}(k^{(t)}, \mathcal{M}\_{all})$，直到收敛
  - **输出**：任务无关世界观、通用原则 $\mathcal{M}\_{principle} = \{k_i\}\_{i=1}^m$
  - **时间尺度**：周-月
  - **代表工作**：Matrix (Liu et al., 2024)

  <br>

#### 技术实现

##### Level 1: Local Consolidation（局部巩固）

**Local Consolidation（局部巩固）**的目标是压缩单个会话内的多条记录，生成会话摘要。形式化地，局部巩固可以表示为 $\mathcal{CON}\_{Local}: \mathcal{E}\_{session} \rightarrow \mathcal{S}\_{session}$，其中 $\mathcal{E}\_{session} = \{e_i\}\_{i=1}^n$ 是会话内的事件集合，$\mathcal{S}\_{session} = \{s_j\}\_{j=1}^m$ 是会话摘要集合，满足 $m < n$。

**触发条件**包括固定间隔、容量阈值、任务完成：

- **固定间隔**：每 N 步（如每 10 步）触发一次，形式化地可以表示为 $\text{trigger} = (t \bmod N = 0)$

- **容量阈值**：上下文窗口达到一定使用率（如 80%），形式化地可以表示为 $\text{trigger} = (\frac{|\mathcal{E}\_{session}|}{|C|} \geq \theta_{capacity})$，其中 $|C|$ 是上下文窗口大小，$\theta_{capacity} = 0.8$

- **任务完成**：任务或子任务完成时，形式化地可以表示为 $\text{trigger} = (\text{task\_status} = \text{complete})$

**技术方法**包括 Context Folding、AgentFold、Clustering-based Consolidation：

- **Context Folding**（Zhang et al., 2025q）：
  - **算法**：
    1. 收集当前会话的 N 条记录：$\mathcal{E}\_{fold} = \{e_i\}\_{i=t-N+1}^t$
    2. 使用 LLM 生成摘要：$s = f_{LLM}(\mathcal{E}\_{fold}, \text{prompt})$，其中 $\text{prompt} = \text{"Summarize the key actions and outcomes from the following interactions: [records]"}$
    3. 用摘要替换原始记录：$\mathcal{E}\_{session}' = (\mathcal{E}\_{session} \setminus \mathcal{E}\_{fold}) \cup \{s\}$
    4. 继续后续交互
  
  - **压缩比**：$\text{ratio} = \frac{|\mathcal{E}\_{fold}|}{1} = N$，通常 $N = 5$ 到 $10$

- **AgentFold**（Ye et al., 2025a）：
  - 专门针对工作记忆压缩，形式化地可以表示为 $\mathcal{M}\_{working}' = f_{AgentFold}(\mathcal{M}\_{working})$
  - 使用分层压缩策略，形式化地可以表示为 $\mathcal{M}\_{working}' = f_{hierarchical}(\mathcal{M}\_{working}, \text{levels})$
  - 保留关键决策点，形式化地可以表示为 $\mathcal{M}\_{key} = \{m: \text{importance}(m) > \theta_{key}\}$

- **Clustering-based Consolidation**：
  - **算法**：
    1. 将会话内的记录嵌入为向量：$\mathbf{V} = \{\mathbf{v}_i = f_{embed}(e_i)\}\_{i=1}^n$
    2. 使用 K-means 或 DBSCAN 聚类：$\mathcal{C} = \text{cluster}(\mathbf{V}, k)$，其中 $k$ 是聚类数量
    3. 对每个聚类生成摘要：$s_j = f_{summarize}(\mathcal{C}_j)$，其中 $\mathcal{C}_j$ 是第 $j$ 个聚类
    4. 保留聚类中心作为代表：$\mathcal{S}\_{session} = \{s_j\}\_{j=1}^k$

**参数设置**：

- **压缩比**：通常 5:1 到 10:1（5-10 条记录压缩为 1 条摘要），形式化地可以表示为 $\text{ratio} = \frac{|\mathcal{E}\_{session}|}{|\mathcal{S}\_{session}|} \in [5, 10]$

- **保留策略**：保留关键事件（如用户明确偏好、错误、重要决策），形式化地可以表示为 $\mathcal{E}\_{kept} = \{e: \text{importance}(e) > \theta_{keep}\}$，其中 $\text{importance}(e)$ 是重要性评分

- **质量评估**：使用 LLM 评估摘要是否保留了关键信息，形式化地可以表示为 $\text{quality}(s) = f_{LLM}(s, \mathcal{E}\_{session}, \text{"Does this summary preserve key information?"})$

**风险与缓解**：

- **Information Smoothing（信息平滑）**：
  - **问题**：独特事件被平均化，丧失个性，形式化地可以表示为 $\text{loss} = \text{uniqueness}(\mathcal{E}\_{session}) - \text{uniqueness}(\mathcal{S}\_{session})$
  - **缓解**：显式标记重要事件，避免压缩，即 $\mathcal{E}\_{important} = \{e: \text{important}(e)\}$，$\mathcal{E}\_{fold} = \mathcal{E}\_{session} \setminus \mathcal{E}\_{important}$；使用重要性评分 $\text{importance}(e)$ 指导压缩

##### Level 2: Cluster Fusion（聚类融合）

**Cluster Fusion（聚类融合）**的目标是识别跨会话的相似事件模式，生成行为模板。形式化地，聚类融合可以表示为 $\mathcal{CON}\_{Cluster}: \mathcal{M}\_{session} \rightarrow \mathcal{M}\_{pattern}$，其中 $\mathcal{M}\_{session} = \bigcup_{i=1}^k \mathcal{S}\_{session_i}$ 是跨会话的摘要集合，$\mathcal{M}\_{pattern} = \{p_j\}\_{j=1}^l$ 是行为模式集合。

**技术流程**包括事件嵌入、时间窗口聚类、模式提取三个阶段：

- **事件嵌入**：
  - 对每个 episodic record 进行嵌入：$\mathbf{v}_i = f_{embed}(e_i) \in \mathbb{R}^d$，其中 $d$ 是嵌入维度
  - 存储嵌入和元数据：$\mathcal{D} = \{(\mathbf{v}_i, \text{metadata}_i, t_i)\}\_{i=1}^n$，其中 $t_i$ 是时间戳

- **时间窗口聚类**：
  - **算法**：
    1. 选择时间窗口（如最近 30 天）：$\mathcal{E}\_{window} = \{e: t_{current} - t(e) \leq T_{window}\}$，其中 $T_{window} = 30$ 天
    2. 提取事件嵌入：$\mathbf{V}\_{window} = \{\mathbf{v}_i: e_i \in \mathcal{E}\_{window}\}$
    3. 使用 DBSCAN 或 Hierarchical Clustering：$\mathcal{C} = \text{cluster}(\mathbf{V}\_{window}, \text{method})$，其中 $\text{method} \in \{\text{DBSCAN}, \text{Hierarchical}\}$
    4. 识别相似事件簇：$\mathcal{C} = \{\mathcal{C}_1, \mathcal{C}_2, \ldots, \mathcal{C}_k\}$，其中每个簇 $\mathcal{C}_j$ 包含相似事件

- **模式提取**（MOOM 方法）：
  - **算法**：
    - 对每个聚类 $\mathcal{C}_j$：
      - **输入**：来自多个会话的相似事件 $\mathcal{E}_j = \{e: e \in \mathcal{C}_j\}$
      - **Prompt**：$\text{prompt} = \text{"Describe the common pattern in these events: [events]. Focus on: - User preferences, - Common behaviors, - Recurring patterns"}$
      - **输出**：行为模式模板 $p_j = f_{LLM}(\mathcal{E}_j, \text{prompt})$

**示例**：
- **聚类**：$\mathcal{C} = \{\text{Event1: User booked direct flight NYC→LAX}, \text{Event2: User booked direct flight NYC→SFO}, \text{Event3: User selected direct option over connecting}\}$
- **提取的模式**：$p = \text{"User consistently prefers direct flights over connecting flights"}$

**应用**：

- **用户画像构建**：从行为模式中提取用户特征，形式化地可以表示为 $p_u = f_{extract}(\mathcal{M}\_{pattern}, u)$

- **偏好推断**：识别用户的稳定偏好，形式化地可以表示为 $\text{preferences}(u) = \{p: p \in \mathcal{M}\_{pattern} \land \text{applies}(p, u)\}$

- **异常检测**：识别偏离模式的行为，形式化地可以表示为 $\text{anomaly}(e) = \min_{p \in \mathcal{M}\_{pattern}} \text{dist}(e, p) > \theta_{anomaly}$，其中 $\text{dist}$ 是距离函数，$\theta_{anomaly}$ 是异常阈值

<br>

##### Level 3: Global Integration（全局整合）

**Global Integration（全局整合）**的目标是从全部历史记忆中提炼**任务无关的通用原则**。形式化地，全局整合可以表示为 $\mathcal{CON}\_{Global}: \mathcal{M}\_{pattern} \rightarrow \mathcal{M}\_{principle}$，其中 $\mathcal{M}\_{pattern} = \bigcup_{t=1}^T \mathcal{M}\_{pattern}^{(t)}$ 是全部历史模式，$\mathcal{M}\_{principle} = \{k_i\}\_{i=1}^m$ 是通用原则集合。

**技术流程**（Matrix 方法）包括迭代反思循环、原则表示、原则质量评估：

- **迭代反思循环**：
  - **Round 1**: Initial Principle Extraction
    - **输入**：全部历史记忆 $\mathcal{M}\_{all} = \bigcup_{t=1}^T \mathcal{M}\_{pattern}^{(t)}$
    - **输出**：初始原则 $k^{(1)} = f_{extract}(\mathcal{M}\_{all})$，可能包含噪声
  - **Round 2**: Refinement
    - **输入**：初始原则 + 新证据 $k^{(2)} = f_{refine}(k^{(1)}, \mathcal{E}\_{new})$
    - **输出**：精炼后的原则
  - **Round 3**: Validation
    - **输入**：精炼后的原则 + 测试案例 $k^{(3)} = f_{validate}(k^{(2)}, \mathcal{E}\_{test})$
    - **输出**：验证后的原则
  - **迭代过程**：$k^{(t+1)} = f_{refine}(k^{(t)}, \mathcal{E}\_{new}, \text{feedback})$，直到收敛 $||k^{(t+1)} - k^{(t)}|| < \epsilon$

- **原则表示**：
  - **格式**：$k = \langle \text{id}, \text{statement}, \text{confidence}, \text{support\_count}, \text{violation\_count}, \text{context}, \text{derived\_from} \rangle$
  - **示例**：$k = \langle \text{"P001"}, \text{"Always verify user's timezone before scheduling"}, 0.95, 150, 3, \text{"scheduling tasks"}, \{\text{"event_123"}, \text{"event_456"}\} \rangle$

- **原则质量评估**：
  - **支持度**：$\text{support}(k) = |\{e: k \text{ applies to } e\}|$，有多少事件支持该原则
  - **置信度**：$\text{confidence}(k) = \frac{\text{support}(k)}{\text{support}(k) + \text{violation}(k)}$，原则的可靠性，其中 $\text{violation}(k)$ 是违反次数
  - **泛化能力**：$\text{generalization}(k) = \frac{|\{e \in \mathcal{E}\_{test}: k \text{ applies to } e\}|}{|\mathcal{E}\_{test}|}$，原则在新任务上的应用效果
  - **一致性**：$\text{consistency}(\{k_i\}) = 1 - \frac{|\text{conflicts}(\{k_i\})|}{|\{k_i\}|}$，原则之间是否冲突

**输出示例**：
- $k_1 = \langle \text{"P001"}, \text{"Users prefer concise answers over detailed explanations"}, 0.92, 200, 16 \rangle$
- $k_2 = \langle \text{"P002"}, \text{"Always confirm important details before taking actions"}, 0.88, 180, 23 \rangle$
- $k_3 = \langle \text{"P003"}, \text{"When multiple options exist, present them in order of user preference"}, 0.85, 150, 26 \rangle$

**挑战**：

- **计算成本**：需要处理全部历史，计算量大，形式化地可以表示为 $\text{cost}(\mathcal{CON}\_{Global}) = O(|\mathcal{M}\_{all}| \cdot \text{cost}\_{LLM})$

- **原则冲突**：不同原则可能相互矛盾，形式化地可以表示为 $\exists k_1, k_2: \text{conflict}(k_1, k_2)$，需要冲突解决机制

- **过度泛化**：从特定场景中提取过于宽泛的原则，形式化地可以表示为 $\text{scope}(k) > \text{support}(k)$，需要最小支持度约束

<br>

### Updating（更新）

#### 动机与挑战

**动机**：
- **世界变化**：外部世界不断变化（用户偏好改变、API 更新、知识更新）
- **错误修正**：发现记忆中的错误需要更正
- **知识演进**：新知识需要整合到现有记忆中

**核心挑战**：
- **变化检测**：如何识别需要更新的记忆？
- **冲突解决**：新旧记忆冲突时如何处理？
- **噪声区分**：如何区分**噪声**（one-off behavior）与**真实变化**？

#### 更新策略

##### Conflict Detection（冲突检测）

**机制**：
```
Algorithm:
1. 新证据到达
2. 检索相关旧记忆
3. 检测冲突：
   IF new_evidence contradicts old_memory:
       Calculate confidence scores
       Apply recency weighting
       Decide: update / keep / flag_for_review
```

**冲突示例**：
```
Old Memory: "User said hates coffee" (timestamp: 2025-01-01)
New Evidence: "User ordered coffee today" (timestamp: 2025-12-20)

Conflict Detected!
```

**解决方案**：

1. **置信度评分（Confidence Scoring）**：
   ```
   old_confidence = calculate_confidence(old_memory)
   new_confidence = calculate_confidence(new_evidence)
   
   IF new_confidence > old_confidence * threshold:
       Update memory
   ELSE:
       Keep old memory, flag for review
   ```

2. **时间衰减（Recency Weighting）**：
   ```
   weight = exp(-decay_rate * time_difference)
   effective_confidence = confidence * weight
   
   # 更近期的证据权重更高
   ```

3. **证据聚合（Evidence Aggregation）**：
   ```
   IF multiple_new_evidences support change:
       Update memory with higher confidence
   ELSE IF single_evidence:
       Flag for review, don't update immediately
   ```

##### Incremental Update（增量更新）

**机制**：
- **部分更新**：只更新记忆的特定字段，而非整体替换
- **版本控制**：保留历史版本，支持回滚
- **原子操作**：确保更新的原子性

**示例**：
```
Memory Before:
{
  "user_id": "U123",
  "preferences": {
    "flight_type": "direct",
    "seat_preference": "window"
  },
  "last_updated": "2025-01-01"
}

New Evidence: User now prefers aisle seats

Memory After:
{
  "user_id": "U123",
  "preferences": {
    "flight_type": "direct",  // unchanged
    "seat_preference": "aisle"  // updated
  },
  "last_updated": "2025-12-20",
  "version": 2
}
```

##### Parameter-based Update（基于参数的更新）

**定义**：将高频、关键的记忆**微调进模型参数**，而非存储在外部数据库。

**适用场景**：
- **用户身份**：姓名、角色、核心偏好
- **领域知识**：高频使用的专业知识
- **系统配置**：Agent 的行为模式

**技术实现**：

1. **LoRA Adapter**：
   ```
   Base Model: GPT-4
   ↓
   Fine-tune LoRA adapter on user-specific data
   ↓
   User-specific Model: GPT-4 + LoRA(user_id)
   ```

2. **Embedding Fine-tuning**：
   ```
   Fine-tune embedding model to better represent
   user-specific concepts and preferences
   ```

3. **Prompt Tuning**：
   ```
   Learn soft prompts that encode user preferences
   ```

**优势**：
- **快速访问**：无需检索，直接体现在模型输出中
- **个性化强**：模型本身适应特定用户

**劣势**：
- **成本高**：需要为每个用户维护模型副本
- **灵活性低**：难以快速更新

##### Temporal Update（时序更新）

**机制**：基于时间序列分析检测趋势变化。

```
Algorithm:
1. 收集时间序列数据：
   [preference_value(t1), preference_value(t2), ...]
   
2. 检测趋势：
   IF trend indicates change:
       Update memory
   ELSE IF single_outlier:
       Treat as noise, don't update
```

**示例**：
```
Coffee Preference Over Time:
  Jan: [hate, hate, hate] → Stable: "hates coffee"
  Dec: [like, like, like] → Trend detected: Update to "likes coffee"
```

#### 更新策略对比

| 策略 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **Conflict Detection** | 明确的冲突证据 | 精确、可控 | 需要明确的冲突信号 |
| **Incremental Update** | 部分信息变化 | 保留历史、可回滚 | 需要版本管理 |
| **Parameter-based** | 高频关键记忆 | 快速访问、个性化强 | 成本高、灵活性低 |
| **Temporal Update** | 渐进式变化 | 检测趋势、抗噪声 | 需要足够的历史数据 |


<br>

### Forgetting（遗忘）

#### 必要性与挑战

**必要性**：
- **存储限制**：防止存储空间无限增长
- **性能优化**：减少检索时间，提高系统响应速度
- **信息时效性**：移除过时、不再相关的信息
- **隐私合规**：遵守 GDPR "Right to be Forgotten" 等法规
- **认知合理性**：模拟人类记忆的自然遗忘过程

**核心挑战**：
- **重要性判断**：如何判断记忆的重要性？
- **删除 vs 归档**：是彻底删除还是归档？
- **影响评估**：删除记忆对系统性能的影响？

<br>

#### 遗忘策略

##### Time-based Expiration（基于时间的过期）

**机制**：
```
Algorithm:
FOR each memory:
    IF current_time - memory.timestamp > TTL:
        DELETE memory
```

**参数设置**：
- **TTL（Time To Live）**：根据记忆类型设置
  - Episodic memory: 30-90 天
  - Semantic memory: 1-2 年（或永久保留）
  - Procedural memory: 根据使用频率动态调整

**变体**：
- **Soft Expiration**：过期后标记为"不活跃"，延迟删除
- **Tiered Expiration**：不同重要性级别使用不同 TTL

**示例**：
```python
def expire_memories(memories, ttl_days=30):
    current_time = datetime.now()
    expired = []
    
    for memory in memories:
        age = (current_time - memory.timestamp).days
        if age > ttl_days:
            expired.append(memory)
    
    return expired
```

<br>

##### Access Frequency（访问频率）

**机制**：使用 **LRU（Least Recently Used）** 或 **LFU（Least Frequently Used）** 算法。

**LRU 实现**：
```
Algorithm:
1. 维护访问时间戳
2. 当存储空间不足时：
   DELETE memories with oldest access_time
```

**LFU 实现**：
```
Algorithm:
1. 维护访问计数器
2. 当存储空间不足时：
   DELETE memories with lowest access_count
```

**混合策略**：
```
Score = α * recency_score + β * frequency_score
DELETE memories with lowest scores
```

<br>

##### Redundancy Removal（冗余移除）

**机制**：识别并移除高度相似的记忆。

**技术流程**：
```
Algorithm:
1. 计算记忆之间的相似度矩阵
2. 识别相似度 > threshold 的记忆对
3. 合并或删除冗余记忆：
   Option A: 保留最详细的版本
   Option B: 合并为摘要
   Option C: 删除较旧的版本
```

**相似度计算**：
- **向量相似度**：cosine similarity > 0.95
- **文本相似度**：Jaccard similarity、编辑距离
- **语义相似度**：使用嵌入模型计算

**示例**：
```
Memory 1: "User prefers direct flights"
Memory 2: "User likes non-stop flights"  # Similarity: 0.98

Action: Merge into "User prefers direct/non-stop flights"
```
<br>

##### LLM-assisted Evaluation（LLM 辅助评估）

**机制**：使用 LLM 判断记忆的重要性。

**Prompt 设计**：
```
Evaluate the importance of the following memory for 
future agent tasks:

Memory: [memory_content]
Context: [current_context]

Questions:
1. Is this memory critical for future tasks? (Yes/No)
2. How frequently will this memory be needed? (High/Medium/Low)
3. Can this information be easily reacquired? (Yes/No)

Based on your evaluation, should this memory be:
- KEPT (high importance)
- ARCHIVED (medium importance, can be retrieved if needed)
- DELETED (low importance, can be reacquired)
```

**评估流程**：
```
FOR each memory:
    importance_score = llm_evaluate(memory)
    IF importance_score < threshold:
        DELETE or ARCHIVE memory
```

**优势**：
- **语义理解**：LLM 能理解记忆的语义重要性
- **上下文感知**：可以考虑当前上下文

**劣势**：
- **成本高**：每次评估都需要调用 LLM
- **一致性**：不同时间的评估可能不一致

<br>

##### Importance-based Forgetting（基于重要性的遗忘）

**重要性评分**：
```
importance_score = f(
    access_frequency,
    recency,
    user_feedback,
    task_success_contribution,
    uniqueness
)
```

**综合遗忘策略**：
```
Algorithm:
1. Calculate importance_score for each memory
2. Sort memories by importance_score
3. DELETE bottom N% memories
4. ARCHIVE memories in middle range
5. KEEP top memories
```
<br>

#### 遗忘策略对比

| 策略 | 触发条件 | 优势 | 劣势 | 适用场景 |
|------|---------|------|------|---------|
| **Time-based** | 超过 TTL | 简单、可预测 | 可能删除重要但旧的记忆 | 临时数据、日志 |
| **Access Frequency** | 很少被访问 | 保留常用记忆 | 可能删除重要但很少访问的记忆 | 缓存系统 |
| **Redundancy Removal** | 高度相似 | 减少存储、提高检索效率 | 可能丢失细微差异 | 大规模记忆库 |
| **LLM-assisted** | LLM 判断不重要 | 语义理解、上下文感知 | 成本高、可能不一致 | 高质量记忆库 |
| **Importance-based** | 重要性评分低 | 综合考虑多因素 | 需要设计评分函数 | 通用场景 |

<br>


#### 代码实现

**记忆清理任务（Memory Garbage Collection）**：
```python
def memory_garbage_collection(memory_db):
    """
    定期运行的记忆清理任务
    """
    # Step 1: Time-based expiration
    expired = expire_by_time(memory_db, ttl_days=30)
    
    # Step 2: Redundancy removal
    redundant = find_redundant(memory_db, similarity_threshold=0.95)
    
    # Step 3: Low-importance removal
    low_importance = filter_by_importance(memory_db, threshold=0.3)
    
    # Step 4: Archive (don't delete immediately)
    to_archive = expired + redundant + low_importance
    archive_memories(to_archive)
    
    # Step 5: Final deletion (after archive period)
    really_old = get_archived_older_than(days=90)
    delete_memories(really_old)
```

**最佳实践**：
- **渐进式删除**：先归档，再删除，避免误删
- **备份机制**：删除前备份重要记忆
- **监控指标**：跟踪删除率、检索性能变化
- **用户控制**：允许用户标记"重要记忆"，避免被删除

<br>

## 6.资源与框架

### Benchmark

#### 专为记忆设计的基准

##### AgentBoard（Zhou et al., 2024）

**特点**：
- **多任务场景**：涵盖对话、任务规划、工具使用等多种任务类型
- **长期交互**：模拟长期用户-Agent 交互（数周至数月）
- **用户模拟**：使用模拟用户生成多样化的交互模式
- **记忆显式评估**：直接测试记忆的保留、检索、更新能力

**评估指标**：
- **记忆保留率（Memory Retention Rate）**：长期记忆的保留比例
- **个性化准确率（Personalization Accuracy）**：Agent 行为与用户偏好的匹配度
- **检索精度（Retrieval Precision）**：检索到的记忆与查询的相关性
- **一致性（Consistency）**：Agent 在不同时间对同一用户的行为一致性

**任务示例**：
```
Task: Personal Assistant
- Day 1: User says "I prefer morning meetings"
- Day 10: User asks "Schedule a meeting"
- Expected: Agent should schedule in the morning
- Evaluation: Does agent remember the preference?
```

<br>

##### MemBench（Memory Benchmark）

**特点**：
- **显式测试记忆 CRUD 操作**：Create、Read、Update、Delete
- **覆盖记忆全生命周期**：从存储到检索到更新到遗忘
- **多记忆形式支持**：Episodic、Semantic、Procedural

**评估指标**：
- **检索精度（Retrieval Precision）**：Recall@K、Precision@K、MRR
- **更新一致性（Update Consistency）**：更新后记忆的一致性
- **遗忘合规性（Forgetting Compliance）**：遗忘策略的正确性
- **冲突解决（Conflict Resolution）**：新旧记忆冲突的处理能力

**测试场景**：
```
Scenario 1: Memory Creation
  - Store: "User prefers coffee"
  - Test: Can retrieve this memory?

Scenario 2: Memory Update
  - Old: "User prefers coffee"
  - New: "User now prefers tea"
  - Test: Is memory updated correctly?

Scenario 3: Memory Deletion
  - Store then delete: "Temporary note"
  - Test: Is memory deleted? Can it be retrieved?
```

<br>


##### LifelongQA（持续问答基准）

**特点**：
- **持续学习**：Agent 需要从持续到来的问题中学习
- **知识累积**：测试知识随时间累积的效果
- **遗忘测试**：测试长期记忆的保留能力

**评估指标**：
- **回答正确率随时间变化曲线**：学习曲线
- **知识保留率**：旧知识的记忆保留情况
- **新知识整合能力**：新知识的快速学习能力

**数据集**：
- 时间跨度：数月至数年
- 问题类型：事实性、推理性、多跳推理
- 知识领域：通用知识、领域特定知识

<br>

#### 通用 Agent 基准（隐含记忆需求）

这些基准虽未显式标注"memory"，但其特性天然要求强大的记忆系统：

##### WebArena（Zhou et al., 2023）

**特点**：
- **长周期网页操作**：需要执行多步骤、长时间的任务
- **状态管理**：需要记住之前的操作和状态
- **上下文依赖**：后续操作依赖前面的操作结果

**记忆需求**：
- **工作记忆压缩**：长序列任务需要压缩中间步骤
- **状态记忆**：记住当前任务的状态和进度
- **操作历史**：记录操作序列，支持错误回溯

**示例任务**：
```
Task: Book a flight
Steps:
  1. Search flights (NYC → LAX)
  2. Filter by preferences (direct, morning)
  3. Select flight
  4. Enter passenger info
  5. Confirm booking

Memory needed: Remember preferences, selected flight, 
               passenger info across steps
```

<br>


##### AgentBench（Liu et al., 2024）

**特点**：
- **工具使用**：测试 Agent 调用外部工具的能力
- **多步推理**：需要多步骤的推理和规划
- **任务多样性**：涵盖代码生成、数据分析、网页操作等

**记忆需求**：
- **程序记忆复用**：复用成功的工具调用模式
- **案例检索**：检索相似的历史任务案例
- **错误记忆**：记住失败的操作，避免重复错误


<br>

##### GAIA（Mialon et al., 2024）

**特点**：
- **真实世界问题**：需要整合来自多个来源的信息
- **多文档推理**：需要跨文档的记忆整合
- **长期依赖**：问题可能依赖之前的知识

**记忆需求**：
- **跨文档记忆整合**：整合来自多个文档的信息
- **知识图谱构建**：构建跨文档的知识图谱
- **长期知识保留**：保留之前学到的知识

<br>

#### 基准对比总结

| 基准 | 记忆显式性 | 时间跨度 | 主要评估维度 | 适用场景 |
|------|-----------|---------|------------|---------|
| **AgentBoard** | 显式 | 长期（周-月） | 保留、个性化、一致性 | 长期交互 Agent |
| **MemBench** | 显式 | 短期 | CRUD 操作、更新、遗忘 | 记忆系统开发 |
| **LifelongQA** | 显式 | 长期（月-年） | 知识累积、保留 | 持续学习系统 |
| **WebArena** | 隐含 | 短期（单任务） | 工作记忆、状态管理 | 网页操作 Agent |
| **AgentBench** | 隐含 | 短期（单任务） | 程序记忆、案例检索 | 通用 Agent |
| **GAIA** | 隐含 | 短期（单任务） | 跨文档整合 | 知识问答 |



<br>

### 开源框架与工具

#### 框架概览

##### G-Memory（Zhang et al., 2025c）

**核心创新**：
- **多 Agent 个性化记忆**：为每个 Agent 维护独立的记忆空间
- **图结构存储**：使用知识图谱组织记忆
- **角色特定提示**：将通用记忆转换为角色特定的 prompt

**技术栈**：
- **存储**：Neo4j 图数据库
- **检索**：Cypher 查询语言 + 向量相似性搜索
- **个性化**：LLM-based role prompting

**使用场景**：
- 多 Agent 协作系统
- 需要个性化记忆的场景
- 复杂关系查询需求

**示例**：
```python
# G-Memory 使用示例
memory = GMemory(user_id="U123", agent_role="medical_assistant")

# 存储记忆
memory.store(
    fact="User is a doctor",
    context="medical"
)

# 检索时，自动转换为角色特定提示
prompt = memory.retrieve_as_prompt(query="user background")
# Output: "The user is a medical professional (doctor), 
#          emphasize accuracy and evidence-based responses"
```

<br>

##### MA-RAG（Multi-Agent RAG, Nguyen et al., 2025）

**核心创新**：
- **检索后细粒度提取**：使用多个 Agent 对检索结果进行精细处理
- **查询分解**：将复杂查询分解为子查询
- **多 Agent 协作**：不同 Agent 负责不同的处理任务

**技术架构**：
```
Query → Query Decomposer Agent
      ↓
      [Sub-query 1, Sub-query 2, ...]
      ↓
      Retrieval (for each sub-query)
      ↓
      Extractor Agent (for each result)
      ↓
      Integration Agent
      ↓
      Final Context
```

**优势**：
- **精确提取**：只提取最相关的信息片段
- **减少噪声**：过滤掉不相关的检索结果
- **结构化输出**：输出结构化的上下文

<br>

##### ComoRAG（Wang et al., 2025f）

**核心创新**：
- **全局上下文整合**：Integration Agent 对多个检索文档进行语义对齐
- **Broad Contextual Grounding**：生成统一的、广泛的上下文摘要
- **避免信息碎片化**：将分散的信息整合为连贯的上下文

**技术流程**：
```
Multiple Retrieved Documents
    ↓
Integration Agent:
  - Semantic alignment
  - Cross-document reasoning
  - Unified summary generation
    ↓
Broad Contextual Grounding
    ↓
LLM with enriched context
```

**应用场景**：
- 需要整合多个来源信息的任务
- 复杂推理任务
- 知识问答系统

<br>


##### Matrix（Liu et al., 2024）

**核心创新**：
- **经验提炼为原则**：从 Agent 轨迹中提炼任务无关原则
- **迭代反思循环**：多轮迭代优化原则质量
- **原则驱动决策**：使用提炼的原则指导 Agent 行为

**技术流程**：
```
Agent Trajectories
    ↓
Reflection Loop:
  Round 1: Initial principle extraction
  Round 2: Refinement
  Round 3: Validation
  ...
    ↓
Task-agnostic Principles
    ↓
Principle-guided Agent Behavior
```

**优势**：
- **高度抽象**：原则可跨任务复用
- **持续改进**：通过迭代优化原则质量
- **可解释性**：原则提供决策的可解释性

<br>

#### 检索后处理（Post-Retrieval Processing）关键技术

**核心问题**：原始检索结果常**冗余、噪声大、缺乏结构**，需要后处理才能有效利用。
- **记忆质量 = 检索质量 × 后处理能力**。优秀的记忆系统不仅需要高效的检索，更需要智能的后处理来提取、整合、个性化记忆。

##### Query Decomposition（查询分解）

**方法**（MA-RAG）：
```
Complex Query: "What are the user's preferences for flights and hotels?"

Decomposed:
  - Sub-query 1: "User's flight preferences"
  - Sub-query 2: "User's hotel preferences"

For each sub-query:
  - Retrieve relevant memories
  - Extract most relevant snippets
  - Combine results
```

**优势**：
- **精确匹配**：每个子查询更精确，检索结果更相关
- **并行处理**：可以并行处理多个子查询
- **模块化**：便于组合和复用

<br>

##### Semantic Alignment（语义对齐）

**方法**（ComoRAG）：
```
Retrieved Documents:
  Doc 1: "User prefers direct flights"
  Doc 2: "User likes non-stop flights"
  Doc 3: "User booked direct flight NYC→LAX"

Semantic Alignment:
  - Recognize "direct" = "non-stop"
  - Align temporal information
  - Identify core concept: "prefers direct flights"

Unified Summary:
  "User consistently prefers direct/non-stop flights, 
   as evidenced by multiple bookings and stated preferences"
```

**技术实现**：
- **实体对齐**：识别不同文档中的相同实体
- **关系对齐**：识别相同的关系模式
- **时间对齐**：整合时间序列信息

<br>

##### Role-Specific Prompting（角色特定提示）

**方法**（G-Memory）：
```
Generic Memory: "User is a doctor"

For Medical Agent:
  Prompt: "The user is a medical professional (doctor). 
           Emphasize accuracy, evidence-based responses, 
           and use appropriate medical terminology."

For Customer Service Agent:
  Prompt: "The user is a doctor (medical professional). 
           They likely have limited time and value efficiency. 
           Be concise and respect their time constraints."
```

**优势**：
- **上下文适应**：同一记忆在不同上下文中发挥不同作用
- **个性化**：根据 Agent 角色定制记忆的使用方式
- **灵活性**：支持多 Agent 系统中的个性化


<br>

##### Reranking（重排序）

**方法**：
```
Initial Retrieval (Top 20):
  [Result 1, Result 2, ..., Result 20]

Cross-Encoder Reranking:
  For each (query, result) pair:
    score = cross_encoder(query, result)
  
  Sort by score
  Return Top 5
```

**优势**：
- **提高精度**：交叉编码器比双编码器更准确
- **考虑交互**：直接建模查询和结果的交互

**劣势**：
- **计算成本**：需要为每个候选结果计算，成本高
- **延迟**：增加检索延迟


<br>

##### Deduplication（去重）

**方法**：
```
Retrieved Results:
  Result 1: "User prefers direct flights"
  Result 2: "User likes non-stop flights"  # Similar
  Result 3: "User booked flight NYC→LAX"

Deduplication:
  - Calculate similarity between results
  - If similarity > threshold (e.g., 0.95):
      Keep most detailed version
      Remove duplicates
```

**实现**：
- **向量相似度**：使用嵌入计算相似度
- **文本相似度**：使用编辑距离、Jaccard 相似度
- **语义相似度**：使用 LLM 判断语义等价性
<br>

#### 其他框架

##### LangChain Memory

**特点**：
- **多种记忆后端**：支持向量数据库、SQL、内存等
- **易于集成**：与 LangChain 生态无缝集成
- **灵活配置**：支持多种记忆配置

**使用示例**：
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_texts(["..."])
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever()
)
```
<br>

##### LlamaIndex

**特点**：
- **高级检索策略**：子查询分解、时间加权检索等
- **多模态支持**：支持文本、图像等多模态记忆
- **可扩展性**：支持大规模记忆库

<br>

##### Mem0

**特点**：
- **自动记忆提取**：从对话中自动提取记忆
- **向量检索**：使用向量数据库进行高效检索
- **用户友好**：提供简单的 API
<br>



<br>

## 7. 未来展望

### Automation-Oriented Memory Design（自动化导向的记忆设计）

#### 研究动机

**当前问题**：
- **人工设计成本高**：需要领域专家设计记忆 schema
- **灵活性不足**：预定义的 schema 难以适应新场景
- **可扩展性差**：新任务需要重新设计记忆结构

**目标**：实现**自适应记忆架构**，系统能够自动发现和优化记忆形式。


#### 技术路线

##### 路线 1: Meta-Learning Memory Formats（元学习记忆格式）

**核心思想**：使用元学习算法学习最优的记忆表示形式。

**方法**：
```
Meta-Learning Framework:
  1. Task Distribution: Sample tasks from task distribution
  2. Memory Format Learning: Learn optimal format for each task
  3. Generalization: Extract general principles for format design
  4. Adaptation: Adapt format to new tasks
```

**挑战**：
- **评估标准**：如何定义"好"的记忆形式？
- **搜索空间**：记忆格式的搜索空间巨大
- **计算成本**：元学习需要大量计算资源


<br>

##### 路线 2: Automatic Schema Discovery（自动 Schema 发现）

**核心思想**：从原始日志中自动发现结构化的 schema。

**方法**：
```
Raw Logs → Pattern Mining → Schema Discovery → Structured Memory

Techniques:
  - Frequent Pattern Mining
  - Clustering-based Schema Inference
  - LLM-based Schema Extraction
```

**示例**：
```
Raw Logs:
  "User searched for flights NYC to LAX"
  "User booked flight DL456"
  "User canceled booking AA123"

Discovered Schema:
  {
    "event_type": "flight_operation",
    "fields": ["action", "origin", "destination", "flight_id"],
    "relationships": ["search → book → cancel"]
  }
```


<br>

##### 路线 3: Neural Architecture Search (NAS) for Memory

**核心思想**：使用神经架构搜索自动设计记忆架构。

**方法**：
```
Search Space:
  - Memory types (episodic, semantic, procedural)
  - Storage backends (vector, graph, SQL)
  - Retrieval strategies (dense, sparse, hybrid)
  - Consolidation mechanisms

Search Algorithm:
  - Reinforcement Learning
  - Evolutionary Algorithms
  - Gradient-based Methods
```

#### 代表工作

- **AutoMemory**（研究进行中）：使用强化学习自动设计记忆架构
- **SchemaLearner**：从日志中自动学习记忆 schema

#### 挑战与未来

**主要挑战**：
1. **评估难题**：如何评估记忆形式的质量？
2. **可解释性**：自动发现的架构可能难以解释
3. **计算成本**：搜索过程计算量大

**未来方向**：
- 开发记忆形式的质量评估指标
- 结合领域知识约束搜索空间
- 设计更高效的搜索算法


<br>

### Reinforcement Learning Integration（强化学习集成）

#### 研究动机

**核心思想**：将记忆操作（读/写/更新/遗忘）建模为**强化学习的动作空间**，让 Agent 学习何时、如何操作记忆。

**优势**：
- **自适应**：Agent 可以根据任务需求自适应地管理记忆
- **优化**：自动优化记忆策略，平衡存储成本和任务性能
- **端到端**：记忆管理和任务执行联合优化

#### 技术框架

##### 框架设计

**状态空间（State Space）**：
```
State = {
  current_task,
  memory_state,  # 当前记忆库状态
  context_window_usage,
  available_memory_space
}
```

**动作空间（Action Space）**：
```
Actions:
  - READ(memory_id, query)
  - WRITE(memory_content)
  - UPDATE(memory_id, new_content)
  - DELETE(memory_id)
  - CONSOLIDATE(memory_ids)
  - NO_OP  # 不操作记忆
```

**奖励函数（Reward Function）**：
```
Reward = α * task_success_reward 
       + β * memory_efficiency_reward
       - γ * memory_operation_cost

Where:
  task_success_reward: 任务完成质量
  memory_efficiency_reward: 记忆使用效率（如检索精度）
  memory_operation_cost: 记忆操作的计算成本
```

##### 算法设计

**DQN-based Memory Management**：
```python
class MemoryDQN:
    def __init__(self):
        self.q_network = QNetwork(state_dim, action_dim)
        self.memory_buffer = ReplayBuffer()
    
    def select_action(self, state):
        # ε-greedy exploration
        if random() < epsilon:
            return random_action()
        else:
            return argmax(self.q_network(state))
    
    def update(self, batch):
        # Standard DQN update
        loss = mse(q_predicted, q_target)
        optimize(loss)
```

**Policy Gradient Methods**：
```python
class MemoryPolicyGradient:
    def __init__(self):
        self.policy_network = PolicyNetwork()
    
    def select_action(self, state):
        action_probs = self.policy_network(state)
        return sample(action_probs)
    
    def update(self, trajectory):
        # REINFORCE or PPO update
        policy_loss = -log_prob * advantage
        optimize(policy_loss)
```

#### 挑战与解决

**挑战 1: 稀疏奖励**

**问题**：记忆操作的效果可能延迟显现，导致稀疏奖励。

**解决方案**：
- **Reward Shaping**：设计中间奖励信号
- **Hierarchical RL**：分层强化学习，不同层次不同时间尺度
- **Intrinsic Motivation**：使用内在动机（如好奇心）鼓励探索

**挑战 2: 长期信用分配**

**问题**：如何将任务成功归因到具体的记忆操作？

**解决方案**：
- **Attention Mechanisms**：使用注意力机制识别关键记忆
- **Credit Assignment**：设计专门的信用分配机制
- **Causal Analysis**：分析记忆操作与任务成功的因果关系

**挑战 3: 动作空间巨大**

**问题**：记忆操作的动作空间可能非常大（如选择哪个记忆读取）。

**解决方案**：
- **Hierarchical Actions**：分层动作空间
- **Parameterized Actions**：参数化动作
- **Neural Architecture**：使用神经网络压缩动作空间

#### 代表工作

- **MemRL**（研究进行中）：使用 DQN 学习记忆管理策略
- **Adaptive Memory Agent**：结合 PPO 和记忆管理的端到端系统

#### 未来方向

- **多任务学习**：在不同任务间共享记忆管理策略
- **迁移学习**：将学到的策略迁移到新任务
- **可解释性**：理解 Agent 为什么选择特定的记忆操作


<br>

### Multimodal Memory（多模态记忆）

#### 研究动机

**需求场景**：
- **具身智能体（Embodied AI）**：机器人需要记忆视觉场景
- **AR/VR 交互**：虚拟助手需要记忆用户的视觉环境
- **多媒体对话**：对话中包含图像、视频等多模态内容
- **跨模态检索**：用文本查询图像记忆，或反之

#### 技术路线

##### 路线 1: Multimodal Embedding Alignment（多模态嵌入对齐）

**核心思想**：将不同模态的内容映射到统一的嵌入空间。

**技术实现**：

1. **CLIP-based Alignment**：
   ```
   Text: "A red apple" → [0.1, 0.2, ..., 0.9]
   Image: [apple_image] → [0.12, 0.18, ..., 0.88]
   
   Similarity = cosine(text_embedding, image_embedding)
   ```

2. **LLM + Vision Encoder**：
   ```
   Text: LLM Embedding Model → Text Embedding
   Image: Vision Encoder (ViT) → Image Embedding
   Alignment: Cross-modal Attention → Unified Embedding
   ```

3. **Multimodal Fusion**：
   ```
   Text + Image → Multimodal Encoder → Joint Embedding
   ```

<br>

##### 路线 2: Cross-Modal Retrieval（跨模态检索）

**场景 1: Text → Image Memory**：
```
Query: "Show me the restaurant I visited last week"
↓
Text Embedding
↓
Similarity Search in Image Memory
↓
Retrieved Images: [restaurant_photo_1, restaurant_photo_2]
```

**场景 2: Image → Text Memory**：
```
Query Image: [screenshot of booking confirmation]
↓
Image Embedding
↓
Similarity Search in Text Memory
↓
Retrieved Text: "Flight booking confirmed: DL456"
```


<br>

##### 路线 3: Multimodal Consolidation（多模态巩固）

**挑战**：如何整合来自不同模态的记忆？

**方法**：
```
Multimodal Events:
  - Text: "User visited restaurant X"
  - Image: [photo of restaurant X]
  - Audio: [voice note about restaurant X]

Consolidation:
  1. Extract key information from each modality
  2. Align semantic content
  3. Generate unified multimodal memory
```


<br>

#### 代表工作

##### Voyager（Minecraft Agent, Wang et al., 2023）

**特点**：
- **视觉-动作记忆**：记忆 Minecraft 世界的视觉场景
- **技能复用**：将成功的视觉-动作序列存储为可复用技能
- **跨场景泛化**：将学到的技能应用到新场景

**技术**：
- 使用视觉编码器提取场景特征
- 将视觉特征与动作序列关联
- 存储为可执行的程序记忆

##### Multimodal RAG Systems

**特点**：
- 支持文本、图像、视频等多种模态
- 跨模态检索和整合
- 统一的多模态上下文

#### 挑战与未来方向

**主要挑战**：
1. **模态对齐**：不同模态的语义对齐困难
2. **存储开销**：多模态数据存储开销大
3. **检索效率**：跨模态检索计算成本高
4. **质量评估**：如何评估多模态记忆的质量？

**未来方向**：
- 开发更高效的多模态嵌入模型
- 设计专门的多模态记忆架构
- 建立多模态记忆的评估基准


<br>

### Shared Memory in Multi-Agent Systems（多 Agent 系统中的共享记忆）

####  研究动机

**需求场景**：
- **团队协作**：多个 Agent 协同完成复杂任务
- **知识共享**：Agent 之间共享学习到的知识
- **一致性保证**：确保不同 Agent 对同一知识的理解一致

#### 架构模式

##### 模式 1: Centralized（中心化）

**架构**：
```
[Agent 1] ──┐
[Agent 2] ──┼──> [Shared Memory Server]
[Agent 3] ──┘
```

**优势**：
- **简单**：架构简单，易于实现
- **一致性强**：所有 Agent 访问同一记忆库，一致性高
- **易于管理**：集中管理，便于监控和调试

**劣势**：
- **单点故障**：Shared Memory Server 故障影响所有 Agent
- **扩展性差**：成为系统瓶颈
- **延迟**：所有 Agent 都需要通过网络访问


<br>

##### 模式 2: Decentralized（去中心化）

**架构**：
```
[Agent 1] <──> [Agent 2]
   ↕              ↕
[Agent 3] <──> [Agent 4]
```

**优势**：
- **高可用**：无单点故障
- **可扩展**：易于水平扩展
- **低延迟**：本地访问，延迟低

**劣势**：
- **一致性协议**：需要复杂的一致性协议（如 Raft、Paxos）
- **复杂度高**：实现和维护复杂
- **冲突解决**：需要处理冲突和合并


<br>

##### 模式 3: Federated（联邦）

**架构**：
```
[Local Memory 1] ──┐
[Local Memory 2] ──┼──> [Aggregated Knowledge]
[Local Memory 3] ──┘
```

**优势**：
- **隐私保护**：原始数据不离开本地
- **分布式**：支持大规模分布式部署
- **合规性**：符合数据保护法规

**劣势**：
- **聚合质量**：聚合后的知识质量可能下降
- **通信成本**：需要频繁通信同步
- **延迟**：聚合过程增加延迟



<br>

#### 关键技术

##### 技术 1: Consensus Protocols（一致性协议）

**Raft 算法**：
```
Leader Election → Log Replication → Safety Guarantees

Key Features:
  - Leader-based: 选举 leader 协调更新
  - Majority Consensus: 需要多数节点同意
  - Strong Consistency: 保证强一致性
```

**CRDTs（Conflict-free Replicated Data Types）**：
```
Key Idea: 设计数据结构，使得任意顺序的更新都能收敛到相同状态

Example: G-Counter (Grow-only Counter)
  - Each node maintains local counter
  - Merge: sum of all local counters
  - Always converges to correct value
```

<br>

##### 技术 2: Knowledge Fusion（知识融合）

**冲突解决策略**：
```
Strategy 1: Time-based
  - Newer memory wins
  - Use timestamp to resolve conflicts

Strategy 2: Confidence-based
  - Higher confidence wins
  - Use confidence scores

Strategy 3: Voting
  - Majority vote
  - Use consensus among agents

Strategy 4: Merge
  - Combine conflicting memories
  - Use LLM to merge semantically
```


<br>

##### 技术 3: Privacy-Preserving Sharing（隐私保护共享）

**Differential Privacy**：
```
Add noise to shared memories:
  memory' = memory + Laplace(0, ε)

Properties:
  - Privacy: Individual data protected
  - Utility: Aggregate statistics preserved
```

**Federated Learning**：
```
Only share model parameters, not raw data:
  1. Each agent trains local model
  2. Share model parameters (not data)
  3. Aggregate parameters
  4. Distribute aggregated model
```

**Homomorphic Encryption**：
```
Compute on encrypted data:
  Encrypted_Memory_1 + Encrypted_Memory_2
  = Encrypt(Memory_1 + Memory_2)

Allows computation without decryption
```

<br>

####  代表工作

- **G-Memory**（Zhang et al., 2025c）：多 Agent 个性化记忆系统
- **AutoGen**（Wu et al., 2023）：多 Agent 对话框架，支持记忆共享
- **CrewAI**：多 Agent 协作框架


<br>

#### 挑战与未来

**主要挑战**：
1. **一致性 vs 性能**：强一致性影响性能
2. **冲突解决**：如何智能地解决冲突？
3. **隐私保护**：如何在共享的同时保护隐私？
4. **可扩展性**：如何扩展到大规模系统？

**未来方向**：
- 设计更高效的一致性协议
- 开发智能的冲突解决机制
- 研究隐私保护的新技术
- 建立多 Agent 记忆系统的评估基准



<br>

### Trustworthy Memory Systems（可信记忆系统）

#### 研究动机

**可信性维度**：
- **Factuality（事实性）**：记忆是否准确？
- **Explainability（可解释性）**：为什么使用这个记忆？
- **Robustness（鲁棒性）**：能否抵抗攻击？
- **Ethics & Privacy（伦理与隐私）**：是否符合伦理和隐私要求？

#### 关键技术

##### 技术 1: Factuality（事实性）

**问题**：记忆可能包含错误信息，如何保证事实性？

**解决方案**：

1. **Provenance Tracking（引用溯源）**：
   ```
   Memory:
     Content: "User prefers coffee"
     Provenance:
       - Source: User statement on 2025-01-01
       - Confidence: 0.95
       - Supporting Evidence: [event_1, event_2, event_3]
   ```

2. **Fact Verification（事实核查）**：
   ```
   Before storing memory:
     1. Extract claim from memory
     2. Verify against knowledge base
     3. Check consistency with existing memories
     4. Assign confidence score
     5. Store with verification metadata
   ```

3. **Confidence Scoring（置信度评分）**：
   ```
   Confidence = f(
       source_reliability,
       supporting_evidence_count,
       consistency_with_other_memories,
       recency
   )
   ```


<br>


##### 技术 2: Explainability（可解释性）

**问题**：用户和开发者需要理解 Agent 为什么使用特定记忆。

**解决方案**：

1. **Memory Usage Logging（记忆使用日志）**：
   ```
   Log Entry:
     Timestamp: 2025-12-20 14:30:00
     Query: "What are user's flight preferences?"
     Retrieved Memories: [mem_123, mem_456]
     Reasoning: "mem_123 matches query semantically, 
                 mem_456 provides additional context"
     Final Decision: Used mem_123 as primary, 
                     mem_456 as supporting
   ```

2. **Attention Visualization（注意力可视化）**：
   ```
   Show which parts of memory were most important:
     Memory: "User prefers direct flights"
     Attention: [0.1, 0.8, 0.1]  # "direct" has high attention
   ```

3. **Explanation Generation（解释生成）**：
   ```
   LLM-generated Explanation:
     "I retrieved this memory because it directly answers 
      your question about flight preferences. The memory 
      indicates you prefer direct flights, which matches 
      the current query context."
   ```

<br>

##### 技术 3: Robustness（鲁棒性）

**问题**：记忆系统可能受到攻击（如记忆污染、对抗样本）。

**攻击类型**：
- **Memory Poisoning**：注入错误记忆
- **Adversarial Retrieval**：构造查询导致检索错误记忆
- **Data Manipulation**：篡改存储的记忆

**防御措施**：

1. **Adversarial Training（对抗训练）**：
   ```
   Training with adversarial examples:
     - Generate adversarial memories
     - Train model to resist attacks
     - Improve robustness
   ```

2. **Input Validation（输入验证）**：
   ```
   Before storing memory:
     1. Validate format
     2. Check for suspicious patterns
     3. Verify source authenticity
     4. Sanitize content
   ```

3. **Anomaly Detection（异常检测）**：
   ```
   Monitor for anomalies:
     - Unusual memory patterns
     - Suspicious retrieval patterns
     - Unexpected memory updates
   ```

   
<br>

##### 技术 4: Ethics & Privacy（伦理与隐私）

**伦理问题**：
- **偏见**：记忆可能包含偏见信息
- **公平性**：不同用户群体的记忆质量是否公平？
- **透明度**：用户是否知道 Agent 记住了什么？

**隐私问题**：
- **数据收集**：收集哪些数据？
- **数据使用**：如何使用收集的数据？
- **数据删除**：如何删除用户数据（GDPR Right to be Forgotten）？

**解决方案**：

1. **User Control（用户控制）**：
   ```
   Features:
     - View stored memories
     - Edit memories
     - Delete memories
     - Control what is remembered
     - Opt-out of memory collection
   ```

2. **Privacy-Preserving Techniques（隐私保护技术）**：
   ```
   - Differential Privacy
   - Federated Learning
   - Homomorphic Encryption
   - Data Minimization
   ```

3. **Bias Mitigation（偏见缓解）**：
   ```
   - Regular auditing for bias
   - Diverse training data
   - Fairness constraints
   - Bias detection and correction
   ```

4. **Compliance（合规性）**：
   ```
   - GDPR compliance
   - CCPA compliance
   - Automatic data deletion
   - Consent management
   ```



<br>

#### 评估框架

**可信性评估指标**：

| 维度 | 评估指标 | 方法 |
|------|---------|------|
| **Factuality** | 准确率、置信度、一致性 | 人工评估、自动验证 |
| **Explainability** | 解释质量、用户理解度 | 用户研究、专家评估 |
| **Robustness** | 对抗准确率、攻击成功率 | 对抗测试、压力测试 |
| **Ethics & Privacy** | 合规性、用户满意度 | 合规审计、用户调查 |


<br>

#### 代表工作

- **Factual Memory Systems**：专注于事实性保证的记忆系统
- **Explainable Memory Agents**：提供可解释性的 Agent
- **Privacy-Preserving Memory**：隐私保护记忆系统

#### 挑战与未来

**主要挑战**：
1. **平衡**：可信性与性能的平衡
2. **标准**：缺乏统一的可信性标准
3. **评估**：可信性的评估方法不完善
4. **成本**：可信性技术增加系统成本

**未来方向**：
- 建立可信性标准和评估框架
- 开发更高效的可信性技术
- 研究可信性与性能的平衡
- 加强跨学科合作（AI、法律、伦理）

<br>

### 核心洞见与未来方向

#### 核心洞见

**记忆是 Agent 认知跃迁的关键**：从"反应式"到"认知式"，记忆系统使 Agent 能够个性化、提升效率、保持一致性、持续学习。

**Forms–Functions–Dynamics 三维协同**：三个维度必须协同设计，孤立优化任一维度效果有限。Forms 决定 Functions，Functions 影响 Dynamics，Dynamics 优化 Forms。

**当前局限**：主流方法依赖人工设计、静态架构、单一形式。未来方向是自动化、自适应、混合架构。

#### 研究趋势

**技术趋势**：
- **自动化与自适应**：从人工设计转向自动发现和优化记忆架构（Meta-learning、NAS、RL）
- **多模态扩展**：从文本记忆扩展到图像、音频、视频等多模态记忆
- **可信性增强**：关注事实性、可解释性、鲁棒性、隐私保护
- **大规模部署**：从研究原型转向生产环境部署（分布式系统、边缘计算）

**应用趋势**：
- **领域扩展**：从对话系统扩展到具身智能、AR/VR、多 Agent 协作、边缘设备
- **用户参与**：从黑盒系统转向透明、可控、可修正的记忆系统

#### 未来建议

**研究建议**：
1. **建立综合 Benchmark**：覆盖存储、检索、更新、遗忘全生命周期的评估框架
2. **开发轻量动态机制**：适用于边缘设备的记忆系统（模型压缩、高效检索、增量更新）
3. **人本记忆设计**：透明性、可控性、可修正性，让用户参与记忆管理
4. **神经符号融合**：结合 LLM 灵活性与符号系统可验证性（混合架构、符号引导、神经执行）

**工程建议**：
1. **模块化设计**：记忆系统作为独立模块，标准化接口，支持插件扩展
2. **性能优化**：缓存策略、异步处理、批量操作
3. **监控与调试**：记忆使用监控、性能指标跟踪、调试工具
4. **安全与隐私**：数据加密、访问控制、审计日志

<br>