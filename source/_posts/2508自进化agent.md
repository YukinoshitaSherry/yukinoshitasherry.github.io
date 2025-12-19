---
title: Self-Evolving Agent综述整理 
date: 2025-08-21
categories:
  - 学AI/DS
tags:
  - LLM
  - Agent
desc: 整理自 https://arxiv.org/abs/2507.21046 《A Survey of Self-Evolving Agents- On Path to Artificial Super Intelligence》
---

综述地址：
- <a href="https://arxiv.org/abs/2507.21046"> A Survey of Self-Evolving Agents- On Path to Artificial Super Intelligence </a>
<br>


此方向是直接面向未来的、关于从AGI到ASI的研究的组成部分，值得探索。
正在开启相关项目，先进行初步整理。

> "It is not the most intellectual of the species that survives; it is not the strongest that survives; but the species that survives is the one that is able best to adapt and adjust to the changing environment in which it finds itself."  
> —— Charles Darwin


<br>

## 背景


当前主流大型语言模型（Large Language Models, LLMs）在完成预训练和指令微调（Instruction Tuning）后即进入“冻结”状态（frozen state），其行为完全由输入提示（prompt）驱动。这种**静态部署范式**（static deployment paradigm）虽在封闭任务上表现优异，但在开放、动态、长期交互环境中暴露出根本性局限：

- **知识固化**（Knowledge Staleness）：无法吸收新事实（如 2025 年 FDA 批准的新药名称），导致回答过时；
- **策略僵化**（Strategy Rigidity）：面对结构新颖的任务（如多跳工具调用链、跨域推理）缺乏适应能力；
- **错误不可逆**（Error Irreversibility）：一旦推理路径偏离，无法在后续步骤中自我纠正或从失败中学习。

为突破上述瓶颈，研究者提出 **Self-Evolving Agents（自进化智能体）** ——一种能在部署后持续优化自身能力的系统。其核心理念并非“一次训练、终身使用”，而是“**终身学习、动态演化**”（lifelong learning with dynamic evolution）。这类智能体不仅响应环境，更能主动改造自身内部结构以提升未来性能。



<br>

## 概念


自进化智能体标志着 AI 从 **静态模型** 向 **动态生命体** 的范式转变。其核心在于：
- **多组件协同演化**（model + context + tool + architecture）；
- **主动环境交互与自我反思**；
- **无需人工干预的终身学习能力**。

<br>


### 形式化定义

论文将外部环境建模为 **部分可观测马尔可夫决策过程**
（Partially Observable Markov Decision Process, POMDP）：
$$
E = (G, S, A, T, R, \Omega, O, \gamma)
$$
其中各要素定义如下：
- $ G $：目标集合（goal set），通常由用户自然语言指令实例化；
- $ S $：状态空间（state space），包含环境状态与智能体内存状态；
- $ A $：动作空间（action space），包括工具调用（tool invocation）、推理步骤生成（reasoning step generation）、记忆更新（memory update）等；
- $ T(s'|s,a) $：状态转移函数（transition function），描述执行动作 $ a $ 后状态如何变化；
- $ R(s,a) $：奖励函数（reward function），可为标量数值（如任务完成度）或文本反馈（如“你的计划忽略了权限限制”）；
- $ \Omega $：观测空间（observation space），即智能体在每一步可感知的信息（如网页 HTML、API 返回结果）；
- $ O(o|s) $：观测概率（observation probability），表示在真实状态 $ s $ 下获得观测 $ o $ 的可能性；
- $ \gamma \in [0,1] $：折扣因子（discount factor），用于权衡即时与长期回报。

在此框架下，智能体系统被形式化为四元组：
$$
\psi = (\mathcal{M}, \mathcal{C}, \mathcal{T}, \mathcal{A})
$$
- $ \mathcal{M} $：模型组件（Model），即底层 LLM backbone（如 Llama-3、Qwen-Max）；
- $ \mathcal{C} $：上下文组件（Context），包含 prompt 模板、短期记忆缓存、长期记忆库；
- $ \mathcal{T} $：工具组件（Tool），即可供调用的外部功能集合（如搜索引擎、代码解释器、数据库接口）；
- $ \mathcal{A} $：架构组件（Architecture），指智能体内部模块的组织方式（如是否包含 planner、executor、reflector 等角色，以及它们之间的数据流拓扑）。

**自进化的目标**是寻找一个演化策略 $ f $，使得在给定任务序列 $ \{T\_j\}\_{j=1}^n $ 上最大化累计效用（cumulative utility）：
$$
\max_f \sum_{j=1}^n U(\Pi\_j, T\_j)
$$
其中：
- $ \Pi_j = f(\psi\_{j-1}, T\_j) $ 表示第 $ j $ 个任务所采用的策略（即更新后的智能体配置）；
- $ U $ 是一个复合效用函数，综合衡量任务成功率（task success rate）、执行效率（steps/time）、安全性（safety compliance）、资源消耗（compute cost）等多个维度。

该公式揭示了自进化智能体的本质：**通过策略 $ f $ 对四元组 $ \psi $ 进行迭代优化，以实现跨任务性能的持续提升**。

<br>


### 对比其他范式


<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251220012415249.png" style="width:80%"><br>


**关键区别**：自进化智能体是唯一同时具备 **结构性自修改能力**（structural self-modification）、**主动环境交互**（active exploration） 和 **多组件协同演化**（multi-component co-evolution） 的范式。其他方法仅在单一维度上改进，无法实现系统级适应。



<br>

## 核心维度：What / When / How / Where

为系统理解自进化机制，论文提出四维分析框架（Four-Dimensional Framework），分别对应“演化什么”、“何时演化”、“如何演化”、“在何场景演化”。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251220012300831.png"/><br>

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251220012247691.png"/><br>

### What to Evolve（演化对象）

自进化智能体可对以下四个核心组件进行修改：

#### Model（模型参数）
- **机制**：
  - **监督微调**（Supervised Fine-Tuning, SFT）：利用 self-generated 成功轨迹作为训练数据。
    - 例：**SCA**（Self-Challenging Agent）让 LLM 生成 Code-as-Task 问题（如"实现快速排序"）→ 执行代码 → 验证输出正确性 → 保留成功样本 → 微调模型。此过程形成闭环，使模型在编程能力上持续提升。
  - **强化学习**（Reinforcement Learning, RL）：以任务完成度为 reward，通过策略梯度方法（如 PPO）更新 policy。
    - 例：**RAGen**（Reinforcement Learning for Agent Generation）在 multi-turn 对话环境中使用 PPO 优化 agent 的 action selection policy，同时联合优化 memory retrieval 策略。
- **挑战**：计算成本高、需高质量 reward signal、易受 reward hacking 影响。

#### Context（上下文）
上下文包含 **Prompt** 与 **Memory**，是无需修改模型即可实现快速适应的关键。

- **Prompt Optimization**（提示优化）：
  - **SPO**（Self-supervised Prompt Optimization）：LLM 自动生成 instruction-response 对，通过 self-preference 比较（如"哪个回答更简洁？"）优化 prompt 模板。
  - **PromptAgent**：将 prompt 优化视为 planning 问题，使用 Monte Carlo Tree Search（MCTS）搜索最优 prompt 序列（如"先 plan 再 act" vs "边做边想"）。
  - **TextGrad**：将 prompt 视为可微变量，通过"文本梯度"反向传播误差。例如，若输出被评价为"太冗长"，则自动在 prompt 中加入"concise"或"limit to 3 sentences"等约束。

- **Memory Evolution**（记忆演化）：
  - **短期记忆**（Short-term Memory）：上下文窗口内的 token 缓存，支持 ADD/UPDATE/DELETE 操作。
    - **Mem0**：构建 production-ready 的向量数据库存储 episodic memory（如"用户昨天问过糖尿病用药"），支持基于语义相似度的检索。
  - **长期记忆**（Long-term Memory）：结构化知识图谱或摘要库，用于跨会话知识积累。
    - **Agent Workflow Memory**：记录 multi-turn interaction 中的 workflow state（如 tool call history、subgoal status、失败原因），并在 future task 中复用。例如，若上次因"未登录"失败，则下次自动插入 login 步骤。
  - **动态剪枝**（Dynamic Pruning）：基于遗忘曲线（forgetting curve）或 utility score（如访问频率、任务相关性）删除低价值记忆，防止 memory bloat。

#### Tool（工具）
工具是智能体与外部世界交互的桥梁，其演化包括创建、掌握与选择。

- **Tool Creation**（工具创建）：
  - **Voyager**：在《我的世界》（Minecraft）中自主发明新工具（如自动采矿脚本），通过"生成代码 → 执行 → 验证效果 → 保存成功脚本"闭环实现 open-ended invention。
  - **CREATOR**：区分 abstract reasoning（高层规划）与 concrete reasoning（底层执行），自动生成专用工具函数（如"calculate_tax(income)"）。
  - **CRAFT**：构建 specialized toolset，通过 retrieval-augmented generation 动态组合基础工具（如"search + summarize + translate"）。

- **Tool Mastery**（工具掌握）：
  - **LearnAct**：通过分析 API 错误返回（如 HTTP 400 Bad Request），迭代优化 tool usage documentation（如修正参数格式）。
  - **DRAFT**（Dynamic Retrieval and Feedback Tool）：self-driven interaction with tools → build internal model of tool behavior（如"该 API 响应延迟高，需加 retry 机制"）。

- **Tool Selection & Retrieval**（工具选择与检索）：
  - **ToolGen**：将工具编码为 semantic tokens，统一进行 retrieval 与 calling via generation（即"生成工具名"而非"选择工具名"）。
  - **ATLASS**（Autonomous Tool Learning and Selection System）：closed-loop framework where LLM selects tools based on task decomposition and past success rate.

#### Architecture（架构）
架构指智能体内部模块的组织方式，其演化涉及单智能体流程优化与多智能体协作拓扑调整。

- **单智能体架构优化**：
  - **TextGrad**：将 workflow 中每个节点（如 "plan", "act", "reflect"）视为可优化模块，通过 textual feedback 反向调整节点 prompt（如将"plan"改为"generate three alternative plans and select the best"）。
  - **AlphaEvolve**：使用 evolutionary algorithm 搜索最优 node composition（如是否需要 separate verifier module）。

- **多智能体架构优化**：
  - **ReMA**（Role-Evolving Multi-Agent）：通过群体强化学习协调"thinker"与"executor"角色，动态调整通信拓扑（如 thinker 直接向 executor 发送 refined plan）。
  - **Multi-Agent Design**：搜索 multi-agent collaboration graph（谁与谁通信）以最大化 team performance（如在医疗诊断中，symptom-analyzer 与 drug-recommender 需高频交互）。
  - **Agentsquare**：automatic agent search in modular design space（如从 planner/retriever/executor 模块库中自动组合最优 agent）。

<mark style="background: #FFF3A3A6;">【理解：传统 AI 系统仅更新模型参数；而自进化智能体实现了 **四维协同演化**（model + context + tool + architecture），从而获得更强的适应性与泛化能力。】</mark>

<br>

### When to Evolve（演化时机）

演化可在两个时间尺度发生：

#### Intra-test-time Evolution（测试时演化）
- **定义**：在单次任务执行过程中实时调整，不涉及模型参数更新。
- **技术**：
  - **AdaPlanner**：根据环境反馈动态切换 planning mode（如从 BFS 切换到 heuristic search，当发现状态空间过大时）。
  - **Self-Refine**：generate → critique → refine 循环，最多迭代 N 次。例如，初始回答"巴黎是法国首都" → critique："未说明埃菲尔铁塔位置" → refine："巴黎是法国首都，埃菲尔铁塔位于其市中心"。
  - **LADDER**（Learning by Adapting During Deployment with Examples and Reflection）：当遇到难题时，自动生成变体问题集，通过 test-time RL 快速适应（如调整 temperature 或 top-p）。
- **特点**：低延迟、高灵活性，适用于 online 场景。

#### Inter-test-time Evolution（测试间演化）
- **定义**：跨任务间积累经验并更新系统，通常涉及离线训练。
- **技术**：
  - **Expel**（Experiential Prompting for Lifelong Learning）：将 past experiences 存入 memory bank，新任务中检索相似案例用于 in-context learning（ICL）。
  - **AgentGen**：生成 synthetic environments（PDDL/Gym 格式），构建 curriculum for offline training（如从简单导航到复杂多目标规划）。
  - **STaR**（Self-Taught Reasoner）：将失败轨迹中的 self-generated explanations（如“我忽略了边界条件”）转化为训练数据，用于 SFT。
- **特点**：性能提升显著，但需计算资源与时间。

<mark style="background: #FFF3A3A6;">【理解：Intra-test-time 演化提供 immediate adaptation，而 inter-test-time 演化实现 long-term capability growth。二者常结合使用，形成“在线微调 + 离线精炼”的混合范式。】</mark>

<br>

### How to Evolve（演化机制）

演化机制可分为三类：

#### Reward-Based Evolution（基于奖励的演化）
- **Outcome-Based Reward**（结果导向奖励）：
  - 仅在 episode 结束时获得稀疏奖励。
  - 代表工作：
    - **AutoWebGLM**：使用预训练 reward model 判断最终网页是否满足 query（如"找到 iPhone 16 价格" → 检查页面是否含价格信息）。
    - **DigiRL**：Vision-Language Model（VLM）作为 evaluator，输入 (instruction, final screenshot) → 输出 success/failure。
    - **WebRL**：提出 ORM（Outcome-Supervised Reward Model），在动态网页环境中判断任务完成度，支持 curriculum learning。

- **Process-Based Reward**（过程导向奖励）：
  - 对每一步推理打分，提供 dense supervision，解决 sparse reward 问题。
  - **Math-Shepherd**：用 MCTS 生成多条解题路径，统计每个 step 在成功路径中的出现频率 → 作为 correctness label → 训练 PRM（Process Reward Model）。
  - **rStar-Math**：迭代 co-evolve policy 与 PRM：policy 生成轨迹 → PRM 打分 → DPO（Direct Preference Optimization）更新 policy。
  - **Agent Q**：在 MCTS 中集成 step-wise verifier（LLM 判断当前 step 是否合理），仅保留 verified 路径用于训练，避免错误累积。

- **Hybrid Reward**（混合奖励）：
  - **GiGPO**（Generalized Incentive-Guided Policy Optimization）：同时使用 episode-level reward（任务是否完成）和 step-level reward（由轻量 critic 提供），解决 long-horizon credit assignment 问题。

#### Imitation and Demonstration（模仿与示范）
- **Self-Generated Demos**：
  - **SELF**（Self-Explaining Learning Framework）：self-feedback + self-correction → generate high-quality instruction-following pairs.
  - **STaR**：将错误答案 + self-explanation 转化为训练样本（如“错误：未考虑负数；正确：应先取绝对值”）。

- **Cross-Agent Learning**：
  - **SiriuS**：构建 experience library，agents 共享 successful trajectories.
  - **SOFT**（Self-Optimized Feedback Training）：通过 internal feedback 优化示范质量。

#### Population-Based Evolution（基于种群的演化）
- **Darwin Gödel Machine**：直接修改自身 Python 代码实现 self-improvement（如优化搜索算法）。
- **EvoMAC**（Evolutionary Multi-Agent Collaboration）：multi-agent co-evolution via textual backpropagation.
- **EvoAgent**：使用 evolutionary algorithms 自动生成 multi-agent systems.

<mark style="background: #FFF3A3A6;">【理解：Reward-based 方法依赖 explicit feedback，而 imitation 方法利用 implicit knowledge；population-based 则探索 diverse strategies。三者可融合，如用 population 生成 demos → 用 reward 过滤 → 用于 SFT。】</mark>

<br>

### Where to Evolve（应用场景）

| 场景 | 特点 | 代表工作 |
|------|------|--------|
| **General-Purpose** | 开放域任务（问答、工具使用） | AutoWebGLM, Expel, Alita |
| **Web Navigation** | 动态 HTML 环境，DOM 结构复杂 | WebArena, MiniWoB++, WebRL |
| **GUI Automation** | Android/iOS 屏幕操作，需视觉理解 | DigiRL, AITW |
| **Code Generation** | 编程任务，需执行验证 | Voyager, SCA |
| **Strategic Games** | 多人博弈、谈判，需心理建模 | Richelieu (AI Diplomacy) |
| **Mathematical Reasoning** | 形式化证明、解题，需符号推理 | Math-Shepherd, rStar-Math |

<br>

<mark style="background: #FFF3A3A6;">【理解：不同场景对演化维度有不同侧重。例如，Web Navigation 侧重 tool mastery 与 process reward，而 Strategic Games 侧重 architecture evolution（如增加 deception detection module）。】</mark>



<br>

## 代表性工作

### SCA（Self-Challenging Agent）
- **机制**：
  1. LLM 生成 Code-as-Task 问题（如“写一个排序算法”）；
  2. 执行代码，验证输出；
  3. 保留成功轨迹（problem + solution）；
  4. 用这些轨迹微调 LLM。
- **效果**：在 HumanEval 上提升 19% pass@1，证明 self-challenge 可有效提升 coding 能力。

### AgentGen
- **双向演化循环**：
  - **Environment → Agent**：生成 PDDL 环境 → agent 学习规划；
  - **Agent → Environment**：根据 agent 能力调整环境难度（如增加障碍物）。
- **结果**：agent 在 unseen 环境中泛化能力显著提升，体现 co-evolution 优势。

### Reflexion
- **Self-Reflective Mechanism**：
  - 每次任务后，LLM 生成 natural-language critique（如“我忽略了边界条件”）；
  - critique 存入 memory，下次类似任务时作为 few-shot example。
- **应用**：在 ALFWorld（文本环境导航）中提升成功率 35%，证明 textual reflection 有效。

### Alita（Generalist Agent）
- **设计**：
  - Minimal predefinition：无预设 workflow；
  - Maximal self-evolution：通过 trial-and-error 自主发现 optimal reasoning path.
- **机制**：结合 SFT + RL + memory augmentation，支持 scalable agentic reasoning.

### SkillWeaver
- **Web Agent Self-Improvement**：
  - **Skill Discovery**：通过 exploration 发现可复用 action sequence（如“login → search → filter”）；
  - **Skill Honing**：通过 repeated execution 优化 skill parameters（如 wait time, retry logic）。
- **结果**：在 WebShop 任务中减少 40% 操作步数，提升 efficiency。

<br>

## 评估、安全与未来

### 评估基准
- **Tool Usage**：ToolBench, ToolLLM benchmark（16000+ APIs）
- **Web Navigation**：WebArena, MiniWoB++
- **GUI Automation**：AITW, DigiRL benchmark
- **Collaboration**：Overcooked, AI Diplomacy
- **Specialized Domains**：GSM8K（math）, HumanEval（code）

 **挑战**：long-horizon 任务缺乏 standardized metric（如“部分成功”如何评分）。

### 安全挑战
- **Reward Hacking**：agent 利用 reward model 漏洞（如生成虚假 success 页面）。
- **Unsafe Tool Use**：调用危险 API（如删除文件）。
- **Deception**：在 multi-agent 环境中欺骗其他 agent（如 Richelieu）。
- **对策**：
  - **Evolutionary Safeguards**：“三大定律”：
    1. Maintain safety；
    2. Preserve or improve performance；
    3. Enable autonomous optimization.

### 未来问题
1. **Stable Reward Modeling**：如何构建 robust、generalizable reward function？
2. **Efficiency-Effectiveness Trade-off**：MCTS + LLM verification 成本过高。
3. **Transferability**：在 one domain 演化的 prompt/tool/arch 能否迁移到 another？
4. **Theoretical Foundation**：缺乏对 self-evolution convergence 的 formal guarantee。

未来工作需聚焦 **安全对齐**、**高效演化** 与 **标准化评估**，以迈向人工超级智能（Artificial Super Intelligence, ASI）。



<br>

## 附录：关键参考文献索引

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251220012322319.png"/>
<br>

| 工作 | 核心贡献 | 来源 |
|------|--------|------|
| Promptbreeder | Evolutionary prompt optimization | arXiv:2309.16797 |
| SCA | Self-challenging for code tasks | NeurIPS 2023 |
| TextGrad | Textual gradient-based optimization | ICLR 2024 |
| MAS-Zero | Multi-agent self-evolution | arXiv:2502.0xxxx |
| AgentGen | Synthetic environment generation | ICML 2024 |
| Reflexion | Self-reflection via natural language | NeurIPS 2023 |
| AdaPlanner | Closed-loop adaptive planning | AAAI 2024 |
| Self-Refine | Iterative self-critique & refine | NeurIPS 2023 |
| RAGen | RL for agent generation | arXiv:2504.20073 |
| Mem0 | Scalable long-term memory | arXiv:2501.xxxx |
| Expel | Experiential learning | AAAI 2024 |
| Agent Workflow Memory | Multi-turn workflow storage | arXiv:2503.xxxx |
| Richelieu | Structural self-modification in diplomacy | NeurIPS 2024 |
| PromptAgent | Strategic prompt planning | ACL 2024 |
| SPO | Self-supervised prompt optimization | arXiv:2502.xxxx |
| EvoAgent | Evolutionary multi-agent generation | GECCO 2024 |
| Voyager | Open-ended embodied agent | NeurIPS 2023 |
| Alita | Generalist self-evolving agent | arXiv:2505.xxxx |
| ATLASS | Closed-loop tool selection | arXiv:2504.xxxx |
| CREATOR | Tool creation for reasoning | ICLR 2025 |
| SkillWeaver | Web agent skill discovery | WWW 2025 |
| CRAFT | Specialized toolset customization | arXiv:2503.xxxx |
| LearnAct | Action learning for tool mastery | EMNLP 2024 |
| DRAFT | Self-driven tool interaction | arXiv:2502.xxxx |
| ToolLLM | Mastering 16000+ APIs | NeurIPS 2024 |
| ToolGen | Unified tool retrieval & calling | ACL 2025 |
| Agentsquare | Automatic agent search | arXiv:2501.xxxx |
| Darwin Gödel Machine | Open-ended self-improvement | arXiv:2505.22954 |
| AlphaEvolve | Evolutionary workflow search | GECCO 2025 |
| ReMA | Multi-agent role coordination | AAMAS 2025 |
| SiriuS | Experience sharing | arXiv:2312.17025 |
| WebRL | ORM for web tasks | arXiv:2406.12373 |
| DigiRL | VLM-based sparse reward for GUI | arXiv:2405.xxxx |
| SOFT | Self-optimized feedback | arXiv:2502.xxxx |
| EvoMAC | Multi-agent co-evolution | arXiv:2406.xxxx |
| Math-Shepherd | MCTS-based process annotation | ICLR 2025 |
| Agent Q | Step-wise verification + DPO | NeurIPS 2024 |
| GiGPO | Dual-level reward for stability | ICML 2025 |
| AutoWebGLM | Outcome-based self-evolution | arXiv:2406.12373 |
| rStar-Math | Iterative PRM evolution | arXiv:2501.xxxx |

<br>