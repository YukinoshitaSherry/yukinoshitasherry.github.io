---
title: Harness Engineering 学习笔记
date: 2026-04-13
categories:
  - 学AI/DS
tags:
  - Agent
desc: 面向 LLM Agent 的 Harness Engineering 概念框架、证据结构与工程实践综述。
---

参考：
- <a href="https://arxiv.org/pdf/2603.28052">Meta-Harness: End-to-End Optimization of Model Harnesses 论文</a>
- <a href="https://tingdeliu.github.io/Harness-Engineering-Survey/">Harness Engineering 综述</a>
- <a href="https://www.preprints.org/manuscript/202604.0428">Agent Harness for Large Language Model Agents: A Survey</a>
- <a href="https://javaguide.cn/ai/agent/harness-engineering.html">一文搞懂 Harness Engineering：六层架构、上下文管理与一线团队实战</a>
- <a href="https://zhuanlan.zhihu.com/p/2014014859164026634">知乎 Harness Engineering 深度解析：AI Agent 时代的工程范式革命</a>
- <a href="https://www.xiaohongshu.com/discovery/item/69d61348000000001b02312c?source=webshare&xhsshare=pc_web&xsec_token=CBfL41Koe77TlZEK2WQMyd2LXAL8RIXKNR5srBWvfFIn4=&xsec_source=pc_share">小红书 Harness 不难啊，加这 4 个文件足够了</a>

<br>

## 背景

### 核心命题

**Agent = Model + Harness**

Phil Schmid 打了个比方：模型是 CPU，Harness 是操作系统。

Harness 指模型之外的执行基础设施，包括系统提示、工具调用、文件系统、沙箱环境、编排逻辑、中间件、反馈回路与约束机制。模型提供能力上限，Harness 决定能力能否转化为稳定、可复现、可维护的系统行为。

Harness Engineering 的核心命题是：**限制才能解放。** 在长时程、工具密集、可执行的 Agent 任务中，系统可靠性主要受执行壳层质量约束，而非仅由模型能力单独决定。  
这与工程实践中反复出现的现象一致：当模型权重保持不变、仅调整 Harness（工具暴露、上下文装配、验证回路、权限约束）时，任务成功率与稳定性可出现显著变化。由此可将其归纳为工程结论：模型决定能力上限，Harness 决定交付下限。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/20260413201023290.png"/><br>

### 原因

【为什么瓶颈常出现在 Harness，而非模型本体？】

该判断可由两类互补证据支持。第一类是“接口/工具格式变化导致的能力跃迁”：<a href="Can.ac">Can.ac</a> 的 Harness 实验显示，在模型权重不变的条件下，仅替换代码编辑与工具调用格式，部分模型的任务通过率可从 6.7% 提升至 68.3%。第二类是“执行治理变化导致的系统级改进”：LangChain 在 Terminal Bench 2.0 上仅通过调整 system prompt、tools 与 middleware（未更换模型）将得分从 52.8% 提升至 66.5%。

上述现象说明，Agent 性能并非由“模型智力”单变量决定，而是由“模型能力 × Harness 适配质量”共同决定。对工程系统而言，Harness 实际承担三项决定性功能：  
(1) 将任务表达为模型可执行的操作接口；  
(2) 将中间状态组织为可持续推理的上下文结构；  
(3) 将输出结果置于可验证、可回滚的控制回路。  
当任一功能缺失时，模型能力无法稳定转化为可交付结果，表现为高波动、低复现与高返工。

<br>

与“瓶颈在 Harness”密切相关的经验现象是“上下文退化”：上下文窗口并非越满越好，随着无关轨迹累积，信息密度下降、推理焦点被稀释，常见外显结果是循环编辑、验证不足与过早结束。实践上应将“上下文容量”问题转化为“上下文质量”问题，通过分层文档、渐进披露、压缩与结构化交接维持稳定性。

这一现象可以进一步细化为几类可操作的失败模式（可视为“上下文退化 + 执行治理不足”的组合后果）：
1. 首个答案偏见（first-answer bias）：Agent 容易将首个可行方案视为“已足够正确”，在缺乏外部验证压力时倾向于停止进一步反证与替代方案搜索，导致隐藏缺陷在后续阶段集中暴露。
2. 一步到位倾向（one-shotting）：Agent 试图在单会话内完成完整交付，导致上下文预算在中途耗尽，留下缺乏交接信息的半成品状态。后续会话需消耗大量 token 进行状态重建，显著降低有效开发时间。
3. 过早宣布完成：当局部功能已形成可见进展时，Agent 倾向于将“阶段性可用”误判为“全局完成”，提前结束任务循环，导致剩余需求被遗漏。
4. 过早标记功能就绪：在缺少明确验证门禁时，Agent 可能以代码编译通过或局部命令通过替代端到端可用性验证，形成“测试通过但场景不可用”的假阳性。
5. 会话冷启动成本过高：新会话需要重复理解环境启动方式、服务依赖与运行命令，token 预算被环境摸索消耗，而非用于真实功能推进。

上述失败模式的共同指向是：问题并非单纯来自模型推理能力，而是来自缺少持续状态管理与强制验证机制。对应治理策略包括：结构化交接文档、可执行启动脚本、阶段化验收清单、端到端门禁测试、以及在生命周期钩子中注入“未完成项复核”步骤。


> [!INFO]+ 适用范围
>
> 本文讨论的主要对象是“长链路任务 + 外部工具调用 + 可执行反馈回路”的 Agent 系统。  
> 对于一次性问答、短上下文生成、无外部执行环境的任务，Harness 的边际影响通常弱于模型与提示策略。

<br>

## 概念

### Agent Harness 的定义

Agent Harness 可定义为：  
围绕语言模型构建的执行基础设施集合，用于将模型推理转化为可追踪、可验证、可恢复的系统行为。

其最小功能集通常包括：
- 执行环境（Execution Environment）
- 工具集成（Tool Integration）
- 上下文管理（Context Management）
- 任务范围协商（Scope Negotiation）
- 循环生命周期管理（Loop Management）
- 验证与评估接口（Verification Interface）

### 辨析

与 Prompt Engineering、Context Engineering 的关系

三者为包含关系而非并列关系：

`Prompt Engineering ⊂ Context Engineering ⊂ Harness Engineering`

三者的差异不仅在聚焦的关注点，还体现在作用对象、时间尺度与评估标准上。


对应三个层面：


#### Prompt

**Prompt：表达控制（如何提问、如何约束输出格式）**

- Prompt Engineering 主要作用于单轮或短链路交互，目标是提升单次输出质量，典型评估是回答相关性、格式遵循率与任务完成度。
- 技术可包括：系统提示词（System Prompt）、Few-shot 示例构造、思维链引导（CoT）与输出格式约束。

#### Context

**Context：信息控制（在何时向模型提供何种信息）**

- Context Engineering 作用于多轮任务中的信息供给链路，目标是在正确时机提供正确信息，核心问题是检索质量、上下文装配、压缩与记忆读写。
- 技术可包括：上下文管理策略、RAG 检索增强、记忆注入（短期/长期）、Token 预算优化与上下文裁剪。

#### Harness

**Harness：执行治理（如何保证系统稳定、可审计、可演化）**

- Harness Engineering 作用于端到端执行系统，关注 Agent 在长周期运行中的稳定性、可验证性与可恢复性。次级技术可包括：文件系统组织、沙箱执行、约束执行（lint/CI/权限门禁）、熵管理、反馈回路、生命周期编排与可观测性。换言之，Harness 是将 Prompt 与 Context 纳入统一执行治理框架的上位工程范式。

进一步从任务类型看，三层技术的主导权会发生转移：简单任务通常以 Prompt 质量为主导，外部知识任务以 Context 质量为主导，长链路生产任务则以 Harness 质量为主导。






<br>

## 框架

### 六层工程框架

- <a href="https://javaguide.cn/ai/agent/harness-engineering.html"> 一文搞懂 Harness Engineering：六层架构、上下文管理与一线团队实战</a> 中将 Harness 划分为六层：
- 信息边界层：定义任务目标、角色边界与禁止域
- 工具系统层：定义工具白名单、调用协议、返回格式
- 执行编排层：管理任务分解、阶段切换与循环策略
- 记忆与状态层：维护短期状态、长期记忆、检查点
- 评估与观测层：采集日志、指标、Trace 与验证结果
- 约束与恢复层：执行策略门禁、回滚、重试与降级

### 六组件研究框架

<a href="https://www.preprints.org/manuscript/202604.0428">Agent Harness for Large Language Model Agents: A Survey</a> 给出的抽象可写为 `H = (E, T, C, S, L, V)`：
- E：Execution Loop
- T：Tool Registry
- C：Context Manager
- S：State Store
- L：Lifecycle Hooks
- V：Verification Interface

两套框架并不冲突，前者面向工程实施，后者面向跨系统比较与研究归纳。

### 五个子系统实践框架

<a href="https://www.xiaohongshu.com/discovery/item/69d61348000000001b02312c?source=webshare&xhsshare=pc_web&xsec_token=CBfL41Koe77TlZEK2WQMyd2LXAL8RIXKNR5srBWvfFIn4=&xsec_source=pc_share">小红书 Harness 不难啊，加这 4 个文件足够了</a> 整合基模厂的实践总结的。

- Instructions（指令系统）：定义任务目标、流程规则、禁止项与质量标准，强调渐进式披露而非单一超长提示。
- State（状态系统）：显式记录 feature 状态与会话进度，解决跨会话失忆与状态漂移问题。
- Verification（验证系统）：将“是否完成”的判断权从 Agent 自评转为自动化检查与门禁判定。
- Scope（作用域系统）：限制单轮任务边界，避免一次会话并行改动过多目标导致漂移。
- Session Lifecycle（会话周期系统）：标准化“初始化 -> 工作 -> 验证 -> 提交 -> 交接”流程，降低冷启动与交接成本。

可将该五子系统映射到六组件框架：
- Instructions -> C / L（上下文装配与生命周期约束）
- State -> S（状态存储与进度持久化）
- Verification -> V（验证接口与门禁回路）
- Scope -> E / L（执行循环边界与钩子控制）
- Session Lifecycle -> E / L / S（流程编排 + 阶段钩子 + 会话交接）

### 四文件最小落地框架

<a href="https://www.xiaohongshu.com/discovery/item/69d61348000000001b02312c?source=webshare&xhsshare=pc_web&xsec_token=CBfL41Koe77TlZEK2WQMyd2LXAL8RIXKNR5srBWvfFIn4=&xsec_source=pc_share">小红书 Harness 不难啊，加这 4 个文件足够了</a>提出的方法。

“四文件”不是独立理论框架，而是上述五子系统的最小实现载体，属于工程模板层：

- `AGENTS.md`：承载 Instructions 与 Scope 的核心规则（目标、流程、约束、禁止项）。
- `init.sh`：承载 Session Lifecycle 的初始化步骤（环境检查、依赖准备、基础验证）。
- `feature_list.json`：承载 State 的结构化任务状态（feature id、状态、优先级、验收条件）。
- `progress.md`：承载 Session Lifecycle + State 的会话交接信息（已完成项、阻塞项、下一步）。

这种“文件级最小 Harness”与六层/六组件同样不冲突：它是将抽象能力压缩为可维护、可迁移、可复用的轻量起步方案。

>[!INFO]+ 框架理解
>可以把本章中的三套表述理解为同一体系的不同分辨率：
> - 综述的六组件（E/T/C/S/L/V）回答“系统必须具备哪些能力”；
> - 六层工程框架回答“这些能力在工程实现上如何分层组织”；
> - 五子系统 + 四文件回答“团队如何最低成本把这些能力落地并持续运行”。
> 因此，三者关系不是替代关系，而是“研究抽象 -> 架构分层 -> 运行模板”的递进关系。实践时可先用六组件校验完整性，再用六层确定责任边界，最后以类似五子系统和四文件的形式启动，完成最小可运行实现。

<br>

## 组件框架与系统机制整合

本节将四类来源中的核心信息统一到同一框架下：  
`Harness Engineering 综述` 侧重“三大支柱 + 技术体系 + 实践案例”；  
`Agent Harness for Large Language Model Agents: A Survey` 侧重“形式化定义 + 六组件抽象”；  
`JavaGuide` 与知乎长文侧重“六层落地结构 + 团队实践经验”。  
三者可以互补，不存在原则性冲突。

## 组件细节

### 上下文组件

上下文组件（C）：信息相关性优先于上下文长度

上下文治理目标并非“输入越多越好”，而是保持高相关、低污染。  
在长任务中，冗余轨迹会降低决策质量，表现为重复尝试、提前收敛、无效工具调用增加。
Dex Horthy 观察到一个现象：168K token 的上下文窗口，用到大约 40% 的时候，Agent 的输出质量就开始明显下降。

| 区间 | 占比 | 表现 |
| --- | --- | --- |
| Smart Zone | 0 - ~40% | 推理聚焦、工具调用准确、代码质量高 |
| Dumb Zone | 超过 ~40% | 幻觉增多、兜圈子、格式混乱、低质量代码 |

上下文管理的目标不是“持续扩容输入”，而是“维持高相关信息密度”。在长任务中，当无关历史轨迹持续累积时，推理焦点会被稀释，常见表现包括重复尝试、错误收敛与过早结束。  
因此，成熟 Harness 通常采用分层文档、渐进式披露、上下文压缩与结构化交接（context reset）等机制，控制上下文熵增，并将验证信号持续注入执行回路。

核心策略：
- 渐进式披露：入口文档（如 `AGENTS.md`）作为索引，不承载全部规则
- 分层知识组织：规范、架构、任务状态分文件维护
- 上下文压缩：保留当前决策所需状态，移除历史噪声
- Context Reset：在上下文接近饱和时进行结构化交接并重启会话

> [!INFO]+ 上下文预算观测指标
>
> 建议至少跟踪三项指标：  
> 1) 有效信息密度（有效事实 token / 总 token）  
> 2) 历史污染比（过时轨迹 token / 总 token）  
> 3) 验证信号占比（测试、错误、检查点 token / 总 token）

### 验证与执行组件

验证组件（V）与执行组件（E）：Build-Verify 闭环

在缺乏验证压力时，Agent 往往出现“生成即结束”的行为倾向。  
因此，执行循环必须内建验证阶段，形成完整闭环：
- 生成候选方案
- 执行测试、静态检查、契约检查
- 根据失败信号修复
- 满足门禁后再结束任务

关键设计要点：
- 生成与验证职责分离，减少自评偏差
- 验证标准外显化、机器可执行、可复现实验
- 失败样本沉淀为后续优化数据集
- 验证维度同时覆盖功能正确性、架构一致性、运行安全性



### 工具与生命周期组件

工具组件（T）与生命周期组件（L）：治理面与控制面

“工具越多越好”并不成立；过宽工具暴露通常会放大探索噪声与误用风险。  
工具治理与生命周期钩子应联合设计：
- 任务级工具白名单：按任务类型动态裁剪
- 返回格式规范化：降低上下文污染与解析歧义
- 高风险操作检查点：关键步骤确认与审计
- 循环检测与提醒注入：避免重复无效编辑
- 时间预算提醒：在关键阶段提示验证与收敛策略

### 状态组件与熵管理

状态组件（S）与熵管理：长期可维护性条件

高频 AI 生成会导致代码库出现文档漂移、规则退化、重复实现与隐式耦合。  
状态管理与熵管理结合后，才能形成可持续系统。

典型机制：
- 定期一致性巡检（文档、架构、依赖）
- 冗余与失效实现清理
- 规则违规自动检测
- 失败模式知识化沉淀
- 基于检查点的跨会话交接

### 外循环优化

外循环优化：Meta-Harness 对组件框架的补充

`Meta-Harness` 的价值在于将“Harness 本身”设为优化对象，而非仅优化提示词文本。  
其优化空间覆盖工具策略、记忆读写、验证路径、恢复逻辑等系统级变量。

方法学特征：
- 在代码与流程空间搜索候选 Harness
- 读取历史执行日志与失败轨迹进行迭代
- 在准确率、成本、上下文开销之间做多目标权衡


### 可观测性设计

可观测性是验证回路能够持续收敛的前提。缺少可观测性时，团队只能看到“是否完成”，难以解释“为何失败”“失败发生在何阶段”，也就无法稳定优化 Harness。

建议将可观测性指标分为执行结果、过程行为、资源成本、质量覆盖四个层面：

| 观测维度 | 具体指标 | 主要用途 |
| --- | --- | --- |
| 任务成功率 | 通过/失败/超时 | 衡量整体 Harness 运行效果与稳定性 |
| 推理轨迹 | 每步决策过程与阶段切换 | 诊断失败成因与错误传播路径 |
| 工具调用分布 | 各工具调用频次、失败率、重试率 | 识别工具设计缺陷与暴露面问题 |
| Token 消耗 | 各阶段 token 用量与峰值 | 进行成本分析与预算优化 |
| 循环次数 | 重试次数、重复编辑次数 | 识别高难任务与无效探索模式 |
| 验证覆盖率 | 实际执行的测试与检查项比例 | 评估验证强度与质量保障水平 |



<br>


## 关键挑战

Harness Engineering 的现实难点不在“是否需要 Harness”，而在“何处最容易失效、如何持续纠偏”。结合综述与工程案例，可归纳为五类高频挑战。

### 首答锚定偏差（First-Answer Anchoring）

Agent 在生成首个可行方案后，常出现过早收敛，不再主动扩展验证路径。其直接后果是“表面完成、深层缺陷未暴露”，尤其在边界条件与回归兼容性上问题突出。

工程上通常通过外部验证压力进行纠偏：将测试执行、失败回路与门禁判定前置到任务流程中，并通过中间件在“准备结束”阶段强制注入验证步骤。

### 代码库熵增与结构退化（Repository Entropy）

在高频 AI 生成场景中，新增代码速度通常快于治理速度，导致文档滞后、重复实现、依赖层次破坏与模块边界模糊。短期内表现为开发效率提升，长期则转化为维护成本上升与故障率增加。

有效治理依赖持续性机制，而非一次性清理：包括约束扫描、文档一致性巡检、冗余实现回收、失败模式沉淀与定期质量审计。

### 上下文缺口（Context Gap）

项目中的关键知识往往分散在会议结论、IM 消息与个人经验中，未进入仓库可执行语义层。Agent 即使具备编码能力，也无法访问这些隐性约束，导致“局部正确、全局偏离”。

解决路径是将隐性知识转化为机器可读资产：以入口文档索引规则、分层维护架构与约束文档，并将验收标准与反模式写入可执行检查流程。

### 模型升级脆弱性（Version-Coupling Fragility）

许多 Harness 设计隐含依赖于特定模型行为特征（输出风格、工具调用习惯、指令服从模式）。模型版本升级后，这些隐式假设可能失效，进而导致原有策略效果衰减甚至反向影响。

应将“模型升级”视为 Harness 重新评估触发器：同步执行回归基准、失败类型对比与约束强度重整，而非仅替换模型版本号。

### 静态过拟合与过度工程化（Static Over-Engineering）

当 Harness 长期固化为复杂且不可拆分的控制逻辑时，会出现“历史正确、当下低效”的结构性问题：模型能力提升后，旧约束继续存在，导致额外 token 成本、执行路径冗余与创新空间受限。

实践上应采用可拆卸（rippable）设计原则：每项约束都应具备明确的保留条件与移除条件，并通过数据驱动决定“加”或“减”。

<br>


## 争议观点

<a href="https://www.xiaohongshu.com/discovery/item/69dc4b55000000001d0187a4?source=webshare&xhsshare=pc_web&xsec_token=CBcJK23FEqaaPn5QqURLkDjNrHU1_NVXjbEB6kUiEjOSY=&xsec_source=pc_share">【Harness刚火起来 就要被模型进化 吞掉了吗 - 随机场 | 小红书 - 你的生活兴趣社区】</a>

### 内生推理机制进展是否会替代 Harness

参考文献：
- [Reasoning Shift: How Context Silently Shortens LLM Reasoning](https://arxiv.org/abs/2604.01161)  
- [Emotion Concepts and their Function in a Large Language Model](https://arxiv.org/abs/2604.07729)  
- [Anthropic 研究页面](https://www.anthropic.com/research/emotion-concepts-function)


- Yandex 论文 `Reasoning Shift: How Context Silently Shortens LLM Reasoning`（arXiv:2604.01161）指出：在长上下文与多任务条件下，模型可能出现推理轨迹压缩，伴随自检与不确定性处理行为下降。该结论支持“长链路质量问题不只与长度上限有关，还与模型内部策略变化有关”。  
- Anthropic 研究 `Emotion Concepts and their Function in a Large Language Model`（arXiv:2604.07729）指出：模型内部存在可解释的情绪概念表征，并对行为产生可测因果影响。该结论支持“部分行为偏差与内部表示有关，且内部干预具有研究价值”。

上述两篇研究共同说明：Agent 失败并非完全是外部编排问题，模型内生决策机制同样关键。

成立的部分：
- 将长上下文退化简单归因为“token 变长”过于粗糙，内部策略（例如推理压缩）是更贴近机制层的解释。  
- 仅依赖外部约束难以彻底修复模型内部决策偏差，尤其在复杂推理与长期任务中。

推断过度的部分：
- 从“存在内部可干预信号”直接推到“Harness 将彻底失去价值”缺乏充分证据。当前研究更多证明“可解释、可局部干预”，尚未证明“可在真实工程场景中全面替代外部治理”。  
- Harness 处理的不仅是推理质量，还包括权限控制、工具编排、审计追踪、回滚恢复、跨会话状态管理等系统职责；这些职责具有明确的基础设施属性，不会因为模型更聪明而自然消失。

笔者个人认为更合理的判断不是“内生机制 vs 外部 Harness 二选一”，而是“双层协同”：
- 模型层：持续改进内部推理策略与状态校准，降低内生偏差；
- 系统层：通过 Harness 提供可验证、可审计、可恢复的执行边界。

可预期的演化方向是：随着模型能力提升，Harness 会从“强约束防失控”逐步转向“轻约束高观测”，但不会消失。其角色将从“纠错主力”转为“系统安全与治理底座”。


<br>


## 总结

- Harness 的核心价值不是替代模型，而是将模型能力转化为可验证、可治理、可演化的系统能力。  
- 组件完整性（E/T/C/S/L/V）与分层实现质量共同决定最终稳定性。  
- 上下文治理、验证闭环、工具约束、熵管理是最具可迁移性的四个主轴。  
- 对于真实工程场景，Harness 优化应被视为持续过程，而非一次性配置。

<br>
