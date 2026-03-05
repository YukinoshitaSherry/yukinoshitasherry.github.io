---
title: Agent Skills入门
date: 2026-01-30 
categories:
  - 学AI/DS
tags:
  - Agent
desc: Claude带火了skills,学习一下Agent未来工程新范式。感觉现在有点像tools刚出来时候，后续应该会发展的。
---

## 一、概念区分：Agent、Tool、Skill

### 1.1 Agent Skill 定义

**Agent Skill = 大模型 Agent 的「可复用小能力模块」**，作用是将对话型模型升级为可执行任务的智能体。

### 1.2 三者关系

| 概念 | 定义 | 业务作用 |
| :--- | :--- | :--- |
| **Agent（智能体）** | 有目标、有记忆，会自主规划并调用工具的大模型应用；比「单轮大模型 API」多了决策与执行 | 独立编排任务、根据执行结果调整，适用于客户服务、报告生成、代码协助等多步复杂场景 |
| **Tool / Function（工具）** | 具体能力的底层接口：查库、调 HTTP API、执行脚本等；更接近「原子操作」 | Agent 与外部世界交互的执行单元，处理超出 LLM 知识范围或需实时数据的需求 |
| **Skill（技能）** | 站在 Agent 层的抽象；将若干工具 + 规则 + Prompt 封装成「对 Agent 而言一键可用的能力单元」 | 可即插即用的能力包，对上层屏蔽复杂细节，提升可复用性与模块化 |

### 1.3 层级类比

> [!ABSTRACT]+ Tool / Skill / Agent 层级类比
>
> - **Tool**：锤子、螺丝刀（最低层工具）
> - **Skill**：会「安装书架」这门手艺（组合使用多种工具完成目标）
> - **Agent**：根据需求决定「要不要装书架」的装修工（理解需求、决策、编排 Skill 与 Tool）

<br>

## 二、Agent Skill 的原理：从调用函数到组合能力

### 2.1 技能发现机制

**问题**：大模型如何知道有哪些 Skill？

**主流做法**：将 Skill 作为「可调用选项」暴露给模型，并明确描述其能力边界。描述信息通常通过 **System Prompt / 工具列表** 提供给 Agent。

### 2.2 技能描述关键要素

| 要素 | 说明 |
| :--- | :--- |
| **技能名** | 唯一标识符，供模型调用 |
| **功能描述** | 业务用途与预期结果 |
| **入参 schema** | 所需字段及类型 |
| **返回 schema** | 返回数据结构 |
| **使用场景** | 何时应调用该技能（触发条件） |

> [!EXAMPLE]+ query_user_profile 技能描述示例
>
> - **技能名**：`query_user_profile`
> - **功能描述**：根据 user_id 查询用户画像
> - **入参**：user_id (string)
> - **返回**：用户画像结构
> - **场景**：需获取特定用户信息以做个性化服务或分析时

### 2.3 模型推理流程

1. **判断需求**：当前任务是否需要调用某 Skill
2. **参数构造**：按 Skill 的入参 schema 构造调用参数
3. **Skill 执行**：调用并执行
4. **结果推理**：基于返回结果继续推理

该流程对应 **Function Calling / Tool Calling** 的底层逻辑（OpenAI、Anthropic 等平台通用）。

### 2.4 Skill 的构成：策略而非函数

Skill 不应理解为「一个函数」，而应视为「一段策略」。现代实践中，一个 Skill 通常包含：

- **一个或多个底层工具**：封装、协调 API 调用、脚本执行等
- **专门给该 Skill 用的 Prompt 模板**：引导模型正确调用、格式化输入输出
- **必要的前后处理**：入参校验、结果解析、异常兜底等

<br>

## 三、为什么要引入 Skill

### 3.1 避免「冗长 Prompt」

若不使用 Skill，复杂业务逻辑会堆积在 Prompt 中：「当用户这样说就先查 A，再判断 B……」→ Prompt 越写越长、越难维护、难以复用。

### 3.2 职责划分

| 角色 | 职责 |
| :--- | :--- |
| **Prompt** | 定义高层意图与策略 |
| **Skill** | 负责具体执行步骤 |

将 if-else 逻辑从自然语言中抽出，变为**可测试、可复用的代码模块**。

### 3.3 Skill 对 Agent 的抽象价值

Agent 仅需识别存在可调用的分析日报 Skill，无需理解底层实现（如多次库查询与 LLM 总结调用）。Skill 封装复杂逻辑，降低 Agent 的认知负载。

### 3.4 Skill 内部能力

- **入参校验**：确保输入有效、合规
- **默认值处理**：未指定参数提供预设值
- **输出结构化封装**：统一、规范的返回结构
- **异常兜底**：错误处理、重试、降级
- **可选小状态**：分页游标、上次查询结果缓存等

### 3.5 多 Agent 协作与任务编排

Skill 支持**多 Agent 协作与复杂任务编排**。在多 Agent 架构（Planner / Worker / Critic）中，Skill 是任务分配与协调的基础单元。

<br>

## 四、多 Agent 架构与 Skill 治理

### 4.1 角色职责

| 角色 | 职责 |
| :--- | :--- |
| **Planner** | 决定需要哪些 Skill，以及执行顺序 |
| **Worker** | 执行对应 Skill 并返回结果 |
| **Critic** | 利用自身 Skill 做质量检测、重写、过滤 |

### 4.2 技能抽象稳定后的优势

- **新任务组合高效**：通过「组合已有 Skill + 少量新 Skill」即可完成
- **流程复用**：避免重复构建流程

### 4.3 企业落地考量

- **权限控制**：按 Skill、角色做细粒度访问控制
- **可观测性**：分离的 Skill 与角色便于监控、诊断、审计

### 4.4 Skill 维度治理

在 Skill 维度配置**权限、Quota、监控、熔断、审计**，避免依赖分散的 user prompt 进行问题定位。管理问题包括：

- 哪些 Skill 允许哪个业务线 / 哪类用户使用？
- 每个 Skill 被调用次数、平均耗时、失败率？
- 某次错误是模型决策错误，还是 Skill 本身故障？

<br>

## 五、从 0 到 1：设计 Agent Skill

### 5.1 第一步：拆解稳定子能力

从用户任务中拆出「稳定的子能力」——独立、可复用、接口清晰。

### 5.2 Skill 识别三问

满足以下三点，则值得抽象为 Skill：

1. **是否会被多次复用？**
2. **能否明确写出「输入是什么 → 输出是什么」？**
3. **能否在业务上单独监控其效果？**

### 5.3 业务示例

> [!EXAMPLE]+ 可抽象为 Skill 的能力
>
> - **根据用户行为生成个性化推荐理由**：推荐系统、转化率
> - **根据用户问题选择合适部门工单并创建**：客服自动化、工单路由
> - **把原始日志转换成结构化字段**：运维、监控、合规审计

### 5.4 第二步：封装接口与内部逻辑

完成识别后，定义对外接口并实现内部业务逻辑。

<br>

## 六、Skill 的接口与实现

### 6.1 接口定义（面向 Agent）

通常用 JSON Schema 描述 Agent 的调用规范。

> [!EXAMPLE]+ create_support_ticket 接口定义
>
> ```json
> {
>   "name": "create_support_ticket",
>   "description": "根据用户问题创建客服工单，并返回工单ID",
>   "parameters": {
>     "type": "object",
>     "properties": {
>       "user_id": { "type": "string" },
>       "issue_summary": { "type": "string" },
>       "priority": { "type": "string", "enum": ["low", "medium", "high"] }
>     },
>     "required": ["user_id", "issue_summary"]
>   }
> }
> ```

### 6.2 内部实现（面向工程）

- 入参校验 → 风险过滤
- 调业务 API 创建工单
- 接口错误重试 / 兜底处理
- 返回规范化结果（如 `{ticket_id, status}`）

### 6.3 Skill 级 Prompt（需模型参与时）

Skill 内部可包含模型调用。例如：先将用户自由文本压缩为 `issue_summary` 和 `priority`，该过程视为 Skill 内部一步，端到端完成从原始输入到结构化输出。

<br>

## 七、在 Agent 侧注册与使用 Skill

### 7.1 注册

在 Agent 配置中将 Skill 暴露为可用工具。

### 7.2 System Prompt 指导内容

- **触发条件**：何时优先使用该 Skill（语义条件、关键词、意图模式）
- **结果解释策略**：调用 Skill 后如何向用户说明结果

### 7.3 运行时链路

1. 用户提问 → Agent 分析意图
2. 决定调用 `create_support_ticket` Skill
3. 构造参数 → 执行 Skill → 获得 `ticket_id`
4. 结合上下文生成最终对话回复

业务层仅感知自然语言到工单的转换，Skill 调用细节对上层透明。

<br>

## 八、Skill 发展趋势

### 8.1 图编排 (Skill Graph / Workflow Graph)

Skill 由线性工具集合演进为**有条件分支、循环、回退的有向图**：

- **条件分支**：按条件选择不同 Skill 路径
- **循环**：重复执行某 Skill 直到满足退出条件
- **回退**：异常时撤销已执行操作

### 8.2 Skill 市场 / 生态

标准化 Skill 作为**可分发的插件**，包含四要素：

| 要素 | 说明 |
| :--- | :--- |
| **接口描述** | 输入、输出、参数、调用规范 |
| **运行镜像/代码** | 业务逻辑与运行环境 |
| **安全策略** | 权限、数据访问、风险控制 |
| **计费方式** | 按次、订阅等 |

与早期云函数 / API 市场类似，未来可能形成 Agent Skill 市场；企业可专注于「装配与编排」，而非从零开发所有 Skill。

<br>

## 九、Claude / Anthropic 实现

### 9.1 Skill 与 MCP 的差异

| 维度 | Skill | MCP |
| :--- | :--- | :--- |
| 受众 | 面向所有人（含非开发者） | 面向开发者 |
| 层级 | 上层应用：封装业务能力 | 底层协议：Agent 与外部服务通信 |
| 加载 | 渐进式懒加载 | 一次性加载，上下文占用大 |

MCP 负责与外部服务通信；工具编写、流程封装更适合用 Skill，以降低上下文消耗。

### 9.2 SKILL.md 结构与加载机制

Skill 以文件夹形式存放于 `.claude/skills`（如 `frontend-design`）。至少包含 **SKILL.md**，含 YAML front matter。

| Level | 文件 | 上下文窗口 | # Tokens |
| :---- | :--- | :--------- | :------- |
| 1 | SKILL.md Metadata (YAML) | Always loaded | ~100 |
| 2 | SKILL.md Body (Markdown) | Skill 触发时加载 | <5k |
| 3+ | Bundled files | Claude 按需加载 | unlimited* |

`description` 需精确，用于 Skill 发现与匹配。可捆绑参考文档、脚本、Schema。

### 9.3 运行环境与安全

| 环境 | 说明 |
| :--- | :--- |
| **Claude API** | 沙箱：无网络、不可随意 pip install |
| **Claude.ai** | 可上传 Zip Skill，可有部分联网 |
| **Claude Code** | 本地运行，具备完整系统权限 |

> [!WARNING]+ 安全注意
> 勿使用来源不明的 Skill Zip；使用前解压检查 `SKILL.md` 与 `.py` 的实际行为。自建 Agent SDK 环境需做好隔离。

### 9.4 预置 / 内置 / 自定义技能

- **预置**：Anthropic 官方提供，`https://github.com/anthropics/skills/tree/main/skills`
- **内置（隐式）**：pptx / xlsx / docx / pdf 等在 Claude.ai 中自动启用
- **自定义**：封装工作流，上传后通过自然语言触发

### 9.5 frontend-design 示例

> [!ABSTRACT]- frontend-design 技能定义与设计思维
> **技能定义**：创建独特、生产级前端界面；适用于网页组件、着陆页、仪表盘、React 组件等。来源：[Claude Blog](https://www.claude.com/blog/improving-frontend-design-through-skills)
> **设计思维**：目的 → 基调（极简/极繁/复古未来等）→ 约束 → 差异化。关键：意图性比强度更重要。
> **实现标准**：生产级、视觉一致、细节打磨到位。

<!-- -->

> [!ABSTRACT]- 前端美学指南要点
> **字体**：避免 Arial、Inter；搭配独特展示字体与精致正文。**色彩**：CSS 变量；主色 + 锐利点缀。**动效**：HTML 纯 CSS；React 可用 Motion；聚焦高影响力时刻。**空间**：不对称、重叠、对角线；留白或密度控制。**背景**：渐变网格、噪音、几何、阴影等。**避免**：通用 AI 美学、白底紫渐变、可预测布局。

<br>

## 十、总结

Agent Skills 实现：

1. **知识持久化**：文件系统
2. **Token 高效利用**：渐进式披露
3. **逻辑确定性**：脚本 / 代码执行

建议从简单场景入手（如周报格式 Skill），实践「定义一次、到处运行」的工作流封装。
