---
title: DeepSeek Harness & Super-agent 技术报告整理
date: 2026-11-01
categories:
  - 学AI/DS
tags:
  - LLM
  - Agent
  - DeepSeek
  - Harness
desc: 整理自 DeepSeek Harness 官方仓库文档、Cordis 论文 (arXiv 2608.25512)、DeepSeek Harness 团队长文《Exploring Super-agent Design - Towards a More Autonomous Agent System》及其开源原型 dsh-superagent-prototype 的 000~008 设计文档。按"原文要点 + 我的理解"逐节梳理。
---

## 原始出处

**官方一手资料**

| 类型 | 资料 | 链接 |
|---|---|---|
| 产品页 | DeepSeek Harness 官网 | <a href="https://www.deepseek.com/en/harness/">deepseek.com/en/harness</a> |
| 代码仓库 | deepseek-ai/deepseek-harness（MIT；引用条目 *DeepSeek Harness: Everything is a Plugin*, DeepSeek-AI, 2026） | <a href="https://github.com/deepseek-ai/deepseek-harness">GitHub</a> |
| 开发者文档 | 文档站 | <a href="https://deepseek-harness.github.io/deepseek-harness/">deepseek-harness.github.io</a> |
| 架构文档 | architecture / agent-lifecycle / tool-execution-pipeline / cordis-primer / capability-seams / glossary / tool-catalog | <a href="https://github.com/deepseek-ai/deepseek-harness/tree/master/docs">docs/</a> |
| 安全说明 | SAFETY.md | <a href="https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.md">SAFETY.md</a> |
| 论文 | *A Programming Paradigm for Spatiotemporal Composability*（Yifan Shi, Wei Zhang, Tianyi Cui；北大 & DeepSeek-AI；2026.08.26；92 页） | <a href="https://arxiv.org/abs/2608.25512">arXiv 2608.25512</a> |
| 设计长文 | *Exploring Super-agent Design: Towards a More Autonomous Agent System*（DeepSeek Harness 团队，作者 Yujia Qiao；2026.11.01） | <a href="https://zhuanlan.zhihu.com/p/2099987320443237060">知乎中文原文</a> / <a href="https://x.com/DeepSeekHarness/status/2116857257245987236">X 英文版（机翻）</a> |
| 开源原型 | deepseek-ai/dsh-superagent-prototype（MIT，研究原型） | <a href="https://github.com/deepseek-ai/dsh-superagent-prototype">GitHub</a>，docs/000~008 + SETUP.md |

**推荐的解析文章**

| 文章 | 适合看什么 |
|---|---|
| <a href="https://habr.com/en/articles/1070958/">Inside DeepSeek Harness: Cordis, Session Events, Tool Pipelines, and Permission Boundaries (Habr)</a> | Harness 源码层面最清楚的一篇，含插件示例和 Python SDK |
| <a href="https://explainx.ai/blog/deepseek-super-agent-architecture-router-worker-triggers-watchdog-2026">DeepSeek Super-agent: Router Beat One Session by 39% (explainx.ai)</a> | Super-agent 各模块 + 实验表格 |
| <a href="https://github.com/yu-iskw/agent-works/issues/184">agent-works Issue #184</a> | 按原文 §2–§8 逐节摘要 |
| <a href="https://dev.to/paullovelace/deepseek-super-agent-a-proactive-open-source-coding-agent-prototype-what-it-actually-is-and-438a">What It Actually Is, and What It Is Not Yet (DEV)</a> | 讲清"文章设计 vs. 开源原型"的边界 |
| <a href="https://www.marktechpost.com/2026/11/01/deepseek-dsh-superagent-router-worker-coding-agent/">MarkTechPost</a> | 发布新闻 + 原型概览 |
| <a href="https://www.ithome.com/1/020/615.htm">IT之家</a>、<a href="https://zhuanlan.zhihu.com/p/2100268290350368746">知乎：突发！DeepSeek重磅开源：超级Agent架构！</a>、<a href="https://36kr.com/p/4050650194630536">36氪：详解Super Agent：会主动干活，更懂何时闭嘴</a> | 中文快速入门 |

> **几点说明**
> 1. DeepSeek **没有**发布一份叫 "Harness technical report" 的正式论文。所谓"技术报告"由三部分构成：Cordis 论文（理论底座）、Harness 仓库 `docs/`（运行时架构）、Super-agent 长文 + 原型 docs（上层系统设计）。本文按这三层由下到上整理。
> 2. Super-agent 长文：本文按原文 §1–§9 的结构逐节转述并保留原文术语、数据与图注，但不是全文翻译，原文请看上面的链接。Harness 与原型文档是 MIT 许可，本文对其做了较多的摘译。
> 3. 每节末尾的 **「💡理解」** 是我自己的解读，不属于原文。

<br>

## 0. 总览

### 0.1 三层关系一图看懂

```text
┌──────────────────────────────────────────────────────────────────┐
│ ③ Super-agent（设计长文 + dsh-superagent-prototype）                 │
│    Router 常驻会话 ── 派单 ──> Worker 隔离会话 × N                    │
│    Trigger / Thread / Inline&Spill / Chain of Recall / 推测执行        │
│    全双工 · 多渠道 · 计算-状态-环境分离(BYOM) · Watchdog               │
│    → 解决"人的注意力只能盯 3~5 个 session"                            │
├──────────────────────────────────────────────────────────────────┤
│ ② DeepSeek Harness（dsh，Agent 运行时）                              │
│    Session 事件日志 · Agent Loop · 工具流水线 · 沙箱/审批 · LLM 适配     │
│    Profile / Bundle / Patch 组装 · Web / Headless / SDK / ACP / 桌面   │
│    → "模型可见即已记录"，一切能力都是可替换的 seam                       │
├──────────────────────────────────────────────────────────────────┤
│ ① Cordis（插件元框架，arXiv 2608.25512）                              │
│    Context · Service · inject · 类型化事件 · 可逆副作用 · Fiber · HMR    │
│    → 时间可组合（卸得干净）+ 空间可组合（依赖响应式管理）                  │
└──────────────────────────────────────────────────────────────────┘
```

一句话概括：**Cordis 让一切都能热插拔且卸得干净；Harness 用插件拼出一个可回放、可审计的 Agent 运行时；Super-agent 在 Harness 之上用"一个 Router + 多个 Worker"，把 Agent 从"人盯着的会话"变成"常驻的团队成员"。**

### 0.2 时间线

| 时间 | 事件 |
|---|---|
| 2026.08.26 | Cordis 论文提交 arXiv（cs.PL / cs.SE） |
| 2026 年内 | DeepSeek Harness 开源，developer preview，官网称 "now in public preview worldwide" |
| 2026.10.31 | dsh-superagent-prototype 开源（MIT） |
| 2026.11.01 | Super-agent 设计长文发布（知乎中文原文，X 发英文机翻版） |

### 0.3 推荐阅读路线

1. 只想知道 Super-agent 是什么：读本文 §4（长文）+ §6（设计与原型对照）。
2. 想基于 Harness 开发插件：读 §2.4（Cordis 入门）→ §3（运行时）→ 官方 `docs/cookbook/`。
3. 想自己部署原型：读 §5（原型）→ 原型仓库 `SETUP.md` 和 `docs/006-deployment.md`。
4. 对理论感兴趣：读 §2.1~2.3（论文）。

<br>

## 1. 背景：什么是 Harness，为什么 DeepSeek 要做

**Harness（智能体运行框架，直译"挽具"）** 指包在 LLM 外面、让它真正能干活的那一层软件。它负责组装提示词、注册与执行工具、管理上下文与会话、权限与沙箱、出错重试、上下文压缩、子 agent 调度等。Claude Code、Codex CLI 都属于这一类。行业已有共识：模型能力相同时，harness 设计对 agent 实际效果影响很大。

DeepSeek Harness（命令行名 `dsh`）的基本信息：

| 项 | 内容 |
|---|---|
| 定位 | DeepSeek 开发的开源 agent harness，"一切皆插件"，由 Cordis 驱动 |
| 状态 | **开发者预览**，快速迭代，"未来将出现破坏兼容性的变更"；CLI 仍是 release candidate |
| 许可 | MIT；第三方依赖见 `THIRD_PARTY_NOTICES.md` |
| 形态 | Web UI、Headless 一次性运行、TypeScript SDK、Python SDK、ACP 服务器、Electron 桌面端（macOS Apple silicon / Windows x64） |
| 默认模型 | 路由 `deepseek-official`，模型 `deepseek-v4-flash` / `deepseek-v4-pro`（均标称 1M 上下文） |
| 官方插件 | Agent teams、Auto approval reviewer、Scheduled tasks、Voice input、Terminal、Agent loop、Subagents（多数标 Experimental） |
| 插件生态 | 插件仓库加 GitHub topic `dsh-plugin` 便于被发现；官网支持在聊天里用 "Creator mode" 让 agent 帮你写插件 |
| 社区 | GitHub Discussions、Discord、企业微信群 |

**快速上手**（摘自 README）：

```sh
# 方式一：npm，默认在 http://127.0.0.1:3080 启动 Web UI，本机启动时自动打开浏览器
npx @deepseek-ai/dsh web          # 加 --no-open 只跑服务器；SSH 下只打印宿主机 URL

# 方式二：源码
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm run build     # 准备仓库产物
pnpm dsh web       # 直接用已构建产物，不会重新构建

# 开发
pnpm run dev:web   # 一个终端里完成构建、启动、源码修改时重建 client bundle
make help          # 列出 Web 与 Desktop 的 Make target

# 其他运行方式
dsh --profile headless "<task>"    # 无界面一次性运行
```

Python SDK 用法（据 Habr 解析）：`DeepSeekHarness(provider, model, cwd, session_root, cordis)` 创建客户端，`.run(task, session_id=...)` 返回 `result.final_response`；复用同一 `session_id` 会保留对话状态和会话自有的 shell 进程。官方 `BENCHMARK.md` 建议跑评测时用 Python SDK 的 `jsonrpc-agent` 极简变体，**每个独立任务用独立的工作区和 session ID**。

<br>

## 2. 第一层：Cordis —— 插件化的形式基础

### 2.1 论文动机：为什么需要"动态可组合性"

论文开篇的论点：传统软件组合是**静态**的（函数调用、模块 import、继承在编译期就确定）；而插件系统和"自进化 agent harness"需要**动态组合**，即在运行时加载、卸载、重配置组件。现实中大家只能退回到"重启"这种粗粒度手段，代价是丢掉所有运行时状态。

论文把问题拆成两个**正交**维度：

| 维度 | 含义 | 静态场景下退化成 | 动态场景下的难点 |
|---|---|---|---|
| **时间可组合性** (temporal) | 组件被移除时，它对共享环境的修改必须被**完全、安全地撤销**，需要追踪每一次资源分配、事件注册、状态变更 | 词法作用域（RAII、bracket 模式） | 副作用长期存在、有状态，作用域不受词法约束 |
| **空间可组合性** (spatial) | 组件能以结构化、可验证的方式**声明、发现、解析**彼此依赖，并在依赖变化时协调生命周期 | 模块 import 解析 | 依赖会在运行中出现、消失、更换身份 |

**动机例子一：插件系统（以 VSCode 为例）**

- *时间方面*：所有扩展跑在共享的 extension host 进程里，没有卸载单个扩展代码的机制；一旦 `activate` 执行过，禁用或卸载就得**重启整个 host**。论文统计：安装量前 100 的扩展里 **87 个含可执行代码**，卸载时都需要重启。`deactivate` 钩子只是进程终止时的优雅关闭回调，而且把"副作用的清理"和"副作用的创建"分在两个函数里，违反了关注点局部性，很难保证清理完整。
- *空间方面*：VSCode 有 `extensionDependencies`，但前 100 里**只有 7 个**声明了对非内置扩展的依赖；扩展间交互靠 `vscode.extensions.getExtension(...).exports`，返回值默认是 `any`，没有结构化契约。

**动机例子二：自进化 Agent Harness**

现代 harness 要组合工具集、执行环境、权限与沙箱、会话持久化、上下文与记忆、子 agent 与多 agent 工作流，以及对人和自动化的接口。未来的 harness 可能**一边持续服务请求、一边生成并部署对自身组件的修改**，每一次修改都是一次动态组合。修改频繁且少有人工监督时：

- 没有时间可组合性：每次自修改都要全量重启，丢掉进程内积累的状态，正在进行的任务反复被打断；更糟的是，一次错误的自修改可能把负责恢复的那个进程也弄坏。
- 没有空间可组合性：每个模块都得用临时手段自己检测依赖模块的出现、消失或更换；朴素的代码替换会悄悄破坏依赖方，或引入只在 reload 时才暴露的循环依赖。

**粗粒度替代方案的代价**：操作系统在进程粒度上提供时间可组合性（坏了就重启进程），容器编排在服务粒度上提供空间可组合性。但重启要丢掉缓存、连接、部分计算，重建要几秒到几分钟，还得靠冗余副本维持可用性；容器编排表达不了同一地址空间内组件间的依赖，而且把本可以是本地函数调用的交互变成了网络调用。论文称之为**粒度错配**。

> **💡理解**：这段动机正好解释了 DeepSeek 为什么要先做一个"框架的框架"。agent harness 的组件（工具、提示词片段、模型适配器、监听器）数量多、变化快，还可能由 agent 自己生成。如果每装一个插件、改一段提示词都要重启进程，长时间运行的 agent（比如 §4 的常驻 Router）就无从谈起。

### 2.2 论文贡献与理论要点

论文的切入点：**effect（效应）** 系统形式化"计算如何修改环境"，**coeffect（余效应）** 系统形式化"计算对环境有什么要求"。但已有工作都只做编译期、固定词法作用域上的分析。论文把两者**提升为运行时机制**，得到语言无关的动态可组合性基础。

五项贡献：

1. **可逆副作用 (revertible effects，§3.1)**：每一次上下文变换都附带一个**由运行时持有的显式逆操作**，追踪和恢复都保持组合性，所以组件被移除时上下文能被恢复。→ 局部时间可组合性。
2. **响应式余效应 (reactive coeffects，§3.2)**：组件把自己需要的余效应声明为一个**规约 (specification)**；上下文每次变化都对照该规约被分类为 **激活 / 停用 / 中性** 三类之一，以此驱动组件的激活与停用。→ 局部空间可组合性。
3. **上下文范式 (context paradigm，§3.3)**：把效应上下文和余效应上下文**统一成一个 Context 类型**，所有效应和余效应都经它中介；这种中介诱导出一种**观察等价**，在此意义下不同组件的副作用可以交错而互不干扰。
4. **动态组合演算 (§4)**：把两种机制合成"组件 (component)"的概念并给出操作语义，元理论（Preservation、Temporal / Spatial Composability、Progress、Confluence）把时空可组合性从单个组件推广到整个系统。
5. **Cordis 实现 (§5)**：一个"时空可组合性元框架"，包括带副作用追踪和余效应解析的核心库，以及带配置协调与热模块替换的声明式组件加载器。

**理论构造 ↔ Cordis 运行时对应表**（论文 Table 2 节选）：

| 理论 | 实现 |
|---|---|
| Γ∞ | `ctx`，一等公民的上下文 |
| effectΓ(e) | `ctx.effect(callback)`，callback 返回（或 yield）逆操作 |
| get(k) / set(k, v) | `ctx.get(key)` / `ctx.set(key, value)` |
| isolate(k, r) | `ctx.isolate(key, realm)`，按 realm 隔离某个服务 |
| intercept(k, ν) | `ctx.intercept(key, metadata)`，在上下文上附加拦截元数据 |
| ⟨d, p, e, π, σ, τ, θ⟩ | `fiber`，组件的一次实例化 |
| d（依赖）/ p（提供） | `fiber.inject` / 组件的 `provide` |
| e（效应） | `fiber.apply` |
| π（父） | `fiber.parent.fiber.uid` |
| 派生实现 | `fiber.ctx`，fiber 运行所在的子上下文 |
| θ（生命周期状态） | `fiber.state` |

论文 §5 把实现分三层：**(1) 核心库**直接实现效应与余效应系统；**(2) 组件加载器**加上配置协调和 HMR；**(3) 应用框架**（如 Koishi）在其上构建领域功能。Cordis 被定位为"元框架"：不规定具体场景（不像 Web 路由、ORM、UI 渲染框架），唯一职责是提供通用的动态组合语义。

**§6 讨论里两个和 agent 安全直接相关的点**：

- **系统边界 (§6.1)**：每个效应都带逆操作，但"逆操作能做到什么"取决于系统边界。一个位置如果系统能**独占修改并恢复**，它就在边界内，操作会被记录、可回滚；否则在边界外，操作视作恒等，不追踪也不回滚。余效应可以通过"具象化 (reify)"外部位置来移动边界：把对它的所有访问收敛到一组自己能提供逆操作的操作上。例如私有路径下的临时文件在边界内，别的程序也会读写的路径就在边界外。
- **访问控制与沙箱 (§6.3)**：`inject` 依赖声明本身就是一种**基于能力 (capability-based)** 的访问控制：组件只能访问声明过的依赖，未声明的访问会报错；声明是静态的，所以编排者可以在**加载时**审阅批准。更细的策略通过 interception 实现，例如给文件系统依赖附加"可读写哪些路径"的元数据，provider 每次调用时检查。拦截挂在 context 上，可以运行时安装、重配、移除，不触发重载。但**沙箱隔离不可信代码需要外部沙箱**，Cordis 不负责。

**案例研究 (§5.3)：Koishi**。一个基于 Cordis 的开源聊天机器人框架，开发四年、社区插件超过 **4000 个**（IM 适配器、数据库驱动、管理后台、终端功能等）。它的 Web 控制台是另一个独立的 Cordis 应用，跑在浏览器里。论文用它说明模型既有表达力（原语足以承载完整生产系统，宿主只提供领域词汇），又有通用性（不预设领域和运行时）。Koishi 可以在控制台里禁用插件并就地回滚其副作用；开发时 HMR 在保存时重新应用改动的插件，同时保留别处的缓存和活跃连接。（注：Koishi 目前用 Cordis v3，论文介绍的是 v4，后者细化了效应/余效应语义并重新设计了 loader。）

> **💡理解**：
> - 把论文翻成工程话：**"每次注册都必须返回一个撤销函数"（时间维度）**，**"声明你依赖谁，依赖到位你才活，依赖走了你就自动停"（空间维度）**，**"所有注册都经过同一个 ctx，所以框架能记账"（上下文范式）**。
> - 这和 React 的 `useEffect(() => { ...; return cleanup })`、Go 的 `defer`、RAII 是同一思想，只是从"词法作用域"推广到"运行时任意长的插件生命周期"。
> - 对 agent 安全的启示：Cordis 能保证"插件卸得干净、只能用声明过的能力"，但保证不了"插件不去碰边界外的东西"，所以 Harness 另外有 sandbox seam，SAFETY.md 也强调沙箱不是安全保证。

### 2.3 Vendored 的 Cordis 包

Harness 把 Cordis 及其基础库以源码形式放在 `vendor/` 下，并以 `@deepseek-ai` scope 重新发布（因为每个 harness 包都把框架声明为 peer dependency，用上游名发布等于占用别人的名字）。摘自 `docs/rescope.zh.md`：

| 发布名 | 上游名 | 上游版本 | 角色 |
|---|---|---|---|
| `@deepseek-ai/cordis` | `cordis` | 4.0.0-rc.11 | 框架核心：`Context`、`Service`、`Fiber`、事件 |
| `@deepseek-ai/cosmokit` | `cosmokit` | 1.8.1 | 框架与 Schemastery 共用的基础工具 |
| `@deepseek-ai/schemastery` | `schemastery` | 3.18.0 | 配置 schema（`Schema`），每个插件的 `Config` 基于它 |
| `@deepseek-ai/cordis-plugin-loader` | `@cordisjs/plugin-loader` | 1.0.0-rc.5 | `cordis.yml` 装载、插件解析、repository 缓存 |
| `@deepseek-ai/cordis-plugin-include` | `@cordisjs/plugin-include` | 1.0.4 | 配置包含与 patch 叠加 |
| `@deepseek-ai/cordis-plugin-group` | `@cordisjs/plugin-group` | 1.0.0 | 嵌套插件分组 |
| `@deepseek-ai/cordis-plugin-timer` | `@cordisjs/plugin-timer` | 1.1.4 | `ctx` 上随 disposal 回收的定时器 |
| `@deepseek-ai/cordis-plugin-hmr` | `@cordisjs/plugin-hmr` | 1.0.15 | 插件与配置的热替换 |
| `@deepseek-ai/cordis-plugin-logger-console` | `@cordisjs/plugin-logger-console` | 1.0.0 | 控制台日志导出 |

迁移时自己的代码要改：`import { Context } from 'cordis'` → `from '@deepseek-ai/cordis'`；`declare module 'cordis'` → `declare module '@deepseek-ai/cordis'`；`package.json` 依赖键和 `cordis.yml` 的插件 `name` 同理。仓库提供 `pnpm run rescope-vendor [--apply] [--reverse]` 脚本自动完成。

### 2.4 Cordis 入门（摘译自 `docs/cordis-primer.zh.md`）

**五个核心概念**

1. **插件是实现 Service 的对象。** 可以是一个带可选 `inject` 和 `apply(ctx)` 字段的函数，也可以是一个 `Service` 子类，其生命周期由 Cordis 挂载到当前上下文。
2. **上下文是服务的容器。** 一个服务占据一个稳定的 `ctx.<key>`（如 `ctx.tools`、`ctx.llm`、`ctx.sessions`）；其他插件通过 key 查找服务，而不是导入具体实现。
3. **通过 `inject` 声明服务依赖。** 插件声明所需服务后，会等这些服务就绪才启动；加载顺序通过服务依赖表达，而不是手动编排启动序列。
4. **类型化事件用于通信。** 服务通过 TypeScript 声明合并注册事件名，再以 `emit` / `waterfall` / `parallel` / `serial` / `bail` 分发。
5. **注册是可逆的副作用。** 提示词片段、工具 schema、适配器、提供方和监听器通过 `ctx.effect()` 或 `ctx.on()` 安装，reload 和 teardown 时按预期撤销。

一个最小插件长这样（据 Habr 解析整理）：

```ts
import { Context } from '@deepseek-ai/cordis'

export const name = 'my-tool'
export const inject = ['tools']          // 等 ctx.tools 就绪才加载

export function apply(ctx: Context) {
  ctx.tools.register({ /* name, description, parameters, execute */ })  // 插件卸载时自动注销
  ctx.effect(() => {
    const timer = setInterval(() => { /* ... */ }, 1000)
    return () => clearInterval(timer)    // 显式 disposer
  })
}
```

**分发模式**（原文表格）

| 模式 | 是否 await？ | 分发顺序 | 是否有返回值？ |
|---|---|---|---|
| `emit` | 否 | 监听器按注册顺序观察 | 否 |
| `waterfall` | 否 | 监听器按注册顺序观察 | 是 |
| `parallel` | 是 | 所有监听器并行观察事件 | 否 |
| `serial` | 是 | 监听器按注册顺序观察 | 是 |
| `bail` | 否 | 监听器按注册顺序观察，直到某个监听器返回 bail 值 | 是 |

分发模式是事件公开约定的一部分，新事件用 `@mode` 标签记录，生成的目录会把声明和分发调用点交叉校验。

**Waterfall 语义**：`ctx.waterfall` 是**环绕中间件**。监听器接收 `(...args, next)`，调用 `next()` 执行下游监听器，下游返回值经 `next()` 回到当前层，可以包装后继续往外返回；**不调用 `next()` 直接返回就是短路**。协作式监听器通常修改共享的请求/决策对象再委托；也可以整个替换结果。对单决策事件，短路是设计意图：**有决策权的策略监听器可以不调用 `next()`，只做标注或观察的监听器必须委托**。只有必须在普通注册之前运行时才用 `prepend: true`。

**Loader 配置**：`@deepseek-ai/cordis-plugin-include` 把 YAML 里的 `!!js` 解析为表达式节点（例如 `disabled: !!js process.platform === 'win32'`）。Loader 在注入激活后基于插件上下文插值 `config`，在每次挂载决策时插值 `disabled`。按环境选择插件时用 overlay。

**实践规则**：行为封装成插件；工具流水线事件属于 `ctx.tools`，模型流式输出属于 `ctx.llm`，实时 agent 协调属于 `ctx.agents`；拦截和策略优先用事件，直接能力调用优先用服务方法；**每个注册都要有对应的 disposer**，teardown 顺序有要求时把相关工作放进同一个 effect。

> **💡理解**：`waterfall` 是整个 Harness 扩展模型的核心。它就是 Koa / Express 的洋葱中间件：审批插件可以在 `tools/pre-execute` 里不调 `next()` 直接返回 deny；计时插件包在 `tools/execute` 外层，调用 `next()` 前后各记一次时间。

<br>

## 3. 第二层：DeepSeek Harness 运行时架构

> 本节主要摘译自 `docs/architecture.zh.md`、`agent-lifecycle.zh.md`、`tool-execution-pipeline.zh.md`、`capability-seams.zh.md`、`glossary.zh.md`、`tool-catalog.zh.md`。官方建议：改动 `packages/` 下任何内容前先读架构文档。

### 3.1 设计原则："没有特权内核"

原文要点：产品的每一部分都是插件，**包括模型适配器、工具注册表、会话日志，以及 agent loop 本身**，所以每一个都可以从配置替换。不存在需要打补丁的特权内核：扩展 dsh 的方式是把插件挂到其他插件旁边，各项注册都是副作用，插件卸载时撤销。

### 3.2 Profile / Bundle / Patch：一棵在启动时组装的插件树

运行中的 `dsh` 是一棵插件树，由启动时按序叠加的各层组合而成：

```text
空条目列表
  └─ + profile 中有序的 bundles（如 dsh-base → dsh-web-app）
       └─ + profile 自己的 cordis.patch.yml
            └─ + Harness home 的 cordis.patch.yml（用户全局）
                 └─ + 命令行 --patch overlay
                      = 最终插件树（dsh --profile web --dump-config 可打印）
```

- **Profile**：Harness home 里一个具名组装，列出叠放的 bundle、存放自己安装的树外插件、保存用户自己的 `cordis.patch.yml`。随发行版提供 `web`、`headless`、`sdk`、`sdk-minimal`、`acp` 五个模板。
- **Bundle（组合包）**：Cordis 配置行及其挂载代码的分发格式，插入的内容始终可被上层 patch。两者都在 `package.json` 的 `dsh` 字段声明自己：`dsh.profile` 列出 bundle，`dsh.bundle` 指向 patch 文件。

| Bundle | 作用 |
|---|---|
| `dsh-base` | `web` / `headless` / `sdk` / `acp` 共享的第一层：模型适配器、工具、持久化、沙箱与审批策略、设置、凭据、遥测；还提供 Web 和 Agent 用的插件管理器 |
| `dsh-web-app` | 浏览器应用 |
| `dsh-headless` | 不带服务器的一次性运行器 |
| `dsh-sdk-app` | SDK JSON-RPC 服务器 |
| `dsh-acp-app` | 仅用于自动化的 ACP 服务器 |
| `dsh-sdk-minimal` | 刻意保留的例外：拥有完整显式 SDK 配置树，**不应用 `dsh-base`** |

- **Patch**：按 id 定位条目并**替换其整个 config**，或插入新条目；`preset` patch 在目标声明的 `config.plugins` 内应用这些操作，沿用相同的层优先级。
- **HMR 由 YAML 控制**：base 启用仅监视配置的 `dsh-hmr`；Headless、SDK、ACP 禁用它；`sdk-minimal` 不包含它。

**应用启动路径**

- 受支持的 Node 应用**只能**通过具名 `dsh` profile 启动（`dsh --profile <name>` 或 `dsh <name>`；`plugin` 是管理命令，同名 profile 要写 `--profile plugin`）。脚本 `verify-application-entrypoints` 会拒绝任何绕过 `dsh` 的 Node 应用路径。
- TypeScript SDK 解析同版本 `dsh` 依赖并选择 `sdk` profile。
- Python SDK 的运行时 wheel 把普通 `dsh` CLI 打包为 `deepseek-harness-sdk-runtime-<platform>-<arch>`，默认以显式 Harness home 启动 `dsh --profile sdk`；极简示例选 `sdk-minimal`。Python 只暴露 profile 选择与有序 patch 文件，持久外部插件通过 `dsh plugin` 安装。
- **桌面端**：Electron 在签名资源中携带精确匹配的 dsh 生产运行时，拥有保留的 `$DSH_HOME/profiles/desktop`。Electron 以 Node 模式启动私有 Desktop Host，Host 调用共享 CLI profile runner 与完整 Web 应用；窗口先加载打包的 Web 资源，等启动注入后在同一文档中激活客户端插件。Node IPC 承载启动注入、就绪、致命错误与关闭。

> **💡理解**：这套机制类似"Docker 镜像分层 + Kustomize patch"。好处是：同一份代码组合出 Web、CLI、SDK、桌面四种产品；用户只写一个 patch 就能替换任何一个内置插件，不用 fork。代价是要读懂一棵配置树，所以官方提供 `--dump-config`。

### 3.3 核心包

| 包 | 职责 | `ctx` 键 |
|---|---|---|
| `core/session` | 仅追加的 `SessionEvent` 日志和内存存储 | `ctx.sessions` |
| `core/system-prompt` | 提示词片段与工具 schema 的组装 | `ctx.systemPrompt` |
| `core/tools` | 作用域化的工具注册表和带把关的执行流水线 | `ctx.tools` |
| `core/agent` | `Agent` 接口、活跃 agent 注册表和 `agent/*` 事件 | `ctx.agents` |
| `core/agent-loop` | 实现该接口的默认驱动器 | `ctx.agentLoop` |
| `core/scope` | 按 agent 划分作用域的注册原语 | 库，无 ctx 键 |
| `llm/llm` | 消息与流式词汇表，以及适配器 seam | `ctx.llm` |

### 3.4 三类事件域："事件就是扩展点"

原文：事件就是扩展点，选对事件域是大多数改动的第一个决定。

| 事件域 | 例子 | 持久？ | 什么时候用 |
|---|---|---|---|
| **会话事件** | `turn/*`、`step/*`、`system/message`、`user/message`、`assistant/message`、`assistant/attempt`、`tool/*` | 是，追加到日志并经 `session/event` 广播 | 某个事实必须在重新加载后仍然存在 |
| **Agent 事件** | `agent/pre-step`、`agent/request`、`agent/request-error`、`agent/turn-stopping`、`agent/inbox/*`、`agent/status`、`agent/assistant-stream` | 否，携带活跃 `Agent` | 观察或拦截进行中的工作（inbox、步骤、状态、请求、验证、续跑） |
| **能力事件** | `fs/*`、`tools/*`、`telemetry/*` | 否 | 不用 import 循环，就能在某个 seam 上附加策略和适配器 |

另有生命周期细节：AgentLoop 初始化 Agent，失败则回滚，并在启动已排队工作前等待串行的 `agent/created`；工厂 setup 在 Session 进入 SessionStore 并完成 `session/created` 之后、`agent/created` 之前运行。官方 `event-producer-consumer.zh.md` 列出每个事件的生产方与消费方。

### 3.5 Agent Loop：轮次 (turn) 与步骤 (step)

**术语**（摘自 glossary）：

| 术语 | 定义 |
|---|---|
| **步骤 (step)** | 一次模型请求，加上由模型响应引发的工具执行 |
| **轮次 (turn)** | 会话中一次对已接纳输入的排空过程，包含零个或多个步骤；在领取首条输入之前打开，不再欠任何工作时关闭 |
| **Round** | 承载一个轮次的外层策略迭代，如一个 Goal Round 或一次 Ralph 尝试；计数器归策略所有 |
| **交互段 (exchange)** | 从首个步骤接纳真人输入的轮次起，到下一个这样的轮次为止的连续轮次；被通知（子 agent 完成、Goal Round、定时任务）唤醒的轮次归入前一个交互段 |

**轮次流程**（原文代码块）：

```text
turn/start
  claim next-step input plus one queued message
  snapshot Session model selection; assemble prompt sections + step-admitted tool schemas
  -> agent/pre-step; project runtime context at fallback
                                     reject | enter(messages, startsRequestSeries?)
     reject, or a first enter rewritten empty -> close the turn with no step
     step/start
     agent/request -> prepareCall (cancellation commits neither system nor users)
     reconcile system/message using the prepared call capability
     append entered messages as user/message; reconcile retained runtime context
     log request/header and request/context as needed
     derive and freeze model history from the log
     stream the bound prepared call -> llm/stream -> agent/assistant-stream start
       agent/assistant-stream chunk*
       assistant/message | assistant/attempt -> agent/assistant-stream end
     tool/call* -> tools/pre-execute -> tools/execute -> tools/post-execute -> tool/result*
     step/end
     tools owe another request, or next-step input arrived -> claim -> next step
  -> agent/turn-stopping
turn/end
```

逐步解读：

1. **领取输入**：领取下一步待处理输入加一条排队消息。所有输入（用户追加消息、注入的上下文）都走同一个 inbox；注入的上下文要等一条唤醒消息。AgentLoop 的持久 `inbox` 投影让待处理输入在没有活跃 Agent 时也能读取。
2. **快照与组装**：把 Session 的模型选择快照到 `AssembleContext.modelSelection` 和 `agent/pre-step.modelSelection`（重试沿用；省略推理强度时用适配器默认值），组装提示词片段和本步获准的工具 schema。
3. **`agent/pre-step`（waterfall）**：监听器接纳、改写或拒绝已领取的消息。**首次领取被拒绝或改写为空 → 关闭一个不含步骤的轮次**。enter 决策可设 `startsRequestSeries`，此时循环记录新的 `request/header`。
4. **`step/start` → `agent/request` → `prepareCall()`**：绑定适配器。**在这两个异步阶段取消，系统提示词和用户消息都不会被提交**。
5. **落日志**：按已准备调用的能力协调 `system/message`；把接纳的消息记为 `user/message`；按需记录 `request/header`、`request/context`。
6. **从日志派生并冻结请求**，经 `llm/stream`（waterfall）流式发送，实时 chunk 经 `agent/assistant-stream` 推给 UI。
7. **结算**：成功 → `assistant/message`；失败、重试、取消、流错误 → `assistant/attempt`（**保留审计但不进模型历史**）。
8. **工具**：按 `executionMode` 分类待执行调用，在屏障和有界滚动池里执行（启动前会重新分类）：每个调用 `tool/call` → 有序的 pre → 并发 execute → 有序的 post → 按模型顺序记录 `tool/result`。
9. **`step/end`**，释放本步工具集。工具结果还要回给模型，或者有新输入到达，就领取输入进入下一步。
10. 自然停止且下一步 inbox 为空时，触发 **`agent/turn-stopping`（serial，终止检查点，没有 `next()`）**，然后 `turn/end`，`agent/status` 变为 idle。

**几个重要机制**

- **中途引导 (steering)**：用户在运行中追加的消息在下一个 step 被领取，不用等整轮结束。steering 和注入的上下文都经过同一个 `agent/pre-step` waterfall。
- **失败重试**：请求失败时触发 `agent/request-error` waterfall，监听器可以返回重试动作。**重试发生在仍然打开的 step 内**，会重新准备调用并协调保留的已渲染组装结果，**不重复组装、pre-step 和用户消息准入**。
- **上下文压缩**：`dsh-compaction-basic` 在派生请求之前通过 `agent/pre-step` 处理上下文压力，`agent/request-error` 只用于规范的上下文溢出。任一条件触发后，**先做可选的工具结果剪枝，再选择摘要**。只有剪枝或摘要推进了 surface replacement generation 才重试，否则仍以原始错误为准。注意 `ctx.compaction` 是 seam，**不存在面向模型的压缩工具**。
- **失败的 step 会记录缺失的工具结果**，保证工具调用与结果总是成对，历史可以继续。
- `assistant/message` 记录**每次成功的提供方调用**，包括返回空内容或以 `max-tokens` 结束的调用，并嵌入精确的紧凑带时间 stream；空内容不进入派生历史。进程如果在结算前硬中断，就不会留下持久的 attempt stream。
- SDK 用户如果需要可回放的 transcript，应消费 `session/event`；`agent/*` 是用于队列、状态、提示词拦截、请求构造、steering、续跑和错误处理的实时协调接口。

> **💡理解**：大多数 harness 的循环是"调模型 → 执行工具 → 再调模型"的一个 while 循环。dsh 的循环把每个阶段都拆成可拦截的事件，并严格区分两类东西：**持久事实**（写进 session 日志）和**实时控制**（`agent/*`）。所以 UI、压缩、审批、重试都能以插件形式插进来，又不会让"模型当时到底看到了什么"变得说不清。

### 3.6 会话日志："模型可见即已记录"

这是 Harness 最核心的工程原则（原文加粗）：

> **模型可见即已记录。** 每个模型请求都必须能从日志重建。新增模型可见输入需要会话事件。

展开：

- 会话日志是模型所见上下文的**唯一来源**，`deriveMessages()` 从中投影出模型历史。修改现有消息内容的插件要注册**纯消息投影**，独立读取器显式传入相同的处理器。
- 每个 `assistant/message` 嵌入产生其内容的精确紧凑带时间 stream；`assistant/attempt` 保留已结算的失败、重试、取消与流错误尝试，但不添加模型历史。
- **fork、恢复、transcript、遥测与持久化都从这些持久结算派生**，实时 UI 增量则来自 `agent/assistant-stream`。
- **版本化与迁移**：Session 消费方只了解当前逻辑格式。JSONL v0 用 `session.jsonl[.zstd]`，v1 及之后用 `session.vN.jsonl[.zstd]`。打开已存储的 Session 时选择数值最高的规范 generation、拒绝未来版本、只组合构建时确定的相邻迁移链；写 open 会在源文件旁边排他发布新版本的后继文件。**已提交的 generation 路径绝不重命名、替换或删除**。每个迁移包只负责一步 `vN → vN+1`。
- **投影 seam**：`dsh-session-projection` 提供 `ctx.sessionProjections`，注册单元折叠已提交事件，客户端通过 `snapshot()` 读视图。`AgentRegistry` 通过 `turnBoundaryOf(session)` 提供轮次和步骤状态。
- **fork 会话**：`ctx.agents.create({ sessionId, seed, inheritedEventCount, meta: { isSeeded: true } })`，只有经 agent-loop 发布的会话才持久化。

> **💡理解**：这是把**事件溯源 (event sourcing)** 用在 agent 上下文上。很多 harness 的"上下文"是内存里一个可变的 messages 数组，压缩、重试、注入都直接改它，事后很难说清模型当时看到了什么。dsh 的做法是：上下文是日志的投影，任何变化都是一条新事件。直接收益有四个：(1) 任意时刻可 resume / fork；(2) 评测可以精确复现一次请求；(3) 调试时能看到失败的 attempt；(4) 格式升级有可靠的迁移路径。Super-agent 原型能"重启后恢复每个 Worker 读到哪条消息"，就依赖这一点。

### 3.7 工具执行流水线

原文要点：内置执行器负责原始批次解析、调度、取消和 Session 调用/结果日志；共享调用流水线负责策略、钩子、沙箱、文件系统守卫、结果改写、最终结果观测和 UI 呈现，**而无需改动循环**。

```text
assistant 消息里的 tool-call 块
   │
   ▼
ToolExecutor.execute（原始批次 + 并发上限 + 可选 logging step）
   │  记录 tool/call（执行前先落日志） ──> UI pending 卡片 presentCall(args)
   ▼
tools/pre-execute (waterfall)：hooks、权限、沙箱
   ├─ deny ────────────────────────────────────┐
   ├─ ask ──> ctx.approval 一次性询问             │
   │            ├─ allowed-once ─┐                │
   │            └─ 拒绝/取消/不可答 ──> denied ─────┤
   └─ allow ─────────────────────┤                │
                                 ▼                │
      已注册的单调守卫（只能 deny 或弃权，保护身份）    │
                                 ▼                │
   tools/execute (waterfall)：超时、重试、指标（环绕分发） │
                                 ▼                │
                       工具 execute() 本体          │  denied 时跳过工具本体
     写/改文件经 fs/write-intent / fs/edit-intent    │
     工具自有事件：todo/write、fs/observed、          │
       hook/invoked、hook/result、tool/ptc-dispatch  │
                                 ▼                ▼
                 ToolDefinition.projectContent（准备好的文本与图片）
                                 ▼
   tools/post-execute (waterfall)：accept、block、replace、add context
                                 ▼
      规范化（任何环节抛异常 → isError 结果）
                                 ▼
      finalizeContent（最后的、仅内容的不变式）
                                 ▼
      tools/result（同步通知；冻结的权威结果）
                                 ▼
      记录 tool/result（唯一面向模型的结果）──> UI 完成卡片 presentResult
                                 ▼
      整批结算后，additionalContexts 以 FIFO 方式作为 user/message 注入
```

补充要点（原文）：

- 三个 waterfall 都可以转换调用。
- 文件系统的**先读后编辑检查**位于 `tool-fs` 之下，通过 `fs/*` 事件实现。
- 通用的 pre/post waterfall 承载钩子与审批策略；`ctx.approval` 在单调守卫之前处理询问；**不得重新排序的所有者策略作为已注册的守卫**存在。
- 注册表会对候选结果做无损快照；`tools/result` 观察的是不可变、可由 JSON 无损表示的结果。这样钩子能跨不同工具族生效，工具不必与某个策略服务耦合。
- **PTC 模式**（程序化工具调用，`run_code`）：保留的 `run_code` 传输及其序列化子调用都进入流水线；子调用携带父级 token、记录 `tool/ptc-dispatch`、把拒绝呈现为有约束力的驳回，并省略 `additionalContexts`，以保持调用与结果相邻。

### 3.8 权限：沙箱 ≠ 审批

| 控制 | 回答的问题 | 实现 |
|---|---|---|
| **沙箱 (sandbox)** | 这个动作**能触及**哪些资源？ | `ctx.sandbox` seam：消费方交出即将 spawn 的确切 argv，后端按每次调用的策略包装 argv 并报告强制执行情况；`ctx.sandboxPolicy` 统一保存部署默认模式和工作区根目录，bash 与 fs 读同一份，不会限制到不同的根目录 |
| **审批 (approval)** | 这个动作**要不要**用户确认？ | `ctx.approval` seam：一次性权限决策经 `approval/request` waterfall 分派，**没有回答方时以 `unavailable` 关闭（fail-closed）** |

预设举例：`workspace-write`（沙箱只允许写工作区，审批 `ask`）、`danger-full-access`（不设沙箱，审批 `never`，只应在一次性 checkout 或容器里使用）。官方另有 Auto approval reviewer 插件，用模型代替人做审批。

**SAFETY.md 要点**（摘译）：

- 实验性开发者预览软件，**未经安全审计**，不能当作安全或生产就绪的软件。
- 它会执行模型生成的代码和命令、加载第三方插件、访问网络、进程、凭据和文件。错误的模型输出、缺陷、错误配置、恶意输入或不可信插件都可能损坏主机、改删文件、泄露数据或凭据。
- **沙箱、审批和权限控制可以降低风险，但不能保证隔离或防止损坏**；即使限制正确执行，也保护不了项目被允许访问的资源。不要把 DeepSeek Harness 当作不可信工作负载的唯一安全控制。
- 负责任使用：最小权限；优先用一次性虚拟机、容器或专用环境；备份它能访问的文件；不暴露敏感凭据；运行前审阅插件、配置和命令。

### 3.9 能力 Seam：换一个 provider 就换一个产品

**定义**（摘自 glossary）：seam 是一种包含三种角色的可替换能力：

- **Service Definition**：拥有自身 `ctx.<key>` 和词汇类型的 Cordis `Service`（可以是 `ShellExecutor` 这样的抽象类，也可以是 `WebRuntime` 这样的具体注册表，**绝不是 TypeScript `interface`**）；
- 一个或多个 **Service Provider**；
- 一个或多个注入该服务的 **Consumer**（通常是面向模型的工具）。

规范范例是 `packages/shell`：`dsh-shell`（Definition）、`dsh-bash-local` / `dsh-bash-sandbox`（Provider）、`dsh-tool-bash`（Consumer）。单一角色不算 seam，新增一项能力意味着把三者一起设计。

原文举例：文件系统与进程提供方共享同一个执行世界，**把它们指向远程沙箱，就把 Bash、PTY 和 LSP 一起搬了过去**，无需提供方专用 fork。subagent 提供方也一样，可以是新建一个子 agent，也可以是把一个轮次委派给另一个产品（如进程外 ACP、Codex、Claude Code subagent 后端）。

**核心 seam / 服务一览**（摘自 `capability-seams.zh.md`）：

| ctx 键 | 角色 | 说明 |
|---|---|---|
| `ctx.llm` | seam | 适配器注册提供方实现；agent loop 与压缩调用提供方无关的流服务 |
| `ctx.sessions` | core | 拥有仅追加的 Session 实例，发出持久会话事件流 |
| `ctx.systemPrompt` | core | 为每个步骤收集提示词各部分和工具 schema |
| `ctx.tools` | core | 注册能力和作用域执行器；原生与 PTC 提供者可选用共享流水线 |
| `ctx.agents` | core | 实时 Agent 句柄、创建/恢复工厂 seam |
| `ctx.agentLoop` | bundle | **唯一的具体循环插件**；扩展包依赖 dsh-agent 的事件和服务，而不依赖它 |
| `ctx.credentials` | seam | 配置只携带机密的**引用**，提供方拥有实际值；按操作解析，轮换后下一次请求即生效；设置页只提供不含实际值的视图和只写存储 |
| `ctx.subprocess` | seam | Bash、PTY、LSP Host、进程外 ACP / Codex / Claude Code subagent 后端都经它 spawn |
| `ctx.shell` | seam | 沙箱、远程或 PowerShell 执行器可替换 bash-local，消费方不用改 |
| `ctx.terminals` | seam | 持久化终端；后端负责终端机制，tool-terminal 提供模型接口 |
| `ctx.sandbox` / `ctx.sandboxPolicy` | seam / core | 见 §3.8 |
| `ctx.approval` | seam | 见 §3.8 |
| `ctx.fs` | seam | tool-fs 经它读写编辑；fs-sandbox 按沙箱模式限制变更；fs-observation-policy 基于观测状态做检查 |
| `ctx.compaction` | seam | 消费步骤后的压力事件和请求错误恢复事件 |
| `ctx.subagents` | seam | 提供方准备本地子级或执行外部任务；activation 管理器持有两类生命周期 |
| `ctx.agentTeams` | core（实验） | 隐式 Root roster、peer inbox 直投、共享任务 DAG、可续跑子 agent 生命周期 |
| `ctx.jobs` | seam | 后台 bash/pwsh 与 PTY 登记的运行中工作；`job_*` 工具读取、列出、终止 |
| `ctx.skills` | seam | 合并提供方的 skill 目录，加载完整 skill 正文 |
| `ctx.commands` | core | 直接面向人的斜杠命令，不发给模型 |
| `ctx.schedule` | core | 独立于 Session 存储定时任务，到期消息排入目标 Session |
| `ctx.goals` | core | 从日志折叠带修订号的目标状态 |
| `ctx.webhookRuntime` | core | 已认证的 webhook 交付转成普通 Session |
| `ctx.mcpResources` | seam | MCP 资源访问 |
| `ctx.browserUse` / `ctx.computerUse` | seam | 浏览器操作（Playwright MCP、Chrome DevTools MCP、Stagehand）/ 电脑操作（CUA driver） |
| `ctx.worktrees` | seam（实验） | 从固定本地提交创建 git worktree |

**新行为的归属位置**（原文表格，完整）：

| 目标 | 机制 |
|---|---|
| 添加模型提供方 | 在 `ctx.llm` 上注册适配器 |
| 添加面向模型的能力 | 在 `ctx.tools` 上注册；schema 加入提示词组装 |
| 让某个会话拥有不同的能力集合 | 组装一个 agent preset；其中的服务行需要 `isolate` realm |
| 添加 shell 执行 | 注册 `ctx.shell` 后端；本地后端经 `ctx.subprocess` spawn |
| 添加持久化终端执行 | 注册 `ctx.terminals` 后端和 `dsh-tool-terminal` |
| 添加用户命令 | 在 `ctx.commands` 上注册，无需模型轮次即可分派 |
| 管理后台任务 | 在 `ctx.jobs` 上注册；`job_*` 工具读取或停止 |
| 从外部 webhook 启动 Session | 在 `ctx.webhookRuntime` 上注册可信规则，挂载提供方适配器 |
| 文件系统访问或策略 | 注册 `ctx.fs` 提供方，或监听 `fs/*` |
| 限制所启动的进程 | 用 `ctx.sandbox` 后端；消费方在 spawn 前包装 argv |
| 拦截请求、工具或轮次 | 用相应的 `agent/*` 或 `tools/*` 事件；`agent/turn-stopping` 会停止轮次 |
| 添加模型可见上下文 | 调用 `agent.inject()`，落到下一次获准的请求中 |
| 添加 UI 或编辑器集成 | 驱动 `ctx.agents` 并从 `session/event` 渲染 |
| 添加 Web Client Chat 节点 | 注册 `ConversationNodeDefinition` + keyed renderer |
| 添加持久会话状态 | 扩展 `SessionEventMap`；从日志渲染和回放 |
| 生成会话标题 | 注册唯一的 `ctx.sessionTitle` 提供方 |
| 管理同会话目标 | 用 `ctx.goals`；经 `agent/*` 续跑 |
| 在轮次边界 fork 会话 | `ctx.agents.create({ sessionId, seed, inheritedEventCount, meta: { isSeeded: true } })` |
| 在新后端存储会话 | 基于共享句柄脚手架实现 `SessionPersistence`（`create`/`open`/`stat`/`list`/`export`） |
| 把注册项限定到单个 agent | 用该 agent 的 `agent.ctx` |

### 3.10 Agent 作用域 (scope)

摘自 glossary：

- **scope**：按 agent 划分的注册单位。一项贡献（工具、提示词段、变量、限制、监听器）要么是**全局的**（对所有 agent 可见），要么**带作用域**（归属于恰好一个 scope key）。只有两层、扁平结构：带作用域的注册**不会向下继承给 subagent**，父子关系用 lineage 数据表达（`parentSession`、持久的 `delegationDepth`、运行时 `subagentDepth`），从不影响可见性。
- **shadowing**：最具体者胜出。带作用域的工具 / 片段 / 变量只在该 scope 内替换同名全局项，这是按 agent 定制 persona 和工具变体的机制。
- **restriction**：`tools.restrict` 为单个 scope 过滤全局工具集合（多个 restriction 取交集）；**被过滤掉的工具既不出现在提示词中，也拒绝执行，与不存在的工具无法区分**。
- **setup window**：创建 agent 时的一个时隙，scope 和 agent 对象已存在，但会话尚未发布、`agent/created` 未触发、首次提示词未组装；setup 只做注册，从不驱动 agent。

> **💡理解**：Super-agent 原型里 Router 和 Worker 用不同 preset、Worker 拿不到"问人""改 profile"等工具，靠的就是 preset + restriction 这套机制。被过滤的工具对模型来说"根本不存在"，比在提示词里写"别用这个工具"可靠得多。

### 3.11 内置工具目录

摘自 `tool-catalog.zh.md` 的工具包映射（模型可见名称）。这份目录由脚本**真实启动每个工具插件**后读取 `ctx.tools.schemas()` 生成，因为 schema 无法完全静态分析（运行时展开的枚举、拼接的描述、配置决定的名称、MCP 原始 JSON Schema）。

| 类别 | 工具包 → 模型可见工具 |
|---|---|
| 文件 | `dsh-tool-fs` → `read`、`read_image`、`write`、`edit`、`apply_patch`；`dsh-tool-fs-search` → `glob`、`grep`；`dsh-tool-str-replace-editor` → `str_replace_editor` |
| Shell / 终端 | `dsh-tool-bash` → `bash`；`dsh-tool-pwsh` → `pwsh`；`*-persistent` 版本；`dsh-tool-terminal` → `terminal_open/send/read/signal/list/close`；`dsh-tool-jobs` → `job_output`、`job_list`、`job_kill` |
| 代码智能 | `dsh-tool-lsp` → `lsp`；`dsh-tool-executor-ptc` → `run_code` |
| 子 agent / 团队 | `dsh-tool-subagent` → `subagent`、`list_subagent_models`；`dsh-tool-subagent-control` → `send_message`、`list_agents`、`interrupt_agent`；实验 `dsh-experimental-tool-agent-team` → `spawn_teammate`、`team_task_create/get/list/update`、`wait_agent` 等；实验 `subagent_oar` |
| 长任务策略 | `dsh-tool-goal` → `create_goal`、`get_goal`、`update_goal`；`dsh-tool-ralph` → `ralph`；`dsh-tool-workflow` → `workflow`；`dsh-tool-schedule` → `schedule_create/edit/list/view` |
| 交互 | `dsh-tool-ask-user` → `ask_user_question`；`dsh-tool-present` → `present`；`dsh-plan-mode` → `exit_plan_mode`；`dsh-action-cards` → `propose_action`；`dsh-tool-todo` → `todo_write` |
| 会话检索 | `dsh-tool-session-query` → `session_search`、`session_trace`、`session_event_read/search/trace` |
| 其他 | `dsh-tool-web` → `web_fetch`、`web_search`；`dsh-tool-skill` → `skill`；`dsh-mcp-resources` → MCP 资源三件套；`dsh-plugin-manager` / `dsh-plugin-market`；`dsh-tool-cordis` → `cordis_inspect_list/query`；`dsh-tool-working-directory`；实验 `create_worktree`；Stagehand 浏览器工具 |

**Goal 与 Ralph**（摘自 glossary）：

- **目标 (Goal)**：附着在现有会话上的单个持久完成目标，阶段为 `active` / `paused` / `blocked` / `complete`，按修订号演进，有 Goal Round 上限。它是一种状态，不是调度器，会话日志仍是真源。**目标激活**是进程本地权限（`armed` / `disarmed`），有意不参与持久回放：恢复或 fork 后，只有经人类授权的恢复变更（`/goal` 或模型工具）才能重新开始自动工作。
- **Ralph 循环**：一次面向不可变目标的前台"全新 agent"工作流。每个 **Ralph Round** 是一个全新子会话，**不接收父会话或此前子会话的对话种子**；跨 Round 的状态靠共享工作区和一份有界的结构化 **Ralph 交接**（状态、摘要、证据、后续步骤、阻塞说明）传递。

> **💡理解**：Ralph 对应社区里流行的 "Ralph Wiggum loop"：与其让一个会话无限变长、反复压缩，不如每轮开一个全新 agent，把进度写进工作区和交接报告。这和 Super-agent 里"Worker 上下文隔离、Router 只看摘要"是同一思路：**用结构化交接代替无限增长的上下文**。

<br>

## 4. 第三层：Super-agent 设计长文

> 原文开头的两条声明：英文版是中文原文的机器翻译；**本文讨论的是架构设计思路，不是对 DeepSeek Harness 现有功能的介绍**。
> 原文结构：§1 Super-agent、§2 双层架构、§3 事件驱动、§4 全双工多渠道、§5 计算/状态/环境分离、§6 Watchdog、§7 初步实验、§8 开放问题、§9 结语。

### 4.1 §1 Super-agent：问题与定义

**背景**

- 过去一年，coding agent 等单任务 agent 证明了模型能用工具解决复杂问题，而且需要的人工持续监控和逐步干预越来越少；模型协调多个 agent 的能力也在变强，跨会话组织和调度工作变得可行。人参与底层执行越来越少，角色逐渐上移到**更高层的决策与签收 (sign-off)**。
- 更强的模型也催生了新应用：最近出现很多**个人 agent**，直接接入日常的消息、日历、邮件和工作平台，监听信息流并帮忙处理任务。作者指出，这种"常驻在线、事件驱动"的工作方式并不限于个人场景，在软件开发等商业工作里同样有价值。

**问题**

- 主流 agent 设计仍以**单个会话**为基本工作单元，会话之间相互隔离。用户并行跑多个任务时得手动切换会话、记住每个会话里有什么。**按作者经验，一个人通常只能同时跟进 3 到 5 个活跃会话**。结论：在这种设计下，**单个任务受限于模型能力，而多任务受限于人的注意力**。
- 给本地单会话 agent 加上 CLI、MCP 或外部连接器，确实能让它收到外部消息，但根本限制还在：仍是一个会阻塞在长操作上的同步会话；不同任务的上下文混在一起；本地设备一离线一切都停；维持这套东西运转要大量人工。

**定义**：要突破这些限制，就要把 agent 从任何单个、手动驱动的会话中解耦出来，让它成为**一个持续存在的实体**：跨会话保持上下文，同时处理多个复杂任务且互不干扰。作者把这种 agent 及其系统架构称为 **Super-agent**。

**内部实例**：他们内部基于 DeepSeek Harness 运行的 Super-agent 能在 IM 和 GitHub 之间延续同一个开发任务，还会监控公开群聊，**在没人 @ 它的情况下**自己找到根因并提 PR 修复。原文配图：<a href="https://pbs.twimg.com/media/HWB9aQxbsAI02Ht.jpg">图 1：同一任务在 IM 群聊和 GitHub PR review 之间延续</a>、<a href="https://pbs.twimg.com/media/HWB9kRhagAA9Qv_.jpg">图 2：未被 @ 时在公开群里发现 bug 并提 PR</a>。

**四个基本特征**

| 特征 | 原文含义 | 例子 |
|---|---|---|
| **连续 (Continuous)** | 用户在原本就在用的地方（IM、GitHub 评论）和它交互，始终感觉是**同一个** agent，永远不用想"X 会话的上下文里有没有 Y 信息" | 用户先在 IM 要方案 A，后来在 PR 评论里改成方案 B，agent 能理解这个变化并调整正在做的事 |
| **高效 (Productive)** | 响应快，能并行完成多个复杂任务且每个都高质量；**上一个命令或任务还没完成就能接下一条指令** | — |
| **主动 (Proactive)** | 不等逐步指令，监听环境事件，值得处理就处理，**否则保持安静** | — |
| **持久 (Persistent)** | 始终在线，直接连接用户真实的工作环境，长时间运行中能从异常自动恢复 | — |

**核心论断**：**没有任何单个 ReAct 循环能同时提供这四个特征，它们必须来自循环之外的系统架构。** 相似的设计约束往往导出相似的架构：当目标是"常驻在线、并发响应环境事件"的系统时，作者预计个人 agent 和软件开发这类商业 agent 最终会走向相似的设计。构建这样的系统也会遇到很多与计算机系统领域长期研究的问题相似的问题，所以那个领域的许多经典设计思想在这里同样适用。

**五个模块与四个特征的对应**：

| 模块 | 支撑的特征 |
|---|---|
| §2 双层架构 | 连续交互 + 高效并发 |
| §3 事件驱动 | 主动 |
| §4 全双工多渠道交互 | 让每次交互既连续又高效 |
| §5 计算/状态/环境分离 + §6 Watchdog | 跨设备持久在线、自行从异常恢复 |

原文总架构图：<a href="https://pbs.twimg.com/media/HWB910MbcAAF68B.jpg">图 3：Super-agent 整体架构，标号 [2]~[6] 对应 §2~§6</a>。

> **💡理解**：这一节最重要的判断是**瓶颈转移**：单任务时代的瓶颈是模型能力，多任务时代的瓶颈是人的注意力。所以 Super-agent 不是要让单个任务做得更好（§7 实验也明确不测这个），而是要**把"管理会话"这件事从人身上拿走**。全文的设计都可以用一个问题检验：这个设计是否减少了人需要盯的东西？

### 4.2 §2 双层架构

**核心权衡：连续性 vs 吞吐量**

| 方案 | 连续性 | 吞吐量 | 问题 |
|---|---|---|---|
| 所有事件路由到**一个会话** | 最好，它看到一切 | 差：ReAct 一次走一步，单会话速度有上限，任务耗时长；不同任务的中间细节交织在同一上下文里，降低结果质量 | — |
| 每个事件一个**新会话** | 差：任务不完全独立时，缺少共享的全局上下文，结果彼此割裂；多个会话分别回复用户，交互混乱 | 最好 | — |
| **两层：Router + Worker** | 长驻的 Router 层负责所有人类交互和全局协调，保持全局连续 | 上下文隔离的 Worker 层高并发执行任务 | 多花 token（§7） |

原文配图：<a href="https://pbs.twimg.com/media/HWB-BsLasAAi5Qe.jpg">图 4：Router 层负责所有路由和对外交互，多个隔离的 Worker Session 并发执行任务</a>。

```text
             人（飞书群 / GitHub / 邮件 / 日历 / CI ...）
                          │  Trigger 事件流
                          ▼
   ┌────────────────────────────────────────────┐
   │  Router 层（长驻全局会话，轻量模型即可）         │
   │  · 承接所有入口的人类交互，汇聚跨任务信息与决策    │
   │  · 全局序列点 / 协调点 / 记忆                   │
   │  · 只交互和派单，不执行任务（可以不给 Bash）       │
   │  · 唯一对外说话的角色                           │
   └──────┬──────────────┬──────────────┬───────┘
     派单 │ 状态汇报 ↑     │              │
          ▼              ▼              ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ Worker A  │  │ Worker B  │  │ Worker C  │   每个任务一个隔离会话
   │ 独立上下文 │  │ 全套工具   │  │ 最强模型   │   内部可再用 subagent / agent team
   └───────────┘  └───────────┘  └───────────┘
       不直接对人说话；推理过程和工具调用不进外部渠道；状态汇报给 Router
```

**会话应当是实现细节**：有了双层架构，系统接管了上下文管理和并发执行，用户很少需要考虑会话，始终在和同一个 Super-agent 打交道。作者认为，让用户手动管理会话，**很大程度上是从底层 LLM API 沿袭下来的设计习惯**；会话应当是系统的实现细节，而不是用户的心智负担。现有设计里会话同时承担两件事：管理模型上下文，以及作为用户的项目管理单元。产品和底层会话解耦之后，项目管理不必再跟着会话边界走（见 §8.2）。

两层是**逻辑上的职责划分**，每层都不必实现成单个会话，可以有自己的内部结构。

**§2.1 Router 层**

- 长驻的全局路由层，接收来自所有入口的人类交互。所有跨任务的信息和决策汇聚于此，所以它是系统的**全局序列点、协调点和记忆**。
- 为了不让这个单一全局节点成为性能瓶颈，Router **不自己执行任务**，只与人交互、分派任务，**甚至可以完全不配 Bash 工具**。
- **和 Agent Team 的区别**：结构上像"一个中心节点 + 多个执行节点"的 Agent Team，但关注点不同。Agent Team 通常把**一个**复杂任务拆成步骤、协调扮演不同角色的成员；步骤间往往紧耦合，Team Lead 要深入理解任务、规划子任务、频繁协调成员，推理负担很重。Super-agent 的 Router **只派单、只在高层追踪任务之间的关系，不拆解单个任务**。反过来，Agent Team 也可以当双层架构用，前提是提示词限制主会话不亲自执行、只做对外交互和派发独立任务，由成员各自独立完成。作者经验：**Router 的工作往往比预想的简单，轻量模型通常就能兼顾速度和质量**。
- Router 内部也不必是单个会话，部分逻辑甚至不需要 LLM。作者选了最直接的设计：**单个长驻 LLM 会话**，并发现它没有想象中那么容易成为吞吐瓶颈。原因是**批处理**：一次推理期间到达的环境事件（Trigger）会排队，在下一轮一起处理，额外开销很小。
  - 事件匀速到达时，Trigger 逐个到达 Router，每次都能很快响应。
  - 事件突发、开始在 Router 排队时，上一轮积压的 Trigger 自动合并，一起注入下一次 LLM 调用。因为 Router 只做轻量交互和派发，每一步通常是**长输入、短输出**；合并进同一次 API 调用的多个 Trigger 在 **Prefill 阶段并行计算**，所以单次请求耗时的增长比批大小慢得多。在一定范围内，**积压越多，摊到每个决策的推理时间越低**。上限取决于上下文长度，以及模型一轮内能准确处理多少个 Trigger。
- 结合订阅与透传（§3.2）和 Chain of Recall（§3.3），**单个长驻 Router 会话就能保持上下文连续，对他们目前的团队规模已经够用**（§7.2 有量化）。
- 规模再大时，Router 内部可以用多级过滤结构，或水平分片成多个会话。这又回到同一个连续性/吞吐量权衡：分片越细，吞吐越高，但分片间共享的记忆和上下文越少。**在全局记忆问题解决之前，这个权衡会一直存在。**

**§2.2 Worker 层**

- 每个 Worker 有隔离的上下文和全套执行工具，专注于单个任务。Worker 直接决定任务结果和代码质量，所以**通常需要最强的模型**。
- **Worker 从不直接与人交互**，推理过程和工具调用不进入外部渠道；每个 Worker 把状态汇报给 Router，由 Router 负责所有对用户的回复。这样既保留了并发和输出质量，又避免了多会话分别回复的混乱。
- **Worker 间通常不直接通信**。冲突往往出现在共享工作环境中（如同一个 PR），只要相关 Worker 都订阅了该环境的 Thread，它们就会收到相同的事件通知，有机会**自己发现并解决冲突**。少数需要跨任务协调的情况由 Router 转达。
- Worker 内部也不必是单个会话：复杂任务可以创建 subagent，或者本身就是一个 Agent Team，这对上层 Router 透明。
- 适用场景：代码协作中复现问题、审 PR、开发功能等独立任务可同时进行、互不阻塞；个人 agent 通常只有一个聊天页面，用户在里面同时聊多个话题时页面保持对话连续，后台为每个话题派一个 Worker 并行干活、起草回复。

> **💡理解**：
> - **Router ≈ 前台 / 调度员，Worker ≈ 工程师**。前台不写代码，但记得每个客户说过什么；工程师不接电话，只对前台汇报。
> - "Router 用轻模型、Worker 用强模型"是很实际的成本结构：Router 的输入长（积压事件 + 摘要）、输出短（几个派单决策），正好是 Prefill 重、Decode 轻的负载，批处理收益最大。
> - 和 Claude Code 等工具里的 subagent 对比：subagent 是主会话**自愿**调用的工具，用不用由模型当场判断；Super-agent 里 Router **根本不干活**，委派是架构强制的。§7 实验就是在验证这个区别。

### 4.3 §3 事件驱动的 Trigger

要让 Super-agent 主动，先要改变驱动系统的东西：主要输入不再是人一条条敲进会话的 prompt，而是**来自环境的半结构化事件**，例如群聊消息、邮件、日历提醒、代码评审意见、CI 结果。这些事件叫 **Trigger**。agent 监听并响应 Trigger，就不必像单会话 agent 那样每次运行前都等人的命令。

一个事件从到达到被处理分四步，层层递进：

```text
环境事件 ──① 过滤(§3.1)──> ② 路由(§3.2)──> ③ 组织上下文(§3.3)──> 模型处理
                                                   └─ ④ 提前行动(§3.4)：后台推测执行
```

**§3.1 事件过滤：快路径 + 慢路径**

接入真实群聊和代码平台后，系统面对的是**噪声大、触发率低**的事件流：绝大多数日常消息都不需要 agent 行动。让大模型从头判断每个事件，延迟和算力都吃不消。作者借鉴计算机系统里的 **Fast Path / Slow Path**：把大模型推理当作灵活但昂贵的慢路径；把频繁、定义清楚、稳定的判断写成确定性代码，作为低延迟、低成本的快路径。

| 阶段 | 做什么 |
|---|---|
| **程序化策略（Fast Path）** | 代码根据事件来源、群组、用户或 bot 白名单等确定性条件做快速准入。例如每个来源可设为"只响应显式请求"、"作为候选事件"或"忽略"。在调用任何模型之前就丢掉大部分无关消息 |
| **语义策略（Slow Path）** | 通过程序化检查的候选事件交给 LLM，根据上下文和**写成 prompt 的策略**决定是否行动。语义阶段同时是一个**自然语言扩展点**：人可以随时通过 prompt 开关策略或新增策略 |

原文配图：<a href="https://pbs.twimg.com/media/HWB-bbPaYAAwiP3.jpg">图 5：两级过滤，程序化策略先筛掉大部分无关消息，语义策略判断其余</a>。

**§3.2 事件路由：Thread、订阅与透传**

如果每个已分派任务的后续事件都还要经过 Router，单一全局节点负载会很重、响应变慢。所以快/慢路径的划分也用在路由上：**Router 只负责新任务的冷启动和跨任务协调，代码把后续事件直接送到拥有它的 Worker**。

- 真实工作流中，同一个 GitHub PR、IM 话题或邮件链上的连续事件构成一个事件流，叫 **Thread**（<a href="https://pbs.twimg.com/media/HWB-jgWbgAASHEr.jpg">图 6</a>）。
- Router 为某个 Thread 的第一个事件创建 Worker 时，该 Worker **直接订阅**这个 Thread。之后路由分两条路（<a href="https://pbs.twimg.com/media/HWB-t4aboAAc4z4.jpg">图 7</a>）：

| 路径 | 行为 |
|---|---|
| **已订阅（Fast Path）** | 代码把 Thread 里的后续事件**直接透传 (pass-through)** 给订阅的 Worker，Router 不需要再发一次工具调用转发。新事件超出当前任务范围时，Worker 可以退订，或把事件**交还**给 Router 重新分派 |
| **未订阅（Slow Path）** | Router 仍能看到所有事件以追踪全局状态，但**只评估和分派还没有订阅者的新事件** |

- **订阅是多对多的**：一个 Worker 可订阅多个相关 Thread，一个 Thread 可有多个订阅 Worker。多个 Worker 在同一工作环境或讨论中重叠时，共享 Thread 的每个新事件会同时透传给所有订阅者，它们因此有机会了解彼此在做什么、**无需点对点通信**就自行解决冲突。输出侧，Worker 仍把关键结论汇报给 Router，由 Router 负责对用户的所有回复并保持全局记忆同步。

**§3.3 上下文组织：Inline & Spill + Chain of Recall**

事件被过滤并路由到目标会话后，下一个问题是：往模型上下文窗口里放多少信息？输入不再是有限的人类 prompt，而是环境持续产生的半结构化事件，所以工具调用和事件注入的设计更重要。**注入太多**，窗口被填满，稀释模型注意力；**注入太少**，模型要反复调工具读上下文，增加执行步数。原文的说法是：浪费上下文窗口，就是浪费模型的智能。

作者把有限的上下文窗口**当作缓存**，借用编译优化和计算机系统的术语，上下文注入归结为两个基本操作：

- **Inline（内联）**：把很可能马上用到的信息直接放进当前窗口，省掉额外的工具调用往返。
- **Spill（溢出）**：超出预算或很少用到的完整信息留在窗口外，窗口里只留引用或摘要，模型需要时按引用读取。

结合计算机系统设计的经典思想**局部性**，可以逐步改进事件注入和工具调用：

1. **从最少信息开始**：一开始只给模型当前对象的 ID 和内容摘要。
2. **按局部性逐步 Inline**：内联模型很可能马上要用的内容，先是紧密相关对象的 ID，再加空间或时间上相邻的内容。例如内容的前 50 个字符、某条 IM 消息周围的消息和它回复的那条、同一代码评审讨论中相邻的评论。
3. **Spill 超限内容**：对不能完整内联的长内容设一个较小的长度上限，超出部分 Spill 到窗口外。简单做法是截断并留一个指向后续内容的指针，模型需要时调工具读剩下的。

**Router 自身的上下文问题**：很多后续事件直接透传给了 Worker，长时间运行又会积累大量历史任务，Router 不可能在窗口里保留所有任务的完整历史；但用户在另一个渠道提起旧任务时，Router 仍要恢复那个任务的上下文。把 Inline & Spill 一致地应用到系统中每一类对象上，就得到一种按需、多跳的检索机制：**Chain of Recall（链式召回）**（<a href="https://pbs.twimg.com/media/HWB-3mwa4AAd7HW.jpg">图 8</a>）。

- 系统给 Trigger、Thread、Worker Session 等对象分配唯一 ID；注入一个对象时，同时内联与之紧密相关对象的 ID。这些 ID 把对象连成一张**引用图**。
- 通过工具调用召回的每个对象也遵循 Inline & Spill，所以返回结果里内联了下一跳对象的摘要和 ID。模型从当前事件出发，可以沿引用**一跳一跳**召回上下文，需要多远就走多远。
- 例子：从当前消息所在的 Thread 出发 → 沿内联的 ID 找到相关的 Worker Session → 再找到与该 Worker 关联的另一个代码评审 Thread。

**§3.4 提前行动：推测执行**

借用计算机系统的**推测执行 (speculative execution)**：根据环境信号，agent 在还不知道结果会不会被用到时就在后台开工，用额外的后台算力换取人更少的等待和介入。有用的结果交付，其余丢弃。

- 个人场景：收到会议邀请邮件或待办消息时，agent 在后台检查日历冲突、收集背景信息、起草回复，用户只需过目并发送。
- 研发协作：他们在内部 DeepSeek Harness 用户群配置了一条 **"静默复现并修复"** 策略。有人报告问题或讨论改动方案时，agent 先在后台尝试复现并找修复方案；**只有确认问题存在且找到了简单修复时才交付结果，否则全程保持安静**（<a href="https://pbs.twimg.com/media/HWB_CSNbAAAq4Ur.jpg">图 9：群里还在讨论方案时，Super-agent 已在后台提了修复 PR</a>）。

> **💡理解**：§3 几乎是一张"计算机系统经典思想 → agent 设计"的映射表：
>
> | 系统思想 | Super-agent 中的对应 |
> |---|---|
> | Fast Path / Slow Path（网络协议栈、JIT） | 规则过滤 vs LLM 判断；订阅透传 vs Router 分派 |
> | 缓存 + 局部性 | 上下文窗口即缓存；按时空局部性 Inline |
> | 寄存器溢出 (spill)、函数内联 (inline) | 超出预算的内容 Spill，留指针 |
> | 指针 / 引用图遍历 | Chain of Recall 沿 ID 多跳召回 |
> | 发布-订阅 (pub/sub) | Thread 订阅、多对多透传 |
> | CPU 推测执行 | 后台先做，有用才交付 |
>
> Chain of Recall 本质上是用**结构化引用**代替向量检索：不靠语义相似度去猜哪段历史相关，而是让模型像点超链接一样顺着明确的关系去找。好处是可解释、不会召回不相关的东西；局限是只能找到"有引用连过来"的内容，没连上的旧信息就找不到，这也是 §8.1 说长期记忆仍是瓶颈的原因。

### 4.4 §4 全双工、多渠道交互

§2 在上下文层面保证了跨渠道连续性，但要让日常交互既连续又高效，还要解决两个问题：**交互如何进行**（耗时的后台操作不能阻塞前台对话）和**在哪里进行**（按渠道调整回复内容和详略）。

**§4.1 全双工**

- 经典 ReAct 循环里，模型发出同步的 `run()` 工具调用后，循环会阻塞到工具完成并返回最终结果，期间会话无法响应用户的任何新消息。要去掉这种阻塞，对话必须从**轮流说话的半双工**变成**全双工**（<a href="https://pbs.twimg.com/media/HWB_UbGbgAANrKk.jpg">图 10：经典 vs 异步 ReAct 循环，异步调用立即返回 Started，后台工具运行时会话仍能响应新消息</a>）。
- 做法：把耗时的命令和工具调用换成异步的 `start()` 和 `stop()`，形成**异步 ReAct 循环**。`start()` 立即返回 "Started"，循环不等工具；工具在后台完成后，系统把最终结果作为**一条新事件消息**送进会话。模型也可以根据上下文调用 `stop()` 结束运行过久的后台命令。
- 效果：前台对话和后台任务解耦，**新输入在当前推理步结束后就被处理**，而不是等后台工具跑完。
- **DeepSeek Harness 已实现这套异步任务机制，但默认未开启。**

**§4.2 多渠道**

人在群聊、邮件、GitHub 讨论里的沟通方式不同，agent 需要按渠道调整详略和回应方式。详细的推理过程和工具日志适合单用户终端，原样贴进群聊会让参与者信息过载。双层架构已经解决了一部分：每个任务冗长的推理和频繁的工具调用都留在 Worker 内部，不直接进外部渠道，Router 负责所有对外交互。此外，系统要接入用户日常依赖的工作环境，并**用 prompt 为每个渠道设定交互规范**：

| 渠道 | 规范 |
|---|---|
| **IM 等即时通讯** | 沟通要轻、尽量少打扰人。像人类同事一样，只在需要决策或签收时通知人，只问直接的问题，给简短结论。作者验证过的一条简单有效的硬规则：**每条回复上限 100 个词** |
| **GitHub、邮件、文档等异步渠道** | 交付结构化、信息密集的输出。例如创建或更新 PR 时，按项目规范写清完整的问题背景、改动说明和验证结果；个人场景下直接起草结构化的邮件和文档摘要 |

> **💡理解**：全双工的思路和操作系统的异步 I/O 一样：发起后立即返回，完成时以事件通知。Harness 里对应的是 `ctx.jobs` + `job_output` / `job_list` / `job_kill` 工具，原型的简化版做法是"**前台命令超过 5 秒自动转后台**"（§5.4）。"IM 回复不超过 100 词"看起来是小细节，却正是 §4.1 理解里那个检验标准的体现：每一条消息都在消耗人的注意力。

### 4.5 §5 计算、状态、执行环境三者分离

本地 agent 默认把三样东西绑在一台设备上：**运行 Agent Loop 的计算**、**必须持久化的会话状态**、**执行环境**（代码仓库、本地工具链）。设备休眠或离线 agent 就停，也够不到其他设备上的环境。为了让 Super-agent 持久在线并连接任务需要的真实工作环境，作者借鉴计算机系统的**存算分离**，分两步解耦。

**§5.1 计算与状态分离**：运行 Agent Loop 的计算可以是无状态的，只有会话状态需要持久化。所以第一层解耦是把计算放进**无状态、可弹性伸缩的云端容器**，会话和其他持久状态放进数据库。容器可以扩容、重启、替换，持久存储保证会话和任务的连续性。本地设备休眠或离线不再让 agent 停下，它持久在线于云端。

**§5.2 计算与执行环境分离**：

- 模型用到的大量上下文来自对代码仓库、本地文件、依赖服务等执行环境的工具调用。所以**广义上，上下文包括执行环境本身**。
- 两种常见做法都有问题：(1) 在云端起沙箱、拉代码、尝试重建并同步环境。但真实环境常含未提交的本地改动、私有凭据、对内网服务的依赖、特定硬件或工具链，全部同步代价高且容易遗漏。(2) 要求用户放弃本地环境，把工作迁到云上，对很多用户来说迁移成本太高。
- 既然难以复制和同步的是环境本身，就没必要把它和 Agent Loop 一起搬上云。第二层解耦：**Agent Loop 在云端，状态重的执行环境留在原地**。每个端点只需一个**轻量 daemon**，它**不保存任何会话状态**，只执行命令、返回结果；agent 按需从执行环境读取上下文。
- **与 Anthropic 的对比**：Anthropic 的 *Decoupling the brain from the hands*（<a href="https://www.anthropic.com/research/managed-agents">anthropic.com/research/managed-agents</a>）走的是同一条路：harness 保持无状态，会话日志存在外部，执行环境通过统一接口按需连接。区别在目标场景：Anthropic 那篇描述的是托管服务，主要讨论按需创建的沙箱和客户 VPC 内的资源；DeepSeek 首先要连接人们每天用的**个人机器**，这些机器上有未提交的改动和本地凭据，而且可能只是间歇在线。所以他们的设计支持 **BYOM（Bring Your Own Machine）**，把用户自己的开发机或个人电脑接到云端 agent 上，同时也支持云托管机器和按需创建的计算环境。
- 结果：agent 不再绑定一台机器，可以在执行设备之间切换，在任务需要的环境里运行命令、读取上下文；甚至能和间歇在线的设备协作：**用户的个人设备重新上线时，云端 agent 自动开始并完成需要在该设备上跑的任务**。

原文配图：<a href="https://pbs.twimg.com/media/HWCACygasAAvhLx.jpg">图 11：存算分离与执行环境解耦，云端 Agent Loop 与持久状态分离，经端点 daemon 按需访问不同执行环境</a>。

```text
                       云端
   ┌─────────────────────────────────────────────┐
   │  无状态容器 × N：跑 Agent Loop（可扩容/重启/替换） │
   │            │ 读写                              │
   │            ▼                                  │
   │  数据库：会话 + 持久状态                           │
   └────────────┬───────────────┬────────────────┘
                │ 下发命令 / 回传结果（按需）
      ┌─────────┼───────────┬───────────────┐
      ▼         ▼           ▼               ▼
  [个人笔记本]  [开发工作站]  [云托管机器]   [按需创建的沙箱]
  轻量 daemon：不存会话状态，只执行命令、返回结果
  未提交改动、私有凭据、内网依赖、特定硬件都留在原地
  设备离线 → 相关任务排队；重新上线 → 自动继续
```

> **💡理解**：这是数据库里"存算分离"（如 Snowflake、Aurora）的 agent 版本，再加一层"执行环境不动、只派命令过去"，类似 CI 系统的 self-hosted runner 或 VS Code Remote 的 server 端。核心洞察是"**广义上下文包括执行环境**"：与其把环境复制到 agent 身边，不如让 agent 远程去用环境。代价在 §8.4：云端 Loop 和所有端点之间形成了信任关系，Loop 一旦被攻破，所有端点都暴露。

### 4.6 §6 Watchdog：系统外的控制回路

- 持久在线不仅需要 §5 的存算分离，还要求系统在长期运行中**自动从异常恢复**。作者指出：一个持久在线的分布式系统，**如果只依赖内部逻辑，很难长期可靠**，需要一个独立于主系统的**外部控制回路 (Control Loop)**。
- agent 系统需要的不止传统的进程存活探针或监控脚本。很多异常不是崩溃：进程还在跑，但**模型陷入了无效循环，或偏离了目标**。所以长时间运行的 agent 系统除了监控基础设施存活，还要**监控任务进展**。
- 他们的外部控制回路是一个独立的 **Watchdog Session**，运行在主系统之外，**由定时消息驱动**。它处理 API 报错、状态恢复失败等基础设施故障，也**用模型判断每个 Worker 是否在实质性推进、是否偏离了总体目标**；及时修复异常，并对暂时性的外部故障持续复查直到解决。
- Watchdog 本身就是一个带全套工具的 agent 会话，所以维护者可以**用自然语言问它某件事为什么失败**，也可以让它在系统运行中修正主系统的行为，例如增删 prompt 或调整触发规则。

> **💡理解**：Watchdog 对应 Kubernetes 的 controller / reconcile loop 和 SRE 的外部探针，但多了一层"语义健康检查"：不只是"进程活着吗"，而是"这个 Worker 最近 30 分钟是在推进，还是在原地打转"。这类检查只有模型做得了。原型里有一个简化落地：**在你平时用的 profile 里建一个每小时运行的定时任务，让 agent 跑 `status` 并按退出码处理**（§5.6）。它放在被监控系统之外，被监控系统挂了它也还在，正符合"外部控制回路"的要求。

### 4.7 §7 初步实验

作者在两个层面对核心调度和路由设计做了初步实验：**整个系统**（端到端比较双层架构与单会话设计，并测订阅透传的作用）和**单独的 Router 层**（测批处理）。实验用合成负载和简化环境，部分设置可以更严谨，**结果仅供参考 (indicative only)**。

**§7.1 消融实验**

- **设置**：四种端到端设计同时运行，由一条群聊 + GitHub CI 消息流驱动。每次运行约 **300 条消息**，其中约 **20 条是工作请求**，其余是闲聊、追问、取消和 CI 消息。**编码工作用一个只消耗时间的工具模拟**，所以测的是系统的调度和协调能力，**不是执行单个任务的能力**。每种设计跑 **5 次**。
- **四种设计**：Super-agent；去掉透传的 Super-agent；两个"单会话 + subagent 工具"版本，区别只在 prompt：
  - **中性 prompt**：大意是"合适时为一项工作使用 subagent"；
  - **优先委派 prompt**：大意是"群里每条消息都会到达你，你自己做的工作会不断被新消息打断，所以优先把每项工作交给它自己的 subagent，只自己做小事"。

| 设计 | 任务完成时间 p50 / p95（分钟） | 每次运行 cache-read tokens（所有会话合计） |
|---|---|---|
| **Super-agent** | **12.4 / 15.7** | 23.2M |
| Super-agent（无透传） | 12.6 / 15.5 | 31.8M |
| 单会话 + subagent 工具，中性 prompt | 20.4 / 29.5 | 6.1M |
| 单会话 + subagent 工具，优先委派 prompt | 13.8 / 29.0 | 10.8M |

注：任务完成时间从工作请求发出算到结果消息发出。原文配图：<a href="https://pbs.twimg.com/media/HWCAXLOaoAELExc.jpg">图 12：四种设计下工作请求的累计到达与累计完成（5 次均值；消息间隔和工作时长按 1/4 缩放，模型调用时间未缩放）</a>。

原文结论：

1. Super-agent p50 12.4 分钟、p95 15.7 分钟，**明显快于**单会话 + subagent 工具。
2. 单会话慢，是因为在这个实验里它**高度依赖 prompt**：主会话常常决定自己干活，导致完成时间长且不稳定。
   - **中性 prompt：5 次运行中模型一次都没调用 subagent**，所有工作都在主会话里跑，不断被新消息打断，p95 达到 29.5 分钟。
   - **优先委派 prompt：5 次里有 4 次委派了**，完成时间有所下降，但整体仍长于 Super-agent。**剩下那 1 次，模型判断工作"很小"就自己做了**，造成了图中紫色的长尾。
   - 单会话只能靠 prompt 工程和模型的临场判断决定是否委派；**双层架构在系统层面缓解了这个问题：Router 不执行工作，每个 Worker 只收到它订阅的 Thread 的消息**。
3. 因为维护多个独立会话，Super-agent 每次运行的 cache-read tokens 比单会话多，**用 token 换取更短、更稳定的完成时间**。
4. 去掉透传后完成时间和 Super-agent 差不多，但 **Router 每次运行多做约 30% 的决策**，cache-read tokens 从 23.2M 增到 31.8M，**多约 37%**。所以在这个负载下，**透传并不让任务完成得更快或更好，但能大幅削减 Router 的 token 用量和推理开销**。

**§7.2 Router 层批处理**

- **设置**：单独压测 Router。Router 调用真实模型，Worker 用 mock 模型。把需要决策的 Trigger 到达率从约 **1 条/分钟逐步提高到约 40 条/分钟**，在每个到达率下测平均批大小、单次请求耗时和摊到每个决策的推理时间。注意：一个决策可能跨多次 LLM 请求（Router 派单前可能多步推理或调工具召回上下文），所以每决策时间可以大于每请求时间。

| 指标 | ~1 条/分钟 | ~40 条/分钟 |
|---|---|---|
| 平均批大小 | 1.0 | 2.8 |
| 单次请求耗时 | 2.5 s | 3.9 s |
| 摊到每个决策的推理时间 | 6.4 s | **1.6 s** |

原文配图：<a href="https://pbs.twimg.com/media/HWCEDgNbIAAzqbX.jpg">图 13：不同 Trigger 到达率下 Router 的批大小、单次请求耗时与每决策摊销推理时间</a>。

- 结论：单次请求耗时的增长慢于批大小，所以每决策摊销时间下降。外部事件突发时，Router 通过自动合并积压事件提高了有效吞吐。这也解释了 §7.1：去掉透传提高了 Router 的峰值负载，但单个 Router 仍没到吞吐上限。**在他们目前的团队规模下，没必要以牺牲全局上下文连续性为代价过早地给 Router 分片。**

> **💡理解**：
> - 媒体标题里的"快 39%"即 p50 从 20.4 降到 12.4 分钟。但更关键的是 **p95**：优先委派 prompt 的 p50（13.8）已经接近 Super-agent，p95（29.0）却几乎没改善。说明靠 prompt 劝模型委派"平均还行，但压不住长尾"，**架构强制分工才能压住长尾**。
> - 成本：Super-agent 的 cache-read tokens 约为单会话方案的 2~4 倍（23.2M vs 6.1M / 10.8M）。注意是 cache-read，单价远低于普通输入 token，这也是这种设计在经济上可行的前提。
> - 局限要说清：合成负载、5 次运行、编码用"睡眠工具"模拟、单一团队规模。它证明了**调度机制可行**，没证明**代码质量更好**。

### 4.8 §8 开放问题

§2~§6 的五个模块构成了 Super-agent 的基本架构，§7 的实验表明其核心调度和路由机制在多任务并发时可行。系统之所以能保持相对简单，很大程度上是因为**现阶段同时简化了工程设计和所支持的场景**。在这些简化之外，四个方面的问题仍然开放：

**§8.1 双层架构如何扩展**

- 到目前为止，一个简单组合以很低的工程成本覆盖了他们大部分日常任务：**单个长驻 Router 会话 + 订阅与透传 + Worker 之间不直接通信**。
- 一旦任务规模更大、耦合更紧，超出单个 Router 能处理的范围，每层的合适结构就还没定论：何时、以多细的粒度给 Router 分片？何时让 Worker 直接通信、用什么协作拓扑？都需要在更多场景中验证。
- **系统与模型协同设计**也值得探索：专门为 Router 层工作（判断事件、批量派单、召回上下文）训练模型，可能进一步提高 Router 的吞吐和决策质量。
- 记忆也限制扩展：长驻 Router 加上引用图上的 Chain of Recall，主要解决的是近期任务和会话历史的按需检索；**系统运行几个月甚至更久后，长期记忆会成为主要瓶颈**。

**§8.2 如何与"没有会话"的 agent 协作**

- 双层架构对用户隐藏了多会话管理，会话不再是产品设计的主要关注点。但底层执行会话不再直接对应界面中的对话后，新问题出现了：**如何向用户呈现任务，什么样的交互和项目管理方式才自然？**
- 作者认为聊天窗口成为交互中心，部分是**路径依赖**。更多交互可能迁移到**领域专用界面**，比如他们的 Super-agent 实例直接在 GitHub Issues 和 PR 里工作。界面里也可以保留会话，因为用户习惯了，但**用户看到的一个对话不必对应底层一个会话**。
- 前面各节主要讨论任务开始和签收时的人机交互。但在模型快速进步的过渡期，很多任务中途仍需要人。值得探索的是：产品如何让用户在关键细节上逐步干预，又不丢掉后台自主工作的效率，例如在后台执行的任何时刻介入、接管或与 agent 实时协作。
- 总体上，用户需要的项目管理和交互方式高度取决于其工作领域和个人风格，最终设计很可能高度定制化。DeepSeek Harness 的开放架构让这种定制更容易支持。

**§8.3 如何划定上下文边界、通过 A2A 协作**

- 到目前为止讨论的是单个 Super-agent 实例，并假设其范围内的所有信息都汇聚在一个全局上下文里。使用范围从个人和小团队扩展到更大组织时，**决定如何部署 Super-agent，实质上就是决定哪些上下文共享、哪些隔离**：需要共享上下文的部分共用一个实例，需要隔离的各用各的。
- 组织内或跨组织存在多个隔离实例后，跨边界协作就是 **A2A（agent-to-agent）通信**，例如每个成员的个人 agent 和团队共享的 Super-agent 之间。这里的 A2A 泛指 agent 间通信，不指某个具体协议。**不共享底层上下文、各自代表不同权限边界的 agent 之间如何安全可靠地通信协作**，仍待解决。

**§8.4 如何保证安全与隐私**

- §3 的事件驱动设计让更广泛的环境能驱动 agent，也让 agent **暴露在更多不可信输入之下**。安全和隐私处理不好，Super-agent 就只能在非常小的可信范围内工作。
- 例如 §5 的设计把 Agent Loop 放在云端、在执行设备上装轻量 daemon，**隐含假设云端与端点互相信任**。但云端 Agent Loop 持续接收外部事件，而端点设备上有私有代码、敏感数据和本地凭据，把两者连起来带来显著的安全与隐私风险：**如果服务端入侵或 prompt injection 攻破了 Agent Loop，所有与之相连的端点设备都可能被接管**。
- **即使云端从未被攻破，凭据也可能泄露**：只要端点上的凭据能进入模型上下文，正常执行就可能把它暴露出去。所以 **agent 需要一种"使用凭据但永远看不到凭据"的方式**。要让这种架构更广泛地使用，就必须在保住"从多种环境召回上下文"能力的同时解决这些安全隐私问题。

> **💡理解**：§8.4 和 Harness 已有的设计能对上：`ctx.credentials` seam "配置只携带机密的引用，提供方拥有实际值"就是"使用凭据但看不见凭据"的一种基础形态。原型里的 `gh-agent` / `lark-bot` 包装脚本 + setup 清除名字含 KEY/PASSWORD/SECRET/TOKEN 的环境变量，是另一种土办法。但官方也明说这些"建立的是默认值，不是安全边界"（§5.4）。

### 4.9 §9 结语

- 作者强调这套设计**没有提出新理论**，而是把计算机系统的经典工程思想用在 agent 系统构建上。具体实现会随技术发展而变，但背后的系统设计思想会更长久地有用。
- 他们内部基于这些思想和 DeepSeek Harness 构建的 Super-agent 实例**已经在持续运行**，每天帮忙复现并修复问题、回应代码评审、开发 DeepSeek Harness 本身。
- 回到开篇：大模型在单个任务上需要的人工干预越来越少，进一步提升生产力的瓶颈正在转向人的注意力。双层架构、事件驱动、异步多渠道交互、存算分离和外部 Watchdog 合起来，让系统接管了管理会话、同步上下文这类繁琐工作，**人可以进一步上移，从执行细节转向更高层的决策与签收**。
- 这样构建的 agent 不再只是一个等待指令的聊天窗口，而成为一个常驻在人们所用数字环境中的**自主协作者**：对个人是跨设备帮忙处理日常事务的个人 agent；对团队是研发流程的一部分，接任务、推进任务。作者相信随着模型和系统架构协同演进，这种更自主的工作方式可能成为人机协作的常态。

**作者与致谢**：本文基于 DeepSeek Harness 团队的共同工作，**Yujia Qiao 实现了系统并撰写了文章**；Huanqi Cao、Zui Chen、Xieting Chu、Tianyi Cui、Shiyuan Feng、Yudong Han、Kexin Liu、Shichun Liu、Gesi Luo、Yifan Shi、Tongrui Xiong、Hanxiang Xu、C.C. Yu、Ziya Zhang、F. Zhou 等参与了评审讨论并给出反馈。文末注明：文中部分设计已在 dsh-superagent-prototype 中实现，该插件**只是验证技术路线的原型，不是 DeepSeek 正式产品**。

<br>

## 5. 开源原型：dsh-superagent-prototype

> 本节摘译自原型仓库 README、SETUP.md、cordis.patch.yml 和 docs/000~008（MIT）。原型文档要求"按编号顺序读，每篇只建立在前面几篇之上"，本节按同样顺序整理。

### 5.0 定位（README）

- **研究原型，仅供参考，不是 DeepSeek 正式产品。** 代码源于团队日常开发工作，**大量工程由 coding agent 完成**；应视为研究代码而非生产就绪软件，可能仍有 bug，编码风格和工程实践可能不一致，使用前请审阅并改造。
- 为公开发布做过改编，所以**与内部版本、与长文所述实现都不完全相同**；**长文里讨论的部分设计是未来开发方向，原型中没有实现**。
- 功能：面向**单个 GitHub 仓库**的主动式 coding agent，打包为 DeepSeek Harness 的一个 bundle；监听 GitHub，可选监听飞书。被允许的人请求工作时，它开一个工作项，在自己的 agent 会话和目录里干活，在原对话里答复（例如提一个 PR）。Web 面板显示需要人处理的事。
- 开发：`pnpm check` 跑类型检查、测试和构建。
- 仓库结构：`docs`、`examples`、`locale`、`prompts`、`scripts`、`skills`、`src`、`tests`，以及 `cordis.patch.yml`、`SETUP.md` 等。

### 5.1 000 哲学

核心原则（原文加粗）：

> **Take as little human attention as possible, and give as much value as possible.**
> 占用尽可能少的人类注意力，提供尽可能多的价值。

项目的每一部分都按它消耗的注意力来评判：一条消息、一个通知、一个问题、一项设置、一个面板元素、一段文档，都在消耗某人的注意力，**每一项都必须给出比这更多的价值，否则就删掉**。

**对 agent 的要求**

- **只在被请求、或团队写好的规则允许时行动。** 请求只有在直接对 agent 说、且来自团队允许的人（在白名单上，或在团队开放请求的对话里）时才算数。没人要求的工作需要团队写的规则，而且**从"只记录 agent 本会做什么"的模式开始**。
- **只在有人必须行动或能用上结果时说话。** 确认一次请求并说明要做什么；干活时保持安静；结果就绪、需要决定、或做不完时再说话。消息要短，发到请求所在的对话。
- **一次只问一个具体问题**，用结构化形式，让问题真正到达人，而不是埋在长报告里。
- **按影响停止，而不是按工作量停止。** 局部改动做到可评审的结果为止；需要设计决策的改动带着一个清晰的问题停下。
- **如实报告。** 草稿标为未完成，失败就报失败，**绝不绕过限制来避免提问**。

**对设计的要求**

- **少概念**：只有领域概念（如 Source）和它们的具体实现（如 GitHub、飞书）。业务需求要么成为某个概念的属性，要么成为 agent 遵循的一段 prompt，**不为它新增概念**。
- **少设置**：一个值适合几乎所有团队时就用默认值代替设置；团队用文字描述的行为是 prompt 文本，不是设置项。
- **面板只显示需要人的东西**：先列等人的事，健康状态不占地方。
- **文档按顺序读**。

**对运营者的责任要求**：agent 以团队每天看到的账号评论、提 PR、发消息，每条消息都在消耗同事的注意力，运营者要负责。**从窄处开始**（短白名单，主动工作只记录不执行，确认 agent 本会做的事看起来对了再放开）；**为更少噪声调优**（agent 吵就把 prompt 改短改严，或暂停它，不要再加消息来解释噪声）；**不要用它刷量**（批量 PR、群发消息、无意义的忙碌对团队的代价大于收益）；**为它说的话负责**（读 agent 以你的名义发的东西，错了就改 prompt）。

目标：**在对的时间给对的帮助，其他时间什么都不做。**

> **💡理解**：这份哲学可以看作长文 §4.2 渠道规范的完整版。很多主动式 agent 失败不是因为不会干活，而是因为**太吵**。"从观察模式开始"相当于主动行为的灰度发布。

### 5.2 001 领域模型

**流程**：工作从 Trigger 流向 Eval 再到执行。Source 产生 Trigger；Trigger 作为上下文扇出给订阅了其 Thread 的 Ticket。经此到达某个 Ticket 的人类消息归该 Ticket 处理；其他 Trigger 进入 Eval，由 Eval 判断是否在请求新工作。Eval 接受时创建一个带 Mandate 的 Ticket；Assignment 把 Ticket 交给 Executor；结果经 Channel 回到 Source。

```text
Source ──Trigger──> Thread ──批次──> Eval ──accept + Mandate / stop──> Ticket ──Assignment──> Executor
                      │                 └─ ignore / unavailable ──> 记录          │
                      └──Subscription 扇出（作为上下文）──────────> Ticket         │
Policy 变更 ──notice──> Ticket                                  Ticket ──Channel──> Source（回复）
```

**概念表**（Source、Thread、Eval、Mandate、Subscription 是核心领域，其余起支撑作用）：

| 概念 | 中文 | 定义 | 保持的规则 | 为什么存在 |
|---|---|---|---|---|
| Source | 来源 | 事件流。Source kind 是可复用实现，Source instance 是一份配置（是否运行、监听什么） | 持有读位置和自己的 bot 身份；定义什么算事件（包括只在一段时间后才成立的条件）；在进入领域前丢掉 bot 自己的消息和监听范围外的 Trigger；把每个 Trigger 翻译成它所属的 Thread 和回复用的 Channel 地址 | 重启不能重放事件 |
| Trigger | 触发 | 一个不可变事件，带 Source 自己的元数据（如发送者） | 属于一个或多个 Thread；同一个 Trigger 被接受两次只建一个 Ticket | 审计和幂等的单位 |
| Thread | 线程 | 按时间排序的 Trigger；由 Source 定义哪些 Trigger 组成一个 Thread | 同一时间最多一个 Eval；可以有多个 Ticket；持有订阅、运行中的 Eval 和决定历史 | 给 Eval 一致的视图和上下文 |
| Correlation | 关联 | 同一个 Trigger 所属的多个 Thread 之间的关系 | **只增加上下文，从不路由** | 让 Eval 追踪相关证据 |
| Channel | 渠道 | Thread 的写侧，地址由 Source 定义 | 一个 Thread 可以有多个 Channel；一个 Ticket 持有一组 | 确认、回复、PR、已解决的 review 都是写操作 |
| Policy | 策略 | 带版本的评估规则，加上每个 Source 的那部分（谁能经该 Source 发起工作，用该 Source 自己的身份；名单如何同步；哪些 bot 能产生候选） | 版本覆盖规则、接单模式和各 Source 部分；影响在途 Ticket 的变更**会通知其 Assignment，但不撤销 Mandate**；**不同 Source 上的身份永不互相映射** | 规则变了，运行中的 Eval 必须丢弃 |
| Eval | 评估 | 对一批 Trigger 的判断，分两层：**Gate（准入）** 和 **Assessment（研判）** | 输入：Trigger 批次、Thread 快照、该 Thread 上所有 Ticket（包括已结束的）、当前 Policy；输出：一个 Decision；**不持有状态** | 决定新工作是否开始、继续或停止 |
| Mandate | 授权 | 接单授予的东西：Policy 版本、范围、强度、匹配规则、发送者在该 Source 上的身份、Source 如何收到 Trigger | **不可变，扩大范围意味着新 Ticket**；Ticket 结束不会结束 Mandate，只有人的 stop 撤回它，Policy 不再支持时撤销它；已结束的 Ticket 恢复前要对当前 Policy 重新验证 | 说明为什么可以做这项工作、能做到多远 |
| Ticket | 工单 | 一个已授权的工作单元 | **只有 Eval 能创建**；引用 Mandate，持有 Channel 和 Input；open → ended，人在其 Thread 上发消息可恢复为 open（除非人叫停过）；**自己定义状态，不跟随 Executor 的状态** | 拥有执行及其报告 |
| Assignment | 派工 | Ticket 与一个 Executor 实例的绑定 | 一个 Ticket 随时间可有多个；pending → running → ended，独立于 Ticket；只有它的最终报告能从 Executor 一侧结束 Ticket | Executor 在领域外，但谁来执行必须在领域内 |
| Subscription | 订阅 | Ticket 与 Thread 的多对多关系，由 Thread 持有 | 匹配的 Trigger 扇出给每个活跃的订阅 Ticket，**不检查发送者**；经此到达 Ticket 的人类消息不走 Eval；仅为上下文跟随的 Thread 不提供授权、不能恢复 Ticket；已结束 Ticket 保留订阅（包括已关闭/合并的 PR），人的消息可在同一 Mandate 下恢复它，bot 和 CI 消息不会；被人叫停的 Ticket 什么都收不到；主会话只有在消息**明确指向**某 Ticket 时才把 Thread 附加给它 | 后续讨论无需再 @ agent 也能到达 Executor |

几条值得注意的设计（原文）：

- **"人"不是领域概念。** Trigger 只携带 Source 给的发送者标识；白名单是某个 Source 的 Policy 部分，用该 Source 的身份表示；同一个人在多个 Source 上有账号，这件事是 agent 需要时读的 prompt 文本（例如一个 `identities` skill）。
- **Executor（含其会话和目录）在领域之外**，经 Assignment 接入。
- **聚合**：Source 持有读位置；Thread 持有订阅、运行中的 Eval（串行运行，期间到达的 Trigger 合并到下一批）和决定历史；Ticket 持有 Assignment、Channel 和 Input，按 id 引用 Decision。

**被否决的概念**（原文表格，体现设计取舍）：

| 被否决的 | 理由 |
|---|---|
| 把 Principal / Membership 做成实体，或跨 Source 映射身份 | 人是 Trigger 元数据；白名单是 Source 的 Policy 部分；跨 Source 映射是 prompt 文本 |
| Eval 调度自己的延时 Trigger | 需要等待的条件是一个普通的 Source，所以 Eval 永远不反向依赖 Source |
| Eval 的"等待中"或"已撤销"状态 | 等待属于 Source；超时的 Eval 把 Trigger 重新排队，或以 `unavailable` 结束 |
| 每个 Thread 最多一个活跃 Ticket | 一个 Thread 可以承载多项独立工作；去重靠评估时参考 Ticket 列表 |
| 一个"追加到 Ticket"的 Decision | 追加就是订阅扇出，不需要 Eval |
| 对订阅做准入检查 | 订阅覆盖整个 Thread；**Mandate 把授权和上下文分开** |
| 会路由的 Correlation | 路由靠订阅，Correlation 只加上下文 |
| Ticket 状态跟随 DSH 会话 | Ticket 可以被移交、恢复或由多个 Executor 执行，所以自己定义状态 |
| Executor 宣称"有人叫我停" | **只有 Eval 决定结束的强度** |
| Policy 变更时取消运行中的 Ticket | 取消会留下做了一半的工作；改为记录变更并通知 |
| Executor 会话失败或阻塞时锁住它所有工具 | 会把直接在会话里工作的人、以及恢复后的报告都拒之门外；暂停自动输入就够了 |
| 从报告的自由文本推断"在等人" | 分类需要结构化事实；Executor 用结构化字段提问 |

**可复用部分 vs 部署相关部分**：上述概念和规则可复用；Source 实现、Policy 内容、Executor 和 Channel 实现属于具体部署。本仓库中：

| 部分 | 本仓库实现 |
|---|---|
| Source | 飞书：实时事件、历史恢复、订阅话题轮询。GitHub：通知、订阅的 Issue 和 PR、CI 变化、PR 合并或关闭；开启时还包括仓库里每个新 Issue、PR、评论和默认分支 CI |
| Policy | 评估规则，以及每个 Source 的白名单、同步、bot 名单和准入表 |
| Executor | 每个 Ticket 一个 DSH Executor 会话和一个目录 |
| Channel | 飞书话题回复、GitHub 评论和 review 线程回复 |

**取值速查**：Decision = accept 接单 / stop 停止 / ignore 忽略 / unavailable 未完成；Strength = explicit 明确 / inferred 推断；Assignment = pending / running / ended；Report = progress 进展 / waiting 等待 / ready 就绪 / blocked 受阻 / failed 失败 / dropped 无需处理。代码和 prompt 里，主会话叫 `orchestrator`，执行器会话叫 `worker`。

**长文术语 ↔ 原型术语**

| 长文 | 原型 |
|---|---|
| Router | 主会话 (Main Session, `orchestrator`) |
| Worker | 执行器会话 (Executor Session, `worker`)，每个 Ticket 一个 |
| Trigger / Thread | Trigger / Thread（同名） |
| 程序化策略 (Fast Path) | Eval 的 Gate（准入）+ Source 自己的过滤 |
| 语义策略 (Slow Path) | Eval 的 Assessment（研判）+ Policy 里的 rules（prompt） |
| 订阅与透传 | Subscription + Fan-out（扇出） |
| 交还给 Router | `orchestrator_trigger_handback` |
| Worker 向 Router 汇报 | Report（progress / waiting / ready / blocked / failed / dropped） |

> **💡理解**：原型的领域模型比长文严谨得多，核心是 **Mandate（授权）与 Subscription（上下文）分离**：agent 可以读到 Thread 上任何人（包括 bot、CI、陌生人）的消息，但这些消息只是上下文，改不了 agent 被允许做什么；权限只来自 Eval 接单时生成的、不可变的 Mandate。这是对 prompt injection 的**结构性防御**。"被否决的概念"表也很值得看，几乎每一条都是"能用已有概念表达，就不新增概念"。

### 5.3 002 授权

总原则（原文）：**宿主强制执行白名单、规则和 Mandate。消息、评论和引用的文档只是决策的输入，不能改变这些权限。**

**Trigger 的两条路径**

1. **Eval：决定是否开始工作。** Source 记录消息是否在对 agent 说话：`mention`（@ 了 bot 或 agent 账号）或 `direct`（私信）。Gate 检查发送者和该 Source 的 Policy：白名单上的人对 agent 说话 = 显式请求；在已有 Ticket 的 Thread 上的跟进按下文规则处理；其他人类消息看接收范围的准入表；**bot 名单上的 bot 只能产生推断候选，永远不能是显式请求**；其他 bot 不准入。
2. **订阅：继续已有工作。** 订阅 Thread 上的每条消息（包括 bot、CI）都作为上下文到达订阅 Ticket。**这种投递绕过 Gate，也不授予任何新权限**，每个 Ticket 仍受 Mandate 限制。Ticket 自己 Thread 上的人类消息由该 Ticket 的 Executor 决定怎么处理（不走 Eval、没有 Decision）：要么在 Ticket 范围内处理，要么交还给别处路由。此时主会话只能应人的要求停止工作，accept / attach / ignore / requeue 都会被拒绝。

**交还与释放**：Executor 对超出 Ticket 范围或属于另一 Ticket 的消息**交还一次**（Ticket 保留订阅）；只有在**没有任何 Ticket 还拥有它**之后，消息才进入新的 Eval，候选会带上 Executor 的理由。Ticket 结束时，它的 Executor 没读、也没处理的跟进会被释放，回到 Eval。被人叫停的 Ticket 不会把叫停消息再送回 Eval。**仅为上下文订阅**的 Thread 上的消息不提供授权或跟进强度，也不能恢复已结束的 Ticket。

**强度 (Strength)**

| 强度 | 情形 |
|---|---|
| explicit | 白名单上的人对 agent 说话，或在已有 Ticket 的 Thread 上跟进；准入表里设为"显式请求"的格子 |
| inferred | 不在白名单的人在有 Ticket 的 Thread 上跟进（除非其准入格是 explicit）；准入表里设为"推断候选"的格子；bot 名单上的 bot 对 agent 说话、或在接收全部消息的范围里发言 |
| 不准入 | 其他 bot 的消息；准入表忽略的一切；接单模式为 `off` 时的所有推断情形 |

**准入表 (Admission)**：每个 Source 的 Policy 有一张表，行是接收范围（飞书：每个配置的群、其他 @ 了 bot 的群、私信；GitHub：新 Issue、新 PR、评论、默认分支 CI），每行最多三格：

| 格子 | 可选 | 默认 |
|---|---|---|
| 在白名单、没对 agent 说话 | inferred、ignore | inferred |
| 不在白名单、对 agent 说话 | explicit、inferred、ignore | ignore |
| 不在白名单、没对 agent 说话 | inferred、ignore | ignore |

没对 agent 说话的格子只在 Source 接收全部消息的范围里可用（如飞书群设为"全部消息"、GitHub 开了 receive-all）。CI 结果没有发送者，只有最后一格。把"不在白名单、对 agent 说话"设为 explicit，就能让该范围内任何人 @ agent 请求工作，适合专门的支持群。用普通账号的自动评审者对宿主来说算"人"，团队 prompt 必须标明这类账号。

**路由候选**：主会话用 `orchestrator_route` 路由显式候选，一次调用可含多个 accept / attach / stop / ignore 决定，**各项独立处理**，一项被拒不影响其他项。推断候选一般要完整评估；唯一捷径是卡片明显不涉及任何启用规则（如闲聊）时可直接忽略，Decision 记 `ignoredFromCard`。**接受推断候选永远需要评估**，直接 accept 会拒绝 Gate 没判为 explicit 的 Trigger。

**接单模式**

| 模式 | 效果 |
|---|---|
| `off` | 推断的东西一律不成为候选（未对 agent 说话的消息、bot 消息、不在白名单的人的跟进） |
| `observe`（**默认**） | 推断候选照常评估；正面评估被记录为标着 **"would accept"** 的 ignore，**什么都不接** |
| `auto` | 推断候选的正面评估会接单 |

显式请求在所有模式下都有效。建议从 `observe` 开始，复查记录的 Decision 后再切到 `auto`。每次评估对每条启用规则恰好判断一次（无论是否匹配）；主动接单至少要匹配一条规则。规则描述一类主动工作，Ticket 的 Mandate 记录匹配的规则，Executor 收到规则名和指令；对该 Ticket，规则中的流程和沟通要求覆盖默认值（例如"静默"规则可以不发确认，"评审"规则可以指定合并前需要的评审人）。

**Policy 版本**：规则、接单模式和各 Source 的 Policy 构成 Policy 版本，每个 Mandate 记录自己被接受时的版本。版本变化会**丢弃运行中的 Eval 并重新排队其 Trigger**，并给每个受影响的活跃 Ticket 发一个 policy-changed Input，说明是匹配的规则变了或不再适用、请求者不再能发起工作，还是不再接受主动工作；不受影响的 Ticket 不收通知；**不立即撤销**活跃 Ticket 的 Mandate。注意 Source 配置（读哪些群、接收全部还是只接收 @）与 Policy 是分开的。

**恢复与停止**：只有三种操作改变 Mandate：**accept 创建它，人的 stop 撤回它，重新验证失败撤销它**。已结束的 Ticket 保留 Mandate，人在其 Thread 上发消息就能在同一 Mandate 下恢复（不需要新的 Eval），bot 和 CI 唤不醒它。恢复前的重新验证：Source 必须仍在监听原 Thread；显式跟进型 Mandate 直接通过；推断型需要 `auto` 模式且匹配规则仍启用；准入表型要在今天的表和名单下仍至少有原来的强度；bot 型要求 bot 仍在名单上；其他要求发送者仍在白名单。

- **人在 Thread 里说"停"是一个普通 Trigger**，由 Eval 决定 stop；**Executor 不能宣称有人叫它停**。
- 面板里的 Stop 是一条命令，直接作用于 Ticket，记为手动停止。
- 等待条件属于 Source，例如"N 小时没人认领"应该是一个自己等待并复查的 Source，Eval 从不等待。

**Eval 没完成时**：时限从候选到达主会话时开始算（等待投递的时间不算，未投递的候选永不过期）。超时的 Eval 把 Trigger 重新排队，**连续重试两次后第三次超时记为 `unavailable`**；评估失败也记 `unavailable`。面板只在这个 unavailable 决定不到 24 小时、仍是其 Thread 上最新的决定、没被忽略、且 Thread 不在排队或评估中时才请人关注。

### 5.4 003 执行

**Ticket 阶段**（从上往下读，第一个匹配的行生效）：

| 阶段 | 条件 | 面板标签 |
|---|---|---|
| person / delivery | 向 Executor 会话投递 Input 失败；2 秒后重试，之后延迟翻倍直到 1 分钟 | Input undelivered |
| person / answer | 有一个 Attention（待人回答的问题） | Awaiting answer |
| person / review | 最新 Report 是 `ready` | Awaiting review |
| person / blocked, failed | 最新 Report 是 `blocked` 或 `failed`，且此后没有人的消息到达 | Blocked · Failed |
| working / queued | 没有 Assignment，或 Assignment 为 pending | Queued |
| idle | 最新 Report 是 `waiting`，或 Assignment 已结束 | Waiting for a Trigger |
| working | 其他所有情况 | Working |

**Executor**

- 每个 Ticket 一个名为 `worker-<ticket id>` 的 Executor 会话，和一个以 Ticket 短名命名的目录（默认在 `<stateDir>/work` 下）。
- **Executor 会话以运营者的 OS 用户身份、免审批运行命令。** 其 `dsh-superagent-worker` preset 只含干活所需的工具，并且**前台 shell 命令 5 秒后自动转为后台任务**，这样 worker 在下一个 step 就能收到新消息，而不是等长命令跑完。
- 主会话用 `dsh-superagent-router` preset，同样有 5 秒的 shell 与等待上限，工具集更小。会话保持创建时选定的 preset。Worker 的 subagent 用 DeepSeek Harness 的 `standard` preset（shell 60 秒才转后台）。
- **后台工作与 worker 槽位**：一轮结束时如果还有 shell 任务或 subagent 在跑，Ticket 保留槽位、保持 working，不报 waiting；任务完成会在同一槽位唤醒 Executor。**每轮结束后后台工作最多再有 30 分钟**，到时杀掉 shell 任务并报 waiting（subagent 允许继续）。交给人（ready 报告或提问之后，或任何 blocked / failed 报告之后）时立即释放槽位，subagent 停止，shell 任务（如为评审开的服务器）可以一直跑到 Assignment 结束。
- **准备目录**：首次 Assignment 前目录不存在时，在执行根目录下用 `sh` 跑一次 prepare 命令，限时 10 分钟；典型做法是 `git worktree` 建一个 checkout。PR 分支名的最后一段与 Ticket 短名匹配时，该 PR 属于这个 Ticket。
- **并发**：配置的并发上限作用于 worker 槽位，**另外多留一个槽位给人的消息或状态请求**，所以新工作永远不会挡住人继续推进已有 Ticket。CI 结果等其他 Input 等普通槽位。
- **执行变量**：`DSH_SUPERAGENT_DIR` / `_NAME` / `_TICKET`（目录、短名、id）、`DSH_SUPERAGENT_BLOCKED_COMMANDS`、`DSH_SUPERAGENT_GITHUB_REPOSITORY` / `_ACCOUNT` / `_COMMIT_EMAIL` / `_NUMBER` / `_PR_HEAD`、`DSH_SUPERAGENT_FEISHU_BOT` / `_CHAT`。**只有面板能改执行环境设置，发给主会话的文字改不了配置的 shell 命令。** Worker 通过 agent 账号的 `gh` 访问 GitHub，通过 bot 的 `lark-cli` 访问飞书；被屏蔽的命令按名字调用时会失败。**原文提醒：这些包装和目录建立的是默认值，不是安全边界，worker 仍然可以按绝对路径执行程序。**

**合并**：自动合并由 **Delivery requirements** prompt 控制，宿主没有合并专用的决策逻辑。默认 Executor **从不以 agent 账号开启自动合并**（GitHub 会把合并记在开启者头上），Delivery requirements 必须指定另一个账号的 CLI 来合并，否则由人合并。默认指令只允许对为该 Ticket 开的 PR 自动合并，且要求：已就绪、所需批准覆盖 head commit、**增删行数合计少于 200 行**；推新修订前先撤回自动合并。真正强制批准要靠分支保护。

**Ticket 如何结束**：`ready` 且无 PR → 完成；无事可做（复现不了、已被取代）→ `dropped`，默认不发消息；有 PR 时，**当它自己开的 PR 全部关闭**（若没开过，则它跟随的 PR 全部关闭）时结束，结局取最后一次相关关闭（merged / closed），并保留订阅，之后人的评论可以恢复它。聊天渠道只收一条关于 PR 结局的最终通知（GitHub 上 PR 自己就显示结局）。

**失败与恢复**（要点）：

- Executor 报 `blocked` / `failed` 时，Ticket 不接收自动 Input，CI 和 bot 消息只记录不投递；**工具仍可用，人可以直接在会话里接着干**。人的消息、后续非 blocked/failed 的报告或一次 retry 会解除这种状态。
- 模型提供方或网络的暂时性故障（HTTP 408、429、5xx，连接被拒、重置、超时）先自动重试 3 次，退避递增；prepare 命令失败不重试。
- **Retry** 结束当前 Assignment，在同一会话、目录、分支和 PR 里开一个新的。
- **重启**：运行中的 Assignment 以 "host restarted" 结束，新 Assignment 带一条恢复说明启动，要求 Executor **在重复任何外部动作之前先检查目录、git 历史和 GitHub**，按最老 Ticket 优先。会话、目录、订阅、待处理消息、存储的工作，以及**每个 Executor 会话把哪些消息读进了历史**，都会在重启后保留。
- **先存后做**：决定之后要做的事（转发候选/跟进/报告给主会话、启动 Assignment、投递 Input）都**先存储再执行**；失败的转发或投递 2 秒后重试、延迟翻倍到 1 分钟；同类工作连续失败 3 次才在系统问题里报错。
- **Source 失败**：GitHub 每个 Issue、PR、Thread 单独读，失败只记在该 Thread 上，下次轮询重读；默认连续 3 次轮询失败才报 Source 错误；404 也会重试（失去私有仓库访问权时 GitHub 也返回 404）。
- **Channel 失败**：外发消息作为存储的工作发送，从 5 秒开始退避、翻倍到 300 秒；同一 Ticket 发往同一对话的消息**按排队顺序发出**（确认一定在结果之前）。
- **暂停处理**：停止派发、新 Eval、Eval 超时、Input 投递和回复；Source 继续记录 Trigger，运行中的 Executor 轮次不被打断；暂停期间 Eval 不会过期，截止时间顺延暂停时长。

> **💡理解**：这一节基本是**分布式系统的可靠性清单**：幂等（Trigger 去重）、先写日志再执行（stored work，类似 outbox pattern）、指数退避、按会话保序、重启后从持久状态恢复并"先检查再重复外部动作"（避免重复提 PR、重复发消息）。长文 §6 的 Watchdog 在原型里没有实现成一个 agent 会话，但这些机制加上 §5.6 的每小时检查，覆盖了它的一部分职责。

### 5.5 004 Prompts

**团队可写的部分**：三个团队段落、团队 skill，以及覆盖任何内置 system prompt 片段。默认内容大多在 `prompts/` 和 `skills/` 下；工具描述、通知和卡片留在代码里，不能通过 prompt 设置修改。

| 段落 | 内容 |
|---|---|
| Communication | 何时给人发消息、消息怎么写、每条 GitHub 消息开头的声明行（默认 `This is by an automated agent, not a human`）、怎么处理 review、哪些普通账号是 review bot |
| Ticket scope | Ticket 什么时候继续、什么时候带着草稿停下 |
| Delivery requirements | PR 里的证据、自动合并条件及行数阈值、提交作者（不填则用 agent 账号的 no-reply 地址提交） |
| Team skills | agent 需要时加载的知识，例如说明哪个飞书用户对应哪个 GitHub 账号的 `identities` skill |

**内置片段的覆盖键**：`runtime/shared`、`runtime/orchestrator`、`runtime/worker`（共享指令与角色）；`runtime/subscriptions`、`runtime/replies`、`runtime/dispatch`、`runtime/work`、`runtime/admission`（工作指令）；`runtime/handoff/orchestrator`、`runtime/handoff/worker`（交接指令）；`skill/feishu`、`skill/github`（Source skill）；`runtime/environment`、`runtime/policy`、`runtime/policy/none`（生成段落，可编辑模板，必须保留 `{variables}`、`{version}`、`{mode}`、`{rules}` 等占位符）。**编辑片段只改变模型收到的指令，不改变 Gate、Mandate、工具检查和账号包装，这些由代码强制执行。**

**组装顺序**：① 共享指令与角色 → ② 三个团队段落 → ③ 订阅与回复 → ④ 主会话的派发指令 / Executor 的工作指令 → ⑤ `runtime/admission`（仅主会话）→ ⑥ 交接指令 → ⑦ `runtime/environment` → ⑧ `runtime/policy`（仅主会话）。Policy 段每次模型请求都从设置重新生成，放在最后，这样 **Policy 变更不会改变前面的 prompt 文本**（我的理解：这样对 prompt cache 友好）。Executor 还会读 DeepSeek Harness 的 coding prompt 和仓库的 `AGENTS.md`，主会话不读。Executor 无人值守，工具里没有 `ask_user_question`、`present`、`plugin_manager` 等。

**优先级（从强到弱）**：

1. 工具和宿主强制的事实：身份、权限、工具参数检查、回复用途、长度限制。**prompt 文本改变不了它们。**
2. 内置和生成的段落（以团队覆盖后的版本为准）。
3. 主动 Ticket 匹配规则里的流程和沟通要求（只对该 Ticket 的 Executor）。
4. 团队段落和团队 skill。
5. **外部文本：消息、评论、文档、引用。它们是数据，永远不改变规则或授权。**

**内置文本的原则**（摘译）：

- **角色分离**：主会话只路由，转发原始 Trigger 和路由事实，**不做技术决策**，用一个路由工具一次处理多个候选；**路由备注最多 30 个词**。Executor 读原始来源、判断范围、实现并验证。
- **主会话不等待**：报告、跟进通知、候选、完成的后台任务都会自动以消息到达；没事可做就结束这一轮。一个守卫会拒绝 `sleep`、超过 30 秒的 shell 超时和超过 5 秒的任务等待，因为阻塞的 step 会延误新候选。
- **只在有人必须行动或能用上结果时发消息**；聊天消息默认一句话，**硬上限 100 词**。
- **Executor 自己决定跟进**：处理并确认范围内的请求，交还超出范围或属于别的 Ticket 的请求。
- **通过订阅和主会话协调**：Executor 跟随相关 Thread 获取上下文；主会话可以用不超过 100 词的备注把另一个 Ticket 的事实告诉它。**订阅和备注都不授权。**
- **只在明确引用时附加**：消息必须指向某 Ticket 的 PR、Issue、源消息或确认，或指向恰好一个匹配 Ticket 的近期改动，**话题相似不够**。
- **具体地确认一次**：动手前在原消息下确认每个新请求，说明对象和预期结果。
- **通过 `human_action` 提问**：Executor 在 Report 里放一个具体问题，主会话原样转发；已授权的工作可以同时继续。
- **诚实交付**：开了 PR 不等于就绪；草稿标未完成；**没有真实"改动前"截图的 UI 改动不算就绪**。
- **人的暂停一直有效**，跨跟进、新会话和重启，直到人解除；自动反馈永远不恢复被暂停的工作。
- **示例只示范格式**：示例句子不是 agent 可以声称的事实。
- **按结果处理 review**：修好并验证的评论静默 resolve；拒绝的给一句理由；每个自动评审者每个 PR 最多三轮。
- **给重复的工作让路**：人的分支或 PR 已经包含这个改动时，Executor 用一句话指向人的 PR 并关掉自己的。

**版本**：每次保存产生一个不可变版本（完整配置、修订号、前驱、时间戳、来源），恢复旧版本或默认值也产生新版本；保存要指明基准修订，冲突会被拒绝但不丢草稿；**会话每次模型请求都读最新版本**，包括同一轮的后续 step；升级时保留团队文本，默认内容变了面板会标 **Default updated** 并显示差异。

> **💡理解**：这份"优先级五层表"是整个原型安全模型最清楚的表述：**能用代码强制的绝不交给 prompt**，prompt 只负责"怎么做得好"，不负责"能不能做"。第 5 层"外部文本永远是数据"直接对应长文 §8.4 的 prompt injection 风险。

### 5.6 005 面板

打开 DeepSeek Harness Web 侧边栏的 **Superagent** 来监督 agent。面板按顺序回答三个问题：

1. 现在有什么需要我处理吗？
2. 系统健康吗？在做对的事吗？
3. 某项工作从 Trigger 经 Eval 到 Ticket 的历史是什么？

| 标签页 | 显示 |
|---|---|
| Overview | 需要处理的 Ticket、系统问题、近 24 小时活动计数、Executor 负载 |
| Source | Source 状态、设置问题、非零计数 |
| Trigger | 按"已订阅 / 评估中 / 24 小时内新 Trigger / 全部"筛选的 Thread，含 Trigger、Decision、订阅 |
| Eval | 带理由和匹配规则的 Decision |
| Execution | Executor 状态、暂停处理、按状态筛选的 Ticket；详情含操作、授权、待处理 Input、订阅和 Assignment |
| Settings | Sources、Eval、Execution、Prompts、Deployment，每组带保存历史 |

面板原则（摘要）：**聚焦重要的事**（每个元素都回答三个问题之一，否则删掉；数字只在会改变监督者行为时才出现）；**精确定义**（分类互斥且全覆盖，标签说明这一行在等谁、需要什么）；**一个概念一个地方**；**状态诚实**（计数等于它链接的列表长度）；**默认安静**（健康状态不占空间，一次性网络错误不算问题）；**控件一致**。

### 5.7 006 部署（要点）

- **运行组成**：一个专用的 DeepSeek Harness profile（文档叫 `superagent`），从 Web 模板创建，在 `127.0.0.1` 上提供面板。**要与你自己工作用的 profile 分开**，因为插件会改进程级 `PATH` 和 git 设置，且需要独立重启和升级。一个 **agent 专用 GitHub 账号**（不能是 GitHub App，因为要读账号通知）。可选一个飞书机器人。
- **权限与隔离**：worker 和 router 两个 preset 都以 **`danger-full-access` + 运营者 OS 用户**运行，并读取订阅 Thread 的消息。账号包装默认选择 agent 的凭据，**但不隔离进程**：worker 可以按路径执行程序、读运营者的 home 目录。**要隔离，就用一个只含 agent 凭据的专用 OS 账号。**
- **前置依赖**：Node.js 22.19+ 或 24+、pnpm、git 2.32+、Python 3.8+、tmux、GitHub CLI 2.48+；仓库必须在 github.com（不支持 GitHub Enterprise Server）；飞书需要 `lark-cli`（测试版本 1.0.90）。
- **账号包装**：给 agent 一个只用自己凭据的 `gh-agent` 脚本（设置独立的 `GH_CONFIG_DIR`）；飞书 bot 用独立的 `LARKSUITE_CLI_CONFIG_DIR` 并开启 bot strict mode，**宿主会拒绝能以用户身份操作的 CLI**。
- **安装**：

```sh
npm install -g @deepseek-ai/dsh@0.2.1-rc.3
git clone <this repository> dsh-superagent && cd dsh-superagent
pnpm install && pnpm build
dsh superagent --from-default-profile web --dump-config > /dev/null   # 仅新装时运行：从 Web 模板创建 ~/.dsh/profiles/superagent
dsh plugin --profile superagent add "link:$PWD"                        # 安装本 bundle
```

- **启停**：`scripts/dsh-superagent start | stop | restart | status`。`start` 在 tmux 会话里跑 `dsh --profile superagent`；`stop` / `restart` 请求优雅关闭，最多等 Executor 5 分钟，**从不强杀**。`status` 退出码：**0** 正常；**1** 没运行 / 运行了两个 / 不响应 / 运行时报错（重启可能修好）；**2** 运行着但某部分（如某个 Source、Policy 同步）报错（重启修不好）；**3** 运行着但在等设置。
- **每小时检查**：在**你平时用的那个 profile 里**（不是 superagent 里，这样服务挂了检查还在）让 agent 建一个每小时定时任务：跑 `status`；退出码 0 什么都不说；2 和 3 报告问题但不重启（同样的问题不重复报）；用户手动 stop 过就保持停止；1 则带 `--respect-stop` 重启，重启还失败就读最新日志和排障文档，能修就修，并告诉用户出了什么问题、做了什么。
- **Setup 检查**：setup 会**删除名字含 KEY、PASSWORD、SECRET、TOKEN 的环境变量**；在 `<stateDir>/worker-bin` 里写命令 shim（`gh` → agent CLI，`lark-cli` → bot CLI，屏蔽的命令直接失败），放在 `PATH` 最前；写一份 `worker-gitconfig` **替换运营者的全局和系统 git 配置**（agent CLI 作为 github.com 的凭据助手、SSH 地址改写为 HTTPS、git 永不弹出凭据提示、移除 `GH_TOKEN` / `GITHUB_TOKEN`）。
- **管理 API**：面板用的本地 HTTP API 也给 agent 用（`SETUP.md` 就用它保存设置），地址 `http://127.0.0.1:<port>/orchestrator/api/`，要求带 `x-orchestrator-request: 1` 头，Host 必须是 `127.0.0.1` / `localhost`，跨站 `Origin` 返回 403。
- **试用**：用白名单上的 GitHub 账号开 Issue 或发评论 `@<agent login> <请求>`，一个轮询周期内（默认 1 分钟）Overview 上就会出现 Ticket。注意：**编辑 Issue 正文加上的 @ 收不到**，要发新评论。
- **升级**：等没有 Executor 在轮次中 → 备份 `<stateDir>/orchestrator.sqlite` → `git pull && pnpm install && pnpm build` → `restart`。新版迁移过的数据库旧版打不开，回滚只能靠备份。

### 5.8 007 架构（代码层面）

```text
Source 插件 ──SourceContext──> Engine <── 会话工具 / 管理路由
                                │
        ┌───────────┬───────────┼────────────┬──────────────┐
        ▼           ▼           ▼            ▼              ▼
   Store(Repos)  Policy   SourceRegistry  MasterAssessor  DshExecutor
                                              │ decide        │ assignmentStarted /
                                              └──> Engine <───┘ inputsDelivered / report
   Engine ──Outbox──> Outbox；Settings ──policyChanged──> Engine
   Assessor / Executor / Sessions / Outbox ──enqueue──> Effects ──> Store
```

- **Runtime**（`src/app/runtime.ts`）是组合根，不含领域规则：锁状态目录、打开 Store、构建组件；启动时注册管理路由、跑 Setup、安装会话和 Executor 钩子、确保主会话存在、启动 Policy 同步、恢复被中断的 Assignment、启动 passes 和 effects。**Passes 每两秒跑一次**（以及命令请求时）：先处理到期的 effects，再 `Engine.evaluateDue`。
- **Engine**（`src/core/engine.ts`）是 Trigger、Thread、Decision、Ticket、Assignment 的**唯一写者**。所有方法都是**同步**的，每个命令在一个事务内修改其聚合；端口在事务内把外部工作入队，适配器在提交后执行。内部分五部分，依赖单向：`Intake`（接收 Trigger、扇出、排队 Eval）→ `Evals`（批处理、Gate 强度、截止时间、超时重排、Decision）→ `Tickets`（生命周期、Mandate、订阅、恢复、重试、重启、停止）→ `Assignments`（Executor 报告）→ `Replies`（经 Outbox 写 Channel）。调用方只拿到自己角色的那部分 Engine（`EngineForSources`、`EngineForMainSession`、`EngineForPeople`、`EngineForSettings`、`EngineForExecutor`）。
- **Policy**（`src/policy.ts`）：`settingsPolicy(settings, scope)` 每次使用返回一个全新的、无状态的 `Policy`，含 `gate()`、`revalidate(mandate)`、`admitsInferred(rules)`；`assessmentOutcome()` 把模型评估转成 Engine 结果，**要求对每条启用规则都有判断、方向确定、至少一条匹配规则以及意图证据**；`policyVersion` 对规则、接单模式和各 Source 的 Policy 部分（含 bot 名单）做哈希。
- **持久化**：一个 SQLite 文件 `<stateDir>/orchestrator.sqlite`，表 `trigger`、`disposition`、`trigger_thread`、`thread`、`ticket`、`decision`、`effect`（effect 队列）、`outbox`（每条外发消息的回执，保证同一 id 不会排队两次）、`kv`。原文坦言：**目前没有任何东西删除旧记录**，这些表会无限增长。
- **主会话的 14 个应用工具**：`orchestrator_read`、`orchestrator_route`（一次 1~20 个决定）、`orchestrator_ticket_retry` / `_request_status` / `_note`、`orchestrator_trigger_requeue`、`orchestrator_eval` / `_eval_read` / `_eval_decide`、`orchestrator_reply`、`orchestrator_thread_unsubscribe`、`orchestrator_settings_save`、`orchestrator_policy_sync`、`orchestrator_prompts_save`。另外还有 Harness 的 `skill`、`bash`、`read`、`read_image`、`job_*`。原文特别说明：**"主会话避免读代码"是 prompt 规则，不是工具限制。**
- **Executor 的 6 个应用工具**：`orchestrator_read`（只能读自己 Ticket 的资源）、`orchestrator_ack`、`orchestrator_report`、`orchestrator_thread_unrelated`、`orchestrator_thread_follow`、`orchestrator_trigger_handback`。Worker **拿不到**向人提问或展示、改 profile、在槽位外调度工作、启动或联系其他会话、编辑外部文档的工具；Ticket 结束后工具调用一律拒绝。
- **`noWait` 守卫**（`src/app/sessions.ts`）：拒绝主会话在命令位置调用 `sleep`、shell `timeoutMs` 超过 30 秒、`job_output` 等待超过 5 秒。
- **评估上下文**（`orchestrator_eval` 返回的是上下文而不是指令）：Trigger 文本最多 4000 字符；最多 8 个相关 Ticket；Thread 窗口取**前 5 条消息 + Trigger 前后各 5 条**，每条通常截到 400 字符（紧邻的两条放宽到 1200），其余消息只给 50 字符的预览存根，用 `orchestrator_eval_read` 按 id 展开。**这就是长文 Inline & Spill 的具体参数。**
- **读资源**：`ticket/<id>`、`thread/<key>`、`trigger/<id>`、`decision/<id>` 等；页面上出现的每个 id 都能通过这些资源读到，所以会话可以从一个对象走到下一个（**Chain of Recall 的落地**）。Executor 只能读自己 Ticket 相关的对象，页面会隐去它不能读的 id。文本页默认 4000 UTF-16 码元（最多 8000），列表默认 20 项（最多 50）。
- **投递给会话的消息**：候选（Trigger 卡片批次等）、`follow_up`、`worker_report`（摘要最多 600 字符）、`worker_attention`（携带问题）会被 steer 进主会话的轮次；`ticket_ended`、`settings_saved` 等通知注入但不唤醒主会话。

**Preset 定义**（摘自 `cordis.patch.yml` 注释）：worker preset 是 DSH 的 `standard` preset 只保留 worker 需要的部分，shell 前台命令 5 秒转后台；**去掉了**：向人提问、展示、plan mode、插件管理器和市场、调度、goals、workflows、web fetch、forked subagent（它会在这个 preset 下运行）。用 spawn 方式创建的 subagent 跑在 `standard` 下。由于 preset 行没有"在选择器里隐藏"的设置，所以靠名字和描述（"dsh-superagent 执行器……不供手动选择"）提示用户。

### 5.9 008 Cookbook 与 SETUP

- **008 Cookbook：添加一个 Source**。以一个虚构的 `webhook` 消息队列（CLI 叫 `hookq`）为例，分 10 步：① 把事件映射到领域（Trigger = 一条消息；Thread = `<conversation>:<首条消息 id>`；Actor = 发送者 id；Channel 地址 = 对话 + 回复目标消息；接收范围 = 一个对话）。**先选稳定的映射再写代码**，因为 Trigger id、Thread id、Channel 地址都会持久化，以后改会断开已有记录；② 配置 schema、字段、事实与健康；③ 把消息翻译成 Trigger；④ 接收范围；⑤ 回复 Channel；⑥ 评估用的 Thread 上下文；⑦ 谁能请求（该 Source 的 Policy 部分）；⑧ 给 Executor 的变量；⑨ 插件入口和轮询循环；⑩ 注册插件；最后是测试和检查清单。
- **SETUP.md**：给 coding agent 读的安装剧本，分 ① Probe（探测环境）→ ② Questions（**一次只问一个短问题**）→ ③ Plan（**完整计划只展示一次**）→ ④ Execute → ⑤ Verify and hand over 五步。README 的建议用法就是对你的 coding agent 说 "Read SETUP.md in this repository and set up dsh-superagent with me."

<br>

## 6. 设计（长文）与实现（原型）对照

| 长文模块 | 原型是否实现 | 原型中的形态 / 差异 |
|---|---|---|
| §2 Router / Worker 双层 | ✅ | 主会话（`dsh-superagent-router`）+ 每 Ticket 一个 Executor 会话（`dsh-superagent-worker`）；主会话仍有 `bash` 等工具，"不读代码"靠 prompt 约束 |
| §2.1 Router 批处理 | ✅ | 运行中到达的 relay 在下一步经 Harness inbox 一起投递；`orchestrator_route` 一次 1~20 个决定 |
| §3.1 两级过滤 | ✅ | Source 自身过滤 + Gate（白名单、准入表、bot 名单）+ Assessment（rules 写成 prompt）；接单模式 off / observe / auto |
| §3.2 Thread 订阅与透传 | ✅ | Subscription + Fan-out；handback、unrelated、follow-for-context；比长文多了严格的"跟进归属"规则 |
| §3.3 Inline & Spill / Chain of Recall | ✅（具体参数化） | Eval 上下文窗口（前 5 + 前后各 5 条，400/1200 字符，50 字符存根）；读资源的 id 互相可达 |
| §3.4 推测执行 / 静默复现并修复 | 部分 | 可写成 Policy rule（"silence rule" 可以不发确认）；默认 observe 模式下只记录不接单 |
| §4.1 全双工异步 ReAct | 简化 | 前台命令 5 秒自动转后台 + 新消息下一 step 领取；Harness 的完整 `start()/stop()` 机制默认未开启 |
| §4.2 多渠道规范 | ✅ | Communication prompt；聊天默认一句话、硬上限 100 词；GitHub 消息带自动化声明行 |
| §5 计算 / 状态 / 环境分离、BYOM | ❌ | 原型是单机部署（本地 SQLite + 本机目录 + tmux），没有云端 Loop 和端点 daemon |
| §6 Watchdog 会话 | ❌（有替代） | 用"另一个 profile 里的每小时定时任务 + `status` 退出码"做外部健康检查；没有语义层面的"Worker 是否在推进"判断 |
| §8.4 凭据不进上下文 | 部分 | 账号包装 + 清除敏感环境变量 + 专用 gitconfig；官方明确说这不是安全边界 |

> **💡理解**：原型实现的是长文里"**单机、单仓库、单团队**"能验证的部分（双层、事件驱动、订阅、上下文组织），没有实现需要基础设施的部分（云端存算分离、BYOM、Watchdog agent）。读原型代码时要记住：它是长文 §2~§3 的参考实现，不是 §5~§6 的。

<br>

## 7. 个人梳理与点评

### 7.1 和其他 harness 的对比

| 维度 | 常见本地 coding agent（单会话） | DeepSeek Harness | Super-agent |
|---|---|---|---|
| 扩展方式 | 配置 + hooks + MCP | 一切皆 Cordis 插件，**loop 本身可替换** | 以 Harness bundle 形式实现 |
| 上下文真相源 | 内存消息列表 | **仅追加事件日志，模型可见即已记录** | 外部对象图（ID 引用）+ 按需召回 |
| 并行 | subagent 工具（模型自己决定用不用） | subagent / agent team / Ralph 插件 | **架构强制**：Router 派单、Worker 隔离 |
| 人机交互 | 人发起、人盯着 | 同左（+ 定时任务、webhook、goal） | 事件驱动、主动、只在需要时打扰人 |
| 权限 | 审批 + 沙箱 | 审批 seam 与沙箱 seam 分离、fail-closed | Mandate 与上下文分离；外部文本永远是数据 |
| 运行位置 | 本机 | 本机 / 远端沙箱（换 provider） | 设计：云端 Loop + 本地 daemon（BYOM）；原型：单机 |

### 7.2 我认为最值得借鉴的五点

1. **事件溯源式会话日志**（§3.6）：可回放、可分叉、可审计，对评测和调试 agent 价值巨大，也是"重启后精确恢复"的基础。
2. **授权与上下文分离**（§5.2、§5.3）：agent 可以读大量不可信输入，但这些输入改不了它的权限。这是结构性防注入，比在 prompt 里叮嘱模型"别听"可靠得多。
3. **能用代码强制的绝不交给 prompt**（§5.5 的五层优先级）：prompt 负责做得好，代码负责能不能做。工具被过滤后对模型"根本不存在"（§3.10）也是同一思路。
4. **用架构而不是 prompt 保证关键行为**（§4.7）：优先委派 prompt 仍有 1/5 的长尾，说明关键行为不应依赖模型每次都自觉。
5. **把"注意力"当作一等资源**（§4.1、§5.1）：每条消息、每个面板元素都要证明自己的价值，默认安静。

### 7.3 需要保持清醒的地方

- 实验是**合成负载、自报数据、每种设计 5 次**，编码用耗时工具模拟，只验证调度，不代表真实任务质量提升。
- 成本：Super-agent 的 cache-read token 是单会话方案的 2~4 倍。
- 原型 Worker **免审批、以 OS 用户身份、`danger-full-access` 运行**，必须用专用 OS 账号或虚拟机隔离；SAFETY.md 明确沙箱不是安全保证。
- Harness 仍是 developer preview，配置格式和插件 API 会有破坏性变更；原型的 SQLite 记录目前不会清理。
- 长文的 BYOM、Watchdog、默认全双工都还是**设计**；§8 的长期记忆、A2A、凭据隔离都是**开放问题**。

### 7.4 一个可以带走的思考框架

长文 §1 说，相似的设计约束会导出相似的架构。我把 Super-agent 的设计归纳成"三个分离"：

| 分离 | 分开了什么 | 解决什么 |
|---|---|---|
| **交互与执行分离** | Router（对人）/ Worker（干活） | 连续性 vs 吞吐量 |
| **授权与上下文分离** | Mandate / Subscription | 主动性 vs 安全性 |
| **计算、状态与环境分离** | 云端 Loop / 数据库 / 端点 daemon | 持久在线 vs 真实环境 |

再加上 Harness 层的"**事实与控制分离**"（持久会话事件 / 实时 `agent/*` 事件）和 Cordis 层的"**注册与撤销成对**"，这五个分离基本就是整套技术报告的骨架。

<br>

## 参考资料

**原始出处**
1. DeepSeek Harness 官网：https://www.deepseek.com/en/harness/
2. DeepSeek Harness 仓库：https://github.com/deepseek-ai/deepseek-harness
3. 架构文档：https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.zh.md
4. Agent 生命周期：https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/agent-lifecycle.zh.md
5. 工具执行流水线：https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/tool-execution-pipeline.zh.md
6. Cordis 入门：https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.zh.md
7. 能力 Seams：https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/capability-seams.zh.md
8. 术语表：https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/glossary.zh.md
9. 工具目录：https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/tool-catalog.zh.md
10. SAFETY：https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.md
11. Cordis 论文：https://arxiv.org/abs/2608.25512
12. Super-agent 长文（知乎）：https://zhuanlan.zhihu.com/p/2099987320443237060
13. Super-agent 长文（X）：https://x.com/DeepSeekHarness/status/2116857257245987236
14. dsh-superagent-prototype：https://github.com/deepseek-ai/dsh-superagent-prototype （docs/000~008 + SETUP.md）
15. 长文引用的 Anthropic 文章 *Decoupling the brain from the hands*：https://www.anthropic.com/research/managed-agents

**解析**
- Habr：https://habr.com/en/articles/1070958/
- explainx.ai：https://explainx.ai/blog/deepseek-super-agent-architecture-router-worker-triggers-watchdog-2026
- agent-works Issue #184：https://github.com/yu-iskw/agent-works/issues/184
- DEV Community：https://dev.to/paullovelace/deepseek-super-agent-a-proactive-open-source-coding-agent-prototype-what-it-actually-is-and-438a
- MarkTechPost：https://www.marktechpost.com/2026/11/01/deepseek-dsh-superagent-router-worker-coding-agent/
- IT之家：https://www.ithome.com/1/020/615.htm
- 知乎解读：https://zhuanlan.zhihu.com/p/2100268290350368746
- 36氪：https://36kr.com/p/4050650194630536
