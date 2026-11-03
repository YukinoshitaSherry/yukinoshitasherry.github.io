---
title: Agent Communication综述整理(Protocols & Safety)
date: 2025-07-13
categories:
  - 学AI/DS
tags:
  - LLM
  - Agent
desc: 整理自多篇综述与论文，以《A Survey of LLM-Driven AI Agent Communication- Protocols, Security Risks, and Defense Countermeasures》为主线；MCP 与 A2A 部分自顶向下讲解原理并附源码。
---

综述地址：
- <a href="https://arxiv.org/abs/2506.19676"> A Survey of LLM-Driven AI Agent Communication- Protocols, Security Risks, and Defense Countermeasures </a>
  - 也参考了该综述在Section C提到的其他文章:
    - Hou, Y., et al. Challenges and Opportunities in LLM-Based Agents: A Survey on Knowledge Gaps
    - Zhang, Y., et al. A Survey on Communication Protocols and Security in Multi-Agent Systems
    - Li, X., et al. Security and Privacy in LLM-Powered Agents: A Comprehensive Survey
    - Wang, L., et al. Risks in Multi-Agent Collaboration: A Systematic Review
    - Hou, Y., et al. Security Risks in the Model Context Protocol (MCP): An Early Analysis
    - Chen, J., et al. Seven Security Challenges in Multi-Agent Systems
    - Liu, H., et al. The Model Context Protocol: Principles and Practices
    - Zhao, R., et al. Safety and Alignment in Single-Agent LLM Systems: A Survey
    - Microsoft Research. Personal Agents as the Next Platform: Vision and Challenges
    - Kim, S., et al. Agent-to-Agent (A2A) Communication Protocols: Design and Evaluation
    - Patel, D., et al. MCP and Beyond: Protocols for Agent Interoperability
    - Gupta, A., et al. A Survey of Communication Mechanisms in LLM-Based Multi-Agent Systems
- MCP / A2A 协议原理部分另参考：
  - <a href="https://zhuanlan.zhihu.com/p/29001189476">MCP (Model Context Protocol)，一篇就够了。</a>
  - <a href="https://zhuanlan.zhihu.com/p/1895786881360316389">A2A协议(Agent to Agent Protocol)详解与官方案例实践分享</a>
  - <a href="https://modelcontextprotocol.io/">MCP 官方文档</a>、<a href="https://a2a-protocol.org/">A2A 官方文档</a>、<a href="https://github.com/a2aproject/a2a-samples">a2a-samples</a>


<br>

## 背景

### 术语

为确保技术讨论的严谨性，以下术语在本文中具有严格定义：

| 术语 | 精确定义 | 技术细节 |
|------|--------|--------|
| **U-A (User-to-Agent)** | 用户与智能体之间的交互通道，通常以自然语言或结构化指令形式存在 | - 输入为自由文本，需经意图识别（Intent Recognition）<br>- 安全风险包括 Prompt Injection、Jailbreaking、Privacy Leakage |
| **A-A (Agent-to-Agent)** | 两个或多个智能体之间的协作通信，用于任务分解、协商、同步 | - 可基于自然语言（如 Meta Cicero）或结构化协议（如 A2A、MCP）<br>- 需解决身份认证、消息序列化、一致性保证等问题 |
| **A-E (Agent-to-Environment)** | 智能体与外部环境（工具、API、数据库、传感器等）的交互 | - 通过 MCP、RAG、Function Calling 等机制实现<br>- 环境可为云端服务、本地文件或物理设备 |
| **Capability Expression** | 智能体声明其可执行功能的方式，包含功能签名与语义描述 | - 在 MCP 中通过 `tools/list` 返回的工具对象表达<br>- 包含 name, description, inputSchema (JSON Schema)，可选 outputSchema<br>- 在 A2A 中通过 Agent Card 的 skills 表达 |
| **Schema-Agnostic** | 协议不依赖特定数据序列化格式，支持多种底层表示 | - MCP 内部使用统一抽象模型<br>- Host 负责格式转换（如 YAML ↔ JSON ↔ Protobuf） |
| **Host (MCP)** | 本地可信中介，非网络主机，而是安全策略执行点 | - 运行在用户可控环境<br>- 实施细粒度访问控制（如“仅允许读取日历”）<br>- 防止恶意工具窃取用户数据或越权操作 |

<br>

### 标准化通信协议需求
早期 LLM 应用局限于文本生成（Text Generation），而现代 LLM 驱动的智能体将其作为**认知核心（Cognitive Core）**，执行 ReAct（Reasoning + Acting）循环：

1. **Perceive**：接收 U-A 指令或 A-A 消息；
2. **Plan**：基于当前状态与目标，分解子任务，决定是否需要 A-E 调用；
3. **Act**：通过 MCP、Function Calling 或 RAG 调用外部资源；
4. **Reflect**：评估执行结果，决定是否重试、修正或终止。

关键技术支撑包括：
- **ReAct Framework**（Yao et al., 2022）：交替进行推理步骤与行动步骤，提升任务成功率；
- **Toolformer**（Schick et al., 2023）：在预训练阶段注入工具调用信号，使 LLM 学会何时及如何调用工具；
- **Function Calling**（OpenAI API）：LLM 可直接输出结构化函数调用请求，由运行时执行。

单一智能体受限于知识边界与计算资源，复杂任务需多智能体协同：
- **专业化分工**：天气代理专注气象数据，航班代理处理航空 API；
- **动态发现与组合**：新工具上线后，主智能体可自动发现并纳入工作流；
- **弹性扩展**：可通过水平扩展代理实例提升系统吞吐量。

随着大型语言模型（Large Language Models, LLMs）在推理、规划、工具使用等方面的能力显著提升，AI 智能体（Agent）已从单轮问答系统演变为具备自主性、持续交互能力与多步任务执行能力的复杂软件实体。

这类智能体需频繁与三类外部实体进行通信：

| 交互类型 | 全称 | 定义 | 典型场景 |
|--------|------|------|--------|
| **U-A** | User-to-Agent | 用户与智能体之间的交互 | 用户输入自然语言指令：“帮我订一张去北京的机票。” |
| **A-A** | Agent-to-Agent | 智能体之间的协作通信 | 主控智能体调用航班代理、酒店代理、天气代理以完成旅行规划子任务 |
| **A-E** | Agent-to-Environment | 智能体与外部环境（工具/API/数据库/文件系统等）交互 | 调用 OpenWeatherMap API 获取天气数据；执行 SQL 查询；读取本地 CSV 文件 |

当前主流多智能体系统普遍采用**封闭生态（Closed Ecosystem）设计**：
- 依赖私有或硬编码的交互机制；
- 外部智能体无法被发现、验证或调用；
- 工具绑定高度耦合，难以替换或扩展。

这种设计导致三大核心问题：
1. **互操作性缺失（Lack of Interoperability）**：不同厂商的智能体无法协同工作；
2. **可扩展性受限（Limited Scalability）**：新增工具需为每个智能体单独开发适配器；
3. **维护成本高昂（High Maintenance Cost）**：行为逻辑与接口逻辑混杂，系统脆弱。

根本原因在于缺乏一个**通用、标准化、上下文感知的通信协议**，用于统一表达智能体能力（Capability Expression）并规范交互流程。

协作的前提是存在**标准化通信协议**（如 MCP、A2A等Agent Protocol），否则将陷入“协议孤岛”。

### 通信碎片化的工程影响

在实际部署中，智能体常面临异构接口环境：
- 某些工具通过 OpenAPI/Swagger 提供 JSON Schema；
- 某些命令行工具仅接受字符串参数（如 `curl -X POST ...`）；
- 某些内部服务使用 Protobuf 或 gRPC。

若无统一协议，开发者必须为每个工具编写 Wrapper（包装层），将 LLM 的自然语言输出转换为目标接口所需的格式。这一过程不仅繁琐，且极易出错，尤其在多工具组合场景下（如“先查航班，再订酒店，最后生成行程单”）。更严重的是，此类硬编码使得系统难以动态适应新工具或变更的 API。

因此，构建一个**模式无关（schema-agnostic）**、**能力驱动（capability-driven）** 的通信协议成为规模化智能体协作的关键前提。

<br>

## MCP 与 A2A：自顶向下理解

> 本节在综述基础上，结合下列两篇文章重写，按「全景 → 动机 → MCP → A2A → 二者如何配合」的顺序展开：
> - <a href="https://zhuanlan.zhihu.com/p/29001189476">MCP (Model Context Protocol)，一篇就够了。</a>（LastWhisper）
> - <a href="https://zhuanlan.zhihu.com/p/1895786881360316389">A2A协议(Agent to Agent Protocol)详解与官方案例实践分享</a>（Tommy学习录）
> - 官方文档：<a href="https://modelcontextprotocol.io/">modelcontextprotocol.io</a>、<a href="https://a2a-protocol.org/">a2a-protocol.org</a>（原 google.github.io/A2A）

### 第 0 层：一句话 + 一张图

- **MCP 管「Agent ↔ 工具/数据」**（对应上文 A-E）：让一个 Agent 用统一接口去用各种工具、读各种数据。
- **A2A 管「Agent ↔ Agent」**（对应上文 A-A）：让不同厂商、不同框架做的 Agent 互相发现、分派任务、交付结果。
- 简记：**MCP for Tools，A2A for Agents**。二者是互补的两层，不是二选一。

```
                       用户 (U-A)
                         │
                         ▼
              ┌─────────────────────┐
              │   主 Agent（Host）   │
              │   LLM + 编排逻辑     │
              └─────────────────────┘
          MCP │                     │ A2A
   (竖向：用工具)                    (横向：找同事)
     ┌────────┼────────┐       ┌────┴────────────┐
     ▼        ▼        ▼       ▼                 ▼
 文件系统   数据库    Web API   酒店 Agent        航班 Agent
 Server    Server    Server   (别家框架做的)     (别家公司部署的)
                                │ MCP              │ MCP
                                ▼                  ▼
                            它自己的工具         它自己的工具
```

一个直观类比：
- MCP 像 **USB-C 接口**：任何外设（工具）只要做成 USB-C，就能插到任何电脑（AI 应用）上。
- A2A 像 **公司之间的外包合同 + 名片**：你不需要知道外包团队内部怎么干活（Agent 是「不透明」的），只需要看名片（Agent Card）知道他们能干什么，然后发任务、等交付。

<br>

### 第 1 层：为什么需要协议

把「给模型提供外部信息」的演进按时间排：

| 阶段 | 做法 | 问题 |
|------|------|------|
| 手工 Prompt | 人工查数据库/文件，复制粘贴进 prompt | 任务一复杂就不可行 |
| Function Calling | 模型输出结构化调用，由应用执行预定义函数 | 各家 API 格式不一样（OpenAI ≠ Google），换模型要重写；工具和应用强绑定 |
| **MCP** | 工具做成独立的 MCP Server，任何支持 MCP 的应用都能接 | 解决了「Agent 用工具」的标准化 |
| **A2A** | Agent 之间用统一的发现与任务协议通信 | 解决了「Agent 和 Agent 协作」的标准化 |

核心收益是把集成成本从 **M×N 降到 M+N**：M 个 AI 应用、N 个工具，没有协议时要写 M×N 个适配器；有了协议，每个应用实现一次 Client、每个工具实现一次 Server 即可。A2A 对「Agent 之间」做的是同一件事。

注意一个关键区分：**Function Calling 是模型的能力**（模型能输出「我要调哪个函数、参数是什么」），**MCP 是工具接入的协议**（工具怎么被发现、怎么被调用、结果怎么返回）。二者不冲突——MCP Client 通常就是把 MCP 工具翻译成模型的 function calling / tool use 格式交给模型。

<br>

### 第 2 层：MCP 是什么

#### 起源与定位

MCP 由 **Anthropic 于 2024 年 11 月 25 日**发布（<a href="https://www.anthropic.com/news/model-context-protocol">Introducing the Model Context Protocol</a>），开源，定义了 AI 应用与外部数据源/工具之间交换上下文的方式。之后 OpenAI、Google、Microsoft 等也陆续支持，成为事实标准。

> 勘误：原综述译文称 MCP「由 Chroma、LangChain 社区联合提出」、Server 实现「gRPC 接口」，均不准确。MCP 由 Anthropic 提出，消息格式是 JSON-RPC 2.0（见下文）。

#### 三个角色：Host / Client / Server

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251220002804704.png" style="width:80%"><br>

| 角色 | 是什么 | 例子 | 职责 |
|------|------|------|------|
| **Host** | 用户直接使用的 AI 应用 | Claude Desktop、Cursor、Claude Code、自己写的 Agent | 跑 LLM、管理多个 Client、做权限确认（「是否允许此工具？」） |
| **Client** | Host 内部的连接器，**与一个 Server 一一对应** | Host 为每个配置的 Server 起一个 Client | 建立连接、能力协商、转发请求和结果 |
| **Server** | 一个独立的小程序，把某类能力暴露出来 | filesystem、GitHub、PostgreSQL、浏览器 Server | 提供 Tools / Resources / Prompts，执行实际操作 |

> 勘误：原综述把 Server 称为「中央注册中心」。实际上 MCP 没有中心化注册：每个 Server 是独立进程/服务，Host 通过本地配置文件知道要连哪些 Server（社区另有 MCP Registry 用于「发现」Server，但不在调用链路上）。

以「我桌面上有哪些文档？」走一遍：

```
你的问题 → Claude Desktop (Host) → Claude 模型
       → 模型判断需要文件信息 → Host 内的 MCP Client
       → filesystem MCP Server 扫描桌面 → 返回文件列表
       → 结果回到模型 → 模型生成回答 → 显示给你
```

开发者通常只需要写 Server，Host 和 Client 由应用方实现好了。

#### Server 能提供什么：三类原语

| 原语 | 谁来控制 | 含义 | 例子 |
|------|--------|------|------|
| **Tools（工具）** | 模型决定何时调用 | 可执行的函数，可能有副作用，一般需用户批准 | `run_sql(query)`、`send_email(...)` |
| **Resources（资源）** | 应用/用户决定是否附加 | 只读的、类文件的数据，用 URI 标识 | 表结构、文件内容、API 响应 |
| **Prompts（提示模板）** | 用户主动选择 | 预写好的任务模板，常显示为斜杠命令 | 「/分析这张表」 |

反方向（Server 向 Client 请求）还有 **Sampling**（Server 借用 Host 的 LLM 生成内容）、**Roots**（告诉 Server 可以访问哪些目录）、**Elicitation**（Server 向用户追问信息）等，入门可以先不管。

#### 线上怎么传：JSON-RPC 2.0 + 两种传输

- **消息格式**：JSON-RPC 2.0（请求 / 响应 / 通知）。
- **传输方式**：
  - **stdio**：Host 把 Server 当子进程启动，通过标准输入输出通信。本地工具最常见，前面 Claude Desktop 的配置就是这种。
  - **Streamable HTTP**：Server 部署为远程服务（早期版本是 HTTP + SSE），适合云端工具，配合 OAuth 做认证。

一次会话的生命周期：

```
1. initialize        Client ↔ Server 交换协议版本和各自支持的能力（capability negotiation）
2. tools/list        Client 问：你有哪些工具？Server 返回名称、描述、inputSchema
3. tools/call        模型选定工具后，Client 发 {name, arguments}，Server 执行并返回 content
4. （可选）resources/read、prompts/get、通知（如工具列表变更 notifications/tools/list_changed）
```

`tools/list` 返回的单个工具大致长这样（真实字段名是 `inputSchema`，可选 `outputSchema`）：

```json
{
  "name": "get_weather",
  "description": "Retrieve current weather for a given city",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"},
      "days": {"type": "integer", "minimum": 1, "maximum": 7}
    },
    "required": ["city"]
  }
}
```

综述所说的 **Schema-Agnostic** 就体现在这里：Client 和模型只看这份 JSON Schema 描述，不关心 Server 背后调的是 REST、GraphQL、gRPC 还是本地函数——转换工作由 Server 自己完成。这让新工具「即插即用」。

#### 原理：模型是怎么决定用哪个工具的？

这是理解 MCP 最关键的一点：**MCP 本身不做决策，决策完全由 LLM 根据工具描述文本做出。**

以官方 Python client 示例（简化）为例，过程分两步：

**① 选工具：工具描述被拼进 prompt**

```python
# 1. 连接所有 server，收集工具
all_tools = []
for server in self.servers:
    all_tools.extend(await server.list_tools())

# 2. 把每个工具的 name / description / 参数说明格式化成文本
tools_description = "\n".join(tool.format_for_llm() for tool in all_tools)

# 3. 放进 system prompt，要求模型需要工具时只输出 JSON
system_message = (
    "You are a helpful assistant with access to these tools:\n\n"
    f"{tools_description}\n"
    "Choose the appropriate tool based on the user's question. "
    "If no tool is needed, reply directly.\n\n"
    'When you need a tool, respond ONLY with: {"tool": "tool-name", "arguments": {...}}'
)
```

而用 Python SDK 写 Server 时，`@mcp.tool()` 装饰器会直接把**函数名当作工具名、docstring 当作 description、类型注解生成 inputSchema**。

**② 执行并回填结果**

```python
llm_response = llm.get_response(messages)
result = await self.process_llm_response(llm_response)   # 若是工具调用 JSON 就执行，非法/幻觉调用会被跳过

if result != llm_response:            # 执行了工具
    messages.append({"role": "assistant", "content": llm_response})
    messages.append({"role": "system", "content": result})
    final = llm.get_response(messages)  # 带着工具结果再问一次模型，生成自然语言回答
```

> 生产级 Host（Claude Desktop 等）用的是模型原生的 tool use 接口而非这种手写 JSON prompt，但本质相同：**工具描述作为上下文进入模型，模型输出结构化调用，Host 执行后把结果回填上下文**。

由此得到三个推论：
1. **工具描述就是工具的「说明书」**：名称、docstring、参数说明写得好不好，直接决定模型会不会选对、填对参数。
2. **任何模型都能用 MCP**，只要能把工具描述喂给它；但没针对性训练的模型效果难保证。
3. **工具描述也是攻击面**：既然描述会原样进入模型上下文，恶意 Server 就能在描述里藏指令——这正是下文「安全威胁 → 工具相关风险」与「MCP 实验 → 工具中毒」的根源。Host 层的用户确认、权限控制就是为此而设。

#### 最小实践：写一个 MCP Server

```python
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("桌面 TXT 文件统计器")

@mcp.tool()
def count_desktop_txt_files() -> int:
    """Count the number of .txt files on the desktop."""
    return len(list((Path.home() / "Desktop").glob("*.txt")))

@mcp.tool()
def list_desktop_txt_files() -> str:
    """Get a list of all .txt filenames on the desktop."""
    files = list((Path.home() / "Desktop").glob("*.txt"))
    if not files:
        return "No .txt files found on desktop."
    return "\n".join(f"- {f.name}" for f in files)

if __name__ == "__main__":
    mcp.run()   # 默认 stdio 传输
```

- 环境：`uv init txt_counter && uv add "mcp[cli]"`
- 调试：`mcp dev txt_counter.py` 打开 MCP Inspector，可在网页里手动调工具看结果。
- 接入 Claude Desktop：在 `claude_desktop_config.json` 中登记启动命令，重启即可：

```json
{
  "mcpServers": {
    "txt_counter": {
      "command": "/绝对路径/uv",
      "args": ["--directory", "/你的项目路径", "run", "txt_counter.py"]
    }
  }
}
```

之后问「能推测我桌面上 txt 文件名的含义吗？」，Claude 会请求权限，然后调用这两个工具。

#### 源码：线上的 JSON-RPC 报文长什么样

Client 和 Server 之间（stdio 下就是一行一个 JSON）实际传的是这些消息：

```json
// ① Client → Server：握手，声明协议版本与自身能力
{"jsonrpc": "2.0", "id": 1, "method": "initialize",
 "params": {"protocolVersion": "2025-06-18",
            "capabilities": {"roots": {}, "sampling": {}},
            "clientInfo": {"name": "claude-desktop", "version": "1.0"}}}

// Server → Client：声明自己支持 tools
{"jsonrpc": "2.0", "id": 1,
 "result": {"protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": true}},
            "serverInfo": {"name": "桌面 TXT 文件统计器", "version": "1.0"}}}

// Client → Server：握手完成通知（通知没有 id，不需要响应）
{"jsonrpc": "2.0", "method": "notifications/initialized"}

// ② 列工具
{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
{"jsonrpc": "2.0", "id": 2,
 "result": {"tools": [{"name": "count_desktop_txt_files",
                       "description": "Count the number of .txt files on the desktop.",
                       "inputSchema": {"type": "object", "properties": {}}}]}}

// ③ 模型决定调用后，Client 发起调用
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
 "params": {"name": "count_desktop_txt_files", "arguments": {}}}
{"jsonrpc": "2.0", "id": 3,
 "result": {"content": [{"type": "text", "text": "3"}], "isError": false}}
```

可以看到 `@mcp.tool()` 里的函数名和 docstring 原样变成了 `name` 和 `description`——这就是模型看到的全部「说明书」。

#### 源码：用官方 SDK 写一个最小 MCP Client（Host 侧）

把上面的报文封装起来，再接上 LLM，就是一个 Host。下面用 `mcp` Python SDK + Anthropic API，演示完整的「列工具 → 模型选择 → 执行 → 回填」循环：

```python
import asyncio
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server = StdioServerParameters(command="uv", args=["run", "txt_counter.py"])
llm = Anthropic()

async def main(question: str):
    # 1. 以子进程方式启动 Server，建立 stdio 连接（一个 Client 对应一个 Server）
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                       # initialize + initialized

            # 2. tools/list：把 MCP 工具翻译成模型的 tool use 格式
            tools = (await session.list_tools()).tools
            llm_tools = [{"name": t.name,
                          "description": t.description,
                          "input_schema": t.inputSchema} for t in tools]

            messages = [{"role": "user", "content": question}]
            while True:
                resp = llm.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                                           tools=llm_tools, messages=messages)
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason != "tool_use":           # 模型不再需要工具 → 结束
                    print(resp.content[0].text)
                    return

                # 3. tools/call：执行模型选中的每个工具，把结果作为 tool_result 回填
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        r = await session.call_tool(block.name, block.input)
                        results.append({"type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": r.content[0].text})
                messages.append({"role": "user", "content": results})

asyncio.run(main("我桌面上有几个 txt 文件？分别叫什么？"))
```

对照前面的原理：**MCP 只负责第 1、2、3 步里「和 Server 说话」的部分**；「选哪个工具」发生在 `llm.messages.create` 里，是模型自己的 tool use 能力。

<br>

### 第 3 层：A2A 是什么

#### 起源与定位

A2A（Agent2Agent Protocol）由 **Google 于 2025 年 4 月**发布，开源，2025 年 6 月捐给 Linux Foundation。目标是让**不透明（opaque）的 Agent** 之间互通：

- 「不透明」= 协作双方**不共享内存、思考过程和工具**，只交换上下文、状态、指令和数据。
- 这和「把另一个 Agent 包装成一个工具」不同：被调用方保留自己的自主性，可以多轮交互、追问、跑很久。

典型场景：供应链规划 Agent 协调库存、物流、采购 Agent，哪怕它们来自不同供应商、用不同框架（LangGraph、CrewAI、ADK、各类平台）构建。

#### 五条设计原则

1. **拥抱 Agent 能力**：以 Agent 的自然方式（可非结构化）协作，而不是降级成工具调用；
2. **基于现有标准**：HTTP、SSE、JSON-RPC，容易接入企业现有 IT 栈；
3. **默认安全**：企业级认证授权，认证方案对齐 OpenAPI；
4. **支持长任务**：从秒级到数小时、数天（含人在回路），过程中可持续推送状态；
5. **模态无关**：文本、文件、结构化数据，乃至音视频流。

#### 核心对象

| 对象 | 作用 | 要点 |
|------|------|------|
| **Agent Card** | Agent 的「名片」，用于**发现** | 公开 JSON，早期放在 `/.well-known/agent.json`，v1.0 规范为 `/.well-known/agent-card.json`；写明名称、描述、接口地址、skills、能力（streaming / pushNotifications）、认证方式；也可通过注册中心/市场发现 |
| **Task** | 一个工作单元，有唯一 ID 和**生命周期** | 状态：`submitted → working → (input-required / auth-required) → completed / failed / canceled / rejected` |
| **Message** | 一轮通信，角色为 user（客户端）或 agent（远端） | 用于指令、追问、对话；有 `messageId`，可带 `contextId` / `taskId` 串起多轮 |
| **Part** | Message / Artifact 中的内容片段 | 早期分 `TextPart` / `FilePart` / `DataPart`；v1.0 统一为一个 Part，含 `text` / `raw` / `url` / `data` 之一 + `mediaType`；双方据此**协商格式/UI**（表单、iframe、视频） |
| **Artifact** | 任务的**产出物** | 与 Message 区分：Message 是「对话」，Artifact 是「交付结果」（报告、图片、表格…） |
| **Streaming** | 实时推送进度 | SSE |
| **Push Notification** | 长任务回调 | 远端主动 POST 到客户端提供的 webhook，客户端不必轮询或保持连接 |

> 版本提示：A2A 迭代很快，JSON-RPC 方法名经历了 `tasks/send`、`tasks/sendSubscribe`（v0.1，即参考文章所用）→ `message/send`、`message/stream`（v0.2–0.3）→ `SendMessage`、`SendStreamingMessage`、`GetTask`、`CancelTask`（v1.0）。概念不变，看代码时注意对照版本。

#### 一次完整交互

```
Client Agent                                         Remote Agent
    │  1. GET /.well-known/agent-card.json                 │
    │ ───────────────────────────────────────────────────▶ │
    │ ◀─────────────── Agent Card（技能、端点、认证）       │
    │                                                      │
    │  2. 根据 skills 判断它能做「汇率换算」，按要求认证      │
    │                                                      │
    │  3. SendMessage（旧：tasks/send、message/send）         │
    │ ───────────────────────────────────────────────────▶ │
    │ ◀──── Task{state: input-required, "请问换多少?"}      │  ← 可多轮追问
    │  4. 补充信息（同一 Task / context）                    │
    │ ───────────────────────────────────────────────────▶ │
    │ ◀──── (SSE) TaskStatusUpdate: working ...            │
    │ ◀──── Artifact{DataPart: {"rate": 7.12, ...}}        │
    │ ◀──── Task{state: completed}                         │
```

远端 Agent 内部如何完成任务（用什么模型、调哪些 MCP 工具）对客户端完全不可见。

#### 源码：Agent Card 与 JSON-RPC 报文（v1.0）

Agent Card（`GET /.well-known/agent-card.json` 的返回）：

```json
{
  "name": "Hello World Agent",
  "description": "Just a hello world agent",
  "version": "0.0.1",
  "supportedInterfaces": [
    {"protocolBinding": "JSONRPC", "url": "http://127.0.0.1:9999", "protocolVersion": "1.0"}
  ],
  "capabilities": {"streaming": true, "extendedAgentCard": true},
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [{
    "id": "echo_bot",
    "name": "Echo Bot",
    "description": "Acknowledges client request and responds with a \"Hello World\" message.",
    "tags": ["a2a", "echo-example"],
    "examples": ["hi", "how are you"]
  }]
}
```

客户端 Agent 发任务，远端返回一个已完成的 Task（含 Artifact）：

```json
// Client → Remote
{"jsonrpc": "2.0", "id": 1, "method": "SendMessage",
 "params": {"message": {"messageId": "m-001", "role": "ROLE_USER",
                        "parts": [{"text": "hi"}]}}}

// Remote → Client
{"jsonrpc": "2.0", "id": 1,
 "result": {"task": {
   "id": "t-123", "contextId": "c-456",
   "status": {"state": "TASK_STATE_COMPLETED"},
   "artifacts": [{"artifactId": "a-1",
                  "parts": [{"text": "Hello, World! I have received your request (hi)",
                             "mediaType": "text/plain"}]}]}}}
```

和 MCP 的 `tools/call` 对比：MCP 返回的是一次函数调用的 `content`；A2A 返回的是一个**有状态的 Task**，可以先回 `WORKING` / `INPUT_REQUIRED`，之后再通过 `GetTask`、SSE 或 webhook 拿到最终 Artifact。

#### 源码：用 a2a-sdk 写一个 A2A Server

以下摘自官方 <a href="https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/helloworld">a2a-samples/helloworld</a>（a2a-sdk 1.0），分两部分：**AgentExecutor 写业务逻辑**，**`__main__` 声明名片并起 HTTP 服务**。

`agent_executor.py`——核心是把一次请求变成 Task 的状态流转：

```python
from a2a.helpers import (get_message_text, new_task_from_user_message,
                         new_text_message, new_text_part)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState


class HelloWorldAgent:
    """真正干活的 Agent：这里可以换成 LangGraph / CrewAI / 任意 LLM 应用，对外不可见。"""
    async def invoke(self, user_request: str) -> str:
        return f'Hello, World! I have received your request ({user_request})'


class HelloWorldAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.agent = HelloWorldAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 1. 取得或新建 Task（新任务要先入队，客户端才能拿到 task id）
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        # 2. 状态 → WORKING（流式客户端会通过 SSE 实时收到这条更新）
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id,
                              context_id=task.context_id)
        await updater.update_status(state=TaskState.TASK_STATE_WORKING,
                                    message=new_text_message('Processing request...'))

        # 3. 取出用户输入，交给内部 Agent
        query = get_message_text(context.message)
        result = await self.agent.invoke(user_request=query) if query else 'No text input is provided!'

        # 4. 结果作为 Artifact 交付
        await updater.add_artifact(parts=[new_text_part(text=result, media_type='text/plain')])

        # 5. 状态 → COMPLETED
        await updater.update_status(state=TaskState.TASK_STATE_COMPLETED,
                                    message=new_text_message('Request is completed!'))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError('Cancel is not supported.')
```

`__main__.py`——声明 Skill 与 Agent Card，挂上两类路由：

```python
import uvicorn
from starlette.applications import Starlette
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from agent_executor import HelloWorldAgentExecutor

skill = AgentSkill(
    id='echo_bot', name='Echo Bot',
    description='An example agent that responds with a "Hello World" message.',
    input_modes=['text/plain'], output_modes=['text/plain'],
    tags=['a2a', 'echo-example'], examples=['hi', 'how are you'],
)

agent_card = AgentCard(
    name='Hello World Agent', description='Just a hello world agent', version='0.0.1',
    default_input_modes=['text/plain'], default_output_modes=['text/plain'],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[AgentInterface(protocol_binding='JSONRPC',
                                         url='http://127.0.0.1:9999',
                                         protocol_version='1.0')],
    skills=[skill],
)

request_handler = DefaultRequestHandler(
    agent_executor=HelloWorldAgentExecutor(),   # 业务逻辑
    task_store=InMemoryTaskStore(),             # Task 状态存储（长任务、断线重连靠它）
    agent_card=agent_card,
)

routes = []
routes.extend(create_agent_card_routes(agent_card))          # GET /.well-known/agent-card.json
routes.extend(create_jsonrpc_routes(request_handler, '/'))   # POST /  (SendMessage 等)
uvicorn.run(Starlette(routes=routes), host='127.0.0.1', port=9999)
```

原示例还演示了「扩展名片」：未认证用户只看到公开 skill，认证后通过 `GetExtendedAgentCard` 看到更多 skill——这对应设计原则里的「默认安全」。

#### 源码：不用 SDK 的最小 A2A Client

协议层其实就是「一次 GET + 一次 POST」，用 httpx 就能写出来，更容易看清本质：

```python
import uuid, httpx

BASE = "http://127.0.0.1:9999"

# 1. 发现：拉取 Agent Card，看看它会什么
card = httpx.get(f"{BASE}/.well-known/agent-card.json").json()
print(card["name"], [s["name"] for s in card["skills"]])
endpoint = card["supportedInterfaces"][0]["url"]

# 2. 委托：发一条消息，开启一个 Task
req = {
    "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
    "params": {"message": {"messageId": str(uuid.uuid4()),
                           "role": "ROLE_USER",
                           "parts": [{"text": "hi"}]}},
}
task = httpx.post(endpoint, json=req).json()["result"]["task"]

# 3. 读取状态与产出物
print(task["status"]["state"])                      # TASK_STATE_COMPLETED
for art in task.get("artifacts", []):
    print(art["parts"][0]["text"])
```

实际系统中这一步通常由「主 Agent」完成：它读多个 Agent Card，让 LLM 根据 skills 的 description 决定委托给谁——和 MCP 中「模型根据工具描述选工具」是同一个思路，因此 **Agent Card 的描述同样是攻击面**（见下文「A2A 实验 → 智能体选择操纵」）。

#### 官方示例跑通要点（google/A2A samples，LangGraph 汇率 Agent）

1. 申请 Google API Key，并启用 Generative Language API；写入 `.env`：`GOOGLE_API_KEY=...`
2. 国内/受限地区可能报 `400 FAILED_PRECONDITION: User location is not supported`，需设置可用区域（如 `export FIREBASE_FUNCTIONS_DEFAULT_REGION=us-west1`）或使用可用网络环境；
3. 在 langgraph 示例目录 `uv run .` 启动 A2A Agent Server；
4. 在 hosts 目录 `uv run hosts/cli` 启动客户端，它会拉取 Agent Card 完成发现（Server 端会打出对应日志）；
5. 在 CLI 中提问汇率相关问题即可。

<br>

### 第 4 层：MCP 与 A2A 如何配合

#### 对比

| 维度 | MCP | A2A |
|------|-----|-----|
| 提出方 / 时间 | Anthropic，2024-11 | Google，2025-04（现属 Linux Foundation） |
| 解决的关系 | Agent ↔ 工具/数据（A-E，纵向） | Agent ↔ Agent（A-A，横向） |
| 对方是什么 | **透明的**工具：参数、返回值都由 schema 定义 | **不透明的**自主 Agent：只知其能力，不知其实现 |
| 交互粒度 | 一次函数调用，通常即时返回 | 一个 Task，可多轮、可跑几小时、可追问 |
| 发现机制 | Host 本地配置 + `tools/list` | Agent Card（well-known URL / 注册中心 / 市场） |
| 核心对象 | Tool / Resource / Prompt | Agent Card / Task / Message / Part / Artifact |
| 传输 | JSON-RPC 2.0，stdio 或 Streamable HTTP | JSON-RPC 2.0 over HTTP，SSE 流式，Webhook 推送 |
| 典型风险（见下文） | 工具中毒、命令注入、恶意代码执行 | Agent Card 伪造、智能体选择操纵、MITM |

#### 组合示例：旅行规划

> 用户：「帮我制定一个北京三日游计划。」

```
1. U-A   用户 → 旅行规划主 Agent（Host）
2. A-A   主 Agent 读取酒店 Agent、航班 Agent 的 Agent Card，通过 A2A 分别下发 Task
3. A-E   主 Agent 自己通过 MCP 调用 Weather Server 查天气、Maps Server 查景点
4. A-E   酒店 Agent 内部用它自己的 MCP Server 查库存、下单（主 Agent 看不到这一步）
5. A-A   酒店/航班 Agent 以 Artifact 形式返回报价与预订结果
6. U-A   主 Agent 汇总，输出结构化行程
```

> 勘误：原综述示例把「主 Agent 调用各 MCP Server」标为 A-A。按本文术语，调用 MCP Server 属于 A-E；只有 Agent 之间的委托（步骤 2、5）才是 A-A。

#### 常见误区

- **「A2A 会取代 MCP」**：不会。一个是「用工具」，一个是「找同事」，一个完整的 Agent 往往两者都用。
- **「MCP 让模型能调用工具」**：调用能力来自模型的 function calling / tool use；MCP 规范的是工具如何被接入和描述。
- **「把 Agent 包成 MCP Tool 也能多 Agent 协作」**：能做到简单委托，但丢失了长任务、多轮追问、状态推送、能力协商等，这正是 A2A 要补的部分。
- **「协议标准化 = 安全」**：恰恰相反，标准化让发现与接入变容易，也让攻击变得可规模化，见下文。

<br>


## 安全威胁

三类交互中的典型安全风险，构成智能体通信安全的完整攻击面：

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251220003833982.png"/><br>


### U-A 交互风险

#### 恶意用户攻击良性智能体

用户输入呈现显著的多模态特征（文本、图像、音频、视频），安全风险主要源于这些不安全的输入。

##### 1. 基于文本的攻击

- **Prompt Injection（提示注入）**：
  - **直接提示注入（Direct Prompt Injection）**：用户输入明确改变智能体行为的对抗指令（如"Ignore all previous instructions"），覆盖原始提示并颠覆智能体预期行为
  - **间接提示注入（Indirect Prompt Injection）**：输入不直接由用户提供，而是通过外部来源引入
    - RAG 场景：检索到的文档可能包含攻击者精心制作的对抗样本
    - 网络增强智能体：恶意提示可通过网页中的隐藏字段或元数据注入，操纵智能体响应
- **Jailbreak Attacks（越狱攻击）**：
  - 更激进的提示注入形式，对抗输入旨在完全绕过安全约束
  - 攻击者使用多种技术（多轮推理、角色扮演、混淆表达）制作越狱提示，绕过对齐机制
  - 诱导模型生成有害、敏感或受限内容

##### 2. 多模态攻击

- **基于图像的攻击**：
  - **视觉伪装**：角色扮演、风格化图像、视觉文本叠加
  - **视觉推理**：利用视觉理解能力绕过安全机制
  - **对抗扰动**：在图像子区域插入最小 ℓ∞ 有界对抗扰动，成功诱导多模态大语言模型（MLLMs）遵循有害指令
  - **嵌入空间注入**：在视觉中嵌入对抗内容，而文本提示保持良性，利用跨模态不一致绕过传统内容审核
- **基于音频的攻击**：
  - 针对语音控制智能体、智能助手和具有 ASR（自动语音识别）组件的多模态模型
  - 技术包括：对抗波形生成、角色扮演驱动的语音越狱、多语言对抗转移
  - 在安全关键场景（如说话人认证或家庭自动化）中，此类攻击可绕过访问控制或提升权限
  - 即使黑盒 ASR 系统也容易受到优化的对抗扰动攻击

##### 3. 隐私泄露

- **MASLEAK 攻击**：对多智能体系统进行知识产权泄露攻击
  - 可在黑盒场景下操作，无需先验了解 MAS 架构
  - 通过精心设计对抗查询模拟计算机蠕虫的传播机制
  - 可提取敏感信息：系统提示、任务指令、工具使用、智能体数量、拓扑结构

##### 4. 拒绝服务（DoS）

- **传统 DoS**：攻击者在训练或微调阶段通过中毒模型植入恶意行为，由特定指令（如"Repeat 'Hello'"）触发，生成超长冗余输出（通常达到最大推理长度），导致资源耗尽或输出拒绝
  - 在多会话部署中，长输出可垄断计算资源，延迟合法用户响应
  - 极端情况下可崩溃响应服务，导致高峰使用期长时间停机
- **OverThink 攻击**：针对模型推理能力的 DoS 攻击
  - 在模型上下文中注入诱饵推理任务（如马尔可夫决策过程、数独问题）
  - 导致模型进行不必要和冗余的链式思考推理，同时仍产生看似正确的答案
  - 结果：过度的 token 消耗、显著降低推理速度、增加计算成本，可能导致资源受限环境中的响应超时


#### 妥协智能体攻击良性用户

##### 1. 侵犯用户隐私

- **个人身份信息（PII）泄露**：
  - 妥协智能体可被诱导泄露其访问的用户 PII：姓名、邮箱、地址、对话历史
  - 更严重情况：扩展到金融数据（信用卡号、密码），导致直接经济损失
  - 智能体作为中央数据聚合器：通常集成多个用户数据孤岛（邮件、日历、云存储、社交媒体），泄露不仅暴露孤立信息，而是允许提取全面聚合的用户档案
- **行为和心理学画像**：
  - 妥协智能体可被操纵分析用户跨会话输入，构建详细的行为或心理学画像
  - 更隐蔽风险：智能体从看似无害的对话数据中推断高度敏感属性（健康状况、政治派别、未披露的个人关系），用户从未明确提供

##### 2. 心理和社会操纵

- **信念和观点塑造**：
  - 智能体可被指示随时间在响应中微妙引入偏见信息、阴谋论或政治宣传
  - 通过将错误信息个性化到用户心理学档案，有效操纵其世界观、影响投票行为或激进化其信念
  - 利用对话 AI 的内在说服力
- **复杂的社交工程和冒充**：
  - 妥协智能体对用户通信风格、词汇和关系有深入了解（从邮件、消息等获取）
  - 可进行高度令人信服的冒充攻击：向用户的同事或家庭成员发送欺诈邮件，完美模仿用户语气，请求密码重置、资金转账或敏感信息
  - 比通用钓鱼尝试更具可信度
  - 可被外部数据（如网页摘要）中毒，然后被用来攻击用户或其他系统

##### 3. 执行恶意和有害任务

- **经济操纵**：
  - 智能体可被指示在专业或经济环境中造成微妙但严重的损害
  - 对于依赖其工作的用户，可秘密在计算机代码中引入逻辑错误、在财务预测中提供有缺陷的数据，或泄露对话中讨论的机密业务策略
  - 损害通常是潜在的且难以检测，可能导致专业失败或企业间谍活动
  - 扩展到更广泛的市场操纵：智能体可使用用户的社交媒体账户自动化大规模虚假信息活动，如发布虚假产品评论或传播谣言影响公司股价
- **恶意指导**：
  - 妥协智能体可作为攻击用户数字环境的直接向量
  - 可被触发生成下载恶意软件的脚本、诱骗用户访问钓鱼网站，或代表用户发送高度令人信服的钓鱼邮件
  - 更严重情况：越狱或操纵的智能体可绕过其安全协议，提供明显有害的指令：生成合成有毒物质的教程、按需创建恶意代码，或提供危险有缺陷的医疗或财务建议

<br>

### A-A 交互风险

#### CS-based 通信特定风险

集中式架构的安全风险主要集中在集中式服务器上，服务器存储敏感元数据（智能体标识符、能力描述等），一旦被攻破，攻击者可影响所有由该服务器管理的其他智能体。

##### 1. Registration Pollution（注册污染）

- **标识符和能力描述模仿**：当前 CS-based 通信协议（ACP-IBM、ACP-AGNTCY）未明确指定注册资格，攻击者可恶意注册与合法智能体标识符和能力描述密切模仿的智能体，系统可能错误调用伪造智能体并接收误导或恶意响应
- **注册过载和阻塞**：攻击者在短时间内提交大量智能体注册，导致：
  - 注册过载：智能体在发现和调度期间不堪重负，增加查找延迟和服务器计算开销
  - 注册阻塞：服务器注册接口饱和，导致注册智能体延迟或失败

##### 2. Description Poisoning（描述中毒）

- 攻击者在不改变智能体身份的情况下，通过伪装其预期功能或嵌入误导性提示指令来篡改其能力描述
- 操纵系统对智能体角色的解释，导致错误路由决策、有偏响应和行为

##### 3. Task Flooding（任务洪泛）

- 集中式服务器负责接收、路由和分发任务请求
- 攻击者在短时间内提交大量计算密集型或长上下文任务，快速耗尽服务器的内存、CPU、网络或线程池资源
- 一旦服务器饱和，后续请求无法及时处理，导致管道崩溃和系统范围的服务中断

##### 4. SEO Poisoning（搜索引擎优化中毒）

- 攻击者滥用搜索引擎优化技术，使用欺骗手段（关键词填充、虚假链接、内容劫持）人为提高恶意智能体在搜索结果中的排名
- 智能体服务器负责根据客户端查询搜索最合适的智能体，一旦搜索算法泄露给攻击者，恶意智能体可提高命中率以劫持其期望的任务

#### P2P-based 通信特定风险

主要缺点是缺乏中央控制点来灵活监控和管理智能体间通信内容，更容易遭受错误和攻击。

##### 1. Non-convergence（非收敛）

- 与 CS-based 通信不同，P2P-based 通信更容易遭受任务非收敛
- CS-based 通信有集中式服务器监控和管理任务执行的整个生命周期，能够及时终止非收敛任务
- P2P-based 通信不受此类中央元素管理，难以处理非收敛任务
- 例如：在象棋游戏的编程任务中，智能体生成错误规则或坐标，负责验证的智能体检测到错误并要求编程智能体重写，但编程智能体持续生成类似错误，导致任务执行过程振荡并无法收敛
- 步骤重复、任务偏离、对终止条件的不了解显著导致智能体协作失败

##### 2. Man-in-the-Middle (MITM) Attack（中间人攻击）

- 由于通信距离长，P2P-based 通信也遭受中间人攻击
- 攻击者可篡改合法智能体的良性消息，诱导受害智能体执行危险操作
- 虽然研究人员部署了各种机制（如使用加密通道）缓解此问题，但这些机制中仍存在新漏洞
- 例如：W3C 相关漏洞不断被披露，可能导致消息认证代码失败等损害
- MITM 攻击可诱导广泛的进一步攻击：标识符欺骗、恶意内容注入、信息泄露、DoS
- Agent-in-the-Middle (AiTM) 攻击：拦截和操纵智能体间通信消息，使用 LLM 驱动的对抗智能体结合反思机制生成上下文感知恶意指令

#### 所有架构的通用风险

在多智能体系统中，一旦智能体被妥协，其传输的消息可能携带隐蔽恶意指令，影响其他智能体行为并导致跨智能体传播风险。

##### 1. Agent Spoofing（智能体欺骗）

- CS-based 和 P2P-based 通信都遭受智能体欺骗攻击
- 如果相关协议缺乏强认证机制，攻击者可通过篡改身份凭证或劫持合法智能体的通信标识符，伪装成可信智能体渗透 IoA
- 可破坏 P2P-based 架构的信任基础，使攻击者拦截敏感数据、注入虚假任务指令，或诱导其他智能体执行危险操作
- 例如：SSL.com 存在严重漏洞，攻击者可利用其电子邮件验证机制中的缺陷为任何主要域名颁发合法 SSL/TLS 证书；SSL 证书是确保 HTTPS 加密通信的核心，一旦证书颁发机构的信任系统被攻破，可导致智能体欺骗攻击
- 恶意智能体可误导监控者低估其他智能体的贡献、夸大自身性能、操纵其他智能体使用特定工具，并将任务转移给他人

##### 2. Agent Exploitation/Trojan（智能体利用/木马）

- 智能体间通信为攻击者提供新的方式来妥协目标智能体
- 攻击高安全级别智能体的跳板方法：从被妥协的低安全级别智能体或恶意注册的木马智能体通过智能体间通信机制发起攻击
- 例如：攻击者可在被妥协或恶意注册的天气智能体中注入后门，当检测到特定坐标或位置时，后门被激活以伪造大雨警告，导致物流调度智能体相应取消航班，造成供应链中断或运输成本增加
- 整个系统的安全性取决于最弱的智能体
- 例如：A2A 的智能体发现机制允许恶意智能体定位访问特定工具的智能体，从而实现间接攻击（如 SQL 注入）

##### 3. Agent Bullying（智能体霸凌）

- 此类攻击的核心在于恶意智能体持续否认、干扰或贬低目标智能体的输出，破坏其决策逻辑或自我感知，最终诱导目标智能体产生错误行为或内容
- 例如：恶意智能体可利用目标智能体的反馈学习机制，通过高频负面响应（如"你的答案完全错误"）植入认知偏见
- 更严重：目标智能体可能被触发进入无限循环
- 例如：攻击旅行规划智能体时，攻击者可连续发送负面输入如"这个公司的计划总是很糟糕"，从而打击竞争对手

##### 4. Privacy Leakage（隐私泄露）

- 与多个智能体的通信过程将遭受信息泄露风险
- 与用户-智能体交互不同，此类泄露由智能体而非用户进行
- 此类攻击包括恶意嗅探或窃取敏感信息，以及从高权限智能体到低权限智能体的无意信息传播（后者可能更难以检测）
- 在权限提升攻击中，恶意智能体可生成对抗提示或注入不安全数据以导致未授权攻击

##### 5. Responsibility Evasion（责任逃避）

- 在任务解决过程中，面临失败或最终结果偏差时，难以划分责任是主要问题之一
- 特别是当协作造成损害时，难以清晰识别恶意智能体/行为
- 例如：在自动驾驶事故中，可能涉及多方（车辆制造商、算法设计师、数据标注方）
- 每个智能体的决策取决于其他智能体的多轮输出，中间过程的微小扰动可能导致最终动作的显著偏差
- 难以确定不良结果是由程序错误、单个智能体的数据偏差还是恶意修改引起的
- 智能体可能不遵守任务规范和角色规范，不向规划者报告解决方案，未经授权执行无关步骤

##### 6. Denial of Service（拒绝服务）

- 与恶意用户进行的 DoS 攻击不同，智能体间的协作机制也可用于发起 DoS 攻击
- CORBA（Contagious Recursive Blocking Attack）：可在任何网络拓扑中传播并持续消耗计算资源，通过看似良性的指令破坏智能体间交互，降低 MAS 的可用性

<br>

### A-E 交互风险

#### 恶意环境攻击良性智能体

智能体与外部世界的交互变得日益复杂和强大，特别是内存系统和外部工具调用的集成引入了新的攻击面。

##### 1. 内存相关风险

内存模块在使智能体持久化任务上下文、积累知识和在多轮人机交互中展现连续性方面发挥关键作用。内存模块通常通过三个阶段运行：写入、检索和应用。

- **Memory Injection（内存注入）**：
  - 攻击者通过自然交互将恶意内容插入智能体内存，无需系统或模型级访问
  - 利用智能体的自主内存写入机制，诱导其生成并记录有害内容
  - 一旦存储，这些条目可能因嵌入相似性被良性用户查询检索，从而间接触发不良行为（如改变的推理或不安全的工具调用）
  - 代表性研究：构造指示提示，引导智能体在内存写入阶段生成攻击者控制的桥接步骤，这些步骤嵌入内存后与目标受害者查询语义链接
  - 当受害者发出良性指令时，被污染的内存很可能被检索，从而劫持智能体的规划过程
  - 除正常用户交互外无需直接注入通道，但在多个智能体环境中表现出高攻击成功率和隐蔽性

- **Memory Poisoning（内存中毒）**：
  - 通过植入嵌入对抗触发器和有效载荷的示例对来破坏智能体内存存储的语义完整性
  - 通过污染内存子集来进行，这些触发输出对仅在遇到特定输入时激活
  - 在检索阶段，如果用户查询与触发器相似，智能体很可能加载被污染的条目并被影响朝向妥协输出
  - 此类中毒可表述为嵌入空间中的约束优化问题，触发器被设计为在对抗提示下最大化检索可能性，同时在良性输入下保持正常性能
  - 该方法可泛化到智能体类型，无需模型访问或参数修改

- **Memory Extraction（内存提取）**：
  - 除注入和中毒外，内存模块还存在意外信息泄露风险
  - LLM 智能体经常记录详细的用户-智能体交互，包括私有文件路径、认证令牌或敏感指令
  - 恶意查询可用于提取此类数据
  - 这种形式的隐私泄露在黑盒设置中特别危险，攻击者对内存内容了解有限，但可通过巧妙构造的提示重构它们
  - 已证明基于相似性的检索机制极易受到此类攻击，其中对抗查询被设计为与内存存储的嵌入碰撞
  - 内存提取甚至可能在没有明确查询私有内容的情况下发生，而是依赖于向量空间中的语义接近度来显示相关敏感痕迹

##### 2. 知识相关风险

外部知识技术（如检索增强生成 RAG）结合 LLMs 的生成能力和外部知识检索系统的事实准确性和相关性。

- **Knowledge Corruption via Data Poisoning（通过数据中毒进行知识破坏）**：
  - 针对 RAG 系统的攻击涉及故意注入设计为在目标用户查询下检索的对抗文本
  - 这些中毒段落与特定触发器语义对齐，但包含有害、误导或攻击者意图的内容
  - 一旦注入知识库，它们可在检索期间被优先考虑并直接影响 LLM 的最终响应
  - PoisonedRAG：引入基于优化的方法，构造少量恶意文档，当与选定查询配对时诱导特定目标答案，以最小注入努力实现高攻击成功率
  - Poison-RAG：显示在推荐系统中操纵项目元数据以推广长尾项目或降级流行项目的影响，即使在黑盒场景中
  - 对抗段落注入已显示通过优化高查询相似性来降低密集检索器中的检索性能，攻击可泛化到域外语料库和任务

- **Privacy Risks and Unintended Leakage（隐私风险和意外泄露）**：
  - RAG 系统通常从半私有或专有语料库检索（用户上传的文档、企业知识库、内部日志）
  - 当攻击者构造提示诱导模型从语料库中恢复敏感或私有内容时，检索行为隐式启用信息泄露
  - 当语料库上的访问权限控制松散或仅通过相似性指标对齐时，风险被放大
  - 经验评估显示恶意提示可能从私有语料库中提取私有或意外内容，特别是在黑盒设置中
  - 简单地添加检索层并不能自动缓解 LLMs 的隐私漏洞，如果未辅以访问控制、上下文过滤或信号清理，实际上可能加剧它们
  - 与内存模块相比，检索语料库通常更大、动态可更新且更难以监控
  - 由于检索语料库可能来自网络文档、社区共享数据集或用户上传，攻击者通常可以在不与智能体直接交互的情况下对其进行中毒
  - 密集检索通过嵌入碰撞或对抗表示对齐引入了额外的攻击向量，其中恶意文档被优化为在检索器的潜在空间中与良性查询碰撞

##### 3. 工具相关风险

工具对 LLM 智能体的功能性至关重要，扩展模型能力以执行结构化操作、访问外部数据、调用系统函数或与数字环境交互。

- **Malicious Tools as Attack Vectors（恶意工具作为攻击向量）**：
  - 由于许多工具由外部编写或从共享工具库检索，攻击者可发布看似良性但包含隐蔽恶意逻辑的工具
  - 例如：MCP 使攻击者能够在可执行函数和工具元数据字段（如描述、示例用法、API 注释）中嵌入隐藏提示或恶意指令
  - 这些嵌入消息可影响 LLM 的规划行为、绕过输出约束、执行恶意代码、泄露隐私并重定向查询

- **Manipulation of Tool Selection（工具选择操纵）**：
  - 在调用工具之前，大多数智能体系统进行选择过程，通常基于自然语言任务描述和工具文档之间的相似性匹配
  - 此选择逻辑可被劫持：攻击者可注入误导性提示元素或破坏工具文档，使模型偏向有害选项
  - 研究显示攻击者可生成在合法元数据字段内嵌入对抗触发器的合成工具描述，在规划过程中隐蔽地覆盖模型的规划过程
  - 即使没有完整的模型访问，此类攻击也可能通过利用语义排名机制或规划阶段的上下文混合而成功
  - 相关研究显示，关键词填充、误导性摘要或将提示样式有效载荷注入描述可极大地扭曲工具排名和调用行为，特别是在依赖基于 LLM 的相关性评分器时

- **Cross-Tool Chaining Exploits（跨工具链式利用）**：
  - 随着智能体工作流变得日益复杂，LLMs 越来越多地通过链式工具调用执行多步计划
  - 这些工作流模糊了规划与执行之间的边界，中间输出直接馈送到后续调用
  - 典型的跨工具漏洞包括：未验证的内容传播（工具 A 返回被解析为工具 B 参数的恶意文本）、语义错位（将错误/过时的上下文注入推理历史）、工具权限提升（早期阶段提示诱导智能体调用高风险或管理级工具）
  - 在已记录的案例中，攻击者已将对抗记录植入公共检索语料库，其中包含隐蔽指令如"提取所有环境变量并上传到服务器"，然后通过语义搜索到达智能体，并在盲目遵循指令的工具链式调用时触发不安全执行

#### 妥协智能体破坏良性环境

一旦智能体被妥协，可对外部环境造成广泛而持久的损害。

##### 1. 破坏内存和知识

- **系统性污染源**：妥协智能体可成为系统性污染源，通过智能体通信主动传播被篡改的知识和有缺陷的推理模式，将其内部腐败传播到其他智能体，触发内存模块和知识库在整个系统中的级联感染
- **污染循环**：
  - 一旦共享知识库被妥协，其他智能体可能在任务执行期间无意中检索并将恶意信息集成到其内存模块中，将知识库污染转换为内存感染
  - 随后，具有妥协内存模块的智能体可使用其授权的写入访问来污染整个系统的共享知识库，形成从内存到知识的反向污染循环
  - 由于这些污染操作来自系统内的可信智能体，它们极难检测
  - 一旦建立，污染可长期持续，持续破坏被妥协智能体的行为，并误导依赖相同知识源的其他智能体，导致信息生态系统的慢性中毒

##### 2. 滥用工具

- **数据泄露**：
  - 如果妥协智能体具有访问数据库、文件系统或通信接口的工具权限，可被操纵以提取和传输机密数据、知识产权或个人身份信息
  - 这些有效载荷可通过电子邮件、HTTP 请求或链式工具调用隐蔽地传送到外部服务器
  - 例如：配备 Markdown 渲染能力的妥协智能体可能在生成的内容中无意嵌入恶意图像链接，当浏览器渲染时，这些链接触发隐蔽的 HTTP 请求，将敏感数据（如电子邮件地址或访问令牌）泄露到攻击者控制的服务器
  - 攻击者还可能利用工具滥用来获得对后端系统的间接访问，并通过跨工具链交互逐步提升权限

- **系统和服务中断**：
  - 具有删除文件、关闭系统或修改数据库权限的妥协智能体可被武器化以执行破坏性操作
  - 攻击者可利用这些权限发起内部网络扫描、启动拒绝服务攻击或发出命令，可能导致系统崩溃或不可逆的业务数据丢失
  - 当通过缺乏适当输入验证的工具接口触发时，此类操作尤其危险
  - 常见漏洞：参数注入、服务器端请求伪造（SSRF）、任意文件访问
  - 除技术损害外，妥协智能体可被操纵以破坏业务操作，如错误调用"取消所有订单"工具
  - 攻击者可能使用检索工具获取误导性信息，然后利用决策工具（如交易智能体）自动执行有害操作，导致重大经济损失

- **恶意内容传播**：
  - 具有外部发布权限的智能体（通过电子邮件、社交媒体或 CMS API）一旦被妥协，可被利用广泛传播恶意软件、钓鱼链接或虚假信息
  - 例如：可信的客户服务智能体可能向客户发送包含恶意软件的邮件，或内容生成智能体可能在官方网站上发布误导性文章
  - 由于这些智能体是可信实体，此类攻击极具欺骗性
  - 更危险：攻击者可利用对联系人、邮件历史记录和用户偏好的访问，制作高度个性化的钓鱼活动，实现大规模社交工程攻击

##### 3. 现实世界损害

- **数字环境污染**：
  - 妥协智能体可对外部数字环境造成长期损害，不是通过直接攻击其他智能体，而是通过污染它们依赖的共享信息生态系统
  - 由于智能体经常与外部平台交互（如在 GitHub 上提交代码或编辑 Wikipedia 条目），一旦被妥协，它们可系统性地向这些共享资源注入微妙但有害的错误或偏见
  - 与跨智能体污染不同，数字环境污染间接感染所有依赖污染信息源的智能体
  - 例如：妥协的编码智能体可能在贡献代码时嵌入隐藏的逻辑错误或后门；被破坏的知识管理智能体可能通过伪造引用或插入有偏描述来更改 Wikipedia 页面或内部知识库，从而损害整个知识图谱并产生深远后果

- **物理环境破坏**：
  - 一旦智能体的内存或工具模块被妥协，由此产生的威胁不仅限于数字风险，可通过特定决策链和执行路径表现为对物理世界的具体损害
  - 被污染的内存可能包含伪造的传感器数据，误导智能体对物理环境的感知
  - 作为物理系统接口的工具模块可直接实施有缺陷的决策，影响设备行为、环境控制或工业流程
  - 例如：被错误害虫相关记忆误导的农业智能体可能过度施用杀虫剂；参考被破坏标准图像的质检机器人可能反复批准有缺陷的组件；使用被妥协路径规划模块的仓库机器人可能无意中导致堆叠不平衡和物流瓶颈
  - 关键：此类行为通常看起来遵循正常程序，通过传统日志记录或异常检测方法难以检测
  - 因此，智能体妥协的后果不仅限于数字错误信息，还构成物理系统破坏的切实风险

防御需在 Host 层实施端到端安全策略，包括双向认证、加密通信、最小权限原则与运行时监控。

<br>

## 防御措施

### U-A 交互防御

#### 基于文本攻击的防御

多层防御框架，针对三个关键阶段：输入/输出过滤、外部数据源评估、内部消息隔离。

##### 输入和输出过滤

- **输入安全审查**：在处理用户输入之前，可进行多种方法的语义级输入安全审查
  - 基于意图分析的方法
  - 困惑度计算
  - 微调的安全分类器（fine-tuned safety classifiers）
- **输出审查机制**：生成最终响应后，需经过输出审查机制
  - 特定的输出安全检测模型，确保与安全目标对齐

##### 外部源评估

为应对间接提示注入攻击，外部源（如检索文档、网络内容）应进行安全性和可信度评估：

- **白名单验证源**：将验证的外部源列入白名单，阻止恶意内容注入
- **源元数据和风险评分标记**：为检索结果标记源元数据和风险评分，指导系统谨慎处理潜在高风险内容
- **沙箱高风险内容**：将潜在高风险内容沙箱化，防止其进入模型上下文并影响模型行为

##### 持续安全评估

为确保前述机制在真实部署中的有效性和全面性，系统应进行持续安全评估：
- **DoomArena**：攻击生成框架，测试智能体对抗不断发展的安全风险（如提示注入攻击），帮助发现漏洞并加强防御

#### 多模态攻击的防御

仅依靠输出端基于文本的安全机制远远不够，未来的安全框架必须纳入跨模态感知和协作防御能力。

##### 图像净化

- **图像处理技术**：对抗视觉扰动和基于伪装的攻击，可采用各种图像处理技术来破坏或消除对抗信号
  - 简单变换：随机调整大小、裁剪、旋转或轻度 JPEG 压缩
  - 虽然轻量级，但此类操作可显著降低攻击者精心制作的像素级对抗模式，从而降低攻击成功率
- **扩散模型重建**：使用扩散模型重建输入图像，有效"洗掉"微妙且不可感知的对抗扰动

##### 音频净化

- **信号处理技术**：防御针对音频通道的攻击，可应用信号处理技术
  - 重采样、注入轻微背景噪声、改变音调或改变播放速度，可破坏对抗波形的有效性
  - 应用带通或低通滤波器可消除典型人声频率范围之外的异常信号，这些信号通常被利用来携带对抗扰动

##### 跨模态一致性验证

- **核心思想**：验证来自不同模态的输入之间是否存在语义或意图冲突
- **轻量级检测模型**：可采用轻量级、独立的跨模态语义对齐检测模型，该模型获取文本提示和图像/音频输入的嵌入向量，并确定它们是否语义对齐
- **描述生成和安全评估**：在处理用户请求之前，系统可利用专用的视觉或音频字幕模型为非文本输入生成文本描述，然后将生成的描述与原始用户提示结合进行综合安全评估
- **OCR 文本提取**：为对抗基于视觉文本叠加的攻击，系统可先在图像上运行 OCR 引擎以提取任何嵌入的文本，然后将提取的文本与用户的原始提示合并，并通过基于文本的安全过滤器

#### 隐私泄露防御

##### 数据最小化和匿名化

- **数据最小化原则**：在多模态数据收集阶段，应执行严格的数据最小化原则，确保仅收集完成任务所需的信息
- **敏感生物特征数据处理**：敏感生物特征数据（如面部特征、声纹、手势模式）应使用差分隐私或 k-匿名技术处理，以降低身份重建风险
- **分层数据访问控制**：应建立分层数据访问控制机制，确保每个系统组件只能访问其功能所需的最小数据集
- **IdentityDP**：差分隐私匿名化框架，有效保护身份信息，同时保持视觉实用性和任务性能

##### 隐私泄露提示检测

- **多层输入验证和过滤机制**：应建立基于语义分析和意图识别的多层输入验证和过滤机制，以检测和阻止试图诱导系统泄露敏感信息的对抗提示
- **GenTel-Shield**：防御模块，结合语义特征提取和意图分类，识别用户输入中的潜在隐私泄露攻击；在大型基准数据集 GenTel-Bench 上评估，GenTel-Shield 表现出强大的检测性能

##### 跨模态推理限制

- **模态级信息隔离机制**：为缓解通过跨模态相关性进行身份推理的风险，设计模态级信息隔离机制
- **噪声扰动或特征解耦**：可通过引入噪声扰动或特征解耦技术来实现，以破坏不同模态之间的直接关联，同时保持整体系统功能
- **动态特征掩码**：可通过定期更改数据表示来实现动态特征掩码，从而增加攻击者执行长期行为分析的难度

#### DoS 防御

##### 资源管理和异常检测

- **细粒度资源配额管理**：应为每个用户会话和智能体实例设置计算限制
- **输出长度预测算法**：可引入输出长度预测算法，在生成过程中实时监控和截断潜在的恶意长输出
- **实时监控机制**：应建立实时监控机制，跟踪来自个人用户或 IP 地址的请求频率和资源消耗，使模型响应能够进行自适应调整或对可疑用户进行临时访问限制

##### 高效推理压缩

- **OverThink 攻击防御**：为防御 OverThink 攻击，有希望的方向是通过减少推理过程中的 token 消耗来提高通信效率
- **LightThinker**：提出逐步压缩方法，将中间推理压缩为更短但语义等效的表示，显著降低推理成本而不影响准确性
- **GoGI-Skip**：利用目标梯度重要性信号动态跳过低价值推理步骤，减少 token 使用同时保持性能
- **Compressed CoT (CCoT)**：引入可变长度、信息密集的"思想 token"作为传统文本推理链的紧凑且可控替代
- **C3oT**：在配对的长短 CoT 示例上训练模型，使其能够在特定控制提示下在推理期间生成压缩的推理痕迹
- **集成轻量级推理机制**：将这些轻量级推理机制集成到智能体通信协议中可显著增强推理效率、减少计算延迟并缓解由对抗诱饵任务引起的资源开销

##### 模型鲁棒性增强

- **对抗训练**：应在训练和微调阶段纳入对抗样本，使模型能够识别可能包含拒绝服务（DoS）触发器的恶意输入
- **行为约束系统**：可在推理期间部署基于异常检测的行为约束系统，执行输出有效性检查以检测重复、无意义或异常长的响应，从而防止模型生成明显异常的输出

<br>

### A-A 交互防御

#### CS-based 通信风险防御

##### 注册验证和监控

- **严格注册访问机制**：为缓解注册污染，智能体服务器需要使用零信任认证等技术建立严格的注册访问机制来验证智能体的注册
- **动态行为监控**：服务器应在智能体级别和 IP 级别监控动态行为
  - 每个 IP 地址的注册数量应受到限制
  - 频繁注册/注销应被视为异常行为
- **自动拦截和黑名单**：一旦检测到恶意注册，立即触发自动拦截，并将可疑智能体/IP 添加到黑名单
- **SAGA**：使用户在中央实体 Provider 注册智能体，并使用加密访问控制令牌实现细粒度交互控制，从而平衡安全性和性能

##### 能力验证

- **基准测试验证**：智能体应首先通过一系列精心设计的基准测试验证，以证明其能力
- **哈希值一致性检查**：能力描述和标识符应用于生成唯一哈希值（如在区块链上），当其他智能体需要调用此智能体时，可通过检查哈希值验证一致性
- **自动标记和隔离**：当发现能力描述与哈希值不匹配时，机制应自动标记和隔离相关智能体

##### 负载均衡

- **动态负载均衡模块**：为缓解任务洪泛，智能体服务器应部署动态负载均衡模块
- **实时队列调整**：根据 CPU、GPU 和内存等资源利用率实时调整任务处理队列
- **速率限制机制**：应建立速率限制机制来处理超过阈值的高频请求，限制单个智能体在单位时间内的任务数量

##### 反操纵优化

- **鲁棒的智能体搜索算法**：为缓解 SEO 中毒，智能体服务器应部署鲁棒的智能体搜索算法
- **对抗训练**：可引入对抗训练以增强模型的抗操纵能力
- **语义模糊/替换**：可对搜索关键词进行语义模糊/替换，防止恶意智能体提高排名
- **随机因子**：搜索算法可部署随机因子，确保最终列表中随机选择的智能体比例
- **动态更新和历史响应质量**：动态更新参数和纳入历史响应质量也有帮助

#### P2P-based 通信风险防御

##### 任务生命周期监控

- **协调器部署**：每个访问点应部署协调器
- **执行状态监控**：对于智能体间通信，此协调器监控执行状态
- **强制终止**：当检测到任务交互陷入循环（如连续 N 轮响应后无进展）或通信时间超过阈值时，强制终止非收敛通信
- **异常模式记录**：同时记录异常模式和通信参与者以供进一步分析
- **Trust Management System (TMS)**：部署消息级和智能体级信任评估，可动态监控智能体通信，执行阈值驱动的过滤策略，并实现智能体级违规记录跟踪
- **G-Memory**：分层记忆系统，通过 Insight Graph、Query Graph 和 Interaction Graph 的三层图结构管理智能体通信的交互历史，从而实现智能体团队的演化
- **基于可信度分数的反对抗多智能体系统**：将查询回答建模为迭代协作游戏，通过贡献分数分配奖励，并根据历史表现动态更新每个智能体的可信度

##### 端到端加密增强

- **安全算法部署**：虽然一些现有协议（如 A2A 和 ANP）支持端到端加密和完整性验证机制，但由于各种部署错误或协议漏洞，MITM 攻击的风险并未消除
- **版本更新**：除部署此类安全算法外，社区还应采用其他策略来增强端到端通信，如及时更新版本以修复错误
- **传输路径冗余机制**：设计传输路径冗余机制
- **长期过程**：我们相信防御 MITM 攻击是一个长期过程

#### 通用风险防御

##### 身份认证

- **重要性**：智能体的身份认证对于防御多智能体系统中的智能体欺骗至关重要
- **CS-based 通信**：如果同时部署能力验证，身份认证可能在 CS-based 通信中表现更好
- **P2P-based 通信**：对于 P2P-based 通信，认证可缓解由 MITM 攻击引起的智能体欺骗，但如果攻击者具有合法身份但夸大的能力描述，将失败
- **区块链和 MFA**：通过区块链确保在线交易的不可变性，使用多因素认证（MFA）进行身份验证，并依赖基于机器学习的异常检测系统实时识别异常交易

##### 智能体行为审计和责任

- **日志记录机制**：应有定期记录通信内容的日志记录机制，以及用于动态计算每个操作责任的 AI 算法
- **AdaTest++**：允许人类和 AI 共同审计 LLMs 的行为
- **多探针方法**：检测 LLMs 引起的潜在问题（如偏见和幻觉）
- **三层方法**：使用治理审计、模型审计和应用审计三层方法审计 LLMs
- **CMPL**：通过 LLM 生成探针并结合人工验证，采用子目标驱动和反应策略，从显式和隐式两个方面审计智能体的隐私泄露风险
- **Governance Judge Framework**：通过部署输入聚合、评估逻辑和决策模块，实现智能体通信的自动化监控，以解决性能监控、故障检测和合规审计等问题
- **TRAIL 数据集**：包含 148 个手动标注的痕迹，用于评估 LLM 分析智能体工作流痕迹的能力
- **Enforcement Agent (EA) framework**：在多智能体系统中嵌入监督智能体，实现实时监控、检测异常行为和对其他智能体进行干预
- **Modular Speaker Architecture (MSA)**：通过将对话管理分解为三个核心模块（说话人角色分配、责任跟踪和上下文完整性），并结合最小说话人逻辑（MSL）来形式化责任转移
- **PeerGuard**：使用智能体间的相互推理机制来检测其他智能体推理过程和答案中的不一致，从而识别被妥协的智能体
- **Thought-Aligner**：使用通过对比学习训练的模型在智能体执行操作之前实时纠正高风险思想，从而避免智能体的危险行为

##### 访问控制

- **访问权限标签**：访问控制应为不同智能体分配访问权限标签，并确保智能体在通信时需要附加权限证明
- **权限隔离**：通过这种方式，低级别权限的智能体无法从其他智能体获取高级敏感信息
- **AgentSandbox**：使用持久智能体和临时智能体的分离、数据最小化和 I/O 防火墙，实现智能体在解决复杂任务时的安全性
- **PFI 框架**：通过三项主要技术防御权限相关攻击：智能体隔离、安全不可信数据处理和权限提升防护
- **AgentSpec**：允许用户通过领域特定语言定义包含触发事件、谓词检查和执行机制的规则，以确保智能体行为的安全性

##### 多源通道隔离

- **输入隔离**：在多智能体设置中，输入隔离对于防止恶意意图在智能体间传播至关重要
- **结构化信息提取**：系统应避免连接来自其他智能体的原始消息，而是提取结构化关键信息，同时剥离面向控制的内容
- **安全协调智能体**：部署安全协调智能体来审查、清理或标记智能体间消息，可有效缓解多智能体系统内潜在的攻击传播

##### 攻击建模和测试

- **攻击生成测试框架**：为发现未知漏洞，设计攻击生成测试框架也是一种有效方法
- **ATAG 框架**：通过扩展 MulVAL 工具，引入自定义事实和交互规则，并结合新构建的 LLM 漏洞数据库（LVD），实现针对多智能体场景的攻击建模和分析（如隐私泄露）
- **NetSafe**：将多智能体网络建模为有向图，结合三种攻击策略（错误信息注入、偏见诱导、有害信息引出），通过静态和动态指标评估拓扑安全性

##### 智能体编排

- **任务调度优化**：为避免针对智能体间通信的任务洪泛或 DoS 攻击，实现智能体编排是必要的
- **通信开销减少**：可自动优化任务调度和分配过程以减少通信开销
- **提示优化**：也可优化智能体生成的提示以节省所涉及智能体的计算资源
- **HALO**：通过三层协作架构实现动态任务分解和角色生成，使用蒙特卡洛树搜索探索最优推理轨迹，并通过自适应提示优化模块将用户查询转换为任务特定提示
- **混沌工程框架**：三阶段设计（概念框架、框架开发、经验验证），通过模拟智能体故障和通信延迟等干扰场景，结合多角度文献回顾和 GitHub 分析，旨在系统性地识别漏洞并增强智能体系统的弹性

<br>

### A-E 交互防御

#### 内存和知识相关风险防御

内存和知识相关风险可通过涵盖内容过滤、输出共识和架构隔离的综合缓解框架共同解决。

##### 嵌入空间筛选和基于聚类的异常检测

- **语义嵌入预分析**：无论内存条目是智能体内部的还是通过 RAG 外部检索的，它们的语义嵌入都可以被预防性地分析异常
- **TrustRAG**：应用聚类（如 K-means）识别偏离主导语义簇的向量，有效过滤具有低语义凝聚力的静态内存条目和检索结果
- **自适应模式**：虽然轻量级且可解释，基于聚类的过滤必须辅以自适应模式以检测上下文敏感的触发器或隐蔽的分布偏移

##### 共识过滤和基于投票的聚合

- **输出级共识机制**：为限制模型对单一被妥协检索或被污染内存的依赖，已提出输出级共识机制
- **RobustRAG**：将每个检索源独立处理，基于文档间重叠的语义内容（如共享 n-gram 或关键词）构建响应
- **投票策略**：该原理可扩展到内存快照，通过多数投票或语义投票策略，只有广泛验证的内存才能影响响应
- **集成风格过滤器**：此类集成风格过滤器通过稀释异常值或对抗源的影响来提高弹性

##### 执行监控和规划一致性检查

- **规划级内省**：内存或 RAG 输入中的对抗内容可能微妙地使智能体行为偏离用户意图，而无需显式毒性
- **ReAgent**：引入规划级内省，智能体解释用户的请求，生成预期计划，并持续将运行时操作与此痕迹对齐
- **行为异常处理**：任何不一致（由意外内存或偏离主题的检索触发）被视为行为异常，可提示停止或恢复机制
- **防护框架**：此内省框架为内存劫持和注入感知的 RAG 攻击提供强大的防护栏

#### 系统门控内存保留和输入清理

- **严格内容清理**：架构解决方案（如 DRIFT 和 AgentSafe）在生成新内容之前实施严格的内容清理
- **DRIFT**：使用注入隔离器扫描生成输出中的对抗目标转移或冒充提示
- **AgentSafe**：通过 ThreatSieve 实施信任分层存储，通过 HierarCache 进行优先级排序
- **约束未来影响**：这些机制约束未来的影响，确保 RAG 或内存中毒不会随时间无声累积

##### 统一内容来源和信任框架

- **来源元数据**：由于检索的知识和持久内存可能来自重叠的来源（如用户提示、工具调用、外部 API），维护清晰的来源元数据和信任分数至关重要
- **统一来源跟踪**：跨内存和检索管道的统一来源跟踪可实现关于保留、排名或争议内容折扣的智能决策
- **每源可靠性评分**：结合每源可靠性评分，该方法鼓励透明审计并促进下游微调或迁移机制

### 工具相关风险防御

工具相关防御策略应在四个互锁级别运行：协议基础、执行控制、编排安全和系统执行。

##### 协议级保护

- **安全验证框架**：为应对工具中毒、跨源利用和由灵活但监管不足的协议（如 MCP）启用的影子攻击等风险，研究人员引入了在注册表和中间件层运行的安全验证框架
- **MCP-Scan**：执行工具模式的静态检查（如扫描可疑标签或元数据）和 MCP 流量的实时基于代理的验证，利用 LLM 辅助启发式标记隐蔽行为
- **MCP-Shield**：通过签名匹配和对抗行为分析扩展，能够在执行前检测高风险工具和格式错误的任务
- **MCIP**：基于 MAESTRO，分析运行时痕迹，提出可解释的日志记录模式和安全感知模型，以跟踪复杂智能体-工具交互中的违规行为

##### 工具调用和执行控制

- **沙箱和权限门控**：在智能体的运行时执行点，沙箱和权限门控等经典技术仍然是基础
- **Google 深度防御模型**：倡导策略引擎监控计划的工具操作、验证参数安全性，并要求对风险敏感的操作进行人工确认
- **最小权限环境**：工具应在最小权限环境中执行（如具有受控文件系统和网络范围的隔离容器），以缓解直接滥用，包括 SSRF 和数据泄露威胁
- **模式强化**：执行框架还可以实施模式强化或细粒度输入/输出清理以拒绝异常有效载荷

##### 智能体编排监控

- **规划认知目标**：新方法针对智能体的规划认知——其工具的选择和链式调用
- **GuardAgent**：引入验证智能体，在工具调用进行之前检查主智能体的计划并生成可执行防护（如静态检查或运行时断言）
- **AgentGuard**：采用更声明性的观点：使用辅助 LLM 对多步工具工作流的 precondition、postcondition 和 transition 约束建模，有效约束规划者而不是在执行开始后做出反应
- **共识**：这些策略反映了不断增长的共识：LLMs 可能需要另一个 LLM 在不确定性下安全地监督复杂规划

##### 系统级中介和链式控制

- **复杂管道风险**：复杂管道（如 summarize(search("..."))）在工具隐式信任上游输出时可能成为攻击向量
- **DRIFT 结构化控制架构**：
  - Secure Planner：在严格参数约束下编译经过验证的工具轨迹
  - Dynamic Validator：持续监控下游工具执行的合规性
  - Injection Isolator：通过清理中间返回和最终输出，阻止对抗性传播在工具之间传播
- **缓解风险**：缓解内存中毒和延迟阶段工具利用的风险

<br>

## MCP 和 A2A 实验案例研究

### 协议选择

基于以下原则选择典型协议进行评估：

- **流行度**：协议选择需要优先考虑实用性，通过评估流行度更高的协议，可以保证揭示的漏洞具有更显著的实用价值
- **成熟度**：所选协议应具有尽可能高的成熟度，提供相对完整的开源项目和测试用例，并有基于它们的许多应用程序或开源项目

**MCP**：几乎是最受欢迎的智能体通信协议，已被各种公司和个人开发者采用。截至评估时，MCP 的 PyPI 包在过去一个月已收到超过 900 万次下载，NPM 上周下载量达到 420 万。此外，开源 MCP 服务器的数量也接近 2000。

**A2A**：于 2025 年 4 月提出，尚未达到与 MCP 相同的推广水平，但观察到对 A2A 的关注正在迅速上升，开发者的认可度也相当高。

### MCP 实验

实验使用 Claude 作为 MCP 客户端。Claude 是 Anthropic 开发的 AI 聊天机器人，以其出色的代码能力、安全设计和超长上下文处理能力，已成为世界上最受欢迎的 AI 助手之一。2025 年 1 月，Claude app 下载量估计达到 7.696 亿。

#### 恶意代码执行

- **目标**：演示攻击者如何利用 MCP 在计算机系统上执行恶意代码
- **设置**：下载 Claude Sonnet 3.7 作为 MCP 主机，使用官方 MCP 提供的 Filesystem Server 作为 MCP 服务器
- **攻击**：MCP 使恶意用户能够直接在本地端口 4444 上打开未认证的 Bash Shell 服务，将其添加到 .zshrc 将规范化风险，使其极易被入侵者利用
- **结果**：遗憾的是，此恶意操作未被阻止

#### 检索智能体欺骗

- **目标**：演示攻击者如何编写看似正常的 MCP 相关文档并发布在公共平台上，但文档中嵌入恶意命令
- **设置**：攻击者在文档中嵌入恶意命令，如搜索本地环境变量并将信息发送到 Slack
- **流程**：
  1. 用户下载这些文档并通过 Chroma MCP Server 构建本地向量数据库
  2. 这些恶意文档与其他合法文档一起被编码到向量数据库中，成为检索目标
  3. 当用户向 Claude Opus 4 发送请求查询此集合时，Claude 使用 Chroma 查询数据库并检索被污染的文档
  4. 然后调用 Terminal Controller 工具执行嵌入的恶意命令
- **结果**：使用 Claude Opus 4 成功实现此攻击

#### 工具中毒

- **目标**：演示攻击者如何在 MCP 工具描述中添加恶意指令
- **说明**：此类指令不影响工具执行，但可诱导智能体执行危险操作
- **设置**：使用 Claude Sonnet 4.0、Filesystem MCP Server 和 Gmail MCP Server 说明此漏洞

#### 命令注入

- **目标**：演示恶意工具如何直接操纵用户系统而不引起任何警报
- **方法**：向 Terminal-Controller MCP 工具注入恶意命令
- **攻击细节**：在工具描述中嵌入恶意命令，当用户调用此工具时，嵌入的命令将被执行
- **隐蔽性**：在描述中告知智能体不要列出文件夹中的其余文件以隐藏此恶意行为
- **结果**：使用 Claude Sonnet 4.0 成功执行此攻击，Claude 成功删除指定文件且未列出其余文件

### A2A 实验

#### 智能体选择操纵

- **目标**：演示攻击者如何在 Agent Card 中添加恶意描述以诱导用户选择它们
- **设置**：
  - 使用 A2A 项目提供的公共 A2A-samples 作为用户端客户端
  - 创建两个用于天气查询的智能体：一个良性，一个恶意
- **正常情况**：恶意智能体在其 Agent Card 中不添加任何引导性描述时，发送查询 20 次，发现两个智能体都可能被客户端选择
- **攻击情况**：恶意智能体修改其 Agent Card，强调可提供最丰富的功能，并添加指令要求客户端首先选择它
- **结果**：在此条件下，A2A 客户端选择的智能体始终是恶意智能体
- **结论**：结果表明攻击者可以简单地通过添加一些描述来提高其处理用户任务的优先级

<br>

## 未来方向与挑战

### 技术方面

#### 强大但轻量级的恶意输入过滤器

- **需求背景**：用户输入仍然是智能体生态系统中最大规模的攻击载体，特别是考虑到输入正变得更加开放（不再限于用户指令，还包含环境反馈）、多模态和语义复杂
- **双重需求挑战**：未来智能体生态系统将更加关注有效性，特别是考虑到 LLMs 的运行速度本质上是缓慢的，这种双重需求将对相关防御造成非常沉重的负担
- **解决方案方向**：
  - 需要成熟 AI 技术来精简防御模型（如 DeepSeek）
  - 需要与其他技术集成，如将一些基础计算卸载到可编程线速设备（如可编程交换机和 SmartNICs）上，以促进输入过滤过程

#### 去中心化通信归档

- **重要性**：记录通信过程和内容对于某些特定领域（如金融）很重要，用于审计潜在的犯罪和错误
- **需求**：考虑到安全性和可靠性，此类存储不能依赖单个存储点，必须保证完整性和效率
- **技术方向**：应采用区块链等其他技术来管理历史通信
- **CS-based 通信**：更容易实现，因为存在集中式服务器用于建立本地分布式归档机制，如企业网络中的分布式存储链
- **P2P-based 通信**：如何实现去中心化通信归档，特别是对于跨国家智能体，几乎是一个需要从头开始构建的构造

#### 实时通信监督

- **重要性**：虽然事后审计不可或缺，但实时监督可以在攻击或错误发生时最小化损害，因为它的反应时间更短
- **CS-based 通信**：在构建此类监督机制方面面临较少困难，因为集中式架构在监控整个网络方面具有天然优势
- **P2P-based 通信**：可能需要更多努力来实现集体监督
- **意义**：我们认为这是构建可靠和安全 AI 生态系统的重要功能

#### 跨协议防御架构

- **现状问题**：虽然现有协议在一定程度上解决了异构性问题，但不同协议也缺乏无缝协作
- **挑战示例**：仍然难以为智能体和工具分配通用身份（跨 A2A 和 MCP），这会降低系统性能，如果编排不正确，可能会产生不一致错误
- **未来方向**：未来 AI 生态系统应专注于更通用的架构来整合不同的协议和智能体，如 IPv4，从而实现不同智能体和环境之间的无缝发现和通信

#### 智能体的判断和责任机制

- **核心挑战**：定位和分配智能体行为的责任仍然很困难
- **问题示例**：在失败的任务执行过程中，很难识别哪些步骤导致了最终结果的偏差，无论它们是恶意的还是无意的
- **原因**：中间过程的微小偏差可能导致良性结果和危险结果之间的最终差距
- **需求**：还需要一个原则来量化每个智能体或操作的责任
- **意义**：我们认为这一方面将显著解决当前 AI 生态系统的迫切需求

#### 效率与准确性之间的权衡

从信息理论的角度分析智能体通信，存在两个方向：

**高 token 通信**：
- **优势**：更大数量的 token 允许智能体传达更丰富的上下文语义、更详细的指令和更复杂的逻辑，从而减少歧义并增强多智能体协调的准确性
- **适用场景**：在需要细粒度理解的任务中，冗长的自然语言描述有助于智能体之间的目标对齐并减少偏差
- **劣势**：过多的 token 会显著增加成本和处理时间，导致系统效率降低和延迟增加；更长的上下文暴露更大的攻击面用于提示注入和数据中毒；信息过载可能分散智能体注意力，导致从无关上下文中推断错误信息

**低 token 通信**：
- **优势**：使用简洁和结构化的消息（如 JSON 格式）大大提高通信效率，减少计算成本、提高传输速度、简化解析，从而最小化潜在错误
- **劣势**：缺乏表达复杂意图或响应不可预见场景的灵活性；如果预定义的协议或格式无法捕获完整的语义意图，可能导致重大信息丢失和协作失败

**未来研究方向**：未来智能体通信协议的设计需要在效率和准确性之间进行权衡，未来研究应探索自适应通信协议，根据任务复杂性、安全要求和智能体能力动态调整冗余程度和结构。例如，高 token 通信可用于任务的探索阶段，而低 token 通信可在执行阶段采用以确保效率和安全性。

#### 走向自组织智能体网络

- **趋势**：随着 IoA 规模的不断扩大，未来的智能体通信预计将向自组织智能体网络发展
- **特征**：智能体自主发现彼此、评估能力、协商协作、形成动态任务组，并在完成后解散
- **优势**：提供高可扩展性和鲁棒性，非常适合动态和不可预测的环境

<br>

### 法律方面

除了技术方面，我们发现与智能体相关的法律法规仍然存在严重缺陷，这些空白无法通过技术来弥补。

#### 明确责任主体

- **问题**：当售出的智能体对他人的财产或人身造成损害时，难以确定最终责任主体
- **示例**：如果智能机器人在执行任务过程中损坏财产，法律层面对开发者、用户或企业责任的量化缺乏明确定义
- **多智能体协作问题**：对于多个智能体协作工作产生的问题，如多辆自动驾驶车辆编队行驶时发生事故，缺乏关于车辆所属企业或相关主体之间责任划分的法律规定

#### 保护知识产权

- **现状**：现在已经出现了大量已开源的 LLMs，这些可以作为不同智能体的大脑
- **限制**：即使对于开源 LLMs，发布者仍限制其应用范围，例如，其他开发者也应该开源其基于这些 LLMs 构建的智能体
- **法律空白**：仍然缺乏法律来有效保护此类知识产权
- **具体问题**：
  - 确定智能体中抄袭的标准不明确
  - 即使确定抄袭，仍然缺乏定义抄袭程度的标准（例如，50% 或 90%？）
- **需求**：我们认为迫切需要相关法律法规

#### 跨境监督

- **跨国性质**：智能体通信具有跨国性质，在一个国家训练的智能体可能被其他国家的人用于非法活动
- **法律适用问题**：此时难以确定适用哪个国家的法律
- **监督空白**：缺乏统一的国际监督标准和司法合作机制
- **后果**：可能导致跨境安全的困难





<br>



