---
title: SFT 监督微调
date: 2026-03-05
categories:
  - 学AI/DS
tags:
  - LLM
  - SFT
desc: 从原理到上手的 SFT 完整教程：损失函数推导、只对 response 算 loss 的 mask 机制、Alpaca 数据格式、LLaMA-Factory 手把手配置与训练，以及 LoRA/QLoRA 选型与常见坑。
---

参考：
- [大模型训练三阶段总览](Pretrain-SFT-RLHF.md)（Pre-Training、SFT、RLHF 关系）
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 官方文档

<br>

## 定义

**SFT（Supervised Fine-Tuning，监督微调）**：在预训练模型基础上，用**人工标注的高质量 (prompt, response) 数据**做有监督训练，让模型从「通用」转向「特定任务可用」。

| 阶段 | 数据 | 目标 |
| :--- | :--- | :--- |
| **Pre-Training** | 海量无标签连续文本 | 学习语言规律、世界知识、生成能力 |
| **SFT** | 任务相关的 (问题, 答案) 对 | 学会按指令回答问题、格式化输出 |
| **RLHF** | 人类偏好数据 | 对齐价值观、提升体验 |

### 为何需要 SFT

- **指令遵循**：预训练模型不一定按「指令 → 回答」模式输出，SFT 教会这一模式
- **格式规范**：输出结构（如 JSON、步骤列表）需通过样例学习
- **领域适配**：垂直场景（法律、医疗、代码）需注入领域知识与表述方式
- **为 RLHF 打基础**：对齐类训练通常在 SFT 模型上进行

<br>

## 原理与公式

### 数学目标

语言模型用**自回归**生成：给定前文 $x_1,\ldots,x_{t-1}$，预测下一个 token $x_t$ 的分布。SFT 的数据形式为 $(c, r)$，其中 $c$ 为上下文（指令+输入），$r = r_1 r_2 \cdots r_m$ 为期望的回答。

**目标**：最大化模型生成 $r$ 的似然，等价于最小化**负对数似然**：

$$\mathcal{L}\_{SFT}(\theta) = -\sum_{t=1}^{|r|} \log P_\theta(r_t \mid c, r_{1:t-1})$$

其中 $r_{1:t-1}$ 表示 $r_1$ 到 $r_{t-1}$。

**与预训练的区别**：预训练在整段文本上对所有 token 算 loss；SFT 通常**只对 response 部分**算 loss，prompt 部分不参与（见下节）。

<br>

### Loss Mask：只对 response 算 loss

完整输入序列为：`[BOS] prompt [sep] response [EOS]`。若对**整个序列**算 loss：

- 模型会学习「如何生成 prompt」，但推理时 prompt 由用户提供，无需模型生成
- 容易学到「复述问题」等无效模式，浪费算力且影响质量

**正确做法**：只对 response 部分计算 loss，prompt 对应位置的 loss 置为 0（不参与梯度）。

**实现**：CrossEntropyLoss 支持 `ignore_index`。将 prompt 位置的 label 设为 `-100`（PyTorch 默认忽略），response 位置的 label 为真实 token id。

```text
input_ids:  [BOS] p1 p2 p3 ... [sep] r1 r2 r3 ... [EOS]
labels:     -100 -100 -100 ... -100   r1 r2 r3 ... r_EOS
```

只有 $r_1, r_2, \ldots$ 参与 loss 计算与反向传播。

<br>

### 从公式到实现

设完整序列为 $s = [s_1,\ldots,s_n]$，其中 $s_1,\ldots,s_k$ 为 prompt，$s_{k+1},\ldots,s_n$ 为 response。

- 前向：模型输出 logits $\hat{y}_t = f_\theta(s_{<t})$，预测分布 $P_\theta(\cdot \mid s_{<t})$
- Loss：仅对 $t \in [k+1, n]$ 计算交叉熵：

$$\mathcal{L} = -\sum_{t=k+1}^{n} \log P_\theta(s_t \mid s_{<t})$$

**实现要点**：

1. 构造 `labels`：prompt 段填 `-100`，response 段填真实 token id
2. 使用 `model(input_ids, labels=labels)` 或 `CrossEntropyLoss(ignore_index=-100)` 计算 loss

> [!NOTE]+ 常见坑
>
> 若 `labels` 与 `input_ids` 错位（例如 labels 比 input_ids 少一位），会导致预测与标签不对齐。通常做法是 `labels[i]` 对应预测 `input_ids[i+1]`，即 labels 整体左移一位，最后一个 token 不参与 loss。

<br>

## 数据准备

### Alpaca 格式

斯坦福 Alpaca 的标准字段：

| 字段 | 必填 | 说明 |
| :--- | :--- | :--- |
| **instruction** | 是 | 指令/问题 |
| **input** | 否 | 附加输入（可为空） |
| **output** | 是 | 期望回答 |

```json
{
  "instruction": "解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支，使计算机能够从数据中学习并改进，而无需显式编程..."
}
```

```json
{
  "instruction": "将以下英文翻译成中文",
  "input": "Machine learning is a method of data analysis.",
  "output": "机器学习是一种数据分析方法。"
}
```

**数据文件**：JSON Lines（`.jsonl`），每行一个 JSON 对象；或 JSON 数组 `[{...},{...}]`。

<br>

### ShareGPT 格式

多轮对话常用字段：

| 字段 | 说明 |
| :--- | :--- |
| **conversations** | 列表，每项含 `from`（human/gpt）和 `value`（内容） |
| **system** | 可选，系统提示 |

```json
{
  "conversations": [
    {"from": "human", "value": "什么是 SFT？"},
    {"from": "gpt", "value": "SFT 是监督微调，用标注数据对预训练模型进行二次训练..."},
    {"from": "human", "value": "和预训练有什么区别？"},
    {"from": "gpt", "value": "预训练用无标签文本学习通用知识；SFT 用 (问题, 答案) 学习特定任务..."}
  ]
}
```

<br>

### 格式对比

| 维度 | Alpaca | ShareGPT |
| :--- | :--- | :--- |
| **结构** | 单轮：instruction + input → output | 多轮：conversations 列表，human/gpt 交替 |
| **典型场景** | 指令跟随、单问单答、格式转换 | 多轮对话、追问、上下文依赖 |
| **优势** | 结构简单，易于构造与清洗；单轮 loss 清晰 | 保留对话流，模型学会多轮交互、指代消解 |
| **劣势** | 无多轮能力，复杂任务需拆成多单轮 | 数据来源多为爬取对话，质量参差；格式复杂 |
| **数据来源** | 人工构造或 GPT 生成（如 self-instruct） | 用户与 ChatGPT 等产品的真实对话导出 |

**为何需要两类**：

- **Alpaca**：适合「一问一答」类任务（翻译、摘要、分类、简单推理）。训练快、数据易控，是 SFT 的起点。
- **ShareGPT**：适合「对话式」产品，用户会追问、改需求、补充上下文。单轮数据无法学到「听懂上文」「承接上文回答」。

**选型建议**：指令模型底座通常先用 Alpaca 类数据打指令能力，再混入 ShareGPT 提升多轮体验；纯单轮任务用 Alpaca 即可。

<br>

### dataset_info 配置

LLaMA-Factory 通过 `data/dataset_info.json` 注册数据集。新建数据集时需：

1. 在 `data/` 下放置数据文件（如 `my_sft.jsonl`）
2. 在 `dataset_info.json` 中增加配置，指定列名映射与模板

**Alpaca 风格配置示例**：

```json
{
  "my_sft": {
    "file_name": "my_sft.jsonl",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
}
```

- `prompt`：对应 instruction
- `query`：对应 input（可为空）
- `response`：对应 output
- `system`：可选系统提示

**ShareGPT 风格**：使用 `sharegpt` 格式，`columns` 中指定 `messages` 等字段，详见官方文档。

<br>

### 数据质量

- **多样性**：覆盖任务类型、表述方式、长度
- **一致性**：指令风格、输出格式统一
- **量级**：百～万级视任务而定；过多低质数据不如少而精
- **划分**：预留 5%–10% 作验证集，用于监控过拟合

<br>

## LLaMA-Factory 实践

### 环境准备

```bash
# 1. 克隆
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 2. 创建环境
conda create -n llama_factory python=3.10 -y
conda activate llama_factory

# 3. 安装依赖（含 metrics，可选）
pip install -e ".[metrics]"
```

**硬件**：7B 模型 LoRA 微调，建议 24GB 显存（如 RTX 3090/4090）；QLoRA 可降至 12GB 左右。

<br>

### 准备数据

> [!EXAMPLE]+ 数据准备示例
>
> **Step 1**：创建 `data/my_sft.jsonl`：
>
> ```jsonl
> {"instruction": "什么是 SFT？", "input": "", "output": "SFT（监督微调）是在预训练模型基础上，用标注数据做有监督训练，使模型适配特定任务。"}
> {"instruction": "解释机器学习中的过拟合", "input": "", "output": "过拟合指模型在训练集上表现很好，但在测试集上泛化较差，通常因模型过于复杂或训练数据不足导致。"}
> ```
>
> **Step 2**：编辑 `data/dataset_info.json`，添加：
>
> ```json
> {
>   "my_sft": {
>     "file_name": "my_sft.jsonl",
>     "columns": {
>       "prompt": "instruction",
>       "query": "input",
>       "response": "output"
>     }
>   }
> }
> ```

<br>

### 训练配置

复制官方示例并修改，例如 `examples/train_lora/llama3_lora_sft.yaml`，新建 `my_sft.yaml`：

```yaml
### 模型
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct   # 或本地路径

### 训练阶段与微调方式
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### 数据集
dataset: my_sft
template: llama3
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true

### 输出
output_dir: saves/llama3-8b-lora-sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### 训练超参
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### 验证
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

**关键参数**：

| 参数 | 说明 | 典型值 |
| :--- | :--- | :--- |
| `finetuning_type` | `lora` / `qlora` / `full` | `lora` 省显存 |
| `lora_target` | LoRA 作用的模块 | `all` 或 `q_proj,v_proj` |
| `cutoff_len` | 最大序列长度 | 512–2048 |
| `learning_rate` | 学习率 | 1e-4 ~ 5e-5 |
| `num_train_epochs` | 训练轮数 | 2–5 |

<br>

### 启动训练

```bash
llamafactory-cli train my_sft.yaml
```

或使用 `examples` 下已有配置：

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

训练结束后，LoRA 权重保存在 `output_dir` 下。

<br>

### 推理验证

```bash
# Web UI 对话
llamafactory-cli webui

# 或 API 服务
llamafactory-cli api
```

在 WebUI 中选择训练好的 adapter 路径，加载后即可对话测试。

**合并 LoRA 到基座（可选）**：

```bash
llamafactory-cli export --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_name_or_path saves/llama3-8b-lora-sft \
  --template llama3 \
  --export_dir ./merged_model
```

<br>

## LoRA / QLoRA

### LoRA 原理

**LoRA（Low-Rank Adaptation）**：在原有线性层旁路增加低秩矩阵 $B \cdot A$，训练时只更新 $A, B$，原权重冻结。

$$W' = W + B \cdot A, \quad A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$$

- $r$：秩，通常 8–64
- 可训练参数约为 $2 \times r \times d$，远小于全量 $d^2$

**优点**：显存占用小、训练快、易于切换不同 adapter；**缺点**：表达能力略弱于全参微调。

<br>

### QLoRA

**QLoRA**：基座模型用 4-bit 量化加载，LoRA 用 fp16/bf16 训练。显存可再降一半左右，适合单卡 12GB 场景。

配置中设置 `finetuning_type: qlora` 即可。

<br>

### 选型

| 显存 | 模型规模 | 建议 |
| :--- | :--- | :--- |
| 12GB | 7B | QLoRA |
| 24GB | 7B–8B | LoRA 或 QLoRA |
| 40GB+ | 7B–13B | LoRA，batch 可更大 |
| 80GB+ | 13B–70B | LoRA；全参需多卡 |

<br>

## 常见问题

### Loss 为 0 或 NaN

- **label 全被 mask**：检查 prompt/response 划分，确认 response 段没有被填成 -100
- **数据为空**：确认 `dataset_info` 中列名与 JSON 字段对应正确
- **学习率过大**：尝试 5e-5 或更小

### 过拟合

- 减小 `num_train_epochs`
- 增加数据量或做数据增强
- 增大 `lora_dropout`、加 weight_decay

### 只会复述问题

- 确认 **只对 response 算 loss**，prompt 已正确 mask
- 检查模板是否把 instruction 和 output 正确分隔（如 `### 回答:\n` 后才是 response）

### 显存不足

- 减小 `per_device_train_batch_size`，增大 `gradient_accumulation_steps`
- 减小 `cutoff_len`
- 使用 `finetuning_type: qlora`
- 启用 gradient checkpointing（若框架支持）

<br>

## 小结

| 环节 | 要点 |
| :--- | :--- |
| **原理** | 最大化 response 的似然；只对 response 算 loss，prompt 用 -100 mask |
| **数据** | Alpaca（instruction/input/output）或 ShareGPT 多轮；注意列名映射 |
| **实践** | LLaMA-Factory：dataset_info → 配置 yaml → `llamafactory-cli train` |
| **选型** | 显存紧张用 QLoRA；一般场景 LoRA 即可 |

<br>

