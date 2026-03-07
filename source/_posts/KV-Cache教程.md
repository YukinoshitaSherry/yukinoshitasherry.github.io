---
title: KV Cache
date: 2026-02-25
categories:
  - 学AI/DS
tags:
  - LLM
  - 推理优化
desc: KV Cache 完整教程：为什么需要、使用场景、注意力公式推导、为何只缓 K/V 不缓 Q、显存计算、PyTorch 代码实现、PagedAttention 等优化，以及面试要点。
---

**参考**：

- [大模型参数与显存](大模型参数与显存.md)（显存估算）
- vLLM 文档：<https://docs.vllm.ai/>
- Andrew Szot：<https://www.andrewszot.com/posts/kv_cache/>

<br>


## 定义

**KV Cache（Key-Value Cache）**：在 Transformer 自回归生成时，把每一层、每个 token 的 **Key** 和 **Value** 向量**缓存到显存**，后续生成步骤直接复用，避免重复计算。

### 朴素推理的问题

自回归生成时，每生成一个 token，模型都要「回顾」所有历史 token 做注意力计算。

**无 KV Cache 的朴素做法**：每生成第 $t$ 个 token，都把**整个序列** $[x_1, x_2, \ldots, x_t]$ 再过一遍 Transformer，重新计算所有 token 的 Q、K、V。

| 生成步骤 | 重新计算的 token |
| :--- | :--- |
| 预测 token 2 | 重新算 token 1 的 K、V |
| 预测 token 3 | 重新算 token 1、2 的 K、V |
| 预测 token 4 | 重新算 token 1、2、3 的 K、V |
| 预测 token $n$ | 重新算 token 1 到 $n-1$ 的 K、V |

**冗余量**：到第 $n$ 个 token 时，历史 token 的 K、V 已被重复计算 $(n-1) \times L \times H$ 次（$L$ 层、$H$ 个头）。

### 复杂度对比

| 方式 | 每步计算量 | 总复杂度 |
| :--- | :--- | :--- |
| **朴素** | 对 $t$ 个 token 做完整前向 | $O(n^2)$（$n$ 为生成长度） |
| **KV Cache** | 只算当前 token 的 Q、K、V，K/V 从缓存读 | $O(n)$ |

### 硬件瓶颈

- GPU 算力很强，但**内存带宽**有限
- 朴素方式每步都要从显存反复读入整段历史的 K、V，带宽成为瓶颈
- 注意力变成 **memory-bound**，GPU 大量时间在等数据
- KV Cache 把历史 K、V 留在显存，减少重复搬运，显著加速

<br>

## 使用场景

| 场景 | 是否用 KV Cache |
| :--- | :--- |
| **训练** | 不用。整段序列并行计算，无自回归逐步生成 |
| **推理 / 文本生成** | 用。逐 token 自回归，KV Cache 可大幅加速 |
| **预填充（Prefill）** | 第一次处理 prompt 时，可批量算完 prompt 的 K、V 并写入 cache |
| **解码（Decode）** | 每生成 1 个 token，只算新 token 的 Q、K、V，K、V 追加到 cache |

**结论**：KV Cache 是**推理优化**手段，训练不涉及。

<br>

## 原理与公式

### 自注意力回顾

设输入 $X \in \mathbb{R}^{n \times d}$，$n$ 为序列长度，$d$ 为隐藏维度。单头注意力：

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

$$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

因果（Causal）注意力下，位置 $t$ 只能看到 $t$ 及之前，即 $Q_t$ 只与 $K_{1:t}, V_{1:t}$ 做注意力。

### 自回归生成时的计算

**Step 1**：给定 prompt $[x_1, \ldots, x_n]$，算 $Q_{1:n}, K_{1:n}, V_{1:n}$，得到输出，预测 token $n+1$。

**Step 2**：生成 $x_{n+1}$ 后，要预测 token $n+2$。此时只需：

- 新 token $x_{n+1}$ 的 embedding 经过各层，得到 $Q_{n+1}, K_{n+1}, V_{n+1}$
- 注意力计算：$Q_{n+1}$ 与 $[K_{1:n}, K_{n+1}]$ 做注意力，再乘 $[V_{1:n}, V_{n+1}]$

$$\text{Attn}(X_{n+1}) = \text{Softmax}\left(\frac{Q_{n+1} [K_{1:n}; K_{n+1}]^\top}{\sqrt{d_k}}\right) [V_{1:n}; V_{n+1}]$$

**关键**：$K_{1:n}, V_{1:n}$ 在 Step 1 已经算过，且**不会再变**（权重固定、输入固定），因此可直接复用，无需重算。

### 为何可缓存 K、V

- 推理时**模型权重不变**
- 历史 token 的 embedding 不变
- 因此 $K_i = x_i W^K$、$V_i = x_i W^V$ 是**确定性的**，算一次即可
- 后续所有生成步骤都可复用 $K_{1:t}, V_{1:t}$

### 为何不缓存 Q

| 向量 | 含义 | 是否复用 |
| :--- | :--- | :--- |
| **K, V** | 历史 token 的「被查询」表示，代表过去 | 每个历史 token 只算一次，可缓存 |
| **Q** | 当前 token 的「查询」向量，代表当下 | 每步都不同，只用于当前步，缓存无意义 |

Q 只在当前时间步用于「问」历史，下一步会换成新的 Q，因此不需要、也不值得缓存。

<br>

## 显存占用

### 单层与总公式

每层有 K、V 两个张量。设：

- $B$：batch size
- $H$：注意力头数
- $d$：每头维度（$d = h/H$，$h$ 为隐藏维度）
- $S$：当前序列长度
- `bytes`：2（fp16）或 4（fp32）

**单层**：$2 \times B \times H \times S \times d$ 个元素（K 一份、V 一份）。

$$\boxed{\text{KV cache (bytes)} = 2 \times L \times B \times h \times S \times \text{bytes}}$$

其中 $L$ 为层数，$h = H \times d$ 为隐藏维度。单条序列、fp16：$\text{KV cache} = 4 \times L \times h \times S \text{ bytes}$。

> [!EXAMPLE]+ 手算例题
>
> **LLaMA 7B**：$L=32$，$h=4096$，fp16，$S=2048$：
> $$4 \times 32 \times 4096 \times 2048 = 1.07 \times 10^9 \text{ bytes} \approx 1 \text{ GB}$$
>
> **LLaMA 70B**：$L=80$，$h=8192$，$S=8192$：
> $$4 \times 80 \times 8192 \times 8192 \approx 21 \text{ GB}$$
>
> 长上下文（如 128K token）时，KV Cache 会超过模型权重本身，成为显存主因。

<br>

## 代码实现

### 朴素生成（无 Cache）

```python
def generate_naive(model, input_ids, max_new_tokens):
    """每步都把整段序列重新过一遍模型，无 KV cache"""
    cur_ids = input_ids
    for _ in range(max_new_tokens):
        logits = model(cur_ids).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_token], dim=1)
    return cur_ids
```

问题：`cur_ids` 越来越长，每步计算量线性增加，总复杂度 $O(n^2)$。

<br>

### 带 KV Cache 的生成

```python
def generate_with_kv_cache(model, input_ids, max_new_tokens):
    """使用 KV cache，每步只处理新 token"""
    batch_size = input_ids.shape[0]
    past_key_values = None  # 初始无缓存
    cur_ids = input_ids

    for _ in range(max_new_tokens):
        if past_key_values is None:
            # 第一步：处理完整 prompt
            outputs = model(cur_ids, use_cache=True)
        else:
            # 后续步：只输入最后一个 token
            outputs = model(cur_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_token], dim=1)

    return cur_ids
```

HuggingFace `transformers` 中，`use_cache=True` 时，`model()` 会返回 `past_key_values`，下一轮传入即可复用。

<br>

### 从零实现

```python
import torch
import torch.nn.functional as F

def attention_forward_with_cache(
    q, k, v,           # 当前步的 Q,K,V，shape [B, H, 1, d] 或 [B, H, seq_len, d]
    past_k, past_v,    # 缓存，shape [B, H, past_len, d]，首次为 None
    causal_mask=True
):
    """单层注意力的 KV cache 版前向"""
    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)  # 拼接历史 K
        v = torch.cat([past_v, v], dim=2)  # 拼接历史 V

    # 新缓存：整段 [1..t] 的 K, V
    new_past_k, new_past_v = k, v

    # 注意力：Q @ K^T / sqrt(d)
    scale = k.size(-1) ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale

    if causal_mask:
        # 因果 mask：当前 query 只能看 past + 自己
        seq_len = attn.size(-1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=attn.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn = F.softmax(attn, dim=-1)
    out = attn @ v

    return out, new_past_k, new_past_v
```

**要点**：每步只算当前 token 的 Q、K、V；把 K、V 拼到 `past_k`、`past_v` 后面；注意力用 `q @ k.T` 时，`k` 已包含历史。

<br>

### 预分配 Cache

```python
def create_kv_cache(batch_size, num_layers, num_heads, head_dim, max_seq_len, device, dtype):
    """预分配 KV cache 张量"""
    # shape: [num_layers, 2, batch_size, num_heads, max_seq_len, head_dim]
    # 2 表示 K 和 V
    cache = torch.zeros(
        num_layers, 2, batch_size, num_heads, max_seq_len, head_dim,
        device=device, dtype=dtype
    )
    return cache

# 每步写入新 token 的 K、V 到 cache[:, :, :, :, pos:pos+1, :]
# 读取时取 cache[:, :, :, :, :pos+1, :]
```

预分配可避免频繁 `torch.cat`，利于 `torch.compile` 等优化，实际推理框架（如 vLLM）均采用此类方式。

<br>

## 优化技术

### MQA / GQA

| 类型 | K、V 头数 | 显存 |
| :--- | :--- | :--- |
| **MHA** | 每层 $H$ 组 K、V | 标准 |
| **MQA** | 全层共享 1 组 K、V | 约 $1/H$ |
| **GQA** | $H$ 个 Q 头共享 $G$ 组 K、V（$G<H$） | 介于两者之间 |

LLaMA 2、3 等已广泛使用 GQA，在长上下文下显著降低 KV cache 占用。

### PagedAttention

**问题**：传统做法为每个请求预分配**连续**显存，易产生碎片、浪费。

**思路**：借鉴操作系统分页，把 KV cache 切成固定大小的 **Block**，按需分配非连续块。

- 每个序列维护 **Block Table**，记录「逻辑位置 → 物理 Block」
- 不同请求的 Block 可交错存放，减少碎片
- 相同 prompt 前缀可**共享**物理 Block，进一步提升复用

**效果**：vLLM 相比 HuggingFace 最高可达约 24× 吞吐提升。

### 量化

将 KV cache 存为 int8/int4，可减半或更多显存，配合反量化计算，精度损失较小。

<br>

## 常见问题

### 为什么只缓 K、V 不缓 Q？

Q 表示「当前要查什么」，每步不同；K、V 表示「历史有什么」，算一次即可，后续复用。

### 训练为什么不用 KV Cache？

训练时整段序列一次性前向，所有 token 并行算注意力，不存在「逐步生成、重复算历史」的场景。

### KV Cache 显存公式？

$\text{bytes} = 2 \times L \times B \times h \times S \times \text{bytes\_per\_elem}$，fp16 时 `bytes_per_elem=2`。

### 长上下文为何吃显存？

KV cache 与 $S$ 线性相关，128K token 时 $S$ 很大，cache 可达数十 GB，超过模型权重。

### 时空权衡

KV Cache 用**显存**换**计算与带宽**：多占一块 cache，少做重复计算、少搬数据，推理显著加速。

<br>

## 小结

| 要点 | 说明 |
| :--- | :--- |
| **动机** | 自回归生成时避免重复计算历史 K、V，降低复杂度和带宽压力 |
| **原理** | K、V 只依赖历史 token 和固定权重，算一次可复用 |
| **公式** | $\text{KV cache} = 2 L B h S \times \text{bytes}$ |
| **实现** | 每步只算新 token 的 Q、K、V，K、V 追加到 cache，注意力时拼接使用 |
| **优化** | GQA/MQA、PagedAttention、量化等 |

<br>