---
title: AI Infra：从一块 GPU 到万卡集群的训练与推理系统
date: 2026-11-17
categories:
  - 学AI/DS
tags:
  - AI Infra
  - LLM
  - 分布式训练
  - 推理优化
  - GPU
desc: 从单卡的 roofline 讲到万卡训练和大规模推理：kernel 与融合、DP/ZeRO/TP/PP/SP/CP/EP、训练稳定性、确定性训练与 replay、continuous batching / PagedAttention / 量化 / 投机解码 / PD 分离、编译器、互连与调度、MoE 与超节点、RL 后训练系统、成本指标。附推导和代码。
---

参考：
- <a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook: Training LLMs on GPU Clusters</a>（Hugging Face，2025）
  - 1 至 512 块 GPU 规模下的并行策略选择，含 4000 余组实验数据
- <a href="https://jax-ml.github.io/scaling-book/">How to Scale Your Model（JAX Scaling Book）</a>
  - roofline、通信代价、分片推导；TPU 一章讲脉动阵列和 ICI
- <a href="https://github.com/stas00/ml-engineering">Machine Learning Engineering Open Book</a>（Stas Bekman）
  - 训练集群工程实践手册，故障排查部分较完整
- <a href="https://github.com/chenzomi12/AISystem">AISystem（ZOMI 酱）</a>
  - 中文体系化课程，AI 芯片和编译器两部分对应第 6、7 章
- <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/">CUDA C++ Programming Guide</a>
  - 线程层级、SM、warp、存储层级
- <a href="https://resources.nvidia.com/en-us-tensor-core">NVIDIA H100 Tensor Core GPU Architecture Whitepaper</a>
  - Hopper 的 Tensor Core、TMA、FP8
- <a href="https://arxiv.org/abs/1704.04760">In-Datacenter Performance Analysis of a Tensor Processing Unit</a>（Jouppi et al.，ISCA 2017）
  - TPU v1 的脉动阵列设计，和同期 CPU / GPU 的对比
- <a href="https://arxiv.org/abs/2304.01433">TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning</a>（ISCA 2023）
  - OCS 光交换、3D torus、SparseCore
- <a href="https://www.hiascend.com/document">昇腾 CANN / Ascend C 文档</a>
  - 达芬奇架构的 Cube / Vector 单元、Ascend C 编程范式
- <a href="https://dl.acm.org/doi/10.1145/1498765.1498785">Roofline: An Insightful Visual Performance Model</a>（Williams et al.，CACM 2009）
  - roofline 原始论文
- <a href="https://arxiv.org/abs/1710.03740">Mixed Precision Training</a>（ICLR 2018）
  - FP16 训练、master weights、loss scaling
- <a href="https://arxiv.org/abs/2209.05433">FP8 Formats for Deep Learning</a>（2022）
  - E4M3 / E5M2 的定义和训练效果
- <a href="https://arxiv.org/abs/2205.14135">FlashAttention</a>
  - IO-aware 注意力、online softmax、IO 复杂度证明
- <a href="https://arxiv.org/abs/2307.08691">FlashAttention-2</a>
  - 并行划分和 warp 分工的改进
- <a href="https://arxiv.org/abs/2407.08608">FlashAttention-3</a>
  - Hopper 上的异步和 FP8
- <a href="https://triton-lang.org/main/getting-started/tutorials/index.html">Triton Tutorials</a>
  - 从向量加法到矩阵乘，自己写 kernel
- <a href="https://arxiv.org/abs/1909.08053">Megatron-LM</a>（2019）
  - 张量并行的切法和 f / g 算子
- <a href="https://arxiv.org/abs/2104.04473">Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM</a>（SC 2021）
  - TP + PP + DP 组合、interleaved 1F1B
- <a href="https://arxiv.org/abs/2205.05198">Reducing Activation Recomputation in Large Transformer Models</a>（2022）
  - 激活显存公式、sequence parallel、选择性重算
- <a href="https://arxiv.org/abs/1910.02054">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</a>（SC 2020）
  - 三个阶段的显存和通信量
- <a href="https://arxiv.org/abs/2304.11277">PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel</a>（VLDB 2023）
  - FSDP 的实现和调参经验
- <a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers</a>（2023）
  - 上下文并行，KV 环形传递
- <a href="https://arxiv.org/abs/2309.14509">DeepSpeed-Ulysses</a>（2023）
  - 用 All-to-All 在序列和头之间转置
- <a href="https://arxiv.org/abs/2402.15627">MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs</a>（NSDI 2024）
  - 万卡训练的 MFU 优化、故障诊断
- <a href="https://arxiv.org/abs/2407.21783">The Llama 3 Herd of Models</a>
  - §3.3 的基础设施部分：故障统计、4D 并行
- <a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 Technical Report</a>
  - FP8 训练、DualPipe、EP 和推理部署
- <a href="https://pytorch.org/docs/stable/notes/randomness.html">PyTorch Reproducibility</a>
  - 官方的确定性开关清单
- <a href="https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/">Defeating Nondeterminism in LLM Inference</a>（Thinking Machines，2025）
  - 推理非确定性的真正来源、batch invariant kernel
- <a href="https://arxiv.org/abs/2511.17826">Deterministic Inference across Tensor Parallel Sizes</a>（2025）
  - 不同 TP 大小之间也逐位一致的做法
- <a href="https://arxiv.org/abs/2309.06180">Efficient Memory Management for LLM Serving with PagedAttention（vLLM）</a>（SOSP 2023）
  - KV cache 分页、写时复制
- <a href="https://www.usenix.org/conference/osdi22/presentation/yu">Orca: A Distributed Serving System for Transformer-Based Generative Models</a>（OSDI 2022）
  - iteration-level scheduling
- <a href="https://arxiv.org/abs/2403.02310">Sarathi-Serve</a>（OSDI 2024）
  - chunked prefill、stall-free batching
- <a href="https://arxiv.org/abs/2401.09670">DistServe</a>（OSDI 2024）
  - prefill / decode 分离和按 SLO 配比
- <a href="https://arxiv.org/abs/2312.07104">SGLang</a>（2023）
  - RadixAttention 前缀缓存
- <a href="https://arxiv.org/abs/2407.00079">Mooncake</a>（2024）
  - 以 KV cache 为中心的分离式架构
- <a href="https://arxiv.org/abs/2211.17192">Fast Inference from Transformers via Speculative Decoding</a>（ICML 2023）
  - 投机解码的接受规则和加速比
- <a href="https://arxiv.org/abs/2210.17323">GPTQ</a>（ICLR 2023）
  - 基于二阶信息的逐列量化
- <a href="https://arxiv.org/abs/2306.00978">AWQ</a>（MLSys 2024）
  - 按激活幅值保护重要通道
- <a href="https://arxiv.org/abs/2211.10438">SmoothQuant</a>（ICML 2023）
  - 把激活离群值挪到权重上
- <a href="https://pytorch.org/assets/pytorch2-2.pdf">PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation</a>（ASPLOS 2024）
  - Dynamo、AOTAutograd、Inductor
- <a href="https://openxla.org/xla">OpenXLA</a>
  - XLA 的 HLO 和融合
- <a href="https://tvm.apache.org/">Apache TVM</a>
  - 调度原语和自动调优
- <a href="https://github.com/NVIDIA/TensorRT-LLM">TensorRT-LLM</a>
  - NVIDIA 的推理引擎和插件

<br>

## 0. 概述：AI Infra 的范围与核心约束

### 0.1 定义与成本构成

#### 0.1.1 定义


AI Infra（AI 基础设施）指支撑大模型在给定的预算、电力与时间约束下完成训练，并以较低成本提供推理服务的软硬件系统。分为三层：
- 下层是硬件资源
  - GPU
  - HBM（High Bandwidth Memory，与 GPU 芯片封装在一起的高带宽显存）
  - NVLink（NVIDIA 的 GPU 间高速互连）
  - InfiniBand（数据中心常用的低延迟高带宽网络）
  - 存储与供电
- 上层服务于以下场景
  - 预训练
  - SFT（Supervised Fine-Tuning，监督微调）
  - RL（Reinforcement Learning，强化学习后训练）
  - 在线推理
- 中间层是软件系统
  - kernel（在 GPU 上执行的单个计算函数）
  - 通信库
  - 并行框架
  - 编译器
  - 调度器
  - serving 引擎（接收推理请求、组织批处理并返回结果的服务程序）

#### 0.1.2 成本构成

| 资源 | 构成 | Infra 的优化方向 |
|---|---|---|
| GPU 时间 | GPU 数 × 小时 × 单价 | 提高每块 GPU 的有效利用率（MFU） |
| 显存容量 | 权重、优化器状态、激活、KV cache 均需驻留显存 | 切分、重算、量化、分页 |
| 网络带宽 | 梯度同步、激活交换、专家分发 | 减少通信量、通信与计算重叠、按拓扑放置 |

GPU 时间：
- GPU 时间是训练与推理的主要成本项
  - 费用等于 GPU 数量、使用时长与单价的乘积
- 在 GPU 数量与单价固定时，降低成本的途径是让每块 GPU 在单位时间内完成更多有效计算
- 衡量这一点的指标是 MFU（Model FLOPs Utilization）
  - 即模型实际完成的有效浮点运算速率与硬件理论峰值之比
  - 大规模稠密模型训练的 MFU 通常在 35% 到 55% 之间
  - 其余时间消耗在通信等待、访存瓶颈与流水线空闲上

显存容量：
- 显存容量决定一块 GPU 能容纳多大的模型、多少并发请求
- 需要驻留显存的数据包括四类
  - 权重
  - 优化器状态：优化器为每个参数维护的辅助变量，例如 Adam 的一阶矩与二阶矩
  - 激活：前向传播中为反向计算梯度而保存的中间张量
  - KV cache：推理时缓存的已处理 token 的 Key 与 Value 张量
- 容量不足时的应对方式
  - 把数据切分到多块 GPU 上
  - 以重新计算代替存储
  - 用更低位宽的格式量化
  - 对 KV cache 按固定大小的页分配，以减少碎片

网络带宽：
- 网络带宽决定多块 GPU 协同工作时的通信开销
- 通信开销的来源
  - 数据并行训练每一步都要同步梯度
  - 张量并行与流水线并行需要在 GPU 之间交换激活
  - MoE（Mixture of Experts，混合专家）模型需要把 token 分发到位于其他 GPU 上的专家子网络
- 对应的优化方向有三类
  - 通过算法设计减少通信量
  - 让通信与计算在时间上重叠执行
  - 按网络拓扑放置计算任务，使通信量大的 GPU 位于高带宽链路的两端

#### 0.1.3 训练成本估算

训练一个参数量为 70B 的稠密模型：
- 稠密模型指每个 token 都经过全部参数的模型，与 MoE 相对
- 训练数据为 15T token
- 总计算量约为（系数 6 的来源见 0.3.1 节）：

$$6 \times 70\times10^9 \times 15\times10^{12} \approx 6.3\times10^{24}\ \text{FLOPs}$$

算力与耗时：
- H100 在 BF16（一种 16 位浮点格式，见 1.5 节）下的稠密峰值算力约为 989 TFLOPS
  - TFLOPS 表示每秒 $10^{12}$ 次浮点运算
- 按 MFU 为 40% 估算：

$$\frac{6.3\times10^{24}}{0.4\times 989\times10^{12}} \approx 1.6\times10^{10}\ \text{GPU·s} \approx 4.4\times10^{6}\ \text{GPU·h}$$

成本：
- 按每 GPU 小时 2 美元计，总成本约 900 万美元
- 若 MFU 由 40% 提升至 50%，所需 GPU 时间缩短 20%
  - 可节省约 180 万美元

### 0.2 核心术语

下表列出全文出现频率最高的 12 个术语。

| 术语 | 英文 | 含义 |
|---|---|---|
| 算力 | FLOPS | 每秒浮点运算次数；FLOPs（小写 s）指浮点运算总次数 |
| 显存 | HBM | GPU 的板载主存，容量大，但带宽与延迟均劣于片上存储 |
| 带宽 | Bandwidth | 单位时间内可传输的字节数。显存带宽（HBM 与 SM 之间，SM 为 GPU 的计算单元，见 1.1 节）与互连带宽（GPU 与 GPU 之间）是两个不同的指标 |
| Kernel | Kernel | 在 GPU 上执行的一个函数，一次矩阵乘或一次 softmax 均对应一个 kernel |
| 融合 | Fusion | 将多个 kernel 合并为一个，中间结果保留在片上存储，不写回显存 |
| 并行 | Parallelism | 按数据（DP）、矩阵（TP）、层（PP）、序列（SP/CP）、专家（EP）切分计算任务 |
| 集合通信 | Collective | 多块 GPU 协同完成的通信原语：AllReduce、AllGather、ReduceScatter、All-to-All |
| MFU | Model FLOPs Utilization | 有效算力与理论峰值之比 |
| Prefill / Decode | — | 推理的两个阶段：prefill 一次性处理整个 prompt，属于计算密集型；decode 逐 token 生成，属于带宽密集型 |
| KV cache | — | 缓存已处理 token 的 Key/Value 张量，避免每步重复计算 |
| 吞吐 / 延迟 | Throughput / Latency | 吞吐为单位时间处理的 token 数，延迟为单个请求的等待时间，二者通常相互制约 |
| SLO | Service Level Objective | 服务等级目标，例如首 token 延迟 ≤ 500 ms、每 token 间隔 ≤ 50 ms |

### 0.3 五类资源约束：算力、带宽、显存、延迟、成本

后续各章的分析均围绕这五类约束展开。

#### 0.3.1 计算量

设 $N$ 为模型参数量，每个 token 的计算量如下：

| 场景 | 每 token FLOPs |
|---|---|
| 前向 | $\approx 2N$ |
| 训练（前向 + 反向） | $\approx 6N$ |
| 训练 + 全量激活重算 | $\approx 8N$ |
| 推理 | $\approx 2N$，长上下文需另加注意力项 |

推导如下：
- $[m,k]\times[k,n]$ 的矩阵乘需要 $2mkn$ 次浮点运算
  - 每次乘加计为 2 次
- 线性层 $W\in\mathbb{R}^{k\times n}$ 对单个 token 的前向计算量为 $2kn$
  - 即参数量的 2 倍
- 反向传播需要计算 $\partial L/\partial x = \delta W^\top$ 与 $\partial L/\partial W = x^\top\delta$ 两个规模相同的矩阵乘
  - 因此反向计算量为前向的 2 倍
  - 前向与反向合计 $6N$
- 激活重算（activation recomputation）：前向时不保存激活，反向时重新执行一次前向以恢复激活
  - 额外的一次前向使总量增至 $8N$

注意力项：
- 除参数相关的矩阵乘外，注意力中的 $QK^\top$ 与 $PV$ 每层每 token 的前向计算量约为 $4sd_{\text{model}}$
  - $s$ 为序列长度
  - $d_{\text{model}}$ 为隐藏维度
- 设层数为 $L$，计入反向后，训练的每 token 计算量为

$$C_{\text{train}} \approx 6N + 12\,L\,d_{\text{model}}\,s \quad (\text{FLOPs/token})$$

#### 0.3.2 带宽

带宽约束分为两类：
- 第一类是显存带宽
  - 一个 kernel 读写的总字节数除以 HBM 带宽，得到该 kernel 的耗时下限
  - 例如在 H100（HBM 带宽 3.35 TB/s）上读写 1 GB 数据，耗时不少于约 0.3 ms，无论 kernel 的计算多么简单
- 第二类是网络带宽
  - 一次集合通信传输的字节数除以链路带宽，得到该次通信的耗时下限

一次数据传输的耗时可以近似写成

$$T \approx T_{\text{setup}} + \frac{D}{B}$$

各符号含义：
- $T_{\text{setup}}$ 是与数据量无关的固定开销
  - 包括软件调用、链路握手与首字节到达的时间，即通常所说的延迟（latency）
- $D$ 为传输字节数
- $B$ 为有效带宽

由该式得到的结论：
- 延迟与带宽是两个独立的指标
- 传输量小时，固定开销占主导，提高带宽几乎没有收益
- 传输量大时，$D/B$ 占主导，耗时近似与数据量成正比
- 以 $T_{\text{setup}} = 10\ \mu\text{s}$、$B = 50$ GB/s 的跨节点链路为例
  - 传输 1 KB 约需 $10.02\ \mu\text{s}$，几乎全部是固定开销
  - 传输 1 GB 约需 20 ms，固定开销可以忽略
- 这一规律决定了分布式系统倾向于把多次小通信合并为一次大通信
  - 例如数据并行训练中把多个梯度张量打包成一个缓冲区后一次性同步

#### 0.3.3 显存

- 训练时的显存占用包括六部分
  - 权重
  - 梯度
  - 优化器状态
  - 激活
  - 临时缓冲区（kernel 执行时申请的工作空间）
  - 显存分配器碎片（已分配块之间无法利用的空隙）
- 推理时的显存占用包括三部分
  - 权重
  - KV cache
  - 激活
  - 其中 KV cache 随并发请求数与上下文长度线性增长，在长上下文场景中往往超过权重本身
- 详细计算见 1.7 节

#### 0.3.4 延迟

$$T \approx \max(T_{\text{compute}}, T_{\text{memory}}) + T_{\text{launch}} + T_{\text{comm}} + T_{\text{queue}}$$

各项含义：
- $T_{\text{compute}}$ 与 $T_{\text{memory}}$ 分别为计算耗时与访存耗时
  - 二者可以重叠执行，因此取最大值
- $T_{\text{launch}}$ 为 kernel 启动开销
- $T_{\text{comm}}$ 为通信耗时
- $T_{\text{queue}}$ 为请求排队时间

#### 0.3.5 成本

$$\text{每百万 token 成本} = \frac{\text{单 GPU 小时价格}\times \text{GPU 数}}{\text{吞吐（token/s）}\times 3600}\times 10^6$$

- 以 8 块 H100、每 GPU 小时 2 美元、吞吐 20,000 token/s 计算：$\frac{16}{20000\times3600}\times10^6 \approx 0.22$ 美元 / 百万 token

由该式得到的结论：
- 在硬件价格给定时，推理成本与吞吐成反比
  - 吞吐提高一倍，每百万 token 的成本减半
- 推理系统的大部分优化最终都体现为在满足 SLO 的前提下提高这一吞吐
  - 包括批处理、量化、投机解码与 prefill/decode 分离（第 5 章）

#### 0.3.6 端到端加速比：Amdahl 定律

- 优化单个组件时，端到端收益受该组件在总耗时中所占比例的限制
- 设某组件原本占总耗时的比例为 $f$，优化后该组件加速 $s$ 倍，其余部分不变，则端到端加速比为

$$S = \frac{1}{(1-f) + f/s}$$

- 这一关系称为 Amdahl 定律

例子：
- 某个 kernel 被加速 2 倍，但它只占一次请求端到端耗时的 20%，则

$$S = \frac{1}{0.8 + 0.2/2} \approx 1.11$$

- 端到端只快了约 11%
- 即使把该 kernel 的耗时降为零（$s\to\infty$），加速比的上限也只有 $1/(1-f) = 1.25$

推论：
- 一次完整的推理请求要经过以下环节
  - 网络接入
  - 路由
  - 排队
  - CPU 端的分词与预处理
  - GPU 计算
  - KV cache 管理与后处理
- 因此性能优化应先通过 profile 确定关键路径上占比最大的部分，再决定优化对象

### 0.4 AI Infra 技术栈全景

下图自上而下列出 AI Infra 的主要层次及各层的代表性系统。

```text
┌──────────────────────────────────────────────────────────────────────┐
│ 应用层   Chat / Agent / RAG / 代码助手 / 多模态                         │
├──────────────────────────────────────────────────────────────────────┤
│ 推理服务  vLLM · SGLang · TensorRT-LLM · LMDeploy                       │
│          continuous batching · Paged/Radix KV · 量化 · 投机解码          │
│          PD 分离 · 路由 · autoscaling · SLO                            │
├──────────────────────────────────────────────────────────────────────┤
│ 训练框架  Megatron-LM · DeepSpeed · FSDP/TorchTitan · JAX/MaxText       │
│          DP/ZeRO · TP · PP · SP/CP · EP · 重算 · checkpoint · 容错      │
│          RL 后训练: verl · OpenRLHF · NeMo-RL                           │
├──────────────────────────────────────────────────────────────────────┤
│ 编译/图层 torch.compile(Dynamo+Inductor) · XLA · TVM · TensorRT         │
├──────────────────────────────────────────────────────────────────────┤
│ 算子/通信 cuBLAS · cuDNN · CUTLASS · Triton · FlashAttention           │
│          NCCL · NVSHMEM · DeepEP · Transformer Engine(FP8)              │
├──────────────────────────────────────────────────────────────────────┤
│ 运行时    CUDA Driver/Runtime · CUDA Graph · Stream · 显存分配器          │
├──────────────────────────────────────────────────────────────────────┤
│ 集群      K8s / Slurm · gang 调度 · 健康检查 · 分布式存储 · 监控         │
├──────────────────────────────────────────────────────────────────────┤
│ 硬件      GPU(SM/Tensor Core/HBM) · NVLink/NVSwitch · IB/RoCE · 存储/电力 │
└──────────────────────────────────────────────────────────────────────┘
```

各层的职责自下而上如下：
- 硬件层提供以下资源
  - 计算单元
  - 显存
  - GPU 间互连（NVLink/NVSwitch）
  - 跨节点网络（InfiniBand 或 RoCE，即基于以太网的 RDMA）
- 集群层负责资源分配与作业调度
  - 其中 gang 调度指一个分布式作业所需的全部 GPU 必须同时分配，否则整个作业不启动
- 运行时层管理以下事项
  - kernel 的提交顺序（CUDA stream）
  - 显存的申请与回收
  - 将一串 kernel 录制后整体重放以降低启动开销的 CUDA Graph
- 算子与通信层提供以下能力
  - 高性能的矩阵乘、注意力等计算 kernel
  - NCCL 等集合通信库
- 编译层把模型的计算图转换为优化后的 kernel 序列，完成融合与调度
- 训练框架与推理服务分别把上述能力组织成完整的训练流程与在线服务
- 第 2 章至第 8 章大致按这一层次依次展开

### 0.5 四类系统瓶颈

| 瓶颈 | 表现 | 量化数据 | 应对方法 |
|---|---|---|---|
| 内存墙 Memory Wall | 算力增速高于显存带宽增速 | 2012→2024 峰值算力增长约 $10^3$ 倍，HBM 带宽约 $10^1$–$10^2$ 倍 | 融合、FlashAttention、量化、提高算术强度 |
| 容量墙 Capacity Wall | 模型规模超过单块 GPU 显存 | 70B 模型 BF16 权重 140 GB，单块 GPU 80 GB | 并行切分、ZeRO、offload、量化 |
| 通信墙 Communication Wall | GPU 间带宽比 GPU 内低一个数量级以上 | HBM 3.35 TB/s，NVLink 450 GB/s/方向，IB 50 GB/s/网卡 | 拓扑感知并行、计算通信重叠 |
| 功耗墙 Power Wall | 供电与散热成为上限 | 单块 GPU 700 W 至 1 kW 以上，万卡集群达数十 MW | 低精度、液冷、能效调度 |

内存墙：
- 内存墙指计算单元的处理速度增长远快于数据供给速度，导致大量计算单元处于等待数据的状态
- 成因
  - 计算单元的数量可以随晶体管密度提升而成倍增加
  - 显存带宽受限于芯片边缘可布置的 I/O 引脚数与信号速率，增长缓慢
- 衡量一个计算任务受内存墙影响程度的指标是算术强度
  - 即其浮点运算次数与访存字节数之比（1.3 节）
- 应对方法的共同思路是减少每次计算所需搬运的字节数
  - 融合避免中间结果写回显存
  - FlashAttention 通过分块计算避免在显存中保存完整的注意力矩阵
  - 量化用更少的位表示每个数值

容量墙：
- 容量墙指模型与训练状态的总大小超过单块 GPU 的显存
- 70B 模型仅 BF16 权重就需要 140 GB
  - 训练时的模型状态约为权重的 8 倍（1.7.1 节），远超单块 80 GB GPU 的容量
- 应对方法包括
  - 用并行策略把模型切分到多块 GPU
  - ZeRO（Zero Redundancy Optimizer）把优化器状态、梯度与权重分片存储，消除数据并行各副本之间的冗余（第 3 章）
  - offload 把暂时不用的数据卸载到 CPU 内存或 SSD，需要时再取回
  - 量化降低每个参数占用的字节数

通信墙：
- 通信墙指 GPU 之间的数据交换速度远低于 GPU 内部的数据访问速度
- 以 H100 为例
  - 访问本地 HBM 的带宽为 3.35 TB/s
  - 经 NVLink 访问同节点其他 GPU 为 450 GB/s/方向
  - 经 InfiniBand 访问其他节点仅为每网卡 50 GB/s
  - 逐级相差约一个数量级
- 并行策略因此需要按拓扑放置
  - 通信最频繁的张量并行放在节点内的 NVLink 域中
  - 通信较少的数据并行跨节点展开
- 同时让通信与计算重叠，使通信时间被计算时间掩盖

功耗墙：
- 功耗墙指供电与散热能力成为部署规模的上限
- 单块 H100 的功耗约 700 W，新一代 GPU 超过 1 kW
  - 万卡集群的总功率达数十 MW
  - 数据中心的供电容量因此成为扩容的前置条件
- 功耗主要来自数据在导线上的移动与晶体管的开关
  - 因此降低数值位宽在减少带宽需求的同时也直接降低能耗
- 高功率密度的机柜则需要液冷才能将热量及时导出

四类瓶颈之间的关系：
- 本文后续内容可以看作在这四类瓶颈之间寻求平衡
- 缓解一类瓶颈的手段，往往会加重另一类
  - 用重算缓解容量墙会增加计算量
  - 用张量并行缓解容量墙会增加通信量

### 0.6 发展历程：主要矛盾的三次转移

| 阶段 | 时间 | 主要矛盾 | 代表事件 |
|---|---|---|---|
| 单卡计算加速 | 2006–2016 | 算力不足，GPU 开始用于通用计算 | CUDA (2007)、AlexNet 双 GPU 训练 (2012)、cuDNN (2014) |
| 分布式扩展 | 2017–2022 | 显存与通信成为瓶颈，训练走向分布式 | Transformer (2017)、混合精度 (2018)、Megatron (2019)、ZeRO (2020)、GPT-3 (2020) |
| 成本与可靠性 | 2022– | 推理成本、万卡集群稳定性与能耗 | FlashAttention (2022)、vLLM (2023)、Llama 3 16K GPU 训练 (2024)、DeepSeek-V3 FP8+MoE (2024)、NVL72 超节点 (2025)、推理时计算 (2025–) |

- 第一阶段的核心问题是计算速度
  - CUDA 使 GPU 可以用类 C 语言编写通用程序
  - AlexNet 用两块 GPU 在数天内完成了 CPU 需要数周的训练
  - cuDNN 把卷积等常用算子封装为高度优化的库
- 第二阶段的核心问题是规模
  - Transformer 架构使模型参数量迅速增长到单块 GPU 无法容纳的程度
  - 混合精度把每个参数的存储与计算位宽减半
  - Megatron 的张量并行与 ZeRO 的状态分片让模型可以跨数百块 GPU 训练
- 第三阶段的核心问题是经济性与稳定性
  - 模型进入大规模在线服务后，推理成本超过训练成本成为主要开支
  - vLLM 等系统通过显存管理与调度大幅提高了推理吞吐
  - 训练集群扩展到上万块 GPU，硬件故障从偶发事件变为每天都会发生的常态
  - 容错与快速恢复成为必要能力

三次转移的关系：
- 主要矛盾依次由算力转向显存与通信，再转向成本与可靠性
- 前一阶段的问题并没有消失，而是被解决到足以让下一类约束成为主导

### 0.7 学习路线

```text
                       ┌── A 训练系统：ch3 并行 → ch4 训练 → ch8 MoE / RL
 ch0 → ch1 单卡 → ch2 ─┼── B 推理系统：ch5 推理 → ch6 编译 → ch8 推理时计算
       （必修）   单机  └── C 集群平台：ch7 集群 → ch4.2 容错 → ch9 指标成本
```

- 第 0 章至第 2 章是三条路线的共同基础
  - 第 1 章建立单块 GPU 的性能模型
  - 第 2 章在此基础上讨论单机范围内的优化
- 此后可以按方向选择路线
  - 路线 A 面向训练系统，依次学习并行策略、训练稳定性与容错、MoE 与 RL 后训练
  - 路线 B 面向推理系统，依次学习推理优化、编译器与推理时计算
  - 路线 C 面向集群平台，依次学习互连与调度、训练容错与成本指标

| 背景 | 建议路线 |
|---|---|
| 算法方向转系统方向 | 先完成 A 或 B 其中一条，再补充 C |
| 后端 / 平台工程师 | C → B |
| kernel 开发 | ch1 → ch2 → ch6，随后阅读 FlashAttention 源码 |

### 0.8 前置知识与工作特点

#### 0.8.1 前置知识

| 知识领域 | 要求程度 |
|---|---|
| Python / PyTorch | 能独立编写训练循环 |
| 线性代数 | 能推算矩阵乘的形状与 FLOPs |
| Transformer | 理解 attention、MLP、LayerNorm 的计算过程 |
| 计算机组成 | 了解缓存、带宽、延迟的概念（1.2 节补充） |
| C++ / CUDA | 仅编写 kernel 时需要；入门可先学习 Triton（基于 Python 的 GPU kernel 编程语言） |

- 线性代数与 Transformer 是理解全文的必要基础
  - 后续的计算量、显存与通信量估算都建立在矩阵形状之上
- CUDA 编程只在第 2 章与第 6 章的 kernel 实现部分需要
  - Triton 以 block 为单位编写程序，由编译器处理线程级细节，入门门槛低于 CUDA

#### 0.8.2 工作特点

- Infra 工作的主要内容不是设计新算法，而是反复执行一个迭代过程
  - 用 profile 工具记录各 kernel 与通信操作的时间线（timeline），定位耗时最长的瓶颈
  - 用本章的模型估算该瓶颈的理论下限
  - 做局部修改
  - 再次 profile 验证收益
- 一次有效的优化通常需要多轮这样的迭代
  - 单轮的收益往往只有几个百分点
  - 但在万卡规模上，几个百分点的 MFU 对应数十万美元的成本（0.1.3 节）

### 0.9 阅读建议

1. 每章先阅读开头与末尾小结，建立整体框架，再进入细节。
2. 文中算例建议亲自推算一遍。
   - Infra 的工程判断依赖对数量级的熟悉
   - 例如能够迅速判断一个操作受算力还是带宽限制、一份模型状态需要多少显存
3. 代码建议实际运行。
   - 一块 NVIDIA GPU 即可完成大部分实验，Colab 提供的 T4 也能运行大部分示例
   - 涉及 FP8 或 Hopper 特性的代码需要 H100 级别的硬件

<br>

## 1. 单卡性能模型：算力、带宽与数值格式

讨论多 GPU 系统之前，需要先明确单块 GPU 的两个性能上限：
- 每秒可执行的浮点运算次数（FLOPS）
- 每秒可在显存与计算单元之间传输的字节数（Bytes/s）

本章还包含两项内容：
- 给出判断一个程序受哪一个上限约束的方法
- 介绍决定每个数值占用多少字节的数值格式

### 1.1 GPU 执行模型：SIMT 与线程层级

#### 1.1.1 CPU 与 GPU 的设计取向

- CPU 由少量复杂核心构成
  - 每个核心配有大容量缓存、分支预测器与乱序执行逻辑
  - 目标是让单个线程尽快完成，即降低单线程延迟
- GPU 由大量简单核心构成
  - 单个线程的执行速度不如 CPU，但可以同时运行数万个线程
  - 目标是提高单位时间内完成的总工作量，即吞吐
- 神经网络的主要计算是矩阵乘与逐元素运算
  - 大量数据执行相同的操作，彼此之间没有依赖
  - 因此适合 GPU 的设计取向

#### 1.1.2 软件视角的线程层级

CUDA 程序把线程组织为四级层次：
- 层次中的 SM（Streaming Multiprocessor）是 GPU 上独立调度并执行线程块的硬件计算单元
  - 一块 GPU 包含上百个 SM

```text
Grid（一次 kernel 启动）
 └── Block / CTA（线程块，最多 1024 线程，整体调度到一个 SM 上）
      └── Warp（32 个线程，同时执行同一条指令，即 SIMT）
           └── Thread（各自有寄存器）
```

各级含义如下：
- Grid 对应一次 kernel 启动所创建的全部线程
- Block（又称 CTA，Cooperative Thread Array）是被整体分配到某一个 SM 上执行的一组线程
  - 同一 block 内的线程可以通过 shared memory 交换数据并相互同步
- warp 是 GPU 调度与执行的最小单位，由 32 个线程组成
- SIMT（Single Instruction, Multiple Threads）指同一条指令由一个 warp 内的多个线程在各自的数据上同时执行
- 每个线程拥有私有的寄存器

#### 1.1.3 硬件视角（H100 SXM）

| 部件 | 规格 | 作用 |
|---|---|---|
| SM | 132 个 | GPU 的基本计算单元，可同时驻留多个 block |
| Warp 调度器 | 4 个 / SM | 每个时钟周期各发射一条 warp 指令 |
| Tensor Core | 4 个 / SM（第四代） | 专用于小矩阵块乘加的硬件单元，提供绝大部分算力 |
| 寄存器文件 | 256 KB / SM | 访问最快的存储，线程私有 |
| Shared Memory（SMEM）/ L1 | 最多 228 KB / SM | 片上 SRAM，同一 block 内的线程共享 |
| L2 Cache | 50 MB | 全卡共享的片上缓存 |
| HBM3 | 80 GB，3.35 TB/s | 显存 |

表中涉及两类存储技术：
- SRAM（Static RAM，静态随机存储器）是集成在 GPU 芯片上的高速存储
  - 寄存器、shared memory 与 L2 cache 均由其实现
  - 它的访问延迟低，但每比特占用的芯片面积大，因此容量有限
- HBM 由多层 DRAM（Dynamic RAM，动态随机存储器）芯片垂直堆叠而成
  - 与 GPU 芯片并排封装，通过大量并行的数据通道获得高带宽
  - 它的容量远大于 SRAM，但访问延迟高出一个数量级以上

#### 1.1.4 SIMT 与分支发散

- 同一 warp 的 32 个线程在同一时刻执行同一条指令
- 若程序中的 `if` 分支使一半线程进入 A 路径、另一半进入 B 路径
  - 硬件无法同时执行两条路径
  - 只能先执行 A 路径（此时 B 路径的线程被屏蔽、处于闲置状态），再执行 B 路径
- 这一现象称为分支发散（warp divergence）
  - 极端情况下，若 32 个线程各走一条不同的路径，该 warp 的执行效率降为 1/32
- 编写 kernel 时应尽量让同一 warp 内的线程走相同的分支

#### 1.1.5 延迟隐藏与 occupancy

- 一次 HBM 访问需要数百个时钟周期才能返回数据
- GPU 通过延迟隐藏（latency hiding）应对这一延迟
  - SM 上同时驻留大量 warp
  - 当某个 warp 因等待数据而停顿时，调度器在下一个周期即切换到另一个就绪的 warp 执行
  - 每个 warp 的寄存器状态始终保存在寄存器文件中，因此这种切换没有额外开销
  - 只要驻留的 warp 足够多，计算单元就始终有工作可做，访存延迟被其他 warp 的计算掩盖

衡量驻留 warp 数量的指标是 occupancy：
- occupancy 是 SM 上活跃 warp 数与其最大可驻留 warp 数之比
  - H100 每个 SM 最多驻留 64 个 warp
- 寄存器与 shared memory 是 SM 上所有驻留线程共享的有限资源
  - 每个线程使用的寄存器越多、每个 block 使用的 shared memory 越多，SM 上能同时驻留的 block 就越少
  - occupancy 随之降低
  - 例如每个 SM 有 64K 个 32 位寄存器，若每个线程使用 255 个寄存器，则最多只能驻留约 256 个线程，即 8 个 warp，occupancy 仅为 12.5%
- occupancy 并非越高越好
  - 许多高性能矩阵乘 kernel 有意用较低的 occupancy 换取每个线程更多的寄存器，以提高数据复用

#### 1.1.6 kernel 示例：向量加法

```cpp
// 向量加法：每个线程算一个元素
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程编号
    if (i < n) c[i] = a[i] + b[i];
}
// 启动：每 block 256 线程
// vec_add<<<(n + 255) / 256, 256>>>(a, b, c, n);
```

- `__global__` 声明该函数是从 CPU 端启动、在 GPU 上执行的 kernel
- 每个线程计算出全局编号 `i`，并负责第 `i` 个元素
  - 依据是自身所在 block 的编号 `blockIdx.x`、block 大小 `blockDim.x` 与 block 内编号 `threadIdx.x`
- 启动参数 `<<<(n + 255) / 256, 256>>>` 表示每个 block 含 256 个线程
  - block 数向上取整以覆盖全部 $n$ 个元素
  - `if (i < n)` 用于屏蔽最后一个 block 中超出范围的线程

访存与计算量的对比：
- 每个元素需要读取 8 字节（两个 FP32 输入）、写入 4 字节，仅执行 1 次加法
- 对 $n=10^8$ 的向量，总访存量为 1.2 GB
  - 在 3.35 TB/s 的 HBM 上至少需要约 0.36 ms
- $10^8$ 次加法在 67 TFLOPS 的 FP32 算力下只需约 1.5 µs
- 该 kernel 的速度完全由显存带宽决定

#### 1.1.7 合并访存

- 当一个 warp 的 32 个线程访问一段连续且对齐的 128 字节地址时，硬件将这些访问合并为一次内存事务
  - 这称为合并访存（coalesced access）
- 若各线程访问的地址不连续，例如按步长跨越访问
  - 硬件需要拆分为多次事务
  - 每次事务取回的数据中只有一小部分被使用，有效带宽成倍下降
- 在上面的向量加法中，相邻线程的 `i` 相邻，访问的地址也相邻，因此满足合并访存
- 编写 kernel 的首要原则是让相邻线程访问相邻地址
  - 对矩阵而言，这通常意味着让 warp 内的线程沿内存中连续存储的维度展开

### 1.2 存储层级与内存墙

#### 1.2.1 存储层级

| 层级 | 容量（H100） | 带宽 | 延迟 | 可见范围 |
|---|---|---|---|---|
| 寄存器 | 256 KB/SM，全卡约 33 MB | 最高 | ~1 cycle | 单线程 |
| SMEM / L1 | 228 KB/SM | 全卡合计几十 TB/s | ~20–30 cycles | 同一 block |
| L2 | 50 MB | ~10 TB/s 级 | ~200 cycles | 全卡 |
| HBM | 80 GB | 3.35 TB/s | ~400–600 cycles | 全卡 |
| 主机内存（PCIe Gen5 x16） | TB 级 | ~64 GB/s/方向 | µs 级 | CPU/GPU |
| 同节点其他 GPU（NVLink 4） | — | 450 GB/s/方向 | µs 级 | 节点内 |
| 其他节点（IB NDR 400G） | — | 50 GB/s/网卡 | µs 级以上 | 跨节点 |

- 层级越靠下，容量越大、速度越慢，带宽大约每下降一级减少一个数量级
- 各级的物理位置：
  - 上三级（寄存器、SMEM/L1、L2）位于 GPU 芯片内部，由 SRAM 实现
  - HBM 位于芯片封装内
  - 主机内存经 PCIe 总线连接
  - 其他 GPU 与其他节点则需经过 NVLink 或网络
- 这种分层是成本与速度的折中
  - 速度快的存储每比特面积与成本高，只能做小
  - 容量大的存储必须远离计算单元，访问代价随之上升

多数性能优化的目标是让数据尽可能停留在较高层级，并在被逐出之前尽可能多次复用：
- 矩阵乘 kernel 把输入矩阵的一个子块从 HBM 读入 SMEM 后，让该子块参与多次乘加运算再丢弃（1.4.2 节）
- 算子融合让中间结果停留在寄存器中，而不写回 HBM

#### 1.2.2 内存墙的量化

- H100 每秒可执行 989 万亿次 BF16 运算，但每秒只能从 HBM 读取 3.35 万亿字节
- 若要使计算单元不因等待数据而空闲，每读取 1 字节需要执行约 $989/3.35\approx295$ 次运算
- 加法、激活函数、LayerNorm 等逐元素操作对每个读入的数值只做几次运算
  - 折合每字节不足 1 次至数次
  - 其性能必然受带宽限制，计算单元的利用率不到 1%
- 只有矩阵乘这类每个数据被多次复用的操作，才能达到每字节数百次运算的水平

### 1.3 Roofline 模型

Roofline 模型用两条上限，即算力峰值与显存带宽，界定一个程序在给定硬件上可达到的最高性能，并据此判断该程序的瓶颈类型。

#### 1.3.1 算术强度

算术强度（arithmetic intensity）定义为一个 kernel 执行的浮点运算次数与其在显存上读写的字节数之比，单位为 FLOP/Byte：

$$I = \frac{\text{FLOPs}}{\text{Bytes}}$$

- 算术强度描述的是程序本身的性质，与硬件无关
- 它表示每从显存取得 1 字节数据，程序在这份数据上做了多少次运算
- 同一操作的算术强度可以因实现方式而改变
  - 例如融合两个逐元素操作后，读写字节数减半而运算次数不变，算术强度翻倍

#### 1.3.2 Roofline 公式

$$P = \min\left(P_{\text{peak}},\ I\times B_{\text{mem}}\right),\qquad I^* = \frac{P_{\text{peak}}}{B_{\text{mem}}}$$

- $P$ 为可达性能的上限，$P_{\text{peak}}$ 为硬件的峰值算力，$B_{\text{mem}}$ 为显存带宽
- $I\times B_{\text{mem}}$ 表示带宽能够支撑的运算速率
  - 每秒供给 $B_{\text{mem}}$ 字节，每字节对应 $I$ 次运算
- 可达性能取两者中较小的一个

$I^*$ 称为脊点（ridge point）：
- 即 roofline 图中带宽斜线与算力水平线交点所对应的算术强度
- 数值上等于峰值算力除以显存带宽
- $I<I^*$ 时，带宽能支撑的运算速率低于峰值算力，程序受带宽限制（memory-bound）
- $I>I^*$ 时，程序受算力限制（compute-bound）

```text
 性能 (FLOP/s)
   │              ______________________  算力屋顶 P_peak (989 TFLOPS)
   │             /
   │            /  斜率 = 带宽 B (3.35 TB/s)
   │           /
   │          /
   │  memory-bound | compute-bound
   └──────────┴──────────────────────── 算术强度 I (FLOP/Byte)
            I* ≈ 295
```

举例说明：
- 设某加速器的峰值算力为 100 TOPS（TOPS 为每秒 $10^{12}$ 次运算，常用于整数运算），显存带宽为 2 TB/s
  - 则其脊点为 50 ops/Byte
- 某算子的算术强度为 10 ops/Byte
  - 带宽能支撑的运算速率为 $2\times10^{12}\times10 = 2\times10^{13}$ ops/s，即 20 TOPS
- 该算子在这台硬件上最多只能达到 20 TOPS，峰值算力的 80% 无法利用
- 此时提高峰值算力没有任何收益
  - 只有提高带宽或提高算子的算术强度才能提升性能

#### 1.3.3 H100 各精度的脊点

| 精度 | 稠密峰值 | $I^*$ |
|---|---|---|
| FP32（CUDA core） | 67 TFLOPS | ≈ 20 |
| TF32 Tensor Core | 495 TFLOPS | ≈ 148 |
| BF16/FP16 Tensor Core | 989 TFLOPS | ≈ 295 |
| FP8 Tensor Core | 1979 TFLOPS | ≈ 590 |

- CUDA core 是 SM 中执行标量浮点与整数运算的通用计算单元，各数值格式的定义见 1.5 节
- 表中显存带宽不变，而峰值算力随精度降低成倍提高，因此脊点随之右移
- 这意味着改用更低精度的计算后，一个程序需要更高的算术强度才能充分利用算力
  - 原本在 BF16 下受算力限制的操作，改用 FP8 后可能变为受带宽限制

#### 1.3.4 常见操作的算术强度（BF16）

| 操作 | FLOPs | Bytes | $I$ |
|---|---|---|---|
| 向量加 $c=a+b$ | $n$ | $6n$ | 0.17 |
| GELU / SiLU | ~$10n$ | $4n$ | ~2.5 |
| LayerNorm / RMSNorm | ~$5n$ | $4n$ | ~1 |
| GEMM $[M,K]\times[K,N]$ | $2MKN$ | $2(MK+KN+MN)$ | 看形状 |

- BF16 每个元素占 2 字节
  - 向量加读两个输入、写一个输出，共 $6n$ 字节
  - 激活函数与归一化读一个输入、写一个输出，共 $4n$ 字节
- 这些操作的算术强度都在 1 到数个 FLOP/Byte 之间，比 H100 的脊点低两个数量级
- GEMM（General Matrix Multiply，通用矩阵乘）读入两个输入矩阵、写出一个结果矩阵
  - 其算术强度取决于三个维度的大小

对于方阵 $M=N=K$：

$$I = \frac{2M^3}{2\cdot 3M^2} = \frac{M}{3}$$

- 算术强度随矩阵边长线性增长
- 令 $M/3 > 295$，可知边长超过约 885 时方阵乘才进入算力受限区

#### 1.3.5 decode 阶段的算术强度

- decode 阶段每一步只为每个请求生成 1 个新 token
- 设一个批次内共有 $b$ 个 token 与形状为 $[K,N]$ 的权重相乘（$b\ll K,N$）：

$$I = \frac{2bKN}{2(bK + KN + bN)} \approx b$$

- $b$ 远小于 $K$ 与 $N$，分母中权重的读取量 $KN$ 占主导
  - 算术强度近似等于 batch size
- $b=1$ 时 $I\approx1$，与 295 的脊点相比，GPU 的算力利用率不足 1%
  - 每一步都要把全部权重从 HBM 读一遍，却只对其做一次乘加
- 因此推理系统需要把多个请求合并为一个批次执行
  - 使同一份权重被 $b$ 个 token 复用（第 5 章）

#### 1.3.6 基于 roofline 的性能诊断

1. 统计 kernel 的 FLOPs 与访存字节数，计算其算术强度 $I$。
2. 比较 $I$ 与硬件的 $I^*$，确定该 kernel 在理论上受哪一个上限约束。
3. 测量实际耗时，换算为实际达到的 FLOP/s 或 Bytes/s，评估与对应上限的差距。差距大说明实现本身存在问题，例如访存未合并或 occupancy 过低。
4. 若受带宽限制，则减少访存字节数，方法包括融合、量化与数据复用；若受算力限制，则使用 Tensor Core、降低计算精度或消除冗余计算。

### 1.4 矩阵乘与 Tensor Core

#### 1.4.1 Tensor Core

- CUDA core 每条指令执行一次标量 FMA（Fused Multiply-Add，乘加融合运算，即 $a\leftarrow a + b\times c$ 一步完成）
- Tensor Core 每条指令完成一个小矩阵块的乘加
  - Hopper 的 `wgmma` 指令一次处理 $64\times N\times16$ 规模的块
- 一条指令完成的运算量大得多，因此 Tensor Core 的吞吐远高于 CUDA core
  - H100 的 BF16 算力中，Tensor Core 提供 989 TFLOPS，而 CUDA core 的 FP16 算力仅为 134 TFLOPS
- 大模型的计算量绝大部分来自矩阵乘
  - 因此其性能几乎完全取决于 Tensor Core 的利用率
  - 不经过 Tensor Core 的计算在总计算时间中的占比可以忽略

#### 1.4.2 分块矩阵乘

GPU 上的矩阵乘按 tile（子块）划分。每个 block 负责输出矩阵 $C$ 的一个 $B_M\times B_N$ 子块，沿 $K$ 维循环：

```text
for k_tile in range(0, K, B_K):
    从 HBM 加载 A[m_tile, k_tile]、B[k_tile, n_tile] 到 SMEM
    用 Tensor Core 累加 C_tile += A_tile @ B_tile（寄存器中，FP32）
写回 C_tile
```

- 每处理一个 $K$ 方向的 tile
  - 需要从 HBM 读取 $(B_M + B_N)B_K$ 个元素
  - 执行 $2B_MB_NB_K$ 次运算
- 以元素为单位的算术强度为

$$I_{\text{tile}} = \frac{B_MB_N}{B_M+B_N}$$

- $B_M=B_N=128$ 时为 64 FLOP/元素，即 BF16 下 32 FLOP/Byte
- 不同 block 读取的输入子块有重叠
  - 这些重叠部分可在 L2 cache 中命中而不必再访问 HBM
  - 使实际的有效算术强度进一步提高

tile 大小的权衡：
- tile 越大，每个元素被复用的次数越多，算术强度越高
- 但每个 block 占用的 SMEM 与寄存器也越多
  - SM 上能同时驻留的 block 越少
  - occupancy 随之降低，延迟隐藏能力减弱
- tile 过大还会使 tile 总数过少，无法填满全部 SM
- autotuning 指对 tile 尺寸等参数的多组候选配置逐一实测并选取最快者
  - 它搜索的正是这一权衡点（6.4 节）

#### 1.4.3 Hopper 架构的新特性

| 特性 | 作用 |
|---|---|
| TMA（Tensor Memory Accelerator） | 由专用硬件完成 HBM 与 SMEM 之间的块拷贝，不占用线程 |
| WGMMA | 4 个 warp 组成 warpgroup，异步发射大规模矩阵乘 |
| Warp specialization | producer warp 仅负责数据传输，consumer warp 仅负责计算，二者构成流水线 |
| Thread Block Cluster / DSMEM | 相邻 SM 之间可直接访问彼此的 shared memory |

- 这些特性的共同目标是让数据传输与计算并行进行
- 在此前的架构上，线程需要亲自执行加载指令把数据从 HBM 搬到 SMEM
  - 加载期间这些线程无法参与计算
- TMA 把块拷贝交给专用硬件，线程只需发出一条指令
- WGMMA 以异步方式发射矩阵乘，发射后线程可以继续执行其他指令
- warp specialization 在此基础上把 warp 分为两组
  - producer 组持续预取下一批数据
  - consumer 组持续计算当前数据
  - 形成生产者与消费者流水线
- Thread Block Cluster 允许若干相邻 SM 上的 block 组成一个簇
  - 彼此直接读写对方的 shared memory（DSMEM，Distributed Shared Memory）
  - 从而在 SM 之间共享已加载的数据
- FlashAttention-3、DeepGEMM 均基于上述特性实现

#### 1.4.4 张量形状的选择

- 矩阵维度应取 64 或 128 的倍数，以对齐 Tensor Core 与 tile 的尺寸
  - 维度不对齐时，边缘的 tile 只有部分有效数据，但仍需完整执行一次 Tensor Core 指令，部分算力被浪费
  - 某些库在不对齐时还会退回到较慢的实现

词表大小应填充（pad）至 64 或 128 的倍数：
- 输出层的矩阵乘维度等于词表大小
- GPT-2 的词表大小为 50257，不是 64 的倍数
- 将其填充为 50304 后，输出层矩阵乘可以使用对齐的 tile，可测得明显的加速
  - 多出的 47 个词表位置在训练中不会被使用

Wave quantization 是 tile 数量与 SM 数量不匹配造成的效率损失：
- 一个 kernel 的全部 tile 按「波」（wave）调度，每一波最多同时占用全部 132 个 SM
- 若 tile 数不是 SM 数的整数倍，最后一波只使用部分 SM
- 例如 133 个 tile 需要执行 2 波
  - 第二波只有 1 个 SM 在工作，131 个 SM 处于闲置状态
  - kernel 耗时接近 132 个 tile 时的两倍

### 1.5 数值格式

#### 1.5.1 二进制编码与浮点数的位结构

位与编码数：
- 数字电路中的每一位（bit）只有 0 与 1 两种状态，$b$ 位最多提供 $2^b$ 种不同的编码
  - 例如 8 位可以表示 256 种编码，4 位只能表示 16 种
- 实数是连续且无限的
  - 有限位数只能从中选出有限个代表值，其余实数都必须近似为某个代表值
- 数值格式的设计，就是决定这有限个编码对应哪些实数
- 把连续数值映射为有限个离散值的过程称为量化（quantization）

两类格式：
- 整数格式（如 INT8）的代表值等间隔分布
  - INT8 通常表示 $-128$ 到 $127$ 之间的整数
  - 需要额外的缩放因子把整数映射回真实数值
- 浮点格式把位分为三个字段：符号位 S、指数位 E 与尾数位 M
  - 一个正规浮点数的值为

$$x = (-1)^S \times 2^{E-\text{bias}} \times (1.M)$$

- 指数（exponent）决定数值的数量级，即可表示的范围
- 尾数（mantissa）是二进制小数点之后的部分，决定在给定数量级内能区分多细，即精度
- bias 是一个固定偏移量，使指数字段只需存储非负整数就能表示负指数
- 正规数的尾数前隐含一个 1，不占存储位

举例说明：
- 尾数字段为 $01$ 时，$(1.01)_2 = 1 + 0\times2^{-1} + 1\times2^{-2} = 1.25$
  - 若 $E-\text{bias}=2$，符号位为 0，则该数为 $1.25\times2^2=5$
  - 若指数改为 $E-\text{bias}=-3$，同一尾数表示 $1.25\times2^{-3}=0.15625$
- 浮点格式在每个数量级内提供相同数目的代表值，因此保持的是相对精度
  - 数值越大，相邻代表值之间的绝对间距越大
- 这种设计以固定的相对精度换取跨越多个数量级的表示范围
  - 适合神经网络中数值跨度很大的权重、激活与梯度

#### 1.5.2 格式表

| 格式 | 位数 | E | M | 最大值 | 用途 |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | $3.4\times10^{38}$ | master weight、优化器状态、累加 |
| TF32 | 19 | 8 | 10 | 同 FP32 | Ampere 起矩阵乘的内部格式 |
| FP16 | 16 | 5 | 10 | 65504 | 老一代混合精度，需要 loss scaling |
| BF16 | 16 | 8 | 7 | 同 FP32 | 当前训练主流 |
| FP8 E4M3 | 8 | 4 | 3 | 448 | 前向权重 / 激活 |
| FP8 E5M2 | 8 | 5 | 2 | 57344 | 反向梯度 |
| FP4 E2M1 | 4 | 2 | 1 | 6 | 推理权重 / 激活，需配合块缩放 |
| MXFP8 / MXFP4 / NVFP4 | 8 / 4 | — | — | — | Blackwell 支持的块缩放格式，每组元素共享一个缩放因子 |
| INT8 / INT4 | 8 / 4 | — | — | 127 / 7 | 推理量化 |

- 表中每种格式除 E 与 M 外还有 1 位符号位
- master weight 指混合精度训练中保存的 FP32 权重副本（1.6.1 节）
- loss scaling 指为防止梯度下溢而对 loss 进行的缩放（1.6.2 节）
- TF32 并非存储格式，而是 Tensor Core 在矩阵乘内部使用的格式
  - 输入以 FP32 存储，计算时截断为 10 位尾数
  - 以 FP32 的范围、与 FP16 相同的尾数精度运算

FP8 有两种布局：
- E4M3 的尾数多一位，精度较高但最大值只有 448
  - 适合数值范围较稳定的前向权重与激活
- E5M2 的指数多一位，范围扩大到 57344
  - 适合跨度更大的反向梯度

FP4 E2M1 的可表示值：
- FP4 E2M1 只有 16 种编码
- 按其规范（指数 bias 为 1，指数字段为 0 时表示次正规数），它能表示的非负值只有 $0, 0.5, 1, 1.5, 2, 3, 4, 6$ 这 8 个
  - 加上符号共 15 个不同的值（正负零重合）
- 这 8 个值的间距在 0 附近为 0.5，在 4 到 6 之间为 2
  - 任何落在其间的数值都必须舍入到最近的一个
- 可表示的值极少，因此 FP4 必须与细粒度的缩放因子配合使用（1.5.5 节）
  - 否则大部分数值会被舍入到 0 或饱和到 6

#### 1.5.3 BF16 与 FP16 的比较

二者均为 16 位格式，区别在于指数与尾数的位数分配：
- FP16 有 5 位指数、10 位尾数，精度较高
  - 但最大值只有 65504，最小正规数约为 $6\times10^{-5}$
  - 训练中大量很小的梯度会下溢为 0
- BF16 有 8 位指数、7 位尾数，精度较低
  - 但指数位数与 FP32 相同，表示范围也与 FP32 一致
  - 梯度几乎不会下溢，通常无需 loss scaling
  - BF16 与 FP32 之间的转换只需截去低 16 位尾数，实现简单
- 训练中范围不足造成的失败远比精度不足更难处理，因此大模型训练普遍采用 BF16

#### 1.5.4 机器精度与权重更新

- 机器精度 $\epsilon = 2^{-M}$，即 1 与比 1 大的下一个可表示数之间的间距
- BF16 的 $M=7$，$\epsilon = 2^{-7}\approx 0.0078$
  - 因此在 BF16 中，凡是小于 $\epsilon/2$ 的增量加到 1 上都会被舍入抹去，例如 $1 + 0.003 = 1$

这一性质直接影响训练：
- 学习率为 $10^{-4}$ 时，每步对权重的更新量通常是权重本身的 $10^{-4}$ 到 $10^{-3}$ 量级，远小于 BF16 的相对精度
- 若直接把更新量加到 BF16 权重上，大部分更新会被舍入抹去，模型无法学习
- 因此权重更新必须在 FP32 master weight 上执行
  - FP32 的 $\epsilon = 2^{-23}\approx1.2\times10^{-7}$，足以累积这些微小的更新

#### 1.5.5 缩放粒度与块缩放

缩放的动机与做法：
- 低位宽格式的代表值很少，直接用来存储原始数值会浪费大部分编码
  - 一部分区间没有数值落入
  - 另一部分区间的数值又被挤到同一个编码上
- 缩放的做法是先用一个缩放因子（scale）$s$ 把数值变换到格式最适合表示的范围
  - 存储 $x/s$ 的量化结果，使用时再乘回 $s$
- 缩放因子通常取一组数值的最大绝对值除以格式的最大可表示值
  - 使这组数中最大的一个恰好映射到格式的上限

一个缩放因子覆盖多少个元素，称为缩放粒度。

| 粒度 | 含义 | 优点 | 代价 |
|---|---|---|---|
| Per-tensor | 整个张量共用一个 scale | 元数据最少，计算最简单 | 受离群值影响大 |
| Per-channel | 每个输出通道一个 scale | 适应不同通道的数值差异 | 需要存储额外的 scale |
| Per-group / block | 每 $g$ 个相邻元素一个 scale | 量化误差通常最小 | scale 的存储、读取与运算更复杂 |
| Dynamic | 运行时根据当前激活计算 scale | 适应输入分布的变化 | 每次需额外执行一次求最大值的归约 |

粒度的影响可以用一个例子说明：
- 设四个权重为 $[0.02, 0.03, 0.04, 3.0]$，用 FP4 E2M1（最大值 6）量化
- 若四个数共用一个 scale
  - 为容纳 3.0，取 $s = 3.0/6 = 0.5$
  - 前三个数除以 $s$ 后为 $0.04、0.06、0.08$，全部舍入为 0
  - 三个权重的信息完全丢失
- 若把它们分为 $[0.02, 0.03, 0.04]$ 与 $[3.0]$ 两组
  - 第一组取 $s = 0.04/6\approx0.0067$
  - 三个数除以 $s$ 后为 $3、4.5、6$，可以分别舍入到 3、4、6
  - 反量化后为 $0.02、0.027、0.04$，误差显著减小
- 分组越细，每组数值的跨度越小，量化误差越小
  - 但需要存储的 scale 越多，硬件在计算中处理 scale 的开销也越大

块缩放：
- FP8 与 FP4 的表示范围很小，整个张量共用一个 scale 会把大部分数值压缩到少数几个量化级别上
- 因此通常采用块缩放（block scaling）：每组 $g$ 个元素共享一个 scale
  - MX 格式为 32 个，NVFP4 为 16 个
  - DeepSeek-V3 对激活采用 $1\times128$、对权重采用 $128\times128$ 的分组

$$\hat{x}_i = \text{round}\left(\frac{x_i}{s_g}\right),\quad s_g = \frac{\max_{j\in g}|x_j|}{\text{FP8}_{\max}}$$

- MX（Microscaling）是一类把块缩放写入格式规范的数值格式
- 在 MX 格式中，每 32 个相邻元素组成一个块
  - 块内元素用 FP8、FP6 或 FP4 等低位宽格式存储
  - 整块共享一个 8 位的缩放因子，该缩放因子只编码 2 的整数次幂（E8M0 格式）
  - 缩放因子是 2 的幂，因此乘以它只需调整指数，硬件实现代价很低
- 以 MXFP4 为例，每个元素 4 位，每 32 个元素另需 8 位 scale，平均每个元素 4.25 位
- NVFP4 则把块大小缩小到 16，并用 FP8 E4M3 格式存储缩放因子
  - 以更多的元数据换取更小的量化误差

#### 1.5.6 浮点加法不满足结合律

- $(a+b)+c \ne a+(b+c)$
- 在 BF16 中，$(1 + 2^{-8}) + 2^{-8} = 1$
  - $1+2^{-8}$ 恰好位于 1 与下一个可表示数 $1+2^{-7}$ 的中点
  - 按就近舍入到偶数的规则被舍为 1，第二次相加同理
- 而 $1 + (2^{-8} + 2^{-8}) = 1 + 2^{-7} = 1.0078$
  - 先相加的两个小数之和恰好达到可表示的间距
- 只要求和顺序发生变化，结果的低位就可能不同
- 实际系统中求和顺序改变的原因包括：
  - 并行度改变导致归约树的形状不同
  - 更换了 kernel 或 kernel 选择了不同的分块方式
  - atomic 操作（多个线程对同一地址的原子累加）的完成先后不确定
- 4.10 节讨论的确定性训练问题均源于此

```python
import torch
a = torch.tensor([1.0, 2**-8, 2**-8], dtype=torch.bfloat16)
print((a[0] + a[1]) + a[2])   # tensor(1., dtype=torch.bfloat16)
print(a[0] + (a[1] + a[2]))   # tensor(1.0078, dtype=torch.bfloat16)
```

#### 1.5.7 均匀量化与量化误差

均匀量化（uniform quantization）把实数映射到等间隔的整数网格上。对称均匀量化的形式为

$$q=\operatorname{clip}\left(\operatorname{round}(x/s),\ q_{\min},\ q_{\max}\right),\qquad \hat x=s\,q$$

各符号含义：
- $s$ 为缩放因子，决定一个整数单位对应多大的实数间隔
- round 把 $x/s$ 舍入到最近的整数
- clip 把超出格式范围的值截断到 $q_{\min}$ 或 $q_{\max}$，这一操作称为饱和（saturation）
- $\hat x$ 为反量化（dequantization）得到的近似值
- 量化误差为 $e = x - \hat x$
  - 常用均方误差 $\frac1n\sum_i(x_i-\hat x_i)^2$ 衡量一个张量的整体误差

算例：
- 取 $s=0.25$，$x=0.63$，则 $x/s = 2.52$，舍入后 $q=3$，$\hat x = 0.75$，误差为 $-0.12$
  - 舍入误差的绝对值不超过 $s/2 = 0.125$
- 若某个离群值 $x=20$ 超出 INT4 的范围（$q_{\max}=7$，对应 $\hat x$ 最大为 1.75）
  - 则被饱和为 1.75，误差高达 18.25，远大于舍入误差
- $s$ 的选择因此是一个权衡
  - $s$ 大，饱和少但舍入误差大
  - $s$ 小，舍入误差小但更多数值被饱和

均方误差的局限：
- 均方误差是一个方便计算的代理指标，但它与模型最终的准确率并不完全一致
- 某些层（如注意力的输出投影、模型的首尾层）对误差更敏感
- 少数离群值对输出的影响也可能远大于其对均方误差的贡献
- 因此量化方案最终需要以任务准确率来评估

#### 1.5.8 非均匀码本量化与率失真权衡

- 神经网络的权重通常集中分布在 0 附近，只有少数权重的绝对值较大
- 均匀量化在整个范围内等间隔放置代表值
  - 使大量编码落在很少有权重出现的区间
  - 而权重密集的 0 附近区间反而分辨率不足
- 非均匀量化让代表值的分布与数据分布匹配，在高概率区间放置更密集的代表值

设代表值的集合（码本，codebook）为 $\mathcal C=\{c_1,\ldots,c_K\}$，量化操作把每个数值映射到码本中距离最近的元素：

$$Q(x)=c_{\arg\min_j |x-c_j|}$$

- 例如码本可以取 $\{-1, -0.25, -0.05, 0, 0.05, 0.25, 1\}$
  - 在 0 附近的间距为 0.05，在远离 0 处的间距为 0.75
- 码本可以通过统计权重分布、最小化重构误差（如 k-means 聚类）或直接最小化任务损失来设计
- NF4（NormalFloat4）即按正态分布的分位数放置 16 个代表值的码本格式

非均匀量化的代价在硬件计算上：
- 乘法器只能直接处理标准数值格式
- 码本量化存储的是码本索引而非数值本身
  - 计算前必须通过查找表（lookup table）把索引转换为实际数值，再送入乘法器
  - 这一步需要额外的指令或专用电路，也使数据无法直接送入 Tensor Core
- 因此码本量化目前主要用于存储压缩
  - 权重以码本索引的形式存放在显存中以减少容量与带宽
  - 计算前在片上解码为 BF16 或 FP8

量化方案的设计可以用率失真（rate–distortion）的框架描述：
- 率（rate）指编码每个数值所需的位数
- 失真（distortion）指近似引入的信息损失
- 在给定的位数预算下，目标是使失真尽可能小
- 反之，在给定的失真容忍度下，目标是使位数尽可能少
- 对 AI 硬件而言还需要加入第三个维度：硬件执行的代价
  - 最终的方案需要在模型质量、存储与带宽、计算电路的复杂度三者之间折中
  - 失真最小的格式未必是实际系统中最优的格式

#### 1.5.9 量化矩阵乘的计算方式

- 神经网络的核心运算是 $y=Wx$
- 硬件中执行这一运算的基本单元是 MAC（Multiply–Accumulate，乘累加），每次执行 $a\leftarrow a+w_i x_i$
- 若权重与输入分别以缩放因子 $s_w$、$s_x$ 量化为整数 $q_{w,i}$、$q_{x,i}$，即 $w_i\approx s_w q_{w,i}$、$x_i\approx s_x q_{x,i}$，则

$$y\approx s_w s_x\sum_i q_{w,i}\,q_{x,i}$$

- 缩放因子对一组元素是常数，可以提到求和号之外
  - 硬件因此可以先用低位宽的整数 MAC 完成全部乘累加
  - 最后只做一次乘以 $s_w s_x$ 的缩放
- 低位宽乘法器的面积与能耗远小于浮点乘法器
  - INT8 乘法的能耗不到 FP32 乘法的 1/10
  - 这是量化计算节能的来源

这一推导的前提与限制：
- 前提是同一次求和中的所有元素共享缩放因子
- 采用块缩放时，每个块有自己的 scale
  - 硬件需要在每个块内先完成整数累加
  - 乘以该块的 scale 后再与其他块的结果相加
- 累加器的位宽必须足够宽
  - 两个 INT8 相乘的结果需要 16 位，数千次累加后需要 32 位才能避免溢出
- 这些因素共同决定了量化 Tensor Core 的电路设计

### 1.6 混合精度训练与 loss scaling

#### 1.6.1 AMP 的标准流程

AMP（Automatic Mixed Precision，自动混合精度）指在训练中的一种方案：
- 矩阵乘等计算密集的操作使用低精度
- 数值敏感的操作保留高精度

标准流程来自 Micikevicius et al. (2018)：

1. 保留一份 FP32 的 master weights，即权重的高精度主副本，参数更新始终在这份副本上进行。
2. 每一步把 master weights 转换为 FP16/BF16，用于前向与反向计算
   - 矩阵乘在 Tensor Core 上执行，乘法以低精度进行，内部累加以 FP32 进行
   - 这样避免大量乘积相加时损失精度
3. 把反向得到的低精度梯度转换回 FP32，在 FP32 中完成参数更新，原因见 1.5.4 节。
4. softmax、LayerNorm、loss 等数值敏感的操作在 FP32 中计算。这些操作包含指数、求和与除法，低精度下容易溢出或产生较大的舍入误差。

该方案使矩阵乘获得低精度的速度与显存收益，同时把精度损失控制在不影响收敛的范围内。

#### 1.6.2 Loss scaling

- FP16 的最小正规数约为 $6\times10^{-5}$，而训练中大量梯度值小于这一数值
  - 转换为 FP16 后会下溢为 0，对应的参数无法得到更新
- loss scaling 的做法是在反向传播之前把 loss 乘以一个缩放因子 $S$
  - 根据链式法则，所有梯度随之放大 $S$ 倍，原本过小的梯度被移入 FP16 可表示的范围
  - 在参数更新之前再除以 $S$，恢复梯度的真实大小：

$$g = \frac{1}{S}\nabla_\theta (S\cdot L)$$

- 例如取 $S = 2^{16} = 65536$，一个 $10^{-7}$ 量级的梯度被放大到约 $6.6\times10^{-3}$，可以在 FP16 中正常表示

#### 1.6.3 动态 loss scaling

- $S$ 取得过小，梯度仍会下溢
- $S$ 取得过大，较大的梯度会超过 FP16 的最大值 65504 而上溢为 inf
- 合适的 $S$ 还会随训练进程变化
- 动态 loss scaling 在训练中自动调整 $S$
  - 一旦检测到梯度中出现 inf 或 NaN，说明发生了上溢，跳过本步更新并把 $S$ 减半
  - 若连续若干步（如 2000 步）未发生溢出，则把 $S$ 加倍，尝试利用更大的表示范围

```python
class DynamicLossScaler:
    def __init__(self, init_scale=2.**16, growth=2.0, backoff=0.5, interval=2000):
        self.scale, self.growth, self.backoff, self.interval = init_scale, growth, backoff, interval
        self.good_steps = 0

    def step(self, params, optimizer):
        grads = [p.grad for p in params if p.grad is not None]
        found_inf = any(not torch.isfinite(g).all() for g in grads)
        if found_inf:                      # 溢出：跳过本步，缩小 scale
            self.scale *= self.backoff
            self.good_steps = 0
            optimizer.zero_grad()
            return False
        for g in grads:                    # 正常：反缩放后更新
            g.div_(self.scale)
        optimizer.step()
        self.good_steps += 1
        if self.good_steps % self.interval == 0:   # 连续 interval 步无溢出，放大
            self.scale *= self.growth
        return True

# loss = model(x); (loss * scaler.scale).backward(); scaler.step(params, opt)
```

- PyTorch 中对应的实现为 `torch.amp.GradScaler`
- BF16 的表示范围与 FP32 相同，梯度几乎不会下溢，因此 BF16 训练通常不需要这一机制

#### 1.6.4 FP8 训练

Transformer Engine（NVIDIA 提供的 FP8 训练库）与 DeepSeek-V3 采用的方案如下：

| 环节 | 做法 |
|---|---|
| GEMM | FP8 输入，FP32 累加 |
| Scale 管理 | delayed scaling（依据历史 amax 估计 scale），或 current / 块级 scaling |
| 保留高精度 | embedding、输出层、norm、attention softmax、MoE gating、优化器状态 |
| 累加精度 | DeepSeek-V3 发现 H800 上 FP8 GEMM 的累加器只保留约 14 位精度，于是每累加 128 个元素就把部分和提升到 CUDA core 的 FP32 寄存器中继续累加 |

FP8 的表示范围远小于 BF16，每个张量都必须配合缩放因子使用。scale 的确定方式有两类：
- delayed scaling 记录每个张量最近若干步的 amax（张量中绝对值的最大值），用历史 amax 计算当前步的 scale
  - 优点是不需要在计算前额外扫描一遍张量
  - 缺点是数值分布突变时 scale 可能不合适
- current scaling 在每次计算前统计当前张量的 amax
  - 更准确但多一次归约开销
- 块级 scaling 为每个块单独计算 scale（1.5.5 节）
  - 能更好地处理离群值

并非所有操作都适合 FP8：
- 以下部分都保留 BF16 或 FP32
  - embedding 与输出层直接决定 token 的表示与预测概率
  - norm、softmax 与 MoE 的 gating 涉及归一化与指数运算
  - 优化器状态需要累积微小的更新
- 累加精度问题则说明，即使输入是 FP8，累加器的精度不足也会在长求和中积累误差
  - 需要定期把部分和转移到更高精度的寄存器中

#### 1.6.5 随机舍入

- 常规的舍入方式是舍入到最近的可表示值（round-to-nearest）
  - 它对单个数值误差最小，但对重复的微小增量存在系统性偏差
  - 若每次增量都小于相邻可表示值间距的一半，每次都会被舍去，累积再多次也不改变结果，如 1.5.4 节所述

随机舍入（stochastic rounding）按数值到两个相邻可表示值的距离，以相应概率向上或向下舍入：
- 设 $x$ 位于相邻可表示值 $a<x<b$ 之间
  - 以概率 $(x-a)/(b-a)$ 舍入为 $b$
  - 以概率 $(b-x)/(b-a)$ 舍入为 $a$
- 这样舍入结果的期望恰好等于 $x$，舍入误差在多次累积后相互抵消而不再系统性地丢失
- 例如在间距为 1 的格式中把 0.3 累加 10 次
  - 舍入到最近值时每次都被舍去，结果仍为 0
  - 随机舍入时每次有 30% 的概率加 1，结果的期望为 3

随机舍入的作用与代价：
- 随机舍入使低精度格式能够在一定程度上累积小于其精度的更新，因此在低精度训练与 FP4 训练中受到关注
- 它的代价是需要硬件生成随机数，并且单次结果的方差大于确定性舍入
- 它降低了系统性偏差，但不能单独保证训练收敛
  - 通常仍需与高精度主权重或其他手段配合使用

### 1.7 显存占用分析：训练与推理

#### 1.7.1 训练：模型状态

使用 Adam 优化器与 BF16 混合精度训练时，每个参数占用的字节数如下：

| 项 | 字节 |
|---|---|
| BF16 权重 | 2 |
| BF16 梯度 | 2 |
| FP32 master 权重 | 4 |
| Adam 一阶矩 $m$ | 4 |
| Adam 二阶矩 $v$ | 4 |
| 合计 | 16 |

- Adam 为每个参数维护两个状态
  - 一阶矩 $m$ 是梯度的指数移动平均
  - 二阶矩 $v$ 是梯度平方的指数移动平均
  - 二者都需要累积微小的变化，因此以 FP32 保存
- 权重与梯度各 2 字节，FP32 master 权重与 Adam 的两个状态共 12 字节
  - 即 ZeRO 论文中的 $2\Psi+2\Psi+12\Psi$（$\Psi$ 为参数量）
- 一个 7B 模型的模型状态为 $7\times10^9\times16 = 112$ GB
  - 单块 80 GB 的 GPU 无法容纳，尚未计入激活
- 这是即使中等规模的模型也必须采用分布式训练或状态分片的直接原因

#### 1.7.2 训练：激活

- 激活指前向传播中为反向计算梯度而保存的中间张量
- 根据 Korthikanti et al. (2022)，在 BF16、不做重算、不使用 TP/SP（TP 为张量并行，SP 为序列并行，见第 3 章）的条件下，每个 Transformer 层的激活显存为

$$M_{\text{act}}^{\text{layer}} = sbh\left(34 + \frac{5as}{h}\right)\ \text{bytes}$$

- 其中 $s$ 为序列长度，$b$ 为 micro-batch 大小，$h$ 为隐藏维度，$a$ 为注意力头数
- $34sbh$ 一项来自各线性层的输入、norm 的输入、dropout mask 等与 $h$ 成正比的张量
- $5as^2b$ 一项来自注意力分数矩阵及其 softmax 结果，它们的大小与 $s^2$ 成正比
- 使用 FlashAttention 后，注意力分数矩阵不再显式存储，反向时按块重新计算，第二项消失

算例：Llama-2-7B（$h=4096$，32 层），$s=4096$，$b=1$，启用 FlashAttention：

$$34\times4096\times4096\times32 \approx 18.3\ \text{GB}$$

- 激活显存与序列长度和 micro-batch 成正比，序列长度翻倍则激活翻倍
- 启用 TP=$t$ 并配合 SP 后，每块 GPU 的激活降至约 $34sbh/t$
- 采用全量重算时，每层只需保存该层的输入，即 $2sbh$
  - 其余激活在反向时重新计算
  - 代价是增加约一次前向的计算量（0.3.1 节）

#### 1.7.3 推理

$$M_{\text{infer}} = N\cdot p_w + 2\cdot L\cdot n_{kv}\cdot d_h\cdot p_{kv}\cdot T_{\text{total}} + M_{\text{act}}$$

符号含义：
- $N$ 为参数量，$p_w$ 与 $p_{kv}$ 分别为权重与 KV cache 每个元素的字节数
- $L$ 为层数，$n_{kv}$ 为 KV 头数，$d_h$ 为每个头的维度
- $T_{\text{total}}$ 为所有并发请求的 token 总数
- 系数 2 对应 Key 与 Value 两份张量

各项特点：
- 第一项为权重，大小固定
- 第二项为 KV cache，随并发数与上下文长度线性增长
- 第三项为激活
  - 推理时不需要为反向保存激活，只需保留当前层的中间结果
  - 通常远小于前两项

#### 1.7.4 每 token 的 KV cache

表中的注意力变体决定了 $n_{kv}$ 的大小：
- MHA（Multi-Head Attention，多头注意力）中每个查询头对应一组独立的 KV 头
- GQA（Grouped-Query Attention，分组查询注意力）让多个查询头共享一组 KV 头
  - 从而减少需要缓存的 KV 头数
- MLA（Multi-head Latent Attention，多头潜在注意力）把 K 与 V 压缩为一个低维的潜在向量后再缓存

| 模型 | $L$ | $n_{kv}$ | $d_h$ | 每 token（BF16） |
|---|---|---|---|---|
| Llama-2-7B（MHA） | 32 | 32 | 128 | 512 KB |
| Llama-3-8B（GQA） | 32 | 8 | 128 | 128 KB |
| Llama-3-70B（GQA） | 80 | 8 | 128 | 320 KB |
| DeepSeek-V3（MLA） | 61 | latent 512 + RoPE 64 | — | $61\times576\times2 \approx 70$ KB |

- 以 Llama-2-7B 为例，每 token 的 KV cache 为 $2\times32\times32\times128\times2 = 524288$ 字节，即 512 KB
- Llama-3-8B 的层数与头维度相同，但 KV 头数从 32 减为 8
  - KV cache 相应降为 1/4
- DeepSeek-V3 每层只缓存 512 维的潜在向量与 64 维的位置编码分量
  - 尽管模型规模大得多，每 token 的 KV cache 反而最小

Llama-3-70B 的 KV cache 规模：
- 在 32 个并发请求、每个请求 8K 上下文的条件下，KV cache 占用 $32\times8192\times320\text{ KB}\approx 84$ GB
  - 超过其 BF16 权重（140 GB）的一半
- 在长上下文推理中，KV cache 是显存的主要占用项
  - 它直接限制了一块 GPU 能同时服务的请求数，进而决定吞吐与成本（0.3.5 节）

#### 1.7.5 权重存储与位宽

一个含 $N$ 个参数、每个参数用 $b$ 位存储的模型，其权重占用约

$$M_w \approx \frac{N\cdot b}{8}\ \text{bytes}$$

- 70B 模型以 BF16 存储时，$70\times10^9\times16/8 = 140$ GB，至少需要两块 80 GB 的 GPU
- 以 4 位存储时降为 35 GB，一块 GPU 即可容纳并留出约 45 GB 给 KV cache
- 该式给出的是理论下限，实际占用还需加上缩放因子等元数据
  - 例如 MXFP4 每 32 个元素附带 8 位 scale，平均每个参数 4.25 位

位宽降低的其他收益：
- 位宽的降低同时减少了带宽需求
  - decode 阶段每生成一个 token 都要把全部权重从 HBM 读一遍（1.3.5 节）
  - 在带宽受限的条件下，单步耗时近似与权重字节数成正比
  - 权重从 16 位降到 4 位，单步的权重读取时间可缩短至约 1/4
- 数据在存储层级之间的移动所消耗的能量往往高于对其进行的算术运算
  - 因此减少字节数也降低了能耗
- 这些收益成立的前提：
  - 硬件支持高效的低精度计算或片上解码
  - 量化与反量化本身的开销不抵消节省的时间

### 1.8 实验：测量显存带宽与算力

#### 1.8.1 测试脚本

```python
import torch

def bench(fn, iters=50):
    for _ in range(5): fn()                      # warmup
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # 秒/次

# 1) 显存带宽：拷贝 n 字节 = 读 n + 写 n
n = 1 << 30
x = torch.empty(n, dtype=torch.uint8, device="cuda"); y = torch.empty_like(x)
t = bench(lambda: y.copy_(x))
print(f"HBM copy bandwidth: {2 * n / t / 1e12:.2f} TB/s")

# 2) BF16 GEMM 算力
for M in [512, 1024, 2048, 4096, 8192]:
    a = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)
    t = bench(lambda: a @ b)
    print(f"M={M:5d}  {2 * M**3 / t / 1e12:7.1f} TFLOPS   AI={M/3:.0f}")
```

- 脚本中的 `bench` 函数先执行 5 次预热，使 GPU 完成时钟提升与 kernel 选择
  - 再用 CUDA event 记录 50 次执行的平均耗时
- GPU 上的操作是异步提交的
  - 必须调用 `torch.cuda.synchronize()` 等待其完成后再读取时间
  - 否则测得的只是 CPU 端提交指令的时间
- 带宽测试用 1 GB 的拷贝，读写各 1 GB
- 算力测试用不同边长的方阵乘
  - 输出中的 `AI` 为按 1.3.4 节公式计算的算术强度

#### 1.8.2 预期结果（H100 SXM）

| 测试 | 典型值 |
|---|---|
| 拷贝带宽 | 2.8–3.0 TB/s，峰值的 85–90% |
| GEMM M=512 | 远低于峰值，kernel 启动开销与带宽占主导 |
| GEMM M≥4096 | 650–800 TFLOPS，峰值的 70–80% |

- 实测带宽达不到标称峰值，原因包括 DRAM 的刷新、行切换等协议开销
- $M=512$ 时算术强度约为 171，低于脊点 295，处于带宽受限区
  - 同时总计算量只有约 0.27 GFLOP，kernel 耗时仅为微秒量级
  - 启动开销在其中占有不可忽略的比例
- $M\ge4096$ 时算术强度超过 1000，进入算力受限区
  - 实测值接近但达不到峰值
  - 差距来自 wave quantization、流水线的启动与收尾，以及高负载下 GPU 因功耗限制而降频

#### 1.8.3 绘制实测 roofline

- 把不同 $M$ 下的实测点（算术强度, TFLOPS）与理论上限 $\min(P_{\text{peak}}, I\times B)$ 画在双对数坐标上
- 实测点在算术强度较低时沿带宽斜线上升，越过脊点后趋于平坦，并贴近算力上限的下方
- 实测点与理论上限之间的距离，即为该 kernel 实现的优化空间

### 1.9 小结

- 单块 GPU 有两个性能上限：算力与显存带宽
  - 二者之比 $I^*$ 为脊点，H100 BF16 约为 295 FLOP/Byte
  - 算术强度低于脊点的程序受带宽限制，高于脊点的受算力限制
- 逐元素操作与小 batch 的 decode 受带宽限制，只有大规模 GEMM 才受算力限制
  - 提高性能的方法取决于瓶颈类型
  - 带宽受限时减少字节数，算力受限时使用 Tensor Core 与更低精度
- 数值格式决定每个数值占用的字节数。低位宽格式必须配合缩放因子使用，缩放粒度越细误差越小、元数据越多；FP4 等极低位宽格式依赖块缩放。
- 混合精度训练的模型状态为每参数 16 字节；长上下文推理中 KV cache 是显存的主要占用项。
- 浮点加法不满足结合律，这是确定性问题的根源。

<br>

## 2. 单机优化：访存、kernel 融合与显存管理

单块 GPU 上的性能瓶颈与显存不足，可以从三个方向处理：
- 减少 HBM 访存字节数（算子融合、FlashAttention）
- 减少 kernel 启动次数（算子融合、CUDA Graph）
- 减少显存碎片（缓存分配器配置）
- 此外，激活重算以额外计算换取显存

### 2.1 基本概念：kernel、kernel launch 与算子融合

#### 2.1.1 术语定义

| 术语 | 定义 |
|---|---|
| Kernel | 在 GPU 上由大量线程并行执行的函数。PyTorch 的每个算子在底层至少对应一个 kernel |
| Kernel launch | CPU 将 kernel 及其参数提交到 GPU 执行队列的操作，每次产生约 3–10 µs 的 CPU 侧开销 |
| Stream | GPU 上的有序执行队列。同一 stream 内的操作按提交顺序执行，不同 stream 之间可以并发，计算与通信的重叠依赖多个 stream 实现 |
| 算子融合（fusion） | 将多个 kernel 合并为一个 kernel，使中间结果保留在寄存器或 SMEM（共享内存，即 SM 内的片上 SRAM）中，不写回 HBM |

#### 2.1.2 launch-bound

- GPU 相对 CPU 异步执行
  - 若 CPU 提交 kernel 的速度高于 GPU 执行的速度，launch 开销被 GPU 执行时间掩盖
  - 反之，GPU 执行完当前 kernel 后需要等待 CPU 提交下一个
    - 此时整体性能受 CPU 侧提交速度限制，称为 CPU-bound 或 launch-bound
- launch-bound 在 Nsight Systems 中的特征
  - GPU 时间线上存在大量空隙
  - 单个 kernel 只持续数微秒
  - CPU 线程时间线接近满载
- 该现象多见于小模型推理和小 batch 的 decode 阶段
  - 对应的优化手段是算子融合与 CUDA Graph（6.6 节）

以一个数量级估算说明其影响：
- 设某模型的一个 decode step 包含约 1000 个 kernel，每次 launch 的 CPU 侧开销为 5 µs
  - 则仅提交这些 kernel 就需要约 5 ms
- 若这些 kernel 在 GPU 上的执行时间合计只有 3 ms，GPU 约有 40% 的时间处于空闲
  - 此时提高 GPU 算力不会缩短 step 时间
- 只有两类手段有效
  - 减少 kernel 数量（融合）
  - 消除逐个提交的开销（CUDA Graph 将整段 kernel 序列录制后一次性提交）

### 2.2 算子融合的访存收益

#### 2.2.1 示例：bias 加法与 GeLU

`y = gelu(x + bias)`，$x$ 含 $n$ 个 BF16 元素（每元素 2 字节）：

| 实现方式 | Kernel 数 | HBM 读写量 |
|---|---|---|
| 不融合：`t = x + bias`; `y = gelu(t)` | 2 | 读 $x$ + 写 $t$ + 读 $t$ + 写 $y$ = $8n$ 字节 |
| 融合 | 1 | 读 $x$ + 写 $y$ = $4n$ 字节 |

- 中间张量 $t$ 先写回 HBM、再由下一个 kernel 读出的过程称为一次 HBM 往返（round trip）
  - 融合消除的正是这类往返
- 两个操作的算术强度都远低于脊点（算力峰值与带宽之比，见 1.3.3）
  - 属于带宽受限操作，耗时近似正比于 HBM 读写字节数
  - 融合后字节数减半，耗时也约减半
- 一个 Transformer 层中有十余个逐元素操作（bias、残差、norm、激活、dropout、RoPE）
  - 全部融合可以显著缩短该层非 GEMM 部分的耗时

#### 2.2.2 常见融合模式

| 模式 | 示例 | 收益来源 |
|---|---|---|
| 逐元素链 | bias + GeLU + dropout | 中间张量不写回 HBM |
| 归约 + 逐元素 | residual add + RMSNorm | 输入只读一次，在片上完成归一化 |
| GEMM epilogue | matmul + bias + 激活 | 在 GEMM 结果写回前完成（CUTLASS epilogue） |
| 注意力 | QK^T + mask + softmax + ×V | $s\times s$ 矩阵不写回 HBM |
| 跨算子 | MoE 的 permute + GEMM + unpermute | 减少大张量的 HBM 读写 |

- GEMM epilogue 指 GEMM kernel 在累加完成之后、结果写回 HBM 之前执行的后处理阶段
  - 逐元素操作放在这一阶段执行不产生额外的 HBM 访问

各模式的收益机制如下：
- 逐元素链
  - 由多个逐点操作首尾相接组成，每个操作的输出只被下一个操作使用一次
  - 未融合时，链上每个中间张量都要经历一次 HBM 往返
  - 融合后整条链只读一次输入、写一次输出
  - 链越长，节省比例越高：长度为 $k$ 的链，HBM 读写量从约 $2k$ 份张量降至 2 份
- 归约与逐元素的融合，以 RMSNorm 为典型
  - RMSNorm 需要先对一行求平方和（归约），再用结果缩放该行每个元素（逐元素）
  - 若分两个 kernel 实现，同一行数据要从 HBM 读取两次
  - 融合后，一个程序实例把整行加载到寄存器，完成归约后直接在寄存器中缩放并写出，输入只读一次
  - 前面的 residual add 也可以并入同一 kernel，2.7 节给出该融合链的完整访存量计算
- GEMM epilogue 融合利用了 GEMM 的结构
  - GEMM 的累加结果本来就在寄存器中，写回之前加上 bias、施加激活函数几乎不增加耗时
  - 若 bias 和激活作为独立 kernel 执行，需要额外读写一次大小为输出矩阵的张量
- 注意力融合
  - 将 $QK^\top$、mask、softmax 和与 $V$ 的乘积合并为一个 kernel
  - 使 $s\times s$ 的分数矩阵始终不出现在 HBM 中
  - 这是 2.3 节 FlashAttention 的内容
- 跨算子融合，针对 MoE 等结构中的数据重排
  - MoE 层需要先按专家对 token 重新排列（permute），再做矩阵乘，最后恢复原顺序（unpermute）
  - 重排本身是纯访存操作
  - 把它并入 GEMM 的加载与写回阶段，可以省去两次对整个激活张量的读写

### 2.3 FlashAttention：IO 感知的精确注意力

#### 2.3.1 标准注意力的访存问题

$$S = \frac{QK^\top}{\sqrt{d}},\quad P = \text{softmax}(S),\quad O = PV$$

- $Q,K,V\in\mathbb{R}^{N\times d}$，$S$、$P$ 均为 $N\times N$
- $N=8192$、32 头、BF16 时，单层的 $S$ 占用 $8192^2\times32\times2\approx 4.3$ GB
- 标准实现将 $S$、$P$ 写回 HBM 后再读出，HBM 访问量为 $\Theta(Nd + N^2)$

标准注意力的计算量与存储量需要分开分析：
- 计算量方面
  - $QK^\top$ 需要对 $N^2$ 个 query-key 对各做一次 $d$ 维点积
  - $PV$ 同样是 $N^2 d$ 量级的乘加
  - 因此计算量为 $O(N^2 d)$
- 存储量方面
  - 若显式保存 $S$ 或 $P$，需要 $O(N^2)$ 的空间，与 $d$ 无关
- 序列长度 $N$ 从 8K 增加到 32K 时，两者都增加为原来的 16 倍
  - 但 $O(N^2)$ 的存储先成为瓶颈
  - 上例中单层的 $S$ 将从 4.3 GB 增至约 69 GB，超出单块 GPU 的显存容量

关于 $O(N^2)$ 存储的适用范围：
- 上述 $O(N^2)$ 存储针对的是在 HBM 中物化完整矩阵的实现
- 注意力的数学定义本身并不要求物化 $S$ 和 $P$
- 因此不能把定义式的存储需求直接等同于所有实现的 HBM 占用
- FlashAttention 正是利用了这一点

#### 2.3.2 分块计算思路

- 分块（tiling）指将矩阵划分为能够放入 SMEM 的子块，逐块加载并计算
- FlashAttention 对 $Q$、$K$、$V$ 分块
  - 使 $S$ 和 $P$ 在任一时刻只有一个子块驻留于 SMEM，不写回 HBM
- 难点在于 softmax 需要整行的最大值与指数和
  - 分块计算时一次只能看到一行的一部分
- 这一问题由 online softmax 解决
  - online softmax 对每一行维护运行中的最大值与指数和
  - 新块到来时对已有结果做缩放修正
  - 最终结果与一次性计算整行完全相同

#### 2.3.3 Online softmax 推导

先回顾 softmax 的数值稳定写法：
- 定义式为 $p_i = e^{x_i}/\sum_j e^{x_j}$
  - 当 $x_i$ 较大时 $e^{x_i}$ 会溢出（FP32 下 $x > 88$ 即超出表示范围，BF16 同样如此）
- 实际实现先减去行最大值 $m = \max_j x_j$：

$$p_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

- 分子分母同乘 $e^{-m}$，结果不变
  - 所有指数的参数都不大于 0，不会溢出
- 问题在于 $m$ 是整行的最大值
  - 分块计算时处理第一块时尚不知道后续块中是否有更大的值
- online softmax 的做法
  - 先用当前已知的最大值计算
  - 发现更大的值后，再对已有结果乘以修正因子 $e^{m - m'}$（$m'$ 为更新后的最大值）
  - 推导如下

将一行分数分为两块 $s = [s^{(1)}, s^{(2)}]$，记 $m^{(j)} = \max s^{(j)}$，$\ell^{(j)} = \sum e^{s^{(j)} - m^{(j)}}$，则

$$m = \max(m^{(1)}, m^{(2)}),\qquad \ell = e^{m^{(1)}-m}\ell^{(1)} + e^{m^{(2)}-m}\ell^{(2)}$$

因为

$$\sum_i e^{s_i - m} = \sum_{i\in 1} e^{s_i - m^{(1)}}e^{m^{(1)}-m} + \sum_{i\in 2} e^{s_i - m^{(2)}}e^{m^{(2)}-m}$$

输出同理：

$$O = \frac{e^{m^{(1)}-m}\ell^{(1)}O^{(1)} + e^{m^{(2)}-m}\ell^{(2)}O^{(2)}}{\ell},\qquad O^{(j)} = \text{softmax}(s^{(j)})V^{(j)}$$

因此，每处理一个 K/V 块只需维护三个量：
- 当前最大值 $m$
- 当前分母 $\ell$
- 未归一化的输出累加 $\tilde O$

新块到来时的更新方式：
- 先将旧的 $\ell$ 与 $\tilde O$ 乘以缩放因子 $e^{m_{\text{old}}-m_{\text{new}}}$
- 再加上新块的贡献

#### 2.3.4 FA2 前向算法

```text
for each Q 块 Q_i (B_r 行)：                     # 外循环的各次迭代分配给不同线程块并行
    m_i = -inf, l_i = 0, O_i = 0
    for each K/V 块 K_j, V_j (B_c 行)：
        S_ij = Q_i K_j^T / sqrt(d)               # B_r × B_c，在 SMEM
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)
        l_i = exp(m_i - m_new) * l_i + rowsum(P_ij)
        O_i = exp(m_i - m_new) * O_i + P_ij V_j
        m_i = m_new
    O_i = O_i / l_i
    写回 O_i 和 logsumexp L_i = m_i + log(l_i)    # L_i 供反向传播使用
```

#### 2.3.5 NumPy 参考实现

```python
import numpy as np

def attention_ref(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1])
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P / P.sum(axis=1, keepdims=True)) @ V

def flash_attention(Q, K, V, Br=64, Bc=64):
    N, d = Q.shape
    O = np.zeros_like(Q); scale = 1 / np.sqrt(d)
    for i in range(0, N, Br):
        Qi = Q[i:i+Br]
        m = np.full(len(Qi), -np.inf); l = np.zeros(len(Qi)); Oi = np.zeros((len(Qi), d))
        for j in range(0, N, Bc):
            Sij = Qi @ K[j:j+Bc].T * scale
            m_new = np.maximum(m, Sij.max(axis=1))
            Pij = np.exp(Sij - m_new[:, None])
            alpha = np.exp(m - m_new)                 # 旧值的缩放因子
            l = alpha * l + Pij.sum(axis=1)
            Oi = alpha[:, None] * Oi + Pij @ V[j:j+Bc]
            m = m_new
        O[i:i+Br] = Oi / l[:, None]
    return O

Q, K, V = (np.random.randn(256, 64) for _ in range(3))
print(np.abs(flash_attention(Q, K, V) - attention_ref(Q, K, V)).max())   # ~1e-15
```

#### 2.3.6 IO 复杂度

SMEM 容量为 $M$ 时，FlashAttention 的 HBM 访问量为

$$\Theta\left(\frac{N^2d^2}{M}\right)$$

- 标准实现为 $\Theta(Nd + N^2)$
- $d=128$、$M\approx100$ KB 时 $d^2/M\ll1$，访问量低约一个数量级
- 论文还证明，对 $M\in[d, Nd]$，任何精确注意力算法的 HBM 访问量都不可能渐近低于该值

FlashAttention 减少的是 HBM 访问量和临时存储，而不是浮点运算量：
- 它计算的仍是精确的标准注意力
  - $N^2$ 个 query-key 对的点积一个也没有省略，计算量仍为 $O(N^2 d)$
  - 反向传播中重算 $S$、$P$（2.3.7 节）还使总 FLOPs 略有增加
- 它加速的原因是标准实现在长序列下受 HBM 带宽限制
  - $S$ 和 $P$ 的读写量远超 $Q$、$K$、$V$、$O$ 本身，GPU 大部分时间在等待访存
  - FlashAttention 去掉这部分访存后，注意力从带宽受限转为计算受限，实际耗时随之下降
- 因此，FlashAttention 不改变注意力的渐近计算复杂度
  - 若需要把序列维的计算复杂度降到线性，需要改变注意力函数本身（如线性注意力）
  - 那属于模型结构的改动，结果与标准注意力不再等价

#### 2.3.7 反向传播中的重算

- FlashAttention 前向不保存 $P$，只保存 $O$ 与 logsumexp $L$
- 反向传播时由 $Q$、$K$ 和 $L$ 重新计算 $S$、$P$
- 在片上重算的耗时低于从 HBM 读取 $N\times N$ 矩阵的耗时，即以额外计算换取访存带宽

#### 2.3.8 版本演进

| 版本 | 改进 | H100 BF16 前向利用率 |
|---|---|---|
| FA1 (2022) | 分块 + online softmax + 反向重算 | — |
| FA2 (2023) | 外循环改为 Q 块（沿序列并行）、减少非 matmul FLOPs、改进 warp 间的工作划分 | ~35% |
| FA3 (2024) | TMA 异步数据传输、WGMMA、warp specialization、softmax 与 GEMM 乒乓重叠、FP8 | ~75% |

表中术语的含义：
- TMA（Tensor Memory Accelerator）是 Hopper 架构上在 HBM 与 SMEM 之间异步传输数据块的硬件单元
- WGMMA 是以 warpgroup（4 个 warp）为单位发射的异步矩阵乘指令
- warp specialization 指将同一线程块内的 warp 分为数据加载与计算两类角色，使两者并行推进

此后的工作包括：
- Blackwell 上的 FA4
- 面向 decode 的 Flash-Decoding
  - 沿 KV 长度切分，最后用同一套 $m,\ell$ 公式合并各分段结果
- FlashInfer
- FlashMLA

### 2.4 激活重算与 offload：以计算和传输换取显存

#### 2.4.1 激活重算

- 激活重算（activation recomputation，又称梯度检查点）指前向传播时不保存中间激活，只保存每层（或每段）的输入
  - 反向传播时重新执行该段的前向计算以恢复所需激活

| 方式 | 保存内容 | 额外计算 |
|---|---|---|
| 全量重算 | 每层输入 $2sbh$ | 多一次前向，训练计算量从 $6N$ 增至 $8N$，+33% |
| 选择性重算（Korthikanti 2022） | 只丢弃显存占用大而重算开销小的部分，如 softmax 输出与 dropout mask | 2–3% |

全量重算的额外开销可以从训练计算量的构成推出：
- 对参数量为 $N$ 的模型，每个 token 的前向传播约需 $2N$ FLOPs
- 反向传播约需 $4N$ FLOPs（对激活和对权重各求一次梯度），合计 $6N$
- 全量重算在反向之前再执行一次前向，增加 $2N$，总计 $8N$，即计算量增加 33%

选择性重算的依据是不同激活的显存占用与重算代价差别很大：
- 未使用 FlashAttention 时，softmax 输出和 dropout mask 的大小为 $a s^2 b$（$a$ 为头数）
  - 随序列长度平方增长，在长序列下占激活显存的大部分
  - 而重算它们只需一次逐元素运算和一次 softmax，计算量远小于 GEMM
- 只丢弃这类激活，就能以 2–3% 的额外计算节省大部分激活显存
- FlashAttention 本身不保存注意力矩阵
  - 因此目前选择性重算主要作用于 MLP 激活和 norm

#### 2.4.2 $\sqrt{n}$ 检查点

$n$ 层网络每 $k$ 层保存一个检查点，峰值显存约为

$$M(k) \approx \frac{n}{k}\cdot a + k\cdot a$$

- 前一项为全部检查点
- 后一项为反向时重算一段所需的激活，$a$ 为单层激活
- 令导数为 0 得 $k^*=\sqrt{n}$，$M^* = 2\sqrt{n}\,a$

#### 2.4.3 Offload

- Offload 指将部分张量从 GPU 显存转移到 CPU 内存或 NVMe 存储，在需要时再传回 GPU

| 对象 | 方案 | 适用场景 |
|---|---|---|
| 优化器状态 | ZeRO-Offload，优化器在 CPU 上执行更新 | GPU 数量少、模型大、对训练吞吐要求不高 |
| 参数 | ZeRO-Infinity，存放于 NVMe | 显存与 CPU 内存均不足的极端规模 |
| 激活 | 前向后传输到 CPU，反向前预取回 GPU | PCIe 带宽足以掩盖传输时 |
| KV cache | GPU → CPU → SSD 分层存储 | 长上下文、前缀缓存（5.4 节） |

- Offload 能否被计算掩盖取决于一个不等式：传输时间 < 可重叠的计算时间
  - PCIe Gen5 x16 单向带宽约 64 GB/s，传输 10 GB 约需 160 ms
  - 若一层计算只需 20 ms，传输无法被掩盖

#### 2.4.4 显存优化手段的优先级

- 按额外开销从低到高排列
  - FlashAttention 与算子融合（无额外开销）
  - 选择性重算（开销约 2–3%）
  - 并行切分（第 3 章）
  - 全量重算（计算量 +33%）
  - offload（可能显著降低吞吐）

这一顺序的理由如下：
- FlashAttention 与算子融合
  - 在减少显存的同时还减少 HBM 访问，通常使训练更快，没有任何代价
  - 应首先采用
- 选择性重算
  - 只重算低成本的激活，开销在 2–3% 范围内
  - 可作为默认配置
- 并行切分（TP、PP、ZeRO 等）
  - 把模型状态和激活分摊到多块 GPU 上
  - 代价是通信开销和工程复杂度，其开销取决于互连带宽
  - 第 3 章给出定量分析
- 全量重算
  - 33% 的计算开销是确定的
  - 只在其他手段不足时使用
- offload
  - 代价取决于 PCIe 带宽能否被计算掩盖
  - 掩盖失败时吞吐可能下降数倍，因此放在最后

### 2.5 缓存分配器与显存碎片

#### 2.5.1 缓存分配器

- `cudaMalloc` / `cudaFree` 调用开销大，并且会引入设备同步
- PyTorch 因此使用缓存分配器（caching allocator）
  - 它向 CUDA 申请大块显存后自行管理
  - 张量释放后，其显存块归还到缓存池而不是交还 CUDA
  - 后续相同或相近尺寸的申请直接复用缓存块

| 接口 | 含义 |
|---|---|
| `torch.cuda.memory_allocated()` | 张量实际占用的显存 |
| `torch.cuda.memory_reserved()` | 分配器向 CUDA 申请的显存总量 |
| 二者之差 | 缓存池中的空闲显存与碎片 |

#### 2.5.2 案例一：空闲显存充足时的 OOM

- 报错形如 `Tried to allocate 2.00 GiB (... 9.8 GiB free; ... reserved but unallocated 9.5 GiB)`
- 显存碎片（fragmentation）指空闲显存被分割为大量不连续的小块
  - 总量充足，但没有一块满足单次申请所需的连续大小
  - 本例中空闲显存约 9.8 GB，却不存在连续的 2 GB
- 变长序列是常见诱因
  - 每个 step 的张量形状不同，缓存块的尺寸随之分散

```bash
# 用 CUDA 虚拟内存 API，让物理页映射到连续的虚拟地址
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

其他缓解手段：
- 将序列长度填充到若干固定档位（bucketing）
- 训练开始前按最大形状执行一次热身 step，使分配器预先保留足够大的块
- 设置 `max_split_size_mb`，禁止分配器切分超过该尺寸的块

#### 2.5.3 案例二：显存随训练步数缓慢增长直至 OOM

显存在数千步内持续增长，常见原因如下：

1. 计算图随 loss 累积
   - `total_loss += loss` 中的 `loss` 是带有 `grad_fn` 的张量
   - 累加结果会保留对每一步计算图的引用，使各步的中间激活都无法释放，显存随步数线性增长
   - 应改为 `total_loss += loss.item()`（转为 Python 标量）或 `loss.detach()`（切断与计算图的联系）
2. 日志列表中保存了 GPU 张量
   - 例如每步把 `logits` 或 `grad_norm` 张量追加到列表中用于统计，这些张量的引用使其显存无法回收
   - 若每步保存一个 4 MB 的张量，1 万步后就占用 40 GB
   - 应先转为 CPU 标量或在追加前调用 `.cpu()`
3. 反复新建通信组
   - 每次调用 `dist.new_group` 时，NCCL 都会为新的通信器分配缓冲区（每个通信器通常为数十到数百 MB）
   - 在训练循环或评估函数中重复创建通信组而不销毁，缓冲区会持续增加
   - 通信组应在初始化阶段一次性创建并复用
4. 形状多样导致碎片累积
   - 即使没有引用泄漏，若每步张量形状都不同，缓存池中会逐渐积累大量尺寸不一、无法被后续申请复用的块
   - `memory_reserved()` 持续上升而 `memory_allocated()` 平稳
   - 处理方法与 2.5.2 节相同

- 区分前三类（真实泄漏）与第四类（碎片）的方法是观察 `memory_allocated()`
  - 前者持续增长，后者基本不变

#### 2.5.4 诊断工具

```python
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... 执行若干训练 step ...
torch.cuda.memory._dump_snapshot("mem.pickle")   # 上传至 https://pytorch.org/memory_viz 可视化
print(torch.cuda.memory_summary())
```

#### 2.5.5 推理引擎中的预分配策略

- vLLM 在启动时为 KV cache 一次性预分配一大块显存，再以固定大小的 block 为单位管理（PagedAttention，5.3 节）
- 这与操作系统以分页机制避免内存碎片的思路一致
  - 因此推理过程中不会产生分配器层面的碎片

### 2.6 Triton 编程模型

#### 2.6.1 Triton 概述

- Triton 是 OpenAI 开源的 GPU kernel 编程语言，以 Python DSL 的形式嵌入
- 其编程模型以程序实例（program instance）为单位
  - 一次 kernel launch 启动由多个程序实例组成的网格（grid）
  - 每个实例处理一块数据，对应 CUDA 中的一个线程块（block）
- 线程映射、合并访存、SMEM 分配和软件流水线均由编译器完成
- 相比 CUDA，Triton 的开发成本低得多，性能通常可达手写 CUDA 的 80–100%
- `torch.compile` 生成的 GPU 代码大部分也是 Triton

#### 2.6.2 Triton kernel 的基本结构

一个 Triton kernel 通常由三部分组成：

1. `program_id`：获取当前程序实例的编号，确定其负责的数据块（相当于 CUDA 的 `blockIdx`）。
2. `offsets` 与 `mask`：计算块内各元素的下标，并用掩码屏蔽越界访问。
3. `tl.load` → 计算 → `tl.store`：从 HBM 加载数据，在片上计算，再写回 HBM。

#### 2.6.3 例一：向量加法

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # 1. 当前实例负责第 pid 块
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 2. 块内下标
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)         # 3. 加载、计算、写回
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x); n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

#### 2.6.4 例二：融合 residual add 与 RMSNorm

每个程序实例处理一行（一个 token 的隐藏向量）：

```python
@triton.jit
def add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr,
                       stride, H, eps, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H); mask = cols < H
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.).to(tl.float32)
    r = tl.load(res_ptr + row * stride + cols, mask=mask, other=0.).to(tl.float32)
    h = x + r                                             # residual add
    tl.store(res_out_ptr + row * stride + cols, h.to(tl.bfloat16), mask=mask)
    rms = tl.sqrt(tl.sum(h * h, axis=0) / H + eps)        # 行内归约，FP32
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    tl.store(out_ptr + row * stride + cols, (h / rms * w).to(tl.bfloat16), mask=mask)

def add_rmsnorm(x, res, w, eps=1e-6):
    T, H = x.shape
    out, res_out = torch.empty_like(x), torch.empty_like(x)
    add_rmsnorm_kernel[(T,)](x, res, w, out, res_out, x.stride(0), H, eps,
                             BLOCK_H=triton.next_power_of_2(H))
    return out, res_out
```

- `tl.sum` 的归约顺序在形状与配置相同时是固定的
  - 块大小或切分方式改变时，归约树随之改变，结果的低位可能不同
  - 4.10 和 5.15 节将讨论这一点对数值确定性的影响

#### 2.6.5 分块与编译器的职责

- 上面两个例子都是逐元素或逐行操作，数据只需加载一次
- 矩阵乘这类操作的每个输入元素要参与多次运算，分块的意义在于数据复用
  - 将 $C = AB$ 的输出按 $B_M\times B_N$ 分块
  - 每个程序实例沿 $K$ 维逐块加载 $A$ 的 $B_M\times B_K$ 子块和 $B$ 的 $B_K\times B_N$ 子块到 SMEM
  - 在片上完成 $B_M B_N B_K$ 次乘加
  - 每加载一个元素被复用 $B_N$ 或 $B_M$ 次，算术强度随块大小线性增长
- 块越大复用越充分，但块的总字节数必须小于 SMEM 和寄存器容量，否则无法驻留于片上
  - 以 BF16、$B_M=B_N=128$、$B_K=64$ 为例
  - 两个输入子块共 $2\times128\times64\times2 = 32$ KB
  - FP32 累加器为 $128\times128\times4 = 64$ KB
  - 均需放在片上存储中

从计算图到可执行 kernel，编译器（`torch.compile` 的 Inductor 后端、Triton 编译器、XLA 等）需要完成以下工作：

- 计算图分解与算子融合：将模型的计算图划分为若干子图，决定哪些相邻操作合并为一个 kernel。
  - 融合边界的选择直接决定 HBM 往返次数
- 分块大小选择：为每个 kernel 选择 $B_M$、$B_N$、$B_K$ 等参数，在片上存储容量、并行度和数据复用之间取得平衡。
  - 并行度方面：块越大，程序实例越少，SM 可能填不满
  - Triton 通过 `triton.autotune` 在若干候选配置中实测选择
- 数据布局：决定张量在内存中的维度顺序和对齐方式。
  - 使同一 warp 的访存落在连续地址上（合并访存）
  - 并满足 Tensor Core 指令对矩阵形状的要求
- 内存分配：为中间张量规划显存。
  - 复用生命周期不重叠的缓冲区，减少峰值显存
- 指令调度与通信调度
  - 在 kernel 内部安排加载与计算的流水，使下一块数据的加载与当前块的计算重叠
  - 在分布式场景下，还要安排集合通信与计算的重叠顺序

- kernel 的最终性能由硬件特性与这些编译决策共同决定
  - 同一个矩阵乘，分块参数选择不当时性能可能只有最优配置的一半

### 2.7 算例：residual add 与 RMSNorm 的 HBM 访存量

#### 2.7.1 设定

decoder 层中的一段计算，$T = 8192$ 个 token，$h = 4096$，BF16：

```text
y = RMSNorm(x + residual) * w
```

#### 2.7.2 未融合实现

| 步骤 | 读 | 写 |
|---|---|---|
| `h = x + residual` | $4Th$ | $2Th$ |
| `sq = h * h` | $2Th$ | $2Th$ |
| `ms = sq.mean(-1)` | $2Th$ | 可忽略 |
| `h * rsqrt(ms + eps)` | $2Th$ | $2Th$ |
| `* w` | $2Th$ | $2Th$ |
| 合计 | $12Th$ | $8Th$ |

- 共 $20Th \approx 671$ MB
- 至少需要 5 个 kernel

#### 2.7.3 融合实现

- 读取 $x$ 与 residual 共 $4Th$
- 写出 out 与新 residual 共 $4Th$
- 合计 $8Th \approx 268$ MB，只需 1 个 kernel

#### 2.7.4 耗时下限

- 以 H100 HBM 带宽 3.35 TB/s 的 85%（约 2.85 TB/s）计算
  - 未融合实现约 235 µs，另有 5 次 kernel launch 开销
  - 融合实现约 94 µs
- 融合实现快 2.5 倍，而浮点运算量完全相同
- 对带宽受限的操作，性能分析应统计访存字节数，而不是 FLOPs

### 2.8 补充：性能剖析方法

- 性能剖析（profiling）指在程序运行时采集各算子、kernel 与硬件单元的耗时及资源使用数据，用于定位性能瓶颈

#### 2.8.1 常用工具

| 工具 | 粒度 | 主要观测内容 |
|---|---|---|
| `torch.profiler` | 算子 | 各算子 CPU/GPU 耗时、调用栈、显存，导出 Chrome trace |
| Nsight Systems (`nsys`) | 系统时间线 | CPU 线程、CUDA API、kernel、NCCL、内存拷贝之间是否重叠，GPU 空隙 |
| Nsight Compute (`ncu`) | 单个 kernel | 带宽与算力占峰值的比例、occupancy、访存合并、roofline |
| `nvidia-smi dmon` / DCGM | 设备 | SM 活跃度、显存带宽利用率、功耗、温度、XID 错误 |

#### 2.8.2 使用示例

```python
from torch.profiler import profile, ProfilerActivity, schedule
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=1, warmup=1, active=3),
             on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace"),
             record_shapes=True, profile_memory=True, with_stack=True) as prof:
    for step in range(5):
        train_step(); prof.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o report python train.py
ncu --set full -k regex:flash -o fa_report python bench_attention.py
```

#### 2.8.3 分析流程

1. 观察整体时间线，判断 GPU 是否存在空隙
   - 用 Nsight Systems 采集若干个训练 step，查看 GPU 时间线上 kernel 之间的间隔
   - 若空隙占比明显（例如超过 10%），瓶颈不在 kernel 本身，而在 GPU 之外
     - CPU 侧 Python 开销或 kernel launch 过慢（表现为 CPU 线程满载、kernel 很短）
     - 数据加载跟不上（空隙集中出现在每个 step 开头）
     - 通信未与计算重叠（空隙对应 NCCL kernel 的执行期间）
2. 找出耗时最长的若干 kernel，判断其类型
   - 用 `torch.profiler` 按 `cuda_time_total` 排序，通常前 10 个 kernel 占总时间的大部分
   - GEMM 和注意力 kernel 属于计算密集型，逐元素、归约和数据重排 kernel 属于带宽受限型
   - 若逐元素 kernel 合计占比很高（例如超过 30%），说明融合不足
3. 对这些 kernel 做 roofline 分析
   - roofline 以算术强度和硬件峰值刻画性能上限（见 1.3）
   - 用 Nsight Compute 读出 kernel 的实际 FLOPs 和 HBM 字节数，计算算术强度，再与脊点比较，判断它应受算力还是带宽限制
   - 并计算实际性能与对应上限的比值
   - 例如一个带宽受限 kernel 只达到 HBM 峰值带宽的 40%，说明其访存模式存在问题（如未合并访存），而不是融合问题
4. 修改后重新测量
   - 每次只改动一处，并在相同的形状和步数下对比
   - 避免将测量波动误认为优化效果

#### 2.8.4 GPU-Util 与真实利用率

- `nvidia-smi` 报告的 GPU-Util 只表示采样周期内是否有 kernel 在执行
  - 一个只占用 1 个 SM 的 kernel 也会显示 100%
- 真实利用率应以 MFU（实际达到的模型 FLOPs 与硬件峰值之比）衡量
  - 或参考 DCGM 的 `SM_ACTIVE`、`PIPE_TENSOR_ACTIVE` 指标

### 2.9 小结

- 带宽受限操作的优化目标是减少 HBM 读写字节数，首选手段是算子融合。
- FlashAttention 由分块计算、online softmax 和反向重算组成，注意力的额外显存从 $O(N^2)$ 降至 $O(N)$；它减少的是 HBM 访问量而非 FLOPs，计算量仍为 $O(N^2 d)$。
- 分块使数据在片上被多次复用，块大小受 SMEM 与寄存器容量约束；融合边界、分块参数和数据布局由编译器决定。
- 显存不足时的处理顺序：融合与 FlashAttention → 选择性重算 → 并行切分 → 全量重算 → offload。
- 空闲显存充足却发生 OOM，原因通常是碎片，首先尝试 `expandable_segments:True`。
- 优化前先做 profiling；GPU-Util 不能代替 MFU。

<br>

## 3. 分布式并行：数据、张量、流水线、序列与专家并行

需要将计算分布到多块 GPU 上的两种情形：
- 模型状态超出单块 GPU 的显存容量
- 单块 GPU 的训练时间不可接受

切分方式有五种：
- 按数据（数据并行）
- 按权重矩阵（张量并行）
- 按层（流水线并行）
- 按序列（序列并行与上下文并行）
- 按专家（专家并行）

每种切分都引入通信开销，本章逐一推导各自的通信量。

### 3.1 并行的必要性：显存容量与计算时间

#### 3.1.1 显存

- 70B 模型使用 Adam 优化器和 BF16 混合精度训练时，每个参数的模型状态为 16 字节
  - BF16 参数与梯度各 2 字节
  - FP32 主参数、一阶矩、二阶矩各 4 字节
- 共 $70\times10^9\times16 = 1.12$ TB
  - 至少需要 14 块 80 GB GPU 才能容纳
  - 尚未计入激活

#### 3.1.2 计算时间

- 0.1.3 节已估算，70B 模型训练 15T token 约需 $4.4\times10^6$ GPU·h
  - 单块 GPU 需要约 500 年
  - 1 万块 GPU 约需 18 天

#### 3.1.3 五种并行方式概览

| 切分对象 | 名称 | 每块 GPU 存储的内容 | 通信操作 | 通信量级 |
|---|---|---|---|---|
| 数据 | DP / ZeRO / FSDP | 完整模型（或其 1/N 分片） | 梯度 AllReduce / 参数 AllGather | 每步 $\sim 2\Psi$ |
| 矩阵 | TP | 每层权重的 1/t | 每层激活 AllReduce | 每层 $\sim 4sbh$ ×2 |
| 层 | PP | 1/p 的层 | 相邻 stage 之间点对点传递激活 | 每 micro-batch $sbh$ |
| 序列 | SP / CP | 1/c 的 token | KV 的 AllGather / Ring P2P / All-to-All | 与序列长度成正比 |
| 专家 | EP | 1/e 的专家 | token 分发的 All-to-All | 与 top-k 和 token 数成正比 |

下面逐一说明表中各方式的基本思想，详细推导见 3.3 至 3.7 节。

数据并行（DP）让每块 GPU 处理不同的训练样本：
- 每块 GPU 持有完整模型，独立完成前向与反向
- 再通过 AllReduce 对梯度求平均，使所有副本执行相同的更新
- 它不减少单块 GPU 的模型状态，只提高吞吐
  - ZeRO 与 FSDP 在此基础上把模型状态也分片，解决显存问题
- 通信每步发生一次
  - 数据量与模型大小成正比
  - 可以与反向计算重叠

张量并行（TP）切分单个层内的矩阵乘：
- 一个线性层的权重按行或列分到 $t$ 块 GPU 上
  - 每块 GPU 计算结果的一部分，再通过 AllReduce 组合
- TP 同时减少权重和计算量
- 每层都要通信，且通信位于前向和反向的关键路径上
  - 因此对带宽和延迟最敏感

流水线并行（PP）按层切分模型：
- 连续的若干层组成一个 stage，分配到一块 GPU 上
- 样本被切分为 micro-batch 依次流过各 stage
- stage 之间只传递边界处的激活，通信量小
- 流水线的填充与排空阶段会产生空闲时间（气泡）

序列并行与上下文并行（SP / CP）沿序列维切分激活，用于长上下文训练：
- 逐 token 独立的操作切分后无需通信
- 注意力需要跨 GPU 交换 K、V

专家并行（EP）用于 MoE 模型：
- 把不同专家放在不同 GPU 上
- token 按路由结果通过 All-to-All 发送到对应专家所在的 GPU

实际系统通常组合使用多种方式（3.9 节给出一个完整配置）：
- 选择依据是模型大小、序列长度、显存容量、互连拓扑和通信量

### 3.2 集合通信原语

- 集合通信（collective communication）：指一组进程共同参与、按固定语义交换或归约数据的通信操作
- NCCL 是 NVIDIA GPU 上的集合通信库，本节的原语均由它实现

#### 3.2.1 常用原语

设共有 $p$ 块 GPU，每块 GPU 持有大小为 $M$ 字节的数据。

| 原语 | 操作前 | 操作后 | 典型用途 |
|---|---|---|---|
| Broadcast | GPU 0 持有 $x$ | 所有 GPU 持有 $x$ | 初始化权重 |
| Reduce | 各 GPU 持有 $x_i$ | GPU 0 持有 $\sum x_i$ | 较少直接使用 |
| AllReduce | 各 GPU 持有 $x_i$ | 所有 GPU 持有 $\sum x_i$ | DP 梯度同步、TP 激活求和 |
| ReduceScatter | 各 GPU 持有 $x_i$（分 $p$ 段） | GPU $j$ 持有 $\sum_i x_i[j]$ | ZeRO-2/3 梯度、SP |
| AllGather | GPU $j$ 持有 $x[j]$ | 所有 GPU 持有完整 $x$ | ZeRO-3 参数、SP |
| All-to-All | GPU $i$ 持有 $x_i[0..p-1]$ | GPU $j$ 持有 $x_0[j],\dots,x_{p-1}[j]$ | MoE、Ulysses |

各原语的语义如下：
- Broadcast：将一块 GPU 上的数据复制到所有 GPU
- Reduce：将所有 GPU 的数据逐元素归约（通常为求和）到一块 GPU
- AllReduce：在归约后将结果分发到所有 GPU
- ReduceScatter：将数据分为 $p$ 段，第 $j$ 段的归约结果只保留在 GPU $j$ 上
- AllGather：将各 GPU 持有的分段拼接为完整数据并分发到所有 GPU
- All-to-All：每块 GPU 向其他每块 GPU 发送一段不同的数据
  - 相当于对 $p\times p$ 的分段矩阵做转置

三者之间存在恒等关系：AllReduce = ReduceScatter + AllGather。
- 先做 ReduceScatter，每块 GPU 得到完整和的一段
- 再做 AllGather，把各段拼回完整结果并分发到所有 GPU
  - 最终状态与 AllReduce 相同
- ZeRO 与 Megatron SP 都利用这一分解
  - 把一次 AllReduce 拆成两半，在中间插入只需处理分片数据的计算

除集合通信外，还有点对点通信（point-to-point，P2P）：
- 一块 GPU 向另一块指定的 GPU 发送数据（`send` / `recv`），不涉及其他 GPU
- 流水线并行在相邻 stage 之间传递激活，Ring Attention 沿环传递 KV 块
  - 使用的都是 P2P 通信

#### 3.2.2 α-β 代价模型

- α-β 模型是估算点对点通信时间的线性模型：发送 $n$ 字节的时间为 $T = \alpha + n/\beta$
  - $\alpha$ 为与消息大小无关的固定延迟（µs 量级）
  - $\beta$ 为链路带宽

延迟与带宽是两个独立的参数，分别主导不同规模的消息：
- 设 $\alpha = 5$ µs、$\beta = 50$ GB/s
  - 发送 4 KB 时带宽项只有 0.08 µs，总时间几乎全部是延迟
  - 发送 1 GB 时带宽项为 20 ms，延迟可以忽略
- 两项相等时的消息大小 $n^* = \alpha\beta$，在此例中约为 250 KB
  - 小于 $n^*$ 的消息受延迟限制，优化方向是合并消息、减少通信次数
  - 大于 $n^*$ 的消息受带宽限制，优化方向是减少数据量或使用更快的链路
- DDP 将梯度分桶为约 25 MB 再通信（3.3.2 节）
  - 目的是使每次通信都处于带宽主导区间

#### 3.2.3 Ring AllReduce 推导

$p$ 块 GPU 连成逻辑环，数据切为 $p$ 段，每段 $M/p$ 字节。

1. ReduceScatter 阶段，共 $p-1$ 步
   - 每一步，每块 GPU 将一段发送给环上的下一块 GPU，并将收到的段累加到本地对应的段上
   - $p-1$ 步后，每块 GPU 持有一段的完整归约结果
2. AllGather 阶段，共 $p-1$ 步：每块 GPU 沿环转发已完成归约的段，$p-1$ 步后所有 GPU 都持有全部段。

每步每块 GPU 发送 $M/p$ 字节，共 $2(p-1)$ 步：

$$T_{\text{ring}} = 2(p-1)\alpha + \frac{2(p-1)}{p}\cdot\frac{M}{\beta}$$

- 带宽项 $\frac{2(p-1)}{p}\frac{M}{\beta}\to\frac{2M}{\beta}$，与 GPU 数量几乎无关
  - 这是 ring 算法的优点
- 延迟项随 $p$ 线性增长
  - 这是其缺点

| 原语 | Ring 带宽项 |
|---|---|
| AllReduce | $\frac{2(p-1)}{p}\frac{M}{\beta}$ |
| ReduceScatter / AllGather | $\frac{p-1}{p}\frac{M}{\beta}$ |
| All-to-All | $\frac{p-1}{p}\frac{M}{\beta}$（每块 GPU 发出 $M$ 中的 $(p-1)/p$） |

#### 3.2.4 Ring 与 Tree 算法

| 算法 | 延迟项 | 带宽项 | 适用场景 |
|---|---|---|---|
| Ring | $O(p)\alpha$ | 最优 $\approx 2M/\beta$ | 大消息 |
| Tree（double binary tree） | $O(\log p)\alpha$ | 接近最优 | 小消息、超大规模 |

- NCCL 根据消息大小和拓扑自动选择算法（可用 `NCCL_ALGO` 强制指定）
- 传输协议也有 LL、LL128、Simple 三种
- 自动选择使同一 AllReduce 在不同规模下的求和顺序不同
  - 4.10 节讨论它对数值确定性的影响

#### 3.2.5 busbw 与 algbw

`nccl-tests` 报告两个带宽指标：

- algbw（算法带宽）= $M / T$，即数据量除以耗时；
- busbw（总线带宽）= algbw × $\frac{2(p-1)}{p}$（AllReduce 情形），按 ring 算法中每条链路实际传输的数据量折算，可直接与硬件链路带宽比较。

典型数值：
- H100 节点内 8 块 GPU 的 AllReduce busbw 一般为 350–450 GB/s
- 跨节点使用 IB 400G、每块 GPU 配一块网卡时约为 40–48 GB/s

#### 3.2.6 PyTorch 接口示例

```python
import torch, torch.distributed as dist
# torchrun --nproc_per_node=8 collectives.py
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())

x = torch.full((4,), float(rank), device="cuda")
dist.all_reduce(x, op=dist.ReduceOp.SUM)                   # 所有 rank 得到 0+1+...+7
shards = torch.empty(world * 4, device="cuda")
dist.all_gather_into_tensor(shards, torch.full((4,), float(rank), device="cuda"))
out = torch.empty(4, device="cuda")
dist.reduce_scatter_tensor(out, torch.ones(world * 4, device="cuda"))   # 每段 = world
a2a_out = torch.empty(world, device="cuda")
dist.all_to_all_single(a2a_out, torch.arange(world, device="cuda", dtype=torch.float) + rank * 100)
print(rank, x[0].item(), out[0].item(), a2a_out.tolist())
```

### 3.3 数据并行与 ZeRO

#### 3.3.1 数据并行（DP / DDP）

数据并行指每块 GPU 持有完整的模型副本，各自处理不同的数据分片：
- 反向传播后对梯度做 AllReduce 求平均
- 再各自执行相同的参数更新

$$g = \frac{1}{N_d}\sum_{i=1}^{N_d} g_i$$

- 每步通信量为 $2\Psi$ 字节量级（$\Psi$ 为梯度字节数）

#### 3.3.2 DDP 的工程实现

| 机制 | 实现 |
|---|---|
| 梯度分桶（bucket） | 按反向传播顺序将梯度装入约 25 MB 的桶，桶满即发起 AllReduce |
| 计算与通信重叠 | 靠后的层梯度先计算完成、先发起通信，与靠前层的反向计算重叠 |
| 梯度累积 | 前几个 micro-step 用 `no_sync()` 跳过通信，只在最后一个 micro-step 同步梯度 |
| 全局 batch | global batch = micro-batch × 累积步数 × DP 规模 |

- micro-batch 是单次前向与反向传播实际处理的样本子集
- 若干 micro-batch 的梯度累积后构成一次参数更新
- 每处理一个 micro-batch 称为一个 micro-step

#### 3.3.3 ZeRO 的三个阶段

- DP 的缺点是每块 GPU 都保存完整的模型状态（每参数 16 字节）
- ZeRO（Zero Redundancy Optimizer）将模型状态分片到 $N_d$ 块 GPU 上，以消除这种冗余

| 阶段 | 分片对象 | 每块 GPU 显存 | 通信量（相对 DP） |
|---|---|---|---|
| DP | 不分片 | $16\Psi$ | 1× |
| ZeRO-1 | 优化器状态 | $4\Psi + \frac{12\Psi}{N_d}$ | 1× |
| ZeRO-2 | + 梯度 | $2\Psi + \frac{14\Psi}{N_d}$ | 1× |
| ZeRO-3 / FSDP | + 参数 | $\frac{16\Psi}{N_d}$ | 1.5× |

- 此处 $\Psi$ 表示参数个数
- 以 7.5B 模型、$N_d=64$ 为例
  - DP 为 120 GB
  - ZeRO-1 为 31.4 GB
  - ZeRO-2 为 16.6 GB
  - ZeRO-3 为 1.9 GB

#### 3.3.4 通信量推导

以下按 ring 算法、忽略 $(p-1)/p$ 因子，以每块 GPU 的发送量计（单位为一份完整梯度或参数的大小 $\Psi$）。

DP：
- 梯度 AllReduce 等价于一次 ReduceScatter 加一次 AllGather
  - 每次各发送约 $\Psi$，合计约 $2\Psi$

ZeRO-1 和 ZeRO-2 把同一过程拆开使用：
- 反向结束后对梯度做 ReduceScatter（$\Psi$）
  - 每块 GPU 只得到自己负责的那 $1/N_d$ 段参数的完整梯度
  - 这正好够它更新自己持有的那段优化器状态和参数
- 更新完成后，对参数做 AllGather（$\Psi$）
  - 使每块 GPU 重新获得完整参数，供下一步前向使用
- 合计仍为 $2\Psi$，与 DP 相同
  - ZeRO-1/2 只是把 DP 本来就要做的 AllReduce 拆成两半，在中间插入分片更新
  - 因此显存大幅减少而通信量不变

ZeRO-3 连参数也只保留分片：
- 前向计算每一层之前，必须先 AllGather 该层的完整参数（合计 $\Psi$）
  - 用完即释放以节省显存
- 反向传播计算每一层的梯度时需要再次使用完整参数
  - 它们已在前向后释放，因此要再 AllGather 一次（$\Psi$）
- 最后对梯度做 ReduceScatter（$\Psi$）
- 合计 $3\Psi$，是 DP 的 1.5 倍
  - 多出的 $\Psi$ 是以通信换显存的代价

#### 3.3.5 FSDP 的执行流程

- FSDP（Fully Sharded Data Parallel）是 PyTorch 对 ZeRO-3 的实现
- 以模块为单元对参数、梯度和优化器状态分片

```text
前向 layer i:  AllGather(参数分片) → 计算 → 释放完整参数
               （同时预取 layer i+1 的参数）
反向 layer i:  AllGather(参数分片) → 计算梯度 → ReduceScatter(梯度) → 释放
优化器:        每块 GPU 只更新自己的参数分片
```

#### 3.3.6 FSDP2 代码示例

```python
import torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

dist.init_process_group("nccl"); torch.cuda.set_device(dist.get_rank() % 8)
model = build_transformer().cuda()
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.layers:            # 每个 Transformer block 作为一个 FSDP 单元
    fully_shard(block, mp_policy=mp)
fully_shard(model, mp_policy=mp)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)   # 参数已是 DTensor 分片

for x, y in loader:
    loss = model(x, y)
    loss.backward()                    # 自动执行 AllGather / ReduceScatter
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
```

#### 3.3.7 HSDP

- 万卡规模下，ZeRO-3 跨全部节点的 AllGather 开销过高
- HSDP（Hybrid Sharded DP）的做法：
  - 在节点内（或一小组节点内）做 FSDP 分片
  - 在组间做普通 DP 复制
  - 使大部分通信经由 NVLink 完成

### 3.4 张量并行（TP）

#### 3.4.1 列切分与行切分

张量并行指将单个层的权重矩阵切分到 $t$ 块 GPU 上，每块 GPU 计算矩阵乘的一部分。对线性层 $Y = XA$：

- 列切分（column parallel）：$A = [A_1, A_2]$，$Y_i = XA_i$。每块 GPU 得到输出的一部分列，无需通信。
- 行切分（row parallel）：$A = \begin{bmatrix}A_1\\A_2\end{bmatrix}$，$X = [X_1, X_2]$，$Y = X_1A_1 + X_2A_2$。每块 GPU 得到部分和，需要 AllReduce 求和。

#### 3.4.2 Megatron-LM 的 MLP 切分

$$Z = \text{GeLU}(XA)\,B$$

- $A$ 列切分、$B$ 行切分
- GeLU 是逐元素函数，可以直接作用于列切分后的 $XA_i$，中间无需通信：

$$Z = \sum_i \text{GeLU}(XA_i)\,B_i$$

- 整个 MLP 只在最后做一次 AllReduce

#### 3.4.3 注意力层的切分

按注意力头切分：
- $W_Q, W_K, W_V$ 列切分
  - 每块 GPU 负责 $a/t$ 个头并在本地完成注意力计算
- $W_O$ 行切分，最后做 AllReduce
- 使用 GQA 时，KV 头数也需能被 $t$ 整除
  - 否则需要复制 KV 头

#### 3.4.4 共轭算子 f 与 g

Megatron 用一对共轭算子描述 TP 的通信：

| 算子 | 前向 | 反向 |
|---|---|---|
| $f$（进入 TP 区域） | identity | AllReduce |
| $g$（离开 TP 区域） | AllReduce | identity |

- 每个 Transformer 层有 attention 和 MLP 两个 TP 区域
  - 因此前向共 2 次 AllReduce，反向共 2 次
- 每次数据量为 $sbh$ 个元素
  - $s$ 为序列长度
  - $b$ 为 micro-batch 大小
  - $h$ 为隐藏维度

#### 3.4.5 代码示例

```python
import torch, torch.nn as nn, torch.distributed as dist

class _AllReduceFwd(torch.autograd.Function):          # g：前向 AllReduce，反向 identity
    @staticmethod
    def forward(ctx, x): dist.all_reduce(x); return x
    @staticmethod
    def backward(ctx, g): return g

class _AllReduceBwd(torch.autograd.Function):          # f：前向 identity，反向 AllReduce
    @staticmethod
    def forward(ctx, x): return x
    @staticmethod
    def backward(ctx, g): g = g.clone(); dist.all_reduce(g); return g

class ColumnParallelLinear(nn.Module):
    def __init__(self, d_in, d_out, tp):
        super().__init__(); self.w = nn.Parameter(torch.randn(d_out // tp, d_in) * 0.02)
    def forward(self, x):
        return _AllReduceBwd.apply(x) @ self.w.t()         # 输出是列分片

class RowParallelLinear(nn.Module):
    def __init__(self, d_in, d_out, tp):
        super().__init__(); self.w = nn.Parameter(torch.randn(d_out, d_in // tp) * 0.02)
    def forward(self, x_shard):
        return _AllReduceFwd.apply(x_shard @ self.w.t())   # 部分和 → AllReduce

class TPMLP(nn.Module):
    def __init__(self, h, tp):
        super().__init__()
        self.up = ColumnParallelLinear(h, 4 * h, tp); self.down = RowParallelLinear(4 * h, h, tp)
    def forward(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x)))
```

#### 3.4.6 Sequence Parallel（Megatron SP）

- TP 区域之外的 LayerNorm、dropout 和残差连接在每块 GPU 上都处理完整的 $sbh$ 激活
  - 同一份激活被重复存储 $t$ 份
- SP 将这些操作沿序列维切分，每块 GPU 只存储 $s/t$ 部分
- 通信相应改为：

$$\text{AllReduce} \;\to\; \text{ReduceScatter（离开 TP 区）} + \text{AllGather（进入 TP 区）}$$

- 通信总量不变（AllReduce 恰好等于二者之和）
- 激活显存从 $sbh(10 + 24/t)$ 降至 $34sbh/t$
- 因此启用 TP 时通常同时启用 SP

#### 3.4.7 TP 的通信开销

- 每层前向与反向共 4 次 AllReduce，每次 $2sbh$ 字节（BF16）
- 按 ring 公式，每层通信时间约为

$$T_{\text{TP}}^{\text{layer}} \approx 4\times\frac{2(t-1)}{t}\cdot\frac{2sbh}{\beta}$$

- 每层计算量约为 $24sbh^2\times 3$ FLOPs
  - 前向 $24sbh^2$，含 attention 投影和 MLP
  - 训练时乘以 3
  - 每块 GPU 承担 $1/t$
- 设单块 GPU 可达算力为 $P$，通信与计算时间之比为

$$\frac{T_{\text{comm}}}{T_{\text{comp}}} \approx \frac{16sbh/\beta}{72sbh^2/(tP)} = \frac{2tP}{9h\beta}$$

- H100 可达算力 $P\approx 800$ TFLOPS，$h=8192$
  - 使用 NVLink（$\beta=450$ GB/s）、$t=8$ 时比值约为 0.39，通信可以与计算重叠
  - 使用跨节点 IB（$\beta=50$ GB/s）时约为 3.5，通信时间超过计算时间
- 因此 TP 只部署在节点内（或 NVL72 这类超节点内）

### 3.5 流水线并行（PP）

#### 3.5.1 基本原理

- 流水线并行将 $L$ 层划分为 $p$ 个 stage，每块 GPU（或每组 GPU）负责一个 stage
- stage 之间只传递前向激活和反向的激活梯度
  - 采用点对点通信，每个 micro-batch 传输 $sbh$ 个元素
- 通信量小，适合跨节点部署

#### 3.5.2 流水线气泡

- 一个 batch 被切分为 $m$ 个 micro-batch 依次送入流水线
- 在流水线的填充阶段和排空阶段，部分 stage 没有可执行的计算而处于空闲
  - 这些空闲时间称为气泡（bubble）：

```text
GPipe / 1F1B，p = 4，m = 8（F = 前向，B = 反向）
stage0: F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 B6 B7 B8
stage1:    F1 F2 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 B7 B8 ...
stage2:       F1 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 F8 B7 B8 ...
stage3:          F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8
        └─┬─┘ 填充阶段的空闲                         排空阶段的空闲
```

- 设每个 micro-batch 的前向加反向耗时为 $t_{mb}$
  - 理想耗时为 $m\,t_{mb}$
  - 实际耗时为 $(m+p-1)\,t_{mb}$
- 气泡占比为

$$\text{bubble} = \frac{p-1}{m+p-1}$$

- 相对理想耗时的比例为 $\frac{p-1}{m}$
  - $p=8$、$m=32$ 时约为 18%
- $m$ 越大气泡越小
  - 但 global batch 存在上限（过大会损害收敛）

#### 3.5.3 调度策略的演进

- 1F1B（one-forward-one-backward）调度在流水线进入稳态后，令每个 stage 交替执行一个 micro-batch 的前向和另一个 micro-batch 的反向
  - 使已完成反向的激活及时释放
  - 从而限制同时驻留的激活数量

| 调度 | 思路 | 气泡（相对理想） | 激活显存 |
|---|---|---|---|
| GPipe | 先执行全部前向，再执行全部反向 | $\frac{p-1}{m}$ | 保存 $m$ 个 micro-batch 的激活 |
| 1F1B（PipeDream-Flush） | 稳态阶段每个 stage 交替执行一次前向和一次反向 | $\frac{p-1}{m}$ | 最多保存 $p$ 个 |
| Interleaved 1F1B（Megatron） | 每块 GPU 负责 $v$ 段不连续的层（虚拟 stage） | $\frac{p-1}{vm}$ | 略增，通信 ×$v$ |
| Zero Bubble（2023） | 反向拆分为激活梯度计算 B 和权重梯度计算 W，用 W 填充气泡 | 接近 0 | 增加 |
| DualPipe（DeepSeek-V3） | 双向流水，前向与反向块内的计算和通信重叠 | 进一步降低 | 保存两份参数 |

#### 3.5.4 PP 的其他约束

负载均衡：
- 流水线的吞吐由最慢的 stage 决定，各 stage 的耗时需要尽量相等
- 首尾两个 stage 除 Transformer 层外，还分别包含 embedding 层和 LM head
  - LM head 是一个 $h\times V$ 的矩阵乘（$V$ 为词表大小）
  - $h=8192$、$V=128\text{K}$ 时其参数量约 10 亿，计算量相当于 2 到 3 个 Transformer 层
  - 输出的 logits 张量和 softmax 交叉熵也占用大量显存
- 因此首尾 stage 通常少分配一到两层

反向依赖：
- stage $i$ 计算某个 micro-batch 的反向时，需要 stage $i+1$ 先传回该 micro-batch 的激活梯度
- 依赖链贯穿整条流水线
  - 任意一个 stage 变慢（例如某块 GPU 温度过高而降频、某条链路拥塞），其上游所有 stage 都要等待
  - 整个流水线的速度随之降到最慢 stage 的水平

与 TP/DP 的耦合：
- 三者之间满足 global batch = micro-batch 大小 × $m$ × DP 规模
- global batch 由收敛性决定，通常固定
- 为了压低气泡需要增大 $m$
  - 这就要求减小 DP 规模或 micro-batch 大小
- DP 规模又由 GPU 总数除以 TP 与 PP 的乘积决定
- 因此调整任一维度都会影响其他维度
  - 3.9 节给出一个完整的求解过程

### 3.6 序列并行与上下文并行（SP/CP）

#### 3.6.1 长序列的激活显存

- 128K 上下文下，即使使用 FlashAttention，单层激活 $34sbh$ 也达到 $34\times131072\times8192\approx 36$ GB（单层、未切分）
- TP + SP 最多只能切分 $t\le8$ 份，仍然不够
  - 需要沿序列维进一步切分

#### 3.6.2 注意力的跨 token 依赖

- MLP 与 norm 对每个 token 独立计算，沿序列切分不需要任何通信
- 注意力则不同：每个 query 需要与全部 key 计算注意力分数

#### 3.6.3 Ring Attention（CP）

- 上下文并行（Context Parallelism，CP）沿序列维切分全部激活，Ring Attention 是其主要实现方式
- $c$ 块 GPU 各持有 $s/c$ 个 token 的 Q、K、V
- KV 块沿环传递 $c-1$ 步
  - 每收到一块就用 online softmax（2.3.3 节）将其贡献累加到本地输出：

```text
for step in range(c):
    O, m, l = flash_attn_block(Q_local, K_cur, V_cur, O, m, l)   # 本地计算
    K_cur, V_cur = ring_send_recv(K_cur, V_cur)                   # 与计算重叠
```

- 每块 GPU 每步发送 $2\cdot\frac{s}{c}\cdot h_{kv}$ 个元素（$h_{kv}$ 为全部 KV 头的总维度）
- 通信能被计算掩盖的条件是块计算时间不小于块传输时间：

$$\frac{4(s/c)^2 h}{P} \ge \frac{2\cdot 2 (s/c) h_{kv}}{\beta} \quad\Rightarrow\quad \frac{s}{c} \ge \frac{P\,h_{kv}}{\beta\, h}$$

- 即每块 GPU 上的序列块必须足够长
- causal mask 下各 GPU 的工作量不均衡（靠后的 token 需要关注的 key 更多）
  - 常用 zigzag / striped 切分：将序列切为 $2c$ 段，GPU $i$ 持有第 $i$ 段和第 $2c-1-i$ 段

#### 3.6.4 DeepSpeed-Ulysses

Ulysses 采用另一种方式：
- 在注意力计算前做一次 All-to-All，将按序列切分的布局转换为按头切分
  - 每块 GPU 持有完整序列的 $a/c$ 个头并在本地计算完整注意力
- 之后再做一次 All-to-All，恢复按序列切分的布局

| | Ring Attention | Ulysses |
|---|---|---|
| 通信 | P2P 环，$c-1$ 轮 | 2 次 All-to-All（Q/K/V 和 O） |
| 并行度上限 | 不受头数限制 | $c \le$ 头数（GQA 时受 KV 头数限制） |
| 通信与计算重叠 | 可以 | 较难 |
| 组合 | 两者常组合使用（USP / 2D 混合） | |

### 3.7 专家并行（EP）

#### 3.7.1 MoE 前向

$$y = \sum_{i\in \text{TopK}(g(x))} g_i(x)\,E_i(x)$$

- 门控网络（router）$g$ 为每个 token 计算各专家的得分，并选出得分最高的 $k$ 个专家
- $g_i(x)$ 为专家 $E_i$ 对应的门控权重

MoE（Mixture of Experts）层的结构：
- 用 $E$ 个结构相同、参数独立的 FFN（称为专家）替代 dense 模型中的单个 FFN
- 每个 token 只经过其中 $k$ 个
  - 因此模型的总参数量可以很大，而每个 token 激活的参数量较小
- 以 DeepSeek-V3 为例，总参数 671B，每个 token 激活约 37B
  - 单 token 的计算量接近一个 37B 的 dense 模型

稀疏激活降低的是单 token 的 FLOPs，并不意味着端到端成本按同样比例下降：
- 全部 671B 参数仍需存放在显存中，显存需求按总参数计算
- token 需要通过 All-to-All 发送到专家所在的 GPU
  - 引入 dense 模型中不存在的通信
- 各专家接收的 token 数不均衡，导致部分 GPU 空等
- 这三方面构成 MoE 系统设计的主要问题，3.7.3 至 3.7.5 节分别讨论

#### 3.7.2 EP 的数据流

专家并行将 $E$ 个专家分布到多块 GPU 上，每块 GPU 只持有其中一部分专家：

```text
token（按数据分布在各 GPU）──Router──> 决定每个 token 发往哪些专家
   │ All-to-All dispatch：将 token 发送到专家所在的 GPU
   ▼
各 GPU 用本地专家计算（grouped GEMM）
   │ All-to-All combine：将结果送回 token 原来所在的 GPU，按门控权重加权求和
   ▼
输出
```

- grouped GEMM 指在一次 kernel launch 中完成多个形状不同的矩阵乘
  - 此处每个本地专家对应一个矩阵乘

#### 3.7.3 通信量

- 每个 token 发出 $k$ 份 $h$ 维激活，dispatch 和 combine 各一次
- 每块 GPU 有 $T$ 个 token 时，每层单向通信量约为 $2\cdot k\cdot T\cdot h$ 字节（BF16）
  - 前向加反向翻倍
- 与 TP 不同，该通信量与专家总数无关，只取决于 top-k

#### 3.7.4 负载均衡

负载不均的来源：
- router 的选择由数据决定，不受系统约束
- 训练过程中，若某个专家的输出稍好，router 会更多地选择它
  - 它因此得到更多训练而变得更好，形成正反馈
  - 最终少数专家接收大部分 token
- 在 EP 中，专家所在的 GPU 需要处理的 token 数正比于其被选中的次数
- All-to-All 和后续计算是同步的，一层的耗时由负载最重的 GPU 决定
  - 若最热的专家收到平均值 2 倍的 token，其余 GPU 约有一半时间空等

容量因子（capacity factor）：
- 容量因子 $C$ 为每个专家设定接收上限
  - 平均每个专家应收到 $\frac{kT}{E}$ 个 token，上限设为 $C\cdot\frac{kT}{E}$
- 超出上限的 token 被丢弃（drop）
  - 不经过该专家，只通过残差连接传到下一层
- $C$ 常取 1.0–1.25
- 容量上限使每块 GPU 的缓冲区大小和计算量有确定的上界
  - 便于预分配显存和对齐 grouped GEMM 的形状
- 代价是被丢弃的 token 损失了一次专家计算
  - $C$ 过小会影响模型质量
  - $C$ 过大则失去约束作用、浪费填充计算

辅助负载均衡损失：
- 在训练目标中加入 $\mathcal{L}_{\text{aux}} = \alpha E\sum_i f_i P_i$
  - $f_i$ 为分配到专家 $i$ 的 token 比例
  - $P_i$ 为专家 $i$ 的平均门控概率
- 在 $\sum_i f_i$ 和 $\sum_i P_i$ 固定的约束下，均匀分布时该乘积之和最小
  - 因此最小化该损失会推动 router 均匀分配
- 系数 $\alpha$ 通常取 $10^{-2}$ 量级
  - 过大会干扰主任务的优化
  - 过小则无法抑制正反馈

无辅助损失的均衡：
- DeepSeek-V3 为每个专家引入偏置 $b_i$
  - 它只加在选择专家时使用的得分上，不参与输出的加权求和
- 每个训练 step 后统计各专家的负载
  - 过载的专家调低 $b_i$，使其更难被选中
  - 负载不足的专家调高 $b_i$
- 这种方法通过调整路由实现均衡，不向主损失加入额外梯度
  - 避免了辅助损失对模型质量的干扰

#### 3.7.5 通信优化

| 方法 | 实现 |
|---|---|
| 节点受限路由 | 每个 token 最多发往 $M$ 个节点（DeepSeek-V3 为 4），先经 IB 传至目标节点，再经 NVLink 分发 |
| 专用通信 kernel | DeepEP：高吞吐 kernel（训练、prefill）与低延迟 kernel（decode，纯 RDMA） |
| 与计算重叠 | 两个 micro-batch 交错执行，一个计算 attention 时另一个执行 All-to-All |
| 共享专家 | 每个 token 都经过的共享专家无需通信，其计算可与 dispatch 重叠 |

节点受限路由针对跨节点带宽不足的问题：
- 若不加限制，一个 token 选中的 8 个专家可能分布在 8 个不同节点上
  - 需要发送 8 份数据经过 IB
- 限定最多发往 4 个节点后，每个 token 跨节点只发送 4 份
  - 到达目标节点后再经带宽高约 9 倍的 NVLink 分发给节点内的多个专家
  - 跨节点流量减半

DeepEP 等专用通信库针对 MoE 的通信模式优化：
- 通用 All-to-All 假设各 GPU 之间的数据量已知且规则
  - 而 MoE 的发送量由路由结果动态决定
- 高吞吐 kernel 用于训练和 prefill 这类 token 数多的场景
  - 充分利用 NVLink 与 IB 带宽
- 低延迟 kernel 用于 decode
  - 此时每步 token 数少，通信受延迟主导
  - 直接通过 RDMA 读写远端显存以减少同步次数

计算与通信重叠的做法是把一个 batch 拆成两个 micro-batch 交替执行：
- micro-batch A 执行 attention 计算时，micro-batch B 进行 All-to-All
- 然后两者互换
- 只要 attention 计算时间不短于 All-to-All 时间，通信就被完全掩盖

### 3.8 互连带宽模型与并行维度放置

#### 3.8.1 互连带宽对比

| 链路 | 单向带宽（每 GPU） | 相对 HBM |
|---|---|---|
| HBM3（H100） | 3.35 TB/s | 1 |
| NVLink 4（H100，经 NVSwitch） | 450 GB/s | 1/7 |
| NVLink 5（B200 / GB200） | 900 GB/s | — |
| PCIe Gen5 x16 | ~64 GB/s | 1/50 |
| IB NDR 400G（每块 GPU 一块网卡） | 50 GB/s | 1/67 |
| IB XDR 800G / CX-8 | 100 GB/s | — |

- H100 节点内与跨节点的带宽相差约 9 倍

#### 3.8.2 各并行方式的通信需求

| 并行方式 | 每步通信 | 是否位于关键路径 | 部署位置 |
|---|---|---|---|
| TP | 每层 4 次 $2sbh$ | 是，频繁且同步执行 | 节点内 / 超节点内 |
| CP | 每层 $c-1$ 轮 KV P2P | 可与计算重叠 | 优先节点内 |
| EP | 每层 2–4 次 All-to-All | 部分可重叠 | 节点内 + 节点受限路由 |
| PP | 每 micro-batch $sbh$ P2P | 可重叠，通信量小 | 跨节点 |
| DP / ZeRO | 每步 $2\Psi$–$3\Psi$ | 大部分可重叠 | 跨节点，最外层 |

表中两个属性决定了各并行方式对链路的要求：通信频率和能否与计算重叠。
- TP 每层都要做 AllReduce，并且下一步计算必须等它完成
  - 通信时间直接计入每层耗时
  - 3.4.7 节已算出，跨节点时 TP 的通信时间是计算时间的 3.5 倍
- CP 和 EP 每层也有通信
  - 可以通过环形传递或 micro-batch 交错与计算重叠
  - 对带宽的要求略低于 TP
- PP 只在 stage 边界传递一份激活，数据量为 $sbh$
  - 比 TP 每层的通信量小一个数量级，跨节点也不会成为瓶颈
- DP 每步只同步一次
  - 可以在整个反向传播期间分桶进行，跨节点带宽足以掩盖

#### 3.8.3 放置原则

- 通信最频繁、同步性最强的并行维度部署在带宽最高的链路上
  - 由内到外依次为 TP → CP/EP → PP → DP

在框架中，这一原则体现为进程组的排列顺序：
- 以 8 卡节点、TP=8、PP=4、DP=32 为例
  - rank 编号按 TP 维度变化最快的方式排列，使 rank 0–7（同一节点）组成一个 TP 组
  - PP 组的成员相隔 8 个 rank，位于不同节点
  - DP 组的成员相隔 32 个 rank
- 若顺序设置错误，TP 组可能跨越节点
  - 导致 3.10 节所述的性能下降

#### 3.8.4 Scale-up 与 scale-out

GPU 之间的互连分为两个层级。

Scale-up：
- 指在一个高带宽域内增加互连紧密的 GPU 数量
- 域内的 GPU 通过 NVLink 与 NVSwitch 直接相连
  - 可以像访问本地显存一样读写其他 GPU 的显存
  - 每块 GPU 的带宽为数百 GB/s
- 传统的 8 卡服务器是一个 scale-up 域
  - GB200 NVL72 将其扩大到 72 块 GPU

Scale-out：
- 指通过数据中心网络（InfiniBand 或 RoCE 以太网）连接更多服务器来扩大规模
- 每块 GPU 的带宽通常为 50–100 GB/s
- 延迟更高，并且受网络拓扑和拥塞影响

两个层级的带宽相差近一个数量级：
- 这正是 3.8.3 节放置原则的物理基础
  - scale-up 域的大小决定了 TP（以及 EP、CP）能扩展到多大
  - scale-out 网络承载 PP 和 DP
- NVL72 这类超节点的意义在于扩大 scale-up 域
  - 使 TP=16 或 EP=64 这类配置无需经过 IB 即可完成

实际的通信性能不仅取决于链路的标称带宽，还取决于：
- 拓扑结构（如胖树网络的收敛比）
- 链路利用率（多个任务共享网络时的拥塞）
- 通信与计算重叠的程度

- 同一个 AllReduce 在空闲网络上与繁忙网络上的耗时可能相差数倍
- 因此规划并行配置时应以 `nccl-tests` 实测的 busbw 为准，而不是标称值

### 3.9 并行配置示例：70B 模型与 1024 块 GPU

#### 3.9.1 设定

- Llama 架构的 70B 模型：$L=80$，$h=8192$，$a=64$，GQA 8 个 KV 头
- 1024 块 H100（128 节点），序列长度 8192
- global batch 约 4M token（512 条序列）

#### 3.9.2 第一步：确定 TP

- TP 的通信只适合由节点内 NVLink 承担，因此 TP 取节点内的 GPU 数 8
- 每块 GPU 持有 140 GB/8 = 17.5 GB 的 BF16 权重

#### 3.9.3 第二步：确定 PP

- 模型状态 $1.12\text{ TB}/8 = 140$ GB，仍超过 80 GB
- 取 PP=4，每个 stage 20 层
  - 每块 GPU 的模型状态为 35 GB

#### 3.9.4 第三步：确定 DP

- $\text{DP} = 1024/(8\times4) = 32$
- 启用 ZeRO-1，将优化器状态再分为 32 份
  - 每块 GPU 的参数量为 $70\times10^9/(8\times4)\approx2.2\times10^9$
  - 模型状态为 $2.2\times10^9\times(2+2+12/32)\approx 9.6$ GB

#### 3.9.5 第四步：估算激活显存

- micro-batch=1，$s=8192$，TP=8 并启用 SP，每层激活 $34sbh/8 \approx 0.29$ GB
- 每个 stage 20 层，1F1B 调度下第一个 stage 最多同时保存 $p=4$ 个 micro-batch 的激活
  - $0.29\times20\times4\approx 23$ GB
- 加上模型状态 9.6 GB 和通信缓冲区，总量在 80 GB 以内
  - 无需启用重算

#### 3.9.6 第五步：估算气泡

- 每个 DP rank 每步处理 $512/32 = 16$ 条序列，micro-batch=1 时 $m=16$
- 气泡比例 $\frac{p-1}{m} = 3/16\approx 19\%$，偏高
- 改用 interleaved 调度、$v=5$（每块 GPU 负责 5 段，每段 4 层）后为 $\frac{3}{80}\approx 4\%$

#### 3.9.7 最终配置

| 维度 | 取值 | 依据 |
|---|---|---|
| TP | 8 | TP 通信限于节点内 NVLink |
| PP | 4，interleaved $v=5$ | 容纳模型状态，气泡约 4% |
| DP | 32，ZeRO-1 | 优化器状态再分为 32 份 |
| SP | 启用 | 激活显存降为 1/8 |
| 重算 | 不启用（序列更长时启用选择性重算） | 每块 GPU 约 35–45 GB |
| micro-batch | 1 | — |

- 预期 MFU 为 40–50%。

### 3.10 案例分析：跨节点张量并行导致的性能下降

#### 3.10.1 现象

- 某团队为容纳更大的模型，将 TP 从 8 调整为 16，MFU 从 45% 降至 18%
- Nsight Systems 显示每层都有持续时间很长的 NCCL AllReduce
  - GPU 大部分时间处于等待状态

#### 3.10.2 原因分析

- TP=16 跨越两台机器，每层 4 次 AllReduce 中有一部分经过 IB
- 按 3.4.7 节的比值公式
  - 节点内 $\beta=450$ GB/s 时通信与计算之比约为 0.4
  - IB $\beta=50$ GB/s 时约为 7
- ring AllReduce 的速度由环上最慢的链路决定
  - 环一旦跨越节点，整体速度即降至 IB 水平

#### 3.10.3 解决方案

- 将 TP 恢复为 8，新增的显存需求由 PP=2 承担
  - PP 通信量小，跨节点部署影响不大
- MFU 回升至 43%

#### 3.10.4 经验总结

第一，TP 不应跨越 scale-up 域（节点或超节点）：
- TP 的通信每层发生、位于关键路径上，其带宽需求只有 NVLink 能够满足
- 显存不足时，应优先增加 PP 或启用 ZeRO 分片
  - 这两者的通信量小或可重叠，适合跨节点

第二，修改并行配置前先做定量估算：
- 用 3.4.7 节的公式和 3.8 节的带宽表计算通信与计算时间之比
  - 比值明显大于 1 的配置无需实测即可排除
- 本例中跨节点 TP 的比值约为 7
  - 事前计算即可预见 MFU 的大幅下降

第三，用 `nccl-tests` 实测 busbw，确认链路工作正常：
- 估算使用的是标称带宽
  - 实际带宽可能因网卡故障、链路降速或拓扑配置错误而偏低
- 例如某节点 IB 的 busbw 只有 20 GB/s 而非预期的 45 GB/s
  - 所有涉及该节点的通信都会变慢
  - 仅凭估算无法发现

### 3.11 小结

- AllReduce = ReduceScatter + AllGather；ring 算法的带宽项约为 $2M/\beta$，与 GPU 数量无关。
- ZeRO-1/2 不增加通信量，ZeRO-3 增加 50%。
- TP 每层 4 次 AllReduce，只能部署在节点内；SP 不增加通信量即可减少激活显存。
- PP 气泡比例为 $\frac{p-1}{m}$，interleaved 调度将其降至 $\frac{p-1}{vm}$。
- 长序列使用 CP（Ring Attention / Ulysses），MoE 使用 EP（All-to-All 与负载均衡）。
- MoE 的稀疏激活降低单 token 计算量，但显存需求按总参数计算，并引入 All-to-All 通信与负载不均问题。
- 并行维度由内到外的放置顺序：TP → CP/EP → PP → DP；scale-up 域（NVLink）承载 TP、CP、EP，scale-out 网络承载 PP、DP。

<br>

## 4. 大规模训练系统：容错、效率与可复现性

- 第 3 章讨论了模型如何在数据、张量、流水线等维度上切分到多块 GPU
- 本章讨论切分方案确定之后，训练作业如何在数千至上万块 GPU 上连续运行数周乃至数月
- 在这一规模下，硬件与软件故障会频繁发生，因此训练系统需要同时满足三项要求：
  - 容错：故障发生时损失的计算时间尽可能少
  - 效率：GPU 的算力被尽可能充分地利用
  - 可复现性：出现异常现象时能够复现并定位原因

### 4.1 主流训练框架：Megatron、DeepSpeed、FSDP 与 Transformer Engine

#### 4.1.1 框架定位与适用场景

下表中的并行维度沿用第 3 章的记号：
- TP（张量并行）：把单层的矩阵按行或列切到多块 GPU
- SP（序列并行）：在 TP 组内沿序列维切分 norm、dropout 等逐元素算子的激活
- PP（流水线并行）：把不同层分配到不同 GPU
- CP（上下文并行）：沿序列维切分注意力计算
- EP（专家并行）：把 MoE 的不同专家放在不同 GPU
- ZeRO：把优化器状态（ZeRO-1）、梯度（ZeRO-2）乃至参数（ZeRO-3）分片到各 DP rank 上以降低单卡显存的方法（见 3.3）
  - FSDP 是 PyTorch 对 ZeRO-3 的原生实现

| 框架 | 开发方 | 核心能力 | 适用场景 |
|---|---|---|---|
| Megatron-LM / Megatron-Core | NVIDIA | TP/SP/PP/CP/EP 全套，性能最好，和 TE 深度集成 | 超大规模预训练 |
| DeepSpeed | Microsoft | ZeRO 1/2/3、offload、Ulysses、推理 | 中等规模、显存紧张 |
| PyTorch FSDP / FSDP2 | Meta | 原生 ZeRO-3，DTensor 抽象 | 研究、微调、中等规模预训练 |
| TorchTitan | Meta | 基于 FSDP2 + TP + PP + CP 的参考实现，可组合，支持 `torch.compile`、Float8 | 原生 PyTorch 预训练 |
| JAX / MaxText | Google | GSPMD 自动分片，TPU 原生 | TPU 集群 |
| Transformer Engine | NVIDIA | FP8 / MXFP8 GEMM、融合的 attention 和 norm 层 | 被 Megatron、TorchTitan 等调用 |

表中几个术语的含义如下：
- offload：把优化器状态或参数暂存到 CPU 内存乃至 NVMe，用 PCIe 传输时间换取 GPU 显存
- Ulysses：DeepSpeed 的序列并行方案
  - 用 All-to-All 通信在序列维切分与注意力头维切分之间转换
- DTensor：PyTorch 的分布式张量抽象
  - 用一个张量对象描述它在设备网格上的切分或复制方式
- GSPMD：XLA 编译器的自动分片机制
  - 用户只标注少量张量的切分方式
  - 编译器推导其余张量的切分并插入通信
- Transformer Engine（TE）本身不是完整的训练框架
  - 它是一组支持 FP8 等低精度格式的算子库

各框架的设计取舍可以从三个方面理解：

- Megatron-LM 的出发点是在 NVIDIA GPU 上取得最高吞吐
  - 它把 TP、PP、CP、EP 等并行方式实现为显式的代码路径，每种切分方式下的通信位置都由开发者手工安排
    - 因此可以与计算精确重叠
    - 并直接调用 TE 的 FP8 GEMM 和融合算子（多个逐元素操作合并成一个 kernel，减少显存读写，见 2.2）
  - 代价是代码与模型结构耦合较紧
    - 新增一种模型结构往往需要修改并行逻辑
  - 公开报告中 MFU 在 40%–55% 区间的万卡级稠密模型训练，多数基于 Megatron 或其衍生实现
- DeepSpeed 的核心贡献是 ZeRO 系列显存优化和 offload
  - 以 70B 模型的混合精度 Adam 训练为例，参数、梯度和优化器状态合计约 $16\Psi \approx 1.1$ TB，远超单块 80 GB GPU 的容量
  - ZeRO-3 把这部分状态均分到 64 块 GPU 上后，每块 GPU 只需约 17.5 GB，再加上激活即可放下
  - offload 进一步把优化器状态移到 CPU 内存，使少量 GPU 也能训练较大的模型
    - 但每步需要经 PCIe（每块 GPU 约 50 GB/s）往返传输状态，吞吐会明显下降
- PyTorch FSDP2 与 TorchTitan 的目标是用原生 PyTorch 组件组合出完整的预训练系统
  - FSDP2 以 DTensor 表示每个参数的分片方式
    - TP、PP、CP 可以在同一套抽象上叠加
    - 并能与 `torch.compile` 编译优化和 Float8 训练配合
  - 它的代码量远小于 Megatron，便于研究人员修改
  - 规模扩展到数千块 GPU 时性能接近专用框架
  - JAX/MaxText 则依靠 GSPMD 在编译期完成切分，主要面向 TPU

#### 4.1.2 后训练（RL）框架

- 后训练：在预训练模型上进行的监督微调与强化学习阶段
- RL 后训练的特点是一次迭代包含两类负载：
  - rollout：用当前策略模型自回归生成回答，属于推理负载
  - 训练：根据奖励计算梯度并更新参数
- 因此框架需要同时调度训练引擎和推理引擎

| 框架 | 特点 |
|---|---|
| verl（字节） | HybridFlow：单控制器调度、训练与 rollout 共置，Megatron / FSDP + vLLM / SGLang |
| OpenRLHF | Ray + vLLM + DeepSpeed，部署和使用门槛较低 |
| NeMo-RL（NVIDIA） | Megatron 后端 |

三者的主要差别在于训练引擎与推理引擎的部署方式：
- verl 采用共置（colocate）方案
  - 同一组 GPU 在 rollout 阶段运行 vLLM 或 SGLang 生成回答，在训练阶段切换为 Megatron 或 FSDP 计算梯度
  - 两个阶段之间需要在两种引擎的参数布局之间转换权重
  - 这种方式避免了 GPU 在某一阶段空闲，但要求权重转换和显存释放足够快
- OpenRLHF 用 Ray 把训练进程和推理进程分别部署为独立的 actor（Ray 中可远程调度的有状态进程）
  - 结构清晰，适合中小规模实验
- NeMo-RL 直接复用 Megatron 的并行实现
  - 适合已有 Megatron 训练流程的团队
- RL 后训练系统的详细设计见 8.8 节

### 4.2 容错机制：checkpoint、弹性训练与 straggler 检测

#### 4.2.1 大规模集群的故障统计

- Llama 3 405B 在 16,384 块 H100 上预训练，54 天内发生 466 次作业中断
  - 其中 47 次为计划内中断（维护、升级）
  - 419 次为意外中断，约 78% 的意外中断由硬件问题引起

| 原因（意外中断） | 占比 |
|---|---|
| GPU 故障（含 NVLink） | 30.1% |
| GPU HBM3 | 17.2% |
| 软件 bug | 12.9% |
| 网络交换机 / 线缆 | 8.4% |
| 主机维护 | 7.6% |
| 其他（SRAM、GPU 系统处理器、静默数据损坏、电源等） | 其余 |

- 表中 HBM3 是 H100 使用的高带宽显存
- 按 54 天约 1296 小时计算，平均约每 3 小时发生一次意外中断
  - 在这一故障频率下，该作业的有效训练时间仍保持在 90% 以上
  - 这一结果依赖于本节后续介绍的 checkpoint、快速恢复和自动化运维机制

#### 4.2.2 作业级 MTBF 与集群规模

- MTBF（Mean Time Between Failures，平均无故障时间）：系统相邻两次故障之间运行时间的期望值
- 同步训练作业中任何一个部件失效都会导致整个作业中断
- 设单个部件的 MTBF 为 $M_0$，各部件故障相互独立且服从指数分布，则由 $N$ 个部件组成的作业的 MTBF 为：

$$M_{\text{job}} = \frac{M_0}{N}$$

- 若单块 GPU 的 MTBF 为 5 万小时（约 5.7 年），则 16K 块 GPU 的作业 MTBF 约为 3 小时
  - 与 Llama 3 的实测结果吻合
- 单块 GPU 极少出故障，但作业规模增大到上万块时，故障间隔会缩短到小时量级

同一结论也可以用某一时间窗口内的故障概率表述：
- 设每个部件在该窗口内独立发生故障的概率为 $p$，则 $N$ 个部件中至少有一个发生故障的概率为：

$$P(\ge 1\ \text{failure}) = 1-(1-p)^N$$

- 当 $p$ 很小时，$1-(1-p)^N \approx 1-e^{-Np}$
- 以 1 小时为窗口，单块 GPU 在 1 小时内故障的概率 $p = 1/50000 = 2\times10^{-5}$
  - 对 1024 块 GPU，$Np \approx 0.02$，1 小时内至少一次故障的概率约为 2%
  - 对 16,384 块 GPU，$Np \approx 0.33$，概率约为 $1-e^{-0.33}\approx 28\%$
  - 若窗口扩大到 24 小时，16,384 块 GPU 的 $Np \approx 7.9$，至少发生一次故障的概率超过 99.9%
- 也就是说，在万卡规模上，连续一天不发生任何故障几乎不可能
  - 训练系统必须把故障当作每天都会发生的事件来设计

上述模型假设各部件的故障相互独立，实际集群中这一假设并不成立：
- 一台机柜的电源或配电单元故障会同时使柜内所有节点断电
- 一台叶交换机（leaf switch，直接连接服务器网卡的第一级交换机）故障会使其下连接的数十块 GPU 同时失去网络
- 同一批次的硬件可能带有相同的制造缺陷
- 一次错误的固件或驱动升级会同时影响所有升级过的节点
- 这类相关故障（correlated failure）的特点是单次事件波及大量设备
  - 因此独立模型会低估同时失去大量节点的概率
- 对训练系统的直接影响有两点：
  - 热备节点池的容量要按一个故障域（共享同一电源、交换机等单点的设备集合）的规模而不是按单个节点来预留
  - checkpoint 的副本不能只存放在同一故障域内的邻居节点上
    - 否则一次机柜断电会同时丢失原始状态和备份

#### 4.2.3 最优 checkpoint 间隔（Young / Daly 公式）

- checkpoint：训练过程中定期写入持久存储的状态快照
  - 包含从该时刻恢复训练所需的全部状态（参数、优化器状态、数据位置、随机数状态等，见 4.7.1）
- 作业中断后，训练从最近一次 checkpoint 重新开始
  - 此后到中断时刻之间完成的计算全部丢失

- 设保存一次 checkpoint 的耗时为 $\delta$，保存间隔为 $\tau$，故障率 $\lambda = 1/M$（$M$ 为作业 MTBF）
- 单位时间内浪费的比例由两部分组成：
  - 保存 checkpoint 的开销 $\delta/\tau$
  - 故障发生在间隔内的位置近似均匀分布，平均丢失半个间隔的计算，对应 $\lambda\tau/2$
- 两者相加：

$$W(\tau) = \frac{\delta}{\tau} + \frac{\tau}{2M}$$

- 对 $\tau$ 求导并令导数为 0，得到最优间隔与对应的最小浪费比例：

$$\tau^* = \sqrt{2\delta M},\qquad W^* = \sqrt{\frac{2\delta}{M}}$$

- 这一结果称为 Young 公式
  - Daly 在此基础上给出了包含高阶项和重启时间的修正
  - 在 $\delta \ll M$ 时两者结果接近

数值示例：
- 取 $M = 3$ h，同步 checkpoint 耗时 $\delta = 5$ min：$\tau^*=\sqrt{2\times5\times180}\approx 42$ min，$W^*=\sqrt{10/180}\approx 24\%$
- 若用异步 checkpoint 把 $\delta$ 降到 30 s：$\tau^*\approx 13$ min，$W^*\approx 7.5\%$
- 降到 5 s：$\tau^*\approx 5.5$ min，$W^*\approx 3\%$
- 可见 checkpoint 开销越低，最优保存频率越高，故障造成的损失越小

重启时间 $R$：
- 此外还需计入重启时间 $R$，即从故障发生到训练恢复所需的时间
  - 包括重新调度节点、加载 checkpoint 和初始化 NCCL 通信器
- 每次故障都固定损失 $R$，单位时间内这部分损失为 $R/M$
  - 因此缩短重启时间与降低 checkpoint 开销同等重要

#### 4.2.4 降低 checkpoint 开销 $\delta$：异步、分片与分层保存

| 手段 | 做法 |
|---|---|
| 异步 checkpoint | 先把状态拷到 CPU 内存（几秒），训练继续，后台线程写分布式存储 |
| 分片保存 | 每个 rank 只写自己的分片，并行写入（PyTorch DCP、Megatron dist-ckpt） |
| 内存 checkpoint | 存到邻居节点的 CPU 内存（如 Gemini、ByteCheckpoint 的思路），故障后从内存恢复 |
| 分层 | 高频存内存、低频存远端存储 |

异步 checkpoint 把保存过程拆成两段：
- GPU 到 CPU 内存的拷贝会阻塞训练，但耗时很短
- CPU 内存到分布式存储的写入在后台进行，与后续训练步重叠
- 因此对训练而言，$\delta$ 只剩第一段的耗时
- 以 4.7.3 的 70B 模型为例，同步写入 10 GB/s 的存储需要 84 s
  - 改为异步后阻塞时间降到 0.1–0.2 s
  - 代价是每个节点需要预留与本节点状态等量的 CPU 内存（约 6.6 GB）作为暂存区

分片保存解决的是带宽问题：
- 若所有状态先汇总到 rank 0 再写一个文件，写入带宽受限于单个节点的网卡（如 400 Gb/s，即 50 GB/s）
- 每个 rank 并行写自己的分片后，总带宽可以接近分布式存储系统的聚合带宽
- 分片同时带来一个要求：checkpoint 需要附带元数据
  - 元数据记录每个分片对应原张量的哪一部分
  - 否则无法在不同的并行配置下重新组装

内存 checkpoint 利用的是 CPU 内存与网络远快于分布式存储这一事实：
- 每个节点把自己的状态复制一份到另一个节点的 CPU 内存中
- 单个节点故障时，替换节点可以从备份节点的内存中直接读回状态，恢复时间从分钟级降到秒级
- CPU 内存是易失的，备份节点也可能同时故障（如 4.2.2 所述的相关故障）
  - 所以内存 checkpoint 需要与低频的持久化存储配合使用
- 这种组合称为分层 checkpoint
  - 例如每 5 分钟写一次内存副本，每 1 小时写一次分布式存储

#### 4.2.5 降低重启时间 $R$：故障恢复流程

重启时间 $R$ 由若干串行环节组成，每个环节都有对应的优化手段：

- 热备节点池
  - 集群调度器预留一定数量的空闲节点
  - 作业中某个节点故障后，调度器直接用空闲节点替换它，作业无需重新进入队列等待资源
  - 若不预留，在资源紧张的共享集群上，重新申请 1024 块 GPU 可能需要等待数小时
  - 预留比例需要在资源浪费和恢复速度之间权衡
    - 常见做法是按作业规模预留 1%–3% 的节点，并按 4.2.2 的讨论覆盖至少一个故障域
- 预热的容器镜像和数据缓存
  - 训练容器镜像通常有数十 GB
  - 若替换节点在恢复时才从镜像仓库拉取，单这一步就可能耗时数分钟
  - 备用节点应预先拉取作业所用的镜像，并在本地缓存训练数据的索引和高频访问的分片
- NCCL 初始化优化
  - NCCL 创建通信器时，各 rank 需要交换网络地址、探测拓扑并建立连接
  - 在上万个 rank 的规模下，若所有 rank 都通过同一个协调点交换信息，这一过程可能需要数分钟
  - 常用的优化包括：
    - 分层建立通信组
    - 复用已有通信器的拓扑信息
    - 延迟到首次使用时再建立连接
- 自动化恢复流程
  - 故障检测、节点隔离、重新调度和从最近 checkpoint 重启的全过程应由程序自动完成
  - 若依赖人工值守，夜间故障的响应时间可能达到数十分钟到数小时
    - 按 3 小时一次故障计算，这部分人工延迟会成为有效训练时间的主要损失来源
  - 自动化流程通常先在故障节点上运行健康检查（GPU 诊断、NCCL 带宽测试、ECC 计数检查）
    - 确认故障后将节点标记为不可调度，再拉起作业

#### 4.2.6 弹性训练

- 弹性训练（elastic training）：部分节点失效时作业不终止，而是在剩余节点上缩小并行规模继续训练（例如 DP 从 32 降到 31）
- 实现弹性训练需要解决三个问题：

1. global batch 与学习率的调整：若保持 global batch 不变，需要相应增加梯度累积步数；若改变 global batch，需要按相应规则调整学习率。
2. 数据顺序的衔接：sampler 需要支持按 consumed samples（已消耗的样本总数）定位续接位置，保证缩容前后不重复、不遗漏样本。
3. checkpoint 的重切分（resharding）：并行拓扑改变后，原先按旧拓扑切分保存的参数和优化器状态需要按新拓扑重新切分加载。

- TorchElastic、DeepSpeed Elastic 支持 DP 维度的弹性
- TP/PP 维度的弹性一般通过 checkpoint 重切分加重启实现
  - 因为 TP/PP 组的成员变化会改变每块 GPU 持有的参数切片

#### 4.2.7 Straggler 的影响与检测

- straggler（慢节点）：同步训练中步耗时显著长于其他 rank 的 GPU 或节点
- 同步训练的每一步都需要所有 rank 完成计算并参与集合通信，因此步耗时由最慢的 rank 决定：

$$T_{\text{step}} = \max_i T_i$$

- 若一块 GPU 因降频（过热、功耗封顶）、HBM 纠错、PCIe 链路降速或网卡抖动而变慢 20%，整个万卡作业也会变慢 20%

straggler 难以发现的原因在于它不会报错：
- 作业仍在正常运行，只是整体变慢
- 在同步训练中，所有 rank 的步耗时看起来都相同
  - 快的 rank 把多出来的时间花在了等待集合通信上
- 因此不能用总步耗时判断，而要把每个 rank 的时间拆分为计算时间和通信等待时间分别观察

常用的检测方法有三种：

- 第一种是按 rank 拆分步耗时
  - 每个 rank 在每步记录自身的计算时间（从步开始到发起梯度通信）和通信等待时间
  - 正常情况下各 rank 的计算时间接近
  - 若某个 rank 的计算时间比中位数高出 15%，同时其他所有 rank 的通信等待时间都相应增加了约同样的数值，这个 rank 就是 straggler
- 第二种是集合通信到达时间统计，MegaScale 采用了这一方法
  - 在每次集合通信开始前记录时间戳，汇总后统计每个 rank 最后一个到达的次数
  - 正常情况下最后到达的 rank 是随机的
  - 若某个 rank 在 1000 次通信中有 900 次最后到达，即可判定它是慢节点
  - 这种方法可以在不修改模型代码的前提下，只在通信库层面实现
- 第三种是主动基准测试
  - 定期（如每次作业启动前，或每天一次）在每块 GPU 上运行固定规模的 GEMM，在节点之间运行 NCCL 带宽测试
  - 将结果与同型号 GPU 的性能分布比较
  - 例如 H100 上某一固定尺寸 BF16 GEMM 的吞吐若普遍为 700 TFLOPS 左右，而某块 GPU 只有 600 TFLOPS
    - 即使它尚未在作业中造成明显影响，也应将其剔除或送检
  - 这种方法能在作业受影响之前发现降频、散热不良和链路降速等问题

#### 4.2.8 容错能力的评测指标

- 训练与推理系统的性能通常以健康状态下的平均吞吐（token/s）衡量
- 但在设备数量达到上万块时，硬件故障、网络中断、人为操作失误和恢复过程本身可能主导实际可用的吞吐
  - 一个健康状态下吞吐高 10%、但每次故障需要 1 小时才能恢复的系统，在 3 小时一次故障的条件下，实际产出会低于吞吐稍低、恢复只需 5 分钟的系统
- 因此评测一个大规模系统时，除平均吞吐外还应纳入以下指标：

- 有效训练时间比例（goodput）
  - goodput 指实际推进训练进度的时间占作业总运行时间的比例
  - 它扣除了 checkpoint 开销、故障后丢失的计算和重启时间
  - 4.2.1 中 Llama 3 的 90% 以上即是这一指标
  - 按 4.2.3 的模型，goodput 约为 $1 - W(\tau) - R/M$
- 故障恢复时间
  - 从故障发生到训练恢复到故障前吞吐水平所经历的时间，即 4.2.3 中的 $R$ 加上故障检测时间
  - 评测时应分别统计不同故障类型（单 GPU 故障、单节点故障、交换机故障）的恢复时间
    - 因为它们涉及的恢复路径不同
- 尾部延迟（P95/P99）
  - P95 指将所有测量值从小到大排序后第 95% 位置的数值，P99 同理
  - 对训练而言，关注的是步耗时的 P99
    - 平均步耗时 10 s 而 P99 为 15 s，说明有约 1% 的步明显变慢
    - 这通常对应 straggler、存储抖动或垃圾回收停顿
  - 对推理服务而言，用户感知的是单个请求的延迟
    - P99 延迟比平均延迟更能反映服务质量
- 任务成功率
  - 指提交的作业（或推理请求、智能体任务）在不需要人工介入的前提下正常完成的比例
  - 自动恢复流程有缺陷的系统，平均吞吐可能很高，但会有一定比例的作业因无法自动恢复而需要人工重启

除了这些指标，系统设计上还有两项支撑能力：
- 故障域隔离
  - 调度和数据放置时把可能同时失效的设备（同一电源、同一交换机下的节点）视为一个故障域
  - 保证单个故障域失效时作业能够在其他故障域的资源上恢复
  - checkpoint 副本也分布在不同故障域中
- 可观测性（observability）
  - 系统持续采集每块 GPU 的温度、功耗、频率、ECC 错误计数，NVLink 和网卡的错误计数，以及每个 rank 的步耗时和通信时间
  - 使故障发生后能够迅速定位到具体的设备
  - 并在设备完全失效之前根据指标的变化趋势提前将其替换

### 4.3 数据管线：吞吐、预处理与可恢复性

#### 4.3.1 吞吐需求估算

- 以 1024 块 H100 训练 70B 模型、MFU 为 45%（MFU 的定义见 4.4）为例
- H100 的 BF16 稠密峰值算力为 989 TFLOPS，每个 token 的训练计算量约为 $6N$（$N$ 为参数量）
- 每秒处理的 token 数为：

$$\frac{1024\times 0.45\times 989\times10^{12}}{6\times70\times10^9} \approx 1.09\times10^6\ \text{token/s}$$

- 若 token id 以 uint16 存储，每个 token 占 2 字节，所需读取带宽仅约 2 MB/s
  - 存储带宽不构成瓶颈
- 数据管线的主要问题在于：
  - 读取延迟的抖动
  - CPU 预处理的开销
  - 多模态数据中计算量很大的图像与视频解码

#### 4.3.2 常用设计

| 环节 | 做法 |
|---|---|
| 离线预处理 | 提前分词，存成二进制 token 文件 + 索引（Megatron 的 `.bin` / `.idx`），训练时 mmap 读取 |
| Packing | 多条短文档拼成定长序列，减少 padding；配合 document mask 防止跨文档注意力 |
| 数据混合 | 按权重从多个数据源采样，采样索引由 seed 预先算好（Megatron blended dataset） |
| 预取 | DataLoader 多 worker + `pin_memory` + 异步 H2D 拷贝 |
| 可恢复 | 记录 consumed samples 或用 `StatefulDataLoader`，续训时精确接上 |

表中术语：
- mmap：把文件映射到进程的虚拟地址空间，按需由操作系统分页读入，无需一次性加载整个文件
- Packing：把多条短文档拼接成固定长度的序列，以减少填充（padding）造成的无效计算
- document mask：一种注意力掩码，使拼接序列中的每个 token 只关注同一文档内的 token
- `pin_memory`：把数据放入页锁定内存，使主机到设备（H2D）的拷贝可以异步进行并与计算重叠

离线预处理的作用是把分词从训练关键路径上移除：
- 对 4.3.1 中每秒 109 万 token 的吞吐，若在训练时实时分词，按单个 CPU 核每秒处理约 1 MB 文本、每个 token 约 4 字节文本计算，需要数十个 CPU 核持续工作
- 分词耗时随文本内容波动，会造成步耗时抖动
- 预先分词并存成二进制文件后，训练时读取一个样本只需按索引计算偏移量，再从 mmap 映射的文件中取出一段连续字节
  - 开销可以忽略

Packing 的收益取决于文档长度分布：
- 若训练序列长度为 8192，而网页文档的平均长度约 1000 token，不做 packing 时每条序列约有 88% 是 padding
  - 这些位置的计算全部浪费
- packing 后填充比例可以降到 1% 以下
- 拼接后若不加 document mask，序列中后一篇文档的 token 会关注到前一篇无关文档的内容，引入噪声
- FlashAttention 等实现支持变长序列接口
  - 可以在不显式构造掩码矩阵的情况下实现文档级隔离

数据混合的采样索引需要预先确定：
- 若在训练时按权重实时随机选择数据源，续训后很难复现同样的选择序列
- Megatron 的 blended dataset 在训练开始前就根据权重和种子生成全部样本的来源与下标，训练时只按 step 查表
- 这一设计也是 4.3.3 确定性索引的基础

#### 4.3.3 确定性的样本索引

- 确定性训练（4.10 节）和断点续训都要求：给定 step，能够计算出该步每个 rank 读取哪些样本
- 最直接的实现方式是让样本索引只依赖 (seed, epoch, global_step, dp_rank)，而不依赖迭代器的内部状态
- 这样无论从哪一步恢复，都无需重放之前的迭代过程：

```python
import numpy as np

class DeterministicSampler:
    """第 step 步、第 dp_rank 个 rank 的样本下标只由 (seed, step, rank) 决定。"""
    def __init__(self, n_samples, global_batch, dp_size, seed=1234):
        self.n, self.gb, self.dp, self.seed = n_samples, global_batch, dp_size, seed

    def indices(self, step, dp_rank):
        per_epoch = self.n // self.gb
        epoch, k = divmod(step, per_epoch)
        perm = np.random.default_rng(self.seed + epoch).permutation(self.n)
        batch = perm[k * self.gb:(k + 1) * self.gb]
        mb = self.gb // self.dp
        return batch[dp_rank * mb:(dp_rank + 1) * mb]

s = DeterministicSampler(n_samples=10_000_000, global_batch=512, dp_size=32)
print(s.indices(step=123456, dp_rank=7)[:4])   # 任意时刻重新计算，结果相同
```

### 4.4 MFU 与 HFU：算力利用率的度量

#### 4.4.1 定义

- MFU（Model FLOPs Utilization，模型算力利用率，由 PaLM 论文提出）：实际训练吞吐对应的模型必需计算量与集群峰值算力之比：

$$\text{MFU} = \frac{\text{tokens/s}\times C_{\text{model}}}{N_{\text{GPU}}\times P_{\text{peak}}},\qquad C_{\text{model}} = 6N + 12Lhs$$

- 其中各符号含义如下：
  - $C_{\text{model}}$：每个 token 前向加反向所需的计算量
  - $N$：参数量
  - $L$：层数
  - $h$：隐藏维度
  - $s$：序列长度
  - $P_{\text{peak}}$：单块 GPU 的峰值算力
- $6N$ 对应所有权重矩阵乘法（前向 $2N$，反向 $4N$）
- $12Lhs$ 对应注意力中 $QK^\top$ 与注意力权重乘 $V$ 两次矩阵乘法
  - 它们不涉及权重，因此不包含在 $6N$ 中

- MFU 只计入模型本身必需的计算
- HFU（Hardware FLOPs Utilization，硬件算力利用率）则计入硬件实际执行的全部计算，包括激活重算等额外计算：

$$\text{HFU} = \frac{\text{tokens/s}\times C_{\text{actual}}}{N_{\text{GPU}}\times P_{\text{peak}}}$$

- 激活重算：反向传播时重新执行前向计算以恢复激活值，用计算量换取显存
- 开启全量重算时前向执行两次，$C_{\text{actual}}\approx 8N$，HFU 比 MFU 高约 1/3
- 开启重算会提高 HFU 而不提高训练速度
  - 因此比较不同系统的真实效率应使用 MFU

#### 4.4.2 MFU 计算示例

70B 模型，1024 块 H100，实测吞吐 1.0M token/s，$L=80$，$h=8192$，$s=8192$：

$$C_{\text{model}} = 6\times70\times10^9 + 12\times80\times8192\times8192 \approx 4.2\times10^{11} + 6.4\times10^{10} = 4.84\times10^{11}$$

$$\text{MFU} = \frac{10^6\times4.84\times10^{11}}{1024\times989\times10^{12}} \approx 47.8\%$$

- 本例中注意力项约占总计算量的 13%
- 序列越长，该项占比越高
  - 因此长上下文训练计算 MFU 时不能省略

#### 4.4.3 公开报告的 MFU

| 来源 | 规模 | MFU |
|---|---|---|
| PaLM 540B | 6144 TPU v4 | 46.2% |
| MegaScale 175B | 12,288 GPU | 55.2% |
| Llama 3 405B | 16K H100，BF16 | 38–43% |
| DeepSeek-V3 | 2048 H800，FP8，MoE | MoE 的 MFU 定义不同，一般更低 |

- MoE 模型每个 token 只激活部分专家，计算量按激活参数而非总参数统计
- MoE 的 All-to-All 通信占比较高
- 因此其 MFU 与稠密模型不能直接比较

#### 4.4.4 MFU 损失的构成

| 损失 | 典型量级 |
|---|---|
| GEMM 本身达不到峰值 | 70–85% |
| 非 GEMM 算子（norm、激活、softmax） | 5–15% |
| 未重叠的通信 | 5–20% |
| PP 气泡 | 3–20% |
| 数据加载、Python 开销、同步 | 1–5% |
| straggler | 视情况 |

- 第一行表示即使在大尺寸矩阵乘法上，实际吞吐通常也只能达到峰值的 70–85%
- 其余各行为相对总步耗时的额外损失

各项损失的成因如下：

- GEMM 达不到峰值，原因包括：
  - 矩阵尺寸不是 tile 大小的整数倍时边缘 tile 利用率低
  - GPU 在高负载下因功耗封顶而降频
  - kernel 的流水线在启动和结束阶段无法填满
  - H100 上大尺寸 BF16 GEMM 的实测吞吐一般在 650–800 TFLOPS，即峰值的 66%–81%
- 非 GEMM 算子
  - 包括 LayerNorm/RMSNorm、激活函数、softmax、残差相加等
  - 这些算子每读写一个元素只做少量运算
    - 按 roofline 模型（以算术强度和峰值算力、显存带宽判断性能上限的模型，见 1.3）属于访存受限
    - 其耗时由 HBM 带宽而不是算力决定
    - 在计算 MFU 时这部分时间几乎没有贡献有效 FLOPs
  - 算子融合可以减少这部分损失
- 未重叠的通信
  - 指 GPU 在等待集合通信完成而不进行计算的时间
  - TP 的 AllReduce 位于每层前向和反向的关键路径上，难以与计算重叠
    - 因此 TP 通常限制在 NVLink 连接的单个节点内
  - DP 的梯度通信可以与反向计算重叠
    - 但最后几层的梯度通信往往无法完全隐藏
- PP 气泡
  - 指流水线并行中各 stage 等待前后 stage 数据而空闲的时间（见 3.5）
  - 对 1F1B 调度，气泡比例约为 $(p-1)/(m+p-1)$，其中 $p$ 为 stage 数，$m$ 为每步的 micro-batch 数
  - $p=8$、$m=32$ 时气泡约为 18%
  - 增大 $m$ 或采用交错调度可以降低这一比例
- 数据加载、Python 开销与同步
  - Python 解释器逐个发射 kernel 的开销在模型较小、每个 kernel 耗时很短时尤为明显
  - `.item()` 等调用会强制 CPU 等待 GPU，打断异步执行
  - CUDA Graph 把一系列 kernel 录制为一次提交，可以消除大部分发射开销

### 4.5 实验管理与可追溯性

#### 4.5.1 训练运行的元数据要求

每一次训练运行（run）都应能准确回答以下问题，否则其结果无法复现，也无法与其他运行比较：

1. 代码对应哪个 commit，是否存在未提交的改动？
2. 配置是什么（模型、并行、优化器、数据配比）？
3. 数据是哪个版本？
4. 运行环境是什么（镜像、CUDA、NCCL、驱动、框架版本）？
5. 随机种子是什么？
6. 从哪个 checkpoint 开始？

#### 4.5.2 记录规范

| 项 | 做法 |
|---|---|
| 配置 | 单一配置文件，启动时整份写进日志和 checkpoint |
| 代码 | 启动时记录 `git rev-parse HEAD` 和 `git diff`，有未提交改动就告警 |
| 环境 | 固定容器镜像 tag，记录 `torch.__version__`、`torch.version.cuda`、NCCL 版本、驱动版本 |
| 日志 | 每步记录 loss、grad norm、lr、loss scale、tokens/s、MFU、显存峰值 |
| 观测 | 每层的激活 / 权重 / 梯度范数，定期记录，出问题时能定位到层 |
| 小规模先行 | 新改动先在小模型上做 loss 曲线对比，再上大规模 |

- grad norm：全部参数梯度拼接后的 L2 范数（计算方式见 4.6.2）
- loss scale：FP16 混合精度训练中为防止梯度下溢而乘到 loss 上的缩放系数

表中各项的目的可以概括为：任何一次训练结果都能追溯到确定的代码、配置、数据和环境。
- 配置与代码版本必须写进 checkpoint 而不仅是日志
  - checkpoint 会被复制、迁移和长期保存，日志则可能丢失
  - 数月后从某个 checkpoint 续训时，只有 checkpoint 内的配置能可靠地说明它是如何产生的
- 环境版本的记录同样必要
  - cuBLAS 或 NCCL 的小版本升级可能改变 kernel 的选择，使 loss 在低位上产生差异（见 4.10.3）
  - 若未记录版本，这类差异无法与代码改动造成的差异区分

逐层范数的记录频率需要权衡：
- 每步记录所有层的激活、权重和梯度范数会产生额外的归约计算和日志量
- 常见做法是每 10–100 步记录一次
  - 出现 loss spike 时再通过 replay（4.10.8）补全异常前后每一步的数据
- 小规模先行的依据是，大多数代码错误和数值问题在 1B 以下的模型上同样会表现为 loss 曲线的偏离
  - 小模型的一次实验只需数小时
  - 成本比在大规模上发现问题低两到三个数量级

#### 4.5.3 启动清单（manifest）的生成

- manifest 是在作业启动时生成的一份元数据文件，记录 4.5.1 中列出的全部信息
- 以下函数在 rank 0 上收集这些信息：

```python
import json, os, subprocess, torch

def run_manifest(cfg: dict) -> dict:
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    return {
        "git_commit": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "gpu": torch.cuda.get_device_name(), "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "CUDA_", "PYTORCH_", "NVTE_"))},
        "config": cfg,
    }

# rank 0 启动时：json.dump(run_manifest(cfg), open(f"{out_dir}/manifest.json", "w"), indent=2)
```

### 4.6 单个训练 step 的执行流程：FSDP 与 Megatron 对比

#### 4.6.1 各阶段的计算与通信

- 下表对比两种典型配置在一个训练 step 内各阶段的行为
- AllGather、ReduceScatter、AllReduce 是三种集合通信原语：
  - AllGather：把各 rank 的分片拼接成完整张量并分发给所有 rank
  - ReduceScatter：先求和再把结果按分片分发
  - AllReduce：等价于二者的组合
- P2P 指点对点的发送与接收

| 阶段 | FSDP（ZeRO-3） | Megatron（TP + PP + DP + ZeRO-1） |
|---|---|---|
| 取数据 | 每个 rank 取自己的 micro-batch | 只有 PP 第一个 stage 取输入，最后一个 stage 取 label；同一 TP 组拿相同数据 |
| 前向 | 逐层 AllGather 参数 → 计算 → 释放 | 每层计算 + TP AllReduce/RS/AG；stage 之间 P2P 传激活 |
| 反向 | 逐层 AllGather 参数 → 算梯度 → ReduceScatter 梯度 | 1F1B 交错，TP 通信，P2P 传梯度 |
| 梯度同步 | 已在反向中 ReduceScatter 完成 | DP 组内 ReduceScatter（分布式优化器）或 AllReduce |
| 梯度裁剪 | 各分片算范数平方 → AllReduce → 开方 | 跨 TP、PP、DP 组 AllReduce 范数 |
| 优化器 | 每 rank 更新自己的分片 | 分布式优化器：每 rank 更新 1/DP 的参数，然后 AllGather |

- 1F1B：一种流水线调度策略
  - 每个 stage 在稳态下交替执行一个 micro-batch 的前向与另一个 micro-batch 的反向
  - 以限制同时驻留的激活数量（见 3.5）
- Megatron 的分布式优化器即 ZeRO-1
  - 优化器状态按 DP rank 分片，每个 rank 只更新自己负责的那部分参数
  - 再通过 AllGather 同步完整参数

两种方案的根本区别在于通信量与切分粒度：
- FSDP 在每层前向和反向时都要 AllGather 该层的完整参数
  - 每步的参数通信量约为参数量的 3 倍（前向一次 AllGather、反向一次 AllGather、梯度一次 ReduceScatter）
  - 这些通信可以与相邻层的计算重叠，但要求 DP 组之间的网络带宽足够高
- Megatron 的 TP 把单层的计算本身切开
  - 通信量与激活大小成正比而与参数量无关，适合放在 NVLink 带宽很高的节点内
- PP 只在 stage 边界传递激活
  - 通信量最小，适合跨节点
- 因此在大规模下，Megatron 的典型配置是节点内 TP=8、跨节点 PP 和 DP
  - FSDP 则依赖高带宽网络在 DP 维度上扩展

取数据一行也反映了这种区别：
- 在 Megatron 中，同一 TP 组的各 GPU 计算的是同一批样本的不同切片
  - 因此必须拿到完全相同的输入
- PP 的中间 stage 的输入来自上一个 stage 的激活
  - 不需要读数据
- FSDP 中每个 rank 都是完整的数据并行副本
  - 各自读取不同的 micro-batch

#### 4.6.2 分片条件下的全局梯度范数

- 梯度裁剪（gradient clipping）：当全部梯度的 L2 范数超过阈值时，按比例缩小所有梯度
- 参数分片之后，每个 rank 只持有部分梯度，裁剪前必须先求出全局范数：

$$\|g\|_2 = \sqrt{\sum_{r}\|g_r\|_2^2}$$

- 实现上，各 rank 先计算本地梯度的平方和，经 AllReduce 求和后再开方
- TP 中被复制到每个 rank 上的参数（如 LayerNorm 权重）在求和时只能计入一次
  - 否则会被重复计数，这是自行实现 TP 时的常见错误

#### 4.6.3 框架选型

| 情况 | 选择 |
|---|---|
| 模型 ≤ 30B，GPU 数 ≤ 数百块 | FSDP2 / TorchTitan，代码简单 |
| 模型 ≥ 70B 或需要长上下文、MoE | Megatron-Core，或 TorchTitan 的 TP+PP+CP 组合 |
| 显存极紧、GPU 数很少 | DeepSpeed ZeRO-3 + offload |

选型的分界点主要由两个因素决定：
- 其一是单层的大小
  - 70B 以上的稠密模型，单层参数和激活已经很大，纯 FSDP 每层 AllGather 的通信量随之增加
  - DP 规模扩大后每块 GPU 分到的 batch 变小，计算不足以掩盖通信
  - 此时需要 TP 和 PP 把单层切开
- 其二是模型结构
  - 长上下文需要 CP 切分序列，MoE 需要 EP 分配专家
  - 这些并行方式在 Megatron-Core 中实现最为完整
- 对于 30B 以下的模型和数百块 GPU 的规模，FSDP2 的性能已足够
  - 代码简单带来的开发和调试效率通常更重要
- 只有在 GPU 数量很少、即使 ZeRO-3 分片后仍放不下模型时，才需要用 offload 以吞吐换取显存

### 4.7 checkpoint 的内容、格式与 IO 开销

#### 4.7.1 checkpoint 的组成

下表中 $\Psi$ 为模型参数量：
- 优化器状态以 Adam 为例
  - $m$ 为梯度的一阶矩（动量）估计
  - $v$ 为二阶矩估计
  - 二者均以 FP32 存储
- FP32 master 权重：混合精度训练中保留的 FP32 参数副本
  - 参数更新在这份副本上进行
  - 前向和反向使用其 BF16 副本

| 内容 | 大小 | 续训必需 |
|---|---|---|
| 模型权重（FP32 master 或 BF16） | $4\Psi$ 或 $2\Psi$ | 是 |
| 优化器状态（$m$、$v$） | $8\Psi$ | 是 |
| LR scheduler、step、consumed samples | 很小 | 是 |
| loss scaler 状态 | 很小 | FP16 时是 |
| 各 rank 的 RNG 状态 | 很小 | 确定性续训时是 |
| 数据加载器状态 | 很小 | 是 |
| 配置、代码版本 | 很小 | 建议 |

- RNG 状态：伪随机数生成器的内部状态
  - 给定这一状态，此后生成的随机数序列完全确定（详见 4.10.5）

从大小上看，checkpoint 的主体是优化器状态和 FP32 master 权重：
- 二者合计 $12\Psi$，约占全部内容的 99% 以上
- 其余各项虽然只有 KB 到 MB 量级，却同样是续训必需的
  - 例如只保存了权重和优化器状态而遗漏 consumed samples，续训时数据会从头开始读取，模型将重复训练已见过的数据
  - 遗漏 LR scheduler 的状态，学习率会回到 warmup 起点，可能直接引发 loss spike

#### 4.7.2 存储格式

| 格式 | 特点 |
|---|---|
| `torch.save` 单文件 | 只适合小模型 |
| PyTorch DCP（Distributed Checkpoint） | 每个 rank 写自己的分片 + 元数据，加载时支持不同并行配置的重切分 |
| Megatron dist-ckpt | 类似，支持 TP/PP 变化后加载 |
| safetensors | 发布推理权重用，零拷贝 mmap，不能执行代码，安全 |

- `torch.save` 基于 Python 的 pickle 序列化，加载时可能执行任意代码
- safetensors 只存储张量数据和元数据，不包含可执行内容
  - 因此适合公开发布权重

`torch.save` 单文件不适合大模型的原因有两个：
- 保存前需要把所有分片汇总到一个进程
  - 单个进程的内存需要容纳整个模型（70B 模型的完整状态为 840 GB）
- 写入只能由一个节点完成，带宽受限

DCP 和 dist-ckpt 的做法：
- 让每个 rank 写自己的分片，同时保存一份全局元数据
  - 元数据记录每个分片在原始张量中的偏移和形状
- 加载时，新的并行配置下每个 rank 根据元数据计算自己需要的那部分数据来自哪些文件的哪些位置，按需读取
  - 从而实现重切分
- 例如以 TP=8 保存的 checkpoint，可以直接以 TP=4 加载
  - 每个新 rank 读取原来两个 rank 的分片并拼接

#### 4.7.3 IO 耗时估算

- 70B 模型的完整 checkpoint（FP32 master 权重加优化器状态）大小为 $70\times10^9\times(4+8) = 840$ GB
- 若另存 BF16 权重则为 980 GB

| 存储写带宽 | 同步写耗时 |
|---|---|
| 10 GB/s | 84 s |
| 100 GB/s | 8.4 s |

采用异步 checkpoint 时，训练只需等待 GPU 到 CPU 内存的拷贝完成：
- 以 1024 块 GPU、128 个节点计，状态全部分片后每个节点需拷贝约 $840/128\approx 6.6$ GB，每块 GPU 约 0.82 GB
- 每块 GPU 的 PCIe 带宽约 50 GB/s
- 计入同一节点 8 块 GPU 共享主机内存带宽等因素，约 0.1–0.2 s 后训练即可继续

#### 4.7.4 基于 PyTorch DCP 的异步保存

- `dcp.async_save` 在完成 GPU 到 CPU 的拷贝后立即返回一个 Future 对象，写入存储的过程在后台进行
- 下一次保存前需要等待上一次写入完成，以免两次保存同时占用内存和存储带宽：

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

def save_async(model, opt, step, extra, path):
    model_sd, opt_sd = get_state_dict(model, opt)
    state = {"model": model_sd, "optim": opt_sd, "step": step, **extra}
    return dcp.async_save(state, checkpoint_id=f"{path}/step_{step:08d}")   # 返回 Future

# 训练循环中
pending = None
if step % ckpt_every == 0:
    if pending is not None: pending.result()      # 等待上一次保存写入完成
    pending = save_async(model, opt, step, {"rng": rng_state_dict(), "data": sampler_state}, ckpt_dir)
```

### 4.8 工程估算示例：straggler 成本、数据配比与 MFU

#### 4.8.1 straggler 的经济成本

一个使用 1024 块 GPU 的作业：
- 正常步耗时 10 s
- 若其中一块 GPU 因散热不良降频而慢 15%
  - 步耗时变为 11.5 s
  - 整体吞吐下降约 13%
- 按每 GPU 小时 2 美元计算，一天的浪费为 $1024\times24\times2\times0.13\approx 6400$ 美元
- 检测并替换这块 GPU 通常只需几分钟
  - 因此持续的 straggler 检测有明确的经济价值

#### 4.8.2 数据配比与重复轮数

- 计划训练 15T token，配比为网页 50%、代码 20%、数学 10%、书籍 10%、多语言 10%
- 代码部分需要 3T token，而代码子集只有 1.5T 不重复的 token
  - 因此需要重复 2 个 epoch
- 经验结果表明，重复 4 个 epoch 以内对模型效果的影响很小
  - 超过后额外训练的收益明显下降（Muennighoff et al., 2023, *Scaling Data-Constrained Language Models*）

#### 4.8.3 由步耗时推算 MFU

已知条件：
- 8B 模型（$L=32$，$h=4096$），序列长度 $s=8192$
- 256 块 H100
- global batch 为 4M token
  - 即 512 条 8192 长度的序列，$4.19\times10^6$ token
- 步耗时 1.9 s

$$C = 6\times8\times10^9 + 12\times32\times4096\times8192 \approx 4.8\times10^{10}+1.29\times10^{10} = 6.09\times10^{10}$$

$$\text{MFU} = \frac{(4.19\times10^6/1.9)\times6.09\times10^{10}}{256\times989\times10^{12}}\approx 53\%$$

### 4.9 训练异常的诊断：loss spike、NaN、通信超时与节点故障

#### 4.9.1 常见异常汇总

本节涉及的几类异常定义如下：
- loss spike：训练损失在少数几步内突然大幅上升的现象
  - 之后可能逐渐回落，也可能持续发散
- NaN（Not a Number）：无定义浮点运算（如 $0/0$）的结果
- Inf：数值溢出产生的无穷大
  - NaN 和 Inf 一旦出现在 loss 或梯度中，会经参数更新扩散到全部参数
- NCCL 超时：某次集合通信在设定时限（PyTorch 默认 10 分钟）内未完成
  - 通常因为部分 rank 没有进入这次通信
- XID：NVIDIA 驱动写入内核日志的 GPU 错误码
- DCGM（Data Center GPU Manager）：NVIDIA 的 GPU 监控与诊断工具

| 现象 | 常见原因 | 排查 | 处理 |
|---|---|---|---|
| loss spike（突然升高后恢复或不恢复） | 坏数据批次；学习率过大；Adam 二阶矩估计在罕见梯度方向上过小；注意力 logits 爆炸 | 看 spike 前后的 grad norm、各层激活范数、对应样本 | 回退到 spike 前的 checkpoint，跳过对应数据；降 lr；QK-norm、z-loss；调 Adam $\epsilon$ 和 $\beta_2$ |
| NaN / Inf | FP16 溢出；除零；log(0)；坏数据；硬件错误 | 打开 anomaly 检测定位第一个 NaN 的算子；同数据换节点重跑 | 修数值问题；换 BF16；隔离坏节点 |
| NCCL 超时 / 挂起 | 某个 rank 崩了或卡住；各 rank 执行的集合通信顺序不一致；网络故障 | `NCCL_DEBUG=INFO`；PyTorch Flight Recorder（`TORCH_NCCL_TRACE_BUFFER_SIZE`）看每个 rank 最后一个集合通信 | 修分支不一致的代码；隔离坏网卡 / 交换机 |
| 节点掉线 | GPU 掉卡（XID 79）、HBM ECC 错误（XID 48/94/95）、主机故障 | `dmesg`、DCGM、`nvidia-smi -q` | 自动隔离 + 从 checkpoint 恢复 |
| 吞吐缓慢下降 | 显存碎片、straggler、存储变慢、日志堆积 | 每 rank 步耗时分布、显存曲线 | 见 2.5、4.2.7 |
| loss 不降 / 发散 | 配置错（lr、warmup、init）；数据错（label 错位、mask 错） | 小模型复现；单卡对拍 | 修配置 |

下面按现象分别说明表中各行的排查思路。

loss spike 的排查：
- 首先看 grad norm
  - 若 spike 前一两步 grad norm 先出现大幅上升，说明问题源于某一批数据或某几层的数值不稳定
- 再看各层激活范数，找出最先出现异常增长的层
- 最后取出该步的样本，检查是否存在大量重复、乱码或格式异常的数据
- 成因和对策详见 4.9.2

NaN/Inf 的排查：
- NaN/Inf 的特点是会迅速扩散
  - 某个算子输出一个 NaN，经过矩阵乘法后整行输出都变为 NaN
  - 一步之后所有参数都被污染
- 因此排查的关键是找到第一个产生 NaN 的算子
- 若同样的数据在其他节点上重跑不产生 NaN，则应怀疑原节点的硬件（见 4.9.4）
- FP16 的最大可表示值约为 65504
  - 注意力 logits 或某些激活很容易超过这一范围
  - 改用 BF16（最大值约 $3.4\times10^{38}$）是最直接的解决方法

NCCL 超时的排查：
- NCCL 超时通常不是网络本身慢，而是参与通信的 rank 没有全部到达
- 集合通信要求组内所有 rank 以相同的顺序调用相同的通信操作
  - 若某个 rank 因为代码中的条件分支（例如只在部分 rank 上执行的日志或评估逻辑）多调用或少调用了一次通信
  - 其余 rank 就会一直等待，直到超时
- 定位方法见 4.9.3

节点掉线的排查：
- 节点掉线可以通过 XID 错误码区分原因
  - XID 79 表示 GPU 已从 PCIe 总线上脱落，驱动无法再访问它
  - XID 48 表示 HBM 发生了不可纠正的双比特 ECC 错误
  - XID 94 和 95 分别表示被控制的和未被控制的 ECC 错误
    - 后者意味着错误可能已经影响了其他进程
- 这类故障无法在软件层面修复，只能隔离节点后从 checkpoint 恢复

吞吐缓慢下降的排查：
- 吞吐缓慢下降指在数小时到数天内步耗时逐渐增加
- 可能的原因
  - 显存碎片会使分配器频繁整理内存（见 2.5）
  - 日志文件不断增大、监控数据积压会占用 CPU 和存储带宽
  - 共享存储在其他作业负载增加时变慢，会延长数据读取和 checkpoint 时间
- 按 rank 记录步耗时分布可以区分是个别 rank 变慢（straggler）还是全体变慢（共享资源问题）

loss 不降或发散的排查：
- loss 不降或从训练开始就发散，通常是配置或数据错误而非硬件问题
- 常见的数据错误
  - label 与输入没有正确错位一个 token，使模型直接看到答案
  - attention mask 错误，使模型看到未来的 token
- 这类问题在小模型上同样能复现
  - 应先在单块 GPU 上与参考实现对拍

表中术语说明：
- anomaly 检测：指 PyTorch 的 `torch.autograd.set_detect_anomaly`
  - 它在反向传播中检查每个算子的输出
  - 报告第一个产生 NaN 的算子及其前向调用栈
- BF16 与 FP16 同为 16 位浮点格式
  - BF16 的指数位与 FP32 相同，表示范围大
  - 因而基本不会溢出
- 单卡对拍：用同样的输入在单块 GPU 上运行未并行化的参考实现
  - 与分布式实现的结果逐项比较

#### 4.9.2 loss spike 的成因与对策

loss spike 在大模型训练中较为常见。目前公认的机制和对应措施有以下几类：

1. 注意力 logits 增长：
   - $QK^\top$ 的数值过大导致 softmax 饱和，梯度异常
   - 对策是 QK-norm（对 Q、K 分别做 LayerNorm 后再计算内积）或对 logits 施加软截断
2. 输出 logits 漂移：
   - 输出层 softmax 的归一化项不受约束地增大
   - 对策是加入 z-loss，$\mathcal{L}_z = 10^{-4}\cdot\log^2 Z$，$Z$ 为 softmax 的分母（PaLM 的做法）
     - 使 $\log Z$ 保持在 0 附近
3. Adam 的更新尺度：
   - 某些参数的梯度长期很小，其二阶矩估计 $v$ 也很小
   - 突然出现一个大梯度时，更新量 $m/\sqrt{v}$ 会急剧增大
   - 对策是减小 $\beta_2$（如 0.95，使 $v$ 更快跟上梯度变化）、增大 $\epsilon$ 以及梯度裁剪
4. 异常数据：
   - 如大量重复字符、超长乱码
   - PaLM 的做法是回退到 spike 前约 100 步的 checkpoint，并跳过之后的 200–500 个 batch

#### 4.9.3 NCCL 挂起的定位方法

相关机制：
- PyTorch 的 ProcessGroupNCCL 中有一个 watchdog 线程
  - 它周期性检查已提交的集合通信是否完成
  - 超过时限即判定为超时，中止通信器并报错
- Flight Recorder 是 PyTorch 内置的环形缓冲区
  - 记录每个 rank 最近若干次集合通信的类型、序号、张量大小、完成状态和调用栈
  - 超时时可将内容写入文件

定位挂起问题时通常开启以下设置：

```bash
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000        # Flight Recorder：记录最近的集合通信
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1          # 超时后抛错退出，避免进程无限期挂起
```

超时后的分析：
- 超时后每个 rank 会输出最近的集合通信记录
- 比较各 rank 的最后一条记录
  - 若大部分 rank 停在第 $k$ 次 AllReduce，而个别 rank 停在第 $k-1$ 次或正在执行其他通信，问题就出在这些 rank 上
- 常见原因
  - 这些 rank 崩溃
  - 这些 rank 卡在计算中
  - 因代码分支条件不同而执行了不同的通信序列

#### 4.9.4 静默数据损坏（SDC）

- 静默数据损坏（silent data corruption，SDC）：硬件产生了错误的计算结果，但没有触发任何错误报告或异常
- Meta、Google 都报告过大规模集群中 SDC 的实际发生率
  - Llama 3 的中断原因中也列出了 SDC

典型表现：
- 某一步的 loss 或 grad norm 突然异常，且无法复现
- 模型质量出现原因不明的下降

检测方法：

1. 同一 DP 组内各 rank 更新后的参数应当完全一致（ZeRO-1/2 下指 AllGather 之后的参数）
   - 定期对参数计算哈希并比对
   - 不一致说明有 rank 计算出错
2. 定期在每块 GPU 上用固定输入运行确定性计算（GEMM、小模型前向），与参考结果逐位比对
3. 出现 spike 时，用 replay（4.10 节）在不同节点上重新计算同一步
   - 结果与其他节点不同的节点即为嫌疑节点

这一方法把确定性与稳定性联系起来：
- 只有训练过程是确定性的，才能通过重新计算并比对结果来识别硬件错误

SDC 的危害在于影响可能长期潜伏：
- 一次单比特翻转若发生在某个权重的高位指数上
  - 可能使该权重变成极大的数值，立即引发 loss spike
- 若发生在尾数的低位
  - 影响可能小到无法从 loss 曲线上察觉
  - 但错误会随参数更新一直保留在模型中
- 对于 DP 组内参数哈希比对，若比对周期为每 1000 步
  - 一旦发现不一致，只需回退到上一次比对一致时的 checkpoint
  - 即可保证损失不超过 1000 步的计算

#### 4.9.5 硬件纠错与冗余计算：ECC、冗余与三模冗余

应对硬件错误的手段按代价从低到高可以分为四类：纠错码、冗余、三模冗余和 checkpoint。

ECC（Error-Correcting Code，纠错码）：
- 在存储或传输的数据之外附加若干校验位，使硬件能够检测并纠正一定数量的位错误
- 数据中心 GPU 的 HBM 和片上 SRAM 普遍采用 SECDED（Single Error Correction, Double Error Detection）编码
  - 例如每 64 位数据附加 8 位校验位，可以纠正任意 1 位错误，并检测出 2 位错误
- 单比特错误被纠正后计数加一，不影响计算
- 双比特错误无法纠正，驱动会报告 XID 48 并终止相关进程（4.9.1）
- ECC 的局限在于它只保护存储和传输的数据，不保护运算本身
  - 若 Tensor Core 的乘加单元因制造缺陷或老化而算错，写入 HBM 的已经是错误的结果
  - ECC 会如实地保护这个错误值
  - 这正是 SDC 无法被 ECC 消除的原因

冗余（redundancy）：为同一计算、数据或系统资源保留多份副本以提高可靠性
- 在训练中，数据层面的冗余包括 checkpoint 的多副本存储（4.2.4）
- 资源层面的冗余包括热备节点池（4.2.5）
- 计算层面的冗余是让同一计算执行多次并比对结果
- 4.9.4 中 DP 组内参数哈希比对利用的是数据并行天然存在的计算冗余
  - 同一 DP 组的各 rank 用相同的梯度更新相同的参数，结果理应完全相同
  - 这份冗余无需额外计算即可获得

三模冗余（Triple Modular Redundancy，TMR）：计算冗余的典型形式
- 同一计算由三个独立的单元同时执行，由表决器取多数结果作为输出
  - 只要三个单元中至多一个出错，表决结果就是正确的
  - 若只用两份副本，只能检测出不一致，无法判断哪一份是正确的
- 设单个单元出错的概率为 $q$，各单元独立
  - TMR 输出错误要求至少两个单元同时出错，概率为 $3q^2(1-q) + q^3 \approx 3q^2$
  - $q=10^{-4}$ 时，TMR 的错误概率约为 $3\times10^{-8}$，降低了三个数量级以上
- TMR 的代价是计算资源和功耗增加到原来的 3 倍以上
  - 因此主要用于航天、汽车控制等对单次错误零容忍的场景
  - 大模型训练不会对全部计算采用 TMR
- 训练中可以借鉴的做法是选择性冗余
  - 只对少量关键计算（如梯度范数、loss 的归约）或抽样的部分步进行重复计算和比对
  - 以较小的开销获得 SDC 检测能力

checkpoint：代价最低、覆盖范围最广的手段
- 它不阻止错误发生，而是保证任何错误被发现之后都能回退到错误发生前的状态
- 三者分工构成训练系统应对硬件错误的完整链条
  - ECC 处理高频的单比特错误
  - 参数哈希比对和抽样冗余计算负责发现 SDC
  - checkpoint 负责在发现后恢复

### 4.10 确定性训练与 replay

- 确定性训练：给定相同的代码、配置、数据和初始状态，训练过程每次产生相同的结果
  - 本节讨论的最高要求是逐位确定性（bitwise determinism），即两次运行得到的每个浮点数在二进制表示的每一位上都相同
- replay（重放）：从异常发生前的 checkpoint 出发，用与原运行完全相同的数据和设置重新执行若干步
  - 目的是检验异常能否复现

#### 4.10.1 确定性的应用场景

| 场景 | 需要什么 |
|---|---|
| loss spike 调查 | 回退到 spike 前的 checkpoint 重跑，看 spike 是否复现。复现了才能二分定位是数据、代码还是硬件 |
| SDC 排查 | 同一步在不同节点上重算，比对结果 |
| 代码重构 / 框架升级 | 新旧代码跑出逐位相同的 loss，才能确认重构没有改变语义 |
| 断点续训验证 | 「连续跑 2N 步」和「跑 N 步、存、读、再跑 N 步」结果相同 |
| 并行配置迁移 | TP=8 换 TP=4，loss 曲线应该一致（逐位相同很难，统计一致是底线） |
| RL 后训练 | 训练侧和 rollout 侧对同一序列算出的 logprob 应一致（5.15、8.8 节） |

在训练不具备确定性时，任何对比实验都必须先判断观察到的差异是否只是噪声
- 调试成本因此大幅增加

以代码重构为例：
- 若训练是逐位确定的
  - 重构后在相同输入下跑 100 步，loss 与重构前逐位相同即可确认语义没有改变
  - 验证在几分钟内完成
- 若训练不确定
  - 两次运行的 loss 在第三四位有效数字上本来就会有差异
  - 要证明重构没有引入错误，就必须多次运行并做统计比较
  - 小的语义错误（例如某个 bias 项少加了一次）可能淹没在噪声中无法发现
- loss spike 调查也是如此
  - 只有 replay 能逐位复现原运行，才能断定 spike 由数据或数值引起而不是偶然现象

#### 4.10.2 确定性的分级

| 级别 | 定义 | 难度 |
|---|---|---|
| L0 统计可复现 | 同配置多次运行，loss 曲线在噪声范围内一致 | 默认即可 |
| L1 运行间逐位一致 | 同代码、同配置、同硬件、同卡数，两次运行每步 loss 逐位相同 | 中等，主要是关掉非确定性算子 |
| L2 续训逐位一致 | 从 checkpoint 恢复后与不中断的运行逐位相同 | 需要完整保存所有状态 |
| L3 跨并行配置逐位一致 | 换 TP/PP/DP 大小或卡数后仍逐位相同 | 很难，需要和切分方式无关的归约顺序 |

- 工程上以 L1 + L2 为目标较为合理
- L3 一般只在以下场景中追求
  - 推理场景的 batch invariance，即同一请求的输出不随它所在 batch 的大小和组成而变化
  - RL 的训练与推理结果对齐

#### 4.10.3 非确定性的来源

非确定性的根源是 1.5.6 节所述的浮点加法不满足结合律：
- 只要求和顺序改变，结果的低位就可能不同

具体来源如下：

| 来源 | 机制 | 影响级别 |
|---|---|---|
| 随机数种子 | Python / NumPy / PyTorch CPU / CUDA 各自有生成器；每个 rank 的种子 | L1 |
| 数据顺序 | shuffle 种子；DataLoader worker 数量和调度；多数据源混合采样 | L1、L2 |
| atomic 加法 | `index_add_`、`scatter_add_`、embedding 反向、某些 attention 反向用 atomicAdd，线程到达顺序不定 | L1 |
| cuBLAS / cuDNN 算法选择 | `cudnn.benchmark=True` 时按测速选算法；cuBLAS 的 split-K 和 workspace 配置 | L1 |
| 通信归约顺序 | NCCL 按消息大小、拓扑选 ring / tree、协议、通道数；不同算法求和顺序不同 | L1（通常同配置下稳定）、L3 |
| 并行切分 | TP 大小决定行切矩阵乘的部分和怎么加；DP 大小决定梯度怎么加；梯度累积顺序 | L3 |
| kernel 配置 | Triton / Inductor autotune 按测速选 tile；split-K 选择依赖形状 | L1 |
| Dropout 在 TP 下的 RNG | 被 TP 切开的区域和被复制的区域需要不同的随机流 | L1 正确性 |
| MoE 路由 | top-k 并列时的选择；capacity 截断时谁被丢弃取决于顺序 | L1 |
| 软件 / 硬件版本 | CUDA、cuBLAS、NCCL、驱动、GPU 型号不同，kernel 不同 | 跨环境 |
| 硬件错误 | SDC | 意外 |

表中三个机制需要说明：
- atomic 加法（atomicAdd）：GPU 上多个线程对同一内存地址执行的不可分割的读取、相加、写回操作
  - 各线程完成的先后由硬件调度决定
  - 因此浮点累加的顺序每次运行都可能不同
- split-K：GEMM 的一种并行策略
  - 把归约维 $K$ 切成若干段，由不同线程块分别计算部分和后再合并
  - 段数的选择改变了求和顺序
- autotune：在运行时对同一算子的多种 kernel 配置（如 tile 大小）测速并选用最快者
  - 测速结果受噪声影响，不同运行可能选中不同配置

表中来源的对应措施：
- 表中的 L1 来源大多可以通过设置消除，相应措施见 4.10.4
- L3 来源则源于并行方式本身，见 4.10.6

MoE 路由一行的说明：
- 若两个专家的路由得分在 BF16 精度下恰好相等
  - top-k 选择哪一个取决于排序实现
- 若某个专家接收的 token 数超过容量上限（capacity）
  - 超出部分被丢弃
  - 丢弃哪些 token 取决于它们在 batch 中的顺序
- 这两种情况下，路由前的微小数值差异会被放大成离散的选择差异
  - 使后续计算完全不同

低精度训练中的随机舍入是另一个需要纳入确定性管理的随机源：
- FP8 等低精度格式的可表示值很稀疏
  - 普通的就近舍入（round-to-nearest）在参数更新量远小于参数本身时会把更新量舍掉
  - 例如 BF16 只有 8 位有效数字（含隐含位），对一个值为 1.0 的参数，任何小于约 $0.004$ 的更新量经就近舍入后都会被吞掉，参数始终不变
- 随机舍入（stochastic rounding）：按待舍入值与相邻两个可表示值的距离，以相应概率向上或向下舍入
  - 若 $x$ 位于可表示值 $a$ 与 $b$ 之间，则以概率 $(b-x)/(b-a)$ 舍入为 $a$、以概率 $(x-a)/(b-a)$ 舍入为 $b$
  - 这样舍入结果的期望等于 $x$ 本身
  - 多次小更新累积后在统计上不会丢失，从而减小就近舍入的系统性偏差
  - 但这并不能单独保证训练收敛
- 随机舍入需要随机数
  - 因此在追求 L1/L2 确定性时，舍入所用的随机数生成器同样要固定种子，并在 checkpoint 中保存其状态

低精度训练还需要区分权重、激活、梯度和累加器各自的精度：
- 典型做法是 GEMM 的输入用 FP8，累加器用 FP32，梯度与优化器状态用 BF16 或 FP32
- MXFP8、MXFP4 等 microscaling 格式让每 32 个元素共享一个尺度因子
  - 在每个小块内充分利用有限的动态范围
  - 代价是需要额外存储和处理尺度因子
- 这些尺度因子的计算方式（如取块内最大值）是确定的
  - 但若采用延迟缩放（根据前若干步的统计量确定当前步的尺度），尺度历史也属于续训必须保存的状态

#### 4.10.4 L1：运行间逐位一致

##### 4.10.4.1 随机数种子的设置

```python
import os, random, numpy as np, torch

def seed_everything(seed: int, rank: int = 0):
    s = seed + rank                      # 每个 rank 不同但确定（数据增强等需要不同随机流时）
    random.seed(s); np.random.seed(s)
    torch.manual_seed(seed)              # 模型初始化用全局相同的种子，保证各 rank 初始权重一致
    torch.cuda.manual_seed_all(seed)
```

- 模型初始化在各 rank 上必须一致
  - 或者只在 rank 0 上初始化后广播（broadcast）到其他 rank
- 数据增强、dropout 等则按需要为各 rank 设置不同的随机流

##### 4.10.4.2 PyTorch 确定性模式

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # cuBLAS 确定性需要固定 workspace，必须在 CUDA 初始化前设置
torch.use_deterministic_algorithms(True)            # 遇到没有确定性实现的算子会直接报错
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False       # 可选：排除 TF32 带来的差异
```

`use_deterministic_algorithms(True)` 的作用：
- 遇到没有确定性实现的算子时直接报错，而不是静默地产生非确定性结果
- 报错的算子就是需要替换实现的位置
- 常见的报错算子包括 `index_add_`、`scatter_add_` 的 CUDA 反向、`nll_loss2d` 反向以及部分插值算子的反向

##### 4.10.4.3 固定集合通信算法与协议

```bash
export NCCL_ALGO=Ring          # 不让 NCCL 根据消息大小在 ring / tree 之间切换
export NCCL_PROTO=Simple
# Transformer Engine：禁止非确定性的 attention 反向实现
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
```

- Megatron-LM 提供 `--deterministic-mode` 参数
  - 启用后会检查上述环境变量是否已设置，并选用确定性 kernel
- 在同一配置下，NCCL 的算法选择通常本身是稳定的
  - 显式固定可以排除这一变量
  - 也能防止规模变化时 NCCL 自动切换算法

##### 4.10.4.4 张量并行下的 dropout 随机流

dropout mask 在 TP 区域内外的要求：
- 在 TP 切分区域内，每块 GPU 持有张量的不同切片
  - 各自的 dropout mask 应当不同
- 在 TP 区域外（未开启 SP 时，激活在 TP 组内各 GPU 上是复制的）
  - 各 GPU 的 mask 必须相同
  - 否则逻辑上同一个张量在各 GPU 上的值会不一致
- Megatron 用 `CudaRNGStatesTracker` 维护两条随机流：

```python
# Megatron 的实现思路（简化）
tracker = get_cuda_rng_tracker()
tracker.add("model-parallel-rng", seed + 2718 + tp_rank)   # 每个 TP rank 不同
# 默认流：所有 TP rank 相同（seed 只和 PP/DP 有关）

with tracker.fork():          # 进入 TP 切分区域，用 model-parallel 流
    attn_probs = dropout(attn_probs)
hidden = dropout(hidden)      # 复制区域，用默认流，各 TP rank 一致
```

激活重算时的 RNG：
- 激活重算时也必须恢复与第一次前向相同的 RNG 状态
  - 否则重算得到的 dropout mask 与第一次前向不同，梯度计算会出错
- `torch.utils.checkpoint` 默认设置 `preserve_rng_state=True`
  - 目的正是保存并恢复这一状态

##### 4.10.4.5 消除 autotune 引入的非确定性

- `torch.compile` 的 `max-autotune` 模式按测速结果选择 kernel，不同运行可能选中不同配置
  - 确定性运行时应使用默认模式，或缓存 autotune 结果并在后续运行中复用
- Triton kernel 的 `@triton.autotune` 同理，应固定配置或缓存结果

#### 4.10.5 L2：续训逐位一致

续训时需要恢复的状态不仅是权重和优化器状态：

| 状态 | 容易漏的地方 |
|---|---|
| 模型、优化器 | FP32 master weight 要存，不能只存 BF16 |
| step、LR scheduler | warmup / cosine 的当前位置 |
| 数据位置 | consumed samples；DataLoader 的预取缓冲里已经取出但没用的 batch |
| 各 rank 的 RNG 状态 | Python、NumPy、torch CPU、每块 GPU 的 CUDA 生成器，以及 TP 的 RNG tracker |
| loss scaler | FP16 时的 scale 和 good_steps |
| MoE 均衡偏置 | DeepSeek-V3 式的 bias 是非梯度状态，要单独存 |
| EMA 权重、梯度累积缓冲 | 如有 |

表中两个概念的说明：
- MoE 均衡偏置：DeepSeek-V3 为实现无辅助损失的负载均衡，在路由打分上加的一组偏置
  - 它按专家负载直接调整，不经过梯度更新
  - 因此不在优化器状态中
- EMA 权重：参数的指数滑动平均

数据位置一行中预取缓冲的问题较为隐蔽：
- DataLoader 通常会提前准备若干个 batch
  - 保存 checkpoint 时，迭代器可能已经从数据集中取出了第 $k+1$ 到 $k+4$ 个 batch 放入缓冲区，而模型只消耗到第 $k$ 个
- 若以迭代器的内部位置作为数据位置保存
  - 续训时会跳过这几个 batch
- 若以 consumed samples 保存并按 4.3.3 的方式重新计算索引
  - 则不受缓冲区的影响

FP32 master weight 一行同理：
- 若只保存 BF16 权重
  - 续训时从 BF16 恢复的 FP32 副本丢失了低 16 位尾数
  - 后续的累积更新会与不中断的运行不同

RNG 状态的保存与恢复：

```python
def rng_state_dict():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        "tp_tracker": get_cuda_rng_tracker().get_states() if HAS_TP else None,
    }

def load_rng_state(sd):
    random.setstate(sd["python"]); np.random.set_state(sd["numpy"])
    torch.set_rng_state(sd["torch"]); torch.cuda.set_rng_state_all(sd["cuda"])
    if sd["tp_tracker"] is not None: get_cuda_rng_tracker().set_states(sd["tp_tracker"])
```

- 各 rank 的 RNG 状态互不相同，必须按 rank 分别保存

验证方法：

```text
运行 A：从 step 0 连续跑到 step 2N，记录每步 loss
运行 B：从 step 0 跑到 N，存 checkpoint，进程退出；重新启动，加载，跑到 2N
断言：A 和 B 在 [N, 2N] 每一步的 loss 逐位相同，最终参数哈希相同
```

- 应将这一测试加入 CI（使用小模型，运行几十步）
  - 每次修改框架代码后自动执行

#### 4.10.6 L3：跨并行配置逐位一致

不一致的来源：
- TP 大小改变后，行切矩阵乘的部分和按不同的分组相加
  - 以 8 个部分和为例，TP=2 时计算 $(a_1+a_2+a_3+a_4) + (a_5+\dots+a_8)$，TP=4 时计算 $(a_1+a_2)+(a_3+a_4)+\dots$
  - 由于浮点加法不满足结合律，两者的结果一般不同

实现 L3 的思路是使归约顺序与切分方式无关：

1. 固定一棵全局归约树
   - 例如始终沿 $K$ 维切成固定的 $c$ 块（$c$ 取所有可能 TP 大小的公倍数）
   - 每块内部的累加顺序固定，块之间按固定的二叉树合并
   - TP=2 时每块 GPU 负责 $c/2$ 块，在本地按树合并到某一层，再通过通信完成上层的合并
   - *Deterministic Inference across Tensor Parallel Sizes*（2025）采用的正是这一思路，用树形归约使不同 TP 大小得到逐位相同的结果
2. DP 维度同理
   - 梯度 AllReduce 采用固定的求和顺序（例如先收集到固定拓扑，再按 rank 顺序相加）
   - 代价是失去 ring 算法的带宽最优性
3. 代价：性能下降，并且需要自行实现通信和 GEMM 的归约逻辑

因此训练中一般不要求 L3 级别的逐位一致，而是用统计一致性来验证：
- 在相同的数据和初始化下，改变并行配置后 loss 曲线的差异应处于多次 L0 运行的噪声范围之内

#### 4.10.7 确定性的性能代价

| 措施 | 性能影响 |
|---|---|
| 固定种子、数据顺序 | 无 |
| `use_deterministic_algorithms` | 取决于模型，atomic 改为排序后归约的算子会变慢，通常 0–10% |
| 确定性 attention 反向 | FlashAttention 的确定性反向会更慢（不能用 atomic 累加 dK、dV） |
| 固定 NCCL 算法 | 小消息上 ring 比 tree 慢，大规模下可能明显 |
| 关闭 autotune | 失去部分 kernel 性能 |

各项代价的来源如下：
- atomic 加法改为确定性实现
  - 通常要先对写入位置排序，再按固定顺序归约，多出一次排序和额外的显存读写
  - embedding 反向在词表很大时这一开销较明显
- FlashAttention 的反向
  - 多个线程块并行处理不同的 query 块时，需要把各自对 dK、dV 的贡献累加到同一位置
  - 非确定性版本用 atomic 加法完成
  - 确定性版本则需要按固定顺序依次累加或额外分配缓冲区再归约，速度会下降
- 固定 NCCL 使用 ring 算法
  - 小消息的延迟随 rank 数线性增长，而 tree 算法为对数增长
  - 在数千个 rank 的规模上，梯度桶较小的 AllReduce 可能慢数倍

常见的做法：
- 正式训练使用高性能模式，但保证 L2 级别的完整状态保存
- 调试时切换到确定性模式，在相同数据上进行 replay
- 这样正式训练不承担确定性的性能代价
  - 而一旦出现异常，仍然可以从完整保存的状态出发，在确定性模式下复现和定位问题

需要注意：
- 高性能模式下的原运行本身不是逐位确定的
  - 因此 replay 的结果与原运行在低位上可能不同
- 判断异常是否复现时，看的是 spike 或 NaN 等现象是否在同一步出现
- replay 之间的比较（如在不同节点上分别 replay）才要求逐位相同

#### 4.10.8 Replay：异常步骤的重放与定位

##### 4.10.8.1 常规运行中的日志记录

- replay 的前提是异常发生之前已经保存了足够的信息
- 训练时每步写入一份轻量日志：

| 字段 | 用途 |
|---|---|
| step、各 rank 的样本 id（或样本哈希） | 确定这一步使用了哪些数据 |
| loss、grad norm、各层 grad / 激活范数 | 判断 spike 起点和层 |
| lr、loss scale | 排除调度问题 |
| 每 rank 的步耗时、所在节点 | 关联硬件 |
| 参数哈希（每 K 步，DP 组内比对） | 发现 SDC |

- 此外还需要足够密集的 checkpoint（见 4.2.3 节）

##### 4.10.8.2 replay 的判定流程

```text
1. 找到异常 step S（例如 loss spike、NaN、grad norm 突增）
2. 取 S 之前最近的 checkpoint C（step = S0）
3. 用确定性模式，从 C 加载，喂入完全相同的数据（按日志里的样本 id），重跑 S0 → S
4. 判断：
   a. 异常复现，且逐位相同 → 问题在数据或代码/数值，与硬件无关
        → 二分：只喂 step S 的 batch / 去掉某些样本 / 某层用 FP32，定位到具体样本或算子
   b. 异常不复现 → 怀疑硬件或非确定性
        → 在原节点和其他节点上分别 replay，比较结果；
          只在某个节点上复现 → 隔离该节点（SDC、坏 HBM）
   c. 结果不同但都正常 → 原运行存在非确定性，需先实现 L1 再继续排查
```

##### 4.10.8.3 replay 脚本示例

以下脚本的行为：
- 从 checkpoint 加载状态，按日志中的样本 id 重新执行指定范围的 step
- 与原运行的 loss 逐位比较
  - 比较使用 `==` 而非 `isclose`，因为目标是检验逐位一致

```python
import hashlib, torch

def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]

def replay(ckpt_path, step_log, start, end, deterministic=True):
    if deterministic:
        enable_determinism()                         # 4.10.4 的全部设置
    model, opt, sched, extra = load_checkpoint(ckpt_path)
    load_rng_state(extra["rng"][rank])
    records = []
    for step in range(start, end):
        ids = step_log[step]["sample_ids"][dp_rank]  # 原运行在这一步使用的样本
        batch = dataset.fetch(ids)
        loss = model(batch).float()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec = {"step": step, "loss": loss.item(), "gnorm": gnorm.item(),
               "param_hash": tensor_hash(flat_params(model)) if step % 10 == 0 else None}
        orig = step_log[step]
        rec["loss_match"] = rec["loss"] == orig["loss"]          # 逐位比较，不用 isclose
        records.append(rec)
        if not rec["loss_match"]:
            print(f"diverge at step {step}: replay {rec['loss']!r} vs orig {orig['loss']!r}")
    return records
```

##### 4.10.8.4 逐层哈希比对：定位首个不一致的算子

两次运行结果不同时，可以用前向 hook 记录每层输出的哈希值，找出第一个出现差异的层：

```python
def register_hash_hooks(model, store):
    for name, mod in model.named_modules():
        mod.register_forward_hook(
            lambda m, i, o, name=name: store.append((name, tensor_hash(o))) if torch.is_tensor(o) else None)

run_a, run_b = [], []
# 分别在两次运行里注册 hook 并跑同一个 batch，然后：
first = next(((a, b) for a, b in zip(run_a, run_b) if a[1] != b[1]), None)
print("first mismatch:", first)
```

- 反向传播的比对方法相同
  - 可使用 `register_full_backward_hook`，或比较各参数梯度的哈希值

##### 4.10.8.5 案例：loss spike 的 replay 分析

某 70B 模型训练到 step 182,340 时，loss 从 1.82 跳升到 3.1，此后约两百步内逐渐回落。分析过程如下：

1. 从 step 182,000 的 checkpoint 以确定性模式 replay
   - spike 在 step 182,340 准确复现，loss 逐位相同
   - 结论：与硬件无关
2. 仅将 step 182,340 的 batch 替换为下一步的 batch，spike 消失
   - 再将 step 182,340 的 batch 拆成 512 条序列逐一排除
   - 定位到 3 条由同一网页模板生成、包含数万个重复 token 的样本
3. 开启逐层激活范数日志
   - 发现第 61 层注意力 logits 的最大值从约 30 增长到约 400
4. 处理措施
   - 数据侧增加重复 n-gram 过滤
   - 模型侧的 QK-norm 留待下一代模型采用
   - 本次训练回退到 step 182,000，跳过这些样本后继续

- 若第 1 步中 spike 未能复现，下一步应在其他节点上 replay，排查 SDC

#### 4.10.9 训练确定性检查清单

- [ ] 所有随机源有种子，种子写进 manifest
- [ ] 数据索引只依赖 (seed, step, rank)
- [ ] `use_deterministic_algorithms(True)` 下能跑通（调试模式）
- [ ] `CUBLAS_WORKSPACE_CONFIG`、`NCCL_ALGO`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO` 已设置（调试模式）
- [ ] TP 区域内外的 dropout 随机流区分正确；重算时恢复 RNG
- [ ] checkpoint 包含 RNG（每 rank）、数据位置、scheduler、scaler、非梯度状态
- [ ] CI 里有「连续 vs 中断续训」逐位比对测试
- [ ] 每步日志记录样本 id、loss、grad norm；定期比对 DP 组内参数哈希

### 4.11 小结

本章讨论了大规模训练系统在容错、效率和可复现性三方面的设计。

容错方面：
- 作业 MTBF 与 GPU 数量成反比
  - 万卡作业的 MTBF 只有数小时
  - 按独立故障模型计算，一天内至少发生一次故障的概率接近 1
  - 电源、交换机等相关故障还会使独立模型低估同时失去大量节点的风险
- 最优 checkpoint 间隔为 $\sqrt{2\delta M}$
  - 降低 checkpoint 开销 $\delta$（异步、分片、分层保存）和重启时间 $R$（热备节点、自动化恢复）共同决定有效训练时间
- 评价一个系统的容错能力，应看 goodput、故障恢复时间、P99 步耗时和任务成功率
  - 而不只是健康状态下的平均吞吐

效率方面：
- MFU 按 $6N + 12Lhs$ 计算模型必需的计算量，是比较不同系统效率的标准指标
  - HFU 计入重算，数值会被抬高，不宜用于跨系统比较
- MFU 的损失来自 GEMM 本身、访存受限的非 GEMM 算子、未重叠的通信、PP 气泡和 straggler
- straggler 使整个作业以最慢的 GPU 的速度运行，且不会报错
  - 需要通过按 rank 拆分步耗时、通信到达统计和主动基准测试持续检测

可复现性方面：
- 确定性分为四级，L1 + L2 是合理目标
  - 代价约为 0–10% 的性能，通常在调试时开启
- 处理 loss spike 时先进行 replay
  - 能够复现说明问题在数据或数值，不能复现再排查硬件
- SDC 无法被 ECC 消除
  - 需要借助 DP 组内的参数哈希比对、抽样冗余计算和 replay 发现
  - 再依靠 checkpoint 回退恢复

<br>

## 5. 推理系统：调度、KV cache 管理与解码加速

- 训练是一次性投入，推理是持续运行的开销
  - 模型上线后，累计推理成本通常很快超过训练成本
- 本章从 prefill 与 decode 两个阶段的负载差异出发，依次讨论：
  - 请求调度
  - KV cache 管理与分层存储
  - 量化
  - 投机解码
  - PD 分离部署
  - 长上下文 serving
  - 容量规划
- 然后讨论推理结果的确定性问题和 Agent 工作负载的服务方式

### 5.1 推理负载的两个阶段：prefill 与 decode

#### 5.1.1 阶段定义与特征对比

自回归生成分为两个阶段：
- Prefill 阶段对输入 prompt 的全部 token 做一次前向计算
  - 生成第一个输出 token
  - 把每层的 Key、Value 写入 KV cache
  - KV cache 是缓存已处理 token 的注意力 Key/Value 张量的显存区域，见 1.7.4 节
- Decode 阶段每次前向只为每个请求生成一个新 token
  - 计算时需要读取该请求此前全部 token 的 KV cache

衡量两阶段延迟的指标分别是 TTFT 和 TPOT：
- TTFT（Time To First Token）：从请求到达到输出第一个 token 的时间
- TPOT（Time Per Output Token）：生成第一个 token 之后，平均每个输出 token 所需的时间

| | Prefill | Decode |
|---|---|---|
| 计算内容 | 一次处理整个 prompt，生成第一个 token，写入 KV cache | 每步生成一个 token，读取全部 KV cache |
| 每步 token 数 | prompt 长度 $s$（几百到几十万） | 每个请求 1 个 |
| 算术强度 | 高，约等于 $s$ | 低，约等于 batch size |
| 瓶颈 | 算力 | 显存带宽 |
| 决定的指标 | TTFT（首 token 延迟） | TPOT（每 token 延迟） |

- 算术强度：每读写 1 字节数据所执行的浮点运算次数
- 按 roofline 模型，以脊点为界
  - 脊点是峰值 FLOPS 与显存带宽之比，见 1.3.3 节
  - 算术强度高于脊点时，kernel 受算力限制
  - 算术强度低于脊点时，kernel 受带宽限制
- Prefill 的算术强度约为 $s$
  - 每个权重元素参与 $s$ 个 token 的计算
- Decode 的算术强度约为 batch size
  - 每个权重元素只参与 batch 内各请求的 1 个 token

#### 5.1.2 Prefill 耗时估算

- Prefill 的计算量约为 $2N\cdot s$ FLOPs
  - $N$ 是模型参数量，$s$ 是 prompt 长度
- 设 $N_{\text{GPU}}$ 为 GPU 数，$P$ 为单块 GPU 的峰值算力，$\eta$ 为实际达到的算力利用率，则

$$T_{\text{prefill}} \approx \frac{2N\cdot s}{N_{\text{GPU}}\cdot P\cdot \eta}$$

以 70B 模型、8 块 H100、$s=2000$、$\eta=50\%$ 为例：

$$\frac{2\times70\times10^9\times2000}{8\times989\times10^{12}\times0.5}\approx 71\ \text{ms}$$

#### 5.1.3 Decode 耗时下限

- Decode 每一步至少要把全部权重和本 batch 所有请求的 KV cache 从 HBM 读取一遍
  - HBM 是 GPU 上的高带宽显存

$$T_{\text{decode}} \ge \frac{N\cdot p_w + \sum_{i\in\text{batch}} s_i\cdot k_{\text{tok}}}{N_{\text{GPU}}\cdot B_{\text{mem}}}$$

- 公式中各符号：
  - $p_w$ 是每个权重的字节数
  - $s_i$ 是第 $i$ 个请求当前的序列长度
  - $k_{\text{tok}}$ 是每个 token 的 KV 字节数
  - $B_{\text{mem}}$ 是单块 GPU 的显存带宽
- 算例：70B BF16 模型、TP=8、batch=1
  - TP=8 指张量并行度为 8，权重均分到 8 块 GPU
  - 下限为 $140\text{ GB}/26.8\text{ TB/s}\approx 5.2$ ms，即单请求最多约 190 token/s
- 实际系统一般能达到该下限对应速度的 60–80%
- 差距来自：
  - kernel 启动开销
  - 注意力 kernel 对不规则 KV 布局的访存效率
  - TP 通信
    - 每层 AllReduce，即各 GPU 把部分结果求和并广播回所有 GPU
  - 采样等 CPU 侧开销

公式中的 KV 项需要单独估算。KV cache 的总字节数为

$$M_{KV}=2\times B\times L\times n_{\text{layers}}\times n_{\text{kv\_heads}}\times d_{\text{head}}\times s_{\text{elem}}$$

- 公式中各符号：
  - 系数 2 对应 Key 和 Value 两个张量
  - $B$ 是并发序列数
  - $L$ 是每条序列缓存的 token 数
  - $n_{\text{layers}}$ 是层数
  - $n_{\text{kv\_heads}}$ 是 KV 头数
  - $d_{\text{head}}$ 是每个头的维度
  - $s_{\text{elem}}$ 是每个元素的字节数
- 去掉 $B$ 和 $L$ 后的部分就是每 token 的 KV 字节数 $k_{\text{tok}}$

以一个 32 层、8 个 KV 头、头维度 128 的模型为例，BF16（$s_{\text{elem}}=2$）下 10 条并发序列各缓存 4096 个 token：

$$M_{KV}=2\times10\times4096\times32\times8\times128\times2\approx 5.37\times10^9\ \text{bytes}\approx 5.0\ \text{GiB}$$

- 同样的公式用于 Llama-3-70B（80 层、8 个 KV 头、头维度 128）
  - 每 token 的 KV 为 $2\times80\times8\times128\times2 = 327{,}680$ 字节，即本章反复使用的 320 KB/token
  - 改用 FP8 存储时为 160 KB/token
- 公式也说明了 GQA/MQA 压缩 KV 的方式
  - 它们减少的是 $n_{\text{kv\_heads}}$
  - KV 总量与该项成正比

#### 5.1.4 Batch size 与单 token 成本

- Batch 从 1 增大到 $b$ 时，每步读取的权重字节数不变，产出的 token 数变为 $b$ 倍
  - 在进入算力受限区之前，吞吐随 $b$ 近似线性增长
  - 单 token 成本近似线性下降
- 令每步计算时间等于读权重时间，可得交叉点：

$$\frac{2Nb}{P} = \frac{N p_w}{B}\quad\Rightarrow\quad b^* = \frac{P\,p_w}{2B} = \frac{I^*\,p_w}{2}$$

- 其中 $I^* = P/B$ 是脊点
  - BF16（$p_w=2$）时 $b^*\approx I^*\approx 295$
- 实际中交叉点会进一步后移，而且显存容量通常先达到上限
  - KV cache 的读取量也随 batch 增长
- 因此 decode 基本始终处于带宽受限区
  - 优化目标是在显存容量和延迟约束允许的范围内尽量增大 batch

增大 batch 也有代价：
- 第一，每步读取的 KV 随 batch 线性增长，每步耗时随之增加，TPOT 变差
  - 以 5.13.5 节的算例计，batch 从 128 增至 512 时每步耗时从约 6 ms 增至约 14 ms
- 第二，若调度器为凑满 batch 而让请求排队等待，等待时间直接计入 TTFT
- 第三，同一 batch 内请求的输入、输出长度差异越大，越容易出现部分槽位空转
- 5.2 节的 continuous batching 和 5.10 节的 chunked prefill 分别针对后两个问题

### 5.2 Continuous batching 与迭代级调度

#### 5.2.1 静态 batching 的局限

- 静态 batching 指一批请求同时开始、同时结束，批内成员在整批完成前不变
- 请求的生成长度差异很大（例如 10 token 与 2000 token）
  - 短请求结束后，其槽位一直空闲，直到批内最长的请求完成
  - 新到达的请求也必须等整批结束后才能加入

以 batch 为 8 为例：
- 若 7 个请求各生成 50 token，1 个请求生成 2000 token
  - 前 50 步之后只有 1 个槽位在工作
  - 剩余 1950 步的平均槽位利用率只有 1/8
- 在此期间到达的新请求全部排队
  - 其 TTFT 包含了这 1950 步的等待时间

#### 5.2.2 迭代级调度（Orca, OSDI 2022）

- Continuous batching（连续批处理）：批内成员在每次前向迭代之间动态变化的批处理方式
- 其实现基础是迭代级调度（iteration-level scheduling）
  - 调度粒度从「请求」缩小到「一次前向迭代」

```text
每次迭代：
  1. 移除已完成（遇到 EOS 或达到 max_tokens）的请求，释放 KV
  2. 在显存允许时，从等待队列里加入新请求
  3. 对当前所有活跃请求跑一次前向（新请求做 prefill，旧请求做 decode）
```

- 请求完成后立即退出，新请求在下一次迭代即可加入
  - GPU 槽位几乎始终处于占用状态
- Orca 论文报告，在相同延迟下，其吞吐比当时的 FasterTransformer 高一个数量级以上

#### 5.2.3 变长序列的张量组织

- 不同请求的序列长度不同，无法组成规整的 $[b, s]$ 张量
  - 如果用 padding 补齐到最长序列，补齐部分的计算全部是浪费
  - 同一次迭代中既有 prefill 请求（每个几千 token）又有 decode 请求（每个 1 token），补齐的浪费会非常大

通行做法是把本次迭代所有请求的 token 拼接成一维序列：
- 张量形状为 $[T_{\text{total}}, h]$
  - $T_{\text{total}}$ 是所有请求本步 token 数之和，$h$ 是隐藏维度
  - 例如 100 个 decode 请求加 1 个 1948 token 的 prefill 块，$T_{\text{total}}=2048$
- 线性层和 MLP 对每个 token 独立计算，不需要知道 token 属于哪个请求
  - 因此可以直接对这个 $[2048, h]$ 矩阵做一次矩阵乘法，权重只读一次

只有注意力需要区分请求：
- 每个 token 只能关注同一请求中位于它之前的 token
- 这一信息由两种方式提供：
  - 变长注意力 kernel 接收一个前缀和数组 `cu_seqlens`
    - 第 $i$ 个元素记录第 $i$ 条序列在拼接序列中的起始位置
  - PagedAttention 通过每个请求的 block table 找到该请求的历史 KV 所在的物理块（5.3 节）

### 5.3 PagedAttention 与 KV cache 管理

#### 5.3.1 KV cache 的显存碎片

- 早期系统按 `max_seq_len` 为每个请求预留连续的 KV 空间
  - 实际生成长度事先未知，预留空间的大部分未被使用
- vLLM 论文测得，已有系统中 KV 显存的有效利用率只有 20–40%

| 浪费类型 | 原因 |
|---|---|
| 预留浪费 | 按最大长度预留，实际未使用 |
| 内部碎片 | 已分配区域中请求结束时仍未写入的部分 |
| 外部碎片 | 大小不一的连续分配区之间无法利用的空隙 |

- 预留浪费是其中最大的一项
  - 假设 `max_seq_len` 为 4096，而某请求的 prompt 加输出共 600 token
  - 按 Llama-3-70B 的 320 KB/token 计，系统为它预留约 1.3 GB，实际只用约 190 MB
  - 其余 85% 在请求存续期间无法分配给其他请求
- 外部碎片的成因：
  - 请求在不同时间结束，释放出的连续空间大小不一
  - 新请求需要的连续空间可能比任何一段空隙都大
  - 即便空闲总量足够也无法分配

#### 5.3.2 分页式 KV 管理

- PagedAttention 借鉴操作系统的虚拟内存分页机制管理 KV cache
- 它把 KV 显存池划分为固定大小的 block
  - 每个 block 存放固定数量（vLLM 默认 16 个）token 在所有层、所有 KV 头上的 Key 和 Value
  - 按 Llama-3-70B 的 320 KB/token 计算，一个 block 约 5 MB
- 每个请求维护一张 block table，记录逻辑块号到物理块号的映射
  - 逻辑块 $i$ 存放该请求第 $16i$ 到 $16i+15$ 个 token 的 KV
  - 对应的物理块可以位于池中任意位置
- 注意力 kernel 计算时先查 block table，再按物理块号读取 KV
  - 因此物理块不需要连续

分配是按需进行的：
- prefill 时按 prompt 长度一次分配所需的块数
- decode 时每写满 16 个 token 再追加一个新块
- 外部碎片不再存在
  - 所有块大小相同，任何空闲块都能满足任何请求
- 预留浪费也消失了
  - 不再按最大长度预留
- 剩下的浪费只出现在每个请求最后一个未写满的块中
  - 平均不到半个块，即约 8 个 token
  - vLLM 论文报告浪费降至 4% 以内

- PagedAttention 本身是一种显存分配与注意力访存机制
  - 并不包含把 KV 迁移到 CPU 内存或磁盘的功能
  - 分层存储与卸载在 5.4.5 节以后讨论

```text
请求 A 的逻辑 KV:  [块0][块1][块2]          block table A: 0→7, 1→2, 2→9
请求 B 的逻辑 KV:  [块0][块1]               block table B: 0→3, 1→5
物理 KV 池:  0  1  2(A1) 3(B0) 4  5(B1) 6  7(A0) 8  9(A2) ...
```

#### 5.3.3 块共享与写时复制

- Parallel sampling（对同一 prompt 生成 $n$ 个回答）和 beam search 中，多个序列共享 prompt 部分的 KV 块
  - 这些块的引用计数大于 1
- 写时复制（copy-on-write）：共享块在被某个序列写入新 token 时才复制出一份私有副本
  - 未写入前各序列只持有引用

以 2000 token 的 prompt、$n=4$ 的 parallel sampling 为例：
- prompt 占 125 个块
  - 不共享时 4 个序列需要 500 个块
  - 共享后只需 125 个块，加上各序列自己生成部分的块
- 只有 prompt 的最后一个块可能未写满
  - 4 个序列在 decode 第一步都会向它写入
  - 此时触发写时复制，各自得到一个私有副本
- 其余 124 个块始终是只读共享的

#### 5.3.4 Block manager 代码骨架

```python
class BlockManager:
    def __init__(self, num_blocks, block_size=16):
        self.block_size = block_size
        self.free = list(range(num_blocks))
        self.ref = [0] * num_blocks
        self.tables = {}                      # seq_id -> [physical block ids]

    def can_allocate(self, n_tokens):
        return len(self.free) >= -(-n_tokens // self.block_size)

    def allocate(self, seq_id, n_tokens):     # prefill
        n = -(-n_tokens // self.block_size)
        blocks = [self.free.pop() for _ in range(n)]
        for b in blocks: self.ref[b] = 1
        self.tables[seq_id] = blocks

    def append_slot(self, seq_id, seq_len):   # decode：需要时追加一个块
        table = self.tables[seq_id]
        if seq_len % self.block_size == 1 or not table:   # 上一块刚好写满
            if not self.free: raise MemoryError("preempt")  # 交给调度器抢占
            b = self.free.pop(); self.ref[b] = 1; table.append(b)
        last = table[-1]
        if self.ref[last] > 1:                # 写时复制
            new = self.free.pop(); self.ref[new] = 1; self.ref[last] -= 1
            table[-1] = new
            return ("copy", last, new)
        return None

    def fork(self, parent, child):            # parallel sampling 共享 prompt
        self.tables[child] = list(self.tables[parent])
        for b in self.tables[child]: self.ref[b] += 1

    def free_seq(self, seq_id):
        for b in self.tables.pop(seq_id):
            self.ref[b] -= 1
            if self.ref[b] == 0: self.free.append(b)
```

#### 5.3.5 显存不足时的抢占

- 抢占（preemption）：KV 块耗尽时，调度器暂停部分正在运行的请求并收回其 KV 块，待显存释放后再恢复这些请求
- 恢复方式有两种：

| 方式 | 做法 | 代价 |
|---|---|---|
| Swap | 将被抢占请求的 KV 块换出到 CPU 内存，恢复时换入 | PCIe 传输 |
| Recompute | 直接丢弃 KV，恢复时重新执行 prefill | 重复计算 |

- vLLM 默认采用 recompute
  - prefill 速度快，经 PCIe 换出再换入的耗时通常更长
- 以 Llama-3-70B 的 2000 token 请求为例，KV 约 640 MB
  - TP=8 时每块 GPU 持有 80 MB，单向 PCIe 传输只需几毫秒
  - 但换出和换入要与正在进行的计算争用 PCIe 和 HBM 带宽
  - 还要求 CPU 侧预留锁页内存
  - 重算 2000 token 的 prefill 约 71 ms（5.1.2 节），并且可以和其他请求合并进同一次迭代
  - 两者的一般性比较见 5.4.7 节
- 被抢占的请求通常选择最晚到达的请求
  - 以保证先到的请求优先完成
- 抢占发生得越频繁，重复计算越多，有效吞吐越低
  - `--max-num-seqs` 过大或 `--gpu-memory-utilization` 过小都会使抢占增多（5.14 节）

### 5.4 前缀缓存

#### 5.4.1 动机

- 前缀缓存（prefix caching）：保留已计算过的 token 序列前缀的 KV cache，供后续具有相同前缀的请求直接复用，跳过这部分 prefill 计算
- 大量请求共享相同前缀，例如：
  - 系统提示词
  - few-shot 示例
  - 多轮对话历史
  - Agent 的工具说明
  - 针对同一文档的多个提问
- 这部分 KV 只需计算一次

#### 5.4.2 两种实现

- 基数树（radix tree）：一种压缩前缀树
  - 每条边上存放一段 token 序列
  - 共享前缀的序列共享从根出发的同一条路径

| | vLLM Automatic Prefix Caching | SGLang RadixAttention |
|---|---|---|
| 数据结构 | 每个 KV 块按「前缀全部 token + 本块 token」计算哈希，通过哈希表查找 | 基数树，边上是 token 序列 |
| 匹配粒度 | 块（16 token） | token |
| 淘汰策略 | 引用计数为 0 的块按 LRU 淘汰 | 叶子节点按 LRU 淘汰 |

块哈希按如下方式递归定义，保证只有前缀完全相同时才能命中：

$$h_i = \text{hash}(h_{i-1},\ \text{tokens}_{[i\cdot B,\,(i+1)\cdot B)})$$

其中 $B$ 为块大小（token 数）。

- 复用的前提是 token 级完全一致，而不是语义相近
  - 第 $i$ 层第 $t$ 个位置的 KV 取决于前 $t$ 个 token 的全部内容（通过前面各层的注意力）
  - 因此两个 prompt 只要在某个位置出现一个不同的 token，从该位置起的 KV 都不同
  - 即使两段文字意思相同也不能共享
- 同理，模型权重、量化方式、位置编码配置不同时，KV 也不能共享
- 工程上的推论：
  - 系统提示词、工具说明等固定内容应放在 prompt 最前面
  - 时间戳、用户 ID 等每次变化的内容应放在后面
  - 否则一个变化的 token 会使其后所有块都无法命中

两种实现的差异主要体现在匹配粒度上：
- vLLM 以块为单位
  - 一个 2010 token 的公共前缀只能复用前 125 个完整块（2000 token），最后 10 个 token 需要重算
- SGLang 的基数树以 token 为单位匹配
  - 可以复用全部 2010 个 token
  - 但树的插入、分裂和查找开销更高

#### 5.4.3 计算量收益

- 命中率为 $r$ 时，prefill 计算量变为原来的 $(1-r)$ 倍
  - 多轮对话和 Agent 场景中 $r$ 常达到 50–90%
  - 许多 API 对命中缓存的输入 token 只收取标准价格的几分之一，依据即在于此

以多轮对话为例：
- 第 10 轮请求的 prompt 包含前 9 轮的全部问答，共 8000 token，其中本轮新增 300 token
- 如果前 9 轮的 KV 仍在缓存中，本轮只需对 300 token 做 prefill
  - 计算量约为无缓存时的 4%
  - TTFT 相应从数百毫秒降至几十毫秒
- 命中率的上限受缓存容量限制
  - 缓存的前缀越多，占用的 KV 显存越多，可用于活跃请求的 batch 就越小
  - 因此缓存容量与 batch 大小之间需要权衡

#### 5.4.4 多副本下的缓存感知路由

- 前缀缓存只在单个推理实例内有效
- 多副本部署时，需要将共享前缀的请求路由到同一副本，即 cache-aware routing
  - 常见做法是按前缀哈希做一致性哈希
  - 或根据各副本当前缓存的内容为候选副本打分
- 若用轮询等与缓存无关的方式路由
  - $k$ 个副本时同一前缀落到已有缓存副本上的概率约为 $1/k$
  - 副本数越多命中率越低
- 按前缀路由则可能造成负载不均
  - 某个热门系统提示词的请求全部集中到一个副本上
- 实际路由器通常同时考虑两项指标，在缓存命中收益和各副本的排队长度之间取折中
  - 例如当目标副本的排队请求超过阈值时，改为发往负载最低的副本，接受一次缓存未命中

#### 5.4.5 分层 KV 存储

显存容纳不下的 KV 可以逐级下沉到更大、更慢的存储。推理系统涉及的存储层级如下：

| 层级 | 位置与特性 | 在 KV 管理中的角色 |
|---|---|---|
| SRAM | GPU 片上存储（寄存器、共享内存、L1/L2 cache），容量为 MB 级，延迟最低，单位容量面积成本最高 | 注意力 kernel 计算时暂存 KV 分块，不用于长期存放 |
| HBM | 与 GPU 封装在一起的高带宽显存，单块 GPU 为 80–192 GB，带宽为 3–8 TB/s | 存放活跃请求的 KV，是最紧缺的资源 |
| CPU DRAM | 主机内存，单节点可达 1–2 TB，经 PCIe 访问，单块 GPU 方向的有效带宽为几十 GB/s | 存放暂不活跃但可能很快复用的 KV |
| 本地 SSD | 非易失存储，单节点可达数十 TB，带宽为每块盘数 GB/s，延迟与随机访问性能远逊于 DRAM | 存放复用概率较低的冷前缀 |
| 远端内存或存储 | 其他节点的 DRAM、SSD 或分布式存储，经 RDMA 或 TCP 访问，受网络带宽、延迟和拥塞影响 | 集群级共享的 KV 池 |

```text
GPU HBM（热） → CPU DRAM（温） → 本地 SSD / 分布式存储（冷）
```

带宽和延迟是两个不同的指标。一次传输的耗时可近似为

$$T_{\text{move}} \approx T_{\text{setup}} + \frac{D}{B}$$

- 其中 $T_{\text{setup}}$ 是与数据量无关的固定开销
  - 包括发起请求、查找元数据、建立 DMA 描述符、网络往返
- $D$ 是数据量，$B$ 是有效带宽
- 传输小块数据时耗时主要由 $T_{\text{setup}}$ 决定，传输大块数据时主要由 $D/B$ 决定
  - 因此 KV 迁移通常以多个 block 为单位批量进行，而不是逐个 16 token 的块单独传输

- Mooncake（Kimi 的服务架构）将集群中所有节点的 CPU 内存和 SSD 组成一个全局 KV 池
  - prefill 节点可以直接复用其他节点已计算过的前缀
- 从存储中取回 KV 比重新计算更快的条件是：

$$\frac{s\cdot k_{\text{tok}}}{B_{\text{transfer}}} < T_{\text{prefill}}(s)$$

- 以 Llama-3-70B、8K 前缀为例
  - KV 约 2.6 GB，经 8 张 RDMA 网卡（合计 400 GB/s）传输需 6.5 ms
  - 而重新计算约需 300 ms

- LMCache 是另一类实现：作为推理引擎之下的 KV 存储与复用层
  - 把引擎产生的 KV 按前缀组织后存入 CPU 内存、本地磁盘或远端存储
  - 在后续请求命中时取回并注入引擎
  - 也可以在不同推理实例之间迁移 KV
- 它与 vLLM、SGLang 的关系是存储层与执行层的关系
  - 引擎负责在 HBM 内分配块和执行注意力
  - LMCache 负责 HBM 之外的存放与检索

#### 5.4.6 卸载、逐出与预取

KV 在存储层级之间的流动包含三种不同的操作，它们回答不同的问题：

- 卸载（offloading）：把暂时不用的 KV 从 HBM 移到更低一级的存储
  - 数据仍然保留，只是换了位置
  - 它回答的是「数据放在哪里」
- 逐出（eviction）：决定哪些数据离开某一层存储
  - 逐出可以伴随卸载（移到下一层），也可以直接丢弃（以后需要时重算）
  - 它回答的是「谁离开」
  - 常用策略有 LRU、按引用计数、按预测的复用概率等
  - 5.4.2 节中引用计数为 0 的块按 LRU 淘汰就是一种逐出策略
- 预取（prefetching）：在数据被使用之前，提前把它从低层存储搬回 HBM，使传输时间与其他计算重叠
  - 它回答的是「何时回来」
  - 预取依赖对未来访问的预测
  - 预测错误时，搬回的数据占用了 HBM 却没有被使用，浪费了带宽和容量

三者组合起来构成完整的 KV 层级管理策略：
- 逐出策略选出牺牲者
- 卸载决定它去哪一层
- 预取决定它何时提前回到 HBM

#### 5.4.7 迁移与重算的决策

- 当一段 KV 不在 HBM 中而又需要使用时，有两种获得方式：
  - 从其他存储层级搬回
  - 重新执行 prefill 计算
- 决策的基本比较是

$$T_{\text{move}}\approx T_{\text{setup}}+\frac{D}{B} \quad \text{vs.}\quad T_{\text{recompute}}$$

仅比较两个时间还不够，还需要考虑以下三个因素：

- 第一，重叠能力
  - DMA（Direct Memory Access，直接内存访问）允许设备之间直接搬运数据，不需要 CPU 逐字节复制
  - GPU 的 copy engine 执行 DMA 传输时，计算单元可以同时执行 kernel
  - 若搬运可以与其他请求的计算完全重叠，它对 GPU 时间的实际占用接近零，而重算必然占用计算资源
  - RDMA（Remote DMA）把这一机制扩展到跨节点
    - 网卡直接读写远端内存，省去对端 CPU 和内核协议栈的参与
    - 但仍需处理内存注册、一致性、访问权限和传输失败后的恢复
- 第二，资源争用
  - 搬运占用 PCIe 或网络带宽，在 PD 分离部署中，这些链路同时承担 prefill 到 decode 的 KV 传输
  - 重算占用算力，在 prefill 负载高时会推迟其他请求的 TTFT
  - 哪一种资源更紧张，就应少用哪一种
- 第三，HBM 的机会成本
  - 为了随时可用而把 KV 一直保留在 HBM 中，本身也有代价
  - 这部分显存无法分配给其他请求，batch 因此变小
  - 一段 10 GB 的 KV 在 decode 节点上保留 20 s，相当于在这 20 s 内少服务了数十个并发请求

以 Llama-3-70B、32K token 前缀为例：
- KV 约 10.7 GB（BF16）
- 从 CPU DRAM 经 PCIe 搬回约需 30 ms
  - 按每块 GPU 约 50 GB/s 有效带宽、TP=8 各搬 1/8 计算
- 从 8 张网卡合计 400 GB/s 的远端存储取回约需 27 ms
- 在 8 块 H100 上重算，仅线性层部分就约需 1.2 s
  - 注意力的 $O(s^2)$ 部分还会使其更长
- 这一规模下搬运明显更优
- 反之，对于几百 token 的短前缀，重算只需几毫秒
  - 搬运的固定开销 $T_{\text{setup}}$ 和元数据查找可能已与之相当
  - 直接重算更简单

### 5.5 量化：权重、激活与 KV cache

#### 5.5.1 量化对 decode 的加速原理

- 量化：用低位宽的整数或浮点格式（INT8、INT4、FP8 等）表示原本以 BF16/FP16 存储的权重、激活或 KV cache
- Decode 受带宽限制，耗时与读取的字节数成正比
  - 权重从 BF16 降到 INT4 后读取字节数变为 1/4
  - decode 理论上最多可加速接近 4 倍
  - 同时释放的显存可以容纳更多 KV，从而支持更大的 batch

- 权重存储量约为 $N\cdot b/8$ 字节，$b$ 为每个权重的位数
  - 70B 模型在 BF16 下为 $70\times10^9\times2 = 140$ GB，理想 INT4 下为 35 GB
- 实际量化模型略大于这个值
  - 每组权重还要存储 scale（以及非对称量化的 zero point）
  - per-group 量化、组大小 128、scale 为 FP16 时，每个权重额外摊到 $16/128 = 0.125$ bit
- KV cache、激活和运行时缓冲区不随权重位宽变化
  - 部署时的总显存需求不会按同一比例下降

量化能否转化为实际加速，还取决于两个条件：
- 硬件是否支持该格式的高效计算
- 量化和反量化本身的开销是否小于节省的访存时间

#### 5.5.2 量化公式与粒度

- 量化通过缩放因子（scale）$s$ 将浮点值映射到整数网格
- 非对称量化还使用零点（zero point）$z$
  - 即浮点 0 所对应的整数值
  - 用于让整数范围覆盖不以 0 为中心的数值区间

对称量化（$b$ 位）：

$$s = \frac{\max|w|}{2^{b-1}-1},\qquad q = \text{clip}\left(\text{round}\left(\frac{w}{s}\right), -2^{b-1}, 2^{b-1}-1\right),\qquad \hat w = s\cdot q$$

非对称量化：

$$s = \frac{w_{\max}-w_{\min}}{2^b-1},\qquad z = \text{round}\left(-\frac{w_{\min}}{s}\right),\qquad q = \text{clip}(\text{round}(w/s)+z, 0, 2^b-1)$$

- 公式中各操作的含义：
  - round 把值舍入到最近的整数
  - clip 把超出可表示范围的值截断到边界（称为饱和）
  - $\hat w$ 是反量化得到的近似值
  - 量化误差为 $e = w - \hat w$
- 算例：$s=0.25$、$w=0.63$
  - $w/s=2.52$，舍入得 $q=3$
  - 反量化得 $\hat w=0.75$，误差为 $-0.12$
- 若某个值超过格式最大值，它会被截断到边界
  - 误差可能远大于 $s/2$
- 误差常用均方误差 $\frac1n\sum_i(w_i-\hat w_i)^2$ 衡量
  - 但均方误差小并不保证模型精度损失小
  - 敏感层和离群值对最终精度的影响往往更大

- 量化粒度：共享同一个 scale 的元素范围
- 从粗到细依次为：
  - per-tensor：整个张量一个 scale
  - per-channel：每个输出通道一个 scale
  - per-group：每连续若干元素一个 scale，例如每 128 个元素
- 粒度越细，量化误差越小，scale 的存储和计算开销越大

粒度的影响可以用一个小例子说明：
- 设四个权重为 $[0.02, 0.03, 0.04, 3.0]$，做 INT4 对称量化（$q_{\max}=7$）
- 四个值共用一个 scale 时，$s = 3.0/7 \approx 0.43$
  - 前三个值除以 $s$ 后都小于 0.1，全部舍入为 0，信息完全丢失
- 若把前三个值分为一组，$s = 0.04/7 \approx 0.0057$
  - 三个值分别量化为 4、5、7
  - 反量化误差不超过 $0.003$
- 代价是多存一个 scale，矩阵乘法时也要多处理一次 scale

| 粒度 | 含义 | 优点 | 代价 |
|---|---|---|---|
| Per-tensor | 整个张量共用一个 scale | 元数据最少，计算最简单 | 受离群值影响最大 |
| Per-channel | 每个通道一个 scale | 适应不同通道的幅度差异 | 需要存储与通道数相同的 scale |
| Per-group / per-block | 每组连续元素一个 scale | 误差通常最小 | scale 的访存和计算更复杂 |
| 动态 scaling | 运行时根据当前激活计算 scale | 适应输入分布的变化 | 每次需要额外的归约（求最大值）和同步 |

- 权重是静态的，scale 可以在离线量化时算好
- 激活随输入变化，有两种方式确定 scale：
  - 用校准数据预先确定一个静态 scale
  - 在运行时按 token 或按块动态计算
- 动态 scaling 精度更高
  - 但每次计算前都要先扫描一遍数据求最大值

#### 5.5.3 主流量化方案

- 方案名中 W 表示权重位宽，A 表示激活位宽
  - 例如 W4A16 表示权重 4 位、激活 16 位

| 方案 | 权重 | 激活 | 加速来源 | 适用场景 |
|---|---|---|---|---|
| W8A16 / W4A16（GPTQ、AWQ） | INT8 / INT4 | BF16 | 权重读取字节数减少 | 小 batch、显存紧张 |
| W8A8 INT8（SmoothQuant） | INT8 | INT8 | INT8 Tensor Core 与字节数减少 | 大 batch |
| FP8（W8A8） | FP8 | FP8 | FP8 Tensor Core，精度损失很小 | Hopper / Blackwell 的默认选择 |
| FP4（NVFP4 / MXFP4） | FP4 | FP4 / FP8 | Blackwell FP4 Tensor Core | 新一代硬件 |
| KV cache FP8 / INT4 | — | — | KV 读取字节数减少，容量翻倍 | 长上下文 |

两类方案的加速机制不同：
- W8A16、W4A16 只量化权重，矩阵乘法仍在 BF16 下进行
  - kernel 从 HBM 读取 INT4 权重，在寄存器中反量化为 BF16，再送入 Tensor Core
  - 它节省的是带宽，不节省计算
  - 因此适合 decode 这类带宽受限的场景
- W8A8 和 FP8 同时量化权重和激活，矩阵乘法直接在低精度 Tensor Core 上执行
  - 设 $w_i\approx s_w q_{w,i}$、$x_i\approx s_x q_{x,i}$，则

$$y=\sum_i w_i x_i\approx s_w s_x\sum_i q_{w,i}\,q_{x,i}$$

- 即先用低精度乘加单元累加整数乘积，最后乘一次 scale
  - MAC 是执行 $a\leftarrow a+w_i x_i$ 的电路
  - 累加器使用 INT32 或 FP32 以防溢出
- H100 的 FP8 Tensor Core 峰值是 BF16 的 2 倍
  - 因此 W8A8 在 prefill 和大 batch decode 等算力受限的场景下也能加速
- per-group 量化时 scale 随组变化
  - 需要在每组累加结束时分别乘 scale，kernel 实现更复杂

- W4A16 的优势在大 batch 下会缩小
  - batch 增大后计算进入算力受限区
  - 反量化本身也需要额外计算

#### 5.5.4 激活离群值

- 离群值（outlier）：数值幅度远大于同一张量中其他元素的少数元素
- LLM 的激活中存在少数通道，其数值比其他通道大几十到上百倍
  - LLM.int8() 论文称之为 emergent outliers
- 对激活做 per-tensor 量化时，scale 由离群值决定而被放大
  - 其他通道的值几乎全部被量化为 0
- 权重的分布相对平滑，较易量化
- 5.11 节介绍 SmoothQuant 和 AWQ 对这一问题的处理方法

#### 5.5.5 非均匀量化与码本

- 5.5.2 节的公式属于均匀量化：可表示的值在数轴上等间隔分布
- LLM 权重近似服从以 0 为中心的钟形分布
  - 大部分值集中在 0 附近，少数值很大
  - 等间隔网格在 0 附近点太稀，在两端点太密而很少用到

非均匀量化用一个码本（codebook）$\mathcal C=\{c_1,\ldots,c_K\}$ 表示所有可取的值，量化即选取最近的码字：

$$Q(x)=c_{j^*},\qquad j^*=\arg\min_j |x-c_j|$$

- $K=2^b$，每个权重只存储码字的索引 $j^*$
  - 例如 3 bit 码本可以取 $\{-1, -0.25, -0.05, 0, 0.05, 0.25, 1\}$ 加一个备用值
  - 0 附近的点比均匀网格更密
- 码本有两种来源：
  - 根据权重分布的统计特性设计（NF4 按正态分布的分位数放置码字）
  - 通过最小化量化误差或任务损失来学习

- 非均匀量化的难点在硬件执行
  - Tensor Core 只支持标准数值格式的乘法，任意码本的值无法直接相乘
  - 需要先通过查表把索引解码为 BF16 再计算
  - 因此码本量化一般只用于 W4A16 这类只量化权重的方案，加速来源仍是带宽节省
- 量化方案的设计是三方权衡：
  - 模型质量
  - 存储与带宽
  - 计算电路的复杂度
- 误差最小的格式不一定在实际硬件上最快

- FP4 E2M1（1 位符号、2 位指数、1 位尾数）本身也是一种非均匀表示
  - 它的 16 个编码（含 +0 与 −0）对应 $0, \pm0.5, \pm1, \pm1.5, \pm2, \pm3, \pm4, \pm6$
  - 相邻值的间隔随幅度增大
- 浮点格式的一般形式为 $x=(-1)^{\text{sign}}(1.f)_2\cdot 2^{e-\text{bias}}$
  - 以相对精度换取跨数量级的动态范围
  - FP4 是这一形式在极少位数下的特例，而且硬件可以直接计算

#### 5.5.6 Microscaling 格式

- FP4 只有 16 个编码，动态范围很窄
  - 直接用于整个张量会产生严重饱和或下溢
- Microscaling（MX）格式的做法是让每一小块元素共享一个 scale
  - 每 32 个元素（MXFP4、MXFP8）共享一个 8 位的 2 的幂次 scale（E8M0 格式）
  - 块内每个元素用 FP4 或 FP8 存储
- NVIDIA 的 NVFP4 采用类似结构
  - 块大小为 16，scale 为 FP8 E4M3
  - 并在张量级再加一个 FP32 scale

- 这相当于把 per-group 量化标准化为硬件格式
  - Blackwell 的 Tensor Core 在计算时直接读取块 scale 并完成缩放
  - 不需要软件 kernel 额外处理
- 按 MXFP4 计算，scale 的开销为每元素 $8/32=0.25$ bit，等效位宽为 4.25 bit
- 块越小，量化越精确，scale 的存储和处理开销越大
- 不同 4 bit 格式在编码、块大小和 scale 规则上各不相同
  - 选用时需要以具体格式规范为准

### 5.6 投机解码

#### 5.6.1 基本流程

- 投机解码（speculative decoding）：先用低成本的草稿模型（draft model）连续生成若干候选 token，再由目标模型一次前向并行验证
  - 目的是减少目标模型的串行前向次数
- 其依据是 decode 阶段 GPU 算力大量闲置
  - 目标模型验证 $\gamma$ 个 token 与生成 1 个 token 的耗时接近
  - 两者都主要花在读取一遍权重上

流程如下：

1. 草稿模型连续生成 $\gamma$ 个候选 token；
2. 目标模型做一次前向，并行计算这 $\gamma$ 个位置的输出分布；
3. 按接受规则接受一个前缀，并在第一个被拒绝的位置重新采样一个 token。

- 每次目标模型前向至少产出 1 个 token，最多产出 $\gamma+1$ 个

#### 5.6.2 接受规则

- 设草稿分布为 $q$，目标分布为 $p$
- 对草稿 token $x\sim q$，以概率 $\min\left(1, \frac{p(x)}{q(x)}\right)$ 接受
- 若被拒绝，则从下式重新采样：

$$p'(x) = \frac{\max(0,\, p(x)-q(x))}{\sum_{x'}\max(0,\, p(x')-q(x'))}$$

- 按此规则，输出分布严格等于 $p$，不引入任何质量损失（证明见 5.12 节）
- 草稿 token 被接受的概率称为接受率（acceptance rate）
- 贪心解码时，规则退化为「草稿 token 等于目标模型的 argmax 时接受」

#### 5.6.3 草稿的来源

| 方法 | 草稿来源 | 特点 |
|---|---|---|
| 独立小模型 | 同系列小模型（如用 1B 模型为 70B 模型生成草稿） | 实现简单，要求词表相同 |
| Medusa | 在目标模型上增加若干预测头，分别预测第 $+1, +2, \dots$ 个 token | 无需额外模型，采用树形验证 |
| EAGLE / EAGLE-2 / EAGLE-3 | 一个很小的自回归头，输入为目标模型的隐状态 | 接受率高，是目前的主流方案 |
| MTP（DeepSeek-V3） | 训练阶段已有的多 token 预测模块，推理时直接用作草稿 | 第二个 token 的接受率为 85–90% |
| n-gram / prompt lookup | 从 prompt 或已生成文本中匹配 n-gram | 几乎无额外成本，在代码编辑、RAG 场景中效果显著 |

#### 5.6.4 适用条件

- 投机解码的加速依赖「验证多个 token 的成本接近验证 1 个 token」这一前提
- 该前提在以下两种情况下不成立：
  - 第一种是大 batch
    - batch 为 $b$、草稿长度为 $\gamma$ 时，目标模型每次前向要处理 $b(\gamma+1)$ 个 token
    - batch 为 4、$\gamma=4$ 时为 20 个 token，远低于 5.1.4 节的交叉点 $b^*\approx 295$，验证几乎不增加耗时
    - batch 为 128 时为 640 个 token，已超过交叉点，前向进入算力受限区，耗时约随 token 数线性增长
    - 此时被拒绝的草稿 token 消耗的计算是实打实的浪费，加速比下降，甚至慢于普通解码
  - 第二种是接受率低的任务
    - 高温采样和创意写作中，目标分布 $p$ 较平坦，草稿模型很难猜中
    - 接受率 $\alpha$ 可能低于 0.5
    - 按 5.12.3 节的公式，$\alpha=0.4$、$\gamma=4$、$c=0.05$ 时加速比约为 1.4，扣除调度开销后收益很小
- 因此投机解码主要用于低延迟、小 batch 的场景
  - 生产系统通常根据负载动态启用或关闭
  - 负载低时开启以降低 TPOT，负载高时关闭以保证吞吐

### 5.7 Prefill/Decode 分离部署

#### 5.7.1 混合部署的相互干扰

- 在同一块 GPU 上同时运行 prefill 和 decode 时，一个长 prompt 的 prefill 需要几百毫秒
  - 同一 batch 中的 decode 请求都要等待它完成，TPOT 出现尖峰
- 反之，若为保证 TPOT 而限制 prefill，TTFT 会变差
- 此外，两个阶段对并行方式和 batch 大小的最优选择也不同

#### 5.7.2 PD 分离架构

- PD 分离（prefill/decode disaggregation）：把 prefill 和 decode 部署在不同的 GPU 资源池上
  - prefill 池完成计算后，将生成的 KV cache 传给 decode 池继续生成

```text
            ┌─────────────┐   KV cache (RDMA)   ┌─────────────┐
 请求 ────> │ Prefill 池  │ ──────────────────> │ Decode 池   │ ──> token 流
            │ 算力密集     │                     │ 带宽密集     │
            │ 小 TP / 高 MFU│                    │ 大 batch / 大 EP│
            └─────────────┘                     └─────────────┘
```

- 图中符号：
  - MFU 是模型实际达到的 FLOPS 与硬件峰值 FLOPS 之比
  - EP 是专家并行（MoE 的不同专家分布在不同 GPU 上）
- 代表性工作包括：
  - DistServe（OSDI 2024）：以 goodput 为目标优化两池的资源配比，goodput 定义见 5.9.1 节
  - Splitwise（ISCA 2024）：两阶段使用不同代际的硬件
  - Mooncake：以 KV 存储为中心的架构
  - DeepSeek-V3 推理部署：prefill 使用 EP32，decode 使用 EP320

#### 5.7.3 KV 传输开销

以 4K token 的 prompt、Llama-3-70B 为例，KV 大小为 $4096\times320\text{ KB}\approx 1.3$ GB。

| 链路 | 传输时间 |
|---|---|
| 8 张 400G 网卡并行（400 GB/s） | 3.3 ms |
| 单张 400G 网卡（50 GB/s） | 26 ms |
| TCP 10 GB/s | 130 ms |

- 该 prompt 的 prefill 本身约需 150 ms，使用 RDMA 时传输时间可以忽略
- 此外还可以逐层传输
  - 即每算完一层就发送该层的 KV，使传输与计算重叠

#### 5.7.4 分离部署的适用条件

| 适合分离 | 无需分离 |
|---|---|
| prompt 长、SLO 严格 | prompt 短、负载小 |
| 规模大，两个池都能各自达到高利用率 | GPU 数量少 |
| 具备 RDMA 网络 | 只有普通以太网 |

三个条件各有依据：
- prompt 越长，单次 prefill 越久，对同一 GPU 上 decode 的干扰越大
  - 分离的收益越明显
- SLO 越严格，越难在混合部署下同时满足 TTFT 和 TPOT
- 分离后两个池的 GPU 数必须按负载比例配置
  - 例如 5.13 节的算例中 prefill 与 decode 的节点数约为 5:3
  - GPU 总数很少时，这一比例无法精确实现，某个池会长期闲置
  - 资源利用率反而低于混合部署
- KV 传输的耗时按 5.7.3 节的表格随链路带宽变化
  - 只有普通以太网时，传输时间可能与 prefill 本身相当
  - 分离带来的 TTFT 改善被抵消

不做分离时，可以用 chunked prefill（5.10 节）减轻两阶段的干扰。
### 5.8 长上下文 serving：KV 压缩、稀疏注意力与序列并行

#### 5.8.1 问题规模

128K 上下文下有三个问题：
- Llama-3-70B 每个请求的 KV 达到 40 GB
- prefill 中 $O(s^2)$ 的注意力计算量超过线性层
- decode 每步读取的 KV 字节数远大于权重

针对这三个问题，有三类方法：
- 压缩 KV
- 减少注意力计算
- 跨多块 GPU 切分序列

#### 5.8.2 压缩 KV cache

| 方法 | 原理 | 压缩比 |
|---|---|---|
| GQA / MQA | 多个 query 头共享一组 KV 头 | 4–8× |
| MLA（DeepSeek-V2/V3） | 将 KV 压缩为低秩 latent 向量，推理时把上投影矩阵吸收进 Q 和 O 的投影 | 相对 MHA 约 50× |
| KV 量化 | FP8 / INT4 / 2bit（KIVI） | 2–8× |
| KV 淘汰 | StreamingLLM（保留 attention sink 与最近窗口）、H2O（保留累计注意力权重大的 token）、SnapKV（按观察窗口选择 token） | 视配置而定，存在精度风险 |
| 跨层共享 | 相邻层共享 KV（CLA、YOCO） | 2× |

这几类方法作用于 5.1.3 节 KV 公式的不同因子：
- GQA/MQA 减少 $n_{\text{kv\_heads}}$
  - Llama-3-70B 有 64 个 query 头、8 个 KV 头
  - KV 是同规模 MHA 的 1/8
- MLA 不再缓存每个头的 K 和 V，而是缓存一个低维 latent 向量
  - DeepSeek-V3 中每层每 token 为 512 维加 64 维位置编码部分
  - 注意力计算时再由 latent 恢复各头所需的信息
- KV 量化减少 $s_{\text{elem}}$
- KV 淘汰减少 $L$，即只保留部分 token 的 KV
  - 这改变了模型实际看到的上下文
  - 被丢弃的 token 若在后文被需要，模型将无法准确回忆
  - 这是它与前几类方法在精度风险上的根本区别
- 跨层共享减少需要单独存储 KV 的层数

#### 5.8.3 稀疏与局部注意力

| 方法 | 原理 |
|---|---|
| 滑动窗口与全局层混合 | 多数层只关注局部窗口，少数层关注全局（Gemma、GPT-OSS 等） |
| 稀疏注意力 | NSA（压缩、选择、滑窗三路分支）、DeepSeek Sparse Attention（轻量 indexer 选出 top-k token） |
| 线性注意力 / SSM 混合 | 部分层替换为 Mamba 或线性注意力，其状态不随序列长度增长 |

滑动窗口注意力中每个 token 只关注前 $w$ 个 token（例如 $w=4096$）：
- 这些层的 KV 只需保留最近 $w$ 个 token
- 计算量从 $O(s^2)$ 降为 $O(sw)$
- 全局层保留完整上下文，负责长距离依赖

稀疏注意力保留全部 KV，但每个 query 只选取一小部分 token 参与计算：
- 节省的是计算和读取量，而不是 KV 容量

线性注意力需要单独推导。标准 softmax 注意力对第 $t$ 个 query 的输出为

$$o_t=\frac{\sum_{i\le t}\exp(q_t^\top k_i/\sqrt d)\,v_i}{\sum_{i\le t}\exp(q_t^\top k_i/\sqrt d)}$$

- 每一步都要与所有历史 $k_i$ 计算相似度
  - 因此必须保留全部 KV
  - decode 第 $t$ 步的读取量与 $t$ 成正比
- 线性注意力用核特征映射 $\phi$ 把相似度改写为内积 $\phi(q_t)^\top\phi(k_i)$，于是

$$o_t=\frac{\phi(q_t)^\top\left(\sum_{i\le t}\phi(k_i)v_i^\top\right)}{\phi(q_t)^\top\left(\sum_{i\le t}\phi(k_i)\right)}$$

- 括号内的两个求和与 $q_t$ 无关，可以递推维护：

$$S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad z_t=z_{t-1}+\phi(k_t),\qquad o_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}$$

- $S_t$ 是一个 $d'\times d$ 的矩阵（$d'$ 为特征维度），$z_t$ 是一个 $d'$ 维向量
  - 两者大小与序列长度无关
- decode 每步只需读写这个固定大小的状态，不再扫描历史 token
- prefill 的计算量也从 $O(s^2)$ 降为 $O(s)$

线性注意力不是 softmax 注意力的等价变换：
- $\exp(q^\top k)$ 无法精确分解为有限维特征的内积
- 固定大小的状态 $S_t$ 也无法无损保存任意长的历史
  - 模型精确检索远处某个 token 的能力因此下降
- 实践中线性注意力（及 Mamba 等 SSM）通常与少量 softmax 注意力层混合使用
  - 由后者负责精确的长距离检索

FlashAttention 与上述方法的性质也不同：
- 它不改变注意力的数学定义
- 它通过分块（tiling）把 Q、K、V 的小块载入 SRAM
  - 并用 online softmax 逐块更新运行最大值与归一化分母
  - 避免在 HBM 中写出 $s\times s$ 的分数矩阵
- 它减少的是 HBM 读写和临时存储，成对交互的 $O(s^2)$ 计算量不变

#### 5.8.4 跨卡切分长序列

prefill 阶段使用 context parallel（3.6 节）沿序列维度切分输入：
- 128K token 的 prompt 分给 8 块 GPU，每块处理 16K token 的 query
- 通过环形通信依次获取其他 GPU 上的 K、V 分块
- 注意力的 $O(s^2)$ 计算被均分，TTFT 约降为原来的 1/8（扣除通信开销）

decode 阶段，单个请求的 40 GB KV 可能超过单块 GPU 能分出的容量，也会使每步读取时间过长：
- 做法是把 KV 沿序列维度切分到多块 GPU
  - 每块 GPU 用当前 query 对本地的那一段 KV 计算部分注意力
  - 得到局部输出、局部最大值和局部归一化分母
  - 再用 online softmax 的合并规则把各段结果合成精确的最终输出
- 这相当于 Flash-Decoding 在单块 GPU 内按 KV 长度切分的跨卡版本
- 通信量只有每段的局部输出和两个标量，与 KV 长度无关

访问频率低的 KV 的处理：
- 可以下沉到 CPU 内存或 SSD，按需取回
- 取回与重算的决策见 5.4.5 至 5.4.7 节

#### 5.8.5 各类方法的比较

上述方法节省的资源不同，改变模型的程度也不同：

| 方法 | 主要节省的资源 | 是否改变模型的计算 | 主要风险 |
|---|---|---|---|
| FlashAttention | HBM 读写、临时分数矩阵 | 否，在数值误差范围内与标准注意力一致 | kernel 实现复杂度，受片上存储容量约束 |
| 线性注意力 / SSM | 历史 KV 的扫描与存储，状态大小固定 | 是，改变了注意力函数 | 精确长距离检索能力下降，需验证模型质量 |
| MLA | KV cache 容量 | 是，改变了 KV 的参数化方式 | 训练与推理 kernel 实现复杂 |
| GQA / MQA | KV head 数量，即 KV 容量与读取量 | 是，改变了头结构 | 共享程度过高时质量下降 |
| 混合架构（滑窗、稀疏、线性层与全局层组合） | 视组合而定，通常同时节省计算和 KV | 是 | 系统实现与调度复杂，不同层需要不同的 KV 管理方式 |

选择时需要区分两类方法：
- FlashAttention 和序列并行属于纯系统优化
  - 可以直接用于任何已训练的模型
- 其余方法改变了模型结构
  - 必须在训练阶段采用
  - 推理系统只能支持而无法单方面引入

### 5.9 调度与 SLO：TTFT 与 TPOT 的权衡

#### 5.9.1 指标定义

- SLO（Service Level Objective）是服务对延迟等指标承诺的目标值
  - 通常以分位数表示，例如 TTFT P99 < 2 s
- ITL（Inter-Token Latency）是相邻两个输出 token 之间的实际间隔
  - TPOT 是其平均值
- Goodput 是单位时间内满足 SLO 的请求数
  - 与不区分是否达标的吞吐不同

| 指标 | 定义 | 典型目标（对话场景） |
|---|---|---|
| TTFT | 请求到达至第一个 token 输出 | P99 < 1–2 s |
| TPOT / ITL | 相邻两个输出 token 的间隔 | P99 < 50–100 ms |
| E2E latency | TTFT + TPOT × (输出长度 − 1) | — |
| Throughput | 每秒输出 token 数（或每秒总 token 数） | 越高越好 |
| Goodput | 每秒满足 SLO 的请求数 | DistServe 提出的优化目标 |

- 人的阅读速度约为每秒 5–10 个词
- TPOT 为 50 ms 时对应每秒 20 token，已能保证流畅的阅读体验

#### 5.9.2 参数对各指标的影响

| 调大 | TTFT | TPOT | 吞吐 |
|---|---|---|---|
| batch size | 排队时间缩短 | 变差（每步耗时增加） | 上升 |
| prefill 优先级 | 变好 | 变差（decode 被打断） | — |
| decode 优先级 | 变差 | 变好 | — |
| TP 度 | 变好 | 变好 | 每块 GPU 的吞吐下降 |

表中各项的机制如下：
- 增大 batch
  - 使更多请求同时得到服务，排队时间缩短
  - 每步读取的 KV 增多，每步耗时变长
- 调度器在每次迭代中优先放入新请求的 prefill
  - 可以缩短 TTFT
  - 长 prefill 会拉长该次迭代，正在 decode 的请求的 ITL 出现尖峰
  - 反之优先 decode 则新请求排队更久
- 提高 TP 度
  - 使每块 GPU 读取的权重减少，单步延迟下降，TTFT 和 TPOT 都改善
  - AllReduce 通信占比上升，每块 GPU 贡献的吞吐降低
- 不存在同时优化所有指标的配置
  - 调参的目标是在 SLO 约束下最大化 goodput

#### 5.9.3 调度策略

先来先服务（FCFS）：
- 是默认策略，实现简单且公平
- 一个长请求会使其后的短请求等待，平均延迟较高

分级队列按优先级或 SLO 等级划分请求：
- 不同类别的请求对延迟的敏感度不同
  - 交互式对话要求 TTFT 在 1 s 以内
  - 批量离线任务（数据标注、评测）可以容忍分钟级延迟
  - Agent 的中间步骤（规划、工具参数生成）的 TPOT 要求可以比面向用户的最终回答宽松
- 为所有请求设定同一组 SLO 会使调度过于保守
- 按类别设定不同目标后，调度器可以：
  - 在交互请求空闲时用离线请求填满 batch
  - 在交互请求到达时优先服务它们

最短作业优先：
- 可以降低平均延迟，但输出长度事先未知
- 一些系统用一个小模型或基于 prompt 特征的回归器预测输出长度，再据此近似排序
- 预测误差较大时，长请求可能长期得不到调度
  - 需要按等待时间提高其优先级以避免饥饿

过载控制：
- 指当到达率超过系统容量时，主动拒绝部分请求或降级服务
  - 降级的例子：缩短最大输出长度、改用更小的模型
  - 目的是避免让所有请求排队直到全部超时
- 按 5.13.1 节的 Little 定律，到达率持续超过服务率时队列长度无限增长
  - 所有请求的延迟都会违反 SLO，goodput 降为零

调度器的决策范围还可以更大：
- 请求路由到哪个副本
- 是否与其他请求合并进 batch
- 何时执行
- 是否迁移其 KV
- 如何避免某个副本过热
- 是否优先利用已有的前缀缓存

这些决策相互影响：
- 例如为了命中缓存而路由到繁忙的副本，可能使排队时间超过节省的 prefill 时间
- 需要用统一的端到端成本模型来比较
  - 而不是各个组件独立做贪心选择

#### 5.9.4 端到端延迟与 Amdahl 定律

用户感知的延迟不只由 GPU kernel 决定：
- 一个请求依次经过网络接入、负载均衡与路由、排队、CPU 侧的 tokenize 与预处理、GPU 前向计算、KV 管理、采样与 detokenize、流式输出

Amdahl 定律给出了局部优化对整体的影响：
- 若被优化部分占端到端时间的比例为 $f$，该部分加速 $s$ 倍，则整体加速比为

$$S=\frac{1}{(1-f)+f/s}$$

- 若某个 kernel 加速 2 倍，而它只占端到端时间的 20%，则 $S = 1/(0.8 + 0.2/2) \approx 1.11$
  - 整体只快 11%
- 即使该 kernel 被优化到耗时为零，整体加速也不超过 $1/0.8 = 1.25$ 倍

推理系统的优化应从端到端的关键路径分析开始：
- 先测量各阶段的耗时分布，找出占比最大的部分
- 实践中常见的情况是，GPU 计算已经足够快
  - CPU 侧的调度循环、tokenize 或 Python 开销成为瓶颈
  - 小模型上尤其明显
- 评测时也不能只看平均值
  - P95/P99 尾延迟往往由排队、抢占和偶发的长 prefill 决定
  - 与平均吞吐关系不大

### 5.10 批处理的演进：静态、连续与分块

#### 5.10.1 三种方式对比

| 代 | 方法 | 遗留问题 |
|---|---|---|
| 1 | 静态 batching：整批进入、整批退出 | 短请求等待长请求，GPU 槽位空闲 |
| 2 | continuous batching（Orca、vLLM）：迭代级调度 | 新请求的长 prefill 打断 decode，TPOT 出现尖峰 |
| 3 | chunked prefill（Sarathi-Serve）：长 prompt 分块，每步 token 预算固定 | 每个块需要重复读取之前各块的 KV，开销略有增加 |

#### 5.10.2 Chunked prefill

- Chunked prefill 是指将长 prompt 的 prefill 切分为多个块，分散到多次迭代中执行
  - 每次迭代与 decode 请求共同组成一个 batch
- 每次迭代设定一个 token 预算（如 `max_num_batched_tokens = 2048`）
  - 先放入所有 decode 请求（每个占 1 token）
  - 剩余预算分配给某个 prefill 请求的一个块：

```text
迭代 t:   [decode × 100 token] + [请求 X 的 prompt 第 0–1947 token]
迭代 t+1: [decode × 100 token] + [请求 X 的 prompt 第 1948–3895 token]
...
```

- 每步耗时保持稳定，decode 不会被长 prefill 阻塞
- decode 本身受带宽限制，搭载一块 prefill 相当于利用了原本闲置的算力
  - 这种做法称为 piggybacking

#### 5.10.3 Chunk 大小的选择

- Chunk 越大，prefill 效率越高、TTFT 越短，但每步耗时越长、TPOT 越差
- 一般取 512–4096，通过压测找出满足 TPOT SLO 的最大值

两端的极限可以帮助理解这一权衡：
- chunk 过小（如 128）
  - prefill 块的算术强度过低，矩阵乘法无法达到算力上限
  - 一个 8000 token 的 prompt 要分 60 多次迭代才能完成，TTFT 明显增加
  - 每个块还要重新读取此前所有块的 KV，重复读取量随块数增加
- chunk 过大（如 16384）
  - 一次迭代的 prefill 部分接近不分块的情况
  - decode 请求的 ITL 尖峰重新出现
- 按 5.1.4 节的分析，每次迭代的总 token 数达到数百后，计算已进入算力受限区
  - 此后再增大 chunk 主要增加每步耗时，对 prefill 效率的提升有限

### 5.11 量化算法：GPTQ、AWQ、SmoothQuant 与 KV 量化

#### 5.11.1 问题定义

训练后量化通常逐层进行：
- 给定校准数据在该层的输入激活 $X$，寻找量化权重 $\hat W$，使该层输出误差最小：

$$\min_{\hat W}\ \|WX - \hat WX\|_F^2$$

- 该目标在权重矩阵的各行之间相互独立
- 对每一行 $w$，误差是 $w$ 的二次型
  - 其 Hessian 矩阵（误差对权重的二阶导数矩阵）为 $H = 2XX^\top$

#### 5.11.2 OBS 更新

- OBS（Optimal Brain Surgeon）是一种基于 Hessian 的剪枝/量化方法
  - 每量化（或剪除）一个权重后，利用 Hessian 的逆调整其余未量化的权重，使总误差的增量最小
- 量化第 $q$ 个权重时，补偿量和误差增量为：

$$\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}}\cdot (H_F^{-1})_{:,q},\qquad \Delta E = \frac{(w_q - \text{quant}(w_q))^2}{2[H_F^{-1}]_{qq}}$$

- 其中 $F$ 是尚未量化的权重集合

#### 5.11.3 GPTQ 的三项改进

1. 固定量化顺序：所有行按相同的列顺序量化，不再逐个贪心选择误差最小的权重
   - 这样 $H^{-1}$ 的更新可在所有行之间共享
   - 复杂度从 $O(d_{\text{row}}\cdot d_{\text{col}}^3)$ 降至 $O(\max(d_{\text{row}} d_{\text{col}}^2, d_{\text{col}}^3))$；
2. 延迟批量更新：以 128 列为一块，块内逐列更新，块处理完后再一次性更新后续各列
   - 目的是提高算术强度；
3. Cholesky 分解：预先对 $H^{-1}$ 做 Cholesky 分解
   - 目的是保证数值稳定性。

175B 模型的 4-bit 量化在单块 GPU 上约需 4 小时。

#### 5.11.4 GPTQ 核心代码

```python
import torch

def gptq_quantize(W, X, bits=4, group=128, damp=0.01, blocksize=128):
    """W: [rows, cols]，X: [cols, n_samples] 校准激活。返回反量化后的权重。"""
    W = W.clone().float(); cols = W.shape[1]
    H = 2 * X @ X.T
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)   # 阻尼
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qmax = 2 ** (bits - 1) - 1
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1, Q1, Err1 = W[:, i1:i2].clone(), torch.zeros_like(W[:, i1:i2]), torch.zeros_like(W[:, i1:i2])
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            if col % group == 0:                                   # 每组重新算 scale
                scale = W[:, col:col + group].abs().amax(dim=1).clamp(min=1e-8) / qmax
            w = W1[:, j]
            q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
            Q1[:, j] = q
            err = (w - q) / Hinv1[j, j]
            W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)  # 补偿块内后续列
            Err1[:, j] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]                      # 懒更新块外的列
    return Q
```

以上为简化版本，完整实现与之的差异：
- 完整实现会在每组开始时使用当时已更新的权重计算 scale
- 完整实现会保存整数权重和 scale，以配合专用的反量化 kernel

#### 5.11.5 AWQ：按激活幅度保护重要通道

- AWQ 基于以下观察：只有约 1% 的权重输入通道（对应激活幅度大的通道）对量化误差影响显著
- AWQ 不对这些通道保留高精度，而是做等价缩放：

$$y = Wx = (W\cdot \text{diag}(s))(\text{diag}(s)^{-1}x)$$

- 将重要通道的权重放大（$s_j>1$），可减小其相对量化误差
- 激活相应缩小，该缩放可融合进前一个算子
- 缩放因子取 $s = s_x^\alpha$
  - $s_x$ 是该通道激活的平均幅度
  - $\alpha\in[0,1]$ 通过网格搜索确定，搜索目标是量化后输出误差最小
- AWQ 不做基于 Hessian 的误差补偿
  - 对校准集细节的依赖较小，泛化性更好

#### 5.11.6 SmoothQuant：将激活的量化难度迁移到权重

- W8A8 量化中，激活因离群值而难以量化，权重则较易量化
- SmoothQuant 对每个输入通道 $j$ 做如下等价变换：

$$Y = (X\,\text{diag}(s)^{-1})(\text{diag}(s)\,W),\qquad s_j = \frac{\max|X_j|^\alpha}{\max|W_j|^{1-\alpha}}$$

- $\alpha=0.5$ 时，激活与权重两侧的最大值被调整到同一量级
- $\text{diag}(s)^{-1}$ 可以融合进前一层 LayerNorm 的权重，不产生额外开销

#### 5.11.7 KV cache 量化

| 观察 | 做法 |
|---|---|
| Key 的离群值集中在固定通道 | Key 按通道（per-channel）量化 |
| Value 没有固定的离群通道 | Value 按 token（per-token）量化 |
| 最近的 token 对注意力影响最大 | 最近一个窗口保持全精度，更早的 token 量化（KIVI） |
| FP8 E4M3 误差小 | 生产中最常用 FP8 KV，配合 per-tensor 或 per-head scale |

Key 和 Value 采用不同量化方向的原因在于它们在注意力中的作用不同：
- Key 与 query 做内积，少数固定通道的大幅值主导内积结果
  - 按通道量化时，每个通道有自己的 scale，大幅值通道不会挤压其他通道的精度
- Value 按注意力权重加权求和，误差在 token 之间平均
  - 按 token 量化更合适
  - 新生成的 token 可以独立量化后追加，不需要重新计算已有 token 的 scale

FP8 KV 的收益：
- 使 KV 容量翻倍，在长上下文场景下基本等价于 batch 翻倍
- 以 5.13.5 节的 decode 节点为例
  - 约 500 GB 的 KV 空间在 BF16 下可容纳约 700 个平均上下文 2250 token 的请求
  - FP8 下约 1400 个
- 同时每步读取的 KV 字节数减半，TPOT 也随之下降

### 5.12 投机解码的数学：接受率与加速比

#### 5.12.1 正确性证明

- 记 $\beta = \sum_x \min(p(x), q(x))$ 为一次接受的总概率
- 输出 token 为 $x$ 的概率为：

$$P(x) = \underbrace{q(x)\min\left(1,\frac{p(x)}{q(x)}\right)}_{\text{草稿被接受}} + \underbrace{(1-\beta)\,p'(x)}_{\text{被拒后重采样}} = \min(p,q) + (1-\beta)\frac{p(x)-\min(p(x),q(x))}{1-\beta} = p(x)$$

- 推导中用到 $\sum_x\max(0,p-q) = 1-\beta$
- 因此无论草稿质量如何，输出分布都严格等于目标模型的分布

#### 5.12.2 期望产出

- 假设每个草稿 token 独立地以概率 $\alpha$ 被接受（$\alpha$ 是 $\beta$ 的期望），草稿长度为 $\gamma$
- 一次验证产出的 token 数记为 $X$：
  - 前 $k$ 个被接受、第 $k+1$ 个被拒绝时，产出 $k+1$ 个（含重采样的 1 个）
  - 全部接受时，产出 $\gamma+1$ 个（含目标模型额外给出的 1 个）

$$\mathbb{E}[X] = \sum_{k=0}^{\gamma} \alpha^k = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

#### 5.12.3 加速比

- 设草稿模型一次前向的耗时是目标模型的 $c$ 倍
- 则一轮投机解码耗时为 $\gamma c + 1$（以目标模型一次前向为单位）

$$\text{Speedup} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$$

| $\alpha$ | $\gamma$ | $c$ | $\mathbb{E}[X]$ | 加速比 |
|---|---|---|---|---|
| 0.6 | 3 | 0.05 | 2.18 | 1.89 |
| 0.8 | 4 | 0.05 | 3.36 | 2.80 |
| 0.8 | 8 | 0.05 | 4.33 | 3.09 |
| 0.9 | 5 | 0.02 | 4.69 | 4.26 |

#### 5.12.4 最优草稿长度

- 随 $\gamma$ 增大
  - $\mathbb{E}[X]$ 的增长逐渐放缓（几何级数趋于饱和）
  - 草稿成本线性增长
- 对加速比关于 $\gamma$ 求最大值即可得到最优草稿长度
  - $\alpha$ 在 0.6–0.8 时，最优 $\gamma$ 一般在 3–6
- EAGLE-2 等方法根据草稿置信度动态决定候选树的形状和深度

#### 5.12.5 模拟验证

```python
import numpy as np
rng = np.random.default_rng(0)

def speculative_step(p, q, gamma):
    """p, q: 词表上的分布（这里假设各位置同分布，只为验证期望产出）"""
    out = 0
    for _ in range(gamma):
        x = rng.choice(len(q), p=q)
        if rng.random() < min(1, p[x] / q[x]):
            out += 1
        else:
            return out + 1                     # 拒绝：重采样 1 个
    return out + 1                             # 全部接受：外加 1 个

V = 50
p = rng.dirichlet(np.ones(V)); q = 0.7 * p + 0.3 * rng.dirichlet(np.ones(V))
alpha = np.minimum(p, q).sum()
emp = np.mean([speculative_step(p, q, 4) for _ in range(20000)])
print(f"alpha={alpha:.3f}  E[X] theory={(1 - alpha**5) / (1 - alpha):.3f}  empirical={emp:.3f}")
```

### 5.13 SLO 与容量规划算例

#### 5.13.1 Little 定律

- Little 定律指出，稳态系统中的平均请求数等于到达率与平均停留时间之积：

$$L = \lambda W$$

- 例如 QPS 为 100、每个请求平均处理 20 s 时，系统中平均同时存在 2000 个请求
  - KV cache 需要能够容纳这些请求

#### 5.13.2 问题设定

- 部署 Llama-3-70B（FP8 权重与 FP8 KV），峰值 100 QPS
- 平均输入 2000 token、平均输出 500 token
- SLO 为 TTFT P99 < 2 s、TPOT P99 < 50 ms
- 硬件为 8×H100 节点，采用 PD 分离
- 求所需节点数

#### 5.13.3 负载需求

- prefill：$100\times2000 = 2\times10^5$ token/s；
- decode：$100\times500 = 5\times10^4$ token/s；
- 同时处于 decode 阶段的请求数（由 Little 定律）：$\lambda\times$ 输出时间 $= 100\times500\times\text{TPOT}$
  - 按 SLO 上限 TPOT = 50 ms 计算为 2500
  - 按下文算得的实际 TPOT 约 14 ms 计算为 700

#### 5.13.4 Prefill 节点数

每个节点的 prefill 吞吐（FP8 峰值 1979 TFLOPS，MFU 50%）：

$$\frac{8\times1979\times10^{12}\times0.5}{2\times70\times10^9}\approx 5.6\times10^4\ \text{token/s}$$

- 所需节点数为 $2\times10^5/5.6\times10^4\approx 3.6$，取 4 个
- 单请求 prefill 约 $2000/5.6\times10^4\approx 36$ ms，为排队留有充足余量

#### 5.13.5 Decode 节点数

每个节点运行一个 TP=8 实例，batch 为 128，平均上下文 2250 token：

| 项 | 值 |
|---|---|
| 权重（FP8） | 70 GB |
| KV（FP8，160 KB/token） | $128\times2250\times160\text{ KB}\approx 46$ GB |
| 每步读取量 | 116 GB |
| 每步耗时（26.8 TB/s × 70% 有效带宽） | $116/18.8\approx 6.2$ ms |
| 每步计算 | $2\times70\times10^9\times128 / (8\times1979\times10^{12}\times0.5)\approx 2.3$ ms，被带宽时间覆盖 |

- 计入调度与采样开销后取 8 ms/步，TPOT 为 8 ms，远低于 50 ms
- 吞吐为 $128/0.008 = 1.6\times10^4$ token/s/节点
- Batch 仍可继续增大
  - 640 GB 显存扣除 70 GB 权重和预留空间后，约 500 GB 可用于 KV，可容纳约 1400 个请求
  - 但每步读取量随 batch 增长
  - batch 为 512 时每步约读取 $70 + 184 = 254$ GB，耗时约 14 ms
  - 吞吐约 $3.7\times10^4$ token/s/节点，TPOT 仍满足 SLO

节点数的两种算法：
- 按吞吐计算需要 $5\times10^4/3.7\times10^4\approx 1.4$ 个节点
- 按并发计算，700 个请求除以每节点 512 并发，同样为 1.4 个节点
- 取 2 个，再增加 1 个用于容忍单节点故障，共 3 个

#### 5.13.6 结论

- prefill 4 节点加 decode 3 节点共 7 节点
  - prefill 侧再预留 1 个节点应对突发流量，合计 8 节点
- 按每块 GPU 每小时 2 美元计算，每天成本为 $8\times8\times24\times2\approx 3100$ 美元
- 平均负载按峰值的 50% 计算，每天输出 token 数为 $5\times10^4\times86400\times0.5\approx 2.2\times10^9$
  - 即每百万输出 token 约 1.4 美元
  - 该数字已包含输入 token 的处理成本，因为 prefill 节点也计入了总成本

该算法的要点：
- 两个阶段按各自的瓶颈分别估算
  - prefill 按算力
  - decode 按带宽和显存容量
- 每个阶段取约束更紧的结果
- 实际部署中还需通过 5.14.4 节的压测校正 MFU 和带宽效率的假设值

### 5.14 案例：vLLM 配置调优

#### 5.14.1 初始配置

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 8
```

压测结果：
- 吞吐较好
- 突发流量下 TTFT P99 达到 8 s
- 长 prompt 到达时 TPOT 出现尖峰

#### 5.14.2 参数调整

| 参数 | 作用 | 调整 |
|---|---|---|
| `--gpu-memory-utilization` | 显存中分配给权重与 KV 的比例，默认 0.9 | 调至 0.92，增加的显存用于 KV |
| `--max-model-len` | 最大上下文长度，决定单请求 KV 上限 | 按业务需求设为 32768，不使用 128K 默认值 |
| `--max-num-seqs` | 最大并发序列数 | 从 256 提高到 512 |
| `--enable-chunked-prefill` | 分块 prefill | 开启（新版本默认开启） |
| `--max-num-batched-tokens` | 每步 token 预算 | 设为 4096，在 TTFT 与 TPOT 之间折中 |
| `--enable-prefix-caching` | 前缀缓存 | 开启，系统提示词命中率高 |
| `--kv-cache-dtype fp8` | FP8 KV | 开启，KV 容量翻倍 |
| `--quantization fp8` | FP8 权重 | 开启，权重为 70 GB |
| `--speculative-config` | 投机解码 | 在低负载时段的低延迟实例上开启 |

#### 5.14.3 调整后的配置

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --max-num-seqs 512 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --enable-prefix-caching
```

#### 5.14.4 压测方法

```bash
# vLLM 自带基准：固定输入输出长度分布，按 Poisson 过程发送请求
vllm bench serve --model meta-llama/Llama-3.1-70B-Instruct \
  --dataset-name random --random-input-len 2000 --random-output-len 500 \
  --request-rate 20 --num-prompts 2000
```

- 逐步提高 `--request-rate`，记录每一档的吞吐、TTFT P50/P99 和 TPOT P50/P99
- SLO 被违反之前的最大 QPS 即为单实例容量

#### 5.14.5 常见配置错误

第一类错误是对小模型使用过大的 TP：
- 8B 模型的 BF16 权重为 16 GB，单块 GPU 即可容纳
- TP=8 时每层要做两次 AllReduce，而每块 GPU 上的矩阵乘法规模很小
  - 计算时间与通信时间相当，通信占比过高
- 改为 8 个 TP=1 的独立副本
  - 每个副本没有通信开销，总吞吐通常高得多
  - 只有在单副本延迟无法满足 SLO 时，才考虑 TP=2 等较小的并行度

第二类错误是 `max-model-len` 设得过大：
- 该参数决定单个请求的 KV 上限，也影响调度器的准入判断
- 设为 128K 时，一个长请求最多可占用约 40 GB KV（Llama-3-70B，BF16）
  - 会挤占大量并发请求的空间
  - 并在 KV 紧张时触发频繁抢占
- 应按业务的实际上下文需求设定，超长请求路由到专门配置的实例

第三类错误是压测数据与线上分布不一致：
- `random` 数据集的输入不共享前缀，前缀缓存的命中率为零
  - 线上请求则可能有很高的系统提示词命中率
- 输入、输出长度的分布形状（尤其是长尾）也会显著影响 KV 占用和抢占频率
- 用固定长度的合成数据得到的容量，可能与线上相差数倍
  - 应尽量用线上日志回放或按线上统计分布生成压测数据

### 5.15 推理侧的确定性与 batch invariance

#### 5.15.1 温度为 0 时输出仍不确定的原因

- temperature=0 即贪心解码，通常被认为输出确定
- 但在推理服务上，同一个 prompt 多次请求得到的输出可能不同
  - Thinking Machines 的实验中，一个大型 MoE 模型在温度为 0 时采样 1000 次，得到 80 种不同的结果

常见解释与根本原因：
- 常见解释是「GPU 并行计算加上浮点加法不满足结合律」，但单个 kernel 在相同输入形状下通常是确定的
- 根本原因在于 batch invariance 不成立
  - Batch invariance 是指同一条请求的计算结果与它所在 batch 的大小和组成无关
  - 服务端的 batch 大小由当时的并发请求数决定
  - batch 大小变化后，kernel 选择的切分策略（split-K、注意力的 split-KV、归约分块）随之变化，同一条请求的数值结果也随之改变
  - 当两个候选 token 的 logits 非常接近时，argmax 可能翻转，此后的整段输出都会不同

#### 5.15.2 需要满足 batch invariance 的三类 kernel

| Kernel | 不变性被破坏的原因 | 修复方法 |
|---|---|---|
| RMSNorm | batch 小时为占满 SM，改为由多个 block 共同归约一行 | 始终由一个 block 处理一行（data-parallel），放弃小 batch 下的额外并行度 |
| MatMul | batch 小时用 split-K 增加并行度，归约顺序随之变化 | 固定 tile 配置，不使用 split-K |
| Attention | decode 时按 KV 长度切分（split-KV），切分方式取决于 batch 和已缓存长度；prefill、decode 和 chunked prefill 的边界也影响归约顺序 | 固定每个 split 的长度（而非 split 的数量），使同一 token 位置的归约顺序与 batch 和 chunk 划分无关 |

#### 5.15.3 性能开销

- Thinking Machines 的初始实现约慢 60%
- SGLang 的确定性推理模式（`--enable-deterministic-inference`）经优化后报告的开销约为 34%
- vLLM 也提供 batch invariant 运行模式

#### 5.15.4 跨 TP 大小的确定性

- 即使满足 batch invariance，训练侧（如 TP=8 的 Megatron）与推理侧（如 TP=2 的 vLLM）的张量切分方式不同
  - 行切分矩阵乘中部分和的累加顺序仍然不同
- *Deterministic Inference across Tensor Parallel Sizes*（2025）采用固定的树形归约顺序
  - 使不同 TP 大小下得到逐位相同的结果
- 原理见 4.10.6 节

#### 5.15.5 与 RL 训推一致性的关系

- On-policy RL 中，rollout 由推理引擎生成，训练时再由训练框架重新计算这些 token 的 logprob
- 若两侧数值不一致，名义上的 on-policy 训练实际上变成了 off-policy：

$$\frac{\pi_{\text{train}}(a\mid s)}{\pi_{\text{rollout}}(a\mid s)} \ne 1$$

- MoE 模型上这一差异更大，因为两侧的路由可能选中不同的专家
- 这种不一致会导致训练不稳定，甚至 reward 崩溃

处理方法有两类：

1. 使两侧数值一致：采用 batch invariant kernel、跨 TP 确定性归约，以及相同的 kernel 实现；
2. 在算法层面修正：用重要性采样比值修正梯度，并截断过大的比值（truncated importance sampling），或在 MoE 上回放 rollout 时的路由结果（routing replay）。

8.8 节将在 RL 系统的整体框架中讨论这一问题。

### 5.16 Agent 工作负载的推理服务

#### 5.16.1 负载特点

- Agent 工作负载是指由模型驱动、包含多轮推理与外部工具调用的任务
- 一个典型的 Agent 任务依次执行：规划，调用搜索工具，等待返回，阅读结果，调用代码执行器，等待返回，再次规划，最后生成报告

与单轮对话相比，它对推理系统有三方面不同：
- 第一，上下文持续增长且高度重复
  - 每一步的 prompt 包含系统提示词、工具说明和此前所有步骤的记录
  - 相邻两步的 prompt 只在末尾相差一次工具返回的内容
  - 前缀缓存的命中率可以超过 90%，KV 的复用价值很高
- 第二，请求之间存在不可预测的空闲间隔
  - 模型生成一次工具调用后，推理侧就进入等待，等待时间取决于外部工具
  - 一次 API 调用可能 200 ms，一次网页检索可能 20 s，一次代码编译与测试可能数分钟
  - 这些间隔内，该任务的 KV 不被读取，但很可能在间隔结束后马上被复用
- 第三，端到端目标是任务完成，而不是单次请求的延迟
  - 一个任务可能包含几十次模型调用
  - 评测指标除 token/s 外，还应包括任务完成率、端到端完成时间的 P95、重试次数和每个任务的成本

#### 5.16.2 五类状态

Agent 任务涉及多种状态，它们的生命周期、大小和存储要求不同，不应混为一谈：

| 状态类型 | 内容 | 生命周期 | 存放位置与要求 |
|---|---|---|---|
| KV cache | 模型推理过程中的 Key/Value 张量 | 任务进行期间，可随时重算 | HBM 或分层 KV 存储，丢失后可由文本重建 |
| 工作记忆 / 任务状态 | 计划、工具调用结果、中间变量、任务进度 | 任务进行期间 | Agent 框架的内存或数据库，以文本或结构化数据存储 |
| 检查点 / 快照 | 为长时间任务的故障恢复而持久化的完整状态 | 任务进行期间，需在故障后保留 | 持久化存储，需保证一致性 |
| 产出物（artifact） | 最终的文件、报告、代码、图像 | 任务结束后长期保留 | 对象存储或文件系统，需可靠存储 |
| 长期记忆 | 跨任务可检索的知识与用户偏好 | 跨任务长期存在 | 向量数据库或检索系统，以文本形式存储 |

- 推理系统直接管理的只有 KV cache
- KV 是工作记忆在特定模型下的派生表示
  - 给定相同的 token 序列和相同的模型，KV 总可以重新计算出来
  - 因此 KV 丢失只损失性能，不损失正确性
  - 工作记忆、检查点和产出物的丢失会导致任务失败
- 长期记忆与 KV cache 的区别在于
  - 前者是按语义检索后以文本形式重新输入模型的
  - 不能直接复用为 KV

#### 5.16.3 工具调用等待期间的 KV 处置

模型发出工具调用后，该任务的 KV 有三种处置方式：
- 保留在 HBM 中
- 卸载到 CPU DRAM 或远端存储
- 直接丢弃

决策依据是 5.4.7 节的迁移与重算比较，加上对等待时长的估计。

以 Llama-3-70B 上下文 32K token 的任务为例，KV 约 10.7 GB（BF16）：
- 若工具调用等待 20 s
  - 保留在 HBM 中意味着这 10.7 GB 在 20 s 内无法服务其他请求
  - 卸载到 CPU DRAM 和取回各需约 30 ms（5.4.7 节），与 20 s 的等待相比可以忽略
  - 丢弃后重算需 1 s 以上，并占用 prefill 算力
  - 此时卸载最优
- 若工具调用只等待 200 ms
  - 卸载与取回的 60 ms 已占等待时间的 30%，还要与其他传输争用 PCIe 带宽
  - 保留在 HBM 中更合理
- 若任务大概率不再继续（例如用户已经离开）
  - 则直接丢弃，等待逐出策略回收

单一的逐出策略（如 LRU）不适合 Agent 负载：
- LRU 在等待 20 s 后可能已逐出该任务的 KV，而它恰恰即将被复用
- 同时又可能长时间保留已结束任务的 KV
- 更合理的做法是让 Agent 框架向推理系统传递提示信息
  - 例如「本请求结束后预计等待某工具，预计耗时若干秒」
  - 由推理系统据此选择处置方式

#### 5.16.4 基于预测的预取

- 卸载之后，KV 应在任务恢复前提前取回
  - 否则恢复时的第一次请求要承担取回延迟
- 预取的难点在于 Agent 的下一步动作和恢复时间都不确定
- 可利用的预测信息包括：
  - 已知工作流的结构（例如固定的「检索、阅读、总结」流程）
  - 各类工具的历史耗时分布
  - 当前任务的缓存命中统计
  - 工具调用的模式
- 例如某个检索工具的 P50 耗时为 3 s，则可以在工具调用发出 2.5 s 后开始预取该任务的 KV

预测错误有代价：
- 过早预取的 KV 在 HBM 中闲置，占用了本可服务其他请求的容量
- 预测会继续而实际未继续时，预取浪费了带宽
- 因此只有当预期收益（节省的取回延迟乘以预测准确率）超过带宽与容量的开销时才应预取
- 在动态性很强的 Agent 工作流中如何做出可靠预测，目前仍是开放问题

#### 5.16.5 Agent 框架、推理运行时与系统软件的分工

Agent 系统可以分为三层，各自负责不同的问题：
- Agent 框架（agent harness）负责任务如何组织
  - 包括工具调用循环、每一步的上下文组装、错误处理与重试、状态转移、权限控制和工作流控制
  - 它掌握任务级的信息，例如下一步要调用哪个工具、该工具通常耗时多久
- 推理运行时（serving runtime）负责模型如何高效运行
  - 包括模型加载、GPU 调度、batching、KV cache 管理和前向执行，即本章前面各节讨论的内容
  - 它掌握请求级和硬件级的信息，例如当前 KV 占用和各副本的负载
- 系统软件负责数据与设备如何协同
  - 包括内存管理、跨设备通信、设备驱动、集群资源调度和故障恢复

5.16.3 和 5.16.4 节的优化需要跨层协作：
- 框架知道任务何时恢复
- 运行时知道 KV 在哪里、HBM 是否紧张
- 系统软件提供 DMA/RDMA 传输与存储能力

各层各自独立做贪心决策时，常常得到次优结果：
- 例如框架为降低延迟而把同一任务的请求发往不同副本，导致前缀缓存全部失效
- 联合优化请求路由、GPU 调度和 KV 放置，需要一个覆盖各层的端到端成本模型

#### 5.16.6 按请求类别设定 SLO

Agent 任务中的模型调用对延迟的要求并不相同：
- 直接面向用户、流式输出的最终回答
  - 需要与对话场景相同的 TTFT 和 TPOT 目标
- 用户看不到的中间步骤（规划、工具参数生成、结果摘要）
  - 其延迟只影响任务的总完成时间
  - 可以采用更宽松的 TPOT 目标，换取更大的 batch 和更高的吞吐
- 后台运行的长时间任务
  - 可以只设定总完成时间的目标

按类别设定 SLO 之后：
- 调度器可以采用 5.9.3 节的分级队列
  - 交互请求优先
  - 中间步骤和后台任务填充剩余容量
- 评测 goodput 时，也应分别统计各类请求的 SLO 达成率
  - 而不是对所有请求使用同一阈值

### 5.17 小结

- prefill 受算力限制，决定 TTFT；decode 受带宽限制，决定 TPOT。
- continuous batching 与 PagedAttention 是现代推理引擎的基础，chunked prefill 用于减轻两阶段之间的干扰。
- 前缀缓存在不增加计算的前提下提高吞吐，多副本部署时需配合 cache-aware 路由。
- 量化按读取字节数的减少直接加速 decode；激活离群值由 SmoothQuant / AWQ 处理；KV 通常采用 FP8。
- 投机解码严格保持输出分布不变，加速比为 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$，适用于小 batch 场景。
- PD 分离消除两阶段干扰，需要 RDMA 网络传输 KV。
- 容量规划：先计算两阶段的 token 需求，再分别按算力和带宽 / 显存容量估算节点数，取约束更紧的结果。
- 温度为 0 时输出不确定的根源是 batch invariance 被破坏，RL 的训推一致性依赖于它。
- KV 在 HBM、CPU DRAM、SSD 与远端存储之间的管理由卸载、逐出、预取三种操作组成
  - 取回还是重算，取决于传输时间、能否与计算重叠以及 HBM 的机会成本
- Agent 负载上下文重复度高、存在不可预测的工具等待，需要 Agent 框架与推理运行时协同决定 KV 的保留、卸载与预取，并按请求类别设定 SLO。

<br>

## 6. 编译与图优化

第 2 章介绍的算子融合依赖手写 kernel（kernel 是在 GPU 上并行执行的一个函数，见 2.1）：
- 一个模型通常包含几十种算子组合
  - 逐一手写融合 kernel 并不可行
- 深度学习编译器以整张计算图为输入，自动完成算子融合、kernel 选择与参数调优
- 编译器同时消除 Python 解释执行和 kernel launch 带来的开销
  - kernel launch：CPU 向 GPU 提交一次 kernel 执行请求的操作

### 6.1 从 eager 模式到图模式：图捕获与形状特化

#### 6.1.1 eager 模式与图模式

两种执行模式的定义：
- eager 模式（eager mode）：每个算子在 Python 代码执行到它时立即发射对应 kernel 的执行方式
  - PyTorch 默认采用这种模式
- 图模式（graph mode）：先把程序表示为一张完整的计算图，对整张图做优化后再执行

两者的差异如下：

| | Eager | Graph |
|---|---|---|
| 执行方式 | Python 逐行执行，每个算子立即发射 kernel | 先获得整张图，优化后再执行 |
| 优点 | 易于调试，支持任意 Python 控制流 | 具备全局视野，可做跨算子优化 |
| 缺点 | 执行当前算子时无法得知后续算子，因而无法融合；存在 Python 解释开销 | 需要先捕获图，对动态行为的支持受限 |

两者的根本区别在于优化时可见的范围：
- eager 模式下，框架在执行 `y = x + b` 时并不知道下一行是 `z = gelu(y)`
  - 因此只能为加法单独发射一个 kernel，把 `y` 完整写回 HBM，再由 gelu 的 kernel 读出
  - 以形状为 [4096, 4096] 的 BF16 张量为例，单个中间结果占 32 MiB
  - 一次写回加一次读出就产生 64 MiB 的 HBM 流量
  - 这部分流量对最终结果没有贡献
- 每个算子都要经过 Python 解释器、PyTorch 的算子分发和一次 kernel launch
  - 单个算子的 CPU 侧开销在几微秒到十几微秒之间
  - 算子数量多、单个 kernel 执行时间短时，这部分开销会占据可观的比例
- 图模式先拿到整段计算
  - 编译器可以把加法和 gelu 合并为一个 kernel，`y` 只在寄存器中存在
  - 同时把多次 launch 合并为一次
- 图模式的代价：程序必须先被转换成图
  - Python 的任意控制流和副作用不一定能用图表示

#### 6.1.2 图捕获的方式

图捕获（graph capture）指把一段 Python 程序中的张量运算提取为计算图的过程。
- PyTorch 中常用的图表示是 FX 图（FX graph）
  - 一种以算子调用为节点、以张量数据依赖为边的中间表示

主流的捕获方式有四种：

| 方式 | 代表 | 原理 | 问题 |
|---|---|---|---|
| Tracing | `torch.jit.trace`、JAX `jit` | 用示例输入执行一遍，记录经过的算子 | 数据相关的控制流被固化为示例输入所走的分支 |
| 脚本解析 | `torch.jit.script` | 解析 Python 源码 | 只支持 Python 的一个子集 |
| 字节码捕获 | TorchDynamo（`torch.compile`） | 在 CPython 执行帧时拦截字节码，把其中的 PyTorch 操作提取为 FX 图 | 遇到无法捕获的代码时产生 graph break（6.1.3） |
| 惰性张量 | PyTorch/XLA LazyTensor | 先记录操作，直到需要结果时再编译执行 | 每次形状变化都可能触发重编译 |

Tracing 的局限来自它只观察一次执行：
- 若函数中有 `if x.sum() > 0: y = x * 2 else: y = -x`，用某个示例输入追踪时只会走其中一条分支
  - 生成的图永远执行这条分支
  - 即使之后输入的和变为负数，结果也不会切换，并且不会报错

脚本解析通过分析源码保留了控制流：
- 它要求代码只使用 TorchScript 支持的 Python 子集
- 调用 NumPy、使用复杂的数据结构或动态类型时需要改写代码

TorchDynamo 的思路是在更低的层次上工作：
- CPython 执行每个函数时会创建一个执行帧（frame），其中包含该函数的字节码
- Dynamo 通过 PEP 523 提供的帧求值接口，在字节码被解释之前接管执行
  - 逐条分析字节码，把其中的张量运算收集为 FX 图
  - 对非张量的普通 Python 代码保持原有语义
- 用户代码因此无须修改，只在遇到无法分析的部分时才截断图（6.1.3）

惰性张量在每个算子调用时只记录操作而不执行：
- 直到程序需要具体数值时，才把积累的操作序列编译为一张图并执行
  - 例如调用 `.item()`、打印张量或显式调用同步点
- 每次记录的序列都依赖当次的形状
  - 形状频繁变化时会反复编译

#### 6.1.3 Graph break 的定义与成因

graph break 指图捕获过程中遇到无法表示为图的代码时，计算图在该处被截断的现象。
- Dynamo 遇到这类代码时，把图切成前后两段分别编译，中间这段代码回到 eager 模式执行
  - 打印张量的值
  - 调用不受支持的第三方库
  - `if tensor.item() > 0` 这类依赖张量数据的分支
- graph break 越多，每张子图覆盖的算子越少，跨算子融合的机会也随之减少

可以用下面的方式统计 graph break 的数量和原因：

```python
import torch
torch._dynamo.config.verbose = True
explanation = torch._dynamo.explain(model)(x)
print(explanation.graph_break_count, explanation.break_reasons)
```

`break_reasons` 会给出每个断点的源码位置和原因。消除 graph break 的常见方法：
- 把数据相关的分支改写为张量运算
  - 例如用 `torch.where(cond, a, b)` 代替 `if cond.item(): ... else: ...`
  - 这样两条分支都在 GPU 上计算，由掩码选择结果
  - 图不再依赖某个运行时数值
- 调试用的打印语句应在编译前移除
- 必须调用的外部库函数可以封装为自定义算子注册给 PyTorch
  - 使 Dynamo 把它当作图中的一个不透明节点
- 对于返回标量的 `.item()`，可以设置 `torch._dynamo.config.capture_scalar_outputs = True`
  - 让 Dynamo 把该标量作为符号值保留在图中，而不是截断图

#### 6.1.4 静态形状、动态形状与重编译

编译生成的 kernel 针对特定的输入形状进行了特化。相关概念：
- 静态形状（static shape）：编译时各维大小均为确定常数
- 动态形状（dynamic shape）：某些维度在编译时被当作符号变量、运行时才确定具体值
- 重编译（recompilation）：已编译的图因输入条件变化而失效、需要重新捕获和编译的过程

形状变化时的处理方式：
- 当 batch 大小或序列长度发生变化时，静态编译的结果需要重编译
- 另一种做法是编译成支持动态形状的版本
  - 其性能通常略低于静态版本
- `torch.compile` 在 `dynamic=None`（默认值）下的行为
  - 第一次遇到某个形状时按静态形状编译
  - 第二次遇到该维度的不同取值时，把这一维标记为动态并重编译
- 推理服务中常用的做法是把形状对齐到若干档位
  - 例如 batch 取 1, 2, 4, 8, ..., 512
  - 每个档位编译一次

静态形状带来的性能优势来自编译器可以利用的确定信息：
- 循环次数已知，可以完全展开或选择恰好整除的 tile 大小
- 不需要生成处理边界的掩码分支
- 缓冲区大小在编译期确定，可以预先规划内存

动态形状与重编译的代价：
- 动态形状版本必须对任意长度都正确
  - 通常需要在运行时计算循环边界并保留越界检查
- 重编译本身也有成本
  - 一个中等规模模型的完整编译可能需要数十秒到数分钟
  - 在线服务中途触发重编译会造成明显的延迟尖峰

形状分档的代价是填充（padding）浪费：
- 若实际 batch 为 37，对齐到 64 档后有 27 行是无效计算
  - 浪费比例为 27/64 ≈ 42%
- 档位越密，浪费越小，但需要编译和缓存的版本越多
  - 因此实际系统在小 batch 区间按 2 的幂分档，在较大区间改用固定步长（如每 8 或 16 一档）
- 已知某一维必然变化时，可以用 `torch._dynamo.mark_dynamic(x, 0)` 在首次编译前就把第 0 维标记为动态
  - 避免先静态编译、再重编译的过程

#### 6.1.5 守卫（guard）

守卫（guard）是 Dynamo 为每张已编译的图记录的一组前提条件，包括输入的 dtype、形状、设备以及部分 Python 变量的值。
- 每次调用时先检查 guard
  - 全部满足则复用缓存中的已编译图
  - 否则重新捕获并编译
- 同一段代码的重编译次数超过上限（`torch._dynamo.config.cache_size_limit`）后，Dynamo 不再编译，直接回退到 eager 模式执行

guard 之所以需要覆盖 Python 变量：
- Dynamo 在捕获时会把非张量的 Python 值当作常量折叠进图中
- 例如函数中有 `if self.training: x = dropout(x)`
  - 捕获时 `self.training` 为 True，图中就包含 dropout
  - 若之后调用 `model.eval()`，guard 检测到该值变化，触发重新捕获
  - 得到一张不含 dropout 的图

guard 的开销与排查：
- guard 的检查发生在每次调用时，本身也有微秒级的 CPU 开销
  - guard 数量过多会抵消一部分编译收益
- 排查重编译原因可以设置环境变量 `TORCH_LOGS="recompiles"`
  - 日志会指出是哪一条 guard 失败

#### 6.1.6 静态调度与动态调度

两种调度方式的定义：
- 静态调度（static scheduling）：在编译期就确定算子的执行顺序、计算资源的分配以及数据在各级存储之间的搬运时机
- 动态调度（dynamic scheduling）：在运行时根据数据依赖是否满足、硬件资源是否空闲来决定下一步执行什么
- 两者是一组设计取舍，而不是非此即彼

静态调度的优势在于可预测性和硬件简化：
- 编译器掌握完整的计算图和每个张量的大小后，可以精确规划
  - 每块缓冲区的地址
  - 每次数据搬运的起止时刻
  - 计算与搬运的重叠方式
- 硬件不再需要复杂的运行时控制逻辑
  - 例如乱序发射、动态依赖检测
  - 节省的晶体管和功耗可以用于计算单元
  - 执行时间也更加稳定
- XLA 面向 TPU 的编译路径偏向这一端
  - 它要求编译期形状已知，正是为了在编译时完成上述规划
- GPU 则在硬件层面保留了较强的动态调度能力
  - warp 调度器在每个周期从就绪的 warp 中选择一个发射指令
  - 某个 warp 等待访存时自动切换到其他 warp
  - 这种机制容忍了不规则的访存延迟
  - 但需要额外的调度硬件和大容量寄存器文件来保存所有驻留 warp 的状态

静态调度的难点集中在动态性上：
- 动态形状、数据相关的分支和不规则负载都会使编译期的规划失效
  - 不规则负载的例子：MoE 中每个专家接收的 token 数量随输入变化
- 需要额外的机制来处理
  - 按档位编译多个版本（6.1.4）
  - 把输入填充到固定大小
  - 把动态部分留在运行时调度
  - 或者在硬件中保留部分动态控制能力
- 实际的加速器架构也并非所有操作都完全静态
- 本章介绍的形状分档和 CUDA Graph（6.6）都可以理解为在动态负载下尽量恢复静态调度优势的手段
  - 前者把无限多的形状收敛为有限个静态版本
  - 后者把运行时逐个提交的 kernel 序列固定为一次性提交的静态图

### 6.2 算子融合与模式匹配

算子融合（fusion）指把多个算子合并为一个 kernel：
- 中间结果保留在寄存器或共享内存中，而不必写回 HBM 再读出（2.2）
  - HBM：GPU 的片外高带宽显存
- 编译器在图上自动识别可融合的算子组合
  - 目的是减少 HBM 的读写字节数

#### 6.2.1 融合的分类

下表涉及的概念：
- 生产者：产生某个张量的算子
- 消费者：读取该张量的算子
- 纵向融合（vertical fusion）：合并存在生产者与消费者关系的算子
- 横向融合（horizontal fusion）：合并互不依赖但读取相同输入的算子
- epilogue：附加在 GEMM 末尾、对其输出直接做的逐元素运算

| 类型 | 规则 | 例子 |
|---|---|---|
| 逐元素 + 逐元素（vertical） | 生产者和消费者形状相同 | add → gelu → mul |
| 逐元素 + 归约 | 归约的输入由逐元素算子产生 | `(x*x).sum(-1)` |
| 归约 + 逐元素 | 归约结果广播后参与逐元素运算 | softmax、RMSNorm |
| 水平融合（horizontal） | 互不依赖、读同一输入的算子合并 | Q、K、V 三个投影合成一个 GEMM |
| GEMM + epilogue | GEMM 之后的逐元素算子 | matmul + bias + relu |

各类融合节省的 HBM 流量可以逐一计算。以长度为 N 的 BF16 张量为例：

纵向逐元素融合：
- `t1 = a + b; t2 = gelu(t1); y = t2 * c` 不融合时需要三个 kernel
  - 加法读 a、b 写 t1
  - gelu 读 t1 写 t2
  - 乘法读 t2、c 写 y
  - 共访问 7N 个元素
- 融合后只读 a、b、c 并写 y，共 4N 个元素
  - HBM 流量降为原来的 4/7
  - launch 次数从 3 次降为 1 次
- 逐元素算子的计算强度很低，执行时间几乎完全由访存决定
  - 因此流量的比例基本就是耗时的比例

逐元素与归约融合：
- `(x*x).sum(-1)` 不融合时先把 `x*x` 完整写回 HBM，再由归约 kernel 读出
- 融合后每个线程读入 x 的元素后直接平方并累加
  - 中间张量不会出现在 HBM 中
  - 流量从约 3N 降为约 N

归约与逐元素融合：
- softmax 需要先求每行的最大值和指数和，再对每个元素做归一化
- RMSNorm 需要先求每行的平方均值，再把每个元素乘以缩放因子
- 当一行能够放进单个线程块的寄存器或共享内存时
  - 例如 hidden 维为 4096、BF16 时一行仅 8 KiB
  - 整行在片上完成归约后可以直接用于逐元素计算
  - 每行只从 HBM 读一次、写一次

横向融合：
- Q、K、V 三个投影共享输入 X
  - 分别计算时 X 要被读三遍、发射三个 GEMM
- 把三个权重矩阵沿输出维拼接为一个矩阵后，只需一个 GEMM 读一遍 X
- 更大的 GEMM 也能产生更多的线程块
  - 更充分地占满 GPU 的所有 SM

GEMM epilogue 融合：
- GEMM 的结果在写回 HBM 之前位于累加寄存器中
- 此时直接加上 bias 并执行 relu，再一次性写出
  - 可以消除输出矩阵的一次完整写回和一次完整读出
  - 对于 [8192, 8192] 的 BF16 输出，单次往返约为 256 MiB

#### 6.2.2 模式匹配（pattern matching）

模式匹配（pattern matching）指编译器在计算图中查找与预定义模板一致的子图，并将其替换为等价的高性能实现。
- 部分融合无法由通用规则推导得到
- 编译器为此维护一个模式库，常见条目如下：

| 模式 | 替换成 |
|---|---|
| matmul → div → softmax → matmul | `scaled_dot_product_attention`（FlashAttention） |
| 三个共享输入的 Linear | 一个拼接权重的 GEMM |
| 小 GEMM 序列 | 批量 GEMM |

注意力模式的替换收益最为显著：
- 按定义计算注意力时，`Q @ K^T` 的结果是一个 S×S 的分数矩阵
  - 需要写回 HBM 后再做 softmax，再读出与 V 相乘
  - 序列长度 S = 8192 时，单个注意力头的 BF16 分数矩阵为 8192² × 2 B = 128 MiB
  - 32 个头合计 4 GiB
  - 仅这一个中间结果的读写就远超 Q、K、V 本身
- FlashAttention 按块计算并在片上维护 softmax 的运行统计量
  - 分数矩阵从不写回 HBM
- 通用的融合规则无法推导出这种改写
  - 它依赖 online softmax 这一数学变换
  - 所以只能以模式的形式写入编译器

权重拼接和批量 GEMM 的作用与 6.2.1 中的横向融合相同：
- 减少 kernel launch 次数
- 把多个小矩阵乘合并成一个能够占满全部 SM 的大任务
- 例如 MoE 中每个专家处理的 token 很少
  - 单独发射几十个小 GEMM 时每个 GEMM 只占用少数 SM
  - 合并为一次分组 GEMM 后利用率显著提高

模式匹配要求替换前后的数值语义一致：
- 例如注意力模式中除以 √d 的缩放、可选的因果掩码和 dropout 都必须与目标实现的参数逐一对应
  - 否则编译器不会执行替换
- 模型代码的写法稍有不同就可能导致匹配失败
  - 例如先缩放 Q 再相乘，而不是先相乘再缩放
  - 这时直接调用 `scaled_dot_product_attention` 是更可靠的做法

#### 6.2.3 融合的代价与判定

融合并非在所有情况下都能提升性能，主要有两类代价。

第一类是寄存器压力：
- 融合后的 kernel 要同时保存多个算子的中间值
  - 每个线程需要的寄存器数随之增加
  - occupancy（每个 SM 上同时驻留的 warp 数占硬件上限的比例）随之下降
- 以 H100 为例，每个 SM 有 65536 个 32 位寄存器，最多驻留 64 个 warp
  - 若每个线程使用 255 个寄存器（按分配粒度计为 256 个）
  - 一个 SM 最多容纳 65536 / 256 = 256 个线程，即 8 个 warp
  - occupancy 只有 12.5%
- 驻留 warp 越少，某个 warp 等待访存时可供切换的其他 warp 就越少
  - 访存延迟越难被隐藏
- 寄存器不足时编译器还会把部分变量溢出（spill）到本地内存
  - 本地内存实际位于 HBM，反而增加访存

第二类是重复计算：
- 若某个中间结果 t 被两个消费者使用，而两个消费者分别融合到不同的 kernel 中
  - t 就会在两个 kernel 内各计算一次
- t 是逐元素运算时，重算的计算量远小于节省的访存，这样做是划算的
- t 是 GEMM 或归约时，重算代价可能超过收益

编译器使用代价模型（cost model）做判定：
- 代价模型是一个对候选方案的执行时间进行估算的函数
- Inductor 的做法是估计融合前后的 HBM 访问字节数和计算量
  - 并检查融合是否会引入重复计算或超出寄存器预算
  - 据此决定是否融合

### 6.3 主流编译器：Inductor、XLA、TVM

#### 6.3.1 TorchInductor

TorchInductor 是 `torch.compile` 的默认后端。整条编译链路分为三个阶段：

```text
Python 代码
  └─ TorchDynamo：捕获 FX 图
      └─ AOTAutograd：生成前向 + 反向的联合图，做分解（decomposition）和重算决策
          └─ TorchInductor：
               ├─ 降低到 loop-level IR（define-by-run 的循环表示）
               ├─ 调度与融合
               └─ 代码生成：GPU → Triton；CPU → C++/OpenMP；GEMM → cuBLAS 或 Triton/CUTLASS 模板
```

各阶段涉及的概念如下：
- AOTAutograd：在编译期（ahead of time）对前向图执行自动微分、生成前向与反向联合图的组件
- 分解（decomposition）：把复合算子改写为少量基础算子的组合
  - 以缩小后端需要支持的算子集合
- IR（intermediate representation，中间表示）：编译器内部在源程序与目标代码之间使用的程序表示
- 降低（lowering）：把高层 IR 逐步转换为更接近硬件的低层 IR 的过程
- 代码生成（codegen）：由最低层 IR 输出可编译的目标代码
  - Inductor 在 GPU 上输出的是 Triton 源码
  - Triton 是以 block 为编程单位的 GPU kernel 语言，见 2.6

Inductor 的 loop-level IR 把每个算子表示为「给定输出下标，如何计算该位置的值」的函数，而不是一次完整的张量运算：
- 在这种表示下，融合生产者与消费者就是把生产者的函数内联到消费者的循环体中
  - 消费者在计算第 i 个输出时直接调用生产者的函数求出所需的输入值
  - 中间张量因此不再需要分配缓冲区
- define-by-run 指这些函数以普通 Python 代码的形式定义
  - Inductor 通过执行它们来记录具体的计算步骤

AOTAutograd 还会执行基于 min-cut 的重算决策：
- 在前向与反向之间寻找一个图切分
  - 使保存到反向的激活字节数与反向中需要重新计算的计算量之和最小
  - 由此实现自动的选择性重算
- 典型的结果是 GEMM 的输出被保存下来
  - 因为重新计算 GEMM 代价很高
- 其后的 gelu、dropout 掩码应用等逐元素算子的输出不保存，在反向中由已保存的 GEMM 输出重新算出
  - 这类运算的重算几乎不增加耗时
  - 却能节约与激活同等大小的显存

#### 6.3.2 XLA

XLA（Accelerated Linear Algebra）是 Google 开发的编译器：
- 它是 JAX 和 TensorFlow 的后端，也通过 PyTorch/XLA 支持 PyTorch
- 它的 IR 称为 HLO（High Level Operations）
  - StableHLO 是其版本化、可跨框架交换的形式
- GSPMD 是 XLA 中的自动分片机制
  - 用户在少量张量上标注分片方式
  - 编译器把分片信息传播到全图并插入所需的集合通信

| 特点 | 说明 |
|---|---|
| IR | HLO / StableHLO，算子集合小而规整 |
| 融合 | 激进的融合（loop fusion、input fusion），以静态形状为前提 |
| 分布式 | GSPMD：在张量上标注分片，编译器自动插入通信 |
| 硬件 | TPU 的原生编译路径，也支持 GPU |

HLO 的算子集合只包含数量有限的基础操作：
- 包括逐元素运算、广播、归约、点积、卷积、切片、拼接等
- 所有高层算子都先分解为这些操作
  - 因此融合规则只需针对这个小集合编写

XLA 中的两种融合：
- loop fusion：把具有生产者与消费者关系的逐元素操作合并为一个循环
- input fusion：把归约操作的输入端的逐元素操作并入归约 kernel
  - 对应 6.2.1 中的「逐元素 + 归约」

XLA 要求编译期形状已知：
- 它会在编译时完成全部缓冲区的分配和复用规划（6.3.5）
- 形状变化即意味着重新编译
  - JAX 用户通常通过填充到固定长度来规避

GSPMD 的使用方式可以用一个例子说明：
- 在 4 块 GPU（或 TPU 芯片）上，用户把权重 W（形状 [d, 4d]）标注为沿第 1 维切成 4 份
- 编译器据此推断出 `X @ W` 的输出也沿最后一维切分
- 若后续算子需要完整的输出，编译器会自动在该处插入 all-gather
- 若下一个矩阵乘的权重沿第 0 维切分，则输出是部分和
  - 编译器插入 reduce-scatter 或 all-reduce
- 用户只写单设备的计算逻辑，切分和通信由编译器推导

#### 6.3.3 TVM

Apache TVM 采用「张量表达式 + 调度原语 + 自动搜索」的技术路线：
- 其核心是把「计算什么」与「如何计算」分开描述

张量表达式描述计算的数学定义，即输出的每个元素如何由输入得到：
- 矩阵乘写作 `C[i, j] = sum(A[i, k] * B[k, j], axis=k)`
- 其中不包含任何关于循环顺序、并行方式或存储位置的信息

调度（schedule）描述计算的执行方式，由一系列调度原语组合而成：
- tiling：把 i、j 循环各切成大小为 64 的块
  - 使每块数据能够放入共享内存
- reorder：调整嵌套循环的次序
  - 使最内层访问连续地址
- vectorize：让最内层循环使用向量指令一次处理多个元素
- bind：把外层循环映射到 GPU 的线程块下标（blockIdx），把内层循环映射到线程下标（threadIdx）
- 同一个张量表达式配合不同的调度，可以生成性能相差数倍的代码

AutoTVM、Ansor、MetaSchedule 在调度空间中自动搜索：
- 调度空间是所有合法调度方案构成的集合
  - 属于 6.4 中 autotuning 搜索空间（search space）的一种
- 每个循环的切分大小有多种选择，多个循环的组合再乘以循环次序和绑定方式
  - 一个 GEMM 的调度空间规模可以达到百万量级以上
  - 逐一实测不可行
- Ansor 和 MetaSchedule 的做法是训练一个代价模型预测候选方案的执行时间
  - 只把预测排名靠前的少量方案放到真实硬件上测量
  - 测量结果再用于更新代价模型

TVM 的适用范围：
- TVM 的优势在于跨硬件支持（CPU、移动端和各类加速器）
  - 同一份张量表达式配合针对不同硬件搜索出的调度，即可在不同设备上获得较好的性能
  - 无须为每种硬件手写 kernel
- 在 NVIDIA GPU 上的大模型场景中，Triton 和厂商库的使用更为普遍
  - 因为它们对 Tensor Core 等新硬件特性的支持更及时
- TVM 的设计思想延续到了 MLC-LLM 等端侧推理项目中

#### 6.3.4 三者对比

| | Inductor | XLA | TVM |
|---|---|---|---|
| 前端 | PyTorch | JAX / TF / PyTorch | 多框架导入 |
| 形状 | 静态 + 动态 | 以静态为主 | 静态为主，有动态支持 |
| GPU 代码生成 | Triton | 自研 + LLVM | 自研 + 搜索 |
| 典型用户 | PyTorch 训练 / 推理 | TPU、JAX 研究 | 端侧、异构硬件 |

三者的差异可以归结为设计前提的不同：
- Inductor 以兼容既有 PyTorch 代码为首要目标
  - 因此必须容忍 graph break 和动态形状
  - 代码生成依托 Triton 以便快速覆盖大量算子组合
- XLA 假设整张图和全部形状在编译期已知
  - 换取了更彻底的全局优化，包括静态内存规划和全图分片推导
- TVM 以跨硬件为目标，用搜索代替针对每种硬件的人工调优
  - 代价是较长的调优时间

#### 6.3.5 编译器的核心任务

上述编译器的结构各不相同，但它们在把计算图映射到硬件时都要完成以下六项任务：
- 最终性能取决于这些决策与硬件特性的匹配程度
  - 因此硬件与编译器需要协同设计

计算图划分（graph partitioning）：
- 把整张计算图切分为若干子图，并为每个子图指定执行它的后端或设备
- 例如 Inductor 把 GEMM 交给 cuBLAS 或模板 kernel
  - 把逐元素和归约算子交给自身生成的 Triton kernel
- 在多设备系统中，划分还决定哪些子图在哪块 GPU 上执行
  - 流水线并行的阶段划分就是一种图划分

算子融合（fusion）：
- 在 6.2 中已经详细讨论
- 它决定子图内部哪些算子合并为同一个 kernel

内存分配（memory allocation）：
- 为计算图中的每个张量分配存储位置，并在张量的生命周期互不重叠时复用同一块缓冲区
- 编译器通过活跃性分析（liveness analysis）得知每个张量从产生到最后一次使用的区间
  - 两个区间不相交的张量可以共用地址
- XLA 在编译期完成全部规划，Inductor 也会在生成的 wrapper 中复用缓冲区
- 这一步直接决定峰值显存
  - 峰值显存又决定了单块 GPU 能容纳的 batch 大小

tile 大小选择：
- 决定把大矩阵切成多大的块进行计算
- tiling 的目的是让每块数据能放入片上存储（寄存器和共享内存）
  - 从而在片上被多次复用，减少 HBM 读写
- 以 GEMM 为例，若 tile 为 128×128、K 方向步长 BK = 32、数据类型为 BF16
  - 每一级流水线需要 A 的 128×32 块和 B 的 32×128 块，各 8 KiB，共 16 KiB
  - 4 级流水线共需 64 KiB，在 H100 每个 SM 最多 228 KiB 的共享内存范围之内
- tile 越大，数据复用率越高
  - 但占用的片上存储和寄存器越多，可同时驻留的线程块越少
- 6.4 的 autotuning 主要就是在搜索这一参数

数据布局（data layout）：
- 张量在内存中的排列方式
  - 例如行主序与列主序
  - 卷积中的 NCHW 与 NHWC
  - 为匹配矩阵计算单元而采用的分块交错布局
- 布局决定了 kernel 能否以连续、对齐的方式读取数据
  - Tensor Core 和脉动阵列都要求输入按特定的块形状组织
  - 布局不匹配时需要插入转置或重排操作，每次重排都是一次额外的 HBM 读写
- 编译器在全图范围内选择布局，目标是让布局转换的次数最少

通信调度（communication scheduling）：
- 在分布式执行中决定集合通信的时机和顺序，使通信与计算尽可能重叠
- 数据并行（DP，每块 GPU 持有完整模型、处理不同数据）的反向传播中
  - 梯度 all-reduce 可以按层分桶
  - 某一层的梯度一旦算完就立即开始通信，而不必等全部反向结束
- 张量并行（TP，把单层的权重矩阵切分到多块 GPU）中
  - 编译器可以把 all-reduce 与下一个不依赖其结果的计算并行执行

#### 6.3.6 异构硬件与可移植性

AI 计算系统中的硬件种类正在增加：
- 除 GPU 外还有 TPU、CPU、各类定制 ASIC，以及不同类型的存储和网络设备
- 不同云服务商和芯片厂商之间的架构差异也在扩大
- 异构（heterogeneity）：一个系统或一个软件栈需要同时面对多种体系结构不同的计算设备

异构给编译与运行时带来几类具体问题：
- 第一是数据布局
  - GPU 的 Tensor Core、TPU 的脉动阵列和其他 ASIC 的矩阵单元对输入块的形状和排列有不同要求
  - 同一个张量在不同设备之间传递时往往需要重排
- 第二是数据拷贝
  - 不同设备通常拥有各自独立的内存，数据跨设备时必须显式拷贝
  - 拷贝的带宽和延迟远差于设备内部访存
- 第三是路由与一致性
  - 数据经过哪条互连路径传输，以及多个设备同时访问共享数据时如何保证各自看到一致的值（coherence 与 consistency）
  - 都需要软件或硬件协议来保证
- 第四是调度
  - 同一张计算图的不同部分分配给哪类设备执行、如何安排设备之间的依赖，比单一设备上的调度复杂得多
- 随着系统规模增大，这些问题可能比单个 kernel 的性能更重要

硬件异构与软件抽象的关系：
- 硬件异构带来的是专用效率
  - 针对矩阵乘、注意力或特定数据格式设计的硬件，其能效可以显著高于通用硬件
- 软件抽象的作用则是避免把每个应用绑定在某一种设备上
  - StableHLO 这类可跨框架交换的 IR
  - 支持多家 GPU 的 Triton
  - 基于 MLIR 的多层 IR 体系
  - 它们都试图让同一份模型描述能够编译到多种后端
- 两者之间存在取舍
  - 抽象层次越高，可移植性越好，但越难利用某一种设备的独有特性
  - 因此 TensorRT-LLM（6.4.1）这类针对单一厂商硬件深度优化的库仍然有其位置
- 实际系统通常采用分层方案
  - 模型和大部分算子通过可移植的编译路径生成
  - 少数性能关键的 kernel 针对每种硬件单独实现

### 6.4 TensorRT-LLM 与 autotuning

#### 6.4.1 TensorRT-LLM

TensorRT-LLM 是 NVIDIA 的大语言模型推理库，由以下组件构成：

| 组件 | 内容 |
|---|---|
| 模型定义 | Python API 定义网络，编译成 TensorRT engine（新版本也支持 PyTorch 后端） |
| 手写 kernel | FMHA、XQA（GQA / MQA 的 decode 注意力）、MoE、量化 GEMM |
| 运行时 | in-flight batching（即 continuous batching）、paged KV、chunked prefill |
| 量化 | FP8、INT8 SmoothQuant、INT4 AWQ / GPTQ、NVFP4 |
| 并行 | TP、PP、EP，多节点 |

TensorRT engine：
- TensorRT 对一个网络完成图优化、kernel 选择和内存规划后生成的可执行文件
- 它针对特定的 GPU 型号、精度和形状范围构建
  - 换用不同型号的 GPU 时需要重新构建

与 Inductor 依靠代码生成覆盖算子组合不同，TensorRT-LLM 的性能主要来自手写 kernel：
- FMHA：融合的多头注意力 kernel，用于 prefill
- XQA：针对 GQA 和 MQA 的 decode 注意力 kernel
  - 它利用多个查询头共享同一组 K、V 的结构
  - 使一次读入的 KV 被多个查询头复用
  - 从而降低 decode 阶段的 HBM 流量

运行时组件：
- in-flight batching：允许请求在任意一步加入或离开 batch
- paged KV：以固定大小的块管理 KV cache，以减少显存碎片
- chunked prefill：把长提示词的 prefill 切成多段，与其他请求的 decode 交替执行

TensorRT-LLM 的特点是最早支持 NVIDIA 硬件的新特性：
- 包括 FP8、FP4 数据格式和新架构的 Tensor Core 指令
- 因为这些特性往往先在厂商自己的库中实现，再逐步进入通用编译器

#### 6.4.2 Autotuning 的定义

相关定义：
- autotuning（自动调优）：对一个 kernel 的若干候选配置在目标硬件上逐一实测运行时间，并选出最快配置的过程
- 搜索空间（search space）：所有候选配置构成的集合

需要实测而非直接推算，是因为 kernel 的性能对配置参数非常敏感，且难以用解析模型准确预测。以 GEMM 为例，可调参数包括：
- tile 大小（BM、BN、BK）
  - 决定每个线程块计算的输出块大小和每次从 HBM 读入的数据量
- warp 数
  - 决定每个线程块内的并行度和每个线程的寄存器负担
- 流水线级数（num_stages）
  - 决定在共享内存中同时预取多少个 K 方向的块，以便访存与计算重叠
- split-K
  - 把 K 维切成几段由不同线程块并行计算后再归约
  - 适用于 M、N 较小而 K 很大的矩阵

配置的影响：
- 同一个 GEMM 在不同配置下性能可以相差 2–3 倍
- 最优配置依赖具体的形状和硬件
- 例如 M = 16 的 decode 阶段 GEMM 若使用 BM = 128 的 tile
  - 每个线程块有 7/8 的行是填充
  - 此时较小的 BM 配合 split-K 往往更快
- M = 8192 的 prefill GEMM 则适合大 tile 以提高数据复用

#### 6.4.3 Triton autotune 示例

下面的 kernel 列出四个候选配置：
- Triton 在首次遇到某一组 (M, N, K) 时逐一运行这些配置并缓存最快的一个
- 之后相同形状的调用直接使用缓存结果

```python
import triton, triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_stages=3, num_warps=8),
        triton.Config({"BM": 64,  "BN": 256, "BK": 32}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],          # 形状变化时才重新调优
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a, mask=(rk[None, :] + k) < K, other=0.),
                      tl.load(b, mask=(rk[:, None] + k) < K, other=0.))
        a += BK * sak; b += BK * sbk
    c = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))
```

#### 6.4.4 Autotuning 的工程问题

| 问题 | 处理 |
|---|---|
| 调优耗时长 | 结果缓存到磁盘，按 (kernel, 形状, 硬件) 做键 |
| 不同机器选到不同配置 | 线上部署前统一调优，下发配置；需要确定性时固定配置（4.10.4.5） |
| 形状太多 | 按档位对齐形状，只调有限几档 |
| 测量噪声 | 多次测量取中位数，预热，清 L2 |

调优耗时的问题来自实测本身：
- 每个候选配置都要先编译，再运行数十次取统计值
- 一个 kernel 的候选配置若有几十个，一次完整调优就需要数秒
  - 整个模型的全部 GEMM 形状累计可达数分钟
- 把结果按 (kernel, 形状, 硬件) 缓存到磁盘后，同一台机器再次启动时可以直接读取

不同机器选到不同配置：
- 两个配置的实测性能可能非常接近，测量误差足以改变排序
- 不同配置的累加顺序不同（例如 split-K 的段数不同），浮点结果会有最低位上的差异
  - 多机部署时这会导致同一输入在不同机器上的输出不完全一致
- 因此线上部署通常在一台基准机器上统一调优并下发配置
- 需要逐位可复现时直接固定配置（4.10.4.5）

测量噪声的来源：
- GPU 频率随温度和功耗变化
- 首次运行时的编译和缓存未命中
- L2 缓存中残留的上一次运行的数据
  - 残留数据会使第二次及之后的运行比真实场景更快
- 对策
  - 测量前先预热若干次
  - 每次测量前用一次大块写入清空 L2
  - 最后取多次结果的中位数以排除偶发的异常值

`torch.compile(mode="max-autotune")` 的行为与适用场景：
- 会对 GEMM 和卷积在 cuBLAS、Triton 模板、CUTLASS 模板之间做 autotune
- 编译时间显著增加
- 适合在编译时间不敏感、运行时间敏感的场景中使用
  - 例如长时间训练和离线构建的推理服务

### 6.5 自动并行的现状与边界

#### 6.5.1 问题定义

自动并行（auto-parallelism）的目标：
- 用户只编写在单块 GPU 上运行的模型代码
- 由系统自动决定以下内容
  - 数据并行（DP）、张量并行（TP）、流水线并行和序列并行的组合方式
  - 每个张量在各块 GPU 之间的切分方式
- 概念
  - 流水线并行（PP）：把模型的不同层分配给不同 GPU
  - 序列并行（SP）：沿序列维切分激活

这个问题的难度在于决策之间的耦合：
- 某一层的权重采用哪种切分，决定了该层输出的分布方式
  - 进而决定下一层的输入是否需要先做通信转换
- 流水线划分又决定了每个阶段有多少显存可用于激活
  - 从而影响能否同时开启某种张量切分
- 系统需要在通信量、计算负载均衡和显存峰值之间同时求解

#### 6.5.2 两类技术路线

| 路线 | 做法 | 代表 |
|---|---|---|
| 标注 + 传播 | 用户标注少量张量的分片，编译器传播到全图并插入通信 | GSPMD（XLA）、PyTorch DTensor、JAX `shard_map` / `jit` with shardings |
| 搜索 | 枚举并行策略，用代价模型或实测选最优 | Alpa（ILP + 动态规划）、FlexFlow、nnScaler |

标注加传播的路线把决策权留给用户，系统只负责推导：
- 用户通常只需标注每个 Transformer block 中的几个权重矩阵
- 编译器根据算子的数学性质把分片信息传播到其余张量
  - 例如矩阵乘中若权重按列切分，输出也按列切分
  - 在分片方式不一致的位置插入 all-gather、reduce-scatter 或 all-reduce
- 这种方式的结果可预测，便于调试

搜索路线由系统自动做决策：
- Alpa 把问题分为两层
  - 层间用动态规划决定流水线阶段的划分
  - 层内把每个算子的切分方式建模为整数线性规划（ILP），以通信代价最小为目标求解
- FlexFlow 在并行策略空间中做随机搜索
  - 用模拟器估计每个候选的执行时间
- 这类方法在结构不规则的模型上可能找到人工难以想到的方案
  - 但求解时间和代价模型的准确度是主要限制（6.5.4）

#### 6.5.3 DTensor 示例

DTensor 相关概念：
- DTensor（distributed tensor）：PyTorch 中带有分布信息的张量类型
  - 它在逻辑上表示一个完整的张量，并记录该张量在一组设备上的实际存放方式
- 设备网格（device mesh）：把参与计算的 GPU 排列成的多维逻辑阵列
  - 每一维对应一种并行方式
- placement：描述 DTensor 在设备网格某一维上的分布方式，主要有三种
  - Shard(d)：沿张量的第 d 维切分，每块 GPU 持有一段
  - Replicate：每块 GPU 持有完整副本
  - Partial：每块 GPU 持有部分和，需要归约后才是完整的值

下例在 8 块 GPU 上构造 DP = 2、TP = 4 的二维网格，并对每个 block 应用张量并行：

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))   # 8 块 GPU：DP=2 × TP=4
tp_mesh = mesh["tp"]
for block in model.layers:
    parallelize_module(block, tp_mesh, {
        "attn.wq": ColwiseParallel(), "attn.wk": ColwiseParallel(), "attn.wv": ColwiseParallel(),
        "attn.wo": RowwiseParallel(),
        "mlp.w1": ColwiseParallel(), "mlp.w3": ColwiseParallel(), "mlp.w2": RowwiseParallel(),
    })
# 再在 dp 维度上 fully_shard(model, mesh=mesh["dp"]) 即得 FSDP + TP
```

这段代码中：
- ColwiseParallel 把线性层的权重按输出维切分
  - 输出的 placement 为 Shard(-1)
- RowwiseParallel 按输入维切分
  - 输出为 Partial，随后归约
- 注意力中 wq、wk、wv 按列切分，使每块 GPU 负责一部分注意力头
  - wo 按行切分，各 GPU 的部分结果经一次 all-reduce 合并
- MLP 中 w1、w3 按列切分，w2 按行切分
  - 同样只在 block 末尾需要一次 all-reduce
- 这是 Megatron-LM 风格的张量并行
  - 每个 block 的前向只需两次 all-reduce

这种方式属于半自动并行：
- 每一层采用何种切分由人决定
- 通信操作由框架根据 placement 自动插入

#### 6.5.4 全自动并行尚未成为主流的原因

- 第一，搜索空间巨大，代价模型不准确
  - 搜索空间是每层每个张量的切分方式、流水线的阶段划分和微批调度方式三者的乘积
  - 一个几十层的模型即可达到天文数字的规模，只能依赖代价模型剪枝
  - 而代价模型对通信与计算重叠、显存碎片等效应的估计误差较大
- 第二，最优解与硬件拓扑、通信库版本和 kernel 性能高度相关
  - 同样的切分在 NVLink 全互连的节点内和跨节点的 InfiniBand 网络上代价完全不同
  - 通信库或 kernel 的版本更新也会改变最优解
  - 模拟器难以准确跟踪这些变化
- 第三，大模型的结构高度同质
  - 主流 LLM 由几十层完全相同的 Transformer block 堆叠而成
  - 专家根据经验公式很快就能得到接近最优的配置
    - 例如 TP 不跨节点、PP 用于跨节点扩展、DP 填满剩余的 GPU
  - 自动搜索能带来的额外收益有限
- 第四，工程上要求方案可解释、可调试
  - 训练任务出现性能下降或数值异常时，工程师需要知道每个张量如何切分、通信在何处发生
  - 自动生成的方案出问题时难以定位

因此实际做法是：
- 结构化的部分（DP、TP、PP 的三维并行组合）用经验公式加小规模实测确定
- 局部的张量切分用 DTensor 或 GSPMD 标注

### 6.6 CUDA Graph 捕获与重放

#### 6.6.1 工作原理

CUDA Graph 是 CUDA 提供的一种执行机制：
- 把一串 kernel launch 连同它们的参数和依赖关系录制为一张图
- 之后调用一次 `graph.replay()` 即可重放整串 kernel，CPU 侧只需提交一次

普通执行方式下的开销：
- 每个 kernel 都要由 CPU 单独准备参数、经过驱动提交到 GPU 的命令队列
  - 单次 launch 的 CPU 开销在数微秒量级
- 如果 GPU 执行一个 kernel 的时间短于 CPU 提交下一个 kernel 的时间
  - GPU 就会在两个 kernel 之间空闲等待
  - 此时整体速度由 CPU 的提交速度决定
  - 这种状态称为 CPU 受限（CPU-bound）或 launch 受限

CUDA Graph 的做法：
- 在录制阶段把整串 kernel 的参数和依赖一次性交给驱动
  - 驱动预先完成校验和准备工作
- 重放时只需一次提交，GPU 可以连续执行所有 kernel

#### 6.6.2 性能收益

launch 开销的占比因模型而异：
- decode 一步可能包含几百个 kernel，每个 kernel 的执行时间只有几微秒
- 70B 模型在 TP=8 下 decode 一步约 8 ms
  - 单个 kernel 的计算量较大，launch 开销占比不明显
- 8B 模型在单块 GPU 上 decode 一步只有约 10 ms，且 kernel 很小
  - launch 开销可占 20–40%
- 以 300 个 kernel、每次 launch 约 5 μs 计，仅 CPU 提交就需要约 1.5 ms
  - 再加上 Python 和框架的调度开销，占比相当可观

CUDA Graph 的收益范围：
- 对小模型、小 batch decode 的提升最大
- prefill 阶段每个 kernel 本身耗时较长，GPU 始终处于忙碌状态，收益较小

#### 6.6.3 使用约束

静态内存地址（static memory address）：
- 录制时每个 kernel 读写的显存地址被固定写入图中、重放时不可更改
- 这是 CUDA Graph 大部分约束的来源：

| 限制 | 后果 | 对策 |
|---|---|---|
| 形状和指针固定 | 每种 batch 大小要录一张图 | 推理引擎按 batch 档位（1, 2, 4, 8, 16, ..., 512）预录多张图，实际 batch pad 到最近的档 |
| 内存地址固定 | 输入输出必须用固定的静态缓冲区 | 每次 replay 前把数据拷进静态缓冲 |
| 不能有 CPU 同步 | 图里不能有 `.item()`、CPU 判断 | 把控制流移到图外 |
| 显存 | 每张图有自己的内存池 | 共享内存池 |

形状和指针固定意味着一张图只对应一种 batch 大小：
- 推理引擎因此按档位预先录制多张图
  - 实际 batch 为 13 时填充到 16，选用 16 档的图执行
- 档位划分与 6.1.4 中的形状分档是同一个取舍
  - 档位越密，填充浪费越少
  - 但需要录制和保存的图越多

内存地址固定意味着不能在每次调用时传入新分配的张量：
- 录制时使用一组预先分配的静态缓冲区作为输入和输出
- 每次重放前把新数据拷贝进输入缓冲区，重放后从输出缓冲区读取结果
- 这次拷贝本身是一个很小的 kernel
  - 开销远小于节省的 launch 时间

不能有 CPU 同步，是因为图中只记录 GPU 上的操作：
- `.item()` 需要把数据从 GPU 拷回 CPU 并等待
  - 基于该值的 Python 分支在录制时只会走其中一条，重放时不会重新判断
- 凡是依赖运行时数值的逻辑，都必须放在图外，在两次重放之间由 CPU 执行
  - 例如判断生成是否结束
  - 例如根据采样结果选择下一步操作

显存方面的约束：
- 每张图在录制时从一个专属的内存池分配中间张量，并在图的生命周期内一直占用这些显存
  - 几十张图各自占用一份中间缓冲区，总量会相当可观
- PyTorch 允许多张图共享同一个内存池（录制时传入同一个 `pool` 参数）
  - 前提是这些图不会同时执行

#### 6.6.4 代码示例

下面的代码依次完成预热、录制和重放三个步骤：
- 预热在一个独立的 stream 上运行几次模型
  - 使显存分配器和 cuBLAS 完成各自的初始化
  - 初始化过程中的内存分配和工作空间申请不能出现在录制中

```python
import torch

static_x = torch.zeros(BATCH, HIDDEN, device="cuda", dtype=torch.bfloat16)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):                     # 预热，让分配器和 cuBLAS 完成初始化
    for _ in range(3): static_y = model(static_x)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):                      # 录制
    static_y = model(static_x)

def run(x):
    static_x.copy_(x)                          # 写入静态缓冲
    g.replay()                                 # 一次提交重放全部 kernel
    return static_y.clone()
```

`run` 返回 `static_y.clone()` 而不是 `static_y` 本身：
- 下一次重放会覆盖 `static_y` 的内容
- 调用方若持有引用，读到的值会被改写

框架中的使用：
- `torch.compile(mode="reduce-overhead")` 会自动使用 CUDA Graph，并负责静态缓冲区的管理
- vLLM、SGLang 在 decode 阶段默认使用 CUDA Graph
- prefill 阶段通常不使用
  - 序列长度变化范围大，按档位录制的图数量会过多
  - prefill 的 kernel 耗时较长、launch 开销占比低

### 6.7 torch.compile 的完整编译流程：从 Python 到 kernel

#### 6.7.1 示例函数

以下函数包含一个 MLP 形式的计算和一个残差连接后的 RMSNorm：
- 其中既有 GEMM，也有逐元素和归约算子
- 适合用来观察编译的各个阶段

```python
import torch

def block(x, w1, w2, g):
    h = x @ w1
    h = torch.nn.functional.silu(h)
    h = h @ w2
    y = x + h
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6) * g

cblock = torch.compile(block)
```

#### 6.7.2 各阶段的产物

| 阶段 | 产物 |
|---|---|
| Dynamo | FX 图：`matmul → silu → matmul → add → pow → mean → add → rsqrt → mul → mul`，加上 guard（dtype、形状、设备） |
| AOTAutograd | 前向 + 反向联合图；`silu`、`rsqrt` 等分解成基础算子；决定反向需要保存哪些张量 |
| Inductor 降低 | 两个 matmul 留给 cuBLAS（或 Triton 模板）；其余逐元素和归约降低成循环 IR |
| 调度融合 | `silu` 融进第一个 matmul 的 epilogue（模板 GEMM 时）或成为单独 kernel；`add → pow → mean → rsqrt → mul → mul` 融成一个 Triton kernel（一次读 x、h，写 y） |
| 代码生成 | 生成 Triton 源码，编译成 PTX → cubin；调用顺序写成一个 Python wrapper |
| 运行 | 每次调用检查 guard，命中就直接执行 wrapper |

从 HBM 流量看这次编译的效果：
- eager 模式下，matmul 之后的 8 个算子各自是一个 kernel
  - 每个都要从 HBM 读入上一步的结果并写出自己的结果
  - `y.pow(2)` 这样的中间张量与 x 大小相同
- 编译后，残差加法、平方、求均值、rsqrt 和两次乘法合并为一个 Triton kernel
  - 每个线程块负责若干行，读入 x 和 h 的对应行
  - 在片上算出 y 和该行的均方根，完成缩放后只写出最终结果
  - 中间张量全部不出现在 HBM 中
- kernel 数量从 10 个左右减少到 3 至 4 个
  - 逐元素和归约部分的 HBM 流量降为原来的一小部分

两个 matmul 的处理：
- 两个 matmul 不由 Inductor 生成循环代码
  - cuBLAS 和经过调优的 GEMM 模板在这类计算上的性能难以超越
- `silu` 能否融入第一个 GEMM，取决于 GEMM 采用哪种实现
  - 调用 cuBLAS 时，GEMM 是一个不可修改的库函数，`silu` 只能单独成为一个 kernel
  - 采用 Triton 模板时（例如 `max-autotune` 模式下模板胜出），`silu` 可以作为 epilogue 插入模板代码中

#### 6.7.3 查看生成代码

```bash
TORCH_LOGS="output_code" python my_script.py     # 打印 Inductor 生成的 Triton / wrapper 代码
TORCH_COMPILE_DEBUG=1 python my_script.py        # 把每个阶段的 IR 写进 torch_compile_debug/
```

阅读生成的 Triton 代码是理解融合最直接的方式：
- 可以看到哪些算子被合并进同一个 kernel，以及每个 kernel 读写了哪些张量
- Inductor 生成的 kernel 名称通常带有被融合算子的名字
  - 例如包含 `add`、`pow`、`mean`、`rsqrt`、`mul` 字样
- wrapper 代码则按顺序列出每个 kernel 的调用和缓冲区的分配与复用

#### 6.7.4 使用建议

- 先在 eager 模式下确认模型的正确性，再启用 `torch.compile`
  - 编译引入的问题（graph break、重编译、数值差异）与模型本身的错误混在一起时很难区分
  - 先有一个正确的 eager 基线，后续的每一处差异才有明确的比较对象
- 用 `torch._dynamo.explain` 找出并消除 graph break（6.1.3）
  - graph break 数量为零是理想状态
  - 少量 graph break 位于计算量很小的位置时，对性能的影响可以忽略
- 训练时对每个 Transformer block 分别编译，而不是编译整个模型
  - 由于所有 block 结构相同，编译一次的结果可以被所有层复用，总编译时间与层数基本无关
  - 整模型编译则需要处理一张包含全部层的巨大图，编译时间长
  - 并且模型中任意一处的改动都会导致整图重编译
- 动态形状较多的场景，预先用 `dynamic=True` 或 `mark_dynamic` 声明动态维度，或在数据层面做形状分档（6.1.4）
  - 避免运行过程中反复重编译
- 精度对比时，编译前后的 loss 应在 BF16 的误差范围内一致，但不会逐位相同
  - 原因是融合改变了浮点运算的执行顺序，例如归约的累加顺序不同
  - 浮点加法不满足结合律，结果在最低几位上会有差异
  - 若差异明显超出这一范围（例如训练 loss 曲线出现可见的偏离），应逐个子图对比定位问题

### 6.8 小结

- 编译器解决两个问题：通过算子融合减少 HBM 访问字节数，以及消除 Python 解释和 kernel launch 的开销。
- `torch.compile` 由三部分组成：Dynamo 负责图捕获，AOTAutograd 生成前向与反向联合图，Inductor 生成 Triton 代码。
- graph break 和形状变化引起的重编译是编译后性能下降的主要原因
  - 静态调度与动态形状之间的取舍贯穿整个编译栈
  - 形状分档和 CUDA Graph 都是在动态负载下恢复静态优势的手段
- 编译器的核心任务包括计算图划分、算子融合、内存分配、tile 大小选择、数据布局和通信调度
  - 面对异构硬件，软件抽象需要在可移植性与设备专用优化之间取得平衡
- autotuning 通过实测选择 kernel 配置，结果需要缓存；需要确定性时应固定配置。
- CUDA Graph 把一串 kernel launch 合并为一次提交，在小 batch decode 场景中收益最大。
- 全自动并行尚未取代人工配置，目前的主流是标注加传播的半自动方式。

<br>

## 7. 集群与硬件：加速器、互连与调度

前几章讨论的是单台机器内部的计算与访存，本章转向集群层面：

- 依次讨论加速器的代际演进与架构选型
- 先进封装
- 机器之间的互连网络
- 作业调度
- 存储系统
- 供电、散热和能耗等物理约束

### 7.1 NVIDIA 数据中心 GPU 的代际演进

#### 7.1.1 主要型号规格

| 代 | 型号 | 年份 | BF16 稠密 | FP8 稠密 | 显存 | 显存带宽 | NVLink 单向 / GPU |
|---|---|---|---|---|---|---|---|
| Volta | V100 | 2017 | 125（FP16） | — | 32 GB HBM2 | 0.9 TB/s | 150 GB/s |
| Ampere | A100 | 2020 | 312 | — | 80 GB HBM2e | 2.0 TB/s | 300 GB/s |
| Hopper | H100 SXM | 2022 | 989 | 1979 | 80 GB HBM3 | 3.35 TB/s | 450 GB/s |
| Hopper | H200 | 2024 | 989 | 1979 | 141 GB HBM3e | 4.8 TB/s | 450 GB/s |
| Blackwell | B200 | 2025 | ~2250 | ~4500 | 180–192 GB HBM3e | 8 TB/s | 900 GB/s |
| Blackwell Ultra | B300 / GB300 | 2025 | ~2250 | ~4500（FP4 更高） | 288 GB HBM3e | 8 TB/s | 900 GB/s |
| Rubin | R200（Vera Rubin） | 2026 | — | — | 288 GB HBM4 | ~20 TB/s 级 | NVLink 6，约为上一代 2 倍 |

- 单位：TFLOPS
- 「稠密」指不利用结构化稀疏时的峰值算力
  - NVIDIA 规格表中常见的「稀疏」数字是在 2:4 结构化稀疏（每 4 个元素中至少 2 个为零）条件下的理论值，约为稠密值的 2 倍
  - 实际训练很少能用上，因此本章一律采用稠密值

- HBM（High Bandwidth Memory，高带宽显存）：将多层 DRAM 裸片垂直堆叠、并与 GPU 封装在同一基板上的显存，其结构见 7.2.9.3
- NVLink：NVIDIA 的 GPU 间高速点对点互连，表格最后一列为每块 GPU 所有 NVLink 链路的单向总带宽

- 从表中可以看到，从 V100 到 B200 的八年间：
  - BF16 算力增长约 18 倍
  - 显存带宽增长约 9 倍
  - 显存容量增长约 6 倍
  - 算力的增长快于带宽和容量，这一趋势决定了 7.1.2 中各项比值的变化方向

- H800 / H20 是面向特定市场的版本，算力或互连带宽有所削减
  - H800 的 NVLink 带宽降至 400 GB/s（双向），多卡张量并行的通信更慢，但单卡算力与 H100 相同
  - H20 的算力大幅降低，但保留了与 H100 同级别的 HBM 带宽和更大的显存容量
    - 由于 decode 阶段主要受显存带宽限制（7.1.3），H20 在 decode 上的单卡吞吐与 H100 的差距远小于算力差距

#### 7.1.2 代际比较的三个指标

- 比较不同代 GPU 时，比绝对算力更有参考价值的是以下三个比值
- 算术强度（arithmetic intensity）：每字节访存对应的浮点运算次数
- 脊点（ridge point）：定义为峰值算力与显存带宽之比，即 kernel 从带宽受限转为算力受限时的算术强度（1.3.3）

| 比值 | 意义 | 趋势 |
|---|---|---|
| 算力 / 显存带宽（脊点） | 脊点越高，kernel 越容易受带宽限制 | 持续升高，算子融合与量化的重要性随之上升 |
| 显存容量 / 算力 | 单位算力可配置的模型状态容量 | B300、Rubin 提高了容量，有利于长上下文与 MoE |
| NVLink 带宽 / 算力 | GPU 间通信的相对成本 | NVL72 超节点使 TP / EP 可扩展到 72 卡 |

- 第一项比值决定了 kernel 需要多高的算术强度才能用满算力
  - H100 的脊点约为 $989/3.35\approx295$ FLOP/B，意味着一个 kernel 每从 HBM 读取 1 字节，必须完成约 295 次浮点运算，才不会受带宽限制
  - 逐元素运算（如加法、激活函数）的算术强度通常低于 1，远低于脊点，因此只能以带宽速度运行
  - 脊点逐代升高，意味着越来越多的算子落入带宽受限区，必须通过以下方式提高有效算术强度：
    - 算子融合：将多个算子合并为一个 kernel，中间结果留在片上
    - 量化：减少每个元素的字节数

- 第二项比值决定了单位算力能配多少模型状态
  - 模型权重、优化器状态、激活和 KV cache 都需要放在显存中
  - 容量不足时只能把模型切分到更多 GPU 上，从而引入额外通信
  - 长上下文推理的 KV cache 随序列长度线性增长
  - MoE 模型的总参数量远大于激活参数量
  - 二者都对容量提出了比算力更高的要求

- 第三项比值决定了 GPU 之间通信的相对代价
  - TP（张量并行）：将一层内的矩阵切分到多块 GPU 上
  - EP（专家并行）：将 MoE 的不同专家放在不同 GPU 上
  - TP 与 EP 在每一层都要通信，数据量大且频繁，因此其规模受 NVLink 域大小的约束
  - NVLink 域：指通过 NVLink 与 NVSwitch 实现全互连的一组 GPU（7.3.1）
  - 超节点：指由高速互连连接成单一 NVLink 域的多机系统
    - NVL72 指 GB200 / GB300 NVL72 机柜，其中 72 块 GPU 构成一个 NVLink 域
    - 在 H100 时代，NVLink 域通常仅限于一台 8 卡服务器

#### 7.1.3 按负载选型的依据

- 不同负载对硬件的需求差别很大，选型时应先确定主要负载，再确定优先考察的规格

- 训练负载主要考察 BF16/FP8 算力、NVLink 域的规模、节点间网络，以及软件栈的成熟度
  - 训练中的矩阵乘规模大、batch 大，算术强度高，主要受算力限制
  - 但大模型训练必然跨多块 GPU 乃至多个节点，梯度同步和张量并行的通信会占据可观的时间，因此网络带宽同样关键
  - 软件栈的成熟度决定了理论算力中有多少能转化为实测 MFU（7.2.8.2）

- 推理 decode（逐 token 生成阶段）主要考察显存带宽和容量，算力是次要因素
  - decode 每生成一个 token，都要从 HBM 读取一遍全部激活权重和该请求的 KV cache
  - 而计算量只有每个参数约 2 次浮点运算，算术强度很低（1.3.5）
  - 因此 decode 的速度近似为「显存带宽 ÷ 每步需要读取的字节数」
  - H20、H200 一类带宽高而算力相对较低的型号在 decode 上性价比较高

- 推理 prefill（并行处理输入提示的阶段）主要考察算力
  - prefill 一次处理输入提示中的所有 token，相当于一次 batch 很大的前向计算，算术强度高，行为接近训练中的前向传播

- 软件生态方面，采用 CUDA 以外的平台时，需要评估迁移与调优所需的人力成本（7.2.8）
  - 这部分成本不体现在硬件价格中，但往往决定了一个平台在实际项目中能否达到其标称性能

### 7.2 加速器架构比较：GPU、TPU 与 NPU

#### 7.2.1 术语与分类

| 术语 | 含义 | 例子 |
|---|---|---|
| CPU | 少量高性能核心，配有大容量缓存、分支预测和乱序执行，适合控制流密集的任务 | Xeon、EPYC、Grace |
| GPU / GPGPU | 由成千上万个简单核心组成，以 SIMT（Single Instruction, Multiple Threads，单指令多线程）方式执行；最初用于图形渲染，后扩展到通用计算（GPGPU） | H100、B200、MI300X |
| ASIC（Application-Specific Integrated Circuit，专用集成电路） | 为特定应用定制的芯片，不以通用性为目标 | 所有 TPU、NPU 均属于此类 |
| DSA（Domain-Specific Architecture，领域专用架构） | 面向某一类工作负载（此处为深度学习）优化的体系结构，保留一定的可编程性 | TPU、昇腾、Trainium |
| TPU（Tensor Processing Unit） | Google 自研 DSA 的产品名，不是通用术语 | TPU v4、v5p、v6e、v7 |
| NPU（Neural Processing Unit） | 神经网络加速器的统称，没有统一架构。在国内数据中心语境中多指昇腾，在端侧多指手机 SoC 中的 AI 单元 | 昇腾 910、Apple Neural Engine、高通 Hexagon |
| LPU / 晶圆级芯片 | 更激进的设计路线：片上存储全部采用 SRAM 并静态调度，或以整片晶圆制成一颗芯片 | Groq LPU、Cerebras WSE |

- ASIC 与 DSA 描述的是不同层面的属性
  - ASIC：一种实现方式，指芯片为特定用途定制流片
  - DSA：一种体系结构设计方法，指针对一个工作负载族（例如所有以矩阵乘为主的神经网络）选择计算单元、存储层次和互连
  - DSA 可以用 ASIC 实现，也可以采用可编程加速器或二者混合的形式
  - DSA 并非只能运行一种固定算法：TPU 和昇腾都提供指令集、编译器和 kernel 编程接口，可以运行各种模型结构，只是对矩阵乘以外的运算效率较低
  - 以固定功能电路实现单一算法的芯片（例如比特币矿机）才属于完全不可编程的 ASIC

- TPU 是品牌，NPU 是类别，GPU 是一种架构
  - 因此三者的比较可归结为两种设计思路的比较：GPU 采用「通用并行处理器 + 矩阵单元」的结构，DSA 则以矩阵单元为主体，并尽可能简化控制逻辑

#### 7.2.2 共性设计问题：维持乘加单元的利用率

##### 7.2.2.1 Transformer 的计算负载特征

- Transformer 的 FLOPs 有 95% 以上来自矩阵乘（1.3.4）
- 因此各类 AI 芯片都需要回答同一组设计问题：

1. 乘加单元（MAC，Multiply-Accumulate，完成一次 $a\times b+c$ 运算的单元）如何排布，单个周期能完成多大的矩阵块；
2. 数据如何从 HBM 送达 MAC，片上存储由硬件管理还是由软件管理；
3. 非矩阵运算（softmax、归一化、激活函数）由哪个单元执行；
4. 多颗芯片之间如何互连；
5. 向程序员和编译器提供怎样的编程模型。

- 第 1、2 个问题决定了矩阵单元能否持续满载：MAC 阵列每个周期都需要新的操作数，如果数据供给的速度跟不上，再多的 MAC 也只能空转
- 第 3 个问题决定了非矩阵运算是否成为瓶颈：这些运算的 FLOPs 占比虽小，但如果执行它们的单元吞吐太低，矩阵单元就要等待
- 第 4 个问题决定了多芯片扩展的效率
- 第 5 个问题决定了软件能否把硬件能力发挥出来

##### 7.2.2.2 芯片面积的分配

- 芯片面积是固定的，各类架构的差别在于面积如何分配
  - CPU 将大量面积用于缓存、分支预测和乱序执行，用于乘加运算的面积占比很小
  - GPU 将面积分配给大量 SM（Streaming Multiprocessor，流式多处理器，GPU 的基本计算单元）
    - 每个 SM 内仍包含寄存器堆、warp 调度器和 L1 缓存等控制与缓冲结构
  - DSA 进一步削减通用控制逻辑，将尽可能多的面积用于矩阵单元和片上 SRAM
    - 其代价是灵活性下降：控制逻辑越简单，越依赖编译器在编译期完成全部调度安排

- 面积分配直接影响能效
  - 乱序执行、分支预测和缓存标签比较等控制电路在每条指令执行时都会消耗能量，却不产生任何乘加结果
  - 在 CPU 上，一次浮点乘加本身的能耗只占执行一条指令总能耗的一小部分，其余消耗在取指、译码、寄存器读写和调度上
  - 矩阵单元一条指令完成成百上千次乘加，把这部分开销分摊到大量运算上
  - 因此 GPU 的 Tensor Core 和 DSA 的矩阵阵列在能效上比标量执行高出一到两个数量级

##### 7.2.2.3 片上数据传输的管理方式

数据在存储层级之间的传输由谁决定，是区分各类架构最有效的标准：

| 方式 | 数据传输的决定者 | 代表 |
|---|---|---|
| 硬件缓存 + 线程切换隐藏延迟 | 硬件（L1/L2 缓存）与程序员（shared memory） | GPU |
| 软件管理的片上存储（scratchpad） | 编译器或 kernel 作者显式发起 DMA | TPU、昇腾、Trainium |
| 全部数据驻留 SRAM，编译期静态排程 | 编译器，精确到每个时钟周期 | Groq |

- shared memory：GPU 每个 SM 内由程序员显式管理的片上存储
- scratchpad memory（暂存存储器）：指由软件显式分配和读写的片上 SRAM，硬件不自动决定其中存放哪些数据
- DMA（Direct Memory Access，直接存储器访问）：指由专用引擎在存储层级之间传输数据，无需计算单元逐条执行访存指令

- 三种方式在硬件复杂度与编程负担之间形成不同的取舍
  - 硬件缓存对程序员透明，代码写法简单，但需要标签存储、替换逻辑和一致性维护电路，且命中与否在运行时才能确定，性能难以精确预测
  - scratchpad 省去了这些电路，面积和能耗更低，数据何时到达也完全可知，但要求编译器或程序员提前规划每一块数据的搬入、使用和搬出时机
  - 全 SRAM 静态排程把这一思路推到极致，硬件完全不做运行时决策

- 近几代 GPU 也在向第二种方式靠拢：
  - Hopper 的 TMA（Tensor Memory Accelerator）是一个异步 DMA 引擎，kernel 只需描述要传输的张量块，由 TMA 在后台完成地址计算和传输
  - Blackwell 又为 Tensor Core 增设了专用的 Tensor Memory，用于存放矩阵乘的累加结果

##### 7.2.2.4 硬件的评价指标：有效吞吐、能耗与成本

- 宣传材料中最常见的指标是峰值 FLOPS，但它只描述理想条件下矩阵单元的最大吞吐
- 一颗芯片在实际部署中的价值取决于以下几项：

- 有效吞吐（useful work per second）：在真实负载下单位时间完成的有用工作
  - 对训练是每秒处理的 token 数
  - 对推理是满足延迟目标前提下每秒生成的 token 数
  - 它等于峰值算力乘以实际利用率，而利用率受带宽、通信、kernel 质量和调度共同影响
- 能耗：完成单位工作所消耗的电能，例如每 token 的焦耳数（7.7.2）
  - 在电力供应受限的数据中心，能效直接决定了同一电力预算下可部署的总算力
- 成本：包括芯片采购价格、配套的服务器与网络、机房与电力，以及软件适配的人力投入，通常以总拥有成本（TCO，Total Cost of Ownership）衡量
- 可用性：能否按需采购到足够数量，以及软件生态是否支持目标模型

- 峰值算力高的芯片在上述指标上不一定占优
  - 例如 decode 负载受带宽限制，一颗峰值算力翻倍但带宽不变的芯片，decode 吞吐几乎不会提高
  - LLM 推理服务同时包含长短不一的输入、长度不同的输出、不断变化的并发量和多种模型结构，没有任何一个单一算子能代表全部负载
  - 因此评价硬件时应以端到端的实测指标为准

##### 7.2.2.5 专用化程度与过时风险

- 一款加速器从立项定义到量产通常需要两到三年，部署后又要服役三到五年
- 设计者必须在立项时预测五年以后的模型形态，而在这段时间内，模型结构、上下文长度、MoE 的普及程度和低精度格式都可能发生显著变化
  - 例如，若一款芯片为 4K 上下文的稠密模型优化了片上存储容量
    - 在模型普遍转向 128K 上下文和 MoE 之后，其 KV cache 容量和专家并行所需的互连带宽都可能不足

- 专用化程度越高，对目标负载的效率越高，但工作负载改变后过时的风险也越大
  - GPU 保留了较高的通用性，单位面积效率较低，却能适应新出现的算子
  - DSA 效率更高，但如果新模型结构不能高效映射到其矩阵单元上，性能会显著下降
  - 实际设计中常用的折中：
    - 矩阵单元针对矩阵乘专门优化，同时保留足够灵活的向量单元和标量单元，以覆盖新的非矩阵运算
    - 数值格式支持多种精度
    - 编译器和 kernel 接口保持可编程，以便在硬件不变的前提下适配新模型

#### 7.2.3 GPU：SIMT 执行与 Tensor Core

##### 7.2.3.1 SM 结构回顾

- 1.1 节已介绍 H100 的结构：
  - 全片有 132 个 SM
  - 每个 SM 分为 4 个子分区，每个子分区包含一个 warp 调度器、一组 CUDA core、一个 Tensor Core 和一块寄存器堆
  - 每个 SM 内的 L1 缓存与 shared memory 共用 256 KB
  - 全片共享 50 MB L2 缓存
- CUDA core：执行标量浮点与整数运算的单元
- Tensor Core：执行小矩阵块乘加的专用单元

##### 7.2.3.2 SIMT 执行与延迟隐藏

- 程序员编写单个线程的代码，硬件将 32 个线程编为一个 warp（线程束），同一 warp 内的线程同时执行同一条指令，各自处理不同的数据
  - 这种执行方式称为 SIMT
- 与 CPU 的 SIMD 向量指令相比，SIMT 允许每个线程有独立的寄存器和程序计数器
  - 线程之间出现分支时，硬件会分别执行两条分支路径并屏蔽不参与的线程，编程模型更接近普通的标量代码

- 一个 SM 上同时驻留数十个 warp
  - 当某个 warp 等待访存时，调度器在下一个周期即可切换到另一个就绪的 warp 发射指令
    - 切换本身没有额外开销
    - 因为每个 warp 的寄存器始终保存在寄存器堆中
  - HBM 访问的延迟约为数百个周期，只要 SM 上有足够多的 warp 轮流发射指令，访存等待的时间就会被其他 warp 的计算填满
  - GPU 依靠始终存在可调度的 warp 来隐藏访存延迟，而不依赖缓存命中
  - 这也是 GPU 需要容量很大的寄存器堆（H100 每个 SM 有 256 KB）的原因

- Tensor Core 以单条指令完成一个小矩阵块的乘加
  - Hopper 的 wgmma 指令由 4 个 warp（称为一个 warpgroup）协同发射
  - 块大小可达 64×N×16
  - 操作数可以直接从 shared memory 读取，不必先装入寄存器

##### 7.2.3.3 GPU 在大模型训练中占主导地位的原因

- GPU 的第一项优势是通用性
  - 同一块 GPU 可以执行矩阵乘、注意力、MoE 路由、采样和各类自定义 kernel，新算子提出后通常数周内即有 CUDA 实现
  - 研究人员提出一种新的注意力变体时，可以先用 PyTorch 写出参考实现，再用 Triton 或 CUDA 逐步优化，整个过程不需要等待硬件厂商的支持

- 第二项优势是对动态形状友好
  - SIMT 线程模型以单个线程为编程单位，变长序列只需让多余的线程提前退出，分支和稀疏计算也能直接表达
  - 相比之下，依赖编译期静态排程的架构在形状变化时往往需要重新编译（7.2.4.9）

- 第三项优势是软件生态
  - CUDA、cuBLAS、cuDNN、NCCL、Triton、FlashAttention、vLLM 等构成完整的工具链，多数新工作首先在 GPU 上发布
  - 这种生态优势会自我强化：新工作先在 GPU 上实现，使 GPU 上的软件更完善，又吸引更多新工作

- 第四项优势是互连
  - NVLink 与 NVSwitch 将 scale-up 域扩展到 72 卡（7.3、8.2）
  - NVSwitch：连接多块 GPU 的 NVLink 端口的交换芯片，使域内任意两块 GPU 都能以全带宽通信
  - scale-up：指在单个高速互连域内增加加速器数量；与之相对，scale-out 指通过节点间网络连接更多节点（7.3.1）

##### 7.2.3.4 GPU 架构的局限

- GPU 的通用性需要付出面积和能耗代价
  - warp 调度器、大容量寄存器堆、缓存标签和一致性逻辑都占用芯片面积，并在每个周期消耗功耗，而它们本身不完成矩阵乘
  - 在矩阵乘占绝对主导的负载上，这部分开销使 GPU 的单位面积和单位功耗算力低于同工艺的 DSA

- 使 Tensor Core 达到满载需要精心编写 kernel
  - kernel 必须让 TMA 传输、Tensor Core 计算和 softmax 等向量运算在多个 warpgroup 之间流水重叠，才能接近峰值
  - 例如 FlashAttention-2 在 H100 上只能达到约 35% 的峰值利用率
  - FlashAttention-3 专门针对 Hopper 的 wgmma 和 TMA 重新实现后，BF16 利用率提高到约 75%

- 此外，高端 GPU 的采购价格高，且长期供不应求，大规模集群的建设周期常受交货时间约束

##### 7.2.3.5 AMD Instinct 系列

AMD Instinct 系列（MI300X / MI325X / MI355X）与 NVIDIA GPU 采用相同的架构思路：

| | NVIDIA | AMD |
|---|---|---|
| 计算单元 | SM | CU（Compute Unit） |
| 线程束 | warp，32 线程 | wavefront，64 线程（CDNA） |
| 矩阵单元 | Tensor Core | Matrix Core |
| 编程 | CUDA | HIP（语法与 CUDA 几乎相同，可用 hipify 自动转换） |
| 库 | cuBLAS / NCCL | rocBLAS、hipBLASLt / RCCL |
| 封装 | 单 die 或双 die | chiplet：MI300X 由 8 个计算 die 和 4 个 IO die 组成 |

- wavefront：AMD GPU 上以锁步方式执行同一条指令的线程组，对应 NVIDIA 的 warp
- CDNA：AMD 数据中心 GPU 的架构名称
- die：指从晶圆上切割下来的单个裸片
- chiplet（芯粒）：指将一颗芯片的功能拆分到多个较小的 die 上，再通过先进封装互连为一个整体（7.2.9）
  - MI300X 的 8 个计算 die 以 3D 方式堆叠在 4 个 IO die 之上，IO die 再与 8 个 HBM 堆栈一起放置在硅中介层上，是 2.5D 与 3D 封装同时使用的例子

- MI300X 配备 192 GB HBM3，带宽为 5.3 TB/s，容量是 H100 的两倍以上
  - 单卡可以容纳更大的模型和 KV cache：例如 70B 模型的 BF16 权重约 140 GB，在 MI300X 上单卡即可放下，而在 H100 上至少需要两块卡做张量并行
  - 因此 MI300X 在推理场景中具有性价比优势

#### 7.2.4 TPU：基于脉动阵列的 DSA

##### 7.2.4.1 TPU 芯片组成

- 一颗 TPU 芯片包含 1–2 个 TensorCore
  - 此处的 TensorCore 指一个完整的计算核心，与 NVIDIA 的 Tensor Core 含义不同
- 每个 TensorCore 包含以下部件：

- MXU（Matrix Multiply Unit，矩阵乘单元）：一个脉动阵列，v2–v5 为 128×128，自 v6e（Trillium）起为 256×256
  - 每个 TensorCore 通常有多个 MXU，承担全部矩阵乘运算，是芯片算力的主要来源
- VPU（Vector Processing Unit，向量处理单元）：执行逐元素运算、归约、softmax 等非矩阵运算
  - VPU 以二维向量寄存器为操作对象，一条指令同时处理数千个元素，吞吐远低于 MXU，但足以覆盖 Transformer 中非矩阵运算的计算量
- 标量单元：负责取指、控制流和发起 DMA
  - 它向 MXU、VPU 和 DMA 引擎发射指令，功能相当于整个核心的控制器
- VMEM：容量为数十 MB 量级的片上 SRAM，由软件管理，不是缓存
  - MXU 和 VPU 的操作数都从 VMEM 读取，HBM 中的数据必须先由 DMA 搬入 VMEM 才能参与计算
- SparseCore：自 v4 起加入，专门处理推荐模型中的 embedding 查找
  - embedding 查找是对大表的随机稀疏访问，算术强度极低，不适合在 MXU 上执行

##### 7.2.4.2 脉动阵列的矩阵乘数据流

- 脉动阵列（systolic array）是由 $N\times N$ 个处理单元（PE，Processing Element）组成的二维网格，每个 PE 只与相邻的 PE 相连
  - 每个 PE 在每个时钟周期（下文称「拍」）执行相同的操作
    - 接收左侧和上方传入的操作数
    - 相乘并累加
    - 再将两个操作数分别传给右侧和下方的相邻 PE
  - 数据逐拍、有节奏地流过整个阵列
  - systolic 一词借自医学中心脏的收缩搏动，用以比喻这种节律

- 以输出驻留（output-stationary）方式为例：每个输出元素固定在一个 PE 中累加，输入数据流经阵列
- 计算 $C=AB$，其中 $A$ 为 $M\times K$，$B$ 为 $K\times N$，阵列大小为 $M\times N$：

- PE$(i,j)$ 负责 $C_{ij}$，部分和始终保留在该 PE 内累加；
- $A$ 的第 $i$ 行从左侧第 $i$ 行送入，每行比上一行延迟一拍；
- $B$ 的第 $j$ 列从上方第 $j$ 列送入，每列比左侧一列延迟一拍；
- 第 $t$ 拍时，PE$(i,j)$ 恰好收到 $A_{i,k}$ 和 $B_{k,j}$，其中 $k=t-i-j$。

```text
            B 列从上往下流（逐列错开一拍）
              b00   .     .
              b10  b01    .
              b20  b11   b02
               ↓    ↓     ↓
 a02 a01 a00 → PE00 → PE01 → PE02
     a12 a11 a10 → PE10 → PE11 → PE12
         a22 a21 a20 → PE20 → PE21 → PE22
 A 行从左往右流（逐行错开一拍）
```

- 行与列逐个错开一拍的目的，是让 $A_{i,k}$ 和 $B_{k,j}$ 这两个需要相乘的操作数恰好在同一拍到达 PE$(i,j)$
  - $A_{i,k}$ 从左边进入后，每拍向右移动一格，经过 $j$ 拍到达第 $j$ 列
  - $B_{k,j}$ 从上边进入后，每拍向下移动一格，经过 $i$ 拍到达第 $i$ 行
  - 两者送入时各自的延迟分别为 $i$ 拍和 $j$ 拍，再加上第 $k$ 个元素本身在序列中的位置 $k$，到达时刻都是 $t=i+j+k$

下面用一个 $2\times2$ 的例子逐拍验证。设

$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\qquad B=\begin{bmatrix}5&6\\7&8\end{bmatrix},\qquad AB=\begin{bmatrix}19&22\\43&50\end{bmatrix}$$

- 下标从 0 开始，$M=N=K=2$，共需 $T=K+M+N-2=4$ 拍
- 按 $k=t-i-j$ 逐拍列出每个 PE 的运算（「—」表示该拍空闲）：

| 拍 $t$ | PE(0,0) | PE(0,1) | PE(1,0) | PE(1,1) |
|---|---|---|---|---|
| 0 | $k=0$：$1\times5=5$ | — | — | — |
| 1 | $k=1$：$5+2\times7=19$ | $k=0$：$1\times6=6$ | $k=0$：$3\times5=15$ | — |
| 2 | — | $k=1$：$6+2\times8=22$ | $k=1$：$15+4\times7=43$ | $k=0$：$3\times6=18$ |
| 3 | — | — | — | $k=1$：$18+4\times8=50$ |

- 4 拍结束后，四个 PE 中分别保存 19、22、43、50，与 $AB$ 一致
- 以 PE(0,1) 为例
  - $A_{0,0}=1$ 在第 0 拍进入 PE(0,0)，第 1 拍向右传到 PE(0,1)
  - $B_{0,1}=6$ 因所在列比第 0 列延迟一拍，也在第 1 拍从上方进入 PE(0,1)
  - 二者在第 1 拍相遇
- 4 个 PE 在 4 拍中共有 16 个 PE·拍的计算能力，实际完成 8 次乘加，利用率为 50%，与 7.2.4.4 的公式 $K/(K+M+N-2)=2/4$ 一致

- TPU 实际采用权重驻留（weight-stationary）方式：
  - 先将一块 $128\times128$ 的权重载入阵列并固定在各 PE 中，再让激活逐行流过阵列，部分和沿列方向向下传递并累加，从阵列底部流出
  - 其原理与输出驻留相同，区别只在于固定在 PE 中的对象由输出换成了权重
  - 神经网络推理中同一块权重要与 batch 中所有样本的激活相乘，权重驻留使每块权重只需载入一次，就能被整个 batch 复用

##### 7.2.4.3 脉动阵列的数据复用与能耗

- 一个元素进入阵列后，沿所在的行或列被 $N$ 个 PE 依次复用，中间结果直接在相邻 PE 之间传递，不写回寄存器堆或 SRAM
- 对于 $N\times N$ 的阵列：

$$\text{每拍 MAC 数} = N^2,\qquad \text{每拍从外部读入的元素数} = 2N$$

$$\text{片上复用比} = \frac{N^2}{2N} = \frac{N}{2}$$

- $N=128$ 时，每从外部读入一个元素可完成 64 次乘加；$N=256$ 时为 128 次
- GPU 的 Tensor Core 同样存在数据复用，但数据需要在寄存器堆与 shared memory 之间多次传输，每一级存储都要消耗面积和功耗

- 局部通信节省能耗的原因可以从 CMOS 电路的动态功耗来理解
  - 数字电路中，一根导线每次从 0 翻转到 1 或从 1 翻转到 0，都要对导线及其所连接晶体管的等效电容充电或放电
  - 由此产生的动态功耗近似为

$$P_{\text{dynamic}}\approx\alpha\, C\, V^2 f$$

- 其中 $\alpha$ 是活动因子（每个周期发生翻转的概率），$C$ 是被驱动的等效电容，$V$ 是电源电压，$f$ 是时钟频率
- 导线的电容大致与长度成正比
  - 因此把一个数从片外 HBM 读入、穿过芯片上长距离的全局布线送到计算单元，需要驱动的电容远大于在两个相邻 PE 之间传递同一个数
- 脉动阵列中操作数每拍只移动到相邻 PE，连线极短，每次传递的能耗很低；同一个数被复用 $N$ 次，长距离读取的能耗也被摊到 $N$ 次乘加上
- 这一公式是简化模型，实际芯片还有漏电功耗、时钟树功耗和 I/O 功耗，但它足以说明数据移动距离对能耗的影响
- 7.7.3 将从存储层级的角度给出数据移动能耗的数量级

##### 7.2.4.4 流水线的填充与排空开销

- 数据从阵列的一角传播到对角需要时间
- 在输出驻留方式下，$M\times N$ 阵列计算长度为 $K$ 的点积，从第一拍到最后一个 PE 完成计算共需

$$T = K + M + N - 2 \text{ 拍}$$

- 其中 $K$ 拍用于让每个 PE 完成 $K$ 次乘加，另外的 $M+N-2$ 拍是数据从左上角 PE(0,0) 传播到右下角 PE$(M-1,N-1)$ 所需的时间
- 在开始阶段，只有靠近左上角的 PE 收到了数据，其余 PE 空闲，称为填充（fill）
- 在结束阶段，左上角的 PE 已经算完，只有右下角附近的 PE 还在工作，称为排空（drain）

- 有效乘加次数为 $MNK$，阵列在这段时间内的乘加能力为 $MN\cdot T$，利用率为

$$U = \frac{MNK}{MN(K+M+N-2)} = \frac{K}{K+M+N-2}$$

- $M=N=128$ 时，若要求利用率达到 90%，需满足 $K/(K+254)\ge0.9$，即 $K\ge 2286$
- 实际硬件会在前一块数据之后紧接着送入下一块，使流水线保持满载，因此连续的大矩阵乘可以接近满利用率
- 但当矩阵较小，或维度不是 128 的整数倍时，填充、排空和补零造成的浪费会很明显
  - 例如一个维度为 130 的矩阵需要补零到 256，浪费接近一半的计算
  - TPU 偏好大 batch、要求维度对齐到 128（自 v6e 起为 256），原因即在于此

##### 7.2.4.5 脉动阵列的逐拍模拟

下面的代码按上述数据流逐拍模拟输出驻留阵列，并统计 PE 的利用率：

```python
import numpy as np

def systolic_matmul(A, B):
    """输出驻留脉动阵列的逐拍模拟。PE(i, j) 在第 t 拍处理 k = t - i - j。"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N))
    T = K + M + N - 2
    busy = 0
    for t in range(T):
        for i in range(M):
            for j in range(N):
                k = t - i - j
                if 0 <= k < K:
                    C[i, j] += A[i, k] * B[k, j]
                    busy += 1
    util = busy / (M * N * T)
    return C, T, util

for K in [16, 128, 1024]:
    A, B = np.random.randn(8, K), np.random.randn(K, 8)
    C, T, util = systolic_matmul(A, B)
    assert np.allclose(C, A @ B)
    print(f"K={K:5d}  拍数={T:5d}  利用率={util:.1%}  公式={K/(K+14):.1%}")
```

- 代码中最内层的判断 `0 <= k < K` 对应 PE 是否在该拍收到了有效数据：$k<0$ 表示数据尚未到达（填充阶段），$k\ge K$ 表示数据已经流过（排空阶段）
- 阵列大小固定为 $8\times8$，因此公式中的 $M+N-2=14$
- 模拟得到的利用率与公式 $K/(K+M+N-2)$ 完全一致：$K=16$ 时仅为 53%，$K=1024$ 时达到 98.7%

##### 7.2.4.6 芯片间互连：ICI 与 OCS

- TPU 芯片之间通过 ICI（Inter-Chip Interconnect，芯片间互连）直接相连，不经过网卡和交换机
  - 每颗芯片集成了若干条高速串行链路，直接连到相邻芯片
  - v4 / v5p 组成 3D torus（三维环面）拓扑
    - 芯片排列成三维网格，每颗芯片在 X、Y、Z 三个维度上各与两颗相邻芯片相连，共 6 个邻居
    - 每个维度的首尾芯片也相连，形成环
  - 环形回绕使网格边缘的芯片与中心芯片具有相同的连接数，任意两颗芯片之间的最大跳数减半

- v4 引入了 OCS（Optical Circuit Switch，光路交换机）
  - OCS 通过 MEMS（微机电系统）微镜改变光束的反射方向，从而改变光纤之间的连接关系
  - 它只切换光路，不解析数据包，因此不增加转发延迟
  - 借助 OCS，以 64 颗芯片（$4\times4\times4$）为单位的立方体可以按需组合成不同形状的切片（slice），例如为一个作业分配 $8\times8\times16$ 的拓扑
  - 某个立方体出现故障时，也可以重新连接光路将其绕开，而不影响其余部分

- pod：指通过 ICI 互连的一套完整 TPU 系统，其规模为：v4 有 4096 颗芯片，v5p 有 8960 颗，v7（Ironwood）有 9216 颗
- GPU 的 NVLink 域目前最大为 72 卡（NVL72），更大范围的通信需要经过 IB / RoCE 网络（7.3）
- TPU 的 ICI 域则直接覆盖整个 pod，数千颗芯片之间的集合通信都不经过以太网或 InfiniBand

- 在 torus 上执行集合通信时，可沿三个维度分别执行 ring 算法
  - 例如对一个三维张量做 AllReduce，先沿 X 维的环归约，再沿 Y 维，最后沿 Z 维
    - 三个维度的链路同时工作，带宽相互叠加
    - 推导见 JAX Scaling Book 的 TPU 一章

##### 7.2.4.7 静态调度与动态调度

- 处理器的调度方式分为两类
- 静态调度（static scheduling）：指在编译期确定指令的执行顺序、各功能单元的分配和数据传输的时机，硬件按编译器给出的顺序执行，不做运行时决策
- 动态调度（dynamic scheduling）：指由硬件在运行时根据数据依赖和资源占用情况决定下一条执行的指令，CPU 的乱序执行和 GPU 的 warp 调度都属于动态调度

- TPU 的核心采用 VLIW（Very Long Instruction Word，超长指令字）指令集
  - 一条指令同时编码 MXU、VPU、标量单元和 DMA 引擎各自要执行的操作
  - 由编译器在编译期决定哪些操作并行执行
  - 这种方式省去了动态调度所需的依赖检查和发射逻辑，硬件更简单、能效更高，执行时间也可以精确预测，便于编译器把 DMA 传输与计算完全重叠

- 静态调度的局限在于，它要求编译期已知计算的形状和控制流
  - 输入形状改变、存在数据相关的分支或不规则的稀疏访存时，编译器无法提前排好最优顺序
    - 只能重新编译
    - 或按最坏情况保守安排
    - 或把数据补齐到固定形状
  - TPU 并非所有操作都完全静态：DMA 的完成需要同步等待，标量单元也支持循环和条件跳转
  - 但其主要性能来源是编译器对规则计算的静态安排

- 在静态调度的架构上，编译器承担了 GPU 上由硬件完成的大部分工作
  - 包括计算图分解、算子融合、片上存储分配、数据布局和通信调度
  - 还包括分块（tiling）大小选择
  - tiling 即把大矩阵切成可以放入片上存储的小块，以减少 HBM 读写
- 硬件的实际性能取决于硬件与编译器的协同设计

##### 7.2.4.8 TPU 编程模型：JAX、XLA 与 Pallas

- TPU 的主要编程方式是 JAX + XLA
  - 用户用 JAX 编写完整的计算图，JAX 将其转换为 XLA 的中间表示，由 XLA 编译器负责算子融合、分块、DMA 安排和 MXU 流水线（6.3.2）
  - 用户通常不需要关心数据如何在 HBM 与 VMEM 之间传输

- 分布式训练中，用户通过 `jax.sharding` 标注张量的切分方式（例如将权重矩阵的第二维切分到 8 颗芯片上）
  - 编译器（GSPMD / Shardy）根据标注推导出每个算子的切分方式，并自动插入所需的通信（6.5）

- 需要手写 kernel 时使用 Pallas
  - 用户声明每个块从 HBM 读取哪一部分到 VMEM，块内计算用 `jnp` 编写
  - 以下是 Pallas 官方教程中的分块矩阵乘：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def matmul_kernel(x_ref, y_ref, z_ref):
    # 网格第三维是 K 方向，第一块时清零，之后累加
    @pl.when(pl.program_id(2) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)
    z_ref[...] += x_ref[...] @ y_ref[...]

def matmul(x, y, bm=512, bk=512, bn=512):
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
                  pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
    )(x, y)
```

- 代码中 `grid` 定义了三维的块网格，每个网格点 `(i, j, kk)` 计算输出块 $(i,j)$ 在 $K$ 方向第 `kk` 段的贡献
- `BlockSpec` 中的 lambda 函数把网格坐标映射为输入块的位置，例如 `x` 的块取第 `i` 行段、第 `kk` 列段
- 编译器根据这些映射生成 DMA 指令，并自动在计算当前块时预取下一块

- 与 2.6 节的 Triton 相比：
  - Triton 描述的是「一个程序实例处理一个块」，块内线程的划分由编译器决定
  - Pallas 描述的是「每个块从 HBM 读取哪一部分」，数据传输由 `BlockSpec` 声明，流水线由编译器生成
  - 两者的编程抽象很接近，都由用户指定分块方式，其余细节由编译器完成

##### 7.2.4.9 TPU 的优势与限制

- TPU 的优势首先是大矩阵乘的效率
  - 脉动阵列的复用比高、控制逻辑少，相同功耗下可提供更多算力
- 其次是互连规模
  - ICI 使数千颗芯片组成的高带宽互连域成为可能，大规模张量并行和专家并行可以在整个 pod 内展开
  - 而在 GPU 集群上超出 NVLink 域的通信要走带宽低一个数量级的 scale-out 网络
- 此外，Google 的自研模型（Gemini）与硬件、编译器和软件栈协同设计，模型结构可以针对 TPU 的特点选择

- TPU 的限制主要来自静态调度
  - 它要求静态形状，形状改变即触发重新编译，一次编译可能耗时数十秒到数分钟
  - 动态控制流、稀疏计算和变长序列需要借助 padding 或 bucketing 处理，二者都会引入无效计算
    - padding：补齐到固定长度
    - bucketing：将长度相近的序列分入同一组，每组使用一种固定形状
  - 此外，TPU 仅能在 Google Cloud 上使用；PyTorch 需通过 PyTorch/XLA 接入，成熟度不及 JAX

#### 7.2.5 NPU：以华为昇腾为例

##### 7.2.5.1 达芬奇架构与 AI Core

昇腾 910 系列的计算核心称为 AI Core，采用达芬奇（Da Vinci）架构。每个 AI Core 包含三类计算单元：

| 单元 | 功能 | GPU 中的对应部件 |
|---|---|---|
| Cube | 矩阵乘，每拍完成一次 $16\times16$ 与 $16\times16$ 的 FP16 矩阵块乘法，即 4096 次乘加 | Tensor Core |
| Vector | 逐元素运算、归约、激活、类型转换 | CUDA core |
| Scalar | 控制流、地址计算、向其他单元发射指令 | warp 调度器和整数单元 |

- Cube 是三维立方结构的矩阵单元
  - 与 TPU 的二维脉动阵列不同，它在一拍内并行完成整个 $16^3$ 块的乘加，不存在脉动阵列中逐拍传播的填充与排空过程
  - 阵列规模小，矩阵维度只需对齐到 16，对小矩阵更友好
  - 代价是每次乘加的操作数需要通过更宽的数据通路同时送达，单位算力的布线开销高于脉动阵列

- 三类单元可以并行工作
  - Scalar 单元负责循环控制和地址计算，并把指令分别发往 Cube、Vector 和数据传输引擎的指令队列
  - 各队列独立执行
  - 单元之间通过同步指令协调依赖关系

##### 7.2.5.2 AI Core 的片上存储层次

AI Core 的片上存储全部由软件管理，各级 buffer 的连接关系如下：

```text
         HBM（Global Memory）
           │  MTE（Memory Transfer Engine，DMA）
           ▼
          L1 Buffer ───────────────┐
           │                      │
     ┌─────┴─────┐                │
     ▼           ▼                ▼
  L0A（左矩阵）  L0B（右矩阵）   Unified Buffer（UB）
     └─────┬─────┘                │
           ▼                      ▼
         Cube ──► L0C（累加结果）──► Vector
```

- 矩阵乘的输入存放在 L0A / L0B，结果写入 L0C；Vector 单元只能读写 UB
- L1 Buffer 是较大的中间缓冲，用于暂存从 HBM 读入、尚未分发到 L0A / L0B 的数据，使同一块数据可以被多次送入 Cube
- 数据在各级 buffer 之间的传输由 MTE 完成，需要在程序中显式安排，包括传输的源地址、目的地址、数据量，以及传输与计算之间的同步

- 自 910B 起，Cube 和 Vector 被拆分到不同的核（AIC 与 AIV），二者之间的中间结果需要经由 L2 或 HBM 传递
  - 融合「矩阵乘 + 激活」这类算子时需要考虑这一点
    - 在 GPU 上，Tensor Core 的累加结果位于寄存器中，可以直接在寄存器中施加激活函数
    - 在 910B 上，AIC 的结果必须先写出，再由 AIV 读入，多出一次数据传输
  - 算子开发者通常通过让 AIC 与 AIV 按流水方式交替处理不同的数据块来掩盖这部分开销

##### 7.2.5.3 Ascend C 编程模型

- Ascend C 是基于 C++ 的扩展语言
- 算子按「搬入（CopyIn）→ 计算（Compute）→ 搬出（CopyOut）」三个阶段编写，阶段之间通过队列连接
- 将队列深度设为 2 即实现双缓冲（double buffering）：两块缓冲区交替使用，一块参与计算时另一块进行数据传输，从而使传输与计算重叠
- 以下是官方样例中加法算子的结构（节选）：

```cpp
constexpr int32_t BUFFER_NUM = 2;   // 双缓冲

class KernelAdd {
public:
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileNum * BUFFER_NUM; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t i) {
        LocalTensor<half> x = inQueueX.AllocTensor<half>();
        LocalTensor<half> y = inQueueY.AllocTensor<half>();
        DataCopy(x, xGm[i * tileLength], tileLength);   // HBM -> UB
        DataCopy(y, yGm[i * tileLength], tileLength);
        inQueueX.EnQue(x);
        inQueueY.EnQue(y);
    }
    __aicore__ inline void Compute(int32_t i) {
        LocalTensor<half> x = inQueueX.DeQue<half>();
        LocalTensor<half> y = inQueueY.DeQue<half>();
        LocalTensor<half> z = outQueueZ.AllocTensor<half>();
        Add(z, x, y, tileLength);                       // Vector 单元
        outQueueZ.EnQue(z);
        inQueueX.FreeTensor(x);
        inQueueY.FreeTensor(y);
    }
    __aicore__ inline void CopyOut(int32_t i) {
        LocalTensor<half> z = outQueueZ.DeQue<half>();
        DataCopy(zGm[i * tileLength], z, tileLength);   // UB -> HBM
        outQueueZ.FreeTensor(z);
    }
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;
    int32_t tileNum, tileLength;
};
```

- 代码中 `GlobalTensor` 表示位于 HBM 的数据，`LocalTensor` 表示位于 UB 的数据
- `AllocTensor` 从队列中申请一块空闲缓冲，`EnQue` 把填好数据的缓冲交给下一阶段，`DeQue` 在数据就绪前会阻塞等待，`FreeTensor` 把用完的缓冲归还队列
- 三个阶段的指令分别进入数据传输队列和 Vector 指令队列并行执行
  - 队列机制自动处理阶段之间的同步
  - 第 $i$ 块在 Vector 单元上计算时，第 $i+1$ 块已经在由 MTE 搬入

- 同样的向量加法，CUDA 版本（1.1.6）只需一行 `c[i] = a[i] + b[i]`，访存延迟由 warp 切换隐藏
- Ascend C 则需要开发者自行分块、安排数据传输并实现双缓冲
- 两段代码在长度上的差异，反映了硬件管理与软件管理片上存储这两种方式在编程负担上的差别

##### 7.2.5.4 CANN 软件栈

| 层 | 昇腾 | 对应 NVIDIA |
|---|---|---|
| 驱动与运行时 | CANN（Compute Architecture for Neural Networks） | CUDA Driver / Runtime |
| 算子开发 | Ascend C、TBE | CUDA C++、Triton |
| 算子库 | aclnn、ATB（Ascend Transformer Boost） | cuBLAS、cuDNN |
| 图编译 | GE（Graph Engine） | TensorRT、Inductor |
| 通信库 | HCCL | NCCL |
| 框架接入 | MindSpore、torch_npu（PyTorch 插件） | PyTorch 原生 |
| 训练框架 | MindSpeed（Megatron 适配） | Megatron-LM |
| 推理 | MindIE、vLLM-Ascend | vLLM、TensorRT-LLM |

- 昇腾的软件栈与 NVIDIA 在层次上一一对应
  - 最底层的 CANN 提供设备管理、内存分配和 kernel 启动接口
  - 其上是算子开发语言和算子库
  - 图编译器 GE 对整个计算图做融合与内存规划
  - HCCL 提供集合通信
  - 最上层是框架接入和训练、推理框架

- PyTorch 用户最常用的接入方式是 `torch_npu`：执行 `import torch_npu` 后，将代码中的 `cuda` 替换为 `npu`，多数模型代码即可直接运行
  - 这种方式下，每个 PyTorch 算子被映射到 aclnn 库中对应的 NPU 算子逐个执行，性能通常不及 GPU 上的同等实现
  - 进一步优化时，需要将注意力、RMSNorm、RoPE 等性能关键的算子替换为 ATB 或 MindSpeed 提供的融合算子

##### 7.2.5.5 昇腾芯片型号与超节点

- 910B 是单芯片产品，FP16 算力约 300 余 TFLOPS，配备 64 GB HBM2e，常见于 8 卡服务器
  - 其算力约为 H100 的三分之一，HBM 带宽约为 H100 的一半

- 910C 将两颗 910B 级别的 die 合封在同一封装内，BF16 算力约 780 TFLOPS，配备 128 GB HBM
  - 两颗 die 之间通过封装内互连通信，对软件呈现为一个设备

- CloudMatrix 384 是由 384 颗 910C 组成的超节点，芯片之间通过全光互连连接，任意两卡之间的带宽一致，不存在节点内与节点间的带宽差异
  - 其设计思路与 NVL72 相同，都是扩大 scale-up 域，使张量并行和专家并行可以在更多芯片之间高效展开
  - CloudMatrix 384 以更多的芯片数量弥补单卡算力的差距，整体算力可与 NVL72 相当甚至更高
  - 代价是芯片数量更多、光模块用量更大，整体功耗明显更高

##### 7.2.5.6 端侧 NPU

- 手机和 PC 中的 NPU（Apple Neural Engine、高通 Hexagon、Intel NPU）属于另一类器件
  - 它们的功耗为数瓦，受电池容量和无风扇散热条件限制
  - 以 INT8 / INT4 运算为主，仅用于推理
  - 没有独立的 HBM，权重存放在与 CPU、GPU 共享的 LPDDR 内存中，带宽只有数十到一百余 GB/s，约为数据中心 GPU 的几十分之一

- 端侧运行大模型同样受 decode 阶段的带宽限制
  - 例如一个 7B 模型以 BF16 存储时权重约 14 GB，在 60 GB/s 的带宽下，每生成一个 token 至少需要约 0.23 s
  - 量化到 4 bit 后权重约 3.5 GB，同样带宽下每 token 约 0.06 s
  - 因此将权重量化到 4 bit 是端侧部署的通行做法

#### 7.2.6 其他架构路线

##### 7.2.6.1 AWS Trainium

- Trainium 的计算核心称为 NeuronCore，包含：
  - Tensor Engine（脉动阵列，执行矩阵乘）
  - Vector Engine（向量运算）
  - Scalar Engine（逐元素的标量运算与激活函数）
  - GpSimd Engine（由通用处理器核组成，用于执行难以映射到其他引擎的自定义运算）
  - 片上 SRAM 由软件管理，架构思路与 TPU 接近

- 软件栈为 Neuron SDK，它通过 PyTorch/XLA 或 JAX 接收模型，经编译器生成 NeuronCore 的指令
  - 需要手写 kernel 时使用 NKI（Neuron Kernel Interface），其编程方式与 Pallas、Triton 类似，以张量块为单位描述计算和数据传输
  - Anthropic 的 Project Rainier 集群采用 Trainium 2

##### 7.2.6.2 Groq LPU

- Groq LPU 不使用 HBM，权重全部存放在片上 SRAM 中，每颗芯片约 230 MB
  - SRAM 与计算单元位于同一颗芯片上，访问带宽可达数十 TB/s 量级，比 HBM 高出一个数量级

- LPU 的编译器在编译期将每条指令和每次数据传输安排到确定的时钟周期
  - 硬件既没有缓存，也没有动态调度
  - 芯片之间的数据传输同样按编译期确定的时刻进行
  - 运行时间因此完全可预测，同一请求每次执行的延迟相同

- 单芯片容量小的直接后果是芯片数量多
  - 一个 70B 模型的 8 bit 权重约 70 GB，按每颗芯片 230 MB 计算，至少需要约 300 颗芯片，实际部署需要数百颗芯片级联

- 这种设计的优点是 decode 延迟极低，因为 decode 受带宽限制，而 SRAM 带宽远高于 HBM
- 缺点是单芯片容量小、所需芯片数量多，KV cache 也占用宝贵的 SRAM 容量，且不适合训练，因为训练所需的优化器状态和激活远超片上容量

##### 7.2.6.3 Cerebras WSE

- Cerebras WSE（Wafer-Scale Engine）以整片 300 mm 晶圆制成一颗芯片
  - 常规芯片从晶圆上切下，单颗面积受光刻机单次曝光范围（约 26 mm × 33 mm，即约 858 mm²）限制
  - WSE 则在曝光区域之间也布置连线，使整片晶圆上的所有核心连成一个整体
  - WSE-3 约有 90 万个核心和 44 GB 片上 SRAM

- 片内带宽极高，原本需要跨芯片的通信都变成了片内通信
- 主要难点有三方面：
  - 一是良率，整片晶圆上必然存在制造缺陷，WSE 通过预留冗余核心并在测试后绕开缺陷核心来解决
  - 二是散热，整片晶圆的功耗达到十几千瓦，需要专门设计的水冷系统
  - 三是容量，44 GB 的 SRAM 放不下大模型的全部权重，需要将权重存放在外部存储中，按层流式送入晶圆

#### 7.2.7 架构对比

##### 7.2.7.1 架构特征对比

| | GPU（H100） | TPU（v5p / v7） | 昇腾 NPU（910B / C） | Groq LPU |
|---|---|---|---|---|
| 矩阵单元 | 每 SM 4 个 Tensor Core | 128×128 / 256×256 脉动阵列 | 16×16×16 Cube | 矩阵单元 + 向量单元 |
| 执行模型 | SIMT，硬件动态调度 warp | VLIW，编译器静态调度 | 标量单元发射指令，多单元并行 | 完全静态调度 |
| 片上存储 | 硬件缓存（L1/L2）+ shared memory | 软件管理的 VMEM | 软件管理的 L1 / L0 / UB | 全 SRAM |
| 延迟隐藏 | warp 切换 | 编译器安排的 DMA 流水线 | 队列双缓冲 | 时序确定，无需隐藏延迟 |
| 芯片互连 | NVLink（≤72 卡）+ IB/RoCE | ICI 3D torus + OCS，覆盖整个 pod | HCCS / 灵衢 + RoCE | 芯片间直连 |
| 编程 | CUDA、Triton | JAX / XLA、Pallas | CANN、Ascend C | 由编译器自动完成 |
| 动态形状 | 支持良好 | 支持较差，需要重新编译或 padding | 中等，静态图性能更好 | 支持较差 |

- VLIW 的定义见 7.2.4.7
- HCCS 与灵衢是华为的芯片间高速互连

- 表中各行之间存在对应关系
  - 执行模型越静态，片上存储越倾向于由软件管理，延迟隐藏越依赖编译器的提前安排，对动态形状的支持也越差
  - GPU 处于这一谱系的动态一端，Groq LPU 处于静态一端，TPU 和昇腾位于二者之间

##### 7.2.7.2 Roofline 指标对比

- Roofline 模型以算术强度为横轴、可达性能为纵轴，性能上限为 $\min(\text{峰值算力},\ \text{带宽}\times\text{算术强度})$，两段上限的交点即为脊点（1.3）
- 按 1.3 节的定义计算各芯片的脊点（BF16 稠密，Ironwood 使用 FP8）：

| 芯片 | 峰值（TFLOPS） | HBM 带宽（TB/s） | 脊点（FLOP/B） |
|---|---|---|---|
| H100 SXM | 989 | 3.35 | ~295 |
| B200 | ~2250 | 8 | ~280 |
| MI300X | ~1307 | 5.3 | ~247 |
| TPU v5p | 459 | 2.76 | ~166 |
| TPU v7（Ironwood，FP8） | ~4600 | ~7.4 | ~620 |
| 昇腾 910B（FP16） | ~320 | ~1.6 | ~200 |

- 脊点越高，kernel 越需要大 batch 和算子融合才能达到算力上限
- 以 BF16 矩阵乘为例，batch 为 $b$ 时，每读取一个 2 字节的权重完成 $2b$ 次浮点运算，算术强度约为 $b$ FLOP/B
  - 在 H100 上，batch 需要达到约 300 才能进入算力受限区
- decode 阶段的算术强度只有 1 左右（1.3.5），在上述任何一颗芯片上都受带宽限制，此时应比较的是单位成本所能获得的 HBM 带宽和容量

##### 7.2.7.3 负载与架构的匹配

| 负载 | 推荐架构 | 原因 |
|---|---|---|
| 新模型结构的研究 | GPU | 算子覆盖最全，代码修改与迭代最快 |
| 超大规模稠密预训练 | GPU 或 TPU | TPU 的 pod 级 ICI 适合大规模切分，GPU 的生态和框架成熟 |
| MoE 训练和推理 | GPU（NVL72）、TPU、CloudMatrix | 需要较大的 scale-up 域支撑 EP |
| 大规模在线推理 | 取决于每 token 成本 | decode 取决于带宽和容量，prefill 取决于算力 |
| 极低延迟 decode | Groq、Cerebras | SRAM 带宽高 |
| 端侧 | 手机 / PC NPU | 受功耗约束 |

- 研究新模型结构时，迭代速度比单次运行的效率更重要，新算子往往只能先在 GPU 上实现，因此 GPU 是默认选择
- 超大规模稠密预训练的计算形态规则、持续时间长，TPU 的静态编译限制影响较小，pod 级互连的优势则能充分发挥
- MoE 模型的每一层都需要在专家之间做 All-to-All 通信，scale-up 域的大小直接决定了专家并行能否高效扩展
- 在线推理的选型取决于 prefill 与 decode 的比例和延迟目标，需要按 7.2.7.4 的方法测量实际负载后再定

##### 7.2.7.4 负载画像：选型前需要测量的指标

- 无论是设计一款加速器还是为一项业务选择硬件，都应先对目标负载做画像（workload profiling）
  - 即测量负载在各维度上的分布，而不是只看平均值或峰值规格
- 需要测量的指标包括以下几类：

- 输入与输出长度的分布
  - 除均值外还应统计 P50 和 P95（第 50 和第 95 百分位数）
  - 长度分布决定了 prefill 的计算量、KV cache 的容量需求和 decode 的步数
  - 例如，均值为 2K token 而 P95 为 32K token 的负载，KV cache 容量必须按长尾请求配置，否则少数长请求会挤占全部显存

- 并发量与流量模式
  - 包括平均并发、峰谷比和请求到达过程（均匀到达还是突发到达）
  - 并发量决定了 decode 能组成多大的 batch，进而决定了算术强度和带宽利用率
  - 峰谷比决定了需要按峰值配置多少冗余容量

- prefill 与 decode 的比例，以及每个请求要求的 token 生成速度
  - prefill 占比高的负载（如长文档摘要）主要受算力限制
  - decode 占比高的负载（如长篇生成、多轮 Agent 调用）主要受带宽限制
  - 二者适合的硬件不同

- 各类数据的大小
  - 包括模型权重、KV cache、激活和临时缓冲区
  - 这些数据的总量决定了单个模型实例至少需要多少块卡，以及是否需要跨节点切分

- 资源利用率
  - 包括显存带宽利用率、计算利用率和跨设备通信量
  - 利用率能直接指出当前的瓶颈所在：带宽利用率接近上限而计算利用率很低，说明负载受带宽限制，增加算力不会带来收益

- 成本与可靠性
  - 包括每 token 能耗、每请求成本、故障率和重试率
  - 一个平台的单次执行效率再高，如果故障率高、重试频繁，有效吞吐也会下降

- 模型演化风险
  - 需要评估未来模型可能的变化方向，例如从稠密模型转向 MoE、注意力结构的变体（如 MLA、线性注意力）、更低精度的数值格式
  - 硬件服役期长，选型时应考虑这些变化是否会使所选硬件的优势消失（7.2.2.5）

#### 7.2.8 跨平台迁移成本

##### 7.2.8.1 迁移评估清单

从 GPU 迁移到其他平台，或在多个平台之间并行部署时，应逐项评估以下内容。

1. 关键算子是否有高效实现，包括 FlashAttention、MLA、MoE 路由与 grouped GEMM、paged attention 和 FP8 GEMM
   - grouped GEMM：将多个形状不同的小矩阵乘合并为一次调用，用于 MoE 中各专家的计算
   - 这些算子在 GPU 上都有经过大量优化的实现
   - 如果目标平台缺少其中任何一个，就需要自行开发，或者退回到由小算子拼接的低效实现
2. 通信库是否成熟，即与 NCCL 对应的 RCCL、HCCL 或 ICI 集合通信
   - 应实测 All-to-All 和大规模 AllReduce（各 rank 的数据归约后将结果分发给所有 rank）的 busbw
   - busbw 是 nccl-tests 报告的总线带宽，已按集合通信算法的数据量折算，与参与的 GPU 数量无关
   - busbw 可以直接与链路的标称带宽比较
3. 框架是否有官方适配，即 Megatron / FSDP / vLLM 在目标平台上是否有官方维护的版本，以及与上游版本相比滞后多少
   - 滞后的版本可能缺少最新的并行策略和推理优化
4. 数值是否一致，即同一模型在两个平台上的 loss 曲线能否对齐
   - 不同平台的累加顺序、舍入方式和算子实现各不相同，无法做到逐位一致，只能要求统计意义上的一致（4.10.2 的 L0 级）
   - 迁移时应先用小模型对比两个平台的 loss 曲线，确认偏差在随机种子造成的波动范围以内，再进行大规模训练
5. 工具是否完备，包括 profiler、内存分析和故障诊断工具。缺少这些工具时，性能问题和训练中断的排查时间会成倍增加。
6. 人员是否充足，即能为该平台编写和调优 kernel 的工程师数量。这往往是迁移中最难解决的约束。

##### 7.2.8.2 性价比的度量方法

- 比较对象不应是峰值 TFLOPS，而应是「实测 MFU × 峰值 ÷ 每小时价格」，其中 MFU（Model FLOPs Utilization）为模型实际完成的有效算力与硬件峰值算力之比
  - 峰值高 30% 而 MFU 低 30% 的平台，有效算力为原来的 $1.3\times0.7=0.91$ 倍，反而更低
  - MFU 只能在目标模型和目标并行配置下实测得到，不能根据其他模型的公开数据推断

- 推理场景比较的是「相同 SLO 下每百万 token 的成本」（计算方法见 5.13）
  - SLO（Service Level Objective，服务水平目标）：指服务承诺的延迟指标，例如 95% 的请求首 token 延迟小于 1 秒
  - 放宽 SLO 可以组成更大的 batch，降低每 token 成本，因此比较不同平台时必须固定相同的 SLO

#### 7.2.9 先进封装与 HBM

前面各节讨论的是芯片内部的架构，本节介绍与 AI 加速器相关的封装基本概念：

- 现代加速器的显存带宽、多 die 互连和功耗上限，在很大程度上由封装技术决定

##### 7.2.9.1 基本概念：晶圆、裸片、封装与中介层

- 晶圆（wafer）：制造集成电路的圆形硅片，直径通常为 300 mm，一片晶圆上同时制造数十到数百颗芯片
- 裸片（die）：晶圆切割后得到的单块芯片，即包含完整电路的一小片硅
  - 单个 die 的面积受光刻机单次曝光范围的限制，最大约 858 mm²，称为光罩极限（reticle limit）
  - H100 的 die 面积约 814 mm²，已接近这一极限
- 封装（package）：包裹 die 的结构，为 die 提供机械支撑、对外的电气连接、散热路径和环境保护
- 封装基板（substrate）：位于 die 与电路板之间的多层布线板，通常由有机材料制成
  - 负责将 die 上密集的信号和电源连接扇出到间距更大的焊球，再连接到电路板
- 中介层（interposer）：位于 die 与封装基板之间的高密度布线层，通常由硅制成
  - 硅中介层可以用芯片制造工艺刻出微米级的细线，布线密度比有机基板高一到两个数量级，用于连接并排放置的多个 die
- 芯粒（chiplet）：指将一个系统的功能拆分到多个较小的 die 上，再通过封装互连组合为一个整体
  - 拆分的动机有两个：
    - 一是突破光罩极限，例如 B200 由两颗接近光罩极限的 die 组成，二者之间通过 10 TB/s 的片间接口互连
    - 二是提高良率，制造缺陷的数量与面积成正比，小 die 的良率高于大 die，且不同功能的 die 可以采用不同的工艺制造

##### 7.2.9.2 2D、2.5D 与 3D 集成

按 die 之间的连接方式，封装可以分为三类：

- 2D 封装中，多个芯片各自封装后焊接在电路板上，或并排放在同一块有机封装基板上，芯片之间的信号经过基板或电路板走线
  - 连线长、密度低，芯片之间的带宽一般只有数百 GB/s

- 2.5D 封装中，多个 die 并排放置在同一块硅中介层（或等效的高密度重布线层）上，die 之间的信号经过中介层上的细密走线
  - 连线密度和带宽远高于 2D，但 die 仍然处于同一平面
  - 台积电的 CoWoS（Chip-on-Wafer-on-Substrate）是这一类技术的代表：先将计算 die 和 HBM 堆栈安装在硅中介层上，再将中介层安装在有机封装基板上
  - H100、B200、MI300X 等数据中心加速器都采用 CoWoS 类的 2.5D 封装

- 3D 封装中，die 在垂直方向上堆叠，上下层之间通过穿过硅的垂直连接通信
  - 垂直连线的长度只有几十微米，单位面积可以容纳的连接数远高于平面走线，带宽和能效更高，但散热更困难（7.7.6）
  - AMD 的 3D V-Cache 将一层 SRAM die 堆叠在 CPU die 之上，MI300X 将计算 die 堆叠在 IO die 之上

- 实际产品中，2.5D 与 3D 经常同时使用
  - HBM 本身是 3D 堆叠结构，又与计算 die 一起放在中介层上构成 2.5D 封装
  - MI300X 在此基础上，又在计算 die 与 IO die 之间采用了 3D 堆叠

##### 7.2.9.3 HBM 的结构与 2.5D 集成

- HBM 由多层 DRAM die 垂直堆叠而成，底部是一层逻辑 die（base die），负责与处理器通信和管理各层 DRAM
  - HBM3e 通常堆叠 8 层或 12 层 DRAM die，HBM4 最多可达 16 层
  - 各层 die 之间通过硅通孔（7.2.9.4）垂直连接

- HBM 获得高带宽的方式是加宽接口，而非提高单根信号线的速率
  - 每个 HBM 堆栈有 1024 位宽的数据接口（HBM4 为 2048 位），每根信号线的速率只有每秒数 Gbit，单个 HBM3e 堆栈的带宽约 1.2 TB/s 左右
  - 相比之下，GDDR 显存每颗芯片只有 32 位接口，依靠每根信号线 20 Gbit/s 以上的高速率获得带宽
  - 较低的信号速率降低了每比特的传输能耗，但 1024 位的接口加上地址、控制和电源线，每个堆栈需要数千条连线，有机基板的布线密度无法容纳
  - 因此 HBM 必须与计算 die 并排放在硅中介层上，通过中介层上的细密走线连接，两者的距离只有几毫米

- H100 的计算 die 周围放置了 5 个有效的 HBM3 堆栈，总带宽 3.35 TB/s
- B200 的两颗计算 die 共连接 8 个 HBM3e 堆栈，总带宽 8 TB/s
- HBM 堆栈的数量受计算 die 边长的限制，因为每个堆栈需要占用 die 边缘的一段接口，这也是 HBM 容量和带宽难以随算力同步增长的原因之一

##### 7.2.9.4 垂直互连：TSV、微凸点与混合键合

三种连接结构是 3D 堆叠和 2.5D 封装都依赖的技术：

- TSV（Through-Silicon Via，硅通孔）：在硅片上刻蚀出的贯穿孔，内部填充铜等导电材料，形成穿过整个硅片厚度的垂直导线
  - HBM 中每层 DRAM die 都有数千个 TSV，使信号和电源可以从底部逻辑 die 一直传到最上层
  - 硅中介层中的 TSV 则把中介层上表面的走线连接到下方的封装基板
  - TSV 占用硅片面积，且刻蚀和填充过程会在周围硅中产生应力，因此其密度和位置都受到限制

- 微凸点（micro-bump）：die 之间的小尺寸焊料连接点，间距通常为 40–55 µm
  - 两个 die 对准后，通过加热使焊料熔化，形成电气和机械连接
  - 微凸点工艺成熟，是当前 HBM 堆叠和 2.5D 封装的主流方式，但焊料凸点的间距难以继续缩小，凸点本身也会增加层间高度和热阻

- 混合键合（hybrid bonding）：不使用焊料，而是将两个 die 表面的铜焊盘和周围的介质层直接键合在一起
  - 先将表面抛光到原子级平整，在室温下使介质层贴合，再加热使铜焊盘相互扩散形成连接
  - 连接间距可以缩小到 10 µm 以下，单位面积的连接数比微凸点高一个数量级以上，层间距离更短，电阻、电容和热阻都更低
  - 代价是对表面平整度、洁净度和对准精度的要求极高，任何微小颗粒都可能导致键合失败，制造成本和良率风险都高于微凸点

- 三种技术的共同取舍是：连接越密、距离越短，带宽密度和能效越高，但对准精度、良率、热机械可靠性（7.7.7）和制造成本的要求也越高
### 7.3 互连：scale-up 与 scale-out 网络

GPU 之间的互连分为两个层级，二者的带宽相差约一个数量级。

#### 7.3.1 两级网络结构

两级网络的定义：

- scale-up 网络在一个高速互连域内连接加速器
  - 在 NVIDIA 平台上由 NVLink 和 NVSwitch 构成
  - 所覆盖的 GPU 集合称为 NVLink 域
- scale-out 网络连接不同的节点
  - 通常采用 InfiniBand（IB）或 RoCE

下图是 8 卡节点的典型连接方式：

```text
         ┌──────────── scale-up 网络（NVLink / NVSwitch）────────────┐
         │  GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7  （一个 NVLink 域）│
         └────┬────┬────┬────┬────┬────┬────┬────┬──────────────────┘
              NIC  NIC  NIC  NIC  NIC  NIC  NIC  NIC    每卡一张 400G 网卡
              │    │    │    │    │    │    │    │
         ═════╪════╪════╪════╪════╪════╪════╪════╪═══  scale-out 网络（IB / RoCE）
              rail0 rail1 ...                  rail7
```

图中的连接方式：

- 每块 GPU 配备一张网卡（NIC，Network Interface Card）
  - 各节点的同号网卡接入同一组交换机，构成一条 rail（7.4.3）
- 节点内任意两块 GPU 通过 NVSwitch 以 NVLink 带宽直接通信
- 不同节点的 GPU 之间则要经过各自的网卡和 scale-out 网络

| 网络 | 连接范围 | 带宽（每 GPU 单向） | 延迟 | 用途 |
|---|---|---|---|---|
| NVLink 4 + NVSwitch | 节点内 8 卡 | 450 GB/s | 1 µs 以内 | TP、EP、CP |
| NVLink 5（NVL72） | 72 卡一个域 | 900 GB/s | 同上 | 更大规模的 TP / EP |
| InfiniBand NDR / XDR | 跨节点 | 50 / 100 GB/s | 1–2 µs | DP、PP、跨节点 EP |
| RoCE v2（以太网 RDMA） | 跨节点 | 与 IB 同一量级 | 略高，依赖无损网络配置 | 同上，成本更低 |

表中的缩写与速率换算：

- TP、EP、CP、PP、DP 分别指以下几种并行方式
  - 张量并行（TP）
  - 专家并行（EP）
  - 上下文并行（CP）：将长序列切分到多块 GPU 上
  - 流水线并行（PP）：将模型的不同层放在不同 GPU 上
  - 数据并行（DP）：每块 GPU 持有完整模型副本，处理不同数据
- InfiniBand NDR 的单端口速率为 400 Gbit/s，即 50 GB/s
- XDR 为 800 Gbit/s，即 100 GB/s

两级网络的带宽差决定了各类并行的放置方式：

- TP 和 EP 每一层都要通信
  - 通信量与激活大小成正比
  - 只有放在 NVLink 域内才不会成为瓶颈
- DP 每个训练步只同步一次梯度
  - 且可以与反向计算重叠
  - 能够容忍 scale-out 网络的较低带宽
- PP 只在相邻 stage 之间传递激活
  - 通信量最小

#### 7.3.2 RDMA 和 GPUDirect

RDMA（Remote Direct Memory Access，远程直接内存访问）指网卡直接读写远端主机的内存：

- 不经过对端的 CPU 和操作系统内核协议栈
- 传统的 TCP/IP 通信中
  - 数据要在用户态缓冲区和内核缓冲区之间拷贝
  - 并由 CPU 处理协议，每条消息都有数十微秒的软件开销
- RDMA 由网卡硬件完成协议处理和数据搬运
  - 延迟可降到微秒级
  - 且不占用 CPU

GPUDirect RDMA 是 NVIDIA 的一项技术，使网卡可以直接读写 GPU 显存，不经过主机内存中转：

- 没有 GPUDirect 时，发送数据需要先从 GPU 显存拷贝到主机内存，再由网卡从主机内存读出
  - 接收端同样多一次拷贝
  - 带宽受 PCIe 与主机内存的双重限制

二者结合后，跨节点 GPU 通信的数据路径为：GPU 显存 → PCIe 交换芯片 → 网卡 → 网络 → 网卡 → GPU 显存。

- 数据只经过 PCIe 交换芯片，不经过 CPU 和主机内存
- 为了使这条路径最短，GPU 与网卡应挂接在同一个 PCIe 交换芯片下（拓扑亲和性，7.4 节）
  - 如果二者分属不同的 CPU 插槽，数据要跨越 CPU 之间的互连
  - 带宽会显著下降

#### 7.3.3 InfiniBand 与 RoCE v2

| | InfiniBand | RoCE v2 |
|---|---|---|
| 协议栈 | 原生 RDMA，基于信用的流控，天然无损 | RDMA over UDP/IP，需要 PFC + ECN 实现无损 |
| 管理 | 由 Subnet Manager（子网管理器）集中管理 | 采用标准以太网运维方式 |
| 拥塞控制 | 硬件内建 | DCQCN 等算法，需要调参 |
| 成本与生态 | 价格较高，供应商以 NVIDIA（Mellanox）为主 | 价格较低，多厂商供货 |
| 大规模案例 | 多数 NVIDIA 参考架构 | Meta（Llama 3 使用 RoCE 24K 卡集群）、阿里 HPN、字节 |

「无损」的含义与必要性：

- 「无损」指网络不因交换机缓冲区溢出而丢包
- RDMA 的传输协议在丢包后的重传代价很高
  - 少量丢包就会使吞吐大幅下降
  - 因此要求网络无损

InfiniBand 的无损机制是基于信用的流控：

- 接收端预先告知发送端自己还有多少缓冲区空间（信用）
- 发送端只在有信用时才发送
  - 从机制上保证缓冲区不会溢出

RoCE v2 将 RDMA 承载在普通以太网的 UDP/IP 之上：

- 以太网本身不保证无损，因此需要额外的机制
- PFC（Priority Flow Control，基于优先级的流控）
  - 在接收端缓冲区接近满载时，向上游发送暂停帧
  - 使上游暂停发送对应优先级的流量
  - PFC 能防止丢包，但暂停会逐跳向上游传播，严重时造成大范围阻塞
  - 因此还需要拥塞控制来从源头降低发送速率
- ECN（Explicit Congestion Notification，显式拥塞通知）
  - 由交换机在队列长度超过阈值时，在报文头部标记拥塞状态
- DCQCN（Data Center Quantized Congestion Notification）是 RoCE 上常用的端到端拥塞控制算法
  - 接收端收到带 ECN 标记的报文后通知发送端
  - 发送端据此降低发送速率
- DCQCN 的多个参数（标记阈值、降速比例、恢复速度）需要根据网络规模和流量模式调整
  - 这是 RoCE 部署运维成本较高的主要原因

Subnet Manager 是 InfiniBand 网络中集中计算路由表并配置各交换机的管理程序。

#### 7.3.4 显存与互连带宽的数量级对比

- 3.8.1 节已给出各级带宽的对比，此处只保留其比值
  - H100 上 HBM : NVLink : IB ≈ 67 : 9 : 1
  - 对应的带宽分别为 3350、450 和 50 GB/s
- 同样大小的数据
  - 经过 IB 传输所需的时间是经过 NVLink 的 9 倍
  - 是从 HBM 读取的 67 倍
- 设计并行策略时，应让通信量最大的操作使用带宽最高的链路

### 7.4 网络拓扑与亲和性

#### 7.4.1 节点内拓扑

```bash
nvidia-smi topo -m
```

该命令输出一个矩阵，给出节点内每对 GPU、网卡和 CPU 之间的连接类型：

- `NV18` 表示两卡之间有 18 条 NVLink
- `PIX` 表示经过同一个 PCIe 交换芯片
- `NODE` 表示跨越 PCIe 根复合体但位于同一 NUMA 节点
  - 根复合体（root complex）：CPU 上连接 PCIe 设备的控制器
- `SYS` 表示跨越 CPU 插槽，速度最慢

NUMA（Non-Uniform Memory Access，非一致内存访问）节点指一个 CPU 插槽及其直连内存构成的单元：

- 双路服务器有两个 NUMA 节点
- CPU 访问本节点的内存速度快
- 访问另一节点的内存需要经过 CPU 之间的互连
  - 延迟更高、带宽更低

#### 7.4.2 亲和性配置

亲和性（affinity）指将相互通信的设备和进程放置在物理路径最短的位置上。

| 对象 | 做法 |
|---|---|
| GPU 与网卡 | 每块 GPU 使用与其位于同一 PCIe 交换芯片下的网卡（NCCL 会自动探测，在容器中需保证拓扑信息可见） |
| GPU 与 CPU | 将数据加载进程绑定到 GPU 所在 NUMA 节点的 CPU 核（`numactl`），避免跨插槽访存 |
| rank 映射 | 同一 TP 组的 rank 必须位于同一节点（或 NVLink 域）内，`torchrun` 按连续方式分配 local rank |

GPU 与网卡的亲和性影响跨节点通信带宽：

- NCCL 通过读取 `/sys` 下的 PCIe 拓扑信息为每块 GPU 选择最近的网卡
- 在容器中，如果没有挂载这些信息，NCCL 可能选错网卡
  - 通信路径会跨越 CPU 插槽
  - 带宽下降一半以上

GPU 与 CPU 的亲和性影响数据加载：

- 数据加载进程把样本从主机内存拷贝到 GPU
- 如果进程运行在另一个 NUMA 节点的 CPU 核上，每次拷贝都要跨越 CPU 之间的互连

rank 映射决定了并行组的物理位置：

- `torchrun` 将同一节点上的进程分配连续的 rank 编号
- 因此按 rank 编号划分 TP 组时，应使每个 TP 组的 rank 编号连续
  - 保证 TP 组落在同一节点内

#### 7.4.3 跨节点拓扑：fat-tree 与 rail-optimized

数据中心网络通常分为两层或三层：

- leaf 交换机是直接连接服务器的接入层交换机
- spine 交换机是连接各 leaf 交换机的上层交换机
- 规模更大的集群在 spine 之上还有第三层（core 层）

Fat-tree（胖树）是由多层交换机组成的树形拓扑：

- 与普通树不同，越靠近根部的链路总带宽越高，使上层不成为瓶颈
- 任意两台服务器之间存在多条经过不同 spine 交换机的等价路径
  - 流量可以分散到这些路径上
- 二分带宽（将网络划分为两半时，两半之间的总带宽）满配时
  - 任意一半服务器都能以全速与另一半通信
  - 任意通信模式都不会阻塞

Rail-optimized 是针对多网卡 GPU 服务器的连接方式：

- 每台机器的第 $i$ 张网卡都连接到第 $i$ 组 leaf 交换机，构成第 $i$ 条 rail
  - 例如 8 卡服务器有 8 条 rail，所有服务器的 GPU0 网卡都接在 rail0 的 leaf 交换机上
- 同一 rail 内的通信只经过一跳 leaf 交换机，延迟最低
- 跨 rail 的通信则要经过 spine 交换机
- NCCL 的 ring / tree 算法会优先安排同号 GPU 之间的通信
  - 使流量尽量留在 rail 内
- PXN（PCIe × NVLink）功能处理目标位于其他 rail 的情况
  - 先经节点内 NVLink 将数据转发到位于目标 rail 的那块 GPU
  - 再由该 GPU 的网卡发出
  - 以节点内的 NVLink 转发代替跨 rail 的 spine 转发

收敛比（oversubscription）定义为交换机下行总带宽与上行总带宽之比：

- 例如一台 leaf 交换机有 32 个 400G 端口连接服务器、16 个 400G 端口连接 spine，收敛比为 2:1
  - 意味着当所有服务器同时向外通信时，每台服务器只能获得一半带宽
- 1:1 表示无阻塞
- 训练集群一般要求 spine 层的收敛比接近 1:1
  - 因为数据并行的梯度同步会使大量服务器同时跨 leaf 通信

#### 7.4.4 并行组的物理放置

| 并行组 | 放置 |
|---|---|
| TP | 节点内（或 NVL72 域内） |
| EP | 优先放在节点内；跨节点时采用节点受限路由（每个 token 只路由到有限个节点上的专家） |
| PP | 相邻 stage 尽量位于同一 leaf 交换机下 |
| DP | 跨 leaf / spine 分布，梯度同步与计算重叠 |

放置原则是按通信量从大到小，依次占用带宽从高到低的链路：

- TP 的通信量最大，放在 NVLink 域内
- EP 的 All-to-All 通信量也很大
  - 跨节点时通过限制每个 token 最多发往几个节点来控制跨节点流量
- PP 只在相邻 stage 之间点对点传递激活
  - 放在同一 leaf 下即可
- DP 的梯度同步每步只发生一次
  - 且可以与反向计算重叠
  - 放在最外层

调度器为作业分配节点时应考虑上述约束：

- 将一个作业的节点集中在少数 leaf 交换机下，比分散在整个集群中通信更快
- 受其他作业的干扰也更少

### 7.5 作业调度：gang 调度、资源碎片与抢占

#### 7.5.1 训练作业的调度需求

| 需求 | 含义 |
|---|---|
| Gang 调度 | 一个作业所需的全部资源必须同时分配，否则不启动。例如，需要 1024 块卡的作业只获得 1000 块时不能先行运行 |
| 拓扑感知 | 分配的节点尽量位于同一 leaf / spine 交换机下 |
| 整机分配 | 训练作业通常以整台 8 卡机为单位分配 |
| 长时运行 | 运行时间从数天到数月不等，需要在故障后自动恢复 |

gang 调度（gang scheduling，成组调度）的必要性来自分布式训练的同步特性：

- 所有 rank 在每一步都要参与集合通信
  - 任何一个 rank 未启动，其余 rank 都会在第一次通信时阻塞等待
- 如果调度器像调度普通任务那样逐个分配资源
  - 两个大作业可能各自拿到一半资源后互相等待，形成死锁
  - 已分配的资源也全部闲置

其余三项需求的依据：

- 拓扑感知的依据见 7.4.4
- 整机分配的原因
  - TP 组需要整个 NVLink 域
  - 多个作业共享一台机器时会相互争用 PCIe、网卡和主机内存带宽
- 长时运行意味着作业在运行期间几乎必然遇到硬件故障（7.8.2）
  - 调度系统需要能自动检测故障、替换节点并从 checkpoint 重启作业

#### 7.5.2 Slurm 与 Kubernetes

| | Slurm | Kubernetes |
|---|---|---|
| 起源 | HPC | 云原生微服务 |
| gang 调度 | 原生支持（一个 job 申请 N 个节点） | 需要插件：Volcano、Kueue、Coscheduling |
| 拓扑感知 | `--switches`、topology 插件 | 需要自定义调度插件或节点标签 |
| 适用场景 | 大规模训练 | 推理服务、混合负载、多租户 |

Slurm 起源于高性能计算领域，其调度单位是作业：

- 一个作业申请 N 个节点
  - 调度器要么一次分配全部节点，要么让作业排队
  - 天然满足 gang 调度
- `--switches` 参数可以限制作业所用节点跨越的交换机数量

Kubernetes 起源于微服务部署，其调度单位是单个 Pod（一组共享网络和存储的容器）：

- 默认调度器逐个调度 Pod，不保证一组 Pod 同时启动
- Volcano、Kueue 等插件在默认调度器之上增加了作业级的排队和成组调度
- Kubernetes 的优势在于服务管理、弹性扩缩容和多租户隔离
  - 这些正是推理服务所需要的

实际选择：

- 许多公司在训练中使用 Slurm，或在 Kubernetes 上运行 Slurm 风格的调度器
- 推理服务则使用 Kubernetes

#### 7.5.3 资源碎片

资源碎片指空闲资源的总量足够，但在拓扑上过于分散而无法满足大作业需求的状态：

- 例如，集群有 1000 台机器，空闲的 50 台分散在 20 个 leaf 交换机下
  - 一个需要 32 台机器的作业要么等待，要么跨越多个 leaf 运行
  - 通信性能随之下降
- 调试、评测等小作业若不加约束地占用资源，会将成块的拓扑切分得零散

应对措施包括以下几项：

- 按作业规模划分资源池
  - 例如将集群划分为大作业池和小作业池，小作业只能使用小作业池
  - 大作业池中的 leaf 交换机保持完整
  - 代价是两个池之间无法互相借用空闲资源
- 小作业优先填入已经零散的位置（bin packing，装箱式分配）
  - 调度器在为小作业选择节点时，优先选择已有其他作业运行、剩余空闲节点较少的 leaf 交换机
  - 把完全空闲的 leaf 留给大作业
- 定期整理资源
  - 可以迁移正在运行的小作业以腾出完整的 leaf
  - 或者在作业结束时重新规划分配
  - 迁移需要作业支持从 checkpoint 恢复（7.5.4）

#### 7.5.4 抢占与优先级

抢占（preemption）指高优先级作业到达时，调度器中止低优先级作业以释放其资源：

- 例如主模型训练这类高优先级作业到达时抢占低优先级的实验作业
- 被抢占的作业应能按以下顺序处理：收到信号 → 保存 checkpoint → 退出 → 之后从 checkpoint 恢复
- 因此所有作业都需要支持优雅退出（graceful shutdown）
  - 即在收到终止信号后先保存状态再退出
- 调度器通常先发送 SIGTERM 信号
  - 等待一段宽限期（例如几十秒到几分钟）
  - 若进程仍未退出，则发送无法捕获的 SIGKILL 强制终止

```python
import signal, sys

stop = False
def handle(sig, frame):
    global stop; stop = True
signal.signal(signal.SIGTERM, handle)

for step in range(start, total):
    train_step()
    if stop or step % ckpt_every == 0:
        save_checkpoint(step)
        if stop: sys.exit(0)
```

代码中的信号处理函数只设置一个标志，不在处理函数内直接保存 checkpoint：

- 原因是信号可能在训练步的任意时刻到达，此时参数可能正处于更新的中途
- 等当前训练步完成后再检查标志并保存
  - 可以保证 checkpoint 中的状态一致
- 保存 checkpoint 所需的时间必须短于调度器的宽限期

#### 7.5.5 推理服务的调度

推理服务的调度需求与训练不同，主要有以下三方面：

- 自动扩缩容（autoscaling）指根据负载自动增减服务副本数
  - 扩缩容的依据可以是 QPS（每秒请求数）、队列长度或 KV cache 使用率
  - 对 LLM 服务而言，KV cache 使用率比 QPS 更能反映实际负载
    - 因为请求长度差异很大
  - 扩容的速度受模型加载时间限制
    - 一个 70B 模型的 BF16 权重约 140 GB，从远端对象存储加载可能需要数分钟
  - 常见的加速方法
    - 将权重预先缓存到本地 NVMe
    - 或使用共享存储并行加载（7.6.4）
- 潮汐调度利用推理负载的昼夜变化
  - 白天为推理高峰，夜间请求量下降
  - 可以将空闲 GPU 借给训练或离线批处理任务（如数据合成、模型评测），早高峰前再收回
  - 被借用的作业必须支持抢占
- 多模型混部指多个小模型或 LoRA 适配器共享一组 GPU
  - 单个小模型往往无法用满一块 GPU 的算力和显存，多个模型共享可以提高利用率
  - 多个 LoRA 适配器共享同一个基座模型时，只需为每个适配器额外存储很小的低秩矩阵

### 7.6 存储与 checkpoint IO

#### 7.6.1 训练集群的存储需求

| 数据 | 访问模式 | 量级 |
|---|---|---|
| 训练数据 | 顺序读，读多写少 | 几十 TB 到 PB |
| checkpoint | 突发大量写入，偶尔读取 | 每次 TB 级 |
| 日志、指标 | 小文件持续写入 | 小 |
| 模型权重分发（推理） | 突发并发读取 | 每副本几十到几百 GB |

四类数据的访问模式差别很大，对存储系统的要求也不同：

- 训练数据量大但读取速率稳定
  - 要求总容量大、顺序读吞吐足够
- checkpoint 平时不产生流量，保存时所有 rank 在同一时刻写入
  - 要求峰值写带宽高
- 日志和指标数据量小，但写入频繁
  - 要求小文件和元数据操作的性能
- 推理权重分发在扩容时集中发生，数百个副本同时读取同一份权重
  - 要求高并发读吞吐

#### 7.6.2 存储层级

| 层 | 例子 | 特点 |
|---|---|---|
| 本地 NVMe | 每节点 8×3.84 TB | 每节点 20–50 GB/s，数据随节点故障丢失 |
| 并行文件系统 | Lustre、GPFS、WekaFS、DDN、3FS（DeepSeek） | 聚合带宽达 TB/s 量级，提供 POSIX 语义 |
| 对象存储 | S3、OSS | 容量大、成本低、延迟高，适合冷数据和归档 |

各层的特点：

- 本地 NVMe 是插在每台服务器上的固态硬盘
  - 带宽高、延迟低
  - 数据只存在于该节点上，节点故障后无法访问
  - 因此只能作为缓存或临时存储
- 并行文件系统将数据分条存放在多个存储节点上
  - 客户端可以同时从多个节点读写
  - 聚合带宽随存储节点数量线性增长
  - POSIX 语义指它对外表现为普通的文件系统，程序可以用标准的文件读写接口访问，无需修改代码
- 对象存储以「键-对象」的形式存储数据，通过 HTTP 接口访问
  - 单位容量成本最低
  - 单次访问延迟为数十毫秒，且不支持对文件的局部修改
  - 适合存放原始数据集和历史 checkpoint
- DeepSeek 报告 3FS 在 180 个存储节点上达到约 6.6 TB/s 的聚合读吞吐
  - 平均每个存储节点约 37 GB/s
  - 用于数据加载、checkpoint 以及推理时的 KV cache 存储

#### 7.6.3 checkpoint 带宽算例

以 1024 卡训练 70B 模型为例：

- checkpoint 包含 BF16 权重、FP32 主权重和 Adam 优化器的两个 FP32 动量
  - 每个参数约 12 字节，总大小约 840 GB
- 若要求同步写入部分小于 30 s，需要 $840/30=28$ GB/s 的写入带宽
- 若每 15 分钟保存一次（4.2.3 节）
  - 平均写入带宽仅约 $840/900\approx1$ GB/s
  - 但峰值仍需 28 GB/s

同步与异步 checkpoint 的区别：

- 存储系统需要按峰值配置
  - 因此同步 checkpoint 会使存储带宽在绝大多数时间闲置
- 异步 checkpoint 将峰值分摊到更长的时间内
  - 训练进程只需在阻塞期间把模型状态从 GPU 显存复制到 CPU 内存
    - 每个节点通过 PCIe 并行拷贝，数秒即可完成
  - 之后训练继续进行
  - 由后台线程以较低速率将 CPU 内存中的状态写入存储

#### 7.6.4 checkpoint 与权重的加载开销

checkpoint 加载：

- 1024 个 rank 同时读取 checkpoint 时，如果所有 rank 都打开同一个文件并读取其中的不同部分
  - 并行文件系统的元数据服务器需要在同一时刻处理大量请求，会出现过载
  - 元数据服务器：负责管理文件名、权限和数据块位置的服务
- 分片 checkpoint 将状态按 rank 拆分为多个文件
  - 每个 rank 只读取属于自己的部分
  - 元数据请求和数据流量都分散到不同的文件和存储节点上

推理扩容时，权重加载往往占冷启动时间的大部分。缩短加载时间的方法有两种：

- 将权重预先缓存到每个节点的本地 NVMe
  - 扩容时从本地读取
- 通过 RDMA 从已在运行的副本直接复制权重到新副本的显存
  - 复制速度受网卡带宽限制
  - 在 400G 网卡上传输 140 GB 约需 3 秒

### 7.7 能耗与每 token 成本的物理下限

#### 7.7.1 功耗构成

| 部件 | 功耗 |
|---|---|
| H100 SXM | TDP 最高 700 W |
| B200 | TDP 最高约 1000 W |
| GB200 NVL72 整柜 | 约 120–130 kW |
| 节点其他部分（CPU、内存、网卡、风扇） | GPU 功耗的 20–40% |
| 数据中心开销（制冷、配电） | PUE 1.1–1.5 |

两个指标的定义：

- TDP（Thermal Design Power，热设计功耗）指芯片在持续负载下的设计功耗上限
  - 散热系统必须能持续带走这一功率
- PUE（Power Usage Effectiveness，电能使用效率）定义为数据中心总耗电与 IT 设备耗电之比
  - 越接近 1，表示制冷和配电的额外开销越小
  - PUE 为 1.2 意味着 IT 设备每消耗 1 kW，制冷和配电再额外消耗 0.2 kW

单柜功耗的上升改变了散热方式：

- 传统风冷机柜的功耗一般在 20–40 kW 以内
  - 因为空气的比热容和导热能力有限，再高的功耗需要极大的风量
- GB200 NVL72 整柜功耗达到 120 kW 以上
  - 风冷已无法带走这些热量，必须采用液冷
  - 冷却液通过贴在 GPU 和 CPU 上的冷板直接吸收热量，再经机柜外的换热设备排出
- 液冷还降低了风扇功耗，使 PUE 更接近 1

#### 7.7.2 每 token 能耗估算

H100 的能效基准：

- BF16 峰值为 989 TFLOPS，功耗为 700 W
- 即每焦耳约 $989\times10^{12}/700\approx1.4\times10^{12}$ FLOPs

训练 70B 模型时：

- 每 token 的计算量为 $6N = 4.2\times10^{11}$ FLOPs（$N$ 为参数量）
- 设 MFU 为 45%：

$$E_{\text{token}} = \frac{4.2\times10^{11}}{1.4\times10^{12}\times0.45}\approx 0.67\ \text{J}$$

由此推出整体能耗与成本：

- 计入节点其他部件的开销（×1.3）和 PUE（×1.2）后，约为 1 J/token
- 训练 15T token 共消耗约 $1.5\times10^{13}$ J，即约 4.2 GWh
  - 按电价 0.08 美元/kWh 计算，电费约 33 万美元
  - 远低于 GPU 折旧（约 900 万美元）
- 但几十 MW 量级的电力接入是建设集群的硬性约束
  - 一个 10 万卡的 H100 集群，按每卡 700 W、节点开销 1.3 倍和 PUE 1.2 计算，总功率约 110 MW
  - 相当于一座小型城市的用电负荷
  - 新建这一规模的电网接入通常需要数年

推理 decode 的能耗：

- decode 受带宽限制，算力利用率下降时功耗并不会相应大幅下降
  - 因为 HBM 读取、片上数据传输和静态漏电都在持续消耗功率
- 因此小 batch decode 每 token 的能耗远高于按 FLOPs 计算的理论值
- 以下方法在降低成本的同时也降低了每 token 能耗
  - 增大 batch 使每次权重读取服务更多 token
  - 量化减少每次读取的字节数
  - 投机解码使一次权重读取验证多个 token

#### 7.7.3 数据移动能耗的物理下限

各类操作的能耗量级：

- 从 HBM 读取一个字节的能耗约为数 pJ
- 读取片上 SRAM 的能耗低一个数量级以上
- 跨芯片通信的能耗则再高一个数量级
- 作为比较，一次 BF16 乘加运算本身的能耗在 1 pJ 以下

能耗差异的物理原因：

- 按 7.2.4.3 的动态功耗模型，能耗主要取决于被驱动导线的电容，而电容随距离增长
- 片上 SRAM 与计算单元的距离以微米到毫米计
- HBM 与计算 die 之间要经过中介层上数毫米的走线
- 跨芯片通信则要驱动封装外的长距离电缆或光模块

优化方向：

- 因此能效优化与性能优化的方向一致：使数据尽量留在靠近计算单元的位置
- 以下做法都是在减少长距离数据移动
  - 算子融合
  - 分块
  - 脉动阵列的局部传递
  - 先进封装缩短 die 间距离

#### 7.7.4 摩尔定律与 Dennard 缩放

摩尔定律（Moore's law）是对集成电路晶体管数量增长趋势的经验观察：

- 单颗芯片上可集成的晶体管数量大约每两年翻一番
- 它描述的是制造工艺的进步速度，不是物理定律
- 近年来晶体管尺寸仍在缩小，但每代工艺的周期变长
  - 单个晶体管的成本下降速度明显放缓，先进工艺下甚至不再下降

Dennard 缩放（Dennard scaling）描述的是晶体管等比例缩小时功耗的变化规律。设晶体管的线性尺寸缩小为原来的 $1/\kappa$，同时将电源电压和电流也缩小为 $1/\kappa$，则：

- 单个晶体管的面积缩小为 $1/\kappa^2$，单位面积可容纳的晶体管数量增加到 $\kappa^2$ 倍；
- 电容随尺寸缩小为 $1/\kappa$，开关速度提高，频率可以提高到 $\kappa$ 倍；
- 由 $P\approx\alpha CV^2f$，单个晶体管的功耗变为 $\frac{1}{\kappa}\cdot\frac{1}{\kappa^2}\cdot\kappa=\frac{1}{\kappa^2}$；
- 单位面积的功耗（功率密度）为 $\kappa^2\times\frac{1}{\kappa^2}=1$，保持不变。

Dennard 缩放成立时期的效果：

- 每一代工艺都能在相同面积和相同功耗下提供更多晶体管和更高频率
- 性能的提升几乎不需要付出功耗代价

Dennard 缩放在 2005 年前后失效：

- 原因是电源电压无法继续按比例降低
  - 晶体管的阈值电压（开启所需的最低栅极电压）降低时，关断状态下的漏电流呈指数增长
  - 漏电功耗变得不可接受
  - 因此阈值电压和电源电压都停留在约 1 V 甚至略低的水平
- 电压不再下降后，按上式
  - 单个晶体管的功耗只随电容下降为 $1/\kappa$（频率不变时）
  - 而单位面积的晶体管数量仍增加到 $\kappa^2$ 倍
  - 功率密度随之上升为 $\kappa$ 倍

失效之后的影响：

- 功率密度取代晶体管数量成为芯片设计的首要约束
- 芯片能集成的晶体管越来越多，但受散热能力限制，无法让所有晶体管同时以最高频率工作
  - 这种现象称为暗硅（dark silicon）
- CPU 的主频从此基本停止增长，设计转向多核并行
- 在晶体管数量继续增长而功耗预算固定的条件下
  - 提高性能的主要途径变成提高每焦耳完成的有用运算，即专用化
  - GPU 的 Tensor Core、TPU 的脉动阵列等面向矩阵乘的专用单元，以及 7.2 中讨论的各种 DSA，都是在这一约束下出现的

#### 7.7.5 供电网络：焦耳热、IR 压降与电迁移

供电网络（PDN，Power Delivery Network）指将电能从电源模块经电路板、封装基板、中介层和片上电源网格送达每个晶体管的整条路径：

- AI 加速器的 PDN 面临的核心问题是电流极大
- 功耗 $P=VI$，在功耗固定时，电压越低，电流越大
- H100 的核心电压约 0.7–0.8 V
  - 700 W 的功耗对应约 $700/0.75\approx930$ A 的电流
- B200 的功耗约 1000 W，电流超过 1000 A

电流流过有电阻的导体时会发热，称为焦耳热：

$$P_{\text{Joule}}=I^2R$$

- 焦耳热与电流的平方成正比
  - 在同一电阻下，电流翻倍，发热变为原来的 4 倍
- 因此芯片功耗从 700 W 增加到 1000 W 以上时，PDN 上的损耗和发热增长得更快
- 这部分热量产生在电源模块、电路板走线、封装中的电源连接和片上电源网格中
  - 与计算单元本身的发热叠加

电流流过电阻还会产生电压降，称为 IR 压降（IR drop）：$\Delta V=IR$。

- 芯片能够正常工作的电压范围很窄
  - 假设电压容差为核心电压的 5%，即约 37.5 mV
  - 在 1000 A 的电流下，从电源模块到晶体管的总电阻必须低于 $37.5\ \text{mV}/1000\ \text{A}=37.5\ \mu\Omega$
- 为满足这一要求
  - 封装中需要用大量并联的凸点和 TSV 专门传输电源和地
  - 电路板上的电源模块也要尽量靠近芯片放置
- 除稳态压降外，还有瞬态压降
  - 计算负载突变（例如大量矩阵单元同时开始工作）时，电流在纳秒级时间内急剧变化
  - PDN 的电感会产生额外的瞬态压降
  - 需要依靠封装和片上的去耦电容提供瞬时电流

电迁移（electromigration）是高电流密度下的金属失效机制：

- 导线中高速运动的电子不断撞击金属原子，使原子沿电子运动方向逐渐迁移
- 长时间作用后
  - 原子流失的位置形成空洞，导致电阻升高甚至断路
  - 原子堆积的位置形成凸起，可能与相邻导线短路
- 电迁移导致的失效时间随电流密度升高和温度升高而显著缩短
  - 常用 Black 方程描述：平均失效时间与电流密度的负 $n$ 次方（$n$ 通常约为 2）和温度的 Arrhenius 项成正比
- PDN 设计必须将每根导线、每个凸点和每个 TSV 的电流密度控制在工艺允许的上限以内
  - 这也是封装中需要大量电源连接的原因之一

#### 7.7.6 散热：热传导、热阻与热点

热量的散热路径：

- 芯片产生的热量需要依次穿过硅 die、热界面材料、封装盖板、散热器或冷板
- 最终由空气或冷却液带走
- 热界面材料（TIM，Thermal Interface Material）：填充芯片与散热器之间微小间隙的导热材料

热量在固体中的传导遵循傅里叶定律：

$$\mathbf q=-k\nabla T$$

- 其中 $\mathbf q$ 是热流密度（单位面积上每秒通过的热量，单位 W/m²）
- $k$ 是材料的热导率（单位 W/(m·K)）
- $\nabla T$ 是温度梯度
- 负号表示热量从高温流向低温
- 热导率越高，同样的温差下能传导的热流越大
  - 硅的热导率约为 150 W/(m·K)，铜约为 400 W/(m·K)
  - 而常用热界面材料只有几到十几 W/(m·K)

工程上常用热阻对整条散热路径做一阶估算：

- 热阻 $R_{\text{th}}$（单位 K/W）表示每传导 1 W 热量所需的温差
- 串联的各层热阻相加
- 芯片相对冷却介质的温升约为

$$\Delta T\approx P\cdot R_{\text{th}}$$

- 例如一颗功耗 1000 W 的芯片，若从芯片结点到冷却液的总热阻为 0.05 K/W
  - 则结点温度比冷却液高约 50 K
  - 在冷却液温度为 35 °C 时，结点温度约 85 °C，接近多数芯片允许的上限
- 功耗继续增加时，有三种选择
  - 降低总热阻（更好的界面材料、更高效的冷板）
  - 降低冷却液温度
  - 限制功耗

热点（hotspot）：

- 上述估算假设热量在芯片表面均匀分布
- 实际芯片中，矩阵单元等高活跃度区域的功率密度可能是平均值的数倍
  - 形成局部高温区域，称为热点
- 热点温度比平均温度高出很多
  - 芯片的降频保护和可靠性寿命都由最高温度决定
  - 因此热点比平均温度更危险

3D 堆叠使散热更加困难：

- 在平面芯片中，热量只需穿过一层硅即可到达散热器
- 在 3D 堆叠中，位于下层的 die 产生的热量必须穿过上层 die 和层间的键合层
  - 垂直方向的热阻增加，下层 die 的温度更高
- HBM 堆栈同样存在这一问题，且 DRAM 对温度敏感
  - 温度升高时 DRAM 存储单元的电荷泄漏加快，需要提高刷新频率
  - 按 JEDEC 规范，温度超过 85 °C 后刷新频率需要加倍
  - 这会占用带宽并增加功耗
- HBM 堆栈紧邻高功耗的计算 die 放置，还会受到计算 die 传来的热量影响

#### 7.7.7 热机械应力：热膨胀系数失配

材料受热时会膨胀，长度变化量为

$$\Delta L=\alpha L_0\Delta T$$

- 其中 $\alpha$ 为线膨胀系数（CTE，Coefficient of Thermal Expansion，单位 1/K 或 ppm/K）
- $L_0$ 为原始长度
- $\Delta T$ 为温度变化

封装由多种 CTE 不同的材料组成：

- 硅的 CTE 约为 3 ppm/K，有机封装基板约为 17 ppm/K，焊料和塑封材料又各不相同
- 以两段原长均为 10 mm 的材料为例，温度升高 100 K 时
  - CTE 为 $3\times10^{-6}$/K 的材料伸长 $3\times10^{-6}\times10\ \text{mm}\times100=3\ \mu$m
  - CTE 为 $17\times10^{-6}$/K 的材料伸长 17 µm
  - 二者相差 14 µm
- 如果两种材料可以自由伸缩，这一差异不会产生问题
- 但在封装中，它们通过焊点、键合层或粘接剂固定在一起，无法自由伸缩
  - 长度差转化为界面上的应力

热机械应力会导致以下几类失效：

- 翘曲（warpage）：整个封装或其中的某一层发生弯曲。翘曲使焊点或凸点的一部分受拉、一部分受压，严重时在组装过程中就无法形成可靠连接。
- 分层（delamination）：不同材料之间的界面在剪切应力作用下分离，使电气连接断开，或在散热路径中形成隔热的空隙。
- 开裂（crack）：脆性材料（如硅、介质层）或焊点在应力集中处产生裂纹。
- 疲劳（fatigue）：芯片在工作中反复经历负载变化引起的温度循环，每次循环都使焊点承受一次应力，累积到一定次数后焊点开裂失效。

应力的来源与封装尺寸的影响：

- 应力不仅来自工作时的温度变化，也来自制造过程
  - 键合、回流焊和固化都在高温下进行
  - 冷却到室温后，各层材料因收缩量不同而残留应力，称为残余应力（residual stress）
  - 即使芯片工作温度正常，残余应力也可能使其从制造完成时起就带有潜在的失效风险
- 封装尺寸越大，热机械问题越严重
  - 因为长度差与 $L_0$ 成正比
  - 为容纳更多 HBM 堆栈和更大的计算 die，CoWoS 中介层的面积已达到光罩极限的数倍
  - 翘曲控制成为大尺寸封装的主要工艺难点之一

#### 7.7.8 电、热、力的多物理场耦合

供电、散热和热机械应力并不是三个独立的问题，而是相互影响的耦合系统：

- 电流通过导体产生焦耳热，使温度升高
- 温度升高使金属电阻增大
  - 在相同电流下焦耳热和 IR 压降进一步增加
- 温度变化通过热膨胀产生应力和形变
- 形变改变了接触界面的状态
  - 界面分层或焊点开裂又使局部电阻和热阻升高，引起更多发热
- 这种正反馈可能使局部缺陷逐步扩大，最终导致失效

因此封装设计需要进行多物理场分析，常用方法是有限元分析（FEA，Finite Element Analysis）：

- 将封装的复杂几何结构划分为大量小单元
- 在每个单元上联立求解电场、温度场和应力场方程
  - 得到整个结构中的电流分布、温度分布和应力分布
- 仿真结果高度依赖输入的材料参数（这些参数本身随温度变化）、边界条件、网格划分和制造工艺历史
  - 因此必须用实测数据校准

对 AI 硬件而言，多物理场约束最终体现为系统层面的上限：

- 单芯片功耗受供电和散热能力限制
- HBM 堆栈的数量和层数受封装面积、翘曲和散热限制
- die 之间的连接密度受键合工艺和热机械可靠性限制
- 算力、带宽和容量的每一次提升，都需要同时满足这些物理约束

### 7.8 算例：AllReduce 的链路差异与万卡集群故障间隔

#### 7.8.1 算例一：AllReduce 在不同链路上的耗时

对 2 GB 的梯度消息（1B 参数，BF16）执行 AllReduce，采用 ring 算法：

- 带宽项为 $\frac{2(p-1)}{p}\frac{M}{\beta}$
  - $p$ 为参与的 GPU 数
  - $M$ 为消息大小
  - $\beta$ 为每块 GPU 的链路带宽
- 系数 $\frac{2(p-1)}{p}$ 来自 ring AllReduce 的两个阶段
  - reduce-scatter 和 all-gather 各需每块 GPU 发送 $\frac{p-1}{p}M$ 的数据

| 场景 | $p$ | $\beta$ | 时间 |
|---|---|---|---|
| 节点内 NVLink（H100） | 8 | 450 GB/s（实测 busbw 约 360） | $1.75\times2/360\approx 9.7$ ms |
| 跨节点 IB，每块 GPU 配一张 400G 网卡 | 64 | 50 GB/s（实测约 45） | $1.97\times2/45\approx 88$ ms |
| 跨节点，仅经 PCIe 与单张网卡（无 GPUDirect，8 块 GPU 共享一张网卡） | 64 | ~6 GB/s | ~650 ms |

第三种场景的原因：

- 8 块 GPU 共享一张 400G 网卡，每块 GPU 分到的带宽只有约 6 GB/s
- 由于没有 GPUDirect，数据还要经主机内存中转
- 这种配置常见于为通用计算设计、未针对分布式训练优化的服务器

三种场景的对比与验证方法：

- 同一个操作，在正确与错误的硬件配置下耗时可相差两个数量级
- 因此新集群上线时，首先应运行 `nccl-tests` 确认 busbw

```bash
# 节点内 8 卡
./build/all_reduce_perf -b 8M -e 4G -f 2 -g 8
# 跨节点（MPI 启动），每进程一卡
mpirun -np 64 -N 8 --hostfile hosts ./build/all_reduce_perf -b 8M -e 4G -f 2 -g 1
```

参数与输出的读法：

- 参数 `-b` 和 `-e` 分别指定测试消息大小的起点和终点
- `-f 2` 表示每次将消息大小翻倍
- `-g` 指定每个进程使用的 GPU 数
- 输出中应重点查看大消息（数百 MB 以上）时的 busbw
  - 节点内 H100 应接近 360 GB/s
  - 跨节点 400G IB 应接近 45 GB/s
- 明显低于这些数值时，通常是网卡选择、PCIe 拓扑或网络配置有问题

#### 7.8.2 算例二：万卡集群的故障间隔

MTBF（Mean Time Between Failures，平均故障间隔时间）指相邻两次故障之间的平均运行时间：

- $n$ 个独立部件同时运行时，任何一个部件故障都会中断作业
- 集群层面的 MTBF 等于单件 MTBF 除以 $n$
- 下表按各部件的假设 MTBF 进行估算：

| 部件 | 单件 MTBF（假设） | 10,000 卡集群中的数量 | 集群层面 MTBF |
|---|---|---|---|
| GPU | 50,000 h | 10,000 | 5 h |
| HBM（属于 GPU，单独统计） | 80,000 h | 10,000 | 8 h |
| 光模块 / 线缆 | 200,000 h | ~30,000 | 6.7 h |
| 主机（CPU、内存、电源） | 100,000 h | 1,250 | 80 h |

- 单件 MTBF 为 50,000 小时意味着一块 GPU 平均约 5.7 年才发生一次故障
  - 单独看非常可靠
  - 但一万块 GPU 同时运行时，平均每 5 小时就有一块发生故障

将各部件的故障率相加，得到集群的合并故障率：

$$\lambda = \sum_i \frac{n_i}{M_i} = \frac{1}{5}+\frac{1}{8}+\frac{1}{6.7}+\frac{1}{80}\approx 0.49\ \text{/h}$$

- 作业 MTBF 为 $1/\lambda$，约为 2 小时
  - 与 Llama 3 的实测量级一致（16K 卡约 3 小时）
- 这一估算假设各部件的故障相互独立
  - 实际中，供电中断或交换机故障会同时影响大量设备
  - 独立假设往往低估系统级风险

代入 4.2.3 节的公式：

- 异步 checkpoint 的保存开销 $\delta=10$ s 时
  - 最优保存间隔 $\tau^*=\sqrt{2\times(10/3600)\times2}\approx 0.105$ h ≈ 6.3 min
  - 浪费比例为 $\sqrt{2\delta/M}\approx 5.3\%$
- 再加上每次重启耗时 10 分钟（$10/120\approx 8\%$），有效训练时间约为 87%
- 若将重启时间压缩到 3 分钟（$3/120=2.5\%$），有效训练时间可恢复到 90% 以上

结论与措施：

- 因此在万卡训练中，重启速度与 checkpoint 开销同等重要
- 缩短重启时间的措施包括
  - 预留热备节点以免等待硬件更换
  - 预先拉取容器镜像
  - 从 CPU 内存或本地 NVMe 而非远端存储恢复 checkpoint
  - 缩短 NCCL 通信初始化时间

### 7.9 小结

- 选型时关注三个比值：脊点、容量 / 算力、NVLink 带宽 / 算力；decode 负载主要取决于显存带宽和容量。
- GPU 以 SIMT 和硬件动态调度换取通用性，TPU、昇腾等 DSA 以软件管理的片上存储和编译器静态调度换取能效
  - 评价硬件应以真实负载下的有效吞吐、能耗和成本为准
  - 选型前应完成负载画像
- HBM 依靠宽接口获得高带宽，必须通过 2.5D 封装与计算 die 并排放在硅中介层上；TSV、微凸点和混合键合在连接密度与制造难度之间各有取舍。
- scale-up 网络（NVLink）与 scale-out 网络（IB/RoCE）的带宽相差约一个数量级，TP / EP 应限制在 NVLink 域内。
- GPU 与网卡、CPU 之间的亲和性，以及 rail-optimized 拓扑，直接影响实测带宽。
- 训练调度需要 gang 调度与拓扑感知，所有作业都应支持被抢占后从 checkpoint 恢复。
- Dennard 缩放失效后，功率密度成为芯片设计的首要约束
  - 供电网络的焦耳热与 IR 压降、散热路径的热阻和封装材料的热膨胀失配共同限制了单芯片功耗与封装规模
- 万卡集群的 MTBF 约为 2–3 小时，checkpoint 开销与重启时间共同决定有效训练时间。

<br>

## 8. 前沿方向：MoE、超节点、推理时计算与 RL 系统

本章讨论 2024 年以来演进最快的四个方向：
- MoE 成为大模型的主流结构
- 超节点扩大了高带宽互连的范围，改变了并行策略的边界
- 推理模型（reasoning model）使服务成本集中到 decode 阶段
- RL 后训练成为训练负载与推理负载交替运行的系统问题

### 8.1 MoE 系统：专家并行与 DeepSeek 实践

#### 8.1.1 MoE 的容量与计算解耦

- MoE（Mixture of Experts，混合专家）层把 Transformer 中的稠密 FFN 替换为多个结构相同、参数独立的 FFN
  - 每个 FFN 称为一个专家（expert）
- 路由器（router，也称门控网络 gating network）是一个小型线性层
  - 根据 token 的隐藏状态为每个专家打分
  - 只把该 token 送往得分最高的 $k$ 个专家（top-$k$ 路由）
  - 以打分作为权重合并这些专家的输出

由此，两类参数量分别决定不同的量：
- 模型的总参数量决定容量
- 每个 token 实际经过的激活参数量决定计算量
  - DeepSeek-V3 总参数 671B，每 token 激活 37B
  - 训练 FLOPs 按 $6\times37\text{B}$ 估算（0.3.1 节的 $6N$ 规则），而不是 $6\times671\text{B}$

计算量的节省以系统开销为代价：
- 显存必须容纳全部专家的参数
- 每个 token 必须被发送到其所选专家所在的 GPU 上
  - 在专家并行（EP，expert parallelism，把不同专家放置到不同 GPU 上的并行方式）下，这一步由 All-to-All 集合通信完成
  - All-to-All 即每块 GPU 向其他每块 GPU 发送一份不同的数据

#### 8.1.2 DeepSeek-V3 的结构要点

下表中有两个术语需要先说明：
- MLA（Multi-head Latent Attention，多头潜在注意力）
  - 把每个 token 的 K、V 压缩为一个低维潜向量（latent）进行缓存
  - 计算注意力时再经上投影矩阵恢复
  - KV cache（推理时缓存的历史 token 的 K、V 张量）因此大幅缩小
- 无辅助损失的负载均衡（auxiliary-loss-free balancing）：为每个专家维护一个偏置项
  - 该偏置只加在用于选择专家的打分上，不参与输出加权
  - 专家负载偏高时调低其偏置，偏低时调高
  - 因此无需在训练损失中加入可能损害模型质量的均衡损失项

| 项 | 值 |
|---|---|
| 层数 | 61（前 3 层稠密，其余 MoE） |
| 隐藏维 | 7168 |
| 每层专家 | 1 个共享专家 + 256 个路由专家，每 token 选 8 个 |
| 专家中间维 | 2048 |
| 注意力 | MLA，KV latent 512 维 + RoPE 64 维 |
| 路由约束 | 每个 token 最多发往 4 个节点 |
| 负载均衡 | 无辅助损失的偏置调节 + 很小的序列级辅助损失 |
| 训练 | 2048 块 H800，FP8 混合精度，约 278.8 万 H800 GPU·h |

#### 8.1.3 训练侧的系统设计

表中涉及的几项技术定义如下：
- 节点受限路由（node-limited routing）：限制每个 token 的目标专家最多分布在 4 个节点上
  - 以此控制跨节点流量
- DualPipe：DeepSeek 提出的双向流水并行调度
  - 从流水线两端同时注入 micro-batch
  - 使一个 micro-batch 的前向计算与另一个 micro-batch 的反向计算在同一时段执行
  - 从而把 All-to-All 通信隐藏在计算之后
- SM（Streaming Multiprocessor）：GPU 上执行计算的基本单元
  - 通信 kernel 占用的 SM 越多，留给计算的 SM 越少
- warp specialization：让同一 kernel 内不同的 warp 承担不同类型的任务
- MTP（multi-token prediction，多 token 预测）：在主模型之外附加的预测模块
  - 每个位置额外预测后续若干个 token
  - 训练时提供额外信号
  - 推理时可用作投机解码的草稿
- EMA：模型参数的指数滑动平均

| 问题 | DeepSeek-V3 的做法 |
|---|---|
| 跨节点 All-to-All 带宽低 | 节点受限路由：先按节点打分选出 4 个节点，再在节点内选专家；数据经 IB 发送到目标节点上编号相同的 GPU，再经 NVLink 转发到目标 GPU |
| 通信占比高（计算与通信耗时约 1 : 1） | DualPipe 双向流水：把一个 chunk 拆分为 attention、dispatch、MLP、combine 四段，前向块和反向块交错执行，使计算与通信完全重叠 |
| 通信占用 SM | 定制 All-to-All kernel 只使用 20 个 SM，借助 warp specialization 分别处理 IB 发送、IB 到 NVLink 的转发和 NVLink 接收 |
| 显存 | 不使用 TP；重算 RMSNorm 和 MLA 上投影；EMA 参数放在 CPU 内存；MTP 模块与主模型共享 embedding 和输出头 |
| 精度 | FP8 GEMM，采用细粒度 tile/block 量化，并提高累加精度（1.6.4 节） |

#### 8.1.4 推理侧的系统设计

推理采用 prefill/decode 分离部署，相关术语如下：
- prefill：一次性处理输入 prompt 并生成首个 token 的阶段
  - 属于算力受限
- decode：逐个生成后续 token 的阶段
  - 属于访存受限
- SP（sequence parallelism）：沿序列维度切分 attention 之外的计算
- EPLB（Expert Parallelism Load Balancer）：根据专家负载统计计算冗余专家放置方案的工具

| 阶段 | 部署 | 要点 |
|---|---|---|
| Prefill | 以 4 节点 32 块 GPU 为一个部署单元；attention 采用 TP4 + SP + DP8，MoE 采用 EP32 | 两个 micro-batch 交错执行，一个计算时另一个通信 |
| Decode | 以 40 节点 320 块 GPU 为一个部署单元；attention 采用 TP4 + SP + DP80，MoE 采用 EP320 | 每块 GPU 放置一个路由专家，另有 64 块 GPU 放置冗余专家和共享专家；All-to-All 使用纯 RDMA 的低延迟 kernel |
| 负载均衡 | 冗余专家（EPLB） | 根据负载统计把热门专家复制多份，并定期调整放置 |

DeepSeek 随后公开的推理系统统计数据显示：
- 每个 H800 节点 prefill 阶段平均输入吞吐约 7.37 万 token/s（含前缀缓存命中）
- decode 阶段平均输出吞吐约 1.48 万 token/s

#### 8.1.5 专家并行的开源实现

MoE 层的通信与相关概念如下：
- MoE 层的通信分为两步
  - dispatch：把每个 token 发送到其所选专家所在的 GPU
  - combine：把各专家的输出送回 token 原来所在的 GPU，并按路由权重求和
- grouped GEMM：把多个尺寸不同的矩阵乘合并为一次 kernel 调用
  - 适合每个专家各自处理不定数量 token 的情形
- token dropping：专家接收的 token 超过容量上限时丢弃多余部分
- dropless：不丢弃任何 token

| 工具 | 作用 |
|---|---|
| DeepEP | DeepSeek 开源的 MoE dispatch / combine 通信库，提供高吞吐模式（NVLink + RDMA 转发）和低延迟模式（纯 RDMA，以 hook 方式与计算重叠，不占用 SM） |
| DeepGEMM | FP8 grouped GEMM，支持 MoE 的连续布局和 masked 布局 |
| EPLB | 专家负载均衡器，根据负载统计计算冗余专家的放置方案 |
| Megatron-Core MoE | 提供 grouped GEMM、token dropping / dropless 两种模式，以及 EP + TP + PP 的组合并行 |

DeepGEMM 的两种布局分别对应 prefill 和 decode：
- 连续布局：把发往同一专家的 token 紧密排列
  - 适合 token 数多且可变的 prefill
- masked 布局：为每个专家预留固定大小的缓冲区，并用掩码标记有效 token
  - 形状固定，便于在 decode 中配合 CUDA Graph 使用

### 8.2 超节点与 scale-up 互连

#### 8.2.1 超节点的定义与典型系统

GPU 集群的互连分为两个层次：
- scale-up 域（scale-up domain）：GPU 之间通过 NVLink 一类高带宽、低延迟并支持内存语义的互连直接相连的范围
  - 内存语义即 load/store 直接访问对方显存
- scale-out：跨越 scale-up 域、经 InfiniBand 或 RoCE 等数据中心网络的扩展
- 两者的带宽相差约一个数量级
  - H100 的 NVLink 每块 GPU 双向 900 GB/s
  - 每块 GPU 配一块 400 Gb/s 网卡时，scale-out 带宽单向只有约 50 GB/s

超节点把 scale-up 域扩展到整机柜乃至多个机柜：
- 传统服务器的 scale-up 域仅限一台机器内的 8 块 GPU，跨机器的通信都必须走网络
- 超节点（superpod，也称 rack-scale system）使原本的机间通信变为域内通信
- 以 GB200 NVL72 为例
  - 一个机柜内 72 块 Blackwell GPU 通过 NVLink Switch 全互连
  - 每块 GPU 的 NVLink 双向带宽为 1.8 TB/s
  - 域内聚合带宽约 130 TB/s
  - 任意两块 GPU 之间可以直接读写对方的显存

| 系统 | scale-up 域大小 |
|---|---|
| DGX H100 | 8 |
| GB200 / GB300 NVL72 | 72 |
| Vera Rubin NVL72 / Rubin Ultra NVL576 | 72 / 576（规划） |
| 华为 CloudMatrix 384 | 384 颗昇腾 910C |
| Google TPU v4 / v5p pod | 4096 / 8960 颗芯片（ICI + 光交换） |

TPU pod 的组织方式与 NVLink 不同：
- 芯片之间通过 ICI（Inter-Chip Interconnect）以环面（torus）拓扑与相邻芯片直连
- 再由光路交换机（OCS）动态重组拓扑
- 因此其 scale-up 域规模可达数千颗芯片
- 但任意两颗芯片之间的通信可能需要多跳

#### 8.2.2 对并行策略的影响

- 第 3 章的并行策略设计遵循一条基本原则
  - 通信量大、对延迟敏感的并行维度（TP、EP、CP）放在 scale-up 域内
  - 通信量小的维度（PP、DP）跨越 scale-out 网络
- scale-up 域从 8 块 GPU 扩大到 72 块以后，这条原则的约束随之放宽
  - 各并行维度的取值范围变化如下表

| 并行 | 8 卡 NVLink 域 | 72 卡 NVLink 域 |
|---|---|---|
| TP | ≤ 8 | 可扩展到 16–32（通信仍是瓶颈，但不再受服务器边界限制） |
| EP | 跨节点 All-to-All 经 IB 传输 | 64–72 个专家分组可以全部放在 NVLink 域内 |
| CP | 长序列需跨节点切分 | 在域内完成 |
| PP | 跨节点扩展的主要手段 | 需求下降 |
| 推理 | 大 MoE 需要多节点 + RDMA | 一个 NVL72 可容纳 DeepSeek-V3 的全部专家，decode 阶段的 All-to-All 完全在 NVLink 上完成 |

以推理为例：
- 8.1.4 节中 DeepSeek-V3 的 decode 单元横跨 40 个节点
  - 每一步 All-to-All 都经过 RDMA 网络
  - 单次通信延迟在数十微秒量级
- 在 NVL72 上，同样的通信走 NVLink
  - 带宽提高一个数量级
  - 延迟降到微秒量级
  - TPOT（time per output token，每个输出 token 的生成时间）随之下降

#### 8.2.3 故障域、功耗密度与成本

超节点在扩大高带宽域的同时，引入了三类工程代价。

故障域：
- 爆炸半径（blast radius）：单个部件故障所波及的范围
- 在 8 卡服务器上，一块 GPU 或一个 NVLink 链路故障只影响这台服务器
- 在 NVL72 中，72 块 GPU 共享同一组 NVLink Switch 托盘和铜缆背板
  - 一个交换芯片或一组线缆的故障可能使整个域的 NVLink 拓扑不完整
- 同时，TP、EP 等并行维度已经铺满整个域
  - 任何一块 GPU 退出都会使占用该域的作业中断
- 因此运维上通常采用域内降级（in-domain degradation）策略
  - 作业只使用 72 块中的 64 或 68 块 GPU，其余作为热备
  - 某块 GPU 故障时，调度器用热备 GPU 替换故障 GPU 并从 checkpoint 恢复
  - 而不必把整个机柜下线
- 代价是 5%–10% 的 GPU 平时处于闲置状态

功耗密度：
- GB200 NVL72 单柜功耗在 120 kW 以上，而风冷机柜的散热能力通常只有约 20–40 kW
  - 风冷依靠空气带走热量，空气的比热容和密度都很低
  - 要带走 120 kW 热量所需的风量和风速超出机柜与机房风道的物理上限
- 因此超节点必须采用液冷
  - 冷却液通过冷板直接贴附在 GPU 和 CPU 上
  - 液体单位体积的热容约为空气的数千倍
- 液冷还要求机房配套冷却分配单元（CDU）、二次侧管路和更高的单柜供电能力
  - 老旧机房往往无法直接部署

成本：
- 超节点的额外成本来自三部分
  - NVLink Switch 芯片及其托盘
  - 连接 72 块 GPU 的铜缆背板
    - 铜缆在短距离内功耗低于光模块
    - 但布线密度极高，制造和维护复杂
  - 液冷系统和机房改造
- 单位 GPU 的系统成本因此高于同代 8 卡服务器
- 只有当工作负载确实受益于更大的 scale-up 域（如大规模 MoE 推理、超长序列训练）时，这部分投入才能通过更高的吞吐收回

### 8.3 推理时计算：推理模型的服务负载与成本

- 推理时计算（test-time compute）：在推理阶段投入更多计算以提升答案质量
  - 具体形式包括生成更长的思维链
  - 对同一问题多次采样后投票
  - 在多条推理路径上做搜索
- 推理模型（如 OpenAI o 系列、DeepSeek-R1）通过 RL 训练学会在作答前生成较长的思考过程
  - 这使推理时计算成为常态

#### 8.3.1 负载特征的变化

下表比较普通对话与推理模型的负载特征，其中的术语如下：
- TTFT（time to first token）：从请求到达至首个 token 输出的时间
  - 主要由 prefill 决定
- TPOT：由 decode 决定
- best-of-n：采样 $n$ 个回答后由打分器选出最优者
- 多数投票：选择 $n$ 个回答中出现次数最多的最终答案

| | 普通对话 | 推理模型 |
|---|---|---|
| 输出长度 | 几百 token | 几千到几万 token（思考过程） |
| 成本重心 | prefill 和 decode 相当 | decode 占绝大部分 |
| KV cache | 主要来自输入 | 随生成不断增长 |
| 延迟体验 | TTFT 和 TPOT | 总思考时长，等于 TPOT × 输出长度，TPOT 直接决定用户的等待时间 |
| 并行采样 | 少 | best-of-n、多数投票、树搜索较为常见 |

成本重心转移的原因在于 decode 的访存特性：
- decode 每生成一个 token，都要从 HBM（GPU 上的高带宽显存）读取全部权重和该请求的全部 KV cache
  - 其算术强度远低于 roofline 模型中的脊点（峰值算力与显存带宽之比，1.3.3 节）
  - 属于带宽受限
- 输出长度从几百增加到几万 token，decode 步数成比例增加
- KV cache 随生成持续增长，每一步需要读取的字节数也在增加
  - 因此单 token 成本随输出变长而上升

#### 8.3.2 系统层面的优化手段

| 手段 | 作用 |
|---|---|
| 降低 TPOT | 投机解码（MTP、EAGLE）、更小的 TP 组配合更大的 batch、FP8/FP4 权重 |
| KV 压缩 | MLA、FP8 KV、稀疏注意力；长输出下 KV 读取是 decode 的主要开销 |
| 并行采样共享前缀 | $n$ 个样本共享同一份 prompt KV（5.3.3 节的 copy-on-write） |
| 早停 | 多个样本的答案已经一致时，提前终止其余样本 |
| 动态思考预算 | 按问题难度限制思考 token 数 |

- 投机解码（speculative decoding）：用一个轻量的草稿模型或 MTP 模块一次提出多个候选 token
  - 再由主模型在一次前向中并行验证
  - 从而用一次权重读取产出多个 token（5.6 节）
- 早停的依据是：在多数投票场景下，若 8 个样本中已有 5 个得出相同答案，其余样本无论结果如何都不会改变投票结论
  - 可以立即释放其 KV cache 和计算槽位

#### 8.3.3 decode 阶段的长尾效应

推理模型的输出长度分布呈明显的长尾：
- 一个 batch 中，大部分请求在几千 token 内结束，少数请求需要几万 token
- continuous batching（迭代级调度，某个请求结束后立即由新请求填补其位置，5.2 节）可以填补空出的槽位
  - 使 GPU 保持较高利用率
- 但单个长请求的完成时间仍由其自身的 TPOT × 输出长度决定
  - 调度无法缩短它

在 RL 训练中这一问题更为突出：
- rollout：RL 中用当前策略模型对一批 prompt 生成回答的过程（8.8 节）
- 训练步必须等这一批中所有回答都生成完毕才能开始
  - 因此最长的那几个回答决定了整步的耗时
- 例如一批 512 个回答中，99% 在 8000 token 内结束，最长的 5 个需要 30000 token
  - 大部分时间里 GPU 只在为少数几个请求做 decode
  - batch 很小，利用率很低

#### 8.3.4 Agent 工作负载的状态管理与软件分层

- Agent：由模型反复调用工具（网页搜索、代码执行、外部 API）、根据结果重新规划，直至完成任务的多步工作流
- 与单轮对话相比，Agent 请求的执行过程被外部调用切分为多个片段
  - 片段之间存在不可预测的等待时间
  - 服务系统需要管理多种性质不同的状态

| 状态类型 | 内容 | 存储与生命周期 |
|---|---|---|
| KV cache | Transformer 推理中可复用的中间张量 | GPU 显存，可卸载到 CPU 内存或 SSD；随会话结束释放 |
| 工作记忆 / 任务状态 | 计划、工具调用结果、中间变量、任务进度 | 由 Agent 框架维护，通常在 CPU 侧 |
| checkpoint / snapshot | 为长时间任务的故障恢复而持久化的状态 | 持久化存储 |
| 产出物（artifact） | 最终生成的文件、报告或图像 | 独立的可靠存储 |
| 长期记忆 | 跨任务可检索的持久知识 | 数据库或向量索引；与 KV cache 是不同层次的概念 |

这些状态中，KV cache 的管理最直接影响服务成本：
- Agent 调用外部网页并等待 20 秒时，如果让该会话的全部 KV cache 继续驻留 HBM
  - 这部分显存在 20 秒内无法服务其他请求
- 如果立即卸载到 CPU 内存，而工具调用在 1 秒内就返回
  - 卸载和重新加载的搬运开销又是浪费
- Agent 的下一步动作和等待时长事先未知
  - 因此 KV cache 的卸载（offloading）、预取（prefetching）和淘汰（eviction）策略需要结合对工具调用耗时的预测、请求路由和存储层次一起设计

从软件分层看，Agent 系统由三层组成，各自负责不同的问题：

- Agent 执行框架（agent harness）：负责任务如何组织，包括工具调用循环、上下文拼装、错误处理、状态转移、权限与工作流控制；
- 模型服务运行时（model serving / runtime）：负责模型如何高效运行，包括模型加载、GPU 调度、batching、KV cache 管理和推理执行；
- 系统软件（system software）：负责数据与设备如何协同，包括内存管理、通信、设备驱动、资源调度和故障恢复。

三层之间的接口决定了优化空间：
- 例如，只有当 harness 向 serving 层传递「该会话正在等待一个预计耗时较长的工具调用」这类提示时
  - serving 层才能提前把 KV cache 卸载出 HBM

Agent 负载的另一个特点是各子任务对延迟的要求差异很大：
- 面向用户的交互回复要求低 TTFT
- 后台的长时间检索和汇总只要求总完成时间
- 因此 SLO（service level objective，服务水平目标，例如「95% 的请求 TTFT 小于 1 s」）宜按请求类别、交互阶段和任务重要性分别制定
  - 而不是对所有请求采用同一个延迟阈值

### 8.4 训推一体与弹性算力

#### 8.4.1 训推一体的三个层面

训推一体指训练与推理在资源、软件和工作流上共享同一套基础设施，具体包含以下三个层面。

| 层面 | 内容 |
|---|---|
| 资源 | 同一批 GPU 白天承担在线推理、夜间承担训练，或训练集群空闲时承接批量推理 |
| 软件 | 训练和推理共用模型定义、kernel 和数值实现，避免两套实现带来的数值不一致 |
| RL | 同一组 GPU 在 rollout（推理）和训练之间交替切换（8.8.3 节的共置模式） |

- 资源层面的动机是在线推理流量的日内波动
- 软件层面的动机是数值一致性
  - 如果训练框架和推理引擎各自实现注意力和 MoE kernel，同一组权重在两边算出的 logits 会有差异
  - 这种差异在评测时导致结果不可比
  - 在 RL 中则直接影响训练稳定性（8.8.6 节）

#### 8.4.2 弹性调度的技术难点

把同一批 GPU 在不同负载之间切换，需要解决三个问题。

模型加载与权重分发：
- 70B 模型的 BF16 权重为 140 GB，从对象存储冷加载时受限于存储带宽，通常需要数分钟
- 常用的加速方法有三种
  - 在节点本地 NVMe 上缓存常用模型的权重
  - 在节点之间以 P2P 方式分发权重，避免所有节点同时访问对象存储
  - 从已经在运行的副本经高速网络直接拷贝权重

显存布局切换：
- 训练时显存中存放参数、梯度、优化器状态和激活
- 推理时存放权重和 KV cache
- 两者的布局和大小完全不同
- 切换时有两种做法
  - 完全释放后重新分配
  - 借助 CUDA 虚拟内存管理 API 把物理显存与虚拟地址解绑，在不销毁进程的前提下换出、换入部分数据

状态保存：
- 被推理负载抢占的训练作业必须能在短时间内完成 checkpoint（4.2 节），并在 GPU 归还后从断点恢复
- 否则抢占带来的重复计算会抵消回收算力的收益

#### 8.4.3 收益估算

- 设推理集群日间峰值利用率为 80%，夜间为 20%
- 对于 1000 块 GPU 的集群，夜间 10 小时内约有 600 块 GPU 空闲
  - 每天合计 6000 卡时
  - 按每卡时 2 美元计算，相当于每天 1.2 万美元的算力
- 将这部分算力用于离线评测、数据合成或小规模训练，可以直接提高集群的有效利用率

### 8.5 开放问题

下表列出当前尚未解决或仍在快速演进的系统问题，其中的术语如下：
- 确定性（同样的输入与配置得到逐位相同的结果）训练的级别沿用 4.10 节的划分
  - 级别越高要求越严格
  - L3 指改变并行配置后结果仍逐位一致
- 静默数据错误（SDC，silent data corruption）：硬件产生了错误的计算结果却没有报告任何错误
- off-policy 程度：训练所用样本的生成策略与当前被训练策略之间的偏离程度（8.8.5 节）

| 问题 | 现状 |
|---|---|
| 万卡以上的确定性训练 | L1/L2 级已可实现；L3 级（跨并行配置逐位一致）代价高，只在推理和 RL 的训推对齐中局部使用 |
| 低精度训练的边界 | FP8 训练已成熟，FP4 训练仍在探索，其稳定性和收敛规律尚不明确 |
| 静默数据错误 | 检测依赖冗余计算和定期自检，检测代价与覆盖率之间的取舍尚无定论 |
| MoE 负载均衡 | 无辅助损失的均衡方法已被验证有效，但细粒度专家和动态专家数量下的通信与显存调度仍在演进 |
| 长上下文 | 稀疏注意力、线性注意力与全注意力的混合结构在逐步接近全注意力的质量；系统侧的 KV 分层存储正在成熟 |
| RL 系统 | rollout 长尾、训推数值一致、异步 RL 中 off-policy 程度的控制 |
| 自动并行 | 结构化的并行组合仍依赖人工设计，自动搜索只在特定场景有效 |
| 跨地域训练 | 数据中心之间带宽低、延迟高，需要低通信量的优化算法，如 DiLoCo 一类方法（各数据中心本地执行多步更新，再低频同步参数） |
| 能耗 | 电力接入能力成为集群规模的上限（8.9 节） |

### 8.6 MoE 通信与显存算例

#### 8.6.1 训练阶段：每层 All-to-All 通信量

考虑 DeepSeek-V3 结构的 MoE 层：隐藏维 $h=7168$，top-8 路由，每块 GPU 每个 micro-batch 处理 $T=4096$ 个 token。

- dispatch（FP8，每元素 1 字节）：$T\times8\times h = 4096\times8\times7168\approx 235$ MB
- combine（BF16，每元素 2 字节）：约 470 MB
- 前向每层合计约 0.7 GB；反向传播需要再传输一次同等数据量，前向与反向合计约 1.4 GB

通信时间的估算：
- 在节点受限路由下，跨节点部分约占总量的一半
  - 4 个目标节点中有 3 个是外部节点
  - 但发往同一节点的多个专家选择被合并为一次发送
- 按每块 GPU 50 GB/s 的 IB 带宽估算，前向每层 dispatch 与 combine 的 IB 传输时间约为 $0.35/50\approx 7$ ms

同一层每块 GPU 的计算量为：

$$2\times T\times(\text{37B 激活参数中本层部分})\approx 2\times4096\times(3\times7168\times2048\times9)\approx 3.2\times10^{12}\ \text{FLOPs}$$

- 公式中的 9 个专家是 8 个路由专家加 1 个共享专家，每个专家包含 3 个矩阵
- H800 的 FP8 稠密峰值约 2000 TFLOPS，按 50% 利用率计约 1000 TFLOPS
  - 对应计算时间约 3.2 ms
- 通信时间约为计算时间的两倍
  - 若串行执行，GPU 有三分之二的时间处于等待状态
  - 因此必须采用 DualPipe 一类的调度，使一个 micro-batch 的 All-to-All 与另一个 micro-batch 的计算同时进行

#### 8.6.2 推理阶段：单块 GPU 的显存分配

decode 采用 EP320 部署，每块 GPU 放置 1 个路由专家：

| 项 | 量 |
|---|---|
| 每个路由专家 | $3\times7168\times2048\approx 44$ M 参数 |
| 每块 GPU 的专家参数（58 个 MoE 层 × 1 个专家，FP8） | $58\times44\text{M}\approx 2.6$ GB |
| 非专家参数（attention、共享专家、embedding 等，每块 GPU 各存一份） | 约 15–20 GB（FP8） |
| 剩余可用于 KV cache 的显存（80 GB GPU） | 约 50 GB |
| MLA 每 token 的 KV | 约 70 KB（BF16），FP8 约 35 KB |
| 每块 GPU 可容纳的 token 数 | 约 140 万（FP8） |

作为对比：
- 稠密 70B 模型在 8 块 H100 上以 TP8 部署时
  - 每块 GPU 的权重为 17.5 GB（BF16）
  - 每 token 的 KV 为 320 KB，分摊到 8 块 GPU 后为每块 40 KB
- DeepSeek-V3 的配置下
  - 每块 GPU 上的专家权重只有 2.6 GB，绝大部分显存留给 KV cache
  - MLA 又使每 token 的 KV 只有约 35 KB（FP8）
- 两者结合，每块 GPU 能够同时服务的上下文总量远大于稠密模型
  - 这是 DeepSeek 推理成本较低的主要原因之一

#### 8.6.3 专家负载不均的代价

- MoE 层的执行时间由负载最重的那块 GPU 决定，因为 combine 必须等所有专家完成计算
- 在 320 块 GPU 中，如果最热门的专家收到的 token 数是平均值的 2 倍
  - 其余 GPU 在该层有一半时间处于等待
  - 有效吞吐只剩一半
- 冗余专家方案把最热门的若干专家复制多份放到不同 GPU 上
  - 按负载在副本之间分流 token
  - 代价是额外的显存占用（每个副本约 2.6 GB 的 FP8 参数）

### 8.7 推理模型服务的成本算例

#### 8.7.1 问题设定

- 沿用 5.13 节的 Llama-3-70B（FP8）decode 节点
- 普通对话：输入 2000 token、输出 500 token
- 推理模式：输入 1000 token、输出 10,000 token
- 比较两种场景下单次请求的成本

#### 8.7.2 普通对话场景

- 根据 5.13.5 节的计算：batch 为 512 时每个 decode 步约 14 ms，每节点吞吐约 $3.7\times10^4$ token/s
- 每个请求输出 500 token，因此每节点每秒完成约 74 个请求的 decode
- 节点按每小时 16 美元计：

$$\frac{16}{3600\times74}\approx 6\times10^{-5}\ \text{美元/请求}$$

#### 8.7.3 推理模式场景

- 生成过程中上下文从 1000 token 增长到 11000 token
  - 平均上下文约为 $1000 + 10000/2 = 6000$ token
- 按 FP8 KV 每 token 160 KB 计，batch 512 的 KV cache 总量为 $512\times6000\times160\text{ KB}\approx 490$ GB
- 每个 decode 步需要读取约 70 GB 的 FP8 权重和 490 GB 的 KV
  - 合计约 560 GB
  - 按节点聚合显存带宽计约 30 ms

| 项 | 值 |
|---|---|
| TPOT | 约 30 ms |
| 每节点吞吐 | $512/0.03\approx 1.7\times10^4$ token/s |
| 每请求输出 | 10,000 token |
| 每节点每秒完成 | 约 1.7 个请求 |
| 单请求 decode 时间 | $10000\times0.03 = 300$ s |
| 单请求成本 | $\frac{16}{3600\times1.7}\approx 2.6\times10^{-3}$ 美元 |

#### 8.7.4 结果分析

- 推理模式下单次请求的成本约为普通对话的 40 多倍
  - 约 20 倍来自输出 token 数的增加（10,000 对 500）
  - 另外约 2 倍来自平均上下文变长
    - 每个 decode 步读取的 KV 字节数增加
    - TPOT 从 14 ms 升至 30 ms，每 token 的成本相应上升
- 单个请求的 decode 时间约为 300 s，即 5 分钟

降低成本的手段按效果排序依次为：
- 缩小 KV（MLA、FP8/FP4 KV、稀疏注意力）
  - 它同时降低每步读取量和显存占用
  - 从而允许更大的 batch
- 降低 TPOT（投机解码）
- 限制思考长度（8.3.2 节的动态思考预算）

### 8.8 RL 后训练系统

#### 8.8.1 RL 迭代的执行流程

RL 后训练在预训练和 SFT 之后，用奖励信号进一步优化模型的策略，相关概念如下：
- 策略（policy）：模型在给定上下文下输出各 token 的概率分布
- PPO（Proximal Policy Optimization）
  - 需要额外训练一个 critic 模型估计状态价值
  - 用 GAE（Generalized Advantage Estimation）计算每个 token 的优势
- GRPO（Group Relative Policy Optimization）
  - 不使用 critic
  - 而是对同一 prompt 采样的一组回答的奖励做组内标准化，作为每个回答的优势
- 优势（advantage）：衡量某个回答比平均水平好多少
  - 决定梯度更新的方向与幅度
- logprob：模型对某个 token 赋予的对数概率

以 GRPO / PPO 为例，一次迭代包含六个步骤：

```text
1. Rollout：   用当前策略对一批 prompt 生成回答（每个 prompt 采 n 个）   ← 推理负载
2. 打分：     奖励模型 / 规则验证器 / 代码执行沙箱给每个回答打分
3. 计算 logprob：训练框架重新算当前策略（和参考模型）对这些 token 的 logprob
4. 计算优势：  GRPO 用组内标准化，PPO 用 critic + GAE
5. 训练：     对策略做若干步梯度更新                                       ← 训练负载
6. 权重同步：  把新权重推给 rollout 引擎
```

#### 8.8.2 迭代耗时分布

- 在长思维链任务中，rollout 通常占一次迭代 60%–80% 的时间
  - rollout 是 decode 密集的推理负载
  - 且受 8.3.3 节所述长尾效应的影响：一批回答中最长的几个决定了这一阶段的结束时间
- 打分阶段的耗时取决于奖励来源（8.8.7 节）
  - 规则验证几乎不耗时
  - 代码执行和奖励模型则可能占到 10% 以上
- logprob 重算与训练阶段的计算特征与普通 SFT 类似
  - 可以达到较高的 MFU（模型实际完成的有效 FLOPs 与硬件峰值 FLOPs 之比，4.4 节）

#### 8.8.3 共置与分离两种部署模式

RL 系统同时包含推理负载（rollout）和训练负载，两者在 GPU 上的部署有两种基本方式：
- 共置（colocate）：rollout 和训练使用同一组 GPU，分时交替执行
- 分离（disaggregate）：rollout 和训练各自使用独立的一组 GPU
  - 两组之间通过网络传输数据和权重

| | 共置（colocate） | 分离（disaggregate） |
|---|---|---|
| 做法 | 同一组 GPU 先执行 rollout，再切换为训练 | rollout 和训练各用一组 GPU |
| 优点 | 没有 GPU 闲置；权重同步在本机显存或 NVLink 内完成 | 两组可以同时工作，各自采用最适合的并行配置 |
| 缺点 | 两个阶段串行执行；切换时需要重新安排显存（卸载 KV cache、加载优化器状态） | 一组繁忙时另一组可能空闲；权重同步需经过网络 |
| 代表实现 | verl（HybridFlow）默认采用 | OpenRLHF、slime、AReaL 等支持此模式 |

两种模式的取舍取决于 rollout 与训练的耗时比例：
- 若 rollout 占 70% 的时间，分离部署时训练组有大量空闲
  - 除非引入 8.8.5 节的异步执行
- 共置部署则避免了这种闲置
  - 但每次切换都要在推理显存布局（权重 + KV cache）和训练显存布局（参数 + 梯度 + 优化器状态 + 激活）之间转换
  - 切换开销通常在数秒到数十秒

#### 8.8.4 权重同步

- 权重同步（weight sync）：每次训练更新后，把新的策略权重从训练框架传送到 rollout 推理引擎
  - 使下一轮 rollout 使用最新策略
- 以 70B 模型为例，BF16 权重共 140 GB：

| 方式 | 时间量级 |
|---|---|
| 共置：训练侧的 TP/FSDP 分片在本机重排为推理侧的 TP 分片 | 秒级 |
| 分离：通过 NCCL broadcast 传输，每节点 8 块 400 Gb/s 网卡（合计约 400 GB/s） | 理论 0.35 s，实际数秒 |
| 经过存储：训练侧写出 checkpoint，推理侧重新加载 | 分钟级，只适合低频更新 |

- 训练侧和推理侧的并行切分方式通常不同
  - 例如训练采用 FSDP 把每个参数矩阵按行均分到 64 块 GPU，推理采用 TP4 按注意力头切分
- 因此同步时必须做重新分片（resharding）
  - 按推理侧的 TP 切分方式，从训练侧的各个分片中取出并拼接每块 GPU 所需的部分
- 分离模式下实际耗时远高于理论值，原因主要有三点
  - resharding 带来的额外拷贝
  - 多次小规模通信的启动开销
  - 推理引擎接收新权重时需要暂停服务

#### 8.8.5 异步 RL

- 同步 RL 中，rollout 和训练互相等待
  - 训练要等 rollout 生成完整批数据
  - rollout 要等训练更新完权重
- 异步 RL 允许 rollout 使用落后 $k$ 步的旧权重生成数据
  - 训练同时消费这些数据，从而消除互相等待
- $k$ 称为陈旧度（staleness）
  - 即生成样本所用的策略版本落后于当前训练策略的步数

异步 RL 的收益与代价：
- 收益是 rollout 组和训练组都可以持续满负荷运行
- 代价是数据变为 off-policy
  - 样本由旧策略生成，而梯度要对当前策略求期望，两者分布不同
- 修正方法是重要性采样（importance sampling）
  - 用当前策略与采样策略对同一动作的概率比 $\pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)$ 为每个样本加权
  - 使加权后的期望等于在当前策略下的期望
- $k$ 越大，两个策略的差异越大
  - 概率比的方差越大，修正后的梯度估计越不稳定
  - 因此 $k$ 通常限制在 1–4 步

partial rollout 是针对长尾的配套手段：
- 长回答不必在一次迭代内生成完毕
  - 到达本轮截止时间时先暂停，把已生成的部分保存下来
  - 在下一轮迭代中用新权重继续生成
- 这样每轮迭代的 rollout 时间不再由最长的回答决定
- 代价是同一个回答的不同片段由不同版本的策略生成

#### 8.8.6 训推数值不一致及其修正

- 如 5.15.5 节所述，rollout 引擎和训练框架即使加载相同的权重，对同一 token 算出的概率也会不同
  - 原因包括 kernel 实现不同
  - 浮点累加顺序不同
  - batch 大小不同导致的数值差异
  - 推理侧的量化
- 记 $\pi_{\text{inf}}$ 为 rollout 侧的概率，$\pi_\theta$ 为训练侧的概率
  - 策略梯度应当按 $\pi_\theta$ 计算期望，但样本实际来自 $\pi_{\text{inf}}$
  - 因此即便权重完全同步，数据也是轻微 off-policy 的
- 用截断重要性采样修正：

$$\nabla J \approx \mathbb{E}_{a\sim\pi_{\text{inf}}}\left[\min\left(\frac{\pi_\theta(a\mid s)}{\pi_{\text{inf}}(a\mid s)},\ C\right)\hat A\,\nabla\log\pi_\theta(a\mid s)\right]$$

- $C$ 是截断上限
  - 当某个 token 在 rollout 侧概率极低而在训练侧概率较高时，比值可能达到数十甚至上百
  - 若不截断，单个样本就会主导整批梯度，导致方差急剧增大
- 对 MoE 模型还可以采用路由回放（routing replay）
  - 在 rollout 时记录每个 token 选中的专家，训练时强制使用相同的路由结果
  - MoE 中路由是离散选择，数值上的微小差异就可能使某个 token 在两侧选中不同的专家
  - 输出随之出现大幅差异，因此路由不一致是训推差异的最大来源
  - 路由回放可以直接消除这一部分

从系统角度看：
- 根本的解决办法是让两侧数值一致
  - 即采用 5.15 节的 batch invariant kernel（输出不随 batch 大小和组成变化的 kernel）
  - 并保证跨 TP 配置的确定性
- 重要性采样等算法修正是在无法做到数值一致时的补救措施

#### 8.8.7 奖励计算的系统需求

| 奖励来源 | 系统需求 |
|---|---|
| 规则验证（如数学答案比对） | 计算开销很低，可在 CPU 上同步完成 |
| 代码执行 | 需要大规模沙箱，解决隔离、超时和并发问题；通常部署在独立的 CPU 集群上 |
| 奖励模型 | 构成额外的推理负载，需要与 rollout 统一调度 GPU 资源 |
| Agent 环境（浏览器、工具调用） | 多轮交互，单步延迟不可控，需要异步化执行 |

代码执行类奖励的规模可以很大：
- 一批 512 个 prompt、每个采样 16 个回答、每个回答运行 20 个单元测试，每次迭代就要执行约 16 万次测试
- 每次执行都要在隔离的容器中运行，并设置超时以防止死循环占用资源
- 沙箱集群的吞吐或故障率一旦成为瓶颈，GPU 就会在打分阶段空等

Agent 环境的情形与 8.3.4 节相同：
- 一次 rollout 包含多轮模型生成与工具调用
- 工具调用的耗时从毫秒到数十秒不等
  - 同步执行会使 GPU 长时间等待
  - 因此需要把等待工具返回的轨迹挂起，让 GPU 继续处理其他轨迹的生成

#### 8.8.8 RL 系统监控检查项

- [ ] 监控 rollout 与训练的时间占比，以及 rollout 输出长度的长尾分布
- [ ] 监控训练侧重算的 logprob 与 rollout 侧 logprob 的差异（如每步的 KL 散度或概率比分布）
- [ ] 每次权重同步后执行一致性检查（同一输入在两侧的输出应在阈值内一致）
- [ ] 异步程度（权重落后步数 $k$）可配置，并在日志中记录
- [ ] 监控沙箱与奖励服务的失败率和超时率

### 8.9 能耗约束与超越 CMOS 的计算范式

- 前面各节讨论的优化都在现有的 CMOS 数字电路与冯·诺依曼架构之内进行
- 本节讨论一个更长期的问题
  - 当集群规模持续扩大时，什么会成为根本性的上限
  - 有哪些计算范式在尝试突破这一上限

#### 8.9.1 能耗作为长期的规模上限

- 8.5 节已经指出，电力接入正在成为集群规模的直接约束
- 从长期看，如果数据中心的能耗需求保持指数增长，迟早会触及任何有限能源供给的边界
  - 设当前能耗为 $E_0$，年增长率为 $r$，则

$$E(t)=E_0(1+r)^t$$

- 若能耗每 5 年翻一番，则 $E(t)=E_0\cdot2^{t/5}$
  - 25 年后为当前的 32 倍
  - 50 年后约为 1000 倍
  - 100 年后约为 100 万倍
- 这个模型只用于说明指数趋势不能无限外推，不是对未来能耗的预测
  - 实际增长率会受到技术进步、电价、监管、产业结构的影响
  - 还会受到效率提升后需求反而增加（回弹效应）等因素的影响

扩大能源供给与降低单位任务能耗：
- 扩大能源供给（如建设更多核电或可再生能源设施）可以缓解这一约束，但无法消除以下问题
  - 芯片和机房的散热能力
  - 电力基础设施的建设周期
  - 能量转换过程中的损耗
  - 整体成本
- 因此长期的改进方向是把完成单位有用智能任务所需的能量（energy per useful task）降低若干个数量级
  - 而不是单纯增加供给

这一方向与过去几十年半导体发展的背景有关：
- 摩尔定律（Moore's Law）：描述集成电路上晶体管数量约每两年翻倍的历史趋势
- Dennard 缩放（Dennard scaling）：描述在理想缩放下，晶体管尺寸缩小时电压和电流按比例降低，使单位面积功耗保持不变
  - Dennard 缩放在 2000 年代中期已经失效
  - 晶体管继续缩小，但每代工艺带来的能效改善远不如从前
- 两者都是经验规律，不是自然定律
- 在制程红利减弱的情况下，能效的提升越来越依赖架构层面的创新
  - 例如第 1、2 章讨论的低精度格式、算子融合和减少数据搬运

#### 8.9.2 人脑作为能效的存在性证明

- 人脑的功耗约为 20 W，却能完成复杂的感知、语言、运动控制和持续学习
  - 这说明以极低能耗实现通用智能在物理上是可能的
  - 可以视为一个存在性证明（existence proof）

不能据此把人脑折算为若干 FLOPS 再与 GPU 比较，原因是两者存在多方面差异：
- 两者完成的任务不同
- 信息的表示方式不同
  - 生物神经元以离散脉冲和化学信号传递信息
  - GPU 以浮点数矩阵运算
- 学习过程不同
  - 人脑在发育过程中持续学习
  - 模型在训练阶段一次性学习
- 能量的统计边界也不同
- 这类折算得出的数字没有可比的物理意义

有意义的问题：
- 生物系统中是否存在现有 AI 尚未利用的高效学习机制和架构原则
  - 例如局部计算、模块化和层次化结构

#### 8.9.3 神经形态计算与脉冲神经网络

- 神经形态计算（neuromorphic computing）：模仿生物神经系统的结构或机制来设计计算硬件
- 脉冲神经网络（SNN，spiking neural network）是其中最常见的模型
  - 神经元以离散的脉冲（spike）在时间上传递信息
  - 只有当膜电位超过阈值时才发放脉冲
  - 没有脉冲时不产生计算和通信

SNN 的现状：
- 这种事件驱动的特性在输入稀疏的场景下可以减少无效计算
  - 但事件驱动本身并不自动带来大脑级的能效
- SNN 经过长期研究，至今在通用任务上没有形成相对于常规深度学习的明确优势，主要障碍包括
  - 脉冲的离散性使梯度难以直接计算，训练方法不成熟
  - 现有的大规模模型和软件生态都建立在稠密矩阵运算之上

另一种研究思路：
- 不局限于模仿脉冲的形式，而是研究生物神经系统更深层的组织原则
  - 如模块化、层次化、局部计算
  - 以及在能耗、速度和鲁棒性约束下进化形成的结构
- 再在电路仿真（如 SPICE）中建立等效模型
  - 验证其能耗与功能
  - 而不是停留在概念类比

#### 8.9.4 模拟计算、存内计算与事件驱动计算

除神经形态计算之外，主要的非传统方向还有以下三类。

| 方向 | 原理 | 主要问题 |
|---|---|---|
| 模拟计算（analog computing） | 用电压、电流等连续物理量直接表示数值，利用电路的物理规律（如欧姆定律和基尔霍夫电流定律）完成乘加运算 | 器件噪声和温度漂移限制了有效精度，通常只有 4–8 bit；ADC/DAC 转换的开销可能抵消计算本身的节能 |
| 存内计算（in-memory computing） | 在存储阵列内部完成计算，避免权重在存储器与计算单元之间搬运，直接针对存储墙问题 | 器件一致性差、写入寿命有限、精度受限；阵列规模固定，难以适应模型结构变化 |
| 事件驱动计算（event-driven computing） | 只在输入发生变化时触发计算，没有事件时电路处于静默状态 | 在稠密负载（如 Transformer 的矩阵乘）下优势不明显；编程模型与现有框架差异大 |

- 这些方向的共同挑战是噪声与精度、制造良率、可编程性和可靠性
- 数字电路的优势在于
  - 结果可精确复现
  - 可以通过编译器把任意模型映射到同一套硬件上
- 非传统计算要进入大规模应用
  - 必须在这些方面达到可用水平
  - 并建立相应的软件栈

#### 8.9.5 CMOS 的长期地位

- 新的计算范式不一定需要全新的材料或器件
  - 建筑行业引入新结构形式之后，木材、砖和混凝土仍在使用
  - 同样，成熟的 CMOS 工艺在可预见的时间内仍将是计算硬件的基础
  - 新的架构（如存内计算阵列、事件驱动电路）大多也构建在 CMOS 工艺之上
- 因此近中期更现实的路径是在 CMOS 基础上结合先进封装和新架构提升能效
  - 先进封装如 2.5D/3D 集成，缩短计算与存储之间的距离
  - 而不是等待某种替代 CMOS 的器件成熟

### 8.10 小结

- MoE 降低了每个 token 的计算量
  - 系统压力相应转移到显存容量和 All-to-All 通信
  - DeepSeek-V3 通过节点受限路由、DualPipe、DeepEP、FP8 训练和大规模 EP 推理部署解决了这些问题
- 超节点把 TP、EP 等并行维度的边界从 8 块 GPU 扩展到 72 块以上
  - 代价是更大的故障域
  - 液冷带来的功耗密度要求
  - 更高的系统成本
- 推理模型使 decode 成为服务成本的主体
  - KV cache 大小和 TPOT 是降低成本的主要手段
  - Agent 负载还要求按状态类型管理 KV cache 与任务状态
- RL 后训练是推理与训练混合的系统
  - rollout 长尾、权重同步和训推数值一致是三个核心问题
- 从长期看，能耗是集群规模的根本约束
  - 神经形态、模拟和存内计算等方向仍面临精度、可编程性和可靠性问题
  - CMOS 将长期作为计算硬件的基础

<br>

## 9. 系统指标、成本模型与故障案例

前八章分别讨论了各个子系统，本章的内容是：
- 把它们放回模型从数据处理到上线服务的完整生命周期中
- 说明每个环节的关键指标、成本构成和常见故障

### 9.1 LLM 生产流水线

#### 9.1.1 全流程概览

```text
数据         预训练          中期训练        后训练             评测          上线            反馈
──────>  ──────────>  ──────────>  ───────────>  ──────────>  ──────────>  ──────────┐
清洗/去重   万卡 × 数月    长上下文/      SFT → RL          离线评测      推理集群        日志/偏好   │
分词/配比   (ch3, ch4)    领域数据退火    (ch4, ch8.8)      (批量推理)    (ch5, ch7)     数据回流 ──┘
```

上图的说明：
- 中期训练（mid-training）：预训练主体完成后，用长文本或特定领域的高质量数据继续训练一段时间
  - 同时逐步降低学习率（退火）
- 括号内标注了对应的章节

#### 9.1.2 各环节的基础设施需求

| 环节 | 规模 | 主要负载 | 基础设施重点 |
|---|---|---|---|
| 数据处理 | CPU 集群数千核，PB 级存储 | 去重、过滤、分词、质量打分（小模型推理） | 分布式数据处理（Spark、Ray），GPU 打分的吞吐 |
| 预训练 | 数千到数万块 GPU，数周到数月 | 训练 | MFU、稳定性、checkpoint、确定性回放（replay） |
| 中期训练 / 长上下文扩展 | 与预训练相近，周期较短 | 长序列训练 | CP（context parallelism，沿序列维度切分）、激活显存 |
| SFT | 数十到数百块 GPU | 训练 | 变长样本打包、快速迭代 |
| RL | 数百到数千块 GPU | rollout + 训练 | 8.8 节所述的全部问题 |
| 评测 | 数十到数百块 GPU | 批量推理 | 吞吐优先、前缀缓存、确定性（保证不同版本的结果可比） |
| 上线 | 数千块 GPU 起，7×24 小时运行 | 在线推理 | SLO、成本、弹性扩缩容 |

各环节的规模和负载性质差异很大：
- 预训练是长时间运行的单个超大作业
  - 关注点是 MFU 和稳定性
- 评测和数据处理是大量短作业
  - 关注点是吞吐和调度效率
- 上线服务是持续运行的在线负载
  - 关注点是延迟 SLO 和单位成本
- 同一个集群往往要同时承载这几类负载
  - 这也是 8.4 节训推一体的出发点

#### 9.1.3 环节之间的接口

| 接口 | 要求 |
|---|---|
| 训练 → 评测 | checkpoint 格式转换（训练分片 → 推理权重），并自动触发评测 |
| 评测 → 上线 | 推理引擎与训练框架数值一致（同一输入的 logits 误差在阈值内） |
| 训练 → RL | 权重同步，模型定义一致 |
| 上线 → 数据 | 日志脱敏、采样、标注 |

接口处的问题往往比单个环节内部的问题更难发现：
- 例如训练框架与推理引擎对 RoPE 的实现存在细微差异
  - 评测分数可能只下降不到 1 分
  - 却会在长上下文任务上造成明显的质量下降
- 只有在接口处做逐层的 logits 比对，才能尽早发现这类问题

### 9.2 关键指标与典型量级

本节中的百分位延迟采用以下定义：
- P99：将所有请求的延迟从小到大排序后第 99 百分位处的值，即 99% 的请求延迟不超过该值
- P95、P50 同理
  - P50 即中位数

#### 9.2.1 训练指标

| 指标 | 定义 | 健康量级 |
|---|---|---|
| MFU | 4.4 节 | 稠密模型 40%–55%，MoE 更低 |
| tokens/s/GPU | 每块 GPU 每秒处理的训练 token 数 | H100 BF16、MFU 45%、8K 序列：8B 模型约 7000，70B 模型约 900 |
| 有效训练时间比例（training goodput） | 产生有效训练进展的时间 / 作业总时间 | > 90% |
| 作业 MTBF | 平均无故障时间，即两次意外中断之间的平均间隔 | 万卡规模约 2–3 h |
| 恢复时间 | 从故障发生到重新开始训练的时间 | < 10 min，先进做法 < 3 min |
| checkpoint 阻塞时间 | 保存 checkpoint 时训练被阻塞的时间 | 异步保存 < 10 s |
| loss spike 次数 | 每万步出现的 loss 突增次数 | 越少越好，每次都应能定位原因 |
| step 时间方差 | 各 rank 单步耗时的分布 | 尾部与中位数的差距 < 5% |

训练 goodput 中不计入有效进展的时间包括：
- 故障后从上一个 checkpoint 到故障点之间需要重算的步数
- 故障检测与恢复所需的时间
- checkpoint 阻塞时间
- 作业启动和初始化时间

以万卡作业为例：
- 万卡作业 MTBF 为 2–3 h 时，即使每次恢复只需 5 min，每天也有约 10 次中断
- 若 checkpoint 间隔为 30 min，平均每次还要重算 15 min
  - 合计每天约 3.3 h 的损失，goodput 仅约 86%
- 这说明恢复时间和 checkpoint 间隔（4.2.3 节）对 goodput 的影响与 MFU 同等重要

#### 9.2.2 推理指标

| 指标 | 健康量级（对话类服务） |
|---|---|
| TTFT P99 | < 1–2 s |
| TPOT P99 | < 50–100 ms |
| 吞吐（每块 GPU 每秒输出 token 数） | 70B FP8 约 2000–5000，8B 约 1 万–3 万 |
| goodput | 满足 SLO 的请求比例 > 99% |
| KV cache 利用率 | 70%–95%；长期处于 100% 说明正在发生抢占 |
| 前缀缓存命中率 | 多轮对话场景 50%–90% |
| 抢占率 | 越低越好 |
| 每百万 token 成本 | 视模型规模，0.1 美元到数美元 |

推理中的 goodput 与训练中的含义不同：
- 指同时满足 TTFT 与 TPOT SLO 的请求在全部请求中所占的比例（或满足 SLO 的请求每秒吞吐）
- 单看总吞吐会掩盖问题
  - 把 batch 调大可以提高每秒输出的 token 数
  - 但若 TPOT 因此超出 SLO，这些 token 对用户而言并不合格
- 抢占（preemption）：KV cache 显存不足时，调度器暂停部分正在运行的请求并释放其 KV cache
  - 待显存空出后再重新计算或换入

#### 9.2.3 集群指标

| 指标 | 含义 |
|---|---|
| GPU 分配率 | 被作业占用的 GPU 数 / GPU 总数 |
| GPU 有效利用率 | 已分配的 GPU 中实际在高效计算的比例（以 MFU 加权） |
| 排队时间 | 从作业提交到开始运行的时间 |
| 碎片率 | 处于空闲但无法满足任何排队作业需求的 GPU 所占比例 |
| 节点健康率 | 通过健康检查的节点所占比例 |
| 故障率 | 每千块 GPU 每天的故障次数，按故障类型分别统计 |

碎片的典型来源是拓扑约束：
- 例如一个需要 64 块 GPU 且要求在同一 IB 子网内的作业
  - 即使集群中零散地空闲着 100 块 GPU，只要它们分布在不同子网，就无法调度
  - 这些 GPU 计入碎片

#### 9.2.4 分配率与有效利用率的区别

- GPU 分配率达到 95% 并不表示集群得到了高效利用
  - 如果已分配 GPU 上作业的 MFU 只有 20%，或者一半时间在等待数据加载和故障重启，实际有效算力只有标称算力的一小部分
- 例如分配率 95%、有效利用率 40% 时，集群的实际效率只有 $0.95\times0.40=38\%$
- 评估集群效率应当同时考察两者
  - 以分配率 × 有效利用率作为综合指标

#### 9.2.5 可靠性指标

- 设备数量增加以后，硬件故障、网络中断、人为误操作以及故障恢复过程本身，可能主导系统实际可用的吞吐
  - 因此可靠性必须作为性能指标的一部分，而不是单独的运维事项
- 只在所有设备健康时测量平均 token/s 的基准测试，会系统性地高估大规模系统的实际能力

完整的评测应包含以下几项。

| 指标 | 含义 | 原因 |
|---|---|---|
| 故障恢复时间 | 从故障发生到服务或训练恢复正常的时间，包括检测、定位、隔离和重启 | 决定每次故障造成的损失；万卡训练中每天发生多次故障，恢复时间直接决定 goodput |
| 尾延迟（P95 / P99） | 慢请求的延迟 | 平均延迟会被大量快速请求拉低，掩盖少数请求的长时间等待；交互服务的用户体验由尾部决定 |
| 任务成功率 | 请求或作业正常完成、未因超时或错误而失败的比例 | 对 Agent 和长任务尤其重要，一次任务包含数十次模型调用和工具调用，任一环节失败都会导致整个任务失败 |
| 故障后的降级吞吐 | 部分设备失效时系统仍能提供的吞吐 | 衡量故障域隔离是否有效：一块 GPU 故障是只损失 1/72 的能力，还是整个机柜下线 |

支撑这些指标的工程手段包括：
- ECC（错误检测与纠正码）：在显存和片上存储中检测并纠正位翻转
- 冗余：备用 GPU、冗余专家、多副本服务
- checkpoint 与快速重启
- 可观测性（9.6 节的监控看板）
- 故障域隔离（8.2.3 节）

任务成功率的计算还说明了可靠性为何在长任务中更加重要：
- 若单次模型或工具调用的成功率为 99.5%，一个包含 50 次调用的 Agent 任务的整体成功率只有 $0.995^{50}\approx78\%$

#### 9.2.6 FLOPs 之外的系统瓶颈

只用峰值 FLOPs 或 FLOPs 利用率衡量 AI 系统是不充分的：
- 实际推理与训练受到以下多个层次的共同制约
- 某一层的性能提升可能被另一层的瓶颈抵消

| 瓶颈 | 机制 | 典型表现 |
|---|---|---|
| 存储墙（memory wall） | 过去二十年 GPU 的峰值算力增长远快于显存带宽和容量的增长，大量负载的算术强度低于 roofline 脊点 | decode 阶段每步都要读取全部权重和 KV cache，算力利用率常低于 10% |
| 能耗与功率 | 集群总功率受电力接入和散热能力限制，单块 GPU 的功率受供电和封装限制 | 功耗封顶导致降频（9.4.1 节）；新集群的规模由可用电力而非预算决定（8.9.1 节） |
| 长上下文与缓存 | KV cache 随上下文长度线性增长，注意力计算随长度平方增长 | 长上下文请求挤占显存，batch 变小，吞吐下降 |
| 可靠性 | 设备数量越多，单位时间内发生故障的概率越高 | 万卡作业每 2–3 h 中断一次（9.2.1 节） |
| 异构性 | 集群中混合多代 GPU、TPU、CPU 和定制 ASIC，以及不同的存储与网络设备 | 同一模型需要在多种硬件上分别优化，跨设备的数据传输和调度复杂 |

异构性正在成为大规模系统的常态：
- 不同云服务商和芯片厂商的架构差异在扩大
  - GPU、TPU 和各类定制 ASIC 的数据布局要求、内存层次、互连拓扑和编程模型各不相同
- 在这样的环境中，以下问题都可能成为比单个 kernel 性能更关键的问题
  - 跨设备的数据布局转换
  - 显存拷贝
  - 请求路由
  - 内存一致性（coherence / consistency）
  - 任务调度
- 硬件异构带来了专用化的效率
- 软件抽象层的任务是让应用不被绑定在某一种设备上，同时不丢失这部分效率

#### 9.2.7 端到端加速比与 Amdahl 定律

- 优化单个组件时，其对整体性能的贡献受该组件在端到端耗时中所占比例的限制
- Amdahl 定律给出了这一关系：设某部分在总耗时中的占比为 $f$，该部分被加速 $s$ 倍，则整体加速比为

$$S=\frac{1}{(1-f)+f/s}$$

一个在线推理请求依次经过以下环节：
- 网络接入
- 请求路由
- 排队
- CPU 预处理（分词、模板拼装）
- GPU kernel 执行
- KV cache 管理
- 后处理
- 流式输出

数值示例：
- 假设某个 attention kernel 被优化到原来的 2 倍速度，而它只占端到端时间的 20%，则

$$S=\frac{1}{0.8+0.2/2}\approx1.11$$

- 整体只加速 11%
- 即使把这个 kernel 的耗时降为零，整体加速比也不超过 $1/0.8=1.25$
- 因此系统优化应当从端到端关键路径的剖析开始，优先处理占比最大的部分

端到端视角下还需要处理几组相互冲突的目标：
- 吞吐与延迟
  - 更大的 batch 提高吞吐但增加每个请求的等待时间
- TTFT 与 TPOT
  - 优先调度新请求的 prefill 可以降低 TTFT
  - 但会打断正在 decode 的请求，使其 TPOT 升高
- 平均延迟与尾延迟

### 9.3 成本模型：两个算例

#### 9.3.1 TCO 构成

- TCO（total cost of ownership，总拥有成本）：一套系统在其生命周期内的全部成本，包括购置成本和运行成本
  - 前者称为资本支出（CapEx，capital expenditure）
    - 即硬件采购等一次性投入，在会计上按折旧年限分摊到每一年
  - 后者称为运营支出（OpEx，operating expenditure）
    - 即电费、机房租金、运维人力等持续发生的费用
- 自建集群的年度 TCO 可写为：

$$\text{TCO}_{\text{年}} = \underbrace{\frac{\text{硬件采购}}{\text{折旧年限}}}_{\text{资本}} + \underbrace{P_{\text{集群}}\times 8760\times\text{PUE}\times\text{电价}}_{\text{电费}} + \text{机房} + \text{运维人力} + \text{网络与存储}$$

公式中各项的含义：
- $P_{\text{集群}}$ 为集群 IT 设备的平均功率（kW）
- 8760 为一年的小时数
- PUE（power usage effectiveness，电能使用效率）为数据中心总耗电与 IT 设备耗电之比
  - 反映制冷、配电等辅助设施的额外能耗
  - 风冷机房通常为 1.3–1.6，液冷机房可低至 1.1–1.2

#### 9.3.2 算例一：单块 H100 的每卡时成本

| 项 | 假设 | 每卡时成本 |
|---|---|---|
| 采购（含服务器、网络分摊） | 每块 GPU 3.5 万美元，4 年折旧 | $35000/(4\times8760)\approx 1.00$ 美元 |
| 电力 | 每块 GPU 含分摊功率 1.2 kW，PUE 1.3，电价 0.08 美元/kWh | $1.2\times1.3\times0.08\approx 0.12$ 美元 |
| 机房、运维、存储 | 经验值 | ~0.3 美元 |
| 合计 | 满负荷利用 | ~1.4 美元 |
| 实际利用率 70% 时 | | ~2.0 美元 |

- 表中「每块 GPU 含分摊功率 1.2 kW」高于 H100 自身 700 W 的 TDP
  - 因为还要分摊 CPU、内存、网卡、交换机和存储的功耗
- 折旧占成本的约 70%，而折旧不随使用量变化
  - GPU 空闲时，这部分成本照样发生
- 有效卡时成本等于满负荷成本除以利用率
  - 利用率从 70% 提高到 90%，每有效卡时成本从 2.0 美元降至 1.6 美元
- 集群利用率与 MFU 一样，直接决定单位算力的成本

#### 9.3.3 算例二：训练与推理成本对比

模型：70B，预训练 15T token。

- 训练：按 0.1.3 节的计算需要 $4.4\times10^6$ GPU·h
  - 按每卡时 2 美元计约 880 万美元
- 推理：日均处理 1000 亿 token（输入与输出合计）
  - 按 9.3.2 节和 5.13 节的量级，每百万 token 的综合成本约 0.5 美元
  - 每天 5 万美元，一年约 1800 万美元

- 一年的推理花费约为训练的 2 倍，且推理成本随模型的服务时间持续累积
- 因此 GQA、MLA、量化友好的结构和 MoE 等面向推理效率的设计，需要在模型设计阶段就纳入考虑
  - 例如采用 MLA 可能使训练成本增加数个百分点
  - 但若它使每块 GPU 可服务的并发量提高数倍，推理阶段节省的成本将远超训练阶段的额外投入

### 9.4 故障案例

本节的八个案例均按「现象、排查、原因、处理」四个环节组织：
- 它们覆盖了大规模系统中最常见的三类根因：硬件异常、状态保存不完整和数值不一致

#### 9.4.1 案例一：MFU 由 45% 骤降至 30%

- 现象：作业重启后训练吞吐明显下降，但 loss 曲线正常，说明问题在于性能而非正确性。
- 排查：各 rank 单步耗时的分布显示有 8 个 rank 明显偏慢，且都位于同一台服务器上
  - 在同步数据并行中，每一步都要等最慢的 rank 完成，因此单台服务器变慢会拖慢整个作业
  - `nvidia-smi -q` 显示这台服务器的 GPU 时钟频率被限制，原因是触发了功耗封顶
- 原因：机房对该机柜的配电进行调整后，该节点被设置了更低的功耗上限，GPU 为满足功耗限制而自动降频。
- 处理：调度器把该节点标记为降级并替换为其他节点；在节点健康检查中加入 GPU 时钟频率和功耗上限的检查项，使同类问题在作业启动前即被发现。

#### 9.4.2 案例二：训练每隔数小时出现 NCCL 超时

- 现象：NCCL watchdog（检测集合通信长时间未完成的监控线程）在随机时间点报告超时，作业中断；重启后又能正常运行数小时。
- 排查：Flight Recorder（NCCL 记录每个 rank 最近若干次集合通信状态的诊断工具）显示
  - 每次超时时最后完成通信的总是同一对节点
  - 交换机日志中对应端口的 CRC 校验错误计数持续增长
- 原因：一个光模块老化，产生的误码触发链路层重传
  - 重传在大部分时间里只造成轻微延迟
  - 但在偶发的连续误码时使单次集合通信的耗时超过超时阈值
- 处理：更换该光模块；在网络监控中加入交换机端口误码率告警，在误码率开始上升时即提前更换部件。

#### 9.4.3 案例三：从 checkpoint 恢复后 loss 跳变

- 现象：从 checkpoint 恢复训练后，loss 比中断前高约 0.05，之后经过较长时间才逐渐回落。
- 排查：用 4.10.5 节的「连续训练与中断恢复」对比测试复现该问题，发现恢复后的学习率与中断前不一致。
- 原因：学习率调度器（LR scheduler）的状态没有写入 checkpoint，恢复时调度器从 warmup 阶段重新开始，学习率出现突变。
- 处理：在 checkpoint 中保存所有训练状态，包括优化器、学习率调度器、数据加载器位置和随机数生成器状态；把续训一致性测试加入 CI。

#### 9.4.4 案例四：推理服务 P99 TTFT 偶发升至 10 秒

- 现象：平均 TTFT 正常，但 P99 TTFT 偶尔高达 10 秒，且与流量峰值并不完全对应。
- 排查：日志显示，每个高 TTFT 时间点都有超长 prompt（10 万 token 以上）进入系统。
- 原因：服务没有启用 chunked prefill（把长 prompt 的 prefill 切成多段，与其他请求的 decode 交替执行）
  - 一个超长 prompt 的 prefill 独占了一个调度周期，耗时数秒
  - 其后到达的所有请求都在队列中等待
- 处理：启用 chunked prefill；把超长请求路由到专门的长上下文实例，避免与普通对话请求争用同一调度队列。

#### 9.4.5 案例五：新量化模型上线后部分任务质量下降

- 现象：离线评测的总分只下降了 0.3 分，但上线后代码类请求的用户投诉明显增加。
- 排查：按任务类型分别评测，发现代码和数学任务下降 3–5 分，通用对话几乎没有变化；总分的下降被大量通用任务平均掉了。
- 原因：量化校准（用一批样本统计各通道的激活范围以确定量化参数）只使用了通用对话文本
  - 与代码相关的激活通道中的离群值没有被覆盖
  - 导致这些通道的量化误差较大
- 处理：在校准集中加入代码和数学数据；上线前按任务类型分别评测并分别设置质量门槛。

#### 9.4.6 案例六：RL 训练中途 reward 崩溃

- 现象：reward 持续上升一段时间后突然下降，模型输出开始出现重复和乱码。
- 排查：监控显示，训练侧 logprob 与 rollout 侧 logprob 的概率比分布在崩溃前几十步明显变宽。
- 原因：推理引擎升级后更换了 MoE kernel，训推数值不一致加剧（8.8.6 节），重要性采样比值的方差急剧增大，少数样本主导了梯度。
- 处理：引入截断重要性采样；规定推理引擎升级前必须先通过训推一致性测试。

#### 9.4.7 案例七：单块 GPU 持续产生错误结果

- 现象：某次 loss spike 无法复现；在 DP 组内比对参数哈希时，发现某个 rank 的参数与其他 rank 不同
  - 正常情况下，同一 DP 组内所有 rank 的参数在每步 AllReduce 之后应逐位一致
- 排查：在该 GPU 上反复运行固定输入的 GEMM 自检，约每 1000 次出现一次与参考结果逐位不同的输出。
- 原因：静默数据错误（SDC）
  - 该 GPU 的某个计算单元存在硬件缺陷，在特定条件下产生错误结果，却不触发 ECC 或任何错误报告
- 处理：隔离该 GPU；把参数哈希比对和定期 GEMM 自检纳入常规监控。

#### 9.4.8 案例八：checkpoint 写入过程中存储空间耗尽

- 现象：保存 checkpoint 时作业阻塞数十分钟，最后报告 IO 错误。
- 排查：存储配额已满，历史 checkpoint 没有按策略清理。
- 处理：制定保留策略，例如保留最近 N 个 checkpoint，另每隔 M 步永久保留一个；每次写入前检查剩余空间是否足够容纳一个完整的 checkpoint。

### 9.5 两个案例研究

#### 9.5.1 Llama 3 405B 的训练基础设施

| 方面 | 做法 |
|---|---|
| 规模 | 最多 16,384 块 H100，每块 GPU 功耗 700 W |
| 网络 | 2.4 万块 GPU 规模的 RoCE 集群（另有 IB 集群），三层 Clos 拓扑，400 Gb/s；采用 E-ECMP 负载均衡和深缓冲交换机 |
| 存储 | Tectonic 分布式文件系统，容量 240 PB，持续吞吐 2 TB/s，峰值 7 TB/s |
| 并行 | 4D 并行：TP 8 × CP（长上下文阶段）× PP 16 × DP（FSDP） |
| MFU | 38%–43%（BF16） |
| 稳定性 | 54 天内发生 466 次中断，有效训练时间 > 90%；使用 NCCLX（改进版 NCCL），并实现快速故障检测与定位 |
| 观察 | 白天气温升高导致吞吐波动 1%–2%；数万块 GPU 同时开始或停止计算时，数据中心功耗瞬间波动数十 MW，对电网构成挑战 |

- 466 次中断平均每 2.8 小时发生一次，与 9.2.1 节给出的万卡 MTBF 量级一致
- 在这一中断频率下仍保持 90% 以上的有效训练时间
  - 依赖的是自动化的故障检测、节点替换和快速重启
- 功耗瞬间波动的问题说明，在这一规模下，集群已经成为电网中不可忽视的负荷
  - 训练作业的同步特性（所有 GPU 同时进入计算或同时等待通信）会直接反映为电力负荷的剧烈变化

#### 9.5.2 DeepSeek-V3 的成本

| 项 | 数据 |
|---|---|
| 预训练 | 14.8T token，266.4 万 H800 GPU·h |
| 上下文扩展 | 11.9 万 GPU·h |
| 后训练 | 0.5 万 GPU·h |
| 合计 | 278.8 万 GPU·h，按 2 美元约 558 万美元 |
| 每万亿 token | 约 18 万 GPU·h |

- 上述数字仅为最终一次训练的 GPU 租赁成本
  - 不包括前期研究、消融实验、数据和人力成本
- 作为参照，按 0.1.3 节的方法估算，稠密 70B 模型训练 15T token 约需 440 万 GPU·h，即每万亿 token 约 29 万 GPU·h
  - 而 DeepSeek-V3 的总参数量接近其 10 倍
- 成本较低的系统层面原因包括：
  - MoE 结构使每 token 只激活 37B 参数
  - FP8 训练提高了 GEMM 吞吐
  - DualPipe 使 EP 的 All-to-All 通信与计算重叠
  - 不使用 TP，省去了大量 AllReduce 通信
  - 针对通信和 GEMM 深度优化的 kernel

### 9.6 监控看板的最小集合

#### 9.6.1 训练看板

| 面板 | 内容 | 告警 |
|---|---|---|
| 进度 | loss（训练 / 验证）、累计 token 数、学习率 | loss spike、NaN |
| 效率 | tokens/s、MFU、单步耗时 P50/P99 | MFU 下降 10% 以上 |
| 稳定性 | grad norm、loss scale、各层激活 / 梯度范数 | grad norm 突增 |
| 硬件 | 每个 rank 的单步耗时热力图、GPU 温度 / 时钟 / 功耗、XID 错误 | straggler、XID |
| 网络 | NCCL 操作耗时、交换机端口误码 | 误码增长 |
| 可靠性 | 中断次数、恢复时间、有效训练时间比例 | 恢复超时 |

表中术语的含义：
- straggler：单步耗时明显长于其他 rank、拖慢整个同步作业的 rank
- XID：NVIDIA 驱动报告的 GPU 错误代码
  - 不同编号对应显存 ECC 错误、NVLink 错误、GPU 掉卡等不同故障
- grad norm 突增通常先于 loss spike 出现，是更早的预警信号

#### 9.6.2 推理看板

| 面板 | 内容 | 告警 |
|---|---|---|
| SLO | TTFT / TPOT 的 P50/P99、goodput | P99 超过阈值 |
| 负载 | QPS、并发请求数、排队长度 | 队列长度持续增长 |
| 资源 | KV cache 利用率、抢占次数、GPU 利用率 | 抢占频繁 |
| 缓存 | 前缀缓存命中率 | 命中率骤降（通常是请求路由出现问题） |
| 质量 | 输出长度分布、错误率、拒答率 | 分布突变 |
| 可靠性 | 请求失败率、重试率、副本故障与恢复时间 | 失败率上升 |
| 成本 | 每百万 token 成本、副本数 | 成本突增 |

- 前缀缓存命中率骤降通常意味着请求路由出了问题
  - 多轮对话的后续请求本应被路由到持有其前缀 KV cache 的副本
  - 若路由策略失效而把请求随机分发，每个请求都要重新计算完整的 prefill

#### 9.6.3 采集要点

- 所有指标按 rank 和节点打标签，热力图才能显示出 straggler 所在的位置；
- 延迟指标应以分布（直方图）形式存储，不能只存平均值，否则无法计算 P99；
- 训练日志和集群监控使用同一套时间轴，才能把 loss spike 与同一时刻的硬件事件（如 XID 错误、链路抖动）对应起来。

#### 9.6.4 工作负载画像清单

- 监控看板描述系统运行时的状态
- 在设计新系统、做容量规划或评估一款加速器之前，还需要先对目标工作负载进行画像（workload profiling）
  - 即系统地测量负载的特征
  - 需要测量的项目如下

| 类别 | 测量项 | 用途 |
|---|---|---|
| 请求长度 | 输入与输出长度的分布，包括 P50、P95，而不只是均值 | 长度分布决定 KV cache 需求和 decode 长尾；均值相同的两种负载，长尾差异可能使所需 GPU 数相差一倍 |
| 流量模式 | 并发量、峰谷比、请求到达过程（是否突发） | 决定副本数、弹性扩缩容策略和排队延迟 |
| 阶段比例 | prefill 与 decode 的耗时比例、每请求 token/s | 决定是否值得 PD 分离，以及两类节点的配比 |
| 显存构成 | 模型权重、KV cache、激活和临时缓冲区各自的大小 | 决定并行配置和单块 GPU 能容纳的 batch |
| 资源利用 | 显存带宽利用率、算力利用率、跨设备通信量 | 判断负载在 roofline 上的位置，确定瓶颈属于带宽、算力还是通信 |
| 效率与可靠性 | 每 token 能耗、每请求成本、故障率、重试率 | 计算 TCO 和实际可用吞吐 |
| 模型演化风险 | 稠密或 MoE 结构、注意力变体、低精度格式的变化趋势 | 评估硬件和系统设计在未来模型上是否仍然适用 |

最后一项对硬件设计尤其重要：
- 一款专用加速器从定义到量产通常需要两到三年
  - 在这段时间内，模型结构、上下文长度、MoE 的专家粒度和主流量化格式都可能发生变化
- 专用化带来效率
  - 但过度针对当前负载的专用化可能在芯片上市时已经不适用

### 9.7 小结

- 在一个模型的生命周期中，推理花费常常超过训练，模型设计阶段就应考虑推理成本。
- 集群效率等于分配率与有效利用率的乘积，两者需要同时考察。
- 每卡时成本的主要部分是折旧，利用率直接决定有效成本。
- 可靠性应作为性能指标的一部分，评测需包含故障恢复时间、尾延迟和任务成功率
  - 系统瓶颈不止于 FLOPs，还包括存储墙、能耗、长上下文、可靠性和异构性
- 端到端优化受 Amdahl 定律约束，应从关键路径的剖析出发，优先处理占比最大的环节。
- 故障的根因主要集中在硬件异常、状态保存不完整和数值不一致三类，监控和测试应覆盖这三类问题。

<br>

## 附录

### A. 术语表

#### A.1 硬件、加速器与性能模型

| 术语 | 英文 | 定义 |
|---|---|---|
| SM | Streaming Multiprocessor | NVIDIA GPU 中包含 CUDA Core、Tensor Core、寄存器堆和共享内存的基本计算单元。 |
| Warp | — | 由 32 个线程组成、在 SIMT 模型下同步执行同一条指令的调度单位。 |
| Tensor Core | — | SM 内执行小尺寸矩阵乘加（MMA）运算的专用硬件单元。 |
| HBM | High Bandwidth Memory | 通过硅中介层与计算芯片封装在一起的堆叠式 DRAM，是 GPU 的主存（显存）。 |
| SMEM | Shared Memory | 位于 SM 内、由同一线程块内线程共享、由程序显式管理的片上存储。 |
| TMA | Tensor Memory Accelerator | Hopper 架构引入的硬件单元，负责在全局内存与共享内存之间异步搬运多维数据块。 |
| NVLink / NVSwitch | — | NVLink 是 GPU 之间的点对点高速互连链路，NVSwitch 是连接多条 NVLink 的交换芯片。 |
| IB | InfiniBand | 面向数据中心、原生支持 RDMA 的低延迟高带宽网络协议与设备体系。 |
| RoCE | RDMA over Converged Ethernet | 在以太网上承载 RDMA 语义的网络协议。 |
| GPUDirect RDMA | — | 允许网卡绕过 CPU 和主机内存、直接读写 GPU 显存的数据通路。 |
| PUE | Power Usage Effectiveness | 数据中心总功耗与 IT 设备功耗之比，衡量供电与制冷开销。 |
| XID | — | NVIDIA 驱动在内核日志中报告的 GPU 错误事件编号。 |
| SDC | Silent Data Corruption | 硬件产生错误计算结果而未触发任何错误检测或告警的故障。 |
| 算术强度 | Arithmetic Intensity | 一个 kernel 执行的浮点运算次数与其访问主存的字节数之比，单位为 FLOP/Byte。 |
| 脊点 | Ridge Point | 峰值算力与主存带宽之比 $P_{\text{peak}}/B$，即 kernel 由带宽受限转为算力受限时对应的算术强度。 |
| busbw | Bus Bandwidth | 将集合通信的算法带宽乘以与算法相关的系数（AllReduce 为 $2(p-1)/p$）后得到的、反映链路实际负载的带宽指标。 |
| DSA | Domain-Specific Architecture | 针对特定应用领域（如深度学习）的计算模式定制数据通路与存储层次的处理器架构。 |
| NPU | Neural Processing Unit | 专为神经网络计算设计的 DSA 类加速器的统称。 |
| TPU | Tensor Processing Unit | Google 设计的、以脉动阵列矩阵单元（MXU）为核心的深度学习加速器。 |
| 脉动阵列 | Systolic Array | 由规则排列的乘加单元组成、数据在相邻单元间逐拍传递并复用的矩阵计算阵列。 |
| ICI | Inter-Chip Interconnect | TPU 芯片之间直接相连、构成 2D/3D torus 拓扑的高速互连链路。 |
| OCS | Optical Circuit Switch | 通过微镜阵列重定向光路、在 TPU pod 内动态重构芯片间拓扑的光电路交换机。 |
| Cube 单元 | Cube Unit | 昇腾达芬奇架构中每拍完成一个 $16\times16\times16$ 矩阵乘加的矩阵计算单元。 |
| scratchpad 存储 | Scratchpad Memory | 由软件或编译器显式分配和搬运数据、不具备硬件缓存一致性管理的片上存储。 |
| VLIW | Very Long Instruction Word | 由编译器在一条长指令中静态打包多个可并行操作、硬件不做动态调度的指令集架构。 |

#### A.2 训练

| 术语 | 定义 |
|---|---|
| DP / DDP | 数据并行：每个设备持有完整模型副本，处理不同数据分片，并在反向后对梯度做 AllReduce。 |
| ZeRO / FSDP | 将优化器状态、梯度和参数切分到数据并行组各成员上的数据并行方法。 |
| HSDP | 在组内使用 FSDP 切分、在组间使用普通 DP 复制的混合分片数据并行。 |
| TP | 张量并行：将单层内的权重矩阵按行或列切分到多个设备上协同计算。 |
| SP | 序列并行：Megatron 中指在 TP 组内沿序列维切分 norm 与 dropout 的激活，广义上也包括 Ulysses 式的注意力序列切分。 |
| CP | 上下文并行：沿序列维切分整个注意力计算，以 Ring Attention 等方式在设备间传递 KV。 |
| PP | 流水线并行：将模型的层按深度划分为多个阶段，分布在不同设备上依次执行。 |
| EP | 专家并行：将 MoE 层的不同专家放置在不同设备上，通过 All-to-All 分发 token。 |
| 1F1B | 稳态阶段每个流水线阶段交替执行一次前向与一次反向的流水线调度策略。 |
| Bubble | 流水线并行中因阶段间依赖导致设备空闲等待的时间段。 |
| Micro-batch | 流水线调度和梯度累积中一次前向/反向处理的最小样本批。 |
| Activation recomputation | 反向传播时重新执行前向计算以恢复激活、从而减少激活显存的技术。 |
| MFU | Model FLOPs Utilization：按模型理论计算量（不含重计算）折算的实际算力与硬件峰值算力之比。 |
| HFU | Hardware FLOPs Utilization：按硬件实际执行的计算量（含重计算）折算的算力与峰值算力之比。 |
| Goodput | 训练中指用于有效推进训练的时间占总时间的比例，推理中指满足 SLO 的请求吞吐。 |
| Straggler | 因硬件、网络或负载原因明显慢于其他节点、拖慢同步步骤的节点。 |
| MTBF | Mean Time Between Failures：系统相邻两次故障之间的平均运行时间。 |
| Loss spike | 训练过程中 loss 在少数步内突然大幅升高的现象。 |
| Deterministic training | 在相同代码、数据、配置与硬件下逐位复现训练结果的训练方式。 |
| Replay | 从某个 checkpoint 出发、以相同数据顺序重新执行一段训练以复现或定位问题的方法。 |

#### A.3 推理

| 术语 | 定义 |
|---|---|
| Prefill / Decode | Prefill 指并行处理整个 prompt 并生成 KV cache 的阶段，Decode 指逐个生成输出 token 的阶段。 |
| TTFT | Time To First Token：从请求到达至返回第一个输出 token 的延迟。 |
| TPOT | Time Per Output Token：首 token 之后平均每生成一个输出 token 所需的时间。 |
| ITL | Inter-Token Latency：相邻两个输出 token 之间的实际间隔，其均值近似等于 TPOT。 |
| Continuous batching | 以单步迭代为粒度将新请求加入、将完成请求移出 batch 的调度方式。 |
| PagedAttention | 以固定大小的块为单位分配和索引 KV cache、消除连续分配碎片的注意力实现。 |
| Prefix caching | 在请求之间复用相同 prompt 前缀已计算的 KV cache。 |
| Chunked prefill | 将长 prompt 的 prefill 切分为多个块，与 decode 请求混合调度。 |
| PD disaggregation | 将 prefill 和 decode 部署在不同实例上，并在两者之间传输 KV cache 的架构。 |
| Speculative decoding | 由草稿模型或草稿头先生成多个候选 token，再由目标模型一次前向并行验证的解码方法。 |
| GQA / MQA / MLA | 分别为多个查询头共享一组 KV 头、所有查询头共享单个 KV 头、以及将 KV 压缩为低秩潜在向量的注意力变体。 |
| W4A16 / W8A8 | 量化配置记法，W 后数字为权重位宽，A 后数字为激活位宽。 |
| Batch invariance | 单个请求的输出与其所在 batch 的大小和组成无关的性质。 |

#### A.4 编译

| 术语 | 定义 |
|---|---|
| Graph break | 图捕获过程中遇到无法追踪的代码而被迫拆分计算图的位置。 |
| Guard | 编译器为已编译图记录的、决定其能否被后续调用复用的运行时条件。 |
| Epilogue fusion | 在 GEMM 结果写回主存之前，在同一 kernel 内完成偏置、激活等后续逐元素操作。 |
| Autotuning | 对候选 kernel 配置逐一实测并选取最快配置的过程。 |
| CUDA Graph | 将一串 kernel 启动录制为图、之后以单次提交重放以降低 CPU 启动开销的机制。 |
| Tiling | 将大规模计算按块切分，使每个块的输入和中间结果能驻留在寄存器、SMEM 等片上存储中反复复用，从而减少主存访问。 |

#### A.5 数值格式、量化与数据搬运

| 术语 | 定义 |
|---|---|
| Scale factor | 缩放因子：把低位宽整数或浮点编码映射回实数的乘数，$\hat{x}=s\,q$；可按张量、按通道或按块各存一个。 |
| Codebook | 码本：某一数值格式能够表示的全部离散数值的集合，量化即把每个实数映射到码本中最接近的元素。FP4 E2M1 的码本只有 $\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ 共 15 个不同值。 |
| Microscaling（MX） | OCP 标准定义的块缩放格式：每 32 个元素共享一个 8 bit 指数（E8M0）缩放因子，元素本身为 FP8、FP6、FP4 或 INT8，如 MXFP4。 |
| Stochastic rounding | 随机舍入：以与距离成反比的概率舍入到相邻两个可表示值之一，使舍入结果的期望等于原值，从而避免低精度累加中小更新量被系统性舍掉。 |
| Offloading | 卸载：把暂时不用的权重、优化器状态或 KV cache 从 HBM 移到 CPU 内存、SSD 等容量更大、成本更低的存储层级。 |
| Eviction | 逐出：存储空间不足时按 LRU 等策略从高速层级中移除部分数据（如 KV 块），被逐出的数据需重新计算或从下层存储取回。 |
| Prefetching | 预取：在数据被使用之前提前发起搬运，使传输时间与计算重叠。 |
| DMA / RDMA | DMA 是由专用引擎在设备与内存之间搬运数据、无需 CPU 逐字节参与的机制；RDMA 把这一机制扩展到网络，使一台机器的网卡可直接读写另一台机器的内存。 |
| ECC | Error-Correcting Code：在存储数据旁附加冗余校验位，常见配置可纠正单比特错误并检测双比特错误（SECDED）。 |
| TMR | Triple Modular Redundancy：三模冗余，用三份相同电路并行计算并多数表决，可屏蔽任一份电路的单点错误，代价约为三倍面积与功耗。 |
| Linear attention | 线性注意力：用特征映射替代 softmax，使注意力可写成固定大小状态的递推更新，每个 token 的计算与显存开销不随上下文长度增长。 |
| SLO | Service Level Objective：服务水平目标，即以可测量指标表述的服务承诺，如 P99 TTFT 小于 2 s、P99 TPOT 小于 50 ms。 |

#### A.6 封装、供电、功耗与新型计算

| 术语 | 定义 |
|---|---|
| Chiplet | 小芯片：把一颗大芯片拆成多颗功能裸片分别制造、再通过先进封装互连成一个整体，以提高良率并允许不同工艺混用。 |
| Interposer | 中介层：位于裸片与封装基板之间、带有高密度布线的硅或有机层，用于连接多颗裸片与 HBM，如 CoWoS 中的硅中介层。 |
| TSV | Through-Silicon Via：硅通孔，即垂直贯穿硅片的金属导通孔，用于 HBM 内 DRAM 裸片的堆叠互连和 3D 集成。 |
| Hybrid bonding | 混合键合：把两颗裸片表面的铜焊盘与介质层直接键合，无需焊料凸点，互连间距可降到 10 µm 以下。 |
| CTE | Coefficient of Thermal Expansion：热膨胀系数，即温度每升高 1 K 时材料长度的相对变化量。 |
| FEA | Finite Element Analysis：有限元分析，把复杂结构离散为大量小单元后数值求解温度场、应力场等物理量的方法。 |
| Electromigration | 电迁移：高电流密度下电子动量把金属原子沿导线推移，逐渐形成空洞或凸起并导致断路或短路的失效机制。 |
| PDN | Power Delivery Network：供电网络，即从电压调节模块经封装、凸点到晶体管的整条供电通路。 |
| IR drop | 电流 $I$ 流过 PDN 电阻 $R$ 产生的电压下降；电压低于阈值时电路时序失效，需降频或提高供电电压补偿。 |
| CMOS | Complementary Metal-Oxide-Semiconductor：由 NMOS 与 PMOS 晶体管互补构成逻辑门的集成电路技术，是当前数字芯片的主流工艺。 |
| Dennard scaling | 晶体管尺寸缩小时电压与电流同比例下降、单位面积功耗保持不变的缩放规律；约在 2005 年后因漏电和阈值电压限制而失效。 |
| In-memory computing | 存内计算：在存储阵列内部或紧邻处直接完成乘加运算（如利用阻变存储器阵列的欧姆定律与基尔霍夫定律做模拟矩阵向量乘），以减少数据搬运能耗。 |
| Neuromorphic computing | 神经形态计算：借鉴生物神经系统的脉冲、事件驱动与存算一体特征构建的计算硬件，如 Intel Loihi、IBM TrueNorth。 |
| Heterogeneity | 异构性：同一系统中 CPU、GPU、加速器以及 SRAM、HBM、DRAM、SSD 等不同计算单元与存储层级协同工作的特征。 |

### B. 公式速查

符号约定：
- $N$ 为参数量
- $\Psi$ 在显存与通信公式中同样表示参数量
- $D$ 为训练 token 数
- $L$ 为层数
- $h$ 为隐藏维度
- $s$ 为序列长度
- $b$ 为 batch 大小
- $a$ 为注意力头数
- $p$ 在通信公式中为参与设备数、在 KV 与 MoE 公式中为每个元素的字节数
- $P_{\text{peak}}$ 为峰值算力
- $B$ 为主存带宽

#### B.1 计算量与算力利用率

| 量 | 公式 | 说明 |
|---|---|---|
| 矩阵乘 FLOPs | $2mkn$ | $(m\times k)$ 与 $(k\times n)$ 矩阵相乘，每个输出元素需要 $k$ 次乘法和 $k$ 次加法。 |
| 训练 FLOPs / token | $6N + 12Lhs$ | 前向 $2N$、反向 $4N$；$12Lhs$ 为注意力分数与加权求和的计算量。 |
| 推理 FLOPs / token | $2N + 4Lhs$ | 仅含前向；注意力项为近似值。 |
| 训练总 FLOPs | $\approx 6ND$ | 当 $12Lhs\ll 6N$（对标准 Transformer 约为 $s\ll 6h$）时忽略注意力项后的估算。 |
| MFU | $\frac{\text{tokens/s}\times(6N+12Lhs)}{N_{\text{GPU}}P_{\text{peak}}}$ | 分子按模型理论计算量计，不计激活重计算；大规模稠密训练的 MFU 通常在 35%–55%。 |

#### B.2 算术强度与 Roofline 模型

| 量 | 公式 | 说明 |
|---|---|---|
| 算术强度 | $I = \text{FLOPs}/\text{Bytes}$ | 分母为 kernel 与主存之间实际传输的字节数。 |
| Roofline | $P = \min(P_{\text{peak}}, I\cdot B)$ | kernel 可达到的性能上界由算力上限和带宽上限中较低者决定。 |
| 脊点 | $I^* = P_{\text{peak}}/B$ | H100 SXM 的 BF16 脊点约为 $989/3.35\approx 295$ FLOP/Byte；$I<I^*$ 时 kernel 受带宽限制。 |
| 方阵 GEMM 强度 | $M/3$ | $M\times M$ 的 BF16 方阵乘法，计算量 $2M^3$，访存 $3\times 2M^2$ 字节。 |
| decode GEMM 强度 | $\approx b$ | 权重读取一次被 $b$ 个 token 复用，强度与 batch 大小近似成正比。 |

#### B.3 显存占用

| 量 | 公式 | 说明 |
|---|---|---|
| 权重显存 | $M_w \approx N\cdot b_w/8$ 字节 | $b_w$ 为每个权重的位宽；未计入量化缩放因子等元数据。70B 模型在 BF16 下约 140 GB，在 4 bit 下约 35 GB。 |
| 模型状态（Adam 混合精度） | $16\Psi$ | BF16 参数与梯度各 $2\Psi$，FP32 主参数、一阶矩、二阶矩共 $12\Psi$。 |
| ZeRO-1 / 2 / 3 | $4\Psi + 12\Psi/N_d$ / $2\Psi + 14\Psi/N_d$ / $16\Psi/N_d$ | 依次切分优化器状态、再加梯度、再加参数；$N_d$ 为数据并行度。 |
| 每层激活 | $sbh(34 + 5as/h)$；FlashAttention 后 $34sbh$；TP+SP 后 $34sbh/t$ | $5as/h$ 项来自注意力分数矩阵，FlashAttention 不保存该矩阵；$t$ 为 TP 度。 |
| KV / token | $2Ln_{kv}d_hp$ | 系数 2 对应 K 与 V；$n_{kv}$ 为 KV 头数，$d_h$ 为每头维度。 |
| 推理总显存 | $Np_w + T_{\text{total}}\cdot 2Ln_{kv}d_hp + M_{\text{act}}$ | 依次为权重、全部在途 token 的 KV cache、激活与临时缓冲。 |

#### B.4 通信开销

| 量 | 公式 | 说明 |
|---|---|---|
| α-β 模型 | $T = \alpha + n/\beta$ | $\alpha$ 为单次传输的固定启动延迟，$\beta$ 为链路带宽，$n$ 为消息字节数。 |
| 数据搬运时间 | $T \approx T_{\text{setup}} + D/B$ | 对 DMA、RDMA、PCIe 拷贝等同样适用；数据量小时 $T_{\text{setup}}$ 主导，数据量大时 $D/B$ 主导。 |
| Ring AllReduce | $2(p-1)\alpha + \frac{2(p-1)}{p}\frac{M}{\beta}$ | ReduceScatter 与 AllGather 各 $p-1$ 步；带宽项与 $p$ 基本无关。 |
| ReduceScatter / AllGather | $\frac{p-1}{p}\frac{M}{\beta}$ | 每个设备发送和接收 $(p-1)/p$ 份数据。 |
| busbw（AllReduce） | $\text{algbw}\times\frac{2(p-1)}{p}$ | $\text{algbw}=M/T$；busbw 可直接与链路峰值带宽比较。 |
| TP 每层通信 | 前向 2 次 + 反向 2 次 AllReduce，每次 $2sbh$ 字节 | 注意力块与 MLP 块各在前向、反向中做一次 AllReduce。 |
| TP 通信计算比 | $\approx\frac{2tP}{9h\beta}$ | 比值随 $t$ 增大、随 $h$ 减小，因此 TP 一般限制在节点内。 |
| DP / ZeRO-1/2 / ZeRO-3 | $2\Psi$ / $2\Psi$ / $3\Psi$ | 每步每个设备的通信量（以参数元素计）；ZeRO-3 额外在前向、反向各做一次参数 AllGather。 |
| MoE 每层 dispatch | $T\cdot k\cdot h\cdot p$ | $T$ 为 token 数，$k$ 为每个 token 选中的专家数；combine 阶段通信量相同。 |
| Amdahl 定律 | $S = \frac{1}{(1-f) + f/s}$ | $f$ 为可加速部分占比，$s$ 为该部分的加速倍数；$f=0.9$ 时即使 $s\to\infty$，$S$ 也不超过 10。 |

#### B.5 流水线与可靠性

| 量 | 公式 | 说明 |
|---|---|---|
| PP 气泡 | $\frac{p-1}{m}$；interleaved $\frac{p-1}{vm}$ | $p$ 为流水线阶段数，$m$ 为 micro-batch 数，$v$ 为每个设备上的虚拟阶段数。 |
| 至少一处故障概率 | $P(\ge 1) = 1-(1-p_f)^N$ | $p_f$ 为单个部件在给定时间内的故障概率，假设各部件相互独立；$p_f=10^{-4}$、$N=10^4$ 时约为 63%。 |
| 作业 MTBF | $M_0/N$ | $M_0$ 为单个节点的 MTBF，$N$ 为节点数；在故障独立且服从指数分布时成立。 |
| 最优 checkpoint 间隔 | $\tau^* = \sqrt{2\delta M}$，浪费 $\sqrt{2\delta/M}$ | Young 公式；$\delta$ 为单次保存 checkpoint 的耗时，$M$ 为作业 MTBF。 |

#### B.6 推理性能与成本

| 量 | 公式 | 说明 |
|---|---|---|
| Prefill 时间 | $\frac{2Ns}{N_{\text{GPU}}P\eta}$ | prefill 受算力限制；$\eta$ 为实际算力利用率。 |
| Decode 步时间下限 | $\frac{Np_w + \sum s_i k_{\text{tok}}}{N_{\text{GPU}}B}$ | 每步需读完全部权重和 batch 内所有请求的 KV；$k_{\text{tok}}$ 为每个 token 的 KV 字节数。 |
| decode 进入算力受限的 batch | $b^*\approx I^* p_w/2$ | H100 上 BF16 权重对应 $b^*$ 约为 300。 |
| 投机解码期望产出 | $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ | $\alpha$ 为单个草稿 token 的接受率，$\gamma$ 为每轮草稿长度；结果为每轮目标模型前向产出的期望 token 数。 |
| 投机解码加速比 | $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$ | $c$ 为草稿模型与目标模型单次前向耗时之比。 |
| Little 定律 | $L = \lambda W$ | 系统内平均请求数等于到达率乘以平均停留时间。 |
| 每百万 token 成本 | $\frac{\text{卡时价}\times\text{卡数}}{\text{token/s}\times3600}\times10^6$ | token/s 为整个实例的总吞吐。 |

#### B.7 数值表示、注意力变体与物理约束

| 量 | 公式 | 说明 |
|---|---|---|
| 对称量化 | $q=\mathrm{clip}(\mathrm{round}(x/s),\,q_{\min},\,q_{\max})$，$\hat{x}=s\,q$ | $s$ 为缩放因子；INT8 对称量化常取 $s=\max\lvert x\rvert/127$。舍入引入误差，裁剪限制离群值。 |
| 线性注意力状态更新 | $S_t = S_{t-1} + \phi(k_t)v_t^\top$，$o_t = \phi(q_t)^\top S_t$ | $\phi$ 为特征映射；状态 $S_t$ 大小固定为 $d_k\times d_v$，每个 token 的计算量与序列长度无关，替代了随长度增长的 KV cache。带门控的变体写作 $S_t = G_t\odot S_{t-1} + \phi(k_t)v_t^\top$。 |
| 动态功耗 | $P_{\text{dyn}}\approx\alpha C V^2 f$ | $\alpha$ 为翻转活动因子，$C$ 为负载电容，$V$ 为电源电压，$f$ 为时钟频率；功耗与电压平方成正比。 |
| 焦耳热 | $P = I^2R$ | 供电网络与互连中电流 $I$ 流经电阻 $R$ 产生的热功率。 |
| 热传导（Fourier 定律） | $\mathbf{q}=-k\nabla T$；$\Delta T\approx P\cdot R_{\text{th}}$ | $k$ 为热导率，$R_{\text{th}}$ 为热阻（K/W）；1000 W 芯片在 $R_{\text{th}}=0.05$ K/W 时温升约 50 K。 |
| 热膨胀 | $\Delta L=\alpha_{\text{CTE}} L_0\Delta T$ | 硅的 CTE 约 2.6 ppm/K，有机基板约 15–17 ppm/K，二者之差在温度循环中产生翘曲和焊点应力。 |

### C. 参考资源

#### C.1 核心论文

| 主题 | 论文 |
|---|---|
| 性能模型 | Roofline (2009) |
| 精度 | Mixed Precision Training (2018)；FP8 Formats (2022) |
| Kernel | FlashAttention 1/2/3 (2022–2024) |
| 并行 | Megatron-LM (2019)；Efficient Large-Scale LM Training (2021)；Reducing Activation Recomputation (2022)；ZeRO (2020)；PyTorch FSDP (2023)；GPipe (2019)；PipeDream (2019)；Zero Bubble PP (2023) |
| 长序列 | Ring Attention (2023)；DeepSpeed-Ulysses (2023) |
| MoE | GShard (2020)；Switch Transformer (2021)；DeepSeekMoE (2024)；DeepSeek-V3 (2024) |
| 大规模训练 | MegaScale (2024)；Llama 3 (2024) §3.3 |
| 推理 | Orca (2022)；vLLM (2023)；SGLang (2023)；Sarathi-Serve (2024)；DistServe (2024)；Splitwise (2024)；Mooncake (2024) |
| 推理算法 | Speculative Decoding (2023)；Medusa (2024)；EAGLE (2024)；LLM.int8() (2022)；GPTQ (2023)；AWQ (2024)；SmoothQuant (2023)；KIVI (2024) |
| 编译 | PyTorch 2 (2024)；Triton (2019)；TVM (2018) |
| 确定性 | Defeating Nondeterminism in LLM Inference (2025)；Deterministic Inference across TP Sizes (2025) |

#### C.2 代码仓库

| 仓库 | 重点阅读内容 |
|---|---|
| NVIDIA/Megatron-LM | `megatron/core/tensor_parallel/`、`pipeline_parallel/schedules.py`、`transformer/moe/` |
| pytorch/torchtitan | FSDP2 + TP + PP + CP 组合的参考实现 |
| deepspeedai/DeepSpeed | `runtime/zero/` |
| Dao-AILab/flash-attention | `csrc/flash_attn/`，Hopper 部分 `hopper/` |
| triton-lang/triton | `python/tutorials/` |
| vllm-project/vllm | `vllm/v1/core/sched/`（调度器）、`vllm/v1/core/kv_cache_manager.py` |
| sgl-project/sglang | `srt/managers/scheduler.py`、radix cache |
| deepseek-ai/DeepEP、DeepGEMM、EPLB、3FS | MoE 通信、FP8 GEMM、专家均衡、分布式存储 |
| NVIDIA/nccl-tests | 通信基准 |
| karpathy/nanoGPT、llm.c | 单卡训练的最小参考实现 |

#### C.3 课程与博客

| 资源 | 内容 |
|---|---|
| Hugging Face Ultra-Scale Playbook | 并行训练全景 |
| JAX Scaling Book | 分片与 roofline 推导 |
| Stas Bekman, ML Engineering | 工程实践手册 |
| CMU 15-442 / 15-779 Machine Learning Systems | 机器学习系统课程 |
| Stanford CS336 Language Modeling from Scratch | 从零实现语言模型，含系统方向作业 |
| GPU MODE（社区与讲座） | GPU kernel 编程 |
| Horace He, *Making Deep Learning Go Brrrr From First Principles* | 计算、访存、开销三类瓶颈的分析框架 |

### D. 实践计划

#### D.1 实践项目与验证标准

| # | 项目 | 验证标准 |
|---|---|---|
| 1 | 测量所用 GPU 的实际带宽与算力，绘制 Roofline 图 | 1.8 节 |
| 2 | 用 Triton 实现融合的 add + RMSNorm，并与 PyTorch 实现对比 | 加速比与访存字节数的估算一致（2.7） |
| 3 | 先用 NumPy、再用 Triton 实现 FlashAttention 前向 | 输出与标准注意力实现逐元素比对一致 |
| 4 | 基于 `torch.distributed` 的 send/recv 实现 Ring AllReduce | 结果与 `all_reduce` 一致，并测量带宽 |
| 5 | 实现张量并行的 MLP 与 1F1B 流水线调度 | 结果与单卡实现逐位一致或在数值误差范围内一致 |
| 6 | 确定性训练测试：连续训练与中断后续训的结果逐位比对 | 4.10.5 |
| 7 | 实现支持 continuous batching 与分页 KV cache 的简化推理引擎 | 吞吐随并发数增加而提升 |
| 8 | 实现投机解码，以小模型作为大模型的草稿模型 | 实测接受率与加速比符合理论公式 |
| 9 | vLLM 负载测试与参数调优 | 5.14 |
| 10 | 编写一份完整的推理容量规划文档 | 5.13 |

### E. 检查清单

#### E.1 新集群验收

- [ ] `nvidia-smi topo -m` 显示的拓扑符合设计：GPU 之间为 NVLink（NV18 等）连接，每块 GPU 与其对应网卡位于同一 PCIe 交换机（PIX/PXB）和同一 NUMA 节点
- [ ] 每台机器完成单机 GEMM 与 HBM 带宽测试，结果达到同型号中位数的 95% 以上
  - 低于该阈值的节点先排查散热、时钟与功耗上限
  - 无法修复的从资源池剔除
- [ ] 节点内 `nccl-tests` 的 AllReduce 在大消息（如 1 GB 以上）下 busbw 达到 NVLink 单向峰值的 80% 左右，H100 节点通常约 360–480 GB/s（启用 NVLS 时更高）
- [ ] 跨节点 `nccl-tests` 在全规模下 busbw 接近单块网卡线速（400G 网卡约 45–50 GB/s）
  - 按 2、4、8…节点逐步扩大规模
  - 并对各节点对单独测试以定位故障链路或交换机端口
- [ ] 存储读写带宽测试达标：聚合写带宽应能在目标时间内（如 1 分钟以内）写完一次完整 checkpoint，读带宽满足数据加载与故障恢复需求
- [ ] 健康检查接入调度器：DCGM 诊断、XID 错误监控、ECC 可纠正与不可纠正错误计数、时钟与功耗上限检查，异常节点自动隔离
- [ ] 用小模型完成端到端多节点训练，loss 正常下降，MFU 与同规模参考值一致

#### E.2 大规模训练启动前

- [ ] 已按附录 B 的公式估算并行配置的每块 GPU 显存占用（预留 10%–20% 余量）与各维度通信计算比（3.9）
- [ ] 已在小规模上用相同配置训练数百至数千步，loss 曲线与参考实现或上一版本一致
- [ ] checkpoint 保存、恢复与续训一致性测试通过：续训后的 loss 与不中断训练逐位一致或在预期误差内（4.10.5）
- [ ] 异步 checkpoint 已启用，保存间隔按 $\tau^*=\sqrt{2\delta M}$ 设定，保留策略与存储配额已配置
- [ ] 故障自动恢复流程已完成演练：人为终止进程或下线节点后，作业能自动替换节点并从最近 checkpoint 恢复，恢复耗时已测量
- [ ] 日志记录每步的 loss、grad norm、样本 id 及各 rank 的步耗时
- [ ] 监控看板与告警已配置，覆盖 loss 异常、步耗时突增、MFU 下降与通信超时（9.6）
- [ ] manifest 已记录代码版本、配置与运行环境（4.5.3）

#### E.3 推理服务上线前

- [ ] 完成推理引擎与训练框架在同一输入上的 logits 一致性检查，差异应在所用精度的数值误差范围内
- [ ] 量化模型已按任务类型（对话、代码、数学、长上下文等）分别评测，与原精度模型的指标差距在可接受范围内
- [ ] 按真实请求的输入与输出长度分布完成负载测试，确定在满足 SLO 前提下单实例可承载的最大并发与吞吐
- [ ] 已针对 P99 TTFT、P99 TPOT 与错误率配置 SLO 告警，自动扩缩容策略以队列长度或 KV cache 占用率为触发指标
- [ ] 超长请求与异常请求的限流和路由策略已配置，避免单个长 prompt 的 prefill 阻塞同一实例上其他请求的 decode
- [ ] 灰度发布与回滚方案已就绪：新版本先承接小比例流量，关键指标异常时可在分钟级回滚

### F. 环境变量与命令速查

#### F.1 NCCL 环境变量

| 变量 | 作用 |
|---|---|
| `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,COLL` | 输出初始化与集合通信的调试日志 |
| `NCCL_ALGO=Ring/Tree` | 强制使用指定的集合通信算法 |
| `NCCL_PROTO=Simple/LL/LL128` | 强制使用指定的传输协议 |
| `NCCL_IB_HCA` | 指定 NCCL 使用的 IB 网卡 |
| `NCCL_SOCKET_IFNAME` | 指定 bootstrap 阶段使用的网络接口 |
| `NCCL_NVLS_ENABLE` | 启用 NVLink SHARP，即在 NVSwitch 上执行归约 |
| `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` | 故障排查时禁用对应的传输路径 |

#### F.2 PyTorch 环境变量与接口

| 变量 / 接口 | 作用 |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 启用可扩展内存段以减少显存碎片 |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 使 cuBLAS 结果确定 |
| `torch.use_deterministic_algorithms(True)` | 启用 PyTorch 确定性算法模式 |
| `TORCH_NCCL_TRACE_BUFFER_SIZE`、`TORCH_NCCL_DUMP_ON_TIMEOUT=1` | 配置 Flight Recorder，在通信超时时转储集合通信记录 |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | 通信超时或出错后终止进程 |
| `TORCH_LOGS=output_code`、`TORCH_COMPILE_DEBUG=1` | 输出 torch.compile 生成的代码与调试信息 |
| `CUDA_LAUNCH_BLOCKING=1` | 同步执行 kernel 以定位报错位置；会显著降低性能，仅用于调试 |

#### F.3 其他框架环境变量

| 变量 | 作用 |
|---|---|
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | 禁止 Transformer Engine 使用非确定性算法 |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | 在 Megatron 中保证 TP 通信与计算按发射顺序执行，以实现两者重叠 |

#### F.4 常用诊断与启动命令

```bash
nvidia-smi topo -m                             # 拓扑
nvidia-smi -q -d CLOCK,POWER,ECC,PERFORMANCE   # 时钟、功耗、ECC、降频原因
dcgmi diag -r 3                                # DCGM 诊断
ibstat; ibv_devinfo                            # IB 网卡状态
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$HEAD:29500 train.py
nsys profile -t cuda,nvtx,osrt -o report python train.py
ncu --set full -k regex:gemm -o report python bench.py
```

### G. 硬件参数速查

#### G.1 GPU 与 AI 加速器

| 型号 | BF16 稠密 (TFLOPS) | FP8 稠密 (TFLOPS) | 显存 | 带宽 | 片间互连单向带宽（每芯片） | TDP |
|---|---|---|---|---|---|---|
| A100 SXM 80G | 312 | — | 80 GB | 2.0 TB/s | 300 GB/s | 400 W |
| H100 SXM | 989 | 1979 | 80 GB | 3.35 TB/s | 450 GB/s | 700 W |
| H200 | 989 | 1979 | 141 GB | 4.8 TB/s | 450 GB/s | 700 W |
| B200 | ~2250 | ~4500 | 180–192 GB | 8 TB/s | 900 GB/s | ~1000 W |
| GB300（单 GPU） | ~2250 | ~4500 | 288 GB | 8 TB/s | 900 GB/s | ~1400 W |
| MI300X | ~1307 | ~2615 | 192 GB HBM3 | 5.3 TB/s | Infinity Fabric 448 GB/s | 750 W |
| TPU v5p | 459 | — | 95 GB HBM2e | 2.76 TB/s | ICI | — |
| TPU v7 Ironwood | — | ~4600 | 192 GB HBM3e | ~7.4 TB/s | ICI | — |
| 昇腾 910B | ~320（FP16） | — | 64 GB HBM2e | ~1.6 TB/s | HCCS | — |

#### G.2 网络与总线带宽

| 链路 | 单向带宽 |
|---|---|
| PCIe Gen4 x16 | ~32 GB/s |
| PCIe Gen5 x16 | ~64 GB/s |
| IB HDR 200G | 25 GB/s |
| IB NDR 400G | 50 GB/s |
| IB XDR 800G | 100 GB/s |
| NVLink 3 / 4 / 5（每 GPU） | 300 / 450 / 900 GB/s |

#### G.3 典型操作延迟

| 操作 | 延迟量级 |
|---|---|
| 寄存器 / SMEM 访问 | ns 级 |
| HBM 访问 | 数百 ns |
| kernel launch（CPU 侧） | 3–10 µs |
| NVLink 小消息 | 1–2 µs |
| IB 小消息 | 2–5 µs |
| 节点内 8 卡 AllReduce 小消息 | ~10 µs |
| 千卡 AllReduce 小消息 | 数十到数百 µs |
