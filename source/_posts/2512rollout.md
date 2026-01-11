---
title: RL中的 Rollout 与 Training
date: 2026-01-02
categories:
    - 学AI/DS
tags:
    - LLM
    - Agent
    - RL
desc: 经常听到rollout这个词， 周围人张口闭口就是。详细解释辨析一下。
---



在强化学习（Reinforcement Learning, RL）和基于强化学习的语言模型训练（如 RLHF）中，**rollout** 和 **training** 是两个核心且必须分离的阶段。理解这两个阶段的区别、作用及其分离的必要性，对于深入理解现代 RL 算法至关重要。



## Rollout

### 概念


**Rollout** 可以理解为"数据收集"阶段。这个术语源自经典强化学习，其含义是："**实际运行当前策略一次，观察它展现出什么样的行为**"。

在深度强化学习的语境中，rollout 指的是使用当前策略（policy）与环境交互，生成经验数据的过程。这些数据记录了策略在执行任务时的完整轨迹（trajectory），包括状态、动作、奖励等信息。

### 步骤

在大语言模型（LLM）的强化学习训练中，一次完整的 rollout 通常包括以下步骤：

1. **冻结当前策略**：将当前策略参数固定为 $π_{θ_{old}}$，确保在数据收集过程中策略保持不变。
2. **前向生成**：使用这个冻结的策略对一批提示（prompts）进行前向生成，产生模型输出序列。
3. **采样与记录**：对模型的输出序列进行采样，同时记录相关信息，包括：
   - 生成的 tokens
   - 每个 token 的对数概率（log-probability）
   - 奖励（rewards，可能来自奖励模型或人类评估）
4. **参数保持不变**：整个过程中不更新任何模型参数，只是收集数据。

### 数值示例

假设：

- 提示批次大小（prompt batch size）= 512
- 每个提示采样响应数 = 16

那么，**一次 rollout 将产生 512 × 16 = 8192 条模型生成的轨迹（trajectories）**。

这些轨迹不会立即用于更新模型，而是首先被存储为一个固定不变的数据集，供后续的 training 阶段使用。

<br>

## Training

### 概念

**Training** 是 rollout 之后的"真正训练"阶段。在这个阶段，模型参数会被实际更新，以优化策略性能。

### 步骤

在 training 阶段中：

1. **数据固定不变**：rollout 产生的数据是固定且不可变的。这些数据被视为来自旧策略 $π_{θ_{old}}$ 的"历史经验"。
2. **计算梯度**：使用各种目标函数（如 PPO、GRPO、DAPO 等）计算梯度。这些目标函数通常包含：
   - 策略改进项：鼓励高奖励动作
   - 策略约束项：限制新策略与旧策略的偏离（如 KL 散度约束）
3. **更新参数**：根据计算的梯度对模型参数进行更新，得到新策略 $π_{θ_{new}}$。

### 数值示例

假设：

- Mini-batch size = 512
- Rollout 总共生成了 8192 条样本

那么，可以进行 **8192 / 512 = 16 次梯度更新**。

这意味着**同一批 rollout 数据会被重复使用 16 次来更新模型参数**。这是离线策略学习（off-policy learning）的典型特征：用一批固定数据多次更新模型，提高数据利用效率。

### Warm-up

很多论文会提到"在前 20 个 rollout step 里做线性 warm-up"。这里需要特别注意：

- **20 个 rollout step** 指的是 **20 次完整的 rollout + training 循环**
- 而不是 20 次梯度更新

例如，如果每次 rollout 生成 8192 条数据，mini-batch size 为 512，那么一个完整的 rollout + training 循环包含：

- 1 次 rollout（数据收集）
- 16 次梯度更新（training）

因此，"20 个 rollout step" 实际上相当于 20 次 rollout 和 320 次梯度更新（20 × 16）。

<br>

## 分开原因

将 rollout 和 training 分离是现代强化学习算法（特别是 PPO 等策略梯度方法）的核心设计原则。这种分离有两个重要原因：

### 稳定性

#### 理论基础

PPO、GRPO 等算法基于**重要性采样（Importance Sampling）** 和 **信任区域（Trust Region）** 理论。这些方法的核心假设是：

- 数据来自一个**固定的旧策略** $π_{θ_{old}}$
- 新策略 $π_{θ_{new}}$ 被限制在旧策略的"信任区域"内，不能偏离太远


为什么不能同时进行？
如果一边采样（rollout）一边更新模型（training），会导致：

1. **重要性采样比失效**：重要性采样比 $r(θ) = \frac{π_{θ_{new}}(a|s)}{π_{θ_{old}}(a|s)}$ 的计算依赖于"旧策略"是固定的。如果策略在采样过程中不断变化，这个比值将失去意义。

2. **信任区域失效**：信任区域方法（如 PPO 的 clipped objective）依赖于新旧策略的 KL 散度约束。如果策略在采样过程中更新，新旧策略的定义变得模糊，约束将失效。

3. **训练不稳定**：策略的频繁变化会导致目标函数变化剧烈，梯度估计不准确，训练过程可能发散。

#### 解决方案

通过先完整执行 rollout（冻结策略），收集固定数据，再执行 training（更新策略），可以确保：

- 所有训练数据都来自同一个固定策略 $π_{θ_{old}}$
- 目标函数和梯度估计更加稳定和准确

<br>

### 效率

#### 计算成本

在 LLM 的强化学习训练中，不同操作的计算成本差异巨大：

1. **生成文本（Rollout）非常耗时**：
   - 需要完整的前向传播生成整个序列
   - 对于长序列，计算量随序列长度线性增长
   - 通常需要 GPU 的完整计算资源
   - 例如：生成 8192 条轨迹可能需要数分钟到数十分钟

2. **反向传播（Training）相对便宜**：
   - 虽然也需要计算梯度，但可以利用已生成的数据
   - 可以批量处理，GPU 利用率高
   - 一次梯度更新的时间通常远小于生成相同数量数据的时间

#### 效率提升策略

##### 分离 rollout 和 training

通过分离 rollout 和 training，可以实现：

- **一次昂贵的采样**（rollout）：生成 8192 条轨迹
- **多次参数更新**（training）：用这 8192 条数据更新 16 次（或更多次）

这种设计相当于"**用一次数据收集的成本，换取多次模型改进的机会**"，整体效率显著提高。

##### 数据复用机制

在 training 阶段，同一批 rollout 数据会被重复使用多次：

- 每次 mini-batch 训练后，模型参数更新
- 虽然数据相同，但模型已经改变，因此每次训练实际上是在用"新模型"学习"旧数据"
- 这种机制允许模型从同一批数据中提取更多信息，提高数据利用效率

<br>


## 对比

### 理解

>[!note]+ 类比：学生做题与总结反思
> 
>
> **Rollout（模型去"做题"）**：
> - 模型就像学生，拿着当前的"解题方法"（策略 $π_{θ_{old}}$）
> - 出去做一批题目（处理一批 prompts）
> - 把自己的解题过程完整记录下来（生成轨迹，记录 tokens、log-probs、rewards）
> - 这个过程中，学生的"解题方法"保持不变，只是执行和记录
>
> **Training（模型批改分析"总结反思"）**：
> 
> - 模型来到"office hour"（训练环境）
> - 拿着这些解题记录（rollout 产生的固定数据）
> - 认真分析哪里做得好、哪里做得不好（计算梯度，评估策略性能）
> - 根据分析结果调整自己的解题方法（更新模型参数，得到新策略 $π_{θ_{new}}$）
> 
> **关键点**：
> 
> - 这两个阶段节奏不同：rollout 是"执行"，training 是"反思"
> - 但缺一不可：没有 rollout 就没有数据，没有 training 就没有改进
> - 必须分开进行：不能一边做题一边改方法，否则会混乱

<br>

### 关键区别

| 特性 | Rollout | Training |
| :--- | :------ | :------- |
| **目的** | 数据收集 | 参数更新 |
| **策略状态** | 冻结（$π_{θ_{old}}$） | 更新（$π_θ$） |
| **计算成本** | 高（生成文本） | 相对低（反向传播） |
| **数据状态** | 生成新数据 | 使用固定数据 |
| **执行次数** | 一次完整 rollout | 多次梯度更新 |

<br>

### 实际应用

在 RLHF（Reinforcement Learning from Human Feedback）等实际应用中，理解 rollout 和 training 的分离至关重要：

- **数据收集效率**：合理设计 rollout 批次大小和采样数量
- **训练稳定性**：确保 rollout 数据的一致性
- **计算资源分配**：平衡 rollout 和 training 的计算时间
- **超参数调优**：理解 rollout step、gradient step、mini-batch size 的关系


<br>

### 常见误区

1. **误区一**：认为 rollout 和 training 可以交替进行
   - **错误**：在 rollout 过程中更新模型
   - **正确**：必须完整执行 rollout，再完整执行 training

2. **误区二**：混淆 rollout step 和 gradient step
   - **错误**：认为"20 个 rollout step"是 20 次梯度更新
   - **正确**："20 个 rollout step"是 20 次完整的 rollout + training 循环

3. **误区三**：认为 rollout 数据只能用一次
   - **错误**：每次 rollout 的数据只能训练一次
   - **正确**：同一批 rollout 数据通常会被用于多次梯度更新（如 16 次）

<br>



## 技术细节

### PPO

在 PPO（Proximal Policy Optimization）算法中，rollout 和 training 的分离更加明显：

**PPO 的目标函数**：
$$L^{CLIP}(θ) = \mathbb{E}_t[\min(r_t(θ)\hat{A}_t, \text{clip}(r_t(θ), 1-ε, 1+ε)\hat{A}_t)]$$

其中：

- $r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_{old}}(a_t|s_t)}$ 是重要性采样比
- $\hat{A}_t$ 是优势估计
- $ε$ 是裁剪参数（通常为 0.1 或 0.2）

**关键点**：

- $π_{θ_{old}}$ 必须在 rollout 阶段固定
- 在 training 阶段，$π_θ$ 会更新，但 $π_{θ_{old}}$ 保持不变
- 只有当 $r_t(θ)$ 接近 1 时（新旧策略相似），重要性采样才有效

<br>

### GRPO 与 DAPO

**GRPO（Group Relative Policy Optimization）**：

- 专门为 LLM 设计，使用组内相对比较而非绝对奖励
- 同样需要 rollout 和 training 分离
- 在 rollout 阶段生成多个响应，在 training 阶段基于组内排名更新策略

**DAPO（Direct Alignment from Preference Optimization）**：

- 直接优化偏好，无需显式奖励模型
- Rollout 阶段生成候选响应
- Training 阶段基于偏好信号更新模型

<br>

### 实际训练流程

一个完整的训练循环通常如下：

```python
for epoch in range(num_epochs):
    # ========== Rollout 阶段 ==========
    # 1. 冻结当前策略
    policy_old = copy.deepcopy(policy)
    
    # 2. 生成数据
    trajectories = []
    for batch in prompt_batches:
        responses = policy_old.generate(batch, num_samples=16)
        rewards = reward_model.evaluate(responses)
        trajectories.extend(collect_trajectories(responses, rewards))
    
    # 3. 存储固定数据集
    dataset = TrajectoryDataset(trajectories)
    
    # ========== Training 阶段 ==========
    # 4. 使用固定数据多次更新
    for update_step in range(num_updates):
        batch = dataset.sample(mini_batch_size)
        loss = compute_ppo_loss(batch, policy, policy_old)
        optimizer.step(loss)
    
    # 5. 更新旧策略（为下一轮 rollout 准备）
    policy_old = copy.deepcopy(policy)
```



<br>