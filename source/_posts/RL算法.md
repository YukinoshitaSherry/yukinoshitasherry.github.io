---
title: RL 算法详解：从 Q-learning 到 GRPO 系列
date: 2026-03-07
categories:
  - 学AI/DS
tags:
  - RL
  - LLM
desc: 系统梳理强化学习算法：从定义、马尔可夫、Q-learning 起步，经 PPO、VAPO、DPO，到 GRPO、GMPO、GSPO、GFPO、GAPO、Dr.GRPO，含详细数学推导、伪代码、优缺点与继承关系。
---

参考：
- <a href="https://datawhalechina.github.io/easy-rl/#/">蘑菇书</a>
-  <a href="http://xhslink.com/o/7jGW68bHvfl ">小红书@古希腊掌管代码的神 大模型RL总结：PPO、DPO、GRPO、DAPO、GSPO</a>


论文与技术报告：
- [PPO - Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [DPO 原论文](https://arxiv.org/abs/2305.18290)
- [DeepSeek-Math 技术报告](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 技术报告](https://arxiv.org/abs/2501.12948)
- [VAPO 论文](https://arxiv.org/abs/2504.05118)
- [GMPO 论文](https://arxiv.org/abs/2507.20673)
- [From GRPO to DAPO and GSPO](https://huggingface.co/blog/NormalUhr/grpo-to-dapo-and-gspo)
- [GFPO 论文](https://arxiv.org/abs/2508.09726)
- [GAPO 论文](https://arxiv.org/abs/2503.20194)
- [DAPO 论文](https://arxiv.org/abs/2503.14476)
- [Phi-4-reasoning 技术报告](https://arxiv.org/abs/2504.21318)
- [Qwen2.5-Math 技术报告](https://arxiv.org/abs/2409.12122)

<br>


## 强化学习基础

### 定义与动机

**强化学习（Reinforcement Learning, RL）**：智能体（Agent）通过与环境的交互，根据获得的**奖励信号**学习如何在给定状态下选择动作，以最大化**累积奖励**。

| 概念 | 定义 |
| :--- | :--- |
| **Agent** | 做决策的主体，输出动作 $a$ |
| **Environment** | 接收动作、返回状态与奖励的外部系统 |
| **State $s$** | 环境的当前描述，Agent 据其决策 |
| **Action $a$** | Agent 的可选行为 |
| **Reward $r$** | 环境对动作的即时反馈标量 |
| **Policy $\pi(a\|s)$** | 状态到动作的概率分布，即策略 |

**动机**：监督学习需要大量标注；RL 只需稀疏的奖励信号，通过试错学习，适用于游戏、机器人控制、LLM 对齐等难以提供逐样本标签的场景。

> 本文 PPO、DPO、GRPO 等算法在 LLM 对齐中有广泛应用。训练流程与阶段划分见 [Pre-Training、SFT 与 RLHF](Pretrain-SFT-RLHF.md)，SFT 原理与实践见 [SFT 教程：原理与实践](SFT教程-原理与实践.md)。

<br>

### 马尔可夫性与 MDP

**马尔可夫性**：下一状态只依赖当前状态，与更早历史无关：

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

> [!INFO]+ 推导：马尔可夫性的意义
> 若满足马尔可夫性，则「当前状态」携带了做最优决策所需的全部信息，无需记忆完整历史。这使价值函数可递归定义，从而导出贝尔曼方程。
>
> **直观例子**：下棋时，只需知道当前棋盘布局即可决策，不必记住每一步如何走到这里；若不满足马尔可夫性（如部分可观测），则需隐状态或历史窗口。

<br>

**马尔可夫决策过程（MDP）** 由五元组 $(S, A, P, R, \gamma)$ 定义：

| 符号 | 含义 |
| :--- | :--- |
| $S$ | 状态空间 |
| $A$ | 动作空间 |
| $P(s'|s,a)$ | 状态转移概率 |
| $R(s,a,s')$ 或 $r(s,a)$ | 奖励函数 |
| $\gamma \in [0,1]$ | 折扣因子，权衡即时与未来奖励 |

**回报（Return）**：从 $t$ 时刻起的累积折扣奖励

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

$\gamma$ 接近 0 则更重视短期；接近 1 则更重视长期。

<br>

### 价值函数与贝尔曼方程

**状态价值函数** $V^\pi(s)$：从状态 $s$ 出发，按策略 $\pi$ 的期望回报

$$V^\pi(s) = \mathbb{E}\_{\pi}[G_t \mid s_t = s]$$

**动作价值函数（Q 函数）** $Q^\pi(s,a)$：在 $s$ 选择 $a$，之后按 $\pi$ 的期望回报

$$Q^\pi(s,a) = \mathbb{E}\_{\pi}[G_t \mid s_t = s, a_t = a]$$

> [!INFO]+ 推导：贝尔曼期望方程
> **第一步**：回报可拆分为即时奖励与折扣后的未来回报
> $$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = r_t + \gamma G_{t+1}$$
>
> **第二步**：对 $V^\pi(s) = \mathbb{E}\_{\pi}[G_t \mid s_t = s]$ 代入上式，由期望的线性性：
> $$V^\pi(s) = \mathbb{E}\_{\pi}[r_t + \gamma G_{t+1} \mid s_t = s] = \mathbb{E}\_{\pi}[r_t \mid s_t = s] + \gamma \mathbb{E}\_{\pi}[G_{t+1} \mid s_t = s]$$
>
> **第三步**：对 $a \sim \pi(\cdot|s)$、$s' \sim P(\cdot|s,a)$ 取期望，且 $\mathbb{E}\_{\pi}[G_{t+1} \mid s_t=s] = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) V^\pi(s')$（因为 $G_{t+1}$ 从 $s'$ 起由 $\pi$ 生成）：
> $$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \big[ R(s,a,s') + \gamma V^\pi(s') \big]$$
>
> **第四步**：$Q^\pi$ 的贝尔曼方程同理，注意下一时刻动作由 $\pi(a'|s')$ 采样：
> $$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \Big[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a') \Big]$$

<br>

**最优价值函数**：

$$V^{\ast}(s) = \max_{\pi} V^\pi(s), \quad Q^{\ast}(s,a) = \max_{\pi} Q^\pi(s,a)$$

> [!INFO]+ 推导：贝尔曼最优性方程
> **最优策略的刻画**：最优策略 $\pi^{\ast}$ 必然在每个状态 $s$ 选择使 $Q^{\ast}(s,a)$ 最大的动作（贪心），故：
> $$V^{\ast}(s) = \max_a Q^{\ast}(s,a)$$
>
> **$Q^{\ast}$ 的递归**：在 $s$ 选 $a$ 后，转移到 $s'$ 并获得 $R(s,a,s')$，从 $s'$ 起的最优回报为 $V^{\ast}(s') = \max_{a'} Q^{\ast}(s',a')$，故：
> $$Q^{\ast}(s,a) = \sum_{s'} P(s'|s,a) \Big[ R(s,a,s') + \gamma V^{\ast}(s') \Big] = \sum_{s'} P(s'|s,a) \Big[ R(s,a,s') + \gamma \max_{a'} Q^{\ast}(s',a') \Big]$$
>
> 这是 Q-learning 等算法的理论基石：$Q^{\ast}$ 是上述算子的不动点，可用迭代更新逼近。

<br>

---

## Q-learning

*Q-learning（经典 TD 控制算法）*

### 定义与动机

**Q-learning**：无模型、离策略的 TD 控制算法，直接学习最优动作价值函数 $Q^{\ast}(s,a)$，无需显式估计环境转移 $P$。

**动机**：贝尔曼最优性方程给出 $Q^{\ast}$ 的固定点；可用**时序差分（TD）**在线更新，每步用当前估计改进自身。

<br>

### 流程概述

1. 初始化 Q 表（或 Q 网络）为 0。
2. 每个 episode：从初始状态 $s$ 开始，循环直到终止。
3. 每步：用 $\varepsilon$-greedy 选动作 $a$，执行后得 $r, s'$，用 TD 更新 $Q(s,a)$，然后 $s \leftarrow s'$。
4. 重复 2–3，直到收敛。

**核心循环**：采样一步 $(s,a,r,s')$ → 计算 TD 目标 $y = r + \gamma \max_{a'} Q(s',a')$ → 更新 $Q(s,a) \leftarrow Q(s,a) + \alpha(y - Q(s,a))$。

<br>

### 原理简述

Q-learning 直接学习**最优 Q 函数** $Q^{\ast}$，不显式学策略；策略由「每步选 $\arg\max_a Q(s,a)$」隐式给出。利用贝尔曼最优性方程的**不动点**：$Q^{\ast}$ 满足 $Q^{\ast}(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^{\ast}(s',a')]$。用一步采样做 TD 近似，每次更新把当前 $Q$ 向该目标拉近，迭代收敛到 $Q^{\ast}$。

<br>

### 数学推导

> [!INFO]+ 推导：Q-learning 更新规则
> **贝尔曼最优性方程**：$Q^{\ast}(s,a) = \mathbb{E}\_{s'}\big[ r + \gamma \max_{a'} Q^{\ast}(s',a') \big]$
>
> **TD 思想**：不用完整回报 $G_t$，只用一步采样 $(s,a,r,s')$ 估计。**TD 目标**为：
> $$y = r + \gamma \max_{a'} Q(s', a')$$
> 即用当前 $Q$ 对下一状态的估计替代 $V^{\ast}(s')$。
>
> **TD 误差**：$\delta = y - Q(s,a)$，表示当前估计与一步自举目标的偏差。
>
> **更新规则**：沿减小 $\delta^2$ 的梯度方向，得到：
> $$Q(s,a) \leftarrow Q(s,a) + \alpha \big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \big]$$
>
> **收敛性**：当 $\alpha$ 满足 Robbins-Monro 条件（$\sum \alpha = \infty$，$\sum \alpha^2 < \infty$）且每个 $(s,a)$ 被无限次访问时，Q-learning 以概率 1 收敛到 $Q^{\ast}$。离策略性来自：行为策略可用 $\varepsilon$-greedy 探索，而更新目标总是 $\max$（贪心）。

<br>

### 伪代码

```text
算法：Q-learning
输入：环境 env，学习率 α，折扣因子 γ，探索率 ε
初始化：Q(s,a) = 0 对所有 s,a

for episode = 1, 2, ... do
    s = env.reset()
    while s 非终止 do
        a = ε-greedy(Q, s)
        执行 a，得到 r, s'
        Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') - Q(s,a) ]
        s ← s'
    end while
end for
```

<br>

> [!EXAMPLE]+ 例1：Q-learning 单步更新
> 状态 $s$，动作 $a$，得奖 $r=1$，转移至 $s'$。设 $Q(s,a)=0.5$，$Q(s',a_1)=0.3$，$Q(s',a_2)=0.8$，$\alpha=0.1$，$\gamma=0.99$。
>
> TD 目标：$y = r + \gamma \max_{a'} Q(s',a') = 1 + 0.99 \times 0.8 = 1.792$
> 更新：$Q(s,a) \leftarrow 0.5 + 0.1 \times (1.792 - 0.5) = 0.6292$

> [!EXAMPLE]+ 例2：ε-greedy 探索
> 设 $\varepsilon=0.1$，状态 $s$ 有 3 个动作，$Q(s,a_1)=0.2$，$Q(s,a_2)=0.9$，$Q(s,a_3)=0.5$。以 0.1 概率随机选动作，0.9 概率选 $\arg\max_a Q(s,a)=a_2$。即 90%  exploitation、10% exploration，避免过早收敛到次优。

<br>

### 优缺点

| 优点 | 缺点 |
| :--- | :--- |
| 无模型，不需 $P(s'\|s,a)$ | 仅适用于离散、低维状态/动作 |
| 离策略，可用经验回放 | 表格法无法泛化到未见过状态 |
| 理论保证收敛到最优 | 需大量采样；探索-利用需调参 |
| 实现简单 | 连续动作需函数逼近（如 DQN） |

<br>

---

## PPO

*Proximal Policy Optimization（近端策略优化）*

> RLHF 中 PPO 的流程与角色见 [Pre-Training、SFT 与 RLHF](Pretrain-SFT-RLHF.md)。

### 定义与动机

**PPO**：策略梯度算法，通过**裁剪（clip）**限制策略更新幅度，在实现简洁与训练稳定之间取得平衡。

> [!INFO]+ 概念：clip（裁剪）详解
>
> **定义**：clip 即把数值**夹紧**到指定区间 $[a, b]$。数学上 $\mathrm{clip}(x, a, b) = \min(\max(x, a), b)$：若 $x < a$ 取 $a$，若 $x > b$ 取 $b$，否则取 $x$。
>
> **在 PPO 中的角色**：PPO 对重要性采样比 $r_t = \pi_\theta(a_t|s_t) / \pi_{\mathrm{old}}(a_t|s_t)$ 做 clip，限制在 $[1-\epsilon, 1+\epsilon]$（常取 $\epsilon=0.2$）。目的是防止 $r_t$ 过大导致策略更新过大、训练崩塌。
>
> **直观例子**：$r_t=2$ 表示新策略对该动作概率是旧策略的 2 倍，梯度会被放大；clip 到 1.2 后，目标贡献被「截断」，更新更温和。
>
> **应用**：GRPO、GMPO、DAPO、GSPO 等均沿用 clip 思想，对概率比或序列级权重做裁剪，是 PPO 系算法的核心稳定机制。

<br>

**动机**：策略梯度若更新过大，易导致性能崩溃；TRPO 用复杂二阶约束保证稳定，PPO 用一阶裁剪近似实现类似效果，更易实现。

<br>

### RLHF 中的四个模型

用于 LLM 对齐时，PPO 需要**四个模型**协同工作：

| 模型 | 符号 | 作用 | 是否更新 |
| :--- | :--- | :--- | :--- |
| **策略模型** | $\pi_\theta$ | Actor，生成回答 | 是 |
| **参考模型** | $\pi_{\mathrm{ref}}$ | 提供 KL 约束基准，防止偏离 SFT | 否（冻结） |
| **奖励模型** | $R(q,o)$ | 对 $(q,o)$ 打分 | 否（冻结） |
| **价值模型** | $V_\phi$ | Critic，估计状态价值，用于 GAE | 是 |

**数据流**：prompt $q$ → $\pi_\theta$ 采样回答 $o$ → $R(q,o)$ 得奖励 → $V_\phi$ 估计各步价值 → 计算优势 $\hat{A}_t$ → 用 clip 目标更新 $\pi_\theta$，用 MSE 更新 $V_\phi$。KL 项 $D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$ 约束策略不要离参考太远。

<br>

### 流程概述

1. **采样**：对每个 prompt $q$，用当前 $\pi_\theta$ 采样回答 $o$。
2. **打分**：用 $R(q,o)$ 得奖励，减去 $\beta \cdot \mathrm{KL}$ 得有效奖励；用 $V_\phi$ 估计各步价值，GAE 算优势 $\hat{A}_t$。
3. **策略更新**：对多轮 epoch，用 minibatch 计算 $L^{\mathrm{CLIP}}$、$L^{\mathrm{VF}}$、熵项，更新 $\theta$；KL 超 target 则提前停。
4. **价值更新**：用 MSE 更新 $V_\phi$。
5. 重复 1–4。

<br>

### 原理简述

PPO 基于**策略梯度**，用**重要性采样**复用旧策略数据；但比率 $r_t$ 过大会导致更新过大。**clip** 把 $r_t$ 限制在 $[1-\epsilon, 1+\epsilon]$，实现「近端」约束，避免策略突变。GAE 用多步 TD 估计优势，在偏差与方差间折中；KL 约束防止偏离 SFT 参考。

<br>

### 策略梯度基础

策略参数化为 $\pi_\theta(a|s)$，目标为最大化期望回报：

$$J(\theta) = \mathbb{E}\_{\tau \sim \pi_\theta}\Big[ \sum_t \gamma^t r_t \Big]$$

> [!INFO]+ 推导：策略梯度定理
> **目标**：$J(\theta) = \mathbb{E}\_{\tau \sim \pi_\theta}[R(\tau)]$，其中 $\tau = (s_0, a_0, r_0, s_1, \ldots)$，$R(\tau) = \sum_t \gamma^t r_t$。
>
> **关键恒等式**：$\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau)$，故
> $$\nabla_\theta J(\theta) = \mathbb{E}\_{\tau \sim \pi_\theta}\Big[ R(\tau) \nabla_\theta \log \pi_\theta(\tau) \Big]$$
>
> **分解**：$\log \pi_\theta(\tau) = \sum_t \log \pi_\theta(a_t|s_t) + \sum_t \log P(s_{t+1}|s_t,a_t)$，转移 $P$ 与 $\theta$ 无关，故 $\nabla_\theta \log \pi_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$。代入并利用因果性（$t$ 时刻梯度只依赖 $t$ 之后奖励）：
> $$\nabla_\theta J(\theta) = \mathbb{E}\_{\tau}\Big[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \sum_{k \geq t} \gamma^{k-t} r_k \Big]$$
>
> **引入优势**：$\sum_{k \geq t} \gamma^{k-t} r_k$ 的期望即 $Q^\pi(s_t,a_t)$，减去基线 $V^\pi(s_t)$ 得 $A^\pi(s_t,a_t)$，可减少方差：
> $$\nabla_\theta J(\theta) = \mathbb{E}\_{s,a}\Big[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s,a) \Big]$$

<br>

### 重要性采样与 PPO-Clip

旧策略 $\pi_{\theta_{old}}$ 采样的数据用于更新新策略 $\pi_\theta$，需**重要性采样**：

$$\mathbb{E}\_{a \sim \pi_{\theta_{\mathrm{old}}}}[f(a)] = \mathbb{E}\_{a \sim \pi_\theta}\left[ \frac{\pi_{\theta_{\mathrm{old}}}(a|s)}{\pi_\theta(a|s)} f(a) \right]$$

记比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，原始目标为：

$$L^{PG}(\theta) = \mathbb{E}_t\big[ r_t(\theta) \hat{A}_t \big]$$

> [!INFO]+ 推导：PPO-Clip 目标
> **问题**：重要性采样比 $r_t(\theta)$ 过大时，策略更新过大，易导致性能崩塌。
>
> **解决**：对 $r_t$ 做 clip，限制在 $[1-\epsilon, 1+\epsilon]$（常用 $\epsilon=0.2$）：
> $$L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\Big[ \min\big( r_t(\theta) \hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \big) \Big]$$
>
> **直观**：$\hat{A}_t > 0$（好动作）时，若 $r_t > 1+\epsilon$ 说明新策略对该动作概率增加过多，clip 到 $1+\epsilon$ 限制更新；$\hat{A}_t < 0$（坏动作）时同理。取 $\min$ 实现悲观更新。
>
> [!EXAMPLE]+ 例：PPO-Clip 的 clip 效果
> 设 $\hat{A}_t = 0.5$，$\epsilon=0.2$。若 $r_t=1.5$，clip 后为 $\min(1.5, 1.2)=1.2$，目标贡献 $1.2 \times 0.5 = 0.6$ 而非 $0.75$，抑制了过大更新。

<br>

### 完整目标与 GAE

PPO 常用目标（含价值损失与熵正则）：

$$L^{PPO}(\theta) = \mathbb{E}_t\Big[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 H[\pi_\theta](s_t) \Big]$$

**广义优势估计（GAE）**：
$$\hat{A}_t^{\mathrm{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$\lambda \in [0,1]$ 控制偏差-方差权衡：$\lambda=0$ 只用单步 TD，$\lambda=1$ 等价于蒙特卡洛，常用 $\lambda=0.95$。

<br>

### 伪代码

```text
算法：PPO（RLHF 语境）
输入：策略模型 π_θ，参考模型 π_ref（冻结），奖励模型 R（冻结），价值模型 V_φ
超参：clip_ratio ε，γ，λ（GAE），β_KL

for iteration = 1, 2, ... do
    对每个 prompt q：采样回答 o ~ π_θ(·|q)
    计算 r = R(q,o)，KL = D_KL(π_θ || π_ref)
    用 GAE 计算优势 A_t（基于 r - β·KL 与 V）
    for epoch = 1 to K do
        对 minibatch 计算 L_CLIP、L_VF、熵项
        更新 θ 最大化 L_PPO
        若 mean(KL) > target_kl 则提前停止
    end for
    更新价值网络 V_φ
end for
```

<br>

### 优缺点

| 优点 | 缺点 |
| :--- | :--- |
| 实现简单，无需二阶优化 | 需单独的价值模型，显存占用大 |
| 训练稳定，clip 有效限制更新 | 长序列时 GAE 方差大 |
| 可重复使用数据（多轮 epoch） | 工程调参多（clip、KL、GAE λ） |
| RLHF 中应用最广 | 对 LLM 长输出扩展性差 |

<br>

---

## VAPO

*Value-based Augmented PPO（基于价值的增强 PPO）*

### 定义与动机

**VAPO**：面向推理模型的**价值型**强化学习框架，在 PPO 基础上系统缓解价值模型偏差、异质序列长度、稀疏奖励三大问题。基模 Qwen 32B，AIME 2024 达 60.4 分，优于 DeepSeek-R1-Zero、DAPO 10+ 分，5000 步内收敛且无训练崩溃。

**动机**：GRPO/DAPO 等 value-free 方法无法做细粒度信用分配；长 CoT 推理中单步错误可导致全链失败，需价值模型追踪每步对回报的贡献。VAPO 保留 value-based 范式，针对性设计解决其固有难点。

<br>

### 流程概述

与 PPO 相同的大框架：策略采样 → RM 打分 → 价值模型估计 → GAE 算优势 → clip 更新。区别在**数据与目标的设计**：对价值偏差做增强、对异质长度做归一化、对稀疏奖励做塑形，使每步优势估计更可靠，从而在长 CoT 上稳定收敛。

<br>

### 原理简述

VAPO 的核心是**改进 value-based 的可靠性**，而不是去掉价值模型。通过缓解 V 的偏差、长度主导、稀疏信号三大问题，使 PPO+GAE 在推理任务上可用；细粒度信用分配让模型知道「错在哪一步」，而非仅知道整句好坏。

<br>

### 核心设计

1. **Value model bias**：通过数据增强、正则化减轻价值模型估计偏差，避免 V 过高/过低导致的优势估计失真。
2. **Heterogeneous sequence lengths**：对异质长度做标准化/归一化，避免长序列主导梯度；否则短链与长链混训时，长链的梯度会压倒短链。
3. **Sparse rewards**：结合密集辅助信号或回报塑形（如中间步骤正确性），缓解稀疏奖励下的学习困难；纯端到端奖励在长 CoT 中难以提供有效学习信号。

### 与 PPO 的关系

VAPO 仍使用 PPO 的 clip 目标与 GAE，区别在于针对推理任务的三大痛点做了专门设计，使 value-based 方法在长链推理上可行。

<br>

### 优缺点

| 优点 | 缺点 |
| :--- | :--- |
| 细粒度信用分配，适合长 CoT | 需价值模型，显存高于 value-free |
| 训练稳定，多轮无崩溃 | 理论上有价值表示能力瓶颈 |
| 推理 SOTA（AIME 等） | 稀疏奖励仍需辅助设计 |

<br>

---

## DPO

*Direct Preference Optimization（直接偏好优化）*

> DPO 在 RLHF 流程中的定位及与 PPO 的对比见 [Pre-Training、SFT 与 RLHF](Pretrain-SFT-RLHF.md)。

### 定义与动机

**DPO**：直接从偏好数据 $(x, y_w, y_l)$ 优化策略，**无需**训练奖励模型或运行强化学习。将 RLHF 的「RM + RL」流程压缩为单一监督式目标。

**动机**：RLHF 需先训 RM、再训策略，工程复杂；DPO 证明在 Bradley-Terry 偏好模型下，最优策略可写成参考策略与隐式奖励的闭式，从而用分类式损失直接优化。只需一个策略模型 + 参考模型，显存友好。

<br>

### 流程概述

1. 从偏好数据 $D = \{(x, y_w, y_l)\}$ 采样。
2. 计算 $r_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}$，$r_l$ 同理。
3. 损失 $L = -\log \sigma(\beta(r_w - r_l))$，梯度下降更新 $\theta$。
4. 重复 1–3，无 RM、无 RL 循环、无采样生成。

<br>

### 原理简述

DPO 把 RLHF 的「RM + RL」**压缩为监督学习**。在 Bradley-Terry 假设下，最优策略满足 $\pi^{\ast} \propto \pi_{\mathrm{ref}} \exp(r^{\ast}/\beta)$，可解出隐式奖励 $r^{\ast} = \beta \log(\pi^{\ast}/\pi_{\mathrm{ref}}) + c$。把 $\pi_\theta$ 当作 $\pi^{\ast}$ 的估计，则偏好概率 $P(y_w \succ y_l|x)$ 可写成 $\sigma(\beta(r_w - r_l))$，最大化该似然即直接优化策略，无需显式 RM。

<br>

### 数学推导

> [!INFO]+ 推导：Bradley-Terry 与 DPO 损失
> 假设 $P(y_w \succ y_l \mid x) = \sigma(r^{\ast}(y_w) - r^{\ast}(y_l))$，RLHF 最优策略 $\pi^{\ast} \propto \pi_{\mathrm{ref}} \exp(r^{\ast}/\beta)$。
>
> 可解出 $r^{\ast}(x,y) = \beta \log \frac{\pi^{\ast}(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)$，代入得：
> $$P_\theta(y_w \succ y_l \mid x) = \sigma\big( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \big)$$
>
> 定义 $r_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}$，$r_l$ 同理。最大化偏好对数似然：
> $$\mathcal{L}\_{\mathrm{DPO}} = -\mathbb{E}\Big[ \log \sigma\big( \beta(r_w - r_l) \big) \Big]$$
> 即让 $\pi_\theta$ 对 $y_w$ 的相对对数概率（相对于 $\pi_{\mathrm{ref}}$）高于 $y_l$。

> [!EXAMPLE]+ 例：DPO 单步
> 给定 $x$="写首诗"，$y_w$="春眠不觉晓..."，$y_l$="今天天气不错"。计算 $\log \pi_\theta(y_w|x) - \log \pi_{\mathrm{ref}}(y_w|x)$ 与 $y_l$ 的对应项，损失推动前者大于后者，使策略偏好诗歌式回答。

<br>

### 伪代码

```text
算法：DPO
输入：偏好数据 D = {(x, y_w, y_l)}，参考模型 π_ref，策略 π_θ，超参 β

for epoch = 1, 2, ... do
    for (x, y_w, y_l) in DataLoader(D) do
        r_w = log π_θ(y_w|x) - log π_ref(y_w|x)
        r_l = log π_θ(y_l|x) - log π_ref(y_l|x)
        L = -log σ(β(r_w - r_l))
        更新 θ 最小化 L
    end for
end for
```

<br>

### 优缺点

| 优点 | 缺点 |
| :--- | :--- |
| 无需 RM、无需 RL 循环 | 依赖 Bradley-Terry 假设 |
| 训练稳定，实现简单 | 易过拟合，泛化到新分布较弱 |
| 显存友好，单模型即可 | 难以处理多目标、复杂奖励 |

<br>

---

## GRPO

*Group Relative Policy Optimization（组相对策略优化）*

> GRPO 在 RLHF 流程中的特点及与 PPO、GSPO 的关系见 [Pre-Training、SFT 与 RLHF](Pretrain-SFT-RLHF.md)。

### 定义与动机

**GRPO**：对同一 prompt 采样多个回答，用**组内相对奖励**（均值、方差标准化）估计优势，**无需价值模型**。DeepSeek-Math、DeepSeek-R1 采用。

**动机**：PPO 需 Critic 估计 $V(s)$，长文本显存压力大；GRPO 用同 prompt 下多回答的统计量替代，省去 Critic。组内标准化后，每个回答的优势 $A_i = (r_i - \mu_G)/(\sigma_G + \varepsilon)$ 表示其相对好坏，可直接用于 PPO-Clip。

<br>

### 流程概述

1. 对每个 prompt $q$，用 $\pi_\theta$ **采样 G 个回答** $o_1, \ldots, o_G$。
2. 用 $R(q, o_i)$ 得 $r_i$，组内标准化 $A_i = (r_i - \mu_G)/(\sigma_G + \varepsilon)$。
3. 对每个 $(q, o_i)$ 的每个 token $t$：算 $r_{i,t} = \pi_\theta/\pi_{\mathrm{old}}$，用 PPO-Clip 目标 $\min(r_{i,t} A_i, \mathrm{clip}(r_{i,t}) A_i) - \beta \mathrm{KL}$。
4. 聚合所有 token 的损失，更新 $\theta$。
5. 重复 1–4。

<br>

### 原理简述

GRPO 用**组内相对奖励**替代价值模型：同一 prompt 下多个回答的 $r_i$ 的均值和方差，足以构造「谁比谁好」的优势信号。$A_i > 0$ 表示该回答好于组平均，$A_i < 0$ 反之。整句共享同一 $A_i$，对每个 token 做 clip，实现无 Critic 的 PPO 式更新。

<br>

### 数学推导

> [!INFO]+ 推导：GRPO 优势与目标
> 对 prompt $q$，采样 $G$ 个回答，奖励 $r_i = R(q, o_i)$，组内标准化：
> $$A_i = \frac{r_i - \mu_G}{\sigma_G + \varepsilon}$$
>
> 在 PPO-Clip 框架下，对每个 token 使用组级**同一**优势 $A_i$（整句共享）：
> $$\mathcal{J}\_{\mathrm{GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \Big( \min\big(r_{i,t}(\theta) A_i,\; \mathrm{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) A_i\big) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \Big) \Bigg]$$
> 其中 $r_{i,t}(\theta) = \pi_\theta(o_{i,t}|q, o_{i,1:t-1}) / \pi_{\theta_{\mathrm{old}}}(o_{i,t}|q, o_{i,1:t-1})$。

> [!EXAMPLE]+ 例：GRPO 组内优势
> 对 prompt $q$ 采样 4 个回答，奖励 $r=[0.2, 0.8, 0.4, 0.6]$，$\mu=0.5$，$\sigma \approx 0.26$。$A_2 \approx 1.15$（好），$A_1 \approx -1.15$（差），用于各 token 的 clip 目标。

<br>

### 伪代码

```text
算法：GRPO
输入：策略 π_θ，参考 π_ref，奖励模型 R，组大小 G
超参：ε，β

for iteration = 1, 2, ... do
    对每个 prompt q：采样 G 个回答 o_1,...,o_G ~ π_θ(·|q)
    计算 r_i = R(q, o_i)，A_i = (r_i - mean(r)) / (std(r) + ε)
    for 每个 (q, o_i, t) do
        r_{i,t} = π_θ(o_{i,t}|...) / π_θ_old(o_{i,t}|...)
        L_t = min(r_{i,t} A_i, clip(r_{i,t},1-ε,1+ε) A_i) - β·KL
    end for
    更新 θ 最大化 Σ L_t
end for
```

<br>

### 优缺点

| 优点 | 缺点 |
| :--- | :--- |
| 无需 Critic，显存减半 | token 级优化与序列级评估错位 |
| 组内相对排序提供稳定信号 | 长度偏差；MoE 下 per-token 方差大 |
| 适合可验证奖励（数学、代码） | 易产生冗余、啰嗦输出 |

<br>

---

## GMPO

*Geometric-Mean Policy Optimization（几何平均策略优化）*

### 定义与动机

**GMPO**：用**几何平均**替代 GRPO 的算术平均聚合 token 级奖励，抑制 outlier 重要性采样比，提升策略更新稳定性。ICLR 2026，基模 Qwen2.5-Math、DeepSeek-R1-Distill-Qwen 等。

**动机**：GRPO 的算术平均 $\frac{1}{|o|}\sum_t (\cdot)$ 对极端 $\rho_t = \pi_\theta/\pi_{\mathrm{old}}$ 敏感，若某一 token 的 $\rho_t$ 极大，会主导梯度导致剧烈更新与奖励崩塌；几何平均 $(\prod_t (\cdot))^{1/|o|}$ 乘性聚合，单个 outlier 影响被稀释。

<br>

### 流程概述

与 GRPO 相同：采样 G 个回答 → 组内标准化得 $A_i$ → 对每个 token 计算 clip 项。区别仅在**聚合方式**：GRPO 用 $\frac{1}{|o|}\sum_t (\cdot)$，GMPO 用 $\big(\prod_t |(\cdot)|\big)^{1/|o|} \cdot \mathrm{sgn}(\hat{A})$。clip 范围通常取 $(e^{-0.4}, e^{0.4})$，较 GRPO 更宽。

<br>

### 原理简述

算术平均对 outlier 敏感：一个极大的 $\rho_t$ 会拉高整体目标。几何平均的乘性结构使单个异常值影响被开方稀释，更新更平滑。同时 GMPO 放宽 clip 范围，在稳定与探索间取得更好平衡。

<br>

### 数学形式

> [!INFO]+ 推导：GMPO 目标
> **GRPO**（算术平均）：
> $$\frac{1}{|o|}\sum_t \min(\rho_t \hat{A}, \mathrm{clip}(\rho_t, \epsilon_{\mathrm{low}}, \epsilon_{\mathrm{high}}) \hat{A})$$
>
> **GMPO**（几何平均）：
> $$\Big\{\prod_t \big|\min(\rho_t \hat{A}, \mathrm{clip}(\rho_t, \epsilon_{\mathrm{low}}, \epsilon_{\mathrm{high}}) \hat{A})\big|\Big\}^{1/|o|} \cdot \mathrm{sgn}(\hat{A})$$
> 取绝对值后开 $1/|o|$ 次方，再用 $\mathrm{sgn}(\hat{A})$ 恢复符号。
>
> **关键设计**：token 级 clip（非序列级）；clip 范围 $(e^{-0.4}, e^{0.4}) \approx (0.67, 1.49)$ 较 GRPO 的 $(0.8, 1.2)$ 更宽，平衡探索与稳定。

> [!EXAMPLE]+ 例：几何平均对 outlier 的抑制
> 设 $|o|=4$，四个 token 的 clip 后项为 $[1.0, 1.0, 1.0, 10]$。算术平均 $\approx 3.25$，被 10 拉高；几何平均 $(1 \cdot 1 \cdot 1 \cdot 10)^{1/4} \approx 1.78$，影响更小。

<br>

### 优缺点

| 优点 | 缺点 |
| :--- | :--- |
| 抑制 outlier，更新更稳定 | 需替换 GRPO 目标，实现略改 |
| 保持更高熵，探索更好 | clip 超参需调 |
| Pass@1 较 GRPO 提升约 4.1% | 与 Dr.GRPO 等可叠加 |

<br>

---

## DAPO

*GRPO 的工程优化变体（无独立全称）*

### 定义与动机

DAPO 在 GRPO 框架上做三项工程优化，提升训练效率与稳定性。

<br>

### 流程概述

与 GRPO 相同的主循环，但在三处做改动：(1) clip 用非对称区间（好动作更宽）；(2) 动态丢弃全对/全错 prompt；(3) 归一化用 $\frac{1}{\sum_i |o_i|}$ 替代 $\frac{1}{|o_i|}$。实现时只需在 GRPO 代码上改这三处。

<br>

### 原理简述

Clip-Higher 让好动作有更大更新空间，鼓励探索；动态采样避免无效 batch；统一归一化缓解「短句权重过高」的长度偏差。三者都是工程级改进，不改变 GRPO 的组相对优势框架。

<br>

### 核心设计

1. **Clip-Higher（非对称 clip）**：对好动作（$A_i>0$）使用更宽松的上界，鼓励探索；对坏动作保持严格 clip，抑制恶化。
2. **动态采样**：丢弃全对或全错的 prompt，避免无信息量的 batch 浪费算力；只保留有对有错的 prompt，提供有效梯度。
3. **Token 级归一化**：用 $\frac{1}{\sum_i |o_i|}$ 替代每句的 $\frac{1}{|o_i|}$，避免长句被过度惩罚，缓解长度偏差。

### 与 GSPO 的区别

DAPO 仍为 **token 级**优化；GSPO 改为**序列级**，对整句做重要性采样与 clip，方差更低。

<br>

---

## GSPO

*Group Sequence Policy Optimization（组序列策略优化）*

### 定义与动机

将 GRPO 的 **token 级**优化改为**序列级**，对整句做重要性采样与 clip。GRPO 对每个 token 计算 $r_{i,t}$ 并平均，方差大；GSPO 将整句概率比作为标量权重，方差更低，尤其适合 MoE（专家路由导致 per-token 方差大）。

<br>

### 流程概述

与 GRPO 相同：采样 G 个回答 → 组内标准化得 $A_i$。区别在**优化粒度**：GRPO 对每个 token 有 $r_{i,t}$，GSPO 对整句算一个 $s_i = (\pi_\theta(y_i|x)/\pi_{\mathrm{old}}(y_i|x))^{1/|y_i|}$，对 $s_i$ 做 clip 后乘 $A_i$，**不再对 token 展开**。一次前向即可得整句梯度。

<br>

### 原理简述

token 级优化时，每个 token 的 $r_{i,t}$ 方差叠加，MoE 下专家路由更放大方差。序列级把整句视为一个单位，$s_i$ 是标量，方差小；且与「整句一个奖励」的评估粒度一致，减少错位。

<br>

### 数学形式

**序列级权重**（几何平均形式的概率比）：
$$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\mathrm{old}}}(y_i|x)}\right)^{1/|y_i|}$$

**目标**：
$$L_{\mathrm{GSPO}} = \mathbb{E}\big[ \min(s_i \hat{A}_i, \mathrm{clip}(s_i, 1-\varepsilon, 1+\varepsilon) \hat{A}_i) \big]$$

与 GRPO 的区别：优化粒度序列级，单句一个 $s_i$，不再对 token 展开；方差更低，尤其适合 MoE。

<br>

---

## GFPO

*Group Filtered Policy Optimization（组过滤策略优化）*

### 定义与动机

在 GRPO 基础上**采样更大组**（如 N=16），再按长度或 token 效率**筛选**部分回答用于训练。目标是抑制模型通过「写更长」来刷奖励的长度偏差，控制输出膨胀。

<br>

### 流程概述

1. 对每个 prompt 采样 **更大组**（如 16 个）回答。
2. 按选定策略**过滤**：如取最短 k 个、或按 $r_i/|o_i|$ 取前 k 个。
3. 仅用过滤后的回答参与 GRPO 的 clip 目标与梯度更新。
4. 重复 1–3。

<br>

### 原理简述

GRPO 易产生「写长刷分」：长回答 tokens 多，总奖励可能更高，模型倾向啰嗦。GFPO 通过**过滤**隐式塑形奖励：只让短而好或单位 token 效率高的回答参与训练，等价于惩罚冗长，抑制输出膨胀。

<br>

### 过滤策略

1. **Shortest k/N**：只取组内最短的 k 个回答参与梯度，鼓励简洁。
2. **Token Efficiency**：按 $r_i / |o_i|$ 排序，取高效回答（单位 token 奖励高）。
3. **Adaptive Difficulty**：根据当前能力调整过滤阈值，避免过滤过严或过松。

过滤等价于隐式奖励塑形，Phi-4-reasoning 等技术报告显示可抑制长输出膨胀约 46–71%。

<br>

---

## GAPO

*Generative Adversarial Policy Optimization（生成对抗策略优化）*

### 定义与动机

结合 **GAN** 与 **PPO**，用 encoder-only 的 RM 在对抗训练中学习约束型 prompt-response 对应关系。适用于细粒度约束（如格式、关键词、长度等），RM 可判别生成是否满足约束。

<br>

### 流程概述

1. **Warmup**：用标注数据训 RM，学会区分满足/不满足约束的回答。
2. **对抗循环**：Generator（策略）生成回答 → RM 打分 → 用 PPO 目标 + RM 信号更新 Generator；交替更新 RM 使其更准。
3. 重复 2，直到收敛。

<br>

### 原理简述

传统 RM 用 decoder 对 $(q,o)$ 打分；GAPO 用 **encoder-only** RM 做二分类（满足/不满足约束），可与 PPO 联合对抗训练。Generator 试图骗过 RM，RM 试图更准，形成博弈；约束信号通过 RM 注入 PPO 目标，实现细粒度控制。

<br>

### 框架

1. **Warmup**：先训练 RM 区分满足/不满足约束的回答。
2. **对抗阶段**：交替更新 Generator（策略）与 RM；Generator 试图生成能骗过 RM 的回答，RM 试图更准确判别。
3. **与 PPO 结合**：在 PPO 目标中加入 RM 的判别信号，形成对抗+策略梯度的混合目标。

细粒度约束遵循优于 PPO、DPO、KTO。

<br>

---

## Dr.GRPO

*修正 GRPO 长度偏差的变体*

### 定义与动机

GRPO 目标中对每个回答按 $\frac{1}{|o_i|}$ 归一化，导致**长度偏差**：短句每个 token 权重高，模型倾向生成更短回答以获取更高有效梯度。Dr.GRPO 用**组平均长度**作为统一缩放因子，缓解该偏差。

<br>

### 流程概述

与 GRPO 完全相同，仅把归一化分母从每句的 $|o_i|$ 改为组平均 $\bar{L} = \frac{1}{G}\sum_i |o_i|$。即所有回答共享同一缩放因子，短句不再被过度加权。

<br>

### 原理简述

$\frac{1}{|o_i|}$ 使短句每 token 的梯度贡献更大，模型学到「写短」能提高有效梯度。Dr.GRPO 用 $\bar{L}$ 统一分母，使不同长度的回答在梯度尺度上更公平，缓解长度偏好。

<br>

### 数学形式

用 $\bar{L} = \frac{1}{G}\sum_{i=1}^G |o_i|$ 替代每句的 $|o_i|$，即所有回答共享同一归一化分母，避免短句被过度加权。

### 与 λ-GRPO

λ-GRPO 引入可学习 token 权重，更灵活，通常优于 Dr.GRPO。

<br>

---

## 总结

### 算法对比表

| 算法 | RM | Critic | KL/Ref | 优化粒度 | 主要改进点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Q-learning | — | — | — | — | 表格式 $Q^{\ast}$ 学习 |
| PPO | 需 | 需 | 需 | token | clip 限制更新 |
| VAPO | 需 | 需 | 需 | token | 价值增强，推理 SOTA |
| DPO | 不需 | 不需 | 隐式 | 偏好对 | 直接偏好优化 |
| GRPO | 需 | 不需 | 需 | token | 组内相对优势，无 Critic |
| GMPO | 需 | 不需 | 需 | token | 几何平均，稳定 GRPO |
| DAPO | 需 | 不需 | 需 | token | Clip-Higher、动态采样、归一化 |
| GSPO | 需 | 不需 | 需 | 序列 | 序列级 clip，降方差 |
| GFPO | 需 | 不需 | 需 | token | 采样过滤，控长度 |
| GAPO | encoder-only | 有 | 需 | token | 对抗约束学习 |
| Dr.GRPO | 需 | 不需 | 需 | token | 修正长度归一化偏差 |

<br>

### 继承与改进关系

```
Q-learning (TD, 无模型)
    ↓
策略梯度 / Actor-Critic
    ↓
PPO (clip, GAE, 稳定)
    ├→ VAPO (value 增强，推理专用)
    └→ GRPO (无 Critic，组相对优势)
           ├→ GMPO (几何平均，稳定)
           ├→ DAPO (clip、动态采样、归一化)
           ├→ GSPO (序列级，降方差)
           ├→ GFPO (过滤，控长度)
           └→ Dr.GRPO (修正长度偏差)

DPO (独立分支：无 RM/RL，Bradley-Terry)
GAPO (独立分支：GAN + PPO，约束学习)
```

<br>

### 使用建议

| 场景 | 推荐 |
| :--- | :--- |
| 通用 RLHF、资源充足 | PPO |
| 推理 SOTA、可接受 value 成本 | VAPO |
| 偏好数据多、追求简单 | DPO |
| 长文本、可验证奖励、省显存 | GRPO / GMPO |
| MoE、序列级评估 | GSPO |
| 推理链冗长、需控长度 | GFPO |
| 细粒度约束、prompt 敏感 | GAPO |
| 长度偏差明显 | Dr.GRPO 或 λ-GRPO |

<br>

