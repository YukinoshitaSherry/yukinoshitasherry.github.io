---
title: ResNet论文精读
date: 2024-03-25
categories:
    - 学AI/DS
tags:
    - LLM
    - NLP
    - CV
desc: ResNet精读与整理，缓解梯度爆炸的数学原理。
---

## 参考
<br>

### 资料
- [ResNet 论文概览与精读 - 周弈帆的博客](https://zhouyifan.net/2022/08/09/20220807-ResNet/)
- [ResNet论文+复现 - Alaskaboo的博客](https://blog.alaskaboo.cn/2025/07/25/ResNet%E8%AE%BA%E6%96%87/)
- [ResNet论文详解 - 知乎](https://zhuanlan.zhihu.com/p/159162779)
- [ResNet论文详解 - CSDN](https://blog.csdn.net/xiaoyuting999/article/details/135077588)
- [李沐论文精读系列一： ResNet、Transformer、GAN、BERT_李沐读论文-CSDN博客](https://blog.csdn.net/qq_56591814/article/details/127313216)
<br>

### 论文
**Deep Residual Learning for Image Recognition.** `CVPR` 2016
[[1512.03385] Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
代码：[pytorch/vision: Data loaders and abstractions for computer vision datasets](https://github.com/pytorch/vision/tree/main/torchvision/models)

<br>






## 退化问题概况

【深度网络的训练困境】

### 论文实验

论文开篇即指出：**"Deeper neural networks are more difficult to train."**（更深的神经网络更难训练）

论文通过实验观察到，随着网络深度的增加，出现了**退化（Degradation）问题**：

**实验设置**：
- 20层普通网络（Plain-20）
- 56层普通网络（Plain-56）
- 相同的训练设置、数据增强、优化器

**实验结果**：
- 20层网络的训练误差：$\epsilon_{train}^{(20)}$
- 56层网络的训练误差：$\epsilon_{train}^{(56)} > \epsilon_{train}^{(20)}$
- 56层网络的测试误差：$\epsilon_{test}^{(56)} > \epsilon_{test}^{(20)}$

<br>

### 特征分析
> "This reveals that not all systems are similarly easy to optimize."

**这不是过拟合问题**：
- 如果是过拟合，训练误差应该降低：$\epsilon_{train}^{(56)} < \epsilon_{train}^{(20)}$
- 但实际观察到：$\epsilon_{train}^{(56)} > \epsilon_{train}^{(20)}$
- **训练误差和测试误差都增加**，说明问题不在泛化能力

**这是优化问题**：
- 训练误差增加说明模型**无法在训练集上拟合得更好**
- 更深网络的**模型容量**是足够的（理论上可以表示浅层网络的解）
- 问题在于**优化器无法找到更好的解**

<br>

## 残差思想


论文提出一个关键假设：**"If the added layers can be constructed as identity mappings, a deeper model should have training error no greater than its shallower counterpart."**

**数学表述**：如果较深的网络 $f_L$ 可以构造为恒等映射，即：
$$f_L(x) = f_{L-1}(x)$$

那么更深网络的训练误差应该**不大于**浅层网络。

**论文原文的推理**：
- 如果恒等映射是最优的，那么将新增层学习为恒等映射应该不会增加训练误差
- 但实际训练中，优化器很难将新增层学习为恒等映射
- 因此需要**显式地提供恒等映射的路径**（快捷连接）



<br>


## 数学原理

### 基本表示

**传统网络**：直接学习期望映射 $H(x)$
$$y = H(x)$$

传统深度网络的前向传播

对于 $L$ 层深度网络，输入 $x$ 经过逐层变换：

$$x_0 = x$$
$$x_1 = H_1(x_0) = \sigma(W_1 x_0 + b_1)$$
$$x_2 = H_2(x_1) = \sigma(W_2 x_1 + b_2)$$
$$\vdots$$
$$x_L = H_L(x_{L-1}) = \sigma(W_L x_{L-1} + b_L)$$

其中 $H_i$ 表示第 $i$ 层的映射函数，$\sigma$ 是激活函数（如ReLU）。

**残差网络**：学习残差映射 $F(x) = H(x) - x$
$$y = F(x) + x = H(x)$$

**关键洞察**：
- 如果最优映射是恒等映射，即 $H^{\ast}(x) = x$，那么 $F^{\ast}(x) = H^{\ast}(x) - x = 0$
- 学习 $F(x) = 0$ 比学习 $H(x) = x$ **更容易** 【这个思想相当重要！】

> [!info]+ 残差学习的优势并不依赖于最优映射一定是恒等映射，而是基于以下两个关键点
> 
> 1. **如果最优映射接近恒等映射**（$H^{\ast}(x) \approx x$）：
>    - 那么 $F^{\ast}(x) = H^{\ast}(x) - x \approx 0$
>    - 学习 $F(x) = 0$ 比学习 $H(x) = x$ **更容易**（初始状态已接近目标）
> 
> 2. **如果最优映射不是恒等映射**（$H^{\ast}(x) \neq x$）：
>    - 残差网络只需要学习**增量** $F^{\ast}(x) = H^{\ast}(x) - x$
>    - 学习增量比学习完整的 $H^{\ast}(x)$ **更容易**，因为：
>      - 增量通常比完整映射更小、更简单
>      - 初始状态 $F(x; \theta_0) \approx 0$ 提供了良好的起点
>      - 快捷连接保证了信息的直接流动
> 
> **为什么残差学习总是更容易？**
> 
> 关键在于：**学习增量（残差）比学习完整映射更容易**，无论最优映射是否接近恒等映射。
> 
> - **传统网络**：需要从随机函数 $H(x; \theta_0)$ 学习到完整的最优映射 $H^{\ast}(x)$
> - **残差网络**：只需要从接近零的函数 $F(x; \theta_0) \approx 0$ 学习到增量 $F^{\ast}(x) = H^{\ast}(x) - x$
> 
> 即使 $H^{\ast}(x)$ 与 $x$ 差异很大，学习增量 $H^{\ast}(x) - x$ 仍然比学习完整的 $H^{\ast}(x)$ 更容易，因为增量通常具有更简单的结构。



<br>

### 具体推导

为什么学习残差更容易？

#### 传统网络学习恒等映射的困难

**目标**：找到参数 $\theta = \{W\_i, b\_i\}\_{i=1}^L$ 使得：
$$H(x; \theta) = x \quad \forall x \in \mathbb{R}^d$$

**约束条件分析**：

对于线性层 $H(x) = Wx + b$，要满足 $H(x) = x$，需要：
$$Wx + b = x \quad \forall x$$

这要求：
- $W = I$（单位矩阵）
- $b = 0$（零向量）

**对于深度网络**：
$$H(x) = H\_L(H\_{L-1}(\cdots H\_1(x))) = x$$

这要求**每一层**都近似恒等映射，或者多层组合后等于恒等映射。

**优化难度**：
- 需要同时优化所有层的参数
- 初始状态：$H(x; \theta\_0) \approx \text{随机函数}$（远离恒等映射）
- 目标：$H(x; \theta^*) = x$（精确的恒等映射）
- **优化路径很长**，容易陷入局部最优

<br>

#### 残差网络学习零映射的优势

**目标**：找到参数 $\theta$ 使得：
$$F(x; \theta) = 0 \quad \forall x$$

**关键洞察**：如果权重初始化为接近0（如He初始化），则：

$$F(x; \theta\_0) = W\_2 \sigma(W\_1 x + b\_1) + b\_2 \approx 0$$

因为：
- $W_1, W_2$ 初始值接近0
- $b_1, b_2$ 初始值为0
- 因此 $W_1 x + b_1 \approx 0$，$\sigma(0) = 0$（对于ReLU），$W_2 \cdot 0 + b_2 \approx 0$

**初始状态分析**：

对于残差块 $y = F(x) + x$：
- 初始状态：$F(x; \theta_0) \approx 0$
- 因此：$y = F(x; \theta_0) + x \approx 0 + x = x$

**初始状态已经接近恒等映射！**

#### 优化难度的数学对比

**传统网络**：
- 初始误差：$||H(x; \theta_0) - x||\_2$ 很大（随机函数与恒等映射的差距）
- 目标误差：$||H(x; \theta^*) - x||\_2 = 0$
- 需要优化的距离：$\Delta\_{plain} = ||H(x; \theta_0) - x||\_2$（很大）

**残差网络**：
- 初始误差：$||F(x; \theta_0) + x - x||\_2 = ||F(x; \theta_0)||\_2 \approx 0$
- 目标误差：$||F(x; \theta^*) + x - H^*(x)||]_2 = ||F(x; \theta^*) - (H^*(x) - x)||\_2$
- 需要优化的距离：$\Delta\_{res} = ||F(x; \theta_0) - (H^*(x) - x)||\_2 \approx ||H^*(x) - x||\_2$

**关键结论**：

**情况1：最优映射接近恒等**（$H^{\ast}(x) \approx x$）
- 则 $\Delta_{res} \approx ||H^{\ast}(x) - x||\_2 \approx 0$，优化很容易
- 残差函数 $F^{\ast}(x) \approx 0$，初始状态已接近目标

**情况2：最优映射不是恒等**（$H^{\ast}(x) \neq x$）
- 残差网络需要学习增量：$\Delta_{res} = ||H^{\ast}(x) - x||\_2$
- 传统网络需要学习完整映射：$\Delta\_{plain} = ||H(x; \theta_0) - H^{\ast}(x)||_2$

**为什么学习增量仍然更容易？**

1. **增量通常更小**：$||H^{\ast}(x) - x||\_2 \leq ||H^{\ast}(x)||\_2 + ||x||\_2$，而 $||H(x; \theta_0) - H^{\ast}(x)||\_2$ 可能很大（随机函数与最优映射的差距）

2. **增量具有更简单的结构**：$H^{\ast}(x) - x$ 通常比 $H^{\ast}(x)$ 本身更平滑、更容易学习

3. **初始状态优势**：$F(x; \theta_0) \approx 0$ 提供了良好的起点，而 $H(x; \theta_0)$ 是随机函数，远离目标

4. **优化路径更短**：从 $F(x; \theta_0) \approx 0$ 到 $F^{\ast}(x) = H^{\ast}(x) - x$ 的路径，比从 $H(x; \theta_0)$ 到 $H^{\ast}(x)$ 的路径更短、更直接

**因此，无论最优映射是否接近恒等映射，残差学习都更容易优化。**

#### 总结

**定理**：对于深度网络，学习残差映射 $F(x) = H(x) - x$ 比直接学习 $H(x)$ 更容易优化。

**证明**：
1. 初始状态：$F(x; \theta_0) \approx 0$，因此 $y = F(x; \theta_0) + x \approx x$（初始时接近恒等映射）
2. **无论最优映射是否接近恒等映射**，残差网络都只需要学习增量 $F^{\ast}(x) = H^{\ast}(x) - x$
3. 学习增量比学习完整的 $H^{\ast}(x)$ 更容易，因为：
   - 初始状态 $F(x; \theta_0) \approx 0$ 提供了良好的起点
   - 增量通常比完整映射更小、结构更简单
   - 优化路径更短、更直接
4. 梯度信号更稳定（通过快捷连接直接传播）

**结论**：残差学习将"学习完整映射 $H^{\ast}(x)$"转化为"学习增量 $H^{\ast}(x) - x$"，无论最优映射是否接近恒等映射，都大大降低了优化难度。

<br>


### 为什么更深网络更难优化？

#### 优化空间的复杂性

考虑损失函数 $L(\theta)$，其中 $\theta = \{W_i, b_i\}\_{i=1}^L$。

对于 $L$ 层网络，参数空间维度为：
$$d_L = \sum\_{i=1}^L (n_i \times n\_{i-1} + n_i)$$

其中 $n_i$ 是第 $i$ 层的神经元数量。

**关键问题**：随着 $L$ 增加，优化空间呈指数级增长：
- 局部最优解数量：$O(e^{d_L})$
- 鞍点数量：$O(e^{d_L})$
- 平坦区域：$O(e^{d_L})$

<Br>

#### 梯度传播的详细数学分析

考虑反向传播，损失函数对第 $l$ 层参数的梯度。根据链式法则：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial x_L} \cdot \frac{\partial x_L}{\partial x\_{L-1}} \cdot \frac{\partial x\_{L-1}}{\partial x\_{L-2}} \cdots \frac{\partial x\_{l+1}}{\partial x_l} \cdot \frac{\partial x_l}{\partial W_l}$$

**步骤1：展开链式法则**

将中间项合并为连乘形式：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial x_L} \prod_{i=l+1}^{L} \frac{\partial x_i}{\partial x\_{i-1}} \cdot \frac{\partial x_l}{\partial W_l}$$

**步骤2：计算每层的雅可比矩阵**

对于每一层 $i$，前向传播为：
$$x_i = \sigma(W_i x\_{i-1} + b_i)$$

其中 $\sigma$ 是激活函数（如ReLU）。

对 $x_{i-1}$ 求偏导：

$$\frac{\partial x_i}{\partial x\_{i-1}} = \frac{\partial \sigma(W_i x_{i-1} + b_i)}{\partial x_{i-1}}$$

使用链式法则：
$$\frac{\partial x_i}{\partial x\_{i-1}} = \frac{\partial \sigma(z_i)}{\partial z_i} \cdot \frac{\partial z_i}{\partial x\_{i-1}}$$

其中 $z_i = W_i x_{i-1} + b_i$。

计算：
- $\frac{\partial z_i}{\partial x_{i-1}} = W_i$
- $\frac{\partial \sigma(z_i)}{\partial z_i} = \sigma'(z_i)$（激活函数的导数）

因此：
$$\frac{\partial x_i}{\partial x_{i-1}} = \sigma'(z_i) \cdot W_i$$

**步骤3：梯度消失的数学原因**

将雅可比矩阵代入梯度公式：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial x_L} \prod_{i=l+1}^{L} [\sigma'(z_i) \cdot W_i] \cdot \frac{\partial x_l}{\partial W_l}$$

**关键问题**：当 $L-l$ 很大时，连乘项 $\prod_{i=l+1}^{L} [\sigma'(z_i) \cdot W_i]$ 会发生什么？

**情况1：使用sigmoid/tanh激活函数**

对于sigmoid：$\sigma'(z) = \sigma(z)(1-\sigma(z)) \in (0, 0.25]$

对于tanh：$\sigma'(z) = 1 - \tanh^2(z) \in (0, 1]$

如果 $|\sigma'(z_i)| < 1$，则：

$$\left|\prod\_{i=l+1}^{L} [\sigma'(z_i) \cdot W_i]\right| \leq \prod\_{i=l+1}^{L} |\sigma'(z_i)| \cdot ||W_i||\_2$$

当 $L-l$ 很大时，如果 $|\sigma'(z_i)| \cdot ||W_i||\_2 < 1$，这个乘积会**指数级衰减**到接近0。

**情况2：使用ReLU激活函数**

ReLU的导数为：
$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

理论上，如果 $z_i > 0$，$\sigma'(z_i) = 1$，梯度不会消失。

**但实际情况更复杂**：

1. **死ReLU问题**：如果 $z_i \leq 0$，$\sigma'(z_i) = 0$，梯度完全消失
2. **权重矩阵的谱范数**：即使 $\sigma'(z_i) = 1$，如果 $||W_i||_2 < 1$，梯度仍会衰减
3. **权重初始化**：如果权重初始化不当，可能导致大部分神经元处于死区

**数学证明梯度消失**：

假设每层的雅可比矩阵的谱范数满足：
$$||\frac{\partial x_i}{\partial x\_{i-1}}||\_2 = ||\sigma'(z_i) \cdot W_i||\_2 \leq \lambda < 1$$

则：
$$||\prod\_{i=l+1}^{L} \frac{\partial x_i}{\partial x\_{i-1}}||\_2 \leq \prod\_{i=l+1}^{L} ||\frac{\partial x_i}{\partial x\_{i-1}}||\_2 \leq \lambda^{L-l}$$

当 $L-l \to \infty$ 时，$\lambda^{L-l} \to 0$，梯度消失。

**即使使用Batch Normalization**：
- BN可以稳定激活值的分布，减少死ReLU
- 但**不能完全解决**梯度消失问题
- 深层网络的梯度仍然很弱，优化困难

#### 退化问题的本质

**数学表述**：对于深度网络 $f_L(x; \theta_L)$，即使存在参数 $\theta_L^*$ 使得 $f_L(x; \theta_L^*) = f\_{L-1}(x; \theta\_{L-1}^*)$，优化算法也很难找到这样的参数。

**原因**：
1. **优化空间太大**：需要同时优化所有层的参数，参数空间维度呈指数级增长
2. **梯度信息衰减**：深层参数的梯度信号很弱，梯度消失导致优化困难
3. **初始化敏感**：初始参数的选择严重影响最终性能，难以找到接近恒等映射的初始状态

<br>

### 残差网络的梯度流动分析

#### 单个残差块的梯度计算

对于残差块 $y = F(x) + x$，我们需要计算损失函数 $L$ 对输入 $x$ 的梯度。

**使用链式法则**：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

**计算 $\frac{\partial y}{\partial x}$**：

由于 $y = F(x) + x$，有：
$$\frac{\partial y}{\partial x} = \frac{\partial (F(x) + x)}{\partial x} = \frac{\partial F(x)}{\partial x} + \frac{\partial x}{\partial x} = \frac{\partial F}{\partial x} + I$$

其中 $I$ 是单位矩阵。

**因此**：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + I\right) = \frac{\partial L}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

**关键洞察**：
- $\frac{\partial L}{\partial y}$ 项提供了梯度直接传播的路径（通过快捷连接）
- 即使 $\frac{\partial F}{\partial x}$ 很小（接近0），梯度仍可通过恒等项 $I$ 传播
- **梯度不会完全消失**：只要 $\frac{\partial L}{\partial y}$ 不为零，$\frac{\partial L}{\partial x}$ 就不会为零

#### 多层残差网络的梯度传播

考虑 $n$ 层残差网络，每层输出为：
$$x_{l+1} = F_l(x_l) + x_l, \quad l = 0, 1, \ldots, n-1$$

**计算第 $l$ 层的梯度**：

根据链式法则：
$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_n} \cdot \frac{\partial x_n}{\partial x\_{n-1}} \cdot \frac{\partial x\_{n-1}}{\partial x\_{n-2}} \cdots \frac{\partial x\_{l+1}}{\partial x_l}$$

**计算每层的雅可比矩阵**：

对于 $x_{i+1} = F_i(x_i) + x_i$：
$$\frac{\partial x_{i+1}}{\partial x_i} = \frac{\partial (F_i(x_i) + x_i)}{\partial x_i} = \frac{\partial F_i}{\partial x_i} + I = I + \frac{\partial F_i}{\partial x_i}$$

**代入链式法则**：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_n} \prod_{i=l}^{n-1} \frac{\partial x\_{i+1}}{\partial x_i} = \frac{\partial L}{\partial x_n} \prod_{i=l}^{n-1} \left(I + \frac{\partial F_i}{\partial x_i}\right)$$

**关键分析**：

对于标量情况（简化理解），有：
$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_n} \prod_{i=l}^{n-1} \left(1 + \frac{\partial F_i}{\partial x_i}\right)$$

**与普通网络的对比**：

普通网络：$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_n} \prod_{i=l}^{n-1} \frac{\partial F_i}{\partial x_i}$

残差网络：$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_n} \prod_{i=l}^{n-1} \left(1 + \frac{\partial F_i}{\partial x_i}\right)$

**关键差异**：
- 普通网络：如果 $\frac{\partial F_i}{\partial x_i} < 1$，梯度会指数衰减
- 残差网络：即使 $\frac{\partial F_i}{\partial x_i} \approx 0$，仍有 $1 + 0 = 1$，梯度不会消失

#### 梯度消失的数学证明

**普通网络的梯度消失**：

如果 $|\frac{\partial F_i}{\partial x_i}| < \lambda < 1$，则：
$$\left|\prod\_{i=l}^{n-1} \frac{\partial F_i}{\partial x_i}\right| \leq \lambda^{n-l}$$

当 $n-l \to \infty$ 时，$\lambda^{n-l} \to 0$，**梯度消失**。

**残差网络的梯度保持**：

即使 $\frac{\partial F_i}{\partial x_i} \approx 0$，有：
$$\prod\_{i=l}^{n-1} \left(1 + \frac{\partial F_i}{\partial x_i}\right) \approx \prod\_{i=l}^{n-1} 1 = 1$$

**更一般的情况**：

如果 $\frac{\partial F_i}{\partial x_i}$ 有正有负，但平均接近0，则：
$$\prod\_{i=l}^{n-1} \left(1 + \frac{\partial F_i}{\partial x_i}\right) \approx 1 + \sum_{i=l}^{n-1} \frac{\partial F_i}{\partial x_i} + O\left(\left(\frac{\partial F_i}{\partial x_i}\right)^2\right)$$

只要 $\sum_{i=l}^{n-1} \frac{\partial F_i}{\partial x_i}$ 不会很大（负值），梯度就不会消失。

#### 梯度爆炸的预防

**潜在问题**：如果 $\frac{\partial F_i}{\partial x_i} > 0$ 且很大，可能导致梯度爆炸。

**解决方案**：
1. **Batch Normalization**：稳定激活值的分布
2. **权重初始化**：使用He初始化，确保初始梯度合理
3. **梯度裁剪**：限制梯度的最大值

**数学分析**：

如果 $|\frac{\partial F_i}{\partial x_i}| < M$（有界），则：
$$\left|\prod\_{i=l}^{n-1} \left(1 + \frac{\partial F_i}{\partial x_i}\right)\right| \leq (1 + M)^{n-l}$$

虽然可能增长，但比普通网络的指数衰减要好得多。

#### 信息流动的数学保证

**前向传播**：
$$x_{l+1} = F_l(x_l) + x_l$$

**关键性质**：
- 即使 $F_l(x_l) = 0$（层没有学到有用信息），输入 $x_l$ 仍能直接传递到 $x_{l+1}$
- 这保证了信息的直接流动，避免了信息丢失

**数学证明**：

如果 $F_l(x_l) = 0$，则：
$$x_{l+1} = 0 + x_l = x_l$$

**恒等映射得到保证**，信息不会丢失。

**与普通网络对比**：

普通网络：$x_{l+1} = F_l(x_l)$
- 如果 $F_l(x_l) = 0$，则 $x_{l+1} = 0$，**信息完全丢失**

残差网络：$x_{l+1} = F_l(x_l) + x_l$
- 如果 $F_l(x_l) = 0$，则 $x_{l+1} = x_l$，**信息完全保留**

<br>

## 残差块设计

### 残差块结构

#### 论文定义

对于输入 $x$，残差块定义为：
$$y = F(x, \{W_i\}) + x$$

其中：
- $F(x, \{W_i\})$ 是学习的残差映射（通常包含2-3个卷积层）
- $x$ 是输入（通过快捷连接直接传递）
- $y$ 是输出

#### 维度匹配问题

当输入和输出的维度不匹配时（$F$ 和 $x$ 的维度不同），使用线性投影 $W_s$：
$$y = F(x, \{W_i\}) + W_s x$$

**快捷连接的类型**：
- **恒等映射**：当维度匹配时，$y = F(x) + x$
- **投影映射**：当维度不匹配时，$y = F(x) + W_s x$

#### 论文实验的三种配置

论文实验了三种快捷连接的配置：
- **A**：全部使用恒等映射（维度不匹配时用零填充）
- **B**：仅在维度变化时使用投影映射
- **C**：全部使用投影映射

**实验结果**：B > A > C（B是公认的ResNet标配）

**原因分析**：
- 配置A：零填充会引入额外的零值，可能影响学习
- 配置B：只在必要时使用投影，既保持了恒等映射的优势，又解决了维度问题
- 配置C：所有连接都投影，增加了参数但可能引入不必要的复杂性

<br>

### 基本残差块

#### 数学结构

对于输入 $x$，基本残差块包含：
1. 两个3×3卷积层：$F(x) = W_2 \sigma(W_1 x + b_1) + b_2$
2. 快捷连接：$x$
3. 输出：$y = \sigma(F(x) + x)$

其中 $\sigma$ 是ReLU激活函数。

**完整公式**（包含Batch Normalization）：
$$F(x) = BN(W_2 \sigma(BN(W_1 x)))$$
$$y = \sigma(F(x) + x)$$

#### 维度匹配的数学处理

当输入输出维度不匹配时：
$$y = \sigma(F(x) + W_s x)$$

其中 $W_s$ 是1×1卷积，用于维度调整。

### 瓶颈残差块

#### 数学结构

对于输入 $x$，瓶颈块包含：
1. 1×1卷积（降维）：$x_1 = W_1 x$
2. 3×3卷积（特征提取）：$x_2 = W_2 x_1$
3. 1×1卷积（升维）：$F(x) = W_3 x_2$
4. 快捷连接：$x$ 或 $W_s x$
5. 输出：$y = \sigma(F(x) + x)$

**完整公式**：
$$x_1 = BN(W_1 x)$$
$$x_2 = BN(W_2 \sigma(x_1))$$
$$F(x) = BN(W_3 \sigma(x_2))$$
$$y = \sigma(F(x) + x)$$

#### 参数效率的数学分析

假设输入输出都是 $C$ 维通道：

**基本块**：
- 参数数量：$2 \times (3 \times 3 \times C^2) = 18C^2$

**瓶颈块**：
- 1×1降维：$1 \times 1 \times C \times C/4 = C^2/4$
- 3×3特征提取：$3 \times 3 \times (C/4)^2 = 9C^2/16$
- 1×1升维：$1 \times 1 \times C/4 \times C = C^2/4$
- 总参数：$C^2/4 + 9C^2/16 + C^2/4 = 17C^2/16$

**参数效率对比**：
- 基本块：$18C^2$ 参数
- 瓶颈块：$17C^2/16 \approx 1.06C^2$ 参数（当 $C$ 较大时）

当 $C$ 较大时（如 $C = 256$），瓶颈块参数更少，计算更高效。

<br>

### 激活函数位置

#### 后激活（Post-activation）

论文采用。

**结构**：Conv → BN → ReLU → Conv → BN → ReLU（相加后）

**数学**：
$$F(x) = BN(W_2 \sigma(BN(W_1 x)))$$
$$y = \sigma(F(x) + x)$$

#### 前激活（Pre-activation）- 后续改进

**结构**：BN → ReLU → Conv → BN → ReLU → Conv（相加后）

**数学**：
$$F(x) = W_2 \sigma(BN(W_1 \sigma(BN(x))))$$
$$y = F(x) + x$$

**优势**：更好的梯度流动，更易训练（梯度可以直接通过快捷连接传播，不受BN和ReLU影响）

<br>

## 网络架构

### ResNet架构设计

#### ResNet-18/34（浅层网络）

**基础块**：两个3×3卷积层

**完整结构**：
- **初始层**：7×7卷积，64通道，stride=2
- **MaxPool**：3×3，stride=2
- **4个残差层**：
  - Layer1：64通道，2个（ResNet-18）或3个（ResNet-34）基础块
  - Layer2：128通道，2个或4个基础块，stride=2
  - Layer3：256通道，2个或6个基础块，stride=2
  - Layer4：512通道，2个或3个基础块，stride=2
- **全局平均池化** + **全连接层**

**数学表示**：
$$x_0 = \text{Input}$$
$$x_1 = \text{Conv7×7}(x_0)$$
$$x_2 = \text{MaxPool}(x_1)$$
$$x_3 = \text{Layer1}(x_2) = F_1(x_2) + x_2$$
$$x_4 = \text{Layer2}(x_3) = F_2(x_3) + x_3$$
$$x_5 = \text{Layer3}(x_4) = F_3(x_4) + x_4$$
$$x_6 = \text{Layer4}(x_5) = F_4(x_5) + x_5$$
$$y = \text{FC}(\text{AvgPool}(x_6))$$

#### ResNet-50/101/152（深层网络）

**瓶颈块（Bottleneck Block）**：1×1、3×3、1×1卷积

**设计目的**：减少参数数量和计算量

**完整结构**：
- **初始层**：7×7卷积，64通道，stride=2
- **MaxPool**：3×3，stride=2
- **4个残差层**：
  - Layer1：256通道（64×4），3个（ResNet-50）或更多瓶颈块
  - Layer2：512通道（128×4），4个或更多瓶颈块，stride=2
  - Layer3：1024通道（256×4），6个或更多瓶颈块，stride=2
  - Layer4：2048通道（512×4），3个瓶颈块，stride=2
- **全局平均池化** + **全连接层**

**层数计算**：
- ResNet-50：1 + 1 + (3 + 4 + 6 + 3) × 3 + 1 = 50层
- ResNet-101：1 + 1 + (3 + 4 + 23 + 3) × 3 + 1 = 101层
- ResNet-152：1 + 1 + (3 + 8 + 36 + 3) × 3 + 1 = 152层


<br>

## 代码实现

代码实现遵循由简入深的原则，结构如下：

1. **基础残差块**：最简单的残差块实现（BasicBlock）
2. **瓶颈残差块**：更高效的残差块（Bottleneck）
3. **ResNet主网络**：完整的ResNet架构
4. **ResNet变体**：不同深度的ResNet模型
5. **CIFAR-10适配**：针对小数据集的调整版本
6. **训练示例**：完整的训练流程
7. **梯度验证**：验证残差连接的梯度流动优势

### 基础残差块

最基础的残差块实现，包含两个3×3卷积层和快捷连接。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet基础残差块（用于ResNet-18和ResNet-34）"""
    expansion = 1  # 扩展因子，基础块为1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层：W_1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # BN
        
        # 第二个卷积层：W_2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # BN
        
        # 快捷连接（下采样）：W_s（如果需要）
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x  # 快捷连接：x
        
        # 残差路径：F(x) = BN(W_2 σ(BN(W_1 x)))
        out = self.conv1(x)      # W_1 x
        out = self.bn1(out)      # BN(W_1 x)
        out = F.relu(out)        # σ(BN(W_1 x))
        
        out = self.conv2(out)    # W_2 σ(BN(W_1 x))
        out = self.bn2(out)      # BN(W_2 σ(BN(W_1 x)))
        
        # 快捷连接：如果维度不匹配，使用投影
        if self.downsample is not None:
            identity = self.downsample(x)  # W_s x
        
        # 残差相加：y = F(x) + x（或 y = F(x) + W_s x）
        out += identity
        out = F.relu(out)        # σ(F(x) + x)
        
        return out
```

<br>

### 瓶颈残差块

```python
class Bottleneck(nn.Module):
    """ResNet瓶颈残差块（用于ResNet-50/101/152）"""
    expansion = 4  # 扩展因子，瓶颈块为4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1卷积降维：W_1，输出通道 = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷积特征提取：W_2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1卷积升维：W_3，输出通道 = out_channels * 4
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # 快捷连接
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        # 残差路径：F(x) = BN(W_3 σ(BN(W_2 σ(BN(W_1 x)))))
        out = self.conv1(x)      # W_1 x
        out = self.bn1(out)       # BN(W_1 x)
        out = F.relu(out)         # σ(BN(W_1 x))
        
        out = self.conv2(out)    # W_2 σ(BN(W_1 x))
        out = self.bn2(out)       # BN(W_2 σ(BN(W_1 x)))
        out = F.relu(out)         # σ(BN(W_2 σ(BN(W_1 x))))
        
        out = self.conv3(out)    # W_3 σ(BN(W_2 σ(BN(W_1 x))))
        out = self.bn3(out)       # BN(W_3 σ(BN(W_2 σ(BN(W_1 x)))))
        
        # 快捷连接
        if self.downsample is not None:
            identity = self.downsample(x)  # W_s x
        
        # 残差相加
        out += identity
        out = F.relu(out)
        
        return out
```
<br>

### ResNet主网络

```python
class ResNet(nn.Module):
    """ResNet主网络"""
    
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        """
        Args:
            block: 残差块类型（BasicBlock或Bottleneck）
            layers: 每个残差层的块数量，如[2, 2, 2, 2]表示ResNet-18
            num_classes: 分类类别数
            zero_init_residual: 是否将最后一个BN层的权重初始化为0
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # 初始层：7×7卷积，64通道，stride=2
        # 数学：x_1 = σ(BN(Conv7×7(x_0)))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层：x_{l+1} = F_l(x_l) + x_l
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化：确保F(x) ≈ 0（初始状态接近恒等映射）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)  # γ = 1
                nn.init.constant_(m.bias, 0)    # β = 0
        
        # 零初始化残差块的最后一个BN层（可选）
        # 数学：确保初始时F(x) ≈ 0
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建一个残差层
        
        数学对应：
        - 第一个块可能需要下采样：y = F(x) + W_s x
        - 后续块：y = F(x) + x
        """
        downsample = None
        
        # 如果stride不为1或通道数变化，需要下采样
        # 数学：y = F(x) + W_s x（投影映射）
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),  # W_s
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        # 第一个块可能需要下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # 后续块stride=1，不需要下采样
        # 数学：y = F(x) + x（恒等映射）
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        数学对应：
        x_0 = x
        x_1 = σ(BN(Conv7×7(x_0)))
        x_2 = MaxPool(x_1)
        x_3 = Layer1(x_2) = F_1(x_2) + x_2
        x_4 = Layer2(x_3) = F_2(x_3) + x_3
        x_5 = Layer3(x_4) = F_3(x_4) + x_4
        x_6 = Layer4(x_5) = F_4(x_5) + x_5
        y = FC(AvgPool(x_6))
        """
        # 初始层
        x = self.conv1(x)      # Conv7×7(x_0)
        x = self.bn1(x)         # BN(Conv7×7(x_0))
        x = F.relu(x)          # σ(BN(Conv7×7(x_0)))
        x = self.maxpool(x)    # MaxPool(x_1)
        
        # 残差层：每层都有快捷连接
        x = self.layer1(x)     # x = F_1(x) + x
        x = self.layer2(x)     # x = F_2(x) + x
        x = self.layer3(x)     # x = F_3(x) + x
        x = self.layer4(x)     # x = F_4(x) + x
        
        # 分类层
        x = self.avgpool(x)    # AvgPool(x_6)
        x = torch.flatten(x, 1)
        x = self.fc(x)         # FC(AvgPool(x_6))
        
        return x
```
<br>

### ResNet变体

```python
def resnet18(pretrained=False, **kwargs):
    """ResNet-18模型"""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(pretrained=False, **kwargs):
    """ResNet-34模型"""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(pretrained=False, **kwargs):
    """ResNet-50模型"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(pretrained=False, **kwargs):
    """ResNet-101模型"""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(pretrained=False, **kwargs):
    """ResNet-152模型"""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
```
<br>


### CIFAR-10适配版本

CIFAR-10数据集输入尺寸为32×32，因此需要调整ResNet的初始层。
- 初始卷积核从7×7改为3×3
- 初始通道数16
- 3个残差层

```python
class ResNetCIFAR(nn.Module):
    """适用于CIFAR-10的ResNet（输入32x32）"""
    
    def __init__(self, block, layers, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_channels = 16
        
        # CIFAR-10使用较小的初始卷积核（3×3）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 残差层（只有3层）
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        # 权重初始化（与ResNet相同）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """构建残差层（与ResNet的_make_layer方法相同）"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet20_cifar():
    """ResNet-20 for CIFAR-10"""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=10)

def resnet32_cifar():
    """ResNet-32 for CIFAR-10"""
    return ResNetCIFAR(BasicBlock, [5, 5, 5], num_classes=10)

def resnet56_cifar():
    """ResNet-56 for CIFAR-10"""
    return ResNetCIFAR(BasicBlock, [9, 9, 9], num_classes=10)

def resnet110_cifar():
    """ResNet-110 for CIFAR-10"""
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes=10)
```
<br>

### 训练示例

完整的训练流程示例，展示如何使用ResNet进行CIFAR-10分类。

```python
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def train_resnet():
    """完整的训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # 创建模型
    net = resnet20_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    
    # 训练循环
    for epoch in range(200):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')
        
        # 每10个epoch测试一次
        if (epoch + 1) % 10 == 0:
            net.eval()
            correct = total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train_resnet()
```

<br>

### 梯度流动的代码验证

```python
def analyze_gradient_flow(model, x, y):
    """
    分析梯度流动
    
    数学对应：
    - 计算每层的梯度范数
    - 验证：∂L/∂x_l = ∂L/∂x_n ∏(1 + ∂F_i/∂x_i)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # 前向传播
    output = model(x)
    loss = criterion(output, y)
    
    # 反向传播
    loss.backward()
    
    # 分析每层的梯度
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[name] = grad_norm
    
    return grad_norms

# 对比普通网络和残差网络的梯度
def compare_gradients():
    """对比普通网络和残差网络的梯度流动"""
    # 创建数据
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    
    # 普通网络
    plain_net = PlainNet(num_layers=20)
    plain_grads = analyze_gradient_flow(plain_net, x, y)
    
    # 残差网络
    res_net = ResNet(BasicBlock, [3, 3, 3], num_classes=10)
    res_grads = analyze_gradient_flow(res_net, x, y)
    
    # 打印结果
    print("普通网络梯度范数（前几层）：")
    for i, (name, norm) in enumerate(list(plain_grads.items())[:5]):
        print(f"  {name}: {norm:.6f}")
    
    print("\n残差网络梯度范数（前几层）：")
    for i, (name, norm) in enumerate(list(res_grads.items())[:5]):
        print(f"  {name}: {norm:.6f}")
    
    # 分析：残差网络的梯度应该更稳定
    print("\n分析：残差网络通过快捷连接提供了梯度直接传播路径")
```

**预期结果**：
- 普通网络：深层梯度很小（梯度消失）
- 残差网络：梯度更稳定（快捷连接提供直接路径）

<br>

