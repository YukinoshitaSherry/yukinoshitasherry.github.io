---
title: 斯坦福CS229：Machine Learning
date: 2026-09-22
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS229 详细课程笔记：从线性回归、GLM 与 SVM，到泛化、EM、PCA、扩散模型、LLM 与强化学习。
hidden: true
---

- **主讲**
  - 2026 当前主页：[Jehangir Amjad](https://cs229.stanford.edu/)、[Anand Avati](https://cs229.stanford.edu/)；
  - 2025 Autumn：Moses Charikar、Carlos Guestrin、Andrew Ng。
- **入口**
  - [Stanford Bulletin：CS229](https://bulletin.stanford.edu/courses/1057501)
  - [课程主页](https://cs229.stanford.edu/)
  - [Fall 2025 课表](https://cs229.stanford.edu/index.html-fall25)
  - [主讲义 PDF](https://cs229.stanford.edu/main_notes.pdf)
  - [公开作业族（Summer 2020）](https://cs229.stanford.edu/summer2020/)

<br>

> [!INFO]+ 版本范围
>
> Stanford Bulletin 列出：统计模式识别（statistical pattern recognition）、线性与非线性回归、非参数方法（non-parametric methods）、指数族（exponential family）、GLM、支持向量机（support vector machines）、核方法（kernel methods）、深度学习、模型与特征选择、学习理论（learning theory）、ML advice、聚类（clustering）、密度估计（density estimation）、EM、降维（dimensionality reduction）、ICA、PCA、强化学习（reinforcement learning）与自适应控制（adaptive control）、MDP、近似动态规划（approximate dynamic programming）与策略搜索（policy search）。
>
> 算法推导以 Tengyu Ma 与 Andrew Ng 的 *CS229 Lecture Notes*（2026-08-23）为准。决策树、Boosting、评估指标、公平与隐私、LLM 的 RAG 与微调按 2025 Autumn 课表补入。后续学期讲座题目若有调整，回归、GLM、核、SVM、EM、PCA 与 MDP 通常仍在。

<br>


# 总览

- **监督学习（supervised learning）**
  - 线性回归（linear regression）、LMS、正规方程（normal equations）与高斯噪声下的最大似然（maximum likelihood）；
  - 局部加权回归（locally weighted linear regression）；
  - 逻辑回归（logistic regression）、感知机（perceptron）、softmax、牛顿法（Newton's method）；
  - 指数族与广义线性模型（generalized linear models, GLM）；
  - 高斯判别分析（Gaussian discriminant analysis, GDA）与朴素贝叶斯（naive Bayes）；
  - 核方法与支持向量机（support vector machines, SVM）；
  - 决策树（decision trees）与 Boosting。
- **深度与泛化（generalization）**
  - 多层感知机（multilayer perceptron）、模块化反向传播（backpropagation）；
  - 偏差-方差（bias-variance）、双重下降（double descent）、有限/无限假设类（hypothesis class）的样本复杂度（sample complexity）；
  - 正则（regularization）、隐式正则（implicit regularization）、交叉验证（cross-validation）与贝叶斯解释。
- **无监督（unsupervised learning）**
  - $k$-means；
  - 高斯混合（Gaussian mixture model, GMM）与一般 EM、ELBO、变分推断（variational inference）；
  - 因子分析（factor analysis）、PCA 与 ICA。
- **生成与基础模型（foundation models）**
  - 扩散模型（diffusion models）与连续时间反向过程；
  - 表示学习（representation learning）、对比学习（contrastive learning）、检索与 RAG；
  - 分词（tokenization）、自回归（autoregressive）损失、Transformer、MoE、SFT 与推理。
- **控制与决策**
  - MDP、值迭代（value iteration）、策略迭代（policy iteration）、值函数逼近（value function approximation）；
  - LQR、DDP、LQG；
  - REINFORCE 与 PPO。
- **课堂扩展**
  - 数据划分与 ML advice；
  - 评估指标（evaluation metrics）；
  - 公平性（fairness）、算法偏差（algorithmic bias）、可解释性（interpretability）与隐私（privacy）。

- **记录逻辑**

$$
\text{任务与数据}
\rightarrow\text{假设类 }
h
\rightarrow\text{损失或似然}
\rightarrow\text{约束或先验}
\rightarrow\text{求解器}
\rightarrow\text{泛化诊断}
$$

- **分数校准**
  - 训练 $J(\theta)$ 下降不等于测试风险下降；
  - 准确率在类别不平衡时会掩盖少数类错误；
  - AUC、F1、校准误差、回报与约束违反各自对应不同代价。
- **实现要求**
  - 能够从零实现正规方程、逻辑回归牛顿法、GDA、核 SVM 对偶、GMM-EM、PCA 与值迭代；
  - 能够说明成熟库中求解器、核缓存、EM 责任度、以及策略梯度基线的实际语义。
- **主线**
  - **前半：目标与求解**
    - 写清优化对象、梯度与是否存在闭式解。
  - **后半：误差来源**
    - 区分偏差、方差、模型设定错误、优化失败与标注噪声；
    - 间隔分类看支持向量，概率模型看校准，聚类看局部最优，PCA 看保留方差。

$$
\begin{aligned}
\text{最小二乘}
&\rightarrow\text{指数族 GLM}
\rightarrow\text{逻辑回归 / softmax},\\
\text{高斯条件}
&\rightarrow\text{GDA / 朴素贝叶斯}
\rightarrow\text{生成-判别对照},\\
\text{特征映射}
&\rightarrow\text{核}
\rightarrow\text{SVM 对偶},\\
\text{完全数据似然}
&\rightarrow\text{ELBO}
\rightarrow\text{EM / 因子分析 / 变分},\\
\text{MDP 贝尔曼}
&\rightarrow\text{值迭代}
\rightarrow\text{策略梯度}.
\end{aligned}
$$

> [!NOTE]+ 概念链
>
> - 记号：$X\in\mathbb{R}^{n\times(d+1)}$，截距在 $\theta_0$。
> - 线性回归：平方损失、LMS、正规方程、高斯 MLE。
> - 逻辑回归：Bernoulli 似然，梯度仍是残差乘特征。
> - GLM：指数族加上 $\eta=\theta^{\top}x$。
> - 生成模型：先建 $p(x\mid y)$，再由贝叶斯得后验。
> - 核：算法只出现 $K(x,z)=\phi(x)^{\top}\phi(z)$。
> - SVM：同一特征空间上的最大间隔。
> - 树做轴对齐切分；Boosting 把弱分类器相加。
> - 网络用可微激活，反向传播同时更新各层。
> - 泛化比较 $R$ 与 $\hat R_n$；正则与交叉验证用来控制容量。
> - 无标签：$k$-means、EM、PCA、ICA。
> - 扩散用 ELBO 做去噪；其后是表示适配与语言建模。
> - RL 没有固定 $(x,y)$。LQR 给出线性二次的反馈；PPO 是带裁剪的策略梯度。
> - 公平与实践处理分组指标、划分与项目流程。




<br>



## 数学预备

课程反复使用六类数学对象。

$$
J(\theta)=\frac{1}{2}\sum_{i=1}^{n}\big(h_{\theta}(x^{(i)})-y^{(i)}\big)^{2}
$$

- **经验风险（empirical risk）：把数据集上的损失之和当作可微目标**
  - 单样本残差描述预测与标签的局部代价；
  - 参数更新针对 $J$ 或对数似然 $\ell$，而不是针对单个样本残差本身；
  - CS229 的线性回归常用 $1/2$ 而不除以 $n$，使 $\nabla J$ 少一个常数，正规方程少一次缩放。
- **梯度与 Hessian**
  - 梯度 $\nabla_{\theta}J$ 指向上升最快方向，下降更新减它；
  - Hessian $H_{jk}=\partial^{2}J/\partial\theta_j\partial\theta_k$ 描述曲率；
  - 牛顿法用 $H^{-1}$ 校正步长，二次函数一步到位。
- **最大似然（maximum likelihood）**
  - 先写 $p(y\mid x;\theta)$，再写 $L(\theta)=\prod_i p(y^{(i)}\mid x^{(i)};\theta)$；
  - 对数把乘积变求和：$\ell(\theta)=\log L(\theta)$；
  - 高斯噪声下最大化 $\ell$ 等价于最小化平方损失。
- **线性代数**
  - 设计矩阵（design matrix）$X\in\mathbb{R}^{n\times(d+1)}$ 的每一行是一个样本；
  - $X^{\top}X\theta=X^{\top}y$ 是法方程，不是“把 $X$ 除掉”；
  - 对称半正定矩阵的特征值全非负，这是凸二次的来源。
- **概率与指数族（exponential family）**
  - 伯努利、高斯、泊松都能写成 $b(y)\exp(\eta^{\top}T(y)-a(\eta))$；
  - 自然参数（natural parameter）$\eta$ 与均值的关系给出 sigmoid、恒等、指数等响应函数。
- **约束优化与对偶（duality）**
  - 原问题加不等式约束后，Lagrangian 引入乘子 $\alpha_i\ge 0$；
  - KKT 条件把“哪些点贴在间隔边界上”变成支持向量（support vectors）。

## 复杂度

- **时间：先问哪一个维度增长**
  - 正规方程：形成 $X^{\top}X$ 为 $O(nd^{2})$，分解为 $O(d^{3})$；
  - 批量梯度：每步 $O(nd)$；随机梯度每步 $O(d)$；
  - 逻辑回归牛顿法：每步形成 Hessian $O(nd^{2})$ 再求逆 $O(d^{3})$；
  - 核方法：核矩阵 $K\in\mathbb{R}^{n\times n}$，存贮 $O(n^{2})$，许多求解器至少 $O(n^{2})$；
  - GMM-EM：$k$ 个分量，每轮 $O(nkd^{2})$（若协方差完整）；
  - PCA：对 $d\times d$ 协方差特征分解 $O(d^{3})$，或对 $X$ 做 SVD；
  - 值迭代：状态 $S$、动作 $A$，每轮 $O(|S|^{2}|A|)$。
- **空间**
  - 参数模型只需存 $\theta$，预测不必再看训练集；
  - 非参数模型（局部加权、核 SVM）预测时仍要访问训练点或支持向量；
  - 核矩阵 $n=10^{5}$ 时双精度约 $80$ GB，必须用稀疏或 SMO 式分解。

## 数值计算

- $X^{\top}X$ 接近奇异时不要写 `inv(X.T @ X)`，改用 `np.linalg.lstsq` 或加 $\lambda I$。
- 概率与指数在未归一化 logits 上计算，不是先 `exp` 再求和。
- softmax 应减去每行最大值。
- sigmoid 在大正负输入上应改写为稳定形式。
- 对数似然中的 $\log 0$ 必须用 $\varepsilon$ 截断，但不能用截断代替正确的 Bernoulli 模型。
- EM 的责任度在分量几乎不重叠时会下溢，应在 log-space 用 log-sum-exp。
- 比较不同长度序列的 log-likelihood 前，说明是否按 token 数归一化。

## Python 实现基础

- **角色**
  - NumPy 用来验证算法：先在小数组上对齐 shape 与梯度，再交给求解器或框架；
  - 数值内核优先 broadcasting 与线性代数，避免在样本维上写 Python `for` loop；
  - 向量化不改变渐近复杂度，只是把循环移到底层 BLAS。
- **索引**
  - 整数索引消去一个维度；切片通常保留该维度；
  - `X[0]` 的 shape 是 `(d+1,)`；`X[0:1]` 的 shape 是 `(1, d+1)`；
  - 把前者与 `(n, d+1)` 相加会触发意外广播。
- **数值与随机**
  - `/` 总是浮点除法；
  - 随机性来自 `numpy.random`；
  - 复现实验必须固定 seed。
- **广播**
  - 从尾维对齐；维长相等、其中一方为 $1$、或该维不存在时可以扩展；
  - `(n, 1)` 权重与 `(n,)` 残差相乘得到逐样本加权。

```python
import numpy as np

def add_intercept(X):
    """X: (n, d) -> (n, d+1)，首列全 1。"""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])

def assert_design_matrix(X, name="X"):
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {X.shape}")
    if not np.allclose(X[:, 0], 1.0):
        raise ValueError(f"{name}[:, 0] should be intercept ones")
```

- **计算图**
  - 节点：$z=\theta^{\top}x$，$h=g(z)$，$\mathcal{L}(h,y)$；
  - 前向沿拓扑序求值并缓存 $x,z,h$；
  - 反向从 $\mathrm{d}\mathcal{L}$ 出发，每个节点只用局部导数。



<br>



## 概念记号

- **问题**
  - 监督学习（supervised learning）把输入 $x$ 映射到标签 $y$；
  - 假设（hypothesis）$h_{\theta}:\mathcal{X}\rightarrow\mathcal{Y}$ 由有限参数 $\theta$ 描述，或由训练集本身描述（非参数，non-parametric）；
  - 记号不统一时，正规方程与对偶公式无法逐行核对。
- **范围**
  - 固定 CS229 讲义中的上下标约定；
  - 输入是原始特征，输出是预测 $h_{\theta}(x)$ 与标量目标 $J$ 或 $\ell$。
- **章节衔接**
  - 本章固定样本、特征与参数的 shape；
  - 下一章的线性回归是该记号下最简单的可训练模型。

上标圆括号表示样本，不是幂：

$$
x^{(i)}\in\mathbb{R}^{d+1},\qquad
y^{(i)}\in\mathbb{R},\qquad
\theta\in\mathbb{R}^{d+1}.
$$

- $n$：训练样本数。
- $d$：输入特征维（不含截距）。
- $x_0^{(i)}=1$：截距位，于是 $h_{\theta}(x)=\theta^{\top}x=\sum_{j=0}^{d}\theta_j x_j$。
- $X\in\mathbb{R}^{n\times(d+1)}$：第 $i$ 行是 $(x^{(i)})^{\top}$。
- $\vec{y}\in\mathbb{R}^{n}$：标签列向量。
- $h_{\theta}$：假设（hypothesis），历史名词，就是预测函数。
- $\mathcal{X},\mathcal{Y}$：输入空间与输出空间。
- 回归：$\mathcal{Y}=\mathbb{R}$。分类：$\mathcal{Y}$ 有限，二分类常取 $\{0,1\}$；SVM 章改用 $\{-1,+1\}$。

一个样本的线性预测：

$$
h_{\theta}(x^{(i)})=\theta^{\top}x^{(i)}.
$$

整批一次计算：

$$
X\theta-\vec{y}
=
\begin{pmatrix}
h_{\theta}(x^{(1)})-y^{(1)} \\
\vdots \\
h_{\theta}(x^{(n)})-y^{(n)}
\end{pmatrix}.
$$

- 实现注释中写明每个数组的 shape。
- 若 $X$ 被误写成 $(d+1,n)$，后续所有公式都会差一个转置。

逐层对照表（线性回归）：

| 量 | shape |
| :--- | :--- |
| $X$ | $(n, d+1)$ |
| $\theta$ | $(d+1,)$ 或 $(d+1, 1)$ |
| $X\theta$ | $(n,)$ 或 $(n, 1)$ |
| $\vec{y}$ | 与 $X\theta$ 相同 |
| $\nabla_{\theta}J$ | 与 $\theta$ 相同 |
| $X^{\top}X$ | $(d+1, d+1)$ |

- **命名**
  - $\nabla_{\theta}J$ 在代码里常叫 `grad`，不是 $J$ 的微分形式；
  - 实现时用 `assert grad.shape == theta.shape` 卡住转置错误。
- **常用恒等式**

$$
\nabla_{x}(b^{\top}x)=b,\qquad
\nabla_{x}(x^{\top}Ax)=2Ax\quad(A=A^{\top}),\qquad
z^{\top}z=\sum_{i}z_{i}^{2}.
$$

> [!NOTE]+ 衔接
>
> 记号固定 $X$ 为 $(n,d+1)$、$\theta$ 含截距。下一章把 $h_{\theta}(x)=\theta^{\top}x$ 配上平方损失，导出 LMS 与正规方程。

<br>

# 线性回归

## 概念：线性回归

- **问题**
  - 回归（regression）：$y\in\mathbb{R}$，输入 $x\in\mathbb{R}^{d+1}$（已含 $x_0=1$）；
  - 模型 $h_{\theta}(x)=\theta^{\top}x$，预测房价、计数的连续近似等。
- **范围**
  - 由平方损失导出 LMS 更新、正规方程（normal equations），以及高斯噪声下的最大似然（maximum likelihood）；
  - 输入 $(X,\vec{y})$，输出 $\theta$ 与预测。
- **章节衔接**
  - 上一章固定 $X\theta$ 的乘法顺序；
  - 本章给出“损失 → 梯度 → 更新 / 闭式解”的完整模板；
  - 下一章只改响应与似然：$h$ 换成 sigmoid，平方损失换成对数似然。

Portland 房价是讲义中的标准例子：居住面积与价格成对出现。把面积记为 $x_1$，价格记为 $y$，学习目标是得到 $h:\mathcal{X}\rightarrow\mathcal{Y}$，使 $h(x)$ 对新房子仍是好的预测。$y$ 连续时称为回归；$y$ 只取少数离散值时称为分类。

加入卧室数后，$x^{(i)}\in\mathbb{R}^{2}$，再补截距：

$$
h_{\theta}(x)=\theta_0+\theta_1 x_1+\theta_2 x_2=\theta^{\top}x.
$$

$\theta_j$ 称为参数或权重。特征选什么由建模者决定：壁炉、浴室数都可以进 $x$，但必须在训练与预测时使用同一套定义。

## 损失

$$
J(\theta)=\frac{1}{2}\sum_{i=1}^{n}\big(h_{\theta}(x^{(i)})-y^{(i)}\big)^{2}.
$$

- $1/2$ 使后面求导时消去平方带来的 $2$，不是概率里的归一化。
- $J$ 对 $\theta$ 是凸二次：等高线是椭圆，只有一个全局最小。
- 若特征尺度相差几个数量级，椭圆被拉成长谷，梯度下降会走之字形。

> [!EXAMPLE]+ 例1 两个样本的手工计算
>
> 取 $x^{(1)}=(1,1)$，$y^{(1)}=1$；$x^{(2)}=(1,2)$，$y^{(2)}=2$；$\theta=(0,0)$。则 $h=(0,0)$，$J=\frac12(1^{2}+2^{2})=2.5$。若改 $\theta=(0,1)$，则 $h=(1,2)$，$J=0$。该数据集上最优就是过原点斜率 $1$ 的直线。真实作业里 $n\gg d$，不能靠观察得到 $\theta$。

<br>

## LMS

目标是最小化 $J(\theta)$。梯度下降从初始 $\theta$ 出发，对每个坐标同时执行

$$
\theta_j \leftarrow \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta).
$$

$\alpha>0$ 是学习率（learning rate）。需要先算出偏导。先看单样本、略去求和：

$$
\begin{aligned}
\frac{\partial}{\partial\theta_j}J(\theta)
&=
\frac{\partial}{\partial\theta_j}\frac12\big(h_{\theta}(x)-y\big)^{2}
=
\big(h_{\theta}(x)-y\big)\frac{\partial}{\partial\theta_j}\big(\theta^{\top}x-y\big)
=
\big(h_{\theta}(x)-y\big)x_j.
\end{aligned}
$$

因此单样本更新是

$$
\theta_j \leftarrow \theta_j + \alpha\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}_{j}.
$$

该式称为 LMS（least mean squares）更新，也叫 Widrow-Hoff 规则。误差 $y-h$ 大则步子大；已经拟合好的点几乎不推动 $\theta$。

> [!NOTE]+ LMS
>
> 是最小均方意义下的随机梯度规则：每看到一个样本，沿 $\nabla_{\theta}(h-y)^{2}/2$ 的反方向走一步。来源是 1960 年代的自适应滤波（Widrow-Hoff），不是为闭式最小二乘新造的算法。相对一次性分解 $X^{\top}X$，它不要求全部数据同时在内存里，也适用于样本陆续到达的情形。$\alpha$ 选不好会振荡或发散，这是用迭代换闭式解必须付的代价。

<br>

> [!INFO]+ 从平方到 $x_j$ 的每一步
>
> 令 $r=h_{\theta}(x)-y=\sum_{k=0}^{d}\theta_k x_k-y$。则 $J=r^{2}/2$。复合函数求导：$\partial J/\partial\theta_j=r\cdot\partial r/\partial\theta_j$。而 $\partial r/\partial\theta_j=x_j$，因为只有第 $j$ 项含 $\theta_j$，且系数为 $x_j$。符号写成 $\theta\leftarrow\theta+\alpha(y-h)x$ 时，加号来自把 $-\partial J$ 与 $(h-y)$ 中的负号相消。实现时不要把两个负号再乘一次。
>
> 感性理解：残差 $r$ 对 $\theta_j$ 的依赖只经过 $x_j$，故偏导多一个因子 $x_j$。特征尺度相差一个数量级时，对应坐标的梯度也会差一个数量级，面积会压过卧室数。标准化是为了让各坐标的步长可比较，不是为了改变最小二乘的几何。

<br>

多样本有两种用法。

**批量梯度下降（batch gradient descent）**每一步用全体样本：

$$
\theta \leftarrow \theta + \alpha\sum_{i=1}^{n}\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}.
$$

这就是对原始 $J$ 做梯度下降。线性回归的 $J$ 是凸二次，只要 $\alpha$ 不太大，批量法收敛到全局最小。讲义在面积-价格数据上得到 $\theta_0\approx 71.27$，$\theta_1\approx 0.1345$；再加入卧室数后 $\theta_0\approx 89.60$，$\theta_1\approx 0.1392$，$\theta_2\approx -8.738$。

**随机（增量）梯度下降（stochastic / incremental gradient descent）**每次只看一个样本：

$$
\theta \leftarrow \theta + \alpha\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}.
$$

批量法必须扫完整个训练集才走一步；$n$ 很大时昂贵。随机法每看到一个点就更新，往往更快靠近最小点，但会在最小点附近振荡。实践中可令 $\alpha$ 缓慢衰减到 $0$，以保证收敛而不只是振荡。

## 正规方程（normal equations）

梯度下降是迭代求解。也可以直接令导数为零。

> [!NOTE]+ 正规方程
>
> 令 $\nabla_{\theta}J=0$ 得到的线性系统 $X^{\top}X\theta=X^{\top}\vec{y}$。名称里的 “normal” 来自残差与 $X$ 的列正交（normal equations），不是“普通方程”。相对 LMS：二次凸目标有唯一驻点时，一次分解即可，不必调 $\alpha$。相对后来的岭回归：不加 $\lambda I$ 时，$X^{\top}X$ 奇异则解不唯一。$n$ 或 $d$ 很大、或需要流式更新时，仍应回到迭代法。

<br>

先定义矩阵导数：对 $f:\mathbb{R}^{n\times d}\rightarrow\mathbb{R}$，

$$
\nabla_{A}f(A)
=
\Big(\frac{\partial f}{\partial A_{ij}}\Big)_{ij}.
$$

例如 $A\in\mathbb{R}^{2\times 2}$ 且 $f=\frac32 A_{11}+5A_{12}^{2}+A_{21}A_{22}$，则

$$
\nabla_{A}f
=
\begin{pmatrix}
3/2 & 10A_{12} \\
A_{22} & A_{21}
\end{pmatrix}.
$$

设计矩阵与标签：

$$
X
=
\begin{pmatrix}
(x^{(1)})^{\top} \\
\vdots \\
(x^{(n)})^{\top}
\end{pmatrix},\qquad
\vec{y}
=
\begin{pmatrix}
y^{(1)} \\
\vdots \\
y^{(n)}
\end{pmatrix}.
$$

因为 $h_{\theta}(x^{(i)})=(x^{(i)})^{\top}\theta$，有 $X\theta-\vec{y}$ 的第 $i$ 个分量恰为残差。对任意向量 $z$，$z^{\top}z=\sum_i z_i^{2}$，故

$$
J(\theta)=\frac12(X\theta-\vec{y})^{\top}(X\theta-\vec{y}).
$$

> [!INFO]+ 正规方程推导
>
> 展开：
>
> $$
> J(\theta)=\frac12\Big(\theta^{\top}X^{\top}X\theta-2(X^{\top}\vec{y})^{\top}\theta+\vec{y}^{\top}\vec{y}\Big).
> $$
>
> $\vec{y}^{\top}\vec{y}$ 不含 $\theta$。对对称矩阵 $A=X^{\top}X$ 使用 $\nabla_{\theta}(\theta^{\top}A\theta)=2A\theta$，以及对 $\nabla_{\theta}(b^{\top}\theta)=b$：
>
> $$
> \nabla_{\theta}J(\theta)=X^{\top}X\theta-X^{\top}\vec{y}.
> $$
>
> 令梯度为零，得到法方程 $X^{\top}X\theta=X^{\top}\vec{y}$。若 $X^{\top}X$ 可逆，
>
> $$
> \theta=(X^{\top}X)^{-1}X^{\top}\vec{y}.
> $$
>
> $X^{\top}X$ 不可逆的常见原因：独立样本数少于特征数，或特征线性相关（例如同时放入华氏与摄氏温度）。此时应删共线特征、用伪逆，或加 $\lambda I$（岭回归）。
>
> 感性理解：$X\theta$ 是用 $X$ 的各列（截距、面积、卧室数）去拼一个最接近 $\vec{y}$ 的向量。拼完后的残差必须与每一列都垂直，否则还能沿某一列再挪一点，使平方误差继续下降。垂直条件写成矩阵就是 $X^{\top}(\vec{y}-X\theta)=0$，整理即正规方程。$(X^{\top}X)^{-1}X^{\top}$ 是把 $\vec{y}$ 投影到列空间的算子，不是“把 $X$ 除掉”。

<br>

几何图像：$X\theta$ 是 $X$ 列空间中的一点；正规方程说残差 $\vec{y}-X\theta$ 与每一列正交，即预测是 $\vec{y}$ 到列空间的正交投影。

## 概率解释（probabilistic interpretation）

假设

$$
y^{(i)}=\theta^{\top}x^{(i)}+\varepsilon^{(i)},\qquad
\varepsilon^{(i)}\stackrel{\mathrm{iid}}{\sim}\mathcal{N}(0,\sigma^{2}).
$$

$\varepsilon$ 吸收未建模效应与噪声。密度为

$$
p(\varepsilon)=\frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac{\varepsilon^{2}}{2\sigma^{2}}\Big),
$$

因而

$$
p(y^{(i)}\mid x^{(i)};\theta)
=
\frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac{(y^{(i)}-\theta^{\top}x^{(i)})^{2}}{2\sigma^{2}}\Big).
$$

记号 $p(y\mid x;\theta)$ 表示给定 $x$、由 $\theta$ 参数化的分布；$\theta$ 不是随机变量，故不写成 $p(y\mid x,\theta)$。似然把上式看成 $\theta$ 的函数：

$$
L(\theta)=p(\vec{y}\mid X;\theta)=\prod_{i=1}^{n}p(y^{(i)}\mid x^{(i)};\theta).
$$

最大似然选择使数据概率最大的 $\theta$。对数似然

$$
\ell(\theta)
=
n\log\frac{1}{\sqrt{2\pi}\sigma}
-\frac{1}{\sigma^{2}}\cdot\frac12\sum_{i=1}^{n}(y^{(i)}-\theta^{\top}x^{(i)})^{2}.
$$

第一项与 $\theta$ 无关。最大化 $\ell$ 等价于最小化原来的 $J(\theta)$。因此：在高斯 IID 噪声下，最小二乘就是 $\theta$ 的最大似然估计。$\sigma^{2}$ 不出现在最优 $\theta$ 中；即使 $\sigma^{2}$ 未知，同一 $\theta$ 仍然最优。该事实在指数族与 GLM 中会再用到。

> [!INFO]+ 从高斯密度到平方损失
>
> 单条密度是 $p(y\mid x;\theta)\propto\exp\big(-(y-\theta^{\top}x)^{2}/(2\sigma^{2})\big)$。独立样本的联合密度是各条相乘，取对数后指数变成求和：
>
> $$
> \ell(\theta)=\mathrm{const}(\sigma)-\frac{1}{2\sigma^{2}}\sum_{i}(y^{(i)}-\theta^{\top}x^{(i)})^{2}.
> $$
>
> $\sigma$ 只出现在与 $\theta$ 无关的常数，以及整体缩放 $1/\sigma^{2}$ 里。对 $\theta$ 求最大，等价于最小化 $\sum(y-\theta^{\top}x)^{2}$，也就是原来的 $J$。
>
> 感性理解：高斯密度的指数里是 $-(y-\theta^{\top}x)^{2}/(2\sigma^{2})$。最大化对 $\theta$ 的似然，等价于最小化平方残差。$\sigma$ 只缩放目标而不移动驻点，故最优 $\theta$ 与噪声方差无关。若噪声是拉普拉斯，同一推导给出绝对偏差而不是平方。

<br>

> [!WARNING]+ 高斯假设的适用范围
>
> 高斯噪声足以推出平方损失对应最大似然，但并非必要。噪声重尾时，平方损失会被离群点主导，绝对偏差或 Huber 损失更稳。似然模型设错时，MLE 仍会给出一个估计，该估计不必对应真实的数据生成过程。

<br>

## 局部加权（locally weighted）

全局拟合一条直线会欠拟合弯曲数据；五次多项式又会过拟合。局部加权线性回归（LWR）在查询点 $x$ 处重新加权：

1. 最小化 $\sum_{i}w^{(i)}(y^{(i)}-\theta^{\top}x^{(i)})^{2}$；
2. 输出 $\theta^{\top}x$。

常用权重

$$
w^{(i)}=\exp\Big(-\frac{\|x^{(i)}-x\|_{2}^{2}}{2\tau^{2}}\Big).
$$

靠近查询点的样本权重大，远处几乎被忽略。$\tau$ 是带宽：太小则只看邻近两三个点，曲线锯齿；太大则退回普通最小二乘。向量情形可把欧氏距离换成马氏距离 $(x^{(i)}-x)^{\top}\Sigma^{-1}(x^{(i)}-x)$。

闭式解是加权正规方程：

$$
\theta=(X^{\top}WX)^{-1}X^{\top}W\vec{y},\qquad
W=\mathrm{diag}(w^{(1)},\ldots,w^{(n)}).
$$

普通线性回归是参数方法：$\theta$ 个数固定，拟合后可丢掉训练集。LWR 是非参数方法：表示假设所需的存贮随 $n$ 线性增长，每次预测都要重拟合。权重公式看起来像高斯密度，但 $w^{(i)}$ 不是随机变量。

> [!NOTE]+ 局部加权回归
>
> 在查询点附近重新做一次加权最小二乘，而不是先学一条全局直线再到处用。来自核回归 / Nadaraya-Watson 一类非参数思想：远处的点对局部斜率贡献应接近 $0$。相对全局线性，它能跟上缓弯；相对全局高次多项式，它不在全域抬高次数。代价是每个查询都要重解，且 $\tau$ 必须用验证集选，否则锯齿或退回普通最小二乘。

<br>

## 算法

1. 构造 $X$：原始特征按列排列，左侧加全 $1$ 列。
2. 检查列尺度；必要时用训练集均值方差标准化，测试集复用同一统计量。
3. $n$ 不大且 $X^{\top}X$ 良态：用 `lstsq` 解正规方程。
4. $n$ 很大：批量或小批量梯度；监控 $J$，过大则减小 $\alpha$。
5. 查询点附近需要弯曲拟合：对每个查询做一次 LWR，$\tau$ 与模型系数一样需要用验证集选定。

- 正规方程时间 $O(nd^{2}+d^{3})$。
- 批量每步 $O(nd)$，随机每步 $O(d)$。
- LWR 每个查询 $O(nd^{2}+d^{3})$。

## 实现

```python
import numpy as np

def linear_regression_normal_eq(X, y, reg=0.0):
    """X: (n, d+1), y: (n,)。reg 为岭参数 λ，0 表示普通最小二乘。"""
    A = X.T @ X
    if reg:
        A = A + reg * np.eye(A.shape[0])
    theta, *_ = np.linalg.lstsq(A, X.T @ y, rcond=None)
    return theta

def batch_lms(X, y, lr=1e-8, num_steps=10000):
    """批量 LMS。房价等未标准化特征需要很小的 lr。"""
    n, d1 = X.shape
    theta = np.zeros(d1)
    history = []
    for _ in range(num_steps):
        residual = X @ theta - y
        history.append(0.5 * float(residual @ residual))
        theta -= lr * (X.T @ residual)
    return theta, history

def stochastic_lms(X, y, lr=1e-8, epochs=50, rng=None):
    rng = np.random.default_rng(rng)
    n, d1 = X.shape
    theta = np.zeros(d1)
    for _ in range(epochs):
        for i in rng.permutation(n):
            theta += lr * (y[i] - X[i] @ theta) * X[i]
    return theta

def locally_weighted(X, y, x_query, tau):
    """在 x_query 处做一次 LWR。X, x_query 均已含截距。"""
    diff = X - x_query
    w = np.exp(-np.sum(diff * diff, axis=1) / (2.0 * tau ** 2))
    WX = X * w[:, None]
    theta, *_ = np.linalg.lstsq(X.T @ WX, X.T @ (w * y), rcond=None)
    return float(x_query @ theta), theta
```

> [!EXAMPLE]+ 例2 正规方程与梯度必须对齐
>
> 随机生成 $X\in\mathbb{R}^{20\times 3}$、$y=X\theta^{\ast}+\varepsilon$。`lstsq` 得到的 $\theta$ 与批量 LMS 在足够步数、足够小 $\alpha$ 后应在 $10^{-4}$ 内一致。若不一致，先查 $X$ 是否漏了截距列，再查更新写成了 `theta += lr * X.T @ residual`（符号反了）。

<br>

> [!NOTE]+ 对照：参数与非参数
>
> 普通最小二乘拟合后只保留 $\theta$，训练集可以丢掉。局部加权在每个查询点重解一次 $\theta$，存贮随 $n$ 增长。后面的核 SVM 也是非参数：决策依赖支持向量。神经网络又回到参数：层宽固定后，预测不必扫描全部训练点。

<br>

> [!NOTE]+ 衔接
>
> 线性回归在高斯噪声下是最大似然。下一章把 $y$ 换成 $\{0,1\}$，似然换成 Bernoulli，预测换成 $g(\theta^{\top}x)$。梯度仍写成残差乘特征，只是残差里的 $h$ 不再是 $\theta^{\top}x$。

<br>

# 逻辑回归

## 概念：逻辑回归

- **问题**
  - 二分类（binary classification）：$y\in\{0,1\}$，输入 $x\in\mathbb{R}^{d+1}$；
  - 模型输出 $h_{\theta}(x)=g(\theta^{\top}x)\in(0,1)$，解释为 $P(y=1\mid x;\theta)$。
- **范围**
  - 由 Bernoulli 最大似然导出对数损失与梯度，并写出牛顿法（Newton's method）；
  - 输入 $(X,\vec{y})$，输出 $\theta$ 与类别预测。
- **章节衔接**
  - 上一章是高斯噪声下的平方损失；
  - 本章是 Bernoulli 下的对数似然，梯度外形仍是残差乘 $x$；
  - 下一章用指数族说明这两次选择不是两套无关算法。

直接把线性回归用在 $\{0,1\}$ 标签上会出现 $h>1$ 或 $h<0$，且离群的 $x$ 会把直线拽歪。改用 logistic / sigmoid

$$
g(z)=\frac{1}{1+e^{-z}},\qquad
h_{\theta}(x)=g(\theta^{\top}x).
$$

- $z\rightarrow+\infty$ 时 $g\rightarrow 1$；$z\rightarrow-\infty$ 时 $g\rightarrow 0$。
- $g(z)+g(-z)=1$。
- $g'(z)=g(z)(1-g(z))$。
- $|z|$ 很大时饱和，梯度接近 $0$。

> [!NOTE]+ 逻辑回归
>
> 把 $P(y=1\mid x)$ 写成 $\sigma(\theta^{\top}x)$，用 Bernoulli 对数似然训练。来源是把线性分数压进 $(0,1)$，而不是先发明 S 形再找概率解释。相对把线性回归直接套在 $\{0,1\}$ 上：输出不再越出区间，离群 $x$ 对直线的杠杆也小得多。相对感知机：优化的是似然而不是仅改错分，因而给出校准概率。线性可分且无正则时 $\|\theta\|$ 仍会发散。

<br>

## 似然

$$
P(y=1\mid x;\theta)=h_{\theta}(x),\qquad
P(y=0\mid x;\theta)=1-h_{\theta}(x).
$$

紧凑写法：

$$
p(y\mid x;\theta)=h_{\theta}(x)^{y}\big(1-h_{\theta}(x)\big)^{1-y}.
$$

独立样本的似然与对数似然：

$$
\begin{aligned}
L(\theta)&=\prod_{i=1}^{n}h_{\theta}(x^{(i)})^{y^{(i)}}\big(1-h_{\theta}(x^{(i)})\big)^{1-y^{(i)}},\\
\ell(\theta)&=\sum_{i=1}^{n}\Big[y^{(i)}\log h_{\theta}(x^{(i)})+(1-y^{(i)})\log\big(1-h_{\theta}(x^{(i)})\big)\Big].
\end{aligned}
$$

最大化 $\ell$ 等价于最小化平均二元交叉熵（binary cross-entropy）。$y=1$ 时只惩罚 $-\log h$；$y=0$ 时只惩罚 $-\log(1-h)$。

> [!INFO]+ 梯度为什么是 $(h-y)x$
>
> 令 $z=\theta^{\top}x$，$h=g(z)$。单样本
>
> $$
> \frac{\partial\ell}{\partial h}=\frac{y}{h}-\frac{1-y}{1-h},\qquad
> \frac{\partial h}{\partial z}=h(1-h).
> $$
>
> 相乘：
>
> $$
> \frac{\partial\ell}{\partial z}
> =
> \Big(\frac{y}{h}-\frac{1-y}{1-h}\Big)h(1-h)
> =
> y-h.
> $$
>
> 再乘 $\partial z/\partial\theta=x$，得 $\nabla_{\theta}\ell=(y-h)x$。批量下降最大化 $\ell$ 时
>
> $$
> \theta\leftarrow\theta+\alpha\sum_{i=1}^{n}\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}.
> $$
>
> 与 LMS 外形相同，但 $h$ 是 sigmoid 而不是 $\theta^{\top}x$。若实现最小化交叉熵 $J=-\ell/n$，则梯度反号：$\nabla_{\theta}J=\frac1n X^{\top}(h-y)$。
>
> 不要用平方损失配 sigmoid：$\partial\mathcal{L}/\partial z=(h-y)h(1-h)$，预测极错时 $h(1-h)\rightarrow 0$，梯度消失。交叉熵在同一极限下 $\partial J/\partial z$ 仍约为 $h-y$，信号不消失。平方损失对 $\theta$ 非凸；Bernoulli 对数似然对 $\theta$ 是凹的，因而 $J$ 凸。
>
> 感性理解：交叉熵的梯度方向仍是特征 $x$，幅度是概率误差 $h-y$。$h(1-h)$ 在链式法则里与分母相消，因此 $h\rightarrow 0$ 且 $y=1$ 时 $\partial J/\partial z$ 仍接近 $-1$。平方损失保留 $h(1-h)$，同一极限下梯度消失。配对损失必须和输出激活一起选，不能把回归用的平方损失直接套在 sigmoid 上。

<br>

## 感知机（perceptron）

把 $g$ 换成硬阈值：$g(z)=1$（$z\ge 0$）否则 $0$，再套同一更新，得到感知机学习规则。感知机不估计概率，只推动被分错的点。线性可分时存在有限步收敛到分隔超平面；不可分时振荡。它是 SVM 与神经网络历史中的祖先，不是现代默认分类器。

> [!NOTE]+ 感知机
>
> 错分时把 $y x$ 加到 $\theta$ 上，分对则不动。来源是 Rosenblatt 的线性阈值单元，更新与 LMS 外形相近，但 $h$ 是硬阈值而不是线性或 sigmoid。相对逻辑回归：不给出 $P(y=1\mid x)$，也没有似然可最大化。相对 SVM：只要可分就停，不最大化间隔。线性不可分时振荡，不能当默认分类器。

<br>

## 多类

$k$ 类时用 softmax（也叫多项逻辑回归）：

$$
h_{\theta}(x)_{c}
=
p(y=c\mid x;\theta)
=
\frac{\exp(\theta_c^{\top}x)}{\sum_{j=1}^{k}\exp(\theta_j^{\top}x)}.
$$

其中一个 $\theta_c$ 可固定为 $0$ 以消除平移不确定性。对数似然是

$$
\ell(\theta)=\sum_{i=1}^{n}\log h_{\theta}(x^{(i)})_{y^{(i)}}.
$$

对正确类的 logits 求导后，输出层误差仍是 $\hat{p}-e_y$（one-hot）。一对多（one-vs-rest）训练 $k$ 个独立二分类器，分数不可比；softmax 在同一归一化下比较 $k$ 类，后验可直接对照。

> [!INFO]+ softmax 的梯度为何仍是 $\hat p-e_y$
>
> 令 $z_c=\theta_c^{\top}x$，$\hat p_c=e^{z_c}/\sum_j e^{z_j}$。单样本对数似然是 $\ell=\log\hat p_y=z_y-\log\sum_j e^{z_j}$。对任意 logit：
>
> $$
> \frac{\partial\ell}{\partial z_c}
> =
> \mathbf{1}\{c=y\}-\frac{e^{z_c}}{\sum_j e^{z_j}}
> =
> \mathbf{1}\{c=y\}-\hat p_c.
> $$
>
> 再乘 $\partial z_c/\partial\theta_c=x$，得到 $\nabla_{\theta_c}\ell=(e_{y,c}-\hat p_c)x$。向量写法即输出层误差 $\hat p-e_y$。Jacobian $\partial\hat p_c/\partial z_{c'}=\hat p_c(\delta_{cc'}-\hat p_{c'})$ 与交叉熵的 $1/\hat p_y$ 相乘后，同样消成 $\hat p-e_y$。
>
> 感性理解：$\partial\ell/\partial z_c$ 是“正确类指示”减去“模型分给该类的概率”。正确类为 $1-\hat p_y$，其余类为 $-\hat p_c$，合起来即 $\hat p-e_y$。二分类时 $\hat p_1=h$、$\hat p_0=1-h$，退回 $h-y$。实现时不要再乘一遍 softmax 的 Jacobian。

<br>

## 牛顿法（Newton's method）

先看标量：要解 $f(\theta)=0$，用切线

$$
\theta\leftarrow\theta-\frac{f(\theta)}{f'(\theta)}.
$$

最大化 $\ell$ 时令 $f=\ell'$，得到

$$
\theta\leftarrow\theta-\frac{\ell'(\theta)}{\ell''(\theta)}.
$$

$\theta$ 为向量时推广为 Newton-Raphson：

$$
\theta\leftarrow\theta-H^{-1}\nabla_{\theta}\ell(\theta),
$$

其中 $H_{jk}=\partial^{2}\ell/\partial\theta_j\partial\theta_k$ 是 Hessian。用于最大化逻辑回归对数似然时，该方法也叫 Fisher scoring。二次收敛：靠近最优点时有效数字大致倍增。每步要形成并分解 $d\times d$ 的 $H$，故 $d$ 很大时不如（随机）梯度便宜；$d$ 中等时总时间通常远小于批量梯度。

> [!NOTE]+ 牛顿法
>
> 用当前点的二阶泰勒代替目标，一步走到该二次型的驻点。来自解非线性方程 $f(\theta)=0$ 的切线法，把 $f$ 换成 $\nabla\ell$ 即得最大化版本。相对固定步长的梯度上升：步长被 $H^{-1}$ 按曲率缩放，椭圆谷里不必手调各方向的 $\alpha$。代价是每步 $O(nd^{2}+d^{3})$，且 $H$ 接近奇异时需要阻尼或改回梯度。

<br>

对逻辑回归，令 $h^{(i)}=g(\theta^{\top}x^{(i)})$，

$$
\nabla_{\theta}\ell=X^{\top}(\vec{y}-h),\qquad
H=-X^{\top}RX,\qquad
R=\mathrm{diag}\big(h^{(i)}(1-h^{(i)})\big).
$$

若改为最小化 $J=-\ell$，则牛顿步为 $\theta\leftarrow\theta-(X^{\top}RX)^{-1}X^{\top}(h-\vec{y})$。$R$ 的对角元在 $h\approx 0$ 或 $1$ 时接近 $0$，对应“已经非常确定”的样本对曲率贡献小。

> [!INFO]+ Hessian $H=-X^{\top}RX$ 从哪来
>
> 单样本 $\nabla_{\theta}\ell=(y-h)x$。再对 $\theta$ 求导：$h=g(\theta^{\top}x)$，$g'=h(1-h)$，于是
>
> $$
> \frac{\partial h}{\partial\theta}=h(1-h)\,x,\qquad
> \nabla^{2}_{\theta}\ell=-h(1-h)\,xx^{\top}.
> $$
>
> $n$ 个样本相加：$H=-X^{\top}RX$，$R$ 的对角是各点的 $h^{(i)}(1-h^{(i)})$。牛顿步用 $H^{-1}$ 去“除以曲率”：山谷窄的方向步子自动变小，山谷宽的方向步子变大。梯度下降对所有方向用同一个 $\alpha$，所以在被拉长的椭圆上走之字。
>
> 感性理解：牛顿步用当前点的二阶泰勒代替 $\ell$，并走到该二次型的驻点。逻辑回归的 $\ell$ 在最优点附近接近二次，故后几步收敛很快。$h\in\{0,1\}$ 的样本满足 $h(1-h)\approx 0$，对 Hessian 几乎无贡献；曲率主要由预测仍接近 $1/2$ 的点决定。

<br>

> [!EXAMPLE]+ 例3 一维牛顿法
>
> 令 $\ell(\theta)=-(\theta-3)^{2}$，则 $\ell'= -2(\theta-3)$，$\ell''=-2$。任取 $\theta=10$，一步 $\theta\leftarrow 10-(-14)/(-2)=3$。二次目标的牛顿步一次到达驻点。逻辑回归的 $\ell$ 不是二次，但在最优点附近近似二次，因此后几步收敛很快。

<br>

## 算法

1. 标签必须是 $\{0,1\}$，不能是 $\{-1,+1\}$，除非改写损失。
2. 初始化 $\theta=0$ 合法：没有隐层对称性。
3. 梯度上升：重复计算 $h=g(X\theta)$，再 $\theta\leftarrow\theta+\alpha X^{\top}(y-h)$。
4. 牛顿法：重复计算 $R$ 与 $H$，解线性系统而不是显式求逆。
5. 预测：阈值 $0.5$，或按假阳性代价移动阈值。阈值不改变 $\ell$，只改变准确率。

- 线性可分且不加正则时，$\|\theta\|\rightarrow\infty$，$h$ 被推到 $0/1$，似然趋向 $1$。必须加 $L_2$ 或早停。
- 特征尺度仍影响梯度法；牛顿法对仿射尺度不那么敏感，因为 $H^{-1}$ 吸收了曲率。

## 实现

```python
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out

def logistic_loglik(X, y, theta, eps=1e-12):
    h = np.clip(sigmoid(X @ theta), eps, 1.0 - eps)
    return float(y @ np.log(h) + (1.0 - y) @ np.log(1.0 - h))

def logistic_gradient_ascent(X, y, lr=0.1, num_steps=2000):
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(num_steps):
        h = sigmoid(X @ theta)
        history.append(logistic_loglik(X, y, theta))
        theta += lr * (X.T @ (y - h))
    return theta, history

def logistic_newton(X, y, num_steps=20, l2=1e-6):
    n, d1 = X.shape
    theta = np.zeros(d1)
    for _ in range(num_steps):
        h = sigmoid(X @ theta)
        r = h * (1.0 - h)
        H = X.T @ (X * r[:, None]) + l2 * np.eye(d1)
        grad = X.T @ (y - h) - l2 * theta
        step, *_ = np.linalg.lstsq(H, grad, rcond=None)
        theta += step
    return theta

def softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
```

> [!EXAMPLE]+ 例4 线性可分与不可分
>
> 二维点若正类都在 $x_1+x_2>1$ 一侧，牛顿法会把 $\|\theta\|$ 推得很大，决策边界仍是那条直线，只是概率更“硬”。若四点构成异或，$h$ 无法同时接近四个标签，训练 $\ell$ 有有限上界。这是后面引入特征映射、核与隐层的动机。

<br>

> [!NOTE]+ 对照：平方损失与交叉熵
>
> 二者梯度都能写成“残差 × 特征”。差别在残差如何定义：回归是 $\theta^{\top}x-y$，分类是 $\sigma(\theta^{\top}x)-y$。把平方损失硬套在 sigmoid 上，会多乘 $h(1-h)$，极错时梯度消失。线性可分时交叉熵会把 $\|\theta\|$ 推向无穷，必须靠正则或早停。

<br>

> [!NOTE]+ 衔接
>
> 逻辑回归给出凸的 $\ell$ 与牛顿步。下一章不再把 sigmoid 当成独立发明：它是 Bernoulli 指数族的典范响应。高斯族收回最小二乘，泊松族给出计数回归。

<br>

# 广义线性模型

## 概念：GLM

- **问题**
  - 响应 $y$ 不必是高斯或 Bernoulli，还可以是计数、正实数等；
  - 指数族（exponential family）把这些分布写成同一模板，再令自然参数（natural parameter）线性依赖于 $x$。
- **范围**
  - 识别指数族三件套 $(T,a,b)$，并按三条假设构造 GLM；
  - 输入特征 $x$ 与指数族选择，输出 $E[y\mid x]$ 的预测。
- **章节衔接**
  - 上一章已写出 $h=\sigma(\theta^{\top}x)$；
  - 本章说明 sigmoid 来自 Bernoulli 的典范响应，OLS 来自高斯；
  - 下一章离开只建 $p(y\mid x)$，改建 $p(x\mid y)$。

## 指数族（exponential family）

一类分布属于指数族，若可写成

$$
p(y;\eta)=b(y)\exp\big(\eta^{\top}T(y)-a(\eta)\big).
$$

- $\eta$：自然参数（canonical parameter）。
- $T(y)$：充分统计量；本章常见 $T(y)=y$。
- $a(\eta)$：对数配分函数；$e^{-a(\eta)}$ 保证积分或求和为 $1$。
- $b(y)$：基测度，不含 $\eta$。

固定 $T,a,b$ 后，改变 $\eta$ 就在该族内移动。

> [!NOTE]+ 指数族与 GLM
>
> 指数族把高斯、Bernoulli、泊松等写成同一模板 $b(y)\exp(\eta^{\top}T(y)-a(\eta))$。GLM 再令 $\eta=\theta^{\top}x$，并预测 $E[T(y)\mid x]$。来源是统一此前两章看起来不像的算法：平方损失对应高斯，交叉熵对应 Bernoulli。相对“每种响应另写一套模型”：梯度一律是 $(y-\mu)x$，换分布只换联系函数。$\eta$ 对 $x$ 必须线性；要非线性应改广义可加模型或网络，指数族部分仍可保留。

<br>

**Bernoulli。** $p(y;\phi)=\phi^{y}(1-\phi)^{1-y}$。取对数：

$$
p(y;\phi)
=
\exp\big(y\log\phi+(1-y)\log(1-\phi)\big)
=
\exp\Big(y\log\frac{\phi}{1-\phi}+\log(1-\phi)\Big).
$$

因而

$$
\eta=\log\frac{\phi}{1-\phi},\qquad
\phi=\frac{1}{1+e^{-\eta}},\qquad
T(y)=y,\qquad
a(\eta)=-\log(1-\phi)=\log(1+e^{\eta}),\qquad
b(y)=1.
$$

sigmoid 由 Bernoulli 自然参数与均值之间的反函数给出，不是额外指定的响应曲线。

> [!INFO]+ Bernoulli 怎样变成 sigmoid
>
> 从 $p(y)=\phi^{y}(1-\phi)^{1-y}$ 取指数：
>
> $$
> \log p=y\log\phi+(1-y)\log(1-\phi)=y\log\frac{\phi}{1-\phi}+\log(1-\phi).
> $$
>
> 与标准形 $\eta T(y)-a(\eta)$ 对照，$T(y)=y$，故 $\eta=\log(\phi/(1-\phi))$。两边取指数再整理：
>
> $$
> e^{\eta}=\frac{\phi}{1-\phi}\implies\phi=\frac{e^{\eta}}{1+e^{\eta}}=\frac{1}{1+e^{-\eta}}.
> $$
>
> 这就是 sigmoid。GLM 再令 $\eta=\theta^{\top}x$，于是 $h_{\theta}(x)=\sigma(\theta^{\top}x)$。
>
> 感性理解：均值 $\phi\in(0,1)$，自然参数 $\eta$ 却在 $\mathbb{R}$ 上取值。对数几率 $\log(\phi/(1-\phi))$ 是二者之间的标准双射，其反函数即 sigmoid。GLM 再令 $\eta=\theta^{\top}x$，逻辑回归的响应函数由此确定。

<br>

**高斯（方差固定）。** 把 $\mathcal{N}(\mu,1)$ 写成指数族：

$$
\begin{aligned}
p(y;\mu)
&=
\frac{1}{\sqrt{2\pi}}\exp\Big(-\frac12(y-\mu)^{2}\Big)
=
\frac{1}{\sqrt{2\pi}}\exp\Big(-\frac12 y^{2}\Big)
\exp\Big(\mu y-\frac12\mu^{2}\Big).
\end{aligned}
$$

故 $\eta=\mu$，$T(y)=y$，$a(\eta)=\eta^{2}/2$，$b(y)=(2\pi)^{-1/2}\exp(-y^{2}/2)$。若把 $\sigma^{2}$ 也当作未知，需要二维自然参数或带色散参数 $\tau$ 的更一般写法 $p(y;\eta,\tau)=b(y,\tau)\exp((\eta^{\top}T(y)-a(\eta))/c(\tau))$；对 GLM 构造，固定 $\sigma^{2}$ 已足够。

**泊松。** $p(y;\lambda)=e^{-\lambda}\lambda^{y}/y!$，$y\in\{0,1,2,\ldots\}$。则 $\eta=\log\lambda$，$\lambda=e^{\eta}$，$T(y)=y$，$a(\eta)=e^{\eta}$，$b(y)=1/y!$。适合到店人数、网页浏览次数等计数。

指数族的两个常用事实：

$$
E[T(y);\eta]=\nabla_{\eta}a(\eta),\qquad
\mathrm{Var}(T(y);\eta)=\nabla_{\eta}^{2}a(\eta).
$$

对数配分函数的一阶导是均值，二阶导是方差。牛顿法里的对角权重 $h(1-h)$ 因此等于 Bernoulli 方差，而不是另设的经验系数。

> [!INFO]+ $a'(\eta)$ 为何是均值
>
> $\int p(y;\eta)\,dy=1$。把 $p=b(y)\exp(\eta T(y)-a(\eta))$ 代入，对 $\eta$ 求导（在积分号下）：
>
> $$
> 0=\int p(y;\eta)\big(T(y)-a'(\eta)\big)\,dy=E[T(y)]-a'(\eta).
> $$
>
> 再导一次得到 $a''(\eta)=\mathrm{Var}(T(y))$。Bernoulli 时 $a(\eta)=\log(1+e^{\eta})$，$a'=\sigma(\eta)=\phi$，$a''=\phi(1-\phi)$。牛顿法里的 $R_{ii}=h^{(i)}(1-h^{(i)})$ 就是这一点的方差：模型越确定，曲率越小。
>
> 感性理解：$a(\eta)$ 是指数族的对数配分函数，用来保证密度积分为 $1$。$\eta$ 增大时，质量向 $T(y)$ 较大的一侧移动，$a'(\eta)$ 恰为该移动的均值；$a''(\eta)$ 描述质量铺开的程度，即方差。牛顿法里的对角权重 $h(1-h)$ 因此不是另加的经验系数。

<br>

## 构造

要对 $x$ 预测 $y$，GLM 采用三条假设。

1. $y\mid x;\theta\sim\mathrm{ExponentialFamily}(\eta)$。
2. 给定 $x$，目标是预测 $T(y)$ 的条件期望；通常 $T(y)=y$，即预测 $\mu=E[y\mid x]$。
3. 自然参数与输入线性：$\eta=\theta^{\top}x$（向量 $\eta$ 时为 $\eta=\Theta x$）。

于是

$$
h_{\theta}(x)=E[T(y)\mid x;\theta]=g(\eta)=g(\theta^{\top}x),
$$

其中 $g(\eta)=E[T(y);\eta]$ 称为典范响应函数，$g^{-1}$ 称为典范联系函数。

**普通最小二乘。** 选高斯族。前面已得 $\mu=\eta$，故

$$
h_{\theta}(x)=E[y\mid x;\theta]=\mu=\eta=\theta^{\top}x.
$$

**逻辑回归。** 选 Bernoulli 族。$\phi=1/(1+e^{-\eta})$ 且 $E[y\mid x]=\phi$，故

$$
h_{\theta}(x)=\frac{1}{1+e^{-\theta^{\top}x}}.
$$

**泊松回归。** $E[y\mid x]=\lambda=e^{\eta}=\exp(\theta^{\top}x)$。计数预测自动为正。对数似然（略去 $y!$）为 $\ell=\sum_i(y^{(i)}\theta^{\top}x^{(i)}-e^{\theta^{\top}x^{(i)}})$，梯度

$$
\nabla_{\theta}\ell=\sum_{i=1}^{n}\big(y^{(i)}-e^{\theta^{\top}x^{(i)}}\big)x^{(i)}.
$$

仍然是“残差乘特征”。这是指数族配典范联系时的一般模式：$\nabla\ell=\sum(y-\mu)x$。

> [!INFO]+ 为何三条假设刚好够用
>
> 假设 1 指定响应的指数族（连续、二元或计数）。假设 3 将条件分布的自由度限制为有限维 $\theta$。假设 2 规定预测对象是 $E[T(y)\mid x]$，一般为均值。若目标改为分位数，则不再是标准 GLM。若 $\eta$ 对 $x$ 非线性，则进入广义可加模型或神经网络，指数族部分仍可保留。
>
> 感性理解：假设 1 选定 $y$ 的指数族，假设 2 指定预测均值，假设 3 把自然参数限制为 $\theta^{\top}x$。联系函数随之确定：高斯为恒等，Bernoulli 为 sigmoid，泊松为指数。更换损失对应更换指数族，而不是另写一套梯度公式。

<br>

## 实现

```python
def poisson_regression(X, y, lr=1e-3, num_steps=2000):
    """X: (n, d+1), y: 非负计数。"""
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(num_steps):
        mu = np.exp(X @ theta)
        history.append(float(np.sum(y * (X @ theta) - mu)))
        theta += lr * (X.T @ (y - mu))
    return theta, history
```

> [!EXAMPLE]+ 例5 到店人数
>
> 特征含是否周末、是否促销。泊松回归输出 $\hat\lambda=\exp(\theta^{\top}x)$。若误用线性回归，周末系数很大时预测可能为负，计数解释崩溃。若数据方差明显大于均值（过离散），泊松方差 $=$ 均值的假设失败，应改负二项或给色散参数。

<br>

> [!NOTE]+ 衔接
>
> GLM 统一了“选指数族 + 令 $\eta=\theta^{\top}x$”。到此为止的模型都直接建 $p(y\mid x)$。下一章改建 $p(x\mid y)$，再用贝叶斯法则回到后验。同一条线性边界可以由两条完全不同的估计路线得到。

<br>

# 生成学习

## 概念：生成

- **问题**
  - 判别模型（discriminative model）直接建 $p(y\mid x)$，如逻辑回归；
  - 生成模型（generative model）建 $p(x\mid y)$ 与 $p(y)$，再用贝叶斯法则得 $p(y\mid x)$。
- **范围**
  - 写出 GDA 与朴素贝叶斯的 MLE，并与逻辑回归对照；
  - 输入带标签样本，输出类条件密度参数与后验。
- **章节衔接**
  - 上一章的 GLM 是判别模型：直接建 $p(y\mid x)$；
  - 本章建 $p(x\mid y)p(y)$，GDA 在共享 $\Sigma$ 时边界仍线性，但 $\theta$ 的估计量不同；
  - 下一章回到判别路线，只把 $x$ 换成 $\phi(x)$。

贝叶斯法则：

$$
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
$$

$p(x)=\sum_{y'}p(x\mid y')p(y')$ 对决策只是归一化。生成模型可以采样 $x$，也可以处理缺失特征；判别模型通常在 $p(y\mid x)$ 被正确设定时渐近更省数据。

## 多元正态（multivariate Gaussian）

$$
p(x;\mu,\Sigma)
=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\Big(-\frac12(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\Big).
$$

- $\mu=E[x]\in\mathbb{R}^{d}$。
- $\Sigma=E[(x-\mu)(x-\mu)^{\top}]$ 对称半正定；$|\Sigma|$ 为行列式。
- 等高线是椭圆，主轴沿 $\Sigma$ 的特征向量。
- $\Sigma=\sigma^{2}I$ 时椭圆变圆，坐标独立。

二次型 $(x-\mu)^{\top}\Sigma^{-1}(x-\mu)$ 是马氏距离的平方：先按相关结构“拉白”，再量欧氏长度。

## GDA

高斯判别分析：类先验为 Bernoulli，类条件为高斯，且两类共享 $\Sigma$。

$$
\begin{aligned}
y &\sim \mathrm{Bernoulli}(\phi),\\
x\mid y=0 &\sim \mathcal{N}(\mu_0,\Sigma),\\
x\mid y=1 &\sim \mathcal{N}(\mu_1,\Sigma).
\end{aligned}
$$

> [!NOTE]+ GDA
>
> 类条件取共享协方差的高斯，类先验取 Bernoulli，再用贝叶斯得 $p(y\mid x)$。来源是生成路线：先写 $p(x\mid y)p(y)$，而不是直接写 $p(y\mid x)$。相对逻辑回归：小样本、近似高斯时估计更省数据，且能采样 $x$；假设错了（离散词袋、重尾）则偏差更大。共享 $\Sigma$ 把二次项抵消，决策面仍是线性，这是它能和逻辑回归比的前提。

<br>

对数似然对 $n$ 个 IID 样本求最大，得到

$$
\begin{aligned}
\phi
&=
\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\{y^{(i)}=1\},\\
\mu_0
&=
\frac{\sum_{i}\mathbf{1}\{y^{(i)}=0\}x^{(i)}}{\sum_{i}\mathbf{1}\{y^{(i)}=0\}},\\
\mu_1
&=
\frac{\sum_{i}\mathbf{1}\{y^{(i)}=1\}x^{(i)}}{\sum_{i}\mathbf{1}\{y^{(i)}=1\}},\\
\Sigma
&=
\frac{1}{n}\sum_{i=1}^{n}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^{\top}.
\end{aligned}
$$

即：$\phi$ 是正类比例；$\mu_c$ 是该类样本均值；$\Sigma$ 是相对各类中心的合并协方差。预测时比较 $p(y=1\mid x)$ 与 $1/2$，或比较对数后验。

> [!INFO]+ 共享协方差为何给出线性边界
>
> $\log p(x\mid y=1)-\log p(x\mid y=0)$ 中，二次项 $-x^{\top}\Sigma^{-1}x/2$ 两类相同，相减后消失。剩下
>
> $$
> (\mu_1-\mu_0)^{\top}\Sigma^{-1}x
> -\frac12\mu_1^{\top}\Sigma^{-1}\mu_1
> +\frac12\mu_0^{\top}\Sigma^{-1}\mu_0
> +\log\frac{\phi}{1-\phi}.
> $$
>
> 这是 $x$ 的线性函数。因此 $p(y=1\mid x)$ 仍具有 sigmoid 形式 $\sigma(\theta^{\top}x)$，只是 $\theta$ 由类高斯参数算出来，不是直接最大化判别似然。
>
> 若两类各有自己的 $\Sigma_0,\Sigma_1$（QDA），二次项不再相消，决策边界是二次曲面。数据少时共享 $\Sigma$ 方差更小；两类形状差很多时 QDA 偏差更小。

<br>

GDA 与逻辑回归的对照：

- 二者都能给出 $p(y=1\mid x)=\sigma(\theta^{\top}x)$ 的模型族（共享 $\Sigma$ 时）。
- GDA 对 $x\mid y$ 做了更强假设。假设近似正确时，GDA 更数据高效，小样本往往更好。
- $x$ 明显非高斯（例如离散指示变量）时，逻辑回归更稳健。
- GDA 是生成模型，逻辑回归是判别模型：即使 $p(x\mid y)$ 错了，逻辑回归仍可能学到好的 $p(y\mid x)$。

## 朴素贝叶斯（naive Bayes）

文本分类等情形下，$x$ 是高维离散特征，例如词表指示。朴素贝叶斯假设给定 $y$ 后各特征条件独立：

$$
p(x\mid y)=\prod_{j=1}^{d}p(x_j\mid y).
$$

该假设几乎从不严格成立（“cheap” 与 “viagra” 在垃圾邮件里相关），但作为偏置很强的生成模型，小样本上常常好用。

> [!NOTE]+ 朴素贝叶斯
>
> 在 $p(x\mid y)=\prod_j p(x_j\mid y)$ 下做生成分类。名称里的 naive 就是这条条件独立，不是对方法本身的贬义。来源是高维离散 $x$ 上完整联合 $p(x\mid y)$ 无法估计：词表大小为 $V$ 时，无独立假设要估 $2^{V}$ 项。相对 GDA：不需要连续高斯，参数个数随 $V$ 线性增长。相对逻辑回归：同样给出线性对数后验，但估计量不同，零计数必须平滑。

<br>

**Bernoulli 事件模型。** $x_j\in\{0,1\}$ 表示词 $j$ 是否出现。参数

$$
\phi_{j\mid y=c}=p(x_j=1\mid y=c),\qquad
\phi_y=p(y=1).
$$

MLE 是计数比例。若某词在某类从未出现，$\phi_{j\mid y=c}=0$，一篇含该词的测试文档后验变成 $0$。Laplace（加一）平滑：

$$
\phi_{j\mid y=c}
=
\frac{1+\sum_{i}\mathbf{1}\{y^{(i)}=c\wedge x_j^{(i)}=1\}}{2+\sum_{i}\mathbf{1}\{y^{(i)}=c\}}.
$$

一般地，对取 $|V|$ 个值的离散变量，分子加 $1$，分母加 $|V|$。这对应于 Dirichlet / Beta 先验下的后验均值。

**多项事件模型。** 文档是词序列，同一词可出现多次。令 $x_j$ 为词 $j$ 的出现次数，$\sum_j x_j=L$（文档长度）。

$$
p(x\mid y)=\frac{L!}{\prod_j x_j!}\prod_{j=1}^{|V|}\phi_{j\mid y}^{x_j}.
$$

预测时阶乘对所有类相同，可忽略。MLE 同样是计数，Laplace 分母加 $|V|$。

多项式模型吃词频，Bernoulli 模型只吃是否出现。长文档里词频更有信息；极短文本或只关心“出现过没有”时 Bernoulli 更稳。

## 实现

```python
def gda_fit(X, y):
    """X: (n, d) 不含截距。y in {0, 1}。"""
    X0, X1 = X[y == 0], X[y == 1]
    phi = float(np.mean(y))
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    Xc = np.vstack([X0 - mu0, X1 - mu1])
    Sigma = (Xc.T @ Xc) / X.shape[0]
    return {"phi": phi, "mu0": mu0, "mu1": mu1, "Sigma": Sigma}

def gda_predict_proba(X, params, ridge=1e-6):
    mu0, mu1, Sigma, phi = params["mu0"], params["mu1"], params["Sigma"], params["phi"]
    S = Sigma + ridge * np.eye(Sigma.shape[0])
    # 解 Sigma^{-1}(x-μ) 而不是显式求逆
    def quad(mu):
        dlt = X - mu
        sol = np.linalg.solve(S, dlt.T).T
        return np.sum(dlt * sol, axis=1)
    sign, logdet = np.linalg.slogdet(S)
    logp0 = -0.5 * (logdet + quad(mu0)) + np.log(1.0 - phi)
    logp1 = -0.5 * (logdet + quad(mu1)) + np.log(phi)
    return sigmoid(logp1 - logp0)

def naive_bayes_bernoulli(X, y, alpha=1.0):
    """X: (n, V) 的 0/1 矩阵。alpha=1 为 Laplace。"""
    n, V = X.shape
    phi_y = (y.sum() + alpha) / (n + 2.0 * alpha)
    phi_pos = (X[y == 1].sum(axis=0) + alpha) / (y.sum() + 2.0 * alpha)
    phi_neg = (X[y == 0].sum(axis=0) + alpha) / ((1 - y).sum() + 2.0 * alpha)
    return phi_y, phi_pos, phi_neg

def naive_bayes_predict(X, phi_y, phi_pos, phi_neg, eps=1e-12):
    lp1 = np.log(phi_y) + X @ np.log(phi_pos + eps) + (1 - X) @ np.log(1.0 - phi_pos + eps)
    lp0 = np.log(1.0 - phi_y) + X @ np.log(phi_neg + eps) + (1 - X) @ np.log(1.0 - phi_neg + eps)
    return (lp1 >= lp0).astype(int)
```

> [!EXAMPLE]+ 例6 垃圾邮件里的零概率
>
> 词表含 “lottery”。训练正类 100 封都没出现该词，不加平滑则 $\phi_{\mathrm{lottery}\mid y=1}=0$。测试信一旦出现 “lottery”，$p(x\mid y=1)=0$，后验崩溃。Laplace 后该条件概率是 $1/102$，文档仍可被其他强垃圾词推到正类。平滑强度 $\alpha$ 过大则所有词概率被拉向均匀，判别力下降。

<br>

> [!NOTE]+ 对照：判别与生成
>
> 逻辑回归最大化 $p(y\mid x)$，不对 $x$ 的密度负责。GDA / 朴素贝叶斯最大化 $p(x,y)$，因此能采样 $x$，也能在特征缺失时边缘化。$x$ 近似高斯且 $n$ 小时，GDA 往往更省数据；特征离散或明显非高斯时，逻辑回归更稳。共享 $\Sigma$ 的 GDA 与逻辑回归假设类同形，估计量不同，测试误差不必相同。

<br>

> [!NOTE]+ 衔接
>
> 生成模型在原始坐标上建密度，非线性边界要靠改 $p(x\mid y)$ 的形状。下一章留在判别路线，只把 $x$ 映射到 $\phi(x)$，并用核避免写出 $\phi$。

<br>

# 核方法

## 概念：核

- **问题**
  - 原始属性（attributes）$x$ 上的线性模型表达力不足时，先映射到特征（features）$\phi(x)$，再在 $\phi$ 上做线性运算；
  - 核（kernel）$K(x,z)=\phi(x)^{\top}\phi(z)$ 使算法只依赖内积，不必显式构造 $\phi$。
- **范围**
  - 把 LMS 改写成对偶形式，并给出核的合法判定；
  - 输入是属性空间中的点对，输出是核矩阵与核化预测。
- **章节衔接**
  - 上一章在原始 $x$ 上建密度；非线性分类需要新特征；
  - 本章用 $K(x,z)=\phi(x)^{\top}\phi(z)$ 避免显式 $\phi$；
  - 下一章把同一内积放进最大间隔的对偶。

房价若更接近三次函数，定义

$$
\phi(x)=(1,x,x^{2},x^{3})^{\top}\in\mathbb{R}^{4},
$$

则 $\theta_3 x^3+\theta_2 x^2+\theta_1 x+\theta_0=\theta^{\top}\phi(x)$。$x$ 仍叫属性，$\phi(x)$ 叫特征，$\phi$ 叫特征映射。多项式、交互项、直方图编码都是 $\phi$ 的例子。

## 特征上的 LMS

在 $\phi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{p}$ 上拟合 $\theta^{\top}\phi(x)$，$\theta\in\mathbb{R}^{p}$。把普通 LMS 中的 $x^{(i)}$ 全部换成 $\phi(x^{(i)})$：

$$
\theta\leftarrow\theta+\alpha\sum_{i=1}^{n}\big(y^{(i)}-\theta^{\top}\phi(x^{(i)})\big)\phi(x^{(i)}).
$$

随机版本每次只用一个 $i$。当 $\phi$ 含 $x\in\mathbb{R}^{d}$ 的全部三次单项式时，$p=O(d^{3})$，显式存 $\theta$ 与 $\phi(x)$ 开始变贵。

## 核技巧（kernel trick）

若 $\theta$ 从 $0$ 出发，每次更新都沿某个 $\phi(x^{(i)})$ 的方向走，则始终有

$$
\theta=\sum_{i=1}^{n}\beta_i\phi(x^{(i)}).
$$

预测变成

$$
\theta^{\top}\phi(x)
=
\sum_{i=1}^{n}\beta_i\,\phi(x^{(i)})^{\top}\phi(x)
=
\sum_{i=1}^{n}\beta_i K(x^{(i)},x).
$$

> [!NOTE]+ 核技巧
>
> 凡算法只通过内积 $\phi(x)^{\top}\phi(z)$ 访问特征，就把该内积换成 $K(x,z)$，不必构造 $\phi$。来源是再生核希尔伯特空间与 Mercer 定理：合法的 $K$ 对应某个（可能无限维的）$\phi$。相对显式多项式特征：$d$ 维三次单项式是 $O(d^{3})$，高斯核对应的 $\phi$ 甚至无限维，计算仍是 $O(n^{2})$ 的核矩阵。相对直接在原始 $x$ 上做线性模型：表达力上升，但存贮从 $O(d)$ 变成 $O(n^{2})$，预测必须回看训练点。

<br>

只需核

$$
K(x,z)=\phi(x)^{\top}\phi(z).
$$

$d$ 维三次多项式核的闭式是

$$
K(x,z)=(x^{\top}z+c)^{3},
$$

计算 $O(d)$，等价特征维却是 $O(d^{3})$。这就是核技巧：算法改写成只出现内积，再把内积换成 $K$。

对偶系数的批量更新（由 $\theta$ 的更新代入）可写成只含 $K$ 的形式。令 $K_{ij}=K(x^{(i)},x^{(j)})$，则

$$
\theta^{\top}\phi(x^{(j)})=\sum_{i}\beta_i K_{ij},
$$

$\beta$ 的更新与残差 $y^{(j)}-\sum_i\beta_i K_{ij}$ 成正比。训练要反复读核矩阵，代价至少随 $n^{2}$ 增长。

> [!INFO]+ 为何 $\theta$ 永远活在训练点张成的空间里
>
> 从 $\theta=0$ 出发。第 $i$ 个样本的 LMS 更新是 $\theta\leftarrow\theta+\alpha(y^{(i)}-\theta^{\top}\phi(x^{(i)}))\phi(x^{(i)})$，每一步只沿某个 $\phi(x^{(i)})$ 走。归纳：$t$ 步之后
>
> $$
> \theta^{(t)}=\sum_{i=1}^{n}\beta_i^{(t)}\phi(x^{(i)}).
> $$
>
> 预测 $\theta^{\top}\phi(x)=\sum_i\beta_i\phi(x^{(i)})^{\top}\phi(x)=\sum_i\beta_i K(x^{(i)},x)$。$\phi$ 本身不必出现。
>
> 感性理解：从 $\theta=0$ 出发的梯度步始终沿某个 $\phi(x^{(i)})$ 进行，故 $\theta$ 始终落在训练特征张成的子空间里。预测只需 $\theta^{\top}\phi(x)=\sum_i\beta_i K(x^{(i)},x)$，不必构造 $\phi$ 本身。核矩阵就是这组内积。

<br>

## 性质

不是任意 $K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$ 都对应某个 $\phi$ 的内积。Mercer 条件：对任意有限集 $\{x^{(1)},\ldots,x^{(n)}\}$，核矩阵 $K$ 必须对称半正定，即对任意 $z$，

$$
z^{\top}Kz\ge 0.
$$

等价地，$K$ 的特征值全非负。若某数据集上 $K$ 有负特征值，该 $K$ 不能作为核。

常用核：

| 核 | 公式 | 对应直觉 |
| :--- | :--- | :--- |
| 线性 | $K(x,z)=x^{\top}z$ | $\phi$ 为恒等 |
| 多项式 | $K(x,z)=(x^{\top}z+c)^{m}$ | 次数 $\le m$ 的单项式 |
| 高斯 / RBF | $K(x,z)=\exp(-\|x-z\|_{2}^{2}/(2\sigma^{2}))$ | 无限维特征，局部相似 |
| 字符串核等 | 由领域定义 | 离散结构上的加权共现 |

高斯核 $K(x,x)=1$，随距离衰减。$\sigma$ 太小则几乎只有对角，过拟合；太大则所有点彼此相似，欠拟合。核必须在训练与测试间保持同一公式、同一尺度；先标准化特征再算 RBF。

> [!INFO]+ 为何半正定不可少
>
> 若 $K=\Phi\Phi^{\top}$，其中 $\Phi$ 的第 $i$ 行是 $\phi(x^{(i)})^{\top}$，则 $z^{\top}Kz=\|\Phi^{\top}z\|_{2}^{2}\ge 0$。反过来，对称半正定 $K$ 必有平方根，可把该平方根的行当作有限样本上的特征。优化问题里 $K$ 出现在二次项 $\alpha^{\top}(K\odot yy^{\top})\alpha$ 中；若 $K$ 不定，对偶目标不再凹，二次项可以沿某方向趋于 $-\infty$。
>
> 感性理解：若 $z^{\top}Kz<0$，则 $K$ 不能写成任何 $\Phi\Phi^{\top}$，对应的“内积”在实特征空间中不存在。对偶目标的凹性依赖 $K$ 半正定；一旦不定，二次项可沿某方向无下界。Mercer 条件因此是可优化性条件，而不只是形式上的对称。

<br>

## 实现

```python
def polynomial_kernel(X, Z, degree=3, c=1.0):
    """X: (n, d), Z: (m, d) -> (n, m)。"""
    return (X @ Z.T + c) ** degree

def rbf_kernel(X, Z, sigma=1.0):
    X2 = np.sum(X * X, axis=1)[:, None]
    Z2 = np.sum(Z * Z, axis=1)[None, :]
    sq = np.maximum(X2 + Z2 - 2.0 * X @ Z.T, 0.0)
    return np.exp(-sq / (2.0 * sigma ** 2))

def kernel_lms_predict(X_train, beta, x, kernel_fn):
    return float(beta @ kernel_fn(X_train, x[None, :]).ravel())
```

> [!EXAMPLE]+ 例7 二次核展开
>
> $x,z\in\mathbb{R}^{2}$，$K=(x^{\top}z)^{2}=x_1^{2}z_1^{2}+x_2^{2}z_2^{2}+2x_1 x_2 z_1 z_2$。对应 $\phi(x)=(x_1^{2},x_2^{2},\sqrt{2}x_1 x_2)$。显式 $\phi$ 在 $d=2$ 时可写；$d=100$ 的三次核就不应再展开。

<br>

> [!NOTE]+ 衔接
>
> 核把 LMS 改写成只依赖 $K_{ij}$。下一章不再最小化平方损失，而是最大化几何间隔；对偶目标里出现的仍是同一类内积，因此核可以原样代入。

<br>

# 支持向量机

## 概念：SVM

- **问题**
  - 二分类，标签改为 $y\in\{-1,+1\}$；
  - 寻找间隔（margin）最大的分隔超平面（separating hyperplane），使预测正确且离边界远。
- **范围**
  - 从几何间隔推出原问题与对偶，并处理软间隔与 SMO；
  - 输入带标签点，输出 $w,b$ 或对偶系数 $\alpha$ 与支持向量。
- **章节衔接**
  - 上一章给出合法的 $K$；
  - 本章在 $\phi$ 空间最大化几何间隔，对偶只出现 $K_{ij}$；
  - 树与网络是另外两条得到非线性边界的路，不经过核矩阵。

逻辑回归里 $|\theta^{\top}x|$ 大表示更有把握。点 $A$ 远离边界时，边界微扰不易改预测；点 $C$ 贴在边界上则把握低。SVM 把“所有训练点都正确且尽量远离边界”写成优化问题。

> [!NOTE]+ 支持向量机
>
> 在间隔最大的超平面上做分类，最优法向由贴在间隔边界上的训练点（支持向量）决定。来源是统计学习理论里的间隔界：几何间隔大则 VC 型容量更小，不是为了输出概率。相对逻辑回归：hinge 让间隔外的正确点梯度为 $0$，解稀疏，便于核化；输出不是 $P(y=1\mid x)$。相对硬编码的感知机：目标是最大间隔而不是任意可分平面。

<br>

分类器写成

$$
h_{w,b}(x)=g(w^{\top}x+b),\qquad
g(z)=\begin{cases}+1 & z\ge 0\\ -1 & z<0\end{cases}.
$$

此处不再把 $x_0=1$ 塞进 $w$，$b$ 单独出现。输出直接是 $\pm 1$，不经过概率。

## 间隔（margin）

功能间隔（functional margin）：

$$
\hat\gamma^{(i)}=y^{(i)}(w^{\top}x^{(i)}+b).
$$

$\hat\gamma^{(i)}>0$ 表示分类正确。把 $(w,b)$ 乘 $2$，功能间隔也乘 $2$，但 $g$ 只看符号，预测不变。因此功能间隔可被任意放大，单独拿来当目标没有尺度意义。

几何间隔（geometric margin）是点到超平面的有符号距离。$w$ 垂直于 $\{x:w^{\top}x+b=0\}$。正类点 $x^{(i)}$ 沿 $-w/\|w\|$ 走到平面上的距离 $\gamma^{(i)}$ 满足

$$
w^{\top}\Big(x^{(i)}-\gamma^{(i)}\frac{w}{\|w\|}\Big)+b=0,
$$

解出

$$
\gamma^{(i)}=y^{(i)}\Big(\Big(\frac{w}{\|w\|}\Big)^{\top}x^{(i)}+\frac{b}{\|w\|}\Big)=\frac{\hat\gamma^{(i)}}{\|w\|}.
$$

几何间隔对 $(w,b)$ 的正尺度缩放不变。训练集上的间隔取最小值：$\gamma=\min_i\gamma^{(i)}$，$\hat\gamma=\min_i\hat\gamma^{(i)}$。

> [!INFO]+ 从平面方程到距离
>
> 平面 $w^{\top}x+b=0$。单位法向是 $w/\|w\|$。从正类点 $x$ 沿 $-w/\|w\|$ 走距离 $\gamma$，落到平面上的点是 $x-\gamma w/\|w\|$。该点满足平面方程：
>
> $$
> w^{\top}x-\gamma\|w\|+b=0\implies\gamma=\frac{w^{\top}x+b}{\|w\|}.
> $$
>
> 负类点需要乘 $y=-1$，否则距离会带错符号。与功能间隔对照：$\hat\gamma=y(w^{\top}x+b)$，故 $\gamma=\hat\gamma/\|w\|$。
>
> 感性理解：$(w,b)$ 同时乘正数 $c$，功能间隔按 $c$ 缩放，点到超平面的欧氏距离不变。几何间隔才是尺度不变的间隔。把功能间隔固定为 $1$ 后，最大化几何间隔等价于最小化 $\|w\|$。

<br>

## 最优间隔（optimal margin）

线性可分时，最大化几何间隔：

$$
\max_{\gamma,w,b}\ \gamma
\quad\mathrm{s.t.}\quad
y^{(i)}(w^{\top}x^{(i)}+b)\ge\gamma,\ 
\|w\|=1.
$$

$\|w\|=1$ 使功能间隔等于几何间隔，但该约束非凸。改写为最大化 $\hat\gamma/\|w\|$，再固定尺度 $\hat\gamma=1$（合法，因为可整体缩放），目标变成最大化 $1/\|w\|$，即

$$
\min_{w,b}\ \frac12\|w\|_{2}^{2}
\quad\mathrm{s.t.}\quad
y^{(i)}(w^{\top}x^{(i)}+b)\ge 1,\quad i=1,\ldots,n.
$$

这是凸二次目标加线性约束，可用二次规划求解。最优处至少有一些点贴在 $\hat\gamma=1$ 的两条边界上，这些点就是支持向量。$w$ 只由它们决定。

> [!INFO]+ 为何最小化 $\|w\|^{2}/2$ 就是最大间隔
>
> 几何间隔 $\gamma=\hat\gamma/\|w\|$。固定 $\hat\gamma=1$ 后 $\gamma=1/\|w\|$。最小化 $\|w\|$ 与最大化 $\gamma$ 等价。除以 $2$ 与改成平方都不改变最优点，却让梯度等于 $w$，Hessian 等于单位阵，二次规划更标准。约束 $y(w^{\top}x+b)\ge 1$ 是“功能间隔至少为 1”，不是几何距离至少为 1；几何距离是 $1/\|w\|$。
>
> 感性理解：两类支撑超平面之间的宽度为 $2/\|w\|$。放大 $w$ 只改变功能间隔的数值，不改变该宽度。将边界上的功能间隔固定为 $1$ 后，缩短 $\|w\|$ 才真正加宽间隔。目标写成 $\|w\|_{2}^{2}/2$ 是为了使梯度等于 $w$，最优解不变。

<br>

## 对偶（duality）

带不等式的原问题：

$$
\min_{w}f(w)
\quad\mathrm{s.t.}\quad
g_i(w)\le 0,\ 
h_i(w)=0.
$$

广义 Lagrangian

$$
\mathcal{L}(w,\alpha,\beta)=f(w)+\sum_{i}\alpha_i g_i(w)+\sum_{i}\beta_i h_i(w),\qquad\alpha_i\ge 0.
$$

定义 $\theta_P(w)=\max_{\alpha\ge 0,\beta}\mathcal{L}$。若 $w$ 违反约束，可令对应乘子趋于无穷，$\theta_P=+\infty$；若 $w$ 可行且取合适乘子，$\theta_P=f(w)$。因此最小化 $\theta_P$ 等价于原问题。对偶函数 $\theta_D(\alpha,\beta)=\min_{w}\mathcal{L}$，对偶问题是 $\max_{\alpha\ge 0,\beta}\theta_D$。始终有弱对偶：对偶最优值 $\le$ 原最优值。在凸问题且约束 qualification（如 Slater：存在严格可行点）成立时，强对偶：两者相等。此时 KKT 条件成立：

- 原可行、对偶可行；
- $\nabla_w\mathcal{L}=0$；
- 互补松弛（complementary slackness）：$\alpha_i g_i(w)=0$。

对最优间隔分类器，$g_i(w)=1-y^{(i)}(w^{\top}x^{(i)}+b)$，$f=\|w\|^{2}/2$。Lagrangian

$$
\mathcal{L}=\frac12\|w\|_{2}^{2}-\sum_{i=1}^{n}\alpha_i\big[y^{(i)}(w^{\top}x^{(i)}+b)-1\big].
$$

对 $w,b$ 求最小：

$$
w=\sum_{i=1}^{n}\alpha_i y^{(i)}x^{(i)},\qquad
\sum_{i=1}^{n}\alpha_i y^{(i)}=0.
$$

代回得到对偶：

$$
\max_{\alpha}\ 
\sum_{i=1}^{n}\alpha_i
-\frac12\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_j y^{(i)}y^{(j)}(x^{(i)})^{\top}x^{(j)}
\quad\mathrm{s.t.}\quad
\alpha_i\ge 0,\ 
\sum_{i}\alpha_i y^{(i)}=0.
$$

互补松弛：$\alpha_i>0$ 仅当 $y^{(i)}(w^{\top}x^{(i)}+b)=1$，即支持向量。预测

$$
w^{\top}x+b=\sum_{i\in\mathrm{SV}}\alpha_i y^{(i)}(x^{(i)})^{\top}x+b.
$$

把 $(x^{(i)})^{\top}x^{(j)}$ 换成 $K(x^{(i)},x^{(j)})$ 即核 SVM，特征空间中的最大间隔分类器不必出现 $\phi$。

> [!INFO]+ 从 Lagrangian 到对偶二次型
>
> $\mathcal{L}=\|w\|_{2}^{2}/2-\sum_i\alpha_i\big[y^{(i)}(w^{\top}x^{(i)}+b)-1\big]$。对 $w$ 求导并令为零：
>
> $$
> w-\sum_i\alpha_i y^{(i)}x^{(i)}=0\implies w=\sum_i\alpha_i y^{(i)}x^{(i)}.
> $$
>
> 对 $b$ 求导：$\sum_i\alpha_i y^{(i)}=0$。把 $w$ 代回 $\mathcal{L}$。二次项给出
>
> $$
> \frac12\|w\|_{2}^{2}=\frac12\sum_{i,j}\alpha_i\alpha_j y^{(i)}y^{(j)}(x^{(i)})^{\top}x^{(j)}.
> $$
>
> 约束项里的 $w^{\top}x^{(i)}$ 再贡献同样一份（带负号），线性项 $\sum_i\alpha_i$ 留下。整理后对偶目标是
>
> $$
> \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j y^{(i)}y^{(j)}(x^{(i)})^{\top}x^{(j)}.
> $$
>
> 约束只剩 $\alpha_i\ge 0$ 与 $\sum_i\alpha_i y^{(i)}=0$。互补松弛 $\alpha_i\big(y^{(i)}(w^{\top}x^{(i)}+b)-1\big)=0$ 说明：不在间隔边界上的点 $\alpha_i=0$，不参与 $w$。
>
> 感性理解：最优法向落在训练点张成的锥里，$w=\sum_i\alpha_i y^{(i)}x^{(i)}$。互补松弛迫使间隔严格大于 $1$ 的点满足 $\alpha_i=0$，故 $w$ 只由支持向量决定。对偶目标中的二次项仅为样本内积；换 $K_{ij}$ 即在特征空间做同一间隔问题，不必写出 $\phi$。

<br>

$b$ 由任意支持向量的等式约束恢复：$b=y^{(s)}-\sum_i\alpha_i y^{(i)}K(x^{(i)},x^{(s)})$。数值上应对多个支持向量取平均，减轻舍入误差。

## 软间隔（soft margin）

线性不可分或需要容忍噪声时，引入松弛 $\xi_i\ge 0$：

$$
\min_{w,b,\xi}\ 
\frac12\|w\|_{2}^{2}+C\sum_{i=1}^{n}\xi_i
\quad\mathrm{s.t.}\quad
y^{(i)}(w^{\top}x^{(i)}+b)\ge 1-\xi_i,\ 
\xi_i\ge 0.
$$

$C>0$ 权衡间隔与违反。$C\rightarrow\infty$ 逼近硬间隔；$C$ 小则允许更多点进入间隔带。对偶只改盒约束：

$$
0\le\alpha_i\le C,\qquad
\sum_{i}\alpha_i y^{(i)}=0.
$$

$\alpha_i=0$：点在间隔外侧且正确； $0<\alpha_i<C$：落在间隔边界上的支持向量；$\alpha_i=C$：违反间隔或被分错的点。

> [!NOTE]+ 软间隔
>
> 允许点进入间隔带或被分错，违反量记为松弛 $\xi_i$，惩罚系数是 $C$。来源是硬间隔在线性不可分或含噪声时无可行解。相对硬间隔：$C<\infty$ 保证二次规划有解，并对离群点不那么敏感。相对交叉熵：远处正确点仍不拉 $w$，只有 $\xi_i>0$ 的点进入对偶的盒约束上界。$C$ 本身是超参数，必须用验证集选。

<br>

等价的无约束写法使用 hinge 损失：

$$
\min_{w,b}\ 
\frac12\|w\|_{2}^{2}
+C\sum_{i=1}^{n}\max\big(0,1-y^{(i)}(w^{\top}x^{(i)}+b)\big).
$$

hinge 在功能间隔大于 $1$ 后梯度为 $0$，因此远处正确点不再拉 $w$；这与交叉熵永远给一点梯度不同。SVM 的输出不是校准概率；需要概率时应另做 Platt scaling，不能把 $w^{\top}x+b$ 直接当 $\sigma$ 的输入而不校验校准。

## SMO

对偶是盒约束上的二次规划。坐标上升每次只改一个 $\alpha_i$，但等式 $\sum\alpha_i y^{(i)}=0$ 使单坐标动不了。SMO（sequential minimal optimization）每次选两个 $\alpha_i,\alpha_j$，在其余固定时解析求解。

设 $i,j$ 被选中，由等式得 $\alpha_j$ 由 $\alpha_i$ 线性确定，再把 $\alpha_i$ 限制在由 $0\le\alpha\le C$ 推出的区间 $[L,H]$ 上，目标成为一元二次，顶点若出界就投影到端点。启发式：优先选违反 KKT 最严重的一对。线性核或核缓存下，SMO 避免 $n\times n$ 稠密求解器。

> [!NOTE]+ SMO
>
> 对偶二次规划每次只动两个 $\alpha$，因为等式 $\sum\alpha_i y^{(i)}=0$ 使单坐标更新不可行。来源是 Platt 对核 SVM 的分解算法，不是新的对偶目标。相对一次性解 $n\times n$ 稠密 QP：内存与时间从整核矩阵降到缓存若干列。相对坐标上升：最小可行工作集大小是 $2$。启发式选违反 KKT 的一对，否则收敛很慢。

<br>

> [!EXAMPLE]+ 例8 二维硬间隔
>
> 正类 $(2,2),(3,3)$，负类 $(0,0),(0,1)$。最优边界大致沿两簇之间，支持向量是离对方最近的那几个点，不是全部四点。若把 $(2,2)$ 挪到 $(10,10)$，它离开间隔带，$\alpha$ 变为 $0$，$w$ 不变。这是支持向量“稀疏”的含义：多数训练点可以删掉而不改决策面。

<br>

## 实现

```python
def linear_soft_svm_subgradient(X, y, C=1.0, lr=0.01, num_steps=2000):
    """y in {-1, +1}。X 不含截距；b 单独更新。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for t in range(1, num_steps + 1):
        margin = y * (X @ w + b)
        viol = margin < 1.0
        grad_w = w - C * (y[viol, None] * X[viol]).sum(axis=0)
        grad_b = -C * y[viol].sum()
        step = lr / np.sqrt(t)
        w -= step * grad_w
        b -= step * grad_b
    return w, b

def svm_dual_objective(alpha, y, K):
    """对偶目标，越大越好。"""
    return float(alpha.sum() - 0.5 * alpha @ ((y[:, None] * K * y[None, :]) @ alpha))

def predict_kernel_svm(X_train, y, alpha, b, x, kernel_fn):
    k = kernel_fn(X_train, x[None, :]).ravel()
    return float(np.sum(alpha * y * k) + b)
```

子梯度法适合教学与线性核；生产环境的核 SVM 应使用 SMO 或 libsvm 一类分解算法，并缓存核列。

> [!NOTE]+ 对照：逻辑回归与 SVM
>
> 二者都能给出线性（或核后的线性）边界。逻辑回归输出校准概率，所有点都拉 $\theta$，线性可分时 $\|\theta\|$ 无界。SVM 用 hinge，间隔外的正确点梯度为 $0$，解由支持向量决定。需要 $P(y=1\mid x)$ 时用逻辑回归或对 SVM 另做校准；需要稀疏对偶与核时用 SVM。

<br>

> [!NOTE]+ 衔接
>
> 核 SVM 用光滑的 $\phi$ 弯曲边界，代价是 $n\times n$ 核矩阵。下一章改用轴对齐切分：不写内积，也能表示非线性规则，但单棵深树方差大。

<br>

# 决策树

## 概念：树

- **问题**
  - 把输入空间递归切成轴对齐区域（axis-aligned regions），每个叶放一个预测（类或均值）；
  - 2025 Autumn 课堂主题，讲义主 PDF 不单列一章。
- **范围**
  - 用信息增益或 Gini 选择切分，并控制深度以防过拟合；
  - 输入特征表，输出一棵树或树的集成前体。
- **章节衔接**
  - 上一章的核 SVM 用光滑间隔切开空间；
  - 本章用轴对齐矩形逼近同一决策，不写 $\theta^{\top}x$；
  - 单棵深树方差大，下一章把它当作弱学习器来加。

切分选择一个特征 $j$ 与阈值 $t$（连续）或子集（离散），把数据分成 $S_L,S_R$。分类常用不纯度：

$$
\mathrm{Gini}(S)=\sum_{c}\hat p_c(1-\hat p_c),\qquad
H(S)=-\sum_{c}\hat p_c\log\hat p_c.
$$

信息增益（information gain）是切分前不纯度减去切分后加权不纯度。贪心每步选增益最大的切。回归用叶内平方误差。

> [!NOTE]+ 决策树
>
> 用轴对齐切分把输入空间递归分成矩形，叶上放类或均值。来源是规则清单与 CART / ID3：每次选让不纯度下降最多的一刀。相对线性或核：不写 $\theta^{\top}x$，对特征单调变换不敏感，能混用类别与缺失；斜边界只能用多层阶梯逼近。相对后面的提升：单棵深树方差大，训练误差可以到 $0$，测试误差通常先升。

<br>

停止条件：深度上限、叶最小样本数、增益下限。不剪枝的深树能记住训练集，训练误差可到 $0$，测试误差通常更差。代价复杂度剪枝用验证集在“叶数”与“训练不纯度”之间折中。

缺失值可用代理切分或把缺失当作单独水平。类别不平衡时，叶预测应报告概率或按代价加权，而不是只报多数类。

轴对齐切分无法用一次切切开斜对角的两类；需要多层阶梯近似。这与线性/核分类器互补：树擅长规则清单与异质表格，不擅长光滑的斜边界。

## 实现

```python
def gini(y):
    if y.size == 0:
        return 0.0
    p = np.bincount(y).astype(float) / y.size
    return float(1.0 - p @ p)

def best_threshold_split(x, y):
    """一维连续特征上的最优阈值。y 为非负整数类标。"""
    order = np.argsort(x)
    x, y = x[order], y[order]
    best, thr = np.inf, None
    for i in range(1, y.size):
        if x[i] == x[i - 1]:
            continue
        impurity = (i * gini(y[:i]) + (y.size - i) * gini(y[i:])) / y.size
        if impurity < best:
            best, thr = impurity, 0.5 * (x[i - 1] + x[i])
    return thr, best
```

> [!EXAMPLE]+ 例9 一次切分
>
> 特征“年龄”，标签违约。阈值 $35$ 把样本分成 $20$ 与 $30$ 人，Gini 从 $0.5$ 降到 $0.32$。信息增益是 $0.18$。若另一特征“收入”增益 $0.25$，根节点选收入。增益大不自动等于因果：收入可能只是职业的代理。

<br>

> [!NOTE]+ 对照：核、树、线性
>
> 线性模型要斜边界一次切成。核把斜边界搬到 $\phi$ 空间。树用多层阶梯逼近斜线，对单调变换不敏感，也难表示“$x_1+x_2>1$”这种简单倾斜。异质表格、缺失、类别特征常先试树；光滑几何关系常先试线性或核。

<br>

> [!NOTE]+ 衔接
>
> 单棵树是强、高方差的假设。下一章把它削弱成桩，再按指数损失逐个加上去。加法模型与后面的网络是同一思路：复杂 $h$ 由简单模块叠加。

<br>

# 提升

## 概念：提升

- **问题**
  - 把多个弱分类器（weak learner，略优于随机）加成强分类器；
  - AdaBoost 按错分的指数损失（exponential loss）重加权样本。
- **范围**
  - 写出 AdaBoost 的权重更新与加法模型解释；
  - 输入弱学习器接口，输出加权投票。
- **章节衔接**
  - 上一章给出可当作 $h_t$ 的切分；
  - 本章在指数损失下做前向阶段极小化，间隔被逐步拉大；
  - 下一章把“相加的弱学习器”换成可微层，用梯度同时更新全部模块。

AdaBoost 维护样本权重 $w_i$，初始均匀。第 $t$ 轮：

1. 在 $w$ 上训练弱分类器 $h_t:\mathcal{X}\rightarrow\{-1,+1\}$；
2. 加权错误率 $\varepsilon_t=\sum_i w_i\mathbf{1}\{h_t(x^{(i)})\ne y^{(i)}\}/\sum_i w_i$；
3. $\alpha_t=\frac12\log\frac{1-\varepsilon_t}{\varepsilon_t}$；
4. $w_i\leftarrow w_i\exp(-\alpha_t y^{(i)}h_t(x^{(i)}))$，再归一化。

最终 $H(x)=\mathrm{sign}(\sum_t\alpha_t h_t(x))$。$\varepsilon_t<1/2$ 时 $\alpha_t>0$；被分错的点权重上升，下一棵树更关注它们。

> [!NOTE]+ 提升
>
> 把多个略优于随机的弱分类器加成一个强分类器，样本权重随错分上升。AdaBoost 来自 Freund 与 Schapire，后来被证明等价于指数损失的前向阶段极小化。相对单棵深树：每步只加一个浅桩，训练误差有乘积上界，方差通常更低。相对后面的梯度提升：指数损失对离群点过陡，可换成 logistic 或 Huber。弱学习器必须真的弱于 $1/2$ 错误率，否则 $\alpha_t\le 0$，加法没有收益。

<br>

> [!INFO]+ $\alpha_t=\frac12\log\frac{1-\varepsilon_t}{\varepsilon_t}$ 从哪来
>
> 指数损失 $L(f)=\sum_i\exp(-y^{(i)}f(x^{(i)}))$。第 $t$ 步令 $f_t=f_{t-1}+\alpha h$，样本权重取 $w_i\propto\exp(-y^{(i)}f_{t-1}(x^{(i)}))$。于是
>
> $$
> L(f_{t-1}+\alpha h)=\sum_i w_i\exp(-\alpha y^{(i)}h(x^{(i)})).
> $$
>
> $y h\in\{+1,-1\}$，把对与错拆开。令 $W=\sum_i w_i$，$\varepsilon=\sum_{i:yh=-1}w_i/W$，则
>
> $$
> L=W\big((1-\varepsilon)e^{-\alpha}+\varepsilon e^{\alpha}\big).
> $$
>
> 对 $\alpha$ 求导并令为零：$(1-\varepsilon)e^{-\alpha}=\varepsilon e^{\alpha}$，两边取对数得 $\alpha=\frac12\log\frac{1-\varepsilon}{\varepsilon}$。权重更新 $w_i\leftarrow w_i\exp(-\alpha y^{(i)}h(x^{(i)}))$ 正是把新的指数因子乘进 $w$，供下一轮使用。
>
> 感性理解：$\alpha_t$ 是该轮弱分类器在加法模型中的最优步长，随 $\varepsilon_t$ 减小而增大。$\varepsilon_t=1/2$ 时 $\alpha_t=0$；$\varepsilon_t>1/2$ 时 $\alpha_t<0$，等价于翻转 $h_t$ 的符号。错分样本的权重乘 $e^{\alpha_t}>1$，下一轮再拟合时这些点占更大比例。指数损失在负间隔上比 hinge 更陡，噪声点的权重会持续上升，提升轮数需要早停或缩小步长。

<br>

梯度提升回归树（GBRT）第 $t$ 步拟合

$$
r_i^{(t)}=-\frac{\partial}{\partial f(x^{(i)})}\mathcal{L}\big(y^{(i)},f_{t-1}(x^{(i)})\big),
$$

再用学习率 $\nu\in(0,1]$ 作 $f_t=f_{t-1}+\nu h_t$。$\nu$ 小、树多通常优于 $\nu=1$、树少。列采样与子采样是随机性正则。

## 实现

```python
def adaboost_stumps(X, y, num_rounds=20):
    """y in {-1, +1}。弱学习器为一维阈值桩。"""
    n, d = X.shape
    w = np.ones(n) / n
    stumps, alphas = [], []
    for _ in range(num_rounds):
        best = None
        for j in range(d):
            thr, _ = best_threshold_split(X[:, j], (y > 0).astype(int))
            if thr is None:
                continue
            for s in (+1, -1):
                pred = np.where(X[:, j] >= thr, s, -s)
                err = float(np.sum(w * (pred != y)))
                if best is None or err < best[0]:
                    best = (err, j, thr, s, pred)
        err, j, thr, s, pred = best
        err = min(max(err, 1e-12), 1.0 - 1e-12)
        alpha = 0.5 * np.log((1.0 - err) / err)
        w *= np.exp(-alpha * y * pred)
        w /= w.sum()
        stumps.append((j, thr, s))
        alphas.append(alpha)
    return stumps, np.asarray(alphas)

def adaboost_predict(X, stumps, alphas):
    scores = np.zeros(X.shape[0])
    for (j, thr, s), a in zip(stumps, alphas):
        scores += a * np.where(X[:, j] >= thr, s, -s)
    return np.sign(scores), scores
```

> [!EXAMPLE]+ 例10 弱学习器必须真的弱而有效
>
> 若桩的加权错误率稳定在 $0.49$，二十轮后组合间隔仍可分开简单的二维两类。若桩错误率 $0.5$，则 $\alpha_t=0$，提升停住。若第一棵树已经训练误差 $0$，后续权重不再有信息，等价于单棵树。弱学习器太强会失去“逐步修补”的意义。

<br>

> [!NOTE]+ 衔接
>
> 提升按阶段把 $h_t$ 冻住再加下一棵。下一章把所有模块同时写成 $g(Wx+b)$，用一条链式法则更新。非线性的来源从“切分”换成“逐点激活”。

<br>

# 神经网络

## 概念：网络

- **问题**
  - 把仿射变换（affine map）与逐点非线性堆成 $h_{\theta}(x)$，以逼近非线性 $y$；
  - 反向传播（backpropagation）用链式法则把 $\partial J/\partial(\cdot)$ 送到每个模块。
- **范围**
  - 写清一层的前向/反向，以及整网向量化；
  - 输入 $X$，输出预测与参数梯度。
- **章节衔接**
  - 核把非线性放进 $\phi$，树放进切分；本章放进 $g(z)$ 并反传；
  - 无非线性时多层仿射等价于一层；
  - 下一章解释：训练损失小不等于存在风险小。

一层：

$$
z=Wx+b,\qquad a=g(z).
$$

$g$ 常用 ReLU $g(z)=\max(0,z)$、sigmoid、tanh。两层网络

$$
a^{[1]}=g(W^{[1]}x+b^{[1]}),\qquad
\hat y=g_{\mathrm{out}}(W^{[2]}a^{[1]}+b^{[2]}).
$$

二分类 $g_{\mathrm{out}}=\sigma$，多类 $g_{\mathrm{out}}=\mathrm{softmax}$，回归 $g_{\mathrm{out}}$ 为恒等。参数量约为 $\sum_l n^{[l]}(n^{[l-1]}+1)$。足够宽的单隐层可逼近紧集上的连续函数，但宽度可能指数级，且训练不保证找到该逼近。

> [!NOTE]+ 多层网络
>
> 仿射与逐点非线性交替堆叠，用反向传播同时更新全部层。来源是感知机加上可微激活与链式法则，不是核或树的改写。相对核：非线性在 $g$ 里，不形成 $n\times n$ 矩阵，预测不必回看训练点。相对提升：模块可微，全部参数一起走梯度，而不是逐个冻结再加。无非线性时多层仿射塌成一层，深度本身没有表达力。

<br>

现代模块还包括 Dropout、BatchNorm、残差 $y=F(x)+x$、注意力。CS229 要求能把这些模块的局部 Jacobian 接到总损失上；卷积与 Transformer 的尺寸公式在 TA 讲座与后续 LLM 章展开。

## 反向

令 $J$ 为平均损失。对仿射 $z=Wx+b$：

$$
\frac{\partial J}{\partial W}=\frac{\partial J}{\partial z}\,x^{\top},\qquad
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial z},\qquad
\frac{\partial J}{\partial x}=W^{\top}\frac{\partial J}{\partial z}.
$$

对逐点 $a=g(z)$：

$$
\frac{\partial J}{\partial z}=\frac{\partial J}{\partial a}\odot g'(z).
$$

sigmoid + 交叉熵或 softmax + 交叉熵在输出层简化为 $\partial J/\partial z=\hat y-y$（one-hot 或标量）。不要再乘一次 $g'$。

向量化：令 $X$ 为 $(d,n)$ 或按 CS229 常用的 batch 在前。关键是 $W$ 左乘激活时，

$$
\mathrm{d}W=\mathrm{d}Z\,A^{\top}/n,\qquad
\mathrm{d}A=W^{\top}\mathrm{d}Z.
$$

实现时每个数组旁写 shape，并用 `assert dW.shape == W.shape`。

> [!INFO]+ 链式法则只需局部 Jacobian
>
> $J$ 对 $W^{[1]}$ 的依赖路径是 $W^{[1]}\rightarrow z^{[1]}\rightarrow a^{[1]}\rightarrow\cdots\rightarrow J$。不必一次写出总公式。每个模块只实现 `backward(d_out, cache) -> d_in, d_params`。这与自动微分的 reverse-mode 相同：标量损失、高维参数时，反向只需一次扫描。漏乘 $g'$、忘记除以 $n$、把 $A^{\top}$ 写成 $A$，是三种最常见的实现错误。
>
> 矩阵情形：$\mathrm{d}Z=\mathrm{d}A\odot g'(Z)$，$\mathrm{d}W=\mathrm{d}Z\,A_{\mathrm{prev}}^{\top}/n$。形状必须满足 $(n,n^{[l]})=(n,n^{[l]})\times(n^{[l]},n^{[l-1]})$ 的右乘。
>
> 感性理解：反向模式把标量 $J$ 对中间量的偏导从输出层传回输入。每个模块只需要局部 Jacobian：上游传来的 $\mathrm{d}_{\mathrm{out}}$ 左乘或按元素乘 $g'$，再分配给输入与参数。线性层 $Z=A_{\mathrm{prev}}W$ 对 $W$ 的梯度是 $\mathrm{d}Z\,A_{\mathrm{prev}}^{\top}$，形状由矩阵乘法决定。各模块局部正确则总梯度正确，不必展开整条计算图。

<br>

初始化不能全零：隐层对称，所有神经元收到同一梯度。He 初始化对 ReLU 使用 $\mathrm{Var}(W)=2/n_{\mathrm{in}}$；Xavier 对饱和激活使用 $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$。学习率过大出现 `nan`；过小则数百步不动。

## 实现

```python
def relu(Z):
    return np.maximum(0.0, Z)

def mlp_forward(X, params):
    """X: (n, d)。params 含 W1,b1,W2,b2。"""
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    return Z2, (X, Z1, A1)

def mlp_backward(dZ2, caches, params):
    X, Z1, A1 = caches
    n = X.shape[0]
    dW2 = (A1.T @ dZ2) / n
    db2 = dZ2.sum(axis=0) / n
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (X.T @ dZ1) / n
    db1 = dZ1.sum(axis=0) / n
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def he_init(d_in, d_out, rng):
    return rng.normal(scale=np.sqrt(2.0 / d_in), size=(d_in, d_out))
```

> [!EXAMPLE]+ 例11 异或
>
> 四点 $(\pm 1,\pm 1)$ 的异或不能被一条直线切开。单隐层两个 ReLU 单元可以把平面折成两块，再由输出层拼出异或。若隐层激活改成恒等，两层合成一层，四点仍分不开。训练损失有下界不是优化失败，而是假设类不够。

<br>

> [!NOTE]+ 对照：核、提升、网络
>
> 三条路都在放大假设类。核：特征维上升，优化仍常是凸的。提升：凸的指数或 logistic 损失，弱学习器离散。网络：参数与损失对 $\theta$ 非凸，但反向传播把所有层一起动。训练误差到 $0$ 只说明 $\mathcal{H}$ 够大或优化走到了插值，下一步必须谈 $R$ 而不是 $\hat R_n$。

<br>

> [!NOTE]+ 衔接
>
> 网络（以及此前所有模型）都能把训练集拟合到很低。下一章把经验风险与期望风险拆开，并用偏差-方差、VC 维描述“假设类有多丰富、样本要多少”。

<br>

# 泛化

## 概念：泛化

- **问题**
  - 训练误差小不等于对新样本误差小；
  - 偏差（bias）、方差（variance）、样本复杂度（sample complexity）描述假设类（hypothesis class）、数据量与风险的关系。
- **范围**
  - 写出回归的偏差-方差分解，并给出有限/无限假设类的均匀收敛轮廓；
  - 输入是假设类 $\mathcal{H}$ 与样本数 $n$，输出是对 excess risk 的定性与定量界。
- **章节衔接**
  - 前文章节都在最小化 $\hat R_n$；本章把它与 $R$ 分开；
  - 偏差-方差、VC 与双重下降描述容量、样本与风险；
  - 下一章给出可操作的惩罚、早停与交叉验证。

期望风险（expected risk）与经验风险（empirical risk）：

$$
R(h)=E_{(x,y)\sim\mathcal{D}}\big[\ell(h(x),y)\big],\qquad
\hat R_n(h)=\frac1n\sum_{i=1}^{n}\ell(h(x^{(i)}),y^{(i)}).
$$

ERM 在 $\mathcal{H}$ 上最小化 $\hat R_n$。关心的是 $R(\hat h)$，不是 $\hat R_n(\hat h)$。

> [!NOTE]+ 期望风险与经验风险
>
> $R$ 是未知分布 $\mathcal{D}$ 上的期望损失，$\hat R_n$ 是训练集上的平均损失。ERM 最小化后者，真正要控制的是前者。来源是把“训练误差小”和“新样本上误差小”拆开，否则无法讨论过拟合。相对只报训练 $J$：$\hat R_n$ 下降不蕴含 $R$ 下降。下一节的偏差-方差与再下一节的均匀收敛，分别从估计波动和假设类大小两条路逼近这个缺口。

<br>

## 偏差方差（bias-variance）

回归中平方损失、数据 $(x,y)$、$y=f(x)+\varepsilon$、$E[\varepsilon\mid x]=0$。对固定 $x$，把模型输出看成训练集的随机变量 $\hat f(x)$，则

$$
E\big[(y-\hat f(x))^{2}\big]
=
\big(f(x)-E[\hat f(x)]\big)^{2}
+E\big[(\hat f(x)-E[\hat f(x)])^{2}\big]
+\sigma^{2}.
$$

三项依次是偏差平方、方差、不可约噪声。高偏差：直线拟合抛物线，换训练集也系统性地弯错。高方差：五次多项式随样本跳，预测乱晃。不可约噪声：即使 $\hat f=f$ 也消不掉。

> [!NOTE]+ 偏差与方差
>
> 把测试平方误差拆成：平均模型离真实有多远、换训练集模型跳多远、标签自身抖动。来源是对 $E[(y-\hat f)^{2}]$ 做代数展开，不是另设的三种错误类型。相对只说“过拟合”：可以判断下一步该换假设类（压偏差）还是加数据 / 加惩罚（压方差）。$\sigma^{2}$ 与模型无关，再复杂的 $\hat f$ 也消不掉。双重下降不推翻这三项，只说明“容量”不能单用参数个数度量。

<br>

> [!INFO]+ 平方误差怎样拆成三项
>
> 固定 $x$，令 $\bar f=E[\hat f(x)]$（对训练集取期望），$y=f(x)+\varepsilon$，$E[\varepsilon]=0$ 且 $\varepsilon$ 与 $\hat f$ 独立。展开
>
> $$
> (y-\hat f)^{2}=(f+\varepsilon-\hat f)^{2}=(f-\bar f)^{2}+(\bar f-\hat f)^{2}+\varepsilon^{2}+2\text{交叉项}.
> $$
>
> 对训练集与 $\varepsilon$ 取期望。$E[\varepsilon]=0$ 使 $2\varepsilon(f-\hat f)$ 消失；$E[\hat f-\bar f]=0$ 使 $2(f-\bar f)(\bar f-\hat f)$ 消失。剩下
>
> $$
> E[(y-\hat f)^{2}]=(f-\bar f)^{2}+E[(\hat f-\bar f)^{2}]+\sigma^{2}.
> $$
>
> 第一项是“平均模型离真实有多远”，第二项是“换一袋数据模型跳多远”，第三项是标签自身的抖动。
>
> 感性理解：对固定 $x$，把 $\hat f(x)$ 看作训练集的随机变量。$E[\hat f]$ 相对 $f$ 的偏移是偏差，$E[(\hat f-E[\hat f])^{2}]$ 是该估计随样本的波动。线性模型偏差大、方差小；高次多项式相反。增大 $n$ 主要压方差；更换假设类主要压偏差。$\sigma^{2}$ 来自标签噪声，与 $\hat f$ 无关，不能靠加复杂模型消去。

<br>

学习曲线：固定模型，增大 $n$，训练误差上升、验证误差下降并靠拢，间隙是方差的经验影子。固定 $n$，增大容量，训练误差下降、验证误差先降后升，这是经典 U 形。

## 双重下降（double descent）

在插值阈值附近（模型刚好能记住训练集）测试误差往往出现峰值；再增大模型，测试误差可以再次下降。过参数化区存在许多插值解，梯度下降的隐式偏置（例如最小范数）会挑出更平滑的一条。双重下降不推翻偏差-方差，而是说“容量”不能只用参数个数一条轴；有效容量还依赖优化路径与数据谱。若测试误差落在插值阈值附近的峰值上，加宽网络、加强正则或增加样本，都有可能离开该区域。

> [!NOTE]+ 双重下降
>
> 测试误差在模型刚能插值训练集时升高，过了该阈值后再降。来源是 Belkin 等人对现代过参数化实践与经典 U 形的对照，不是新的损失定义。相对经典偏差-方差图：参数个数不再是容量的唯一坐标，优化路径的隐式偏置会在许多插值解里挑一条更平滑的。相对“更大一定过拟合”：过参数化区加宽网络有时比停在插值阈值更稳。仍要用验证集确认，不能把下降当成定理。

<br>

## 样本复杂度（sample complexity）

目标：以高概率对所有 $h\in\mathcal{H}$ 同时有 $|\hat R_n(h)-R(h)|\le\varepsilon$，从而 ERM 的 excess risk 也小。

**有限 $\mathcal{H}$。** 令 $|\mathcal{H}|=m$。对单个 $h$，Hoeffding 给出

$$
P\big(|\hat R_n(h)-R(h)|>\varepsilon\big)\le 2e^{-2n\varepsilon^{2}}
$$

（损失在 $[0,1]$ 时）。对 $m$ 个假设做并界，再令失败概率 $\le\delta$，需要

$$
n\ge\frac{1}{2\varepsilon^{2}}\log\frac{2m}{\delta}.
$$

样本数随 $\log|\mathcal{H}|$ 增长。这解释了“多试几个模型”必须用验证集付对数代价，而不是在测试集上付。

> [!INFO]+ 从 Hoeffding 到 $n\ge\frac1{2\varepsilon^{2}}\log\frac{2m}{\delta}$
>
> 损失落在 $[0,1]$ 时，单条假设的经验风险是 $n$ 个有界随机变量的均值。Hoeffding 给出
>
> $$
> P\big(|\hat R_n(h)-R(h)|>\varepsilon\big)\le 2\exp(-2n\varepsilon^{2}).
> $$
>
> $\mathcal{H}$ 里有 $m$ 个假设，关心的是“至少有一个 $h$ 偏离超过 $\varepsilon$”。并界（union bound）
>
> $$
> P(\exists h:|\hat R_n-R|>\varepsilon)\le 2m\exp(-2n\varepsilon^{2}).
> $$
>
> 令右端 $\le\delta$，取对数：$2n\varepsilon^{2}\ge\log(2m/\delta)$，即 $n\ge\frac1{2\varepsilon^{2}}\log\frac{2m}{\delta}$。指数里的 $2$ 来自 Hoeffding 对 $[0,1]$ 变量的常数；换成其他有界区间只改常数，不改 $\log m$ 与 $1/\varepsilon^{2}$ 的形状。
>
> 感性理解：单条假设的 $\hat R_n$ 以指数速度集中到 $R$。同时控制 $|\mathcal{H}|=m$ 条假设时，失败事件要做并界，样本需求只增加 $\log m$。模型选择应在验证集上支付这项对数代价；测试集若再用来选模型，并界中的 $m$ 继续变大，$\varepsilon$ 不再对应所报数字。

<br>

**无限 $\mathcal{H}$。** 用 VC 维 $d_{\mathrm{VC}}$ 描述打散能力：能实现所有 $2^{d}$ 种标记的最大点集大小。线性分类器在 $\mathbb{R}^{d}$ 的 VC 维是 $d+1$（含截距）。均匀收敛界的形状为

$$
R(h)\le\hat R_n(h)+O\Big(\sqrt{\frac{d_{\mathrm{VC}}\log n+\log(1/\delta)}{n}}\Big).
$$

VC 维有限则一致可学；无限则存在分布使任意大 $n$ 仍可被打穿。RBF-SVM 的假设类可以很丰富，有效容量由核带宽与 $C$ 共同限制，不能只看“核是无限维”。

> [!NOTE]+ VC 维
>
> 假设类能打散的最大点集大小，用来给无限 $\mathcal{H}$ 标一个有限容量。来源是 Vapnik-Chervonenkis 的均匀收敛理论：有限类用 $\log|\mathcal{H}|$，无限类用打散数代替计数。相对 Hoeffding 并界：$|\mathcal{H}|=\infty$ 时不能再对每个假设加一项。相对参数个数：线性分类器的 VC 维是 $d+1$，与是否过参数化的现代网络不是同一尺度。界的常数通常松，用来判断 $1/\sqrt{n}$ 与 $\log$ 的形状，不用来预报精确测试误差。

<br>

> [!WARNING]+ 界是最坏情形
>
> PAC / VC 界给出充分条件，常数往往松。真实图像上深度网上的泛化远好于按参数个数代入 VC 界的预测。界用来判断论证方向（容量、对数、$\sqrt{1/n}$），不用来预报精确测试误差。

<br>

> [!EXAMPLE]+ 例12 有限假设类点数
>
> $\mathcal{H}$ 有 $1000$ 个阈值分类器，$\varepsilon=0.05$，$\delta=0.01$。$n\ge \frac{1}{2\cdot 0.0025}\log(2\cdot 1000/0.01)\approx 200\cdot\log(2\cdot 10^{5})\approx 2440$。把阈值换成连续区间后 $|\mathcal{H}|=\infty$，必须改用 VC：一维阈值的 VC 维为 $1$，同样 $\varepsilon$ 下 $n$ 仍是几千量级而不是无穷。

<br>

> [!NOTE]+ 衔接
>
> 泛化章给出 excess risk 随容量与 $n$ 的变化。下一章用 $\lambda$、早停与交叉验证来限制有效容量。若不留验证折，$\lambda$ 本身也会被训练集拟合。

<br>

# 正则与选择

## 概念：正则

- **问题**
  - 在经验风险上加约束或惩罚，降低有效容量（effective capacity）；
  - 用验证或交叉验证（cross-validation）选择惩罚强度，不用训练集直接选。
- **范围**
  - 写出 $L_2$/$L_1$、早停与贝叶斯对应，并给出交叉验证流程；
  - 输入候选超参数，输出验证风险最小的模型。
- **章节衔接**
  - 上一章给出 excess risk 的形状；
  - 本章用 $\lambda$、验证折与先验把容量按下去；
  - 下一章没有 $y$，但选择 $k$ 仍走同一“留出一块数据”的逻辑。

岭回归（ridge regression）：

$$
J_{\lambda}(\theta)=\frac12\|X\theta-\vec y\|_{2}^{2}+\frac{\lambda}{2}\|\theta\|_{2}^{2},
\qquad
\theta=(X^{\top}X+\lambda I)^{-1}X^{\top}\vec y.
$$

$\lambda I$ 使 $X^{\top}X$ 可逆，并把 $\theta$ 拉向 $0$。通常不惩罚截距。Lasso 用 $\lambda\|\theta\|_1$，最优解稀疏，兼做特征选择。弹性网同时加 $L_1$ 与 $L_2$。

> [!NOTE]+ 岭回归与 Lasso
>
> 在经验风险上加 $\|\theta\|_{2}^{2}$ 或 $\|\theta\|_{1}$，把有效容量按下去。岭来自 Hoerl 与 Kennard，也对应高斯先验下的 MAP；Lasso 对应 Laplace 先验，并给出稀疏解。相对裸的正规方程：$\lambda I$ 保证可逆，并减小高方差方向上的系数。相对只靠早停：惩罚写在目标里，与优化器无关。$\lambda$ 不能在训练集上选，否则等于没有惩罚。

<br>

逻辑回归同样可加 $\frac{\lambda}{2}\|\theta\|_{2}^{2}$，既避免线性可分时 $\|\theta\|\rightarrow\infty$，也降低方差。

隐式正则：早停、SGD 噪声、最小范数插值都在不写 $\lambda$ 的情况下偏向简单解。同一架构、不同优化器，测试误差可以不同，因为隐式偏置不同。

## 交叉验证（cross-validation）

把数据分成训练 / 验证 / 测试。超参数只许看验证集。$k$ 折：轮流把 $k$ 块之一当验证，平均 $k$ 个分数后再在全部训练数据上用选定的 $\lambda$ 重训。测试集只许用一次，用来报告。

> [!NOTE]+ 交叉验证
>
> 用未参与拟合的一块数据估计超参数下的风险，再在测试集上报告一次。来源是：$\hat R_n$ 对用来选择 $\lambda$ 的那份数据是乐观的。相对单次训练/测试切分：$k$ 折降低验证分数的方差，小样本上更稳。相对在测试集上调参：并界中的假设个数继续变大，所报数字不再对应 $R$。时间顺序数据必须按时间切，随机打乱会把未来泄漏进训练。

<br>

嵌套交叉验证在小样本上估计“带选择过程的”泛化，计算更贵。时间序列必须按时间切，不能随机打乱，否则未来泄漏。

## 贝叶斯

高斯先验 $\theta\sim\mathcal{N}(0,\tau^{2}I)$ 配高斯似然时，MAP 估计恰为岭回归，$\lambda$ 与 $1/\tau^{2}$ 成正比。Laplace 先验对应 Lasso。贝叶斯观点把 $\lambda$ 对应到先验方差 $1/\tau^{2}$。$\tau$ 仍需选择，交叉验证仍然必要。

全贝叶斯预测积掉 $\theta$：$p(y\mid x,\mathrm{data})=\int p(y\mid x,\theta)p(\theta\mid\mathrm{data})\,d\theta$。MAP 只取后验众数，忽略参数不确定度。线性高斯可以闭式；一般模型用变分或 MCMC。

## 实现

```python
def ridge_cv(X, y, lambdas, k=5, rng=None):
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    scores = []
    for lam in lambdas:
        errs = []
        for i in range(k):
            va = folds[i]
            tr = np.concatenate([folds[j] for j in range(k) if j != i])
            theta = linear_regression_normal_eq(X[tr], y[tr], reg=lam)
            pred = X[va] @ theta
            errs.append(np.mean((pred - y[va]) ** 2))
        scores.append(float(np.mean(errs)))
    return lambdas[int(np.argmin(scores))], scores
```

> [!EXAMPLE]+ 例13 不要在测试集上选 $\lambda$
>
> 十个 $\lambda$ 在测试集上挑最小误差，相当于对测试集做了十次假设检验，报告值乐观。正确流程：验证集选 $\lambda=10^{-2}$，再在测试集上只报一次。若测试误差仍差，应回到特征与模型设定，而不是继续扫测试集。

<br>

> [!NOTE]+ 对照：显式正则与隐式正则
>
> 岭与 Lasso 把惩罚写进 $J$。早停、SGD 噪声、最小范数插值不写 $\lambda$，却同样偏向简单解。双重下降说明：过参数化区的测试误差可以再降，前提是优化路径挑中了“简单”的插值。换优化器而不换 $\mathcal{H}$，测试误差仍可能变，因为隐式偏置变了。

<br>

> [!NOTE]+ 衔接
>
> 有监督部分到此为止：假设、求解、容量与模型选择。下一章不再使用 $y$。簇数 $k$ 不能靠最小化训练失真来定，否则最优是 $k=n$。需要留出数据，或检查聚类在重抽样下是否稳定。

<br>

# 聚类

## 概念：聚类

- **问题**
  - 无标签时把 $\{x^{(i)}\}$ 分成 $k$ 组，使组内相似、组间不相似；
  - $k$-means 最小化到组中心的平方距离（distortion）。
- **范围**
  - 写出失真函数与坐标下降；
  - 输入 $X$ 与 $k$，输出中心与分配。
- **章节衔接**
  - 以上为有监督：目标、求解与选择协议；
  - 本章在没有 $y$ 时最小化到中心的平方距离；
  - 下一章把硬分配 $c^{(i)}$ 换成后验责任度 $w_j^{(i)}$。

失真：

$$
J(c,\mu)=\sum_{i=1}^{n}\|x^{(i)}-\mu_{c^{(i)}}\|_{2}^{2}.
$$

$c^{(i)}\in\{1,\ldots,k\}$ 是分配，$\mu_j$ 是第 $j$ 组中心。$J$ 对 $(c,\mu)$ 非凸。Lloyd 算法交替：

1. 分配：$c^{(i)}=\arg\min_j\|x^{(i)}-\mu_j\|_{2}^{2}$；
2. 更新：$\mu_j=$ 分到 $j$ 的点的均值；空簇需重初始化。

每步不增 $J$，有限配置下必停，但停在局部最小。应多次随机初始化（或 $k$-means++）取最小 $J$。$k$ 不能靠最小化 $J$ 选择：$k=n$ 时 $J=0$。用肘部、轮廓系数、或下游任务指标，并配合稳定性（bootstrap 后簇是否还在）。

> [!NOTE]+ $k$-means
>
> 把点分给最近中心，再把中心更新为组内均值，交替到失真不再下降。来源是向量量化与 Lloyd 算法，目标是 $L_2$ 失真，不是似然。相对有监督分类：没有 $y$，不能用交叉熵选 $k$。相对下一章的 GMM：分配是 $0/1$，簇被当成各向同性球。$J$ 非凸，必须多随机启动；空簇要重抽样。

<br>

$k$-means 假设簇大致球形、方差相近。长条、环形、不同密度的簇会切错。特征必须同一尺度，否则“距离”被某一维主导。

## 实现

```python
def kmeans(X, k, num_restarts=10, max_iter=100, rng=None):
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    best = None
    for _ in range(num_restarts):
        mu = X[rng.choice(n, size=k, replace=False)]
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
            c = d2.argmin(axis=1)
            new_mu = np.stack([
                X[c == j].mean(axis=0) if np.any(c == j) else mu[j]
                for j in range(k)
            ])
            if np.allclose(new_mu, mu):
                break
            mu = new_mu
        J = float(((X - mu[c]) ** 2).sum())
        if best is None or J < best[0]:
            best = (J, mu, c)
    return best
```

> [!EXAMPLE]+ 例14 空簇
>
> 初始化把两个中心放在同一稠密团外，所有点分给更近的那个，另一中心永远分不到点。实现必须检测空簇并重新抽样，否则 `mean` 遇到空切片会得到 `nan`，后续距离全坏。

<br>

> [!NOTE]+ 衔接
>
> $k$-means 的分配是 $0/1$。下一章引入潜变量 $z$，用后验 $p(z\mid x)$ 做软分配，并证明 EM 每步不降低边缘似然。GMM 可看作带协方差的软分配聚类。

<br>

# EM

## 概念：EM

- **问题**
  - 观测 $x$ 与潜变量（latent variable）$z$ 的联合模型 $p(x,z;\theta)$，边缘似然 $p(x;\theta)=\sum_z p(x,z;\theta)$ 难优化；
  - EM（expectation-maximization）交替构造下界并抬升下界。
- **范围**
  - 从 Jensen 推出 ELBO，再特化到高斯混合；
  - 输入不完全数据，输出 $\theta$ 与后验责任度。
- **章节衔接**
  - 上一章的 $J(c,\mu)$ 是完全数据平方损失；
  - 本章对边缘似然构造 ELBO，GMM 是软分配特例；
  - 下一章的 PCA / 因子分析是另一类潜变量：连续 $z$，不是离散簇。

> [!NOTE]+ EM
>
> 边缘似然 $\log p(x)=\log\sum_z p(x,z)$ 不好直接最大化时，交替：用当前 $\theta$ 算 $p(z\mid x)$（E），再最大化完全数据似然的期望（M）。来源是 Dempster 等人对不完全数据的最大似然，下界就是 ELBO。相对硬分配的 $k$-means：重叠点以分数责任度同时更新多个分量，似然单调不降。相对直接对 $\log\sum_z$ 做梯度：E 步在指数族上常有闭式，M 步回到完全数据 MLE。不保证全局最大。

<br>

## Jensen

对凸 $f$，$f(E[t])\le E[f(t)]$；$\log$ 凹，故 $\log E[t]\ge E[\log t]$。对正数 $t(z)$ 与分布 $Q$，

$$
\log\sum_z Q(z)\frac{p(x,z;\theta)}{Q(z)}
\ge
\sum_z Q(z)\log\frac{p(x,z;\theta)}{Q(z)}.
$$

左边是 $\log p(x;\theta)$。右边定义为

$$
\mathrm{ELBO}(x;Q,\theta)
=
\sum_z Q(z)\log\frac{p(x,z;\theta)}{Q(z)}
=
E_{z\sim Q}[\log p(x,z;\theta)]+H(Q).
$$

对任意 $Q$，$\log p(x;\theta)\ge\mathrm{ELBO}(x;Q,\theta)$。缺口是 $\mathrm{KL}(Q\|p(z\mid x;\theta))$，因此当 $Q=p(z\mid x;\theta)$ 时等号成立。

> [!INFO]+ Jensen 怎样把边缘似然托成下界
>
> 边缘似然 $p(x)=\sum_z p(x,z)$。随便选一个分布 $Q(z)>0$，改写成加权平均：
>
> $$
> \log p(x)=\log\sum_z Q(z)\frac{p(x,z)}{Q(z)}=\log E_{z\sim Q}\Big[\frac{p(x,z)}{Q(z)}\Big].
> $$
>
> $\log$ 凹，Jensen 给出 $\log E[t]\ge E[\log t]$，于是
>
> $$
> \log p(x)\ge E_Q\big[\log p(x,z)-\log Q(z)\big]=\mathrm{ELBO}(Q,\theta).
> $$
>
> 缺口可直接算出来：
>
> $$
> \log p(x)-\mathrm{ELBO}=\sum_z Q(z)\log\frac{Q(z)}{p(z\mid x)}=D_{\mathrm{KL}}(Q\|p_{z\mid x})\ge 0.
> $$
>
> E 步把 $Q$ 设成当前后验，KL 归零，下界贴住 $\ell(\theta^{(t)})$。M 步抬高下界；由于 $\ell$ 始终在下界上方，似然不会下降。
>
> 感性理解：$\log p(x)$ 要对潜变量边缘化，直接求和或积分通常不可行。引入任意 $Q$ 后，Jensen 给出可计算的下界；缺口等于 $D_{\mathrm{KL}}(Q\|p_{z\mid x})$。E 步取 $Q=p(z\mid x;\theta^{(t)})$，下界与 $\ell(\theta^{(t)})$ 相切；M 步抬高下界。由于 $\ell$ 始终在 ELBO 上方，边缘似然单调不降，但不保证到达全局最大。

<br>

多样本：

$$
\ell(\theta)=\sum_{i=1}^{n}\log p(x^{(i)};\theta)
\ge
\sum_{i=1}^{n}\mathrm{ELBO}(x^{(i)};Q_i,\theta).
$$

EM：

- E 步：令 $Q_i^{(t)}(z)=p(z\mid x^{(i)};\theta^{(t)})$，此时下界贴住 $\ell(\theta^{(t)})$；
- M 步：$\theta^{(t+1)}=\arg\max_{\theta}\sum_i\mathrm{ELBO}(x^{(i)};Q_i^{(t)},\theta)$，即最大化完全数据对数似然在 $Q_i$ 下的期望。

单调性：$\ell(\theta^{(t+1)})\ge\mathrm{ELBO}(Q^{(t)},\theta^{(t+1)})\ge\mathrm{ELBO}(Q^{(t)},\theta^{(t)})=\ell(\theta^{(t)})$。EM 不保证全局最大，只保证似然不降。

ELBO 的另两种写法：

$$
\begin{aligned}
\mathrm{ELBO}
&=
E_Q[\log p(x\mid z;\theta)]-\mathrm{KL}(Q\|p_z),\\
\mathrm{ELBO}
&=
\log p(x;\theta)-\mathrm{KL}(Q\|p_{z\mid x}).
\end{aligned}
$$

第二式直接说明最优 $Q$ 是真实后验。

## 高斯混合（Gaussian mixture）

$$
p(z^{(i)}=j)=\phi_j,\qquad
x^{(i)}\mid z^{(i)}=j\sim\mathcal{N}(\mu_j,\Sigma_j).
$$

E 步的责任度（responsibilities）

$$
w_j^{(i)}
=
p(z^{(i)}=j\mid x^{(i)};\theta)
=
\frac{\phi_j\,\mathcal{N}(x^{(i)};\mu_j,\Sigma_j)}{\sum_{\ell}\phi_{\ell}\,\mathcal{N}(x^{(i)};\mu_{\ell},\Sigma_{\ell})}.
$$

M 步与带权的 GDA 更新相同：

$$
\begin{aligned}
\phi_j&=\frac1n\sum_i w_j^{(i)},\\
\mu_j&=\frac{\sum_i w_j^{(i)}x^{(i)}}{\sum_i w_j^{(i)}},\\
\Sigma_j&=\frac{\sum_i w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^{\top}}{\sum_i w_j^{(i)}}.
\end{aligned}
$$

$k$-means 可视为协方差各向同性、责任度变硬（$0/1$）的极限。GMM 能表达椭圆簇与重叠簇；代价是局部最优与奇异协方差（某分量贴住一个点，$\Sigma\rightarrow 0$，似然 $\rightarrow\infty$）。应对 $\Sigma$ 加岭、设最小特征值，或多随机启动。

> [!INFO]+ 责任度与 M 步加权公式
>
> 完全数据对数似然是 $\sum_i\sum_j\mathbf{1}\{z^{(i)}=j\}\big(\log\phi_j+\log\mathcal{N}(x^{(i)};\mu_j,\Sigma_j)\big)$。E 步用后验替换指示变量：
>
> $$
> w_j^{(i)}=E[\mathbf{1}\{z^{(i)}=j\}\mid x^{(i)}]=p(z^{(i)}=j\mid x^{(i)}).
> $$
>
> Bayes：先验 $\phi_j$ 乘该类高斯密度，再对 $j$ 归一化，即正文中的 $w_j^{(i)}$。M 步对 $\phi,\mu,\Sigma$ 分别求最大。$\phi$ 在单纯形上，拉格朗日给出 $\phi_j=\frac1n\sum_i w_j^{(i)}$。$\mu_j$ 的驻点是加权均值
>
> $$
> \mu_j=\frac{\sum_i w_j^{(i)}x^{(i)}}{\sum_i w_j^{(i)}}.
> $$
>
> $\Sigma_j$ 同理，是相对 $\mu_j$ 的加权散度矩阵。这与 GDA 的闭式解相同，只是每点以分数 $w$ 同时属于各类。
>
> 感性理解：$w_j^{(i)}$ 是点 $i$ 属于分量 $j$ 的后验质量。M 步把完全数据的 MLE 换成按该质量加权的均值与协方差，形式与 GDA 相同。重叠区域的点对多个分量同时贡献分数权重。$k$-means 是各向同性协方差、后验取 $0/1$ 的极限，重叠点只能整份归入一个中心。

<br>

## 变分（variational inference）

当 $p(z\mid x)$ 不可解析时，在受限族 $\mathcal{Q}$（例如平均场 $Q(z)=\prod_j Q_j(z_j)$）上最大化 ELBO。变分自编码器把 $Q(z\mid x)$ 参数化为编码器网络，用重参数化估计对 $z$ 的期望，解码器建 $p(x\mid z)$。训练目标仍是 ELBO，不是随意的重构平方差；KL 项防止编码器把后验坍缩到与先验无关的点。

## 实现

```python
def log_gaussian(X, mu, Sigma, ridge=1e-6):
    d = X.shape[1]
    S = Sigma + ridge * np.eye(d)
    diff = X - mu
    sol = np.linalg.solve(S, diff.T).T
    sign, logdet = np.linalg.slogdet(S)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + np.sum(diff * sol, axis=1))

def gmm_em(X, k, num_iters=50, ridge=1e-6, rng=None):
    rng = np.random.default_rng(rng)
    n, d = X.shape
    mu = X[rng.choice(n, size=k, replace=False)]
    Sigma = np.stack([np.cov(X, rowvar=False) + ridge * np.eye(d) for _ in range(k)])
    phi = np.ones(k) / k
    for _ in range(num_iters):
        logcomp = np.stack([np.log(phi[j]) + log_gaussian(X, mu[j], Sigma[j], ridge) for j in range(k)], axis=1)
        m = logcomp.max(axis=1, keepdims=True)
        resp = np.exp(logcomp - m)
        resp /= resp.sum(axis=1, keepdims=True)
        n_k = resp.sum(axis=0) + 1e-12
        phi = n_k / n
        mu = (resp.T @ X) / n_k[:, None]
        for j in range(k):
            dlt = X - mu[j]
            Sigma[j] = (dlt.T @ (dlt * resp[:, j:j + 1])) / n_k[j] + ridge * np.eye(d)
    return phi, mu, Sigma, resp
```

> [!EXAMPLE]+ 例15 两个高斯重叠
>
> 数据来自 $0.5\,\mathcal{N}((-2,0),I)+0.5\,\mathcal{N}((2,0),I)$。E 步给左峰点高 $w_1$，右峰点高 $w_2$，中间点约 $0.5$。M 步把中心拉回 $\pm 2$。若初始化两个中心都在右侧，可能两个分量抢同一簇，另一簇被忽略，似然停在较差局部最优。多次重启并比较 $\ell$ 是标准补救。

<br>

> [!NOTE]+ 对照：硬分配与软分配
>
> $k$-means 每点只属于一个中心，目标是失真 $J$。GMM 每点对每个分量有责任度 $w_j^{(i)}$，目标是边缘似然。协方差各向同性且 $w$ 变硬时，二者重合。需要椭圆簇或重叠密度时用 GMM；只想要可解释的分区且实现要稳时，先跑多次 $k$-means。

<br>

> [!NOTE]+ 衔接
>
> EM 的 $z$ 是离散簇。下一章的潜变量改成连续坐标：PCA 找方差最大的方向，因子分析给低秩加对角的生成模型。ELBO 工具仍可用，只是 $Q(z\mid x)$ 从多项换成高斯。

<br>

# 主成分

## 概念：PCA

- **问题**
  - 把 $x\in\mathbb{R}^{d}$ 投影到 $k\ll d$ 维子空间，使保留方差最大，或重构误差（reconstruction error）最小；
  - 两种目标在 $L_2$ 下给出同一特征问题。
- **范围**
  - 从约束优化推出协方差的特征向量，并说明 SVD 算法；
  - 输入中心化数据，输出主成分、投影与重构。
- **章节衔接**
  - 上一章的 $z$ 是离散簇指标；
  - 本章的 $z$ 是连续坐标，目标是保方差或低秩重构；
  - 下一章在非高斯源上把“不相关”加强为“独立”。

先中心化 $x^{(i)}\leftarrow x^{(i)}-\bar x$。单位向量 $u$ 上的投影方差为

$$
\frac1n\sum_{i=1}^{n}\big((x^{(i)})^{\top}u\big)^{2}=u^{\top}\Sigma u,\qquad
\Sigma=\frac1n\sum_{i=1}^{n}x^{(i)}(x^{(i)})^{\top}.
$$

最大化 $u^{\top}\Sigma u$ 约束 $\|u\|_2=1$，拉格朗日 $u^{\top}\Sigma u-\lambda(u^{\top}u-1)$，得 $\Sigma u=\lambda u$。最大方差方向是最大特征值对应的特征向量。前 $k$ 个主成分是最大的 $k$ 个特征值方向，且彼此正交。

> [!NOTE]+ PCA
>
> 找一组正交方向，使投影方差依次最大，等价于 $L_2$ 重构误差最小。来源是 Pearson / Hotelling 的主轴，不是聚类，也不是因果因子。相对直接在原始高维上算距离：先丢掉末尾特征值，可降噪并可视化。相对下一节因子分析：PCA 通常不建各维不同的噪声 $\Psi$，旋转后主成分跟着转，轴没有“真实名字”。

<br>

> [!INFO]+ 从拉格朗日到 $\Sigma u=\lambda u$
>
> 目标 $u^{\top}\Sigma u$，约束 $u^{\top}u=1$。拉格朗日 $L=u^{\top}\Sigma u-\lambda(u^{\top}u-1)$。$\Sigma$ 对称，对 $u$ 求导：
>
> $$
> \nabla_u L=2\Sigma u-2\lambda u=0\implies\Sigma u=\lambda u.
> $$
>
> 代回目标：$u^{\top}\Sigma u=\lambda$。因此在单位球上，投影方差等于对应特征值。最大方差取最大 $\lambda$，次大方差在与第一方向正交的子空间上继续，得到其余特征向量。重构误差
>
> $$
> \sum_i\|x^{(i)}-UU^{\top}x^{(i)}\|_{2}^{2}=n\sum_{j=k+1}^{d}\lambda_j
> $$
>
> 说明丢掉的方差正好是末尾特征值之和，所以“最大方差”与“最小重构”是同一件事。
>
> 感性理解：$\Sigma u=\lambda u$ 表示 $u$ 已是协方差的主轴：$\Sigma$ 沿 $u$ 的作用不产生正交分量，投影方差等于 $\lambda$。前 $k$ 个特征向量张成最大方差子空间，也是 $L_2$ 重构误差最小的 $k$ 维子空间。对 $X$ 做 SVD 得到同一组右奇异向量，避免显式形成 $X^{\top}X$。

<br>

重构误差 $\sum_i\|x^{(i)}-UU^{\top}x^{(i)}\|_{2}^{2}$ 的最小化（$U^{\top}U=I_k$）给出同一组向量。解释方差比例 $\lambda_j/\sum_{\ell}\lambda_{\ell}$ 用于选 $k$：保留 $90\%$ 或看碎石图肘部。$k$ 不是越多越好：末尾成分常是噪声，下游分类有时只用前几个。

数值上对 $X\in\mathbb{R}^{n\times d}$（已中心化）做 SVD：$X=U S V^{\top}$，右奇异向量 $V$ 的前 $k$ 列就是主成分，$\Sigma$ 的特征值是 $s_j^{2}/n$。不要先形成 $X^{\top}X$ 再特征分解，若 $d$ 大且病态，SVD 更稳。

PCA 不是聚类，也不是因果因子。旋转不变：任意正交混合后，主成分跟着转。标准化与否会改变方向：未标准化时大尺度特征主导第一主成分。可视化前两维不等于“数据是二维的”，只是方差最大的二维窗。

概率 PCA / 因子分析把“投影”改回生成模型。经典因子分析

$$
z\sim\mathcal{N}(0,I_k),\qquad
\varepsilon\sim\mathcal{N}(0,\Psi),\qquad
x=\mu+\Lambda z+\varepsilon,
$$

其中 $\Psi$ 对角。边缘 $x\sim\mathcal{N}(\mu,\Lambda\Lambda^{\top}+\Psi)$。当 $n<d$ 或样本协方差奇异时，完整 $\Sigma$ 的 MLE 不存在，低秩加对角的结构仍然可估。$\Psi=\sigma^{2}I$ 时称为概率 PCA：$\Lambda$ 的列张成的空间与普通 PCA 的主空间在 $\sigma\rightarrow 0$ 时重合。参数用 EM：E 步算 $E[z\mid x]$ 与 $E[zz^{\top}\mid x]$，M 步更新 $\Lambda,\Psi$。载荷 $\Lambda$ 仍有正交旋转不定性，不能把某一列直接命名为“真实因子”而不加约束。

> [!NOTE]+ 因子分析
>
> 连续潜变量 $z$ 经载荷 $\Lambda$ 线性生成 $x$，再加对角噪声 $\Psi$。来源是心理测量里的公共因子模型，CS229 把它写成高斯生成模型并用 EM 估计。相对普通 PCA：允许各维观测噪声不同，$n<d$ 时完整协方差的 MLE 不存在，该低秩加对角结构仍可估。相对 ICA：因子是高斯，因此正交旋转不可辨识，不能把某列当作独立物理源。

<br>

## 实现

```python
def pca_fit(X, k):
    """X: (n, d)。返回均值、主成分 (d, k)、解释方差比。"""
    mu = X.mean(axis=0)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k].T
    ev = (S ** 2) / X.shape[0]
    return mu, V, ev[:k] / ev.sum()

def pca_transform(X, mu, V):
    return (X - mu) @ V

def pca_reconstruct(Z, mu, V):
    return Z @ V.T + mu
```

> [!EXAMPLE]+ 例16 人脸与特征脸
>
> $100\times 100$ 灰度脸拉成 $10^{4}$ 维。前几十个主成分（特征脸）主要对应光照与轮廓。用 $k=50$ 重构仍可辨认，用 $k=2$ 只剩低频块。若训练脸都是正面、测试是侧面，主成分子空间外的能量很大，距离比较会失败。训练与部署的姿态分布不一致时，保留方差最大的坐标也不一定对下游任务有用。

<br>

> [!NOTE]+ 衔接
>
> PCA 给出不相关的正交轴。高斯变量不相关即独立，因此高斯源上无法再往下分解。下一章处理非高斯源：目标从“保方差”换成“统计独立”。

<br>

# 独立成分

## 概念：ICA

- **问题**
  - 观察 $x=As$，源（sources）$s$ 的各维统计独立且非高斯；$A$ 未知；
  - 目标是恢复解混矩阵 $W\approx A^{-1}$，使 $y=Wx$ 尽量独立。
- **范围**
  - 写出密度变换与 Bell-Sejnowski 一类目标；
  - 输入混合信号，输出解混矩阵与源估计。
- **章节衔接**
  - 上一章的主成分在高斯下已是独立方向；
  - 本章恢复线性混合的非高斯源，尺度与次序不可辨识；
  - 下一章离开线性混合，用逐步加噪的生成模型处理高维 $x$。

鸡尾酒会：麦克风记录的是说话人的线性叠加。PCA 只能把观察变成不相关；若源是高斯，不相关已等于独立，正交旋转无法辨识，$A$ 在旋转下不可恢复。因此 ICA 要求源非高斯。

> [!NOTE]+ ICA
>
> 假设 $x=As$ 且各源独立、非高斯，估计 $W\approx A^{-1}$。来源是鸡尾酒会与盲源分离，密度变换里的 $|W|$ 保证概率质量守恒。相对 PCA：目标是统计独立而不是最大方差，不要求轴正交。相对因子分析：源必须非高斯，否则正交旋转不可辨识。次序与符号无法恢复，分量不能直接当成“第 $j$ 个说话人”而不加额外约束。

<br>

不可辨识性：

- 置换：源的次序可任意交换；
- 尺度与符号：$s_j$ 乘 $c$ 等价于 $A$ 的第 $j$ 列除以 $c$。

通常约定各 $s_j$ 方差为 $1$，并在预处理中白化（whitening：先 PCA 再缩放到 $\Sigma=I$），把 $A$ 缩到正交矩阵，再在正交群上搜索。

密度变换：若 $x=A s$、$s$ 密度为 $\prod_j p_s(s_j)$，则

$$
p_x(x)=p_s(Wx)\,|W|=\prod_{j=1}^{d}p_s(w_j^{\top}x)\,|W|.
$$

> [!INFO]+ 密度变换里的 $|W|$ 从哪来
>
> $x=As$，$W=A^{-1}$，$s=Wx$。概率质量在可逆变换下守恒：小盒子 $ds$ 映成 $dx$，体积比是 $|\det A|=1/|W|$。因此
>
> $$
> p_x(x)\,|dx|=p_s(s)\,|ds|\implies p_x(x)=p_s(Wx)\,|W|.
> $$
>
> 源独立则 $p_s(s)=\prod_j p_s(s_j)$，取对数后出现 $\sum_j\log p_s(w_j^{\top}x)+\log|W|$。梯度里 $(W^{\top})^{-1}$ 一项正是 $\nabla_W\log|W|$。高斯源的 $p_s$ 只依赖 $\|s\|_{2}$，在正交旋转下不变，于是 $W$ 无法从似然里辨识。
>
> 感性理解：可逆线性变换改变体积元，$p_x$ 必须乘 $|W|$ 才能保持概率质量。缺少 $\log|W|$ 时，似然可通过缩小 $W$ 而被人为抬高。白化把观察协方差变成 $I$，搜索范围缩到正交群，$\log|W|$ 成为常数，只剩源密度的非高斯性用来辨识旋转。

<br>

对数似然

$$
\ell(W)=\sum_{i=1}^{n}\sum_{j=1}^{d}\log p_s(w_j^{\top}x^{(i)})+n\log|W|.
$$

$p_s$ 未知时，选一个超高斯先验，例如 $p_s'(s)/p_s(s)=1-2\sigma(s)$（logistic 源），得到随机梯度

$$
W\leftarrow W+\alpha\big((1-2\sigma(Wx))x^{\top}+(W^{\top})^{-1}\big).
$$

自然梯度右乘 $W^{\top}W$，避免显式求逆并改善几何。实现前必须中心化、白化；实现后按方差排序分量，承认符号可翻转。

ICA 失败模式：源不足 $d$ 个、非线性混合、高斯源、样本太少。脑电伪迹分离是成功案例；把 ICA 分量直接当“真实物理源”需要额外实验约束。

## 实现

```python
def whiten(X):
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    # 右奇异方向上缩放到方差 1
    Z = U[:, :X.shape[1]] * np.sqrt(X.shape[0])
    return Z

def ica_logistic(X, lr=1e-3, num_steps=500, rng=None):
    """X: (n, d) 已白化。"""
    rng = np.random.default_rng(rng)
    n, d = X.shape
    W = rng.normal(size=(d, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    for _ in range(num_steps):
        Y = X @ W.T
        S = 1.0 - 2.0 * sigmoid(Y)
        grad = (S.T @ X) / n + np.linalg.inv(W.T)
        W += lr * grad
    return W, X @ W.T
```

> [!EXAMPLE]+ 例17 两个正弦加噪声
>
> $s_1=\sin t$，$s_2=\mathrm{sign}(\sin(0.3t))$，$x=As$。PCA 轴对准最大方差，仍是混合波形。ICA 后一条接近正弦，一条接近方波，次序与正负号可能对调。若把 $s_2$ 也改成高斯白噪声，恢复只到旋转不定。

<br>

> [!NOTE]+ 对照：PCA、因子分析、ICA
>
> PCA：正交、按方差排序、无噪声模型（或 $\Psi=\sigma^{2}I$ 的概率 PCA）。因子分析：低秩载荷加对角 $\Psi$，允许各维噪声不同，有旋转不定。ICA：源独立且非高斯，不要求正交，次序与尺度不可辨识。三者都在找 $x$ 的线性潜结构，目标函数不同，不能互换解释。

<br>

> [!NOTE]+ 衔接
>
> 线性潜变量到此结束。高维图像或波形很少是 $As+\varepsilon$。下一章用逐步加噪与去噪学习 $p(x)$，潜变量变成整条噪声轨迹，目标仍是 ELBO。

<br>

# 扩散模型

## 概念：扩散

- **问题**
  - 前向过程（forward process）把数据 $x_0$ 逐步加高斯噪声直到近似 $\mathcal{N}(0,I)$；
  - 学习反向去噪（reverse denoising），从而从噪声采样新 $x_0$。
- **范围**
  - 写出离散前向、反向参数化与 ELBO 训练目标；
  - 输入数据分布样本，输出可采样的反向网络。
- **章节衔接**
  - 训练目标仍是 ELBO，与 EM / VAE 同族，只是潜变量变成噪声轨迹；
  - 下一章讨论如何把已预训练的表示用于下游任务，而不再从零估计 $p(x)$。

离散时间前向（DDPM）：

$$
q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I).
$$

> [!NOTE]+ 扩散模型
>
> 先按固定日程把 $x_0$ 加到近高斯噪声，再学反向去噪以采样新样本。来源是非平衡热力学与 DDPM：潜变量是整条噪声轨迹，训练目标仍是 ELBO。相对 GAN：似然有明确下界，训练更稳，不必同时训一对网络。相对 VAE：不把 $x$ 压进低维 $z$，而是在同维噪声上逐步恢复。代价是逐步采样慢，工程上用蒸馏或潜空间扩散补救。

<br>

$\beta_t\in(0,1)$ 是噪声日程。边缘有闭式

$$
q(x_t\mid x_0)=\mathcal{N}(x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I),\qquad
\alpha_t=1-\beta_t,\ 
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

因而可以一步从 $x_0$ 跳到任意 $t$：$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$，$\varepsilon\sim\mathcal{N}(0,I)$。

> [!INFO]+ 为何 $q(x_t\mid x_0)$ 仍是高斯
>
> 单步 $x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,z_t$，$z_t\sim\mathcal{N}(0,I)$ 独立。从 $x_0$ 展开：
>
> $$
> x_t=\sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_1}\,x_0+\text{若干独立高斯之和}.
> $$
>
> 前缀系数是 $\sqrt{\bar\alpha_t}$。独立高斯之和仍是高斯，方差逐项累加：
>
> $$
> \mathrm{Var}(x_t\mid x_0)=(1-\alpha_t)\bar\alpha_{t-1}+(1-\alpha_{t-1})\bar\alpha_{t-2}+\cdots+(1-\alpha_1).
> $$
>
> 这是望远镜和：$\bar\alpha_{s-1}-\bar\alpha_s= (1-\alpha_s)\bar\alpha_{s-1}$，从 $s=1$ 加到 $t$ 得到 $1-\bar\alpha_t$。因此
>
> $$
> x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon.
> $$
>
> 训练时不必逐步加噪，对均匀抽样的 $t$ 一步到位。反向网络预测 $\varepsilon$，等价于在已知 $x_t$ 时估计 $x_0$ 的后验均值。
>
> 感性理解：线性高斯转移的复合仍是高斯。$\sqrt{\bar\alpha_t}$ 是 $x_0$ 的剩余系数，$1-\bar\alpha_t$ 是累积噪声方差。$t$ 足够大时边缘接近 $\mathcal{N}(0,I)$。训练对均匀抽样的 $t$ 一步得到 $x_t$，网络估计该步的噪声 $\varepsilon$，等价于估计 $q(x_0\mid x_t)$ 的后验均值，而不必逐步模拟前向过程。

<br>

反向 $p_\theta(x_{t-1}\mid x_t)$ 参数化为高斯，均值由网络给出。一种等价且稳定的训练是预测噪声 $\varepsilon_\theta(x_t,t)$，损失

$$
L=\mathbb{E}_{x_0,\varepsilon,t}\big[\|\varepsilon-\varepsilon_\theta(x_t,t)\|_{2}^{2}\big].
$$

该式来自 ELBO 各项在高斯假设下的加权简化；实践常对 $t$ 均匀采样并忽略复杂权重。采样：从 $x_T\sim\mathcal{N}(0,I)$ 出发，按

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\Big)+\sigma_t z
$$

逐步走到 $t=0$。连续时间观点把前向写成 SDE，反向 SDE 或概率流 ODE 给出同一边缘；实现时仍是离散积分。

扩散比 GAN 训练稳，似然有明确下界；代价是逐步采样慢。蒸馏、一致性模型与潜空间扩散是工程补救，不改变“加噪-去噪”的核心。

## 实现

```python
def q_sample(x0, t, alphas_bar, rng):
    """t: (n,) 的整数时间。"""
    eps = rng.normal(size=x0.shape)
    a = alphas_bar[t].reshape(-1, *([1] * (x0.ndim - 1)))
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * eps, eps

def ddpm_loss(model, x0, alphas_bar, rng):
    n = x0.shape[0]
    t = rng.integers(0, len(alphas_bar), size=n)
    xt, eps = q_sample(x0, t, alphas_bar, rng)
    pred = model(xt, t)
    return float(np.mean((pred - eps) ** 2))
```

> [!EXAMPLE]+ 例18 一维双峰
>
> $x_0$ 来自两个相距很远的高斯。$t$ 小时 $\varepsilon_\theta$ 只需轻微去噪；$t$ 大时 $x_t$ 已分不清来自哪一峰，网络必须学会把质量拆回两峰。若日程 $\beta_t$ 过快，中间 $t$ 的信噪比塌缩，训练信号变差。噪声日程 $\beta_t$ 直接决定中间时刻的信噪比，需要与学习率一并调节。

<br>

> [!NOTE]+ 衔接
>
> 扩散从噪声生成新的 $x$。不少下游任务只需要一个已在大规模数据上训练过的表示 $f$。下一章讨论冻结、微调与 LoRA；表示 $f$ 本身的训练放在再下一章。

<br>

# 基础模型

## 概念：基础模型

- **问题**
  - 在大规模数据上预训练（pretraining）一个通用表示或生成模型，再适配下游；
  - 适配方式包括线性探针（linear probe）、全量微调（fine-tuning）与低秩适配（LoRA）。
- **范围**
  - 分清冻结表示与更新参数的代价；
  - 输入预训练权重与下游标注，输出适配后的预测头或增量权重。
- **章节衔接**
  - 上一章说明如何采样新 $x$；本章说明如何复用已有 $f(x)$；
  - 探针、微调与 LoRA 改变的是如何使用已有 $f$，并不引入新的似然形式；
  - 下一章写 $f$ 本身如何用对比与检索学出来。

线性探针：冻结主干 $f$，只训 $W$ 使 $Wf(x)$ 拟合 $y$。用于诊断表示里已经线性可分的信息。全量微调更新全部参数，表达力强，小数据上易毁预训练特征（灾难遗忘）。折中：只解冻最后几块，或使用 LoRA。

LoRA：对权重 $W\in\mathbb{R}^{m\times n}$ 学习低秩增量

$$
W'=W+\frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{m\times r},A\in\mathbb{R}^{r\times n}.
$$

$r\ll\min(m,n)$ 时增量参数约为 $r(m+n)$。推理可把 $BA$ 合并进 $W$，无额外延迟。$\alpha/r$ 控制步幅。LoRA 不是正则的替代：下游仍需验证集与早停。

> [!EXAMPLE]+ 例19 何时探针就够
>
> 图像主干在 ImageNet 上预训练，下游是十类物体，线性探针准确率已接近全量微调，说明类别信息已近线性可分，优先探针以节省算力。若下游是细粒度病斑且探针远差于微调，才值得解冻或 LoRA。项目里通常先做探针，确认表示不足后再微调或加 LoRA。

<br>

> [!NOTE]+ 衔接
>
> 探针与 LoRA 默认 $f$ 已经有用。下一章补上 $f$ 的来源：监督预训练、对比学习、语义检索，以及把检索接到生成上的 RAG。

<br>

# 表示学习

## 概念：表示

- **问题**
  - 学习 $f(x)$ 使下游任务变容易：同类靠近、检索可召回、生成可条件化；
  - 监督预训练、对比学习（contrastive learning）、语义检索（semantic retrieval）与 RAG 是四条常用路。
- **范围**
  - 写出对比损失与检索-生成接口；
  - 输入原始对象，输出向量与最近邻或增强后的生成。
- **章节衔接**
  - 上一章冻结或低秩更新 $f$；本章回答 $f$ 从哪来；
  - 对比学习给出无标签的几何；检索把几何变成可检查的邻居；
  - RAG 把邻居接到下一章的语言模型，而不是用参数记住全部事实。

监督预训练：在源任务上最小化标准损失，取倒数第二层当 $f$。迁移是否成功取决于源与目标的共享结构，不是参数量本身。

对比学习：正对 $(x,x^{+})$（同一图像的两种增强、同一文档的两句），负对 $(x,x^{-})$。InfoNCE

$$
\mathcal{L}=-\log\frac{\exp(\mathrm{sim}(f(x),f(x^{+}))/\tau)}{\exp(\mathrm{sim}(f(x),f(x^{+}))/\tau)+\sum_{x^{-}}\exp(\mathrm{sim}(f(x),f(x^{-}))/\tau)}.
$$

$\tau$ 是温度：过小则只关注最难负例，训练不稳；过大则梯度平坦。批次要够大才能提供有信息的负例，或使用动量队列。对比学习不需要类标，但正对的两种增强应保持语义，否则梯度会把不相关的视图拉近。

> [!NOTE]+ InfoNCE
>
> 把“认出正对、推开负对”写成多项交叉熵：分母是正例与若干负例的相似度。来源是对比估计（NCE）与 SimCLR / CLIP 一类表示学习，不是生成模型的似然。相对分类交叉熵：不需要类标，正对由增强或配对文本定义。相对重构损失：优化的是几何上的可分性，而不是像素误差。温度 $\tau$ 与负例数量决定梯度集中在多难的负对上。

<br>

语义检索：把查询与文档编码到同一空间，按内积或余弦取 top-$k$。评估用 Recall@$k$、nDCG，而不是生成 BLEU。索引（IVF、HNSW）是近似最近邻，召回率与延迟必须同时报。

RAG：先检索再生成。生成条件于查询与检索片段，减少幻觉并允许更新知识而不重训全部权重。若检索错误，生成仍可能写出通顺但不正确的句子。检索命中与生成忠实度需要分开评估。

> [!NOTE]+ RAG
>
> 生成时条件于检索到的片段，而不是把全部事实存进参数。来源是开放域问答：参数记忆难以及时更新，也难以审计出处。相对纯参数模型：知识可换索引、可检查邻居。相对只做检索：仍要一个条件语言模型把片段写成答案。检索错误会原样进入通顺的句子，因此命中率与忠实度必须分开报。

<br>

## 实现

```python
def infonce(z_i, z_j, tau=0.07):
    """z_i, z_j: (n, d) 的 L2 归一化正对。"""
    logits = (z_i @ z_j.T) / tau
    labels = np.arange(z_i.shape[0])
    # 对称两项：i->j 与 j->i
    def ce(logp, y):
        z = logp - logp.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.log(p[labels, y] + 1e-12).mean())
    return 0.5 * (ce(logits, labels) + ce(logits.T, labels))

def retrieve(query_vec, doc_vecs, k=5):
    scores = doc_vecs @ query_vec
    idx = np.argpartition(-scores, kth=min(k, scores.size - 1))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]
```

> [!EXAMPLE]+ 例20 检索与生成必须拆开评估
>
> 问“某药的禁忌”。检索返回过期说明书，模型忠实复述过期内容。终端答案错误，但生成模块没有胡编，错误在索引。下一实验是更新语料或改排序，不是加大解码温度。若检索正确而答案仍编造剂量，才去改提示或微调忠实度。

<br>

> [!NOTE]+ 对照：生成新样本与检索旧样本
>
> 扩散与语言模型从噪声或前缀采样新对象；检索在已有语料中找邻居。RAG 先取证据，再条件生成。检索命中率与生成忠实度应分开报告。只看终端流畅度，无法判断下一步该改索引还是改解码。

<br>

> [!NOTE]+ 衔接
>
> 表示给出向量空间。下一章在同一空间上对 token 序列做自回归：损失是 $\sum_t-\log p(x_t\mid x_{<t})$，核心模块是带因果掩码的注意力。

<br>

# 大语言模型

## 概念：LLM

- **问题**
  - 在 token 序列上做自回归（autoregressive）语言建模，再用同一模型做提示、检索与微调；
  - 架构核心是 Transformer 与其注意力（attention）变体。
- **范围**
  - 写出 next-token 损失、缩放点积注意力与常见适配；
  - 输入文本，输出 token 分布或解码序列。
- **章节衔接**
  - 上一章的向量可用于检索；本章在同一向量空间上做自回归生成；
  - 损失是 next-token 负对数似然，架构是带掩码的注意力；
  - 下一章在这条损失之上加思维链与可验证奖励，不改注意力公式。

## 分词（tokenization）

文本先切成词表中的子词（BPE、Unigram）。词表大小 $V$ 决定 softmax 最后一维。切分必须与训练时完全一致：多一个空格、不同 Unicode 规范化，都会改变 token 序列，使困惑度不可比。BOS、EOS、PAD 会进入模型，与普通 token 一样参与注意力与损失。

## 自回归（autoregressive）

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

训练损失是平均负对数似然

$$
J=-\frac1N\sum_{\mathrm{seq}}\sum_{t}\log p_{\theta}(x_t\mid x_{<t}).
$$

$N$ 是 token 总数时，不同长度文档才可比较。教师强制：训练时条件于真实前缀，不是模型自己的采样；推理时条件于已生成前缀，分布偏移称为暴露偏差。解码：贪心快但不多样；温度采样改变锐度；nucleus（top-$p$）截掉长尾。评估：困惑度 $\exp(J)$；任务指标（准确率、F1）必须在下游协议上另测。

## Transformer

对序列 $X\in\mathbb{R}^{T\times d}$，

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}+M\Big)V.
$$

$M$ 在因果语言模型里对 $t'<t$ 为 $0$、对未来为 $-\infty$。多头把 $d$ 切成 $h$ 组再拼接。位置信息由正弦位置编码或 RoPE 注入，否则自注意力置换等变，无法区分“A 打 B”与“B 打 A”。FFN 是逐位置双层 MLP；Pre-Norm + 残差使深层可训。复杂度 $O(T^{2}d)$ 来自 $QK^{\top}$，这是长上下文的主瓶颈。

> [!NOTE]+ 自注意力
>
> 同一序列里每个位置用 $Q$ 去对所有位置的 $K$ 打分，再对 $V$ 做加权平均。来源是把编码器-解码器注意力改成序列内部交互，从而去掉 RNN 的逐步依赖。相对循环：训练可并行，长程依赖不经过逐步隐状态。相对卷积：感受野是全序列，而不是固定核宽。必须另加位置编码，否则对置换不变；因果掩码防止读到未来 token。除以 $\sqrt{d_k}$ 是为了避免点积方差随维度增大而导致 softmax 饱和。

<br>

变体：多查询 / 分组查询注意力减少 KV 缓存；滑动窗口降低 $T^{2}$；交叉注意力用于编码器-解码器。MoE：FFN 换成若干专家，路由器只激活少数专家，参数量上升而每 token FLOP 近似不变；训练须防止专家崩塌（所有 token 进同一专家）。

## 使用方式

- **零样本提示**：只给指令，不给例子。依赖预训练里见过的任务格式。
- **上下文学习**：提示里放若干 $(x,y)$ 对。这是推理期条件，不是梯度更新；例子顺序与标签空间都会改变准确率。
- **SFT**：在指令-回答对上继续最小化 next-token 损失。数据质量比数量更敏感：错误示范会被模仿。
- **RAG / 提示优化**：改检索与提示模板，不改权重。必须版本化提示，否则实验不可复现。

安全：预训练会复述有害或隐私内容。部署需要拒绝策略、输出过滤与评估红队，不能只报困惑度。

## 实现

```python
def scaled_dot_product_attention(Q, K, V, causal=True):
    """Q,K,V: (n, h, T, d_k)。"""
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if causal:
        T = scores.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    z = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(z)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V

def next_token_nll(logits, targets):
    """logits: (n, T, V), targets: (n, T)。"""
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    n, T = targets.shape
    return float(-logp[np.arange(n)[:, None], np.arange(T)[None, :], targets].mean())
```

> [!EXAMPLE]+ 例21 因果掩码漏了
>
> 训练时若不对未来位置置 $-\infty$，自注意力会读到 $x_{t+1}$ 再预测 $x_t$，训练损失会异常偏低；推理阶段没有未来 token，性能会明显下降。可用 $T=3$ 的小例子检查位置 $0$ 的权重是否只落在位置 $0$。

<br>

> [!NOTE]+ 衔接
>
> 预训练与 SFT 都在最小化 next-token 损失。下一章改变的是推理期计算和奖励，不是注意力公式：思维链增加中间 token，RLVR 用可验证回报更新 $\pi$。

<br>

# LLM 推理

## 概念：推理

- **问题**
  - 在生成过程中显式写出中间步骤，并用可验证奖励（verifiable rewards）做强化学习；
  - 思维链（chain of thought）改变的是推理期计算，不一定改变预训练损失。
- **范围**
  - 区分提示出的链与 RLVR 训出的链；
  - 输入题目，输出带步骤的答案与可检查的正确性。
- **章节衔接**
  - 上一章的解码只最大化 $p(x_t\mid x_{<t})$；
  - 本章把中间 token 当作额外计算，并用可检查的奖励更新策略；
  - 下一章给出一般 MDP 与贝尔曼方程，策略梯度是其中一条求解器。

思维链：提示逐步书写中间步骤后再给答案。对算术与符号操作，中间 token 增加可用的计算步数。自洽性对多条采样链的最终答案投票。延迟与费用随链长上升；链更长并不保证正确，错误中间步骤会被后续条件化并延续。

RLVR（reinforcement learning with verifiable rewards）：环境给可程序化检查的奖励，例如单位测试通过、数学表达式等于标准答案。策略是语言模型，动作是 token。与 RLHF 的区别：奖励不是学出来的偏好模型，减少了奖励黑客的一种来源，但模型仍可能钻验证器的空子（硬编码通过用例）。长链推理把“思考 token”纳入动作空间，优化目标是期望可验证回报，不是模仿教师的中间步骤。

> [!WARNING]+ 不可验证任务不要假装可验证
>
> 开放写作没有自动判分器。强行用另一个 LLM 当奖励，会把该模型的偏好写进策略。CS229 的 RLVR 讨论限定在能写检查器的任务；其余应回到 SFT 或人工评估。

<br>

> [!NOTE]+ 衔接
>
> RLVR 已经在用策略梯度，但环境被简化成“答案对错”。下一章给出一般 MDP：状态、动作、转移、折扣。值迭代在已知 $P$ 时求最优值；语言模型只是策略的一种参数化。

<br>

# 强化学习

## 概念：RL

- **问题**
  - 智能体（agent）在环境中选动作，得到奖励与下一状态；目标是最大化期望累积回报（return）；
  - 与监督学习的关键差别：没有现成的 $(s,a^{\ast})$ 标签，且动作影响未来数据。
- **范围**
  - 定义 MDP，写出贝尔曼方程、值迭代与策略迭代；
  - 输入转移与奖励，或与环境交互的轨迹，输出策略。
- **章节衔接**
  - 监督学习有固定 $(x,y)$；本章的数据分布依赖于正在执行的 $\pi$；
  - 值迭代在已知 $P$ 时求 $V^{\ast}$；连续状态必须逼近；
  - 下一章在线性动态、二次代价下给出闭式反馈。

## MDP

马尔可夫决策过程 $(S,A,P,R,\gamma)$：

- $S$：状态集；$A$：动作集；
- $P(s'\mid s,a)$：转移；
- $R(s,a)$ 或 $R(s,a,s')$：奖励；
- $\gamma\in[0,1)$：折扣。

策略 $\pi(a\mid s)$。回报 $G_t=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k}$。状态值与动作值：

$$
V^{\pi}(s)=E_{\pi}[G_t\mid s_t=s],\qquad
Q^{\pi}(s,a)=E_{\pi}[G_t\mid s_t=s,a_t=a].
$$

贝尔曼期望方程：

$$
V^{\pi}(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^{\pi}(s')\big].
$$

最优 $V^{\ast}(s)=\max_{\pi}V^{\pi}(s)$ 满足贝尔曼最优方程：

$$
V^{\ast}(s)=\max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^{\ast}(s')\big].
$$

马尔可夫性：给定当前 $s$，未来与更早历史条件独立。若真实决策依赖未写入 $s$ 的记忆，模型设错，再强的算法也在优化错误对象。

> [!NOTE]+ MDP
>
> 用 $(S,A,P,R,\gamma)$ 描述序贯决策：当前状态已包含对未来有用的历史。来源是动态规划与最优控制，不是把监督学习的 $(x,y)$ 换成 $(s,a)$。相对监督：没有现成的 $a^{\ast}$，且 $\pi$ 改变后数据分布跟着变。相对多臂老虎机：状态会转移，当前动作影响以后能到达的状态。$\gamma$ 把无限时域的回报变成有限期望；$\gamma=1$ 且无吸收态时目标可能发散。

<br>

## 值迭代与策略迭代（value / policy iteration）

值迭代对所有 $s$ 同步更新

$$
V(s)\leftarrow\max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V(s')\big],
$$

直到 $\|V_{\mathrm{new}}-V\|_{\infty}$ 小于阈值。收敛到 $V^{\ast}$ 后，贪心

$$
\pi(s)=\arg\max_a\sum_{s'}P(s'\mid s,a)\big[R+\gamma V(s')\big]
$$

即最优策略（并列最优时可任选）。压缩映射：贝尔曼最优算子是 $\gamma$-收缩，因此迭代线性收敛，速率由 $\gamma$ 决定，$\gamma$ 越接近 $1$ 越慢。

策略迭代交替：

1. 策略评估：解线性方程组 $V=T^{\pi}V$，或迭代评估到近似；
2. 策略改进：对 $V^{\pi}$ 贪心得到 $\pi'$。

有限 MDP 上策略单调改进，有限步到达最优。每轮评估比值迭代更“彻底”，轮数通常更少，但单轮更贵。

> [!NOTE]+ 值迭代与策略迭代
>
> 值迭代反复用贝尔曼最优算子更新 $V$，再贪心出 $\pi$。策略迭代先把当前 $\pi$ 评估到底，再改进。二者都来自动态规划，要求已知 $P,R$ 且 $S,A$ 有限。相对监督学习的一次梯度：这里是压缩映射迭代，初值不影响最优解，只影响步数。相对后面的策略梯度：必须能对 $a$ 做 $\arg\max$，并且用得起模型或表格。$\gamma$ 接近 $1$ 时收缩变慢。

<br>

## 学模型

$P,R$ 未知时，用轨迹计数估计 $\hat P(s'\mid s,a)=N(s,a,s')/N(s,a)$，$\hat R$ 为平均奖励。再在 $\hat P,\hat R$ 上跑值迭代。这是基于模型的 RL。未访问的 $(s,a)$ 计数为 $0$，估计不可靠；需要探索（$\varepsilon$-贪心、RMax、后验采样）。小状态空间可行；连续或巨大 $S$ 时模型本身要参数化。

## 连续状态

离散化：把 $\mathbb{R}^{d}$ 切成格子，当表格 MDP。格子粗则量化误差大；格子细则 $|S|$ 指数爆炸（维数灾难）。只适合 $d$ 很小且动态光滑。

值函数逼近：用 $V(s;w)$ 或 $Q(s,a;w)$ 代替表格。线性 $V(s)=w^{\top}\phi(s)$ 时，可对贝尔曼残差或 TD 目标做随机梯度。非线性（神经网络）即 DQN 一类：目标 $r+\gamma\max_{a'}Q(s',a';w^{-})$，定期复制目标网 $w^{-}$，并用回放缓冲打破轨迹相关。逼近使“收敛到 $V^{\ast}$”的保证变弱；必须监控评估回报，不能只看 TD 损失。

> [!INFO]+ 贝尔曼方程在算什么
>
> $V^{\pi}(s)$ 是“从 $s$ 出发按 $\pi$ 走到无穷的折扣奖励期望”。把第一步的期望奖励写成 $\sum_a\pi(a\mid s)\sum_{s'}P R$，其余从 $s'$ 出发的期望恰是 $V^{\pi}(s')$，再乘 $\gamma$。这不是新假设，只是全期望定律。值迭代把等号换成赋值，并用 $\max_a$ 代替 $\sum_a\pi$，相当于同时做评估与改进。初值 $V=0$ 即可；最优解与初值无关，只影响迭代次数。
>
> 写成一步：
>
> $$
> V^{\pi}(s)=E_{a\sim\pi,\,s'\sim P}\big[R(s,a,s')+\gamma V^{\pi}(s')\big].
> $$
>
> 最优方程把期望里的 $\pi$ 换成 $\max_a$。收缩映射保证 $\gamma<1$ 时迭代收敛到唯一不动点。
>
> 感性理解：$V^{\pi}(s)$ 拆成即时奖励与从后继状态继续按 $\pi$ 行走的价值。$\gamma$ 接近 $1$ 时远期奖励与即时奖励几乎同等计入；接近 $0$ 时只需看一步。值迭代反复用贝尔曼最优算子更新表格；该算子是模为 $\gamma$ 的压缩映射，与初值无关地收敛到唯一不动点。连续状态没有有限表格，只能逼近，收敛到 $V^{\ast}$ 的保证随之变弱。

<br>

## 实现

```python
def value_iteration(P, R, gamma=0.95, tol=1e-8, max_iter=10000):
    """P: (S, A, S), R: (S, A, S) 或 (S, A)。"""
    S, A, _ = P.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = np.zeros((S, A))
        for a in range(A):
            if R.ndim == 3:
                reward = (P[:, a, :] * R[:, a, :]).sum(axis=1)
            else:
                reward = R[:, a]
            Q[:, a] = reward + gamma * P[:, a, :] @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    pi = Q.argmax(axis=1)
    return V, pi, Q

def policy_evaluation(P, R, pi, gamma=0.95, tol=1e-8):
    S, A, _ = P.shape
    V = np.zeros(S)
    while True:
        V_new = np.zeros(S)
        for s in range(S):
            a = pi[s]
            reward = (P[s, a] * R[s, a]).sum() if R.ndim == 3 else R[s, a]
            V_new[s] = reward + gamma * P[s, a] @ V
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
```

> [!EXAMPLE]+ 例22 网格世界
>
> $4\times 4$ 格子，目标格奖励 $+1$ 并终止，其余 $-0.04$，四方向动作，撞墙停留。$\gamma=0.99$ 时最优策略沿最短路走，因为活着的每一步都扣分。若把逐步奖励改为 $0$，多种绕路与最优路的 $V$ 只差在数值精度，策略可能绕远。奖励塑形改变的是最优策略，不是“同一任务的另一写法”。

<br>

> [!NOTE]+ 对照：监督与 RL
>
> 监督：数据分布固定，损失对每个 $(x,y)$ 可单独计算。RL：动作改变下一状态，也改变以后能看见的数据。值函数把“以后的奖励”收成一个数，贝尔曼方程是全期望定律，不是新的建模假设。没有 $P$ 时，要么先估模型，要么直接估 $Q$ 或 $\nabla J$。

<br>

> [!NOTE]+ 衔接
>
> 值迭代对一般有限 $S$ 反复扫描。若动态是线性的、代价是二次的，动态规划可以一次回传到底，并得到线性反馈。下一章写这个闭式解，以及如何对非线性 $f$ 做局部 LQR。

<br>

# LQR

## 概念：LQR

- **问题**
  - 有限时域（finite horizon）、线性动态、二次代价的控制；
  - 最优策略是状态的线性反馈（linear feedback），可用动态规划反推。
- **范围**
  - 写出离散 LQR 的 Riccati 回传，并说明线性化与 LQG；
  - 输入 $(A,B,Q,R)$，输出增益序列 $K_t$。
- **章节衔接**
  - 上一章的值迭代对一般有限 $S$ 迭代；本章对线性-二次一次回传到底；
  - 非线性时在名义轨迹上线性化，即 DDP；
  - 下一章不再需要可逆的 $P$，直接对 $\pi_{\theta}$ 求 $J$ 的梯度。

有限时域 MDP：时间 $t=0,\ldots,T$，代价可加。线性动态与二次代价

$$
x_{t+1}=A x_t+B u_t,\qquad
J=x_T^{\top}Q_T x_T+\sum_{t=0}^{T-1}\big(x_t^{\top}Q x_t+u_t^{\top}R u_t\big).
$$

$Q,Q_T$ 半正定，$R$ 正定（控制要花代价）。值函数保持二次 $V_t(x)=x^{\top}P_t x$（可加线性项若有仿射动态）。从 $P_T=Q_T$ 回传

$$
\begin{aligned}
K_t&=(R+B^{\top}P_{t+1}B)^{-1}B^{\top}P_{t+1}A,\\
P_t&=Q+A^{\top}P_{t+1}A-A^{\top}P_{t+1}B K_t,\\
u_t^{\ast}&=-K_t x_t.
\end{aligned}
$$

无限时域定常 $(A,B)$ 时 $P_t$ 收敛到代数 Riccati 方程的解，增益恒定。可控性不足则无法任意配置闭环极点；$R$ 过小则控制猛、对模型误差敏感。

> [!INFO]+ Riccati 回传在最小化什么
>
> 假设从 $t+1$ 起最优代价已是二次，$V_{t+1}(x)=x^{\top}P_{t+1}x$。当前步
>
> $$
> V_t(x)=\min_u\Big[x^{\top}Qx+u^{\top}Ru+(Ax+Bu)^{\top}P_{t+1}(Ax+Bu)\Big].
> $$
>
> 对 $u$ 求导：$2Ru+2B^{\top}P_{t+1}(Ax+Bu)=0$，解出
>
> $$
> u^{\ast}=-(R+B^{\top}P_{t+1}B)^{-1}B^{\top}P_{t+1}A\,x=-K_t x.
> $$
>
> 代回二次型，收集 $x^{\top}(\cdot)x$ 的系数，得到 $P_t=Q+A^{\top}P_{t+1}A-A^{\top}P_{t+1}BK_t$。终点 $P_T=Q_T$ 启动回传。线性动态加二次代价保证值函数永远是二次，归纳不断裂。
>
> 感性理解：二次代价在线性动态下对值函数封闭：若 $V_{t+1}$ 是二次，则 $V_t$ 仍是二次。对当前 $u$ 求最小得到线性反馈 $u=-K_t x$。$R$ 增大则控制更贵，$K_t$ 减小；$P_{t+1}$ 增大则未来状态代价更高，当前步更愿意施加控制。$P_t$ 把 $t$ 之后的全部二次代价折进当前状态。

<br>

非线性 $x_{t+1}=f(x_t,u_t)$：在名义轨迹 $(\bar x_t,\bar u_t)$ 上线性化 $A_t=D_x f$，$B_t=D_u f$，再解时变 LQR，得到修正 $\delta u_t=-K_t\delta x_t$，即迭代 LQR / DDP 的核心一步。需要多次前向滚出与回传，直到轨迹稳定。局部二次模型失效时（接触、不连续），该方法发散，应换接触感知或采样型 MPC。

LQG：线性动态、二次代价、高斯过程噪声与测量噪声。最优仍是二次值函数；确定性等价：先用卡尔曼滤波估计 $\hat x$，再套 LQR 的 $u=-K\hat x$。分离原理在线性高斯下成立，非线性下一般不成立，不能把“先滤后控”当成普遍定理。

> [!NOTE]+ LQR
>
> 线性动态加二次代价时，最优反馈是 $u=-Kx$，增益由 Riccati 回传得到。来源是最优控制，不是把值迭代的表格换成矩阵记号。相对一般 MDP：值函数保持二次，不必扫状态空间。相对策略梯度：必须有 $A,B$ 或可线性化的 $f$；$R$ 过小则控制猛、对模型误差敏感。迭代 LQR / DDP 把同一回传用在名义轨迹的局部线性化上，接触或不连续时局部二次失效。

<br>

## 实现

```python
def discrete_lqr(A, B, Q, R, QT, T):
    P = QT
    Ks = []
    for _ in range(T):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        Ks.append(K)
    return list(reversed(Ks)), P

def rollout_lqr(x0, A, B, Ks):
    xs = [x0]
    us = []
    x = x0
    for K in Ks:
        u = -K @ x
        x = A @ x + B @ u
        us.append(u)
        xs.append(x)
    return np.stack(xs), np.stack(us)
```

> [!EXAMPLE]+ 例23 一维小车
>
> $\dot v=u$，$x=(p,v)$。离散一步：$p\leftarrow p+\Delta v+\Delta^{2}u/2$，$v\leftarrow v+\Delta u$。$Q$ 惩罚位置，$R$ 惩罚推力。$R$ 很大时 $K$ 小，车慢慢停；$R$ 很小时起步猛、超调。把 $Q$ 的速度项加大，等于要求少晃，闭环更阻尼。$Q,R$ 带物理单位，取值应与状态量纲一致。

<br>

> [!NOTE]+ 衔接
>
> LQR / DDP 要模型或线性化。许多环境只有采样接口。下一章对 $\pi_{\theta}$ 直接上升 $J(\theta)$，转移 $P$ 不出现在梯度里。方差过大时，估计器形式上无偏，更新仍会被噪声淹没。

<br>

# 策略梯度

## 概念：策略梯度

- **问题**
  - 直接参数化策略 $\pi_{\theta}(a\mid s)$，沿 $J(\theta)=E_{\pi_{\theta}}[G_0]$ 做策略梯度（policy gradient）；
  - 不要求可逆动态，也不要求 $Q$ 表格。
- **范围**
  - 写出 REINFORCE 与 PPO 的目标；
  - 输入轨迹，输出 $\theta$ 的更新。
- **章节衔接**
  - 值迭代与 LQR 都要用模型或线性化；本章只采样轨迹；
  - 对数导数给出无偏梯度，基线与 PPO 裁剪管方差；
  - LLM 推理中的 RLVR 使用同一估计器，动作是 token。

轨迹 $\tau=(s_0,a_0,\ldots)$，$p_{\theta}(\tau)=p(s_0)\prod_t\pi_{\theta}(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$。$J(\theta)=E_{\tau\sim p_{\theta}}[R(\tau)]$。对数导数技巧：

$$
\nabla_{\theta}J(\theta)
=
E_{\tau}\Big[R(\tau)\sum_{t}\nabla_{\theta}\log\pi_{\theta}(a_t\mid s_t)\Big].
$$

$P$ 不出现在梯度里，因此是无模型的。REINFORCE：采样若干条轨迹，用 $\sum_t\nabla\log\pi\cdot G_t$ 的平均当梯度。$G_t$ 从时刻 $t$ 算起，避免把未来奖励算到过去动作上（因果）。

方差巨大：乘上一条轨迹的总回报，噪声可淹没信号。减基线 $b(s_t)$（例如值函数 $V_w(s_t)$）不改变期望：

$$
\nabla J=E\Big[\sum_t\nabla\log\pi_{\theta}(a_t\mid s_t)\big(G_t-b(s_t)\big)\Big].
$$

优势 $A(s,a)=Q(s,a)-V(s)$ 是更紧的乘子。Actor-Critic 同时学 $\pi_{\theta}$ 与 $V_w$。

## PPO

朴素大步更新会让 $\pi$ 离开采样策略太远，重要性采样（importance sampling）比爆炸。PPO 用裁剪代理目标

$$
L^{\mathrm{CLIP}}(\theta)
=
E_t\Big[\min\big(
r_t(\theta)A_t,\ 
\mathrm{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)A_t
\big)\Big],
$$

其中 $r_t(\theta)=\pi_{\theta}(a_t\mid s_t)/\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$。$A_t>0$ 时不允许 $r$ 超过 $1+\varepsilon$；$A_t<0$ 时不允许 $r$ 低于 $1-\varepsilon$。再加熵奖励鼓励探索，加值函数损失训 critic。PPO 不是新的最优方程，是工程上稳定的策略梯度。超参数 $\varepsilon$、更新轮数、batch 长度彼此耦合，必须在验证环境上报平均回报，不能只报代理损失下降。

> [!NOTE]+ REINFORCE 与 PPO
>
> REINFORCE 用 $\nabla\log\pi\cdot G_t$ 的样本平均当 $\nabla J$，来自对数导数恒等式，不需要 $P$。相对值迭代：动作可以是连续的或离散的 token，不必 $\arg\max_a Q$。代价是方差大。PPO 在重要性比率上加裁剪，阻止一次更新让 $\pi$ 离开采样策略太远。相对 TRPO：用一阶裁剪代替二阶约束，实现更简单，不是新的最优方程。$\varepsilon$ 与更新轮数必须在验证环境上看回报。

<br>

> [!INFO]+ 对数导数为何合法
>
> $\nabla_{\theta}p_{\theta}(\tau)=p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)$。$p_{\theta}(\tau)$ 中只有 $\pi_{\theta}$ 含 $\theta$，故 $\nabla\log p_{\theta}=\sum_t\nabla\log\pi_{\theta}$。两边乘 $R(\tau)$ 再对 $\tau$ 积分，得到 $\nabla J$。该恒等式对任何可微密度成立，与奖励是否稀疏无关；稀疏只影响方差。实现时对 $\log\pi$ 做数值裁剪，避免 $\log 0$。
>
> 轨迹密度 $p_{\theta}(\tau)=p(s_0)\prod_t\pi_{\theta}(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$。$\log$ 之后 $P$ 与 $p(s_0)$ 对 $\theta$ 的导数为零，只留下 $\sum_t\nabla\log\pi_{\theta}$。减去基线 $b(s)$ 不改变期望（因为 $E[\nabla\log\pi]=0$），只减小方差。
>
> 感性理解：$\nabla\log\pi$ 给出提高当前动作概率的方向，轨迹回报决定该方向的符号与幅度。转移核 $P$ 不含 $\theta$，对数导数后从梯度中消失，因此无需环境模型。无偏代价是方差：同一 $\pi$ 下不同轨迹的回报可以相差一个数量级。减去只依赖状态的基线不改变期望，只减小方差。

<br>

## 实现

```python
def reinforce_update(log_probs, returns, lr, baseline=0.0):
    """log_probs, returns: 一条轨迹上各步的 log π 与 G_t。"""
    adv = returns - baseline
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    loss = -np.sum(log_probs * adv)
    # 真实项目里由自动微分回传到 theta；此处只返回标量目标
    return float(loss)

def discount_returns(rewards, gamma):
    G = 0.0
    out = np.zeros_like(rewards, dtype=float)
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        out[t] = G
    return out

def ppo_clip_objective(ratio, adv, eps=0.2):
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
    return float(np.mean(np.minimum(unclipped, clipped)))
```

> [!EXAMPLE]+ 例24 基线把方差砍下来
>
> 两臂老虎机，期望奖励 $1$ 与 $1.1$，每条轨迹长度 $1$。无基线时乘子是 $1$ 或 $1.1$，梯度符号对，但幅度抖。减掉平均奖励 $1.05$ 后，差臂得到负优势，好臂得到正优势，同一样本数下分得更开。基线选错（例如用未来才知道的全局常数之外的、与动作相关的量）会引入偏差。值函数基线只依赖 $s$，不依赖 $a$，因此保持无偏。

<br>

> [!NOTE]+ 对照：值方法与策略梯度
>
> 值迭代 / Q 学习：先得到 $V$ 或 $Q$，再贪心出 $\pi$。要求动作空间可 $\arg\max$，模型或 $Q$ 逼近要稳。策略梯度：$\pi_{\theta}$ 直接输出分布，适合大的或连续的 $A$，也适合语言模型。PPO 用裁剪防止一次更新走太远。LLM 推理章的 RLVR 属于后者，不是值迭代。

<br>

> [!NOTE]+ 衔接
>
> 算法章优化的是平均损失或平均回报。部署还要问：对哪个群体错、解释能否审计、模型是否记住个人记录。下一章把分组误差、可解释性与隐私写成与平均损失并列的约束。

<br>

# 公平与隐私

## 概念：公平

- **问题**
  - 模型在不同群体上的错误率、校准与资源分配可以系统性地不同；
  - 可解释性（interpretability）与隐私（privacy）是部署约束，与 $J(\theta)$ 不是同一目标。
- **范围**
  - 列出常用公平定义、解释工具与隐私风险，并说明彼此不可同时任意满足；
  - 输入带敏感属性的评估集，输出分组指标与限制条件。
- **章节衔接**
  - 前文章节优化平均 $J$ 或平均回报；
  - 本章把平均拆成群体，并写出不能单靠更多数据消掉的约束；
  - 下一章把划分、指标与项目节奏写成可执行协议。

## 偏差

训练数据记录的是历史决策，不是应然分配。平均准确率高，少数群体假阴性可以仍然高。敏感属性 $A$（受保护的类别）上常见定义：

| 定义 | 要求 | 失败时的含义 |
| :--- | :--- | :--- |
| 人口统计均等 | $P(\hat y=1\mid A=a)$ 相同 | 不同组正预测率不同 |
| 均等机会 | $P(\hat y=1\mid y=1,A=a)$ 相同 | 真阳性率不等 |
| 预测校准 | $P(y=1\mid\hat p,A=a)=\hat p$ | 同一分数在不同组含义不同 |

这些定义一般不能同时成立，除非基线风险 $P(y=1\mid A)$ 已经相同。选哪一个是规范问题，不是纯优化问题。必须先写清伤害：拒贷、误诊、多判刑，再选对应的等式。事后阈值调整能满足某些等式，但会改校准；对抗去除 $A$ 的表示不能证明公平，只能说线性探针读不出 $A$。

> [!NOTE]+ 分组公平
>
> 把平均准确率拆成敏感属性 $A$ 上的一组等式，例如正预测率相同或真阳性率相同。来源是社会选择与算法公平文献，不是 $J(\theta)$ 的正则项。相对只报总体指标：同一模型可以在少数组上系统性地差。相对“删掉 $A$ 这一列”：代理变量仍在，反馈回路会把模型决策写进明天的标签。各定义一般不能同时成立，选择是规范问题，必须先写清伤害再选等式。

<br>

算法偏差还包括度量偏置（标签本身歧视）、反馈回路（模型决策变成明天的训练数据）、代理变量（邮编代替种族）。只删列 $A$ 而留下强代理，不等于公平。

## 解释

全局：线性系数、树的分裂、置换特征重要性。局部：LIME、SHAP 用局部线性或博弈值解释单次预测。注意力权重不是因果解释。反事实：“若收入高 $10\%$，决策是否改变”需要结构假设，不能从观测数据唯一识别。解释用于调试与审计，不能单独当合规证明。

## 隐私

模型会记忆训练点。成员推断：判断某条记录是否在训练集。训练数据含医疗或标识时，发布模型等于间接发布样本。差分隐私通过在梯度或输出加噪，使“含不含单条记录”的分布接近，代价是效用下降。聚合统计仍可能再识别。项目阶段应写明：数据许可、是否可识别个人、日志保留、以及只在聚合层面报告的指标。

> [!EXAMPLE]+ 例25 两个群体的阈值
>
> 组 $A$ 基线正率为 $0.3$，组 $B$ 为 $0.1$。同一阈值 $0.5$ 下，均等机会可能破裂。把 $B$ 的阈值降到 $0.3$ 可拉齐真阳性率，但 $B$ 的精确率下降，校准曲线分裂。报告必须同时给出两组的 TPR、FPR、校准，并写清业务接受哪一种破裂。只报总体 AUC 会把该冲突藏起来。

<br>

> [!NOTE]+ 衔接
>
> 公平定义不能同时任意满足，选择是规范问题。下一章把同一套记号接到作业：书面推导加 `src/` 里的 NumPy 实现。

<br>

# 作业

## 概念：作业

- **问题**
  - 当前学期四份 Problem Set 加一份不计分的 PS0，书面推导与 Python / NumPy 同交；
  - 2026 与 2025 Autumn 的 zip 需 Stanford 登录。公开可对拍的完整题面是 [Summer 2020](https://cs229.stanford.edu/summer2020/) 的 `ps1.pdf` / `ps2.pdf` / `ps3.pdf`。2018 至 2024 多数学期沿用同一作业族；Fall 2025 把强化学习拆成 PS4，无监督仍在 PS3。
- **范围**
  - 概括每题问什么，给出可对拍的推导、数值检查与带注释实现；
  - 不照抄题面长文。学期微调时以 Gradescope zip 里的 `*.tex` 为准。
- **章节衔接**
  - 前面各章给目标与求解器；
  - 本章接到 `src/*/foo.py` 与排版 PDF。

| 编号 | 公开题面 | 主题 |
| :--- | :--- | :--- |
| PS0 | 课内 Ed / Canvas | NumPy、提交流程，不计分 |
| PS1 | [ps1.pdf](https://cs229.stanford.edu/summer2020/ps1.pdf) | 逻辑回归牛顿法、GDA、正样本部分标签、泊松回归、GLM 凸性、特征映射 |
| PS2 | [ps2.pdf](https://cs229.stanford.edu/summer2020/ps2.pdf) | 逻辑回归不收敛、垃圾短信、核的封闭性、核感知机、MNIST 浅层网 |
| PS3 | [ps3.pdf](https://cs229.stanford.edu/summer2020/ps3.pdf) | 倒立摆、KL、颜色压缩、$k$-means、半监督 EM、PCA、ICA |
| PS4 | Fall 2025 起单开 | 值迭代 / 策略迭代、模型估计、策略评估 |

提交：书面 PDF 加源码 zip（常用 `make_zip.py`）。禁用 scikit-learn。样本数记为 $n$（与 CS230 的 $m$ 不同）。$y$ 在分类题里是 $\{0,1\}$，不是 $\pm 1$。

> [!NOTE]+ 读法（先看这个）
>
> 每题只做三件事：**建模**（似然或损失是什么）、**推断**（梯度、闭式或牛顿步）、**对拍**（先用 $2$ 到 $4$ 个点手算）。callout 顺序：题在问什么、符号、先算一个数、再写一般式、圈出要交的答案。
>
> | 作业 | 先盯这个 |
> | :--- | :--- |
> | PS1 | $y\in\{0,1\}$；牛顿法用 Hessian 不是学习率；GDA 的 $\Sigma$ 除以 $n$ 不是 $n-1$ |
> | PS2 | 线性可分时无正则逻辑回归的 $\|\theta\|\rightarrow\infty$；朴素贝叶斯必须走 $\log$ |
> | PS3 | 半监督 EM 的 $Q$ 只对未标注样本更新；PCA 先中心化 |
> | PS4 | 值迭代用 $V_{\mathrm{new}}$ 覆盖整张表后再比较 $\infty$ 范数 |

<br>

> [!WARNING]+ 对拍与荣誉守则
>
> 笔记里的解答用于理解，不能整段贴进作业。课程禁止对照往年官方解答或他人代码。生成式工具若学期允许使用，必须按规范交 transcript。公开 PDF 与当年 zip 可能改数字。

<br>

## PS1 线性分类器

逻辑回归是判别模型，直接建 $p(y\mid x)$。高斯判别分析（Gaussian discriminant analysis, GDA）是生成模型，建 $p(x\mid y)p(y)$，再用贝叶斯反推后验。两者决策边界都可以写成 $\theta^{\top}x+\theta_0=0$，估计量不同。

> [!INFO]+ PS1.1 逻辑回归 Hessian 正半定
>
> **题在问什么**。平均经验损失
>
> $$
> J(\theta)=-\frac1n\sum_{i=1}^{n}\Big[y^{(i)}\log h_{\theta}(x^{(i)})+(1-y^{(i)})\log(1-h_{\theta}(x^{(i)}))\Big],
> $$
>
> $h_{\theta}(x)=g(\theta^{\top}x)$，$g(z)=(1+e^{-z})^{-1}$。要证明 Hessian $H$ 对一切 $z$ 满足 $z^{\top}Hz\ge 0$，从而 $J$ 凸。
>
> **先对一个样本求导**。令 $h=g(\theta^{\top}x)$。$\partial J/\partial\theta=(h-y)x$（单个样本、不除 $n$）。再对 $\theta$ 求导：$h(1-h)\,xx^{\top}$，因为 $g'=g(1-g)$。
>
> **一般式**。$H=\frac1n\sum_i h^{(i)}(1-h^{(i)})x^{(i)}(x^{(i)})^{\top}$。对任意向量 $z$，
>
> $$
> z^{\top}Hz=\frac1n\sum_i h^{(i)}(1-h^{(i)})(z^{\top}x^{(i)})^{2}\ge 0,
> $$
>
> 因为 $h(1-h)\in(0,1/4]$。故 $H\succeq 0$，$J$ 没有异于全局最小的局部最小。
>
> **要交的答案**：$H$ 的表达式，加上 $z^{\top}Hz\ge 0$。

<br>

> [!EXAMPLE]+ PS1.1 牛顿法实现
>
> **题在问什么**。从 $\theta=0$ 走牛顿步，直到 $\|\theta_{\mathrm{new}}-\theta_{\mathrm{old}}\|_1<10^{-5}$。把验证集概率写成指定文件，并画 $p=0.5$ 的直线。
>
> 牛顿步：$\theta\leftarrow\theta-H^{-1}\nabla J$。$X$ 为 $(n,d)$，含偏置列。决策边界：$\theta_0+\theta_1 x_1+\theta_2 x_2=0$，即 $x_2=-(\theta_0+\theta_1 x_1)/\theta_2$。
>
> **要交的答案**：验证集概率文件、图、以及“从 $\theta=0$ 迭代到 $L_1$ 步长小于 $10^{-5}$”。

```python
import numpy as np

def sigmoid(z):
    # 大正负输入分别改写，避免 overflow
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out

def logistic_newton(X, y, eps=1e-5, max_iter=50):
    """X: (n, d) 已含全 1 列；y: (n,) 取值 {0, 1}。"""
    n, d = X.shape
    theta = np.zeros(d)
    for _ in range(max_iter):
        h = sigmoid(X @ theta)                 # (n,)
        # 平均梯度 (1/n) X^T (h - y)
        grad = (X.T @ (h - y)) / n
        # Hessian = (1/n) X^T diag(h(1-h)) X
        w = h * (1.0 - h)
        H = (X.T * w) @ X / n
        # solve 比显式求逆稳
        delta = np.linalg.solve(H + 1e-12 * np.eye(d), grad)
        theta_new = theta - delta
        if np.abs(theta_new - theta).sum() < eps:
            return theta_new
        theta = theta_new
    return theta
```

<br>

> [!INFO]+ PS1.1 GDA 后验是逻辑形式
>
> **题在问什么**。共享协方差的两类高斯，证明
>
> $$
> p(y=1\mid x)=\frac{1}{1+\exp(-(\theta^{\top}x+\theta_0))}.
> $$
>
> **先写对数后验比**。
>
> $$
> \log\frac{p(y=1\mid x)}{p(y=0\mid x)}
> =\log\frac{\phi}{1-\phi}
> -\frac12(x-\mu_1)^{\top}\Sigma^{-1}(x-\mu_1)
> +\frac12(x-\mu_0)^{\top}\Sigma^{-1}(x-\mu_0).
> $$
>
> 二次项 $x^{\top}\Sigma^{-1}x$ 相消，剩下 $x$ 的一次式。令
>
> $$
> \theta=\Sigma^{-1}(\mu_1-\mu_0),\qquad
> \theta_0=\log\frac{\phi}{1-\phi}
> -\frac12\mu_1^{\top}\Sigma^{-1}\mu_1
> +\frac12\mu_0^{\top}\Sigma^{-1}\mu_0.
> $$
>
> 即得逻辑形式。**共享 $\Sigma$ 是线性边界的原因**；两类各用一个 $\Sigma$ 会留下二次项，变成二次判别。
>
> **MLE（要交的闭式）**。$\phi$ 是正类比例。$\mu_0,\mu_1$ 是各类样本均值。
>
> $$
> \Sigma=\frac1n\sum_{i=1}^{n}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^{\top}.
> $$
>
> 分母是 $n$ 不是 $n-1$。证明：对数似然对 $\mu_c$ 求导，令其为零，即该类均值；对 $\Sigma$ 求导用矩阵微积分，得到上述平均外积。
>
> **数据集对照**。ds1 两类大致椭圆、协方差接近，GDA 与逻辑回归边界接近。ds2 往往一类被拉成长条或边缘非高斯，GDA 更差。常见补救：对偏斜坐标取 $\log(1+|x|)$，或先标准化再拟合。
>
> **要交的答案**：$\theta,\theta_0$ 的式子、四个 MLE、两套图、以及“哪一套 GDA 更差、为何、用什么变换”。

```python
def gda_fit(X, y):
    """X: (n, d) 不含偏置；y: (n,) 取值 {0, 1}。"""
    n, d = X.shape
    phi = y.mean()
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)
    # 每行减去对应类均值后再做外积平均
    mu = np.where(y[:, None] == 1, mu1, mu0)
    xc = X - mu
    Sigma = (xc.T @ xc) / n
    return phi, mu0, mu1, Sigma

def gda_theta(phi, mu0, mu1, Sigma):
    iS = np.linalg.inv(Sigma)
    theta = iS @ (mu1 - mu0)
    theta0 = (
        np.log(phi / (1.0 - phi))
        - 0.5 * mu1 @ iS @ mu1
        + 0.5 * mu0 @ iS @ mu0
    )
    return theta, theta0
```

<br>

> [!INFO]+ PS1.2 只有部分正标签
>
> **题在问什么**。真实标签 $t$ 看不到。观测 $y$：若 $t=1$，以概率 $\alpha$ 标成 $1$，否则标 $0$；若 $t=0$，永远标 $0$。要用 $(x,y)$ 去逼近 $p(t=1\mid x)$。
>
> **先用贝叶斯**。$p(y=1\mid t=0,x)=0$，故 $y=1$ 只能来自 $t=1$：
>
> $$
> p(t=1\mid y=1,x)=1.
> $$
>
> 再拆 $p(y=1\mid x)=p(y=1\mid t=1,x)p(t=1\mid x)=\alpha\,p(t=1\mid x)$。故
>
> $$
> p(t=1\mid x)=\frac1\alpha\,p(y=1\mid x).
> $$
>
> **估计 $\alpha$**。再假设真实过程无噪声，即 $p(t=1\mid x)\in\{0,1\}$。当 $y=1$ 时必有 $t=1$，于是 $h(x)=p(y=1\mid x)=\alpha$。因此
>
> $$
> \alpha=\mathbb{E}[h(x)\mid y=1]\approx\frac{1}{|V_+|}\sum_{x\in V_+}h(x),
> $$
>
> $V_+$ 是验证集里 $y=1$ 的点。用训练集估 $\alpha$ 会偏乐观，题面指定验证集。
>
> **实现顺序**。先用 $t$ 训一条上界；再用 $y$ 训朴素模型；最后把朴素概率除以 $\hat\alpha$。只做排序时 $\alpha$ 可省略，因为 $1/\alpha$ 不改变次序。
>
> **要交的答案**：上面三式、三张测试图、验证集上的 $\hat\alpha$。

<br>

> [!INFO]+ PS1.3 泊松回归
>
> **题在问什么**。计数 $y=0,1,2,\ldots$，$p(y;\lambda)=e^{-\lambda}\lambda^y/y!$。写成指数族，再导出典范响应与梯度上升。
>
> $$
> p(y;\lambda)=\frac{1}{y!}\exp\big(y\log\lambda-\lambda\big).
> $$
>
> 故 $b(y)=1/y!$，$\eta=\log\lambda$，$T(y)=y$，$a(\eta)=e^{\eta}$。均值 $\lambda=e^{\eta}$，这就是典范响应：$h_{\theta}(x)=\exp(\theta^{\top}x)$。
>
> 单样本对数似然（略去 $y!$）：$\ell=y\,\theta^{\top}x-e^{\theta^{\top}x}$。对 $\theta_j$ 求导：$(y-e^{\theta^{\top}x})x_j$。随机梯度**上升**（最大化似然）：
>
> $$
> \theta\leftarrow\theta+\alpha\big(y-\exp(\theta^{\top}x)\big)x.
> $$
>
> 全批把括号换成 $\frac1n\sum_i(y^{(i)}-\exp(\theta^{\top}x^{(i)}))x^{(i)}$。停止条件：参数改变的欧氏范数小于 $10^{-5}$。验证图：横轴真计数、纵轴预测均值。学习率过大 $\exp$ 会爆。
>
> **要交的答案**：$b,\eta,T,a$、响应 $e^{\eta}$、更新式、散点图。

```python
def poisson_fit(X, y, lr=1e-5, eps=1e-5, max_iter=10000):
    """X: (n, d) 已含偏置列；y: (n,) 非负计数。全批梯度上升。"""
    theta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        lam = np.exp(X @ theta)            # 预测均值，必须为正
        grad = X.T @ (y - lam) / len(y)
        theta_new = theta + lr * grad
        if np.linalg.norm(theta_new - theta) < eps:
            return theta_new
        theta = theta_new
    return theta
```

<br>

> [!INFO]+ PS1.4 GLM 的 NLL 凸
>
> **题在问什么**。标量自然参数 $\eta=\theta^{\top}x$，$T(y)=y$，$p(y;\eta)=b(y)\exp(\eta y-a(\eta))$。证明负对数似然对 $\theta$ 凸。
>
> 密度对 $\eta$ 积分恒为 $1$，两边对 $\eta$ 求导：$\mathbb{E}[y]=a'(\eta)$。再导一次：$\mathrm{Var}(y)=a''(\eta)\ge 0$。
>
> 单样本 NLL：$\ell(\theta)=-y\,\theta^{\top}x+a(\theta^{\top}x)-\log b(y)$。Hessian：
>
> $$
> \nabla^{2}\ell=a''(\theta^{\top}x)\,xx^{\top}=\mathrm{Var}(y\mid x)\,xx^{\top}\succeq 0.
> $$
>
> 方差非负，故对一切 $\theta$ 正半定。这就是“换指数族只换 $a$，凸性自动继承”。
>
> **要交的答案**：均值 $=$ $a'$、方差 $=$ $a''$、Hessian $=$ $\mathrm{Var}\,xx^{\top}$。

<br>

> [!EXAMPLE]+ PS1.5 特征映射
>
> **题在问什么**。$h_{\theta}(x)=\theta_3 x^3+\theta_2 x^2+\theta_1 x+\theta_0$ 对 $x$ 非线性，对 $\theta$ 仍线性。令 $\phi(x)=(1,x,x^2,x^3)$，在新特征上做普通线性回归。
>
> 目标 $J(\theta)=\frac1{2n}\sum_i(\theta^{\top}\phi(x^{(i)})-y^{(i)})^{2}$。全批梯度：$\nabla J=\frac1n\Phi^{\top}(\Phi\theta-y)$。闭式用 `np.linalg.solve(Phi.T @ Phi, Phi.T @ y)`，不要写 `inv`。
>
> $k$ 升高时训练曲线更贴合，验证会先好后坏。数据由 $\sin x$ 加噪生成时，把 $\sin x$ 放进 $\phi$ 后，$k=0$ 已经能拟合主体。小样本上 $k>n$ 时曲线会“发疯”，这是过拟合：假设类比数据更灵活，开始解释噪声。
>
> **要交的答案**：$J$ 与梯度、三张图（$k=3$；$k\in\{3,5,10,20\}$；加 $\sin$；小样本）、以及过拟合一两句。

<br>

## PS2 稳定、核与浅层网

> [!INFO]+ PS2.1 逻辑回归在线性可分数据上不收敛
>
> **题在问什么**。同一份实现，数据集 A 正常停，数据集 B 损失一直降、$\|\theta\|$ 一直涨。
>
> **原因**。B 线性可分。最大似然要把正确类的 $h$ 推到 $0$ 或 $1$，需要 $|\theta^{\top}x|\rightarrow\infty$，所以 $\|\theta\|\rightarrow\infty$。A 有重叠，最优 $\theta$ 有限。这不是溢出。
>
> | 改动 | 能否在 B 上收敛 | 理由 |
> | :--- | :--- | :--- |
> | 换一个固定学习率 | 否 | 只改步长，不改“最优在无穷远” |
> | 学习率按 $1/t^{2}$ 衰减 | 否 | 参数仍被往无穷推，只是更慢 |
> | 线性缩放输入 | 否 | 可分性不变 |
> | 加 $\lambda\|\theta\|_2^{2}$ | 是 | 岭正则把最优拉回有限点 |
> | 给 $x$ 或 $y$ 加噪声 | 通常是 | 破坏完全可分 |
>
> SVM 有间隔约束或 $\|\theta\|$ 惩罚，最优在有限点，对这类 B 不爆炸。
>
> **要交的答案**：B 线性可分的证据（例如画出分隔直线、或训练准确率 $=1$ 且 $\|\theta\|$ 单调增）、上表、SVM 一句。

<br>

> [!EXAMPLE]+ PS2.2 垃圾短信朴素贝叶斯
>
> **题在问什么**。词袋加拉普拉斯平滑，预测垃圾 / 非垃圾。词表大小、测试准确率、最能指示垃圾的五个词、以及验证集上最好的 RBF 半径。
>
> 词表：小写、按空白切、出现次数低于阈值的丢掉。 multinomial：$p(w\mid y)=\frac{c_{y,w}+1}{C_y+|V|}$。预测必须走对数，否则连乘下溢成 $0$：
>
> $$
> \log p(y\mid x)=\log p(y)+\sum_w c_w\log p(w\mid y).
> $$
>
> 指示度 $\log\frac{p(w\mid\mathrm{spam})}{p(w\mid\mathrm{ham})}$ 最大的五个词。SVM 半径在验证集上扫，不要看测试。
>
> **要交的答案**：$|V|$、测试准确率、五个词、验证集最优半径。公开学期 $|V|$ 与准确率会变，以当次 `spam_dictionary` 为准。

```python
def fit_naive_bayes(X, y):
    """X: (n, V) 词频；y: (n,) 取值 {0, 1}。返回 log 先验与 log 词概率。"""
    V = X.shape[1]
    log_prior = np.array([
        np.log((y == 0).mean()),
        np.log((y == 1).mean()),
    ])
    log_cond = np.zeros((2, V))
    for c in (0, 1):
        counts = X[y == c].sum(axis=0) + 1.0      # Laplace +1
        log_cond[c] = np.log(counts / counts.sum())
    return log_prior, log_cond

def predict_naive_bayes(X, log_prior, log_cond):
    # (n, V) @ (V, 2) + (2,) 得到两类 log 联合，再取 argmax
    scores = X @ log_cond.T + log_prior
    return scores.argmax(axis=1)
```

<br>

> [!INFO]+ PS2.3 核的封闭性
>
> Mercer's 定理：$K$ 是核当且仅当一切有限 Gram 矩阵对称且正半定。$K_1,K_2$ 已是核，$a>0$，$p$ 是正系数多项式。
>
> | $K$ | 是核？ | 理由 |
> | :--- | :--- | :--- |
> | $K_1+K_2$ | 是 | PSD 对加法封闭 |
> | $K_1-K_2$ | 否 | $K_2=2K_1$ 时变成 $-K_1$ |
> | $aK_1$ | 是 | 正数倍保持 PSD |
> | $-aK_1$ | 否 | 负特征值 |
> | $K_1K_2$（逐点积） | 是 | Schur 积定理：PSD 的 Hadamard 积仍 PSD |
> | $f(x)f(z)$ | 是 | $\phi(x)=f(x)$ 的内积 |
> | $K_3(\phi(x),\phi(z))$ | 是 | 先映射再套核 |
> | $p(K_1)$ | 是 | 正系数加与积的复合 |
>
> **要交的答案**：八行是 / 否，否的给反例，是的给 PSD 或显式 $\phi$。

<br>

> [!EXAMPLE]+ PS2.4 核感知机
>
> 一次遍历更新 $\theta\leftarrow\theta+\alpha(y-h(x))\phi(x)$，但 $\phi$ 无穷维，不能存 $\theta$。$\theta^{(0)}=0$，故
>
> $$
> \theta^{(i)}=\sum_{t=1}^{i}\beta_t\phi(x^{(t)}),\qquad
> \theta^{\top}\phi(x)=\sum_{t}\beta_t K(x^{(t)},x).
> $$
>
> 预测：$h(x)=\mathbf{1}\{\sum_t\beta_t K(x^{(t)},x)\ge 0\}$。更新只改对应的 $\beta$。
>
> 点积核能分线性可分数据。RBF 通常最好。题面给的“$x=z$ 为 $-1$ 否则 $0$”不是 PSD，会完全失败。

```python
def predict(state, x, kernel):
    # state["x"] 存见过的样本，state["beta"] 存系数
    s = 0.0
    for xt, b in zip(state["x"], state["beta"]):
        s += b * kernel(xt, x)
    return 1 if s >= 0 else 0

def update_state(state, x, y, kernel, alpha=1.0):
    yhat = predict(state, x, kernel)
    # 分对则 beta=0，分错才留下这个支持向量
    state["x"].append(x)
    state["beta"].append(alpha * (y - yhat))
    return state
```

<br>

> [!INFO]+ PS2.5 MNIST 单隐层
>
> 前向：$z^{[1]}=W^{[1]\top}x+b^{[1]}$，$a^{[1]}=\sigma(z^{[1]})$，$z^{[2]}=W^{[2]\top}a^{[1]}+b^{[2]}$，$\hat y=\mathrm{softmax}(z^{[2]})$。损失是交叉熵。反向输出层 $\partial\ell/\partial z^{[2]}=\hat y-y$（one-hot）。隐层 $\partial\ell/\partial z^{[1]}=W^{[2]}(\hat y-y)\odot a^{[1]}(1-a^{[1]})$。实现用小批量、把 $\log$ 与 softmax 合成稳定形式。
>
> **要交的答案**：训练 / 开发准确率曲线、实现 `forward` / `backward` / `update`。

<br>

## PS3 无监督与倒立摆

Fall 2025 起倒立摆常放到 PS4，推导不变。

> [!EXAMPLE]+ PS3.1 / PS4 倒立摆
>
> **题在问什么**。状态已被离散成整数。动作只有左 / 右。失败给负回报并重置，其余回报 $0$。$\gamma=0.995$。周期地用计数估 $T,R$，再值迭代，再贪心。
>
> 初值：$R(s)=0$，$T(s'\mid s,a)$ 均匀。维护 $N(s,a,s')$ 与回报和。每次失败后：
>
> $$
> \hat T(s'\mid s,a)=\frac{N(s,a,s')}{\sum_{u}N(s,a,u)},\qquad
> \hat R(s)=\text{该状态观测回报的平均}.
> $$
>
> 未见过的 $(s,a)$ 保持均匀。值迭代：
>
> $$
> V(s)\leftarrow \hat R(s)+\gamma\max_a\sum_{s'}\hat T(s'\mid s,a)V(s').
> $$
>
> 直到 $\max_s|V_{\mathrm{new}}-V_{\mathrm{old}}|<\mathrm{tol}$。连续若干次值迭代第一步就收敛，则认为模型不再变。正确实现时，平滑后的“撑住步数”大约在 $60$ 次失败附近变平。种子 $1,2,3$ 会得到不同曲线：离散化加随机重置，算法对初始化敏感。
>
> **要交的答案**：失败次数、学习曲线、三种种子的观察。

```python
def value_iteration(T, R, gamma=0.995, tol=1e-3, max_iter=200):
    """T[s, a, s2] = P(s2 | s, a)，R[s] 只依赖状态。"""
    nS, nA, _ = T.shape
    V = np.zeros(nS)
    for it in range(max_iter):
        # Q(s,a) = R(s) + gamma * sum_{s'} T(s'|s,a) V(s')
        Q = R[:, None] + gamma * (T * V[None, None, :]).sum(axis=2)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1), it
        V = V_new
    return V, Q.argmax(axis=1), max_iter
```

<br>

> [!INFO]+ PS3.2 KL 与最大似然
>
> $D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$。$f=-\log$ 严格凸，Jensen：
>
> $$
> D_{\mathrm{KL}}(P\|Q)=-\sum_x P(x)\log\frac{Q(x)}{P(x)}
> \ge -\log\sum_x Q(x)=0,
> $$
>
> 等号当且仅当 $Q/P$ 为常数，即 $P=Q$。
>
> 链式法则：把 $\log\frac{P(x,y)}{Q(x,y)}$ 拆成 $\log\frac{P(x)}{Q(x)}+\log\frac{P(y\mid x)}{Q(y\mid x)}$，再按 $P(x,y)$ 求和。
>
> 经验分布 $\hat P$ 在每个训练点上质量 $1/n$。
>
> $$
> D_{\mathrm{KL}}(\hat P\|P_{\theta})
> =\mathrm{const}-\frac1n\sum_i\log P_{\theta}(x^{(i)}).
> $$
>
> 故最小化 KL 等价于最大似然。
>
> **要交的答案**：非负证明、链式法则、KL $=$ 常数 $-$ 平均对数似然。

<br>

> [!EXAMPLE]+ PS3.3 $k$-means 压颜色
>
> 像素看成 $\mathbb{R}^3$ 的 $(r,g,b)$。在小图上跑 $k=16$，中心初始化为随机像素，至少 $30$ 轮。大图每个像素换成最近中心。
>
> 压缩比：原图每像素 $24$ 位。新图每像素 $4$ 位索引（$16$ 色）再加调色板 $16\times 24$ 位，调色板相对 $512^2$ 可忽略，约 $24/4=6$ 倍。
>
> **要交的答案**：大图对比、约 $6$ 倍。

<br>

> [!INFO]+ PS3.4 半监督 EM
>
> 目标 $\ell_{\mathrm{semi}}=\ell_{\mathrm{unsup}}+\alpha\ell_{\mathrm{sup}}$。E 步只对**未标注**样本算 $Q_i^{(t)}(z)=p(z\mid x^{(i)};\theta^{(t)})$。M 步把未标注的期望完全数据对数似然，加上 $\alpha$ 倍已标注的真实完全数据对数似然。
>
> 单调性：标准 EM 已证 $\ell_{\mathrm{unsup}}(\theta^{(t+1)})\ge\ell_{\mathrm{unsup}}(\theta^{(t)})$；$\ell_{\mathrm{sup}}$ 在 M 步被显式最大化，故加权和上升。
>
> GMM：未标注责任度
>
> $$
> w_j^{(i)}=\frac{\phi_j\,\mathcal{N}(x^{(i)};\mu_j,\Sigma_j)}{\sum_\ell\phi_\ell\,\mathcal{N}(x^{(i)};\mu_\ell,\Sigma_\ell)}.
> $$
>
> 已标注样本的责任度是 one-hot，再乘 $\alpha$。均值 / 协方差 / $\phi$ 用“未标注权重 $+$ $\alpha$ 已标注计数”的加权公式。半监督通常更稳、更少轮，并能分开低方差三簇与一簇高方差噪声。
>
> **要交的答案**：单调性、$Q$ 公式、$\mu,\Sigma,\phi$ 的加权更新、六张散点、三点对照。

<br>

> [!INFO]+ PS3.5 PCA 的重构误差
>
> $f_u(x)=(u^{\top}x)u$。$\|x-f_u(x)\|_2^2=\|x\|_2^2-(u^{\top}x)^2$。最小化平均重构误差等价于最大化 $u^{\top}(\sum_i x^{(i)}x^{(i)\top})u$，约束 $u^{\top}u=1$。拉格朗日乘数给出 $Su=\lambda u$，最大特征值对应第一主成分。数据须先中心化。
>
> **要交的答案**：从 $\|x\|^2-(u^{\top}x)^2$ 走到特征方程。

<br>

> [!INFO]+ PS3.6 ICA
>
> 源为标准正态时，$\log g'(w^{\top}x)=-\frac12(w^{\top}x)^2+\mathrm{const}$。最大化 $\log|W|-\frac12\|XW^{\top}\|_F^2$ 只约束 $W$ 使 $XW^{\top}$ 白化，右乘任意旋转矩阵似然不变。高斯源不可辨识。
>
> Laplace 密度 $\frac12 e^{-|s|}$。单样本随机梯度：
>
> $$
> W\leftarrow W+\alpha\big((W^{-1})^{\top}-\mathrm{sign}(Wx)\,x^{\top}\big).
> $$
>
> 实现时对学习率退火，并打乱样本顺序。正确 $W$ 应让 `split_0.wav` 接近参考音轨。
>
> **要交的答案**：高斯下的旋转歧义、Laplace 更新、`W.txt`。

<br>

> [!NOTE]+ 衔接
>
> 作业按 Problem Set 组织。下一章把同一张表对回项目：划分、指标与非深度基线，而不是“短信”或“倒立摆”。

<br>

# 实践

## 概念：建议

- **问题**
  - 接到真实数据时，常见错误来自划分、指标与泄漏，梯度公式本身往往已经写对；
  - 2025 Autumn 的 Lecture 13 与 TA 评估指标讲座对应本章。
- **范围**
  - 给出划分、基线、误差分析与项目节奏；
  - 输入原始数据集，输出可复现的实验记录。
- **章节衔接**
  - 公式正确仍可能因泄漏或指标错位而失败；
  - 本章固定划分、基线与误差表；
  - 下一章把公式收成默写清单。

## 划分

训练 / 验证 / 测试必须在同一条协议下切开。时间数据按时间；患者、用户、文档必须整实体划分，否则同一人的两张切片分别进入训练与测试，指标虚高。若根据测试结果回头改模型，测试集即失去一次评估的意义。标准化、词表与目标编码只允许在训练折上估计。

## 指标

先写错误代价，再选指标。2025 Autumn 的 TA Lecture 4 单独讲评估，原因是作业与项目里最常见的失败不是公式写错，而是报了一个与决策无关的数字。

| 任务 | 常用指标 | 陷阱 |
| :--- | :--- | :--- |
| 平衡二分类 | 准确率、AUC | 阈值 $0.5$ 不必最优 |
| 不平衡分类 | PR-AUC、F1、召回@固定精度 | 准确率被多数类刷高 |
| 概率输出 | ECE、可靠性图 | SVM 间隔不是概率 |
| 回归 | MAE、MSE、分位损失 | MSE 被离群点主导 |
| 检索 / 排序 | Recall@$k$、nDCG | 只报 MRR 会忽略长尾 |
| 生成 | 任务准确率、忠实度 | 流畅 $\ne$ 正确 |
| RL | 评估回报、约束违反 | TD 损失下降不等于会玩 |

宜指定一个优化指标，其余作为满意度约束。误差分析至少包含混淆矩阵、阈值曲线，以及按长度、人群或时间的切片。下一步实验应由占比最大的错误类型决定，而不是默认更换更大模型。

## 基线

结果表应包含常数预测，以及线性模型、逻辑回归、$k$-NN 或单棵树之一。深度模型或 LLM 若只优于未写明的方法，不能说明任务需要深度结构。去掉正则、核或检索时应分别报告。负结果应保留，用以说明假设类或数据是否不足。

## 项目

公开时间线（2025 Autumn）：提案、中期、报告、海报。提案写清问题、数据许可与规模、划分、指标、最简基线、风险。中期必须有数字与误差表。最终报告让他人能复现划分与命令。与其他课合并时，CS229 部分必须能单独拆出：假设、似然或损失、与非深度基线的比较。

常见失败：选题过大、中期仍无标签、用测试调参、不固定 seed、把作业数据改标题当项目。范围宜先缩小，跑通后再扩展。

> [!WARNING]+ 项目失败模式
>
> 公式正确仍可能因实体泄漏、用测试选 $\lambda$、只报总体准确率、缺少非深度基线或误差表无切片而失分。应先固定评估协议，再增加模型容量。

<br>

> [!NOTE]+ 衔接
>
> 划分、指标、基线与误差表是把前面算法接到未布置数据上的接口。下一章把公式收成清单；清单用来答题，协议用来做项目。

<br>

# 复习

课程把机器学习写成可声明的目标加可实现的求解器。线性模型给出闭式解与凸优化；指数族把回归与分类收成同一构造；核与网络扩大假设类；无监督补上没有 $y$ 的密度与表示；RL 把预测换成决策。

$$
\begin{aligned}
\text{拟合训练} &\leftarrow \text{假设类、优化、可辨识性},\\
\text{拟合验证} &\leftarrow \text{正则、数据、划分},\\
\text{拟合部署} &\leftarrow \text{验证分布是否等于现实，以及公平与隐私约束}.
\end{aligned}
$$

- 回归与分类：正规方程、$h-y$ 与 $g(\theta^{\top}x)$、牛顿 Hessian $X^{\top}RX$。
- GLM：$\eta=\theta^{\top}x$，$g(\eta)=E[T(y)]$。
- 生成：GDA 线性边界；Laplace 平滑。
- 核与 SVM：Mercer；$w=\sum\alpha_i y^{(i)}\phi(x^{(i)})$；$0\le\alpha_i\le C$。
- 树与提升：不纯度；指数损失前向阶段。
- 网络：局部 Jacobian；输出层 $\hat y-y$。
- 泛化：偏差-方差；有限类 $\log|\mathcal{H}|/n$；VC。
- 无监督：Lloyd；ELBO；$\Sigma u=\lambda u$；因子分析 $x=\mu+\Lambda z+\varepsilon$；ICA 非高斯。
- 生成与语言：$\varepsilon$-预测；InfoNCE；$\mathrm{softmax}(QK^{\top}/\sqrt{d_k})$。
- 控制：贝尔曼；$P_t$ Riccati；$\nabla J=E[\nabla\log\pi\,A]$。

期末闭卷部分通常覆盖前半：LMS 与正规方程、逻辑回归梯度与牛顿、GLM、GDA/NB、核与间隔、偏差-方差、EM 责任度。编程作业覆盖 shape、稳定 softmax、EM 下溢。项目覆盖能否把上述工具用在未布置过的数据上。

默写清单：

- $J(\theta)=\frac12\|X\theta-y\|_{2}^{2}$，$\theta=(X^{\top}X)^{-1}X^{\top}y$。
- $\nabla_{\theta}\ell_{\mathrm{logistic}}=X^{\top}(y-h)$，$H=-X^{\top}RX$。
- Bernoulli 的 $\eta=\log\frac{\phi}{1-\phi}$，$\phi=\sigma(\eta)$。
- GDA 共享 $\Sigma$ 时边界线性；QDA 二次。
- Laplace：分子 $+1$，分母 $+|V|$。
- 几何间隔 $\gamma=\hat\gamma/\|w\|$；硬间隔 $\min\|w\|^{2}/2$ s.t. $y(w^{\top}x+b)\ge 1$。
- 对偶 $\max\sum\alpha_i-\frac12\sum\alpha_i\alpha_j y_i y_j K_{ij}$。
- ELBO $=\sum Q\log\frac{p(x,z)}{Q}$；E 步 $Q=p(z\mid x)$。
- PCA：$\Sigma u=\lambda u$；SVD 右向量。
- $V(s)\leftarrow\max_a\sum_{s'}P[R+\gamma V(s')]$。
- REINFORCE：$\sum\nabla\log\pi(G_t-b(s_t))$。
- PPO：$r_t=\pi/\pi_{\mathrm{old}}$，裁剪 $1\pm\varepsilon$。

闭卷实现题常考：$X$ 的行列约定、对 $X^{\top}X$ 直接求逆、EM 未加岭、因果掩码缺失、测试集被当成验证集。策略题会给分组错误表，要求指出下一步实验及其指标上限。

> [!NOTE]+ 三条失败轴
>
> - 拟合训练失败：假设类不够、优化发散、实现 shape 错误、不可辨识。
> - 拟合验证失败：正则过弱或过强、数据量不够、标注噪声、用测试调参。
> - 拟合部署失败：划分泄漏、验证分布不等于现实、公平约束被平均指标掩盖。
> - 课堂扩展主要落在部署轴：评估协议、分组指标、隐私与项目节奏。

<br>

# 参考文献

## 课程

1. Stanford CS229. *Machine Learning* 课程主页。[https://cs229.stanford.edu/](https://cs229.stanford.edu/)
2. Stanford CS229. *Fall 2025* 课表。[https://cs229.stanford.edu/index.html-fall25](https://cs229.stanford.edu/index.html-fall25)
3. Stanford Bulletin. *CS229: Machine Learning*. [https://bulletin.stanford.edu/courses/1057501](https://bulletin.stanford.edu/courses/1057501)
4. Ma T, Ng A. *CS229 Lecture Notes*. 2026-08-23（Spring 2026 主讲义）。[PDF](https://cs229.stanford.edu/main_notes.pdf)

## 监督学习与核

1. Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. Springer.
2. Boyd S, Vandenberghe L. *Convex Optimization*. Cambridge University Press.
3. Cortes C, Vapnik V. *Support-Vector Networks*. Machine Learning, 1995.
4. Platt J. *Sequential Minimal Optimization*. 1998.
5. Freund Y, Schapire R. *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*. JCSS, 1997.

## 泛化、无监督与生成

1. Vapnik V. *Statistical Learning Theory*. Wiley, 1998.
2. Dempster A, Laird N, Rubin D. *Maximum Likelihood from Incomplete Data via the EM Algorithm*. JRSS B, 1977.
3. Hyvärinen A, Oja E. *Independent Component Analysis: Algorithms and Applications*. Neural Networks, 2000.
4. Kingma DP, Welling M. *Auto-Encoding Variational Bayes*. ICLR, 2014.
5. Ho J et al. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020.
6. Belkin M et al. *Reconciling modern machine-learning practice and the classical bias-variance trade-off*. PNAS, 2019.（双重下降）

## 语言、控制与公平

1. Vaswani A et al. *Attention Is All You Need*. NeurIPS, 2017.
2. Sutton R, Barto A. *Reinforcement Learning: An Introduction*. MIT Press.
3. Williams RJ. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. Machine Learning, 1992.
4. Schulman J et al. *Proximal Policy Optimization Algorithms*. 2017.
5. Barocas S, Hardt M, Narayanan A. *Fairness and Machine Learning*. [https://fairmlbook.org/](https://fairmlbook.org/)
6. Dwork C et al. *Calibrating Noise to Sensitivity in Private Data Analysis*. TCC, 2006.

<br>

# 工具

## 数值

- [NumPy](https://numpy.org/)：shape 对齐、向量化与从零实现；CS229 约定样本在第 $0$ 维。
- [SciPy](https://scipy.org/)：`linalg.cho_solve`、稀疏核、优化器。
- [scikit-learn](https://scikit-learn.org/)：对照正规方程、SVM、PCA、GMM；作业仍要求自写核心循环。

## 深度学习与检索

- [PyTorch](https://pytorch.org/)：自动微分；默认 batch-first。
- [Hugging Face Transformers / tokenizers](https://huggingface.co/)：分词必须与检查点绑定。
- FAISS / 其他 ANN：检索召回与延迟同时记录。

## 控制与评估

- 简单网格世界自写即可；连续控制再考虑标准基准环境。
- 分组指标与校准曲线应写进验证脚本，而不是事后用表格软件手算。

## 线性代数与概率复习

- CS229 课程页的 Linear Algebra Review、Probability Review 与 Python/NumPy 讲义。
- 作业 Problem Set 0 用于在正式算法前对齐矩阵微积分与广播。

<br>




