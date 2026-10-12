---
title: 斯坦福CS229：Machine Learning
date: 2026-09-22
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS229 详细课程笔记：从线性回归、GLM 与 SVM，到泛化、EM、PCA、扩散模型、LLM 与强化学习。
---

- **主讲**
  - 2026 年课程主页上列出的讲师是 [Jehangir Amjad](https://cs229.stanford.edu/) 与 [Anand Avati](https://cs229.stanford.edu/)；
  - 2025 Autumn 学期由 Moses Charikar、Carlos Guestrin 与 Andrew Ng 主讲。
- **入口**
  - [Stanford Bulletin：CS229](https://bulletin.stanford.edu/courses/1057501)：官方课程目录中的课程描述。
  - [课程主页](https://cs229.stanford.edu/)：当前学期的课表、讲义与作业入口。
  - [Fall 2025 课表](https://cs229.stanford.edu/index.html-fall25)：本笔记补充课堂专题（决策树、Boosting、评估指标、公平与隐私、LLM）时依据的课表。
  - [主讲义 PDF](https://cs229.stanford.edu/main_notes.pdf)：Tengyu Ma 与 Andrew Ng 编写的 *CS229 Lecture Notes*，本笔记的推导以它为准。
  - [公开作业族（Summer 2020）](https://cs229.stanford.edu/summer2020/)：公开可下载的一组作业与题目。

<br>

> [!INFO]+ 版本范围
>
> Stanford Bulletin 给出的课程范围包括：统计模式识别（statistical pattern recognition）、线性与非线性回归、非参数方法（non-parametric methods）、指数族（exponential family）、GLM、支持向量机（support vector machines）、核方法（kernel methods）、深度学习、模型与特征选择、学习理论（learning theory）、ML advice、聚类（clustering）、密度估计（density estimation）、EM、降维（dimensionality reduction）、ICA、PCA、强化学习（reinforcement learning）与自适应控制（adaptive control）、MDP、近似动态规划（approximate dynamic programming）与策略搜索（policy search）。
>
> 算法推导以 Tengyu Ma 与 Andrew Ng 的 *CS229 Lecture Notes*（2026-08-23 版本）为准。决策树、Boosting、评估指标、公平与隐私、LLM 的 RAG 与微调这几部分讲义覆盖较少，按 2025 Autumn 课表补入。不同学期的讲座题目会有调整，但回归、GLM、核、SVM、EM、PCA 与 MDP 这几块核心内容通常都会保留，因此笔记以它们为骨架。

<br>


# 总览：课程地图、建模流程与全书约定

CS229 覆盖的模型非常多：从最朴素的线性回归一路走到扩散模型、大语言模型和强化学习。初学时最容易出现的问题不是某个公式看不懂，而是“学了十几个模型，却说不清它们之间有什么关系”。这一章的作用就是在进入具体模型之前，先给出一张地图，再把后面反复用到的数学工具、计算代价估算方法、数值稳定性技巧、Python 实现习惯与记号统一固定下来。

核心想法可以用一句话概括：几乎每个 CS229 模型都可以拆成“假设类 + 损失（或似然）+ 约束（或先验）+ 求解器”四个零件，再加上一步“泛化诊断”。看懂了这四个零件分别是什么，新模型就只是零件的新组合。

本章之后紧接着的是线性回归，它是这套记号下最简单、可以完整手算的可训练模型，后面的逻辑回归、GLM、核方法等都在它的基础上逐步推广。

## 课程内容地图

CS229 的内容大致分成五个板块，再加上若干课堂扩展专题。下面按“学什么、为什么放在这里”的顺序列出。

- **监督学习（supervised learning）**：给定带标签的数据 $(x,y)$，学一个从输入到标签的映射。这一板块是全课的基础，依次包括：
  - 线性回归（linear regression）、LMS 更新规则、正规方程（normal equations），以及“高斯噪声下的最大似然（maximum likelihood）恰好等于最小二乘”这一概率解释；
  - 局部加权回归（locally weighted linear regression），它是第一个非参数方法的例子；
  - 逻辑回归（logistic regression）、感知机（perceptron）、多分类的 softmax，以及用于更快求解的牛顿法（Newton's method）；
  - 指数族与广义线性模型（generalized linear models, GLM），它把线性回归、逻辑回归、softmax 统一成同一个框架；
  - 高斯判别分析（Gaussian discriminant analysis, GDA）与朴素贝叶斯（naive Bayes），它们是“生成式”建模的代表；
  - 核方法与支持向量机（support vector machines, SVM），用核技巧在高维特征空间里做线性分类；
  - 决策树（decision trees）与 Boosting，它们是非线性、可组合的另一类模型。
- **深度学习与泛化（generalization）**：模型能在训练集上拟合好之后，接下来要问“它在没见过的数据上表现如何”。
  - 多层感知机（multilayer perceptron）与模块化的反向传播（backpropagation）；
  - 偏差-方差（bias-variance）分解、双重下降（double descent）现象，以及有限/无限假设类（hypothesis class）情形下的样本复杂度（sample complexity）；
  - 正则化（regularization）、隐式正则（implicit regularization）、交叉验证（cross-validation），以及正则化的贝叶斯解释（相当于给参数加先验）。
- **无监督学习（unsupervised learning）**：没有标签，只有输入 $x$，目标是发现数据中的结构。
  - $k$-means 聚类；
  - 高斯混合模型（Gaussian mixture model, GMM）、一般形式的 EM 算法、证据下界 ELBO，以及变分推断（variational inference）；
  - 因子分析（factor analysis）、主成分分析 PCA 与独立成分分析 ICA。
- **生成模型与基础模型（foundation models）**：较新的内容，把前面的概率工具用到现代大模型上。
  - 扩散模型（diffusion models）及其连续时间的反向过程；
  - 表示学习（representation learning）、对比学习（contrastive learning）、检索与 RAG；
  - 分词（tokenization）、自回归（autoregressive）语言模型的损失、Transformer、混合专家 MoE、监督微调 SFT 与推理。
- **控制与决策**：数据不再是固定的 $(x,y)$ 对，而是智能体与环境交互产生的轨迹。
  - 马尔可夫决策过程 MDP、值迭代（value iteration）、策略迭代（policy iteration）、值函数逼近（value function approximation）；
  - 线性二次调节器 LQR、微分动态规划 DDP、线性二次高斯控制 LQG；
  - 策略梯度方法 REINFORCE 与 PPO。
- **课堂扩展专题**：
  - 数据集划分与 ML advice（如何诊断和推进一个机器学习项目）；
  - 评估指标（evaluation metrics）；
  - 公平性（fairness）、算法偏差（algorithmic bias）、可解释性（interpretability）与隐私（privacy）。

## 贯穿全课的建模流程

### 每个模型都按同一条链来记录

面对一个新模型，最有效的读法是依次回答五个问题：任务和数据是什么？假设类 $h$ 长什么样？用什么损失或似然衡量好坏？有没有额外的约束或先验？用什么求解器把参数解出来？最后，训练好的模型泛化得怎样？本笔记记录每个模型时都遵循这条链：

$$
\text{任务与数据}
\rightarrow\text{假设类 }
h
\rightarrow\text{损失或似然}
\rightarrow\text{约束或先验}
\rightarrow\text{求解器}
\rightarrow\text{泛化诊断}
$$

以线性回归为例：任务是回归，假设类是 $h_\theta(x)=\theta^\top x$，损失是平方损失，没有约束（加 $\ell_2$ 正则时相当于高斯先验），求解器可以是梯度下降或正规方程，泛化诊断看测试误差与偏差-方差。以 SVM 为例：假设类是线性分类器，损失是 hinge 损失（或等价的最大间隔约束），求解器是对偶问题上的 SMO。

### 前半看目标与求解，后半看误差来源

课程的主线可以分成两半：

- **前半：目标与求解。** 对每个模型都要写清楚三件事：优化的对象到底是什么（哪个函数、对哪些参数），它的梯度是什么，以及它是否存在闭式解（closed-form solution）。例如线性回归有正规方程这一闭式解，逻辑回归没有闭式解，只能迭代求解。
- **后半：误差来源。** 模型表现不好时，要能区分问题出在哪里：偏差（模型太简单）、方差（模型对训练集过度敏感）、模型设定错误（假设的分布形式本身不对）、优化失败（目标函数没有真正被优化好），还是标注噪声（标签本身有错）。不同模型族有各自应重点检查的对象：间隔分类器看支持向量，概率模型看校准，聚类看是否陷入局部最优，PCA 看保留了多少方差。

### 模型之间的推广关系

把各章串起来，可以得到五条主要的“推广链”。每一条链上，后一个模型都是在前一个模型的基础上换掉或推广某个零件得到的：

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

逐条解释如下：

1. 最小二乘可以看成“高斯分布 + 线性均值”的最大似然；把高斯换成任意指数族分布，就得到 GLM；其中取伯努利分布得到逻辑回归，取多项分布得到 softmax 回归。
2. 假设每一类内部的 $x$ 服从高斯分布，就得到 GDA；假设各特征在给定类别下条件独立，就得到朴素贝叶斯。它们与逻辑回归对照，引出“生成式模型 vs 判别式模型”的比较。
3. 把输入 $x$ 映射到高维特征 $\phi(x)$ 后再做线性模型，算法里只出现内积 $\phi(x)^\top\phi(z)$，于是可以用核函数代替；SVM 的对偶形式正好只依赖内积，因此天然适合核化。
4. 有隐变量时，完全数据（含隐变量）的似然容易写，但边缘似然难以直接优化；构造它的下界 ELBO，交替优化就得到 EM，因子分析和变分推断都是这一思路的变体。
5. MDP 的贝尔曼方程给出最优值函数的不动点刻画，迭代求不动点就是值迭代；当状态空间太大、无法枚举时，转而直接对策略参数求梯度，就是策略梯度。

### 训练目标下降不代表模型变好

训练时盯着的是目标函数 $J(\theta)$，但真正关心的是模型在新数据上的表现与实际代价，两者之间有几道鸿沟，需要在“分数校准”时注意：

- 训练目标 $J(\theta)$ 下降，不等于测试风险下降。训练损失可以一直下降，而测试误差先降后升，这就是过拟合。
- 准确率在类别不平衡时会掩盖少数类的错误。例如 99% 的样本是负类时，全部预测为负类也有 99% 的准确率，但少数类一个都没识别出来。
- AUC、F1、校准误差、强化学习中的回报与约束违反次数，各自对应不同的实际代价。应当根据应用里“哪种错误更贵”来选指标，而不是默认看准确率。

### 实现层面的目标

学完这门课，应当达到两个层面的实现能力：

- 能够从零实现以下算法：正规方程、逻辑回归的牛顿法、GDA、核 SVM 的对偶求解、GMM 的 EM、PCA 与值迭代。
- 使用成熟库（如 scikit-learn、PyTorch）时，能够说清楚库里各个概念的实际语义：求解器在优化什么、核缓存为何存在、EM 输出的责任度（responsibility）代表什么、策略梯度里的基线（baseline）为什么能降低方差而不引入偏差。

> [!NOTE]+ 概念链
>
> - 记号：$X\in\mathbb{R}^{n\times(d+1)}$，截距放在 $\theta_0$。
> - 线性回归：平方损失、LMS 更新、正规方程，以及高斯噪声下的 MLE 解释。
> - 逻辑回归：使用 Bernoulli 似然，梯度的形式仍然是“残差乘特征”，与线性回归同构。
> - GLM：指数族分布，加上线性设定 $\eta=\theta^{\top}x$。
> - 生成模型：先建模 $p(x\mid y)$，再由贝叶斯公式得到后验 $p(y\mid x)$。
> - 核方法：算法中只出现 $K(x,z)=\phi(x)^{\top}\phi(z)$，因此无需显式计算 $\phi$。
> - SVM：在同一个特征空间上找最大间隔的分类超平面。
> - 决策树做轴对齐（axis-aligned）切分；Boosting 把多个弱分类器加权相加。
> - 神经网络使用可微激活函数，反向传播一次性给出各层参数的梯度并同时更新。
> - 泛化理论比较真实风险 $R$ 与经验风险 $\hat R_n$；正则化与交叉验证用来控制模型容量。
> - 无标签数据：$k$-means、EM、PCA、ICA。
> - 扩散模型用 ELBO 推出去噪目标；其后是表示的适配与语言建模。
> - 强化学习没有固定的 $(x,y)$ 样本对。LQR 在线性动力学、二次代价下给出线性反馈律；PPO 是带裁剪（clipping）的策略梯度。
> - 公平性与实践部分处理分组指标、数据划分与项目流程。




<br>



## 数学预备

CS229 几乎所有推导都只用到六类数学对象：经验风险、梯度与 Hessian、最大似然、线性代数、概率与指数族、约束优化与对偶。把它们事先理顺，后面每章的推导就只是“套公式 + 化简”。下面依次说明每类对象在课程中扮演的角色，并以线性回归的平方损失作为贯穿例子：

$$
J(\theta)=\frac{1}{2}\sum_{i=1}^{n}\big(h_{\theta}(x^{(i)})-y^{(i)}\big)^{2}
$$

这里 $n$ 是训练样本数，$h_\theta(x^{(i)})$ 是模型对第 $i$ 个样本的预测，$y^{(i)}$ 是其标签，$J$ 是一个标量。

### 经验风险：把整个数据集上的损失变成一个可微目标

真正想最小化的是模型在未知分布上的期望损失（风险），但分布未知，只能用训练集上的平均（或求和）损失来代替，这就是经验风险（empirical risk）。上式 $J(\theta)$ 就是平方损失下的经验风险。

- 单个样本的残差 $h_\theta(x^{(i)})-y^{(i)}$ 只描述“在这一个点上预测偏了多少”，是局部的代价。
- 参数更新针对的是整体目标 $J$（或对数似然 $\ell$），而不是针对单个样本的残差本身。即使是随机梯度下降，每步用一个样本，也是在用该样本的梯度近似 $\nabla J$。
- CS229 的线性回归习惯在前面乘 $1/2$，而不除以 $n$。乘 $1/2$ 是为了让求导时平方项落下的 2 被抵消，$\nabla J$ 少一个常数因子；不除以 $n$ 则让正规方程里少一次缩放。这两个常数都不影响最优解的位置，只影响梯度的大小（因而影响合适的学习率）。

### 梯度与 Hessian：一阶告诉方向，二阶告诉曲率

- **梯度** $\nabla_{\theta}J$ 是由各偏导 $\partial J/\partial\theta_j$ 组成的向量，指向 $J$ 上升最快的方向。梯度下降沿它的反方向走一小步：$\theta:=\theta-\alpha\nabla_\theta J$，其中 $\alpha>0$ 是学习率。
- **Hessian** 是二阶偏导组成的矩阵，元素为 $H_{jk}=\partial^{2}J/\partial\theta_j\partial\theta_k$，形状是 $(d+1)\times(d+1)$。它描述目标函数在各方向上的弯曲程度（曲率）：曲率大的方向应当走小步，曲率小的方向可以走大步。
- **牛顿法**用 $H^{-1}$ 来校正步长和方向：$\theta:=\theta-H^{-1}\nabla_\theta J$。对二次函数，牛顿法一步就能到达最优点。原因是：若 $J(\theta)=\frac12\theta^\top A\theta-b^\top\theta$（$A$ 对称正定），则 $\nabla J=A\theta-b$，$H=A$，于是从任意 $\theta$ 出发，

$$
\theta-H^{-1}\nabla J=\theta-A^{-1}(A\theta-b)=A^{-1}b,
$$

  这正是 $\nabla J=0$ 的解。对一般函数，牛顿法相当于每步用二次近似代替原函数，再一步跳到该近似的最小点。代价是每步要构造并求解一个 $(d+1)\times(d+1)$ 的线性系统，因此只适合维数不太大的问题。讲义指出，牛顿法用于最大化逻辑回归对数似然时也称 Fisher scoring。

### 最大似然：从概率模型推出损失函数

很多损失函数并不是“拍脑袋”定的，而是从一个概率模型出发用最大似然（maximum likelihood）推出来的。流程固定为三步：

1. 先写出给定输入时标签的条件分布 $p(y\mid x;\theta)$。
2. 假设样本独立，写出似然函数 $L(\theta)=\prod_i p(y^{(i)}\mid x^{(i)};\theta)$。
3. 取对数把乘积变成求和：$\ell(\theta)=\log L(\theta)=\sum_i\log p(y^{(i)}\mid x^{(i)};\theta)$。对数是单调函数，不改变最大值位置；求和比连乘更容易求导，也避免了很多小概率相乘导致的数值下溢。

一个关键结论是：若假设 $y^{(i)}=\theta^\top x^{(i)}+\epsilon^{(i)}$，噪声 $\epsilon^{(i)}\sim\mathcal N(0,\sigma^2)$ 独立同分布，则

$$
\ell(\theta)=n\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\cdot\frac12\sum_{i=1}^n\big(y^{(i)}-\theta^\top x^{(i)}\big)^2 .
$$

第一项与 $\theta$ 无关，第二项正是 $-J(\theta)/\sigma^2$。所以在高斯噪声下，最大化 $\ell$ 等价于最小化平方损失，而且结果与 $\sigma^2$ 的取值无关。线性回归一章会给出完整推导。

### 线性代数：设计矩阵与正规方程

- **设计矩阵**（design matrix）$X\in\mathbb{R}^{n\times(d+1)}$ 把所有样本按行堆起来，每一行是一个样本（含截距位）。这样整批预测就是一次矩阵-向量乘法 $X\theta$。
- **正规方程** $X^{\top}X\theta=X^{\top}y$ 是令 $\nabla J=0$ 得到的线性方程组（也称法方程）。它不是“把 $X$ 除掉”：$X$ 一般不是方阵，没有逆，不能直接写 $\theta=X^{-1}y$；正确的做法是求解这个 $(d+1)\times(d+1)$ 的方程组。
- **对称半正定矩阵**（symmetric positive semidefinite, PSD）的特征值全部非负，这是凸二次函数的来源。$X^\top X$ 对任意 $v$ 满足 $v^\top X^\top Xv=\|Xv\|^2\ge 0$，因此是 PSD 的；而平方损失的 Hessian 恰好是 $X^\top X$，所以 $J$ 是凸函数，任何驻点都是全局最小点。

### 概率与指数族：统一常见分布

- 伯努利、高斯、泊松等常见分布都可以写成指数族（exponential family）的统一形式 $p(y;\eta)=b(y)\exp(\eta^{\top}T(y)-a(\eta))$。其中 $\eta$ 是自然参数，$T(y)$ 是充分统计量（多数情况下 $T(y)=y$），$a(\eta)$ 是对数配分函数，负责归一化，$b(y)$ 是只与 $y$ 有关的底测度。
- 自然参数（natural parameter）$\eta$ 与分布均值之间的关系决定了 GLM 的响应函数：伯努利分布下均值 $\phi=1/(1+e^{-\eta})$，即 sigmoid；高斯分布下均值等于 $\eta$，即恒等函数；泊松分布下均值 $\lambda=e^\eta$，即指数函数。这解释了逻辑回归为什么用 sigmoid——它不是随意选择，而是伯努利分布写成指数族后自然得到的。

以伯努利分布为例，$p(y;\phi)=\phi^y(1-\phi)^{1-y}=\exp\big(y\log\frac{\phi}{1-\phi}+\log(1-\phi)\big)$，对照可得 $\eta=\log\frac{\phi}{1-\phi}$，反解即 $\phi=1/(1+e^{-\eta})$。

### 约束优化与对偶：SVM 的数学基础

- 原问题加上不等式约束 $g_i(w)\le 0$ 后，可以构造 Lagrangian，为每个约束引入一个乘子 $\alpha_i\ge 0$，把约束“搬进”目标函数。对偶（duality）问题是关于这些乘子的优化问题，在满足一定条件时与原问题的最优值相等。
- KKT 条件中的互补松弛条件 $\alpha_i g_i(w)=0$ 说明：要么约束取等号（点恰好贴在间隔边界上），要么乘子为零。于是只有贴在间隔边界上的点乘子非零，它们就是支持向量（support vectors），最终的分类器只由它们决定。

<br>

## 计算复杂度与内存估算

选择算法之前，应当先问清楚：数据规模中哪一个维度在增长——样本数 $n$、特征维数 $d$，还是分量数 $k$、状态数 $|S|$？同一个任务，不同算法对各维度的依赖差别很大。例如样本很多、特征不多时，正规方程很便宜；特征维数极高时，它的 $O(d^3)$ 就成了瓶颈，迭代方法更合适。

### 时间复杂度

| 算法 | 主要开销 | 说明 |
| :--- | :--- | :--- |
| 正规方程 | 形成 $X^{\top}X$ 为 $O(nd^{2})$，分解为 $O(d^{3})$ | $n\gg d$ 时前者占主导 |
| 批量梯度下降 | 每步 $O(nd)$ | 每步遍历全部样本 |
| 随机梯度下降 | 每步 $O(d)$ | 每步只看一个样本，步数更多但单步便宜 |
| 逻辑回归牛顿法 | 每步形成 Hessian $O(nd^{2})$，再求逆 $O(d^{3})$ | 收敛步数少，但单步贵 |
| 核方法 | 核矩阵 $K\in\mathbb{R}^{n\times n}$，存储 $O(n^{2})$，许多求解器至少 $O(n^{2})$ | 复杂度随样本数平方增长 |
| GMM-EM | $k$ 个分量，每轮 $O(nkd^{2})$（若协方差为完整矩阵） | 对角协方差可降到 $O(nkd)$ |
| PCA | 对 $d\times d$ 协方差矩阵做特征分解 $O(d^{3})$，或对 $X$ 做 SVD | 只要前几个主成分时可用截断方法 |
| 值迭代 | 状态数 $\lvert S\rvert$、动作数 $\lvert A\rvert$，每轮 $O(\lvert S\rvert^{2}\lvert A\rvert)$ | 每个状态、每个动作都要对所有后继状态求和 |

几点补充说明：

- 正规方程的 $O(nd^2)$ 来自 $X^\top X$ 的 $(d+1)^2$ 个元素各需要 $n$ 次乘加；$O(d^3)$ 来自对这个 $(d+1)\times(d+1)$ 矩阵做 Cholesky 或 LU 分解。
- 值迭代的 $O(|S|^2|A|)$ 来自贝尔曼更新 $V(s):=R(s)+\max_a\gamma\sum_{s'}P_{sa}(s')V(s')$：$|S|$ 个状态 × $|A|$ 个动作 × $|S|$ 个后继状态。若转移是稀疏的（每个状态只有少数可能的后继），实际代价会小得多。

### 空间复杂度：参数模型与非参数模型的区别

- **参数模型**（如线性回归、逻辑回归）训练完只需存下 $\theta$，预测时不必再访问训练集，内存与 $n$ 无关。
- **非参数模型**（如局部加权回归、核 SVM）预测时仍然要访问训练点或支持向量。局部加权回归每次预测都要用全部训练数据重新拟合；核 SVM 要对所有支持向量计算核函数。
- **核矩阵的内存**：$n=10^{5}$ 时，核矩阵有 $10^{10}$ 个元素，按双精度每个 8 字节计算约为 $8\times10^{10}$ 字节，即约 $80$ GB，普通机器根本放不下。因此大规模核方法必须使用稀疏近似，或像 SMO 那样每次只处理少数几个乘子、按需计算核矩阵的部分列（这也是核缓存存在的原因）。

<br>

## 数值稳定性

数学上等价的公式，在浮点数下可能结果完全不同。双精度浮点数能表示的最大值约为 $1.8\times10^{308}$，`np.exp(710)` 就会溢出为 `inf`；而 `np.exp(-750)` 会下溢为 0，再取对数得到 `-inf`。机器学习里大量出现指数、对数和接近奇异的矩阵，下面这些规则在后面的实现中会反复用到。

- **不要显式求逆。** $X^{\top}X$ 接近奇异（例如特征之间高度共线）时，不要写 `inv(X.T @ X)`；应改用 `np.linalg.lstsq`（基于 QR 或 SVD 的最小二乘求解），或加上 $\lambda I$ 做岭回归式的正则，使矩阵良态。即使矩阵不奇异，`np.linalg.solve` 也比先求逆再相乘更快、更准。
- **在 logits 上计算，而不是先 `exp`。** 概率与指数应当在未归一化的 logits（对数尺度的分数）上计算，而不是先 `exp` 再求和。所有“先指数、再求和、再取对数”的运算都应写成 log-sum-exp：

$$
\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m},\qquad m=\max_j z_j .
$$

  减去最大值后，最大的指数项是 $e^0=1$，不会上溢；求和结果至少为 1，取对数也不会出现 $\log 0$。
- **softmax 要减去每行最大值。** 原因同上：$\mathrm{softmax}(z)_j=e^{z_j}/\sum_k e^{z_k}$ 对所有 $z_j$ 同时减去一个常数不变，减去每行（每个样本）的最大值后就不会溢出。
- **sigmoid 在大正负输入上要用稳定形式。** $g(z)=1/(1+e^{-z})$ 在 $z$ 很负时 $e^{-z}$ 会溢出。稳定写法是分情况：$z\ge 0$ 时用 $1/(1+e^{-z})$，$z<0$ 时用 $e^{z}/(1+e^{z})$，两种情况里指数的参数都不为正。计算对数概率时，$\log g(z)=-\log(1+e^{-z})$ 可直接写成 `-np.logaddexp(0, -z)`（注意前面的负号），在 $z=-800$ 时得到 $-800$ 而不是 `-inf`。
- **截断 $\log 0$，但不要用截断掩盖模型错误。** 对数似然中出现 $\log 0$ 时必须用一个小量 $\varepsilon$ 截断（例如把概率限制在 $[\varepsilon,1-\varepsilon]$），否则会得到 `-inf` 或 `nan`。但截断只是数值保护，不能用来代替正确的 Bernoulli 模型：如果模型输出的“概率”本身不在 $(0,1)$ 内（例如直接用线性回归的输出当概率），问题出在模型设定上，截断只会掩盖它。
- **EM 的责任度要在 log-space 计算。** EM 的 E 步要计算每个样本属于各分量的后验概率（责任度）。当各分量几乎不重叠、样本离某些分量很远时，高斯密度值会极小，直接相乘相除会下溢为 0，出现 $0/0$。应当先在对数空间计算 $\log\phi_j+\log\mathcal N(x;\mu_j,\Sigma_j)$，再用 log-sum-exp 归一化。
- **比较序列似然前先说明归一化方式。** 比较不同长度序列的 log-likelihood 前，需要说明是否按 token 数归一化。序列越长，总对数似然越负，不归一化时长序列天然吃亏；语言模型通常报告每 token 平均负对数似然或困惑度（perplexity）。

<br>

## Python 实现基础

课程作业用 Python 和 NumPy 完成。这一节不教语法，只总结实现机器学习算法时最容易出错的几个环节：数组的角色、索引规则、随机性与广播。

### NumPy 在课程中的角色

- NumPy 用来验证算法：先在小数组上把 shape 和梯度对齐（例如用数值差分检查解析梯度），确认无误后再交给求解器或深度学习框架。
- 数值内核优先使用广播（broadcasting）和线性代数运算，避免在样本维度上写 Python `for` 循环。
- 向量化不改变算法的渐近复杂度（$O(nd)$ 还是 $O(nd)$），只是把循环从解释执行的 Python 层移到底层编译好的 BLAS 库里执行，常数因子可以小几十到上百倍。

### 索引：整数索引会消去维度

- 整数索引会消去一个维度；切片通常保留该维度。
- 因此 `X[0]` 的 shape 是 `(d+1,)`（一维数组），而 `X[0:1]` 的 shape 是 `(1, d+1)`（仍是二维的一行）。
- 把前者与 `(n, d+1)` 的数组相加时不会报错，而是触发广播，把这一行加到全部 $n$ 行上。如果本意不是如此，这类错误就会悄悄混过去，因此要时刻清楚每个数组到底是几维。

### 数值类型与随机性

- Python 3 中 `/` 总是浮点除法（`7 / 2` 得 `3.5`），整除要用 `//`。
- 随机性来自 `numpy.random`（如随机初始化、打乱样本、划分数据集）。
- 要复现实验必须固定随机种子（seed），例如 `rng = np.random.default_rng(0)`，之后所有随机操作都通过 `rng` 进行。

### 广播规则

- 广播从尾维（最右边的维度）开始逐维对齐。某一维可以扩展的条件是：两边维长相等，或其中一方为 $1$，或该维在一方中不存在（自动视为 1）。
- 逐样本加权时，权重与残差的 shape 必须一致：`(n,)` 权重与 `(n,)` 残差相乘，或 `(n, 1)` 与 `(n, 1)` 相乘，得到逐样本加权的结果。

需要注意：原笔记写的是“`(n, 1)` 权重与 `(n,)` 残差相乘得到逐样本加权”，这是错的。按上面的规则，`(n,)` 先补成 `(1, n)`，再与 `(n, 1)` 两维都扩展，结果是 `(n, n)` 的矩阵，而不是长度为 $n$ 的加权残差。这正是最常见、也最隐蔽的广播错误（不会报错），此处已改正。

```python
import numpy as np
n = 4
w = np.ones((n, 1))
r = np.arange(n, dtype=float)
(w * r).shape            # (4, 4)：意外广播成矩阵
(w[:, 0] * r).shape      # (4,)：逐样本加权的正确写法
```

下面两个辅助函数在后续各章会反复使用：一个给原始特征矩阵加上截距列，另一个检查设计矩阵是否为二维且首列全为 1。

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

### 计算图：前向缓存、反向只用局部导数

即使在线性模型里，把计算写成计算图也有好处：它把“求整体梯度”分解成“每个节点只算自己的局部导数”，这正是后面神经网络一章中反向传播的思想。

- 节点依次为：线性部分 $z=\theta^{\top}x$，激活 $h=g(z)$，损失 $\mathcal{L}(h,y)$。
- 前向传播沿拓扑序逐个求值，并缓存中间结果 $x,z,h$，供反向时使用。
- 反向传播从 $\mathrm{d}\mathcal{L}$ 出发，按链式法则逆序传播；每个节点只需要用到自己的局部导数，例如 $\partial z/\partial\theta=x$，$\partial h/\partial z=g'(z)$。

以逻辑回归为例，$g$ 为 sigmoid、$\mathcal L$ 为交叉熵时，$\partial\mathcal L/\partial h\cdot g'(z)$ 化简为 $h-y$，于是 $\partial\mathcal L/\partial\theta=(h-y)x$——这就是“残差乘特征”的形式。



<br>



## 记号约定

记号看似琐碎，却直接决定能否把推导与代码逐行对上：如果样本是按行还是按列排、截距放在哪里没有统一，正规方程和 SVM 对偶公式就会差一个转置，根本无法核对。这一节固定 CS229 讲义中的上下标约定，后面所有章节都沿用。

### 监督学习问题的基本设定

监督学习（supervised learning）要做的事情是把输入 $x$ 映射到标签 $y$。完成这个映射的函数叫假设（hypothesis）$h_{\theta}:\mathcal{X}\rightarrow\mathcal{Y}$。它可以由有限个参数 $\theta$ 描述（参数模型，如线性回归），也可以由训练集本身描述（非参数模型，non-parametric，如局部加权回归、$k$ 近邻）。

整个流程的输入是原始特征，输出是预测 $h_{\theta}(x)$，训练时优化的是一个标量目标：要最小化的损失 $J$，或要最大化的对数似然 $\ell$。本节固定样本、特征与参数的 shape；下一章的线性回归就是这套记号下最简单的可训练模型。

### 样本、特征与参数

上标圆括号表示第几个样本，不是幂次：$x^{(i)}$ 是第 $i$ 个训练样本的输入，与 $x$ 的 $i$ 次方无关。下标表示特征分量，$x_j^{(i)}$ 是第 $i$ 个样本的第 $j$ 个特征。

$$
x^{(i)}\in\mathbb{R}^{d+1},\qquad
y^{(i)}\in\mathbb{R},\qquad
\theta\in\mathbb{R}^{d+1}.
$$

各符号含义如下：

- $n$：训练样本数。
- $d$：输入特征的维数，不含截距。
- $x_0^{(i)}=1$：约定每个样本的第 0 个分量恒为 1，作为截距位。这样截距 $\theta_0$ 就被并进了参数向量，预测可以统一写成内积 $h_{\theta}(x)=\theta^{\top}x=\sum_{j=0}^{d}\theta_j x_j$，不必单独写偏置项。这也是 $x^{(i)}$ 与 $\theta$ 都是 $d+1$ 维的原因。
- $X\in\mathbb{R}^{n\times(d+1)}$：设计矩阵，第 $i$ 行是 $(x^{(i)})^{\top}$。注意 CS229 是**按行**放样本的，与 CS230 等深度学习课程“每一列是一个样本”的约定相反。
- $\vec{y}\in\mathbb{R}^{n}$：把所有标签排成的列向量。
- $h_{\theta}$：假设（hypothesis）。这是历史沿用下来的名词，含义就是预测函数。
- $\mathcal{X},\mathcal{Y}$：输入空间与输出空间。
- 回归问题中 $\mathcal{Y}=\mathbb{R}$。分类问题中 $\mathcal{Y}$ 是有限集合，二分类通常取 $\{0,1\}$；但 SVM 一章改用 $\{-1,+1\}$，因为这样可以把“分类正确”统一写成 $y\,(w^\top x+b)>0$，间隔的公式更简洁。

### 单样本预测与整批预测

对单个样本，线性预测是一个内积：

$$
h_{\theta}(x^{(i)})=\theta^{\top}x^{(i)}.
$$

把 $n$ 个样本按行堆成 $X$ 之后，$X\theta$ 的第 $i$ 个分量正是 $(x^{(i)})^\top\theta=h_\theta(x^{(i)})$。因此所有样本的残差可以一次算出：

$$
X\theta-\vec{y}
=
\begin{pmatrix}
h_{\theta}(x^{(1)})-y^{(1)} \\
\vdots \\
h_{\theta}(x^{(n)})-y^{(n)}
\end{pmatrix}.
$$

有了这个向量，平方损失就可以写成 $J(\theta)=\frac12(X\theta-\vec y)^\top(X\theta-\vec y)$，这是线性回归一章推导正规方程的出发点。

实现时有两条习惯：

- 在实现代码的注释中写明每个数组的 shape。
- 若把 $X$ 误写成 $(d+1,n)$，后续所有公式都会差一个转置，而且很多情况下因为广播或方阵而不报错，极难排查。

### 形状对照表（以线性回归为例）

| 量 | shape |
| :--- | :--- |
| $X$ | $(n, d+1)$ |
| $\theta$ | $(d+1,)$ 或 $(d+1, 1)$ |
| $X\theta$ | $(n,)$ 或 $(n, 1)$ |
| $\vec{y}$ | 与 $X\theta$ 相同 |
| $\nabla_{\theta}J$ | 与 $\theta$ 相同 |
| $X^{\top}X$ | $(d+1, d+1)$ |

表中“$\vec y$ 与 $X\theta$ 相同”非常关键：如果 $\theta$ 是 `(d+1,)` 而 $\vec y$ 是 `(n, 1)`，那么 `X @ theta - y` 会按上一节的广播规则变成 `(n, n)`，正是前面提到的那类隐蔽错误。

### 梯度的命名与检查

- $\nabla_{\theta}J$ 在代码里通常命名为 `grad`。它是与 $\theta$ 同形的向量，而不是 $J$ 的微分形式 $\mathrm dJ$（后者是一个线性映射）。
- 实现时用 `assert grad.shape == theta.shape` 卡住转置错误，这是最便宜、最有效的检查。

### 常用矩阵求导恒等式

推导正规方程等结论时，最常用的是下面三个恒等式：

$$
\nabla_{x}(b^{\top}x)=b,\qquad
\nabla_{x}(x^{\top}Ax)=2Ax\quad(A=A^{\top}),\qquad
z^{\top}z=\sum_{i}z_{i}^{2}.
$$

- 第一个：$b^\top x=\sum_i b_ix_i$，对 $x_j$ 求偏导得 $b_j$，所以梯度就是 $b$。
- 第二个：$x^\top Ax=\sum_{i,k}A_{ik}x_ix_k$，对 $x_j$ 求偏导得 $\sum_k A_{jk}x_k+\sum_i A_{ij}x_i=(Ax)_j+(A^\top x)_j$，即一般情形下 $\nabla_x(x^\top Ax)=(A+A^\top)x$；当 $A$ 对称时化为 $2Ax$。
- 第三个：向量与自身的内积是各分量的平方和，它让平方损失之和可以写成残差向量的内积 $(X\theta-\vec y)^\top(X\theta-\vec y)$。

把 $J(\theta)=\frac12(X\theta-\vec y)^\top(X\theta-\vec y)$ 展开为 $\frac12\big(\theta^\top X^\top X\theta-2(X^\top\vec y)^\top\theta+\vec y^\top\vec y\big)$，套用前两个恒等式（取 $A=X^\top X$，它是对称的；$b=X^\top\vec y$），即得 $\nabla_\theta J=X^\top X\theta-X^\top\vec y$，令其为零就是正规方程。完整推导见线性回归一章。

> [!NOTE]+ 小结
>
> - 每个模型都按“任务与数据 → 假设类 → 损失或似然 → 约束或先验 → 求解器 → 泛化诊断”来理解。
> - 平方损失前乘 $1/2$ 不除以 $n$；高斯噪声下最大似然等价于最小二乘；$X^\top X$ 半正定保证平方损失是凸的。
> - 选算法前先看哪一个维度在增长；$n=10^5$ 的核矩阵约 $80$ GB，需要稀疏或 SMO 式分解。
> - 数值上：不显式求逆，log-sum-exp 与 softmax 减最大值，sigmoid 用稳定形式，EM 在 log-space 计算。
> - 广播从尾维对齐，`(n, 1)` 与 `(n,)` 相乘得到 `(n, n)`；用 `assert grad.shape == theta.shape` 卡住转置错误。
> - 记号：$X$ 为 $(n,d+1)$、按行放样本，$\theta$ 含截距。下一章把 $h_{\theta}(x)=\theta^{\top}x$ 配上平方损失，导出 LMS 与正规方程。

<br>

<br>

# 线性回归（Linear Regression）

线性回归要解决的是最朴素的预测问题：给定一组特征（房子的面积、卧室数……），预测一个连续的实数（房价）。它是监督学习里最简单的可训练模型，却把后面几乎所有模型都要用到的套路完整走了一遍：先写出模型 $h_{\theta}$，再写出衡量好坏的损失 $J(\theta)$，然后要么沿梯度一步步走（迭代法），要么令梯度为零直接解（闭式解），最后再从概率角度解释“为什么选这个损失”。

一句话的直觉：用一个超平面 $\theta^{\top}x$ 去贴近所有数据点，“贴近”的标准是残差平方和最小；而平方和恰好是“高斯噪声下最大似然”的另一种说法。

和前后章节的关系：上一章（总览）的“记号约定”一节固定了 $X\in\mathbb{R}^{n\times(d+1)}$、$\theta$ 含截距、预测写成 $X\theta$ 的乘法顺序；本章在这套记号下给出“损失 → 梯度 → 更新 / 闭式解”的完整模板；下一章逻辑回归只改两处——把响应 $h$ 换成 sigmoid，把平方损失换成 Bernoulli 对数似然——其余骨架原样沿用。

## 问题设定：从房价例子到线性假设

### 回归与分类

回归（regression）指标签 $y\in\mathbb{R}$ 连续取值的预测问题，例如房价，或者计数类目标的连续近似；输入 $x\in\mathbb{R}^{d+1}$，其中已经含有恒为 $1$ 的截距位 $x_0=1$。模型取线性形式 $h_{\theta}(x)=\theta^{\top}x$。本章的范围是：由平方损失导出 LMS 更新与正规方程（normal equations），并说明它在高斯噪声下就是最大似然（maximum likelihood）；输入是训练数据 $(X,\vec{y})$，输出是参数 $\theta$ 以及对新样本的预测。

讲义中的标准例子是 Portland 房价数据：每条记录是一对（居住面积，价格）。把面积记为 $x_1$，价格记为 $y$，学习的目标是得到一个映射 $h:\mathcal{X}\rightarrow\mathcal{Y}$（$\mathcal{X}$ 是输入空间，$\mathcal{Y}$ 是输出空间），使 $h(x)$ 对**没见过的新房子**仍然是好的预测——只把训练集背下来不算成功。当 $y$ 连续时称为回归；当 $y$ 只取少数几个离散值（比如“是公寓还是独栋”）时称为分类（classification）。

### 线性假设

如果再加入卧室数 $x_2$，每个样本的原始特征就变成 $x^{(i)}\in\mathbb{R}^{2}$（上标 $(i)$ 表示第 $i$ 个样本，不是幂）。再补上截距位 $x_0=1$，线性假设可以写成

$$
h_{\theta}(x)=\theta_0+\theta_1 x_1+\theta_2 x_2=\theta^{\top}x.
$$

$\theta_j$ 称为参数（parameters）或权重（weights）；$\theta_0$ 是截距，它让超平面不必经过原点。把截距并进 $x$ 的好处是所有公式都可以统一写成内积 $\theta^{\top}x=\sum_{j=0}^{d}\theta_j x_j$，不必单独处理 $\theta_0$。

特征选什么完全由建模者决定：壁炉个数、浴室数都可以放进 $x$。唯一的硬性要求是**训练与预测必须使用同一套特征定义**（同样的列顺序、同样的单位、同样的预处理），否则学到的 $\theta$ 在预测时会被套在含义不同的数上。

## 损失函数：最小二乘（least squares）

有了模型，还需要一个标量来衡量某个 $\theta$ 好不好。最常用的是残差平方和，称为最小二乘代价函数（least-squares cost function）：

$$
J(\theta)=\frac{1}{2}\sum_{i=1}^{n}\big(h_{\theta}(x^{(i)})-y^{(i)}\big)^{2}.
$$

其中 $n$ 是训练样本数，$h_{\theta}(x^{(i)})-y^{(i)}$ 是第 $i$ 个样本的残差（residual）。关于这个式子有三点需要注意：

- 前面的 $1/2$ 只是为了求导时消去平方带来的因子 $2$，让梯度式子干净一些；它不是概率里的归一化常数，也不改变最优 $\theta$ 的位置。
- $J$ 对 $\theta$ 是凸二次函数（$\theta$ 的二次型加一次项加常数，二次项矩阵 $X^{\top}X$ 半正定）。在二维参数空间里画等高线是一族同心椭圆，只有一个全局最小、没有别的局部最小，所以“找到最小值”是一个良定义且不会被困住的问题。（严格说，$X^{\top}X$ 可逆时 $J$ 严格凸、最小点唯一；$X^{\top}X$ 奇异时最小值仍是全局的，但最小点构成一个仿射子空间，见下文正规方程一节。）
- 如果各特征的尺度相差几个数量级（例如面积以平方英尺计、取值上千，卧室数只有个位数），椭圆会被拉成狭长的山谷。梯度垂直于等高线，在长谷里大多指向谷壁而不是谷底，梯度下降就会左右来回走之字形，收敛很慢。这正是后面要标准化特征的原因。

> [!EXAMPLE]+ 例1 两个样本的手工计算
>
> 取 $x^{(1)}=(1,1)$，$y^{(1)}=1$；$x^{(2)}=(1,2)$，$y^{(2)}=2$（每个 $x$ 的第一个分量是截距位 $x_0=1$）；先令 $\theta=(0,0)$。则预测 $h=(0,0)$，两个残差为 $-1,-2$，于是 $J=\frac12(1^{2}+2^{2})=2.5$。若改 $\theta=(0,1)$，则 $h=(1,2)$，与标签完全一致，$J=0$。由于 $J\ge 0$，该数据集上的最优解就是过原点、斜率为 $1$ 的直线。
>
> 这个例子小到可以“看出来”答案。真实作业里通常 $n\gg d$，数据点不会恰好落在一条直线上，不能靠观察得到 $\theta$，需要下面的系统方法。

<br>

## 梯度下降与 LMS 更新规则

### 梯度下降的一般形式

既然目标是最小化 $J(\theta)$，最直接的想法是：从某个初始 $\theta$（比如全零）出发，每次朝 $J$ 下降最快的方向——负梯度方向——走一小步。对每个坐标 $j=0,\ldots,d$ **同时**执行

$$
\theta_j \leftarrow \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta).
$$

$\alpha>0$ 是学习率（learning rate），控制每一步走多远。“同时”的意思是所有坐标都用更新前的同一个 $\theta$ 来算偏导，而不是更新完 $\theta_0$ 再用新值算 $\theta_1$。

### 单样本梯度的推导

要使用上面的规则，必须先算出偏导。为了看清结构，先只考虑一个训练样本 $(x,y)$，略去求和：

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

第一个等号是定义；第二个等号用了链式法则，外层 $\frac12 u^2$ 对 $u$ 求导得 $u$，$1/2$ 正好与 $2$ 相消；第三个等号是因为 $\theta^{\top}x-y=\sum_k\theta_k x_k-y$ 中只有第 $j$ 项含 $\theta_j$，其系数为 $x_j$。

把偏导代回梯度下降，负号与 $(h-y)$ 合并成 $(y-h)$，得到单样本更新：

$$
\theta_j \leftarrow \theta_j + \alpha\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}\_{j}.
$$

该式称为 LMS（least mean squares，最小均方）更新规则，也叫 Widrow-Hoff 规则。它的形式非常符合直觉：更新量与误差 $y-h$ 成正比。预测偏差大的样本把 $\theta$ 推得多；已经拟合得很好的样本（$y\approx h$）几乎不推动 $\theta$。

> [!NOTE]+ LMS 规则的来历与定位
>
> LMS 是最小均方意义下的随机梯度规则：每看到一个样本，就沿 $\nabla_{\theta}(h-y)^{2}/2$ 的反方向走一步。它源自 1960 年代的自适应滤波（Widrow-Hoff），并不是为闭式最小二乘专门新造的算法。与一次性分解 $X^{\top}X$ 的闭式解相比，它不要求全部数据同时在内存里，也适用于样本陆续到达（在线学习）的情形。代价是学习率 $\alpha$ 选得不好时会振荡甚至发散——这是用迭代换闭式解必须付出的代价。

<br>

> [!INFO]+ 从平方到 $x_j$ 的每一步
>
> 令 $r=h_{\theta}(x)-y=\sum_{k=0}^{d}\theta_k x_k-y$。则 $J=r^{2}/2$。复合函数求导：$\partial J/\partial\theta_j=r\cdot\partial r/\partial\theta_j$。而 $\partial r/\partial\theta_j=x_j$，因为只有第 $j$ 项含 $\theta_j$，且系数为 $x_j$。符号写成 $\theta\leftarrow\theta+\alpha(y-h)x$ 时，加号来自把 $-\partial J$ 与 $(h-y)$ 中的负号相消。实现时不要把两个负号再乘一次，否则就变成了梯度**上升**，$J$ 会越来越大。
>
> 感性理解：残差 $r$ 对 $\theta_j$ 的依赖只经过 $x_j$，故偏导多一个因子 $x_j$。特征尺度相差一个数量级时，对应坐标的梯度也会差一个数量级，面积会压过卧室数——同一个 $\alpha$ 对面积坐标可能已经太大，对卧室坐标却还太小。标准化是为了让各坐标的步长可比较，不是为了改变最小二乘的几何：标准化后的最优解经过反变换，与原始特征上的最小二乘解给出同样的预测。

<br>

### 批量梯度下降与随机梯度下降

单样本规则推广到 $n$ 个样本有两种用法，区别在于“每走一步看多少数据”。

<strong>批量梯度下降（batch gradient descent）</strong>每一步用全体样本的梯度之和：

$$
\theta \leftarrow \theta + \alpha\sum_{i=1}^{n}\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}.
$$

这里写成了向量形式（对所有 $j$ 同时成立）。由于 $J$ 是各样本平方误差之和，求和的梯度就是梯度的求和，所以这正是对原始 $J$ 做梯度下降。线性回归的 $J$ 是凸二次，只有一个全局最优、没有其他局部最优，因此只要 $\alpha$ 不太大，批量法一定收敛到全局最小。讲义在面积-价格数据上用批量梯度下降得到 $\theta_0\approx 71.27$，$\theta_1\approx 0.1345$（价格以千美元计时，即每平方英尺约 134.5 美元）；再加入卧室数后得到 $\theta_0\approx 89.60$，$\theta_1\approx 0.1392$，$\theta_2\approx -8.738$。卧室数的系数为负并不奇怪：在面积固定时，卧室更多意味着每间更小，这个系数衡量的是“控制面积后”卧室数的边际效应。

<strong>随机（增量）梯度下降（stochastic / incremental gradient descent）</strong>每次只看一个样本，看完立刻更新：

$$
\theta \leftarrow \theta + \alpha\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}.
$$

外层循环反复遍历训练集（通常每轮打乱顺序），内层对 $i=1,\ldots,n$ 依次执行上式。两者的取舍是：

- 批量法必须扫完整个训练集才能走一步；$n$ 很大（数百万条）时每一步都很昂贵。
- 随机法每看到一个点就更新，往往能更快地靠近最小点；但单样本梯度只是全梯度的有噪声估计，所以参数会在最小点附近持续振荡，未必真正“收敛”。实践中最小点附近的值通常已经是足够好的近似。
- 若需要真正的收敛，可以让 $\alpha$ 随训练缓慢衰减到 $0$，这样振荡幅度逐渐缩小，参数收敛到全局最小而不只是在它周围打转。

训练集很大时，随机梯度下降（以及折中的小批量梯度下降）通常比批量法更受欢迎。

## 正规方程（normal equations）：一步求出闭式解

梯度下降是迭代求解，需要选学习率、判断何时停止。对最小二乘这种凸二次目标，还有另一条路：直接令梯度为零，解出驻点。因为 $J$ 是凸的，驻点就是全局最小。

> [!NOTE]+ 正规方程是什么
>
> 正规方程是令 $\nabla_{\theta}J=0$ 得到的线性系统 $X^{\top}X\theta=X^{\top}\vec{y}$。名称里的 “normal” 意为“法向、正交”，来自最优解处残差与 $X$ 的列正交（normal equations，也译“法方程”），不是“普通方程”的意思。与 LMS 相比：二次凸目标有唯一驻点时，一次矩阵分解即可，不必调 $\alpha$，也不必迭代。与后面的岭回归相比：不加 $\lambda I$ 时，$X^{\top}X$ 奇异则解不唯一。当 $n$ 或 $d$ 很大、或者需要流式更新时，仍应回到迭代法。

<br>

### 矩阵导数

为了不用逐个坐标求偏导，先定义对矩阵的导数。对函数 $f:\mathbb{R}^{n\times d}\rightarrow\mathbb{R}$（输入是矩阵、输出是标量），它关于 $A$ 的梯度是一个与 $A$ 同形的矩阵，第 $(i,j)$ 元是 $f$ 对 $A_{ij}$ 的偏导：

$$
\nabla_{A}f(A)
=
\Big(\frac{\partial f}{\partial A_{ij}}\Big)_{ij}.
$$

例如 $A\in\mathbb{R}^{2\times 2}$ 且 $f=\frac32 A_{11}+5A_{12}^{2}+A_{21}A_{22}$，则逐元素求偏导（$\partial f/\partial A_{11}=3/2$，$\partial f/\partial A_{12}=10A_{12}$，$\partial f/\partial A_{21}=A_{22}$，$\partial f/\partial A_{22}=A_{21}$），排回原位置得到

$$
\nabla_{A}f
=
\begin{pmatrix}
3/2 & 10A_{12} \\
A_{22} & A_{21}
\end{pmatrix}.
$$

向量是只有一列的矩阵，所以 $\nabla_{\theta}J$ 就是与 $\theta$ 同形的偏导向量，这一点可以用 `assert grad.shape == theta.shape` 检查。

### 把 $J$ 写成矩阵形式

把训练输入按行堆成设计矩阵（design matrix）$X\in\mathbb{R}^{n\times(d+1)}$，标签堆成列向量 $\vec{y}\in\mathbb{R}^{n}$：

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

因为 $h_{\theta}(x^{(i)})=(x^{(i)})^{\top}\theta$，所以 $X\theta-\vec{y}$ 的第 $i$ 个分量恰为第 $i$ 个样本的残差。对任意向量 $z$，有 $z^{\top}z=\sum_i z_i^{2}$，取 $z=X\theta-\vec{y}$ 就得到

$$
J(\theta)=\frac12(X\theta-\vec{y})^{\top}(X\theta-\vec{y}).
$$

### 推导与解

> [!INFO]+ 正规方程推导
>
> 展开乘积。注意 $\theta^{\top}X^{\top}\vec{y}$ 与 $\vec{y}^{\top}X\theta$ 都是标量且互为转置，所以相等，可以合并成 $2(X^{\top}\vec{y})^{\top}\theta$：
>
> $$
> J(\theta)=\frac12\Big(\theta^{\top}X^{\top}X\theta-2(X^{\top}\vec{y})^{\top}\theta+\vec{y}^{\top}\vec{y}\Big).
> $$
>
> $\vec{y}^{\top}\vec{y}$ 不含 $\theta$，求导为零。对对称矩阵 $A=X^{\top}X$ 使用 $\nabla_{\theta}(\theta^{\top}A\theta)=2A\theta$，以及 $\nabla_{\theta}(b^{\top}\theta)=b$（取 $b=X^{\top}\vec{y}$）：
>
> $$
> \nabla_{\theta}J(\theta)=X^{\top}X\theta-X^{\top}\vec{y}.
> $$
>
> 令梯度为零，得到法方程（正规方程）$X^{\top}X\theta=X^{\top}\vec{y}$。若 $X^{\top}X$ 可逆，
>
> $$
> \theta=(X^{\top}X)^{-1}X^{\top}\vec{y}.
> $$
>
> $X^{\top}X$ 不可逆的常见原因有两个：独立样本数少于特征数（方程比未知数少），或特征线性相关（例如同时放入华氏与摄氏温度，两列只差一个仿射变换）。此时满足正规方程的 $\theta$ 有无穷多个，应删掉共线特征、用伪逆（取最小范数解），或加 $\lambda I$（岭回归）使矩阵可逆。
>
> 感性理解：$X\theta$ 是用 $X$ 的各列（截距、面积、卧室数）按权重 $\theta$ 去拼一个最接近 $\vec{y}$ 的向量。拼完后的残差必须与每一列都垂直，否则还能沿某一列再挪一点，使平方误差继续下降。垂直条件写成矩阵就是 $X^{\top}(\vec{y}-X\theta)=0$，整理即正规方程。$(X^{\top}X)^{-1}X^{\top}$ 是把 $\vec{y}$ 映射到投影系数的算子（再左乘 $X$ 就是投影到列空间的投影矩阵），不是“把 $X$ 除掉”。

<br>

几何图像：$X\theta$ 是 $X$ 列空间中的一点；正规方程说残差 $\vec{y}-X\theta$ 与每一列正交，即最优预测 $X\theta$ 是 $\vec{y}$ 到列空间的正交投影。用例1 验证：$X=\begin{pmatrix}1&1\\1&2\end{pmatrix}$ 可逆，$\vec{y}=(1,2)^{\top}$ 本身就在列空间里，投影就是它自己，解出 $\theta=(0,1)$，与手算一致。

实际怎么用：在数值计算中不要显式求逆 $(X^{\top}X)^{-1}$，而是调用线性方程求解或最小二乘求解（如 `np.linalg.lstsq`、QR 或 SVD 分解）；直接对 $X$ 做最小二乘比先形成 $X^{\top}X$ 更稳定，因为 $X^{\top}X$ 的条件数是 $X$ 的平方。

## 概率解释（probabilistic interpretation）：为什么是平方损失

前面把平方损失当作“自然的选择”直接拿来用。但为什么是平方而不是绝对值或四次方？这一节给出一个答案：只要假设误差服从独立同分布的高斯分布，最小二乘就恰好是参数的最大似然估计。这个视角很重要，因为下一章逻辑回归、再下一章广义线性模型都沿着“先写分布、再最大化似然”的路线推出损失函数。

### 高斯噪声模型

假设目标值与输入之间的关系为

$$
y^{(i)}=\theta^{\top}x^{(i)}+\varepsilon^{(i)},\qquad
\varepsilon^{(i)}\stackrel{\mathrm{iid}}{\sim}\mathcal{N}(0,\sigma^{2}).
$$

误差项 $\varepsilon^{(i)}$ 吸收了未建模的效应（比如没放进特征的学区、装修）以及随机噪声；iid 表示各样本的误差独立且同分布。由中心极限定理，许多小的独立因素叠加的结果常近似高斯，这是该假设的常见理由。误差的密度为

$$
p(\varepsilon)=\frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac{\varepsilon^{2}}{2\sigma^{2}}\Big),
$$

把 $\varepsilon^{(i)}=y^{(i)}-\theta^{\top}x^{(i)}$ 代入，因而

$$
p(y^{(i)}\mid x^{(i)};\theta)
=
\frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac{(y^{(i)}-\theta^{\top}x^{(i)})^{2}}{2\sigma^{2}}\Big).
$$

也就是 $y^{(i)}\mid x^{(i)};\theta\sim\mathcal{N}(\theta^{\top}x^{(i)},\sigma^{2})$：给定输入，标签围绕线性预测值做高斯波动。

记号 $p(y\mid x;\theta)$ 表示“给定 $x$、由 $\theta$ 参数化的 $y$ 的分布”。分号强调 $\theta$ 在频率学派视角下不是随机变量，只是一个未知的固定参数，故不写成 $p(y\mid x,\theta)$（那样写暗示是在对随机变量 $\theta$ 条件化）。

### 似然与对数似然

同一个式子，固定 $\theta$ 看成 $y$ 的函数就是概率密度；固定数据、看成 $\theta$ 的函数就称为似然（likelihood）。利用各样本独立，整个训练集的似然是各条密度之积：

$$
L(\theta)=p(\vec{y}\mid X;\theta)=\prod_{i=1}^{n}p(y^{(i)}\mid x^{(i)};\theta).
$$

最大似然原则是：选择使观测数据出现概率最大的 $\theta$。乘积不便求导，而 $\log$ 单调递增、不改变最大点，所以改为最大化对数似然：

$$
\ell(\theta)
=
n\log\frac{1}{\sqrt{2\pi}\sigma}
-\frac{1}{\sigma^{2}}\cdot\frac12\sum_{i=1}^{n}(y^{(i)}-\theta^{\top}x^{(i)})^{2}.
$$

第一项与 $\theta$ 无关；第二项是 $-\frac{1}{\sigma^{2}}J(\theta)$。因此最大化 $\ell$ 等价于最小化原来的 $J(\theta)$。结论是：**在高斯 IID 噪声假设下，最小二乘就是 $\theta$ 的最大似然估计。**

还要注意，$\sigma^{2}$ 不出现在最优 $\theta$ 中；即使 $\sigma^{2}$ 未知，同一个 $\theta$ 仍然最优。这个“方差不影响均值参数的估计”的事实在指数族与广义线性模型中会再次用到（在那里 $\sigma^2$ 扮演离散参数的角色）。

> [!INFO]+ 从高斯密度到平方损失
>
> 单条密度是 $p(y\mid x;\theta)\propto\exp\big(-(y-\theta^{\top}x)^{2}/(2\sigma^{2})\big)$。独立样本的联合密度是各条相乘，取对数后乘积变成求和，指数函数的指数被“拿下来”：
>
> $$
> \ell(\theta)=\mathrm{const}(\sigma)-\frac{1}{2\sigma^{2}}\sum_{i}(y^{(i)}-\theta^{\top}x^{(i)})^{2}.
> $$
>
> $\sigma$ 只出现在与 $\theta$ 无关的常数，以及整体缩放 $1/\sigma^{2}$ 里。对 $\theta$ 求最大，等价于最小化 $\sum(y-\theta^{\top}x)^{2}$，也就是原来的 $J$。
>
> 感性理解：高斯密度的指数里是 $-(y-\theta^{\top}x)^{2}/(2\sigma^{2})$。最大化对 $\theta$ 的似然，等价于最小化平方残差。$\sigma$ 只缩放目标而不移动驻点，故最优 $\theta$ 与噪声方差无关。若噪声是拉普拉斯分布（密度 $\propto\exp(-|\varepsilon|/b)$），同一推导给出绝对偏差 $\sum|y-\theta^{\top}x|$ 而不是平方。也就是说，“选哪种损失”和“假设哪种噪声分布”是同一个决定的两种说法。

<br>

> [!WARNING]+ 高斯假设的适用范围
>
> 高斯噪声足以推出“平方损失对应最大似然”，但并非必要：最小二乘本身作为一个合理的拟合准则，在其他假设下也可以被论证。噪声重尾（偶尔有极端离群值）时，平方损失会被离群点主导——一个残差为 $10$ 的点贡献 $100$，压过一百个残差为 $1$ 的点——这时绝对偏差或 Huber 损失更稳健。还要记住，似然模型设错时，MLE 仍会照样给出一个估计，但该估计不必对应真实的数据生成过程。

<br>

## 局部加权线性回归（locally weighted linear regression）

### 动机：欠拟合与过拟合之间

全局拟合一条直线，遇到弯曲的数据会欠拟合（underfitting）：无论 $\theta$ 怎么选，直线都跟不上曲线。反过来，若加入 $x^2,\ldots,x^5$ 拟合五次多项式，曲线可以穿过每个训练点，却在点与点之间剧烈摆动，即过拟合（overfitting）。特征怎么选会极大地影响结果。

局部加权线性回归（LWR）提供了第三条路：不追求一条在全域都好用的曲线，而是“在哪里预测，就只认真拟合哪里”。核心直觉是，任何光滑曲线放大到足够小的局部都近似直线，所以对每个查询点 $x$，用它附近的点拟合一条局部直线即可。具体做法是，在查询点 $x$ 处重新加权：

1. 选择 $\theta$ 以最小化 $\sum_{i}w^{(i)}(y^{(i)}-\theta^{\top}x^{(i)})^{2}$；
2. 输出 $\theta^{\top}x$。

与普通最小二乘唯一的区别是每个样本前面多了非负权重 $w^{(i)}$。常用的权重是

$$
w^{(i)}=\exp\Big(-\frac{\|x^{(i)}-x\|_{2}^{2}}{2\tau^{2}}\Big).
$$

权重依赖于查询点 $x$：当 $x^{(i)}$ 离 $x$ 很近时 $w^{(i)}\approx 1$，很远时 $w^{(i)}\approx 0$。于是靠近查询点的样本权重大，远处的样本几乎被忽略。

### 带宽 $\tau$ 与距离的选择

$\tau$ 称为带宽（bandwidth），控制“邻域”有多宽，作用类似于偏差-方差之间的旋钮：

- $\tau$ 太小时，每次只看邻近两三个点，局部拟合噪声很大，曲线呈锯齿状（过拟合）；
- $\tau$ 太大时，所有点权重都接近 $1$，退回普通最小二乘（欠拟合）。

对向量输入，还可以把欧氏距离换成马氏距离 $(x^{(i)}-x)^{\top}\Sigma^{-1}(x^{(i)}-x)$，让不同方向的邻域宽度不同（例如某个特征尺度大，就在该方向放宽）。

### 加权正规方程

权重固定后，目标仍是 $\theta$ 的凸二次函数，重复上一节的推导即可得到闭式解——加权正规方程：

$$
\theta=(X^{\top}WX)^{-1}X^{\top}W\vec{y},\qquad
W=\mathrm{diag}(w^{(1)},\ldots,w^{(n)}).
$$

可以看到，当 $W=I$ 时它退回普通正规方程。注意 $W$ 依赖查询点，所以每个新查询都要重新算一次。

### 参数方法与非参数方法

普通线性回归是参数（parametric）方法：参数 $\theta$ 的个数固定、有限，拟合完把 $\theta$ 存下来，训练集就可以丢掉。LWR 是讲义里第一个非参数（non-parametric）方法：为了做预测，必须一直保留整个训练集，表示假设所需的存储随 $n$ 线性增长，而且每次预测都要重新拟合。“非参数”并不是说没有参数，而是说假设的复杂度随数据量增长。

另外需要注意：权重公式外形像高斯密度，但 $w^{(i)}$ 只是一个确定性的权重，不是随机变量，也与“误差服从高斯”的假设无关；分母里的 $\tau$ 也不是噪声标准差。

> [!NOTE]+ 局部加权回归的定位
>
> LWR 在查询点附近重新做一次加权最小二乘，而不是先学一条全局直线再到处用。它来自核回归 / Nadaraya-Watson 一类非参数思想：远处的点对局部斜率的贡献应接近 $0$。与全局线性相比，它能跟上缓慢的弯曲；与全局高次多项式相比，它不必在全域抬高次数。代价是每个查询都要重解一次，且 $\tau$ 必须用验证集选，否则要么锯齿，要么退回普通最小二乘。

<br>

## 线性回归的完整算法与复杂度

把前面的内容串起来，一个实际可用的线性回归流程如下：

1. 构造 $X$：原始特征按列排列，左侧加一列全 $1$ 作为截距列。
2. 检查各列的尺度；必要时用**训练集**的均值和方差标准化，测试集复用同一组统计量（不能用测试集自己的均值方差，否则会泄露信息，且两套特征定义不一致）。
3. 若 $n$ 不大且 $X^{\top}X$ 良态（条件数不大）：用 `lstsq` 解正规方程。
4. 若 $n$ 很大：用批量或小批量梯度下降；训练中监控 $J$，若 $J$ 上升或振荡（说明 $\alpha$ 过大），则减小 $\alpha$。
5. 若查询点附近需要弯曲的拟合：对每个查询做一次 LWR，带宽 $\tau$ 与模型系数一样，需要用验证集选定。

各方法的时间复杂度（$n$ 为样本数，$d$ 为特征维数）：

| 方法 | 时间复杂度 | 说明 |
| :--- | :--- | :--- |
| 正规方程 | $O(nd^{2}+d^{3})$ | 形成 $X^{\top}X$ 需 $O(nd^{2})$，分解 $d\times d$ 矩阵需 $O(d^{3})$ |
| 批量梯度下降 | 每步 $O(nd)$ | 一次 $X\theta$ 与一次 $X^{\top}r$ |
| 随机梯度下降 | 每步 $O(d)$ | 只用一个样本 |
| LWR | 每个查询 $O(nd^{2}+d^{3})$ | 每个查询都要解一次加权正规方程 |

由此可以看出：$d$ 在几千以内时正规方程通常很快；$d$ 很大时 $d^3$ 项不可接受，应改用梯度法；LWR 的总代价随查询次数线性增长。

## 线性回归的 NumPy 实现

下面四个函数分别对应正规方程（可选岭正则）、批量 LMS、随机 LMS 和 LWR。约定 `X` 的形状为 `(n, d+1)`，已经含截距列。

```python
import numpy as np

def linear_regression_normal_eq(X, y, reg=0.0):
    """X: (n, d+1), y: (n,)。reg 为岭参数 λ，0 表示普通最小二乘。"""
    A = X.T @ X                       # (d+1, d+1)
    if reg:
        A = A + reg * np.eye(A.shape[0])
    theta, *_ = np.linalg.lstsq(A, X.T @ y, rcond=None)  # 解 A θ = Xᵀy，不显式求逆
    return theta

def batch_lms(X, y, lr=1e-8, num_steps=10000):
    """批量 LMS。房价等未标准化特征需要很小的 lr。"""
    n, d1 = X.shape
    theta = np.zeros(d1)
    history = []
    for _ in range(num_steps):
        residual = X @ theta - y      # (n,)，即 h - y
        history.append(0.5 * float(residual @ residual))  # 记录 J(θ)
        theta -= lr * (X.T @ residual)  # 梯度 Xᵀ(h - y)，沿负梯度走
    return theta, history

def stochastic_lms(X, y, lr=1e-8, epochs=50, rng=None):
    rng = np.random.default_rng(rng)
    n, d1 = X.shape
    theta = np.zeros(d1)
    for _ in range(epochs):
        for i in rng.permutation(n):  # 每轮打乱样本顺序
            theta += lr * (y[i] - X[i] @ theta) * X[i]  # 单样本 LMS 更新
    return theta

def locally_weighted(X, y, x_query, tau):
    """在 x_query 处做一次 LWR。X, x_query 均已含截距。"""
    diff = X - x_query
    w = np.exp(-np.sum(diff * diff, axis=1) / (2.0 * tau ** 2))  # (n,) 高斯核权重
    WX = X * w[:, None]               # 等价于 W @ X，但不构造 n×n 的对角阵
    theta, *_ = np.linalg.lstsq(X.T @ WX, X.T @ (w * y), rcond=None)  # 加权正规方程
    return float(x_query @ theta), theta
```

几点实现细节：`X * w[:, None]` 用广播把每一行乘以对应权重，避免构造 $n\times n$ 的对角矩阵 $W$；截距列的差 `diff[:, 0]` 恒为 $0$，所以把截距放进距离计算不影响权重；`batch_lms` 里用的是梯度 $X^{\top}(X\theta-\vec{y})$ 而不是逐坐标循环，与前文的批量更新公式完全等价。

> [!EXAMPLE]+ 例2 正规方程与梯度必须对齐
>
> 随机生成 $X\in\mathbb{R}^{20\times 3}$、$y=X\theta^{\ast}+\varepsilon$。由于两种方法最小化的是同一个凸二次 $J$，`lstsq` 得到的 $\theta$ 与批量 LMS 在足够步数、足够小 $\alpha$ 后应在 $10^{-4}$ 内一致。若不一致，先查 $X$ 是否漏了截距列，再查更新是否写成了 `theta += lr * X.T @ residual`（符号反了：`residual` 是 $h-y$，加号会变成梯度上升，$J$ 越来越大）。这种“闭式解对拍迭代解”的检查成本极低，值得在每次实现后做一遍。

<br>

> [!NOTE]+ 对照：参数与非参数
>
> 普通最小二乘拟合后只保留 $\theta$，训练集可以丢掉。局部加权在每个查询点重解一次 $\theta$，存储随 $n$ 增长。后面的核 SVM 也是非参数的：决策函数依赖支持向量。神经网络又回到参数方法：层宽固定后，预测不必扫描全部训练点。

<br>

> [!NOTE]+ 小结与衔接
>
> - 线性回归的模型是 $h_{\theta}(x)=\theta^{\top}x$，损失是最小二乘 $J(\theta)$，它是凸二次，只有一个全局最小。
> - 迭代求解得到 LMS（Widrow-Hoff）规则：更新量 = 学习率 × 残差 × 特征；批量法每步用全体样本，随机法每步用一个样本。
> - 闭式解是正规方程 $X^{\top}X\theta=X^{\top}\vec{y}$，几何上是 $\vec{y}$ 到 $X$ 列空间的正交投影。
> - 在高斯 IID 噪声下，最小二乘就是最大似然，且最优 $\theta$ 与 $\sigma^2$ 无关。
> - 局部加权回归是非参数方法，用带宽 $\tau$ 控制局部性。
>
> 线性回归在高斯噪声下是最大似然。下一章把 $y$ 换成 $\{0,1\}$，似然换成 Bernoulli，预测换成 $g(\theta^{\top}x)$。梯度仍写成残差乘特征，只是残差里的 $h$ 不再是 $\theta^{\top}x$。

<br>

# 逻辑回归（Logistic Regression）

很多预测问题的答案不是一个实数，而是“是 / 否”：邮件是不是垃圾邮件，肿瘤是良性还是恶性，用户会不会点击广告。这就是二分类（binary classification）：标签 $y\in\{0,1\}$（$1$ 称为正类，$0$ 称为负类），输入仍是含截距的 $x\in\mathbb{R}^{d+1}$。逻辑回归的做法是不直接输出类别，而是输出“属于正类的概率”：模型 $h_{\theta}(x)=g(\theta^{\top}x)\in(0,1)$，解释为 $P(y=1\mid x;\theta)$，需要硬判决时再按阈值切分。

一句话的直觉：先像线性回归那样算一个线性分数 $\theta^{\top}x$，再用 S 形函数把它压进 $(0,1)$ 当作概率，然后用最大似然训练。

和前后章节的关系：上一章是高斯噪声下的平方损失；本章换成 Bernoulli 分布下的对数似然，推出对数损失（交叉熵）与梯度，并引入收敛更快的牛顿法（Newton's method）；神奇的是，梯度的外形仍然是“残差乘 $x$”。输入仍为 $(X,\vec{y})$，输出是 $\theta$ 与类别预测。下一章用指数族说明，线性回归与逻辑回归这两次选择并不是两套无关的算法，而是同一个框架的两个特例。

## 从线性分数到概率：sigmoid 函数

### 为什么不能直接用线性回归做分类

最省事的想法是把 $\{0,1\}$ 标签直接当实数做线性回归，再以 $0.5$ 为界判类。这样做有两个问题：其一，$\theta^{\top}x$ 可以取任意实数，会出现 $h>1$ 或 $h<0$，无法解释成概率；其二，平方损失会惩罚“分得太对”的点——一个远离边界、明显属于正类的离群 $x$ 会产生很大的预测值和很大的平方误差，从而把整条直线拽歪，使边界附近本来分对的点被分错。

### logistic / sigmoid 函数

改用 logistic 函数（也叫 sigmoid 函数）把线性分数映射到 $(0,1)$：

$$
g(z)=\frac{1}{1+e^{-z}},\qquad
h_{\theta}(x)=g(\theta^{\top}x).
$$

$g$ 的几条性质在后面的推导里会反复用到：

- 当 $z\rightarrow+\infty$ 时 $g\rightarrow 1$；当 $z\rightarrow-\infty$ 时 $g\rightarrow 0$；$g(0)=1/2$。因此决策边界 $h=1/2$ 对应 $\theta^{\top}x=0$，是一个超平面——逻辑回归仍是线性分类器。
- 对称性 $g(z)+g(-z)=1$：把分数取负，正负类概率互换。
- 导数 $g'(z)=g(z)(1-g(z))$。推导：$g'(z)=\frac{e^{-z}}{(1+e^{-z})^{2}}=\frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}}=g(z)\big(1-g(z)\big)$。导数可以用函数值本身表示，这让链式法则特别简洁。
- 当 $|z|$ 很大时函数饱和，导数 $g(1-g)$ 接近 $0$。这一点决定了“sigmoid 不能配平方损失”（见下文）。

> [!NOTE]+ 逻辑回归的定位
>
> 逻辑回归把 $P(y=1\mid x)$ 写成 $\sigma(\theta^{\top}x)$（$\sigma$ 与 $g$ 是同一个 sigmoid 函数），用 Bernoulli 对数似然训练。它的出发点是把线性分数压进 $(0,1)$，而不是先发明 S 形曲线再找概率解释（等价地说，它假设对数几率 $\log\frac{h}{1-h}=\theta^{\top}x$ 是线性的）。与把线性回归直接套在 $\{0,1\}$ 上相比：输出不再越出区间，离群 $x$ 对直线的杠杆也小得多。与感知机相比：优化的是似然而不是仅改正错分点，因而能给出校准的概率。需要警惕的是，线性可分且无正则时 $\|\theta\|$ 仍会发散。

<br>

## 似然函数、交叉熵与梯度

### Bernoulli 模型与似然

逻辑回归的概率假设是：给定 $x$，$y$ 服从参数为 $h_{\theta}(x)$ 的 Bernoulli 分布，即

$$
P(y=1\mid x;\theta)=h_{\theta}(x),\qquad
P(y=0\mid x;\theta)=1-h_{\theta}(x).
$$

利用 $y$ 只取 $0,1$，两种情况可以合并成一个紧凑写法（$y=1$ 时第二个因子为 $1$，$y=0$ 时第一个因子为 $1$）：

$$
p(y\mid x;\theta)=h_{\theta}(x)^{y}\big(1-h_{\theta}(x)\big)^{1-y}.
$$

假设 $n$ 个训练样本独立生成，似然是各条概率之积，取对数后变成求和：

$$
\begin{aligned}
L(\theta)&=\prod_{i=1}^{n}h_{\theta}(x^{(i)})^{y^{(i)}}\big(1-h_{\theta}(x^{(i)})\big)^{1-y^{(i)}},\\
\ell(\theta)&=\sum_{i=1}^{n}\Big[y^{(i)}\log h_{\theta}(x^{(i)})+(1-y^{(i)})\log\big(1-h_{\theta}(x^{(i)})\big)\Big].
\end{aligned}
$$

最大化 $\ell$ 等价于最小化平均二元交叉熵（binary cross-entropy）$-\ell/n$，也称对数损失（log loss）。逐项看：$y=1$ 时只有 $-\log h$ 这一项起作用，模型给正类的概率 $h$ 越小惩罚越大，$h\rightarrow 0$ 时惩罚趋于无穷；$y=0$ 时只惩罚 $-\log(1-h)$。也就是说，“自信地答错”会受到极重的惩罚。

与线性回归不同，令 $\nabla_{\theta}\ell=0$ 得到的是关于 $\theta$ 的非线性方程，**没有闭式解**，只能用迭代法（梯度上升或牛顿法）求解。

### 梯度的推导

> [!INFO]+ 梯度为什么是 $(h-y)x$（以及 $(y-h)x$）
>
> 令 $z=\theta^{\top}x$，$h=g(z)$。对单个样本的对数似然 $\ell=y\log h+(1-y)\log(1-h)$，按链式法则 $\theta\rightarrow z\rightarrow h\rightarrow\ell$ 分三段求导。前两段是
>
> $$
> \frac{\partial\ell}{\partial h}=\frac{y}{h}-\frac{1-y}{1-h},\qquad
> \frac{\partial h}{\partial z}=h(1-h).
> $$
>
> 相乘，分母 $h$ 与 $1-h$ 被 $h(1-h)$ 恰好消掉：
>
> $$
> \frac{\partial\ell}{\partial z}
> =
> \Big(\frac{y}{h}-\frac{1-y}{1-h}\Big)h(1-h)
> =
> y-h.
> $$
>
> （中间一步是 $y(1-h)-(1-y)h=y-h$。）再乘 $\partial z/\partial\theta=x$，得 $\nabla_{\theta}\ell=(y-h)x$。对全部样本求和，用批量梯度上升最大化 $\ell$ 时，更新为
>
> $$
> \theta\leftarrow\theta+\alpha\sum_{i=1}^{n}\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}.
> $$
>
> 它与 LMS 外形完全相同，但 $h$ 是 sigmoid 而不是 $\theta^{\top}x$，所以这是一个不同的算法。若实现中改为最小化交叉熵 $J=-\ell/n$，则梯度反号：$\nabla_{\theta}J=\frac1n X^{\top}(h-y)$，配合梯度**下降**使用。
>
> 不要用平方损失配 sigmoid：此时 $\partial\mathcal{L}/\partial z=(h-y)h(1-h)$，预测极错时（例如 $y=1$ 而 $h\rightarrow 0$）$h(1-h)\rightarrow 0$，梯度消失，模型错得越离谱反而学得越慢。交叉熵在同一极限下 $\partial J/\partial z$ 仍约为 $h-y$，信号不消失。此外，平方损失配 sigmoid 对 $\theta$ 非凸；Bernoulli 对数似然对 $\theta$ 是凹的，因而 $J$ 凸，梯度法不会陷入局部最优。
>
> 感性理解：交叉熵的梯度方向仍是特征 $x$，幅度是概率误差 $h-y$。$h(1-h)$ 在链式法则里与分母相消，因此 $h\rightarrow 0$ 且 $y=1$ 时 $\partial J/\partial z$ 仍接近 $-1$。平方损失保留 $h(1-h)$，同一极限下梯度消失。配对损失必须和输出激活一起选，不能把回归用的平方损失直接套在 sigmoid 上。

<br>

外形相同并非巧合：线性回归（高斯）和逻辑回归（Bernoulli）都属于指数族配典范响应函数的情形，这类模型的对数似然梯度总是“（观测 − 预测均值）× 特征”。后面广义线性模型一章会给出统一解释。

## 感知机（perceptron）：硬阈值的近亲

### 从 sigmoid 到硬阈值

如果把 $g$ 换成硬阈值函数，即 $g(z)=1$（$z\ge 0$）否则 $0$，让输出只能是 $0$ 或 $1$：

$$
g(z)=\begin{cases}1, & z\ge 0,\\ 0, & z<0,\end{cases}
$$

仍令 $h_{\theta}(x)=g(\theta^{\top}x)$，再套用同一个更新 $\theta\leftarrow\theta+\alpha\big(y^{(i)}-h_{\theta}(x^{(i)})\big)x^{(i)}$，就得到感知机学习规则（perceptron learning algorithm）。由于 $y-h\in\{-1,0,1\}$：分对的样本 $y-h=0$，参数不动；把正类错分为负类时加上 $\alpha x$，把负类错分为正类时减去 $\alpha x$。所以感知机不估计概率，只推动被分错的点。

### 收敛性与历史地位

线性可分时（存在一个超平面把两类完全分开），感知机能在有限步内收敛到某个分隔超平面（Novikoff 定理给出了以间隔表示的步数上界）；数据不可分时，它会一直振荡，不会收敛。它在 1960 年代被视为生物神经元的粗略模型，是 SVM 与神经网络历史中的祖先，并且因为简单，后面学习理论部分还会拿它作分析起点；但它不是现代默认的分类器。

> [!NOTE]+ 感知机的定位
>
> 感知机在错分时把 $yx$ 加到 $\theta$ 上，分对则不动（这里的 $yx$ 写法对应标签取 $\{-1,+1\}$ 的惯例；在 $\{0,1\}$ 标签下等价于上面的 $(y-h)x$）。它来自 Rosenblatt 的线性阈值单元，更新与 LMS 外形相近，但 $h$ 是硬阈值而不是线性函数或 sigmoid。与逻辑回归相比：它不给出 $P(y=1\mid x)$，也没有似然可最大化，很难赋予其预测有意义的概率解释。与 SVM 相比：只要可分就停，不最大化间隔，最终停在哪个超平面取决于样本顺序。线性不可分时振荡，不能当默认分类器。

<br>

## 多类分类：softmax 回归

### 模型

当类别数 $k>2$ 时（例如把邮件分成垃圾、个人、工作三类），每个类别 $c$ 配一个参数向量 $\theta_c\in\mathbb{R}^{d+1}$，先算 $k$ 个线性分数（logits）$z_c=\theta_c^{\top}x$，再用 softmax 把它们变成和为 $1$ 的概率分布。这称为 softmax 回归，也叫多项逻辑回归（multinomial logistic regression）：

$$
h_{\theta}(x)_{c}
=
p(y=c\mid x;\theta)
=
\frac{\exp(\theta_c^{\top}x)}{\sum_{j=1}^{k}\exp(\theta_j^{\top}x)}.
$$

指数保证每个概率为正，除以总和保证归一化。把所有 $\theta_j$ 同时加上同一个向量，分子分母同乘一个因子，概率不变——参数存在平移不确定性。因此可以把其中一个 $\theta_c$ 固定为 $0$（例如 $\theta_k=0$）以消除这种冗余；$k=2$ 时固定 $\theta_0=0$ 就恰好退回逻辑回归。

训练仍然是最大似然：每个样本贡献其**真实类别**的对数概率，

$$
\ell(\theta)=\sum_{i=1}^{n}\log h_{\theta}(x^{(i)})_{y^{(i)}}.
$$

### 梯度与一对多的比较

对 logits 求导后，负对数似然在输出层的误差仍是 $\hat{p}-e_y$，其中 $\hat{p}$ 是模型预测的概率向量，$e_y$ 是真实类别的 one-hot 向量（第 $y$ 位为 $1$、其余为 $0$）；相应地，$\ell$ 本身的梯度是 $e_y-\hat{p}$。

另一种做多类的办法是一对多（one-vs-rest）：训练 $k$ 个独立的二分类器，每个区分“第 $c$ 类 vs 其他”。但这些分类器各自训练，输出的分数不在同一个尺度上，不可直接比较，加起来也不等于 $1$；softmax 在同一个归一化下同时比较 $k$ 类，得到的后验概率可以直接对照。

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
> 第一项来自 $z_y$（只有 $c=y$ 时对 $z_c$ 求导为 $1$），第二项来自 $\log\sum_j e^{z_j}$ 的导数正好是 softmax 本身。再乘 $\partial z_c/\partial\theta_c=x$，得到 $\nabla_{\theta_c}\ell=(e_{y,c}-\hat p_c)x$。向量写法即输出层误差 $\hat p-e_y$（对负对数似然而言）。若按“先求 $\partial\ell/\partial\hat p$ 再乘 Jacobian”的顺序：Jacobian $\partial\hat p_c/\partial z_{c'}=\hat p_c(\delta_{cc'}-\hat p_{c'})$ 与交叉熵的 $1/\hat p_y$ 相乘后，同样消成 $\hat p-e_y$。
>
> 感性理解：$\partial\ell/\partial z_c$ 是“正确类指示”减去“模型分给该类的概率”。正确类为 $1-\hat p_y$，其余类为 $-\hat p_c$，合起来即 $\hat p-e_y$（差一个符号）。二分类时 $\hat p_1=h$、$\hat p_0=1-h$，退回 $h-y$。实现时不要再乘一遍 softmax 的 Jacobian——它已经被消掉了。

<br>

## 牛顿法（Newton's method）：利用曲率的快速优化

### 动机

梯度上升对所有方向用同一个步长 $\alpha$，在狭长的山谷里会走之字形，需要成百上千步。牛顿法利用二阶导数（曲率）信息自动决定每个方向走多远，通常只需十几步就能收敛到很高的精度。这一节先从一维求根讲起，再推广到向量情形，最后套用到逻辑回归。

### 一维：切线法求根

先看标量情形：要解方程 $f(\theta)=0$。在当前点用切线近似 $f$，切线与横轴的交点作为下一个点：

$$
\theta\leftarrow\theta-\frac{f(\theta)}{f'(\theta)}.
$$

（切线方程是 $f(\theta)+f'(\theta)(t-\theta)$，令其为零解出 $t$ 即得上式。）最大化 $\ell$ 的问题可以转化为求根：最大值点满足 $\ell'(\theta)=0$。令 $f=\ell'$，得到

$$
\theta\leftarrow\theta-\frac{\ell'(\theta)}{\ell''(\theta)}.
$$

### 多维：Newton-Raphson

$\theta$ 为向量时，一阶导数变成梯度，二阶导数变成 Hessian 矩阵，除法变成乘逆矩阵，推广为 Newton-Raphson 方法：

$$
\theta\leftarrow\theta-H^{-1}\nabla_{\theta}\ell(\theta),
$$

其中 $H\in\mathbb{R}^{(d+1)\times(d+1)}$ 是 Hessian，$H_{jk}=\partial^{2}\ell/\partial\theta_j\partial\theta_k$。用于最大化逻辑回归对数似然时，该方法也叫 Fisher scoring（对逻辑回归这类典范链接的模型，观测 Hessian 与期望 Hessian 相同，两者一致）。

它的收敛是二次的（quadratic convergence）：足够靠近最优点时，每迭代一次，有效数字位数大致倍增（误差从 $0.01$ 变到约 $0.0001$）。代价是每步要形成并分解 $d\times d$ 的 $H$（严格说是 $(d+1)\times(d+1)$，复杂度分析中略去截距的 $+1$），所以 $d$ 很大时单步不如（随机）梯度便宜；但 $d$ 中等时，由于迭代次数少得多，总时间通常远小于批量梯度。

> [!NOTE]+ 牛顿法的思想
>
> 牛顿法用当前点的二阶泰勒展开代替目标函数，然后一步走到这个二次近似的驻点。它来自解非线性方程 $f(\theta)=0$ 的切线法，把 $f$ 换成 $\nabla\ell$ 即得最大化版本。与固定步长的梯度上升相比：步长被 $H^{-1}$ 按曲率缩放，椭圆形山谷里不必手调各方向的 $\alpha$。代价是每步 $O(nd^{2}+d^{3})$，且 $H$ 接近奇异时需要阻尼（加 $\lambda I$）或改回梯度法。还需注意，牛顿法本身只是找驻点，不区分极大与极小；逻辑回归的 $\ell$ 是凹的，所以找到的驻点就是全局最大。

<br>

### 逻辑回归的梯度与 Hessian

对逻辑回归，令 $h^{(i)}=g(\theta^{\top}x^{(i)})$，把 $h^{(i)}$ 排成向量 $h\in\mathbb{R}^n$，则梯度与 Hessian 的矩阵形式是

$$
\nabla_{\theta}\ell=X^{\top}(\vec{y}-h),\qquad
H=-X^{\top}RX,\qquad
R=\mathrm{diag}\big(h^{(i)}(1-h^{(i)})\big).
$$

由于 $R$ 的对角元都是正数，$X^{\top}RX$ 半正定，$H$ 半负定——这正是 $\ell$ 为凹函数的证明。若改为最小化 $J=-\ell$，则牛顿步为 $\theta\leftarrow\theta-(X^{\top}RX)^{-1}X^{\top}(h-\vec{y})$，两种写法完全等价。$R$ 的对角元在 $h\approx 0$ 或 $1$ 时接近 $0$，对应“已经非常确定”的样本对曲率贡献小。

每一步牛顿迭代其实是在解一个加权最小二乘问题（权重矩阵正是 $R$，并随 $\theta$ 更新），所以这一算法在统计学里也叫迭代重加权最小二乘（IRLS）。

> [!INFO]+ Hessian $H=-X^{\top}RX$ 从哪来
>
> 单样本 $\nabla_{\theta}\ell=(y-h)x$。再对 $\theta$ 求导：$y$ 与 $x$ 都是常数，只有 $h=g(\theta^{\top}x)$ 依赖 $\theta$，且 $g'=h(1-h)$，于是
>
> $$
> \frac{\partial h}{\partial\theta}=h(1-h)\,x,\qquad
> \nabla^{2}\_{\theta}\ell=-h(1-h)\,xx^{\top}.
> $$
>
> $n$ 个样本相加：$\sum_i -h^{(i)}(1-h^{(i)})x^{(i)}(x^{(i)})^{\top}=-X^{\top}RX$，$R$ 的对角是各点的 $h^{(i)}(1-h^{(i)})$。牛顿步用 $H^{-1}$ 去“除以曲率”：山谷窄（曲率大）的方向步子自动变小，山谷宽（曲率小）的方向步子变大。梯度下降对所有方向用同一个 $\alpha$，所以在被拉长的椭圆上走之字。
>
> 感性理解：牛顿步用当前点的二阶泰勒代替 $\ell$，并走到该二次型的驻点。逻辑回归的 $\ell$ 在最优点附近接近二次，故后几步收敛很快。$h\in\{0,1\}$ 的样本满足 $h(1-h)\approx 0$，对 Hessian 几乎无贡献；曲率主要由预测仍接近 $1/2$ 的点决定。

<br>

> [!EXAMPLE]+ 例3 一维牛顿法
>
> 令 $\ell(\theta)=-(\theta-3)^{2}$，则 $\ell'= -2(\theta-3)$，$\ell''=-2$。任取 $\theta=10$，此时 $\ell'(10)=-14$，一步 $\theta\leftarrow 10-(-14)/(-2)=3$。二次目标的二阶泰勒展开就是它本身，所以牛顿步一次到达驻点，与起点无关。逻辑回归的 $\ell$ 不是二次，但在最优点附近近似二次，因此后几步收敛很快。

<br>

## 逻辑回归的完整算法与注意事项

1. 标签必须编码为 $\{0,1\}$，不能是 $\{-1,+1\}$，除非相应地改写损失（$\{-1,+1\}$ 标签下，与 $-\ell$ 等价的待最小化损失是 $\sum_i\log\big(1+e^{-y^{(i)}\theta^{\top}x^{(i)}}\big)$）。
2. 初始化 $\theta=0$ 是合法的：逻辑回归没有隐层，不存在神经网络中“同层神经元对称、学到一样东西”的问题，且目标是凹的，只要最大值点存在（数据不线性可分），初值不影响最终结果。
3. 梯度上升：重复计算 $h=g(X\theta)$，再执行 $\theta\leftarrow\theta+\alpha X^{\top}(y-h)$。
4. 牛顿法：每轮重新计算 $R$ 与 $H$，**解线性系统**求牛顿步，而不是显式求逆 $H$（更快也更稳定）。
5. 预测：默认以 $0.5$ 为阈值；若假阳性与假阴性的代价不同，可以移动阈值（例如癌症筛查宁可降低阈值多报）。阈值不改变 $\ell$ 与训练出的 $\theta$，只改变准确率等分类指标。

还有两个常见的坑：

- 线性可分且不加正则时，把 $\theta$ 沿分隔方向放大能让每个 $h$ 更接近其标签，于是 $\|\theta\|\rightarrow\infty$，$h$ 被推到 $0/1$，似然趋向 $1$（对数似然趋向 $0$），最大似然估计不存在。必须加 $L_2$ 正则或早停。
- 特征尺度仍影响梯度法的收敛速度，需要标准化；牛顿法对特征的仿射变换（缩放、平移）不那么敏感，因为 $H^{-1}$ 吸收了曲率。

## 逻辑回归的 NumPy 实现

```python
import numpy as np

def sigmoid(z):
    # 数值稳定版：分正负两支计算，避免 exp 溢出
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))   # z≥0 时 exp(-z) ≤ 1，不会溢出
    expz = np.exp(z[~pos])                      # z<0 时 exp(z) < 1
    out[~pos] = expz / (1.0 + expz)
    return out

def logistic_loglik(X, y, theta, eps=1e-12):
    h = np.clip(sigmoid(X @ theta), eps, 1.0 - eps)  # 截断避免 log(0)
    return float(y @ np.log(h) + (1.0 - y) @ np.log(1.0 - h))

def logistic_gradient_ascent(X, y, lr=0.1, num_steps=2000):
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(num_steps):
        h = sigmoid(X @ theta)
        history.append(logistic_loglik(X, y, theta))
        theta += lr * (X.T @ (y - h))   # 梯度上升：∇ℓ = Xᵀ(y - h)
    return theta, history

def logistic_newton(X, y, num_steps=20, l2=1e-6):
    n, d1 = X.shape
    theta = np.zeros(d1)
    for _ in range(num_steps):
        h = sigmoid(X @ theta)
        r = h * (1.0 - h)                               # R 的对角元
        H = X.T @ (X * r[:, None]) + l2 * np.eye(d1)    # 即 -Hessian = XᵀRX + λI
        grad = X.T @ (y - h) - l2 * theta               # 带 L2 惩罚的 ∇ℓ
        step, *_ = np.linalg.lstsq(H, grad, rcond=None) # 解线性系统，不显式求逆
        theta += step                                   # θ - (∇²)⁻¹∇ = θ + (XᵀRX+λI)⁻¹ grad
    return theta

def softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)  # 每行减最大值，利用平移不变性防溢出
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
```

实现说明：

- `sigmoid` 对 $z\ge 0$ 用 $1/(1+e^{-z})$，对 $z<0$ 用等价的 $e^{z}/(1+e^{z})$，保证 `exp` 的参数永远不为大正数，避免溢出。
- `logistic_newton` 里的 `H` 存的是负 Hessian $X^{\top}RX$（再加 `l2` 倍单位阵），所以牛顿步写成加号；它最大化的是 $\ell(\theta)-\frac{\lambda}{2}\|\theta\|^2$，其中 $\lambda$ 即 `l2`。这个很小的 $L_2$ 项既让 `H` 在 $h$ 饱和时仍可解，也防止线性可分数据上 $\|\theta\|$ 无限增长。
- `softmax_rows` 每行减去最大 logit 再取指数，利用 softmax 的平移不变性防止 $e^{z}$ 溢出，结果与原定义完全相同。

> [!EXAMPLE]+ 例4 线性可分与不可分
>
> 二维点若正类都在 $x_1+x_2>1$ 一侧、负类都在另一侧，数据线性可分，牛顿法会把 $\|\theta\|$ 推得很大：决策边界 $\theta^{\top}x=0$ 仍是一条把两类分开的直线，其方向趋于稳定（但不一定恰好是 $x_1+x_2=1$，具体是哪条取决于样本位置），变的只是尺度——概率越来越“硬”，每个点的 $h$ 逼近 $0$ 或 $1$。若四点构成异或（例如 $(0,0),(1,1)$ 为负类，$(0,1),(1,0)$ 为正类），没有任何直线能分开它们，$h$ 无法同时接近四个标签，训练 $\ell$ 有严格小于 $0$ 的有限上界，并在有限的 $\theta$ 处取到。对这个对称例子可以直接验证：$\theta=0$ 时 $X^{\top}(\vec{y}-h)=0$，由凹性知它就是最优解，所有点 $h=1/2$，$\ell=4\log\frac12$——线性模型对异或毫无区分能力。这是后面引入特征映射、核方法与隐层的动机。

<br>

> [!NOTE]+ 对照：平方损失与交叉熵
>
> 二者的梯度都能写成“残差 × 特征”。差别在残差如何定义：回归是 $\theta^{\top}x-y$，分类是 $\sigma(\theta^{\top}x)-y$。把平方损失硬套在 sigmoid 上，会多乘 $h(1-h)$，极错时梯度消失。线性可分时交叉熵会把 $\|\theta\|$ 推向无穷，必须靠正则或早停。

<br>

> [!NOTE]+ 小结与衔接
>
> - 逻辑回归用 $h_{\theta}(x)=g(\theta^{\top}x)$ 建模 $P(y=1\mid x)$，决策边界 $\theta^{\top}x=0$ 是超平面。
> - Bernoulli 最大似然给出交叉熵损失，梯度为 $X^{\top}(\vec{y}-h)$，外形与 LMS 相同；$\ell$ 是凹的，但没有闭式解。
> - 感知机是硬阈值版本，只在错分时更新，不给概率，仅在线性可分时收敛。
> - softmax 回归把二分类推广到 $k$ 类，logit 上的梯度仍是 $\hat p-e_y$。
> - 牛顿法 $\theta\leftarrow\theta-H^{-1}\nabla\ell$ 二次收敛，逻辑回归的 Hessian 为 $-X^{\top}RX$；线性可分时需要正则。
>
> 逻辑回归给出凸的 $J$（凹的 $\ell$）与牛顿步。下一章不再把 sigmoid 当成独立发明：它是 Bernoulli 指数族的典范响应函数。高斯族收回最小二乘，泊松族给出计数回归。

<br>

<br>

# 广义线性模型（Generalized Linear Models, GLM）

前面两章各自给出了一种看起来互不相关的模型：线性回归用平方损失、预测 $\theta^{\top}x$；逻辑回归用交叉熵、预测 $\sigma(\theta^{\top}x)$。实际问题里的响应 $y$ 还有更多类型：一家店每天的到店人数是非负整数（计数），一次故障前的等待时间是正实数，网页浏览次数也是计数。如果每遇到一种响应就重新设计一套假设函数、损失和梯度，既繁琐又容易出错。

广义线性模型给出的统一答案是：先把 $y$ 的条件分布选成<strong>指数族（exponential family）</strong>中的某一员，再令该分布的**自然参数（natural parameter）** $\eta$ 线性依赖于输入 $x$，最后预测 $y$ 的条件期望。高斯、Bernoulli、泊松都可以写成同一个模板，因此最小二乘、逻辑回归、泊松回归只是“选了不同指数族”的同一个算法。

本章的任务分三步：第一，学会识别一个分布的指数族三件套 $(T,a,b)$；第二，按三条假设构造 GLM，输入是特征 $x$ 与选定的指数族，输出是 $E[y\mid x]$ 的预测；第三，给出泊松回归的实现。与上下文的关系是：上一章已经写出了 $h=\sigma(\theta^{\top}x)$，本章说明 sigmoid 其实来自 Bernoulli 分布的典范响应函数，最小二乘（OLS）则来自高斯分布；本章及之前的模型都直接对 $p(y\mid x)$ 建模，下一章将换一条路线，改为对 $p(x\mid y)$ 建模。

## 指数族：把常见分布写成同一个模板

### 定义与各部分的含义

一类分布属于指数族，若其概率质量函数或密度可以写成

$$
p(y;\eta)=b(y)\exp\big(\eta^{\top}T(y)-a(\eta)\big).
$$

式中各项的含义如下：

- $\eta$ 是自然参数（natural parameter，也称 canonical parameter），可以是标量也可以是向量。它是“控制分布形状的旋钮”。
- $T(y)$ 是充分统计量（sufficient statistic），与 $\eta$ 同维；本章遇到的分布都有 $T(y)=y$。
- $a(\eta)$ 是对数配分函数（log partition function）。因子 $e^{-a(\eta)}$ 的作用是归一化，保证 $p(y;\eta)$ 对 $y$ 积分或求和等于 $1$。
- $b(y)$ 是基测度（base measure），只依赖 $y$，不含 $\eta$。

固定 $T,a,b$ 就确定了一个分布“族”；在族内改变 $\eta$，就得到族中的不同分布。例如固定 Bernoulli 的三件套后，调节 $\eta$ 相当于调节正类概率 $\phi$。

> [!NOTE]+ 指数族与 GLM 的关系
>
> 指数族把高斯、Bernoulli、泊松等分布写成同一模板 $b(y)\exp(\eta^{\top}T(y)-a(\eta))$。GLM 在此基础上再令 $\eta=\theta^{\top}x$，并以 $E[T(y)\mid x]$ 作为预测。这一框架的来由是统一此前两章看起来不相像的算法：平方损失对应高斯分布，交叉熵对应 Bernoulli 分布。与“每种响应各写一套模型”相比，GLM 的好处是梯度一律具有 $(y-\mu)x$ 的形式，更换分布只是更换联系函数。需要注意的限制是：$\eta$ 对 $x$ 必须是线性的；若需要非线性，应改用广义可加模型或神经网络，指数族这一部分仍可保留。

<br>

### 例：Bernoulli 分布与 sigmoid 的来历

Bernoulli 分布描述取值为 $0/1$ 的随机变量，均值为 $\phi\in(0,1)$：$p(y;\phi)=\phi^{y}(1-\phi)^{1-y}$。为了和模板 $\exp(\eta T(y)-a(\eta))$ 对照，先把它写成“$\exp$ 一个对数”，再把含 $y$ 的项合并：

$$
p(y;\phi)
=
\exp\big(y\log\phi+(1-y)\log(1-\phi)\big)
=
\exp\Big(y\log\frac{\phi}{1-\phi}+\log(1-\phi)\Big).
$$

第二个等号只是把 $y\log\phi-y\log(1-\phi)$ 合并成 $y\log\frac{\phi}{1-\phi}$。逐项对照模板，读出三件套：

$$
\eta=\log\frac{\phi}{1-\phi},\qquad
\phi=\frac{1}{1+e^{-\eta}},\qquad
T(y)=y,\qquad
a(\eta)=-\log(1-\phi)=\log(1+e^{\eta}),\qquad
b(y)=1.
$$

其中 $a(\eta)=\log(1+e^{\eta})$ 来自把 $1-\phi=1/(1+e^{\eta})$ 代入 $-\log(1-\phi)$。这意味着：sigmoid 是 Bernoulli 分布中“自然参数 $\to$ 均值”的反函数，并不是额外人为指定的响应曲线。

> [!INFO]+ Bernoulli 怎样变成 sigmoid
>
> 从 $p(y)=\phi^{y}(1-\phi)^{1-y}$ 取对数：
>
> $$
> \log p=y\log\phi+(1-y)\log(1-\phi)=y\log\frac{\phi}{1-\phi}+\log(1-\phi).
> $$
>
> 与标准形 $\eta T(y)-a(\eta)$ 对照，$T(y)=y$，故 $\eta=\log(\phi/(1-\phi))$。两边取指数再解出 $\phi$：
>
> $$
> e^{\eta}=\frac{\phi}{1-\phi}\implies\phi=\frac{e^{\eta}}{1+e^{\eta}}=\frac{1}{1+e^{-\eta}}.
> $$
>
> 这就是 sigmoid。GLM 再令 $\eta=\theta^{\top}x$，于是 $h_{\theta}(x)=\sigma(\theta^{\top}x)$。
>
> 直观上看：均值 $\phi$ 只能在 $(0,1)$ 内取值，自然参数 $\eta$ 却可以取遍 $\mathbb{R}$。对数几率（log-odds）$\log(\phi/(1-\phi))$ 是两者之间的标准双射，它的反函数正是 sigmoid。线性函数 $\theta^{\top}x$ 的值域是整个实轴，正好可以放在 $\eta$ 的位置上，逻辑回归的响应函数由此确定。

<br>

### 例：方差固定的高斯分布

把 $\mathcal{N}(\mu,1)$ 写成指数族。做法是展开平方 $(y-\mu)^{2}=y^{2}-2\mu y+\mu^{2}$，把只含 $y$ 的部分拿出去作为 $b(y)$：

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

对照模板得 $\eta=\mu$，$T(y)=y$，$a(\eta)=\eta^{2}/2$，$b(y)=(2\pi)^{-1/2}\exp(-y^{2}/2)$。

这里把方差固定为 $1$ 并不失一般性：在线性回归里 $\sigma^{2}$ 不影响 $\theta$ 的最优值（它只给对数似然整体乘一个常数）。若把 $\sigma^{2}$ 也当作未知参数，则需要二维自然参数，或者使用带色散参数（dispersion parameter）$\tau$ 的更一般写法 $p(y;\eta,\tau)=b(y,\tau)\exp((\eta^{\top}T(y)-a(\eta))/c(\tau))$；对 GLM 的构造而言，固定 $\sigma^{2}$ 已经足够。

### 例：泊松分布

泊松分布刻画单位时间内事件发生的次数：$p(y;\lambda)=e^{-\lambda}\lambda^{y}/y!$，$y\in\{0,1,2,\ldots\}$，$\lambda>0$ 是均值（也是方差）。写成 $\frac{1}{y!}\exp(y\log\lambda-\lambda)$ 即可读出：$\eta=\log\lambda$，$\lambda=e^{\eta}$，$T(y)=y$，$a(\eta)=e^{\eta}$，$b(y)=1/y!$。它适合建模到店人数、网页浏览次数等计数型响应。

把三种分布并列，便于对照：

| 分布 | 自然参数 $\eta$ | 均值（用 $\eta$ 表示） | $a(\eta)$ | $b(y)$ |
| :--- | :--- | :--- | :--- | :--- |
| Bernoulli$(\phi)$ | $\log\frac{\phi}{1-\phi}$ | $\phi=1/(1+e^{-\eta})$ | $\log(1+e^{\eta})$ | $1$ |
| 高斯 $\mathcal{N}(\mu,1)$ | $\mu$ | $\mu=\eta$ | $\eta^{2}/2$ | $(2\pi)^{-1/2}e^{-y^{2}/2}$ |
| 泊松$(\lambda)$ | $\log\lambda$ | $\lambda=e^{\eta}$ | $e^{\eta}$ | $1/y!$ |

### 对数配分函数的导数：均值与方差

指数族有两个常用事实：

$$
E[T(y);\eta]=\nabla_{\eta}a(\eta),\qquad
\mathrm{Var}(T(y);\eta)=\nabla_{\eta}^{2}a(\eta).
$$

也就是说，对数配分函数的一阶导数是均值，二阶导数是方差。可以用上表验证：Bernoulli 的 $a'(\eta)=e^{\eta}/(1+e^{\eta})=\phi$；高斯的 $a'(\eta)=\eta=\mu$；泊松的 $a'(\eta)=e^{\eta}=\lambda$，且 $a''(\eta)=\lambda$，正是“泊松方差等于均值”。

这一事实的实际意义：由于方差 $a''(\eta)\ge 0$，$a$ 是凸函数，GLM 的负对数似然对 $\theta$ 是凸的；而且上一章逻辑回归牛顿法里的对角权重 $h(1-h)$ 恰好是 Bernoulli 方差，而不是另设的经验系数。

> [!INFO]+ $a'(\eta)$ 为何是均值
>
> 归一化条件是 $\int p(y;\eta)\,dy=1$，对任意 $\eta$ 都成立。把 $p=b(y)\exp(\eta T(y)-a(\eta))$ 代入，对 $\eta$ 求导（在积分号下求导；$\exp$ 的导数会带出指数里对 $\eta$ 的导数 $T(y)-a'(\eta)$）：
>
> $$
> 0=\int p(y;\eta)\big(T(y)-a'(\eta)\big)\,dy=E[T(y)]-a'(\eta).
> $$
>
> 再求导一次可得 $a''(\eta)=\mathrm{Var}(T(y))$。Bernoulli 时 $a(\eta)=\log(1+e^{\eta})$，$a'=\sigma(\eta)=\phi$，$a''=\phi(1-\phi)$。牛顿法里的 $R_{ii}=h^{(i)}(1-h^{(i)})$ 就是该样本点处的方差：模型越确定（$h$ 越接近 $0$ 或 $1$），曲率越小。
>
> 直观上看：$a(\eta)$ 的职责是保证密度积分为 $1$。$\eta$ 增大时，概率质量向 $T(y)$ 较大的一侧移动，$a'(\eta)$ 恰好记录了这次移动后的均值；$a''(\eta)$ 描述质量铺开的程度，即方差。因此牛顿法里的对角权重 $h(1-h)$ 不是另加的经验系数。

<br>

## 用三条假设构造 GLM

### 三条假设

有了指数族这个“模板库”，构造回归或分类模型只需要回答三个问题：$y$ 服从哪种分布？要预测什么？分布的参数怎样依赖于 $x$？GLM 对这三个问题各给一条假设：

1. $y\mid x;\theta\sim\mathrm{ExponentialFamily}(\eta)$：给定 $x$ 和 $\theta$，$y$ 服从某个以 $\eta$ 为自然参数的指数族分布。
2. 给定 $x$，目标是预测 $T(y)$ 的条件期望；通常 $T(y)=y$，即预测 $\mu=E[y\mid x]$。
3. 自然参数与输入呈线性关系：$\eta=\theta^{\top}x$（当 $\eta$ 是向量时写成 $\eta=\Theta x$）。

于是假设函数自动确定为

$$
h_{\theta}(x)=E[T(y)\mid x;\theta]=g(\eta)=g(\theta^{\top}x),
$$

其中 $g(\eta)=E[T(y);\eta]$ 称为典范响应函数（canonical response function），其反函数 $g^{-1}$ 称为典范联系函数（canonical link function）。由上一节，$g=a'$。

> [!INFO]+ 为何三条假设刚好够用
>
> 假设 1 指定响应所属的指数族（连续、二元或计数）。假设 3 把条件分布的自由度限制为有限维参数 $\theta$。假设 2 规定预测对象是 $E[T(y)\mid x]$，一般就是均值。若目标改为预测分位数，就不再是标准 GLM；若 $\eta$ 对 $x$ 是非线性的，则进入广义可加模型或神经网络的范围，指数族部分仍可保留。
>
> 换句话说：选定 $y$ 的指数族、规定预测均值、把自然参数限制为 $\theta^{\top}x$ 之后，联系函数就随之确定——高斯对应恒等函数，Bernoulli 对应 sigmoid，泊松对应指数函数。更换损失函数等价于更换指数族，而不必另写一套梯度公式。

<br>

### 三个实例：OLS、逻辑回归、泊松回归

**普通最小二乘（ordinary least squares）。** 选高斯族。前面已得 $\mu=\eta$，故

$$
h_{\theta}(x)=E[y\mid x;\theta]=\mu=\eta=\theta^{\top}x.
$$

这正是线性回归的假设函数；对应的负对数似然就是平方损失。

**逻辑回归（logistic regression）。** 选 Bernoulli 族。由 $\phi=1/(1+e^{-\eta})$ 且 $E[y\mid x]=\phi$，得

$$
h_{\theta}(x)=\frac{1}{1+e^{-\theta^{\top}x}}.
$$

**泊松回归（Poisson regression）。** 选泊松族。$E[y\mid x]=\lambda=e^{\eta}=\exp(\theta^{\top}x)$，所以计数的预测值自动为正。对数似然（略去与 $\theta$ 无关的 $\log y!$）为 $\ell=\sum_i(y^{(i)}\theta^{\top}x^{(i)}-e^{\theta^{\top}x^{(i)}})$。对 $\theta$ 求导时，第一项给出 $y^{(i)}x^{(i)}$，第二项由链式法则给出 $e^{\theta^{\top}x^{(i)}}x^{(i)}$，故梯度为

$$
\nabla_{\theta}\ell=\sum_{i=1}^{n}\big(y^{(i)}-e^{\theta^{\top}x^{(i)}}\big)x^{(i)}.
$$

这仍然是“残差乘特征”。这是指数族配典范联系函数时的一般规律：对任意此类 GLM，单样本对数似然为 $y\,\theta^{\top}x-a(\theta^{\top}x)+\log b(y)$，求导得 $(y-a'(\theta^{\top}x))x=(y-\mu)x$，因此 $\nabla\ell=\sum(y-\mu)x$。实际使用时，三种模型的梯度上升代码只差一行：计算 $\mu$ 时用恒等、sigmoid 还是指数。

<br>

## 泊松回归的实现

下面的代码用批量梯度上升最大化泊松对数似然，并记录每一步的对数似然值（同样略去 $\log y!$）。

```python
def poisson_regression(X, y, lr=1e-3, num_steps=2000):
    """X: (n, d+1), y: 非负计数。"""
    theta = np.zeros(X.shape[1])
    history = []
    for _ in range(num_steps):
        mu = np.exp(X @ theta)          # 预测均值 λ = exp(θᵀx)
        history.append(float(np.sum(y * (X @ theta) - mu)))   # 对数似然（略去 log y!）
        theta += lr * (X.T @ (y - mu))  # 梯度上升：残差 × 特征
    return theta, history
```

需要注意的实现细节：$\exp(\theta^{\top}x)$ 增长很快，学习率过大时 $\mu$ 会溢出，因此 `lr` 通常取得较小，并建议先对特征做标准化。

> [!EXAMPLE]+ 例5 到店人数
>
> 特征包括“是否周末”“是否促销”。泊松回归输出 $\hat\lambda=\exp(\theta^{\top}x)$，例如周末系数为正，意味着周末的期望人数按 $e^{\theta_{\text{周末}}}$ 倍放大——系数在对数尺度上可加、在原尺度上相乘。若误用线性回归，系数在原尺度上相加：周末系数很大时，为拟合周末的高值，截距或其他系数可能被压成负数，某些特征组合（如非周末、无促销）的预测就可能为负，计数的解释随之崩溃。若数据方差明显大于均值（过离散，overdispersion），泊松“方差 $=$ 均值”的假设失败，应改用负二项回归或引入色散参数。

<br>

> [!NOTE]+ 小结
>
> - 指数族模板 $b(y)\exp(\eta^{\top}T(y)-a(\eta))$ 覆盖 Bernoulli、高斯、泊松等常见分布。
> - $a'(\eta)$ 是均值，$a''(\eta)$ 是方差；后者保证了凸性，也解释了牛顿法里的 $h(1-h)$。
> - GLM 三条假设：响应服从指数族、预测条件期望、$\eta=\theta^{\top}x$。
> - 高斯 $\to$ OLS，Bernoulli $\to$ 逻辑回归，泊松 $\to$ 泊松回归；梯度统一为 $\sum(y-\mu)x$。

> [!NOTE]+ 衔接
>
> GLM 统一了“选指数族 + 令 $\eta=\theta^{\top}x$”这一套路。到此为止的模型都直接对 $p(y\mid x)$ 建模。下一章改为对 $p(x\mid y)$ 建模，再用贝叶斯法则回到后验。同一条线性边界可以由两条完全不同的估计路线得到。

<br>

# 生成学习算法（Generative Learning Algorithms）

设想要区分大象（$y=1$）和狗（$y=0$）。逻辑回归的做法是直接在特征空间里找一条分界线，即对 $p(y\mid x)$ 建模，这类方法称为**判别模型（discriminative model）**。另一种思路是：先只看大象，建立“大象长什么样”的模型 $p(x\mid y=1)$；再只看狗，建立 $p(x\mid y=0)$；同时估计类别先验 $p(y)$。来了一只新动物，就看它更像哪一类的模型。这类方法称为**生成模型（generative model）**，因为学到的 $p(x\mid y)p(y)$ 可以用来“生成”样本。

从生成模型回到分类，需要贝叶斯法则：

$$
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
$$

其中 $p(x)=\sum_{y'}p(x\mid y')p(y')$。对于决策而言 $p(x)$ 只是一个与 $y$ 无关的归一化常数，所以预测只需比较 $p(x\mid y)p(y)$ 的大小：$\arg\max_y p(y\mid x)=\arg\max_y p(x\mid y)p(y)$。

两条路线各有长处。生成模型可以采样新的 $x$，也可以在部分特征缺失时对缺失部分边缘化；它对 $x$ 的分布做了更强的假设，假设近似成立时更省数据。判别模型不对 $x$ 的分布负责，假设更少、更稳健，在生成假设不成立时通常能达到更低的渐近误差。

> [!WARNING]+ 关于“谁更省数据”的修正
>
> 原笔记此处写的是“判别模型通常在 $p(y\mid x)$ 被正确设定时渐近更省数据”，这一说法与课程讲义及 Ng & Jordan（2002）的结论不符，此处改正如下。
>
> - 课程讲义（main notes 第 4 章）：当 $p(x\mid y)$ 确实是共享协方差的高斯时，GDA 是渐近有效（asymptotically efficient）的，即训练集很大时没有算法在估计 $p(y\mid x)$ 上严格优于它；即使训练集较小，一般也预期 GDA 更好。总之，生成假设正确或近似正确时，生成模型更省数据。
> - Ng & Jordan（NIPS 2001，论文集 2002 年出版）比较朴素贝叶斯与逻辑回归：逻辑回归的渐近误差不高于朴素贝叶斯（生成假设不成立时通常更低）；朴素贝叶斯则以少得多的样本（约 $O(\log d)$ 对 $O(d)$，$d$ 为特征维数）接近其自身的渐近误差，因此小样本时生成模型常常领先，样本充足后判别模型反超。
> - 判别模型（如逻辑回归）假设更弱，对错误的建模假设不敏感，更稳健。

本章的内容是：写出高斯判别分析（GDA）与朴素贝叶斯的极大似然估计，并与逻辑回归对照；输入是带标签样本，输出是类条件密度的参数和后验。与上下文的关系是：上一章的 GLM 是判别模型，直接对 $p(y\mid x)$ 建模；本章对 $p(x\mid y)p(y)$ 建模，GDA 在共享协方差 $\Sigma$ 时决策边界仍是线性的，但参数 $\theta$ 的估计量不同；下一章回到判别路线，只是把 $x$ 换成特征映射 $\phi(x)$。

## 多元正态分布（Multivariate Gaussian）

GDA 要用高斯分布描述每一类的 $x$，而 $x\in\mathbb{R}^{d}$ 是向量，因此先回顾 $d$ 维正态分布 $\mathcal{N}(\mu,\Sigma)$。它的密度是

$$
p(x;\mu,\Sigma)
=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\Big(-\frac12(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\Big).
$$

各参数的含义如下：

- 均值向量 $\mu=E[x]\in\mathbb{R}^{d}$，决定分布的中心位置。
- 协方差矩阵 $\Sigma=E[(x-\mu)(x-\mu)^{\top}]\in\mathbb{R}^{d\times d}$ 对称半正定；要使上式密度存在（$\Sigma^{-1}$ 存在），需要 $\Sigma$ 正定。$|\Sigma|$ 表示其行列式。
- 密度的等高线是椭圆（高维时为椭球），主轴方向沿 $\Sigma$ 的特征向量，半轴长度与对应特征值的平方根成正比。
- 当 $\Sigma=\sigma^{2}I$ 时椭圆变成圆，各坐标相互独立且方差相同；非对角元素不为零时，等高线沿相关方向倾斜。

指数里的二次型 $(x-\mu)^{\top}\Sigma^{-1}(x-\mu)$ 是马氏距离（Mahalanobis distance）的平方：先按协方差结构把数据“白化”（拉成各向同性），再度量欧氏长度。因此密度只取决于 $x$ 在“考虑了相关性之后”离中心有多远。

## 高斯判别分析（Gaussian Discriminant Analysis, GDA）

### 模型

GDA 适用于输入 $x$ 是连续实值向量的二分类问题。模型假设类先验服从 Bernoulli 分布，每一类的 $x$ 服从高斯分布，且两类**共享**同一个协方差矩阵 $\Sigma$：

$$
\begin{aligned}
y &\sim \mathrm{Bernoulli}(\phi),\\
x\mid y=0 &\sim \mathcal{N}(\mu_0,\Sigma),\\
x\mid y=1 &\sim \mathcal{N}(\mu_1,\Sigma).
\end{aligned}
$$

参数为 $\phi\in(0,1)$、$\mu_0,\mu_1\in\mathbb{R}^{d}$、$\Sigma\in\mathbb{R}^{d\times d}$。几何上，可以把它想象成两团形状、朝向相同的椭圆云，中心分别在 $\mu_0$ 和 $\mu_1$。

> [!NOTE]+ GDA 的定位
>
> GDA 把类条件分布取为共享协方差的高斯、类先验取为 Bernoulli，再用贝叶斯法则得到 $p(y\mid x)$。它体现的是生成路线：先写 $p(x\mid y)p(y)$，而不是直接写 $p(y\mid x)$。与逻辑回归相比：在小样本且数据近似高斯时，GDA 的估计更省数据，还能采样 $x$；假设错误（如离散词袋特征、重尾分布）时偏差更大。共享 $\Sigma$ 使二次项相互抵消，决策面仍是线性的，这是它能与逻辑回归直接比较的前提。

<br>

### 极大似然估计

生成模型最大化的是**联合**对数似然

$$
\ell(\phi,\mu_0,\mu_1,\Sigma)=\log\prod_{i=1}^{n}p(x^{(i)}\mid y^{(i)};\mu_0,\mu_1,\Sigma)\,p(y^{(i)};\phi),
$$

而不是逻辑回归那样的条件似然 $\prod_i p(y^{(i)}\mid x^{(i)})$。对 $n$ 个独立同分布样本求最大（分别对各参数求导置零），得到闭式解：

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

这些式子都有直白的解释：$\phi$ 是训练集中正类所占的比例；$\mu_c$ 是第 $c$ 类样本的平均值；$\Sigma$ 是每个样本相对于**其所属类中心**的偏差外积的平均，即两类的合并协方差（pooled covariance）。之所以能分开求解，是因为联合似然的对数分解成 $\sum_i\log p(y^{(i)})$ 与 $\sum_i\log p(x^{(i)}\mid y^{(i)})$ 两部分，前者只含 $\phi$。与逻辑回归需要迭代优化不同，GDA 的训练只需要一次遍历统计量。

预测时比较 $p(y=1\mid x)$ 与 $1/2$，或者等价地比较两类的对数后验 $\log p(x\mid y)+\log p(y)$。

### 共享协方差导致线性决策边界

> [!INFO]+ 共享协方差为何给出线性边界
>
> 把两类的对数高斯密度相减。展开 $(x-\mu)^{\top}\Sigma^{-1}(x-\mu)=x^{\top}\Sigma^{-1}x-2\mu^{\top}\Sigma^{-1}x+\mu^{\top}\Sigma^{-1}\mu$，归一化常数 $(2\pi)^{d/2}|\Sigma|^{1/2}$ 两类相同。$\log p(x\mid y=1)-\log p(x\mid y=0)$ 中，二次项 $-x^{\top}\Sigma^{-1}x/2$ 两类相同，相减后消失。再加上先验的对数比，对数后验几率只剩下
>
> $$
> (\mu_1-\mu_0)^{\top}\Sigma^{-1}x
> -\frac12\mu_1^{\top}\Sigma^{-1}\mu_1
> +\frac12\mu_0^{\top}\Sigma^{-1}\mu_0
> +\log\frac{\phi}{1-\phi}.
> $$
>
> 这是 $x$ 的线性函数（仿射函数）。因此 $p(y=1\mid x)$ 仍具有 sigmoid 形式 $\sigma(\theta^{\top}x)$，其中 $\theta$ 的非截距部分是 $\Sigma^{-1}(\mu_1-\mu_0)$，截距由其余常数项构成——只是 $\theta$ 由类高斯参数算出来，而不是直接最大化判别似然得到的。
>
> 若两类各有自己的协方差 $\Sigma_0,\Sigma_1$（二次判别分析，QDA），二次项不再相消，决策边界是二次曲面。数据少时共享 $\Sigma$ 的估计方差更小；两类形状差别很大时 QDA 的偏差更小。

<br>

### GDA 与逻辑回归的对照

上面的推导说明两种方法有密切联系，但不等价：

- 在共享 $\Sigma$ 时，二者都给出 $p(y=1\mid x)=\sigma(\theta^{\top}x)$ 这一模型族。
- 两者的蕴含关系是单向的：若 $x\mid y$ 是共享协方差的高斯，则 $p(y\mid x)$ 一定是 logistic 形式；反之不成立——例如 $x\mid y$ 为泊松分布时 $p(y\mid x)$ 也是 logistic。所以 GDA 对 $x\mid y$ 做了更强的假设。假设近似正确时，GDA 更数据高效，小样本上往往表现更好。
- 当 $x$ 明显非高斯（例如离散的指示变量）时，逻辑回归更稳健。
- GDA 是生成模型，逻辑回归是判别模型：即使 $p(x\mid y)$ 设定错误，逻辑回归仍可能学到良好的 $p(y\mid x)$。

## 朴素贝叶斯（Naive Bayes）

### 动机：高维离散特征无法直接建模

GDA 要求 $x$ 是连续向量。在垃圾邮件过滤这样的文本分类问题里，常用的表示是：取一个大小为 $V$（如 50000）的词表，$x\in\{0,1\}^{V}$，第 $j$ 维表示词 $j$ 是否在邮件中出现。若直接用多项分布对 $x$ 的全部 $2^{V}$ 种取值建模，参数个数约为 $2^{V}$，根本无法估计。

朴素贝叶斯的解决办法是做一条很强的假设：**给定类别 $y$ 后，各特征条件独立**，

$$
p(x\mid y)=\prod_{j=1}^{d}p(x_j\mid y).
$$

式中 $d$ 为特征维数，在上述词表表示中 $d=V$。这条假设几乎从不严格成立（例如在垃圾邮件中，“cheap”和“viagra”经常一起出现，并不独立），但作为一个偏置很强的生成模型，它在小样本上往往很好用。

> [!NOTE]+ 朴素贝叶斯的定位
>
> 朴素贝叶斯在 $p(x\mid y)=\prod_j p(x_j\mid y)$ 的假设下做生成式分类。名称中的 naive 指的就是这条条件独立假设，并非对方法本身的贬义。它的来由是：在高维离散 $x$ 上，完整的联合分布 $p(x\mid y)$ 无法估计——词表大小为 $V$ 时，没有独立假设就要估计 $2^{V}$ 项。与 GDA 相比，它不需要连续高斯，参数个数随 $V$ 线性增长。与逻辑回归相比，它同样给出关于 $x$ 线性的对数后验，但估计量不同，而且零计数必须平滑。

<br>

### Bernoulli 事件模型与极大似然估计

在 Bernoulli 事件模型（Bernoulli event model）中，$x_j\in\{0,1\}$ 表示词 $j$ 是否出现。可以把生成过程想象成：先按先验决定这封邮件是不是垃圾邮件，再对词表中的每个词独立地抛一枚硬币，决定它是否出现。参数为

$$
\phi_{j\mid y=c}=p(x_j=1\mid y=c),\qquad
\phi_y=p(y=1).
$$

对联合似然求最大，得到的 MLE 是计数比例：$\phi_{j\mid y=c}$ 等于类 $c$ 中含有词 $j$ 的文档占类 $c$ 文档的比例，$\phi_y$ 是正类文档的比例。预测时代入贝叶斯法则，比较 $p(y=1)\prod_j p(x_j\mid y=1)$ 与 $p(y=0)\prod_j p(x_j\mid y=0)$；实际计算取对数，把连乘变成求和以避免下溢。

### 零概率问题与 Laplace 平滑

MLE 有一个致命缺陷：若某个词在某一类的训练文档中从未出现，则 $\phi_{j\mid y=c}=0$，于是任何含有该词的测试文档在这一类下的似然都是 $0$，后验也变成 $0$。若两类都出现零，后验甚至成为 $0/0$。仅仅因为训练集有限没见过某个事件，就断定它不可能发生，这在统计上是不合理的。

Laplace 平滑（Laplace smoothing，又称加一平滑）在每个计数上加 $1$：

$$
\phi_{j\mid y=c}
=
\frac{1+\sum_{i}\mathbf{1}\{y^{(i)}=c\wedge x_j^{(i)}=1\}}{2+\sum_{i}\mathbf{1}\{y^{(i)}=c\}}.
$$

分母加 $2$ 是因为 $x_j$ 有两种取值，这样两个取值的平滑概率之和仍为 $1$。一般地，对取 $k$ 个值的离散变量，分子加 $1$，分母加 $k$（多项事件模型中 $k=|V|$）；这样平滑后的各概率之和仍为 $1$，且没有一个被估计为 $0$。从贝叶斯角度看，这对应于在 Dirichlet（二值时为 Beta）均匀先验下的后验均值。

### 多项事件模型

Bernoulli 事件模型只记录“出现与否”，丢掉了词频信息。多项事件模型（multinomial event model）把文档看作词序列，同一词可以出现多次：生成过程是先定类别，再从类别对应的词分布中独立地抽取 $L$ 个词。令 $x_j$ 为词 $j$ 的出现次数，$\sum_j x_j=L$（文档长度），$\phi_{j\mid y}$ 为类 $y$ 中抽到词 $j$ 的概率（$\sum_j\phi_{j\mid y}=1$），则

$$
p(x\mid y)=\frac{L!}{\prod_j x_j!}\prod_{j=1}^{|V|}\phi_{j\mid y}^{x_j}.
$$

课程讲义采用另一种等价的记法：令 $x_j\in\{1,\ldots,|V|\}$ 为文档第 $j$ 个位置上的词在词表中的编号，假设每个位置的词都独立地从同一个与位置无关的分布 $p(x_j=k\mid y)=\phi_{k\mid y}$ 中抽取，则 $p(x\mid y)=\prod_{j=1}^{L}\phi_{x_j\mid y}$。两种记法只差上式中的多项式系数。预测时多项式系数 $L!/\prod_j x_j!$ 对所有类相同，可以忽略。MLE 同样是计数：$\phi_{j\mid y=c}$ 等于类 $c$ 所有文档中词 $j$ 的总次数除以类 $c$ 所有文档的总词数。Laplace 平滑时分子加 $1$、分母加 $|V|$，即 $\phi_{j\mid y=c}=\big(1+\sum_i\mathbf{1}\{y^{(i)}=c\}x_j^{(i)}\big)/\big(|V|+\sum_i\mathbf{1}\{y^{(i)}=c\}L^{(i)}\big)$，这里 $L^{(i)}$ 是第 $i$ 篇文档的长度。

两种事件模型如何选择：多项式模型利用词频，Bernoulli 模型只利用是否出现。长文档里词频更有信息量，多项式模型通常更好；极短文本，或只关心“是否出现过”时，Bernoulli 模型更稳。

## 实现

下面的代码实现了 GDA 的拟合与预测、Bernoulli 朴素贝叶斯的拟合与预测。

```python
def gda_fit(X, y):
    """X: (n, d) 不含截距。y in {0, 1}。"""
    X0, X1 = X[y == 0], X[y == 1]
    phi = float(np.mean(y))                    # 正类比例
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)  # 各类均值
    Xc = np.vstack([X0 - mu0, X1 - mu1])       # 相对各自类中心的偏差
    Sigma = (Xc.T @ Xc) / X.shape[0]           # 合并协方差
    return {"phi": phi, "mu0": mu0, "mu1": mu1, "Sigma": Sigma}

def gda_predict_proba(X, params, ridge=1e-6):
    mu0, mu1, Sigma, phi = params["mu0"], params["mu1"], params["Sigma"], params["phi"]
    S = Sigma + ridge * np.eye(Sigma.shape[0])  # 加小对角项防止奇异
    # 解 Sigma^{-1}(x-μ) 而不是显式求逆
    def quad(mu):
        dlt = X - mu
        sol = np.linalg.solve(S, dlt.T).T
        return np.sum(dlt * sol, axis=1)        # 每行的马氏距离平方
    sign, logdet = np.linalg.slogdet(S)
    logp0 = -0.5 * (logdet + quad(mu0)) + np.log(1.0 - phi)
    logp1 = -0.5 * (logdet + quad(mu1)) + np.log(phi)
    return sigmoid(logp1 - logp0)               # 对数后验几率 -> 概率

def naive_bayes_bernoulli(X, y, alpha=1.0):
    """X: (n, V) 的 0/1 矩阵。alpha=1 为 Laplace。"""
    n, V = X.shape
    phi_y = (y.sum() + alpha) / (n + 2.0 * alpha)
    phi_pos = (X[y == 1].sum(axis=0) + alpha) / (y.sum() + 2.0 * alpha)
    phi_neg = (X[y == 0].sum(axis=0) + alpha) / ((1 - y).sum() + 2.0 * alpha)
    return phi_y, phi_pos, phi_neg

def naive_bayes_predict(X, phi_y, phi_pos, phi_neg, eps=1e-12):
    # 对数空间里累加：出现的词贡献 log φ，未出现的词贡献 log(1-φ)
    lp1 = np.log(phi_y) + X @ np.log(phi_pos + eps) + (1 - X) @ np.log(1.0 - phi_pos + eps)
    lp0 = np.log(1.0 - phi_y) + X @ np.log(phi_neg + eps) + (1 - X) @ np.log(1.0 - phi_neg + eps)
    return (lp1 >= lp0).astype(int)
```

几点说明：`gda_predict_proba` 中两类共用同一个 `logdet`，相减时会抵消，保留它只是为了让 `logp0`、`logp1` 是完整的对数密度（差一个常数）；`sigmoid` 沿用前文逻辑回归中定义的函数；`naive_bayes_bernoulli` 对先验 $\phi_y$ 也做了同样强度的平滑（讲义指出，实践中是否平滑 $\phi_y$ 影响不大，因为两类通常都有相当比例的样本，$\phi_y$ 本就远离 $0$）；预测函数在对数空间中计算，避免数千个小概率连乘造成下溢。

> [!EXAMPLE]+ 例6 垃圾邮件里的零概率
>
> 词表中含有 “lottery”。训练集中正类（垃圾邮件）共 100 封，都没有出现这个词。不加平滑时 $\phi_{\mathrm{lottery}\mid y=1}=0$。测试邮件一旦出现 “lottery”，就有 $p(x\mid y=1)=0$，后验崩溃。使用 Laplace 平滑后，该条件概率为 $(0+1)/(100+2)=1/102$，这封邮件仍可被其他强垃圾词推到正类。反过来，平滑强度 $\alpha$ 过大时，所有词的概率都被拉向均匀分布，判别力下降。

<br>

> [!NOTE]+ 对照：判别与生成
>
> 逻辑回归最大化条件似然 $p(y\mid x)$，不对 $x$ 的密度负责。GDA 和朴素贝叶斯最大化联合似然 $p(x,y)$，因此能采样 $x$，也能在特征缺失时对其边缘化。$x$ 近似高斯且 $n$ 较小时，GDA 往往更省数据；特征离散或明显非高斯时，逻辑回归更稳。共享 $\Sigma$ 的 GDA 与逻辑回归的假设函数族相同，但估计量不同，测试误差不必相同。

<br>

> [!NOTE]+ 衔接
>
> 生成模型在原始坐标上建立密度，想要非线性边界就只能修改 $p(x\mid y)$ 的形状（例如改用 QDA）。下一章留在判别路线上，只把 $x$ 映射为 $\phi(x)$，并用核函数避免显式写出 $\phi$。

<br>

# 核方法（Kernel Methods）

线性模型的表达力有限：如果房价与面积之间更接近三次函数关系，那么在原始输入 $x$ 上无论怎样调 $\theta$，直线都拟合不好。标准的补救办法是先把输入映射成一组新的特征，再在新特征上做线性回归。例如定义

$$
\phi(x)=(1,x,x^{2},x^{3})^{\top}\in\mathbb{R}^{4},
$$

则三次函数 $\theta_3 x^3+\theta_2 x^2+\theta_1 x+\theta_0=\theta^{\top}\phi(x)$ 关于 $\theta$ 仍是线性的，前面所有的线性模型算法都可以照搬。术语上，原始输入 $x$ 称为属性（attributes），映射后的 $\phi(x)$ 称为特征（features），$\phi$ 称为特征映射（feature map）。多项式、交互项、直方图编码都是 $\phi$ 的例子。

问题在于特征维数可能爆炸：$x\in\mathbb{R}^{d}$ 的全部三次以内单项式，若按讲义的约定允许重复计数（$x_1x_2$ 与 $x_2x_1$ 各算一项），共有 $1+d+d^{2}+d^{3}$ 个，$d=1000$ 时约为 $10^{9}$，每次更新都要计算并存储一个 $10^{9}$ 维向量，比普通 LMS 慢约 $10^{6}$ 倍；高斯核对应的特征甚至是无限维的。核方法（kernel methods）的核心想法是：若算法只通过内积 $\phi(x)^{\top}\phi(z)$ 访问特征，就把这个内积替换为一个可以直接计算的函数——核（kernel）$K(x,z)=\phi(x)^{\top}\phi(z)$，从而完全不必构造 $\phi$。

本章先把 LMS 改写成只依赖内积的对偶形式，再给出判定一个函数是否为合法核的条件；输入是属性空间中的点对，输出是核矩阵与核化后的预测。与上下文的关系是：上一章在原始 $x$ 上建立密度，而非线性分类需要新的特征；本章用 $K(x,z)=\phi(x)^{\top}\phi(z)$ 避免显式构造 $\phi$；下一章把同样的内积放进最大间隔分类器的对偶问题中。

## 在特征上运行 LMS

先看最直接的做法：显式构造特征，在 $\phi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{p}$ 上拟合 $\theta^{\top}\phi(x)$，其中 $\theta\in\mathbb{R}^{p}$。只需把普通 LMS（最小均方，least mean squares）更新中的 $x^{(i)}$ 全部换成 $\phi(x^{(i)})$，批量梯度下降的更新为

$$
\theta\leftarrow\theta+\alpha\sum_{i=1}^{n}\big(y^{(i)}-\theta^{\top}\phi(x^{(i)})\big)\phi(x^{(i)}).
$$

随机梯度下降版本每次只用一个样本 $i$。这种做法的代价随 $p$ 增长：当 $\phi$ 包含 $x\in\mathbb{R}^{d}$ 的全部三次单项式时，$p=O(d^{3})$，显式存储 $\theta$ 和 $\phi(x)$、计算 $\theta^{\top}\phi(x)$ 都开始变得昂贵。若 $p$ 是无限维，这条路就完全走不通了。

## 核技巧（Kernel Trick）

### 关键观察：$\theta$ 始终是训练特征的线性组合

看上面的更新式：每一步加到 $\theta$ 上的量，都是若干个 $\phi(x^{(i)})$ 的线性组合（系数是 $\alpha$ 乘残差）。若 $\theta$ 从 $0$ 出发，那么无论迭代多少步，始终有

$$
\theta=\sum_{i=1}^{n}\beta_i\phi(x^{(i)}).
$$

其中 $\beta=(\beta_1,\ldots,\beta_n)\in\mathbb{R}^{n}$ 称为对偶系数。这样，$p$ 维（可能无限维）的 $\theta$ 被 $n$ 个数 $\beta_i$ 完全表示。代入预测式：

$$
\theta^{\top}\phi(x)
=
\sum_{i=1}^{n}\beta_i\,\phi(x^{(i)})^{\top}\phi(x)
=
\sum_{i=1}^{n}\beta_i K(x^{(i)},x).
$$

预测只用到核函数在训练点与新点之间的取值，$\phi$ 本身没有出现。

> [!INFO]+ 为何 $\theta$ 永远落在训练点张成的空间里
>
> 从 $\theta=0$ 出发。第 $i$ 个样本的 LMS 更新是 $\theta\leftarrow\theta+\alpha(y^{(i)}-\theta^{\top}\phi(x^{(i)}))\phi(x^{(i)})$，每一步只沿某个 $\phi(x^{(i)})$ 的方向移动。用归纳法：初始时 $\beta^{(0)}=0$；若第 $t$ 步时 $\theta$ 是训练特征的线性组合，加上一个 $\phi(x^{(i)})$ 的倍数后仍然是。因此 $t$ 步之后
>
> $$
> \theta^{(t)}=\sum_{i=1}^{n}\beta_i^{(t)}\phi(x^{(i)}).
> $$
>
> 预测 $\theta^{\top}\phi(x)=\sum_i\beta_i\phi(x^{(i)})^{\top}\phi(x)=\sum_i\beta_i K(x^{(i)},x)$，$\phi$ 本身不必出现。
>
> 直观上看：$\theta$ 无法“跑出”训练特征张成的子空间，因为梯度从来不往别的方向推它。于是只需记录在每个训练点方向上走了多少（$\beta_i$），而所有需要的几何信息都包含在训练点两两之间的内积里。预测只需 $\theta^{\top}\phi(x)=\sum_i\beta_i K(x^{(i)},x)$，核矩阵就是这组内积。

<br>

> [!NOTE]+ 核技巧的要点
>
> 凡是只通过内积 $\phi(x)^{\top}\phi(z)$ 访问特征的算法，都可以把该内积换成 $K(x,z)$，而不必构造 $\phi$。其理论来源是再生核希尔伯特空间（reproducing kernel Hilbert space, RKHS）与 Mercer 定理：合法的 $K$ 对应某个（可能无限维的）$\phi$。与显式多项式特征相比：$d$ 维三次单项式有 $O(d^{3})$ 个，高斯核对应的 $\phi$ 甚至是无限维，而计算量始终是 $O(n^{2})$ 个核矩阵元素。与直接在原始 $x$ 上做线性模型相比：表达力上升，但存储从 $O(d)$ 变成 $O(n^{2})$，而且预测时必须回看训练点。

<br>

### 核函数可以比特征便宜得多

整个算法只需要核

$$
K(x,z)=\phi(x)^{\top}\phi(z).
$$

关键在于很多 $K$ 有远比 $\phi$ 便宜的闭式。例如对三次以内全部单项式（允许重复）构成的特征 $\phi(x)=(1,x_1,\ldots,x_d,x_1x_1,x_1x_2,\ldots,x_1x_1x_1,\ldots)$，内积为 $1+\sum_i x_iz_i+\sum_{i,j}x_ix_jz_iz_j+\sum_{i,j,k}x_ix_jx_kz_iz_jz_k$，后两项分别等于 $(x^{\top}z)^{2}$ 与 $(x^{\top}z)^{3}$，于是内积可以写成 $1+x^{\top}z+(x^{\top}z)^{2}+(x^{\top}z)^{3}$：先用 $O(d)$ 时间算一次 $x^{\top}z$，再做常数次运算即可。更常用的 $d$ 维三次多项式核的闭式是

$$
K(x,z)=(x^{\top}z+c)^{3},
$$

计算代价是 $O(d)$，等价的特征维数却是 $O(d^{3})$（展开后对应所有次数不超过 $3$ 的单项式，各项带不同的权重，$c\ge 0$ 控制低次项的相对权重）。一般地，$(x^{\top}z+c)^{k}$ 对应由全部次数不超过 $k$ 的单项式构成的 $\binom{d+k}{k}$ 维特征空间，而计算 $K(x,z)$ 仍只需 $O(d)$ 时间。这就是核技巧：把算法改写成只出现内积的形式，再把内积换成 $K$。

### 核化 LMS 的训练

对偶系数的批量更新可以由 $\theta$ 的更新代入得到，并且只含 $K$。令核矩阵（kernel matrix，也称 Gram 矩阵）$K\in\mathbb{R}^{n\times n}$ 的元素为 $K_{ij}=K(x^{(i)},x^{(j)})$，则训练点上的预测为

$$
\theta^{\top}\phi(x^{(j)})=\sum_{i}\beta_i K_{ij},
$$

把 $\theta=\sum_j\beta_j\phi(x^{(j)})$ 代入 LMS 更新，比较 $\phi(x^{(j)})$ 前的系数，可得每个 $\beta_j$ 的更新与该点的残差 $y^{(j)}-\sum_i\beta_i K_{ij}$ 成正比，即 $\beta_j\leftarrow\beta_j+\alpha\big(y^{(j)}-\sum_i\beta_i K_{ij}\big)$（对所有 $j$ 同时更新，右端用旧的 $\beta$），向量形式为 $\beta\leftarrow\beta+\alpha(y-K\beta)$（$K$ 对称，故 $\sum_i\beta_iK_{ij}=(K\beta)_j$）。预先算好核矩阵后，每个 $\beta_j$ 的更新只需 $O(n)$ 时间，与特征维数 $p$ 无关。

实际使用时，核矩阵可以在训练前一次性算好；但每一步都要用到整个核矩阵，训练代价至少随 $n^{2}$ 增长。因此核方法在 $n$ 适中（几千到几万）、特征维数很高时最划算；$n$ 达到百万级时，存储 $n\times n$ 的核矩阵本身就成了瓶颈。

## 合法核的判定与常用核

### Mercer 条件：核矩阵必须对称半正定

给定任意函数 $K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$，并不一定存在某个 $\phi$ 使 $K(x,z)=\phi(x)^{\top}\phi(z)$。判定条件（Mercer 条件）是：对任意有限点集 $\{x^{(1)},\ldots,x^{(n)}\}$，对应的核矩阵 $K$ 必须对称半正定，即对任意 $z\in\mathbb{R}^{n}$，

$$
z^{\top}Kz\ge 0.
$$

等价地说，$K$ 的特征值全部非负。对称性来自内积的对称性 $\phi(x)^{\top}\phi(z)=\phi(z)^{\top}\phi(x)$。实际用法是反向排除：若在某个数据集上算出的 $K$ 有负特征值，该函数就不能作为核。

> [!INFO]+ 为何半正定不可少
>
> 必要性：若 $K=\Phi\Phi^{\top}$，其中 $\Phi$ 的第 $i$ 行是 $\phi(x^{(i)})^{\top}$，则 $z^{\top}Kz=z^{\top}\Phi\Phi^{\top}z$，即 $z^{\top}Kz=\|\Phi^{\top}z\|_{2}^{2}\ge 0$。充分性（在有限样本上）：对称半正定的 $K$ 必有平方根 $K=K^{1/2}K^{1/2}$，可以把该平方根的行当作这些样本上的特征。在优化问题中，$K$ 出现在二次项 $\alpha^{\top}(K\odot yy^{\top})\alpha$ 里（下一章的支持向量机对偶即如此）；若 $K$ 不定，对偶目标不再是凹函数，二次项可以沿某个方向趋于 $-\infty$。
>
> 换个角度：若 $z^{\top}Kz<0$，则 $K$ 不能写成任何 $\Phi\Phi^{\top}$，所谓的“内积”在实特征空间中不存在。对偶目标的凹性依赖于 $K$ 半正定，一旦不定，二次项可以沿某方向无下界。因此 Mercer 条件是可优化性的条件，而不只是形式上的对称要求。

<br>

### 常用核

| 核 | 公式 | 对应直觉 |
| :--- | :--- | :--- |
| 线性 | $K(x,z)=x^{\top}z$ | $\phi$ 为恒等 |
| 多项式 | $K(x,z)=(x^{\top}z+c)^{m}$ | 次数 $\le m$ 的单项式 |
| 高斯 / RBF | $K(x,z)=\exp(-\|x-z\|_{2}^{2}/(2\sigma^{2}))$ | 无限维特征，局部相似 |
| 字符串核等 | 由领域定义 | 离散结构上的加权共现 |

可以把核理解为相似度：$x$ 与 $z$ 越相似，$K(x,z)$ 越大。高斯核（也称径向基函数核，radial basis function, RBF）满足 $K(x,x)=1$，并随距离增大而衰减到 $0$。带宽 $\sigma$ 是最关键的超参数：$\sigma$ 太小时，核矩阵几乎只有对角元非零，每个训练点只“记住”自己，导致过拟合；$\sigma$ 太大时，所有点彼此都很相似，模型退化为近乎常数，导致欠拟合。还需注意：训练和测试必须使用同一个核公式和同一个尺度；由于 RBF 依赖欧氏距离，量纲不同的特征会主导距离，应当先标准化特征再计算 RBF 核。

## 实现

下面的代码给出多项式核、RBF 核和核化 LMS 的预测函数，均按矩阵批量计算。

```python
def polynomial_kernel(X, Z, degree=3, c=1.0):
    """X: (n, d), Z: (m, d) -> (n, m)。"""
    return (X @ Z.T + c) ** degree

def rbf_kernel(X, Z, sigma=1.0):
    # 利用 ||x - z||² = ||x||² + ||z||² - 2 xᵀz 一次算出所有距离平方
    X2 = np.sum(X * X, axis=1)[:, None]
    Z2 = np.sum(Z * Z, axis=1)[None, :]
    sq = np.maximum(X2 + Z2 - 2.0 * X @ Z.T, 0.0)  # 截断浮点误差导致的微小负数
    return np.exp(-sq / (2.0 * sigma ** 2))

def kernel_lms_predict(X_train, beta, x, kernel_fn):
    # 预测 = Σ_i β_i K(x^(i), x)，需要回看全部训练点
    return float(beta @ kernel_fn(X_train, x[None, :]).ravel())
```

`rbf_kernel` 用展开式代替逐对求差，复杂度相同但能利用矩阵乘法加速；`np.maximum(..., 0.0)` 防止相减时的舍入误差产生负的距离平方。训练 $\beta$ 时可以先算 `K = kernel_fn(X_train, X_train)`，再反复执行 $\beta\leftarrow\beta+\alpha(y-K\beta)$。

> [!EXAMPLE]+ 例7 二次核展开
>
> 取 $x,z\in\mathbb{R}^{2}$，$(x^{\top}z)^{2}=(x_1z_1+x_2z_2)^{2}$，展开得 $K=(x^{\top}z)^{2}=x_1^{2}z_1^{2}+x_2^{2}z_2^{2}+2x_1 x_2 z_1 z_2$。这正是两个三维向量的内积，对应的特征映射为 $\phi(x)=(x_1^{2},x_2^{2},\sqrt{2}x_1 x_2)$。$d=2$ 时显式写出 $\phi$ 还很容易；但 $d=100$ 的三次核对应约 $10^{6}$ 维特征（按允许重复的计数为 $1+100+100^{2}+100^{3}=1010101$；即使只计不同单项式也有 $\binom{103}{3}=176851$ 个），就不应再展开，直接用 $(x^{\top}z+c)^{3}$ 计算即可。

<br>

> [!NOTE]+ 小结
>
> - 特征映射 $\phi$ 能让线性模型表达非线性关系，但显式特征的维数可能爆炸。
> - 从 $\theta=0$ 出发的梯度更新使 $\theta=\sum_i\beta_i\phi(x^{(i)})$，训练与预测都只需内积。
> - 用核 $K(x,z)=\phi(x)^{\top}\phi(z)$ 替换内积，计算代价从特征维数转移到 $O(n^{2})$ 的核矩阵。
> - 合法核的核矩阵必须对称半正定；RBF 核的带宽 $\sigma$ 控制过拟合与欠拟合，使用前应标准化特征。

> [!NOTE]+ 衔接
>
> 核把 LMS 改写成只依赖 $K_{ij}$ 的形式。下一章不再最小化平方损失，而是最大化几何间隔；对偶目标里出现的仍是同一类内积，因此核可以原样代入。

<br>

<br>

# 支持向量机（Support Vector Machines）

逻辑回归已经能给出线性分类边界，但它只要求“把训练点分对、并让对数似然尽量大”，并没有直接追求“边界离两类点都尽量远”。直观上，离边界越远的预测越可靠：在逻辑回归里，$|\theta^{\top}x|$ 越大，$\sigma(\theta^{\top}x)$ 越接近 $0$ 或 $1$，模型越有把握。设想平面上三个点：点 $A$ 远离决策边界，边界稍作扰动也不会改变对它的预测；点 $C$ 几乎贴在边界上，边界轻轻一动它就可能被判到另一类，对它的预测把握很低；点 $B$ 介于二者之间。支持向量机（support vector machine，SVM）的出发点，就是把“所有训练点都分对，并且尽量远离边界”写成一个明确的优化问题。

核心想法一句话：在所有能把两类分开的超平面里，选间隔（margin）最大的那一个，即最大间隔分隔超平面（separating hyperplane）。本章的问题设定是二分类，但为了让公式简洁，标签改记为 $y\in\{-1,+1\}$（而不是逻辑回归里的 $\{0,1\}$）。

本章的路线是：先定义两种间隔（功能间隔与几何间隔），由几何间隔推出“最优间隔分类器”的原问题；再借助 Lagrange 对偶把它改写成只含样本内积的对偶问题；然后加入松弛变量处理线性不可分与噪声（软间隔）；最后介绍求解对偶的 SMO 算法与代码实现。输入是一组带标签的训练点，输出是参数 $w,b$，或者等价地，对偶系数 $\alpha$ 与支持向量。

与前后章节的关系：核方法一章给出了合法的核函数 $K$（即对应某个特征映射 $\phi$ 的内积）。本章在 $\phi$ 空间中最大化几何间隔，并且会看到对偶问题中训练数据只以内积 $K_{ij}=K(x^{(i)},x^{(j)})$ 的形式出现，从而可以直接核化。之后的决策树与神经网络是得到非线性边界的另外两条路，它们都不经过核矩阵。

> [!NOTE]+ 支持向量机的定位
>
> SVM 在间隔最大的超平面上做分类，最优法向量只由贴在间隔边界上的训练点（支持向量，support vectors）决定。它的理论来源是统计学习理论里的间隔界：几何间隔越大，VC 型容量（capacity）越小，泛化越有保证；它的设计目标不是输出概率。
>
> - 相对逻辑回归：SVM 使用 hinge 损失，间隔之外、已经分对的点梯度为 $0$，因此解是稀疏的（只依赖少数支持向量），也便于核化；代价是输出不是 $P(y=1\mid x)$。
> - 相对感知机（perceptron）这类“找到任意一个可分平面即可”的方法：SVM 的目标是最大间隔，而不是任意一个可分平面。

<br>

## 分类器的形式与记号

SVM 的分类器写成

$$
h_{w,b}(x)=g(w^{\top}x+b),\qquad
g(z)=\begin{cases}+1 & z\ge 0\\ -1 & z<0\end{cases}.
$$

其中 $x\in\mathbb{R}^{d}$ 是输入，$w\in\mathbb{R}^{d}$ 是超平面的法向量，$b\in\mathbb{R}$ 是截距。与线性回归、逻辑回归的写法不同，这里不再把 $x_0=1$ 塞进参数向量，截距 $b$ 单独出现——因为后面 $w$ 与 $b$ 在优化中扮演不同角色（目标函数只惩罚 $\|w\|$，不惩罚 $b$）。另外，$g$ 直接输出 $\pm 1$，中间不经过任何概率估计。

## 功能间隔与几何间隔（margin）

要“最大化间隔”，先得把“间隔”定义清楚。直觉上的间隔是点到超平面的距离，但最先想到的量 $y(w^{\top}x+b)$ 并不是距离，它会随参数的尺度变化。下面分别定义这两个量，并说明为什么真正要最大化的是第二个。

### 功能间隔（functional margin）

训练样本 $(x^{(i)},y^{(i)})$ 关于 $(w,b)$ 的功能间隔定义为

$$
\hat\gamma^{(i)}=y^{(i)}(w^{\top}x^{(i)}+b).
$$

当 $y^{(i)}=+1$ 时，要让它大，需要 $w^{\top}x^{(i)}+b$ 是一个大的正数；当 $y^{(i)}=-1$ 时，需要 $w^{\top}x^{(i)}+b$ 是一个绝对值大的负数。因此 $\hat\gamma^{(i)}>0$ 恰好表示该样本被分对，并且数值越大看起来“越有把握”。

问题在于尺度：把 $(w,b)$ 同时乘以 $2$，功能间隔也乘以 $2$，但 $g$ 只看 $w^{\top}x+b$ 的符号，预测完全不变。也就是说，只要放大参数，功能间隔就能被任意放大，而分类器本身没有任何改进。因此功能间隔单独拿来当优化目标没有尺度意义，必须配合某种归一化。

### 几何间隔（geometric margin）

几何间隔是点到超平面的有符号欧氏距离。首先，$w$ 垂直于超平面 $\{x:w^{\top}x+b=0\}$：平面上任意两点 $x_1,x_2$ 满足 $w^{\top}(x_1-x_2)=0$。于是 $w/\|w\|$ 是单位法向量。

设 $x^{(i)}$ 是正类点，它到平面的距离记为 $\gamma^{(i)}$。从 $x^{(i)}$ 出发沿 $-w/\|w\|$ 方向走 $\gamma^{(i)}$，正好落到平面上，所以落点满足平面方程：

$$
w^{\top}\Big(x^{(i)}-\gamma^{(i)}\frac{w}{\|w\|}\Big)+b=0,
$$

展开并利用 $w^{\top}w=\|w\|^{2}$，再乘上 $y^{(i)}$ 让负类点也得到正的距离，解出

$$
\gamma^{(i)}=y^{(i)}\Big(\Big(\frac{w}{\|w\|}\Big)^{\top}x^{(i)}+\frac{b}{\|w\|}\Big)=\frac{\hat\gamma^{(i)}}{\|w\|}.
$$

可以看到几何间隔就是“用 $\|w\|$ 归一化后的功能间隔”。把 $(w,b)$ 乘以任意正数 $c$，分子分母同乘 $c$，几何间隔不变——它对参数的正尺度缩放不变，这正是功能间隔缺少的性质。另外，若 $\|w\|=1$，两种间隔相等。

一个分类器在整个训练集上的间隔取所有样本中的最小值，即“最差的那个点”：

$$
\gamma=\min_i\gamma^{(i)},\qquad \hat\gamma=\min_i\hat\gamma^{(i)}.
$$

> [!INFO]+ 从平面方程到距离（推导细节）
>
> 平面为 $w^{\top}x+b=0$，单位法向是 $w/\|w\|$。从正类点 $x$ 沿 $-w/\|w\|$ 走距离 $\gamma$，落到平面上的点是 $x-\gamma w/\|w\|$。该点满足平面方程：
>
> $$
> w^{\top}x-\gamma\|w\|+b=0\implies\gamma=\frac{w^{\top}x+b}{\|w\|}.
> $$
>
> 对负类点，$w^{\top}x+b<0$，需要再乘 $y=-1$，否则距离会带错符号。与功能间隔 $\hat\gamma=y(w^{\top}x+b)$ 对照，得到 $\gamma=\hat\gamma/\|w\|$。
>
> 感性理解：$(w,b)$ 同时乘正数 $c$ 时，功能间隔按 $c$ 缩放，而点到超平面的欧氏距离不变，所以几何间隔才是尺度不变的间隔。下一节会利用这一点：把功能间隔固定为 $1$ 之后，最大化几何间隔就等价于最小化 $\|w\|$。

<br>

## 最优间隔分类器（optimal margin classifier）

有了几何间隔，就可以把“找最大间隔超平面”写成优化问题。本节先假设训练集线性可分（存在超平面把两类完全分开），目标是让训练集上最小的几何间隔尽量大。难点在于最直接的写法不是凸问题，需要借助尺度不变性做两步改写，最终得到一个标准的凸二次规划。

### 第一步：直接写法

最大化训练集的几何间隔：

$$
\max_{\gamma,w,b}\ \gamma
\quad\mathrm{s.t.}\quad
y^{(i)}(w^{\top}x^{(i)}+b)\ge\gamma,\ 
\|w\|=1.
$$

约束 $\|w\|=1$ 使功能间隔等于几何间隔，于是第一组约束说的是“每个点的几何间隔都至少为 $\gamma$”。但 $\{w:\|w\|=1\}$ 是一个球面，不是凸集，这个问题无法直接交给凸优化求解器。

### 第二步：去掉非凸约束并固定尺度

利用 $\gamma=\hat\gamma/\|w\|$，可以改为最大化 $\hat\gamma/\|w\|$，约束变为 $y^{(i)}(w^{\top}x^{(i)}+b)\ge\hat\gamma$，从而去掉 $\|w\|=1$。此时目标 $\hat\gamma/\|w\|$ 仍非凸，但还剩一个自由度：$(w,b)$ 可以整体缩放而不改变分类器。于是可以固定尺度 $\hat\gamma=1$（这是合法的，因为任何可行解都能缩放到训练集最小功能间隔恰为 $1$）。目标变成最大化 $1/\|w\|$，等价于最小化 $\|w\|^{2}$：

$$
\min_{w,b}\ \frac12\|w\|_{2}^{2}
\quad\mathrm{s.t.}\quad
y^{(i)}(w^{\top}x^{(i)}+b)\ge 1,\quad i=1,\ldots,n.
$$

这里 $n$ 是训练样本数。这是凸二次目标加线性不等式约束，属于二次规划（quadratic programming，QP），可以用现成的 QP 软件求解，且全局最优可得。

### 解的结构：支持向量

在最优解处，至少有一些点使约束取等号，即 $y^{(i)}(w^{\top}x^{(i)}+b)=1$，它们恰好落在两条间隔边界 $w^{\top}x+b=\pm 1$ 上。这些点就是支持向量。后面对偶推导会严格说明：$w$ 只由支持向量决定，其他点即使删除也不改变解。通常支持向量的数目远少于训练样本数。

> [!INFO]+ 为何最小化 $\|w\|^{2}/2$ 就是最大间隔
>
> 几何间隔 $\gamma=\hat\gamma/\|w\|$。固定 $\hat\gamma=1$ 后 $\gamma=1/\|w\|$，因此最小化 $\|w\|$ 与最大化 $\gamma$ 等价。除以 $2$ 与改成平方都是单调变换，不改变最优点，却让梯度等于 $w$、Hessian 等于单位阵，问题成为更标准的二次规划。
>
> 需要注意：约束 $y(w^{\top}x+b)\ge 1$ 说的是“功能间隔至少为 1”，不是“几何距离至少为 1”；最近点到超平面的几何距离是 $1/\|w\|$。
>
> 感性理解：两类各自的支撑超平面 $w^{\top}x+b=+1$ 与 $w^{\top}x+b=-1$ 之间的宽度为 $2/\|w\|$。单纯放大 $w$ 只改变功能间隔的数值，不改变真实宽度；在把边界上的功能间隔固定为 $1$ 之后，缩短 $\|w\|$ 才真正加宽了间隔带。

<br>

## Lagrange 对偶（duality）与对偶问题

上一节的原问题已经能用 QP 求解，为什么还要推对偶？有两个原因。第一，对偶问题中训练数据只以内积 $(x^{(i)})^{\top}x^{(j)}$ 的形式出现，把它换成核函数 $K$ 就能在高维甚至无穷维的特征空间里做最大间隔分类，而不必显式写出 $\phi$。第二，对偶变量 $\alpha_i$ 直接指出哪些点是支持向量，并且引出了高效的 SMO 算法。本节先回顾一般约束优化的 Lagrange 对偶与 KKT 条件，再套用到 SVM 上。

### 一般形式：原问题、对偶问题与 KKT 条件

考虑带不等式与等式约束的原问题（primal problem）：

$$
\min_{w}f(w)
\quad\mathrm{s.t.}\quad
g_i(w)\le 0,\ 
h_i(w)=0.
$$

引入乘子 $\alpha_i$（对应不等式约束）与 $\beta_i$（对应等式约束），定义广义 Lagrangian（generalized Lagrangian）：

$$
\mathcal{L}(w,\alpha,\beta)=f(w)+\sum_{i}\alpha_i g_i(w)+\sum_{i}\beta_i h_i(w),\qquad\alpha_i\ge 0.
$$

**原问题的另一种写法。** 定义 $\theta_P(w)=\max_{\alpha\ge 0,\beta}\mathcal{L}$（下标 $P$ 表示 primal）。若 $w$ 违反某个约束（例如某个 $g_i(w)>0$ 或 $h_i(w)\ne 0$），可以令对应的乘子趋于无穷，使 $\theta_P=+\infty$；若 $w$ 可行，最优选择是让 $\alpha_i g_i(w)$ 项全取 $0$，于是 $\theta_P=f(w)$。因此 $\theta_P$ 相当于“在可行域上等于 $f$、在可行域外等于 $+\infty$”的函数，最小化 $\theta_P$ 等价于原问题，记其最优值为 $p^{*}=\min_w\max_{\alpha\ge0,\beta}\mathcal{L}$。

**对偶问题。** 交换 min 与 max 的顺序：定义对偶函数 $\theta_D(\alpha,\beta)=\min_{w}\mathcal{L}$，对偶问题（dual problem）是 $\max_{\alpha\ge 0,\beta}\theta_D$，最优值记为 $d^{*}$。

**弱对偶与强对偶。** 始终有弱对偶（weak duality）：对偶最优值 $\le$ 原最优值，即 $d^{*}\le p^{*}$（“max min $\le$ min max”）。在凸问题（$f$ 与 $g_i$ 凸、$h_i$ 仿射）且满足约束规范（constraint qualification，如 Slater 条件：存在严格可行点，使所有 $g_i(w)<0$）时，强对偶（strong duality）成立：两者相等。此时存在 $w^{*},\alpha^{*},\beta^{*}$ 满足 KKT（Karush–Kuhn–Tucker）条件：

- 原可行：$g_i(w^{*})\le 0$，$h_i(w^{*})=0$；对偶可行：$\alpha_i^{*}\ge 0$；
- 平稳性：$\nabla_w\mathcal{L}=0$（在 $(w^{*},\alpha^{*},\beta^{*})$ 处取值）；
- 互补松弛（complementary slackness）：$\alpha_i g_i(w)=0$（在最优点处）。

互补松弛是 SVM 稀疏性的来源：若某个约束不起作用（$g_i(w^{*})<0$），对应乘子必须为 $0$；只有约束取等号（$g_i(w^{*})=0$，称为“活跃”约束）时，$\alpha_i^{*}$ 才可能大于 $0$。反过来，满足 KKT 条件的点也就是原问题与对偶问题的最优解，因此 KKT 条件还可以当作检验收敛的标准（后面 SMO 正是这么用的）。

### 套用到最优间隔分类器

对最优间隔分类器，$f=\|w\|^{2}/2$，每个样本贡献一个不等式约束 $g_i(w)=1-y^{(i)}(w^{\top}x^{(i)}+b)\le 0$，没有等式约束。Lagrangian 为

$$
\mathcal{L}=\frac12\|w\|_{2}^{2}-\sum_{i=1}^{n}\alpha_i\big[y^{(i)}(w^{\top}x^{(i)}+b)-1\big].
$$

为求对偶函数，对 $w,b$ 求最小（$\mathcal{L}$ 对 $w$ 是凸二次函数，令梯度为零即可）：

$$
w=\sum_{i=1}^{n}\alpha_i y^{(i)}x^{(i)},\qquad
\sum_{i=1}^{n}\alpha_i y^{(i)}=0.
$$

第二个式子来自 $\partial\mathcal{L}/\partial b=-\sum_i\alpha_i y^{(i)}=0$。注意 $\mathcal{L}$ 对 $b$ 是线性的：如果 $\sum_i\alpha_i y^{(i)}\ne 0$，对 $b$ 取最小会得到 $-\infty$，所以这个等式实际上是对偶问题的一个约束。把 $w$ 代回 $\mathcal{L}$，得到对偶问题：

$$
\max_{\alpha}\ 
\sum_{i=1}^{n}\alpha_i
-\frac12\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_j y^{(i)}y^{(j)}(x^{(i)})^{\top}x^{(j)}
\quad\mathrm{s.t.}\quad
\alpha_i\ge 0,\ 
\sum_{i}\alpha_i y^{(i)}=0.
$$

原问题是凸的，且线性可分时满足 Slater 类条件（约束是仿射的），强对偶成立，所以解对偶问题与解原问题得到相同的最优值；由 $\alpha^{*}$ 可以通过 $w=\sum_i\alpha_i y^{(i)}x^{(i)}$ 还原 $w^{*}$。

> [!INFO]+ 从 Lagrangian 到对偶二次型（逐步代入）
>
> $\mathcal{L}=\|w\|_{2}^{2}/2-\sum_i\alpha_i\big[y^{(i)}(w^{\top}x^{(i)}+b)-1\big]$。对 $w$ 求导并令为零：
>
> $$
> w-\sum_i\alpha_i y^{(i)}x^{(i)}=0\implies w=\sum_i\alpha_i y^{(i)}x^{(i)}.
> $$
>
> 对 $b$ 求导：$\sum_i\alpha_i y^{(i)}=0$。把 $w$ 代回 $\mathcal{L}$，二次项给出
>
> $$
> \frac12\|w\|_{2}^{2}=\frac12\sum_{i,j}\alpha_i\alpha_j y^{(i)}y^{(j)}(x^{(i)})^{\top}x^{(j)}.
> $$
>
> 约束项里的 $-\sum_i\alpha_i y^{(i)}w^{\top}x^{(i)}$ 恰好等于 $-\|w\|_2^2$，即再贡献同样一份二次项（带负号）；含 $b$ 的项 $-b\sum_i\alpha_i y^{(i)}$ 因等式约束为 $0$；线性项 $\sum_i\alpha_i$ 留下。整理后对偶目标是
>
> $$
> \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j y^{(i)}y^{(j)}(x^{(i)})^{\top}x^{(j)}.
> $$
>
> 约束只剩 $\alpha_i\ge 0$ 与 $\sum_i\alpha_i y^{(i)}=0$。互补松弛 $\alpha_i\big(y^{(i)}(w^{\top}x^{(i)}+b)-1\big)=0$ 说明：不在间隔边界上的点 $\alpha_i=0$，不参与 $w$。
>
> 感性理解：最优法向量是训练点的带符号加权和，$w=\sum_i\alpha_i y^{(i)}x^{(i)}$。互补松弛迫使功能间隔严格大于 $1$ 的点满足 $\alpha_i=0$，故 $w$ 只由支持向量决定。对偶目标中的二次项只涉及样本两两内积；把它换成 $K_{ij}$，就是在特征空间做同一个间隔问题，不必写出 $\phi$。

<br>

### 支持向量、预测与核化

由互补松弛，$\alpha_i>0$ 仅当 $y^{(i)}(w^{\top}x^{(i)}+b)=1$，即该点是支持向量。把 $w$ 的表达式代入决策函数，预测只需要计算新点与支持向量（SV）的内积：

$$
w^{\top}x+b=\sum_{i\in\mathrm{SV}}\alpha_i y^{(i)}(x^{(i)})^{\top}x+b.
$$

由于大多数 $\alpha_i=0$，求和实际上只遍历少数支持向量，预测代价与支持向量个数成正比，而不是与训练集大小成正比。

把对偶目标和预测式中的 $(x^{(i)})^{\top}x^{(j)}$ 换成 $K(x^{(i)},x^{(j)})$，即得到核 SVM（kernel SVM）：它等价于在特征空间 $\phi(x)$ 中求最大间隔分类器，但整个训练与预测过程不必出现 $\phi$。

### 截距 $b$ 的恢复

对偶问题里没有 $b$，需要事后求出。任取一个支持向量 $x^{(s)}$，它满足等式约束 $y^{(s)}(w^{\top}x^{(s)}+b)=1$；两边乘 $y^{(s)}$ 并利用 $(y^{(s)})^{2}=1$，得到

$$
b=y^{(s)}-\sum_i\alpha_i y^{(i)}K(x^{(i)},x^{(s)}).
$$

理论上任何一个支持向量算出的 $b$ 都相同，但数值上应对多个支持向量取平均，以减轻舍入误差。CS229 主讲义还给出一个等价的对称写法（线性核情形）：$b^{*}=-\tfrac12\big(\max_{i:y^{(i)}=-1}w^{*\top}x^{(i)}+\min_{i:y^{(i)}=+1}w^{*\top}x^{(i)}\big)$，即把超平面放在两类最近点的正中间。在下一节的软间隔中，用于求 $b$ 的应是满足 $0<\alpha_i<C$ 的支持向量（只有它们严格落在间隔边界上）。

## 软间隔 SVM（soft margin）

硬间隔 SVM 有两个现实问题：其一，数据线性不可分（哪怕在特征空间里）时，约束 $y^{(i)}(w^{\top}x^{(i)}+b)\ge 1$ 根本没有可行解；其二，即使可分，单个离群点也可能把最大间隔超平面拽得很偏、间隔变得很窄。软间隔的想法是：允许少数点进入间隔带甚至被分错，但每违反一点就付出代价，在“间隔宽”与“违反少”之间做权衡。

### 原问题

对每个样本引入松弛变量（slack variable）$\xi_i\ge 0$，表示它违反间隔约束的程度：

$$
\min_{w,b,\xi}\ 
\frac12\|w\|_{2}^{2}+C\sum_{i=1}^{n}\xi_i
\quad\mathrm{s.t.}\quad
y^{(i)}(w^{\top}x^{(i)}+b)\ge 1-\xi_i,\ 
\xi_i\ge 0.
$$

$\xi_i=0$ 表示该点满足原来的间隔要求；$0<\xi_i\le 1$ 表示点进入了间隔带但仍分对；$\xi_i>1$ 表示点被分错。$C>0$ 是权衡间隔与违反的超参数：$C\rightarrow\infty$ 时任何违反都代价无穷，逼近硬间隔；$C$ 小则允许更多点进入间隔带，换取更宽的间隔与对离群点更强的鲁棒性。

### 对偶：只多了一个盒约束

引入乘子 $\alpha_i\ge 0$（间隔约束）与 $r_i\ge 0$（约束 $\xi_i\ge 0$），对 $\xi_i$ 求导得 $C-\alpha_i-r_i=0$。由于 $r_i\ge 0$，这就给出 $\alpha_i\le C$；其余推导与硬间隔完全相同，$\xi$ 从对偶目标中消失。因此对偶目标不变，只是约束改为盒约束（box constraint）：

$$
0\le\alpha_i\le C,\qquad
\sum_{i}\alpha_i y^{(i)}=0.
$$

结合 KKT 条件，$\alpha_i$ 的取值把训练点分成三类（记 $f(x)=w^{\top}x+b$）：

- $\alpha_i=0$：点在间隔外侧且分对，$y^{(i)}f(x^{(i)})\ge 1$；
- $0<\alpha_i<C$：此时 $r_i>0$ 故 $\xi_i=0$，点恰好落在间隔边界上，$y^{(i)}f(x^{(i)})=1$，是“边界上”的支持向量；
- $\alpha_i=C$：违反间隔或被分错的点，$y^{(i)}f(x^{(i)})\le 1$。

> [!NOTE]+ 软间隔的定位
>
> 软间隔允许点进入间隔带或被分错，违反量记为松弛 $\xi_i$，惩罚系数是 $C$。它的来源是：硬间隔在线性不可分或含噪声时无可行解。
>
> - 相对硬间隔：$C<\infty$ 保证二次规划总有可行解，并且对离群点不那么敏感。
> - 相对交叉熵（逻辑回归）：远处已经分对的点仍然不拉动 $w$，只有间隔被违反的点（$\xi_i>0$）在对偶中顶到盒约束上界 $C$。
> - $C$ 本身是超参数，必须用验证集（或交叉验证）选取。

<br>

### 等价的无约束写法：hinge 损失

在最优解处，松弛变量一定取能满足约束的最小值，即 $\xi_i=\max\big(0,1-y^{(i)}(w^{\top}x^{(i)}+b)\big)$。代回目标，得到等价的无约束写法，使用 hinge 损失 $\max(0,1-z)$：

$$
\min_{w,b}\ 
\frac12\|w\|_{2}^{2}
+C\sum_{i=1}^{n}\max\big(0,1-y^{(i)}(w^{\top}x^{(i)}+b)\big).
$$

这个形式把 SVM 放进了“正则化 + 经验损失”的统一框架：$\frac12\|w\|^2$ 是 L2 正则项，hinge 是对功能间隔 $z=y(w^{\top}x+b)$ 的损失函数。与逻辑回归的 logistic 损失 $\log(1+e^{-z})$ 对比：hinge 在功能间隔大于 $1$ 后恒为 $0$、梯度为 $0$，因此远处已分对的点不再拉 $w$；而交叉熵对任何点都永远给一点梯度。这种写法也直接引出下文“实现”中的子梯度法。

需要注意的常见坑：SVM 的输出 $w^{\top}x+b$ 不是校准概率。需要概率时，应另外做 Platt scaling（在验证集上拟合一个 $\sigma(a f(x)+c)$），不能把 $w^{\top}x+b$ 直接当作 $\sigma$ 的输入而不检验校准。

## SMO 算法（Sequential Minimal Optimization）

软间隔对偶是一个带盒约束和一个线性等式约束的二次规划。通用 QP 求解器需要存储和处理整个 $n\times n$ 核矩阵，样本数上万时内存和时间都难以承受。SMO（sequential minimal optimization，Platt 1998）是专门为 SVM 对偶设计的分解算法：每次只优化极少数变量，并且这个小子问题可以解析求解。

### 为什么每次要选两个变量

坐标上升（coordinate ascent）每次固定其余变量、只优化一个 $\alpha_i$。但在 SVM 对偶中，等式约束 $\sum\alpha_i y^{(i)}=0$ 意味着：其余 $\alpha$ 固定时，$\alpha_i=-y^{(i)}\sum_{k\ne i}\alpha_k y^{(k)}$ 已被完全确定，单个坐标根本动不了。因此 SMO 每次选两个 $\alpha_i,\alpha_j$，在其余固定时解析求解——这是满足等式约束的最小可行工作集。

### 两变量子问题的解析解

设 $i,j$ 被选中。由等式约束，$\alpha_i y^{(i)}+\alpha_j y^{(j)}=\zeta$（$\zeta$ 是由其他 $\alpha$ 决定的常数），所以一个变量由另一个线性确定。再结合 $0\le\alpha\le C$，可以推出被优化变量的可行区间 $[L,H]$。以 $\alpha_j$ 为自由变量（Platt 原文的常见写法）：

- 若 $y^{(i)}\ne y^{(j)}$：$L=\max(0,\alpha_j-\alpha_i)$，$H=\min(C,C+\alpha_j-\alpha_i)$；
- 若 $y^{(i)}=y^{(j)}$：$L=\max(0,\alpha_i+\alpha_j-C)$，$H=\min(C,\alpha_i+\alpha_j)$。

把 $\alpha_i$ 用 $\alpha_j$ 表示后，目标成为 $\alpha_j$ 的一元二次函数。记 $f(x)=\sum_k\alpha_k y^{(k)}K(x^{(k)},x)+b$，误差 $E_k=f(x^{(k)})-y^{(k)}$，曲率 $\eta=K_{ii}+K_{jj}-2K_{ij}$。无约束顶点为

$$
\alpha_j^{\mathrm{new}}=\alpha_j+\frac{y^{(j)}(E_i-E_j)}{\eta},
$$

顶点若出界就投影（clip）到端点 $L$ 或 $H$；然后由等式约束回代 $\alpha_i^{\mathrm{new}}=\alpha_i+y^{(i)}y^{(j)}(\alpha_j-\alpha_j^{\mathrm{new,clipped}})$，并更新 $b$。

### 选对与收敛

启发式：优先选违反 KKT 条件最严重的一对（例如先找违反最厉害的 $\alpha_i$，再选使 $|E_i-E_j|$ 最大的 $\alpha_j$，以求步长最大）；若随意选取，收敛会很慢。当所有 $\alpha$ 都在容差内满足上一节列出的三条 KKT 条件时停止。SMO 每步只需要核矩阵的若干列，在线性核或配合核缓存时，避免了 $n\times n$ 的稠密求解器。

> [!NOTE]+ SMO 的定位
>
> SMO 对对偶二次规划每次只动两个 $\alpha$，因为等式 $\sum\alpha_i y^{(i)}=0$ 使单坐标更新不可行。它来源于 Platt 对核 SVM 的分解算法，并不是新的对偶目标。
>
> - 相对一次性求解 $n\times n$ 稠密 QP：内存与时间从“整个核矩阵”降到“缓存若干列”。
> - 相对普通坐标上升：最小可行工作集大小是 $2$。
> - 启发式应选违反 KKT 的一对，否则收敛很慢。

<br>

> [!EXAMPLE]+ 例8 二维硬间隔
>
> 正类 $(2,2),(3,3)$，负类 $(0,0),(0,1)$。最优边界大致沿两簇之间，支持向量是离对方最近的那几个点，不是全部四点。
>
> 具体算一下：两类凸包之间最近的一对点是正类的 $(2,2)$ 与负类的 $(0,1)$，最优法向量与它们的连线 $(2,1)$ 同向。令 $w=c(2,1)$，由两个支持向量取等号 $w^{\top}(2,2)+b=1$、$w^{\top}(0,1)+b=-1$，解得 $c=0.4$，即 $w=(0.8,0.4)$，$b=-1.4$。验证四个点的功能间隔分别为 $1,\ 2.2,\ 1.4,\ 1$：只有 $(2,2)$ 与 $(0,1)$ 取等号，它们是支持向量；几何间隔为 $1/\|w\|=\sqrt5/2\approx1.118$，正好是两点距离 $\sqrt5$ 的一半。对偶系数为 $\alpha_{(2,2)}=\alpha_{(0,1)}=0.4$，其余为 $0$：确实 $0.4\cdot(2,2)-0.4\cdot(0,1)=(0.8,0.4)=w$，且 $\sum_i\alpha_iy^{(i)}=0$；对偶目标 $0.8-0.4=0.4$ 等于原目标 $\|w\|^2/2=0.4$，强对偶成立。
>
> 稀疏性：若把非支持向量 $(3,3)$ 挪到 $(10,10)$，它离间隔带更远，$\alpha$ 仍为 $0$，$w$ 不变；同理删掉 $(0,0)$ 也不改变解。这是支持向量“稀疏”的含义：多数训练点可以删掉而不改决策面。（原笔记此处写的是把 $(2,2)$ 挪到 $(10,10)$，但 $(2,2)$ 本身是支持向量，挪走后最近点对变为 $(3,3)$ 与 $(0,1)$，决策面会改变，故改为挪动非支持向量 $(3,3)$。）

<br>

## 实现

下面三个函数分别对应本章的三种视角：`linear_soft_svm_subgradient` 直接在原问题的 hinge 形式上做子梯度下降；`svm_dual_objective` 计算对偶目标，可用于检查 SMO 等对偶算法是否在单调上升；`predict_kernel_svm` 用对偶系数和核函数做预测。

子梯度推导：对单个样本，若功能间隔 $y^{(i)}(w^{\top}x^{(i)}+b)<1$，hinge 项关于 $w$ 的（子）梯度是 $-y^{(i)}x^{(i)}$，关于 $b$ 是 $-y^{(i)}$；否则为 $0$。加上正则项的梯度 $w$，即得代码中的 `grad_w` 与 `grad_b`。步长按 $1/\sqrt t$ 衰减，这是子梯度法保证收敛的常用选择（子梯度方向不一定是下降方向，固定步长会在最优点附近振荡）。

```python
def linear_soft_svm_subgradient(X, y, C=1.0, lr=0.01, num_steps=2000):
    """y in {-1, +1}。X 不含截距；b 单独更新。"""
    n, d = X.shape                      # X: (n, d)，y: (n,)
    w = np.zeros(d)
    b = 0.0
    for t in range(1, num_steps + 1):
        margin = y * (X @ w + b)        # 功能间隔，(n,)
        viol = margin < 1.0             # 违反间隔的点才贡献 hinge 梯度
        grad_w = w - C * (y[viol, None] * X[viol]).sum(axis=0)
        grad_b = -C * y[viol].sum()
        step = lr / np.sqrt(t)          # 子梯度法的递减步长
        w -= step * grad_w
        b -= step * grad_b
    return w, b

def svm_dual_objective(alpha, y, K):
    """对偶目标，越大越好。"""
    # (y_i y_j K_ij) 构成的矩阵，再做二次型 alpha^T Q alpha
    return float(alpha.sum() - 0.5 * alpha @ ((y[:, None] * K * y[None, :]) @ alpha))

def predict_kernel_svm(X_train, y, alpha, b, x, kernel_fn):
    k = kernel_fn(X_train, x[None, :]).ravel()   # 新点与每个训练点的核值，(n,)
    return float(np.sum(alpha * y * k) + b)      # 只有 alpha>0 的支持向量有贡献
```

子梯度法适合教学与线性核；生产环境的核 SVM 应使用 SMO 或 libsvm 一类分解算法，并缓存核列。

> [!NOTE]+ 对照：逻辑回归与 SVM
>
> 二者都能给出线性（或核化后的线性）边界。逻辑回归输出校准概率，所有点都拉 $\theta$，线性可分时 $\|\theta\|$ 无界（不加正则会一直增大）。SVM 用 hinge，间隔外的正确点梯度为 $0$，解由支持向量决定。需要 $P(y=1\mid x)$ 时用逻辑回归或对 SVM 另做校准；需要稀疏对偶与核时用 SVM。

<br>

> [!NOTE]+ 小结与衔接
>
> - 几何间隔 $\gamma=\hat\gamma/\|w\|$ 对尺度不变；固定 $\hat\gamma=1$ 后，最大间隔等价于 $\min\frac12\|w\|^2$。
> - 对偶问题只含样本内积，换成 $K_{ij}$ 即为核 SVM；互补松弛使解只依赖支持向量。
> - 软间隔引入 $\xi_i$ 与 $C$，对偶只多出盒约束 $0\le\alpha_i\le C$，等价于 L2 正则 + hinge 损失。
> - SMO 每次解析更新一对 $\alpha$，是核 SVM 的标准求解方法。
>
> 核 SVM 用光滑的 $\phi$ 弯曲边界，代价是 $n\times n$ 的核矩阵。下一章改用轴对齐切分：不写内积，也能表示非线性规则，但单棵深树方差大。

<br>

# 决策树（Decision Trees）

线性模型和核 SVM 都通过某种“内积”来做决策：$\theta^{\top}x$ 或 $\sum_i\alpha_iy^{(i)}K(x^{(i)},x)$。但很多实际问题更像一张规则清单：“如果收入低于某值且年龄小于某值，则大概率违约”。这种规则天然是按单个特征设阈值、层层细分的，而且常见于含有类别特征、缺失值、量纲各异的表格数据。决策树（decision tree）正是把这种规则清单自动学出来的模型。

核心想法一句话：把输入空间递归地切成轴对齐区域（axis-aligned regions），每一刀只看一个特征是否超过某个阈值，最后在每个叶节点放一个预测（分类时放类别或类别比例，回归时放均值）。每一步贪心地选择“让叶内样本最纯”的那一刀。本章讨论如何用信息增益或 Gini 指数选择切分，以及如何控制树的深度以防过拟合。输入是一张特征表，输出是一棵树——它也是下一章集成方法（提升）的基本构件。决策树是 2025 Autumn 的课堂主题，CS229 主讲义 PDF 不单列一章（往年有单独的决策树讲义）。

与前后章节的关系：上一章的核 SVM 用光滑的间隔边界切开空间；本章用轴对齐矩形的拼接来逼近同一类决策，完全不写 $\theta^{\top}x$。单棵深树的方差很大，下一章会把它削弱成很浅的树，作为弱学习器一棵棵加起来。

> [!NOTE]+ 决策树的定位
>
> 决策树用轴对齐切分把输入空间递归分成矩形，叶上放类别或均值。它的来源是规则清单以及 CART / ID3 等算法：每次选让不纯度下降最多的一刀。
>
> - 相对线性或核方法：不写 $\theta^{\top}x$，对特征的单调变换不敏感（切分只依赖样本的排序），能混用类别特征与缺失值；但斜边界只能用多层阶梯逼近。
> - 相对后面的提升：单棵深树方差大，训练误差可以降到 $0$，测试误差却通常随深度增加先降后升（过深时上升）。

<br>

## 切分准则：不纯度与信息增益

### 一次切分是什么

一次切分（split）选择一个特征 $j$ 与一个阈值 $t$（连续特征），或者一个取值子集（离散/类别特征），把当前节点的数据集 $S$ 分成左右两部分：记为 $S_L,S_R$，例如 $S_L=\{x\in S:x_j<t\}$，$S_R=\{x\in S:x_j\ge t\}$。建树就是从根节点开始，对每个节点重复“选最好的一刀、再递归处理两个子节点”，直到满足停止条件。

### 不纯度（impurity）

“最好的一刀”需要一个衡量节点“纯不纯”的指标。记 $\hat p_c$ 为节点 $S$ 中类别 $c$ 的样本比例。分类问题常用两种不纯度：Gini 指数与熵（entropy），

$$
\mathrm{Gini}(S)=\sum_{c}\hat p_c(1-\hat p_c),\qquad
H(S)=-\sum_{c}\hat p_c\log\hat p_c.
$$

二者都在节点只含一个类时取 $0$，在各类均匀混合时最大（二分类时 Gini 最大为 $0.5$，以 2 为底的熵最大为 $1$ bit）。Gini 也可以写成 $1-\sum_c\hat p_c^{2}$，下文代码就用这个形式。

### 信息增益（information gain）

信息增益是切分前的不纯度减去切分后两个子节点按样本数加权的不纯度：

$$
\mathrm{Gain}=I(S)-\Big(\frac{|S_L|}{|S|}I(S_L)+\frac{|S_R|}{|S|}I(S_R)\Big),
$$

其中 $I$ 可以取 $H$（狭义的信息增益）或 Gini。贪心算法每一步枚举所有特征与候选阈值，选增益最大的切分。回归树则用叶内平方误差（即叶内方差乘样本数）作为不纯度，叶节点预测取叶内 $y$ 的均值。

### 为什么不直接用错分率

最终关心的是错分率（misclassification loss）$1-\max_c\hat p_c$，但它不适合用来选切分。CS229 决策树讲义的解释是：错分率不是严格凹函数，很多确实有用的切分在错分率上看不出任何改进。例如父节点有 $900$ 个正例、$100$ 个负例（错分 $100$ 个）；切成 $(700^{+},100^{-})$ 与 $(200^{+},0^{-})$ 后，两个子节点都预测正类，错分仍然是 $100$ 个，增益为 $0$。但第二个子节点已经完全纯了，这显然是一次好切分。改用熵：父节点约 $0.469$ bit，子节点加权后约 $0.8\times0.544+0.2\times0=0.435$ bit，增益为正；改用 Gini：从 $0.18$ 降到 $0.8\times0.219\approx0.175$，增益同样为正。熵与 Gini 都是严格凹的，只要两个子节点的类别比例不同，加权后的不纯度就严格小于父节点，所以建树时用它们。

> [!EXAMPLE]+ 例9 一次切分
>
> 特征“年龄”，标签是否违约。阈值 $35$ 把样本分成 $20$ 与 $30$ 人，Gini 从 $0.5$ 降到 $0.32$，信息增益（Gini 下降量）是 $0.5-0.32=0.18$。若另一特征“收入”的增益为 $0.25$，根节点选收入。
>
> 需要注意：增益大不自动等于因果。收入可能只是职业的代理变量，树只是发现了它与标签的相关性。

<br>

## 停止条件、剪枝与实际问题

### 何时停止生长

如果不加限制，树可以一直切到每个叶只剩一个样本（或只剩同类样本），训练误差到 $0$——相当于把训练集记住了，测试误差通常更差。常用的停止条件有：深度上限、叶节点最小样本数、增益下限（增益小于阈值就不再切）。

### 剪枝

另一种做法是先长一棵很深的树，再往回剪。代价复杂度剪枝（cost-complexity pruning）在“叶数”（模型复杂度）与“训练不纯度”（拟合程度）之间折中，折中系数用验证集选取：剪掉那些对验证误差没有帮助的子树。

### 缺失值与类别不平衡

缺失值可以用代理切分（surrogate split，即找一个与主切分结果最相似的备用特征，在主特征缺失时使用）处理，也可以把“缺失”当作该特征的一个单独水平。类别不平衡时，叶节点的预测应报告类别概率（叶内比例），或者按错分代价加权，而不是只报多数类——否则少数类可能在每个叶里都被多数类淹没。

### 轴对齐的局限

轴对齐切分无法用一次切分切开沿斜对角分布的两类，例如“$x_1+x_2>1$”这条斜线，只能用多层阶梯来近似，需要很多次切分。这与线性/核分类器恰好互补：树擅长规则清单与异质表格，不擅长光滑的斜边界。

## 实现

下面的 `gini` 计算一个节点的 Gini 不纯度（类标需为非负整数，`np.bincount` 统计各类数目）；`best_threshold_split` 在一个连续特征上枚举所有相邻取值的中点作为阈值，返回使加权 Gini 最小的阈值。排序后只需扫描 $n-1$ 个候选位置；相等取值之间不能放阈值，所以跳过。

```python
def gini(y):
    if y.size == 0:
        return 0.0
    p = np.bincount(y).astype(float) / y.size   # 各类比例
    return float(1.0 - p @ p)                   # 1 - sum_c p_c^2

def best_threshold_split(x, y):
    """一维连续特征上的最优阈值。y 为非负整数类标。"""
    order = np.argsort(x)
    x, y = x[order], y[order]
    best, thr = np.inf, None
    for i in range(1, y.size):
        if x[i] == x[i - 1]:
            continue                            # 相同取值之间不能切
        # 左边 i 个样本、右边 n-i 个样本的加权 Gini
        impurity = (i * gini(y[:i]) + (y.size - i) * gini(y[i:])) / y.size
        if impurity < best:
            best, thr = impurity, 0.5 * (x[i - 1] + x[i])
    return thr, best
```

这个实现每个候选位置都重新计算两侧的 Gini，复杂度是 $O(n^2)$；实用实现会在扫描时增量更新各类计数，降到排序的 $O(n\log n)$。

> [!NOTE]+ 对照：核、树、线性
>
> 线性模型要把斜边界一次切成。核把斜边界搬到 $\phi$ 空间里变成线性的。树用多层阶梯逼近斜线，对单调变换不敏感，但也难以表示“$x_1+x_2>1$”这种简单倾斜。实践上，异质表格、缺失值、类别特征常先试树；光滑的几何关系常先试线性或核。

<br>

> [!NOTE]+ 小结与衔接
>
> - 决策树递归地做轴对齐切分，叶上放类别（比例）或均值。
> - 切分用 Gini 或熵的下降量（信息增益）贪心选择；错分率不够敏感，不适合用来选切分。
> - 用深度上限、叶最小样本数、增益下限或代价复杂度剪枝控制过拟合。
>
> 单棵树是强、高方差的假设。下一章把它削弱成桩（stump，只有一次切分的树），再按指数损失逐个加上去。加法模型与之后的神经网络是同一个思路：复杂的 $h$ 由简单模块叠加而成。

<br>

# 提升方法（Boosting）

假设手上有一个“弱”分类器（weak learner）：它只比随机猜测略好一点，例如一个只切一刀的决策桩，在某个数据集上加权错误率是 $0.45$。能否通过反复调用它，组合出一个错误率任意低的强分类器？提升（boosting）给出了肯定的回答。

核心想法一句话：按顺序训练一串弱分类器，每一轮都提高上一轮被分错样本的权重，迫使下一个弱分类器重点修补这些错误，最后按各自的可靠程度加权投票。AdaBoost（adaptive boosting）是最经典的版本，它按错分的指数损失（exponential loss）给样本重新加权。本章写出 AdaBoost 的权重更新规则，并从“加法模型 + 指数损失”的角度解释它从何而来；输入是一个弱学习器的训练接口，输出是加权投票的组合分类器。

与前后章节的关系：上一章的决策树（尤其是浅树、决策桩）正好可以当作第 $t$ 轮的 $h_t$。本章在指数损失下做前向分阶段极小化（forward stagewise minimization），训练集上的间隔被逐步拉大。下一章的神经网络把“逐个相加、加完就冻结的弱学习器”换成可微的层，用梯度同时更新全部模块。

## AdaBoost 算法

AdaBoost 维护一组样本权重 $w_i$（$i=1,\dots,n$），初始均匀（$w_i=1/n$）。第 $t$ 轮（$t=1,\dots,T$）：

1. 在当前权重 $w$ 下训练弱分类器 $h_t:\mathcal{X}\rightarrow\{-1,+1\}$，即让它尽量减小加权错误；
2. 计算加权错误率 $\varepsilon_t=\sum_i w_i\mathbf{1}\{h_t(x^{(i)})\ne y^{(i)}\}/\sum_i w_i$；
3. 计算该弱分类器的投票权重 $\alpha_t=\frac12\log\frac{1-\varepsilon_t}{\varepsilon_t}$；
4. 更新样本权重 $w_i\leftarrow w_i\exp(-\alpha_t y^{(i)}h_t(x^{(i)}))$，再归一化使 $\sum_iw_i=1$。

最终分类器为

$$
H(x)=\mathrm{sign}\Big(\sum_t\alpha_t h_t(x)\Big).
$$

逐条理解：当 $\varepsilon_t<1/2$（比随机好）时 $\alpha_t>0$，且错误率越低、$\alpha_t$ 越大，投票越有分量。第 4 步中，分对的点 $y^{(i)}h_t(x^{(i)})=+1$，权重乘以 $e^{-\alpha_t}<1$；分错的点 $y^{(i)}h_t(x^{(i)})=-1$，权重乘以 $e^{\alpha_t}>1$。于是被分错的点权重上升，下一棵树会更关注它们。

> [!NOTE]+ 提升方法的定位
>
> 提升把多个略优于随机的弱分类器加成一个强分类器，样本权重随错分而上升。AdaBoost 由 Freund 与 Schapire 提出，后来被证明等价于指数损失下的前向分阶段极小化。
>
> - 相对单棵深树：每步只加一个浅桩，训练误差有乘积形式的上界，方差通常更低。
> - 相对后面的梯度提升：指数损失对离群点过陡，可以换成 logistic 损失或 Huber 损失。
> - 弱学习器的加权错误率必须真的低于 $1/2$，否则 $\alpha_t\le 0$，加法没有收益。

<br>

### 训练误差的上界

AdaBoost 有一个干净的收敛保证。记第 $t$ 轮的“优势”为 $\gamma_t=\frac12-\varepsilon_t>0$，则组合分类器 $H$ 的训练误差满足

$$
\frac1n\sum_{i=1}^{n}\mathbf{1}\{H(x^{(i)})\ne y^{(i)}\}\le\prod_{t=1}^{T}2\sqrt{\varepsilon_t(1-\varepsilon_t)}=\prod_{t=1}^{T}\sqrt{1-4\gamma_t^{2}}\le\exp\Big(-2\sum_{t=1}^{T}\gamma_t^{2}\Big).
$$

推导思路是：$\mathbf{1}\{H(x)\ne y\}\le\exp(-y\sum_t\alpha_th_t(x))$（指数损失是 0-1 损失的上界），而每一轮最优 $\alpha_t$ 恰好让归一化常数 $Z_t=2\sqrt{\varepsilon_t(1-\varepsilon_t)}$。这意味着：只要每一轮都有一个固定的优势 $\gamma_t\ge\gamma>0$，训练误差就随轮数指数下降。这就是“弱学习器只需略好于随机”的含义。

> [!INFO]+ $\alpha_t=\frac12\log\frac{1-\varepsilon_t}{\varepsilon_t}$ 从哪来
>
> 把组合分类器看成加法模型 $f(x)=\sum_t\alpha_th_t(x)$，用指数损失 $L(f)=\sum_i\exp(-y^{(i)}f(x^{(i)}))$ 衡量它。第 $t$ 步保持前面的项不动，令 $f_t=f_{t-1}+\alpha h$，样本权重取 $w_i\propto\exp(-y^{(i)}f_{t-1}(x^{(i)}))$。于是
>
> $$
> L(f_{t-1}+\alpha h)=\sum_i w_i\exp(-\alpha y^{(i)}h(x^{(i)})).
> $$
>
> 由于 $y h\in\{+1,-1\}$，可以把分对与分错的样本拆开。令 $W=\sum_i w_i$，$\varepsilon=\sum_{i:yh=-1}w_i/W$，则
>
> $$
> L=W\big((1-\varepsilon)e^{-\alpha}+\varepsilon e^{\alpha}\big).
> $$
>
> 这个式子还说明，对固定的 $\alpha>0$，最好的 $h$ 就是加权错误率 $\varepsilon$ 最小的那个，这正是算法第 1 步“在权重 $w$ 下训练弱分类器”。再对 $\alpha$ 求导并令为零：$(1-\varepsilon)e^{-\alpha}=\varepsilon e^{\alpha}$，即 $e^{2\alpha}=(1-\varepsilon)/\varepsilon$，两边取对数得 $\alpha=\frac12\log\frac{1-\varepsilon}{\varepsilon}$。权重更新 $w_i\leftarrow w_i\exp(-\alpha y^{(i)}h(x^{(i)}))$ 正是把新的指数因子乘进 $w$，得到下一轮的 $\exp(-y^{(i)}f_t(x^{(i)}))$。
>
> 感性理解：$\alpha_t$ 是该轮弱分类器在加法模型中的最优步长，随 $\varepsilon_t$ 减小而增大。$\varepsilon_t=1/2$ 时 $\alpha_t=0$；$\varepsilon_t>1/2$ 时 $\alpha_t<0$，等价于翻转 $h_t$ 的符号。错分样本的权重乘 $e^{\alpha_t}>1$，下一轮拟合时这些点占更大比例。需要注意的坑：指数损失在负间隔上比 hinge 更陡，噪声点（标签错误的点）的权重会持续上升，因此提升轮数需要早停，或者缩小步长。

<br>

## 梯度提升（Gradient Boosting）

AdaBoost 绑定了指数损失。梯度提升把同一个“前向分阶段加法模型”推广到任意可微损失 $\mathcal{L}(y,f)$：每一步不再重新加权样本，而是让新的弱学习器去拟合损失对当前预测的负梯度，相当于在“函数空间”里做梯度下降。梯度提升回归树（gradient boosted regression trees，GBRT）第 $t$ 步先计算伪残差（pseudo-residual）

$$
r_i^{(t)}=-\frac{\partial}{\partial f(x^{(i)})}\mathcal{L}\big(y^{(i)},f_{t-1}(x^{(i)})\big),
$$

然后用一棵回归树 $h_t$ 拟合 $\{(x^{(i)},r_i^{(t)})\}$，再用学习率（收缩系数）$\nu\in(0,1]$ 作 $f_t=f_{t-1}+\nu h_t$。对平方损失 $\mathcal{L}=\frac12(y-f)^2$，伪残差就是普通残差 $y^{(i)}-f_{t-1}(x^{(i)})$，“拟合残差”因此得名；换成 logistic 或 Huber 损失，就得到对离群点更稳健的版本。

实际经验：$\nu$ 小、树多，通常优于 $\nu=1$、树少（代价是训练更慢）。列采样（每棵树只用部分特征）与子采样（每棵树只用部分样本）是随机性正则，能进一步降低方差。XGBoost、LightGBM 等库都是这一框架的高效实现。

## 实现

`adaboost_stumps` 用一维阈值桩作为弱学习器：对每个特征先用上一章的 `best_threshold_split` 找阈值，再尝试两个方向 $s=\pm1$（“$x_j\ge$ 阈值时预测 $s$”），选加权错误率最小的桩。`err` 被截断到 $[10^{-12},1-10^{-12}]$，避免完美分类时 $\log(1/0)$ 发散。

```python
def adaboost_stumps(X, y, num_rounds=20):
    """y in {-1, +1}。弱学习器为一维阈值桩。"""
    n, d = X.shape
    w = np.ones(n) / n                  # 样本权重，初始均匀
    stumps, alphas = [], []
    for _ in range(num_rounds):
        best = None
        for j in range(d):
            thr, _ = best_threshold_split(X[:, j], (y > 0).astype(int))
            if thr is None:
                continue
            for s in (+1, -1):          # 两个方向
                pred = np.where(X[:, j] >= thr, s, -s)
                err = float(np.sum(w * (pred != y)))   # 加权错误率（w 已归一化）
                if best is None or err < best[0]:
                    best = (err, j, thr, s, pred)
        err, j, thr, s, pred = best
        err = min(max(err, 1e-12), 1.0 - 1e-12)        # 防止 log(0)
        alpha = 0.5 * np.log((1.0 - err) / err)
        w *= np.exp(-alpha * y * pred)  # 错分点权重乘 e^alpha
        w /= w.sum()
        stumps.append((j, thr, s))
        alphas.append(alpha)
    return stumps, np.asarray(alphas)

def adaboost_predict(X, stumps, alphas):
    scores = np.zeros(X.shape[0])
    for (j, thr, s), a in zip(stumps, alphas):
        scores += a * np.where(X[:, j] >= thr, s, -s)
    return np.sign(scores), scores      # 符号是预测，scores 是组合间隔
```

需要注意一个实现细节：`best_threshold_split` 计算的是不加权的 Gini，不使用样本权重 `w`，所以每个特征的阈值在各轮之间都是同一个，权重只影响“选哪个特征、哪个方向”。这样候选桩只有 $2d$ 个，提升的能力受限。更标准的做法是在每一轮按加权错误率直接搜索阈值，例如下面的函数（排序后增量更新错误率，一次扫描完成）。它直接返回 `(err, thr, s)`，替换时在 `for j` 循环里调用 `best_weighted_stump(X[:, j], y, w)`、保留错误率最小的 `(err, j, thr, s)`，再用 `np.where(X[:, j] >= thr, s, -s)` 算出 `pred` 即可，内层的 `for s` 循环可以省去：

```python
def best_weighted_stump(x, y, w):
    """一维特征上按加权错误率搜索阈值与方向。y in {-1,+1}，w 非负且和为 1。"""
    order = np.argsort(x)
    x, y, w = x[order], y[order], w[order]
    err = float(np.sum(w[y < 0]))       # 阈值在最左：全部预测 +1，错的是负类
    best = (min(err, 1.0 - err), -np.inf, +1 if err <= 0.5 else -1)
    for i in range(1, x.size):
        # 第 i-1 个点移到阈值左侧（规则 s=+1 下预测 -1）
        err += w[i - 1] if y[i - 1] > 0 else -w[i - 1]
        if x[i] == x[i - 1]:
            continue
        s = +1 if err <= 0.5 else -1    # s=-1 时错误率为 1-err
        cand = min(err, 1.0 - err)
        if cand < best[0]:
            best = (cand, 0.5 * (x[i - 1] + x[i]), s)
    return best                          # (加权错误率, 阈值, 方向 s)
```

> [!EXAMPLE]+ 例10 弱学习器必须“弱而有效”
>
> 若桩的加权错误率稳定在 $0.49$，每轮优势只有 $\gamma_t=0.01$，但二十轮后组合间隔仍可分开简单的二维两类。需要注意，这只是经验现象，而不是上界保证的结果：代入训练误差上界，$20$ 轮后只得到 $\exp(-2\times20\times0.01^{2})=e^{-0.004}\approx0.996$，几乎没有约束力；要让上界降到 $1/n$ 以下（从而训练误差为 $0$），需要约 $\ln n/(2\gamma^{2})$ 轮，即 $\gamma=0.01$ 时约 $5000\ln n$ 轮。若桩的错误率是 $0.5$，则 $\alpha_t=0$，权重不再变化，提升停住。反过来，若第一棵树已经把训练误差降到 $0$，后续权重不再携带信息，结果等价于单棵树。弱学习器太强会失去“逐步修补”的意义。

<br>

> [!NOTE]+ 小结与衔接
>
> - AdaBoost 每轮在加权数据上训练弱分类器，按 $\alpha_t=\frac12\log\frac{1-\varepsilon_t}{\varepsilon_t}$ 投票，并放大错分样本的权重。
> - 它等价于指数损失下的前向分阶段加法建模；训练误差以 $\exp(-2\sum_t\gamma_t^2)$ 的速度下降。
> - 梯度提升把这一框架推广到任意可微损失：每步拟合负梯度，配合小学习率与子采样。
>
> 提升按阶段把 $h_t$ 冻住再加下一棵。下一章把所有模块同时写成 $g(Wx+b)$，用一条链式法则一起更新。非线性的来源从“切分”换成“逐点激活”。

<br>

# 神经网络与反向传播（Neural Networks）

前面几章得到非线性模型的方式各不相同：核方法把非线性放进特征映射 $\phi$，决策树放进切分，提升方法把很多简单模块逐个相加。神经网络的做法是：把仿射变换（affine map）$Wx+b$ 与逐点非线性函数 $g$ 交替堆叠，组成 $h_{\theta}(x)$，用它去逼近任意的非线性关系；所有参数 $\theta$（所有层的 $W,b$）通过梯度下降一起训练。

使这一切可行的算法是反向传播（backpropagation）：用链式法则，把损失对各中间量的偏导 $\partial J/\partial(\cdot)$ 从输出层一层层送回到每个模块，一次扫描就得到全部参数的梯度。本章写清一层的前向与反向计算，以及整个网络的向量化实现：输入一批样本 $X$，输出预测与所有参数的梯度。

与前后章节的关系：核把非线性放进 $\phi$，树放进切分，本章放进激活函数 $g(z)$ 并用反向传播训练。需要强调的是，如果没有非线性，多层仿射变换的复合仍是一个仿射变换，等价于一层。下一章（泛化）会解释：训练损失小不等于期望风险小。

## 网络结构：仿射层与激活函数

### 一层的计算

一层网络做两件事：先做仿射变换，再逐元素套一个激活函数：

$$
z=Wx+b,\qquad a=g(z).
$$

若输入 $x\in\mathbb{R}^{n^{[l-1]}}$、本层有 $n^{[l]}$ 个神经元，则 $W\in\mathbb{R}^{n^{[l]}\times n^{[l-1]}}$，$b,z,a\in\mathbb{R}^{n^{[l]}}$。$g$ 常用 ReLU $g(z)=\max(0,z)$、sigmoid $\sigma(z)=1/(1+e^{-z})$、tanh。ReLU 在正半轴梯度恒为 $1$，不易出现 sigmoid/tanh 在饱和区梯度消失的问题，是现代网络的默认选择。

### 两层网络与输出层

两层网络（一个隐层）写成

$$
a^{[1]}=g(W^{[1]}x+b^{[1]}),\qquad
\hat y=g_{\mathrm{out}}(W^{[2]}a^{[1]}+b^{[2]}).
$$

方括号上标 $[l]$ 表示第 $l$ 层。输出层的激活 $g_{\mathrm{out}}$ 由任务决定：二分类用 $g_{\mathrm{out}}=\sigma$（输出 $P(y=1\mid x)$），多类用 $g_{\mathrm{out}}=\mathrm{softmax}$，回归时 $g_{\mathrm{out}}$ 为恒等。这与广义线性模型一章的对应关系完全一致：网络的最后一层就是一个 GLM，前面各层相当于“学出来的特征映射”。

参数量约为 $\sum_l n^{[l]}(n^{[l-1]}+1)$：每层有 $n^{[l]}\times n^{[l-1]}$ 个权重加 $n^{[l]}$ 个偏置。

### 表达能力

万能逼近定理（universal approximation）说：足够宽的单隐层网络可以以任意精度逼近紧集上的连续函数。但这个结论有两点保留：所需宽度可能是指数级的；而且它只说明“存在”这样的参数，训练算法并不保证能找到该逼近。

> [!NOTE]+ 多层网络的定位
>
> 多层网络让仿射与逐点非线性交替堆叠，用反向传播同时更新全部层。它的来源是感知机加上可微激活与链式法则，并不是核或树的改写。
>
> - 相对核方法：非线性在 $g$ 里，不形成 $n\times n$ 矩阵，预测时也不必回看训练点。
> - 相对提升：模块可微，全部参数一起沿梯度更新，而不是逐个训练、冻结再相加。
> - 没有非线性时，多层仿射会塌缩成一层，深度本身不带来表达力。

<br>

现代网络的模块还包括 Dropout、BatchNorm、残差连接 $y=F(x)+x$、注意力等。CS229 的要求是：能把这些模块各自的局部 Jacobian 接到总损失上（即写出它们的反向传播）；卷积与 Transformer 的尺寸公式在 TA 讲座与后面的大语言模型相关章节展开。

## 反向传播（Backpropagation）

训练需要 $J$ 对每一层 $W^{[l]},b^{[l]}$ 的梯度。网络是一长串函数的复合，直接写出总公式既繁琐又容易出错。反向传播的思路是：每个模块只负责“已知损失对本模块输出的梯度，算出损失对本模块输入和本模块参数的梯度”，然后从输出层开始依次往回传。本节先给出两类基本模块（仿射层、逐点激活）的局部规则，再给出整批数据的矩阵形式。

### 仿射层的梯度

令 $J$ 为（平均）损失。对单样本的仿射变换 $z=Wx+b$，已知上游梯度 $\partial J/\partial z$（与 $z$ 同形），有

$$
\frac{\partial J}{\partial W}=\frac{\partial J}{\partial z}\,x^{\top},\qquad
\frac{\partial J}{\partial b}=\frac{\partial J}{\partial z},\qquad
\frac{\partial J}{\partial x}=W^{\top}\frac{\partial J}{\partial z}.
$$

理由：$z_k=\sum_j W_{kj}x_j+b_k$，所以 $\partial z_k/\partial W_{kj}=x_j$，得到 $\partial J/\partial W_{kj}=(\partial J/\partial z_k)\,x_j$，写成矩阵就是外积 $\frac{\partial J}{\partial z}x^{\top}$；又 $\partial z_k/\partial x_j=W_{kj}$，对 $k$ 求和得 $\sum_k W_{kj}\,\partial J/\partial z_k$，即 $W^{\top}\frac{\partial J}{\partial z}$。检查形状：$\partial J/\partial W$ 为 $(n^{[l]},1)\times(1,n^{[l-1]})$，与 $W$ 同形。

### 逐点激活的梯度

对逐点激活 $a=g(z)$，每个 $a_k$ 只依赖 $z_k$，Jacobian 是对角阵，所以链式法则化为逐元素乘法：

$$
\frac{\partial J}{\partial z}=\frac{\partial J}{\partial a}\odot g'(z).
$$

对 ReLU，$g'(z)=\mathbf{1}\{z>0\}$；对 sigmoid，$g'(z)=\sigma(z)(1-\sigma(z))$。

### 输出层的简化

sigmoid + 交叉熵，或 softmax + 交叉熵，在输出层可以合并求导，结果非常简单：$\partial J/\partial z=\hat y-y$（$y$ 是 one-hot 向量或标量）。以二分类为例：$J=-y\log\hat y-(1-y)\log(1-\hat y)$，$\hat y=\sigma(z)$，则 $\partial J/\partial\hat y=\frac{\hat y-y}{\hat y(1-\hat y)}$，乘以 $\sigma'(z)=\hat y(1-\hat y)$ 恰好约掉分母，得到 $\hat y-y$。这与逻辑回归梯度的形式一致。常见坑：用了这个合并后的公式，就不要再乘一次 $g'$。

### 向量化：整批数据

实际训练时一次处理 $n$ 个样本。有两种排布：令 $X$ 为 $(d,n)$（每列一个样本，CS230 的写法），或按 CS229 代码常用的 batch 在前，$X$ 为 $(n,d)$（每行一个样本）。两种排布下公式的乘法顺序不同。

列排布（$Z=WA+b$，$W$ 左乘激活）时，关键是

$$
\mathrm{d}W=\mathrm{d}Z\,A^{\top}/n,\qquad
\mathrm{d}A=W^{\top}\mathrm{d}Z.
$$

其中 $\mathrm{d}Z$ 表示每个样本各自的 $\partial\ell/\partial z$ 并排成的矩阵，$A$ 是本层输入（上一层激活）；对样本求和的过程藏在矩阵乘法里，除以 $n$ 得到平均损失的梯度。行排布（$Z=A_{\mathrm{prev}}W+b$，$W$ 右乘激活，下文代码采用）时，对应为 $\mathrm{d}W=A_{\mathrm{prev}}^{\top}\mathrm{d}Z/n$、$\mathrm{d}A_{\mathrm{prev}}=\mathrm{d}Z\,W^{\top}$。

实现时在每个数组旁写上 shape，并用 `assert dW.shape == W.shape` 检查：梯度必须与参数同形，这是最便宜的查错手段。

> [!INFO]+ 链式法则只需局部 Jacobian
>
> $J$ 对 $W^{[1]}$ 的依赖路径是 $W^{[1]}\rightarrow z^{[1]}\rightarrow a^{[1]}\rightarrow\cdots\rightarrow J$。不必一次写出总公式，每个模块只实现 `backward(d_out, cache) -> d_in, d_params`。这与自动微分的反向模式（reverse-mode）相同：损失是标量、参数是高维时，反向只需一次扫描就得到全部梯度，代价与一次前向同阶。漏乘 $g'$、忘记除以 $n$、把 $A^{\top}$ 写成 $A$，是三种最常见的实现错误。
>
> 矩阵情形：$\mathrm{d}Z=\mathrm{d}A\odot g'(Z)$。列排布下 $\mathrm{d}W=\mathrm{d}Z\,A_{\mathrm{prev}}^{\top}/n$；行排布（$Z=A_{\mathrm{prev}}W$，$A_{\mathrm{prev}}$ 为 $(n,n^{[l-1]})$，$W$ 为 $(n^{[l-1]},n^{[l]})$）下 $\mathrm{d}W=A_{\mathrm{prev}}^{\top}\mathrm{d}Z/n$，形状为 $(n^{[l-1]},n)\times(n,n^{[l]})=(n^{[l-1]},n^{[l]})$；传给上一层的梯度 $\mathrm{d}A_{\mathrm{prev}}=\mathrm{d}Z\,W^{\top}$ 必须满足 $(n,n^{[l-1]})=(n,n^{[l]})\times(n^{[l]},n^{[l-1]})$ 的右乘。（原笔记把行排布下的 $\mathrm{d}W$ 写成 $\mathrm{d}Z\,A_{\mathrm{prev}}^{\top}$，并把上式左边写成 $(n,n^{[l]})$，两处混用了两种排布，这里按形状核对后改正；下文代码中的 `A1.T @ dZ2` 与改正后的公式一致。）
>
> 感性理解：反向模式把标量 $J$ 对中间量的偏导从输出层传回输入。每个模块只需要局部 Jacobian：上游传来的 $\mathrm{d}_{\mathrm{out}}$ 左乘（或右乘）权重的转置、或按元素乘 $g'$，再分配给输入与参数，乘法顺序由矩阵形状唯一决定。各模块局部正确，则总梯度正确，不必展开整条计算图。

<br>

### 初始化与学习率

初始化不能全零：若隐层权重全相同，隐层各神经元对称，前向输出相同、反向收到同一梯度，永远学成一样的特征，等于只有一个神经元。正确做法是随机初始化，并控制方差，使各层激活与梯度的尺度在前向、反向传播中大致保持不变：He 初始化对 ReLU 使用 $\mathrm{Var}(W)=2/n_{\mathrm{in}}$；Xavier（Glorot）初始化对 tanh、sigmoid 等饱和激活使用 $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$。

学习率过大，损失会发散并出现 `nan`；过小，则数百步损失几乎不动。调参时通常按对数尺度（如 $10^{-1},10^{-2},10^{-3}$）搜索。

## 实现

下面是一个单隐层 ReLU 网络的前向与反向（行排布：`X` 为 $(n,d)$，`W1` 为 $(d,h)$，`W2` 为 $(h,k)$）。`mlp_forward` 返回输出层的线性部分 `Z2`（损失与输出激活在外部计算），并缓存反向所需的中间量。`mlp_backward` 接收 `dZ2`，即每个样本的 $\partial\ell/\partial z^{[2]}$（对 sigmoid/softmax + 交叉熵就是 $\hat y-y$，形状 $(n,k)$），在计算参数梯度时除以 $n$ 得到平均损失的梯度。

```python
def relu(Z):
    return np.maximum(0.0, Z)

def mlp_forward(X, params):
    """X: (n, d)。params 含 W1,b1,W2,b2。"""
    Z1 = X @ params["W1"] + params["b1"]      # (n, h)
    A1 = relu(Z1)                             # (n, h)
    Z2 = A1 @ params["W2"] + params["b2"]     # (n, k)
    return Z2, (X, Z1, A1)

def mlp_backward(dZ2, caches, params):
    X, Z1, A1 = caches
    n = X.shape[0]
    dW2 = (A1.T @ dZ2) / n                    # (h, k)，与 W2 同形
    db2 = dZ2.sum(axis=0) / n                 # (k,)
    dA1 = dZ2 @ params["W2"].T                # (n, h)
    dZ1 = dA1 * (Z1 > 0)                      # ReLU 的导数是指示函数
    dW1 = (X.T @ dZ1) / n                     # (d, h)
    db1 = dZ1.sum(axis=0) / n                 # (h,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def he_init(d_in, d_out, rng):
    return rng.normal(scale=np.sqrt(2.0 / d_in), size=(d_in, d_out))   # Var = 2 / d_in
```

验证实现的标准办法是梯度检查（gradient checking）：对少量参数用中心差分 $\big(J(\theta+\epsilon e_k)-J(\theta-\epsilon e_k)\big)/(2\epsilon)$ 近似偏导，与 `mlp_backward` 的结果比较相对误差。

> [!EXAMPLE]+ 例11 异或
>
> 四点 $(\pm 1,\pm 1)$ 上的异或（两坐标符号不同为正类）不能被一条直线切开。单隐层两个 ReLU 单元可以把平面折成两块，再由输出层拼出异或。一种具体构造：$h_1=\mathrm{ReLU}(x_1-x_2)$，$h_2=\mathrm{ReLU}(x_2-x_1)$，输出 $\hat y=h_1+h_2-1$。在 $(1,-1)$ 与 $(-1,1)$ 上 $h_1+h_2=2$，$\hat y=1$；在 $(1,1)$ 与 $(-1,-1)$ 上 $h_1=h_2=0$，$\hat y=-1$，按符号即可完全分对（实际上 $h_1+h_2=|x_1-x_2|$）。
>
> 若隐层激活改成恒等，两层合成一层线性函数，四点仍分不开。此时训练损失有一个下界降不下去，这不是优化失败，而是假设类不够。

<br>

> [!NOTE]+ 对照：核、提升、网络
>
> 三条路都在放大假设类。核：特征维数上升，但优化通常仍是凸的。提升：使用凸的指数或 logistic 损失，弱学习器是离散的。网络：损失对 $\theta$ 非凸，但反向传播能把所有层一起更新。训练误差降到 $0$ 只说明假设类 $\mathcal{H}$ 足够大，或者优化走到了插值解；下一步必须讨论期望风险 $R$，而不是经验风险 $\hat R_n$。

<br>

> [!NOTE]+ 小结与衔接
>
> - 网络由仿射层与逐点激活交替组成；没有非线性时多层塌缩为一层。
> - 反向传播只需每个模块的局部规则：仿射层 $\mathrm{d}W=\mathrm{d}z\,x^{\top}$、$\mathrm{d}x=W^{\top}\mathrm{d}z$，激活层 $\mathrm{d}z=\mathrm{d}a\odot g'(z)$；sigmoid/softmax + 交叉熵在输出层合并为 $\hat y-y$。
> - 向量化时乘法顺序由排布与形状决定，用 `assert dW.shape == W.shape` 检查；随机初始化（He / Xavier）打破对称。
>
> 网络（以及此前所有模型）都能把训练集拟合到很低的损失。下一章把经验风险与期望风险拆开，并用偏差-方差分解、VC 维来描述“假设类有多丰富、需要多少样本”。

<br>

<br>

# 泛化：从训练误差到测试误差（Generalization）

前面各章（线性回归、逻辑回归、GLM、生成模型、核方法、SVM、树与提升、神经网络）做的其实是同一件事：在某个假设类里找一个让训练集上平均损失最小的模型。可是训练误差小并不等于对新样本误差小。一个能把训练集背下来的高次多项式，训练误差可以是 $0$，换一批数据却错得离谱。真正关心的是模型在“没见过的数据”上表现如何，这就是泛化（generalization）问题。

这一章把“训练集上的表现”和“总体分布上的表现”正式分开，并从三个角度刻画两者之间的缺口：偏差-方差分解（bias-variance decomposition）从“估计量随训练集波动”的角度看回归误差；双重下降（double descent）说明现代过参数化模型为什么不完全符合经典 U 形曲线；样本复杂度（sample complexity）与均匀收敛则从“假设类有多大”的角度，回答“需要多少样本才能让训练误差可信”。本章的输入是假设类（hypothesis class）$\mathcal{H}$ 与样本数 $n$，输出是对超额风险（excess risk）的定性理解与定量界。

本章只负责“诊断”：说明容量、样本数与风险的关系。下一章（正则化与模型选择）给出可操作的手段——惩罚项、早停与交叉验证——来控制有效容量。

## 期望风险与经验风险

### 两种风险的定义

设数据 $(x,y)$ 来自某个未知分布 $\mathcal{D}$，$\ell(\hat y,y)$ 是损失函数（例如平方损失或 0-1 损失），$h\in\mathcal{H}$ 是一个假设（一个具体的预测函数）。训练集是 $n$ 个独立同分布样本 $\{(x^{(i)},y^{(i)})\}_{i=1}^n$。定义期望风险（expected risk，也叫泛化误差）与经验风险（empirical risk，也叫训练误差）：

$$
R(h)=E_{(x,y)\sim\mathcal{D}}\big[\ell(h(x),y)\big],\qquad
\hat R_n(h)=\frac1n\sum_{i=1}^{n}\ell(h(x^{(i)}),y^{(i)}).
$$

$R(h)$ 是在整个分布上的平均损失，是一个确定的数，但因为 $\mathcal{D}$ 未知而无法直接计算；$\hat R_n(h)$ 是在 $n$ 个训练样本上的平均损失，可以算，但它是随机的（换一批训练集就变）。对固定的 $h$，$\hat R_n(h)$ 是 $R(h)$ 的无偏估计。

### ERM 与真正的目标

经验风险最小化（empirical risk minimization，ERM）在 $\mathcal{H}$ 上最小化 $\hat R_n$，得到 $\hat h=\arg\min_{h\in\mathcal{H}}\hat R_n(h)$。需要强调的是：关心的是 $R(\hat h)$，不是 $\hat R_n(\hat h)$。

这里有一个微妙之处：虽然对固定 $h$，$\hat R_n(h)$ 无偏，但 $\hat h$ 本身是根据同一批训练数据挑出来的——它恰好是让 $\hat R_n$ 最小的那个，所以 $\hat R_n(\hat h)$ 系统性地偏乐观，通常低于 $R(\hat h)$。$\mathcal{H}$ 越大、可挑的假设越多，这种“挑出来的乐观”越严重。这正是过拟合的数学根源。

衡量 ERM 好坏的标准是超额风险（excess risk）：$R(\hat h)-\min_{h\in\mathcal{H}}R(h)$，即学到的模型比假设类里最好的模型差多少。

> [!NOTE]+ 期望风险与经验风险
>
> $R$ 是未知分布 $\mathcal{D}$ 上的期望损失，$\hat R_n$ 是训练集上的平均损失。ERM 最小化后者，真正要控制的是前者。之所以要区分两者，是为了把“训练误差小”和“新样本上误差小”拆开，否则无法讨论过拟合。与只报告训练目标 $J$ 相比，这种区分强调：$\hat R_n$ 下降并不蕴含 $R$ 下降。下面的偏差-方差分解与均匀收敛，分别从“估计量的波动”和“假设类的大小”两条路来逼近这个缺口。

<br>

## 偏差-方差分解（Bias–Variance Decomposition）

偏差-方差分解回答这样一个问题：一个回归模型在测试点上的平方误差，究竟来自哪里？是模型“本来就画不出真实曲线”（例如用直线拟合抛物线），还是“模型太灵活，被训练集里的噪声带跑了”，还是“标签本身就有噪声，谁也预测不准”？把这三种来源分清楚，才能决定下一步该换模型、加数据，还是加正则。

### 设定与分解公式

考虑回归问题，使用平方损失。假设数据满足

$$
y=f(x)+\varepsilon,\qquad E[\varepsilon\mid x]=0,\qquad \mathrm{Var}(\varepsilon)=\sigma^2,
$$

其中 $f$ 是未知的真实函数，$\varepsilon$ 是标签噪声。训练集是随机抽取的，所以在训练集上学到的模型 $\hat f$ 也是随机的。对一个固定的测试输入 $x$，把模型输出 $\hat f(x)$ 看成“训练集的随机变量”，测试标签 $y$ 是一个与训练集独立的新样本。则测试平方误差的期望（同时对训练集与测试噪声取期望）可以分解为：

$$
E\big[(y-\hat f(x))^{2}\big]
=
\big(f(x)-E[\hat f(x)]\big)^{2}
+E\big[(\hat f(x)-E[\hat f(x)])^{2}\big]
+\sigma^{2}.
$$

三项依次是：

- **偏差平方（bias²）**：$\big(f(x)-E[\hat f(x)]\big)^2$。设想用无数个不同的训练集各训练一次，再把这些模型取平均得到“平均模型” $E[\hat f(x)]$，偏差衡量平均模型离真实 $f(x)$ 有多远。它反映的是假设类本身的表达能力不足。
- **方差（variance）**：$E\big[(\hat f(x)-E[\hat f(x)])^{2}\big]$。衡量换一个训练集，模型的预测会跳动多少。它反映的是模型对训练数据中偶然因素的敏感程度。
- **不可约噪声（irreducible error）**：$\sigma^2$。即使 $\hat f=f$ 也消不掉，因为标签自身带有随机性。

直观例子：真实函数是抛物线时，用直线去拟合属于**高偏差**——不管换哪批训练集，直线都系统性地在同样的地方弯错；用五次多项式拟合少量样本属于**高方差**——多项式会随样本点剧烈跳动，预测乱晃；而无论模型多好，**不可约噪声**都留在那里。

> [!NOTE]+ 偏差与方差
>
> 这个分解把测试平方误差拆成三部分：平均模型离真实有多远、换训练集模型跳多远、标签自身的抖动。它来自对 $E[(y-\hat f)^{2}]$ 做代数展开，而不是另外假设了三种错误类型。与只说“过拟合”相比，这一分解可以指导下一步：偏差大就换更丰富的假设类（压偏差），方差大就加数据或加惩罚（压方差）。$\sigma^{2}$ 与模型无关，再复杂的 $\hat f$ 也消不掉。后面的双重下降并不推翻这三项，只说明“容量”不能单用参数个数来度量。

<br>

### 推导：平方误差怎样拆成三项

> [!INFO]+ 平方误差怎样拆成三项
>
> 固定 $x$，令 $\bar f=E[\hat f(x)]$（对训练集取期望），$y=f(x)+\varepsilon$，$E[\varepsilon]=0$，且测试噪声 $\varepsilon$ 与 $\hat f$ 独立（测试样本与训练集独立）。下面把 $f(x)$、$\hat f(x)$ 简写为 $f$、$\hat f$。
>
> 第一步，在括号里“加一项减一项” $\bar f$，把误差写成三段之和：
>
> $$
> y-\hat f=(f-\bar f)+(\bar f-\hat f)+\varepsilon .
> $$
>
> 第二步，平方展开：
>
> $$
> (y-\hat f)^{2}=(f+\varepsilon-\hat f)^{2}=(f-\bar f)^{2}+(\bar f-\hat f)^{2}+\varepsilon^{2}+2\text{交叉项}.
> $$
>
> 其中交叉项是 $(f-\bar f)(\bar f-\hat f)$、$(f-\bar f)\varepsilon$、$(\bar f-\hat f)\varepsilon$；后两项合起来就是 $\varepsilon(f-\hat f)$。
>
> 第三步，对训练集与 $\varepsilon$ 取期望，逐个消去交叉项：
>
> - $E[\varepsilon]=0$ 且 $\varepsilon$ 与 $\hat f$ 独立，使 $2\varepsilon(f-\hat f)$ 的期望为 $0$；
> - $f-\bar f$ 是常数（不依赖训练集），而 $E[\hat f-\bar f]=0$（由 $\bar f$ 的定义），使 $2(f-\bar f)(\bar f-\hat f)$ 的期望为 $0$；
> - $E[\varepsilon^2]=\sigma^2$。
>
> 剩下
>
> $$
> E[(y-\hat f)^{2}]=(f-\bar f)^{2}+E[(\hat f-\bar f)^{2}]+\sigma^{2}.
> $$
>
> 第一项是“平均模型离真实有多远”，第二项是“换一袋数据模型跳多远”，第三项是标签自身的抖动。
>
> 感性理解：对固定 $x$，把 $\hat f(x)$ 看作训练集的随机变量。$E[\hat f]$ 相对 $f$ 的偏移是偏差，$E[(\hat f-E[\hat f])^{2}]$ 是该估计随样本的波动。线性模型偏差大、方差小；高次多项式相反。增大 $n$ 主要压方差；更换假设类主要压偏差。$\sigma^{2}$ 来自标签噪声，与 $\hat f$ 无关，不能靠加复杂模型消去。

<br>

需要注意，这个分解是对**一个固定的 $x$** 做的；对整个测试分布的误差，只需再对 $x$ 取期望，三项各自变成“平均偏差平方”“平均方差”“$\sigma^2$”。另外，分解只对平方损失成立得这么干净；对 0-1 损失等其他损失，“偏差”“方差”只能作为定性概念使用。

### 学习曲线：在实践中看出偏差与方差

偏差和方差都涉及“对训练集取期望”，实际中无法直接算，但可以通过学习曲线（learning curve）间接观察：

- **固定模型，增大样本数 $n$**：训练误差上升（样本越多越难全部拟合），验证误差下降（模型被更多数据约束），两条曲线逐渐靠拢。两者之间的间隙是方差的经验影子；两条曲线最终靠拢到的高度则反映偏差加噪声。若两条曲线都停在较高水平且已贴得很近，说明是高偏差，再加数据也没用；若间隙仍然很大，说明是高方差，加数据有帮助。
- **固定 $n$，增大模型容量**：训练误差持续下降，验证误差先降后升，这就是经典的 U 形曲线。U 形左侧是欠拟合（偏差主导），右侧是过拟合（方差主导）。

## 双重下降（Double Descent）

经典 U 形曲线告诉人们“模型太大会过拟合”。可是现代深度网络的参数往往远多于样本数，能把训练集完全记住，测试误差却依然很低。双重下降现象正是为了调和这一矛盾：当横轴是模型规模时，测试误差先按 U 形下降再上升，在“刚好能插值训练集”的位置达到峰值，然后随着模型继续变大**再次下降**。

### 现象与解释

在插值阈值（interpolation threshold）附近——也就是模型参数量恰好能完全记住训练集的位置，例如线性回归中参数个数 $d\approx n$——测试误差往往出现峰值；再增大模型，测试误差可以再次下降。CS229 讲义中还指出了“样本维度上的双重下降”：固定模型、改变样本数 $n$，当 $n$ 接近参数个数时测试误差同样会出现峰值，即“更多数据反而变差”的反常区间；适当调好的正则化可以显著削弱这个峰。

为什么过了阈值反而变好？原因在于：

1. **插值阈值附近解是“被逼出来的”**。此时满足“训练误差为零”的解几乎唯一，模型只能用很大、很扭曲的参数去穿过每一个带噪声的点，因而方差极大。
2. **过参数化区存在许多插值解**。参数远多于样本时，满足训练误差为零的解构成一个很大的集合，算法挑哪一个就变得关键。
3. <strong>梯度下降的隐式偏置（implicit bias）</strong>会挑出更平滑的那一条。例如在过参数化线性回归中，从 $\theta=0$ 出发的梯度下降收敛到所有插值解中 $\|\theta\|_2$ 最小的那个（最小范数解），它往往比阈值处的“唯一解”平滑得多。

因此，双重下降并不推翻偏差-方差分解，而是说“容量”不能只用参数个数一条轴来衡量；有效容量还依赖优化路径与数据谱（数据协方差矩阵特征值的分布）。实践上的含义是：若测试误差恰好落在插值阈值附近的峰值上，加宽网络、加强正则或增加样本，都有可能离开该区域。

> [!NOTE]+ 双重下降
>
> 测试误差在模型刚能插值训练集时升高，过了该阈值后再降。这一现象源自 Belkin 等人把现代过参数化实践与经典 U 形曲线进行对照，并不是新的损失定义。与经典偏差-方差图相比，参数个数不再是容量的唯一坐标，优化路径的隐式偏置会在许多插值解里挑出一条更平滑的。与“模型越大一定越过拟合”的直觉相比，在过参数化区加宽网络有时比停在插值阈值处更稳。但这仍要用验证集确认，不能把“再次下降”当成定理。

<br>

## 样本复杂度与均匀收敛（Sample Complexity）

偏差-方差从“估计量波动”的角度看泛化，而学习理论换了一个更一般的角度：给定假设类 $\mathcal{H}$，需要多少样本，才能保证训练误差是真实误差的可靠代理？这个“需要的样本数”就叫样本复杂度。

### 目标：均匀收敛

ERM 的危险在于“从很多假设中挑出训练误差最小的那个”。只证明“某一个固定的 $h$ 的 $\hat R_n(h)$ 接近 $R(h)$”是不够的，因为 $\hat h$ 是看过数据后才挑出来的。所以目标更强：以高概率对**所有** $h\in\mathcal{H}$ **同时**有

$$
|\hat R_n(h)-R(h)|\le\varepsilon ,
$$

这叫均匀收敛（uniform convergence）。一旦成立，ERM 挑出的 $\hat h$ 自然也满足这个不等式，从而超额风险也小（推导见下文）。

### 有限假设类：Hoeffding 不等式 + 并界

令 $|\mathcal{H}|=m$，即假设类只包含有限 $m$ 个假设，并假定损失取值在 $[0,1]$ 内（例如 0-1 损失）。

**单个假设。** 对一个固定的 $h$，$\hat R_n(h)$ 是 $n$ 个独立、取值在 $[0,1]$ 的随机变量 $\ell(h(x^{(i)}),y^{(i)})$ 的均值，其期望正是 $R(h)$。Hoeffding 不等式给出

$$
P\big(|\hat R_n(h)-R(h)|>\varepsilon\big)\le 2e^{-2n\varepsilon^{2}}
$$

（损失在 $[0,1]$ 时）。也就是说，样本均值偏离期望超过 $\varepsilon$ 的概率随 $n$ 指数下降。

**全部假设。** 对 $m$ 个假设做并界（union bound，即“若干事件中至少发生一个的概率不超过各自概率之和”），再令失败概率 $\le\delta$，得到需要的样本数

$$
n\ge\frac{1}{2\varepsilon^{2}}\log\frac{2m}{\delta}.
$$

这个结论最重要的特点是：样本数只随 $\log|\mathcal{H}|$ 增长，而不是随 $|\mathcal{H}|$ 线性增长。这解释了“多试几个模型”为什么代价不大，但这份对数代价必须在验证集上付，而不是在测试集上付。

> [!INFO]+ 从 Hoeffding 到 $n\ge\frac1{2\varepsilon^{2}}\log\frac{2m}{\delta}$
>
> 损失落在 $[0,1]$ 时，单条假设的经验风险是 $n$ 个有界随机变量的均值。Hoeffding 给出
>
> $$
> P\big(|\hat R_n(h)-R(h)|>\varepsilon\big)\le 2\exp(-2n\varepsilon^{2}).
> $$
>
> $\mathcal{H}$ 里有 $m$ 个假设，关心的是“至少有一个 $h$ 偏离超过 $\varepsilon$”这一坏事件。它是 $m$ 个单独坏事件的并集，用并界（union bound）把各项概率相加：
>
> $$
> P(\exists h:|\hat R_n-R|>\varepsilon)\le 2m\exp(-2n\varepsilon^{2}).
> $$
>
> 令右端 $\le\delta$，即 $2m\exp(-2n\varepsilon^2)\le\delta$，两边取对数：$2n\varepsilon^{2}\ge\log(2m/\delta)$，即 $n\ge\frac1{2\varepsilon^{2}}\log\frac{2m}{\delta}$。指数里的 $2$ 来自 Hoeffding 对 $[0,1]$ 变量的常数；换成其他有界区间只改常数，不改 $\log m$ 与 $1/\varepsilon^{2}$ 的形状。
>
> 感性理解：单条假设的 $\hat R_n$ 以指数速度集中到 $R$。同时控制 $|\mathcal{H}|=m$ 条假设时，失败事件要做并界，样本需求只增加 $\log m$。模型选择应在验证集上支付这项对数代价；测试集若再用来选模型，并界中的 $m$ 继续变大，$\varepsilon$ 不再对应所报数字。

<br>

### 从均匀收敛到 ERM 的超额风险

均匀收敛本身只说“训练误差可信”，还要再走一步才能说明 ERM 学到的模型接近最优。记 $h^*=\arg\min_{h\in\mathcal{H}}R(h)$ 为假设类中真实风险最小者。在均匀收敛事件（概率至少 $1-\delta$）上：

$$
R(\hat h)\le\hat R_n(\hat h)+\varepsilon\le\hat R_n(h^*)+\varepsilon\le R(h^*)+2\varepsilon .
$$

第一个不等号用均匀收敛作用于 $\hat h$；第二个用 ERM 的定义（$\hat h$ 的训练误差最小）；第三个用均匀收敛作用于 $h^*$。把 $\varepsilon$ 用 $n$ 解出来，得到 CS229 讲义中的形式：以至少 $1-\delta$ 的概率，

$$
R(\hat h)\le\min_{h\in\mathcal{H}}R(h)+2\sqrt{\frac{1}{2n}\log\frac{2m}{\delta}} .
$$

这个式子也给出了偏差-方差权衡的学习理论版本：换一个更大的 $\mathcal{H}$，第一项（类似“偏差”）只会变小，第二项（类似“方差”）因 $m$ 变大而变大。

### 无限假设类：VC 维

线性分类器、带连续阈值的分类器等假设类有无穷多个成员，$|\mathcal{H}|=\infty$ 时上面的并界直接失效（$\log m=\infty$）。需要一个能代替“个数”的有限容量度量，这就是 VC 维（Vapnik–Chervonenkis dimension）。

**打散（shattering）。** 称 $\mathcal{H}$ 打散一个点集 $S$，是指对 $S$ 中点的**每一种**可能的 $\pm$ 标记方式，$\mathcal{H}$ 中都存在一个分类器把它们全部分对。$d$ 个点共有 $2^{d}$ 种标记。

**VC 维。** 用 VC 维 $d_{\mathrm{VC}}$ 描述打散能力：它是 $\mathcal{H}$ 能实现所有 $2^{d}$ 种标记的最大点集大小。注意定义中是“存在某个大小为 $d$ 的点集能被打散”，而不是“所有大小为 $d$ 的点集都能被打散”。例如二维平面上的线性分类器能打散处于一般位置的 3 个点（三点共线时不行，但这不影响），却无法打散任何 4 个点（例如对角同号的“异或”标记），所以 VC 维是 3。一般地，线性分类器在 $\mathbb{R}^{d}$ 的 VC 维是 $d+1$（含截距）。

**均匀收敛界。** 对 VC 维有限的假设类，均匀收敛界的形状为

$$
R(h)\le\hat R_n(h)+O\Big(\sqrt{\frac{d_{\mathrm{VC}}\log n+\log(1/\delta)}{n}}\Big).
$$

CS229 讲义中写成更精确的形式 $O\big(\sqrt{\tfrac{d}{n}\log\tfrac{n}{d}+\tfrac1n\log\tfrac1\delta}\big)$，两者形状一致：VC 维扮演了有限情形中 $\log|\mathcal{H}|$ 的角色。由此可得一个经验法则：所需样本数大致与 VC 维成线性关系，对很多“合理”的模型而言，大致与参数个数成正比。

VC 维有限则一致可学；VC 维无限则存在分布，使任意大的 $n$ 仍可被“打穿”（训练误差与真实误差的差距无法一致控制）。需要注意，RBF-SVM 的假设类可以很丰富，其有效容量由核带宽与 $C$ 共同限制，不能只看“核对应的特征空间是无限维”就断言它必然过拟合。

> [!NOTE]+ VC 维
>
> VC 维是假设类能打散的最大点集大小，用来给无限 $\mathcal{H}$ 标一个有限的容量。它来自 Vapnik–Chervonenkis 的均匀收敛理论：有限类用 $\log|\mathcal{H}|$ 计量容量，无限类用打散能力代替计数。与 Hoeffding 加并界相比，$|\mathcal{H}|=\infty$ 时不能再对每个假设加一项。与参数个数相比，线性分类器的 VC 维是 $d+1$，这与过参数化的现代网络并不是同一尺度。界的常数通常很松，适合用来判断 $1/\sqrt{n}$ 与 $\log$ 的形状，而不是预报精确的测试误差。

<br>

> [!WARNING]+ 界是最坏情形
>
> PAC / VC 界给出的是充分条件，对任意分布都成立，因此常数往往很松。在真实图像数据上，深度网络的泛化远好于把参数个数代入 VC 界所得的预测。这些界用来判断论证方向（容量越大越难泛化、模型选择只付对数代价、误差按 $\sqrt{1/n}$ 收缩），不用来预报精确的测试误差。

<br>

### 数值例子

> [!EXAMPLE]+ 例12 有限假设类所需的样本数
>
> $\mathcal{H}$ 有 $1000$ 个阈值分类器，要求 $\varepsilon=0.05$，$\delta=0.01$。代入 $n\ge\frac{1}{2\varepsilon^2}\log\frac{2m}{\delta}$：$\varepsilon^2=0.0025$，所以
>
> $$
> n\ge \frac{1}{2\cdot 0.0025}\log(2\cdot 1000/0.01)\approx 200\cdot\log(2\cdot 10^{5})\approx 2440 .
> $$
>
> 其中 $\log(2\times10^5)\approx 12.2$（自然对数）。可以看到，即使把假设数从 $1000$ 加到 $10^6$，$\log$ 项也只从约 $12.2$ 增加到约 $19.1$，样本需求增长有限。
>
> 把阈值换成在连续区间上任意取值后，$|\mathcal{H}|=\infty$，必须改用 VC 维：以单向阈值分类器 $h_t(x)=\mathbf{1}\{x\ge t\}$ 为例，它能打散 1 个点（$t$ 放在点的左边或右边即可），但打散不了 2 个点：对 $x_1<x_2$，“左正右负”这种标记无法实现（只要 $x_1$ 被判正，$x_2\ge x_1$ 也必被判正），所以 VC 维为 $1$。同样的 $\varepsilon$ 下，$n$ 仍是几千量级，而不是无穷。

<br>

> [!NOTE]+ 小结
>
> - ERM 最小化的是经验风险 $\hat R_n$，真正关心的是期望风险 $R$；被挑出来的 $\hat h$ 的训练误差偏乐观。
> - 回归平方误差 = 偏差² + 方差 + 不可约噪声；加数据主要压方差，换假设类主要压偏差。
> - 双重下降说明参数个数不是容量的唯一度量，隐式偏置会在插值解中挑出平滑的解。
> - 有限 $\mathcal{H}$ 的样本复杂度随 $\log|\mathcal{H}|$ 与 $1/\varepsilon^2$ 增长；无限 $\mathcal{H}$ 用 VC 维代替 $\log|\mathcal{H}|$。
> - 学习理论的界很松，用于理解趋势，不用于预测具体误差。

> [!NOTE]+ 衔接
>
> 本章给出了超额风险随容量与 $n$ 变化的规律。下一章用 $\lambda$、早停与交叉验证来限制有效容量。需要提醒的是：若不留出验证折，$\lambda$ 本身也会被训练集拟合。

<br>

# 正则化与模型选择（Regularization and Model Selection）

上一章说明了：假设类越大，越可能在训练集上“挑”出一个碰巧表现好却泛化差的模型。可是事先往往不知道多大的容量才合适——特征该用几个、多项式该几次、惩罚该多强。正则化（regularization）的思路是：不直接换掉假设类，而是在经验风险上加约束或惩罚，把“有效容量”（effective capacity）按下去；模型选择（model selection）的思路是：用一份没有参与拟合的数据（验证集或交叉验证，cross-validation）来挑惩罚强度等超参数，而不是用训练集直接选。

本章依次介绍显式正则（$L_2$、$L_1$、弹性网）、隐式正则（早停等）、交叉验证流程、正则化的贝叶斯解释，最后给出一个用 $k$ 折交叉验证选择岭参数的实现。输入是一组候选超参数，输出是验证风险最小的模型。它承接上一章的超额风险分析，并为后面的无监督章节做铺垫：聚类中没有 $y$，但选择簇数 $k$ 时仍然遵循同样的“留出一块数据”的逻辑。

## 显式正则：岭回归、Lasso 与弹性网

### 岭回归

最小二乘线性回归中，若特征之间高度相关或特征数接近甚至超过样本数，$X^\top X$ 会接近奇异，求出的 $\theta$ 某些分量会非常大且极不稳定（高方差）。岭回归（ridge regression）在平方误差上加一个 $L_2$ 惩罚：

$$
J_{\lambda}(\theta)=\frac12\|X\theta-\vec y\|_{2}^{2}+\frac{\lambda}{2}\|\theta\|_{2}^{2},
\qquad
\theta=(X^{\top}X+\lambda I)^{-1}X^{\top}\vec y.
$$

这里 $X\in\mathbb{R}^{n\times(d+1)}$ 是设计矩阵（每行一个样本），$\vec y\in\mathbb{R}^n$ 是标签向量，$\lambda\ge 0$ 是正则强度。闭式解的推导与正规方程（normal equations）相同：对 $J_\lambda$ 求梯度 $X^\top(X\theta-\vec y)+\lambda\theta$，令其为 $0$，得 $(X^\top X+\lambda I)\theta=X^\top\vec y$。

$\lambda I$ 有两个作用：一是保证可逆——$X^\top X$ 半正定，加上 $\lambda I$（$\lambda>0$）后所有特征值至少为 $\lambda$，矩阵一定可逆；二是把 $\theta$ 拉向 $0$，在数据信息少（$X^\top X$ 特征值小）的方向上收缩得最厉害，从而减小方差。代价是引入一点偏差，这正是偏差-方差权衡的一个具体旋钮。

通常不惩罚截距：截距只是把预测整体平移，惩罚它会让模型依赖于 $y$ 的原点选取，没有道理。实际做法是把截距对应的那一维从惩罚中剔除，或先把 $X$ 与 $y$ 中心化。另外，$L_2$ 惩罚对特征尺度敏感，使用前应先标准化特征。

### Lasso 与弹性网

Lasso 用 $\lambda\|\theta\|_1$ 代替 $L_2$ 惩罚，其最优解是稀疏的（许多分量恰好为 $0$），因此兼做特征选择。直观原因是：$L_1$ 球在坐标轴上有“尖角”，损失函数的等高线最先碰到约束区域时往往碰在尖角上，此时部分坐标为零；而 $L_2$ 球是光滑的，只会把系数缩小，不会缩到恰好为零。Lasso 没有闭式解，通常用坐标下降或近端梯度法求解。

弹性网（elastic net）同时加 $L_1$ 与 $L_2$ 惩罚，常见写法（glmnet、scikit-learn 等软件的约定）是 $\lambda\big[\alpha\|\theta\|_1+\frac{1-\alpha}{2}\|\theta\|_2^2\big]$，$\alpha\in[0,1]$，$\alpha=1$ 退化为 Lasso、$\alpha=0$ 退化为岭回归，在“特征选择”和“系数小而稳定”之间折中；当多个特征高度相关时，它比纯 Lasso 更不容易随意只挑其中一个。

> [!NOTE]+ 岭回归与 Lasso
>
> 二者都在经验风险上加 $\|\theta\|_{2}^{2}$ 或 $\|\theta\|_{1}$，把有效容量按下去。岭回归由 Hoerl 与 Kennard 提出，也对应高斯先验下的 MAP 估计；Lasso 对应 Laplace 先验，并给出稀疏解。与不加惩罚的正规方程相比，$\lambda I$ 保证了可逆，并减小高方差方向上的系数。与只靠早停相比，惩罚写在目标函数里，与使用哪种优化器无关。$\lambda$ 不能在训练集上选，否则训练误差总是在 $\lambda=0$ 时最小，等于没有惩罚。

<br>

### 逻辑回归中的正则

逻辑回归同样可以在负对数似然上加 $\frac{\lambda}{2}\|\theta\|_{2}^{2}$。这有两个好处：其一，当训练数据线性可分时，不加正则的极大似然会让 $\|\theta\|\rightarrow\infty$（把 sigmoid 推成阶跃函数以使似然趋近 $1$），解不存在，加正则后最优解有限；其二，与回归中一样，它降低了方差。

## 隐式正则

并不是所有正则都写在目标函数里。早停（early stopping，在验证误差开始上升时停止训练）、SGD 的噪声、以及过参数化模型中梯度下降收敛到的最小范数插值解，都在不写 $\lambda$ 的情况下偏向简单解，这类效果称为隐式正则（implicit regularization）。例如，对线性回归从零初始化做梯度下降并提前停止，其效果与某个 $\lambda$ 的岭回归相近：迭代次数越少，相当于 $\lambda$ 越大。

一个推论是：同一架构、不同优化器，测试误差可以不同，因为隐式偏置不同。这也是上一章双重下降现象的关键成因。

## 交叉验证（Cross-Validation）

正则强度 $\lambda$、多项式次数、网络宽度等超参数不能用训练误差来选——训练误差几乎总是偏爱容量最大的选项。需要一份“模型没见过”的数据来估计各个候选的真实风险。

### 训练 / 验证 / 测试三分法

把数据分成三部分：

- **训练集**：用于拟合参数 $\theta$；
- **验证集**（validation set，也叫 hold-out set）：用于比较不同超参数，选出最好的；
- **测试集**：只用于最终报告一次性能。

超参数只许看验证集。测试集只许用一次，用来报告。

### $k$ 折交叉验证

数据少时，单独切出一块验证集既浪费数据，验证分数的方差也大。$k$ 折交叉验证（$k$-fold cross-validation）的流程是：

1. 把训练数据随机分成 $k$ 块（fold）；
2. 对每个候选超参数，轮流把 $k$ 块中的一块当验证集，其余 $k-1$ 块当训练集，得到 $k$ 个验证分数；
3. 平均这 $k$ 个分数，作为该超参数的交叉验证误差；
4. 选出交叉验证误差最小的超参数，再在全部训练数据上用选定的 $\lambda$ 重训。

常用 $k=5$ 或 $10$。$k=n$ 时每次只留一个样本做验证，称为留一交叉验证（leave-one-out cross-validation，LOOCV），数据利用率最高但计算量最大。

> [!NOTE]+ 交叉验证
>
> 交叉验证用未参与拟合的一块数据估计各超参数下的风险，最后再在测试集上报告一次。需要它的原因是：$\hat R_n$ 对用来选择 $\lambda$ 的那份数据是乐观的。与单次训练/验证切分相比，$k$ 折降低了验证分数的方差，在小样本上更稳。与在测试集上调参相比，后者使并界中的假设个数继续变大，所报数字不再对应 $R$。时间顺序数据必须按时间切分，随机打乱会把未来信息泄漏进训练。

<br>

### 嵌套交叉验证与时间序列

注意，交叉验证选出的最优分数本身也是“挑出来的最小值”，同样偏乐观。若想在小样本上估计“带选择过程的”泛化性能，可以用嵌套交叉验证（nested cross-validation）：外层交叉验证负责评估，每个外层训练折内部再做一次内层交叉验证来选超参数。它更诚实，但计算更贵。

时间序列必须按时间切分（例如用过去的数据训练、用之后一段的数据验证，并滚动推进），不能随机打乱，否则未来泄漏，验证分数会虚高。

## 贝叶斯视角：正则化即先验

正则化不只是一个工程技巧，它在贝叶斯框架下有自然的解释：惩罚项就是对参数的先验信念。

### MAP 估计与岭回归、Lasso

极大似然估计（MLE）把 $\theta$ 当作未知常数，最大化 $p(\text{data}\mid\theta)$；贝叶斯方法把 $\theta$ 当作随机变量，给它一个先验 $p(\theta)$，再由 Bayes 公式得到后验 $p(\theta\mid\text{data})\propto p(\text{data}\mid\theta)\,p(\theta)$。取后验的众数，就得到最大后验估计（maximum a posteriori，MAP）：

$$
\theta_{\mathrm{MAP}}=\arg\max_\theta\Big[\log p(\text{data}\mid\theta)+\log p(\theta)\Big].
$$

取高斯先验 $\theta\sim\mathcal{N}(0,\tau^{2}I)$，配上噪声方差为 $\sigma^2$ 的高斯似然 $y^{(i)}\mid x^{(i)},\theta\sim\mathcal{N}(\theta^\top x^{(i)},\sigma^2)$，代入后（去掉与 $\theta$ 无关的常数，并乘以 $-\sigma^2$）：

$$
\theta_{\mathrm{MAP}}=\arg\min_\theta\Big[\frac12\|X\theta-\vec y\|_2^2+\frac{\sigma^2}{2\tau^2}\|\theta\|_2^2\Big],
$$

这恰为岭回归，且 $\lambda=\sigma^2/\tau^2$，即 $\lambda$ 与 $1/\tau^{2}$ 成正比：先验越窄（$\tau$ 越小，越相信 $\theta$ 接近 $0$），惩罚越强。类似地，Laplace 先验 $p(\theta_j)\propto e^{-|\theta_j|/b}$ 取对数后给出 $L_1$ 惩罚，对应 Lasso。

所以贝叶斯观点把 $\lambda$ 对应到先验精度 $1/\tau^{2}$（精度即方差的倒数；原笔记写作“先验方差 $1/\tau^{2}$”，先验方差应是 $\tau^2$，这里改正）。但这并没有免除选择的问题：$\tau$ 仍需选择，交叉验证仍然必要。

### 全贝叶斯预测

MAP 只取后验的一个点，忽略了参数的不确定度。全贝叶斯预测（fully Bayesian prediction）则把 $\theta$ 积掉：

$$
p(y\mid x,\mathrm{data})=\int p(y\mid x,\theta)\,p(\theta\mid\mathrm{data})\,d\theta .
$$

它相当于用后验概率作为权重，对所有可能的参数下的预测取平均，因此能给出带不确定度的预测。线性高斯模型可以闭式计算（后验与预测分布仍是高斯）；一般模型的积分难以解析，要用变分推断或 MCMC 近似——变分推断的思想将在 EM 一章中出现。

## 实现：用 $k$ 折交叉验证选择岭参数

下面的函数对每个候选 $\lambda$ 做 $k$ 折交叉验证，返回平均验证均方误差最小的 $\lambda$ 以及所有候选的分数。其中 `linear_regression_normal_eq(X, y, reg=lam)` 是线性回归一章中定义的正规方程求解函数，`reg` 即岭参数 $\lambda$（该函数对包括截距在内的所有分量都加了 $\lambda$，与“通常不惩罚截距”的建议略有出入，样本较多时影响很小）。

```python
def ridge_cv(X, y, lambdas, k=5, rng=None):
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    idx = rng.permutation(n)            # 随机打乱样本下标（时间序列不能这样做）
    folds = np.array_split(idx, k)      # 切成 k 折，各折大小至多相差 1
    scores = []
    for lam in lambdas:
        errs = []
        for i in range(k):
            va = folds[i]               # 第 i 折做验证
            tr = np.concatenate([folds[j] for j in range(k) if j != i])  # 其余做训练
            theta = linear_regression_normal_eq(X[tr], y[tr], reg=lam)
            pred = X[va] @ theta
            errs.append(np.mean((pred - y[va]) ** 2))   # 验证集 MSE
        scores.append(float(np.mean(errs)))             # k 折平均
    return lambdas[int(np.argmin(scores))], scores
```

返回选定的 $\lambda$ 后，还应在全部训练数据上用它重新训练一次，再在测试集上报告一次。注意所有折共用同一个随机划分 `folds`，这样不同 $\lambda$ 的比较是在相同切分下进行的，减少了比较中的随机噪声。

> [!EXAMPLE]+ 例13 不要在测试集上选 $\lambda$
>
> 若拿十个 $\lambda$ 在测试集上挑误差最小的那个，相当于对测试集做了十次假设检验，报告值偏乐观——用上一章的话说，并界里的 $m$ 从 $1$ 变成了 $10$。正确流程是：在验证集（或交叉验证）上选出 $\lambda=10^{-2}$，再在测试集上只报一次。若测试误差仍然不理想，应回到特征与模型设定上去改进，而不是继续在测试集上扫参数。

<br>

> [!NOTE]+ 对照：显式正则与隐式正则
>
> 岭回归与 Lasso 把惩罚写进目标 $J$。早停、SGD 噪声、最小范数插值不写 $\lambda$，却同样偏向简单解。双重下降说明：过参数化区的测试误差可以再次下降，前提是优化路径挑中了“简单”的插值解。换优化器而不换 $\mathcal{H}$，测试误差仍可能改变，因为隐式偏置变了。

<br>

> [!NOTE]+ 小结
>
> - 正则化通过惩罚项降低有效容量，以少量偏差换取较大的方差下降。
> - 岭回归有闭式解 $(X^\top X+\lambda I)^{-1}X^\top\vec y$，保证可逆；Lasso 给出稀疏解；弹性网二者兼顾。
> - 超参数用验证集或 $k$ 折交叉验证选，测试集只用一次；时间序列按时间切分。
> - 高斯先验的 MAP 即岭回归（$\lambda=\sigma^2/\tau^2$），Laplace 先验对应 Lasso；全贝叶斯预测对参数积分。

> [!NOTE]+ 衔接
>
> 有监督学习部分到此为止：假设、求解、容量与模型选择。下一章不再使用标签 $y$。届时簇数 $k$ 不能靠最小化训练失真来确定，否则最优解是 $k=n$（每个点自成一簇）。同样需要留出数据，或检查聚类在重抽样下是否稳定。

<br>

# 聚类与 k-means（Clustering）

到目前为止，每个样本都带着标签 $y$，学习的目标是从 $x$ 预测 $y$。但很多数据根本没有标签：一堆用户的购买记录、一批基因表达谱、一组图像的特征向量。这时一个自然的问题是：这些点能不能自动分成几组，使得同一组内彼此相似、不同组之间差异明显？这就是聚类（clustering），属于无监督学习（unsupervised learning）。

$k$-means 是最经典的聚类算法，其核心想法一句话就能说清：用 $k$ 个“中心”代表 $k$ 个簇，让每个点到它所属中心的平方距离之和（失真，distortion）尽量小。本章的输入是数据 $X$ 与簇数 $k$，输出是 $k$ 个中心与每个点的分配。它与前面有监督各章的关系是：以前有目标函数、求解算法和模型选择协议，这里同样有，只是目标里不再出现 $y$。它与下一章 EM 的关系是：$k$-means 的“硬分配” $c^{(i)}$ 会被换成概率意义下的“软分配”，即后验责任度 $w_j^{(i)}$。

## 失真函数

给定无标签数据 $\{x^{(1)},\dots,x^{(n)}\}$，$x^{(i)}\in\mathbb{R}^d$。$k$-means 引入两组未知量：

- $c^{(i)}\in\{1,\ldots,k\}$ 是第 $i$ 个点的簇分配；
- $\mu_j\in\mathbb{R}^d$ 是第 $j$ 组的中心（$j=1,\dots,k$）。

失真函数定义为每个点到其所属中心的平方距离之和：

$$
J(c,\mu)=\sum_{i=1}^{n}\|x^{(i)}-\mu_{c^{(i)}}\|_{2}^{2}.
$$

$J$ 越小，说明每个点离它的代表中心越近，簇越“紧”。$J$ 对 $(c,\mu)$ 联合非凸（$c$ 甚至是离散变量），无法一步求出全局最优；但若固定其中一组变量，对另一组的最优化都很容易，这就引出了坐标下降。

## Lloyd 算法：交替的坐标下降

Lloyd 算法（即通常所说的 $k$-means 算法）先初始化 $k$ 个中心（常见做法是从数据中随机选 $k$ 个点），然后交替执行以下两步直到收敛：

1. **分配步**：固定中心，把每个点分给最近的中心，$c^{(i)}=\arg\min_j\|x^{(i)}-\mu_j\|_{2}^{2}$；
2. **更新步**：固定分配，令 $\mu_j=$ 分到 $j$ 的点的均值，即 $\mu_j=\dfrac{\sum_i \mathbf{1}\{c^{(i)}=j\}x^{(i)}}{\sum_i\mathbf{1}\{c^{(i)}=j\}}$；若某个簇没有分到任何点（空簇），需重新初始化该中心（下文的参考实现采用更简单的“保留旧中心”处理，差别见实现后的说明）。

两步都是在对 $J$ 做精确的坐标最小化：

- 分配步中，$J$ 关于 $c$ 可以逐点分解，每个点独立地选择让 $\|x^{(i)}-\mu_{c^{(i)}}\|^2$ 最小的中心，这正是“分给最近的中心”；
- 更新步中，$J$ 关于 $\mu_j$ 只涉及分到 $j$ 的那些点，$\sum_{i:c^{(i)}=j}\|x^{(i)}-\mu_j\|^2$ 是 $\mu_j$ 的凸二次函数，对 $\mu_j$ 求导令其为零，$-2\sum_{i:c^{(i)}=j}(x^{(i)}-\mu_j)=0$，解得组内均值。

## 收敛性、初始化与 $k$ 的选择

**收敛性。** 由于两步都是坐标最小化，每步都不增加 $J$。而分配方式 $c$ 只有有限种（至多 $k^n$ 种），每种分配对应唯一的均值中心，所以 $J$ 不可能无限地严格下降，算法在有限步内必停。但它停在的是局部最小，不一定是全局最小——不同初始化可能收敛到截然不同的聚类。

**初始化。** 应多次随机初始化取最小 $J$：跑若干次 $k$-means，每次随机选初始中心，保留失真最小的结果。更好的初始化是 $k$-means++：第一个中心随机选，之后每个新中心以“与已选中心最近距离的平方”成正比的概率从数据点中抽取，从而让初始中心彼此分散，通常收敛更快、结果更好。

**$k$ 的选择。** $k$ 不能靠最小化 $J$ 来选择：$k$ 越大 $J$ 越小，$k=n$ 时每个点自成一簇，$J=0$。实用的方法有：

- **肘部法（elbow method）**：画出 $J$ 随 $k$ 变化的曲线，选择曲线下降明显变缓的拐点；
- **轮廓系数（silhouette coefficient）**：比较每个点到本簇其他点的平均距离与到最近邻簇的平均距离，越接近 $1$ 表示聚类越清晰；
- **下游任务指标**：若聚类是为了后续某项任务服务，直接用该任务的表现来选；
- 并配合**稳定性检查**：对数据做 bootstrap 重抽样后重新聚类，看簇是否还在。真实存在的结构在重抽样下应当稳定。

> [!NOTE]+ $k$-means
>
> $k$-means 把每个点分给最近的中心，再把中心更新为组内均值，交替进行直到失真不再下降。它源自向量量化与 Lloyd 算法，目标是 $L_2$ 失真，而不是似然。与有监督分类相比，这里没有 $y$，不能用交叉熵之类的指标选 $k$。与下一章的 GMM 相比，$k$-means 的分配是 $0/1$，并且把每个簇当成各向同性的球。$J$ 非凸，必须多次随机启动；出现空簇时要重新抽样中心。

<br>

## $k$-means 的隐含假设与局限

$k$-means 只用“到中心的欧氏距离”来划分，因而隐含地假设簇大致是球形的、各簇方差相近、规模不至于悬殊。长条形、环形、密度不同的簇都会被切错：例如两个同心圆环，任何中心都无法用“离谁近”把内外环分开。此时应考虑 GMM（可表达椭圆簇）、谱聚类或基于密度的方法。

另外，特征必须处于同一尺度，否则“距离”会被数值范围最大的那一维主导。例如一个特征是以元为单位的收入、另一个是年龄，不标准化时聚类几乎只按收入划分。

## 实现

下面的实现带多次随机重启，返回失真最小的一次结果 `(J, mu, c)`。距离计算用广播一次算出 $n\times k$ 的平方距离矩阵。

```python
def kmeans(X, k, num_restarts=10, max_iter=100, rng=None):
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    best = None
    for _ in range(num_restarts):
        mu = X[rng.choice(n, size=k, replace=False)]      # 随机选 k 个数据点作初始中心，(k, d)
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)  # (n, k) 平方距离
            c = d2.argmin(axis=1)                         # 分配步
            new_mu = np.stack([
                X[c == j].mean(axis=0) if np.any(c == j) else mu[j]   # 更新步；空簇保留旧中心
                for j in range(k)
            ])
            if np.allclose(new_mu, mu):                   # 中心不再移动则收敛
                break
            mu = new_mu
        J = float(((X - mu[c]) ** 2).sum())               # 本次重启的失真
        if best is None or J < best[0]:
            best = (J, mu, c)
    return best
```

需要说明两点。第一，这段代码对空簇的处理是“保留旧中心”，这避免了对空切片求均值得到 `nan`，但被困住的中心可能一直分不到点，相当于实际只用了少于 $k$ 个簇；更稳妥的做法是像正文所说那样把它重新抽样到某个数据点上（例如离当前中心最远的点）。第二，若循环因达到 `max_iter` 而结束，最后的 `c` 对应的是更新前的中心，计算 `J` 时与 `mu` 有一步之差；收敛时（`break` 分支）二者一致，不影响结果。

> [!EXAMPLE]+ 例14 空簇
>
> 设初始化时两个中心都落在同一个稠密团的外面，且一个比另一个离这团点更近。分配步中所有点都分给更近的那个中心，另一中心永远分不到点。实现必须检测空簇，否则对空切片调用 `mean` 会得到 `nan`，后续所有距离都会变成 `nan`，整个算法失效。检测到空簇后，稳妥的做法是把该中心重新抽样到某个数据点上；上面的 `kmeans` 用 `np.any(c == j)` 做了检测，但只是保留旧中心——这避免了 `nan`，却不能把被困的中心救回来，按本例的初始化它会一直空着，结果实际上只有一个簇，只能靠多次重启中别的初始化得到正常结果。（顺带一提，上面代码的初始中心取自数据点，每个中心至少拥有自己那个点，所以第一轮不会出现空簇，除非数据中有重复点；空簇主要出现在后续迭代或用其他方式初始化时。）

<br>

> [!NOTE]+ 衔接
>
> $k$-means 的分配是 $0/1$ 的：每个点要么完全属于某簇，要么完全不属于。下一章引入潜变量 $z$，用后验 $p(z\mid x)$ 做软分配，并证明 EM 每一步都不降低边缘似然。高斯混合模型（GMM）可以看作带协方差的软分配聚类。

<br>

# EM 算法与潜变量模型（Expectation–Maximization）

$k$-means 给每个点一个硬标签，但很多时候这并不合理：一个恰好落在两个簇中间的点，说它“百分之百属于左簇”显然过于武断。更自然的做法是建立一个概率模型：假设每个点背后有一个看不见的“簇编号” $z$，先按某个概率抽出 $z$，再从对应的分布中生成 $x$。这个 $z$ 就是潜变量（latent variable）。如果 $z$ 已知，参数估计就是普通的极大似然（类似 GDA）；困难恰恰在于 $z$ 没有观测到。

形式上，观测 $x$ 与潜变量 $z$ 的联合模型是 $p(x,z;\theta)$，而能用于训练的只有边缘似然 $p(x;\theta)=\sum_z p(x,z;\theta)$。对数里面套着求和，$\log\sum_z$ 难以优化：对 $\theta$ 求导后没有闭式解，各分量的参数纠缠在一起。EM（expectation-maximization）算法的核心思想是：交替地构造边缘对数似然的一个下界，并抬升这个下界。E 步“猜”潜变量的分布，M 步在这个猜测下做普通的极大似然。

本章先从 Jensen 不等式推出证据下界（evidence lower bound，ELBO），给出一般的 EM 算法及其单调性证明，再特化到高斯混合模型，最后简述当后验无法精确计算时的变分推断。输入是不完全数据（只有 $x$ 没有 $z$），输出是参数 $\theta$ 与后验责任度。它与上一章的关系是：$k$-means 的 $J(c,\mu)$ 相当于把簇分配当作已知的“完全数据”平方损失，本章的 GMM 是其软分配版本；与下一章的关系是：PCA / 因子分析也是潜变量模型，但 $z$ 是连续坐标，不是离散簇。

> [!NOTE]+ EM
>
> 当边缘似然 $\log p(x)=\log\sum_z p(x,z)$ 不好直接最大化时，EM 交替执行两步：用当前 $\theta$ 计算后验 $p(z\mid x)$（E 步），再最大化完全数据对数似然在该后验下的期望（M 步）。它源自 Dempster、Laird 与 Rubin 对不完全数据极大似然的研究，所用的下界就是 ELBO。与硬分配的 $k$-means 相比，处在重叠区域的点以分数责任度同时参与多个分量的更新，且似然单调不降。与直接对 $\log\sum_z$ 做梯度上升相比，E 步在指数族模型上常有闭式解，M 步又回到熟悉的完全数据 MLE。需要注意，EM 不保证达到全局最大。

<br>

## Jensen 不等式与证据下界（ELBO）

### Jensen 不等式

Jensen 不等式说：对凸函数 $f$ 与随机变量 $t$，$f(E[t])\le E[f(t)]$，即“函数的期望不小于期望的函数”；若 $f$ 严格凸，等号当且仅当 $t$ 以概率 $1$ 为常数时成立。$\log$ 是凹函数（$-\log$ 凸），所以不等号反向：$\log E[t]\ge E[\log t]$。

### 构造下界

对正数 $t(z)=p(x,z;\theta)/Q(z)$ 与 $z$ 上的任意分布 $Q$（要求在 $p(x,z;\theta)>0$ 处 $Q(z)>0$），

$$
\log\sum_z Q(z)\frac{p(x,z;\theta)}{Q(z)}
\ge
\sum_z Q(z)\log\frac{p(x,z;\theta)}{Q(z)}.
$$

左边分子分母的 $Q(z)$ 抵消，就是 $\log\sum_z p(x,z;\theta)=\log p(x;\theta)$。右边定义为证据下界：

$$
\mathrm{ELBO}(x;Q,\theta)
=
\sum_z Q(z)\log\frac{p(x,z;\theta)}{Q(z)}
=
E_{z\sim Q}[\log p(x,z;\theta)]+H(Q).
$$

第二个等号把对数拆开：$\sum_z Q\log p(x,z)$ 是完全数据对数似然在 $Q$ 下的期望，$-\sum_z Q\log Q=H(Q)$ 是 $Q$ 的熵。“证据”（evidence）指的是 $\log p(x;\theta)$。

于是对任意 $Q$，$\log p(x;\theta)\ge\mathrm{ELBO}(x;Q,\theta)$。两者之间的缺口恰好是 $\mathrm{KL}(Q\|p(z\mid x;\theta))$，因此当 $Q=p(z\mid x;\theta)$ 时等号成立。

> [!INFO]+ Jensen 怎样把边缘似然托成下界
>
> 边缘似然 $p(x)=\sum_z p(x,z)$。任选一个分布 $Q(z)>0$，把求和改写成加权平均：
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
> 缺口可以直接算出来。利用 $\sum_z Q(z)=1$ 把 $\log p(x)$ 写成 $\sum_z Q(z)\log p(x)$，再用 $p(x,z)=p(x)\,p(z\mid x)$：
>
> $$
> \log p(x)-\mathrm{ELBO}=\sum_z Q(z)\log\frac{Q(z)}{p(z\mid x)}=D_{\mathrm{KL}}(Q\|p_{z\mid x})\ge 0.
> $$
>
> E 步把 $Q$ 设成当前后验，KL 归零，下界贴住 $\ell(\theta^{(t)})$。M 步抬高下界；由于 $\ell$ 始终在下界上方，似然不会下降。
>
> 感性理解：$\log p(x)$ 要对潜变量边缘化，直接求和或积分通常不可行。引入任意 $Q$ 后，Jensen 给出可计算的下界；缺口等于 $D_{\mathrm{KL}}(Q\|p_{z\mid x})$。E 步取 $Q=p(z\mid x;\theta^{(t)})$，下界与 $\ell(\theta^{(t)})$ 相切；M 步抬高下界。由于 $\ell$ 始终在 ELBO 上方，边缘似然单调不降，但不保证到达全局最大。

<br>

从 Jensen 的等号条件也能看出最优 $Q$：等号要求 $p(x,z;\theta)/Q(z)$ 对 $z$ 是常数，即 $Q(z)\propto p(x,z;\theta)$，归一化后正是 $Q(z)=p(z\mid x;\theta)$。

## EM 算法与单调性

### 多样本下界

对 $n$ 个独立样本，每个样本 $x^{(i)}$ 配一个自己的分布 $Q_i$（因为每个点的潜变量后验不同），对数似然满足：

$$
\ell(\theta)=\sum_{i=1}^{n}\log p(x^{(i)};\theta)
\ge
\sum_{i=1}^{n}\mathrm{ELBO}(x^{(i)};Q_i,\theta).
$$

### 算法

EM 从某个初值 $\theta^{(0)}$ 出发，重复以下两步直到收敛（例如 $\ell$ 的增量小于阈值）：

- **E 步**：令 $Q_i^{(t)}(z)=p(z\mid x^{(i)};\theta^{(t)})$，此时下界贴住 $\ell(\theta^{(t)})$；
- **M 步**：$\theta^{(t+1)}=\arg\max_{\theta}\sum_i\mathrm{ELBO}(x^{(i)};Q_i^{(t)},\theta)$，即最大化完全数据对数似然在 $Q_i$ 下的期望。

M 步之所以能化为“完全数据对数似然的期望”，是因为 ELBO 中的熵项 $H(Q_i)$ 不依赖 $\theta$，最大化时可以丢掉。

### 单调性证明

$$
\ell(\theta^{(t+1)})\ge\mathrm{ELBO}(Q^{(t)},\theta^{(t+1)})\ge\mathrm{ELBO}(Q^{(t)},\theta^{(t)})=\ell(\theta^{(t)}).
$$

三步的理由依次是：第一个不等号是 Jensen 下界对任意 $Q$ 与 $\theta$ 都成立；第二个不等号是因为 $\theta^{(t+1)}$ 是 M 步的最大化者，不会比 $\theta^{(t)}$ 差；最后的等号是因为 E 步选的 $Q^{(t)}$ 恰好让下界在 $\theta^{(t)}$ 处取等。

因此 EM 不保证全局最大，只保证似然不降。从另一个角度看，EM 是在 ELBO 上对 $(Q,\theta)$ 做坐标上升：E 步对 $Q$ 最大化，M 步对 $\theta$ 最大化——这与 $k$-means 对 $(c,\mu)$ 做坐标下降完全平行。实践中应多次随机初始化，比较最终的 $\ell$。

### ELBO 的另两种写法

把 $p(x,z)$ 按不同方式分解，ELBO 还有两种常用形式：

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

第一式用 $p(x,z)=p(z)\,p(x\mid z)$ 得到：第一项是“重构项”（在 $Q$ 下数据被解释得多好），第二项让 $Q$ 不要偏离先验 $p_z$ 太远；这正是后面变分自编码器的训练目标。第二式用 $p(x,z)=p(x)\,p(z\mid x)$ 得到，它直接说明最优 $Q$ 是真实后验：$\log p(x;\theta)$ 与 $Q$ 无关，最大化 ELBO 等价于最小化 KL，KL 在 $Q=p_{z\mid x}$ 时取最小值 $0$。

## 高斯混合模型（Gaussian Mixture Model）

### 模型

高斯混合模型（GMM）是 EM 最经典的应用。它假设每个样本按如下方式生成：先以概率 $\phi_j$ 抽出簇编号 $z^{(i)}=j$，再从第 $j$ 个高斯中抽出 $x^{(i)}$：

$$
p(z^{(i)}=j)=\phi_j,\qquad
x^{(i)}\mid z^{(i)}=j\sim\mathcal{N}(\mu_j,\Sigma_j).
$$

参数 $\theta=(\phi,\mu,\Sigma)$：$\phi\in\mathbb{R}^k$ 满足 $\phi_j\ge0$、$\sum_j\phi_j=1$；$\mu_j\in\mathbb{R}^d$ 是第 $j$ 个分量的均值；$\Sigma_j\in\mathbb{R}^{d\times d}$ 是其协方差（对称正定）。与 GDA 相比，模型结构完全一样，区别只在于 GDA 中类别 $y$ 是观测到的，而这里的 $z$ 是潜变量。

### E 步：计算责任度

E 步的责任度（responsibilities）就是潜变量的后验，由 Bayes 公式计算：

$$
w_j^{(i)}
=
p(z^{(i)}=j\mid x^{(i)};\theta)
=
\frac{\phi_j\,\mathcal{N}(x^{(i)};\mu_j,\Sigma_j)}{\sum_{\ell}\phi_{\ell}\,\mathcal{N}(x^{(i)};\mu_{\ell},\Sigma_{\ell})}.
$$

$w_j^{(i)}\in[0,1]$，且对每个 $i$ 有 $\sum_j w_j^{(i)}=1$，可以理解为“点 $i$ 有多大比例属于分量 $j$”。

### M 步：加权的 GDA 更新

M 步与带权的 GDA 更新相同：

$$
\begin{aligned}
\phi_j&=\frac1n\sum_i w_j^{(i)},\\
\mu_j&=\frac{\sum_i w_j^{(i)}x^{(i)}}{\sum_i w_j^{(i)}},\\
\Sigma_j&=\frac{\sum_i w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^{\top}}{\sum_i w_j^{(i)}}.
\end{aligned}
$$

三个公式的含义都很直接：$\phi_j$ 是分量 $j$ 的“有效样本数” $\sum_i w_j^{(i)}$ 占总数的比例；$\mu_j$ 是以责任度加权的样本均值；$\Sigma_j$ 是以责任度加权、相对 $\mu_j$ 的散度矩阵（这里的 $\mu_j$ 用的是本轮刚更新的值）。

> [!INFO]+ 责任度与 M 步加权公式
>
> 若 $z$ 已知，完全数据对数似然是 $\sum_i\sum_j\mathbf{1}\{z^{(i)}=j\}\big(\log\phi_j+\log\mathcal{N}(x^{(i)};\mu_j,\Sigma_j)\big)$。$z$ 未知时，E 步用后验期望替换指示变量：
>
> $$
> w_j^{(i)}=E[\mathbf{1}\{z^{(i)}=j\}\mid x^{(i)}]=p(z^{(i)}=j\mid x^{(i)}).
> $$
>
> 由 Bayes 公式，后验正比于先验 $\phi_j$ 乘该类高斯密度，再对 $j$ 归一化，即正文中的 $w_j^{(i)}$。M 步要最大化 $\sum_i\sum_j w_j^{(i)}\big(\log\phi_j+\log\mathcal{N}(x^{(i)};\mu_j,\Sigma_j)\big)$，它对 $\phi,\mu,\Sigma$ 可分开求最大。
>
> - $\phi$ 在单纯形上：与 $\phi$ 有关的项是 $\sum_i\sum_j w_j^{(i)}\log\phi_j$，约束 $\sum_j\phi_j=1$。拉格朗日函数对 $\phi_j$ 求导得 $\sum_i w_j^{(i)}/\phi_j=\beta$，即 $\phi_j\propto\sum_i w_j^{(i)}$；又因 $\sum_j\sum_i w_j^{(i)}=n$，归一化给出 $\phi_j=\frac1n\sum_i w_j^{(i)}$。（$\phi_j\ge0$ 的约束自动满足。）
> - $\mu_j$：$\log\mathcal{N}$ 中与 $\mu_j$ 有关的部分是 $-\frac12(x^{(i)}-\mu_j)^\top\Sigma_j^{-1}(x^{(i)}-\mu_j)$，对 $\mu_j$ 求梯度得 $\sum_i w_j^{(i)}\Sigma_j^{-1}(x^{(i)}-\mu_j)=0$，两边左乘 $\Sigma_j$，驻点是加权均值
>
> $$
> \mu_j=\frac{\sum_i w_j^{(i)}x^{(i)}}{\sum_i w_j^{(i)}}.
> $$
>
> - $\Sigma_j$ 同理（对 $\Sigma_j^{-1}$ 求导），结果是相对 $\mu_j$ 的加权散度矩阵。
>
> 这与 GDA 的闭式解形式相同，只是每个点以分数 $w$ 同时属于各类，而不是以 $0/1$ 只属于一类。
>
> 感性理解：$w_j^{(i)}$ 是点 $i$ 属于分量 $j$ 的后验质量。M 步把完全数据的 MLE 换成按该质量加权的均值与协方差，形式与 GDA 相同。重叠区域的点对多个分量同时贡献分数权重。$k$-means 是各向同性协方差、后验取 $0/1$ 的极限，重叠点只能整份归入一个中心。

<br>

### 与 $k$-means 的关系及常见问题

$k$-means 可视为 GMM 在协方差各向同性、责任度变硬（$0/1$）时的极限：若令所有 $\Sigma_j=\sigma^2 I$、$\phi_j$ 相等，并让 $\sigma\to0$，则责任度公式中离 $x^{(i)}$ 最近的那个分量的高斯密度会压倒其余所有分量，$w_j^{(i)}$ 趋于 $0/1$ 的指示变量，E 步变成“分给最近中心”，M 步的 $\mu_j$ 更新变成组内均值。

GMM 能表达椭圆簇与重叠簇，这是它相对 $k$-means 的主要优势；代价是两个问题：

- **局部最优**：与 $k$-means 一样，结果依赖初始化。
- **奇异协方差**：若某分量的均值恰好贴住一个数据点，并把协方差缩向零（$\Sigma\rightarrow 0$），该点处的高斯密度趋于无穷，似然 $\rightarrow\infty$。这说明 GMM 的极大似然在严格意义上没有有限最优解，EM 可能跑向这种退化解。

应对方法是对 $\Sigma$ 加岭（在对角线上加一个小常数）、设定最小特征值下界，或多随机启动并剔除退化解。

## 变分推断（Variational Inference）

GMM 中的后验 $p(z\mid x)$ 能精确计算（只是 $k$ 个数的归一化），所以 E 步可以把 $Q$ 取成真实后验。但在更复杂的模型里（例如 $z$ 是高维连续变量、$p(x\mid z)$ 是神经网络），后验 $p(z\mid x)=p(x,z)/\int p(x,z)\,dz$ 中的积分无法计算，E 步就无法精确执行。

变分推断的做法是：当 $p(z\mid x)$ 不可解析时，不再要求 $Q$ 等于后验，而是在一个易处理的受限族 $\mathcal{Q}$ 上最大化 ELBO。由“ELBO $=\log p(x)-\mathrm{KL}(Q\|p_{z\mid x})$”可知，这等价于在 $\mathcal{Q}$ 中找离真实后验最近（KL 意义下）的那个分布。常见的受限族是平均场（mean-field）假设 $Q(z)=\prod_j Q_j(z_j)$，即假设各潜变量分量在 $Q$ 下相互独立。

变分自编码器（variational autoencoder，VAE）是这一思路与神经网络的结合：

- 把 $Q(z\mid x)$ 参数化为一个编码器网络（例如输出高斯分布的均值与方差），对所有样本共享参数，称为摊销推断（amortized inference）；
- 用重参数化技巧（reparameterization trick）估计对 $z$ 的期望的梯度：把 $z=\mu(x)+\sigma(x)\odot\epsilon$，$\epsilon\sim\mathcal{N}(0,I)$，使采样过程对网络参数可导；
- 解码器网络建模 $p(x\mid z)$。

训练目标仍是 ELBO（即上面第一种写法：重构项减去 $\mathrm{KL}(Q\|p_z)$），而不是随意的重构平方差；KL 项防止编码器把后验坍缩到与先验无关的点上，保证潜空间的结构可用于生成。

## 实现

下面给出 GMM 的 EM 实现。为了数值稳定，全程在对数域计算高斯密度，并用 log-sum-exp 技巧做责任度归一化。

```python
def log_gaussian(X, mu, Sigma, ridge=1e-6):
    d = X.shape[1]
    S = Sigma + ridge * np.eye(d)                 # 加岭，防止奇异
    diff = X - mu                                 # (n, d)
    sol = np.linalg.solve(S, diff.T).T            # S^{-1}(x - mu)，避免显式求逆
    sign, logdet = np.linalg.slogdet(S)           # 稳定地计算 log|S|
    return -0.5 * (d * np.log(2 * np.pi) + logdet + np.sum(diff * sol, axis=1))

def gmm_em(X, k, num_iters=50, ridge=1e-6, rng=None):
    rng = np.random.default_rng(rng)
    n, d = X.shape
    mu = X[rng.choice(n, size=k, replace=False)]  # 随机选 k 个点作初始均值
    Sigma = np.stack([np.cov(X, rowvar=False) + ridge * np.eye(d) for _ in range(k)])  # 初始协方差取全体数据协方差
    phi = np.ones(k) / k                          # 初始混合比例均匀
    for _ in range(num_iters):
        # E 步：log(phi_j) + log N(x; mu_j, Sigma_j)，形状 (n, k)
        logcomp = np.stack([np.log(phi[j]) + log_gaussian(X, mu[j], Sigma[j], ridge) for j in range(k)], axis=1)
        m = logcomp.max(axis=1, keepdims=True)    # log-sum-exp：先减去每行最大值
        resp = np.exp(logcomp - m)
        resp /= resp.sum(axis=1, keepdims=True)   # 责任度 w_j^{(i)}，每行和为 1
        # M 步
        n_k = resp.sum(axis=0) + 1e-12            # 各分量的有效样本数
        phi = n_k / n
        mu = (resp.T @ X) / n_k[:, None]
        for j in range(k):
            dlt = X - mu[j]
            Sigma[j] = (dlt.T @ (dlt * resp[:, j:j + 1])) / n_k[j] + ridge * np.eye(d)
    return phi, mu, Sigma, resp
```

几点说明：减去每行最大值再取指数，是为了避免高维时 $\log\mathcal{N}$ 很负导致 `exp` 下溢为 $0$，归一化后结果不变；`n_k` 加 `1e-12` 防止某分量责任度全为零时除以零；每次更新 $\Sigma_j$ 都加 `ridge`，对应正文中“对 $\Sigma$ 加岭”的防退化措施。若要监控收敛，可以在 E 步中顺便计算 $\ell(\theta)=\sum_i\big(m_i+\log\sum_j e^{\text{logcomp}_{ij}-m_i}\big)$，它应单调不降。

> [!EXAMPLE]+ 例15 两个高斯重叠
>
> 数据来自 $0.5\,\mathcal{N}((-2,0),I)+0.5\,\mathcal{N}((2,0),I)$，即两个单位协方差的二维高斯，中心分别在 $(-2,0)$ 与 $(2,0)$。E 步给左峰附近的点高 $w_1$，右峰附近的点高 $w_2$，中间（横坐标接近 $0$）的点两者都约为 $0.5$（拟合出的两个分量越对称越接近 $0.5$；有限样本下估计略有偏差，中间点的责任度会偏向某一侧）。M 步按这些权重做加权平均，把两个中心拉回 $\pm 2$ 附近。
>
> 若初始化时两个中心都在右侧，可能出现两个分量“抢”同一个簇、另一个簇被忽略的情况，似然停在较差的局部最优。多次重启并比较最终的 $\ell$ 是标准补救。

<br>

> [!NOTE]+ 对照：硬分配与软分配
>
> $k$-means 中每个点只属于一个中心，目标是失真 $J$。GMM 中每个点对每个分量都有责任度 $w_j^{(i)}$，目标是边缘似然。当协方差各向同性且 $w$ 变硬时，二者重合。需要椭圆簇或重叠密度时用 GMM；只想要可解释的分区、且希望实现稳定时，先跑多次 $k$-means（也常用 $k$-means 的结果来初始化 GMM）。

<br>

> [!NOTE]+ 小结
>
> - 潜变量模型的边缘对数似然 $\log\sum_z p(x,z)$ 难以直接优化。
> - Jensen 不等式给出下界 ELBO，缺口是 $\mathrm{KL}(Q\|p_{z\mid x})$，$Q$ 取后验时等号成立。
> - EM = 对 ELBO 的坐标上升：E 步取后验，M 步最大化期望完全数据对数似然；似然单调不降，但只到局部最优。
> - GMM 的 E 步算责任度，M 步是加权 GDA；$k$-means 是其硬分配、各向同性极限；需防范协方差奇异。
> - 后验不可算时，在受限族上最大化 ELBO，即变分推断；VAE 用编码器与重参数化实现。

> [!NOTE]+ 衔接
>
> EM 中的 $z$ 是离散的簇编号。下一章的潜变量改成连续坐标：PCA 寻找方差最大的方向，因子分析给出“低秩加对角”协方差的生成模型。ELBO 这套工具仍然适用，只是 $Q(z\mid x)$ 从多项分布换成了高斯分布。

<br>

<br>

# 主成分分析（Principal Component Analysis, PCA）

高维数据往往“看起来”维数很高、实际上只在少数几个方向上有明显变化。例如 $100\times 100$ 的灰度人脸图像有一万个像素，但相邻像素高度相关，所有人脸大致落在一个低维的“脸空间”附近；又如一份问卷的几十道题，答案可能主要由“外向程度”“焦虑程度”等少数几个因素决定。PCA 要解决的问题是：给定数据 $x\in\mathbb{R}^{d}$，找一个 $k\ll d$ 维的线性子空间，把数据投影上去，使得投影后**保留的方差最大**，或者等价地，使得从低维坐标回到原空间时的**重构误差（reconstruction error）最小**。

核心想法一句话：数据云沿哪个方向“拉得最长”，那个方向就最值得保留；依次找出彼此正交、方差递减的方向，前 $k$ 个就构成最好的 $k$ 维近似。后面会看到，“最大方差”与“最小 $L_2$ 重构误差”这两种目标在 $L_2$ 下给出同一特征问题——协方差矩阵的特征分解。

与相邻章节的关系：上一章（EM 算法与潜变量模型，如高斯混合）中的潜变量 $z$ 是离散的簇指标，每个样本被归到某一簇；本章的 $z$ 换成连续坐标，目标是保方差或做低秩重构。本章后半部分的因子分析把 PCA 改写回生成模型，与 EM 再次衔接。下一章（ICA）则在非高斯源上把“不相关”加强为“独立”。

<br>

## 问题设定与预处理

本章输入是 $n$ 个样本 $x^{(1)},\ldots,x^{(n)}\in\mathbb{R}^{d}$，输出是主成分（一组正交方向）、每个样本在这些方向上的投影坐标，以及由坐标得到的重构。

第一步必须**中心化**：先中心化 $x^{(i)}\leftarrow x^{(i)}-\bar x$，其中 $\bar x=\frac1n\sum_i x^{(i)}$ 是样本均值。不中心化时，“方差最大的方向”会被均值本身拉偏：如果数据整体远离原点，第一主成分会大致指向均值方向，而不是数据内部的变化方向。

第二步视情况**标准化**（每一维除以它的标准差）。标准化与否会改变方向：未标准化时大尺度特征主导第一主成分。比如一维是以“米”计的身高、另一维是以“毫米”计的臂长，后者的数值方差大上百万倍，第一主成分几乎就是“臂长轴”，这只反映单位选择，而不是数据结构。各维单位不同或量纲不可比时应先标准化；各维本来同质（如像素灰度）时通常只中心化。

## 最大方差视角：从约束优化到特征向量

### 投影方差

取一个单位向量 $u\in\mathbb{R}^{d}$（$\|u\|_2=1$）。样本 $x^{(i)}$ 在 $u$ 方向上的投影长度是标量 $(x^{(i)})^{\top}u$。由于数据已中心化，投影坐标的均值为 $0$，所以投影方差就是投影长度平方的平均。单位向量 $u$ 上的投影方差为

$$
\frac1n\sum_{i=1}^{n}\big((x^{(i)})^{\top}u\big)^{2}=u^{\top}\Sigma u,\qquad
\Sigma=\frac1n\sum_{i=1}^{n}x^{(i)}(x^{(i)})^{\top}.
$$

中间一步是把 $\big((x^{(i)})^{\top}u\big)^{2}=u^{\top}x^{(i)}(x^{(i)})^{\top}u$ 写成二次型，再把求和移到中间。$\Sigma\in\mathbb{R}^{d\times d}$ 是（中心化后的）样本协方差矩阵，对称半正定。

### 拉格朗日乘子求解

要最大化 $u^{\top}\Sigma u$ 约束 $\|u\|_2=1$。约束不能省：没有约束时把 $u$ 放大就能让目标任意大。

> [!INFO]+ 从拉格朗日到 $\Sigma u=\lambda u$
>
> 目标 $u^{\top}\Sigma u$，约束 $u^{\top}u=1$。拉格朗日 $L=u^{\top}\Sigma u-\lambda(u^{\top}u-1)$。$\Sigma$ 对称，所以 $\nabla_u(u^{\top}\Sigma u)=2\Sigma u$，$\nabla_u(u^{\top}u)=2u$。对 $u$ 求导并令其为零：
>
> $$
> \nabla_u L=2\Sigma u-2\lambda u=0\implies\Sigma u=\lambda u.
> $$
>
> 也就是说，驻点 $u$ 必须是 $\Sigma$ 的特征向量，拉格朗日乘子 $\lambda$ 就是对应的特征值。
>
> 代回目标：$u^{\top}\Sigma u=\lambda u^{\top}u=\lambda$。因此在单位球上，投影方差等于对应特征值。最大方差取最大 $\lambda$；次大方差在与第一方向正交的子空间上继续求同样的问题（额外约束 $u^{\top}u_1=0$），得到第二大特征值的特征向量，依此类推，得到其余特征向量。
>
> 感性理解：$\Sigma u=\lambda u$ 表示 $u$ 已是协方差的主轴：$\Sigma$ 沿 $u$ 的作用不产生正交分量，只做伸缩，投影方差等于 $\lambda$。

结论：最大方差方向是最大特征值对应的特征向量。把特征值排成 $\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d\ge 0$，前 $k$ 个主成分是最大的 $k$ 个特征值方向 $u_1,\ldots,u_k$，且彼此正交（实对称矩阵不同特征值的特征向量必正交，重特征值时可以选成正交）。把它们按列排成 $U=[u_1,\ldots,u_k]\in\mathbb{R}^{d\times k}$，满足 $U^{\top}U=I_k$。

样本的低维坐标是 $z^{(i)}=U^{\top}x^{(i)}\in\mathbb{R}^{k}$，重构是 $\hat x^{(i)}=UU^{\top}x^{(i)}$（再加回均值）。$UU^{\top}$ 是到主子空间的正交投影矩阵。

## 最小重构误差视角：同一组向量

另一种出发点：要求一个列正交的 $U$（$U^{\top}U=I_k$），使所有样本被投影到 $U$ 的列空间后丢掉的部分最少。重构误差 $\sum_i\|x^{(i)}-UU^{\top}x^{(i)}\|_{2}^{2}$ 的最小化（$U^{\top}U=I_k$）给出同一组向量。

推导要点：由勾股定理，$\|x\|^2=\|UU^{\top}x\|^2+\|x-UU^{\top}x\|^2$。左边与 $U$ 无关，所以最小化残差等价于最大化投影部分 $\sum_i\|U^{\top}x^{(i)}\|^2=n\sum_{j=1}^{k}u_j^{\top}\Sigma u_j$，这正是上一节的最大方差目标。取 $U$ 为前 $k$ 个特征向量时，残差恰好是被丢掉的那些特征方向上的方差：

$$
\sum_i\|x^{(i)}-UU^{\top}x^{(i)}\|_{2}^{2}=n\sum_{j=k+1}^{d}\lambda_j
$$

（因子 $n$ 来自 $\Sigma$ 定义中的 $\frac1n$）。这说明丢掉的方差正好是末尾特征值之和，所以“最大方差”与“最小重构”是同一件事。前 $k$ 个特征向量张成最大方差子空间，也是 $L_2$ 重构误差最小的 $k$ 维子空间。

## 如何选择 $k$

第 $j$ 个成分的解释方差比例 $\lambda_j/\sum_{\ell}\lambda_{\ell}$ 用于选 $k$：常见做法是累计解释方差保留 $90\%$，或者画特征值随序号下降的碎石图（scree plot），取曲线明显变平的肘部。

需要注意，$k$ 不是越多越好：末尾成分常是噪声，丢掉它们反而起到降噪作用；下游分类有时只用前几个成分就够，保留太多会把噪声维度重新带回来。反过来，方差大也不等于对下游任务有用，后文人脸例子会说明这一点。

## 数值实现：用 SVD 而不是显式协方差

数值上对 $X\in\mathbb{R}^{n\times d}$（已中心化，每行一个样本）做 SVD：$X=U S V^{\top}$。注意这里的 $U$ 是 SVD 的左奇异向量（$n\times n$ 或 $n\times d$），与上文主成分矩阵同名但不是一回事。因为

$$
\Sigma=\frac1n X^{\top}X=\frac1n VS^{2}V^{\top},
$$

所以右奇异向量 $V$ 的前 $k$ 列就是主成分，$\Sigma$ 的特征值是 $s_j^{2}/n$（$s_j$ 为第 $j$ 个奇异值）。不要先形成 $X^{\top}X$ 再特征分解：显式相乘会把条件数平方，若 $d$ 大且病态，小特征值会被舍入误差吞掉，SVD 更稳。换言之，对 $X$ 做 SVD 得到同一组右奇异向量，避免显式形成 $X^{\top}X$。

下面的实现直接采用这一做法。`pca_fit` 返回均值、主成分矩阵和前 $k$ 个成分的解释方差比；`pca_transform` 求低维坐标；`pca_reconstruct` 从坐标回到原空间。

```python
def pca_fit(X, k):
    """X: (n, d)。返回均值、主成分 (d, k)、解释方差比。"""
    mu = X.mean(axis=0)          # 中心化用的均值 (d,)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k].T                 # 前 k 个右奇异向量，按列排 (d, k)
    ev = (S ** 2) / X.shape[0]   # 协方差特征值 s_j^2 / n
    return mu, V, ev[:k] / ev.sum()

def pca_transform(X, mu, V):
    return (X - mu) @ V          # 低维坐标 z = (x - mu)^T V，形状 (n, k)

def pca_reconstruct(Z, mu, V):
    return Z @ V.T + mu          # 重构 x_hat = V z + mu，形状 (n, d)
```

`np.linalg.svd` 返回的奇异值已按降序排列，所以 `Vt[:k]` 直接对应最大的 $k$ 个特征值。

> [!EXAMPLE]+ 例16 人脸与特征脸
>
> $100\times 100$ 灰度脸拉成 $10^{4}$ 维向量。对一批人脸做 PCA，每个主成分本身也是一张 $100\times 100$ 的“图”，称为特征脸（eigenface）。前几十个主成分（特征脸）主要对应光照与轮廓等大尺度变化。用 $k=50$ 重构仍可辨认，用 $k=2$ 只剩低频块。
>
> 失败情形：若训练脸都是正面、测试是侧面，测试样本在主成分子空间外的能量很大（即重构误差很大），在低维坐标上做距离比较会失败。这说明训练与部署的姿态分布不一致时，保留方差最大的坐标也不一定对下游任务有用。

<br>

## 性质与常见误区

PCA 的定位：找一组正交方向，使投影方差依次最大，等价于 $L_2$ 重构误差最小。来源是 Pearson / Hotelling 的主轴，不是聚类，也不是因果因子。

- **用途**：相对直接在原始高维上算距离，先丢掉末尾特征值，可降噪并可视化，也能减少下游模型的输入维数和计算量。
- **不是聚类，也不是因果因子**：PCA 只描述二阶统计（协方差），主成分是数学上的方差方向，不保证对应任何可解释的“因素”。
- **旋转不变**：任意正交混合后，主成分跟着转。也就是说，若把数据先乘一个正交矩阵 $R$，新的主成分正是 $R$ 乘旧主成分，PCA 的结果与坐标系的选择“同步旋转”。由此也可知，轴没有“真实名字”。
- **尺度敏感**：前文已述，标准化与否会改变方向。
- **可视化陷阱**：可视化前两维不等于“数据是二维的”，只是方差最大的二维窗。若前两个成分只解释了 $30\%$ 的方差，二维散点图会遗漏大量结构。

## 从投影到生成模型：概率 PCA 与因子分析

### 动机

普通 PCA 只是一个投影算法，没有概率模型，因此无法给出似然、无法处理缺失值、也不能说明“各维观测噪声多大”。概率 PCA / 因子分析把“投影”改回生成模型：假设观测由少数连续潜因子线性生成，再加噪声。另一个重要动机来自小样本：当 $n<d$ 或样本协方差奇异时，完整 $\Sigma$ 的 MLE 不存在（高斯似然中需要 $\Sigma^{-1}$，而秩不足的样本协方差不可逆），低秩加对角的结构仍然可估。

### 模型

经典因子分析

$$
z\sim\mathcal{N}(0,I_k),\qquad
\varepsilon\sim\mathcal{N}(0,\Psi),\qquad
x=\mu+\Lambda z+\varepsilon,
$$

其中 $z\in\mathbb{R}^{k}$ 是潜因子，$\Lambda\in\mathbb{R}^{d\times k}$ 是载荷矩阵（factor loadings），$\mu\in\mathbb{R}^{d}$ 是均值，$\Psi\in\mathbb{R}^{d\times d}$ 对角，表示每一维有自己独立的观测噪声方差。由于 $z$ 与 $\varepsilon$ 独立且都是高斯，$x$ 是高斯的线性组合，边缘 $x\sim\mathcal{N}(\mu,\Lambda\Lambda^{\top}+\Psi)$：均值是 $\mu$，协方差是 $\mathrm{Cov}(\Lambda z)+\mathrm{Cov}(\varepsilon)=\Lambda\Lambda^{\top}+\Psi$。这里需要估计的参数只有 $dk+d$ 个左右，而不是完整协方差的 $d(d+1)/2$ 个，这正是 $n<d$ 时仍能估计的原因。

$\Psi=\sigma^{2}I$（各维噪声相同）时称为概率 PCA：$\Lambda$ 的列张成的空间与普通 PCA 的主空间在 $\sigma\rightarrow 0$ 时重合。

### 用 EM 估计参数

$z$ 未观测，因此参数用 EM：E 步算 $E[z\mid x]$ 与 $E[zz^{\top}\mid x]$（联合高斯的条件分布仍是高斯，后验均值和二阶矩有闭式），M 步把这些期望代入完全数据对数似然，对 $\Lambda,\Psi$ 求导得到闭式更新。整个流程与高斯混合模型的 EM 同构，只是潜变量从离散簇标签变成连续向量。

### 旋转不定性

载荷 $\Lambda$ 仍有正交旋转不定性：对任意正交矩阵 $R$，$(\Lambda R)(\Lambda R)^{\top}=\Lambda\Lambda^{\top}$，而 $Rz$ 仍服从 $\mathcal{N}(0,I_k)$，所以 $\Lambda$ 和 $\Lambda R$ 给出完全相同的 $x$ 分布，数据无法区分二者。因此不能把某一列直接命名为“真实因子”而不加约束。

> [!NOTE]+ 因子分析
>
> 连续潜变量 $z$ 经载荷 $\Lambda$ 线性生成 $x$，再加对角噪声 $\Psi$。来源是心理测量里的公共因子模型，CS229 把它写成高斯生成模型并用 EM 估计。相对普通 PCA：PCA 通常不建各维不同的噪声 $\Psi$，因子分析允许各维观测噪声不同；$n<d$ 时完整协方差的 MLE 不存在，该低秩加对角结构仍可估。相对 ICA：因子是高斯，因此正交旋转不可辨识，不能把某列当作独立物理源。

<br>

> [!NOTE]+ 小结与衔接
>
> - PCA 求协方差 $\Sigma$ 的前 $k$ 个特征向量；最大方差与最小 $L_2$ 重构误差等价，丢掉的误差是 $n\sum_{j>k}\lambda_j$。
> - 实现用中心化数据的 SVD，主成分是右奇异向量，特征值为 $s_j^2/n$。
> - 先中心化；量纲不同时先标准化；用解释方差比或碎石图选 $k$。
> - 因子分析与概率 PCA 是对应的生成模型，用 EM 估计，有旋转不定性。
> - PCA 给出不相关的正交轴。高斯变量不相关即独立，因此高斯源上无法再往下分解。下一章处理非高斯源：目标从“保方差”换成“统计独立”。

<br>

# 独立成分分析（Independent Component Analysis, ICA）

经典场景是鸡尾酒会问题（cocktail party problem）：房间里有 $d$ 个人同时说话，$d$ 个麦克风放在不同位置，每个麦克风记录的是说话人的线性叠加（离得近的人声音大，权重大）。能否只凭麦克风录音，把每个人的声音分离出来？这称为盲源分离（blind source separation），“盲”指既不知道源信号，也不知道混合方式。

模型写成：观察 $x=As$，其中源（sources）$s\in\mathbb{R}^{d}$ 的各维统计独立且非高斯；混合矩阵 $A\in\mathbb{R}^{d\times d}$ 未知（假设可逆）。目标是恢复解混矩阵 $W\approx A^{-1}$，使 $y=Wx$ 的各分量尽量独立，从而 $y$ 就是源的估计。

为什么 PCA 不够：PCA 只能把观察变成不相关，而“不相关”只约束二阶矩。若源是高斯，不相关已等于独立，正交旋转无法辨识，$A$ 在旋转下不可恢复。因此 ICA 要求源非高斯：正是高阶统计量（非高斯性）提供了 PCA 没有用到的信息。与相邻章节的关系：上一章的主成分在高斯下已是独立方向；本章恢复线性混合的非高斯源，尺度与次序不可辨识；下一章离开线性混合，用逐步加噪的生成模型处理高维 $x$。

<br>

## 不可辨识性：哪些东西永远恢复不了

即使源非高斯，有些信息在原理上也无法从 $x$ 中恢复：

- **置换**：源的次序可任意交换。把 $s$ 的分量重新排列、同时把 $A$ 的列做同样排列，$x=As$ 不变，所以无法知道恢复出的第 1 个分量对应哪位说话人。
- **尺度与符号**：$s_j$ 乘 $c$ 等价于 $A$ 的第 $j$ 列除以 $c$。$c=-1$ 时就是符号翻转，对音频而言听起来一样。

因此通常约定各 $s_j$ 方差为 $1$，以消除尺度不定；次序与符号则只能接受。实际应用中（如鸡尾酒会）这些不定性通常无关紧要。

### 预处理：白化

在预处理中白化（whitening：先 PCA 再缩放到 $\Sigma=I$）。白化后的观察 $\tilde x$ 协方差为单位阵；若源也已约定为单位方差，则 $\tilde x=\tilde A s$ 中 $\tilde A\tilde A^{\top}=I$，即把 $A$ 缩到正交矩阵。于是只需在正交群上搜索，未知数从 $d^2$ 个降到约 $d(d-1)/2$ 个。白化本身就是“PCA + 每个方向除以标准差”，它把能用二阶统计做的事都做完了，剩下的旋转只能靠非高斯性确定。

## 密度变换与最大似然

### 线性变换下的密度

ICA 的经典解法（Bell–Sejnowski 一类目标）是写出 $x$ 的似然再最大化。为此需要把 $s$ 的密度变换成 $x$ 的密度。密度变换：若 $x=A s$、$s$ 密度为 $\prod_j p_s(s_j)$，则

$$
p_x(x)=p_s(Wx)\,|W|=\prod_{j=1}^{d}p_s(w_j^{\top}x)\,|W|.
$$

这里 $|W|$ 指行列式的绝对值 $|\det W|$，$w_j^{\top}$ 是 $W$ 的第 $j$ 行，所以 $w_j^{\top}x$ 就是第 $j$ 个源的估计。一个常见误区是直接写 $p_x(x)=p_s(Wx)$ 而漏掉 $|W|$。

> [!INFO]+ 密度变换里的 $|W|$ 从哪来
>
> $x=As$，$W=A^{-1}$，$s=Wx$。概率质量在可逆变换下守恒：小盒子 $ds$ 映成 $dx$，体积比是 $|\det A|=1/|W|$。因此
>
> $$
> p_x(x)\,|dx|=p_s(s)\,|ds|\implies p_x(x)=p_s(Wx)\,|W|.
> $$
>
> 一维例子：$s\sim\mathrm{Uniform}[0,1]$，$x=2s$，则 $x\sim\mathrm{Uniform}[0,2]$，密度是 $\frac12$。此时 $W=\frac12$，$p_s(Wx)\,|W|=1\cdot\frac12$，正确；若漏掉 $|W|$ 会得到密度 $1$，在 $[0,2]$ 上积分为 $2$，不是合法分布。
>
> 源独立则 $p_s(s)=\prod_j p_s(s_j)$，取对数后出现 $\sum_j\log p_s(w_j^{\top}x)+\log|W|$。梯度里 $(W^{\top})^{-1}$ 一项正是 $\nabla_W\log|W|$。高斯源的 $p_s$ 只依赖 $\|s\|_{2}$，在正交旋转下不变，于是 $W$ 无法从似然里辨识。
>
> 感性理解：可逆线性变换改变体积元，$p_x$ 必须乘 $|W|$ 才能保持概率质量。缺少 $\log|W|$ 时，似然可通过缩小 $W$ 而被人为抬高（所有 $w_j^{\top}x$ 都被压到密度峰值附近）。白化把观察协方差变成 $I$，搜索范围缩到正交群，$\log|W|$ 成为常数，只剩源密度的非高斯性用来辨识旋转。

### 对数似然

对 $n$ 个独立样本求和，得到对数似然

$$
\ell(W)=\sum_{i=1}^{n}\sum_{j=1}^{d}\log p_s(w_j^{\top}x^{(i)})+n\log|W|.
$$

### 选择源密度与梯度上升

$p_s$ 未知时，需要人为指定一个密度。它不必精确，只要“形状对”：语音等自然信号通常是超高斯（super-Gaussian，峰更尖、尾更厚）。CS229 讲义的做法是取源的累积分布函数为 sigmoid $\sigma(s)$，于是密度为 $p_s(s)=\sigma'(s)$，即 logistic 分布。选一个超高斯先验，例如 $p_s'(s)/p_s(s)=1-2\sigma(s)$（logistic 源）：由 $\sigma'=\sigma(1-\sigma)$ 得 $\log p_s=\log\sigma+\log(1-\sigma)$，求导为 $(1-\sigma)-\sigma=1-2\sigma(s)$。

对单个样本的 $\ell$ 求梯度：第一项对 $W$ 的梯度是向量 $\big(1-2\sigma(w_j^{\top}x)\big)_j$ 与 $x^{\top}$ 的外积，第二项用恒等式 $\nabla_W\log|W|=(W^{\top})^{-1}$。得到随机梯度上升

$$
W\leftarrow W+\alpha\big((1-2\sigma(Wx))x^{\top}+(W^{\top})^{-1}\big).
$$

其中 $\sigma(Wx)$ 逐元素作用，$\alpha$ 是学习率。

### 自然梯度与实践要点

每步计算 $(W^{\top})^{-1}$ 既慢又可能数值不稳。自然梯度右乘 $W^{\top}W$，避免显式求逆并改善几何：记 $y=Wx$，

$$
\big((1-2\sigma(y))x^{\top}+(W^{\top})^{-1}\big)W^{\top}W=\big(I+(1-2\sigma(y))y^{\top}\big)W,
$$

右边不再有矩阵求逆。

实践清单：实现前必须中心化、白化；实现后按方差排序分量，承认符号可翻转。

### 何时失败

ICA 失败模式：源不足 $d$ 个、非线性混合、高斯源、样本太少（高阶统计量估计噪声大）。脑电伪迹分离（把眨眼、肌电等伪迹从 EEG 中分出来）是成功案例；但把 ICA 分量直接当“真实物理源”需要额外实验约束。

## 实现

`whiten` 先中心化再做 SVD。设 $X_c=USV^{\top}$，则 $X_c V S^{-1}=U$，所以左奇异向量 $U$ 就是投影到主方向并除以奇异值后的坐标；$U^{\top}U=I$，再乘 $\sqrt n$ 后样本协方差 $\frac1n Z^{\top}Z=I$，即完成白化。`ica_logistic` 用全批量平均梯度执行上文的更新（`sigmoid` 假定已定义为逐元素的 $1/(1+e^{-x})$）。

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
    W /= np.linalg.norm(W, axis=1, keepdims=True)   # 每行归一化，初始尺度适中
    for _ in range(num_steps):
        Y = X @ W.T                                   # (n, d)，每行是 y = W x
        S = 1.0 - 2.0 * sigmoid(Y)                    # 1 - 2σ(y)
        grad = (S.T @ X) / n + np.linalg.inv(W.T)     # 平均梯度 + (W^T)^{-1}
        W += lr * grad
    return W, X @ W.T
```

> [!EXAMPLE]+ 例17 两个正弦加噪声
>
> 取两个源：$s_1=\sin t$，$s_2=\mathrm{sign}(\sin(0.3t))$（一条正弦波和一条方波，二者都非高斯），用某个可逆 $A$ 混合得 $x=As$。PCA 轴对准最大方差，得到的两条信号仍是混合波形。ICA 后一条接近正弦，一条接近方波，次序与正负号可能对调，这正是前文的不可辨识性。若把 $s_2$ 也改成高斯白噪声，恢复只到旋转不定：两个分量都无法干净分离。
>
> 需要注意源密度的选择：正弦与方波都是亚高斯（sub-Gaussian，超额峰度分别约为 $-1.5$ 与 $-2$），与上文 logistic（超高斯）先验的形状相反。实测把这组信号白化后直接跑 `ica_logistic`，得到的两个分量与两个源的相关系数都约为 $0.7$，即停在约 $45^\circ$ 的混合上，并没有分离；把 $1-2\sigma(y)$ 换成适合亚高斯源的得分函数 $-y^{3}$（对应 $p_s\propto e^{-s^{4}/4}$）后，相关系数可达约 $1.0$。改用 Laplace 这类超高斯源时，logistic 版本可以正常分离。所以“一条接近正弦、一条接近方波”的结果，前提是源密度的形状（超高斯或亚高斯）选对了，或者使用 FastICA 这类对两种情形都适用的方法。

<br>

## 对照：PCA、因子分析、ICA

ICA 的定位：假设 $x=As$ 且各源独立、非高斯，估计 $W\approx A^{-1}$。来源是鸡尾酒会与盲源分离，密度变换里的 $|W|$ 保证概率质量守恒。相对 PCA：目标是统计独立而不是最大方差，不要求轴正交。相对因子分析：源必须非高斯，否则正交旋转不可辨识。次序与符号无法恢复，分量不能直接当成“第 $j$ 个说话人”而不加额外约束。

| 方法 | 潜变量假设 | 噪声模型 | 轴的约束 | 不可辨识部分 |
| :--- | :--- | :--- | :--- | :--- |
| PCA | 无概率模型（只看方差） | 无，或 $\Psi=\sigma^{2}I$ 的概率 PCA | 正交，按方差排序 | 重特征值时子空间内的旋转 |
| 因子分析 | 高斯 $z\sim\mathcal{N}(0,I_k)$ | 对角 $\Psi$，各维可不同 | 低秩载荷 | 载荷的正交旋转 |
| ICA | 独立且非高斯的源 | 通常无噪声 | 不要求正交 | 次序、尺度与符号 |

> [!NOTE]+ 对照：PCA、因子分析、ICA
>
> PCA：正交、按方差排序、无噪声模型（或 $\Psi=\sigma^{2}I$ 的概率 PCA）。因子分析：低秩载荷加对角 $\Psi$，允许各维噪声不同，有旋转不定。ICA：源独立且非高斯，不要求正交，次序与尺度不可辨识。三者都在找 $x$ 的线性潜结构，目标函数不同，不能互换解释。

<br>

> [!NOTE]+ 小结与衔接
>
> - ICA 假设 $x=As$，源独立且非高斯；高斯源只能恢复到正交旋转。
> - 次序、尺度、符号永远不可辨识；约定源方差为 1。
> - 似然中的 $\log|W|$ 来自密度变换，梯度为 $(W^{\top})^{-1}$；logistic 源给出 $1-2\sigma(\cdot)$ 项。
> - 先中心化、白化，把搜索限制到正交矩阵；自然梯度可避免求逆。
> - 线性潜变量到此结束。高维图像或波形很少是 $As+\varepsilon$。下一章用逐步加噪与去噪学习 $p(x)$，潜变量变成整条噪声轨迹，目标仍是 ELBO。

<br>

# 扩散模型（Diffusion Models）

前两章的潜变量模型都是线性的，而真实图像、音频的分布高度非线性、多峰，很难写成 $As+\varepsilon$。扩散模型换了一个思路：把“生成一张图”这件难事拆成很多步容易的小事。前向过程（forward process）按固定日程把数据 $x_0$ 逐步加高斯噪声直到近似 $\mathcal{N}(0,I)$；这一步不需要学习。然后学习反向去噪（reverse denoising）：每一步只需把略带噪声的样本变得稍微干净一点。学会之后，从纯噪声出发反复去噪，就能采样出新的 $x_0$。

直觉：往一滴墨水里不断加清水，最后只剩均匀的淡色（对应纯高斯噪声）；如果能学会“时间倒流”的每一小步，就能从均匀淡色里重新凝聚出一滴墨。每一小步的加噪幅度很小，所以对应的反向步近似是高斯，可以用神经网络预测其均值。

与相邻章节的关系：训练目标仍是 ELBO，与 EM / VAE 同族，只是潜变量变成噪声轨迹 $x_1,\ldots,x_T$。下一章讨论如何把已预训练的表示用于下游任务，而不再从零估计 $p(x)$。

<br>

## 前向过程：逐步加噪

### 单步转移

离散时间前向（DDPM，Ho et al. 2020）：

$$
q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I).
$$

这里 $t=1,\ldots,T$（$T$ 通常取 $1000$ 量级），$x_t$ 与 $x_0$ 同维；$\beta_t\in(0,1)$ 是噪声日程（noise schedule），通常从很小的值（如 $10^{-4}$）递增到较大的值（如 $0.02$）。系数 $\sqrt{1-\beta_t}$ 先把信号略微缩小，再加方差为 $\beta_t$ 的噪声；这样设计的好处是，若 $x_{t-1}$ 方差为 $1$，$x_t$ 方差仍为 $(1-\beta_t)+\beta_t=1$，整个过程不会让数值爆炸。

### 边缘的闭式：一步跳到任意 $t$

记 $\alpha_t=1-\beta_t$。边缘有闭式

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
> 前缀系数是 $\sqrt{\bar\alpha_t}$。第 $s$ 步加入的噪声 $z_s$ 之后还要再经过第 $s+1,\ldots,t$ 步的缩放，所以它在 $x_t$ 中的系数是 $\sqrt{(1-\alpha_s)\alpha_{s+1}\cdots\alpha_t}$。独立高斯之和仍是高斯，方差逐项累加：
>
> $$
> \mathrm{Var}(x_t\mid x_0)=(1-\alpha_t)+\alpha_t(1-\alpha_{t-1})+\alpha_t\alpha_{t-1}(1-\alpha_{t-2})+\cdots+\alpha_t\cdots\alpha_2(1-\alpha_1)=\sum_{s=1}^{t}(1-\alpha_s)\frac{\bar\alpha_t}{\bar\alpha_s}.
> $$
>
> 原笔记此处写成 $(1-\alpha_t)\bar\alpha_{t-1}+(1-\alpha_{t-1})\bar\alpha_{t-2}+\cdots+(1-\alpha_1)$，逐项系数不对（例如 $z_t$ 是最后一步加入的，系数应为 $1-\alpha_t$ 而不是 $(1-\alpha_t)\bar\alpha_{t-1}$）；但两种写法的总和都等于 $1-\bar\alpha_t$，所以结论不变。
>
> 这是望远镜和：记 $P_s=\alpha_{s+1}\cdots\alpha_t$（$P_t=1$），则每一项 $(1-\alpha_s)P_s=P_s-P_{s-1}$，其中 $P_{s-1}=\alpha_sP_s$、$P_0=\bar\alpha_t$。从 $s=1$ 加到 $t$，中间项全部抵消，得到 $P_t-P_0=1-\bar\alpha_t$。（原笔记用的恒等式 $\bar\alpha_{s-1}-\bar\alpha_s= (1-\alpha_s)\bar\alpha_{s-1}$ 同样可以把它那一列求和成 $1-\bar\alpha_t$。）因此
>
> $$
> x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon.
> $$
>
> 训练时不必逐步加噪，对均匀抽样的 $t$ 一步到位。反向网络预测 $\varepsilon$，等价于在已知 $x_t$ 时估计 $x_0$ 的后验均值：由上式 $x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon)/\sqrt{\bar\alpha_t}$，对 $x_t$ 取条件期望可知 $E[x_0\mid x_t]$ 与 $E[\varepsilon\mid x_t]$ 是线性一一对应的。
>
> 感性理解：线性高斯转移的复合仍是高斯。$\sqrt{\bar\alpha_t}$ 是 $x_0$ 的剩余系数，$1-\bar\alpha_t$ 是累积噪声方差。$t$ 足够大时 $\bar\alpha_t\to 0$，边缘接近 $\mathcal{N}(0,I)$，与 $x_0$ 几乎无关。训练对均匀抽样的 $t$ 一步得到 $x_t$，网络估计该步的噪声 $\varepsilon$，等价于估计 $q(x_0\mid x_t)$ 的后验均值，而不必逐步模拟前向过程。

信噪比视角：$x_t$ 中信号方差与噪声方差之比为 $\bar\alpha_t/(1-\bar\alpha_t)$，随 $t$ 单调下降。噪声日程本质上就是在设计这条信噪比曲线。

## 反向过程与训练目标

### 反向参数化

真正的反向条件 $q(x_{t-1}\mid x_t)$ 依赖整个数据分布，无法计算；但当 $\beta_t$ 很小时它近似高斯。于是反向 $p_\theta(x_{t-1}\mid x_t)$ 参数化为高斯，均值由网络给出，方差通常固定为 $\sigma_t^2 I$（DDPM 取 $\sigma_t^2=\beta_t$ 或后验方差 $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$）。生成模型为 $p_\theta(x_{0:T})=p(x_T)\prod_t p_\theta(x_{t-1}\mid x_t)$，其中 $p(x_T)=\mathcal{N}(0,I)$。

### 从 ELBO 到噪声预测损失

与 VAE 一样，$\log p_\theta(x_0)$ 无法直接计算，于是最大化它的下界 ELBO，前向过程 $q(x_{1:T}\mid x_0)$ 扮演“编码器 / 后验近似”的角色，只是它是固定的、不含参数。ELBO 可以拆成每个 $t$ 上的一项 KL 散度 $\mathrm{KL}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)$。关键在于：**以 $x_0$ 为条件**时，$q(x_{t-1}\mid x_t,x_0)$ 是有闭式的高斯，其均值为

$$
\tilde\mu_t(x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon\Big),
$$

其中 $\varepsilon$ 是由 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$ 反解出的噪声。两个方差相同的高斯之间的 KL 只剩均值差的平方；若把网络均值写成同样的形式、只把真噪声 $\varepsilon$ 换成预测 $\varepsilon_\theta(x_t,t)$，每一项 KL 就正比于 $\|\varepsilon-\varepsilon_\theta(x_t,t)\|^2$，系数依赖于 $t$。

于是得到一种等价且稳定的训练：预测噪声 $\varepsilon_\theta(x_t,t)$，损失

$$
L=\mathbb{E}\_{x_0,\varepsilon,t}\big[\|\varepsilon-\varepsilon_\theta(x_t,t)\|_{2}^{2}\big].
$$

该式来自 ELBO 各项在高斯假设下的加权简化；实践常对 $t$ 均匀采样并忽略复杂权重（Ho et al. 称之为 $L_{\text{simple}}$，发现去掉权重后样本质量反而更好）。网络 $\varepsilon_\theta$ 输入带噪样本和时间步，输出与 $x_t$ 同形的噪声估计，图像上通常用 U-Net 或 Transformer。

### 采样

采样：从 $x_T\sim\mathcal{N}(0,I)$ 出发，按

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\Big)+\sigma_t z
$$

逐步走到 $t=0$。其中 $z\sim\mathcal{N}(0,I)$ 每步重新采样（最后一步 $t=1$ 时取 $z=0$），括号里的部分就是把 $\tilde\mu_t$ 中的真噪声换成网络预测；注意 $1-\alpha_t=\beta_t$。每一步都要调用一次网络，$T$ 步就是 $T$ 次前向计算，这正是扩散模型采样慢的来源。

连续时间观点把前向写成随机微分方程（SDE），反向 SDE 或概率流 ODE 给出同一边缘；实现时仍是离散积分。

## 实现

`q_sample` 用闭式一步得到 $x_t$ 及所用噪声；`ddpm_loss` 对每个样本均匀抽一个时间步，计算 $L_{\text{simple}}$。代码中时间下标从 $0$ 开始，`alphas_bar[t]` 对应 $\bar\alpha_{t+1}$。

```python
def q_sample(x0, t, alphas_bar, rng):
    """t: (n,) 的整数时间。"""
    eps = rng.normal(size=x0.shape)
    # 把 (n,) 的 alpha_bar 变形为 (n, 1, 1, ...)，以便对任意维度的 x0 广播
    a = alphas_bar[t].reshape(-1, *([1] * (x0.ndim - 1)))
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * eps, eps

def ddpm_loss(model, x0, alphas_bar, rng):
    n = x0.shape[0]
    t = rng.integers(0, len(alphas_bar), size=n)   # 每个样本独立均匀抽时间步
    xt, eps = q_sample(x0, t, alphas_bar, rng)
    pred = model(xt, t)                            # 网络预测噪声
    return float(np.mean((pred - eps) ** 2))
```

`alphas_bar` 可由 `np.cumprod(1.0 - betas)` 得到。

> [!EXAMPLE]+ 例18 一维双峰
>
> $x_0$ 来自两个相距很远的高斯。$t$ 小时 $x_t$ 仍清楚地位于某一峰附近，$\varepsilon_\theta$ 只需轻微去噪；$t$ 大时 $x_t$ 已分不清来自哪一峰，网络必须学会把质量拆回两峰（在两峰中间的点上，预测的去噪方向要随 $x_t$ 的微小变化向左或向右分岔）。若日程 $\beta_t$ 过快，中间 $t$ 的信噪比塌缩，可学习的过渡区间很短，训练信号变差。噪声日程 $\beta_t$ 直接决定中间时刻的信噪比，需要与学习率一并调节。

<br>

## 与其他生成模型的比较

扩散模型的定位：先按固定日程把 $x_0$ 加到近高斯噪声，再学反向去噪以采样新样本。来源是非平衡热力学（Sohl-Dickstein et al. 2015）与 DDPM：潜变量是整条噪声轨迹，训练目标仍是 ELBO。

- **相对 GAN**：似然有明确下界，训练更稳（只是一个回归损失），不必同时训一对网络，也较少出现模式崩塌。
- **相对 VAE**：不把 $x$ 压进低维 $z$，而是在同维噪声上逐步恢复；编码器是固定的加噪过程，不需要学习。
- **代价**：逐步采样慢。蒸馏、一致性模型与潜空间扩散（在自编码器的低维潜空间中做扩散，如 Stable Diffusion）是工程补救，不改变“加噪-去噪”的核心。

> [!NOTE]+ 小结与衔接
>
> - 前向 $q(x_t\mid x_{t-1})$ 固定、不学习；闭式 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$ 让训练可一步加噪。
> - 训练最小化噪声预测误差，它是 ELBO 的加权简化；预测噪声等价于估计后验均值。
> - 采样从 $\mathcal{N}(0,I)$ 出发逐步去噪，步数多因此慢。
> - 噪声日程决定各时刻信噪比，是最重要的超参之一。
> - 扩散从噪声生成新的 $x$。不少下游任务只需要一个已在大规模数据上训练过的表示 $f$。下一章讨论冻结、微调与 LoRA；表示 $f$ 本身的训练放在再下一章。

<br>

# 基础模型（Foundation Models）与适配

从零训练一个模型需要大量标注数据和算力，而很多下游任务（某医院的影像分类、某公司的客服问答）恰恰数据少。基础模型的思路是：先在大规模数据上预训练（pretraining）一个通用表示或生成模型，例如在数亿张图像或数万亿 token 的文本上；然后把它适配到具体下游任务。适配方式包括线性探针（linear probe）、全量微调（fine-tuning）与低秩适配（LoRA）。

核心问题是：在“只用不改”和“全部重训”之间选在哪里。改得越少越省算力、越不容易破坏预训练特征，但表达力也越弱。需要分清冻结表示与更新参数的代价。本章的输入是预训练权重与下游标注，输出是适配后的预测头或增量权重。

与相邻章节的关系：上一章说明如何采样新 $x$；本章说明如何复用已有 $f(x)$。探针、微调与 LoRA 改变的是如何使用已有 $f$，并不引入新的似然形式。下一章写 $f$ 本身如何用对比与检索学出来。

<br>

## 线性探针、全量微调与部分解冻

记预训练主干为 $f:\mathcal{X}\to\mathbb{R}^{p}$，它把输入映射成 $p$ 维特征。

- **线性探针**：冻结主干 $f$，只训 $W$ 使 $Wf(x)$ 拟合 $y$。这只是在固定特征上做一次逻辑回归或 softmax 回归，是凸问题，训练快、几乎不会过拟合。它用于诊断表示里已经线性可分的信息：探针准确率高，说明需要的信息已经“摆在表面”。
- **全量微调**：更新全部参数，表达力强，但在小数据上易毁预训练特征（灾难遗忘，catastrophic forgetting）：几百个样本的梯度足以把主干推离预训练时学到的通用结构。通常要配合小学习率与早停。
- **折中**：只解冻最后几块（靠近输出、最任务相关的层），或使用 LoRA。

## 低秩适配（LoRA）

### 动机与公式

对大模型做全量微调，每个下游任务都要保存一份完整权重（数十 GB），优化器状态还要再翻几倍。LoRA（Low-Rank Adaptation）的假设是：适配所需的权重改变量本身是低秩的，因此冻结原权重，只学一个低秩增量。对权重 $W\in\mathbb{R}^{m\times n}$ 学习低秩增量

$$
W'=W+\frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{m\times r},A\in\mathbb{R}^{r\times n}.
$$

$W$ 冻结，只训练 $A$ 与 $B$；$r$ 是秩，$\alpha$ 是缩放超参。通常把 $B$ 初始化为零、$A$ 随机初始化，这样训练开始时 $BA=0$，模型恰好等于预训练模型。

### 参数量与推理代价

$r\ll\min(m,n)$ 时增量参数约为 $r(m+n)$，而全量更新是 $mn$。例如 $m=n=4096$、$r=8$ 时，增量约 $8\times 8192=65536$ 个参数，只占 $4096^2\approx 1.68\times 10^{7}$ 的约 $0.4\%$。推理可把 $BA$ 合并进 $W$（直接存 $W'$），无额外延迟；也可以保留 $W$ 不变、为不同任务切换不同的小增量。

$\alpha/r$ 控制步幅：改变 $r$ 时保持 $\alpha$ 不变，增量的整体量级大致稳定，不必重新调学习率。需要注意，LoRA 不是正则的替代：参数少不等于不会过拟合，下游仍需验证集与早停。

> [!EXAMPLE]+ 例19 何时探针就够
>
> 图像主干在 ImageNet 上预训练，下游是十类物体，线性探针准确率已接近全量微调，说明类别信息已近线性可分，优先探针以节省算力。若下游是细粒度病斑且探针远差于微调（预训练数据里少有这类细微纹理差别），才值得解冻或 LoRA。项目里通常先做探针，确认表示不足后再微调或加 LoRA。

<br>

| 方式 | 训练参数 | 优点 | 风险 |
| :--- | :--- | :--- | :--- |
| 线性探针 | 只有头部 $W$ | 快、稳、可诊断表示质量 | 表示不够时上限低 |
| 部分解冻 | 最后几块 + 头部 | 折中 | 需选择解冻深度 |
| LoRA | 约 $r(m+n)$ 每矩阵 | 省显存、可合并、多任务切换 | 仍需验证集与早停 |
| 全量微调 | 全部 | 表达力最强 | 小数据灾难遗忘、存储昂贵 |

> [!NOTE]+ 小结与衔接
>
> - 先探针后微调：探针用来诊断表示是否已足够。
> - 全量微调表达力强，但小数据易灾难遗忘。
> - LoRA 学 $\frac{\alpha}{r}BA$ 的低秩增量，参数约 $r(m+n)$，推理时可合并无延迟。
> - 探针与 LoRA 默认 $f$ 已经有用。下一章补上 $f$ 的来源：监督预训练、对比学习、语义检索，以及把检索接到生成上的 RAG。

<br>

# 表示学习（Representation Learning）与检索增强

上一章默认已有一个好用的 $f$。什么叫“好用”？学习 $f(x)$ 使下游任务变容易：同类靠近（简单分类器就能分开）、检索可召回（语义相近的对象在向量空间里是近邻）、生成可条件化（生成模型可以“看着”这些向量写答案）。监督预训练、对比学习（contrastive learning）、语义检索（semantic retrieval）与 RAG 是四条常用路。

核心想法：把原始对象（图像、句子、文档）编码成向量，让“语义上的相似”变成“几何上的接近”。本章写出对比损失与检索-生成接口；输入原始对象，输出向量与最近邻或增强后的生成。

与相邻章节的关系：上一章冻结或低秩更新 $f$；本章回答 $f$ 从哪来。对比学习给出无标签的几何；检索把几何变成可检查的邻居；RAG 把邻居接到下一章的语言模型，而不是用参数记住全部事实。

<br>

## 监督预训练

最直接的办法：监督预训练，即在源任务上最小化标准损失（如 ImageNet 上的交叉熵），取倒数第二层当 $f$。倒数第二层已经把输入变换到“对分类有用”的空间，最后一层只是在它之上的线性分类器。迁移是否成功取决于源与目标的共享结构，不是参数量本身：在自然图像上训练的特征迁移到其他自然图像很好，迁移到频谱图或医学切片则未必。

## 对比学习与 InfoNCE

### 动机

监督预训练需要大量类标。对比学习不需要类标，而是自己构造“哪些应该相似”的信号：正对 $(x,x^{+})$（同一图像的两种增强、同一文档的两句，或配对的图像与描述文本），负对 $(x,x^{-})$（批次里的其他样本）。目标是让正对的表示靠近、负对的表示远离。

### 损失

用 $\mathrm{sim}(\cdot,\cdot)$ 表示相似度（通常是 L2 归一化后的内积，即余弦相似度），InfoNCE

$$
\mathcal{L}=-\log\frac{\exp(\mathrm{sim}(f(x),f(x^{+}))/\tau)}{\exp(\mathrm{sim}(f(x),f(x^{+}))/\tau)+\sum_{x^{-}}\exp(\mathrm{sim}(f(x),f(x^{-}))/\tau)}.
$$

读法：把它看成一个分类问题——在“1 个正例 + 若干负例”中认出正例，分子是正例的得分，分母是所有候选的得分之和，整体就是多项交叉熵（softmax 交叉熵），“正确类别”是正例。

### 温度与负例

$\tau$ 是温度：过小则 softmax 极尖，只关注最难负例，训练不稳；过大则所有候选得分趋于平均，梯度平坦。批次要够大才能提供有信息的负例（SimCLR 用数千的批次），或使用动量队列（MoCo）缓存以往批次的表示作为负例。

需要注意，对比学习不需要类标，但正对的两种增强应保持语义，否则梯度会把不相关的视图拉近。例如对数字图像做大角度旋转，“6”会变成“9”，却被当成正对。

> [!NOTE]+ InfoNCE
>
> 把“认出正对、推开负对”写成多项交叉熵：分母是正例与若干负例的相似度。来源是对比估计（NCE）与 SimCLR / CLIP 一类表示学习，不是生成模型的似然。相对分类交叉熵：不需要类标，正对由增强或配对文本定义。相对重构损失：优化的是几何上的可分性，而不是像素误差。温度 $\tau$ 与负例数量决定梯度集中在多难的负对上。

<br>

## 语义检索

有了编码器，检索就变成向量空间里的最近邻：语义检索把查询与文档编码到同一空间，按内积或余弦取 top-$k$。

- **评估**：用 Recall@$k$（正确文档是否出现在前 $k$ 个结果里）、nDCG（考虑排序位置的加权增益），而不是生成 BLEU。检索的产出是一个排序列表，应按排序指标评估。
- **索引**：文档上百万时逐一计算内积太慢，需要近似最近邻（ANN）索引，如 IVF（先聚类，只在最近的几个簇里搜）、HNSW（分层可导航小世界图）。索引（IVF、HNSW）是近似最近邻，召回率与延迟必须同时报：只报延迟不报召回率，无法判断是否以漏检为代价换来的速度。

## 检索增强生成（RAG）

RAG（retrieval-augmented generation）：先检索再生成。生成条件于查询与检索片段，减少幻觉并允许更新知识而不重训全部权重（换一份索引即可）。

但它并不保证正确：若检索错误，生成仍可能写出通顺但不正确的句子。检索命中与生成忠实度（答案是否忠于检索到的片段）需要分开评估。

> [!NOTE]+ RAG
>
> 生成时条件于检索到的片段，而不是把全部事实存进参数。来源是开放域问答：参数记忆难以及时更新，也难以审计出处。相对纯参数模型：知识可换索引、可检查邻居。相对只做检索：仍要一个条件语言模型把片段写成答案。检索错误会原样进入通顺的句子，因此命中率与忠实度必须分开报。

<br>

## 实现

`infonce` 实现批内对比：第 $i$ 行与第 $i$ 列是正对，同一行的其他列都当负例，因此正确“类别”是对角线；取 $i\to j$ 与 $j\to i$ 两个方向的平均（CLIP 风格的对称损失）。内部函数 `ce` 的第一个参数实际是 logits，先减行最大值再 softmax 以保证数值稳定。`retrieve` 用 `argpartition` 在 $O(N)$ 时间内取出前 $k$ 大，再只对这 $k$ 个排序。

```python
def infonce(z_i, z_j, tau=0.07):
    """z_i, z_j: (n, d) 的 L2 归一化正对。"""
    logits = (z_i @ z_j.T) / tau          # (n, n) 相似度矩阵，对角线为正对
    labels = np.arange(z_i.shape[0])
    # 对称两项：i->j 与 j->i
    def ce(logp, y):
        z = logp - logp.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.log(p[labels, y] + 1e-12).mean())
    return 0.5 * (ce(logits, labels) + ce(logits.T, labels))

def retrieve(query_vec, doc_vecs, k=5):
    scores = doc_vecs @ query_vec          # (N,) 每篇文档的内积得分
    idx = np.argpartition(-scores, kth=min(k, scores.size - 1))[:k]
    idx = idx[np.argsort(-scores[idx])]    # 只对前 k 个排序
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

> [!NOTE]+ 小结与衔接
>
> - 监督预训练取倒数第二层为表示，迁移效果取决于源与目标的共享结构。
> - InfoNCE 是“在正例与负例中认出正例”的交叉熵；温度和负例数量决定难度。
> - 语义检索用排序指标（Recall@$k$、nDCG）评估，ANN 索引需同时报召回率与延迟。
> - RAG 把检索接到生成，命中率与忠实度分开评估。
> - 表示给出向量空间。下一章在同一空间上对 token 序列做自回归：损失是 $\sum_t-\log p(x_t\mid x_{<t})$，核心模块是带因果掩码的注意力。

<br>

# 大语言模型（Large Language Models, LLM）

大语言模型做的事情听起来极其朴素：给定一段文本的前缀，预测下一个 token。在 token 序列上做自回归（autoregressive）语言建模，再用同一模型做提示、检索与微调。之所以有效，是因为要把“下一个词”预测准，模型必须隐式掌握语法、事实、常识乃至一定的推理模式；同一个预训练好的模型再通过提示或少量微调，就能被用于翻译、问答、写代码等各种任务。架构核心是 Transformer 与其注意力（attention）变体。

本章写出 next-token 损失、缩放点积注意力与常见适配；输入文本，输出 token 分布或解码序列。与相邻章节的关系：上一章的向量可用于检索；本章在同一向量空间上做自回归生成，损失是 next-token 负对数似然，架构是带掩码的注意力。下一章在这条损失之上加思维链与可验证奖励，不改注意力公式。

<br>

## 分词（tokenization）

模型不直接处理字符串，而是处理整数 id 序列。文本先切成词表中的子词（BPE、Unigram）：BPE（byte-pair encoding）从字符出发，反复把语料中最常见的相邻对合并成新符号；Unigram 则从大词表出发，按概率模型逐步删减。子词的好处是既不会像按字符切那样序列过长，也不会像按整词切那样遇到大量未登录词。

几个实际要点：

- 词表大小 $V$ 决定 softmax 最后一维，也决定输入嵌入矩阵和输出层的参数量（各为 $V\times d$）。
- 切分必须与训练时完全一致：多一个空格、不同 Unicode 规范化，都会改变 token 序列，使困惑度不可比。不同分词器的模型之间比较困惑度同样没有意义，因为“每个 token”的含义不同。
- BOS、EOS、PAD 等特殊符号会进入模型，与普通 token 一样参与注意力与损失；PAD 位置通常需要在注意力和损失中显式屏蔽。

## 自回归语言建模

### 链式分解

由概率的链式法则，任何序列分布都可以精确分解为一串条件分布的乘积，没有任何近似：

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

语言模型用一个神经网络参数化每个条件分布 $p_\theta(x_t\mid x_{<t})$，它是词表上的 $V$ 维 softmax。

### 训练损失

训练损失是平均负对数似然

$$
J=-\frac1N\sum_{\mathrm{seq}}\sum_{t}\log p_{\theta}(x_t\mid x_{<t}).
$$

外层对语料中所有序列求和，内层对序列中每个位置求和。$N$ 是 token 总数时，不同长度文档才可比较：若按序列数平均，长文档的损失天然更大。这本质上就是每个位置上的多分类交叉熵，所以训练 LLM 与训练 softmax 回归在损失层面是同一件事，只是“特征”由 Transformer 计算。

### 教师强制与暴露偏差

教师强制（teacher forcing）：训练时条件于真实前缀，不是模型自己的采样。这使得所有位置可以并行计算（配合因果掩码，一次前向就得到全部 $T$ 个位置的预测）。推理时条件于已生成前缀，训练与推理的条件分布不一致，这种分布偏移称为暴露偏差（exposure bias）：模型一旦生成了一个训练中少见的前缀，后续错误可能累积。

### 解码

训练好的模型给出每一步的分布，从中如何选出 token 由解码策略决定：

- **贪心**：每步取概率最大的 token，快但不多样，容易陷入重复。
- **温度采样**：把 logits 除以温度再 softmax，改变分布锐度；温度小于 1 更保守，大于 1 更发散。
- **nucleus（top-$p$）**：只在累计概率达到 $p$ 的最小 token 集合里采样，截掉长尾中的低概率怪词。

### 评估

评估用困惑度 $\exp(J)$：直观上它是模型在每一步“平均犹豫于多少个等可能的选项之间”，越低越好。但困惑度只衡量语言建模本身；任务指标（准确率、F1）必须在下游协议上另测。

## Transformer 架构

### 缩放点积注意力

对序列 $X\in\mathbb{R}^{T\times d}$（$T$ 个位置，每个位置 $d$ 维），先用三个可学习矩阵线性投影得到查询 $Q=XW_Q$、键 $K=XW_K$、值 $V=XW_V$（这里的 $V$ 指值矩阵，与上文词表大小同字母但含义不同），其中 $Q,K$ 的维度为 $d_k$。注意力为

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}+M\Big)V.
$$

读法：$QK^{\top}$ 是 $T\times T$ 的打分矩阵，第 $(t,t')$ 个元素表示位置 $t$ 对位置 $t'$ 的关注程度；对每一行做 softmax 得到权重，再对 $V$ 的各行做加权平均，所以每个输出位置是所有（允许看到的）位置的值的凸组合。

**为何除以 $\sqrt{d_k}$**：若 $q,k$ 的各分量独立、均值 0、方差 1，则点积 $q^{\top}k$ 的方差为 $d_k$。$d_k$ 大时打分数值很大，softmax 会饱和成接近 one-hot，梯度几乎为零。除以 $\sqrt{d_k}$ 把方差拉回 1。

**因果掩码**：$M$ 在因果语言模型里对 $t'\le t$（当前及之前的位置）为 $0$、对未来（$t'>t$）为 $-\infty$。原笔记写作“对 $t'<t$ 为 $0$”，漏掉了 $t'=t$；位置 $t$ 需要能看到自身（下面代码中 `np.triu(..., k=1)` 只屏蔽严格上三角，也是允许看自身的）。加上 $-\infty$ 后 softmax 中对应权重恰好为 $0$，所以预测 $x_{t+1}$ 时只用到 $x_{\le t}$。

### 多头、位置编码、FFN 与归一化

- **多头**：多头把 $d$ 切成 $h$ 组，每组 $d/h$ 维独立做注意力，再拼接并线性投影。不同的头可以关注不同类型的关系（如句法依存、指代）。
- **位置信息**：位置信息由正弦位置编码或 RoPE（旋转位置编码，把位置编码为对 $Q,K$ 的旋转，使打分只依赖相对位置）注入，否则自注意力置换等变：打乱输入顺序只会同样打乱输出，模型无法区分“A 打 B”与“B 打 A”。
- **FFN**：FFN 是逐位置双层 MLP（通常中间层宽度为 $4d$），对每个位置独立施加同一个非线性变换。注意力负责“位置之间交换信息”，FFN 负责“每个位置内部的加工”。
- **归一化与残差**：Pre-Norm（先 LayerNorm 再进子层）+ 残差使深层可训，残差路径让梯度能直接流过几十上百层。

### 复杂度

复杂度 $O(T^{2}d)$ 来自 $QK^{\top}$：$T\times d$ 乘 $d\times T$。序列长度翻倍，注意力计算与显存约变为四倍，这是长上下文的主瓶颈。

> [!NOTE]+ 自注意力
>
> 同一序列里每个位置用 $Q$ 去对所有位置的 $K$ 打分，再对 $V$ 做加权平均。来源是把编码器-解码器注意力改成序列内部交互，从而去掉 RNN 的逐步依赖。相对循环：训练可并行，长程依赖不经过逐步隐状态。相对卷积：感受野是全序列，而不是固定核宽。必须另加位置编码，否则对置换不变（严格说是置换等变：输出随输入顺序一起被打乱）；因果掩码防止读到未来 token。除以 $\sqrt{d_k}$ 是为了避免点积方差随维度增大而导致 softmax 饱和。

<br>

### 常见变体

- **多查询 / 分组查询注意力**（MQA / GQA）：多个查询头共享同一组（或少数几组）键值头，减少 KV 缓存。自回归推理时每生成一个 token 都要读取所有历史位置的 $K,V$，缓存大小直接决定显存和速度。
- **滑动窗口注意力**：每个位置只看最近的固定窗口，降低 $T^{2}$ 的代价。
- **交叉注意力**：查询来自解码器、键值来自编码器输出，用于编码器-解码器结构（如翻译）。
- **MoE**（mixture of experts）：FFN 换成若干专家，路由器只激活少数专家（如每个 token 选 top-2），参数量上升而每 token FLOP 近似不变；训练须防止专家崩塌（所有 token 进同一专家），常加负载均衡辅助损失。

## 使用方式：从提示到微调

同一个预训练模型有几种由轻到重的使用方式：

- **零样本提示**：只给指令，不给例子。依赖预训练里见过的任务格式。
- **上下文学习**（in-context learning）：提示里放若干 $(x,y)$ 对。这是推理期条件，不是梯度更新，权重完全不变；例子顺序与标签空间都会改变准确率，因此结果需要在多个排列上平均报告。
- **SFT**（supervised fine-tuning，监督微调）：在指令-回答对上继续最小化 next-token 损失（通常只在回答部分计算损失）。数据质量比数量更敏感：错误示范会被模仿。
- **RAG / 提示优化**：改检索与提示模板，不改权重。必须版本化提示，否则实验不可复现。

安全：预训练语料来自网络，模型会复述有害或隐私内容。部署需要拒绝策略、输出过滤与评估红队，不能只报困惑度。

## 实现

`scaled_dot_product_attention` 对批次和多头同时计算；`np.triu(..., k=1)` 生成严格上三角掩码（未来位置为 True），用 $-10^9$ 近似 $-\infty$。`next_token_nll` 用 log-sum-exp 技巧稳定地计算 log-softmax，再用高级索引取出每个位置目标 token 的对数概率。

```python
def scaled_dot_product_attention(Q, K, V, causal=True):
    """Q,K,V: (n, h, T, d_k)。"""
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # (n, h, T, T)
    if causal:
        T = scores.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # 严格上三角 = 未来位置
        scores = np.where(mask, -1e9, scores)
    z = scores - scores.max(axis=-1, keepdims=True)      # 数值稳定的 softmax
    w = np.exp(z)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V

def next_token_nll(logits, targets):
    """logits: (n, T, V), targets: (n, T)。"""
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))   # log-softmax
    n, T = targets.shape
    return float(-logp[np.arange(n)[:, None], np.arange(T)[None, :], targets].mean())
```

使用 `next_token_nll` 时，`targets` 应是输入右移一位后的序列：位置 $t$ 的 logits 预测的是 $x_{t+1}$。

> [!EXAMPLE]+ 例21 因果掩码漏了
>
> 训练时若不对未来位置置 $-\infty$，自注意力会读到 $x_{t+1}$ 再预测 $x_{t+1}$ 本身——等于把答案直接抄进输入，训练损失会异常偏低；推理阶段没有未来 token，性能会明显下降。（原笔记写作“读到 $x_{t+1}$ 再预测 $x_t$”；按上文“位置 $t$ 的输出预测 $x_{t+1}$”的约定，泄漏的实质是目标 token 本身出现在可见输入里，这里改成了这种说法。）可用 $T=3$ 的小例子检查：位置 $0$ 的权重是否只落在位置 $0$，位置 $1$ 的权重是否只落在位置 $0,1$。

<br>

> [!NOTE]+ 小结与衔接
>
> - 语言模型用链式法则分解 $p(x_{1:T})$，训练就是每个位置上的交叉熵，按 token 总数平均。
> - 分词必须与训练一致，否则困惑度不可比。
> - 注意力 $\mathrm{softmax}(QK^{\top}/\sqrt{d_k}+M)V$，因果掩码只允许看当前及之前位置；需另加位置编码。
> - 复杂度 $O(T^2d)$ 是长上下文瓶颈，GQA、滑动窗口、MoE 是常见工程变体。
> - 预训练与 SFT 都在最小化 next-token 损失。下一章改变的是推理期计算和奖励，不是注意力公式：思维链增加中间 token，RLVR 用可验证回报更新 $\pi$。

<br>

# LLM 推理：思维链与可验证奖励

上一章的解码只最大化 $p(x_t\mid x_{<t})$：模型读完题目就直接写答案，每个答案 token 只能分到固定量的计算（一次前向）。对多步算术、符号推导、编程这类题目，这往往不够。本章的思路有两层：其一，在生成过程中显式写出中间步骤，把中间 token 当作额外计算——这就是思维链（chain of thought）；其二，用可验证奖励（verifiable rewards）做强化学习，直接按“最终答案是否正确”来更新模型，而不是模仿人写的步骤。需要注意，思维链改变的是推理期计算，不一定改变预训练损失。

本章要区分提示出的链与 RLVR 训出的链；输入题目，输出带步骤的答案与可检查的正确性。与相邻章节的关系：本章把中间 token 当作额外计算，并用可检查的奖励更新策略；下一章（强化学习）给出一般 MDP 与贝尔曼方程，策略梯度是其中一条求解器。

<br>

## 思维链提示与自洽性

思维链：提示逐步书写中间步骤后再给答案（例如在提示中加入带推导过程的示例，或直接要求“一步一步地想”）。为什么有用：对算术与符号操作，中间 token 增加可用的计算步数——每多写一个 token，模型就多做一次完整的前向计算，并且可以把中间结果“写下来”供后面读取，相当于获得了草稿纸。

自洽性（self-consistency）对多条采样链的最终答案投票：用温度采样生成多条推理链，取出现次数最多的最终答案。直觉是错误的推理路径五花八门，正确路径往往殊途同归。

代价与局限：延迟与费用随链长上升（自洽性还要乘上采样条数）；链更长并不保证正确，错误中间步骤会被后续条件化并延续——一旦写下错误的中间结果，后面的 token 会把它当作已知事实继续推下去。

## 可验证奖励的强化学习（RLVR）

### 设定

RLVR（reinforcement learning with verifiable rewards）：环境给可程序化检查的奖励，例如单位测试通过、数学表达式等于标准答案。用强化学习的语言描述：策略是语言模型 $\pi_\theta$，动作是 token，状态是题目加已生成的前缀，一条完整回答结束后由验证器给出奖励（如对为 1、错为 0）。优化目标是期望可验证回报 $\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]$，可用策略梯度类方法优化。

### 与 RLHF 和 SFT 的区别

- **与 RLHF 的区别**：RLHF 的奖励来自从人类偏好数据学出来的奖励模型；RLVR 的奖励不是学出来的偏好模型，而是程序检查，减少了奖励黑客的一种来源（策略钻奖励模型误差的空子）。但模型仍可能钻验证器的空子（硬编码通过用例、只输出测试检查的格式）。
- **与 SFT 的区别**：长链推理把“思考 token”纳入动作空间，优化目标是期望可验证回报，不是模仿教师的中间步骤。模型可以自己摸索出人类示范中没有的推理方式，只要最终答案对就会被强化。

> [!WARNING]+ 不可验证任务不要假装可验证
>
> 开放写作没有自动判分器。强行用另一个 LLM 当奖励，会把该模型的偏好写进策略。CS229 的 RLVR 讨论限定在能写检查器的任务；其余应回到 SFT 或人工评估。

<br>

> [!NOTE]+ 小结与衔接
>
> - 思维链用中间 token 换取更多推理期计算；自洽性对多条链投票。
> - 链更长不保证更对，错误中间步骤会被延续；成本随链长上升。
> - RLVR 用程序可检查的奖励做策略优化，奖励不是学出来的，但验证器本身仍可能被钻空子。
> - 只有能写检查器的任务才适合 RLVR。
> - RLVR 已经在用策略梯度，但环境被简化成“答案对错”。下一章给出一般 MDP：状态、动作、转移、折扣。值迭代在已知 $P$ 时求最优值；语言模型只是策略的一种参数化。

<br>

<br>

# 强化学习（Reinforcement Learning）：MDP、值迭代与策略迭代

前面各章讨论的大多是监督学习：手里有固定的一批 $(x,y)$，模型的任务是把 $x$ 映射到正确的 $y$。很多实际问题并不长这样。让一台机器人学走路、让程序下棋、让直升机悬停，都没有人能在每一步告诉算法“此刻正确的动作 $a^{\ast}$ 是什么”；能拿到的只是一个事后才知道好坏的信号（摔倒了、赢了、偏离了目标）。而且算法自己选的动作会决定下一时刻看到什么状态，也就决定了以后能收集到什么数据。

强化学习（reinforcement learning, RL）研究的正是这种情形：智能体（agent）在环境中观察状态、选择动作，环境返回奖励（reward）与下一状态；目标是找到一个策略，使**期望累积回报**（return）最大。与监督学习相比有两点根本差别：

- 没有现成的 $(s,a^{\ast})$ 标签，只有对动作结果的评分；
- 动作影响未来的数据：监督学习的数据分布是固定的，而 RL 中看到的状态分布依赖于正在执行的策略 $\pi$，策略一改，数据分布跟着变。

本章的路线是：先用马尔可夫决策过程（MDP）把问题写成数学对象，再写出贝尔曼方程（Bellman equation），然后给出两种在已知转移与奖励时求最优策略的动态规划算法——值迭代（value iteration）与策略迭代（policy iteration）。整体上，输入是转移概率与奖励（或者与环境交互得到的轨迹），输出是策略。之后讨论两个现实障碍：模型未知时要先从数据里学模型；状态连续时表格存不下，只能离散化或用函数逼近值函数。下一章会看到，当动态是线性的、代价是二次的，同样的动态规划可以写成闭式的线性反馈。

<br>

## 马尔可夫决策过程（Markov Decision Process, MDP）

### 五元组定义

马尔可夫决策过程用五元组 $(S,A,P,R,\gamma)$ 描述一个序贯决策问题：

- $S$：状态集（state space），例如网格世界中的所有格子、机器人的位置与速度；
- $A$：动作集（action space），例如上下左右四个方向；
- $P(s'\mid s,a)$：转移概率（transition probability），在状态 $s$ 执行动作 $a$ 后到达 $s'$ 的概率，对每个 $(s,a)$ 有 $\sum_{s'}P(s'\mid s,a)=1$；
- $R(s,a)$ 或 $R(s,a,s')$：奖励函数，表示这一步得到的即时奖励（CS229 讲义中常简化为只依赖状态的 $R(s)$，三种写法本质相同，只是把期望放在不同位置）；
- $\gamma\in[0,1)$：折扣因子（discount factor），用来给远期奖励打折。

需要注意记号冲突：本章的 $A$ 表示动作集，策略梯度一章里 $A_t$ 表示优势，LQR 一章里 $A$ 表示动态矩阵，公平一章里 $A$ 表示敏感属性；根据上下文区分即可。

### 策略、回报与值函数

策略（policy）$\pi(a\mid s)$ 给出在状态 $s$ 下选每个动作的概率；确定性策略就是 $\pi(s)$ 直接输出一个动作。从时刻 $t$ 起的回报定义为折扣奖励之和：

$$
G_t=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k},
$$

其中 $R_{t+k}$ 是第 $t+k$ 步得到的奖励。折扣的作用有两个：一是数学上，若奖励有界 $|R|\le R_{\max}$，则 $|G_t|\le R_{\max}/(1-\gamma)$，无限时域的和一定收敛；二是建模上，表达“越早得到的奖励越值钱”。

衡量一个策略好坏的工具是值函数。状态值函数（state value function）与动作值函数（action value function，也叫 $Q$ 函数）分别是

$$
V^{\pi}(s)=E_{\pi}[G_t\mid s_t=s],\qquad
Q^{\pi}(s,a)=E_{\pi}[G_t\mid s_t=s,a_t=a].
$$

$V^{\pi}(s)$ 是“从 $s$ 出发、以后一直按 $\pi$ 行动，能拿到的折扣奖励期望”；$Q^{\pi}(s,a)$ 多固定了第一步的动作 $a$。两者关系是 $V^{\pi}(s)=\sum_a\pi(a\mid s)Q^{\pi}(s,a)$。

### 贝尔曼期望方程

值函数满足一个递归关系。把回报拆成“第一步的奖励”加“从下一状态起的折扣回报”：$G_t=R_t+\gamma G_{t+1}$。对两边取条件期望，第一步先按 $\pi$ 选动作 $a$，再按 $P$ 转移到 $s'$；而从 $s'$ 出发的期望回报恰好是 $V^{\pi}(s')$（这里用到马尔可夫性：到了 $s'$ 以后的事只和 $s'$ 有关）。于是得到贝尔曼期望方程（Bellman expectation equation）：

$$
V^{\pi}(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^{\pi}(s')\big].
$$

对有限的 $S$，这是关于 $|S|$ 个未知数 $V^{\pi}(s)$ 的线性方程组。

### 贝尔曼最优方程

最优值函数定义为所有策略中能取得的最大值 $V^{\ast}(s)=\max_{\pi}V^{\pi}(s)$。它满足贝尔曼最优方程（Bellman optimality equation）：

$$
V^{\ast}(s)=\max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^{\ast}(s')\big].
$$

与期望方程的唯一区别是：把“按 $\pi$ 对动作加权平均”换成“选最好的那个动作”。直觉是，最优地行动就是：这一步选使“即时奖励 + 折扣后的最优未来价值”最大的动作，以后也继续最优地行动。由于 $\max$ 的存在，它不再是线性方程组，需要迭代求解（见下一节）。

### 马尔可夫性与建模风险

马尔可夫性（Markov property）指：给定当前状态 $s$，未来与更早的历史条件独立。整个框架——值函数只是 $s$ 的函数、贝尔曼方程成立——都依赖这一点。因此状态必须包含所有对未来有用的信息。例如只给小车位置而不给速度，同一位置可能向左也可能向右运动，下一状态就不能只由当前“状态”决定。若真实决策依赖未写入 $s$ 的记忆，模型就设错了，再强的算法也只是在优化一个错误的对象。常见的补救是把历史（如最近几帧、速度）拼进状态。

> [!NOTE]+ MDP
>
> 用 $(S,A,P,R,\gamma)$ 描述序贯决策：当前状态已包含对未来有用的历史。这一框架来源于动态规划与最优控制，不是把监督学习的 $(x,y)$ 换成 $(s,a)$ 那么简单。相对监督学习：没有现成的 $a^{\ast}$，且 $\pi$ 改变后数据分布跟着变。相对多臂老虎机（multi-armed bandit）：老虎机只有一个状态，而 MDP 中状态会转移，当前动作影响以后能到达的状态。$\gamma$ 把无限时域的回报变成有限期望；$\gamma=1$ 且没有吸收态（absorbing state，进入后不再离开、奖励为零的终止态）时，目标可能发散。

<br>

## 值迭代与策略迭代（Value Iteration / Policy Iteration）

当 $P$ 与 $R$ 都已知、$S$ 与 $A$ 都有限时，求最优策略是一个纯计算问题，不需要和环境交互。两种经典的动态规划算法都把贝尔曼方程“当成更新规则来用”：值迭代直接迭代最优方程；策略迭代则交替地“评估当前策略”和“针对评估结果改进策略”。

### 值迭代

值迭代对所有 $s$ 同步更新（synchronous update，先用旧的 $V$ 算出全部新值再整体替换）：

$$
V(s)\leftarrow\max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V(s')\big],
$$

直到 $\|V_{\mathrm{new}}-V\|_{\infty}$ 小于阈值。也可以异步更新（asynchronous update，逐个状态更新并立刻使用新值），同样收敛，往往还更快。初值取 $V=0$ 即可。

收敛到 $V^{\ast}$ 后，对 $V$ 做一步贪心（greedy）：

$$
\pi(s)=\arg\max_a\sum_{s'}P(s'\mid s,a)\big[R+\gamma V(s')\big]
$$

就得到最优策略（若有多个动作并列最优，任选其一都最优）。

**为什么收敛。** 把右端看成作用在整个向量 $V\in\mathbb{R}^{|S|}$ 上的算子，记为贝尔曼最优算子 $T$。对任意两个值函数 $V,V'$，利用 $|\max_a f(a)-\max_a g(a)|\le\max_a|f(a)-g(a)|$ 以及 $\sum_{s'}P(s'\mid s,a)=1$，可得

$$
\|TV-TV'\|_{\infty}\le\gamma\|V-V'\|_{\infty}.
$$

即 $T$ 是模为 $\gamma$ 的压缩映射（$\gamma$-contraction）。由压缩映射定理（Banach 不动点定理），$T$ 有唯一不动点，就是 $V^{\ast}$，且从任何初值出发，误差每轮至少缩小为原来的 $\gamma$ 倍：$\|V_k-V^{\ast}\|_{\infty}\le\gamma^{k}\|V_0-V^{\ast}\|_{\infty}$。所以迭代线性收敛，速率由 $\gamma$ 决定，$\gamma$ 越接近 $1$ 越慢。例如 $\gamma=0.99$ 时，由 $\gamma^{k}=1/e$ 得 $k=1/(-\ln 0.99)\approx 99.5$，即误差上界缩小到原来的 $1/e$ 大约需要 $100$ 轮（一般地约 $1/(1-\gamma)$ 轮）；要把误差上界缩小 $10^{-6}$ 倍则需要约 $1400$ 轮。

### 策略迭代

策略迭代从任意策略出发，交替进行两步：

1. **策略评估（policy evaluation）**：求出当前策略的 $V^{\pi}$。由于贝尔曼期望方程 $V=T^{\pi}V$ 是线性的，可以直接解线性方程组：记 $R_{\pi}(s)$ 为按 $\pi$ 的期望即时奖励、$P_{\pi}$ 为按 $\pi$ 的状态转移矩阵，则 $V^{\pi}=(I-\gamma P_{\pi})^{-1}R_{\pi}$；也可以反复迭代 $V\leftarrow T^{\pi}V$ 评估到近似。
2. **策略改进（policy improvement）**：对 $V^{\pi}$ 贪心，得到新策略 $\pi'(s)=\arg\max_a\sum_{s'}P(s'\mid s,a)[R+\gamma V^{\pi}(s')]$。

可以证明（策略改进定理）新策略不会比旧策略差：$V^{\pi'}(s)\ge V^{\pi}(s)$ 对所有 $s$ 成立。有限 MDP 上确定性策略只有 $|A|^{|S|}$ 个，而每轮严格改进直到无法再改进，因此策略单调改进，并在有限步内到达最优；停下时 $\pi$ 对自己的值函数贪心，说明 $V^{\pi}$ 满足最优方程。

### 两者对比

策略迭代每轮评估比值迭代“彻底”（把当前策略的值算到底），因此轮数通常更少，但单轮更贵：解线性方程组的代价约为 $O(|S|^{3})$。值迭代每轮只做一次 $O(|S|^{2}|A|)$ 的扫描，轮数多但每轮便宜。小规模问题常用策略迭代，状态多时值迭代更实用。

> [!NOTE]+ 值迭代与策略迭代
>
> 值迭代反复用贝尔曼最优算子更新 $V$，再贪心出 $\pi$。策略迭代先把当前 $\pi$ 评估到底，再改进。二者都来自动态规划，要求已知 $P,R$ 且 $S,A$ 有限。相对监督学习里的一次梯度下降：这里是压缩映射迭代，初值不影响最优解，只影响步数。相对后面的策略梯度：必须能对 $a$ 做 $\arg\max$，并且用得起模型或表格。$\gamma$ 接近 $1$ 时收缩变慢。

<br>

## 模型未知时：从数据中学习模型

前面假设 $P,R$ 已知。现实中常常只能和环境交互：执行若干条轨迹，记录每一步的 $(s,a,r,s')$。一个直接的思路是**先估计模型，再规划**，这称为基于模型的强化学习（model-based RL）。

用轨迹计数估计转移概率：

$$
\hat P(s'\mid s,a)=\frac{N(s,a,s')}{N(s,a)},
$$

即 $\hat P(s'\mid s,a)=N(s,a,s')/N(s,a)$，其中 $N(s,a)$ 是在状态 $s$ 执行动作 $a$ 的次数，$N(s,a,s')$ 是其中转移到 $s'$ 的次数；这正是多项分布的极大似然估计。$\hat R$ 取该 $(s,a)$ 下观测到的平均奖励。然后在 $\hat P,\hat R$ 上跑值迭代（或策略迭代）得到策略，再用新策略继续收集数据、更新估计，如此循环。

需要注意两个坑：

- **未访问的状态–动作对。** 若某个 $(s,a)$ 从未被尝试，$N(s,a)=0$，上式无定义，估计也不可靠（CS229 讲义中的做法是此时令 $\hat P$ 为均匀分布）。如果总是按当前最优策略行动，算法可能永远不去尝试那些“看起来不好但其实很好”的动作。因此需要探索（exploration），常见方法有 $\varepsilon$-贪心（以概率 $\varepsilon$ 随机选动作）、RMax（对没见过的状态乐观地假设奖励最大，从而主动去访问）、后验采样（posterior sampling / Thompson sampling，从模型的后验中抽一个模型并对它最优地行动）。
- **规模。** 计数表的大小是 $|S|^{2}|A|$，只在小状态空间可行；连续或巨大的 $S$ 时，模型本身也要参数化，例如用线性回归或神经网络拟合 $s_{t+1}\approx f(s_t,a_t)$。

<br>

## 连续状态空间：离散化与值函数逼近

机器人、车辆、直升机的状态是 $\mathbb{R}^{d}$ 中的连续向量（位置、速度、角度……），值函数无法再存成一张有限的表格。有两条路。

### 离散化

最朴素的做法是离散化（discretization）：把 $\mathbb{R}^{d}$ 切成格子，每个格子当成一个离散状态，然后按表格 MDP 处理。它的问题在于两头为难：

- 格子粗，则量化误差大：同一格内的状态被迫共用一个值和一个动作，而真实值函数在格内可能变化很大；
- 格子细，则 $|S|$ 指数爆炸：每维切 $k$ 份，总格子数是 $k^{d}$，这就是维数灾难（curse of dimensionality）。例如 $d=10$、每维 $10$ 份就有 $10^{10}$ 个状态。

因此离散化只适合 $d$ 很小且动态光滑的情形。按 CS229 讲义的经验，离散化在 $1$–$2$ 维问题上效果很好，精心设计（如选好每维的格点）时可用到约 $4$ 维，极少用于 $6$ 维以上。

### 值函数逼近

更通用的做法是值函数逼近（value function approximation）：用带参数的函数 $V(s;w)$ 或 $Q(s,a;w)$ 代替表格，参数 $w$ 的个数远小于状态数，相近的状态通过共享参数自动“泛化”。

- **线性逼近。** 取特征 $\phi(s)$（例如状态分量及其平方、交叉项），令 $V(s)=w^{\top}\phi(s)$。训练时可以对贝尔曼残差，或对 TD 目标（temporal-difference target）$r+\gamma V(s';w)$ 与当前估计之差做随机梯度。CS229 讲义中的拟合值迭代（fitted value iteration）就是这一思路：在一批采样状态上用一步贝尔曼备份算出目标值 $y^{(i)}$，再用线性回归拟合 $w$，然后重复。
- **非线性逼近（神经网络）。** 即 DQN（Deep Q-Network）一类方法。训练目标是

  $$
  r+\gamma\max_{a'}Q(s',a';w^{-}),
  $$

  其中 $w^{-}$ 是目标网络（target network）的参数，定期从 $w$ 复制过来，使回归目标在一段时间内保持不动；同时用经验回放缓冲（replay buffer）存储过去的转移并随机抽取小批量，打破同一条轨迹内样本的强相关性。

逼近的代价是理论保证变弱：贝尔曼算子是压缩映射，但“贝尔曼备份 + 投影到函数类”不一定还是压缩，迭代可能振荡甚至发散，“收敛到 $V^{\ast}$”不再有保证。实践中必须监控策略的评估回报（实际跑若干回合的平均回报），不能只看 TD 损失下降——TD 损失小只说明网络和自己的目标一致，不代表策略好。

> [!INFO]+ 贝尔曼方程在算什么
>
> $V^{\pi}(s)$ 是“从 $s$ 出发按 $\pi$ 走到无穷的折扣奖励期望”。把第一步的期望奖励写成 $\sum_a\pi(a\mid s)\sum_{s'}P R$，其余从 $s'$ 出发的期望恰是 $V^{\pi}(s')$，再乘 $\gamma$。这不是新假设，只是全期望定律（law of total expectation）加上马尔可夫性。值迭代把等号换成赋值，并用 $\max_a$ 代替 $\sum_a\pi$，相当于同时做评估与改进。初值 $V=0$ 即可；最优解与初值无关，只影响迭代次数。
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

## 值迭代与策略评估的 NumPy 实现

下面的实现把转移存成形状为 $(S,A,S)$ 的数组 `P`，`P[s, a, s2]` 即 $P(s_2\mid s,a)$；奖励 `R` 可以是 $(S,A,S)$（依赖下一状态）或 $(S,A)$（只依赖当前状态与动作）。`value_iteration` 每轮先算出所有 $Q(s,a)$，再对动作取最大值得到新的 $V$；收敛后对 $Q$ 取 `argmax` 即贪心策略。`policy_evaluation` 用迭代法求确定性策略 `pi`（形状 $(S,)$，每个状态一个动作）的 $V^{\pi}$，可作为策略迭代中的评估步。

```python
# 需要先 import numpy as np
def value_iteration(P, R, gamma=0.95, tol=1e-8, max_iter=10000):
    """P: (S, A, S), R: (S, A, S) 或 (S, A)。"""
    S, A, _ = P.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = np.zeros((S, A))
        for a in range(A):
            if R.ndim == 3:
                # 期望即时奖励：对下一状态按转移概率加权
                reward = (P[:, a, :] * R[:, a, :]).sum(axis=1)
            else:
                reward = R[:, a]
            # 贝尔曼备份：即时奖励 + 折扣后的下一状态值期望
            Q[:, a] = reward + gamma * P[:, a, :] @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:   # 无穷范数收敛判据
            V = V_new
            break
        V = V_new
    pi = Q.argmax(axis=1)   # 对 Q 贪心得到策略
    return V, pi, Q

def policy_evaluation(P, R, pi, gamma=0.95, tol=1e-8):
    S, A, _ = P.shape
    V = np.zeros(S)
    while True:
        V_new = np.zeros(S)
        for s in range(S):
            a = pi[s]
            reward = (P[s, a] * R[s, a]).sum() if R.ndim == 3 else R[s, a]
            V_new[s] = reward + gamma * P[s, a] @ V   # 贝尔曼期望备份
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
```

把 `policy_evaluation` 与“对 $V^{\pi}$ 贪心”交替调用，直到策略不再变化，就得到完整的策略迭代。需要注意，停止判据 `tol` 衡量的是相邻两轮之差，不是与真值的误差；由压缩性，真实误差上界约为 $\gamma\,\mathrm{tol}/(1-\gamma)$，$\gamma=0.95$ 时约是 `tol` 的 $19$ 倍。

> [!EXAMPLE]+ 例22 网格世界
>
> $4\times 4$ 格子，目标格奖励 $+1$ 并终止，其余每步 $-0.04$，四方向动作，撞墙停留。$\gamma=0.99$ 时最优策略沿最短路走，因为活着的每一步都扣分，多走一步就多扣 $0.04$。用上面的 `value_iteration` 实算（目标设在右下角），离目标 $k$ 步的格子 $V$ 依次为 $1,\ 0.95,\ 0.9005,\ 0.8515,\ \ldots$，左上角（$6$ 步）为 $0.7074$，每多一步约少 $0.05$。
>
> 若把逐步奖励改为 $0$，绕路与最短路的区别只剩折扣：$\gamma=0.99$ 时 $V=\gamma^{k}$，左上角为 $0.9415$，相邻步只差约 $1\%$，最短路仍严格最优，但值的差距小，近似误差或噪声稍大就可能盖过它；若进一步取 $\gamma=1$，所有格子的 $V$ 都等于 $1$，多种绕路与最优路的 $V$ 完全相同（只差在数值精度），贪心策略可能绕远。（原笔记把 $\gamma=0.99$ 与“只差数值精度”放在一起，严格说只有 $\gamma=1$ 时才是真正的并列。）这说明奖励塑形（reward shaping）改变的是最优策略本身，而不是“同一任务的另一写法”；设计奖励时要想清楚它鼓励的到底是什么行为。

<br>

> [!NOTE]+ 对照：监督与 RL
>
> 监督：数据分布固定，损失对每个 $(x,y)$ 可单独计算。RL：动作改变下一状态，也改变以后能看见的数据。值函数把“以后的奖励”收成一个数，贝尔曼方程是全期望定律，不是新的建模假设。没有 $P$ 时，要么先估模型（基于模型的 RL），要么直接估 $Q$（如 DQN）或 $\nabla J$（如策略梯度）。

<br>

> [!NOTE]+ 小结与衔接
>
> - MDP 用 $(S,A,P,R,\gamma)$ 描述序贯决策，值函数满足贝尔曼期望方程与最优方程。
> - 值迭代与策略迭代在已知 $P,R$ 时求 $V^{\ast}$ 与最优策略；贝尔曼最优算子的 $\gamma$-压缩性保证收敛。
> - 模型未知时可计数估计 $\hat P,\hat R$，但需要探索；连续状态必须离散化或用函数逼近，逼近会削弱收敛保证。
> - 值迭代对一般有限 $S$ 反复扫描。若动态是线性的、代价是二次的，动态规划可以一次回传到底，并得到线性反馈。下一章写这个闭式解，以及如何对非线性 $f$ 做局部 LQR。

<br>

# 线性二次调节器（LQR）：Riccati 回传、DDP 与 LQG

上一章的值迭代对一般的有限状态空间反复扫描；连续状态时只能离散化或逼近。但有一类连续控制问题可以**精确**求解：动态是线性的、代价是二次的。这类问题叫线性二次调节器（Linear Quadratic Regulator, LQR）。典型场景是让小车停在原点、让倒立摆保持竖直、让飞行器沿给定轨迹飞：状态偏离目标要付代价，用力过猛也要付代价。

核心想法一句话：在线性动态下，二次的值函数经过一步动态规划后仍是二次的，所以值函数始终可以用一个矩阵 $P_t$ 表示；从终点往回推这个矩阵（Riccati 回传），每一步的最优控制都是状态的线性反馈（linear feedback）$u_t=-K_tx_t$。输入是 $(A,B,Q,R)$，输出是增益序列 $K_t$。

本章依次给出：有限时域（finite horizon）离散 LQR 的推导与 Riccati 递推；非线性系统如何在名义轨迹上线性化，得到迭代 LQR / 微分动态规划（DDP）；状态不可直接观测且有高斯噪声时的 LQG。下一章则不再需要模型或线性化，直接对参数化策略 $\pi_{\theta}$ 求目标 $J$ 的梯度。

<br>

## 问题设定：线性动态与二次代价

考虑有限时域 MDP：时间 $t=0,\ldots,T$，总代价是各步代价之和（可加）。状态 $x_t\in\mathbb{R}^{n}$，控制 $u_t\in\mathbb{R}^{m}$，线性动态与二次代价为

$$
x_{t+1}=A x_t+B u_t,\qquad
J=x_T^{\top}Q_T x_T+\sum_{t=0}^{T-1}\big(x_t^{\top}Q x_t+u_t^{\top}R u_t\big).
$$

其中 $A\in\mathbb{R}^{n\times n}$ 描述系统自身如何演化，$B\in\mathbb{R}^{n\times m}$ 描述控制如何作用到状态；$Q,Q_T\in\mathbb{R}^{n\times n}$ 半正定，惩罚状态偏离原点（$Q_T$ 是终端代价）；$R\in\mathbb{R}^{m\times m}$ 正定，表示控制要花代价——正定性保证下面对 $u$ 的最小化有唯一解。这里的目标是**最小化**代价 $J$，相当于奖励取负。（CS229 讲义还允许高斯过程噪声 $w_t$，由于期望下噪声只给值函数加常数，不改变最优增益。）

## Riccati 回传

### 结论

值函数保持二次形式 $V_t(x)=x^{\top}P_t x$（若动态是仿射的，即带常数项，则还要加线性项与常数项）。从终点 $P_T=Q_T$ 开始向前回传：

$$
\begin{aligned}
K_t&=(R+B^{\top}P_{t+1}B)^{-1}B^{\top}P_{t+1}A,\\
P_t&=Q+A^{\top}P_{t+1}A-A^{\top}P_{t+1}B K_t,\\
u_t^{\ast}&=-K_t x_t.
\end{aligned}
$$

第二式就是离散 Riccati 方程（discrete Riccati equation），代入 $K_t$ 可写成 $P_t=Q+A^{\top}P_{t+1}A-A^{\top}P_{t+1}B(R+B^{\top}P_{t+1}B)^{-1}B^{\top}P_{t+1}A$。注意 $K_t$ 与 $P_t$ 都不依赖于具体的 $x_t$，可以离线一次算好；运行时只需做矩阵乘法 $-K_tx_t$。

### 推导

推导用归纳法加动态规划。

> [!INFO]+ Riccati 回传在最小化什么
>
> 假设从 $t+1$ 起最优代价已是二次，$V_{t+1}(x)=x^{\top}P_{t+1}x$。当前步的最优代价等于“当前步代价 + 下一状态的最优代价”对 $u$ 取最小：
>
> $$
> V_t(x)=\min_u\Big[x^{\top}Qx+u^{\top}Ru+(Ax+Bu)^{\top}P_{t+1}(Ax+Bu)\Big].
> $$
>
> 方括号里是 $u$ 的二次函数，二次项系数 $R+B^{\top}P_{t+1}B$ 正定（$R$ 正定、$P_{t+1}$ 半正定），所以驻点就是唯一的全局最小点。对 $u$ 求导：$2Ru+2B^{\top}P_{t+1}(Ax+Bu)=0$，即 $(R+B^{\top}P_{t+1}B)u=-B^{\top}P_{t+1}Ax$，解出
>
> $$
> u^{\ast}=-(R+B^{\top}P_{t+1}B)^{-1}B^{\top}P_{t+1}A\,x=-K_t x.
> $$
>
> 代回二次型，收集 $x^{\top}(\cdot)x$ 的系数：展开得 $x^{\top}(Q+A^{\top}P_{t+1}A)x+2x^{\top}A^{\top}P_{t+1}Bu^{\ast}+u^{\ast\top}(R+B^{\top}P_{t+1}B)u^{\ast}$，再用 $(R+B^{\top}P_{t+1}B)u^{\ast}=-B^{\top}P_{t+1}Ax$ 把最后一项化为 $-x^{\top}A^{\top}P_{t+1}Bu^{\ast}$，合并得 $P_t=Q+A^{\top}P_{t+1}A-A^{\top}P_{t+1}BK_t$。终点 $P_T=Q_T$ 启动回传。线性动态加二次代价保证值函数永远是二次，归纳不断裂。
>
> 感性理解：二次代价在线性动态下对值函数封闭：若 $V_{t+1}$ 是二次，则 $V_t$ 仍是二次。对当前 $u$ 求最小得到线性反馈 $u=-K_t x$。$R$ 增大则控制更贵，$K_t$ 减小；$P_{t+1}$ 增大则未来状态代价更高，当前步更愿意施加控制。$P_t$ 把 $t$ 之后的全部二次代价折进当前状态。

### 无限时域与实际注意事项

无限时域、定常 $(A,B)$ 时，回传足够多步后 $P_t$ 收敛到代数 Riccati 方程（algebraic Riccati equation）$P=Q+A^{\top}PA-A^{\top}PB(R+B^{\top}PB)^{-1}B^{\top}PA$ 的解，增益 $K$ 恒定，得到一个时不变的线性控制器。

两个常见坑：

- **可控性。** 若 $(A,B)$ 可控性不足（某些状态方向任何控制都影响不到），就无法任意配置闭环极点，那些方向上的不稳定模态无法被镇定。
- **$R$ 的大小。** $R$ 过小意味着控制几乎免费，求出的增益很大，控制动作猛烈，对模型误差和执行器饱和都很敏感；$R$ 太大则响应迟缓。

<br>

## 非线性系统：线性化、迭代 LQR 与 DDP

真实系统的动态通常是非线性的 $x_{t+1}=f(x_t,u_t)$。思路是：LQR 只能解线性-二次问题，那就在一条名义轨迹（nominal trajectory）附近把问题**局部**近似成线性-二次问题。

设名义轨迹为 $(\bar x_t,\bar u_t)$，对 $f$ 做一阶泰勒展开：

$$
x_{t+1}\approx f(\bar x_t,\bar u_t)+A_t(x_t-\bar x_t)+B_t(u_t-\bar u_t),\qquad A_t=D_x f,\ B_t=D_u f,
$$

其中雅可比矩阵 $A_t=D_x f$、$B_t=D_u f$ 都在 $(\bar x_t,\bar u_t)$ 处取值。以偏差 $\delta x_t=x_t-\bar x_t$、$\delta u_t=u_t-\bar u_t$ 为变量，这就是一个时变线性系统（若代价也非二次，就在轨迹附近做二阶展开）。解这个时变 LQR，得到修正 $\delta u_t=-K_t\delta x_t$，这就是迭代 LQR（iLQR）/ 微分动态规划（Differential Dynamic Programming, DDP）的核心一步。完整算法是循环：

1. 用当前控制序列前向滚出（rollout）真实的非线性系统，得到名义轨迹；
2. 沿轨迹线性化动态、二次展开代价；
3. 用 Riccati 回传求时变增益与修正量；
4. 用修正后的控制再前向滚出，回到第 1 步，直到轨迹稳定。

需要多次前向滚出与回传，直到轨迹稳定。局部近似只在名义轨迹附近有效；当局部二次模型失效时（接触、碰撞、摩擦切换等不连续动态），该方法会发散，应换用接触感知（contact-aware）的方法或采样型模型预测控制（sampling-based MPC）。

## 部分可观测与噪声：LQG

前面假设状态 $x_t$ 能被精确观测。实际中往往只能拿到带噪声的测量 $y_t$。LQG（Linear Quadratic Gaussian）问题的设定是：线性动态、二次代价、高斯过程噪声与测量噪声，即

$$
x_{t+1}=Ax_t+Bu_t+w_t,\qquad y_t=Cx_t+v_t,\qquad w_t\sim\mathcal{N}(0,\Sigma_w),\ v_t\sim\mathcal{N}(0,\Sigma_v).
$$

此时最优值函数仍是二次的，解法遵循确定性等价（certainty equivalence）：

1. 先用卡尔曼滤波（Kalman filter）根据历史测量估计状态 $\hat x$——在线性高斯模型下，后验 $p(x_t\mid y_{1:t})$ 恰好是高斯分布，卡尔曼滤波递推地维护它的均值 $\hat x_t$ 与协方差，每步分为“预测”（用动态推进均值与协方差）和“更新”（用新测量修正）两步；
2. 再把估计值当作真实状态，套用 LQR 的控制律 $u=-K\hat x$，增益 $K$ 与无噪声时完全相同。

估计和控制可以分开设计，这称为分离原理（separation principle）。它在线性高斯的设定下成立，非线性下一般不成立，不能把“先滤后控”当成普遍定理：在非线性系统中，控制动作可能影响将来能观测到多少信息（例如为了看清楚而主动移动），估计与控制是耦合的。

> [!NOTE]+ LQR
>
> 线性动态加二次代价时，最优反馈是 $u=-Kx$，增益由 Riccati 回传得到。来源是最优控制，不是把值迭代的表格换成矩阵记号。相对一般 MDP：值函数保持二次，不必扫状态空间。相对策略梯度：必须有 $A,B$ 或可线性化的 $f$；$R$ 过小则控制猛、对模型误差敏感。迭代 LQR / DDP 把同一回传用在名义轨迹的局部线性化上，接触或不连续时局部二次失效。

<br>

## LQR 的 NumPy 实现

`discrete_lqr` 从 $P=Q_T$ 开始回传 $T$ 步，每步先算 $K$ 再更新 $P$；由于回传是从 $t=T-1$ 到 $t=0$，收集到的增益顺序是倒着的，最后需要 `reversed` 翻转，使 `Ks[t]` 对应时刻 $t$。用 `np.linalg.solve` 代替显式求逆，数值上更稳定。`rollout_lqr` 从初始状态出发，按 $u=-Kx$ 和线性动态向前模拟，返回状态与控制序列。

```python
def discrete_lqr(A, B, Q, R, QT, T):
    P = QT
    Ks = []
    for _ in range(T):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)      # K = S^{-1} B^T P A
        P = Q + A.T @ P @ A - A.T @ P @ B @ K    # Riccati 更新
        Ks.append(K)
    return list(reversed(Ks)), P   # Ks[t] 对应时刻 t；P 为 P_0

def rollout_lqr(x0, A, B, Ks):
    xs = [x0]
    us = []
    x = x0
    for K in Ks:
        u = -K @ x          # 线性反馈
        x = A @ x + B @ u   # 线性动态
        us.append(u)
        xs.append(x)
    return np.stack(xs), np.stack(us)
```

> [!EXAMPLE]+ 例23 一维小车
>
> 连续时间动态为 $\dot v=u$（控制是加速度），状态 $x=(p,v)$ 是位置与速度。以步长 $\Delta$ 离散一步：$p\leftarrow p+\Delta v+\Delta^{2}u/2$，$v\leftarrow v+\Delta u$。写成矩阵即
>
> $$
> A=\begin{pmatrix}1&\Delta\\0&1\end{pmatrix},\qquad B=\begin{pmatrix}\Delta^{2}/2\\ \Delta\end{pmatrix}.
> $$
>
> $Q$ 惩罚位置，$R$ 惩罚推力。$R$ 很大时 $K$ 小，车慢慢停；$R$ 很小时起步猛、超调。把 $Q$ 的速度项加大，等于要求少晃，闭环更阻尼。$Q,R$ 带物理单位，取值应与状态量纲一致：例如位置以米、速度以米每秒计时，同样的数值权重含义不同，换单位就要重新调权重。

<br>

> [!NOTE]+ 小结与衔接
>
> - 线性动态 + 二次代价时，值函数始终是二次型 $x^{\top}P_tx$，从 $P_T=Q_T$ 用 Riccati 方程回传，最优控制是线性反馈 $u_t=-K_tx_t$。
> - 无限时域定常系统得到常增益；可控性不足或 $R$ 过小都会带来问题。
> - 非线性系统在名义轨迹上线性化，反复滚出与回传，即 iLQR / DDP；不连续动态下会失效。
> - LQG 在线性高斯下满足分离原理：卡尔曼滤波估计 $\hat x$，再用 $u=-K\hat x$。
> - LQR / DDP 要模型或线性化，而许多环境只有采样接口。下一章对 $\pi_{\theta}$ 直接上升 $J(\theta)$，转移 $P$ 不出现在梯度里。代价是方差：估计器形式上无偏，方差过大时更新仍会被噪声淹没。

<br>

# 策略梯度（Policy Gradient）：REINFORCE 与 PPO

值迭代与 LQR 都需要模型（已知 $P$，或已知 / 可线性化的动态），值迭代还要求能对动作做 $\arg\max$。可是很多环境只提供一个“黑箱”采样接口：给一个动作，返回奖励和下一状态，此外什么都不告诉；动作空间也可能是连续的，或者像语言模型那样是巨大的词表。

策略梯度（policy gradient）换了一个角度：不去求值函数再贪心，而是**直接参数化策略** $\pi_{\theta}(a\mid s)$（例如一个输出动作分布的神经网络），把期望回报 $J(\theta)=E_{\pi_{\theta}}[G_0]$ 当成 $\theta$ 的函数，用梯度上升去最大化它。它既不要求可逆或已知的动态，也不要求 $Q$ 表格，只需要采样轨迹。核心工具是对数导数技巧（log-derivative trick），它给出梯度的无偏估计；而基线（baseline）与 PPO 的裁剪（clipping）则是用来控制方差、稳定更新的工程手段。本章输入是采样得到的轨迹，输出是 $\theta$ 的更新。大语言模型推理一章中的 RLVR（可验证奖励的强化学习）使用的就是同一个估计器，只不过动作是 token。

<br>

## 目标函数与对数导数技巧

### 轨迹分布与目标

一条轨迹（trajectory）$\tau=(s_0,a_0,\ldots)$（依次为 $s_0,a_0,s_1,a_1,\ldots$）的概率由初始分布、策略和环境转移共同决定：

$$
p_{\theta}(\tau)=p(s_0)\prod_t\pi_{\theta}(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

记 $R(\tau)$ 为整条轨迹的（折扣）总回报，目标是

$$
J(\theta)=E_{\tau\sim p_{\theta}}[R(\tau)].
$$

难点在于：$\theta$ 出现在“对谁取期望”的分布里，而不是被积函数里，不能直接把梯度挪进期望。

### 策略梯度定理

对数导数技巧利用恒等式 $\nabla_{\theta}p_{\theta}(\tau)=p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)$，把“对分布求导”变回“在分布下求期望”：

$$
\nabla_{\theta}J=\nabla_{\theta}\int p_{\theta}(\tau)R(\tau)\,d\tau=\int p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)R(\tau)\,d\tau.
$$

再对 $\log p_{\theta}(\tau)$ 展开：$\log p(s_0)+\sum_t\log\pi_{\theta}(a_t\mid s_t)+\sum_t\log P(s_{t+1}\mid s_t,a_t)$，其中只有中间一项含 $\theta$。于是

$$
\nabla_{\theta}J(\theta)
=
E_{\tau}\Big[R(\tau)\sum_{t}\nabla_{\theta}\log\pi_{\theta}(a_t\mid s_t)\Big].
$$

$P$ 不出现在梯度里，因此这是无模型（model-free）的：只要能采样轨迹、能算 $\log\pi_{\theta}$ 的梯度就行。

> [!INFO]+ 对数导数为何合法
>
> $\nabla_{\theta}p_{\theta}(\tau)=p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)$（这只是 $\nabla\log p=\nabla p/p$ 的变形）。$p_{\theta}(\tau)$ 中只有 $\pi_{\theta}$ 含 $\theta$，故 $\nabla\log p_{\theta}=\sum_t\nabla\log\pi_{\theta}$。两边乘 $R(\tau)$ 再对 $\tau$ 积分，得到 $\nabla J$。该恒等式对任何可微密度成立，与奖励是否稀疏无关，奖励本身也不需要可微；稀疏只影响方差。实现时对 $\log\pi$ 做数值裁剪，避免 $\log 0$。
>
> 轨迹密度 $p_{\theta}(\tau)=p(s_0)\prod_t\pi_{\theta}(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$。$\log$ 之后 $P$ 与 $p(s_0)$ 对 $\theta$ 的导数为零，只留下 $\sum_t\nabla\log\pi_{\theta}$。减去基线 $b(s)$ 不改变期望（因为 $E[\nabla\log\pi]=0$），只减小方差。
>
> 感性理解：$\nabla\log\pi$ 给出提高当前动作概率的方向，轨迹回报决定该方向的符号与幅度。转移核 $P$ 不含 $\theta$，对数导数后从梯度中消失，因此无需环境模型。无偏的代价是方差：同一 $\pi$ 下不同轨迹的回报可以相差一个数量级。减去只依赖状态的基线不改变期望，只减小方差。

## REINFORCE 算法

REINFORCE 就是用蒙特卡洛样本估计上式：采样若干条轨迹，用

$$
\frac{1}{N}\sum_{i=1}^{N}\sum_t\nabla_{\theta}\log\pi_{\theta}(a_t^{(i)}\mid s_t^{(i)})\,G_t^{(i)}
$$

（即 $\sum_t\nabla\log\pi\cdot G_t$ 的平均）当梯度，然后做梯度上升 $\theta\leftarrow\theta+\alpha\nabla J$。

这里把整条轨迹的 $R(\tau)$ 换成了从时刻 $t$ 算起的回报 $G_t$（reward-to-go）。理由是因果性（causality）：时刻 $t$ 的动作不可能影响 $t$ 之前已经拿到的奖励，那部分奖励乘在 $\nabla\log\pi(a_t\mid s_t)$ 上期望为零，只贡献噪声。因此用 $G_t$ 避免把未来奖励算到过去动作上，期望不变、方差更小。

## 方差与基线

REINFORCE 的方差巨大：每个动作的梯度方向都乘上一整段轨迹的回报，而回报受后面所有随机动作和随机转移的影响，噪声可能淹没信号。另一个问题是，若所有回报都是正的，被采到的动作概率都会被推高，只是幅度不同；算法要靠大量样本才能分辨“哪个更好”。

解决办法是减去一个基线 $b(s_t)$（例如学出来的值函数 $V_w(s_t)$）：

$$
\nabla J=E\Big[\sum_t\nabla\log\pi_{\theta}(a_t\mid s_t)\big(G_t-b(s_t)\big)\Big].
$$

**为什么不改变期望。** 对固定的 $s$，

$$
E_{a\sim\pi_{\theta}}\big[\nabla_{\theta}\log\pi_{\theta}(a\mid s)\big]=\sum_a\pi_{\theta}(a\mid s)\frac{\nabla_{\theta}\pi_{\theta}(a\mid s)}{\pi_{\theta}(a\mid s)}=\nabla_{\theta}\sum_a\pi_{\theta}(a\mid s)=\nabla_{\theta}1=0.
$$

所以任何只依赖状态、不依赖当前动作的 $b(s)$ 乘上去，期望都是零，减掉它不引入偏差，却能让乘子在零附近正负分开，大幅减小方差。

更进一步，乘子可以换成优势函数（advantage）$A(s,a)=Q(s,a)-V(s)$，它衡量“动作 $a$ 比该状态下的平均水平好多少”，是更紧的乘子。Actor-Critic 方法同时学习两个网络：策略（actor）$\pi_{\theta}$ 与值函数（critic）$V_w$，用 critic 估计优势来指导 actor 的更新。

## 近端策略优化（PPO）

### 动机：步子太大的问题

策略梯度每次更新后，旧数据就“过时”了，因为数据是旧策略采的。为了提高样本利用率，希望用同一批数据做多次更新；这需要重要性采样（importance sampling）比率 $r_t(\theta)$ 来修正分布差异。但朴素大步更新会让 $\pi$ 离开采样策略太远，重要性采样比爆炸，估计变得极不可靠；而且策略一旦崩坏，下一轮采到的数据也是坏的，难以恢复。TRPO 用 KL 散度约束限制每步的变化，需要二阶优化；PPO（Proximal Policy Optimization）用更简单的裁剪达到类似效果。

### 裁剪代理目标

PPO 用裁剪代理目标（clipped surrogate objective）

$$
L^{\mathrm{CLIP}}(\theta)
=
E_t\Big[\min\big(
r_t(\theta)A_t,\ 
\mathrm{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)A_t
\big)\Big],
$$

其中 $r_t(\theta)=\pi_{\theta}(a_t\mid s_t)/\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$ 是新旧策略在采样动作上的概率比，$A_t$ 是优势估计，$\varepsilon$ 常取 $0.2$。

逐项理解这个 $\min$：

- $A_t>0$（动作好于平均）时，目标希望增大 $r_t$；但 $r$ 超过 $1+\varepsilon$ 后被裁剪项封顶，梯度为零，不允许 $r$ 超过 $1+\varepsilon$ 继续获益；
- $A_t<0$（动作差于平均）时，目标希望减小 $r_t$；但 $r$ 低于 $1-\varepsilon$ 后同样不再获益，不允许 $r$ 低于 $1-\varepsilon$；
- 取 $\min$ 使得这是未裁剪目标的一个悲观下界：朝“变差”方向偏离时不会被裁剪掩盖，仍受惩罚。

实际的 PPO 损失还会再加熵奖励（entropy bonus）鼓励探索，加值函数损失训练 critic。

需要强调：PPO 不是新的最优方程，只是工程上稳定的策略梯度。它的超参数 $\varepsilon$、每批数据的更新轮数、batch 长度彼此耦合；评估时必须在验证环境上报平均回报，不能只报代理损失下降——代理损失是相对旧策略定义的，每轮都在变，其下降不代表策略变好。

> [!NOTE]+ REINFORCE 与 PPO
>
> REINFORCE 用 $\nabla\log\pi\cdot G_t$ 的样本平均当 $\nabla J$，来自对数导数恒等式，不需要 $P$。相对值迭代：动作可以是连续的或离散的 token，不必 $\arg\max_a Q$。代价是方差大。PPO 在重要性比率上加裁剪，阻止一次更新让 $\pi$ 离开采样策略太远。相对 TRPO：用一阶裁剪代替二阶约束，实现更简单，不是新的最优方程。$\varepsilon$ 与更新轮数必须在验证环境上看回报。

<br>

## 策略梯度的 NumPy 实现

下面三个函数分别对应：REINFORCE 的损失（取负号是因为优化器做最小化，最小化 $-J$ 等于最大化 $J$；优势还做了标准化，这是常用的方差缩减技巧）、从奖励序列倒序计算折扣回报 $G_t=r_t+\gamma G_{t+1}$、PPO 的裁剪目标。

```python
def reinforce_update(log_probs, returns, lr, baseline=0.0):
    """log_probs, returns: 一条轨迹上各步的 log π 与 G_t。"""
    adv = returns - baseline                       # 减基线
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 标准化优势
    loss = -np.sum(log_probs * adv)                # 负号：最小化 -J
    # 真实项目里由自动微分回传到 theta；此处只返回标量目标
    return float(loss)

def discount_returns(rewards, gamma):
    G = 0.0
    out = np.zeros_like(rewards, dtype=float)
    for t in range(len(rewards) - 1, -1, -1):  # 从后往前：G_t = r_t + gamma * G_{t+1}
        G = rewards[t] + gamma * G
        out[t] = G
    return out

def ppo_clip_objective(ratio, adv, eps=0.2):
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
    return float(np.mean(np.minimum(unclipped, clipped)))  # 取悲观的那一项
```

> [!EXAMPLE]+ 例24 基线把方差砍下来
>
> 两臂老虎机，两臂的期望奖励分别为 $1$ 与 $1.1$，每条轨迹长度 $1$（只做一次选择）。无基线时乘子是 $1$ 或 $1.1$，都是正数：不论采到哪一臂，都会提高它的概率，只是好臂提高得稍多。期望梯度的符号是对的，但单个样本的幅度抖动大，需要很多样本才能把两臂分开。减掉平均奖励 $1.05$ 后，差臂得到负优势 $-0.05$，好臂得到正优势 $+0.05$，同一样本数下分得更开。具体算一下：设 softmax 策略的两个 logit 为参数，当前两臂概率各 $0.5$，则 $\nabla\log\pi(a)=e_a-\pi$。无基线时单样本梯度为 $1\cdot(0.5,-0.5)$ 或 $1.1\cdot(-0.5,0.5)$，均值 $(-0.025,0.025)$，每个分量方差约 $0.2756$，标准差约是均值的 $21$ 倍；减去 $1.05$ 后两种样本的梯度都恰好是 $(-0.025,0.025)$，均值不变而方差为 $0$（奖励无噪声时的极端情形）。基线若选错——例如用与当前动作相关的量——会引入偏差；值函数基线只依赖 $s$，不依赖 $a$，因此保持无偏。

<br>

> [!NOTE]+ 对照：值方法与策略梯度
>
> 值迭代 / Q 学习：先得到 $V$ 或 $Q$，再贪心出 $\pi$。要求动作空间可 $\arg\max$，模型或 $Q$ 逼近要稳。策略梯度：$\pi_{\theta}$ 直接输出分布，适合大的或连续的 $A$，也适合语言模型。PPO 用裁剪防止一次更新走太远。大语言模型推理一章中的 RLVR 属于后者，不是值迭代。

<br>

> [!NOTE]+ 小结与衔接
>
> - 策略梯度直接对 $\pi_{\theta}$ 做梯度上升，对数导数技巧使转移 $P$ 从梯度中消失，因而无需模型。
> - REINFORCE 用 $\sum_t\nabla\log\pi\cdot G_t$ 的样本平均估计梯度；用 $G_t$ 而非总回报利用了因果性。
> - 只依赖状态的基线不改变期望、减小方差；优势 $A=Q-V$ 是更紧的乘子，Actor-Critic 同时学策略与值函数。
> - PPO 裁剪重要性比率，防止一次更新离旧策略太远；评估看验证回报，不看代理损失。
> - 前面各章优化的是平均损失或平均回报。部署还要问：对哪个群体错、解释能否审计、模型是否记住个人记录。下一章把分组误差、可解释性与隐私写成与平均损失并列的约束。

<br>

# 公平、可解释性与隐私（Fairness, Interpretability and Privacy）

前面所有章节优化的都是一个平均量：平均损失、平均似然或平均回报 $J$。可是一个总体准确率 $95\%$ 的贷款审批或疾病筛查模型，完全可能在某个少数群体上系统性地出错——错误率更高、校准更差、资源分配更少。平均数把这些差异藏起来了。

本章把“平均”拆成“各群体”，讨论三类部署约束：公平性（不同群体上的错误与分配是否合理）、可解释性（interpretability，能否说清模型为什么这样判）、隐私（privacy，模型是否泄露训练数据中的个人信息）。它们与 $J(\theta)$ 不是同一个目标，也不能单靠更多数据或更大模型自动消掉。下面会列出常用的公平定义、解释工具与隐私风险，并说明各公平定义彼此不能同时任意满足。实际操作上，输入是带敏感属性的评估集，输出是分组指标与限制条件。紧接着的作业一章把同一套记号接到具体作业上；再之后的“机器学习实践”一章会把数据划分、评估指标与项目节奏写成可执行的协议。

<br>

## 分组公平的定义与不可兼得

### 偏差从哪里来

训练数据记录的是历史决策，不是应然分配：如果过去的放贷、招聘、执法本身带有偏见，标签就把这种偏见原样教给了模型。即使数据本身没有问题，平均准确率高，少数群体的假阴性率仍可以很高，因为少数群体样本少，对总体损失贡献小，模型“不在乎”在他们身上犯错。

### 三种常见定义

记敏感属性（受保护的类别，如性别、种族）为 $A$，取值 $a$；$y\in\{0,1\}$ 为真实标签，$\hat y$ 为预测标签，$\hat p$ 为模型给出的预测概率。常见定义如下：

| 定义 | 要求 | 失败时的含义 |
| :--- | :--- | :--- |
| 人口统计均等 | $P(\hat y=1\mid A=a)$ 相同 | 不同组正预测率不同 |
| 均等机会 | $P(\hat y=1\mid y=1,A=a)$ 相同 | 真阳性率不等 |
| 预测校准 | $P(y=1\mid\hat p,A=a)=\hat p$ | 同一分数在不同组含义不同 |

- 人口统计均等（demographic parity）只看预测结果，要求各组获得正预测（如批准贷款）的比例相同，不管真实标签；
- 均等机会（equal opportunity）只看真正合格者（$y=1$），要求他们在各组被判为正的概率相同，即真阳性率（TPR）相等；若同时要求假阳性率（FPR）相等，则称均等几率（equalized odds）；
- 预测校准（calibration within groups）要求分数在每个组内都“说话算数”：被打 $0.7$ 分的人，无论属于哪组，真实阳性比例都应是 $0.7$。

### 为什么不能同时满足

这些定义一般不能同时成立，除非基线风险 $P(y=1\mid A)$ 在各组已经相同。更精确的已知结论（都假设两组基线正率不同）：

- Kleinberg、Mullainathan 与 Raghavan（2016）证明：组内校准，与“正类在各组的平均分数相同、负类在各组的平均分数相同”（错误率平衡的分数版本）三者不能同时成立，唯一例外是模型完美预测；
- Chouldechova（2017）证明：对二值预测，正预测值（PPV，即精确率）在各组相等，与假阳性率、假阴性率都在各组相等不能同时成立，除非完美预测；
- 人口统计均等要求两组正预测率相同，而基线正率不同时，完美预测（或任何与 $y$ 足够一致的预测）必然给出不同的正预测率，所以它与“准确地反映 $y$”本身就有张力。

需要注意，“完美预测”这一例外只适用于前两条；而且不可能性针对的是“同时满足多条”，只要求其中某一条（例如只拉齐真阳性率）通常是能做到的。直觉上：若两组基线正率不同，一个校准的分数在两组的分布必然不同，用同一阈值切下去，两组的真阳性率、假阳性率与正预测率就不会都相等；强行拉齐其中一个，必然破坏另一个。

因此选哪一个是规范问题，不是纯优化问题。必须先写清伤害是什么——拒贷（错误拒绝合格者）、误诊（漏诊病人）、多判刑（错误地把低风险者判为高风险）——再选对应的等式。例如漏诊代价高时应重点拉齐真阳性率。

### 常见的“修复”手段及其局限

- 事后阈值调整（post-processing）：对不同组用不同阈值，能满足某些等式（如均等机会），但会改变校准，并且需要在决策时使用敏感属性。
- 对抗去除 $A$ 的表示（adversarial debiasing）：训练表示使判别器读不出 $A$。这不能证明公平，只能说明（某类）探针读不出 $A$——例如线性探针读不出，不代表非线性模型读不出，更不代表决策结果公平。

> [!NOTE]+ 分组公平
>
> 把平均准确率拆成敏感属性 $A$ 上的一组等式，例如正预测率相同或真阳性率相同。来源是社会选择与算法公平文献，不是 $J(\theta)$ 的正则项。相对只报总体指标：同一模型可以在少数组上系统性地差。相对“删掉 $A$ 这一列”：代理变量仍在，反馈回路会把模型决策写进明天的标签。各定义一般不能同时成立，选择是规范问题，必须先写清伤害再选等式。

<br>

### 算法偏差的其他来源

除了历史标签，算法偏差还包括：

- 度量偏置（measurement bias）：标签本身就带歧视，例如用“被逮捕”代替“犯罪”，而逮捕率受执法强度影响；
- 反馈回路（feedback loop）：模型的决策变成明天的训练数据，例如被拒贷的人永远没有还款记录，模型永远学不到他们其实会还；
- 代理变量（proxy）：与敏感属性强相关的特征，如邮编可以代替种族。

因此只删掉 $A$ 这一列而留下强代理，不等于公平（这种做法常被称为“通过无知实现公平”，fairness through unawareness），而且删掉之后反而无法测量各组差异。

## 可解释性工具

可解释性回答“模型为什么做出这个预测”，主要用于调试（发现模型依赖了错误特征）与审计（检查是否依赖了敏感属性或其代理）。工具分为两类：

- **全局解释**：描述模型整体的行为。例如线性模型的系数（注意需要特征标准化后才可比较）、决策树的分裂规则、置换特征重要性（permutation feature importance：随机打乱某个特征的列，看性能下降多少）。
- **局部解释**：解释单次预测。LIME 在该样本附近采样并拟合一个局部线性模型；SHAP 用博弈论中的 Shapley 值把一次预测的输出公平地分配给各特征。二者分别用局部线性或博弈值解释单次预测。

常见误区：

- 注意力权重不是因果解释：某个 token 注意力高，不代表去掉它预测就会变。
- 反事实问题（counterfactual），如“若收入高 $10\%$，决策是否改变”，需要关于变量之间因果关系的结构假设（收入变了，其他特征会不会跟着变？），不能从观测数据唯一识别。
- 解释用于调试与审计，不能单独当合规证明：解释方法本身也有近似误差，甚至可以被刻意操纵。

## 隐私风险与差分隐私

模型会记忆训练点，尤其是参数多、训练轮次长的模型，对罕见样本可能几乎原样记住。由此产生的典型攻击是成员推断（membership inference）：判断某条记录是否在训练集中——模型对训练样本往往更自信、损失更低，攻击者据此区分。若训练集是“某医院的癌症患者”，仅仅“在训练集中”这件事就泄露了病情。因此训练数据含医疗记录或个人标识时，发布模型等于间接发布样本。

差分隐私（differential privacy, DP）是目前最主流的形式化防御：通过在梯度（如 DP-SGD：先裁剪每个样本的梯度范数，再加高斯噪声）或输出中加噪，使训练集“含不含某条记录”时输出的分布接近，从而任何攻击者都难以判断某个人是否参与。代价是效用下降：噪声越大隐私越强，模型越不准。

还要注意，即使只发布聚合统计，也仍可能被再识别（re-identification），例如把多个统计量相减，或与外部数据集交叉比对。

项目阶段应写明：数据许可（是否允许用于此用途）、是否可识别个人、日志保留期限，以及哪些指标只在聚合层面报告。

> [!EXAMPLE]+ 例25 两个群体的阈值
>
> （为避免与敏感属性 $A$ 混淆，这里的两个群体记为甲组与乙组。）甲组基线正率为 $0.3$，乙组为 $0.1$。用同一阈值 $0.5$ 时，即使模型在两组都校准良好，乙组的阳性样本分数整体更低，被判为正的比例更小，均等机会可能破裂。把乙组的阈值降到 $0.3$ 可拉齐真阳性率，但乙组被判为正的人里真阳性的比例更低，精确率下降，两组的分数含义也不再一致，校准曲线分裂。
>
> 一个数值示意（假设甲组分数 $\hat p\sim\mathrm{Beta}(1.5,3.5)$、乙组 $\hat p\sim\mathrm{Beta}(0.6,5.4)$，均值分别为 $0.3$ 与 $0.1$，且 $y\sim\mathrm{Bernoulli}(\hat p)$，即两组都完美校准；模拟 $10^{6}$ 人）：阈值 $0.5$ 下甲组 TPR 约 $0.33$、乙组仅约 $0.056$；乙组阈值降到 $0.3$ 后 TPR 升到约 $0.28$，与甲组接近，但乙组精确率从约 $0.57$ 降到约 $0.40$，FPR 从约 $0.005$ 升到约 $0.047$。具体数字取决于假设的分数分布，趋势是一般性的。
>
> 报告必须同时给出两组的 TPR、FPR、校准，并写清业务接受哪一种破裂。只报总体 AUC 会把该冲突藏起来。

<br>

> [!NOTE]+ 小结与衔接
>
> - 平均指标会掩盖群体差异，评估时必须按敏感属性分组报告 TPR、FPR、校准等。
> - 人口统计均等、均等机会、预测校准在基线风险不同时一般不能同时成立，选择是规范问题，先写清伤害再选等式。
> - 删除敏感属性列不能消除代理变量与反馈回路带来的偏差。
> - 解释工具（系数、置换重要性、LIME、SHAP）用于调试与审计；注意力不是因果，反事实需要结构假设。
> - 模型会记忆训练数据，成员推断可泄露隐私；差分隐私以效用为代价提供形式化保证。
> - 下一章把同一套记号接到作业：书面推导加 `src/` 里的 NumPy 实现。

<br>

<br>

# 作业（Problem Sets）题目讲解

前面各章分别讲了模型（目标函数是什么）和求解器（梯度、闭式解、牛顿法、EM、值迭代）。但只读讲义很难发现自己哪里没懂：符号写反一个转置、协方差除错一个分母、把梯度上升写成下降，往往要到真正写代码、画图、对数字的时候才暴露。CS229 的作业（problem set, PS）正是为此设计的：每道题都要求先在纸上推出公式，再在 `src/*/foo.py` 这样的 starter code 里实现，最后把图和结论排版进 PDF。

本章按 problem set 组织，逐题说明三件事：题目到底在问什么、推导的关键中间步骤、实现与数值检查时容易踩的坑。它和前面各章的关系是“落地”：前面给出目标与求解器，本章把它们接到具体的源文件与要提交的 PDF 上。

<br>

## 作业概览：题目来源、提交方式与阅读方法

### 作业的组成与公开题面

当前学期的作业由四份计分的 Problem Set 加一份不计分的 PS0 组成，每份都要同时提交书面推导和 Python / NumPy 代码。

需要注意题面的获取渠道：2026 与 2025 Autumn 学期的作业 zip 需要 Stanford 账号登录才能下载。公开可查、可用来对照数字的完整题面是 [Summer 2020](https://cs229.stanford.edu/summer2020/) 的 `ps1.pdf` / `ps2.pdf` / `ps3.pdf`。2018 至 2024 年的多数学期沿用同一个“作业族”（题目骨架相同，只微调数字、数据集或小问顺序）；Fall 2025 把强化学习单独拆成 PS4，无监督学习仍留在 PS3。

本章的定位是：概括每题问什么，给出可以对照检查的推导、数值检查方法与带注释的实现；不照抄题面长文。不同学期会有微调，具体要求以 Gradescope 上当次 zip 里的 `*.tex` 源文件为准。

| 编号 | 公开题面 | 主题 |
| :--- | :--- | :--- |
| PS0 | 课内 Ed / Canvas | NumPy、提交流程，不计分 |
| PS1 | [ps1.pdf](https://cs229.stanford.edu/summer2020/ps1.pdf) | 逻辑回归牛顿法、GDA、正样本部分标签、泊松回归、GLM 凸性、特征映射 |
| PS2 | [ps2.pdf](https://cs229.stanford.edu/summer2020/ps2.pdf) | 逻辑回归不收敛、垃圾短信、核的封闭性、核感知机、MNIST 浅层网 |
| PS3 | [ps3.pdf](https://cs229.stanford.edu/summer2020/ps3.pdf) | 倒立摆、KL、颜色压缩、$k$-means、半监督 EM、PCA、ICA |
| PS4 | Fall 2025 起单开 | 值迭代 / 策略迭代、模型估计、策略评估 |

Summer 2020 的 PS2 末尾还有一道“正则化的贝叶斯解释”（MAP 估计与 L2 / L1 正则的对应），上表没有列出，本章在 PS2 部分作为补充简要说明。

### 提交规范与记号约定

提交内容是两部分：书面答案的 PDF（推荐用 LaTeX 排版，Summer 2020 对排版提交额外加 1 分），以及源代码 zip（通常用 starter code 自带的 `make_zip.py` 生成）。代码只能使用 `environment.yml` 中列出的库，明确禁用 scikit-learn 之类的机器学习库——这正是要求“从零实现”的原因。提交后可能会被自动评分器在私有测试集上运行，所以代码必须能无错跑通。

记号上有两点与其他课程不同，做题时最容易混：

- 样本数记为 $n$（CS230 记为 $m$），特征维数记为 $d$。
- 分类题里标签 $y$ 取值 $\{0,1\}$，而不是 SVM 常用的 $\pm 1$。因此逻辑回归梯度写作 $(h-y)x$，而不是 $-y\,x\,g(-y\theta^{\top}x)$ 那种 $\pm1$ 形式。

### 做题的三步法

每道题本质上都在做三件事：

1. **建模**：写出似然或损失——数据如何生成、要最大化 / 最小化什么。
2. **推断**：求梯度、闭式解或牛顿步。
3. **对拍**：先用 $2$ 到 $4$ 个点手算一个具体数，再和代码输出对比；手算对上之后再写一般式。

下文每道题的讲解也按这个顺序：题在问什么、符号含义、先算一个数、再写一般式、最后圈出要交的答案。

> [!NOTE]+ 读法（先看这个）
>
> 每题只做三件事：**建模**（似然或损失是什么）、**推断**（梯度、闭式或牛顿步）、**对拍**（先用 $2$ 到 $4$ 个点手算）。callout 顺序：题在问什么、符号、先算一个数、再写一般式、圈出要交的答案。
>
> 下表是每份作业里最容易错、应当最先盯住的点：
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
> 笔记里的解答用于理解，不能整段贴进作业。课程荣誉守则（honor code）禁止对照往年官方解答或他人代码。生成式工具若当学期允许使用，必须按课程规范提交对话记录（transcript）。另外，公开 PDF 与当年 zip 里的数字可能不同，对拍时以当年 zip 为准。

<br>

## PS1：线性分类器、广义线性模型与特征映射

PS1 的核心是对比两种“线性分类器”。逻辑回归是判别模型（discriminative model），直接对 $p(y\mid x)$ 建模；高斯判别分析（Gaussian discriminant analysis, GDA）是生成模型（generative model），对 $p(x\mid y)p(y)$ 建模，再用贝叶斯公式反推后验。两者的决策边界都可以写成 $\theta^{\top}x+\theta_0=0$，区别在于估计 $\theta$ 的方式：逻辑回归直接最大化条件似然，GDA 先估计均值与协方差再换算。因此当数据真的接近“共享协方差的高斯”时 GDA 更省样本，偏离高斯时逻辑回归更稳健——这正是 PS1.1 最后要求在两个数据集上观察的现象。

PS1 后半部分把视角推广到广义线性模型（GLM）：泊松回归是 GLM 的一个新实例，GLM 凸性一题说明“换一个指数族分布，凸性自动保留”；特征映射一题则说明“线性回归”的“线性”指的是对参数线性，而非对输入线性。

### PS1.1 逻辑回归的 Hessian 半正定

**动机**：牛顿法和梯度下降能找到全局最优，前提是目标函数凸。这一小问用 Hessian 半正定（positive semi-definite, PSD）证明逻辑回归的平均经验损失是凸的。

> [!INFO]+ PS1.1 逻辑回归 Hessian 半正定
>
> **题在问什么**。平均经验损失
>
> $$
> J(\theta)=-\frac1n\sum_{i=1}^{n}\Big[y^{(i)}\log h_{\theta}(x^{(i)})+(1-y^{(i)})\log(1-h_{\theta}(x^{(i)}))\Big],
> $$
>
> 其中 $x^{(i)}\in\mathbb{R}^d$，$y^{(i)}\in\{0,1\}$，$h_{\theta}(x)=g(\theta^{\top}x)$，$g(z)=(1+e^{-z})^{-1}$ 是 sigmoid 函数。要证明 Hessian $H$ 对一切 $z$ 满足 $z^{\top}Hz\ge 0$，从而 $J$ 凸。
>
> **先对一个样本求导**。令 $h=g(\theta^{\top}x)$。利用 $g'=g(1-g)$，有 $\partial \log h/\partial\theta=(1-h)x$，$\partial\log(1-h)/\partial\theta=-hx$。代回单样本损失 $-[y\log h+(1-y)\log(1-h)]$：
>
> $$
> \frac{\partial}{\partial\theta}\Big(-y\log h-(1-y)\log(1-h)\Big)=-y(1-h)x+(1-y)hx=(h-y)x.
> $$
>
> 即 $\partial J/\partial\theta=(h-y)x$（单个样本、不除 $n$）。再对 $\theta$ 求导一次，$y$ 是常数，只有 $h$ 依赖 $\theta$：$\partial h/\partial\theta^{\top}=h(1-h)x^{\top}$，于是得到 $h(1-h)\,xx^{\top}$，因为 $g'=g(1-g)$。
>
> **一般式**。$H=\frac1n\sum_i h^{(i)}(1-h^{(i)})x^{(i)}(x^{(i)})^{\top}$。对任意向量 $z$，利用题面提示 $\sum_i\sum_j z_ix_ix_jz_j=(x^{\top}z)^2\ge 0$，
>
> $$
> z^{\top}Hz=\frac1n\sum_i h^{(i)}(1-h^{(i)})(z^{\top}x^{(i)})^{2}\ge 0,
> $$
>
> 因为 $h(1-h)\in(0,1/4]$。故 $H\succeq 0$，$J$ 没有异于全局最小的局部最小。
>
> **要交的答案**：$H$ 的表达式，加上 $z^{\top}Hz\ge 0$。

这个结论的实际意义是：从任何初始点出发，只要优化器正常工作，就不会卡在“坏的”局部最小。需要注意 $H$ 只是半正定而不一定正定——当数据线性可分时，最优解跑到无穷远（PS2.1 会专门讨论），此时 $h(1-h)\to 0$，Hessian 会趋于奇异。

### PS1.1 用牛顿法训练逻辑回归

**动机**：梯度下降只用一阶信息，需要调学习率；牛顿法（Newton's method）用 Hessian 描述曲率，步长由曲率自动决定，对逻辑回归这种光滑凸问题通常只需要个位数到十来次迭代就收敛（二次收敛）。

> [!EXAMPLE]+ PS1.1 牛顿法实现
>
> **题在问什么**。从 $\theta=0$ 走牛顿步，直到 $\|\theta_{\mathrm{new}}-\theta_{\mathrm{old}}\|_1<10^{-5}$（第一次满足 $L_1$ 范数小于 $\epsilon=10^{-5}$ 的迭代即停）。把验证集概率写成 starter code 指定的文件，并在 $x_1$-$x_2$ 平面上用不同符号画出两类点和 $p=0.5$ 的直线。
>
> 牛顿步：$\theta\leftarrow\theta-H^{-1}\nabla J$。这里没有学习率，“步长”完全由 $H^{-1}$ 决定。$X$ 为 $(n,d)$，含偏置列（全 1 列）。
>
> 决策边界：$p(y=1\mid x)=0.5$ 等价于 $\theta^{\top}x=0$，即 $\theta_0+\theta_1 x_1+\theta_2 x_2=0$，解出 $x_2=-(\theta_0+\theta_1 x_1)/\theta_2$，画图时在 $x_1$ 的取值范围内取两点连线即可。
>
> **要交的答案**：验证集概率文件、图、以及“从 $\theta=0$ 迭代到 $L_1$ 步长小于 $10^{-5}$”。

下面的实现有两处数值上的细节：sigmoid 对正负输入分开计算以避免 `exp` 溢出；牛顿方向用 `np.linalg.solve` 求解线性方程组而不是显式求逆，并加一个极小的对角项 `1e-12` 防止 $H$ 奇异时报错。

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

一个便宜的检查：`(X.T * w) @ X` 利用广播把每个样本的第 $i$ 列乘以权重 $w_i$，等价于 $X^{\top}\mathrm{diag}(w)X$，但不需要构造 $n\times n$ 的对角矩阵。梯度和 Hessian 是否都除以了 $n$ 并不影响牛顿步（两者的 $1/n$ 相消），但必须一致——只除一个会让步长差 $n$ 倍。

### PS1.1 GDA 的后验是逻辑函数形式

**动机**：GDA 是生成模型，它的分类器长什么样并不显然。这一问证明：只要两类共享协方差矩阵，GDA 推出的后验 $p(y=1\mid x)$ 恰好是逻辑回归的形式，所以它也是线性分类器。

GDA 的模型是：$y\sim\mathrm{Bernoulli}(\phi)$；$x\mid y=c\sim\mathcal{N}(\mu_c,\Sigma)$，$c\in\{0,1\}$。其中 $\phi\in(0,1)$ 是正类先验，$\mu_0,\mu_1\in\mathbb{R}^d$ 是两类均值，$\Sigma\in\mathbb{R}^{d\times d}$ 是共享的对称正定协方差。

> [!INFO]+ PS1.1 GDA 后验是逻辑形式
>
> **题在问什么**。共享协方差的两类高斯，证明
>
> $$
> p(y=1\mid x)=\frac{1}{1+\exp(-(\theta^{\top}x+\theta_0))}.
> $$
>
> **先写对数后验比**。由贝叶斯公式，$p(y=1\mid x)/p(y=0\mid x)=p(x\mid y=1)\phi\,/\,[p(x\mid y=0)(1-\phi)]$；两个高斯密度的归一化常数 $(2\pi)^{d/2}|\Sigma|^{1/2}$ 相同，直接约掉，于是
>
> $$
> \log\frac{p(y=1\mid x)}{p(y=0\mid x)}
> =\log\frac{\phi}{1-\phi}
> -\frac12(x-\mu_1)^{\top}\Sigma^{-1}(x-\mu_1)
> +\frac12(x-\mu_0)^{\top}\Sigma^{-1}(x-\mu_0).
> $$
>
> 展开二次型：$(x-\mu_c)^{\top}\Sigma^{-1}(x-\mu_c)=x^{\top}\Sigma^{-1}x-2\mu_c^{\top}\Sigma^{-1}x+\mu_c^{\top}\Sigma^{-1}\mu_c$（用到 $\Sigma^{-1}$ 对称）。二次项 $x^{\top}\Sigma^{-1}x$ 在两项中系数相反、相消，剩下 $x$ 的一次式 $(\mu_1-\mu_0)^{\top}\Sigma^{-1}x+\text{常数}$。令
>
> $$
> \theta=\Sigma^{-1}(\mu_1-\mu_0),\qquad
> \theta_0=\log\frac{\phi}{1-\phi}
> -\frac12\mu_1^{\top}\Sigma^{-1}\mu_1
> +\frac12\mu_0^{\top}\Sigma^{-1}\mu_0.
> $$
>
> 对数后验比等于 $\theta^{\top}x+\theta_0$，再利用 $p(y=1\mid x)+p(y=0\mid x)=1$ 解出 $p(y=1\mid x)$，即得逻辑形式。**共享 $\Sigma$ 是线性边界的原因**；两类各用一个 $\Sigma$ 会留下二次项，变成二次判别（quadratic discriminant analysis, QDA）。

需要注意这个结论只有单向：GDA 的假设能推出逻辑形式的后验，但逻辑形式的后验不能反推出 $x$ 服从高斯。所以 GDA 的假设更强——假设对时更省数据，假设错时会被拖累。

### PS1.1 GDA 的极大似然估计

> [!INFO]+ PS1.1 GDA 的 MLE 与数据集对照
>
> **MLE（要交的闭式）**。对数似然为 $\ell=\sum_i\log p(x^{(i)}\mid y^{(i)};\mu_0,\mu_1,\Sigma)+\sum_i\log p(y^{(i)};\phi)$。四个估计分别是：$\phi$ 是正类比例 $\frac1n\sum_i\mathbf{1}\{y^{(i)}=1\}$；$\mu_0,\mu_1$ 是各类样本均值 $\mu_c=\sum_i\mathbf{1}\{y^{(i)}=c\}x^{(i)}/\sum_i\mathbf{1}\{y^{(i)}=c\}$；协方差为
>
> $$
> \Sigma=\frac1n\sum_{i=1}^{n}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^{\top}.
> $$
>
> 分母是 $n$ 不是 $n-1$（MLE 是有偏估计，不是统计课里的无偏样本协方差）。
>
> **证明思路**。
>
> - 对 $\phi$：$\ell$ 中与 $\phi$ 有关的部分是 $n_1\log\phi+n_0\log(1-\phi)$（$n_c$ 为第 $c$ 类样本数），求导置零得 $\phi=n_1/n$。
> - 对 $\mu_c$：对数似然对 $\mu_c$ 求导得 $\sum_{i:y^{(i)}=c}\Sigma^{-1}(x^{(i)}-\mu_c)$，令其为零，即该类均值。
> - 对 $\Sigma$：用矩阵微积分。记 $\Lambda=\Sigma^{-1}$，相关项是 $\frac n2\log|\Lambda|-\frac12\sum_i(x^{(i)}-\mu_{y^{(i)}})^{\top}\Lambda(x^{(i)}-\mu_{y^{(i)}})$；利用 $\nabla_\Lambda\log|\Lambda|=\Lambda^{-1}$ 与 $\nabla_\Lambda\,a^{\top}\Lambda a=aa^{\top}$，置零得 $\Lambda^{-1}=\frac1n\sum_i(\cdot)(\cdot)^{\top}$，即上述平均外积。题面允许假设正负样本至少各有一个，保证 $\mu_c$ 的分母非零。
>
> **数据集对照**。ds1 两类大致椭圆、协方差接近，GDA 与逻辑回归边界接近。ds2 往往一类被拉成长条或边缘非高斯，GDA 更差。常见补救：对偏斜坐标取 $\log(1+|x|)$，或先标准化再拟合——目的都是让变换后的特征更接近高斯，从而让 GDA 的假设重新成立。
>
> **要交的答案**：$\theta,\theta_0$ 的式子、四个 MLE、两套图、以及“哪一套 GDA 更差、为何、用什么变换”。

实现时，协方差的计算要先把每个样本减去**它所属类别**的均值（不是全体均值），再求平均外积：

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

得到 $(\theta,\theta_0)$ 后，GDA 的预测与画边界的方式和逻辑回归完全相同：$p(y=1\mid x)=g(\theta^{\top}x+\theta_0)$。一个对拍办法是在 ds1 上比较两条边界，它们应当接近但不重合。

### PS1.2 只有部分正标签的分类

**动机**：现实中常有“只知道一部分正例”的数据。题面的例子是蛋白质数据库：库里的蛋白质都参与跨膜信号传递（正例），但还有很多参与信号传递的蛋白质没被收录。没被收录不代表是负例。如果直接把“未收录”当负例训练，模型学到的是“是否被收录”的概率，而不是“是否真的是正例”的概率。这道题说明：在一个简单的标注机制假设下，两者只差一个常数因子，而这个常数可以从验证集估计出来。

> [!INFO]+ PS1.2 只有部分正标签
>
> **题在问什么**。真实标签 $t$ 看不到。观测 $y$ 的生成方式：若 $t=1$，以概率 $\alpha\in(0,1)$ 标成 $1$，否则标 $0$；若 $t=0$，永远标 $0$。且 $\alpha$ 与 $x$ 无关（对一切 $x$ 成立）。要只用 $(x,y)$ 去逼近 $p(t=1\mid x)$。
>
> **先用贝叶斯**。
>
> $$
> p(t=1\mid y=1,x)=\frac{p(y=1\mid t=1,x)\,p(t=1\mid x)}{p(y=1\mid t=1,x)\,p(t=1\mid x)+p(y=1\mid t=0,x)\,p(t=0\mid x)}.
> $$
>
> 由于 $p(y=1\mid t=0,x)=0$，分母第二项为零，分子分母相同，故 $y=1$ 只能来自 $t=1$：
>
> $$
> p(t=1\mid y=1,x)=1.
> $$
>
> 再用全概率公式拆 $p(y=1\mid x)=p(y=1\mid t=1,x)p(t=1\mid x)+p(y=1\mid t=0,x)p(t=0\mid x)$，第二项为零，即 $p(y=1\mid x)=p(y=1\mid t=1,x)p(t=1\mid x)=\alpha\,p(t=1\mid x)$。故
>
> $$
> p(t=1\mid x)=\frac1\alpha\,p(y=1\mid x).
> $$
>
> **估计 $\alpha$**。设分类器 $h$ 恰好等于 $p(y=1\mid x)$，再假设真实过程无噪声，即 $p(t=1\mid x)\in\{0,1\}$（注意只假设 $t$ 无噪声，观测 $y$ 仍然是有噪声的）。当 $y=1$ 时必有 $t=1$，于是在这个 $x$ 处 $p(t=1\mid x)$ 不可能是 $0$，只能是 $1$，代入上式得 $h(x)=p(y=1\mid x)=\alpha$。因此
>
> $$
> \alpha=\mathbb{E}[h(x)\mid y=1]\approx\frac{1}{|V_+|}\sum_{x\in V_+}h(x),
> $$
>
> $V_+$ 是验证集里 $y=1$ 的点。用训练集估 $\alpha$ 会偏乐观（模型在自己见过的正例上给出的概率偏高），题面指定验证集。
>
> **实现顺序**。先用 $t$ 训一条上界（理想情形，实际拿不到 $t$）；再用 $y$ 训朴素模型（它学到的是 $p(y=1\mid x)$，会系统性低估）；最后把朴素概率除以 $\hat\alpha$。只做排序时 $\alpha$ 可省略，因为 $1/\alpha$ 不改变次序。
>
> **要交的答案**：上面三式、三张测试图、验证集上的 $\hat\alpha$。

校正后的决策边界可以直接写出来：$h(x)/\hat\alpha=0.5$ 等价于 $h(x)=\hat\alpha/2$，即 $\theta^{\top}x=\log\frac{\hat\alpha/2}{1-\hat\alpha/2}$。所以校正不改变边界的方向，只把朴素模型的边界平移到更“宽松”的位置（因为 $\hat\alpha/2<1/2$，更多点被判为正）。三张测试图里，校正后的边界应当明显比朴素边界更接近用 $t$ 训练的上界。

### PS1.3 泊松回归

**动机**：要预测的量是计数（一天的网站访问量、一小时进店顾客数），取值 $0,1,2,\ldots$。高斯回归会给出负数或非整数均值，伯努利只适合二值。泊松分布支撑在非负整数上且属于指数族，因此可以套用 GLM 的标准流程：写成指数族 → 读出典范响应函数 → 求对数似然梯度。

> [!INFO]+ PS1.3 泊松回归
>
> **题在问什么**。计数 $y=0,1,2,\ldots$，$p(y;\lambda)=e^{-\lambda}\lambda^y/y!$，$\lambda>0$ 既是均值也是方差。写成指数族 $p(y;\eta)=b(y)\exp(\eta^{\top}T(y)-a(\eta))$，再导出典范响应与梯度上升。
>
> 利用 $\lambda^y=\exp(y\log\lambda)$：
>
> $$
> p(y;\lambda)=\frac{1}{y!}\exp\big(y\log\lambda-\lambda\big).
> $$
>
> 故 $b(y)=1/y!$，$\eta=\log\lambda$，$T(y)=y$，$a(\eta)=e^{\eta}$（把 $\lambda$ 用 $\eta$ 表示）。均值 $\lambda=e^{\eta}$，这就是典范响应函数（canonical response function）：在 GLM 中令 $\eta=\theta^{\top}x$，得到 $h_{\theta}(x)=\exp(\theta^{\top}x)$。指数保证了预测均值恒为正。
>
> 单样本对数似然（略去与 $\theta$ 无关的 $y!$）：$\ell=y\,\theta^{\top}x-e^{\theta^{\top}x}$。对 $\theta_j$ 求导：$(y-e^{\theta^{\top}x})x_j$。随机梯度**上升**（最大化似然）：
>
> $$
> \theta\leftarrow\theta+\alpha\big(y-\exp(\theta^{\top}x)\big)x.
> $$
>
> 这里的 $\alpha$ 是学习率（与 PS1.2 的 $\alpha$ 无关）。这个形式和线性回归、逻辑回归的更新 $\theta\leftarrow\theta+\alpha(y-h_\theta(x))x$ 完全一致，是 GLM 使用典范响应时的共同结构。
>
> 全批把括号换成 $\frac1n\sum_i(y^{(i)}-\exp(\theta^{\top}x^{(i)}))x^{(i)}$。停止条件：参数改变的欧氏范数小于 $10^{-5}$。验证图：横轴真计数、纵轴预测均值（真计数是整数，预测均值一般是实数）。学习率过大 $\exp$ 会爆。
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

调试提示：如果 `lam` 出现 `inf`，说明某一步 $\theta^{\top}x$ 过大，应当减小学习率；如果损失长期几乎不变，可能是学习率过小或特征尺度差异太大。散点图上点应大致沿对角线分布。

### PS1.4 广义线性模型的负对数似然是凸的

**动机**：PS1.1 用手算证明了逻辑回归的凸性，PS1.3 又引入了泊松回归。是否每换一种分布都要重新证一次？这一题给出统一答案：对任何使用指数族、自然参数 $\eta=\theta^{\top}x$ 的 GLM，负对数似然（negative log-likelihood, NLL）都对 $\theta$ 凸。证明顺带得到指数族一个漂亮的性质：均值和方差可以通过对 $a(\eta)$ 求导得到，而不必做积分。

> [!INFO]+ PS1.4 GLM 的 NLL 凸
>
> **题在问什么**。标量自然参数 $\eta=\theta^{\top}x$，$T(y)=y$，$p(y;\eta)=b(y)\exp(\eta y-a(\eta))$。证明负对数似然对 $\theta$ 凸。$a(\eta)$ 称为对数配分函数（log-partition function）。
>
> **均值**。密度对 $y$ 积分恒为 $1$：$\int b(y)\exp(\eta y-a(\eta))\,dy=1$。两边对 $\eta$ 求导（交换求导与积分）：$\int (y-a'(\eta))\,p(y;\eta)\,dy=0$，即 $\mathbb{E}[y]=a'(\eta)$。
>
> **方差**。对 $\int(y-a'(\eta))p(y;\eta)dy=0$ 再对 $\eta$ 求导一次：$\int\big[-a''(\eta)+(y-a'(\eta))^2\big]p(y;\eta)\,dy=0$，即 $\mathrm{Var}(y)=a''(\eta)\ge 0$。
>
> **Hessian**。单样本 NLL：$\ell(\theta)=-y\,\theta^{\top}x+a(\theta^{\top}x)-\log b(y)$。由链式法则，梯度是 $(a'(\theta^{\top}x)-y)x$，再求一次导得 Hessian：
>
> $$
> \nabla^{2}\ell=a''(\theta^{\top}x)\,xx^{\top}=\mathrm{Var}(y\mid x)\,xx^{\top}\succeq 0.
> $$
>
> 方差非负，$xx^{\top}$ 是 PSD 的秩一矩阵，故对一切 $\theta$ 半正定；多个样本求和仍 PSD。这就是“换指数族只换 $a$，凸性自动继承”。
>
> **要交的答案**：均值 $=$ $a'$、方差 $=$ $a''$、Hessian $=$ $\mathrm{Var}\,xx^{\top}$。

可以用前面两题对拍：伯努利的 $a(\eta)=\log(1+e^{\eta})$，$a''=h(1-h)$，正好是 PS1.1 Hessian 里的权重；泊松的 $a(\eta)=e^{\eta}$，$a''=e^{\eta}=\lambda$，正好是泊松分布的方差。

### PS1.5 特征映射：线性回归“线性”在哪里

**动机**：数据明显是曲线时，直线拟合不好。但“线性回归”的“线性”是指假设对**参数** $\theta$ 线性，而对输入 $x$ 可以任意非线性——只要先把 $x$ 映射成一组特征 $\phi(x)$。题面区分两个术语：原始的 $x$ 称为输入属性（input attributes），$\phi(x)$ 称为特征（features），$\phi$ 称为特征映射（feature map）。这也是后面核方法的出发点。

> [!EXAMPLE]+ PS1.5 特征映射
>
> **题在问什么**。$h_{\theta}(x)=\theta_3 x^3+\theta_2 x^2+\theta_1 x+\theta_0$ 对 $x$ 非线性，对 $\theta$ 仍线性。令 $\phi(x)=(1,x,x^2,x^3)\in\mathbb{R}^4$，则 $h_\theta(x)=\theta^{\top}\phi(x)$，在新特征上做普通线性回归即可。
>
> 目标 $J(\theta)=\frac1{2n}\sum_i(\theta^{\top}\phi(x^{(i)})-y^{(i)})^{2}$。记 $\Phi\in\mathbb{R}^{n\times(k+1)}$ 为第 $i$ 行是 $\phi(x^{(i)})^{\top}$ 的设计矩阵，全批梯度：$\nabla J=\frac1n\Phi^{\top}(\Phi\theta-y)$，批量梯度下降更新为 $\theta\leftarrow\theta-\alpha\nabla J$。闭式（正规方程，normal equations）用 `np.linalg.solve(Phi.T @ Phi, Phi.T @ y)`，不要写 `inv`——$\Phi^{\top}\Phi$ 在高次多项式下条件数极大甚至不可逆，显式求逆数值很不稳定。
>
> 推广到 $\phi(x)=(1,x,\ldots,x^k)\in\mathbb{R}^{k+1}$ 后，$k$ 升高时训练曲线更贴合，验证会先好后坏。数据由 $\sin x$ 加高斯噪声生成，把 $\sin x$ 放进 $\phi$（即 $\phi(x)=(1,x,\ldots,x^k,\sin x)\in\mathbb{R}^{k+2}$，题面扫 $k\in\{0,1,2,3,5,10,20\}$）后，$k=0$ 已经能拟合主体，因为“对的特征”比“更高的次数”有效得多。小样本（`small.csv`，题面扫 $k\in\{1,2,5,10,20\}$）上 $k>n$ 时（更准确地说，特征数 $k+1$ 超过样本数 $n$ 时）曲线会“发疯”，这是过拟合：假设类比数据更灵活，能以不自然的方式穿过每个数据点，开始解释噪声，于是在新样本上预测很差。
>
> **要交的答案**：$J$ 与梯度、四张图（$k=3$；$k\in\{3,5,10,20\}$；加 $\sin$；小样本）、以及过拟合一两句。

这道题提前预告了后面学习理论（偏差–方差权衡）和正则化要解决的问题：模型容量相对数据量过大时，训练误差低不代表泛化好。

<br>

## PS2：训练稳定性、朴素贝叶斯、核方法与浅层神经网络

PS2 的几道题看似互不相关，实际上围绕同一个问题：模型“学到了什么、为什么能学 / 学不到”。PS2.1 是调试题，训练不收敛的原因不在代码而在数据的几何性质；PS2.2 用朴素贝叶斯和 SVM 做真实的文本分类；PS2.3 与 PS2.4 讨论什么样的函数可以当作核、以及如何把一个只会算内积的算法“核化”；PS2.5 从零实现一个单隐层神经网络。Summer 2020 版本末尾还有一道把正则化解释为贝叶斯先验的题。

### PS2.1 逻辑回归在线性可分数据上不收敛

**动机**：机器学习的调试和普通软件调试不同——代码没有 bug，算法也可能“不正常”。这一题给定一份不许修改的逻辑回归实现（`src/stability/stability.py`）和两个数据集 A、B，要求解释为什么它在 B 上停不下来，并给出“硬证据”（数学、代码或图）。

> [!INFO]+ PS2.1 逻辑回归在线性可分数据上不收敛
>
> **题在问什么**。同一份实现，数据集 A 正常停，数据集 B 损失一直降、$\|\theta\|$ 一直涨。
>
> **原因**。B 线性可分。最大似然要把正确类的 $h$ 推到 $0$ 或 $1$，需要 $|\theta^{\top}x|\rightarrow\infty$，所以 $\|\theta\|\rightarrow\infty$。更具体地说：若某个 $\theta^\ast$ 把所有点分对，则沿方向 $c\,\theta^\ast$ 放大 $c$，每个样本的对数似然都单调增加，似然的上确界 $0$ 只在 $c\to\infty$ 时逼近，有限处不存在最大值。A 有重叠，总有点会被分错，把 $\|\theta\|$ 放大会让这些点的损失变大，因此最优 $\theta$ 有限。这不是溢出（题面也提示问题与数值舍入或上下溢无关）。
>
> | 改动 | 能否在 B 上收敛 | 理由 |
> | :--- | :--- | :--- |
> | 换一个固定学习率 | 否 | 只改步长，不改“最优在无穷远” |
> | 学习率按 $1/t^{2}$ 衰减 | 会停下，但不是真正收敛 | $\sum_t 1/t^{2}$ 有限，参数总位移有界，更新量最终小于阈值而停止；但停在的点由初始学习率和衰减序列决定，并不是似然的最优点（它不存在）。梯度方向仍指向“把 $\|\theta\|$ 放大”，只是步长耗尽，$\theta$ 停在某个有限处 |
> | 线性缩放输入 | 否 | 可分性不变 |
> | 加 $\lambda\|\theta\|_2^{2}$ | 是 | 岭正则把最优拉回有限点：$\|\theta\|$ 增大时惩罚项二次增长，而对数似然的收益有上界 |
> | 给 $x$ 或 $y$ 加噪声 | 通常是 | 破坏完全可分 |
>
> SVM 有间隔约束或 $\|\theta\|$ 惩罚，最优在有限点，对这类 B 不爆炸：硬间隔 SVM 固定函数间隔为 $1$ 后最小化 $\|w\|$，软间隔 SVM 的目标里直接含 $\|w\|^2$。
>
> **要交的答案**：B 线性可分的证据（例如画出分隔直线、或训练准确率 $=1$ 且 $\|\theta\|$ 单调增）、上表、SVM 一句。

关于 $1/t^2$ 衰减一行，一种常见答法是直接写“否”（理由是“参数仍被往无穷推，只是更慢”）。更准确的说法是：$\sum_t 1/t^2=\pi^2/6$ 有限，参数的总位移有界，所以算法一定会因为更新量变小而停下（若按 `stability.py` 的“参数变化小于阈值”判据来判断，答案就是“会收敛”），但这只是学习率耗尽，得到的 $\theta$ 不是似然的最优点、没有统计意义。作答时把这两层意思都写清楚最稳妥。

### PS2.2 垃圾短信分类：朴素贝叶斯与 SVM

**动机**：这是一个真实的文本分类任务，数据来自公开的 SMS Spam Collection。要从零实现两步：把文本变成词频矩阵，再用多项式事件模型（multinomial event model）的朴素贝叶斯分类；最后用题目提供的 RBF 核 SVM 做对比，并在验证集上选核半径。

> [!EXAMPLE]+ PS2.2 垃圾短信朴素贝叶斯
>
> **题在问什么**。词袋加拉普拉斯平滑，预测垃圾 / 非垃圾。要报告词表大小、测试准确率、最能指示垃圾的五个词、以及验证集上最好的 RBF 半径。
>
> **预处理**。词表：小写、按空白切、出现次数低于阈值的丢掉（starter code 注释给出具体阈值，常见版本是“至少出现在 5 条短信中”，以当次代码为准）。每条短信表示为长度 $|V|$ 的词频向量 $c$，$c_w$ 是词 $w$ 在该短信中的出现次数。
>
> **模型**。multinomial 模型下，每个词位置独立地从类条件分布 $p(w\mid y)$ 抽取。拉普拉斯平滑后
>
> $$
> p(w\mid y)=\frac{c_{y,w}+1}{C_y+|V|},
> $$
>
> 其中 $c_{y,w}$ 是类 $y$ 的所有训练短信中词 $w$ 的总次数，$C_y$ 是类 $y$ 的总词数。加 $1$ 是为了避免测试时遇到某类没见过的词，概率为 $0$ 导致整个乘积为 $0$。
>
> **预测必须走对数**，否则几十上百个小于 $1$ 的数连乘会下溢成 $0$（underflow）：
>
> $$
> \log p(y\mid x)=\log p(y)+\sum_w c_w\log p(w\mid y).
> $$
>
> 上式右边省略了与 $y$ 无关的归一化常数 $-\log p(x)$，比较两类大小时不影响结果，取较大者即可。
>
> **指示词**。指示度 $\log\frac{p(w\mid\mathrm{spam})}{p(w\mid\mathrm{ham})}$ 最大的五个词。SVM 半径在验证集上扫，不要看测试——测试集只用来报告最终结果。
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

对照公式检查：`counts.sum()` 等于 $C_y+|V|$（每个词都加了 $1$），所以 `counts / counts.sum()` 正是平滑后的 $p(w\mid y)$。五个指示词可以直接由 `np.argsort(log_cond[1] - log_cond[0])[-5:]` 取得。

### PS2.3 哪些运算保持“核”的合法性

**动机**：使用核方法时，常常是先凭直觉写出一个“相似度函数” $K(x,z)$，再问它能不能放进 SVM。合法的核必须对应某个特征映射下的内积。Mercer 定理把这个条件变成可以检验的形式，这一题练习用它判断常见的构造方式。

Mercer 定理（Mercer's theorem）：$K$ 是核当且仅当对任意有限点集 $\{x^{(1)},\ldots,x^{(n)}\}$，Gram 矩阵 $K_{ij}=K(x^{(i)},x^{(j)})$ 对称且半正定。设 $K_1,K_2$ 是 $\mathbb{R}^d\times\mathbb{R}^d$ 上的核，$a>0$，$f:\mathbb{R}^d\to\mathbb{R}$ 是实值函数，$\phi:\mathbb{R}^d\to\mathbb{R}^p$，$K_3$ 是 $\mathbb{R}^p\times\mathbb{R}^p$ 上的核，$p$ 是正系数多项式。

> [!INFO]+ PS2.3 核的封闭性
>
> 下面记 $K_1,K_2$ 在同一点集上的 Gram 矩阵也为 $K_1,K_2$，$v$ 为任意向量。
>
> | $K$ | 是核？ | 理由 |
> | :--- | :--- | :--- |
> | $K_1+K_2$ | 是 | PSD 对加法封闭：$v^{\top}(K_1+K_2)v=v^{\top}K_1v+v^{\top}K_2v\ge0$ |
> | $K_1-K_2$ | 否 | $K_2=2K_1$ 时变成 $-K_1$ |
> | $aK_1$ | 是 | 正数倍保持 PSD |
> | $-aK_1$ | 否 | 负特征值：只要 $K_1$ 的 Gram 矩阵非零，就有 $v^{\top}(-aK_1)v<0$ |
> | $K_1K_2$（逐点积） | 是 | Schur 积定理：PSD 的 Hadamard 积仍 PSD |
> | $f(x)f(z)$ | 是 | $\phi(x)=f(x)$ 的内积；$v^{\top}Kv=(\sum_i v_if(x^{(i)}))^2\ge0$ |
> | $K_3(\phi(x),\phi(z))$ | 是 | 先映射再套核：Gram 矩阵就是 $K_3$ 在点集 $\{\phi(x^{(i)})\}$ 上的 Gram 矩阵 |
> | $p(K_1)$ | 是 | 正系数加与积的复合 |
>
> **要交的答案**：八行是 / 否，否的给反例，是的给 PSD 或显式 $\phi$。

逐点积一行是题面标注的难点。除了引用 Schur 积定理，也可以给出显式特征映射：若 $K_1(x,z)=\phi_1(x)^{\top}\phi_1(z)$，$K_2(x,z)=\phi_2(x)^{\top}\phi_2(z)$，则

$$
K_1(x,z)K_2(x,z)=\sum_{i,j}\phi_{1,i}(x)\phi_{2,j}(x)\,\phi_{1,i}(z)\phi_{2,j}(z),
$$

即特征映射 $\phi(x)=\big(\phi_{1,i}(x)\phi_{2,j}(x)\big)_{i,j}$（张量积）下的内积。多项式一行则由前面几行组合得到：$K_1^k$ 由逐点积封闭性归纳得出，乘正系数保持核，相加保持核；常数项 $a_0>0$ 可以写成 $f(x)f(z)$ 且 $f\equiv\sqrt{a_0}$。

### PS2.4 核感知机

**动机**：感知机（perceptron）是最简单的在线线性分类器，每次只看一个样本、错了就修正。把它核化，就能在无穷维特征空间里学习非线性边界，而从不显式计算 $\phi(x)$。关键观察是：参数始终是见过样本特征的线性组合，因此可以只存组合系数。

感知机的设定：$y\in\{0,1\}$，$h(x)=g(\theta^{\top}x)$，其中 $g(z)=1$ 若 $z\ge0$，否则为 $0$；对训练集只遍历一遍，每次用一个样本更新。

> [!EXAMPLE]+ PS2.4 核感知机
>
> 一次遍历更新 $\theta\leftarrow\theta+\alpha(y-h(x))\phi(x)$，但 $\phi$ 无穷维，不能存 $\theta$。$\theta^{(0)}=0$，每次更新只加上某个 $\phi(x^{(t)})$ 的倍数，归纳可知
>
> $$
> \theta^{(i)}=\sum_{t=1}^{i}\beta_t\phi(x^{(t)}),\qquad
> \theta^{\top}\phi(x)=\sum_{t}\beta_t K(x^{(t)},x).
> $$
>
> 这回答了题面的三个小问：
>
> 1. **表示**：用见过的样本列表和系数列表 $\{\beta_t\}$ 隐式表示 $\theta^{(i)}$；$\theta^{(0)}=0$ 对应空列表。
> 2. **预测**：$h(x)=\mathbf{1}\{\sum_t\beta_t K(x^{(t)},x)\ge 0\}$，只需要核函数值。
> 3. **更新**：看到新样本 $(x^{(i+1)},y^{(i+1)})$ 时，追加 $\beta_{i+1}=\alpha\big(y^{(i+1)}-h^{(i)}(x^{(i+1)})\big)$。更新只改对应的 $\beta$。由于 $y,h\in\{0,1\}$，$\beta$ 只能取 $0$ 或 $\pm\alpha$。
>
> 题面要求比较三个“核”：点积核 $K(x,z)=x^{\top}z$、RBF 核 $K(x,z)=\exp(-\|x-z\|^2/(2\sigma^2))$、以及一个非法函数。点积核等于不做映射的普通感知机，只能给出线性边界，能分线性可分数据，对非线性边界的数据只能部分分对。RBF 通常最好，因为它对应无穷维特征，能拟合弯曲的边界。题面给的“$x=z$ 为 $-1$ 否则 $0$”不是 PSD，会完全失败：测试点与所有训练点都不同，求和恒为 $0$，而 $0\ge0$ 判为 $1$，于是所有点都被预测成同一类。

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

这段实现对每个样本都追加一项，分对时追加的系数为 $0$，不影响预测，只是多占内存；若想节省，可以只在 `y != yhat` 时追加。预测一次的代价是 $O(\text{已见样本数})$ 次核计算，这是核方法用“不存 $\theta$”换来的代价。

### PS2.5 MNIST 单隐层神经网络

**动机**：前面的模型要么线性，要么靠固定的核引入非线性。神经网络让特征本身也可以学习。这一题从零实现一个单隐层网络对手写数字（$0$ 到 $9$，每张 $28\times28$ 像素展平为 $784$ 维）分类：自己推导 softmax 交叉熵的梯度、写前向和反向传播、用小批量梯度下降训练，再加 L2 正则观察差别。

符号与形状：$x\in\mathbb{R}^d$，$W^{[1]}\in\mathbb{R}^{d\times h}$，$b^{[1]}\in\mathbb{R}^h$，$W^{[2]}\in\mathbb{R}^{h\times K}$，$b^{[2]}\in\mathbb{R}^K$，$h$ 为隐层单元数，$K=10$ 为类别数；$y\in\mathbb{R}^K$ 是 one-hot 标签。

> [!INFO]+ PS2.5 MNIST 单隐层
>
> **前向**：$z^{[1]}=W^{[1]\top}x+b^{[1]}$，$a^{[1]}=\sigma(z^{[1]})$（sigmoid），$z^{[2]}=W^{[2]\top}a^{[1]}+b^{[2]}$，$\hat y=\mathrm{softmax}(z^{[2]})$。损失是交叉熵 $\mathrm{CE}(y,\hat y)=-\sum_k y_k\log\hat y_k$，在小批量上取平均。
>
> **输出层梯度**。设真实类别为 $l$，则 $\mathrm{CE}=-\log\hat y_l=-z^{[2]}_l+\log\sum_k e^{z^{[2]}_k}$。对 $z^{[2]}_j$ 求导：$-\mathbf{1}\{j=l\}+\hat y_j$。写成向量形式，反向输出层 $\partial\ell/\partial z^{[2]}=\hat y-y$（one-hot）。
>
> **隐层梯度**。由 $z^{[2]}=W^{[2]\top}a^{[1]}+b^{[2]}$ 得 $\partial\ell/\partial a^{[1]}=W^{[2]}(\hat y-y)$；再乘 sigmoid 的导数 $\sigma'=\sigma(1-\sigma)$，隐层 $\partial\ell/\partial z^{[1]}=W^{[2]}(\hat y-y)\odot a^{[1]}(1-a^{[1]})$。参数梯度是外积：$\partial\ell/\partial W^{[2]}=a^{[1]}(\hat y-y)^{\top}$，$\partial\ell/\partial W^{[1]}=x\,(\partial\ell/\partial z^{[1]})^{\top}$，偏置梯度就是对应的 $\partial\ell/\partial z$。
>
> 实现用小批量、把 $\log$ 与 softmax 合成稳定形式（先减去 $\max_k z_k$ 再取指数，即 log-sum-exp 技巧）。
>
> **要交的答案**：训练 / 开发准确率曲线、实现 `forward` / `backward` / `update`。

Summer 2020 题面给出的具体设置，可用于对拍：

- 60,000 张训练图被 starter code 切成 50,000 张训练集和 10,000 张开发集（dev set），另有 10,000 张测试图。
- 权重从标准正态分布初始化，偏置初始化为 $0$；隐层 $300$ 个单元，学习率 $5$，小批量大小 $B=1000$，因此每个 epoch 迭代 $50$ 次；训练 $30$ 个 epoch，每个 epoch 末在完整训练集和开发集上计算平均损失与准确率并画曲线（损失一张图、准确率一张图）。
- 正则化版本在损失上加 $\lambda(\|W^{[1]}\|^2+\|W^{[2]}\|^2)$，$\lambda=0.0001$；不正则化偏置，梯度里多出 $2\lambda W$；画图用的损失**不含**正则项。
- 测试集只在最后各用一次。题面给的参考测试准确率约为：无正则 $0.92870$，有正则 $0.96760$。正则化版本训练与开发曲线的差距更小，测试准确率更高，符合“正则化减小过拟合”的预期。

### PS2.6 正则化的贝叶斯解释

这一题在本章开头的表格里没有列出，但确实是 Summer 2020 公开 `ps2.pdf` 的第 6 题（Bayesian Interpretation of Regularization），与前面线性回归、正则化相关章节直接相连。它说明：常用的 L2 / L1 正则化，等价于给参数加高斯 / 拉普拉斯先验后做最大后验估计（maximum a posteriori, MAP）。

- 若 $p(\theta)=p(\theta\mid x)$（$x$ 不被 $\theta$ 建模，如线性回归），则 $\theta_{\mathrm{MAP}}=\arg\max_\theta p(y\mid x,\theta)p(\theta)$。
- 先验 $\theta\sim\mathcal{N}(0,\eta^2I)$ 时，$-\log p(\theta)=\frac{1}{2\eta^2}\|\theta\|_2^2+\text{常数}$，所以 MAP 等价于 $\arg\min_\theta\,-\log p(y\mid x,\theta)+\lambda\|\theta\|_2^2$，$\lambda=\frac{1}{2\eta^2}$。
- 线性回归 $y=\theta^{\top}x+\epsilon$，$\epsilon\sim\mathcal{N}(0,\sigma^2)$，再配合上述高斯先验，目标是 $\frac{1}{2\sigma^2}\|X\theta-y\|^2+\frac{1}{2\eta^2}\|\theta\|^2$，令梯度为零得闭式解 $\theta_{\mathrm{MAP}}=\big(X^{\top}X+\frac{\sigma^2}{\eta^2}I\big)^{-1}X^{\top}y$，即岭回归（ridge regression）。
- 先验各分量独立服从拉普拉斯分布 $\theta_i\sim\mathcal{L}(0,b)$ 时，$-\log p(\theta)=\frac1b\|\theta\|_1+\text{常数}$，乘以 $2\sigma^2$ 后 MAP 等价于最小化 $\|X\theta-y\|_2^2+\gamma\|\theta\|_1$，$\gamma=\frac{2\sigma^2}{b}$，即 Lasso；它没有闭式解，需要数值优化，并且倾向于产生稀疏参数。

<br>

## PS3：强化学习（倒立摆）与无监督学习

PS3 由两部分组成：开头一道基于模型的强化学习编程题（倒立摆），其余各题是无监督学习——KL 散度与极大似然的关系、$k$-means 图像压缩、半监督 EM、PCA 的重构误差解释、ICA 的高斯不可辨识性与拉普拉斯源实现。Fall 2025 起倒立摆常放到 PS4，推导不变。

### PS3.1 / PS4 倒立摆：估计模型再做值迭代

**动机**：倒立摆（inverted pendulum / pole balancing）要求小车左右加速，让铰接在车上的杆不倒、车不出界。这里不使用任何物理方程，而是让智能体从交互中**估计** MDP 的转移概率和回报，再对估计出的 MDP 解 Bellman 方程、按值函数贪心行动。这种“先学模型再规划”的思路称为基于模型的强化学习（model-based RL）。

> [!EXAMPLE]+ PS3.1 / PS4 倒立摆
>
> **题在问什么**。连续状态 $(x,\dot x,\theta,\dot\theta)$（车位置、车速、杆角度、角速度）已被离散成整数 $0$ 到 `NUM_STATES-1`。动作只有左 / 右（没有“不动”）。失败（杆倒或车出界）给负回报并重置，其余回报 $0$。$\gamma=0.995$。周期地用计数估 $T,R$，再值迭代，再贪心。
>
> 初值：$R(s)=0$，$T(s'\mid s,a)$ 均匀。维护 $N(s,a,s')$ 与回报和。每次失败后重估模型：
>
> $$
> \hat T(s'\mid s,a)=\frac{N(s,a,s')}{\sum_{u}N(s,a,u)},\qquad
> \hat R(s)=\text{该状态观测回报的平均}.
> $$
>
> 未见过的 $(s,a)$ 保持均匀。之所以转移是随机的，是因为离散化把多个相近但不同的连续状态映射到同一个整数状态；回报则是确定的。值迭代：
>
> $$
> V(s)\leftarrow \hat R(s)+\gamma\max_a\sum_{s'}\hat T(s'\mid s,a)V(s').
> $$
>
> 直到 $\max_s|V_{\mathrm{new}}-V_{\mathrm{old}}|<\mathrm{tol}$。连续若干次（参数 `NO_LEARNING_THRESHOLD`）值迭代第一步就收敛，则认为模型不再变，整个学习过程结束。正确实现时，平滑后的“撑住步数”（图中红线，对数步数的平滑值）大约在 $60$ 次失败附近变平。种子 $1,2,3$ 会得到不同曲线：离散化加随机重置，算法对初始化敏感。
>
> **要交的答案**：失败次数、学习曲线、三种种子的观察。

下面的值迭代是“同步”版本：每一轮用旧的 $V$ 一次算出整张 $Q$ 表，得到 $V_{\mathrm{new}}$ 后再与旧表比较 $\infty$ 范数，这正是作业表格里强调的写法。

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

函数返回的 `it` 就是收敛所用的轮数；`it == 0` 表示“第一步就收敛”，外层循环据此累计连续次数、判断是否达到 `NO_LEARNING_THRESHOLD`。由于 $\gamma=0.995$ 很接近 $1$，压缩率慢，首次求解可能需要较多轮，`max_iter` 不宜设得太小。

### PS3.2 KL 散度与极大似然

**动机**：KL 散度（Kullback–Leibler divergence）衡量用分布 $Q$ 去近似真实分布 $P$ 时“多付出的代价”。从信息论角度，交叉熵 $H(P,Q)=-\sum_xP(x)\log Q(x)$ 是用为 $Q$ 设计的最优编码去传输服从 $P$ 的符号时的平均码长，熵 $H(P)$ 是最优码长，二者之差就是 $D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P)$。这一题证明它的三个基本性质，最后一个说明极大似然估计本质上是在最小化经验分布与模型之间的 KL 散度。

> [!INFO]+ PS3.2 KL 与最大似然
>
> **定义**。$D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$，题面为方便假设 $P(x)>0$。
>
> **非负性**。$f=-\log$ 严格凸，把 $Q(x)/P(x)$ 看成在 $P$ 下取值的随机变量，用 Jensen 不等式 $\mathbb{E}[f(X)]\ge f(\mathbb{E}[X])$：
>
> $$
> D_{\mathrm{KL}}(P\|Q)=-\sum_x P(x)\log\frac{Q(x)}{P(x)}
> \ge -\log\sum_x Q(x)=0,
> $$
>
> 其中 $\sum_xP(x)\frac{Q(x)}{P(x)}=\sum_xQ(x)\le1$（在 $P$ 的支撑上求和），故右边 $\ge0$。等号当且仅当 $Q/P$ 为常数（严格凸函数的 Jensen 取等条件），再由两者都归一化得常数为 $1$，即 $P=Q$。
>
> **链式法则**。条件 KL 定义为 $D_{\mathrm{KL}}(P(Y\mid X)\|Q(Y\mid X))=\sum_xP(x)\sum_yP(y\mid x)\log\frac{P(y\mid x)}{Q(y\mid x)}$，即按 $P(x)$ 加权的各条件分布 KL 的期望。证明时把 $\log\frac{P(x,y)}{Q(x,y)}$ 拆成 $\log\frac{P(x)}{Q(x)}+\log\frac{P(y\mid x)}{Q(y\mid x)}$，再按 $P(x,y)$ 求和：第一项对 $y$ 求和后剩下 $D_{\mathrm{KL}}(P(X)\|Q(X))$，第二项正是条件 KL。于是 $D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))=D_{\mathrm{KL}}(P(X)\|Q(X))+D_{\mathrm{KL}}(P(Y\mid X)\|Q(Y\mid X))$。
>
> **与极大似然**。经验分布 $\hat P(x)=\frac1n\sum_i\mathbf{1}\{x^{(i)}=x\}$ 在每个训练点上质量 $1/n$（重复点的质量相加）。
>
> $$
> D_{\mathrm{KL}}(\hat P\|P_{\theta})
> =\mathrm{const}-\frac1n\sum_i\log P_{\theta}(x^{(i)}).
> $$
>
> 这里常数是 $\sum_x\hat P(x)\log\hat P(x)=-H(\hat P)$，与 $\theta$ 无关；第二项来自 $-\sum_x\hat P(x)\log P_\theta(x)=-\frac1n\sum_i\log P_\theta(x^{(i)})$。故最小化 KL 等价于最大似然：$\arg\min_\theta D_{\mathrm{KL}}(\hat P\|P_\theta)=\arg\max_\theta\sum_i\log P_\theta(x^{(i)})$。
>
> **要交的答案**：非负证明、链式法则、KL $=$ 常数 $-$ 平均对数似然。

题面的附注把链式法则和朴素贝叶斯联系起来：模型 $P_\theta(x,y)=p(y)\prod_{j=1}^dp(x_j\mid y)$ 时，KL 可以拆成类先验一项加上每个特征、每个类别各一项条件 KL，因此极大似然估计分解成 $2d+1$ 个互相独立的小问题，各自做计数即可。这解释了为什么朴素贝叶斯的参数可以“逐个数出来”。

### PS3.3 用 $k$-means 压缩图像颜色

**动机**：一张 24 位真彩色图片可能用到上万种颜色，但人眼通常分辨不出其中很多细微差别。把所有像素颜色聚成 $16$ 类，每个像素只记“属于哪一类”，就能大幅压缩而画面基本可辨。这是 $k$-means 作为矢量量化（vector quantization）的典型应用。

> [!EXAMPLE]+ PS3.3 $k$-means 压颜色
>
> 像素看成 $\mathbb{R}^3$ 的 $(r,g,b)$，每个分量是 $0$ 到 $255$ 的 8 位整数。大图 $512\times512$ 共 262144 个像素，直接聚类太慢，所以在小图（`peppers-small.tiff`）上跑 $k=16$，中心初始化为随机像素，最好迭代到收敛，至少 $30$ 轮。然后把大图（`peppers-large.tiff`）每个像素换成最近中心，与原图并排比较。
>
> 压缩比：原图每像素 $24$ 位。新图每像素 $4$ 位索引（$16=2^4$ 色）再加调色板 $16\times 24$ 位，调色板相对 $512^2$ 可忽略，约 $24/4=6$ 倍。
>
> **要交的答案**：大图对比、约 $6$ 倍。

实现上，“最近中心”可以用广播一次算出所有像素到 $16$ 个中心的距离矩阵（形状为像素数 $\times16$），再按行取 `argmin`；读入的像素若是 `uint8`，计算距离前要先转成浮点数，否则平方差会溢出回绕。

### PS3.4 半监督 EM

**动机**：标准 EM 只用无标签数据，结果依赖初始化，可能把簇分错。如果另外有少量**带标签**的样本（知道它属于哪个高斯分量），就可以把它们一起放进目标函数，让它们“锚定”各个簇。这一题推导这种半监督 EM，证明它仍单调收敛，并在高斯混合模型（GMM）上实现。

符号：$n$ 个无标签样本 $x^{(i)}$，隐变量 $z^{(i)}$ 未观测；$\tilde n$ 个有标签样本 $(\tilde x^{(i)},\tilde z^{(i)})$；$\alpha>0$ 是有标签部分的权重超参数。

> [!INFO]+ PS3.4 半监督 EM
>
> **目标**。$\ell_{\mathrm{semi}}=\ell_{\mathrm{unsup}}+\alpha\ell_{\mathrm{sup}}$，其中 $\ell_{\mathrm{unsup}}(\theta)=\sum_{i=1}^n\log\sum_{z}p(x^{(i)},z;\theta)$ 是无标签样本的边缘对数似然，$\ell_{\mathrm{sup}}(\theta)=\sum_{i=1}^{\tilde n}\log p(\tilde x^{(i)},\tilde z^{(i)};\theta)$ 是有标签样本的完全数据对数似然。
>
> **E 步与 M 步**。E 步只对**未标注**样本算 $Q_i^{(t)}(z)=p(z\mid x^{(i)};\theta^{(t)})$——有标签样本的 $z$ 已知，没有需要估计的隐变量。M 步把未标注的期望完全数据对数似然，加上 $\alpha$ 倍已标注的真实完全数据对数似然，对 $\theta$ 最大化：
>
> $$
> \theta^{(t+1)}=\arg\max_\theta\Big[\sum_{i=1}^n\sum_zQ_i^{(t)}(z)\log\frac{p(x^{(i)},z;\theta)}{Q_i^{(t)}(z)}+\alpha\sum_{i=1}^{\tilde n}\log p(\tilde x^{(i)},\tilde z^{(i)};\theta)\Big].
> $$
>
> **单调性**。记方括号内为 $L(\theta;Q^{(t)})$。由 Jensen 不等式，对任意 $\theta$ 有 $\ell_{\mathrm{unsup}}(\theta)\ge\sum_i\sum_zQ_i^{(t)}(z)\log\frac{p(x^{(i)},z;\theta)}{Q_i^{(t)}(z)}$，因此 $\ell_{\mathrm{semi}}(\theta)\ge L(\theta;Q^{(t)})$；而 E 步取后验使该不等式在 $\theta^{(t)}$ 处取等。于是
>
> $$
> \ell_{\mathrm{semi}}(\theta^{(t+1)})\ge L(\theta^{(t+1)};Q^{(t)})\ge L(\theta^{(t)};Q^{(t)})=\ell_{\mathrm{semi}}(\theta^{(t)}),
> $$
>
> 中间的不等号来自 M 步的最大化。直观地说：标准 EM 用来证明 $\ell_{\mathrm{unsup}}(\theta^{(t+1)})\ge\ell_{\mathrm{unsup}}(\theta^{(t)})$ 的那套下界论证，在下界上再加上 $\alpha\ell_{\mathrm{sup}}$（它在 M 步与下界一起被显式最大化，且本身精确、不需要下界近似）后照样成立，故加权和上升。需要注意，单调上升的是加权和 $\ell_{\mathrm{semi}}$；“$\ell_{\mathrm{unsup}}$ 上升、$\ell_{\mathrm{sup}}$ 也上升，所以和上升”这种分项论证并不成立，因为 M 步最大化的是两项之和，两项各自并不保证单独上升。
>
> **GMM：E 步**。未标注责任度
>
> $$
> w_j^{(i)}=\frac{\phi_j\,\mathcal{N}(x^{(i)};\mu_j,\Sigma_j)}{\sum_\ell\phi_\ell\,\mathcal{N}(x^{(i)};\mu_\ell,\Sigma_\ell)}.
> $$
>
> **GMM：M 步**。已标注样本的责任度是 one-hot（$\mathbf{1}\{\tilde z^{(i)}=j\}$），再乘 $\alpha$。均值 / 协方差 / $\phi$ 用“未标注权重 $+$ $\alpha$ 已标注计数”的加权公式：
>
> $$
> \phi_j=\frac{\sum_{i=1}^nw_j^{(i)}+\alpha\sum_{i=1}^{\tilde n}\mathbf{1}\{\tilde z^{(i)}=j\}}{n+\alpha\tilde n},\qquad
> \mu_j=\frac{\sum_{i=1}^nw_j^{(i)}x^{(i)}+\alpha\sum_{i=1}^{\tilde n}\mathbf{1}\{\tilde z^{(i)}=j\}\tilde x^{(i)}}{\sum_{i=1}^nw_j^{(i)}+\alpha\sum_{i=1}^{\tilde n}\mathbf{1}\{\tilde z^{(i)}=j\}},
> $$
>
> $$
> \Sigma_j=\frac{\sum_{i=1}^nw_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^{\top}+\alpha\sum_{i=1}^{\tilde n}\mathbf{1}\{\tilde z^{(i)}=j\}(\tilde x^{(i)}-\mu_j)(\tilde x^{(i)}-\mu_j)^{\top}}{\sum_{i=1}^nw_j^{(i)}+\alpha\sum_{i=1}^{\tilde n}\mathbf{1}\{\tilde z^{(i)}=j\}}.
> $$
>
> 当 $\tilde n=0$ 时这三式退化为普通 GMM 的 M 步，可以用来对拍。
>
> **实验观察**。题面先跑普通 EM（只用无标签数据）三次，再跑半监督 EM（每簇 5 个标签）三次，比较收敛轮数、稳定性与分配质量。半监督通常更稳、更少轮，并能分开低方差三簇与一簇高方差噪声（数据由三个低方差高斯和一个高方差高斯混合生成）；普通 EM 不同初始化下的结果差别较大，还常把高方差那一簇和别的簇混在一起。
>
> **要交的答案**：单调性、$Q$ 公式、$\mu,\Sigma,\phi$ 的加权更新、六张散点、三点对照。

实现时的常见坑：高斯密度直接相乘容易下溢，计算责任度应在对数域做并用 log-sum-exp 归一化；收敛判据通常用 $\ell_{\mathrm{semi}}$ 的变化量小于阈值，监控这个值是否单调上升也是检查 M 步是否写对的最直接办法。

### PS3.5 PCA 的另一种解释：最小重构误差

**动机**：讲义里 PCA 是“找方差最大的投影方向”。这一题从另一个角度得到同一个答案：找一条过原点的直线，使所有点投影到它上面后的平方误差最小。两种解释等价，说明 PCA 同时是“保留最多信息”和“压缩损失最小”。

> [!INFO]+ PS3.5 PCA 的重构误差
>
> **投影**。对单位向量 $u$，$f_u(x)=\arg\min_{v\in\{\lambda u\}}\|x-v\|^2$ 是 $x$ 在 $u$ 方向上的投影。对 $\lambda$ 求导置零得 $\lambda=u^{\top}x$，所以 $f_u(x)=(u^{\top}x)u$。
>
> **重构误差**。展开并利用 $u^{\top}u=1$：
>
> $$
> \|x-f_u(x)\|_2^2=\|x\|_2^2-2(u^{\top}x)^2+(u^{\top}x)^2u^{\top}u=\|x\|_2^2-(u^{\top}x)^2.
> $$
>
> 即 $\|x-f_u(x)\|_2^2=\|x\|_2^2-(u^{\top}x)^2$：重构误差等于原长度平方减去投影长度平方（勾股定理）。$\|x\|^2$ 与 $u$ 无关，而 $\sum_i(u^{\top}x^{(i)})^2=u^{\top}(\sum_i x^{(i)}x^{(i)\top})u$，因此最小化平均重构误差等价于最大化 $u^{\top}(\sum_i x^{(i)}x^{(i)\top})u$，约束 $u^{\top}u=1$。
>
> **特征方程**。记 $S=\frac1n\sum_ix^{(i)}x^{(i)\top}$。拉格朗日函数 $u^{\top}Su-\lambda(u^{\top}u-1)$ 对 $u$ 求导置零，拉格朗日乘数给出 $Su=\lambda u$，此时目标值 $u^{\top}Su=\lambda$，所以最大特征值对应第一主成分。数据须先中心化（题面假设已做零均值、单位方差预处理）：只有中心化后 $S$ 才是协方差矩阵，否则第一方向会被均值方向“抢走”。
>
> **要交的答案**：从 $\|x\|^2-(u^{\top}x)^2$ 走到特征方程。

题面附注：若要找 $k$ 维子空间使重构平方误差最小，应取前 $k$ 个主成分张成的子空间；本题证明的是 $k=1$ 的情形。

### PS3.6 独立成分分析：高斯源的歧义与拉普拉斯源

**动机**：ICA 的鸡尾酒会问题是：$d$ 个独立声源 $s\in\mathbb{R}^d$ 被未知的混合矩阵 $A$ 混合成 $x=As$，要求出解混矩阵 $W=A^{-1}$，使 $s=Wx$。讲义中提到源分布不能是高斯，但只给出了直观论证。这一题先严格说明高斯源为何不可辨识，再推导拉普拉斯源下的随机梯度上升规则并实现。

符号：$w_j^{\top}$ 是 $W$ 的第 $j$ 行；训练集写成设计矩阵 $X\in\mathbb{R}^{n\times d}$（每行一个样本）；对数似然为 $\ell(W)=\sum_{i=1}^n\big(\log|W|+\sum_{j=1}^d\log g'(w_j^{\top}x^{(i)})\big)$，$g'$ 是源分布的密度。

> [!INFO]+ PS3.6 ICA
>
> **高斯源**。源为标准正态时，$\log g'(w^{\top}x)=-\frac12(w^{\top}x)^2+\mathrm{const}$。对 $n$ 个样本求和，最大化 $n\log|W|-\frac12\|XW^{\top}\|_F^2$。（原笔记把第一项写成 $\log|W|$，漏了因子 $n$，这里补上；它只影响下面闭式里的常数。）
>
> 对 $W$ 求梯度并置零：$nW^{-\top}-WX^{\top}X=0$，整理得 $W^{\top}W=n(X^{\top}X)^{-1}$。这个条件只约束 $W$ 使 $XW^{\top}$ 白化（样本协方差为单位阵），并不唯一确定 $W$：对任意正交（旋转）矩阵 $R$，$RW$ 满足 $(RW)^{\top}(RW)=W^{\top}W$，且 $|\det(RW)|=|\det W|$、$\|XW^{\top}R^{\top}\|_F=\|XW^{\top}\|_F$。即把 $W$ 左乘任意旋转矩阵（等价地，把 $XW^{\top}$ 右乘 $R^{\top}$），似然不变。原因是标准多元高斯是旋转对称的，旋转后的源仍是独立标准高斯，数据无法区分。高斯源不可辨识。
>
> **拉普拉斯源**。Laplace 密度 $\frac12 e^{-|s|}$，于是 $\log g'(s)=-|s|-\log2$，其导数为 $-\mathrm{sign}(s)$。单样本对数似然 $\log|W|+\sum_j\log g'(w_j^{\top}x)$ 对 $W$ 的梯度是 $(W^{-1})^{\top}-\mathrm{sign}(Wx)\,x^{\top}$（第一项用 $\nabla_W\log|W|=(W^{-1})^{\top}$）。单样本随机梯度上升：
>
> $$
> W\leftarrow W+\alpha\big((W^{-1})^{\top}-\mathrm{sign}(Wx)\,x^{\top}\big).
> $$
>
> **实现**。数据 `mix.dat` 是 5 列，每列一路混合信号，所以 $W$ 是 $5\times5$。实现时对学习率退火（逐渐减小，每个学习率跑一整遍数据），并打乱样本顺序（随机排列后按顺序做随机梯度上升）。正确 $W$ 应让 `split_0.wav` 接近参考音轨 `correct_split_0.wav`；分离结果允许有少量串音或噪声，但各路声音应清楚可辨。
>
> **要交的答案**：高斯下的旋转歧义、Laplace 更新、`W.txt`。

与讲义中 logistic 源的更新 $W\leftarrow W+\alpha\big((W^{-1})^{\top}+(1-2g(Wx))x^{\top}\big)$ 对比，只是把 $1-2g(\cdot)$ 换成了 $-\mathrm{sign}(\cdot)$：两者都是对应源密度的对数导数（score function）。即便换成非高斯源，ICA 仍然无法确定各源的排列顺序；对拉普拉斯这类对称密度，各源的符号也无法确定（若源密度的尺度未知，还有尺度歧义）。这些歧义对“把声音分开”无害，与高斯源下任意旋转都无法区分的歧义性质不同。

<br>

> [!NOTE]+ 小结
>
> - PS1 通过 Hessian 半正定与 GLM 凸性说明逻辑回归、泊松回归等都是凸问题；GDA 在共享协方差时给出与逻辑回归同形的线性边界，但假设更强。
> - PS1.2 的正样本部分标签问题中，$p(t=1\mid x)$ 与 $p(y=1\mid x)$ 只差常数 $1/\alpha$，$\alpha$ 用验证集正例上的平均预测估计。
> - PS2.1 中线性可分导致无正则逻辑回归的 $\|\theta\|\to\infty$，正则化或 SVM 能把解拉回有限点。
> - 核的合法性靠 Mercer 定理检验；核感知机只存系数 $\beta$，非 PSD 的“核”会使算法完全失效。
> - PS3 的推导共用两件工具：Jensen 不等式（KL 非负、EM 单调）与特征分解 / 旋转对称性（PCA、ICA 高斯歧义）。

<br>

> [!NOTE]+ 衔接
>
> 本章的作业按 Problem Set 组织。下一章把同一张表对回项目：数据划分、评价指标与非深度学习基线，关注的是实际项目中的问题，而不是“短信”或“倒立摆”这类作业设定。

<br>

<br>

# 机器学习实践：诊断、评估与项目流程

前面各章解决的是“给定一个问题，怎样写出目标函数并求解”。但拿到一份真实数据集之后，最常见的失败往往不在梯度公式上：梯度多半已经写对，真正出错的是数据怎样划分、报告了什么指标、有没有信息泄漏、下一步该改什么。一个逻辑回归在验证集上 99% 准确率，可能只是因为同一位患者的切片同时出现在训练集和测试集；一个垃圾邮件分类器测试误差 20%，可能该加数据，也可能该换目标函数，盲目尝试会浪费数周时间。

这一章的核心想法是：**先诊断，再修改**。把“模型为什么不够好”拆成可以用实验回答的问题（偏差还是方差？优化算法没收敛还是目标函数选错了？流水线里哪个部件拖了后腿？），再根据答案决定下一步。2025 Autumn 的 Lecture 13 与 TA 的评估指标讲座对应本章内容；诊断方法部分取自 Andrew Ng 的经典讲义 *Advice for applying Machine Learning*（[ML-advice.pdf](https://cs229.stanford.edu/materials/ML-advice.pdf)）。

本章的输入是一份原始数据集，输出是一份可复现的实验记录：固定的划分、明确的指标、可对照的基线、带切片的误差表。顺序是：数据划分 → 评估指标 → 基线 → 诊断学习算法（偏差/方差、优化/目标）→ 误差分析与消融分析 → 项目节奏。它与前后章的关系是：即使前面所有公式都推导正确，仍可能因为泄漏或指标错位而失败，所以本章先把评估协议固定下来；下一章再把全课公式收成一张默写清单。

## 数据划分：训练集、验证集与测试集

划分要解决的问题是：用一份数据既训练模型、又挑选超参数、又报告最终效果，三件事会相互污染。标准做法是切成三份，各司其职：

- **训练集（training set）**：用来拟合参数 $\theta$。
- **验证集（validation / dev set）**：用来选择超参数（如正则强度 $\lambda$、核带宽、树深度）、比较模型、做误差分析。
- **测试集（test set）**：只在最后用一次，给出对未见数据性能的无偏估计。

三份数据必须在**同一条协议**下切开，而且协议要符合模型部署时面对的情形：

- **时间数据按时间切。** 部署时模型只能看到过去、预测未来，因此训练集应取较早的时段，验证与测试取较晚的时段。随机打乱会让模型“偷看未来”。
- **实体数据整实体划分。** 患者、用户、文档这类实体，同一实体的所有样本必须进入同一份数据。否则同一个人的两张切片分别进入训练与测试，模型只需记住这个人的特征就能答对，指标会虚高，部署到新患者时却会大幅下降。
- **测试集只用一次。** 如果根据测试结果回头改模型（换特征、调 $\lambda$），测试集实际上已经成了验证集的一部分，它就失去了一次无偏评估的意义。需要反复比较时，应使用验证集或在训练集上做交叉验证（cross-validation）。
- **所有“从数据估计”的预处理只在训练折上估计。** 标准化用的均值与方差、词表、目标编码（target encoding）、PCA 的主方向等，都只能用训练折计算，再原样应用到验证与测试。若用全量数据估计，验证与测试的信息就通过这些统计量泄漏进了训练。

> [!WARNING]+ 常见泄漏
>
> 泄漏的共同点是：训练时用到了部署时拿不到的信息。典型形式有四种：随机划分时间序列、同一实体跨越训练与测试、在全量数据上做标准化或建词表、用测试集挑选超参数。它们都会让离线指标好于真实表现，而且很难从训练曲线上看出来。

## 评估指标：先写错误代价，再选指标

指标要回答的问题是“模型的哪种错误代价最高”。同样是 95% 准确率，在正负样本各占一半的任务上可能不错，在 1% 阳性率的疾病筛查上却可能只是“全部预测阴性”。因此顺序应当是先写清错误代价（漏诊与误诊哪个更贵？排序靠前的错误是否更严重？），再选指标。2025 Autumn 的 TA Lecture 4 专门讲评估，原因正是作业与项目里最常见的失败不是公式写错，而是报告了一个与决策无关的数字。

下表列出各类任务的常用指标及其陷阱：

| 任务 | 常用指标 | 陷阱 |
| :--- | :--- | :--- |
| 平衡二分类 | 准确率、AUC | 阈值 $0.5$ 不必最优 |
| 不平衡分类 | PR-AUC、F1、召回@固定精度 | 准确率被多数类刷高 |
| 概率输出 | ECE、可靠性图 | SVM 间隔不是概率 |
| 回归 | MAE、MSE、分位损失 | MSE 被离群点主导 |
| 检索 / 排序 | Recall@$k$、nDCG | 只报 MRR 会忽略长尾 |
| 生成 | 任务准确率、忠实度 | 流畅 $\ne$ 正确 |
| RL | 评估回报、约束违反 | TD 损失下降不等于会玩 |

对表中各行的补充说明如下：

- **平衡二分类。** AUC 衡量的是排序能力，与阈值无关；准确率则依赖阈值。阈值 $0.5$ 只在两类错误代价相同且概率校准良好时才合理，实际应在验证集上按代价选阈值。
- **不平衡分类。** 精确率（precision）是预测为正的样本中真正为正的比例，召回率（recall）是真正为正的样本中被找出的比例，F1 是二者的调和平均。PR-AUC 关注正类，比 ROC-AUC 更能反映稀有类的表现。“召回@固定精度”适合业务上有硬性误报上限的场景。
- **概率输出。** 期望校准误差（expected calibration error, ECE）把预测概率分桶，比较每桶的平均预测概率与实际正例频率；可靠性图（reliability diagram）是它的可视化。SVM 输出的是到超平面的有符号距离（间隔），不是概率，需要另做校准（如 Platt scaling）才能当概率用。
- **回归。** MSE 对误差平方求平均，一个离群点的大误差会主导整个指标；MAE 更稳健；分位损失（quantile loss）用于预测区间或不对称代价。
- **检索 / 排序。** Recall@$k$ 看前 $k$ 个结果召回了多少相关项，nDCG 对排位靠前的相关项给更高权重。MRR 只看第一个相关结果的位置，会忽略其余相关项与长尾查询。
- **生成。** 语言流畅不等于内容正确，应报告任务准确率与忠实度（faithfulness，即输出是否被输入或检索内容支持）。
- **强化学习。** TD 损失下降只说明值函数拟合了自身的自举目标，不说明策略变好；应报告评估回报与约束违反次数。

### 优化指标与满足指标

当关心的数字不止一个时（准确率、延迟、模型大小），宜指定**一个优化指标（optimizing metric）**，其余作为**满足指标（satisficing metric）**，即只需达到阈值的约束。例如“在单次推理延迟不超过 100 ms 的前提下最大化 F1”。这样每次比较两个模型时都有唯一答案，不会陷入多指标间的反复权衡。

### 误差分析的最低要求

只报一个总体数字不足以指导下一步。误差分析至少应包含：

1. **混淆矩阵（confusion matrix）**：看错误集中在哪些类别之间。
2. **阈值曲线**：精确率、召回率随阈值变化的曲线，用来选工作点。
3. **切片（slice）**：按输入长度、人群、时间段等分组报告指标。总体平均可能掩盖某个子群体上的严重失败。

下一步实验应由**占比最大的错误类型**决定，而不是默认换一个更大的模型。例如错误中 60% 来自模糊图像、5% 来自标注错误，那么修正标注最多只能消除 5% 的错误，这就是该方向的“指标上限”。

## 基线：证明复杂模型确实必要

基线要回答的问题是：任务到底有多难，复杂模型的提升从哪里来。结果表应至少包含：

- **常数预测**：分类时总是预测多数类，回归时总是预测训练集均值。它给出“什么都不学”能达到的水平；不平衡数据上这个数字往往高得惊人。
- **至少一个简单模型**：线性模型、逻辑回归、$k$-NN 或单棵决策树之一。

如果深度模型或 LLM 只是优于一个没有写明的方法，并不能说明任务需要深度结构；只有当它显著超过合理调参的简单基线时，才有理由使用更复杂的假设类。同理，模型里的每个组件（正则、核、检索）都应分别去掉并报告结果，这与下文的消融分析是同一思路。负结果也应保留：简单模型与复杂模型都做不好，往往说明问题出在假设类之外（特征不足或数据本身不够），这本身就是有价值的结论。

## 诊断学习算法

### 动机：不要靠运气修模型

讲义的引例是垃圾邮件过滤。从 50000 多个英文词中精心挑出 100 个词作为特征，用梯度下降训练贝叶斯逻辑回归（Bayesian logistic regression，即带 $\lambda\|\theta\|^2$ 正则的逻辑回归，可看作参数上加高斯先验的 MAP 估计），测试误差为 20%，高得无法接受。接下来该做什么？

常见做法是凭直觉逐个尝试：多收集训练样本、用更少的特征、用更多的特征、改用邮件头而非正文的特征、梯度下降多跑几轮、换牛顿法、换一个 $\lambda$、换成 SVM。这些尝试也许有效，但非常耗时，能否碰巧修到真正的问题很大程度上靠运气。更好的做法是：**先跑诊断确定问题是什么，再针对性地修。**

### 偏差-方差诊断与学习曲线

假设怀疑的原因只有两种：过拟合（高方差，high variance），或者 100 个词的特征太少、模型表达能力不足（高偏差，high bias）。两者可以用训练误差与测试误差的关系区分：

- **高方差**：训练误差远低于测试误差。
- **高偏差**：训练误差本身也很高。

更细的工具是**学习曲线（learning curve）**：横轴是训练集大小 $m$，纵轴是误差，同时画出训练误差与测试（或验证）误差，再画一条“期望性能”水平线。

- **高方差的典型形态**：训练误差低于期望水平，测试误差高于期望水平，二者之间有**很大的间隙**；而且随着 $m$ 增大，测试误差仍在下降。这说明继续增加训练数据会有帮助。
- **高偏差的典型形态**：**连训练误差都高于**期望水平，训练与测试误差之间**间隙很小**，两条曲线早早变平。此时加再多数据也无济于事，因为模型连训练集都拟合不好。

直觉上，$m$ 越大，训练误差会上升（更难记住所有样本），测试误差会下降（估计更稳定），两者逐渐靠拢到模型族能达到的最好水平。若这个水平本身就不够好，就是偏差问题；若离它还很远，就是方差问题。

### 优化算法还是目标函数

偏差-方差只是最常见的一种诊断。讲义的第二个例子说明，很多时候需要自己设计诊断。

在同一个垃圾邮件任务上：贝叶斯逻辑回归（BLR）在垃圾邮件上误差 2%、在正常邮件上误差 2%，后者无法接受（把正常邮件误判为垃圾的代价很高）；线性核 SVM 在垃圾邮件上误差 10%、在正常邮件上误差 0.01%，可以接受。但出于计算效率等原因，仍然希望部署逻辑回归。下一步怎么办？

这里有两个常见疑问：梯度下降是否已经收敛？优化的是不是正确的函数（对 BLR 而言，$\lambda$ 取值是否正确；对 SVM 而言，$C$ 取值是否正确）？仅从 $J(\theta)$ 随迭代次数的曲线，往往很难判断算法是否已经收敛。下面的诊断把两个疑问分开。

记 $\theta_{\mathrm{SVM}}$ 为 SVM 学到的参数，$\theta_{\mathrm{BLR}}$ 为贝叶斯逻辑回归学到的参数。真正关心的是**加权准确率**（正常邮件的样本权重更高）：

$$
a(\theta)=\sum_i w^{(i)}\,1\{h_\theta(x^{(i)})=y^{(i)}\},
$$

其中非垃圾邮件的权重 $w^{(i)}$ 高于垃圾邮件。已知 SVM 在这个指标上更好，即 $a(\theta_{\mathrm{SVM}})>a(\theta_{\mathrm{BLR}})$。而 BLR 实际最大化的是正则化对数似然：

$$
J(\theta)=\sum_i \log p\big(y^{(i)}\mid x^{(i)},\theta\big)-\lambda\|\theta\|^2 .
$$

诊断方法是：把两组参数都代入 BLR 自己的目标 $J$，比较 $J(\theta_{\mathrm{SVM}})$ 与 $J(\theta_{\mathrm{BLR}})$。

- **情形 1：$J(\theta_{\mathrm{SVM}})>J(\theta_{\mathrm{BLR}})$。** BLR 本来就在最大化 $J$，却找到了一个比 SVM 的参数还差的点，说明 BLR 没能把 $J$ 优化好，**问题出在优化算法**（未收敛等）。
- **情形 2：$J(\theta_{\mathrm{SVM}})\le J(\theta_{\mathrm{BLR}})$。** BLR 成功地最大化了 $J$，但在 $J$ 上更差的 SVM 反而在关心的 $a(\theta)$ 上更好。这说明 $J$ 不是该最大化的函数，**问题出在目标函数**。

于是前面那张“可尝试的修法”清单可以按诊断结果归类：

| 修法 | 针对的问题 |
| :--- | :--- |
| 收集更多训练样本 | 高方差 |
| 使用更少的特征 | 高方差 |
| 使用更多的特征 | 高偏差 |
| 加入邮件头特征 | 高偏差 |
| 梯度下降多跑几轮 | 优化算法 |
| 改用牛顿法 | 优化算法 |
| 换一个 $\lambda$ | 优化目标 |
| 改用 SVM | 优化目标 |

这张表的意义是：同样一句“效果不好”，诊断结果不同，正确的修法完全不同；加数据对高偏差无效，多跑几轮对目标函数错位无效。

### 例：自主直升机的强化学习诊断

讲义的第三个例子是斯坦福自主直升机（载荷 14 磅、自重 32 磅）。做法分三步：

1. 搭建直升机的模拟器；
2. 选择代价函数，例如 $J(\theta)=\|x-x_{\mathrm{desired}}\|^2$，其中 $x$ 为直升机位置；
3. 在模拟器里运行强化学习（RL）算法，求 $\theta_{\mathrm{RL}}=\arg\min_\theta J(\theta)$。

如果得到的控制器在真机上远不如人类飞行员，该改进模拟器、改代价函数，还是改 RL 算法？思路是先写出“一切正常”时应成立的三条假设：

1. 模拟器是准确的；
2. RL 算法在模拟中正确地最小化了 $J(\theta)$；
3. 最小化 $J(\theta)$ 对应着良好的自主飞行。

三条都成立，$\theta_{\mathrm{RL}}$ 就应当在真机上飞得好。于是逐条检验：

1. 若 $\theta_{\mathrm{RL}}$ 在模拟中飞得好、在真机上飞得差，问题在**模拟器**。否则继续：
2. 记 $\pi_{\mathrm{human}}$ 为人类飞行员的控制策略。若 $J(\pi_{\mathrm{human}})<J(\theta_{\mathrm{RL}})$，即人类在 RL 自己的代价上都更好，说明 RL 没有把 $J$ 最小化，问题在 **RL 算法**。
3. 若 $J(\pi_{\mathrm{human}})\ge J(\theta_{\mathrm{RL}})$，RL 在 $J$ 上已经不比人差，但飞得更差，说明最小化 $J$ 并不对应良好飞行，问题在**代价函数**。

这与上一小节的“优化还是目标”诊断是同一个模式：拿一个已知更好的参照物，把它代入自己的目标函数，看问题是“没优化好”还是“优化错了东西”。

### 诊断的其他价值

很多时候需要自己发明诊断。即使算法已经工作良好，也值得跑诊断以理解它为什么有效：长期做一个应用时，这能积累关于“什么有效、什么无效”的直觉；写论文时，诊断与误差分析能支撑结论，与其只说“这个算法有效”，不如说“它有效是因为组件 X，证据如下”。

## 误差分析与消融分析

### 误差分析：当前性能离完美差在哪里

许多应用是由多个学习组件串成的**流水线（pipeline）**。讲义用一个人为构造的人脸识别例子：相机图像 → 预处理（去背景）→ 人脸检测 → 眼睛 / 鼻子 / 嘴巴分割 → 逻辑回归 → 输出标签。整个系统准确率 85%，误差分别来自哪个组件？

**误差分析（error analysis）** 的做法是：从流水线前端开始，**依次把各组件的输出替换为真值（ground truth）**，每替换一个就记录整体准确率。由于替换是累积的，相邻两行之差就是该组件造成的损失：

| 组件（替换为真值，累积） | 整体准确率 |
| :--- | :--- |
| 整个系统 | 85% |
| 预处理（去背景） | 85.1% |
| 人脸检测 | 91% |
| 眼睛分割 | 95% |
| 鼻子分割 | 96% |
| 嘴巴分割 | 97% |
| 逻辑回归 | 100% |

预处理只带来 0.1 个百分点的提升，人脸检测带来 5.9 个百分点，眼睛分割带来 4 个百分点，鼻子分割与嘴巴分割各带来 1 个百分点，最后的逻辑回归带来 3 个百分点。结论：最大的改进空间在人脸检测和眼睛分割；在去背景上投入大量精力几乎没有回报。

### 消融分析：当前性能比基线好在哪里

误差分析解释的是“当前性能与完美性能之间的差距”；**消融分析（ablative analysis）** 解释的是“某个（差得多的）基线与当前性能之间的差距”。

例：给逻辑回归垃圾邮件分类器加了许多精巧特征：拼写纠正、发件主机特征、邮件头特征、正文解析特征、JavaScript 解析器、内嵌图片特征。不带任何精巧特征的简单逻辑回归准确率 94%，完整系统 99.9%。从 94% 到 99.9% 的提升到底来自哪里？做法是**从完整系统开始，一次去掉一个组件（累积地）**，看系统怎样变差：

| 组件（依次去掉，累积） | 准确率 |
| :--- | :--- |
| 整个系统 | 99.9% |
| 拼写纠正 | 99.0% |
| 发件主机特征 | 98.9% |
| 邮件头特征 | 98.9% |
| 正文解析特征 | 95% |
| JavaScript 解析器 | 94.5% |
| 图片特征 | 94.0%（基线） |

去掉正文解析特征时准确率从 98.9% 跌到 95%，降幅最大。结论：正文解析特征贡献了大部分提升。需要注意，由于去掉的顺序是累积的，结果依赖顺序；组件之间有交互时，换个顺序可能得出不同的归因，必要时可以分别单独去掉每个组件再比较。

## 如何开始一个机器学习问题

### 精心设计与快速迭代

讲义对比了两种开局方式：

- **方式一：精心设计（careful design）。** 花很长时间设计恰当的特征、收集恰当的数据集、设计恰当的算法结构，然后实现并期待它能工作。好处是可能得到更优雅、更可扩展的算法，甚至提出新的学习算法、为基础研究做贡献。
- **方式二：构建再修正（build-and-fix）。** 先实现一个“快而粗糙”（quick-and-dirty）的版本，再用误差分析和诊断找出问题并逐个修复。好处是通常能更快让应用跑起来，更快上线。

### 过早的统计优化

很多时候事先并不清楚系统的哪部分容易、哪部分难、哪部分值得投入大量时间。上面那条人脸识别流水线作为第一次尝试就过于复杂了；设计学习系统的第一步应当是**把数据画出来看**。弄清楚哪里需要下功夫的唯一办法，是快速实现一个版本，看哪里坏掉。讲义把在不知道瓶颈时就精雕细琢某个部件称为“过早的（统计）优化”（premature statistical optimization），并用一张“过度理论化”的示意图提醒：从送信机器人出发，一路钻进导航、物体检测、三维流形的微分几何、VC 维、收敛界……离原本要解决的问题越来越远（该图基于 Papadimitriou, 1995）。

讲义也指出，这条建议在目标是提出新的机器学习算法时未必适用。

> [!NOTE]+ 诊断方法小结
>
> - 花在设计诊断上的时间是值得的，而合适的诊断常常需要自己想出来。
> - 偏差-方差诊断看训练误差与验证误差的关系与学习曲线。
> - “优化还是目标”诊断把一个已知更好的参照解代入自己的目标函数比较。
> - 误差分析解释当前性能与完美性能的差距，消融分析解释基线与当前性能的差距。
> - 应用问题宜先做快而粗糙的原型，再诊断和修复，避免过早的统计优化。

## 课程项目：时间线、要求与常见失败

公开时间线（2025 Autumn）包括四个节点：提案、中期、报告、海报。各节点的要求如下：

- **提案**应写清：要解决的问题；数据的许可与规模；划分方式；评估指标；最简单的基线；主要风险。
- **中期**必须已经有数字和误差表，不能只有计划。
- **最终报告**应让他人能复现划分与运行命令。
- **与其他课程合并项目时**，CS229 部分必须能单独拆出来，包括：假设（模型类）、似然或损失、与非深度基线的比较。

常见的失败方式有：选题过大；到中期仍没有标签；用测试集调参；不固定随机种子（seed）；把作业数据换个标题当作项目。建议先把范围缩小，跑通完整流程后再扩展，这也正是上一节“构建再修正”的思路。

> [!WARNING]+ 项目失败模式
>
> 公式正确仍可能因实体泄漏、用测试选 $\lambda$、只报总体准确率、缺少非深度基线或误差表无切片而失分。应先固定评估协议，再增加模型容量。

<br>

> [!NOTE]+ 衔接
>
> 划分、指标、基线与误差表是把前面各章算法接到未布置过的数据上的接口。下一章把公式收成清单：清单用来答题，协议用来做项目。

<br>

# 全课复习：公式与算法速查

复习的目的是把全课串成一条线，并把考试和作业最常用的公式集中在一处。整门课可以看成同一种写法的反复使用：**写出一个可声明的目标（似然、损失、ELBO、期望回报），再给出一个可实现的求解器（闭式解、梯度下降、牛顿法、EM、动态规划）。** 各主题依次扩展这个模板：

- 线性模型给出闭式解与凸优化；
- 指数族与广义线性模型把回归与分类收成同一种构造；
- 核方法与神经网络扩大假设类；
- 无监督学习补上没有标签 $y$ 时的密度估计与表示学习；
- 强化学习把“预测”换成“决策”。

## 三条失败轴

模型失败可以按“拟合到哪一层”分成三类，每一类对应不同的原因：

$$
\begin{aligned}
\text{拟合训练} &\leftarrow \text{假设类、优化、可辨识性},\\
\text{拟合验证} &\leftarrow \text{正则、数据、划分},\\
\text{拟合部署} &\leftarrow \text{验证分布是否等于现实，以及公平与隐私约束}.
\end{aligned}
$$

读法是：训练集都拟合不好，要看假设类够不够、优化有没有收敛、参数是否可辨识；训练好而验证差，要看正则、数据量和划分；验证好而部署差，要看验证分布是否代表真实场景，以及公平、隐私等约束。这与实践一章的偏差-方差诊断是一致的：第一轴大致对应高偏差或优化问题，第二轴对应高方差，第三轴对应泄漏与分布偏移。

> [!NOTE]+ 三条失败轴
>
> - 拟合训练失败：假设类不够、优化发散、实现 shape 错误、不可辨识。
> - 拟合验证失败：正则过弱或过强、数据量不够、标注噪声、用测试调参。
> - 拟合部署失败：划分泄漏、验证分布不等于现实、公平约束被平均指标掩盖。
> - 课堂扩展主要落在部署轴：评估协议、分组指标、隐私与项目节奏。

## 各主题一句话回顾

下面按课程顺序列出每个主题最需要记住的结论：

- **回归与分类**：线性回归的正规方程；LMS 与逻辑回归的梯度都具有“误差 × 特征”的形式，误差为 $h-y$，逻辑回归中 $h=g(\theta^{\top}x)$，$g$ 为 sigmoid；牛顿法用的 Hessian 为 $X^{\top}RX$（负对数似然的 Hessian），其中 $R=\mathrm{diag}\big(h^{(i)}(1-h^{(i)})\big)$。
- **GLM**：自然参数与输入线性相关，$\eta=\theta^{\top}x$；假设输出为期望 $g(\eta)=E[T(y)]$，$g$ 为正则响应函数（canonical response function）。
- **生成学习**：GDA 在两类共享协方差时决策边界是线性的；朴素贝叶斯用 Laplace 平滑避免零概率。
- **核与 SVM**：Mercer 定理给出合法核的条件（核矩阵对称半正定）；最优权重可写成 $w=\sum\alpha_i y^{(i)}\phi(x^{(i)})$，因而只需内积；软间隔下 $0\le\alpha_i\le C$。
- **树与提升**：决策树按不纯度（熵、Gini）下降选择分裂；AdaBoost 等价于对指数损失做前向分阶段（forward stagewise）加性建模。
- **神经网络**：反向传播就是把每层的局部 Jacobian 依次相乘；sigmoid / softmax 输出配交叉熵时，输出层梯度简化为 $\hat y-y$。
- **泛化**：偏差-方差分解；有限假设类的一致收敛界中出现 $\log|\mathcal{H}|/n$ 这一项（泛化间隙约为 $\sqrt{\log|\mathcal{H}|/n}$ 量级，$n$ 为样本数）；无限假设类用 VC 维 $d$ 大致替代 $\log|\mathcal{H}|$（界中还带有对数因子）。
- **无监督**：k-means 的 Lloyd 迭代（分配—更新交替）；EM 交替最大化 ELBO；PCA 求协方差特征方程 $\Sigma u=\lambda u$；因子分析 $x=\mu+\Lambda z+\varepsilon$；ICA 依赖源信号的非高斯性才能辨识（高斯源存在旋转歧义）。
- **生成与语言**：扩散模型训练网络做 $\varepsilon$-预测（预测加入的噪声）；对比学习用 InfoNCE 损失；Transformer 注意力为 $\mathrm{softmax}(QK^{\top}/\sqrt{d_k})$ 再乘 $V$。
- **控制**：MDP 的贝尔曼方程；LQR 中二次值函数矩阵 $P_t$ 满足 Riccati 递推；策略梯度 $\nabla J=E[\nabla\log\pi\,A]$，$A$ 为优势函数。

## 考试与作业的侧重点

期末闭卷部分通常覆盖课程前半：LMS 与正规方程、逻辑回归梯度与牛顿法、GLM、GDA/NB、核与间隔、偏差-方差、EM 的责任度（responsibility，即后验 $p(z\mid x)$）。编程作业侧重实现细节：矩阵 shape、数值稳定的 softmax（先减去最大值再取指数）、EM 中的下溢（在对数域计算并用 log-sum-exp）。项目考察的是能否把上述工具用在未布置过的数据上。

## 默写清单

下面每条公式后附一句说明，便于核对含义：

- **最小二乘**：$J(\theta)=\frac12\|X\theta-y\|_{2}^{2}$，$\theta=(X^{\top}X)^{-1}X^{\top}y$。这里 $X\in\mathbb{R}^{n\times d}$ 每行一个样本。令梯度 $X^{\top}(X\theta-y)=0$ 即得正规方程 $X^{\top}X\theta=X^{\top}y$。实现时应解线性方程组（`np.linalg.solve` 或 `lstsq`），不要显式求逆。
- **逻辑回归**：$\nabla_{\theta}\ell_{\mathrm{logistic}}=X^{\top}(y-h)$，$H=-X^{\top}RX$。这里 $\ell$ 是对数似然（要最大化），$h$ 是各样本预测概率组成的向量，$R=\mathrm{diag}(h_i(1-h_i))$。$H$ 负定说明对数似然是凹的，牛顿更新为 $\theta\leftarrow\theta-H^{-1}\nabla_\theta\ell$。
- **Bernoulli 写成指数族**：$\eta=\log\frac{\phi}{1-\phi}$，反过来 $\phi=\sigma(\eta)$。这正是逻辑回归使用 sigmoid 的来源。
- **GDA**：共享 $\Sigma$ 时，两类对数后验比中的二次项相消，边界是线性的；各类用各自的 $\Sigma_k$ 时（QDA）边界是二次的。
- **Laplace 平滑**：分子 $+1$，分母 $+|V|$。这是多项式事件模型的写法，$|V|$ 为词表大小（一般地，分母加的是取值个数；伯努利事件模型中每个特征只有两种取值，分母加 2）。
- **间隔**：几何间隔 $\gamma=\hat\gamma/\|w\|$，$\hat\gamma$ 为函数间隔；硬间隔 SVM 为 $\min\|w\|^{2}/2$ s.t. $y(w^{\top}x+b)\ge 1$（把函数间隔固定为 1 后，最大化几何间隔等价于最小化 $\|w\|$）。
- **SVM 对偶**：$\max\sum\alpha_i-\frac12\sum\alpha_i\alpha_j y_i y_j K_{ij}$，约束为 $\alpha_i\ge0$（软间隔为 $0\le\alpha_i\le C$）与 $\sum_i\alpha_i y_i=0$。只有支持向量的 $\alpha_i>0$。
- **EM**：ELBO $=\sum Q\log\frac{p(x,z)}{Q}$；E 步取 $Q=p(z\mid x)$，此时由 Jensen 不等式得到的下界取等号；M 步对参数最大化 ELBO。
- **PCA**：$\Sigma u=\lambda u$，取最大的几个特征值对应的特征向量；等价地，对中心化后的数据矩阵做 SVD，取右奇异向量。
- **值迭代**：$V(s)\leftarrow\max_a\sum_{s'}P[R+\gamma V(s')]$，其中 $P=P_{sa}(s')$ 为转移概率，$\gamma$ 为折扣因子。
- **REINFORCE**：$\sum\nabla\log\pi(G_t-b(s_t))$，$G_t$ 为从 $t$ 起的回报，减去基线 $b(s_t)$ 不改变期望但降低方差。
- **PPO**：概率比 $r_t=\pi/\pi_{\mathrm{old}}$，把它裁剪到 $1\pm\varepsilon$ 范围内，限制每次策略更新的幅度。

## 常考的实现错误与策略题

闭卷实现题常考以下错误：$X$ 的行列约定搞反（样本在行还是在列）；对 $X^{\top}X$ 直接求逆（病态时数值不稳定，应解方程组或加岭）；EM 未加岭（协方差矩阵奇异）；Transformer 缺少因果掩码（模型偷看未来 token）；把测试集当成验证集使用。

策略题会给出一张按组分类的错误表，要求指出下一步实验及其指标上限。思路即实践一章的误差分析：某类错误占总错误的比例，就是彻底解决该类错误后整体误差最多能下降的幅度。

<br>

# 参考文献

## 课程

1. Stanford CS229. *Machine Learning* 课程主页。[https://cs229.stanford.edu/](https://cs229.stanford.edu/)
2. Stanford CS229. *Fall 2025* 课表。[https://cs229.stanford.edu/index.html-fall25](https://cs229.stanford.edu/index.html-fall25)
3. Stanford Bulletin. *CS229: Machine Learning*. [https://bulletin.stanford.edu/courses/1057501](https://bulletin.stanford.edu/courses/1057501)
4. Ma T, Ng A. *CS229 Lecture Notes*. 2026-08-23（Spring 2026 主讲义）。[PDF](https://cs229.stanford.edu/main_notes.pdf)
5. Ng A. *Advice for applying Machine Learning*（实践一章诊断方法的来源）。[PDF](https://cs229.stanford.edu/materials/ML-advice.pdf)

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

# 工具与软件

完成作业与项目所需的工具不多。原则是：作业中的核心算法循环（梯度下降、EM、值迭代等）要自己用 NumPy 写，现成库只用来对照结果、加速非核心部分，以及在项目中做大规模实验。

## 数值计算

- [NumPy](https://numpy.org/)：用于 shape 对齐、向量化与从零实现。CS229 约定样本在第 $0$ 维，即数据矩阵每行一个样本。
- [SciPy](https://scipy.org/)：提供 `linalg.cho_solve`（用 Cholesky 分解解对称正定方程组，比显式求逆更稳定）、稀疏核矩阵与优化器。
- [scikit-learn](https://scikit-learn.org/)：可用来对照正规方程、SVM、PCA、GMM 的结果；但作业仍要求自己写核心循环。

## 深度学习与检索

- [PyTorch](https://pytorch.org/)：提供自动微分；张量默认 batch-first（第 $0$ 维为批）。
- [Hugging Face Transformers / tokenizers](https://huggingface.co/)：分词器必须与模型检查点绑定使用，混用会导致 token 编号错位。
- FAISS / 其他近似最近邻（ANN）库：做检索时应同时记录召回与延迟，两者之间存在权衡。

## 控制与评估

- 简单的网格世界环境自己写即可；连续控制问题再考虑标准基准环境。
- 分组指标与校准曲线应写进验证脚本，随每次实验自动生成，而不是事后用表格软件手算。

## 线性代数与概率复习材料

- CS229 课程页提供 Linear Algebra Review、Probability Review 与 Python/NumPy 讲义。
- 作业 Problem Set 0 用于在正式学习算法前，对齐矩阵微积分与广播（broadcasting）的约定。

<br>
