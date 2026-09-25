---
title: 斯坦福CS230：Deep Learning
date: 2026-09-22
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS230 详细课程笔记：从逻辑回归与反向传播，到正则化、优化、CNN、序列模型、注意力与 Transformer。
hidden: true
---


- **全名**
  - *Deep Learning*。
- **主讲**
  - Andrew Ng；
  - Kian Katanforoosh。
- **形式**
  - 翻转课堂（flipped classroom）；
  - 课前：deeplearning.ai Deep Learning Specialization 的视频、测验与编程作业；
  - 课上：项目周期、对抗稳健性（adversarial robustness）、生成模型（generative model）、深度强化学习（deep reinforcement learning）、LLM 应用、可解释性（interpretability）；
  - 期末为开放项目。
- **入口**
  - [Stanford Bulletin：CS230](https://bulletin.stanford.edu/courses/2198851)
  - [课程主页](https://cs230.stanford.edu/)
  - [Syllabus](https://cs230.stanford.edu/syllabus/)



<br>


# 总览

- **表示与浅层模型**
  - 监督学习（supervised learning）记号、向量化（vectorization）与广播（broadcasting）；
  - 逻辑回归（logistic regression）作为单层网络；
  - 浅层网络、激活函数（activation function）与决策边界（decision boundary）。
- **深层网络与训练**
  - $L$ 层前向与反向传播（forward and backpropagation）；
  - 初始化、梯度检查（gradient checking）、消失与爆炸梯度（vanishing and exploding gradients）；
  - 正则化（regularization）、Dropout、BatchNorm；
  - mini-batch、Momentum、RMSprop、Adam 与学习率衰减（learning rate decay）。
- **项目策略**
  - 正交化（orthogonalization）、单一评估指标与人类水平（human-level performance）；
  - 误差分析（error analysis）、分布不匹配（data mismatch）、迁移学习（transfer learning）与多任务（multitask learning）；
  - 端到端（end-to-end）与数据流水线。
- **卷积与视觉**
  - 卷积（convolution）、池化（pooling）、CNN 维度计算；
  - ResNet、Inception、MobileNet 与迁移学习；
  - YOLO 检测、U-Net 分割、人脸验证（face verification）与神经风格迁移（neural style transfer）。
- **序列与语言**
  - RNN、GRU、LSTM；
  - 词嵌入（word embeddings）、seq2seq 与注意力（attention）；
  - Transformer、位置编码（positional encoding）与多头注意力（multi-head attention）。
- **课堂扩展**
  - 对抗样本（adversarial examples）与生成模型；
  - 深度强化学习；
  - LLM 应用层；
  - 期末项目的数据、指标与迭代。

- **记录逻辑**

$$
\text{任务与数据}
\rightarrow\text{表示与假设}
\rightarrow\text{损失与约束}
\rightarrow\text{前向计算}
\rightarrow\text{反向与更新}
\rightarrow\text{泛化诊断}
$$

- **分数校准**
  - 训练损失下降不等于 dev 上的目标指标改善；
  - 准确率在类别不平衡时会掩盖少数类错误；
  - BLEU、IoU、mAP、EER 各自对应不同的错误代价。
- **实现要求**
  - 能够从零实现逻辑回归、$L$ 层全连接（fully connected）网络、卷积前向、RNN/LSTM 单元与 scaled dot-product attention；
  - 能够说明成熟框架中优化器、正则项、BatchNorm 运行统计与 teacher forcing 的实际语义。
- **主线**
    - **前半：实现与梯度**
    - 回答前向缓存什么、反向需要什么、shape 如何对齐。
    - **后半：误差归因**
        - 回答偏差、方差、数据不匹配（data mismatch）、优化失败与标注噪声各是什么表现；
        - 检测框的定位误差与分类误差、翻译的覆盖不足与重复解码，分属不同修复路径。

$$
\text{样本 }(x,y)
\rightarrow\text{线性变换 }z=Wx+b
\rightarrow\text{非线性 }a=g(z)
\rightarrow\text{堆叠、共享权重与注意力}
\rightarrow\text{任务损失与部署约束}
$$



$$
\begin{aligned}
\text{向量化逻辑回归}
&\rightarrow\text{浅层网络}
\rightarrow\text{深层反向传播（backpropagation）}
\rightarrow\text{正则与优化},\\
\text{全连接层}
&\rightarrow\text{卷积与权值共享（weight sharing）}
\rightarrow\text{检测/分割头},\\
\text{RNN 隐状态（hidden state）}
&\rightarrow\text{注意力}
\rightarrow\text{自注意力（self-attention）与 Transformer}.
\end{aligned}
$$


<br>



## 数学预备

课程反复使用五类数学对象。

$$
J(W,b)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}\big(\hat{y}^{(i)},y^{(i)}\big)
$$

- **经验风险（empirical risk）：把数据集上的平均损失当作可微目标**
  - 单样本损失 $\mathcal{L}$ 描述预测与标签的局部代价；
  - 参数更新针对 $J$，不是针对某一个样本的瞬时损失。
- **计算图（computational graph）：把复合函数拆成局部导数的乘积**
  - 前向保存中间量，反向沿边回传 $\partial J/\partial(\cdot)$；
  - 链式法则不要求一次写出总公式，只要求每个节点知道局部 Jacobian。
- **期望与 mini-batch 估计**
  - 全批量梯度是 $J$ 的精确梯度；
  - mini-batch 用子集估计同一期望，引入噪声但提高吞吐；
  - Adam 等自适应方法跟踪梯度的一阶与二阶矩。
- **线性代数与广播**
  - 一层仿射变换（affine transformation）是矩阵-矩阵乘法，不是 Python 层循环；
  - 偏置 $b\in\mathbb{R}^{n^{[l]}\times 1}$ 沿 batch 维广播；
  - shape 错误会产生数值合理但语义错误的结果。
- **概率与信息量**
  - sigmoid 输出解释为 Bernoulli 参数；
  - softmax 输出解释为类别分布；
  - 交叉熵（cross-entropy）是真实分布与模型分布之间的编码长度差。

## 复杂度

- **时间：先问哪一个维度增长**
  - 全连接 $n^{[l-1]}\rightarrow n^{[l]}$、batch 大小 $m$：矩阵乘 $O(m n^{[l]} n^{[l-1]})$；
  - 深度 $L$：前向与反向同阶，反向常数更大，因为要缓存激活；
  - 卷积：输入 $H\times W$，通道 $C_{\mathrm{in}}\rightarrow C_{\mathrm{out}}$，核 $f\times f$，约 $O(m H W C_{\mathrm{out}} C_{\mathrm{in}} f^{2})$；
  - 自注意力：长度 $T$、宽度 $d$，为 $O(T^{2}d)$，这是长序列的主要瓶颈。
- **空间**
  - 反向传播必须保存各层 $A^{[l]}$，以及 BatchNorm 的均值方差、Dropout 的掩码；
  - 激活往往比权重更占显存；
  - batch 从 32 增到 256，显存近似线性上升，参数量不变。

## 数值计算

- 概率与指数在未归一化 logits 上计算，不是先 `exp` 再求和。
- softmax 应减去每行最大值。
- sigmoid 在大正负输入上应改写为稳定形式。
- 比较 log-likelihood 前，说明是否按序列长度或像素数归一化。
- 梯度检查：双精度；关闭 Dropout 与非确定性增强。
- 混合精度（mixed precision）需要 loss scaling，否则小梯度会下溢为 0。

## Python 实现基础

- **角色**
  - NumPy 用来验证算法：先在小数组上对齐 shape 与梯度，再交给 TensorFlow 或 PyTorch；
  - 数值内核优先 broadcasting 与线性代数，避免在样本维或通道维上写 Python `for` loop；
  - 向量化不改变渐近复杂度，只是把循环移到底层 BLAS。
- **存贮约定**
  - CS230 作业：$X$ 的 shape 是 $(n_x, m)$（特征 × 样本）；
  - 机器学习社区更常见 $(m, n_x)$；
  - PyTorch 默认 NCHW 或框架相关的 batch-first，迁移代码必须显式转置。
- **索引**
  - 整数索引消去一个维度；切片通常保留该维度；
  - `A[0, :]` 的 shape 是 `(m,)`；`A[0:1, :]` 的 shape 是 `(1, m)`；
  - 把前者与 `(n, m)` 矩阵相乘会触发意外广播。
- **数值与随机**
  - `/` 总是浮点除法；
  - 随机性来自 `numpy.random` 或框架 generator；
  - 复现实验必须固定 seed，并记录是否启用了非确定卷积算法。
- **参数 vs 运行统计**
  - BatchNorm 的 $\gamma,\beta$ 是参数；
  - moving mean/variance 是评估期使用的估计。
- **广播**
  - 从尾维对齐；维长相等、其中一方为 1、或该维不存在时可以扩展；
  - `(n, 1)` 与 `(n, m)` 相加：偏置复制到每一列；
  - `(n,)` 与 `(n, m)` 相加会失败或得到错误语义，因为 `(n,)` 被当成最后一维长度为 $n$；
  - 作业里所有偏置写成 `(n^{[l]}, 1)`，并用 `keepdims=True`。

```python
def assert_col_major(X, name="X"):
    """CS230 约定：二维数组必须是 (features, m)。"""
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {X.shape}")

def vectorized_vs_loop(X, w, b):
    """对比逐样本循环与一次矩阵乘，数值应在 1e-10 内一致。"""
    n_x, m = X.shape
    z_loop = np.empty((1, m))
    for i in range(m):
        z_loop[0, i] = float(w.T @ X[:, i:i + 1] + b)
    z_vec = w.T @ X + b
    return z_vec, float(np.max(np.abs(z_vec - z_loop)))
```

- **计算图**
  - 节点：$z=w^{\top}x+b$，$a=\sigma(z)$，$\mathcal{L}(a,y)$；
  - 前向沿拓扑序求值并缓存 $x,z,a$；
  - 反向从 $\mathrm{d}\mathcal{L}$ 出发，每个节点只用局部导数；
  - 与自动微分的 reverse-mode 相同：标量损失、高维参数时，反向模式只需一次扫描。
- **图像预处理**
  - 作业样本常为 $64\times 64\times 3$；
  - 展开顺序必须固定，例如 `x.reshape((n_x, 1), order="C")`；
  - 否则同一张图会得到不同特征向量，训练与推理预处理不一致；
  - 像素先除以 255 映射到 $[0,1]$，再视情况减训练集均值。

<br>

# 记号

## 概念记号

- **问题**
  - 监督学习把输入 $x$ 映射到标签 $y$；
  - 深度网络把该映射参数化为一串仿射变换与逐点非线性；
  - 记号不统一时，反向传播公式无法逐层核对。
- **范围**
  - 固定 CS230 / Andrew Ng 讲义中的上下标约定；
  - 输入是原始特征或张量，输出是预测 $\hat{y}$ 与标量损失 $J$。
- **章节衔接**
  - 上一节的广播与列存约定，在本章统一为同一套上下标；
  - 记号回答“哪个矩阵乘哪个激活”；
  - 下一章的逻辑回归是该记号下最简单的可训练模型。

上标圆括号表示样本，方括号表示层，尖括号表示时间步：

$$
x^{(i)},\qquad
W^{[l]},\qquad
a^{<t>}.
$$

- $m$：训练样本数。
- $n_x=n^{[0]}$：输入特征维。
- $n^{[l]}$：第 $l$ 层神经元数。
- $L$：网络层数，输出层为第 $L$ 层。
- $X\in\mathbb{R}^{n_x\times m}$：列向量样本拼成的设计矩阵。
- $Y\in\mathbb{R}^{n_y\times m}$：标签矩阵；二分类时 $n_y=1$。
- $W^{[l]}\in\mathbb{R}^{n^{[l]}\times n^{[l-1]}}$，$b^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}$。
- $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$，$A^{[l]}=g^{[l]}(Z^{[l]})$，$A^{[0]}=X$。
- $\hat{Y}=A^{[L]}$。

一个样本的前向：

$$
z^{[l](i)}=W^{[l]}a^{[l-1](i)}+b^{[l]},\qquad
a^{[l](i)}=g^{[l]}\big(z^{[l](i)}\big).
$$

整批样本一次计算：

$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}.
$$

- $b^{[l]}$ 沿列广播。
- 实现注释中写明每个数组的 shape。
- 若 $W^{[l]}$ 被误写成 $(n^{[l-1]}, n^{[l]})$，后续所有梯度公式都会差一个转置。

逐层 shape 表（二分类，$n_y=1$）：

| 量 | shape |
| :--- | :--- |
| $X=A^{[0]}$ | $(n^{[0]}, m)$ |
| $W^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ |
| $b^{[l]}$ | $(n^{[l]}, 1)$ |
| $Z^{[l]}, A^{[l]}$ | $(n^{[l]}, m)$ |
| $\mathrm{d}Z^{[l]}, \mathrm{d}A^{[l]}$ | 与 $Z^{[l]}$ 相同 |
| $\mathrm{d}W^{[l]}, \mathrm{d}b^{[l]}$ | 与对应参数相同 |

- **命名**
  - `dW` 表示 $\partial J/\partial W$，不是 $W$ 的微分形式；
  - 实现时用 `assert dW.shape == W.shape` 卡住转置错误。
- **额外维度**
  - 时间序列另加一维 $T$；
  - 图像另加 $H,W,C$；
  - 全连接作业：batch 维在最后一维；
  - 卷积作业：batch 维在第一维 `(m, n_H, n_W, n_C)`（NHWC）；
  - 两套约定不能混用。
- **常用恒等式**

$$
\frac{\partial}{\partial W}(WA)=A^{\top}\quad\text{（左侧乘 $W$）},\qquad
\frac{\partial}{\partial A}(WA)=W^{\top}.
$$

- 外积 $\mathrm{d}W=\mathrm{d}Z A^{\top}/m$：把 $m$ 个样本的 $(n^{[l]},1)\times(1,n^{[l-1]})$ 加起来再平均。

> [!NOTE]+ 衔接
>
> 记号固定 $X$ 为 $(n_x,m)$、$W$ 为 $(n^{[l]},n^{[l-1]})$。下一章把 $L=1$、$g=\sigma$ 写成可训练的逻辑回归，并导出 $A-Y$。

<br>

# 逻辑回归

## 概念：逻辑回归

- **问题**
  - 二分类：$y\in\{0,1\}$，输入 $x\in\mathbb{R}^{n_x}$；
  - 模型输出 $\hat{y}=\sigma(w^{\top}x+b)\in(0,1)$，解释为 $P(y=1\mid x)$。
- **范围**
  - 由最大似然（maximum likelihood）导出二元交叉熵（binary cross-entropy），再写出向量化梯度；
  - 输入 $(X,Y)$，输出参数 $(w,b)$ 与预测。
- **章节衔接**
  - 上一章的 $Z=WA+b$ 在这里取 $L=1$；
  - 逻辑回归是没有隐层（hidden layer）的神经网络；
  - 下一章只在其前面插入一层非线性。

## 模型

$$
z=w^{\top}x+b,\qquad
\hat{y}=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

- $\sigma$ 把实数压到 $(0,1)$。
- $\sigma(z)+\sigma(-z)=1$。
- $\sigma'(z)=\sigma(z)(1-\sigma(z))$。
- $|z|$ 很大时 $\sigma(z)$ 饱和，梯度接近 0；这是后续要用更好初始化与归一化的原因之一。

Bernoulli 似然：

$$
P(y\mid x)=\hat{y}^{y}(1-\hat{y})^{1-y}.
$$

最大化对数似然等价于最小化二元交叉熵：

$$
\mathcal{L}(\hat{y},y)=-y\log\hat{y}-(1-y)\log(1-\hat{y}).
$$

- $y=1$ 时只惩罚 $-\log\hat{y}$。
- $y=0$ 时只惩罚 $-\log(1-\hat{y})$。
- 把 $\hat{y}$ 截断到 $[\varepsilon,1-\varepsilon]$，避免 $\log 0$。

数据集目标：

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}\big(\hat{y}^{(i)},y^{(i)}\big).
$$

- $1/m$ 使损失与样本数无关，换 batch 大小时学习率更可比较。
- 正则项是否同样除以 $m$，必须与梯度公式一致。
- 独立同分布样本的对数似然：$\sum_i [y^{(i)}\log a^{(i)}+(1-y^{(i)})\log(1-a^{(i)})]$。
- 最大化该式等价于最小化 $J$。
- 判别模型：直接建 $P(y\mid x)$，不建 $P(x)$。

> [!INFO]+ 为何不用 MSE 做分类
>
> 若 $\mathcal{L}=(a-y)^{2}/2$ 且 $a=\sigma(z)$，则 $\partial\mathcal{L}/\partial z=(a-y)\,a(1-a)$。当预测极错、$a\rightarrow 0$ 但 $y=1$ 时，$a(1-a)\rightarrow 0$，梯度消失。交叉熵在同一极限下 $\partial\mathcal{L}/\partial z=a-y\rightarrow -1$，仍有强信号。MSE 对 sigmoid 输出不是凸的；二元交叉熵对 $(w,b)$ 是凸的。

<br>

> [!INFO]+ 从损失到 $dz$
>
> 令 $a=\sigma(z)$。对单个样本：
>
> $$
> \frac{\partial\mathcal{L}}{\partial a}=-\frac{y}{a}+\frac{1-y}{1-a},\qquad
> \frac{\partial a}{\partial z}=a(1-a).
> $$
>
> 相乘后：
>
> $$
> \frac{\partial\mathcal{L}}{\partial z}=\left(-\frac{y}{a}+\frac{1-y}{1-a}\right)a(1-a)=a-y.
> $$
>
> 因此 sigmoid + 二元交叉熵的输出层误差是 $a-y$，不需要再显式乘 $\sigma'(z)$。多分类中 softmax + 交叉熵有同一简化：$\partial\mathcal{L}/\partial z = a-y$。

<br>

## 梯度

单样本：

$$
\frac{\partial\mathcal{L}}{\partial w}=(a-y)x,\qquad
\frac{\partial\mathcal{L}}{\partial b}=a-y.
$$

- 来源：$z=w^{\top}x+b$，故 $\partial z/\partial w=x$，$\partial z/\partial b=1$。
- 再乘 $\partial\mathcal{L}/\partial z=a-y$。

- 整批向量化，令 $A=\sigma(w^{\top}X+b)\in\mathbb{R}^{1\times m}$：

$$
\mathrm{d}w=\frac{1}{m}X(A-Y)^{\top},\qquad
\mathrm{d}b=\frac{1}{m}\sum_{i=1}^{m}(A-Y)_{i}.
$$

- 循环写法：对每个 $i$ 执行 $\mathrm{d}w \mathrel{+}= (a^{(i)}-y^{(i)})x^{(i)}$，再除以 $m$。
- $X$ 为 $(n_x,m)$，$(A-Y)^{\top}$ 为 $(m,1)$，乘积为 $(n_x,1)$。
- 误用 `X @ (A-Y)` 且不转置，会因广播得到错误 shape。

$\mathrm{d}w$ 与 $w$ 同形，均为 $(n_x,1)$。学习率 $\alpha$ 的梯度下降（gradient descent）：

$$
w\leftarrow w-\alpha\,\mathrm{d}w,\qquad
b\leftarrow b-\alpha\,\mathrm{d}b.
$$

- $\alpha$ 过大：$J$ 震荡或出现 `nan`。
- $\alpha$ 过小：数百步几乎不动。
- 猫分类作业上 $\alpha=0.005$、$2000$ 步是常见能下降的起点，不是普遍最优。

## 算法

1. 预处理：图像 `(64, 64, 3)` 展成 `(12288, 1)`，像素除以 $255$，按列拼成 $X$。训练与推理必须使用同一 `reshape` 顺序。
2. 初始化 $w=0$、$b=0$。逻辑回归没有隐层对称性问题，零初始化合法。
3. 重复：计算 $A=\sigma(w^{\top}X+b)$；计算 $J$；计算 $\mathrm{d}w,\mathrm{d}b$；更新参数。
4. 每隔固定步记录 $J$。若 $J$ 不降，检查 shape、学习率与标签是否为 $\{0,1\}$ 而非 $\{-1,1\}$。
5. 预测时用阈值 $0.5$，或按假阳性代价调整。准确率不是训练目标；移动阈值会改准确率但不改 $J$。

- 每次迭代时间 $O(m n_x)$。
- 逻辑回归的 $J$ 对 $(w,b)$ 是凸的，梯度下降可到全局最优。
- 一旦加入隐层，凸性消失。
- 特征尺度相差几个数量级时，等高线被拉成长谷。
- 只用训练集统计做标准化；测试集复用同一 $\mu,\sigma$。

## 实现

```python
import numpy as np

def sigmoid(z):
    """稳定 sigmoid；大正负输入分别改写，避免 overflow。"""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out

def logistic_loss(a, y, eps=1e-12):
    """二元交叉熵，a 与 y 的 shape 均为 (1, m)。"""
    a = np.clip(a, eps, 1.0 - eps)
    m = y.shape[1]
    return float(-(y * np.log(a) + (1.0 - y) * np.log(1.0 - a)).sum() / m)

def logistic_train(X, Y, lr=0.01, num_steps=2000):
    """X: (n_x, m), Y: (1, m)。返回 w, b, 损失轨迹。"""
    n_x, m = X.shape
    w = np.zeros((n_x, 1))
    b = 0.0
    history = []
    for _ in range(num_steps):
        A = sigmoid(w.T @ X + b)
        history.append(logistic_loss(A, Y))
        dZ = A - Y
        dw = (X @ dZ.T) / m
        db = float(dZ.sum() / m)
        w -= lr * dw
        b -= lr * db
    return w, b, history

def logistic_predict(X, w, b, threshold=0.5):
    return (sigmoid(w.T @ X + b) >= threshold).astype(int)
```

> [!EXAMPLE]+ 例1 决策边界
>
> - 二维输入 $x=(x_1,x_2)$：决策边界是 $w_1 x_1+w_2 x_2+b=0$ 的直线。
> - 正类线性可分时，训练后法向量 $w$ 指向正类一侧。
> - 异或四点：逻辑回归损失有下界，无法同时分开。
> - 这是引入隐层的动机。

<br>

> [!NOTE]+ 衔接
>
> 逻辑回归给出凸的 $J$ 与 $A-Y$。下一章在 $x$ 与输出之间插入 $g^{[1]}$，决策边界才能弯曲。没有非线性时，两层仿射仍等价于一层。

<br>

# 浅层网络

## 概念：浅层网络

- **问题**
  - 一层隐层加一层输出，即 $L=2$；
  - 隐层激活常用 tanh 或 ReLU，输出层二分类用 sigmoid、多分类用 softmax。
- **范围**
  - 写清两层前向与反向，并理解非线性如何折叠空间；
  - 输入 $X$，输出 $A^{[2]}$。
- **章节衔接**
  - 上一章是 $L=1$ 的特例；
  - 本章把反向传播模式写成两层，下一章推广到任意 $L$；
  - 没有非线性时，两层仿射变换等价于一层，深度无意义。

## 前向

$$
\begin{aligned}
Z^{[1]} &= W^{[1]}X + b^{[1]}, &
A^{[1]} &= g^{[1]}(Z^{[1]}),\\
Z^{[2]} &= W^{[2]}A^{[1]} + b^{[2]}, &
A^{[2]} &= g^{[2]}(Z^{[2]}).
\end{aligned}
$$

- $W^{[1]}$：$(n^{[1]}, n_x)$
- $A^{[1]}$：$(n^{[1]}, m)$
- $W^{[2]}$：$(n_y, n^{[1]})$
- $A^{[2]}$：$(n_y, m)$

常用 $g^{[1]}$：

- **sigmoid**：输出有界，深层易饱和。
- **tanh**：零中心，浅层隐层通常优于 sigmoid。
- **ReLU**：$g(z)=\max(0,z)$，$g'(z)=\mathbf{1}_{z>0}$。计算便宜，负半轴梯度为 0，出现 dying ReLU。
- **Leaky ReLU**：$g(z)=\max(\alpha z, z)$，$\alpha$ 如 $0.01$，减轻死亡神经元。

> [!INFO]+ ReLU
>
> - 是什么：整流线性，$g(z)=\max(0,z)$，正半轴导数为 1。
> - 来自：Nair & Hinton (2010)；AlexNet 之后成为卷积网络的默认隐层激活。
> - 相对 sigmoid / tanh：深层不再把梯度乘进 $0.1$ 量级的饱和导数。
> - 代价：负半轴永久为零（dying ReLU）；Leaky ReLU 用小斜率补这条通路。

<br>

- **通用逼近**
  - 足够宽的单隐层网络可逼近紧集上的连续函数；
  - 所需宽度可能指数级；
  - 训练不保证找到该逼近；
  - 深度用组合的方式复用特征。
- **参数量**：$n^{[1]}(n_x+1)+n_y(n^{[1]}+1)$。
  - 平面分类作业常用 $n_x=2$、$n^{[1]}=4$、$n_y=1$，共 $4\cdot 3+1\cdot 5=17$ 个参数；
  - 隐单元太少：无法拼出弯曲边界；
  - 隐单元太多而数据只有几百点：过拟合。
- **非线性不可省略**
  - 若 $g^{[1]}$ 为恒等，则

$$
A^{[2]}=g^{[2]}\big(W^{[2]}W^{[1]}X + W^{[2]}b^{[1]}+b^{[2]}\big),
$$

  - 等价于单层，决策边界仍是直线；
  - 异或四个点需要把空间折叠，至少要有弯曲的隐层响应。

> [!EXAMPLE]+ 例1b 激活导数在实现中的取值
>
> $z=2$ 时，$\sigma(2)\approx 0.88$，$\sigma'(2)\approx 0.105$。tanh$(2)\approx 0.96$，导数 $1-0.96^{2}\approx 0.07$。ReLU$(2)=2$，导数 $1$。深层中若每层都乘 $0.1$ 量级的饱和导数，十层后梯度衰减约 $10^{-10}$。这是隐层改用 ReLU、输出层才用 sigmoid/softmax 的原因。

<br>


## 反向

- 二分类取 $g^{[2]}=\sigma$，则 $\mathrm{d}Z^{[2]}=A^{[2]}-Y$。

$$
\begin{aligned}
\mathrm{d}W^{[2]} &= \frac{1}{m}\mathrm{d}Z^{[2]}A^{[1]\top},&
\mathrm{d}b^{[2]} &= \frac{1}{m}\sum_{i}\mathrm{d}Z^{[2](:,i)},\\
\mathrm{d}A^{[1]} &= W^{[2]\top}\mathrm{d}Z^{[2]},&
\mathrm{d}Z^{[1]} &= \mathrm{d}A^{[1]}\odot g^{[1]\prime}(Z^{[1]}),\\
\mathrm{d}W^{[1]} &= \frac{1}{m}\mathrm{d}Z^{[1]}X^{\top},&
\mathrm{d}b^{[1]} &= \frac{1}{m}\sum_{i}\mathrm{d}Z^{[1](:,i)}.
\end{aligned}
$$

- $\odot$ 是逐元素乘。
- $g=\tanh$ 时 $g'(z)=1-\tanh^{2}(z)=1-(a)^{2}$，用前向已算的 $A^{[1]}$ 即可，不必另存 $Z^{[1]}$。
- ReLU 必须知道 $Z^{[1]}$ 的符号。

> [!INFO]+ 两层网络的链式法则
>
> $J$ 对 $W^{[2]}$ 的依赖路径是 $W^{[2]}\rightarrow Z^{[2]}\rightarrow A^{[2]}\rightarrow J$。矩阵微积分给出 $\mathrm{d}W^{[2]}=\mathrm{d}Z^{[2]}A^{[1]\top}/m$，因为 $Z^{[2]}$ 的第 $k$ 行来自 $W^{[2]}$ 第 $k$ 行与 $A^{[1]}$ 的内积。对 $A^{[1]}$，$W^{[2]\top}$ 把输出误差投回隐层坐标。再乘局部激活导数，才得到 $\mathrm{d}Z^{[1]}$。漏乘 $g'$ 是实现中最常见的错误之一。

<br>

## 实现

```python
def tanh_forward(Z):
    A = np.tanh(Z)
    return A, Z

def tanh_backward(dA, cache):
    Z = cache
    A = np.tanh(Z)
    return dA * (1.0 - A ** 2)

def relu_forward(Z):
    A = np.maximum(0.0, Z)
    return A, Z

def relu_backward(dA, cache):
    Z = cache
    dZ = dA.copy()
    dZ[Z <= 0] = 0.0
    return dZ

def two_layer_forward(X, params):
    """params 含 W1,b1,W2,b2。缓存反向所需激活。"""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = W1 @ X + b1
    A1, cache1 = relu_forward(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    return A2, (X, A1, cache1, Z2)

def two_layer_backward(A2, Y, caches, params):
    X, A1, cache1, Z2 = caches
    m = X.shape[1]
    W2 = params["W2"]
    dZ2 = A2 - Y
    dW2 = (dZ2 @ A1.T) / m
    db2 = dZ2.sum(axis=1, keepdims=True) / m
    dA1 = W2.T @ dZ2
    dZ1 = relu_backward(dA1, cache1)
    dW1 = (dZ1 @ X.T) / m
    db1 = dZ1.sum(axis=1, keepdims=True) / m
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
```

- **初始化**
  - 不能全零：若 $W^{[1]}=W^{[2]}=0$，隐单元对称，梯度相同，永远无法分裂特征；
  - 常用小高斯数，或按后文 Xavier/He 缩放；
  - $b$ 可零初始化。
- **训练循环**
  - 四步：前向、损失、反向、更新；
  - 更新必须用当前步的梯度，不能把所有层的梯度算完后再用过期的激活；
  - 学习率对浅层同样敏感：平面数据上 $\alpha=1.2$ 可能合适，$\alpha=0.01$ 会显得“网络学不会”；
  - 先画 $J$ 随迭代的曲线，再谈架构。

```python
def two_layer_update(params, grads, lr):
    for k in ("W1", "b1", "W2", "b2"):
        params[k] = params[k] - lr * grads["d" + k]
    return params

def two_layer_train(X, Y, params, lr=1.2, steps=10000):
    history = []
    for _ in range(steps):
        A2, caches = two_layer_forward(X, params)
        history.append(logistic_loss(A2, Y))
        grads = two_layer_backward(A2, Y, caches, params)
        params = two_layer_update(params, grads, lr)
    return params, history
```

- 决策边界：$A^{[2]}=0.5$ 即 $Z^{[2]}=0$，在输入平面上一般是曲线。
- 把网格点送入前向即可画出。
- 边界仍接近直线：隐层过窄、激活近线性、或训练未收敛。

> [!NOTE]+ 衔接
>
> 两层已经具备完整的前向缓存与反向外积。下一章只是把这两步放进 $l=1\to L$ 与 $l=L\to 1$ 的循环，接口不变。

<br>

# 深层网络

## 概念：深层网络

- **问题**
  - $L$ 层：$[n^{[0]}, n^{[1]}, \ldots, n^{[L]}]$；
  - 隐层激活通常相同，输出层由任务决定。
- **范围**
  - 把前向、损失、反向、更新写成可循环的层接口；
  - 输入 $X$，输出 $A^{[L]}$ 与梯度字典。
- **章节衔接**
  - 上一章的两层公式是本章循环的一次展开；
  - 深层公式是后续 Dropout、BatchNorm、残差连接的宿主；
  - 卷积与 RNN 只是把 $W^{[l]}A^{[l-1]}$ 换成带共享权重的线性算子，反向接口不变。

- **经验动机：特征层次**
  - 像素 → 边缘 → 部件 → 物体；
  - 同样参数预算下，深度组合比单层极宽更省，中间表示可复用；
  - 不是定理：过深而无残差、无合适初始化时，训练误差会上升；
  - 作业中的 $L$ 层模型通常 2-5 层全连接，目标是把循环接口写对，不是追 ImageNet 精度。

## 前向

对 $l=1,\ldots,L$：

$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]},\qquad
A^{[l]}=g^{[l]}(Z^{[l]}).
$$

输出层：

- 二分类：$g^{[L]}=\sigma$，$J$ 为二元交叉熵。
- $K$ 类：$g^{[L]}=\mathrm{softmax}$，

$$
a_k=\frac{e^{z_k-\max_j z_j}}{\sum_{j=1}^{K}e^{z_j-\max_j z_j}},\qquad
\mathcal{L}=-\sum_{k=1}^{K}y_k\log a_k.
$$

$y$ 为 one-hot 时 $\mathcal{L}=-\log a_{k^{\ast}}$。

> [!INFO]+ softmax 交叉熵的 $dz$
>
> 令 $a_k=\mathrm{softmax}(z)_k$，$\mathcal{L}=-\sum_k y_k\log a_k$。
>
> $$
> \frac{\partial a_k}{\partial z_j}=a_k(\delta_{kj}-a_j),\qquad
> \frac{\partial\mathcal{L}}{\partial z_j}=\sum_k \frac{\partial\mathcal{L}}{\partial a_k}\frac{\partial a_k}{\partial z_j}=a_j-y_j.
> $$
>
> 与 sigmoid + BCE 相同：配对使用时输出层误差是 $a-y$，不要再乘一遍 softmax 的 Jacobian。若损失改成 MSE，该简化不成立。

<br>


- 参数量约 $\sum_l (n^{[l]}n^{[l-1]}+n^{[l]})$。
- 全连接在图像上浪费：相邻像素应共享权重；这是卷积的动机。

作业常见结构 `[12288, 20, 7, 5, 1]`：

| 层 $l$ | $W^{[l]}$ | $b^{[l]}$ | $A^{[l]}$ |
| :--- | :--- | :--- | :--- |
| 1 | $(20, 12288)$ | $(20, 1)$ | $(20, m)$ |
| 2 | $(7, 20)$ | $(7, 1)$ | $(7, m)$ |
| 3 | $(5, 7)$ | $(5, 1)$ | $(5, m)$ |
| 4 | $(1, 5)$ | $(1, 1)$ | $(1, m)$ |

- 第一层独占几乎全部参数。
- 缓存 `caches[l-1]` 存 `(lin_cache, act_cache)`，其中 `lin_cache=(A^{[l-1]}, W^{[l]}, b^{[l]})`。
- 丢掉 $A^{[l-1]}$ 就无法形成 $\mathrm{d}W^{[l]}$。
- 前向必须 $l=1\to L$，因为 $A^{[l]}$ 依赖 $A^{[l-1]}$。
- 反向必须 $l=L\to 1$。
- 把反向写成与前向相同的升序循环是逻辑错误，梯度检查会立刻失败。


## 反向

- 令 $\mathrm{d}A^{[L]}=\partial J/\partial A^{[L]}$。
- sigmoid + BCE：直接设 $\mathrm{d}Z^{[L]}=A^{[L]}-Y$。
- 然后对 $l=L,\ldots,1$：

$$
\begin{aligned}
\mathrm{d}W^{[l]} &= \frac{1}{m}\mathrm{d}Z^{[l]}A^{[l-1]\top},\\
\mathrm{d}b^{[l]} &= \frac{1}{m}\sum_{i=1}^{m}\mathrm{d}Z^{[l](:,i)},\\
\mathrm{d}A^{[l-1]} &= W^{[l]\top}\mathrm{d}Z^{[l]},\\
\mathrm{d}Z^{[l-1]} &= \mathrm{d}A^{[l-1]}\odot g^{[l-1]\prime}(Z^{[l-1]}).
\end{aligned}
$$

- 若损失已与输出激活配对，最后一层不必乘 $g^{[L]\prime}$。
- 每层必须缓存 $A^{[l-1]}$，以及激活所需的 $Z^{[l]}$ 或掩码。

> [!INFO]+ 反向传播作为动态规划
>
> 从 $A^{[L]}$ 到 $W^{[1]}$ 的计算图是一条长链。朴素差分需要多次前向。反向传播对每条边只访问一次，总代价与一次前向同阶。缓存是空间换时间：不存 $A^{[l]}$ 就无法用外积形成 $\mathrm{d}W^{[l+1]}$。checkpointing 可丢掉部分激活并在反向时重算，用计算换显存。

<br>

## 实现

```python
def softmax(Z):
    """Z: (K, m)。按列做稳定 softmax。"""
    Zc = Z - Z.max(axis=0, keepdims=True)
    expZ = np.exp(Zc)
    return expZ / expZ.sum(axis=0, keepdims=True)

def linear_forward(A_prev, W, b):
    Z = W @ A_prev + b
    return Z, (A_prev, W, b)

def linear_activation_forward(A_prev, W, b, activation):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu_forward(Z)
    elif activation == "sigmoid":
        A, act_cache = sigmoid(Z), Z
    else:
        raise ValueError(activation)
    return A, (lin_cache, act_cache)

def l_model_forward(X, params, hidden_activation="relu"):
    """params 键为 W1,b1,...,WL,bL。"""
    caches = []
    A = X
    L = len(params) // 2
    for l in range(1, L):
        A, cache = linear_activation_forward(
            A, params[f"W{l}"], params[f"b{l}"], hidden_activation
        )
        caches.append(cache)
    AL, cache = linear_activation_forward(
        A, params[f"W{L}"], params[f"b{L}"], "sigmoid"
    )
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (dZ @ A_prev.T) / m
    db = dZ.sum(axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ
    return dA_prev, dW, db

def l_model_backward(AL, Y, caches, hidden_activation="relu"):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dZ = AL - Y
    current_cache = caches[L - 1]
    lin_cache, act_cache = current_cache
    dA_prev, dW, db = linear_backward(dZ, lin_cache)
    grads[f"dW{L}"], grads[f"db{L}"] = dW, db
    dA = dA_prev
    for l in reversed(range(1, L)):
        lin_cache, act_cache = caches[l - 1]
        if hidden_activation == "relu":
            dZ = relu_backward(dA, act_cache)
        else:
            dZ = tanh_backward(dA, act_cache)
        dA, dW, db = linear_backward(dZ, lin_cache)
        grads[f"dW{l}"], grads[f"db{l}"] = dW, db
    return grads

def l_model_update(params, grads, lr):
    L = len(params) // 2
    for l in range(1, L + 1):
        params[f"W{l}"] -= lr * grads[f"dW{l}"]
        params[f"b{l}"] -= lr * grads[f"db{l}"]
    return params
```

- 每步更新 $O(\sum_l n^{[l]}n^{[l-1]}m)$，与前向同阶。
- 不要在反向过程中立刻改 $W^{[l]}$。
- 后续层的 $\mathrm{d}A^{[l-1]}=W^{[l]\top}\mathrm{d}Z^{[l]}$ 需要用**更新前**的权重。
- 正确顺序：全部 `dW` 算完，再统一更新。

## 梯度检查

数值梯度：

$$
\frac{\partial J}{\partial\theta_i}\approx\frac{J(\theta_i+\varepsilon)-J(\theta_i-\varepsilon)}{2\varepsilon}.
$$

相对误差：

$$
\frac{\|\mathrm{d}\theta_{\mathrm{num}}-\mathrm{d}\theta_{\mathrm{bp}}\|_2}{\|\mathrm{d}\theta_{\mathrm{num}}\|_2+\|\mathrm{d}\theta_{\mathrm{bp}}\|_2}.
$$

- $\varepsilon=10^{-7}$、双精度：
  - 相对误差 $<10^{-7}$：实现通常正确；
  - $10^{-5}$ 量级：需要排查；
  - $10^{-2}$：公式或 shape 错误。
- 代价高：只在小模型、关闭 Dropout、BatchNorm 固定统计时使用。

```python
def gradient_check(forward_loss, params, grads, eps=1e-7):
    """forward_loss(params) 返回标量 J。params/grads 为同键字典。"""
    theta, grad = [], []
    keys = sorted(params)
    for k in keys:
        theta.append(params[k].reshape(-1))
        grad.append(grads["d" + k].reshape(-1))
    theta = np.concatenate(theta)
    grad = np.concatenate(grad)
    num = np.zeros_like(theta)
    for i in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += eps
        minus[i] -= eps

        def load(vec):
            out, offset = {}, 0
            for k in keys:
                size = params[k].size
                out[k] = vec[offset:offset + size].reshape(params[k].shape)
                offset += size
            return out

        num[i] = (forward_loss(load(plus)) - forward_loss(load(minus))) / (2 * eps)
    diff = np.linalg.norm(num - grad) / (np.linalg.norm(num) + np.linalg.norm(grad))
    return float(diff)
```

> [!NOTE]+ 衔接
>
> $L$ 层接口已写完。下一章回答信号与梯度的尺度：全零对称、过小收缩、过大饱和。先让预激活方差保持 $O(1)$，再谈正则与优化。

<br>

# 初始化

## 概念：初始化

- **问题**
  - 权重的尺度决定预激活方差，从而决定激活是否饱和、梯度是否消失或爆炸。
- **范围**
  - 给出零初始化失败原因，以及 Xavier 与 He 的方差公式；
  - 输入是层宽 $n^{[l-1]}, n^{[l]}$ 与激活类型，输出是 $W^{[l]}$ 的采样分布。
- **章节衔接**
  - 上一章给出可循环的 $L$ 层接口；
  - 本章管预激活方差，否则深层信号先死；
  - 下一章在能传到深层的梯度上加偏差-方差约束。

- 全零 → 对称性：同一层神经元接收相同梯度，表达能力退化为每层一个神经元。
- 方差过大：tanh/sigmoid 进入饱和区。
- 方差过小：每层信号幅度收缩，深层输出接近常数。

- 对线性层 $z=\sum_{j=1}^{n_{\mathrm{in}}} w_j a_j$，若 $a_j$ 零均值、方差 $v_a$，$w_j$ 独立零均值方差 $v_w$，则

$$
\mathrm{Var}(z)=n_{\mathrm{in}} v_w v_a.
$$

- 希望 $\mathrm{Var}(z)\approx\mathrm{Var}(a)$，则 $v_w\approx 1/n_{\mathrm{in}}$（Xavier/Glorot 前向约束）。
- 同时考虑反向 $\mathrm{Var}(\mathrm{d}a)\approx\mathrm{Var}(\mathrm{d}z)$，得到 $v_w\approx 1/n_{\mathrm{out}}$。
- 折中：

$$
W^{[l]}\sim\mathcal{N}\left(0,\frac{2}{n^{[l-1]}+n^{[l]}}\right)
\quad\text{或均匀分布等价尺度}.
$$

- ReLU 把负半轴置零，大约折半方差。
- He 初始化取

$$
W^{[l]}\sim\mathcal{N}\left(0,\frac{2}{n^{[l-1]}}\right).
$$

> [!INFO]+ Xavier 与 He
>
> - 是什么：按层宽设定 $W$ 的方差，使预激活保持 $O(1)$。
> - Xavier / Glorot (2010)：同时约束前向与反向方差，适合 tanh / sigmoid。
> - He et al. (2015)：ReLU 丢掉负半轴，方差约折半，故取 $2/n_{\mathrm{in}}$。
> - 相对固定小方差（如 $0.01$）：深层不再逐层收缩或饱和。

<br>

```python
def init_params(layer_dims, mode="he", seed=1):
    """layer_dims 如 [n_x, 20, 7, 5, 1]。"""
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        n_l, n_prev = layer_dims[l], layer_dims[l - 1]
        if mode == "he":
            scale = np.sqrt(2.0 / n_prev)
        elif mode == "xavier":
            scale = np.sqrt(2.0 / (n_prev + n_l))
        elif mode == "small":
            scale = 0.01
        else:
            raise ValueError(mode)
        params[f"W{l}"] = rng.normal(0.0, scale, size=(n_l, n_prev))
        params[f"b{l}"] = np.zeros((n_l, 1))
    return params
```

- 连乘理解：反向经过 $\prod_l W^{[l]\top} \mathrm{diag}(g'(z^{[l]}))$。
- 每层谱半径 $>1$：梯度指数增长（爆炸）。
- 每层谱半径 $<1$：梯度指数衰减（消失）。
- 残差连接把乘积改成 $1+$ 小残差，缓解该问题。

> [!EXAMPLE]+ 例5 过小与过大初始化
>
> 设十层 tanh 网络，$n^{[l]}=100$。若 $W_{ij}\sim\mathcal{N}(0,0.01^{2})$，每层方差约 $100\times 10^{-4}=0.01$，激活幅度逐层收缩，最后一层几乎是常数，梯度接近 0。若 $W_{ij}\sim\mathcal{N}(0,1)$，方差约 $100$，tanh 饱和，导数接近 0，同样学不动。He 对 ReLU 取 $\mathrm{Var}=2/n_{\mathrm{in}}=0.02$，标准差约 $0.14$，使 $\mathrm{Var}(z)$ 保持 $O(1)$。
>
> 作业对比图通常显示：零初始化训练准确率停在随机水平；过大初始化损失曲线震荡；He 初始化损失平稳下降。这是 C2M1 编程作业的核心观察。

<br>

> [!NOTE]+ 衔接
>
> 初始化解决“训练误差下不去、梯度消失或爆炸”。训练已经能拟合、dev 却高，是下一章正则的对象。二者不要混用：Dropout 不能代替 He 初始化。

<br>

# 正则化

## 概念：正则化

- **问题**
  - 训练误差与泛化误差之间的缺口；
  - 通过约束假设类或注入噪声，降低对训练集 idiosyncrasy 的拟合。
- **范围**
  - 区分偏差、方差与数据不匹配；
  - 给出 $L_2$、Dropout、数据增强（data augmentation）、early stopping 的公式与实现。
- **章节衔接**
  - 正则化在损失上加项或在前向中改计算图；
  - 优化章节处理的是如何下降，不自动解决过拟合。

## 偏差与方差

在可达到的最优误差（Bayes 误差或人类水平）附近分解：

- **偏差（bias / underfitting）**：训练误差明显高于最优误差。
- **方差（variance / overfitting）**：dev 误差明显高于训练误差。
- **数据不匹配**：dev 分布与训练分布不同，即使方差小也会差。

CS230 的实用判据（分类误差，最优误差接近 0 时）：

| 训练误差 | Dev 误差 | 诊断 |
| :--- | :--- | :--- |
| 高 | 高且接近训练 | 偏差：更大网络、更久训练、更好优化、换特征 |
| 低 | 高 | 方差：更多数据、正则、简化模型 |
| 高 | 更高 | 偏差与方差同时存在 |
| 低 | 低 | 再看 test 与部署分布 |

- “更大网络一定过拟合”并不成立。
- 足够大的网络配合正则与早停（early stopping），常常同时降低训练与 dev 误差。
- 先保证模型装得下训练数据，再压方差。

## $L_2$ 权重衰减（weight decay）

> [!INFO]+ $L_2$
>
> - 是什么：在损失上加 $\|W\|_F^{2}$，更新时等价于每次把 $W$ 乘以略小于 1 的因子。
> - 来自：岭回归；神经网络里常与 SGD 写在同一条更新式中。
> - 相对不加惩罚：权重变小，有效假设类收缩，dev 与 train 的缺口通常下降。
> - 不惩罚 $b$；$\lambda$ 过大则欠拟合。

<br>

$$
J_{\mathrm{reg}}=J+\frac{\lambda}{2m}\sum_{l}\|W^{[l]}\|_F^{2}.
$$

$\|W\|_F^{2}=\sum_{i,j}W_{ij}^{2}$。偏置通常不惩罚。梯度多一项 $\lambda W^{[l]}/m$，更新等价于

$$
W\leftarrow\left(1-\frac{\alpha\lambda}{m}\right)W-\alpha\,\mathrm{d}W_{\mathrm{data}}.
$$

- 权重每次被乘以略小于 1 的因子，故称 weight decay。
- $L_1$ 把平方换成绝对值，倾向稀疏；CS230 作业以 $L_2$ 为主。
- Frobenius 惩罚使权重变小，激活更可能落在 tanh/sigmoid 的线性区，有效降低复杂度。
- $\lambda$ 过大则欠拟合。

## Dropout

> [!INFO]+ Dropout
>
> - 是什么：训练时按 keep probability $p$ 随机置零神经元，推理时用完整网络。
> - 来自：Srivastava et al. (2014)。作业用 inverted dropout，训练除以 $p$。
> - 相对只加 $L_2$：迫使表示分布式，近似大量子网络的集成。
> - 卷积层参数已共享，收益较小；常只加在全连接头。

<br>

- 训练时每个神经元以概率 $1-p$ 被置零；$p$ 为 keep probability。
- Inverted dropout：训练时把保留值除以 $p$，使期望与测试时一致。
- 测试不必再缩放：

$$
A^{[l]}\leftarrow\frac{D^{[l]}\odot A^{[l]}}{p},\qquad D^{[l]}_{ij}\sim\mathrm{Bernoulli}(p).
$$

- 反向：只对保留位置传梯度，并同样除以 $p$。
- 直观：网络不能依赖某个特定隐单元，特征必须分布式编码。
- Dropout 相当于指数级共享权重的子网络集成。
- $p$ 常用：输入 $0.8$，隐层 $0.5$。
- 卷积层因参数已共享，Dropout 较弱；常改用空间 Dropout，或只在全连接头使用。
- 与 BatchNorm 同时使用时，二者都在改变激活分布，调参需更小心。
- 许多现代卷积架构主要靠 BatchNorm、数据增强与权重衰减。

```python
def inverted_dropout_forward(A, keep_prob, rng):
    """A: (n_l, m)。返回缩放后的激活与掩码。"""
    if keep_prob <= 0.0 or keep_prob > 1.0:
        raise ValueError("keep_prob must be in (0, 1]")
    mask = (rng.random(A.shape) < keep_prob).astype(A.dtype)
    A_drop = A * mask / keep_prob
    return A_drop, mask

def inverted_dropout_backward(dA, mask, keep_prob):
    return dA * mask / keep_prob
```

## 其他正则

- **数据增强**：图像翻转、裁剪、色移；文本同义改写。增强必须保持标签语义：水平翻转车牌可能破坏字符顺序。
- **Early stopping**：监控 dev 损失，回滚到最优检查点。它把有效训练步数当成超参数，与 $L_2$ 的作用有重叠。
- **输入归一化**：每维减均值除标准差，条件数更好，水平方向的细长损失谷被拉圆。
- **标签平滑**：把 one-hot 换成 $(1-\varepsilon)y+\varepsilon/K$，减轻过度自信。

- 训练时 Dropout 必须开启，推理必须关闭。
- Keras/TF 用 `training=` 切换；推理仍除以 $p$ 且随机置零，会使 dev 准确率无故抖动。
- Inverted dropout 的除 $p$ 只属于训练分支。
- $L_2$ 与 Dropout 可叠加，但二者都在压有效容量。
- 同时把 $\lambda$ 和 $1-p$ 开到很大，会回到欠拟合。
- 正交化：先用更大网络把训练误差降下来，再加一种正则观察 dev。
- 数据增强是“免费”样本，但标签必须不变。
- 分类可以水平翻转；检测框坐标要跟着翻转。
- 文字与医学左右不对称的影像不能盲目翻转。
- 增强过猛等于换了数据分布；dev 若未增强会出现人为不匹配。

```python
def l2_cost(params, lambda_, m):
    """只惩罚 W，不惩罚 b。"""
    s = 0.0
    L = len(params) // 2
    for l in range(1, L + 1):
        s += np.sum(params[f"W{l}"] ** 2)
    return lambda_ * s / (2.0 * m)

def l2_grad_add(grads, params, lambda_, m):
    L = len(params) // 2
    for l in range(1, L + 1):
        grads[f"dW{l}"] = grads[f"dW{l}"] + (lambda_ / m) * params[f"W{l}"]
    return grads
```

> [!WARNING]+ 正则不能修复错误标签或泄漏
>
> - test 经增强后泄漏进训练集：dev 指标虚高。
> - Dropout 不能补偿训练与部署的分布差。
> - 先检查划分与预处理，再加正则。

<br>

> [!NOTE]+ 衔接
>
> 正则针对的是方差。损失本身不降、曲线震荡或出现 `nan`，属于优化问题，加大 $\lambda$ 没有帮助。下一章处理下降轨迹。

<br>

# 优化

## 概念：优化

- **问题**
  - 随机估计下的高维非凸最小化；
  - 学习率、batch 大小与自适应矩决定轨迹。
- **范围**
  - 从 BGD/SGD/mini-batch 写到 Momentum、RMSprop、Adam 与学习率衰减；
  - 输入是梯度 $\mathrm{d}W$，输出是参数更新。
- **章节衔接**
  - 优化失败表现为损失不降或剧烈震荡；
  - 这与方差（dev 高、训练低）不同，不可用 Dropout 代替调学习率。

## Mini-batch

> [!INFO]+ Mini-batch
>
> - 是什么：用大小 $n_b$ 的子集估计全数据梯度，每批更新一次。
> - 相对全批量：吞吐高，可在 GPU 上向量化；相对 $n_b=1$ 的 SGD：噪声更小。
> - 改进的是更新频率与硬件效率，不是换了一种损失。
> - $n_b$ 过小则 BN 统计不稳；过大则每 epoch 步数变少。

<br>

把 $m$ 个样本分成大小 $n_b$ 的批。一批一次更新：

$$
J_t=\frac{1}{n_b}\sum_{i\in\mathcal{B}_t}\mathcal{L}^{(i)}.
$$

- $n_b=m$：批量梯度，方向稳定，每 epoch 更新 1 次。
- $n_b=1$：SGD，噪声大，难以向量化。
- $n_b\in\{32,64,128,256,512\}$：折中。常用 2 的幂以契合 GPU tile。

- 一个 epoch = 全部样本被看见一次。
- 损失曲线呈云状下降；若要平滑，对 $J_t$ 做指数滑动平均。
- 每个 epoch 开始时打乱样本，否则同一 mini-batch 偏差会被记住。

## 指数滑动平均（exponentially weighted moving average）

$$
v_t=\beta v_{t-1}+(1-\beta)\theta_t.
$$

$v_t$ 大约平均最近 $1/(1-\beta)$ 个值：$\beta=0.9$ 约 10 步，$\beta=0.999$ 约 1000 步。偏差修正：

$$
\hat{v}_t=\frac{v_t}{1-\beta^{t}}.
$$

$t=1$ 且 $v_0=0$ 时，$\hat{v}_1=\theta_1$，避免初期被零拉低。

## Momentum

> [!INFO]+ Momentum
>
> - 是什么：用梯度的指数滑动平均当速度，再沿速度更新。
> - 来自：Polyak 重球法；深度学习里 $\beta$ 常取 $0.9$。
> - 相对朴素 SGD：狭长谷里振荡方向相消，沿谷底更快。

<br>

$$
\begin{aligned}
v_{\mathrm{d}W} &= \beta v_{\mathrm{d}W} + (1-\beta)\,\mathrm{d}W,\\
W &\leftarrow W-\alpha v_{\mathrm{d}W}.
\end{aligned}
$$

$\beta$ 常用 $0.9$。在狭长谷中，振荡方向的梯度正负相消，沿谷底方向累积。有的实现把 $(1-\beta)$ 省略，此时 $\alpha$ 的有效尺度改变，超参数不可直接抄。

## RMSprop

> [!INFO]+ RMSprop
>
> - 是什么：用梯度平方的滑动平均缩放每维步长。
> - 来自：Hinton 课程；针对 AdaGrad 学习率单调衰减过快。
> - 相对 Momentum：自动压低长期大梯度坐标，抬高低梯度坐标。

<br>

$$
\begin{aligned}
s_{\mathrm{d}W} &= \beta_2 s_{\mathrm{d}W} + (1-\beta_2)\,\mathrm{d}W\odot\mathrm{d}W,\\
W &\leftarrow W-\alpha\frac{\mathrm{d}W}{\sqrt{s_{\mathrm{d}W}}+\varepsilon}.
\end{aligned}
$$

$\varepsilon$ 如 $10^{-8}$，防止除零。梯度长期偏大的坐标步长被缩小，偏小的被放大，自动调整各维学习率。

## Adam

> [!INFO]+ Adam
>
> - 是什么：Momentum 的一阶矩加 RMSprop 的二阶矩，并对早期 $v,s$ 做偏差修正。
> - 来自：Kingma & Ba (2015)。默认 $\beta_1=0.9$，$\beta_2=0.999$。
> - 相对只选其中一种：稀疏或尺度不一的参数也能用同一 $\alpha$ 起步。
> - 仍要选 $\alpha$；漏计步数 $t$ 等于修正永久停在 $t=1$。

<br>

- Adam 同时维护一阶矩与二阶矩，并做偏差修正。
- 默认 $\beta_1=0.9$，$\beta_2=0.999$，$\varepsilon=10^{-8}$。

$$
\begin{aligned}
v_{\mathrm{d}W} &= \beta_1 v_{\mathrm{d}W}+(1-\beta_1)\,\mathrm{d}W,\\
s_{\mathrm{d}W} &= \beta_2 s_{\mathrm{d}W}+(1-\beta_2)\,\mathrm{d}W\odot\mathrm{d}W,\\
\hat{v}_{\mathrm{d}W} &= \frac{v_{\mathrm{d}W}}{1-\beta_1^{t}},\qquad
\hat{s}_{\mathrm{d}W} = \frac{s_{\mathrm{d}W}}{1-\beta_2^{t}},\\
W &\leftarrow W-\alpha\frac{\hat{v}_{\mathrm{d}W}}{\sqrt{\hat{s}_{\mathrm{d}W}}+\varepsilon}.
\end{aligned}
$$

- $t$ 是更新步数，从 1 计。
- 每个参数张量（每个 $W^{[l]},b^{[l]}$）都有自己的 $v,s$。
- Adam 对初始学习率仍敏感。
- 常用 $\alpha\in\{10^{-3},3\times10^{-4},10^{-4}\}$，再配合衰减。

> [!INFO]+ 偏差修正为何必要
>
> $s$ 的 $\beta_2=0.999$ 使早期 $s_t\approx(1-\beta_2)\sum_{k} \beta_2^{t-k} g_k^{2}$ 远小于稳态。若不除 $1-\beta_2^{t}$，分母过小，最初若干步更新会过大。$\beta_1$ 同样需要修正，但 $\beta_1=0.9$ 时偏差消失更快。实现中漏掉 $t$ 递增，等于永久处于 $t=1$ 的修正，学习率标度错误。

<br>

```python
def adam_init(params):
    v, s = {}, {}
    for k, val in params.items():
        v["d" + k] = np.zeros_like(val)
        s["d" + k] = np.zeros_like(val)
    return v, s, 0

def adam_update(params, grads, v, s, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    for k in params:
        gk = grads["d" + k]
        v["d" + k] = beta1 * v["d" + k] + (1.0 - beta1) * gk
        s["d" + k] = beta2 * s["d" + k] + (1.0 - beta2) * (gk ** 2)
        v_hat = v["d" + k] / (1.0 - beta1 ** t)
        s_hat = s["d" + k] / (1.0 - beta2 ** t)
        params[k] = params[k] - lr * v_hat / (np.sqrt(s_hat) + eps)
    return params, v, s, t
```

## 学习率衰减

$$
\alpha_t=\frac{\alpha_0}{1+\delta\,\mathrm{epoch}},\qquad
\alpha_t=\alpha_0\cdot 0.95^{\mathrm{epoch}},\qquad
\alpha_t=\frac{k\alpha_0}{\sqrt{t}}.
$$

- Transformer 中更常见：余弦退火、线性 warmup 后衰减。
- $\alpha$ 过大：损失爆炸或震荡；过小：下降极慢，易停在尖锐区域。
- 可先找使损失开始发散的 $\alpha$，再取大约 $1/3$ 到 $1/10$。
- 高维中局部最优较少成为主因；鞍点与平台更常见。
- 随机 mini-batch 噪声有助于逃离平坦区。
- Batch 过小：噪声过大；过大：泛化有时变差；与学习率线性缩放等经验规则一起调。
- Nesterov：先沿速度走一步再求梯度；对凸问题有更紧收敛率；课程以普通 Momentum 为主。
- AdaGrad：历史平方梯度一直累加，学习率单调下降，后期可能过早停止。
- RMSprop：用滑动平均避免 AdaGrad 的过早停止。
- Adam = Momentum + RMSprop，不是“一定最好”。
- 视觉分类上，调好的 SGD+Momentum 仍常与 Adam 打平或更好。
- Adam 对初始 $\alpha$ 更宽容，适合作业与项目起步。
- 损失仍陡降时不要过早衰减 $\alpha$；先用常数 $\alpha$ 观察到平台，再衰减。
- $\alpha$ 衰减与 early stopping 同时打开时，dev 变好可能只是步长变小，不一定是泛化改善。

```python
def mini_batches(X, Y, batch_size, rng):
    """按列打乱后切片。最后一批可以小于 batch_size。"""
    m = X.shape[1]
    perm = rng.permutation(m)
    Xs, Ys = X[:, perm], Y[:, perm]
    batches = []
    for i in range(0, m, batch_size):
        batches.append((Xs[:, i:i + batch_size], Ys[:, i:i + batch_size]))
    return batches
```

- 打乱必须每个 epoch 重新做。
- 固定 batch 顺序等于让网络记住“第 3 个 batch 总是那些困难样本”，梯度噪声不再无偏。

> [!NOTE]+ 衔接
>
> Adam、Momentum、衰减决定轨迹。$\alpha$、$\beta$、batch 大小本身不是梯度，下一章当作超参数搜索。

<br>

# 超参数

## 概念：超参数

- **问题**
  - 不能由梯度直接学习、但控制学习过程的量：$\alpha,\beta_1,\beta_2,\lambda$，层宽、层数，batch 大小，keep_prob。
- **范围**
  - 用随机搜索（random search）与尺度变换代替网格穷举；
  - 输入是搜索空间，输出是一组可复现配置。
- **章节衔接**
  - 上一章给出 Adam 等更新规则；
  - 本章决定 $\alpha$、batch、层宽等不能由梯度学习的量；
  - 下一章用 BatchNorm 把激活尺度从学习率里拆出来。

重要性大致分层：

1. 学习率 $\alpha$。
2. Momentum $\beta$、隐层宽度、mini-batch 大小。
3. 层数、学习率衰减、Adam 的 $\beta_1,\beta_2$、$\varepsilon$。

- 网格搜索：低维且参数同等重要时可用。
- 超参数很多时：随机采样能探索更多“重要轴”的取值。
- 学习率在对数轴采样：例如 $10^{-4}$ 到 $10^{-1}$ 上取 $\alpha=10^{u}$，$u\sim\mathrm{Unif}(-4,-1)$。
- $\beta=0.9,0.99,0.999$ 对应 $1-\beta$ 的对数轴。
- Panda：盯一组配置仔细调；计算预算小时用。
- Caviar：并行训练多组；有集群时用。
- 所有试验记录 seed、代码版本、数据版本与最终 dev 指标，否则无法复现“偶然得到的好点”。
- 超参数不要在 test 上选。
- 每在 test 上看一次并据此改配置，test 就变成第二块 dev，最终报告的泛化被高估。
- 正确：train 拟合，dev 选模型，test 只评一次。
- C2M3：用编程框架搭同一模型。
- 框架把 `GradientTape` 或静态图当成计算图；`optimizer.apply_gradients` 对应手写 `W -= lr * dW`。
- 仍须手动保证：

- 输入 shape 与作业约定一致，或显式 `transpose`；
- `training=True/False` 切换 Dropout 与 BatchNorm；
- 分类用 `from_logits=True` 的交叉熵，避免 `softmax` 后再 `log` 的不稳定；
- 指标在 epoch 结束用完整 dev 集计算，不要只用最后一个 mini-batch。

```python
import tensorflow as tf

def tf_two_layer(n_x, n_h, n_y, lr=0.01):
    """Keras 对应手写两层网络；此处仅示范接口，不是作业标准答案。"""
    inputs = tf.keras.Input(shape=(n_x,))
    h = tf.keras.layers.Dense(n_h, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(n_y, activation="sigmoid")(h)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model
```

- Keras 默认 `(m, n_x)`，与 NumPy 作业的 `(n_x, m)` 相反。
- 把同一组权重搬到 Keras 时需要转置 $W$。

> [!NOTE]+ 衔接
>
> 超参数在 dev 上选，test 只评一次。激活尺度仍随下层参数漂移时，用下一章的 BatchNorm 固定激活尺度，不能只调 $\alpha$。

<br>

# BatchNorm

## 概念：BatchNorm

> [!INFO]+ BatchNorm
>
> - 是什么：对当前 batch 的预激活标准化，再用可学习的 $\gamma,\beta$ 做仿射。
> - 来自：Ioffe & Szegedy (2015)。论文动机写内部协变量偏移；实践上主要是让损失面更好走。
> - 相对只调初始化与 $\alpha$：可用更大学习率，对起始尺度不那么敏感。
> - 推理必须改用 running mean / variance，不能沿用当批统计。

<br>

- **问题**
  - 内部协变量偏移（internal covariate shift）：深层输入分布随下层参数变化；
  - 对每个 mini-batch 的预激活做标准化，再用仿射参数恢复表达能力。
- **范围**
  - 写训练期与推理期公式；
  - 输入 $Z^{[l]}$，输出 $Z_{\mathrm{BN}}^{[l]}$。
- **章节衔接**
  - 上一章用搜索选择 $\alpha$ 与 batch；
  - 本章标准化每层激活，降低对初始化与学习率的敏感；
  - 下一章按偏差、方差与分布不匹配做误差分解。

对当前 batch 的每个特征维（全连接：神经元；卷积：通道）：

$$
\mu=\frac{1}{n_b}\sum_{i}z^{(i)},\qquad
\sigma^{2}=\frac{1}{n_b}\sum_{i}(z^{(i)}-\mu)^{2},\qquad
\tilde{z}=\frac{z-\mu}{\sqrt{\sigma^{2}+\varepsilon}}.
$$

$$
z_{\mathrm{BN}}=\gamma\odot\tilde{z}+\beta.
$$

- $\gamma,\beta$ 可学习。
- 若 $\gamma=\sqrt{\sigma^{2}+\varepsilon}$，$\beta=\mu$，理论上可还原恒等映射，因此不损失一层的表示能力。
- 加入 BatchNorm 后，$b$ 与 $\beta$ 冗余，实现中常省略 $b$。

- 训练：用 batch 统计。
- 推理：用运行平均。

$$
\bar{\mu}\leftarrow \eta\bar{\mu}+(1-\eta)\mu_{\mathrm{batch}},
$$

- $\eta$ 如 $0.9$。
- 评估模式必须切换到 running statistics。
- 否则 batch 大小为 1 时 $\sigma^{2}=0$，行为与训练不一致。

> [!INFO]+ BatchNorm 的反向要点
>
> $\partial J/\partial z$ 不能只通过 $\gamma/\sqrt{\sigma^{2}+\varepsilon}$ 缩放，因为 $\mu$ 与 $\sigma^{2}$ 也依赖 batch 内每个 $z^{(i)}$。完整 Jacobian 含三项：经 $\tilde{z}$ 的直接路径、经 $\mu$ 的路径、经 $\sigma^{2}$ 的路径。框架会自动求导；手写时需对这三项求和。$\gamma,\beta$ 的梯度是 $\partial J/\partial z_{\mathrm{BN}}$ 分别与 $\tilde{z}$、$1$ 的 batch 求和。

<br>

```python
def batchnorm_forward(Z, gamma, beta, cache_bn, eps=1e-8, momentum=0.9, train=True):
    """Z: (n, m)。cache_bn 存 running_mean, running_var。"""
    if train:
        mu = Z.mean(axis=1, keepdims=True)
        var = Z.var(axis=1, keepdims=True)
        cache_bn["running_mean"] = momentum * cache_bn["running_mean"] + (1 - momentum) * mu
        cache_bn["running_var"] = momentum * cache_bn["running_var"] + (1 - momentum) * var
    else:
        mu = cache_bn["running_mean"]
        var = cache_bn["running_var"]
    Z_hat = (Z - mu) / np.sqrt(var + eps)
    out = gamma * Z_hat + beta
    return out, {"Z": Z, "Z_hat": Z_hat, "mu": mu, "var": var, "gamma": gamma, "eps": eps}

def batchnorm_backward(d_out, cache):
    Z, Z_hat, mu, var, gamma, eps = (
        cache["Z"], cache["Z_hat"], cache["mu"], cache["var"], cache["gamma"], cache["eps"]
    )
    m = Z.shape[1]
    dgamma = (d_out * Z_hat).sum(axis=1, keepdims=True)
    dbeta = d_out.sum(axis=1, keepdims=True)
    dZ_hat = d_out * gamma
    inv_std = 1.0 / np.sqrt(var + eps)
    dvar = (dZ_hat * (Z - mu) * (-0.5) * inv_std ** 3).sum(axis=1, keepdims=True)
    dmu = (dZ_hat * (-inv_std)).sum(axis=1, keepdims=True) + dvar * (-2.0 * (Z - mu).mean(axis=1, keepdims=True))
    dZ = dZ_hat * inv_std + dvar * 2.0 * (Z - mu) / m + dmu / m
    return dZ, dgamma, dbeta
```

- 效果：学习率可以更大；对初始化不那么敏感；本身带轻微正则（batch 统计噪声）。
- 小 batch 时统计不可靠：改 LayerNorm（layer normalization）、GroupNorm，或增大 batch。
- RNN 中 BatchNorm 不如 LayerNorm 常见。
- 稳定深层训练的工具还有：softmax 温度、梯度裁剪（gradient clipping）、残差与归一化。
- BatchNorm 是固定激活尺度的一种做法。

> [!NOTE]+ 对照：BatchNorm 与 LayerNorm
>
> - BatchNorm：沿 batch 维；训练用 batch 统计，推理用 running average。
> - LayerNorm：沿特征维；训练与推理公式相同。
> - 全连接与卷积训练常用 BatchNorm；RNN 与 Transformer 常用 LayerNorm。
> - 小 batch 时 BatchNorm 统计不可靠。

<br>

> [!NOTE]+ 衔接
>
> C1-C2 给出可训练、可正则、可下降的网络。下一章按训练误差、dev 误差与部署误差分开处理。

<br>

# 项目策略

## 概念：策略

- **问题**
  - 把项目误差拆开：容量、正则、数据划分与部署分布不要一次全改。
- **范围**
  - 建立单一评估指标、划分数据、做误差分析、决定下一实验；
  - 输入是当前系统的错误样本，输出是优先级列表。
- **章节衔接**
  - C1-C2 已经给出可训练模型；
  - 策略决定何时加数据、何时加正则、何时换架构；
  - 卷积与序列章节提供新的算子，仍用本章的误差表决定要不要换。

## 正交化

> [!INFO]+ 正交化
>
> - 是什么：把拟合训练、拟合 dev、拟合部署拆开，一次只改一类误差来源。
> - 来自：Andrew Ng 在 C3 的项目策略；对应实验设计里一次改一个因子。
> - 相对同时加宽网络又加大 $\lambda$：否则无法判断是容量还是正则在起作用。

<br>

正交化要求一次只改一类误差来源：

- 拟合训练集：更大模型、更好优化、更久训练。
- 拟合 dev 集：正则、更多与 train 同分布的数据。
- 拟合 test 集：dev/test 同分布且足够大。
- 拟合现实：确认 dev 分布就是部署分布。

- 一个改动同时影响拟合能力与正则强度时，因果被混淆。
- 例如一边加宽网络一边把 $\lambda$ 提高一个数量级，无法解释哪一项起作用。

## 评估指标

- 优化指标（optimizing metric）只留一个，例如准确率。
- 满意度指标（satisficing metric）设阈值，例如运行时间 $<100\,\mathrm{ms}$、内存 $<1\,\mathrm{GB}$。
- 多个优化指标应加权成单一数字，否则无法比较模型 A/B。
- 类别不平衡时准确率失效：改用 precision、recall、F1、PR AUC、代价加权错误率。
- 检测用 mAP；分割用 IoU 或 Dice；检索用 Recall@$k$。
- 指标必须在项目开始时与业务代价对齐，不是训练后再挑选当时最高的曲线。
- Dev/test 必须来自同一分布，且尽量等于部署分布。
- 上线数据是手机模糊图时，dev 不能只用单反清晰图。
- Train 可以来自更大、略有差异的来源，但要单独做训练-开发不匹配分析。
- 大数据下划分不必 70/30：例如 $10^{6}$ 样本可用 98/1/1。
- 条件：dev/test 大到能分辨 $0.1\%$ 的指标差。
- Dev 太小会使超参数选择变成对噪声拟合。

## 人类水平

> [!INFO]+ 人类水平
>
> - 是什么：用一组人的误差 $h$ 代理不可观测的 Bayes 误差。
> - 来自：C3 把偏差拆成“相对人类”与“可避免”两部分。
> - 相对只看训练误差：训练 $\gg h$ 才优先加容量；已经贴着 $h$ 再加大模型收益很小。
> - $h$ 取决于典型人还是专家委员会；标注不一致时 $h$ 会被高估。

<br>

把 Bayes 误差的代理取为人类水平误差 $h$。

- 训练误差 $\gg h$：偏差，优先模型容量与优化。
- 训练误差 $\approx h$，dev 误差 $\gg$ 训练误差：方差。
- 训练与 dev 都接近 $h$：继续加大模型的收益很小，应改任务定义、输入信号或产品范围。

- 人类水平本身有定义问题：典型人、专家、专家委员会的误差不同。
- 应取与产品目标匹配的那一层。
- 模型超过单人水平后，误差分析仍有用，不宜再把人类水平当作不可逾越的上限。
- 可避免偏差（avoidable bias） = 训练误差 $-$ $h$。
- 方差 = dev 误差 $-$ 训练误差。
- 两者都大时先打较大的那一项。
- 贝叶斯误差不可直接得，$h$ 只是代理。
- 若标注本身不一致，$h$ 被高估，模型会看起来“已经超过人类”而其实在拟合噪声。
- 此时应先改善标注协议。

<br>

## 误差分析

- 手动检查 $n$ 个错误 dev 样本（如 100 个）。
- 统计可修复类别：标注错、模糊、过曝、类间混淆、定位框不准等。
- 某一类只占错误的 $2\%$：即使完美修复也只提升 $2\%$，不应成为主攻。
- 并行：把同一批错误按多个标签打勾，比较上限。
- 这比凭直觉加新模块便宜。

> [!EXAMPLE]+ 例2 猫识别器的误差表
>
> 100 张误分类中：狗 8、模糊 61、滤镜 21、标注错误 10。即使狗分类器做到完美，上限约 $8\%$。优先收集模糊与滤镜数据，或在训练中加模糊增强，比设计狗检测器更合理。

<br>

## 分布不匹配

- Train 来自网络图片、dev 来自用户手机：训练-开发不匹配。
- 构造训练-训练 dev：从训练集划出一块与训练同分布的 holdout。
- 比较：

- 训练误差 vs 训练-训练 dev：方差。
- 训练-训练 dev vs 开发集：数据不匹配。

- 修复不匹配：收集更接近 dev 的训练数据；合成（须验证合成有效）；或领域自适应。
- 不要在完全不同的 dev 上做模型选择还自称泛化。

## 迁移、多任务与端到端

- **迁移学习**：在源任务大数据上预训练，再在目标任务上替换输出层并微调。源与目标应共享低层特征（边缘、音素、词法）。目标数据很少时冻结主体，只训头；数据多时解冻更多层、用小学习率。
- **多任务学习**：一个网络多个输出，当任务共享表示且数据量不平衡时可能互相帮助。若任务噪声结构冲突，可能负迁移。
- **端到端**：数据极大时，直接 $x\mapsto y$ 可胜过手工流水线。数据不足时，拆成有中间监督的模块（如检测再识别）更稳。端到端把模块误差变成不可诊断的黑箱，需要更多数据来补偿。

## 数据与迭代

- 先搭能跑通的简单基线（甚至逻辑回归或小 CNN）。
- 建立训练-评估管道。
- 每周只改一件有误差分析支持的事。
- 不要第一周就上未验证的巨大 Transformer。
- 数据量不足：先估计人工标注成本与增强上限。
- 标签质量差会形成不可逾越的误差地板。

## 案例

Course 3 的两个书面案例是策略章的考试重点。

- **鸟类识别（Peacetopia）**
  - 城市要识别鸟；
  - 准确率是优化指标；运行时间、内存是满意度指标；
  - 人类水平约 $0\%$；若训练误差 $4\%$、dev $9\%$，先降方差（数据、正则），不是无限制加深网络；
  - 照片从互联网换成手机拍摄后，dev 分布变了：必须重建与部署一致的 dev/test；
  - 不能继续用旧互联网 dev 做模型选择；
  - 错误分析若显示“雾天”占大部分，优先收集雾天数据；
  - 把测试集拿来调参会污染最终数字。
- **自动驾驶**
  - 检测交通灯、行人等；
  - 可做成多任务：一个网络多个头；
  - 某任务数据极少时，多任务可能被大数据任务主导，需要加权或分阶段训练；
  - 端到端“像素到方向盘”在数据不足时不可诊断；
  - 拆成检测、跟踪、规划后，每一段都有自己的误差分析；
  - 雷达与摄像头融合时，传感器不同步属于数据问题，不是换 Adam 能修的。

> [!EXAMPLE]+ 例6 避免不可比的指标
>
> - 模型 A：准确率 $97\%$，延迟 $150\,\mathrm{ms}$。
> - 模型 B：准确率 $95\%$，延迟 $40\,\mathrm{ms}$。
> - 产品要求 $<100\,\mathrm{ms}$ 时，A 直接不合格。
> - 先过滤满意度约束，再在合格者中比优化指标。
> - 加权成单一数字也可以，但权重必须事先写进提案。

<br>

> [!NOTE]+ 四类失败
>
> - 训练误差高：偏差或优化失败。先看 $J$ 是否下降，再谈容量。
> - 训练低、dev 高、且同分布：方差。加数据或正则。
> - 训练-训练 dev 低、真实 dev 高：分布不匹配。
> - 开发集指标高、线上差：dev 不是现实。先改划分，再改模型。

<br>

> [!NOTE]+ 衔接
>
> 策略不指定层类型。图像上全连接浪费邻近先验。下一章把 $WA$ 换成卷积，反向接口仍是 $\mathrm{d}W,\mathrm{d}X$。

<br>

# 卷积

## 概念：卷积

> [!INFO]+ 卷积
>
> - 是什么：局部核在空间上滑动，权值共享，输出特征图。
> - 来自：LeNet-5（1998）；AlexNet / VGG 之后成为视觉默认算子。
> - 相对把图像展平后全连接：参数不随 $H,W$ 增长，并利用邻近像素相关这一先验。
> - 深度学习实现里的“卷积”多数是互相关，核不翻转。

<br>

- **问题**
  - 网格结构数据：图像、特征图；
  - 用局部核在空间上滑动，实现权值共享与平移等变（translation equivariance）。
- **范围**
  - 计算输出尺寸、参数量、卷积与池化前向；
  - 输入 $n_H\times n_W\times n_C$，输出更小或同尺寸的特征图。
- **章节衔接**
  - 上一章的策略决定要不要换架构；本章给出图像上的默认算子；
  - 卷积层替换全连接中的 $WA$，反向仍是线性算子的伴随；
  - 下一章用残差把深层卷积训下去；再下一章在骨干上接检测或分割头。

- 全连接：把 $32\times32\times3$ 展平后，第一层若有 1000 个单元，参数约 $3\times10^{6}$，且忽略邻近像素更相关这一先验。
- 卷积核 $f\times f\times n_C$ 在所有位置共享。
- 参数量 $f^{2} n_C n_C'$（再加偏置 $n_C'$）。

## 运算

- 单通道、步幅（stride） $s=1$、无填充（padding）。
- 核 $K\in\mathbb{R}^{f\times f}$ 与输入 $X$ 的互相关（深度学习中常仍称为卷积）：

$$
Z_{ij}=\sum_{u=0}^{f-1}\sum_{v=0}^{f-1}K_{uv}X_{i+u,j+v}+b.
$$

- 多通道：核为 $f\times f\times n_C$，在通道维求和后得到一张输出图。
- $n_C'$ 个核得到 $n_C'$ 张输出。

输出空间尺寸：

$$
n_H'=\left\lfloor\frac{n_H+2p-f}{s}\right\rfloor+1,\qquad
n_W'=\left\lfloor\frac{n_W+2p-f}{s}\right\rfloor+1.
$$

- **valid**：无需填充，$p=0$，输出缩小。
- **same**：希望 $n'=n$（在 $s=1$ 时），则 $p=(f-1)/2$，故 $f$ 常取奇数。
- **步幅 $s$**：每隔 $s$ 个像素计算一次，兼具下采样。

- 参数量与输入的 $n_H,n_W$ 无关，只与核尺寸和通道有关。
- 计算量仍随空间尺寸线性增长。

> [!EXAMPLE]+ 例3 尺寸计算
>
> 输入 $32\times32\times3$，核 $5\times5$，步幅 1，same padding：$p=2$，输出 $32\times32\times n_C'$。改为 valid、步幅 2：
>
> $$
> n'=\left\lfloor\frac{32-5}{2}\right\rfloor+1=14.
> $$
>
> 若再接 $2\times2$ 步幅 2 的池化，空间变为 $7\times7$。

<br>

```python
def conv2d_forward(X, W, b, stride=1, padding=0):
    """X: (m, n_H, n_W, n_C), W: (f, f, n_C, n_C_out), b: (1,1,1,n_C_out)。"""
    m, n_H, n_W, n_C = X.shape
    f, _, _, n_C_out = W.shape
    n_H_out = (n_H + 2 * padding - f) // stride + 1
    n_W_out = (n_W + 2 * padding - f) // stride + 1
    Xp = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    Z = np.zeros((m, n_H_out, n_W_out, n_C_out))
    for i in range(m):
        for h in range(n_H_out):
            for w in range(n_W_out):
                for c in range(n_C_out):
                    hs, ws = h * stride, w * stride
                    window = Xp[i, hs:hs + f, ws:ws + f, :]
                    Z[i, h, w, c] = np.sum(window * W[:, :, :, c]) + b[0, 0, 0, c]
    return Z, (X, W, b, stride, padding)

def pool_forward(A, f=2, stride=2, mode="max"):
    m, n_H, n_W, n_C = A.shape
    n_H_out = (n_H - f) // stride + 1
    n_W_out = (n_W - f) // stride + 1
    out = np.zeros((m, n_H_out, n_W_out, n_C))
    for i in range(m):
        for h in range(n_H_out):
            for w in range(n_W_out):
                hs, ws = h * stride, w * stride
                window = A[i, hs:hs + f, ws:ws + f, :]
                if mode == "max":
                    out[i, h, w, :] = window.max(axis=(0, 1))
                else:
                    out[i, h, w, :] = window.mean(axis=(0, 1))
    return out
```

- 教学实现：$O(m n_H' n_W' n_C' f^{2} n_C)$ 的六重循环。
- 生产：滑动窗口展开为 im2col 矩阵乘，或 Winograd / FFT / cuDNN。
- $1\times 1$ 卷积不看邻域，只在通道维做线性组合；参数 $n_C n_C'$；常用于降维或升维。
- 不是“没用的卷积”，而是 Inception 与 bottleneck 的计算开关。
- `np.pad` 在高、宽两侧各加 $p$ 列零。
- same 且 $s=1$、$f$ 为奇数：两侧对称。
- $f$ 为偶数时 same 无法完全对称，框架把多出的 1 像素分到一侧；作业一般避开偶数核。

> [!INFO]+ 卷积反向
>
> 记前向 $Z=X * K$。$\partial J/\partial K$ 是 $X$ 与 $\mathrm{d}Z$ 的互相关：每个核位置把所有窗口与对应输出梯度相乘再求和。$\partial J/\partial X$ 是 $\mathrm{d}Z$ 与 $180^{\circ}$ 翻转核的卷积，并处理步幅造成的空洞（把 $\mathrm{d}Z$ 的像素之间插入 $s-1$ 个零）。偏置梯度是 $\mathrm{d}Z$ 在 batch 与空间维上的和。池化反向不学习参数，只路由梯度。

<br>

```python
def conv2d_backward(dZ, cache):
    """与 conv2d_forward 配对的教学反向；shape 与前向缓存一致。"""
    X, W, b, stride, padding = cache
    m, n_H, n_W, n_C = X.shape
    f, _, _, n_C_out = W.shape
    Xp = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    dXp = np.zeros_like(Xp)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    _, n_H_out, n_W_out, _ = dZ.shape
    for i in range(m):
        for h in range(n_H_out):
            for w in range(n_W_out):
                hs, ws = h * stride, w * stride
                for c in range(n_C_out):
                    window = Xp[i, hs:hs + f, ws:ws + f, :]
                    dW[:, :, :, c] += window * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
                    dXp[i, hs:hs + f, ws:ws + f, :] += W[:, :, :, c] * dZ[i, h, w, c]
    dX = dXp[:, padding:padding + n_H, padding:padding + n_W, :] if padding else dXp
    return dX, dW / m, db / m
```

- 卷积层是否除以 $m$，取决于损失是否已对 batch 平均。
- 与全连接保持同一约定：损失含 $1/m$ 则梯度也含 $1/m$。


## 池化

> [!INFO]+ 池化
>
> - 是什么：窗口内取最大或平均，降低空间分辨率，无（或极少）可学参数。
> - 来自：早期 ConvNet 的下采样层。
> - 相对只用步幅：提供局部平移稳健性；max 反向只把梯度送到 argmax。
> - 现代骨干常改用 stride 卷积下采样，分类头仍常用 global average pooling。

<br>

- Max pooling：在 $f\times f$ 窗口取最大；无学习参数；提供局部平移稳健性。
- Average pooling：取均值。
- 反向：max 把梯度路由到 argmax，其余为零；average 把梯度均分。
- 现代网络常用 stride 卷积代替大范围池化。
- 分类头前仍广泛使用 global average pooling。

## 典型层序列

$$
\mathrm{Conv}\rightarrow\mathrm{ReLU}\rightarrow\mathrm{Conv}\rightarrow\mathrm{ReLU}\rightarrow\mathrm{Pool}
$$

- 重复若干次后展平或全局平均，再接全连接或卷积分类头。
- LeNet、AlexNet、VGG：空间下降、通道上升。
- VGG 用堆叠 $3\times3$ 代替大核：两层 $3\times3$ 的感受野（receptive field）等于一层 $5\times5$，参数更少且多一层非线性。

- 感受野：输出一个像素所依赖的输入区域。
- 步幅与池化会快速扩大感受野。
- 过早下采样会丢掉定位信息，对检测与分割不利。

- 一层 $f\times f$、步幅 1：感受野增加 $f-1$。
- 堆叠三层 $3\times3$：感受野为 $7$，与一层 $7\times7$ 相同。
- 参数：$3\cdot 9\cdot C^{2}$ 对 $49 C^{2}$，且多两次 ReLU。
- 这是 VGG 用小核堆叠的定量原因。
- 通道通常按 stage 倍增：64 → 128 → 256 → 512，同时空间减半，使每层计算量大致均衡。
- Global average pooling：最后一张特征图每通道求平均，得到 $n_C$ 维向量再分类。
- 去掉大全连接层，减轻过拟合。

> [!NOTE]+ 衔接
>
> 卷积尺寸与反向已经对齐。直接堆很多层时训练误差可能上升。下一章的残差针对这条优化通路，加 Dropout 解决不了。

<br>

# 深度卷积

## 概念：深度卷积

- **问题**
  - 把卷积块堆深之后的优化、计算量与迁移。
- **范围**
  - 残差、Inception、深度可分卷积与预训练微调；
  - 输入仍是特征图，输出是更可训的深层表示。
- **章节衔接**
  - 上一章给出单层卷积；
  - 残差对应初始化章的连乘消失；
  - 迁移学习把策略章的“源任务大数据”落到卷积骨干上。

## 残差

> [!INFO]+ 残差
>
> - 是什么：块输出 $y=F(x)+x$，捷径把输入原样加上。
> - 来自：He et al., ResNet (2016)。针对“再加层，训练误差反而上升”。
> - 相对纯堆叠：反向多一条完整通路，最优映射接近恒等时 $F$ 只需学接近 0。
> - 修的是优化，不是过拟合；加 Dropout 解决不了这条训练误差上升。

<br>

- 深层网络直接堆叠会出现训练误差上升，原因是优化困难，不是过拟合。
- 残差块学习 $F(x)$，输出

$$
y=F(x)+x.
$$

- $F$ 与 $x$ 通道或尺寸不同时，用 $1\times1$ 卷积或步幅卷积做投影 $W_s x$。
- 恒等捷径使反向梯度多一条完整通路，减轻消失。
- 最优映射接近恒等时，$F$ 可学成接近 0，比直接学恒等映射容易。

```python
def residual_block_forward(X, conv1, conv2):
    """简化的两层 3x3 残差块，假设空间尺寸与通道已对齐。"""
    Z1, _ = conv2d_forward(X, conv1["W"], conv1["b"], stride=1, padding=1)
    A1 = np.maximum(Z1, 0)
    Z2, _ = conv2d_forward(A1, conv2["W"], conv2["b"], stride=1, padding=1)
    return np.maximum(Z2 + X, 0)
```

- ResNet-50/101 用 bottleneck：$1\times1$ 降维 $\rightarrow 3\times3 \rightarrow 1\times1$ 升维，降低 $3\times3$ 的通道成本。
- 例：输入通道 $C=256$，瓶颈宽 $C/4=64$。
- $3\times3$ 乘加从 $9\cdot 256\cdot 256$ 降到 $256\cdot 64 + 9\cdot 64\cdot 64 + 64\cdot 256$。
- 降采样块：第一层卷积 stride-2；捷径同步用 stride-2 的 $1\times1$。
- 作业：identity shortcut 与 convolutional shortcut 必须分支清楚，否则相加 shape 不对。

```python
def conv_bn_relu(X, W, b, gamma, beta, bn_cache, stride, padding, train):
    Z, _ = conv2d_forward(X, W, b, stride=stride, padding=padding)
    # 卷积 BatchNorm 对每个通道在 (m,H,W) 上求均值，这里用展平示意。
    n_C = Z.shape[-1]
    Zf = Z.reshape(-1, n_C).T
    Zb, _ = batchnorm_forward(Zf, gamma, beta, bn_cache, train=train)
    A = np.maximum(Zb.T.reshape(Z.shape), 0)
    return A
```

- ResNet-18 阶段：conv1 后 $56\times56$；layer1 保持；layer2-4 各减半空间、通道倍增。
- 实现时先让一个 block 的梯度检查通过，再堆 stage。

## Inception 与深度可分


> [!INFO]+ Inception 与深度可分
>
> - Inception（Szegedy et al.）：同一层并行多种核宽，再沿通道拼接，让网络选尺度。
> - $1\times1$ 先降维，否则 $5\times5$ 的通道乘加过贵。
> - MobileNet 深度可分：先逐通道空间卷积，再 $1\times1$ 混合通道，参数约为普通卷积的 $1/C+1/f^{2}$。
> - 相对 VGG 式一律 $3\times3$：用结构换计算量，而不是只靠堆层。

<br>

- Inception：同一层并行 $1\times1$、$3\times3$、$5\times5$ 与池化，再沿通道拼接，让网络选择尺度。
- $1\times1$ 卷积在通道维做线性组合，可降维以控制计算。

MobileNet 把标准卷积分解为：

- depthwise：每个输入通道单独 $f\times f$ 卷积；
- pointwise：$1\times1$ 混合通道。

- 计算量从 $f^{2} C_{\mathrm{in}} C_{\mathrm{out}}$ 降到 $f^{2} C_{\mathrm{in}}+C_{\mathrm{in}} C_{\mathrm{out}}$。
- 适合移动端。
- CS230 作业：用 MobileNet 做迁移学习。
  - 冻结卷积骨干；
  - 替换并训练分类头；
  - 必要时用小学习率微调后几层。
- DenseNet：每层输入接到后续所有层的拼接上。
  - 特征复用更强，参数更省；
  - 拼接占用显存；
  - 公开大纲列为选读。

## 迁移学习

> [!INFO]+ 迁移学习
>
> - 是什么：把在大数据源任务上训好的骨干接到新任务头上。
> - 来自：ImageNet 预训练成为视觉默认；CS230 作业用 MobileNet 做。
> - 相对从零训练：低层边缘与纹理核可复用，目标数据少时只训头。
> - 域差大（医疗、卫星）时仍常起步于预训练，但应更快解冻或换领域权重。

<br>

- ImageNet 预训练的低层核接近 Gabor 式边缘检测器，对多数自然图像可复用。
- 步骤：

1. 加载预训练骨干，去掉原分类头。
2. 加任务头，随机初始化。
3. 目标数据少：冻结骨干，只训头。
4. 目标数据多：解冻后几段，学习率比从头训练小一个数量级。

- 医学影像、卫星图与 ImageNet 域差大：仍常从预训练起步，但应更快解冻，或改用领域内预训练。
- 迁移不是免费性能：输入通道数不同（灰度、多光谱）时，第一层核需要适配或重训。

- Inception 计算控制：$5\times5$ 在通道很大时极贵，先 $1\times1$ 降到 $n_{\mathrm{bot}}$ 再做空间卷积。
- 参数从 $25 C C'$ 变为 $C n_{\mathrm{bot}}+25 n_{\mathrm{bot}} C'$。
- 作业不要求手写 Inception，但应能指出：并行分支空间尺寸必须一致才能拼接，因此 same padding 与步幅要匹配。
- MobileNet 的 depthwise 不混合通道；没有随后的 $1\times1$，网络无法组合红/绿边缘。
- 冻结 MobileNet 骨干时，输入预处理必须与预训练一致（尺寸、均值、缩放），否则等于给网络看另一种分布。
- 残差反向：$\partial J/\partial x = \partial J/\partial y \cdot (1 + \partial F/\partial x)$。
- 即使 $F$ 的雅可比很小，仍有 $1$ 这条路。
- 与 LSTM 的 $c^{<t>}=c^{<t-1>}+\cdots$ 同一思想：加法路径保梯度。

> [!NOTE]+ 衔接
>
> 骨干给出空间特征图。分类头假定一张图一个标签。下一章输出个数不固定：每个物体一个框。

<br>

# 检测

## 概念：检测

- **问题**
  - 一张图中多个物体的类别与边界框 $(b_x,b_y,b_h,b_w)$；
  - 与分类的差别是输出个数不固定，且含定位。
- **范围**
  - 从滑动窗口到 YOLO 风格的网格预测、IoU、非极大值抑制；
  - 输入图像，输出框列表。
- **章节衔接**
  - 上一章骨干输出特征图；
  - 分类假定一张图一个标签，检测输出个数不固定；
  - 下一章的人脸验证进一步把特征图变成可比较的嵌入。

## 滑动窗口

- 在多尺度窗口上跑分类器。
- 计算浪费：相邻窗口特征重复。
- 把全连接改为卷积：一次前向得到空间图上每点的分类，等价于密集滑动窗口。
- 仍难以精确框回归，且宽高比不灵活。

## IoU 与 NMS

> [!INFO]+ IoU 与 NMS
>
> - IoU：交并比，衡量两框重叠。检测正样本与 mAP 都用它，但阈值含义不同。
> - NMS（非极大值抑制）：同一物体的重复框里只留最高分。
> - 相对滑动窗口的原始输出：一张图会对同一目标给出多个高度重叠的框。
> - Soft-NMS 降低重叠框分数而不直接删除，减轻密集目标被误杀。

<br>

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

- 检测正样本通常要求与真值 IoU 超过 $0.5$。
- 非极大值抑制（NMS）：
  1. 丢掉置信度低于阈值的框。
  2. 选最高置信度框，删除所有与它 IoU 高于阈值（如 $0.5$）的同类框。
  3. 重复直至为空。
- NMS 默认一个物体对应一个框；物体密集重叠时会误删。
- Soft-NMS 等变体降低重叠框分数，而不是直接删除。

## YOLO

> [!INFO]+ YOLO
>
> - 是什么：一次前向同时回归框与类别，把图划成网格，中心落入的格子负责该物体。
> - 来自：Redmon et al. (2016)。相对两阶段检测（先提案再分类），延迟更低。
> - 相对滑动窗口：全图一张特征图出全部框，宽高比由 anchor 承担。
> - 作业要写清格子、anchor 与 NMS 的对应，而不是复现当时的 SOTA 精度。

<br>

- 把图划分为 $S\times S$ 网格。
- 物体中心落入的格子负责预测该物体。
- 每个格子预测 $B$ 个框及置信度，以及 $C$ 类条件概率。
- 输出张量形状约为

$$
S\times S\times\big(B\cdot(5+C)\big)
$$

或后来版本中把类别与框解耦。框常用相对格子的中心偏移与宽高。训练损失包含：

- 有物体格子的坐标回归（平方或平滑 $L_1$）；
- 置信度（有物体 / 无物体，后者权更小）；
- 分类交叉熵。

- Anchor box：预定义若干宽高比。
- 每个格子每个 anchor 预测一个框，解决同一格子两物体中心重叠。
- 匹配：与真值 IoU 最高的 anchor 负责该真值。

```python
def iou(box1, box2):
    """box = (x1, y1, x2, y2)，坐标为绝对像素或归一化均可。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def nms(boxes, scores, iou_thr=0.5, score_thr=0.6):
    """boxes: (N, 4), scores: (N,)。返回保留的下标。"""
    keep = []
    order = np.argsort(-scores)
    order = [i for i in order if scores[i] >= score_thr]
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thr]
    return keep
```

- mAP：对每类按置信度排序，算 precision-recall 曲线下面积，再对类平均。
- IoU 阈值集合（如 $0.5$ 到 $0.95$）决定定位严格程度。
- YOLO 训练：每个真值框只匹配一个格子（中心所在格）和一个 anchor（IoU 最大者）。
- 无物体格子的置信度损失权重更小，否则背景淹没梯度。
- 坐标：sigmoid 把中心约束在格内；指数把 anchor 宽高缩放到正数。
- 损失是多项之和；权重不匹配会出现“分类还行、框乱飘”或反过来。
- 滑动窗口转卷积：分类器输入 $14\times14$ 时，全连接改为 $1\times1$ 卷积，更大图像一次前向得到空间分数图，等价于所有对齐窗口。
- 这解释了检测网络为何是全卷积的。
- Anchor 与特征图步幅共同决定最小可检物体。
- 物体小于一个格子：中心仍只能进一个格，但回归难度上升。

> [!EXAMPLE]+ 例7 IoU 与 NMS 的数值
>
> 框 $A=[0,0,10,10]$，$B=[0,0,10,8]$，交集面积 $80$，并集 $100+80-80=100$，$\mathrm{IoU}=0.8$。若 NMS 阈值为 $0.5$ 且二者同类，低分框被删。若其实是并排两辆车，阈值过低会误删，这是密集场景换 Soft-NMS 或更小阈值的原因。阈值不是 IoU 评估里的 $0.5$ 同一件事：一个用于后处理，一个用于算 mAP。

<br>


## 分割

> [!INFO]+ U-Net
>
> - 是什么：编码器下采样、解码器上采样，并用 skip 把高分辨率特征拼回去。
> - 来自：Ronneberger et al. (2015)，面向生物医学分割。
> - 相对只做编码器再插值：边缘与定位来自 skip，不是来自最后一层语义。
> - skip 是通道拼接，与残差的相加不同。

<br>

- 语义分割（semantic segmentation）：对每个像素分类。
- U-Net：编码器下采样抓上下文；解码器上采样恢复空间。
- skip connection：把编码器的高分辨率特征接到对应解码层，保留边缘。
- 损失：逐像素交叉熵；类别不平衡时用加权或 Dice。
- 实例分割还要区分同类不同物体，超出 CS230 作业范围。
- 评估不能把 IoU 与像素准确率混用：后者会被大背景类主导。
- skip：编码器第 $k$ 个下采样前的特征拼到解码器对应上采样后。
- 通道维拼接，而非相加（与残差不同）。
- 上采样：转置卷积，或插值加 $3\times3$。
- 医学分割正类像素很少，交叉熵会偏向背景。
- Dice

$$
1-\frac{2\sum_p \hat{y}_p y_p}{\sum_p \hat{y}_p+\sum_p y_p+\varepsilon}
$$

- Dice 直接优化重叠。
- 作业重点：对称裁剪或 padding，使 skip 两端空间尺寸一致，否则无法 `concatenate`。

> [!NOTE]+ 衔接
>
> 检测输出框与类别；人脸验证学的是度量，不是 $K$ 类 softmax。风格迁移甚至不训练网络，只优化像素。二者都建立在卷积特征上。

<br>

# 人脸与风格

## 概念：人脸与风格

- **问题**
  - 验证 / 识别需要可比较的嵌入；
  - 风格迁移需要可分离的内容特征与 Gram 统计。
- **范围**
  - triplet loss 与阈值 $\tau$；
  - 内容损失（content loss）、风格损失（style loss）与对 $G$ 的梯度。
- **章节衔接**
  - 上一章的卷积骨干提供 $f(x)$ 或 $a^{[l]}$；
  - 嵌入思想下一章会在词向量上重现；
  - 优化像素而不是权重，与对抗章对输入求梯度同类。

## 人脸验证

> [!INFO]+ Siamese 与 triplet
>
> - 是什么：共享权重的双塔把图映成嵌入；triplet 要求锚点离正例近、离负例至少远 $\alpha$。
> - 来自：Chopra et al. 的 Siamese；FaceNet (Schroff et al., 2015) 用 triplet。
> - 相对 $K$ 类 softmax：库里加人不必重训分类头，只需加入新嵌入。
> - 随机负例大多已经满足间隔，必须做半困难负例挖掘。

<br>

- 验证：输入两张图，判断是否同一人。
- 识别：在库中找匹配。
- 二者都需要度量学习（metric learning），不是每次用 $K$ 类 softmax（$K$ 会随新员工变化）。

- Siamese：共享权重的 CNN 把脸映射为 $d$ 维嵌入 $f(x)$。
- 同一人的 $\|f(x^{(i)})-f(x^{(j)})\|_2^{2}$ 应较小，不同人应较大。
- 二元监督可用对比损失。
- FaceNet 使用 triplet loss。
- 三元组 $(A,P,N)$：锚点、正例、负例。

$$
\mathcal{L}=\max\big(\|f(A)-f(P)\|_2^{2}-\|f(A)-f(N)\|_2^{2}+\alpha,0\big).
$$

- $\alpha>0$ 为间隔。
- 嵌入通常 $L_2$ 归一化到单位球面。
- 三元组挖掘很关键：随机负例大多已经满足间隔，梯度为 0。
- 应选半困难负例：比正例远，但仍在间隔内。

```python
def triplet_loss(fA, fP, fN, alpha=0.2):
    """f*: (m, d)，已 L2 归一化更好。"""
    pos = np.sum((fA - fP) ** 2, axis=1)
    neg = np.sum((fA - fN) ** 2, axis=1)
    return np.maximum(pos - neg + alpha, 0.0).mean()
```

- 验证时比较 $\|f(x)-f(x_{\mathrm{db}})\|$ 与阈值 $\tau$。
- $\tau$ 在验证集上用 FAR/FRR 或 EER 选择。
- 光照、姿态、年龄会造成嵌入漂移；部署采集协议须与训练分布对齐。
- 验证（verification）：一对一，门禁刷脸。
- 识别（recognition）：一对多，在库中找最近邻。
- 二者共享嵌入，不共享决策。
- 库新增人员不必重训 softmax 头，只需加入新嵌入；这是用度量学习而不是 $K$ 类分类器的工程原因。
- 三元组构造：每个锚点的正例来自同一身份的另一张图。
- 简单负例已满足间隔，损失为 0。
- 困难负例 $\|f(A)-f(N)\|<\|f(A)-f(P)\|$：往往是噪声或标注错。
- 半困难负例位于二者之间，梯度最有信息。
- 在线挖掘：一个 mini-batch 内选；离线挖掘：周期更新。
- 都要防止坍缩到所有嵌入为零（尺度任意时间隔仍可被满足）。
- $L_2$ 归一化把嵌入钉在球面，消除尺度作弊。

```python
def embed_and_verify(xa, xb, network, tau):
    """network 返回 (d,) 嵌入。"""
    ea, eb = network(xa), network(xb)
    ea = ea / np.linalg.norm(ea)
    eb = eb / np.linalg.norm(eb)
    return float(np.sum((ea - eb) ** 2)) < tau
```


## 神经风格迁移

> [!INFO]+ 风格迁移
>
> - 是什么：固定预训练 CNN，把生成图 $G$ 当像素变量优化。
> - 来自：Gatys et al. (2016)。内容用某层特征，风格用 Gram 矩阵。
> - 相对再训一个生成网络：不更新 $W$，只对 $G$ 求梯度，与对抗章对 $x$ 求梯度同类。
> - Gram 捕捉通道相关（纹理），不锁定空间布局。

<br>

- 把内容图 $C$ 的结构与风格图 $S$ 的纹理合成到生成图 $G$。
- $G$ 是像素变量，不是网络权重；CNN 只提供特征。
- 内容损失用某层特征响应：

$$
J_{\mathrm{content}}(C,G)=\frac{1}{2}\|a^{[l]}(C)-a^{[l]}(G)\|_F^{2}.
$$

- 风格用 Gram 矩阵 $G^{[l]}=A^{[l]}A^{[l]\top}$。
- $A^{[l]}$ 把该层特征图展成 $n_C\times (n_H n_W)$：

$$
J_{\mathrm{style}}^{[l]}(S,G)=\frac{1}{2}\|G^{[l]}(S)-G^{[l]}(G)\|_F^{2}.
$$

总损失：

$$
J=\alpha J_{\mathrm{content}}+\beta\sum_{l}\lambda_l J_{\mathrm{style}}^{[l]}.
$$

- 对 $G$ 做梯度下降（可加 total variation 平滑）。
- 浅层偏纹理，深层偏大结构。
- 这是优化问题不是训练问题：每张图都要重新迭代。

```python
def gram_matrix(A):
    """A: (n_C, n_H * n_W)。"""
    return A @ A.T

def style_loss(Gs, Gg):
    n_C = Gs.shape[0]
    return np.sum((Gs - Gg) ** 2) / (4.0 * n_C ** 2)

def total_style_loss(style_grams, gen_acts, weights):
    """style_grams: 层名 -> Gram(S)；gen_acts: 层名 -> A(G)。"""
    loss = 0.0
    for layer, w in weights.items():
        A = gen_acts[layer]
        n_C, n_HW = A.shape[0], A.shape[1]
        Gg = gram_matrix(A) / n_HW
        Gs = style_grams[layer] / n_HW
        loss += w * style_loss(Gs, Gg)
    return loss
```

- 预训练 CNN 冻结，只对像素求 $\partial J/\partial G$。
- 初始化 $G$：内容图加噪声。
- 迭代不够：风格未融合。
- $\beta/\alpha$ 过大：内容结构崩坏。
- 与训练分类器的“epoch”不是同一件事：每张新图都从零优化 $G$。

> [!NOTE]+ 对照：验证、识别、分类
>
> - $K$ 类 softmax：类别集合固定。
> - 验证：一对一，比距离与 $\tau$。
> - 识别：一对多，在库中找最近邻；库可增员而不重训头。
> - 风格迁移：变量是像素 $G$，CNN 冻结。

<br>

> [!NOTE]+ 衔接
>
> 卷积共享的是空间。序列的默认共享在时间轴。下一章的 $W_{aa}$ 对每个 $t$ 重复使用，对应卷积核对每个位置重复使用。

<br>

# 循环网络

## 概念：循环

- **问题**
  - 变长序列 $x^{<1>},\ldots,x^{<T_x>}$ 到输出序列或单一标签；
  - 用隐状态 $a^{<t>}$ 压缩到当前步为止的历史。
- **范围**
  - 写出 RNN、GRU、LSTM 的前向与时间反向传播；
  - 输入序列张量，输出每步预测或最后一步表示。
- **章节衔接**
  - 上一章的权值共享在空间；本章改到时间；
  - 循环共享时间上的权重，类似卷积共享空间上的权重；
  - 长程依赖（long-range dependency）困难引出门控（gating）；固定向量瓶颈引出再下一章的注意力。

## RNN 单元

> [!INFO]+ RNN
>
> - 是什么：同一组 $W_{aa},W_{ax}$ 在每个时间步重复使用，隐状态压缩到当前步为止的历史。
> - 来自：Elman 等早期循环网络；CS230 作业从零实现字符级语言模型。
> - 相对把变长序列 pad 成固定向量再全连接：参数不随 $T$ 增长，能处理不等长输入。
> - 代价：时间维串行；长程梯度经反复相乘易消失，后面用门控与注意力补。

<br>

$$
\begin{aligned}
a^{<t>} &= g\big(W_{aa}a^{<t-1>}+W_{ax}x^{<t>}+b_a\big),\\
\hat{y}^{<t>} &= g_y\big(W_{ya}a^{<t>}+b_y\big).
\end{aligned}
$$

- $g$ 常用 tanh。
- 可等价写成把 $[a^{<t-1>}; x^{<t>}]$ 拼起来乘 $W_a$。
- 参数不随 $t$ 改变。
- $a^{<0>}$ 通常为零向量。

架构变体：

- many-to-many，等长：每步都有标签，如词性标注。
- many-to-many，不等长：编码器-解码器（encoder-decoder），如翻译。
- many-to-one：情感分类，只用 $a^{<T>}$。
- one-to-many：音乐生成，从种子或零输入展开。

- 字符级语言模型：$\hat{y}^{<t>}$ 是词表上的 softmax。
- 训练：每步交叉熵。
- 生成：从 $\hat{y}^{<t>}$ 采样（或取 argmax）作为下一步输入。
- 温度 $T$：logits 除以 $T$ 再 softmax。
- $T>1$ 更均匀；$T\rightarrow0$ 接近贪心。

```python
def rnn_cell_forward(xt, a_prev, params):
    """xt: (n_x, m), a_prev: (n_a, m)。"""
    Wax, Waa, Wya = params["Wax"], params["Waa"], params["Wya"]
    ba, by = params["ba"], params["by"]
    a_next = np.tanh(Wax @ xt + Waa @ a_prev + ba)
    yt_pred = softmax(Wya @ a_next + by)
    return a_next, yt_pred, (a_next, a_prev, xt, params)

def rnn_forward(X, a0, params):
    """X: (n_x, m, T)。返回每步预测列表与缓存。"""
    n_x, m, T = X.shape
    n_y = params["Wya"].shape[0]
    a = a0
    y_preds, caches = [], []
    for t in range(T):
        a, yt, cache = rnn_cell_forward(X[:, :, t], a, params)
        y_preds.append(yt)
        caches.append(cache)
    return np.stack(y_preds, axis=-1), caches
```

## BPTT（backpropagation through time）

- 损失 $J=\sum_t \mathcal{L}^{<t>}$。
- $W_{aa}$ 的梯度来自所有时刻：同一矩阵被反复使用。
- 从 $t=T$ 向 $t=1$ 回传 $\mathrm{d}a^{<t>}$。
- 连乘 $\prod_{k} W_{aa}^{\top}\mathrm{diag}(1-(a^{<k>})^{2})$ 导致：

- 谱半径 $>1$：爆炸，用梯度裁剪 $\mathrm{clip}(g, -\tau, \tau)$ 或全局范数裁剪；
- 谱半径 $<1$：消失，早期步对 $J$ 几乎无贡献，无法学长距离依赖。

```python
def clip_gradients(grads, max_value=5.0):
    for k in grads:
        np.clip(grads[k], -max_value, max_value, out=grads[k])
    return grads

def rnn_cell_backward(da_next, cache):
    """单步 BPTT。da_next 含本步输出误差与未来回传。"""
    a_next, a_prev, xt, params = cache
    Wax, Waa, Wya = params["Wax"], params["Waa"], params["Wya"]
    dtanh = (1.0 - a_next ** 2) * da_next
    dWax = dtanh @ xt.T
    dWaa = dtanh @ a_prev.T
    dba = dtanh.sum(axis=1, keepdims=True)
    dxt = Wax.T @ dtanh
    da_prev = Waa.T @ dtanh
    return {"dWax": dWax, "dWaa": dWaa, "dba": dba}, dxt, da_prev
```

- 整段 BPTT：从 $t=T-1$ 走到 $t=0$。
- 各步 `dWax, dWaa` 求和（权值共享）。
- `da_prev` 传入前一步。
- 字符级作业里 $T$ 可达数百，必须裁剪。
- 恐龙名字生成作业用采样而不是 teacher forcing，演示是否学会合法字符转移。

```python
def sample_char_model(params, seed_idx, ix_to_char, n_chars, rng, temperature=1.0):
    """从训练好的字符 RNN 采样一段文本。"""
    n_a = params["Waa"].shape[0]
    n_x = params["Wax"].shape[1]
    a = np.zeros((n_a, 1))
    x = np.zeros((n_x, 1))
    x[seed_idx] = 1.0
    out = [ix_to_char[seed_idx]]
    for _ in range(n_chars):
        a, y, _ = rnn_cell_forward(x, a, params)
        logits = np.log(np.clip(y.ravel(), 1e-12, 1.0)) / temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        idx = int(rng.choice(n_x, p=p))
        out.append(ix_to_char[idx])
        x = np.zeros((n_x, 1))
        x[idx] = 1.0
        if ix_to_char[idx] == "\n":
            break
    return "".join(out)
```

## GRU

> [!INFO]+ GRU
>
> - 是什么：用更新门与重置门控制写入与读取，隐状态本身兼记忆。
> - 来自：Cho et al. (2014)，为 seq2seq 提出。
> - 相对朴素 RNN：$\Gamma_u\approx 0$ 时沿时间拷贝旧状态，梯度有高速路。
> - 参数比 LSTM 少，作业与 LSTM 二选一即可，不必两套都从零调到最优。

<br>

- 用更新门决定保留多少旧状态，减轻消失梯度。
- 令 $[x^{<t>}; a^{<t-1>}]$ 为输入：

$$
\begin{aligned}
\Gamma_u &= \sigma(W_u[a^{<t-1>},x^{<t>}]+b_u),\\
\Gamma_r &= \sigma(W_r[a^{<t-1>},x^{<t>}]+b_r),\\
\tilde{c}^{<t>} &= \tanh\big(W_c[\Gamma_r\odot a^{<t-1>},x^{<t>}]+b_c\big),\\
a^{<t>} &= \Gamma_u\odot\tilde{c}^{<t>}+(1-\Gamma_u)\odot a^{<t-1>}.
\end{aligned}
$$

- $\Gamma_u\approx1$：写入新候选。
- $\Gamma_u\approx0$：拷贝旧状态，梯度可沿时间高速路传递。
- $\Gamma_r$：控制候选计算时看见多少历史。

## LSTM

> [!INFO]+ LSTM
>
> - 是什么：单独的细胞状态 $c^{<t>}$，加遗忘、输入、输出三个门。
> - 来自：Hochreiter & Schmidhuber (1997)。
> - 相对 GRU：记忆与输出解耦，$c$ 上主要是加法和门控，更适合很长的依赖。
> - 作业默写六式；实现错误常见于门的拼接顺序与 `c` / `a` 谁进入下一门。

<br>

- 增加记忆单元 $c^{<t>}$。
- 三个门：遗忘 $\Gamma_f$、输入 $\Gamma_i$、输出 $\Gamma_o$。

$$
\begin{aligned}
\Gamma_f &= \sigma(W_f[a^{<t-1>},x^{<t>}]+b_f),\\
\Gamma_i &= \sigma(W_i[a^{<t-1>},x^{<t>}]+b_i),\\
\tilde{c}^{<t>} &= \tanh(W_c[a^{<t-1>},x^{<t>}]+b_c),\\
c^{<t>} &= \Gamma_f\odot c^{<t-1>}+\Gamma_i\odot\tilde{c}^{<t>},\\
\Gamma_o &= \sigma(W_o[a^{<t-1>},x^{<t>}]+b_o),\\
a^{<t>} &= \Gamma_o\odot\tanh(c^{<t>}).
\end{aligned}
$$

- 遗忘门：把 $c^{<t-1>}$ 的部分清零。
- 输入门：写入候选。
- 输出门：决定暴露多少记忆到 $a^{<t>}$。
- $c$ 的加法更新是长程信息的主通路。
- 把 $b_f$ 初始化为小正数，有助于早期偏向“先记住”。

```python
def lstm_cell_forward(xt, a_prev, c_prev, params):
    concat = np.vstack([a_prev, xt])
    gf = sigmoid(params["Wf"] @ concat + params["bf"])
    gi = sigmoid(params["Wi"] @ concat + params["bi"])
    go = sigmoid(params["Wo"] @ concat + params["bo"])
    cct = np.tanh(params["Wc"] @ concat + params["bc"])
    c_next = gf * c_prev + gi * cct
    a_next = go * np.tanh(c_next)
    return a_next, c_next, (a_next, c_next, a_prev, c_prev, concat, gf, gi, go, cct, xt)
```

- 双向 RNN：分别跑正向与反向，拼接 $[a^{\rightarrow <t>}; a^{\leftarrow <t>}]$。
- 适合标注任务，不适合在线因果生成。
- 多层 RNN：下层 $a^{<t>}$ 当作上层输入。

> [!INFO]+ Teacher forcing
>
> - 是什么：训练时解码器每步吃真值上一步，而不是自己的预测。
> - 来自：Williams & Zipser 的 teacher forcing；seq2seq 作业默认这样训。
> - 相对用模型自己的输出做输入：收敛快，梯度稳定。
> - 代价：训练与推理分布不一致；推理第一步错会级联。

<br>

- Teacher forcing：训练时解码器输入真值上一步，不是自己的预测；加快收敛，但造成训练-推理落差。
- LSTM 反向：对 $c^{<t>}$ 与 $a^{<t>}$ 分别回传。
- $\partial c^{<t>}/\partial c^{<t-1>}=\Gamma_f$；遗忘门接近 1 时梯度几乎原样传递。
- 门的饱和（$\sigma$ 两端）仍会削弱更新。
- GRU 参数更少；序列不是极长时与 LSTM 往往接近。
- 作业要求能默写门的形状：每个门矩阵是 $(n_a, n_a+n_x)$。
- 爵士乐作业：和弦与音高序列送入 LSTM，输出下一步音符的 softmax。
- 评估不能只看交叉熵：还要听是否跑调、是否只会重复短动机。
- 生成模型的损失下降不等于样本质量；GAN 与 LLM 仍然成立。

> [!NOTE]+ 衔接
>
> RNN 的输入 $x^{<t>}$ 若是 one-hot，内积恒为 0。下一章用嵌入表 $E$ 把离散记号变成可比较的向量，再送进 RNN 或平均后分类。

<br>

# 词嵌入

## 概念：嵌入

> [!INFO]+ 词嵌入
>
> - 是什么：把词号换成稠密向量，使类比在向量空间里近似线性。
> - 来自：word2vec（Mikolov et al., 2013）的 Skip-gram / CBOW；GloVe 用共现矩阵。
> - 相对 one-hot：内积不再恒为 0，相似词可靠近。
> - 负采样用来避开词表上的满 softmax；嵌入会复制语料里的偏见。

<br>

- **问题**
  - 词的离散编号无法表达相似；
  - 用稠密向量 $e\in\mathbb{R}^{d}$ 使类比关系近似线性。
- **范围**
  - word2vec / GloVe 的目标直觉、OOV、偏见；
  - 输入词表 index，输出嵌入矩阵 $E$。
- **章节衔接**
  - 上一章的 RNN 吃的是向量，不是词号；
  - 本章把离散记号变成可加减的向量；
  - 下一章用注意力在变长源序列上取加权，而不是只靠最后一个隐状态。

- one-hot $o_j\in\mathbb{R}^{|V|}$：内积恒为 0。
- 嵌入表 $E\in\mathbb{R}^{d\times |V|}$，$e=E o_j$ 即取第 $j$ 列。
- 语言模型、情感分类、翻译都可以把 $E$ 当作第一层，可冻结或微调。
- Skip-gram：由中心词预测窗口内上下文。
- softmax 在大词表上昂贵，用负采样（negative sampling）近似。
- CBOW：由上下文预测中心词。
- GloVe：对共现对数做加权最小二乘，使 $w_i^{\top}w_j\approx\log X_{ij}$。
- 训练后常见类比：

$$
e_{\mathrm{king}}-e_{\mathrm{man}}+e_{\mathrm{woman}}\approx e_{\mathrm{queen}}.
$$

- 评价：余弦相似度。
- OOV：子词模型（BPE、WordPiece）或随机向量。
- CS230 作业含词向量去偏：对性别子空间做投影消除，同时检查是否破坏中性语义。
- 嵌入会编码语料中的社会偏见；用于招聘或刑侦时必须单独审计。

```python
def cosine(u, v, eps=1e-12):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + eps))

def analogy(e, a, b, c, words):
    """解 a:b :: c:?  即 argmax cos(e[w], e[b]-e[a]+e[c])。"""
    target = e[b] - e[a] + e[c]
    best, best_s = None, -1.0
    banned = {a, b, c}
    for w in words:
        if w in banned:
            continue
        s = cosine(e[w], target)
        if s > best_s:
            best, best_s = w, s
    return best, best_s
```

- 情感分类可先把句中词向量平均后接逻辑回归，作为强基线。
- 再对比 LSTM 或 Transformer 是否真有增益。

- Skip-gram 精确目标：窗口半径 $c$，中心词 $w_t$，

$$
\sum_{t}\sum_{\substack{-c\le j\le c\\ j\neq 0}}\log P(w_{t+j}\mid w_t),\qquad
P(w_O\mid w_I)=\frac{\exp(v_{w_O}^{\prime\top} v_{w_I})}{\sum_{w}\exp(v_w^{\prime\top} v_{w_I})}.
$$

- 分母遍历词表，计算 $O(|V|)$。
- 负采样：对一对 $(w_I,w_O)$ 与 $k$ 个噪声词做逻辑回归。

$$
\log\sigma(v_{w_O}^{\prime\top}v_{w_I})+\sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}[\log\sigma(-v_{w_i}^{\prime\top}v_{w_I})].
$$

- $P_n(w)\propto f(w)^{3/4}$：压低超高频词。
- GloVe 的加权最小二乘对稀有共现降权，避免 $\log 0$。
- 去偏：估计性别子空间方向 $g$，例如 $\frac{1}{2}[(e_{\mathrm{he}}-e_{\mathrm{she}})+(e_{\mathrm{man}}-e_{\mathrm{woman}})]$。
- 中性词减去 $\mathrm{proj}_g e$；成对词做均衡。
- 作业检查：中性职业词与性别方向的余弦应下降；`king-queen` 类比不应崩溃。
- 去偏不保证下游公平，只是表示空间上的线性修正。

```python
def neutralize(e, gender_dir):
    """去掉 e 在性别方向上的分量。"""
    g = gender_dir / np.linalg.norm(gender_dir)
    return e - (e @ g) * g

def equalize(e1, e2, gender_dir):
    """让成对词到性别方向的距离对称。"""
    g = gender_dir / np.linalg.norm(gender_dir)
    mu = (e1 + e2) / 2.0
    mu_b = (mu @ g) * g
    mu_orth = mu - mu_b
    e1_orth = e1 - (e1 @ g) * g
    e2_orth = e2 - (e2 @ g) * g
    e1_new = mu_orth + np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * (e1_orth / np.linalg.norm(e1_orth))
    e2_new = mu_orth + np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * (e2_orth / np.linalg.norm(e2_orth))
    return e1_new, e2_new
```

- Emojify：句子平均嵌入送入 softmax，再换成 LSTM。
- 若平均嵌入已经够用，加 LSTM 的增益应在误差分析里写明（否定词、词序），否则是多余复杂度。

> [!NOTE]+ 衔接
>
> 嵌入解决了离散符号。seq2seq 仍把整句压进 $a^{<T_x>}$。下一章让解码器每步对源隐状态取加权和，不再依赖这一个向量。

<br>

# 注意力

## 概念：注意力

- **问题**
  - 编码器隐状态序列过长，单一向量 $a^{<T_x>}$ 成为瓶颈；
  - 解码器每步从源序列取加权和。
- **范围**
  - Bahdanau 注意力与 seq2seq；
  - 输入源隐状态与当前解码状态，输出上下文向量。
- **章节衔接**
  - 上一章的嵌入进入编码器；
  - RNN 章的 $a^{<T_x>}$ 是瓶颈；
  - 下一章把注意力改成自注意力，并去掉循环。

## Seq2seq

> [!INFO]+ Seq2seq
>
> - 是什么：编码器读完整源序列，解码器自回归生成目标序列。
> - 来自：Sutskever et al. (2014)；Cho et al. 同期的编码器-解码器。
> - 相对固定输出长度的 many-to-one：源与目标长度可以不同。
> - 整句压进最后一个隐状态，长句会丢前面的信息，下一节用注意力拆这个瓶颈。

<br>

- 编码器 RNN 读入源句；解码器 RNN 生成目标句。
- 训练用 $<\mathrm{SOS}>$、`<EOS>` 定界。
- 推理：贪心或束搜索（beam search）。
- 束宽 $B$：每步保留 $B$ 条最高对数概率路径。
- 比较路径时应除以长度惩罚（指数约 $0.7$），否则短句占优：

$$
\mathrm{score}(y)=\frac{1}{T_y^{\alpha}}\sum_{t=1}^{T_y}\log P(y^{<t>}\mid y^{<t-1>},x).
$$

BLEU 用修正的 $n$-gram 精确率与短句惩罚衡量翻译，对同义改写不敏感，只是自动代理。

```python
def beam_search_step(prev_live, logp_step, beam_width):
    """prev_live: list[(seq_tuple, logprob)]；logp_step[i] 为第 i 条活路在词表上的 log softmax。"""
    cand = []
    for i, (seq, lp) in enumerate(prev_live):
        topk = np.argpartition(logp_step[i], -beam_width)[-beam_width:]
        for tok in topk:
            cand.append((seq + (int(tok),), lp + float(logp_step[i, tok])))
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[:beam_width]

def length_norm(logprob, length, alpha=0.7):
    return logprob / (((5 + length) / 6) ** alpha)
```

- 束搜索是启发式，不保证最大似然句。
- $B=1$ 即贪心。
- $B$ 过大则短句与重复 $n$-gram 需要额外惩罚。


## Bahdanau 注意力

> [!INFO]+ 注意力
>
> - 是什么：解码每步对源隐状态做加权和，权来自当前解码状态与各源位置的能量。
> - 来自：Bahdanau et al. (2015)，为神经机器翻译提出。
> - 相对 seq2seq 的单一向量 $a^{<T_x>}$：信息不必挤进固定长度瓶颈。
> - Luong 后来改了能量函数；作业实现 Bahdanau 即可。高权重不是因果证明。

<br>

解码步 $t$，源步 $t'$ 的能量

$$
e^{<t,t'>}=v^{\top}\tanh\big(W_a[s^{<t-1>};h^{<t'>}]\big),
$$

$$
\alpha^{<t,t'>}=\frac{\exp(e^{<t,t'>})}{\sum_{k=1}^{T_x}\exp(e^{<t,k>})},\qquad
c^{<t>}=\sum_{t'=1}^{T_x}\alpha^{<t,t'>}h^{<t'>}.
$$

- $s^{<t>}$ 由解码器在 $[c^{<t>}; y^{<t-1>}]$ 上更新。
- 注意力权重可视化可检查对齐，但高权重不是因果证明。

> [!INFO]+ 注意力是期望
>
> $\alpha^{<t,\cdot>}$ 是源位置上的离散分布，$c^{<t>}$ 是隐状态的期望。与硬对齐（每次只选一个源词）相比，软注意力可微。复杂度每解码步 $O(T_x d)$，总 $O(T_y T_x d)$。源或目标很长时这仍贵，但比让单一向量承担全部记忆要稳。

<br>

```python
def bahdanau_attention(s_prev, H, Wa, va):
    """s_prev: (n_s, m), H: (n_h, m, T_x)。返回 context (n_h, m) 与 alpha (T_x, m)。"""
    n_h, m, T_x = H.shape
    s_exp = np.repeat(s_prev[:, :, None], T_x, axis=2)
    concat = np.concatenate([s_exp, H], axis=0)
    n_cat = concat.shape[0]
    e = np.tanh(np.einsum("ij,jmt->imt", Wa, concat))
    scores = np.einsum("i,imt->mt", va.ravel(), e)
    scores -= scores.max(axis=1, keepdims=True)
    alpha = np.exp(scores)
    alpha /= alpha.sum(axis=1, keepdims=True)
    context = np.einsum("mt,hmt->hm", alpha, H)
    return context, alpha.T
```

- 触发词：注意力或 RNN 输出接到每帧二分类。
- 正类极稀疏：需加权损失或下采样静音段。
- CTC：允许输出与输入不对齐；用空白符与动态规划边缘化对齐。
- CS230 课堂曾作为进阶讲座出现。

- BLEU 的核心是修正 $n$-gram 精确率。
- 对 $n=1,\ldots,N$（常 $N=4$），

$$
p_n=\frac{\sum_{\mathrm{clip}}\mathrm{count}(\mathrm{ngram})}{\sum \mathrm{count}_{\mathrm{sys}}(\mathrm{ngram})},
$$

- 短句惩罚：输出长 $c$ 小于参考长 $r$ 时 $\mathrm{BP}=\exp(1-r/c)$，否则为 1。

$$
\mathrm{BLEU}=\mathrm{BP}\cdot\exp\Big(\frac{1}{N}\sum_{n=1}^{N}\log p_n\Big).
$$

- clip：每个 n-gram 计数不超过参考中出现次数，防止“the the the”刷精确率。
- BLEU 不理解同义，不能单独作为项目唯一指标。
- Luong：$e^{<t,t'>}=s^{<t>\top} W h^{<t'>}$（双线性）或点积 $s^{\top}h$，不必再过一层 tanh。
- 与 Bahdanau 的差别：能量函数，以及 $s$ 用 $t$ 还是 $t-1$。
- 作业实现 Bahdanau 即可；复杂度同为 $O(T_x T_y d)$。
- 翻译作业的 teacher forcing 使训练损失偏低；推理时第一步就错，后续会级联。
- Scheduled sampling：以概率把真值换成模型预测；是缓解手段，不是作业必做。
- 触发词标签：每 $10\,\mathrm{ms}$ 一帧的 $0/1$。
- 正帧极少：损失写成 $w_1 y\log\hat{y}+w_0(1-y)\log(1-\hat{y})$，或只在正例附近保留负例。
- 评估用检测延迟与误触发率，不用帧级准确率。后者会因全预测 0 而虚高。

> [!NOTE]+ 衔接
>
> 注意力仍依附 RNN 编码器-解码器，时间维串行。下一章让 $Q,K,V$ 都来自同一序列（自注意力），去掉循环，用位置编码补回顺序。

<br>

# Transformer

架构、多头与位置编码的展开见 [Transformer论文精读](https://yukinoshitasherry.github.io/Transformer/)。

## 概念：Transformer

> [!INFO]+ Transformer
>
> - 是什么：用自注意力和位置编码代替循环，一层内任意两个位置可以直接交互。
> - 来自：Vaswani et al., *Attention Is All You Need* (2017)。
> - 相对 RNN + Bahdanau：训练可并行，感受野不再随层沿时间累积。
> - 代价：$O(T^{2}d)$；顺序必须显式注入。细节见精读笔记。

<br>

- **问题**
  - 去掉循环，用自注意力在序列内部做全局交互，并用位置编码注入顺序；
  - 机器翻译上的编码器-解码器，以及后来的仅解码器语言模型。
- **范围**
  - 写出 scaled dot-product、多头、残差+LayerNorm、位置编码与掩码；
  - 输入 token 嵌入，输出下一 token 的 logits 或编码器表示。
- **章节衔接**
  - 自注意力是注意力中 query、key、value 都来自同一序列的情形；
  - 课堂 LLM 应用默认建立在 Transformer 之上。

## Scaled dot-product

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V.
$$

$Q,K\in\mathbb{R}^{T\times d_k}$，$V\in\mathbb{R}^{T\times d_v}$。不除 $\sqrt{d_k}$ 时，点积方差约为 $d_k$，softmax 进入饱和，梯度消失。除以 $\sqrt{d_k}$ 使尺度与 $d_k$ 无关。

```python
def scaled_dot_product_attention(Q, K, V, mask=None, eps=1e-9):
    """Q,K,V: (..., T, d)。mask 中 0 表示禁止位置。"""
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + eps)
    return weights @ V, weights
```

## 多头

- 把 $d_{\mathrm{model}}$ 拆成 $h$ 个头，每头 $d_k=d_{\mathrm{model}}/h$：

$$
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V),
$$

$$
\mathrm{MultiHead}=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O.
$$

- 不同头可以学到不同对齐，例如句法、共指、位置邻近。
- 实现上通常一次线性投射再 reshape 为 $(h,T,d_k)$，不必写 $h$ 个独立矩阵循环。

## 块结构

- 编码器一层：多头自注意力，残差与 LayerNorm，再两层逐位置前馈。

$$
\mathrm{FFN}(x)=\max(0, xW_1+b_1)W_2+b_2.
$$

$$
\begin{aligned}
z &= \mathrm{LayerNorm}\big(x+\mathrm{MultiHead}(x,x,x)\big),\\
y &= \mathrm{LayerNorm}\big(z+\mathrm{FFN}(z)\big).
\end{aligned}
$$

- LayerNorm 沿特征维标准化，与 batch 大小无关，适合变长序列。
- Pre-LN 与 Post-LN 的稳定性不同；原论文为 Post-LN。
- 解码器每层多一块：对编码器输出的交叉注意力，以及带因果掩码（causal mask）的自注意力。
- 因果掩码使位置 $t$ 看不见 $t'>t$，训练可以并行，并与自回归推理一致。

## 位置编码

> [!INFO]+ 位置编码
>
> - 是什么：把位置信息加到嵌入上，因为自注意力对置换等变。
> - 来自：原论文的正弦 / 余弦；后来常见可学习嵌入、相对偏置、RoPE。
> - 相对依赖 RNN 的递推顺序：并行计算时顺序必须显式写入。
> - 正弦编码在长度外推上通常优于绝对可学习嵌入。

<br>

- 自注意力置换等变，必须加入位置信息。
- 原论文用正弦：

$$
\mathrm{PE}(pos,2i)=\sin\big(pos/10000^{2i/d}\big),\qquad
\mathrm{PE}(pos,2i+1)=\cos\big(pos/10000^{2i/d}\big).
$$

- 正弦位置编码加在嵌入上。
- 后续模型多用可学习位置嵌入或相对位置偏置（如 T5、RoPE）。
- 序列长于训练长度时，正弦编码可以外推；绝对可学习嵌入通常不能。

> [!INFO]+ 复杂度对比
>
> - 自注意力：$O(T^{2} d)$ 时间，$O(T^{2})$ 注意力矩阵。
> - RNN：$O(T d^{2})$，时间维串行。
> - 卷积核宽 $k$ 时 $O(T k d^{2})$，感受野受层数限制。
> - Transformer 一层即可覆盖全序列，代价是对长度的二次复杂度。

<br>

```python
def positional_encoding(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / (10000 ** (2 * (i // 2) / d))
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

def causal_mask(T):
    return np.tril(np.ones((T, T)))
```

- 仅解码器 LM 的训练目标：下一步预测交叉熵。
- 推理为自回归：新 token 追加后再跑一遍，或用 KV cache 避免重算过去的键值。
- Beam search、采样、top-$k$、nucleus（top-$p$）改变解码多样性。
- 作业用 TensorFlow 搭编码器-解码器 Transformer。
- 重点是 mask、shape 与残差是否对齐，不是从零训到 SOTA。

LayerNorm 对最后一个特征维：

$$
\mu=\frac{1}{d}\sum_{j=1}^{d}x_j,\qquad
\sigma^{2}=\frac{1}{d}\sum_{j}(x_j-\mu)^{2},\qquad
\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^{2}+\varepsilon}}+\beta.
$$

- 与 BatchNorm 不同：不依赖 batch，训练与推理公式相同，无 running statistics。
- RNN 与 Transformer 几乎都用 LayerNorm。
- 编码器-解码器注意力：$Q$ 来自解码器状态，$K,V$ 来自编码器输出。
- Padding mask：源句 pad 位置的 logits 设为很大的负数，避免质量泄漏到 pad。
- 因果 mask 与 padding mask 应相乘（或逻辑与）。
- 只做其中一个会在 batched 翻译里出错。

```python
def layer_norm(x, gamma, beta, eps=1e-6):
    """x: (..., d)。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def transformer_block(x, Wq, Wk, Wv, Wo, W1, W2, gamma1, beta1, gamma2, beta2, mask=None):
    """单层编码器：MHA + 残差 LN + FFN + 残差 LN。x: (T, d)。"""
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    attn, _ = scaled_dot_product_attention(Q, K, V, mask)
    h = layer_norm(x + attn @ Wo, gamma1, beta1)
    ff = np.maximum(h @ W1, 0.0) @ W2
    return layer_norm(h + ff, gamma2, beta2)
```

- 多头 reshape：$(T, d)$ 的 $Q$ 变成 $(h, T, d_k)$，再做 batched 矩阵乘，最后拼回 $(T, d)$。
- 漏掉 `swapaxes`：头与时间维缠在一起，梯度检查失败。
- KV cache：第 $t$ 步只算当前 $q_t$，与已缓存的 $K_{1:t}, V_{1:t}$ 做注意力，把新 $k_t,v_t$ 追加。
- 复杂度从每步重复计算 $O(t^{2}d)$ 降到 $O(td)$。
- 作业不一定实现 cache，但应能解释为何线上生成比训练时的并行 teacher forcing 慢。
- 正弦 $10000^{2i/d}$：不同维度波长从 $2\pi$ 到 $10000\cdot 2\pi$，同时编码短距与长距。
- 可学习位置嵌入在固定 $T_{\max}$ 内更灵活；超出 $T_{\max}$ 必须外推或插值。
- RoPE：旋转施加在 $Q,K$ 上，相对位置直接进入点积。
- 课程要求能说明：位置信息必须显式注入。

> [!EXAMPLE]+ 例8 掩码错误
>
> - 忘记因果掩码：位置 $t$ 看见 $t+1$ 的 token。
> - 训练损失异常低，推理崩盘，因为推理没有未来词。
> - 这是数据泄漏的序列版；梯度检查不一定能发现。
> - 单元测试：把未来位置换成随机词，训练损失应上升。

<br>

> [!NOTE]+ 对照：RNN、注意力、Transformer
>
> - RNN：时间串行，$O(Td^{2})$，长程靠门控。
> - Bahdanau：仍有 RNN，另加 $O(T_x T_y d)$ 的源侧加权。
> - Transformer：自注意力 $O(T^{2}d)$，一层即全局感受野；顺序靠位置编码。
> - 因果掩码是语言模型的训练-推理对齐条件。

<br>

> [!NOTE]+ 衔接
>
> Transformer 在 token 上做前向。课堂扩展转到输入空间：对抗沿 $\nabla_x J$ 上升；生成器沿判别器下降。二者都对 $x$ 求梯度，不再只更新 $W$。

<br>

# 对抗与生成

## 概念：对抗

- **问题**
  - 人眼不可察的扰动使分类器高置信度出错；
  - 生成模型学习 $P(x)$ 或条件 $P(x\mid y)$，用于采样而不是只输出标签。
- **范围**
  - Fast Gradient Sign 等攻击的公式、对抗训练（adversarial training）与生成模型谱系；
  - 这些内容来自 2025 Autumn Lecture 4。Coursera C1-C5 没有对应编程作业，但是课堂核心扩展。
- **章节衔接**
  - 上一章的网络给出 $J(\theta,x,y)$；
  - 对抗对 $x$ 上升，生成对 $x$ 建模；
  - 强化学习进一步去掉固定标签 $y$，改用回报。

## 对抗样本

> [!INFO]+ FGSM
>
> - 是什么：沿 $\mathrm{sign}(\nabla_x J)$ 走一小步，得到人眼难察、模型高置信出错的 $x_{\mathrm{adv}}$。
> - 来自：Goodfellow et al. (2015)。PGD 是同一方向上的多步投影。
> - 相对只在 $W$ 上做梯度：攻击的是输入空间。
> - 对抗训练把 $x_{\mathrm{adv}}$ 加进训练集，是 min-max；只报 FGSM 准确率会高估稳健性。

<br>

线性分类器 $w^{\top}x$ 在高维中对沿 $w$ 方向的小扰动敏感。FGSM：

$$
x_{\mathrm{adv}}=x+\varepsilon\,\mathrm{sign}(\nabla_x J(\theta,x,y)).
$$

$\varepsilon$ 很小（图像上常对应每个像素几级灰度）即可跨过决策面。迭代 FGSM / PGD 在 $\ell_{\infty}$ 球内多步投影：

$$
x^{k+1}=\Pi_{\mathcal{B}_{\varepsilon}(x)}\big(x^{k}+\alpha\,\mathrm{sign}(\nabla_x J)\big).
$$

- 对抗训练：把 $x_{\mathrm{adv}}$ 加入训练集，最小化最坏扰动下的损失，是 min-max。
- 代价：训练变慢；对攻击范数外的扰动不一定迁移。
- 防御不能只靠把输入模糊一下：自适应攻击会把防御编进梯度。
- 评估必须用强攻击（PGD、AutoAttack），不能只报 FGSM 准确率。

```python
def fgsm(x, grad_x, eps):
    """grad_x = dJ/dx，与 x 同形。像素若在 [0,1] 则裁剪。"""
    x_adv = x + eps * np.sign(grad_x)
    return np.clip(x_adv, 0.0, 1.0)
```

## 生成模型

> [!INFO]+ VAE、GAN、扩散
>
> - VAE（Kingma & Welling, 2014）：用 ELBO 下界 $\log p(x)$，重参数化让采样可微。
> - GAN（Goodfellow et al., 2014）：生成器与判别器对打，采样快，训练不稳，似然不可直接算。
> - 扩散（Ho et al., 2020 等）：加噪再学去噪，训练稳，采样步数多。
> - 三者都对 $x$ 建模；相对分类器只输出标签。课堂要能对谱系，不必推完 SDE。

<br>

- **自回归**：$P(x)=\prod_t P(x_t\mid x_{<t})$，像素 RNN、GPT 属此类。似然可精确计算，采样是串行的。
- **VAE**：编码器 $q(z\mid x)$，解码器 $p(x\mid z)$，最大化 ELBO（evidence lower bound）
  $$
  \mathbb{E}_{q}[\log p(x\mid z)]-\mathrm{KL}(q(z\mid x)\|p(z)).
  $$
  重参数化 $z=\mu+\sigma\odot\varepsilon$，$\varepsilon\sim\mathcal{N}(0,I)$，使采样路径可微。
- **GAN**：生成器 $G(z)$ 与判别器 $D(x)$。原目标
  $$
  \min_G\max_D \mathbb{E}_{x}[\log D(x)]+\mathbb{E}_{z}[\log(1-D(G(z)))].
  $$
  训练不稳定：模式崩溃、梯度消失。WGAN 等用不同距离缓解。似然不可直接得。
- **扩散**：逐步加噪到高斯，再学逐步去噪。训练稳定，采样步数多。课堂以直觉与应用为主，不要求推导全部 SDE。

- 条件生成（text-to-image、语音 vocoder）把 $y$ 注入归一化层或注意力。
- 评估用 FID、IS、人工偏好，不要只用训练损失。

> [!INFO]+ VAE 的 ELBO
>
> - $\log p(x)=\log\int p(x\mid z)p(z)\,\mathrm{d}z$。
> - 引入 $q(z\mid x)$ 后
>
> $$
> \log p(x)=\mathrm{KL}(q\|p(z\mid x))+\mathbb{E}_{q}[\log p(x\mid z)]-\mathrm{KL}(q(z\mid x)\|p(z)).
> $$
>
> - KL 非负，故 ELBO 是 $\log p(x)$ 的下界。
> - 最大化 ELBO = 重构 − 正则。
> - $q=\mathcal{N}(\mu(x),\mathrm{diag}(\sigma^{2}(x)))$、$p(z)=\mathcal{N}(0,I)$ 时，KL 有闭式
>
> $$
> \frac{1}{2}\sum_j\big(\mu_j^{2}+\sigma_j^{2}-\log\sigma_j^{2}-1\big).
> $$
>
> - 重参数化使 $\partial/\partial\mu$、$\partial/\partial\sigma$ 能穿过采样。
> - KL 过强：后验坍缩到先验，解码器忽略 $z$（posterior collapse）。

<br>

```python
def vae_elbo(x, mu, logvar, decoder_logpx, kl_weight=1.0):
    """mu, logvar: (m, d_z)；decoder_logpx 已对像素求和。"""
    recon = decoder_logpx.mean()
    kl = 0.5 * np.mean(np.sum(mu ** 2 + np.exp(logvar) - logvar - 1.0, axis=1))
    return recon - kl_weight * kl

def reparameterize(mu, logvar, rng):
    eps = rng.normal(size=mu.shape)
    return mu + np.exp(0.5 * logvar) * eps

def pgd_linf(x, loss_grad_fn, eps, alpha, steps):
    """loss_grad_fn(x_adv) 返回 dJ/dx。"""
    x_adv = x.copy()
    for _ in range(steps):
        g = loss_grad_fn(x_adv)
        x_adv = np.clip(x_adv + alpha * np.sign(g), x - eps, x + eps)
        x_adv = np.clip(x_adv, 0.0, 1.0)
    return x_adv
```

- GAN 训练交替：固定 $G$ 升 $D$，固定 $D$ 降 $\log(1-D(G(z)))$，或升 $\log D(G(z))$（非饱和启发式）。
- 判别器过强：生成器梯度消失。
- 判别器过弱：$G$ 收不到有用信号。
- 模式崩溃：$G$ 只输出少数几种图，$D$ 在这些点上仍可能被骗。
- WGAN-GP 用 1-Lipschitz 约束替代 JS 散度，训练更稳，不是作业必实现。
- 扩散前向：$q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)$。
- 闭式：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$。
- 训练预测 $\varepsilon$ 或 $x_0$；采样从 $x_T\sim\mathcal{N}(0,I)$ 逐步去噪。
- 课堂要求掌握噪声预测的直觉，以及似然可算但采样步多、GAN 采样快但不稳的对比。

> [!NOTE]+ 对照：对 $x$ 的两种梯度
>
> - FGSM：沿 $\nabla_x J$ 上升一小步，制造错分类。
> - 生成器：沿判别器损失下降，制造更像真的 $x$。
> - 二者都对输入空间求梯度，不再只更新 $W$。
> - 评估协议必须与攻击或生成过程分开写清，否则数字不可比。

<br>

> [!WARNING]+ 评估
>
> - 只报 FGSM 准确率会高估稳健性。
> - 必须用 PGD 或 AutoAttack 等强攻击。
> - 生成模型不要只用训练损失；FID、IS 与人工偏好正交。

<br>

> [!NOTE]+ 衔接
>
> 对抗与生成仍假设标签或判别信号存在。强化学习去掉固定 $(x,y)$：环境给奖励，网络近似 $Q$ 或 $\pi$。策略章的偏差-方差与分布不匹配在此重现为函数近似偏差、回报方差与仿真到真实。

<br>

# 强化学习

## 概念：强化学习

> [!INFO]+ DQN 与 PPO
>
> - DQN（Mnih et al., 2015）：用网络回归 TD 目标；经验回放打散相关，目标网络稳住 bootstrapping。
> - 相对表格 Q 学习：状态是图像时必须函数近似。
> - REINFORCE / PPO：直接对 $\log\pi$ 加权回报；PPO 裁剪概率比，避免一次更新毁掉策略。
> - 相对监督学习：没有固定 $(x,y)$，数据由当前策略产生，探索失败会先于实现错误出现。

<br>

- **问题**
  - 智能体（agent）与环境交互：状态 $s$、动作 $a$、奖励 $r$、转移 $P$；
  - 目标是最大化期望回报（expected return） $G_t=\sum_{k=0}^{\infty}\gamma^{k} r_{t+k}$。
- **范围**
  - 对应 2025 Autumn Lecture 5：用深度网络近似策略或价值；
  - 输入轨迹，输出策略 $\pi(a\mid s)$ 或 $Q(s,a)$。
- **章节衔接**
  - 上一章仍有 $J(\theta,x,y)$ 或判别器信号；
  - 本章没有固定标签，梯度来自回报；
  - 下一章回到有预训练 $P(y\mid x)$ 的应用层，调整的是条件 $x$ 的构造。

- MDP（Markov decision process）：$(S,A,P,R,\gamma)$。
- 价值函数（value function）$V^{\pi}(s)=\mathbb{E}_{\pi}[G_t\mid s_t=s]$。
- $Q^{\pi}(s,a)=\mathbb{E}_{\pi}[G_t\mid s_t=s,a_t=a]$。
- Bellman：

$$
V^{\pi}(s)=\mathbb{E}_{a\sim\pi,s'}[r+\gamma V^{\pi}(s')].
$$

- 最优 $Q^{\ast}$ 满足 $Q^{\ast}(s,a)=\mathbb{E}[r+\gamma\max_{a'}Q^{\ast}(s',a')]$。

- Deep Q-Network：网络 $Q(s,a;\theta)$ 回归 TD 目标。
- 目标网络（target network） $\theta^{-}$ 缓慢更新，稳定 bootstrapping。
- 经验回放（experience replay）打散样本相关性。
- 动作离散时可行；连续动作常用策略梯度（policy gradient）。

REINFORCE：

$$
\nabla_{\theta}J=\mathbb{E}_{\pi_{\theta}}\big[\nabla_{\theta}\log\pi_{\theta}(a\mid s)\,G_t\big].
$$

- 基线 $b(s)$（如 $V(s)$）不改变无偏性，但降低方差，形成 actor-critic。
- PPO 用裁剪的概率比限制一次更新幅度：

$$
L^{\mathrm{CLIP}}=\mathbb{E}\big[\min\big(\rho\hat{A},\,\mathrm{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat{A}\big)\big],
$$

$$
\rho=\frac{\pi_{\theta}(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}.
$$

> [!WARNING]+ 奖励设计
>
> - 智能体会利用奖励漏洞（reward hacking）。
> - 稀疏奖励使探索困难。
> - 函数近似引入偏差；蒙特卡洛回报方差大。
> - 仿真到真实是分布不匹配。
> - 项目若做 RL：先在可重置、有明确成功条件的环境验证学习曲线，再谈迁移。

<br>

```python
def discounted_returns(rewards, gamma=0.99):
    """rewards: 一条轨迹上的 r_0,...,r_{T-1}。"""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return np.array(out[::-1])

def reinforce_loss(log_probs, returns):
    """最大化期望回报等价于最小化 -log pi * G。"""
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return -np.sum(log_probs * returns)

def dqn_td_target(r, done, q_next, gamma=0.99):
    """r, done, q_next: (batch,)。目标网络给出 q_next = max_a Q(s',a; theta-)。"""
    return r + gamma * (1.0 - done) * q_next
```

DQN 算法要点：

1. 用 $\varepsilon$-greedy 从 $Q(s,\cdot;\theta)$ 选动作，与环境交互，把 $(s,a,r,s',\mathrm{done})$ 写入 replay buffer。
2. 均匀采样一批转移。
3. 目标 $y=r+\gamma\max_{a'}Q(s',a';\theta^{-})$（终止状态无后续项）。
4. 对 $\theta$ 做 MSE 或 Huber 回归。
5. 每隔 $C$ 步 $\theta^{-}\leftarrow\theta$。

- 没有 replay：样本强相关。
- 没有目标网络：$y$ 与 $Q$ 同时移动。
- 二者都是发散的常见原因。
- 连续动作不能对 $a$ 做 $\max$，改用 DDPG/PPO 等策略方法。
- 轨迹概率 $P(\tau)=\rho(s_0)\prod_t \pi(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$。
- $\nabla\log P(\tau)=\sum_t\nabla\log\pi(a_t\mid s_t)$；转移 $P$ 不出现在梯度里。
- 因此可以在未知模型上学习（model-free）。
- 优势 $\hat{A}_t=G_t-V(s_t)$ 或 GAE 进一步降方差。
- PPO 的 clip 把 $\rho$ 限制在 $[1-\varepsilon,1+\varepsilon]$，防止一次更新毁掉策略。
- $\varepsilon$ 常取 $0.1$-$0.2$。
- 探索：$\varepsilon$-greedy、熵奖励、噪声网络。
- 奖励稀疏（只有终局 $+1$）时，先塑造稠密奖励或用课程学习。
- 曲线长期为零：不能只归因于算法写错，也可能是没探索到。
- 仿真到真实：像素 vs 状态向量、延迟、摩擦都会造成不匹配。
- 先在状态可完整观测的玩具环境验证实现，再上像素输入。
- 对应策略章：先保证能拟合训练。

> [!NOTE]+ 对照：监督、对抗、RL
>
> - 监督：$\nabla_\theta J(\theta,x,y)$，标签固定。
> - 对抗：$\nabla_x J$，标签仍在，更新的是输入。
> - RL：$\nabla_\theta\log\pi\cdot G$，标签换成回报，数据由策略自己产生。
> - 策略自己产生数据，因此探索失败与分布漂移比监督学习更早出现。

<br>

> [!NOTE]+ 衔接
>
> RL 的数据由策略与环境交互产生。LLM 应用层相反：骨干已经预训练，调整的是提示、检索与工具，即条件 $x$ 的构造。评估仍按任务拆开，不能只报笼统的主观质量。

<br>

# LLM 应用

## 概念：应用层

- **问题**
  - 预训练 Transformer 已经存在时，产品性能往往取决于检索、工具、记忆、评估与安全，而不是再训一个骨干；
  - 对应 2025 Autumn Lecture 8：Beyond the model。
- **范围**
  - 把提示、检索增强（retrieval-augmented）、工具调用（tool use）与评价协议写成可测量流水线；
  - 输入用户请求与外部知识源，输出带引用或工具痕迹的回答。
- **章节衔接**
  - 上一章没有固定 $(x,y)$；
  - 本章已有预训练 $P(y\mid x)$，主要调整条件 $x$；
  - 下一章用内部分析检查模型盯着什么，以及切片是否掩盖失败。

- 预训练模型是条件分布 $P(y\mid x)$。
- 应用层控制 $x$ 的构造：$x=$ 系统指令 + 检索片段 + 对话历史 + 用户问题。
- 上下文窗口（context window）有限，必须做选择与压缩。
> [!INFO]+ RAG 与 LoRA
>
> - RAG（Lewis et al., 2020）：检索 $k$ 段证据再生成，改的是条件 $x$，不更新骨干。
> - 相对闭卷提示：事实可随检索库更新；检索失败时模型仍会流畅编造，故召回要单测。
> - LoRA（Hu et al., 2021）：在线性层上加低秩 $\Delta W=BA$，适合风格与格式，不适合补过时事实。
> - 误差表决定走检索、提示还是低秩微调。

<br>

- RAG（retrieval-augmented generation）：检索器返回 $k$ 段文档，生成器只在这些证据上回答。
- 检索失败时生成器仍会流畅地编造。
- 必须单独测检索召回，不能只看最终 BLEU 或人工分。
- 工具调用把输出约束为可解析动作（搜索、计算器、数据库）。
- 解析失败、权限过宽与注入攻击（指令藏在检索文档里）是系统风险。
- 这些风险不是再加一点温度能解决的。
- 评估应分任务：闭卷事实、开卷引用、数学、代码、拒答。
- 笼统的主观质量无法拆开改进。
- 项目若做 LLM 应用，基线应包括：无检索零样本（zero-shot）、少样本（few-shot）、简单 RAG。
- 误差分析标注：检索错 / 证据够但推理错 / 格式错 / 拒答不当。
- LoRA（low-rank adaptation）等微调：仅当误差表显示风格或格式占主导、且有足够私有数据时才值得做。

RAG 流水线可以写成与模型无关的步骤：

1. 文档切块：按标题或固定 token 数，块过大则噪声多，过小则上下文碎。
2. 嵌入并建索引；查询时对问题向量取 $k$ 近邻。
3. 组装提示：系统规则 + 引用块（带编号）+ 用户问题 + “仅依据编号证据回答”。
4. 生成；解析引用是否指向真实块号。
5. 分别记录检索召回（金标准块是否进入 top-$k$）与生成正确率。

```python
def retrieve(query_vec, doc_vecs, k=5):
    """doc_vecs: (n_doc, d)，已 L2 归一化时内积即余弦。"""
    scores = doc_vecs @ query_vec
    idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

def build_rag_prompt(question, chunks, idx):
    evidence = "\n".join(f"[{i}] {chunks[j]}" for i, j in enumerate(idx, 1))
    return (
        "只根据下列证据回答。若证据不足，明确写出不知道。\n"
        f"{evidence}\n问题：{question}\n回答："
    )
```

- 少样本提示：把 $k$ 个输入-输出示范放进上下文，不更新权重。
- 示范必须与测试分布同格式，否则学到的是表面模板。
- 温度、top-$p$ 属于解码超参数，应在 dev 上选，并在报告中固定。
- 工具调用：模型输出 JSON 或特定标记，程序执行后再把结果追加进上下文。
- 失败模式：字段缺失、类型错误、重复调用、对不可用工具幻想。
- 应用层用 schema 校验，而不是再提示“请一定输出合法 JSON”。
- 提示注入（prompt injection）：不可信文档里的“忽略以上指令”应被视为数据。
- 系统消息与工具权限隔离。
- LoRA：线性层上加低秩 $\Delta W=BA$，$B\in\mathbb{R}^{d\times r}$，$A\in\mathbb{R}^{r\times k}$，$r\ll \min(d,k)$。
- 可训参数远小于全量微调，适合项目算力。
- 误差来自事实过时：更新检索库比 LoRA 便宜。
- 评估协议先写再测。
- 闭卷 QA：exact match / F1。
- 开卷：检查引用是否支撑答案。
- 代码：单测通过率。
- 安全：红队集。
- 不要用同一聊天记录既调提示又报最终分。

> [!NOTE]+ 对照：改权重还是改上下文
>
> - 零样本 / 少样本 / RAG：不更新 $W$，改 $x$。
> - LoRA：更新低秩 $\Delta W$，适合风格与格式。
> - 事实过时优先改检索库，不是再训骨干。
> - 误差表决定下一步改检索、提示还是微调，对应策略章的正交化。

<br>

> [!WARNING]+ 注入与评估
>
> - 检索文档是数据，不是指令。
> - 工具权限过宽与解析失败属于系统风险。
> - 检索召回与生成正确率必须分开记。

<br>

> [!NOTE]+ 衔接
>
> 应用层调整条件 $x$。内部分析检查模型依据哪些特征，以及子群是否被总体准确率掩盖。这些量用来决定下一实验，不是用来事后解释。

<br>

# 内部分析

## 概念：可解释

- **问题**
  - 网络是否学到预期的特征、失败模式能否定位到层或样本；
  - 对应 Lecture 10：What’s Going On Inside My Model。
- **范围**
  - 用激活、显著性（saliency）、对抗点与切片指标做诊断，并据此选择下一实验。
- **章节衔接**
  - 上一章把失败标成检索 / 推理 / 格式；
  - 本章把失败标到层、样本或捷径特征；
  - 下一章把同一套诊断用到自选项目数据上。

可用工具：

- **学习曲线**：训练与 dev 损失、指标同图，区分欠拟合、过拟合、优化发散。
- **混淆矩阵（confusion matrix）与切片**：按光照、语言、设备切片，避免总体准确率掩盖子群失败。
- **显著性 / Grad-CAM**：$\partial y_c/\partial x$ 或特征图层的加权，检查模型是否盯着物体而非水印。
- 显著图可被输入变换愚弄，只能当假设生成器。
- **最近邻嵌入**：把验证样本嵌到表示空间，看错例是否与另一类混叠。
- **消融（ablation）**：去掉增强、正则或某一数据源，量化贡献。

- 注意力权重、某神经元的最大化图都不是因果证明。
- 可解释性服务于误差分析与安全审查：决定下一周改数据还是改模型。

> [!INFO]+ Grad-CAM
>
> - 是什么：用 $\partial y_c/\partial A^{k}$ 对特征图加权，得到类别热力图。
> - 来自：Selvaraju et al. (2017)，由 CAM 推广到任意卷积网络。
> - 相对只看准确率：能提出“模型盯着哪里”的假设。
> - 热力图可被输入变换愚弄；必须再用遮挡或切片检验。

<br>

Grad-CAM 对卷积特征图 $A^{k}$ 与类别分数 $y_c$：

$$
\alpha_k=\frac{1}{Z}\sum_{i,j}\frac{\partial y_c}{\partial A^{k}_{ij}},\qquad
L_{\mathrm{CAM}}=\mathrm{ReLU}\Big(\sum_k \alpha_k A^{k}\Big).
$$

- 热力图上采样到原图尺寸。
- ReLU 丢掉对类别有负贡献的区域。
- 热力图落在角落水印或床头文字而不是病灶：模型可能在抄捷径（shortcut）。
- 显著性图对输入变换不稳定，只能提出假设。
- 再用切片指标或遮挡实验验证：把高显著区域涂掉，若 $y_c$ 不变，则解释不可信。

```python
def confusion_and_slices(y_true, y_pred, slice_ids):
    """slice_ids 如设备型号。返回总体混淆与各切片准确率。"""
    k = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = {}
    for s in np.unique(slice_ids):
        m = slice_ids == s
        acc[str(s)] = float((y_true[m] == y_pred[m]).mean()) if m.any() else None
    return cm, acc
```

学习曲线诊断：

- 训练损失不降：学习率、bug、标签错、初始化爆炸。
- 训练损失降、训练指标差：损失与指标不对齐（例如用 Acc 看交叉熵任务的阈值问题）。
- 训练好、dev 差且同分布：方差。
- 训练-训练 dev 好、真实 dev 差：不匹配。
- 损失出现 NaN：溢出、未裁剪、学习率过大、除零。

- 内部分析服务于误差分析：决定下一周改数据还是改模型。
- 期中若问“下一步做什么”，答案必须引用上述某一种曲线或误差表，而不是报层名。

> [!NOTE]+ 对照：解释与因果
>
> - Grad-CAM 与显著性只生成假设。
> - 遮挡、切片、消融才能检验假设。
> - 对抗章已经说明：输入梯度可被利用，也可被解释图愚弄。
> - 期中问“下一步做什么”时，必须引用曲线或误差表，而不是报层名。

<br>

> [!NOTE]+ 衔接
>
> 内部分析给出可引用的曲线与误差表。下一章先把 Coursera 编程作业对回公式，再进入自选项目。Coursera 作业数据集不能直接充当项目。

<br>

# 作业

## 概念：作业

- **问题**
  - CS230 的每周编程与测验在 Coursera 的 Deep Learning Specialization；截止日期以 [Syllabus](https://cs230.stanford.edu/syllabus/) 为准，不是 Coursera 默认周历；
  - 最新公开清单对齐 2025 Autumn：每周二 11:00 a.m. PST，晚于该时刻才算迟交。
- **范围**
  - 概括每份编程作业在实现什么，给出带注释的核心函数与对拍检查；
  - 不照抄 notebook 题面。测验只列必须能口述的结论。
- **章节衔接**
  - 前面各章给向量化公式；
  - 本章接到 Coursera 自动评分的函数签名。

| 周 | 模块 | 编程作业 |
| :--- | :--- | :--- |
| L2 | C1M1 / C1M2 | Python Basics with Numpy（可选）、Logistic Regression with a neural network mindset |
| L3 | C1M3 / C1M4 | Planar data classification、Building your Deep Neural Network: step by step、Deep Neural Network Application |
| L4 | C2M1 / C2M2 | Initialization、Regularization、Gradient Checking、Optimization |
| L5 | C2M3 / C3 | Tensorflow |
| L6 | C4M1 / C4M2 | Convolutional Model step by step / application、Residual Networks、Transfer Learning with MobileNet |
| L7 | C4M3 / C4M4 | YOLO、Neural Style Transfer、Face Recognition、U-Net |
| L8 | C5M1 | RNN step by step、Dinosaur Island、Jazz LSTM |
| L9 | C5M2 / C5M3 | Word Vectors Debiasing、Emojify、NMT Attention、Trigger Word |
| L10 | C5M4 | Transformers Architecture with Tensorflow |

记号与课程一致：样本数是 $m$，特征矩阵 $X\in\mathbb{R}^{n_x\times m}$。标签 $y\in\{0,1\}$。迟交共 $10$ 天，单份最多 $3$ 天。

> [!NOTE]+ 读法（先看这个）
>
> 每份作业先在纸上写出 **shape**，再写代码。自动评分最常见的失败是 `(n_x, m)` 与 `(m, n_x)` 对调，或 `keepdims` 丢了。
>
> | 作业 | 先盯这个 |
> | :--- | :--- |
> | 逻辑回归 | $X$ 是 $(12288, m)$；像素先 `/255` |
> | 深层网 | 缓存必须按层存 $A,W,b$；反向用前向的激活 |
> | 正则 | $L_2$ 只加 $\|W\|^{2}$，不加 $\|b\|^{2}$；Dropout 正向反向共用同一掩码 |
> | 卷积 | 输出空间尺寸 $\lfloor(n+2p-f)/s\rfloor+1$ |
> | RNN | 时间步循环在 $T_x$ 上，权跨步共享 |
> | 注意力 | $\alpha$ 对编码器时间求和为 $1$ |

<br>

> [!WARNING]+ 对拍与荣誉守则
>
> 函数必须写在指定单元格内。笔记用于理解，不能整段贴进 Coursera。项目禁止把作业猫图或恐龙名字当新数据。

<br>

## C1 浅层与深层

> [!EXAMPLE]+ 逻辑回归猫分类
>
> **题在问什么**。`(64,64,3)` 图像展成 $12288$ 维列向量，学 $w,b$，输出 $P(y=1\mid x)$。
>
> **先算一个数**。单像素三通道 $x=(0.2,0.8,0.4)$，$w=(1,-1,0.5)$，$b=0$。$z=0.2-0.8+0.2= -0.4$，$a=\sigma(-0.4)\approx 0.401$。若 $y=1$，损失 $-\log 0.401\approx 0.91$，梯度 $dw=(0.401-1)x$。
>
> 向量化：$A=\sigma(w^{\top}X+b)$，$\mathrm{d}w=\frac1m X(A-Y)^{\top}$，$\mathrm{d}b=\frac1m\sum(A-Y)$。零初始化合法，因为没有隐层。图像必须先 `/255`，训练与测试用同一 `reshape` 顺序。学习率 $0.005$、$2000$ 步是该作业的常见能下降起点。
>
> **要交的答案**：训练准确率接近 $99\%$、测试大约 $70\%$ 量级（线性模型对猫图有上限）、学习曲线下降。

```python
def propagate(w, b, X, Y, eps=1e-12):
    """X: (n_x, m), Y: (1, m), w: (n_x, 1)。返回代价与梯度。"""
    m = X.shape[1]
    A = 1.0 / (1.0 + np.exp(-(w.T @ X + b)))   # (1, m)
    A = np.clip(A, eps, 1.0 - eps)
    cost = float(-(Y * np.log(A) + (1 - Y) * np.log(1 - A)).sum() / m)
    dZ = A - Y
    dw = (X @ dZ.T) / m                         # (n_x, 1)
    db = float(dZ.sum() / m)
    return cost, dw, db
```

<br>

> [!EXAMPLE]+ 平面数据与 $L$ 层网络
>
> **平面分类**。花瓣形二维点，单隐层 $\tanh$ + 输出 sigmoid。隐单元太少（例如 $1$）画不出花瓣；太多且无正则会绕噪声。
>
> **逐步深层**。第 $\ell$ 层：$Z^{[\ell]}=W^{[\ell]}A^{[\ell-1]}+b^{[\ell]}$，隐层 ReLU，输出 sigmoid。$W^{[\ell]}$ 的 shape 是 $(n^{[\ell]}, n^{[\ell-1]})$。He 初始化：$W\sim\mathcal{N}(0,2/n^{[\ell-1]})$。反向：输出 $\mathrm{d}Z^{[L]}=A^{[L]}-Y$。ReLU 的 $\mathrm{d}Z$ 在 $Z\le 0$ 处置零。`db` 必须 `keepdims=True`。
>
> **要交的答案**：平面图决策边界、$L$ 层前向 / 反向通过单元测试、应用作业的 train / test 准确率。

```python
def linear_forward(A_prev, W, b):
    # A_prev: (n_prev, m), W: (n, n_prev), b: (n, 1)
    Z = W @ A_prev + b
    return Z, (A_prev, W, b)

def relu(Z):
    return np.maximum(0.0, Z), Z                 # 反向需要 Z 才能知道哪边是 0

def L_model_forward(X, parameters):
    """parameters 含 W1,b1,...,WL,bL。返回 AL 与 caches。"""
    caches = []
    A = X
    L = len(parameters) // 2
    for ell in range(1, L):
        Z, lin = linear_forward(A, parameters[f"W{ell}"], parameters[f"b{ell}"])
        A, act = relu(Z)
        caches.append((lin, act))
    ZL, lin = linear_forward(A, parameters[f"W{L}"], parameters[f"b{L}"])
    AL = 1.0 / (1.0 + np.exp(-ZL))
    caches.append((lin, ZL))
    return AL, caches
```

<br>

## C2 正则与优化

> [!INFO]+ 初始化、正则、梯度检查
>
> **初始化**。全零：隐层对称，每个单元学同一函数。过大高斯：ReLU 前的 $Z$ 进饱和，出现 `nan`。He 对 ReLU，Xavier 对 $\tanh$。
>
> **$L_2$**。代价加 $\frac{\lambda}{2m}\sum_\ell\|W^{[\ell]}\|_F^{2}$。梯度 $\mathrm{d}W$ 多一项 $\frac{\lambda}{m}W$。偏置不加。
>
> **Dropout**。前向 $A\leftarrow A\odot D/\mathrm{keep}$，$D\sim\mathrm{Bernoulli}(\mathrm{keep})$。反向必须用**同一** $D$，再除 $\mathrm{keep}$。测试阶段关掉 Dropout。
>
> **梯度检查**。$\frac{J(\theta+\varepsilon)-J(\theta-\varepsilon)}{2\varepsilon}$ 与解析梯度的相对误差应 $<10^{-7}$。Dropout 与 BatchNorm 的随机性会毁掉检查，必须先关掉。
>
> **要交的答案**：三种初始化的损失曲线、有 / 无 $L_2$ 与 Dropout 的决策边界、梯度检查相对误差。

```python
def compute_cost_with_regularization(AL, Y, parameters, lambd, eps=1e-12):
    m = Y.shape[1]
    AL = np.clip(AL, eps, 1.0 - eps)
    cross = -(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)).sum() / m
    l2 = 0.0
    for k, W in parameters.items():
        if k.startswith("W"):
            l2 += np.sum(W ** 2)
    return float(cross + (lambd / (2.0 * m)) * l2)

def backward_relu_dropout(dA, cache, D, keep):
    # cache 是前向 ReLU 的 Z；D 必须是前向那张掩码
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    dZ = (dZ * D) / keep
    return dZ
```

<br>

> [!EXAMPLE]+ Momentum / RMSprop / Adam
>
> mini-batch 把 $m$ 换成批大小。Momentum：$v\leftarrow\beta v+(1-\beta)\mathrm{d}\theta$，$\theta\leftarrow\theta-\alpha v$。RMSprop 用 $\mathrm{d}\theta^{2}$ 的指数平均做分母。Adam 两者都要做偏差修正：
>
> $$
> \hat v_t=\frac{v_t}{1-\beta_1^{t}},\qquad
> \hat s_t=\frac{s_t}{1-\beta_2^{t}},\qquad
> \theta\leftarrow\theta-\alpha\frac{\hat v_t}{\sqrt{\hat s_t}+\varepsilon}.
> $$
>
> $t$ 从 $1$ 起算。漏掉偏差修正，前几步会过小。作业常用 $\beta_1=0.9$，$\beta_2=0.999$，$\varepsilon=10^{-8}$。
>
> TensorFlow 作业只要求用 `tf.Variable`、`GradientTape` 与 `optimizer.apply_gradients` 复现同一逻辑，不要手写数值循环去“绕过”框架。
>
> **要交的答案**：三种优化器的损失曲线、Adam 通常最快到低损失。

<br>

## C4 卷积、残差、检测

> [!INFO]+ 卷积尺寸与前向
>
> $$
> n_{\mathrm{out}}=\Big\lfloor\frac{n_{\mathrm{in}}+2p-f}{s}\Big\rfloor+1.
> $$
>
> 例：$n=8$、$f=3$、$p=1$、$s=2$，输出 $4$。多通道：$W$ 的 shape `(f, f, n_C_prev, n_C)`。池化没有要学的权，MaxPool 反向只把梯度送回 argmax 位置。残差块：$a_{\mathrm{out}}=\mathrm{ReLU}(F(a)+W_s a)$，捷径维数不对时 $W_s$ 是 $1\times 1$ 卷积。MobileNet 冻住卷积核，只训顶部分类头。
>
> **要交的答案**：逐步卷积的单元测试、残差块图、迁移学习准确率。

```python
def conv_forward(A_prev, W, b, hparams):
    """A_prev: (m, n_H, n_W, n_C_prev)。返回 Z 与 cache。"""
    (m, n_H, n_W, n_C_prev) = A_prev.shape
    (f, f, _, n_C) = W.shape
    stride, pad = hparams["stride"], hparams["pad"]
    n_H_out = (n_H + 2 * pad - f) // stride + 1
    n_W_out = (n_W + 2 * pad - f) // stride + 1
    A_pad = np.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    Z = np.zeros((m, n_H_out, n_W_out, n_C))
    for i in range(m):
        for h in range(n_H_out):
            for w in range(n_W_out):
                for c in range(n_C):
                    hs, ws = h * stride, w * stride
                    window = A_pad[i, hs:hs + f, ws:ws + f, :]
                    Z[i, h, w, c] = np.sum(window * W[:, :, :, c]) + float(b[:, :, :, c])
    return Z, (A_prev, W, b, hparams)
```

<br>

> [!NOTE]+ YOLO、风格、人脸、U-Net
>
> **YOLO**。每格预测边界框与类别。过滤：置信度阈值，再非极大值抑制（non-max suppression, NMS），IoU 超过阈值的重叠框只留分数最高的。
>
> **风格迁移**。内容损失用深层特征 $\|a_C-a_G\|^{2}$。风格损失用 Gram 矩阵 $G=aa^{\top}$ 的差。总损失 $\alpha\ell_{\mathrm{content}}+\beta\ell_{\mathrm{style}}$。优化的是生成图像像素，不是网络权。
>
> **人脸**。验证是一次比较两张图的编码距离；识别是在库里找最近邻。三元组损失：$\|f(A)-f(P)\|^{2}+m\le\|f(A)-f(N)\|^{2}$。
>
> **U-Net**。编码器下采样，解码器上采样，同分辨率的特征按通道拼接。分割输出每像素一个类。
>
> **要交的答案**：NMS 后的框、风格图、编码距离阈值、U-Net 分割叠加图。

<br>

## C5 序列与 Transformer

> [!EXAMPLE]+ RNN 一步
>
> $$
> a^{\langle t\rangle}=\tanh(W_{aa}a^{\langle t-1\rangle}+W_{ax}x^{\langle t\rangle}+b_a),\qquad
> y^{\langle t\rangle}=\mathrm{softmax}(W_{ya}a^{\langle t\rangle}+b_y).
> $$
>
> 权在时间上共享。恐龙作业是字符级：输入一个前缀，输出下一个字符的分布。Jazz LSTM 把忘门 / 输入门 / 输出门按笔记逐步实现。
>
> **词向量去偏**。把性别方向 $g$ 从职业词里减掉投影：$e\leftarrow e-(e\cdot \hat g)\hat g$。中性词应靠近，成对词关于 $g$ 对称。
>
> **Emojify**。句子词向量平均后接 softmax，或改 LSTM。平均袋会丢掉词序：`not good` 与 `good` 可能撞车。
>
> **注意力翻译**。对齐分数 $e_{t,t'}=\tanh$ 或加性注意力，再 $\alpha_{t,t'}=\mathrm{softmax}_{t'}(e_{t,\cdot})$，上下文 $c_t=\sum_{t'}\alpha_{t,t'}h_{t'}$。$\alpha$ 对 $t'$ 求和必须为 $1$。
>
> **触发词**。频谱图进一维卷积或 RNN，输出每帧是否在唤醒词上。类别极不平衡，正帧要加权。
>
> **Transformer**。$Q=XW_Q$，$K=XW_K$，$V=XW_V$，
>
> $$
> \mathrm{Attention}(Q,K,V)=\mathrm{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)V.
> $$
>
> 多头把 $d_{\mathrm{model}}$ 切开再拼回。位置编码加在嵌入上。作业用 TensorFlow 搭编码器块，不是从零写反向。
>
> **要交的答案**：RNN 单元测试、一段恐龙名、去偏前后的余弦表、注意力对齐图、触发词 F1、Transformer 的一次前向 shape。

```python
def softmax(Z):
    # 减最大值，避免 exp 溢出
    Z = Z - Z.max(axis=0, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=0, keepdims=True)

def rnn_cell_forward(xt, a_prev, parameters):
    """xt: (n_x, m)；a_prev: (n_a, m)。权跨时间共享。"""
    Wax, Waa, Wya = parameters["Wax"], parameters["Waa"], parameters["Wya"]
    ba, by = parameters["ba"], parameters["by"]
    a_next = np.tanh(Waa @ a_prev + Wax @ xt + ba)
    yt_pred = softmax(Wya @ a_next + by)
    return a_next, yt_pred, (a_next, a_prev, xt, parameters)

def rnn_forward(x, a0, parameters):
    # x: (n_x, m, T_x)。沿时间把隐状态传下去
    n_x, m, T_x = x.shape
    n_y = parameters["Wya"].shape[0]
    a, y_pred = a0, np.zeros((n_y, m, T_x))
    caches = []
    for t in range(T_x):
        a, yt, cache = rnn_cell_forward(x[:, :, t], a, parameters)
        y_pred[:, :, t] = yt
        caches.append(cache)
    return y_pred, a, caches
```

<br>

> [!NOTE]+ 衔接
>
> 作业按 Coursera 周组织。下一章把同一套模块接到自选数据：先基线、再误差表，不要把猫图作业改标题当项目。

<br>

# 项目

## 概念：项目

- **问题**
  - CS230 的主交付物是可复现的深度学习项目，不能把 Coursera 作业改个标题再交一遍；
  - 公开时间线（2025 Autumn）：提案、中期、期末报告与海报。
- **范围**
  - 选定任务与数据、事先约定指标、跑通基线、按误差分析迭代；
  - 输出提案、里程碑、最终报告与演示。
- **章节衔接**
  - 前面各章给出模块与诊断语言；
  - 本章要求在新数据上依次检查：拟合训练、拟合 dev、拟合部署；
  - 复习章汇总默写清单与评分口径。

- Section 材料给出的步骤：选题、文献、数据、指标与第一模型、迭代。
- 数据应优先用已有、许可清晰、标签定义明确的集合。
- 自建数据采集成本高，且常在标注协议上失败。
- 项目可与其他课合并，但必须是深度学习焦点。
- 提案应写清：问题、数据规模与划分、评估指标、最简单基线、风险（数据不够、标签噪、算力）。
- 里程碑应有数字：基线指标、误差分析表、下一步实验。
- 最终报告应能让他人复现划分与训练命令，并诚实报告负结果。

> [!EXAMPLE]+ 例4 可执行的项目循环
>
> - 任务：胸片是否含肺炎。
> - 指标：AUROC 为主；灵敏度在特异度 $\ge 0.8$ 时为满意度指标。
> - 基线：预训练 ResNet-18 微调。
> - 误差分析：心影重叠区域假阴性高。
> - 下一实验：加大该区域采样，或改 U-Net 先分割肺野再分类。
> - 不把下一实验设为“换更大的 ViT”，除非切片结果说明容量不足。

提案模板（应对应 Section 1/2 的检查项）：

- 问题一句话：输入、输出、谁用、错误代价。
- 数据：来源、许可、规模、正负比、划分方式、是否与部署同分布。
- 指标：一个优化指标 + 满意度约束。
- 基线：非深度学习或最浅网络的数字目标。
- 方法：为何该架构匹配该输入结构（网格 / 序列 / 图）。
- 算力：一次实验多久、计划实验次数。
- 风险与备选：数据不够时如何降级任务。

- 里程碑应包含：可复现的训练命令、基线表、至少 50-100 个错误样本的标注表、一张学习曲线。
- 最终报告写清负结果：哪些改动没有提升。
- 海报突出问题、数据、指标、误差分析驱动的一次关键改进，而不是层数列表。
- 与其他课合并时，CS230 部分必须能单独拆出：深度模型、训练曲线、与非深度基线的比较。
- 只交一个软件演示而没有误差分解，不符合课程评分。
- 常见失败：选题过大（“做通用自动驾驶”）、自采数据到中期仍无标签、用 test 调参、不固定 seed、把 Coursera 作业数据集改个标题当项目。
- 先做小、做完、再加范围。

复现检查：

- 代码仓库含 `requirements` 或 lock 文件、随机种子、数据下载或划分脚本。
- 表格同时列出基线、本方法、消融（无增强 / 无预训练 / 无注意力）。
- 图的坐标轴有单位；学习曲线含 train 与 dev。
- 伦理：医疗与人脸数据的许可、是否可识别个人、是否只在聚合层面报告。

- 公开数据集起点（仍须自己钉指标与划分）：
  - 图像分类：CIFAR / ImageNet 子集；
  - 检测：PASCAL 或自车前视小集；
  - 医疗：课程允许的去标识胸部影像；
  - 文本：公开情感或翻译对；
  - 音频：触发词或乐器。
- 选用数据集后第一件事是统计标签分布与缺损，而不是先下预训练权重。
- 评分通常看：问题是否清楚、方法是否匹配数据、实验是否可复现、误差分析是否驱动迭代、写作是否诚实。
- 模型更大或损失更低本身不加分。

> [!WARNING]+ 项目失败模式
>
> - 选题过大，中期仍无标签。
> - 用 test 调参或不固定 seed。
> - 把 Coursera 作业数据集改个标题当项目。
> - 只交软件演示，没有误差分解。

<br>

> [!NOTE]+ 衔接
>
> 项目需检查拟合训练、拟合 dev 与拟合部署。复习章汇总公式与误差归因，两者都要能独立写出来。

<br>

# 复习

- 课程把深度学习当成可微分模块的组合：仿射或卷积、逐点非线性、归一化、注意力、损失。
- 把模块写对之后，项目成败由数据、指标与误差归因决定。

$$
\begin{aligned}
\text{拟合训练} &\leftarrow \text{容量、优化、初始化},\\
\text{拟合开发} &\leftarrow \text{正则、数据、分布对齐},\\
\text{拟合部署} &\leftarrow \text{dev 是否等于现实}.
\end{aligned}
$$

- 逻辑回归与 $L$ 层网络：shape、BCE 与 $A-Y$、反向缓存。
- 正则与优化：偏差-方差、Dropout、$L_2$、Adam 偏差修正、BatchNorm 推理统计。
- 策略：单一指标、人类水平、误差上限、不匹配。
- CNN：尺寸公式、残差、YOLO+NMS、triplet、Gram 矩阵。
- 序列：BPTT、LSTM 门、注意力期望、$\mathrm{softmax}(QK^{\top}/\sqrt{d_k})$。
- 课堂扩展：FGSM、GAN/VAE/扩散、DQN/PPO、RAG 与切片评估。

- 期末闭卷通常覆盖前半：前向反向、超参数、策略案例、卷积尺寸。
- 编程作业覆盖实现细节。
- 项目覆盖能否把上述方法用到自选数据上。

默写清单（与作业/期中对齐）：

- $A^{[l]}$ 与 $W^{[l]}$ 的 shape；`dW = dZ @ A_prev.T / m`。
- sigmoid/softmax 配交叉熵时 $\mathrm{d}Z=A-Y$。
- He：$\mathrm{Var}=2/n^{[l-1]}$；Xavier：$2/(n_{\mathrm{in}}+n_{\mathrm{out}})$。
- inverted dropout 训练除以 $p$，推理不丢弃。
- Adam 的 $t$ 从 1 计，两个矩都要偏差修正。
- BatchNorm 训练用 batch 统计，推理用 running average；$\gamma,\beta$ 可学习。
- $n'=\lfloor(n+2p-f)/s\rfloor+1$；same 且 $s=1$ 时 $p=(f-1)/2$。
- ResNet：$y=F(x)+x$；triplet：$\max(\|f_A-f_P\|^{2}-\|f_A-f_N\|^{2}+\alpha,0)$。
- LSTM 六式；注意力 $\alpha=\mathrm{softmax}(e)$，$c=\sum\alpha h$。
- $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^{\top}/\sqrt{d_k})V$；因果掩码。
- 策略：单一优化指标、dev=部署分布、误差表算上限。

- 不能只背公式。
- 实现题会要求指出 shape 错误、忘记转置、推理期仍开 Dropout、或把 train 统计用在 BatchNorm 评估。
- 策略题会给一张误差表，要求选出下一实验并给出百分点上限。

> [!NOTE]+ 三条失败轴
>
> - 拟合训练失败：容量、优化、初始化、实现 bug。
> - 拟合开发失败：正则、数据量、标注噪声。
> - 拟合部署失败：dev 与现实分布不一致。
> - 课堂扩展主要讨论部署阶段：对抗扰动、仿真到真实、检索与协议。

<br>


# 参考文献

## 课程

1. Stanford CS230. *Deep Learning* 课程主页与大纲。[https://cs230.stanford.edu/](https://cs230.stanford.edu/)；[Syllabus](https://cs230.stanford.edu/syllabus/)
2. Stanford Bulletin. *CS230: Deep Learning*. [https://bulletin.stanford.edu/courses/2198851](https://bulletin.stanford.edu/courses/2198851)
3. deeplearning.ai. *Deep Learning Specialization*（C1-C5，CS230 翻转课堂视频与作业来源）。

## 优化与正则

1. Srivastava N et al. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 2014. [论文](https://jmlr.org/papers/v15/srivastava14a.html)
2. Ioffe S, Szegedy C. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML, 2015.
3. Kingma DP, Ba J. *Adam: A Method for Stochastic Optimization*. ICLR, 2015.
4. He K et al. *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV, 2015.（He 初始化）
5. Glorot X, Bengio Y. *Understanding the difficulty of training deep feedforward neural networks*. AISTATS, 2010.（Xavier）

## 卷积与检测

1. He K et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
2. Huang G et al. *Densely Connected Convolutional Networks*. CVPR, 2017.（大纲选读）
3. Howard AG et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. 2017.
4. Redmon J et al. *You Only Look Once: Unified, Real-Time Object Detection*. CVPR, 2016.
5. Ronneberger O et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI, 2015.
6. Schroff F et al. *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR, 2015.
7. Gatys LA et al. *Image Style Transfer Using Convolutional Neural Networks*. CVPR, 2016.

## 序列与注意力

1. Hochreiter S, Schmidhuber J. *Long Short-Term Memory*. Neural Computation, 1997.
2. Cho K et al. *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP, 2014.
3. Sutskever I et al. *Sequence to Sequence Learning with Neural Networks*. NeurIPS, 2014.
4. Bahdanau D et al. *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR, 2015.
5. Vaswani A et al. *Attention Is All You Need*. NeurIPS, 2017.
6. Mikolov T et al. *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS, 2013.

## 对抗、生成与强化学习

1. Goodfellow IJ et al. *Explaining and Harnessing Adversarial Examples*. ICLR, 2015.
2. Goodfellow I et al. *Generative Adversarial Nets*. NeurIPS, 2014.
3. Kingma DP, Welling M. *Auto-Encoding Variational Bayes*. ICLR, 2014.
4. Ho J et al. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020.
5. Mnih V et al. *Human-level control through deep reinforcement learning*. Nature, 2015.
6. Schulman J et al. *Proximal Policy Optimization Algorithms*. 2017.

<br>

# 工具

## 数值与框架

- [NumPy](https://numpy.org/)：shape 对齐、向量化与作业中的从零实现；CS230 约定特征在第 0 维、样本在第 1 维。
- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)：课程编程框架；注意 `training=` 对 Dropout 与 BatchNorm 的切换。
- [PyTorch](https://pytorch.org/)：研究常用；默认 batch-first，与课程讲义转置不同。

## 训练配套

- [TensorBoard](https://www.tensorflow.org/tensorboard) / [Weights & Biases](https://wandb.ai/)：损失、学习率、切片指标与超参数记录。
- cuDNN / 混合精度：吞吐；记录是否启用非确定算法。

## 视觉与检测

- [OpenCV](https://opencv.org/)：读图、仿射增强、框可视化。
- COCO 评估工具：mAP；IoU 阈值与类别映射必须与标注一致。

## 语言与检索

- [spaCy](https://spacy.io/) / 分词器文档：BPE 与词表；OOV 与特殊符号。
- 向量检索库（FAISS 等）：RAG 的 $k$ 近邻；度量（L2 / cosine）须与嵌入归一化一致。

## 环境

- Conda lock、container digest：固定 CUDA、驱动与 Python。只写“TensorFlow 2”不能复现 BatchNorm 与 GPU kernel 差异。

- 工具选择取决于输入张量约定、是否需要自定义反向、部署延迟和许可证。
- 论文、权重版本、预处理与随机种子应同时记录。
- 只写模型 zoo 里的名字，实验无法复现。
