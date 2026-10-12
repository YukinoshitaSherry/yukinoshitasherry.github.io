---
title: 斯坦福CS230：Deep Learning
date: 2026-09-22
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS230 详细课程笔记：从感知机、逻辑回归与反向传播，到正则化、优化、CNN、序列模型、注意力与 Transformer。
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
  - [CS230 Cheatsheets（Afshine & Shervine Amidi）](https://stanford.edu/~shervine/teaching/cs-230/)
- **参考书**
  - 斋藤康毅《深度学习入门：基于 Python 的理论与实现》（*Deep Learning from Scratch*），下文简称“鱼书”。

全文的主线是：先说明一个神经网络**算什么**（前向传播），再说明它**怎么学**（损失函数、梯度、反向传播），然后说明**怎样学得好**（初始化、正则化、优化器、BatchNorm、项目策略），最后进入**特定数据的结构**（图像用卷积，序列用循环网络与注意力）。

<br>


# 预备知识

这一部分不讲模型，只统一两件事：一是全文使用的记号与矩阵形状，二是估算计算量与显存的方法。后面每一条反向传播公式都要靠形状来核对，所以先把记号固定下来。

KL 散度的定义、非负证明、数值例子，以及蒸馏、VAE、RLHF 与最大似然四种用法，见 [信息论基础 · 相对熵](/信息论熵/#kl)。

## 记号

### 监督学习的设定

监督学习（supervised learning）给定成对的数据 $(x,y)$，希望学到一个映射 $x\mapsto\hat{y}$，使预测 $\hat{y}$ 尽量接近标签 $y$。深度网络把这个映射写成“一串仿射变换 + 逐点非线性”的复合：每一层先做 $Wa+b$，再套一个激活函数 $g$。衡量 $\hat{y}$ 与 $y$ 差距的标量称为损失 $J$，训练就是调整所有 $W,b$ 让 $J$ 变小。

### 三种上标

一个量可能同时属于“第几个样本”“第几层”“第几个时间步”，CS230 用三种括号区分：

$$
x^{(i)}\ \text{（第 $i$ 个样本，圆括号）},\qquad
W^{[l]}\ \text{（第 $l$ 层，方括号）},\qquad
a^{\langle t\rangle}\ \text{（第 $t$ 个时间步，尖括号）}.
$$

例如 $a^{[2](5)}$ 表示第 5 个样本在第 2 层的激活。

### 维度符号

- $m$：训练样本数。
- $n_x=n^{[0]}$：输入特征维数。
- $n^{[l]}$：第 $l$ 层神经元个数。
- $L$：网络层数，输出层是第 $L$ 层；输入层记作第 0 层，不计入层数。
- $X\in\mathbb{R}^{n_x\times m}$：设计矩阵，**每一列是一个样本**。
- $Y\in\mathbb{R}^{n_y\times m}$：标签矩阵；二分类时 $n_y=1$。
- $W^{[l]}\in\mathbb{R}^{n^{[l]}\times n^{[l-1]}}$，$b^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}$。
- $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$ 是线性部分，$A^{[l]}=g^{[l]}(Z^{[l]})$ 是激活，$A^{[0]}=X$。
- $\hat{Y}=A^{[L]}$。

$W^{[l]}$ 的形状之所以是“本层 × 上一层”，是因为它要左乘上一层的激活：$(n^{[l]},n^{[l-1]})\times(n^{[l-1]},m)=(n^{[l]},m)$。

### 单样本与整批

对单个样本，第 $l$ 层的计算是

$$
z^{[l](i)}=W^{[l]}a^{[l-1](i)}+b^{[l]},\qquad
a^{[l](i)}=g^{[l]}\big(z^{[l](i)}\big).
$$

把 $m$ 个样本的列向量并排放成矩阵，同一个式子就能一次算完整批：

$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}.
$$

这里 $W^{[l]}A^{[l-1]}$ 的形状是 $(n^{[l]},m)$，而 $b^{[l]}$ 只有 $(n^{[l]},1)$。两者能相加，靠的是 NumPy 的**广播**（broadcasting）：形状不同的数组做逐元素运算时，长度为 1 的维度会被自动复制，直到两边形状一致。所以 $b^{[l]}$ 会沿列复制 $m$ 份，每个样本加上同一个偏置。

> [!EXAMPLE]+ 例0 广播与矩阵乘法的形状规则
>
> 广播：标量与数组运算时，标量会与每个元素运算；$(2,2)$ 的矩阵与 $(2,)$ 的向量相加时，向量被复制成两行。
>
> ```python
> import numpy as np
> A = np.array([[1, 2], [3, 4]])
> B = np.array([10, 20])
> A * B          # array([[10, 40], [30, 80]])
> 1.0 / np.array([1.0, 2.0, 3.0])   # array([1., 0.5, 0.33333333])
> ```
>
> 矩阵乘法要求**左矩阵的列数等于右矩阵的行数**，结果的形状是“左行数 × 右列数”：
>
> ```python
> A = np.array([[1, 2, 3], [4, 5, 6]])     # (2, 3)
> B = np.array([[1, 2], [3, 4], [5, 6]])   # (3, 2)
> np.dot(A, B)   # array([[22, 28], [49, 64]])，形状 (2, 2)
> C = np.array([[1, 2], [3, 4]])           # (2, 2)
> np.dot(A, C)   # ValueError: shapes (2,3) and (2,2) not aligned
> ```
>
> 正因为形状不对就会直接报错（或者更糟，被广播成错误的形状却不报错），实现时应在注释里写明每个数组的 shape。

### 逐层形状表

二分类（$n_y=1$）时各量的形状如下。梯度与对应的量同形，这是检查实现最便宜的办法。

| 量 | shape |
| :--- | :--- |
| $X=A^{[0]}$ | $(n^{[0]}, m)$ |
| $W^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ |
| $b^{[l]}$ | $(n^{[l]}, 1)$ |
| $Z^{[l]}, A^{[l]}$ | $(n^{[l]}, m)$ |
| $\mathrm{d}Z^{[l]}, \mathrm{d}A^{[l]}$ | 与 $Z^{[l]}$ 相同 |
| $\mathrm{d}W^{[l]}, \mathrm{d}b^{[l]}$ | 与对应参数相同 |

- **命名**
  - 代码里的 `dW` 表示 $\partial J/\partial W$，不是数学上的微分 $\mathrm{d}W$；
  - 每算完一个梯度就 `assert dW.shape == W.shape`，可以卡住绝大多数转置错误；
  - 若 $W^{[l]}$ 被误写成 $(n^{[l-1]}, n^{[l]})$，后续每条梯度公式都会差一个转置。
- **额外维度**
  - 时间序列另加一维 $T$；
  - 图像另加 $H,W,C$（高、宽、通道）；
  - 全连接作业：batch 维在**最后**一维；
  - 卷积作业：batch 维在**第一**维 `(m, n_H, n_W, n_C)`（NHWC）；
  - 两套约定不能混用。

> [!WARNING]+ 两种排布：列样本与行样本
>
> CS230 把样本排成**列**，所以写 $Z=WA+b$，$W$ 的形状是（输出, 输入）。鱼书与 PyTorch 的 `nn.Linear` 把样本排成**行**，写 $Y=XW+b$，$W$ 的形状是（输入, 输出）。两者只差一个整体转置，但反向公式的写法会跟着变：
>
> - 列样本：$\mathrm{d}W=\frac{1}{m}\,\mathrm{d}Z\,A^{\top}$，$\mathrm{d}A_{\text{prev}}=W^{\top}\mathrm{d}Z$；
> - 行样本：$\mathrm{d}W=X^{\top}\,\mathrm{d}Y$，$\mathrm{d}X=\mathrm{d}Y\,W^{\top}$。
>
> 对照两边资料时，先确认样本放在行还是列。

### 两个常用的矩阵求导恒等式

反向传播反复用到下面两条：

$$
\frac{\partial}{\partial W}(WA)\ \leadsto\ \text{右乘 } A^{\top},\qquad
\frac{\partial}{\partial A}(WA)\ \leadsto\ \text{左乘 } W^{\top}.
$$

精确的含义是：若 $Z=WA$ 且已知上游梯度 $\mathrm{d}Z=\partial J/\partial Z$，则

$$
\frac{\partial J}{\partial W}=\mathrm{d}Z\,A^{\top},\qquad
\frac{\partial J}{\partial A}=W^{\top}\mathrm{d}Z.
$$

验证方法是逐元素写出来：$Z_{kj}=\sum_{r}W_{kr}A_{rj}$，所以 $\partial Z_{kj}/\partial W_{kr}=A_{rj}$，对所有 $j$ 求和得 $\partial J/\partial W_{kr}=\sum_j \mathrm{d}Z_{kj}A_{rj}=(\mathrm{d}Z\,A^{\top})_{kr}$。写不出转置放哪边时，用形状反推最快：$\mathrm{d}W$ 必须是 $(n^{[l]},n^{[l-1]})$，而 $\mathrm{d}Z$ 是 $(n^{[l]},m)$、$A$ 是 $(n^{[l-1]},m)$，唯一能拼出来的就是 $\mathrm{d}Z\,A^{\top}$。

外积视角：$\mathrm{d}W=\mathrm{d}Z A^{\top}/m$ 等于把 $m$ 个样本各自的外积 $(n^{[l]},1)\times(1,n^{[l-1]})$ 加起来再平均。矩阵乘法只是把这个求和交给 BLAS 一次做完。

<br>

## 复杂度

训练一个网络之前，应当能粗略估计它要算多久、占多少显存。估算的原则是：**只看随规模增长的那几个维度，常数忽略**。

- **时间复杂度**
  - 全连接层 $n^{[l-1]}\rightarrow n^{[l]}$、batch 大小 $m$：一次矩阵乘 $O(m\, n^{[l]} n^{[l-1]})$；
  - 深度 $L$ 的网络：前向与反向同阶，反向的常数更大（每层要做两次矩阵乘：一次求 $\mathrm{d}W$，一次求 $\mathrm{d}A$），一般估为前向的 2 倍；
  - 卷积层：输入 $H\times W$，通道 $C_{\mathrm{in}}\rightarrow C_{\mathrm{out}}$，核 $f\times f$，约 $O(m H W C_{\mathrm{out}} C_{\mathrm{in}} f^{2})$；
  - 自注意力：长度 $T$、宽度 $d$，为 $O(T^{2}d)$，这是长序列的主要瓶颈。
- **空间复杂度**
  - 反向传播需要前向时的中间结果，所以必须保存各层 $A^{[l]}$，以及 BatchNorm 的均值方差、Dropout 的掩码；
  - 因此**激活往往比权重更占显存**；
  - batch 从 32 增到 256，显存近似线性上升（激活随 $m$ 线性增长），参数量不变。

<br>

> [!NOTE]+ 衔接
>
> 记号已经固定：$X$ 为 $(n_x,m)$、$W^{[l]}$ 为 $(n^{[l]},n^{[l-1]})$，梯度与原量同形。下一部分从最简单的“神经元”——感知机——出发，一步步加上可学习的参数、平滑的激活和隐藏层。

<br>


# 浅层网络

这一部分的路线是：

1. **感知机**：一个神经元能表示什么、不能表示什么（XOR）；
2. **逻辑回归**：把阶跃换成 sigmoid，让参数可以用梯度自动学习，并引出损失函数与梯度下降；
3. **浅层网络**：插入一层隐藏层，使决策边界可以弯曲；
4. **反向传播**：用计算图把梯度的计算拆成局部步骤，得到两层网络的完整训练算法。

## 为什么是现在

神经网络的基本想法在 1980 年代就有了，C1M1 用一张图解释为什么到 2010 年代才突然好用：横轴是带标签数据量，纵轴是模型表现。

- **传统算法**（逻辑回归、SVM 等）：数据少时表现不错，但数据再多，曲线很快变平，多出的数据用不上；
- **小型神经网络**：曲线比传统算法稍高、平得稍晚；
- **大型神经网络**：数据越多表现越好，在很大的数据量范围内仍不饱和。

所以要拿到最好的表现，需要同时具备两件事：**足够大的网络**和**足够多的数据**。数据少时，各种算法的排序主要取决于特征工程和细节，大网络未必占优；只有在大数据区间，大网络的优势才稳定地体现出来。

推动这件事的三股力量，课上称为“规模驱动深度学习的进步”（scale drives deep learning progress）：

1. **数据**：手机、传感器、互联网让带标签的数据量暴增；
2. **算力**：GPU 和专用硬件让训练大网络变得可行；
3. **算法**：很多算法改进的目的就是让计算更快。例如把 sigmoid 换成 ReLU，消除了饱和区的梯度消失，梯度下降快得多。

算力还有一层间接作用：深度学习是**高度经验化**的迭代过程，“想法 → 写代码 → 实验 → 再改想法”。一次实验从一个月缩短到十分钟，同样时间内能试的想法就多了上百倍，进步因此加快。后面“项目策略”部分的很多建议，都是为了让这个循环转得更快。

C1M1 还按输入类型对应常见结构：房价、广告点击这类表格数据用标准全连接网络；图像用卷积网络；音频、文本这类序列用循环网络或 Transformer；自动驾驶这类多模态输入用定制的混合结构。前者称为**结构化数据**（每个特征有明确含义），后者称为**非结构化数据**（像素、音频采样、单词），深度学习最大的突破在非结构化数据上。

<br>

## 感知机（Perceptron）

### 定义

感知机由 Frank Rosenblatt 于 1957 年提出，是神经网络的起点。它接收多个输入信号 $x_1,x_2,\dots$，每个输入乘以各自的**权重** $w_i$，求和后与**阈值** $\theta$ 比较，超过阈值就输出 1（称为神经元“被激活”），否则输出 0：

$$
y=\begin{cases}0 & w_1x_1+w_2x_2\le\theta\\ 1 & w_1x_1+w_2x_2>\theta\end{cases}
$$

把 $\theta$ 移到左边并记 $b=-\theta$，就得到更常用的形式：

$$
y=\begin{cases}0 & b+w_1x_1+w_2x_2\le 0\\ 1 & b+w_1x_1+w_2x_2>0\end{cases}
$$

- **权重** $w_i$ 控制各个输入的重要性：权重越大，对应输入对结果的影响越大。
- **偏置** $b$ 控制神经元被激活的难易程度：$b=-0.1$ 时加权和超过 0.1 就激活；$b=-20$ 时加权和要超过 20 才激活。输入全为 0 时，输出只由 $b$ 决定。

### 逻辑门

用感知机实现逻辑门，就是找一组 $(w_1,w_2,b)$ 使它满足真值表。

| $x_1$ | $x_2$ | AND | NAND | OR | XOR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 1 | 0 | 0 |
| 1 | 0 | 0 | 1 | 1 | 1 |
| 0 | 1 | 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 0 | 1 | 0 |

- **与门（AND）**：$(w_1,w_2,b)=(0.5,0.5,-0.7)$。只有两个输入都是 1 时，加权和 $1.0$ 才超过 $0.7$。满足条件的参数有无穷多组，如 $(0.5,0.5,-0.8)$、$(1.0,1.0,-1.0)$。
- **与非门（NAND）**：把与门的参数全部取反，$(w_1,w_2,b)=(-0.5,-0.5,0.7)$。
- **或门（OR）**：$(w_1,w_2,b)=(0.5,0.5,-0.2)$，只要有一个输入为 1，加权和 $0.5$ 就超过 $0.2$。

三个门的**结构完全相同，只有参数不同**。这正是机器学习的出发点：结构（模型）由人设计，参数应该由计算机从数据（这里是真值表）中自动确定。

```python
import numpy as np

def perceptron(x1, x2, w, b):
    tmp = np.sum(np.array(w) * np.array([x1, x2])) + b
    return 0 if tmp <= 0 else 1

def AND(x1, x2):  return perceptron(x1, x2, [0.5, 0.5], -0.7)
def NAND(x1, x2): return perceptron(x1, x2, [-0.5, -0.5], 0.7)
def OR(x1, x2):   return perceptron(x1, x2, [0.5, 0.5], -0.2)
```

### 局限：XOR

异或门（XOR）在两个输入恰好一个为 1 时输出 1。单个感知机无法实现它，原因可以画图看出来。

以或门为例，取 $(b,w_1,w_2)=(-0.5,1.0,1.0)$，感知机在平面上画出直线 $-0.5+x_1+x_2=0$：直线一侧输出 0（只含点 $(0,0)$），另一侧输出 1（含 $(1,0),(0,1),(1,1)$）。**单个感知机的决策边界永远是一条直线**。

而 XOR 的四个点中，$(0,0),(1,1)$ 输出 0，$(1,0),(0,1)$ 输出 1，两类点在对角线上交错，任何一条直线都分不开。能被一条直线分开的问题称为**线性可分**，由直线划分的区域称为线性空间；需要曲线才能划分的称为非线性空间。

### 多层感知机

把门“叠起来”就能绕过这个限制。XOR 可以写成

$$
\mathrm{XOR}(x_1,x_2)=\mathrm{AND}\big(\mathrm{NAND}(x_1,x_2),\ \mathrm{OR}(x_1,x_2)\big).
$$

记 $s_1=\mathrm{NAND}(x_1,x_2)$，$s_2=\mathrm{OR}(x_1,x_2)$，逐行验证：

| $x_1$ | $x_2$ | $s_1$ | $s_2$ | $y$ |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 1 | 0 | 0 |
| 1 | 0 | 1 | 1 | 1 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 1 | 0 | 1 | 0 |

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)
```

画成网络：第 0 层是输入 $x_1,x_2$，第 1 层是 $s_1,s_2$，第 2 层是输出 $y$。有权重的层只有两层（0→1、1→2），所以称为 **2 层感知机**（有的文献按神经元列数称为 3 层）。这种叠加了多层的感知机称为**多层感知机**（multi-layered perceptron, MLP）。

几何上，第一层把输入空间变换成 $(s_1,s_2)$ 平面，在新平面里四个点变得线性可分，第二层再用一条直线切开。**加一层就能表示单层表示不了的东西**，这是“深度”最原始的动机。

> [!INFO]+ 从 NAND 到计算机
>
> 只用 NAND 门的组合就能搭出加法器、编码器乃至整台计算机（参见《计算机系统要素：从零开始构建现代计算机》）。既然感知机能实现 NAND，多层感知机理论上就能表示计算机能做的任何处理。理论上 2 层感知机（激活函数为 sigmoid 等非线性函数）已能表示任意函数，但实际搭建时，分模块逐层构造（与门 → 半加器 → 全加器 → ALU → CPU）要自然得多，这与深度网络“逐层组合特征”的思路一致。

### 从阶跃到平滑

感知机还有一个问题：**参数要靠人凑**。为了让计算机自动学习参数，需要对激活函数做一处关键修改。

把感知机写成两步：先算加权和 $a=b+w_1x_1+w_2x_2$，再用函数 $h$ 转换为输出 $y=h(a)$。这个 $h$ 称为**激活函数**（activation function）。感知机用的是**阶跃函数**：

$$
h(a)=\begin{cases}0 & a\le 0\\ 1 & a>0\end{cases}
$$

阶跃函数的导数几乎处处为 0（在 $a=0$ 处不存在）。这意味着微调参数时输出要么不变、要么突然跳变，无法告诉算法“往哪个方向调、调多少”。把它换成平滑的 **sigmoid 函数**

$$
h(a)=\sigma(a)=\frac{1}{1+e^{-a}}
$$

就进入了神经网络的范畴。例如 $\sigma(1.0)=0.731\ldots$，$\sigma(2.0)=0.880\ldots$。

- **相同点**：两者都是“输入小则输出接近 0，输入大则输出接近 1”，输出都在 $[0,1]$ 内，而且都是**非线性**函数。
- **不同点**：阶跃函数只输出 0 或 1，像“竹筒敲石”一样只有传/不传两种动作；sigmoid 输出连续实数，像“水车”一样按水量调节输出。**正是这种平滑性，使得输出对参数可导，梯度法才能工作。**

> [!INFO]+ 激活函数为什么必须是非线性的
>
> 若激活函数是线性的 $h(x)=cx$，三层叠加 $y=h(h(h(x)))=c^{3}x$，等价于一次乘法 $y=ax$（$a=c^{3}$），即一个没有隐藏层的网络。更一般地，任意多个仿射变换的复合仍是一个仿射变换，加深层数毫无意义。所以**激活函数必须使用非线性函数**。

<br>

> [!NOTE]+ 衔接
>
> 感知机说明了两件事：单层只能画直线，多层才能处理 XOR；阶跃函数不可导，必须换成平滑函数才能自动学习。下一节先做第二件事：只保留一个神经元，把激活换成 sigmoid，并给出自动学习参数的完整方法——这就是逻辑回归。

<br>

## 逻辑回归（Logistic Regression）

### 定义

逻辑回归（logistic regression）是**只有一个 sigmoid 神经元、没有隐藏层**的网络，用于二分类 $y\in\{0,1\}$。它输出的不是 0/1，而是“$y=1$ 的概率”：

$$
\hat{y}=\sigma(w^{\top}x+b)=P(y=1\mid x).
$$

用前面的记号，它就是 $L=1$、$g=\sigma$ 的 $Z=WA+b$。输入是设计矩阵与标签 $(X,Y)$，要学的是参数 $(w,b)$。

它和感知机的区别只有激活函数：阶跃换成 sigmoid。因此它的决策边界 $w^{\top}x+b=0$ 依然是一条直线（超平面），同样**无法处理 XOR**；要处理 XOR，必须像多层感知机那样在前面再插一层非线性。

### 模型

$$
z=w^{\top}x+b,\qquad
\hat{y}=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

sigmoid 的几个性质在推导中会反复用到：

- $\sigma$ 把任意实数压到 $(0,1)$，所以输出可以解释为概率。
- 对称性：$\sigma(z)+\sigma(-z)=1$。
- 导数可以用函数值自身表示：$\sigma'(z)=\sigma(z)(1-\sigma(z))$。
- $|z|$ 很大时 $\sigma(z)$ 贴近 0 或 1，曲线变平，导数接近 0，称为**饱和**。饱和会让梯度消失，这是后文需要更好的初始化与归一化的原因之一。

### 损失函数

#### 由最大似然得到交叉熵

有了模型，需要一个标量来衡量“预测得有多差”，称为**损失函数**（loss function）。逻辑回归的损失不是拍脑袋定的，而是由最大似然推出来的。

模型认为 $y=1$ 的概率是 $\hat{y}$，$y=0$ 的概率是 $1-\hat{y}$。这两种情况可以合写成一个 Bernoulli 似然：

$$
P(y\mid x)=\hat{y}^{y}(1-\hat{y})^{1-y}.
$$

代入 $y=1$ 得 $\hat{y}$，代入 $y=0$ 得 $1-\hat{y}$。好的参数应让观测到的标签概率尽量大。取对数（单调变换，不改变最大值位置，还把乘积变成求和）再取负号，"最大化似然"就变成"最小化损失"，得到**二元交叉熵**（binary cross-entropy）：

$$
\mathcal{L}(\hat{y},y)=-y\log\hat{y}-(1-y)\log(1-\hat{y}).
$$

- $y=1$ 时只剩 $-\log\hat{y}$：预测 $\hat{y}$ 越接近 1 损失越小，$\hat{y}\to 0$ 时损失趋于无穷。
- $y=0$ 时只剩 $-\log(1-\hat{y})$：对称地惩罚把负例预测成正例。
- 实现时把 $\hat{y}$ 截断到 $[\varepsilon,1-\varepsilon]$，避免 $\log 0$。

对整个训练集取平均，得到代价函数（cost function）：

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}\big(\hat{y}^{(i)},y^{(i)}\big).
$$

- 除以 $m$ 让损失与样本数无关，换 batch 大小时学习率仍可比较。
- 加正则项时，正则项是否同样除以 $m$，必须与梯度公式保持一致。
- 独立同分布样本的对数似然是 $\sum_i [y^{(i)}\log a^{(i)}+(1-y^{(i)})\log(1-a^{(i)})]$，最大化它等价于最小化 $J$。
- 逻辑回归是**判别模型**：直接建模 $P(y\mid x)$，不建模 $P(x)$。

#### 两种常用损失：均方误差与交叉熵

一般的神经网络输出向量 $y_k$，标签 $t_k$ 常用 one-hot 表示（正确类为 1，其余为 0）。最常用的两种损失是：

$$
\text{均方误差（MSE）}\quad E=\frac{1}{2}\sum_k (y_k-t_k)^2,
\qquad
\text{交叉熵}\quad E=-\sum_k t_k\log y_k.
$$

由于 $t$ 是 one-hot，交叉熵实际上只看正确类的输出：$E=-\log y_{\text{正确类}}$。

> [!EXAMPLE]+ 例1 两种损失的数值比较（MNIST 十分类）
>
> 设正确答案是“2”，即 $t=(0,0,1,0,0,0,0,0,0,0)$。
>
> - 网络输出 $y=(0.1,0.05,\mathbf{0.6},0,0.05,0.1,0,0.1,0,0)$，认为“2”的概率最高（预测正确）：
>   - MSE $=0.0975$；交叉熵 $=-\log 0.6=0.51$。
> - 网络输出 $y=(0.1,0.05,0.1,0,0.05,0.1,0,\mathbf{0.6},0,0)$，认为“7”的概率最高（预测错误），“2”只有 0.1：
>   - MSE $=0.5975$；交叉熵 $=-\log 0.1=2.30$。
>
> 两种损失都在预测错误时变大，与直觉一致。实现交叉熵时写成 `-np.sum(t * np.log(y + 1e-7))`，加一个极小值 $\delta=10^{-7}$ 防止 `log(0)` 得到 `-inf`。

> [!INFO]+ 为何不用 MSE 做分类
>
> 若 $\mathcal{L}=(a-y)^{2}/2$ 且 $a=\sigma(z)$，由链式法则
> $$
> \frac{\partial\mathcal{L}}{\partial z}=\frac{\partial\mathcal{L}}{\partial a}\cdot\frac{\partial a}{\partial z}=(a-y)\cdot a(1-a).
> $$
> 当预测极错时，例如 $y=1$、$z=-4$，则 $a=\sigma(z)\approx 0.018$，故 $a(1-a)\approx 0.018$，总梯度 $(a-y)\cdot a(1-a)\approx -0.017$，几乎学不动。交叉熵在同一点 $\partial\mathcal{L}/\partial z=a-y\approx -0.98$，仍有强信号。另外，MSE 配 sigmoid 输出对参数不是凸的；二元交叉熵对 $(w,b)$ 是凸的。

#### 为什么优化损失而不是准确率

最终关心的是准确率，为什么不直接优化它？因为**准确率对参数几乎处处不可导**。100 个样本中答对 32 个，准确率是 32%；把某个权重微调一点，答对的个数通常不变，准确率仍是 32%，导数为 0；即使变了也是跳到 33%，是离散的跳变。这和阶跃函数不能用于学习是同一个道理。损失函数则随参数连续变化，微调参数就能看到损失如何变化，梯度才有意义。

同理，预测时把阈值从 0.5 移开会改变准确率，但不改变 $J$；准确率是评估指标，不是训练目标。

#### Mini-batch 损失

训练集很大时（MNIST 有 6 万张），每一步都在全部数据上算损失太慢。常用做法是每次随机抽一小批（mini-batch），例如 100 个样本，用这一批的平均损失近似整体损失：

```python
batch_mask = np.random.choice(train_size, batch_size)
x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]
```

这就像用电视收视率抽样估计全体观众。mini-batch 的大小、遍历方式与噪声的影响在“优化”一节详细讨论。

### 梯度

#### 导数与数值微分

有了损失 $J(w,b)$，下一步是知道“参数往哪边动，损失会下降”，这由导数给出。导数的定义是

$$
\frac{\mathrm{d}f(x)}{\mathrm{d}x}=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
$$

直接用很小的 $h$ 近似这个极限，称为**数值微分**（numerical differentiation）。实现时有两个陷阱：

1. **舍入误差**：$h$ 不能太小。`np.float32(1e-50)` 会直接变成 `0.0`，一般取 $h=10^{-4}$。
2. **截断误差**：前向差分 $(f(x+h)-f(x))/h$ 算的其实是 $x$ 与 $x+h$ 之间割线的斜率，与 $x$ 处切线有偏差。改用**中心差分**可把误差从 $O(h)$ 降到 $O(h^2)$：

$$
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}.
$$

```python
def numerical_diff(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)
```

> [!EXAMPLE]+ 例2 数值微分与解析微分
>
> $f(x)=0.01x^{2}+0.1x$，解析导数 $f'(x)=0.02x+0.1$。
>
> - $x=5$：数值微分得 $0.1999999999990898$，真值 $0.2$；
> - $x=10$：数值微分得 $0.2999999999986347$，真值 $0.3$。
>
> 误差非常小。数值微分简单可靠，但每个参数都要算两次 $f$，参数一多就太慢；所以训练用解析梯度（反向传播），数值微分只用来**检查**解析梯度写得对不对（见“梯度检查”）。

#### 偏导数与梯度

有多个参数时，对其中一个求导、其余固定，称为**偏导数**。把所有偏导数排成向量，就是**梯度**：

$$
\nabla f=\left(\frac{\partial f}{\partial x_0},\frac{\partial f}{\partial x_1},\dots\right).
$$

> [!EXAMPLE]+ 例3 梯度的方向
>
> $f(x_0,x_1)=x_0^{2}+x_1^{2}$，$\nabla f=(2x_0,2x_1)$。
>
> - 在 $(3,4)$：数值偏导为 $6.00000000000378$ 与 $7.999999999999119$，即梯度 $(6,8)$；
> - 在 $(0,2)$：梯度 $(0,4)$；在 $(3,0)$：梯度 $(6,0)$。
>
> 把各点的**负梯度**画成箭头，箭头都指向最低点 $(0,0)$，离最低点越远箭头越长。**负梯度是函数值在该点下降最快的方向**。注意它指向的是局部下降方向，不保证指向全局最小值。

#### 梯度下降

既然负梯度指向下降最快的方向，就沿它走一小步，重复多次。这就是**梯度下降**（gradient descent）：

$$
\theta\leftarrow\theta-\alpha\,\frac{\partial J}{\partial\theta}.
$$

$\alpha$ 称为**学习率**（learning rate），决定每一步走多远。它不是学出来的，需要人工设定，属于**超参数**。

> [!EXAMPLE]+ 例4 学习率过大或过小
>
> 对 $f(x_0,x_1)=x_0^{2}+x_1^{2}$，从 $(-3,4)$ 出发做 100 步梯度下降：
>
> - $\alpha=0.1$：到达 $(-6.1\times 10^{-10},\ 8.1\times 10^{-10})$，几乎就是真正的最小值 $(0,0)$；
> - $\alpha=10$：发散到 $(-2.58\times 10^{13},\ -1.29\times 10^{12})$；
> - $\alpha=10^{-10}$：几乎不动，停在 $(-2.99999994,\ 3.99999992)$。
>
> 实际训练中，$\alpha$ 过大表现为 $J$ 震荡或出现 `nan`，过小表现为数百步几乎不动。

#### 逻辑回归的解析梯度

对逻辑回归，可以直接用链式法则求出梯度的解析式，不必数值微分。

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
> \frac{\partial\mathcal{L}}{\partial z}=\left(-\frac{y}{a}+\frac{1-y}{1-a}\right)a(1-a)=-y(1-a)+(1-y)a=a-y.
> $$
>
> 因此 sigmoid + 二元交叉熵的输出层误差就是“预测减标签” $a-y$，不需要再显式乘 $\sigma'(z)$。多分类中 softmax + 交叉熵有同样的简化：$\partial\mathcal{L}/\partial z = a-y$。这个简洁的结果不是巧合，而是交叉熵被设计成与 sigmoid / softmax 配套的结果。

再由 $z=w^{\top}x+b$ 得 $\partial z/\partial w=x$、$\partial z/\partial b=1$，乘上 $\partial\mathcal{L}/\partial z=a-y$，得到单样本梯度：

$$
\frac{\partial\mathcal{L}}{\partial w}=(a-y)x,\qquad
\frac{\partial\mathcal{L}}{\partial b}=a-y.
$$

含义很直观：预测偏高（$a>y$）时，把 $w$ 往 $-x$ 方向推，使下次对同一个 $x$ 的输出变小。

对 $m$ 个样本求平均。循环写法是对每个 $i$ 执行 $\mathrm{d}w \mathrel{+}= (a^{(i)}-y^{(i)})x^{(i)}$，最后除以 $m$。向量化写法令 $A=\sigma(w^{\top}X+b)\in\mathbb{R}^{1\times m}$：

$$
\mathrm{d}w=\frac{1}{m}X(A-Y)^{\top},\qquad
\mathrm{d}b=\frac{1}{m}\sum_{i=1}^{m}(A-Y)_{i}.
$$

- 形状核对：$X$ 为 $(n_x,m)$，$(A-Y)^{\top}$ 为 $(m,1)$，乘积为 $(n_x,1)$，与 $w$ 同形。
- 误用 `X @ (A-Y)` 而不转置，会因形状不匹配报错或被广播成错误的形状。

最后用梯度下降更新：

$$
w\leftarrow w-\alpha\,\mathrm{d}w,\qquad
b\leftarrow b-\alpha\,\mathrm{d}b.
$$

在猫分类作业上，$\alpha=0.005$、$2000$ 步是一个能让损失下降的常见起点，但不是普遍最优。

### 算法

1. **预处理**：图像 `(64, 64, 3)` 展平成 `(12288, 1)`，像素除以 $255$ 缩放到 $[0,1]$，按列拼成 $X$。训练与推理必须使用同一 `reshape` 顺序。
2. **初始化**：$w=0$、$b=0$。逻辑回归只有一个神经元，不存在隐藏单元的对称性问题，零初始化合法。
3. **迭代**：计算 $A=\sigma(w^{\top}X+b)$；计算 $J$；计算 $\mathrm{d}w,\mathrm{d}b$；更新参数。
4. **监控**：每隔固定步记录 $J$。若 $J$ 不降，依次检查 shape、学习率，以及标签是否为 $\{0,1\}$ 而非 $\{-1,1\}$。
5. **预测**：$\hat{y}\ge 0.5$ 判为正类；也可以按假阳性与假阴性的代价调整阈值。准确率不是训练目标：移动阈值会改变准确率，但不改变 $J$。

几点性质：

- 每次迭代的时间为 $O(m n_x)$。
- 逻辑回归的 $J$ 对 $(w,b)$ 是凸的，梯度下降能到达全局最优；一旦加入隐藏层，凸性就消失了。
- 若特征尺度相差几个数量级，$J$ 的等高线会被拉成狭长的山谷，梯度下降来回震荡。因此要做标准化，而且**只用训练集的统计量** $\mu,\sigma$，测试集复用同一组 $\mu,\sigma$。

### 实现

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

> [!EXAMPLE]+ 例5 决策边界
>
> - 二维输入 $x=(x_1,x_2)$ 时，$\hat{y}=0.5$ 等价于 $z=0$，决策边界是直线 $w_1 x_1+w_2 x_2+b=0$。
> - 正类线性可分时，训练后法向量 $w$ 指向正类一侧。
> - 对异或四点，无论怎么训练，损失都有一个正的下界，四个点无法同时分对。
> - 这就是引入隐藏层的动机。

<br>

> [!NOTE]+ 衔接
>
> 逻辑回归给出了“模型 → 损失 → 梯度 → 更新”的完整训练闭环，以及输出层误差 $A-Y$。但它的边界仍是直线。下一节在输入与输出之间插入一层带非线性激活的隐藏层，让边界可以弯曲——相当于把多层感知机的结构和逻辑回归的学习方法合在一起。

<br>

## 浅层网络（Shallow Neural Network）

### 结构

浅层网络指**一层隐藏层加一层输出层**，即 $L=2$。隐藏层的神经元从输入中提取新的特征，输出层再在这些特征上做逻辑回归。隐藏层常用 tanh 或 ReLU，输出层二分类用 sigmoid、多分类用 softmax。

层数的数法：输入层、隐藏层、输出层共三列神经元，但只有两层带权重，所以称为“2 层网络”（鱼书同样按带权重的层数计）。

### 前向

$$
\begin{aligned}
Z^{[1]} &= W^{[1]}X + b^{[1]}, &
A^{[1]} &= g^{[1]}(Z^{[1]}),\\
Z^{[2]} &= W^{[2]}A^{[1]} + b^{[2]}, &
A^{[2]} &= g^{[2]}(Z^{[2]}).
\end{aligned}
$$

各量形状：

- $W^{[1]}$：$(n^{[1]}, n_x)$
- $A^{[1]}$：$(n^{[1]}, m)$
- $W^{[2]}$：$(n_y, n^{[1]})$
- $A^{[2]}$：$(n_y, m)$

每一层都是“矩阵乘 + 加偏置 + 激活”，用 `np.dot` 一次算完一层所有神经元。不用矩阵乘法就得为每个神经元写一个循环，所以向量化在实现层面非常关键。

> [!EXAMPLE]+ 例6 一个 3 层网络的前向计算（鱼书 3.4 节）
>
> 网络为 2 → 3 → 2 → 2，隐藏层用 sigmoid，输出层用恒等函数。鱼书使用行样本约定（$a=xW+b$）：
>
> - 输入 $x=(1.0,\ 0.5)$；
> - $W^{(1)}=\begin{pmatrix}0.1&0.3&0.5\\0.2&0.4&0.6\end{pmatrix}$，$b^{(1)}=(0.1,0.2,0.3)$；
> - $W^{(2)}=\begin{pmatrix}0.1&0.4\\0.2&0.5\\0.3&0.6\end{pmatrix}$，$b^{(2)}=(0.1,0.2)$；
> - $W^{(3)}=\begin{pmatrix}0.1&0.3\\0.2&0.4\end{pmatrix}$，$b^{(3)}=(0.1,0.2)$。
>
> 逐层计算：
>
> - 第 1 层加权和 $a^{(1)}=xW^{(1)}+b^{(1)}=(0.3,\ 0.7,\ 1.1)$，例如第一个分量 $1.0\times 0.1+0.5\times 0.2+0.1=0.3$；激活 $z^{(1)}=\sigma(a^{(1)})=(0.574,\ 0.668,\ 0.750)$；
> - 第 2 层 $a^{(2)}=(0.516,\ 1.214)$，$z^{(2)}=(0.626,\ 0.771)$；
> - 输出 $y=z^{(2)}W^{(3)}+b^{(3)}=(0.3168,\ 0.6963)$。
>
> 形状链为 $(2,)\times(2,3)\to(3,)\times(3,2)\to(2,)\times(2,2)\to(2,)$，每一步“相邻维度一致”。

### 激活函数

隐藏层的激活函数决定了网络能表示什么样的非线性。常用的选择如下，每个都给出公式、导数、用途与缺点。

> [!NOTE]+ 激活函数
>
> **sigmoid（logistic function，逻辑函数）**
>
> $$
> g(z)=\frac{1}{1+e^{-z}},\qquad g'(z)=g(z)\big(1-g(z)\big).
> $$
>
> 用途：值域 $(0,1)$，二分类输出直接当概率。缺点：$|z|$ 大时导数接近 $0$（最大值也只有 $0.25$），多层连乘后梯度消失；输出不以 $0$ 为中心，下一层输入整体偏正。
>
> **tanh（hyperbolic tangent，双曲正切）**
>
> $$
> g(z)=\tanh z=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}},\qquad g'(z)=1-\tanh^{2}z.
> $$
>
> 用途：值域 $(-1,1)$ 且零中心，浅层隐藏层和 RNN 中比 sigmoid 好训。缺点：$|z|$ 大时同样饱和，深层仍会梯度消失。
>
> **ReLU（rectified linear unit，整流线性单元）**
>
> $$
> g(z)=\max(0,z),\qquad g'(z)=\begin{cases} 0 & z<0 \\ 1 & z>0 \end{cases}.
> $$
>
> 用途：正半轴导数恒为 $1$，不饱和，计算只是一次比较，所以是卷积网络隐藏层的默认选择。缺点：负半轴导数恒为 $0$，单元若一直停在负区就不再更新，称为死亡 ReLU（dying ReLU）。$z=0$ 处导数不唯一，实现里通常把 $z\le 0$ 的导数写成 $0$。
>
> **Leaky ReLU（leaky rectified linear unit，渗漏整流线性）**
>
> $$
> g(z)=\max(\alpha z, z),\qquad g'(z)=\begin{cases} \alpha & z<0 \\ 1 & z>0 \end{cases},\quad \alpha=0.01.
> $$
>
> 用途：负半轴仍有梯度，避免永久死亡。缺点：$\alpha$ 很小，形状仍接近 ReLU，负信号几乎传不远。
>
> **ELU（exponential linear unit，指数线性单元）**
>
> $$
> g(z)=\begin{cases} z & z>0 \\ a(e^{z}-1) & z\le 0 \end{cases},\qquad
> g'(z)=\begin{cases} 1 & z>0 \\ g(z)+a & z\le 0 \end{cases}.
> $$
>
> 常取 $a=1$。用途：负值平滑，激活均值更靠近 $0$，比固定小斜率更容易把单元拉回零附近；处处可导（$a=1$ 时）。缺点：负半轴多一次指数，比 ReLU 贵。
>
> **GELU（Gaussian error linear unit，高斯误差线性单元）**
>
> $$
> g(z)=z\,\Phi(z),\qquad g'(z)=\Phi(z)+z\,\phi(z).
> $$
>
> $\Phi$ 是标准正态累积分布，$\phi$ 是其密度。用途：负半轴按概率缩小输入，而不是硬截成 $0$，不会死亡；Transformer 与语言模型的隐藏层常用它。缺点：要比 ReLU 多算 $\Phi$。实现里常用 $z\,\sigma(1.702z)$ 近似。
>
> **hard tanh（hard hyperbolic tangent，硬双曲正切）**
>
> $$
> g(z)=\begin{cases} -1 & z<-1 \\ z & -1\le z\le 1 \\ 1 & z>1 \end{cases},\qquad
> g'(z)=\begin{cases} 1 & |z|<1 \\ 0 & |z|>1 \end{cases}.
> $$
>
> 用途：CS224n 把它当作 tanh 的便宜替代，区间内就是恒等，没有指数。缺点：区间外饱和，导数为 $0$，和 tanh 一样会消失。
>
> **softsign（软符号）**
>
> $$
> g(z)=\frac{z}{1+|z|},\qquad g'(z)=\frac{1}{(1+|z|)^{2}}.
> $$
>
> 用途：值域 $(-1,1)$，零中心，导数恒正；尾部按多项式衰减，比 tanh 更晚贴到 $\pm 1$，比 tanh 少一次指数。缺点：仍会在远处变平，深层一样会饱和；比 hard tanh 多一次除法。
>
> **Softplus（软加）**
>
> $$
> g(z)=\log(1+e^{z}),\qquad g'(z)=\frac{e^{z}}{1+e^{z}}=\sigma(z).
> $$
>
> 用途：ReLU 的光滑版本，值域 $(0,\infty)$，常用来输出必须为正的量（方差、尺度参数）。缺点：负半轴导数 $\sigma(z)$ 也趋于 $0$；做隐藏层激活时一般不比 ReLU 好。
>
> **PReLU（parametric ReLU，参数化 ReLU）**
>
> $$
> g(z)=\max(\alpha z,z),\qquad \frac{\partial g}{\partial z}=\begin{cases}\alpha & z<0\\ 1 & z>0\end{cases},\qquad
> \frac{\partial g}{\partial \alpha}=\begin{cases} z & z<0\\ 0 & z>0\end{cases}.
> $$
>
> 与 Leaky ReLU 形状相同，但 $\alpha$ 是可学习参数（通常每个通道一个），由反向传播决定负半轴斜率。He 初始化的论文（He et al., 2015）就是用 PReLU 在 ImageNet 上首次超过人类水平。缺点：多了参数，小数据上可能过拟合。
>
> **SELU（scaled ELU，缩放指数线性单元）**
>
> $$
> g(z)=\lambda\begin{cases} z & z>0\\ a(e^{z}-1) & z\le 0\end{cases},\qquad \lambda\approx1.0507,\ a\approx1.6733.
> $$
>
> 两个常数是解出来的：在 LeCun 初始化（$\mathrm{Var}(w)=1/n_{\mathrm{in}}$）下，若输入均值 $0$、方差 $1$，输出的均值、方差仍为 $0$、$1$，称为自归一化（self-normalizing）。用途：不加 BatchNorm 的深层全连接网络。缺点：只对全连接、特定初始化、配合 AlphaDropout 时成立，卷积与残差网络中优势不明显。
>
> **SiLU / Swish（sigmoid-weighted linear unit）**
>
> $$
> g(z)=z\,\sigma(z),\qquad g'(z)=\sigma(z)+z\,\sigma(z)\big(1-\sigma(z)\big)=\sigma(z)\big(1+z(1-\sigma(z))\big).
> $$
>
> 形状与 GELU 几乎重合（GELU 的近似式 $z\,\sigma(1.702z)$ 就是 Swish 加了一个系数）。光滑、非单调：在 $z\approx-1.28$ 处有最小值约 $-0.28$，负半轴先降后回到 $0$。EfficientNet、YOLO 系列与很多大语言模型使用它。
>
> **GLU 系列（gated linear unit，门控线性单元）**
>
> $$
> \mathrm{GLU}(x)=(xW+b)\odot\sigma(xV+c),\qquad
> \mathrm{SwiGLU}(x)=\mathrm{SiLU}(xW)\odot(xV).
> $$
>
> 它不是逐元素函数，而是两路线性投影相乘：一路给内容，一路当门，思路与 LSTM 的门相同。LLaMA 等模型的 FFN 用 SwiGLU 代替 ReLU。为了保持参数量不变，隐藏宽度常从 $4d$ 改成约 $\frac{8}{3}d$（三个矩阵 $W,V,W_2$ 而不是两个）。
>
> **选择**：隐藏层默认 ReLU；需要光滑负值时用 GELU 或 ELU。输出层按任务：回归用恒等函数，二分类用 sigmoid，多分类用 softmax，不把 ReLU 放在概率输出上。

<br>

> [!INFO]+ ReLU 为什么成为默认
>
> ReLU 由 Nair & Hinton (2010) 提出，AlexNet 之后卷积网络的隐藏层基本都用它。原因在于导数：sigmoid 的导数最大只有 $0.25$，tanh 在 $|z|$ 稍大时导数也只有 $0.1$ 左右，十层乘下来梯度接近 0；ReLU 在正半轴导数恒为 1，梯度可以原样穿过很多层。代价是负半轴梯度为 0，神经元可能“死掉”，Leaky ReLU 给负半轴一个很小的斜率来缓解。

<br>

> [!EXAMPLE]+ 例7 反向传播中如何计算激活函数的导数
>
> 前向已经得到 $a=g(z)$。反向时直接用下面的式子，多数情况下能复用前向结果，不必重新计算：
>
> - sigmoid：$g'(z)=a(1-a)$
> - tanh：$g'(z)=1-a^{2}$
> - ReLU：$z>0$ 时为 $1$，否则为 $0$
> - Leaky ReLU：$z>0$ 时为 $1$，否则为 $\alpha$
> - ELU：$z>0$ 时为 $1$，否则为 $g(z)+a$
> - GELU：$g'(z)=\Phi(z)+z\,\phi(z)$
> - hard tanh：$|z|<1$ 时为 $1$，否则为 $0$
> - softsign：$g'(z)=1/(1+|z|)^{2}$
>
> sigmoid 与 tanh 在远离 $0$ 时导数接近 $0$，多层连乘后梯度消失；ReLU、Leaky ReLU、ELU 在正半轴导数为 $1$。所以隐藏层用后者，输出层才用 sigmoid 或 softmax。

NumPy 实现：每个激活写成“前向 + 导数”一对，反向时 `dZ = dA * g_prime(Z, A)`。

```python
import numpy as np
from math import sqrt, pi

def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))

ACTIVATIONS = {
    #  名称:   (前向 g(z),                         导数 g'(z)，可复用前向结果 a)
    "sigmoid": (_sig,                              lambda z, a: a * (1 - a)),
    "tanh":    (np.tanh,                           lambda z, a: 1 - a ** 2),
    "relu":    (lambda z: np.maximum(0, z),        lambda z, a: (z > 0).astype(z.dtype)),
    "leaky":   (lambda z: np.where(z > 0, z, 0.01 * z),
                lambda z, a: np.where(z > 0, 1.0, 0.01)),
    "elu":     (lambda z: np.where(z > 0, z, np.expm1(z)),
                lambda z, a: np.where(z > 0, 1.0, a + 1.0)),
    "gelu":    (lambda z: z * _sig(1.702 * z),     # 近似 z·Φ(z)
                lambda z, a: _sig(1.702 * z) * (1 + 1.702 * z * (1 - _sig(1.702 * z)))),
    "softsign": (lambda z: z / (1 + np.abs(z)),    lambda z, a: 1 / (1 + np.abs(z)) ** 2),
}

def activation_backward(dA, Z, A, name):
    """dA, Z, A 同形 (n_l, m)。返回 dZ = dA ⊙ g'(Z)。"""
    _, g_prime = ACTIVATIONS[name]
    return dA * g_prime(Z, A)
```

PyTorch 用 autograd 核对上面的导数：对 `g(z).sum()` 求反向，`z.grad` 就是逐点的 $g'(z)$。

```python
import torch
import torch.nn.functional as F

z = torch.linspace(-3, 3, 13, dtype=torch.float64, requires_grad=True)
for name, fn in {"sigmoid": torch.sigmoid, "tanh": torch.tanh,
                 "relu": F.relu, "elu": F.elu}.items():
    z.grad = None
    fn(z).sum().backward()
    g, g_prime = ACTIVATIONS[name]
    zn = z.detach().numpy()
    ours = g_prime(zn, g(zn))
    print(name, np.abs(ours - z.grad.numpy()).max())   # 约 1e-16
```

ReLU 在 $z=0$ 处 PyTorch 取导数 $0$，与上面 `z > 0` 的写法一致；网格若恰好落在 $0$ 上，换别的约定就会对不上。

#### 导数推导

上面的导数都可以一步步推出来，推导本身也说明了“为什么能复用前向结果”。

**sigmoid**。把 $\sigma(z)=(1+e^{-z})^{-1}$ 看成复合函数，用链式法则：

$$
\sigma'(z)=-(1+e^{-z})^{-2}\cdot(-e^{-z})
=\frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}}
=\sigma(z)\cdot\big(1-\sigma(z)\big).
$$

最后一步用了 $\frac{e^{-z}}{1+e^{-z}}=1-\frac{1}{1+e^{-z}}$。$\sigma(1-\sigma)$ 在 $\sigma=\frac12$（即 $z=0$）时取最大值 $\frac14$，这个 $\frac14$ 是后面梯度消失估计的关键数字。

**tanh**。用商的求导法则，记 $u=e^{z}-e^{-z}$、$w=e^{z}+e^{-z}$，则 $u'=w$、$w'=u$：

$$
\tanh'(z)=\frac{u'w-uw'}{w^{2}}=\frac{w^{2}-u^{2}}{w^{2}}=1-\tanh^{2}z.
$$

$z=0$ 时导数为 $1$，比 sigmoid 的 $\frac14$ 大 4 倍。tanh 其实就是平移、缩放过的 sigmoid：

$$
\tanh z=\frac{1-e^{-2z}}{1+e^{-2z}}=\frac{2}{1+e^{-2z}}-1=2\sigma(2z)-1.
$$

所以二者表达能力相同，区别只在“是否零中心”和“$0$ 附近的斜率”。

**Softplus 与 SiLU**。$\frac{\mathrm d}{\mathrm dz}\log(1+e^{z})=\frac{e^{z}}{1+e^{z}}=\sigma(z)$：Softplus 的导数正好是 sigmoid，sigmoid 的积分就是 Softplus。SiLU 是乘积 $z\cdot\sigma(z)$，用乘积法则得 $\sigma(z)+z\sigma'(z)=\sigma(z)+z\sigma(z)(1-\sigma(z))$。

**GELU**。$g(z)=z\Phi(z)$，乘积法则加上 $\Phi'=\phi$，得 $g'(z)=\Phi(z)+z\phi(z)$，其中 $\phi(z)=\frac{1}{\sqrt{2\pi}}e^{-z^{2}/2}$。

**ReLU 在 0 处**。$\max(0,z)$ 在 $z=0$ 处不可导，但左右导数 $0$ 和 $1$ 之间的任何值都是次梯度（subgradient）。实际训练中 $z$ 恰好等于 $0$ 的概率几乎为零，取 $0$ 或 $1$ 都不影响结果。

#### 为什么零中心重要

考虑下一层的一个神经元 $z=\sum_i w_i a_i+b$，它的权重梯度是

$$
\frac{\partial L}{\partial w_i}=\frac{\partial L}{\partial z}\cdot a_i.
$$

若上一层用 sigmoid，所有 $a_i>0$，那么同一个样本给出的所有 $\partial L/\partial w_i$ 都与 $\partial L/\partial z$ **同号**：这一步要么所有 $w_i$ 一起变大，要么一起变小。若最优方向是“$w_1$ 增大、$w_2$ 减小”，就只能交替走“全增”“全减”两种方向，路径呈之字形，收敛变慢。tanh 的输出有正有负，没有这个限制。这也是数据预处理要减均值、BatchNorm 要把预激活拉回零均值的原因之一。mini-batch 平均会把这个问题冲淡一些，但不能消除。

#### 梯度消失的定量估计

反向传播时，每经过一层，梯度都乘一次 $g'(z)$（还有权重矩阵）。只看激活函数这一项：

| 激活 | $g'$ 的最大值 | 10 层连乘的上界 |
| :--- | :--- | :--- |
| sigmoid | $1/4$ | $(1/4)^{10}\approx 9.5\times10^{-7}$ |
| tanh | $1$（仅在 $z=0$） | 实际 $\lvert z\rvert$ 稍大就远小于 $1$ |
| ReLU | $1$（$z>0$ 时恒为 $1$） | 激活路径上不衰减 |

sigmoid 即使在最理想的位置，10 层也让梯度缩小约一百万倍：前几层几乎学不动。ReLU 在激活的位置导数恰好为 $1$，梯度大小只由权重决定，这时配合 He 初始化让权重项的连乘也保持在 $1$ 附近，深层网络才训得动。

#### 死亡 ReLU

ReLU 单元若对**所有**训练样本都有 $z\le 0$，它的输出恒为 $0$、梯度恒为 $0$，权重再也不会更新，这个单元就“死”了。常见原因：

- 学习率过大：一次大更新把偏置推到很负的位置；
- 偏置初始化为较大的负数；
- 输入没有归一化，某些方向尺度极大。

诊断方法是统计每个隐藏单元在一个 batch 上的激活率：

```python
def dead_relu_fraction(Z):
    """Z: (n_l, m) 某层 ReLU 之前的预激活。返回对整批都不激活的单元比例。"""
    return np.mean(np.all(Z <= 0, axis=1))
```

比例超过 10% 到 20% 时，可以减小学习率、改用 Leaky ReLU / GELU，或把偏置初始化为小的正数（如 $0.01$）。

#### 对比与选择

| 激活 | 值域 | 零中心 | 饱和 | 会死亡 | 计算 | 常见位置 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sigmoid | $(0,1)$ | 否 | 两侧 | 否 | 指数 | 二分类输出、门 |
| tanh | $(-1,1)$ | 是 | 两侧 | 否 | 指数 | RNN 隐状态、候选值 |
| ReLU | $[0,\infty)$ | 否 | 负侧梯度为 0 | 会 | 比较 | CNN 隐藏层 |
| Leaky / PReLU | $\mathbb{R}$ | 近似 | 否 | 否 | 比较 | 担心死亡时 |
| ELU / SELU | $(-a,\infty)$ | 近似 | 负侧缓饱和 | 否 | 负侧指数 | 无 BN 的全连接 |
| GELU / SiLU | 约 $(-0.17,\infty)$ / $(-0.28,\infty)$ | 近似 | 否 | 否 | 指数或误差函数 | Transformer、现代 CNN |
| softmax | 单纯形 | 否 | 是 | 否 | 指数 | 多分类输出 |

选择的原则：

1. **输出层由任务决定，并且和损失函数配对**（见下面的“损失函数的选择”）：回归用恒等，二分类和多标签用 sigmoid，单标签多分类用 softmax，必须为正的量用 Softplus 或 $\exp$。
2. **隐藏层先用 ReLU**，配 He 初始化。出现大量死亡单元时换 Leaky ReLU；做 Transformer 或追求最后一点精度时用 GELU / SiLU；FFN 可以用 SwiGLU。
3. **门控结构保留 sigmoid 和 tanh**：LSTM / GRU 的门需要 $(0,1)$ 之间的“开关比例”，只能用 sigmoid；候选值需要有正有负，用 tanh。这里的饱和反而是想要的：门要么开、要么关。
4. **激活与初始化要成对**：ReLU 系列配 He（$\mathrm{Var}=2/n_{\mathrm{in}}$），tanh / sigmoid 配 Xavier，SELU 配 LeCun（$1/n_{\mathrm{in}}$）。
5. 换激活函数通常只带来很小的提升，排在数据、架构、学习率之后再调。

补充的几种激活的 NumPy 实现，并用 PyTorch 核对导数：

```python
SELU_L, SELU_A = 1.0507009873554805, 1.6732632423543772

MORE_ACTIVATIONS = {
    "softplus": (lambda z: np.logaddexp(0, z),          # log(1+e^z)，大 z 不溢出
                 lambda z, a: _sig(z)),
    "silu":     (lambda z: z * _sig(z),
                 lambda z, a: _sig(z) * (1 + z * (1 - _sig(z)))),
    "selu":     (lambda z: SELU_L * np.where(z > 0, z, SELU_A * np.expm1(z)),
                 lambda z, a: np.where(z > 0, SELU_L, a + SELU_L * SELU_A)),
}

def swiglu(x, W, V):
    """x: (m, d)；W, V: (d, d_ff)。返回 (m, d_ff)，后面再乘 W2 回到 d 维。"""
    h = x @ W
    return h * _sig(h) * (x @ V)

z = torch.linspace(-4, 4, 17, dtype=torch.float64, requires_grad=True)
for name, fn in {"softplus": F.softplus, "silu": F.silu, "selu": F.selu}.items():
    z.grad = None
    fn(z).sum().backward()
    g, g_prime = MORE_ACTIVATIONS[name]
    zn = z.detach().numpy()
    print(name, np.abs(g(zn) - fn(z).detach().numpy()).max(),
          np.abs(g_prime(zn, g(zn)) - z.grad.numpy()).max())    # 都约 1e-16

# SELU 的自归一化：标准正态输入、LeCun 初始化，20 层后均值方差仍接近 0 与 1
rng = np.random.default_rng(0)
a = rng.standard_normal((256, 2000))
for _ in range(20):
    a = MORE_ACTIVATIONS["selu"][0](rng.standard_normal((256, 256)) / np.sqrt(256) @ a)
print(a.mean().round(3), a.var().round(3))                     # 0.003 1.035
```

SELU 的导数在 $z\le0$ 处可以复用前向结果：$g=\lambda a(e^{z}-1)$，所以 $g'=\lambda a e^{z}=g+\lambda a$，代码里的 `a + SELU_L * SELU_A` 就是这个式子（`a` 是前向输出，`SELU_A` 是常数 $a$）。

<br>

### 隐藏层的作用

- **非线性不可省略**。若 $g^{[1]}$ 为恒等函数，则

$$
A^{[2]}=g^{[2]}\big(W^{[2]}W^{[1]}X + W^{[2]}b^{[1]}+b^{[2]}\big),
$$

  记 $W'=W^{[2]}W^{[1]}$、$b'=W^{[2]}b^{[1]}+b^{[2]}$，就退化成单层的 $g^{[2]}(W'X+b')$，决策边界仍是直线。异或四个点需要把空间“折叠”，隐藏层的响应必须是弯曲的。
- **通用逼近**（universal approximation）
  - 足够宽的单隐藏层网络可以以任意精度逼近紧集上的连续函数；
  - 但所需宽度可能是指数级的；
  - 定理只说明这样的参数存在，不保证训练能找到它；
  - 深度用“组合”的方式复用特征，常常比单纯加宽更省参数。
- **参数量**：$n^{[1]}(n_x+1)+n_y(n^{[1]}+1)$（每个神经元有若干权重和一个偏置）。
  - 平面分类作业常用 $n_x=2$、$n^{[1]}=4$、$n_y=1$，共 $4\times 3+1\times 5=17$ 个参数；
  - 隐藏单元太少：拼不出弯曲的边界；
  - 隐藏单元太多而数据只有几百点：过拟合。

### 输出层

输出层的激活函数由任务决定：**回归**（预测连续值，如由照片预测体重 57.4 kg）用恒等函数，直接输出；**二分类**用 sigmoid；**多分类**用 softmax。输出层的神经元个数一般等于类别数，例如手写数字 0–9 有 10 个输出神经元。

#### softmax

$K$ 类分类时，输出层先算出 $K$ 个分数 $z_1,\dots,z_K$，softmax 把它们变成概率：

$$
a_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}.
$$

分子是第 $k$ 个分数的指数，分母是所有指数之和，所以每个输出都受**所有**输入影响。

- **输出可当概率**：每个 $a_k\in(0,1)$，且 $\sum_k a_k=1$。例如 $z=(0.3,2.9,4.0)$ 时 $a=(0.018,\ 0.245,\ 0.737)$，可以说“有 74% 的概率是第 2 类，25% 的概率是第 1 类，1% 的概率是第 0 类”。
- **不改变大小顺序**：指数函数单调递增，所以 $\arg\max_k a_k=\arg\max_k z_k$。推理时只需取最大分数对应的类别，**softmax 可以省略**以节省计算；训练时需要它，是因为损失函数要用到概率。
- 二分类时，softmax 退化为 sigmoid：$\frac{e^{z_1}}{e^{z_1}+e^{z_0}}=\sigma(z_1-z_0)$。

#### 溢出问题

指数增长极快：$e^{10}>20000$，$e^{100}$ 后面有 40 多个 0，$e^{1000}$ 直接返回 `inf`，两个 `inf` 相除得到 `nan`。解决办法是分子分母同乘一个常数 $C$：

$$
a_k=\frac{Ce^{z_k}}{C\sum_j e^{z_j}}=\frac{e^{z_k+\log C}}{\sum_j e^{z_j+\log C}}=\frac{e^{z_k-c}}{\sum_j e^{z_j-c}}.
$$

即**所有分数同时减去同一个常数，结果不变**。取 $c=\max_j z_j$，最大的指数变成 $e^{0}=1$，其余都小于 1，不会溢出。

> [!EXAMPLE]+ 例8 softmax 的溢出与修正
>
> ```python
> a = np.array([1010, 1000, 990])
> np.exp(a) / np.sum(np.exp(a))          # array([nan, nan, nan])
> c = np.max(a)                           # 1010
> np.exp(a - c) / np.sum(np.exp(a - c))  # array([9.9995e-01, 4.5398e-05, 2.0611e-09])
> ```

### 损失函数的选择

> [!INFO]+ 一条主线
>
> 几乎所有常用损失都是**负对数似然**：先假设“给定 $x$ 时 $y$ 服从什么分布”，输出层给出这个分布的参数，损失就是 $-\log p(y\mid x)$。假设高斯得到 MSE，假设拉普拉斯得到 MAE，假设伯努利得到二元交叉熵，假设类别分布得到交叉熵。输出激活与损失因此成对出现，配对正确时 $\partial L/\partial z$ 都化简成“预测减标签”。

<br>

#### 总表

| 任务 | 假设的分布 | 输出激活 | 损失 | $\partial L/\partial z$ | PyTorch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 回归 | 高斯 | 恒等 | MSE $\frac12(\hat y-y)^2$ | $\hat y-y$ | `MSELoss` |
| 回归，有离群点 | 拉普拉斯 | 恒等 | MAE $\lvert\hat y-y\rvert$ | $\mathrm{sign}(\hat y-y)$ | `L1Loss` |
| 回归，折中 | 高斯与拉普拉斯的拼接 | 恒等 | Huber | 截断的 $\hat y-y$ | `HuberLoss`、`SmoothL1Loss` |
| 二分类 | 伯努利 | sigmoid | BCE | $a-y$ | `BCEWithLogitsLoss` |
| 多标签 | 每类独立伯努利 | 每维 sigmoid | 各维 BCE 之和 | $a_k-y_k$ | `BCEWithLogitsLoss` |
| 单标签多分类 | 类别分布 | softmax | 交叉熵 | $a_k-t_k$ | `CrossEntropyLoss` |
| 计数 | 泊松 | $\exp$ | $\hat y-y\log\hat y$ | $\hat y-y$ | `PoissonNLLLoss` |

最后一列的 PyTorch 损失都直接接收 **logits**（激活之前的 $z$），自己在内部做 sigmoid / softmax。在模型里再加一层 `Sigmoid` 或 `Softmax` 是最常见的错误之一：激活被做了两次，梯度变小，训练变慢但不报错。

#### 回归：MSE、MAE 与 Huber

**由似然推 MSE**。假设 $y=\hat y+\varepsilon$，$\varepsilon\sim\mathcal N(0,\sigma^2)$，则

$$
-\log p(y\mid x)=-\log\Big(\frac{1}{\sqrt{2\pi}\sigma}e^{-(y-\hat y)^2/(2\sigma^2)}\Big)
=\frac{(y-\hat y)^2}{2\sigma^2}+\log(\sqrt{2\pi}\sigma).
$$

$\sigma$ 固定时，第二项是常数，第一项正比于平方误差。所以最小化 MSE 就是在高斯噪声假设下做最大似然。同理，拉普拉斯噪声 $p(\varepsilon)\propto e^{-\lvert\varepsilon\rvert/b}$ 给出 MAE。

**两者预测的是不同的量**。对一组观测 $y_1,\dots,y_n$ 只用一个常数 $c$ 去拟合：

- MSE：$\frac{\mathrm d}{\mathrm dc}\sum_i\frac12(c-y_i)^2=\sum_i(c-y_i)=0$，得 $c=\bar y$，**均值**；
- MAE：$\frac{\mathrm d}{\mathrm dc}\sum_i\lvert c-y_i\rvert=\#\{y_i<c\}-\#\{y_i>c\}=0$，比 $c$ 小和比 $c$ 大的点一样多，得**中位数**。

> [!EXAMPLE]+ 离群点对 MSE 与 MAE 的影响
>
> 房价数据 $\{1,2,3,4,100\}$（单位：百万），最后一个是录入错误。MSE 的最优常数是均值 $22$，被一个错误值拉得远离其他所有点；MAE 的最优常数是中位数 $3$，几乎不受影响。从梯度看：MSE 对这个点的梯度是 $\hat y-y$，误差 $97$ 就给出 $97$ 倍的推力；MAE 的梯度只有 $\pm1$，离群点和普通点的发言权一样大。

**Huber 损失**把两者拼在一起：误差小时用平方（$0$ 附近光滑，梯度随误差减小，便于精确收敛），误差大时用绝对值（梯度有界，不被离群点带跑）：

$$
L_\delta(r)=\begin{cases}\frac12 r^2, & \lvert r\rvert\le\delta,\\ \delta\big(\lvert r\rvert-\frac12\delta\big), & \lvert r\rvert>\delta,\end{cases}
\qquad
L_\delta'(r)=\begin{cases} r, & \lvert r\rvert\le\delta,\\ \delta\,\mathrm{sign}(r), & \lvert r\rvert>\delta,\end{cases}
$$

其中 $r=\hat y-y$。在 $\lvert r\rvert=\delta$ 处两段的值和导数都连续：$\frac12\delta^2=\delta(\delta-\frac12\delta)$，导数都是 $\delta$。它的梯度就是“裁剪到 $[-\delta,\delta]$ 的 MSE 梯度”。DQN 的 TD 误差和检测网络的边框回归都用它。`SmoothL1Loss(beta)` 等于 `HuberLoss(delta=beta)` 再除以 `beta`，二者在 $\delta=1$ 时相同。

| | MSE | MAE | Huber |
| :--- | :--- | :--- | :--- |
| 最优常数 | 均值 | 中位数 | 介于两者之间 |
| 梯度大小 | 正比于误差 | 恒为 $1$ | 小误差正比，大误差恒为 $\delta$ |
| 对离群点 | 敏感 | 稳健 | 稳健 |
| $0$ 处 | 光滑 | 不可导，梯度不变小，在最优点附近震荡 | 光滑 |

#### 分类：交叉熵家族

**二元交叉熵的梯度**。由链式法则，

$$
\frac{\partial L}{\partial a}=-\frac{y}{a}+\frac{1-y}{1-a}=\frac{a-y}{a(1-a)},\qquad
\frac{\partial L}{\partial z}=\frac{\partial L}{\partial a}\cdot\sigma'(z)=\frac{a-y}{a(1-a)}\cdot a(1-a)=a-y.
$$

sigmoid 的导数 $a(1-a)$ 恰好被消掉。这正是“为何不用 MSE 做分类”的另一面：MSE 配 sigmoid 时，这个 $a(1-a)$ 留在梯度里，饱和区梯度消失。

**数值稳定的写法**。直接算 $\log\sigma(z)$，在 $z=-800$ 时 $\sigma(z)$ 下溢为 $0$，得到 $\log 0=-\infty$。把损失用 $z$ 重写：利用 $\log\sigma(z)=-\log(1+e^{-z})$、$\log(1-\sigma(z))=-z-\log(1+e^{-z})$，

$$
L=-y\log\sigma(z)-(1-y)\log(1-\sigma(z))=\log(1+e^{z})-yz=\max(z,0)-yz+\log\big(1+e^{-\lvert z\rvert}\big).
$$

最后一步用了 $\log(1+e^{z})=\max(z,0)+\log(1+e^{-\lvert z\rvert})$：指数上永远是非正数，不会溢出。`BCEWithLogitsLoss` 内部就是这个式子。多分类同理：$\log y_k=z_k-\operatorname{logsumexp}(z)$，`log_softmax` 先减去最大值再求和。

**多标签与多分类的区别**。一张图里同时有猫和狗，标签是 $(1,1,0)$，各类之间不互斥，每一维单独做 sigmoid + BCE，概率之和不必为 $1$。只有“恰好一个类别正确”时才用 softmax：它强制概率和为 $1$，一个类升高必然压低其他类。

**类别不平衡**。正例只占 1% 时，模型全猜负例就有 99% 准确率，交叉熵也已经不小。两种常见对策：

- **加权交叉熵**：$L=-\sum_k w_k t_k\log y_k$，少数类权重大，常取 $w_k\propto 1/\text{频率}$ 或其平方根。PyTorch 中是 `CrossEntropyLoss(weight=...)`，二分类是 `BCEWithLogitsLoss(pos_weight=...)`。
- **Focal loss**（Lin et al., 2017，RetinaNet）：记正确类的概率为 $p_t$，

$$
\mathrm{FL}(p_t)=-(1-p_t)^{\gamma}\log p_t,\qquad \gamma=2.
$$

$(1-p_t)^{\gamma}$ 把已经分对的简单样本压下去，让梯度集中在难样本上：

| $p_t$ | 交叉熵 $-\log p_t$ | Focal（$\gamma=2$） | 压低倍数 |
| :--- | :--- | :--- | :--- |
| $0.9$（简单） | $0.105$ | $0.0011$ | $100$ |
| $0.5$ | $0.693$ | $0.173$ | $4$ |
| $0.1$（困难） | $2.303$ | $1.865$ | $1.23$ |

检测中每张图有上万个背景候选框，几乎都是简单负例。不压低它们，简单负例的损失总和会淹没少数难样本。

**Hinge 损失**（SVM）。标签取 $y\in\{-1,+1\}$，分数为 $z$，定义间隔（margin）$yz$：

$$
L_{\mathrm{hinge}}=\max(0,1-yz),\qquad
\frac{\partial L}{\partial z}=\begin{cases}-y, & yz<1,\\ 0, & yz\ge 1.\end{cases}
$$

同样用 $\pm1$ 标签，逻辑回归的损失可以写成 $\log(1+e^{-yz})$。两者都随间隔增大而下降，区别在于 hinge 在间隔超过 $1$ 后损失和梯度都恰好为 $0$，“分对且有余量”的样本不再贡献任何梯度；逻辑损失永远大于 $0$，还会继续把分数推大。多分类版本（CS231n）是 $\sum_{j\neq y}\max(0,z_j-z_y+1)$：要求正确类的分数比每个错误类至少高 $1$。神经网络中交叉熵更常用，因为它输出校准过的概率。

**交叉熵与 KL 散度**。对标签分布 $p$ 和预测分布 $q$，

$$
H(p,q)=-\sum_k p_k\log q_k=\underbrace{-\sum_k p_k\log p_k}_{H(p)}+\underbrace{\sum_k p_k\log\frac{p_k}{q_k}}_{\mathrm{KL}(p\Vert q)}.
$$

$H(p)$ 与模型无关，所以对 $q$ 最小化交叉熵等价于最小化 KL 散度。标签是 one-hot 时 $H(p)=0$，两者完全相等；标签是软分布时（标签平滑、知识蒸馏），二者相差一个常数，梯度相同。蒸馏用 `F.kl_div(F.log_softmax(z_s / T), F.softmax(z_t / T), reduction="batchmean") * T**2`：学生向教师在温度 $T$ 下的软分布靠拢，乘 $T^2$ 是为了让梯度量级不随 $T$ 改变。

#### 其他任务的损失

后面各章还会遇到专门的损失，它们都建立在上面的基本形式上：

- **Triplet loss**（人脸）：$\max(\lVert f(A)-f(P)\rVert^2-\lVert f(A)-f(N)\rVert^2+\alpha,0)$，是作用在距离上的 hinge；
- **对比损失 NT-Xent**（自监督）：在 batch 内找出正样本对的交叉熵；
- **Dice loss**（分割）：直接优化预测与真值的重叠率，前景很小时比逐像素交叉熵稳；
- **内容与风格损失**（风格迁移）：特征图和 Gram 矩阵上的 MSE；
- **YOLO 损失**：坐标的平方误差、置信度与类别的交叉熵加权求和；
- **VAE 损失**：重构的负对数似然加 KL 项。

#### 实现与核对

每个损失返回 `(loss, dz)`，`dz` 是对 logits 或预测值的梯度（已除以 batch 大小），再用 PyTorch autograd 核对：

```python
def mse(yhat, y):
    r = yhat - y
    return 0.5 * np.mean(r ** 2), r / r.size

def mae(yhat, y):
    r = yhat - y
    return np.mean(np.abs(r)), np.sign(r) / r.size

def huber(yhat, y, delta=1.0):
    r = yhat - y
    quad = np.abs(r) <= delta
    loss = np.where(quad, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))
    return loss.mean(), np.clip(r, -delta, delta) / r.size

def bce_logits(z, y):
    loss = np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))
    return loss.mean(), (sigmoid(z) - y) / z.size

def softmax_ce(z, labels):
    """z: (m, K) logits；labels: (m,) 类别下标。"""
    m = z.shape[0]
    log_p = z - z.max(1, keepdims=True)
    log_p = log_p - np.log(np.exp(log_p).sum(1, keepdims=True))     # log_softmax
    dz = np.exp(log_p)
    dz[np.arange(m), labels] -= 1.0                                   # y - t
    return -log_p[np.arange(m), labels].mean(), dz / m

def focal_binary(z, y, gamma=2.0):
    p = sigmoid(z)
    p_t = np.where(y == 1, p, 1 - p)
    return np.mean(-(1 - p_t) ** gamma * np.log(np.clip(p_t, 1e-12, 1)))

def hinge(z, y_pm):
    """y_pm ∈ {-1, +1}。"""
    margin = y_pm * z
    return np.mean(np.maximum(0, 1 - margin)), np.where(margin < 1, -y_pm, 0.0) / z.size
```

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

rng = np.random.default_rng(0)
z = rng.standard_normal(8) * 3; y = (rng.random(8) > 0.5).astype(float)
zt = torch.tensor(z, requires_grad=True); yt = torch.tensor(y)
checks = {
    "mse":   (mse(z, y),            lambda: 0.5 * F.mse_loss(zt, yt)),
    "mae":   (mae(z, y),            lambda: F.l1_loss(zt, yt)),
    "huber": (huber(z, y),          lambda: F.huber_loss(zt, yt, delta=1.0)),
    "bce":   (bce_logits(z, y),     lambda: F.binary_cross_entropy_with_logits(zt, yt)),
    "hinge": (hinge(z, 2 * y - 1),  lambda: F.relu(1 - (2 * yt - 1) * zt).mean()),
}
for name, ((loss, dz), torch_fn) in checks.items():
    zt.grad = None
    lt = torch_fn(); lt.backward()
    print(name, abs(loss - lt.item()), np.abs(dz - zt.grad.numpy()).max())   # 都约 1e-16

Z = rng.standard_normal((4, 5)); lab = np.array([0, 3, 1, 4])
Zt = torch.tensor(Z, requires_grad=True)
F.cross_entropy(Zt, torch.tensor(lab)).backward()
loss, dZ = softmax_ce(Z, lab)
print("ce", np.abs(dZ - Zt.grad.numpy()).max())                              # 约 1e-17

print(bce_logits(np.array([-800.0]), np.array([1.0]))[0])                   # 800.0，而不是 inf
```

`F.mse_loss` 没有 $\frac12$，所以上面乘了 $0.5$ 才能对上；自己写的损失与框架对拍时，常数因子和“是否对 batch 取平均”（`reduction="mean"` 或 `"sum"`）是最常见的不一致来源。

### 推理示例：MNIST

把前面的部件拼起来，就能完成一个真实任务的推理（假设参数已训练好）。

- **数据**：MNIST 手写数字，训练 6 万张、测试 1 万张，每张 $28\times 28$ 灰度图，像素取值 0–255。
- **预处理**：展平为 784 维向量；像素除以 255 缩放到 $[0,1]$（称为**正规化**，normalization）。更常见的预处理是减均值、除以标准差，使数据以 0 为中心；还有使分布形状均匀化的白化（whitening）。
- **网络**：784 → 50 → 100 → 10，隐藏层 sigmoid，输出层 softmax。权重形状（行样本约定）为 $(784,50),(50,100),(100,10)$。
- **结果**：取输出中概率最大的类别作为预测（`np.argmax`），测试集准确率 93.52%。

**批处理**：一次只输入一张图时，形状链是 $(784,)\to(10,)$。若一次输入 100 张，输入变为 $(100,784)$，输出为 $(100,10)$，第 $i$ 行就是第 $i$ 张图的结果。这样打包的输入称为一个**批**（batch）。批处理更快，因为数值库对大矩阵运算做了高度优化，而且减少了数据搬运的开销，让更多时间花在计算上。

```python
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)        # (100, 10)
    p = np.argmax(y_batch, axis=1)              # 每行取最大值的下标
    accuracy_cnt += np.sum(p == t[i:i + batch_size])
print("Accuracy:", accuracy_cnt / len(x))
```

<br>

## 反向传播（Backpropagation）

### 为什么需要反向传播

训练需要 $J$ 对每个参数的梯度。用数值微分，每个参数都要前向两次；一个有一百万参数的网络，一步更新就要两百万次前向，完全不可行。**反向传播**（误差反向传播法，error backpropagation）利用链式法则，只需一次前向 + 一次反向，就能得到所有参数的梯度，代价与前向同阶。

理解反向传播最直观的工具是**计算图**（computational graph）：把计算拆成一个个节点，每个节点只做一件简单的事。

### 计算图

> [!EXAMPLE]+ 例9 买苹果（鱼书 5.1 节）
>
> 买 2 个单价 100 日元的苹果，消费税 10%，应付多少？
>
> 前向（从左到右）：$100\xrightarrow{\times 2}200\xrightarrow{\times 1.1}220$。
>
> 现在问：苹果单价涨一点，总价涨多少？即求 $\partial(\text{总价})/\partial(\text{单价})$。反向（从右到左）传递导数：
>
> - 起点是 $\partial(\text{总价})/\partial(\text{总价})=1$；
> - 经过“$\times 1.1$”节点，乘以 1.1，得 1.1；
> - 经过“$\times 2$”节点，乘以 2，得 **2.2**。
>
> 所以单价每涨 1 日元，总价涨 2.2 日元。顺带还能得到：对苹果个数的导数是 **110**，对消费税的导数是 **200**。

计算图的两个好处：

1. **局部计算**：每个节点只需关心自己的输入输出，不管全局多复杂。例如在超市买了很多东西，“$\times$ 税率”节点只需要知道税前总额。
2. **中间结果可复用**：前向时存下每个节点的值，反向时一次就能求出所有变量的导数。

### 链式法则

反向传播的数学基础是**链式法则**：复合函数的导数等于各环节导数的乘积。例如 $z=t^{2}$，$t=x+y$：

$$
\frac{\partial z}{\partial x}=\frac{\partial z}{\partial t}\cdot\frac{\partial t}{\partial x}=2t\cdot 1=2(x+y).
$$

在计算图上，反向传播就是：**把从右边传来的上游梯度，乘以本节点的局部导数，再传给左边**。

### 基本节点

- **加法节点** $z=x+y$：$\partial z/\partial x=\partial z/\partial y=1$，所以上游梯度**原样**传给两个输入。
- **乘法节点** $z=xy$：$\partial z/\partial x=y$，$\partial z/\partial y=x$，所以上游梯度乘以**另一个输入**再传回（输入“交换”）。因此乘法节点反向时需要前向的输入值，必须缓存。

> [!EXAMPLE]+ 例10 买苹果和橘子
>
> 2 个苹果（单价 100）、3 个橘子（单价 150），税率 1.1。
>
> 前向：苹果 $100\times 2=200$，橘子 $150\times 3=450$，加法得 $650$，乘税得 $715$。
>
> 反向：
>
> - 乘税节点：对税前总额的导数是 $1.1$，对税率的导数是 **650**；
> - 加法节点：把 $1.1$ 原样传给苹果总价和橘子总价；
> - 苹果的乘法节点：对单价的导数 $1.1\times 2=$ **2.2**，对个数的导数 $1.1\times 100=$ **110**；
> - 橘子的乘法节点：对单价的导数 $1.1\times 3=$ **3.3**，对个数的导数 $1.1\times 150=$ **165**。

### 激活函数层

把神经网络的每种运算实现成一个“层”，每层提供 `forward` 和 `backward` 两个方法，网络就像积木一样拼起来。

- **ReLU 层**：前向时 $x>0$ 输出 $x$，否则输出 0；反向时 $x>0$ 的位置梯度原样通过，$x\le 0$ 的位置梯度置 0。前向时记下布尔掩码 `mask = (x <= 0)`，反向时 `dout[mask] = 0`。ReLU 就像一个开关：前向时导通的位置，反向时也导通。
- **Sigmoid 层**：$y=1/(1+e^{-x})$ 可以拆成“$\times(-1)$ → $\exp$ → $+1$ → 取倒数”四个节点。逐个反向：
  - 取倒数 $y=1/u$：局部导数 $-1/u^{2}=-y^{2}$；
  - $+1$：原样通过；
  - $\exp$：局部导数 $e^{-x}$；
  - $\times(-1)$：乘以 $-1$。
  
  合起来，上游梯度 $\frac{\partial L}{\partial y}$ 变为

$$
\frac{\partial L}{\partial y}\,y^{2}e^{-x}=\frac{\partial L}{\partial y}\cdot\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}=\frac{\partial L}{\partial y}\,y(1-y).
$$

  反向只需前向的输出 $y$，与 $\sigma'(z)=\sigma(z)(1-\sigma(z))$ 一致。

### Affine 层

神经网络前向中的矩阵乘加偏置，在几何上称为仿射变换（affine transformation），对应的层称为 Affine 层。按上一节的恒等式，列样本约定 $Z=WA+b$ 下：

$$
\frac{\partial J}{\partial W}=\mathrm{d}Z\,A^{\top},\qquad
\frac{\partial J}{\partial A}=W^{\top}\mathrm{d}Z,\qquad
\frac{\partial J}{\partial b}=\sum_{\text{样本}}\mathrm{d}Z.
$$

偏置在前向时被广播到每个样本上，所以反向时要把每个样本的梯度**加起来**（沿样本维求和）。鱼书的行样本约定 $Y=XW+b$ 下对应为 $\partial L/\partial X=\frac{\partial L}{\partial Y}W^{\top}$、$\partial L/\partial W=X^{\top}\frac{\partial L}{\partial Y}$、$\partial L/\partial b$ 沿第 0 轴求和。

### Softmax-with-Loss 层

把 softmax 和交叉熵合成一个层，反向时的梯度极其简洁：

$$
\frac{\partial L}{\partial z_k}=y_k-t_k.
$$

即“输出减标签”，与 sigmoid + 二元交叉熵的 $a-y$ 完全一致。批处理时再除以批大小。

**推导**分两步：先求 softmax 的 Jacobian，再与交叉熵的梯度相乘。

第一步，softmax $y_k=e^{z_k}/S$，$S=\sum_j e^{z_j}$。对 $z_i$ 求导，分 $i=k$ 和 $i\neq k$ 两种情况（商的求导法则，注意 $\partial S/\partial z_i=e^{z_i}$）：

$$
\frac{\partial y_k}{\partial z_i}=
\begin{cases}
\dfrac{e^{z_k}S-e^{z_k}e^{z_k}}{S^{2}}=y_k(1-y_k), & i=k,\\[2ex]
\dfrac{0\cdot S-e^{z_k}e^{z_i}}{S^{2}}=-y_k y_i, & i\neq k.
\end{cases}
$$

两种情况合写为 $\frac{\partial y_k}{\partial z_i}=y_k(\delta_{ki}-y_i)$，矩阵形式是 $\frac{\partial y}{\partial z}=\mathrm{diag}(y)-yy^{\top}$。对角线上的 $y_k(1-y_k)$ 与 sigmoid 的导数形式相同；非对角项为负，因为一个类的分数变大，会把其他类的概率挤小。

第二步，交叉熵 $L=-\sum_k t_k\log y_k$，所以 $\frac{\partial L}{\partial y_k}=-\frac{t_k}{y_k}$。链式法则对所有 $k$ 求和：

$$
\frac{\partial L}{\partial z_i}
=\sum_k\frac{\partial L}{\partial y_k}\frac{\partial y_k}{\partial z_i}
=\sum_k\Big(-\frac{t_k}{y_k}\Big)y_k(\delta_{ki}-y_i)
=-\sum_k t_k\delta_{ki}+y_i\sum_k t_k
=-t_i+y_i.
$$

最后一步用了 $\sum_k t_k=1$（one-hot 或任何概率分布标签都成立）。分母里的 $y_k$ 被 Jacobian 里的 $y_k$ 消掉了，所以合并成一层后既简单又数值稳定：不需要计算 $1/y_k$，$y_k$ 很小时也不会除以接近 $0$ 的数。这就是框架把 softmax 和交叉熵合成 `CrossEntropyLoss` 一个函数的原因。

更直接的推法是先化简损失：$\log y_k=z_k-\log S$，于是

$$
L=-\sum_k t_k z_k+\log\sum_j e^{z_j},\qquad
\frac{\partial L}{\partial z_i}=-t_i+\frac{e^{z_i}}{\sum_j e^{z_j}}=y_i-t_i.
$$

$\log\sum_j e^{z_j}$ 称为 log-sum-exp，它的梯度正好是 softmax。

```python
def softmax_jacobian(y):
    """y: (K,) softmax 输出。返回 (K, K) 的 dy/dz。"""
    return np.diag(y) - np.outer(y, y)

z = np.array([0.3, 2.9, 4.0]); t = np.array([0.0, 1.0, 0.0])
y = np.exp(z - z.max()); y /= y.sum()
dL_dy = -t / y
print(softmax_jacobian(y).T @ dL_dy, y - t)    # 两种算法结果相同：[ 0.018 -0.755  0.737]
```

> [!EXAMPLE]+ 例11 误差的大小反映错误的程度
>
> 标签 $t=(0,1,0)$：
>
> - softmax 输出 $y=(0.3,0.2,0.5)$，正确类概率只有 0.2，识别错误。反向传回 $(0.3,\,-0.8,\,0.5)$，误差很大，前面的层会做较大调整。
> - softmax 输出 $y=(0.01,0.99,0)$，识别得相当准确。反向传回 $(0.01,\,-0.01,\,0)$，误差很小，前面的层只做微小调整。
>
> 回归问题中“恒等函数 + 均方误差”也有同样的结果 $y-t$。这不是巧合：交叉熵与 softmax、均方误差与恒等函数，都是为了得到这个简洁形式而成对设计的。

### 两层网络的反向公式

把以上节点串起来，就得到浅层网络的完整反向传播。二分类取 $g^{[2]}=\sigma$，则 $\mathrm{d}Z^{[2]}=A^{[2]}-Y$，其余按 Affine 层与激活层依次往回传：

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

- $\odot$ 是逐元素乘：激活函数逐元素作用，所以它的反向也是逐元素乘局部导数。
- $g=\tanh$ 时 $g'(z)=1-\tanh^{2}(z)=1-a^{2}$，用前向已算的 $A^{[1]}$ 即可，不必另存 $Z^{[1]}$。
- ReLU 必须知道 $Z^{[1]}$ 的符号，需要缓存 $Z^{[1]}$（或掩码）。

> [!INFO]+ 两层网络的链式法则
>
> $J$ 对 $W^{[2]}$ 的依赖路径是 $W^{[2]}\rightarrow Z^{[2]}\rightarrow A^{[2]}\rightarrow J$。$\mathrm{d}W^{[2]}=\mathrm{d}Z^{[2]}A^{[1]\top}/m$，因为 $Z^{[2]}$ 的第 $k$ 行来自 $W^{[2]}$ 第 $k$ 行与 $A^{[1]}$ 的内积。对 $A^{[1]}$ 求导时，$W^{[2]\top}$ 把输出层的误差“投影”回隐藏层的坐标；再乘以隐藏层的局部激活导数，才得到 $\mathrm{d}Z^{[1]}$。漏乘 $g'$ 是实现中最常见的错误之一。

### 实现

激活函数层与两层网络的前向、反向：

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
  - 不能全零：若 $W^{[1]}=W^{[2]}=0$，所有隐藏单元计算同一个函数，收到同样的梯度，更新后仍然相同，永远无法分化出不同的特征（对称性问题）；
  - 常用小的高斯随机数，或按后文 Xavier/He 缩放；
  - $b$ 可以零初始化，因为 $W$ 的随机性已经打破了对称。
- **训练循环**
  - 每步四件事：前向、算损失、反向、更新；
  - 更新必须用当前步的梯度，不能把所有层的梯度算完后再用过期的激活；
  - 学习率对浅层网络同样敏感：平面数据上 $\alpha=1.2$ 可能合适，而 $\alpha=0.01$ 会显得“网络学不会”；
  - 损失曲线还没下降时，先不要换架构，先查学习率和实现。

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

- **决策边界**：$A^{[2]}=0.5$ 即 $Z^{[2]}=0$，在输入平面上一般是曲线。把网格上的点送入前向，按输出着色即可画出。
- 若边界仍接近直线，可能是隐藏层过窄、激活近似线性（输入落在 tanh 的线性区），或训练未收敛。

同一个两层网络的 PyTorch 写法。注意两点：PyTorch 的 `nn.Linear` 按**行样本**排布（输入 `(m, n_x)`，权重 `(n_out, n_in)`，计算 $XW^{\top}+b$）；输出层不写 sigmoid，而是把 logits 交给 `BCEWithLogitsLoss`，它内部用数值稳定的方式合并 sigmoid 与交叉熵，对应前面的 $\mathrm{d}Z^{[2]}=A^{[2]}-Y$。

```python
import torch
import torch.nn as nn

class TwoLayer(nn.Module):
    def __init__(self, n_x, n_h):
        super().__init__()
        self.fc1 = nn.Linear(n_x, n_h)
        self.fc2 = nn.Linear(n_h, 1)

    def forward(self, x):                 # x: (m, n_x)
        return self.fc2(torch.relu(self.fc1(x)))   # logits: (m, 1)

def train_two_layer(X, Y, n_h=4, lr=1.2, steps=10000):
    """X: (n_x, m), Y: (1, m) 的 NumPy 数组，沿用 CS230 的列样本约定。"""
    x = torch.tensor(X.T, dtype=torch.float32)        # 转成行样本
    y = torch.tensor(Y.T, dtype=torch.float32)
    model = TwoLayer(x.shape[1], n_h)
    loss_fn = nn.BCEWithLogitsLoss()                  # 默认对 m 取平均，对应 1/m
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()                               # 梯度默认累加，每步先清零
        loss = loss_fn(model(x), y)                   # 前向 + 损失
        loss.backward()                               # 反向：填好每个参数的 .grad
        opt.step()                                    # W -= lr * dW
    return model
```

- `model.fc1.weight.grad` 与手写的 `dW1` 同形 `(n_h, n_x)`，可以直接对拍：同一组初值下两者应一致到 $10^{-6}$ 量级（float32）。
- 漏写 `zero_grad()` 时梯度会跨步累加，等价于学习率越来越大，损失往往先降后炸。

> [!EXAMPLE]+ 例12 鱼书的两层网络训练（MNIST）
>
> 784 → 50 → 10 的两层网络，mini-batch 大小 100，学习率 0.1，迭代 10000 次。训练集 60000 张、batch 100，所以遍历一遍训练集需要 600 次迭代，称为一个 **epoch**。每个 epoch 记录一次训练集与测试集准确率，两条曲线几乎重合且一起上升，说明没有过拟合。用数值微分做同样的训练会慢几个数量级，这正是反向传播的价值。

<br>

> [!NOTE]+ 衔接
>
> 两层网络已经具备完整的“前向缓存 + 反向外积”结构，每一层都是一个有 `forward` / `backward` 的模块。下一部分只是把这两步放进 $l=1\to L$ 与 $l=L\to 1$ 的循环，接口不变；随后讨论网络变深之后才出现的问题：初始化、过拟合、优化速度。

<br>


# 深层网络与训练

浅层网络的前向与反向已经写清楚，把它推广到 $L$ 层只是加一个循环。真正的困难在于：**网络一深，训练本身就变难了**。这一部分按问题出现的顺序组织：

1. **深层网络**：$L$ 层的通用接口，以及用梯度检查验证反向传播写对了；
2. **初始化**：权重的初始尺度决定信号能否穿过很多层；
3. **优化**：mini-batch、Momentum、AdaGrad / RMSprop、Adam、学习率衰减，决定损失下降得快不快、稳不稳；
4. **批归一化**：把每层激活的分布强制拉回合适范围，降低对初始化和学习率的敏感度；
5. **正则化**：训练误差降下来之后，缩小训练集与开发集之间的差距（过拟合）；
6. **超参数**：上述方法引入的 $\alpha,\beta,\lambda$ 等如何在验证集上搜索。

前四节回答“训练误差降不降得下去”，第五节回答“能不能泛化”，第六节回答“这些旋钮怎么拧”。

<br>

## 深层网络（Deep Neural Network）

### 为什么要深

深层网络把浅层的前向、反向两步写成对 $l=1\to L$ 的循环。层宽记为 $[n^{[0]},\ldots,n^{[L]}]$，隐藏层的激活函数通常相同，输出层由任务决定。后面的卷积网络和 RNN 只是把 $W^{[l]}A^{[l-1]}$ 换成带权重共享的线性算子，反向接口不变；Dropout、BatchNorm、残差也都挂在这套层接口上。

为什么要加深而不是加宽？

- **特征层次**（经验动机）：浅层学到边缘，中层学到纹理和部件，深层学到物体。像素 → 边缘 → 部件 → 物体，每一层都在上一层的基础上组合出更抽象的特征。
- **参数效率**：同样的参数预算下，深度组合比单层极宽更省，中间表示可以被后续层复用。通用逼近定理说单隐藏层足够，但所需宽度可能是指数级的。
- **这不是定理**：过深而没有残差连接、没有合适的初始化时，训练误差反而会上升（见卷积部分的 ResNet）。

作业中的 $L$ 层模型通常是 2–5 层全连接，目标是把循环接口写对，不是追求 ImageNet 精度。

### 前向

对 $l=1,\ldots,L$：

$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]},\qquad
A^{[l]}=g^{[l]}(Z^{[l]}).
$$

输出层按任务选择：

- 二分类：$g^{[L]}=\sigma$，$J$ 为二元交叉熵。
- $K$ 类：$g^{[L]}=\mathrm{softmax}$，减去最大值保证数值稳定：

$$
a_k=\frac{e^{z_k-\max_j z_j}}{\sum_{j=1}^{K}e^{z_j-\max_j z_j}},\qquad
\mathcal{L}=-\sum_{k=1}^{K}y_k\log a_k.
$$

$y$ 为 one-hot 时 $\mathcal{L}=-\log a_{k^{\ast}}$，$k^{\ast}$ 是正确类别。

> [!INFO]+ softmax 交叉熵的 $dz$
>
> 令 $a_k=\mathrm{softmax}(z)_k$，$\mathcal{L}=-\sum_k y_k\log a_k$。softmax 的每个输出依赖所有输入，所以先求 Jacobian：
>
> $$
> \frac{\partial a_k}{\partial z_j}=a_k(\delta_{kj}-a_j),
> $$
>
> 其中 $\delta_{kj}$ 在 $k=j$ 时为 1，否则为 0。再由链式法则，并利用 $\sum_k y_k=1$：
>
> $$
> \frac{\partial\mathcal{L}}{\partial z_j}=\sum_k \frac{\partial\mathcal{L}}{\partial a_k}\frac{\partial a_k}{\partial z_j}
> =\sum_k\left(-\frac{y_k}{a_k}\right)a_k(\delta_{kj}-a_j)=-y_j+a_j\sum_k y_k=a_j-y_j.
> $$
>
> 与 sigmoid + BCE 相同：配对使用时输出层误差就是 $a-y$，不要再乘一遍 softmax 的 Jacobian。若损失改成 MSE，该简化不成立。

<br>

参数量约为 $\sum_l (n^{[l]}n^{[l-1]}+n^{[l]})$。图像若先展平再接全连接，参数量随图像长宽增长，而且模型默认任意两个像素同等相关，丢掉了“相邻像素关系更密切”的空间结构。让相邻位置共享权重，正是后面卷积网络的动机。

作业中常见的结构是 `[12288, 20, 7, 5, 1]`（输入是 $64\times 64\times 3$ 的展平图像）：

| 层 $l$ | $W^{[l]}$ | $b^{[l]}$ | $A^{[l]}$ |
| :--- | :--- | :--- | :--- |
| 1 | $(20, 12288)$ | $(20, 1)$ | $(20, m)$ |
| 2 | $(7, 20)$ | $(7, 1)$ | $(7, m)$ |
| 3 | $(5, 7)$ | $(5, 1)$ | $(5, m)$ |
| 4 | $(1, 5)$ | $(1, 1)$ | $(1, m)$ |

- 第一层独占几乎全部参数（$20\times 12288=245760$ 个权重）。
- 前向时每层存一个缓存 `caches[l-1] = (lin_cache, act_cache)`，其中 `lin_cache=(A^{[l-1]}, W^{[l]}, b^{[l]})`。丢掉 $A^{[l-1]}$ 就无法计算 $\mathrm{d}W^{[l]}$。
- 前向必须按 $l=1\to L$，因为 $A^{[l]}$ 依赖 $A^{[l-1]}$；反向必须按 $l=L\to 1$，因为 $\mathrm{d}A^{[l-1]}$ 依赖 $\mathrm{d}Z^{[l]}$。把反向写成与前向相同的升序循环是逻辑错误，梯度检查会立刻失败。

### 反向

反向从输出层开始：

- 一般情况令 $\mathrm{d}A^{[L]}=\partial J/\partial A^{[L]}$，再乘 $g^{[L]\prime}$ 得 $\mathrm{d}Z^{[L]}$。
- sigmoid + BCE：直接设 $\mathrm{d}Z^{[L]}=A^{[L]}-Y$。
- softmax + 分类交叉熵：若 $Y\in\{0,1\}^{K\times m}$ 是 one-hot 标签，同样有 $\mathrm{d}Z^{[L]}=A^{[L]}-Y$。此时 $A^{[L]}$ 和 $Y$ 都是 $(K,m)$，每列和为 $1$。

然后对 $l=L,\ldots,1$ 重复同一组公式（就是浅层网络的反向公式加上层号）：

$$
\begin{aligned}
\mathrm{d}W^{[l]} &= \frac{1}{m}\mathrm{d}Z^{[l]}A^{[l-1]\top},\\
\mathrm{d}b^{[l]} &= \frac{1}{m}\sum_{i=1}^{m}\mathrm{d}Z^{[l](:,i)},\\
\mathrm{d}A^{[l-1]} &= W^{[l]\top}\mathrm{d}Z^{[l]},\\
\mathrm{d}Z^{[l-1]} &= \mathrm{d}A^{[l-1]}\odot g^{[l-1]\prime}(Z^{[l-1]}).
\end{aligned}
$$

- 若损失已与输出激活配对，最后一层不必再乘 $g^{[L]\prime}$。
- 每层必须缓存 $A^{[l-1]}$，以及激活函数求导所需的 $Z^{[l]}$ 或掩码。

> [!EXAMPLE]+ 例1 多分类输出（multiclass output）
>
> 三分类时 $W^{[L]}$ 的 shape 是 $(3,n^{[L-1]})$，而不是 $(1,n^{[L-1]})$。前向和损失为
>
> $$
> A^{[L]}=\operatorname{softmax}(Z^{[L]}),\qquad
> J=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{3}
> Y_{ki}\log A_{ki}^{[L]}.
> $$
>
> 反向仍从 $\mathrm{d}Z^{[L]}=A^{[L]}-Y$ 开始。这个简化只适用于 softmax 与分类交叉熵的配对。若 $Y$ 只是类别号 $(m,)$，应先转成 one-hot，或使用框架提供的 sparse categorical cross-entropy，不能直接依赖 NumPy 广播相减。

> [!INFO]+ 反向传播作为动态规划
>
> 从 $A^{[L]}$ 到 $W^{[1]}$ 的计算图是一条长链。朴素的数值差分需要对每个参数各做两次前向。反向传播从输出往回走，每一层复用上一层算好的 $\mathrm{d}Z$，每条边只访问一次，总代价与一次前向同阶，这本质上是动态规划。缓存是空间换时间：不存 $A^{[l]}$ 就无法用外积形成 $\mathrm{d}W^{[l+1]}$。梯度检查点（checkpointing）可以丢掉部分激活、在反向时重算，用计算换显存。

<br>

### 实现

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

- 每步更新的代价为 $O(\sum_l n^{[l]}n^{[l-1]}m)$，与前向同阶。
- 不要在反向过程中立刻修改 $W^{[l]}$：后续层的 $\mathrm{d}A^{[l-1]}=W^{[l]\top}\mathrm{d}Z^{[l]}$ 需要用**更新前**的权重。正确顺序是全部 `dW` 算完，再统一更新。

### 梯度检查

反向传播的公式很容易写错一个转置或漏乘一个导数，而且错了之后网络往往仍能“训练”，只是效果差，很难察觉。**梯度检查**（gradient checking）用数值微分作为参照：数值微分慢但几乎不会写错，反向传播快但容易写错，两者结果一致就说明反向传播正确。

对每个参数分量用中心差分：

$$
\frac{\partial J}{\partial\theta_i}\approx\frac{J(\theta_i+\varepsilon)-J(\theta_i-\varepsilon)}{2\varepsilon}.
$$

把所有参数拉平成一个长向量，用相对误差衡量两者差距（相对误差不受梯度整体大小影响）：

$$
\frac{\|\mathrm{d}\theta_{\mathrm{num}}-\mathrm{d}\theta_{\mathrm{bp}}\|_2}{\|\mathrm{d}\theta_{\mathrm{num}}\|_2+\|\mathrm{d}\theta_{\mathrm{bp}}\|_2}.
$$

- 判据（$\varepsilon=10^{-7}$、双精度）：
  - 相对误差 $<10^{-7}$：实现通常正确；
  - $10^{-5}$ 量级：需要排查；
  - $10^{-2}$：公式或 shape 有错误。
- 数值梯度要对每个参数前向两次，代价极高，只在小模型上检查，检查完就关掉。
- 检查时关闭 Dropout（随机掩码会让每次前向的 $J$ 不同），BatchNorm 使用固定统计量。
- 若加了正则项，$J$ 中必须包含正则项，否则两边的梯度对不上。

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

> [!EXAMPLE]+ 例2 鱼书两层网络的梯度确认
>
> 在 MNIST 两层网络上，比较数值微分与反向传播得到的梯度，各参数的平均绝对误差为：$W_1$：$2.2\times 10^{-13}$，$b_1$：$9.7\times 10^{-13}$，$W_2$：$8.4\times 10^{-13}$，$b_2$：$1.2\times 10^{-10}$。误差极小且不为零，后者源于浮点运算精度有限，说明反向传播实现正确。

$\varepsilon$ 的取法是一个折中：太大，差分本身对导数的近似变差（截断误差约 $O(\varepsilon^{2})$）；太小，$J(\theta+\varepsilon)$ 与 $J(\theta-\varepsilon)$ 几乎相等，相减时有效数字被抵消（舍入误差约 $O(\epsilon_{\text{mach}}/\varepsilon)$）。双精度下取 $10^{-7}$ 左右，单精度下应放大到 $10^{-3}$ 左右。

### 先过拟合一个小 batch

梯度检查只说明“反向传播与前向一致”，不说明整个训练管道没有问题。CS230 速查表（*Deep Learning Tips and Tricks*）推荐的另一项检查是：**取一个很小的 batch，看模型能不能把它背下来**。

- **做法**：固定一个 mini-batch（例如 $16$–$64$ 个样本），关掉 Dropout、数据增强与权重衰减，在这一批上反复迭代几百步。
- **预期**：损失接近 $0$，训练准确率接近 $100\%$。参数量远大于样本数，网络完全有能力记住它们。
- **背不下来时**，先查这些，而不是先调网络结构：
  - 标签与输入是否错位（打乱 $X$ 时没有同步打乱 $Y$）；
  - 损失与输出层是否匹配（sigmoid 输出却用了 softmax 交叉熵，或 logits 被重复过了一次 softmax）；
  - 学习率是否过大或过小；梯度是否真的流到了每一层（参数是否注册进了优化器、是否误加了 `stop_gradient`）；
  - 预处理是否把输入变成了常数（例如整数除法把像素全变成 $0$）。
- 速查表的说法是：若网络连一个小 batch 都过拟合不了，那它要么太复杂而训不动，要么太简单，正常大小的训练集就更不用谈了。

这项检查只花几分钟，应当在每次搭新管道、换损失或换数据格式后先跑一次，再开始正式训练。

```python
import torch

def overfit_one_batch(model, loss_fn, loader, steps=300, lr=1e-3):
    """取 loader 的第一个 batch 反复训练，返回损失轨迹。"""
    x, y = next(iter(loader))
    model.train()                       # 若模型含 Dropout，先在构造时把 p 设为 0
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # 不加 weight_decay
    history = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history          # 预期 history[-1] 接近 0
```

PyTorch 也自带梯度检查：`torch.autograd.gradcheck` 用中心差分核对自定义算子的反向，要求输入为双精度。

```python
from torch.autograd import gradcheck

def my_layer(x, W, b):                     # 想检查的任意可微函数
    return torch.tanh(x @ W.T + b)

x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
W = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
b = torch.randn(4, dtype=torch.float64, requires_grad=True)
print(gradcheck(my_layer, (x, W, b), eps=1e-6, atol=1e-4))   # True
```

自己写了 `torch.autograd.Function`（手写 `backward`）时，这一步与前面 NumPy 的 `gradient_check` 作用相同。

<br>

> [!NOTE]+ 衔接
>
> $L$ 层接口已写完，梯度也能验证。但同一份代码，权重初始尺度不同，结果可能是正常训练，也可能完全学不动。下一节讨论初始化：全零导致对称，过小使信号逐层收缩，过大使激活饱和或爆炸。

<br>

## 初始化（Initialization）

### 问题

初始化规定 $W^{[l]}$ 从什么分布采样。它要解决两个问题：

1. **打破对称**：同一层的神经元必须从不同的起点出发，否则它们永远一样。
2. **保持尺度**：信号经过很多层后，既不能缩到 0，也不能涨到无穷大，否则梯度会消失或爆炸。

Xavier 与 He 初始化按层宽设定方差，使初始化时刻每层的预激活保持 $O(1)$。深层网络中信号要先传得动，才谈得上正则。

### 不能全零

**全零（或全部相同）会导致对称性问题**。若同一层每一行权重都相等，前向时这一层所有神经元的预激活相等、输出相等；反向时每一行收到的梯度也相等（回忆乘法节点：梯度依赖另一个输入，而这些输入都相同），更新后仍然相等。于是整层只学到同一个特征，有 100 个神经元也只相当于 1 个。随机初始化打破这组相等。偏置仍可取 $0$，因为权重已经不对称了。

注意，问题不在于“零”，而在于“相同”：同层权重若完全相同，即使不是零，梯度也相同，对称性同样破不掉。

### 尺度太小或太大

> [!EXAMPLE]+ 例3 激活值分布的实验（鱼书 6.2 节，参考 CS231n）
>
> 5 层网络，每层 100 个神经元，激活函数为 sigmoid，输入 1000 个高斯随机样本，观察各层激活值的直方图：
>
> - **标准差为 1**：各层激活值集中在 0 和 1 两端。sigmoid 在两端导数接近 0，反向传播时梯度越来越小，最终消失，即**梯度消失**（gradient vanishing）。
> - **标准差为 0.01**：激活值集中在 0.5 附近。梯度不会消失，但 100 个神经元输出几乎相同的值，和 1 个神经元没区别，称为**表现力受限**。
> - **Xavier 初始化**（标准差 $1/\sqrt{n}$，$n$ 为前一层节点数）：各层激活值都有较宽的分布，后面几层略有歪斜。把 sigmoid 换成关于原点对称的 tanh，分布会变成漂亮的钟形。
>
> 结论：**各层激活值的分布需要有适当的广度**。偏向两端会梯度消失，挤在一起会表现力受限。

更一般地：

- **方差过小**（标准差 $0.01$）：激活和梯度的标准差逐层一起变小，信号消失。
- **方差过大**（标准差 $1$）：ReLU 与线性层上两者逐层变大，梯度爆炸；tanh / sigmoid 则进入饱和区，局部导数接近 $0$，同样学不动。
- **按 Xavier 或 He 取值**：激活与梯度跨层保持同一量级。

### Xavier 初始化

推导的出发点是：让每层输出的方差与输入的方差相等。对线性层 $z=\sum_{j=1}^{n_{\mathrm{in}}} w_j a_j$，若 $a_j$ 零均值、方差 $v_a$，$w_j$ 独立零均值、方差 $v_w$，则

$$
\mathrm{Var}(z)=n_{\mathrm{in}}\, v_w\, v_a.
$$

（$n_{\mathrm{in}}$ 项独立随机变量之和，每项的方差是 $v_w v_a$。）

- 希望前向时 $\mathrm{Var}(z)\approx\mathrm{Var}(a)$，需要 $v_w\approx 1/n_{\mathrm{in}}$。
- 同样要求反向时 $\mathrm{Var}(\mathrm{d}a)\approx\mathrm{Var}(\mathrm{d}z)$，得到 $v_w\approx 1/n_{\mathrm{out}}$。
- 两者通常不能同时满足，取调和折中（Xavier / Glorot）：

$$
W^{[l]}\sim\mathcal{N}\left(0,\frac{2}{n^{[l-1]}+n^{[l]}}\right)
\quad\text{或均匀分布等价尺度}.
$$

Caffe 等框架常简化为只看输入，即 $v_w=1/n_{\mathrm{in}}$、标准差 $1/\sqrt{n}$。直观上，前一层节点越多，求和项越多，每个权重就应越小。

Xavier 的推导假设激活函数在 0 附近近似线性。sigmoid 和 tanh 关于中心对称、中间段近似线性，所以适用 Xavier。

### He 初始化

ReLU 把负半轴置零。对零均值对称的输入，这会使二阶矩 $\mathbb{E}[a^{2}]$ 大约折半（注意不是严格把方差折半）。为了补偿丢掉的一半，He 初始化把方差乘 2：

$$
W^{[l]}\sim\mathcal{N}\left(0,\frac{2}{n^{[l-1]}}\right).
$$

> [!EXAMPLE]+ 例4 ReLU 网络的三种初始化（鱼书 6.2.3 节）
>
> 同样的 5 层、每层 100 个神经元的网络，激活换成 ReLU：
>
> - **std = 0.01**：各层激活的均值依次约为 $0.0396,\ 0.00290,\ 0.000197,\ 1.32\times 10^{-5},\ 9.46\times 10^{-7}$，每层缩小一个数量级以上，反向梯度同样极小，学习基本没有进展。
> - **Xavier**：随层数加深，分布逐渐偏向 0，层再多就会梯度消失。
> - **He**：各层分布的广度保持一致。
>
> 在 MNIST 上训练（5 层，每层 100 个 ReLU 神经元）：std = 0.01 完全学不动；Xavier 与 He 都能顺利学习，He 更快一些。
>
> 当前的最佳实践：**ReLU 用 He 初始化，sigmoid / tanh 用 Xavier 初始化**。

> [!INFO]+ Xavier 与 He
>
> Xavier / Glorot (2010) 按层宽设定 $\mathrm{Var}(W)$，使初始化时预激活保持 $O(1)$，适合 tanh 与 sigmoid。He et al. (2015) 考虑到 ReLU 丢掉负半轴、二阶矩约折半，改用 $2/n_{\mathrm{in}}$。二者都只约束初始化时刻；训练开始后方差仍会变化，这也是后面需要 BatchNorm 的原因之一。

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

用 NumPy 复现例 4 的观察：5 层、每层 100 个 ReLU，看各层激活的标准差怎么变。

```python
def activation_stats(scale_fn, n=100, L=5, m=1000, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m))
    stds = []
    for _ in range(L):
        W = rng.normal(0.0, scale_fn(n), size=(n, n))
        A = np.maximum(0.0, W @ A)
        stds.append(round(float(A.std()), 4))
    return stds

print(activation_stats(lambda n: 0.01))               # [0.058, 0.0043, 0.0003, 0.0, 0.0]
print(activation_stats(lambda n: np.sqrt(1.0 / n)))   # [0.58, 0.43, 0.31, 0.24, 0.16]
print(activation_stats(lambda n: np.sqrt(2.0 / n)))   # [0.83, 0.87, 0.88, 0.95, 0.91]
```

- std $=0.01$：每层缩小一个数量级以上；
- Xavier（ReLU 下）：每层约乘 $1/\sqrt2$；
- He：各层基本不变。

PyTorch 的 `nn.Linear` 默认用 Kaiming-uniform（$a=\sqrt5$），方差比 He 小，等价于 $\mathrm{Var}(W)=1/(3n_{\mathrm{in}})$。要与作业一致，需要显式重设：

```python
import torch.nn as nn

def init_he(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")   # std = sqrt(2/fan_in)
        nn.init.zeros_(module.bias)

def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)    # std = sqrt(2/(fan_in+fan_out))
        nn.init.zeros_(module.bias)

model = nn.Sequential(nn.Linear(12288, 20), nn.ReLU(),
                      nn.Linear(20, 7), nn.ReLU(),
                      nn.Linear(7, 5), nn.ReLU(),
                      nn.Linear(5, 1))
model.apply(init_he)          # apply 会递归作用到每个子模块
```

### 梯度消失与爆炸的连乘解释

反向时，梯度从第 $L$ 层传到第 1 层，要依次乘上每层的 $W^{[l]\top}\mathrm{diag}(g'(z^{[l]}))$，即经过连乘 $\prod_l W^{[l]\top} \mathrm{diag}(g'(z^{[l]}))$。

- 每层的“放大倍数”（谱半径）$>1$：梯度随层数指数增长，即**爆炸**。
- 每层的谱半径 $<1$：梯度随层数指数衰减，即**消失**。
- 好的初始化让每层倍数接近 1；残差连接把每层的乘子改成“$1+$ 小残差”，从结构上缓解这个问题。

> [!EXAMPLE]+ 例5 过小与过大初始化的数量级
>
> 设十层 tanh 网络，$n^{[l]}=100$。若 $W_{ij}\sim\mathcal{N}(0,0.01^{2})$，每层方差约乘 $100\times 10^{-4}=0.01$，激活幅度逐层收缩，最后一层几乎是常数，梯度接近 0。若 $W_{ij}\sim\mathcal{N}(0,1)$，每层方差约乘 $100$，tanh 饱和，导数接近 0，同样学不动。同一套 $\mathcal{N}(0,1)$ 若接 ReLU，正半轴不截断，激活二阶矩大约每层乘 $n_{\mathrm{in}}/2=50$，激活与梯度一起爆炸。He 对 ReLU 取 $\mathrm{Var}=2/n_{\mathrm{in}}=0.02$，标准差约 $0.14$，使 $\mathrm{Var}(z)$ 保持 $O(1)$。
>
> 作业对比图通常显示：零初始化训练准确率停在随机水平；过大初始化损失曲线震荡；He 初始化损失平稳下降。这是 C2M1 编程作业的核心观察。

<br>

> [!NOTE]+ 衔接
>
> 初始化保证了训练开始时信号和梯度的尺度合适。接下来的问题是：有了梯度，参数该怎么走？下一节讨论优化算法。注意 Dropout 等正则方法不能代替好的初始化，它们解决的是另一类问题。

<br>

## 优化（Optimization）

### 问题

优化（optimization）决定参数沿梯度怎么走。损失 $J$ 已经给定，这里只选**更新规则**：输入是梯度 $\mathrm{d}W$，输出是新的 $W$。

神经网络的优化很难：参数空间维度极高、形状复杂，不可能像解方程那样一步求出最小值。鱼书用一个比喻描述这种处境：一位蒙着眼睛、没有地图的探险家，要在广袤的山地中找到最深的谷底，唯一的线索是脚下地面的坡度。SGD 的策略就是“朝坡度最陡的方向走”，而本节的各种方法都是在改进这个策略。

损失不降、曲线剧烈震荡或出现 `nan`，属于优化失败；加大 $\lambda$ 或加 Dropout 改变的是模型的有效容量，不能代替调学习率。

### Mini-batch

> [!INFO]+ Mini-batch
>
> Mini-batch 用 $n_b$ 条样本估计梯度并更新一次，损失定义不变。$n_b=m$ 是全批量，方向稳、每个 epoch 只走一步；$n_b=1$ 是随机梯度，噪声大。常用 $32$ 到 $512$。过小则 BatchNorm 统计不稳，过大则一个 epoch 内更新次数变少。

<br>

把 $m$ 个样本打乱后分成大小 $n_b$ 的批，每批更新一次参数。第 $t$ 批的损失为：

$$
J_t=\frac{1}{n_b}\sum_{i\in\mathcal{B}_t}\mathcal{L}^{(i)}.
$$

由于 mini-batch 是随机抽取的，它的梯度是全批量梯度的**无偏估计**，只是带有噪声，所以这种做法称为**随机梯度下降**（stochastic gradient descent, SGD）。

- $n_b=m$：批量梯度下降，方向稳定，但每个 epoch 只更新 1 次，大数据集上太慢。
- $n_b=1$：严格意义的 SGD，噪声大，而且无法利用向量化加速。
- $n_b\in\{32,64,128,256,512\}$：折中。常用 2 的幂以契合 GPU 的计算块。

几个相关概念：

- **一个 epoch** 等于全部训练样本都被看过一次。例如 6 万个样本、batch 大小 100，一个 epoch 是 600 次迭代。
- mini-batch 的损失曲线呈云状下降，不是单调的；若要平滑，对 $J_t$ 做指数滑动平均。
- 每个 epoch 开始时都要重新打乱样本，否则同一 mini-batch 的偏差会被网络记住。

> [!NOTE]+ 批量大小
>
> batch 增大时，梯度估计方差（variance of the gradient estimate）下降，矩阵乘吞吐通常提高，但一个 epoch 内的参数更新次数减少。显存占用增加，而且低噪声轨迹有时收敛到泛化较差的尖锐区域（sharp minima）。
>
> - 小批量（small batch）：更新噪声大、每个 epoch 更新多，吞吐可能较低。
> - 大批量（large batch）：梯度稳定、硬件利用率高，但需要更多显存，学习率通常也要重新调。
> - 线性缩放规则（linear scaling rule）只是经验起点：batch 放大 $k$ 倍时把学习率放大约 $k$ 倍，并配学习率预热（learning-rate warmup）。它不是对所有优化器和任务都成立的定理。
> - 比较实验时要同时说明 batch size、每个 epoch 的更新次数、总样本数与总优化步数，否则“收敛更快”没有统一含义。

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

PyTorch 把“打乱 + 切片”交给 `DataLoader`；`shuffle=True` 会在每个 epoch 重新打乱。

```python
from torch.utils.data import TensorDataset, DataLoader

ds = TensorDataset(torch.tensor(X.T, dtype=torch.float32),    # 行样本 (m, n_x)
                   torch.tensor(Y.T, dtype=torch.float32))
g = torch.Generator().manual_seed(0)                            # 固定打乱顺序，便于复现
loader = DataLoader(ds, batch_size=64, shuffle=True, generator=g, drop_last=False)

for epoch in range(num_epochs):            # 一个 epoch = 遍历一次 loader
    for xb, yb in loader:                  # xb: (64, n_x)，最后一批可能更小
        ...
```

- `drop_last=True` 会丢掉最后不满的一批。带 BatchNorm 时，如果最后一批只剩 1 个样本，方差为 0，训练会报错或不稳定，这时应开启它。
- dev/test 的 loader 用 `shuffle=False`，保证每次评估顺序一致。

### SGD 的缺点

普通 SGD 的更新是 $W\leftarrow W-\eta\,\frac{\partial L}{\partial W}$（$\eta$ 即学习率 $\alpha$）。它简单，但在某些地形上效率很低。

> [!EXAMPLE]+ 例6 狭长山谷中的 SGD（鱼书 6.1.3 节）
>
> 考虑
>
> $$
> f(x,y)=\frac{1}{20}x^{2}+y^{2}.
> $$
>
> 它的图形是沿 $x$ 轴拉长的碗，等高线是沿 $x$ 方向拉长的椭圆。梯度 $(x/10,\ 2y)$ 在 $y$ 方向大、$x$ 方向小，在大多数位置**并不指向最小值 $(0,0)$**。从 $(-7,2)$ 出发做 SGD，路径呈“之”字形：在陡峭的 $y$ 方向来回震荡，在平缓的 $x$ 方向缓慢前进。
>
> 这种各方向曲率差别很大的函数称为**非均向**（anisotropic）的。SGD 低效的根本原因是梯度方向没有指向最小值。下面的 Momentum、AdaGrad / RMSprop、Adam 分别从“平滑方向”和“按维度调步长”两个角度改进。

### 指数滑动平均（exponentially weighted moving average）

后面几种方法都依赖一个工具：指数滑动平均（EWMA）。它用上一时刻的平滑值和新观测做凸组合，得到“近期数据的平均”：

$$
v_t=\beta v_{t-1}+(1-\beta)\theta_t.
$$

展开后 $v_t=(1-\beta)\sum_{k\le t}\beta^{t-k}\theta_k$，越早的观测权重按 $\beta$ 指数衰减。$v_t$ 大约平均了最近 $1/(1-\beta)$ 个值：$\beta=0.9$ 约 10 步，$\beta=0.999$ 约 1000 步。Momentum、RMSprop、Adam 里的 $v,s$，以及损失曲线的平滑，用的都是它。

**偏差修正**：初始 $v_0=0$，前几步的 $v_t$ 会被这个 0 拉低。例如 $\beta=0.9$、$\theta_1=10$ 时 $v_1=1$，远小于真实水平 10。除以 $1-\beta^t$ 可以修正：

$$
\hat{v}_t=\frac{v_t}{1-\beta^{t}}.
$$

$t=1$ 时 $\hat{v}_1=1/(1-0.9)=10=\theta_1$。随着 $t$ 增大，$\beta^t\to 0$，修正自动失效。

```python
def ewma(xs, beta=0.9, bias_correction=True):
    """返回逐步的滑动平均；xs 为一维序列（如每步的损失）。"""
    v, out = 0.0, []
    for t, x in enumerate(xs, start=1):
        v = beta * v + (1.0 - beta) * x
        out.append(v / (1.0 - beta ** t) if bias_correction else v)
    return np.array(out)

ewma([10, 10, 10], bias_correction=False)   # array([1.  , 1.9 , 2.71])
ewma([10, 10, 10])                          # array([10., 10., 10.])
```

### Momentum

> [!INFO]+ Momentum
>
> Momentum 把梯度的指数滑动平均当作速度，再沿速度更新。来自 Polyak 重球法，$\beta$ 常取 $0.9$。狭长谷里左右振荡的分量会互相抵消，沿谷底的分量会累积。

<br>

Momentum（动量）借用物理直觉：参数像一个在地面上滚动的小球，梯度是作用在它上面的力，力改变的是**速度**，速度再改变位置。

$$
\begin{aligned}
v_{\mathrm{d}W} &= \beta v_{\mathrm{d}W} + (1-\beta)\,\mathrm{d}W,\\
W &\leftarrow W-\alpha v_{\mathrm{d}W}.
\end{aligned}
$$

在例 6 的山谷中：$x$ 方向的力虽小，但始终同向，速度不断累积；$y$ 方向的力虽大，但正负交替，在平均中相互抵消。于是“之”字形明显减弱，更快地沿谷底前进。

这一点可以定量算出来。只看一个坐标，分两种极端情况求 $v$ 的稳态（令 $v_t=v_{t-1}$）：

- **梯度恒为 $g$**（沿谷底的方向）：$v=\beta v+(1-\beta)g$，解得 $v=g$。速度最终等于梯度本身，每步走 $\alpha g$，与 SGD 相同，没有被削弱。
- **梯度正负交替** $+g,-g,+g,\dots$（横跨山谷的方向）：稳态时相邻两步的 $v$ 符号相反、大小相等，设为 $\pm v$。代入 $v=\beta(-v)+(1-\beta)g$，得

$$
v=\frac{1-\beta}{1+\beta}\,g.
$$

$\beta=0.9$ 时 $v=0.0526\,g$：震荡方向的步长被压到原来的约 $\frac{1}{19}$，而一致方向的步长不变。这就是 Momentum 的全部作用：**一致的分量保留，来回抖的分量抵消**。$\beta$ 越大，抵消越彻底，但对方向变化的反应也越慢（大约要 $1/(1-\beta)$ 步才能“转过弯来”）。

鱼书写法是 $v\leftarrow\alpha v-\eta\frac{\partial L}{\partial W}$，$W\leftarrow W+v$，其中 $\alpha v$ 项相当于摩擦或空气阻力，让小球在没有外力时逐渐减速（$\alpha$ 取 0.9 左右）。它与上式只差一个尺度：令 $\alpha_{\text{鱼书}}=\beta$，则鱼书的 $v$ 等于 $-\frac{\eta}{1-\beta}v_{\mathrm{d}W}$，即有效学习率为 $\eta/(1-\beta)$。所以把 $(1-\beta)$ 省略的实现，$\alpha$ 的有效尺度会改变，超参数不能直接照抄。

Nesterov 动量先沿速度方向走一步，再在新位置求梯度，对凸问题有更紧的收敛率；课程以普通 Momentum 为主。

### AdaGrad 与 RMSprop

> [!INFO]+ RMSprop
>
> RMSprop 用梯度平方的滑动平均去除以当前梯度，长期偏大的坐标步长变小，偏小的变大。AdaGrad 把历史梯度平方累加后开方去除以步长，学习率单调衰减，后期几乎走不动。RMSprop 改成滑动平均，避免这一项只增不减。

<br>

另一个思路是**给每个参数分配自己的学习率**。学习率衰减是让所有参数的学习率一起变小；AdaGrad（Ada 取自 adaptive）进一步按参数的每个元素分别衰减：

$$
\begin{aligned}
h &\leftarrow h+\frac{\partial L}{\partial W}\odot\frac{\partial L}{\partial W},\\
W &\leftarrow W-\eta\,\frac{1}{\sqrt{h}+\varepsilon}\,\frac{\partial L}{\partial W}.
\end{aligned}
$$

$h$ 累加了历史上所有梯度的平方。某个参数过去变动越大，$h$ 越大，它的步长就越小。在例 6 中，$y$ 方向梯度大，一开始变动大，但很快被按比例压小，“之”字形明显减弱。$\varepsilon$（鱼书取 $10^{-7}$）防止除零。

AdaGrad 的问题是 $h$ **只增不减**：学得越久，步长越小，最终完全停止更新。RMSprop 把累加改成指数滑动平均，逐渐遗忘久远的梯度：

$$
\begin{aligned}
s_{\mathrm{d}W} &= \beta_2 s_{\mathrm{d}W} + (1-\beta_2)\,\mathrm{d}W\odot\mathrm{d}W,\\
W &\leftarrow W-\alpha\frac{\mathrm{d}W}{\sqrt{s_{\mathrm{d}W}}+\varepsilon}.
\end{aligned}
$$

$\varepsilon$ 如 $10^{-8}$，防止除零。梯度长期偏大的坐标步长被缩小，偏小的被放大，自动调整各维学习率。

**为什么是除以 $\sqrt{s}$**。$s$ 是梯度平方的平均，$\sqrt{s}$ 就是梯度的均方根（RMS，RMSprop 名字的由来），和梯度同一量纲。三个角度理解这一除：

1. **步长与梯度尺度无关**。若梯度一直是常数 $g$，则 $s\to g^2$，更新为 $\alpha g/\lvert g\rvert=\alpha\,\mathrm{sign}(g)$：无论 $g$ 是 $1000$ 还是 $0.001$，每步都走 $\alpha$。把损失整体乘以 $c$，SGD 的步长也乘以 $c$，RMSprop 的步长不变。于是 $\alpha$ 的含义变成“参数每步大约移动多少”，比 SGD 的学习率直观得多。
2. **各维度一视同仁**。例 6 中 $y$ 方向梯度是 $x$ 方向的约 $20$ 倍，SGD 只能按 $y$ 方向选一个小学习率，$x$ 方向因此走得极慢。除以各自的 RMS 后，两个方向的步长同量级。
3. **近似的曲率修正**。对一维二次函数 $f=\frac12 h w^2$，牛顿法的步长是 $g/h$：曲率大的方向走小步。梯度 $g=hw$ 的大小正比于曲率 $h$，所以除以 $\lvert g\rvert$ 粗略地起到了除以曲率的作用。它不是真正的二阶方法，只是一个便宜的对角近似。

AdaGrad 用的是累加 $h=\sum_k g_k^2$，到第 $t$ 步大约是 $t\cdot\overline{g^2}$，步长按 $1/\sqrt{t}$ 衰减，这是它后期停滞的原因；RMSprop 的滑动平均只记最近约 $1/(1-\beta_2)$ 步，$s$ 不会无限增长。

Momentum 与 RMSprop 的 NumPy 更新，接口与后面的 `adam_update` 相同：

```python
def momentum_update(params, grads, v, lr, beta=0.9):
    for k in params:
        v["d" + k] = beta * v["d" + k] + (1.0 - beta) * grads["d" + k]
        params[k] = params[k] - lr * v["d" + k]
    return params, v

def rmsprop_update(params, grads, s, lr, beta2=0.999, eps=1e-8):
    for k in params:
        g = grads["d" + k]
        s["d" + k] = beta2 * s["d" + k] + (1.0 - beta2) * g * g
        params[k] = params[k] - lr * g / (np.sqrt(s["d" + k]) + eps)
    return params, s
```

在例 6 的 $f(x,y)=x^{2}/20+y^{2}$ 上直接对比路径：

```python
def run(update, steps=30, **kw):
    p = {"W": np.array([-7.0, 2.0])}
    state = {"dW": np.zeros(2)}
    path = [p["W"].copy()]
    for _ in range(steps):
        x, y = p["W"]
        g = {"dW": np.array([x / 10.0, 2.0 * y])}
        p, state = update(p, g, state, **kw)
        path.append(p["W"].copy())
    return np.array(path)

sgd = lambda p, g, s, lr: ({"W": p["W"] - lr * g["dW"]}, s)
for name, fn, kw in [("SGD", sgd, {"lr": 0.95}),
                     ("Momentum", momentum_update, {"lr": 0.95}),
                     ("RMSprop", rmsprop_update, {"lr": 0.3, "beta2": 0.9})]:
    print(name, np.round(run(fn, **kw)[-1], 3))
# SGD      [-0.35   0.085]
# Momentum [ 0.75   0.228]
# RMSprop  [-0.154  0.   ]
```

- SGD 在 $y$ 方向每步乘 $1-2\times 0.95=-0.9$，符号来回翻转，这就是之字形；$x$ 方向每步只乘 $0.905$，走得很慢。
- Momentum 把 $y$ 方向正负交替的梯度平均掉，$x$ 方向持续累积速度；速度大了会冲过最低点再折返，所以第 30 步时离原点未必更近。把 `path` 画出来，能看到它的轨迹比 SGD 平滑得多。
- RMSprop 按各自的梯度尺度归一化，两个方向的步长同量级，最先把 $y$ 压到 $0$。

### Adam

> [!INFO]+ Adam
>
> Adam（Kingma & Ba, 2015）同时保存梯度的一阶矩（Momentum）和二阶矩（RMSprop），并用 $1-\beta^t$ 修正 $v_0=s_0=0$ 带来的初期偏差。默认 $\beta_1=0.9$，$\beta_2=0.999$。学习率 $\alpha$ 仍要选；实现时 $t$ 必须从 1 递增。

<br>

Momentum 平滑了方向，RMSprop 按维度调整了步长，Adam（adaptive moment estimation）把两者合在一起，再加上偏差修正：

$$
\begin{aligned}
v_{\mathrm{d}W} &= \beta_1 v_{\mathrm{d}W}+(1-\beta_1)\,\mathrm{d}W,\\
s_{\mathrm{d}W} &= \beta_2 s_{\mathrm{d}W}+(1-\beta_2)\,\mathrm{d}W\odot\mathrm{d}W,\\
\hat{v}\_{\mathrm{d}W} &= \frac{v_{\mathrm{d}W}}{1-\beta_1^{t}},\qquad
\hat{s}\_{\mathrm{d}W} = \frac{s_{\mathrm{d}W}}{1-\beta_2^{t}},\\
W &\leftarrow W-\alpha\frac{\hat{v}\_{\mathrm{d}W}}{\sqrt{\hat{s}\_{\mathrm{d}W}}+\varepsilon}.
\end{aligned}
$$

- 默认 $\beta_1=0.9$，$\beta_2=0.999$，$\varepsilon=10^{-8}$。在例 6 中，Adam 的路径也像小球在碗中滚动，但左右摇晃比 Momentum 更轻。
- $t$ 是更新步数，从 1 计。
- 每个参数张量（每个 $W^{[l]},b^{[l]}$）都有自己的 $v,s$。
- Adam 对初始学习率仍敏感，常用 $\alpha\in\{10^{-3},3\times10^{-4},10^{-4}\}$，再配合衰减。

> [!INFO]+ 偏差修正为何必要
>
> $s$ 的 $\beta_2=0.999$ 使早期 $s_t\approx(1-\beta_2)\sum_{k} \beta_2^{t-k} g_k^{2}$ 远小于稳态值。若不除以 $1-\beta_2^{t}$，分母过小，最初若干步的更新会过大。$\beta_1$ 同样需要修正，但 $\beta_1=0.9$ 时偏差消失更快。实现中若漏掉 $t$ 的递增，等于永远按 $t=1$ 修正，学习率标度错误。

<br>

#### 一步步推导 Adam

**第 1 步：想要什么**。把每个坐标的梯度 $g_t$ 看成随机变量（mini-batch 带来噪声），有均值 $\mathbb E[g]$ 和二阶矩 $\mathbb E[g^2]$。理想的更新应该：

- 方向跟着**真实梯度的均值** $\mathbb E[g]$ 走，而不是跟着单个 batch 的噪声走；
- 步长按这个坐标的**梯度尺度** $\sqrt{\mathbb E[g^2]}$ 归一化，使各维度、各层的步长同量级。

于是目标形如 $\Delta w=-\alpha\,\mathbb E[g]/\sqrt{\mathbb E[g^2]}$。Adam 的名字 adaptive moment estimation 指的就是“估计一阶矩和二阶矩”。

**第 2 步：用滑动平均估计两个矩**。两个期望都不能直接算，只能用历史梯度估计。用指数滑动平均（越近的梯度权重越大，适应参数在变化的事实）：

$$
v_t=\beta_1v_{t-1}+(1-\beta_1)g_t\approx\mathbb E[g],\qquad
s_t=\beta_2s_{t-1}+(1-\beta_2)g_t^2\approx\mathbb E[g^2].
$$

$v$ 就是 Momentum 的速度，$s$ 就是 RMSprop 的分母。$\beta_1=0.9$ 约平均最近 10 步，方向反应快；$\beta_2=0.999$ 约平均最近 1000 步，尺度估计稳。尺度要稳是因为它在分母上，抖动会直接放大成步长的抖动。

**第 3 步：发现估计有偏**。$v_0=0$，把递推展开：

$$
v_t=(1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k}g_k.
$$

假设梯度分布不随时间变（$\mathbb E[g_k]=\mathbb E[g]$），取期望，并用等比数列求和 $\sum_{k=1}^{t}\beta_1^{t-k}=\frac{1-\beta_1^{t}}{1-\beta_1}$：

$$
\mathbb E[v_t]=(1-\beta_1)\cdot\frac{1-\beta_1^{t}}{1-\beta_1}\cdot\mathbb E[g]=(1-\beta_1^{t})\,\mathbb E[g].
$$

所以 $v_t$ 系统性地偏小，偏小的比例正好是 $1-\beta_1^{t}$。原因是滑动平均的权重 $(1-\beta_1)\beta_1^{t-k}$ 加起来只有 $1-\beta_1^t$，缺的那部分权重分给了初始值 $v_0=0$。$s_t$ 同理：$\mathbb E[s_t]=(1-\beta_2^{t})\mathbb E[g^2]$。

**第 4 步：修正**。既然偏小的比例已知，除掉即可：

$$
\hat v_t=\frac{v_t}{1-\beta_1^{t}},\qquad \hat s_t=\frac{s_t}{1-\beta_2^{t}},\qquad
\mathbb E[\hat v_t]=\mathbb E[g],\quad \mathbb E[\hat s_t]=\mathbb E[g^2].
$$

**第 5 步：合成更新**，加上 $\varepsilon$ 防止除零，就得到 Adam：

$$
w_t=w_{t-1}-\alpha\,\frac{\hat v_t}{\sqrt{\hat s_t}+\varepsilon}.
$$

#### 不修正会怎样

两个矩的偏差大小不同，所以不修正时它们不会互相抵消。看第 1 步（$t=1$）：

$$
v_1=(1-\beta_1)g_1=0.1\,g_1,\qquad s_1=(1-\beta_2)g_1^2=0.001\,g_1^2,\qquad
\frac{v_1}{\sqrt{s_1}}=\frac{0.1}{\sqrt{0.001}}\,\mathrm{sign}(g_1)\approx 3.16\,\mathrm{sign}(g_1).
$$

不修正，第一步就走 $3.16\alpha$；修正后 $\hat v_1=g_1$、$\hat s_1=g_1^2$，正好走 $\alpha$。梯度恒为常数时，逐步的步长（以 $\alpha$ 为单位）如下：

| $t$ | 1 | 2 | 3 | 4 | 5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 不修正 | 3.16 | 4.25 | 4.95 | 5.44 | 5.80 |
| 修正 | 1 | 1 | 1 | 1 | 1 |

不修正时，前几千步的步长是设定值的数倍（直到 $\beta_2^t$ 衰减掉，约需几千步），训练初期最容易发散，正好落在参数还很差、最需要稳的阶段。

#### 每一步走多远

修正后，更新量 $\Delta_t=\alpha\,\hat v_t/\sqrt{\hat s_t}$ 中的比值有一个很好的解释。设梯度的均值为 $\mu$、方差为 $\sigma^2$，则 $\mathbb E[g^2]=\mu^2+\sigma^2$，

$$
\frac{\hat v_t}{\sqrt{\hat s_t}}\approx\frac{\mu}{\sqrt{\mu^2+\sigma^2}}\in[-1,1].
$$

它是梯度的**信噪比**：

- 梯度方向稳定（$\sigma\ll\lvert\mu\rvert$）：比值接近 $\pm1$，每步走满 $\alpha$；
- 梯度噪声大（$\sigma\gg\lvert\mu\rvert$），通常是接近最优点时：比值接近 $0$，步长自动变小，相当于自带了学习率退火。

用均值为 $\mu$、标准差为 $1$ 的随机梯度跑 5000 步，统计稳定后平均每步走多远（以 $\alpha$ 为单位）：

| $\mu$ | 理论 $\mu/\sqrt{\mu^2+1}$ | 实测 |
| :--- | :--- | :--- |
| $1.0$ | $0.707$ | $0.717$ |
| $0.1$ | $0.0995$ | $0.117$ |
| $0.0$ | $0$ | $0.019$ |

另外 $\lvert\hat v_t\rvert\le\sqrt{\hat s_t}$ 大致成立（均值的平方不超过平方的均值），所以 $\lvert\Delta_t\rvert$ 一般不超过 $\alpha$。Adam 论文给出的严格上界分两种参数情形：$1-\beta_1\le\sqrt{1-\beta_2}$ 时 $\lvert\Delta_t\rvert\le\alpha$；$1-\beta_1>\sqrt{1-\beta_2}$ 时（默认的 $0.1>0.0316$ 属于这种）$\lvert\Delta_t\rvert\le\alpha(1-\beta_1)/\sqrt{1-\beta_2}\approx3.16\alpha$。后者只在梯度长期接近 $0$、某一步突然变大时才会接近，常见情形下 $\lvert\Delta_t\rvert$ 不超过 $\alpha$。因此 $\alpha$ 可以理解为参数空间里的**信赖域半径**：知道参数大致的合理尺度，就能估出 $\alpha$ 的量级。

梯度正负交替（$+1,-1,+1,\dots$）时，修正后的步长依次是 $1,-0.053,0.336,-0.053,0.204,\dots$，奇数步逐渐减小、偶数步稳定在 $-0.053$：与 Momentum 的 $\frac{1-\beta}{1+\beta}$ 一样，来回抖的方向被自动压低。

#### 超参数各管什么

| 超参数 | 默认 | 作用 | 何时改 |
| :--- | :--- | :--- | :--- |
| $\alpha$ | $10^{-3}$ | 每步大约移动多少 | 最需要调；Transformer 常用 $10^{-4}$ 到 $3\times10^{-4}$ 并加 warmup |
| $\beta_1$ | $0.9$ | 方向平滑的时间窗约 10 步 | GAN 常取 $0.5$，让方向反应更快 |
| $\beta_2$ | $0.999$ | 尺度估计的时间窗约 1000 步 | 大模型训练常取 $0.95$ 或 $0.98$，让 $s$ 更快跟上梯度尺度的突变，减少损失尖峰 |
| $\varepsilon$ | $10^{-8}$ | 防止除零 | 梯度极小或用 fp16 时可增大到 $10^{-6}$ 甚至更大；$\varepsilon$ 很大时 Adam 退化为带动量的 SGD，学习率为 $\alpha/\varepsilon$ |

#### 变体

- **AMSGrad**：用 $\hat s_t=\max(\hat s_{t-1},s_t)$，分母只增不减，修复了 Adam 在某些构造的凸问题上不收敛的反例（Reddi et al., 2018）；实践中提升不稳定，PyTorch 中为 `Adam(amsgrad=True)`。
- **Nadam**：把 Nesterov 动量用到 $\hat v$ 上，先“往前看一步”。
- **AdamW**：解耦权重衰减，见下面的推导；现在训练 Transformer 的默认选择。
- **Adafactor / 8-bit Adam**：Adam 对每个参数要额外存 $v,s$ 两份，显存是参数的 2 倍。Adafactor 把矩阵的 $s$ 分解成行、列两个向量，8-bit Adam 把 $v,s$ 量化，都是为了省显存。

> [!NOTE]+ $L_2$ 与 AdamW
>
> 对普通 SGD，在损失中加入 $\lambda\|W\|^{2}/2$ 与每步按比例缩小 $W$ 基本等价。Adam 会按坐标缩放梯度，因此把 $\lambda W$ 混进梯度，不再等价于统一的权重衰减。
>
> AdamW 使用解耦权重衰减（decoupled weight decay）：
>
> $$
> W\leftarrow
> (1-\alpha\lambda)W
> -\alpha\frac{\hat v}{\sqrt{\hat s}+\varepsilon}.
> $$
>
> 这里衰减项不经过 Adam 的二阶矩缩放。英文题目中的 `L2 regularization with Adam` 与 `AdamW weight decay` 不能自动当作同一件事。

<br>

**为什么不等价，推一遍**。损失加 $\frac{\lambda}{2}\lVert w\rVert^2$ 后，梯度变成 $g+\lambda w$。

- SGD：$w\leftarrow w-\alpha(g+\lambda w)=(1-\alpha\lambda)w-\alpha g$。每个参数都按同一个比例 $1-\alpha\lambda$ 收缩，L2 正则与权重衰减完全相同。
- Adam：$\lambda w$ 进入 $v$ 和 $s$，最后被除以 $\sqrt{\hat s}$。忽略动量，衰减部分约为 $\alpha\lambda w/\sqrt{\hat s}$，**每个参数的衰减强度反比于它的梯度 RMS**。

举个数：两个权重都等于 $1$，一个梯度 RMS 是 $0.1$，另一个是 $0.001$。L2 + Adam 下，后者每步的衰减是前者的 $100$ 倍；梯度大的权重几乎不被正则。这与“惩罚大权重”的初衷不符：梯度大的往往正是最活跃、最需要约束的权重。

AdamW 把衰减从梯度中拿出来，直接作用在参数上：

$$
w\leftarrow w-\alpha\Big(\frac{\hat v}{\sqrt{\hat s}+\varepsilon}+\lambda w\Big).
$$

衰减比例对每个参数都是 $\alpha\lambda$，与梯度尺度无关。另一个好处是 $\lambda$ 与 $\alpha$ 的作用分开了，两者可以分别调。PyTorch 中 `AdamW` 的默认 `weight_decay=0.01`，乘的是当前学习率，所以学习率衰减时权重衰减也一起变小。

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

PyTorch 中的对应写法：

```python
import torch

params = list(model.parameters())     # parameters() 是生成器，复用前先转成列表
opt_sgd  = torch.optim.SGD(params, lr=0.1)
opt_mom  = torch.optim.SGD(params, lr=0.1 * (1 - 0.9), momentum=0.9)    # 见下方说明
opt_rms  = torch.optim.RMSprop(params, lr=1e-3, alpha=0.999, eps=1e-8)  # alpha 即 beta2
opt_adam = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
opt_adw  = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)        # 解耦权重衰减
```

- `torch.optim.SGD(momentum=β)` 用的是鱼书的写法：$v\leftarrow\beta v+g$，$W\leftarrow W-\alpha v$，**没有** $(1-\beta)$。要与 CS230 公式在同一个 $\alpha$ 下等价，PyTorch 的学习率应取 $\alpha(1-\beta)$，这正是前面说的“超参数不能直接照抄”。
- `RMSprop` 的平滑系数叫 `alpha`（默认 $0.99$），不是学习率。
- `Adam(weight_decay=λ)` 是把 $\lambda W$ 加进梯度（经过二阶矩缩放），`AdamW(weight_decay=λ)` 才是解耦衰减。
- 一个优化器只应创建一次。若在训练循环里反复新建，$v,s$ 和步数 $t$ 每步都被清零，Adam 就退化成每步都按 $t=1$ 做偏差修正。

用上面的 `adam_update` 与 `torch.optim.Adam` 逐步对拍，并验证 AdamW 的解耦写法：

```python
def adamw_update(params, grads, v, s, t, lr, wd, beta1=0.9, beta2=0.999, eps=1e-8):
    for k in params:
        params[k] = params[k] * (1.0 - lr * wd)          # 解耦衰减：先按比例收缩
    return adam_update(params, grads, v, s, t, lr, beta1, beta2, eps)

def compare(torch_opt_cls, ours, steps=50, **kw):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5)); A = A @ A.T + np.eye(5)     # 正定二次型 f = ½ wᵀAw
    w0 = rng.standard_normal(5)
    wt = torch.tensor(w0, requires_grad=True)
    opt = torch_opt_cls([wt], **kw)
    p = {"W": w0.copy()}; v, s, t = adam_init(p)
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * wt @ torch.tensor(A) @ wt).backward()
        opt.step()
        g = {"dW": A @ p["W"]}
        p, v, s, t = ours(p, g, v, s, t)
    return np.abs(p["W"] - wt.detach().numpy()).max()

print(compare(torch.optim.Adam, lambda p, g, v, s, t: adam_update(p, g, v, s, t, lr=0.1),
              lr=0.1))                                          # 约 1e-16
print(compare(torch.optim.AdamW, lambda p, g, v, s, t: adamw_update(p, g, v, s, t, lr=0.1, wd=0.05),
              lr=0.1, weight_decay=0.05))                       # 约 1e-16
```

PyTorch 内部先算分母 $\sqrt{s}/\sqrt{1-\beta_2^t}+\varepsilon$（即 $\sqrt{\hat s}+\varepsilon$），再用步长 $\alpha/(1-\beta_1^t)$ 乘 $v$，与上面的公式逐项相同；`AdamW` 则在这一步之前先执行 $w\leftarrow(1-\alpha\lambda)w$，与 `adamw_update` 的顺序一致。

### 选哪个

> [!EXAMPLE]+ 例7 MNIST 上四种方法的比较（鱼书 6.1.8 节）
>
> 5 层网络，每层 100 个 ReLU 神经元。Momentum、AdaGrad、Adam 的损失下降速度大致相同，都明显快于 SGD；细看 AdaGrad 稍快一点。结果会随学习率等超参数和网络结构而变化，但一般而言，后三者比 SGD 学得快，有时最终精度也更高。

- 不存在在所有问题上都最好的方法，各自有擅长和不擅长的问题。
- AdaGrad：历史平方梯度一直累加，学习率单调下降，后期可能过早停止。
- RMSprop：用滑动平均避免 AdaGrad 的过早停止。
- Adam = Momentum + RMSprop，但不是“一定最好”。视觉分类上，调好的 SGD + Momentum 仍常与 Adam 打平或更好。
- Adam 对初始 $\alpha$ 更宽容，适合作业与项目起步。

### 学习率衰减

训练初期离最优点远，可以走大步；后期接近最优点，大步会在最小值附近来回跳，需要减小步长。常见的衰减方式：

$$
\alpha_t=\frac{\alpha_0}{1+\delta\,\mathrm{epoch}},\qquad
\alpha_t=\alpha_0\cdot 0.95^{\mathrm{epoch}},\qquad
\alpha_t=\frac{k\alpha_0}{\sqrt{t}}.
$$

第一种是反比衰减（$\delta$ 为衰减率），第二种是指数衰减，第三种按迭代次数的平方根衰减。另外还有阶梯式衰减和手动衰减。

```python
def lr_inverse(alpha0, epoch, decay_rate=1.0):
    return alpha0 / (1.0 + decay_rate * epoch)

def lr_exponential(alpha0, epoch, base=0.95):
    return alpha0 * base ** epoch

def lr_step(alpha0, epoch, decay_rate=1.0, interval=1000):
    """C2 作业中的 schedule_lr_decay：每 interval 个 epoch 才衰减一次。"""
    return alpha0 / (1.0 + decay_rate * np.floor(epoch / interval))

def lr_warmup_cosine(alpha0, step, warmup, total):
    if step < warmup:
        return alpha0 * (step + 1) / warmup                  # 线性升温
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * alpha0 * (1.0 + np.cos(np.pi * progress))  # 余弦降到 0
```

PyTorch 用调度器（scheduler）包住优化器，每个 epoch（或每步）调用一次 `step()`：

```python
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, CosineAnnealingLR

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = LambdaLR(opt, lr_lambda=lambda epoch: 1.0 / (1.0 + 1.0 * epoch))   # 反比衰减
# sched = ExponentialLR(opt, gamma=0.95)                                   # 指数衰减
# sched = CosineAnnealingLR(opt, T_max=num_epochs)                         # 余弦退火

for epoch in range(num_epochs):
    for x, y in train_loader:
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    sched.step()                              # 先 opt.step() 再 sched.step()
    print(epoch, sched.get_last_lr())
```

- Transformer 中更常见：余弦退火、线性 warmup 后衰减。
- $\alpha$ 过大：损失爆炸或震荡；过小：下降极慢，易停在尖锐区域。
- 一个实用做法是先找使损失开始发散的 $\alpha$，再取它的 $1/3$ 到 $1/10$。
- 损失仍陡降时不要过早衰减 $\alpha$；先用常数 $\alpha$ 观察到平台，再衰减。
- $\alpha$ 衰减与 early stopping 同时打开时，dev 变好可能只是步长变小，不一定是泛化改善。

### 局部最优与鞍点

- 低维直觉里，优化最怕陷入局部最优。但在高维空间中，梯度为零的点要成为局部最小值，需要所有方向都向上弯，这概率极低；绝大多数梯度为零的点是**鞍点**（有的方向向上、有的向下）。
- 真正拖慢训练的是**平台**（plateau）：梯度长时间接近 0 的大片区域。Momentum、Adam 等方法和 mini-batch 的随机噪声都有助于更快地离开平坦区。
- Batch 过小：噪声过大；过大：泛化有时变差。要与学习率线性缩放等经验规则一起调。

<br>

> [!NOTE]+ 衔接
>
> 优化器决定了参数怎么走，但每层激活的分布仍会随前面层的参数更新而漂移，使得对初始化和学习率都很敏感。下一节用 BatchNorm 把每层激活的分布强制拉回合适的范围。

<br>

## 批归一化（Batch Normalization）

### 动机

初始化一节说明：各层激活值的分布有适当的广度，训练才能顺利。初始化只能保证**开始时**如此；训练开始后，参数不断变化，各层分布也跟着变。BatchNorm 的思路很直接：既然需要合适的分布，就在每层**强制**调整一次。

> [!INFO]+ BatchNorm
>
> BatchNorm（Ioffe & Szegedy, 2015）对当前 batch 的预激活做标准化，再乘 $\gamma$、加 $\beta$。没有 $\gamma,\beta$，一层只能输出近似零均值、单位方差的量。训练用当批统计，推理改用训练期累积的 running mean / variance，并仍然使用 $\gamma,\beta$。batch 统计带噪声，有轻微正则作用，但不能代替 Dropout 或 $L_2$。

<br>

论文给出的动机是**内部协变量偏移**（internal covariate shift）：下层参数一更新，上层看到的输入分布就跟着变，上层要不断适应新分布。BatchNorm 用标准化把每层的尺度固定下来，学习率因而可以更大。论文把这当作动机；实践上更常观察到的是损失曲面变得更平滑好走、对初始化不那么敏感。

BatchNorm 的三个优点：

- 学习更快（可以使用更大的学习率）；
- 不那么依赖初始值；
- 抑制过拟合（降低 Dropout 等的必要性）。

### 算法

对当前 mini-batch 的每个特征维（全连接：每个神经元；卷积：每个通道），先标准化为均值 0、方差 1：

$$
\mu=\frac{1}{n_b}\sum_{i}z^{(i)},\qquad
\sigma^{2}=\frac{1}{n_b}\sum_{i}(z^{(i)}-\mu)^{2},\qquad
\tilde{z}=\frac{z-\mu}{\sqrt{\sigma^{2}+\varepsilon}}.
$$

$\varepsilon$ 是防止除零的小常数。然后做可学习的缩放和平移：

$$
z_{\mathrm{BN}}=\gamma\odot\tilde{z}+\beta.
$$

- $\gamma,\beta$ 可学习，初始 $\gamma=1$、$\beta=0$。
- 为什么需要 $\gamma,\beta$：强制均值 0、方差 1 不一定是最好的分布（例如 sigmoid 在 0 附近近似线性，非线性就浪费了）。若 $\gamma=\sqrt{\sigma^{2}+\varepsilon}$，$\beta=\mu$，可以还原为恒等映射，所以 BatchNorm 不损失一层的表示能力，由网络自己决定要多大的均值和方差。
- 加入 BatchNorm 后，$b$ 与 $\beta$ 冗余（减均值会把 $b$ 抵消），实现中常省略 $b$。
- 位置：通常插在 Affine 之后、激活函数之前，即 Affine → BatchNorm → ReLU。也有放在激活之后的做法，文献中都有讨论。

### 训练与推理

- **训练**：用当前 batch 的 $\mu,\sigma^2$。
- **推理**：可能一次只来一个样本，无法计算 batch 统计（batch 大小为 1 时 $\sigma^{2}=0$）。因此训练时同时维护运行平均，推理时使用它：

$$
\bar{\mu}\leftarrow \eta\bar{\mu}+(1-\eta)\mu_{\mathrm{batch}},
$$

$\eta$ 如 $0.9$，$\bar{\sigma}^2$ 同理。评估模式必须切换到运行统计量，否则行为与训练不一致。

> [!INFO]+ BatchNorm 的反向要点
>
> $\partial J/\partial z$ 不能只通过 $\gamma/\sqrt{\sigma^{2}+\varepsilon}$ 缩放，因为 $\mu$ 与 $\sigma^{2}$ 也依赖 batch 内每个 $z^{(i)}$。完整 Jacobian 含三项：经 $\tilde{z}$ 的直接路径、经 $\mu$ 的路径、经 $\sigma^{2}$ 的路径。框架会自动求导；手写时需对这三项求和。$\gamma,\beta$ 的梯度是 $\partial J/\partial z_{\mathrm{BN}}$ 分别与 $\tilde{z}$、$1$ 的 batch 求和。画出计算图再逐节点反向推导会清楚得多（参见 Frederik Kratzert 的博客 *Understanding the backward pass through Batch Normalization Layer*）。

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

上面的反向可以化简成一行。推导只看一个特征维（一行），batch 内有 $m$ 个样本，记 $\tilde\sigma=\sqrt{\sigma^2+\varepsilon}$，$\hat z_i=(z_i-\mu)/\tilde\sigma$。

第一步，两个统计量对每个输入的导数：

$$
\frac{\partial\mu}{\partial z_j}=\frac1m,\qquad
\frac{\partial\sigma^2}{\partial z_j}=\frac2m(z_j-\mu),\qquad
\frac{\partial\tilde\sigma}{\partial z_j}=\frac{1}{2\tilde\sigma}\cdot\frac2m(z_j-\mu)=\frac{\hat z_j}{m}.
$$

求 $\partial\sigma^2/\partial z_j$ 时，$\mu$ 也依赖 $z_j$，但那一项是 $-\frac2m\sum_i(z_i-\mu)\cdot\frac1m=0$，因为偏差之和为零。

第二步，$\hat z_i$ 对 $z_j$ 的导数（商的求导法则，分子分母都依赖 $z_j$）：

$$
\frac{\partial\hat z_i}{\partial z_j}
=\frac{1}{\tilde\sigma}\Big(\delta_{ij}-\frac1m\Big)-\frac{z_i-\mu}{\tilde\sigma^2}\cdot\frac{\hat z_j}{m}
=\frac{1}{\tilde\sigma}\Big(\delta_{ij}-\frac1m-\frac{\hat z_i\hat z_j}{m}\Big).
$$

三项分别对应三条路径：直接路径 $\delta_{ij}$，经过 $\mu$ 的路径 $-\frac1m$，经过 $\sigma$ 的路径 $-\frac{\hat z_i\hat z_j}{m}$。

第三步，链式法则，$\mathrm d\hat z_i=\gamma\,\mathrm dy_i$，对 $i$ 求和：

$$
\mathrm dz_j=\sum_i\mathrm d\hat z_i\frac{\partial\hat z_i}{\partial z_j}
=\frac{1}{\tilde\sigma}\Big(\mathrm d\hat z_j-\frac1m\sum_i\mathrm d\hat z_i-\hat z_j\cdot\frac1m\sum_i\mathrm d\hat z_i\hat z_i\Big)
=\frac{\gamma}{m\tilde\sigma}\Big(m\,\mathrm dy_j-\sum_i\mathrm dy_i-\hat z_j\sum_i\mathrm dy_i\hat z_i\Big).
$$

括号里是上游梯度减去它的均值分量、再减去它在 $\hat z$ 方向上的分量。所以 $\mathrm dz$ 满足 $\sum_j\mathrm dz_j=0$ 和 $\sum_j\mathrm dz_j\hat z_j=0$。这有直观的原因：把所有 $z_j$ 加同一个常数，或整体乘一个正数，BN 的输出不变，所以梯度在这两个方向上的分量必然为零。$\sum_i\mathrm dy_i$ 与 $\sum_i\mathrm dy_i\hat z_i$ 正好是 $\mathrm d\beta$ 与 $\mathrm d\gamma$，可以直接复用。写成矩阵形式：

$$
\mathrm{d}Z=\frac{\gamma}{m\sqrt{\sigma^{2}+\varepsilon}}\Big(m\,\mathrm{d}\hat{Z}'-\textstyle\sum_j \mathrm{d}\hat{Z}'_{\cdot j}-\hat{Z}\odot\sum_j\big(\mathrm{d}\hat{Z}'\odot\hat{Z}\big)_{\cdot j}\Big),
$$

其中 $\mathrm{d}\hat{Z}'$ 即 `d_out`，求和沿 batch 维：

```python
def batchnorm_backward_compact(d_out, cache):
    Z_hat, var, gamma, eps = cache["Z_hat"], cache["var"], cache["gamma"], cache["eps"]
    m = Z_hat.shape[1]
    dgamma = (d_out * Z_hat).sum(axis=1, keepdims=True)
    dbeta = d_out.sum(axis=1, keepdims=True)
    dZ = gamma / (m * np.sqrt(var + eps)) * (
        m * d_out - dbeta - Z_hat * dgamma
    )
    return dZ, dgamma, dbeta
```

PyTorch 的 `nn.BatchNorm1d` 按**行样本**沿第 0 维求统计量，`model.train()` / `model.eval()` 决定用 batch 统计还是运行平均：

```python
import torch.nn as nn

bn = nn.BatchNorm1d(num_features=20, eps=1e-5, momentum=0.1)
# PyTorch 的 momentum 指“新值所占比例”：running = (1-0.1)*running + 0.1*batch，
# 对应上面 NumPy 代码里的 momentum=0.9。

block = nn.Sequential(nn.Linear(64, 20, bias=False),   # BN 会抵消 b，故省略
                      nn.BatchNorm1d(20),
                      nn.ReLU())

block.train()
out = block(torch.randn(32, 64))       # 用这 32 个样本的均值方差，并更新 running_*
block.eval()
out = block(torch.randn(1, 64))        # 用 running_mean / running_var，单样本也能推理
print(block[1].running_mean.shape, block[1].weight.shape)   # γ 即 weight，β 即 bias
```

- PyTorch 训练时用有偏方差（除以 $m$）做标准化，但更新 `running_var` 时用无偏方差（除以 $m-1$）。与自己的 NumPy 实现对拍时，要注意这一点差异。
- 忘记 `model.eval()` 是推理结果随 batch 组成而变的最常见原因。

> [!EXAMPLE]+ 例8 BatchNorm 的效果（鱼书 6.3.2 节）
>
> 在 MNIST 上比较有无 BatchNorm 的训练准确率：使用 BatchNorm 后学习明显更快。再把权重初始标准差从 $1.0$ 到 $0.0001$ 取 16 个值分别实验：几乎所有情况下有 BatchNorm 都学得更快；而没有 BatchNorm 时，初始尺度不合适，学习完全无法进行。这直接体现了“对初始值健壮”。

### 局限与替代

- 效果：学习率可以更大；对初始化不那么敏感；本身带轻微正则（batch 统计带噪声）。
- 小 batch 时统计不可靠：改用 LayerNorm（layer normalization）、GroupNorm，或增大 batch。
- RNN 中 BatchNorm 不如 LayerNorm 常见。
- 稳定深层训练的其他工具：softmax 温度、梯度裁剪（gradient clipping）、残差连接与归一化。

> [!NOTE]+ 对照：BatchNorm 与 LayerNorm
>
> - BatchNorm：沿 batch 维求统计量；训练用 batch 统计，推理用运行平均。
> - LayerNorm：沿特征维求统计量（每个样本自己归一化）；训练与推理公式相同。
> - 全连接与卷积网络常用 BatchNorm；RNN 与 Transformer 常用 LayerNorm。
> - 小 batch 时 BatchNorm 统计不可靠。

<br>

### 归一化家族

#### 统一的写法

BatchNorm、LayerNorm、InstanceNorm、GroupNorm 的公式完全相同，**唯一的区别是在哪些维度上求均值和方差**。以卷积特征图 $x\in\mathbb R^{N\times C\times H\times W}$（batch、通道、高、宽）为例，选定一个维度集合 $\mathcal S$：

$$
\mu=\frac{1}{\lvert\mathcal S\rvert}\sum_{\mathcal S}x,\qquad
\sigma^2=\frac{1}{\lvert\mathcal S\rvert}\sum_{\mathcal S}(x-\mu)^2,\qquad
y=\gamma_c\,\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta_c.
$$

| 方法 | 求统计的维度 $\mathcal S$ | 每组统计用多少个数 | 依赖 batch | 训练与推理相同 | 典型场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BatchNorm | $N,H,W$（每个通道一组） | $NHW$ | 是 | 否，推理用运行平均 | CNN，batch 较大 |
| LayerNorm | $C,H,W$（每个样本一组） | $CHW$ | 否 | 是 | Transformer、RNN |
| InstanceNorm | $H,W$（每个样本的每个通道一组） | $HW$ | 否 | 是 | 风格迁移、图像生成 |
| GroupNorm | 每个样本内，$C/G$ 个通道一组，再加 $H,W$ | $\frac{C}{G}HW$ | 否 | 是 | 检测、分割等小 batch 任务 |

GroupNorm 是 LN 与 IN 之间的连续谱：$G=1$ 时所有通道一组，就是 LayerNorm；$G=C$ 时每个通道一组，就是 InstanceNorm。缩放参数 $\gamma,\beta$ 在四种方法中都是**每个通道一个**，共 $2C$ 个参数。

Transformer 里的张量是 $(N,T,d)$（batch、序列位置、特征），LayerNorm 只在最后一维 $d$ 上求统计：**每个 token 自己归一化**，与 batch 中其他句子、同一句的其他位置都无关。

> [!EXAMPLE]+ 同一个矩阵上的 BN 与 LN
>
> 2 个样本、3 个特征（行是样本），第二个样本恰好是第一个的 3 倍：
>
> $$
> X=\begin{pmatrix}1&2&3\\3&6&9\end{pmatrix}.
> $$
>
> **BatchNorm 按列**（每个特征在 2 个样本上求统计）。第 1 列均值 2、标准差 1；第 2 列均值 4、标准差 2；第 3 列均值 6、标准差 3。结果
>
> $$
> \mathrm{BN}(X)=\begin{pmatrix}-1&-1&-1\\1&1&1\end{pmatrix}.
> $$
>
> 各特征原本的尺度差异（$1,2,3$ 与 $3,6,9$）被抹掉，剩下的只是“这个样本在这个特征上比 batch 平均大还是小”。
>
> **LayerNorm 按行**（每个样本在 3 个特征上求统计）。第 1 行均值 2、方差 $2/3$；第 2 行均值 6、方差 $6$。结果
>
> $$
> \mathrm{LN}(X)=\begin{pmatrix}-1.22&0&1.22\\-1.22&0&1.22\end{pmatrix}.
> $$
>
> 两个样本的输出完全相同：LN 去掉了“整体有多大”，只保留“各特征之间的相对模式”。第二个样本是第一个的 3 倍，这个信息被 LN 丢掉了，被 BN 保留了（体现为两行符号相反）。选哪种归一化，就是在选“哪种差异对任务无关、可以丢掉”。

#### 为什么各有各的位置

**BatchNorm**。统计量来自 $NHW$ 个数，在 CNN 中通常有几万个，估计准，带来的噪声恰好起到正则作用。它的问题都来自对 batch 的依赖：

- batch 很小（每张卡 1 到 2 张图，检测和分割中常见）时，均值方差的估计噪声太大，精度明显下降；
- 训练用 batch 统计、推理用运行平均，两者不一致；训练数据和测试数据分布不同时更明显；
- 多卡训练时每张卡各算各的统计，需要 `SyncBatchNorm` 才能用到全局 batch；
- RNN 每个时间步的统计不同，变长序列难以处理。

**LayerNorm**。每个样本、每个 token 自己算，与 batch 大小、序列长度都无关，训练和推理完全一样，所以成了 Transformer 和 RNN 的标准选择。用在 CNN 上效果一般：它把所有通道放在一起归一化，假设各通道尺度相近，而卷积网络不同通道的激活尺度往往差别很大。

**InstanceNorm**。对每张图的每个通道单独去均值、除标准差。风格迁移的研究发现，一个通道在空间上的均值和方差主要编码了“风格”（对比度、色调、笔触强弱），去掉它们就去掉了原图的风格，生成网络更容易换上新风格。AdaIN 把这一点直接写成公式：$\mathrm{AdaIN}(x,y)=\sigma(y)\frac{x-\mu(x)}{\sigma(x)}+\mu(y)$，用风格图 $y$ 的统计量替换内容图 $x$ 的统计量。分类任务中它会丢掉有用的全局信息，一般不用。

**GroupNorm**（Wu & He, 2018）。把通道分组（默认 $G=32$），每组内的通道往往编码相关的特征（例如不同方向的边缘），一起归一化比 LN 的“全部一起”更合理，又比 IN 的“各自为政”保留更多信息。它不依赖 batch：在 batch 为 2 时，ResNet-50 的 ImageNet 错误率比 BN 低约 10 个百分点；batch 大时与 BN 相当。

#### RMSNorm

LayerNorm 做两件事：减均值（re-centering）和除尺度（re-scaling）。Zhang & Sennrich（2019）发现起作用的主要是后者，于是去掉减均值，也去掉 $\beta$：

$$
\mathrm{RMS}(x)=\sqrt{\frac1d\sum_{j=1}^{d}x_j^2+\varepsilon},\qquad
y=\gamma\odot\frac{x}{\mathrm{RMS}(x)}.
$$

- 少算一次均值、少一组参数，在大模型中能省下可观的时间；
- 当 $x$ 的均值恰好为 $0$ 时，RMS 就是标准差，RMSNorm 与 LayerNorm 相同；
- LLaMA、T5 等大语言模型使用 RMSNorm。

反向推导与 BN 类似，记 $r=\mathrm{RMS}(x)$、$u=\gamma\odot\mathrm dy$，用 $\partial r/\partial x_j=x_j/(dr)$：

$$
\mathrm dx_j=\frac{u_j}{r}-\frac{x_j}{d\,r^3}\sum_i u_ix_i.
$$

与 BN 的结果对照：少了“减去均值分量”那一项（因为前向没有减均值），只剩“减去沿 $x$ 方向的分量”，所以 $\sum_j\mathrm dx_j x_j\approx 0$：把 $x$ 整体放大，输出不变，梯度在这个方向上为零。

#### 归一化权重而不是激活

另一类方法不碰激活，直接约束权重：

- **Weight Normalization**：把权重向量写成 $w=g\,\frac{v}{\lVert v\rVert}$，方向 $v/\lVert v\rVert$ 与长度 $g$ 分开学习。没有 batch 依赖，也没有运行统计，但通常不如 BN 稳。
- **Spectral Normalization**（Miyato et al., 2018）：用 $W/\sigma_{\max}(W)$ 代替 $W$，$\sigma_{\max}$ 是最大奇异值。这样每一层都是 1-Lipschitz 的：输入变化 $\delta$，输出变化不超过 $\lVert\delta\rVert$。GAN 的判别器常用它防止梯度过大、训练发散。$\sigma_{\max}$ 不需要做完整 SVD，用幂迭代每步迭代一次即可。

#### 尺度不变性与有效学习率

BN 带来一个容易被忽略的性质。设一层权重为 $W$，后面接 BN。把 $W$ 乘以任意正数 $c$，预激活整体乘 $c$，BN 减均值除标准差后输出不变：

$$
\mathrm{BN}(cWx)=\mathrm{BN}(Wx),\qquad\text{所以}\quad L(cW)=L(W).
$$

两边对 $W$ 求导：$c\,\nabla L(cW)=\nabla L(W)$，即 $\nabla L(cW)=\frac1c\nabla L(W)$：**权重越大，梯度越小**。另外，对 $L(cW)=L(W)$ 关于 $c$ 求导并令 $c=1$，得 $\langle\nabla L(W),W\rangle=0$，梯度总是与 $W$ 正交。

这两条合起来有两个推论：

1. SGD 的一步 $W\leftarrow W-\alpha\nabla L$ 沿正交方向走，由勾股定理 $\lVert W\rVert$ 只增不减。权重越来越大，梯度越来越小，真正起作用的角度变化约为 $\alpha\lVert\nabla L\rVert/\lVert W\rVert\propto\alpha/\lVert W\rVert^2$，**有效学习率随训练自动下降**。
2. 对 BN 前面的权重做权重衰减，几乎不改变函数（尺度不影响输出），主要作用是让 $\lVert W\rVert$ 保持较小，**维持有效学习率**。所以在 BN 网络里，权重衰减的作用更像调学习率，而不是传统意义上的正则。这也是 BN 网络中 $\lambda$ 与 $\alpha$ 需要一起调的原因。

#### 部署时把 BN 折叠进卷积

推理时 BN 用的是固定的 $\bar\mu,\bar\sigma^2$，它就是一个逐通道的仿射变换，可以并入前面的卷积或全连接层。对输出通道 $c$：

$$
\gamma_c\frac{(W_c*x+b_c)-\bar\mu_c}{\sqrt{\bar\sigma_c^2+\varepsilon}}+\beta_c
=\underbrace{\frac{\gamma_c}{\sqrt{\bar\sigma_c^2+\varepsilon}}W_c}_{W_c'}*x+\underbrace{\frac{\gamma_c(b_c-\bar\mu_c)}{\sqrt{\bar\sigma_c^2+\varepsilon}}+\beta_c}_{b_c'}.
$$

折叠后网络少了一层，推理更快，结果在浮点误差内不变。移动端部署框架都会自动做这一步。LN 和 GN 的统计量依赖每个输入，不能这样折叠。

#### 放在哪里

- CNN：`Conv → BN → ReLU`，BN 前面的卷积不需要偏置；残差块中，最后一个 BN 的 $\gamma$ 初始化为 $0$，可以让每个残差块一开始等于恒等映射，深层网络更容易训练。
- Transformer：Post-LN 写成 $x_{l+1}=\mathrm{LN}(x_l+F(x_l))$，原论文结构；Pre-LN 写成 $x_{l+1}=x_l+F(\mathrm{LN}(x_l))$。Pre-LN 展开后 $x_L=x_0+\sum_{l}F_l(\cdot)$，从输出到任何一层都有一条不经过 LN 的恒等路径，$\partial x_L/\partial x_l$ 中总有一项单位阵，深层时梯度更稳，可以少用甚至不用 warmup；Post-LN 每一层的梯度都要穿过 LN，层数多时需要仔细的 warmup。现代大模型几乎都用 Pre-LN（常配 RMSNorm），最后再加一个 LN。

#### 如何选择

| 场景 | 推荐 |
| :--- | :--- |
| CNN 分类，每卡 batch $\ge16$ | BatchNorm |
| 检测、分割、视频，每卡 batch 很小 | GroupNorm（$G=32$），或 SyncBatchNorm |
| Transformer、RNN、变长序列 | LayerNorm；大模型用 RMSNorm |
| 风格迁移、图像生成器 | InstanceNorm / AdaIN |
| GAN 判别器 | Spectral Normalization |
| 不想要训练与推理的差异 | 任何不依赖 batch 的方法（LN、GN、RMSNorm） |

#### 实现与核对

一个函数实现四种归一化，只改求统计的轴；再分别和 PyTorch 对拍：

```python
def normalize(x, axes, gamma, beta, eps=1e-5):
    """x: (N, C, H, W)。gamma, beta: (C,)，按通道缩放平移。"""
    mu = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma[None, :, None, None] * x_hat + beta[None, :, None, None]

def group_norm(x, G, gamma, beta, eps=1e-5):
    N, C, H, W = x.shape
    xg = x.reshape(N, G, C // G, H, W)                    # 把通道拆成 G 组
    mu = xg.mean(axis=(2, 3, 4), keepdims=True)
    var = xg.var(axis=(2, 3, 4), keepdims=True)
    x_hat = ((xg - mu) / np.sqrt(var + eps)).reshape(N, C, H, W)
    return gamma[None, :, None, None] * x_hat + beta[None, :, None, None]

def rms_norm(x, gamma, eps=1e-6):
    """x: (..., d)，在最后一维上归一化。"""
    r = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * x / r, r

def rms_norm_backward(dy, x, gamma, r):
    d = x.shape[-1]
    u = gamma * dy
    return u / r - x * np.sum(u * x, axis=-1, keepdims=True) / (d * r ** 3)

rng = np.random.default_rng(0)
N, C, H, W = 4, 8, 5, 5
x = rng.standard_normal((N, C, H, W)) * 3 + 1
g, b = rng.standard_normal(C), rng.standard_normal(C)
xt, gt, bt = (torch.tensor(a) for a in (x, g, b))

ours = {
    "BN": normalize(x, (0, 2, 3), g, b),
    "LN": normalize(x, (1, 2, 3), g, b),
    "IN": normalize(x, (2, 3), g, b),
    "GN": group_norm(x, 2, g, b),
}
theirs = {
    "BN": F.batch_norm(xt, None, None, gt, bt, training=True),
    "LN": F.layer_norm(xt, (C, H, W)) * gt[:, None, None] + bt[:, None, None],
    "IN": F.instance_norm(xt, weight=gt, bias=bt),
    "GN": F.group_norm(xt, 2, gt, bt),
}
for k in ours:
    print(k, np.abs(ours[k] - theirs[k].numpy()).max())          # 都约 1e-15

# RMSNorm：前向与 nn.RMSNorm 对拍，反向与 autograd 对拍
d = 16
x2 = rng.standard_normal((3, d)); g2 = rng.standard_normal(d); dy = rng.standard_normal((3, d))
y2, r = rms_norm(x2, g2)
rms = nn.RMSNorm(d, eps=1e-6).double()
with torch.no_grad():
    rms.weight.copy_(torch.tensor(g2))
x2t = torch.tensor(x2, requires_grad=True)
yt = rms(x2t); (yt * torch.tensor(dy)).sum().backward()
print("RMS fwd", np.abs(y2 - yt.detach().numpy()).max(),
      "bwd", np.abs(rms_norm_backward(dy, x2, g2, r) - x2t.grad.numpy()).max())
```

LayerNorm 在 PyTorch 中的 `weight` 形状是 `normalized_shape`，即 $(C,H,W)$ 每个位置一个，而不是每个通道一个；上面为了与其他三种比较，用逐通道的 $\gamma$ 在外面手动缩放。Transformer 中 `nn.LayerNorm(d)` 的 $\gamma$ 是 $d$ 维，两种说法在那里一致。

BN 折叠与谱归一化的幂迭代：

```python
def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std                                         # γ / σ̃，每个输出通道一个
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, bias=True)
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale[:, None, None, None])
        fused.bias.copy_((b - bn.running_mean) * scale + bn.bias)
    return fused

conv, bn = nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8)
bn.train(); [bn(conv(torch.randn(16, 3, 8, 8))) for _ in range(20)]    # 积累运行统计
bn.eval()
xin = torch.randn(2, 3, 8, 8)
print((bn(conv(xin)) - fold_bn(conv, bn)(xin)).abs().max().item())       # 约 1e-7

def spectral_norm_power(W, iters=50, rng=None):
    rng = rng or np.random.default_rng(0)
    u = rng.standard_normal(W.shape[0])
    for _ in range(iters):
        v = W.T @ u; v /= np.linalg.norm(v)
        u = W @ v;   u /= np.linalg.norm(u)
    return u @ W @ v                                                # ≈ 最大奇异值

Wm = rng.standard_normal((64, 32))
print(spectral_norm_power(Wm), np.linalg.norm(Wm, 2))               # 12.61189 与 12.61191，迭代越多越接近
```

PyTorch 中谱归一化是 `nn.utils.parametrizations.spectral_norm(layer)`，每次前向自动做一步幂迭代并把权重除以 $\sigma_{\max}$。

<br>

> [!NOTE]+ 衔接
>
> 初始化、优化器和 BatchNorm 一起保证训练误差能降下来。但训练误差低不等于模型好：若开发集误差明显更高，模型只是记住了训练集。下一节讨论如何诊断并缩小这个差距。

<br>

## 正则化（Regularization）

### 问题

机器学习的目标是**泛化**：对训练时没见过的数据也能预测正确。只能拟合训练数据、不能拟合其他数据的状态称为**过拟合**（overfitting）。过拟合主要有两个原因：

- 模型参数多、表现力强；
- 训练数据少。

正则化用来缩小训练误差与开发集误差的差距：在损失上加惩罚，或在前向中注入噪声，使模型少拟合训练集中的偶然细节。它不管损失降不降得下去，那是优化的事。正则改的是损失或计算图，优化改的是参数沿梯度怎么走，两件事不要混在一次实验里。

> [!EXAMPLE]+ 例9 人为制造过拟合（鱼书 6.4.1 节）
>
> 从 MNIST 的 6 万个训练样本中只取 300 个，用 7 层网络（每层 100 个 ReLU 神经元）、学习率 0.01 的 SGD 训练 200 个 epoch。约 100 个 epoch 后，训练集准确率几乎达到 100%，但测试集准确率与之相差很大——模型只拟合了训练数据。后面的权值衰减和 Dropout 都以这个实验为对照。

### 偏差与方差

先诊断再开药。以可达到的最优误差（Bayes 误差，或用人类水平近似）为基准：

- **偏差（bias / underfitting）**：训练误差明显高于最优误差，模型连训练集都没学好。
- **方差（variance / overfitting）**：开发集误差明显高于训练误差，模型学到了训练集特有的细节。
- **数据不匹配**：开发集分布与训练分布不同，即使方差小也会差（见“项目策略”）。

CS230 的实用判据（分类误差，最优误差接近 0 时）：

| 训练误差 | Dev 误差 | 诊断 |
| :--- | :--- | :--- |
| 高 | 高且接近训练 | 偏差：更大网络、更久训练、更好优化、换特征 |
| 低 | 高 | 方差：更多数据、正则、简化模型 |
| 高 | 更高 | 偏差与方差同时存在 |
| 低 | 低 | 接着核对 test 与部署分布是否仍一致 |

- “更大网络一定过拟合”并不成立。足够大的网络配合正则与早停（early stopping），常常同时降低训练与 dev 误差。
- 顺序是：先保证模型能装下训练数据（降偏差），再压方差。

### $L_2$ 权重衰减（weight decay）

> [!INFO]+ $L_2$
>
> $L_2$ 正则在损失上加 $\|W\|_F^{2}$。梯度多一项 $\lambda W/m$，更新等价于每次把 $W$ 乘以略小于 1 的因子，所以也叫权重衰减。偏置通常不惩罚。$\lambda$ 过大会欠拟合。

<br>

很多过拟合是因为权重取值过大。权重衰减在损失上加一项惩罚大权重：

$$
J_{\mathrm{reg}}=J+\frac{\lambda}{2m}\sum_{l}\|W^{[l]}\|_F^{2}.
$$

$\|W\|_F^{2}=\sum_{i,j}W_{ij}^{2}$（Frobenius 范数，即所有元素的平方和）。$\lambda$ 控制正则强度，越大对大权重的惩罚越重；系数 $\frac{1}{2}$ 是为了让求导结果干净。偏置通常不惩罚。

梯度多一项 $\lambda W^{[l]}/m$，更新等价于

$$
W\leftarrow\left(1-\frac{\alpha\lambda}{m}\right)W-\alpha\,\mathrm{d}W_{\mathrm{data}}.
$$

- 权重每次先被乘以一个略小于 1 的因子，再做普通梯度更新，故称 weight decay。
- 为什么能抑制过拟合：权重变小后，激活更可能落在 tanh / sigmoid 的近似线性区，每层更接近线性，整个网络的有效复杂度降低。
- $\lambda$ 过大则欠拟合。
- 范数的其他选择：$L_1$ 范数（各元素绝对值之和 $|w_1|+\dots+|w_n|$）倾向产生稀疏权重；$L_\infty$ 范数（Max 范数，取绝对值最大的元素）。CS230 作业以 $L_2$ 为主。

CS230 速查表把权重正则的三种常见写法并列（$\theta$ 为全部权重，$\lambda\ge 0$）：

| 名称 | 加到损失上的项 | 效果 |
| :--- | :--- | :--- |
| LASSO（$L_1$） | $\lambda\lVert\theta\rVert_1$ | 把一部分系数压到恰好为 $0$，相当于做特征选择 |
| Ridge（$L_2$） | $\lambda\lVert\theta\rVert_2^{2}$ | 所有系数整体缩小，但一般不为 $0$ |
| Elastic Net | $\lambda\big[(1-\alpha)\lVert\theta\rVert_1+\alpha\lVert\theta\rVert_2^{2}\big]$，$\alpha\in[0,1]$ | 在“稀疏”与“整体缩小”之间折中 |

$L_1$ 产生稀疏的原因在于梯度：$\partial|\theta_j|/\partial\theta_j=\mathrm{sign}(\theta_j)$，无论 $\theta_j$ 多小，都以固定大小把它往 $0$ 推；$L_2$ 的梯度 $2\theta_j$ 随 $\theta_j$ 变小而变小，只会让它越来越接近 $0$，却很少恰好等于 $0$。深度网络中 $L_1$ 用得少，因为稀疏权重在 GPU 上并不能自动变快；需要压缩模型时更常用剪枝或蒸馏。

课程中出现的各种范数及其用途：

| 范数 | 定义 | 梯度（次梯度） | 课程中的用途 |
| :--- | :--- | :--- | :--- |
| $L_1$ | $\sum_j\lvert w_j\rvert$ | $\mathrm{sign}(w)$ | 稀疏正则、MAE |
| $L_2$ | $\sqrt{\sum_j w_j^2}$ | $w/\lVert w\rVert_2$ | 梯度裁剪、嵌入归一化、triplet 距离 |
| $L_2^2$ | $\sum_j w_j^2$ | $2w$ | 权重衰减、MSE |
| $L_\infty$ | $\max_j\lvert w_j\rvert$ | 只在最大的分量上为 $\pm1$ | 对抗扰动的约束球（FGSM、PGD） |
| Frobenius | $\sqrt{\sum_{ij}W_{ij}^2}$ | $W/\lVert W\rVert_F$ | 矩阵的 $L_2$，权重衰减 |
| 谱范数 | $\sigma_{\max}(W)$ | $u_1v_1^{\top}$（最大奇异向量） | 谱归一化、Lipschitz 常数 |

梯度裁剪（gradient clipping）用的就是 $L_2$ 范数：所有参数的梯度拼成一个长向量，若总范数超过阈值 $c$，整体按比例缩小到 $c$，方向不变。

$$
g\leftarrow g\cdot\min\Big(1,\frac{c}{\lVert g\rVert_2}\Big).
$$

```python
def clip_grad_norm(grads, max_norm):
    total = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    scale = min(1.0, max_norm / (total + 1e-6))
    return {k: g * scale for k, g in grads.items()}, total

# PyTorch：在 loss.backward() 之后、opt.step() 之前调用
# total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

按范数裁剪保留了梯度方向；按值裁剪（`clip_grad_value_`，每个分量单独截断到 $[-c,c]$）会改变方向，一般不如前者。RNN 和 Transformer 训练中几乎总会开启按范数裁剪，返回的 `total_norm` 也是监控训练是否即将发散的好指标。

> [!EXAMPLE]+ 例10 权值衰减的效果
>
> 在例 9 的过拟合实验中加入 $\lambda=0.1$ 的权值衰减：训练集与测试集准确率之间仍有差距，但比不加时小得多；同时训练集准确率不再达到 100%。这正是正则化的典型表现：牺牲一点训练集拟合，换取更好的泛化。

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

### Dropout

> [!INFO]+ Dropout
>
> Dropout（Srivastava et al., 2014）：训练时每个神经元以概率 $1-p$ 置零，推理时用全部神经元。作业用 inverted dropout，训练阶段把保留值除以 $p$，使期望与推理一致。它近似许多子网络的平均，不能代替 $L_2$。卷积层权值已共享，通常只加在全连接头。

<br>

网络很复杂时，单靠权重衰减难以应对，这时常用 Dropout：**训练时随机删除隐藏层的神经元**，被删除的神经元不再传递信号。每传入一次数据（每个 mini-batch），都重新随机选择要删除的神经元。

设 $p$ 为保留概率（keep probability）。CS230 作业使用 **inverted dropout**：

$$
A^{[l]}\leftarrow\frac{D^{[l]}\odot A^{[l]}}{p},\qquad D^{[l]}\_{ij}\sim\mathrm{Bernoulli}(p).
$$

- 训练时，每个神经元以概率 $1-p$ 被置零，保留下来的值除以 $p$。
- 为什么除以 $p$：保留后激活的期望变成原来的 $p$ 倍，除以 $p$ 把期望拉回原值，这样测试时用全部神经元、不做任何缩放即可。
- 反向：只对保留的位置传梯度，并同样除以 $p$。行为与 ReLU 类似：前向传了信号的神经元，反向原样传梯度；前向被删除的，反向梯度停在那里。
- 鱼书的写法是另一种等价实现：训练时只乘掩码、不缩放，**测试时**把每个神经元的输出乘以保留比例 $1-\text{dropout\_ratio}$。两者期望一致，框架多采用 inverted 版本，因为推理代码不用改。

为什么有效：

- **不能依赖个别神经元**：任何一个隐藏单元都可能被删掉，网络只能把特征分散编码在多个单元中，效果类似于把权重的“负担”摊开，与 $L_2$ 收缩权重有相似作用。
- **集成学习的近似**：集成学习（ensemble）训练多个模型、推理时取平均，通常能把精度提高几个百分点。Dropout 每次训练的都是一个随机子网络，相当于训练了指数级多个共享权重的子网络；推理时按比例缩放，相当于对这些子网络取平均。即**用一个网络近似实现了集成学习的效果**。

> [!EXAMPLE]+ 例11 Dropout 的效果（鱼书 6.4.3 节）
>
> 同样的 7 层网络，加入 dropout 比例 0.15（即保留 85%）：训练集与测试集准确率的差距变小，训练集准确率也不再达到 100%。即使是表现力很强的网络，也能用 Dropout 抑制过拟合。

实践要点：

- $p$ 常用：输入层 $0.8$，隐藏层 $0.5$；参数多、易过拟合的层可以用更小的 $p$。
- 卷积层因参数已共享，Dropout 效果较弱；常改用空间 Dropout，或只在全连接头使用。
- 与 BatchNorm 同时使用时，二者都在改变激活分布，调参需更小心。许多现代卷积架构主要靠 BatchNorm、数据增强与权重衰减。
- 使用 Dropout 后损失函数 $J$ 不再有明确定义（每次前向都不同），调试时先关掉 Dropout 确认 $J$ 单调下降，再打开。

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

PyTorch 的 `nn.Dropout(p)` 中 `p` 是**丢弃**概率，即 CS230 的 $1-\text{keep\_prob}$；它同样是 inverted 实现，`eval()` 时直接恒等：

```python
drop = nn.Dropout(p=0.2)            # keep_prob = 0.8
x = torch.ones(1, 10)
drop.train(); print(drop(x))        # 约 2 个 0，其余为 1/0.8 = 1.25
drop.eval();  print(drop(x))        # 全为 1，不缩放

mlp = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(256, 10))
```

$L_2$ 在 PyTorch 里通常不写进损失，而是交给优化器的 `weight_decay`。只惩罚 $W$、不惩罚 $b$ 时，要把参数分组：

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if name.endswith("bias") or p.ndim == 1 else decay).append(p)   # 偏置与 BN 参数不衰减
opt = torch.optim.SGD([{"params": decay, "weight_decay": 1e-4},
                       {"params": no_decay, "weight_decay": 0.0}], lr=0.1, momentum=0.9)
```

SGD 的 `weight_decay=w` 等价于在损失里加 $\frac{w}{2}\lVert W\rVert^{2}$。对应 CS230 的 $\frac{\lambda}{2m}\lVert W\rVert^{2}$，取 $w=\lambda/m$。

- 训练时 Dropout 必须开启，推理必须关闭。Keras/TF 用 `training=` 切换；推理时若仍除以 $p$ 且随机置零，dev 准确率会无故抖动。
- Inverted dropout 的除以 $p$ 只属于训练分支。
- $L_2$ 与 Dropout 可以叠加，但二者都在压有效容量；同时把 $\lambda$ 和 $1-p$ 开到很大，会回到欠拟合。
- 正交化：先用更大网络把训练误差降下来，再加一种正则观察 dev。

### 其他正则

- **数据增强**（data augmentation）：从已有数据中生成更多数据，是“免费”的样本。图像可以翻转、旋转、随机裁剪、颜色偏移、加噪声、遮挡部分区域、改变对比度；文本可以同义改写。增强通常在训练时实时（on the fly）进行。
  - 增强必须保持标签语义：分类可以水平翻转，但检测框坐标要跟着翻转；文字与左右不对称的医学影像不能盲目翻转；水平翻转车牌会破坏字符顺序。
  - 增强过猛等于换了数据分布；dev 若未增强会出现人为的分布不匹配。
  - CS230 速查表按“模拟哪一种真实变化”列出图像增强：

    | 变换 | 做法 | 模拟的真实情况 |
    | :--- | :--- | :--- |
    | 翻转（flip） | 沿不改变语义的轴翻转 | 左右镜像的同类物体 |
    | 旋转（rotation） | 小角度旋转 | 拍摄时地平线没放平 |
    | 随机裁剪（random crop） | 随机聚焦图像的一部分，可连续裁多次 | 物体位置、取景范围不同 |
    | 颜色偏移（color shift） | 轻微改变 RGB 各通道 | 光照、曝光带来的色偏 |
    | 加噪声（noise addition） | 叠加随机噪声 | 输入质量参差（传感器噪声、压缩） |
    | 信息丢失（information loss） | 随机遮掉图像的一部分 | 物体被部分遮挡 |
    | 对比度变化（contrast change） | 调整亮度与对比度 | 一天中不同时段的曝光差异 |

  - 选哪几种，取决于部署时真的会出现哪些变化：只在白天固定机位拍摄的数据，对比度增强意义不大；手机随手拍的数据，旋转和颜色偏移通常有用。
  - NumPy 写法（单张图 `(H, W, C)`，取值 $[0,1]$）：

    ```python
    def augment(img, rng):
        if rng.random() < 0.5:
            img = img[:, ::-1, :]                                  # 水平翻转
        H, W, _ = img.shape
        top, left = rng.integers(0, H // 8 + 1), rng.integers(0, W // 8 + 1)
        img = img[top:top + H - H // 8, left:left + W - W // 8]    # 随机裁剪 7/8
        img = img * rng.uniform(0.8, 1.2) + rng.normal(0, 0.02, img.shape)   # 对比度 + 噪声
        if rng.random() < 0.5:                                     # 信息丢失：遮一块
            h, w = img.shape[0] // 4, img.shape[1] // 4
            y, x = rng.integers(0, img.shape[0] - h), rng.integers(0, img.shape[1] - w)
            img = img.copy(); img[y:y + h, x:x + w] = 0.0
        return np.clip(img, 0.0, 1.0)
    ```

  - PyTorch（torchvision）把表中各项写成一条流水线，只挂在训练集上：

    ```python
    from torchvision import transforms as T

    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),       # 随机裁剪
        T.RandomHorizontalFlip(),                         # 翻转
        T.RandomRotation(10),                             # 小角度旋转
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),  # 颜色、对比度
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25),                          # 信息丢失（作用在张量上）
    ])
    eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ```

    dev/test 只做确定性的缩放、裁剪和归一化。`DataLoader` 每次取样都会重新随机增强，这就是速查表说的 on the fly。
- **Early stopping**：训练过程中监控 dev 损失，在它开始上升时停止，回滚到最优检查点。它把有效训练步数当成超参数，与 $L_2$ 的作用有重叠（训练越早停，权重离初始的小值越近）。缺点是把“优化 $J$”和“防止过拟合”两件事耦合在一起，违背正交化。
- **输入归一化**：每维减均值、除标准差。各维尺度一致后，损失函数的等高线从狭长的椭圆变圆，梯度下降可以用更大的学习率、更快收敛。训练集与测试集必须使用同一组 $\mu,\sigma$。
- **标签平滑**：把 one-hot 换成 $(1-\varepsilon)y+\varepsilon/K$，减轻过度自信。
- **噪声注入（noise injection）**：训练时用 $\tilde{x}=x+\varepsilon$，$\varepsilon\sim\mathcal{N}(0,\sigma^{2}I)$，迫使模型在样本邻域保持稳定。输入噪声、Dropout 和随机增强都属于随机正则（stochastic regularization），但三者作用位置不同。

Early stopping 与标签平滑的实现：

```python
import copy

def fit_with_early_stopping(model, train_one_epoch, eval_dev, max_epochs=100, patience=5):
    """dev 损失连续 patience 个 epoch 不降就停，回滚到最好的检查点。"""
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        train_one_epoch(model)
        dev_loss = eval_dev(model)
        if dev_loss < best - 1e-4:
            best, best_state, bad = dev_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best

def smooth_labels(Y_onehot, eps=0.1):
    """Y_onehot: (K, m)。返回 (1-eps)·y + eps/K。"""
    K = Y_onehot.shape[0]
    return (1.0 - eps) * Y_onehot + eps / K

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)   # PyTorch 内置，输入整数标签即可
```

> [!NOTE]+ 加噪为什么能正则（why noise regularizes）
>
> 对小噪声作二阶 Taylor 展开，线性项在零均值噪声下取期望后消失，留下与曲率有关的二阶项：
>
> $$
> \mathbb{E}\_{\varepsilon}[\mathcal{L}(x+\varepsilon)]
> \approx \mathcal{L}(x)
> +\frac{\sigma^{2}}{2}\operatorname{tr}
> \big(\nabla_x^{2}\mathcal{L}(x)\big).
> $$
>
> 因而训练不仅压低样本点处的损失，还惩罚输入附近过大的弯曲，使决策函数更平滑（smoother decision function）。噪声过大会改变标签语义，形成分布偏移而不是正则。

> [!WARNING]+ 正则不能修复错误标签或泄漏
>
> - test 经增强后泄漏进训练集：dev 指标虚高。
> - Dropout 不能补偿训练与部署的分布差。
> - 先检查划分与预处理，再加正则。

<br>

> [!NOTE]+ 衔接
>
> 到这里引入了一批不能由梯度学出来的量：学习率 $\alpha$、动量 $\beta$、正则强度 $\lambda$、保留概率 $p$、batch 大小、层数与层宽。下一节讨论如何在验证集上系统地选择它们。

<br>

## 超参数（Hyperparameter Tuning）

### 定义

超参数（hyperparameter）不能由梯度直接学习，但控制着训练过程和模型结构：$\alpha$、$\beta$、$\lambda$、层宽、层数、batch 大小、keep_prob 等。设置不当，模型性能会很差；而寻找好的取值通常需要大量试错。

### 验证数据

**不能用测试数据调超参数**。如果用测试集判断超参数的好坏，超参数就会被调成只适合测试集，测试集上的成绩不再代表泛化能力。因此要专门留出一份数据用于调超参数，称为**验证数据**（validation data，CS230 中称为 dev set）：

- 训练数据：学习参数（权重和偏置）；
- 验证数据：评估超参数；
- 测试数据：最后评估泛化能力，理想情况下只用一次。

每在 test 上看一次并据此改配置，test 就变成了第二块 dev，最终报告的泛化能力会被高估。若数据集只分了训练集和测试集，最简单的做法是先打乱训练集（数据可能按类别排序），再切出 20% 作为验证集：

```python
x_train, t_train = shuffle_dataset(x_train, t_train)
validation_num = int(x_train.shape[0] * 0.20)
x_val, t_val = x_train[:validation_num], t_train[:validation_num]
x_train, t_train = x_train[validation_num:], t_train[validation_num:]
```

### 重要性排序

不是所有超参数都同等重要。大致分层：

1. 学习率 $\alpha$（最重要）。
2. Momentum $\beta$、隐藏层宽度、mini-batch 大小。
3. 层数、学习率衰减、Adam 的 $\beta_1,\beta_2$、$\varepsilon$（后两者几乎总用默认值）。

### 搜索方法

- **随机搜索优于网格搜索**。网格搜索在低维且参数同等重要时可用；超参数很多时，若按网格取 $5\times 5$ 共 25 组，最重要的参数（如 $\alpha$）只试了 5 个不同值；随机采样 25 组，$\alpha$ 就试了 25 个不同值。因为各超参数对结果的影响程度不同，随机采样能在“重要的轴”上探索更多取值。
- **由粗到细**：先在大范围内随机采样，找到表现好的区域，再在这个区域内更密集地采样。
- **在对数尺度上采样**：范围以“10 的幂”指定。例如学习率在 $10^{-4}$ 到 $10^{-1}$ 之间时，若均匀采样，90% 的样本落在 $[0.01,0.1]$；应取 $\alpha=10^{u}$，$u\sim\mathrm{Unif}(-4,-1)$，使每个数量级获得同样多的样本。
- **$\beta$ 的采样**：$\beta=0.9,0.99,0.999$ 分别对应平均约 10、100、1000 步，对 $1-\beta$ 在对数轴上采样。原因是 $\beta$ 接近 1 时极其敏感：从 0.9 变到 0.9005 几乎没影响，从 0.999 变到 0.9995 却让平均窗口从 1000 步变成 2000 步。
- **缩短单次评估**：深度学习一次训练可能要几天，搜索时应减少 epoch，尽早放弃明显不好的配置。
- **贝叶斯优化**（Bayesian optimization）：以贝叶斯定理为基础，比随机搜索更严密、高效。

鱼书总结的流程：

1. 设定超参数的范围；
2. 从范围中随机采样；
3. 用采样到的值训练（epoch 设小），在验证集上评估；
4. 重复 2、3 步（如 100 次），根据结果缩小范围，再重复，直到范围足够小后选定一个值。

> [!EXAMPLE]+ 例12 学习率与权值衰减系数的搜索（鱼书 6.5.3 节，参考 CS231n）
>
> 在 MNIST 上搜索两个超参数，初始范围为权值衰减系数 $10^{-8}$ 到 $10^{-4}$、学习率 $10^{-6}$ 到 $10^{-2}$，在对数尺度上随机采样：
>
> ```python
> weight_decay = 10 ** np.random.uniform(-8, -4)
> lr = 10 ** np.random.uniform(-6, -2)
> ```
>
> 按验证准确率排序，前 5 名为：
>
> | 排名 | val acc | lr | weight decay |
> | :--- | :--- | :--- | :--- |
> | 1 | 0.83 | 0.0092 | $3.86\times 10^{-7}$ |
> | 2 | 0.78 | 0.00956 | $6.04\times 10^{-7}$ |
> | 3 | 0.77 | 0.00571 | $1.27\times 10^{-6}$ |
> | 4 | 0.74 | 0.00626 | $1.43\times 10^{-5}$ |
> | 5 | 0.73 | 0.0052 | $8.97\times 10^{-6}$ |
>
> 可见学习率在 0.001 到 0.01、权值衰减系数在 $10^{-8}$ 到 $10^{-6}$ 之间时学习顺利。下一轮就在这个缩小的范围内重复搜索。

### 两种策略

- **Panda 策略**（熊猫一次只养一个孩子）：同一时间只训练一个模型，每天观察曲线、手动调整。计算资源少时使用。
- **Caviar 策略**（鱼一次产很多卵）：并行训练多组配置，最后挑 dev 最好的。有集群时使用。
- 无论哪种，所有试验都要记录 seed、代码版本、数据版本与最终 dev 指标，否则无法复现“偶然得到的好点”。超参数也会随数据和硬件变化而“过期”，需要定期重新评估。

### 编程框架

C2M3 用编程框架搭建同一个模型。框架把 `GradientTape` 或静态图当成计算图，自动完成反向传播；`optimizer.apply_gradients` 对应手写的 `W -= lr * dW`。但以下几点仍须手动保证：

- 输入 shape 与作业约定一致，或显式 `transpose`；
- 用 `training=True/False` 切换 Dropout 与 BatchNorm 的行为；
- 分类使用 `from_logits=True` 的交叉熵，避免先 `softmax` 再 `log` 的数值不稳定；
- 指标在 epoch 结束时用完整 dev 集计算，不要只用最后一个 mini-batch。

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

- Keras 默认输入形状为 `(m, n_x)`（行样本），与 NumPy 作业的 `(n_x, m)` 相反；把同一组权重搬到 Keras 时需要转置 $W$。

TensorFlow 的 `GradientTape` 与 PyTorch 的 autograd 是同一件事的两种写法：都在前向时记录计算图，反向时按链式法则回放。

```python
# TensorFlow：显式“录像”
with tf.GradientTape() as tape:
    logits = model(x, training=True)
    loss = tf.keras.losses.binary_crossentropy(y, logits, from_logits=True)
grads = tape.gradient(loss, model.trainable_variables)
opt.apply_gradients(zip(grads, model.trainable_variables))

# PyTorch：张量自带记录，loss.backward() 把梯度写进每个参数的 .grad
model.train()
loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
opt.zero_grad(); loss.backward(); opt.step()
```

一个完整的 PyTorch 训练与评估骨架，后面各章的模型都可以套进去：

```python
def train_epoch(model, loader, loss_fn, opt, device="cpu"):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(x); n += len(x)
    return total / n

@torch.no_grad()                                  # 评估时不建计算图，省显存
def evaluate(model, loader, loss_fn, device="cpu"):
    model.eval()                                  # 关 Dropout，BN 用运行平均
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_fn(logits, y).item() * len(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(x)
    return total / n, correct / n                 # 在完整 dev 集上算指标
```

- `torch.manual_seed(0)` 与 `np.random.seed(0)` 一起设，实验才可复现。
- 在 GPU 上训练时，模型与数据都要 `.to(device)`，两者不在同一设备会直接报错。

<br>

> [!NOTE]+ 衔接
>
> 到这里已经有了一个可以训练、可以正则、超参数可以搜索的网络。但在真实项目中，“下一步该做什么”往往比“怎么实现”更难：收集更多数据？换更大的网络？调正则？下一部分把训练误差、开发集误差和部署误差分开看，用正交化的思路决定每次只改哪一类问题。

<br>

# 项目策略（ML Strategy）

## 动机

前两部分给出了一个能训练、能正则的网络。真实项目中更难的问题是：模型效果不够好时，**下一步该做什么**？可选的想法很多：

- 收集更多数据，或收集更多样的数据；
- 训练更久，换 Adam，换更大或更小的网络；
- 加 Dropout 或 $L_2$；
- 改网络结构、激活函数、隐藏单元数……

每个想法都可能要花几天到几个月，选错方向就白费时间。机器学习策略（ML strategy）讨论的不是某一层怎么实现，而是**实验的次序**：根据指标变化，判断下一步应该改数据、改正则还是改结构。

做法分两步：

1. 固定一个评估指标，以及一份与部署同分布的开发集（相当于先立好靶子）；
2. 根据训练误差、开发集误差和误差分析表，只安排下一件实验（再瞄准射击）。

同一轮改两件不相关的事，结果无法归因，这就是正交化要避免的。

<br>

## 正交化

正交化（orthogonalization）：把影响结果的因素分成几类，**每一类只由一个对应的手段调节**，一次实验只动一类。这样指标上升或下降时，才能把结果归因到那一类，而不是若干改动叠在一起。

Ng 的比喻是老式电视机：一个旋钮只调高度，一个只调宽度，一个只调梯形失真。如果一个旋钮同时改变高度、宽度和亮度，几乎不可能把画面调正。汽车也一样：方向盘管方向，油门和刹车管速度，两者分开才好控制。

监督学习项目要依次满足四个目标，每个目标只用表里的手段：

| 目标 | 只用这些手段 |
| :--- | :--- |
| 降低训练误差 | 加大网络、换优化器、加长训练 |
| 缩小 dev 与训练的差距 | 正则、增加与 train 同分布的数据 |
| 让 test 上的数字可信 | 保证 dev 与 test 同分布，并且两边都够大 |
| 让数字对部署有意义 | 保证 dev/test 的采集方式与部署数据一致 |

若 test 上好但实际效果差，要么 dev/test 分布不对，要么指标不对。

Early stopping 是一个“不正交”的例子：它同时影响训练误差（停得早，训练集拟合得差）和开发集误差，一个旋钮调两件事，所以 Ng 更倾向于用其他正则化手段。

> [!NOTE]+ 反例
>
> 同一轮既把隐藏层从 64 加到 512，又把 $\lambda$ 从 $10^{-4}$ 调到 $10^{-3}$。dev 无论变好还是变差，都无法判断是容量还是正则在起作用。训练误差仍然高时就加 Dropout，也会把两件事搅在一起。

<br>

## 评估指标

### 单一数字指标

评估指标（evaluation metric）是用来比较模型的数字。没有事先约定的指标，就无法判断一次改动是进步还是退步。理想情况是**一个数字**，这样可以立刻排出模型的好坏，快速迭代。

> [!NOTE]+ 分类指标（classification metrics）
>
> 混淆矩阵（confusion matrix）包含真阳性（true positive, TP）、假阳性（false positive, FP）、真阴性（true negative, TN）与假阴性（false negative, FN）。
>
> $$
> \begin{aligned}
> \mathrm{precision} &= \frac{TP}{TP+FP},&
> \mathrm{recall} &= \frac{TP}{TP+FN},\\
> \mathrm{sensitivity} &= \mathrm{recall},&
> \mathrm{specificity} &= \frac{TN}{TN+FP},\\
> F_1 &= \frac{2\,\mathrm{precision}\,\mathrm{recall}}
> {\mathrm{precision}+\mathrm{recall}}.&
> \end{aligned}
> $$
>
> Precision（查准率）回答“判为正的样本里有多少真是正类”；recall（查全率）回答“全部正类里找回了多少”。$F_1$ 是两者的调和平均，任何一个很低都会把它拉低。调高二分类阈值通常提高 precision、降低 recall，但具体结果取决于分数排序。
>
> ROC 曲线（receiver operating characteristic curve）画 true positive rate 对 false positive rate；AUROC 衡量随机抽一个正样本与负样本时，正样本得分更高的概率。正类极少时，precision-recall curve 与 PR AUC 通常比 ROC AUC 更能暴露假阳性。
>
> 多分类的 macro average 先逐类算再等权平均，强调少数类；micro average 先汇总所有类别的 TP、FP、FN 再计算，更受大类影响。

> [!EXAMPLE]+ 例1 两个指标无法直接比较
>
> 猫分类器 A：precision $95\%$，recall $90\%$；分类器 B：precision $98\%$，recall $85\%$。A 的 recall 高，B 的 precision 高，看不出谁更好。合成为 $F_1$：A 为 $92.4\%$，B 为 $91.0\%$，于是选 A。

- 类别不平衡时准确率失效（99% 都是负类时，全判负也有 99% 准确率）：改用 precision、recall、$F_1$、PR AUC、代价加权错误率。
- 检测用 mAP；分割用 IoU 或 Dice；检索用 Recall@$k$。
- 多个地区、多个子集各有一个误差时，取平均作为单一数字。

```python
def binary_metrics(y_true, y_score, threshold=0.5):
    """y_true: (m,) 取 0/1；y_score: (m,) 为预测概率。"""
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "specificity": tn / max(tn + fp, 1)}

def auroc(y_true, y_score):
    """随机抽一正一负，正样本得分更高的概率（平分记 0.5）。O(m log m)。"""
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score)); ranks[order] = np.arange(1, len(y_score) + 1)
    for s in np.unique(y_score):              # 同分取平均秩
        idx = y_score == s
        ranks[idx] = ranks[idx].mean()
    n_pos, n_neg = y_true.sum(), (1 - y_true).sum()
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

例 1 的两个分类器用 `binary_metrics` 算出的 $F_1$ 分别为 $0.924$ 与 $0.910$。工程上也可以直接用 `sklearn.metrics` 的 `precision_recall_fscore_support`、`roc_auc_score`、`average_precision_score`（PR AUC）对拍。

### 优化指标与满意度指标

有些要求本质上不是“越大越好”，而是“达标即可”。Ng 把指标分成两类：

- **优化指标**（optimizing metric）：只留一个，例如准确率，在满足约束的模型里越高越好。
- **满意度指标**（satisficing metric）：设阈值，例如运行时间 $<100\,\mathrm{ms}$、内存 $<1\,\mathrm{GB}$。达不到阈值的模型直接淘汰，达到之后再好也不加分。

一般地，$N$ 个指标中选 1 个作为优化指标，其余 $N-1$ 个作为满意度指标。多个优化指标若必须同时考虑，应加权成单一数字，权重必须在看结果之前写进提案。

> [!EXAMPLE]+ 例2 满意度约束优先
>
> 三个模型：A 准确率 $90\%$、延迟 $80\,\mathrm{ms}$；B 准确率 $92\%$、延迟 $95\,\mathrm{ms}$；C 准确率 $95\%$、延迟 $1500\,\mathrm{ms}$。要求延迟 $\le 100\,\mathrm{ms}$ 时，C 被淘汰，在 A、B 中比准确率，选 B。
>
> 又如唤醒词检测（"Hey Siri"）：优化指标是唤醒准确率，满意度指标是“每 24 小时误唤醒不超过 1 次”。

### 何时修改指标

指标与真实需求不一致时，要果断改指标。

> [!EXAMPLE]+ 例3 加权错误率
>
> 猫图推荐应用中，算法 A 错误率 $3\%$，但会把色情图片当成猫推给用户；算法 B 错误率 $5\%$，不会。按错误率应选 A，但用户和公司都会选 B，说明指标错了。改成加权错误率：
>
> $$
> \mathrm{Error}=\frac{1}{\sum_i w^{(i)}}\sum_{i}w^{(i)}\,\mathbb{1}\{\hat{y}^{(i)}\ne y^{(i)}\},\qquad
> w^{(i)}=\begin{cases}1 & x^{(i)}\text{ 非色情}\\ 10 & x^{(i)}\text{ 色情}\end{cases}.
> $$
>
> 色情图片被误判的代价是普通错误的 10 倍。实现时需要把这类图片标出来。

这本身也是正交化：**定义指标**（立靶子）与**如何在指标上做好**（瞄准）是两件独立的事，分开处理。

- 指标必须在项目开始时与业务代价对齐。训练后再挑选当时最高的那条曲线，等于用 test 调参。
- 在“指标 + dev/test”上表现好，但在实际应用中表现差，就改指标或改 dev/test。例如 dev/test 都是高清图，用户上传的却是模糊图。

<br>

## 数据划分

### 三个子集

划分（split）把带标签样本分到互不重叠的子集，避免用同一批数据既训练又评估。

- **训练集 train**：估计参数 $W,b$。
- **开发集 dev**（hold-out / validation）：选超参数、比较模型、做误差分析。日常迭代看的就是这块上的指标。
- **测试集 test**：最终评估用一次，无偏地估计模型在未见样本上的表现。

若只有 train/test，又在 test 上调学习率和结构，test 就失去了“未见”的含义，报告中的泛化会被高估。可以暂时不设 test、只留 dev，但终稿需写明没有盲测集。

### 按规模选比例

小数据年代常用 70/30 或 60/20/20。样本到百万以后，再把三成留给 test 会浪费训练数据。C3 的做法是：先确定 dev、test 需要多少条才够比较两个模型，其余全部放入 train。

| 总样本大约 | 常见划法 | 理由 |
| :--- | :--- | :--- |
| 几百到约 $10^{4}$ | 60/20/20 或 70/15/15；更老的写法是 70/30（train/test，再从 train 划 dev） | 每一块都要有足够样本，误差估计才稳 |
| $10^{4}$–$10^{5}$ | 80/10/10 或 90/5/5 | 开始把更多样本留给训练 |
| 约 $10^{6}$ | 98/1/1，例如 98 万 / 1 万 / 1 万 | 1 万条 dev 通常能看出约 $0.1\%$ 的差别 |
| 更大，到亿级 | 99.5/0.25/0.25，或固定划出 1 万–10 万做 dev/test，其余全进 train | 比例继续缩小，绝对数量不动 |

- 几百条还做深度网络：优先加数据或用强预训练，不先纠结 2% 的划分差。
- 约 1000 条 dev：大致能分辨 $1\%$ 的指标差；要稳稳看出 $0.1\%$，课上按约 1 万条估。
- 广告、搜索这类 $0.01\%$ 也值钱的任务，dev 可以远大于 1 万。
- test 的大小按“对最终数字有没有信心”定，不是按总数据的固定百分比。
- 类别不平衡时做**分层抽样**（stratified split），保证少数类在三块里都出现。正类只有几十个时，随机切可能让 test 里一个正类都没有。

> [!NOTE]+ 训练、dev 与 test
>
> - 训练误差高：模型还没拟合，先查学习率、bug、容量。
> - 训练低、dev 高，且两块同分布：方差，加数据或正则。
> - 训练和 dev 都低、test 突然高：多半在 test 上调过参（对 dev 过拟合了），或 test 被换过分布。
> - Dev 只有两三百条时，$0.8\%$ 的波动常被采样噪声淹没，不足以支持换架构。

> [!EXAMPLE]+ 例3a 课上的昼夜分类：1 万张怎么切
>
> CS230 Lecture 2 的热身题：给一张户外照片，判断白天（$0$）还是夜晚（$1$）。
>
> - **要多少数据**：参照猫分类器约 1 万张图就能做好，估计这个任务难度相当，先收集约 1 万张。任务定义会改变这个数：只在一个固定机位判断昼夜很容易；要求世界任何地点、室内、黎明黄昏、极昼地区都能判断，1 万张远远不够。
> - **怎么切**：约 $80\%$ 进 train，其余做 dev/test。数据更多时，留给评估集的**比例**更小、留给 train 的更大。
> - **分层**：不是整体随机切，而是**按类别分别切**：白天图的 $80\%$ 和夜晚图的 $80\%$ 各自进 train，保证每块里两类比例与总体一致。
> - **输入分辨率**：把同一批图缩成不同分辨率给人看，找出人还能稳定判断的最低分辨率，结果约 $64\times 64\times 3$。颜色要保留，因为蓝天本身就是白天的线索，去掉颜色后人也更难判断。分辨率太低会丢信息（例如看不清画面里的钟），太高则参数多、训练慢、迭代周期长。
> - **输出与损失**：输出 $0/1$，末层 sigmoid，二元交叉熵；任务简单，浅层网络或小 CNN 即可。

### Dev 要多大

“1 万条 dev 能看出 $0.1\%$ 的差别”是经验说法，可以用误差估计的标准误来核对。

在 $n$ 条 dev 上测得错误率 $\hat{p}$，它是 $n$ 个独立 $0/1$ 变量的平均，标准误（standard error）为

$$
\mathrm{SE}(\hat{p})=\sqrt{\frac{p(1-p)}{n}},\qquad
\text{95\% 置信区间}\approx\hat{p}\pm 1.96\,\mathrm{SE}.
$$

取真实错误率 $p=10\%$：

| dev 大小 $n$ | $\mathrm{SE}$ | 95% 区间半宽 |
| :--- | :--- | :--- |
| $100$ | $3.0\%$ | 约 $\pm 5.9\%$ |
| $1000$ | $0.95\%$ | 约 $\pm 1.9\%$ |
| $10^{4}$ | $0.30\%$ | 约 $\pm 0.59\%$ |
| $10^{5}$ | $0.095\%$ | 约 $\pm 0.19\%$ |

单看一个模型，1 万条 dev 的误差估计仍有约 $\pm 0.6\%$ 的不确定性。但比较两个模型时，两者是在**同一批** dev 上评估的，大部分样本两个模型同对或同错，抵消掉了。真正起作用的只有“两者判断不同”的样本：记 $n_{10}$ 为 A 错 B 对的条数，$n_{01}$ 为 A 对 B 错的条数，则

$$
\hat{\Delta}=\frac{n_{10}-n_{01}}{n},\qquad
z=\frac{n_{10}-n_{01}}{\sqrt{n_{10}+n_{01}}}.
$$

$|z|>1.96$ 时，大约可以认为差别不是采样噪声（这就是 McNemar 检验的正态近似）。

> [!EXAMPLE]+ 例3b 1 万条 dev 上差 $0.1\%$ 算不算数
>
> - 两个模型只在 $30$ 条样本上判断不同，其中 A 错 B 对 $20$ 条、A 对 B 错 $10$ 条：$\hat\Delta=10/10^{4}=0.1\%$，$z=10/\sqrt{30}\approx 1.8$，接近显著。
> - 两个模型在 $300$ 条样本上判断不同，A 错 B 对 $155$、A 对 B 错 $145$：$\hat\Delta$ 同样是 $0.1\%$，但 $z=10/\sqrt{300}\approx 0.58$，完全可能是噪声。
>
> 所以“dev 多大才够”不只看总条数，还看两个候选模型的分歧有多大。改动越小（只调一个超参数），分歧越少，同样的 dev 越容易看出差别。

- 反过来估算：希望单个模型的 $\mathrm{SE}$ 不超过 $s$，需要 $n\ge p(1-p)/s^{2}$。$p=10\%$、$s=0.1\%$ 时 $n\ge 9\times 10^{4}$。
- 错误率越低，同样的 $n$ 能看出的差别越小：$p=1\%$ 时 1 万条的 $\mathrm{SE}$ 约 $0.1\%$。
- test 只评一次，按“最终报告的数字需要多大的置信区间”来定大小，再写进报告。
- 正类极少时，上式中的 $n$ 应换成**正类的条数**来估 recall 的不确定性：1 万条 dev 里只有 50 个正类，recall 的 $\mathrm{SE}$ 约为 $\sqrt{0.5\times 0.5/50}\approx 7\%$。

### 小数据：k 折交叉验证

样本只有几百到几千条时，固定切出 $20\%$ 做 dev，dev 小到噪声很大，train 也少了一块。这时常用 **k 折交叉验证**（k-fold cross-validation）：

1. 把数据（先扣除 test）随机分成 $k$ 份，常取 $k=5$ 或 $10$；
2. 轮流用其中 $1$ 份做验证、其余 $k-1$ 份训练，共训练 $k$ 次；
3. 报告 $k$ 次验证误差的均值与标准差，用均值选超参数；
4. 选定超参数后，在全部训练数据上重训一次，最后在 test 上评一次。

- $k=m$（每次只留一个样本）称为留一法（leave-one-out），偏差最小但要训练 $m$ 次。
- 代价是训练 $k$ 次。深度学习里数据多、单次训练贵，一般只用固定 dev；医学影像、小样本科研数据才更常用交叉验证。
- 若交叉验证同时用来选超参数和报告最终性能，结果会偏乐观。要么另留 test，要么用嵌套交叉验证（nested CV）：外层折估计性能，内层折选超参数。
- 前面“常见切法错误”里的规则同样适用于每一折：
  - **分层 k 折**（stratified k-fold）：每折里类别比例一致；
  - **分组 k 折**（group k-fold）：同一病人、同一用户的样本只进同一折；
  - **时间序列**用前向滚动（forward chaining）：第 $i$ 次用前 $t_i$ 段训练、紧接着的一段验证，绝不让验证段早于训练段。

```python
def kfold_indices(m, k=5, seed=0):
    """返回 k 个 (train_idx, val_idx)。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(m)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, val))
    return out
```

<br>

### Dev 与 test 的分布

- **Dev 和 test 必须取自同一分布**，并且尽量就是部署后真正要处理的数据。否则团队花几个月在 dev 上优化，到 test 上才发现瞄错了靶。课上的例子：dev 来自美国、英国等地区，test 来自其他地区，结果白忙。
- 部署输入是手机模糊图，dev 就不能只放单反爬图。
- Train 可以更大，也可以来自网上爬取；来源不同时，另做训练-开发不匹配分析（见“分布不匹配”）。
- 随机打乱再按比例切，隐含三块同分布。来源不同时，这个假设不成立。

> [!EXAMPLE]+ 例4 课上的猫图：20 万网上 + 1 万用户上传
>
> 网上图清晰，用户上传的是手机拍照，部署后要分的是后者。
>
> - **方案一**：混在一起随机切成 20.5 万 / 2500 / 2500。dev 中平均只有 $2500\times 10000/210000\approx 119$ 张手机图，绝大多数仍是网上图，团队实际上在优化网上图，部署到手机拍照时误差会明显升高。
> - **方案二**（课上推荐）：20 万网上图加 5000 张用户图进 train；剩下 5000 张用户图对半切成 dev 和 test（各 2500）。dev/test 对准了目标分布，train 里也出现目标分布样本，后面分析不匹配时更好用。
>
> 方案二的代价是 train 与 dev 分布不同，需要用后文的 train-dev 集来区分方差和不匹配。
>
> - **方案二的变体**：20 万网上图全部进 train，1 万张用户图对半切成 dev 和 test（各 5000），train 里不放用户图。dev/test 更大、比较模型更稳，但 train 中完全没有目标分布样本，不匹配误差会更大。目标分布样本较多时，再从中拿出约 5000 张放进 train，就回到方案二。
> - 选哪种取决于目标分布样本有多少：先保证 dev/test 够用来比较模型（见上文“Dev 要多大”），剩下的目标分布样本放进 train。

### 项目里常见的切法错误

- **泄漏**（data leakage）：本应只出现在评估集里的信息，在训练时已经被模型看见。同一病人的多张片子、同一段视频的相邻帧、同一篇文章切出来的句子，分别进了 train 和 test，就属于这一类。按“组”整段划分，不要按单张图随机。
- **时间序列、日志、股价**：按时间切，前面训、后面测。打乱等于把未来泄漏给训练。
- **先增强再划分**：同一张图的两个裁剪分别进 train 和 test，等于抄作业。先划分，增强只做在 train 上。
- **用 test 估均值方差或建词表**：预处理统计只在 train 上算，dev/test 复用同一套。
- **每次实验重新随机切**：数字在跳，不是模型变强。写死随机种子，把 index 存进仓库。

```python
def stratified_split(y, train_frac=0.8, dev_frac=0.1, seed=0):
    """按标签分层。y: (m,) 整数标签。返回 train/dev/test 下标。"""
    rng = np.random.default_rng(seed)
    train_idx, dev_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * train_frac))
        n_dv = int(round(n * dev_frac))
        train_idx.append(idx[:n_tr])
        dev_idx.append(idx[n_tr : n_tr + n_dv])
        test_idx.append(idx[n_tr + n_dv :])
    return (
        np.concatenate(train_idx),
        np.concatenate(dev_idx),
        np.concatenate(test_idx),
    )
```

- `train_frac + dev_frac` 应小于 1，剩余进 test。
- 某类只有 3 条时，分层也会把某块切空，这时先合并稀有类或改用交叉验证，不要硬切三块。

### 何时改划分

- 产品目标变了：从网页猫图改成 App 拍照，旧的互联网 dev 不能再用来选模型。
- 指标不反映真实代价（例 3）：改加权误差或加满意度约束，并重建 dev/test。
- 改了划分或指标之后，历史模型要在新集合上重评。旧表上的数字不能和新表混着比。

<br>

## 人类水平

### 为什么要和人比

模型准确率随时间的典型曲线是：快速上升直到接近人类水平，然后放缓，渐近逼近一个永远达不到的上限——**Bayes 误差**（Bayes error），即任何分类器在该数据分布上能达到的最低错误率。标签本身有噪声、或输入信息不足以区分两类（图片太模糊、音频噪声太大）时，它大于 0。这个量通常无法直接计算。

超过人类水平后进展变慢，原因有二：一是人类水平在很多任务上已经接近 Bayes 误差，剩余空间很小；二是只要模型还不如人，就有一套好用的工具——让人来标更多数据、让人分析错误原因、更准确地估计偏差与方差，超过人类后这些工具都失效了。

### 人类水平作为 Bayes 误差的代理

人类水平误差（human-level error）$h$ 是一组标注者在同一任务上的错误率，当作 Bayes 误差的可观测上界（近似）。取普通用户还是专家委员会，数值差很大，需要与目的一致。

> [!EXAMPLE]+ 例5 医学影像中的“人类水平”取哪个
>
> 某放射影像分类任务：普通人 $3\%$，普通医生 $1\%$，经验丰富的医生 $0.7\%$，经验丰富的医生团队讨论后 $0.5\%$。
>
> 若目的是估计 Bayes 误差，应取 $0.5\%$：Bayes 误差不可能高于已经达到的最好水平。若目的是写论文或部署（“超过普通医生即可用”），可以取 $1\%$。

### 可避免偏差

有了 $h$，就能把“偏差”说得更精确：

- **可避免偏差**（avoidable bias）$=$ 训练误差 $-\ h$。训练误差不可能低于 Bayes 误差，所以只有超出 $h$ 的部分才“可避免”。
- **方差** $=$ 开发集误差 $-$ 训练误差（训练与开发集分布不同时，改用 train-dev 误差 $-$ 训练误差）。

> [!EXAMPLE]+ 例6 同样的误差，不同的结论
>
> 训练误差 $8\%$、开发集误差 $10\%$：
>
> - 若人类水平为 $1\%$（清晰的猫图）：可避免偏差 $7\%$，方差 $2\%$，应**先降偏差**：加大网络、训练更久、换更好的优化器或架构。
> - 若人类水平为 $7.5\%$（模糊的猫图）：可避免偏差只有 $0.5\%$，方差 $2\%$，应**先降方差**：加数据、正则。
>
> 不知道 $h$，就无法决定先做哪个。

> [!NOTE]+ 怎么用 $h$
>
> 训练误差明显高于 $h$：可避免偏差大，应加容量或检查优化。训练误差已经接近 $h$、dev 仍高：按方差处理。标注者互相矛盾时，$h$ 会被抬高，模型可能只是在拟合噪声。

<br>

- 两项都大时，先处理数值更大的那一项。
- 训练和 dev 都已经接近 $h$：再加大模型收益很小，应改任务定义、输入信号或产品范围。
- **超过人类水平之后**：例如人类团队 $0.5\%$，训练误差 $0.3\%$，开发集 $0.4\%$。这时无法判断还有多少可避免偏差（Bayes 误差可能是 $0.1\%$，也可能是 $0.3\%$），方向不再清晰。误差分析仍有用，只是不宜再把 $h$ 当成硬上限。
- 机器已明显超过人类的领域多是**结构化数据**：在线广告、商品推荐、物流时间预测、贷款审批。它们数据量巨大，且不是人类擅长的自然感知任务。在视觉、语音等感知任务上超过人类要难得多，但也已在部分任务上实现。

### 小结：提升模型表现的两条路

监督学习要做好，依赖两个基本假设：

1. 能把训练集拟合得很好（可避免偏差小）；
2. 训练集上的表现能推广到开发集和测试集（方差小）。

| 问题 | 手段 |
| :--- | :--- |
| 可避免偏差 | 更大的模型；训练更久或更好的优化算法（Momentum、RMSprop、Adam）；更好的架构（RNN、CNN）或超参数搜索 |
| 方差 | 更多数据；正则化（$L_2$、Dropout、数据增强）；更好的架构或超参数搜索 |

<br>

## 误差分析

### 做法

误差分析（error analysis）：在 dev 的错误样本上**人工**统计失败类型，用各类所占比例估计“若该类被完全修好，指标最多升多少”，再决定下一实验。

> [!EXAMPLE]+ 例7 要不要做狗分类器
>
> 猫分类器在开发集上错误率 $10\%$。同事发现一些狗被误认为猫，提议花几个月专门处理狗。先别动手：从开发集中抽 100 个误判样本，数一数其中有几只狗。
>
> - 若只有 5 只：这类问题彻底解决，错误率最多从 $10\%$ 降到 $9.5\%$，不值得花几个月。
> - 若有 50 只：最多从 $10\%$ 降到 $5\%$，值得做。
>
> 这个“最多能改善多少”称为该方向的**性能上限**（ceiling）。

- 从 dev 里抽出约 100 个错误样本（错误很少就全看）。
- 同一张表上**并行**评估多个想法，打勾统计：标注错、狗、大型猫科动物（狮子、豹）、模糊、滤镜、过曝、框不准、光线等。一张图可以属于多列，表中还可加备注列。
- 某列只占错误的 $2\%$：这列做到完美，总体也只改善约 $2\%$ 的错误，不应成为主攻。
- 上限 $=$ 该列占比 $\times$ 当前错误率。下一实验优先做上限更高的那一列。
- 这比凭直觉加新模块便宜得多，也是期中案例题的标准答法。

误差表本身就是一张“错误样本 × 标签”的布尔矩阵，算上限只需几行代码：

```python
def error_ceiling(tags, dev_error):
    """tags: {类型: (n_err,) 布尔数组}，第 i 个错误样本是否属于该类（可多选）。"""
    n_err = len(next(iter(tags.values())))
    rows = []
    for name, mask in tags.items():
        frac = mask.mean()
        rows.append((name, int(mask.sum()), frac, frac * dev_error))
    rows.sort(key=lambda r: -r[2])
    for name, cnt, frac, ceil in rows:
        print(f"{name:8s} {cnt:4d}/{n_err}  占比 {frac:5.1%}  最多降低 {ceil:.2%}")

rng = np.random.default_rng(0)
n = 100
error_ceiling({"狗": np.arange(n) < 8, "模糊": rng.random(n) < 0.61,
               "滤镜": rng.random(n) < 0.21, "标注错": rng.random(n) < 0.10},
              dev_error=0.10)
```

各列可以重叠，所以“占比”之和可以超过 $100\%$；每一列的上限要单独看，不能相加。

> [!EXAMPLE]+ 例8 猫识别器的误差表
>
> 100 张误分类中：狗 8、模糊 61、滤镜 21、标注错误 10。即使狗分类器做到完美，上限约 $8\%$ 的错误。优先收集模糊与滤镜数据，或在训练中加模糊增强，比设计狗检测器更合理。

### 标错的标签

数据中有标签本身是错的（例如把狗标成了猫），怎么办？

- **训练集**：深度学习对**随机**标签错误相当稳健，只要错误比例不高、数据量够大，往往不必修正；但对**系统性**错误（例如标注员把所有白色的狗都标成猫）不稳健。
- **开发 / 测试集**：在误差表里单开一列“标签错”，按其占比决定是否值得修正。

> [!EXAMPLE]+ 例9 标签错误值不值得修
>
> - 开发集整体错误 $10\%$，其中标签错误导致 $0.6\%$，其他原因 $9.4\%$：标签错误只占 $6\%$ 的错误，先解决其他问题。
> - 开发集整体错误降到 $2\%$，标签错误仍导致 $0.6\%$：已占 $30\%$ 的错误，而且会干扰对模型 A、B 的比较（A 错 $2.1\%$、B 错 $1.9\%$，已分不清谁更好），这时值得修正。

修正开发 / 测试集标签时的原则：

- 对 dev 和 test **同时**修正，保证两者仍同分布；
- 不仅检查模型判错的样本，也检查判对的样本（模型可能因标签错而“碰巧”判对），虽然后者数量大、成本高；
- train 可以不修，于是 train 与 dev/test 的分布会略有不同，这通常可以接受；
- 错标集中在某类、或占错误的大头：回头改标注协议，再重标这批；
- 改标签后重新算 $h$ 和误差表，旧的“已经超过人类”可能消失；
- 清洗标签的成本要和它在误差表上的上限比较，不必无期限洗全部数据。

### 先快速搭一个系统

面对新问题（例如语音识别），可改进的方向非常多：嘈杂背景（咖啡馆、车内）、口音、离麦克风远、儿童语音、口吃……不可能一开始就全部考虑。Ng 的建议是：

1. 快速确定 dev/test 和指标；
2. **快速搭出第一个系统**（几天之内）；
3. 用偏差/方差分析和误差分析决定下一步优先做什么。

若问题已有大量成熟研究（如人脸识别），可以直接从文献起步，不必从简单系统开始。

<br>

## 分布不匹配

### 问题

数据量的需求越来越大，常常只能把来源不同的数据混在一起训练（如例 4 的网上图 + 手机图），于是训练集与开发集不再同分布。这时 dev 高于训练误差，不能直接当成方差：一部分来自过拟合，一部分来自两套数据本身不同，这称为**数据不匹配**（data mismatch）。

> [!EXAMPLE]+ 例10 车载语音助手
>
> 要做车内后视镜上的语音助手。已有 50 万条各类语音（购买的数据、智能音箱、其他语音识别项目），但车内后视镜场景的语音只有 2 万条。按例 4 的思路：50 万条全部进 train，2 万条车内语音分给 dev 和 test（也可以拿出 1 万条放进 train）。

### Train-dev 集

为了区分方差和数据不匹配，从训练集中再随机划出一块 **训练-开发集**（train-dev）：与训练集同分布，但不参与训练。

比较四条误差：

| 量 | 含义 |
| :--- | :--- |
| 人类水平 $h$ | Bayes 误差的代理 |
| 训练误差 | 模型装训练数据的能力 |
| 训练-开发误差 | 同分布但未见过的数据上的表现 |
| 真实 dev 误差 | 再加上分布不匹配 |

- 训练误差 $-\ h$：可避免偏差。大则加容量、修优化。
- 训练-开发 $-$ 训练误差：方差。大则加与 train 同分布的数据，或加正则。
- 真实 dev $-$ 训练-开发：数据不匹配。大则让训练数据更接近部署分布。
- 真实 test $-$ 真实 dev：对 dev 的过拟合程度。这里再跳，多半是 test 被看过或切错，需要更大的 dev。

```python
def make_train_dev(train_idx, frac=0.05, seed=0):
    """从训练集下标中随机划出 train-dev（同分布、不参与训练）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(train_idx)
    k = int(len(idx) * frac)
    return idx[k:], idx[:k]            # 新的 train, train-dev

def diagnose(human, train, train_dev, dev, test=None):
    gaps = {"可避免偏差": train - human,
            "方差": train_dev - train,
            "数据不匹配": dev - train_dev}
    if test is not None:
        gaps["对 dev 过拟合"] = test - dev
    for k, v in gaps.items():
        print(f"{k:8s} {v:+.1%}")
    return max(gaps, key=gaps.get)      # 先处理最大的一项

diagnose(0.0, 0.01, 0.015, 0.10)       # -> '数据不匹配'（例 11 第二行）
```

> [!EXAMPLE]+ 例11 四种误差组合（人类水平约 $0\%$）
>
> | 训练 | 训练-开发 | 开发 | 诊断 |
> | :--- | :--- | :--- | :--- |
> | $1\%$ | $9\%$ | $10\%$ | 方差：同分布的新数据上就已经差了 |
> | $1\%$ | $1.5\%$ | $10\%$ | 数据不匹配：同分布泛化得好，换分布才变差 |
> | $10\%$ | $11\%$ | $12\%$ | 可避免偏差 |
> | $10\%$ | $11\%$ | $20\%$ | 可避免偏差 + 数据不匹配 |
>
> 开发集误差也可能低于训练误差，说明开发集分布比训练集“容易”（例如车内语音反而更清晰）。

### 如何处理数据不匹配

数据不匹配没有像偏差、方差那样系统的解法，但有几条实用办法：

- **人工分析差异**：看开发集样本，理解它和训练集差在哪里（例如车内语音多了汽车噪声、很多是门牌号）。注意只看 dev，不看 test，避免过拟合 test。
- **让训练数据更像 dev**：收集更多接近 dev 的数据（更多手机图，而不是更多网上图；更多车内录音），或者人工合成。
- **人工合成数据**（artificial data synthesis）：例如把干净语音和车内噪声叠加，得到“车内语音”。要警惕合成数据只覆盖了真实分布的一小块：
  - 1 万小时语音配 1 小时车噪声循环叠加，人耳听不出区别，但模型可能过拟合这 1 小时的噪声；
  - 从赛车游戏中截取汽车图片，游戏里可能只有 20 种车型，模型会过拟合这 20 种车。
- 合成要在 dev 上验证真的降低了不匹配误差；加了车载噪声却没降低不匹配项，合成无效。
- 让 train 里混入一部分目标分布样本，分析时更稳。不要把网上图和手机图混洗后再随机切，否则 dev 不再代表部署分布。
- 领域自适应、相关对齐等方法可以后做。课上先要求把四条误差写清楚，再决定要不要上这些方法。

<br>

## 迁移、多任务与端到端

### 迁移学习

**迁移学习**（transfer learning）：把在任务 A 上学到的知识用到任务 B 上。例如用 ImageNet 上训练好的图像识别网络做放射影像诊断：换掉最后的输出层，用影像数据重新训练。

- **预训练**（pre-training）：在任务 A 上训练；**微调**（fine-tuning）：在任务 B 上继续训练。
- 目标数据很少时，冻结骨干只训新的输出头；数据变多再解冻后几层；数据很多时全部重新训练，学习率比从头训练小约一个数量级。
- 为什么有效：低层特征（边缘、曲线、纹理；语音中的音素）在很多任务间通用。

迁移学习**有意义的条件**：

1. 任务 A 和 B 的输入类型相同（都是图像，或都是音频）；
2. 任务 A 的数据比任务 B 多得多（例如 A 有 100 万张图，B 只有 100 张）；
3. A 学到的低层特征对 B 有帮助。

反过来，A 只有 1 万张而 B 有 1 万张，迁移就没有多少意义。源和目标差太远（自然图像 $\rightarrow$ 病理切片）仍可作初始化，但要更快解冻，或换领域内预训练。

### 多任务学习

**多任务学习**（multitask learning）：一个网络同时做多个任务，一个共享的骨干接多个输出头。

> [!EXAMPLE]+ 例12 自动驾驶中的多标签检测
>
> 每张图要判断是否有行人、汽车、停车标志、交通灯，标签 $y^{(i)}\in\{0,1\}^{4}$。与 softmax 不同，一张图可以同时有多个 1。损失是 4 个逻辑回归损失之和：
>
> $$
> J=\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{4}\mathcal{L}\big(\hat{y}_j^{(i)},y_j^{(i)}\big),
> $$
>
> 其中 $\mathcal{L}$ 是二元交叉熵。若某些图片只标了一部分（例如只标了有没有行人，其他标“?”），求和时只对标了 0/1 的项求和即可。
>
> 用一个网络做 4 件事，通常比训练 4 个独立网络效果更好，因为前几层学到的特征可以共享。

“只对标了 0/1 的项求和”用一个掩码实现。把未标注的项记为 $-1$：

```python
def multitask_bce(logits, Y):
    """logits, Y: (m, 4)；Y 中 -1 表示未标注。"""
    mask = (Y >= 0).float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, Y.clamp(min=0).float(), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)   # 只按已标注项求平均

backbone = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten())
head = nn.Linear(32, 4)                  # 行人、汽车、停车标志、交通灯，各一个 sigmoid
logits = head(backbone(torch.randn(8, 3, 64, 64)))
Y = torch.tensor([[1, 0, -1, 1]] * 8)
print(multitask_bce(logits, Y))
```

这里不能用 softmax：一张图可以同时有多个 $1$，每个输出单元是独立的二分类。

多任务学习**有意义的条件**：

1. 各任务能从共享的低层特征中受益；
2. 通常各任务的数据量相近（每个任务都能从其他任务的大量数据中获益）；
3. 能训练一个足够大的网络同时做好所有任务（网络不够大时，多任务才会比分开训练差）。

- 数据量差太多时小任务会被大任务淹没，可给小任务加权，或先单独训头。
- 任务互相干扰、合在一起比分开更差，称为负迁移（negative transfer）；这时拆开比硬融更好。
- 实践中迁移学习比多任务学习常用得多；目标检测是多任务学习的典型应用。

### 端到端学习

**端到端学习**（end-to-end learning）：用一个网络直接学 $x\mapsto y$，取代由多个手工设计阶段组成的流水线。

> [!EXAMPLE]+ 例13 什么时候端到端
>
> - **语音识别**：传统流水线是 音频 → 特征（MFCC）→ 音素 → 单词 → 文本；端到端直接 音频 → 文本。数据少（约 3000 小时）时流水线更好；数据多（1 万到 10 万小时）时端到端更好。
> - **门禁人脸识别**：直接从摄像头原图识别身份很难（人可能站在不同位置、距离不同）。分两步更好：先检测人脸并裁剪放大，再把裁剪后的脸与数据库比对。每一步都更简单，而且每一步都有大量数据。
> - **机器翻译**：英文 → 文本分析 → … → 法文的流水线已被端到端取代，因为有大量 (英文, 法文) 句对。
> - **由手部 X 光片估计儿童年龄**：直接 图像 → 年龄 的数据很少；分两步——先从图像中分割出骨头、测量长度，再按骨骼长度与年龄的统计关系估计——效果更好。

- **优点**：让数据说话，不强加人类的先验（例如“音素”可能只是语言学家的概念，未必是最好的中间表示）；需要手工设计的组件更少。
- **缺点**：需要大量 $(x,y)$ 数据；排除了一些可能有用的手工设计组件。数据少时，手工设计的知识是注入先验的主要途径。
- **关键问题**：是否有足够的数据，学到从 $x$ 映射到 $y$ 所需复杂度的函数？
- 数据不够时，应拆成检测再识别、声学模型再语言模型，每一段都有自己的误差表。像素直接映射到方向盘在课上被当作反例：错了无法判断错在感知还是规划。实际的自动驾驶系统是 图像 → 检测车辆和行人（深度学习）→ 路径规划 → 转向控制（运动规划），只在适合的环节用深度学习。

<br>

## 数据与迭代

### 学习曲线

学习曲线（learning curve）把训练误差和 dev 误差画成训练样本量 $m$ 的函数。它回答的是“再加数据有没有用”，而不是再换一种层。

- 两条曲线都高、且已经靠近：偏差大。加数据帮助有限，应加容量、换特征或修优化。
- 训练低、dev 高，dev 曲线仍随 $m$ 下降：方差大，继续加与 train 同分布的数据通常有效。
- 训练低、dev 高，但 dev 已经走平：再爬同类网页帮助变小，应改正则、增强，或改收集更接近部署的数据。
- 横轴必须是样本量，不能只画 epoch。epoch 曲线看的是优化是否在走，学习曲线看的是数据是否够。

### 数据规模与模型

| 带标签样本大约 | 常见做法 |
| :--- | :--- |
| 几十到几百 | 强预训练加线性头，或非深度基线；先把标注协议写稳 |
| 约 $10^{3}$–$10^{4}$ | 小网络或冻结骨干微调；dev 至少约 200–1000 条才比较得出 $1\%$ 级差别 |
| 约 $10^{5}$ | 可以从零训中等 CNN / 浅层 Transformer；仍建议用预训练起步 |
| 约 $10^{6}$ 及以上 | 更深的网才吃得下；dev/test 按绝对数量留 1 万级，不必再按 20% 切 |

- 小模型加数据很快到顶；容量足够时，数据继续增加，误差往往还能降。
- 样本很少时，先估计标注成本和增强上限，再决定是否加深网络。加深而数据不够，训练误差会降、dev 会升。

### 迭代节奏

- 先做一个能跑通的基线：逻辑回归、小 CNN、或冻结骨干加线性头。同一天把训练和 dev 评估管道打通。
- CS230 Section 3 的建议：先找一个与题目接近的开源代码库，复现它报告的结果作为基线（baseline）。基线既是精度的起点，也是后面更复杂模型的比较对象。代码先在本机或 Colab 上用一小部分数据跑通、确认没有 bug 且输出合理，再搬到 GPU 实例上跑全量，避免把算力额度花在调 bug 上。
- 一次迭代只做误差表支持的一件改动。第一周先跑通基线，不换未经验证的大模型。
- 增强（翻转、裁剪、颜色抖动、SpecAugment）只作用在 train 上，并且要在 dev 上确认有用。增强代替不了目标分布的真实样本。
- 标签质量差会形成误差地板：模型再强也超不过互相矛盾的标注。
- 人工标注：先写协议、抽一小批算标注者一致性，再铺开。一致性低就先改协议。

<br>

## 案例

Course 3 的两个书面案例是策略部分的考试重点。

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
  - 可做成多任务：一个网络多个头（例 12）；
  - 某任务数据极少时，多任务可能被大数据任务主导，需要加权或分阶段训练；
  - 端到端“像素到方向盘”在数据不足时不可诊断；
  - 拆成检测、跟踪、规划后，每一段都有自己的误差分析；
  - 雷达与摄像头融合时，传感器时间不同步属于数据对齐问题；换 Adam 或加大网络不能修复时间戳错位。

> [!EXAMPLE]+ 例14 避免不可比的指标
>
> - 模型 A：准确率 $97\%$，延迟 $150\,\mathrm{ms}$。
> - 模型 B：准确率 $95\%$，延迟 $40\,\mathrm{ms}$。
> - 产品要求 $<100\,\mathrm{ms}$ 时，A 直接不合格。
> - 先过滤满意度约束，再在合格者中比优化指标。
> - 加权成单一数字也可以，但权重必须事先写进提案。

<br>

> [!NOTE]+ 四类失败
>
> - 训练误差高：偏差或优化失败。损失若不降，先查学习率与实现；下降后仍高，再加容量。
> - 训练低、dev 高、且同分布：方差。加数据或正则。
> - 训练-开发低、真实 dev 高：分布不匹配。
> - 开发集指标高、部署差：dev 的采集方式与部署不一致。先重建划分，再改模型。

<br>

> [!NOTE]+ 衔接
>
> 策略部分不指定用哪一种层。到目前为止网络都是全连接的，而图像上用全连接会丢掉“邻近像素更相关”这一先验，参数量也会爆炸。下一部分把 $WA$ 换成卷积，反向接口仍是 $\mathrm{d}W,\mathrm{d}X$。

<br>

# 卷积与视觉

前面的网络都是全连接的：每个神经元与上一层所有神经元相连。图像数据有两个特点让全连接很不合适：一是维度高（一张 $1000\times 1000$ 的彩色图有 300 万个输入），二是有空间结构（相邻像素关系密切，远处像素关系弱）。卷积网络（CNN）用“局部连接 + 权重共享”同时解决这两个问题。

这一部分的路线是：

1. **卷积网络**：卷积、填充、步幅、池化这些基本运算，以及它们的实现与反向传播；
2. **深度卷积**：为什么要加深，以及 VGG、ResNet、Inception、MobileNet 这些经典结构如何让深网络训得动、算得起；
3. **目标检测与分割**：输出从“一个标签”变成“若干个框”或“每个像素一个标签”；
4. **人脸与风格**：输出变成嵌入向量，或者干脆优化输入像素本身。

<br>

## 卷积网络（Convolutional Neural Network）

### 动机

> [!INFO]+ 卷积
>
> 一张图若先展平再全连接，参数随长宽暴涨，而且还假定任意两个像素同等相关。卷积用同一块小核在图上滑，邻近像素共享权重。LeNet-5（1998）已经这样用，AlexNet / VGG 之后成为视觉默认。作业和框架里说的“卷积”，多数其实是互相关，核不翻转。

<br>

全连接层的根本问题是**忽视了数据的形状**。图像是高 × 宽 × 通道的三维数据，空间上相邻的像素取值相近，RGB 各通道之间密切相关，相距很远的像素关系不大——这些模式都藏在三维形状里。但输入全连接层时，必须把它拉平成一维（MNIST 的 $(1,28,28)$ 被排成 784 个数），所有像素被当作“同一维度的神经元”平等对待，形状信息全部丢失。

卷积层则**保持形状**：以三维数据接收输入，同样以三维数据输出给下一层。卷积层的输入输出数据称为**特征图**（feature map），分为输入特征图与输出特征图。

卷积的两个关键性质：

- **权值共享**（parameter sharing）：同一个核扫完整张图。一个在左上角有用的特征检测器（如竖直边缘检测器），在右下角同样有用，没必要为每个位置学一套权重。
- **稀疏连接**（sparsity of connections）：每个输出只依赖输入中一个 $f\times f$ 的小区域，与其余像素无关。
- 由此得到**平移等变**（translation equivariance）：输入平移后，特征图上的响应跟着平移，而不是整张图重算一套无关的权重。

> [!EXAMPLE]+ 例1 参数量对比
>
> - 输入 $32\times32\times3$（3072 个数），输出 $28\times28\times6$（4704 个数）。全连接需要 $3072\times 4704\approx 1400$ 万个权重；用 6 个 $5\times5$ 卷积核只需 $(5\times5\times3+1)\times 6=456$ 个参数。
> - 若全连接第一层有 1000 个单元，参数约 $3\times10^{6}$，且忽略了邻近像素更相关这一先验。
> - 卷积核 $f\times f\times n_C$ 在所有位置共享，一层的参数量为 $f^{2} n_C n_C'$（再加偏置 $n_C'$），**与输入的高宽无关**；但计算量仍随空间尺寸线性增长。

卷积仍是线性运算，反向传播是这个线性算子的伴随（转置），层的接口（前向缓存、反向返回 $\mathrm{d}X,\mathrm{d}W,\mathrm{d}b$）与全连接层相同。

<br>

### 卷积运算

卷积运算相当于图像处理中的**滤波器运算**。滤波器（filter，也称核 kernel）以一定间隔在输入上滑动，在每个位置把滤波器元素与对应输入元素相乘再求和（乘积累加运算），结果写到输出的对应位置。

> [!EXAMPLE]+ 例2 二维卷积的数值计算（鱼书 7.2.2 节）
>
> 输入 $4\times 4$，滤波器 $3\times 3$：
>
> $$
> X=\begin{pmatrix}1&2&3&0\\0&1&2&3\\3&0&1&2\\2&3&0&1\end{pmatrix},\qquad
> K=\begin{pmatrix}2&0&1\\0&1&2\\1&0&2\end{pmatrix}.
> $$
>
> 滤波器从左上角开始，每次右移一格，共有 $2\times 2$ 个位置。左上角位置的计算为
>
> $$
> 1\cdot 2+2\cdot 0+3\cdot 1+0\cdot 0+1\cdot 1+2\cdot 2+3\cdot 1+0\cdot 0+1\cdot 2=15.
> $$
>
> 四个位置的结果为 $\begin{pmatrix}15&16\\6&15\end{pmatrix}$。CNN 中滤波器的参数就是权重，此外还有一个偏置：偏置只有一个数，加到所有输出元素上。若偏置为 3，输出变为 $\begin{pmatrix}18&19\\9&18\end{pmatrix}$。

> [!EXAMPLE]+ 例3 边缘检测（Ng 课上的例子）
>
> $6\times 6$ 灰度图，左半边像素值为 10（亮），右半边为 0（暗），中间有一条竖直边缘。用竖直边缘检测器
>
> $$
> K=\begin{pmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{pmatrix}
> $$
>
> 卷积，得到 $4\times 4$ 的输出，每一行都是 $(0,30,30,0)$：中间两列的大值正好标出了边缘位置，两侧平坦区域输出 0。若把图像左右反过来（左暗右亮），输出变为 $(0,-30,-30,0)$，正负号表示边缘是“由亮到暗”还是“由暗到亮”。
>
> 把核转置就得到水平边缘检测器。人工设计的变体还有 Sobel 核 $\begin{pmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{pmatrix}$ 和 Scharr 核 $\begin{pmatrix}3&0&-3\\10&0&-10\\3&0&-3\end{pmatrix}$（给中间行更大权重）。深度学习的思路是：**不手工设计这 9 个数，而是把它们当作参数，用反向传播学出来**，网络可能学到 45°、70° 等任意方向的边缘检测器。

用公式写，单通道、步幅（stride）$s=1$、无填充（padding）时，核 $K\in\mathbb{R}^{f\times f}$ 与输入 $X$ 的运算为

$$
Z_{ij}=\sum_{u=0}^{f-1}\sum_{v=0}^{f-1}K_{uv}X_{i+u,j+v}+b.
$$

数学上严格的卷积要先把核翻转 $180^{\circ}$ 再做上式，上式实际是**互相关**（cross-correlation）。由于核是学出来的，翻不翻转没有区别，深度学习中仍统称为卷积。

### 填充与步幅

#### 填充

对 $4\times 4$ 的输入用 $3\times 3$ 的滤波器，输出缩小为 $2\times 2$。这带来两个问题：一是每卷积一次空间就缩小，深层网络卷几次就缩到 1，无法继续；二是边角像素只被少数几个窗口覆盖，信息被弱化。

**填充**（padding）在输入四周补上固定值（通常是 0）。幅度为 1 的填充让 $4\times 4$ 的输入变成 $6\times 6$，再用 $3\times 3$ 滤波器卷积，输出仍是 $4\times 4$，空间大小保持不变。填充幅度也可以设为 2、3 等任意整数（$4\times4$ 变成 $8\times8$、$10\times10$）。

- **valid**：不填充，$p=0$，输出缩小。
- **same**：填充到输出与输入大小相同。$s=1$ 时需要 $p=(f-1)/2$，所以 $f$ 通常取奇数（3、5、7），这样两侧填充对称，而且滤波器有一个中心像素。
- **full**：填充 $p=f-1$，使滤波器与输入只要有一个像素重叠就计算一次，输出为 $n+f-1$，比输入还大。深度学习中很少直接用，但转置卷积的输出尺寸与它有关。

步幅 $s>1$ 时，“same”的含义推广为输出大小 $\lceil n/s\rceil$（TensorFlow 的 `padding="SAME"`）。所需的总填充是

$$
p_{\text{总}}=\max\big((\lceil n/s\rceil-1)s+f-n,\ 0\big),
$$

两侧各分一半，总数为奇数时多出的 1 像素放在右（下）侧。例如 $n=7$、$f=3$、$s=2$：$\lceil 7/2\rceil=4$，$p_{\text{总}}=3\cdot2+3-7=2$，两侧各填 1，输出 $4\times4$。

#### 步幅

应用滤波器的位置间隔称为**步幅**（stride）。步幅为 2 时，窗口每次移动两格，兼具下采样效果。例如 $7\times 7$ 的输入、$3\times 3$ 滤波器、步幅 2，输出为 $3\times 3$。

增大步幅，输出变小；增大填充，输出变大。

#### 输出尺寸公式

输入 $n_H\times n_W$、滤波器 $f\times f$、填充 $p$、步幅 $s$ 时：

$$
n_H'=\left\lfloor\frac{n_H+2p-f}{s}\right\rfloor+1,\qquad
n_W'=\left\lfloor\frac{n_W+2p-f}{s}\right\rfloor+1.
$$

推导：填充后长度为 $n+2p$，窗口左端可以落在 $0,s,2s,\dots$，只要窗口右端不超过边界，即 $ks+f\le n+2p$，所以 $k$ 最多取 $\lfloor(n+2p-f)/s\rfloor$，再加上 $k=0$ 这个位置。除不尽时向下取整，表示窗口一旦伸出边界就不再计算（有的框架会报错，有的会自动取整）。

> [!EXAMPLE]+ 例4 尺寸计算
>
> 鱼书的三个例子：
>
> - 输入 $(4,4)$，$p=1$，$s=1$，滤波器 $(3,3)$：$\frac{4+2-3}{1}+1=4$，输出 $(4,4)$。
> - 输入 $(7,7)$，$p=0$，$s=2$，滤波器 $(3,3)$：$\frac{7-3}{2}+1=3$，输出 $(3,3)$。
> - 输入 $(28,31)$，$p=2$，$s=3$，滤波器 $(5,5)$：$n_H'=\frac{28+4-5}{3}+1=10$，$n_W'=\frac{31+4-5}{3}+1=11$，输出 $(10,11)$。
>
> 另一个例子：输入 $32\times32\times3$，核 $5\times5$，步幅 1，same padding：$p=2$，输出 $32\times32\times n_C'$。改为 valid、步幅 2：
>
> $$
> n'=\left\lfloor\frac{32-5}{2}\right\rfloor+1=14.
> $$
>
> 若再接 $2\times2$ 步幅 2 的池化，空间变为 $7\times7$。

### 多通道与多个滤波器

#### 三维卷积

图像有 RGB 三个通道。三维数据的卷积要求**滤波器的通道数与输入相同**：输入 $n_H\times n_W\times n_C$，滤波器 $f\times f\times n_C$。在每个位置，把 $f\times f\times n_C$ 个元素逐一相乘后**全部加起来**（各通道分别卷积再相加），得到一个数。所以一个滤波器只输出**一张**特征图（通道数为 1）。

滤波器的空间大小可以任意（$3\times3$、$5\times5$、$1\times1$），但通道数只能等于输入的通道数。若只想检测红色通道的竖直边缘，可以让绿、蓝通道的核全为 0。

#### 多个滤波器

要在通道方向上得到多个输出，就用多个滤波器：$n_C'$ 个滤波器得到 $n_C'$ 张特征图，叠成 $n_H'\times n_W'\times n_C'$ 的输出，再传给下一层。每个滤波器检测一种特征（竖直边缘、水平边缘、某种颜色的斑块……）。

把数据想成长方体的“方块”更清楚（鱼书用通道在前的 $(C,H,W)$ 顺序）：

- 输入 $(C,H,W)$，一个滤波器 $(C,FH,FW)$，输出 $(1,OH,OW)$；
- $FN$ 个滤波器组成四维权重 $(FN,C,FH,FW)$，输出 $(FN,OH,OW)$。例如 20 个通道数为 3、大小为 $5\times5$ 的滤波器，权重形状为 $(20,3,5,5)$；
- 每个滤波器一个偏置，偏置形状 $(FN,1,1)$，靠广播加到对应通道的每个位置上；
- 批处理时在最前面加批维：输入 $(N,C,H,W)$，输出 $(N,FN,OH,OW)$，$N$ 次处理汇成 1 次。

> [!EXAMPLE]+ 例5 一层的参数量
>
> 一层有 10 个 $3\times3\times3$ 的滤波器。每个滤波器有 $27$ 个权重加 $1$ 个偏置，共 $28$ 个参数；10 个滤波器共 $280$ 个参数。无论输入是 $64\times64$ 还是 $1000\times1000$ 的图像，参数都是 280 个，这正是卷积不易过拟合的原因之一。

#### 第 $l$ 层的记号

CS230 对第 $l$ 层卷积的记号：

- 超参数：滤波器大小 $f^{[l]}$，填充 $p^{[l]}$，步幅 $s^{[l]}$，滤波器个数 $n_C^{[l]}$；
- 输入 $n_H^{[l-1]}\times n_W^{[l-1]}\times n_C^{[l-1]}$，输出 $n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}$，其中 $n_H^{[l]}=\lfloor(n_H^{[l-1]}+2p^{[l]}-f^{[l]})/s^{[l]}\rfloor+1$；
- 每个滤波器 $f^{[l]}\times f^{[l]}\times n_C^{[l-1]}$；全部权重 $f^{[l]}\times f^{[l]}\times n_C^{[l-1]}\times n_C^{[l]}$；偏置 $n_C^{[l]}$ 个；
- 激活 $A^{[l]}$：$m\times n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}$（作业中 batch 维在前，NHWC）。

### 实现

#### 朴素实现

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

- 教学实现是 $O(m n_H' n_W' n_C' f^{2} n_C)$ 的多重循环，NumPy 中用 for 循环访问元素非常慢。
- `np.pad` 在高、宽两侧各加 $p$ 行/列零。same 且 $s=1$、$f$ 为奇数时两侧对称；$f$ 为偶数时 same 无法完全对称，框架把多出的 1 像素分到一侧，作业一般避开偶数核。

用 PyTorch 对拍朴素实现。两边的排布不同：作业是 `X: (m, H, W, C)`、`W: (f, f, C_in, C_out)`；PyTorch 是 `X: (m, C, H, W)`、`W: (C_out, C_in, f, f)`。另外，深度学习里的“卷积”实际是互相关，两边都不翻转核，所以转置排布后结果就能一致：

```python
import torch.nn.functional as F

rng = np.random.default_rng(0)
X = rng.standard_normal((2, 7, 7, 3))
W = rng.standard_normal((3, 3, 3, 8))
b = rng.standard_normal((1, 1, 1, 8))
Z_np, _ = conv2d_forward(X, W, b, stride=2, padding=1)

Z_t = F.conv2d(torch.tensor(X).permute(0, 3, 1, 2),        # NHWC -> NCHW
               torch.tensor(W).permute(3, 2, 0, 1),        # (f,f,Cin,Cout) -> (Cout,Cin,f,f)
               torch.tensor(b).reshape(-1), stride=2, padding=1)
print(np.abs(Z_np - Z_t.permute(0, 2, 3, 1).numpy()).max())   # 约 1e-15
print(Z_np.shape)   # (2, 4, 4, 8)：(7 + 2 - 3)//2 + 1 = 4
```

`nn.Conv2d(C_in, C_out, kernel_size=f, stride=s, padding=p)` 是带参数的模块版本，参数量为 $(f^{2}C_{\mathrm{in}}+1)C_{\mathrm{out}}$；`padding="same"` 只在 `stride=1` 时可用。

#### im2col

生产环境把卷积转成一次大矩阵乘法。**im2col**（image to column）把输入中每个滤波器要覆盖的区域（一个 $C\times FH\times FW$ 的小方块）展开成一行，所有位置的行堆成一个二维矩阵；滤波器也展开，每个滤波器成为一列。于是卷积变成：

$$
\underbrace{\text{col}}_{(N\cdot OH\cdot OW,\ C\cdot FH\cdot FW)}\times\underbrace{\text{col\_W}}_{(C\cdot FH\cdot FW,\ FN)}=\underbrace{\text{out}}_{(N\cdot OH\cdot OW,\ FN)},
$$

再 reshape 回 $(N,OH,OW,FN)$ 并转置成 $(N,FN,OH,OW)$。这与全连接的 Affine 层完全相同。

> [!EXAMPLE]+ 例6 im2col 的形状
>
> - 输入 $(1,3,7,7)$，滤波器 $5\times5$，步幅 1，无填充：输出空间 $3\times 3=9$ 个位置，每个位置展开 $3\times5\times5=75$ 个元素，`col` 形状 $(9,75)$。
> - 批大小改为 10，输入 $(10,3,7,7)$：`col` 形状 $(90,75)$。
> - 滤波器 $(10,3,5,5)$ 共 750 个元素，`reshape(10, -1)` 得 $(10,75)$，转置后为 $(75,10)$。

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W, self.b, self.stride, self.pad = W, b, stride, pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + (H + 2 * self.pad - FH) // self.stride
        out_w = 1 + (W + 2 * self.pad - FW) // self.stride
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = col @ col_W + self.b
        return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
```

- 代价：相邻窗口重叠时，同一个输入元素会被复制到多行，**内存消耗比朴素实现大**。
- 好处：归结为大矩阵乘法后，可以利用高度优化的线性代数库（BLAS），GPU 也更擅长处理大块汇总数据。Caffe、Chainer 等框架的卷积层都基于 im2col。
- 反向传播与 Affine 层相同，最后用逆操作 col2im 把梯度加回原位置。
- 其他加速方法：Winograd、FFT、cuDNN。

#### $1\times 1$ 卷积

$1\times 1$ 卷积不看邻域，只在通道维做线性组合（再加非线性），相当于对每个像素位置做一次全连接，参数 $n_C n_C'$。用途是**改变通道数**：例如 $28\times28\times192$ 的输入经过 32 个 $1\times1\times192$ 的滤波器，得到 $28\times28\times32$。它不是“没用的卷积”，而是 Inception 与 bottleneck 结构中控制计算量的开关（见后文）。

### 反向传播

> [!INFO]+ 卷积反向
>
> 记前向 $Z=X * K$。$\partial J/\partial K$ 是 $X$ 与 $\mathrm{d}Z$ 的互相关：每个核位置把所有窗口与对应输出梯度相乘再求和。$\partial J/\partial X$ 是 $\mathrm{d}Z$ 与 $180^{\circ}$ 翻转核的卷积，并处理步幅造成的空洞（把 $\mathrm{d}Z$ 的像素之间插入 $s-1$ 个零）。偏置梯度是 $\mathrm{d}Z$ 在 batch 与空间维上的和。池化反向不学习参数，只路由梯度。

<br>

直观理解：前向时每个输出 $Z_{hw}$ 是一个窗口与核的内积。按乘法节点的规则，$Z_{hw}$ 的梯度会流回两处：

- 流向核：$\mathrm{d}K\mathrel{+}=\text{窗口}\times\mathrm{d}Z_{hw}$（核在所有位置共享，所以梯度要把所有位置**累加**）；
- 流向输入：$\mathrm{d}\text{窗口}\mathrel{+}=K\times\mathrm{d}Z_{hw}$（窗口重叠处的梯度也要累加）。

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

- 卷积层是否除以 $m$，取决于损失是否已对 batch 平均。与全连接保持同一约定：损失含 $1/m$ 则梯度也含 $1/m$。

### 池化

> [!INFO]+ 池化
>
> 池化在 $f\times f$ 窗口内取最大或平均，降低空间分辨率，一般没有可学参数。max 的反向只把梯度送到 argmax。较新的骨干常用步幅卷积做下采样，分类头仍常用全局平均池化。

<br>

池化（pooling）是缩小高、宽方向空间的运算。**最大池化**（max pooling）取窗口中的最大值，**平均池化**（average pooling）取均值。图像识别中主要用最大池化。窗口大小一般与步幅相同（$2\times2$ 配步幅 2，$3\times3$ 配步幅 3）。

> [!EXAMPLE]+ 例7 最大池化（鱼书 7.3 节）
>
> 对 $4\times4$ 输入做 $2\times2$、步幅 2 的最大池化：
>
> $$
> \begin{pmatrix}1&2&1&0\\0&1&2&3\\3&0&1&2\\2&4&0&1\end{pmatrix}
> \ \longrightarrow\ 
> \begin{pmatrix}2&3\\4&2\end{pmatrix}.
> $$
>
> 每个 $2\times2$ 区域只保留最大值。

池化层的三个特征：

- **没有要学习的参数**：只是取最大值或平均值。超参数 $f,s$ 是固定的。
- **通道数不变**：每个通道独立计算，$n_H\times n_W\times n_C\rightarrow n_H'\times n_W'\times n_C$。
- **对微小位置变化稳健**：输入在宽度方向偏移 1 个像素时，最大池化往往返回相同结果。直观上，最大池化表示“这个区域里只要某处检测到该特征就保留”。

反向传播：最大池化只把梯度送到前向时最大值所在的位置，其余位置为 0（与 ReLU 的反向类似，前向时记下 argmax 掩码）；平均池化把梯度均分给窗口内每个位置。

```python
def pool_forward_cached(A_prev, f=2, s=2, mode="max"):
    """向量化版本（只循环输出位置），并返回反向所需的 cache。A_prev: (m, n_H, n_W, n_C)。"""
    m, H, W, C = A_prev.shape
    n_H, n_W = (H - f) // s + 1, (W - f) // s + 1
    A = np.zeros((m, n_H, n_W, C))
    for h in range(n_H):
        for w in range(n_W):
            win = A_prev[:, h * s:h * s + f, w * s:w * s + f, :]   # (m, f, f, C)
            A[:, h, w, :] = win.max(axis=(1, 2)) if mode == "max" else win.mean(axis=(1, 2))
    return A, (A_prev, f, s, mode)

def pool_backward(dA, cache):
    A_prev, f, s, mode = cache
    dA_prev = np.zeros_like(A_prev)
    _, n_H, n_W, _ = dA.shape
    for h in range(n_H):
        for w in range(n_W):
            hs, ws = h * s, w * s
            win = A_prev[:, hs:hs + f, ws:ws + f, :]
            g = dA[:, h, w, :][:, None, None, :]                     # (m, 1, 1, C)
            if mode == "max":
                mask = (win == win.max(axis=(1, 2), keepdims=True))  # argmax 掩码
                dA_prev[:, hs:hs + f, ws:ws + f, :] += mask * g
            else:
                dA_prev[:, hs:hs + f, ws:ws + f, :] += g / (f * f)  # 均分
    return dA_prev

x = np.array([[1, 2, 1, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 4, 0, 1]], float)
print(pool_forward_cached(x[None, :, :, None])[0][0, :, :, 0])   # [[2. 3.] [4. 2.]]，即例 7
```

- 窗口内有并列最大值时，上面的掩码会把梯度同时给所有最大值；C4 作业与 PyTorch 只给其中一个。要对拍时可以改用 `argmax` 只取第一个。
- PyTorch 对应 `nn.MaxPool2d(kernel_size=2, stride=2)` 与 `nn.AvgPool2d(2, 2)`，输入是 NCHW：`x.permute(0, 3, 1, 2)` 把 NHWC 转成 NCHW。

- 现代网络常用步幅卷积代替大范围池化。
- 分类头前仍广泛使用全局平均池化（global average pooling）。

### 典型结构

一个 CNN 由三种层组成：卷积层（CONV）、池化层（POOL）、全连接层（FC）。全连接网络中的 “Affine → ReLU” 被替换为 “Conv → ReLU →（Pool）”（池化有时省略）；靠近输出的层仍使用 “Affine → ReLU”，最后用 “Affine → Softmax” 输出。

$$
\mathrm{Conv}\rightarrow\mathrm{ReLU}\rightarrow\mathrm{Conv}\rightarrow\mathrm{ReLU}\rightarrow\mathrm{Pool}
$$

重复若干次后展平或全局平均，再接全连接或卷积分类头。整体趋势是**空间逐渐变小、通道逐渐变多**（LeNet、AlexNet、VGG 都是如此）。

> [!EXAMPLE]+ 例8 一个类 LeNet-5 的网络（Ng 课上的例子）
>
> 输入 $32\times32\times3$：
>
> | 层 | 输出 | 激活大小 | 参数量 |
> | :--- | :--- | :--- | :--- |
> | 输入 | $32\times32\times3$ | 3072 | 0 |
> | CONV1（$f=5,s=1$，6 个） | $28\times28\times6$ | 4704 | $(5\cdot5\cdot3+1)\cdot6=456$ |
> | POOL1（$f=2,s=2$） | $14\times14\times6$ | 1176 | 0 |
> | CONV2（$f=5,s=1$，16 个） | $10\times10\times16$ | 1600 | $(5\cdot5\cdot6+1)\cdot16=2416$ |
> | POOL2（$f=2,s=2$） | $5\times5\times16$ | 400 | 0 |
> | FC3 | 120 | 120 | $400\cdot120+120=48120$ |
> | FC4 | 84 | 84 | $120\cdot84+84=10164$ |
> | Softmax | 10 | 10 | $84\cdot10+10=850$ |
>
> 规律：池化层没有参数；卷积层参数很少；大部分参数在全连接层；激活大小随深度逐渐减小，但不能减得太快。

例 8 的网络用 PyTorch 写出来，并核对表中的参数量与形状：

```python
lenet = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),    # 32→28→14
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),   # 14→10→5
    nn.Flatten(),                                                   # 5·5·16 = 400
    nn.Linear(400, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10),                    # softmax 交给 CrossEntropyLoss
)
x = torch.randn(1, 3, 32, 32)             # PyTorch 用 NCHW
for layer in lenet:
    x = layer(x)
    n = sum(p.numel() for p in layer.parameters())
    if n or isinstance(layer, nn.MaxPool2d):
        print(f"{layer.__class__.__name__:10s} {tuple(x.shape)!s:18s} params={n}")
# Conv2d (1, 6, 28, 28) 456；MaxPool2d (1, 6, 14, 14) 0；Conv2d (1, 16, 10, 10) 2416；…
# Linear (1, 120) 48120；Linear (1, 84) 10164；Linear (1, 10) 850
```

> [!EXAMPLE]+ 例9 鱼书的 SimpleConvNet
>
> 结构为 Conv（30 个 $5\times5$ 滤波器，$p=0$，$s=1$）→ ReLU → Pool（$2\times2$）→ Affine（100）→ ReLU → Affine（10）→ Softmax。在 MNIST 上训练集识别率 $99.82\%$，测试集 $98.96\%$，对小网络而言已经很高。

#### 感受野

**感受野**（receptive field）指输出中一个像素所依赖的输入区域。

- 一层 $f\times f$、步幅 1 的卷积：感受野增加 $f-1$。
- 一般公式（第 $k$ 层，$F_j$ 为第 $j$ 层核大小，$S_i$ 为第 $i$ 层步幅，约定 $S_0=1$）：

$$
R_k=1+\sum_{j=1}^{k}(F_j-1)\prod_{i=0}^{j-1}S_i.
$$

  例如两层 $3\times3$、步幅 1：$R_2=1+2+2=5$。
- 步幅与池化会快速扩大感受野（后面层的增量要乘以前面所有步幅）。
- 过早下采样会丢掉定位信息，对检测与分割不利。

#### 小核堆叠

VGG 用堆叠的 $3\times3$ 代替大核，原因可以定量说明：

- 两层 $3\times3$ 的感受野等于一层 $5\times5$；参数 $2\times9=18$ 对 $25$（单通道）。
- 三层 $3\times3$ 的感受野为 7，与一层 $7\times7$ 相同；参数 $3\times9=27$ 对 $49$。考虑通道时为 $3\cdot 9\cdot C^{2}$ 对 $49 C^{2}$。
- 而且多了两次 ReLU，非线性表达能力更强。

- 通道通常按阶段（stage）倍增：64 → 128 → 256 → 512，同时空间减半，使每层计算量大致均衡。
- 全局平均池化：最后一张特征图每通道求平均，得到 $n_C$ 维向量再分类；去掉大全连接层，减轻过拟合。

#### 全连接层可以写成卷积

全连接层的每个输出与全部输入相连，这可以用卷积等价实现：对 $32\times10\times10$（通道 32、高宽 10）的输入，一个输出 100 个节点的全连接层，等价于 100 个 $32\times10\times10$ 的滤波器。这个事实在后面的“卷积化滑动窗口”和 FCN 分割中都会用到。

#### 1D 与 3D 卷积

卷积不限于二维图像。心电图等一维信号可以用一维卷积（例如长度 14 的信号用 16 个长度 5 的核，得到 $10\times16$）；CT 等三维体数据或视频（时间作为第三维）可以用三维卷积，滤波器为 $f\times f\times f\times n_C$。

### 可视化

卷积层到底在“看”什么？

- **第一层权重**：鱼书 SimpleConvNet 第一层有 30 个 $5\times5$ 的单通道滤波器，可以直接画成灰度图。学习前是随机噪声；学习后变成有规律的图案，如从白到黑渐变的滤波器（响应竖直或水平**边缘**）和含块状区域的滤波器（响应**斑块**，blob）。
- **深层**：Zeiler & Fergus 等的可视化研究表明，层越深提取的信息越抽象。在 8 层的 AlexNet 中，第 1 层对边缘和斑块响应，第 3 层对纹理响应，第 5 层对物体部件响应，最后的全连接层对物体类别（狗、车）响应。

这与人理解事物“含义”的方式类似：从简单形状逐渐组合成高级概念。

### 经典网络

- **LeNet**（LeCun, 1998）：最早的 CNN，用于手写数字识别，约 6 万参数。连续的卷积层与子采样层，最后接全连接层。与现在的 CNN 相比，激活函数用 sigmoid（现在用 ReLU），用子采样（subsampling）缩小特征图（现在主流是最大池化）。
- **AlexNet**（Krizhevsky et al., 2012）：引发深度学习热潮的网络，结构与 LeNet 基本相同，约 6000 万参数。不同点：激活函数用 ReLU；使用局部响应归一化（LRN，后来被证明作用不大）；使用 Dropout。更大的变化在环境：大规模数据（ImageNet）可得，擅长大规模并行计算的 GPU 普及。**大数据和 GPU 是深度学习发展的原动力**。

<br>

> [!NOTE]+ 衔接
>
> 卷积的尺寸公式、实现与反向已经对齐。下一步自然是把网络加深。但直接再堆很多层时，训练误差可能不降反升。下一节先说明加深的好处，再介绍让深网络训得动（残差）、算得起（Inception、深度可分卷积）的结构。加 Dropout 压的是方差，修不了训练误差上升。

<br>

## 深度卷积（Deep CNN Architectures）

### 加深的动机

单层卷积可以堆很多层，但训练误差可能不降反升，计算量也会涨得很快。残差给反向一条捷径；Inception 和深度可分卷积用结构换计算量；迁移学习把 ImageNet 上训好的骨干接到新任务上。在讨论这些结构之前，先说明为什么要加深。

加深的重要性在理论上尚不透彻，但有几点可以从实验和直觉解释：

1. **参数更少**：如上节所示，堆叠小核可以用更少的参数达到同样的感受野。
2. **表现力更强**：每层之间夹着 ReLU，非线性函数叠加得越多，能表示的函数越复杂。
3. **学习更高效**：网络分层提取信息，浅层只需学边缘这样的简单模式。识别“狗”时，浅层网络必须一下子理解狗的所有特征，狗的种类与拍摄环境千差万别，需要大量多样的数据；深层网络把问题分解，第一层只学边缘——包含边缘的图像远多于包含狗的图像，边缘的模式也简单得多——后一层再利用边缘信息学更高级的模式，每层要学的问题都变简单了。

注意：近年的深层化是由大数据和计算能力支撑的，没有这些，加深层也学不好。

> [!EXAMPLE]+ 例10 鱼书的深层 CNN（MNIST）
>
> 参考 VGG 搭建的网络：卷积层全部用 $3\times3$ 小滤波器，通道数依次为 16、16、32、32、64、64，每两层卷积后接池化；全连接层后接 Dropout；He 初始化；Adam 优化。
>
> 测试集识别率 $99.38\%$，错误率仅 $0.62\%$。被识别错的图像连人也难以判断：“1”和“7”、“0”和“6”、“3”和“5”最容易混淆。
>
> 截至 2016 年 6 月，MNIST 上的最高识别率为 $99.79\%$，所用的 CNN 并不深（2 个卷积层 + 2 个全连接层）：手写数字识别这类简单任务不需要很强的表现力，加深的好处不大；而大规模一般物体识别问题复杂，加深层才大有裨益。进一步提高精度的技巧有集成学习、学习率衰减和数据增强（Data Augmentation），其中数据增强方法简单但效果显著。

### ImageNet 与 ILSVRC

ImageNet 是拥有超过 100 万张带标签图像的数据集，每年举办的 ILSVRC 大赛中有一项是 1000 类分类，以 top-5 错误率（前 5 个预测中都不含正确类别的比例）评价。2012 年 AlexNet 以压倒性优势胜出（top-5 错误率约 $15\%$，第二名约 $26\%$），此后冠军一直是深度学习方法，且网络越来越深。2015 年超过 150 层的 ResNet 把错误率降到 $3.5\%$，据说超过了普通人的识别能力。

### VGG

VGG（Simonyan & Zisserman, 2014）结构非常规整：

- 卷积层全部是 $3\times3$、步幅 1、same 填充；池化全部是 $2\times2$、步幅 2；
- 重复“卷积层堆叠 2 到 4 次，再用池化把空间减半”，通道依次为 64、128、256、512、512；
- 有权重的层共 16 层（VGG-16）或 19 层（VGG-19），VGG-16 约 1.38 亿参数，大部分在最后的全连接层。

VGG 在 2014 年 ILSVRC 中获得第 2 名（第 1 名是 GoogLeNet），性能稍逊，但因结构简单、应用性强，很多工程师喜欢以它为基础。

### 残差

> [!INFO]+ 残差
>
> 残差块（He et al., 2016）的输出是 $y=F(x)+x$。再往上堆层时，若训练误差不降反升，原因通常是优化，不是过拟合。捷径给反向一条不经过 $F$ 的通路；若最优映射接近恒等，$F$ 只需学接近 0。

<br>

#### 退化问题

理论上，更深的网络至少不会比浅的差：多出来的层只要学成恒等映射即可。但实验中，普通网络（plain network）层数超过一定程度后，**训练误差**反而上升。这不是过拟合（过拟合是训练误差低、测试误差高），而是**优化困难**：让一串非线性层精确地学出恒等映射并不容易，梯度也会在长链上消失或爆炸。

#### 残差块

ResNet（微软研究院）引入**快捷连接**（shortcut / skip connection）：把输入 $x$ 跳过两层卷积，直接加到输出上。原来两层卷积的输出 $F(x)$ 变成 $F(x)+x$：

$$
y=F(x)+x.
$$

用 CS230 的记号，对两层的残差块：

$$
a^{[l+2]}=g\big(z^{[l+2]}+a^{[l]}\big).
$$

为什么好学：若最优映射接近恒等，只需让 $F$ 学成接近 0。权重衰减本来就会把 $W^{[l+2]}$、$b^{[l+2]}$ 往 0 推，此时 $a^{[l+2]}=g(a^{[l]})=a^{[l]}$（ReLU 下 $a^{[l]}\ge 0$）。所以加一个残差块，**最坏也不会让网络变差**，好的话还能学到有用的东西。

反向传播时梯度可以“无衰减”地传递：

$$
\frac{\partial J}{\partial x}=\frac{\partial J}{\partial y}\cdot\left(1+\frac{\partial F}{\partial x}\right).
$$

即使 $F$ 的雅可比很小，仍有 $1$ 这条路：上游梯度原封不动地传向下游，不必担心梯度在深层中变小。这与 LSTM 的 $c^{\langle t\rangle}=c^{\langle t-1\rangle}+\cdots$ 是同一思想：**加法路径保梯度**。实验表明，ResNet 即使加深到 150 层以上，识别精度仍持续提高。

- 残差块学习 $F(x)$，$F$ 与 $x$ 通道或尺寸不同时，用 $1\times1$ 卷积或步幅卷积做投影 $W_s x$。
- 恒等捷径使反向梯度多一条完整通路，减轻消失。

```python
def residual_block_forward(X, conv1, conv2):
    """简化的两层 3x3 残差块，假设空间尺寸与通道已对齐。"""
    Z1, _ = conv2d_forward(X, conv1["W"], conv1["b"], stride=1, padding=1)
    A1 = np.maximum(Z1, 0)
    Z2, _ = conv2d_forward(A1, conv2["W"], conv2["b"], stride=1, padding=1)
    return np.maximum(Z2 + X, 0)
```

PyTorch 版的 BasicBlock（ResNet-18/34 的构件），同时处理 identity shortcut 与 convolutional shortcut：

```python
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:          # 形状变了：捷径上用 1x1 卷积投影
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(c_out))
        else:                                     # 形状不变：恒等捷径
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))   # 先相加，再 ReLU

blk = BasicBlock(64, 128, stride=2)
print(blk(torch.randn(2, 64, 56, 56)).shape)        # torch.Size([2, 128, 28, 28])
```

#### Bottleneck 与实现细节

- ResNet-50/101 用 bottleneck：$1\times1$ 降维 $\rightarrow 3\times3 \rightarrow 1\times1$ 升维，降低 $3\times3$ 卷积的通道成本。
  - 例：输入通道 $C=256$，瓶颈宽 $C/4=64$；
  - 每个位置的乘加从 $9\cdot 256\cdot 256\approx 59$ 万降到 $256\cdot 64 + 9\cdot 64\cdot 64 + 64\cdot 256\approx 7$ 万。
- 降采样块：第一层卷积步幅 2；捷径同步用步幅 2 的 $1\times1$ 卷积。
- 作业：identity shortcut（形状不变，直接相加）与 convolutional shortcut（捷径上有 $1\times1$ 卷积）必须分支清楚，否则相加时 shape 不对。

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

- ResNet-18 的阶段：conv1 后 $56\times56$；layer1 保持；layer2–4 各把空间减半、通道倍增。
- 实现时先让一个 block 的梯度检查通过，再堆 stage。

### Inception 与深度可分

> [!INFO]+ Inception 与深度可分
>
> Inception（Szegedy et al.）在同一层并行 $1\times1$、$3\times3$、$5\times5$ 和池化，再沿通道拼接，让网络自己选尺度。$5\times5$ 之前先用 $1\times1$ 降维，否则通道乘加过贵。MobileNet 的深度可分卷积先逐通道做空间卷积，再用 $1\times1$ 混合通道，参数约为普通卷积的 $1/C+1/f^{2}$。

<br>

#### Inception（GoogLeNet）

设计卷积层时要决定用 $1\times1$、$3\times3$ 还是 $5\times5$，要不要池化。Inception 的思路是：**全都要**，在同一层并行使用多种大小的滤波器和池化，再把结果沿通道拼接，让网络自己学习用哪种尺度。GoogLeNet 以 Inception 模块为构件，不仅在纵向上有深度，在横向上也有“宽度”，获得了 2014 年 ILSVRC 第 1 名。

问题是计算量。

> [!EXAMPLE]+ 例11 $1\times1$ 瓶颈层节省的计算量
>
> 输入 $28\times28\times192$，用 32 个 $5\times5$ same 卷积得到 $28\times28\times32$。每个输出要做 $5\times5\times192$ 次乘法，共有 $28\times28\times32$ 个输出：
>
> $$
> 28\cdot28\cdot32\times5\cdot5\cdot192\approx 1.2\times10^{8}.
> $$
>
> 先用 16 个 $1\times1$ 卷积降到 $28\times28\times16$（称为瓶颈层，bottleneck layer），再做 32 个 $5\times5$ 卷积：
>
> $$
> 28\cdot28\cdot16\times192\approx 2.4\times10^{6},\qquad
> 28\cdot28\cdot32\times5\cdot5\cdot16\approx 1.0\times10^{7},
> $$
>
> 合计约 $1.24\times10^{7}$，降为原来的十分之一。只要瓶颈层不压得太狠，对性能影响不大。

- 一般地，参数从 $25 C C'$ 变为 $C n_{\mathrm{bot}}+25 n_{\mathrm{bot}} C'$。
- 池化分支后也要接 $1\times1$ 卷积，否则池化保持通道数（192），会让拼接后的通道过多。
- GoogLeNet 还在中间层接了辅助分类器（side branch），用中间特征也做预测，起正则作用。
- 作业不要求手写 Inception，但应能指出：并行分支的空间尺寸必须一致才能拼接，因此 same padding 与步幅要匹配。

#### 深度可分卷积（MobileNet）

MobileNet 面向手机等算力有限的设备，把标准卷积分解为两步：

- **depthwise**（逐通道卷积）：每个输入通道单独用一个 $f\times f$ 核卷积，输出通道数不变；
- **pointwise**（逐点卷积）：用 $1\times1$ 卷积混合通道，得到想要的输出通道数。

计算量从 $f^{2} C_{\mathrm{in}} C_{\mathrm{out}}$（每个输出位置）降到 $f^{2} C_{\mathrm{in}}+C_{\mathrm{in}} C_{\mathrm{out}}$，比值为

$$
\frac{f^{2}C_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}}}{f^{2}C_{\mathrm{in}}C_{\mathrm{out}}}=\frac{1}{C_{\mathrm{out}}}+\frac{1}{f^{2}}.
$$

> [!EXAMPLE]+ 例12 深度可分卷积的计算量
>
> 输入 $6\times6\times3$，用 5 个 $3\times3$ 滤波器（valid）得到 $4\times4\times5$。
>
> - 标准卷积：$(3\cdot3\cdot3)\times(4\cdot4\cdot5)=2160$ 次乘法。
> - depthwise：3 个 $3\times3$ 核，输出 $4\times4\times3$，$(3\cdot3)\times(4\cdot4\cdot3)=432$ 次；pointwise：5 个 $1\times1\times3$ 核，$(1\cdot1\cdot3)\times(4\cdot4\cdot5)=240$ 次；合计 $672$ 次。
> - 比值 $672/2160\approx 0.31=\frac{1}{5}+\frac{1}{9}$。实际中 $C_{\mathrm{out}}$ 常为几百，$f=3$ 时约降为 $1/9$，快近 10 倍。

PyTorch 用 `groups` 参数实现 depthwise：`groups=C_in` 表示每个输入通道单独一组、各用自己的核。

```python
def depthwise_separable(c_in, c_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False),  # depthwise
        nn.BatchNorm2d(c_in), nn.ReLU6(),
        nn.Conv2d(c_in, c_out, 1, bias=False),                                       # pointwise
        nn.BatchNorm2d(c_out), nn.ReLU6(),
    )

count = lambda m: sum(p.numel() for p in m.parameters() if p.ndim == 4)   # 只数卷积核
std = nn.Conv2d(256, 256, 3, padding=1, bias=False)
print(count(std), count(depthwise_separable(256, 256)))   # 589824 vs 67840，约 0.115 = 1/256 + 1/9
```

- MobileNet v2 在此基础上加入**瓶颈块**：先用 $1\times1$ 卷积扩展通道（例如扩大 6 倍），在高维空间做 depthwise 卷积，再用 $1\times1$ 卷积投影回低维，并加残差连接。扩展让网络在块内学到更丰富的函数，投影让块之间传递的激活保持很小，节省内存。
- depthwise 本身不混合通道；没有随后的 $1\times1$，网络无法组合红/绿边缘。
- EfficientNet 进一步研究如何同时缩放输入分辨率 $r$、深度 $d$、宽度 $w$，在给定算力下取得最好效果。
- DenseNet：每层的输入是前面所有层输出的拼接，特征复用更强、参数更省，但拼接占用显存；公开大纲列为选读。

### 迁移学习

> [!INFO]+ 迁移学习
>
> 迁移学习：在数据量大的源任务上先训练骨干，再换上目标任务的输出头。低层核往往接近边缘与纹理检测，目标样本少时只训练头；样本多了再解冻后几层，并用更小的学习率。源域与目标域相差很大时，仍可作初始化，但应更快解冻或改用领域内权重。

<br>

在 ImageNet 上训练一个大网络要花几周，而别人已经训练好并公开了权重。实践中常把这些权重复制到自己的网络作为初始值，再在新数据集上继续训练（fine-tuning）。手头数据少时，迁移学习尤其有效。ImageNet 预训练的低层核接近 Gabor 式边缘检测器，对多数自然图像可复用。

步骤：

1. 加载预训练骨干，去掉原分类头（例如 ImageNet 的 1000 类 softmax）。
2. 加任务头，随机初始化。
3. 目标数据少：冻结骨干，只训头。一个加速技巧是把训练集图像先过一遍冻结的骨干，把输出特征存到磁盘，之后只在这些特征上训练一个浅层分类器。
4. 目标数据中等：冻结前面若干层，训练后面几层和头。
5. 目标数据多：解冻全部或后几段，用预训练权重初始化，学习率比从头训练小一个数量级。

- 医学影像、卫星图与 ImageNet 域差大：仍常从预训练起步，但应更快解冻，或改用领域内预训练。
- 迁移不是免费性能：输入通道数不同（灰度、多光谱）时，第一层核需要适配或重训。
- 冻结骨干时，输入预处理必须与预训练一致（尺寸、均值、缩放），否则等于给网络看另一种分布。
- CS230 作业用 MobileNet 做迁移学习：冻结卷积骨干；替换并训练分类头；必要时用小学习率微调后几层。

PyTorch 中三档数据量对应三种冻结方式（以 torchvision 的 ResNet-18 为例）：

```python
from torchvision import models

def build_transfer(num_classes, mode="small"):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, num_classes)       # 换新头，随机初始化
    if mode == "small":            # 冻结全部骨干，只训头
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith("fc.")
    elif mode == "medium":         # 再解冻最后一个 stage
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith(("layer4.", "fc."))
    # mode == "large"：全部可训，用预训练权重初始化
    return net

net = build_transfer(5, "medium")
opt = torch.optim.Adam([
    {"params": net.fc.parameters(), "lr": 1e-3},               # 新头学习率大
    {"params": net.layer4.parameters(), "lr": 1e-4},           # 解冻层小一个数量级
])
```

- 冻结骨干时，骨干里的 BatchNorm 仍会在 `train()` 模式下更新运行均值。数据很少时，可以在训练时把这些 BN 层单独设成 `eval()`，保持预训练统计量。
- 预处理要与预训练一致：ImageNet 权重配 224 输入和上面 `Normalize` 的均值方差。
- 第 3 步“先把特征存盘”的写法：`with torch.no_grad(): feats = backbone(x)`，存下 `feats` 后只训练一个线性层。

### 数据增强

计算机视觉几乎总是缺数据，数据增强（data augmentation）对视觉任务几乎总有帮助：

- **几何变换**：镜像翻转、随机裁剪、小角度旋转、平移、缩放、剪切（shearing）、局部扭曲；
- **颜色变换**：对 RGB 通道加不同的偏移（模拟光照变化），PCA 颜色增强（AlexNet 论文）；亮度、对比度变化；
- **其他**：加噪声、随机遮挡部分区域。

只有在不改变标签语义时才能使用（翻转对文字或左右不对称的场景无效）。大规模训练时，常用 CPU 线程实时做增强，同时 GPU 训练，两者并行。

### 计算加速

深度学习需要大量运算。在 AlexNet 的前向计算中，**卷积层的时间占 GPU 总时间的 95%、CPU 的 89%**。卷积归根结底是大量乘积累加运算，所以加速的核心是高效地做大规模乘积累加。

- **GPU**：GPU 擅长大规模并行数值计算（CPU 擅长连续的复杂计算）。用 16 核 Xeon CPU 训练 AlexNet 要 40 天以上，用 GPU 缩短到 6 天左右，再用 NVIDIA 为深度学习优化的 cuDNN 库还能更快。大多数框架基于 NVIDIA 的 CUDA 开发环境。im2col 把卷积汇总成大矩阵乘法，正好发挥 GPU 的优势。
- **分布式学习**：在多个 GPU 或多台机器上并行。TensorFlow、CNTK 等框架重视分布式学习；基于 TensorFlow，100 个 GPU 相比 1 个 GPU 约快 56 倍，原本 7 天的训练 3 小时完成。难点在于机器间通信与数据同步。
- **降低数值精度**：内存容量和总线带宽也会成为瓶颈。神经网络对噪声稳健，并不需要很高的数值精度：16 位半精度浮点数（half float）也能顺利训练，NVIDIA Pascal 架构开始支持 16 位运算，有望比上一代快约 2 倍。更激进的 Binarized Neural Networks 用 1 位表示权重和中间数据。位数缩减对嵌入式部署尤其重要。现代训练常用混合精度（权重主副本用 32 位，前向反向用 16 位）。

NumPy 中可以直接看到半精度的局限：表示范围窄，小梯度会下溢为 0。

```python
print(np.finfo(np.float16))                       # max ≈ 65504，最小正规数 ≈ 6.1e-5
g = np.float32(1e-8)
print(np.float16(g))                               # 0.0：直接下溢
print(np.float32(np.float16(g * 1024)) / 1024)     # ≈ 1e-8：先放大存成 fp16，再在 fp32 里缩回
```

这就是 loss scaling 的原理：把损失乘一个大数 $S$ 再反向，梯度整体放大、不再下溢；更新前再除以 $S$。PyTorch 的自动混合精度（AMP）把两件事都自动化了：

```python
device = "cuda"
scaler = torch.amp.GradScaler(device)
for x, y in loader:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    with torch.autocast(device_type=device, dtype=torch.float16):   # 矩阵乘、卷积用 fp16
        loss = F.cross_entropy(model(x), y)                         # softmax、归约等自动留 fp32
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()      # 损失乘 S
    scaler.unscale_(opt)               # 裁剪前先除回 S
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                   # 梯度出现 inf/nan 时跳过这一步
    scaler.update()                    # 动态调整 S
```

- 支持 bfloat16 的 GPU（A100 及之后）可以用 `dtype=torch.bfloat16`：它与 fp32 的指数位相同、不会下溢，不需要 `GradScaler`。
- 单机多卡用 `torch.nn.parallel.DistributedDataParallel`：每张卡一份模型、各算一部分 batch，反向时对梯度做 all-reduce 平均。

> [!NOTE]+ 衔接
>
> 卷积骨干输出的是空间特征图。到目前为止的分类头假定一张图只有一个标签。下一节改成检测：输出个数不固定，每个物体对应一个类别和一个框。

<br>

## 目标检测（Object Detection）

### 任务

视觉任务按输出的复杂程度可以分为三级：

- **图像分类**：整张图一个类别（“这是车”）。
- **分类 + 定位**：一张图一个主要物体，输出类别和它的边界框。
- **目标检测**（object detection）：一张图中可能有多个、多类物体，要标出每一个的类别和边界框 $(b_x,b_y,b_h,b_w)$。

检测比分类难得多：输出的个数不固定，并且要同时做分类和定位。骨干仍是上一节的卷积特征图。

### 分类与定位

先看最简单的情况：每张图最多一个物体，类别为行人、汽车、摩托车或背景。约定图像左上角为 $(0,0)$、右下角为 $(1,1)$，网络输出

$$
y=\big[p_c,\ b_x,\ b_y,\ b_h,\ b_w,\ c_1,\ c_2,\ c_3\big]^{\top},
$$

其中 $p_c$ 表示是否有物体，$(b_x,b_y)$ 是框的中心，$b_h,b_w$ 是高和宽，$c_1,c_2,c_3$ 是类别。损失按 $p_c$ 分情况：

- $y_1=p_c=1$（有物体）：所有 8 个分量都计入损失；
- $y_1=0$（背景）：只计 $(\hat{y}_1-y_1)^2$，其余分量无意义，不计入。

实践中，$p_c$ 用逻辑回归损失，框坐标用平方误差，类别用 softmax 交叉熵。

**特征点检测**（landmark detection）是同一思路的推广：让网络输出若干关键点坐标，例如人脸的 64 个关键点（眼角、嘴角、下巴轮廓），输出 $1+2\times64=129$ 个数（是否有脸 + 每个点的 $x,y$）。可用于表情识别、AR 滤镜、人体姿态估计。所有训练图像中关键点的编号顺序必须一致。

### 滑动窗口

滑动窗口（sliding window）检测：先训练一个对裁剪好的小图做分类的 CNN，再在大图上按多种尺度、以一定步幅切出矩形窗口，每个窗口送进分类器，判断里面有没有物体。

- 在多尺度窗口上跑分类器，相邻窗口特征重复，计算量巨大；步幅大则定位粗糙，步幅小则更慢。
- **卷积化**：把分类器中的全连接层改写成卷积，一次前向就能得到空间图上每个位置的分类结果，等价于所有对齐的滑动窗口，重叠部分的计算被共享。

> [!EXAMPLE]+ 例13 滑动窗口的卷积实现
>
> 分类器输入 $14\times14\times3$：16 个 $5\times5$ 卷积 → $10\times10\times16$ → $2\times2$ 最大池化 → $5\times5\times16$ → FC 400 → FC 400 → softmax 4 类。
>
> 把第一个 FC 改成 400 个 $5\times5\times16$ 的卷积（输出 $1\times1\times400$），第二个 FC 改成 400 个 $1\times1$ 卷积，softmax 层改成 4 个 $1\times1$ 卷积（输出 $1\times1\times4$）。
>
> 现在输入 $16\times16\times3$ 的图，同一个网络输出 $2\times2\times4$：4 个位置分别对应以步幅 2 滑动的 4 个 $14\times14$ 窗口。输入 $28\times28\times3$ 时输出 $8\times8\times4$，一次前向完成 64 个窗口。这也解释了检测网络为何是全卷积的。

- 仍难以精确回归框，且宽高比不灵活：窗口位置不一定正好框住物体。

### 两阶段检测：R-CNN 系列

R-CNN（Regions with CNN）的思路是**先找候选区域，再分类**：

1. 输入图像；
2. 提取约 2000 个“看起来像物体”的候选区域（region proposals），原论文用 Selective Search 等传统方法（例如基于图像分割的色块）；
3. 把每个候选区域变形为固定大小，用 CNN 计算特征；
4. 用分类器（原论文用 SVM）判断类别，并修正框。

R-CNN 很慢，因为每个候选区域都要单独跑一次 CNN。Fast R-CNN 用卷积化的方法一次算出全图特征，再对各区域取特征；Faster R-CNN 进一步用一个 CNN（区域提议网络，RPN）来产生候选区域，整个流程都由 CNN 完成，可以高速处理。两阶段方法定位较细，但通常比一阶段方法慢。

### IoU 与 NMS

> [!INFO]+ IoU 与 NMS
>
> IoU（intersection over union）是两框交集面积除以并集面积。检测里用它判定正样本，mAP 也用它，但阈值含义不同。一张图通常会对同一物体给出多个高度重叠的框；NMS（non-maximum suppression）只保留分数最高的那个。Soft-NMS 降低重叠框的分数，而不是直接删掉，密集目标少被误杀。

<br>

**交并比**衡量预测框与真实框的重合程度：

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}\in[0,1].
$$

- 完全重合为 1，不相交为 0。按惯例 $\mathrm{IoU}\ge 0.5$ 视为定位正确（更严格可取 0.6、0.7）。检测正样本通常要求与真值 IoU 超过 $0.5$。

**非极大值抑制**（NMS）：同一物体往往被多个格子或窗口检测到，产生多个重叠框，NMS 只保留最可信的那个。

1. 丢掉置信度低于阈值（如 0.6）的框。
2. 选出剩余框中置信度最高的，作为一个检测结果输出；删除所有与它 IoU 高于阈值（如 $0.5$）的同类框。
3. 重复第 2 步直至没有剩余框。多类别时对每一类分别做 NMS。

- NMS 默认一个物体对应一个框；物体密集重叠时会误删。
- Soft-NMS 等变体降低重叠框的分数，而不是直接删除。

### YOLO

> [!INFO]+ YOLO
>
> YOLO（You Only Look Once, Redmon et al., 2016）把图像划成 $S\times S$ 网格，物体中心落入的格子同时预测框和类别。一次前向出全部检测结果，延迟低于先提候选框再分类的两阶段方法。作业要求格子、anchor 与 NMS 对应正确，不要求复现当时精度。

<br>

#### 网格预测

YOLO 把“分类 + 定位”的输出向量用在网格的每个格子上：

- 把图划分为 $S\times S$ 网格（课上用 $3\times3$ 演示，实际常用 $19\times19$）。
- **物体中心落入的格子负责预测该物体**，即使物体跨越多个格子，也只分配给一个格子。
- 每个格子输出一个 $[p_c,b_x,b_y,b_h,b_w,c_1,\dots,c_C]$ 向量。$3\times3$ 网格、3 类时，输出为 $3\times3\times8$。
- $(b_x,b_y)$ 相对于所在格子表示，在 $[0,1]$ 内；$b_h,b_w$ 相对于格子边长，可以大于 1。
- 由于是一个卷积网络一次前向得到所有格子的输出，速度很快，可以实时检测。

更一般地，每个格子预测 $B$ 个框及置信度和 $C$ 类概率，输出张量形状约为

$$
S\times S\times\big(B\cdot(5+C)\big),
$$

或在后来的版本中把类别与框解耦。

#### Anchor box

一个格子只能检测一个物体。若行人和汽车的中心落在同一格子里怎么办？**Anchor box**（锚框）事先给定一组不同形状的框（如一个瘦高的适合行人，一个扁宽的适合汽车），每个格子对每个 anchor 各输出一组预测。2 个 anchor 时，$3\times3$ 网格的输出变为 $3\times3\times16$（$2\times8$）。

- Anchor 是事先给定的宽高比和尺度，不是网络学出来的（可以用 k-means 对训练集的框聚类得到）。
- 匹配：每个真值框分配给中心所在的格子，以及该格子中与它 IoU 最高的 anchor。
- 除了处理重叠，anchor 还让每组输出专门化：瘦高的 anchor 输出专门学行人。
- 两个物体中心在同一格且形状又和同一个 anchor 最匹配，或物体数超过 anchor 数，算法处理不好，但这种情况很少。

> [!EXAMPLE]+ 例14 作业中的 YOLO 输出
>
> 输入 $608\times608\times3$，网格 $19\times19$，5 个 anchor，80 类（COCO）。每个 anchor 输出 $5+80=85$ 个数，所以输出形状为 $19\times19\times5\times85$，展平最后两维为 $19\times19\times425$。
>
> 后处理：对每个 anchor 计算类别得分 $p_c\times c_i$，取最大的类及其得分；丢掉得分低于阈值（如 0.6）的框；对剩下的框按类做 NMS（IoU 阈值 0.5）。

> [!NOTE]+ YOLO 框参数化（bounding-box parameterization）
>
> 设格子左上角坐标为 $(c_x,c_y)$，anchor 宽高为 $(p_w,p_h)$，网络输出原始数 $t_x,t_y,t_w,t_h$。常见解码为
>
> $$
> b_x=\sigma(t_x)+c_x,\qquad
> b_y=\sigma(t_y)+c_y,\qquad
> b_w=p_w e^{t_w},\qquad
> b_h=p_h e^{t_h}.
> $$
>
> $\sigma$ 把中心限制在负责的格子附近，指数保证宽高为正。实现还会按网格或图像尺寸归一化。不同 YOLO 版本的具体参数化不同，考试若给了解码式，应按题目版本计算。
>
> 一阶段检测器（one-stage detector）如 YOLO 直接密集预测类别与框，速度快。两阶段检测器（two-stage detector）如 Faster R-CNN 先产生区域提案（region proposal），再分类与精修框，通常更慢但定位更细。

#### 训练

训练损失包含三部分：

- 有物体格子的坐标回归（平方或平滑 $L_1$）；
- 置信度（有物体 / 无物体，后者权重更小，否则大量背景格子淹没梯度）；
- 分类交叉熵。

- 每个真值框只匹配一个格子（中心所在格）和一个 anchor（IoU 最大者）。
- 坐标：sigmoid 把中心约束在格内；指数把 anchor 宽高缩放到正数。
- 损失是多项之和；权重不匹配会出现“分类还行、框乱飘”或反过来。
- Anchor 与特征图步幅共同决定最小可检物体。物体小于一个格子时，中心仍只能进一个格，但回归难度上升。

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

上面逐对算 IoU，是 $O(N^{2})$ 次 Python 调用。向量化写法一次算出 $N\times M$ 的 IoU 矩阵，NMS 每轮只需一行比较：

```python
def iou_matrix(A, B):
    """A: (N, 4), B: (M, 4)，返回 (N, M)。"""
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return inter / (area(A)[:, None] + area(B)[None, :] - inter)

def nms_vectorized(boxes, scores, iou_thr=0.5):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return keep
```

torchvision 提供 C++/CUDA 实现，可以用来对拍。YOLO 作业里按类别分别做 NMS，对应 `batched_nms`：

```python
from torchvision.ops import nms, batched_nms, box_iou

boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 8], [20, 20, 30, 30]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.7])
classes = torch.tensor([0, 0, 1])
print(box_iou(boxes[:1], boxes[1:2]))               # tensor([[0.8000]])，即例 15
print(nms(boxes, scores, iou_threshold=0.5))        # tensor([0, 2])
print(batched_nms(boxes, scores, classes, 0.5))     # 不同类别的框互不抑制
```

> [!EXAMPLE]+ 例15 IoU 与 NMS 的数值
>
> 框 $A=[0,0,10,10]$，$B=[0,0,10,8]$，交集面积 $80$，并集 $100+80-80=100$，$\mathrm{IoU}=0.8$。若 NMS 阈值为 $0.5$ 且二者同类，低分框被删。若其实是并排两辆车，阈值过低会误删，这是密集场景换 Soft-NMS 或更小阈值的原因。NMS 阈值与 IoU 评估里的 $0.5$ 不是同一件事：一个用于后处理，一个用于算 mAP。

<br>

### 评估：AP 与 mAP

- mAP：对每类按置信度排序，算 precision-recall 曲线下面积，再对类平均。
- IoU 阈值集合（如 $0.5$ 到 $0.95$）决定定位严格程度。

> [!NOTE]+ AP 与 mAP（average precision and mean average precision）
>
> 对某一类别，把全部预测框按 confidence score 从高到低排序。每加入一个预测，按它是否与未匹配真值达到指定 IoU 阈值更新 TP 或 FP，由此得到 precision-recall curve。平均精度（average precision, AP）是该曲线下的插值面积：
>
> $$
> \mathrm{AP}=\int_{0}^{1}p_{\mathrm{interp}}(r)\,\mathrm{d}r,
> \qquad
> \mathrm{mAP}=\frac{1}{C}\sum_{c=1}^{C}\mathrm{AP}_c.
> $$
>
> `AP@0.5` 表示 IoU 阈值固定为 $0.5$。COCO 的 `AP@[.5:.95]` 会在 $0.50,0.55,\ldots,0.95$ 上求平均，更严格。NMS 是推理后处理；AP/mAP 是评估指标，二者不能互换。

<br>

> [!NOTE]+ 衔接
>
> 检测给出的是矩形框，框里既有物体也有背景。很多应用（医学影像中肿瘤的轮廓、自动驾驶中可行驶的路面）需要知道每个像素属于什么。下一节把输出细化到像素级。

<br>

## 语义分割（Semantic Segmentation）

### 任务

**语义分割**对图像中的每个像素分类（道路、车辆、建筑、行人……），训练数据是逐像素着色的标签图。实例分割还要区分同类的不同个体，超出 CS230 作业范围。

最朴素的做法是对每个像素，取以它为中心的一块区域送入分类网络，但这需要做“像素数”次前向，相邻区域大量重复计算。

### FCN

**FCN**（fully convolutional network，全卷积网络）只用卷积层：把分类网络的全连接层替换成等价的卷积层（见“全连接层可以写成卷积”），空间结构就能一直保留到输出。一次前向就能对所有像素分类。

但经过多次下采样，特征图已经很小。FCN 最后加入**扩大空间**的处理，把小特征图一下子放大回输入大小。这个放大用双线性插值，通过**转置卷积**（transposed convolution，也称去卷积、逆卷积）实现。

#### 转置卷积

普通卷积让空间变小；转置卷积让空间变大。做法是：输入的每个元素乘以整个滤波器，结果“盖”到输出的对应位置上（步幅决定移动距离），重叠部分相加，最后去掉填充部分。例如 $2\times2$ 的输入、$3\times3$ 的滤波器、$p=1$、$s=2$，得到 $4\times4$ 的输出。

输出边长 $=(n-1)s-2p+f$（PyTorch 还可加 `output_padding`）。按上面“盖章再相加”的描述写成 NumPy，再与 `ConvTranspose2d` 对拍：

```python
def conv_transpose2d(x, k, s=2, p=1):
    """单通道。x: (n, n)，k: (f, f)。"""
    n, f = x.shape[0], k.shape[0]
    full = np.zeros(((n - 1) * s + f, (n - 1) * s + f))
    for i in range(n):
        for j in range(n):
            full[i * s:i * s + f, j * s:j * s + f] += x[i, j] * k   # 盖章，重叠相加
    return full[p:full.shape[0] - p, p:full.shape[1] - p]          # 去掉填充

x = np.array([[1.0, 2.0], [3.0, 4.0]])
k = np.arange(9.0).reshape(3, 3)
up = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
up.weight.data = torch.tensor(k).reshape(1, 1, 3, 3)
out_t = up(torch.tensor(x).reshape(1, 1, 2, 2)).detach().numpy()[0, 0]
print(conv_transpose2d(x, k).shape, np.abs(conv_transpose2d(x, k) - out_t).max())   # (3, 3) 0.0
```

这里 $(2-1)\cdot2-2+3=3$，PyTorch 给出的也是 $3\times 3$；要得到上文的 $4\times 4$，需加 `output_padding=1`（在输出的一侧多补一行一列）。

### U-Net

> [!INFO]+ U-Net
>
> U-Net（Ronneberger et al., 2015）用于像素级分割：编码器下采样抓上下文，解码器上采样恢复空间，skip 把编码器的高分辨率特征在通道维拼到对应解码层。定位主要来自 skip，不是来自最底层语义。skip 是拼接，残差是相加。

<br>

U-Net 的结构像字母 U：

- **编码器**（左半边）：卷积 + 下采样，空间变小、通道变多，抓取“这里是什么”的上下文信息。
- **解码器**（右半边）：转置卷积上采样，空间变大、通道变少，恢复空间分辨率。
- **跳跃连接**（skip connection）：编码器第 $k$ 个下采样前的高分辨率特征，沿通道维拼接到解码器对应的上采样之后。解码器底层只有低分辨率的高级语义（知道“有一只猫”但不知道精确边界），跳跃连接补上了精细的空间细节（边缘在哪里）。
- 最后用 $1\times1$ 卷积把通道变为类别数，输出 $h\times w\times n_{\mathrm{classes}}$，每个像素取 argmax 得到分割图。

注意 skip 是**通道维拼接**，而非相加（与残差不同）。

两层深的小型 U-Net，结构与作业相同，只是更浅：

```python
def double_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU())

class MiniUNet(nn.Module):
    def __init__(self, c_in=3, n_classes=2, base=32):
        super().__init__()
        self.enc1 = double_conv(c_in, base)               # H
        self.enc2 = double_conv(base, base * 2)           # H/2
        self.bottom = double_conv(base * 2, base * 4)     # H/4
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = double_conv(base * 4, base * 2)       # 拼接后通道翻倍
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = double_conv(base * 2, base)
        self.head = nn.Conv2d(base, n_classes, 1)         # 1x1 卷积到类别数

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottom(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))   # skip：通道维拼接
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)                                   # (m, n_classes, H, W) logits

print(MiniUNet()(torch.randn(1, 3, 96, 128)).shape)   # torch.Size([1, 2, 96, 128])
```

输入边长须能被 $2^{\text{下采样次数}}$ 整除，否则上采样回来的尺寸比 skip 少一像素，`torch.cat` 报错。这就是作业里要求对称裁剪或 padding 的原因。

### 损失与评估

- 损失：逐像素交叉熵；类别不平衡时用加权或 Dice。
- 医学分割中正类像素很少，交叉熵会偏向背景。Dice 损失直接优化重叠：

$$
1-\frac{2\sum_p \hat{y}_p y_p}{\sum_p \hat{y}_p+\sum_p y_p+\varepsilon}.
$$

- 评估不能把 IoU 与像素准确率混用：后者会被大背景类主导。
- 上采样：转置卷积，或插值加 $3\times3$ 卷积。
- 作业重点：对称裁剪或 padding，使 skip 两端空间尺寸一致，否则无法 `concatenate`。

```python
def dice_loss(logits, target, eps=1e-6):
    """二类分割。logits, target: (m, 1, H, W)，target 取 0/1。"""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()

def mean_iou(pred, target, n_classes):
    """pred, target: (H, W) 整数类别图。逐类算 IoU 再平均，忽略两边都没有的类。"""
    ious = []
    for c in range(n_classes):
        inter = np.logical_and(pred == c, target == c).sum()
        union = np.logical_or(pred == c, target == c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious))

# 多类逐像素交叉熵：logits (m, K, H, W)，标签 (m, H, W) 为整数
seg_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1.0]))   # 给少数类更大权重
```

常见的组合是 `CrossEntropy + Dice`：交叉熵提供稳定的逐像素梯度，Dice 直接对准重叠率，缓解背景占多数的问题。

> [!EXAMPLE]+ 例16 自动驾驶中的道路场景分割
>
> 基于 CNN 的 SegNet 对道路场景逐像素分割，能较高精度地识别出道路、建筑物、人行道、树木、车辆等。正确识别时刻变化的环境、自由往来的车辆和行人，是自动驾驶中最关键也最困难的技术之一。

<br>

> [!NOTE]+ 衔接
>
> 检测与分割输出的是框和像素类别。人脸验证学的是嵌入空间里的距离，不再用固定 $K$ 类的 softmax。风格迁移甚至不训练网络，只把生成图当作像素变量来优化。这两件事都建立在卷积特征上。

<br>

## 人脸与风格（Face Recognition and Neural Style Transfer）

人脸验证学的是可比较的嵌入，而不是固定 $K$ 类的 softmax：同一人的两张图距离应小于阈值 $\tau$。风格迁移固定预训练 CNN，把生成图 $G$ 当作像素变量。内容损失比较某层特征图；风格损失比较 Gram 矩阵（Gram matrix），即把特征图展成通道 $\times$ 空间后做 $AA^{\top}$，刻画通道之间的相关，不锁定空间布局。

### 人脸验证与识别

> [!INFO]+ Siamese 与 triplet
>
> Siamese 网络：两个共享权重的 CNN 把图像映成同一嵌入空间。triplet loss（FaceNet, 2015）要求锚点离正例比离负例至少近一个间隔 $\alpha$。身份数目会变时，不必重训 $K$ 类 softmax，只要把新嵌入写入库。随机负例大多已经满足间隔，需要半困难负例。

<br>

#### 两个任务

- **验证**（verification）：输入一张图和一个身份（ID），判断是不是这个人。一对一，例如刷脸门禁、手机解锁。
- **识别**（recognition）：库中有 $K$ 个人，输入一张图，判断是其中哪一个（或都不是）。一对多，难得多：若验证的准确率是 $99\%$，在 100 人的库中做识别，出错的机会约放大 100 倍。
- 二者共享嵌入，不共享决策。

#### 一次学习

人脸识别的难点是**一次学习**（one-shot learning）：公司员工每人往往只有一张照片，要据此识别本人。若当作 $K$ 类分类训练 softmax，每类只有一个样本，根本训练不好；而且每来一个新员工，$K$ 就变，要重新训练。

解决办法是学一个**相似度函数**：

$$
d(\text{img}_1,\text{img}_2)=\text{两张图的差异程度}.
$$

若 $d\le\tau$ 判为同一人，否则不是。识别时把输入与库中每张照片比较，取 $d$ 最小者（若都大于 $\tau$ 则判为陌生人）。新员工入职只需把他的照片加入库，不必重新训练。

#### Siamese 网络

用同一个 CNN（共享权重）把两张图分别编码为向量 $f(x^{(i)})$ 和 $f(x^{(j)})$（例如 128 维），定义

$$
d\big(x^{(i)},x^{(j)}\big)=\big\|f(x^{(i)})-f(x^{(j)})\big\|_2^{2}.
$$

训练目标：同一人的 $d$ 小，不同人的 $d$ 大。这就是**度量学习**（metric learning）：直接学样本之间的距离。

#### Triplet loss

FaceNet 的 triplet loss 每次看三张图：锚点 $A$（anchor）、同一人的正例 $P$（positive）、另一个人的负例 $N$（negative）。希望

$$
\|f(A)-f(P)\|_2^{2}+\alpha\le\|f(A)-f(N)\|_2^{2}.
$$

间隔 $\alpha>0$ 是必需的：若 $\alpha=0$，网络把所有图都编码成同一个向量（或零向量），不等式就以 $0\le 0$ 平凡成立。$\alpha$ 要求正负对的距离至少拉开一段。写成损失：

$$
\mathcal{L}(A,P,N)=\max\big(\|f(A)-f(P)\|_2^{2}-\|f(A)-f(N)\|_2^{2}+\alpha,\ 0\big),
\qquad
J=\sum_{i}\mathcal{L}\big(A^{(i)},P^{(i)},N^{(i)}\big).
$$

- 训练集需要每个人有多张照片（例如 1000 人共 1 万张）：每个锚点的正例来自同一身份的另一张图。部署时每人一张即可。
- 嵌入通常 $L_2$ 归一化到单位球面，消除尺度作弊。
- 三元组挖掘很关键。随机选 $A,P,N$ 时，绝大多数三元组已经满足不等式，损失为 0，梯度为 0，学不到东西。应选**难**的三元组，即 $d(A,P)\approx d(A,N)$ 的那些。
  - 简单负例已满足间隔，损失为 0。
  - 困难负例 $\|f(A)-f(N)\|<\|f(A)-f(P)\|$：往往是噪声或标注错。
  - **半困难负例**（semi-hard negative）位于二者之间：比正例远，但还没有远过 $\alpha$，梯度最有信息。
  - 在线挖掘：在一个 mini-batch 内选；离线挖掘：周期性更新。
  - 都要防止坍缩到所有嵌入为零（尺度任意时间隔仍可被满足），$L_2$ 归一化把嵌入钉在球面上。

```python
def triplet_loss(fA, fP, fN, alpha=0.2):
    """f*: (m, d)，已 L2 归一化更好。"""
    pos = np.sum((fA - fP) ** 2, axis=1)
    neg = np.sum((fA - fN) ** 2, axis=1)
    return np.maximum(pos - neg + alpha, 0.0).mean()
```

PyTorch 中，常把编码网络的输出做 L2 归一化，并在 batch 内挑选半困难负例（semi-hard negative：比正例远、但仍落在间隔之内）：

```python
import torch.nn.functional as F

class FaceEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, d)
        self.net = backbone

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)          # 嵌入落在单位球面上

def batch_semihard_triplet(emb, labels, alpha=0.2):
    """emb: (B, d) 已归一化；labels: (B,) 身份。batch 内每个锚点取一个半困难负例。"""
    D = torch.cdist(emb, emb) ** 2                       # (B, B) 平方距离
    same = labels[:, None] == labels[None, :]
    losses = []
    for a in range(len(emb)):
        pos_idx = torch.where(same[a] & (torch.arange(len(emb)) != a))[0]
        neg_idx = torch.where(~same[a])[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        for p in pos_idx:
            d_ap, d_an = D[a, p], D[a, neg_idx]
            semi = d_an[(d_an > d_ap) & (d_an < d_ap + alpha)]
            d_n = semi.min() if len(semi) else d_an.min()     # 没有半困难的就取最难的
            losses.append(F.relu(d_ap - d_n + alpha))
    return torch.stack(losses).mean() if losses else emb.sum() * 0
```

- 一个 batch 里每个身份至少要有 2 张图，否则没有正例。常见采样是每 batch 取 $P$ 个人、每人 $K$ 张（例如 $P=32$、$K=4$）。
- PyTorch 内置 `nn.TripletMarginLoss(margin=0.2)` 用的是**不平方**的距离 $\lVert a-p\rVert_2$，与 FaceNet 公式差一个平方，间隔 $\alpha$ 的数值不能直接照搬。

#### 当作二分类

另一种训练方式：把一对图的嵌入差输入逻辑回归，判断是否同一人：

$$
\hat{y}=\sigma\Big(\sum_{k=1}^{128}w_k\big|f(x^{(i)})_k-f(x^{(j)})_k\big|+b\Big).
$$

也可以把 $|\cdot|$ 换成 $\chi^{2}$ 形式 $\frac{(f_k^{(i)}-f_k^{(j)})^{2}}{f_k^{(i)}+f_k^{(j)}}$。训练数据是成对的图，标签为“同一人 1 / 不同人 0”。二元监督也可以用对比损失（contrastive loss）：同一人的两张图拉近，不同人的两张图推到间隔以外。

部署技巧：库中员工照片的嵌入可以**预先算好存起来**，来人时只需对新图跑一次 CNN。

#### 阈值与错误率

- 验证时比较 $\|f(x)-f(x_{\mathrm{db}})\|$ 与阈值 $\tau$；$\tau$ 在验证集上用 FAR/FRR 或 EER 选择。
- 光照、姿态、年龄会造成嵌入漂移；部署采集协议须与训练分布对齐。

> [!NOTE]+ FAR、FRR 与 EER
>
> 假接受率（false accept rate, FAR）是不同人却被接受的比例；假拒绝率（false reject rate, FRR）是同一人却被拒绝的比例：
>
> $$
> \mathrm{FAR}=\frac{\text{false accepts}}{\text{impostor attempts}},\qquad
> \mathrm{FRR}=\frac{\text{false rejects}}{\text{genuine attempts}}.
> $$
>
> 距离阈值 $\tau$ 变大时，系统更容易接受，FAR 通常上升而 FRR 下降。等错误率（equal error rate, EER）是两条错误率相等处的值，越低越好。安全门禁通常不直接选 EER 阈值，而是在 FAR 必须低于某上限的约束下再压 FRR。

```python
def embed_and_verify(xa, xb, network, tau):
    """network 返回 (d,) 嵌入。"""
    ea, eb = network(xa), network(xb)
    ea = ea / np.linalg.norm(ea)
    eb = eb / np.linalg.norm(eb)
    return float(np.sum((ea - eb) ** 2)) < tau
```

### 神经风格迁移

> [!INFO]+ 风格迁移
>
> 神经风格迁移（Gatys et al., 2016）固定预训练 CNN 的权重，把生成图 $G$ 当作像素变量来优化。内容损失比较某层特征图，风格损失比较 Gram 矩阵（通道相关，不锁定空间布局）。梯度对 $G$ 回传，不更新 $W$。

<br>

#### 任务

给定**内容图** $C$（如一张照片）和**风格图** $S$（如梵高的《星空》），生成一张新图 $G$：内容像 $C$，风格像 $S$。这篇论文（*A Neural Algorithm of Artistic Style*）一经发表就受到广泛关注。

关键在于：$G$ 是**像素变量**，不是网络权重；CNN（如 ImageNet 上预训练的 VGG）冻结不动，只用来提取特征。

#### 代价函数

$$
J(G)=\alpha J_{\mathrm{content}}(C,G)+\beta J_{\mathrm{style}}(S,G).
$$

$\alpha$ 越大越在意内容，$\beta$ 越大越在意风格。算法：随机初始化 $G$（或用内容图加噪声），然后对 $G$ 的像素做梯度下降 $G\leftarrow G-\frac{\partial J}{\partial G}$。

**内容代价**：选一个中间层 $l$（太浅则要求像素级相同，太深则只要求“有只狗”），比较两图在该层的激活：

$$
J_{\mathrm{content}}(C,G)=\frac{1}{2}\|a^{[l](C)}-a^{[l](G)}\|_F^{2}.
$$

激活相近说明两图内容相似。

**风格代价**：“风格”被定义为同一层中**不同通道激活之间的相关性**。例如某通道检测竖条纹，另一通道检测橙色，两者经常同时出现，就说明这种风格里“竖条纹常是橙色的”。把第 $l$ 层激活 $a^{[l]}\in\mathbb{R}^{n_H\times n_W\times n_C}$ 展成 $A^{[l]}\in\mathbb{R}^{n_C\times(n_H n_W)}$，**风格矩阵**（Gram 矩阵）为

$$
G^{[l]}=A^{[l]}A^{[l]\top},\qquad
G_{kk'}^{[l]}=\sum_{i=1}^{n_H}\sum_{j=1}^{n_W}a_{ijk}^{[l]}a_{ijk'}^{[l]}.
$$

$G_{kk'}$ 是通道 $k$ 与 $k'$ 在所有位置上的乘积之和（未中心化的协方差）。Gram 矩阵对空间位置求和，所以**不锁定空间布局**，只刻画纹理统计。分别对 $S$ 和 $G$ 计算，比较：

$$
J_{\mathrm{style}}^{[l]}(S,G)=\frac{1}{(2n_Hn_Wn_C)^{2}}\big\|G^{[l](S)}-G^{[l](G)}\big\|_F^{2}.
$$

归一化常数只影响尺度，可以被 $\beta$ 吸收（有的实现写成 $\frac{1}{2}\|\cdot\|_F^2$ 或 $\frac{1}{4n_C^2(n_Hn_W)^2}\|\cdot\|_F^2$）。用多层效果更好，浅层偏纹理、深层偏大结构：

$$
J=\alpha J_{\mathrm{content}}+\beta\sum_{l}\lambda_l J_{\mathrm{style}}^{[l]}.
$$

- 对 $G$ 做梯度下降，可加 total variation 正则使图像平滑。
- 这是优化问题不是训练问题：每张新图都要从零迭代优化 $G$，与训练分类器的“epoch”不是同一件事。

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
- 迭代不够：风格未融合。$\beta/\alpha$ 过大：内容结构崩坏。

PyTorch 写法的关键只有一行：把生成图设成 `requires_grad=True`，并把**它**（而不是网络参数）交给优化器。

```python
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
for p in vgg.parameters():
    p.requires_grad_(False)                       # 网络冻结

LAYERS = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1",
          "21": "conv4_2", "28": "conv5_1"}

def features(x):
    out = {}
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in LAYERS:
            out[LAYERS[name]] = x
    return out

def gram(F_):                                      # F_: (1, C, H, W)
    _, C, H, W = F_.shape
    A = F_.view(C, H * W)
    return A @ A.T / (H * W)

def style_transfer(content, style, steps=300, alpha=1.0, beta=1e4):
    """content, style: (1, 3, H, W)，已按 ImageNet 均值方差归一化。"""
    fc, fs = features(content), features(style)
    style_grams = {k: gram(v) for k, v in fs.items() if k != "conv4_2"}
    G = content.clone().requires_grad_(True)       # 从内容图出发
    opt = torch.optim.Adam([G], lr=0.02)           # 优化的是像素 G
    for _ in range(steps):
        fg = features(G)
        j_content = F.mse_loss(fg["conv4_2"], fc["conv4_2"])
        j_style = sum(F.mse_loss(gram(fg[k]), style_grams[k]) for k in style_grams) / len(style_grams)
        loss = alpha * j_content + beta * j_style
        opt.zero_grad(); loss.backward(); opt.step()
    return G.detach()
```

`vgg.eval()` 不能省：VGG 本身没有 BN 和 Dropout，但换成带 BN 的骨干时，`train()` 模式会让特征随 batch 统计变化。

> [!NOTE]+ 对照：验证、识别、分类、风格迁移
>
> - $K$ 类 softmax：类别集合固定。
> - 验证：一对一，比距离与 $\tau$。
> - 识别：一对多，在库中找最近邻；库可增员而不重训头。
> - 风格迁移：变量是像素 $G$，CNN 冻结。

<br>

> [!NOTE]+ 衔接
>
> 卷积在空间上共享同一组核。序列数据（文本、语音、时间序列）的结构在时间轴上：下一部分的循环网络让 $W_{aa}$ 在每个时间步重复使用，正如卷积核在每个空间位置重复使用。

<br>

# 序列与语言

卷积在空间上共享权重；这一部分把共享换到时间轴上。文本、语音、音乐、时间序列都是序列数据，它们有两个图像没有的特点：**长度可变**，而且**顺序有意义**。

路线如下：

1. **循环网络**：每个时间步复用同一组权重，用隐状态记住历史；长程梯度会消失，于是引入 GRU、LSTM 的门。
2. **词嵌入**：把离散的词变成可比较的稠密向量，作为循环网络的输入。
3. **注意力**：编码器–解码器把整句压成一个向量，长句信息会丢；注意力让解码器每一步回看源句的所有位置。
4. **Transformer**：完全去掉循环，只用注意力，序列顺序改由位置编码写入。

<br>

## 循环网络（Recurrent Neural Network）

### 动机

#### 序列问题

很多任务的输入或输出是序列：

| 任务 | 输入 $x$ | 输出 $y$ |
| :--- | :--- | :--- |
| 语音识别 | 音频波形 | 文字序列 |
| 音乐生成 | 空（或一个风格编号） | 音符序列 |
| 情感分类 | 一句评论 | 1–5 星 |
| DNA 序列分析 | 碱基序列 | 每段是否编码蛋白质 |
| 机器翻译 | 一种语言的句子 | 另一种语言的句子 |
| 视频动作识别 | 帧序列 | 动作类别 |
| 命名实体识别 | 句子 | 每个词是否是人名 |

记号：$x^{\langle t\rangle}$ 表示序列的第 $t$ 个元素，$T_x$、$T_y$ 是输入、输出序列的长度；第 $i$ 个样本写作 $x^{(i)\langle t\rangle}$，不同样本长度 $T_x^{(i)}$ 可以不同。

> [!EXAMPLE]+ 例1 命名实体识别的数据表示
>
> 句子 “Harry Potter and Hermione Granger invented a new spell.” 有 $T_x=9$ 个词，标签 $y=(1,1,0,1,1,0,0,0,0)$ 标出哪些词属于人名。
>
> 每个词先用词表表示：取一个 10000 词的词表（实际常用 3 万到 5 万，大公司可达百万），“Harry” 若是第 4075 个词，就表示为第 4075 位为 1、其余为 0 的 one-hot 向量 $x^{\langle 1\rangle}\in\mathbb{R}^{10000}$。不在词表中的词用 `<UNK>`（unknown）表示。

#### 为什么不用标准网络

把 9 个 one-hot 向量拼起来送进全连接网络，有两个问题：

- **长度不固定**：不同句子的 $T_x$、$T_y$ 不同。补零到最大长度可以勉强处理，但并不优雅。
- **不能在位置之间共享特征**：在第 1 个位置学到 “Harry 是人名”，不能自动用到第 5 个位置。这与全连接处理图像时的问题一样；卷积用共享核解决空间问题，循环网络用共享权重解决时间问题。此外，10000 维的输入乘以 $T_x$ 个位置，第一层参数也太多。

### RNN 单元

> [!INFO]+ RNN
>
> 循环网络在每个时间步复用同一组 $W_{aa},W_{ax}$，隐状态 $a^{\langle t\rangle}$ 概括到当前步为止的历史。参数量不随序列长度增长，因此能处理变长输入。计算必须按时间串行；长程梯度反复相乘后容易消失，后面用门控和注意力处理。

<br>

循环网络按时间顺序读输入：读第 $t$ 个词时，除了 $x^{\langle t\rangle}$，还把上一步的隐状态 $a^{\langle t-1\rangle}$ 一起输入，所以第 $t$ 步的预测能利用 $x^{\langle 1\rangle},\ldots,x^{\langle t\rangle}$ 的全部信息。前向传播为

$$
\begin{aligned}
a^{\langle t\rangle} &= g\big(W_{aa}a^{\langle t-1\rangle}+W_{ax}x^{\langle t\rangle}+b_a\big),\\
\hat{y}^{\langle t\rangle} &= g_y\big(W_{ya}a^{\langle t\rangle}+b_y\big).
\end{aligned}
$$

- $g$ 常用 tanh（有时用 ReLU）；$g_y$ 按任务选 sigmoid（二分类）或 softmax（多分类）。
- $a^{\langle 0\rangle}$ 通常为零向量。
- 参数 $W_{aa},W_{ax},W_{ya},b_a,b_y$ **不随 $t$ 改变**，所有时间步共享。
- 下标的读法：$W_{ax}$ 乘以 $x$ 类的量，输出 $a$ 类的量。

**简化记号**：把两个矩阵横向拼成 $W_a=[W_{aa}\mid W_{ax}]$，把两个向量纵向拼成 $[a^{\langle t-1\rangle};x^{\langle t\rangle}]$，则

$$
a^{\langle t\rangle}=g\big(W_a[a^{\langle t-1\rangle};x^{\langle t\rangle}]+b_a\big),\qquad
\hat{y}^{\langle t\rangle}=g\big(W_y a^{\langle t\rangle}+b_y\big).
$$

例如 $n_a=100$、$n_x=10000$：$W_{aa}$ 为 $(100,100)$，$W_{ax}$ 为 $(100,10000)$，拼成 $W_a$ 为 $(100,10100)$。后面的 GRU、LSTM 都用这种写法。

**局限**：上式只用到了过去的信息。“He said, Teddy Roosevelt was a great President.” 与 “He said, Teddy bears are on sale!” 的前三个词完全相同，只看前面无法判断 Teddy 是不是人名，这需要后文的双向 RNN。

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

PyTorch 的 `nn.RNN` 计算同一个递推 $a^{\langle t\rangle}=\tanh(W_{ax}x^{\langle t\rangle}+b_{ih}+W_{aa}a^{\langle t-1\rangle}+b_{hh})$，只是把偏置拆成两个，并默认按 `(T, m, n_x)` 排布（`batch_first=True` 时为 `(m, T, n_x)`）。把作业参数拷进去即可对拍：

```python
n_x, n_a, m, T = 3, 5, 4, 6
rng = np.random.default_rng(0)
params = {"Wax": rng.standard_normal((n_a, n_x)), "Waa": rng.standard_normal((n_a, n_a)),
          "ba": rng.standard_normal((n_a, 1)),
          "Wya": rng.standard_normal((2, n_a)), "by": rng.standard_normal((2, 1))}
X = rng.standard_normal((n_x, m, T)); a0 = np.zeros((n_a, m))

rnn = nn.RNN(n_x, n_a, nonlinearity="tanh", batch_first=True).double()
with torch.no_grad():
    rnn.weight_ih_l0.copy_(torch.tensor(params["Wax"]))
    rnn.weight_hh_l0.copy_(torch.tensor(params["Waa"]))
    rnn.bias_ih_l0.copy_(torch.tensor(params["ba"][:, 0]))
    rnn.bias_hh_l0.zero_()
out, h_T = rnn(torch.tensor(X).permute(1, 2, 0))       # (n_x, m, T) -> (m, T, n_x)

a = a0
for t in range(T):
    a, _, _ = rnn_cell_forward(X[:, :, t], a, params)
print(np.abs(a.T - h_T[0].numpy()).max())             # 约 1e-15
```

`out` 是每一步的隐状态 `(m, T, n_a)`，输出层 $W_{ya}$ 要另外用 `nn.Linear` 接；`h_T` 只有最后一步，many-to-one 任务直接用它。

#### 架构类型

按输入输出的长度关系分为五类：

| 类型 | 示意 | 例子 |
| :--- | :--- | :--- |
| one-to-one | $T_x=T_y=1$ | 普通神经网络 |
| one-to-many | $T_x=1$，$T_y>1$ | 音乐生成：从一个种子（或零输入）展开，每步输出作为下一步输入 |
| many-to-one | $T_x>1$，$T_y=1$ | 情感分类：只用最后一步 $a^{\langle T_x\rangle}$ 输出 |
| many-to-many，等长 | $T_x=T_y$ | 命名实体识别、词性标注：每步都有标签 |
| many-to-many，不等长 | $T_x\ne T_y$ | 机器翻译：编码器（encoder）先读完输入，解码器（decoder）再逐步输出 |

#### 损失

每一步的损失按任务定义。例如命名实体识别每步是二分类，用逻辑回归损失：

$$
\mathcal{L}^{\langle t\rangle}\big(\hat{y}^{\langle t\rangle},y^{\langle t\rangle}\big)=-y^{\langle t\rangle}\log\hat{y}^{\langle t\rangle}-\big(1-y^{\langle t\rangle}\big)\log\big(1-\hat{y}^{\langle t\rangle}\big),
\qquad
\mathcal{L}=\sum_{t=1}^{T_y}\mathcal{L}^{\langle t\rangle}.
$$

### 语言模型与采样

#### 语言模型

**语言模型**给一个句子估计概率 $P(y^{\langle 1\rangle},\ldots,y^{\langle T_y\rangle})$。语音识别要用它在发音相同的候选间选择：“The apple and pair salad” 的概率约为 $3.2\times10^{-13}$，“The apple and pear salad” 约为 $5.7\times10^{-10}$，后者更可能是对的。机器翻译也需要它判断哪个译文更通顺。

训练：在大语料上**分词**（tokenize），把每个词映射为词表编号，句末加 `<EOS>`（end of sentence），未知词用 `<UNK>`。以 “Cats average 15 hours of sleep a day. `<EOS>`” 为例：

- 第 1 步输入 $x^{\langle 1\rangle}=\vec{0}$，$\hat{y}^{\langle 1\rangle}$ 是词表上的 softmax，预测第一个词是各个词的概率 $P(\text{cats})$；
- 第 2 步输入 $x^{\langle 2\rangle}=y^{\langle 1\rangle}$（真实的 “cats”），预测 $P(\,\cdot\mid\text{cats})$；
- 第 3 步输入 “average”，预测 $P(\,\cdot\mid\text{cats average})$；依此类推，直到预测出 `<EOS>`。

每步损失是 softmax 交叉熵 $\mathcal{L}^{\langle t\rangle}=-\sum_i y_i^{\langle t\rangle}\log\hat{y}_i^{\langle t\rangle}$，总损失为各步之和。训练好后，句子的概率由链式法则得到：

$$
P\big(y^{\langle 1\rangle},y^{\langle 2\rangle},y^{\langle 3\rangle}\big)=P\big(y^{\langle 1\rangle}\big)\,P\big(y^{\langle 2\rangle}\mid y^{\langle 1\rangle}\big)\,P\big(y^{\langle 3\rangle}\mid y^{\langle 1\rangle},y^{\langle 2\rangle}\big).
$$

- **字符级语言模型**：词表是字母、数字、标点，$\hat{y}^{\langle t\rangle}$ 是字符表上的 softmax。好处是没有 `<UNK>`；代价是序列长得多，长程依赖更难学，训练更贵。
- 训练：每步交叉熵。

#### 困惑度

困惑度（perplexity, PPL）是平均负对数似然的指数。若有效词元数为 $N$，

$$
\mathrm{PPL}
=\exp\left(-\frac{1}{N}\sum_{t=1}^{N}
\log p(x_t\mid x_{<t})\right).
$$

PPL 越低，模型给真实序列的平均概率越高。一个直观解释：若模型对每个词都均匀地给出 $1/|V|$，则 $\mathrm{PPL}=|V|$，可以理解为模型平均在多少个词之间“犹豫”。只有词元化、词表与计数方式相同时，两个 PPL 才可直接比较。

#### 采样

训练好的语言模型可以生成新序列，用来检查模型学到了什么：

1. 第 1 步输入 $\vec{0}$，从 softmax 分布 $\hat{y}^{\langle 1\rangle}$ 中**随机采样**一个词（`np.random.choice`）；
2. 把采到的词作为第 2 步的输入，再采样；
3. 直到采到 `<EOS>` 或达到最大长度。若不希望出现 `<UNK>`，采到时就重采。

- 生成：从 $\hat{y}^{\langle t\rangle}$ 采样（或取 argmax）作为下一步输入。
- 温度 $T$：logits 除以 $T$ 再 softmax。
- $T>1$ 更均匀；$T\rightarrow0$ 接近贪心。

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

- 恐龙名字生成作业用采样而不是 teacher forcing，演示模型是否学会合法的字符转移。

### BPTT（backpropagation through time）

#### 时间反向传播

时间反向传播（BPTT）就是把展开后的循环网络当成一个很深的前馈网络，从 $t=T$ 向 $t=1$ 回传梯度。$W_{aa}$ 在每个时间步是同一块矩阵，所以它的梯度是所有时刻贡献之和（与卷积核梯度对所有空间位置求和是一回事）。

- 损失 $J=\sum_t \mathcal{L}^{\langle t\rangle}$。
- $W_{aa}$ 的梯度来自所有时刻：同一矩阵被反复使用。
- 从 $t=T$ 向 $t=1$ 回传 $\mathrm{d}a^{\langle t\rangle}$。每一步的 $\mathrm{d}a^{\langle t\rangle}$ 有两个来源：本步输出 $\hat{y}^{\langle t\rangle}$ 的误差，以及从 $t+1$ 步传回的梯度。

tanh 的导数为 $1-\tanh^{2}$，单步反向为

$$
\mathrm{d}z^{\langle t\rangle}=\big(1-(a^{\langle t\rangle})^{2}\big)\odot\mathrm{d}a^{\langle t\rangle},\quad
\mathrm{d}W_{ax}\mathrel{+}=\mathrm{d}z^{\langle t\rangle}x^{\langle t\rangle\top},\quad
\mathrm{d}W_{aa}\mathrel{+}=\mathrm{d}z^{\langle t\rangle}a^{\langle t-1\rangle\top},\quad
\mathrm{d}a^{\langle t-1\rangle}=W_{aa}^{\top}\mathrm{d}z^{\langle t\rangle}.
$$

```python
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

#### 梯度消失与爆炸

> [!EXAMPLE]+ 例2 长程依赖
>
> “The **cat**, which already ate an apple, a banana, ..., **was** full.” 与 “The **cats**, which already ate ..., **were** full.”
>
> 动词用 was 还是 were，取决于很多个词之前的主语是单数还是复数。普通 RNN 很难学到这种长距离依赖：第 $t$ 步的输出主要受附近几步影响。

原因与深层网络中的梯度消失相同。从第 $t$ 步回传到第 $k$ 步，要连乘

$$
\prod_{k} W_{aa}^{\top}\mathrm{diag}\big(1-(a^{\langle k\rangle})^{2}\big),
$$

这导致：

- 谱半径 $>1$：**梯度爆炸**，参数出现 NaN，容易发现。解决办法是**梯度裁剪**：逐值裁剪（clip by value）或全局范数裁剪（global norm clipping）：

$$
g\leftarrow
\begin{cases}
g & \|g\|_2\le \tau,\\
\dfrac{\tau}{\|g\|_2}g & \|g\|_2>\tau.
\end{cases}
$$

  逐值裁剪会分别截断每个坐标，改变梯度方向；全局范数裁剪只统一缩放，保留方向。
- 谱半径 $<1$：**梯度消失**，早期步对 $J$ 几乎无贡献，无法学长距离依赖。消失更常见也更难处理，解决办法是下面的 GRU 和 LSTM。

```python
def clip_gradients(grads, max_value=5.0):
    for k in grads:
        np.clip(grads[k], -max_value, max_value, out=grads[k])
    return grads
```

- 字符级作业里 $T$ 可达数百，必须裁剪。

### GRU

> [!INFO]+ GRU
>
> GRU（Cho et al., 2014）用更新门 $\Gamma_u$ 和重置门 $\Gamma_r$ 控制写入。$\Gamma_u\approx 0$ 时直接拷贝旧状态，梯度可以沿时间传得更远。参数比 LSTM 少；作业实现其中一种即可。

<br>

#### 简化的 GRU

门控循环单元（gated recurrent unit, GRU）引入一个**记忆单元** $c$（GRU 中 $c^{\langle t\rangle}=a^{\langle t\rangle}$）。每一步先算一个候选值 $\tilde{c}^{\langle t\rangle}$，再用**更新门** $\Gamma_u$ 决定是否用它覆盖旧记忆：

$$
\begin{aligned}
\tilde{c}^{\langle t\rangle} &= \tanh\big(W_c[c^{\langle t-1\rangle},x^{\langle t\rangle}]+b_c\big),\\
\Gamma_u &= \sigma\big(W_u[c^{\langle t-1\rangle},x^{\langle t\rangle}]+b_u\big),\\
c^{\langle t\rangle} &= \Gamma_u\odot\tilde{c}^{\langle t\rangle}+(1-\Gamma_u)\odot c^{\langle t-1\rangle}.
\end{aligned}
$$

$\Gamma$ 是 sigmoid 输出，大多数时候非常接近 0 或 1。在例 2 中：读到 “cat” 时 $\Gamma_u\approx 1$，把 “主语是单数” 写入 $c$；之后一长串词 $\Gamma_u\approx 0$，$c^{\langle t\rangle}=c^{\langle t-1\rangle}$，记忆原样保留；到 “was” 时再读取。$\Gamma_u\approx 0$ 时 $c^{\langle t\rangle}\approx c^{\langle t-1\rangle}$，梯度也几乎原样传过，梯度消失大为缓解。$c,\tilde{c},\Gamma_u$ 维度相同，$\odot$ 为逐元素乘：每一维可以独立决定更新与否（有的维记单复数，有的维记话题）。

#### 完整的 GRU

再加一个**相关门**（relevance gate，也称重置门）$\Gamma_r$，控制计算候选值时用多少旧记忆。用 $a$ 写出：

- 用更新门决定保留多少旧状态，减轻消失梯度。
- 令 $[x^{\langle t\rangle}; a^{\langle t-1\rangle}]$ 为输入：

$$
\begin{aligned}
\Gamma_u &= \sigma(W_u[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_u),\\
\Gamma_r &= \sigma(W_r[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_r),\\
\tilde{c}^{\langle t\rangle} &= \tanh\big(W_c[\Gamma_r\odot a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_c\big),\\
a^{\langle t\rangle} &= \Gamma_u\odot\tilde{c}^{\langle t\rangle}+(1-\Gamma_u)\odot a^{\langle t-1\rangle}.
\end{aligned}
$$

- $\Gamma_u\approx1$：写入新候选。
- $\Gamma_u\approx0$：拷贝旧状态，梯度可沿时间高速路传递。
- $\Gamma_r$：控制候选计算时看见多少历史。

```python
def gru_cell_forward(xt, a_prev, params):
    """xt: (n_x, m), a_prev: (n_a, m)；每个 W 为 (n_a, n_a + n_x)。"""
    concat = np.vstack([a_prev, xt])
    gu = sigmoid(params["Wu"] @ concat + params["bu"])                 # 更新门
    gr = sigmoid(params["Wr"] @ concat + params["br"])                 # 相关门
    cct = np.tanh(params["Wc"] @ np.vstack([gr * a_prev, xt]) + params["bc"])
    a_next = gu * cct + (1.0 - gu) * a_prev
    return a_next, (a_prev, xt, gu, gr, cct)
```

PyTorch 的 `nn.GRU` 有两处与上式不同，对拍时要注意：

- 更新门的含义相反：`h = (1 - z) * n + z * h_prev`，即 PyTorch 的 $z$ 对应这里的 $1-\Gamma_u$；
- 重置门作用在**乘完 $W$ 之后**：$n=\tanh(W_{in}x+b_{in}+r\odot(W_{hn}h+b_{hn}))$，而不是先乘 $r$ 再乘 $W$。

两种写法表达能力相近，但权重不能直接互拷。

### LSTM

> [!INFO]+ LSTM
>
> LSTM（Hochreiter & Schmidhuber, 1997）另设细胞状态 $c^{\langle t\rangle}$，用遗忘、输入、输出三个门更新。$c$ 上主要是加法，长程依赖比朴素 RNN 稳定。作业要求能写出六式；常见错误是门的拼接顺序，以及下一门读的是 $c$ 还是 $a$。

<br>

长短期记忆网络（long short-term memory, LSTM）比 GRU 更早提出，也更通用。与 GRU 的区别：

- 记忆单元 $c^{\langle t\rangle}$ 与隐状态 $a^{\langle t\rangle}$ **分开**，$a^{\langle t\rangle}\ne c^{\langle t\rangle}$；
- GRU 的 $\Gamma_u$ 与 $1-\Gamma_u$ 被拆成两个独立的门：**遗忘门** $\Gamma_f$ 和**输入门** $\Gamma_i$（Ng 课上记作更新门 $\Gamma_u$）；
- 多一个**输出门** $\Gamma_o$，决定暴露多少记忆。

- 增加记忆单元 $c^{\langle t\rangle}$。
- 三个门：遗忘 $\Gamma_f$、输入 $\Gamma_i$、输出 $\Gamma_o$。

$$
\begin{aligned}
\Gamma_f &= \sigma(W_f[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_f),\\
\Gamma_i &= \sigma(W_i[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_i),\\
\tilde{c}^{\langle t\rangle} &= \tanh(W_c[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_c),\\
c^{\langle t\rangle} &= \Gamma_f\odot c^{\langle t-1\rangle}+\Gamma_i\odot\tilde{c}^{\langle t\rangle},\\
\Gamma_o &= \sigma(W_o[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_o),\\
a^{\langle t\rangle} &= \Gamma_o\odot\tanh(c^{\langle t\rangle}).
\end{aligned}
$$

- 遗忘门：把 $c^{\langle t-1\rangle}$ 的部分清零。
- 输入门：写入候选。
- 输出门：决定暴露多少记忆到 $a^{\langle t\rangle}$。
- $c$ 的加法更新是长程信息的主通路：只要遗忘门与输入门设得合适，$c^{\langle 0\rangle}$ 的值可以一直传到很后面。
- 把 $b_f$ 初始化为小正数，有助于早期偏向“先记住”。
- 变体：窥视孔连接（peephole connection）让各个门也看到 $c^{\langle t-1\rangle}$。

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

PyTorch 的 `nn.LSTM` 把四组权重竖着拼成一个 `(4·n_a, n_x)` 的 `weight_ih` 和一个 `(4·n_a, n_a)` 的 `weight_hh`，顺序是**输入门、遗忘门、候选、输出门**（i, f, g, o）：

```python
def lstm_params_to_torch(params, n_x, n_a):
    """把作业的 Wf/Wi/Wc/Wo（作用于 [a; x]）拆成 PyTorch 的 ih / hh 两块。"""
    order = ["Wi", "Wf", "Wc", "Wo"]                     # PyTorch 的门顺序
    W = np.vstack([params[k] for k in order])            # (4 n_a, n_a + n_x)
    b = np.vstack([params["b" + k[1:]] for k in order])[:, 0]
    return W[:, n_a:], W[:, :n_a], b                     # weight_ih, weight_hh, bias

lstm = nn.LSTM(input_size=n_x, hidden_size=n_a, num_layers=2,
               batch_first=True, bidirectional=True, dropout=0.2)
x = torch.randn(m, T, n_x)
out, (h_n, c_n) = lstm(x)
print(out.shape)   # (m, T, 2·n_a)：每步把正向、反向隐状态拼接
print(h_n.shape)   # (num_layers·2, m, n_a)
```

- `bidirectional=True` 就是上文的 BRNN，`num_layers=2` 就是深层 RNN；`dropout` 只加在层与层之间。
- 序列长度不一时，用 `nn.utils.rnn.pack_padded_sequence` 包装，否则补齐的 `<pad>` 也会更新隐状态。
- 遗忘门偏置初始化为正数：`lstm.bias_ih_l0.data[n_a:2 * n_a].fill_(1.0)`。

#### 门的作用对照

| 门 | 作用 | 使用者 |
| :--- | :--- | :--- |
| 更新门 / 输入门 $\Gamma_u$ | 新候选写入多少 | GRU、LSTM |
| 相关门 $\Gamma_r$ | 计算候选时用多少旧状态 | GRU |
| 遗忘门 $\Gamma_f$ | 旧记忆保留多少 | LSTM |
| 输出门 $\Gamma_o$ | 记忆暴露多少到 $a$ | LSTM |

#### 反向与比较

- LSTM 反向：对 $c^{\langle t\rangle}$ 与 $a^{\langle t\rangle}$ 分别回传。
- $\partial c^{\langle t\rangle}/\partial c^{\langle t-1\rangle}=\Gamma_f$；遗忘门接近 1 时梯度几乎原样传递。这与残差网络的 $1+\partial F/\partial x$ 是同一思想：加法路径保梯度。
- 门的饱和（$\sigma$ 两端）仍会削弱更新。
- GRU 参数更少、计算更快，容易做得更大；LSTM 更灵活，历史上是默认首选。序列不是极长时二者往往接近。
- 作业要求能默写门的形状：每个门矩阵是 $(n_a, n_a+n_x)$。

### 双向与深层 RNN

**双向 RNN**（bidirectional RNN, BRNN）：分别跑正向和反向两条 RNN，正向从 $t=1$ 读到 $T_x$，反向从 $T_x$ 读到 1，每步把两边的隐状态拼起来做预测：

$$
\hat{y}^{\langle t\rangle}=g\big(W_y[\overrightarrow{a}^{\langle t\rangle};\overleftarrow{a}^{\langle t\rangle}]+b_y\big).
$$

于是第 $t$ 步的预测同时利用了过去和未来，能解决前面 “Teddy” 的歧义。单元可以是 RNN、GRU 或 LSTM，NLP 中 BLSTM 很常见。

- 适合标注任务，不适合在线因果生成：必须看到整个序列才能预测（语音识别要等说话人说完）。

**深层 RNN**：把多层 RNN 叠起来，下层的 $a^{\langle t\rangle}$ 当作上层的输入。第 $l$ 层第 $t$ 步：

$$
a^{[l]\langle t\rangle}=g\Big(W_a^{[l]}\big[a^{[l]\langle t-1\rangle},a^{[l-1]\langle t\rangle}\big]+b_a^{[l]}\Big).
$$

每层每个时间步都要计算，所以 RNN 叠 3 层就已经很深了（不像 CNN 动辄上百层）。有时在顶层 RNN 之上再接几层只在纵向连接、不在时间上连接的全连接层。

### Teacher forcing 与生成评估

> [!INFO]+ Teacher forcing
>
> Teacher forcing：训练时解码器的输入是上一时刻的真值标签，不是模型自己的预测。收敛通常更快，但训练分布与推理分布不一致；推理时第一步出错会向后传播。

<br>

- Teacher forcing：训练时解码器输入真值上一步，不是自己的预测；加快收敛，但造成训练-推理落差。前面语言模型训练时“第 2 步输入真实的 cats”就是 teacher forcing。
- 爵士乐作业：和弦与音高序列送入 LSTM，输出下一步音符的 softmax。
- 评估不能只看交叉熵：还要听是否跑调、是否只会重复短动机。
- 生成模型的损失下降不等于样本质量；GAN 与 LLM 仍然成立。

> [!NOTE]+ 衔接
>
> RNN 的输入 $x^{\langle t\rangle}$ 若是 one-hot，任意两个词的内积恒为 0，模型无法知道 “apple” 和 “orange” 相似。下一节用嵌入表 $E$ 把离散记号变成可比较的向量，再送进 RNN 或平均后分类。

<br>

## 词嵌入（Word Embeddings）

> [!INFO]+ 词嵌入
>
> 词嵌入把离散词号换成 $d$ 维向量。one-hot 的内积恒为 0；稠密向量里相近词可以有较大余弦。word2vec（2013）用 Skip-gram 或 CBOW 训练，GloVe 拟合共现对数。大词表上的 softmax 用负采样近似。嵌入会带上语料中的偏见。

<br>

### 从 one-hot 到特征表示

one-hot $o_j\in\mathbb{R}^{|V|}$ 把每个词看作孤立的符号：任意两个不同词的内积恒为 0，欧氏距离都相同。模型学会了 “I want a glass of orange **juice**”，遇到 “I want a glass of apple ___” 时无法推广，因为在 one-hot 下 apple 与 orange 的关系和 apple 与 king 的关系一样远。

改进方法是用一组特征描述每个词：

> [!EXAMPLE]+ 例3 特征化的词表示
>
> | 特征 | Man | Woman | King | Queen | Apple | Orange |
> | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
> | Gender | $-1$ | $1$ | $-0.95$ | $0.97$ | $0.00$ | $0.01$ |
> | Royal | $0.01$ | $0.02$ | $0.93$ | $0.95$ | $-0.01$ | $0.00$ |
> | Age | $0.03$ | $0.02$ | $0.70$ | $0.69$ | $0.03$ | $-0.02$ |
> | Food | $0.09$ | $0.01$ | $0.02$ | $0.01$ | $0.95$ | $0.97$ |
>
> 实际使用约 300 维。Apple 与 Orange 的向量非常接近，所以学到 “orange juice” 后很容易推广到 “apple juice”。
>
> **类比**：$e_{\mathrm{man}}-e_{\mathrm{woman}}=(-2,-0.01,0.01,0.08)$，$e_{\mathrm{king}}-e_{\mathrm{queen}}=(-1.92,-0.02,0.01,0.01)$，两个差向量几乎相同，主要差在 Gender 维。所以“man 之于 woman 如同 king 之于 ?”可以通过找与 $e_{\mathrm{king}}-e_{\mathrm{man}}+e_{\mathrm{woman}}$ 最接近的词来回答。

学出来的嵌入的各个维度通常不像上表那样可解释，但相近词仍会聚在一起。常用 t-SNE 把 300 维嵌入降到 2 维画图观察：水果聚一团、数字聚一团、人物聚一团。t-SNE 是非线性映射，降维后类比关系的平行四边形一般不再保持。

### 嵌入矩阵

- 嵌入表 $E\in\mathbb{R}^{d\times |V|}$，$e_j=E o_j$ 即取第 $j$ 列。例如 $d=300$、$|V|=10000$，$E$ 为 $(300,10000)$，$e_{6257}=E\,o_{6257}$ 是第 6257 个词（orange）的 300 维嵌入。
- 实现中不做这个矩阵乘法（$o_j$ 几乎全是 0，太浪费），而是直接按列查表（Keras 的 `Embedding` 层）。
- 语言模型、情感分类、翻译都可以把 $E$ 当作第一层，可冻结或微调。

**迁移学习**：在 10 亿到 1000 亿词的无标签语料上学嵌入（或直接下载预训练的），迁移到只有约 10 万词标注数据的新任务（如命名实体识别），可选地再在新任务上微调。下游数据越少越有帮助；在机器翻译、语言模型这类本身数据很多的任务上帮助较小。这与人脸识别中的“编码”（encoding）类似：人脸编码要对任意新图都能计算，词嵌入只针对固定词表。

### 学习嵌入

#### 神经语言模型

Bengio 等的做法：用嵌入做语言模型，嵌入作为副产品被学出来。预测 “I want a glass of orange ___”：把前面 4 个词各自查表得 300 维嵌入，拼成 1200 维向量，送入隐层和 10000 维 softmax，预测下一个词。$E$、隐层和 softmax 的参数一起用反向传播训练。为了预测好下一个词，相似语境中的词（apple、orange）会被学到相似的嵌入。

上下文（context）可以有多种选法：前 4 个词；左右各 4 个词；只用前 1 个词；附近随机的 1 个词。若目的是学嵌入而不是做语言模型，简单的上下文就够用，这引出了 word2vec。

#### Skip-gram

- Skip-gram：由中心词预测窗口内上下文。
- CBOW：由上下文预测中心词。

Skip-gram 从句子中随机取一个上下文词 $c$（如 orange），再在它前后一定窗口内随机取一个目标词 $t$（如 juice），构造一个监督学习问题：给定 $c$ 预测 $t$。模型为

$$
o_c\rightarrow E\rightarrow e_c\rightarrow\mathrm{softmax}\rightarrow\hat{y},\qquad
p(t\mid c)=\frac{\exp(\theta_t^{\top}e_c)}{\sum_{j=1}^{|V|}\exp(\theta_j^{\top}e_c)},
$$

其中 $\theta_t$ 是输出 $t$ 对应的参数，损失为 $\mathcal{L}(\hat{y},y)=-\sum_i y_i\log\hat{y}_i$。这个任务本身很难做好，但目的是学到好的 $E$。

Skip-gram 精确目标：窗口半径 $c$，中心词 $w_t$，

$$
\sum_{t}\sum_{\substack{-c\le j\le c\\ j\neq 0}}\log P(w_{t+j}\mid w_t),\qquad
P(w_O\mid w_I)=\frac{\exp(v_{w_O}^{\prime\top} v_{w_I})}{\sum_{w}\exp(v_w^{\prime\top} v_{w_I})}.
$$

- 分母遍历词表，计算 $O(|V|)$，词表大时太慢。一种办法是分层 softmax（hierarchical softmax）：用一棵二叉树逐层分类，代价降到 $O(\log|V|)$，常用词放在浅层。另一种办法是负采样。
- 上下文词 $c$ 的采样不能完全均匀，否则 the、of、a、and 这类高频词占绝大多数，需要启发式地平衡常见词与罕见词。

#### 负采样

- softmax 在大词表上昂贵，用负采样（negative sampling）近似。

把问题改成二分类：给一对词 $(c,t)$，判断 $t$ 是否真的出现在 $c$ 附近。

> [!EXAMPLE]+ 例4 负采样的训练样本
>
> 从句子 “I want a glass of orange juice to go along with my cereal” 中取一个正样本，再从词表随机取 $k$ 个词作为负样本：
>
> | 上下文 $c$ | 词 $t$ | 标签 $y$ |
> | :--- | :--- | :--- |
> | orange | juice | 1 |
> | orange | king | 0 |
> | orange | book | 0 |
> | orange | the | 0 |
> | orange | of | 0 |
>
> 即使随机抽到的词碰巧出现在上下文中（如 of），也标为 0。小数据集取 $k=5$ 到 $20$，大数据集取 $k=2$ 到 $5$。

模型是逻辑回归：

$$
P(y=1\mid c,t)=\sigma\big(\theta_t^{\top}e_c\big).
$$

相当于有 $|V|$ 个二分类器，但每一步只训练其中 $k+1$ 个（1 个正例、$k$ 个负例），代价从 $O(|V|)$ 降到 $O(k)$。对一对 $(w_I,w_O)$ 与 $k$ 个噪声词做逻辑回归，目标为

$$
\log\sigma(v_{w_O}^{\prime\top}v_{w_I})+\sum_{i=1}^{k}\mathbb{E}\_{w_i\sim P_n}[\log\sigma(-v_{w_i}^{\prime\top}v_{w_I})].
$$

负样本的采样分布：按词频 $f(w)$ 采样会抽到太多 the、of；均匀采样又不能代表英文的词分布。经验折中是

$$
P_n(w_i)=\frac{f(w_i)^{3/4}}{\sum_{j=1}^{|V|}f(w_j)^{3/4}},
$$

即 $P_n(w)\propto f(w)^{3/4}$：压低超高频词。

NumPy 实现一步负采样 SGD。对 $\mathcal{L}=-\log\sigma(\theta_t^{\top}e_c)-\sum_{i}\log\sigma(-\theta_{n_i}^{\top}e_c)$ 求导，与逻辑回归的 $a-y$ 形式一样：

```python
def neg_sampling_step(E, Theta, c, t, word_freq, k=5, lr=0.025, rng=None):
    """E, Theta: (|V|, d) 输入/输出嵌入；c, t: 上下文词与目标词下标。"""
    rng = rng or np.random.default_rng()
    p = word_freq ** 0.75
    p = p / p.sum()
    negs = rng.choice(len(p), size=k, p=p)             # 负样本按 f^{3/4} 抽
    ids = np.concatenate([[t], negs])                  # 1 个正例 + k 个负例
    labels = np.array([1.0] + [0.0] * k)
    e_c = E[c]                                          # (d,)
    scores = Theta[ids] @ e_c                           # (k+1,)
    g = sigmoid(scores) - labels                        # 每个二分类器的 a - y
    loss = -np.log(sigmoid(scores[0])) - np.log(sigmoid(-scores[1:])).sum()
    grad_e = g @ Theta[ids]                             # 对 e_c 的梯度
    np.subtract.at(Theta, ids, lr * g[:, None] * e_c)   # 负样本可能重复，用 at 累加
    E[c] -= lr * grad_e
    return loss
```

PyTorch 中，嵌入矩阵就是 `nn.Embedding`：前向是按下标查行，等价于 one-hot 乘 $E$，但不用真的构造 one-hot。

```python
class SkipGramNS(nn.Module):
    def __init__(self, V, d=300):
        super().__init__()
        self.inp = nn.Embedding(V, d)       # E：查 e_c
        self.out = nn.Embedding(V, d)       # Θ：查 θ_t
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.out.weight)

    def forward(self, c, t, negs):          # c, t: (B,)；negs: (B, k)
        e = self.inp(c)                                          # (B, d)
        pos = (self.out(t) * e).sum(-1)                          # (B,)
        neg = torch.bmm(self.out(negs), e.unsqueeze(-1)).squeeze(-1)   # (B, k)
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()

E = nn.Embedding(10000, 300)
print(E(torch.tensor([6257])).shape)    # (1, 300)：第 6257 个词（orange）的嵌入
```

下游任务加载预训练嵌入时，用 `nn.Embedding.from_pretrained(weights, freeze=True)`；下游数据很少时冻结，较多时可以 `freeze=False` 一起微调。

#### GloVe

- GloVe：对共现对数做加权最小二乘，使 $w_i^{\top}w_j\approx\log X_{ij}$。

GloVe（Global Vectors for Word Representation）的完整目标为

$$
J=\sum_{i,j}f(X_{ij})
\left(w_i^{\top}\tilde{w}_j+b_i+\tilde{b}_j-\log X_{ij}\right)^2.
$$

$X_{ij}$ 是词 $j$ 出现在词 $i$ 上下文中的次数，$f(X_{ij})$ 降低极罕见与极高频共现对目标的支配。Skip-gram 直接做局部预测，GloVe 显式拟合全局共现统计。

- 加权函数满足 $X_{ij}=0$ 时 $f(X_{ij})=0$，约定 $0\log 0=0$，避免 $\log 0$；GloVe 的加权最小二乘对稀有共现降权。
- 窗口对称时 $w$ 与 $\tilde{w}$ 的作用对称，最终嵌入可取二者的平均 $e_w=\frac{1}{2}(w_w+\tilde{w}_w)$。
- 学到的各维不一定对应 Gender、Royal 这类可解释的轴：任意可逆矩阵 $A$ 满足 $(Aw_i)^{\top}(A^{-\top}\tilde{w}_j)=w_i^{\top}\tilde{w}_j$，坐标轴可以被任意旋转、拉伸。

### 相似度与类比

- 训练后常见类比：

$$
e_{\mathrm{king}}-e_{\mathrm{man}}+e_{\mathrm{woman}}\approx e_{\mathrm{queen}}.
$$

- 评价：余弦相似度。两个向量 $u,v$ 的余弦为

$$
\cos(u,v)=\frac{u^{\top}v}{\|u\|_2\,\|v\|_2},
$$

  类比题求 $\arg\max_w \cos\big(e_w,\ e_{\mathrm{king}}-e_{\mathrm{man}}+e_{\mathrm{woman}}\big)$，需要排除题中已出现的词。
- OOV：子词模型（BPE、WordPiece）或随机向量。

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

### 情感分类

情感分类的标注数据通常很少（几万到十万条），嵌入的迁移作用很大。

- **平均模型**：把句中所有词的嵌入取平均（或求和），得到 300 维向量，接 softmax 输出 1–5 星。“The dessert is excellent” 可以正确分类，而且任意长度的句子都适用。
- 平均模型忽略词序。“Completely lacking in good taste, good service, and good ambience.” 有三个 good，平均后偏正面，实际是差评。
- **RNN 模型**：把嵌入按顺序送入 many-to-one 的 LSTM，能处理“lacking ... good”这类结构。由于嵌入来自大语料，训练集中没出现过的近义词（如用 absent 代替 lacking）也能推广。

- 情感分类可先把句中词向量平均后接逻辑回归，作为强基线。
- 再对比 LSTM 或 Transformer 是否真有增益。
- Emojify 作业：句子平均嵌入送入 softmax，再换成 LSTM。
- 若平均嵌入已经够用，加 LSTM 的增益应在误差分析里写明（否定词、词序），否则是多余复杂度。

### 去偏

嵌入会编码语料中的社会偏见。例如在嵌入上做类比，可能得到 “Man : Computer Programmer = Woman : Homemaker”，“Father : Doctor = Mother : Nurse”。用于招聘或刑侦时必须单独审计。CS230 作业含词向量去偏：

1. **确定偏见方向**：估计性别子空间方向 $g$，例如 $\frac{1}{2}[(e_{\mathrm{he}}-e_{\mathrm{she}})+(e_{\mathrm{man}}-e_{\mathrm{woman}})]$（更一般地对多组差向量做 SVD）。
2. **中和**（neutralize）：对定义上与性别无关的词（doctor、babysitter），减去它在 $g$ 方向上的分量 $\mathrm{proj}_g e$。
3. **均衡**（equalize）：对定义上有性别的成对词（grandmother–grandfather、girl–boy），让它们到中性词的距离相等，使 babysitter 与 grandmother、grandfather 的相似度相同。

哪些词“定义上”有性别，可以训练一个分类器判断；需要均衡的词对数量较少，可手工列出。

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

- 作业检查：中性职业词与性别方向的余弦应下降；`king-queen` 类比不应崩溃。
- 去偏不保证下游公平，只是表示空间上的线性修正。

> [!NOTE]+ 衔接
>
> 词嵌入把离散符号换成可比较的向量。下一步是输入、输出都是序列且长度不同的任务（翻译）。seq2seq 把整句压进编码器最后一个隐状态 $a^{\langle T_x\rangle}$；注意力让解码器每一步对源端各隐状态取加权和，不再依赖这一个向量。

<br>

## 注意力（Attention）

注意力让解码器每一步对源端各隐状态加权求和，得到上下文向量。seq2seq 若只把整句压进最后一个隐状态 $a^{\langle T_x\rangle}$，长句前部容易丢失。Bahdanau 注意力按步读取源序列，不再依赖这一个向量。

- 输入是源隐状态与当前解码状态，输出是上下文向量。
- 作业实现 Bahdanau 即可；权重大不表示该位置具有因果关系。

### Seq2seq

> [!INFO]+ Seq2seq
>
> 编码器–解码器（Sutskever et al., 2014）先把源序列读完，再由解码器逐步生成目标序列，因此两端长度可以不同。整句若只压进最后一个隐状态，长句前部容易丢失。下一节的注意力按步读取源端各位置。

<br>

#### 编码器–解码器

- 编码器 RNN 读入源句（如法语 “Jane visite l'Afrique en septembre.”），得到一个概括整句的向量；解码器 RNN 以它为初始状态，逐词生成目标句（“Jane is visiting Africa in September.”）。
- 训练用 $<\mathrm{SOS}>$、`<EOS>` 定界。

同样的结构可以用于**图像描述**（image captioning）：把预训练 CNN（如去掉 softmax 的 AlexNet）当作编码器，输出 4096 维图像特征，再由 RNN 解码器生成句子。鱼书介绍的 NIC（Neural Image Caption）就是“深层 CNN + RNN”：CNN 提取图像特征，RNN 以该特征为初始值递归地生成文本，能为一张图生成 “A person riding a motorcycle on a dirt road.”，连“没有铺装的道路”都能正确描述。组合图像与自然语言等多种信息的处理称为**多模态**（multimodal）处理。

#### 条件语言模型

翻译模型的解码器与语言模型结构相同，区别在于初始状态不是 $\vec{0}$，而是编码器的输出。所以机器翻译是**条件语言模型**：

$$
P\big(y^{\langle 1\rangle},\ldots,y^{\langle T_y\rangle}\mid x\big).
$$

语言模型生成时随机采样；翻译时则希望找到**最可能**的译文：

$$
\arg\max_{y^{\langle 1\rangle},\ldots,y^{\langle T_y\rangle}}P\big(y^{\langle 1\rangle},\ldots,y^{\langle T_y\rangle}\mid x\big).
$$

#### 贪心与束搜索

- 推理：贪心每步取概率最大的词；束搜索（beam search）每步保留 $B$ 条累计对数概率最高的部分序列，最后在这 $B$ 条里选。

**贪心为什么不够**：“Jane is visiting Africa in September.” 比 “Jane is going to be visiting Africa in September.” 更好，但在 “Jane is” 之后，“going” 在英语中比 “visiting” 常见，$P(\text{going}\mid x,\text{Jane is})$ 更大，贪心会选上它，走进更差的句子。而穷举所有句子又不可能（10000 词、10 个词长就有 $10000^{10}$ 种）。

**束搜索**（束宽 $B=3$ 为例）：

1. 第 1 步：在 10000 个词中保留 $P(y^{\langle 1\rangle}\mid x)$ 最高的 3 个，如 in、jane、september；
2. 第 2 步：对这 3 个分别算第二个词的概率，共 $3\times10000=30000$ 个候选，按 $P(y^{\langle 1\rangle},y^{\langle 2\rangle}\mid x)=P(y^{\langle 1\rangle}\mid x)P(y^{\langle 2\rangle}\mid x,y^{\langle 1\rangle})$ 保留最高的 3 个，如 “in september”、“jane is”、“jane visits”（september 开头的分支可能全被淘汰）；
3. 每步都复制 3 份网络，重复直到产生 `<EOS>`。

- 束宽 $B$：每步保留 $B$ 条最高对数概率路径。
- $B=1$ 即贪心。
- $B$ 越大结果越好、越慢。产品中常用 $B=10$，研究中为了榨取最后一点性能可用 $1000$ 到 $3000$；$B$ 从 1 增到 10 收益大，从 1000 增到 3000 收益很小。
- 束搜索是启发式，不保证最大似然句（不同于 BFS、DFS 这类精确搜索）。
- $B$ 过大则短句与重复 $n$-gram 需要额外惩罚。

#### 长度归一化

概率连乘 $\prod_t P(y^{\langle t\rangle}\mid x,y^{\langle 1\rangle},\ldots,y^{\langle t-1\rangle})$ 中每一项都小于 1，乘多了会数值下溢，所以改为对数求和。但每多一个词就多一个负数，目标天然偏爱短句。比较路径时应除以长度惩罚（指数约 $0.7$），否则短句占优：

$$
\mathrm{score}(y)=\frac{1}{T_y^{\alpha}}\sum_{t=1}^{T_y}\log P(y^{\langle t\rangle}\mid y^{\langle t-1\rangle},x).
$$

$\alpha=1$ 为完全按长度归一化，$\alpha=0$ 为不归一化，$\alpha$ 是可调的超参数。实践中对每个长度 $T_y=1,\ldots,30$ 各保留束内的最优，再用上式统一比较。

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

#### 束搜索的误差分析

翻译效果不好时，要判断是 RNN 模型的错还是束搜索的错，以免盲目增大 $B$ 或盲目收集数据。

> [!EXAMPLE]+ 例5 RNN 的错还是束搜索的错
>
> 法语 “Jane visite l'Afrique en septembre.” 人工译文 $y^{*}$ = “Jane visits Africa in September.”，算法输出 $\hat{y}$ = “Jane visited Africa last September.”。用训练好的 RNN 分别计算 $P(y^{*}\mid x)$ 和 $P(\hat{y}\mid x)$：
>
> - $P(y^{*}\mid x)>P(\hat{y}\mid x)$：模型认为 $y^{*}$ 更好，但束搜索没找到，**束搜索的错**；
> - $P(y^{*}\mid x)\le P(\hat{y}\mid x)$：$y^{*}$ 明明更好，模型却给了更低概率，**RNN 的错**（加正则、更多数据、换结构）。
>
> 用了长度归一化时，比较的应是归一化后的得分。对开发集上所有出错的句子逐一判断、统计比例：只有束搜索出错的比例高时，增大 $B$ 才值得。

#### BLEU

翻译常有多个同样好的答案，不能用准确率评价。BLEU（bilingual evaluation understudy）用修正的 $n$-gram 精确率与短句惩罚衡量翻译，对同义改写不敏感，只是自动代理。

> [!EXAMPLE]+ 例6 修正精确率
>
> 法语 “Le chat est sur le tapis.” 有两个参考译文：Reference 1 “The cat is on the mat.”，Reference 2 “There is a cat on the mat.”
>
> **一元组**：机器输出 “the the the the the the the.” 共 7 个词，每个 the 都出现在参考中，普通精确率为 $7/7$，显然不合理。修正精确率把每个词的计数**截断**（clip）到它在单个参考译文中出现的最多次数：the 在 Reference 1 中出现 2 次，所以修正精确率为 $2/7$。
>
> **二元组**：机器输出 “The cat the cat on the mat.” 的二元组为
>
> | 二元组 | 输出中次数 | 截断后次数 |
> | :--- | :--- | :--- |
> | the cat | 2 | 1 |
> | cat the | 1 | 0 |
> | cat on | 1 | 1 |
> | on the | 1 | 1 |
> | the mat | 1 | 1 |
>
> 修正二元组精确率 $p_2=4/6$。若输出与某个参考完全相同，各阶 $p_n$ 都是 1。

- BLEU 的核心是修正 $n$-gram 精确率。
- 对 $n=1,\ldots,N$（常 $N=4$），

$$
p_n=\frac{\sum_{\mathrm{clip}}\mathrm{count}(\mathrm{ngram})}{\sum \mathrm{count}\_{\mathrm{sys}}(\mathrm{ngram})},
$$

- 短句惩罚（brevity penalty）：输出长 $c$ 小于参考长 $r$ 时 $\mathrm{BP}=\exp(1-r/c)$，否则为 1。没有它，只输出一两个高把握的词就能得到很高的精确率。

$$
\mathrm{BLEU}=\mathrm{BP}\cdot\exp\Big(\frac{1}{N}\sum_{n=1}^{N}\log p_n\Big).
$$

- clip：每个 n-gram 计数不超过参考中出现次数，防止“the the the”刷精确率。
- BLEU 让机器翻译、图像描述等有多个正确答案的任务有了单一实数评价指标，极大地加快了迭代。
- BLEU 不理解同义，不能单独作为项目唯一指标。

### Bahdanau 注意力

> [!INFO]+ 注意力
>
> 注意力（Bahdanau et al., 2015）：解码每一步用当前状态给源端各隐状态打分，softmax 后做加权和，得到上下文向量。源句不必压成单一向量 $a^{\langle T_x\rangle}$。作业实现 Bahdanau 即可。权重大不表示该位置具有因果关系。

<br>

#### 长句问题

人工翻译长句时，不会先把整句背下来再一口气译出，而是一段一段地看、一段一段地译。编码器–解码器却要求把整句记进一个向量：句子短时 BLEU 尚可，句子超过约 20 到 30 个词后 BLEU 明显下降。加入注意力后，BLEU 在长句上基本保持平稳。

#### 计算

编码器用双向 RNN（或 LSTM），源端第 $t'$ 步的特征为 $h^{\langle t'\rangle}=[\overrightarrow{a}^{\langle t'\rangle};\overleftarrow{a}^{\langle t'\rangle}]$（Ng 课上记作 $a^{\langle t'\rangle}$）。解码器是另一个单向 RNN，状态为 $s^{\langle t\rangle}$。生成第 $t$ 个词时：

1. 用一个小的神经网络，根据上一步解码状态 $s^{\langle t-1\rangle}$ 和每个源位置 $h^{\langle t'\rangle}$ 打分。解码步 $t$，源步 $t'$ 的能量

$$
e^{\langle t,t'\rangle}=v^{\top}\tanh\big(W_a[s^{\langle t-1\rangle};h^{\langle t'\rangle}]\big),
$$

2. softmax 归一化为注意力权重，再对源特征加权求和得到上下文向量：

$$
\alpha^{\langle t,t'\rangle}=\frac{\exp(e^{\langle t,t'\rangle})}{\sum_{k=1}^{T_x}\exp(e^{\langle t,k\rangle})},\qquad
c^{\langle t\rangle}=\sum_{t'=1}^{T_x}\alpha^{\langle t,t'\rangle}h^{\langle t'\rangle}.
$$

   $\alpha^{\langle t,t'\rangle}$ 表示生成 $y^{\langle t\rangle}$ 时应对 $h^{\langle t'\rangle}$ 给予多少注意力，且 $\sum_{t'}\alpha^{\langle t,t'\rangle}=1$。
3. $s^{\langle t\rangle}$ 由解码器在 $[c^{\langle t\rangle}; y^{\langle t-1\rangle}]$ 上更新，再输出 $y^{\langle t\rangle}$。

为什么 $e$ 要由 $s^{\langle t-1\rangle}$ 和 $h^{\langle t'\rangle}$ 决定：应该关注源句哪里，取决于已经译到哪了（$s^{\langle t-1\rangle}$）以及源端各位置是什么内容（$h^{\langle t'\rangle}$）。这个函数难以手写，所以用小网络学出来，与整个模型一起用梯度下降训练。

- 注意力权重可视化可检查对齐，但高权重不是因果证明。
- 代价是二次的：$T_x$ 个源位置、$T_y$ 个输出，共 $T_xT_y$ 个权重。翻译的句子不长，可以接受。

> [!INFO]+ 注意力是期望
>
> $\alpha^{\langle t,\cdot\rangle}$ 是源位置上的离散分布，$c^{\langle t\rangle}$ 是隐状态的期望。与硬对齐（每次只选一个源词）相比，软注意力可微。复杂度每解码步 $O(T_x d)$，总 $O(T_y T_x d)$。源或目标很长时这仍贵，但比让单一向量承担全部记忆要稳。

<br>

> [!EXAMPLE]+ 例7 日期格式翻译（作业）
>
> 作业用注意力模型把各种人写的日期转成标准格式，例如 “Tuesday 09 Oct 1993” 或 “July 20th 1969” 转成 “1993-10-09” 或 “1969-07-20”。把 $\alpha^{\langle t,t'\rangle}$ 画成热力图，可以看到输出月份 “10” 时注意力集中在输入的 “Oct” 上，输出年份时集中在 “1993” 上。

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

#### Luong 注意力与训练细节

- Luong：$e^{\langle t,t'\rangle}=s^{\langle t\rangle\top} W h^{\langle t'\rangle}$（双线性）或点积 $s^{\top}h$，不必再过一层 tanh。
- 与 Bahdanau 的差别：能量函数，以及 $s$ 用 $t$ 还是 $t-1$。
- 作业实现 Bahdanau 即可；复杂度同为 $O(T_x T_y d)$。
- 翻译作业的 teacher forcing 使训练损失偏低；推理时第一步就错，后续会级联。
- Scheduled sampling：以概率把真值换成模型预测；是缓解手段，不是作业必做。

### 语音识别与触发词

#### 语音识别与 CTC

语音识别的输入是音频（常先转为声谱图），输出是文字。10 秒音频按 100 Hz 采样特征有 1000 个输入步，而转写文字只有几十个字符，$T_x\gg T_y$ 且对齐未知。

- 连接时序分类（Connectionist Temporal Classification, CTC）：允许输出与输入不对齐；用空白符与动态规划边缘化对齐。网络每个输入帧输出一个字符或空白符（blank），例如 `ttt_h_eee___ ___qqq__` 折叠为 “the q”。
- CTC 路径先合并连续重复符号，再删除 blank。例如 `- c c - a - t` 折叠为 `cat`。目标概率是所有能折叠成该标签序列的路径概率之和，不是只选一条对齐。前向后向动态规划（forward-backward dynamic programming）避免枚举指数级路径。
- CS230 课堂曾作为进阶讲座出现。

#### 触发词检测

触发词检测（trigger word detection）识别 “Alexa”、“Hey Siri” 这类唤醒词：

- 触发词：注意力或 RNN 输出接到每帧二分类。
- 触发词标签：每 $10\,\mathrm{ms}$ 一帧的 $0/1$，说完唤醒词之后的若干帧标为 1，其余标为 0。
- 正类极稀疏：需加权损失或下采样静音段。正帧极少：损失写成 $w_1 y\log\hat{y}+w_0(1-y)\log(1-\hat{y})$，或只在正例附近保留负例。一个简单的缓解办法是在唤醒词后连续多帧都标为 1。
- 评估用检测延迟与误触发率，不用帧级准确率。后者会因全预测 0 而虚高。

> [!NOTE]+ 衔接
>
> 到这里注意力仍依附 RNN 编码器-解码器，时间维必须串行计算。下一节让 $Q,K,V$ 都来自同一序列，称为自注意力，从而去掉循环；顺序改由位置编码写入。

<br>

## Transformer

架构、多头与位置编码的展开见 [Transformer论文精读](https://yukinoshitasherry.github.io/Transformer/)。

> [!INFO]+ Transformer
>
> Transformer（Vaswani et al., 2017）用自注意力代替循环：同一层里任意两个位置可以直接交互，训练可以对时间维并行。顺序不由递推产生，必须另加位置编码。复杂度 $O(T^{2}d)$。展开见精读笔记。

<br>

### 动机

复杂度一路上升：RNN → GRU → LSTM，每个单元越来越复杂，但都是**顺序**处理：第 $t$ 步必须等第 $t-1$ 步算完，每个单元都是信息流的瓶颈。CNN 则可以并行处理所有位置。Transformer 把注意力的“按需读取”与 CNN 的“并行计算”结合起来：

- **自注意力**（self-attention）：为序列中每个词计算一个结合了上下文的表示，所有词同时计算；
- **多头注意力**（multi-head attention）：把自注意力并行做多次。

### Scaled dot-product

#### 自注意力的直觉

“Jane visite l'Afrique en septembre.” 中 “l'Afrique” 的表示不应是固定的：它在这里是旅行目的地，在别的句子里可能是历史话题或经济话题。自注意力给每个词 $x^{\langle i\rangle}$ 算三个向量：

$$
q^{\langle i\rangle}=W^{Q}x^{\langle i\rangle},\qquad
k^{\langle i\rangle}=W^{K}x^{\langle i\rangle},\qquad
v^{\langle i\rangle}=W^{V}x^{\langle i\rangle}.
$$

借用数据库检索来理解：查询（query）$q^{\langle 3\rangle}$ 相当于为 l'Afrique 提出的问题“那里发生了什么？”；每个词的键（key）$k^{\langle j\rangle}$ 用来和查询比对，$q^{\langle 3\rangle}\cdot k^{\langle 1\rangle}$ 衡量 Jane 对这个问题有多相关（“访问”这个词可能最相关）；值（value）$v^{\langle j\rangle}$ 是被取出的内容。对内积做 softmax，再对值加权求和：

$$
A\big(q^{\langle 3\rangle},K,V\big)=\sum_{j}\frac{\exp\big(q^{\langle 3\rangle}\cdot k^{\langle j\rangle}\big)}{\sum_{l}\exp\big(q^{\langle 3\rangle}\cdot k^{\langle l\rangle}\big)}\,v^{\langle j\rangle}.
$$

结构与 Bahdanau 注意力相同（打分 → softmax → 加权和），区别在于打分用内积，而且查询、键、值都来自同一序列。

#### 矩阵形式

把所有位置的 $q,k,v$ 按行堆成矩阵，一次算完：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V.
$$

$Q,K\in\mathbb{R}^{T\times d_k}$，$V\in\mathbb{R}^{T\times d_v}$。不除 $\sqrt{d_k}$ 时，点积方差约为 $d_k$，softmax 进入饱和，梯度消失。除以 $\sqrt{d_k}$ 使尺度与 $d_k$ 无关（若 $q,k$ 各分量独立、均值 0、方差 1，则 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ 的方差为 $d_k$）。

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

因果掩码与 padding 掩码的构造，以及与 PyTorch 内置实现的对拍。注意 PyTorch 的布尔 `attn_mask` 中 **True 表示允许**，与上面“0 表示禁止”一致：

```python
def causal_mask(T):
    return np.tril(np.ones((T, T)))                    # 下三角：位置 t 只看 t' <= t

def padding_mask(lengths, T):
    """lengths: (m,) 每句真实长度。返回 (m, 1, T)，pad 位置为 0。"""
    return (np.arange(T)[None, :] < np.asarray(lengths)[:, None])[:, None, :].astype(float)

T, d = 5, 8
rng = np.random.default_rng(0)
Q, K, V = (rng.standard_normal((2, T, d)) for _ in range(3))
mask = causal_mask(T)[None] * padding_mask([5, 3], T)   # 两种掩码逻辑与
out_np, w = scaled_dot_product_attention(Q, K, V, mask)

out_t = F.scaled_dot_product_attention(
    torch.tensor(Q), torch.tensor(K), torch.tensor(V),
    attn_mask=torch.tensor(mask, dtype=torch.bool))
print(np.abs(out_np - out_t.numpy()).max())   # 约 2e-9：来自 NumPy 版分母上的 eps
print(np.round(w[1, 4], 2))                   # 第 2 句第 5 个位置只把权重分给前 3 个真实词
```

只有因果掩码时，可以直接写 `F.scaled_dot_product_attention(q, k, v, is_causal=True)`，它会在支持的 GPU 上自动调用 FlashAttention 等融合内核。

### 多头

一次自注意力只能问一个问题（“那里发生了什么？”）。多头注意力用 $h$ 组不同的 $W_i^{Q},W_i^{K},W_i^{V}$ 问 $h$ 个问题（“什么时候？”“谁？”……），每组称为一个头（head）。

- 把 $d_{\mathrm{model}}$ 拆成 $h$ 个头，每头 $d_k=d_{\mathrm{model}}/h$：

$$
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V),
$$

$$
\mathrm{MultiHead}=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O.
$$

- 不同头可以学到不同对齐，例如句法、共指、位置邻近。
- 各头之间没有依赖，可以并行计算。
- 实现上通常一次线性投射再 reshape 为 $(h,T,d_k)$，不必写 $h$ 个独立矩阵循环。

```python
def multi_head_attention(x, Wq, Wk, Wv, Wo, h, mask=None):
    """x: (m, T, d)；W*: (d, d)。一次投射后拆成 h 个头。"""
    m, T, d = x.shape
    d_k = d // h
    def split(Z):                                     # (m, T, d) -> (m, h, T, d_k)
        return Z.reshape(m, T, h, d_k).transpose(0, 2, 1, 3)
    Q, K, V = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    if mask is not None:
        mask = mask[:, None]                          # 在头维上广播
    heads, weights = scaled_dot_product_attention(Q, K, V, mask)   # (m, h, T, d_k)
    concat = heads.transpose(0, 2, 1, 3).reshape(m, T, d)        # 拼回 (m, T, d)
    return concat @ Wo, weights
```

PyTorch 的模块版本：

```python
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(2, 10, 512)
causal = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)   # True = 禁止！
out, attn = mha(x, x, x, attn_mask=causal, key_padding_mask=None)
print(out.shape, attn.shape)    # (2, 10, 512)，(2, 10, 10)：默认对头取平均
```

`nn.MultiheadAttention` 的布尔掩码约定与 `F.scaled_dot_product_attention` **相反**：这里 True 表示“禁止看”。混用这两个接口时，最常见的 bug 就是掩码取反。

### 块结构

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

- 编码器块重复 $N$ 次（原论文 $N=6$）。
- LayerNorm 沿特征维标准化，与 batch 大小无关，适合变长序列。
- Pre-LN 与 Post-LN 的稳定性不同；原论文为 Post-LN。
- 解码器每层多一块：对编码器输出的交叉注意力，以及带因果掩码（causal mask）的自注意力。解码器从 `<SOS>` 开始，把已经生成的词作为输入；第一个多头注意力用已生成的词产生 $Q$、$K$、$V$；第二个多头注意力的 $Q$ 来自解码器，$K$、$V$ 来自编码器输出；最后经线性层和 softmax 预测下一个词。
- 因果掩码使位置 $t$ 看不见 $t'>t$，训练可以并行，并与自回归推理一致。训练时整句译文已知，掩码把后面的词遮住，模拟“只看到前面”的推理过程。

三类注意力的 $Q,K,V$ 来源不同：

- 自注意力（self-attention）：$Q,K,V$ 来自同一序列。
- 掩码自注意力（masked self-attention）：来源仍相同，但 causal mask 禁止看未来。
- 交叉注意力（cross-attention）：$Q$ 来自解码器，$K,V$ 来自编码器输出。

原始 Transformer 常取前馈隐藏宽度 $d_{\mathrm{ff}}=4d_{\mathrm{model}}$。忽略偏置时，一层 FFN 的参数量为

$$
d_{\mathrm{model}}d_{\mathrm{ff}}
+d_{\mathrm{ff}}d_{\mathrm{model}}
=2d_{\mathrm{model}}d_{\mathrm{ff}}
\approx 8d_{\mathrm{model}}^{2}.
$$

现代架构的倍率可以不是 $4$，考试应按题目给出的 $d_{\mathrm{ff}}$ 计算。Pre-LN 把 LayerNorm 放在子层之前，深层梯度通常更稳定；Post-LN 放在残差相加之后，是原论文结构。

#### LayerNorm

LayerNorm 对最后一个特征维：

$$
\mu=\frac{1}{d}\sum_{j=1}^{d}x_j,\qquad
\sigma^{2}=\frac{1}{d}\sum_{j}(x_j-\mu)^{2},\qquad
\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^{2}+\varepsilon}}+\beta.
$$

- 与 BatchNorm 不同：不依赖 batch，训练与推理公式相同，无 running statistics。
- RNN 与 Transformer 几乎都用 LayerNorm。

#### 掩码

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

PyTorch 的编码器块，写成 Pre-LN 形式（现代实现的默认）；把两处 `self.ln*(x)` 挪到残差相加之后就是原论文的 Post-LN：

```python
class EncoderBlock(nn.Module):
    def __init__(self, d=512, h=8, d_ff=2048, p=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Dropout(p), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, pad_mask=None):          # pad_mask: (m, T)，True = pad
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(a)                       # 残差 1
        x = x + self.drop(self.ffn(self.ln2(x)))   # 残差 2
        return x

blk = EncoderBlock()
print(sum(p.numel() for p in blk.ffn.parameters()))   # 2·512·2048 + 2048 + 512 = 2099712 ≈ 8d²
```

内置的 `nn.TransformerEncoderLayer(d, h, d_ff, batch_first=True, norm_first=True)` 与上面等价；`norm_first=False` 为 Post-LN。

### 位置编码

> [!INFO]+ 位置编码
>
> 自注意力与输入的排列可交换，必须把位置写进表示。原论文把正弦、余弦加在嵌入上；后来常见可学习嵌入、相对偏置和 RoPE。序列长于训练长度时，正弦编码通常比绝对可学习嵌入更容易外推。

<br>

- 自注意力置换等变，必须加入位置信息：把句子中的词打乱顺序，每个词得到的注意力输出不变（只是跟着换位置）。“猫追狗”与“狗追猫”无法区分。
- 原论文用正弦：

$$
\mathrm{PE}(pos,2i)=\sin\big(pos/10000^{2i/d}\big),\qquad
\mathrm{PE}(pos,2i+1)=\cos\big(pos/10000^{2i/d}\big).
$$

- 正弦位置编码加在嵌入上（与词嵌入同为 $d$ 维）。每个位置得到一个唯一的 $d$ 维编码，类似二进制计数中低位变化快、高位变化慢。
- 正弦 $10000^{2i/d}$：不同维度波长从 $2\pi$ 到 $10000\cdot 2\pi$，同时编码短距与长距。
- 后续模型多用可学习位置嵌入或相对位置偏置（如 T5、RoPE）。
- 序列长于训练长度时，正弦编码可以外推；绝对可学习嵌入通常不能。
- 可学习位置嵌入在固定 $T_{\max}$ 内更灵活；超出 $T_{\max}$ 必须外推或插值。
- RoPE：旋转施加在 $Q,K$ 上，相对位置直接进入点积。
- 课程要求能说明：位置信息必须显式注入。

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

> [!INFO]+ 复杂度对比
>
> - 自注意力：$O(T^{2} d)$ 时间，$O(T^{2})$ 注意力矩阵。
> - RNN：$O(T d^{2})$，时间维串行。
> - 卷积核宽 $k$ 时 $O(T k d^{2})$，感受野受层数限制。
> - Transformer 一层即可覆盖全序列，代价是对长度的二次复杂度。

<br>

### 训练与推理

- 仅解码器 LM 的训练目标：下一步预测交叉熵。
- 推理为自回归：新 token 追加后再跑一遍，或用 KV cache 避免重算过去的键值。
- KV cache：第 $t$ 步只算当前 $q_t$，与已缓存的 $K_{1:t}, V_{1:t}$ 做注意力，把新 $k_t,v_t$ 追加。
- 复杂度从每步重复计算 $O(t^{2}d)$ 降到 $O(td)$。
- 作业不一定实现 cache，但应能解释为何推理时自回归生成比训练时的并行 teacher forcing 慢。
- Beam search、采样、top-$k$、nucleus（top-$p$）改变解码多样性。
- 作业用 TensorFlow 搭编码器-解码器 Transformer。
- 重点是 mask、shape 与残差是否对齐，不是从零训到 SOTA。

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
> Transformer 在 token 序列上做前向。课堂扩展转到输入空间：对抗样本沿 $\nabla_x J$ 上升，使分类出错；生成器沿判别器给出的方向下降，用来采样。二者都对输入 $x$ 求梯度，不再只更新权重 $W$。

<br>

# 课堂扩展

Coursera 五门课之外，CS230 课堂还讲了下面这些专题（按 2025 Autumn 的讲次排列）：

1. **决策案例**（Lecture 2）：拿昼夜分类、触发词、人脸验证、风格迁移四个小项目，练习“数据、输入、输出、架构、损失”怎么定；
2. **自监督与弱监督**（Lecture 2）：没有人工标签时，怎样让数据自己提供监督信号；
3. **项目全周期**（Lecture 3）：从定义问题、收集数据到部署、监控与维护；
4. **对抗与生成**（Lecture 4）：前面的训练都是对权重 $W$ 求梯度；这里转而对输入 $x$ 求梯度，既可以构造让模型出错的输入，也可以生成新数据；
5. **强化学习**（Lecture 5）：没有固定的标签 $y$，数据由智能体与环境交互产生，监督信号是奖励；
6. **读论文与职业**（Lecture 6 / 9）：怎样系统地读一个领域的文献，怎样规划知识结构；
7. **LLM 应用**（Lecture 8）：骨干已经预训练好，工作重心变成如何构造输入（提示、检索、工具）以及如何评估；
8. **内部分析**（Lecture 10）：检查模型依据什么做判断，用来决定下一步的实验。

<br>

## 决策案例（Case Studies）

### 一个项目要定的几件事

Lecture 2 的出发点是：**模型 = 架构 + 参数**。架构是选定的计算形式（逻辑回归、浅层网络、CNN……），参数是其中的 $W,b$。训练时，输入经过模型得到输出，与标签算出损失，再由梯度更新参数。在这个闭环里，可以改动的不只是参数，还有：

- 输入（分辨率、采样率、预处理）与输出（标签怎么编码）；
- 架构与激活函数；
- 损失函数；
- 优化器与超参数。

课上每个案例都按同一组问题来定：**要什么数据、要多少？输入是什么？输出是什么？用什么架构？用什么损失、怎么训练？** 这组问题的答案常常不是推导出来的，而是靠“类似项目的经验”和“拿人做实验”快速得到。

> [!INFO]+ 多分类与多标签的输出编码
>
> 猫分类器扩展成“猫 / 狗 / 长颈鹿”时，权重从 $(n_x,1)$ 变成 $(n_x,3)$，每列对应一个类别。标签有三种写法：
>
> - 整数标签 $y\in\{0,1,2\}$；
> - **one-hot**：猫图记作 $[1,0,0]$，每个标签向量恰好一个 $1$，配 softmax；
> - **multi-hot**：一张图里同时有猫和狗时记作 $[1,1,0]$，可以有多个 $1$，每个输出单元配一个 sigmoid、各自用二元交叉熵（与项目策略中的“多任务学习”例 12 相同）。
>
> 图像里可能同时出现多个类别时，one-hot 本身就写不出正确标签，必须用 multi-hot。

> [!NOTE]+ 编码与嵌入
>
> 网络中间层输出的向量称为**编码**（encoding）：浅层检测边缘，再往上组合出眼睛、耳朵，再往上才是整张脸。课上区分了两个词：任何向量表示都可以叫编码；当向量之间的**距离有含义**（同一个人的两张照片距离近，不同人的距离远）时，才称为**嵌入**（embedding）。

### 昼夜分类

见“项目策略 · 数据划分”中的例 3a。这个热身题要记住两点：

1. **记住各类项目大概需要多少数据**（猫分类约 1 万张），遇到新项目时可以类比估算；
2. **用人做实验**：把图缩成不同分辨率给人看，几分钟就能定下输入尺寸 $64\times 64\times 3$。是否保留颜色也可以这样判断：昼夜任务去掉颜色后人更难判断，所以保留 3 个通道。在另一些任务里（例如后面强化学习的 Atari 游戏），去掉颜色几乎不影响效果，训练却简单得多。

输入分辨率还要和硬件、迭代速度一起权衡：分辨率越高，每次实验越慢，一年内能做的实验次数就越少。

### 触发词检测

**问题**：给一段 10 秒音频，检测其中有没有说出 “activate”。

语音助手通常不是一个模型，而是一串由轻到重的模型，以节省能耗：

1. **活动检测**：极轻量，只判断有没有声音；
2. **触发词检测**：只听一个固定的词（“Alexa”“Hey Siri”“OK Google”），本案例做的就是这一层；
3. **完整理解**：较重的模型，理解用户要做什么。

**数据要覆盖什么**：

- 口音：课上第一版模型对说德语母语的人几乎不起作用，只能专门补录他们的数据；
- 年龄与性别：改变声音的频率分布；
- 语速：人耳觉得差不多，频谱上却差别很大；
- 背景噪声：只在校园里录，地铁上的用户就用不了。

触发词本身也要选好：常用词会让设备频繁误唤醒，课上说 “activate” 其实是个坏选择。音频的采样率不必自己推：找一个处理人声的开源项目，沿用它的超参数即可。

**标签怎么定**：课上做了一个人的实验。

- **方案一**：每段 10 秒音频只标一个 $0/1$，表示整段里有没有触发词。学生听了几段陌生语言的录音，几乎没人猜出触发词是什么。
- **方案二**：在触发词出现的时间段标 $1$，其余时间标 $0$。同样几段录音，学生很快听出了那个词。

对人更容易，对模型也更容易。方案一标起来省事，但模型可能需要多约 $1000$ 倍的数据才能学会，而方案二的标注成本远没有高出 $1000$ 倍，所以选方案二。它还能缓解冷启动：只有几千条手工数据时，方案一可能什么都学不到。

**类别不平衡**：10 秒里只有很短一段是 $1$，全预测 $0$ 也有 $99\%$ 以上的准确率。工程上的做法是在触发词结束后**连续多帧标 1**，让正负标签更平衡。

**输出、架构与损失**：每个时间步一个 sigmoid（“序列化的 sigmoid”），每步算二元交叉熵再求和；架构用 RNN，课上幻灯片给出的结构是多层 GRU、每层后接 BatchNorm，编程作业里最前面还有一层一维卷积（详见“序列与语言 · 语音识别与触发词”）。

**数据怎么造**：项目成败的关键在于数据收集策略，其次才是架构搜索与调参。

1. 建三个库：**正词**（录很多人说 “activate”）、**负词**（其他词，包括发音相近的 “deactivate”，以及 kitchen、lion、dog 等）、**背景噪声**（从网络视频里取音频，几乎免费）；
2. 录音时按词的起止精确剪切；使用网上数据时注意许可协议（CC BY、CC BY-NC、MIT、Apache 各自允许做什么）；
3. 写脚本：取一段 10 秒背景噪声，随机插入若干个互不重叠的负词和正词。脚本知道正词插在哪里，**标签自动生成**；
4. 插入时顺便做增强：改变速度、增强或削弱某些频段。

课上几个人拿手机在校园里录了约 3 小时，就合成出了数百万条训练样本：设有 1000 条正词、1 万条负词、无限多的背景噪声，组合数非常大。但合成数据**只用于训练**；test 要尽可能真实，需要手工录制和标注，好在 test 比 train 小得多。

### 人脸验证、识别与聚类

**问题**：学校想在食堂、健身房、游泳馆刷学生卡时，用人脸确认刷卡的人就是卡的主人。

- **数据**：每个学生一张带姓名的照片。
- **输入**：摄像头拍到的图，加上卡上声称的身份。人脸细节比夜空多，用同样的人眼实验得到的分辨率约 $412\times 412\times 3$。
- **输出**：是本人为 $1$，否则为 $0$。

**最简单的方案**是逐像素比较两张图，距离小于阈值就放行。它很快失效：光照不同、留了胡子、证件照过时，像素距离都会很大。

**课上的方案**：用一个深度网络把每张脸编码成向量（课上用 128 维），比较向量距离。数据库里预先存好每个学生的编码；刷卡时只对新照片跑一次网络，与卡主的编码比距离，例如距离 $0.4$ 小于阈值就判 $1$。

- 阈值决定了误拒与误放的折中，需要做一个小研究来选；不同场所对安全的要求不同，阈值也可以不同。
- 学校的数据不够训练编码网络，所以用**公开人脸数据集**训练。训练目标是：同一人的编码要近，不同人的编码要远。

于是训练样本是三元组：锚点 $A$、同一个人的正例 $P$、另一个人的负例 $N$。损失为

$$
\mathcal{L}=\max\Big(\big\lVert \mathrm{Enc}(A)-\mathrm{Enc}(P)\big\rVert_2^{2}-\big\lVert \mathrm{Enc}(A)-\mathrm{Enc}(N)\big\rVert_2^{2}+\alpha,\ 0\Big).
$$

- 第一项要小（拉近正例），第二项要大（推远负例），所以前减后；
- 间隔 $\alpha$ 防止平凡解：若网络把所有图都编码成同一个向量，两项都为 $0$，没有 $\alpha$ 时损失也是 $0$；
- 外面取 $\max(\cdot,0)$，让损失保持非负，已经满足间隔的三元组不再产生梯度。

负例只在训练时用到；部署时只有“新照片”和“库里的编码”，不需要负例。三元组如何挑选、FaceNet 的细节，见“卷积与视觉 · 人脸验证与识别”。

同一个编码网络，换一个用法就能完成另外两个任务：

- **识别**（identification）：没有卡，只有一张摄像头照片，要找出这个人是谁。把库中所有学生的编码与新编码比较，用 **k 近邻**（k-nearest neighbors）找最近的几个向量；遍历整个库的复杂度为 $O(n)$。每个学生存 3 张照片时，可以要求最近的 3 个向量都属于同一个人，才认定身份，更安全（机场这类场景可以用 10 近邻）。不在库里的人，最近距离也会超过阈值，不放行。
- **聚类**（clustering）：手机相册把同一个人的照片自动归到一个文件夹。先把所有照片编码，再跑 **k-means**；新照片与各簇的中心（centroid）比距离，归入最近的那一簇。k-means 是无监督算法，它能工作的前提是编码网络已经训练好。

### 风格迁移

**问题**：给一张内容图和一张风格图，生成一张“内容像前者、风格像后者”的图。

- **架构**：用一个在 ImageNet 上预训练好的网络，它已经很会“理解”图像，不需要重新训练；
- **内容**：取某几层的激活作为内容表示；**风格**：用激活的 Gram 矩阵作为风格表示；
- **损失**：
$$
\mathcal{L}=\big\lVert \mathrm{Content}_C-\mathrm{Content}_G\big\rVert+\big\lVert \mathrm{Style}_S-\mathrm{Style}_G\big\rVert,
$$
  其中 $G$ 是生成图，$C$ 是内容图，$S$ 是风格图；
- **训练什么**：与前面所有案例不同，这里网络参数固定，**被梯度更新的是生成图的像素**。从噪声或内容图出发，迭代多次。

完整的公式与实现见“卷积与视觉 · 神经风格迁移”。

> [!NOTE]+ 四个案例的共同点
>
> - 数据量和输入尺寸都可以靠“类比旧项目”和“拿人做实验”在几分钟内定下来；
> - 损失函数要对准真正想要的东西：昼夜和触发词是逐个预测的交叉熵，人脸是距离关系（三元组），风格迁移是生成图与两张参考图的差距；
> - 对人更容易的标注方式，往往也让模型学得更快；
> - 收集数据的策略，往往比架构选择更决定项目成败。

### AI 与医疗

课程中关于医疗的一讲（嘉宾 Pranav Rajpurkar）用几个斯坦福的项目说明：把上面的决策流程搬到医疗场景时，多出了哪些问题。

- **CheXNet**（Rajpurkar et al., 2017）：输入胸部 X 光片，输出 14 种疾病的概率。数据是 NIH 的 ChestX-ray14（约 11 万张图，标签由放射报告自动抽取，本身带噪声）。模型是 121 层 DenseNet，从 ImageNet 预训练权重迁移，最后一层换成 14 个 sigmoid 输出（多标签，每种疾病一个二分类交叉熵）。在肺炎检测上，与 4 位放射科医生在一个由医生标注的测试集上比较，F1 达到或超过医生平均水平；配合 CAM 热力图指出病灶位置。
- **心电图心律失常检测**（Hannun et al., 2019）：输入单导联动态心电信号（一维时间序列，约 5.4 万名患者的 9.1 万段记录），用 34 层的**一维卷积残差网络**，每隔一段时间输出一个节律类别（共 12 类），即“序列到序列”的逐段分类。测试集由多位心脏科医生共同标注，模型的平均 F1 超过了单个心脏科医生。
- **MRNet**（Bien et al., 2018）：膝关节 MRI 是三维、多个切面的数据。做法是把每个切面的每一张切片都用 2D CNN（AlexNet）提特征，在切片维上做最大池化，再用逻辑回归融合三个切面的结果。数据只有 1,370 例，靠预训练与多切面融合弥补数据不足。

这些项目共同的经验：

- **测试集的标签要比训练集更好**：训练标签可以从报告中自动抽取，测试集必须由多位专家标注、取多数票，否则“超过医生”无从谈起。这对应“项目策略”中的人类水平与 Bayes 误差：用一组专家的共识来近似 Bayes 误差；
- **指标按临床需求选**：筛查要高召回，确诊要高精确；多标签任务按每种疾病分别报告 AUROC；
- **跨医院的分布不匹配**：Zech et al.（2018）发现，胸片模型会利用图上的医院标记、设备类型等与疾病无关的信号，换一家医院准确率明显下降。这正是“分布不匹配”与“抄捷径”的真实案例，要用外部医院的数据做 test，并用 Grad-CAM 检查热力图是否落在病灶上；
- **人机协作**：临床部署时模型通常是辅助而不是替代，要评估“医生 + 模型”比“医生单独”好多少，而不只是“模型对医生”。

<br>

## 自监督与弱监督（Self- and Weakly Supervised Learning）

### 为什么需要

上面四个案例都是**监督学习**：昼夜要人标 $0/1$，触发词要知道词的位置，人脸验证要知道哪些照片属于同一个人。标注很贵。如果只有大量“野生”的、没有标签的人脸照片，怎样训练出一个好的编码网络？

### 对比学习

思路是**让数据自己提供监督信号**。把一张狗的照片旋转 $90^\circ$，人仍然认得出这是同一只狗；遮掉一张脸的左半边或右半边，剩下的部分仍属于同一个人。于是：

1. 取一张无标签的图，用数据增强（旋转、裁剪、平移、加噪声、遮挡……）生成它的两个变体；
2. 要求网络对这两个变体给出**相近**的向量，对其他图的变体给出**远离**的向量。

这类方法称为**对比学习**（contrastive learning），代表工作是 SimCLR。与 FaceNet（2015）相比，它不再需要人工构造的身份三元组，只需要“同一张图的两个变体”这样的正样本对。不需要标注，所以可以在数十亿张无标签图片上预训练，难点主要在算力。这样训出来的模型，即使训练时从没定义过“人脸验证”这个任务，学到的表示也能区分不同的人脸。

SimCLR 的损失称为 NT-Xent（normalized temperature-scaled cross entropy）。一个 batch 有 $N$ 张图、$2N$ 个变体；对变体 $i$，它的另一个变体 $j$ 是正例，其余 $2N-2$ 个是负例：

$$
\ell_{i}=-\log\frac{\exp(\mathrm{sim}(z_i,z_j)/\tau)}{\sum_{k\neq i}\exp(\mathrm{sim}(z_i,z_k)/\tau)},\qquad
\mathrm{sim}(u,v)=\frac{u^{\top}v}{\lVert u\rVert\,\lVert v\rVert}.
$$

它就是一个 $2N-1$ 类的 softmax 分类：在所有其他变体中认出“自己的另一半”。

```python
def nt_xent(z1, z2, tau=0.5):
    """z1, z2: (N, d)，同一批图的两组增强变体经编码器 + 投影头的输出。"""
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1)          # (2N, d)，余弦相似度 = 点积
    sim = z @ z.T / tau                                   # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))                     # 排除 k = i
    target = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)])   # i 的正例下标
    return F.cross_entropy(sim, target)

# 训练一步：
# x1, x2 = augment(x), augment(x)
# loss = nt_xent(proj(encoder(x1)), proj(encoder(x2)))
```

- batch 越大，负例越多，效果越好：SimCLR 用到 4096 的 batch，这正是“难点在算力”的来源。
- 下游任务丢掉投影头 `proj`，只用 `encoder` 的输出做线性分类（linear probe）或微调。

### 预测下一个词

文本上的自监督是 GPT 的核心：**预测下一个词**（next token prediction）。从网上抓来的句子本身就是标签，不需要人工标注。课上用几个填空说明，单单这一个目标，就逼着模型学会了很多从没被单独教过的东西：

| 句子 | 模型被迫学到的 |
| :--- | :--- |
| I poured myself a cup of \_\_\_（tea / coffee / water） | 能“倒”的是液体，而且要装得进杯子；tea 与 coffee 出现在相似语境里，所以表示相近 |
| The capital of France is \_\_\_ | 事实性知识 |
| She unlocked her phone using her \_\_\_（face / fingerprint / password） | 语义：这几样都能用来解锁 |
| The cat chased the \_\_\_（mouse / ball / dog） | 概率性推理：多个答案都合理，各有频率 |
| If it's raining, I should bring an \_\_\_ | 条件与推断 |

这种从简单目标、大规模数据中出现的、没被显式教过的能力，称为**涌现行为**（emergent behavior）。后面强化学习里，智能体学出设计者没想到的策略（例如 AlphaGo），也是同一个现象。

同样的“遮掉一部分再让模型补全”可以用到别的模态：

- 音频：遮掉 20 个时间步，让模型补全；
- 视频：抽掉若干帧；
- 生物：遮掉蛋白质结构或 DNA 序列的一段。

### 弱监督与多模态

要把文本和图像连到同一个向量空间，就需要成对的数据，例如图片配说明文字。这些配对不是专门为训练而标的，而是世界上**自然同时出现**的（社交网络上的图片配文）。这种学习称为**弱监督**（weakly supervised learning）。

课上学生找出的自然配对还有：

- 视频的画面与声音（狗叫的画面配狗叫的声音）；
- 电影的画面与字幕；
- 音乐与歌名；
- 基因型与表型；
- 同一次检查得到的几种医学影像；
- 游戏画面与玩家的键盘操作。

Meta 的 **ImageBind** 进一步指出，不需要每两种模态之间都有配对：多数模态都能通过图像（或文本）连起来，例如热成像与图像配对，图像又与文本配对，于是热成像也间接和文本对齐。结果是所有模态被“绑”进同一个嵌入空间：输入文字 “drums”，可以检索到鼓声的音频和鼓的图片；输入一段狗叫声加一张海滩图，可以检索到“海滩上的狗”。

> [!NOTE]+ 本讲出现的几个概念
>
> - 嵌入：距离有含义的向量表示；
> - 自监督：标签由数据本身构造，例如对比学习、预测下一个词、遮挡后补全；
> - 弱监督：利用世界上自然出现的配对，例如图片与配文；
> - 共享嵌入空间：不同模态映到同一空间，文本通常是连接各模态的枢纽。

<br>

## 项目全周期（Full Cycle of a DL Project）

### 流程

Lecture 3 由 Ng 主讲，以一个门禁人脸识别系统为例：员工刷卡后，系统再确认刷卡的人就是卡的主人，防止有人捡到或偷到卡就能进门。典型流程是：

1. **定义问题**；
2. **收集数据**；
3. **设计模型**；
4. **训练模型**；
5. 分析结果，回到第 2 或第 3 步修改数据或模型，**反复迭代**，直到效果足够好；
6. **部署**；
7. **监控与维护**。

人脸识别常用的结构是**孪生网络**（Siamese network）：输入两张图，输出是否同一个人。这样每个新用户只需拍几张注册照，不必为每户人家重新训练一个网络。

### 数据：先快，再好

课上的问题是：一家三人创业公司的 CTO，被律师告知不能从网上下载人脸数据，该怎么收集？花多久才开始训练？

Ng 的原则是**速度**。他认为执行速度是创业公司（以及大公司内部的小型创新项目）能否成功的最强预测因素之一，所以倾向于选择能在一两天内拿到数据的办法，哪怕数据更少、质量更差：

- 给团队定一个期限：现在是周二 11:52，问团队周四 11:52 前能拿到什么数据，而不是先问需要什么数据、要多久；
- 在人流量大的地方（例如学校食堂）征得知情同意后拍照、录音；尊重隐私，只采集愿意参与者的数据；
- 训练一个模型只需要几小时到几天时，就不该花几个月收集数据。数据收集的时间应当和训练时间相当；小模型的团队常常一天就迭代一轮。

理由是：**事先很难知道数据里有什么问题，也很难知道哪部分数据有价值**。Ng 讲过一个例子：有位 CEO 花了上亿美元收购一家公司，只为得到它的数据，事后却问 Ng 这些数据怎么变现。越早拿到数据、训出模型、看到它错在哪里，就越早知道数据该怎么改。

例外是做过很多次的同类项目：凭经验或文献已经知道，比如某种人脸识别系统至少要 5 万张图才有希望，就可以提前投入去收集。对第一次做的问题，优先保证迭代速度。

基于 LLM API 的应用也一样：提示词的效果事先无法预测，因为没人看过模型的训练数据，与其长时间琢磨，不如直接试，看到问题再改。安全性也是如此：先在沙盒里构建并反复探测系统，找出它会在哪里说出不当的话，测试充分之前不放给真实用户。

### 数据不是一整块

模型效果不好时，有两条路：

1. 改模型：查文献、换架构；
2. **改数据**：系统地改进数据以提升效果，称为**以数据为中心的 AI**（data-centric AI）。

例如误差分析发现系统认不出戴帽子的人，就专门补充戴帽子的数据。不分青红皂白地“收集更多数据”通常低效：长发、短发、胡子、围巾、眼镜、侧脸……可补的子类太多，不先建系统、看错在哪，就无从决定该补哪一类。前沿大模型也不是“什么数据都抓”，而是找出值得投入的子类别。例如要提高写代码的能力，就专门构造高质量的代码数据，而不是更多随意的网络聊天。

数据质量同样重要：编辑过、有见地的书籍文章，比随意的网络聊天更适合训练语言模型；清晰对焦的人脸，比模糊的人脸更有用。

> [!INFO]+ 训练数据一定要和测试同分布吗
>
> Ng 的回答是：重要，但没有多数人想的那么重要。10–15 年前，模型小、参数少，人们担心无关数据会“分散”模型的容量，所以执着于训练与测试同分布。今天网络足够大时，加入一些相关性不强的数据（只要不是**错误**的数据），通常不会有害，甚至略有帮助。例如在人脸数据里加入长相与真人差别很大的卡通人物，大概率无害。帮多少仍要靠实验。
>
> 他的类比是：学弹钢琴并不会让人做 AI 变差，因为大脑足够大。这与福尔摩斯的“阁楼理论”（脑容量有限，学新东西就得忘掉旧东西）相反，后者只在模型很小时才成立。
>
> 这一条与项目策略中“dev/test 必须与部署同分布”并不矛盾：放宽的是**训练集**，dev/test 仍要对准部署分布。

### 部署

模型训好后，还有大量软件工程工作：把模型放到云端或本地服务器上，接收图片、返回“开不开门”。

实际系统往往不能把摄像头视频 24 小时、每秒 30 帧地传到云端逐帧识别，太贵也太慢。家门口的画面大部分时间只是一面墙，什么都没发生。常见做法是在前面加一级**视觉活动检测**（visual activity detection, VAD）：一个低成本、低功耗的模块，先判断画面里有没有人，有人才把图片发给更重的人脸识别网络。VAD 有两种实现：

| 方案 | 做法 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| 一：非机器学习 | 与一秒前相比，RGB 变化超过阈值的像素若超过 $10\%$，就认为有动静 | 几行 Python，20 分钟写完；只有一个参数 $\varepsilon$ | 树叶晃动、路过的车、邻居的猫都会触发 |
| 二：小神经网络 | 训练一个很小的网络，只判断画面里有没有人 | 更准 | 要数据、要训练，至少花几个小时 |

课上学生提出了各种组合（先用方案一过滤再接方案二、先用一后换二）。Ng 的做法仍然是**先做最快的**：先上方案一，部署出去看它在哪里失败。实际经验是，方案一确实不够好，最后会换成方案二；但先做方案一能更早发现问题，例如：

- 人走向摄像头时，有的帧清晰，有的帧因为脸移动快而模糊。专门挑出几帧（例如 5 帧）清晰对焦的高分辨率图送去识别，准确率显著提高；
- 这类发现只有真正搭出系统后才会出现，它又会引出新的子任务：训练一个网络，既判断有没有脸，也判断这张脸是否对焦。

VAD 与识别网络最好都能在**边缘**（edge，即设备本地）运行；只是每分钟上传几张图，则放在云端也没问题。

模型精度多少才算好？最好有一个参照水平，通常是人类水平（见“项目策略 · 人类水平”）。受控环境下，现在的人脸识别系统已经比大多数人更会判断两张照片是不是同一个人；推荐电影、书籍这类连人都做不好的任务，就很难确定参照水平。

### 监控与维护

模型在训练集、测试集上都很好，部署后却因为**世界变了**而失效：

- **数据漂移**（data drift）：输入 $x$ 的分布变了。例如冬天下雨，人们戴围巾、穿雨衣；夏天戴墨镜的人变多；系统从加州部署到穿着、外貌不同的国家；
- **概念漂移**（concept drift）：输入到输出的映射 $x\mapsto y$ 本身变了。

更多例子：

- **搜索引擎**：新政治人物当选、新视频走红、新专辑发布，用户突然大量搜索新的东西；
- **工厂质检**：换了材料或新装了一台机器，手机外壳上出现新的划痕类型；
- **自动驾驶**：在加州训练的模型到了得州就不好用了。得州部分地区风很大，红绿灯的悬挂方式不同（横放或竖放），外观与加州差别很大。

**简单模型更抗漂移**。VAD 方案一只有一个参数 $\varepsilon$（变化像素的比例），夏天大家都戴墨镜，变化像素的比例也不会变多少，所以它很稳健。有成千上万个参数的神经网络，若训练时没人戴墨镜，就更可能过拟合到“不戴墨镜”，分布一变就得更新。

**怎么监控**：

1. 在用户知情同意、隐私做法透明的前提下，把少量数据传回云端；
2. 召集团队，尽量多元地**头脑风暴所有可能出错的地方**，包括数据分布的变化。Ng 说，认真做过这一步之后，他几乎没见过真实世界里出现“清单上没有”的问题，安全攸关的应用同样适用；
3. 据此设计**大量**监控面板（dashboard），例如：
   - 用户需要第二次验证的频率（往往说明第一次失败了，用户已经不耐烦）；
   - 放行与拒绝的比例；
   - 系统延迟；
4. 事先很难知道哪些面板有用。先画很多（30、50、100 个都可以），几天或几周后删掉无聊的（例如云端部署的延迟一直很平稳），保留少数长期跟踪的指标；
5. 摸清各指标的正常范围后，设置**上下限告警**，越界就请人检查。

发现漂移后的维护：收集能代表新分布的数据，刷新训练集，更新模型。

> [!WARNING]+ 工作是做出能用的系统，不只是测试集上的高分
>
> Ng 见过不少工程师认为“测试集上做得很好，工作就完成了”，而产品负责人的反馈是“系统不能用”。他的建议是：把自己的工作定义为**做出真正能用的系统**。测试集表现好、实际表现差时，应当追查原因并修正，常见原因就是数据分布已经改变，需要更新训练数据。这对应项目策略里的正交化目标“让数字对部署有意义”。

<br>

> [!NOTE]+ 衔接
>
> 部署后的输入会自然漂移，也可能被人**故意**改动。下一节的对抗样本就是后者：对输入 $x$ 求梯度，构造人眼看不出、却能让模型出错的扰动；同一套“对输入求梯度”的工具，也能用来生成新数据。

<br>

## 对抗与生成（Adversarial and Generative Models）

对抗样本在输入上加很小的扰动，分类器就会以高置信度出错。生成模型学习 $P(x)$ 或 $P(x\mid y)$，用来采样，而不是只输出标签。二者都对 $x$ 求梯度，不再只更新 $W$。内容来自 2025 Autumn Lecture 4；Coursera C1–C5 没有对应编程作业。

- 需要能写出 FGSM / PGD 与对抗训练，并能对上 VAE、GAN、扩散三条谱系。
- 强化学习进一步去掉固定标签 $y$，改用回报。

### 对抗样本

> [!INFO]+ FGSM
>
> FGSM（Goodfellow et al., 2015）沿 $\mathrm{sign}(\nabla_x J)$ 对输入走一小步，得到扰动很小、分类却错的 $x_{\mathrm{adv}}$。PGD 是同一方向上的多步投影。对抗训练把 $x_{\mathrm{adv}}$ 加入训练集。只报告 FGSM 准确率会高估稳健性。

<br>

#### 现象

> [!EXAMPLE]+ 例1 熊猫与长臂猿
>
> Goodfellow 等的经典例子：一张被 GoogLeNet 以 $57.7\%$ 置信度识别为熊猫的图像，加上 $\varepsilon=0.007$ 倍的符号梯度噪声后，人眼看不出任何区别，模型却以 $99.3\%$ 的置信度判为长臂猿。

对抗样本（adversarial example）对安全敏感的系统是现实威胁：人脸门禁、自动驾驶的路标识别、内容审核都可能被精心构造的输入欺骗。

#### 为什么高维线性模型脆弱

线性分类器 $w^{\top}x$ 在高维中对沿 $w$ 方向的小扰动敏感。取扰动 $\eta=\varepsilon\,\mathrm{sign}(w)$，每个分量的绝对值都不超过 $\varepsilon$（$\|\eta\|_\infty=\varepsilon$），但

$$
w^{\top}(x+\eta)=w^{\top}x+\varepsilon\|w\|_1.
$$

若 $w$ 有 $n$ 维、每维平均大小为 $m$，输出改变约 $\varepsilon mn$：**每个像素只改一点点，但 $n$ 个像素的改变同方向累加**，维度越高，越容易跨过决策面。神经网络中大量使用 ReLU 这类分段线性的单元，局部表现得很像线性模型，所以同样脆弱。

#### FGSM 与 PGD

把 $w$ 换成损失对输入的梯度，就得到快速梯度符号法（fast gradient sign method, FGSM）：

$$
x_{\mathrm{adv}}=x+\varepsilon\,\mathrm{sign}(\nabla_x J(\theta,x,y)).
$$

$\varepsilon$ 很小（图像上常对应每个像素几级灰度）即可跨过决策面。注意这里固定参数 $\theta$，对**输入**求梯度并沿**上升**方向走，与训练时沿下降方向更新 $\theta$ 正好相反。迭代 FGSM / PGD（projected gradient descent）在 $\ell_{\infty}$ 球内多步投影：

$$
x^{k+1}=\Pi_{\mathcal{B}\_{\varepsilon}(x)}\big(x^{k}+\alpha\,\mathrm{sign}(\nabla_x J)\big).
$$

投影 $\Pi$ 把每步结果裁回 $[x-\varepsilon,x+\varepsilon]$ 内，保证扰动始终足够小。

- 定向攻击（targeted attack）：想让模型判成指定类别 $y_{\mathrm{target}}$ 时，改为沿 $-\nabla_x J(\theta,x,y_{\mathrm{target}})$ 下降。
- 白盒攻击能拿到梯度；黑盒攻击拿不到时，可以在替代模型上构造对抗样本，它们往往能迁移到目标模型。

```python
def fgsm(x, grad_x, eps):
    """grad_x = dJ/dx，与 x 同形。像素若在 [0,1] 则裁剪。"""
    x_adv = x + eps * np.sign(grad_x)
    return np.clip(x_adv, 0.0, 1.0)
```

PyTorch 中 $\nabla_x J$ 由 autograd 直接给出：对输入设 `requires_grad`，模型参数保持不动。

```python
def fgsm_torch(model, x, y, eps):
    model.eval()                                    # 攻击时用推理态的 BN / Dropout
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad_x, = torch.autograd.grad(loss, x)          # 只对输入求导，不污染参数的 .grad
    return (x + eps * grad_x.sign()).clamp(0, 1).detach()

def pgd_torch(model, x, y, eps=8/255, alpha=2/255, steps=10, random_start=True):
    model.eval()
    x_adv = x.clone().detach()
    if random_start:                                # 在 ε 球内随机起点，攻击更强
        x_adv = (x_adv + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)   # 投影 Π
    return x_adv.detach()

@torch.no_grad()
def robust_accuracy(model, loader, attack, **kw):
    correct = total = 0
    for x, y in loader:
        with torch.enable_grad():                   # 攻击本身需要梯度
            x_adv = attack(model, x, y, **kw)
        correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total
```

对抗训练就是把内层 max 换成 PGD，外层照常训练：每个 batch 先 `x_adv = pgd_torch(model, x, y)`，再 `model.train()`，对 `F.cross_entropy(model(x_adv), y)` 做一步优化。

#### 把攻击写成优化问题

课上用更一般的形式统一各种攻击：固定网络参数，**把输入当作变量**，设计一个关于输入的损失再做梯度下降。例如想让一张猫的图被判成鬣蜥（iguana），同时看起来仍是猫：

$$
x^{\ast}=\arg\min_{x}\ \underbrace{\big\lVert\hat y(x)-y_{\mathrm{iguana}}\big\rVert_2^2}_{\text{判成目标类}}+\lambda\underbrace{\big\lVert x-x_{\mathrm{cat}}\big\rVert_2^2}_{\text{离原图近}}.
$$

第一项把预测推向目标类，第二项把图像约束在原图附近；$\lambda$ 越大，扰动越不可见，攻击越难成功。FGSM、PGD 是这个思路在 $\ell_\infty$ 约束下的快速近似。只保留第一项、从随机噪声出发，得到的是一张人看来毫无意义、网络却以高置信度判为鬣蜥的图：网络的决策区域远大于自然图像所在的区域。

#### 物理世界中的攻击

对抗样本不只存在于像素层面：

- **打印再拍照**：把对抗图片打印出来，再用手机拍摄，很多仍能骗过分类器（Kurakin et al., 2016），说明扰动能经受一定程度的光照、角度、压缩变化；
- **贴纸**：在停车标志上贴几块黑白胶条，可以让交通标志识别模型稳定地判成限速标志（Eykholt et al., 2018）；
- **对抗补丁**（adversarial patch）：一块打印出来的彩色图案放在任意物体旁边，就能让分类器输出“烤面包机”（Brown et al., 2017）；
- **眼镜框**：戴上特制图案的眼镜框，可以让人脸识别系统认成另一个人。

这些攻击都要在优化目标里对**一族变换**取期望（随机旋转、缩放、光照，称为 expectation over transformation），使扰动在各种拍摄条件下都有效。它们说明对抗稳健性是自动驾驶、人脸门禁等安全攸关系统必须单独测试的指标。

#### 防御

- 对抗训练：把 $x_{\mathrm{adv}}$ 加入训练集，最小化最坏扰动下的损失，是 min-max：

$$
\min_{\theta}\ \mathbb{E}_{(x,y)}\Big[\max_{\|\eta\|_\infty\le\varepsilon}J(\theta,x+\eta,y)\Big].
$$

- 代价：训练变慢；对攻击范数外的扰动不一定迁移。
- 防御不能只靠把输入模糊一下：自适应攻击会把防御编进梯度。
- 评估必须用强攻击（PGD、AutoAttack），不能只报 FGSM 准确率。

### 生成模型

> [!INFO]+ VAE、GAN、扩散
>
> 生成模型学习 $P(x)$，用来采样，不是只输出标签。VAE（2014）最大化 $\log p(x)$ 的 ELBO，重参数化让采样可微。GAN（2014）让生成器与判别器对打，采样快，训练不稳，似然不好算。扩散模型先加噪再学去噪，训练稳，采样步数多。课堂要能对上谱系，不必推完 SDE。

<br>

#### 监督与无监督

之前的问题都是**监督学习**（supervised learning）：数据集以“输入 + 标签”成对给出。生成模型只拿到大量样本（比如一堆卧室照片），没有标签，要学的是数据本身的分布 $P(x)$，这属于**无监督学习**（unsupervised learning）。学会之后，从 $P(x)$ 中采样就得到训练集中不存在的新样本。

#### 四条谱系

- **自回归**：$P(x)=\prod_t P(x_t\mid x_{<t})$，像素 RNN、GPT 属此类。似然可精确计算，采样是串行的（前一部分的语言模型就是这一类）。
- **VAE**：编码器 $q(z\mid x)$，解码器 $p(x\mid z)$，最大化 ELBO（evidence lower bound）
  $$
  \mathbb{E}\_{q}[\log p(x\mid z)]-\mathrm{KL}(q(z\mid x)\|p(z)).
  $$
  重参数化 $z=\mu+\sigma\odot\varepsilon$，$\varepsilon\sim\mathcal{N}(0,I)$，使采样路径可微。
- **GAN**：生成器 $G(z)$ 与判别器 $D(x)$。原目标
  $$
  \min_G\max_D \mathbb{E}\_{x}[\log D(x)]+\mathbb{E}\_{z}[\log(1-D(G(z)))].
  $$
  训练不稳定：模式崩溃、梯度消失。WGAN 等用不同距离缓解。似然不可直接得。
- **扩散**：逐步加噪到高斯，再学逐步去噪。训练稳定，采样步数多。课堂以直觉与应用为主，不要求推导全部 SDE。

- 条件生成（text-to-image、语音 vocoder）把 $y$ 注入归一化层或注意力。
- 评估用 FID、IS、人工偏好，不要只用训练损失。

#### VAE

> [!INFO]+ VAE 的 ELBO
>
> - $\log p(x)=\log\int p(x\mid z)p(z)\,\mathrm{d}z$。
> - 引入 $q(z\mid x)$ 后
>
> $$
> \log p(x)=\mathrm{KL}(q\|p(z\mid x))+\mathbb{E}\_{q}[\log p(x\mid z)]-\mathrm{KL}(q(z\mid x)\|p(z)).
> $$
>
> - KL 非负，故 ELBO 是 $\log p(x)$ 的下界。
> - 最大化 ELBO = 重构 − 正则。写成要最小化的损失，就是 $\mathbb{E}\_{q}[-\log p(x\mid z)]+\beta\,\mathrm{KL}(q(z\mid x)\|p(z))$，$\beta=1$ 时与 ELBO 只差一个符号。定义、不对称性与另外三种用法见 [信息论基础 · 相对熵](/信息论熵/#kl)。
> - $q=\mathcal{N}(\mu(x),\mathrm{diag}(\sigma^{2}(x)))$、$p(z)=\mathcal{N}(0,I)$ 时，KL 有闭式
>
> $$
> \frac{1}{2}\sum_j\big(\mu_j^{2}+\sigma_j^{2}-\log\sigma_j^{2}-1\big).
> $$
>
> - 重参数化使 $\partial/\partial\mu$、$\partial/\partial\sigma$ 能穿过采样。
> - KL 过强：后验坍缩到先验，解码器忽略 $z$（posterior collapse）。

<br>

两项的含义：重构项要求从 $z$ 解码回来的样本像原图；KL 项要求各样本的 $q(z\mid x)$ 都接近标准正态，使隐空间连续、规整。训练好后，从 $\mathcal{N}(0,I)$ 采样 $z$ 送入解码器即可生成新样本。

```python
def vae_elbo(x, mu, logvar, decoder_logpx, kl_weight=1.0):
    """返回要最大化的 ELBO；decoder_logpx 是 log p(x|z) 的像素和。"""
    recon = decoder_logpx.mean()
    kl = 0.5 * np.mean(np.sum(mu ** 2 + np.exp(logvar) - logvar - 1.0, axis=1))
    return recon - kl_weight * kl

def vae_loss(x, mu, logvar, decoder_logpx, kl_weight=1.0):
    """优化器最小化负 ELBO。"""
    return -vae_elbo(x, mu, logvar, decoder_logpx, kl_weight)

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

PyTorch 版 VAE（MNIST，输入展平为 784 维、像素在 $[0,1]$）。编码器输出 $\mu$ 与 $\log\sigma^2$ 而不是 $\sigma$，这样网络输出无需约束为正；重构项取伯努利似然，即逐像素 BCE：

```python
class VAE(nn.Module):
    def __init__(self, d_in=784, d_h=400, d_z=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU())
        self.mu, self.logvar = nn.Linear(d_h, d_z), nn.Linear(d_h, d_z)
        self.dec = nn.Sequential(nn.Linear(d_z, d_h), nn.ReLU(), nn.Linear(d_h, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)    # 重参数化：随机性只在 ε
        return self.dec(z), mu, logvar                              # dec 输出 logits

def vae_loss_torch(logits, x, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1).mean()
    return recon + beta * kl                                        # = 负 ELBO

@torch.no_grad()
def vae_sample(model, n=16, d_z=20):
    z = torch.randn(n, d_z)                                         # 从先验 N(0, I) 采样
    return torch.sigmoid(model.dec(z)).view(n, 1, 28, 28)
```

若写成 `z = torch.normal(mu, std)`，采样操作不可导，梯度无法传回编码器；重参数化把随机性移到与参数无关的 $\varepsilon$ 上。

#### GAN

GAN（generative adversarial network）由两个网络组成：

- **生成器** $G$（generator）：输入随机噪声 $z$，输出一张假图 $G(z)$，目标是以假乱真；
- **判别器** $D$（discriminator）：输入一张图，输出它是真图的概率，目标是分辨真假。

两者以竞争的方式学习：生成器学到越来越精妙的“作假技术”，判别器成长为越来越精准的“鉴定师”，切磋到最后，生成器可能画出足以乱真的图像。

> [!EXAMPLE]+ 例2 DCGAN 生成卧室（鱼书 8.5.2 节）
>
> DCGAN（Deep Convolutional GAN）用大量卧室照片训练后，可以从零生成新的卧室图像。这些图像看上去像真照片，但训练集中并不存在，是谁都没有见过的新图。DCGAN 的生成器和判别器都是卷积网络：生成器用转置卷积把噪声向量逐步放大成图像，判别器是普通的分类 CNN。

**最优判别器**：固定 $G$，对每个 $x$ 最大化 $p_{\mathrm{data}}(x)\log D(x)+p_g(x)\log(1-D(x))$。对 $D$ 求导令其为零，得

$$
D^{\ast}(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}.
$$

代回目标，$G$ 最小化的是 $-\log 4+2\,\mathrm{JS}(p_{\mathrm{data}}\|p_g)$。JS 散度非负，当且仅当 $p_g=p_{\mathrm{data}}$ 时取到全局最小值 $-\log 4$，此时 $D^{\ast}\equiv\frac{1}{2}$：判别器完全分不出真假。

**非饱和损失**：训练初期生成器很差，$D(G(z))\approx 0$，此时 $\log(1-D(G(z)))$ 在 $0$ 附近非常平坦，梯度几乎为零，生成器学不动。改为最大化 $\log D(G(z))$（等价于最小化 $-\log D(G(z))$），它在 $D(G(z))\approx0$ 处梯度很大，最优点不变，但早期学得快。

- GAN 训练交替：固定 $G$ 升 $D$，固定 $D$ 降 $\log(1-D(G(z)))$，或升 $\log D(G(z))$（非饱和启发式）。
- 判别器过强：生成器梯度消失。
- 判别器过弱：$G$ 收不到有用信号。
- 模式崩溃（mode collapse）：生成器只覆盖数据分布里的少数几种模式，输出几乎一样的图；判别器在这些点上仍可能被骗，多样性丢失。
- WGAN-GP 用 1-Lipschitz 约束替代 JS 散度，训练更稳，不是作业必实现。

一步 GAN 训练（非饱和损失）。判别器输出 logits，用 `BCEWithLogitsLoss` 数值更稳：

```python
bce = nn.BCEWithLogitsLoss()

def gan_step(G, D, opt_G, opt_D, x_real, d_z=100):
    m = x_real.size(0)
    ones, zeros = torch.ones(m, 1), torch.zeros(m, 1)

    # 1) 更新 D：最大化 log D(x) + log(1 - D(G(z)))
    z = torch.randn(m, d_z)
    x_fake = G(z)
    loss_D = bce(D(x_real), ones) + bce(D(x_fake.detach()), zeros)   # detach：不更新 G
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # 2) 更新 G：非饱和，最小化 -log D(G(z))，即把假图标成 1
    loss_G = bce(D(x_fake), ones)
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    return loss_D.item(), loss_G.item()
```

- `x_fake.detach()` 不能省：否则判别器的损失会把梯度传进生成器。
- 饱和版本是 `loss_G = -bce(D(x_fake), zeros)`，即最小化 $\log(1-D(G(z)))$；在 $D(G(z))\approx0$ 时梯度几乎为零。
- DCGAN 的经验设置：Adam，$\alpha=2\times10^{-4}$，$\beta_1=0.5$；判别器用 LeakyReLU，生成器用 ReLU，输出层用 tanh。

**训练技巧**（课上与 DCGAN、Salimans et al. 2016 的经验）：

- 输入图像缩放到 $[-1,1]$，与生成器的 tanh 输出一致；
- 生成器和判别器都用 BatchNorm，但真图和假图分别组成 batch，不要混在一起算统计；
- **单侧标签平滑**：真图标签用 $0.9$ 而不是 $1$，防止判别器过于自信、给生成器的梯度太小；
- 判别器更新 $k$ 次、生成器更新 1 次，或反过来，按两者损失的相对强弱调；
- 盯着生成样本看，而不只是看损失：GAN 的损失不随样本质量单调下降。

#### GAN 的有趣性质与应用

**隐空间算术**。DCGAN 论文发现，生成器的输入 $z$ 空间具有类似词嵌入的线性结构。取若干张“戴眼镜的男人”“不戴眼镜的男人”“不戴眼镜的女人”对应的 $z$，各自求平均后做

$$
z_{\text{戴眼镜的男人}}-z_{\text{男人}}+z_{\text{女人}}\approx z_{\text{戴眼镜的女人}},
$$

送入生成器，得到的确实是戴眼镜的女人。在两个 $z$ 之间线性插值，生成的图像也平滑过渡，说明生成器学到的是连续的语义结构，而不是记住训练图片。

**图像到图像的翻译**：

- **pix2pix**（Isola et al., 2017）：有成对数据时（草图与照片、白天与夜晚、分割图与街景），生成器是 U-Net，判别器看“输入 + 输出”这一对是否真实，损失为 GAN 损失加 $L_1$ 重构损失；
- **CycleGAN**（Zhu et al., 2017）：没有成对数据时（马和斑马的照片各有一堆，但没有同一场景的两种版本），用两个生成器 $G:X\to Y$、$F:Y\to X$，加**循环一致性损失**

$$
\mathcal L_{\mathrm{cyc}}=\mathbb E_x\lVert F(G(x))-x\rVert_1+\mathbb E_y\lVert G(F(y))-y\rVert_1,
$$

要求“马 → 斑马 → 马”回到原图，防止生成器随便输出一张与输入无关的斑马；

- **超分辨率**（SRGAN）：从低分辨率图生成高分辨率图，GAN 损失让纹理更真实，而不是 MSE 带来的模糊平均；
- **人脸生成与编辑**（StyleGAN）、医学影像的数据增强。深度伪造（deepfake）也来自这一类技术，检测它因此成了单独的研究方向。

#### 扩散模型

- 扩散前向：$q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)$。
- 闭式：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$，其中 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s\le t}\alpha_s$。任意 $t$ 的加噪样本可以一步得到，训练时随机选 $t$ 即可。
- 训练预测 $\varepsilon$ 或 $x_0$；采样从 $x_T\sim\mathcal{N}(0,I)$ 逐步去噪。常用损失为 $\mathbb{E}_{x_0,\varepsilon,t}\big\|\varepsilon-\varepsilon_\theta(x_t,t)\big\|_2^{2}$。
- 课堂要求掌握噪声预测的直觉，以及似然可算但采样步多、GAN 采样快但不稳的对比。

DDPM 的训练步只有几行：随机取 $t$，用闭式一步加噪，让网络回归噪声。

```python
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alpha_bar = torch.cumprod(1 - betas, dim=0)            # \bar{α}_t

def diffusion_loss(eps_model, x0):
    m = x0.size(0)
    t = torch.randint(0, T, (m,))
    eps = torch.randn_like(x0)
    ab = alpha_bar[t].view(m, *([1] * (x0.dim() - 1)))  # 广播到图像形状
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps        # 闭式加噪
    return F.mse_loss(eps_model(x_t, t), eps)

@torch.no_grad()
def ddpm_sample(eps_model, shape):
    x = torch.randn(shape)                              # x_T ~ N(0, I)
    for t in reversed(range(T)):                        # 串行 T 步：采样慢的根源
        tt = torch.full((shape[0],), t)
        a, ab = 1 - betas[t], alpha_bar[t]
        x = (x - betas[t] / (1 - ab).sqrt() * eps_model(x, tt)) / a.sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
    return x
```

#### 生成评估

> [!NOTE]+ 生成评估（generative evaluation）
>
> Fréchet Inception Distance（FID）把真实图像与生成图像映射到 Inception 特征，分别拟合高斯，再比较均值与协方差：
>
> $$
> \mathrm{FID}
> =\|\mu_r-\mu_g\|_2^{2}
> +\operatorname{tr}\left(
> \Sigma_r+\Sigma_g
> -2(\Sigma_r\Sigma_g)^{1/2}\right).
> $$
>
> FID 越低越好，同时反映样本质量与多样性，但依赖特征提取器、样本数和预处理协议。
>
> Inception Score（IS）看单张图的类别分布是否尖锐，以及全体样本的边缘类别分布是否多样：
>
> $$
> \mathrm{IS}
> =\exp\left(
> \mathbb{E}\_{x}
> D_{\mathrm{KL}}\big(p(y\mid x)\|p(y)\big)\right).
> $$
>
> IS 越高越好，但不直接与真实数据比较，也会受 Inception 类别空间限制。两者都不能代替人工偏好与任务指标。

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
> 对抗与生成仍假设标签或判别信号存在。强化学习去掉固定的 $(x,y)$：环境给奖励，网络近似价值 $Q$ 或策略 $\pi$。策略章里的偏差、方差和分布不匹配，在这里分别变成函数近似偏差、回报方差，以及仿真环境与真实环境不一致。

<br>

## 强化学习（Reinforcement Learning）

> [!INFO]+ DQN 与 PPO
>
> 强化学习没有固定的 $(x,y)$，数据由当前策略与环境交互产生。DQN（2015）用网络拟合 $Q$，经验回放降低样本相关，目标网络稳定 TD 目标。PPO 直接优化策略，并用裁剪限制一次更新的幅度。探索不足时学习曲线长期为零，不一定是实现写错。

<br>

### 基本框架

人学骑自行车不是靠老师逐个动作纠正，而是在不断尝试中自己摸索。让计算机在试错中自主学习，就是**强化学习**（reinforcement learning）。它与监督学习的区别在于：监督学习的每个输入都有“老师”给出的正确答案；强化学习只有事后的、往往延迟的奖励。

基本框架：**智能体**（agent）根据**环境**（environment）的状态选择**动作**（action），动作改变环境，智能体获得**奖励**（reward）。目标是学到一个行动方针（策略），使得到的奖励总和最大。

奖励往往不是即时明确的。在《超级马里奥兄弟》中，向右走一步能得多少奖励并不确定，要从游戏得分（硬币、消灭的敌人）或游戏结束这类明确指标反推出“预期奖励”。这也是强化学习比监督学习难的原因：一个动作的好坏可能要很多步之后才显现（信用分配问题，credit assignment）。

强化学习里没有固定的 $(x,y)$。智能体与环境交互，观察到状态 $s$、采取动作 $a$、得到奖励 $r$，目标是最大化期望回报 $G_t=\sum_{k=0}^{\infty}\gamma^{k} r_{t+k}$。深度网络用来近似策略 $\pi(a\mid s)$ 或价值 $Q(s,a)$。对应 2025 Autumn Lecture 5。

### 回报与价值

**折扣因子** $\gamma\in[0,1)$ 让越远的奖励越不重要：一方面未来不确定，另一方面保证无限长的和收敛。若每步奖励都是 1，$G=\sum_k\gamma^{k}=\frac{1}{1-\gamma}$，$\gamma=0.99$ 时约为 100，可以理解为智能体大约“看” 100 步远。

> [!EXAMPLE]+ 例3 折扣回报
>
> 取 $\gamma=0.9$。
>
> - 奖励序列 $(0,0,1)$：$G_0=0+0.9\cdot0+0.81\cdot1=0.81$。同样的奖励晚两步拿到，价值打了八一折。
> - 奖励序列 $(1,1,1,10)$：$G_0=1+0.9+0.81+0.729\cdot10=10$。
> - 一个立刻拿到 $+2$ 的选择，与一个 3 步后拿到 $+10$ 的选择相比，后者价值 $0.9^{3}\cdot 10=7.29>2$；若 $\gamma=0.5$，后者只值 $1.25<2$。$\gamma$ 决定智能体有多“远视”。

- MDP（Markov decision process）：$(S,A,P,R,\gamma)$。马尔可夫性指下一状态只依赖当前状态和动作，与更早的历史无关。
- 价值函数（value function）$V^{\pi}(s)=\mathbb{E}\_{\pi}[G_t\mid s_t=s]$：在状态 $s$ 按策略 $\pi$ 行动的期望回报。
- $Q^{\pi}(s,a)=\mathbb{E}\_{\pi}[G_t\mid s_t=s,a_t=a]$：在 $s$ 先做动作 $a$，之后按 $\pi$ 行动的期望回报。
- Bellman：回报可以拆成“这一步的奖励 + 折扣后的下一状态价值”，

$$
V^{\pi}(s)=\mathbb{E}\_{a\sim\pi,s'}[r+\gamma V^{\pi}(s')].
$$

- 最优 $Q^{\ast}$ 满足 $Q^{\ast}(s,a)=\mathbb{E}[r+\gamma\max_{a'}Q^{\ast}(s',a')]$。知道 $Q^{\ast}$ 后，最优策略就是每个状态选 $\arg\max_a Q^{\ast}(s,a)$。

### DQN

#### 从 Q 表到 Q 网络

**为什么围棋要用强化学习**。课上先问：能不能把围棋当监督学习做？输入是棋盘局面，标签是职业棋手下的下一步。这有三个问题：

1. 局面数是天文数字（$19\times19$ 棋盘约 $3^{361}$ 种配置），数据集只能覆盖极小一部分，难以泛化；
2. 标签来自人，模型最多学到和职业棋手一样好，不可能超越人类；
3. 只学到“下一步”，没有长期规划：一步看起来不是最好、但能提高最终胜率的棋，这种标签学不到。

真正关心的标签是**延迟的**：最后赢（1）还是输（0）。强化学习就是从这种延迟信号中学长期策略。

> [!EXAMPLE]+ 课上的玩具问题：“回收是好事”（Recycling is good）
>
> 一排 5 个格子。从状态 2 出发，每步只能向左或向右走一格。状态 1 是普通垃圾桶（到达得 $+2$，终止），状态 5 是回收桶（到达得 $+10$，终止），状态 4 有一块巧克力（到达得 $+1$）。最多走 3 步，防止一直吃巧克力。
>
> | 状态 | 1 | 2 | 3 | 4 | 5 |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | 含义 | 普通桶（终止） | 起点 | 空 | 巧克力 | 回收桶（终止） |
> | 到达的奖励 | $+2$ | $0$ | $0$ | $+1$ | $+10$ |
>
> 取 $\gamma=0.9$，用 Bellman 方程 $Q^{\ast}(s,a)=r+\gamma\max_{a'}Q^{\ast}(s',a')$ 从终点往回推。终止状态之后没有价值，$Q^{\ast}(1,\cdot)=Q^{\ast}(5,\cdot)=0$。
>
> - 状态 4：向右到 5，$Q=10+0=10$；向左到 3，$Q=0+0.9\max Q(3,\cdot)$，待定。
> - 状态 3：向右到 4，$Q=1+0.9\cdot10=10$；向左到 2，$Q=0+0.9\max Q(2,\cdot)$，待定。
> - 状态 2：向右到 3，$Q=0+0.9\cdot10=9$；向左到 1，$Q=2$。
> - 回代：$Q(3,\text{左})=0.9\cdot9=8.1$，$Q(4,\text{左})=0.9\cdot10=9$。
>
> | 状态 \ 动作 | 左 | 右 |
> | :--- | :--- | :--- |
> | 1 | 0 | 0 |
> | 2 | 2 | **9** |
> | 3 | 8.1 | **10** |
> | 4 | 9 | **10** |
> | 5 | 0 | 0 |
>
> 每行取最大值就是最优策略：一路向右，路过巧克力，最后进回收桶，回报 $0+0.9\cdot1+0.81\cdot10=9$，等于 $Q^{\ast}(2,\text{右})$。若 $\gamma$ 很小（例如 $0.1$），$Q(2,\text{右})=0.1\cdot(1+0.1\cdot10)=0.2<2$，最优策略变成直接扔进普通桶：**$\gamma$ 是问题定义的一部分，改变它会改变最优策略**。

```python
def q_value_iteration(n_states=5, rewards={1: 2, 4: 1, 5: 10}, terminal={1, 5}, gamma=0.9, iters=50):
    Q = np.zeros((n_states + 1, 2))                  # 下标 1..5；动作 0=左，1=右
    for _ in range(iters):
        for s in range(1, n_states + 1):
            if s in terminal:
                continue
            for a, s2 in enumerate([s - 1, s + 1]):
                future = 0.0 if s2 in terminal else Q[s2].max()
                Q[s, a] = rewards.get(s2, 0) + gamma * future
    return Q[1:]

print(q_value_iteration())                # [[0,0],[2,9],[8.1,10],[9,10],[0,0]]
print(q_value_iteration(gamma=0.1)[1])    # [2, 0.2]：γ 小时向左更好
```

状态和动作都很少时，可以用一张表存所有 $Q(s,a)$，用 Bellman 方程反复更新（Q 学习）。但游戏画面的状态数是天文数字，表存不下，于是用神经网络近似：输入状态 $s$，输出每个动作的 $Q$ 值。

- Deep Q-Network：网络 $Q(s,a;\theta)$ 回归 TD 目标。
- 时序差分目标（temporal-difference target）与 TD error 为

$$
y_t=r_t+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a';\theta^{-}),
\qquad
\delta_t=y_t-Q(s_t,a_t;\theta).
$$

$d_t=1$ 表示终止，后续价值必须置零。DQN 最小化 $\delta_t^{2}$ 或 Huber loss。目标网络参数 $\theta^{-}$ 暂时固定，避免预测值与目标同时快速移动。这相当于一个回归问题，只是“标签” $y_t$ 由网络自己估计出来。

- 目标网络（target network） $\theta^{-}$ 缓慢更新。TD 目标里用下一状态的 $Q$ 估计当前 $Q$，称为 bootstrapping；两边若用同一套正在更新的权重，目标会跟着漂。
- 经验回放（experience replay）：把轨迹存进缓冲区再随机抽样，打散相邻状态之间的相关，也提高样本利用率。
- 动作离散时可行；连续动作常用策略梯度（policy gradient）。

> [!EXAMPLE]+ 例4 DQN 玩 Atari（鱼书 8.5.4 节）
>
> 以往让计算机学玩游戏，要先把游戏状态（角色位置等）人工提取出来交给模型。DQN 的输入**只有游戏画面**：把连续 4 帧画面（灰度化、缩小到 $84\times84$，叠成 $84\times84\times4$，4 帧可以体现运动方向）送入 CNN，输出游戏手柄每个动作（摇杆移动、按钮按下与否）的 $Q$ 值。
>
> 因此不必针对每个游戏改设置：同一个网络结构学会了《吃豆人》等许多 Atari 游戏，并在很多游戏中超过了人类水平。

课上以 Breakout（打砖块）为例，把 DQN 落地要定的几件事逐个列出：

- **奖励**：打碎砖块为正，球漏掉为负；
- **动作**：不动、左移、右移；
- **状态与预处理**：原始画面 $210\times160\times3$。转灰度（颜色对决策无用）、裁掉记分栏等无关区域、缩小到 $84\times84$，再把最近 4 帧叠起来——单独一帧看不出球往哪边飞。预处理的原则与项目策略一致：去掉与任务无关的信息，保留决策需要的信息；
- **终止状态**：用一个布尔量 `terminal` 标记游戏结束。终止时 TD 目标只有 $r$，没有 $\gamma\max Q(s',\cdot)$，否则网络会给“游戏已结束”的状态估出虚假的未来价值；
- **网络**：几层卷积加全连接，输出每个动作的 $Q$，一次前向就能比较所有动作；
- **训练技巧**：经验回放、$\varepsilon$-greedy（$\varepsilon$ 从 $1$ 逐渐衰减到约 $0.1$，先多探索、后多利用）、目标网络。

DQN 算法要点：

1. 用 $\varepsilon$-greedy 从 $Q(s,\cdot;\theta)$ 选动作（以概率 $\varepsilon$ 随机探索，否则选 $Q$ 最大的动作），与环境交互，把 $(s,a,r,s',\mathrm{done})$ 写入 replay buffer。
2. 均匀采样一批转移。
3. 目标 $y=r+\gamma\max_{a'}Q(s',a';\theta^{-})$（终止状态无后续项）。
4. 对 $\theta$ 做 MSE 或 Huber 回归。
5. 每隔 $C$ 步 $\theta^{-}\leftarrow\theta$。

- 没有 replay：样本强相关。
- 没有目标网络：$y$ 与 $Q$ 同时移动。
- 二者都是发散的常见原因。
- 连续动作不能对 $a$ 做 $\max$，改用 DDPG/PPO 等策略方法。

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

PyTorch 版 DQN 的核心：回放缓冲区、$\varepsilon$-greedy、一步 TD 更新和目标网络同步。

```python
import copy, random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)           # 满了自动丢最旧的转移
    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))
    def sample(self, batch_size):
        s, a, r, s2, d = zip(*random.sample(self.buf, batch_size))
        f = lambda v, dt=torch.float32: torch.as_tensor(np.array(v), dtype=dt)
        return f(s), f(a, torch.long), f(r), f(s2), f(d)

def select_action(q_net, s, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)          # 探索
    with torch.no_grad():
        return q_net(torch.as_tensor(s, dtype=torch.float32)[None]).argmax(1).item()

def dqn_update(q_net, target_net, opt, buffer, batch_size=32, gamma=0.99):
    s, a, r, s2, done = buffer.sample(batch_size)
    q_sa = q_net(s).gather(1, a[:, None]).squeeze(1)          # 只取实际执行动作的 Q
    with torch.no_grad():                                      # 目标不回传梯度
        q_next = target_net(s2).max(1).values
        y = r + gamma * (1 - done) * q_next
    loss = F.smooth_l1_loss(q_sa, y)                           # Huber loss
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    opt.step()
    return loss.item()

q_net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))   # CartPole：4 维状态，2 个动作
target_net = copy.deepcopy(q_net).requires_grad_(False)
# 训练循环中每 C 步：target_net.load_state_dict(q_net.state_dict())
```

- `gather` 只取出执行过的动作 $a_t$ 那一列：其他动作没有观测到奖励，不能产生损失。
- Double DQN 只改一行：`a2 = q_net(s2).argmax(1, keepdim=True)`，`q_next = target_net(s2).gather(1, a2).squeeze(1)`。在线网络选动作、目标网络评估，缓解 $\max$ 带来的高估。
- 软更新是硬拷贝的替代：每步 $\theta^{-}\leftarrow\tau\theta+(1-\tau)\theta^{-}$，$\tau$ 约 $0.005$。

### 策略梯度

DQN 先学价值再导出策略；策略梯度直接参数化策略 $\pi_\theta(a\mid s)$（例如网络输出动作上的 softmax），沿提高期望回报的方向更新 $\theta$。

REINFORCE：

$$
\nabla_{\theta}J=\mathbb{E}\_{\pi_{\theta}}\big[\nabla_{\theta}\log\pi_{\theta}(a\mid s)\,G_t\big].
$$

直观含义：回报高的轨迹，提高其中各动作的概率；回报低的，降低其概率。形式上就像把 $G_t$ 当作样本权重的交叉熵。

- 轨迹概率 $P(\tau)=\rho(s_0)\prod_t \pi(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$。
- $\nabla\log P(\tau)=\sum_t\nabla\log\pi(a_t\mid s_t)$；转移 $P$ 不出现在梯度里。
- 因此可以在未知模型上学习（model-free）。
- 基线 $b(s)$（如 $V(s)$）不改变无偏性，但降低方差，形成 actor-critic。
- 优势 $\hat{A}_t=G_t-V(s_t)$ 或 GAE 进一步降方差：衡量这个动作比该状态下的平均表现好多少。
- PPO 用裁剪的概率比限制一次更新幅度：

$$
L^{\mathrm{CLIP}}=\mathbb{E}\big[\min\big(\rho\hat{A},\,\mathrm{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat{A}\big)\big],
$$

$$
\rho=\frac{\pi_{\theta}(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}.
$$

- PPO 的 clip 把 $\rho$ 限制在 $[1-\varepsilon,1+\varepsilon]$，防止一次更新毁掉策略。
- 另一种限制是在目标里加 $\beta D_{\mathrm{KL}}(\pi_{\theta}\|\pi_{\mathrm{ref}})$，使新策略靠近参考策略（reference policy）。RLHF 与 KL 正则强化学习（KL-regularized RL）用的是这一项。
- $\varepsilon$ 常取 $0.1$-$0.2$。

REINFORCE 与 PPO 裁剪目标的 PyTorch 写法。`Categorical.log_prob` 给出 $\log\pi_\theta(a\mid s)$，梯度由 autograd 完成：

```python
from torch.distributions import Categorical

def run_episode(policy, env):
    """policy(s) 输出动作 logits；env 为 gymnasium 风格接口。"""
    s, _ = env.reset()
    log_probs, rewards, done = [], [], False
    while not done:
        dist = Categorical(logits=policy(torch.as_tensor(s, dtype=torch.float32)))
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        s, r, terminated, truncated, _ = env.step(a.item())
        rewards.append(r)
        done = terminated or truncated
    return torch.stack(log_probs), rewards

def reinforce_step(policy, opt, env, gamma=0.99):
    log_probs, rewards = run_episode(policy, env)
    G = torch.as_tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
    G = (G - G.mean()) / (G.std() + 1e-8)          # 标准化：相当于减去基线
    loss = -(log_probs * G).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    return sum(rewards)

def ppo_clip_loss(new_logp, old_logp, adv, clip_eps=0.2):
    ratio = torch.exp(new_logp - old_logp.detach())             # ρ，用 log 差保证数值稳定
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    return -torch.min(unclipped, clipped).mean()                 # 最大化 L^CLIP
```

`G` 只是权重，不能对它求导；`discounted_returns` 在 NumPy 里算出来，天然不带梯度。若用 critic 的 $V(s)$ 作基线，优势要写成 `adv = (G - V).detach()`，critic 另用 `F.mse_loss(V, G)` 训练。

### 探索与奖励设计

- 探索：$\varepsilon$-greedy、熵奖励、噪声网络。
- 奖励稀疏（只有终局 $+1$）时，先塑造稠密奖励或用课程学习。
- 曲线长期为零：不能只归因于算法写错，也可能是没探索到。
- 仿真到真实：像素 vs 状态向量、延迟、摩擦都会造成不匹配。
- 先在状态可完整观测的玩具环境验证实现，再上像素输入。
- 对应策略章：先保证能拟合训练。

**探索为什么难：Montezuma's Revenge**。DQN 在大多数 Atari 游戏上超过人类，却在《蒙特祖玛的复仇》上几乎得零分。这个游戏要先下梯子、跳过骷髅、拿到钥匙，才第一次得分；靠 $\varepsilon$-greedy 随机按键，几乎不可能碰巧完成这一长串动作，于是奖励永远是 0，网络什么也学不到。人一眼就知道“钥匙有用、骷髅危险、梯子能爬”，这些**先验知识**正是智能体缺的。课上提到的解决方向：

- **内在动机**（intrinsic motivation）：除了游戏奖励，再给“到达没见过的状态”一个额外奖励。基于计数的探索（Bellemare et al., 2016）给 $r^{+}\propto 1/\sqrt{N(s)}$，$N(s)$ 是该状态被访问的（伪）次数，越少见奖励越大；
- **模仿学习**（imitation learning）：先看专家怎么玩，从演示中学，再用强化学习继续提高。

**模仿学习**。最简单的是行为克隆（behavior cloning）：把专家的 $(s,a)$ 对当监督数据，训练 $\pi(a\mid s)$。问题是**误差累积**：策略一旦走偏一点，就进入专家从没去过的状态，那里没有训练数据，于是越偏越远。DAgger 让专家在策略自己走到的状态上补标注；GAIL（Ho & Ermon, 2016）用 GAN 的思路，判别器区分“专家轨迹”与“策略轨迹”，策略学着让判别器分不出来。RLHF 的第一步 SFT 本质上就是对人类示范做行为克隆。

**DQN 与 PPO 对照**（Lecture 5 的总结表）：

| | DQN | PPO |
| :--- | :--- | :--- |
| 学什么 | 价值 $Q(s,a)$，策略是 $\arg\max_a Q$ | 直接学策略 $\pi_\theta(a\mid s)$ |
| 动作空间 | 离散 | 离散与连续都行 |
| 目标 | 回归 Bellman 目标 | 裁剪的代理目标，限制每次更新幅度 |
| 数据 | 离策略（off-policy），经验回放反复用旧数据 | 在策略（on-policy），每轮用当前策略新采的数据 |
| 探索 | $\varepsilon$-greedy | 随机策略本身带探索，可加熵奖励 |
| 稳定性 | 依赖目标网络与回放 | 通常更稳，靠裁剪与优势函数 |
| 典型应用 | Atari | 机器人控制、RLHF |

课上最后提到的进阶方向：**树搜索**（AlphaGo 的蒙特卡洛树搜索，在决策时向前模拟多步）、**策略梯度**、**元学习**（meta-learning：任务分布相似时，从一族任务中学会“如何快速学新任务”，不必每次从零训练）。

> [!WARNING]+ 奖励设计
>
> - 智能体会利用奖励漏洞（reward hacking）。
> - 稀疏奖励使探索困难。
> - 函数近似引入偏差；蒙特卡洛回报方差大。
> - 仿真到真实是分布不匹配。
> - 项目若做 RL：先在可重置、有明确成功条件的环境验证学习曲线，再谈迁移。

<br>

> [!EXAMPLE]+ 例5 AlphaGo
>
> AlphaGo 击败围棋世界冠军，内部同时用到深度学习和强化学习：先学习约 3000 万个职业棋手的棋谱局面（监督学习，得到初始策略），再不停地自我对弈积累经验（强化学习），并用价值网络评估局面、配合蒙特卡洛树搜索选点。AlphaGo 与 DQN 都出自 Google DeepMind。

### RLHF

> [!NOTE]+ RLHF 流程（reinforcement learning from human feedback）
>
> 2025 Autumn Lecture 5 把语言模型训练分成以下阶段。
>
> 1. 下一词元预测（next-token prediction）：预训练语言模型（pretrained language model）最大化语料的对数似然。
> 2. 监督微调（supervised fine-tuning, SFT）：使用人工示范（human-written demonstrations）学习有帮助的回答格式。
> 3. 偏好数据（preference data）：对同一提示的多个回答做人类排序（human ranking）。
> 4. 奖励模型（reward model, RM）：学习标量分数 $r_{\phi}(x,y)$，使偏好回答得分更高。常见成对偏好损失（pairwise preference loss）为
>
> $$
> \mathcal{L}\_{\mathrm{RM}}
> =-\log\sigma\big(r_{\phi}(x,y_{\mathrm{chosen}})
> -r_{\phi}(x,y_{\mathrm{rejected}})\big).
> $$
>
> 5. 强化学习微调（reinforcement-learning fine-tuning）：策略生成回答，奖励模型打分；PPO 或相关策略方法提高奖励，同时用 KL penalty 限制策略离开参考模型。
>
> $$
> \max_{\theta}
> \mathbb{E}\_{y\sim\pi_{\theta}(\cdot\mid x)}
> \left[r_{\phi}(x,y)
> -\beta\log\frac{\pi_{\theta}(y\mid x)}
> {\pi_{\mathrm{ref}}(y\mid x)}\right].
> $$
>
> PPO 的旧策略（old policy）$\pi_{\theta_{\mathrm{old}}}$ 用于构造裁剪概率比；RLHF 的参考策略（reference policy）$\pi_{\mathrm{ref}}$ 用于 KL 约束。二者角色不同，不能混成同一个分布。
>
> 奖励黑客（reward hacking）指策略利用奖励模型漏洞得到高分，却没有真正提高人类偏好。KL 太弱会漂移，太强则几乎不离开 SFT 模型。

在 RLHF 中，“状态”是提示加已生成的词，“动作”是下一个词，“奖励”只在整段回答结束时由奖励模型给出：这正是上文说的稀疏、延迟奖励。

**为什么用成对比较，而不是直接打分**。让标注者给一个回答打 1 到 10 分，不同人的尺度差别很大、同一人前后也不一致；判断“A 和 B 哪个更好”要稳定得多。奖励模型的损失来自 Bradley–Terry 模型：假设人类偏好 A 胜过 B 的概率为

$$
P(y_A\succ y_B\mid x)=\sigma\big(r(x,y_A)-r(x,y_B)\big),
$$

对偏好数据做最大似然，就是上面的 $-\log\sigma(r_{\mathrm{chosen}}-r_{\mathrm{rejected}})$。它就是一个二分类交叉熵，logit 是两个回答的分数差，所以奖励只有差值有意义，整体加一个常数不影响。

```python
def reward_model_loss(r_chosen, r_rejected):
    """r_*: (B,) 奖励模型对同一提示的两个回答打的分。"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

**DPO**（direct preference optimization，Rafailov et al., 2023）是课后常见的延伸：对上面带 KL 约束的目标求最优策略，可以解出 $r(x,y)=\beta\log\frac{\pi^{\ast}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\text{常数}$。把它代回 Bradley–Terry 损失，奖励模型和强化学习两步合并成一个直接对策略的分类损失：

$$
\mathcal L_{\mathrm{DPO}}=-\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\Big),
$$

$y_w$、$y_l$ 是偏好和不偏好的回答。不需要单独训练奖励模型，也不需要在训练中采样，实现上只是监督学习。代价是只能利用已有的偏好数据，不能像 PPO 那样在新生成的回答上继续探索。

> [!NOTE]+ 对照：监督、对抗、RL
>
> - 监督：$\nabla_\theta J(\theta,x,y)$，标签固定。
> - 对抗：$\nabla_x J$，标签仍在，更新的是输入。
> - RL：$\nabla_\theta\log\pi\cdot G$，标签换成回报，数据由策略自己产生。
> - 策略自己产生数据，因此探索失败与分布漂移比监督学习更早出现。

<br>

> [!NOTE]+ 衔接
>
> 强化学习的数据由当前策略与环境交互产生。LLM 应用层相反：骨干已经预训练，调整的是提示、检索与工具，也就是条件 $x$ 怎么拼。评估仍按任务拆开，不能只报一个笼统的主观质量分数。

<br>

## 读论文与职业（Reading Papers and Career Advice）

CS230 每学期都有一讲由 Ng 讲怎样读论文、怎样规划职业（2018 Autumn Lecture 8；2025 Autumn 排在 Lecture 6 / 9，课前可选读 Dropout 与 DenseNet 两篇论文）。课上会现场给几分钟，让学生按下面的方法读 DenseNet。

### 怎样读一个领域

想快速进入一个新方向（例如语音识别）时：

1. **列清单**：收集论文，也包括博客、Medium 文章、GitHub 仓库；
2. **跳着读**：同时打开多篇，每篇先只读约 $10\%$–$20\%$，得到大致印象。发现某篇没价值就放弃；发现某篇是奠基性的，就投入时间完整读懂；
3. **顺着引用扩展**：从读懂的论文的参考文献里找新论文加进清单，回头再补读之前略过的论文。

读多少的粗略参考：

| 读过的论文数 | 大致水平 |
| :--- | :--- |
| $5$–$20$ 篇 | 对领域有基本了解，能实现、应用一些算法，但还不足以做前沿研究 |
| $50$–$100$ 篇 | 对领域有很好的理解，足以开始做前沿研究 |

### 怎样读一篇

不要从第一个字读到最后一个字，而是**分几遍读**，先读信息密度高的部分，难的部分留到后面：

1. 标题、摘要、图（尤其是图），必要时加一两个关键实验；
2. 引言、结论、图，其余部分略读；相关工作（related work）可以跳过或略读；
3. 读正文，但跳过或略读数学；
4. 通读全文，跳过实在读不懂的部分。读不懂的部分可能是作者没写清楚，也可能是不那么重要的细节。

读完后用四个问题检查自己：

- 作者想要解决什么问题？
- 方法的关键要素是什么？
- 哪些内容自己可以用上？
- 还有哪些参考文献值得继续读？

要真正理解一个算法：

- **数学**：合上论文，从头推一遍；
- **代码**：先跑通开源代码，再自己从头实现一遍。

### 长期习惯与知识结构

- **稳定地读**，不要突击：比如每周两篇，长期坚持比短期突击有效得多，积累有复利效应。Ng 说他自己随时带着一个“未读论文”文件夹，组里每周讨论两篇，为了选出这两篇，他每周要读五六篇；
- 熟练以后读一篇论文会越来越快（不同作者画表格、画网络结构的方式大同小异）。Ng 读懂一篇大约要半小时，简单的 20 分钟；新手花一小时很正常；
- 跟进顶会论文、开源实现与社区讨论。

职业上，Ng 推荐 **T 型**（T-shaped）的知识结构：

- **横**：对 AI 的许多方向都有基础了解（机器学习、深度学习、概率统计、软件工程……），靠课程和读书会建立；
- **竖**：至少在一个方向上有很深的积累，靠相关项目、实习和研究建立。

只有广度、没有深度，或者只在一个方向很深、对其他方向毫无了解，都不如 T 型。招聘方看重的是有深度的作品，不是项目数量。

<br>

## LLM 应用（LLM Applications）

### 应用层

骨干已经是预训练好的 $P(y\mid x)$ 时，产品效果往往取决于提示、检索、工具和评估，而不是再训一个网络。对应 2025 Autumn Lecture 8。

- 预训练模型是条件分布 $P(y\mid x)$。
- 应用层控制 $x$ 的构造：$x=$ 系统指令 + 检索片段 + 对话历史 + 用户问题。
- 上下文窗口（context window）有限，必须做选择与压缩。

可调的手段按代价从低到高排列：

1. **零样本提示**（zero-shot）：只写指令；
2. **少样本提示**（few-shot）：在上下文中放几个示范；
3. **检索增强**（RAG）：把相关文档放进上下文；
4. **工具调用**：让模型调用搜索、计算器、数据库；
5. **微调**（LoRA 等）：更新权重。

选择哪一种，应由误差分析决定，而不是默认从最贵的开始。

### 提示

- 少样本提示：把 $k$ 个输入-输出示范放进上下文，不更新权重。
- 示范必须与测试分布同格式，否则学到的是表面模板。
- 温度、top-$p$ 属于解码超参数，应在 dev 上选，并在报告中固定。

课上把提示工程称为“第一道优化”，因为它最便宜、迭代最快。从基础到进阶依次是：

**1. 写清楚**。“总结这篇文档”对模型来说缺了三样信息：文档是什么类型、要多长、给谁看。改成“用 5 条要点总结这篇 10 页的可再生能源论文，面向政策制定者，突出关键发现及其政策含义”，同时规定了类型、受众、格式和重点。

**2. 零样本与少样本**。零样本只给指令；少样本先给几个“输入 → 输出”示范，再给待处理的输入。课上的例子：判断 “The product is fine, but I was expecting more.” 的情感，零样本输出 Neutral；给出“It's okay, but I wish it had more features. → Negative”等示范后，输出变成 Negative。示范定义了标注标准，模型按示范的标准判断，而不是按它自己的默认理解。

**3. 思维链**（chain-of-thought, CoT；Wei et al., 2022）。要求模型“一步步思考，不要跳步”，或者直接把步骤写出来：

> 第 1 步：找出论文最重要的三个发现；第 2 步：说明每个发现对政策的影响；第 3 步：写出 5 条要点，每条对应一个发现及其政策含义。

模型每生成一个词，计算量是固定的。让它先写出中间推理，相当于给难题分配了更多计算，算术、逻辑和多步推理题的准确率明显提高。

**4. 提示链**（prompt chaining）。与思维链不同，提示链是**多次调用**：把复杂任务拆成几个提示，前一个的输出作为后一个的输入。课上的例子是回复一封客户投诉：

1. 提示 1：从评论中提取关键问题（晚到三天、包装破损、急用）；
2. 提示 2：根据这些问题写一份回复提纲（致歉、解释原因、给出补偿）；
3. 提示 3：根据提纲写完整回复。

比起一个大提示一步写完，提示链的每一步都可以**单独检查、单独改进**：回复不好时，可以先看是问题提取错了还是行文差，这与误差分析中“把流水线拆成组件、逐个看误差来源”的思路相同。

**5. 提示模板**。带占位符的固定结构，例如“为 [受众] 用 [格式] 总结以下文本：[文本]”。模板让同一类请求的提示一致，便于批量测试和版本管理。

**6. 测试提示**。像测试代码一样测试提示：准备一组测试用例，对每个提示版本打分（PromptFoo 等工具），用数字而不是印象决定换不换。打分可以是规则（是否包含关键字段），也可以用另一个 LLM 当评审（见“评估”一节）。

课上还引用了 BCG 的一项研究：咨询顾问在 AI 能力范围内的任务上使用 AI，效率和质量都显著提高；但在 AI 能力范围之外的任务上，使用 AI 的人反而比不用的人正确率低 19 个百分点。使用方式分两类：**半人马**（centaur）把任务切开，一部分交给 AI、一部分自己做；**赛博格**（cyborg）把 AI 融入每一步，持续交互。关键是知道任务落在 AI 能力边界的哪一侧。

### RAG

> [!INFO]+ RAG 与 LoRA
>
> RAG（Lewis et al., 2020）先检索 $k$ 段文档，再在这些证据上生成，不更新骨干权重。检索库可以更新事实；检索失败时生成仍可能流畅地编造，因此召回率要单独测。LoRA（2021）在线性层上加低秩 $\Delta W=BA$，适合改风格和格式；事实过时时应改检索库。

<br>

模型的知识截止于训练数据，私有文档也不在其中，直接问容易编造（幻觉）。RAG 把“记住事实”的负担交给可更新的检索库。

- RAG（retrieval-augmented generation）：检索器返回 $k$ 段文档，生成器只在这些证据上回答。
- 检索失败时生成器仍会流畅地编造。
- 必须单独测检索召回，不能只看最终 BLEU 或人工分。

RAG 流水线可以写成与模型无关的步骤：

1. 文档切块：按标题或固定 token 数，块过大则噪声多，过小则上下文碎。
2. 嵌入并建索引；查询时对问题向量取 $k$ 近邻（与词嵌入一节的余弦相似度相同）。
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

### 工具与安全

- 工具调用把输出约束为可解析动作（搜索、计算器、数据库）：模型输出 JSON 或特定标记，程序执行后再把结果追加进上下文。
- 失败模式：字段缺失、类型错误、重复调用、对不可用工具幻想。
- 应用层用 schema 校验，而不是再提示“请一定输出合法 JSON”。
- 提示注入（prompt injection）：不可信文档里的“忽略以上指令”应被视为数据。
- 系统消息与工具权限隔离。
- 解析失败、权限过宽与注入攻击（指令藏在检索文档里）是系统风险，不能靠再调一点解码温度解决。

### 智能体

课上的定义：**AI 智能体**（AI agent）是一个自主的 AI 系统，利用**上下文、记忆和工具**进行推理并采取行动，完成复杂的多步任务。

RAG 与智能体的区别，课上用退款的例子说明：

- 用户问“退款政策是什么？”——RAG 检索政策文档，回答“购买 30 天内可退款”，一问一答就结束；
- 用户问“这笔订单能退款吗？”——智能体先用 RAG 查政策，再**反问**订单号，然后**调用订单 API** 查购买日期，判断符合条件后确认“3 到 5 个工作日到账”。这需要多步规划、与用户交互和调用外部系统。

**组成部分**（以课上的旅行预订智能体为例）：

- **提示**：角色与规则（“角色：旅行代理”）；
- **工具**：航班搜索、酒店预订、天气、支付等 API。每个工具都要写清楚“做什么、输入参数、返回什么、什么时候用”，模型靠这段描述决定调哪个工具；
- **记忆**：分为当前对话内的上下文，以及跨会话的长期记忆（“用户偏好直飞”“喜欢地标附近的四星酒店”），下次规划时直接用上；
- **上下文管理**：上下文窗口有限，要决定保留哪些历史、哪些检索结果。

一次任务的执行流程：**规划**（把“订 12 月 15–20 日去巴黎的机票酒店并安排行程”拆成查航班、查酒店、生成景点、与用户确认、付款五步）→ **执行**（按步骤调用工具、汇总结果）→ **与用户确认**→ **完成动作**（支付）→ **写入记忆**。

**智能体软件与传统软件的区别**：

| | 传统软件 | 智能体软件 |
| :--- | :--- | :--- |
| 输入输出 | 结构化、强类型 | 自由文本，需要动态理解 |
| 逻辑 | 确定性规则，可重复 | 概率推理，结果不完全可预测 |
| 开发方式 | 写函数、写流程 | 设计提示、组合链与工具 |
| 维护 | 改一处很少影响无关功能 | 改一个提示可能悄悄弄坏其他流程 |
| 测试 | 给定输入有确定输出 | 需要反复、探索式地测试 |

最后一行是最大的工程难点，这也是智能体系统必须有一套评估集、每次改动都回归测试的原因。

**多智能体系统**。把一个大智能体拆成多个专门的智能体，再由一个**编排者**（orchestrator）协调。课上的例子是智能家居：温控、照明、安防、能源管理、娱乐、通知六个智能体，各管一摊。组织方式有扁平、层级、顺序、并行几种。与单个智能体相比：

- **模块化**：可以单独更新或替换某个智能体；
- **故障隔离**：一个出错不会拖垮整体；
- **专门化**：每个智能体的提示和工具只针对自己的领域，效果更好；
- **调试**：问题更容易定位到某个智能体；
- **并行**：互不依赖的子任务可以同时做。

代价是协调开销和更多的调用成本。智能体之间如何分工，课上的说法是“像经理一样思考”（think like a manager）：先想清楚岗位怎么划分，再决定每个岗位需要什么工具。

一个最小的工具调用循环（与具体模型无关，`llm` 是任意返回文本的函数）：

```python
import json

TOOLS = {
    "get_order": lambda order_id: {"order_id": order_id, "days_since_purchase": 12},
    "refund_policy": lambda: {"window_days": 30},
}

def run_agent(llm, user_msg, max_steps=5):
    history = [{"role": "system", "content":
                "可用工具: get_order(order_id), refund_policy()。"
                "需要调用工具时只输出 JSON: {\"tool\": 名称, \"args\": {...}}；否则直接回答。"},
               {"role": "user", "content": user_msg}]
    for _ in range(max_steps):                          # 步数上限：防止无限循环
        reply = llm(history)
        try:
            call = json.loads(reply)
        except json.JSONDecodeError:
            return reply                                # 不是工具调用，就是最终回答
        if not isinstance(call, dict) or "tool" not in call:
            return reply
        if call["tool"] not in TOOLS:                   # 校验：不执行未知工具
            history.append({"role": "user", "content": f"未知工具 {call.get('tool')}"})
            continue
        result = TOOLS[call["tool"]](**call.get("args", {}))
        history += [{"role": "assistant", "content": reply},
                    {"role": "user", "content": f"工具结果: {json.dumps(result, ensure_ascii=False)}"}]
    return "超过最大步数，转人工处理。"
```

`history` 就是这次任务的 **trace**：每一步模型说了什么、调了什么工具、拿到什么结果。智能体出错时，调试靠的就是逐步读 trace。

### LoRA

课上对微调的态度是“谨慎使用”（proceed with caution），理由有三：

- 需要大量带标签的任务数据；
- 容易过拟合到这批数据，丢掉通用能力（例如有人用 14 万条 Slack 消息微调 GPT-3.5，结果模型学会的是同事的说话方式，回答“好的，明天上午弄”，而不是去完成任务）；
- 费时费钱，而且基座模型更新很快，每出一个新版本就得重新微调一次。

值得微调的情形：任务需要反复输出高精度、固定风格的结果（法律文书、科学解释），或者通用模型在某个领域的专有语言上一直表现不好。其他情况先把提示、提示链和 RAG 用到头。

- LoRA（low-rank adaptation）：线性层上加低秩 $\Delta W=BA$，$B\in\mathbb{R}^{d\times r}$，$A\in\mathbb{R}^{r\times k}$，$r\ll \min(d,k)$。原权重 $W$ 冻结，前向变为 $h=(W+BA)x$。
- 可训参数从 $dk$ 降到 $r(d+k)$，远小于全量微调，适合项目算力。例如 $d=k=4096$、$r=8$ 时，从约 1678 万降到约 6.6 万，约为 $0.4\%$。

```python
class LoRALinear(nn.Module):
    """包住一个已训练好的 nn.Linear：h = W x + (α/r) B A x。"""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)                     # 冻结 W 与 b
        self.A = nn.Parameter(torch.randn(r, base.in_features) / r)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # B=0：初始时 ΔW=0，不改变原模型
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self):
        """部署时把 ΔW 合并进 W，推理零额外开销。"""
        self.base.weight += self.scale * self.B @ self.A
        return self.base

layer = LoRALinear(nn.Linear(4096, 4096), r=8)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print(trainable, trainable / 4096 ** 2)     # 65536，约 0.0039
```

- $B$ 初始化为零、$A$ 随机：训练开始时输出与原模型完全一致，又不会让两者都为零导致梯度为零。
- 实践中常用 Hugging Face `peft` 库：`LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])` 指定只在注意力的 $W^Q,W^V$ 上加 LoRA。
- 仅当误差表显示风格或格式占主导、且有足够私有数据时才值得做。
- 误差来自事实过时：更新检索库比 LoRA 便宜。

### 评估

- 评估应分任务：闭卷事实、开卷引用、数学、代码、拒答。
- 只报一个笼统的主观质量分数，无法判断下一步该改检索、提示还是工具。
- 评估协议先写再测。
- 闭卷 QA：exact match / F1。
- 开卷：检查引用是否支撑答案。
- 代码：单测通过率。
- 安全：红队集。
- 不要用同一聊天记录既调提示又报最终分（这等于在测试集上调参）。
- 项目若做 LLM 应用，基线应包括：无检索零样本（zero-shot）、少样本（few-shot）、简单 RAG。
- 误差分析标注：检索错 / 证据够但推理错 / 格式错 / 拒答不当。这与项目策略中“对出错样本分类计数”的误差分析方法相同。

课上把智能体系统的评估（evals）分成两类：

- **客观指标**：任务成功率（例如“地址修改”这类任务是否真的改成功）、延迟、每次调用的成本、工具调用的出错率。这些可以自动计算；
- **主观质量**：语气是否礼貌、回答是否切题、用户是否满意。这些要人工评，或者用 **LLM 评审**（LLM-as-a-judge）：给另一个模型一份评分细则和待评回答，让它打分或在两个回答中选一个更好的。

LLM 评审便宜、可以大规模跑，但有已知偏差：偏好排在前面的回答（位置偏差）、偏好更长的回答（长度偏差）、偏好与自己风格相近的回答。常用对策是交换两个回答的顺序各评一次、在评分细则中明确长度不加分，并定期抽一部分与人工评分对比，确认二者一致。

误差分析时，还要区分**客观错误**（查错了订单号、调错了工具）和**主观偏好**（用户想要直飞而系统推荐了转机）：前者要修流程或工具，后者要补记忆或改提示里的默认偏好。

```python
JUDGE_PROMPT = """角色：评审。根据评分细则给回答打 1-5 分，只输出 JSON {{"score": 整数, "reason": 字符串}}。
细则：1) 是否依据提供的证据 2) 是否回答了问题 3) 语气是否专业。回答长度本身不加分。
问题：{question}
回答：{answer}"""

def judge_pairwise(llm, question, a, b):
    """交换顺序各评一次，抵消位置偏差；两次结论不一致时记为平局。"""
    ask = lambda x, y: llm(f"问题：{question}\n回答 1：{x}\n回答 2：{y}\n哪个更好？只输出 1 或 2。").strip()
    first, second = ask(a, b), ask(b, a)
    if first == "1" and second == "2":
        return "a"
    if first == "2" and second == "1":
        return "b"
    return "tie"
```

这一讲最后，Kian 给了几条对未来的个人看法：

- **规模定律会不会到顶**：更多数据未必一直带来更好的模型，但投入 AI 的研究者数量远多于当年推动计算机视觉突破的小团队，新突破仍可能来自规模以外的方向；
- **多模态的增益**：视觉与语言结合后，各自都变得更强；
- **类人与非类人方法并行**：模仿人类推理、感知的方法便于理解与应用，非类人的方法可能找到人脑想不到的解法；
- **技能半衰期**：AI 领域的技能过时得越来越快，比掌握某项技能更重要的是**学习速度**；
- **多种方法融合**：监督、无监督、强化学习组合在一个系统里。

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
> 应用层调整的是条件 $x$。内部分析检查模型依据哪些特征做判断，以及某个子群是否被总体准确率掩盖。这些量用来决定下一实验，而不是写完报告再补说明。

<br>

## 内部分析

### 目的

可解释性要回答网络依据哪些特征做判断，以及失败能否定位到某一层、某类样本。工具包括学习曲线、切片指标、显著性和 Grad-CAM。这些量用来决定下一实验，不是写完报告再补的说明。对应 Lecture 10。

可用工具：

- **学习曲线**：训练与 dev 损失、指标同图，区分欠拟合、过拟合、优化发散。
- **混淆矩阵（confusion matrix）与切片**：按光照、语言、设备切片，避免总体准确率掩盖子群失败。
- **显著性 / Grad-CAM**：$\partial y_c/\partial x$ 或特征图层的加权，检查模型是否盯着物体而非水印。
- 显著图可被输入变换愚弄，只能当假设生成器。
- **最近邻嵌入**：把验证样本嵌到表示空间，看错例是否与另一类混叠。
- **消融（ablation）**：去掉增强、正则或某一数据源，量化贡献。

- 注意力权重、某神经元的最大化图都不是因果证明。
- 可解释性服务于误差分析与安全审查：决定下一周改数据还是改模型。

### 显著性与遮挡

- **显著图**（saliency map）：类别分数 $y_c$（softmax 之前的分数）对输入像素的梯度 $\left|\partial y_c/\partial x\right|$，绝对值大的像素表示“稍微改动就会显著影响该类分数”，可以粗略分割出物体。
- **遮挡敏感度**（occlusion sensitivity）：用灰色方块依次遮住图像的各个位置，记录每次的类别概率，画成热力图。遮住某处概率骤降，说明模型依赖该区域。不需要梯度，但要做很多次前向。
- **类别模型可视化**（class model visualization）：固定权重，从零图开始对输入做梯度上升 $\arg\max_x\, y_c(x)-\lambda\|x\|_2^{2}$，得到“模型心目中最像第 $c$ 类”的图像。正则项防止图像变成无意义的高频噪声。
- **中间层可视化**：对某个卷积通道，找出让它激活最大的训练图像块，或用反卷积（deconvolution）把激活投影回像素空间。卷积一节中“第 1 层看边缘、第 3 层看纹理、第 5 层看物体部件”的结论就来自这类研究。

下面把 Lecture 10 中的几种方法逐个展开。

**积分梯度**（integrated gradients，Sundararajan et al., 2017）。显著图只看输入点处的梯度，有一个缺陷：网络在某个特征上已经**饱和**时（例如“大象鼻子”这个特征已经足够强，再强一点分数也不变），该处梯度接近 $0$，显著图会说这个特征不重要，实际上它最重要。积分梯度从一张基线图 $x'$（通常是全黑图）沿直线走到 $x$，把一路上的梯度累加：

$$
\mathrm{IG}_i(x)=(x_i-x_i')\int_0^1\frac{\partial y_c\big(x'+t(x-x')\big)}{\partial x_i}\,\mathrm dt.
$$

它满足**完备性**：所有像素的归因之和等于 $y_c(x)-y_c(x')$，即分数从基线到原图的总变化被完整分配到各个像素上，显著图没有这个性质。实际计算时用 $20$ 到 $300$ 个点做黎曼和。

```python
def integrated_gradients(model, x, c, baseline=None, steps=64):
    """x: (1, 3, H, W)。返回与 x 同形的归因图。"""
    model.eval()
    baseline = torch.zeros_like(x) if baseline is None else baseline
    t = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
    path = (baseline + t * (x - baseline)).requires_grad_(True)       # (steps, 3, H, W)
    grads, = torch.autograd.grad(model(path)[:, c].sum(), path)
    return (x - baseline) * grads.mean(0, keepdim=True)              # 梯度沿路径平均再乘位移

# 完备性检查：ig.sum() ≈ model(x)[0, c] - model(baseline)[0, c]，steps 越大越接近
```

**数据集搜索**（dataset search）。想知道某一层的第 $k$ 个通道在“找”什么，就把整个数据集过一遍，记录每张图在这个通道上的最大激活，取激活最大的前几张图看它们的共同点。再利用感受野，只截出对应的图像块，比看整张图更精确：某个通道的前 9 个图块全是狗脸，它大概就是“狗脸检测器”。它只需要前向，最简单，但只能看到数据集中已有的模式。

**反卷积可视化**（deconvnet，Zeiler & Fergus, 2014）。目标是把某个激活“追溯”回像素空间，看输入中是哪些像素让它亮起来。做法是保留这一个激活、把同层其他激活置零，然后把网络**反过来走**：

- 卷积的反向用**转置卷积**（滤波器翻转，即“反卷积”名字的由来）；
- 最大池化的反向用**反池化**（unpooling）：前向时记下每个窗口最大值的位置（switches），反向时把值放回这个位置、其余填 $0$；
- ReLU 的反向仍然过一次 ReLU，只保留正的信号。

最终得到的图像显示出这个神经元“看到”的结构。Zeiler 和 Fergus 正是用它发现 AlexNet 第一层有失效的滤波器，调整步幅和核大小后得到了 ZFNet，赢下 ILSVRC 2013。

**DeepDream**（Mordvintsev et al., 2015）。思路与类别模型可视化相同，都是对输入做梯度上升；区别在于目标不是某个类别的分数，而是**某一层激活的整体强度**：

$$
x\leftarrow x+\alpha\,\frac{\nabla_x L}{\lVert\nabla_x L\rVert},\qquad L=\frac12\big\lVert a^{[l]}(x)\big\rVert_2^2.
$$

网络在这张图中“隐约看到”的东西会被放大：浅层放大纹理和笔触，深层放大眼睛、狗脸等物体部件。从一张真实照片出发，就得到那种满是狗脸和眼睛的迷幻图像。课上把它作为“网络学到了什么”的直观演示，而不是严格的解释工具。

```python
def deep_dream(model_upto_layer, x, steps=20, lr=0.01):
    """model_upto_layer: 截到目标层为止的子网络；x: (1, 3, H, W)，已归一化。"""
    x = x.clone().requires_grad_(True)
    for _ in range(steps):
        loss = 0.5 * model_upto_layer(x).pow(2).sum()
        g, = torch.autograd.grad(loss, x)
        with torch.no_grad():
            x += lr * g / (g.abs().mean() + 1e-8)    # 归一化步长，避免梯度尺度随层变化
    return x.detach()
```

几种方法的对比：

| 方法 | 回答的问题 | 需要梯度 | 代价 |
| :--- | :--- | :--- | :--- |
| 显著图 | 哪些像素对类别分数影响大 | 是 | 一次反向 |
| 积分梯度 | 同上，且归因之和等于分数变化 | 是 | 几十次反向 |
| 遮挡敏感度 | 遮住哪里分数下降 | 否 | 很多次前向 |
| CAM / Grad-CAM | 类别依据图中哪些区域 | Grad-CAM 需要 | 一次反向 |
| 类别模型可视化 | 模型心中的“典型第 $c$ 类”长什么样 | 是 | 迭代优化 |
| 数据集搜索 | 某个神经元对哪些真实图像响应最强 | 否 | 遍历数据集 |
| 反卷积 | 某个激活由哪些像素引起 | 反向结构 | 一次反向 |
| DeepDream | 某层在图中“看到”了什么 | 是 | 迭代优化 |

### Grad-CAM

> [!INFO]+ Grad-CAM
>
> Grad-CAM（Selvaraju et al., 2017）用 $\partial y_c/\partial A^{k}$ 对卷积特征图加权，得到该类别的热力图。它只提供假设：模型可能依据哪些区域。热力图对输入变换不稳定，需要再用遮挡或切片核对。

<br>

**先说 CAM**（class activation map，Zhou et al., 2016）。全连接层把空间信息打乱了，很难追溯回某个位置。CAM 要求网络末端是“最后一层卷积 → 全局平均池化（GAP）→ 一个全连接层”：最后一层特征图 $A^{k}$（$k=1,\dots,K$）先在空间上取平均得到 $K$ 维向量，再乘全连接权重 $w^{c}_k$ 得到类别分数

$$
y_c=\sum_k w^{c}_k\cdot\frac1Z\sum_{i,j}A^{k}_{ij}=\frac1Z\sum_{i,j}\underbrace{\sum_k w^{c}_kA^{k}_{ij}}_{M_c(i,j)}.
$$

求和可以交换顺序，于是类别分数等于热力图 $M_c=\sum_k w^{c}_kA^{k}$ 的空间平均：$M_c(i,j)$ 直接给出位置 $(i,j)$ 对类别 $c$ 贡献了多少。这就是课上说的“GAP 让信息更容易追溯回局部”。缺点是必须改网络结构并重新训练。

**Grad-CAM 去掉了这个限制**。最后一个卷积层的特征图既保留了空间位置，又已经包含高级语义。Grad-CAM 对任意结构的网络、卷积特征图 $A^{k}$ 与类别分数 $y_c$：

$$
\alpha_k=\frac{1}{Z}\sum_{i,j}\frac{\partial y_c}{\partial A^{k}\_{ij}},\qquad
L_{\mathrm{CAM}}=\mathrm{ReLU}\Big(\sum_k \alpha_k A^{k}\Big).
$$

$\alpha_k$ 是梯度在空间上的全局平均（$Z$ 为空间位置数），表示第 $k$ 个通道对类别 $c$ 有多重要。对上面 CAM 的结构，$\partial y_c/\partial A^{k}_{ij}=w^{c}_k/Z$，所以 $\alpha_k=w^{c}_k/Z$，Grad-CAM 恰好退化为（差一个常数的）CAM：Grad-CAM 是 CAM 对任意网络的推广。

用 forward hook 截取特征图 $A$，用 `torch.autograd.grad` 求 $\partial y_c/\partial A$：

```python
def grad_cam(model, layer, x, target_class=None):
    """model: 分类网络；layer: 最后一个卷积块（如 resnet.layer4）；x: (1, 3, H, W)。"""
    model.eval()
    store = {}
    handle = layer.register_forward_hook(lambda m, inp, out: store.__setitem__("A", out))
    logits = model(x)
    handle.remove()
    c = logits.argmax(1).item() if target_class is None else target_class
    A = store["A"]                                               # (1, K, h, w)
    grads, = torch.autograd.grad(logits[0, c], A)                # ∂y_c/∂A
    alpha = grads.mean(dim=(2, 3), keepdim=True)                 # 空间全局平均 → (1, K, 1, 1)
    cam = F.relu((alpha * A).sum(1, keepdim=True))               # (1, 1, h, w)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)     # 归一化到 [0,1] 便于叠加显示
    return cam[0, 0].detach(), c

# m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
# heat, cls = grad_cam(m, m.layer4, img_tensor)
```

用的是 softmax 之前的分数 $y_c$ 而不是概率：概率还受其他类别影响，梯度会被抵消。

遮挡实验的对照写法：用灰色方块滑过图像，记录 $y_c$ 的下降。

```python
@torch.no_grad()
def occlusion_map(model, x, c, patch=16, stride=8, fill=0.0):
    _, _, H, W = x.shape
    base = model(x)[0, c].item()
    rows = []
    for i in range(0, H - patch + 1, stride):
        row = []
        for j in range(0, W - patch + 1, stride):
            x_occ = x.clone()
            x_occ[..., i:i + patch, j:j + patch] = fill
            row.append(base - model(x_occ)[0, c].item())   # 下降越多，区域越重要
        rows.append(row)
    return torch.tensor(rows)
```

- 热力图上采样到原图尺寸。
- ReLU 丢掉对类别有负贡献的区域。
- 热力图落在角落水印或床头文字而不是病灶：模型可能在抄捷径（shortcut）。
- 显著性图对输入变换不稳定，只能提出假设。
- 再用切片指标或遮挡实验验证：把高显著区域涂掉，若 $y_c$ 不变，则解释不可信。

### 大模型内部

Lecture 10 的后半部分把同样的问题（模型内部在发生什么）搬到 Transformer 与 LLM 上。图像上的工具大多仍然适用，另外多了几样。

**注意力图**。每个注意力头的权重矩阵 $A=\mathrm{softmax}(QK^\top/\sqrt{d_k})$ 本身就是一张“第 $i$ 个词在看哪些词”的热力图（见 Transformer 一章）。翻译模型中能看到近似对角线的对齐；有些头专门把代词指向它指代的名词，有些头只看前一个词。注意事项与 Grad-CAM 一样：注意力权重大不等于这个词对输出的因果贡献大（Jain & Wallace, 2019），要配合遮挡词或替换词的实验检验。

```python
# 取出每层每个头的注意力：HuggingFace 模型加 output_attentions=True
# out = model(**tok("The animal didn't cross the street because it was tired", return_tensors="pt"),
#             output_attentions=True)
# att = out.attentions[layer][0, head]          # (T, T)，第 i 行是第 i 个词对所有词的权重
# att[tokens.index("it")].topk(3)               # "it" 最关注的 3 个词
```

**探针**（probing）。在冻结的中间层表示 $h^{[l]}$ 上训练一个很小的线性分类器，预测某个属性（词性、句法深度、“这句话是否真实”）。线性探针准确率高，说明这个信息以线性可读的形式存在于第 $l$ 层。逐层画出探针准确率，能看到浅层编码词法、中层编码句法、深层编码语义的趋势，与 CNN 的“边缘 → 纹理 → 部件”对应。

**稀疏自编码器**（sparse autoencoder，SAE）。单个神经元往往同时响应好几种不相关的概念（多义性，polysemanticity），直接看神经元很难解释。SAE 把 $h$ 重构为一个过完备、稀疏的特征组合：

$$
f=\mathrm{ReLU}(W_e h+b_e),\qquad \hat h=W_d f,\qquad L=\lVert h-\hat h\rVert^2+\lambda\lVert f\rVert_1.
$$

$f$ 的维度远大于 $h$，但每次只有少数分量非零，每个分量往往对应一个可以命名的概念（例如“金门大桥”“代码中的 bug”）。把某个特征强行调大，模型的输出会随之偏向该概念，可以借此检验这个特征的作用。

**规模定律**（scaling laws，Kaplan et al., 2020；Hoffmann et al., 2022）。测试损失与参数量 $N$、数据量 $D$、计算量 $C$ 近似成幂律：

$$
L(N)\approx\Big(\frac{N_c}{N}\Big)^{\alpha_N},\qquad L(D)\approx\Big(\frac{D_c}{D}\Big)^{\alpha_D},
$$

在双对数坐标下是直线。这和“项目策略”一章的学习曲线是同一个工具：小规模实验拟合出直线，外推到大规模，决定钱花在更大的模型上还是更多的数据上。Chinchilla 的结论是，给定计算量时参数与 token 应同比例增长（约每个参数 20 个 token），此前很多大模型参数过多、训练数据不足。

```python
def fit_power_law(N, L):
    """在 log-log 空间做线性回归：log L = a - alpha * log N。"""
    alpha, a = np.polyfit(np.log(N), np.log(L), 1)
    return -alpha, np.exp(a)

N = np.array([1e6, 1e7, 1e8, 1e9])
L = 5.0 * N ** -0.076                     # 构造数据，指数取 Kaplan 报告的 alpha_N 量级
alpha, c = fit_power_law(N, L)
print(round(alpha, 3), round(c * (1e10) ** -alpha, 3))   # 0.076 ，外推到 1e10 参数的损失 0.869
```

**基准与污染**。MMLU、GSM8K、HumanEval 等基准衡量的是“模型在这批题上的表现”，有两个常见陷阱：

- **污染**（contamination）：题目和答案出现在预训练语料里，模型可能是背下来的。检查方法有在训练语料中搜 $n$-gram 重叠、改写题目后看成绩是否明显下降；
- **饱和与错位**：头部模型都接近满分后，基准失去区分度；基准成绩也不等于在具体产品任务上的表现。所以课上反复强调要有自己任务上的 evals（见 LLM 应用一章的评估）。

**轨迹**（traces）。对智能体系统，“内部”还包括每一步的提示、工具调用与返回结果。逐步读 trace，定位是哪一步出的错（检索没找到、工具参数错、推理错），与对 CNN 的误差表做切片是同一种思路。

**MoE 路由**。混合专家模型（mixture of experts）每个 token 只交给少数几个专家子网络处理。统计每个专家被选中的频率，可以看到有的专家专门处理代码、有的专门处理标点；如果大部分 token 都涌向少数专家（负载不均），训练时要加负载均衡损失。

### 切片与学习曲线

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

# 作业（Assignments）

## 安排

每周编程与测验在 Coursera 的 Deep Learning Specialization，截止日期以 [Syllabus](https://cs230.stanford.edu/syllabus/) 为准，不是 Coursera 默认周历。2025 Autumn：每周二 11:00 a.m. PST 之后算迟交。下面按作业写出核心函数与对拍检查，不照抄 notebook 题面。

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

## 测验

Coursera Quiz 与英文短答必须能口述以下结论。

- **C1M1, Introduction to Deep Learning**
  - Increasing data and model scale helps only when train and dev distributions are appropriate.
  - Structured data often uses standard neural networks; images use CNNs; sequences use RNNs or Transformers.
- **C1M2, Neural Network Basics**
  - Vectorization replaces explicit loops over examples.
  - With column-major examples, $X$ is $(n_x,m)$ and broadcasting adds $b$ along columns.
  - Sigmoid with binary cross-entropy gives $\mathrm{d}Z=A-Y$.
- **C1M3, Shallow Neural Networks**
  - Hidden units must not start with identical weights because symmetry would be preserved.
  - A linear hidden layer collapses with adjacent linear maps into one linear transformation.
  - Tanh is zero-centered but still saturates; ReLU avoids positive-side saturation.
- **C1M4, Deep Neural Networks**
  - Cache $A^{[l-1]}$, $W^{[l]}$ and activation information during the forward pass.
  - Backprop proceeds from layer $L$ to layer $1$ and uses pre-update weights.
- **C2M1, Practical Aspects**
  - High train error indicates avoidable bias or optimization failure; a large train-dev gap indicates variance.
  - Xavier suits tanh-like activations; He suits ReLU.
  - Inverted dropout divides by keep probability during training and is disabled at test time.
  - Gradient checking requires deterministic execution, so Dropout is turned off.
- **C2M2, Optimization**
  - Mini-batch SGD trades gradient noise against hardware throughput.
  - Momentum tracks the first moment; RMSprop tracks squared gradients; Adam combines both with bias correction.
  - Learning-rate decay matters after progress reaches a plateau, not as a repair for incorrect gradients.
- **C2M3, Hyperparameters and BatchNorm**
  - Search learning rate on a logarithmic scale.
  - BatchNorm uses batch statistics during training and running statistics during inference.
  - $\gamma$ and $\beta$ are learnable; running mean and variance are not optimizer parameters.
- **C3, ML Strategy**
  - Use one optimizing metric and satisficing constraints.
  - Dev and test sets should represent the deployment distribution and come from the same distribution.
  - Manual error analysis estimates the maximum possible gain from fixing an error category.
- **C4M1, CNN Foundations**
  - Convolution uses parameter sharing and sparse connections.
  - Output size is $\lfloor(n+2p-f)/s\rfloor+1$; pooling has no learned weights.
- **C4M2, Deep CNNs**
  - Residual connections provide a short gradient path.
  - A $1\times1$ convolution mixes channels while preserving spatial size when stride is $1$.
  - Transfer learning freezes more layers when target data are scarce and fine-tunes more when data are abundant.
- **C4M3/C4M4, Detection and Applications**
  - IoU measures overlap, NMS removes duplicate predictions, and mAP evaluates ranked detections.
  - Verification is one-to-one; identification is one-to-many.
  - U-Net concatenates encoder features into the decoder to recover localization detail.
- **C5M1, Recurrent Models**
  - RNN parameters are shared across time; BPTT sums their gradient contributions over time.
  - GRU and LSTM gates provide paths that mitigate vanishing gradients.
- **C5M2/C5M3, Embeddings and Attention**
  - Similar embeddings have high cosine similarity; negative sampling avoids a full vocabulary softmax.
  - Attention weights sum to $1$ over source positions for each decoder step.
  - Beam search is approximate and is not guaranteed to find the globally most likely sequence.
- **C5M4, Transformers**
  - Scaled dot-product attention divides by $\sqrt{d_k}$ to control score variance.
  - A causal mask prevents access to future tokens.
  - Self-attention uses one sequence for $Q,K,V$; cross-attention takes $Q$ from the decoder and $K,V$ from the encoder.

记号与课程一致：样本数是 $m$，特征矩阵 $X\in\mathbb{R}^{n_x\times m}$。标签 $y\in\{0,1\}$。迟交共 $10$ 天，单份最多 $3$ 天。

> [!NOTE]+ 读法
>
> 每份作业先在纸上写出 **shape**，再写代码。自动评分最常见的失败是 `(n_x, m)` 与 `(m, n_x)` 对调，或 `keepdims` 丢了。
>
> | 作业 | 易错点 |
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
> **初始化**。全零：隐层对称，每个单元学同一函数。过大高斯：ReLU 网络的激活与梯度可能爆炸并数值溢出；sigmoid / tanh 则进入饱和区。He 对 ReLU，Xavier 对 $\tanh$。
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
> **注意力翻译**。对齐分数 $e_{t,t'}=\tanh$ 或加性注意力，再 $\alpha_{t,t'}=\mathrm{softmax}\_{t'}(e_{t,\cdot})$，上下文 $c_t=\sum_{t'}\alpha_{t,t'}h_{t'}$。$\alpha$ 对 $t'$ 求和必须为 $1$。
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
> 作业按 Coursera 周组织。下一章把同一套模块接到自选数据：先基线，再误差表。Coursera 的数据集不能直接当作项目数据。

<br>

# 项目（Project）

## 要求

CS230 的主交付物是可复现的深度学习项目，不能把 Coursera 作业改个标题再交。时间线（2025 Autumn）是提案、中期、期末报告与海报。选题之后先约定指标和划分，跑通基线，再按误差表迭代。

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

# 复习（Review）

- 课程把深度学习当成可微分模块的组合：仿射或卷积、逐点非线性、归一化、注意力、损失。
- 把模块写对之后，项目成败由数据、指标与误差归因决定。

$$
\begin{aligned}
\text{拟合训练} &\leftarrow \text{容量、优化、初始化},\\
\text{拟合开发} &\leftarrow \text{正则、数据、分布对齐},\\
\text{拟合部署} &\leftarrow \text{dev/test 是否与部署同分布}.
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

## 英文

英文题目首先看动作词（command verb），再看限制条件。

- **compute / calculate**：算出数值，必须写中间尺寸或代入过程。
- **derive**：从已知式逐步推导，不只写结论。
- **show that**：证明题，写出等号或不等式成立的理由。
- **prove**：给完整论证，例如用 Jensen inequality 证明 KL non-negativity。
- **state**：直接写定义或结论，不需要展开长推导。
- **explain why / justify**：给因果机制，不能只复述现象。
- **compare and contrast**：同时写共同点与差异。
- **give one reason / in one sentence**：只需一条准确理由，避免堆砌无关结论。
- **circle all that apply**：多选题，可能不止一个正确答案。
- **which statement is false**：找错误项，先圈出题目中的否定词。
- **with respect to**：关于某变量求导，例如 gradient with respect to $W$。
- **element-wise / entry-wise**：逐元素运算，不是矩阵乘法。
- **in expectation**：对随机变量取期望，不表示每次采样都成立。
- **at training time / at test time**：训练阶段 / 测试阶段，常用于 Dropout 与 BatchNorm。

各章需要能从英文直接还原的核心词如下。

- 数学与实现：计算图（computational graph）、链式法则（chain rule）、雅可比矩阵（Jacobian）、向量化（vectorization）、广播（broadcasting）、数值稳定性（numerical stability）、梯度检查（gradient checking）、有限差分（finite difference）。
- 网络：前向传播（forward propagation）、反向传播（backpropagation）、预激活（pre-activation）、隐藏单元（hidden unit）、逐点非线性（pointwise nonlinearity）、通用逼近（universal approximation）、表示能力（representational capacity）。
- 训练：参数（parameter）、超参数（hyperparameter）、损失函数（loss function）、目标函数（objective function）、经验风险（empirical risk）、优化器（optimizer）、收敛（convergence）、鞍点（saddle point）。
- 梯度问题：消失梯度（vanishing gradient）、爆炸梯度（exploding gradient）、梯度裁剪（gradient clipping）、死亡 ReLU（dying ReLU）、饱和区（saturated regime）。
- 初始化：随机初始化（random initialization）、对称性破缺（symmetry breaking）、扇入（fan-in）、扇出（fan-out）、Xavier / Glorot initialization、He / Kaiming initialization。
- 正则：过拟合（overfitting）、欠拟合（underfitting）、权重衰减（weight decay）、反向 Dropout（inverted dropout）、保留概率（keep probability）、数据增强（data augmentation）、提前停止（early stopping）、标签平滑（label smoothing）。
- 优化：批量梯度下降（batch gradient descent）、随机梯度下降（stochastic gradient descent, SGD）、小批量（mini-batch）、动量（momentum）、指数滑动平均（exponentially weighted moving average）、偏差修正（bias correction）、学习率衰减（learning-rate decay）、学习率预热（learning-rate warmup）。
- 归一化：批归一化（batch normalization）、层归一化（layer normalization）、批统计（batch statistics）、运行均值（running mean）、运行方差（running variance）、可学习缩放和平移（learnable scale and shift）。
- 诊断：可避免偏差（avoidable bias）、方差（variance）、贝叶斯误差（Bayes error）、人类水平表现（human-level performance）、误差分析（error analysis）、训练与开发集不匹配（train-dev mismatch）、分布漂移（distribution shift）。
- 指标：混淆矩阵（confusion matrix）、精确率（precision）、召回率（recall）、灵敏度（sensitivity）、特异度（specificity）、假阳性率（false positive rate）、假阴性率（false negative rate）、ROC 曲线下面积（area under the ROC curve, AUROC）。
- 卷积：感受野（receptive field）、参数共享（parameter sharing）、稀疏连接（sparsity of connections）、有效卷积（valid convolution）、同尺寸卷积（same convolution）、步幅（stride）、填充（padding）、池化（pooling）、通道（channel）、过滤器（filter / kernel）、深度可分离卷积（depthwise separable convolution）。
- 视觉任务：边界框（bounding box）、交并比（intersection over union, IoU）、非极大值抑制（non-maximum suppression, NMS）、锚框（anchor box）、均值平均精度（mean average precision, mAP）、语义分割（semantic segmentation）、实例分割（instance segmentation）。
- 序列：时间反向传播（backpropagation through time, BPTT）、隐藏状态（hidden state）、门控循环单元（gated recurrent unit, GRU）、长短期记忆（long short-term memory, LSTM）、双向循环网络（bidirectional RNN）、教师强制（teacher forcing）、暴露偏差（exposure bias）、束搜索（beam search）。
- 语言与注意力：词嵌入（word embedding）、词表外词（out-of-vocabulary word, OOV）、余弦相似度（cosine similarity）、上下文向量（context vector）、查询（query）、键（key）、值（value）、缩放点积注意力（scaled dot-product attention）、因果掩码（causal mask）、位置编码（positional encoding）。
- 生成与强化学习：对抗样本（adversarial example）、生成器（generator）、判别器（discriminator）、模式崩溃（mode collapse）、证据下界（evidence lower bound, ELBO）、重参数化技巧（reparameterization trick）、经验回放（experience replay）、目标网络（target network）、策略梯度（policy gradient）、优势函数（advantage function）、奖励模型（reward model）。

> [!EXAMPLE]+ 公开试卷风格的英文问法
>
> **Why is scaling $\gamma$ and shifting $\beta$ applied after normalization?**
>
> 标准化本身会限制表示只能保持零均值和单位方差。Learnable scale and shift restore the representational capacity of the layer.
>
> **Is it always a good strategy to train with a large batch size? Why or why not?**
>
> No. A large batch gives a lower-variance gradient estimate and better hardware utilization, but uses more memory, performs fewer updates per epoch, and may require retuning the learning rate.
>
> **Which statement about Xavier initialization is false: “the variance remains unchanged throughout training”?**
>
> 该陈述错误。Xavier controls variance only at initialization; subsequent parameter updates can change it.
>
> **Why can a $1\times1$ convolution reduce channels without changing spatial dimensions?**
>
> At every spatial location, it applies a learned linear combination across input channels. With stride $1$ and zero padding, height and width stay unchanged, while the number of filters sets the output channels.
>
> **Why is KL divergence not a distance metric?**
>
> It is generally asymmetric and does not satisfy the triangle inequality, even though it is non-negative.
>
> **What happens if all hidden-layer weights are initialized to the same constant?**
>
> All neurons produce the same output, receive the same gradient, and remain identical after every update. Random initialization is needed for symmetry breaking.

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
> - 拟合部署失败：dev/test 与部署数据不同源。
> - 课堂扩展主要讨论部署阶段：对抗扰动、仿真到真实、检索与协议。

<br>


# 参考文献（References）

## 课程

1. Stanford CS230. *Deep Learning* 课程主页与大纲。[https://cs230.stanford.edu/](https://cs230.stanford.edu/)；[Syllabus](https://cs230.stanford.edu/syllabus/)
2. Stanford Bulletin. *CS230: Deep Learning*. [https://bulletin.stanford.edu/courses/2198851](https://bulletin.stanford.edu/courses/2198851)
3. deeplearning.ai. *Deep Learning Specialization*（C1-C5，CS230 翻转课堂视频与作业来源）。
4. Stanford CS230. *Winter 2021 Exam* 与 [solution](https://cs230.stanford.edu/files/cs230exam_win21_soln.pdf)：英文题干、初始化、BatchNorm、卷积与反向传播题型。
5. Stanford CS230. *Winter 2020 Exam Solution*. [https://cs230.stanford.edu/files/cs230exam_win20_soln.pdf](https://cs230.stanford.edu/files/cs230exam_win20_soln.pdf)
6. Stanford CS230. *Fall 2025 Lecture 5: Deep Reinforcement Learning and RLHF*. [lecture_5.pdf](https://cs230.stanford.edu/syllabus/fall_2025/5/lecture_5.pdf)
7. Amidi A, Amidi S. *CS230 Deep Learning Cheatsheets*（卷积、循环网络、训练技巧速查表）。[https://stanford.edu/~shervine/teaching/cs-230/](https://stanford.edu/~shervine/teaching/cs-230/)
8. 斋藤康毅. *深度学习入门：基于 Python 的理论与实现*. 陆宇杰 译. 人民邮电出版社, 2018.（感知机、计算图、误差反向传播、im2col、CNN 可视化等例题）
9. Katanforoosh K. *CS230 Lecture 2: Deep Learning Intuition / Decision Making in AI Projects*（昼夜分类、人脸验证、风格迁移、触发词案例）。[Fall 2018 slides](http://cs230.stanford.edu/fall2018/slides_week2.pdf)；[Spring 2022 slides](https://cs230.stanford.edu/syllabus/spring_2022/lecture_2.pdf)；[Autumn 2025 视频](https://www.youtube.com/watch?v=DNCn1BpCAUY)
10. Ng A. *CS230 Autumn 2025 Lecture 3: Full Cycle of a DL Project*（数据收集速度、VAD、数据漂移、监控）。[视频](https://www.youtube.com/watch?v=MGqQuQEUXhk)
11. Ng A. *CS230 Autumn 2018 Lecture 8: Career Advice / Reading Research Papers*. [视频](https://www.youtube.com/watch?v=733m6qBH-jI)
12. Amidi A, Amidi S. *CS230 Deep Learning Tips and Tricks Cheatsheet*（数据增强、权重正则、小 batch 过拟合、梯度检查）。[链接](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)
13. Stanford CS230. *Section 3: Baseline and Codebase*. [https://cs230.stanford.edu/section/3/](https://cs230.stanford.edu/section/3/)
14. Deitke M. *CS230 Lecture Notes*（课堂讲座笔记整理）。[https://mattdeitke.com/notes/cs230](https://mattdeitke.com/notes/cs230)

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
8. LeCun Y et al. *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 1998.（LeNet）
9. Krizhevsky A et al. *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS, 2012.（AlexNet）
10. Simonyan K, Zisserman A. *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR, 2015.（VGG）
11. Szegedy C et al. *Going Deeper with Convolutions*. CVPR, 2015.（GoogLeNet / Inception）
12. Zeiler MD, Fergus R. *Visualizing and Understanding Convolutional Networks*. ECCV, 2014.
13. Girshick R et al. *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation*. CVPR, 2014.（R-CNN）
14. Ren S et al. *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS, 2015.
15. Long J et al. *Fully Convolutional Networks for Semantic Segmentation*. CVPR, 2015.（FCN）
16. Badrinarayanan V et al. *SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation*. TPAMI, 2017.
17. Sandler M et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR, 2018.
18. Tan M, Le QV. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML, 2019.
19. Selvaraju RR et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV, 2017.

## 序列与注意力

1. Hochreiter S, Schmidhuber J. *Long Short-Term Memory*. Neural Computation, 1997.
2. Cho K et al. *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP, 2014.
3. Sutskever I et al. *Sequence to Sequence Learning with Neural Networks*. NeurIPS, 2014.
4. Bahdanau D et al. *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR, 2015.
5. Vaswani A et al. *Attention Is All You Need*. NeurIPS, 2017.
6. Mikolov T et al. *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS, 2013.
7. Bengio Y et al. *A Neural Probabilistic Language Model*. JMLR, 2003.
8. Pennington J et al. *GloVe: Global Vectors for Word Representation*. EMNLP, 2014.
9. Bolukbasi T et al. *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*. NeurIPS, 2016.
10. Papineni K et al. *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL, 2002.
11. Vinyals O et al. *Show and Tell: A Neural Image Caption Generator*. CVPR, 2015.（NIC）
12. Graves A et al. *Connectionist Temporal Classification*. ICML, 2006.

## 对抗、生成与强化学习

1. Goodfellow IJ et al. *Explaining and Harnessing Adversarial Examples*. ICLR, 2015.
2. Goodfellow I et al. *Generative Adversarial Nets*. NeurIPS, 2014.
3. Kingma DP, Welling M. *Auto-Encoding Variational Bayes*. ICLR, 2014.
4. Ho J et al. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020.
5. Mnih V et al. *Human-level control through deep reinforcement learning*. Nature, 2015.
6. Schulman J et al. *Proximal Policy Optimization Algorithms*. 2017.
7. Radford A et al. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. ICLR, 2016.（DCGAN）
8. Silver D et al. *Mastering the game of Go with deep neural networks and tree search*. Nature, 2016.（AlphaGo）
9. Lewis P et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS, 2020.
10. Hu EJ et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR, 2022.

<br>

# 工具（Tools）

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
