---
title: 斯坦福CS230习题与解析整理
date: 2026-10-09
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS230 公开期中与练习卷的中英对照题：选择、损失、优化、卷积、反向传播、GAN 与编程作业目标，附解析。
---



<br>


# 参考

## 官方

| 卷 | 公开位置 |
| :--- | :--- |
| Practice Midterm 1（Winter 2020 练习卷） | [Practice_1_Sol.pdf](https://cs230.stanford.edu/section/6/Practice_1_Sol.pdf) |
| Practice Midterm 2（Fall 2019 期中，Section 6 作练习） | [Practice_2_Sol.pdf](https://cs230.stanford.edu/section/6/Practice_2_Sol.pdf) |
| Fall 2020 Midterm | [cs230exam_fall20_soln.pdf](https://cs230.stanford.edu/files/cs230exam_fall20_soln.pdf) |
| Winter 2021 Midterm | [cs230exam_win21_soln.pdf](https://cs230.stanford.edu/files/cs230exam_win21_soln.pdf) |
| Spring 2021 Midterm | [cs230exam_spr21.pdf](https://cs230.stanford.edu/syllabus/fall_2021/cs230exam_spr21.pdf) |
| Fall 2022 Midterm | [CS230_Midterm_Solutions_Fall_2022.pdf](https://cs230.stanford.edu/syllabus/fall_2022/CS230_Midterm_Solutions_Fall_2022.pdf) |
| 入口 | [Section 6：Practice Midterm](https://cs230.stanford.edu/section/6/) |


<br>

## 课程资料

| 仓库 | 内容 | 使用建议 |
| :--- | :--- | :--- |
| [cs230-stanford/cs230-code-examples](https://github.com/cs230-stanford/cs230-code-examples) | Stanford 官方项目代码示例，含 PyTorch / TensorFlow 与 Vision / NLP 四套结构 | 参考项目目录、数据划分、超参搜索、评估入口，不是 Coursera 作业答案 |
| [maxim5/cs230-2018-autumn](https://github.com/maxim5/cs230-2018-autumn) | Fall 2018 讲义、课堂 slides、section、复习材料与期中卷归档 | 补历史课堂内容；`notes/` 按 C1M1–C5M3 排列，`exam/` 含 Fall 2018 试卷与解答 |
| [afshinea/stanford-cs-230-deep-learning](https://github.com/afshinea/stanford-cs-230-deep-learning) | CNN、RNN、训练技巧与 Super VIP 总表 | 考前压缩复习；MIT License，多语言版本 |
| [kozemzak/cs230-cheat-sheet](https://github.com/kozemzak/cs230-cheat-sheet) | Autumn 2018 期中 LaTeX 速查表 | 可修改 `main.tex`；还收入当年课堂的 GAN 内容 |

## 作业归档

| 仓库 | 覆盖 | 版本判断 |
| :--- | :--- | :--- |
| [amanchadha/coursera-deep-learning-specialization](https://github.com/amanchadha/coursera-deep-learning-specialization) | C1–C5 笔记、周测目录、全部编程 notebook；约 2021 年后 TensorFlow 2 版本 | 当前公开归档中最完整，含 MobileNet、U-Net、Transformer、NER 与 QA |
| [ShawnZhong/Coursera-Deep-Learning-Course](https://github.com/ShawnZhong/Coursera-Deep-Learning-Course) | C1–C5 旧版编程作业 | 适合核对早期题型；没有新版 Transformer、U-Net 等更新 |
| [TheKidPadra/DeepLearning.AI-Deep-Learning-Specialization](https://github.com/TheKidPadra/DeepLearning.AI-Deep-Learning-Specialization) | 各周 notebook 与测验 PDF | 可作文件镜像，题目版权仍属于 deeplearning.ai |
| [thanhhff/CS230-Deep-Learning](https://github.com/thanhhff/CS230-Deep-Learning) | 按课程与 week 排列的早期 DLS 作业 | 目录直观，适合查 C1/C2/C4/C5 旧版文件名 |


<br>


# 选择

## 参数与卷积

A 2-layer network has 5 units in the input, the hidden layer, and the output layer, with biases. The parameter count is 60.

- T
- F

输入、隐藏层、输出层宽度均为 5，且含偏置。参数总数为 60。

> [!EXAMPLE]+ T
>
> 结构是输入 $5$、隐藏 $5$、输出 $5$，中间只有两层仿射，两层都带偏置。
>
> - 输入到隐藏的权重：每个隐藏单元连到全部 $5$ 个输入，$5$ 个隐藏单元，共 $5\times 5=25$。
> - 隐藏层偏置：每个隐藏单元一个，共 $5$。
> - 隐藏到输出的权重：每个输出单元连到全部 $5$ 个隐藏单元，$5$ 个输出单元，共 $5\times 5=25$。
> - 输出层偏置：每个输出单元一个，共 $5$。
>
> $25+5+25+5=60$。前一个 $5\times 5$ 是第一层权重，后一个 $5\times 5$ 是第二层权重；两个单独的 $5$ 分别是两层的偏置个数。官方卷把「每层 5 个神经元」按这一宽度理解（Fall 2019 / Practice 2）。

<br>


Input volume $(10,10,3)$. Compare (A) a fully connected layer with 2 units and biases, after flattening, and (B) a conv layer with three $2\times 2$ filters, padding 0, stride 2, and biases. How many fewer parameters does the conv layer have?

输入 $(10,10,3)$。全连接层 2 个神经元、含偏置，输入先展平。卷积层为 3 个 $2\times 2$ 滤波器、padding 0、stride 2、含偏置。卷积比全连接少多少个参数？

> [!EXAMPLE]+ 少 563 个
>
> 输入是高 $10$、宽 $10$、通道 $3$。展平后每个样本的长度是 $10\times 10\times 3=300$。
>
> 全连接有 $2$ 个输出单元。每个单元对这 $300$ 个数各有一个权重，再加 $1$ 个偏置：$300\times 2+2=602$。
>
> 一个 $2\times 2$ 卷积核要看满 $3$ 个输入通道，权重个数是核高 $\times$ 核宽 $\times$ 输入通道 $=2\times 2\times 3=12$，再加 $1$ 个偏置，共 $13$。三个核是 $13\times 3=39$。
>
> 卷积比全连接少 $602-39=563$。
>
> 输出边长用 $\lfloor(H+2p-f)/s\rfloor+1$。这里 $H=10$、$p=0$、$f=2$、$s=2$，所以 $\lfloor(10-2)/2\rfloor+1=5$。三个滤波器，体积是 $(5,5,3)$。

<br>


NumPy broadcasting: which products `v * u` succeed?

- (i) $(5,1)$ and $(5,1)$
- (ii) $(1,5)$ and $(50,5)$
- (iii) $(1,1)$ and $(50,5)$
- (iv) $(1,5)$ and $(1,50)$

下列 `v * u` 哪些能广播成功？

> [!EXAMPLE]+ (i)(ii)(iii)
>
> NumPy 逐元素相乘时，从最后一维往前对齐：要么相等，要么其中一边是 $1$，否则失败。
>
> - (i) $(5,1)$ 与 $(5,1)$：两维都相等，直接相乘。
> - (ii) $(1,5)$ 与 $(50,5)$：第一维 $1$ 对 $50$ 可广播成 $50$；第二维都是 $5$，得到 $(50,5)$。
> - (iii) $(1,1)$ 与 $(50,5)$：两个 $1$ 分别广播到 $50$ 和 $5$，得到 $(50,5)$。
> - (iv) $(1,5)$ 与 $(1,50)$：第一维都是 $1$，可广播；第二维是 $5$ 对 $50$，既不相等也不是 $1$，失败。

<br>


## 梯度与激活

Which statements about vanishing gradients are true?

- (i) Tanh does not suffer from vanishing gradients, so it is preferred to sigmoid.
- (ii) Vanishing gradients make deeper layers learn more slowly than earlier layers.
- (iii) Leaky ReLU is less prone to vanishing gradients than sigmoid.
- (iv) Xavier initialization can help.

关于梯度消失，哪些成立？

> [!EXAMPLE]+ (iii)(iv)
>
> - (i) 不选：tanh 在 $|x|$ 很大时仍饱和，$\tanh'(x)=1-\tanh^2(x)\to 0$，梯度照样消失。它比 sigmoid 零点附近更陡，但不是「不会消失」。
> - (ii) 不选：反传从输出层往输入层走。靠近输出的一层先收到梯度；每过一层 sigmoid/tanh 都可能乘上小于 1 的因子。**更靠近输入**的层乘的次数更多，更新更慢。题干说「更深的层更慢」方向反了。
> - (iii) 选：Leaky ReLU 在 $x<0$ 时斜率是常数 $\alpha>0$，不会像 ReLU 那样整段为 0，也不会像 sigmoid 那样饱和。
> - (iv) 选：Xavier/He 初始化让各层激活方差在起点附近稳定，输入不易一上来就挤进饱和区，减轻消失。

<br>


If $z=\tanh(x)$, then $\partial z/\partial x=1-z^2$. The backpropagated gradient through tanh is always smaller than or equal in magnitude to the upstream gradient.

- T
- F

经过 tanh 反传后，梯度幅度总是不大于上游梯度。

> [!EXAMPLE]+ T
>
> 对 $z=\tanh(x)$，局部导数是 $\partial z/\partial x=1-z^2$。
>
> - 因为 $|\tanh(x)|<1$，所以 $z^2<1$，从而 $1-z^2\in(0,1]$。
> - 反传时 $\partial L/\partial x=(\partial L/\partial z)(1-z^2)$：上游梯度乘一个 $(0,1]$ 内的正数，幅度不会变大，符号不变。
> - 只有 $z\to\pm 1$ 时因子 $\to 0$，梯度被压没，这是饱和区梯度消失，不是「幅度变大」。

<br>


A trained logistic regression has weight $W$, no bias, and test accuracy $A$. Replacing $W$ by $W/2$ leaves $A$ unchanged.

- T
- F

无偏置的逻辑回归，把 $W$ 除以 2，测试准确率不变。

> [!EXAMPLE]+ T
>
> 无偏置时，预测正类当且仅当 $W^\top x\ge 0$（等价于 $\sigma(W^\top x)\ge 0.5$）。
>
> 把 $W$ 换成 $W/2$ 后，预激活变成 $\frac12 W^\top x$。对任意 $x$：$\frac12 W^\top x$ 与 $W^\top x$ 同号（正、负或零），所以每个样本的类别判决不变，测试准确率 $A$ 不变。
>
> 若存在偏置 $b$ 且只缩放 $W$ 不缩放 $b$，边界 $W^\top x+b=0$ 会移动，本题前提是无偏置。

<br>


A 100-layer net uses sigmoid at the output and a mix of tanh and ReLU elsewhere. After one epoch some layers have essentially zero gradients, while the loss is still a reasonable magnitude. Which fixes help?

- (i) A larger training set.
- (ii) Leaky ReLU in place of ReLU.
- (iii) BatchNorm before every activation.
- (iv) A larger learning rate.

百层网络中，部分层梯度在第一轮后几乎为 0，损失数量级仍正常。哪些处理有帮助？（Fall 2020）

> [!EXAMPLE]+ (ii)(iii)
>
> 现象：部分层梯度 $\approx 0$，但总损失仍正常，说明问题在**某些层的局部饱和/死亡**，不是「数据不够」或「学习率太小」 alone 能解释。
>
> - (i) 不选：加数据不改变激活是否饱和、ReLU 是否死亡。
> - (ii) 选：Leaky ReLU 在负半轴仍有常数斜率，死亡 ReLU（恒为 0 输出、梯度为 0）可缓解。
> - (iii) 选：BatchNorm 把每层的预激活拉回均值 0、方差 1 附近，tanh/sigmoid 的输入离开饱和区，导数变大，梯度能继续往更浅的层传。
> - (iv) 不选：学习率只放大已有梯度；该层梯度已经接近 0 时，加大学习率仍几乎不更新。

<br>


Which maps are usable elementwise nonlinearities in practice?

- (i) $f(x)=-\min(2,x)$
- (ii) $f(x)=0.9x+1$
- (iii) $f(x)=\min(x,0.1x)$ for every $x$
- (iv) $f(x)=\max(x,0.1x)$ if $x\ge 0$, and $\min(x,0.1x)$ if $x<0$

哪些能当作实际使用的逐元非线性？

> [!EXAMPLE]+ (i)(iii)
>
> (ii) 是仿射函数，导数为常数，多层复合后仍是仿射。
>
> (iv) 在 $x\ge 0$ 时 $\max(x,0.1x)=x$；在 $x<0$ 时 $x<0.1x$，故 $\min(x,0.1x)=x$。两边都是 $x$，整段就是恒等映射，没有非线性。
>
> (i) 在 $x<2$ 时为 $-x$，在 $x\ge 2$ 时为 $-2$，于 $x=2$ 处弯折。
>
> (iii) 即 $x<0$ 时为 $x$、$x\ge 0$ 时为 $0.1x$，正负半轴斜率不同。因此只取 (i)(iii)。

<br>


As the gradient flows backward through a sigmoid, it always:

梯度反向经过 sigmoid 时，幅度与符号如何变化？

> [!EXAMPLE]+ 幅度变小，符号不变
>
> sigmoid 导数 $\sigma'(z)=\sigma(z)(1-\sigma(z))$。
>
> - 对任意 $z$，$\sigma(z)\in(0,1)$，所以 $\sigma'(z)>0$：反传梯度**符号**与上游相同。
> - 二次函数 $t(1-t)$ 在 $t\in(0,1)$ 上最大值为 $1/4$（在 $t=0.5$），故 $\sigma'(z)\le 1/4<1$：每过一层 sigmoid，梯度**幅度**至少乘一个小于 1 的正数，只会变小不会变大。

<br>


A step function $f(x)=1$ for $x\ge 0$ and $0$ otherwise is a usable conv nonlinearity, and Adam can train it.

- T
- F

阶跃函数可以作为卷积网络的非线性，并用 Adam 训练。

> [!EXAMPLE]+ F
>
> 阶跃函数：$x\ge 0$ 时为 1，否则为 0。几乎处处导数为 0，只有 $x=0$ 处无经典导数。
>
> 反向传播需要 $\partial f/\partial x$；除测度为零的点外梯度为 0，链式法则传不回有效信号。Adam 只改优化器，不能代替缺失的导数。因此不能当作可训练的非线性。

<br>


ELU replaces ReLU by $x$ on the positive side and $\alpha(e^{x}-1)$ on the negative side. What is the derivative, and what failure of ReLU does it avoid?

ELU 在正半轴为 $x$，在负半轴为 $\alpha(e^{x}-1)$。导数是什么，它避开了 ReLU 的哪种失败？

> [!EXAMPLE]+ 正半轴导数为 1，负半轴导数为 $\alpha e^{x}$
>
> $$
> \mathrm{ELU}'(x)=
> \begin{cases}
> 1 & x\ge 0\\
> \alpha e^{x} & x<0
> \end{cases}
> $$
>
> 负半轴导数恒正，死亡 ReLU 那种「一旦进入负区就再也没有梯度」不会出现。$\alpha\neq 1$ 时原点两侧导数不连续，仍是弯折，不是光滑到一阶的恒等。

<br>


Which methods mitigate exploding gradients: ReLU instead of sigmoid, BatchNorm, gradient clipping, residual connections?

缓解梯度爆炸，可以靠哪些办法：ReLU 换掉 sigmoid、BatchNorm、梯度裁剪、残差连接？

> [!EXAMPLE]+ BatchNorm、梯度裁剪、残差连接
>
> ReLU 的导数只有 0 和 1，能减轻消失，却不给梯度加上界。权重大于 1 时，连乘仍会爆炸。
>
> BatchNorm 把层输入的尺度拉回稳定区间。裁剪直接限制梯度范数。残差给梯度一条乘数为 1 的支路，连乘不再是唯一通路。
>
> 另一道只把「用 SGD、过采样少数类、加大 batch、梯度裁剪」放在一起的题，答案只有梯度裁剪。前三项都不约束梯度幅度。

<br>


## 正则、Dropout 与优化器

Which reduce overfitting?

- (i) Data augmentation. (ii) Dropout. (iii) BatchNorm. (iv) Adam instead of SGD.

哪些能减轻过拟合？（Fall 2020）

> [!EXAMPLE]+ (i)(ii)(iii)
>
> 过拟合是模型在训练集上记噪声、验证变差。需要的是**限制有效容量**或**增加有效数据**。
>
> - (i) 选：数据增强（翻转、裁剪等）让模型见到更多变化，减轻死记训练集。
> - (ii) 选：Dropout 训练时随机关掉神经元，等价于平均许多子网络，抑制共适应。
> - (iii) 选：BatchNorm 的小批量均值/方差有噪声，对下一层是弱正则（不是主因，但卷面认）。
> - (iv) 不选：Adam 与 SGD 只是怎么走梯度，不缩小假设空间，不直接当正则。

<br>


Training error on a dog-species classifier is high. Which changes are promising: a bigger network, more training data, a larger $\lambda$, a larger dropout keep probability?

某分类器的训练误差很高。加大网络、增加数据、加大 $\lambda$、提高 Dropout 保留概率，哪些值得试？

> [!EXAMPLE]+ 加大网络，并提高保留概率
>
> **训练误差高**说明模型连训练集都拟合不好，主因通常是**高偏差**（容量不够或训练不足），不是验证集上的过拟合。
>
> - 加大网络：增加容量，直接针对高偏差。
> - 提高 Dropout **保留概率**（keep prob）：少丢神经元，正则变弱，更利于拟合训练集。
> - 增加训练数据：对高偏差帮助有限（已经欠拟合时，数据多也不等于能表达更复杂函数）。
> - 加大 $\lambda$：L2 更强，权重更小，往往更难拟合训练集，偏差更糟。

<br>


Dropout on a net with $N$ units, each dropped independently. How many distinct sub-networks can appear?

- (i) $\lfloor Np\rfloor$ (ii) $(\lfloor Np\rfloor)^N$ (iii) $2^{\lfloor Np\rfloor}$ (iv) $2^N$

$N$ 个单元各自可保留或丢弃。不同子网络共有多少个？（Fall 2022）

> [!EXAMPLE]+ $2^N$
>
> 第 $i$ 个单元独立地「保留」或「丢弃」，两种选择。$N$ 个单元各两种，组合数是 $2\times 2\times\cdots\times 2=2^N$。
>
> 丢弃概率 $p$ 只决定**随机抽到**哪一种子网络的概率，不改变一共有多少种。选项 (i)(ii)(iii) 把 $p$ 或 $\lfloor Np\rfloor$ 混进计数里，都不对。

<br>


At test time, inverted dropout with drop probability $p$ will:

- (i) multiply by $p$ (ii) drop units with probability $p$ (iii) divide by $p$ (iv) none of these

Inverted dropout 在测试时做什么？

> [!EXAMPLE]+ 都不做
>
> 记**丢弃概率**为 $p$，**保留概率**为 $1-p$。Inverted dropout 约定：
>
> - **训练**：先按概率 $p$ 置零，再对保留下来的激活除以 $(1-p)$，使期望与「不 dropout」一致。
> - **测试**：不再随机丢弃，也**不再**乘除任何因子，直接用完整网络前向。
>
> 因此测试时不是 (i) 乘 $p$、(ii) 再丢单元、(iii) 除以 $p$；正确是 (iv) 以上都不做。

<br>


Which statements about dropout are true?

- (i) Dropout makes the trained weights sparse.
- (ii) At test time, dropout is applied with the inverted keep probability.
- (iii) A larger keep probability regularizes that layer more strongly.
- (iv) None of the above.

关于 Dropout，哪些成立？（Fall 2020）

> [!EXAMPLE]+ 都不成立
>
> - (i) 错：Dropout 是训练时随机置零，**学出来的权重通常不稀疏**；稀疏靠 L1。
> - (ii) 错：inverted 的缩放只在**训练**；测试时关闭 dropout，不做「按保留概率缩放」那一套。
> - (iii) 错：**保留概率越大**，丢的神经元越少，正则**越弱**，不是越强。
> - (iv) 对：上面三条都不对，故选「以上都不成立」。

<br>


Training used dropout but forgot to divide activations by the keep probability $p$. Dropout is then switched off at test time. How is the scale repaired?

训练时做了 Dropout，却忘了除以保留概率 $p$，测试时又关掉 Dropout。尺度怎么补？

> [!EXAMPLE]+ 测试时把该层输出乘 $p$
>
> 本题 $p$ 指**保留概率**（keep probability）。标准 dropout 训练时：以概率 $1-p$ 丢弃，**保留**的激活不再额外放大。
>
> - 训练期望：每个单元输出期望约为「无 dropout 时」的 $p$ 倍（因为只有比例 $p$ 的单元平均在起作用）。
> - 测试：不丢弃，输出相当于放大了约 $1/p$ 倍，与训练尺度不一致。
> - **修复**：测试时把该层输出**乘以 $p$**，与训练期望对齐。
>
> Inverted dropout 则在训练时除以 $(1-p_{\mathrm{drop}})$，测试仍什么都不做；不要与「忘记除以保留概率 $p$」这一题混用符号。

<br>


Which are true of momentum?

- (i) It has no hyperparameters.
- (ii) A running average of gradients helps move through ravines and poor local regions.
- (iii) A network with $n$ parameters needs $\Theta(n)$ extra stored values.

关于 momentum，哪些成立？（Spring 2021，推导）

> [!EXAMPLE]+ (ii)(iii)
>
> - (i) 错：momentum 至少有学习率 $\alpha$ 和动量系数 $\beta$（有的实现还有 Nesterov 开关），不是「没有超参数」。
> - (ii) 对：速度 $v$ 是梯度的指数滑动平均，沿 ravine 方向累积，有助于穿过狭长谷地和较差局部区域。
> - (iii) 对：要存与参数同形状的 $v$，$n$ 个参数则额外 $\Theta(n)$ 个浮点数。

<br>


Match the one-line descriptions to SGD, momentum, RMSProp, and Adam. Which line is the benefit of momentum?

- (i) A simple update and few hyperparameters.
- (ii) A running average of gradients helps leave poor local regions.
- (iii) The learning rate is scaled separately in each coordinate.
- (iv) Several of those mechanisms are combined.

下面四句分别在说 SGD、momentum、RMSProp、Adam 中的哪一个？momentum 对应哪一句？

> [!EXAMPLE]+ 只对应 (ii)
>
> (i) 是普通 SGD。(iii) 是 RMSProp 用二阶矩缩放每个坐标。(iv) 是 Adam，把一阶矩和二阶矩合在一起。momentum 的收益是沿持续方向累积速度，从而离开较差的局部区域。它仍然有学习率与 $\beta$，也要存一份与参数同形状的速度。

<br>


Early stopping:

- (i) can reduce the need to hand-tune the epoch budget
- (ii) increases variance
- (iii) triggers when validation accuracy reaches a trough
- (iv) can stop earlier than the training-loss minimum

关于 early stopping，哪些成立？

> [!EXAMPLE]+ (i)(iv)
>
> - (i) 对：不必事先固定 epoch 总数，验证集不再变好就停，相当于自动选训练长度。
> - (ii) 错：early stopping 在验证最优附近停，相当于限制训练，一般**降低方差**（减轻过拟合），不是增加方差。
> - (iii) 错：通常在验证准确率/损失**达到峰值后**、连续 $k$ 轮无改善时停，不是「掉到谷底（trough）」才停。
> - (iv) 对：训练损失最低往往出现在更晚的 epoch；早停常在训练损失还在缓慢下降时就停，因为验证已开始变差。

<br>


Backpropagation:

- (i) costs $O(n^2)$ for $n$ parameters
- (ii) requires a directed acyclic graph, unless recurrence is unrolled
- (iii) Adam’s $\beta$ is updated by backpropagation
- (iv) BatchNorm’s $\gamma$ is updated by backpropagation

关于反向传播，哪些成立？

> [!EXAMPLE]+ (ii)(iv)
>
> - (i) 错：一次前向+反向，每个参数、每条边常数工作量，总复杂度 $O(n)$（参数量 $n$），不是 $O(n^2)$。
> - (ii) 对：标准反向要求计算图是 DAG；RNN 需按时间展开成 DAG 再反传。
> - (iii) 错：Adam 的 $\beta_1,\beta_2$ 是人工设定的超参数，不参与反向传播更新。
> - (iv) 对：BatchNorm 的 $\gamma,\beta$ 与权重一样是可学习参数，损失对它们的梯度由反向传播给出。

<br>


On ImageNet, with wall-clock time as the measure, which can be true?

- (i) SGD can finish sooner than full-batch GD.
- (ii) Mini-batch can finish sooner than SGD.
- (iii) Mini-batch can finish sooner than full-batch.
- (iv) Full-batch can finish sooner than SGD.

以墙上时间为准，哪些比较可能成立？（Fall 2020 官方取前三项）

> [!EXAMPLE]+ (i)(ii)(iii)
>
> 比较的是**达到相近效果所需的墙上时间**，不是单步迭代数。
>
> - (i) 对：mini-batch SGD 每步只扫一部分数据，步数可能多，但每步便宜，总时钟常快于每步扫全数据的 full-batch。
> - (ii) 对：batch size $>1$ 能用矩阵运算，每步比纯 online（batch size 1）更省时间。
> - (iii) 对：结合 (i)(ii)，mini-batch 相对 full-batch 几乎总是时钟上更划算（大数据上 full-batch 一步极慢）。
> - (iv) 一般不选：full-batch 单步最慢，很难在总时间上 beat mini-batch SGD。

<br>


## 尺度、初始化与指标

Ignore biases and numeric overflow. A conv net ends in softmax. Which leave the predicted class unchanged?

- (i) Multiply every weight by 10 at inference.
- (ii) Multiply every weight by 10 and continue training; convergence is unaffected.
- (iii) Multiply the input by 10 at inference.
- (iv) Subtract the per-channel mean of the input at inference.

忽略偏置与溢出。哪些操作不改变 argmax 类别？（Spring 2021，推导）

> [!EXAMPLE]+ (i)(iii)
>
> 忽略偏置时，线性层、卷积、ReLU、max-pool 对输入或权重乘**正数** $c$ 都是齐次的（ReLU 与 max 只关心大小关系）。
>
> - (i) 推理时全体权重 $\times 10$：整条链路的 logits 全体 $\times 10$，softmax 单调变换下 **argmax 不变**。
> - (iii) 推理时输入 $\times 10$：同样 logits 全体 $\times 10$，**argmax 不变**。
> - (ii) 继续训练：动力学、有效学习率都变，**不保证**收敛行为不变，且不是「推理时类别不变」的题意。
> - (iv) 减通道均值：仿射变换，不是整体正缩放，logits 各维改变不同，**类别可能变**。

<br>


A softmax classifier already has $100\%$ accuracy. Replace $W$ by $3W$, with no regularization. Accuracy stays the same and cross-entropy falls.

- T
- F

已达 $100\%$ 准确率的 softmax，把 $W$ 换成 $3W$。准确率不变，交叉熵下降。

> [!EXAMPLE]+ F
>
> 无偏置、或偏置与 $W$ 一起乘 3 时，logits 同乘正数，argmax 不变，分布更尖，交叉熵下降。
>
> 只替换权重矩阵、偏置不动时，新的预激活是 $3Wx+b$，不是 $3(Wx+b)$。反例：两类、$x=1$、$W=(1,0)$、$b=(0,1.5)$。原 logits 为 $(1,1.5)$，第二类胜出；换成 $3W$ 后 logits 为 $(3,1.5)$，第一类胜出。准确率可以下降。
>
> 另有 L2 正则时，权重量级变大本身进入损失，交叉熵即使下降，总目标也不一定下降。

<br>


For a linear classifier, zero initialization makes every entry of the final weight matrix stay identical. A zero image into a conv net yields uniform class probabilities. BatchNorm, added because validation accuracy lags training accuracy, is likely to shrink that gap. $\mathrm{softmax}(cy)=\mathrm{softmax}(y)$ for every real $c$. Which of these hold?

下列哪些成立：线性分类器从全 0 开始后，最终权重的每个分量始终相同；全 0 图像送入卷积网络后类别概率均匀；验证集明显差于训练集时，加上 BatchNorm 后这个差距多半会缩小；对任意实数 $c$，$\mathrm{softmax}(cy)=\mathrm{softmax}(y)$？

> [!EXAMPLE]+ 只有「加上 BatchNorm 后差距多半缩小」成立
>
> 全 0 初始化的线性分类器，每个权重的梯度正比于对应特征。特征不同，更新就不同，分量不会锁在同一个数上。
>
> 全 0 图像只说明卷积项为 0。偏置、以及 BatchNorm 的 $\beta$，仍可让 logits 互不相等，softmax 不必均匀。
>
> BatchNorm 的小批量噪声有弱正则作用，训练–验证差距常常缩小。这不是保证，只是在「验证明显更差」时值得试的方向。
>
> $\mathrm{softmax}(cy)$ 与 $\mathrm{softmax}(y)$ 仅在 $c=1$ 时相同。$c>0$ 且 $c\neq 1$ 时 argmax 不变，但概率向量变尖或变平。$c<0$ 时排序反过来。$c=0$ 时变成均匀分布。

<br>


For data normalization, mean and standard deviation are computed on the whole dataset before the train/dev/test split.

- T
- F

均值和标准差应在划分训练、验证、测试之前，用全部数据计算。

> [!EXAMPLE]+ F
>
> 归一化的均值、标准差是**从数据估计的超参数**，必须在**划分之后、只用训练集**估计。
>
> 若划分前用全数据算均值方差，验证集和测试集的数值已经参与估计，等于把「未来」分布信息泄漏进预处理，验证/测试指标会乐观。正确流程：先 split，再在训练集上 fit 标准化，同一变换应用到 val/test。

<br>


Weights of a 2-layer MLP with hidden sigmoids are drawn from $\mathcal N(10,1)$. From epoch 1, some first-layer weights barely move. Which help?

- (i) Draw weights from $\mathcal N(0,1)$.
- (ii) Add more hidden layers.
- (iii) Switch the hidden activation to tanh.
- (iv) Switch the hidden activation to leaky ReLU.

隐藏层用 sigmoid，权重来自 $\mathcal N(10,1)$，第一层几乎不更新。哪些能缓解？（Fall 2022）

> [!EXAMPLE]+ (i)(iv)
>
> 权重 $\mathcal N(10,1)$ 使预激活偏大，隐藏 sigmoid 进入饱和区，$\sigma'\approx 0$，第一层几乎不更新。
>
> - (i) 选：改用小方差初始化（如 $\mathcal N(0,1)$ 或 Xavier），预激活落在梯度较大的区域。
> - (ii) 不选：加深网络只增加连乘小导数的层数，饱和通常更严重。
> - (iii) 不选：tanh 也会饱和，不能解决「已经太大」的问题。
> - (iv) 选：Leaky ReLU 负半轴有常数斜率，不饱和，梯度能回传。

<br>


Which statement about initialization is false: Xavier keeps output variances unchanged throughout training; a positive constant does not break symmetry; different activations can want different initializations; sampling each weight from $\{-1,1\}$ can break symmetry?

关于初始化，哪一句是错的：Xavier 使各层输出方差在整个训练中保持不变；正常数不能打破对称；不同激活可以配不同初始化；从 $\{-1,1\}$ 里抽样可以打破对称？

> [!EXAMPLE]+ 「整个训练中方差不变」是错的
>
> Xavier 只在初始化的那一刻把各层方差调到合适的量级。训练开始后，激活分布会变。正常数初始化让同一层单元收到相同梯度，对称不会自己破掉。从 $\{-1,1\}$ 独立抽样时，单元的初始权重不同，对称被打破。

<br>


A binary metric should combine precision and recall. Which is a reasonable single score?

- (i) $|\mathrm{precision}-\mathrm{recall}|$
- (ii) $\mathrm{recall}/\mathrm{precision}$
- (iii) $\mathrm{precision}\times\mathrm{recall}$
- (iv) $\max(\mathrm{precision},\mathrm{recall})$

想用一个数同时看 precision 与 recall。哪个还算合理？（Fall 2022）

> [!EXAMPLE]+ 只选乘积
>
> 差的绝对值奖励两者分开。比值可以靠拉高 recall、牺牲 precision 变大。最大值分不清 $(0.9,0.4)$ 与 $(0.9,0.7)$。乘积与几何平均同方向，两者都低时分数才低。F1 是更常见的调和形式，本题只问备选里哪一个说得通。

<br>


# 浅层

来源：Practice Midterm 1，创业公司融资二分类。输入特征 $n_x=300$，标签为第一年是否融资。

Logistic regression outputs $\hat y=\sigma(w^\top x+b)$, and predicts 1 when $\hat y\ge 0.5$. How many weights does it have, and what is the shape of $b$ and of $\hat y$?

逻辑回归有多少个权重？$b$ 与 $\hat y$ 的维度是什么？

> [!EXAMPLE]+ 300 个权重，$b$ 与 $\hat y$ 都是标量
>
> 本节输入特征维是 $n_x=300$。逻辑回归给每个特征一个权重，所以 $w$ 有 $300$ 个分量。
>
> 偏置 $b$ 是一个数，所有样本共用。$\hat y^{(i)}=\sigma(w^\top x^{(i)}+b)$ 也是一个数：每个样本一个概率，不是一个向量。

<br>


A one-hidden-layer net has 80 hidden units, sigmoid output, and biases. How many weights and biases?

单隐藏层 80 个单元。权重与偏置各有多少？

> [!EXAMPLE]+ 权重 24080，偏置 81
>
> 输入仍是本节的 $n_x=300$，隐藏层 $80$ 个单元，输出是一个 sigmoid，所以输出宽度是 $1$。
>
> - 第一层权重：每个隐藏单元连到全部 $300$ 个输入，$80$ 个隐藏单元，共 $300\times 80=24000$。
> - 第二层权重：唯一的输出单元连到全部 $80$ 个隐藏单元，共 $80\times 1=80$。
> - 权重合计 $24000+80=24080$。
>
> 偏置是每层每个单元一个：隐藏层 $80$ 个，输出层 $1$ 个，共 $81$。参数总数 $24080+81=24161$。

<br>


A one-hidden-layer net can approximate continuous functions under standard conditions. Why use more layers?

单隐藏层已能逼近连续函数。多层的理由是什么？

> [!EXAMPLE]+ 同样的函数可以用更少的单元、更少的参数表示
>
> 单隐藏层在宽度足够大时**可以逼近**很多连续函数（普适逼近），但可能需要**极宽**的一层。
>
> 加深网络往往用**更少的总神经元/参数**表示同一函数（组合结构、层次特征）。官方给分点是**参数效率**，不是「多层才能表示」——单层已经能表示，只是不省参数。

<br>


Hidden activations are omitted, so each layer is affine and the output is still a sigmoid. Show that the net is a logistic regression.

中间没有非线性，输出仍是 sigmoid。证明它等价于逻辑回归。

> [!EXAMPLE]+ 仿射复合仍是仿射
>
> 没有非线性时，每层都是仿射，复合仍是仿射。
>
> 1. 隐藏预激活：$z^{[1]}=W^{[1]}x+b^{[1]}$（矩阵乘加）。
> 2. 若省略激活，输出层输入是 $z^{[1]}$ 本身，则 $z^{[2]}=W^{[2]}z^{[1]}+b^{[2]}=W^{[2]}W^{[1]}x+W^{[2]}b^{[1]}+b^{[2]}$。
> 3. 令 $W'=W^{[2]}W^{[1]}$、$b'=W^{[2]}b^{[1]}+b^{[2]}$，则 $z^{[2]}=W'x+b'$，再 $\hat y=\sigma(z^{[2]})=\sigma(w'^\top x+b')$（列向量时 $w'=(W')^\top$）。
>
> 这就是带一个等价权重向量 $w'$ 和偏置 $b'$ 的逻辑回归。

<br>


The output is changed to $\hat y=\sigma(\mathrm{ReLU}(z))$. What breaks?

在 sigmoid 之前再套一层 ReLU。分类会出什么问题？

> [!EXAMPLE]+ 所有样本都会被判成正类
>
> 1. 原输出层预激活为 $z$，现改为先算 $u=\mathrm{ReLU}(z)\ge 0$。
> 2. 预测 $\hat y=\sigma(u)$。因为 $u\ge 0$，sigmoid 在 $[0,\infty)$ 上取值 $\ge 0.5$。
> 3. 阈值 $0.5$ 时 $\hat y\ge 0.5$ 恒成立，**每个样本都会判为正类**，负类召回为 0。

<br>


Final two layers of a binary net are ReLU then sigmoid. What happens to the predictions?

最后两层是 ReLU 再接 sigmoid。预测落在哪里？（Fall 2019）

> [!EXAMPLE]+ 全部至少为 $0.5$
>
> 倒数第二层 ReLU ⇒ 进入 sigmoid 的值 $u\ge 0$ ⇒ $\hat y=\sigma(u)\ge 0.5$。
>
> 与上一题同一机制：阈值 $0.5$ 时**只会预测正类**，无法输出低于 $0.5$ 的负类概率。

<br>


# 损失

## 二分类

西瓜甜度二分类（Practice 1）。标签 $y\in\{0,1\}$。

Why is mean squared error a poor training loss when $\hat y\in(0,1)$?

预测被限制在 $(0,1)$ 时，为什么不宜用均方误差做训练损失？

> [!EXAMPLE]+ 错误再自信，MSE 也被压在 1 以内
>
> 例：$y=0$ 时，$\hat y=0.99$ 错得很离谱，MSE 只有 $(0.99)^2\approx 0.98$，**梯度仍有限**。
>
> 二元交叉熵 $-\log(1-0.99)\to\infty$，梯度大，强烈把 $\hat y$ 推离错误标签。
>
> 此外 $\hat y\in(0,1)$ 时 MSE 天然 $\le 1$，**饱和区**里错误预测的惩罚不够尖，不适合概率校准。

<br>


Write binary cross-entropy for one example. Then evaluate it at $y=0$, $\hat y=0.9$.

写出单样本二元交叉熵，并计算 $y=0,\hat y=0.9$。

> [!EXAMPLE]+ $-\log 0.1$
>
> 单样本二元交叉熵：
> $$
> L=-\big(y\log\hat y+(1-y)\log(1-\hat y)\big).
> $$
>
> 代入 $y=0$、$\hat y=0.9$：
> - $y=0$ 时第一项 $y\log\hat y=0$；
> - 第二项 $-(1-y)\log(1-\hat y)=-\log(1-0.9)=-\log 0.1\approx 2.30$（自然对数）。

<br>


$y^\top=(1,0,0)$, $\hat y^\top=(0.1,0.2,0.7)$. Write the average cost $J$.

三个样本的平均二元交叉熵是什么？

> [!EXAMPLE]+ $J=-\frac13(\log 0.1+\log 0.8+\log 0.3)$
>
> 三个样本按顺序配对 $(y,\hat y)=(1,0.1)$、$(0,0.2)$、$(0,0.7)$。二元交叉熵里，$y=1$ 只留下 $-\log\hat y$，$y=0$ 只留下 $-\log(1-\hat y)$。
>
> - 第 1 个：$y=1$，损失 $-\log 0.1$。
> - 第 2 个：$y=0$，$\hat y=0.2$，损失 $-\log(1-0.2)=-\log 0.8$。
> - 第 3 个：$y=0$，$\hat y=0.7$，损失 $-\log(1-0.7)=-\log 0.3$。
>
> 平均是把这三项相加再除以样本数 $3$，没有权重惩罚。

<br>


L2 penalty on one weight is $\lambda W^2$. Write the gradient step. How do the weights compare with an unregularized run, and how does L1 differ?

惩罚项为 $\lambda W^2$。写出更新，并比较 L1。

> [!EXAMPLE]+ 权重被额外拉向 0
>
> 总损失 $J_{\mathrm{tot}}=J_{\mathrm{BCE}}+\lambda W^2$。对 $W$ 求导：
> $$
> \frac{\partial J_{\mathrm{tot}}}{\partial W}=\frac{\partial J_{\mathrm{BCE}}}{\partial W}+2\lambda W.
> $$
> 梯度下降 $W\leftarrow W-\alpha\frac{\partial J_{\mathrm{tot}}}{\partial W}$ 得
> $$
> W\leftarrow (1-2\alpha\lambda)W-\alpha\frac{\partial J_{\mathrm{BCE}}}{\partial W}.
> $$
> 因子 $(1-2\alpha\lambda)$ 每步把 $W$ **额外缩小**一点。若惩罚写成 $\frac{\lambda}{2}W^2$，则 $\partial/\partial W$ 里是 $\lambda W$，更新变为 $(1-\alpha\lambda)W-\cdots$。
>
> **L1** 加 $\lambda|W|$：更新含符号项，可把坐标**精确压到 0**；L2 只连续缩小，一般不到 exact 0。

<br>


L1-regularized objective $\tilde J=J+\alpha\lVert w\rVert_1$. Under the diagonal-Hessian approximation $\hat J(w)=J(w^\star)+\frac12(w-w^\star)^\top H(w-w^\star)$ with $H_{ii}>0$, what $w_i$ sets the gradient to zero? How does that differ from L2 for large and for small weights?

在对角 Hessian 近似下，L1 正则的驻点是什么？大权重和小权重与 L2 有何不同？

> [!EXAMPLE]+ 软阈值：$w_i=\mathrm{sign}(w^\star_i)\max\big(|w^\star_i|-\alpha/H_{ii},\,0\big)$
>
> 近似目标对 $w_i$ 的导数是 $H_{ii}(w_i-w^\star_i)+\alpha\,\xi_i$，其中 $\xi_i=\mathrm{sign}(w_i)$，而 $w_i=0$ 时 $\xi_i$ 可取 $[-1,1]$ 中的任意值。
>
> $|w^\star_i|\le \alpha/H_{ii}$ 时，取 $w_i=0$ 就能让次梯度含 0。否则 $w_i=w^\star_i-(\alpha/H_{ii})\mathrm{sign}(w^\star_i)$。
>
> L1 对大权重只减去一个与 $|w|$ 无关的常数，对小权重可以直接收到 0。L2 的对应收缩是 $w_i=H_{ii}w^\star_i/(H_{ii}+\lambda)$，大权重按比例变小，小权重变小但一般不是精确 0。

<br>


## 多任务与多标签

One image must predict sweetness (binary) and weight in pounds (real). What is the label shape, and what loss is appropriate?

同一张图同时预测甜度与重量。标签维度与损失如何取？

> [!EXAMPLE]+ 二维标签；甜度用交叉熵，重量用平方误差
>
> - 标签形状：甜度是 $\{0,1\}$，重量是实数磅值，可写成 $y=(y_{\mathrm{sweet}},y_{\mathrm{wt}})^\top\in\mathbb R^2$（或甜度单独 one-hot）。
> - 损失：甜度用二元交叉熵（或 sigmoid + BCE）；重量用 $(\hat w_{\mathrm{wt}}-y_{\mathrm{wt}})^2$ 或 Huber。
> - 总损失常写成 $L=\lambda_1 L_{\mathrm{sweet}}+\lambda_2 L_{\mathrm{wt}}$。重量数值尺度大，若不加权，平方项梯度会 dominate 甜度分支。

<br>


电影海报四分类：comedy, horror, action, romance（Fall 2020）。人类水平估计为 $95\%$。

What does a $95\%$ human-level accuracy say about the Bayes error?

人类准确率 $95\%$ 对贝叶斯误差说明什么？

> [!EXAMPLE]+ 贝叶斯误差不超过 $5\%$
>
> - 人类准确率 $95\%$ ⇒ 人类错误率 $5\%$。
> - **贝叶斯误差** $E_{\mathrm{Bayes}}$：在数据分布上，**最优**分类器能达到的最小错误率。
> - 人类不是最优，故 $E_{\mathrm{Bayes}}\le 5\%$（人能做到的误差是贝叶斯误差的上界估计，不是下界）。
> - 含义：任务「理论上」至少还有 $0\%\sim 5\%$ 的错误空间；训练目标不应假设标签噪声为 0 时还能超过 $95\%$ 太多而不怀疑过拟合/泄漏。

<br>


A horror poster has one-hot label on class 2. The model outputs $(0.1,0.4,0.3,0.2)$, then later $(0,0.4,0,0.6)$. What is the cross-entropy each time?

正确类概率两次都是 $0.4$。交叉熵各是多少？

> [!EXAMPLE]+ 两次都是 $-\log 0.4$
>
> 四分类 one-hot，正确类是 class 2（horror）。交叉熵 $L=-\log \hat y_2$，只依赖**正确类概率**。
>
> - 第一次 $\hat y=(0.1,0.4,0.3,0.2)$：$L=-\log 0.4$。
> - 第二次 $\hat y=(0,0.4,0,0.6)$：正确类仍是 $0.4$，$L=-\log 0.4$。
>
> 错类之间如何重新分配剩余 $0.6$ 的质量，不改变 $-\log \hat y_2$。

<br>


Training accuracy is $90\%$, below the $95\%$ human level. What is the diagnosis?

训练集只有 $90\%$。偏差还是方差？

> [!EXAMPLE]+ 高偏差
>
> 对比对象：人类约 $95\%$ ⇒ 可避免误差下界约 $5\%$；训练只有 $90\%$ ⇒ 训练误差 $10\%$。
>
> - 训练误差远高于「人类/贝叶斯」可达水平 ⇒ **高偏差**（欠拟合）：模型或训练不足以拟合训练分布。
> - 若训练 $99\%$、验证 $90\%$ 才是高方差。
> - 对策：加容量、训更久、减正则；而不是先加 $\lambda$ 或更强 Dropout。

<br>


After the softmax fits the training set perfectly, why is cross-entropy still not zero?

训练准确率已经是 $100\%$，softmax 交叉熵为什么到不了 0？

> [!EXAMPLE]+ 每个指数都为正，正确类概率严格小于 1
>
> Softmax：$p_c=\dfrac{e^{z_c}}{\sum_j e^{z_j}}$。训练「分对类」只要求 $c=\arg\max z$，仍可有 $z_j$ 有限且 $j\neq c$ 时 $e^{z_j}>0$。
>
> 于是 $p_c=\dfrac{e^{z_c}}{e^{z_c}+\sum_{j\neq c}e^{z_j}}<1$，故
> $$
> L=-\log p_c=-z_c+\log\sum_j e^{z_j}>0.
> $$
>
> 要让 $p_c=1$ 需所有 $j\neq c$ 的 $z_j\to-\infty$，有限参数/有限 logits 做不到；损失只能趋近 0，不能精确为 0。

<br>


Some labels may be wrong. Between L1 and L2, which is preferable before cleaning labels, and why does the posted solution pick L1?

怀疑部分标签是错的。官方解答倾向哪一种正则？

> [!EXAMPLE]+ L1
>
> 错标签样本产生大残差时：
>
> - **L2**：惩罚 $\lambda\|w\|^2$ 与残差平方一起，为降低总损失，模型倾向**整体放大权重**去拟合坏点（大权重 + 大残差仍可能划算）。
> - **L1**：$\lambda\|w\|_1$ 鼓励许多坐标为 0，只有部分坐标被坏样本拉偏，**妥协更局部**。
>
> 清洗标签前官方倾向 **L1**（该题给分点）。

<br>


Posters may belong to several genres at once. Why do softmax and categorical cross-entropy fail, and what should replace them?

一部电影可以同时属于多类。为什么不能继续用 softmax？

> [!EXAMPLE]+ softmax 强迫各类概率相加为 1，多标签会被重复惩罚
>
> 若 $K$ 个类都正确，而模型均匀输出 $1/n_y$，损失约为 $K\log n_y$，远大于单标签样本。
>
> 标签改为 multi-hot。输出改为逐类 sigmoid，损失改为各类二元交叉熵的平均。这样每一类是独立的是/否，不再竞争同一个概率质量。

> [!NOTE]+ 下界
>
> 在仍使用 softmax、且有 $K$ 个正确类时，Jensen 不等式给出 $L\ge K\log K$。均匀分给这 $K$ 类、其余类概率为 0 的极限才取等。有限 logits 到不了这个极限。

<br>


## 加权对数与 hinge

长颈鹿检测（Fall 2019）：正类约 2000，负类约 200。损失写成

$$
L^{(i)}=\alpha\, y^{(i)}\log\hat y^{(i)}+\beta\,(1-y^{(i)})\log(1-\hat y^{(i)}),\qquad
J=-\frac1m\sum_i L^{(i)}.
$$

Why introduce $\alpha$ and $\beta$? Give a reasonable pair.

$\alpha$ 与 $\beta$ 在调节什么？一组说得通的数值是什么？

> [!EXAMPLE]+ 稀有类应得到更大的损失权重
>
> 正类约 $2000$，负类约 $200$，个数比是 $2000/200=10$，负类大约只有正类的十分之一。损失里 $\alpha$ 乘在正类项上，$\beta$ 乘在负类项上。要让稀有的负类不被淹没，应使 $\beta/\alpha\approx 10$。一组具体取值是 $\alpha=0.1$、$\beta=1$。任何同号且比值约为 $10$ 的正数都可以。$\alpha$ 或 $\beta$ 取负时，最小化 $J$ 会把该类错误往反方向推。

<br>


A one-hidden-layer net uses

$$
\begin{aligned}
z^{[1]}&=W^{[1]}x+b^{[1]},&
h&=\mathrm{ReLU}(z^{[1]}),\\
z^{[2]}&=W^{[2]}h+b^{[2]},&
\hat y&=\sigma(z^{[2]}),\\
L&=\sum_{i=1}^{k}\max(0,\,1-y_i\hat y_i).
\end{aligned}
$$

$y$ is a vector of zeros and ones. Derive $\partial L/\partial \hat y$, $\partial L/\partial z^{[2]}$, and the parameter gradients. (Fall 2022)

对上述 hinge 形式的损失做完整反向传播。

> [!EXAMPLE]+ 因为 $y_i\hat y_i\in[0,1]$，hinge 始终处于激活段
>
> 损失项 $\max(0,1-y_i\hat y_i)$。已知 $y_i\in\{0,1\}$，$\hat y_i=\sigma(z)\in(0,1)$。
>
> - 若 $y_i=0$：$y_i\hat y_i=0$，项为 $\max(0,1)=1$。
> - 若 $y_i=1$：$y_i\hat y_i=\hat y_i\in(0,1)$，故 $1-y_i\hat y_i\in(0,1)$，仍 $\ge 0$。
>
> 因此恒走 $\max$ 的**第一支** $1-y_i\hat y_i$，不会出现 $\max(0,\cdot)$ 的「零支」导致梯度为 0 的退化（除非单独讨论边界 $\hat y=1$）。
>
> $$
> \frac{\partial L}{\partial \hat y_i}=-y_i,\qquad
> \frac{\partial L}{\partial \hat y}=-y.
> $$
>
> sigmoid 的导数是 $\hat y\odot(1-\hat y)$：
>
> $$
> \frac{\partial L}{\partial z^{[2]}}
> =(-y)\odot \hat y\odot(1-\hat y).
> $$
>
> 记该量为 $\delta^{[2]}$。则
>
> $$
> \frac{\partial L}{\partial W^{[2]}}=\delta^{[2]} h^\top,\quad
> \frac{\partial L}{\partial b^{[2]}}=\delta^{[2]},\quad
> \frac{\partial L}{\partial h}=(W^{[2]})^\top\delta^{[2]}.
> $$
>
> ReLU 的局部导数 $\delta_i=\mathbf 1[z^{[1]}_i>0]$：
>
> $$
> \frac{\partial L}{\partial z^{[1]}}
> =\frac{\partial L}{\partial h}\odot \delta,\qquad
> \frac{\partial L}{\partial W^{[1]}}=\frac{\partial L}{\partial z^{[1]}}\,x^\top,\qquad
> \frac{\partial L}{\partial b^{[1]}}=\frac{\partial L}{\partial z^{[1]}}.
> $$

<br>


# 优化

## 批量

乐器分类，数据按乐器种类整段存放（Practice 1）。

Full-batch gradient descent is used, and the training set is never shuffled. Shuffling the training set fixes a poor training loss.

- T
- F

全量梯度下降不打乱数据。打乱能修好训练损失。

> [!EXAMPLE]+ F
>
> Full-batch 每步梯度 $g=\frac1m\sum_{i=1}^m \nabla L^{(i)}$，**加法可交换**，与样本在求和里的顺序无关。
>
> 打乱训练集只改变 mini-batch SGD 或 online SGD 里「哪些样本在同一批」，不改变 full-batch 的 $g$。因此打乱不能修复 full-batch 下已经很差的训练损失（问题在模型/学习率/实现，不在顺序）。

<br>


One reason to prefer mini-batch over batch size 1, and one reason to prefer it over full-batch.

mini-batch 相对 SGD 与全量梯度下降各有什么好处？

> [!EXAMPLE]+ 向量化，以及更快的第一步
>
> **相对 batch size 1（纯 SGD）**：
> - 一次前向/反向处理 $B$ 个样本，用矩阵乘，GPU 利用率高；
> - 梯度是 $B$ 个样本的平均，方差比单样本小，曲线更平滑。
>
> **相对 full-batch**：
> - 不必每步扫完 $m$ 个样本，单步更快，大数据可训练；
> - 小批量梯度带噪声，有时有助于逃离尖锐极小值/鞍点。

<br>


Three loss curves, from smoothest to noisiest, correspond to which algorithms?

损失曲线从最平滑到最抖，对应哪种梯度下降？

> [!EXAMPLE]+ 全量、mini-batch、batch size 1
>
> 每步用的梯度估计方差大致随 batch size $B$ 按 $1/B$ 缩小。
>
> - **Full-batch**：每步是真实全量梯度，曲线最平滑。
> - **Mini-batch**：中等噪声。
> - **Batch size 1**：每步只看一个样本，估计最噪，曲线最抖。
>
> 从平滑到最抖的顺序：全量 → mini-batch → batch size 1。

<br>


A mini-batch gradient has lower bias, on average, when the batch is drawn i.i.d.

- T
- F

mini-batch 梯度在样本独立同分布时，平均而言偏差更低。

> [!EXAMPLE]+ T
>
> 全量梯度 $g=\mathbb E[\nabla L]$（对数据分布）。大小为 $B$ 的 mini-batch 平均梯度 $\hat g$ 满足 $\mathbb E[\hat g]=g$：在 **i.i.d.** 抽样下**无偏**。
>
> 若 batch 内样本相关（例如按类别整段取 batch），$\mathbb E[\hat g]\neq g$，估计有**偏差**，更新系统性偏离全体目标。i.i.d. 小批量 ⇒ 平均而言偏差更低（相对有偏抽样）。

<br>


One dataset is almost all positive, another is all negative, and the two sets have similar size. How can training on both still fail?

一份数据几乎全是正类，另一份全是负类，规模相近。合在一起训练仍可能学不会。怎样发生？

> [!EXAMPLE]+ 连续的小批量如果先全正、后全负，模型会忘掉前一段
>
> 设数据块 A 几乎全正、块 B 几乎全负，**不打乱**顺序训练。
>
> 1. 在 A 上多步更新 ⇒ 边界移向「预测正」甚至**总是正**。
> 2. 接着在 B 上多步更新 ⇒ 为降 B 上损失，边界移向「总是负」。
> 3. B 足够长时，**灾难性遗忘** A 上学到的边界，最终像只学了 B。
>
> **修复**：shuffle，使每个 mini-batch 里正负都有，梯度同时反映两类。

<br>


A large batch is always a good idea.

- T
- F

大批量总是更好。

> [!EXAMPLE]+ F
>
> 大批量并非总更好：
>
> - **显存**：一次放不下整个训练集。
> - **泛化**：梯度噪声小，易陷 sharp minima，**测试误差有时更高**。
> - **优化**：每步仍可能因学习率过大而发散；大批量只是梯度更准，不保证更快收敛到好解。

<br>


One advantage of Adam over plain gradient descent.

Adam 相对普通梯度下降的一个好处。

> [!EXAMPLE]+ 每个坐标有自己的步长，一阶矩还能穿过鞍点
>
> Adam 维护：
>
> - 一阶矩 $v$：梯度的 EMA，**沿持续方向累积**，有助于穿过 ravine/鞍点附近；
> - 二阶矩 $s$：梯度平方的 EMA，**按坐标**缩放步长（大梯度坐标步长自动变小）。
>
> 普通 GD 所有坐标共用同一学习率 $\alpha$，无法自适应各维尺度。

<br>


## 学习率与搜索

Name one symptom of a learning rate that is too large, and one of a learning rate that is too small.

学习率过大、过小各有什么典型现象？

> [!EXAMPLE]+ 震荡或发散；损失下降极慢
>
> - **过大**：单步跨过谷底，损失**震荡**甚至**发散**（迭代曲线上下跳或上升）。
> - **过小**：每步移动 $\alpha\nabla J$ 极小，损失**单调但几乎水平**，要很多 epoch 才有可见下降。
>
> 调参时在 $\{10^{-4},10^{-3},\ldots\}$ 等**对数间隔**上试，比在同一数量级里盲猜可靠。

<br>


Candidate learning rates: $0.01,0.16,0.21,0.84,0.94$. This is a good grid.

- T
- F

这组学习率的搜索是合理的。（Fall 2019）

> [!EXAMPLE]+ F
>
> 五个候选是 $0.01,0.16,0.21,0.84,0.94$。其中 $0.16,0.21,0.84,0.94$ 四个点都挤在 $[0.1,1]$，只有 $0.01$ 落在 $[0.01,0.1]$。学习率应在对数轴上取样，让每个数量级的密度相近。

<br>


$\beta_1$ for Adam is drawn uniformly from $[0.9,0.999]$. Find $P(\beta_1<0.94)$ and $P(\beta_1>0.99)$. What is a better search?

在该区间上均匀抽 $\beta_1$。两个概率是多少？

> [!EXAMPLE]+ $0.04/0.099$ 与 $0.009/0.099$
>
> 均匀分布落在一段里的概率，等于这段长度除以总长度。总长度是右端减左端：$0.999-0.9=0.099$。
>
> - $\beta_1<0.94$ 的长度是 $0.94-0.9=0.04$，概率 $0.04/0.099$。
> - $\beta_1>0.99$ 的长度是 $0.999-0.99=0.009$，概率 $0.009/0.099$。
>
> $\beta_1$ 靠近 1 时，有效记忆长度 $1/(1-\beta_1)$ 变化很快。应在 $1-\beta_1$ 的对数尺度上搜索，而不是在 $\beta_1$ 上均匀抽。

<br>


Write momentum and the bias-corrected Adam update.

写出 momentum 与 Adam 的更新。

> [!EXAMPLE]+ 一阶矩、二阶矩，再做偏差修正
>
> Momentum：
>
> $$
> v=\beta v+(1-\beta)\frac{\partial J}{\partial W},\qquad
> W\leftarrow W-\alpha v.
> $$
>
> 指数平均抵消来回震荡的分量，沿持续方向加速。
>
> Adam（$t$ 从 1 起）：
>
> $$
> \begin{aligned}
> v&=\beta_1 v+(1-\beta_1)g,\\
> s&=\beta_2 s+(1-\beta_2)g^{\odot 2},\\
> \hat v&=\frac{v}{1-\beta_1^{t}},\qquad
> \hat s=\frac{s}{1-\beta_2^{t}},\\
> W&\leftarrow W-\alpha\,\hat v\oslash(\sqrt{\hat s}+\epsilon).
> \end{aligned}
> $$
>
> 常用起点 $\alpha=10^{-3}$，$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$。$v,s$ 从 0 开始时，前几步的原始矩偏小，除以 $1-\beta^{t}$ 把它们拉回无偏尺度。

<br>


Logistic regression is initialized at $0.5$ for every weight. A 4-layer net is initialized at the same constant. Which initialization is acceptable?

逻辑回归与四层网络都把权重初始化为常数 $0.5$。哪一个可以？

> [!EXAMPLE]+ 只有逻辑回归可以
>
> **逻辑回归**：损失对 $(w,b)$ **凸**，任意初值（含全 $0.5$）都能收敛到同一全局最小（模学习率），**可接受**。
>
> **四层网络**：若每层所有权重初值相同，前向时同层神经元输出相同，反向时梯度相同 ⇒ **对称性永不破**，等价于一个神经元，**不可接受**。
>
> 全 $0$ 或全常数初始化对深层网络都有此问题；逻辑回归没有「多单元对称」问题。

<br>


Early-stopping patience $k=5$ versus $k=0$. What is the benefit of the larger buffer? (Fall 2022)

验证集连续 5 轮不提升才停止，相比「一轮不提升就停」，好处是什么？

> [!EXAMPLE]+ 不被验证曲线的噪声提前打断
>
> 验证 loss/acc 有**随机波动**。$k=0$：一轮不改善就停 ⇒ 可能在**暂时变差**时停，权重还欠拟合。
>
> $k=5$：连续 5 轮无改善才停 ⇒ 允许短期噪声，更可能在**真正 plateau** 后停。
>
> 这不保证更少训练时间（往往更久），也不改变推理成本。

<br>


The training curve oscillates strongly early in training. Which stabilize it?

- (i) Early stopping. (ii) Learning-rate schedules. (iii) Data augmentation. (iv) Gradient clipping.

训练曲线一开始剧烈震荡。哪些有助于稳住？（Fall 2022 选项答案为 (ii)(iv)）

> [!EXAMPLE]+ 学习率日程与梯度裁剪
>
> - (ii) **学习率 schedule**：前期大 lr 探索，后期减小 ⇒ 更新幅度下降，**震荡减弱**。
> - (iv) **梯度裁剪**：限制 $\|g\|$，防止单步过大 ⇒ **稳定早期训练**。
> - (i) early stopping：只决定何时停，**不减小**每步震荡。
> - (iii) 数据增强：改变训练分布/泛化，**不直接**平滑 loss 曲线上的优化噪声。

<br>


# 归一

来源：Practice Midterm 1 的 BatchNorm 题。

Which statements are true?

- (i) BatchNorm makes one batch cheaper, so the same wall-clock time buys more updates.
- (ii) BatchNorm weakens the coupling between earlier and later layers.
- (iii) It makes the distribution seen by the next layer more consistent across dimensions.
- (iv) It removes the damage of a bad initialization and, in particular, allows weights to be initialized near zero.

关于 BatchNorm，哪些成立？

> [!EXAMPLE]+ (ii)(iii)
>
> - (i) **错**：BN 前向要算 batch 均值/方差，反向多传 $\gamma,\beta$，**单步更慢**；好处是常需**更少 epoch** 达到同样 loss，不是「每步更便宜」。
> - (ii) **对**：BN 把每层输入拉回稳定分布，减轻 **internal covariate shift**，浅层变化不会剧烈拖动深层输入。
> - (iii) **对**：归一化后各维尺度更一致，下一层看到的分布**跨特征更稳定**。
> - (iv) **错**：BN 不能「修复任意坏初始化」，更不能作为把权重 init 到 **0 附近** 的理由（ReLU 会死，sigmoid 仍饱和）。

<br>


What does $\epsilon$ do? What fails at batch size 1? A dense layer maps 10 units to 20 and is followed by BatchNorm on the 20 pre-activations, with no redundant bias. How many trainable parameters?

$\epsilon$ 的作用是什么？batch size 为 1 时哪里坏了？$10\to 20$ 的全连接再加 BatchNorm，可训练参数有多少？

> [!EXAMPLE]+ 240 个参数
>
> $\epsilon$ 防止方差为 0 时除零。
>
> batch size 为 1 时，该样本就是均值，归一化结果恒为 0，再乘 $\gamma$ 仍是 0。只剩 $\beta$ 能变，特征学不出来。更一般地，极少样本上的均值和方差噪声极大，滑动平均也会偏。
>
> 全连接把 $10$ 个输入映到 $20$ 个单元。每个输出单元连到全部 $10$ 个输入，权重是 $10\times 20=200$。这一层的偏置本来还有 $20$ 个，但后面紧跟 BatchNorm，$\beta$ 已经能吸收偏置，所以这 $20$ 个偏置不再单列。
>
> BatchNorm 作用在 $20$ 个预激活上：每个通道一个 $\gamma$、一个 $\beta$，共 $20+20=40$。
>
> 可训练参数是 $200+40=240$。

<br>


训练时用当前 batch 的均值方差，并更新滑动平均；测试时只用滑动平均。

```python
# train
mu = np.mean(Z, axis=0)
var = np.var(Z, axis=0)
cache["mu"] = beta_avg * cache["mu"] + (1 - beta_avg) * mu
cache["var"] = beta_avg * cache["var"] + (1 - beta_avg) * var
# test
mu, var = cache["mu"], cache["var"]
Z_norm = (Z - mu) / np.sqrt(var + eps)
out = gamma * Z_norm + beta
```

<br>


A conv block is Conv, then BatchNorm, then the nonlinearity. Setting every conv bias to 0 and never learning it removes a degree of freedom. (Fall 2020)

- T
- F

卷积后面紧跟 BatchNorm。卷积偏置永久置 0，表达能力会少一块。

> [!EXAMPLE]+ F
>
> 卷积输出 $z=\mathrm{Conv}(x)+b_{\mathrm{conv}}$，BN 做 $\gamma\hat z+\beta$，其中 $\hat z$ 是减均值除方差后的 $z$。
>
> 把 $b_{\mathrm{conv}}$ 固定为 0 后，BN 的 $\beta$ 仍可学出等价的**平移**，表达能力不丢一块自由度。ResNet 因此在 Conv→BN 块里常设 `use_bias=False`。
>
> 若 BN 在 ReLU **之后**，或没有 BN，则**不能**随便删 conv bias。

<br>


Why are a learned scale $\gamma$ and shift $\beta$ applied after normalizing a batch?

BatchNorm 在减均值、除标准差之后，为什么还要可学习的 $\gamma$ 和 $\beta$？

> [!EXAMPLE]+ 否则这一层能表示的函数族会被归一化削掉
>
> 只做标准化时，送进 sigmoid 的值挤在 0 附近，网络被锁在近似线性的一段里。$\gamma,\beta$ 把仿射变换还回来，表示能力才恢复。测试时用的是训练过程里的滑动均值和方差，不是把这两层参数丢掉。

<br>


Which BatchNorm statements hold: the layer is skipped at test time; its parameters can be learned only by batch gradient descent; it speeds learning; mini-batch estimates inject noise into the next layer?

关于 BatchNorm：测试时跳过；参数只能用全量梯度下降学；能加快学习；小批量上的均值方差会给下一层带来噪声。哪些成立？

> [!EXAMPLE]+ 加快学习，以及小批量噪声
>
> 测试时不跳过。用的是训练阶段积累的滑动统计量，单个测试样本因此仍能被归一化。$\gamma,\beta$ 对优化器没有限制，Adam 同样能更新它们。噪声来自每个小批量对均值和方差的估计，批量越小越强，这也是弱正则的来源。LayerNorm 按样本而不是按批量计算，批量变大不会增强它的正则。

<br>


# 卷积

## 尺寸与参数

输出边长（方阵、方核）：

$$
H_{\mathrm{out}}=\left\lfloor\frac{H+2p-f}{s}\right\rfloor+1.
$$

卷积参数：$(f\cdot f\cdot C_{\mathrm{in}}+1)\,C_{\mathrm{out}}$。池化与 ReLU 的参数为 0。BatchNorm 每个通道有 $\gamma$ 与 $\beta$，共 $2C$。

Same spatial size as the input, stride 1. What padding does a filter of width $f$ need?

stride 为 1 时，多宽的 padding 能使输出边长等于输入？

> [!EXAMPLE]+ $(f-1)/2$
>
> 要 $H_{\mathrm{out}}=H$，stride $1$：$H_{\mathrm{out}}=\lfloor(H+2p-f)/1\rfloor+1=H+2p-f+1$。令 $H_{\mathrm{out}}=H$ ⇒ $2p=f-1$ ⇒ $p=(f-1)/2$。
>
> $f$ 为奇数时 $p$ 为整数：$f=3\Rightarrow p=1$；$f=5\Rightarrow p=2$。

<br>


Input $32\times 32\times 3$, kernel $5\times 5$, padding 0, stride 1, 2 filters. Give the output volume and the parameter count. Repeat for a 1-D conv of width 5 on $T\times 3$, and a $5\times 5\times 5$ conv on video. (Practice 1)

三种核的参数量。

> [!EXAMPLE]+ 152、32、752
>
> 参数公式都是「一个核的权重个数 $+1$ 个偏置，再乘滤波器个数」。这里滤波器都是 $2$ 个，输入通道都是 $3$。
>
> 图像是 $32\times 32\times 3$，核 $5\times 5$，padding $0$，stride $1$。输出边长 $(32-5)/1+1=28$，两个滤波器，体积 $28\times 28\times 2$。一个核的权重是核高 $\times$ 核宽 $\times$ 输入通道 $=5\times 5\times 3=75$，加偏置 $76$，两个核 $76\times 2=152$。
>
> 一维卷积没有高这一维。宽度 $5$ 的核只看 $3$ 个通道，权重 $5\times 3=15$，加偏置 $16$，两个核 $16\times 2=32$。
>
> 视频核是 $5\times 5\times 5$，还要乘输入通道 $3$，一个核的权重 $5\times 5\times 5\times 3=375$，加偏置 $376$，两个核 $376\times 2=752$。

<br>


Why use a $1\times 1$ convolution?

$1\times 1$ 卷积改变什么、保留什么？

> [!EXAMPLE]+ 混合通道，保留高和宽
>
> $1\times 1$ 卷积在每个 $(h,w)$ 位置对 **$C_{\mathrm{in}}$ 个通道**做线性组合，得到 $C_{\mathrm{out}}$ 个通道。
>
> - **改变**：通道数 $C$（升维/降维、跨通道混合）。
> - **不变**：空间高 $H$、宽 $W$（stride 1、padding 0 时）。
> - 参数量 $(1\cdot 1\cdot C_{\mathrm{in}}+1)C_{\mathrm{out}}$，比同尺寸展平后全连接小得多。

<br>


An input is $64\times 64\times 16$. How many parameters does one $1\times 1$ filter have, including bias?

输入 $64\times 64\times 16$。一个含偏置的 $1\times 1$ 滤波器有多少参数？

> [!EXAMPLE]+ 17
>
> $1\times 1$ 核在每个空间位置只取一个像素，但要看满该位置的全部通道。输入通道是 $16$，所以权重是 $1\times 1\times 16=16$，再加 $1$ 个偏置，共 $17$。$64\times 64$ 只决定这个核会被用到多少个位置，不增加参数。

<br>


Input shape $(n_h,n_w,n_c)$, a $1\times 1$ convolution with stride 1 and padding 0, and ordinary max-pooling. Which are true?

- (i) A $1\times 1$ convolution can reduce $n_c$, not $n_h$ or $n_w$.
- (ii) Max-pooling can reduce $n_h$ and $n_w$, not $n_c$.
- (iii) A $1\times 1$ convolution can reduce all three.
- (iv) Max-pooling can reduce all three.

$1\times 1$ 卷积与普通最大池化，各自能缩小哪些维度？

> [!EXAMPLE]+ (i)(ii)
>
> - (i) **对**：$1\times 1$、$s=1$、$p=0$ 时 $H,W$ 不变；输出通道 $=C_{\mathrm{out}}$，可小于 $C_{\mathrm{in}}$ ⇒ **只缩通道**。
> - (ii) **对**：max-pool 在每个通道**独立**做窗口，**只缩** $H,W$，$C$ 不变。
> - (iii) **错**：$1\times 1$ **不能**缩小 $H,W$（在本题设定下）。
> - (iv) **错**：标准 max-pool **不改变**通道数。

<br>


Fill the table. `CONV5-N` means $N$ filters of size $5$, padding 2, stride 1. `POOL2` is $2\times 2$, stride 2. Input is $32\times 32\times 1$. (Fall 2019)

填每一层的输出体积与参数量。

| 层 | 输出 | 参数 |
| :--- | :--- | :--- |
| Input | $32\times 32\times 1$ | 0 |
| CONV5-10 | $32\times 32\times 10$ | $(25\cdot 1+1)\cdot 10=260$ |
| POOL2 | $16\times 16\times 10$ | 0 |
| CONV5-10 | $16\times 16\times 10$ | $(25\cdot 10+1)\cdot 10=2510$ |
| POOL2 | $8\times 8\times 10$ | 0 |
| FC-10 | $10$ | $(640+1)\cdot 10=6410$ |

输出边长是 $\lfloor(H+2p-f)/s\rfloor+1$。第一层卷积 $f=5$、$p=2$、$s=1$：$(32+4-5)/1+1=32$，通道数等于滤波器个数 $10$。一个核的权重是 $5\times 5\times 1=25$，加 $1$ 个偏置是 $26$，十个核 $26\times 10=260$。

池化窗口 $2$、步长 $2$，边长减半：$32/2=16$，再 $16/2=8$，池化没有参数。

第二次卷积的输入通道已经是 $10$，一个核是 $5\times 5\times 10=250$，加偏置 $251$，十个核 $251\times 10=2510$。

展平：$8\times 8\times 10=640$。全连接 $10$ 个输出，每个连到这 $640$ 个数再加 $1$ 个偏置，$(640+1)\times 10=6410$。

Input $32\times 32\times 3$. `CONV3` uses padding 1 and stride 1. `POOL-n` has window $n$, stride $n$, padding 0. (Fall 2020)

填 Fall 2020 的卷表。

| 层 | 输出 | 参数 |
| :--- | :--- | :--- |
| Input | $32\times 32\times 3$ | 0 |
| CONV3-8 | $32\times 32\times 8$ | $8\cdot(27+1)=224$ |
| Leaky ReLU | 同左 | 0 |
| POOL-2 | $16\times 16\times 8$ | 0 |
| BatchNorm | $16\times 16\times 8$ | $2\cdot 8=16$ |
| CONV3-16 | $16\times 16\times 16$ | $16\cdot(72+1)=1168$ |
| Leaky ReLU | 同左 | 0 |
| POOL-2 | $8\times 8\times 16$ | 0 |
| Flatten | $1024$ | 0 |
| FC-10 | $10$ | $(1024+1)\cdot 10=10250$ |

第一层卷积 $f=3$、$p=1$、$s=1$：$(32+2-3)/1+1=32$。一个核是 $3\times 3\times 3=27$，加偏置 $28$，$8$ 个核 $28\times 8=224$。

`POOL-2` 把边长减半，$32\to 16$，再 $16\to 8$。Leaky ReLU 与池化都没有参数。

BatchNorm 在 $8$ 个通道上各有 $\gamma$ 和 $\beta$，共 $2\times 8=16$。

第二次卷积的输入通道是 $8$，一个核是 $3\times 3\times 8=72$，加偏置 $73$，$16$ 个核 $73\times 16=1168$。输出边长仍是 $(16+2-3)/1+1=16$。

展平：$8\times 8\times 16=1024$。全连接 $10$ 类，$(1024+1)\times 10=10250$。

Input $128\times 128\times 3$. First conv only. (Spring 2021)

只看第一层卷积。

> [!EXAMPLE]+ 三组超参
>
> 输入都是 $128\times 128\times 3$。边长 $\lfloor(128+2p-f)/s\rfloor+1$。一个核的权重是 $f\times f\times 3$，再加 $1$ 个偏置，然后乘滤波器个数。
>
> - $f=3,p=0,s=1$：$ (128-3)/1+1=126$，体积 $126\times 126\times 64$。一个核 $3\times 3\times 3+1=28$，$64$ 个核 $28\times 64=1792$。
> - $f=5,p=2,s=1$：$(128+4-5)/1+1=128$，体积 $128\times 128\times 16$。一个核 $5\times 5\times 3+1=76$，$16$ 个核 $76\times 16=1216$。
> - $f=2,p=0,s=2$：$(128-2)/2+1=64$，体积 $64\times 64\times 32$。一个核 $2\times 2\times 3+1=13$，$32$ 个核 $13\times 32=416$。

<br>


Three identical modules follow: Conv, $2\times 2$ max-pool with stride 2, ReLU. Start from $128\times 128\times 3$. Which setting gives the smallest 10-way fully connected layer, and how many parameters does that layer have?

三个「卷积 + $2\times 2$ 池化 + ReLU」模块之后接 10 类全连接。哪种卷积配置让全连接参数最少？

> [!EXAMPLE]+ stride 2、核 $2\times 2$、32 个滤波器；全连接 1290 个参数
>
> 每个模块先卷积、再做 $2\times 2$ 步长 $2$ 的池化。卷积边长 $\lfloor(H+2p-f)/s\rfloor+1$，这次池化就是把边长大约砍半。全连接看的是最后的高 $\times$ 宽 $\times$ 通道，再加上 $1$ 个偏置，乘 $10$ 类。
>
> - $f=3,p=0,s=1$，通道 $3\to 64$。空间：$128$ 卷积得 $(128-3)+1=126$，池化得 $\lfloor(126-2)/2\rfloor+1=63$；再卷积 $(63-3)+1=61$，池化 $\lfloor(61-2)/2\rfloor+1=30$；再卷积 $(30-3)+1=28$，池化 $\lfloor(28-2)/2\rfloor+1=14$。体积 $14\times 14\times 64=12544$。全连接 $(12544+1)\times 10=125450$。卷积参数：第一层 $(3\times 3\times 3+1)\times 64=1792$；后两层输入通道已是 $64$，每层 $(3\times 3\times 64+1)\times 64=36928$，合计 $1792+36928+36928=75648$。
> - $f=5,p=2,s=1$，通道 $3\to 16$。padding 刚好保住边长，池化才减半：$128\to 64\to 32\to 16$。体积 $16\times 16\times 16=4096$。全连接 $(4096+1)\times 10=40970$。卷积：第一层 $(5\times 5\times 3+1)\times 16=1216$；后两层输入通道 $16$，每层 $(5\times 5\times 16+1)\times 16=6416$，合计 $1216+6416+6416=14048$。
> - $f=2,s=2,p=0$，通道 $3\to 32$。卷积自己先减半，池化再减半：$128$ 卷积得 $(128-2)/2+1=64$，池化得 $32$；接着 $32\to 16\to 8$；$8\to 4\to 2$。体积 $2\times 2\times 32=128$。全连接 $(128+1)\times 10=1290$。卷积：第一层 $(2\times 2\times 3+1)\times 32=416$；后两层输入通道 $32$，每层 $(2\times 2\times 32+1)\times 32=4128$，合计 $416+4128+4128=8672$。
>
> 三种全连接参数是 $125450$、$40970$、$1290$。最少的是第三种。它的空间被步长和池化各砍掉很多，分类层几乎不看细粒度位置。

<br>


## 感受野与小网络

Layers, in order: $3\times 3$ conv stride 2, $2\times 2$ pool stride 2, $3\times 3$ conv stride 2, $2\times 2$ pool stride 2. How large is the support, in input pixels, of one unit in the fourth non-image layer? (Winter 2021 题干)

第四个非图像层上的一个神经元，对应输入上多大的一片？

> [!EXAMPLE]+ $21\times 21$
>
> 感受野 $r$ 与跳步 $j$ 从 $r=1,j=1$ 起，表示一开始一个像素只覆盖自己。经过核宽 $k$、步长 $s$ 的一层：
>
> $$
> r\leftarrow r+(k-1)\,j,\qquad j\leftarrow j\cdot s.
> $$
>
> - $3\times 3$、步长 $2$：$r=1+(3-1)\times 1=3$，$j=1\times 2=2$。
> - $2\times 2$、步长 $2$：$r=3+(2-1)\times 2=5$，$j=2\times 2=4$。
> - $3\times 3$、步长 $2$：$r=5+(3-1)\times 4=13$，$j=4\times 2=8$。
> - $2\times 2$、步长 $2$：$r=13+(2-1)\times 8=21$，$j=8\times 2=16$。
>
> 第四层一个单元对应输入上边长 $21$ 的正方形，共 $21\times 21=441$ 个像素。池化步长若不是 $2$，每一行的 $j$ 都会变，后面的 $r$ 跟着变。

<br>


Classify a $16\times 16\times 3$ digit image into 10 classes with Conv, max-pool, flatten, and a fully connected layer. Constraints: stride 1, exactly 4 filters, conv activation has at most 576 entries, pool stride equals pool size, the classifier has no bias and at most 1440 parameters. Give one feasible design. Which digit augmentations are valid: mild scale, flips, rotations by 90 or 180, mild shear? (Winter 2021)

在内存限制下给一个可行的小 CNN，并判断数字图像的增强。

> [!EXAMPLE]+ 例如 $5\times 5$ 卷积、padding 0，再 $2\times 2$ 池化
>
> 输入 $16\times 16\times 3$，stride 为 $1$，滤波器恰好 $4$ 个。取 $5\times 5$、padding $0$：边长 $(16-5)/1+1=12$，激活条目 $12\times 12\times 4=576$，顶到上限。
>
> 池化窗口等于步长，取 $2$：边长 $(12-2)/2+1=6$，体积 $6\times 6\times 4=144$。无偏置全连接是「每个类别连到全部 $144$ 个位置」，$10$ 类共 $144\times 10=1440$，也顶到上限。输出维度只能是 $10$。
>
> 轻微缩放、轻微剪切保留数字身份。水平或垂直翻转会毁掉 7 这类字形。$90^\circ$ 或 $180^\circ$ 旋转会把 6 转成 9。

<br>


Among conv, max-pool, average-pool, and BatchNorm, which uses the fewest floating-point operations for an activation of size $n$? (Fall 2020)

同样大小的激活上，哪一种层的浮点运算最少？

> [!EXAMPLE]+ 最大池化
>
> 设激活有 $n$ 个元素（或每个池化窗口内 $f^2$ 个值）。
>
> - **Max-pool**：每个窗口做 $f^2-1$ 次**比较**，几乎无乘加。
> - **Average-pool**：要把窗口内 $f^2$ 个数**相加再除**，乘加更多。
> - **Conv / BN**：至少一次完整乘加遍历，常数更大。
>
> 故 FLOPs 最少的是 **max-pool**。

<br>


# 反向

## 温度 softmax

编码器先把 $f_\theta(X)$ 变成 softmax，再想得到 one-hot，然后送入下一层 $g$（Fall 2022）。

$y=g(\mathrm{one\_hot}(\mathrm{softmax}(f(X))))$ is trainable by gradient descent.

- T
- F

one-hot 放在计算图中间，梯度下降仍然可用。

> [!EXAMPLE]+ F
>
> 计算图：$f(X)\to\mathrm{softmax}\to\mathrm{one\_hot}\to g(\cdot)$。
>
> one-hot / argmax 对 softmax 输出的导数**几乎处处为 0**（只在切换点无经典导数），链式法则在 $\mathrm{one\_hot}$ 处断掉，$\partial L/\partial f$ 为 0。
>
> 训练 $f$ 需可微近似（Gumbel-Softmax、直通估计器）或去掉 one-hot，不能硬插离散层。

<br>


$S_\tau(u)=\mathrm{softmax}(u/\tau)$. What happens as $\tau\to\infty$ and as $\tau\to 0$?

温度趋向无穷和趋向 0 时，分布变成什么样？

> [!EXAMPLE]+ 均匀分布，以及 argmax 上的 one-hot
>
> $\tau$ 很大时，logits 的差距被抹平，各类概率趋于 $1/n$。
>
> $\tau$ 很小时，最大的 logit 主导指数，质量集中到 argmax。因此低温度是 one-hot 的可微近似，不必再接一个离散的 one-hot。

<br>


Softmax Jacobian: $y_i=e^{x_i}/\sum_k e^{x_k}$. Derive $\partial y_i/\partial x_j$.

softmax 的雅可比 $\partial y_i/\partial x_j$ 是什么？

> [!EXAMPLE]+ 对角是 $y_i(1-y_i)$，非对角是 $-y_i y_j$
>
> 商法则：分子是 $e^{x_i}$，分母是 $Z=\sum_k e^{x_k}$。
>
> $$
> \frac{\partial y_i}{\partial x_j}=y_i\big(\mathbf 1[i=j]-y_j\big).
> $$
>
> $i=j$ 时为 $y_i(1-y_i)$，$i\neq j$ 时为 $-y_i y_j$。温度 softmax 只是先把 $x$ 换成 $x/\tau$，整个雅可比再乘 $1/\tau$。

<br>


$f(X)=w^\top X$ produces one logit per coordinate of $X$. Derive $\partial S_\tau(w^\top X)_i/\partial w$.

第 $i$ 个温度 softmax 分量对 $w$ 的导数。

> [!EXAMPLE]+ 与普通 softmax Jacobian 只差一个 $1/\tau$
>
> 记 $s=S_\tau(w^\top X)$，$X_i$ 为产生第 $i$ 个 logit 的那一列特征。
>
> $$
> \frac{\partial s_i}{\partial w}
> =\frac{1}{\tau}\,s_i\left(X_i-\sum_j s_j X_j\right).
> $$
>
> $\tau$ 很小会使这个雅可比很大，训练不稳定。官方解答把「过低的温度」对应到高方差、难训练。

<br>


## 孪生网络

两支网络共享 $W_1,b_1$（Fall 2020）：

$$
\begin{aligned}
a_1&=\mathrm{ReLU}(W_1 x+b_1),&
a_2&=\mathrm{ReLU}(W_1 x'+b_1),\\
a&=a_1-a_2,&
\hat y&=\sigma(W_2 a+b_2).
\end{aligned}
$$

单样本损失是二元交叉熵里的正项，成本 $J$ 再取负号平均，因此最小化 $J$ 就是最小化交叉熵。

State the shapes. For one example, simplify $\partial J/\partial z_3$.

参数形状是什么？输出预激活上的梯度如何化简？

> [!EXAMPLE]+ $\partial J/\partial z_3=(\hat y-y)/m$
>
> $W_1\in\mathbb R^{d_a\times d_x}$，$b_1\in\mathbb R^{d_a}$，$W_2\in\mathbb R^{1\times d_a}$，$b_2$ 为标量。按列堆叠样本时 $X\in\mathbb R^{d_x\times m}$。
>
> 交叉熵对 sigmoid 预激活的导数是 $\hat y-y$，再除以 $m$。共享权重的梯度要加起来自 $x$ 与来自 $x'$ 的两支：
>
> $$
> \frac{\partial J}{\partial W_1}
> =\delta_1 x^\top+\delta_2 (x')^\top,
> $$
>
> 其中 $\delta_1$ 经过 $a=a_1-a_2$ 时带正号，$\delta_2$ 带负号，并且都乘上各自的 ReLU 指示函数。

<br>


Continue the same siamese net. Write $\partial z_3/\partial a$, $\partial a/\partial z_1$, $\partial a/\partial z_2$, and the gradient-descent updates for $W_1,b_1,W_2,b_2$.

同一孪生网络。写出 $\partial z_3/\partial a$、$\partial a/\partial z_1$、$\partial a/\partial z_2$，以及四个参数的梯度下降更新。

> [!EXAMPLE]+ $W_2$ 把标量梯度送回差值，$z_2$ 一支再加负号
>
> $\partial J/\partial z_3=(\hat y-y)/m$ 已含 $1/m$，后面不再除一次。
>
> $$
> \frac{\partial z_3}{\partial a}=W_2,\qquad
> \frac{\partial J}{\partial a}=W_2^\top\frac{\partial J}{\partial z_3}.
> $$
>
> $a=a_1-a_2$，ReLU 的指示函数记为 $H$。
>
> $$
> \frac{\partial a}{\partial z_1}=H(z_1),\qquad
> \frac{\partial a}{\partial z_2}=-H(z_2).
> $$
>
> $$
> \begin{aligned}
> \frac{\partial J}{\partial z_1}
> &=\frac{\partial J}{\partial a}\odot H(z_1),&
> \frac{\partial J}{\partial z_2}
> &=-\frac{\partial J}{\partial a}\odot H(z_2),\\
> \frac{\partial J}{\partial W_1}
> &=\frac{\partial J}{\partial z_1}x^\top
> +\frac{\partial J}{\partial z_2}(x')^\top,&
> \frac{\partial J}{\partial b_1}
> &=\frac{\partial J}{\partial z_1}+\frac{\partial J}{\partial z_2},\\
> \frac{\partial J}{\partial W_2}
> &=\frac{\partial J}{\partial z_3}\,a^\top,&
> \frac{\partial J}{\partial b_2}
> &=\frac{\partial J}{\partial z_3}.
> \end{aligned}
> $$
>
> $J$ 已经对 $m$ 取过平均，更新是把各样本的上述梯度相加，再乘学习率：
>
> $$
> W\leftarrow W-\alpha\sum_{i=1}^{m}\frac{\partial J}{\partial W^{(i)}}.
> $$
>
> $b_1,W_2,b_2$ 同样。两支共享 $W_1$ 时，漏掉 $x'$ 那一支，或在 $a_1-a_2$ 上漏掉负号，都会把「相似」和「不同」的方向学反。

<br>


## 图网络

两层图网络（Fall 2022）。节点特征是标量。第 $k$ 层只把至多 $k$ 跳邻居的信息写进该节点。

$$
a^{[1]}_i=\mathrm{ReLU}\!\left(x_i+w^{[1]}\sum_{n\in\mathcal N(i)}x_n+b^{[1]}\right),
$$

第二层把 $x$ 换成 $a^{[1]}$，参数换成 $w^{[2]},b^{[2]}$。

On a path where node 1 and node 6 are more than two hops apart, what is $\partial a^{[2]}_1/\partial x_6$? How can one new node make this derivative nonzero, using as few new edges as possible?

两跳以外的节点，二层输出对其输入的导数是多少？如何用一个新节点把这个导数变成非零？

> [!EXAMPLE]+ 导数为 0；新节点同时连到 1 和 6
>
> 题设 GNN **2 层**消息传递：节点 $1$ 的输出 $a^{[2]}_1$ 只聚合 **$\le 2$ 跳** 内的邻居信息。节点 $6$ 与 $1$ 相距 **$>2$ 跳** ⇒ $\partial a^{[2]}_1/\partial x_6=0$。
>
> **加一个新节点** $7$，连边 $6\to 7$ 与 $7\to 1$（或 $1\leftrightarrow 7$、$6\leftrightarrow 7$）：$6$ 到 $1$ 的最短路径变为 $6\to 7\to 1$，长度 $2$，$x_6$ 可影响 $a^{[2]}_1$，导数**可非零**。

<br>


# 生成

## 对抗样本

Fast Gradient Sign Method: $x^\star=x+\varepsilon\,\mathrm{sign}(\partial J/\partial x)$. Given $x=(1,2,3)^\top$, $\partial J/\partial x=(0.5,-0.5,1)^\top$, $\varepsilon=0.01$, compute $x^\star$. (Practice 1)

计算这一步 FGSM。

> [!EXAMPLE]+ $(1.01,\,1.99,\,3.01)^\top$
>
> 符号只看梯度的正负，不看大小：$\mathrm{sign}(0.5,-0.5,1)=(1,-1,1)$。
>
> 每个坐标加上 $\varepsilon$ 乘这个符号，$\varepsilon=0.01$：
>
> $$
> (1,2,3)^\top+0.01\,(1,-1,1)^\top=(1.01,\,1.99,\,3.01)^\top.
> $$
>
> 梯度绝对值是 $0.5$ 还是 $1$，移动量都是 $0.01$。

<br>


The $\varepsilon$ needed to change a prediction grows with the dimension of $x$.

- T
- F

输入维度变大时，为了改变预测，必须把 $\varepsilon$ 也放大。

> [!EXAMPLE]+ F
>
> 对无偏置逻辑回归，$x^\star=x+\varepsilon\,\mathrm{sign}(w)$ 使 logit 增加 $\varepsilon\lVert w\rVert_1$。若每个权重量级差不多，$\lVert w\rVert_1$ 随维度线性增长。固定的小 $\varepsilon$ 在高维上仍能把 logit 推过决策边界，而每个像素只改了 $\varepsilon$。

<br>


Mark the true statements.

- (i) An example built to fool classifier A can also fool a similar classifier B.
- (ii) FGSM is an iterative attack.
- (iii) Dropout at test time is an effective defense.
- (iv) A black-box model, with no access to weights, can still be attacked.

关于对抗样本，哪些成立？

> [!EXAMPLE]+ (i)(iv)
>
> - (i) 对：结构相似的分类器，决策边界方向接近，**在一个模型上找到的对抗扰动常能迁移**到另一个。
> - (ii) 错：FGSM 是**单步**沿 $\mathrm{sign}(\nabla_x J)$ 走 $\varepsilon$；迭代攻击（如 PGD）是多步。
> - (iii) 错：Dropout 只在训练时随机；测试时关闭，**不是**对抗防御手段。
> - (iv) 对：黑盒可通过**查询输出**估计梯度，或利用迁移攻击，无需权重。

<br>


## GAN 与噪声

After the first generator and discriminator updates, is $D(G(z))$ closer to 0 or to 1? When is training finished? Which generator cost is preferable, the saturating one or the non-saturating one? (Fall 2022 与 Practice 1)

训练刚开始时 $D(G(z))$ 靠近哪一端？何时算训练完成？生成器该用哪种成本？

> [!EXAMPLE]+ 一开始靠近 0；完成时靠近 $1/2$；用 non-saturating
>
> **刚开始**：$G$ 随机，$D(G(z))$ 很小（判别器几乎总判假），sigmoid 输出**靠近 0**。
>
> **训练完成（纳什均衡附近）**：理想判别器 $D(x)\approx 1/2$ 对真假无区分；$D(G(z))\to 1/2$，不是 $\to 1$。若长期 $\to 1$ 只说明 $D$ 太弱或 $G$ 暂时骗过 $D$。
>
> **生成器损失**：saturating 形式 $\log(1-D(G(z)))$ 在 $D(G(z))\approx 0$ 时梯度 $\to 0$；non-saturating 用 $-\log D(G(z))$，早期梯度更大。
>
> **损失数值**：第 1 epoch 与第 100 epoch 损失相近，不代表图像一样好——$D$ 也在变，同一标量不可横比。

<br>


Discriminator likelihood on a minibatch, with sigmoid output and no biases:

$$
L=\frac1n\sum_{i=1}^{n}\Big(\log D(X_i)+\log\big(1-D(G(Z_i))\big)\Big).
$$

The discriminator maximizes $L$. The generator minimizes $L$. Write $\nabla_{\theta_d}L$ in terms of $\nabla_{\theta_d}D$, the last-layer pre-activation gradient, the recursion through earlier layers, and the two parameter updates. A printed variant adds the two scores, $L'=\frac1n\sum_i\big(D(X_i)+D(G(Z_i))\big)$, and calls the result a WGAN objective. What should the discriminator gradient be?

判别器最大化上面的对数似然，生成器最小化它。写出对判别器参数的梯度、最后一层预激活上的梯度、向更浅层的递推，以及两边的更新。卷面上把两项写成相加，并称为 WGAN。判别器梯度实际应是什么？

> [!EXAMPLE]+ 真实样本把 $D$ 推向 1，生成样本把 $D$ 推向 0
>
> $$
> \nabla_{\theta_d}L
> =\frac1n\sum_i\left(
> \frac{\nabla_{\theta_d}D(X_i)}{D(X_i)}
> -\frac{\nabla_{\theta_d}D(G(Z_i))}{1-D(G(Z_i))}
> \right).
> $$
>
> 最后一层是 sigmoid。真实样本对预激活的导数是 $(1-D)/n$，生成样本是 $-D/n$。
>
> 更浅一层，$a^l=g(z^l)$，$z^{l+1}=W^{l+1}a^l$：
>
> $$
> \frac{\partial L}{\partial z^l}
> =\Big((W^{l+1})^\top\frac{\partial L}{\partial z^{l+1}}\Big)\odot g'(z^l).
> $$
>
> 生成器的输出就是判别器的输入。设第一层没有单独的输入激活，则
>
> $$
> \frac{\partial L}{\partial G(Z_i)}=(W^1)^\top\frac{\partial L}{\partial z^1}.
> $$
>
> $$
> \nabla_{\theta_g}L
> =\sum_i\frac{\partial L}{\partial G(Z_i)}\,\nabla_{\theta_g}G(Z_i).
> $$
>
> 判别器沿梯度上升，生成器沿梯度下降，学习率都是 $\alpha$，且没有正则：
>
> $$
> \theta_d\leftarrow\theta_d+\alpha\nabla_{\theta_d}L,\qquad
> \theta_g\leftarrow\theta_g-\alpha\nabla_{\theta_g}L.
> $$
>
> 卷面把两项写成相加。那样真样本和假样本的分数会被一起推高，假样本不会被压下去，判别器完不成「分辨真假」。Wasserstein 批评器要最大化的是真样本分数减去假样本分数，梯度是
>
> $$
> \nabla_{\theta_d}L_{\mathrm{W}}
> =\frac1n\sum_i\Big(\nabla_{\theta_d}D(X_i)-\nabla_{\theta_d}D(G(Z_i))\Big).
> $$
>
> 再对批评器做梯度裁剪（或梯度惩罚），优化的才是推土机距离。

<br>


A generator of reptiles may be mode-collapsing. Which observations fit?

- (i) Every sample is a Komodo dragon.
- (ii) The generator loss oscillates.
- (iii) Generator loss stays low while discriminator loss stays high.
- (iv) The discriminator is accurate on real images and inaccurate on fake ones.

哪些现象更像 mode collapse？（Fall 2020）

> [!EXAMPLE]+ (i)(ii)
>
> **Mode collapse**：生成器只输出少数几种样本，多样性丧失。
>
> - (i) 对：全是同一种（Komodo dragon）⇒ 单一模式。
> - (ii) 对：生成器损失震荡，常对应在少数模式间切换、$D$ 与 $G$ 拉锯。
> - (iii) 不选为主因：$G$ 低、$D$ 高可能是 $D$ 过强或训练阶段问题，多样 $G$ 也可出现。
> - (iv) 不选为主因：$D$ 对真假准确率差，更像 $D$ 未学好，不直接等于 collapse。

<br>


Only a few real octopus photos exist. Which augmentation is safe for a GAN: blur, recoloring, or left-right flips?

数据很少时，模糊、改颜色、左右翻转，哪些适合拿去训生成章鱼的 GAN？

> [!EXAMPLE]+ 只保留左右翻转
>
> 真实章鱼照片极少，增强要**保持流形**、不引入伪影。
>
> - **模糊**：生成器会把模糊当真实纹理学进分布 ⇒ **不宜**。
> - **改色**：颜色偏移可能变成假模式 ⇒ **不宜**。
> - **左右翻转**：章鱼大致左右对称，**仍像章鱼** ⇒ **可用**。

<br>


Linear regression minimizes $J=\frac1m\sum_i(y^{(i)}-x^{(i)\top}\theta)^2$. Features are corrupted by $\delta\sim\mathcal N(0,\sigma^2 I)$. Show that the expected corrupted objective equals $J$ plus a term that does not depend on the data. (Winter 2021)

给特征加各向同性高斯噪声，期望损失比原损失多出什么？

> [!EXAMPLE]+ 多出 $\sigma^2\lVert\theta\rVert_2^2$，等价于 L2
>
> $$
> \frac{\partial J}{\partial\theta}=\frac{2}{m}X^\top(X\theta-Y).
> $$
>
> 展开 $(y-x^\top\theta-\delta^\top\theta)^2$ 后，交叉项含 $\mathbb E[\delta]=0$，消失。二次项
>
> $$
> \mathbb E[(\delta^\top\theta)^2]=\sigma^2\lVert\theta\rVert_2^2.
> $$
>
> 因此 $\mathbb E[J^\star]=J+\sigma^2\lVert\theta\rVert_2^2$。数据无关的这一项就是对 $\theta$ 的 L2 惩罚，系数由噪声方差决定。

<br>


## 自编码

$x\to q(z\mid x)\to z\to p(x\mid z)\to\hat x$。编码器输出的是高斯分布（Fall 2022）。三种损失：

- Alice：$\mathrm{KL}(q(z\mid x)\,\|\,\mathcal N(0,I))$
- Bob：$\mathrm{MSE}(x,\hat x)+\mathrm{KL}(q(z\mid x)\,\|\,\mathcal N(0,I))$
- Carol：$\mathrm{MSE}(x,\hat x)$

二维潜变量上有三张散点图：类别分开且尺度很大；各类混在原点附近的一团高斯里；仍像标准正态但各类分开。

> [!EXAMPLE]+ Alice 对应混在一起的高斯，Bob 对应分开且仍集中的高斯，Carol 对应尺度很大的分类团
>
> Alice 只把编码拉向标准正态，没有重建项，数字类别不会分开。
>
> Carol 只有重建，相似数字会聚成团，但没有力把云团按在原点附近，坐标尺度可以很大。
>
> Bob 两项都有：整体仍像标准正态，类与类之间因为重建误差而分开。这就是变分自编码器的两项损失。

<br>


A scalar autoencoder uses $z=w_1 x$ and $\hat x=w_2 z$, with squared reconstruction error. Which $(w_1,w_2)$ make the loss zero? Is there a saddle?

标量自编码器 $z=w_1 x$、$\hat x=w_2 z$，用平方重建误差。损失为 0 的解在哪里？有没有鞍点？

> [!EXAMPLE]+ $w_2 w_1=1$；$(0,0)$ 是鞍点
>
> 损失正比于 $(1-w_1 w_2)^2$。乘积为 1 时，每个样本都被精确还原。
>
> 在 $(0,0)$，两个偏导都是 0。沿 $w_1=w_2=\varepsilon$ 走，乘积为 $\varepsilon^2$，损失下降；沿 $w_1=\varepsilon$、$w_2=-\varepsilon$ 走，乘积为负，损失上升。同一个驻点，有的方向下降、有的方向上升，所以是鞍点，不是极小。

<br>


# 策略

500 ultrasound cases, 175 preterm. All positive cases are duplicated, and only then is the set split into train, validation, and test. What is wrong? The best model has $100\%$ recall. Is that enough if missing a preterm birth is unacceptable? Which human baseline best estimates Bayes error: one student (F1 0.20), one doctor (0.80), or a group of doctors (0.88)? (Practice 1)

先复制正类再划分数据有什么问题？召回率 100% 是否够用？贝叶斯误差该用哪组人类成绩估计？

> [!EXAMPLE]+ 泄漏；召回率可以被平凡模型刷满；用医生小组
>
> **数据问题**：先复制 175 个早产正例（变多份），再 random split ⇒ **同一原始图像的副本**可能既在 train 又在 test ⇒ **泄漏**，指标虚高。
>
> **正确顺序**：先划分 train/val/test，再**只在训练集**过采样正类。
>
> **召回 100%**：全预测「早产」也能召回 100%，但精度极低；漏诊不可接受时还要看 **precision / specificity**，不能单看 recall。
>
> **贝叶斯误差**：用**最强可用人类表现**估下界；医生小组 F1 $0.88$ 优于单个医生 $0.80$ 或学生 $0.20$（误差量级约 $1-0.88=0.12$，F1 不是误差但方向正确）。

<br>


Chest X-ray cancer screening: accuracy, precision, recall, or F1? A click-through model can be one large net or an average of ten smaller nets, both at 90% training accuracy. The ten-model ensemble was cheaper to train. Deployment is one GPU and must answer in real time. Which system goes to production? (Fall 2020)

胸片筛查看哪个指标？上线时选大模型还是十模型平均？

> [!EXAMPLE]+ 筛查看召回；上线用单个大模型
>
> **指标**：筛查要尽量少漏癌 ⇒ **recall（敏感性）** 优先；precision 也要够，但 accuracy 在稀有癌种上会被「全判阴性」刷高。
>
> **部署**：十模型 ensemble 训练可能更省，但推理要 **10 次前向**；单 GPU、低延迟 ⇒ 选 **一个大模型** 一次前向。
>
> 二者训练准确率都是 90%，不能说明谁泛化更好；决策依据是 **latency 与 recall 需求**。

<br>


100,000 chest CT images, 1,000 labeled, for COVID-19. Pick one:

- (i) Transfer from a ResNet-50 pretrained on chest CT for tumor detection.
- (ii) Train a GAN to synthesize labeled images, then supervise on real plus synthetic labels.
- (iii) Supervise from scratch on the 1,000 labels.
- (iv) Random crops of the 1,000 labels, then supervise from scratch.

标注很少、同模态图像很多。哪一条最有机会做成？（Fall 2022）

> [!EXAMPLE]+ 迁移已在胸片 CT 上预训练的 ResNet
>
> - (i) **选**：$10^5$ 无标 CT + $10^3$ 有标；同模态 CT 上预训练 ResNet，再微调分类头，是标准 **transfer learning**。
> - (ii) 不选：GAN 需大量数据学分布；每类阳性/阴性标注都很少，合成图会把生成 artifact 当标签学进去。
> - (iii) 不选：ResNet-50 参数量 $\gg 1000$，从头训极易过拟合。
> - (iv) 不选：只裁剪 $1000$ 张，增强倍数有限，仍不够覆盖 CT 多样性。

<br>


Lane detection from a forward camera on two-way California streets. Which augmentations are valid?

- (i) Vertical flip. (ii) Horizontal flip. (iii) Synthetic fog. (iv) Random masks on a small region.

车道方向分「同向」和「对向」。哪些增强还成立？

> [!EXAMPLE]+ 雾与小块遮挡
>
> - (i) 垂直翻转：天空/路面互换，**物理上不合理**，标签（车道线）失效。
> - (ii) 水平翻转：美国**靠右行**，对向车道在左；镜像后左右车道语义反了，**标签错**。
> - (iii) **雾**：真实天气，分布仍 plausible，**可用**。
> - (iv) **小块随机遮挡**：模拟污渍/阴影，**可用**。

<br>


An emergency-room ranking model is trained on three months of US hospital data, with ethnicity as an input. It is then deployed in other countries, and it ranks African American and Hispanic patients as less severe than the chart suggests. What fails, and what can be changed? (Fall 2022)

这个分诊模型为什么不能直接拿到别的国家用？族群排序偏差可能从哪来，可以改什么？

> [!EXAMPLE]+ 分布换了；标签里可能已经含有历史待遇差异
>
> 人口结构、病种和就诊方式都变了，美国数据上的输入分布不再成立。
>
> 若训练标签来自历史上对某些族群治疗不足的记录，模型会把这种差异当成「病情较轻」。输入里的族群字段会把这条捷径留住。
>
> 可做的事包括：去掉族群字段、用更少偏倚的标签重新训练、在评估时按族群拆开看召回而不是只看总准确率。去掉字段并不能消除由其他相关变量带来的代理偏差，所以还要分组评估。

<br>


Handwritten letters A–Z, each class balanced. Random horizontal and vertical flips are proposed as augmentation. Separately: KNN with L1 distance, and the same flip applied to every training image and every test image. What happens to accuracy? (Spring 2021)

字母任务能否随机翻转？训练和测试一起翻转时，L1 的 KNN 准确率变不变？

> [!EXAMPLE]+ 随机翻转不合适；训练和测试做同一个置换则 KNN 不变
>
> **随机增强**：水平/垂直翻转会把 **b↔d、p↔q** 等互镜像字母混进错误类，**破坏标签**，不宜随机用。
>
> **KNN + 同一置换**：若对**所有**训练点和**所有**测试点施加**同一个**坐标置换 $\pi$，则 $\|x-y\|_1=\|\pi(x)-\pi(y)\|_1$，近邻集合不变 ⇒ **准确率不变**（不是「随机翻转」）。

<br>


A fuel-economy model is used to choose engine size and weight by maximizing the model’s prediction. That is a sound way to design a car. (Fall 2020)

- T
- F

把回归模型的预测当作设计目标去搜参数，是稳妥的。

> [!EXAMPLE]+ F
>
> 回归学的是 $P(\mathrm{mpg}\mid \mathrm{weight},\mathrm{engine})$ 的**关联**，不是物理因果。
>
> 优化 $\max \widehat{\mathrm{mpg}}(\mathrm{weight},\mathrm{engine})$ 可能得到训练集**从未出现**的极端重量/排量，模型**外推**不可靠，且组合可能不可制造。需要因果/约束优化，不能直接把预测当设计目标。

<br>


The positive class is about one ninth of the data. Oversampling it during training is also applied at test time.

- T
- F

正类大约只占九分之一。训练时把正类过采样。测试时也要这样做。

> [!EXAMPLE]+ F
>
> 训练过采样：复制正类让 batch 里正负比例更平衡，梯度不被 9:1 淹没。
>
> 测试集必须保持**真实先验**（约 1/9 正类）。测试再复制正类 ⇒ 指标按**假频率**算，不能反映部署。流程：**先 split**，再只在 **train** 过采样。

<br>


A criminal-identification system must avoid false imprisonment and still keep the city safe. Which metric?

识别系统既不能把无辜者送进监狱，又要尽量不漏掉应被找到的人。用哪个指标？

> [!EXAMPLE]+ F1，而不是准确率
>
> - **Precision**：判为罪犯的人里有多少真是罪犯 ⇒ 「少冤枉无辜」。
> - **Recall**：真罪犯里有多少被找到 ⇒ 「少漏抓」。
> - **Accuracy**：不平衡时「全判无辜」也能很高，**不适合**此场景。
> - **F1**：precision 与 recall 的调和平均，**同时**惩罚漏抓与冤枉。

<br>


Why add pooling between convolutions if it discards spatial detail?

池化会丢掉空间细节。为什么还要放在卷积之间？

> [!EXAMPLE]+ 缩小特征图，并带来一点平移不变
>
> 1. **计算**：$H,W$ 减半 ⇒ 下一层卷积在更小的网格上算，**FLOPs 降**。
> 2. **感受野**：同样堆叠深度下，输出神经元看到输入的**更大范围**。
> 3. **不变性**：max/avg 在窗口内对**小位移**不敏感（平移 $\le$ 窗口尺度时输出近似不变）。
>
> 代价是**空间分辨率变粗**，细位置信息丢失，故不是每层都该池化。

<br>


The loss curve is completely flat from the start. Give two causes.

损失从第一步起就是平的。两个原因。

> [!EXAMPLE]+ 学习率实际为 0，或者激活已经饱和
>
> **原因一**：有效学习率 $\alpha=0$（或实现里优化器未 step）⇒ $W$ 不变 ⇒ 损失曲线**平**。
>
> **原因二**：**梯度为 0**——例如 sigmoid/tanh 输入很大（饱和，$\sigma'\approx 0$）；ReLU 隐藏层全负（输出 0、导数 0）；或 bug 把 loss 接到常数输入而非 $\hat y$。
>
> 两种都表现为「从第一步起 loss 不动」，排查时先查 lr，再查激活与 loss 接线。

<br>


L1 is sparse, so it is always preferable to L2 even if accuracy drops.

- T
- F

L1 更稀疏、更好解释，所以即使准确率变差也总该优先用 L1。

> [!EXAMPLE]+ F
>
> 稀疏只在「真正起作用的特征很少、且彼此不太相关」时才好解释。特征相关时，L1 会任意留下其中一个、丢掉其余，选出的变量不稳定。准确率下降是否可接受，取决于任务是否真的需要稀疏，而不是一个普遍规则。

<br>


The majority class was downsampled before training. What changes at test time?

训练前对多数类做了下采样。测试时怎么处理？

> [!EXAMPLE]+ 测试分布保持原样，指标不要在下采样后的集合上算
>
> 1. **训练**：对多数类下采样 ⇒ batch 里类别更平衡，**只改训练梯度**。
> 2. **测试**：仍用**真实人口比例**的原始测试集，才能估计部署表现。
> 3. **指标**：不要在「人为下采样后的测试集」上算 accuracy/F1——比例错了，数会误导。
> 4. 若需要校准概率，可在**输出后**按训练先验调整，而不是改测试集组成。

<br>


A fingerprint model exceeds $95\%$ accuracy on a held-out set for the same user, then fails about half the time after deployment. What kind of failure is that?

同一用户的留出集上准确率超过 $95\%$，部署后大约一半次数认证失败。这是哪一类失败？

> [!EXAMPLE]+ 留出集和部署分布不一致
>
> **分布偏移（distribution shift）** 或 **指标/划分问题**：
>
> 1. 若留出集**类别极不平衡**（几乎总是同一用户/同一结果），$95\%$ acc 可能来自**总是猜多数类**。
> 2. 部署时传感器磨损、角度、湿度变 ⇒ **输入分布**与留出集不同 ⇒ 一半失败。
>
> 应对：用与部署一致的指标（如 balanced acc、FRR/FAR）；在新环境数据上**再训练/微调**；留出集高 acc **不能**保证部署。

<br>


# 编码

## 回归网络里的错误

Fall 2022 给出一个输入维 10、隐藏维 20、ReLU、标量输出的 MLP。官方指出的错误如下。

> [!WARNING]+ 行级错误
>
> - 均方误差只做了平方和，没有对样本取平均。应包一层 `np.mean`。系数 $1/2$ 可留可去，但训练和梯度要一致。
> - Dropout 在测试前向里仍然开启。需要训练标志，测试时原样返回。
> - 丢弃概率为 $p$ 时，训练缩放是除以 $1-p$，不是除以 $p$。
> - 掩码下标变量名与 `np.random.choice` 的返回值不一致，索引会在运行时失败。
> - `b1` 应有 20 维，`b2` 应有 1 维。卷面把二者写成了 10 和 20。
> - 前向用了逐元 `*`。输入是 `(batch, 10)`、权重是 `(10, 20)` 时，应使用矩阵乘。
> - 两层线性之间没有 ReLU。
> - 优化器构造时用了未绑定的 `w1` 而不是 `self.w1`。
> - 损失写成了 `mse_loss(x, y)`，第一个自变量应是前向输出，不是输入。

<br>


## 难例挖掘

预测 $\hat y>0.5$ 视为正类。难例定义为：预测类别错误，且 $\lvert y-\hat y\rvert>0.7$。不够数时，只在当前未选中的位置上按均匀概率补齐（Fall 2019）。

```python
misclassified = (y_hat > 0.5) != y
absolute_difference = np.abs(y - y_hat)
mask = misclassified & (absolute_difference > 0.7)
sample_size = int(np.sum(mask))
if sample_size < min_sample_size:
    flip_probs = (~mask) / (len(mask) - sample_size)
    idx = np.random.choice(
        len(mask),
        min_sample_size - sample_size,
        replace=False,
        p=flip_probs,
    )
    mask[idx] = True
```

> [!NOTE]+ 阈值
>
> $\hat y=0.7$ 且 $y=0$：类别错误，但差值恰好是 0.7，若条件是严格大于 0.7，则还不算难例。$\hat y=0.7+\varepsilon$ 才进入掩码。补样本时概率只能放在掩码为假的位置上，并且相加为 1。

<br>


## 全 0 学习率

A 99-way softmax is initialized from $\mathcal N(0,10^{-4})$, the learning rate is 0, and classes are balanced. What is the expected cross-entropy after one epoch?

权重几乎为 0 且完全不更新。99 类交叉熵的期望大约是多少？（Fall 2020）

> [!EXAMPLE]+ $-\log(1/99)=\log 99$
>
> 1. 权重 $\mathcal N(0,10^{-4})$ ⇒ logits $\approx 0$。
> 2. 99 类 softmax 近似均匀：每类概率 $\approx 1/99$。
> 3. 交叉熵 $L=-\log p_{\mathrm{correct}}\approx -\log(1/99)=\log 99$。
> 4. 学习率 $0$ ⇒ 权重不更新，**每个 epoch 期望损失仍是 $\log 99$**（与 $10^{-4}$ 微小差异可忽略）。

<br>


A video tensor has shape $(F,1280,720,3)$. Luminance of a pixel value $x$ is $4(0.2x+1)^{2.2}$. For every frame after the first, form the per-pixel difference from the previous frame. For each difference frame, average the five largest positive changes and the five largest negative changes, and keep the one with larger magnitude, preserving its sign.

视频张量形状为 $(F,1280,720,3)$。亮度是 $4(0.2x+1)^{2.2}$。从第二帧起做逐像素差分。每一帧差分里，五个最大的正变化取平均，五个最负的变化取平均，留下绝对值更大的那一个，并保留符号。

> [!EXAMPLE]+ 通道平均后套亮度公式，再在展平的差分上各取五个极值
>
> 1. `frames` 形状 $(F,H,W,3)$。对 RGB **沿 axis=3 取均值**得灰度，再代入 $L=4(0.2x+1)^{2.2}$ 得每帧亮度图。
> 2. `delta = lum[1:]-lum[:-1]`：共 $F-1$ 帧差分，形状 $(F-1,H,W)$。
> 3. 每帧差分 **reshape** 成一行，长度 $H\cdot W$。
> 4. 正变化：只保留 $>0$ 的像素，**排序**后取**最大 5 个**求平均得 $p$。
> 5. 负变化：只保留 $<0$ 的，取绝对值后排序，**最大 5 个**平均得 $n$。
> 6. 若 $p\ge n$ 输出 $+p$，否则 $-n$；得到 $(F-1,1)$。附录代码若写 `10` 与题干 `5` 不符，以题干为准。

```python
def video_flashinator(frames):
    lum = 4 * (0.2 * frames.mean(axis=3) + 1) ** 2.2
    delta = lum[1:] - lum[:-1]
    flat = delta.reshape(delta.shape[0], -1)
    pos = np.where(flat > 0, flat, 0)
    neg = np.where(flat < 0, -flat, 0)
    pos.sort(axis=1)
    neg.sort(axis=1)
    p = pos[:, -5:].mean(axis=1)
    n = neg[:, -5:].mean(axis=1)
    use_pos = p >= n
    t = np.where(use_pos, p, -n)
    return t.reshape(-1, 1)
```

<br>


# 作业

名称与顺序来自 [2025 Syllabus](https://cs230.stanford.edu/syllabus/) 及 [DLS 完整 GitHub 归档](https://github.com/amanchadha/coursera-deep-learning-specialization)。模块号 `C1M2` 表示专项第 1 门课的第 2 周。Stanford 每季从 DLS 中指定一部分，GitHub 归档则覆盖整个专项；以下按归档逐份列出，不表示每一份都在同一季计分。

## C1 神经网络

### C1M2 Python Basics with NumPy

Implement elementary vectorized utilities without Python loops.

用 NumPy 完成基础向量化函数。

- `basic_sigmoid` 与 `sigmoid`：标量版认识 `math.exp`，数组版使用 `np.exp`；
- `sigmoid_derivative`：由 $s=\sigma(x)$ 得 $s(1-s)$；
- `image2vector`：把 $(H,W,C)$ 改形为 $(HWC,1)$，不改变元素顺序；
- `normalize_rows`：沿 `axis=1` 求 L2 范数，并保持维度以便广播；
- `softmax`：逐行做指数归一化；
- `L1` 与 `L2`：分别求绝对误差和与平方误差和。

> [!EXAMPLE]+ 核心答案
>
> $$\sigma(x)=\frac1{1+e^{-x}},\qquad
> \sigma'(x)=\sigma(x)(1-\sigma(x)).$$
>
> 对 $X\in\mathbb R^{m\times n}$，逐行归一化要使用
> $\lVert X\rVert_2\in\mathbb R^{m\times 1}$，逐行 softmax 要减去每行最大值后再指数化，避免溢出：
>
> $$S_{ij}=\frac{\exp(X_{ij}-\max_k X_{ik})}
> {\sum_k\exp(X_{ik}-\max_kX_{ik})}.$$

<br>


### C1M2 Logistic Regression with a Neural Network Mindset

Build a cat/non-cat classifier from initialization through prediction.

从零完成猫图二分类器。

- 展平 $(H,W,3,m)$ 图像并除以 255；
- 初始化 $w=0,b=0$；
- `propagate`：一次前向计算成本，再求 $dw,db$；
- `optimize`：反复梯度下降并记录成本；
- `predict`：以 $0.5$ 为阈值；
- `model`：串起训练、预测、准确率与学习率实验。

> [!EXAMPLE]+ 形状与梯度
>
> 令 $X\in\mathbb R^{n_x\times m}$、$w\in\mathbb R^{n_x\times1}$：
>
> $$A=\sigma(w^\top X+b),\qquad
> J=-\frac1m\sum_i[y_i\log a_i+(1-y_i)\log(1-a_i)],$$
>
> $$dw=\frac1mX(A-Y)^\top,\qquad
> db=\frac1m\sum_i(a_i-y_i).$$
>
> 所有样本必须一次向量化。循环只用于迭代次数，不能逐样本算梯度。

<br>


### C1M3 Planar Data Classification

Compare logistic regression with a one-hidden-layer network on a nonlinear planar dataset.

在花瓣形数据上比较线性模型与单隐藏层网络。

- 读出 $X,Y$ 的形状与样本数；
- 用 sklearn 逻辑回归建立线性基线；
- `layer_sizes`、`initialize_parameters`；
- `forward_propagation`：tanh 隐藏层与 sigmoid 输出层；
- `compute_cost`、`backward_propagation`、`update_parameters`；
- 改变隐藏单元数，观察欠拟合与过拟合。

> [!EXAMPLE]+ 一层网络反向
>
> $$Z_1=W_1X+b_1,\quad A_1=\tanh Z_1,\quad
> Z_2=W_2A_1+b_2,\quad A_2=\sigma Z_2,$$
>
> $$dZ_2=A_2-Y,\quad
> dW_2=\frac1m dZ_2A_1^\top,$$
>
> $$dZ_1=W_2^\top dZ_2\odot(1-A_1^2),\quad
> dW_1=\frac1m dZ_1X^\top.$$

<br>


### C1M4 Building a Deep Neural Network

Implement reusable $L$-layer forward and backward building blocks.

把深层网络拆成可复用的前向与反向组件。

- `initialize_parameters`：两层版本；
- `initialize_parameters_deep`：按 `layer_dims` 建立 $W^{[l]},b^{[l]}$；
- `linear_forward` 与 `linear_activation_forward`；
- `L_model_forward`：前 $L-1$ 层 ReLU，最后一层 sigmoid；
- `compute_cost`；
- `linear_backward`、`linear_activation_backward`、`L_model_backward`；
- `update_parameters`；
- cache 保存线性输入与激活输入，反向时按相反顺序取回。

> [!EXAMPLE]+ 通用线性层
>
> $$Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]},$$
>
> $$dW^{[l]}=\frac1m dZ^{[l]}A^{[l-1]\top},\quad
> db^{[l]}=\frac1m\sum_i dZ_i^{[l]},\quad
> dA^{[l-1]}=W^{[l]\top}dZ^{[l]}.$$
>
> 初始化通常按 `np.random.randn(...) / sqrt(n_prev)` 或 He 缩放，不能把同一层所有权重设成同一值。

<br>


### C1M4 Deep Neural Network Application

Train a 2-layer model and an $L$-layer model for cat classification, then inspect errors.

复用上一份组件，比较浅层与深层猫图分类器。

- 两层结构：`LINEAR -> RELU -> LINEAR -> SIGMOID`；
- 深层结构：`[LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID`；
- 画训练成本；
- 在训练集与测试集上预测；
- 查看误分类图片，判断角度、背景、亮度、遮挡等误差来源。

> [!NOTE]+ 解析
>
> 这一份不再考单个导数，而是考组件接口、形状与 cache 能否组合。成本下降但测试准确率停滞时，应检查泛化误差，不应只增加迭代次数。

<br>


## C2 训练技巧

### C2M1 Initialization

Compare zero, random-large, and He initialization on the same classifier.

比较全零、大随机数与 He 初始化。

- `initialize_parameters_zeros`；
- `initialize_parameters_random`；
- `initialize_parameters_he`；
- 对比损失曲线与决策边界。

> [!EXAMPLE]+ He 初始化
>
> 对 ReLU 层：
>
> $$W^{[l]}\sim\mathcal N\!\left(0,\frac{2}{n_{l-1}}\right),\qquad
> W^{[l]}=\operatorname{randn}\sqrt{\frac2{n_{l-1}}}.$$
>
> 全零会保留隐藏单元对称性。过大的随机数把 tanh/sigmoid 推入饱和区。He 让 ReLU 激活的方差跨层大致稳定。

<br>


### C2M1 Regularization

Add L2 regularization and inverted dropout to a three-layer network.

在三层网络中分别实现 L2 与 inverted dropout。

- `compute_cost_with_regularization`；
- `backward_propagation_with_regularization`；
- `forward_propagation_with_dropout`；
- `backward_propagation_with_dropout`；
- 比较无正则、L2、dropout 的决策边界。

> [!EXAMPLE]+ 两条关键公式
>
> $$J_{\mathrm{reg}}=J+\frac{\lambda}{2m}\sum_l\lVert W^{[l]}\rVert_F^2,\qquad
> dW^{[l]}\leftarrow dW^{[l]}+\frac{\lambda}{m}W^{[l]}.$$
>
> 若保留概率为 $k$：
>
> $$D^{[l]}\sim\mathrm{Bernoulli}(k),\qquad
> A^{[l]}\leftarrow A^{[l]}\odot D^{[l]}/k.$$
>
> 反向使用同一份 $D^{[l]}$，测试时关闭 dropout。

<br>


### C2M1 Gradient Checking

Verify scalar and deep-network gradients with centered finite differences.

用中心差分检查标量函数与多层网络的反向传播。

- 标量 `forward_propagation` 与 `backward_propagation`；
- `gradient_check`；
- 字典参数展平成向量；
- 对每个坐标构造 $\theta^+,\theta^-$；
- `gradient_check_n` 对整个网络检查；
- 从异常差异定位反向传播中的符号或系数错误。

> [!EXAMPLE]+ 判据
>
> $$g_{\mathrm{approx},i}
> =\frac{J(\theta+\varepsilon e_i)-J(\theta-\varepsilon e_i)}{2\varepsilon},$$
>
> $$\mathrm{difference}
> =\frac{\lVert g-g_{\mathrm{approx}}\rVert_2}
> {\lVert g\rVert_2+\lVert g_{\mathrm{approx}}\rVert_2}.$$
>
> 常取 $\varepsilon=10^{-7}$。约 $10^{-7}$ 或更小通常可信，$10^{-2}$ 量级应视为实现错误。Dropout 有随机性，做 gradient check 时必须关掉或固定掩码。

<br>


### C2M2 Optimization Methods

Implement mini-batches, momentum, and Adam, then compare convergence.

实现随机 mini-batch、momentum 与 Adam。

- 普通梯度下降更新；
- `random_mini_batches`：同一个 permutation 同时打乱 $X,Y$，处理最后一个不足整批的余数；
- `initialize_velocity` 与 momentum 更新；
- `initialize_adam` 与 Adam 更新；
- 比较优化器损失曲线。

> [!EXAMPLE]+ Adam
>
> $$v_t=\beta_1v_{t-1}+(1-\beta_1)g_t,\qquad
> s_t=\beta_2s_{t-1}+(1-\beta_2)g_t^2,$$
>
> $$\hat v_t=\frac{v_t}{1-\beta_1^t},\qquad
> \hat s_t=\frac{s_t}{1-\beta_2^t},\qquad
> \theta_t=\theta_{t-1}-\alpha\frac{\hat v_t}{\sqrt{\hat s_t}+\epsilon}.$$
>
> 偏差修正里的 $t$ 是更新次数，不是 epoch 编号。

<br>


### C2M3 TensorFlow Introduction

Rebuild a small classifier with TensorFlow 2 and automatic differentiation.

用 TensorFlow 2 重写小型分类器。

- `linear_function`：随机 $W,X,b$ 后做矩阵乘；
- `sigmoid`；
- `one_hot_matrix`；
- `initialize_parameters`：Glorot 初始化与 `tf.Variable`；
- `forward_propagation`；
- `compute_cost`；
- `tf.data.Dataset` 的 batch 与 prefetch；
- `GradientTape`、`optimizer.apply_gradients` 与训练循环。

> [!NOTE]+ 解析
>
> 框架替代的是手工求导，不替代形状推理。`GradientTape` 只追踪张量运算；参数必须是可训练变量。多分类 logits 直接交给交叉熵，不能先手动 softmax 再给需要 logits 的接口。

<br>


## C3 项目策略

C3 没有编程 notebook，公开归档只有两个案例测验。

### C3M1 Bird Recognition in Peacetopia

Design a metric and diagnose avoidable bias, variance, and data mismatch for a city bird detector.

为城市鸟类检测器选择指标，并区分偏差、方差与数据不匹配。

- 把多个目标压成单一优化指标，把时延、内存等设成满足指标；
- dev 与 test 必须来自相同目标分布；
- 人类水平近似贝叶斯误差；
- 训练误差减人类误差是可避免偏差；
- dev 误差减训练误差是方差；
- train-dev 与 dev 的差异揭示数据不匹配。

> [!EXAMPLE]+ 诊断顺序
>
> 若人类 $1\%$、训练 $2\%$、train-dev $3\%$、dev $10\%$，则可避免偏差约 $1\%$，方差约 $1\%$，主要问题是分布不匹配约 $7\%$。此时盲目加正则会处理错对象。

<br>


### C3M2 Autonomous Driving

Decide when to use transfer learning, multi-task learning, end-to-end learning, and error analysis.

在自动驾驶案例中判断迁移学习、多任务、端到端与误差分析。

- 迁移学习：源任务数据多、底层特征可复用、目标任务数据少；
- 多任务：多个任务共享输入与低层特征，且每个任务都有足够标签；
- 端到端：数据极多且输入到输出映射能覆盖中间知识；
- 误差分析：先人工统计最大错误类别，再决定采集数据或改模型；
- 合成数据：必须缩小目标分布误差，而不是只增加容易样本。

<br>


## C4 卷积网络

### C4M1 Convolutional Model: Step by Step

Implement convolution and pooling forward/backward passes in NumPy.

用 NumPy 从零实现卷积与池化的前向、反向。

- `zero_pad`；
- `conv_single_step`；
- `conv_forward`；
- `pool_forward`；
- `conv_backward`；
- `create_mask_from_window`；
- `distribute_value`；
- `pool_backward`。

> [!EXAMPLE]+ 卷积反向的三处累加
>
> 对输出位置 $(i,h,w,c)$ 的梯度 $dZ$：
>
> $$dA_{\mathrm{prev}}[\text{slice}]\mathrel{+}=W_c\,dZ,\quad
> dW_c\mathrel{+}=A_{\mathrm{prev}}[\text{slice}]\,dZ,\quad
> db_c\mathrel{+}=dZ.$$
>
> Max-pool 反向把梯度送到窗口最大值的 mask；average-pool 反向把标量均匀分成 $1/(f^2)$。

<br>


### C4M1 Convolutional Neural Networks: Application

Build a sign-language classifier with TensorFlow/Keras Conv2D, pooling, flattening, and dense output.

用框架搭建手势分类 CNN。

- 顺序模型版的卷积、ReLU、最大池化、展平、全连接；
- Functional API 版前向；
- 编译模型，检查参数量；
- 训练并比较 train/dev 曲线。

> [!NOTE]+ 解析
>
> `Conv2D(filters=8, kernel_size=4, strides=1, padding="same")` 的输出通道是 8，参数量是 $(4\cdot4\cdot C_{\mathrm{in}}+1)\cdot8$。池化改变尺寸，不增加参数。

<br>


### C4M2 Keras Happy House

Build a binary smile detector with Keras.

为“是否微笑”搭建二分类 CNN。

- `ZeroPadding2D`、`Conv2D`、BatchNorm、ReLU、MaxPool、Flatten、Dense；
- 编译与训练；
- 用自定义图片做预处理和预测。

> [!EXAMPLE]+ 输出
>
> 二分类标准头：**1 个神经元 + sigmoid + `binary_crossentropy`**，标签 shape $(m,1)$ 取 0/1。
>
> 若误用 **2 神经元 + softmax**，标签需 one-hot $(m,2)$，损失用 `categorical_crossentropy`。
>
> 不能 softmax 两单元却喂标量 BCE 标签，否则 shape 与概率语义都不对。

<br>


### C4M2 Residual Networks

Implement identity and convolutional residual blocks, then assemble ResNet-50.

实现恒等残差块、卷积残差块并拼成 ResNet-50。

- `identity_block`：捷径尺寸不变；
- `convolutional_block`：捷径用 $1\times1$ 卷积改变尺寸或通道；
- 主支路通常是 $1\times1\to f\times f\to1\times1$；
- 加法在最终 ReLU 之前；
- 建立 stages 并做图像分类。

> [!EXAMPLE]+ 残差目标
>
> $$A^{[l+2]}=g(F(A^{[l]})+A^{[l]}).$$
>
> 若 $F$ 学成 0，模块退化为恒等映射，深层网络至少容易保留已有表示。加法要求两支形状一致；不一致时捷径必须投影。

<br>


### C4M2 Transfer Learning with MobileNet

Adapt a pretrained MobileNetV2 to an Alpaca/not-Alpaca classifier.

用 MobileNetV2 做羊驼二分类迁移学习。

- 数据增强管道；
- 加载 ImageNet 权重且去掉顶层；
- 冻结 base model；
- 全局平均池化、dropout、分类头；
- 只训练分类头；
- 解冻靠后的层，小学习率微调；
- 保持 BatchNorm 的推理语义。

> [!EXAMPLE]+ 两阶段训练
>
> **阶段 1**：`base_model.trainable=False`，只训练 GAP + Dropout + Dense 头。随机头的梯度不会回传到 ImageNet 权重。
>
> **阶段 2**：解冻 base **最后几层**，**lr 降 10×～100×** 微调，BN 层在推理模式与训练模式间与 Keras 约定一致。
>
> 小数据若**一次解冻全部层**，预训练特征易被大梯度冲掉，**快速过拟合**。

<br>


### C4M3 Car Detection with YOLO

Decode YOLO outputs, filter low-confidence boxes, and apply non-max suppression.

解码 YOLO 输出并完成后处理。

- `yolo_filter_boxes`；
- `iou`；
- `yolo_non_max_suppression`；
- `yolo_eval`；
- 把网络坐标缩放回原图。

> [!EXAMPLE]+ 分数与 IoU
>
> 每个框的类别分数是
> $p_c\cdot p(\text{class}\mid\text{object})$。先按阈值过滤，再按类别做 NMS。
>
> $$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.$$
>
> NMS 保留当前最高分框，删除与它 IoU 超阈值的同类框，然后继续。不同类别不能互相压掉。

<br>


### C4M3 Image Segmentation with U-Net

Assemble contracting and expanding paths for pixelwise segmentation.

搭建 U-Net 的下采样与上采样路径。

- `conv_block`：两次卷积，可选 dropout，再最大池化；
- `upsampling_block`：转置卷积上采样，与同分辨率编码器特征拼接，再卷积；
- `unet_model`；
- 最后用 $1\times1$ 卷积把通道变成类别数；
- 查看预测掩码。

> [!EXAMPLE]+ 跳连不是残差相加
>
> U-Net 跳连通常在通道维 `concatenate`，把定位细节交给解码器；ResNet 则逐元素相加。若空间尺寸因 padding 不一致，拼接前必须裁剪或统一使用 same padding。

<br>


### C4M4 Neural Style Transfer

Optimize pixels of a generated image using content and style losses from VGG-19.

固定 VGG-19，直接优化生成图像像素。

- `compute_content_cost`；
- `gram_matrix`；
- `compute_layer_style_cost`；
- 多层 `compute_style_cost`；
- `total_cost`；
- `train_step` 更新生成图，而不是 VGG 权重。

> [!EXAMPLE]+ 三部分损失
>
> $$J_C=\frac{1}{4n_Hn_Wn_C}\lVert a^{(C)}-a^{(G)}\rVert_F^2,$$
>
> $$G=A_{\mathrm{flat}}A_{\mathrm{flat}}^\top,\qquad
> J_S^{[l]}\propto\lVert G^{(S)}-G^{(G)}\rVert_F^2,$$
>
> $$J=\alpha J_C+\beta J_S.$$
>
> Gram 矩阵抹去绝对位置，只保留通道激活之间的相关性，因此能描述纹理风格。

<br>


### C4M4 Face Recognition

Use FaceNet encodings for one-shot verification and recognition.

用 FaceNet 的 128 维编码完成验证与识别。

- `triplet_loss`；
- `verify`：输入姓名与照片，判断是否同一人；
- `who_is_it`：在数据库中找最近身份；
- 区分 verification（一对一）与 recognition（一对多）。

> [!EXAMPLE]+ 三元组损失
>
> $$L=\max\bigl(
> \lVert f(A)-f(P)\rVert_2^2
> -\lVert f(A)-f(N)\rVert_2^2+\alpha,\ 0
> \bigr).$$
>
> 验证只比较一个登记向量与阈值；识别要遍历数据库找最小距离，并且最小值仍须小于阈值，否则应返回未知身份。

<br>


## C5 序列模型

### C5M1 Building an RNN Step by Step

Implement vanilla RNN and LSTM cells, sequence forward passes, and optional BPTT.

用 NumPy 实现 RNN、LSTM 单元与整段前向，扩展题做时间反传。

- `rnn_cell_forward`、`rnn_forward`；
- `lstm_cell_forward`、`lstm_forward`；
- 可选 `rnn_cell_backward`、`rnn_backward`；
- 可选 `lstm_cell_backward`、`lstm_backward`。

> [!EXAMPLE]+ LSTM 单元
>
> $$f_t=\sigma(W_f[a_{t-1},x_t]+b_f),\quad
> i_t=\sigma(W_i[a_{t-1},x_t]+b_i),$$
>
> $$\tilde c_t=\tanh(W_c[a_{t-1},x_t]+b_c),\quad
> c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,$$
>
> $$o_t=\sigma(W_o[a_{t-1},x_t]+b_o),\qquad
> a_t=o_t\odot\tanh c_t.$$
>
> 时间反传时，$da_{t-1}$ 与 $dc_{t-1}$ 都要从当前步累加到上一时刻，不能覆盖。

<br>


### C5M1 Dinosaur Land

Train a character-level RNN that generates dinosaur names.

训练字符级 RNN 生成恐龙名。

- `clip`：逐元素裁剪梯度；
- `sample`：从 softmax 分布采样下一字符；
- `optimize`：前向、损失、反向、裁剪、更新；
- `model`：打乱名字，迭代训练并周期性采样。

> [!EXAMPLE]+ 采样而不是 argmax
>
> 每一步从 $p_t=\mathrm{softmax}(W_ya_t+b_y)$ 按概率抽字符。始终 argmax 会生成确定、单调的名字，无法反映模型分布。梯度裁剪
> $g\leftarrow\min(\max(g,-c),c)$
> 用于抑制长序列的梯度爆炸。

<br>


### C5M1 Jazz Improvisation with LSTM

Train an LSTM on musical values and build an inference graph that generates a sequence.

用 LSTM 学习爵士片段，并建立自回归推理模型。

- 训练模型把长度 $T_x$ 的输入映射到每个时刻的音符分布；
- `djmodel`；
- `music_inference_model`：把当前输出转为下一步输入；
- `predict_and_sample`：把概率转为 one-hot 序列；
- 将生成序列写成 MIDI。

> [!NOTE]+ 解析
>
> 训练时常使用真实前一音符，即 teacher forcing；推理时只能喂回模型刚生成的音符。两种输入分布不同，是 exposure bias 的来源。

<br>


### C5M2 Word Vector Representation and Debiasing

Implement cosine similarity, analogy completion, neutralization, and equalization.

做词向量相似度、类比与偏差消除。

- `cosine_similarity`；
- `complete_analogy`；
- 用词对差向量估计偏差方向 $g$；
- `neutralize` 中性词；
- `equalize` 成对的性别相关词。

> [!EXAMPLE]+ 中和
>
> $$e_{\mathrm{bias}}=\frac{e\cdot g}{\lVert g\rVert_2^2}g,\qquad
> e_{\mathrm{debiased}}=e-e_{\mathrm{bias}}.$$
>
> `equalize` 不能简单把两个词都投影到 $g^\perp$；它要保留二者合理的性别差异，同时让二者到中性词的距离对称。

<br>


### C5M2 Emojify

Map short sentences to emoji labels with an average-embedding baseline and an LSTM.

把短句分类为 emoji，先做均值词向量基线，再做 LSTM。

- `sentence_to_avg`；
- `model`：均值词向量接 softmax；
- `sentences_to_indices`；
- `pretrained_embedding_layer`；
- 双层 LSTM、dropout 与 dense softmax；
- 比较词序无关基线和序列模型。

> [!EXAMPLE]+ OOV 与 padding
>
> 句子索引矩阵应固定到最大长度，短句补 0，长句截断。未知词不能直接索引字典；应跳过、映射到 OOV，或使用可训练未知向量。均值基线看不到词序，所以 “not happy” 容易被 “happy” 的向量主导。

<br>


### C5M3 Neural Machine Translation with Attention

Convert human-readable dates to machine-readable dates with an attention model.

用注意力模型把自然语言日期转成标准日期。

- 双向 LSTM 编码输入；
- `one_step_attention`；
- 解码器每个输出时刻重新计算注意力；
- 模型有 $T_y$ 个输出；
- 可视化注意力图。

> [!EXAMPLE]+ 一步注意力
>
> $$e^{\langle t,t'\rangle}
> =v^\top\tanh(W_a a^{\langle t'\rangle}+W_s s^{\langle t-1\rangle}+b),$$
>
> $$\alpha^{\langle t,t'\rangle}
> =\frac{\exp e^{\langle t,t'\rangle}}
> {\sum_k\exp e^{\langle t,k\rangle}},\qquad
> c^{\langle t\rangle}=\sum_{t'}\alpha^{\langle t,t'\rangle}a^{\langle t'\rangle}.$$
>
> softmax 必须沿输入时间轴归一化，使每个输出位置对全部输入位置的权重和为 1。

<br>


### C5M3 Trigger Word Detection

Synthesize audio training examples and detect an activation word in spectrograms.

合成语音训练样本，并在频谱图上检测唤醒词。

- `get_random_time_segment`；
- `is_overlapping`；
- `insert_audio_clip`；
- `insert_ones`：唤醒词结束后的若干时间步标 1；
- `create_training_example`；
- 建立 1-D Conv、GRU、dropout、BatchNorm 模型；
- 对真实音频输出触发概率。

> [!EXAMPLE]+ 标签为何延长
>
> 频谱图时间轴上，若仅在唤醒词**结束那一格**标 $y=1$，正类只占 $1/T$，BCE 极不平衡，GRU 难学。
>
> 作业在结束点之后约 **50 个时间步** 仍标 1，相当于放宽「正类窗口」，增加正梯度。
>
> 合成时 `insert_audio_clip` 必须先 `is_overlapping` 检查，避免两段音频重叠但标签未重叠。

<br>


### C5M4 Transformer Network

Implement positional encoding, attention masks, encoder layers, decoder layers, and a full Transformer.

实现位置编码、掩码、编码器层、解码器层与完整 Transformer。

- `get_angles`、`positional_encoding`；
- `create_padding_mask` 与 `create_look_ahead_mask`；
- scaled dot-product attention；
- `EncoderLayer` 与 `Encoder`；
- `DecoderLayer` 与 `Decoder`；
- 最终线性层输出词表 logits。

> [!EXAMPLE]+ 缩放与掩码
>
> $$\mathrm{Attention}(Q,K,V)
> =\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$
>
> 被遮住的位置在 $M$ 中加一个极大负数，softmax 后概率约为 0。除以 $\sqrt{d_k}$ 是为了防止维度升高后点积方差过大、softmax 饱和。

<br>


### C5 Transformer Applications

新版归档还包含两个应用 notebook，它们未必是每季 CS230 的计分作业。

**Named-Entity Recognition**

- 使用 Hugging Face tokenizer 与 token-classification 模型；
- 把词级标签对齐到 subword；
- padding 与特殊 token 的标签设为忽略索引；
- 以实体级 F1 评估，而非只看 token accuracy。

**Question Answering**

- 输入问题与上下文；
- tokenizer 生成 offset mapping；
- 把字符级答案起止位置映射到 token 起止位置；
- 模型分别输出 start logits 与 end logits；
- 推理时选择合法且长度受限的起止组合。

> [!NOTE]+ 子词标签
>
> 一个词拆成多个 subword 后，只有首个 subword 继承实体标签、其余位置忽略，或让全部子词继承一致的 IOB 标签；训练与评估必须使用同一约定。

> [!INFO]+ 截止时间
>
> 编程作业与测验按 CS230 syllabus 的时间交，不按 Coursera 页面上的时间。近年安排是讲座开始前 30 分钟。周测验的题目在课程邀请的 Coursera 会话里，不在公开 PDF 中。

<br>


# 出处

1. Stanford CS230. *Practice Midterm 1 Solutions*. [https://cs230.stanford.edu/section/6/Practice_1_Sol.pdf](https://cs230.stanford.edu/section/6/Practice_1_Sol.pdf)
2. Stanford CS230. *Practice Midterm 2 Solutions*（Fall 2019 期中）. [https://cs230.stanford.edu/section/6/Practice_2_Sol.pdf](https://cs230.stanford.edu/section/6/Practice_2_Sol.pdf)
3. Stanford CS230. *Midterm Examination, Fall 2020, solutions*. [https://cs230.stanford.edu/files/cs230exam_fall20_soln.pdf](https://cs230.stanford.edu/files/cs230exam_fall20_soln.pdf)
4. Stanford CS230. *Midterm Examination, Winter 2021, solutions*. [https://cs230.stanford.edu/files/cs230exam_win21_soln.pdf](https://cs230.stanford.edu/files/cs230exam_win21_soln.pdf)
5. Stanford CS230. *Midterm Examination, Spring 2021*. [https://cs230.stanford.edu/syllabus/fall_2021/cs230exam_spr21.pdf](https://cs230.stanford.edu/syllabus/fall_2021/cs230exam_spr21.pdf)
6. Stanford CS230. *Midterm Examination, Fall 2022, solutions*. [https://cs230.stanford.edu/syllabus/fall_2022/CS230_Midterm_Solutions_Fall_2022.pdf](https://cs230.stanford.edu/syllabus/fall_2022/CS230_Midterm_Solutions_Fall_2022.pdf)
7. Stanford CS230. *Syllabus*. [https://cs230.stanford.edu/syllabus/](https://cs230.stanford.edu/syllabus/)
8. Stanford CS230. *Section 6*. [https://cs230.stanford.edu/section/6/](https://cs230.stanford.edu/section/6/)
9. Stanford CS230. *Code Examples*. [https://github.com/cs230-stanford/cs230-code-examples](https://github.com/cs230-stanford/cs230-code-examples)
10. maxim5. *CS230 2018 Autumn Archive*. [https://github.com/maxim5/cs230-2018-autumn](https://github.com/maxim5/cs230-2018-autumn)
11. Aman Chadha. *Deep Learning Specialization: Notes, Assignments and Quizzes*. [https://github.com/amanchadha/coursera-deep-learning-specialization](https://github.com/amanchadha/coursera-deep-learning-specialization)
12. Afshine Amidi, Shervine Amidi. *Stanford CS230 Deep Learning Cheatsheets*. [https://github.com/afshinea/stanford-cs-230-deep-learning](https://github.com/afshinea/stanford-cs-230-deep-learning)
13. kozemzak. *CS230 Midterm Cheat Sheet*. [https://github.com/kozemzak/cs230-cheat-sheet](https://github.com/kozemzak/cs230-cheat-sheet)
