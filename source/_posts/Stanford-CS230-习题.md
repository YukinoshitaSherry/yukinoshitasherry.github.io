---
title: 斯坦福CS230：公开试题
date: 2026-10-09
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS230 公开期中与练习卷的中英对照题：选择、损失、优化、卷积、反向传播、GAN 与编程作业目标，附解析。
hidden: true
---


- **对应课程**
  - [Stanford CS230：Deep Learning](https://cs230.stanford.edu/)；
  - 课堂笔记见同目录 `Stanford-CS230.md`。
- **计分（课程主页）**
  - 期末项目 $45\%$；
  - 期中（笔试，约 $180$ 分钟）$35\%$；
  - 编程作业 $12\%$；
  - 周测验 $8\%$。
- **检索范围**
  - 课程官网公开的练习卷与历年期中解答；
  - [Syllabus](https://cs230.stanford.edu/syllabus/) 上列出的编程作业名称。
- **未收入**
  - Coursera 周测验全文与答案键仍在课程会话内，不在此复述；
  - 编程作业的笔记本源码与自动评分参考实现不在此粘贴。下列「作业」一节只保留公开名称与应对应能独立写出的公式。

<br>


# 来源

| 卷 | 公开位置 |
| :--- | :--- |
| Practice Midterm 1（Winter 2020 练习卷） | [Practice_1_Sol.pdf](https://cs230.stanford.edu/section/6/Practice_1_Sol.pdf) |
| Practice Midterm 2（Fall 2019 期中，Section 6 作练习） | [Practice_2_Sol.pdf](https://cs230.stanford.edu/section/6/Practice_2_Sol.pdf) |
| Fall 2020 Midterm | [cs230exam_fall20_soln.pdf](https://cs230.stanford.edu/files/cs230exam_fall20_soln.pdf) |
| Winter 2021 Midterm | [cs230exam_win21_soln.pdf](https://cs230.stanford.edu/files/cs230exam_win21_soln.pdf) |
| Spring 2021 Midterm | [cs230exam_spr21.pdf](https://cs230.stanford.edu/syllabus/fall_2021/cs230exam_spr21.pdf) |
| Fall 2022 Midterm | [CS230_Midterm_Solutions_Fall_2022.pdf](https://cs230.stanford.edu/syllabus/fall_2022/CS230_Midterm_Solutions_Fall_2022.pdf) |
| 入口 | [Section 6：Practice Midterm](https://cs230.stanford.edu/section/6/) |

> [!INFO]+ 使用方式
>
> 题干按公开卷改写为中英对照，数值、维度与官方答案保持一致。解析按推导重写，不替代官网 PDF。Spring 2021 公开的是试题 PDF，该卷选择题答案由推导给出，并在对应卡片中标明。

<br>


# 仓库

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

> [!WARNING]+ 版本与 Honor Code
>
> CS230 当前 syllabus 的截止任务才是当季范围；GitHub 归档同时混有旧版、改版和 DLS 全专项内容，不能把仓库目录当作某一学期的正式清单。
>
> 课程主页明确指出：在读学生不得查看往年书面或代码答案，也不得公开提交代码。下文只把 notebook 改写成复习题、目标与公式，不粘贴可直接提交的完整函数。

> [!NOTE]+ 官方项目模板的目录逻辑
>
> `data/` 明确分成 train/dev/test；`model/` 放模型；`train.py` 训练并逐轮看 dev；`search_hyperparams.py` 启动多组实验；`synthesize_results.py` 汇总实验；`evaluate.py` 只在最终方案确定后运行一次 test。这个结构比把全部逻辑塞进 notebook 更接近课程项目要求。

<br>


# 选择

## 参数与卷积

**EN.** A 2-layer network has 5 units in the input, the hidden layer, and the output layer, with biases. The parameter count is 60.

**中.** 输入、隐藏层、输出层宽度均为 5，且含偏置。参数总数为 60。判断正误。

> [!EXAMPLE]+ 正。计数为 $5\times 5+5+5\times 5+5=60$
>
> 两层均含权重与偏置。官方卷把「每层 5 个神经元」按这一宽度理解，结论为 True（Fall 2019 / Practice 2）。

**EN.** Input volume $(10,10,3)$. Compare (A) a fully connected layer with 2 units and biases, after flattening, and (B) a conv layer with three $2\times 2$ filters, padding 0, stride 2, and biases. How many fewer parameters does the conv layer have?

**中.** 输入 $(10,10,3)$。全连接层 2 个神经元、含偏置，输入先展平。卷积层为 3 个 $2\times 2$ 滤波器、padding 0、stride 2、含偏置。卷积比全连接少多少个参数？

> [!EXAMPLE]+ 少 563 个
>
> 展平后长度 $300$。全连接：$300\times 2+2=602$。
>
> 每个卷积核：$2\times 2\times 3+1=13$，三个核共 $39$。
>
> $602-39=563$。输出空间尺寸 $\lfloor(10-2)/2\rfloor+1=5$，体积为 $(5,5,3)$。

**EN.** NumPy broadcasting: which products `v * u` succeed?
>
> (i) $(5,1)$ and $(5,1)$
> (ii) $(1,5)$ and $(50,5)$
> (iii) $(1,1)$ and $(50,5)$
> (iv) $(1,5)$ and $(1,50)$

**中.** 下列 `v * u` 哪些能广播成功？

> [!EXAMPLE]+ (i)(ii)(iii)
>
> (iv) 的最后一维是 $5$ 对 $50$，既不相等也不能向 $1$ 对齐，广播失败。

## 梯度与激活

**EN.** Which statements about vanishing gradients are true?
>
> (i) Tanh does not suffer from vanishing gradients, so it is preferred to sigmoid.
> (ii) Vanishing gradients make deeper layers learn more slowly than earlier layers.
> (iii) Leaky ReLU is less prone to vanishing gradients than sigmoid.
> (iv) Xavier initialization can help.

**中.** 关于梯度消失，哪些成立？

> [!EXAMPLE]+ (iii)(iv)
>
> tanh 在饱和区导数仍接近 0，只是零点附近导数为 1，比 sigmoid 的 $0.25$ 更陡，并不能说「不会消失」。
>
> 梯度是从输出往输入传。消失时，**更靠近输入**的层更新更慢，而不是更深的层更慢。
>
> Leaky ReLU 在负半轴仍有常数斜率。Xavier 让各层激活方差更稳定，减轻饱和。

**EN.** If $z=\tanh(x)$, then $\partial z/\partial x=1-z^2$. The backpropagated gradient through tanh is always smaller than or equal in magnitude to the upstream gradient.

**中.** 经过 tanh 反传后，梯度幅度总是不大于上游梯度。判断正误。

> [!EXAMPLE]+ 正
>
> $1-z^2\in(0,1]$。上游梯度只被乘上一个不超过 1 的正数，幅度不增、符号不变。$z\to\pm 1$ 时该因子趋于 0。

**EN.** A trained logistic regression has weight $W$, no bias, and test accuracy $A$. Replacing $W$ by $W/2$ leaves $A$ unchanged.

**中.** 无偏置的逻辑回归，把 $W$ 除以 2，测试准确率不变。判断正误。

> [!EXAMPLE]+ 正
>
> 判决只看 $W^\top x$ 的符号。正数缩放不改变符号，因而 $\sigma(W^\top x)\ge 0.5$ 的区域不变。若同时存在偏置且偏置不缩放，边界会移动。

**EN.** A 100-layer net uses sigmoid at the output and a mix of tanh and ReLU elsewhere. After one epoch some layers have essentially zero gradients, while the loss is still a reasonable magnitude. Which fixes help?
>
> (i) A larger training set.
> (ii) Leaky ReLU in place of ReLU.
> (iii) BatchNorm before every activation.
> (iv) A larger learning rate.

**中.** 百层网络中，部分层梯度在第一轮后几乎为 0，损失数量级仍正常。哪些处理有帮助？（Fall 2020）

> [!EXAMPLE]+ (ii)(iii)
>
> 这是局部的梯度消失或 ReLU 死亡，不是数据量不足。学习率乘在已经接近 0 的梯度上，解决不了该层没有信号的问题。
>
> Leaky ReLU 让负区间仍能回传。BatchNorm 把 tanh 的输入拉回靠近 0 的区域，该处导数远离 0。

**EN.** Which maps are usable elementwise nonlinearities in practice?
>
> (i) $f(x)=-\min(2,x)$
> (ii) $f(x)=0.9x+1$
> (iii) $f(x)=\min(x,0.1x)$ for every $x$
> (iv) $f(x)=\max(x,0.1x)$ if $x\ge 0$, and $\min(x,0.1x)$ if $x<0$

**中.** 哪些能当作实际使用的逐元非线性？

> [!EXAMPLE]+ (i)(iii)
>
> (ii) 是仿射函数。(iv) 在正半轴是 $x$、负半轴是 $0.1x$，整体仍是过原点的分段线性，且两边斜率都为正，复合后仍只是线性缩放，没有把半空间折起来的非线性。
>
> (i) 在 $x=2$ 处弯折。(iii) 即 $f(x)=x$ 当 $x<0$、$f(x)=0.1x$ 当 $x\ge 0$，正负半轴斜率不同。官方答案取 (i)(iii)。

**EN.** As the gradient flows backward through a sigmoid, it always:

**中.** 梯度反向经过 sigmoid 时，幅度与符号如何变化？

> [!EXAMPLE]+ 幅度变小，符号不变
>
> $\sigma'(z)=\sigma(z)(1-\sigma(z))\in(0,1/4]$。因子严格为正且严格小于 1，故幅度下降、极性不变。

**EN.** A step function $f(x)=1$ for $x\ge 0$ and $0$ otherwise is recommended as a conv nonlinearity with Adam. Is that usable?

**中.** 阶跃函数能否作为卷积网络的非线性，并用 Adam 训练？

> [!EXAMPLE]+ 不能
>
> 除原点外导数处处为 0，基于梯度的更新几乎收不到信号。

## 正则、Dropout 与优化器

**EN.** Which reduce overfitting?
>
> (i) Data augmentation. (ii) Dropout. (iii) BatchNorm. (iv) Adam instead of SGD.

**中.** 哪些能减轻过拟合？（Fall 2020）

> [!EXAMPLE]+ (i)(ii)(iii)
>
> Adam 改变的是更新规则，不是假设空间的约束。数据增强扩大有效样本，Dropout 在训练时随机子网络，BatchNorm 的 mini-batch 噪声带来弱正则。

**EN.** Dropout on a net with $N$ units, each dropped independently. How many distinct sub-networks can appear?
>
> (i) $\lfloor Np\rfloor$ (ii) $(\lfloor Np\rfloor)^N$ (iii) $2^{\lfloor Np\rfloor}$ (iv) $2^N$

**中.** $N$ 个单元各自可保留或丢弃。不同子网络共有多少个？（Fall 2022）

> [!EXAMPLE]+ $2^N$
>
> 每个单元两种状态，与丢弃概率 $p$ 无关。$p$ 只决定各子网络被抽到的概率，不决定种类数。

**EN.** At test time, inverted dropout with drop probability $p$ will:
>
> (i) multiply by $p$ (ii) drop units with probability $p$ (iii) divide by $p$ (iv) none of these

**中.** Inverted dropout 在测试时做什么？

> [!EXAMPLE]+ 都不做
>
> 训练时把保留下来的激活除以保留概率 $1-p$，测试时关闭随机丢弃，也不再缩放。官方卷里「除以 $p$」与「除以 $1-p$」容易混：丢弃概率是 $p$ 时，训练缩放因子是 $1/(1-p)$。

**EN.** Which are true of momentum?
>
> (i) It has no hyperparameters.
> (ii) A running average of gradients helps move through ravines and poor local regions.
> (iii) A network with $n$ parameters needs $\Theta(n)$ extra stored values.

**中.** 关于 momentum，哪些成立？（Spring 2021，推导）

> [!EXAMPLE]+ (ii)(iii)
>
> 学习率与 $\beta$ 都是超参数。速度 $v$ 与参数同形状，额外存储是 $\Theta(n)$。

**EN.** Early stopping:
>
> (i) can reduce the need to hand-tune the epoch budget
> (ii) increases variance
> (iii) triggers when validation accuracy reaches a trough
> (iv) can stop earlier than the training-loss minimum

**中.** 关于 early stopping，哪些成立？

> [!EXAMPLE]+ (i)(iv)
>
> 它是正则手段，倾向降低方差。停止条件是验证表现不再改善，也就是峰值之后的耐心计数，不是掉到谷底才停。

**EN.** Backpropagation:
>
> (i) costs $O(n^2)$ for $n$ parameters
> (ii) requires a directed acyclic graph, unless recurrence is unrolled
> (iii) Adam’s $\beta$ is updated by backpropagation
> (iv) BatchNorm’s $\gamma$ is updated by backpropagation

**中.** 关于反向传播，哪些成立？

> [!EXAMPLE]+ (ii)(iv)
>
> 反向传播与前向同阶，对参数量是线性的。$\beta_1,\beta_2$ 是超参数。$\gamma,\beta$（BatchNorm 的缩放与平移）是可训练参数，梯度来自反向传播。

**EN.** On ImageNet, with wall-clock time as the measure, which can be true?
>
> (i) SGD can finish sooner than full-batch GD.
> (ii) Mini-batch can finish sooner than SGD.
> (iii) Mini-batch can finish sooner than full-batch.
> (iv) Full-batch can finish sooner than SGD.

**中.** 以墙上时间为准，哪些比较可能成立？（Fall 2020 官方取前三项）

> [!EXAMPLE]+ (i)(ii)(iii)
>
> 全量梯度每步都要扫完数据，步数虽稳，时钟时间往往更差。Mini-batch 兼顾向量化与噪声，常快于 batch size 为 1 的 SGD，也快于全量梯度下降。官方答案未选 (iv)。

## 尺度、初始化与指标

**EN.** Ignore biases and numeric overflow. A conv net ends in softmax. Which leave the predicted class unchanged?
>
> (i) Multiply every weight by 10 at inference.
> (ii) Multiply every weight by 10 and continue training; convergence is unaffected.
> (iii) Multiply the input by 10 at inference.
> (iv) Subtract the per-channel mean of the input at inference.

**中.** 忽略偏置与溢出。哪些操作不改变 argmax 类别？（Spring 2021，推导）

> [!EXAMPLE]+ (i)(iii)
>
> 无偏置时，卷积、ReLU、最大池化对正缩放是齐次的。全部权重乘 10，或只把输入乘 10，最终 logits 同乘一个正数，softmax 的 argmax 不变。
>
> 训练动力学会变，学习率不再匹配。(iv) 不是正缩放，类别可以变。

**EN.** A softmax classifier already has $100\%$ accuracy. Replace $W$ by $3W$, with no regularization. Accuracy stays the same and cross-entropy falls.

**中.** 已达 $100\%$ 准确率的 softmax，把 $W$ 换成 $3W$。准确率不变，交叉熵下降。判断正误。

> [!EXAMPLE]+ 正
>
> logits 乘 3 后分布更尖，正确类概率上升，$\log$ 损失下降，argmax 不变。有 L2 正则时，权重变大本身会被惩罚，该结论不再自动成立。

**EN.** For data normalization, mean and standard deviation are computed on the whole dataset before the train/dev/test split.

**中.** 均值和标准差应在划分训练、验证、测试之前，用全部数据计算。判断正误。

> [!EXAMPLE]+ 误
>
> 统计量只能来自训练集。用到验证集或测试集会把未来样本的尺度信息漏进预处理。

**EN.** Weights of a 2-layer MLP with hidden sigmoids are drawn from $\mathcal N(10,1)$. From epoch 1, some first-layer weights barely move. Which help?
>
> (i) Draw weights from $\mathcal N(0,1)$.
> (ii) Add more hidden layers.
> (iii) Switch the hidden activation to tanh.
> (iv) Switch the hidden activation to leaky ReLU.

**中.** 隐藏层用 sigmoid，权重来自 $\mathcal N(10,1)$，第一层几乎不更新。哪些能缓解？（Fall 2022）

> [!EXAMPLE]+ (i)(iv)
>
> 预激活过大时 sigmoid 饱和，导数接近 0。tanh 同样会饱和。再加层只会让饱和更严重。小方差初始化或带泄漏的 ReLU 能把梯度留住。

**EN.** A binary metric should combine precision and recall. Which is a reasonable single score?
>
> (i) $|\mathrm{precision}-\mathrm{recall}|$
> (ii) $\mathrm{recall}/\mathrm{precision}$
> (iii) $\mathrm{precision}\times\mathrm{recall}$
> (iv) $\max(\mathrm{precision},\mathrm{recall})$

**中.** 想用一个数同时看 precision 与 recall。哪个还算合理？（Fall 2022）

> [!EXAMPLE]+ 只选乘积
>
> 差的绝对值奖励两者分开。比值可以靠拉高 recall、牺牲 precision 变大。最大值分不清 $(0.9,0.4)$ 与 $(0.9,0.7)$。乘积与几何平均同方向，两者都低时分数才低。F1 是更常见的调和形式，本题只问备选里哪一个说得通。

<br>


# 浅层

来源：Practice Midterm 1，创业公司融资二分类。输入特征 $n_x=300$，标签为第一年是否融资。

**EN.** Logistic regression outputs $\hat y=\sigma(w^\top x+b)$, and predicts 1 when $\hat y\ge 0.5$. How many weights does it have, and what is the shape of $b$ and of $\hat y$?

**中.** 逻辑回归有多少个权重？$b$ 与 $\hat y$ 的维度是什么？

> [!EXAMPLE]+ 300 个权重，$b$ 与 $\hat y$ 都是标量
>
> $\hat y^{(i)}=\sigma(w^\top x^{(i)}+b)$。

**EN.** A one-hidden-layer net has 80 hidden units, sigmoid output, and biases. How many weights and biases?

**中.** 单隐藏层 80 个单元。权重与偏置各有多少？

> [!EXAMPLE]+ 权重 24080，偏置 81
>
> $300\times 80+80\times 1=24080$，$80+1=81$，合计 $24161$。

**EN.** A one-hidden-layer net can approximate continuous functions under standard conditions. Why use more layers?

**中.** 单隐藏层已能逼近连续函数。多层的理由是什么？

> [!EXAMPLE]+ 同样的函数可以用更少的单元、更少的参数表示
>
> 官方给分点是参数效率。仅回答「更好优化」或「能表示更复杂的函数」没有拿到该题分数：普适逼近已经说明单层的表示能力。

**EN.** Hidden activations are omitted, so each layer is affine and the output is still a sigmoid. Show that the net is a logistic regression.

**中.** 中间没有非线性，输出仍是 sigmoid。证明它等价于逻辑回归。

> [!EXAMPLE]+ 仿射复合仍是仿射
>
> $$
> \hat y=\sigma\!\left(W^{[2]}(W^{[1]}x+b^{[1]})+b^{[2]}\right)
> =\sigma(w'^\top x+b')
> $$
>
> 列向量约定下 $w'= (W^{[2]}W^{[1]})^\top$，$b'=W^{[2]}b^{[1]}+b^{[2]}$。

**EN.** The output is changed to $\hat y=\sigma(\mathrm{ReLU}(z))$. What breaks?

**中.** 在 sigmoid 之前再套一层 ReLU。分类会出什么问题？

> [!EXAMPLE]+ 所有样本都会被判成正类
>
> $\mathrm{ReLU}(z)\ge 0$，故 $\sigma(\mathrm{ReLU}(z))\ge 0.5$。阈值 $0.5$ 下没有负类预测。

**EN.** Final two layers of a binary net are ReLU then sigmoid. What happens to the predictions?

**中.** 最后两层是 ReLU 再接 sigmoid。预测落在哪里？（Fall 2019）

> [!EXAMPLE]+ 全部至少为 $0.5$
>
> 与上一题相同。二分类阈值若取 $0.5$，模型只会输出正类。

<br>


# 损失

## 二分类

西瓜甜度二分类（Practice 1）。标签 $y\in\{0,1\}$。

**EN.** Why is mean squared error a poor training loss when $\hat y\in(0,1)$?

**中.** 预测被限制在 $(0,1)$ 时，为什么不宜用均方误差做训练损失？

> [!EXAMPLE]+ 错误再自信，MSE 也被压在 1 以内
>
> 对 $y\in\{0,1\}$、$\hat y\in[0,1]$，$(\hat y-y)^2\le 1$。完全判错时交叉熵可以趋向无穷，梯度信号更符合「概率应远离错误标签」。

**EN.** Write binary cross-entropy for one example. Then evaluate it at $y=0$, $\hat y=0.9$.

**中.** 写出单样本二元交叉熵，并计算 $y=0,\hat y=0.9$。

> [!EXAMPLE]+ $-\log 0.1$
>
> $$
> L=-(y\log\hat y+(1-y)\log(1-\hat y))
> $$
>
> 负样本只留下 $-\log(1-0.9)=-\log 0.1$。

**EN.** $y^\top=(1,0,0)$, $\hat y^\top=(0.1,0.2,0.7)$. Write the average cost $J$.

**中.** 三个样本的平均二元交叉熵是什么？

> [!EXAMPLE]+ $J=-\frac13(\log 0.1+\log 0.8+\log 0.3)$
>
> 三项分别是 $-\log 0.1$、$-\log 0.8$、$-\log(1-0.7)$。无权重惩罚。

**EN.** L2 penalty on one weight is $\lambda W^2$. Write the gradient step. How do the weights compare with an unregularized run, and how does L1 differ?

**中.** 惩罚项为 $\lambda W^2$。写出更新，并比较 L1。

> [!EXAMPLE]+ 权重被额外拉向 0
>
> $$
> W\leftarrow (1-2\alpha\lambda)W-\alpha\frac{\partial J_{\mathrm{BCE}}}{\partial W}
> $$
>
> 系数 $2$ 来自 $\partial(W^2)/\partial W$。若课程把惩罚写成 $\frac{\lambda}{2}W^2$，更新里的 $2$ 会消掉。同一数据下，带 L2 的权重量级整体更小。
>
> L1 惩罚 $\lvert W\rvert$，次梯度是符号函数，容易把一部分坐标推到精确的 0。L2 把权重变小，但通常不停在精确 0。

## 多任务与多标签

**EN.** One image must predict sweetness (binary) and weight in pounds (real). What is the label shape, and what loss is appropriate?

**中.** 同一张图同时预测甜度与重量。标签维度与损失如何取？

> [!EXAMPLE]+ 二维标签；甜度用交叉熵，重量用平方误差
>
> $y\in\mathbb R^{2}$。两部分量纲不同，需要各自的权重，否则平方项会淹没对数项。

电影海报四分类：comedy, horror, action, romance（Fall 2020）。人类水平估计为 $95\%$。

**EN.** What does a $95\%$ human-level accuracy say about the Bayes error?

**中.** 人类准确率 $95\%$ 对贝叶斯误差说明什么？

> [!EXAMPLE]+ 贝叶斯误差不超过 $5\%$
>
> 贝叶斯误差是任何决策规则的误差下界。一组人能达到的错误率是这个下界的上界估计。

**EN.** A horror poster has one-hot label on class 2. The model outputs $(0.1,0.4,0.3,0.2)$, then later $(0,0.4,0,0.6)$. What is the cross-entropy each time?

**中.** 正确类概率两次都是 $0.4$。交叉熵各是多少？

> [!EXAMPLE]+ 两次都是 $-\log 0.4$
>
> 单标签交叉熵只看正确类的对数概率。错误类之间怎么分质量，只要正确类仍是 $0.4$，损失不变。第二次把剩余质量从另外两类挪到 romance，并不改变这一项。

**EN.** Training accuracy is $90\%$, below the $95\%$ human level. What is the diagnosis?

**中.** 训练集只有 $90\%$。偏差还是方差？

> [!EXAMPLE]+ 高偏差
>
> 训练集还没接近可达到的水平。优先加大模型容量或延长有效训练，而不是先加正则。

**EN.** After the softmax fits the training set perfectly, why is cross-entropy still not zero?

**中.** 训练准确率已经是 $100\%$，softmax 交叉熵为什么到不了 0？

> [!EXAMPLE]+ 每个指数都为正，正确类概率严格小于 1
>
> $$
> L=-\log\frac{e^{z_c}}{\sum_i e^{z_i}}=-\,z_c+\log\sum_i e^{z_i}>0
> $$
>
> 除非其余 logits 趋向 $-\infty$，损失只能逼近 0，不能在有限权重下取到 0。

**EN.** Some labels may be wrong. Between L1 and L2, which is preferable before cleaning labels, and why does the posted solution pick L1?

**中.** 怀疑部分标签是错的。官方解答倾向哪一种正则？

> [!EXAMPLE]+ L1
>
> 错标签会把权重往错误方向拉。L2 对大残差的惩罚随权重二次增长，模型更愿意扭曲一大块参数去迁就坏标签。L1 允许多数坐标保持较小或为 0，对少数坏样本的妥协更局部。该题给分点是 L1。

**EN.** Posters may belong to several genres at once. Why do softmax and categorical cross-entropy fail, and what should replace them?

**中.** 一部电影可以同时属于多类。为什么不能继续用 softmax？

> [!EXAMPLE]+ softmax 强迫各类概率相加为 1，多标签会被重复惩罚
>
> 若 $K$ 个类都正确，而模型均匀输出 $1/n_y$，损失约为 $K\log n_y$，远大于单标签样本。
>
> 标签改为 multi-hot。输出改为逐类 sigmoid，损失改为各类二元交叉熵的平均。这样每一类是独立的是/否，不再竞争同一个概率质量。

> [!NOTE]+ 下界
>
> 在仍使用 softmax、且有 $K$ 个正确类时，Jensen 不等式给出 $L\ge K\log K$。均匀分给这 $K$ 类、其余类概率为 0 的极限才取等。有限 logits 到不了这个极限。

## 加权对数与 hinge

长颈鹿检测（Fall 2019）：正类约 2000，负类约 200。损失写成

$$
L^{(i)}=\alpha\, y^{(i)}\log\hat y^{(i)}+\beta\,(1-y^{(i)})\log(1-\hat y^{(i)}),\qquad
J=-\frac1m\sum_i L^{(i)}.
$$

**EN.** Why introduce $\alpha$ and $\beta$? Give a reasonable pair.

**中.** $\alpha$ 与 $\beta$ 在调节什么？一组说得通的数值是什么？

> [!EXAMPLE]+ 稀有类应得到更大的损失权重
>
> 负类大约只有正类的十分之一。取 $\alpha=0.1$、$\beta=1$，或任何 $\beta/\alpha$ 约为 10 且两者为正的组合，都会让「没有长颈鹿」的错误对梯度贡献更大。$\alpha,\beta\le 0$ 与「最小化 $J$」的方向冲突。

**EN.** A one-hidden-layer net uses

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

**中.** 对上述 hinge 形式的损失做完整反向传播。

> [!EXAMPLE]+ 因为 $y_i\hat y_i\in[0,1]$，hinge 始终处于激活段
>
> $0\le y_i\hat y_i\le 1$，故 $1-y_i\hat y_i\ge 0$，最大值的第一支不会出现。
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

**EN.** Full-batch gradient descent is used, and the training set is never shuffled. Does shuffling fix a poor training loss?

**中.** 全量梯度下降不打乱数据。打乱能修好训练损失吗？

> [!EXAMPLE]+ 不能
>
> 每一步都使用全部样本，梯度与样本顺序无关。打乱只改变 mini-batch 与 batch size 为 1 的估计。

**EN.** One reason to prefer mini-batch over batch size 1, and one reason to prefer it over full-batch.

**中.** mini-batch 相对 SGD 与全量梯度下降各有什么好处？

> [!EXAMPLE]+ 向量化，以及更快的第一步
>
> 相对 batch size 1：矩阵运算吃满硬件，梯度噪声更小。
>
> 相对全量：大数据放不进显存；一步不必等整个数据集；适度噪声有助于离开鞍点。

**EN.** Three loss curves, from smoothest to noisiest, correspond to which algorithms?

**中.** 损失曲线从最平滑到最抖，对应哪种梯度下降？

> [!EXAMPLE]+ 全量、mini-batch、batch size 1
>
> 噪声随批量减小而增大。全量曲线最平，单样本最抖。

## 学习率与搜索

**EN.** Name one symptom of a learning rate that is too large, and one of a learning rate that is too small.

**中.** 学习率过大、过小各有什么典型现象？

> [!EXAMPLE]+ 震荡或发散；损失下降极慢
>
> 过大时，按迭代画出的成本会剧烈震荡甚至上升。过小时，成本单调但几乎走平。比较相邻数量级的学习率，比在同一条曲线上猜测更可靠。

**EN.** Candidate learning rates: $0.01,0.16,0.21,0.84,0.94$. Is this a good grid?

**中.** 这组学习率的搜索是否合理？（Fall 2019）

> [!EXAMPLE]+ 不合理
>
> 四个点挤在 $[0.1,1]$，只有一个点落在 $[0.01,0.1]$。学习率应在对数轴上随机取样，让每个数量级的密度相近。

**EN.** $\beta_1$ for Adam is drawn uniformly from $[0.9,0.999]$. Find $P(\beta_1<0.94)$ and $P(\beta_1>0.99)$. What is a better search?

**中.** 在该区间上均匀抽 $\beta_1$。两个概率是多少？

> [!EXAMPLE]+ $0.04/0.099$ 与 $0.009/0.099$
>
> 区间长度 $0.099$。
>
> $$
> P(\beta_1<0.94)=\frac{0.04}{0.099},\qquad
> P(\beta_1>0.99)=\frac{0.009}{0.099}.
> $$
>
> $\beta_1$ 靠近 1 时，有效记忆长度 $1/(1-\beta_1)$ 变化很快。应在 $1-\beta_1$ 的对数尺度上搜索，而不是在 $\beta_1$ 上均匀抽。

**EN.** Write momentum and the bias-corrected Adam update.

**中.** 写出 momentum 与 Adam 的更新。

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

**EN.** Logistic regression is initialized at $0.5$ for every weight. A 4-layer net is initialized at the same constant. Which initialization is acceptable?

**中.** 逻辑回归与四层网络都把权重初始化为常数 $0.5$。哪一个可以？

> [!EXAMPLE]+ 只有逻辑回归可以
>
> 逻辑回归的成本对 $(w,b)$ 是凸的，起点只影响走到最优点的步数。
>
> 多层网络里，同一层的单元若权重完全相同，收到的梯度也相同，对称性不会自己打破，隐藏单元学到同一函数。常数初始化，包括全 0，都会留下这个对称。

**EN.** Early-stopping patience $k=5$ versus $k=0$. What is the benefit of the larger buffer? (Fall 2022)

**中.** 验证集连续 5 轮不提升才停止，相比「一轮不提升就停」，好处是什么？

> [!EXAMPLE]+ 不被验证曲线的噪声提前打断
>
> 小验证集的指标会上下跳动。耐心为 0 时，一次随机变差就会停在欠拟合的权重上。$k=5$ 不减少平均训练时间，也不改变推理时间。

**EN.** The training curve oscillates strongly early in training. Which stabilize it?
>
> (i) Early stopping. (ii) Learning-rate schedules. (iii) Data augmentation. (iv) Gradient clipping.

**中.** 训练曲线一开始剧烈震荡。哪些有助于稳住？（Fall 2022 选项答案为 (ii)(iv)）

> [!EXAMPLE]+ 学习率日程与梯度裁剪
>
> 日程把步长从大逐渐减小，裁剪限制单步位移。提前停止只是结束训练。数据增强主要改变泛化，不直接消除更新震荡。

<br>


# 归一

来源：Practice Midterm 1 的 BatchNorm 题。

**EN.** Which statements are true?
>
> (i) BatchNorm makes one batch cheaper, so the same wall-clock time buys more updates.
> (ii) BatchNorm weakens the coupling between earlier and later layers.
> (iii) It makes the distribution seen by the next layer more consistent across dimensions.
> (iv) It removes the damage of a bad initialization and, in particular, allows weights to be initialized near zero.

**中.** 关于 BatchNorm，哪些成立？

> [!EXAMPLE]+ (ii)(iii)
>
> (i) 反了：前向要算均值方差，反向要多传 $\gamma,\beta$，单步更贵。收益是达到同样损失所需的步数往往更少。
>
> (iv) 的前半句成立，后半句不成立。BatchNorm 降低对初始化的敏感度，并没有要求把权重初始化到接近 0。接近 0 的权重仍可能让 ReLU 或 sigmoid 处在不好的区域。

**EN.** What does $\epsilon$ do? What fails at batch size 1? A dense layer maps 10 units to 20 and is followed by BatchNorm on the 20 pre-activations, with no redundant bias. How many trainable parameters?

**中.** $\epsilon$ 的作用是什么？batch size 为 1 时哪里坏了？$10\to 20$ 的全连接再加 BatchNorm，可训练参数有多少？

> [!EXAMPLE]+ 240 个参数
>
> $\epsilon$ 防止方差为 0 时除零。
>
> batch size 为 1 时，该样本就是均值，归一化结果恒为 0，再乘 $\gamma$ 仍是 0。只剩 $\beta$ 能变，特征学不出来。更一般地，极少样本上的均值和方差噪声极大，滑动平均也会偏。
>
> 权重 $10\times 20=200$。BatchNorm 的 $\gamma,\beta$ 各 20。偏置与 $\beta$ 重复，不再单列。合计 $240$。

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

**EN.** A conv block is Conv, then BatchNorm, then the nonlinearity. A colleague sets every conv bias to 0 and never learns it. Does the block lose a degree of freedom? (Fall 2020)

**中.** 卷积后面紧跟 BatchNorm。卷积偏置永久置 0，表达能力会少一块吗？

> [!EXAMPLE]+ 不会
>
> BatchNorm 的 $\beta$ 能吸收卷积偏置。ResNet 的常见实现因此关掉卷积偏置。若 BatchNorm 在激活之后、或根本没有 BatchNorm，偏置不能删。

<br>


# 卷积

## 尺寸与参数

输出边长（方阵、方核）：

$$
H_{\mathrm{out}}=\left\lfloor\frac{H+2p-f}{s}\right\rfloor+1.
$$

卷积参数：$(f\cdot f\cdot C_{\mathrm{in}}+1)\,C_{\mathrm{out}}$。池化与 ReLU 的参数为 0。BatchNorm 每个通道有 $\gamma$ 与 $\beta$，共 $2C$。

**EN.** Same spatial size as the input, stride 1. What padding does a filter of width $f$ need?

**中.** stride 为 1 时，多宽的 padding 能使输出边长等于输入？

> [!EXAMPLE]+ $(f-1)/2$
>
> $f$ 为奇数时这是整数。$f=5$ 对应 padding 2，$f=3$ 对应 padding 1。

**EN.** Input $32\times 32\times 3$, kernel $5\times 5$, padding 0, stride 1, 2 filters. Give the output volume and the parameter count. Repeat for a 1-D conv of width 5 on $T\times 3$, and a $5\times 5\times 5$ conv on video. (Practice 1)

**中.** 三种核的参数量。

> [!EXAMPLE]+ 152、32、752
>
> 输出体积 $(32-5+1)=28$，即 $28\times 28\times 2$。
>
> $$
> \begin{aligned}
> \text{2-D: }&(5\cdot 5\cdot 3+1)\cdot 2=152,\\
> \text{1-D: }&(5\cdot 3+1)\cdot 2=32,\\
> \text{3-D: }&(5\cdot 5\cdot 5\cdot 3+1)\cdot 2=752.
> \end{aligned}
> $$

**EN.** Why use a $1\times 1$ convolution?

**中.** $1\times 1$ 卷积改变什么、保留什么？

> [!EXAMPLE]+ 混合通道，保留高和宽
>
> 每个空间位置上做一次跨通道线性层，用来降维或升维。空间尺寸不变，参数远少于把特征图展平后的全连接。

**EN.** Fill the table. `CONV5-N` means $N$ filters of size $5$, padding 2, stride 1. `POOL2` is $2\times 2$, stride 2. Input is $32\times 32\times 1$. (Fall 2019)

**中.** 填每一层的输出体积与参数量。

| 层 | 输出 | 参数 |
| :--- | :--- | :--- |
| Input | $32\times 32\times 1$ | 0 |
| CONV5-10 | $32\times 32\times 10$ | $(25\cdot 1+1)\cdot 10=260$ |
| POOL2 | $16\times 16\times 10$ | 0 |
| CONV5-10 | $16\times 16\times 10$ | $(25\cdot 10+1)\cdot 10=2510$ |
| POOL2 | $8\times 8\times 10$ | 0 |
| FC-10 | $10$ | $(640+1)\cdot 10=6410$ |

**EN.** Input $32\times 32\times 3$. `CONV3` uses padding 1 and stride 1. `POOL-n` has window $n$, stride $n$, padding 0. (Fall 2020)

**中.** 填 Fall 2020 的卷表。

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

**EN.** Input $128\times 128\times 3$. First conv only. (Spring 2021)

**中.** 只看第一层卷积。

> [!EXAMPLE]+ 三组超参
>
> | 设置 | 输出 | 参数 |
> | :--- | :--- | :--- |
> | $s=1,f=3,p=0,64$ filters | $126\times 126\times 64$ | $64\cdot(27+1)=1792$ |
> | $s=1,f=5,p=2,16$ filters | $128\times 128\times 16$ | $16\cdot(75+1)=1216$ |
> | $s=2,f=2,p=0,32$ filters | $64\times 64\times 32$ | $32\cdot(12+1)=416$ |

**EN.** Three identical modules follow: Conv, $2\times 2$ max-pool with stride 2, ReLU. Start from $128\times 128\times 3$. Which setting gives the smallest 10-way fully connected layer, and how many parameters does that layer have?

**中.** 三个「卷积 + $2\times 2$ 池化 + ReLU」模块之后接 10 类全连接。哪种卷积配置让全连接参数最少？

> [!EXAMPLE]+ stride 2、核 $2\times 2$、32 个滤波器；全连接 1290 个参数
>
> 记模块为卷积后立刻池化。
>
> - $f=3,p=0,64$：空间 $128\to 126\to 63\to 61\to 30\to 28\to 14$。体积 $14\times 14\times 64=12544$。全连接 $(12544+1)\cdot 10=125450$。卷积参数 $1792+36928+36928=75648$。
> - $f=5,p=2,16$：空间每步保持再减半，$128\to 64\to 32\to 16$。体积 $16\times 16\times 16=4096$。全连接 $40970$。卷积参数 $1216+6416+6416=14048$。
> - $f=2,s=2,p=0,32$：卷积先减半，池化再减半，$128\to 64\to 32\to 16\to 8\to 4\to 2$。体积 $2\times 2\times 32=128$。全连接 $(128+1)\cdot 10=1290$。卷积参数 $416+4128+4128=8672$。
>
> 全连接最瘦的是第三种。它的空间被 stride 和池化各砍掉很多，分类层几乎不看细粒度位置。

## 感受野与小网络

**EN.** Layers, in order: $3\times 3$ conv stride 2, $2\times 2$ pool stride 2, $3\times 3$ conv stride 2, $2\times 2$ pool stride 2. How large is the support, in input pixels, of one unit in the fourth non-image layer? (Winter 2021 题干)

**中.** 第四个非图像层上的一个神经元，对应输入上多大的一片？

> [!EXAMPLE]+ $21\times 21$
>
> 感受野 $r$ 与跳步 $j$ 从 $r=1,j=1$ 起。经过核宽 $k$、步长 $s$ 的一层：
>
> $$
> r\leftarrow r+(k-1)\,j,\qquad j\leftarrow j\cdot s.
> $$
>
> | 层 | $r$ | $j$ |
> | :--- | :--- | :--- |
> | $3\times 3$, stride 2 | 3 | 2 |
> | $2\times 2$, stride 2 | 5 | 4 |
> | $3\times 3$, stride 2 | 13 | 8 |
> | $2\times 2$, stride 2 | 21 | 16 |
>
> 支持是边长 21 的正方形。池化步长若不是 2，结果会变。

**EN.** Classify a $16\times 16\times 3$ digit image into 10 classes with Conv, max-pool, flatten, and a fully connected layer. Constraints: stride 1, exactly 4 filters, conv activation has at most 576 entries, pool stride equals pool size, the classifier has no bias and at most 1440 parameters. Give one feasible design. Which digit augmentations are valid: mild scale, flips, rotations by 90 or 180, mild shear? (Winter 2021)

**中.** 在内存限制下给一个可行的小 CNN，并判断数字图像的增强。

> [!EXAMPLE]+ 例如 $5\times 5$ 卷积、padding 0，再 $2\times 2$ 池化
>
> 卷积输出 $12\times 12\times 4=576$，刚好顶到上限。池化后 $6\times 6\times 4=144$。无偏置全连接参数 $144\times 10=1440$，也刚好顶到上限。全连接输出维度只能是 10。
>
> 轻微缩放、轻微剪切保留数字身份。水平或垂直翻转会毁掉 7 这类字形。$90^\circ$ 或 $180^\circ$ 旋转会把 6 转成 9。

**EN.** Among conv, max-pool, average-pool, and BatchNorm, which uses the fewest floating-point operations for an activation of size $n$? (Fall 2020)

**中.** 同样大小的激活上，哪一种层的浮点运算最少？

> [!EXAMPLE]+ 最大池化
>
> 窗口里求最大只需要比较。平均池化要做加法再除，卷积与 BatchNorm 的运算不少于一次完整遍历加额外算术。

<br>


# 反向

## 温度 softmax

编码器先把 $f_\theta(X)$ 变成 softmax，再想得到 one-hot，然后送入下一层 $g$（Fall 2022）。

**EN.** Is $y=g(\mathrm{one\_hot}(\mathrm{softmax}(f(X))))$ trainable by gradient descent?

**中.** one-hot 放在计算图中间，梯度下降还能用吗？

> [!EXAMPLE]+ 不能直接用
>
> $\mathrm{argmax}$ 与 one-hot 在几乎处处的导数为 0，在切换点不可导。下游损失传不回 $f$。

**EN.** $S_\tau(u)=\mathrm{softmax}(u/\tau)$. What happens as $\tau\to\infty$ and as $\tau\to 0$?

**中.** 温度趋向无穷和趋向 0 时，分布变成什么样？

> [!EXAMPLE]+ 均匀分布，以及 argmax 上的 one-hot
>
> $\tau$ 很大时，logits 的差距被抹平，各类概率趋于 $1/n$。
>
> $\tau$ 很小时，最大的 logit 主导指数，质量集中到 argmax。因此低温度是 one-hot 的可微近似，不必再接一个离散的 one-hot。

**EN.** $f(X)=w^\top X$ produces one logit per coordinate of $X$. Derive $\partial S_\tau(w^\top X)_i/\partial w$.

**中.** 第 $i$ 个温度 softmax 分量对 $w$ 的导数。

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

**EN.** State the shapes. For one example, simplify $\partial J/\partial z_3$.

**中.** 参数形状是什么？输出预激活上的梯度如何化简？

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

## 图网络

两层图网络（Fall 2022）。节点特征是标量。第 $k$ 层只把至多 $k$ 跳邻居的信息写进该节点。

$$
a^{[1]}_i=\mathrm{ReLU}\!\left(x_i+w^{[1]}\sum_{n\in\mathcal N(i)}x_n+b^{[1]}\right),
$$

第二层把 $x$ 换成 $a^{[1]}$，参数换成 $w^{[2]},b^{[2]}$。

**EN.** On a path where node 1 and node 6 are more than two hops apart, what is $\partial a^{[2]}_1/\partial x_6$? How can one new node make this derivative nonzero, using as few new edges as possible?

**中.** 两跳以外的节点，二层输出对其输入的导数是多少？如何用一个新节点把这个导数变成非零？

> [!EXAMPLE]+ 导数为 0；新节点同时连到 1 和 6
>
> 二层输出只依赖两跳以内的输入。一条边连到 1、一条边连到 6，则 $6\to 7\to 1$ 的距离变成 2，信息进得了第二层。

<br>


# 生成

## 对抗样本

**EN.** Fast Gradient Sign Method: $x^\star=x+\varepsilon\,\mathrm{sign}(\partial J/\partial x)$. Given $x=(1,2,3)^\top$, $\partial J/\partial x=(0.5,-0.5,1)^\top$, $\varepsilon=0.01$, compute $x^\star$. (Practice 1)

**中.** 计算这一步 FGSM。

> [!EXAMPLE]+ $(1.01,\,1.99,\,3.01)^\top$
>
> 符号是 $(1,-1,1)$。每个坐标最多移动 $\varepsilon$，与梯度幅度无关。

**EN.** Does the $\varepsilon$ needed to change a prediction grow with the dimension of $x$?

**中.** 输入维度变大时，为了改变预测，是不是必须把 $\varepsilon$ 也放大？

> [!EXAMPLE]+ 不必
>
> 对无偏置逻辑回归，$x^\star=x+\varepsilon\,\mathrm{sign}(w)$ 使 logit 增加 $\varepsilon\lVert w\rVert_1$。若每个权重量级差不多，$\lVert w\rVert_1$ 随维度线性增长。固定的小 $\varepsilon$ 在高维上仍能把 logit 推过决策边界，而每个像素只改了 $\varepsilon$。

**EN.** Mark the true statements.
>
> (i) An example built to fool classifier A can also fool a similar classifier B.
> (ii) FGSM is an iterative attack.
> (iii) Dropout at test time is an effective defense.
> (iv) A black-box model, with no access to weights, can still be attacked.

**中.** 关于对抗样本，哪些成立？

> [!EXAMPLE]+ (i)(iv)
>
> 相似决策边界会共享脆弱方向，所以攻击可以迁移。FGSM 只沿符号走一步，迭代版本是别的方法。Dropout 在测试时关闭。黑盒可以通过查询输出或利用迁移来构造扰动。

## GAN 与噪声

**EN.** After the first generator and discriminator updates, is $D(G(z))$ closer to 0 or to 1? When is training finished? Which generator cost is preferable, the saturating one or the non-saturating one? (Fall 2022 与 Practice 1)

**中.** 训练刚开始时 $D(G(z))$ 靠近哪一端？何时算训练完成？生成器该用哪种成本？

> [!EXAMPLE]+ 一开始靠近 0；完成时靠近 $1/2$；用 non-saturating
>
> 初始生成器很容易被判别器认出，所以 $D(G(z))$ 靠近 0（0 表示「判为假」）。
>
> 均衡时判别器分不清真假，输出应靠近 $1/2$，而不是靠近 1。靠近 1 只说明当前判别器被生成器骗过，或判别器自己太弱。
>
> saturating 成本在 $D(G(z))\approx 0$ 时梯度消失。non-saturating 在生成器最差的时候梯度更大，所以训练早期更可用。
>
> 第 1 轮和第 100 轮的损失数值相近，也不表示图像质量相近：损失是相对于当时的对手计算的，对手本身在变。

**EN.** A generator of reptiles may be mode-collapsing. Which observations fit?
>
> (i) Every sample is a Komodo dragon.
> (ii) The generator loss oscillates.
> (iii) Generator loss stays low while discriminator loss stays high.
> (iv) The discriminator is accurate on real images and inaccurate on fake ones.

**中.** 哪些现象更像 mode collapse？（Fall 2020）

> [!EXAMPLE]+ (i)(ii)
>
> 只生成单一模式，或在几个模式之间跳、损失来回震荡，都指向模式塌缩。(iii)(iv) 主要说明判别器弱，生成器即使多样也会呈现这种损失。

**EN.** Only a few real octopus photos exist. Which augmentation is safe for a GAN: blur, recoloring, or left-right flips?

**中.** 数据很少时，模糊、改颜色、左右翻转，哪些适合拿去训生成章鱼的 GAN？

> [!EXAMPLE]+ 只保留左右翻转
>
> 模糊和改色会被生成器当成真实外观学进去。左右翻转不改变章鱼的合理外形。

**EN.** Linear regression minimizes $J=\frac1m\sum_i(y^{(i)}-x^{(i)\top}\theta)^2$. Features are corrupted by $\delta\sim\mathcal N(0,\sigma^2 I)$. Show that the expected corrupted objective equals $J$ plus a term that does not depend on the data. (Winter 2021)

**中.** 给特征加各向同性高斯噪声，期望损失比原损失多出什么？

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


# 策略

**EN.** 500 ultrasound cases, 175 preterm. All positive cases are duplicated, and only then is the set split into train, validation, and test. What is wrong? The best model has $100\%$ recall. Is that enough if missing a preterm birth is unacceptable? Which human baseline best estimates Bayes error: one student (F1 0.20), one doctor (0.80), or a group of doctors (0.88)? (Practice 1)

**中.** 先复制正类再划分数据有什么问题？召回率 100% 是否够用？贝叶斯误差该用哪组人类成绩估计？

> [!EXAMPLE]+ 泄漏；召回率可以被平凡模型刷满；用医生小组
>
> 同一张图可以同时出现在训练和测试里，测试分数会虚高。复制正类本身可以平衡损失，问题出在复制与划分的顺序。应先划分，再只在训练集里平衡。
>
> 永远预测正类的模型召回率也是 100%。还要看精度或特异性。
>
> 贝叶斯误差用所能找到的最强人类组合来估计，这里是医生小组的 F1 0.88，对应误差约 0.12 的量级，而不是单个学生。

**EN.** Chest X-ray cancer screening: accuracy, precision, recall, or F1? A click-through model can be one large net or an average of ten smaller nets, both at 90% training accuracy. The ten-model ensemble was cheaper to train. Deployment is one GPU and must answer in real time. Which system goes to production? (Fall 2020)

**中.** 胸片筛查看哪个指标？上线时选大模型还是十模型平均？

> [!EXAMPLE]+ 筛查看召回；上线用单个大模型
>
> 漏诊比误报更贵，召回率优先，再辅以可接受的精度。准确率在类别不平衡时会掩盖漏诊。
>
> 十个模型在训练时更省时间，但每次请求都要做十次前向。单 GPU 实时服务选训练更贵、推理只有一次前向的那个模型。训练准确率同为 90% 并不能区分二者的泛化。

**EN.** 100,000 chest CT images, 1,000 labeled, for COVID-19. Pick one:
>
> (i) Transfer from a ResNet-50 pretrained on chest CT for tumor detection.
> (ii) Train a GAN to synthesize labeled images, then supervise on real plus synthetic labels.
> (iii) Supervise from scratch on the 1,000 labels.
> (iv) Random crops of the 1,000 labels, then supervise from scratch.

**中.** 标注很少、同模态图像很多。哪一条最有机会做成？（Fall 2022）

> [!EXAMPLE]+ 迁移已在胸片 CT 上预训练的 ResNet
>
> 预训练数据与目标同属 CT，任务从肿瘤换成新冠，属于近邻迁移。1,000 张从头训练，即使用裁剪增强，也容易记住训练集。GAN 要同时学会生成阳性和阴性，而每一类的标注都远少于 GAN 通常需要的量，合成标签还会把生成错误写进监督信号。

**EN.** Lane detection from a forward camera on two-way California streets. Which augmentations are valid?
>
> (i) Vertical flip. (ii) Horizontal flip. (iii) Synthetic fog. (iv) Random masks on a small region.

**中.** 车道方向分「同向」和「对向」。哪些增强还成立？

> [!EXAMPLE]+ 雾与小块遮挡
>
> 垂直翻转会把路面翻到天上。加州靠右行驶，对向车道在画面左侧；水平翻转会把左右方向反掉，标签不再成立。雾对应真实天气。小块遮挡对应路上的障碍物。

**EN.** An emergency-room ranking model is trained on three months of US hospital data, with ethnicity as an input. It is then deployed in other countries, and it ranks African American and Hispanic patients as less severe than the chart suggests. What fails, and what can be changed? (Fall 2022)

**中.** 这个分诊模型为什么不能直接拿到别的国家用？族群排序偏差可能从哪来，可以改什么？

> [!EXAMPLE]+ 分布换了；标签里可能已经含有历史待遇差异
>
> 人口结构、病种和就诊方式都变了，美国数据上的输入分布不再成立。
>
> 若训练标签来自历史上对某些族群治疗不足的记录，模型会把这种差异当成「病情较轻」。输入里的族群字段会把这条捷径留住。
>
> 可做的事包括：去掉族群字段、用更少偏倚的标签重新训练、在评估时按族群拆开看召回而不是只看总准确率。去掉字段并不能消除由其他相关变量带来的代理偏差，所以还要分组评估。

**EN.** Handwritten letters A–Z, each class balanced. Random horizontal and vertical flips are proposed as augmentation. Separately: KNN with L1 distance, and the same flip applied to every training image and every test image. What happens to accuracy? (Spring 2021)

**中.** 字母任务能否随机翻转？训练和测试一起翻转时，L1 的 KNN 准确率变不变？

> [!EXAMPLE]+ 随机翻转不合适；训练和测试做同一个置换则 KNN 不变
>
> b 与 d、p 与 q 会互换，类条件分布被增强本身弄脏。
>
> 同一种翻转是特征坐标的置换。L1 距离对全体样本使用同一置换保持不变，近邻关系不变。

**EN.** A fuel-economy model is used to choose engine size and weight by maximizing the model’s prediction. Is that a sound way to design a car? (Fall 2020)

**中.** 把回归模型的预测当作设计目标去搜参数，是否稳妥？

> [!EXAMPLE]+ 不稳妥
>
> 模型学到的是观测数据里的相关，不是「把重量改到某个外推值就会省油」的因果。搜索还可能走出训练分布，跑到物理上造不出来的组合。

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

## 全 0 学习率

**EN.** A 99-way softmax is initialized from $\mathcal N(0,10^{-4})$, the learning rate is 0, and classes are balanced. What is the expected cross-entropy after one epoch?

**中.** 权重几乎为 0 且完全不更新。99 类交叉熵的期望大约是多少？（Fall 2020）

> [!EXAMPLE]+ $-\log(1/99)=\log 99$
>
> logits 接近 0，softmax 接近均匀。正确类概率约为 $1/99$，与初始化的微小噪声无关。学习率为 0 时这个值不会随 epoch 改变。

<br>


# 作业

名称与顺序来自 [2025 Syllabus](https://cs230.stanford.edu/syllabus/) 及 [DLS 完整 GitHub 归档](https://github.com/amanchadha/coursera-deep-learning-specialization)。模块号 `C1M2` 表示专项第 1 门课的第 2 周。Stanford 每季从 DLS 中指定一部分，GitHub 归档则覆盖整个专项；以下按归档逐份列出，不表示每一份都在同一季计分。

## C1 神经网络

### C1M2 Python Basics with NumPy

**EN.** Implement elementary vectorized utilities without Python loops.

**中.** 用 NumPy 完成基础向量化函数。

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

### C1M2 Logistic Regression with a Neural Network Mindset

**EN.** Build a cat/non-cat classifier from initialization through prediction.

**中.** 从零完成猫图二分类器。

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

### C1M3 Planar Data Classification

**EN.** Compare logistic regression with a one-hidden-layer network on a nonlinear planar dataset.

**中.** 在花瓣形数据上比较线性模型与单隐藏层网络。

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

### C1M4 Building a Deep Neural Network

**EN.** Implement reusable $L$-layer forward and backward building blocks.

**中.** 把深层网络拆成可复用的前向与反向组件。

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

### C1M4 Deep Neural Network Application

**EN.** Train a 2-layer model and an $L$-layer model for cat classification, then inspect errors.

**中.** 复用上一份组件，比较浅层与深层猫图分类器。

- 两层结构：`LINEAR -> RELU -> LINEAR -> SIGMOID`；
- 深层结构：`[LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID`；
- 画训练成本；
- 在训练集与测试集上预测；
- 查看误分类图片，判断角度、背景、亮度、遮挡等误差来源。

> [!NOTE]+ 解析
>
> 这一份不再考单个导数，而是考组件接口、形状与 cache 能否组合。成本下降但测试准确率停滞时，应检查泛化误差，不应只增加迭代次数。

## C2 训练技巧

### C2M1 Initialization

**EN.** Compare zero, random-large, and He initialization on the same classifier.

**中.** 比较全零、大随机数与 He 初始化。

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

### C2M1 Regularization

**EN.** Add L2 regularization and inverted dropout to a three-layer network.

**中.** 在三层网络中分别实现 L2 与 inverted dropout。

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

### C2M1 Gradient Checking

**EN.** Verify scalar and deep-network gradients with centered finite differences.

**中.** 用中心差分检查标量函数与多层网络的反向传播。

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

### C2M2 Optimization Methods

**EN.** Implement mini-batches, momentum, and Adam, then compare convergence.

**中.** 实现随机 mini-batch、momentum 与 Adam。

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

### C2M3 TensorFlow Introduction

**EN.** Rebuild a small classifier with TensorFlow 2 and automatic differentiation.

**中.** 用 TensorFlow 2 重写小型分类器。

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

## C3 项目策略

C3 没有编程 notebook，公开归档只有两个案例测验。

### C3M1 Bird Recognition in Peacetopia

**EN.** Design a metric and diagnose avoidable bias, variance, and data mismatch for a city bird detector.

**中.** 为城市鸟类检测器选择指标，并区分偏差、方差与数据不匹配。

- 把多个目标压成单一优化指标，把时延、内存等设成满足指标；
- dev 与 test 必须来自相同目标分布；
- 人类水平近似贝叶斯误差；
- 训练误差减人类误差是可避免偏差；
- dev 误差减训练误差是方差；
- train-dev 与 dev 的差异揭示数据不匹配。

> [!EXAMPLE]+ 诊断顺序
>
> 若人类 $1\%$、训练 $2\%$、train-dev $3\%$、dev $10\%$，则可避免偏差约 $1\%$，方差约 $1\%$，主要问题是分布不匹配约 $7\%$。此时盲目加正则会处理错对象。

### C3M2 Autonomous Driving

**EN.** Decide when to use transfer learning, multi-task learning, end-to-end learning, and error analysis.

**中.** 在自动驾驶案例中判断迁移学习、多任务、端到端与误差分析。

- 迁移学习：源任务数据多、底层特征可复用、目标任务数据少；
- 多任务：多个任务共享输入与低层特征，且每个任务都有足够标签；
- 端到端：数据极多且输入到输出映射能覆盖中间知识；
- 误差分析：先人工统计最大错误类别，再决定采集数据或改模型；
- 合成数据：必须缩小目标分布误差，而不是只增加容易样本。

## C4 卷积网络

### C4M1 Convolutional Model: Step by Step

**EN.** Implement convolution and pooling forward/backward passes in NumPy.

**中.** 用 NumPy 从零实现卷积与池化的前向、反向。

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

### C4M1 Convolutional Neural Networks: Application

**EN.** Build a sign-language classifier with TensorFlow/Keras Conv2D, pooling, flattening, and dense output.

**中.** 用框架搭建手势分类 CNN。

- 顺序模型版的卷积、ReLU、最大池化、展平、全连接；
- Functional API 版前向；
- 编译模型，检查参数量；
- 训练并比较 train/dev 曲线。

> [!NOTE]+ 解析
>
> `Conv2D(filters=8, kernel_size=4, strides=1, padding="same")` 的输出通道是 8，参数量是 $(4\cdot4\cdot C_{\mathrm{in}}+1)\cdot8$。池化改变尺寸，不增加参数。

### C4M2 Keras Happy House

**EN.** Build a binary smile detector with Keras.

**中.** 为“是否微笑”搭建二分类 CNN。

- `ZeroPadding2D`、`Conv2D`、BatchNorm、ReLU、MaxPool、Flatten、Dense；
- 编译与训练；
- 用自定义图片做预处理和预测。

> [!EXAMPLE]+ 输出
>
> 二分类最后可用一个 sigmoid 单元与 BCE。若最后用了两个 softmax 单元，标签必须与之匹配，不能混用标量 BCE 标签。

### C4M2 Residual Networks

**EN.** Implement identity and convolutional residual blocks, then assemble ResNet-50.

**中.** 实现恒等残差块、卷积残差块并拼成 ResNet-50。

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

### C4M2 Transfer Learning with MobileNet

**EN.** Adapt a pretrained MobileNetV2 to an Alpaca/not-Alpaca classifier.

**中.** 用 MobileNetV2 做羊驼二分类迁移学习。

- 数据增强管道；
- 加载 ImageNet 权重且去掉顶层；
- 冻结 base model；
- 全局平均池化、dropout、分类头；
- 只训练分类头；
- 解冻靠后的层，小学习率微调；
- 保持 BatchNorm 的推理语义。

> [!EXAMPLE]+ 两阶段训练
>
> 第一阶段只学新头，避免随机头的大梯度破坏预训练特征。第二阶段只解冻顶端若干层，并把学习率降一到两个数量级。小数据上把全部层一次性解冻，通常会迅速过拟合。

### C4M3 Car Detection with YOLO

**EN.** Decode YOLO outputs, filter low-confidence boxes, and apply non-max suppression.

**中.** 解码 YOLO 输出并完成后处理。

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

### C4M3 Image Segmentation with U-Net

**EN.** Assemble contracting and expanding paths for pixelwise segmentation.

**中.** 搭建 U-Net 的下采样与上采样路径。

- `conv_block`：两次卷积，可选 dropout，再最大池化；
- `upsampling_block`：转置卷积上采样，与同分辨率编码器特征拼接，再卷积；
- `unet_model`；
- 最后用 $1\times1$ 卷积把通道变成类别数；
- 查看预测掩码。

> [!EXAMPLE]+ 跳连不是残差相加
>
> U-Net 跳连通常在通道维 `concatenate`，把定位细节交给解码器；ResNet 则逐元素相加。若空间尺寸因 padding 不一致，拼接前必须裁剪或统一使用 same padding。

### C4M4 Neural Style Transfer

**EN.** Optimize pixels of a generated image using content and style losses from VGG-19.

**中.** 固定 VGG-19，直接优化生成图像像素。

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

### C4M4 Face Recognition

**EN.** Use FaceNet encodings for one-shot verification and recognition.

**中.** 用 FaceNet 的 128 维编码完成验证与识别。

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

## C5 序列模型

### C5M1 Building an RNN Step by Step

**EN.** Implement vanilla RNN and LSTM cells, sequence forward passes, and optional BPTT.

**中.** 用 NumPy 实现 RNN、LSTM 单元与整段前向，扩展题做时间反传。

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

### C5M1 Dinosaur Land

**EN.** Train a character-level RNN that generates dinosaur names.

**中.** 训练字符级 RNN 生成恐龙名。

- `clip`：逐元素裁剪梯度；
- `sample`：从 softmax 分布采样下一字符；
- `optimize`：前向、损失、反向、裁剪、更新；
- `model`：打乱名字，迭代训练并周期性采样。

> [!EXAMPLE]+ 采样而不是 argmax
>
> 每一步从 $p_t=\mathrm{softmax}(W_ya_t+b_y)$ 按概率抽字符。始终 argmax 会生成确定、单调的名字，无法反映模型分布。梯度裁剪
> $g\leftarrow\min(\max(g,-c),c)$
> 用于抑制长序列的梯度爆炸。

### C5M1 Jazz Improvisation with LSTM

**EN.** Train an LSTM on musical values and build an inference graph that generates a sequence.

**中.** 用 LSTM 学习爵士片段，并建立自回归推理模型。

- 训练模型把长度 $T_x$ 的输入映射到每个时刻的音符分布；
- `djmodel`；
- `music_inference_model`：把当前输出转为下一步输入；
- `predict_and_sample`：把概率转为 one-hot 序列；
- 将生成序列写成 MIDI。

> [!NOTE]+ 解析
>
> 训练时常使用真实前一音符，即 teacher forcing；推理时只能喂回模型刚生成的音符。两种输入分布不同，是 exposure bias 的来源。

### C5M2 Word Vector Representation and Debiasing

**EN.** Implement cosine similarity, analogy completion, neutralization, and equalization.

**中.** 做词向量相似度、类比与偏差消除。

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

### C5M2 Emojify

**EN.** Map short sentences to emoji labels with an average-embedding baseline and an LSTM.

**中.** 把短句分类为 emoji，先做均值词向量基线，再做 LSTM。

- `sentence_to_avg`；
- `model`：均值词向量接 softmax；
- `sentences_to_indices`；
- `pretrained_embedding_layer`；
- 双层 LSTM、dropout 与 dense softmax；
- 比较词序无关基线和序列模型。

> [!EXAMPLE]+ OOV 与 padding
>
> 句子索引矩阵应固定到最大长度，短句补 0，长句截断。未知词不能直接索引字典；应跳过、映射到 OOV，或使用可训练未知向量。均值基线看不到词序，所以 “not happy” 容易被 “happy” 的向量主导。

### C5M3 Neural Machine Translation with Attention

**EN.** Convert human-readable dates to machine-readable dates with an attention model.

**中.** 用注意力模型把自然语言日期转成标准日期。

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

### C5M3 Trigger Word Detection

**EN.** Synthesize audio training examples and detect an activation word in spectrograms.

**中.** 合成语音训练样本，并在频谱图上检测唤醒词。

- `get_random_time_segment`；
- `is_overlapping`；
- `insert_audio_clip`；
- `insert_ones`：唤醒词结束后的若干时间步标 1；
- `create_training_example`；
- 建立 1-D Conv、GRU、dropout、BatchNorm 模型；
- 对真实音频输出触发概率。

> [!EXAMPLE]+ 标签为何延长
>
> 若只在唤醒词结束的单个时间步标 1，正样本极稀疏，模型很难学。作业把后续约 50 个时间步都标为 1，增加正信号。插入片段前必须检查区间重叠，否则合成音频与标签不一致。

### C5M4 Transformer Network

**EN.** Implement positional encoding, attention masks, encoder layers, decoder layers, and a full Transformer.

**中.** 实现位置编码、掩码、编码器层、解码器层与完整 Transformer。

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
