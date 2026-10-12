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
