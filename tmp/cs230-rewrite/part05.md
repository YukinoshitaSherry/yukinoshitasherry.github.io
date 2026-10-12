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

