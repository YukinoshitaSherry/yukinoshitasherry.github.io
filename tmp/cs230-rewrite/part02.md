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

- 训练时 Dropout 必须开启，推理必须关闭。Keras/TF 用 `training=` 切换；推理时若仍除以 $p$ 且随机置零，dev 准确率会无故抖动。
- Inverted dropout 的除以 $p$ 只属于训练分支。
- $L_2$ 与 Dropout 可以叠加，但二者都在压有效容量；同时把 $\lambda$ 和 $1-p$ 开到很大，会回到欠拟合。
- 正交化：先用更大网络把训练误差降下来，再加一种正则观察 dev。

### 其他正则

- **数据增强**（data augmentation）：从已有数据中生成更多数据，是“免费”的样本。图像可以翻转、旋转、随机裁剪、颜色偏移、加噪声、遮挡部分区域、改变对比度；文本可以同义改写。增强通常在训练时实时（on the fly）进行。
  - 增强必须保持标签语义：分类可以水平翻转，但检测框坐标要跟着翻转；文字与左右不对称的医学影像不能盲目翻转；水平翻转车牌会破坏字符顺序。
  - 增强过猛等于换了数据分布；dev 若未增强会出现人为的分布不匹配。
- **Early stopping**：训练过程中监控 dev 损失，在它开始上升时停止，回滚到最优检查点。它把有效训练步数当成超参数，与 $L_2$ 的作用有重叠（训练越早停，权重离初始的小值越近）。缺点是把“优化 $J$”和“防止过拟合”两件事耦合在一起，违背正交化。
- **输入归一化**：每维减均值、除标准差。各维尺度一致后，损失函数的等高线从狭长的椭圆变圆，梯度下降可以用更大的学习率、更快收敛。训练集与测试集必须使用同一组 $\mu,\sigma$。
- **标签平滑**：把 one-hot 换成 $(1-\varepsilon)y+\varepsilon/K$，减轻过度自信。
- **噪声注入（noise injection）**：训练时用 $\tilde{x}=x+\varepsilon$，$\varepsilon\sim\mathcal{N}(0,\sigma^{2}I)$，迫使模型在样本邻域保持稳定。输入噪声、Dropout 和随机增强都属于随机正则（stochastic regularization），但三者作用位置不同。

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

<br>

> [!NOTE]+ 衔接
>
> 到这里已经有了一个可以训练、可以正则、超参数可以搜索的网络。但在真实项目中，“下一步该做什么”往往比“怎么实现”更难：收集更多数据？换更大的网络？调正则？下一部分把训练误差、开发集误差和部署误差分开看，用正交化的思路决定每次只改哪一类问题。

<br>

