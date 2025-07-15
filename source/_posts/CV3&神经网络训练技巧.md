---
title: CV(3)：神经网络训练技巧
date: 2024-02-15
categories:
  - 学AI/DS
tags:
  - CV
desc: CS231n Lec6-8 笔记，资料整合与一些自己的思考。初始化（激活函数、数据预处理、权重初始化、正则化、梯度检查），训练动态（监控学习过程、参数更新、超参数优化），本文讲解训练神经网络的核心方法：优化方式（SGD、动量更新、Nesterov动量、Adagrad、RMSProp、Adam等），正则化（L2、Dropout），迁移学习，模型集成，常见深度学习框架与硬件。

---

- 参考
    - <a href="https://www.showmeai.tech/article-detail/260">`showmeai-斯坦福CS231n教程`</a>


## 激活函数


在全连接层或者卷积层，输入数据与权重相乘后累加的结果送给一个非线性函数，即**激活函数**（activation function）。每个激活函数的输入都是一个数字，然后对其进行某种固定的数学操作。

  

下面是在实践中可能遇到的几种激活函数：

### Sigmoid函数


**数学公式**：$\sigma(x) = 1 / (1 + e^{-x})$

**求导公式**：$\frac{d\sigma(x)}{dx} = (1 - \sigma(x)) \sigma(x)$ （不小于 $0$ ）

**特点**：把输入值「挤压」到 $0$ 到 $1$ 范围内。Sigmoid 函数把输入的实数值「挤压」到 $0$ 到 $1$ 范围内，很大的负数变成 $0$，很大的正数变成 $1$，在历史神经网络中，Sigmoid 函数很常用，因为它对神经元的激活频率有良好的解释：**从完全不激活（$0$）到假定最大频率处的完全饱和（saturated）的激活（$1$）** 。

  

然而现在 Sigmoid 函数已经很少使用了，因为它有三个主要缺点：

**缺点①：Sigmoid 函数饱和时使梯度消失**

- 当神经元的激活在接近 $0$ 或 $1$ 处时（即门单元的输入过或过大时）会饱和：在这些区域，梯度几乎为 $0$。
- 在反向传播的时候，这个局部梯度要与损失函数关于这个门单元输出的梯度相乘。因此，如果局部梯度非常小，那么相乘的结果也会接近零，这会「杀死」梯度，几乎就有没有信号通过神经元传到权重再到数据了。
- 还有，为了防止饱和，必须对于权重矩阵初始化特别留意。比如，如果初始权重过大，那么大多数神经元将会饱和，导致网络就几乎不学习了。

**缺点②：Sigmoid 函数的输出不是零中心的**

- 这个性质会导致神经网络后面层中的神经元得到的数据不是零中心的。
- 这一情况将影响梯度下降的运作，因为如果输入神经元的数据总是正数（比如在 $\sigma(\sum_{i}w_ix_i+b)$ )中每个输入 $x$ 都有 $x > 0$），那么关于 $w$ 的梯度在反向传播的过程中，将会要么全部是正数，要么全部是负数（根据该 Sigmoid 门单元的回传梯度来定，回传梯度可正可负，而 $\frac{d\sigma}{dW}=X^T \cdot\sigma'$ 在 $X$ 为正时恒为非负数）。
- 这将会导致梯度下降权重更新时出现 $z$ 字型的下降。该问题相对于上面的神经元饱和问题来说只是个小麻烦，没有那么严重。

**缺点③： 指数型计算量比较大**。

### tanh函数

  

**数学公式**：$\tanh(x) = 2 \sigma(2x) -1$

**特点**：将实数值压缩到 $[-1,1]$ 之间

和 Sigmoid 神经元一样，它也存在饱和问题，但是和 Sigmoid 神经元不同的是，它的输出是零中心的。因此，在实际操作中，tanh 非线性函数比 Sigmoid 非线性函数更受欢迎。注意 tanh 神经元是一个简单放大的 Sigmoid 神经元。

### ReLU 函数

  

**数学公式**：$f(x) = \max(0, x)$

**特点**：一个关于 $0$ 的阈值

**优点**：

- ReLU 只有负半轴会饱和；节省计算资源，不含指数运算，只对一个矩阵进行阈值计算；更符合生物学观念；加速随机梯度下降的收敛。
- Krizhevsky论文指出比 Sigmoid 和 tanh 函数快6倍之多，据称这是由它的线性，非饱和的公式导致的。

**缺点**：

- 仍有一半会饱和；非零中心；
- 训练时，ReLU 单元比较脆弱并且可能「死掉」。
    - 举例来说，当一个很大的梯度流过 ReLU 的神经元的时候，由于梯度下降，可能会导致权重更新到一种特别的状态（比如大多数的 $w$ 都小于 $0$ ），在这种状态下神经元将无法被其他任何数据点再次激活。如果这种情况发生，那么从此所有流过这个神经元的梯度将都变成 $0$，也就是说，这个 ReLU 单元在训练中将不可逆转的死亡，因为这导致了数据多样化的丢失。
    - 例如，如果学习率设置得太高（本来大多数大于 $0$ 的 $w$ 更新后都小于 $0$ 了），可能会发现网络中40%的神经元都会死掉（在整个训练集中这些神经元都不会被激活）。
    - 通过合理设置学习率，这种情况的发生概率会降低。

### Leaky ReLU

  

**公式**：$f(x) = \mathbb{1}(x < 0) (\alpha x) + \mathbb{1}(x\geq0) (x)$，$\alpha$ 是小常量

**特点**：解决「 ReLU 死亡」问题，$x<0$ 时给出一个很小的梯度值，比如 $0.01$。

Leaky ReLU 修正了 $x<0$ 时 ReLU 的问题，有研究指出这个激活函数表现很不错，但是其效果并不是很稳定。Kaiming He等人在2015年发布的论文 Delving Deep into Rectifiers中介绍了一种新方法 PReLU，把负区间上的斜率当做每个神经元中的一个参数，然而无法确定该激活函数在不同任务中均有益处。

### 指数线性单元（Exponential Linear Units，ELU）

  

**公式**：$f(x)=\begin{cases} x & if\space\space x>0 \\ \alpha(exp(x)-1) & otherwise \end{cases}$

**特点**：介于 ReLU 和Leaky ReLU 之间

具有 ReLU 的所有优点，但是不包括计算量；介于 ReLU 和 Leaky ReLU 之间，有负饱和的问题，但是对噪声有较强的鲁棒性。

### Maxout

$\max \left(w_{1}^T x+b_{1}, w_{2}^T x+b_{2}\right)$

**公式**：$max(w_1^Tx+b_1, w_2^Tx + b_2)$

**特点**：是对 ReLU 和 leaky ReLU 的一般化归纳

对于权重和数据的内积结果不再使用非线性函数，直接比较两个线性函数。ReLU 和 Leaky ReLU 都是这个公式的特殊情况，比如 ReLU 就是当 $w_1=1$，$b_1=0$ 的时候。

Maxout 拥有 ReLU 单元的所有优点（线性操作和不饱和），而没有它的缺点（死亡的 ReLU 单元）。然而和 ReLU 对比，它每个神经元的参数数量增加了一倍，这就导致整体参数量激增。

  

**实际应用Tips** ：

- **用 ReLU 函数**。注意设置好学习率，你可以监控你的网络中死亡的神经元占的比例。
- 如果单元死亡问题困扰你，就试试Leaky ReLU 或者 Maxout，不要再用 Sigmoid 了。也可以试试 tanh，但是其效果应该不如 ReLU 或者 Maxout。

## 数据预处理


关于数据预处理有 3 个常用的符号，数据矩阵 $X$，假设其尺寸是 $[N \times D]$（$N$ 是数据样本的数量，$D$ 是数据的维度）。

### 减均值（Mean Subtraction）

**减均值法**是数据预处理最常用的形式。它对数据中每个独立特征减去平均值，在每个维度上都将数据的中心都迁移到原点。

  

在 numpy 中，该操作可以通过代码 `X -= np.mean(X, axis=0)` 实现。而对于图像，更常用的是对所有像素都减去一个值，可以用 `X -= np.mean(X)` 实现，也可以在 3 个颜色通道上分别操作。

  

具体来讲，假如训练数据是 $50000$ 张 $32 \times 32 \times 3$ 的图片：

- 第一种做法是减去均值图像，即将每张图片拉成长为 $3072$ 的向量，$50000 \times 3072$ 的矩阵按列求平均，得到一个含有 $3072$ 个数的均值图像，训练集测试集验证集都要减去这个均值，AlexNet 是这种方式；
- 第二种做法是按照通道求平均，RGB三个通道每个通道一个均值，即每张图片的 $3072$ 个数中，RGB各有 $32 \times 32$ 个数，要在 $50000 \times 32 \times 32$ 个数中求一个通道的均值，最终的均值有 $3$ 个数字，然后所有图片每个通道都要减去对应的通道均值，VGGNet是这种方式。

之所以执行减均值操作，是因为解决输入数据大多数都是正或者负的问题。虽然经过这种操作，数据变成零中心的，但是仍然只能第一层解决 Sigmoid 非零均值的问题，后面会有更严重的问题。

### 归一化（Normalization）

**归一化**是指将数据的所有维度都归一化，使其数值范围都近似相等。

有两种常用方法可以实现归一化。

- 第一种是先对数据做零中心化（zero-centered）处理，然后每个维度都除以其标准差，实现代码为 `X /= np.std(X, axis=0)`。
- 第二种是对每个维度都做归一化，使得每个维度的最大和最小值是 $1$ 和 $-1$。这个预处理操作只有在确信不同的输入特征有不同的数值范围（或计量单位）时才有意义，但要**注意预处理操作的重要性几乎等同于学习算法本身**。

在图像处理中，由于像素的数值范围几乎是一致的（都在0-255之间），所以进行这个额外的预处理步骤并不是很必要。

  

- **左边**：原始的 2 维输入数据。
- **中间**：在每个维度上都减去平均值后得到零中心化数据，现在数据云是以原点为中心的。
- **右边**：每个维度都除以其标准差来调整其数值范围，红色的线指出了数据各维度的数值范围。

在中间的零中心化数据的数值范围不同，但在右边归一化数据中数值范围相同。

### 主成分分析（PCA）

这是另一种机器学习中比较常用的预处理形式，但在图像处理中基本不用。在这种处理中，先对数据进行零中心化处理，然后计算协方差矩阵，它展示了数据中的相关性结构。

  

```python
# 假设输入数据矩阵X的尺寸为[N x D]
X -= np.mean(X, axis = 0) # 对数据进行零中心化(重要)
cov = np.dot(X.T, X) / X.shape[0] # 得到数据的协方差矩阵，DxD
```

数据协方差矩阵的第 $(i, j)$ 个元素是数据第 $i$ 个和第 $j$ 个维度的**协方差**。具体来说，该矩阵的对角线上的元素是方差。还有，协方差矩阵是对称和半正定的。我们可以对数据协方差矩阵进行 SVD（奇异值分解）运算。

```python
U,S,V = np.linalg.svd(cov)
```

$U$ 的列是特征向量，$S$ 是装有奇异值的1维数组（因为 cov 是对称且半正定的，所以S中元素是特征值的平方）。为了去除数据相关性，将已经零中心化处理过的原始数据投影到特征基准上：

```python
Xrot = np.dot(X,U) # 对数据去相关性
```

`np.linalg.svd` 的一个良好性质是在它的返回值U中，特征向量是按照特征值的大小排列的。我们可以利用这个性质来对数据降维，只要使用前面的小部分特征向量，丢弃掉那些包含的数据没有方差的维度，这个操作也被称为 **主成分分析（Principal Component Analysis 简称PCA）** 降维：

```python
Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced 变成 [N x 100]
```

经过上面的操作，将原始的数据集的大小由 $[N \times D]$ 降到了 $[N \times 100]$，留下了数据中包含最大方差的的 100 个维度。通常使用 PCA 降维过的数据训练线性分类器和神经网络会达到非常好的性能效果，同时还能节省时间和存储器空间。

> **有一问题是为什么使用协方差矩阵进行 SVD 分解而不是使用原 $X$ 矩阵进行？**
> 
> 其实都是可以的，只对数据 $X$（可以不是方阵）进行 SVD 分解，做 PCA 降维（避免了求协方差矩阵）的话一般用到的是右奇异向量 $V$，即 $V$ 的前几列是需要的特征向量（注意 `np.linalg.svd` 返回的是 `V.T`）。$X$ 是$N \times D$，则 $U$ 是 $N \times N$，$V$ 是 $D \times D$；而对协方差矩阵（$D \times D$）做 SVD 分解用于 PCA 降维的话，可以随意取左右奇异向量$U$、$V$（都是 $D \times D$）之一，因为两个向量是一样的。

### 白化（Whitening）

最后一个在实践中会看见的变换是**白化**（whitening）。白化操作的输入是特征基准上的数据，然后对每个维度除以其特征值来对数值范围进行归一化。

  

白化变换的**几何解释**是：如果数据服从多变量的高斯分布，那么经过白化后，数据的分布将会是一个均值为零，且协方差相等的矩阵。

该操作的代码如下：

```python
# 对数据进行白化操作:
# 除以特征值 
Xwhite = Xrot / np.sqrt(S + 1e-5)
```

注意分母中添加了 `1e-5`（或一个更小的常量）来防止分母为 $0$，该变换的一个缺陷是在变换的过程中可能会夸大数据中的噪声，这是因为它将所有维度都拉伸到相同的数值范围，这些维度中也包含了那些只有极少差异性(方差小)而大多是噪声的维度。

在实际操作中，这个问题可以用更强的平滑来解决（例如：采用比 `1e-5` 更大的值）。

  

从左往右4张子图：

- **第1张**：一个用于演示的图片集合，含 49 张图片。
- **第2张**：3072 个特征向量中的前 144 个。靠前面的特征向量解释了数据中大部分的方差。
- **第3张**：49 张经过了PCA降维处理的图片，只使用这里展示的这 144 个特征向量。为了让图片能够正常显示，需要将 144 维度重新变成基于像素基准的 3072 个数值。因为U是一个旋转，可以通过乘以 `U.transpose()[:144,:]` 来实现，然后将得到的 3072 个数值可视化。可以看见图像变得有点模糊了，然而，大多数信息还是保留了下来。
- **第4张**：将「白化」后的数据进行显示。其中 144个 维度中的方差都被压缩到了相同的数值范围。然后 144 个白化后的数值通过乘以 `U.transpose()[:144,:]` 转换到图像像素基准上。

### 实际应用

实际上在卷积神经网络中并不会采用PCA和白化，对数据进行零中心化操作还是非常重要的，对每个像素进行归一化也很常见。

**补充说明**：

进行预处理很重要的一点是：**任何预处理策略（比如数据均值）都只能在训练集数据上进行计算，然后再应用到验证集或者测试集上**。

- 一个常见的错误做法是先计算整个数据集图像的平均值然后每张图片都减去平均值，最后将整个数据集分成训练/验证/测试集。**正确的做法是先分成训练/验证/测试集，只是从训练集中求图片平均值，然后各个集（训练/验证/测试集）中的图像再减去这个平均值**。


## 更好的优化（参数更新）

在训练中每一步迭代都使用训练集的所有内容 $\{x_1, \cdots ,x_n\}$ 以及每个样本对应的输出 $y_i$，用于计算损失和梯度然后使用梯度下降更新参数。

当在整个数据集上进行计算时，只要学习率足够低，总是能在损失函数上得到非负的进展。参考代码如下（其中 `learning_rate` 是一个超参数）：

```python
# 普通梯度下降
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += -learning_rate * weights_grad  # 参数更新
```

- **优点**：由于每一步都利用了训练集中的所有数据，因此当损失函数达到最小值以后，能够保证此时计算出的梯度为 $0$，能够收敛。因此，使用BGD时不需要逐渐减小学习率。
- **缺点**：随着数据集的增大，运行速度会越来越慢。

### 随机梯度下降（SGD）

每次迭代随机抽取一批样本 $\{x_1, \cdots ,x_m\}$ 及 $y_i$，以此来反向传播计算出梯度，然后向负梯度方向更新参数。

SGD的优点是训练速度快，对于很大的数据集，也能够以较快的速度收敛。但是实际应用SGD会有很多问题：

**① 如果损失函数在一个参数方向下降的快另一个方向下降的慢，这样会导致 「 之字形 」下降到最低点，高维中很普遍**。

- 上图是一个山沟状的区域，损失最小点沿着蓝色的线方向。考虑表面上的一个点A梯度，该点的梯度可以分解为两个分量，一个沿着方向 $w_1$，另一个沿着 $w_2$。
- 梯度在 $w_1$ 方向上的分量要大得多，因为在 $w_1$ 方向上每走一步都比在 $w_2$ 方向损失值下降的多，虽然最小值在 $w_2$ 方向上。这样实际走一步在 $w_1$ 方向走的多， $w_2$ 走得少，就会导致在这个沟里反复震荡，「之字形」前往最小值点。

**② 如果损失函数有局部极小值和鞍点（既不是极大值也不是极小值的临界点）时，此时的梯度为$0$，参数更新会卡住，或在极小值附近震荡**。

- 在高维数据中，鞍点的存在是个更普遍也更大的问题，极小值每个梯度方向损失都会变大，而鞍点有的方向变大，有的减小，接近鞍点时更新缓慢。

**③ SGD具有随机性，我们的梯度来自小批量数据（使用全部数据计算真实梯度速度太慢了），可能会有噪声，这样梯度下降的路线会很曲折，收敛的慢**。

下面有一些「**小批量梯度下降**」基础上的优化算法。

### 动量（Momentum）更新

损失值可以理解为是山的高度（因此高度势能是 $U=mgh$），用随机数字初始化参数等同于在某个位置给质点设定初始速度为 $0$ ，这样最优化过程可以看做是参数向量（即质点）在地形上滚动的过程。

质点滚动的力来源于高度势能 $F = - \nabla U$，即损失函数的负梯度。又因为 $F=ma$，质点的加速度和负梯度成正比，所以负梯度方向速度是逐渐增加的。

在 SGD 中，梯度直接影响质点的位置，在梯度为 $0$ 的地方，位置就不会更新了；而在这里，梯度作为作用力影响的是速度，速度再改变位置，即使梯度为 $0$ ，但之前梯度累积下来的速度还在，一般而言，一个物体的动量指的是这个物体在它运动方向上保持运动的趋势，所以此时质点还是有动量的，位置仍然会更新，这样就可以冲出局部最小值或鞍点，继续更新参数。但是必须要给质点的速度一个衰减系数或者是摩擦系数，不然因为能量守恒，质点在谷底会不停的运动。

也就是说，参数更新的方向，不仅由当前点的梯度方向决定，而且由此前累积的梯度方向决定。

计算过程也是每次迭代随机抽取一批样本 $\{x_1, \cdots ,x_m\}$ 及 $y_i$，计算梯度和损失，并更新速度和参数（假设质量为1，v即动量）：

```python
v=0
while True:
    dW =  compute_gradient(W, X_train, y_train)
    v = rho * v - learning_rate * dW
    W += v
```

- `rho` 表示每回合速度 `v` 的衰减程度，每次迭代得到的梯度都是 `dW` 那么最后得到的 `v` 的稳定值为：$\frac{-learning_{rate} * dw}{1-rho}$
- `rho` 为 $0$ 时表示 SGD，`rho` 一般取值 $0.5$、$0.9$、$0.99$，对应学习速度提高两倍、10倍和100倍。

动量更新可以很好的解决上述 SGD 的几个问题：

- 由于参数的更新要累积之前的梯度，所以如果我们分别累加这些梯度的两个分量，那么 $w_1$ 方向上的分量将互相抵消，而 $w_2$ 方向上的分量得到了加强。 但是由于衰减系数，不可能完全抵消，但是已经可以加速通过，很大程度缓解了「之字形」收敛慢的问题。这也是减少震荡的原理。
- 局部最小值和鞍点由于还有之前的速度，会加速冲过去。
- 面对梯度变化比较大的方向，即一些噪声，由于此时质点还有比较大的速度，这时的反方向需要先将速度减小为 $0$ 才能改变参数更新方向，由于速度是累加的，所以个别的噪声的影响不会那么大，就可以平滑快速的收敛。

### Nesterov动量

Nesterov动量与普通动量有些许不同，最近变得比较流行。在理论上对于凸函数它能得到更好的收敛，在实践中也确实比标准动量表现更好一些。

- **普通的动量更新**在某一点处有一个速度，然后计算该点的梯度，实际的更新方向会根据速度方向和梯度方向做一个权衡。
- **Nesterov动量更新**是既然我们知道动量将会将质点带到一个新的位置（即向前看），我们就不要在原来的位置计算梯度了，在这个「向前看」的地方计算梯度，更新参数。

这样代码变为：

```python
v=0
while True:
    W_ahead = W + rho * v
    dW_ahead =  compute_gradient(W_ahead, X_train, y_train)
    v = rho * v - learning_rate * dW_ahead
    W += v
```

动量还是之前的动量，只是梯度变成将来的点的梯度。

而在实践中，人们更喜欢和普通SGD或普通的动量方法一样简单的表达式。通过对 `W_ahead = W + rho * v` 使用变量变换进行改写是可以做到的，然后用 `W_ahead` 而不是 `W` 来表示上面的更新。

也就是说，实际存储的参数总是向前一步的那个版本。 代码如下：

```python
v=0
while True:
    pre_v = v
    dW =  compute_gradient(W, X_train, y_train)
    v = rho * v - learning_rate * dW
    W += -rho * pre_v + (1 + rho) * v
```

推导过程如下：

最初的 Nesterov 动量可以用下面的数学表达式代替：

$$v_{t+1}=\rho v_t - \alpha \nabla f(x_t+\rho v_t)$$
$$x_{t+1}=x_t+v_{t+1}$$

现在令 $\tilde{x}_t =x_t+\rho v_t$，则：

$$v_{t+1}=\rho v_t-\alpha \nabla f(\tilde{x_t})$$
$$\begin{aligned} \tilde{x}_{t+1} &=x_{t+1}+\rho v_{t+1}\\ &=x_{t}+v_{t+1}+\rho v_{t+1}\\ &=\tilde{x}_{t}-\rho v_{t}+v_{t+1}+\rho v_{t+1} \end{aligned}$$

从而有：

$$\tilde{x}_{t+1}=\tilde{x_t}-\rho v_t+(\rho+1)v_{t+1}$$

- 只更新 $v_t$ 和 $\tilde{x}_t$ 即可

### 自适应梯度算法（Adagrad）

代码如下：

```python
eps = 1e-7
grad_squared = 0 
while True:
    dW = compute_gradient(W)
    grad_squared += dW * dW
    W -= learning_rate * dW / (np.sqrt(grad_squared) + eps)
```

AdaGrad 其实很简单，就是将每一维各自的历史梯度的平方叠加起来，然后更新的时候除以该历史梯度值即可。

变量 `grad_squared` 的尺寸和梯度矩阵的尺寸是一样的，用于累加每个参数的梯度的平方和。这个将用来归一化参数更新步长，归一化是逐元素进行的。`eps`（一般设为 `1e-4` 到 `1e-8` 之间）用于平滑，防止出现除以 $0$ 的情况。

- **优点**：能够实现参数每一维的学习率的自动更改，如果某一维的梯度大，那么学习速率衰减的就快一些，延缓网络训练；如果某一维的梯度小，那么学习速率衰减的就慢一些，网络训练加快。
- **缺点**：如果梯度累加的很大，学习率就会变得非常小，就会陷在局部极小值点或提前停（ RMSProp 算法可以很好的解决该问题）。

### 均方根支柱算法（RMSProp）

RMSProp算法在AdaGrad基础上引入了衰减因子，RMSProp在梯度累积的时候，会对「过去」与「现在」做一个平衡，通过超参数 `decay_rate` 调节衰减量，常用的值是 $[0.9,0.99,0.999]$。其他不变，只是 `grad_squared` 类似于动量更新的形式：

```python
grad_squared =  decay_rate * grad_squared + (1 - decay_rate) * dx * dx
```

相比于AdaGrad，这种方法很好的解决了训练过早结束的问题。和 Adagrad 不同，其更新不会让学习率单调变小。

### 自适应-动量优化（Adam）

动量更新在SGD基础上增加了一阶动量，AdaGrad和RMSProp在SGD基础上增加了二阶动量。把一阶动量和二阶动量结合起来，就得到了Adam优化算法：**Adaptive + Momentum**。

代码如下：

```python
eps = 1e-8
first_moment = 0  # 第一动量，用于累积梯度，加速训练
second_moment = 0  # 第二动量，用于累积梯度平方，自动调整学习率
while True:
    dW = compute_gradient(W)
    first_moment = beta1 * first_moment + (1 - beta1) * dW  # Momentum
    second_moment = beta2 * second_moment + (1 - beta2) * dW * dW  # AdaGrad / RMSProp
    W -= learning_rate * first_moment / (np.sqrt(second_moment) + eps)
```

上述参考代码看起来像是 RMSProp 的动量版，但是这个版本的 Adam 算法有个问题：第一步中 `second_monent` 可能会比较小，这样就可能导致学习率非常大，所以完整的 Adam 需要加入偏置。

代码如下：

```python
eps = 1e-8
first_moment = 0  # 第一动量，用于累积梯度，加速训练
second_moment = 0  # 第二动量，用于累积梯度平方，自动调整学习率

for t in range(1, num_iterations+1):
    dW = compute_gradient(W)
    first_moment = beta1 * first_moment + (1 - beta1) * dW  # Momentum
    second_moment = beta2 * second_moment + (1 - beta2) * dW * dW  # AdaGrad / RMSProp
    first_unbias = first_moment / (1 - beta1 ** t)  # 加入偏置，随次数减小，防止初始值过小
    second_unbias = second_moment / (1 - beta2 ** t)
    W -= learning_rate * first_unbias / (np.sqrt(second_unbias) + eps)
```

论文中推荐的参数值 `eps=1e-8`, `beta1=0.9`, `beta2=0.999`, `learning_rate = 1e-3`或`5e-4`，对大多数模型效果都不错。

在实际操作中，我们推荐 Adam 作为默认的算法，一般而言跑起来比 RMSProp 要好一点。

### 学习率退火

通常，实现学习率衰减有3种方式：

**① 随步数衰减**：每进行几个周期（epoch）就根据一些因素降低学习率。典型的值是每过 5 个周期就将学习率减少一半，或者每 20 个周期减少到之前的 10%。

**② 指数衰减**：数学公式是 $\alpha=\alpha_0e^{-kt}$，其中 $\alpha_0,k$ 是超参数， $t$ 是迭代次数（也可以使用周期作为单位）。

**③ 1/t 衰减**：数学公式是 $\alpha=\alpha_0/(1+kt)$，其中 $\alpha_0,k$ 是超参数， $t$ 是迭代次数。

> 一般像SGD这种需要使用学习率退火，Adam等不需要。也不要一开始就使用，先不用，观察一下损失函数，然后确定什么地方需要减小学习率。

### 二阶方法（Second-Order）

在深度网络背景下，第二类常用的最优化方法是基于牛顿方法的，其迭代如下：

$x \leftarrow x - [H f(x)]^{-1} \nabla f(x)$

$H f(x)$ 是 Hessian 矩阵，由 $f(x)$ 的二阶偏导数组成：

$$\mathbf{H}=\left[\begin{array}{cccc} \frac{\partial^{2} f}{\partial x_{1}^{2}} & \frac{\partial^{2} f}{\partial x_{1} \partial x_{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{1} \partial x_{n}} \\ \frac{\partial^{2} f}{\partial x_{2} \partial x_{1}} & \frac{\partial^{2} f}{\partial x_{2}^{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{2} \partial x_{n}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^{2} f}{\partial x_{n} \partial x_{1}} & \frac{\partial^{2} f}{\partial x_{n} \partial x_{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{n}^{2}} \end{array}\right]$$

$x$ 是 $n$ 维的向量，$f(x)$ 是实数，所以海森矩阵是 $n * n$ 的。

$\nabla f(x)$ 是 $n$ 维梯度向量，这和反向传播一样。

这个方法收敛速度很快，可以进行更高效的参数更新。在这个公式中是没有学习率这个超参数的，这相较于一阶方法是一个巨大的优势。

然而上述更新方法很难运用到实际的深度学习应用中去，这是因为计算（以及求逆）Hessian 矩阵操作非常耗费时间和空间。这样，各种各样的**拟-牛顿法**就被发明出来用于近似转置 Hessian 矩阵。

在这些方法中最流行的是L-BFGS，该方法使用随时间的梯度中的信息来隐式地近似（也就是说整个矩阵是从来没有被计算的）。

然而，即使解决了存储空间的问题，L-BFGS应用的一个巨大劣势是需要对整个训练集进行计算，而整个训练集一般包含几百万的样本。和小批量随机梯度下降（mini-batch SGD）不同，让 L-BFGS 在小批量上运行起来是很需要技巧，同时也是研究热点。

### 实际应用

> Tips：默认选择Adam；如果可以承担全批量更新，可以尝试使用L-BFGS。

## 正则化

### 正则化的动机

当我们增加神经网络隐藏层的数量和尺寸时，网络的容量会上升，即神经元可以合作表达许多复杂函数。例如，如果有一个在二维平面上的二分类问题。我们可以训练 3 个不同的神经网络，每个网络都只有一个隐藏层，但是隐藏层的神经元数目不同。

在上图中，可以看见有更多神经元的神经网络可以表达更复杂的函数。然而这既是优势也是不足：

- 优势是可以分类更复杂的数据
- 不足是可能造成对训练数据的过拟合。

**过拟合（Overfitting）** 是网络对数据中的噪声有很强的拟合能力，而没有重视数据间（假设）的潜在基本关系。

那是不是说 「**如果数据不是足够复杂，则小一点的网络似乎更好，因为可以防止过拟合**」？

不是的，防止神经网络的过拟合有很多方法（L2正则化，Dropout和输入噪音等）。在实践中，使用这些方法来控制过拟合比减少网络神经元数目要好得多。

不应该因为害怕出现过拟合而使用小网络。相反，应该尽可能使用大网络，然后使用正则化技巧来控制过拟合。

上图每个神经网络都有 20 个隐藏层神经元，但是随着正则化强度增加，网络的决策边界变得更加平滑。所以，**正则化强度是控制神经网络过拟合的好方法**。

### 正则化方法

有不少方法是通过控制神经网络的容量来防止其过拟合的：

**L2正则化**：最常用的正则化，通过惩罚目标函数中所有参数的平方实现。

- 对于网络中的每个权重 $w$，向目标函数中增加一个 $\frac{1}{2} \lambda w^2$，1/2为了方便求导， $\lambda$ 是正则强度。
- L2 正则化可以直观理解为它对于大数值的权重向量进行严厉惩罚，倾向于更加分散的权重向量。使网络更倾向于使用所有输入特征，而不是严重依赖输入特征中某些小部分特征。

**L1正则化**：是另一个相对常用的正则化方法，对于每个 $w$ 都向目标函数增加一个 $\lambda | w |$。

- L1 正则化会让权重向量在最优化的过程中变得稀疏（即非常接近 $0$）。在实践中，如果不是特别关注某些明确的特征选择，一般说来 L2 正则化都会比 L1 正则化效果好。
- L1 和 L2 正则化也可以进行组合： $\lambda_1 | w | + \lambda_2 w^2$，称作 Elastic net regularization。

**最大范式约束**（Max norm constraints）：要求权重向量 $w$ 必须满足 L2 范式 $\Vert \vec{w} \Vert_2 < c$，$c$ 一般是 3 或 4。这种正则化还有一个良好的性质，即使在学习率设置过高的时候，网络中也不会出现数值「爆炸」，这是因为它的参数更新始终是被限制着的。

但是在神经网络中，最常用的正则化方式叫做 Dropout，下面我们详细展开介绍一下。

### 随机失活（Dropout）

#### Dropout概述

Dropout 是一个简单又极其有效的正则化方法，由 Srivastava 在论文中提出，与 L1 正则化、L2 正则化和最大范式约束等方法互为补充。

在训练的时候，随机失活的实现方法是让神经元以超参数 $p$ （一般是 $0.5$）的概率被激活或者被设置为 $0$ 。常用在全连接层。

一个三层的神经网络 Dropout 示例代码实现：

```python
""" 普通版随机失活"""
p = 0.5   # 神经元被激活的概率。p值越高，失活数目越少

def train_step(X):
  """ X中是输入数据 """
  # 前向传播
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # 第一个随机失活掩模
  H1 *= U1 # U1中False的H1对应位置置零
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # 第二个随机失活掩模
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
```

在上面的代码中，`train_step` 函数在第一个隐层和第二个隐层上进行了两次随机失活。在输入层上面进行随机失活也是可以的，为此需要为输入数据 $X$ 创建一个二值（要么激活要么失活）的掩模。反向传播几乎保持不变，只需回传梯度乘以掩模得到 Dropout 层的梯度。

#### Dropout的理解

为什么这个想法可取呢？一个解释是防止特征间的相互适应：

- 比如每个神经元学到了猫的一个特征比如尾巴、胡须、爪子等，将这些特征全部组合起来可以判断是一只猫。
- 加入随机失活后就只能依赖一些零散的特征去判断不能使用所有特征，这样可以一定程度上抑制过拟合。不然训练时正确率很高，测试时却很低。

另一个比较合理的解释是：

- 在训练过程中，随机失活可以被认为是对完整的神经网络抽样出一些子集，每次基于输入数据只更新子网络的参数。
- 每个二值掩模都是一个模型，有 $n$ 个神经元的网络有 $2^n$ 种掩模。Dropout 相当于数量巨大的网络模型（共享参数）在同时被训练。

#### 测试时避免随机失活

在训练过程中，失活是随机的，但是在测试过程中要避免这种随机性，所以不使用随机失活，要对数量巨大的子网络们做模型集成（model ensemble），以此来计算出一个预测期望。

比如只有一个神经元 $a$：

测试的时候由于不使用随机失活所以：

$\text{E}(a)=w_1x+w_2y$

假如训练时随机失活的概率为 $0.5$，那么：

$$\text{E}(a)=\frac{1}{4}(w_1x+w_2y)+\frac{1}{2}(w_1x+w_2\cdot 0)+\frac{1}{2}(w_1x\cdot 0+w_2)+\frac{1}{4}\cdot 0=\frac{1}{2}(w_1x+w_2y)$$

所以一个不确切但是很实用的做法是在测试时承随机失活概率，这样就能保证预测时的输出和训练时的期望输出一致。所以测试代码：

```python
def predict(X):
  # 前向传播时模型集成
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # 注意：激活数据要乘以p
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # 注意：激活数据要乘以p
  out = np.dot(W3, H2) + b3
```

上述操作不好的地方是必须在测试时对激活数据按照失活概率 $p$ 进行数值范围调整。测试阶段性能是非常关键的，因此实际操作时更倾向使用**反向随机失活（inverted dropout）**

- 在训练时就进行数值范围调整，从而让前向传播在测试时保持不变。

反向随机失活还有一个好处，无论是否在训练时使用 Dropout，预测的代码可以保持不变。参考实现代码如下：

```python
"""
反向随机失活: 推荐实现方式.
在训练的时候drop和调整数值范围，测试时不做任何事.
"""
p = 0.5
def train_step(X):
  # 前向传播
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # 第一个随机失活遮罩. 注意/p!
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # 第二个随机失活遮罩. 注意/p!
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3

def predict(X):
  # 前向传播时模型集成
  H1 = np.maximum(0, np.dot(W1, X) + b1) # 不用数值范围调整了
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
```

在更一般化的分类上，随机失活属于网络在前向传播中有随机行为的方法。这种在训练过程加入随机性，然后在测试过程中对这些随机性进行平均或近似的思想在很多地方都能见到：

- **批量归一化**：训练时的均值和方差来自随机的小批量；测试时使用的是整个训练过程中的经验方差和均值。
- **数据增强（data augmentation）** ：比如一张猫的图片进行训练时，可以随机的裁剪翻转等操作再训练，然后测试过程再对一些固定的位置（四个角、中心及翻转）进行测试。也可以在训练的时候随机改变亮度对比度，色彩抖动PCA降维等。
- **DropConnect**：另一个与 Dropout 类似的研究是 DropConnect，它在前向传播的时候，一系列权重被随机设置为 $0$ 。
- **部分最大池化（Fractional Max Pooling）** ：训练时随机区域池化，测试时固定区域或者取平均值。这个方法并不常用。
- **随机深度（Stochastic Depth）** ：一个比较深的网络，训练时随机选取部分层去训练，测试时使用全部的层。这个研究非常前沿。

总之，这些方法都是在训练的时候增加随机噪声，测试时通过分析法（在使用随机失活的本例中就是乘以 $p$）或数值法（例如通过抽样出很多子网络，随机选择不同子网络进行前向传播，最后对它们取平均）将噪音边缘化。

#### 实践经验

一些常用的实践经验方法：

- 可以通过交叉验证获得一个全局使用的 L2 正则化系数。
- 使用 L2 正则化的同时在所有层后面使用随机失活

**随机失活 $p$值一般默认设为 $0.5$，也可能在验证集上调参**。

## 迁移学习（Transfer Learning）

另一个导致过拟合的原因可能是训练样本过少，这时可以使用迁移学习来解决这个问题，它允许使用很少的数据来训练 CNN。

### 迁移学习的思想

- **第①步**：在大量的数据集上训练一个 CNN，得到模型（比如使用 ImageNet，有 1000 个分类）
- **第②步**：使用一个少量的数据集，最后需要的得到的分类也不再是 1000 而是一个较小的值 $C$，比如10。这时最后一个全连接层的参数矩阵变成 $4096 \times C$，初始化这个矩阵，重新训练这个线性分类器，保持前面的所有层不变，因为前面的层已经训练好了，有了泛化能力。
- **第③步**：当得到较多的训练集后，训练的层数可以增多，比如可以训练最后三个全连接层。可以使用较低的学习率微调参数。

### 应用

在目标检测和图像标记中都会使用迁移学习，图像处理部分都使用一个已经用 ImageNet 数据预训练好的 CNN 模型，然后根据具体的任务微调这些参数。

所以对一批数据集感兴趣但是数量不够时，可以在网上找一个数据很相似的有大量数据的训练模型，然后针对自己的问题微调或重新训练某些层。一些常用的深度学习软件包都含有已经训练好的模型，直接应用就好。

- **Caffe**: https://github.com/BVLC/caffe/wiki/Model-Zoo
- **TensorFlow**: https://github.com/tensorflow/models
- **PyTorch**: https://github.com/pytorch/vision

## 模型集成（Model Ensembles）

在实践的时候，有一个总是能提升神经网络几个百分点准确率的办法，就是在训练的时候训练几个独立的模型，然后在测试的时候平均它们预测结果。

集成的模型数量增加，算法的结果也单调提升（但提升效果越来越少）。

模型之间的差异度越大，提升效果可能越好。

进行集成有以下几种方法：

- **同一个模型，不同的初始化**。使用交叉验证来得到最好的超参数，然后用最好的参数来训练不同初始化条件的模型。这种方法的风险在于多样性只来自于不同的初始化条件。
- **在交叉验证中发现最好的模型**。使用交叉验证来得到最好的超参数，然后取其中最好的几个（比如10个）模型来进行集成。这样就提高了集成的多样性，但风险在于可能会包含不够理想的模型。在实际操作中，这样操作起来比较简单，在交叉验证后就不需要额外的训练了。
- **一个模型设置多个记录点 （ checkpoints ）** 。如果训练非常耗时，那就在不同的训练时间对网络留下记录点（比如每个周期结束），然后用它们来进行模型集成。很显然，这样做多样性不足，但是在实践中效果还是不错的，这种方法的优势是代价比较小。
- **在训练的时候跑参数的平均值**。和上面一点相关的，还有一个也能得到1-2个百分点的提升的小代价方法，这个方法就是在训练过程中，如果损失值相较于前一次权重出现指数下降时，就在内存中对网络的权重进行一个备份。这样你就对前几次循环中的网络状态进行了平均。你会发现这个「平滑」过的版本的权重总是能得到更少的误差。直观的理解就是目标函数是一个碗状的，你的网络在这个周围跳跃，所以对它们平均一下，就更可能跳到中心去。

## 要点总结

- 优化方式：SGD、动量更新、Nesterov动量、Adagrad、RMSProp、Adam等，一般无脑使用Adam。此外还有学习率退火和二阶方法。
- 正则化：L2比较常用，Dropout也是一个很好的正则方法。
- 数据较少时可以使用迁移学习。
- 模型集成。

## 常见深度学习框架介绍

### 深度学习硬件

**GPU**（Graphics Processing Unit）是图形处理单元（又称显卡），在物理尺寸上就比 **CPU**（Central Processing Unit）大得多，有自己的冷却系统。最初用于渲染计算机图形，尤其是游戏。在深度学习上选择 NVIDIA（英伟达）的显卡，如果使用AMD的显卡会遇到很多问题。**TPU**（Tensor Processing Units）是专用的深度学习硬件。

#### CPU / GPU / TPU

- **CPU**一般有多个核心，每个核心速度都很快都可以独立工作，可同时进行多个进程，内存与系统共享，完成序列任务时很有用。图上CPU的运行速度是每秒约 540 GFLOPs 浮点数运算，使用 32位浮点数（注：一个 GFLOPS（gigaFLOPS）等于每秒十亿（$=10^9$）次的浮点运算）。
- **GPU**有上千个核心数，但每个核心运行速度很慢，也不能独立工作，适合大量的并行完成类似的工作。GPU一般自带内存，也有自己的缓存系统。图上GPU的运行速度是CPU的20多倍。
- **TPU**是专门的深度学习硬件，运行速度非常快。TITANV 在技术上并不是一个「TPU」，因为这是一个谷歌术语，但两者都有专门用于深度学习的硬件。运行速度非常快。

若是将这些运行速度除以对应的价格，可得到每美元对应运行速度的对比图。

#### GPU的优势与应用

GPU 在大矩阵的乘法运算中有很明显的优势。

由于结果中的每一个元素都是相乘的两个矩阵的每一行和每一列的点积，所以并行的同时进行这些点积运算速度会非常快。卷积神经网络也类似，卷积核和图片的每个区域进行点积也是并行运算。

CPU 虽然也有多个核心，但是在大矩阵运算时只能串行运算，速度很慢。

可以写出在 GPU 上直接运行的代码，方法是使用NVIDIA自带的抽象代码 CUDA ，可以写出类似 C 的代码，并可以在 GPU 直接运行。

但是直接写 CUDA 代码是一件非常困难的事，好在可以直接使用 NVIDIA 已经高度优化并且开源的API，比如 cuBLAS 包含很多矩阵运算， cuDNN 包含 CNN 前向传播、反向传播、批量归一化等操作；还有一种语言是 OpenCL，可以在 CPU、AMD 上通用，但是没人做优化，速度很慢；HIP可以将CUDA 代码自动转换成可以在 AMD 上运行的语言。以后可能会有跨平台的标准，但是现在来看 CUDA 是最好的选择。

在实际应用中，同样的计算任务，GPU 比 CPU 要快得多，当然 CPU 还能进一步优化。使用 cuDNN 也比不使用要快接近三倍。

实际应用 GPU 还有一个问题是训练的模型一般存放在 GPU，而用于训练的数据存放在硬盘里，由于 GPU 运行快，而机械硬盘读取慢，就会拖累整个模型的训练速度。有多种解决方法：

- 如果训练数据数量较小，可以把所有数据放到 GPU 的 RAM 中；
- 用固态硬盘代替机械硬盘；
- 使用多个 CPU 线程预读取数据，放到缓存供 GPU 使用。

### 深度学习软件

#### DL软件概述

现在有很多种深度学习框架，目前最流行的是 TensorFlow。

第一代框架大多由学术界编写的，比如 Caffe 就是伯克利大学开发的。

第二代往往由工业界主导，比如 Caffe2 是由 Facebook 开发。这里主要讲解 PyTorch 和 TensorFlow。

回顾之前计算图的概念，一个线性分类器可以用计算图表示，网络越复杂，计算图也越复杂。之所以使用这些深度学习框架有三个原因：

- 构建大的计算图很容易，可以快速的开发和测试新想法；
- 这些框架都可以自动计算梯度只需写出前向传播的代码；
- 可以在 GPU 上高效的运行，已经扩展了 cuDNN 等包以及处理好数据如何在 CPU 和 GPU 中流动。

这样我们就不用从头开始完成这些工作了。

我们以前的做法是使用 Numpy 写出前向传播，然后计算梯度，代码如下：

```python
import numpy as np
np.random.seed(0)  # 保证每次的随机数一致

N, D = 3, 4

x = np.random.randn(N, D)
y = np.random.randn(N, D)
z = np.random.randn(N, D)

a = x * y
b = a + z
c = np.sum(b)

grad_c = 1.0
grad_b = grad_c * np.ones((N, D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a * y
grad_y = grad_a * x
```

这种做法 API 干净，易于编写代码，但问题是没办法在 GPU 上运行，并且需要自己计算梯度。所以现在大部分深度学习框架的主要目标是自己写好前向传播代码，类似 Numpy，但能在 GPU 上运行且可以自动计算梯度。

TensorFlow 版本，前向传播构建计算图，梯度可以自动计算：

```python
import numpy as np
np.random.seed(0)
import tensorflow as tf

N, D = 3, 4

# 创建前向计算图
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = tf.placeholder(tf.float32)

a = x * y
b = a + z
c = tf.reduce_sum(b)

# 计算梯度
grad_x, grad_y, grad_z = tf.gradients(c, [x, y, z])

with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
        z: np.random.randn(N, D),
    }
    out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out
    print(c_val)
    print(grad_x_val)
```

PyTorch版本，前向传播与Numpy非常类似，但反向传播可以自动计算梯度，不用再去实现。

```python
import torch

device = 'cuda:0'  # 在GPU上运行，即构建GPU版本的矩阵

# 前向传播与Numpy类似
N, D = 3, 4
x = torch.randn(N, D, requires_grad=True, device=device)
# requires_grad要求自动计算梯度，默认为True
y = torch.randn(N, D, device=device)
z = torch.randn(N, D, device=device)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()  # 反向传播可以自动计算梯度
print(x.grad)
print(y.grad)
print(z.grad)
```

可见这些框架都能自动计算梯度并且可以自动在 GPU 上运行。

#### TensoFlow

下面以一个两层的神经网络为例，非线性函数使用 ReLU 函数、损失函数使用 L2 范式（当然仅仅是一个学习示例）。

实现代码如下：

**神经网络**

```python
import numpy as np
import tensorflow as tf

N, D , H = 64, 1000, 100

# 创建前向计算图
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.placeholder(tf.float32, shape=(D, H))
w2 = tf.placeholder(tf.float32, shape=(H, D))

h = tf.maximum(tf.matmul(x, w1), 0)  # 隐藏层使用折叶函数
y_pred = tf.matmul(h, w2)
diff = y_pred - y  # 差值矩阵
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))  # 损失函数使用L2范数

# 计算梯度
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# 多次运行计算图
with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
        w1: np.random.randn(D, H),
        w2: np.random.randn(H, D),
    }
    out = sess.run([loss, grad_w1, grad_w2], feed_dict=values)
    loss_val, grad_w1_val, grad_w2_val = out
```

整个过程可以分成两部分，`with` 之前部分定义计算图，`with` 部分多次运行计算图。这种模式在TensorFlow 中很常见。

- 首先，我们创建了`x,y,w1,w2`四个 `tf.placeholder` 对象，这四个变量作为「输入槽」，下面再输入数据。
- 然后使用这四个变量创建计算图，使用矩阵乘法 `tf.matmul` 和折叶函数 `tf.maximum` 计算 `y_pred` ，使用 L2 距离计算损失。但是目前并没有实际的计算，因为只是构建了计算图并没有输入任何数据。
- 然后通过一行神奇的代码计算损失值关于 `w1` 和 `w2` 的梯度。此时仍然没有实际的运算，只是构建计算图，找到 loss 关于 `w1` 和 `w2` 的路径，在原先的计算图上增加额外的关于梯度的计算。
- 完成计算图后，创建一个会话 Session 来运行计算图和输入数据。进入到 Session 后，需要提供 Numpy 数组给上面创建的「输入槽」。
- 最后两行代码才是真正的运行，执行 `sess.run` 需要提供 Numpy 数组字典feed_dict`和需要输出的计算值 loss ,`grad_w1`,`grad_w2` ，最后通过解包获取 Numpy 数组。

上面的代码只是运行了一次，我们需要迭代多次，并设置超参数、参数更新方式等：

```python
with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
        w1: np.random.randn(D, H),
        w2: np.random.randn(H, D),
    }
    learning_rate = 1e-5
    for t in range(50):
        out = sess.run([loss, grad_w1, grad_w2], feed_dict=values)
        loss_val, grad_w1_val, grad_w2_val = out
        values[w1] -= learning_rate * grad_w1_val
        values[w2] -= learning_rate * grad_w2_val
```

这种迭代方式有一个问题是每一步需要将Numpy和数组提供给GPU，GPU计算完成后再解包成Numpy数组，但由于CPU与GPU之间的传输瓶颈，非常不方便。

解决方法是将 `w1` 和 `w2` 作为变量而不再是「输入槽」，变量可以一直存在于计算图上。

由于现在 `w1` 和 `w2` 变成了变量，所以就不能从外部输入 Numpy 数组来初始化，需要由 TensorFlow 来初始化，需要指明初始化方式。此时仍然没有具体的计算。

```python
w1 = tf.Variable(tf.random_normal((D, H)))
w2 = tf.Variable(tf.random_normal((H, D)))
```

现在需要将参数更新操作也添加到计算图中，使用赋值操作 `assign` 更新 `w1` 和 `w2`，并保存在计算图中（位于计算梯度后面）：

```python
learning_rate = 1e-5
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)
```

现在运行这个网络，需要先运行一步参数的初始化 `tf.global_variables_initializer()`，然后运行多次代码计算损失值：

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
    }
    for t in range(50):
        loss_val, = sess.run([loss], feed_dict=values)
```

**优化器**

上面的代码，实际训练过程中损失值不会变。

原因是我们执行的 `sess.run([loss], feed_dict=values)` 语句只会计算 `loss`，TensorFlow 非常高效，与损失值无关的计算一律不会进行，所以参数就无法更新。

一个解决办法是在执行 `run` 时加入计算两个参数，这样就会强制执行参数更新，但是又会产生CPU 与 GPU 的通信问题。

一个技巧是在计算图中加入两个参数的依赖，在执行时需要计算这个依赖，这样就会让参数更新。这个技巧是 `group` 操作，执行完参数赋值操作后，执行 `updates = tf.group(new_w1, new_w2)`，这个操作会在计算图上创建一个节点；然后执行的代码修改为 `loss_val, _ = sess.run([loss, updates], feed_dict=values)`，在实际运算时，`updates` 返回值为空。

这种方式仍然不够方便，好在 TensorFlow 提供了更便捷的操作，使用自带的优化器。优化器需要提供学习率参数，然后进行参数更新。有很多优化器可供选择，比如梯度下降、Adam等。

```python
optimizer = tf.train.GradientDescentOptimizer(1e-5)  # 使用优化器
updates = optimizer.minimize(loss)  # 更新方式是使loss下降，内部其实使用了group
```

执行的代码也是：`loss_val, _ = sess.run([loss, updates], feed_dict=values)`

**损失**

计算损失的代码也可以使用 TensorFlow 自带的函数：

```python
loss = tf.losses.mean_squared_error(y_pred, y)  # 损失函数使用L2范数
```

**层**

目前仍有一个很大的问题是 `x,y,w1,w2` 的形状需要我们自己去定义，还要保证它们能正确连接在一起，此外还有偏差。如果使用卷积层、批量归一化等层后，这些定义会更加麻烦。

TensorFlow可以解决这些麻烦：

```python
N, D , H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

init = tf.variance_scaling_initializer(2.0)  # 权重初始化使用He初始化
h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init)
# 隐藏层使用折叶函数
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)

loss = tf.losses.mean_squared_error(y_pred, y)  # 损失函数使用L2范数

optimizer = tf.train.GradientDescentOptimizer(1e-5)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
    }
    for t in range(50):
        loss_val, _ = sess.run([loss, updates], feed_dict=values)
```

上面的代码，`x,y` 的初始化没有变化，但是参数 `w1,w2` 隐藏起来了，初始化使用 He初始化。

前向传播的计算使用了全连接层 `tf.layers.dense`，该函数需要提供输入数据 `inputs`、该层的神经元数目 `units`、激活函数 `activation`、卷积核（权重）初始化方式 `kernel_initializer` 等参数，可以自动设置权重和偏差。

**High level API：tensorflow.keras**

Keras 是基于 TensorFlow 的更高层次的封装，会让整个过程变得简单，曾经是第三方库，现在已经被内置到了 TensorFlow。

使用 Keras 的部分代码如下，其他与上文一致：

```python
N, D , H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

model = tf.keras.Sequential()  # 使用一系列层的组合方式
# 添加一系列的层
model.add(tf.keras.layers.Dense(units=H, input_shape=(D,), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(D))
# 调用模型获取结果
y_pred = model(x)
loss = tf.losses.mean_squared_error(y_pred, y)
```

这种模型已经简化了很多工作，最终版本代码如下：

```python
import numpy as np
import tensorflow as tf

N, D , H = 64, 1000, 100

# 创建模型，添加层
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=H, input_shape=(D,), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(D))

# 配置模型：损失函数、参数更新方式
model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-5), loss=tf.keras.losses.mean_squared_error)

x = np.random.randn(N, D)
y = np.random.randn(N, D)

# 训练
history = model.fit(x, y, epochs=50, batch_size=N)
```

代码非常简洁：

- **定义模型**：`tf.keras.Sequential()` 表明模型是一系列的层，然后添加两个全连接层，并设置激活函数、每层的神经元数目等；
- **配置模型**：用 `model.compile` 方法配置模型的优化器、损失函数等；
- **基于数据训练模型**：使用 `model.fit`，需要设置迭代周期次数、批量数等，可以直接用原始数据训练模型。

**其他知识**

**常见的拓展包**

- Keras (https://keras.io/)
- TensorFlow内置：
    - tf.keras (https://www.tensorflow.org/api_docs/python/tf/keras)
    - tf.layers (https://www.tensorflow.org/api_docs/python/tf/layers)
    - tf.estimator (https://www.tensorflow.org/api_docs/python/tf/estimator)
    - tf.contrib.estimator (https://www.tensorflow.org/api_docs/python/tf/contrib/estimator)
    - tf.contrib.layers (https://www.tensorflow.org/api_docs/python/tf/contrib/layers)
    - tf.contrib.slim (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
    - tf.contrib.learn (https://www.tensorflow.org/api_docs/python/tf/contrib/learn) (弃用)
    - Sonnet (https://github.com/deepmind/sonnet) (by DeepMind)
- 第三方包：
    - TFLearn (http://tflearn.org/)
    - TensorLayer (http://tensorlayer.readthedocs.io/en/latest/) TensorFlow: High-Level

**预训练模型**

TensorFlow已经有一些预训练好的模型可以直接拿来用，利用迁移学习，微调参数。

- tf.keras: (https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- TF-Slim: (https://github.com/tensorflow/models/tree/master/slim/nets)

**Tensorboard**

- 增加日志记录损失值和状态
- 绘制图像

**分布式操作**

可以在多台机器上运行，谷歌比较擅长。

**TPU（Tensor Processing Units）**

TPU是专用的深度学习硬件，运行速度非常快。Google Cloud TPU 算力为180 TFLOPs ，NVIDIA Tesla V100算力为125 TFLOPs。

**Theano**

TensorFlow的前身，二者许多地方都很相似。

#### PyTorch

**基本概念**

- **Tensor**：与Numpy数组很相似，只是可以在GPU上运行；
- **Autograd**：使用Tensors构建计算图并自动计算梯度的包；
- **Module**：神经网络的层，可以存储状态和可学习的权重。

下面的代码使用的是v0.4版本。

**Tensors**

下面使用Tensors训练一个两层的神经网络，激活函数使用ReLU、损失使用L2损失。

代码如下：

```python
import torch

# cpu版本
device = torch.device('cpu')
#device = torch.device('cuda:0')  # 使用gpu

# 为数据和参数创建随机的Tensors
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)
w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)

learning_rate = 1e-6
for t in range(500):
    # 前向传播，计算预测值和损失
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()

    # 反向传播手动计算梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 梯度下降，参数更新
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

- 首先创建 `x,y,w1,w2`的随机 tensor，与 Numpy 数组的形式一致
- 然后前向传播计算损失值和预测值
- 然后手动计算梯度
- 最后更新参数

上述代码很简单，和 Numpy 版本的写法很接近。但是需要手动计算梯度。

**Autograd自动梯度计算**

PyTorch 可以自动计算梯度：

```python
import torch

# 创建随机tensors
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传播
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    # 反向传播
    loss.backward()
    # 参数更新
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
```

与上一版代码的主要区别是：

- 创建 `w1,w2` 时要求 `requires_grad=True`，这样会自动计算梯度，并创建计算图。`x1,x2` 不需要计算梯度。
- 前向传播与之前的类似，但现在不用保存节点，PyTorch 可以帮助我们跟踪计算图。
- 使用 `loss.backward()` 自动计算要求的梯度。
- 按步对权重进行更新，然后将梯度归零。 `Torch.no_grad` 的意思是「不要为这部分构建计算图」。以下划线结尾的 PyTorch 方法是就地修改 Tensor，不返回新的 Tensor。

TensorFlow 与 PyTorch 的区别是 TensorFlow 需要先显式的构造一个计算图，然后重复运行；PyTorch 每次做前向传播时都要构建一个新的图，使程序看起来更加简洁。

PyTorch 支持定义自己的自动计算梯度函数，需要编写 `forward`，`backward` 函数。与作业中很相似。可以直接用到计算图上，但是实际上自己定义的时候并不多。

**NN**

与 Keras 类似的高层次封装，会使整个代码变得简单。

```python
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 定义模型
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLu(),
                            torch.nn.Linear(H, D_out))

learning_rate = 1e-2
for t in range(500):
    # 前向传播
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    # 计算梯度
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    model.zero_grad()
```

- 定义模型是一系列的层组合，在模型中定义了层对象比如全连接层、折叶层等，里面包含可学习的权重；
- 前向传播将数据给模型就可以直接计算预测值，进而计算损失；`torch.nn.functional` 含有很多有用的函数，比如损失函数；
- 反向传播会计算模型中所有权重的梯度；
- 最后每一步都更新模型的参数。

**Optimizer**

PyTorch 同样有自己的优化器：

```python
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 定义模型
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLu(),
                            torch.nn.Linear(H, D_out))
# 定义优化器
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 迭代
for t in range(500):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()
    # 更新参数
    optimizer.step()
    optimizer.zero_grad()
```

- 使用不同规则的优化器，这里使用Adam;
- 计算完梯度后，使用优化器更新参数，再置零梯度。

**定义新的模块**

PyTorch 中一个模块就是一个神经网络层，输入和输出都是 tensors。模块中可以包含权重和其他模块，可以使用 Autograd 定义自己的模块。

比如可以把上面代码中的两层神经网络改成一个模块：

```python
import torch
# 定义上文的整个模块为单个模块
class TwoLayerNet(torch.nn.Module):
    # 初始化两个子模块，都是线性层
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    # 使用子模块定义前向传播，不需要定义反向传播，autograd会自动处理
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# 构建模型与训练和之前类似
model = TwoLayerNet(D_in, H, D_out)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for t in range(500):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

这种混合自定义模块非常常见，定义一个模块子类，然后作为作为整个模型的一部分添加到模块序列中。

比如用定义一个下面这样的模块，输入数据先经过两个并列的全连接层得到的结果相乘后经过 ReLU：

```python
class ParallelBlock(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(ParallelBlock, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)
        self.linear2 = torch.nn.Linear(D_in, D_out)
    def forward(self, x):
        h1 = self.linear1(x)
        h2 = self.linear2(x)
        return (h1 * h2).clamp(min=0)
```

然后在整个模型中应用：

```python
model = torch.nn.Sequential(ParallelBlock(D_in, H),
                            ParallelBlock(H, H),
                            torch.nn.Linear(H, D_out))
```

**DataLoader**

DataLoader 包装数据集并提供获取小批量数据，重新排列，多线程读取等，当需要加载自定义数据时，只需编写自己的数据集类：

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

loader = DataLoader(TensorDataset(x, y), batch_size=8)
model = TwoLayerNet(D_in, H, D_out)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(20):
    for x_batch, y_batch in loader:
        y_pred = model(x_batch)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

上面的代码仍然是两层神经完网络，使用了自定义的模块。这次使用了 DataLoader 来处理数据。最后更新的时候在小批量上更新，一个周期会迭代所有的小批量数据。一般的 PyTorch 模型基本都长成这个样子。

**预训练模型**

使用预训练模型非常简单：https://github.com/pytorch/vision

```python
import torch
import torchvision
alexnet = torchvision.models.alexnet(pretrained=True)
vgg16 = torchvision.models.vggl6(pretrained=-True)
resnet101 = torchvision.models.resnet101(pretrained=True)
```

**Visdom**

可视化的包，类似 TensorBoard，但是不能像 TensorBoard 一样可视化计算图。

**Torch**

PyTorch 的前身，不能使用 Python，没有 Autograd，但比较稳定，不推荐使用。

### 静态与动态图（Static vs Dynamic Graphs ）

TensorFlow使用的是**静态图**（Static Graphs）：

- 构建计算图描述计算，包括找到反向传播的路径；
- 每次迭代执行计算，都使用同一张计算图。

与静态图相对应的是PyTorch使用的**动态图**（Dynamic Graphs），构建计算图与计算同时进行：

- 创建tensor对象；
- 每一次迭代构建计算图数据结构、寻找参数梯度路径、执行计算；
- 每一次迭代抛出计算图，然后再重建。之后重复上一步。

#### 静态图的优势

使用静态图形，由于一张图需要反复运行很多次，这样框架就有机会在计算图上做优化。

- 比如下面的自己写的计算图可能经过多次运行后优化成右侧，提高运行效率。

静态图只需要构建一次计算图，所以一旦构建好了即使源代码使用 Python 写的，也可以部署在C++上，不用依赖源代码；而动态图每次迭代都要使用源代码，构件图和运行是交织在一起的。

#### 动态图的优势

动态图的代码比较简洁，很像 Python 操作。

在条件判断逻辑中，由于 PyTorch 可以动态构建图，所以可以使用正常的 Python 流操作；而TensorFlow 只能一次性构建一个计算图，所以需要考虑到所有情况，只能使用 TensorFlow 流操作，这里使用的是和条件有关的。

在循环结构中，也是如此。

- PyTorch 只需按照 Python 的逻辑去写，每次会更新计算图而不用管最终的序列有多长；
- TensorFlow 由于使用静态图必须把这个循环结构显示的作为节点添加到计算图中，所以需要用到 TensorFlow 的循环流 `tf.foldl`。并且大多数情况下，为了保证只构建一次循环图， TensorFlow 只能使用自己的控制流，比如循环流、条件流等，而不能使用 Python 语法，所以用起来需要学习 TensorFlow 特有的控制命令。

#### 动态图的应用

**循环网络（Recurrent Networks）**

例如图像描述，需要使用循环网络在一个不同长度序列上运行，我们要生成的用于描述图像的语句是一个序列，依赖于输入数据的序列，即动态的取决于输入句子的长短。

**递归网络（Recursive Networks）**

用于自然语言处理，递归训练整个语法解析树，所以不仅仅是层次结构，而是一种图或树结构，在每个不同的数据点都有不同的结构，使用TensorFlow很难实现。在 PyTorch 中可以使用 Python 控制流，很容易实现。

**Modular Networks**

一种用于询问图片上的内容的网络，问题不一样生成的动态图也就不一样。

#### TensorFlow与PyTorch的相互靠拢

TensorFlow 与 PyTorch 的界限越来越模糊，PyTorch 正在添加静态功能，而 TensorFlow 正在添加动态功能。

- TensorFlow Fold 可以把静态图的代码自动转化成静态图
- TensorFlow 1.7增加了Eager Execution，允许使用动态图

```python
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable eager _execution()

N, D = 3, 4
x = tfe.Variable(tf.random_normal((N, D)))
y = tfe.Variable(tf.random_normal((N, D)))
z = tfe.Variable(tf.random_normal((N, D)))

with tfe.GradientTape() as tape:
    a=x * 2
    b=a + z
    c = tf.reduce_sum(b)

grad_x, grad_y, grad_z = tape.gradient(c, [x, y, 2])
print(grad_x)
```

- 在程序开始时使用 `tf.enable_eager_execution` 模式：它是一个全局开关
- `tf.random_normal` 会产生具体的值，无需 placeholders / sessions，如果想要为它们计算梯度，要用tfe.Variable进行包装
- 在`GradientTape` 下操作将构建一个动态图，类似于 PyTorch
- 使用`tape` 计算梯度，类似 PyTorch 中的 `backward`。并且可以直接打印出来
- 静态的 PyTorch 有Caffe2、ONNX Support

