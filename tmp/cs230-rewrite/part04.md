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

### 损失与评估

- 损失：逐像素交叉熵；类别不平衡时用加权或 Dice。
- 医学分割中正类像素很少，交叉熵会偏向背景。Dice 损失直接优化重叠：

$$
1-\frac{2\sum_p \hat{y}_p y_p}{\sum_p \hat{y}_p+\sum_p y_p+\varepsilon}.
$$

- 评估不能把 IoU 与像素准确率混用：后者会被大背景类主导。
- 上采样：转置卷积，或插值加 $3\times3$ 卷积。
- 作业重点：对称裁剪或 padding，使 skip 两端空间尺寸一致，否则无法 `concatenate`。

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

