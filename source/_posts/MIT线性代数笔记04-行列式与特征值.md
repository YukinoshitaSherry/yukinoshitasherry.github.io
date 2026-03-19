---
title: MIT线性代数笔记04-行列式与特征值
date: 2023-03-13
categories: 
- 上浙大
tags: 
- 数学
- 线代
desc: 本笔记涵盖第 18–25 讲：行列式、克拉默法则与体积、特征值与特征向量、对角化与 $A$ 的幂、微分方程与 $\exp(At)$、马尔可夫矩阵与傅立叶级数、复习二。
---

本笔记涵盖第 18–25 讲：行列式、克拉默法则与体积、特征值与特征向量、对角化与 $A$ 的幂、微分方程与 $\exp(At)$、马尔可夫矩阵与傅立叶级数、复习二。

> [!INFO]+ 与 03 的衔接：从「解方程」到「理解变换」
> 前面关注 $Ax=b$ 的解。这里转向**矩阵作为变换**：$A\mathbf{x} = \lambda\mathbf{x}$ 表示沿某方向只伸缩。行列式 $|A|$ 刻画体积伸缩倍数，$|A|=0$ 等价于不可逆。对角化 $A=S\Lambda S^{-1}$ 把 $A^k$、$e^{At}$ 转化为对角阵的幂/指数，是差分方程、微分方程、马尔可夫链的基础。
<br>

<br>

## 一、行列式

### 行列式的性质

$|A|$ 表示方阵 $A$ 的行列式。方阵可逆当且仅当 $|A| \neq 0$。

| 性质 | 内容 |
| :--- | :--- |
| 1 | $|I| = 1$ |
| 2 | 交换两行，行列式变号 |
| 3 | 某行乘 $t$，行列式乘 $t$；行列式对每一行线性 |
| 4 | 两行相等 $\Rightarrow$ $|A| = 0$ |
| 5 | 行 $k$ 减去行 $\ell$ 的 $i$ 倍，行列式不变（消元不改变行列式） |
| 6 | 有一行全为零 $\Rightarrow$ $|A| = 0$ |
| 7 | 上三角矩阵的行列式 = 主对角线元素之积 |
| 8 | $|A| \neq 0$ 当且仅当 $A$ 可逆 |
| 9 | $|AB| = |A| \cdot |B|$，故 $|A^{-1}| = 1/|A|$，$|kA| = k^n |A|$ |
| 10 | $|A^T| = |A|$ |

> [!INFO]+ 二阶行列式
> $\begin{vmatrix} a & b \\\\ c & d \end{vmatrix} = ad - bc$。
<br>

### 行列式公式与代数余子式

$n$ 阶行列式可展开为 $n!$ 项之和，每项为不同行不同列元素的乘积，符号由排列的奇偶性决定。

**代数余子式** $C_{ij}$：去掉第 $i$ 行第 $j$ 列后的 $(n-1)$ 阶行列式乘以 $(-1)^{i+j}$。

**按行展开**：$|A| = a_{i1} C_{i1} + a_{i2} C_{i2} + \cdots + a_{in} C_{in}$。

**逆矩阵公式**：$A^{-1} = \frac{1}{|A|} C^T$（$C$ 为代数余子式矩阵）。

<br>

## 二、克拉默法则、逆矩阵与体积

克拉默法则用行列式表示线性方程组的解；$n \times n$ 矩阵 $A$ 的行列式绝对值表示 $A$ 的列向量张成的平行多面体的**有向体积**。

<br>

## 三、特征值与特征向量

### 定义

若 $A \mathbf{x} = \lambda \mathbf{x}$（$\mathbf{x} \neq \mathbf{0}$），则 $\lambda$ 为**特征值**，$\mathbf{x}$ 为**特征向量**。

> [!INFO]+ 理解
> $A \mathbf{x}$ 与 $\mathbf{x}$ 平行，$\lambda$ 表示伸缩倍数。特征值 0 对应 $A \mathbf{x} = \mathbf{0}$，即 $\mathbf{x} \in N(A)$。
<br>

### 求法

由 $A \mathbf{x} = \lambda \mathbf{x}$ 得 $(A - \lambda I) \mathbf{x} = \mathbf{0}$，故 $A - \lambda I$ 不可逆，即：
$$
|A - \lambda I| = 0
$$

解该**特征方程**得 $\lambda$，再对每个 $\lambda$ 求 $N(A - \lambda I)$ 得特征向量。

> [!INFO]+ 迹与行列式
> $\lambda_1 + \cdots + \lambda_n = \operatorname{tr}(A)$，$\lambda_1 \cdots \lambda_n = |A|$。
<br>

> [!EXAMPLE]+ 例
> $A = \begin{bmatrix} 3 & 1 \\\\ 1 & 3 \end{bmatrix}$，$|A - \lambda I| = (3-\lambda)^2 - 1 = 0$，得 $\lambda = 2, 4$。对应特征向量 $\begin{bmatrix} 1 \\\\ -1 \end{bmatrix}$，$\begin{bmatrix} 1 \\\\ 1 \end{bmatrix}$。
<br>

### 特殊情况

- **旋转矩阵**（如 90°）：特征值为纯虚数 $i, -i$
- **上三角矩阵**：特征值 = 主对角线元素；重复特征值可能导致特征向量不足

<br>

## 四、对角化与 $A$ 的幂

### 对角化

> [!INFO]+ 为什么要对角化？
> $A^k$ 或 $e^{At}$ 直接算很繁。若 $A = S \Lambda S^{-1}$，则 $A^k = S \Lambda^k S^{-1}$，$\Lambda^k$ 只需对对角元求幂；$e^{At} = S e^{\Lambda t} S^{-1}$，$e^{\Lambda t}$ 为对角阵。对角化把问题简化为标量运算。
<br>

若 $A$ 有 $n$ 个线性无关的特征向量 $\mathbf{x}_1, \ldots, \mathbf{x}_n$，对应特征值 $\lambda_1, \ldots, \lambda_n$，令 $S = [\mathbf{x}_1 \cdots \mathbf{x}_n]$，$\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$，则：
$$
A = S \Lambda S^{-1}, \quad S^{-1} A S = \Lambda
$$

### 矩阵的幂

$$
A^k = S \Lambda^k S^{-1}
$$

$A^k \to 0$ 当且仅当所有 $|\lambda_i| < 1$。

### 差分方程

递推 $u_{k+1} = A u_k$ 的通解为 $u_k = A^k u_0$。将 $u_0$ 用特征向量展开：
$$
u_0 = c_1 \mathbf{x}_1 + \cdots + c_n \mathbf{x}_n \quad \Rightarrow \quad u_k = c_1 \lambda_1^k \mathbf{x}_1 + \cdots + c_n \lambda_n^k \mathbf{x}_n
$$

> [!EXAMPLE]+ 斐波那契数列
> $F_{k+2} = F_{k+1} + F_k$ 可化为 $u_{k+1} = A u_k$，其中 $A = \begin{bmatrix} 1 & 1 \\\\ 1 & 0 \end{bmatrix}$。特征值 $\lambda_{1,2} = \frac{1 \pm \sqrt{5}}{2}$，$F_k$ 的增长主要由 $\lambda_1 = \frac{1+\sqrt{5}}{2}$ 主导。
<br>

<br>

## 五、微分方程与 $\exp(At)$

### 一阶线性微分方程组

$\frac{d \mathbf{u}}{dt} = A \mathbf{u}$ 的通解：
$$
\mathbf{u}(t) = c_1 e^{\lambda_1 t} \mathbf{x}_1 + \cdots + c_n e^{\lambda_n t} \mathbf{x}_n
$$

- 所有 $\operatorname{Re}(\lambda_i) < 0$ $\Rightarrow$ $\mathbf{u}(t) \to \mathbf{0}$
- 有 $\lambda_i = 0$ 且其余 $\operatorname{Re}(\lambda_j) < 0$ $\Rightarrow$ 存在稳态
- 有 $\operatorname{Re}(\lambda_i) > 0$ $\Rightarrow$ 解发散

### 解耦与矩阵指数

令 $\mathbf{u} = S \mathbf{v}$，则 $\frac{d \mathbf{v}}{dt} = \Lambda \mathbf{v}$，解耦为 $n$ 个标量方程。可得：
$$
\mathbf{u}(t) = e^{At} \mathbf{u}(0) = S e^{\Lambda t} S^{-1} \mathbf{u}(0)
$$

其中 $e^{At} = I + At + \frac{(At)^2}{2!} + \cdots$，$e^{\Lambda t} = \operatorname{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t})$。

### 高阶微分方程

二阶方程 $y'' + by' + ky = 0$ 可化为 $\mathbf{u}' = B \mathbf{u}$，其中 $\mathbf{u} = [y', y]^T$。

<br>

## 六、马尔可夫矩阵与傅立叶级数

**马尔可夫矩阵**：各列元素非负且列和为 1。必有特征值 1，对应稳态分布。

傅立叶级数将函数表示为正交基（$\sin$, $\cos$）的线性组合，与正交投影、最小二乘思想一致。

<br>
