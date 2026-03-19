---
title: MIT线性代数笔记03-正交与投影
date: 2023-03-12
categories: 
- 上浙大
tags: 
- 数学
- 线代
desc: 本笔记涵盖第 11–17 讲：矩阵空间与秩 1 矩阵、图与网络、正交向量与子空间、子空间投影、投影矩阵与最小二乘、Gram-Schmidt 正交化。
---

本笔记涵盖第 11–17 讲：矩阵空间与秩 1 矩阵、图与网络、正交向量与子空间、子空间投影、投影矩阵与最小二乘、Gram-Schmidt 正交化。

> [!INFO]+ 与 02 的衔接：$Ax=b$ 无解时怎么办？
> 02 告诉我们：$b \notin C(A)$ 时无解。但实际问题（如拟合直线）往往如此。思路：**找 $C(A)$ 中离 $b$ 最近的点**，即把 $b$ 投影到列空间。这就要求正交、投影矩阵、$A^T A \hat{x} = A^T b$。Gram-Schmidt 构造标准正交基，使投影计算变简单，并为 05 的 QR、SVD 打基础。
<br>

<br>

## 一、矩阵空间、秩 1 矩阵

### 矩阵空间

将所有 $3 \times 3$ 矩阵看作线性空间的「向量」，在加法与数乘下封闭，构成矩阵空间 $M$。

- **对称矩阵** $S$、**上三角矩阵** $U$ 均为 $M$ 的子空间
- $S \cap U$ = 对角矩阵 $D$
- $\dim M = 9$，$\dim S = \dim U = 6$，$\dim D = 3$

> [!INFO]+ 维数公式
> $\dim(S) + \dim(U) = \dim(S \cap U) + \dim(S + U)$，即 $6 + 6 = 3 + 9$。
<br>

### 解空间

微分方程 $\frac{d^2 y}{dx^2} + y = 0$ 的解构成线性空间，基为 $\cos x$ 与 $\sin x$，维数为 2。

### 秩 1 矩阵

**秩 1 矩阵**可写成「列 $\times$ 行」：
$$
A = \mathbf{u} \mathbf{v}^T
$$

> [!EXAMPLE]+ 例
> $A = \begin{bmatrix} 1 & 4 & 5 \\\\ 2 & 8 & 10 \end{bmatrix} = \begin{bmatrix} 1 \\\\ 2 \end{bmatrix} \begin{bmatrix} 1 & 4 & 5 \end{bmatrix}$
<br>

秩为 $r$ 的矩阵可表示为 $r$ 个秩 1 矩阵之和。但**所有秩为 $r$ 的矩阵的集合不是子空间**（对加法不封闭：$R(A+B) \leq R(A) + R(B)$）。

<br>

## 二、图与网络

### 关联矩阵

**关联矩阵** $A$：有向图的边–结点关联。每行对应一条边，每列对应一个结点；边从结点 $i$ 出发为 $-1$，指向结点 $j$ 为 $1$。

### $Ax$ 与电势差

设 $x$ 为各结点电势，则 $Ax$ 表示各边上的**电势差**。$Ax = \mathbf{0}$ 表示各边无电势差，即各点电势相等。

### $A^T y$ 与基尔霍夫定律

设 $y$ 为各边电流，则 $A^T y = \mathbf{0}$ 表示**基尔霍夫电流定律**：每个结点流入、流出电流之和为零。

### 欧姆定律与 $A^T C A x = f$

设 $y = C A x$（$C$ 为 conductance 矩阵），结合 $A^T y = f$（$f$ 为外加电流源）得：
$$
A^T C A x = f
$$

<br>

## 三、正交向量与子空间

### 正交

**正交**即垂直。向量 $\mathbf{x}, \mathbf{y}$ 正交当且仅当 $\mathbf{x}^T \mathbf{y} = 0$。

> [!INFO]+ 勾股定理与正交
> $|\mathbf{x}|^2 + |\mathbf{y}|^2 = |\mathbf{x} + \mathbf{y}|^2$ $\Leftrightarrow$ $\mathbf{x}^T \mathbf{y} = 0$。
<br>

**子空间正交**：一个空间中任意向量与另一空间中任意向量正交。

### 四个子空间的正交关系

- $N(A) \perp C(A^T)$（零空间与行空间正交）
- $N(A^T) \perp C(A)$（左零空间与列空间正交）

行空间与零空间将 $\mathbb{R}^n$ 分为两个正交子空间，称为**正交补**。

### 无解方程的最优解

当 $Ax = b$ 无解时，可求解 $A^T A \hat{x} = A^T b$ 得到**最小二乘解** $\hat{x}$。

> [!INFO]+ 为什么是 $A^T A$？
> 投影条件：误差 $\mathbf{e} = \mathbf{b} - A\hat{x}$ 与 $C(A)$ 正交，即与 $A$ 的每一列正交，故 $A^T \mathbf{e} = \mathbf{0}$，即 $A^T(\mathbf{b} - A\hat{x}) = \mathbf{0}$，整理得 $A^T A \hat{x} = A^T \mathbf{b}$。$A^T A$ 为方阵、对称，且 $A$ 列无关时可逆。
<br>

> [!INFO]+ $A^T A$ 的性质
> - $A^T A$ 为方阵且对称
> - $N(A^T A) = N(A)$，$\operatorname{rank}(A^T A) = \operatorname{rank}(A)$
> - $A$ 列线性无关 $\Rightarrow$ $A^T A$ 可逆
<br>

<br>

## 四、子空间投影

### 向量投影到向量

将 $\mathbf{b}$ 投影到 $\mathbf{a}$ 上，投影向量
$$
\mathbf{p} = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}} \mathbf{a} = P \mathbf{b}, \quad P = \frac{\mathbf{a} \mathbf{a}^T}{\mathbf{a}^T \mathbf{a}}
$$

$P$ 为**投影矩阵**，满足 $P^T = P$，$P^2 = P$。

### 向量投影到子空间

将 $\mathbf{b}$ 投影到 $C(A)$（$A$ 的列张成的子空间）。设投影为 $\mathbf{p} = A \hat{x}$，由误差 $\mathbf{e} = \mathbf{b} - A \hat{x}$ 与 $C(A)$ 正交得：
$$
A^T (\mathbf{b} - A \hat{x}) = \mathbf{0} \quad \Rightarrow \quad A^T A \hat{x} = A^T \mathbf{b}
$$

因此
$$
\hat{x} = (A^T A)^{-1} A^T \mathbf{b}, \quad \mathbf{p} = A (A^T A)^{-1} A^T \mathbf{b}
$$

> [!INFO]+ 投影矩阵
> $P = A (A^T A)^{-1} A^T$ 将 $\mathbf{b}$ 投影到 $C(A)$，同样满足 $P^T = P$，$P^2 = P$。
<br>

### 最小二乘初涉

拟合直线 $y = C + Dt$ 时，将数据点写成 $Ax = b$。若无解，用 $A^T A \hat{x} = A^T b$ 求最优 $\hat{x}$，相当于把 $\mathbf{b}$ 投影到 $C(A)$。

<br>

## 五、投影矩阵与最小二乘

### 投影矩阵的性质

- 若 $\mathbf{b} \in C(A)$，则 $P \mathbf{b} = \mathbf{b}$
- 若 $\mathbf{b} \perp C(A)$（即 $\mathbf{b} \in N(A^T)$），则 $P \mathbf{b} = \mathbf{0}$

$\mathbf{b} = \mathbf{p} + \mathbf{e}$，其中 $\mathbf{p} \in C(A)$，$\mathbf{e} \in N(A^T)$。$(I - P)$ 是将 $\mathbf{b}$ 投影到左零空间的投影矩阵。

### 最小二乘拟合

> [!EXAMPLE]+ 例：三点拟合直线
> 点 $(1,1), (2,2), (3,2)$ 拟合 $y = C + Dx$
>
> $A = \begin{bmatrix} 1 & 1 \\\\ 1 & 2 \\\\ 1 & 3 \end{bmatrix}$，$\mathbf{b} = \begin{bmatrix} 1 \\\\ 2 \\\\ 2 \end{bmatrix}$
>
> 解 $A^T A \hat{x} = A^T \mathbf{b}$ 得 $\hat{C} = 2/3$，$\hat{D} = 1/2$，拟合直线 $y = 2/3 + x/2$。
<br>

### $A^T A$ 可逆的证明

若 $A$ 列线性无关，则 $A^T A$ 可逆。证明：设 $A^T A x = 0$，左乘 $x^T$ 得 $x^T A^T A x = |A x|^2 = 0$，故 $A x = 0$，由列无关得 $x = 0$，即 $N(A^T A) = \{0\}$。

### 标准正交基

**标准正交向量组**：$q_i^T q_j = \delta_{ij}$（正交且单位长度）。选用标准正交基时，投影公式简化为 $\hat{x} = Q^T \mathbf{b}$。

<br>

## 六、正交矩阵与 Gram-Schmidt 正交化

### 正交矩阵 $Q$

**正交矩阵**：方阵 $Q$ 满足 $Q^T Q = I$，即 $Q^{-1} = Q^T$。

当 $A = Q$（标准正交列）时，投影矩阵简化为 $P = Q Q^T$。

### Gram-Schmidt 正交化

从线性无关向量组 $\mathbf{a}, \mathbf{b}, \mathbf{c}, \ldots$ 构造标准正交基：

1. $A = \mathbf{a}$
2. $B = \mathbf{b} - \frac{A^T \mathbf{b}}{A^T A} A$（$\mathbf{b}$ 减去在 $A$ 上的投影）
3. $C = \mathbf{c} - \frac{A^T \mathbf{c}}{A^T A} A - \frac{B^T \mathbf{c}}{B^T B} B$
4. 单位化：$q_i = \frac{\text{第 } i \text{ 个正交向量}}{|\cdot|}$

> [!EXAMPLE]+ 例：Gram-Schmidt
> $\mathbf{a} = \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \end{bmatrix}$，$\mathbf{b} = \begin{bmatrix} 1 \\\\ 0 \\\\ 2 \end{bmatrix}$
>
> $A = \mathbf{a}$，$B = \mathbf{b} - \frac{3}{3} A = \begin{bmatrix} 0 \\\\ -1 \\\\ 1 \end{bmatrix}$
>
> 单位化得 $q_1 = \frac{1}{\sqrt{3}}\begin{bmatrix} 1 \\\\ 1 \\\\ 1 \end{bmatrix}$，$q_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 0 \\\\ -1 \\\\ 1 \end{bmatrix}$。
<br>

### $A = QR$ 分解

Gram-Schmidt 等价于 $A = QR$，其中 $Q$ 列标准正交，$R$ 为上三角矩阵。

<br>
