---
title: MIT线性代数笔记05-对称矩阵与分解
date: 2023-03-14
categories: 
- 上浙大
tags: 
- 数学
- 线代
desc: 本笔记涵盖第 26–30 讲：对称矩阵与正定性、复数矩阵与 FFT、正定矩阵与最小值、相似矩阵与若尔当形、奇异值分解。
---


本笔记涵盖第 26–30 讲：对称矩阵与正定性、复数矩阵与 FFT、正定矩阵与最小值、相似矩阵与若尔当形、奇异值分解。

> [!INFO]+ 与 04 的衔接：从一般矩阵到「最好」的分解
> 04 的对角化要求 $n$ 个线性无关特征向量，对称矩阵**一定**满足且特征向量可正交。$A=Q\Lambda Q^T$ 是「正交对角化」，数值稳定。**SVD** 则对任意 $m \times n$ 矩阵都成立，$A=U\Sigma V^T$，是应用（最小二乘、压缩、伪逆）的通用工具。正定矩阵对应「碗状」二次型，是最优化的基础。
> <br>

<br>

## 一、对称矩阵及正定性

### 对称矩阵的性质

对实对称矩阵 $A = A^T$：
- 特征值均为实数
- 特征向量可选取为两两正交
- 可正交对角化：$A = Q \Lambda Q^T$，其中 $Q$ 为正交矩阵

### 正定矩阵

**正定矩阵**：对称矩阵 $A$ 满足对任意非零 $\mathbf{x}$，$\mathbf{x}^T A \mathbf{x} > 0$。

判定方式（等价）：
1. **特征值**：$\lambda_i > 0$
2. **顺序主子式**：均 $> 0$
3. **主元**：消元所得主元均 $> 0$
4. **判据式**：$\mathbf{x}^T A \mathbf{x} > 0$ 对所有 $\mathbf{x} \neq \mathbf{0}$

**半正定**：$\mathbf{x}^T A \mathbf{x} \geq 0$；对应特征值 $\geq 0$，存在零特征值。

> [!INFO]+ 二次型
> $\mathbf{x}^T A \mathbf{x}$ 是二次型。正定对应二次型恒正，图像为「碗状」；非正定可能有鞍点。
> <br>

<br>

## 二、复数矩阵与快速傅里叶变换

### 复向量与复矩阵

复向量长度：$|z|^2 = \bar{z}^T z = z^H z$（$z^H$ 为共轭转置）。

**埃尔米特矩阵**：$A^H = A$，特征值为实数，特征向量正交。

**酉矩阵**：$Q^H Q = I$，即 $Q^{-1} = Q^H$。

### 傅立叶矩阵与 FFT

傅立叶矩阵 $F_n$ 以 $w = e^{i 2\pi/n}$ 为元素，满足 $F_n^H F_n = n I$。FFT 利用 $F_{2n}$ 与 $F_n$ 的关系，将运算量从 $O(n^2)$ 降至 $O(n \log n)$。

<br>

## 三、正定矩阵与最小值

### 正定与最小值

若 $A$ 正定，则 $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ 在 $\mathbf{x} = \mathbf{0}$ 处取得唯一最小值。对应微积分中：二阶导数矩阵正定 $\Rightarrow$ 临界点为极小值。

### 正定矩阵的性质

- $A$ 正定 $\Rightarrow$ $A^{-1}$ 正定
- $A, B$ 正定 $\Rightarrow$ $A + B$ 正定
- $A$ 列满秩 $\Rightarrow$ $A^T A$ 正定（$\mathbf{x}^T A^T A \mathbf{x} = |A \mathbf{x}|^2 \geq 0$，且 $A \mathbf{x} = \mathbf{0}$ 仅当 $\mathbf{x} = \mathbf{0}$）

<br>

## 四、相似矩阵与若尔当形

### 相似矩阵

$A$ 与 $B$ **相似**：存在可逆 $M$ 使 $B = M^{-1} A M$。

相似矩阵具有相同特征值（特征向量一般不同：若 $A \mathbf{x} = \lambda \mathbf{x}$，则 $B (M^{-1} \mathbf{x}) = \lambda (M^{-1} \mathbf{x})$）。

### 若尔当标准形

当 $A$ 无 $n$ 个线性无关特征向量时，不可对角化，但可相似于**若尔当标准形** $J$。

**若尔当块**（以 $\lambda$ 为重复特征值）：
$$
J_i = \begin{bmatrix} \lambda & 1 & & \\\\ & \lambda & \ddots & \\\\ & & \ddots & 1 \\\\ & & & \lambda \end{bmatrix}
$$

每个若尔当块对应一个特征向量；若尔当块的个数 = 线性无关特征向量的个数。

<br>

## 五、奇异值分解（SVD）

### 基本形式

对任意 $m \times n$ 矩阵 $A$：
$$
A = U \Sigma V^T
$$

- $U$：$m \times m$ 正交矩阵，列由 $A A^T$ 的单位特征向量组成
- $V$：$n \times n$ 正交矩阵，列由 $A^T A$ 的单位特征向量组成
- $\Sigma$：$m \times n$ 对角矩阵，对角元 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 为**奇异值**

### 几何意义

$A \mathbf{v}_i = \sigma_i \mathbf{u}_i$：行空间的正交基 $\mathbf{v}_i$ 经 $A$ 映射为列空间的正交基 $\mathbf{u}_i$，$\sigma_i$ 为伸缩因子。

### 计算方法

1. 求 $A^T A$ 的特征值 $\lambda_i$ 与单位特征向量 $\mathbf{v}_i$，$\sigma_i = \sqrt{\lambda_i}$
2. $\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$
3. 或求 $A A^T$ 的特征向量得 $U$

> [!INFO]+ $A^T A$ 与 $A A^T$ 的特征值
> 非零特征值相同；$A^T A$ 的特征值 = $\sigma_i^2$。
> <br>

> [!EXAMPLE]+ 例：秩 1 矩阵的 SVD
> $A = \begin{bmatrix} 4 & 3 \\\\ 8 & 6 \end{bmatrix}$ 秩为 1，行空间为 $(4,3)$ 方向，列空间为 $(4,8)$ 方向，可快速写出 $V$ 与 $U$ 的基，再通过 $A^T A$ 求 $\sigma$。
> <br>

### 应用

SVD 可用于最小二乘、降秩近似、图像压缩等。低秩近似：保留前 $k$ 个奇异值，用 $U_k \Sigma_k V_k^T$ 近似 $A$。

<br>
