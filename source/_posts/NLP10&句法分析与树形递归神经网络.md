---
title: NLP(10)：句法分析与树形递归神经网络
date: 2024-02-10
categories:
  - 学AI/DS
tags:
  - NLP
desc: CS224n Lec18 笔记，资料整合与一些自己的思考。

---

- 参考
    - <a href="https://www.showmeai.tech/tutorials/36">`showmeai-斯坦福CS224n教程`</a>

## 递归神经网络
**Recursive Neural Networks**

### 概述

递归神经网络是一种专门用于处理具有嵌套层次结构和内在递归结构数据的神经网络模型。在自然语言处理中，它特别适合处理句子的语法结构，因为语言的语法规则本身就是高度递归的。

### 基本特点

1. **结构特点**
   - 能够处理任意长度的句子
   - 可以捕捉句子的层次结构
   - 支持递归组合操作

2. **应用优势**
   - 不需要存储无限数量的向量
   - 可以处理未见过的词组合
   - 能够从词向量推导出短语和句子的表示

### 基本结构

标准递归神经网络通过以下步骤处理句子：

1. **输入处理**
   - 获取句子解析树
   - 获取句子中单词的词向量

2. **递归计算**
   - 从叶子节点开始
   - 逐层向上组合
   - 最终得到整个句子的表示

### 数学表示

对于两个相邻节点，其组合公式为：

$$h = f(W \cdot [h_{left}; h_{right}] + b)$$

其中：
- $h_{left}$ 和 $h_{right}$ 是子节点的表示向量
- $W$ 是权重矩阵
- $b$ 是偏置向量
- $f$ 是激活函数

## 语言模型

假设我们的任务是取一个句子，并将其表示为与单词本身语义空间相同的向量。所以像 I went to the mall yesterday 、 We went shopping last week 和 They went to the store 这样的短语，彼此之间的距离都非常近。
我们已经看到了训练单字向量的方法，是否应该对二元组、三元组等也做同样的事情。
但这种想法存在两个主要问题:
1. 单词的组合有无数种可能。存储和训练无限数量的向量将是荒谬的。
2. 有些单词的组合虽然在语言中完全可以听到，但可能永远不会出现在我们的 训练 / 开发 语料库中，所以永远学不会。

我们需要一种方法来取一个句子及其相应的词向量，并推导出嵌入向量应该是什么。
- 语义分析是对句子意义的理解，能够在结构语义空间中把短语表示为一个向量，相似的句子非常近，不相关的句子非常远。
- 语法理解是指已经确定了句子的基本语法结构，句子的哪一部分取决于哪一部分，哪些词在修饰哪些词。输出解析树。

<br>

### 标准递归神经网络（Standard RNN）

#### 原理
标准递归神经网络是最基础的递归神经网络模型，它使用相同的权重矩阵来处理所有类型的组合。其核心思想是：
- 使用统一的组合函数处理所有节点
- 通过递归方式构建句子的表示
- 保持词向量空间的一致性

#### 实现方式
1. **组合函数**
   ```python
   def combine_vectors(left_vec, right_vec, W, b):
       # 连接左右向量
       combined = np.concatenate([left_vec, right_vec])
       # 线性变换
       transformed = np.dot(W, combined) + b
       # 非线性激活
       return np.tanh(transformed)
   ```

2. **递归计算**
   ```python
   def recursive_compute(node):
       if node.is_leaf():
           return node.word_vector
       left_vec = recursive_compute(node.left)
       right_vec = recursive_compute(node.right)
       return combine_vectors(left_vec, right_vec, W, b)
   ```

#### 优缺点
- **优点**：
  - 结构简单，易于实现
  - 计算效率高
  - 可以处理任意长度的句子
- **缺点**：
  - 表达能力有限
  - 难以处理复杂的语法结构
  - 对否定词和修饰词的处理效果不佳

### 语法解耦递归神经网络（SU-RNN）

SU-RNN通过为不同语法类别的输入使用不同的权重矩阵，提高了模型的表达能力：

- **主要改进**
  - 根据语法类别使用不同的权重矩阵
  - 通过PCFG确定语法类别
  - 分别初始化不同类别的权重


#### 实现方式
1. **语法类别判断**
   ```python
   def get_syntax_category(node):
       # 使用PCFG确定语法类别
       if node.is_noun_phrase():
           return "NP"
       elif node.is_verb_phrase():
           return "VP"
       # ... 其他类别
   ```

2. **类别特定的组合**
   ```python
   def combine_by_category(left_vec, right_vec, category):
       # 获取对应类别的权重矩阵
       W = get_weight_matrix(category)
       b = get_bias_vector(category)
       return combine_vectors(left_vec, right_vec, W, b)
   ```

#### 优缺点
- **优点**：
  - 更好地捕捉语法结构
  - 提高了模型的表达能力
  - 能够学习到语言学上有意义的特征
- **缺点**：
  - 需要额外的语法分析
  - 参数数量增加
  - 训练复杂度提高

### 矩阵向量递归神经网络（MV-RNN）

MV-RNN通过引入单词矩阵，增强了模型对词间修饰关系的表达能力：

- **核心思想**
  - 每个词同时具有词向量和词矩阵
  - 词矩阵用于表示词的修饰作用
  - 支持词间的乘法交互


#### 实现方式
1. **词表示**
   ```python
   class WordRepresentation:
       def __init__(self, word):
           self.vector = get_word_vector(word)
           self.matrix = get_word_matrix(word)
   ```

2. **组合计算**
   ```python
   def combine_with_matrices(left_rep, right_rep):
       # 使用词矩阵进行修饰
       modified_left = np.dot(left_rep.matrix, left_rep.vector)
       modified_right = np.dot(right_rep.matrix, right_rep.vector)
       # 组合修饰后的向量
       return combine_vectors(modified_left, modified_right, W, b)
   ```

#### 优缺点
- **优点**：
  - 更好地处理修饰关系
  - 支持词间的乘法交互
  - 提高了模型的表达能力
- **缺点**：
  - 难以处理否定词
  - 对对比连词的处理效果不佳
  - 计算复杂度较高

### 递归神经张量网络（RNTN）

RNTN通过引入张量操作，进一步提高了模型的表达能力：

- **主要特点**
  - 使用三阶张量进行组合
  - 支持词向量间的乘法交互
  - 不需要维护单词矩阵


#### 实现方式
1. **张量操作**
   ```python
   def tensor_combine(left_vec, right_vec, V):
       # V是三阶张量
       combined = np.zeros(V.shape[0])
       for i in range(V.shape[0]):
           # 计算张量切片
           slice = V[i]
           # 二次型计算
           combined[i] = np.dot(left_vec, np.dot(slice, right_vec))
       return combined
   ```

2. **完整组合**
   ```python
   def rntn_combine(left_vec, right_vec, V, W, b):
       # 张量组合
       tensor_part = tensor_combine(left_vec, right_vec, V)
       # 线性组合
       linear_part = np.dot(W, np.concatenate([left_vec, right_vec])) + b
       # 合并结果
       return np.tanh(tensor_part + linear_part)
   ```

#### 优缺点
- **优点**：
  - 更好地处理否定词
  - 能够处理对比连词
  - 提高了模型的整体性能
- **缺点**：
  - 计算复杂度最高
  - 需要更多的训练数据
  - 训练时间较长

## 模型比较与选择

| 模型 | 语法结构 | 否定词 | 对比连词 | 计算效率 |
|------|----------|--------|----------|----------|
| Standard RNN | 一般 | 差 | 差 | 高 |
| SU-RNN | 好 | 一般 | 一般 | 中 |
| MV-RNN | 好 | 一般 | 差 | 中 |
| RNTN | 最好 | 好 | 好 | 低 |

选择建议：
1. **简单任务**：使用Standard RNN
2. **需要语法结构**：使用SU-RNN
3. **需要修饰关系**：使用MV-RNN
4. **复杂语义**：使用RNTN

实际应用：
1. **情感分析**：推荐使用RNTN
2. **句法分析**：推荐使用SU-RNN
3. **语义组合**：推荐使用MV-RNN
4. **实时处理**：推荐使用Standard RNN

<br>

# 句法分析（Syntactic Parsing）

## 概述

句法分析是自然语言处理中的基础任务，主要研究如何将句子分解成其组成部分，并确定这些部分之间的关系。句法分析的主要目标是构建句子的语法结构。

## 类型

1. **成分句法分析（Constituency Parsing）**
   - 将句子分解成短语结构
   - 使用上下文无关文法（CFG）描述句子结构
   - 输出为句法树（Parse Tree）

2. **依存句法分析（Dependency Parsing）**
   - 识别句子中词与词之间的依存关系
   - 每个词都依赖于句子中的另一个词（除了根节点）
   - 输出为依存树（Dependency Tree）

### 成分句法分析

#### 上下文无关文法（CFG）

CFG由以下四个部分组成：
- 终结符集合（Terminals）：词性标签
- 非终结符集合（Non-terminals）：短语类型
- 产生式规则（Production Rules）
- 起始符号（Start Symbol）

##### 产生式规则示例

```
S → NP VP
NP → Det N
VP → V NP
Det → the | a
N → cat | dog
V → chased | saw
```

##### 句法树示例

对于句子 "The cat chased the dog"，其句法树结构为：

```
        S
    /       \
   NP        VP
  /  \     /   \
Det   N   V    NP
 |    |   |   /  \
the  cat chased Det N
                |  |
               the dog
```

- S 代表句子，最高级的结构
- NP 代表名词短语，包括句子的主语和宾语
- VP 代表动词短语，充当谓语
- V 代表动词
- D 代表限定词，例如 the
- N 代表名词

##### 句法树表示

句法树是一种层次结构，其中：
- 叶子节点是词性标签
- 内部节点是短语类型
- 每个节点都有其子节点

### 依存句法分析

#### 依存关系

依存关系表示词与词之间的语法关系，主要包括：
- 主语关系（subject）
- 宾语关系（object）
- 修饰关系（modifier）
- 等等

##### 依存树

依存树是一种有向图，其中：
- 节点是句子中的词
- 边表示依存关系
- 每个词（除根节点外）都只有一个父节点

##### 依存树示例

对于句子 "The cat chased the dog"，其依存树结构为：

```
chased
  /  \
 /    \
cat   dog
 |     |
the   the
```

# 树形递归神经网络（Tree-RNN）

## 基本概念

树形递归神经网络是一种特殊的递归神经网络，专门用于处理树形结构数据。在NLP中，它主要用于处理句法树。

### 模型结构

树形RNN的基本结构包括：
- 组合函数（Composition Function）
- 递归计算过程
- 节点表示

## 数学表示

### 基本公式

对于节点 $n$，其表示向量 $h_n$ 的计算公式为：

$$h_n = f(W \cdot [h_{left}; h_{right}] + b)$$

其中：
- $h_{left}$ 和 $h_{right}$ 是子节点的表示向量
- $W$ 是权重矩阵
- $b$ 是偏置向量
- $f$ 是激活函数

### 组合函数

常用的组合函数包括：
1. **简单加法**：
   $$h_n = f(W \cdot (h_{left} + h_{right}) + b)$$

2. **矩阵乘法**：
   $$h_n = f(W \cdot [h_{left}; h_{right}] + b)$$

### 递归计算过程

1. **自底向上计算**：
   - 从叶子节点开始
   - 逐层向上计算
   - 最终得到根节点的表示

2. **数学表示**：
   对于树中的每个节点 $n$，其表示向量 $h_n$ 的计算过程为：

   $$h_n = \begin{cases}
   f(W \cdot [h_{left}; h_{right}] + b) & \text{if } n \text{ is internal node} \\
   f(W \cdot x_n + b) & \text{if } n \text{ is leaf node}
   \end{cases}$$

   其中 $x_n$ 是叶子节点的词向量。

### 具体例子

考虑句子 "The cat chased the dog" 的句法树：

1. **叶子节点表示**：
   - $h_{the} = f(W \cdot x_{the} + b)$
   - $h_{cat} = f(W \cdot x_{cat} + b)$
   - $h_{chased} = f(W \cdot x_{chased} + b)$
   - $h_{the} = f(W \cdot x_{the} + b)$
   - $h_{dog} = f(W \cdot x_{dog} + b)$

2. **内部节点计算**：
   - $h_{NP1} = f(W \cdot [h_{the}; h_{cat}] + b)$
   - $h_{NP2} = f(W \cdot [h_{the}; h_{dog}] + b)$
   - $h_{VP} = f(W \cdot [h_{chased}; h_{NP2}] + b)$
   - $h_{S} = f(W \cdot [h_{NP1}; h_{VP}] + b)$

## 应用场景

### 句法分析

树形RNN可以用于：
- 句法树的结构化表示
- 句子的语义表示
- 句法分析任务

### 情感分析

通过树形RNN可以：
- 捕捉句子的层次结构
- 分析不同短语的情感极性
- 提高情感分析的准确性

#### 情感分析示例

对于句子 "The movie was not very good"：

1. **短语级情感**：
   - "not very good" → 负面情感
   - "The movie" → 中性

2. **组合计算**：
   $$h_{not\ very\ good} = f(W \cdot [h_{not}; h_{very\ good}] + b)$$
   $$h_{sentence} = f(W \cdot [h_{The\ movie}; h_{was\ not\ very\ good}] + b)$$

## 优势与局限

### 优势
- 能够捕捉句子的层次结构
- 可以处理变长序列
- 具有较好的可解释性

### 局限
- 计算复杂度较高
- 对树结构的质量依赖性强
- 训练过程可能不稳定

## 改进方向

1. **结构优化**
   - 引入注意力机制
   - 使用更复杂的组合函数

2. **训练策略**
   - 采用预训练方法
   - 使用多任务学习

3. **模型架构**
   - 结合Transformer架构
   - 引入图神经网络

### 注意力机制改进

在树形RNN中引入注意力机制：

$$h_n = f(W \cdot [h_{left}; h_{right}] + b + \sum_{i=1}^{k} \alpha_i \cdot h_i)$$

其中 $\alpha_i$ 是注意力权重，$h_i$ 是其他相关节点的表示。

### 多任务学习

同时优化多个相关任务：
- 句法分析
- 情感分析
- 语义相似度

损失函数：
$$L_{total} = \lambda_1 L_{syntax} + \lambda_2 L_{sentiment} + \lambda_3 L_{similarity}$$

其中 $\lambda_i$ 是各任务的权重。

