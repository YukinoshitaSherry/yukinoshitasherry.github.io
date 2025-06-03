---
title: NLP(8)：NLP中的卷积神经网络
date: 2024-02-07 19:00
categories:
  - 学AI/DS
tags:
  - NLP
desc: CS224n Lec11 笔记，资料整合与一些自己的思考。CNN、VD-CNN、Q-RNN。

---

- 参考
    - <a href="https://www.showmeai.tech/tutorials/36">`showmeai-斯坦福CS224n教程`</a>


## 基础知识

###  概念
卷积神经网络（CNN, Convolutional Neural Network）是一类具有局部连接、参数共享、特征提取能力强的神经网络结构，最早应用于图像处理，现已广泛用于NLP等序列建模任务。

#### 一维/二维卷积
- **一维卷积（1D-CNN）**：
  - 主要用于处理序列数据（如文本、语音），卷积核在时间/序列维度滑动。
  - 公式：
    $$y_i = f\left(\sum_{j=0}^{k-1} w_j \cdot x_{i+j} + b\right)$$
    其中 $k$ 为卷积核宽度，$w_j$ 为权重，$x_{i+j}$ 为输入序列片段，$b$ 为偏置。
- **二维卷积（2D-CNN）**：
  - 主要用于图像等二维数据，NLP中可用于字符级建模（如字符嵌入矩阵的卷积）。
  - 公式：
    $$y_{i,j} = f\left(\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} w_{m,n} \cdot x_{i+m, j+n} + b\right)$$
    其中 $M,N$ 为卷积核高宽。

#### 感受野
- 感受野指输出神经元能"看到"的输入区域大小。
- 堆叠多层卷积可扩大感受野，捕获更长距离依赖。

#### Padding 

- **Padding（填充）**：
  - 作用：在输入序列（或特征图）两端补零，使卷积操作后输出长度不变或满足特定需求。
  - 数学推导：
    - 输入长度为 $n$，卷积核宽度为 $k$，步长为 $s$，填充为 $p$，则输出长度：
      $$
      \text{output\_len} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1
      $$
    - 常见填充策略：
      - **valid**（无填充）：$p=0$，输出长度变短。
      - **same**（输出长度与输入相同）：$p = \left\lfloor \frac{k-1}{2} \right\rfloor$。
      - **full**（最大填充）：$p = k-1$，输出长度最大。
  - NLP实际用法：
    - 保证序列边界信息不丢失，常用same padding。
    - 对于变长输入，padding有助于批量处理。
  - 注意事项：
    - 过多padding会引入无效信息，影响特征学习。
    - padding值一般为0，但也可用其他常数。

    
#### Stride

- **Stride（步长）**：
  - 作用：控制卷积核每次滑动的步幅。
  - 数学推导：
    - 步长$s$越大，输出长度越短，特征下采样更明显。
    - 输出长度同上公式。
  - 常见用法：
    - $s=1$时，卷积核逐步滑动，信息保留最完整。
    - $s>1$时，卷积核跳跃滑动，常用于降采样、特征压缩。
  - NLP实际用法：
    - 一般采用$s=1$，以充分捕获序列细节。
    - 也可结合池化层实现多尺度特征提取。
  - 注意事项：
    - 步长过大可能导致信息丢失。
    - 步长与padding、kernel size共同决定输出形状。

**举例说明**：
- 输入序列长度$n=10$，卷积核宽度$k=3$，步长$s=1$，padding $p=1$：
  $$
  \text{output\_len} = \left\lfloor \frac{10 + 2*1 - 3}{1} \right\rfloor + 1 = 10
  $$
- 若$s=2$，则：
  $$
  \text{output\_len} = \left\lfloor \frac{10 + 2*1 - 3}{2} \right\rfloor + 1 = 5
  $$

**总结**：
- padding和stride是卷积神经网络中影响输出形状和特征提取能力的重要参数。
- 合理选择padding和stride有助于提升模型性能和适应不同任务需求。

#### 池化

池化（Pooling）是卷积神经网络中用于降维和提取主要特征的操作，能够增强特征的不变性和鲁棒性。
常见池化方式如下：

1. **最大池化（Max Pooling）**
   - 取池化窗口内的最大值。
   - 公式：
     $$
     h_{max} = \max_{i=1}^n h_i
     $$
     或滑动窗口形式：
     $$
     p_i = \max_{j=0}^{k-1} h_{i+j}
     $$
     其中：
     - $h_{i+j}$：第$i+j$个输入特征
     - $k$：池化窗口大小
     - $p_i$：第$i$个池化窗口的输出
   - 优点：保留最显著的特征，突出局部极值，对噪声有一定抑制能力。
   - 缺点：可能丢失其他有用信息。
   - 适用场景：分类任务、需要突出显著特征时。

2. **平均池化（Average Pooling）**
   - 取池化窗口内的平均值。
   - 公式：
     $$
     h_{avg} = \frac{1}{n}\sum_{i=1}^n h_i
     $$
     或滑动窗口形式：
     $$
     p_i = \frac{1}{k} \sum_{j=0}^{k-1} h_{i+j}
     $$
     其中：
     - $h_{i+j}$：第$i+j$个输入特征
     - $k$：池化窗口大小
     - $p_i$：第$i$个池化窗口的输出
   - 优点：平滑特征，考虑所有信息。
   - 缺点：可能弱化显著特征，受噪声影响。
   - 适用场景：需要整体信息、特征平滑时。

3. **k-max池化（k-max Pooling）**
   - 选取序列或窗口内前$k$大的值（而不是只取最大值）。
   - 公式：
     $$
     p = \text{Top-}k\left(\{h_i\}_{i=1}^n\right)
     $$
     其中：
     - $h_i$：第$i$个输入特征
     - $n$：输入特征总长度
     - $k$：保留的最大值个数
     - $p$：输出的$k$个最大值组成的向量
   - 优点：保留更多重要特征，适合文本长度变化大、需要丰富特征表达的NLP任务。
   - 缺点：输出为定长$k$，需合理选择$k$。
   - 适用场景：句子建模、文本匹配、变长输入。

4. **动态池化（Dynamic Pooling）**
   - 结合最大池化和平均池化的优点。
   - 公式：
     $$
     h_{dynamic} = \alpha h_{max} + (1-\alpha)h_{avg}
     $$
     其中：
     - $h_{max}$：最大池化结果
     - $h_{avg}$：平均池化结果
     - $\alpha$：可学习参数，$0\leq\alpha\leq1$
   - 优点：兼顾显著特征与整体信息，灵活性强。
   - 适用场景：需要自适应特征融合时。

5. **全局池化（Global Pooling）**
   - 对整个序列或特征图做池化（如全局最大池化、全局平均池化）。
   - 公式：
     $$
     p = \max_{i} h_i \quad \text{或} \quad p = \frac{1}{n} \sum_{i=1}^n h_i
     $$
     其中：
     - $h_i$：第$i$个输入特征
     - $n$：输入特征总长度
     - $p$：池化输出（标量或向量）
   - 优点：将变长输入映射为定长输出，便于后续全连接层处理。
   - 适用场景：文本分类、句子/文档级特征提取。

**池化的意义**：
- 降低特征维度，减少计算量
- 提高特征的平移不变性
- 抑制噪声，突出主要特征
- 便于处理变长输入

<br>

**常用池化操作PyTorch代码示例：**

```python
import torch
import torch.nn as nn

# 假设输入为(batch, channel, seq_len)
x = torch.randn(2, 8, 16)  # batch=2, channel=8, 序列长度16

# 最大池化
max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
output_max = max_pool(x)  # 输出shape: (2, 8, 8)

# 平均池化
avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
output_avg = avg_pool(x)  # 输出shape: (2, 8, 8)

# k-max池化（需手动实现）
k = 3
output_kmax, _ = x.topk(k, dim=2)  # 输出shape: (2, 8, 3)

# 全局最大池化
output_global_max = torch.max(x, dim=2).values  # 输出shape: (2, 8)

# 全局平均池化
output_global_avg = torch.mean(x, dim=2)  # 输出shape: (2, 8)
```
<br>

**实际例子：卷积和池化后channel数量的计算**

- 假设输入为(batch, channel_in, seq_len) = (2, 8, 16)
- 经过一层一维卷积：
  ```python
  conv = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
  x_conv = conv(x)  # 输出shape: (2, 32, 16)
  ```
  - 说明：卷积后channel数量由out_channels决定，这里变为32。
- 经过最大池化：
  ```python
  pool = nn.MaxPool1d(kernel_size=2, stride=2)
  x_pool = pool(x_conv)  # 输出shape: (2, 32, 8)
  ```
  - 说明：池化不会改变channel数量，只会改变序列长度（宽度/高度）。


#### 激活函数
- **ReLU**: $f(x) = \max(0, x)$，收敛快，常用于CNN。
- **Tanh/Sigmoid**：早期常用，现多用于特殊场景。
- **LeakyReLU/ELU等**：解决ReLU"死亡"问题。

#### 参数量计算
- 单个卷积核参数量：$k \times d$（宽度$k$，嵌入维度$d$），加上偏置$1$。
- 总参数量：卷积核数$\times$每个卷积核参数量。

### 在NLP中使用CNN

#### 原因

卷积神经网络（CNN）最初是为计算机视觉任务设计的，但在NLP领域也有其独特的优势：

1. **并行计算效率**：
   - RNN需要按顺序处理序列，而CNN可以并行处理整个序列
   - 在GPU上，CNN的并行计算能力可以显著提升训练速度
   - 对于长序列，CNN的计算效率优势更加明显

2. **特征提取能力**：
   - 能够有效捕获局部特征和模式
   - 通过不同大小的卷积核，可以捕获不同尺度的特征
   - 特别适合处理n-gram特征

3. **参数共享**：
   - 同一个卷积核在整个序列上共享参数
   - 大大减少了模型参数量
   - 有助于防止过拟合
   - 提高了模型的泛化能力

4. **层次化特征学习**：
   - 浅层网络捕获基础特征（如词性、短语）
   - 深层网络捕获更抽象的特征（如语义、情感）
   - 这种层次化结构符合语言的特征

#### 基本结构

在NLP中，CNN的基本结构包括：

1. **输入层**：
   - 词嵌入矩阵 $E \in \mathbb{R}^{n \times d}$
   - $n$ 是序列长度
   - $d$ 是词向量维度
   - 每个词被转换为 $d$ 维的密集向量

2. **卷积层**：
   - 使用不同大小的卷积核 $W \in \mathbb{R}^{k \times d}$
   - $k$ 是卷积核大小（通常为2-5）
   - 每个卷积核可以捕获不同长度的n-gram特征
   - 多个卷积核可以并行使用

3. **池化层**：
   - 最大池化：提取最显著的特征
   - 平均池化：考虑所有特征的平均值
   - 动态池化：根据任务需求选择池化策略

4. **全连接层**：
   - 将池化后的特征映射到目标空间
   - 通常使用softmax进行多分类
   - 可以添加dropout防止过拟合

## 数学原理

### 词嵌入层

1. **词嵌入矩阵构建**：
   $$E = [e_1, e_2, ..., e_n]$$
   其中：
   - $e_i \in \mathbb{R}^d$ 是第 $i$ 个词的词向量
   - $d$ 是词向量维度（通常为100-300）
   - $n$ 是序列长度

2. **词向量初始化**：
   - 随机初始化
   - 预训练词向量（如Word2Vec、GloVe）
   - 微调或固定

3. **位置编码**(需要额外加入)：
   - 添加位置信息
   - 使用正弦位置编码
   - 或学习位置编码


### 卷积

对于输入序列 $X = [x_1, x_2, ..., x_n]$，其中每个 $x_i \in \mathbb{R}^d$ 是 $d$ 维的词向量，卷积操作可以表示为：

$$h_i = f(W \cdot X_{i:i+k-1} + b)$$

详细推导过程：

1. **卷积核滑动**：
   - 卷积核 $W \in \mathbb{R}^{k \times d}$ 在序列上滑动
   - 每次滑动步长为1
   - 对于位置 $i$，计算 $X_{i:i+k-1}$ 与 $W$ 的点积

2. **具体计算过程**：
   $$h_i = f(\sum_{j=0}^{k-1} W_j \cdot x_{i+j} + b)$$
   其中：
   - $W_j$ 是卷积核的第 $j$ 行
   - $x_{i+j}$ 是输入序列的第 $i+j$ 个词向量
   - $b$ 是偏置项
   - $f$ 是激活函数（通常使用ReLU）

3. **输出维度**：
   - 输入序列长度：$n$
   - 卷积核大小：$k$
   - 输出序列长度：$n-k+1$
   - 每个位置输出一个标量值



实现详解：

1. **单卷积核实现**：
   $$c_i = \text{ReLU}(W \cdot E_{i:i+k-1} + b)$$
   其中：
   - $W \in \mathbb{R}^{k \times d}$ 是卷积核
   - $b \in \mathbb{R}$ 是偏置项
   - ReLU激活函数：$f(x) = \max(0, x)$

2. **多卷积核实现**：
   - 使用不同大小的卷积核
   - 每个卷积核捕获不同长度的n-gram
   - 所有特征图拼接

3. **残差连接**：
   $$h_i = f(W \cdot E_{i:i+k-1} + b) + E_i$$
   - 帮助训练更深的网络
   - 缓解梯度消失问题

### 多通道卷积

在NLP中，我们经常使用多个卷积核来捕获不同的特征：

$$h_i^j = f(W^j \cdot X_{i:i+k-1} + b^j)$$

详细说明：

1. **多卷积核结构**：
   - 使用 $m$ 个不同的卷积核
   - 每个卷积核大小可以不同
   - 每个卷积核捕获不同的特征模式

2. **特征图计算**：
   - 第 $j$ 个卷积核产生特征图 $h^j = [h_1^j, h_2^j, ..., h_{n-k+1}^j]$
   - 所有特征图组合成特征矩阵 $H \in \mathbb{R}^{(n-k+1) \times m}$

3. **参数共享机制**：
   - 每个卷积核在整个序列上共享参数
   - 大大减少了模型参数量
   - 提高了模型的泛化能力

### 池化层

实现详解：

1. **最大池化实现**：
   $$p = \max_{i=1}^{n-k+1} c_i$$
   - 提取最显著的特征
   - 保持特征的不变性

2. **平均池化实现**：
   $$p = \frac{1}{n-k+1}\sum_{i=1}^{n-k+1} c_i$$
   - 平滑特征
   - 考虑所有位置的信息

3. **注意力池化**：
   $$p = \sum_{i=1}^{n-k+1} \alpha_i c_i$$
   其中：
   - $\alpha_i$ 是注意力权重
   - 通过注意力机制学习重要特征

## 应用示例

### 文本分类

1. **模型结构**：
   $$y = \text{softmax}(W_{fc} \cdot p + b_{fc})$$
   其中：
   - $W_{fc} \in \mathbb{R}^{m \times c}$ 是全连接层权重
   - $b_{fc} \in \mathbb{R}^c$ 是偏置项
   - $c$ 是类别数量
   - $p$ 是池化后的特征向量

2. **损失函数**：
   $$\mathcal{L} = -\sum_{i=1}^c y_i \log \hat{y}_i$$
   其中：
   - $y_i$ 是真实标签
   - $\hat{y}_i$ 是预测概率
   - $c$ 是类别数量



3. **具体实现**：
   ```python
   class TextCNN(nn.Module):
       def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5]):
           super(TextCNN, self).__init__()
           self.embedding = nn.Embedding(vocab_size, embed_dim)
           self.convs = nn.ModuleList([
               nn.Conv2d(1, 100, (k, embed_dim)) for k in kernel_sizes
           ])
           self.dropout = nn.Dropout(0.5)
           self.fc = nn.Linear(len(kernel_sizes) * 100, num_classes)
           
       def forward(self, x):
           x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
           x = x.unsqueeze(1)    # [batch_size, 1, seq_len, embed_dim]
           x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
           x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
           x = torch.cat(x, 1)
           x = self.dropout(x)
           return self.fc(x)
   ```


### 序列标注

1. **滑动窗口方法**：
   $$y_i = \text{softmax}(W_{fc} \cdot h_i + b_{fc})$$
   其中：
   - $h_i$ 是位置 $i$ 的特征
   - $y_i$ 是位置 $i$ 的标签预测
   - $W_{fc}$ 是全连接层权重
   - $b_{fc}$ 是偏置项

2. **CRF层**：
   - 考虑标签之间的依赖关系
   - 使用维特比算法进行解码
   - 提高序列标注的准确性
   - 转移矩阵学习标签间关系

3. **具体实现**：
   ```python
   class CNNCRF(nn.Module):
       def __init__(self, vocab_size, embed_dim, num_tags):
           super(CNNCRF, self).__init__()
           self.embedding = nn.Embedding(vocab_size, embed_dim)
           self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
           self.dropout = nn.Dropout(0.5)
           self.hidden2tag = nn.Linear(128, num_tags)
           self.crf = CRF(num_tags)
           
       def forward(self, x, mask):
           x = self.embedding(x)
           x = x.transpose(1, 2)
           x = F.relu(self.conv(x))
           x = x.transpose(1, 2)
           x = self.dropout(x)
           emissions = self.hidden2tag(x)
           return self.crf(emissions, mask)
   ```


### 情感分析

1. **模型架构**：
   - 词嵌入层
   - 多尺度卷积层
   - 注意力机制
   - 情感分类层

2. **注意力机制**：
   $$a_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$$
   $$c = \sum_i a_i h_i$$
   其中：
   - $e_i$ 是注意力分数
   - $h_i$ 是隐藏状态
   - $c$ 是上下文向量

3. **具体实现**：
   ```python
   class SentimentCNN(nn.Module):
       def __init__(self, vocab_size, embed_dim, num_classes):
           super(SentimentCNN, self).__init__()
           self.embedding = nn.Embedding(vocab_size, embed_dim)
           self.conv1 = nn.Conv1d(embed_dim, 100, 3)
           self.conv2 = nn.Conv1d(embed_dim, 100, 4)
           self.conv3 = nn.Conv1d(embed_dim, 100, 5)
           self.attention = nn.Linear(100, 1)
           self.fc = nn.Linear(300, num_classes)
           
       def forward(self, x):
           x = self.embedding(x)
           x = x.transpose(1, 2)
           c1 = F.relu(self.conv1(x))
           c2 = F.relu(self.conv2(x))
           c3 = F.relu(self.conv3(x))
           p1 = F.max_pool1d(c1, c1.size(2))
           p2 = F.max_pool1d(c2, c2.size(2))
           p3 = F.max_pool1d(c3, c3.size(2))
           p = torch.cat([p1, p2, p3], 1).squeeze(2)
           return self.fc(p)
   ```



### 文本匹配

1. **模型结构**：
   - 双塔结构
   - 交互层
   - 匹配层
   - 预测层

2. **交互计算**：
   $$M_{ij} = f(h_i^1, h_j^2)$$
   其中：
   - $h_i^1$ 是第一个文本的特征
   - $h_j^2$ 是第二个文本的特征
   - $f$ 是交互函数

3. **具体实现**：
   ```python
   class TextMatchingCNN(nn.Module):
       def __init__(self, vocab_size, embed_dim):
           super(TextMatchingCNN, self).__init__()
           self.embedding = nn.Embedding(vocab_size, embed_dim)
           self.conv = nn.Conv1d(embed_dim, 100, 3)
           self.fc = nn.Linear(200, 1)
           
       def forward(self, x1, x2):
           x1 = self.embedding(x1)
           x2 = self.embedding(x2)
           x1 = x1.transpose(1, 2)
           x2 = x2.transpose(1, 2)
           h1 = F.relu(self.conv(x1))
           h2 = F.relu(self.conv(x2))
           p1 = F.max_pool1d(h1, h1.size(2))
           p2 = F.max_pool1d(h2, h2.size(2))
           p = torch.cat([p1, p2], 1).squeeze(2)
           return torch.sigmoid(self.fc(p))
   ```


## 优化技巧

### 跳接结构
**Skip Connections**

跳接结构是一种重要的网络设计技术，通过添加直接连接来改善深层网络的训练。

#### 基本概念
- 跳接结构允许信息在网络中直接传递，绕过某些层
- 最早在ResNet中提出，现已广泛应用于各种网络架构
- 有助于解决深层网络的梯度消失问题

#### 数学表示
对于输入$x$和变换函数$F(x)$，跳接结构可以表示为：
$$y = F(x) + x$$

#### 实现方式
1. **残差连接（Residual Connection）**：
   - 最简单的跳接形式
   - 直接将输入加到变换后的输出上
   - 公式：$h_t = F(h_{t-1}, x_t) + h_{t-1}$

2. **密集连接（Dense Connection）**：
   - 将每一层的输出连接到所有后续层
   - 提供更丰富的特征重用
   - 公式：$h_t = [h_0, h_1, ..., h_{t-1}]$

3. **高速网络（Highway Network）**：
   - 使用门控机制控制信息流
   - 结合了残差连接和门控机制
   - 公式：$h_t = g_t \odot F(h_{t-1}, x_t) + (1-g_t) \odot h_{t-1}$

### 门控单元
**Gated Units**

门控单元是一种动态控制信息流的机制，可以学习性地决定保留或丢弃信息。

#### 垂直门控
- 在深度方向上控制信息流
- 可以学习性地决定是否使用深层特征
- 有助于训练更深的网络

#### 实现方式
1. **门控机制**：
   $$g = \sigma(W_g \cdot [x, h] + b_g)$$
   $$h_{new} = g \odot h_{old} + (1-g) \odot h_{transformed}$$

2. **注意力门控**：
   $$g = \text{softmax}(W_a \cdot [x, h])$$
   $$h_{new} = g \odot h_{old}$$

### Batch Normalization

批归一化是一种重要的正则化技术，通过归一化每层的输入来加速训练。

#### 基本原理
- 对每个mini-batch的数据进行归一化
- 将激活值缩放到零均值和单位方差
- 添加可学习的缩放和平移参数

#### 数学表示
1. **计算均值和方差**：
   $$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$$
   $$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

2. **归一化**：
   $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

3. **缩放和平移**：
   $$y_i = \gamma \hat{x}_i + \beta$$

#### 优势
- 减少内部协变量偏移
- 允许使用更大的学习率
- 减少对初始化的依赖
- 提供正则化效果

### 1x1卷积
1x1卷积是一种特殊的卷积操作，具有重要的功能。

#### 基本概念
- 卷积核大小为1x1
- 可以看作是一个跨通道的全连接层
- 用于调整通道数量和特征融合

#### 数学表示
对于输入特征图$X \in \mathbb{R}^{C \times H \times W}$和1x1卷积核$W \in \mathbb{R}^{C' \times C}$：
$$Y_{c'} = \sum_{c=1}^C W_{c',c} \cdot X_c$$

#### 应用场景
1. **通道降维**：
   - 减少计算量
   - 降低模型复杂度

2. **特征融合**：
   - 跨通道信息交互
   - 非线性特征提取

3. **网络压缩**：
   - 减少参数量
   - 提高计算效率

<br>

## 高级CNN结构

### CNN在机器翻译中的应用

#### 基本架构
- 使用CNN进行编码
- 使用RNN进行解码
- 结合注意力机制

#### 优势
- 并行计算效率高
- 可以捕获局部特征
- 适合处理长序列

#### 经典模型
1. **Kalchbrenner and Blunsom (2013)**：
   - 最早成功的神经机器翻译模型之一
   - 使用CNN进行编码
   - 使用RNN进行解码

2. **Character-Aware Neural Language Models**：
   - 基于字符的单词嵌入
   - 利用卷积、highway network和LSTM
   - 可以处理未知词

### 深度CNN用于文本分类

#### VD-CNN结构
- 类似VGG和ResNet的深度结构
- 使用固定大小的输入
- 每个阶段都有局部池化操作

#### 卷积模块
- 每个卷积块包含两个卷积层
- 每个卷积层后接BatchNorm和ReLU
- 使用padding保持维度

#### 实验结果
- 在大型文本分类数据集上表现优异
- 深度网络效果更好
- 残差连接效果显著
- MaxPooling优于其他池化方法

### Q-RNN模型

#### 基本结构
- 结合CNN和RNN的优点
- 时间上并行的卷积
- 跨通道并行的门控
- 使用卷积层替代RNN的循环连接
- 通过门控机制控制信息流

#### 数学表示
1. **卷积计算候选**：
   $$c_t = \text{conv}(x_{t-k+1:t})$$
   - $k$为卷积核大小
   - $x_{t-k+1:t}$表示时间窗口内的输入
   - 卷积操作可以并行计算

2. **门控机制**：
   $$f_t = \sigma(W_f \cdot [x_t, h_{t-1}])$$
   $$o_t = \sigma(W_o \cdot [x_t, h_{t-1}])$$
   - $f_t$为遗忘门
   - $o_t$为输出门
   - $\sigma$为sigmoid激活函数
   - $W_f, W_o$为可学习参数

3. **状态更新**：
   $$h_t = f_t \odot h_{t-1} + (1-f_t) \odot c_t$$
   - $\odot$表示逐元素乘法
   - $h_t$为当前时刻的隐藏状态
   - $h_{t-1}$为上一时刻的隐藏状态
   - $c_t$为当前时刻的候选状态

#### 优势
- 比LSTM更快：
  - 卷积操作可以并行计算
  - 减少了序列依赖
  - 提高了计算效率
- 可解释性更好：
  - 门控机制更简单
  - 状态更新更直观
  - 易于分析和调试
- 适合并行计算：
  - 卷积层可以并行化
  - 门控计算可以并行化
  - 适合GPU加速

#### 局限性
- 对字符级语言模型效果不如LSTM：
  - 难以捕获长距离依赖
  - 对序列顺序的建模能力较弱
  - 在需要精确字符建模的任务上表现不佳
- 需要更深的网络来获得同等性能：
  - 单层Q-RNN表达能力有限
  - 需要堆叠多层来提升性能
  - 增加了模型复杂度
- 长距离依赖建模能力有限：
  - 卷积操作主要捕获局部特征
  - 难以直接建模远距离关系
  - 需要额外的机制来增强长距离建模能力
<br>

## 总结

### CNN特点
#### 优点

1. **并行计算效率**：
   - 可以并行处理整个序列
   - 在GPU上效率高
   - 训练速度快

2. **特征提取能力**：
   - 有效捕获局部特征
   - 可以学习不同尺度的特征
   - 适合处理n-gram特征

3. **参数共享**：
   - 减少模型参数量
   - 提高训练效率
   - 防止过拟合

4. **训练速度快**：
   - 并行计算
   - 参数共享
   - 结构简单

#### 缺点

1. **长距离依赖**：
   - 难以捕获长距离依赖
   - 需要很深的网络
   - 计算成本高

2. **固定长度输入**：
   - 需要固定长度的输入
   - 可能丢失信息
   - 需要截断或填充

3. **位置信息**：
   - 可能丢失位置信息
   - 需要额外的位置编码
   - 影响序列建模

4. **序列顺序**：
   - 对序列顺序不敏感
   - 可能影响语义理解
   - 需要额外的位置编码

### CNN vs RNN
RNN对**序列标记**和**分类**之类的事情有很好的效果；以及语言模型**预测**下一个单词，结合注意力机制会取得很好的效果。但是对于某个句子的**整体解释**，CNN做的是更好的。

1. **计算效率**：
   - CNN：并行计算，效率高
   - RNN：顺序计算，效率低
   - 在长序列上差异明显

2. **长距离依赖**：
   - CNN：难以捕获
   - RNN：可以捕获
   - 需要不同的解决方案

3. **并行性**：
   - CNN：天然并行
   - RNN：难以并行
   - 影响训练速度

4. **参数数量**：
   - CNN：参数共享，数量少
   - RNN：参数不共享，数量多
   - 影响模型复杂度

### CNN vs Transformer

1. **计算复杂度**：
   - CNN：$O(n)$
   - Transformer：$O(n^2)$
   - 在长序列上差异明显

2. **长距离依赖**：
   - CNN：难以捕获
   - Transformer：可以捕获
   - 通过自注意力机制

3. **并行性**：
   - CNN：天然并行
   - Transformer：天然并行
   - 两者都适合GPU加速

4. **位置编码**：
   - CNN：需要额外编码
   - Transformer：内置位置编码
   - 影响序列建模

### 常见方法对比

| 方法 | 优点 | 局限 | 适用场景 |
|------|------|------|----------|
| 词袋模型 / Bag of Vectors | 实现简单，适合简单分类问题，是很好的基线方法；可与ReLU层结合（如Deep Averaging Networks） | 丢失词序信息，无法建模上下文 | 文本分类、情感分析 |
| 词窗分类 / Window Model | 适合局部问题（如单字分类），不需要广泛上下文；只关注窗口内上下文，计算高效 | 无法捕捉长距离依赖 | 词性标注（POS）、命名实体识别（NER） |
| 卷积神经网络 / CNN | 适合分类任务，能有效提取局部n-gram特征，易于GPU并行化 | 难以解释，长距离依赖建模能力有限 | 文本分类、短语建模 |
| 循环神经网络 / RNN | 天然建模序列顺序，适合序列标注、语言模型等；从左到右的认知更具可信度，结合注意力机制表现更佳 | 分类任务若只用最后状态效果不佳，训练速度慢于CNN | 序列标注、文本生成、语言建模 |



