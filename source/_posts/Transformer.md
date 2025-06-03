---
title: Transformer论文精读
date: 2024-03-23
categories:
    - 学AI/DS
tags:
    - LLM
    - NLP
    - CV
desc: Transformer模型笔记整理，参考资料已附在开头。
---

# 参考
<br>

## 资料
- [Transformer模型 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/Transformer%E6%A8%A1%E5%9E%8B)
- [Transformer模型详解（图解最完整版） - 知乎](https://zhuanlan.zhihu.com/p/338817680)
- [Transformer背景介绍及架构解析 - 知乎](https://zhuanlan.zhihu.com/p/572491345)
- [李沐论文精读系列一： ResNet、Transformer、GAN、BERT_李沐读论文-CSDN博客](https://blog.csdn.net/qq_56591814/article/details/127313216)
- [《The Annotated Transformer》翻译——注释和代码实现《Attention Is All You Need》-CSDN博客](https://blog.csdn.net/qq_56591814/article/details/120278245)
- [Transformer-LLM笔记 - 飞书](https://mj8stxx1cd.feishu.cn/wiki/OCTNw0yhTiuxBZkK6sLcjSBtnac)
<br>

## 论文
**Attention is all you need.** `NeurIPS` 2017
[[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762)
代码：[tensorflow/tensor2tensor: Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.](https://github.com/tensorflow/tensor2tensor)

<br>

### 摘要
主流的序列转换模型都是基于复杂的循环或卷积神经网络，这个模型包含一个编码器和一个解码器。具有最好性能的模型在编码和解码之间**通过一个注意力机制链接编解码器**。我们提出了一个新的简单网络结构——Transformer，其**仅仅是基于注意力机制，而完全不需要之前的循环或卷积**。在两个机器翻译任务上的实验表明，该模型具有更好的性能，同时并行度更好，并且训练时间更少；泛化到其它任务效果也不错。

<br>

### 结论
本文介绍了Transformer，这是**第一个完全基于注意力的序列转换模型**，用**多头自注意力**（`multi-headed self-attention`）代替了 `encoder-decoder` 架构中最常用的**循环层**。
我们对基于注意力的模型的未来感到兴奋， 并计划将Transformer应用于文本之外的涉及输入和输出模式的问题中任务，以有效处理大型输入/输出任务，如图像、音频和视频等。让**生成不那么时序化**是我们的另一个研究目标。


<br>

# 背景
在Transformer模型之前，大多数最先进的NLP系统都依赖于诸如[LSTM](https://zh.wikipedia.org/wiki/%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6 "长短期记忆")、[门控循环单元](https://zh.wikipedia.org/w/index.php?title=%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83&action=edit&redlink=1 "门控循环单元（页面不存在）")（GRU）等门控RNN模型，并在此基础上增加了[注意力机制](https://zh.wikipedia.org/wiki/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6 "注意力机制")。Transformer正是在注意力机制的基础上构建的，但其**没有使用RNN结构**，而是纯基于注意力机制，来构造输入和输出之间的全局依赖关系。

Transformer可以进行更多的并行化，训练时间更短但翻译效果更好。


<br>

## 顺序处理

[RNN](https://zh.wikipedia.org/wiki/RNN "RNN")模型按顺序处理每一个标记（token）并维护一个状态向量，其中包含所有已输入数据的表示。如要处理第n个标记，模型将表示句中到第n−1个标记为止的状态向量与最新的第n个标记的信息结合在一起创建一个新的状态向量，以此表示句中到第n个标记为止的状态。
当前时刻隐藏状态`ht`，是由上一时刻隐藏状态`ht−1`和 `t`时刻输入共同决定的。（把之前的信息都放在隐藏状态里，一个个传递下去，是RNN处理时序信息的关键）
从理论上讲，如果状态向量不断继续编码每个标记的上下文信息，则来自一个标记的信息可以在序列中不断传播下去。

- 但在实践中，这一机制是有缺陷的：
	- [梯度消失问题](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98 "梯度消失问题")使得**长句末尾**的模型状态会**缺少**前面标记的精确信息。
		- 除非你把`ht`维度设置的很高，可以把每一个时间步的信息都存下来。但这样会造成内存开销很大。
	- 每个标记的计算都依赖于先前标记的计算结果，这也使得其很难在现代深度学习硬件上进行**并行**处理，这导致了RNN模型训练效率低下。

<br>

## 自注意力机制

[注意力机制](https://zh.wikipedia.org/wiki/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6 "注意力机制")解决了上述这些问题。这一机制让模型得以提取序列中任意先前点的状态信息。注意力层能够访问所有先前的状态并根据学习到的相关性度量对其进行加权，从而提供相距很远的标记的相关信息。
在RNN模型中添加[注意力机制](https://zh.wikipedia.org/wiki/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6 "注意力机制")能提高模型的性能。
而Transformer架构的发展表明，[注意力机制](https://zh.wikipedia.org/wiki/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6 "注意力机制")本身就足够强大，并且不需要像RNN模型一样再对数据进行顺序循环处理。Transformer模型采用了**没有RNN模型**的[注意力机制](https://zh.wikipedia.org/wiki/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6 "注意力机制")，它能够同时处理所有标记并计算它们之间的注意力权重。由于注意力机制仅使用来自之前层中其他标记的信息，因此可以并行计算所有标记以提高训练速度。

<br>

# 架构

## 总览
<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Transformer%20structure.png" alt="Transformer structure" width="80%">
<br>

- Transformer总体架构可分为四个部分
	- **输入部分**<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250223214300.png" alt="Transformer structure" width="50%" style="display: block; margin: 0 auto;">
		- 源文本嵌入层及其位置编码器
		- 目标文本嵌入层及其位置编码器【注】Outputs（shifted right）：解码器在$t_0$时刻其实是没有输入的，其输入是编码器的输出，所以这里写的是output。shifted right就是逐个右移的意思。
		<br>
	- **编码器部分**<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250223214525.png" alt="Transformer structure" width="25%" style="display: block; margin: 0 auto;">
		- 由N个编码器层堆叠而成【Nx：模块堆叠N次】
		- 每个编码器层由两个子层连接结构组成
		- 第一个子层连接结构包括一个多头自注意子层和规范化层以及一个残差连接
		- 第二个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接
		<br>
	- **解码器部分**<img src="https://raw.githubusercontent.com/yukinoshitasherry/qycf_picbed/main/img/Pasted%20image%2020250223214536.png" alt="Transformer structure" width="25%" style="display: block; margin: 0 auto;">
		- 由N个解码器层堆叠而成
		- 每个解码器层由三个子层连接结构组成
		- 第一个子层连接结构包括一个Masked 多头自注意力子层和规范化层以及一个残差连接
		- 第二个子层连接结构包括一个多头注意力子层和规范化层以及一个残差连接
		- 第三个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接
		<br>
	- **输出部分**<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250223214325.png" alt="Transformer structure" width="25%" style="display: block; margin: 0 auto;">
		- 线性层
		- softmax层

<br>

## 输入
Transformer 中单词的输入表示 **x**由**单词 Embedding** 和**位置 Embedding** （Positional Encoding）相加得到。
![](https://pic2.zhimg.com/v2-b0a11f97ab22f5d9ebc396bc50fa9c3f_1440w.jpg)

<br>

<mark style="background: #FFF3A3A6;">Transformer 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则 Transformer 就是一个词袋模型了。</mark>

<br>

### 单词 Embedding

单词的 Embedding 有很多种方式可以获取，例如采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到。

<br>

#### Word2Vec
- Word2Vec模型Word2Vec能够有效地捕捉词与词之间的语义关系, 包括两种主要的架构：
	- CBOW（Continuous Bag of Words）
		- 通过上下文词汇来预测中心词。
	- Skip-gram
		- 通过中心词来预测上下文词汇。
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# 示例文本数据
sentences = [
    "this is a sentence",
    "this sentence is for word2vec",
    "word2vec is a tool"
]

# 将文本分割成单词列表
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# 训练Word2Vec模型（CBOW）
model_cbow = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)

# 保存模型
model_cbow.save("word2vec_cbow.model")

# 加载模型
model_cbow = Word2Vec.load("word2vec_cbow.model")

# 获取单词的向量表示
vector = model_cbow.wv['word2vec']

# 查找与“word2vec”最相似的单词
similar_words = model_cbow.wv.most_similar('word2vec')
print(similar_words)
```

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# 示例文本数据
sentences = [
    "this is a sentence",
    "this sentence is for word2vec",
    "word2vec is a tool"
]

# 将文本分割成单词列表
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# 训练Word2Vec模型（Skip-gram）
model_skipgram = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# 保存模型
model_skipgram.save("word2vec_skipgram.model")

# 加载模型
model_skipgram = Word2Vec.load("word2vec_skipgram.model")

# 获取单词的向量表示
vector = model_skipgram.wv['word2vec']

# 查找与“word2vec”最相似的单词
similar_words = model_skipgram.wv.most_similar('word2vec')
print(similar_words)
```
<br>

#### GloVe
- GloVe（Global Vectors for Word Representation）考虑了词与词在文本中的全局共现信息，从而能够捕捉词的语义和语法特性。
	- GloVe算法通过构建词共现矩阵并利用矩阵分解技术来学习词向量, 其中矩阵的每个元素表示某两个词在一定上下文窗口内出现的频率。

```python
from collections import Counter
import numpy as np

# 统计词频
word_counts = Counter([word for text in tokenized_text for word in text])

# 生成共现矩阵
def build_cooccurrence_matrix(tokenized_text, window_size=5):
    vocab = list(word_counts.keys())
    cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))
    for text in tokenized_text:
        for i, word in enumerate(text):
            word_idx = vocab.index(word)
            start = max(0, i - window_size)
            end = min(len(text), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    co_word_idx = vocab.index(text[j])
                    cooccurrence_matrix[word_idx][co_word_idx] += 1
    return cooccurrence_matrix, vocab

co_occurrence_matrix, vocab = build_cooccurrence_matrix(tokenized_text)

from glove import Glove

# 训练GloVe模型
glove = Glove().fit(co_occurrence_matrix, epochs=100, no_threads=4, verbose=True)

# 保存模型
glove.save('glove_model.model')

# 加载已训练的模型
glove = Glove().load('glove_model.model')

# 获取某个词的向量
word_vector = glove.word_vectors[glove.dictionary['example']]

# 找到与某个词最相似的词
similar_words = glove.most_similar('example', number=5)  # 获取最相似的5个词
```

<br>

### 位置 Embedding

Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要** 。所以 Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。

位置 Embedding 用 **PE**表示，**PE** 的维度与单词 Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：
$$
\begin{align*}
PE_{(pos,2i)} &= \sin \left( pos / 10000^{2i/d} \right) 

\\

PE_{(pos,2i+1)} &= \cos \left( pos / 10000^{2i/d} \right)
\end{align*}
$$


其中，$pos$ 表示单词在句子中的位置，$d$ 表示 **PE**的维度 (与单词Embedding 一样)，$2i$ 表示偶数的维度，$2i+1$ 表示奇数维度 (即 $2i≤d, 2i+1≤d$)。

使用这种公式计算 PE 有以下的好处：
- 使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
- 可以让模型容易地计算出相对位置，对于固定长度的间距 k，$PE(pos+k)$ 可以表示为$PE(pos)$ 的线性函数。因为: $$Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B) ,  Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)$$
- 最终编码向量每个元素值都是在-1到1之间。

将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 **x**，**x** 就是 Transformer 的输入。

<br>

## 编码器与解码器

<br>

### 整体架构
大部分神经序列转换模型都使用encoder-decoder 结构。
编码器将一个输入序列$(x_1, \ldots, x_n)$ 映射到一个连续的表示 $z = (z_1, \ldots, z_n)$ 中。解码器对 z 中的每个元素，生成输出序列 $(y_1, \ldots, y_m)$。
每一步中，模型都是自回归的（auto-regressive），在生成下一个结果时，会将先前生成的结果加入输入序列来一起预测。
- 自回归模型的特点：过去时刻的输出可以作为当前时刻的输入。

编码器和解码器序列可以不一样长，且编码器可以一次看到整个序列，但是解码器是一步步输出的。
Transformer 遵循这种整体架构，对编码器和解码器使用堆叠的自注意力和逐点全连接层。

<br>

#### 编码器

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250223214525.png" alt="Transformer structure" width="25%" style="display: block; margin: 0 auto;">

<br>

- 由N=6个相同encoder层堆栈组成。每层有两个子层：
	- **Multi-head self-attention**
	- **FFNN**（前馈神经网络层，Feed Forward Neural Network）
		- 其实就是MLP。


- 两个子层都使用残差连接(residual connection)，然后进行层归一化（layer normalization）。
- 为了简单起见，模型中的所有子层以及嵌入层的向量维度都是$d_{model} = 512$
	- 如果输入输出维度不一样，残差连接就需要做投影，将其映射到统一维度。
	- 这种各层统一维度使得模型比较简单，只有N和$d_{\text{model}}$ 两个参数需要调。
	- 这和之前的CNN或MLP做法不一样，之前都会进行一些下采样

通过Multi-Head Attention, Feed Forward, Add & Norm 就可以构造出一个 Encoder block，Encoder block 接收输入矩阵 X(n×d) ，并输出一个矩阵 O(n×d) 。通过多个 Encoder block 叠加就可以组成 Encoder。

第一个 Encoder block 的输入为句子单词的表示向量矩阵，后续 Encoder block 的输入是前一个 Encoder block 的输出，最后一个 Encoder block 输出的矩阵就是**编码信息矩阵 C**，这一矩阵后续会用到 Decoder 中。

<br>

#### 解码器

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250223214536.png" alt="Transformer structure" width="25%" style="display: block; margin: 0 auto;">

<br>

- 同样由 N=6个相同的decoder层堆栈组成，每个层有三个子层：
	- **Masked multi-head self-attention**
		- 第一个 Multi-Head Attention 层
		- 在解码器里，Self Attention 层只允许关注到输出序列中早于当前位置之前的单词。
		- 具体做法是：在 Self Attention 分数经过 Softmax 层之前，使用attention mask，屏蔽当前位置之后的那些位置。
			- 对应masked位置使用一个很大的负数-inf，使得softmax之后其对应值为0
	- **Encoder-Decoder Attention** 
		- 第二个 Multi-Head Attention 层
		- 编码器输出最终向量，将会输入到每个解码器的Encoder-Decoder Attention层，用来帮解码器把注意力集中中输入序列的合适位置。
		- **K, V**矩阵使用 Encoder 的**编码信息矩阵C**进行计算，而**Q**使用上一个 Decoder block 的输出计算。
	- **FFNN**（前馈神经网络层，Feed Forward Neural Network）
		- 其实就是MLP。


<br>

### Add&Norm

Add & Norm 层由 Add 和 Norm 两部分组成，其计算公式如下：

$\text{LayerNorm}\left(X + \text{MultiHeadAttention}(X)\right)$
$\text{LayerNorm}\left(X + \text{FeedForward}(X)\right)$

- 编码器的两个子层都使用残差连接(residual connection)，然后进行层归一化(layer normalization)。
- 每个子层的输出:$\text{LayerNorm}\left(X + \text{FeedForward}(X)\right)$, 也可写成$LayerNorm(x + Sublayer(x))$，其中$Sublayer(x)$(即$\text{FeedForward}(X)$)是当前子层的输出。
- 其中X表示 Multi-Head Attention 或者 Feed Forward 的输入，MultiHeadAttention(X) 和FeedForward(X) 表示输出 (输出与输入X 维度是一样的，所以可以相加)。
- **Add**指**X**+MultiHeadAttention(**X**)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在 ResNet 中经常用到：

![](https://pic4.zhimg.com/v2-4b3dde965124bd00f9893b05ebcaad0f_1440w.jpg)

、
【LayerNorm】
- 为什么这里使用LN而不是BN？
	- `Batch Normalization`：在特征d/通道维度做归一化（均值变0方差变1），即归一化不同样本的同一特征。缺点是：
		- 计算变长序列时，变长序列后面会pad 0，这些pad部分是没有意义的，这样进行特征维度做归一化缺少实际意义。
		- 序列长度变化大时，计算出来的均值和方差抖动很大。
		- 预测时使用训练时记录下来的全局均值和方差。如果预测时新样本特别长，超过训练时的长度，那么超过部分是没有记录的均值和方差的，预测会出现问题。
	- `Layer Normalization`：在样本b维度进行归一化，即归一化一个样本所有特征。
		- NLP任务中一个序列的所有token都是同一语义空间，进行LN归一化有实际意义
		- 因为实是在每个样本内做的，序列变长时相比BN，计算的数值更稳定。
		- 不需要存一个全局的均值和方差，预测样本长度不影响最终结果。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250224005233.png" alt="Transformer structure" width="60%" style="display: block; margin: 0 auto;">

- 李沐画的图，蓝色BN,橙色LN。

<br>

### FFNN 
(Position-wise Feed-Forward Networks）
包括两个线性变换，并在两个线性变换中间有一个ReLU激活函数。
$FFN(x)=\max(0, XW_1 + b_1)W_2 + b_2$
本质是一个单隐藏层MLP：X是输入，Feed Forward 最终得到的输出矩阵的维度与X一致。X维度是512，W1把它扩大四倍到2048，W2把它缩小四倍变回512维。

Position就是序列中每个token，Position-wise 就是把MLP对每个token作用一次，且作用的是同一个MLP。即MLP只作用于最后一个维度d=512。

<br>

## Attention

<br>

### Scaled Dot-Product Attention

<br>

#### 结构
<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250223215933.png" alt="Transformer structure" width="50%" >

<br>	

#### 计算公式

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.  ———— 原文

<br>

   - 其输入为query、key(维度是 $d_{k}$)以及values(维度是 $d_{v}$)。
   - 计算query和所有key的点积，得到两个向量的相似度（结果越大相似度越高）；然后对每个点积结果除以 $\sqrt{d_{k}}$，
   - 点积结果输入softmax函数获得value的权重。
   - 最后对value进行加权求和。

在实践中，同时计算一组query的attention函数，并将它们组合成一个矩阵 $Q$。key和value也一起组成矩阵 $K$ 和 $V$。计算的输出矩阵为：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V
$$

<br>

##### 理解
【理解1】
参考
- b站@404号宇宙观察计划
- [【官方双语】直观解释注意力机制，Transformer的核心 | 【深度学习第6章】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1TZ421j7Ke/?vd_source=fa)
<br>

假设输入是一个句子，每个单词是一个token，首先将每个token进行embedding，映射到一个高维向量空间。这个向量仅表示该词的静态含义。 
有三个**可训练的权重矩阵**$W_Q$、$W_K$和$W_V$： 
- **每个token的embedding向量通过与$W_Q$相乘得到Q**
	- 3B1B为了便于理解抽象了一个“adj修饰n的例子”
	- 可以理解为一系列这个token关心的查询问题，比如“你是一个形容词吗？”，“你是一个动词吗？”，“你是一种颜色吗？” 
- **每个token的embedding向量通过与$W_K$相乘得到K**
	- 可以理解为对于这个token的信息描述，或者说对于Q当中的问题的回答，比如“我是一个动词”，“我是一个形容词”，“我是一个颜色”等信息。 
- **Q⋅K^T得到的就是一个大小为n\*n的相关性矩阵**
	- 可以理解为这个矩阵相当于每一个token通过K来回答其他token的Q的问题，每个位置上的乘积数值表示一个token基于问题的回答得到的对于另一个token的关注程度。 
	- 点乘(内积)有投影的意义，就是表示两组向量的相关关系
- **相关性矩阵需要除以 $\sqrt{d_k}$**
	- 原因是当$d_k$很大的时候，因为两个token的相关性是通过点积得到的，结果会趋向于极大或极小，这会导致softmax函数的结果偏向1和0，进入梯度非常小的区域，影响训练效率。 
- **应用softmax函数，转换为总和为1的概率分布**
	- 表示每个token对其他token的注意力权重。softmax的结果反映了每个token在当前上下文中对于其他token的关注程度。 
- **每个token的embedding向量通过与$W_V$相乘得到V**
	- 这个矩阵可以理解为，如果有一个token B和该token A相关，那么token B(的向量)该偏移多少得到以这个token A作为上下文的语义。 
- **将注意力权重矩阵与V矩阵相乘，得到每个token在上下文中的表示**
	- V之外的部分相当于“修改系数”
	- 这个结果相当于对V矩阵中的值进行加权平均，使每个token的最终表示既保留了它的原始含义，又结合了与其他token的上下文关系。于是，模型能够在当前token的基础上综合其他相关token的信息，得出一个更符合整体语境的表示。

<br>

【理解2】
参考
- [注意力机制的本质|Self-Attention|Transformer|QKV矩阵_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1dt4y1J7ov/)

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250224014228.png" alt="Transformer structure" width="80%">
<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250224014343.png" alt="Transformer structure" width="80%">

- 如果KQV相等就是Self-Attention，记相等矩阵为X，X\*$W_Q$=Q, $X^T$\*$W_Q$=Q，X\*$W_V$=V

<br>

##### 维度相关
【维度相同与否】
- Q和K的维度必须一样，因为要计算点积。
- K、V 矩阵的序列长度是一样的（加权求和），而 Q 矩阵的序列长度可以和前两者不一样。
	- 这种情况发生在：解码器部分的Encoder-Decoder Attention层中，Q 矩阵是来自解码器输出tgt，而 K、V 矩阵则是来自编码器最后的输出memory(编码信息矩阵C)。
		- 即: `tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,key_padding_mask=memory_key_padding_mask)[0]`
			- `tgt`：Target的缩写，指的是解码器的目标输出。在训练过程中，解码器试图生成与目标序列（即真实标签或目标序列）相匹配的输出。在机器翻译任务中，目标序列是翻译后的目标语言句子。 
			- `attn`是指注意力（Attention）机制。在代码中`attn_mask`和`key_padding_mask`是注意力掩码（Attention Mask），用于在计算注意力权重时屏蔽（mask）掉某些位置，例如填充（padding）位置或未来位置（防止在解码时看到未来的信息）。

<br>

##### $\sqrt{d_k}$相关
【为什么需要】<mark style="background: #FFF3A3A6;">为了防止内积过大，因此除以 dk 的平方根。</mark>

解释：
有两个最常用的attention函数：
- 加法attention: $s = A^T \tanh \left( qW + kU \right)$
	- 使用具有单个隐层的前馈网络计算，q和k维度不一致也可以进行；
- 点积（乘法）attention
	- 除了缩放因子 $\frac{1}{\sqrt{d_k}}$ 之外，点积Attention跟本文上述的算法一样（所以作者的注意力叫**缩放**点积注意力）。

虽然理论上点积Attention和加法Attention复杂度相似，但在实践中，点积Attention可以使用高度优化的矩阵乘法来实现，因此点积Attention计算更快、更节省空间。
当$d_k$的值比较小的时候，这两个机制的性能相差相近，当$d_k$比较大时，加法Attention比不带缩放的点积Attention性能好。
作者怀疑，维度$d_k$很大时，点积结果也变得很大（两极分化，有些dk大更靠近1，小的更靠近0），**将softmax函数推向具有极小梯度的区域**。为了抵消这种影响，将点积缩小$\frac{1}{\sqrt{d_k}}$倍。

<br>

### Muti-Head Attention

<br>

#### 背景
使用卷积神经网络CNN替换循环神经网络RNN，并行计算所有输入和输出位置的隐藏表示，是扩展神经GPU，ByteNet和ConvS2S的基础，因为这样可以减少时序计算。
但CNN对长序列难以建模（因为卷积计算时，卷积核/感受野比较小，如果序列很长，需要使用多层卷积才可以将两个比较远的位置关联起来）。
而若使用Transformer的注意力机制，每次（一层）就能看到序列中所有的位置，就不存在这个问题。
- 关联来自两个任意输入或输出位置的数据所需的操作数量，随着距离增长，对于ConvS2S呈线性，对于ByteNet呈对数，而对于Transformer是常数，因为一次就看到了。

但是卷积的好处是，输出可以有**多个通道**，**每个通道可以认为是识别不同的模式**。
作者也想得到这种多通道输出的效果，所以提出了Multi-Head Attention多头注意力机制。（投影h次模拟卷积多通道输出效果）

<br>

- 使用多头自注意力的好处：
	- **多语义匹配**：本身缩放点积注意力是没什么参数可以学习的，就是计算点积、softmax、加权和而已。但是使用Multi-head attention之后，投影到低维的权重矩阵$W^Q$,$W^K$,$W^V$ 是可以学习的，而且有h=8次学习机会。使得模型可以在不同语义空间下学到不同的的语义表示，也扩展了模型关注不同位置的能力。类似卷积中多通道的感觉。
		- 例如,“小明养了一只猫,它特别调皮可爱,他非常喜欢它”。“猫”从指代的角度看,与“它”的匹配度最高,但从属性的角度看,与“调皮”“可爱”的匹配度最高。标准的 Attention 模型无法处理这种多语义的情况。
	- **注意力结果互斥**：自注意力结果需要经过softmax归一化,导致自注意力结果之间是互斥的,无法同时关注多个输入。 使用多组自注意力模型产生多组不同的注意力结果，则不同组注意力模型可能关注到不同的输入，从而增强模型的表达能力。

<br>

#### 结构

<img src="https://raw.githubusercontent.com/yukinoshitasherry/qycf_picbed/main/img/Pasted%20image%2020250223223107.png" alt="Transformer structure" width="50%">

Multi-Head Attention 包含多个 Self-Attention 层，首先将输入**X**分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵**Z**。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵**Z**。

得到 8 个输出矩阵 Z1 到 Z8 之后，Multi-Head Attention 将它们拼接在一起 **(Concat)**，然后传入一个**Linear**层进行线性变换，得到 Multi-Head Attention 最终的输出**Z**。

<br>

### Why Attention

比较self-attention与循环层和卷积层的各个方面，我们使用self-attention是考虑到解决三个问题:
1. **每层计算的总复杂度**，越少越好。
2. **顺序计算量**，越少代表并行度越高。（顺序计算量就是下一步需要前面多少步计算完成）
3. **网络中长距离依赖之间的路径长度**：影响长距离依赖性能力的一个关键因素是前向和后向信号在网络中传播的路径长度。输入和输出序列中任意位置之间的这些路径越短，学习长距离依赖性就越容易。因此，作者还比较了由不同图层类型组成的网络中任意两个输入和输出位置之间的最大路径长度。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250223234940.png" alt="Transformer structure" width="70%">

- Attention：
	- **计算复杂度**：矩阵Q\*K，两个矩阵都是n行d列，所以相乘时复杂度是$O(n^2\cdot d)$, 其它还有一些计算量但影响不大
	- **顺序计算量**：矩阵里面并行度是很高的，整个计算主要就是矩阵乘法，所以可以认为顺序计算量就是O(1)
	- **最大路径长度**：也就是从一个点关联到任何一个点的路径长度。Attention是一次看到整个序列，所以只需要一次操作，复杂度为O(1)

<br>

### 代码
可以参考：
- `《The Annotated Transformer》翻译——注释和代码实现《Attention Is All You Need》`
[《The Annotated Transformer》翻译——注释和代码实现《Attention Is All You Need》-CSDN博客](https://blog.csdn.net/qq_56591814/article/details/120278245)
- [Transformer背景介绍及架构解析 - 知乎](https://zhuanlan.zhihu.com/p/572491345)

<br>

## 输出

<br>	

### Linear
线性层的主要作用是将解码器（Decoder）的输出映射到目标任务的输出形式。它通常使用一个全连接的神经网络层来实现，将解码器的输出向量转换为输出向量。这个输出向量的维度与词汇表的大小一致，其中每个元素代表了对应词汇的得分。

- 具体地，线性层通过对上一步的输出进行线性变换得到指定维度的输出，也就是转换维度的作用。这一步骤不使用激活函数，仅进行线性变换。

<br>

### SoftMax
Softmax层的作用是将线性层输出的logits向量转换为概率分布，以便选择最有可能的词汇。Softmax函数将logits向量中的每个得分转化为概率值，确保所有概率之和为1。

- 具体地，Softmax函数的计算公式如下：$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum\limits_{j=1}^{n} e^{z_j}}$
	- 其中，$z_i​$ 是logits向量中的第 i 个元素，表示第 i 个词汇的得分。Softmax层将这些得分转化为概率，最后通过选择概率最大的元素，来决定该时间步的输出词汇

<br>

# 实验
### 训练数据预处理
在标准的WMT 2014 English-German dataset上进行了训练，其中包含约450万个句子对。这些句子使用byte-pair编码进行编码，源语句和目标语句共享大约37000个词符的词汇表。对于英语-法语翻译，我们使用大得多的WMT 2014 English-French dataset，它包含3600万个句子，并将词符分成32000个word-piece词汇表。序列长度相近的句子一起进行批处理。每个训练批处理的句子对包含大约25000个源词符和25000个目标词符。

- bpe编码，是因为英语/德语中有很多ing之类的词根，又或者一个动词有几种形式。如果直接使用token进行表示，词表就太大了。bpe就可以把词根提取出来，这样词表会小很多，而且还能表示不同时态等等这些相关信息。
- 共用词表可以使编码器和解码器共用一个embedding，权重共享，模型更简单。

<br>

### 硬件与时间
在一台具有8个NVIDIA P100 GPU的机器上训练模型。使用本文描述的超参数的基础模型，每个训练步骤耗时约0.4秒。基础模型共训练了10万步或12小时。对于大型模型（在表3的底部描述），步长为1.0秒。大型模型接受了30万步（3.5天）的训练。

- 因为TPU非常适合做很大的矩阵乘法，所以后面Google都推荐自己的员工多使用TPU

<br>

### 优化器
使用Adam优化器。
Adam优化器（Adaptive Moment estimation，自适应矩估计）是一种用于训练神经网络的优化算法。它结合了Momentum（动量）和RMSProp（Ridged Stochastic Gradient，随机梯度的平方）两种优化算法的优点，旨在解决训练过程中学习率调整的问题，以加快收敛速度并提高模型性能。
Adam优化器的主要特点包括：
1. **动量估计**：Adam优化器通过计算梯度的一阶矩估计（即梯度的平均值）和二阶矩估计（即梯度的无偏方差）来动态调整每个参数的学习率。
2. **自适应性**：对于不同的参数，Adam优化器会计算不同的学习率，这使得它在处理稀疏梯度和非平稳目标时更加有效。
3. **计算效率**：Adam优化器在每次迭代中只需要计算一次梯度，这使得它在大规模数据集上非常高效。
4. **内存效率**：与一些其他优化器相比，Adam优化器需要更多的内存来存储梯度的一阶和二阶矩估计。
5. **参数少**：Adam优化器只有少数几个参数，如学习率、一阶矩估计的指数衰减率（β1​）、二阶矩估计的指数衰减率（β2​）和一个小的常数ϵ来防止除以零。

本文中：
$\beta_1 = 0.9, \beta_2 = 0.98 \text{ 并且 } \epsilon = 10^{-9}$。根据以下公式在训练过程中改变学习率：
$lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\ step\_num \cdot warmup\_steps^{-1.5})$
这对应于在第一次$warmup\_steps$步中线性地增加学习速率，并且随后将其与步数的平方根成比例地减小。使用$warmup\_steps=4000$。

<br>

### 正则化
训练期间我们采用两种正则化：
- Residual Dropout
	- 我们将dropout应用到每个子层的输出，在子层输出进入残差连接之前，和LayerNorm之前，都使用dropout。此外，在编码器和解码器中，token embedding+Positional Encoding时也使用了dropout。对于base模型，我们使用drop概率为 0.1。
- Label Smoothing
	- 在训练过程中使用的label smoothing的值为ϵls= 0.1。这让模型不易理解，因为模型学得更加不确定，但提高了准确性和BLEU得分。（softmax要逼近于1，其输入几乎要无穷大，这是很不合理的，会使模型训练困难）

<br>

### 模型配置
可以看到模型虽然比较复杂，但是没有多少超参数可以调，使得后面的人工作简单很多。

<br>

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/Pasted%20image%2020250223231759.png" alt="Transformer structure" width="70%">

<br>

# 评价
Transformer（attention机制）几乎能用在所有NLP任务上。
类似CNN对整个CV领域的革新（不需要那么多的特征提取或者模型建模，学会CNN就行了），Transformer也一样，不需要那么多的文本预处理，不需要为每个任务设计不同的架构。

而现在Transformer在CV、语音、video等领域也广泛使用，等于一个架构可以适用所有领域，任何一点突破在别的领域都能被使用，减少技术的应用时间。Transformer可以融合多模态的数据（文字、图片、语音等），大家都要同一个架构提取特征的话，可以都抽取到同一个语义空间，使得我们可以用文字、图片、语音等训练更大更好的模型。
<br>
虽然Transformer效果好，但是对它的理解还在初级阶段。
最新的一些结果表明，Attention在里面只是起到一个**聚合序列信息**的作用 ，后面的MLP/残差连接是缺一不可的，如果去掉的话，模型是基本训练不出什么的。

Attention不会对序列的顺序建模，为何能打败RNN？RNN可以显式地建模序列信息，不是应该比Attention更好？
现在大家觉得Attention使用了更广泛的归纳偏置，使得他能处理更一般化的信息；代价就是假设更一般，所以抓取数据信息能力变差，必须使用更大的模型和更多的数据才能训练到较好的效果。

