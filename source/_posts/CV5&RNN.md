---
title: CV(5)：RNN经典结构
date: 2024-02-17
categories:
  - 学AI/DS
tags:
  - CV
desc: CS231n Lec8&11 与序列建模笔记整合：RNN 前向与 BPTT、多种输入输出拓扑、语言模型与采样、梯度消失/爆炸、GRU/LSTM、双向与深层 RNN、图像标注与视觉注意力。
---

- 参考
    - <a href="https://www.showmeai.tech/article-detail/260">`showmeai-斯坦福CS231n教程`</a>



一、为何需要循环神经网络

1. 固定长度全连接网络难以处理变长序列，且无法在不同时间步共享「同一套」模式检测权重。循环神经网络（RNN）在每个时间步接收当前输入与上一隐状态，用共享权重刻画序列依赖。

2. 记号：输入序列 $x^{\langle 1\rangle},\ldots,x^{\langle T_x\rangle}$，隐状态 $a^{\langle t\rangle}$（或记为 $h^{\langle t\rangle}$），输出 $\hat{y}^{\langle t\rangle}$。样本 $i$ 可写作 $X^{(i)\langle t\rangle}$ 等。

<br>

二、Vanilla RNN 前向传播

1. 常用形式（合并 $W_{aa}$ 与 $W_{ax}$）：

$$a^{\langle t\rangle}=g_1\left(W_a\begin{bmatrix}a^{\langle t-1\rangle}\\ x^{\langle t\rangle}\end{bmatrix}+b_a\right),\quad \hat{y}^{\langle t\rangle}=g_2\left(W_{ya}a^{\langle t\rangle}+b_y\right).$$

$g_1$ 常为 $\tanh$ 或 ReLU；$g_2$ 依任务为 sigmoid、softmax 等。

2. 权重在时间上共享：同一组 $W_a,W_{ya}$ 用于所有 $t$，参数量与序列长度无关。

<br>

三、沿时间反向传播（BPTT）

1. 定义逐时刻损失 $L^{\langle t\rangle}$（如交叉熵），总损失 $J=\sum_t L^{\langle t\rangle}$。反向时梯度从 $t=T$ 向 $t=1$ 回传，链式法则穿过时间步时反复乘以 $\partial a^{\langle t\rangle}/\partial a^{\langle t-1\rangle}$，其中含 $W_{aa}$ 与 $\tanh$ 导数。

> [!INFO]+ 梯度随时间连乘的直观
> 若 $\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}}$ 谱范数长期小于 1，梯度指数衰减（消失）；长期大于 1 则爆炸。长序列依赖（如主谓一致跨很多词）对 vanilla RNN 困难，故引入门控结构或梯度裁剪。

<br>

2. **梯度裁剪**：当梯度范数超过阈值 $\tau$ 时整体缩放，缓解爆炸。

<br>

四、多种输入输出拓扑

| 类型 | 条件 | 典型任务 |
| :--- | :--- | :--- |
| 一对一 | $T_x=T_y=1$ | 单帧分类 |
| 一对多 | $T_x=1,T_y>1$ | 看图说话（先 CNN 得图像向量） |
| 多对一 | $T_x>1,T_y=1$ | 情感分类、视频级标签 |
| 多对多（等长） | $T_x=T_y$ | 序列标注、逐帧预测 |
| 多对多（不等长） | $T_x\neq T_y$ | 机器翻译（常配合 encoder–decoder） |

<br>

五、语言模型与序列生成

1. 目标：估计 $P(w_1,\ldots,w_T)$，常分解为 $P(w_1)\prod_{t=2}^{T}P(w_t\mid w_1,\ldots,w_{t-1})$。用 RNN 在每个时间步输出词表上的 softmax，训练时 teacher forcing：输入为前缀真实词，损失为各步交叉熵之和。

2. **采样生成**：从 $t=1$ 起依 softmax 采样一词，作为下一步输入，直到采样到句末符号 EOS。

<br>

六、GRU

1. 更新门 $\Gamma_u$、相关门 $\Gamma_r$ 控制旧记忆与候选更新的混合：

$$\tilde{c}^{\langle t\rangle}=\tanh\left(W_c\left[\Gamma_r\odot c^{\langle t-1\rangle},\,x^{\langle t\rangle}\right]+b_c\right),\quad c^{\langle t\rangle}=(1-\Gamma_u)\odot c^{\langle t-1\rangle}+\Gamma_u\odot\tilde{c}^{\langle t\rangle}.$$

2. 当 $\Gamma_u\approx 0$ 时，记忆可跨多步保持，缓解长程依赖。

<br>

七、LSTM

1. 遗忘门 $\Gamma_f$、输入门 $\Gamma_u$、输出门 $\Gamma_o$ 与细胞状态 $c^{\langle t\rangle}$：

$$c^{\langle t\rangle}=\Gamma_u^{\langle t\rangle}\odot\tilde{c}^{\langle t\rangle}+\Gamma_f^{\langle t\rangle}\odot c^{\langle t-1\rangle},\quad a^{\langle t\rangle}=\Gamma_o^{\langle t\rangle}\odot\tanh(c^{\langle t\rangle}).$$

2. 相对 GRU 更灵活；实践中常配合 dropout、层归一化等稳定训练。

<br>

八、双向与深层 RNN

1. **BRNN**：前向隐状态 $\overrightarrow{a}^{\langle t\rangle}$ 与后向 $\overleftarrow{a}^{\langle t\rangle}$ 拼接后再接输出层，适合标注类任务；推理需完整序列。

2. **DRNN**：同一时刻上堆叠多层 RNN，提升表达能力，代价是训练成本与过拟合风险。

<br>

九、视觉中的 RNN：图像标注与注意力

1. **Image Captioning**：CNN（如 AlexNet/VGG）提取图像向量，作为 RNN 第一时刻输入或初始隐状态，逐词生成描述。

2. **软注意力**：每步对空间位置产生权重分布，对 CNN 特征图做加权求和得到上下文向量，再与 RNN 结合；权重可微，端到端训练。**硬注意力**每次选一个区域，离散决策，优化更难。

3. **VQA** 等任务常将图像特征与问题编码后，用 RNN/Transformer 融合推理。

<br>

十、重参数化与工程提示

1. VAE 中常用的 **reparameterization** 思想类似：把随机性写成确定性函数加噪声，使梯度穿过采样（本笔记以生成模型篇为准展开）。

2. 词表示从 one-hot 到 **词嵌入** 可降低维数并携带语义，预训练嵌入常用于小数据任务。

<br>

十一、推导与掌握要点（增补）

1. **展开图（unfolded）视角**  
   将 RNN 在时间上展开，等价于同一组权重 $W_a,W_{ya}$ 在 $T$ 个「层」上共享的前馈网络。BPTT 即在该展开图上做标准反向传播，梯度路径可沿时间反向流动。

<br>

2. **BPTT：隐状态梯度的连乘形式**  
   记 $\delta^{(t)}=\frac{\partial J}{\partial a^{\langle t\rangle}}$。则
   $$\delta^{(t)}=\frac{\partial J}{\partial a^{\langle t\rangle}}\Big|_{\mathrm{direct}}+\left(\frac{\partial a^{\langle t+1\rangle}}{\partial a^{\langle t\rangle}}\right)^{\mathsf T}\delta^{(t+1)}.$$
   其中 $\frac{\partial a^{\langle t+1\rangle}}{\partial a^{\langle t\rangle}}=W_{aa}^{\mathsf T}\,\mathrm{diag}(g_1'(\cdot))$（以 $\tanh$ 为例）。从 $T$ 回传到 $1$ 时反复左乘此类雅可比，**谱范数**若长期 $<1$ 则梯度消失，$>1$ 则爆炸——与「十、」中直觉一致。

<br>

3. **LSTM 遗忘/输入门的作用（记忆保持）**  
   细胞状态更新 $c^{\langle t\rangle}=\Gamma_f^{\langle t\rangle}\odot c^{\langle t-1\rangle}+\Gamma_u^{\langle t\rangle}\odot\tilde{c}^{\langle t\rangle}$。若 $\Gamma_f\approx 1$ 且 $\Gamma_u\approx 0$，则 $c^{\langle t\rangle}\approx c^{\langle t-1\rangle}$，信息可跨多步无损流过加法通道（乘法门控接近恒等），缓解长程梯度衰减。

> [!INFO]+ GRU 与 LSTM 的选型
> GRU 参数略少、结构简单；LSTM 表达能力与稳定性在极长序列上常略占优。视觉序列（caption）二者均有使用；工业 NLP 早期多用 LSTM，现多被 Transformer 替代，但门控思想仍延续。

<br>

4. **语言模型困惑度（perplexity）**  
   若平均交叉熵为 $H$，则困惑度 $\mathrm{PPL}=2^{H}$（以 2 为底）或 $e^{H}$（自然底），表示模型在下一步预测上「平均等价于从多少个等概率词中选」。**越低越好**。

<br>

5. **软注意力的概率解释**  
   对位置 $i=1,\ldots,L$ 的得分 $s_i$（由当前隐状态与第 $i$ 块图像特征算得），$\alpha_i=\mathrm{softmax}(s)_i$。上下文向量 $\mathbf{c}=\sum_i \alpha_i \mathbf{f}_i$ 为凸组合，$\sum_i\alpha_i=1$，可微，便于端到端训练。硬注意力从多项分布采样位置，期望梯度需 REINFORCE 或近似。

<br>

6. **Encoder–Decoder 与视觉**  
   图像编码器输出固定维向量或空间特征图；解码器为 RNN，每步生成一词，输入含上一词嵌入与（可选）注意力上下文。损失为各步交叉熵之和。beam search 解码可提高输出质量但增加推理耗时。

<br>

十二、逻辑脉络（如何把前面几节串起来）

1. **主线**：变长序列 $\rightarrow$ **权值时间共享** 的 RNN 前向 $\rightarrow$ **BPTT** 理解梯度为何难传 $\rightarrow$ **门控（GRU/LSTM）** 缓解长依赖 $\rightarrow$ 根据任务选 **拓扑**（标注用多对多，摘要/翻译用 encoder–decoder）$\rightarrow$ 视觉里 CNN 提供 **固定向量或特征图**，RNN 负责 **语言侧序列建模**，注意力在二者之间做 **软对齐**。

2. **与 CV4/CV6 的衔接**：Captioning/VQA 不是「另起炉灶」，而是 **CNN 特征 + RNN 序列模型**；注意力权重可视作在特征图上的 **软池化**，梯度仍可回传到 CNN（端到端微调时常用较小学习率或先冻结 CNN 再解冻）。

<br>

十三、分步例题（便于自检）

> [!EXAMPLE]+ 例1：两步前向（展示权值共享，符号与第二节一致）
> 设隐层维数为 1，$g_1=\tanh$，且为简化令 $b_a=b_y=0$，标量输入 $x^{\langle 1\rangle}=1$，$x^{\langle 2\rangle}=-1$，初值 $a^{\langle 0\rangle}=0$，并设合并权重 $W_a=[w_{aa},w_{ax}]=[0.5,\,1]$。则
> $$a^{\langle 1\rangle}=\tanh(0.5\cdot 0+1\cdot 1)=\tanh(1),\quad a^{\langle 2\rangle}=\tanh(0.5\,a^{\langle 1\rangle}+1\cdot(-1)).$$
> 同一条 $(w_{aa},w_{ax})$ 用于 $t=1,2$ 两步，这就是 **时间上的参数共享**；具体数值可用计算器验证 $\tanh$ 大小。

<br>

> [!EXAMPLE]+ 例2：困惑度与交叉熵（与第十一节 4 对照）
> 若某语言模型在大量句子上平均 **负对数似然**（自然底）为 $\ln 2$ nats/词，则等价平均交叉熵 $H=\ln 2$，困惑度 $\mathrm{PPL}=e^{H}=e^{\ln 2}=2$。直观：模型下一步预测等价于在 **2 个** 等概率候选中均匀猜测。困惑度越低，模型对下一词分布越「尖锐」。

<br>

> [!EXAMPLE]+ 例3：软注意力权重（与第十一节 5 对照）
> 设某一时刻对 3 个位置的未归一化得分 $s_1=2,\,s_2=1,\,s_3=0$，则
> $$\alpha_i=\frac{e^{s_i}}{\sum_j e^{s_j}},\quad i=1,2,3.$$
> 分母 $e^2+e^1+e^0$，故 $\alpha_1$ 最大且三者之和为 1。上下文向量 $\mathbf{c}=\sum_i \alpha_i\mathbf{f}_i$ 是特征向量的 **凸组合**，这是「软」的含义。

<br>
