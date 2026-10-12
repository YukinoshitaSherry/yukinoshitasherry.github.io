---
title: 斯坦福CS274：计算分子生物学的表示与算法
date: 2026-09-21
categories:
- 上斯坦福
tags:
- CompBio
desc: Stanford CS274 详细课程笔记：从序列比对、HMM 与系统发育，到蛋白质结构、分子动力学、转录组、化学信息学和网络生物学。
---

Stanford CS274 的全名是 *Representations and Algorithms for Computational Molecular Biology*（计算分子生物学的表示与算法），与 BIOE 214、BMDS 214、GENE 214 交叉挂牌（cross-listed），由 Russ Altman 主讲。它是一门研究生层次的计算分子生物学（computational molecular biology）导论，先修课为 CS106B，课程包含较多 Python 编程。

- [Stanford Bulletin：CS274](https://bulletin.stanford.edu/courses/1410302)

这门课的主线是“生物对象怎样被表示”。同一个蛋白质，可以是一串氨基酸字母，可以是一组原子坐标，也可以是相互作用网络里的一个节点；表示选定之后，能用的算法和能回答的问题也随之确定。课程沿着这条线依次讨论序列比对、概率模型、系统发育、结构计算、分子动力学、组学分析和网络生物学，并在每一处同时考察三件事：算法的复杂度、背后的统计假设，以及结果的生物学解释。

<br>

> [!INFO]+ 版本范围
>
> 当前 Stanford Bulletin 明确列出的主题包括：双序列与多序列的序列/结构比对、系统发育树、隐马尔可夫模型、蛋白质结构计算、蛋白质结构预测、蛋白质语言模型、分子动力学与能量最小化、三维生物数据统计、数据整合、知识表示与受控术语、转录分析、化学信息学、药物遗传学和网络生物学。
>
> 2020 年课程 staff 页面还列出了动态规划、结构叠合、距离信息、1D/3D motif、统计特征检测以及 RNA 序列与结构。文中凡涉及当前目录、历史课程说明或补充材料的内容，都会分别注明出处层级。

<br>

# 课程总览与预备知识

计算分子生物学的难点不在于某一个算法，而在于一条很长的链条：实验产生数据，数据被编码成某种表示，在表示上定义一个“好坏”的分数，用算法找到分数最优的答案，再判断这个分数在统计上是否可信，最后才谈得上生物学含义。链条中任何一环出错，后面的结论都会失真。例如比对算法本身完全正确，但若零模型（null model）选得不对，一个随机出现的高分就会被误读为同源证据。

这一章先给出全课的地图，说明序列、结构、功能与系统三大部分各自讨论什么、彼此怎样依赖；然后补齐后文反复用到的数学工具（log-odds、动态规划、max 与 sum、生成与判别模型）、复杂度与数值稳定性的估算方法，以及课程编程所需的 Python 基础。后面各章的具体算法都可以看成这些工具在不同表示上的实例。

## 课程内容地图

课程内容大致分为三部分，每一部分都从一种基本表示出发。

- **序列部分**以字符串为基本表示，内容包括：
  - 字符串索引与精确匹配；
  - 双序列比对、数据库搜索与多序列比对；
  - motif、HMM 与序列家族模型；
  - 系统发育与 RNA 结构。
- **结构部分**以三维坐标为基本表示，内容包括：
  - 三维坐标、距离与接触（contact）表示；
  - 结构叠合（superposition）、对应搜索与三维 motif；
  - 结构预测、能量函数与分子动力学；
  - 分子动力学轨迹的对齐、降维、聚类与状态模型。
- **功能与系统部分**以矩阵、本体和网络为基本表示，内容包括：
  - 表达矩阵、受控术语与本体（ontology）；
  - GWAS、变异效应与患者级优先级排序；
  - 化学结构、药物遗传学与生物网络；
  - 文献抽取、证据整合与可复现工作流。

### 统一的分析逻辑链

各部分都按同一条逻辑链记录。读后面任何一章时，都可以问：这里的生物问题是什么？用什么表示？目标函数是什么？用什么算法优化？分数如何校准？最终能支持什么生物学结论？

$$
\text{生物问题}
\rightarrow\text{表示}
\rightarrow\text{目标函数}
\rightarrow\text{算法}
\rightarrow\text{统计校准}
\rightarrow\text{生物学解释}
$$

其中“统计校准”一环最容易被忽略，下面是课程中的三个典型例子：

- BLAST 的原始分数本身没有绝对含义，需要结合数据库大小和零模型才能换算成“随机出现这么高分的期望次数”（E-value）；同一个分数在小数据库里可能显著，在大数据库里可能平平无奇。
- 预测结构（如 AlphaFold 模型）要同时查看局部置信度与相对位置置信度：前者说明单个残基周围的局部构象是否可靠，后者说明两个结构域之间的相对摆放是否可靠，局部置信度高并不保证整体排布正确。
- 差异表达的结论依赖实验设计（重复数、批次、配对）和多重检验校正；同时检验两万个基因时，即使没有任何真实差异，按 $p<0.05$ 也会得到约一千个“显著”基因。

课程的实现要求也分两个层次：

- 能够亲手实现 Needleman–Wunsch、Smith–Waterman、Viterbi、Kabsch 等基础算法，理解每个状态和边界条件；
- 对于成熟工具，能够说明它的输入、目标函数、所做的近似、输出以及解释边界，而不是把它当作黑箱。

### 从测量走向决策

课程复习材料给出了另一条主线，强调从实验测量一路走到功能、疾病和药物层面的决策：

$$
\text{DNA/RNA/protein 与实验观测}
\rightarrow\text{序列、结构和表达表示}
\rightarrow\text{比对、预测与统计检验}
\rightarrow\text{功能、疾病和药物解释}
$$

这条主线包含两个关键转换。

- **从测量到计算对象。** 需要回答数据是如何产生的、被怎样编码、表示保留了哪些信息又丢掉了哪些信息。这一问题在“生物表示”与“数据与数据库”两章展开。
- **从模型分数到证据。** 需要回答一个分数能够支持哪一级结论。GWAS 的关联信号、变异的有害性（deleteriousness）、某位患者的致病性判断与药物处方建议分属不同的证据层级：一个位点与疾病统计相关，不代表该变异有害；一个变异在功能上有害，也不代表它导致了眼前这位患者的疾病。同样，GSEA 给出的基因集富集（gene-set enrichment）只说明某组基因的表达整体偏移，并不等于该 pathway 已被因果激活。

### 核心章节之间的依赖

三条基本表示各自衍生出一串依次依赖的方法。序列一侧从双序列动态规划出发，扩展到多序列比对与 profile，再到 motif 与 HMM，最后用于系统发育与 RNA；结构一侧从坐标出发，经距离、刚体叠合、结构对应到三维 motif 与统计检测；图一侧从局部指纹与相似度出发，走向聚类、信息传播与模块发现：

$$
\begin{aligned}
\text{序列字符串}
&\rightarrow\text{pairwise DP}
\rightarrow\text{MSA/profile}
\rightarrow\text{motif 与 HMM}
\rightarrow\text{系统发育、RNA},\\
\text{三维坐标}
&\rightarrow\text{距离信息}
\rightarrow\text{刚体叠合}
\rightarrow\text{结构对应}
\rightarrow\text{3D motif 与统计检测},\\
\text{分子图和生物网络}
&\rightarrow\text{局部指纹/相似度}
\rightarrow\text{聚类、传播与模块}.
\end{aligned}
$$

<br>

## 数学预备：贯穿全课的四类工具

课程中看似各不相同的算法（替换矩阵打分、PWM 扫描、HMM 解码、系统发育似然）反复使用同样四类数学对象。先把它们讲清楚，后面每遇到一个新模型，只需识别它属于哪一类、在哪一点上做了特殊化。

### log-odds：目标模型相对背景模型的证据

许多问题都可以归结为：一条观测 $x$（例如一段序列）更像是由“目标模型” $M_1$（某个蛋白家族、某个结合位点）产生，还是由“背景模型” $M_0$（随机序列）产生？衡量这种相对证据的标准量是对数几率比（log-odds）：

$$
\log\frac{P(x\mid M_1)}{P(x\mid M_0)}
$$

这里 $P(x\mid M)$ 是模型 $M$ 产生观测 $x$ 的概率（似然）。分数大于 0 表示观测更支持 $M_1$，小于 0 表示更支持背景，绝对值越大证据越强。

为什么几乎所有打分都是“加和”形式？原因在于条件独立分解。若模型假设观测由若干条件独立的因子组成，例如 $x=x_1\cdots x_n$ 的各位置独立，那么联合概率是乘积：

$$
P(x\mid M)=\prod_{i=1}^{n}P(x_i\mid M).
$$

代入 log-odds，对数把乘积变成和：

$$
\log\frac{P(x\mid M_1)}{P(x\mid M_0)}
=\sum_{i=1}^{n}\log\frac{P(x_i\mid M_1)}{P(x_i\mid M_0)}.
$$

每个位置贡献一个独立的分数项。这正是替换矩阵（每对对齐残基一个分数）、PWM（每个位置每种碱基一个分数）、HMM 路径分数（每次转移与发射一个分数）以及许多线性分类器都采用加和形式的原因。反过来也要注意，加和打分隐含了“各项独立”的假设；当相邻位置存在相关（如 RNA 碱基配对、蛋白质中协同突变的残基），简单加和就会系统性地失真。

### 动态规划：把指数枚举压缩为子问题复用

两条序列之间可能的比对数目随长度指数增长，逐一枚举不可行。动态规划（dynamic programming, DP）的思路是：最优解由子问题的最优解拼成，而且不同的大问题会反复用到同一批子问题，因此每个子问题只算一次、存下来复用即可。

DP 能够成立需要两个条件：

- **最优子结构（optimal substructure）**：全局最优解的局部也是相应子问题的最优解。
- **重叠子问题（overlapping subproblems）**：不同的决策路径会到达相同的子问题，因此存储有意义。

设计 DP 时最关键的是状态定义：状态必须包含足够的历史，使得未来的决策只依赖当前状态，而不再依赖更早的细节。这一点在序列比对中有具体体现：

- 在线性 gap 罚分（每个 gap 字符罚固定分数）下，下一步的代价与“上一步是不是 gap”无关，因此每个网格位置 $(i,j)$ 只需保存一个最优值。
- 在仿射 gap 罚分（打开 gap 罚得多、延伸 gap 罚得少）下，下一步的代价取决于当前是否已经处在一个 gap 中，因此必须区分三类末端状态：match（$x_i$ 与 $y_j$ 对齐）、gap-in-$x$ 和 gap-in-$y$，每格保存三个值。状态若只存一个值，就会丢失“是否正在延伸 gap”的历史，得到错误的最优解。

### 最大化与求和：两个不同的概率问题

在隐变量模型（如 HMM）中，同一条观测序列可以由许多条隐状态路径产生。这时有两个不同的问题：

- **Viterbi 算法用 max**，找出单条最可能的路径，回答“最可能的解释是什么”。
- **Forward 算法用 sum**，把所有路径的概率加起来，得到观测的总似然 $P(x\mid M)$，回答“模型整体上有多大可能产生这条观测”。

两者的递推形式几乎相同，只是把 $\max$ 换成 $\sum$。需要注意的是，当可能路径非常多、概率分散时，最可能的那一条路径本身的概率可能很低，因此不能用 Viterbi 路径概率代替总似然。比较两个模型、或计算 log-odds 时，应使用 Forward 给出的总似然。

### 生成模型与判别模型：建模目标不同

- **生成模型（generative model）** 描述数据是如何产生的，即建模 $P(x\mid\text{模型})$ 或联合分布。HMM、系统发育中的替换模型、RNA-seq 中的负二项计数模型（negative-binomial count model）都属于这一类。
- **判别模型（discriminative model）** 直接描述给定数据时标签的条件概率 $P(y\mid x)$，例如 logistic regression。

前者可以把机制假设（突变如何发生、读数如何被采样）写进模型，便于解释和模拟；后者不必刻画数据的完整分布，通常更直接地服务于预测。选择哪一类，取决于问题是“理解过程”还是“做准预测”。

<br>

## 复杂度：哪个维度在增长

生物数据的规模差异极大：一个蛋白几百个残基，一个人类基因组约 $3\times10^9$ 个碱基，一个数据库可能有上亿条序列。因此写出 Big-O 之后，还要追问输入中的哪个维度在增长，以及哪部分成本只付一次、哪部分每次查询都要付。

### 时间复杂度的典型例子

- 长度分别为 $n,m$ 的双序列比对（pairwise alignment）要填一张 $(n+1)\times(m+1)$ 的表，每格常数时间，总时间为 $O(nm)$。
- 含 $K$ 个状态、长度为 $T$ 的稠密 HMM（任意两状态间都可转移），每个时间步、每个状态都要考察 $K$ 个前驱，总时间为 $O(TK^2)$。
- 对 $s$ 条长度约为 $n$ 的序列做精确的多序列比对，需要 $s$ 维 DP 表，共有 $O(n^s)$ 个格点。每格的前驱是“每条序列要么前进一位要么插 gap”的所有组合，去掉“全不前进”的情况共 $2^s-1$ 个非空移动，因此朴素时间为 $O((2^s-1)n^s)$，空间为 $O(n^s)$。例如 $s=3$、$n=300$ 时已有约 $2.7\times10^7$ 个格点、每格 7 个移动；序列再多几条就完全不可行，这也是多序列比对必须依赖启发式的原因。
- 数据库搜索还要区分两类成本：一次性建索引的成本，和每次 query 的成本。索引贵但可摊销，查询便宜才是关键。

### 空间复杂度同样重要

空间往往比时间更早成为瓶颈。两条长 $10^5$ 的序列比对，DP 表有 $10^5\times10^5=10^{10}$ 格，即使每格只占 4 bytes，也需要 $4\times10^{10}$ bytes，约 40 GB，即约 37.3 GiB（$4\times10^{10}/2^{30}\approx37.3$）。时间上可以等，但内存放不下就根本无法运行。因此生物规模的数据通常需要以下技巧：

- **带状比对（banded alignment）**：只计算主对角线附近一条带，适用于两条序列已知高度相似的情况；
- **线性空间回溯（linear-space traceback）**：如 Hirschberg 分治，只保留两行即可恢复最优比对，空间降为 $O(\min(n,m))$，时间仍为 $O(nm)$；
- **稀疏图**：只保存实际存在的边或候选匹配；
- **分块计算**：把大问题切块，逐块读入和写出。

<br>

## 数值计算：浮点下溢与长度归一化

概率动态规划不仅要写出正确的递推，还要能在计算机上算得出来。一条长度 1000 的序列，每个位置概率约 0.25，联合概率约为 $0.25^{1000}\approx10^{-602}$，远低于双精度浮点能表示的最小正数（约 $10^{-308}$），直接相乘会下溢为 0，此后所有比较都失去意义。

Forward–backward 算法有两种常见对策：

- **缩放（scaling factor）**：在每个时间步把所有状态的前向变量除以它们的和，并记录这个缩放因子；最终的 log-likelihood 等于各步缩放因子的对数之和。
- **全程在 log-space 计算**：乘法变成加法；而 Forward 中的求和需要用 log-sum-exp 技巧，先提出最大值再求和，避免指数运算溢出或下溢：

$$
\log\sum_k e^{a_k}=m+\log\sum_k e^{a_k-m},\qquad m=\max_k a_k.
$$

提出 $m$ 之后，每一项 $e^{a_k-m}\le1$，且至少有一项等于 1，因此求和既不会上溢也不会全为 0。Viterbi 只用 max，在 log-space 下直接相加取最大即可，不需要 log-sum-exp。

此外，比较 likelihood 时还要说明是否按序列长度归一化。序列越长，乘进去的概率因子越多，原始 log-likelihood 自然越负，因此不同长度序列的原始 log-likelihood 通常不可直接比较。常见做法是使用每个位置的平均 log-likelihood，或者使用相对于背景模型的 log-odds（背景模型同样随长度累积，可以抵消长度效应）。

<br>

## Python 实现基础

课程编程把 Python 当作表示和验证算法的工具。重点不在语法，而在于把状态、边界条件与输入输出明确写进程序：DP 表每一格代表什么、第 0 行第 0 列如何初始化、函数接受什么 shape 的数组、返回什么。下面列出课程作业中最常出错的几类问题。

### NumPy：dtype、向量化与 shape

NumPy 数组要求同质 dtype（所有元素同一类型），这使底层可以用连续内存和编译代码处理。数值内核应优先使用 broadcasting、切片和线性代数操作，避免在大数组上逐元素执行 Python `for` loop。需要明确的是，向量化不会改变渐近复杂度（$O(nm)$ 仍是 $O(nm)$），它的作用是把循环从解释器移到底层编译代码，常数因子往往能快一两个数量级。

NumPy 的整数索引会消去一个维度，切片则通常保留该维度。例如二维数组 `a` 中：

- `a[0, 0]` 是 scalar；
- `a[:, 0]` 的 shape 为 `(n,)`，是一维数组；
- `a[0:1, 0:1]` 的 shape 为 `(1, 1)`，仍是二维数组。

这个区别在动态规划中尤其危险：若把行向量 `(1, m)`、列向量 `(n, 1)` 与一维数组 `(n,)` 混淆，broadcasting 不会报错，而是悄悄生成一个 `(n, m)` 或 `(n, n)` 的数组，得到数值上看似合理但语义错误的结果。稳妥的做法是在关键位置写 `assert x.shape == (...)`，并在注释里记录每个数组的 shape。

### 脚本、类与运算符的常见坑

- **输入方式**：可执行脚本应通过 `sys.argv` 或 `argparse` 接收输入，不应把文件路径与参数硬编码在 notebook 里，否则换一台机器或换一个数据集就无法复现。
- **`self`**：类方法的第一个参数 `self` 指当前实例，方法通过它读写实例的属性。
- **可变默认参数**：默认参数在函数定义时只求值一次。若写成 `def f(x, acc=[])`，多次调用会共享同一个列表，状态在调用间累积；应写成 `acc=None`，在函数体内再新建列表。
- **除法**：`/` 总是浮点除法；`//` 是向下取整的 floor division，对负数并非简单截断。例如 `-7 // 2` 的结果是 `-4`，而截断取整 `int(-7 / 2)` 是 `-3`。在用除法计算下标或区间中点时，这一差别会导致 off-by-one 错误。

### 递归、memoization 与 bottom-up DP

这三者是同一张子问题依赖图（有向无环图）的不同求值方式：

- **递归实现**先写 base case，再保证每次递归调用都更接近 base case，否则不会终止；
- **memoization**（自顶向下加缓存）只计算从原问题出发实际可达的状态，适合状态空间大但真正用到的很少的情况；
- **bottom-up** 则按拓扑顺序填表，保证计算某格时它依赖的格都已算好，没有递归深度限制，也更容易做空间压缩（如只保留两行）。

图遍历中，DFS（深度优先）使用 stack 或递归，BFS（广度优先）使用 queue。二者访问节点的顺序不同，保证也不同：在无权图上，BFS 第一次到达某节点时走的就是最短路，DFS 则没有这种保证。

### 环境与代码规范

Conda 或其他隔离环境负责固定解释器版本与依赖版本；`PYTHONPATH` 只影响模块查找路径，不能替代环境管理。提交代码应遵守 PEP 8，并为函数写明参数、返回值、shape、单位和异常条件。行内注释应解释状态定义、边界和不变量（例如“`dp[i][j]` 是 `x[:i]` 与 `y[:j]` 的最优比对分数”），而不是逐字翻译代码。

> [!NOTE]+ 小结
>
> - 全课按“生物问题 → 表示 → 目标函数 → 算法 → 统计校准 → 生物学解释”组织，统计校准决定分数能支持哪一级结论。
> - log-odds 的加和形式来自条件独立分解；DP 的关键是状态包含足够历史；Viterbi 用 max、Forward 用 sum，二者回答不同问题。
> - 复杂度要指明增长的维度，空间常常先于时间成为瓶颈：$10^5\times10^5$ 的 DP 表约 37.3 GiB。
> - 概率 DP 用 scaling 或 log-sum-exp 防止下溢；不同长度序列的 log-likelihood 需归一化后再比较。
> - NumPy 中整数索引降维、切片保维，shape 混淆会被 broadcasting 掩盖。

<br>

# 生物表示：从分子到可计算对象

计算方法不能直接处理“生命活动”本身，只能处理被编码后的对象。DNA 可以表示为带方向的字符串，基因可以表示为参考基因组上的一个区间，蛋白质可以表示为氨基酸序列或一组原子坐标，表达实验可以表示为“样本 × 基因”的矩阵。每一种表示都保留一部分信息，同时丢弃其余信息：序列字符串保留了残基顺序，却不包含细胞类型、化学修饰和空间构象。

这一章建立后续算法共同使用的基本约定：字母表、链方向、互补关系、坐标系统、变异类型和实验观测的层级。输入通常是实验数据或数据库记录，输出是经过规范化的序列、区间、类别或数值张量。这些约定看似琐碎，但一旦混淆 $5'$/$3'$ 方向、正负链、0-based/1-based 坐标或 transcript isoform，错误会悄无声息地传播到比对、变异注释和结构映射，到最后很难追查。

本章与相邻各章的分工是：生物表示回答“对象在计算机中是什么”；下一章的数据格式回答“这些对象如何存储和交换”；之后的字符串、图、概率模型和坐标算法回答“在这种表示上能计算什么”。

## 中心法则与序列表示

### 第一层抽象：有限字母表上的字符串

最基本的表示是把生物大分子看成有限字母表 $\Sigma$ 上的字符串：

$$
x=x_1x_2\cdots x_n,\qquad x_i\in\Sigma
$$

其中 $n$ 是序列长度，$x_i$ 是第 $i$ 个位置的字符。

- DNA 由四种碱基组成，$\Sigma=\{A,C,G,T\}$；
- RNA 以 U 代替 T；
- 蛋白质通常以 20 种标准氨基酸表示，字母表比 DNA 大得多，这也是蛋白质比对对远缘同源更敏感的原因之一（每个位置携带更多信息）。

这种抽象丢弃了细胞环境、空间构象、化学修饰和实验误差，却换来了巨大的好处：字符串上的索引、匹配、动态规划和概率建模都有成熟的理论和高效算法。后面大半门课都建立在这一抽象之上。

### 核酸化学与链方向

字符串有“左右”，DNA 分子则有“方向”，二者之间的对应来自核酸的化学结构。

- 核苷酸由三部分组成：含氮碱基、五碳糖（脱氧核糖或核糖）和磷酸。
- 相邻核苷酸以磷酸二酯键（phosphodiester bond）连接，连接方式是前一个核苷酸糖环的 $3'$ 碳与后一个的 $5'$ 碳经磷酸相连，因此整条链具有从 $5'$ 端到 $3'$ 端的方向。
- 两条链之间的碱基配对：A–T 形成两个氢键，G–C 形成三个氢键。
- 双螺旋的稳定性还受到碱基堆积（base stacking）、离子与溶剂的影响，而且堆积作用往往贡献更大，因此不能只按氢键数解释 GC 含量高的片段为何更稳定。
- 常见的 B-DNA 和 A-DNA 是右手双螺旋，Z-DNA 则是左手双螺旋，因此“DNA 一定是右手螺旋”的说法并不成立。

### 互补、反向互补、转录与翻译

DNA 双链方向相反（反向平行）且碱基互补。约定正向链按 $5'\rightarrow3'$ 书写。要得到另一条链同样按 $5'\rightarrow3'$ 书写的序列，需要做反向互补（reverse complement）：先按 $A\leftrightarrow T$、$C\leftrightarrow G$ 逐字符取互补，再把整个字符串反转。例如 `5'-ATGC-3'` 的互补为 `TACG`，反转后得到 `5'-GCAT-3'`。只取互补不反转，得到的是按 $3'\rightarrow5'$ 读的另一条链，若当作 $5'\rightarrow3'$ 使用就会出错。

转录（transcription）以 DNA 为模板产生 RNA。翻译（translation）则以三个碱基一组的密码子（codon）读取开放阅读框（open reading frame, ORF），每个 codon 对应一个氨基酸或终止信号。4 种碱基组成 $4^3=64$ 个 codon：在标准遗传密码中，61 个 sense codon 编码 20 种标准氨基酸，另有 3 个 stop codon。因为 61 远大于 20，多个 codon 可编码同一氨基酸，即密码子简并性。线粒体等遗传密码存在例外，例如人类线粒体中 UGA 编码色氨酸而非终止。

### 复制、细胞分裂与重组

这部分生物学背景直接关系到后面系统发育和 GWAS 的建模假设。

- **DNA 复制（replication）**：从复制起点（origin）解开双链，形成复制泡（replication bubble）和两个复制叉（replication fork）。DNA polymerase 只能沿 $5'\rightarrow3'$ 方向合成新链，因此与复制叉前进方向一致的前导链（leading strand）近似连续合成，方向相反的后随链（lagging strand）只能以一段段 Okazaki fragment 不连续合成，再连接起来。
- **有丝分裂（mitosis）**：通常保持体细胞的染色体组不变。
- **减数分裂（meiosis）**：经过一次复制和两次分裂形成单倍体配子，并通过非同源染色体的自由组合（independent assortment）与同源重组（homologous recombination）产生组合多样性。
- **重组的两面性**：重组既产生遗传多样性，也使同一染色体上相邻位点之间的关联随物理距离衰减——相距越远，越容易被重组打断。这正是连锁不平衡（linkage disequilibrium, LD）的生物学基础：附近的位点倾向一起遗传，因此 GWAS 找到的显著位点往往只是与真正致病位点处于 LD 中的“替身”，其定位分辨率受 LD 区块大小限制。

### 中心法则的扩展与限制

经典中心法则描述信息从 DNA 到 RNA 再到蛋白质的流动，但需要几处补充：

- DNA 可以复制，DNA 转录为 RNA，RNA 还可以经逆转录（reverse transcription）生成 DNA（如逆转录病毒、RT-PCR）。
- 但这不表示蛋白质序列能按模板逆向编码回核酸：不存在以蛋白质为模板合成核酸的机制，而且由于密码子简并，从氨基酸也无法唯一确定原来的核酸序列。
- 真核生物的前体 mRNA（pre-mRNA）还要经历 $5'$ 加帽（cap）、剪接（splicing，去除内含子）和 poly(A) 尾加工才成为成熟 mRNA。
- 可变剪接（alternative splicing）使同一个 gene 产生多个 transcript isoform，因此“一个基因对应一条蛋白质序列”的假设通常不成立。
- 翻译之后，磷酸化（phosphorylation）、糖基化（glycosylation）、乙酰化（acetylation）、泛素化（ubiquitination）等翻译后修饰（post-translational modification, PTM）以及膜转运（membrane trafficking）继续决定蛋白质的定位、稳定性与功能。这些信息都不在序列字符串里。

### 扩展字母表与翻译上下文

真实数据的字母表比 $\{A,C,G,T\}$ 大：

- 核酸序列可含 IUPAC 歧义码，例如 R 表示 A 或 G（嘌呤），Y 表示 C 或 T（嘧啶），N 表示任意碱基；
- 蛋白质序列可含 B（D 或 N）、Z（E 或 Q）、J（I 或 L）、X（未知）、U（硒代半胱氨酸）、O（吡咯赖氨酸）以及终止符号（通常写作 `*`）。

翻译结果不仅取决于字符串，还依赖阅读框（reading frame）、所在链（strand）、起始位点和使用的遗传密码表（genetic-code table）。因此“字符串到蛋白质”的映射必须同时记录物种、细胞器和坐标约定，否则同一段 DNA 可能被翻译成完全不同的蛋白。

### 坐标约定

计算时必须保留方向信息。基因组区间（genomic interval）有两种常见坐标约定：

- **0-based half-open**（BED、BAM 内部、Python 切片）：第一个碱基编号为 0，区间 $[i,j)$ 包含起点、不含终点；
- **1-based closed**（GFF/GTF、VCF、SAM 文本、大多数基因组浏览器）：第一个碱基编号为 1，区间两端都包含。

同一个区间在两套约定下的数字不同（1-based 的 $[s,e]$ 对应 0-based 的 $[s-1,e)$），两者混用会产生 off-by-one error。在正链上，半开区间 $[i,j)$ 的长度直接就是 $j-i$，这是半开约定的便利之处。负链 feature 的生物阅读方向与参考坐标递增方向相反：它的“起始”在坐标较大的一端，提取序列后需要做反向互补才得到按 $5'\rightarrow3'$ 读的转录本序列。

<br>

## 变异：尺度、类型与方向

后面的比对、GWAS 和变异效应预测都在处理“序列之间的差异”。这一节先对差异做分类，并说明从比对中解读变异时的一个基本限制。

### 按尺度分类

- **点级（point-level）变化**：包括单碱基替换（substitution）、小片段插入（insertion）和缺失（deletion），后两者合称 indel。
- **染色体级改变（chromosomal alteration）**：包括大段缺失（deletion）、重复（duplication）、倒位（inversion）与易位（translocation）。
- **拷贝数变异（copy-number variation, CNV）**：表示某个片段的拷贝数增加或减少。
- **短串联重复扩增**：其中三核苷酸重复（trinucleotide repeat）可在代际传递时继续扩张，并因是否落在编码区（coding region）而产生不同后果。例如落在编码区的 CAG 重复扩增会在蛋白中产生多聚谷氨酰胺链（如亨廷顿病），落在非编码区的重复则可能影响转录或 RNA 加工。

### 比对中的方向解释

两条序列比对后出现一个 gap，只是假设某处发生过 indel。仅凭两条现存序列，无法判定是一条序列发生了插入，还是另一条发生了缺失——两种历史给出完全相同的比对。要确定方向，需要一个外群（outgroup）序列或一个祖先模型：若外群与其中一条序列一致，通常认为另一条发生了变化。这也是为什么“indel”这一中性术语比直接说“插入”或“缺失”更准确。

### 蛋白质变异的位置依赖

蛋白质变异的影响依赖其所处的位置环境。同样的氨基酸替换（amino-acid substitution），位于活性位点可能使酶失活，位于疏水核心可能破坏折叠，位于无序区或表面则可能几乎没有影响。因此只看序列层面的替换类型不足以判断后果，还需要结构与功能层面的表示。这一点在后面的变异效应预测中会反复出现。

<br>

## 测量：观测值不是生物状态本身

算法处理的是实验观测值，而不是不受测量误差影响的“真实”生物状态。每一种常见数据都有自己的产生过程和误差来源：

- **测序读段（sequencing read）** 是从分子群体中随机采样、并带有测序错误的片段。
- **RT-PCR** 先把 RNA 逆转录（reverse-transcribe）为 cDNA，再以基因特异性引物（gene-specific primer）扩增目标；实时 qPCR 通过荧光循环曲线（fluorescence cycle curve）做相对或绝对定量——目标越多，荧光越早越过阈值。
- **芯片（microarray）强度** 是探针杂交和扫描产生的连续信号，受探针亲和力、背景和饱和影响。
- **RNA-seq 计数（count）** 同时受真实丰度、建库过程和测序深度影响：测得更深的样本每个基因的 count 都更高，未必是表达更高。
- **PDB 坐标** 是由实验电子密度（或其他实验数据）与结构模型共同产生的估计，带有分辨率和 B-factor 所反映的不确定性。
- **相互作用网络中的边（interaction edge）** 依赖实验方法（assay）、阈值和数据库的筛选规则。

因此，表示至少有三层：真实状态、观测过程、文件中的编码。把文件中的数值直接等同于真实状态，就会把技术噪声误当成生物学差异。

### 表达测量技术的演变

表达测量的发展体现了表示粒度的变化：

- **RT-PCR** 只针对少量预先选定的目标（target），精确但通量低；
- **microarray** 依赖预先设计好的探针阵列（probe grid）和杂交强度（hybridization intensity），一次可测成千上万个基因，但只能测探针覆盖到的序列；
- **RNA-seq** 把 cDNA 随机打断成片段，再从单端（single-end）或双端（paired-end）读取，不需要事先知道要测什么。

paired-end 数据对每个片段同时给出两端的两个 read 以及插入片段长度（insert size）信息，有助于跨越重复区和识别剪接异构体。reads 经过比对或伪比对（pseudoalignment）后，再按 gene 或 transcript 计数，得到表达矩阵。需要注意，读长约 250 bp 只是某些平台的特定配置，不是 RNA-seq 的定义。

<br>

## 表观遗传与调控

DNA 序列在一个个体的几乎所有细胞中相同，但肝细胞和神经元表达的基因截然不同，这种差异在细胞分裂后还能保持。表观遗传调控（epigenetic regulation）就是这样一类机制：它改变可遗传或可维持的表达状态，而不改变 DNA 碱基序列。

- DNA 甲基化（DNA methylation）、组蛋白修饰（histone modification）与染色质可及性（chromatin accessibility）共同影响转录机器能否接近启动子（promoter）和增强子（enhancer）。
- 约 22 nt 的 miRNA 可以通过与目标 RNA 部分互补配对，降低其稳定性或抑制翻译。

需要警惕因果方向：表观遗传标记既可能参与调控、导致表达变化，也可能只是细胞类型、发育状态或疾病过程的结果。观察到两组样本之间的甲基化差异，不能直接推出“甲基化导致了疾病”。

<br>

## 表示层次与选择

同一个生物对象可以有多种表示，它们适合不同的算法：

- **序列**：保留残基顺序，适合比对、motif 发现和语言模型。
- **特征向量**：用长度、GC 含量、$k$-mer 频率、理化性质等数值特征表示，适合分类与聚类，但丢失了位置顺序。
- **图**：原子为节点、化学键为边（分子图）；或基因为节点、相互作用为边（生物网络）。适合表示连接关系。
- **三维坐标**：每个原子的 $(x,y,z)$，适合结构叠合、距离图和分子动力学。
- **概率模型**：把一个家族或一个过程表示成分布（如 profile HMM），而不只保存一个共识序列，因此能表达“这个位置允许哪些残基、各有多大概率”。
- **本体与知识图谱**：表示实体、实体间关系、支持证据和层级约束（如“激酶活性”是“催化活性”的子类）。

> [!WARNING]+ 表示决定可回答的问题
>
> FASTA 序列无法表达原子坐标；单个 PDB 构象不等于溶液中的完整构象分布；基因表达矩阵显示的是共变，却不自动给出调控方向。算法输出的上限由输入表示决定——再好的算法也不能从表示中恢复它已经丢弃的信息。

<br>

## 基础代码：反向互补、k-mer 与 FASTA 解析

下面三个函数是后续各章反复使用的基础工具。`reverse_complement` 实现上文的反向互补，并支持 IUPAC 歧义码（如 R=A/G 的互补是 Y=C/T，B=非A 的互补是 V=非T）；`kmers` 按起点顺序产生所有长度为 $k$ 的窗口；`read_fasta` 把 FASTA 文本解析成 `(record_id, sequence)` 对。三者都用生成器或流式处理，避免一次把大文件全部读入内存。

```python
DNA_ALPHABET = set("ACGTRYSWKMBDHVN")
# str.translate 需要等长映射；这里同时覆盖常见 IUPAC 歧义码。
DNA_COMPLEMENT = str.maketrans(
    "ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN"
)

def reverse_complement(seq: str) -> str:
    """返回 DNA 序列的反向互补；非法字符直接报错。"""
    seq = seq.upper()
    invalid = set(seq) - DNA_ALPHABET
    if invalid:
        raise ValueError(f"invalid DNA symbols: {sorted(invalid)}")
    # 先逐字符互补，再反转到 5'→3' 方向。
    return seq.translate(DNA_COMPLEMENT)[::-1]

def kmers(seq: str, k: int):
    """按起点顺序惰性地产生所有长度为 k 的窗口。"""
    if k <= 0:
        raise ValueError("k must be positive")
    # 当 k > len(seq) 时 range 为空，生成器自然不产生结果。
    for i in range(len(seq) - k + 1):
        yield seq[i:i + k]

def read_fasta(lines):
    """从文本行迭代器解析 FASTA，产生 (record_id, sequence)。"""
    name, chunks = None, []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            # 读到新 header 时，先提交上一条已经完成的记录。
            if name is not None:
                yield name, "".join(chunks)
            header = line[1:].strip()
            if not header:
                raise ValueError("empty FASTA header")
            # 本教学实现只取 header 的第一个空白分隔字段作为 ID。
            name, chunks = header.split()[0], []
        else:
            if name is None:
                raise ValueError("sequence encountered before FASTA header")
            chunks.append(line)
    # 文件结尾没有下一个 header，因此最后一条需显式提交。
    if name is not None:
        yield name, "".join(chunks)
```

几点实现细节值得说明：

- `read_fasta` 把多行序列先存入列表 `chunks`，最后用 `"".join` 拼接，而不是反复做字符串 `+=`；后者在 Python 中可能退化为平方时间。
- 长度为 $n$ 的序列共有 $n-k+1$ 个 $k$-mer（当 $k\le n$ 时）。枚举全部 $k$-mer 需要 $O(n)$ 次窗口访问；但 `seq[i:i + k]` 每次都会复制一个长度为 $k$ 的子串，所以实际字符处理量为 $O(nk)$。使用 rolling hash，或把碱基编码为 2-bit 整数、每次移位更新，就可以避免重复复制，使每个窗口的更新为常数时间（见“字符串计算”一章的 $k$-mer 计数部分）。

> [!NOTE]+ 小结
>
> - 序列是有限字母表上的字符串，这一抽象丢弃了环境、构象与修饰，换来高效算法。
> - DNA 有 $5'\rightarrow3'$ 方向；反向互补 = 逐字符互补 + 反转。
> - 0-based half-open 与 1-based closed 坐标混用是 off-by-one 的主要来源；负链 feature 需反向互补。
> - gap 只说明发生过 indel，方向需外群或祖先模型判断。
> - 数据至少分三层：真实状态、观测过程、文件编码；表示决定可回答问题的上限。

<br>

# 数据与数据库

生物信息分析通常不是从手工输入一条序列开始，而是从公共数据库、测序文件或实验平台导出的记录开始。一条数据库记录不只有数据本身，还有标识符、版本、物种、实验方法和注释证据；文件格式则规定这些信息怎样编码。需要牢记的是，数据库记录是某一时间点的整理结果，而不是永远不变的生物学真值：序列会被修正，注释会被更新，记录会被合并或撤销。

这一章有两个任务。第一，正确解析 FASTA、FASTQ、VCF、SAM、mmCIF 等文件，把它们转换成后续算法需要的对象。第二，记录数据来源（provenance），使结果能在数据库更新之后复现。典型的输入是 accession、检索条件或原始文件；输出除了序列、坐标和表达矩阵之外，还应包括数据库 release、下载日期、checksum、解析器版本以及未能映射的记录清单。

后续的序列比对、结构计算和组学分析都默认输入已经通过本章的格式检查与版本固定。还要注意一个解释边界：文件能被程序读入，只说明它在语法上合法，并不说明物种、参考组装、坐标约定或生物学含义是正确的。

## 常见文件格式

### 格式一览

| 格式 | 内容 | 要点 |
| :--- | :--- | :--- |
| FASTA | 序列 | 标识行以 `>` 开头，随后是序列；简单，但元数据结构松散 |
| FASTQ | 测序 reads | 每条 read 包含标识、序列、分隔符和 Phred 质量值四部分 |
| SAM/BAM/CRAM | reads 对参考基因组的比对 | BAM 为 SAM 的二进制压缩形式；CRAM 进一步依赖参考序列做压缩，读取时必须有同一参考 |
| VCF | 基因组变异 | 记录变异位点及各样本的 genotype |
| PDB/mmCIF | 大分子三维结构 | 链、残基、原子坐标、占有率（occupancy）与实验信息 |
| SDF | 小分子 | 可保存分子连接、二维或三维坐标及属性字段 |
| SMILES | 小分子 | 分子图的线性字符串表示，例如乙醇写作 `CCO` |
| GFF/GTF | 基因组注释 | 基因组区间与基因结构（外显子、CDS 等）注释 |
| 表达矩阵 | 表达量 | 行通常为基因，列为样本，元素是 count 或归一化表达值 |

### FASTQ 与 Phred 质量值

测序仪对每个碱基都给出一个“这个碱基被读错的概率”估计，FASTQ 用 Phred 分数记录它：

$$
Q=-10\log_{10}P_e
$$

其中 $P_e$ 是估计的错误概率。反过来 $P_e=10^{-Q/10}$。因此 $Q=30$ 对应估计错误概率 $10^{-3}$，即千分之一。几个常用值如下：

| $Q$ | $P_e$ | 正确率 |
| :--- | :--- | :--- |
| 10 | $10^{-1}$ | 90% |
| 20 | $10^{-2}$ | 99% |
| 30 | $10^{-3}$ | 99.9% |
| 40 | $10^{-4}$ | 99.99% |

质量值是对数尺度的概率编码，不应直接当作线性置信度：$Q=40$ 并不是比 $Q=20$ “可靠一倍”，而是错误率低 100 倍。对质量值取算术平均也会产生误导，平均应在 $P_e$ 尺度上进行。

FASTQ 文件中的质量值以字符存储，常见 Phred+33 等字符编码约定：字符的 ASCII 码减去 33 才是 $Q$。因此解析时，字符 `I` 本身不是质量等级，要先转换为整数：`I` 的 ASCII 码为 73，$73-33=40$，即 $Q=40$。

若假设各碱基的错误相互独立，一条 read 全部碱基都正确的概率为

$$
\prod_i(1-P_{e,i}).
$$

例如一条 100 bp、每个碱基 $Q=30$ 的 read，全部正确的概率约为 $0.999^{100}\approx0.905$，也就是说即使质量很高，约十分之一的 read 至少含一个错误。但实际中错误常受测序周期（越靠 read 末端越差）、局部序列（如均聚物）和仪器状态影响，并不严格独立，所以这个乘积只是近似。

### SAM：比对记录

SAM 记录一条 read 被放在参考基因组的什么位置、以什么方式对齐。核心字段包括：query name（read 名）、按位编码的 FLAG（是否成对、是否反向互补、是否未比对上等）、reference（参考序列名）、position（比对起点，1-based）、mapping quality、CIGAR 和 sequence。

CIGAR 用“长度 + 操作”串描述比对结构，例如 `8M2I4M` 表示 8 个对齐位置、2 个插入、4 个对齐位置。需要注意，`M` 表示 alignment match，即“这个位置 read 与参考对上了一列”，它同时包含碱基相同（sequence match）与碱基不同（mismatch）两种情况。要区分二者，需要使用扩展操作 `=`（相同）/`X`（不同），或者结合 read、reference 序列与 MD tag（记录参考碱基的可选字段）来判断。

mapping quality 不是 base quality，二者回答不同的问题：

- mapping quality（MAPQ）描述 read 被放到该位置的不确定性，同样采用 Phred 尺度，即 $-10\log_{10}P(\text{比对位置错误})$；一条落在重复序列中的 read，碱基质量再高，也可能因为存在多个同样好的位置而 MAPQ 很低。
- base quality 描述碱基本身的测量错误。

### VCF：变异记录

VCF 中的 `REF`/`ALT` 字段必须相对于特定的参考组装（reference assembly）解释：同一个位置编号在 GRCh37 与 GRCh38 上指向不同的碱基。此外，相同的变异可能有不同的表示，尤其是重复序列中的 indel 可以左移或右移而不改变结果序列；比较两个 VCF 之前应先做规范化（normalize，通常是左对齐并去掉冗余碱基），并明确使用的是 GRCh37 还是 GRCh38。

genotype 字段 `0/1` 与 `1|0` 都表示杂合（一个参考等位基因、一个替代等位基因），差异在于后者带相位（phase）：竖线表示已知哪个等位基因在哪条同源染色体上，斜杠表示未定相。

### PDB 与 mmCIF：结构文件

旧的 PDB 格式是固定列宽格式，原子编号、链 ID 等字段宽度有限，在大结构和复杂标识上受限；mmCIF 是当前标准。解析结构时，代码不应假设：

- chain ID 永远是单字符（mmCIF 中可以是多字符）；
- residue number 是连续的（存在缺失残基和编号跳跃）。

插入码（insertion code，同一编号下的 `52A`、`52B`）、非标准残基、缺失坐标与生物学组装体（biological assembly，晶体非对称单元不一定等于功能状态下的复合物）都需要显式处理。

<br>

## 主要数据库

下列数据库是课程中最常用的入口，各自覆盖不同层次的对象：

- [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)：核酸序列与注释。
- [UniProt](https://www.uniprot.org/)：蛋白质序列与功能注释。其中 Swiss-Prot 部分偏人工审校（reviewed），TrEMBL 部分偏自动注释（unreviewed），规模大得多但注释可靠性较低。
- [RCSB PDB](https://www.rcsb.org/)：PDB archive 中由实验测定的结构；RCSB.org 另外也提供对计算预测模型资源（如 AlphaFold DB）的访问，使用时需区分实验结构与预测模型。
- [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/)：转录组和功能基因组数据。
- [Gene Ontology](https://geneontology.org/)：用受控术语描述基因产物的分子功能、生物过程和细胞组分三个方面。
- [ChEMBL](https://www.ebi.ac.uk/chembl/) 与 [PubChem](https://pubchem.ncbi.nlm.nih.gov/)：化合物、靶点与生物活性数据。
- [PharmGKB](https://www.pharmgkb.org/)：基因型、药物和表型之间关系的知识。

数据库记录不是无条件的真值。使用前需要检查 accession、版本、物种、组装版本、实验方法、证据码（evidence code，说明注释来自实验还是推断）、发布日期和许可证。构建机器学习数据集时，训练/测试拆分还需避免信息泄漏：同源序列、同一患者的多个样本或近乎重复的结构若分别落入训练集和测试集，模型的测试性能会被严重高估。

<br>

## 数据谱系与可复现性

同一个检索条件，今天和半年后在 UniProt 中得到的结果可能不同；同一个 accession 指向的序列也可能被修订过。可复现分析因此不能只保存下载文件名，而要记录数据谱系（provenance）：数据从哪里来、哪个版本、怎样查询、何时获取、内容的指纹是什么、用什么软件解析。

### 最小 provenance 记录

一份最小的 provenance 包含以下字段：

```yaml
resource: UniProtKB/Swiss-Prot
release: "2026_03"
query: "organism_id:9606 AND reviewed:true"
retrieved_at: "2026-09-21"
sha256: "..."
software:
  parser: "biopython==1.x"
```

其中 `release` 固定数据库版本，`query` 记录完整检索条件（9606 是人类的 NCBI Taxonomy ID），`retrieved_at` 记录获取日期，`sha256` 是文件内容的哈希值，`software` 记录解析器及其版本。

### 标识符会变化

稳定标识符（stable identifier）也不是永久不变的：记录会被合并、拆分或撤销。可靠的 pipeline 应当保留原始 ID、ID 映射表、目标 namespace、版本，以及每条未能映射记录的原因，而不是把映射失败的记录静默丢弃——静默丢弃会让样本悄悄减少，且无法追查。

### 带 provenance 的下载

下面的函数在下载公开资源的同时，返回一份可审计的最小 provenance：请求的 URL、重定向后的实际 URL、UTC 时间戳、内容类型、服务器提供的 `ETag` 与 `Last-Modified`，以及内容的 SHA-256。

```python
from datetime import datetime, timezone
import hashlib
from urllib.request import Request, urlopen

def fetch_with_provenance(url, timeout=30):
    """下载公开资源并同时返回可审计的最小 provenance。"""
    request = Request(
        url,
        headers={"User-Agent": "cs274-teaching-example/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
        metadata = {
            "requested_url": url,
            "resolved_url": response.geturl(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "content_type": response.headers.get("Content-Type"),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return payload, metadata
```

实际使用时还有几点限制：

- 数据库批量下载要遵守 API 速率限制（rate limit）、认证要求与许可证，并优先保存 release identifier。
- `ETag`、`Last-Modified` 可能缺失（代码中会得到 `None`）；checksum 只能证明本地内容是否改变，不能替代数据库版本——两个不同 release 的同一文件可能哈希相同，而哈希不同也不说明是哪个 release。
- 生产流程应对瞬时网络错误（如超时、503）重试，但不能把 404、权限拒绝或 schema 改变静默当作“空数据”继续往下跑。

### accession 与 version

accession 标识的是一条数据库记录，而 version 才标识该记录具体的序列或注释状态。例如 RefSeq 的 `NM_000314.8` 中，`NM_000314` 是 accession，`.8` 是版本号，序列被修订后版本号递增。可复现的检索应保存数据库 release、完整 query、返回的 `accession.version`、检索日期和 checksum；仅保存 gene symbol 或网页 URL，无法抵抗记录的合并、拆分和更新。

> [!EXAMPLE]+ 例1：同名不是同一实体
>
> `PTEN` 可以指 gene symbol、它编码的蛋白，或论文中的一个概念；Ensembl 的 gene、transcript 与 protein ID 位于不同层次，一个 gene 对应多个 transcript，一个 transcript 对应一个或零个 protein。若表达矩阵按 gene 汇总，而结构数据按 protein isoform 记录，直接用 symbol 做 join 会制造多对多的重复行，使后续统计被重复计数。正确的做法是先声明分析单位（gene 还是 isoform），再通过带版本的关系表进行映射。

> [!NOTE]+ 小结
>
> - 文件能读入只说明语法合法，不说明物种、组装、坐标或含义正确。
> - Phred 值 $Q=-10\log_{10}P_e$ 是对数尺度，Phred+33 下字符 `I` 对应 $Q=40$；MAPQ 与 base quality 含义不同。
> - CIGAR 的 `M` 同时包含 match 与 mismatch；VCF 要绑定参考组装并先 normalize。
> - 数据库记录不是真值；可复现需保存 release、query、`accession.version`、日期与 checksum，并保留未映射记录。

<br>

# 字符串计算：精确匹配、k-mer 与组装

把 DNA、RNA 或蛋白质抽象成有限字母表上的字符串之后，最先要解决的是定位和计数。给定一段长文本 $T$（如参考基因组）和一个较短的模式 $P$（如一条 read 或一个 motif），需要回答模式出现在哪里；给定大量 reads，还需要统计其中的局部短词、建立索引，或者根据 reads 之间的重叠恢复原始序列。这一章暂不考虑复杂替换和长 gap，重点是精确的字符串结构——它们是后面带误差比对的“种子”和加速器。

本章的输入是文本、模式、$k$ 值或 read 集合，输出是匹配位置、前缀表、$k$-mer 频数、后缀索引或 de Bruijn 图。三类核心思想分别是：

- KMP 利用模式内部的前后缀自相似，避免重复比较；
- FM-index 以压缩索引支持在巨大文本中快速检索；
- $k$-mer 把长序列切成局部短词，用局部信息换取速度和可扩展性。

与后续各章的关系是：精确索引先找出 seed，双序列比对再在 seed 周围允许 mismatch 与 gap；$k$-mer 重叠先构建组装图，图算法再从中选择路径。需要注意解释边界：“相同子串”只表示字面相同，不等于进化同源或功能相同——低复杂度序列、重复序列都会产生大量没有生物学意义的字面匹配。

## 精确匹配：从朴素算法到 FM-index

### 各方法的复杂度比较

设文本长 $n$、模式长 $m$。

- **朴素匹配**：对每个起点逐字符比较，最坏为 $O(nm)$，例如在 `AAAA…A` 中找 `AA…AB`，每个起点都要比到最后一位才失败。
- **KMP**：用前缀函数记录模式自身的边界信息，时间为 $O(n+m)$。
- **后缀树（suffix tree）**：可以在线性时间建立索引，查询只需 $O(m)$ 加输出量，但常数和内存都较大（每个字符可能需要数十字节）。
- **后缀数组（suffix array）**：把所有后缀按字典序排序后只存起点，更紧凑，配合二分查找可在 $O(m\log n)$ 时间内查询。
- **FM-index**：基于 Burrows–Wheeler Transform（BWT），以压缩空间支持反向搜索（backward search），是 BWA、Bowtie 等短读段比对器的重要基础。

选择哪种方法取决于场景：只查一次用 KMP；对同一文本（如参考基因组）查询亿万次，就值得先花时间和空间建索引。

### KMP 的前缀函数

朴素匹配的浪费在于：失配后把文本指针退回去重新比较，而那些字符其实刚刚比较过。KMP 的观察是，已经匹配上的那段文本就是模式自己的一个前缀，失配后能从哪里继续，完全由模式自身的结构决定，可以预先算好。

KMP 的前缀函数（prefix function）$\pi[i]$ 定义为模式前缀 `pattern[:i+1]` 的最长真前缀（proper prefix，不等于整个串）与真后缀的公共长度，这个公共部分称为 border。发生失配时，不必把文本指针回退，只把模式指针退到 $\pi[j-1]$（$j$ 为当前已匹配长度）：因为已匹配部分的末尾 $\pi[j-1]$ 个字符恰好等于模式开头的 $\pi[j-1]$ 个字符，它们无需重新比较。这利用的正是已匹配部分的自相似。

> [!EXAMPLE]+ 例2：前缀函数的计算
>
> 模式 `ababaca` 的前缀函数为 `[0, 0, 1, 2, 3, 0, 1]`：
>
> - `ababa` 的最长 border 是 `aba`，所以 $\pi[4]=3$；
> - 下一个字符 `c` 与 `pattern[3]='b'` 不符，退到 $\pi[2]=1$，`c` 与 `pattern[1]='b'` 不符，再退到 $\pi[0]=0$，`c` 与 `pattern[0]='a'` 仍不符，所以 $\pi[5]=0$；
> - 最后的 `a` 与 `pattern[0]` 相符，$\pi[6]=1$。
>
> 若在文本中匹配到 `ababa` 后遇到失配，模式指针直接退到 3，相当于把模式右移两位，并默认前 3 个字符 `aba` 已对齐。

为什么总时间是线性的？在 `kmp_search` 中，`matched` 每读一个文本字符最多加 1，而每次 `while` 回退至少减 1，且 `matched` 不会小于 0，所以回退的总次数不超过前进的总次数 $n$。构造前缀表同理为 $O(m)$。

```python
def kmp_prefix(pattern: str) -> list[int]:
    """构造 KMP prefix table；pi[i] 是最长真前后缀长度。"""
    pi = [0] * len(pattern)
    border = 0
    for i in range(1, len(pattern)):
        # 当前字符不匹配时，退回到更短的候选 border。
        while border > 0 and pattern[i] != pattern[border]:
            border = pi[border - 1]
        if pattern[i] == pattern[border]:
            border += 1
        pi[i] = border
    return pi


def kmp_search(text: str, pattern: str) -> list[int]:
    """返回 pattern 在 text 中全部匹配的 0-based 起点，包含重叠命中。"""
    if pattern == "":
        # 约定空模式在每个字符间隙都匹配。
        return list(range(len(text) + 1))

    pi = kmp_prefix(pattern)
    hits = []
    matched = 0
    for i, char in enumerate(text):
        while matched > 0 and char != pattern[matched]:
            matched = pi[matched - 1]
        if char == pattern[matched]:
            matched += 1
        if matched == len(pattern):
            hits.append(i - len(pattern) + 1)
            # 保留最长可复用 border，因而能发现重叠命中。
            matched = pi[matched - 1]
    return hits
```

### BWT 与 FM-index 的反向搜索

BWT 的构造方式是：先在文本末尾加入一个唯一且字典序最小的终止符 `\$`，再将文本的全部循环旋转（rotation）按字典序排序，取排序后矩阵的最后一列。终止符保证这一变换可逆，并唯一标识文本的结尾。BWT 的价值在于，它倾向于把相同字符聚在一起（便于压缩），同时保留了足够的信息支持搜索。实际的 FM-index 不会显式保存所有旋转（那需要 $O(n^2)$ 空间），而是通过后缀数组构建：排好序的旋转与排好序的后缀一一对应，BWT 的第 $i$ 个字符就是第 $i$ 小后缀前面的那个字符。

反向搜索的思想是：从模式的最后一个字符开始，每次在左侧加一个字符，同时维护“以当前模式片段开头的所有后缀”在后缀数组中的连续区间。采用半开区间 $[l,r)$，定义：

- $C[c]$：文本中字典序小于字符 $c$ 的字符总数，即以 $c$ 开头的后缀在排序中从第 $C[c]$ 行开始；
- $\operatorname{Occ}(c,i)$：`BWT[0:i]` 中字符 $c$ 出现的次数。

当模式从右向左加入字符 $c$ 时，后缀数组区间（suffix-array interval）更新为

$$
l'=C[c]+\operatorname{Occ}(c,l),\qquad
r'=C[c]+\operatorname{Occ}(c,r)
$$

直觉是：区间 $[l,r)$ 内那些 BWT 字符为 $c$ 的行，正是“前面紧跟着 $c$”的后缀；在它们前面加上 $c$ 后，得到的新后缀都以 $c$ 开头，因而落在以 $c$ 开头的区段（从 $C[c]$ 起），并且保持原有的相对顺序（这一性质称为 LF-mapping）。新区间的左端等于 $C[c]$ 加上排在它们之前、同样以 $c$ 结尾的行数。

区间非空（$l'<r'$）即说明存在匹配，区间长度 $r'-l'$ 就是出现次数；区间变空则可以立即停止。每加一个字符只需两次 $\operatorname{Occ}$ 查询，因此查询时间与文本长度无关，只与模式长度成正比。

> [!EXAMPLE]+ 例3：在 banana 中反向搜索 ana
>
> 文本 `banana` 加终止符后，排序的旋转与 BWT 如下：
>
> ```text
> 行  排序后的旋转   最后一列
> 0   $banana        a
> 1   a$banan        n
> 2   ana$ban        n
> 3   anana$b        b
> 4   banana$        $
> 5   na$bana        a
> 6   nana$ba        a
> BWT = "annb$aa"
> ```
>
> 字符计数为 $C[\$]=0$、$C[a]=1$、$C[b]=4$、$C[n]=5$。初始区间 $[0,7)$。
>
> - 加入 `a`：$l=1+\operatorname{Occ}(a,0)=1$，$r=1+\operatorname{Occ}(a,7)=1+3=4$，区间 $[1,4)$，即三个以 a 开头的后缀。
> - 加入 `n`：$l=5+\operatorname{Occ}(n,1)=5$，$r=5+\operatorname{Occ}(n,4)=5+2=7$，区间 $[5,7)$，对应 `na…` 的两个后缀。
> - 加入 `a`：$l=1+\operatorname{Occ}(a,5)=2$，$r=1+\operatorname{Occ}(a,7)=4$，区间 $[2,4)$。
>
> 区间长度为 2，`ana` 在 `banana` 中出现两次（起点 1 和 3，两次出现相互重叠）。

实际实现中，$\operatorname{Occ}$（rank）查询通过每隔若干位置存一个检查点（checkpoint）或使用 wavelet tree 加速，在空间与时间之间折中。若要允许 mismatch 或 indel，每一步可能尝试多个字符，搜索会分支并指数增长，因此实际的比对器把精确 seed、剪枝和动态规划结合起来使用。

<br>

## k-mer：局部短词的计数与比较

### 用途与 Jaccard 相似度

$k$-mer 是长度为 $k$ 的连续子串。把长序列切成局部短词后，许多原本需要比对的问题可以改用“数词”来近似。其用途包括：

- 无需比对的序列相似度估计（alignment-free similarity）；
- de Bruijn 图组装；
- minimizer 索引；
- 物种分类（如宏基因组 reads 的归类）；
- 作为序列模型的输入特征。

若两条序列的 $k$-mer 集合分别为 $A,B$，它们的 Jaccard 相似度为

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

取值在 0（无共同 $k$-mer）到 1（集合完全相同）之间。

> [!EXAMPLE]+ 例4：3-mer Jaccard
>
> `ACGTAC` 的 3-mer 集合为 $\{ACG,CGT,GTA,TAC\}$，`ACGTTC` 的为 $\{ACG,CGT,GTT,TTC\}$。交集 $\{ACG,CGT\}$ 有 2 个元素，并集有 6 个，$J=2/6=1/3$。仅一个碱基不同，Jaccard 就从 1 降到 1/3，因为一个替换会破坏所有覆盖它的 $k$ 个 $k$-mer（此处边界截断后为 2 个）。

$k$ 的选择是一个权衡：增大 $k$ 会降低随机碰撞（两个无关序列偶然共享 $k$-mer），但更怕测序错误和变异，因为一个错误会破坏 $k$ 个 $k$-mer。

对大规模数据，两种技巧进一步压缩成本：

- **MinHash** 用少量哈希函数的最小值近似 Jaccard：对随机哈希函数，两个集合的最小哈希值相等的概率恰好等于 $J(A,B)$，因此只需保存每个集合的若干最小值（sketch）即可估计相似度；
- **minimizer** 在每个滑动窗口内只保留一个代表 $k$-mer（如哈希值最小者），相邻窗口往往选中同一个，从而大幅减少索引条目。

上述 Jaccard 把 $k$-mer 当作集合元素，忽略了同一 $k$-mer 出现的次数（multiplicity）。若 coverage 或 copy number 重要（如比较拷贝数不同的基因组），应使用多重集/加权 Jaccard（multiset/weighted Jaccard，用 $\sum\min/\sum\max$ 代替交并计数），或直接比较 count vector。

### 随机出现概率与 canonical k-mer

若 DNA 背景近似为四种碱基均匀独立，一个指定 $k$-mer 在某个位置随机出现的概率约为 $4^{-k}$；在长 $n$ 的序列中期望出现约 $n\cdot4^{-k}$ 次。例如人类基因组约 $3\times10^9$ bp，$4^{16}\approx4.3\times10^9$，所以 $k\approx16$ 以上时随机 $k$-mer 才倾向于唯一。但真实基因组有 GC 偏好（GC bias）和大量重复序列，因此不能直接用这一近似设定显著性阈值。

由于测序 read 可能来自任一条链，同一位置的 $k$-mer 可能以自身或其反向互补的形式出现。canonical $k$-mer 通常取一个 $k$-mer 与其反向互补中字典序较小者作为代表，从而合并双链方向。例如 `ACG` 的反向互补是 `CGT`，canonical 形式为 `ACG`。

### 计数的工程权衡

对数十亿条 reads 统计 $k$-mer 时，内存往往是瓶颈，常见方案各有取舍：

- **精确哈希表（exact hash table）**：结果准确，但内存高；
- **Bloom filter**：只回答“是否出现过”，节省空间，但有假阳性（不会有假阴性）；
- **Count–Min Sketch**：用多个哈希计数器取最小值，给出频数的上界近似（只会高估不会低估）；
- **rolling 2-bit encoding**：每个碱基编码为 2 bit，窗口右移时整数左移 2 位、加入新碱基，再用 $2k$ 位掩码去掉移出窗口的那个碱基所占的高 2 位，使每次窗口更新为 $O(1)$，避免了“生物表示”一章基础代码中提到的子串复制成本；
- **按丰度过滤**：测序错误通常产生只出现一两次的低频 $k$-mer，可以按 abundance 过滤掉，但这也会误删真实的稀有变异（rare variant）。

<br>

## 序列组装与 de Bruijn 图

测序只能读出短片段，组装（assembly）要从大量重叠的 reads 中恢复原始长序列。直接的思路是构建重叠图（overlap graph）：每条 read 是节点，两条 read 有足够重叠就连边，然后寻找经过每个节点恰好一次的路径。但这是 Hamiltonian path 问题，一般是 NP-hard 的。de Bruijn 图换了一种建图方式，使问题变成可在线性时间求解的 Eulerian path。

### 构图与 Eulerian path

de Bruijn 图把每个 $k$-mer 的长度为 $k-1$ 的前缀和后缀作为节点，把 $k$-mer 本身作为一条从前缀指向后缀的边。这样，“排列 reads”变成“找一条经过每条边恰好一次的路径”，即从 overlap graph 中的序列排序问题转化为 de Bruijn 图上的 Eulerian path。

图的形状直接反映数据的问题：重复序列造成分叉，测序错误造成短小的死胡同分支（tips），覆盖不均造成图的断裂。

> [!EXAMPLE]+ 例5：由三个 3-mer 拼出 ATGCA
>
> reads 中观察到 `ATG`、`TGC`、`GCA`。取 $k=3$，节点为 2-mer，得到边 `AT→TG`、`TG→GC`、`GC→CA`。沿 Eulerian path 依次走过三条边，把第一个节点和之后每个节点的最后一个字符拼起来，得到 `ATGCA`。
>
> 若同时还存在 `TGA`，节点 `TG` 就有两条出边（`TG→GC` 与 `TG→GA`），出现分叉。仅靠图的拓扑不能判断这一分支来自真实变异、重复序列还是测序错误，需要借助覆盖度、paired-end 信息等额外证据。

### Eulerian path 的存在条件与困难所在

有向图存在 Eulerian path，需要同时满足：

- 所有出度或入度非零的节点，在忽略边方向后位于同一个连通部分；
- 至多一个节点满足出度比入度大 1（路径起点），至多一个节点满足入度比出度大 1（路径终点），其余节点入度等于出度。

若所有节点入度都等于出度，则存在 Eulerian 回路。Hierholzer 算法可以在 $O(|E|)$ 时间内找到 Eulerian path：从起点出发沿未用过的边一直走，走不动时回退并从途中节点拼接出新的环。

但组装的真正困难不在于跑 Eulerian path，而在于：noisy reads 产生巨大的多重图（multigraph）；重复序列长于 read 或 $k$ 时，无法唯一确定跨越方式（一个图可能有许多条 Eulerian path，只有一条对应真实基因组）；需要整合 paired-end 的距离约束；多倍体（polyploid）基因组中不同拷贝的差异也会产生分支。

<br>

## 编辑距离：从精确匹配走向带代价的比对

精确匹配只回答“是否完全相同”，而生物序列之间几乎总存在替换和插入缺失。要衡量“差多少”，需要允许编辑操作并给它们代价，这就是从字符串匹配走向序列比对的一步。

### 两种互补视角

- **Levenshtein 编辑距离（edit distance）** 定义为把一个字符串变成另一个所需的插入（insertion）、删除（deletion）与替换（substitution）的最少次数，是“最小代价”视角。例如 `kitten` 变为 `sitting` 需要 3 次编辑（两次替换、一次插入）。
- **最长公共子序列（longest common subsequence, LCS）** 最大化两个字符串中保持相对顺序（不必连续）的公共字符数，是“最大得分”视角。

二者都可以用 $O(nm)$ 的动态规划求解。以编辑距离为例，令 $D[i][j]$ 为 $x$ 的前 $i$ 个字符变为 $y$ 的前 $j$ 个字符的最少编辑次数，则

$$
D[i][j]=\min\begin{cases}
D[i-1][j]+1 & \text{删除 }x_i,\\
D[i][j-1]+1 & \text{插入 }y_j,\\
D[i-1][j-1]+[x_i\ne y_j] & \text{替换或匹配},
\end{cases}
$$

边界为 $D[i][0]=i$、$D[0][j]=j$（与空串之间只能全删或全插）。每格只依赖左、上、左上三格，因此按行填表时只需保留上一行。把单位代价换成替换矩阵分数和 gap 罚分、把 min 换成 max，就得到下一章的 Needleman–Wunsch 全局比对，因此编辑距离是从精确字符串匹配进入生物序列比对的桥梁。

### 两行 DP 实现

```python
def levenshtein(x: str, y: str) -> int:
    """用两行 DP 计算单位代价 Levenshtein distance。"""
    # 令 y 为较短序列，使额外空间为 O(min(n, m))。
    if len(x) < len(y):
        x, y = y, x
    previous = list(range(len(y) + 1))

    for i, char_x in enumerate(x, start=1):
        current = [i]  # x[:i] 变为空串需要 i 次 deletion。
        for j, char_y in enumerate(y, start=1):
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            substitution = previous[j - 1] + (char_x != char_y)
            current.append(min(deletion, insertion, substitution))
        previous = current
    return previous[-1]
```

代码中 `previous` 是上一行 $D[i-1][\cdot]$，`current` 是当前行 $D[i][\cdot]$；`(char_x != char_y)` 作为布尔值参与加法，相等时为 0、不等时为 1，正是上式中的 $[x_i\ne y_j]$。由于编辑距离对称，交换 $x,y$ 不改变结果。代价是只保留两行后无法直接回溯出具体的编辑操作序列；若需要比对本身，要保存整张表，或使用前文提到的线性空间回溯。

### 小结：精确索引与带代价比对的分工

精确索引用于快速定位相同的短词。当存在替换与 gap 时，需要把匹配改写成带代价的最优路径问题，即动态规划比对；BLAST 则结合二者，先用精确 seed 定位候选区域，再在候选附近做局部延伸，以牺牲少量灵敏度换取数量级的速度提升。

> [!NOTE]+ 小结
>
> - 朴素匹配最坏 $O(nm)$；KMP 用前缀函数做到 $O(n+m)$；后缀数组查询 $O(m\log n)$；FM-index 用 BWT 与 $C$、$\operatorname{Occ}$ 做反向搜索，查询时间只与模式长度相关。
> - $k$-mer 的 $k$ 在随机碰撞与错误敏感之间权衡；集合 Jaccard 忽略 multiplicity；canonical $k$-mer 合并双链。
> - de Bruijn 图把组装化为 Eulerian path（Hierholzer 为 $O(|E|)$），真正难点是重复、错误与多倍体造成的分支。
> - 编辑距离与 LCS 是 $O(nm)$ DP，连接精确匹配与序列比对。

<br>

<br>

# 双序列比对（Pairwise Sequence Alignment）

拿到一条新测得的基因或蛋白序列，最先想问的往往是：它和数据库里哪条已知序列“是同一个东西”，或者至少来自同一个祖先？要回答这个问题，需要把两条序列逐位置对应起来，看哪些位置保持不变、哪些位置发生了替换、哪些位置在演化中插入或丢失了。双序列比对（pairwise alignment）就是建立这种逐位置对应关系的计算方法。它是本课程后续几乎所有序列方法的地基：数据库搜索是“快速的局部比对”，多序列比对是“对很多条序列同时比对”，Pair HMM 则是“比对的概率版本”。

核心想法可以用一句话概括：<strong>把所有可能的比对看成网格上的路径，用加性打分给每条路径一个分数，再用动态规划（dynamic programming）在多项式时间内找到最高分路径。</strong>前面字符串计算一章里的编辑距离已经用过同样的网格思想，只不过那里最小化“操作次数”，这里最大化“生物学上的合理程度”，并允许不同字符对、不同 gap 长度有不同的分数。

## 比对在建模什么：对象、输入输出与任务类型

### 比对是演化假设，不只是字符串排版

比对把两条序列写成上下两行、中间插入 `-` 的形式，但这只是结果的可视化。比对真正表达的是一个演化或功能上的对应假设：同一列的两个字符被认为来自祖先序列的同一位置；如果两者不同，就假设发生过一次替换（substitution）；如果某一列一侧是 gap，就假设一侧发生过插入（insertion）或另一侧发生过缺失（deletion），二者合称 indel。因此，比对依据 substitution、insertion 和 deletion 模型建立演化或功能上的对应关系，字符串排齐只是结果的可视化。

由此要区分两个常被混用的词：

- <strong>相似性（similarity）</strong>是可计算量：给定评分系统，比对分数、identity 百分比都可以直接算出来。
- <strong>同源性（homology）</strong>表示两条序列有共同祖先，是一个“是或否”的演化结论，通常不写成“百分之多少同源”。高相似性是推断同源的证据，但“两条序列 40% 同源”这种说法在概念上是错的，应说“40% identity”。

### 输入与输出

比对算法的输入是两条序列，以及一整套评分约定：替换分数 $s(a,b)$（字符 $a$ 与 $b$ 对齐时的得分）、gap-open 罚分、gap-extension 罚分，以及边界约定（序列两端的 gap 是否罚分）。输出是在这一评分系统下的最优分数、一条或多条最优 alignment，以及每一列的对应关系（哪个残基对哪个残基）。

### 全局、局部与半全局：任务由生物问题决定

根据“哪些部分必须参与比对”，比对分为三类：

- <strong>全局比对（global alignment）</strong>要求解释两条完整序列，从头比到尾。适合全长相关、长度相近的序列，例如同一基因在两个近缘物种中的直系同源拷贝。
- <strong>局部比对（local alignment）</strong>只寻找两条序列中得分最高的局部区域，其余部分直接忽略。适合只共享一个结构域（domain）或一个保守片段的序列，例如两个多结构域蛋白只有一个共同结构域。
- <strong>半全局比对（semi-global alignment）</strong>允许指定的末端 gap 不受罚，例如把一条短的测序 read 比对到长参考基因组上：read 必须完整比上，而参考序列两端多出来的部分不应扣分。

选择哪一种由生物问题决定，不是实现细节。用全局比对去比两个只共享一个结构域的蛋白，会被迫对齐大量无关区域并产生误导性的低分；反过来，用局部比对比较两条全长同源基因，可能只报告中间一段，而漏掉末端的真实差异。

给定替换分数 $s(a,b)$ 与 gap 罚分 $g$，全局比对适合全长相关序列，局部比对适合共享局部片段的序列。

### 算法主线与后续章节的关系

本章的几个算法共用同一个骨架，后续章节在其上扩展：

- 动态规划把所有可能 alignment 表示成网格路径，并利用最优子结构避免指数枚举；
- 数据库搜索一章中的 BLAST 用启发式方法缩小局部 DP 的搜索范围；
- 多序列比对把“两个字符之间的打分”推广为“两个 profile 列之间的打分”；
- Pair HMM 把相同的状态结构解释为生成概率，从而把分数和概率联系起来。

<br>

## 比对作为网格路径：表示与加性评分

### 三种合法的列

在讨论算法之前，先要把“所有可能的比对”变成一个可以系统搜索的对象。设两条序列为 $x=x_1x_2\cdots x_n$ 与 $y=y_1y_2\cdots y_m$，长度分别为 $n$ 和 $m$。一个 alignment column（比对列）只有三种合法情况：$(x_i,y_j)$、$(x_i,-)$、$(-,y_j)$，不能是 $(-,-)$——两侧都是 gap 的列不对应任何字符，加入它只会制造无数个等价的冗余比对。

### 比对等价于网格路径

因此比对等价于从 $(n+1)\times(m+1)$ 网格的左上角 $(0,0)$ 走到右下角 $(n,m)$：

- 对角步 $(i-1,j-1)\to(i,j)$ 消耗两个字符，对应列 $(x_i,y_j)$；
- 向下一步 $(i-1,j)\to(i,j)$ 只消耗 $x_i$，对应列 $(x_i,-)$；
- 向右一步 $(i,j-1)\to(i,j)$ 只消耗 $y_j$，对应列 $(-,y_j)$。

每条路径是一种 alignment，反之亦然；路径权重是各列分数之和。路径条数随序列长度指数增长（两条长度为 $n$ 的序列约有 $\binom{2n}{n}\approx 2^{2n}/\sqrt{\pi n}$ 种全局比对），所以不能枚举。

### 为什么动态规划可行

关键在于分数是**加性的（additive）**：整条路径的分数等于各列分数之和，每一列的分数只取决于这一列本身。于是“从 $(0,0)$ 到 $(i,j)$ 的最佳前缀”与“从 $(i,j)$ 到终点的后半段怎么走”互不影响。动态规划在每个格点只保留到达该点的最佳前缀，因为 additive score 使未来与更早路径细节无关。每个格点只需比较三个来源，总共 $(n+1)(m+1)$ 个格点，计算量就从指数降到了 $O(nm)$。

### Gap 罚分为什么不能为零

Gap 不能免费。若 match 为正、mismatch 为负而 gap 为 0，算法可以在两条序列中任意插入 gap，绕开多数 mismatch，只累加偶然 match，得到缺乏生物意义的高分。例如两条随机 DNA 序列，任意一个 `A` 都能通过前后插 gap 和另一条序列中的某个 `A` 对上，最终得分只反映“两条序列里有多少相同字母”，与演化关系无关。

实际的 gap 罚分常分为两部分：gap-open penalty 表达“启动一次 indel 事件”的代价，extension penalty 表达“把同一事件延长一个位置”的代价，后文仿射 gap 一节会详细展开。至于全局与局部比对，它们的递推几乎一样，核心差异在边界初始化、终点选择和 traceback 停止条件这三处。

<br>

## Needleman–Wunsch 全局比对

### 递推式

Needleman–Wunsch 算法（1970）解决全局比对。定义 $F_{i,j}$ 为前缀 $x_1\cdots x_i$ 与 $y_1\cdots y_j$ 的最优全局比对分数（$0\le i\le n$，$0\le j\le m$）。最优比对的最后一列只能是三种合法列之一，分别对应三个来源，取最大即可。这里线性 gap 罚分 $g$ 写成负数（每个 gap 字符加 $g$）：

$$
F_{i,j}=\max\begin{cases}
F_{i-1,j-1}+s(x_i,y_j)\\
F_{i-1,j}+g\\
F_{i,j-1}+g
\end{cases}
$$

三行依次表示：最后一列是 $(x_i,y_j)$；最后一列是 $(x_i,-)$；最后一列是 $(-,y_j)$。

### 边界与复杂度

边界为 $F_{i,0}=ig$、$F_{0,j}=jg$：把长度为 $i$ 的前缀与空序列全局比对，只能是 $i$ 个 gap。这正是“全局”的体现——两端的 gap 也必须付费。

- **时间**：填表为 $O(nm)$ 时间，每格常数次比较；一次 traceback 最多走 $n+m$ 步，为 $O(n+m)$ 时间。
- **空间**：保存全表与回溯指针需 $O(nm)$ 空间；只求分数时，每一行只依赖上一行，可滚动到 $O(\min(n,m))$（让较短序列沿行方向）。
- **线性空间恢复比对**：Hirschberg 分治可在线性空间恢复比对。思路是先用前向和反向两次线性空间 DP 找到最优路径穿过中间行的位置，把问题一分为二再递归；总时间仍是 $O(nm)$（大约加倍），空间降到 $O(n+m)$。

### 最优子结构的证明

动态规划正确的前提是最优子结构（optimal substructure），可用反证法说明：若通往 $(i,j)$ 的最优路径最后一步来自 $(i-1,j-1)$，但此前缀不是 $(i-1,j-1)$ 的最优路径，把它替换成更优前缀即可提高整条路径分数，与“原路径最优”矛盾。来自另两个方向时同理。这个论证依赖加性评分：替换前缀不会改变最后一列的分数。若评分不是加性的（例如某列分数依赖很远处的列），这一论证就失效，后文的仿射 gap 正是通过增加状态把“非加性”变回加性。

### 实现

下面的实现用字典记录三个候选，`back` 存回溯方向。

```python
def needleman_wunsch(x, y, match=1, mismatch=-1, gap=-2):
    """返回全局最优分数及一条最优 alignment。"""
    n, m = len(x), len(y)
    # dp[i][j]：x[:i] 与 y[:j] 的最优全局分数。
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # D/U/L 分别表示 traceback 来自左上、上方和左方。
    back = [[None] * (m + 1) for _ in range(n + 1)]

    # 全局比对必须消费两条完整序列，因此空前缀按 gap 累计罚分。
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i * gap, "U"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j * gap, "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = {
                "D": dp[i - 1][j - 1] + (
                    match if x[i - 1] == y[j - 1] else mismatch
                ),
                "U": dp[i - 1][j] + gap,
                "L": dp[i][j - 1] + gap,
            }
            back[i][j], dp[i][j] = max(
                candidates.items(), key=lambda item: item[1]
            )

    # 从右下角回溯；先逆序收集，最后统一反转。
    ax, ay, i, j = [], [], n, m
    while i or j:
        step = back[i][j]
        if step == "D":
            ax.append(x[i - 1])
            ay.append(y[j - 1])
            i -= 1
            j -= 1
        elif step == "U":
            ax.append(x[i - 1])
            ay.append("-")
            i -= 1
        else:
            ax.append("-")
            ay.append(y[j - 1])
            j -= 1
    return dp[n][m], "".join(reversed(ax)), "".join(reversed(ay))
```

### 同分（tie）与多个最优解

当多个候选同分时，`max` 返回第一个达到最大值的元素，因此代码中 dictionary insertion order 决定优先选 `D`、`U` 或 `L`。不同 tie-breaking 可能返回不同 alignment，但最优分数相同。这一点在实践中很常见：比如 gap 放在一串重复字符（`TT`）的哪一个位置，分数完全一样。科学报告应说明是只需一个 optimum，还是需要枚举、计数或采样多个 optimum；如果下游分析依赖具体哪个残基对哪个残基，只看一个 optimum 可能会给出虚假的确定性。

> [!EXAMPLE]+ 例2：全局比对
>
> 对 `GATTACA` 与 `GCATGCU` 使用 match $+1$、mismatch $-1$、gap $-1$，一种最优解为：
>
> ```text
> G-ATTACA
> GCA-TGCU
> ```
>
> 逐列计分：`G/G` $+1$，`-/C` $-1$，`A/A` $+1$，`T/-` $-1$，`T/T` $+1$，`A/G` $-1$，`C/C` $+1$，`A/U` $-1$，总分为 $0$，与下表右下角 $F_{7,7}=0$ 一致（行为 `GATTACA`，列为 `GCATGCU`）：
>
> | | - | G | C | A | T | G | C | U |
> | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | - | 0 | -1 | -2 | -3 | -4 | -5 | -6 | -7 |
> | G | -1 | 1 | 0 | -1 | -2 | -3 | -4 | -5 |
> | A | -2 | 0 | 0 | 1 | 0 | -1 | -2 | -3 |
> | T | -3 | -1 | -1 | 0 | 2 | 1 | 0 | -1 |
> | T | -4 | -2 | -2 | -1 | 1 | 1 | 0 | -1 |
> | A | -5 | -3 | -3 | -1 | 0 | 0 | 0 | -1 |
> | C | -6 | -4 | -2 | -2 | -1 | -1 | 1 | 0 |
> | A | -7 | -5 | -3 | -1 | -2 | -2 | 0 | 0 |
>
> 第一行、第一列是边界 $jg$、$ig$；例如 $F_{1,1}=\max(0+1,\,-1-1,\,-1-1)=1$。调用 `needleman_wunsch("GATTACA", "GCATGCU", 1, -1, -1)` 返回的正是上面这条比对和分数 0。
>
> 该答案依赖参数。提高 gap-open penalty 后，算法会倾向于选择 mismatch 而不是多个短 gap。所得 alignment 是当前评分模型下的最优解，不是“真实”的演化历史。

<br>

## Smith–Waterman 局部比对

### 动机：只要最好的那一段

两个蛋白可能只共享一个结构域，其余部分毫无关系。全局比对会强迫无关区域也对齐，让真正的同源信号被大量负分淹没。Smith–Waterman 算法（1981）只寻找两条序列中**得分最高的一对子串**的比对。它与 Needleman–Wunsch 只差三处：递推中多了一个 0、边界全为 0、traceback 的起点和终点不同。

### 递推式

局部比对把 0 加入候选：

$$
H_{i,j}=\max\left(0,H_{i-1,j-1}+s(x_i,y_j),H_{i-1,j}+g,H_{i,j-1}+g\right)
$$

$H_{i,j}$ 是“以 $x_i$ 和 $y_j$ 为结尾（或在其附近结束）的最佳局部比对分数”，边界 $H_{i,0}=H_{0,j}=0$。0 相当于允许丢弃负贡献前缀：如果某个前缀累计分数已经变成负数，带着它只会拖累后面，不如从当前位置“重新开始”。

traceback 从全表最大值开始（而不是右下角，因为局部比对可以在任何地方结束），在 0 处停止（因为分数为 0 的格点表示“从这里重新开始”）。该算法保证找到最优局部比对，但长序列数据库上的 $O(nm)$ 成本催生了 BLAST 等启发式方法，见数据库搜索一章。

需要注意，局部比对要求评分系统满足“随机对齐的期望分数为负”：若平均每列期望分为正，最优局部比对会退化成几乎覆盖整条序列的全局比对，“局部”就失去意义（打分矩阵一节还会提到这一条件）。

### 实现

```python
def smith_waterman(x, y, match=3, mismatch=-3, gap=-2):
    """返回局部最优分数、alignment 及两个半开原序列区间。"""
    n, m = len(x), len(y)
    # score[i][j]：以 x[i-1] 或 y[j-1] 附近结束的最佳局部分数。
    score = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best_score, best_pos = 0, (0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                (0, None),  # 负贡献前缀在此截断，重新开始局部比对。
                (
                    score[i - 1][j - 1]
                    + (match if x[i - 1] == y[j - 1] else mismatch),
                    "D",
                ),
                (score[i - 1][j] + gap, "U"),
                (score[i][j - 1] + gap, "L"),
            ]
            score[i][j], back[i][j] = max(
                candidates, key=lambda item: item[0]
            )
            if score[i][j] > best_score:
                best_score, best_pos = score[i][j], (i, j)

    # 局部回溯从全表最大值开始，在分数 0 处终止。
    ax, ay = [], []
    i, j = best_pos
    end_x, end_y = i, j
    while i > 0 and j > 0 and score[i][j] > 0:
        step = back[i][j]
        if step == "D":
            ax.append(x[i - 1])
            ay.append(y[j - 1])
            i -= 1
            j -= 1
        elif step == "U":
            ax.append(x[i - 1])
            ay.append("-")
            i -= 1
        else:
            ax.append("-")
            ay.append(y[j - 1])
            j -= 1

    return (
        best_score,
        "".join(reversed(ax)),
        "".join(reversed(ay)),
        (i, end_x),
        (j, end_y),
    )
```

返回的两个区间是半开区间 `(start, end)`，即 `x[start:end]` 与 `y[start:end]` 就是参与局部比对的子串。由于候选列表把 `(0, None)` 放在第一位，分数恰好为 0 时优先视为“重新开始”。

> [!EXAMPLE]+ Smith–Waterman 手算
>
> 对 `TGTTACGG` 与 `GTTGACTA` 使用 match $+3$、mismatch $-3$、linear gap $-2$，DP 表的最高分为 13。完整 DP 表如下（行为 `TGTTACGG`，列为 `GTTGACTA`）：
>
> | | - | G | T | T | G | A | C | T | A |
> | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
> | T | 0 | 0 | 3 | 3 | 1 | 0 | 0 | 3 | 1 |
> | G | 0 | 3 | 1 | 1 | 6 | 4 | 2 | 1 | 0 |
> | T | 0 | 1 | 6 | 4 | 4 | 3 | 1 | 5 | 3 |
> | T | 0 | 0 | 4 | 9 | 7 | 5 | 3 | 4 | 2 |
> | A | 0 | 0 | 2 | 7 | 6 | 10 | 8 | 6 | 7 |
> | C | 0 | 0 | 0 | 5 | 4 | 8 | 13 | 11 | 9 |
> | G | 0 | 3 | 1 | 3 | 8 | 6 | 11 | 10 | 8 |
> | G | 0 | 3 | 1 | 1 | 6 | 5 | 9 | 8 | 7 |
>
> 一条最优局部比对是：
>
> ```text
> GTT-AC
> GTTGAC
> ```
>
> 五个 match 得 $15$，一个 gap 扣 $2$，合计 $13$。回溯路径为 $13\to10\to7\to9\to6\to3\to0$：从 `C/C` 格点（分数 13）出发，经 `A/A`（10）、一个 $y$ 侧多出的 `G` 对 gap（$9-2=7$）、`T/T`（9）、`T/T`（6）、`G/G`（3），在首次到达 0 时停止，因此不会强制消费两端未参与局部同源区的字符（`x` 开头的 `T` 和结尾的 `GG`，`y` 结尾的 `TA`）。对应的原序列区间是 `x[1:6]="GTTAC"` 与 `y[0:6]="GTTGAC"`。

### 介于两者之间：半全局（overlap）比对

全局与局部之间还有 semi-global/overlap alignment。典型场景是把短 read 比对到一段参考序列：read 的每个碱基都应解释，但参考序列在 read 两侧多出来的部分不应罚分。若令 $x$ 为必须完整消费的短 read、$y$ 为两端免费的 reference，则：

- 初始化 $F_{0,j}=0$（read 可以从参考序列的任意位置开始，前面跳过的参考碱基不罚分），而 $F_{i,0}=ig$ 保持不变（read 的碱基不能免费跳过）；
- 终点取 $\max_jF_{n,j}$（read 必须消费完，但可以在参考序列的任意位置结束）；
- traceback 到第 0 行停止。

其他 overlap 变体（例如测序组装中两条 read 首尾重叠）才可能从最后一行或最后一列选择终点。算法主体相同，差异集中在初始化、终点与停止条件。

| 比对类型 | 边界初始化 | 终点 | traceback 停止 |
| --- | --- | --- | --- |
| 全局（NW） | $F_{i,0}=ig$，$F_{0,j}=jg$ | $(n,m)$ | 回到 $(0,0)$ |
| 局部（SW） | 全为 0，递推中含 0 | 全表最大值 | 遇到 0 |
| 半全局（read 对 reference） | $F_{0,j}=0$，$F_{i,0}=ig$ | $\max_jF_{n,j}$ | 到第 0 行 |

### 局部分数的统计分布

局部 alignment score 的统计分布与全局不同。在满足一定条件时（主要是随机对齐的期望分数为负、且至少存在一个正分），两条随机序列的最大局部得分近似服从极值分布（extreme value distribution，Gumbel 型），其尾部按 $e^{-\lambda S}$ 指数衰减。这正是 BLAST E-value 理论的基础，数据库搜索一章会给出具体公式。全局比对分数没有这种简洁的解析分布。

<br>

## 仿射 gap 罚分与 Gotoh 算法

### 动机：一次长 indel 比多次短 indel 更常见

线性罚分 $G(k)=kg$ 认为长度为 $k$ 的 gap 等价于 $k$ 个独立的单字符 indel。但在生物学上，一次插入或缺失事件常常一次性涉及多个连续残基（例如蛋白质表面的一个环区整体插入），连续插入/删除通常比多次独立 indel 更合理。因此希望“开一个新 gap”很贵、“把已有 gap 延长一格”相对便宜。仿射（affine）罚分写作

$$
G(k)=g_o+(k-1)g_e
$$

其中 $k\ge1$ 是 gap 长度，$g_o$ 是 gap-open（开第一个位置的代价），$g_e$ 是 gap-extension（之后每延长一格的代价），通常 $g_o>g_e\ge0$，这里两者写成非负的罚分，在分数中减去。

### 为什么需要三张表

仿射罚分破坏了单表 DP 的前提：在格点 $(i,j)$ 处，向下一步要扣 $g_o$ 还是 $g_e$，取决于上一步是不是已经在同一方向的 gap 中——这个信息单张 $F$ 表没有保存。解决办法是把“最后一列的类型”纳入状态。Gotoh 算法（1982）用 match、在 $x$ 中开 gap、在 $y$ 中开 gap 三张 DP 表，仍保持 $O(nm)$ 时间（常数约为原来的 3 倍）。

令 $M_{i,j}$ 表示最后一列为字符对，$I^x_{i,j}$ 表示最后一列为 $(x_i,-)$，$I^y_{i,j}$ 表示最后一列为 $(-,y_j)$。三者都是“$x_1\cdots x_i$ 与 $y_1\cdots y_j$ 的最优比对分数，但限定最后一列的类型”。一种约定下：

$$
M_{i,j}=\max(M_{i-1,j-1},I^x_{i-1,j-1},I^y_{i-1,j-1})+s(x_i,y_j)
$$

$$
I^x_{i,j}=\max(M_{i-1,j}-g_o,\ I^x_{i-1,j}-g_e)
$$

$$
I^y_{i,j}=\max(M_{i,j-1}-g_o,\ I^y_{i,j-1}-g_e)
$$

逐式解释：

- $M$：最后一列是 $(x_i,y_j)$，前一列可以是任何类型，所以对三张表在 $(i-1,j-1)$ 处取最大，再加替换分。
- $I^x$：最后一列是 $(x_i,-)$。要么前一列是字符对，现在新开一个 gap（扣 $g_o$）；要么前一列已经是 $(x_{i-1},-)$，现在延长（扣 $g_e$）。
- $I^y$ 与 $I^x$ 对称，沿行方向移动。

最终分数为 $\max(M_{n,m},I^x_{n,m},I^y_{n,m})$。长度为 $k$ 的 gap 在这套递推中恰好被扣 $g_o+(k-1)g_e$：第一格从 $M$ 进入扣 $g_o$，其余 $k-1$ 格自环扣 $g_e$。

### 罚分约定的差异

若定义 $G(k)=g_o+(k-1)g_e$，从 $M$ 开 gap 时扣 $g_o$，延长时扣 $g_e$。有些软件定义 $G(k)=g_o+kg_e$，会在开 gap 的第一格同时扣 extension（递推中开 gap 那项写成 $-g_o-g_e$）。例如 NCBI BLAST 的 “Existence 11, Extension 1” 采用后一种写法，长度为 1 的 gap 罚 $12$。参数数值不能脱离具体约定比较：同样写着“open 10、extend 1”，两种约定下长度为 $k$ 的 gap 相差 $1$ 分。

### 初始化

初始化时 $M_{0,0}=0$，不可能状态设为 $-\infty$（例如 $M_{i,0}$ 对 $i>0$ 不可能，因为没有 $y$ 字符可配对）；第一列只允许 $I^x$ 链，即 $I^x_{i,0}=-g_o-(i-1)g_e$，第一行只允许 $I^y$ 链，即 $I^y_{0,j}=-g_o-(j-1)g_e$。若错误地把所有边界设为 0，就会把 global alignment 变成 free-end-gap 版本——两端的 gap 不再罚分，这恰好是上一节半全局比对的做法，但不再是全局比对。

### 相反方向的相邻 gap

上述递推采用“不允许两个相反方向 gap 紧邻”的规范化 alignment：$I^x$ 不能直接转到 $I^y$，反之亦然。像下面这样一列 $(x_i,-)$ 紧跟一列 $(-,y_j)$：

```text
A-
-C
```

等价于“先插入再立即删除”，通常可以用一列 mismatch `A/C` 代替。常见生物序列实现禁止这种等价于插入后立即删除的冗余列组合；若评分体系允许这种路径并需枚举全部形式，则要显式加入 $I^x\leftrightarrow I^y$ 转移。

### 实现

下面是全局版 Gotoh 算法。注意参数 `gap_open`、`gap_extend` 是非负罚分，代码中减去；`back_m/back_x/back_y` 记录每个状态的前驱状态，traceback 时在三张表之间切换。

```python
def gotoh_global(
    x, y, match=1, mismatch=-1, gap_open=3, gap_extend=1
):
    """Gotoh 全局比对；长度 k 的 gap 扣 gap_open+(k-1)*gap_extend。"""
    if gap_open < 0 or gap_extend < 0:
        raise ValueError("gap penalties must be nonnegative")

    n, m = len(x), len(y)
    neg_inf = float("-inf")
    # M/X/Y 分别表示末列为字符对、(x_i, -)、(-, y_j)。
    M = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    X = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    Y = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    back_m = [[None] * (m + 1) for _ in range(n + 1)]
    back_x = [[None] * (m + 1) for _ in range(n + 1)]
    back_y = [[None] * (m + 1) for _ in range(n + 1)]
    M[0][0] = 0

    # 第一列只能是一条 X gap 链，第一行同理只能是一条 Y 链。
    for i in range(1, n + 1):
        X[i][0] = -gap_open - (i - 1) * gap_extend
        back_x[i][0] = "M" if i == 1 else "X"
    for j in range(1, m + 1):
        Y[0][j] = -gap_open - (j - 1) * gap_extend
        back_y[0][j] = "M" if j == 1 else "Y"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = match if x[i - 1] == y[j - 1] else mismatch

            # M 可由任一前态对角转移，并消费两个字符。
            prev_m = [(M[i - 1][j - 1], "M"),
                      (X[i - 1][j - 1], "X"),
                      (Y[i - 1][j - 1], "Y")]
            best, back_m[i][j] = max(prev_m, key=lambda item: item[0])
            M[i][j] = best + substitution

            # X/Y 分别比较新开 gap 与延长已有 gap。
            choices_x = [(M[i - 1][j] - gap_open, "M"),
                         (X[i - 1][j] - gap_extend, "X")]
            X[i][j], back_x[i][j] = max(
                choices_x, key=lambda item: item[0]
            )
            choices_y = [(M[i][j - 1] - gap_open, "M"),
                         (Y[i][j - 1] - gap_extend, "Y")]
            Y[i][j], back_y[i][j] = max(
                choices_y, key=lambda item: item[0]
            )

    score, state = max(
        [(M[n][m], "M"), (X[n][m], "X"), (Y[n][m], "Y")],
        key=lambda item: item[0],
    )
    ax, ay, i, j = [], [], n, m
    while i > 0 or j > 0:
        if state == "M":
            previous = back_m[i][j]
            ax.append(x[i - 1])
            ay.append(y[j - 1])
            i -= 1
            j -= 1
        elif state == "X":
            previous = back_x[i][j]
            ax.append(x[i - 1])
            ay.append("-")
            i -= 1
        else:
            previous = back_y[i][j]
            ax.append("-")
            ay.append(y[j - 1])
            j -= 1
        state = previous

    return score, "".join(reversed(ax)), "".join(reversed(ay))
```

> [!EXAMPLE]+ 例：仿射罚分偏好一个长 gap
>
> 用 `gotoh_global("ACGTTTACG", "ACGACG", 2, -1, 3, 1)`（match $+2$、mismatch $-1$、$g_o=3$、$g_e=1$）比对，得到
>
> ```text
> ACGTTTACG
> ACG---ACG
> ```
>
> 六个 match 得 $12$，一个长度为 3 的 gap 扣 $G(3)=3+2\times1=5$，总分 $7$。如果改成三个分散的单字符 gap，需扣 $3\times3=9$，因此仿射罚分会把 indel 集中成一个连续区块，这正是设计它的目的。

<br>

## Pair HMM：比对的概率解释

### 三张表就是三个隐状态

到目前为止，替换分数和 gap 罚分都是“凭经验设定的数字”。Pair HMM 给它们一个概率含义：假设存在一个随机过程，一边在隐状态之间跳转，一边同时“吐出”两条序列。Gotoh 的 $M,X,Y$ 三张表也可解释为 pair HMM 的三个隐状态。$M$ 同时发射 $(x_i,y_j)$，$X$ 发射 $(x_i,-)$，$Y$ 发射 $(-,y_j)$。一条比对就是一条隐状态路径，其概率是沿途转移概率与发射概率的乘积。HMM 的一般理论（Forward、Viterbi 等）见隐马尔可夫模型一章。

### 从概率到分数：log-odds

把相关序列 pair-HMM 与独立背景模型（两条序列各自独立地按背景频率 $q_a$ 随机生成）做 log-odds，即取 $\log\frac{P(x,y,\text{path}\mid\text{相关})}{P(x,y\mid\text{随机})}$，乘积变成求和，可得到：

- $M$ 的 emission log-odds 对应 substitution score：$\log\frac{p_{ab}}{q_aq_b}$，这正是下一节打分矩阵的形式；
- $M\rightarrow X/Y$ 的 transition log-odds 对应 gap-open；
- $X\rightarrow X$、$Y\rightarrow Y$ 对应 gap-extension；
- begin/end transition 与允许的入口、出口共同决定 global、local 或 semi-global 边界（局部版本在两端加入发射背景字符的随机状态，相当于“免费跳过”的前后缀）。

在 Durbin 等人的推导中，若 $M\to X$ 的概率为 $\delta$、$X\to X$ 的概率为 $\varepsilon$，则 gap-open 罚分约为 $-\log\frac{\delta(1-\varepsilon)}{1-2\delta}$，extension 罚分约为 $-\log\varepsilon$。这说明 $\varepsilon$ 越接近 1，延长 gap 越便宜，与“长 indel 是一次事件”的直觉一致。

### Viterbi 与全部路径之和

Gotoh 使用 `max`，因此求得最可能 alignment path，即 Viterbi alignment。若把每个状态转移的 `max` 改为 log-sum-exp（即 $\log\sum e^{(\cdot)}$，在概率空间中相当于用求和代替取最大），则得到所有 alignment 路径的总 likelihood，即 Forward 算法给出的 $P(x,y)$。两条序列可能存在许多近似等价 alignment，Viterbi 只保留其中一条；posterior alignment 可进一步计算某对残基同源对应的边缘概率 $P(x_i\diamond y_j\mid x,y)$（结合 Forward 与 Backward），从而知道比对的哪些列可靠、哪些列只是众多同分方案中的一个。

### 结构对应不等于合法概率模型

分数到概率参数并非逐项取指数即可完成。Transition row 必须归一化（从一个状态出发的所有转移概率之和为 1），emission 也必须相对明确的 null model；常数缩放与终止概率会影响概率模型，却可能不改变最优路径。因此“仿射 gap 像 HMM”是结构对应关系，不表示任意经验打分参数天然构成合法概率分布。例如把所有分数乘以 2 不改变 Viterbi 路径，但对应的“概率”在指数化后不再归一。

<br>

## 打分矩阵：替换分数从哪里来

### 动机

DNA 可用 match/mismatch；蛋白质需利用替换频率。20 种氨基酸的理化性质差别很大：亮氨酸（L）换成异亮氨酸（I）几乎不影响结构，换成脯氨酸（P）却可能破坏螺旋。因此蛋白比对需要一个 $20\times20$ 的替换矩阵，告诉算法每种替换“在演化中有多常见”。PAM 从近缘序列估计演化替换再外推，BLOSUM 从保守区块直接统计。

### 最简单的 DNA 矩阵

最简单的 DNA identity matrix 在 A/T/C/G 相同配对上取 1，不同配对上取 0。它适合演示 DP 填表，却没有惩罚 mismatch，也没有表达 transition（嘌呤↔嘌呤、嘧啶↔嘧啶，如 A↔G）、transversion（嘌呤↔嘧啶）或背景碱基频率差异。实际比对通常使用正 match、负 mismatch，并与 gap penalty 联合设定。

### 矩阵元素是 log-odds

矩阵元素本质是 log-odds：

$$
s(a,b)=c\log\frac{p_{ab}}{q_aq_b}
$$

其中 $p_{ab}$ 是在真实同源比对中观察到 $a$ 与 $b$ 对齐的频率，$q_a,q_b$ 是背景频率，$q_aq_b$ 是“两条无关序列偶然在这里对上 $a$ 和 $b$”的概率。比值大于 1（$s>0$）说明该配对在相关序列中比随机背景更常见；比值小于 1（$s<0$）说明更少见。$c>0$ 决定分数单位与取整尺度，不应预先把它等同于后文 Karlin–Altschul 统计参数 $\lambda$。例如 BLOSUM62 以“半 bit”为单位，即 $s=2\log_2\frac{p_{ab}}{q_aq_b}$ 再四舍五入成整数（不同 BLOSUM 矩阵的单位不完全相同，如 BLOSUM50 为 1/3 bit）。正分表示该配对在相关序列中比随机背景更常见。BLOSUM62 是常用默认值，不代表所有演化距离都最优。

一个直观现象：BLOSUM62 中 W–W（色氨酸自身配对）为 11，L–L 只有 4。原因是色氨酸罕见，$q_W$ 很小，偶然对上的概率低，所以一旦在两条序列中对上，提供的证据更强。

### PAM 与 BLOSUM 的构造与编号

- **PAM（Point Accepted Mutation，Dayhoff）**：PAM1 近似每 100 个残基发生 1 个 accepted point mutation（“accepted”指被自然选择保留下来的突变），从非常近缘的序列比对估计；再通过矩阵幂外推较远距离（PAM250 = PAM1 的 250 次幂对应的转移概率再取 log-odds）；编号越大代表距离越远。
- **BLOSUM（BLOcks SUbstitution Matrix，Henikoff & Henikoff 1992）**：BLOSUM 从保守 block（BLOCKS 数据库中无 gap 的保守区段）聚类统计。BLOSUM$N$ 先把 identity 不低于 $N\%$ 的序列聚成一簇并合并计数，以免近缘序列主导统计。编号越低通常代表把相似度较低的序列合并、面向更远关系。

两套编号方向相反：PAM 编号越大越适合远缘，BLOSUM 编号越小越适合远缘。实践中，BLOSUM45 或 PAM250 一类矩阵用于远缘搜索，BLOSUM80 一类用于近缘序列；NCBI 还为很短的 query 提供 PAM30、PAM70。

### 矩阵与 gap 必须联合校准

替换矩阵与 gap penalty 必须联合校准。若平均随机 alignment score 非负，局部比对可以通过无限延长随机区域提高分数，Karlin–Altschul 的标准条件不再成立，局部比对实际上退化成全局比对，E-value 也失去意义。这也是为什么 BLAST 只为每个矩阵提供有限几组允许的 gap 参数组合：每组组合的 $\lambda,K$ 都需要事先估计。

> [!EXAMPLE]+ 例3：比对分数不等于概率
>
> 同一对序列使用不同矩阵、gap 参数或局部/全局目标，会得到不同最优比对。原始分数只有在指定评分系统后才有意义：BLOSUM62 下的 50 分与 PAM250 下的 50 分不能直接比较。跨搜索比较通常使用 bit score 和 E-value，它们把评分系统的尺度差异标准化掉，见数据库搜索一章。

<br>

## 正确性检查与常见陷阱

比对代码很容易“能跑但不对”：边界差一、traceback 方向写反、gap 约定不一致，结果仍是一条看起来合理的比对。下面这些不变量（invariant）可以写成单元测试，快速发现大部分错误：

- `dp[i][j]` 只依赖已计算的左、上、左上，填表顺序必须保证这一点（按行或按反对角线都可以）；
- traceback 每步至少减少 $i$ 或 $j$，必然终止；
- 重算 traceback alignment 的列分数应等于终点 DP 值——这是最有效的单项检查，能同时暴露递推和回溯的错误；
- alignment 去掉 gap 后应恢复原序列（局部比对则应恢复报告区间内的子串）；
- 交换两序列时，若评分对称（$s(a,b)=s(b,a)$），最优分数应相同；
- 将所有分数乘正常数不会改变最优路径；
- local score 不应为负，因为空比对得分为 0。

成熟库与手写实现结果不同时，应先核对 end-gap（两端 gap 是否罚分）、gap convention（$g_o+(k-1)g_e$ 还是 $g_o+kg_e$）、matrix alphabet（是否包含 `B`、`Z`、`X`、`*` 等字符）、unknown residue（`N`、`X` 如何打分）和 tie-breaking，而不是立即判断其中一个错误。

> [!NOTE]+ 小结
>
> - 比对是演化对应假设；相似性可计算，同源性是“是或否”的结论。
> - 所有比对对应网格路径，加性评分保证最优子结构，DP 用 $O(nm)$ 时间求最优。
> - 全局、局部、半全局只在初始化、终点和 traceback 停止条件上不同，选择由生物问题决定。
> - 仿射 gap 通过 $M,I^x,I^y$ 三状态恢复加性，罚分约定（是否在开 gap 时也扣 extension）必须写清。
> - 替换矩阵是 log-odds；PAM 与 BLOSUM 编号方向相反；矩阵与 gap 需联合校准。
> - Pair HMM 把三状态 DP 解释为概率模型：`max` 得 Viterbi 路径，log-sum-exp 得全部路径的 likelihood。

<br>

<br>

# 序列数据库搜索与 BLAST

拿到一条新蛋白序列，最常做的第一件事是把它扔进 NCBI 的 BLAST，看看数据库里有没有相似的已知蛋白。这个问题和上一章的双序列比对不同之处在于规模：一条 query 要和数据库中上亿条序列（数千亿个残基）比较。数据库搜索是“一条 query 对大量候选”的相似性检索；对数据库中的每条序列完整运行 Smith–Waterman 虽然精确，但成本通常不可接受。

核心想法是“先粗筛、再精算”：真正相关的两条序列，其局部比对中几乎总包含若干短的高分片段。BLAST（Basic Local Alignment Search Tool，Altschul et al. 1990）先用短 seed 找可能相关区域，只在候选附近延伸，以牺牲部分敏感度换取速度，并未另行定义一种相似性——它近似的仍是 Smith–Waterman 的局部比对分数。

本章承接局部比对，并新增两个概念：“搜索空间”（搜了多少序列）和“零模型”（随机序列能得多少分）。有了它们，才能判断一个分数是否“高到不像偶然”。搜索得到的命中序列可进入多序列比对、profile 或结构模板搜索，因此本章也是下一章多序列比对的数据来源。

## 搜索的输入、输出与解释边界

### 输入与输出

一次数据库搜索的输入包括 query、数据库版本、替换矩阵、gap 参数、masking 和显著性阈值。输出是若干局部命中（hit），每个命中带 alignment、bit score、E-value、identity 和 coverage。这些量回答不同的问题：

- 原始分数描述当前打分体系下的相似程度；
- E-value 回答给定搜索空间中随机出现同等高分结果的期望数量；
- identity 是比对列中相同残基的比例，coverage 是比对覆盖 query（或 subject）的比例。

### 显著不等于功能相同

低 E-value 支持非随机相似，却不能单独证明功能相同、正交同源（orthology）或相同结构域架构（domain architecture）。例如两个蛋白共享一个常见的激酶结构域，E-value 可以极低，但其他结构域和底物可能完全不同；按“最佳命中”直接转移功能注释，是数据库注释错误传播的主要来源之一。

<br>

## BLAST 的 seed-and-extend 流程

### 完整执行链

BLAST 采用 seed-and-extend（播种—延伸）策略。把课程中的执行链完整展开，可写成：

1. 对 query 做 low-complexity masking（例如蛋白用 SEG、DNA 用 DUST，把 `QQQQQQ` 这类低复杂度区换成 `X` 或 `N`），避免它们产生大量无意义命中；
2. 从 query 枚举长度为 $k$ 的 word（蛋白默认 $k=3$，`blastn` 默认 11，`megablast` 默认 28）；
3. 对蛋白质 query，用替换矩阵生成分数超过阈值 $T$ 的邻居词（neighborhood words），例如 query 词 `PQG` 与 `PEG` 的 BLOSUM62 分数若不低于 $T$，`PEG` 也算作种子；
4. 建立 query word lookup table，某些实现或模式也可预建数据库索引；
5. 在数据库中寻找 word hit（数据库中出现了表中的某个词）；
6. 先做 ungapped extension（沿对角线向两侧不加 gap 地延伸），形成 high-scoring segment pair（HSP）；
7. 用 Karlin–Altschul 零模型筛除不显著 HSP；
8. 合并或连接相容的邻近 HSP，并做 gapped extension；
9. 报告局部比对、bit score、E-value、identity 与 coverage。

最后的 gapped alignment 可视为只在 seed 支持区域附近执行受限的 Smith–Waterman 式搜索。不同 BLAST 版本的具体阶段、two-hit 规则和统计修正并不完全相同，九步链用于理解启发式筛选，不应误当所有实现逐行一致的规范。

### 关键参数：$T$、two-hit 与 $X$-drop

原始 BLAST 对蛋白 query 枚举超过阈值 $T$ 的相似 word，而不是只找完全相同 word。$T$ 控制速度与敏感度的权衡：$T$ 越低，邻居词越多，种子越多，敏感度越高，但延伸工作量越大。

现代实现（Gapped BLAST，Altschul et al. 1997）有 two-hit heuristic：同一对角线上短距离内（窗口 $A$，蛋白默认约 40 个残基）出现两个 seed 后才延伸，从而减少随机 seed 的开销。这样可以把 $T$ 调低而不让延伸成本爆炸，因为真正的同源区域往往在同一对角线上有多个种子，而随机种子很少成对出现。

ungapped extension 的分数从峰值下降超过阈值 $X$ 时停止（$X$-drop 规则）：一旦当前累计分数比目前见过的最高分低出 $X$，就认为后面不会再回升，停止延伸，并截回到峰值位置。通过筛选后再做 gapped extension，gapped 阶段同样使用（更宽松的）$X$-drop 限制 DP 的探索范围。

<br>

## Karlin–Altschul 统计：E-value 与 bit score

### E-value 公式

一个比对分数 50 到底算高还是低？这取决于评分系统，也取决于搜了多大的数据库——搜得越多，偶然出现高分的机会越大。Karlin–Altschul 理论给出

$$
E=Kmn e^{-\lambda S}
$$

各符号的含义：

- $m,n$ 是经过 edge correction 的有效 query 长度与有效数据库搜索空间，并不总等于原始字符数（靠近序列末端的位置没有足够长度形成高分比对，需要扣除一段“有效长度”）；
- $S$ 是原始分数；
- $K,\lambda$ 由评分体系决定（替换矩阵、gap 参数、背景频率）。对无 gap 比对，$\lambda$ 是方程 $\sum_{a,b}q_aq_be^{\lambda s(a,b)}=1$ 的唯一正根；对有 gap 比对，$\lambda,K$ 只能通过随机序列模拟估计，这也是 BLAST 只支持有限几组 gap 参数的原因。

E-value 是随机数据库搜索中分数至少这么高的期望命中数，不是“结果错误的概率”。从公式可以读出两点：$E$ 与搜索空间 $mn$ 成正比，所以数据库越大，同一原始分数的 E-value 越差（数据库扩大 10 倍，E-value 大约变差 10 倍）；$E$ 随 $S$ 指数下降，所以分数每多一点，显著性提升很快。

### Bit score

不同矩阵下的原始分数不可比（BLOSUM62 的 50 分与 PAM250 的 50 分含义不同）。bit score 把不同评分体系标准化：

$$
S'=\frac{\lambda S-\ln K}{\ln 2},
\qquad E=mn\,2^{-S'}
$$

推导很简单：由第一个式子得 $\lambda S=S'\ln2+\ln K$，代入 $E=Kmne^{-\lambda S}=Kmn\,e^{-S'\ln2}/K=mn\,2^{-S'}$。于是 $K,\lambda$ 都被吸收进 $S'$，只要知道 bit score 和搜索空间就能算 E-value，而且 bit score 每增加 1，E-value 减半。

### 如何正确解读 E-value

$E=0.01$ 表示每次搜索的随机高分命中期望数为 0.01；在 Poisson 近似下，随机高分命中数服从均值为 $E$ 的 Poisson 分布，至少出现一个随机命中的概率为 $P=1-e^{-E}$，代入得约 $1-e^{-0.01}\approx0.00995$。当 $E$ 很小时 $P\approx E$，但 $E$ 可以大于 1 而 $P$ 不能。需要强调，这个数不是“该结果为假的概率”，也不校正数据库 annotation error：一个统计上极显著的命中，其注释本身可能就是错的。

### identity、coverage 与 E-value 各回答什么

identity、positive（替换分为正的列所占比例）、coverage 和 E-value 回答不同问题。短片段可以有 100% identity 却覆盖极少；长 alignment 可能 identity 不高但统计上极显著——蛋白序列在约 20%–35% identity 的区间常被称为“暮光区”（twilight zone），此时仅凭 identity 很难判断，长度和 E-value 更有参考价值。判断 domain-level homology 时通常要求 query/subject coverage 与 domain boundary 一并检查。

<br>

## 提高敏感度与报告规范

### 敏感度的几个杠杆

BLAST 的速度来自“跳过大部分搜索空间”，代价是会漏掉一些远缘同源。常见的调节手段和相关方法如下：

- seed 越短（或 $T$ 越低），敏感度越高、候选越多，速度越慢；
- low-complexity masking 防止重复区域支配结果，它降低假阳性，而不是提高敏感度；
- PSI-BLAST 从迭代命中构建 position-specific scoring matrix（PSSM），让每个位置有自己的替换偏好；
- profile HMM（如 HMMER、Pfam）通常比单一 query 更能检出远缘同源，因为它利用了整个家族的位置特异信息，详见隐马尔可夫模型一章；
- reciprocal best hit（A 在 B 物种中的最佳命中是 B，B 在 A 物种中的最佳命中也是 A）只是 ortholog 的启发式证据，不处理复杂复制与丢失历史（例如基因复制后的旁系同源 paralog）。

### 五种 BLAST 程序

`blastn`、`blastp`、`blastx`、`tblastn` 与 `tblastx` 的 query/database 字母表及翻译方向不同，搜索空间也不同：

| 程序 | query | database | 说明 |
| --- | --- | --- | --- |
| `blastn` | 核酸 | 核酸 | 直接比对核酸 |
| `blastp` | 蛋白 | 蛋白 | 直接比对蛋白 |
| `blastx` | 核酸（六框翻译） | 蛋白 | 用于注释未知 DNA 片段可能编码的蛋白 |
| `tblastn` | 蛋白 | 核酸（六框翻译） | 在基因组或 EST 中寻找编码某蛋白的区域 |
| `tblastx` | 核酸（六框翻译） | 核酸（六框翻译） | 两边都翻译，计算量最大 |

蛋白水平的比较通常比 DNA 水平更敏感，因为同义突变在蛋白层面不可见，而替换矩阵能利用氨基酸的相似性。蛋白搜索还可采用 composition-based score adjustment，减少氨基酸组成偏斜造成的虚假高分。报告中还应记录程序类型、矩阵、gap 参数与有效搜索空间。

### PSI-BLAST 与 profile drift

PSI-BLAST 从第一轮命中构建 PSSM，再用 PSSM 重复搜索。远缘 homolog 可逐轮加入：第一轮只找到近缘序列，它们构成的 profile 捕获了家族的保守模式，第二轮便能检出与 query 本身相似度不高、但符合家族模式的序列。风险在于，一次 false positive 也可能污染 profile 并发生 profile drift——profile 逐渐偏向另一个家族，之后的命中越来越不相关。每轮纳入阈值通常应比最终报告阈值更严格（NCBI 网页版的默认纳入阈值为 0.005），并人工检查 domain architecture。

### 应记录什么

实际工作中应记录 BLAST 版本、数据库版本、E-value 阈值、覆盖率、identity、masking 和过滤规则。数据库在不断增长，同一个分数在不同时间搜索得到的 E-value 不同，因此只有记下这些参数，结果才可复现。

> [!EXAMPLE]+ 例4：两个显著命中
>
> 命中 A 覆盖 query 的 95%，identity 32%；命中 B identity 80%，却只覆盖一个 20-residue low-complexity segment。若只按 identity 排序会选 B；综合 coverage、masking、domain 与 E-value 后，A 往往更能支持全长同源。B 的高 identity 很可能来自组成偏斜的低复杂度区，开启 masking 后它甚至可能消失；A 虽然 identity 处于暮光区，但跨越几乎整条 query 的长比对很难由偶然产生。

> [!NOTE]+ 小结
>
> - BLAST = 低复杂度屏蔽 + 短词种子 + ungapped 延伸 + 显著性筛选 + gapped 延伸，近似 Smith–Waterman。
> - $T$、two-hit 窗口和 $X$-drop 控制速度与敏感度的权衡。
> - $E=Kmne^{-\lambda S}$ 是随机期望命中数，随数据库增大而变差；bit score 消去 $K,\lambda$，使 $E=mn\,2^{-S'}$。
> - E-value、identity、coverage 回答不同问题；显著相似不等于功能相同或直系同源。
> - PSI-BLAST 和 profile HMM 提高远缘敏感度，但需警惕 profile drift。

<br>

<br>

# 多序列比对（Multiple Sequence Alignment）

双序列比对只能告诉读者两条序列怎样对应。但很多生物问题天然涉及一整个家族：哪些位点在所有物种中都不变（很可能是功能位点）？哪些位置成对协同变化（可能在三维结构中接触）？要建系统发育树，也需要所有序列共享同一套“位点”。多序列比对（MSA）为三条以上同源序列建立共同列坐标：某一列中的残基被解释为来自同一祖先位置，gap 表示相对该坐标系发生的插入或缺失。

MSA 不是把若干 pairwise alignment 简单并排，因为每次插入 gap 都必须同步更新已有序列的列结构：A–B 比对和 A–C 比对可能在 A 上插入了不同位置的 gap，两者无法直接拼成一个三行矩形。核心困难是计算量：精确求解是多维动态规划，随序列数指数增长。因此实际算法几乎都是近似的——先合并近邻 sequence/profile，再用 refinement 修正早期决策。

本章承接双序列比对（DP 递推、gap 罚分）和数据库搜索（收集同源序列），其输出又是后续 motif（PWM/PSSM）、Profile HMM、系统发育和结构预测各章的输入。

## 问题设定与解释边界

### 输入与输出

输入是一组序列，常伴随 pairwise distance 或 guide tree。输出是带 gap 的矩形 alignment（每行一条序列、每列一个同源位置）、列分布（每列各残基的频率），以及不确定区域（哪些列的对应关系不可靠）。MSA 支持保守位点识别、PWM/PSSM、Profile HMM、系统发育、共进化和结构预测。精确多维 DP 随序列数指数增长，因此实际算法通常先合并近邻 sequence/profile，再用 refinement 修正早期决策。

### 解释边界

- Guide tree 只规定渐进合并顺序，不自动等于物种树；它通常由粗糙的距离和 UPGMA 等简单方法得到，不应作为系统发育结论报告。
- MSA 的列对应不是无误差标签，而是在某个评分模型和启发式下的估计。
- 下游建树、功能位点和结构约束都应检查不稳定列对结论的影响，例如换一个比对工具或去掉不稳定列后结论是否仍成立。

<br>

## 为什么精确 MSA 不可行：复杂度与渐进比对

### 多维动态规划的代价

把双序列比对的网格推广到 $k$ 条序列，就是一个 $k$ 维超立方体网格，每个格点 $(i_1,\dots,i_k)$ 表示各序列分别消耗了多少字符。每一列可以让任意非空子集的序列前进一步（其余填 gap），因此每个格点有 $2^k-1$ 个前驱。对 $k$ 条长度约为 $n$ 的序列，朴素精确 DP 时间为 $O((2^k-1)n^k)$、空间为 $O(n^k)$。以 $k=10$、$n=300$ 为例，$n^k\approx5.9\times10^{24}$ 个格点，完全不可行。在 sum-of-pairs 目标下，精确 MSA 已被证明是 NP-hard 问题。

### 渐进比对（progressive alignment）

常用工具因此采用 progressive alignment：

1. 计算两两距离；
2. 建 guide tree；
3. 按树从近到远合并序列或 profile；
4. 可再迭代优化。

直觉是：最相似的两条序列最容易比对正确，先把它们对齐并“冻结”，再逐步加入较远的序列。Clustal 系列（ClustalW、Clustal Omega）体现 progressive 思路，MUSCLE 加入快速距离估计（$k$-mer 计数）和 refinement，MAFFT 利用 FFT（快速傅里叶变换，用于快速找出同源片段）与不同策略平衡速度和精度。

其根本弱点是“once a gap, always a gap”：早期错误会被 profile 固化，后加入的序列无法修改已经合并好的列。因此 guide tree 不是最终系统发育树，而只是一个合并顺序，它的质量直接影响比对质量。

### Profile 与 profile–profile 打分

渐进合并时，一边可能已经是一组对齐好的序列（profile），所以需要“两列之间”的打分。profile–profile alignment 不再比较两个字符，而是比较两列分布。若列 $i,j$ 的 residue frequency 分别为 $p_i(a)$、$q_j(b)$（$a,b$ 取遍字母表，$\sum_ap_i(a)=1$），期望 substitution score 可写为

$$
s(i,j)=\sum_{a,b}p_i(a)q_j(b)S(a,b)
$$

其中 $S(a,b)$ 是普通的替换矩阵（如 BLOSUM62）。这就是“从一列里随机抽一个残基、从另一列里随机抽一个残基，二者替换分的期望”。当两边各只有一条序列时，它退化为普通的 $S(x_i,y_j)$。

实际中还需对 gap-rich column、sequence weight 和 pseudocount 处理：gap 很多的列要降低其开 gap 的代价（位置特异 gap 罚分）；pseudocount 避免小样本中未观察到的残基概率为 0；sequence weighting 防止某个过度采样物种支配 profile——例如家族中有 20 条几乎相同的人类旁系同源和 1 条细菌序列，不加权时 profile 几乎只反映人类序列。

### 课程复习材料中的 “synthetic sequence”

课程复习材料把 progressive MSA 概括为“pairwise alignment、聚类、synthetic sequence、重新比对”。这里需要澄清几点：

- synthetic sequence 更准确地指由已对齐序列形成的 profile 或 consensus-like 表示，并非伪造一条具有真实生物来源的序列；
- 只取单一 consensus（每列取最常见残基）会丢失列内变异，例如一列中 50% 为 L、50% 为 I 的信息被压成一个字母；
- 实际 profile–sequence 或 profile–profile DP 保留每列 residue frequency、gap frequency 和权重；
- Henikoff weighting（按每列中残基的稀有程度给序列加权）、基于 guide tree 的 weighting（ClustalW 按树上分支长度分配权重）等方法用于降低近重复序列的支配作用。

### 渐进比对的详细链条

把上面的步骤展开成实现层面的流程：

1. 用 $k$-mer 或快速 pairwise score 得到距离矩阵；
2. 用 UPGMA/NJ（邻接法，见系统发育一章）建 guide tree；
3. 叶节点是 sequence profile（只含一条序列的 profile）；
4. 内节点按 pairwise DP 对齐两个 profile；
5. 已对齐 block 作为整体传播到更高节点，内部的列对应不再改变；
6. refinement 拆开部分 alignment 后重新对齐，尝试修复早期错误（例如沿 guide tree 的某条边把序列分成两组，分别作为 profile 重新比对，若分数提高则接受）。

若有已知结构或可靠 motif，可加入 constraint（例如 T-Coffee 的 Expresso 模式利用结构信息）。对于全局同源但有长插入的序列，MAFFT L-INS-i 一类局部 pairwise 信息可能更准确，但成本更高，通常只适合几百条以内的序列。

### 教学实现

下面的教学实现完成一次 profile–profile DP，并按给定 guide order 渐进合并。它直接优化未加权 sum-of-pairs：两列之间的分数等于“左组每条序列 × 右组每条序列”的字符对分数之和，gap 对 gap 记 0（因此组内原有的列对应在合并时分数不变）。成熟软件还会加入 sequence weight、位置特异 gap、局部 alignment 与 refinement。

```python
def profile_profile_align(
    left, right, match=1, mismatch=-1, gap=-2
):
    """合并两个已有 alignment；每个输入都是等长字符串列表。"""
    if not left or not right:
        raise ValueError("both profiles must contain sequences")
    if len({len(seq) for seq in left}) != 1:
        raise ValueError("left profile is not aligned")
    if len({len(seq) for seq in right}) != 1:
        raise ValueError("right profile is not aligned")

    # 把 alignment 转置为列；每列保留所有成员的字符。
    left_cols = list(zip(*left))
    right_cols = list(zip(*right))
    n, m = len(left_cols), len(right_cols)

    def pair_score(a, b):
        if a == "-" and b == "-":
            return 0
        if a == "-" or b == "-":
            return gap
        return match if a == b else mismatch

    def column_score(column_a, column_b):
        # profile 列分数是两个组之间所有序列对分数之和。
        return sum(pair_score(a, b) for a in column_a for b in column_b)

    gap_left = tuple("-" for _ in left)
    gap_right = tuple("-" for _ in right)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + column_score(
            left_cols[i - 1], gap_right
        )
        back[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + column_score(
            gap_left, right_cols[j - 1]
        )
        back[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (
                    dp[i - 1][j - 1]
                    + column_score(left_cols[i - 1], right_cols[j - 1]),
                    "D",
                ),
                (
                    dp[i - 1][j]
                    + column_score(left_cols[i - 1], gap_right),
                    "U",
                ),
                (
                    dp[i][j - 1]
                    + column_score(gap_left, right_cols[j - 1]),
                    "L",
                ),
            ]
            dp[i][j], back[i][j] = max(
                choices, key=lambda item: item[0]
            )

    # 每次回溯都向两组所有序列同步加入一列，保持旧 alignment。
    merged_columns = []
    i, j = n, m
    while i or j:
        step = back[i][j]
        if step == "D":
            merged_columns.append(left_cols[i - 1] + right_cols[j - 1])
            i -= 1
            j -= 1
        elif step == "U":
            merged_columns.append(left_cols[i - 1] + gap_right)
            i -= 1
        else:
            merged_columns.append(gap_left + right_cols[j - 1])
            j -= 1

    merged_columns.reverse()
    merged = [
        "".join(column[row] for column in merged_columns)
        for row in range(len(left) + len(right))
    ]
    return dp[n][m], merged


def progressive_msa(sequences, **score_parameters):
    """按输入顺序形成固定 guide chain；仅用于展示 progressive 机制。"""
    if not sequences:
        return []
    profile = [sequences[0]]
    for sequence in sequences[1:]:
        _, profile = profile_profile_align(
            profile, [sequence], **score_parameters
        )
    return profile


def progressive_msa_with_upgma(sequences, distance, **score_parameters):
    """用预先计算的 pairwise distance 建 UPGMA guide tree 并合并 profile。

    sequences 是 {taxon: sequence}；distance 是同一 taxon 集合的嵌套字典。
    """
    active = list(sequences)
    if set(distance) != set(active) or not active:
        raise ValueError("distance and sequence taxa must match")
    profiles = {taxon: [sequence] for taxon, sequence in sequences.items()}
    size = {taxon: 1 for taxon in active}
    dist = {
        frozenset((left, right)): float(distance[left][right])
        for i, left in enumerate(active)
        for right in active[i + 1:]
    }
    next_id = 0

    while len(active) > 1:
        left, right = min(
            (
                (active[i], active[j])
                for i in range(len(active))
                for j in range(i + 1, len(active))
            ),
            key=lambda pair: dist[frozenset(pair)],
        )
        _, merged = profile_profile_align(
            profiles[left], profiles[right], **score_parameters
        )
        new = ("guide_cluster", next_id)
        next_id += 1
        profiles[new] = merged
        size[new] = size[left] + size[right]

        others = [node for node in active if node not in (left, right)]
        for node in others:
            dist[frozenset((new, node))] = (
                size[left] * dist[frozenset((left, node))]
                + size[right] * dist[frozenset((right, node))]
            ) / size[new]
        active = others + [new]

    return profiles[active[0]]
```

`progressive_msa` 的输入顺序承担了 guide tree 的作用，因此不能用于方法评价：换一个输入顺序，结果可能不同。完整流程应先由 pairwise distance 构建 guide tree，再按内部节点调用 `profile_profile_align`；同一个 UPGMA tree 可用于指导 MSA，但 guide tree 是计算顺序，不应直接报告为物种树。

`progressive_msa_with_upgma` 把这条链实际连通：pairwise distance 决定下一次合并的 profile（每次取当前距离最小的一对簇），profile–profile DP 决定列对应，合并后的簇再按成员数更新距离——新簇到其他簇的距离是两个子簇距离按成员数的加权平均，这正是 UPGMA 的更新规则。初始距离可以来自快速 $k$-mer、编辑距离或 pairwise alignment；若来自未对齐序列，不能直接使用要求同源列的 p-distance（不同位点比例），因为 p-distance 必须先有比对才能定义。最终 alignment 改变后可重新估计距离和 guide tree，这就是 iterative refinement 的一个来源（MUSCLE 的第二阶段正是这样做的）。

> [!EXAMPLE]+ 例：三条短序列的渐进合并
>
> 使用默认参数（match $+1$、mismatch $-1$、gap $-2$），`progressive_msa(["ACGT", "ACT", "AGGT"])` 先把 `ACGT` 与 `ACT` 对齐为 `ACGT` / `AC-T`，再把 `AGGT` 加入这个两行 profile，得到
>
> ```text
> ACGT
> AC-T
> AGGT
> ```
>
> 若改用 `progressive_msa_with_upgma`，并给定距离 $d(a,b)=1$、$d(a,c)=d(b,c)=4$（$a,b,c$ 分别对应上面三条序列），则先合并最近的 $a,b$，新簇到 $c$ 的距离为 $(1\times4+1\times4)/2=4$，再与 $c$ 合并。最终列对应相同，只是行顺序变为 `AGGT`、`ACGT`、`AC-T`（因为新簇被追加在 `active` 列表末尾）。可以看到，第二步加入 `AGGT` 时不能再移动 `AC-T` 中已经插入的 gap，这正是“once a gap, always a gap”。

<br>

## 如何评价一个多序列比对

### 优化目标：sum-of-pairs

sum-of-pairs（SP）分数对每一列中的所有序列对求和：

$$
\text{SP}=\sum_{\text{列 }c}\ \sum_{p<q}s\big(c_p,c_q\big)
$$

其中 $c_p$ 是第 $p$ 条序列在列 $c$ 中的字符，$s$ 包含替换分与 gap 罚分（gap 对 gap 记 0）。它简单、可分解，但与真实演化模型并不等价：一次发生在祖先上的替换，会在所有后代序列对中被重复计数；SP 也不区分系统发育结构。

### 与参考比对比较：SP benchmark 与 TC

benchmark 常看参考结构同源位点能否恢复。BAliBASE、PREFAB、HomFam 等基准数据集以结构比对作为参考答案。需要区分作为优化目标的 sum-of-pairs 总分，与相对参考比对计算的 SP benchmark score：后者通常是参考 alignment 中 residue pair 被预测结果恢复的比例（相当于 recall），部分 benchmark 还另报 precision（预测出的 residue pair 中有多少在参考中出现），必须说明分母。TC（total column）分数要求整列完全正确，即参考比对中的列有多少在预测中被完整重现，更严格。

真实数据通常没有 ground truth，可用结构 alignment、模拟序列（用已知演化模型生成，真实比对已知）或算法间稳定性间接评估：如果多个工具或多组参数给出的某一列都一致，这一列更可信。

### 渐进比对之外的方法

Progressive alignment 之外，consistency-based 方法利用多个 pairwise alignment 间的传递支持：如果 A 的第 5 位在 A–B 比对中对上 B 的第 7 位，B 的第 7 位又在 B–C 比对中对上 C 的第 6 位，那么 A–C 比对中 A5–C6 这对应获得额外支持。T-Coffee 用这种思想从一个 pairwise 比对“库”构造 MSA，通常更准确但更慢。iterative refinement 则反复拆开并重新对齐 profile（MUSCLE 的 refinement 阶段、MAFFT 的 FFT-NS-i 等迭代策略）。对不稳定区应保留 alignment confidence（如 T-Coffee 的 TCS、GUIDANCE 的列置信度）或比较多个合理 MSA，而不是把单一输出视作无误差同源关系。

### 不确定性与下游分析

含大量 indel、低复杂度或无序区时，单一确定比对会隐藏不确定性：这些区域有许多分数相近的比对方案，工具只输出其中一个。多序列比对的下游用途包括保守位点、profile、系统发育、共进化、结构预测和功能注释。任何下游结论都应考虑比对误差传播：错误的列对应会被系统发育误认为替换，被共进化分析误认为协同变化。

下游建树前通常应 trim 极不可靠列（如 trimAl、Gblocks 一类工具），但过度 trimming 会删除快速演化信号，反而降低建树的准确性。分析应报告原始/过滤长度和工具参数。

> [!NOTE]+ 小结
>
> - MSA 给一组同源序列建立共同的列坐标；不能由 pairwise 比对直接拼接。
> - 精确 DP 需 $O((2^k-1)n^k)$ 时间，实际用 progressive alignment：距离 → guide tree → 逐步合并 profile → refinement。
> - profile–profile 打分是列分布下替换分的期望，需配合序列加权、pseudocount 与位置特异 gap。
> - guide tree 只是合并顺序，不是物种树；早期错误会被固化。
> - 评价要区分 SP 优化目标与 SP/TC benchmark 分数；不稳定区应报告置信度或比较多个比对。

<br>

<br>

# 序列 Motif：表示、打分与发现

多序列比对处理的是“整条序列从头到尾的同源对应”。但生物学里大量重要信号只有几个到几十个残基长：转录因子在启动子上结合的 6–20 bp 位点、内含子两端的剪接信号、蛋白质中结合 ATP 的 P-loop、锌指中固定间隔的 Cys/His。这些短的局部模式称为 motif。它们嵌在长度不一、其余部分并不相关的序列里，位置事先未知，整条序列的全局比对既找不到它们，也不适合描述它们。

核心想法可以概括为一句话：motif 是“若干条序列里反复出现、允许部分位置变化的一小段”，用逐位置的概率分布描述它，用这个分布对候选窗口打分，再在未知位置上做统计推断把它找出来。

与前后章节的关系如下。Motif 把多序列比对中“整条序列的共同列”收缩为一个局部窗口里的共同列；PWM 假设 motif 内各位置相互独立；后面的隐马尔可夫模型（HMM）在此基础上加入状态之间的转移，从而能描述可变长度与插入删除；结构章节中的三维 motif 则允许序列上相距很远的残基在空间中共同组成一个位点。

<br>

## Motif 是什么、要解决哪两类计算问题

### 研究对象

Motif 是比完整基因或结构域更短、在一组对象中反复出现的局部模式。序列 motif 可以表示转录因子结合位点（transcription factor binding site, TFBS）、剪接信号或蛋白质功能片段。它的一个关键特征是“允许部分位置高度保守，另一些位置可变”：例如某转录因子位点的第 3、4 位几乎总是 `GA`，而第 1 位 A、C、G 都可以出现。

描述这种“部分保守”有两种粒度。共识字符串（consensus string）只记录每个位置最常见的字符（必要时用 IUPAC 简并码）；位置权重矩阵（position weight matrix, PWM）记录每个位置的完整字符分布。后者保留了“第 1 位 A 占 60%、C 占 30%”这类信息，因此是计算上的标准表示。

### 发现任务与扫描任务

围绕 motif 有两类方向相反的计算任务：

- **发现（discovery）任务**：输入一组被认为共享某个信号的序列（例如 ChIP-seq 峰附近的序列、一组共表达基因的启动子）以及一个背景模型，输出 motif 模型（如 PWM）以及每条序列中的候选出现位置。此时 motif 本身和位置都未知。
- **扫描（scanning）任务**：输入一个已有的 PWM（例如来自 JASPAR 数据库）和新序列，输出每个长度为 motif 宽度的窗口的 log-odds 分数，用来预测新的结合位点。此时模型已知，只需打分。

发现任务还必须先声明 occurrence model，即每条序列里 motif 出现几次。MEME 中使用的三种设定是：OOPS（one occurrence per sequence）规定每条序列恰好出现一次，ZOOPS（zero or one occurrence per sequence）规定至多一次，TCM（two-component mixture）允许出现任意多次。三者对应不同的隐变量结构和似然函数，因此是不同的统计模型，结果也不能直接混比。

<br>

## Motif 的表示：共识串、PWM 与信息量

### 从共识字符串到位置权重矩阵

Motif 是在一组序列或结构中反复出现、与功能或约束相关的局部模式。它通常短于完整结构域，但长度不是定义条件，决定性的是“反复出现”和“与功能/约束相关”。

共识字符串的问题在于会丢失每个位点允许的替换：`TATAAT` 这个写法不能告诉读者第 2 位的 A 有多“必须”，也不能区分“偶尔为 G”和“从不为 G”。PWM 则保留位置特异的分布。设 motif 长度为 $L$，第 $i$ 个位置（$i=1,\dots,L$）出现碱基 $b\in\{A,C,G,T\}$ 的概率为 $p_{i,b}$（每个位置上对 $b$ 求和为 1），背景序列中碱基 $b$ 的频率为 $q_b$。PWM 的每个元素定义为对数几率（log-odds）：

$$
W_{i,b}=\log\frac{p_{i,b}}{q_b}
$$

$W_{i,b}>0$ 表示该碱基在 motif 第 $i$ 位比在背景中更常见，$W_{i,b}<0$ 表示更少见。有些文献把只含频率 $p_{i,b}$ 的矩阵称为 position frequency/probability matrix（PFM/PPM），把取 log-odds 后的矩阵称为 PWM 或 PSSM；阅读工具输出时应先确认是哪一种。

### 窗口打分

对一个长度为 $L$ 的窗口 $x=x_1x_2\cdots x_L$，PWM 分数是各位置对应元素之和：$\sum_{i=1}^{L}W_{i,x_i}$。这个和之所以有意义，是因为在“位置独立”假设下它恰好等于两个模型的对数似然比：

$$
\sum_{i=1}^{L}W_{i,x_i}
=\sum_{i=1}^{L}\log\frac{p_{i,x_i}}{q_{x_i}}
=\log\frac{\prod_i p_{i,x_i}}{\prod_i q_{x_i}}
=\log\frac{P(x\mid\text{motif})}{P(x\mid\text{background})}.
$$

第一步代入定义；第二步把对数之和改写为乘积的对数；第三步用到 motif 模型中各位置独立、背景模型为零阶（各碱基独立同分布）。因此分数越高，窗口越像由 motif 而非背景生成。实际扫描时把一条长序列按步长 1 滑过所有窗口，各窗口分数超过阈值即报告为候选位点。

伪计数（pseudocount）的作用：若训练位点里某位置从未出现某个碱基，频率估计为 0，对数会得到 $-\infty$，任何含该碱基的窗口都被一票否决。这在样本只有十几个位点时过于武断，因此在计数上加一个小的正数（如每个碱基加 0.5）再归一化，这等价于对 $p_{i,\cdot}$ 加一个 Dirichlet 先验。

### 独立性假设及其局限

PWM 假设 motif 内各位置条件独立：给定“这是一个 motif 位点”，第 2 位是什么不影响第 5 位是什么。若 position 2 与 5 之间存在协同约束（例如蛋白质与 DNA 接触时，两个碱基共同决定一个氢键模式，只有 `A…T` 或 `G…C` 组合可行），独立 PWM 只能分别记录两个位置的边缘分布，无法表达这种依赖，还会给 `A…C` 这样的不合法组合打出不低的分数。

能表达依赖的扩展包括二核苷酸模型（dinucleotide model，对相邻位置对建分布）、贝叶斯网络（Bayesian network，学习任意位置间的依赖图）和神经网络（如卷积模型直接学习窗口打分）。代价是参数更多，需要更多训练位点；在只有几十个位点时，简单 PWM 往往更稳健。

### 信息量与 sequence logo

信息量（information content, IC）衡量 motif 某一位置的分布相对背景有多“不同”，也就是这一位置携带多少区分 motif 与背景的信号。在一般背景 $q_b$ 下，它定义为该位置分布 $p_i$ 相对背景 $q$ 的 KL 散度（以 bit 为单位）：

$$
IC_i=D_{\mathrm{KL}}(p_i\|q)
=\sum_b p_{i,b}\log_2\frac{p_{i,b}}{q_b}
$$

KL 散度非负，且只有当 $p_i=q$ 时为 0，所以完全不保守、与背景一样的位置 IC 为 0。注意 $IC_i$ 恰好是该位置 PWM 分数在 motif 分布下的期望（以 2 为底）：$IC_i=\mathbb{E}_{b\sim p_i}[W_{i,b}]$，所以 motif 的总 IC 就是“真实位点平均能比背景高出多少分”，IC 越高，扫描时越容易把真位点和随机窗口分开。

只有 DNA 均匀背景 $q_b=1/4$ 时，上式才化为熟悉的 $IC_i=2-H_i$，推导如下：

$$
\sum_b p_{i,b}\log_2\frac{p_{i,b}}{1/4}
=\sum_b p_{i,b}\log_2 p_{i,b}+\sum_b p_{i,b}\log_2 4
=-H_i+2,
$$

其中 $H_i=-\sum_b p_{i,b}\log_2 p_{i,b}$ 是该位置的 Shannon 熵，并约定 $0\log0=0$（未出现的碱基不贡献熵）。于是 DNA 单个位置的 IC 取值在 0 到 2 bit 之间：完全保守（一种碱基概率为 1）时为 2 bit，均匀分布时为 0。蛋白质（20 种氨基酸、均匀背景）的上限是 $\log_2 20\approx4.32$ bit。当背景不均匀（例如 GC 含量很高的基因组）时，直接用 $2-H_i$ 会误判：一个富含 G/C 的位置在 GC-rich 背景下其实并不特殊，这时必须用一般的 KL 形式。

sequence logo 是 motif 的标准可视化：横轴为位置，每个位置堆叠四个字母。字母高度通常结合 $IC_i$ 与频率，最常见的画法是字母 $b$ 的高度为 $p_{i,b}\cdot IC_i$，即“该位置总高度为 IC，再按频率在各字母间分配”。因此保守位置整体更高，主导字母一眼可见；不保守的位置几乎是平的。

<br>

## Motif 发现：从枚举到 EM 与 Gibbs 采样

### 问题为什么难

发现任务同时有两个未知量：motif 的参数（PWM）和每条序列中 motif 的位置。若知道位置，把各位点对齐后直接数频率就得到 PWM；若知道 PWM，用扫描就能找位置。两者互相依赖，正是“鸡生蛋”式的隐变量问题。穷举所有位置组合不可行：$N$ 条长度为 $\ell$ 的序列在 OOPS 下就有约 $(\ell-L+1)^N$ 种组合。

### 四类主要策略

- **枚举（enumeration）**：对较短的 motif，可以枚举所有长度为 $L$ 的候选字（或允许少量错配的字），统计其在输入集中出现的次数并与期望比较。精确、不依赖初始化，但候选数为 $4^L$，代价随长度指数增长，只适合短 motif。
- **EM（expectation–maximization）**：把 motif 位置当作隐变量，在“给定 PWM 求位置的期望（后验）”与“给定期望位置重估 PWM”之间迭代。MEME（Bailey & Elkan, 1994）就是这一思路，并以多次不同初始化的 EM（Multiple EM）缓解局部最优。
- **Gibbs 采样（Gibbs sampling）**：逐条序列按后验概率重采样 motif 位置，而不是取期望（Lawrence et al., 1993）。
- **富集检验（enrichment test）**：比较某个候选模式在正集（如 ChIP-seq 峰）和背景集（打乱序列或非结合区）中的出现率，用超几何检验或 Fisher 精确检验判断是否显著富集。

需要注意：EM 只保证收敛到局部最优，初始化、背景模型和序列组成偏差（如 CpG island、低复杂度区）都会改变结果。在整基因组上扫描还会引入海量多重检验：人类基因组有约 $3\times10^9$ 个窗口，即使每个窗口的假阳性率只有 $10^{-4}$，也会出现几十万个“命中”，因此阈值必须结合多重检验校正或经验背景分布。

### OOPS 模型下的 EM 推导

以“每条序列恰有一个 motif”（OOPS）为例。设第 $n$ 条序列为 $x_n$，motif 宽度为 $L$，隐变量 $z_{n,j}$ 表示第 $n$ 条序列的 motif 从位置 $j$ 开始（$j$ 取遍所有能放下完整窗口的起点，等价地记作 $z_n=j$）。模型参数 $\theta$ 包括 motif 部分 $\theta_{\text{motif}}$（PWM 的概率 $p_{i,b}$）和背景部分 $\theta_{\text{bg}}$（$q_b$）。

给定起点 $j$，整条序列的概率等于“窗口由 motif 生成”乘以“其余位置由背景生成”。E-step 根据当前 PWM 用贝叶斯公式计算每个起点的后验责任（posterior responsibility）$\gamma_{n,j}$：

$$
\gamma_{n,j}
=P(z_n=j\mid x_n,\theta)
\propto P(z_n=j)P(x_{n,j:j+L}\mid\theta_{\text{motif}})
P(x_{n,\text{background}}\mid\theta_{\text{bg}})
$$

这里 $x_{n,j:j+L}$ 采用半开区间记号，表示从 $j$ 开始的 $L$ 个字符（与 Python 切片一致）。先验 $P(z_n=j)$ 通常取均匀分布。一个实用的化简是：用整条序列的背景概率去除分子，窗口外的背景项全部约掉，只剩窗口内的似然比

$$
\gamma_{n,j}\propto\prod_{i=1}^{L}\frac{p_{i,x_{n,j+i-1}}}{q_{x_{n,j+i-1}}}
=\exp\Big(\sum_{i=1}^{L}W_{i,x_{n,j+i-1}}\Big),
$$

（此处 $W$ 取自然对数），也就是说 E-step 的权重正是扫描分数的指数，再对每条序列内的所有起点归一化。下面的代码正是这样实现的。

M-step 用 $\gamma_{n,j}$ 加权统计每个位置的碱基频率并重新估计 PWM：motif 第 $i$ 位碱基 $b$ 的期望计数是所有序列、所有起点上“该起点使第 $i$ 位落在碱基 $b$”的责任之和，加伪计数后归一化。这是一种“软计数”：EM 不把位点硬性指定到某处，而是按可信度把计数摊到各候选位置。多次随机初始化后比较 held-out likelihood（在未参与训练的序列上的似然），可降低陷入局部最优、或把训练集偶然模式当真 motif 的风险。

### 实现：构建 PWM、扫描与 OOPS-EM

下面三个函数分别对应“由已知位点构建 PWM”“用 PWM 扫描序列”和“OOPS 模型下的 EM 发现”。

```python
import math

DNA = "ACGT"

def build_pwm(sites, pseudocount=0.5):
    """由等长 motif sites 构建按位置归一化的 PWM。"""
    if not sites or len({len(site) for site in sites}) != 1:
        raise ValueError("sites must be a nonempty equal-length collection")
    if pseudocount <= 0 or any(set(site) - set(DNA) for site in sites):
        raise ValueError("use A/C/G/T sites and a positive pseudocount")

    pwm = []
    for column in zip(*sites):
        counts = {base: pseudocount for base in DNA}
        for base in column:
            counts[base] += 1
        total = sum(counts.values())
        pwm.append({base: counts[base] / total for base in DNA})
    return pwm


def scan_pwm(sequence, pwm, background=None):
    """返回每个窗口的 0-based 起点与 PWM/background log-odds。"""
    sequence = sequence.upper()
    background = background or {base: 0.25 for base in DNA}
    if set(sequence) - set(DNA):
        raise ValueError("sequence must contain only A/C/G/T")
    if any(background.get(base, 0) <= 0 for base in DNA):
        raise ValueError("background probabilities must be positive")

    width = len(pwm)
    hits = []
    for start in range(len(sequence) - width + 1):
        window = sequence[start:start + width]
        score = sum(
            math.log(pwm[offset][base] / background[base])
            for offset, base in enumerate(window)
        )
        hits.append((start, score))
    return hits


def motif_em_oops(
    sequences, width, iterations=50, pseudocount=0.5, background=None
):
    """OOPS 模型的 EM：假设每条 DNA 序列恰有一个 motif。"""
    sequences = [seq.upper() for seq in sequences]
    if (
        not sequences
        or width <= 0
        or iterations <= 0
        or any(len(seq) < width for seq in sequences)
    ):
        raise ValueError("each sequence must be at least motif width")
    if any(set(seq) - set(DNA) for seq in sequences):
        raise ValueError("sequences must contain only A/C/G/T")
    background = background or {base: 0.25 for base in DNA}

    # 用各序列首个窗口初始化；实际分析应采用多次随机启动。
    pwm = build_pwm([seq[:width] for seq in sequences], pseudocount)
    responsibilities = None
    for _ in range(iterations):
        responsibilities = []
        for sequence in sequences:
            log_weights = [
                score for _, score in scan_pwm(sequence, pwm, background)
            ]
            maximum = max(log_weights)
            weights = [math.exp(value - maximum) for value in log_weights]
            normalizer = sum(weights)
            responsibilities.append([
                value / normalizer for value in weights
            ])

        # M-step：按窗口 posterior 对每个 motif 位置累计软计数。
        counts = [
            {base: pseudocount for base in DNA} for _ in range(width)
        ]
        for sequence, posterior in zip(sequences, responsibilities):
            for start, probability in enumerate(posterior):
                for offset in range(width):
                    counts[offset][sequence[start + offset]] += probability
        pwm = []
        for column in counts:
            total = sum(column.values())
            pwm.append({base: column[base] / total for base in DNA})

    return pwm, responsibilities
```

几点阅读提示：

- `build_pwm` 返回的是概率矩阵（每个位置一个字典，四个碱基概率和为 1），log-odds 在 `scan_pwm` 里现算，即 `math.log(pwm[offset][base] / background[base])`，对应上文的 $W_{i,b}$（自然对数）。
- E-step 中先减去最大值 `maximum` 再取指数，是为了防止 log 分数很大时 `exp` 溢出；减去同一个常数不改变归一化后的结果。
- `scan_pwm` 计算的是给定模型下的窗口分数，不含全基因组多重扫描校正；把它的输出直接当作“显著位点”会产生大量假阳性。
- `motif_em_oops` 只实现 OOPS occurrence model。ZOOPS 还需要一个“本序列没有 motif”的隐状态（与各起点一起参与归一化），TCM 则允许多个位置同时出现，每个窗口独立地属于 motif 或背景。
- 初始化用的是每条序列的第一批窗口（`seq[:width]`），可能使 EM 停在局部最优，典型情形是得到与真实 motif 错开一两位的相移（phase-shift）解；实际应以不同随机种子（seed）重启，并比较 held-out likelihood。

### Motif EM 与 Baum–Welch 的关系

Motif EM 与后文 HMM 的 Baum–Welch 算法使用同一原则：E-step 对隐变量求后验，M-step 用期望计数更新参数。区别在隐变量的结构。OOPS motif 的隐变量是每条序列的单一起点，候选数只随序列长度线性增长，直接枚举所有起点即可求后验；HMM 的隐变量是受转移概率约束的整条状态路径，路径数随长度指数增长，必须由 forward–backward 动态规划来汇总指数多条路径。

### Gibbs 采样

Gibbs 采样（Lawrence et al., 1993, *Science*）把 EM 中“取期望”换成“随机抽样”。每次暂时移除一条序列，用其余序列当前选中的位点构建 PWM，再按后验概率（与 E-step 中的 $\gamma_{n,j}$ 相同的形式）从被移除序列中随机采样一个新位置，然后换下一条序列重复。由于采样允许偶尔接受较差的位置，它比确定性的 EM 更容易跳出局部最优。

其输出不是单一的点估计，而是一组后验样本（posterior sample）。因此还需要做 MCMC 的常规诊断：丢弃初始的 burn-in 阶段；检查 mixing（链是否在不同解之间充分移动，而不是卡在某个位移解上，例如 motif 整体错开一两位的“相移”解）；运行多条独立链，检查它们是否收敛到一致的 motif。

### Occurrence model 与背景模型

Motif discovery 必须声明 occurrence model：OOPS 假设每条序列恰有一次，ZOOPS 允许零次或一次，TCM 允许多次。若输入集中有不少序列其实不含 motif（ChIP-seq 数据常见），强行用 OOPS 会迫使每条序列贡献一个“最像”的位点，从而稀释并扭曲 PWM。

背景模型也同样关键。最简单的是零阶频率（各碱基独立，频率为 $q_b$）；可以扩展为一阶或更高阶 Markov model，即 $P(x_t\mid x_{t-1},\dots)$。高阶背景能吸收局部组成偏差（如 CpG 二核苷酸缺失、poly-A 片段）和重复序列的影响，否则这些结构会被误报为富集的“假 motif”。MEME 套件中的背景文件就可以指定 Markov 阶数。

### 1D motif 与 3D motif

1D motif 主要约束残基在线性序列中的顺序，可以是连续的一段，也可以包含 wildcard、gap 或可变间隔（PROSITE 模式如 `C-x(2,4)-C-x(3)-[LIVMFYWC]` 即是这一类）。3D motif 则约束指定原子或残基之间的空间关系（距离、角度、相对取向），序列上的顺序与间隔可以很弱。

两者的关系是不对称的。功能相关的 1D motif 在折叠蛋白中通常对应某种 3D 微环境，但逆命题不成立：序列上相距很远的残基可在折叠后共同组成 active site（如丝氨酸蛋白酶的 Ser–His–Asp 催化三联体，三个残基在一级序列上相隔很远）；不同序列甚至不同 fold 也可能通过趋同的几何构型（convergent geometry）形成相似功能位点（胰凝乳蛋白酶与枯草杆菌蛋白酶的催化三联体即为经典例子）。因此一个清晰的 3D motif 未必存在可识别的线性 consensus，这正是结构模式识别需要单独方法的原因。

> [!NOTE]+ 小结
>
> - PWM 用 $W_{i,b}=\log(p_{i,b}/q_b)$ 表示逐位置偏好，窗口分数等于 motif 与背景的对数似然比；伪计数避免 $-\infty$。
> - 信息量是相对背景的 KL 散度；只有均匀 DNA 背景下才等于 $2-H_i$。
> - 发现 motif 是隐变量问题：EM（MEME）求期望位置，Gibbs 采样抽样位置，二者都受初始化与局部最优影响。
> - 必须声明 occurrence model（OOPS/ZOOPS/TCM）和背景模型阶数；全基因组扫描需要多重检验校正。
> - PWM 假设位置独立；需要依赖或可变长度时，转向更复杂模型或下一章的 HMM。

<br>

# 隐马尔可夫模型（Hidden Markov Model, HMM）

PWM 能描述一个固定长度的窗口，却回答不了两类常见问题。第一类是“分段”：一条长 DNA 序列中哪些区段是 GC-rich、哪些是 AT-rich，或哪些是外显子、哪些是内含子？区段的边界和长度都未知。第二类是“带插入删除的家族模型”：一个蛋白质家族的成员在保守位置上相似，但各自在不同地方插入或删除了若干残基，固定宽度的 PWM 无法容纳这种长度变化。

HMM 的核心想法是：观察到的字符序列背后有一条看不见的“状态路径”，每个状态有自己的字符分布（发射），状态之间按概率跳转（转移）。只要状态转移满足 Markov 性质，就可以用动态规划在指数多条可能路径上高效地求和或求最大值。

## HMM 解决的问题与基本组成

### 研究对象

HMM 描述这样一类序列过程：能够观察到字符序列，真正决定字符分布的状态却不可直接观察。几个生物学例子：

- DNA 碱基可以被观察，“GC-rich 区域”或“AT-rich 区域”则是需要推断的隐状态；
- 基因预测中，每个碱基属于外显子、内含子还是基因间区是隐状态；
- 蛋白质家族中某个残基来自 Match、Insert 还是 Delete 状态，也属于隐路径。

### 模型组成与三类问题

HMM 由三组参数组成：初始分布（第一个位置处于各状态的概率）、状态转移概率（从一个状态到下一个状态的概率）和发射概率（某状态下产生各字符的概率）。给定模型与观测序列，可以做三件事：计算整条序列的似然（likelihood），找一条最可能的状态路径，或计算每个位置属于各状态的后验概率。给定未标注序列，还可以用 Baum–Welch 估计参数。

按输入输出归纳，这对应 HMM 的三类经典问题：evaluation（模型 + 序列 → 似然）、decoding（模型 + 序列 → 状态路径或逐位后验）和 learning（序列 → 模型参数）。

### 与序列比对的关系

HMM 与序列比对使用相同的动态规划思想，但目标不同。比对的 DP 在“比对路径”上取最优，HMM 的 DP 在“状态路径”上取最优或求和：Viterbi 用 `max` 保留最可能路径，Forward 用求和汇总所有路径。Pair HMM 把两条序列的 alignment 本身看作隐路径（状态为“匹配 / x 中插入 / y 中插入”）；Profile HMM 把多序列比对的保守列建成 Match 状态，并用 Insert/Delete 状态描述 indel。

### 与其他章节的衔接

HMM 承接多序列比对中的列分布与 Motif 中的位置特异概率，并把“各位置独立”推广为“有状态转移的生成模型”。后续章节中还会反复看到同样的思想：系统发育的似然计算同样在隐藏的祖先状态上求和（Felsenstein 剪枝算法）；RNA 的 covariance model 在 profile HMM 上加入成对位置以描述碱基配对；FATHMM 用 Profile HMM 给出的保守性为变异效应打分。

<br>

## 模型定义：联合概率与 Markov 假设

### 符号

设观测序列为 $x=x_1x_2\cdots x_T$，长度为 $T$，每个 $x_t$ 取自字母表（如 DNA 的 $\{A,C,G,T\}$）；隐状态序列为 $z=z_1\cdots z_T$，每个隐状态 $z_t$ 取自 $K$ 个状态之一。HMM 由隐状态 $z_t$、转移概率 $a_{ij}=P(z_t=j\mid z_{t-1}=i)$ 和发射概率 $e_j(x_t)=P(x_t\mid z_t=j)$ 构成，另有初始分布 $\pi_j=P(z_1=j)$。转移矩阵 $A=(a_{ij})$ 是 $K\times K$ 的，每一行和为 1；每个状态的发射分布在字母表上和为 1。

### 一阶 Markov 假设

一阶 HMM 做两个条件独立假设：当前隐状态只依赖上一个状态；当前观测只依赖当前隐状态：

$$
P(z_t\mid z_{1:t-1})=P(z_t\mid z_{t-1}),\qquad
P(x_t\mid z_{1:t},x_{1:t-1})=P(x_t\mid z_t)
$$

一阶 Markov 假设是为获得可计算模型而采用的近似，真实生物序列显然有更长程的依赖。更长的记忆可以通过扩展状态表示来实现（例如把“上两个状态”组合成一个新状态，或基因预测中按密码子相位拆分外显子状态），代价是状态数和参数量增加。

### 联合概率

在上述假设下，用链式法则展开 $P(x,z)$，每一项都只保留直接依赖，得到状态路径与观测的联合概率：

$$
P(x,z)=P(z_1)e_{z_1}(x_1)\prod_{t=2}^{T}a_{z_{t-1},z_t}e_{z_t}(x_t)
$$

读法是：先按初始分布选第一个状态并发射第一个字符，此后每一步“先转移、再发射”。给定一条具体路径，联合概率很容易算；困难在于路径共有 $K^T$ 条。

### 三个基本问题及复杂度

- **Evaluation**：计算序列在模型下的概率 $P(x)=\sum_z P(x,z)$，用 forward 算法；
- **Decoding**：找最可能的状态路径 $\arg\max_z P(x,z)$，用 Viterbi 算法；
- **Learning**：估计未知参数，用 Baum–Welch（即 HMM 上的 EM）。

三者都利用 Markov 条件独立性避免枚举指数多条路径：时刻 $t$ 的状态把过去和未来“隔开”，所以只需为每个时刻、每个状态保存一个汇总量。Forward 与 Viterbi 在稠密转移下为 $O(TK^2)$（$T$ 个位置，每个位置上 $K$ 个状态各看 $K$ 个前驱）；Baum–Welch 的每次 EM 迭代为 $O(TK^2)$，若迭代 $I$ 次则总计 $O(ITK^2)$。

<br>

## Forward 算法：对所有路径求和

### 动机

要比较“这条序列更像 CpG island 模型还是背景模型”，或要训练参数，需要的是 $P(x)$，即对所有可能路径的概率求和，而不是只看一条最佳路径。直接求和需要 $K^T$ 项；forward 算法的思路是把“到时刻 $t$ 为止、停在状态 $j$”的所有路径合并成一个数。

### 递推

定义前向变量 $\alpha_t(j)=P(x_{1:t},z_t=j)$，即“观察到前 $t$ 个字符且第 $t$ 步处于状态 $j$”的概率。初始化时只有一个字符：

$$
\alpha_1(j)=\pi_j e_j(x_1)
$$

递推时，到达 $(t,j)$ 的路径必然来自某个 $(t-1,i)$，再经转移 $a_{ij}$ 并发射 $x_t$。对前驱 $i$ 求和：

$$
\alpha_t(j)=e_j(x_t)\sum_i\alpha_{t-1}(i)a_{ij}
$$

这一步成立的原因是 Markov 性质：给定 $z_{t-1}=i$，$z_t$ 与 $x_t$ 不再依赖更早的历史，所以可以把 $\alpha_{t-1}(i)$ 作为整体乘上转移和发射。

最终似然为 $P(x)=\sum_j\alpha_T(j)$（若模型有显式 end 状态，则再乘以转向 end 的概率）。forward 汇总了所有可能的状态路径，因此可以用来比较模型（似然比）、训练参数，或与 backward 结合计算后验。

<br>

## Backward 算法与后验解码

### 后向变量

forward 只看到“过去”。要回答“第 $t$ 个碱基属于状态 $i$ 的概率是多少”，还需要考虑“未来”的观测。定义后向变量 $\beta_t(i)=P(x_{t+1:T}\mid z_t=i)$，即在第 $t$ 步处于状态 $i$ 的条件下，生成其余后缀的概率：

$$
\beta_T(i)=1,\qquad
\beta_t(i)=\sum_j a_{ij}e_j(x_{t+1})\beta_{t+1}(j)
$$

末端 $\beta_T(i)=1$ 是因为空后缀的概率为 1；递推从右向左进行，每一步枚举下一个状态 $j$，乘上转移、发射与更远后缀的概率。

### 单点后验

因为给定 $z_t=i$ 时过去与未来条件独立，$P(x,z_t=i)=\alpha_t(i)\beta_t(i)$。除以 $P(x)$ 即得单点后验：

$$
P(z_t=i\mid x)=\frac{\alpha_t(i)\beta_t(i)}{P(x)}
$$

对每个 $t$，这些后验在所有状态上和为 1，这也是检验实现是否正确的简单办法。

### 后验解码与 Viterbi 的区别

后验解码（posterior decoding）在每个位点选边缘概率（marginal probability）最大的状态。它使每个位置的期望正确率最大，但逐位拼接起来的路径不一定是合法转移路径：若某两个状态之间转移概率为 0，后验解码仍可能把它们相邻放置。Viterbi 则保证整条路径合法，并最大化联合概率 $P(x,z)$。两者回答的是不同的问题，不应混称。实际中常把两者并用：Viterbi 给出一条分段结果，后验概率曲线显示哪些边界可信、哪些不确定。

<br>

## Viterbi 算法：最可能的状态路径

### 递推

Viterbi 与 forward 结构相同，只是把求和换成取最大值。为避免下溢直接在 log 空间工作，定义 $\delta_t(j)$ 为“处理到第 $t$ 个字符、以状态 $j$ 结束”的最佳路径 log-score：

$$
\delta_t(j)=\log e_j(x_t)+\max_i
\left[\delta_{t-1}(i)+\log a_{ij}\right]
$$

初始化为 $\delta_1(j)=\log\pi_j+\log e_j(x_1)$。它成立的理由是最优子结构：以 $(t,j)$ 结尾的最佳路径，去掉最后一步后，必然是以某个 $(t-1,i)$ 结尾的最佳路径。

除分数外还要保存回溯指针 $\psi_t(j)=\arg\max_i(\cdots)$，否则只能知道最佳分数而无法 traceback 出路径。终止时取 $\max_j\delta_T(j)$ 对应的状态，再沿 $\psi$ 从后往前恢复整条路径。

### 稀疏转移

许多生物 HMM 的转移图很稀疏（profile HMM 中每个状态只有 3 条出边，基因模型中外显子不能直接跳到任意状态）。只遍历实际存在的边时，每一步的工作量从 $K^2$ 降为实际边数 $|E|$，复杂度为 $O(T|E|)$。

### 实现：log 空间的 Forward–Backward 与 Viterbi

下面的代码把参数都以 log 概率形式传入：`log_start[s]`、`log_trans[p][s]`、`log_emit[s][symbol]`。

```python
import math

def logsumexp(values):
    """稳定计算 log(sum(exp(values)))。"""
    values = list(values)
    if not values:
        return -math.inf
    m = max(values)
    if m == -math.inf:
        return m
    # 先减最大值，避免 exp(很大的数) 溢出。
    return m + math.log(sum(math.exp(v - m) for v in values))


def forward_backward(obs, states, log_start, log_trans, log_emit):
    """返回 log P(obs) 与每个位置的状态 posterior。"""
    if not obs:
        return 0.0, []
    if not states:
        raise ValueError("states must not be empty")

    # alpha[t][s] = log P(obs[:t+1], z_t=s)
    alpha = [
        {s: log_start[s] + log_emit[s][obs[0]] for s in states}
    ]
    for symbol in obs[1:]:
        current = {}
        for state in states:
            incoming = (
                alpha[-1][previous] + log_trans[previous][state]
                for previous in states
            )
            current[state] = log_emit[state][symbol] + logsumexp(incoming)
        alpha.append(current)
    log_likelihood = logsumexp(alpha[-1].values())

    # beta[t][s] = log P(obs[t+1:] | z_t=s)；末端空后缀概率为 1。
    beta = [{s: 0.0 for s in states} for _ in obs]
    for t in range(len(obs) - 2, -1, -1):
        next_symbol = obs[t + 1]
        for state in states:
            beta[t][state] = logsumexp(
                log_trans[state][next_state]
                + log_emit[next_state][next_symbol]
                + beta[t + 1][next_state]
                for next_state in states
            )

    # exp(alpha+beta-log_likelihood) 将每个位置归一化到和为 1。
    posterior = []
    for t in range(len(obs)):
        posterior.append({
            state: math.exp(
                alpha[t][state] + beta[t][state] - log_likelihood
            )
            for state in states
        })
    return log_likelihood, posterior


def viterbi(obs, states, log_start, log_trans, log_emit):
    """返回最佳 joint log-score 与一条最可能合法状态路径。"""
    if not obs:
        return 0.0, []
    if not states:
        raise ValueError("states must not be empty")
    # score[t][s]：处理到位置 t 且以状态 s 结束的最佳路径分数。
    score = [{s: log_start[s] + log_emit[s][obs[0]] for s in states}]
    # back[t-1][s] 保存位置 t 的最佳前驱状态。
    back = []
    for symbol in obs[1:]:
        cur, parent = {}, {}
        for s in states:
            choices = {
                p: score[-1][p] + log_trans[p][s] + log_emit[s][symbol]
                for p in states
            }
            parent[s], cur[s] = max(choices.items(), key=lambda item: item[1])
        score.append(cur)
        back.append(parent)

    # 先选最后位置的最佳状态，再沿 back pointer 反向恢复路径。
    last = max(score[-1], key=score[-1].get)
    path = [last]
    for parent in reversed(back):
        path.append(parent[path[-1]])
    return score[-1][last], list(reversed(path))
```

实现要点：

- 概率连乘会下溢：长度 1000 的序列，每步概率约 0.25 时联合概率约为 $10^{-602}$，远低于双精度浮点的下限。因此实现应采用 log-space。forward 中的求和需要用 `logsumexp`（$\log\sum_k e^{v_k}=m+\log\sum_k e^{v_k-m}$，$m=\max_k v_k$），Viterbi 则把求和替换为最大值，log 空间中直接相加即可。
- `viterbi` 把发射项放进了 `choices` 里一起取最大值；由于发射项与前驱 $p$ 无关，这与公式中“先对前驱取 max、再加 $\log e_j(x_t)$”等价。
- 参数字典必须覆盖所有允许的初始、转移与发射组合；不允许的组合应显式记为 `-math.inf`，否则会出现 `KeyError`。

<br>

## Baum–Welch 算法：无标注数据上的参数学习

### 动机与思路

如果训练序列带有状态标签（例如已注释的外显子/内含子边界），参数估计就是数数：数一数 $i\to j$ 的转移次数和状态 $i$ 下各字符的出现次数，再归一化。参数未知且训练序列没有 state label 时，Baum–Welch 使用 EM：既然不知道真实路径，就用当前参数下的后验去计算“期望计数”，再像有标签时一样归一化。这与 Motif EM 是同一原则，只是隐变量由“单个起点”变成了“整条路径”，期望计数需要靠 forward–backward 求得。

### E-step：期望转移计数与状态占据

E-step 计算每一对相邻位置使用转移 $i\to j$ 的后验概率，即期望转移计数：

$$
\xi_t(i,j)=P(z_t=i,z_{t+1}=j\mid x)
=\frac{\alpha_t(i)a_{ij}e_j(x_{t+1})\beta_{t+1}(j)}{P(x)}
$$

分子四个因子分别对应：到 $t$ 为止停在 $i$ 的前缀概率、转移 $i\to j$、在 $j$ 发射 $x_{t+1}$、以及从 $j$ 出发生成剩余后缀的概率。同时计算状态后验 $\gamma_t(i)=P(z_t=i\mid x)=\alpha_t(i)\beta_t(i)/P(x)$，且有 $\gamma_t(i)=\sum_j\xi_t(i,j)$（对 $t<T$）。

### M-step：归一化期望计数

M-step 把期望计数归一化，得到新参数。转移概率为“$i\to j$ 的期望次数 / 离开 $i$ 的期望次数”：

$$
\hat a_{ij}
=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}
{\sum_{t=1}^{T-1}\gamma_t(i)}
$$

分母只加到 $T-1$，因为最后一个位置不再发生转移。发射概率按状态 $i$ 占据时观察到各符号的期望次数归一化，即 $\hat e_i(b)=\sum_{t:x_t=b}\gamma_t(i)\big/\sum_{t}\gamma_t(i)$；初始分布取 $\hat\pi_i=\gamma_1(i)$。

### 性质与常见坑

每次迭代不会降低训练数据的似然（EM 的一般性质），但可能落入局部最优，并可能把某些概率估为 0：一旦某个转移或发射的估计变成 0，之后的期望计数也永远是 0，这条路就被“永久关闭”。因此初始化、伪计数（pseudocount）与多次重启都很重要。

### 实现：一次带 scaling 的更新

下面实现一次带缩放（scaling）的 Baum–Welch 更新。输入参数使用普通概率而非 log probability，便于直接看到期望计数如何进入 M-step。scaling 的做法（Rabiner 1989）是：forward 每一步把 $\alpha_t$ 归一化为和为 1，记下被除掉的总量 $c_t$；backward 每一步除以同一个 $c_{t+1}$。这样缩放后的 $\hat\alpha_t(i)\hat\beta_t(i)$ 恰好等于 $\gamma_t(i)$，而 $\log P(x)=\sum_t\log c_t$，数值上不再下溢。

```python
import math

def baum_welch_step(
    obs, states, alphabet, start, transition, emission, pseudocount=1e-6
):
    """对一条观测序列执行一次 scaled Baum–Welch EM 更新。"""
    if not obs or any(symbol not in alphabet for symbol in obs):
        raise ValueError("obs must be nonempty and use the given alphabet")
    if pseudocount < 0:
        raise ValueError("pseudocount must be nonnegative")

    # Forward scaling：alpha[t] 每步归一化，scale[t] 保存被除去的质量。
    alpha, scale = [], []
    first = {
        state: start[state] * emission[state][obs[0]]
        for state in states
    }
    total = sum(first.values())
    if total <= 0:
        raise ValueError("observation has zero probability")
    alpha.append({state: first[state] / total for state in states})
    scale.append(total)

    for symbol in obs[1:]:
        current = {}
        for state in states:
            incoming = sum(
                alpha[-1][previous] * transition[previous][state]
                for previous in states
            )
            current[state] = incoming * emission[state][symbol]
        total = sum(current.values())
        if total <= 0:
            raise ValueError("observation has zero probability")
        alpha.append({state: current[state] / total for state in states})
        scale.append(total)

    # 与 forward 使用相同 scaling convention，避免 beta 指数下溢。
    beta = [{state: 1.0 for state in states} for _ in obs]
    for t in range(len(obs) - 2, -1, -1):
        for state in states:
            beta[t][state] = sum(
                transition[state][next_state]
                * emission[next_state][obs[t + 1]]
                * beta[t + 1][next_state]
                for next_state in states
            ) / scale[t + 1]

    gamma = []
    for t in range(len(obs)):
        row = {
            state: alpha[t][state] * beta[t][state] for state in states
        }
        normalizer = sum(row.values())
        gamma.append({
            state: row[state] / normalizer for state in states
        })

    # xi[t][i][j] 是 t→t+1 使用转移 i→j 的 posterior。
    xi = []
    for t in range(len(obs) - 1):
        row = {
            state: {
                next_state: (
                    alpha[t][state]
                    * transition[state][next_state]
                    * emission[next_state][obs[t + 1]]
                    * beta[t + 1][next_state]
                )
                for next_state in states
            }
            for state in states
        }
        normalizer = sum(
            row[state][next_state]
            for state in states
            for next_state in states
        )
        xi.append({
            state: {
                next_state: row[state][next_state] / normalizer
                for next_state in states
            }
            for state in states
        })

    # M-step：加 pseudocount 后逐行归一化，避免出现永久零概率。
    start_counts = {
        state: gamma[0][state] + pseudocount for state in states
    }
    start_total = sum(start_counts.values())
    new_start = {
        state: start_counts[state] / start_total for state in states
    }

    new_transition = {}
    for state in states:
        counts = {
            next_state: (
                sum(row[state][next_state] for row in xi) + pseudocount
            )
            for next_state in states
        }
        total = sum(counts.values())
        new_transition[state] = {
            next_state: counts[next_state] / total for next_state in states
        }

    new_emission = {}
    for state in states:
        counts = {
            symbol: (
                sum(
                    gamma[t][state]
                    for t, observed_symbol in enumerate(obs)
                    if observed_symbol == symbol
                )
                + pseudocount
            )
            for symbol in alphabet
        }
        total = sum(counts.values())
        new_emission[state] = {
            symbol: counts[symbol] / total for symbol in alphabet
        }

    log_likelihood = sum(math.log(value) for value in scale)
    return new_start, new_transition, new_emission, log_likelihood
```

代码中 `gamma` 与 `xi` 都再做了一次显式归一化。对 `gamma` 而言，缩放后的 $\hat\alpha_t(i)\hat\beta_t(i)$ 本身已等于 $\gamma_t(i)$，归一化只是数值保险；对 `xi` 而言，缩放量 $\hat\alpha_t(i)a_{ij}e_j(x_{t+1})\hat\beta_{t+1}(j)$ 等于 $c_{t+1}\,\xi_t(i,j)$，必须再除以 $c_{t+1}$（等价于代码中对所有 $(i,j)$ 的和归一化）才得到真正的 $\xi_t(i,j)$。返回的 `log_likelihood` 是本次更新所用的“旧参数”下的对数似然。

### 多序列训练与停止准则

对多条独立序列训练时，E-step 分别计算每条序列的 $\gamma,\xi$，M-step 再汇总所有序列的期望计数后统一归一化。不能把多条序列直接首尾拼接成一条，否则会人为加入“上一条序列末尾 → 下一条序列开头”的跨序列转移，污染转移估计。

反复调用 `baum_welch_step` 即形成 EM 迭代；判断收敛时应比较所有序列的总 log-likelihood（各序列之和）或参数变化，并用独立的 validation sequence 防止过拟合。训练应以 validation likelihood（独立验证序列上的似然）、参数变化幅度或最大迭代次数作为停止条件，并记录随机初始化以便复现。需要注意，训练似然单调上升不代表验证似然改善（模型可能在过拟合），也不能保证参数可识别：交换两个状态的标签得到完全相同的似然，不同初始化可能学到“同一个模型的不同编号”，甚至学到与生物学含义无关的分段。

<br>

## Profile HMM：把多序列比对变成概率模型

### 动机

PWM 能描述一个蛋白质家族的保守列，却无法处理插入与删除；成对比对的打分矩阵能处理 indel，却对所有位置使用同一套替换分数与 gap 罚分。蛋白质家族真实的情况是：某些列几乎不变（催化残基），某些列可变；某些位置常有插入环，某些位置从不允许 gap。profile HMM 把这两方面同时编码为位置特异的概率。

序列家族的 profile HMM 通常为每个保守列建立一个 Match 状态，并提供 Insert、Delete 状态。它同时表示位置特异替换和位置特异的 indel，比 PWM 更完整。HMMER 用 profile HMM 搜索蛋白质家族（`hmmbuild` 由 MSA 建模型，`hmmsearch`/`hmmscan` 搜索）；Pfam 用它组织蛋白质 domain，每个 Pfam 家族就是一个由种子比对训练的 profile HMM。

### 三类状态

- **Match 状态 $M_k$**：发射第 $k$ 个共识位置的残基分布，相当于 PWM 的第 $k$ 列。
- **Insert 状态 $I_k$**：位于第 $k$ 与 $k+1$ 个共识位置之间，发射背景式残基分布，并可以自循环（$I_k\to I_k$）产生任意长度的插入。自循环概率决定插入长度的几何分布。
- **Delete 状态 $D_k$**：不发射字符，只跳过第 $k$ 个共识位置，用来表示查询序列在该列缺失。

Delete 是 silent state（静默状态）。forward/Viterbi 实现必须按拓扑顺序传播它，不能把它当作消耗一个观测的普通状态：在 $D_k$ 上停留不前进 query 坐标，所以同一 query 位置上的 $D_{k-1}\to D_k$ 必须先算完 $k-1$ 再算 $k$。

### 二维动态规划

与普通 HMM 不同，profile HMM 的 DP 有两个坐标：query 前缀长度 $i$ 与 profile 位置 $k$。三张表的含义是：$M_{i,k}$ 同时消费 query 字符和 profile 列（对角移动），$I_{i,k}$ 只消费 query 字符（纵向移动），$D_{i,k}$ 只推进 profile（横向移动）。这与成对比对中“匹配 / 插入 / 删除”三种移动一一对应，只是每一步的分数换成了位置特异的 log 概率。

下面给出固定转移参数的最小 Viterbi 实现；`match_emit[k]` 与 `insert_emit[k]` 均为 log emission probability，`log_transition` 对所有位置共用同一组 M/I/D 间转移（真实模型中转移也是位置特异的）。

```python
import math

def match_emissions_from_msa(
    alignment, alphabet, match_threshold=0.5, pseudocount=0.5
):
    """由 MSA 的 gap-poor 列估计 1-based Match-state log emissions。"""
    if not alignment or len({len(seq) for seq in alignment}) != 1:
        raise ValueError("alignment must be nonempty and rectangular")
    if not 0 < match_threshold <= 1 or pseudocount <= 0:
        raise ValueError("invalid threshold or pseudocount")

    match_columns = []
    emissions = [None]  # 第 0 项对应虚拟 begin state。
    for column_index, column in enumerate(zip(*alignment)):
        occupancy = sum(symbol != "-" for symbol in column) / len(column)
        if occupancy < match_threshold:
            continue
        counts = {symbol: pseudocount for symbol in alphabet}
        for symbol in column:
            if symbol != "-":
                if symbol not in counts:
                    raise ValueError(f"symbol outside alphabet: {symbol}")
                counts[symbol] += 1
        total = sum(counts.values())
        emissions.append({
            symbol: math.log(counts[symbol] / total)
            for symbol in alphabet
        })
        match_columns.append(column_index)
    return match_columns, emissions


def profile_hmm_viterbi(
    query, match_emit, insert_emit, log_transition
):
    """将 query Viterbi 对齐到含 M/I/D 状态的线性 profile HMM。

    match_emit 和 insert_emit 使用 1-based profile：第 0 项为 None。
    log_transition[(from_state, to_state)] 给出位置无关 log 概率。
    """
    length = len(match_emit) - 1
    if length <= 0 or len(insert_emit) != length + 1:
        raise ValueError("emission lists must use the same 1-based length")

    n = len(query)
    neg_inf = float("-inf")
    M = [[neg_inf] * (length + 1) for _ in range(n + 1)]
    I = [[neg_inf] * (length + 1) for _ in range(n + 1)]
    D = [[neg_inf] * (length + 1) for _ in range(n + 1)]
    back_m = [[None] * (length + 1) for _ in range(n + 1)]
    back_i = [[None] * (length + 1) for _ in range(n + 1)]
    back_d = [[None] * (length + 1) for _ in range(n + 1)]
    M[0][0] = 0.0  # begin state 与虚拟 M_0 合并。

    def transition(previous, current):
        return log_transition.get((previous, current), neg_inf)

    # 空 query 只能沿 silent delete chain 跳过 profile 列。
    for k in range(1, length + 1):
        choices = [
            (M[0][k - 1] + transition("M", "D"), "M"),
            (I[0][k - 1] + transition("I", "D"), "I"),
            (D[0][k - 1] + transition("D", "D"), "D"),
        ]
        D[0][k], back_d[0][k] = max(
            choices, key=lambda item: item[0]
        )

    for i in range(1, n + 1):
        symbol = query[i - 1]
        for k in range(1, length + 1):
            # Match 从左上角三种状态进入，并发射当前 query 字符。
            choices = [
                (M[i - 1][k - 1] + transition("M", "M"), "M"),
                (I[i - 1][k - 1] + transition("I", "M"), "I"),
                (D[i - 1][k - 1] + transition("D", "M"), "D"),
            ]
            best, back_m[i][k] = max(choices, key=lambda item: item[0])
            M[i][k] = best + match_emit[k].get(symbol, neg_inf)

            # Insert 消费 query，但停留在同一 profile 位置 k。
            choices = [
                (M[i - 1][k] + transition("M", "I"), "M"),
                (I[i - 1][k] + transition("I", "I"), "I"),
                (D[i - 1][k] + transition("D", "I"), "D"),
            ]
            best, back_i[i][k] = max(choices, key=lambda item: item[0])
            I[i][k] = best + insert_emit[k].get(symbol, neg_inf)

            # Delete 不发射字符，因此使用本行 k-1 的已计算状态。
            choices = [
                (M[i][k - 1] + transition("M", "D"), "M"),
                (I[i][k - 1] + transition("I", "D"), "I"),
                (D[i][k - 1] + transition("D", "D"), "D"),
            ]
            D[i][k], back_d[i][k] = max(
                choices, key=lambda item: item[0]
            )

    score, state = max(
        [(M[n][length], "M"), (I[n][length], "I"),
         (D[n][length], "D")],
        key=lambda item: item[0],
    )
    if score == neg_inf:
        raise ValueError("query has no finite path through the profile")
    i, k = n, length
    path = []
    while i > 0 or k > 0:
        if state == "M":
            path.append((f"M{k}", query[i - 1]))
            state = back_m[i][k]
            i -= 1
            k -= 1
        elif state == "I":
            path.append((f"I{k}", query[i - 1]))
            state = back_i[i][k]
            i -= 1
        else:
            path.append((f"D{k}", "-"))
            state = back_d[i][k]
            k -= 1
    return score, list(reversed(path))
```

代码把 begin/end 简化为虚拟 $M_0$ 和“在第 $L$ 列（即最后一个 profile 位置）取最大值”，不含 HMMER 的 local-entry、local-exit 与 null model，因此是一种全局（global）比对模式；另外由于第 0 列没有计算 $I_{i,0}$，query 在第一个 match 列之前的 N 端插入也无法表示。其关键仍是 delete state 的更新顺序：同一 query 行内从小 $k$ 向大 $k$ 传播，确保 silent transition 不会错误地消费字符。

### 从 MSA 构建完整的 profile

构建 profile 时，通常先由 MSA 决定哪些列属于 match 列（如上面代码中非 gap 比例 `occupancy` 不小于 `match_threshold`=0.5、即 gap 比例不超过一半的列），再估计参数。为了不让少数相似序列主导模型，需要做序列加权（sequence weighting，下调近乎重复的序列）、估计有效序列数（effective sequence number），并用 Dirichlet mixture prior 平滑发射分布（使从未观察到但理化性质相近的氨基酸获得合理概率）。最终的搜索分数通常是家族模型相对 null background 模型的 log-odds，而不是单纯的 $P(x\mid\text{profile})$：后者会随序列长度和组成变化，只有与背景模型比较才能判断“像家族”还是“只是普通蛋白”。

`match_emissions_from_msa` 只估计了 Match emission。完整训练还要：把 match 列之间的 gap 段映射为 Delete 状态，把非 match 列中的残基映射为 Insert 状态，再统计 M/I/D 之间的转移计数与 Insert emission；序列加权与先验应在计数归一化之前加入。这样形成的 profile HMM 才同时编码 MSA 的列分布与 indel 模式。

### HMMER 的加速流水线

对每条数据库序列都跑完整的 Forward 很昂贵。HMMER3 的 acceleration pipeline 先用便宜的 filter 排除明显不匹配的序列：第一层 MSV filter 只计算多个无 gap 局部片段的最佳和，并用 SIMD 向量化实现；通过者再进入 Viterbi filter，最后才做完整的 Forward/Backward 推断与 domain 定义。默认阈值下绝大多数序列在第一层就被淘汰，而灵敏度损失很小。这与 BLAST 的 seed-and-extend 体现了同一个工程原则：先用高召回、低成本的阶段缩小候选，再在候选上做昂贵的精确计算。

> [!INFO]+ HMM 与比对
>
> 当替换分数和 gap 罚分可以解释为 pair-HMM 路径的 log-odds 时，Needleman–Wunsch 对应该模型的 Viterbi 解；将 max-product 改为 sum-product 可汇总全部 alignment 路径，得到两条序列“以任意方式比对”的总概率。两种递推都是对隐变量路径的动态规划。

<br>

> [!EXAMPLE]+ 例5：基因区段识别
>
> 两状态 HMM 可令 $z_t\in\{\text{GC-rich},\text{AT-rich}\}$，每个状态发射不同碱基分布（例如 GC-rich 状态下 G、C 概率较高），较高的 self-transition（如 $a_{ii}$ 接近 1）表示区段具有连续性：期望区段长度为 $1/(1-a_{ii})$。Viterbi 给出一条最可能的 segmentation；后验概率 $P(z_t=\text{GC-rich}\mid x)$ 则显示边界附近的不确定性：远离边界处后验接近 0 或 1，边界附近则出现中间值。若真实 GC 组成是连续变化的，两个离散状态只是粗略近似。

> [!NOTE]+ 小结
>
> - HMM = 初始分布 + 转移 + 发射；联合概率按“先转移再发射”逐步相乘。
> - Forward 对所有路径求和得 $P(x)$；Backward 补上后缀，二者相乘得单点后验。
> - Viterbi 用 max 代替求和并保存回溯指针，得到一条合法的最可能路径；后验解码逐位最优但路径可能不合法。
> - Baum–Welch 是 HMM 上的 EM：期望计数来自 forward–backward，需 scaling 或 log 空间、伪计数、多次重启和验证集停止。
> - Profile HMM 用 Match/Insert/Delete 把 MSA 编码为位置特异的替换与 indel 模型；Delete 是静默状态，需按拓扑顺序更新。

<br>

<br>

<br>

# 系统发育（Phylogenetics）：从序列推断演化树

多序列比对把同源位置排成了列，但它本身并不回答“这些序列谁和谁更亲近、共同祖先在哪里分开”。例如手里有人、黑猩猩、小鼠和鸡的同一个基因：比对能告诉读者每一列的对应关系，而系统发育分析（phylogenetic analysis）要做的是根据这些列上的差异，推断出一棵描述共同祖先关系的树。这棵树既是演化生物学的研究对象，也是后续很多分析的“背景模型”——判断一个位点是否保守、两个位点是否协同变化，都需要先知道序列之间本来就有多相关。

核心想法可以一句话概括：**序列之间的差异是沿着一棵未知的树逐渐积累起来的，找到最能解释这些差异的树。**“最能解释”有不同的定义，于是产生了两大类方法：距离法先把每对序列的差异压缩成一个数，再由距离矩阵拼出树；字符法直接逐列考察每个位点的状态，用简约性、似然或后验概率比较候选树。本章依次介绍这两类方法，最后讨论如何用 bootstrap 评估分支的可靠性。

## 系统发育树的基本概念

### 树的组成：叶、内部节点、拓扑与枝长

系统发育分析试图从**现存序列**推断共同祖先关系，因此树上的节点分两种：

- <strong>叶（leaf）</strong>是观测到的物种、基因或蛋白，也就是 MSA 的每一行；
- <strong>内部节点（internal node）</strong>表示未被直接观测到的祖先，它们的序列只能推断，不能测量。

一棵树包含两层信息。<strong>拓扑（topology）</strong>表示分支关系，即哪些叶先聚在一起；<strong>枝长（branch length）</strong>表示沿这条边发生了多少演化，可以是预期替换数（expected substitutions per site），也可以是时间。两棵树拓扑相同、枝长不同，讲的是“谁和谁最近”相同、但“变化了多少”不同的故事。

### 有根树与无根树

<strong>有根树（rooted tree）</strong>有一个根节点代表所有叶的共同祖先，因此带有演化方向：从根往叶走就是时间向前。<strong>无根树（unrooted tree）</strong>只表示相对的分裂关系，例如“{人, 黑猩猩} 与 {小鼠, 鸡} 被一条边分开”，但不说明哪一侧更古老。很多方法（如 Neighbor Joining、可逆替换模型下的最大似然）天然只能给出无根树，根的位置需要额外信息确定，后文会反复遇到这一点。

### 输入、输出与两类方法

- **输入**通常是一个 MSA，也可以是由 MSA 计算出的成对距离矩阵（pairwise distance matrix）。
- **输出**是树的拓扑、枝长，以及每个分支的支持度（support）。
- <strong>距离法（distance-based methods）</strong>先把每对序列压缩成一个距离，再用 UPGMA 或 Neighbor Joining 建树。优点是快，缺点是压缩过程丢掉了“哪些列在变”的信息。
- <strong>字符法（character-based methods）</strong>直接使用每个比对列（alignment column），以简约性（parsimony）、似然（likelihood）或贝叶斯后验（Bayesian posterior）比较不同的树。更慢，但能利用全部位点信息并显式建模替换过程。

### 与前后章节的关系，以及 guide tree 的区别

系统发育依赖多序列比对给出的列对应：列对错了，树也会错。反过来，系统发育又影响对协变（covariation）与功能保守性的解释——两条序列在某位点相同，可能是功能约束，也可能只是它们刚刚分开、还没来得及变化。

需要特别区分两种“树”。多序列比对中的 **guide tree** 主要服务于渐进式比对的计算顺序，只要“先合并相似序列”这个目的达到即可，不追求演化上的正确；<strong>系统发育树（phylogenetic tree）</strong>则试图回答真实的演化历史。两者即使拓扑相同，含义也不同，不能把 guide tree 直接当作演化结论引用。

<br>

## 距离法：UPGMA 与 Neighbor Joining

距离法的出发点很朴素：如果能给每对序列一个“演化距离”，而这些距离恰好等于树上两叶之间路径的枝长之和（这种距离称为**加性距离**，additive distance），那么从距离矩阵反推树就成了一个组合问题。实际流程分两步：先从比对计算距离，再由距离建树。下面先讲距离怎么算，再讲两种建树算法。

### 从比对计算距离：p-distance 与 Jukes–Cantor 校正

对已比对的 DNA，最简单的距离是 **p-distance**：两条序列之间不匹配列（mismatch column）所占的比例，记作 $p\in[0,1]$。它的问题在于多次替换（multiple substitution）可能发生在同一位置：某位点先 A→G、再 G→A，观察上看不出任何变化；先 A→G、再 G→T，观察上只算一次差异。因此观察距离会**低估**真实发生的变化数，而且演化时间越长低估越严重——这种现象称为饱和（saturation）。

<strong>Jukes–Cantor 模型（JC69）</strong>是校正这一偏差的最简单模型。它假设四种碱基的平衡频率相同（都是 $1/4$），且任意两种碱基之间的替换率相同。在这些假设下，可以推出两条序列在某位点不同的概率 $p$ 与真实距离 $d$（每位点的预期替换数）之间的关系 $p=\frac34\left(1-e^{-4d/3}\right)$，反解得到修正距离

$$
d_{\mathrm{JC}}=-\frac{3}{4}\ln\left(1-\frac{4}{3}p\right)
$$

这个式子的直觉是：当 $p$ 很小时，$\ln(1-x)\approx -x$，所以 $d_{\mathrm{JC}}\approx p$，几乎不需要校正；$p$ 越大，校正量越大。上限 $0.75$ 也有清楚的含义：演化无限长之后，两条序列在某位点相同只是随机巧合，概率为 $1/4$，因此 $p$ 最多趋近 $3/4$。

> [!EXAMPLE]+ 例1 Jukes–Cantor 校正的数值效果
>
> 按上式计算（自然对数）：
>
> | 观察 p-distance $p$ | $d_{\mathrm{JC}}$ |
> | :--- | :--- |
> | 0.1 | 约 0.107 |
> | 0.3 | 约 0.383 |
> | 0.5 | 约 0.824 |
> | 0.7 | 约 2.031 |
>
> 可以看到 $p=0.1$ 时校正只有 7%，而 $p=0.7$ 时真实替换数估计已接近观察值的 3 倍：观察到的差异越多，被“藏起来”的重复替换就越多。

当 $p\rightarrow0.75^{-}$ 时，$1-\frac43p\to0^+$，距离发散到无穷。有限样本中若得到 $p\ge0.75$，对数的参数小于等于零，JC69 距离没有有限实数估计——这表示数据严重饱和（saturation），或者模型根本不适配，此时应该换更合适的模型或放弃用这对序列估计距离，而不是强行截断。

JC69 的假设很强。更复杂的模型逐步放松它们：**K80**（Kimura 双参数）区分转换（transition，嘌呤↔嘌呤、嘧啶↔嘧啶）与颠换（transversion）的速率；**HKY** 在此基础上允许不同的平衡频率（equilibrium frequency）；**GTR**（general time-reversible）让六类替换各有速率。模型越复杂越贴近真实，但需要估计的参数也越多。

### UPGMA：假设分子钟的平均链接聚类

**UPGMA**（Unweighted Pair Group Method with Arithmetic mean）本质上是层次聚类中的平均链接（average linkage）。算法每次合并当前距离最近的两个簇，并按簇大小更新新簇到其余簇的平均距离。合并簇 $A,B$ 后，它们与簇 $C$ 的距离按成员数加权：

$$
d(A\cup B,C)=
\frac{|A|d(A,C)+|B|d(B,C)}{|A|+|B|}
$$

其中 $|A|$ 是簇 $A$ 中叶的个数。这个加权平均等价于“$A\cup B$ 中每片叶与 $C$ 中每片叶距离的算术平均”，所以叫 arithmetic mean；“unweighted”指每片叶在平均中权重相同。新节点放在高度 $d(A,B)/2$ 处，两侧到叶的距离相等。

这样得到的是一棵**有根的超度量树（rooted ultrametric tree）**：所有叶到根的距离相等。这正对应<strong>分子钟（molecular clock）</strong>假设——所有谱系以相同速率积累替换。UPGMA 的隐含假设就是分子钟；若某些谱系演化得特别快，违反该假设，UPGMA 会产生错误的拓扑，因为快速谱系与其他序列的距离被拉大，最近的一对不一定是真正的姐妹群。

### Neighbor Joining：不要求分子钟的邻接法

<strong>Neighbor Joining（NJ，Saitou 与 Nei，1987）</strong>解决的正是 UPGMA 的这个弱点。它不要求超度量的分子钟，只要距离近似加性即可，输出通常是一棵**无根的加性树（unrooted additive tree）**。

关键在于“选哪一对合并”。直接选距离最小的一对会被长枝误导：一条很短的叶和一条很长的叶可能是姐妹群，但它们的距离并不小。NJ 的修正是同时考虑每个节点与**所有其他节点**的总距离——一个总体离大家都很远的节点（长枝）应当被“扣分”。对当前 $n$ 个节点，NJ 每轮最小化 $Q$-criterion：

$$
Q(i,j)=(n-2)d(i,j)-\sum_kd(i,k)-\sum_kd(j,k)
$$

其中求和遍历当前所有节点 $k$。减去行和 $\sum_kd(i,k)$ 正是在补偿长枝：如果 $i$ 本身枝很长，它的行和大，$Q$ 就被拉低，从而有机会与真正的邻居配对。可以证明，对加性距离，最小 $Q$ 的一对一定是真实树上的一对“邻居”（cherry）。

选出 $i,j$ 后，引入新内部节点 $u$ 连接二者。$u$ 到其余节点 $k$ 的距离为

$$
d(u,k)=\frac{d(i,k)+d(j,k)-d(i,j)}{2}
$$

推导很直接：在加性树上，$d(i,k)=\delta(i,u)+d(u,k)$，$d(j,k)=\delta(j,u)+d(u,k)$，两式相加并利用 $\delta(i,u)+\delta(j,u)=d(i,j)$，即得上式。而 $i$ 侧的分支长度为

$$
\delta(i,u)=\frac12d(i,j)
+\frac{\sum_kd(i,k)-\sum_kd(j,k)}{2(n-2)}.
$$

直觉是：先把 $d(i,j)$ 平分，再根据 $i$ 与 $j$ 平均离其他节点的远近差异做调整——平均更远的一方枝更长。$j$ 侧为 $\delta(j,u)=d(i,j)-\delta(i,u)$。然后用 $u$ 替换 $i,j$，节点数减一，重复直到只剩两个节点。每轮计算 $Q$ 需要 $O(n^2)$，共 $n$ 轮，总复杂度 $O(n^3)$。

NJ 得到的是无根树。若需要根，必须借助外类群（outgroup，已知在所有研究对象之外分出的序列，根放在它的连接处）、中点定根（midpoint rooting，把根放在最长叶间路径的中点，隐含近似等速假设）或其他时间信息。

> [!EXAMPLE]+ 例2 同一距离矩阵上的 UPGMA 与 NJ
>
> 设四个类群的距离矩阵为
>
> | | A | B | C | D |
> | :--- | :--- | :--- | :--- | :--- |
> | A | 0 | 5 | 9 | 9 |
> | B | 5 | 0 | 10 | 10 |
> | C | 9 | 10 | 0 | 8 |
> | D | 9 | 10 | 8 | 0 |
>
> **UPGMA**：最近的是 A、B（5），合并于高度 2.5；$d(AB,C)=d(AB,D)=9.5$，于是 C、D（8）合并于高度 4；最后 $d(AB,CD)=(9+9+10+10)/4=9.5$，根高 4.75。下文代码输出 `((A:2.5,B:2.5):2.25,(C:4,D:4):0.75);`（`upgma` 从 `frozenset` 中取出一对时顺序不固定，同一兄弟节点内的先后可能变成 `(B:2.5,A:2.5)`、`(D:4,C:4)`，树本身不变）。注意 A、B 被强制等长 2.5，树上 A–C 的距离变成 $2\times4.75=9.5$，不再等于原矩阵中的 9。
>
> **NJ**：行和为 $A{:}23,\ B{:}25,\ C{:}27,\ D{:}27$。$Q(A,B)=2\times5-23-25=-38$，$Q(C,D)=2\times8-27-27=-38$，其余四对都是 $-32$。(A,B) 与 (C,D) 并列最小，代码中 `min` 取枚举顺序中的第一个，所以先合并 A、B（先合并 C、D 得到的树相同）。分支长度 $\delta(A,u)=2.5+(23-25)/4=2$，$\delta(B,u)=3$；新节点 $d(u,C)=d(u,D)=(9+10-5)/2=7$。下一轮剩 C、D、$u$ 三个节点，三对的 $Q$ 都等于 $-22$（三个节点时 $Q$ 必然全相等），按枚举顺序合并 C、D，各长 4，剩余内部边长 $(7+7-8)/2=3$。代码输出 `((A:2,B:3):1.5,(C:4,D:4):1.5);`（最后一条边被平分成两段 1.5 写成 Newick）。逐一核对：A–B $=2+3=5$，A–C $=2+3+4=9$，B–C $=3+3+4=10$，C–D $=8$，全部与输入一致。
>
> 这个矩阵是加性的但不是超度量的（B 比 A 演化得快）。NJ 精确恢复了枝长，UPGMA 拓扑虽然这里碰巧正确，但枝长被分子钟假设扭曲了。

### 实现：UPGMA 与 NJ 的教学代码

下面的实现接受嵌套字典形式的距离矩阵 `distance[a][b]`，先由辅助函数转成以 `frozenset({i,j})` 为键的表并检查对称、非负，然后分别执行上述两种算法，返回 Newick 字符串（用括号嵌套表示树、冒号后为枝长的通用文本格式）。

```python
def _pairwise_distance_table(distance):
    """把嵌套距离字典转成以 frozenset({i,j}) 为键的表。"""
    taxa = list(distance)
    if not taxa:
        raise ValueError("distance matrix must contain at least one taxon")
    table = {}
    for i, left in enumerate(taxa):
        for right in taxa[i + 1:]:
            a = distance[left][right]
            b = distance[right][left]
            if a < 0 or abs(a - b) > 1e-12:
                raise ValueError("distance matrix must be symmetric/nonnegative")
            table[frozenset((left, right))] = float(a)
    return taxa, table


def upgma(distance):
    """由距离矩阵构建 rooted ultrametric UPGMA tree，返回 Newick。"""
    active, dist = _pairwise_distance_table(distance)
    tree = {taxon: str(taxon) for taxon in active}
    size = {taxon: 1 for taxon in active}
    height = {taxon: 0.0 for taxon in active}
    next_id = 0

    while len(active) > 1:
        left, right = min(
            (tuple(pair) for pair in dist if pair <= set(active)),
            key=lambda pair: dist[frozenset(pair)],
        )
        new = ("cluster", next_id)
        next_id += 1
        new_height = dist[frozenset((left, right))] / 2.0
        tree[new] = (
            f"({tree[left]}:{new_height-height[left]:.6g},"
            f"{tree[right]}:{new_height-height[right]:.6g})"
        )
        size[new] = size[left] + size[right]
        height[new] = new_height

        others = [node for node in active if node not in (left, right)]
        for node in others:
            # 按簇成员数加权，等价于所有跨簇叶对距离的平均。
            value = (
                size[left] * dist[frozenset((left, node))]
                + size[right] * dist[frozenset((right, node))]
            ) / size[new]
            dist[frozenset((new, node))] = value
        active = others + [new]

    return tree[active[0]] + ";"


def neighbor_joining(distance):
    """由距离矩阵构建 Neighbor-Joining tree，返回一种 unrooted Newick 表示。"""
    active, dist = _pairwise_distance_table(distance)
    if len(active) < 2:
        raise ValueError("Neighbor Joining needs at least two taxa")
    tree = {taxon: str(taxon) for taxon in active}
    next_id = 0

    while len(active) > 2:
        n = len(active)
        row_sum = {
            node: sum(
                dist[frozenset((node, other))]
                for other in active if other != node
            )
            for node in active
        }
        left, right = min(
            (
                (active[i], active[j])
                for i in range(n) for j in range(i + 1, n)
            ),
            key=lambda pair: (
                (n - 2) * dist[frozenset(pair)]
                - row_sum[pair[0]] - row_sum[pair[1]]
            ),
        )
        d_lr = dist[frozenset((left, right))]
        limb_left = 0.5 * d_lr + (
            row_sum[left] - row_sum[right]
        ) / (2 * (n - 2))
        limb_right = d_lr - limb_left

        new = ("node", next_id)
        next_id += 1
        tree[new] = (
            f"({tree[left]}:{limb_left:.6g},"
            f"{tree[right]}:{limb_right:.6g})"
        )
        others = [node for node in active if node not in (left, right)]
        for node in others:
            dist[frozenset((new, node))] = 0.5 * (
                dist[frozenset((left, node))]
                + dist[frozenset((right, node))]
                - d_lr
            )
        active = others + [new]

    left, right = active
    final_length = dist[frozenset((left, right))] / 2.0
    return (
        f"({tree[left]}:{final_length:.6g},"
        f"{tree[right]}:{final_length:.6g});"
    )
```

阅读代码时有几点值得注意：

- UPGMA 中每个簇记录 `height`，子边长度等于“新高度减子簇高度”，因此所有叶到根等距；`size` 用于加权平均。
- NJ 中 `row_sum` 就是公式里的 $\sum_k d(i,k)$；`limb_left` 对应 $\delta(i,u)$。循环在剩两个节点时停止，最后一条边在 Newick 中被平分写成两段，这只是无根树的一种文本表示，根的位置没有生物学含义。
- 两个函数都假设输入距离完整且无缺失。非加性距离、抽样噪声或模型失配可能使 NJ 估计出**负的枝长**（negative branch length）；代码保留该结果以暴露数据与树模型的冲突，生产软件可按明确规则截断（例如置零并把差额转给相邻枝）或重新拟合。

<br>

## 字符法：简约、似然与贝叶斯推断

距离法把两条长度为数百的序列压缩成一个数，必然丢失信息：两对序列的 p-distance 都是 0.2，可能一对是在同样几个快变位点上不同，另一对是在分散的位点上不同，距离法无法区分。<strong>字符法（character-based methods）</strong>直接使用每个类群（taxon）在每个位点的状态，而不是先把整对序列压缩成一个距离。它的问题变成：给定一棵候选树，如何为这棵树打分？三种主流打分方式是：

- **最大简约（maximum parsimony）**：寻找需要最少替换次数的树。直观，不需要替换模型，但会受长枝吸引（long-branch attraction）影响。
- **最大似然（maximum likelihood, ML）**：指定替换模型，在给定树与枝长下计算数据出现的概率（似然），选似然最大的树。
- **贝叶斯推断（Bayesian inference）**：在似然之上结合树与参数的先验，对后验分布进行采样，而不是只给一棵最优树。

### 什么是“字符”

这里的“字符”（character）指一个可观测的离散性状。分子数据中它是一个比对列，状态取自 A/C/G/T 或 20 种氨基酸；也可以是形态学或功能性状的二值状态（binary state），例如“有/无翅膀”“是否具有某酶活性”。

二值编码（binary coding）必须说明 0/1 的含义：是**缺失/存在（absence/presence）**，还是**祖先态/衍生态（ancestral/derived）**。后者带有方向信息，若没有外类群（outgroup），通常无法确定哪个状态是祖先态，也就不能这样编码。

### 最大简约与 Fitch 算法

简约法的思想来自“奥卡姆剃刀”：能用更少变化解释数据的树更可信。给定一棵树，一个位点的<strong>简约分数（parsimony score）</strong>是为所有内部节点指定祖先状态后，树上发生状态变化的最少边数。

对固定二叉树和一个字符，**Fitch 算法**用一次自底向上（后序遍历）的扫描求出这个最小值。它为每个节点维护一个“可取的最优状态集合”：

1. 叶节点的集合就是观测状态（歧义字符可以是多个状态）。
2. 内部节点看两个子节点的集合：两个子集合**有交集**时取交集，不增加代价——祖先取交集中的状态即可让两条子边都不变；**无交集**时取并集，并把变化数加一——无论祖先取什么，至少有一条子边要变一次。
3. 根节点处累计的变化数就是该位点的简约分数。

它在线性时间内给出该拓扑的无序简约分数（unordered parsimony score，任意两状态间变化代价都为 1）。但这只是“给一棵树打分”；寻找全局最优树仍需在巨大的拓扑空间（topology space）中搜索，最大简约树问题本身是 NP-hard。

> [!EXAMPLE]+ 例3 Fitch 算法的手算
>
> 树为 $((A,B),(C,D))$。
>
> - 位点 1 观测为 A=`A`，B=`G`，C=`A`，D=`A`。节点 $(A,B)$：$\{A\}\cap\{G\}=\varnothing$，取并集 $\{A,G\}$，代价 +1；节点 $(C,D)$：$\{A\}\cap\{A\}=\{A\}$；根：$\{A,G\}\cap\{A\}=\{A\}$，不加代价。总分 1。
> - 位点 2 观测为 A=`A`，B=`G`，C=`G`，D=`A`。两个子节点都得到 $\{A,G\}$ 且各 +1；根处交集为 $\{A,G\}$，不再加。总分 2。
>
> 在树 $((A,D),(B,C))$ 上，位点 2 只需 1 次变化，所以这一列“支持”后者。只有至少两种状态各出现至少两次的列才能区分拓扑，这样的列称为简约信息位点（parsimony-informative site）；位点 1 在任何四叶拓扑上都是 1 分，对比较拓扑没有贡献。

```python
def fitch_parsimony_score(root, children, observed):
    """计算固定 rooted binary tree 上一个离散 character 的 Fitch score。"""
    def postorder(node):
        node_children = children.get(node, [])
        if not node_children:
            states = observed.get(node)
            if isinstance(states, str):
                states = {states}
            else:
                states = set(states or [])
            if not states:
                raise ValueError(f"leaf {node} has no compatible state")
            return states, 0
        if len(node_children) != 2:
            raise ValueError("this teaching implementation requires binary nodes")

        left_states, left_score = postorder(node_children[0])
        right_states, right_score = postorder(node_children[1])
        intersection = left_states & right_states
        if intersection:
            return intersection, left_score + right_score
        return left_states | right_states, left_score + right_score + 1

    _, score = postorder(root)
    return score
```

以上例为例，调用方式是 `children = {'r': ['x', 'y'], 'x': ['A', 'B'], 'y': ['C', 'D']}`，`observed = {'A': 'A', 'B': 'G', 'C': 'G', 'D': 'A'}`，返回 2。叶的观测值也可以是集合（如 `{'A', 'G'}` 表示 IUPAC 字符 R）。

对整个 MSA，总简约分数是各列分数之和。使用时需要注意：

- gap、歧义字符与加权替换都需要显式约定（例如 gap 视为第五种状态还是缺失）；
- Fitch 的单位代价模型不区分转换与颠换，也不估计枝长。

### 简约法的局限：长枝吸引与饱和

简约法不需要完整的替换概率，但代价是把所有变化视作同成本，并隐含“变化很少”的假设。当两条不相邻的谱系都演化得很快（长枝）时，它们会在许多位点上独立地变成同一状态（平行变化），简约法把这些巧合误读为共同祖先的证据，从而把快速演化的长枝错误地聚在一起，即**长枝吸引（long-branch attraction）**。这种错误在数据越多时反而越确定，统计上称为不一致（inconsistent）。

另一个角度：最大简约的总分由发生变化的信息位点（informative column）决定；高度可变的列可能贡献大量变化并主导拓扑，但它们也最容易出现饱和（saturation）与同塑性（homoplasy，非同源的趋同相似）。加权简约（weighted parsimony）可以改变不同变化的代价，例如让颠换比转换更“贵”，但权重本身仍是一种模型假设。

需要注意，似然法也并非自动正确：若替换模型（substitution model）严重失配，似然法同样会产生高置信度的错误树。

### 最大似然与 Felsenstein 剪枝算法

最大似然法显式整合枝长与替换概率，因此能表达“同一位点发生多次未观测替换”的可能性，代价是模型选择和树空间搜索更复杂。

<strong>模型设定。</strong>给定有根树，每条边 $(\text{parent},\text{child})$ 有长度 $t$，替换模型给出转移概率矩阵 $P(t)$，其中 $P_{ab}(t)$ 是祖先状态为 $a$ 时、经过长度 $t$ 后子节点状态为 $b$ 的概率。根的状态按平衡频率 $\pi$ 分布。一个位点的似然是“所有叶的观测状态”的概率，需要对所有内部节点的未知状态求和。直接枚举有 $|\Sigma|^{(\text{内部节点数})}$ 种组合，指数爆炸。

<strong>Felsenstein 剪枝算法（pruning algorithm）</strong>在树上做动态规划，把这种指数枚举降为每个位点 $O(|V||\Sigma|^2)$，其中 $|V|$ 是节点数、$|\Sigma|$ 是状态数（DNA 为 4）。对 $L$ 个独立位点和 $R$ 个离散速率类别，总复杂度约为 $O(LR|V||\Sigma|^2)$。

定义条件似然（conditional likelihood）$L_v(a)$：给定节点 $v$ 的状态为 $a$ 时，$v$ 下方所有叶的观测数据出现的概率。

- 对于叶节点 $v$，若观察到碱基 $x$，则 $L_v(a)=1[a=x]$：只有状态等于观测值时概率为 1。
- 内部节点 $v$ 有子节点 $u,w$（对应枝长 $t_u,t_w$）时：

$$
L_v(a)=
\left(\sum_bP_{ab}(t_u)L_u(b)\right)
\left(\sum_cP_{ac}(t_w)L_w(c)\right)
$$

每个括号表示“从 $a$ 经过一条边演化到子节点任意状态，再生成子树数据”的概率；两棵子树在给定 $v$ 的状态后条件独立，所以相乘。这里每个节点、每个 $a$ 都要对 $b$ 求和，正是 $|\Sigma|^2$ 因子的来源。

在根处乘平衡频率并求和，得到该位点的似然：$P(\text{site})=\sum_a\pi_aL_{\text{root}}(a)$。若各列独立，总对数似然（log-likelihood）是各列对数似然之和。

<strong>为什么仍然要搜索。</strong>剪枝算法只解决“给定树求似然”；树拓扑数量随类群数超指数增长（$n$ 个叶的无根二叉树有 $(2n-5)!!$ 种），最大似然软件（如 RAxML、IQ-TREE、PhyML）使用 NNI（nearest-neighbor interchange，交换一条内部边两侧的子树）、SPR（subtree pruning and regrafting，剪下一棵子树接到别处）等局部搜索，而非穷举。

> [!EXAMPLE]+ 例4 两叶树上的剪枝计算与“滑轮原理”
>
> 在 JC69 下，长度为 $t$ 的边上 $P_{aa}(t)=\frac14+\frac34e^{-4t/3}$，$P_{ab}(t)=\frac14-\frac14e^{-4t/3}\ (a\ne b)$。取 $t=0.1$，得 $P_{aa}\approx0.9064$，$P_{ab}\approx0.0312$。
>
> 根下挂两片叶，两条边长都是 0.1，平衡频率均为 0.25：
>
> - 两叶都是 `A`：$L_{\text{root}}(A)=0.9064^2$，$L_{\text{root}}(b)=0.0312^2$（$b\ne A$），似然 $=0.25\,(0.9064^2+3\times0.0312^2)\approx0.2061$；
> - 叶为 `A` 与 `G`：似然 $\approx0.01463$；
> - 一片叶缺失：缺失叶的条件似然向量全为 1，似然恰好为 0.25，即只剩另一片叶的边缘概率。
>
> 第一个数恰好等于 $0.25\times P_{AA}(0.2)$：对可逆模型，根可以沿树滑动而似然不变（Felsenstein 的 pulley principle）。这正是可逆模型下的似然树天然无根的原因。

```python
import numpy as np

def felsenstein_site_likelihood(
    root, children, observed, transition, equilibrium, states
):
    """计算一个 alignment column 在固定有根树与替换模型下的似然。

    children[node] 给出子节点；transition[(parent, child)] 是矩阵
    P[a, b]；observed[leaf] 是叶节点状态，None 表示缺失。
    """
    state_index = {state: i for i, state in enumerate(states)}
    q = len(states)

    def conditional(node):
        node_children = children.get(node, [])
        if not node_children:
            symbol = observed.get(node)
            if symbol is None:
                # 缺失数据与所有状态兼容，不能强制成某个碱基。
                return np.ones(q)
            if symbol not in state_index:
                raise ValueError(f"unknown state at leaf {node}: {symbol}")
            likelihood = np.zeros(q)
            likelihood[state_index[symbol]] = 1.0
            return likelihood

        # L_node(a) 是给定当前节点状态 a 时所有后代观测的概率。
        likelihood = np.ones(q)
        for child in node_children:
            P = np.asarray(transition[(node, child)], dtype=float)
            if P.shape != (q, q):
                raise ValueError("transition matrix has wrong shape")
            child_likelihood = conditional(child)
            likelihood *= P @ child_likelihood
        return likelihood

    equilibrium = np.asarray(equilibrium, dtype=float)
    if equilibrium.shape != (q,) or not np.isclose(equilibrium.sum(), 1.0):
        raise ValueError("equilibrium must be a probability vector")
    return float(equilibrium @ conditional(root))
```

代码中 `P @ child_likelihood` 的第 $a$ 个分量正是 $\sum_bP_{ab}L_{\text{child}}(b)$，对每个子节点相乘即得递推式；它对任意多叉节点同样成立。

**实现中的数值与建模细节：**

- <strong>下溢。</strong>教学实现直接在概率空间递归，深树、多位点时概率连乘会下溢为 0。实际程序通常在每个内部节点缩放条件似然向量（conditional-likelihood vector）并累计对数缩放因子（log scaling factor），或改用对数空间（log-space）计算；多位点计算还会缓存相同的位点模式（site pattern），相同列只算一次再乘以出现次数。
- <strong>缺失与歧义。</strong>叶节点若为缺失、gap 或 IUPAC 歧义字符，应对所有兼容状态置 1，而不是强制 one-hot。例如 `R`（A 或 G）对应向量 $(1,0,1,0)$。
- <strong>位点间速率异质性。</strong>真实位点的演化速率通常不同（功能关键位点几乎不变，第三密码子位置变得快），可用离散 Gamma 类别（discrete Gamma，把速率分布切成 $R$ 类，各算一次似然再平均）和不变位点成分（invariant-site component，+I）近似；这就是复杂度中 $R$ 的来源。
- <strong>方向。</strong>没有外类群或分子钟信息时，无根的似然树本身不提供演化方向。

### 贝叶斯系统发育

最大似然给出一棵“最好的树”，但没有直接回答“这个分支有多大可能是对的”。贝叶斯方法把树当作随机变量，目标是后验分布：

$$
P(T,\ell,\theta\mid X)
\propto P(X\mid T,\ell,\theta)
P(T,\ell,\theta)
$$

其中 $X$ 是比对数据，$T$ 是拓扑，$\ell$ 是枝长向量，$\theta$ 是替换模型参数（如 GTR 速率、Gamma 形状参数）。右边第一项是似然，仍由剪枝算法计算；第二项是先验。

后验的归一化常数需要对所有拓扑求和、对连续参数积分，无法直接计算，因此用 **MCMC**（马尔可夫链蒙特卡洛）采样：在拓扑与连续参数空间中交替提议 NNI/SPR 拓扑移动、枝长更新和模型参数更新，按 Metropolis–Hastings 规则接受或拒绝。链收敛后，某个 clade 在样本中出现的比例就是它的后验概率。代表软件有 MrBayes、BEAST、RevBayes。

**两类分支支持度不可混用：**

- <strong>后验 clade 概率（posterior clade probability）</strong>是在给定模型、先验与数据之后，该 clade 的后验概率质量；
- **bootstrap support** 是重复有放回重采样比对列时，算法恢复出该 clade 的频率（见下一节）；
- 两者即使数值相似，也不能互换解释。经验上后验概率往往偏高，在模型失配时尤其如此。

<strong>MCMC 诊断。</strong>结论可靠的前提是链确实收敛。需要检查多条独立链是否混合良好（如拓扑分裂频率的标准差、PSRF 接近 1），有效样本量（effective sample size, ESS）是否足够（常用经验阈值 200 以上），丢弃合适的预烧期（burn-in），并用后验预测检验（posterior predictive check）评估模型拟合（model fit）。

<br>

## Bootstrap：评估分支的稳定性

得到一棵树之后，最自然的追问是：如果数据稍有不同，这棵树还会是这样吗？<strong>Bootstrap（Felsenstein，1985）</strong>用重采样模拟“稍有不同的数据”：从比对列中有放回地重采样，重复建树，某分支在重复树中出现的比例就是 **bootstrap support**。它衡量的是数据重采样下结果的稳定性，**不是**该 clade 为真的后验概率。

标准 bootstrap 流程是：

1. 原 MSA 有 $L$ 列；
2. 有放回地抽取 $L$ 列，组成一个重复数据集（replicate）——有的列被抽到多次，有的一次也没被抽到；
3. 对每个 replicate 用同一方法独立建树；
4. 把每棵 replicate 树中的分裂（split，一条内部边把叶分成的两部分）映射回主树；
5. 报告主树每个 split 在 replicate 中出现的频率，通常以百分比标在内部边上。

一般重复 100–1000 次。读数时常见的经验是：支持度 70% 以上被认为较可靠，低于 50% 的分支通常不应解释；但这只是惯例，不是显著性阈值。

因为抽样单位是比对列，bootstrap 默认各列近似**独立同分布**。基因连锁、重组和结构上的相关（例如 RNA 中成对配对的两个位点）会削弱这一假设，使支持度被高估。另外，若建树方法本身有系统偏差（如长枝吸引），bootstrap 会以很高的支持度反复给出同一个错误答案——它衡量的是精度，而不是准确性。

> [!WARNING]+ 基因树与物种树
>
> 基因复制（gene duplication）、水平转移（horizontal gene transfer）、不完全谱系排序（incomplete lineage sorting）和重组（recombination）会使基因树（gene tree）不等于物种树（species tree）。例如一个基因在祖先中复制成两个旁系同源拷贝，后续物种各自丢失不同的拷贝，用它建出的树会给出错误的物种关系。建树前还应检查重组、比对质量（alignment quality）和模型适配。

> [!NOTE]+ 小结
>
> - 系统发育树的叶是观测序列，内部节点是推断的祖先；拓扑与枝长分别回答“谁最近”和“变化多少”，有根树才有方向。
> - p-distance 低估真实替换，Jukes–Cantor 等模型做校正；$p\ge0.75$ 时 JC69 无有限解，提示饱和或模型失配。
> - UPGMA 隐含分子钟，输出有根超度量树；NJ 用 $Q$-criterion 补偿长枝，只需加性距离，输出无根树。
> - Fitch 算法线性时间给出固定树的简约分数；Felsenstein 剪枝算法把似然计算降为每位点 $O(|V||\Sigma|^2)$；两者都只解决打分，树空间仍需启发式搜索。
> - 简约易受长枝吸引，似然在模型失配时同样会错；贝叶斯后验概率与 bootstrap 支持度含义不同，不能互换。

<br>

<br>

# RNA 二级结构预测与协变分析

前面几章把生物分子当作字符串处理：比对、搜索、HMM 都只关心一维序列。RNA 打破了这种简化——一条 RNA 链会通过分子内的碱基配对折回自身，形成茎（stem）与环（loop），tRNA 的三叶草形、核糖开关（riboswitch）的配体结合口袋、rRNA 的催化中心都由这种配对结构决定。许多 RNA 的催化、识别和调控功能依赖配对结构，单看一级序列会漏掉重要约束：两条序列可能在序列上差异很大，却折成同样的结构、执行同样的功能。

因此 RNA 既是序列，也是能够通过分子内碱基配对折叠的结构对象。本章关注**二级结构（secondary structure）**：记录哪些位置形成 A–U、G–C 或 G–U 配对，通常先忽略精确的三维坐标。核心想法是：**只要配对之间不交叉，结构就可以按区间递归分解，于是能用区间动态规划在多项式时间内求最优结构。**

本章的方法层次如下：

- **单序列折叠**输入一条 RNA 序列，输出可以是配对集合、点括号表示（dot-bracket）、最小自由能（minimum free energy, MFE）结构，或碱基配对概率（base-pair probability）。
- **Nussinov 算法**用“最大配对数”这一最简目标展示区间 DP；
- **热力学方法**（Zuker/Turner）进一步区分 stacking、hairpin、bulge、internal loop 和 multiloop，最小化自由能；
- **多序列分析**利用补偿突变（compensatory mutation）识别在序列变化后仍被保留的配对。

RNA 这一章连接了前面的字符串 DP、HMM 中的概率集合（ensemble）思想与系统发育背景。协方差模型（covariance model）可以看成同时描述共识序列与成对结构约束的 Profile HMM 扩展；而伪结（pseudoknot）会破坏普通非交叉区间 DP 的分解条件，是本章所有算法的共同边界。

## 二级结构：表示与 Nussinov 算法

### 配对规则与点括号表示

二级结构常允许三种配对：Watson–Crick 配对 A–U、G–C，以及 G–U 摆动配对（wobble pair）。G–C 有三个氢键，最稳定；A–U 有两个；G–U 较弱但在 RNA 中很常见。

一个二级结构是一组配对 $(i,j)$，$i<j$，每个位置最多参与一个配对。为了能用 DP 求解，通常还要求**无伪结（pseudoknot-free）**：任意两对 $(i,j)$、$(k,l)$ 不能交叉，即不允许 $i<k<j<l$。在这一条件下，结构可以写成<strong>点括号（dot-bracket）</strong>字符串：配对的左端写 `(`，右端写 `)`，未配对写 `.`。例如 `((...))` 表示第 1–7 位与第 2–6 位配对，中间 3 个碱基构成发夹环（hairpin loop）。括号能正确匹配，恰好对应“配对不交叉”。

另外，发夹环太短在空间上不可能实现，因此要求配对的两个碱基之间至少隔若干个未配对位置，称为最小环长（minimum loop length），通常取 3。

### Nussinov 递推

Nussinov 算法（1978）的目标最简单：在上述约束下**最大化配对数**。设 $N_{i,j}$ 为子序列 $i..j$ 能形成的最大配对数。考察区间的两端，最优结构只可能是四种情况之一：

$$
N_{i,j}=\max\begin{cases}
N_{i+1,j}\\
N_{i,j-1}\\
N_{i+1,j-1}+\delta(i,j)\\
\max_{k}(N_{i,k}+N_{k+1,j})
\end{cases}
$$

递推中的四项分别表示：

- $i$ 不配对：去掉 $i$ 后剩下区间 $i+1..j$ 的最优解；
- $j$ 不配对：同理，剩下 $i..j-1$；
- $i,j$ 彼此合法配对：得到 1 对，再加上内部区间 $i+1..j-1$ 的最优解；
- 在 $k$ 处分割成两个独立子结构（bifurcation）：左半 $i..k$ 与右半 $k+1..j$ 各自最优，再相加。

其中当 $i,j$ 可合法配对且满足最小环长时 $\delta(i,j)=1$，否则该候选项取 $-\infty$（即不可选）。

为什么四种情况足够？因为无伪结：若 $i$ 与某个 $k<j$ 配对，那么 $k$ 左右两侧的配对不可能跨过 $k$，区间自然分成 $i..k$ 与 $k+1..j$，被分割项覆盖。正是“不交叉”这一条件保证了子问题彼此独立。

**填表顺序**：$N_{i,j}$ 依赖的都是更短的区间，所以按区间长度从短到长填表，保证所需子问题已经计算。初值为长度为 1 的区间 $N_{i,i}=0$。经典实现为 $O(n^3)$ 时间（$O(n^2)$ 个格子，每格枚举 $O(n)$ 个分割点 $k$）、$O(n^2)$ 空间。

**回溯（traceback）**：只有分数还不够，必须记录每格选择了哪一项，才能从 $N_{1,n}$ 一路恢复出配对集合与点括号结构，例如 `((...))`。

### 实现：Nussinov 与点括号转换

```python
PAIR = {("A", "U"), ("U", "A"), ("G", "C"),
        ("C", "G"), ("G", "U"), ("U", "G")}

def nussinov(seq, min_loop=3):
    """最大化非交叉 RNA 碱基对数，返回分数与 0-based 配对列表。"""
    seq = seq.upper().replace("T", "U")
    if set(seq) - set("ACGU"):
        raise ValueError("sequence must contain only A/C/G/U/T")
    if min_loop < 0:
        raise ValueError("min_loop must be nonnegative")
    n = len(seq)
    # dp[i][j] 是闭区间 seq[i:j+1] 的最大配对数。
    dp = [[0] * n for _ in range(n)]
    back = [[None] * n for _ in range(n)]

    # 按区间长度递增填表，确保所有更短子区间已经求解。
    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            choices = [(dp[i + 1][j], ("skip_i",))]
            choices.append((dp[i][j - 1], ("skip_j",)))
            if j - i > min_loop and (seq[i], seq[j]) in PAIR:
                inner = dp[i + 1][j - 1] if i + 1 <= j - 1 else 0
                choices.append((inner + 1, ("pair",)))
            for k in range(i, j):
                choices.append((dp[i][k] + dp[k + 1][j], ("split", k)))
            dp[i][j], back[i][j] = max(choices, key=lambda item: item[0])

    # 用显式 stack 回溯，避免深递归；每个 pair 只记录一次。
    pairs = []
    stack = [(0, n - 1)] if n else []
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        action = back[i][j]
        if action[0] == "skip_i":
            stack.append((i + 1, j))
        elif action[0] == "skip_j":
            stack.append((i, j - 1))
        elif action[0] == "pair":
            pairs.append((i, j))
            stack.append((i + 1, j - 1))
        else:
            k = action[1]
            stack.extend([(i, k), (k + 1, j)])
    return dp[0][n - 1] if n else 0, sorted(pairs)


def pairs_to_dot_bracket(length, pairs):
    """把无伪结、互不复用的 0-based pair 列表转换为 dot-bracket。"""
    structure = ["."] * length
    used = set()
    for left, right in pairs:
        if not 0 <= left < right < length or left in used or right in used:
            raise ValueError("invalid or reused pair index")
        structure[left], structure[right] = "(", ")"
        used.update((left, right))
    ordered = sorted(pairs)
    if any(i < k < j < l for i, j in ordered for k, l in ordered):
        raise ValueError("pseudoknots need an extended bracket alphabet")
    return "".join(structure)
```

关于这段代码：

- 条件 `j - i > min_loop` 表示配对碱基之间至少有 `min_loop` 个位置；默认 3 即要求 $j-i-1\ge3$。
- 分割循环 `for k in range(i, j)` 中 $k=i$ 时 `dp[i][i] + dp[i+1][j]` 与“跳过 $i$”等价，冗余但无害。
- `pairs_to_dot_bracket` 会检查交叉配对：普通点括号只有一种括号，无法表示伪结，伪结需要 `[]`、`{}` 等扩展括号字母表。
- 该代码只最大化配对数，用于展示区间 DP 与回溯，不应替代 ViennaRNA 等热力学实现（thermodynamic implementation）。

> [!EXAMPLE]+ 例5 一个小发夹
>
> 对序列 `GGGAAAUCC`（长度 9），`nussinov` 返回 `(3, [(0, 8), (1, 7), (2, 6)])`，即 G1–C9、G2–C8、G3–U7 三对，点括号为 `(((...)))`：一段 3 bp 的茎加一个 `AAA` 发夹环。若把 `min_loop` 设为 0，`nussinov('GGGAAAUCC', min_loop=0)` 返回 `(3, [(1, 8), (2, 7), (5, 6)])`，其中 A6–U7 是两个相邻碱基直接配对（环长为 0），在空间上不可能实现，这正是需要最小环长约束的原因。

### 从最大配对数到自由能：Turner 模型与 Zuker 算法

Nussinov 的“最大配对数”会把所有合法配对当作等价，这与物理事实相差很远：

- RNA 结构的稳定性主要来自**相邻碱基对的堆积（stacking）**，而不是单个配对本身；孤立的一个配对几乎不提供稳定性；
- 各种环都有熵罚（entropic penalty），环越大越不利；
- G–C 堆积远比 G–U 稳定。

<strong>Turner 最近邻模型（nearest-neighbor model）</strong>把结构分解为若干“环”（loop）——堆积（stacking）、发夹环（hairpin）、凸环（bulge，单侧有未配对碱基）、内环（internal loop，两侧都有）和多分支环（multiloop，三条及以上的茎汇合处）——每类环的自由能由实验测得的参数给出，结构总自由能是各环能量之和，目标改为最小化 $\Delta G$。

Zuker 型 DP 不能只把 Nussinov 的 $+1$ 改成任意配对能量。最近邻能量依赖环的类型与相邻碱基对，因此需要额外状态，例如：

- $W(i,j)$：区间 $i..j$ 上任意结构的最优自由能；
- $V(i,j)$：在“$i,j$ 已配对”前提下，区间内的最优自由能。

$V$ 再区分 $i,j$ 闭合的是发夹环、堆积、内环/凸环还是多分支环，分别查对应的能量表；多分支环通常用线性罚分近似，以保持 $O(n^3)$ 复杂度（内环需限制长度，例如不超过 30 nt）。代表实现有 Zuker 的 mfold/UNAFold、ViennaRNA 的 RNAfold、RNAstructure。

### 结构集合与配分函数：McCaskill 算法

最小自由能结构（MFE structure）仍只是能量曲面上的单一最低点。实际溶液中 RNA 在许多结构间按 Boltzmann 分布涨落，而能量参数本身有误差，排名第二的结构可能只比 MFE 高零点几 kcal/mol。因此更全面的描述是**配分函数（partition function）**

$$
Z=\sum_s e^{-\Delta G(s)/(RT)}
$$

该式适用于 $\Delta G$ 以每摩尔能量（如 kcal/mol）表示的情形，$R$ 是气体常数，$T$ 是绝对温度；若使用单分子能量，分母应为 $k_BT$。它汇总所有允许结构 $s$，结构 $s$ 的概率为 $e^{-\Delta G(s)/(RT)}/Z$。由此可计算碱基配对概率（base-pair probability，某对 $(i,j)$ 在所有结构中出现的总概率）和集合的不确定性（ensemble uncertainty）。

两种算法的分工是：Zuker/Turner DP 寻找最小自由能结构；<strong>McCaskill 配分函数算法（1990）</strong>则对全部允许结构求和。McCaskill 保留与 Zuker 相同的结构分解，只是把“取最小值”替换为“Boltzmann 权重求和”（min-plus 半环换成 sum-product），类似 HMM 中 Viterbi 与前向算法的关系。可靠的报告应同时给出 MFE、集合多样性（ensemble diversity）与高概率配对，而不是只给一个点括号字符串。

### 实现：带配对能量的教学版折叠

下面是从 Nussinov 过渡到能量最小化的教学版本。它区分 GC、AU 和 GU 的配对能量，但没有 stacking 与环罚分，因此不是完整的 Turner/Zuker 实现。与 `nussinov` 相比，变化只有两处：`max` 换成 `min`，`+1` 换成查表得到的负能量。

```python
PAIR_ENERGY = {
    ("G", "C"): -3.0, ("C", "G"): -3.0,
    ("A", "U"): -2.0, ("U", "A"): -2.0,
    ("G", "U"): -1.0, ("U", "G"): -1.0,
}

def weighted_rna_fold(seq, min_loop=3):
    """最小化简化 pair energy，返回教学 MFE 与 dot-bracket。"""
    seq = seq.upper().replace("T", "U")
    if set(seq) - set("ACGU"):
        raise ValueError("sequence must contain only A/C/G/U/T")
    if min_loop < 0:
        raise ValueError("min_loop must be nonnegative")
    n = len(seq)
    energy = [[0.0] * n for _ in range(n)]
    back = [[None] * n for _ in range(n)]

    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            choices = [
                (energy[i + 1][j], ("skip_i",)),
                (energy[i][j - 1], ("skip_j",)),
            ]
            pair_energy = PAIR_ENERGY.get((seq[i], seq[j]))
            if pair_energy is not None and j - i - 1 >= min_loop:
                inner = energy[i + 1][j - 1] if i + 1 <= j - 1 else 0
                choices.append((inner + pair_energy, ("pair",)))
            for k in range(i, j):
                choices.append((
                    energy[i][k] + energy[k + 1][j],
                    ("split", k),
                ))
            energy[i][j], back[i][j] = min(
                choices, key=lambda item: item[0]
            )

    pairs = []
    stack = [(0, n - 1)] if n else []
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        action = back[i][j]
        if action[0] == "skip_i":
            stack.append((i + 1, j))
        elif action[0] == "skip_j":
            stack.append((i, j - 1))
        elif action[0] == "pair":
            pairs.append((i, j))
            stack.append((i + 1, j - 1))
        else:
            k = action[1]
            stack.extend(((i, k), (k + 1, j)))

    mfe = energy[0][n - 1] if n else 0.0
    return mfe, pairs_to_dot_bracket(n, sorted(pairs))
```

对例 5 的 `GGGAAAUCC`，`weighted_rna_fold` 返回 `(-7.0, '(((...)))')`：两个 G–C 各 $-3.0$，一个 G–U $-1.0$。条件 `j - i - 1 >= min_loop` 与 `nussinov` 中的 `j - i > min_loop` 等价。

这段代码只说明目标函数和 DP 状态如何改变。由于没有堆积项，它仍会偏好许多孤立配对，与真实 RNA 不符。真实折叠需要经实验标定的参数集、温度、盐条件和针对各类环的递推（loop-specific recurrence）；生产分析应调用 ViennaRNA 等经过验证的软件。

## 协变：用演化证据确认结构

### 补偿突变为什么是证据

单序列折叠的准确率有限（长 RNA 的 MFE 预测中常有相当比例的配对是错的）。比较序列分析提供了另一种更强的证据：若结构中的一对碱基在演化中发生**补偿突变（compensatory mutation）**，例如某些物种中 G–C 变为 A–U，两个位置同时改变但配对能力不变，这说明自然选择在维持这个配对约束。如果只有一侧突变而破坏了配对，该突变通常会被淘汰。因此，在比对中“一起变、而且变完仍能配对”的两列，强烈提示它们在结构中配对。rRNA 与 tRNA 的经典二级结构很大程度上就是用这种比较方法确定的。

### 协方差模型

<strong>协方差模型（covariance model, CM）</strong>把 profile HMM 的线性依赖扩展到成对位置：profile HMM 中每个状态只发射一个残基，CM 中配对状态一次发射一对碱基，用 $4\times4$ 的发射概率刻画“G–C、A–U 常见、G–A 罕见”之类的协变。形式上它是一种随机上下文无关文法（stochastic context-free grammar, SCFG），可用于结构化 RNA 家族的搜索（如 Infernal 软件与 Rfam 数据库），但计算更昂贵：模型长度与序列长度同为 $n$ 量级时，CM 的 CYK 比对约需 $O(n^3)$ 时间，而 profile HMM 的 Viterbi 只需 $O(n^2)$。

### 互信息的陷阱

衡量两列是否协变的常用统计量是互信息（mutual information）：

$$
M_{ij}=\sum_{x,y}f_{ij}(x,y)\log\frac{f_{ij}(x,y)}{f_i(x)f_j(y)}
$$

其中 $f_i(x)$ 是第 $i$ 列中碱基 $x$ 的频率，$f_{ij}(x,y)$ 是两列联合频率。但单纯的互信息会把<strong>系统发育相关（phylogenetic correlation）</strong>和**间接相关**也算入协变：若比对中一半序列来自某一支，这一支在两列上恰好都是某种组合，两列会显得相关，原因却只是这些序列共享祖先；若 $i$ 与 $j$ 配对、$j$ 与 $k$ 在三维上接触，$i$ 与 $k$ 也可能显示虚假的相关。可靠的协变证据需要足够多样的同源序列、正确的比对，并与共同祖先背景做比较（例如在系统发育树上统计独立发生的补偿替换次数，或用 R-scape 等方法以系统发育零模型评估显著性）。

### 伪结：区间 DP 的边界

伪结使配对弧发生交叉（例如 `((..[[..))..]]`），打破了“区间可以按某个配对独立分解”的假设——本章的 Nussinov、Zuker、McCaskill 以及 CM 都依赖这一假设，相当于上下文无关文法无法描述交叉依赖。允许一般伪结的最小自由能问题是 NP-hard 的，因此伪结通常被排除，或用只覆盖特定伪结类型的高复杂度 DP（如 $O(n^6)$ 的 Rivas–Eddy 算法）及启发式等近似方法处理。

> [!NOTE]+ 小结
>
> - 无伪结的二级结构可以按区间递归分解，因此可用 $O(n^3)$ 时间、$O(n^2)$ 空间的区间 DP 求最优解，回溯给出点括号结构。
> - Nussinov 最大化配对数，仅用于展示 DP 框架；Zuker/Turner 用最近邻自由能参数和 $W/V$ 两类状态最小化 $\Delta G$。
> - 配分函数（McCaskill）对全部结构求和，给出配对概率；报告应同时给出 MFE、集合多样性与高概率配对。
> - 补偿突变是配对约束的演化证据；协方差模型把 profile HMM 推广到成对发射；互信息需排除系统发育与间接相关。
> - 伪结使配对交叉，超出普通区间 DP 的能力范围。

<br>

# 蛋白质结构：层次、表示与实验来源

序列比对和 HMM 处理的是氨基酸字符串，但蛋白质执行功能靠的是三维形状：酶的活性位点需要几个在序列上相距很远的残基在空间中聚到一起，抗体靠表面的凹凸识别抗原。相似的结构常常在序列相似性已经检测不出时仍然保留，因此结构是比序列更保守、更直接联系功能的信息层。后续关于结构叠合、结构模式、结构预测和分子力学的章节都以本章定义的对象和表示为基础。

蛋白质结构是原子在三维空间中的组织方式。氨基酸序列只给出共价连接顺序；结构在此之上加入主链扭转角（backbone torsion）、侧链方向、残基间接触、结构域（domain）排布和多聚体界面。需要牢记两点限制：同一序列可因配体、修饰或环境处于不同构象；一个坐标文件只是某种实验条件下拟合出的结构模型，不是蛋白质的全部状态。

<strong>计算输入与表示。</strong>计算上的输入通常来自 PDB/mmCIF 文件，包含原子坐标、链（chain）、残基编号、占有率（occupancy）、实验方法和缺失信息。常见的派生表示包括笛卡尔坐标（Cartesian coordinate）、距离矩阵（distance matrix）、接触图（contact map）、扭转角（torsion angle）和二级结构标签（secondary-structure label）。这些表示分别适合叠合、检索、统计与预测，但保存的信息不同，转换时往往有损。

<strong>与后续章节的关系。</strong>本章先说明结构对象本身以及实验坐标的可信边界；之后，结构叠合比较两个构象，结构模式寻找局部的功能几何，结构预测从序列估计坐标，分子力学在给定势能模型下研究坐标如何变化。

## 结构的层次与主链几何

### 四个层次

蛋白质结构习惯上分为四个层次，每一层建立在前一层之上：

- **一级结构（primary structure）**：氨基酸序列，即共价连接的顺序。
- **二级结构（secondary structure）**：$\alpha$ 螺旋（$\alpha$-helix）、$\beta$ 折叠（$\beta$-sheet）、转角（turn）等由主链氢键稳定的局部构象。
- **三级结构（tertiary structure）**：单条链的整体三维折叠，即二级结构单元如何在空间中堆积。
- **四级结构（quaternary structure）**：多条链组成的复合物，例如血红蛋白的 $\alpha_2\beta_2$ 四聚体。

### 主链原子与二面角

蛋白质主链的重复单元包含 N、$C_\alpha$、羰基 C 和 O 四个原子，侧链连在 $C_\alpha$ 上。主链的构象可以用沿链的一串<strong>二面角（dihedral angle，又称扭转角 torsion angle）</strong>描述：

- $\phi$：绕 N–$C_\alpha$ 键的旋转（由 $C_{i-1}$–N–$C_\alpha$–C 四个原子定义）；
- $\psi$：绕 $C_\alpha$–C 键的旋转（由 N–$C_\alpha$–C–$N_{i+1}$ 定义）；
- $\omega$：绕肽键 C–N 的旋转。肽键因部分双键性质（羰基与 N 孤对电子共振）而近似平面，$\omega$ 多为 trans（约 $180^\circ$），cis 构象少见，主要出现在脯氨酸之前。

因此主链构象主要由每个残基的 $\phi$、$\psi$ 描述，侧链构象还涉及 $\chi_1,\chi_2,\dots$ 二面角。一般地，二面角由四个连续原子定义：前三个原子确定一个平面，后三个原子确定另一个平面，二面角是两个平面的夹角。

<strong>Ramachandran 图（Ramachandran plot）</strong>以 $\phi$ 为横轴、$\psi$ 为纵轴，展示 $\phi,\psi$ 因立体位阻（steric clash）和局部相互作用形成的允许区域：右手 $\alpha$ 螺旋集中在 $\phi\approx-60^\circ,\psi\approx-45^\circ$ 附近，$\beta$ 链集中在左上方的宽区域。甘氨酸没有侧链，允许区域更大；脯氨酸的环把 $\phi$ 锁定在约 $-60^\circ$。实验结构中落在禁阻区的残基（Ramachandran outlier）常提示建模错误。

### 自由度计数与内坐标

在固定键长（bond length）、键角（bond angle），并把 $\omega$ 近似固定为 trans 的理想化主链模型中，长度为 $N$ 的链主要由约 $2N-2$ 个内部 $\phi/\psi$ 扭转自由度描述：每个残基有一对 $\phi,\psi$，共 $2N$ 个，但第一个残基的 $\phi$ 和最后一个残基的 $\psi$ 缺少完整的四原子定义。例如 100 个残基的蛋白约有 198 个主链扭转自由度，而它的主链重原子笛卡尔坐标有 $4\times100\times3=1200$ 个数。这个计数是简化表示，不包括侧链 $\chi$、cis 肽键、键角涨落和闭环约束。

由于键长和键角在真实结构中变化很小，用**内坐标（internal coordinate）**——键长、键角、二面角——表示可显著减少自由度，这也是许多结构预测和采样方法在扭转角空间中搜索的原因。但内坐标有“杠杆效应”：链中间一个二面角的微小变化会让远端原子移动很大距离。最终与实验结构比较时，通常仍要转换回笛卡尔坐标。

### 侧链性质与稳定性的来源

20 种氨基酸的侧链可以按疏水（hydrophobic）、极性（polar）、带电（charged）、芳香（aromatic）等性质理解，但分类并不是绝对的边界：例如酪氨酸既是芳香族又带极性羟基，组氨酸在生理 pH 附近可带电也可不带电。

驱动折叠的主要力量是**疏水效应（hydrophobic effect）**：非极性侧链埋入蛋白核心，可以释放在其表面被迫有序排列的水分子，增加溶剂熵。氢键（hydrogen bond）、盐桥（salt bridge）、范德华作用（van der Waals）、二硫键（disulfide bond）与溶剂共同稳定结构。

需要注意的是，结构稳定性是折叠态集合（folded ensemble）与去折叠态集合（unfolded ensemble）之间的自由能差 $\Delta G_{\text{fold}}$，而不是“内部相互作用越多越稳定”的简单计数。去折叠态中的极性基团同样能与水形成氢键，所以一个埋在内部的氢键带来的净收益可能很小；典型蛋白的折叠自由能只有每摩尔几到十几 kcal，是大量正负贡献之差。

### 二级结构的指定：DSSP

实验坐标里并没有“螺旋”标签。二级结构常由 **DSSP** 程序根据主链氢键几何（hydrogen-bond geometry）赋值：先用静电模型判断哪些 C=O 与 N–H 之间形成氢键，再按氢键模式识别螺旋（如 $i\to i+4$ 的重复氢键为 $\alpha$ 螺旋）和折叠（相邻链之间的氢键梯子）。helix、sheet 等标签是从连续几何离散化的结果，边界残基的归属可能随结构细节或所用算法（DSSP、STRIDE 等）而改变，用作训练或评测标签时要注意这种不确定性。

## 坐标、距离矩阵与接触图

### 为什么需要不变的表示

同一个蛋白结构，整体平移或旋转之后，坐标文件里的每个数字都变了，但分子本身没有任何变化。比较两个结构、或让机器学习模型学习结构时，这种“坐标随参考系变化”的性质很麻烦。一个解决办法是改用对刚体变换不变的表示。

### 距离矩阵

设第 $i$ 个原子（或残基代表原子，如 $C_\alpha$）的坐标为 $\mathbf{x}_i\in\mathbb{R}^3$，共 $n$ 个。距离矩阵 $D\in\mathbb{R}^{n\times n}$ 的元素为

$$
D_{ij}=\|\mathbf{x}_i-\mathbf{x}_j\|_2
$$

由于平移和旋转都保持两点间距离，距离矩阵天然具有刚体变换不变性。它对称、对角线为零。

但距离矩阵并不是在所有意义上都“更好”。完整且精确的欧氏距离矩阵（Euclidean distance matrix）通常可以把坐标确定到平移、旋转和**反射**为止——镜像也保持所有距离，因此距离矩阵不能区分镜像构象。这对蛋白很关键：L-氨基酸构成的右手螺旋与它的镜像（左手螺旋）有完全相同的距离矩阵，但后者在化学上不对。

### 距离信息的三个层级

距离信息常以三个逐步有损的层级出现：

| 表示 | 保存的内容 | 丢失的内容 | 典型用途 |
| :--- | :--- | :--- | :--- |
| 完整距离矩阵 | 所有 $D_{ij}$ | 手性（镜像） | 结构比较、MDS 重建 |
| distogram | 每个 $D_{ij}$ 落在各距离区间的概率分布 | 区间内精确值 | 结构预测网络的输出（如 AlphaFold 的距离直方图） |
| 接触图（contact map） | 只保存 $1[D_{ij}<c]$ | 距离大小、方向和局部几何 | 接触预测、协变分析、结构检索 |

从坐标到这些表示是逐步有损的：接触图把距离二值化（以阈值 $c$ 判断是否接触），会丢失距离大小、方向和局部几何。

### 从距离恢复坐标：经典多维标度

反过来，若已知完整距离，能否重建坐标？<strong>经典多维标度（classical multidimensional scaling, MDS）</strong>给出了闭式解。关键观察是：把坐标中心化（质心放在原点）后，Gram 矩阵 $B=XX^\top$（$X$ 为 $n\times3$ 坐标矩阵）可以完全由距离算出，因为

$$
D_{ij}^2=\|\mathbf{x}_i\|^2+\|\mathbf{x}_j\|^2-2\mathbf{x}_i^\top\mathbf{x}_j
$$

左右同时做“行减行均值、列减列均值”就能消去前两项。具体地，给定逐元素平方的距离矩阵 $D^{\circ2}$，先做双中心化（double centering）：

$$
B=-\frac12JD^{\circ2}J,\qquad
J=I-\frac1n\mathbf1\mathbf1^\top
$$

其中 $J$ 是中心化矩阵，$\mathbf1$ 是全 1 向量。若 $B$ 是半正定矩阵，对它做特征分解 $B=U\Lambda U^\top$，取前三个正特征值和对应特征向量，$X=U_3\Lambda_3^{1/2}$ 就恢复出一组三维坐标。

实际数据中，负特征值表示距离带噪声或并非严格欧氏的；截断到三维还会产生近似误差。由于 $B$ 只确定到正交变换，恢复出的坐标与原坐标相差一个旋转或反射。

### 实现：距离、接触图与 MDS

下面三个函数分别实现距离矩阵、接触图和 MDS 重建；其中 `pairwise_distances` 会在后文“三维结构模式”一章中直接复用。

```python
import numpy as np

def pairwise_distances(coordinates):
    """从 (n, d) 坐标计算完整 Euclidean distance matrix。"""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be a finite (n, d) array")
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(difference, axis=-1)


def contact_map(coordinates, cutoff=8.0, min_separation=3):
    """按距离阈值构建 residue contact map，并排除近邻序列位置。"""
    if cutoff <= 0 or min_separation < 0:
        raise ValueError("cutoff must be positive")
    distances = pairwise_distances(coordinates)
    contacts = distances < cutoff
    index = np.arange(len(distances))
    local = np.abs(index[:, None] - index[None, :]) < min_separation
    contacts[local] = False
    return contacts


def coordinates_from_distances(distances, dimensions=3):
    """以 classical MDS 从完整距离矩阵恢复中心化坐标。"""
    distances = np.asarray(distances, dtype=float)
    if (
        distances.ndim != 2
        or distances.shape[0] != distances.shape[1]
        or np.any(distances < 0)
        or not np.allclose(distances, distances.T)
    ):
        raise ValueError("distances must be a symmetric nonnegative matrix")
    n = len(distances)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances ** 2) @ centering
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    order = np.argsort(eigenvalue)[::-1]
    positive = order[eigenvalue[order] > 1e-12][:dimensions]
    return eigenvector[:, positive] * np.sqrt(eigenvalue[positive])
```

几点说明：

- `pairwise_distances` 用广播构造 $(n,n,d)$ 的差向量张量再求范数，内存为 $O(n^2d)$，对几千个原子没问题，更大规模应分块或用 `scipy.spatial.distance.cdist`。
- `contact_map` 默认以 $8$ Å 为阈值，并把序列上相距小于 `min_separation` 的残基对（包括对角线）置为非接触，因为相邻残基由共价键连接，必然很近，不携带折叠信息。若传入 `min_separation=0`，对角线会因距离为 0 而被标为接触。
- `coordinates_from_distances` 中，`eigh` 返回升序特征值，代码按降序重排后只保留大于 $10^{-12}$ 的正特征值，自动丢弃噪声造成的负特征值。

`coordinates_from_distances` 返回坐标的方向、符号与原坐标不必相同，需要用 Kabsch 算法（见结构叠合的相关章节）对齐后才可比较；而镜像仍无法由距离矩阵排除，需要额外的手性信息（例如检查螺旋的旋向或 $C_\alpha$ 处的手性）。另外，接触阈值依赖原子定义：$C_\alpha$ 距离、$C_\beta$ 距离与最小重原子距离（minimum heavy-atom distance）给出的接触图不同，不能混用。

> [!EXAMPLE]+ 例6 距离矩阵往返
>
> 取四个点 $(0,0,0)$、$(1.5,0,0)$、$(1.5,1.5,0)$、$(0,1.5,1.0)$，`pairwise_distances` 给出（保留三位小数）
>
> $$
> D=\begin{pmatrix}
> 0 & 1.5 & 2.121 & 1.803\\
> 1.5 & 0 & 1.5 & 2.345\\
> 2.121 & 1.5 & 0 & 1.803\\
> 1.803 & 2.345 & 1.803 & 0
> \end{pmatrix}
> $$
>
> 把 $D$ 交给 `coordinates_from_distances`，得到的新坐标与原坐标数值不同（已中心化且旋转过），但再计算一次 `pairwise_distances` 得到的矩阵与 $D$ 在 `np.allclose` 意义下完全一致。若把所有点的 $z$ 坐标取反（镜像），$D$ 也不会有任何变化。

## 结构数据的实验来源与质量评估

### 为什么要关心坐标从哪里来

前面把坐标当作已知的“真相”，但 PDB 中的每个坐标都是根据某种实验数据拟合出来的模型，其可靠程度随实验方法、分辨率和局部区域而大不相同。直接拿一个结构做分析之前，应先确认它能否回答手头的问题。

### 使用 PDB 条目前的检查清单

PDB 结构需检查：

- 来源是 X 射线晶体学（X-ray）、核磁共振（NMR）、冷冻电镜（cryo-EM）还是计算预测；
- 分辨率（resolution）、R-free 或局部分辨率（local resolution）；
- 缺失残基（missing residue）、交替构象（alternate location, altloc）、占有率（occupancy）；
- 生物学组装体（biological assembly）与不对称单元（asymmetric unit）的区别：不对称单元是晶体学的最小重复单元，可能只含半个二聚体，也可能含两个不相关的拷贝；
- 配体（ligand）、金属、水分子和突变（例如为结晶而引入的突变）。

### 三种实验方法各自给出什么

- <strong>X 射线晶体学（X-ray crystallography）</strong>测量晶体的衍射强度，但丢失了相位，需经相位求解（phasing）、电子密度重建和模型精修（refinement）才得到原子模型。分辨率数值越小，通常表示可分辨的细节越高：大致而言，约 1.2 Å 或更好时可分辨单个原子，数值超过约 3 Å 时侧链位置就不太可靠。R-free 用未参与精修的一小部分衍射数据检验模型，防止过拟合。
- <strong>冷冻电镜（cryo-EM）</strong>从大量单颗粒图像重建出集合平均的库仑势图（ensemble-averaged Coulomb-potential map）。样品中的构象异质性会被平均进图里，局部分辨率在同一个结构内可能显著变化（核心清晰、柔性边缘模糊）。
- <strong>核磁共振（NMR）</strong>条目常提交一组满足实验约束（restraint，如 NOE 距离、化学位移推出的二面角）的构象（conformer）。这些构象说明约束能被多种结构满足，但不能直接当作带真实群体权重的平衡集合（equilibrium ensemble）。

<strong>B 因子（B-factor，温度因子）</strong>描述原子位置的弥散程度，可反映热运动位移、静态无序、建模不确定性和精修模型的因素，不能直接等同于动态性（dynamics）。

### 模型不等于数据：验证指标

坐标文件是拟合出来的模型（fitted model），不是原始实验数据。评价时还应查看几何离群值（geometry outlier，异常键长键角）、clashscore（每千个原子中严重原子重叠的数量）、Ramachandran 离群值（Ramachandran outlier）、配体密度拟合（ligand density fit），以及 wwPDB 提供的验证报告（validation report）。

更一般地，结构质量必须区分三个概念：<strong>精度（precision）</strong>是模型被数据约束得有多紧，<strong>准确度（accuracy）</strong>是模型离真实结构有多近，<strong>构象异质性（conformational heterogeneity）</strong>是分子本身就存在的多种状态。不同指标约束的是不同的误差来源：X 射线的占有率、B 因子和 R-free，冷冻电镜的局部分辨率与图–模型拟合（map–model fit），NMR 的约束违反（restraint violation）各自针对不同问题，不能互相替代。PDB 的生物学组装体也可能来自作者注释或计算推断（如 PISA 界面分析），不必然等于溶液中的真实寡聚状态。

> [!NOTE]+ 小结
>
> - 蛋白结构分为一级到四级；主链构象主要由 $\phi,\psi$ 描述，$\omega$ 近似平面 trans，理想化主链约有 $2N-2$ 个扭转自由度。
> - 稳定性是折叠态与去折叠态集合的自由能差，疏水效应是主要驱动力；DSSP 标签是连续几何的离散化。
> - 距离矩阵对刚体变换不变，但无法区分镜像；distogram、接触图逐步丢失信息；经典 MDS 可由完整距离恢复坐标（差一个旋转/反射）。
> - `pairwise_distances`、`contact_map`、`coordinates_from_distances` 分别给出距离矩阵、接触图与 MDS 重建；接触阈值必须说明所用原子。
> - PDB 坐标是拟合模型：使用前检查实验方法、分辨率/R-free、缺失与 altloc、组装体，并区分精度、准确度与构象异质性。

<br>

<br>

# 结构叠合与结构比对（Structural Superposition and Alignment）

拿到两个蛋白质的三维坐标文件后，最自然的问题是“它们长得像不像”。但两个 PDB 文件的坐标系是各自实验或建模时随意确定的：同一个蛋白放在晶胞的不同位置、朝向不同方向，坐标数值会完全不同。直接逐点比较坐标没有意义，必须先把两者“摆到一起”。结构叠合（structural superposition）就是把两个三维对象放到同一坐标系中再比较：在对应关系（correspondence）已知时，寻找一个旋转 $R$ 和平移 $t$，使对应点之间距离的平方和最小。平移和旋转只反映观察坐标系的差异，不应计入蛋白内部的结构差异，所以要先把它们“优化掉”，剩下的偏差才是真正的结构差异。

这个问题有两个难度层级。若已经知道“第 $i$ 个点对应第 $i$ 个点”（例如同一蛋白的两个构象、或序列比对已给出残基对应），问题就是一个刚体最小二乘问题，Kabsch 算法给出闭式解。若对应关系也未知（例如两个序列相似度很低、长度不同的蛋白），结构比对（structural alignment）还要同时决定哪些残基配对；STRUCTAL、DALI、TM-align 等方法就在“几何对应”和“刚体叠合”之间搜索或迭代。因此 Kabsch 只是结构比对中的一个子问题，不是完整的未知对应结构搜索算法。

本章使用蛋白质结构一章定义的坐标与距离表示，并把序列比对中的“对应关系”推广到三维：序列比对在一维字符串上配对字符，结构比对在三维空间中配对残基。之后的结构模式搜索、模板建模（homology modeling）以及分子动力学（MD）轨迹对齐都会用到这里的叠合与残基映射（residue mapping）。

## 问题的输入、输出与评价方式

**输入**是两组 $C_\alpha$ 坐标或原子坐标，以及残基对应关系（residue correspondence）。**输出**是最优的 $R,t$、叠合后的坐标，以及衡量剩余差异的均方根偏差（root-mean-square deviation，RMSD）：

$$
\mathrm{RMSD}=\sqrt{\frac1n\sum_{i=1}^{n}\left\|\mathbf p_i R+t-\mathbf q_i\right\|^2}
$$

其中 $n$ 是参与叠合的对应点数，单位是 Å。

RMSD 本身很容易被误读，必须与以下三项一起解释：

- **比对长度（aligned length）**：有多少对残基参与了计算；
- **覆盖度（coverage）**：参与叠合的残基占整条链的比例；
- **原子选择（atom selection）**：用的是 $C_\alpha$、主链原子还是全部重原子。

一个极端例子：只取三个局部点做叠合，几乎总能得到很低的 RMSD（三个点只要两两距离相近就能叠得很好），却完全不能说明两条完整蛋白相似。所以“RMSD = 0.8 Å”这种说法，如果不附带“在多少残基上、覆盖多少比例、用哪些原子”，几乎没有信息量。

<br>

## Kabsch 算法：已知对应时的最优刚体叠合

### 问题的矩阵形式

给定已知对应的两组点 $P,Q\in\mathbb{R}^{n\times3}$（$n$ 个点，每点 3 个坐标），本文始终采用“每一行是一个点”的行向量约定（row-vector convention），即变换写作 $\mathbf p R+t$。目标是

$$
\min_{R,t}\frac1n\left\|PR+\mathbf1t-Q\right\|_F^2,
\qquad R^\top R=I,\quad\det R=1
$$

其中 $\|\cdot\|_F$ 是 Frobenius 范数（所有元素平方和再开方），$\mathbf 1\in\mathbb R^{n\times1}$ 是全 1 列向量，$t\in\mathbb R^{1\times3}$ 是行向量平移，$\mathbf1t$ 把同一个平移加到每一行上。约束 $R^\top R=I$ 要求 $R$ 是正交矩阵（保持长度和角度），$\det R=1$ 进一步排除镜像，使 $R$ 是真正的旋转（proper rotation）。

### 第一步：平移可以先消掉

设两组点的质心为 $\bar p,\bar q$（各列均值，$1\times3$ 行向量）。对任意固定的 $R$，目标对 $t$ 是一个普通的二次函数，求导置零可得最优平移使叠合后两组点的质心重合：$\bar pR+t=\bar q$。因此可以先把两组点各自中心化，记

$$
X=P-\mathbf1\bar p,\qquad Y=Q-\mathbf1\bar q
$$

（也常简写为 $X=P-\bar p$、$Y=Q-\bar q$，即每行减去质心），问题化为只关于 $R$ 的 $\min_R\|XR-Y\|_F^2$。

### 第二步：把最小化距离化为最大化迹

展开 Frobenius 范数（$\|A\|_F^2=\operatorname{tr}(A^\top A)$）：

$$
\|XR-Y\|_F^2
=\|X\|_F^2+\|Y\|_F^2-2\operatorname{tr}(R^\top H)
$$

推导要点：$\|XR\|_F^2=\operatorname{tr}(R^\top X^\top XR)=\operatorname{tr}(X^\top X RR^\top)=\|X\|_F^2$，因为 $R$ 正交，旋转不改变长度；交叉项为 $-2\operatorname{tr}(R^\top X^\top Y)$。前两项与 $R$ 无关，所以最小化 RMSD 等价于最大化 $\operatorname{tr}(R^\top H)$，其中

$$
H=X^\top Y=U\Sigma V^\top
$$

$H$ 是 $3\times3$ 的互协方差矩阵（cross-covariance matrix），右边是它的奇异值分解（SVD）：$U,V$ 为正交矩阵，$\Sigma=\operatorname{diag}(\sigma_1,\sigma_2,\sigma_3)$，$\sigma_1\ge\sigma_2\ge\sigma_3\ge0$。

### 第三步：用 SVD 求最大值

代入 SVD，$\operatorname{tr}(R^\top U\Sigma V^\top)=\operatorname{tr}(\Sigma\,V^\top R^\top U)$。记 $M=V^\top R^\top U$，它是正交矩阵，所有元素绝对值不超过 1，于是 $\operatorname{tr}(\Sigma M)=\sum_k\sigma_kM_{kk}\le\sum_k\sigma_k$，等号在 $M=I$ 即 $R=UV^\top$ 时取到。

但 $UV^\top$ 只保证正交，行列式可能是 $-1$（镜像）。若要求 $\det R=1$，在约束下最优的做法是只让最小奇异值对应的那一项取 $-1$，牺牲最少的迹。为此令

$$
D=\operatorname{diag}\left(1,1,\det(UV^\top)\right)
$$

最优旋转和平移为

$$
R=UDV^\top,\qquad t=\bar q-\bar pR
$$

当 $\det(UV^\top)=1$ 时 $D=I$，解就是 $UV^\top$；当 $\det(UV^\top)=-1$ 时，$D$ 把第三个奇异向量方向翻转，得到最接近的合法旋转。该解最大化 $\operatorname{tr}(R^\top H)$，从而最小化 RMSD。

### 行向量与列向量约定

若使用列向量（column vector）并写 $\mathbf q=R_{\mathrm{col}}\mathbf p+t$，则 $R_{\mathrm{col}}=R^\top=VDU^\top$。若省略行列式修正（determinant correction），SVD 可能返回镜像；镜像能进一步降低 RMSD，却不是合法的旋转（rotation）——蛋白质有手性，镜像会把 L-氨基酸变成 D-氨基酸、右手 α 螺旋变成左手螺旋，得到的已不是同一个分子。

### 实现

```python
import numpy as np

def kabsch(P, Q):
    """按 row-vector 约定将 P 刚体叠合到 Q。"""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P and Q must have the same shape (n, 3)")
    if len(P) == 0 or not np.isfinite(P).all() or not np.isfinite(Q).all():
        raise ValueError("coordinates must be nonempty and finite")

    # 去除平移，只在中心化点集之间求最优旋转。
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    X, Y = P - pc, Q - qc
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt

    # det(R)<0 表示包含镜像；翻转一个奇异向量以限制为 proper rotation。
    if np.linalg.det(R) < 0:
        correction = np.eye(3)
        correction[-1, -1] = -1
        R = U @ correction @ Vt

    # row-vector convention 下 fitted = P @ R + t。
    t = qc - pc @ R
    fitted = P @ R + t
    rmsd = np.sqrt(np.mean(np.sum((fitted - Q) ** 2, axis=1)))
    return R, t, rmsd
```

代码与公式一一对应：`X.T @ Y` 就是 $H$；`np.linalg.svd` 返回的 `Vt` 已经是 $V^\top$，所以 `U @ Vt` 即 $UV^\top$；`correction` 就是 $D$（NumPy 的奇异值按降序排列，最后一个对应最小奇异值，所以翻转 `[-1, -1]`）。

### 复杂度

固定三维时，中心化与构建协方差为 $O(n)$，$3\times3$ SVD 为常数时间。一般 $d$ 维时为 $O(nd^2+d^3)$：构建 $d\times d$ 协方差需 $O(nd^2)$，SVD 需 $O(d^3)$。保存坐标需 $O(nd)$ 空间；若以流式方式累计质心与协方差（只需维护和式），可以降低额外空间。

### 常见错误与测试方法

代码采用行向量约定：`P @ R + t`。许多教材使用列向量并写成 $R\mathbf p+t$，此时矩阵公式要整体转置。实现中最常见的错误正是混用这两套约定：$R$ 与 $R^\top$ 搞反时，对很多真实样本 RMSD 仍可能“看起来不太大”，错误不易察觉。可靠的测试应主动构造已知的旋转与平移（例如随机生成 $R_0,t_0$，令 $Q=PR_0+t_0$），确认函数能恢复 $R_0,t_0$ 且 RMSD 近似为 0，而不只检查某个真实样本的 RMSD。

### 与 Arun 等人推导的关系

Arun、Huang 与 Blostein 1987 年的推导同样通过 $3\times3$ 互协方差矩阵的 SVD 求最小二乘意义下的旋转与平移。他们的问题写成带噪声的模型 $\mathbf p_i'=R\mathbf p_i+t+\mathbf n_i$（$\mathbf n_i$ 为噪声），采用列向量约定；与上式比较时需要转置旋转矩阵。Kabsch 与 Arun 算法在无权、已知对应的刚体最小二乘问题上给出同类闭式解，只是文献中矩阵的排列方式和行列式修正的写法可能不同（例如有的文献在 $\det<0$ 时翻转 $V$ 的最后一列，与翻转 $D$ 的最后一个元素等价）。这一问题在统计学中也称为正交 Procrustes 问题（orthogonal Procrustes problem）。

### 加权、离群点与退化情形

- **加权**：若每个点有置信度 $w_i\ge0$（例如按 B-factor 或预测置信度给权重），可改用加权质心 $\bar p=\sum_iw_i\mathbf p_i/\sum_iw_i$ 和加权协方差 $H=\sum_iw_i\mathbf x_i^\top\mathbf y_i$（$\mathbf x_i=\mathbf p_i-\bar p$、$\mathbf y_i=\mathbf q_i-\bar q$ 为相对加权质心的行向量，$\mathbf x_i^\top\mathbf y_i$ 是 $3\times3$ 外积），其余步骤不变。
- **离群点（outlier）**：普通最小二乘对大偏差平方加权，少量大偏差（例如一段柔性 loop 或发生了铰链运动的结构域）会主导结果。可以迭代剔除偏差最大的点，或采用稳健损失（robust loss）；但应避免只保留“看起来最像”的核心区（core）之后，仍然报告关于全结构的结论。
- **退化情形**：当点集共线、共面或包含重复点时，最优 RMSD 仍可定义，但旋转可能不唯一（例如共线点绕该直线任意转动都不改变 RMSD，此时 $H$ 有零奇异值）。生产代码还应检查输入形状、有限值、点数、权重非负性，以及奇异值是否退化。

<br>

## 对应关系未知时的结构比对

### 为什么难：两个互相依赖的子问题

序列比对已知时，可以按对应残基直接叠合；但结构比对通常连对应关系也未知——两个远缘蛋白可能长度不同、有插入缺失、序列几乎没有可识别的相似性。典型方法要么在“建立对应”和“刚体叠合”之间迭代，要么比较不依赖坐标系的局部距离模式。例如 DALI 比较残基–残基距离矩阵（residue–residue distance matrix），TM-align 则优化对蛋白长度更稳健的 TM-score。

结构比对同时优化两个相互依赖的问题，像“先有鸡还是先有蛋”：

- 给定对应关系时，可用 Kabsch 闭式求出最小 RMSD 的叠合；
- 给定空间取向时，可按叠合后残基间的距离相似度寻找对应关系。

目标函数的选取也很关键。若只追求最小 RMSD，算法会倾向保留很短的局部核心，因为点越少越容易叠好；若只追求比对长度，又会纳入大量低质量对应。因此必须联合报告 RMSD 与对应关系/覆盖度，或者使用同时奖励长度、惩罚距离和 gap 的目标函数（TM-score 就是这类目标的代表）。

### STRUCTAL：交替执行动态规划与刚体叠合

STRUCTAL（出自 Stanford 的 Levitt 组，由 Subbiah、Laurents 与 Levitt 提出，Gerstein 与 Levitt 进一步发展）交替执行两个步骤：

1. 在当前相对取向下，根据两条链 $C_\alpha$ 之间的距离构造相似度矩阵（similarity matrix），距离越近得分越高，并用保持序列顺序的动态规划（dynamic programming）求对应关系；
2. 对当前对应做最小二乘刚体叠合，更新取向；

重复直至对应不再改变。每个子问题都有明确的最优解，但这不代表交替过程一定找到全局最优：DP 最大化的是相似度得分，Kabsch 最小化的是对应点 RMSD，两步优化的并不是同一个目标，结果会受初始取向和打分设计影响。这一方法把 Needleman–Wunsch 式动态规划与 Kabsch/Procrustes 叠合连接在同一个迭代框架中——双序列比对一章里的动态规划矩阵，在这里只是把“替换矩阵得分”换成了“空间距离得分”。

下面的代码演示这种交替优化。第一个函数在固定取向下做保持顺序的全局动态规划，相似度为 `offset - 距离`（距离小于 `offset` Å 时得正分），gap 罚分为常数；第二个函数在 DP 与 Kabsch 之间交替。

```python
import numpy as np

# 依赖本章上一节已经定义的 kabsch(P, Q)。
def ordered_structure_correspondence(P_fitted, Q, offset=3.0, gap=-2.0):
    """在固定空间取向下，用顺序约束 DP 建立 C-alpha correspondence。"""
    n, m = len(P_fitted), len(Q)
    dp = np.zeros((n + 1, m + 1))
    back = np.empty((n + 1, m + 1), dtype=object)
    for i in range(1, n + 1):
        dp[i, 0], back[i, 0] = dp[i - 1, 0] + gap, "U"
    for j in range(1, m + 1):
        dp[0, j], back[0, j] = dp[0, j - 1] + gap, "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 距离越小，当前残基对应的奖励越高。
            spatial_score = offset - np.linalg.norm(
                P_fitted[i - 1] - Q[j - 1]
            )
            choices = [
                (dp[i - 1, j - 1] + spatial_score, "D"),
                (dp[i - 1, j] + gap, "U"),
                (dp[i, j - 1] + gap, "L"),
            ]
            dp[i, j], back[i, j] = max(
                choices, key=lambda item: item[0]
            )

    pairs = []
    i, j = n, m
    while i or j:
        step = back[i, j]
        if step == "D":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif step == "U":
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def iterative_structure_alignment(P, Q, max_iter=50, **dp_parameters):
    """交替执行 ordered DP 与 Kabsch，返回变换、对应和对应点 RMSD。"""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1:] != (3,) or Q.shape[1:] != (3,):
        raise ValueError("P and Q must both have shape (n, 3)")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    # 仅以质心平移初始化；真实工具会尝试多个片段或局部 frame。
    R = np.eye(3)
    t = Q.mean(axis=0) - P.mean(axis=0)
    previous_pairs = None
    for _ in range(max_iter):
        fitted = P @ R + t
        pairs = ordered_structure_correspondence(
            fitted, Q, **dp_parameters
        )
        if len(pairs) < 3:
            raise ValueError("fewer than three residue pairs were aligned")
        if pairs == previous_pairs:
            break
        p_index, q_index = zip(*pairs)
        R, t, rmsd = kabsch(P[list(p_index)], Q[list(q_index)])
        previous_pairs = pairs
    else:
        fitted_pairs = P[list(p_index)] @ R + t
        rmsd = np.sqrt(np.mean(np.sum(
            (fitted_pairs - Q[list(q_index)]) ** 2, axis=1
        )))
    return R, t, pairs, rmsd
```

循环的收敛判据是“本轮对应与上一轮完全相同”：此时再做 Kabsch 也只会得到同样的变换，返回的 `rmsd` 正是这组对应上的叠合误差。若迭代用完 `max_iter` 仍未收敛（`for ... else` 分支），则用最后一次 Kabsch 的变换重新计算对应点 RMSD。至少要求 3 对残基，是因为少于 3 个点无法唯一确定三维旋转。

需要强调，这段代码只展示交替优化的思想，不是 STRUCTAL、DALI 或 TM-align 的复刻。其距离得分、全局末端 gap（global end-gap）和单一起点都很简化；真实结构比对会使用饱和的距离权重（如 $1/(1+(d/d_0)^2)$ 形式）、局部片段初始化、覆盖度归一化（coverage normalization）和多起点搜索。若初始取向很差，第一次 DP 就可能形成错误对应，之后 Kabsch 会在错误的盆地（basin）中收敛，再也跳不出来。

### DALI：比较内部距离矩阵

DALI（Holm 与 Sander）换了一个思路：既然坐标依赖坐标系，就干脆不用坐标，而把每个结构转为分子内距离矩阵（intramolecular distance matrix）$D_{ij}=\|\mathbf r_i-\mathbf r_j\|$，然后在两个距离矩阵中寻找具有相似距离模式（distance pattern）的子矩阵（submatrix）。距离模式对整体刚体变换天然不变，也允许在没有初始序列对应的情况下比较内部几何。原始方法先构造局部相似的片段（例如六残基片段对的接触模式），再用 Monte Carlo 优化把这些片段组合成完整比对。数据库搜索报告的 Z-score 依据背景分数分布衡量显著性，不等同于 RMSD：Z-score 高表示“这个相似度在随机结构对中很罕见”。Holm 与 Sander 1996 年的全对全（all-against-all）分析进一步用 DALI 组织已知结构空间，说明即使序列相似度很低，仍可发现共享的折叠（fold）。

### TM-score 与 TM-align：对长度稳健的相似度

对给定对应关系，TM-score 在所有刚体叠合中取最大值：

$$
\operatorname{TM\text{-}score}
=\max_{\text{superposition}}
\frac{1}{L_{\text{norm}}}
\sum_{i=1}^{L_{\text{ali}}}
\frac{1}{1+[d_i/d_0(L_{\text{norm}})]^2}
$$

其中 $L_{\text{ali}}$ 是比对上的残基对数，$d_i$ 是第 $i$ 对残基叠合后的距离，$L_{\text{norm}}$ 是归一化长度（通常取参考结构长度），$d_0$ 是随长度变化的距离尺度。每一项取值在 $(0,1]$：$d_i=0$ 时贡献 1，$d_i=d_0$ 时贡献 $1/2$，$d_i$ 很大时趋于 0。因此 TM-score 取值在 0 到 1 之间，1 表示完全一致。

常用

$$
d_0(L)=1.24\sqrt[3]{L-15}-1.8\ \mathrm{\mathring A}
$$

这个式子大致是随机相关蛋白中对应残基的平均距离随长度的增长规律，用它归一化后，不同长度蛋白的分数才可比较；短链实现还会设置下限（例如长度很短时把 $d_0$ 固定为 0.5 Å）。

关于 TM-score 和 TM-align 的几个要点：

- TM-align 交替更新对应关系和叠合（并用二级结构、无 gap 片段等多种初始比对作起点），以启发式方式寻找高分解，不保证全局最优。
- 归一化到某一条链时，结果可能非对称：以短链长度归一化与以长链长度归一化得到不同数值，平均长度又是另一个数。报告时要写明按哪条链归一化。
- 远离核心的大距离项贡献会饱和（趋于 0 而不是像平方项那样无限增大），因此 TM-score 比 RMSD 不易被少数离群点支配。

### RMSD 与 TM-score 的解释尺度

RMSD 对局部离群点敏感，并随蛋白长度变化：同样“相似程度”的两对蛋白，长的那对 RMSD 往往更大。对典型的单结构域蛋白，TM-score $>0.5$ 常作为属于相同 SCOP/CATH 折叠的经验信号，随机无关结构的平均水平约为 0.17；但这是统计经验，不是适用于短链、多结构域和重复蛋白的硬定理。

### 结构相似与同源性不是一回事

结构相似并不自动证明序列同源（sequence homology）：物理约束可导致趋同折叠（convergent fold），即无共同祖先的蛋白独立演化出相似的结构。反过来，明确同源的蛋白也可能因结构域运动（domain motion）而产生较高的全局 RMSD，例如开放态与闭合态的酶。此时可以分别报告结构域水平（domain-level）的比对和铰链运动（hinge motion）。

因此，一次结构比较的完整报告应包括：原子选择、残基对应关系、比对覆盖度、归一化长度，以及是否允许分结构域比对（domain-wise alignment）。低 RMSD 可能只来自很短的局部核心，高 TM-score 也不能单独证明演化同源。

> [!NOTE]+ 小结
>
> - 已知对应时，最优刚体叠合由 Kabsch 闭式给出：中心化、对 $H=X^\top Y$ 做 SVD、用 $D$ 修正行列式。
> - 行向量约定 $R=UDV^\top$，列向量约定 $R_{\mathrm{col}}=VDU^\top$；混用两套约定是最常见的实现错误。
> - 对应未知时，结构比对在“找对应”和“做叠合”之间交替（STRUCTAL、TM-align），或比较不依赖坐标系的距离矩阵（DALI）。
> - RMSD 必须与比对长度、覆盖度、原子选择一起报告；TM-score 对长度归一化、对离群点饱和，$>0.5$ 是同折叠的经验信号，随机约 0.17。
> - 结构相似不等于同源，同源也可能因结构域运动而全局 RMSD 较高。

<br>

# 三维结构模式与功能位点识别（Structural Motifs and Functional Sites）

前面的 Motif 一章在一维序列上找保守模式，例如“某位置必须是 C、隔 2–4 个任意残基后再出现 H”。但很多功能位点在一级序列上根本看不出规律。经典例子是丝氨酸蛋白酶的催化三联体（catalytic triad）：在胰凝乳蛋白酶（chymotrypsin）编号中，His57、Asp102、Ser195 在序列上相隔几十到上百个残基，折叠后却在空间上紧挨着构成活性中心；而枯草杆菌蛋白酶（subtilisin）的折叠完全不同，却用另一种序列顺序排出了几乎相同的三维几何。这类位点只有在三维空间里才能被识别。

结构模式（structural pattern）研究的就是“某种局部空间安排是否反复出现”。一维 motif 要求关键残基按序列顺序出现；三维 motif 则要求若干残基或原子满足距离、角度、手性和理化环境约束。因为催化残基可以在一级序列上相距很远却在折叠后靠近，三维搜索能发现线性一致序列（consensus）看不到的功能位点。

这一章承接 Motif 和结构叠合两章：搜索阶段解决“结构中哪里像模板”（需要叠合与几何匹配），统计阶段解决“扫描了许多位置之后，最好的分数是否仍然异常”。还要始终记住解释边界：命中只表示局部结构与已知位点相容，不等于功能已经得到实验确认；只报告最低 RMSD 而没有背景分布，无法区分真正保守的位点和数据库规模带来的偶然近似。

## 问题的输入、输出与方法类别

**输入**是一个几何模板或功能位点模型，以及待扫描的结构或结构库。**输出**是候选的残基分配（assignment，即“模板第 1 个位置对应结构中第几号残基”等）、几何误差和统计显著性。

方法大致分两类：TESS、FFF 更接近显式的几何模板匹配，要求模板中的每个关键原子在结构中找到一一对应；FEATURE（Altman 组的方法）则不做一一对应，而把查询点周围的微环境（microenvironment）转成多维属性向量，再用统计模型判断它像不像某类功能位点。

<br>

## 一维模式与三维模式

### 两种约束的区别

一维序列 motif（1D sequence motif）主要约束残基在线性序列中的顺序，可以连续，也可以含通配符（wildcard）、gap 或可变间隔。三维 motif（3D motif）约束关键原子或残基的空间关系，例如催化三联体的成员可在序列上相距很远。

### 三维模式搜索需要的要素

一个可用的三维模式搜索系统需要：

- 原子或残基的类型标签（例如“此位置必须是 Ser 的 OG 原子”）；
- 点与点之间的距离及容差；
- 可选的角度、手性和溶剂可及性（solvent accessibility）约束；
- 对整体旋转和平移不变（同一个位点放在不同坐标系中应得到相同结果）；
- 对缺失原子和构象噪声鲁棒（晶体结构可能缺少侧链原子，不同结构中位点会有零点几 Å 的偏差）。

### 图匹配视角与计算复杂度

可以把 motif 表示成一个小型的带标记图（labeled graph）：节点是带类型标签的残基或原子，边上标注目标距离；把候选结构也表示成距离图，于是搜索就变成子图匹配（subgraph matching）。一般的子图同构判定是 NP-complete 问题，带标签、距离区间与缺失点的近似匹配还会进一步扩大搜索空间；不过生物 motif 通常很小（三到十几个点），实际可解。实际系统先用距离签名（distance signature）或几何哈希（geometric hashing）快速生成候选，再做精确叠合验证。

### 几何哈希

几何哈希的核心想法是：用模板中的几个点建立一个局部坐标系（frame），把其他点在这个 frame 中的坐标作为“指纹”预先存进哈希表，查询时就不必对所有组合做叠合。具体步骤是：

1. 通常由三个非共线且有确定顺序的点建立局部 frame（例如第一个点作原点、指向第二个点为 $x$ 轴、三点所在平面确定 $xy$ 平面）；需要注意交换基点顺序可能改变 frame 的手性。
2. 其余点转为相对 frame 的坐标（frame-relative coordinate），量化后写入哈希表。
3. 扫描候选结构时，枚举其中兼容的 frame，把每个点的相对坐标拿去查表并投票；票数高的 frame 对应潜在命中。
4. 最后用原子类型、手性、RMSD 与完整几何约束验证。

由于 frame 是由结构自身的点定义的，相对坐标天然对整体旋转和平移不变。

### 课程中的三类早期功能位点方法

课程介绍了三种表示方式不同的早期三维功能位点（3D functional-site）方法：

- **FFF（Fuzzy Functional Form，模糊功能形式）**：从实验知识和已知结构手工或半手工定义关键残基的身份、构象与几何约束，并允许一定容差（“模糊”即指距离允许较宽范围），使分辨率较低的模型（例如预测结构）也能被筛查。
- **TESS**：从已知活性位点（active site）推导三维坐标模板，使用几何哈希在结构库中寻找满足模板的候选。
- **FEATURE**：不要求每个关键原子一一对应，而是把查询点周围的同心壳层（concentric shell）表示为理化与结构属性向量，学习某类微环境相对背景的统计偏好。

FFF/TESS 更接近显式几何模板，FEATURE 更接近局部环境分类器（local-environment classifier）。前者精确但需要知道哪些原子关键，后者不需要逐原子对应，能捕捉“周围富含带负电的氧原子”这类分布式特征。三者都依赖正例的定义和背景库；命中表示局部环境与已知功能位点相容，不等于蛋白已被实验确认具有该功能。

### 用距离模板实现一个三维 motif 匹配

下面以距离模板（distance template）表示三维 motif。模板只指定每个位置允许的残基类型，以及各位置之间的目标距离，因此天然不受整体旋转和平移影响。若功能依赖手性或原子方向，距离矩阵还不够——一个结构和它的镜像有完全相同的距离矩阵——这时还需增加有向体积（signed volume）、角度或局部 frame 约束。

```python
from itertools import product
import numpy as np

# 依赖“坐标与距离”一节已经定义的 pairwise_distances()。
def match_distance_motif(
    coordinates, residue_types, motif_types, target_distances, tolerance=1.0
):
    """枚举满足残基标签和两两距离约束的 3D motif 命中。

    target_distances 是 (k, k) 对称矩阵；返回索引 tuple 与距离 RMSE。
    """
    coordinates = np.asarray(coordinates, dtype=float)
    target_distances = np.asarray(target_distances, dtype=float)
    k = len(motif_types)
    if coordinates.shape != (len(residue_types), 3):
        raise ValueError("coordinates and residue_types do not agree")
    if k < 2 or target_distances.shape != (k, k) or tolerance < 0:
        raise ValueError("target distance matrix has wrong shape")

    # 先按残基标签缩小每个 motif 位置的候选集合。
    candidates = [
        [i for i, residue in enumerate(residue_types) if residue in allowed]
        for allowed in motif_types
    ]
    hits = []
    for assignment in product(*candidates):
        if len(set(assignment)) != k:
            continue  # 一个结构残基不能同时占两个 motif 位置。
        observed = pairwise_distances(coordinates[list(assignment)])
        upper = np.triu_indices(k, 1)
        residual = observed[upper] - target_distances[upper]
        if np.all(np.abs(residual) <= tolerance):
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            hits.append((assignment, rmse))
    return sorted(hits, key=lambda item: item[1])
```

这里 `coordinates` 形状为 $(N,3)$，每行一个残基的代表点；`residue_types` 是长度 $N$ 的单字母残基序列；`motif_types` 长度为 $k$，每一项可以是集合，例如 `{"D", "E"}` 表示该位置允许酸性残基。函数要求每对距离的偏差都不超过 `tolerance`，并用距离偏差的均方根（RMSE）给命中排序。注意这里比较的是距离矩阵，不需要先做 Kabsch 叠合。

枚举的复杂度是各位置候选数的乘积：若每个位置有 30 个候选，五位置模板就要检查 $30^5\approx2.4\times10^7$ 种组合。几何哈希的作用正是用局部 frame 和量化坐标先索引兼容的组合，避免对整库执行完整的笛卡尔积。

### 设计功能 motif 搜索时要明确的选择

功能 motif 搜索应事先区分以下几点，因为它们直接影响灵敏度和假阳性：

- 残基身份（residue identity）是否必须完全相同，还是允许同类替换（如 D/E）；
- 代表点用 $C_\alpha$、侧链质心（side-chain centroid）还是具体的功能原子（functional atom，如 Ser 的 OG）；
- 距离容差是否考虑结构分辨率（低分辨率结构需要更宽的容差）；
- 是否允许某个位置缺失；
- 是否要求相同的手性（chirality）；
- 命中是否位于可接近的口袋（pocket）中，而不是埋在蛋白内部、底物无法到达的位置。

<br>

## 搜索结果的统计显著性

找到命中只是第一步。在一个有十几万个结构的数据库中扫描，总能找到几何上“差不多”的组合——就像在足够长的随机序列中总能找到任意短字符串。扫描许多结构和位置会产生选择偏差（selection bias）：报告的最佳分数是大量尝试中的最大值，本身就偏乐观。结构相似是否显著，取决于背景数据库、蛋白长度、残基组成和搜索空间，不能只报告最低 RMSD。

正确做法是用独立的背景集或诱饵数据库（decoy database）估计假阳性率（false-positive rate）。大量 motif、蛋白和局部 frame 的扫描会带来多重检验（multiple testing）问题，数据库冗余（同一家族有大量近似重复结构）又会让命中看起来比实际更“独立”。因此需要同时报告覆盖度、经验 $p$ 值、错误发现率（FDR）和最佳几何分数。

## 统计特征检测：把局部结构变成可检验的分数

### 共同框架

课程中的统计特征检测（statistical feature detection）不是某一种固定检验，而是许多方法共享的同一个问题：先把序列窗口、三维微环境或网络局部结构映射为一个特征分数（feature score），再问该分数相对背景是否异常。回顾前面各章，可以看到它们都是这个框架的实例：

| 方法 | 被打分的对象 | 背景（null） |
| --- | --- | --- |
| PWM log-odds | 序列窗口 | 背景碱基分布 |
| BLAST E-value | 局部比对分数 | 随机序列比对分数的极值分布 |
| DALI Z-score | 结构比较分数 | 结构比较的背景分数分布 |
| FEATURE | 三维微环境 | 非位点微环境的属性分布 |

### FEATURE 的表示与打分

三维 FEATURE 类表示把查询点周围的空间分成若干同心球壳（shell），在每个壳层中统计原子类型（atom type）、电荷（charge）、疏水性（hydrophobicity）、二级结构（secondary structure）与溶剂可及性（solvent accessibility）等属性。于是一个三维位置就变成了“壳层数 × 属性数”维的向量；例如后续的 FEATURE 应用常用 6 个厚 1.25 Å 的同心壳层（总半径 7.5 Å）、每层 80 种属性，合计 $6\times80=480$ 维。

有了正例（已知功能位点）和反例（非位点背景）的向量集合后，最简单的打分是假设各属性独立、服从高斯分布的对数似然比（log-likelihood ratio）。若 $x_j$ 是第 $j$ 项属性，

$$
\log\Lambda(x)
=\sum_j\log
\frac{\mathcal N(x_j;\mu_{1j},\sigma_{1j}^2)}
     {\mathcal N(x_j;\mu_{0j},\sigma_{0j}^2)}
$$

其中下标 1 表示功能位点，0 表示背景微环境，$\mu,\sigma^2$ 是从训练集估计的均值和方差。每一项衡量“这个属性值在位点中出现的可能性比在背景中大多少倍”，取对数后相加；正值偏向位点，负值偏向背景。这与 PWM log-odds 的结构完全相同，只是把“每个位置的碱基”换成了“每个壳层的属性”。（FEATURE 的实际实现使用把属性值离散化后的朴素贝叶斯式打分，上式是其高斯版本的简化示意。）

使用时有两个常见坑：

- 各属性实际上高度相关（例如带电原子多的壳层，极性原子通常也多），独立模型会把同一证据重复计数，因此分数不能直接解释为后验概率（posterior probability），只能作为排序依据。
- 背景集（null set）必须与正例匹配蛋白类别、溶剂暴露、残基组成和结构质量；否则分类器学到的只是“正例来自哪批数据”，而不是功能位点本身的特征。例如正例都在蛋白表面、背景随机取自蛋白内部，模型只会学会区分表面与内部。

### 经验 p 值与 Z-score

给定观察分数和一组背景分数（来自诱饵、置换或非位点），可以计算右尾经验 $p$ 值和描述性的 Z-score：

```python
import math
import numpy as np

def empirical_feature_significance(observed, null_scores):
    """给出右尾经验 p-value 与仅作描述的 z-score。"""
    null_scores = np.asarray(null_scores, dtype=float)
    if null_scores.ndim != 1 or len(null_scores) < 2:
        raise ValueError("at least two null scores are required")
    if not np.isfinite(null_scores).all() or not math.isfinite(observed):
        raise ValueError("scores must be finite")

    # 加一校正避免有限置换下得到 p=0。
    p_value = (
        1 + np.count_nonzero(null_scores >= observed)
    ) / (len(null_scores) + 1)
    standard_deviation = null_scores.std(ddof=1)
    z_score = (
        (observed - null_scores.mean()) / standard_deviation
        if standard_deviation > 0 else math.nan
    )
    return float(p_value), float(z_score)
```

经验 $p$ 值为

$$
p=\frac{1+\#\{b:\ s_b\ge s_{\mathrm{obs}}\}}{B+1}
$$

其中 $B$ 是背景分数个数。加一校正的含义是把观察值本身也算作一次背景抽样，因此即使所有背景分数都低于观察值，也只得到 $p=1/(B+1)$，而不是不合理的 $p=0$；这同时提示：想报告 $p<10^{-4}$，至少需要约 $10^4$ 个背景样本。

两种量的假设不同。经验 $p$ 值只要求背景抽样过程合理；Z-score 还隐含“均值和方差足以概括背景分布”，即近似正态。若背景明显偏态或存在极值尾部，应直接报告经验分位数，或者为最大扫描分数拟合合适的极值模型（如 BLAST 使用的 Gumbel 型极值分布）。对同一结构扫描许多位置时，背景重复（null replicate）必须重复整个扫描过程并取其中的最大值，不能把单个位置的分数当作“最大值”的背景——否则相当于拿“一次抽签的最好结果”去和“一次抽签的普通结果”比较，显著性会被严重高估。

> [!NOTE]+ 小结
>
> - 三维 motif 约束空间关系，能识别序列上相距很远、折叠后聚集的功能位点（如催化三联体）。
> - 显式模板（FFF、TESS）要求原子一一对应，常用几何哈希加速；FEATURE 用同心壳层属性向量描述微环境，作统计分类。
> - 距离模板对刚体变换不变，但无法区分镜像，需要手性约束时要另加有向体积或局部 frame。
> - 命中的显著性必须相对独立背景评估，并处理多重检验、数据库冗余和“最大分数”的选择偏差。
> - 命中只说明局部结构与已知位点相容，不等于功能已被实验确认。

<br>

# 蛋白质结构预测（Protein Structure Prediction）

实验测定一个蛋白质结构（X 射线晶体学、核磁共振、冷冻电镜）通常要数月到数年，而测序一个基因只需要很低的成本。UniProt 中有上亿条序列，PDB 中实验结构只有约二十万个，二者相差三个数量级。结构预测要回答的就是：只给氨基酸序列，能否推断出蛋白质可能采用的三维构象？Anfinsen 的经典实验表明，在合适条件下序列本身就决定了天然结构，这为“从序列算结构”提供了依据。

预测的输出不一定是完整坐标，也可以是中间表示（intermediate representation），例如二级结构、接触图（contact map）、距离分布图（distogram，残基对距离落在各个距离区间的概率）或扭转角分布（torsion distribution）。需要明确解释边界：预测模型给出的是在训练数据、模板和输入多序列比对（MSA）条件下最受支持的结构；它不是实验观测，也不是完整的热力学系综（ensemble）。

**输入**通常包括目标序列、同源序列构成的 MSA 和可用模板；**输出**包括原子坐标以及 pLDDT、PAE、pTM 等置信度。方法上有几条分支：同源建模（homology modeling）依赖相似的已知结构作模板；穿线（threading）判断序列是否适合某个已知折叠；Rosetta 通过片段组装与能量函数搜索构象；AlphaFold 则直接学习序列、MSA、残基对表示（pair representation）与坐标之间的关系。

本章整合了前面的数据库搜索、多序列比对、结构叠合和距离表示：找模板要用数据库搜索，共进化要用 MSA，评价预测要用叠合与 TM-score。下一章的蛋白质语言模型讨论不显式构建 MSA 时，怎样从大规模序列中学习同类统计规律；而分子力学从给定坐标出发计算力和时间演化，回答的是另一类问题（“结构如何运动”而非“结构是什么”）。

## 经典路线与 CASP 评测

### CASP：用盲测衡量预测能力

结构预测方法很容易在“已知答案”的结构上过拟合，因此领域建立了盲测机制。蛋白质结构预测关键评估（Critical Assessment of protein Structure Prediction，CASP）自 1994 年起大约每两年举办一次：组织者收集即将测定但尚未公开实验结构的目标蛋白（target），参赛者只拿到序列并提交预测，方法在提交截止后才与真实结构比较。盲测把“模型能否在已知结构上拟合”与“能否预测未来的 target”分开。

CASP 是周期性的基准评测（benchmark），不是某个预测算法。评价也从单纯的 RMSD 扩展到 GDT、TM-score、局部质量（如 lDDT）和不同的 target 类别（有模板的易目标、无模板的难目标等）。其中 GDT_TS（global distance test total score）在多个叠合下统计预测结构中与真实结构距离在 1、2、4、8 Å 以内的残基比例，再取四个比例的平均，比 RMSD 更不受少数大偏差区域影响。排名还依赖 target 难度、结构域（domain）划分和所选评价指标。

### 数据规律与物理约束

结构预测同时是统计学习问题与物理约束问题。MSA、模板和训练结构提供数据规律（“自然界中类似序列折叠成什么样”）；几何、堆积（packing）与能量则限制哪些构象在物理上可以实现（不能有原子相撞、键长键角要合理、疏水残基倾向埋在内部）。下面几种经典方法在这两种信息之间各有侧重：

- **二级结构预测**：从局部序列窗口统计方法发展到神经网络，为每个残基输出螺旋（helix）、折叠片（sheet）或无规卷曲（coil）。
- **同源建模**：找模板、对齐、复制骨架、建 loop 与侧链、优化与验证。
- **穿线/折叠识别（threading/fold recognition）**：把序列“穿”到已知折叠上，用环境特异势（environment-specific potential）评价相容性。
- **从头预测（ab initio）**：没有模板时，在巨大的构象空间中结合能量函数与搜索算法寻找低能结构。

### 同源建模

同源建模的依据是：结构比序列更保守，序列相似的蛋白折叠也相似。完整流程为：

1. 用 profile 搜索（如 PSI-BLAST 或 profile HMM）找模板；
2. 检查模板的覆盖范围、分辨率（resolution）、结合的配体（ligand）和寡聚状态（oligomeric state）；
3. 构建 target–template 比对；
4. 从模板复制保守的主链（conserved backbone）；
5. 对插入/缺失区域做 loop 建模（loop modeling）；
6. 安放侧链转子异构体（side-chain rotamer）；
7. 局部优化并做立体化学验证（stereochemical validation，如 Ramachandran 图、键长键角检查）；
8. 用独立证据评价模型，而不是只看建模目标函数的分值。

同源建模的误差可来自模板选择、target–template 比对、插入缺失区、侧链堆积、构象状态和后续优化；哪一项占主导取决于模板相似度与覆盖范围，不存在固定的误差大小顺序。模板相似度高也不能自动保证活性状态、复合体界面或无序区正确——例如模板是无配体的非活性构象，模型就会继承这一状态。

约 30% 的 target–template 序列一致度（sequence identity）常被当作同源建模从相对可靠区进入“暮光区”（twilight zone）的经验界线，但它不是硬阈值。比对长度、结构域架构、profile 显著性、保守 motif 和模板质量会共同改变可信度：只覆盖 20 个残基的短片段即使有 35% 一致度也可能没有意义，而覆盖完整结构域的 profile 命中即使一致度更低也可能可靠。

### 穿线与折叠识别

当模板序列一致度很低、普通序列搜索找不到同源物时，穿线法换一个问法：不问“序列像不像”，而问“这条序列放进这个折叠里合不合适”。它把 target 残基放入模板的结构环境中，评分可包括埋藏程度（burial，疏水残基是否被埋）、二级结构相容性、成对接触（pairwise contact）和 gap 罚分。

若评分只含逐位置的环境项，可以直接用一维动态规划求最优穿线；但若成对接触得分依赖相距很远的两个比对位置（第 $i$ 个残基的得分取决于第 $j$ 个残基被放到哪里），标准一维 DP 不再严格可分，问题会变得非常困难，常需近似或迭代求解（例如固定其他位置的环境后做 DP，再更新）。

<br>

## Rosetta：片段组装与蒙特卡洛搜索

### 方法的发展

从头预测面对的核心困难是构象空间极大（Levinthal 悖论：即便每个残基只有少数几种可能构象，穷举也远超宇宙年龄）。Rosetta 的关键想法是不在连续扭转角空间中盲目搜索，而是借用已知结构中真实出现过的局部片段，把搜索空间压缩到“由合理局部构象拼出的结构”。

- **原始的片段组装（fragment assembly）**：Simons 等 1997 年的方法把短片段装配、模拟退火（simulated annealing）与贝叶斯打分（Bayesian scoring）结合。片段库由与目标序列局部窗口相容的已知结构片段组成；Monte Carlo 搜索反复替换局部主链扭转角（backbone torsion），在构象空间中寻找低能区域。
- **后续的两阶段流程**：Rohl 等 2004 年总结的流程中，低分辨率阶段使用残基级统计势，包含紧致性、疏水作用、二级结构配对等项（侧链用质心代表，计算便宜，适合大范围搜索）；高分辨率阶段再加入全原子的堆积、氢键、范德华作用与侧链转子异构体优化。

### 课程串讲中的具体流程

按课程串讲展开，流程如下：

1. 对目标序列做 BLAST/profile 搜索并得到位置特异打分矩阵（PSSM）。
2. 在 PDB 中，为每个相互重叠的 9 残基窗口（9-residue window）选出序列 profile 与局部二级结构都相容的片段（fragment）。
3. 一次 Monte Carlo 移动（move）用某个候选 9-mer 的主链扭转角替换目标窗口的扭转角。
4. 按 Metropolis 准则决定是否接受：

$$
P_{\mathrm{accept}}=\min\left(1,e^{-\Delta E/(k_BT)}\right)
$$

其中 $\Delta E$ 是移动前后的能量差，$k_B$ 是 Boltzmann 常数，$T$ 是（有效）温度。降能移动（$\Delta E<0$）总被接受；升能移动以随温度变化的概率 $e^{-\Delta E/(k_BT)}$ 被接受，这使搜索能够跳出局部盆地（basin）。模拟退火逐步降低有效温度：早期高温时大胆探索，后期低温时逐渐收敛到低能区域。

5. 运行大量独立轨迹（trajectory），得到一组诱饵结构（decoy ensemble），再按得分与结构相似性聚类。大而低能的簇（cluster）及其中心结构通常比孤立的最低分 decoy 更可信：很多独立搜索都落到同一区域，说明这个区域在能量面上确实宽而深。

9-mer 是课程强调的经典片段长度（Rosetta 也常配合 3-mer 片段做精细调整），不表示所有 Rosetta 协议都只使用 9-mer。

### 两项关键近似

方法依赖两项近似。第一，短序列片段的局部构象可由已知结构库覆盖；第二，所用能量函数的低值区域与天然样（native-like）盆地对应。两种失败方式分别对应这两点之外的搜索与打分：搜索不足会停在错误的盆地，能量函数误差则会把非天然构象排在前面。多个独立轨迹形成的低能结构应出现收敛（convergence）；单条最低能 decoy 缺少这种支持。Rosetta 得分由经验项与物理启发项加权构成，不应直接解释为实验测得的折叠自由能。

<br>

## 共进化与接触预测

### 动机：相互接触的残基会协同突变

如果蛋白中两个残基在空间上接触（例如一个带正电、一个带负电形成盐桥），其中一个突变后，另一个往往也要随之突变才能维持结构——否则蛋白会失稳并被自然选择淘汰。因此，在同一家族的多序列比对中，接触残基对应的两列会呈现相关的变化，即共进化（coevolution）。这与 RNA 一章中用协变推断碱基配对是同一个思想，只是蛋白有 20 种氨基酸、关系更复杂。

### 直接耦合分析

但多序列比对中的成对相关不全是接触信号，需要去除两种干扰：

- **共同祖先（系统发育）**：一个分支中的序列整体相似，会让许多不相关的列同时显得相关；
- **间接相关**：若 A 与 B 接触、B 与 C 接触，A 与 C 即使不接触也会显得相关（相关性沿链传递）。

直接耦合分析（Direct Coupling Analysis，DCA）用一个全局 Potts 模型同时拟合所有位点，以分离直接耦合：

$$
P(x)\propto\exp\left(\sum_i h_i(x_i)+\sum_{i<j}J_{ij}(x_i,x_j)\right)
$$

其中 $x=(x_1,\dots,x_L)$ 是长度为 $L$ 的一条比对序列，每个 $x_i$ 取 $q$ 种状态之一（20 种氨基酸加 gap，$q=21$）；$h_i(a)$ 是位点 $i$ 的单点场（反映该列的保守性），$J_{ij}(a,b)$ 是位点 $i,j$ 之间的耦合（$q\times q$ 矩阵）。因为所有耦合在同一个模型中联合估计，A–C 间的相关可以被 A–B、B–C 两个耦合解释掉，不必再赋予 $J_{AC}$ 大的值。

需要注意这种直接耦合的含义：观察到的协方差可能来自共同祖先，也可能是两个位点都与第三个位点相关；DCA 拟合全局模型后，估计的是在模型条件下不能由其他位点简单解释的统计耦合。这种直接耦合不等于直接物理接触，也不能彻底消除系统发育、MSA 错配和共同功能选择带来的影响。

从 $J_{ij}$ 到接触排序还有几个技术步骤。Potts 参数存在规范冗余（gauge redundancy）：把只依赖 $x_i$ 的一项 $g(x_i)$ 从 $J_{ij}(x_i,x_j)$ 中减去、加到 $h_i(x_i)$ 上，概率分布不变，所以 $J_{ij}$ 本身的数值不唯一。接触排序时常先转到零和规范（zero-sum gauge，使 $J_{ij}$ 每行每列和为零），再计算每个耦合矩阵的 Frobenius 范数 $F_{ij}=\|J_{ij}\|_F$ 作为位点对的耦合强度，并施加平均乘积校正（average-product correction，APC）：

$$
F_{ij}^{\mathrm{APC}}=F_{ij}-\frac{F_{i\cdot}\,F_{\cdot j}}{F_{\cdot\cdot}}
$$

其中 $F_{i\cdot}$、$F_{\cdot j}$、$F_{\cdot\cdot}$ 分别是第 $i$ 行、第 $j$ 列和全体的平均值。APC 用来扣除某些位点“和谁都相关”（例如高熵位点或系统发育造成）的背景偏差。

### 计算复杂度

参数量约为 $O(L^2q^2)$：有 $O(L^2)$ 个位点对，每对一个 $q\times q$ 矩阵。稠密平均场（mean-field）方法需要对 $Lq\times Lq$ 的协方差矩阵求逆，直接矩阵求逆可达 $O((Lq)^3)$；伪似然（pseudo-likelihood）方法每次计算目标函数或梯度约为 $O(M_{\mathrm{eff}}L^2q)$（$M_{\mathrm{eff}}$ 是重新加权后的有效序列数；每条序列、每个位点都要对 $q$ 个候选状态累加 $L$ 个耦合），还要再乘以优化的迭代次数。

### 信号质量与用途

有效序列数和 MSA 的多样性决定信号质量：序列太少或太相似时，耦合估计噪声很大。接触预测（contact prediction）可以把高耦合位点对作为折叠约束（folding restraint），交给 Rosetta 等方法使用。序列数据库增长后，更深的 MSA 能提供更多共同演化约束，即使新增序列本身没有结构标签——这正是后来深度学习方法能利用海量无结构序列的原因之一。

<br>

## AlphaFold：从预测距离到端到端预测坐标

### 第一代 AlphaFold（CASP13）

第一代 AlphaFold 在 CASP13（2018 年）中，从 MSA 和序列特征出发，用深度卷积网络预测残基对距离分布（distogram）与主链扭转角分布。它把 distogram 的负对数概率拟合为可微的、针对该蛋白的势函数（protein-specific potential），并加入扭转角势；结构以 $\phi,\psi$ 扭转角参数化，再用 L-BFGS 等梯度方法从多个初始化出发最小化总势能。这里的“能量”主要来自网络预测的分布（“网络认为残基 $i,j$ 最可能相距多远”），不是经典力场意义上的实验自由能。

### AlphaFold2（CASP14）

AlphaFold2 在 CASP14（2020 年）中改变了问题的分解方式：MSA 表示、残基对表示（pair representation）、模板信息与结构模块（structure module）端到端联合训练，并通过循环（recycling）反复细化。可以这样对比：AF1 是“先预测距离/角度，再优化势能”的两段式流程，AF2 则在网络内部迭代更新表示与坐标；二者不能只概括为同一个 CNN 的版本升级。

AF2 的两个核心组件是：

- **Evoformer**
  - 在 MSA 表示（$N_{\text{seq}}\times L$ 个向量）与残基对表示（$L\times L$ 个向量）之间反复交换信息：MSA 中的共进化信息写入 pair 表示，pair 表示又作为 attention 偏置影响 MSA 的更新；
  - 三角乘法更新/三角注意力（triangle multiplicative update/attention）沿残基三角形 $(i,j,k)$ 传播 pair 信息：边 $(i,j)$ 的更新参考共享第三个残基 $k$ 的两条边（如 $(i,k)$ 与 $(j,k)$），设计动机来自几何上的三角不等式；
  - 这种更新帮助网络学习几何一致性，但不在每一步严格保证 pair 表示对应一个合法的欧氏距离矩阵。
- **结构模块与 recycling**
  - 结构模块把每个残基表示为一个刚体 frame（旋转 + 平移），使用不变点注意力（invariant point attention，IPA）更新；
  - 其标量输出对整体刚体变换保持不变（invariant）；
  - 预测的坐标随整体刚体变换等变（equivariant）：输入整体转一下，输出坐标也跟着转；
  - recycling 将预测坐标与内部表示重新输入网络，通常迭代数次。

AlphaFold2 显著提高了单体结构的预测精度，但使用其输出时仍需结合以下信息：

- **pLDDT**：局部残基置信度；
- **PAE**：残基对相对位置误差；
- 多聚体界面、配体、构象状态与无序区等模型没有直接覆盖的方面；
- 训练数据的时间切分和模板泄漏（评测目标的同源结构是否已在训练集或模板库中）。

### 置信度指标的含义

**pLDDT** 是模型对每个残基 lDDT-$C_\alpha$ 分数的预测期望，范围为 0–100。lDDT（local distance difference test）检查一个残基周围局部距离是否与真实结构一致，因此 pLDDT 评价的是局部距离模式，不依赖全局叠合。经验上 pLDDT > 90 表示很高的局部精度，低于 50 的区域常对应无序或不可信区域。

**PAE**$(i,j)$ 以 Å 为单位，表示“若以残基 $j$ 的主链 frame 对齐预测结构与真实结构，残基 $i$ 的位置误差”的预测期望（AlphaFold 数据库的 PAE 图中，即以 $y$ 轴残基对齐、看 $x$ 轴残基的误差；不同文献中 $i,j$ 的角色可能对调，使用时应以所用工具的文档为准）。PAE 一般不对称：以一个刚性核心残基为参照看柔性 loop，与以柔性 loop 为参照看核心，误差可以不同。两个结构域可各自具有高 pLDDT，却因为相对朝向不确定而具有高的结构域间 PAE。

**多聚体预测**还需查看界面预测 TM-score（interface predicted TM-score）、界面 PAE、按物种配对的 MSA 与生物学合理性。pTM 汇总全局拓扑的置信度（预测的 TM-score），AlphaFold-Multimer 的 ipTM 更强调链间的相对排布；二者都是模型置信度，不是实验结合证据。模型可能把非生理接触预测成界面，也可能漏掉需要配体、膜（membrane）或翻译后修饰（PTM）才能稳定的状态。

### 解释边界

预测结构不是实验观测，也不等于完整的动力学系综。高 pLDDT 不能证明蛋白的功能或特定相互作用；低 pLDDT 可以来自真正的无序、柔性、缺少结合伙伴或单纯的预测失败，不能仅凭低分就宣布该区域是内在无序（intrinsically disordered）。还应检查不同随机种子、MSA/模板设置和不同模型之间预测结果的一致性。

> [!EXAMPLE]+ 例6：置信度的两种尺度
>
> 某双结构域蛋白的两个 domain 内 pLDDT 都为 90，但 domain 间 PAE 为 25 Å。合理结论是“两个局部 fold 较可信，二者相对朝向不确定”，而不是“完整蛋白结构高置信”。
>
> 原因在于两个指标的尺度不同：pLDDT 只看每个残基周围的局部距离，两个结构域各自内部折叠正确就能得到高分；PAE 则以一个结构域的残基为参照去看另一个结构域的残基，25 Å 的期望误差意味着两者之间的铰链方向基本不确定。此时若对整条链计算与实验结构的 RMSD，可能会很大，但分别对每个结构域叠合则可能很好。

> [!NOTE]+ 小结
>
> - CASP 以盲测评估预测方法，评价指标从 RMSD 扩展到 GDT、TM-score 和 lDDT。
> - 同源建模依赖模板，约 30% 一致度只是经验界线；穿线在低一致度时评价序列与折叠的相容性。
> - Rosetta 用 9-mer 片段替换加 Metropolis 蒙特卡洛与模拟退火搜索，靠大簇收敛而非单个最低分判断可信度。
> - DCA 用全局 Potts 模型从 MSA 中分离直接耦合，经零和规范、Frobenius 范数和 APC 得到接触排序。
> - AF1 预测距离后优化势能，AF2 用 Evoformer 与结构模块端到端预测坐标；pLDDT 衡量局部、PAE 衡量相对位置，二者都只是模型置信度。

<br>

# 蛋白质语言模型（Protein Language Models）

前面的方法要从序列中提取进化信息，通常都要先做一件昂贵的事：针对目标蛋白搜索同源序列、构建 MSA，再在这个家族上拟合 profile、Profile HMM 或 Potts 模型。对于孤儿蛋白（同源序列很少）或需要处理数百万条序列的场景，这一步要么做不了，要么太慢。蛋白质语言模型（protein language model，PLM）借鉴自然语言处理的思路：把氨基酸序列当作由 token 组成的“句子”，在数以千万计、没有任何标注的天然蛋白上训练一个大模型，让它学会“在这种上下文中，这个位置通常是什么残基”。代表性工作包括 ESM 系列（Rives 等 2021、Lin 等 2023 的 ESM-2/ESMFold）和 ProtTrans（Elnaggar 等 2021）。

核心想法是：如果一个模型能准确补全被遮住的残基，它就必须在内部表示中隐式地掌握替换偏好、保守 motif、二级结构倾向乃至残基间接触等信息。与前几章不同，它不预先写入替换矩阵、motif 或家族 profile，而是通过掩码 token 预测（masked-token prediction）或自回归预测（autoregressive prediction）学习上下文条件分布。

**输入**是一条或多条蛋白质序列；**输出**可以是每个残基的嵌入（embedding）、整条序列的嵌入、条件概率或生成的新序列。嵌入可作为功能分类、亚细胞定位、结构预测和变异效应模型的特征；突变的对数几率（mutation log-odds）衡量某个替换是否符合模型学到的序列分布，但不直接等于折叠自由能或临床致病概率。

和隐马尔可夫模型一章的 Profile HMM 对比最能看清它的定位。两者都是序列的概率分布模型：Profile HMM 针对一个家族建模，状态语义明确（匹配、插入、删除），参数较少；语言模型跨大量家族学习高维上下文，表达能力更强，但也更依赖训练数据和严格的数据划分。无论用哪一种，解释任何分数之前都要固定模型、分词器（tokenizer）、序列预处理和池化（pooling）方法。

## 训练目标

### 掩码语言模型

蛋白质语言模型把每个氨基酸当作一个 token（词表约 20 种标准氨基酸加若干特殊符号）。掩码语言模型（masked language model，MLM，如 ESM、ProtBert）的做法是：随机选一个位置集合 $M$（例如约 15% 的位置），把这些位置遮盖或替换后输入模型，要求模型预测原来的残基。它最小化被遮位置上的交叉熵（masked-token cross-entropy）：

$$
\mathcal L_{\mathrm{MLM}}
=-\mathbb E_M\sum_{i\in M}
\log P_\theta(x_i\mid\widetilde x_M)
$$

其中 $x=(x_1,\dots,x_L)$ 是原始序列，$\widetilde x_M$ 是遮盖或扰动后的输入，$\theta$ 是模型参数，期望对随机选择的遮盖集合取。需要注意，MLM 目标不是规范化的整序列似然（likelihood）：它只给出“已知其余部分时某几个位置的条件分布”，不能直接相乘得到 $P(x)$。

训练得到的嵌入可用于远缘同源识别、功能预测、变异效应、结构预测和序列生成。模型从大规模天然序列中吸收了进化统计——自然选择保留下来的序列就是“能折叠、能工作”的样本——因此不需要为每个任务重新手工设计特征。

### 自回归模型

自回归模型（autoregressive model，如 ProGen、ProtGPT2）则按给定顺序（通常从 N 端到 C 端）分解整条序列的概率：

$$
P(x)=\prod_{i=1}^{L}P(x_i\mid x_{<i})
$$

其中 $x_{<i}$ 表示位置 $i$ 之前的所有残基。

两种目标的取舍是：掩码模型可以双向利用上下文（预测第 $i$ 位时同时看到前后残基），更适合做表示和打分；自回归模型天然给出规范化的序列似然，并便于逐个残基生成新序列。无论哪种，二者学到的都是训练语料分布与目标函数共同决定的表示，不是显式的物理能量。

### 计算成本

PLM 通常基于 Transformer。对长度 $L$、隐藏维度 $d$ 的标准 Transformer 层，自注意力（attention）的主要计算量为 $O(L^2d)$（每对位置算一次 $d$ 维点积），attention 矩阵的显存为 $O(L^2)$；完整的一层还包含 $O(Ld^2)$ 的投影与前馈网络（projection/MLP）成本。只有把 $d$ 当作常数时，才能简称为 $O(L^2)$。因此长蛋白（数千个残基）、多链输入或大量序列同时输入会成为瓶颈，实际使用时常需截断或分段。

<br>

## 提取嵌入：实际使用方法

### 代码示例

下面是用 Hugging Face 风格接口提取残基级和蛋白级嵌入的示意代码，假设 `model`、`tokenizer`、`sequence` 已加载（例如某个 ESM-2 检查点）：

```python
# 示意：真实项目应固定模型版本、tokenizer 和 pooling 方案
# 推理阶段关闭 dropout 等随机训练行为。
model.eval()
encoded = tokenizer(
    sequence,
    return_tensors="pt",
    return_special_tokens_mask=True,
)
# 只把模型真正接收的 tensor 移到同一 device。
model_inputs = {
    key: value.to(model.device)
    for key, value in encoded.items()
    if key != "special_tokens_mask"
}
with torch.no_grad():
    output = model(**model_inputs)
# 同时排除 padding 与 tokenizer 插入的 BOS/EOS 等特殊 token。
valid = (
    encoded["attention_mask"].bool()
    & ~encoded["special_tokens_mask"].bool()
)[0].to(model.device)
residue_embedding = output.last_hidden_state[0, valid]
# 简单 mean pooling 得到定长向量，但会丢失残基位置关系。
protein_embedding = residue_embedding.mean(dim=0)
```

`residue_embedding` 的形状为 $(L,d)$，每行对应一个真实残基；`protein_embedding` 是长度 $d$ 的向量，可直接作为下游分类器的输入。

### 实现中的常见坑

- **特殊 token**：不能通用地假设 `1:-1` 切片恰好去除一个 BOS 和一个 EOS。不同分词器插入的特殊 token 个数和位置不同，批处理时还有 padding；只有特殊 token 掩码（special-token mask）与注意力掩码（attention mask）结合，才能同时排除特殊 token 和 padding。
- **池化**：均值池化（mean pooling）会丢失位点信息，例如两个蛋白在活性位点不同但其余部分相同，均值向量可能几乎一样。
- **数据问题**：长序列截断、训练集与测试集的同源泄漏、物种偏差（训练语料中某些类群过度代表）都会显著影响结果。

### 不同层次的用法

常见的使用层次包括：

- **残基嵌入（residue embedding）**：预测二级结构、结合位点（binding site）、变异效应；
- **序列嵌入（sequence embedding）**：预测蛋白家族、亚细胞定位（localization）、功能；
- **残基对表示/注意力（pair representation/attention）**：预测接触或结构（如 ESMFold 由 ESM-2 表示直接预测结构，无需 MSA）；
- **条件生成（conditional generation）**：按家族、结构或功能设计序列。

评估嵌入质量时有两种方式。线性探针（linear probe）冻结语言模型，只在嵌入上训练一个线性分类器，可测试冻结表示中是否已经包含可线性读出的信息；完整微调（full fine-tuning）同时更新模型参数，容量更高，但更容易过拟合。比较不同方法时，应保持下游的数据划分（split）和标签数量（label budget）一致，否则差异可能只来自数据而非表示。

<br>

## 分数的解释边界与评估

### 高似然不等于好蛋白

语言模型给出的高似然只表示序列符合训练分布，不直接等于蛋白稳定、可表达或具有目标功能。零样本（zero-shot）突变打分常用突变前后的对数似然差，但应以深度突变扫描（deep mutational scanning，DMS）实验数据校准，才能知道分数与实际适应度（fitness）的对应关系。

### 伪对数似然

掩码模型没有规范化的 $P(x)$，常用伪对数似然（pseudo-log-likelihood，PLL）替代：若每次只遮一个位置，

$$
\operatorname{PLL}(x)=\sum_i\log P(x_i\mid x_{\setminus i})
$$

其中 $x_{\setminus i}$ 表示除位置 $i$ 外的整条序列。精确的 PLL 需要依次遮住每一个位置，通常需要 $L$ 次前向传播（forward pass），对长蛋白和大量突变体代价较高。

### 单点突变的掩码边际分数

对单点突变，更常用的是突变位点的掩码边际分数（mutation-site masked marginal，ESM-1v 论文中采用的打分方式）：只遮住突变位点，比较模型给突变残基和野生型残基的概率：

$$
\Delta s=
\log P_\theta(x_i^{\mathrm{mut}}\mid x_{\setminus i})
-\log P_\theta(x_i^{\mathrm{wt}}\mid x_{\setminus i})
$$

其中 $x_i^{\mathrm{wt}}$ 是位置 $i$ 的野生型残基，$x_i^{\mathrm{mut}}$ 是突变后的残基。$\Delta s>0$ 表示模型认为突变残基在此上下文中比野生型更“自然”，$\Delta s<0$ 表示突变偏离训练分布，常提示可能有害。它只需一次前向传播就能对该位置所有 19 种替换打分。

需要注意，$\Delta s$ 不是完整的突变体 PLL 与野生型 PLL 之差：后者还会包含突变对其他位置条件概率的影响。多点突变时必须说明采用的是同时遮盖（simultaneous masking）、逐位分数相加，还是重新计算完整 PLL，三者结果不同。不同蛋白、不同长度或不同模型之间的绝对分数也不可直接混用，比较通常在同一蛋白、同一模型内进行。

### 评估中的同源泄漏

模型评估必须控制同源泄漏。近缘序列、重复的数据库记录以及同一家族跨越训练/测试划分，会使随机拆分的结果严重乐观——模型可能只是“记住了”测试蛋白的近亲。应报告序列一致度阈值（identity threshold，例如测试集与训练集一致度低于 30%）、时间切分，以及按家族汇总的结果。此外，注意力权重（attention weight）也不自动等于接触概率或因果解释：虽然部分注意力头与接触图相关，但这需要用已知结构验证，不能直接把注意力图当作结构证据。

> [!NOTE]+ 小结
>
> - PLM 在大量无标注序列上学习条件分布，不显式构建 MSA，也不预先写入替换矩阵或 profile。
> - MLM 双向利用上下文但不给规范化似然；自回归模型给出 $P(x)=\prod_iP(x_i\mid x_{<i})$，便于生成。
> - 提取嵌入时要用 attention mask 与 special-token mask 排除特殊 token 和 padding，均值池化会丢失位点信息。
> - 单点突变常用掩码边际分数 $\Delta s$，它不等于 PLL 之差，也需要 DMS 数据校准。
> - 高似然不等于稳定或有功能；评估必须控制同源泄漏，注意力权重不能直接当作接触证据。

<br>

<br>

# 分子力学：力场、能量最小化与分子动力学（Molecular Mechanics）

实验测定或结构预测给出的是一组静态坐标，但很多生物学问题需要的是“这个结构稳不稳”“侧链会怎样摆动”“配体结合后口袋会不会张开”。要回答这些问题，必须能够对任意一组原子坐标打分：哪个构象能量更低、每个原子受到多大的力、沿力的方向走下去结构会变成什么样。量子化学可以从电子层面回答，但对一个溶在几万个水分子里的蛋白质来说计算量完全不可承受。分子力学（molecular mechanics）的思路是：放弃显式求解电子，把原子看成带电荷、有质量、由弹簧相连的小球，用一个经验参数化的势能函数近似整个体系的能量。

核心想法一句话：**给定坐标 $\mathbf r$，用一个便宜的经验公式 $E(\mathbf r)$ 打分，再用它的负梯度作为力。** 有了力，就可以做两件事：沿下坡方向走到附近的局部极小值（能量最小化），或者把力代入 Newton 方程，让体系随时间演化（分子动力学，molecular dynamics，MD）。本章承接蛋白质结构、结构预测与结构叠合等主题提供的初始坐标；MD 产生的轨迹随后交给下一章的三维统计，做对齐、降维、聚类和状态建模。

## 分子力学的基本设定与适用边界

### 研究对象与基本近似

分子力学把蛋白质、配体、水和离子表示成一组原子，每个原子带有原子类型（atom type）、质量、部分电荷（partial charge），原子之间有固定的化学键拓扑（bond topology）。参数化势能函数 $E(\mathbf r)$ 近似给定坐标 $\mathbf r$ 时体系的能量，其中 $\mathbf r$ 是所有 $n$ 个原子坐标拼成的 $n\times 3$ 数组。

这个近似有一个关键前提：**成键方式固定**。模型不显式求解电子波函数，而是事先规定哪些原子之间有共价键，再用经验数据和量子化学计算拟合出来的参数，描述键长、键角、二面角、范德华（van der Waals）作用和静电作用。由此带来的代价是：普通力场无法自然地描述化学键的形成和断裂（例如酶催化反应），这类问题需要 QM/MM 或反应力场。

### 从势能到力

原子 $i$ 受到的力等于势能对该原子坐标的负梯度：

$$
\mathbf F_i=-\nabla_iE(\mathbf r)
$$

这里 $\nabla_i$ 表示只对第 $i$ 个原子的三个 Cartesian 坐标求偏导，所以 $\mathbf F_i$ 是一个三维向量。直觉上，力指向“能量下降最快”的方向：两个原子靠得太近时，排斥项的能量急剧升高，负梯度就把它们推开。

### 势能、最小化与动力学的分工

势能函数首先解决的是“怎样比较相邻构象”的问题，并用上式计算每个原子的力。在此基础上有两种用法：

- **能量最小化**沿力的方向移动原子，消除原子间的空间冲突（steric clash），寻找起始构象附近的局部极小值；
- **分子动力学**把力代入 Newton 方程，生成位置和速度随时间变化的轨迹。

需要注意，一次计算的输入远不止 PDB 坐标。还必须给出 atom type、bond topology、质子化状态（protonation）、力场（force field）参数、水模型、边界条件以及温度和压力设置。输出则是势能、力、优化后的坐标或轨迹。初学者最常见的错误，就是以为“把 PDB 文件丢进程序”就够了，而忽略了这些同样决定结果的建模选择。

### 解释边界：势能不是自由能

势能 $E(\mathbf r)$ 不是实验测得的折叠自由能 $\Delta G$。前者评价**一个**微观构象；后者比较的是包含大量构象和溶剂状态的宏观系综（ensemble），同时含焓与熵的贡献。由此有两个直接推论：

- 一次能量最小化不能预测真实的折叠路径，它只是让结构滑进最近的谷底；
- MD 能否估计平衡性质，还取决于力场是否准确、ensemble 是否设置正确、采样是否充分。

### 与相邻主题的关系

本章从实验或预测结构提供的初始坐标出发。可以把两类“能量”对比来看：结构预测中的学习势（learned potential）用数据约束去**生成**坐标；经典力场则为一个给定的原子体系**计算**能量和力。MD 轨迹随后进入三维统计，做对齐、降维、聚类和状态建模。

<br>

## 势能函数：力场的组成

### 为什么低势能不等于稳定

在讨论力场公式之前，先把“稳定性”说清楚。热力学稳定性由自由能而非单一势能决定：

$$
\Delta G=\Delta H-T\Delta S
$$

其中 $\Delta H$ 汇总相互作用能与溶剂等焓贡献，$\Delta S$ 包含蛋白、溶剂和离子的构象熵以及平移、转动熵，$T$ 是绝对温度。某个构象的力场势能较低，不足以证明它所在的宏观态自由能最低；后者还取决于该状态占据的相空间体积。打个比方：一个很深但极窄的坑，可能不如一个稍浅但宽阔的盆地更常被占据，因为后者包含的微观构象多得多（熵大）。

### 成键项与非键项

经典力场把势能分为成键（bonded）项与非键（non-bonded）项：

$$
E=E_{\text{bond}}+E_{\text{angle}}+E_{\text{dihedral}}
+E_{\text{vdW}}+E_{\text{electrostatic}}
$$

前三项只作用于由共价键相连的原子（相隔 1、2、3 根键），描述局部几何；后两项作用于空间中相距较远或不直接成键的原子对。常见的函数形式包括：键长和键角用谐振子（弹簧）、二面角用周期函数（torsion）、范德华用 Lennard–Jones 势、静电用 Coulomb 势。其中 Lennard–Jones 势为

$$
E_{\mathrm{LJ}}(r)=4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12}
-\left(\frac{\sigma}{r}\right)^6\right]
$$

$r$ 是两原子间距离，$\epsilon$ 是势阱深度，$\sigma$ 是势能为零时的距离。$r^{-12}$ 项描述电子云重叠造成的短程强排斥，$r^{-6}$ 项描述色散吸引；对 $r$ 求导令其为零，可得极小值位于 $r_{\min}=2^{1/6}\sigma\approx1.12\sigma$，此处 $E_{\mathrm{LJ}}=-\epsilon$。$r$ 小于 $\sigma$ 后能量急剧上升，这正是“原子不能互相穿透”的数学表达。

再强调一次：力场是参数化近似，不显式求电子波函数，通常不能自然处理成键和断键。

### 课程中的 Levitt 势

课程所称的 Levitt 势（源自 Michael Levitt 早期的蛋白质能量计算工作）可以放在同一框架中理解：它是 bond、angle、dihedral、Coulomb 与 van der Waals 项的加和。各项的作用分别是：

- **bond / angle** 维持局部几何，使键长和键角不偏离化学上合理的数值；
- **dihedral** 控制绕单键旋转的偏好，例如主链 $\phi,\psi$ 和侧链 $\chi$ 角的取值；
- **Coulomb** 表示带电或部分带电原子之间的静电作用，如盐桥和氢键中的静电成分；
- **van der Waals** 包含短程排斥与中程吸引，强烈约束原子的堆积（packing）。

课上常说“vdW 主导 packing”，这句话描述的是范德华项对近接几何的直接作用：原子不能重叠，又倾向于紧密贴合。它并不表示总折叠自由能在所有体系中都由 vdW 单独主导；疏水效应、静电和熵同样重要。

### 一个完整的示意性力场

把上述各项写全，一个示意性的加和式固定电荷力场（additive fixed-charge force field）可写为

$$
\begin{aligned}
E={}&\sum_{\text{bonds}}k_b(b-b_0)^2
+\sum_{\text{angles}}k_\theta(\theta-\theta_0)^2\\
&+\sum_{\text{dihedrals}}k_\phi[1+\cos(n\phi-\delta)]\\
&+\sum_{i<j}4\epsilon_{ij}
\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}
-\left(\frac{\sigma_{ij}}{r_{ij}}\right)^6\right]\\
&+\sum_{i<j}\frac{q_iq_j}{4\pi\epsilon_0\epsilon_r r_{ij}}
\end{aligned}
$$

逐项说明符号：

- $b$ 是实际键长，$b_0$ 是平衡键长，$k_b$ 是键的力常数；键越硬，$k_b$ 越大。
- $\theta,\theta_0,k_\theta$ 分别是键角、平衡键角和角力常数。
- $\phi$ 是二面角，$n$ 是周期数（转一圈出现几个极小值，例如 $sp^3$–$sp^3$ 单键常取 $n=3$），$\delta$ 是相位，$k_\phi$ 是势垒高度参数。
- $r_{ij}$ 是原子 $i,j$ 的距离，$\epsilon_{ij},\sigma_{ij}$ 是这对原子的 LJ 参数，通常由各自原子类型的参数按组合规则（combination rule）得到。
- $q_i,q_j$ 是部分电荷，$\epsilon_0$ 是真空介电常数，$\epsilon_r$ 是相对介电常数。

谐振子项在 $b=b_0$ 时为零、偏离时按平方增长，相当于把化学键当作弹簧；二面角项是余弦函数，自然具有 $2\pi$ 周期性。

### 力场参数是成套的

AMBER、CHARMM、OPLS 等力场家族（force-field family）中的 atom type、partial charge 和各项参数是**成套拟合**的，彼此补偿，不应随意混合：例如把 CHARMM 的电荷配上 AMBER 的二面角参数，两者的误差不再相互抵消，结果可能明显变差。水模型（如 TIP3P、SPC/E、TIP4P）也是参数体系的一部分，应与力场推荐的组合一起使用。

起点的化学状态同样重要。组氨酸的质子化状态（protonation state）、互变异构体（tautomer）和金属配位（metal coordination）若设置错误，长时间模拟也不能自动修正这类起点上的化学错误，因为固定键拓扑的力场根本不允许质子转移或配位键重排。

### 示意式省略了什么

上面的示意式仍省略了不少实际力场中的项和规则：

- **额外项**：improper dihedral（维持平面性和手性）、Urey–Bradley 项（CHARMM 中 1–3 原子间的距离项）、CMAP（CHARMM 中对主链 $\phi/\psi$ 的二维修正网格）、几何约束、极化（polarization）以及特殊的原子对规则。
- **约定差异**：不同力场在谐振子项前是否带 $1/2$ 系数（AMBER 把 $1/2$ 吸收进力常数，所以写成 $k(b-b_0)^2$）、torsion 的符号与相位约定、组合规则（如 Lorentz–Berthelot）、1–2/1–3 排除（exclusion）以及 1–4 缩放（scaling）方面各不相同。
- **非键求和范围**：非键项并非简单覆盖所有 $i<j$。通过 1 根或 2 根键相连的原子对（1–2、1–3）通常被排除，相隔 3 根键的 1–4 对按力场规定的比例缩放。

另外，显式溶剂的固定电荷模型通常取 $\epsilon_r=1$，屏蔽效应由显式的水分子与离子自己产生，而不是靠人为调大介电常数。只有在隐式溶剂或真空近似中，才会使用距离依赖介电常数之类的经验处理。

<br>

## 能量最小化

### 动机：先把明显的坏接触消除

晶体结构、同源建模或刚加完氢和水的体系，往往存在几处原子靠得过近的坏接触。此时 LJ 排斥项的能量巨大，力也巨大，若直接开始 MD，第一步就可能把原子“炸飞”。能量最小化（energy minimization）的任务是在不改变整体结构的前提下，沿着下坡方向把体系移到附近的局部极小值。

目标是找局部极小值，即满足 $\nabla E(\mathbf{x})=0$ 的点，其中 $\mathbf x$ 是所有坐标拼成的 $3n$ 维向量。需要明确：最小化只能消除坏接触并进入附近的能量盆地（basin），它不给出热平衡分布，也不能证明找到了全局最低能量。

### 常用算法

**最速下降（steepest descent）** 每步沿负梯度方向移动：

$$
\mathbf x_{k+1}=\mathbf x_k-\eta_k\nabla E(\mathbf x_k)
$$

其中 $\eta_k$ 是第 $k$ 步的步长。由于负梯度就是力，这一步相当于“顺着力推一小段”。步长过大会在谷底两侧来回震荡，过小则走得很慢。最速下降的优点是稳健，即使起点很差也能快速降低巨大的排斥能；缺点是到了后期在狭长山谷中呈“之”字形前进，收敛缓慢。

**共轭梯度（conjugate gradient）与 L-BFGS** 利用更多曲率信息。共轭梯度让新的搜索方向与之前的方向在 Hessian 意义下共轭，避免重复走回头路。L-BFGS 属于拟牛顿法：它不显式保存完整的 $3n\times3n$ Hessian（对于数万原子的体系根本存不下），而是用最近若干步的梯度差和位置差来近似 inverse Hessian，因此适合高维坐标。实践中常见的组合是先用最速下降消除大的冲突，再换共轭梯度或 L-BFGS 精细收敛。

### 收敛判据

收敛标准可以看最大力（maximum force）、均方根力（RMS force）或能量变化。需要特别警惕的是：仅仅能量变化小，可能只是步长太小导致每步几乎不动，并不代表梯度已经足够接近 0。因此更可靠的判据是力本身，例如 GROMACS 默认要求最大力低于某个阈值。

<br>

## 分子动力学

### 动机：让结构动起来

能量最小化只给出一个静止的谷底，但真实蛋白在室温下不停热运动，功能往往依赖这种运动（如通道开合、结构域转动）。分子动力学用数值方法积分 Newton 第二定律，让体系在给定温度下随时间演化，从而得到构象的分布和动力学信息。

### 运动方程

对每个原子 $i$，

$$
m_i\frac{d^2\mathbf{r}_i}{dt^2}=-\nabla_iE(\mathbf{r})
$$

左边是质量乘加速度，右边是力场给出的力。该方程对数万原子耦合在一起，没有解析解，只能以很小的时间步 $\Delta t$ 一步步推进。

### 时间步长的限制

时间步必须明显小于体系中最快运动的周期，否则积分误差会迅速累积导致能量发散。含氢共价键（X–H，如 C–H、N–H、O–H）的伸缩振动频率最高，周期约 10 fs，因此限制了全原子（all-atom）MD 的时间步：

- 不加约束时常用约 1 fs；
- 用约束算法固定 X–H 键长后，常用约 2 fs；
- 氢质量重分配（hydrogen-mass repartitioning，HMR）等方案把重原子的部分质量挪给相连的氢，使氢振动变慢，可使用约 4 fs，但会改变部分动力学时间尺度。

约束算法、恒温器（thermostat）、恒压器（barostat）、周期性边界条件和长程静电的处理方式共同决定模拟实际采样的是哪个 ensemble。

### Velocity Verlet 积分器

Velocity Verlet 以小时间步同时推进位置和速度，是最常用的 MD 积分器之一。它的核心分两步。先用当前速度和加速度推进位置（相当于二阶 Taylor 展开）：

$$
\mathbf r(t+\Delta t)=\mathbf r(t)+\mathbf v(t)\Delta t
+\frac{\mathbf a(t)}{2}\Delta t^2
$$

其中 $\mathbf a(t)=\mathbf F(t)/m$。然后在新位置上计算力，得到 $\mathbf a(t+\Delta t)$，再用新旧加速度的平均更新速度：

$$
\mathbf v(t+\Delta t)=\mathbf v(t)
+\frac{\mathbf a(t)+\mathbf a(t+\Delta t)}{2}\Delta t
$$

为什么用两端加速度的平均？因为在 $[t,t+\Delta t]$ 区间内加速度在变化，取梯形平均比只用起点加速度精确一阶。这种格式具有时间可逆性和辛（symplectic）性质，长时间模拟中能量不会系统性漂移，这正是它比简单的 Euler 法更适合 MD 的原因。

下面的实现推进一个无约束、恒定粒子数/体积/能量（NVE）的时间步：

```python
import numpy as np

def velocity_verlet(position, velocity, mass, force_fn, dt):
    """推进一个无约束 NVE Velocity-Verlet 时间步。

    position、velocity 为 (n, 3)，mass 为 (n,)；force_fn 接收坐标并
    返回相同 shape 的力。单位必须自洽，例如 nm、ps、amu 与对应力单位。
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if position.shape != velocity.shape or position.ndim != 2:
        raise ValueError("position and velocity must have the same 2-D shape")
    if mass.shape != (len(position),) or np.any(mass <= 0):
        raise ValueError("mass must be a positive vector of length n")
    if dt <= 0:
        raise ValueError("dt must be positive")

    # mass[:, None] 把每个粒子质量广播到其三个 Cartesian 分量。
    force = np.asarray(force_fn(position), dtype=float)
    if force.shape != position.shape:
        raise ValueError("force_fn returned the wrong shape")
    acceleration = force / mass[:, None]
    next_position = (
        position + velocity * dt + 0.5 * acceleration * dt**2
    )

    # 新位置决定下一时刻的力；这是每步主要计算成本。
    next_force = np.asarray(force_fn(next_position), dtype=float)
    if next_force.shape != position.shape:
        raise ValueError("force_fn returned the wrong shape")
    next_acceleration = next_force / mass[:, None]
    next_velocity = velocity + 0.5 * (
        acceleration + next_acceleration
    ) * dt
    return next_position, next_velocity
```

这个函数为了自包含，每次调用都重新计算起点的力；真实程序会把上一步的 `next_force` 缓存下来作为下一步的 `force`，这样每步只需一次力计算。单位方面，常用组合是 nm、ps、amu 和 kJ/mol/nm，此时加速度单位恰好自洽。

### 计算成本：非键力

每步最昂贵的部分通常是非键力。直接对所有原子对计算是 $O(N^2)$（$N$ 为原子数）。短程的 LJ 作用衰减很快，可以设截断半径（cutoff），并用邻居列表（neighbor list）只记录截断内的原子对，使短程部分降到近线性。静电作用按 $1/r$ 衰减，简单截断会引入明显误差；粒子网格 Ewald（Particle Mesh Ewald，PME）把它拆成实空间的短程部分和倒空间的长程部分，长程部分把电荷分配到网格上再用 FFT 求解，典型复杂度为 $O(N\log N)$。

### 积分器与 ensemble 的细节

上面两式只表示无约束、确定性的基本积分器。实际模拟往往还要加入：

- **SHAKE/RATTLE** 等约束算法：每步后把坐标和速度投影回满足键长约束的流形；
- **Langevin thermostat**：加入摩擦和随机力的子步；
- **Trotter 分裂的 thermostat/barostat**：把恒温恒压的耦合拆成若干子步插入积分流程。

关于 ensemble 有两点常被误解。第一，NVE ensemble 固定粒子数、体积和能量（即 $N,V,E$），但有限时间步的数值轨迹通常只是近似守恒能量；辛积分器更准确地守恒的是一个与真实 Hamiltonian 略有差别的影子 Hamiltonian（shadow Hamiltonian），所以能量表现为有界的小幅波动而非漂移。第二，NVT 或 NPT 固定的是 ensemble 的参数及其分布，而不是让瞬时温度或压力每一步都等于目标值。瞬时温度本来就应围绕目标值涨落，涨落幅度随体系变小而增大。

### 典型模拟流程

一次标准 MD 研究通常依次经历以下步骤：

1. **准备结构**（prepare structure）：补全缺失原子和残基，处理多构象；
2. **确定质子化状态**（protonation）：按目标 pH 判断 His、Asp、Glu 等的状态；
3. **溶剂化**（solvation）：把蛋白放进水盒子；
4. **加离子**（ions）：中和净电荷并达到目标盐浓度；
5. **能量最小化**（minimization）：消除坏接触；
6. **平衡**（equilibration）：通常先在 NVT 下升温、再在 NPT 下调整密度，常对蛋白重原子加位置约束；
7. **生产模拟**（production）：去掉约束，长时间采样；
8. **轨迹分析**（trajectory analysis）。

轨迹分析常用的 RMSD、RMSF、回转半径（radius of gyration）和接触占有率（contact occupancy）都只是轨迹的摘要统计量，它们把高维的运动压缩成一两个数字，必然丢失信息。

### RMSF：逐原子的涨落

均方根涨落（root-mean-square fluctuation，RMSF）衡量每个原子围绕其平均位置的波动幅度。对原子 $i$：

$$
\operatorname{RMSF}_i=
\sqrt{\left\langle\|\mathbf r_i(t)-\langle\mathbf r_i\rangle_t\|^2\right\rangle_t}
$$

其中 $\langle\cdot\rangle_t$ 表示对轨迹各帧取时间平均。它和 RMSD 的区别是：RMSD 对每一帧在所有原子上取平均，得到一条随时间变化的曲线；RMSF 对每个原子在所有帧上取平均，得到一条沿序列分布的曲线，常用来找柔性环区和稳定的核心。RMSF 与晶体学 B 因子有关，可近似换算为 $B_i=\tfrac{8\pi^2}{3}\mathrm{RMSF}_i^2$。

计算前通常要把每一帧对齐（用结构叠合中的 Kabsch 方法）到共同的参考结构，去除整体平移和转动，否则 RMSF 会被整体漂移淹没。若蛋白存在真实的结构域运动（domain motion），对全蛋白对齐与只对单个结构域对齐会给出不同的 RMSF：前者把铰链运动摊到两个结构域上，后者让被对齐的结构域显得很刚、另一个结构域显得很柔。解释时必须说明对齐所用的原子集合。

### 从轨迹到自由能

自由能与状态概率满足

$$
\Delta G_{A\rightarrow B}=-k_BT\ln(P_B/P_A)
$$

其中 $k_B$ 是 Boltzmann 常数，$P_A,P_B$ 是体系处于状态 $A$、$B$ 的概率。例如在 300 K 下 $k_BT\approx0.6$ kcal/mol，若 $P_B/P_A=10$，则 $\Delta G\approx-1.4$ kcal/mol。可见 MD 的统计目标并不是“看一段动画”，而是从目标 ensemble 中估计状态概率与各种观测量的期望。用单条轨迹的时间平均近似 ensemble 平均，需要遍历性（ergodicity）和充分采样作为前提。

> [!WARNING]+ 采样瓶颈
>
> 轨迹很长不等于独立样本很多。相关时间、稀有跃迁和初始构象会限制有效样本量。Replica exchange、metadynamics 等增强采样方法引入额外假设，并需要正确地重加权（reweight）。

<br>

> [!EXAMPLE]+ 例7：轨迹帧不是独立样本
>
> 一条 1 μs 轨迹每 1 ps 保存一帧，共有 $10^6$ 帧。设某个观测量的归一化自相关函数为 $\rho(t)$，积分相关时间为 $\tau_{\mathrm{int}}=\int_0^\infty\rho(t)\,dt=10$ ns。相隔约 $2\tau_{\mathrm{int}}$ 的两帧才近似独立，于是
>
> $$
> N_{\mathrm{eff}}\approx\frac{T}{2\tau_{\mathrm{int}}}=\frac{1000\ \text{ns}}{2\times10\ \text{ns}}\approx50,
> $$
>
> 即 $N_{\mathrm{eff}}\approx T/(2\tau_{\mathrm{int}})\approx50$，而不是一百万。保存频率从 1 ps 提高到 0.1 ps 只会让帧数增加十倍，$N_{\mathrm{eff}}$ 却不变。
>
> 需要注意约定：有的软件把因子 2 并入其“相关时间”的定义（报告 $\tau=2\tau_{\mathrm{int}}=20$ ns，再用 $N_{\mathrm{eff}}=T/\tau$，结果同样是 50）；若把只含一倍积分的 $\tau_{\mathrm{int}}=10$ ns 直接代入 $T/\tau$，就会得到约 100。两种算法差两倍，报告时必须注明所用约定。

这个例子说明，用逐帧数据计算标准误时，若把 $10^6$ 帧当作独立样本，标准误会被低估约 $\sqrt{10^6/50}\approx141$ 倍。

## 平衡诊断

模拟结果是否可信，首先取决于它是否真的采样了目标 ensemble。ensemble 由 Hamiltonian（力场）、边界条件、thermostat/barostat、约束与采样权重共同定义。单条长轨迹不能自动证明已经达到平衡：RMSD 曲线变平，可能只是体系被困在某个亚稳态里。

实际检查应包括以下几方面：

- **平衡期（equilibration）**：丢弃初始的弛豫段，并检查结论对丢弃长度是否敏感；
- **多个独立初态**：从不同起始构象或不同随机初速度出发，看是否收敛到相同分布；
- **状态往返**：关心的状态之间是否发生了多次往返跃迁，只跃迁一次的数据无法估计相对概率；
- **观测量的相关时间**：据此计算有效样本量和误差棒；
- **重复（replicate）间一致性**：不同重复给出的估计值是否在误差范围内一致。

增强采样得到的有偏轨迹（biased trajectory）必须用相应的权重（例如 umbrella sampling 用 WHAM/MBAR，metadynamics 用偏置势重加权）重建无偏的平衡概率，不能直接对有偏帧计数。

> [!NOTE]+ 小结
>
> - 力场用经验势能 $E(\mathbf r)$ 近似能量，力是其负梯度；成键方式固定，不能自然描述化学反应。
> - 力场 = 键长、键角、二面角 + LJ + Coulomb；参数成套拟合，不可混用，质子化等起点错误不会被模拟修正。
> - 能量最小化只进入附近局部极小，收敛应看力而不只看能量变化。
> - MD 用 Velocity Verlet 积分 Newton 方程，时间步受 X–H 振动限制（1/2/4 fs）；PME 处理长程静电。
> - 低势能不等于低自由能；轨迹帧高度相关，有效样本量由相关时间决定，平衡需要多重诊断。

<br>

<br>

# 三维统计：构象集合的对齐、降维与状态建模

上一章的 MD 会产生成千上万帧坐标；NMR 结构给出一组 conformer；同源蛋白家族给出一批相似但不同的结构。面对这些**一组**三维观测，需要回答的问题变成：平均形状是什么？主要的运动模式是哪几种？构象可以分成几类？它们之间怎样转换？这就是三维统计要做的事。

难点在于三维坐标带有“几何等价关系”。同一个结构平移或旋转后坐标数值全变了，但生物学上毫无变化；周期性的二面角在 $-\pi$ 和 $\pi$ 处其实是同一个角度，数值上却相差 $2\pi$；同源亚基或对称原子（如 Phe 侧链两侧的环碳）可以互换编号。核心想法一句话：**先把这些不代表生物差异的自由度去掉或用不变量表示，再做普通的统计。** 否则统计模型会把坐标系的差异误当作生物学变化。

## 研究对象、方法与统计边界

### 输入与输出

三维统计的输入可以是对齐后的坐标矩阵，也可以是距离、接触（contact）或扭转角（torsion）等特征。输出包括平均形状、主成分、慢过程、构象簇和状态转移矩阵。几种主要方法的分工是：

- **主成分分析（PCA）** 找方差最大的方向，即“摆动幅度最大”的集体运动；
- **时间结构独立成分分析（time-lagged independent component analysis，tICA）** 找时间相关性最强、也就是变化最慢的方向；
- **Markov 状态模型（Markov state model，MSM）** 把连续轨迹离散为若干状态，并估计在给定滞后时间（lag time）下的转移概率。

### 与前面主题的关系和统计边界

本章承接结构叠合与分子动力学：前者提供去除刚体自由度的工具，后者产生带有时间相关性的样本。由此带来最重要的统计提醒：轨迹帧通常不独立，不能把一百万帧当作一百万个生物学重复。有效样本量、轨迹重复（trajectory replicate）、对称性处理和距离定义共同决定了统计结论该如何解释。

## 不变量与对齐

三维生物数据通常同时存在旋转、平移、尺度、对称和缺失等问题。在做任何统计之前，需要先做出以下几项决定：

- **先对齐坐标，还是直接使用不变量**：对齐（例如 Kabsch 叠合）之后可以直接使用坐标，结果直观但依赖参考结构和对齐原子；距离矩阵、角度等内部坐标天然不受刚体运动影响，但维度更高，且距离矩阵不能区分镜像。
- **对称等价原子如何匹配**：例如 Asp 的两个羧基氧、同源寡聚体中的多条链，应选择使距离最小的编号方式。
- **ensemble 中不同构象如何加权**：来自增强采样或不同长度轨迹的帧，权重未必相等。
- **测量不确定性如何传播**：晶体结构的 B 因子、NMR 约束的松紧都会影响坐标的可信程度。

## 形状分析与降维

### Procrustes 分析与 PCA

Procrustes 分析（Procrustes analysis）在去除平移、旋转（有时也去除尺度）之后比较形状，是结构叠合思想在一组结构上的推广：广义 Procrustes 分析反复把所有结构对齐到当前的平均形状，再更新平均形状，直至收敛。

对齐之后就可以做 PCA。设原始轨迹矩阵为 $X_{\mathrm{raw}}\in\mathbb R^{N\times d}$，每行是一帧，$N$ 为帧数，$d$ 为特征维数（例如 $n$ 个 $C_\alpha$ 原子的坐标时 $d=3n$）。先去除刚体运动，再按列中心化：

$$
X=X_{\mathrm{raw}}-\mathbf1\bar x^\top,\qquad
C=\frac{1}{N-1}X^\top X
$$

这里 $\bar x\in\mathbb R^d$ 是各列均值组成的向量，$\mathbf 1$ 是长度为 $N$ 的全 1 列向量，所以 $\mathbf1\bar x^\top$ 是每行都等于 $\bar x^\top$ 的矩阵；$C$ 是 $d\times d$ 的样本协方差矩阵。$C$ 的特征向量就是主成分方向，对应特征值是该方向上的方差。

几个常见坑：

- **必须先中心化**：若未中心化，$X^\top X/(N-1)$ 是二阶矩而非样本协方差，第一主成分会大致指向均值方向，失去“运动模式”的含义。
- **计算方式**：显式构造协方差约需 $O(Nd^2)$，特征分解需 $O(d^3)$；$d$ 很大时，通常直接对中心化轨迹做 economy SVD 或 randomized SVD，$X=U\Sigma V^\top$ 的右奇异向量即主成分，$\Sigma^2/(N-1)$ 即方差。
- **方差大不等于慢或重要**：最大方差方向不一定是最慢的动力学过程，也不一定具有功能意义。例如一个松散末端的大幅抖动可能主导 PC1，而真正决定功能的口袋开合却幅度很小。

### tICA：寻找慢过程

为了直接找“慢”的方向，tICA 利用时间信息。对中心化后的特征 $x_t$（第 $t$ 帧的特征向量），定义瞬时协方差和时间滞后协方差：

$$
C_0=\mathbb E[x_tx_t^\top],\qquad
C_\tau=\mathbb E[x_tx_{t+\tau}^\top]
$$

然后求解广义特征值问题 $C_\tau v_k=\lambda_k C_0v_k$。直觉是：投影到方向 $v$ 上后，$v^\top C_\tau v / v^\top C_0 v$ 正是该投影在滞后 $\tau$ 下的自相关系数；自相关越接近 1，说明经过 $\tau$ 之后它还“记得”原来的值，即变化越慢。所以 tICA 最大化的是归一化的时间滞后自相关（实际估计时常把 $C_\tau$ 对称化为 $(C_\tau+C_\tau^\top)/2$，以保证特征值为实数）。

需要注意两点：名称中的 independent 并不保证各成分统计独立，它只是沿用了盲源分离文献中的叫法；若把多条非平衡的短轨迹拼接起来估计，还需要重加权（reweighting），否则均值和相关矩阵会偏向初始构象的分布。

### Markov 状态模型

MSM 先把构象（通常在 tICA 空间中）聚类成离散状态，再统计状态之间的跳转。采用行随机（row-stochastic）约定时：

$$
T_{ij}(\tau)=P(s_{t+\tau}=j\mid s_t=i),\qquad
\sum_jT_{ij}=1
$$

其中 $s_t$ 是第 $t$ 时刻所处的状态，$T(\tau)$ 的第 $i$ 行是从状态 $i$ 出发、经过 $\tau$ 后落在各状态的概率分布，所以每行之和为 1。

平稳分布（stationary distribution）$\pi$ 满足 $\pi^\top T=\pi^\top$，即 $T$ 对应特征值 1 的左特征向量；$\pi_i$ 给出状态 $i$ 的平衡概率，结合 $\Delta G=-k_BT\ln(P_B/P_A)$ 就能得到状态间的自由能差。

$T$ 的其余特征值 $|\lambda_i|<1$ 描述弛豫过程：经过 $k$ 个 lag，对应模式衰减为 $\lambda_i^k$。若 $\lambda_i(\tau)$ 是非平凡特征值，令 $\lambda_i^{t/\tau}=e^{-t/t_i}$，即得隐含时间尺度（implied timescale）：

$$
t_i(\tau)=-\frac{\tau}{\ln|\lambda_i(\tau)|}
$$

例如 $\tau=10$ ns 时某个特征值为 $0.9$，则 $t_i=-10/\ln0.9\approx94.9$ ns，表示存在一个约 95 ns 的慢过程。

如何判断模型可信？如果 Markov 假设成立，$t_i(\tau)$ 应当不随 $\tau$ 变化，所以常画 implied timescale 随 lag time 的曲线，看是否出现平台（plateau）。但平台只是必要诊断；Chapman–Kolmogorov 检验还应比较用 $k\tau$ 直接独立估计的 $T(k\tau)$ 与模型预测的 $T(\tau)^k$ 是否一致。状态划分过粗（把动力学上不同的构象合成一个状态）、采样不足或轨迹非平衡，都会破坏 Markov 近似。

## 结构聚类与密度方法

### 距离定义决定簇

结构聚类的结果首先依赖距离怎么定义：主链 RMSD、TM-score、扭转角距离或接触距离会产生不同的簇。RMSD 对局部大偏差敏感、依赖对齐；TM-score 对长度归一化，更看重全局拓扑；扭转角距离不需要对齐；接触距离关注哪些残基相互靠近。选择时应与要回答的问题相符。

算法方面，DBSCAN 基于密度，能识别非球形簇并把稀疏点标为噪声，适合 MD 中“几个密集盆地 + 大量过渡帧”的情形；层次聚类给出树状图，便于观察多尺度关系。聚类数与阈值需要做稳定性分析（例如改变参数或重采样后簇是否保持），不应只看一张图下结论。

### 嵌套采样与 bootstrap

三维数据还经常具有嵌套采样（nested sampling）结构：同一患者的多个细胞、同一蛋白的多帧、同一结构的多条链。把所有观测都当作独立样本，会低估方差、夸大显著性。正确做法是让 bootstrap 在真正独立的最高层级上重采样，例如以整条 trajectory replicate 为单位，而不是逐帧重采样；逐帧重采样会把高度相关的帧当作独立信息，得到过窄的置信区间。

### 周期变量与对称性

二面角是周期变量。直接对角度值做 Euclidean PCA，会在 $-\pi/\pi$ 边界制造假距离：$179^\circ$ 和 $-179^\circ$ 实际只差 $2^\circ$，数值上却差 $358^\circ$。常用做法是把每个角 $\theta$ 改写为 $(\sin\theta,\cos\theta)$ 两个特征，嵌入单位圆后欧氏距离就与真实角度差单调相关。

具有对称侧链、同源亚基或可交换配体时，还要先在对称等价的排列（symmetry-equivalent permutation）下取最小距离，否则同一个构象可能因为原子编号不同被错误地分成多个簇。

> [!NOTE]+ 小结
>
> - 先处理刚体运动、周期性和对称性，再做统计；否则坐标系差异会被当成生物差异。
> - PCA 找方差大的方向（需先中心化），tICA 找慢方向，MSM 估计离散状态间的转移概率与时间尺度。
> - MSM 的可信度要靠 implied timescale 平台与 Chapman–Kolmogorov 检验共同确认。
> - 聚类结果取决于距离定义与参数，需要稳定性分析；bootstrap 应在独立的最高层级上进行。

<br>

# 知识表示：本体、受控术语与数据整合（Knowledge Representation）

同一个基因在不同数据库里可能叫 TP53、p53、P53_HUMAN 或“tumor protein p53”；同一个功能可能被写成“apoptosis”“programmed cell death”或“细胞凋亡”。如果计算机只做字符串比较，这些都会被当成不同的东西；反过来，同一个符号（如 gene symbol 的旧名）也可能指向不同的基因。知识表示（knowledge representation）要解决的问题是：怎样让基因、蛋白质、疾病和功能术语在不同数据库、不同研究者之间保持一致而明确的含义，并且能被程序自动推理。

核心想法一句话：**给每个概念一个稳定的标识符、一条定义，以及它与其他概念之间带语义的关系。** 这样的结构称为本体（ontology）。在此之上，注释（annotation）把具体的基因产物（gene product）连接到本体中的术语（term），并记录证据来源。本章为后面的转录组结果解释、HPO 表型匹配、网络模块注释以及跨数据库 join 提供语义基础。

## 知识表示要解决的三类计算任务

在生物信息学中，本体和注释主要服务于三类计算：

- **语义规范化与推理**：把各种写法映射到同一个概念，并沿允许的关系推出隐含的事实，例如沿 Gene Ontology 的允许关系把注释传播到上位术语；
- **富集分析**：输入候选基因集、背景全集和本体版本，输出每个 term 的命中数、富集倍数（fold enrichment）、$p$ 值与 FDR，用来概括一组基因“共同在做什么”；
- **数据整合**：处理 gene、transcript、protein 和 phenotype 之间的一对多映射以及相互冲突的证据。

有两组概念需要从一开始就分清。其一，term 不是 annotation：term 是本体中定义的概念，annotation 是“某个基因产物具有这个概念”的断言。其二，pathway 不是普通的 gene set：把 pathway 降成一个无序的成员集合之后可以做富集分析，但会丢失方向、反应和拓扑信息。

## 受控术语与本体

### 为什么需要受控术语

自由文本存在同义词、缩写和粒度不一致的问题：一篇文献写“DNA repair”，另一篇写“base excision repair”，后者是前者的一种，但字符串毫无相似之处。本体用唯一标识符、定义和关系来规范语义，使得“base excision repair 属于 DNA repair”这件事可以被程序读出来。

### Gene Ontology 的三个分支

Gene Ontology（GO）是生物学中使用最广的本体，包含三个相互独立的分支：

- **Molecular Function（分子功能）**：基因产物在分子层面的活性，如“protein kinase activity”；
- **Biological Process（生物过程）**：由多个分子活动组成的更大过程，如“DNA repair”“cell cycle”；
- **Cellular Component（细胞组分）**：基因产物发挥作用的位置，如“nucleus”“mitochondrial inner membrane”。

### GO 是有向无环图

GO 是有向无环图（directed acyclic graph，DAG）而不是简单的树：一个 term 可以有多个 parent。例如“mitochondrial DNA repair”既是一种“DNA repair”，又与线粒体 DNA 代谢相关的上位概念相连。图中的边带有不同语义，`is_a` 表示子类关系（A 是 B 的一种），`part_of` 表示部分关系（A 是 B 的组成部分），此外还有 `regulates` 等关系；不同关系在推理时的传递规则不同。

基因注释还附带证据代码（evidence code），记录这条注释的证据来源，例如实验（如 IDA、IMP）、系统发育推断（如 IBA）、计算推断（如 ISS）、作者陈述（TAS、NAS）或自动注释（IEA）。需要注意：evidence code 表示的是证据**类型**，不是适用于所有场景的统一质量等级。IEA 注释虽未经人工审核，但在某些用途上覆盖面和可靠性都足够；实验注释也可能来自单一条件下的结果。

### 本体的构成：class、relation、axiom 与 identifier

从形式上看，本体由类（class）、关系（relation）、公理（axiom）与标识符（identifier）构成。由于是 DAG，一个具体 term 可能同时继承多个上位概念的性质。必须区分两样东西：

- **ontology term** 定义概念本身，例如 GO:0006281 “DNA repair”的定义与它在图中的位置；
- **数据库 annotation** 断言“某个 gene product 与该概念之间存在带证据的关系”，例如“人 TP53 参与 DNA repair，证据为 IMP，来源文献为某 PMID”。

### 从受控词表到本体：表达能力的层次

Stevens、Goble 与 Bechhofer 在讨论生物信息学中的本体表示时，沿用了 Gruber 的定义，把 ontology 解释为“对概念化的显式规范”（explicit specification of a conceptualization），并结合生物信息学说明其用途。可以按表达能力从弱到强排列：

- **受控词表（controlled vocabulary）**：主要统一名称，规定“用哪个词”；
- **分类体系（taxonomy）**：在词表基础上增加层级（上下位）关系；
- **本体**：进一步给出关系约束和可供推理机（reasoner）使用的公理。

建模粒度取决于用途。用于检索的轻量本体未必足以支持自动一致性检查；而用于严格推理的表示则需要声明关系的定义域（domain）、值域（range）、类之间的互斥（disjointness）和等价（equivalence）关系，例如在 OWL 中声明两个类互斥（disjoint）之后，若某个个体或子类被同时归入这两个类，推理机就能自动报告不一致。需要注意，互斥约束的是类的成员资格，并不禁止一个蛋白同时具有两条不同的定位注释（许多蛋白本来就在核与胞质之间穿梭）。

### 注释的附加信息与 NOT 限定词

一条 annotation 除 term 和 evidence code 外，还可能带限定词（qualifier）、亚型（isoform）、文献来源（reference）和注释扩展（annotation extension，例如说明活性发生在哪种细胞或以哪个分子为底物）。

其中 `NOT` 限定词最容易误用。`NOT` 表示有证据**明确否定**某项 GO 关联，例如实验显示该蛋白没有某种酶活性；它不是“没有显著结果”，也不是“缺少注释”。传播规则方向相反：正注释通常沿 GO 允许的关系向更一般的祖先（ancestor）传播；`NOT` 注释则约束该 term 及其更具体的后代（descendant），因为“没有 kinase activity”意味着也没有任何一种更具体的 kinase activity。富集分析应排除 `NOT` 注释，不能把被明确否定的功能当作阳性计数。

## Pathway 与 gene set 的区别

富集分析的输入单位是 gene set，但 gene set 的来源多种多样，含义并不相同。Gene set 只是基因标识符的集合，本身不编码顺序、方向或机制。Pathway（如 KEGG、Reactome 中的通路）通常还包含反应、调控方向、复合物、细胞区室和上下游关系。把 pathway 降成 gene set 之后才能使用许多富集算法，但边的方向、拓扑以及基因之间的逻辑关系（如“A 激活 B、B 抑制 C”）也随之丢失。

GO term 对应的注释基因集、人工整理的 pathway、共表达模块（module）和实验得到的表达特征（signature）都可以作为 gene set 使用，但它们的生物语义不同：前两者来自知识库，后两者来自数据，解释富集结果时应说明用的是哪一类。

## 注释传播与富集分析

### 注释传播

若一个基因被注释到某个具体 term，通常也隐含它属于由 GO 关系规则允许的某些祖先 term。例如注释到“base excision repair”的基因，沿 `is_a` 也应计入“DNA repair”。但不能把图上所有可达的祖先都机械地视为自动成立：例如 `regulates` 边不意味着“调控某过程的基因本身就参与该过程”，传播时应只沿 `is_a`、`part_of` 等允许传递的关系。

计算富集时还有两条基本要求：必须使用与实验实际可检测基因一致的背景全集（background universe）；并且要处理 GO 层级结构造成的检验间相关（各 term 共享基因，检验并不独立，具体处理方法见下文“超几何 null 的含义与常见偏差”）。

### 超几何检验

最常用的过表示分析（over-representation analysis，ORA）问的是：候选集中带有某个 term 的基因数，是否比随机抽取时预期的多？用超几何分布计算右尾概率：

$$
P(X\ge k)=\sum_{i=k}^{\min(K,n)}
\frac{\binom{K}{i}\binom{N-K}{n-i}}{\binom{N}{n}}
$$

符号含义：$N$ 是背景基因数，$K$ 是背景中带有该 term 的基因数，$n$ 是候选集大小，$k$ 是候选集中带该 term 的命中数。分子 $\binom{K}{i}\binom{N-K}{n-i}$ 数的是“从带 term 的 $K$ 个里抽 $i$ 个、从不带 term 的 $N-K$ 个里抽其余 $n-i$ 个”的方式数，分母是从 $N$ 个里任取 $n$ 个的总方式数；对 $i\ge k$ 求和即得“至少命中 $k$ 个”的概率。这等价于单侧 Fisher 精确检验。

几条实现上的要求：候选集必须先与背景取交集，并在同一 ID 空间去重；$K,k$ 应按相同的 GO 版本、物种、evidence/qualifier 过滤规则与传播规则计算。否则候选集中可能出现背景里根本没有的基因，概率模型就不成立了。

> [!EXAMPLE]+ 小例子：超几何检验与富集倍数
>
> 设背景有 $N=10000$ 个可检测基因，其中 $K=100$ 个带有某个 GO term；候选集 $n=200$ 个基因中有 $k=8$ 个命中。随机抽取时期望命中数为 $nK/N=2$。代入上式得 $P(X\ge8)\approx8.6\times10^{-4}$，富集倍数为
>
> $$
> \frac{8/200}{100/10000}=\frac{0.04}{0.01}=4.
> $$
>
> 若该分析同时检验了数千个 term，这个 $p$ 值还需要经过多重检验校正（见转录分析中的多重检验一节）才能下结论。

### 超几何 null 的含义与常见偏差

超几何检验的零假设是：候选集是从固定背景中随机抽取的 $n$ 个基因。由此产生两个常见偏差：

- **背景选错**：若背景误用全基因组，而实验平台实际只能检测其中一部分（例如芯片只有部分基因的 probe，或 RNA-seq 中低表达基因被过滤），那么在组织中表达的基因相关的 term 都会显得“富集”，结果有偏。
- **term 之间相关**：GO term 因为共享基因以及祖先传播而高度相关，父 term 显著时子 term 往往也显著。BH 方法在独立或特定的正相关（PRDS）条件下仍能控制 FDR，但显著的 term 不能解释成彼此独立的生物过程。若需要在任意依赖下的保守保证，可以考虑 Benjamini–Yekutieli 方法、置换检验或利用 GO 结构的层级检验（如 topGO 的 elim/weight 算法）。

### 富集倍数

富集倍数（fold enrichment，FE）定义为候选集中命中比例与背景比例之比：

$$
\operatorname{FE}=\frac{k/n}{K/N}
$$

FE 是效应量，$p$ 值是证据强度，二者都要看。小 term 可能有很高的 FE 却只包含两三个基因（例如 $K=3$、$k=2$ 时 FE 可达几十），因此报告时还要给出命中基因数、FDR、term 大小和具体命中的基因。

## 推理、版本与结果可复现

GO 本身在持续更新：term 会新增、合并、废弃（obsolete）或重新定义，注释也每月变化。同一组基因在两年前和今天做富集，结果可能明显不同。因此分析应固定 ontology release、annotation release、物种、evidence-code 过滤规则与 qualifier 规则。

一份可复现的富集结果至少应保存：background、ID 映射前后的基因数量、未映射（unmapped）和重复（duplicate）的 ID、term 大小、命中基因、原始 $p$ 值、校正方法和 GO 版本。

此外要警惕上游筛选带来的偏差。若候选基因是由表达阈值或模型筛选产生的，简单超几何 null 未必反映上游的选择过程：例如高表达、长基因在 RNA-seq 中更容易被检测为差异表达，于是与它们相关的 term 会系统性地“富集”。可以使用按表达量匹配的置换、基于排序的 GSEA 或模拟来做敏感性分析。最后，显著的 GO term 只是当前 gene set 的统计摘要，不等于该过程被因果性地激活。

## 数据整合

### 数据库 join 不是字符串 join

把两个数据库按基因连接起来，看似只是一个 `JOIN`，实际上陷阱很多。gene symbol 会变更（HGNC 会重命名基因），也可能有歧义（同一 symbol 历史上指向不同基因，或被 Excel 自动转换成日期）；同一个基因可以有多个 transcript 和 protein isoform；基因组坐标依赖参考基因组版本（genome build，如 GRCh37 与 GRCh38）。可靠的整合应保留以下信息：

- 命名空间（namespace）与稳定标识符（stable identifier），例如 Ensembl ID、Entrez Gene ID、UniProt accession；
- 数据库版本和映射日期；
- 一对多（one-to-many）映射，而不是随意取第一个；
- 数据来源（provenance）与证据；
- 缺失的机制（为什么某条记录没有值）；
- 冲突处理规则。

### 三个不同的子问题

数据整合可以拆成三个性质不同的问题：

1. **模式对齐（schema alignment）**：两个数据源的字段、单位和粒度如何对应，例如一个按基因记录、另一个按转录本记录；
2. **实体解析（entity resolution）**：哪些记录指向同一个实体，例如不同数据库中的同一蛋白；
3. **证据调和（evidence reconciliation）**：相互冲突的断言如何保留、评级或汇总，例如一个数据库说某蛋白在核内，另一个说在胞质。

### 多模态整合策略与缺失偏差

当需要把序列、结构、表达、相互作用等多种模态合起来建模时，有三种常见策略：

- **早期整合（early integration）**：把不同模态的特征拼接成一个大向量后统一建模，简单但维度高、尺度不一；
- **晚期整合（late integration）**：对每个模态分别建模，再组合各自的预测；
- **中间整合（intermediate integration）**：学习一个各模态共享的潜在表示（shared latent representation）。

缺失模态通常不是随机缺失。例如研究热门的蛋白更可能拥有结构和功能实验数据，直接删除有缺失的样本，会让模型只在“被充分研究的蛋白”上训练和评估，从而引入选择偏差（selection bias），高估在冷门蛋白上的表现。

> [!NOTE]+ 小结
>
> - 本体用稳定 ID、定义和带语义的关系组织概念；GO 分为 MF、BP、CC 三支，是 DAG 而非树。
> - term 是概念，annotation 是带证据的断言；evidence code 表示证据类型而非统一质量等级；`NOT` 是明确否定，富集时要排除。
> - 超几何检验依赖正确的背景全集与一致的 ID、版本、过滤规则；同时报告 FE、命中数和 FDR。
> - 数据整合要保留稳定 ID、版本、一对多映射与来源，并注意缺失不是随机的。

<br>

# 转录分析：从表达测量到差异表达与功能解释（Transcriptional Analysis）

细胞对药物、疾病或发育信号的响应，很大一部分体现在哪些基因的 RNA 变多、哪些变少。转录分析（transcriptional analysis）研究不同样本中 RNA 丰度如何变化。数据通常整理成“基因（或 transcript）× 样本”的矩阵，但矩阵中的数字并不直接等于细胞内真实的分子数：RT-qPCR、微阵列（microarray）和 RNA-seq 通过不同的实验链获得信号，分别受扩增效率、杂交、测序深度和组成偏差的影响。因此，在问“哪些基因变了”之前，必须先理解每种测量的数字是怎么来的。

核心想法一句话：**用一个符合测量特性的统计模型（对 RNA-seq 是负二项 GLM），在明确的实验设计和对比下估计每个基因的效应量和不确定性，再对成千上万个检验做多重校正。** 本章从表达测量进入统计推断，再把差异基因或全基因排序交给 GO 富集、GSEA、聚类和网络分析，与上一章的本体和富集分析直接衔接。

## 输入、输出与解释边界

差异表达分析的输入是原始 count 矩阵、样本元数据（metadata）和设计矩阵（design matrix）。输出是在指定对比（contrast）下每个基因的 log fold change、标准误、$p$ 值、校正后 $p$ 值与置信区间。模型中各部分分工如下：

- **负二项模型**处理 RNA-seq 中方差随均值变化的问题；
- **size factor** 处理各样本测序尺度不同的问题；
- **batch、配对和交互项**由实验设计编码进设计矩阵。

两条解释边界：TPM 适合描述样本内部的相对组成，但不是 DESeq2 所需的原始 count；统计显著性不能替代效应量与实验重复。

## RT-PCR 与 qPCR

### 原理

逆转录 PCR（reverse-transcription PCR，RT-PCR）先以 RNA 为模板，用逆转录酶合成 cDNA，再用基因特异性引物（gene-specific primer）扩增目标片段。终点 RT-PCR（end-point RT-PCR）主要在反应结束后观察产物（例如跑胶），只能粗略判断有无。实时定量 PCR（real-time quantitative PCR，qPCR）在扩增过程中每个循环记录荧光（fluorescence），并以荧光首次越过阈值时的循环数，即阈值循环（threshold cycle，Ct，也记作 Cq），来近似反映起始模板量：起始模板越多，越早越过阈值，Ct 越小。

理想扩增效率为 2（每循环产物加倍）时，Ct 相差一个循环约对应起始量相差两倍。实际效率常略低于 2，必须用标准曲线验证引物特异性（primer specificity）与扩增效率（amplification efficiency）。

### $\Delta\Delta Ct$ 相对定量

相对定量常用参考基因（reference gene，如 GAPDH、ACTB）来校正上样量差异：

$$
\Delta Ct=Ct_{\mathrm{target}}-Ct_{\mathrm{reference}}
$$

比较两个条件时再取差：$\Delta\Delta Ct=\Delta Ct_{\text{处理}}-\Delta Ct_{\text{对照}}$，fold change 为 $2^{-\Delta\Delta Ct}$。负号的来源是：Ct 越小表示表达越高。

> [!EXAMPLE]+ 小例子：$\Delta\Delta Ct$ 计算
>
> 处理组目标基因 Ct 为 22、参考基因为 18，$\Delta Ct=4$；对照组目标基因 Ct 为 25、参考基因为 18，$\Delta Ct=7$。于是 $\Delta\Delta Ct=4-7=-3$，fold change $=2^{3}=8$，即处理后目标基因约上调 8 倍。

该公式依赖两个前提：目标基因与参考基因的扩增效率接近（否则应使用 Pfaffl 等效率校正方法），并且参考基因在各条件间表达稳定。RT-qPCR 通量低，但灵敏、便宜，适合验证预先选定的少数基因。它与 RNA-seq 是相互独立的测量链，常用来验证 RNA-seq 结果的方向和大致幅度，但不应期待二者的数值尺度直接相同。

## 微阵列

### 原理与处理流程

微阵列（microarray）在芯片上固定大量已知序列的探针（probe），让荧光标记的样本 cDNA/cRNA 与之杂交（probe hybridization），用荧光强度测量相对丰度。标准处理流程包括背景校正、归一化（normalization）、探针汇总（probe summarization）、批次检查和差异表达。交叉杂交（相似序列也会结合同一探针）与动态范围（低端被背景淹没、高端信号饱和）是它的固有限制，这也是 RNA-seq 逐渐取代芯片的主要原因。

### 双色芯片与单通道芯片

- **双色芯片（two-color array）** 把两个样本分别用两种荧光染料（如 Cy3、Cy5）标记后同时杂交，比较两个荧光通道，需要处理染料偏差（dye bias），常用染料互换（dye swap）设计抵消。
- **单通道芯片**（如 Affymetrix）每张芯片只测一个样本，通常用多个 probe 组成一个探针集（probe set）代表一个基因。

RMA（Robust Multi-array Average）是单通道芯片最常用的预处理方法，包含三步：背景校正、分位数归一化（quantile normalization）和 median-polish 汇总。分位数归一化把每个样本的强度排序后替换为各样本同秩次值的平均，从而强制各样本的经验分布完全相同。它的隐含假设是“大部分基因不变、整体分布应相同”；若处理确实引起全局表达分布的变化（例如转录整体上调），该假设可能不成立，归一化会把真实信号抹掉。

### 重分析时的注释问题

探针序列可能映射到多个 transcript，旧芯片的注释也会随着基因组注释更新而过时。重分析公开芯片数据时，应更新 probe 到基因的映射（例如使用基于新注释重新定义的 probe set），同时保存原平台和所用注释的版本。

## RNA-seq

### 从样本到 count 矩阵

RNA-seq 的数据从测序读段（reads）计数出发。先看建库层面的实验链：RNA 提取 → 富集 mRNA（poly-A 选择）或去除 rRNA → 逆转录 → cDNA 片段化与长度筛选（fragmentation/size selection）→ 接头连接（adapter ligation）→ 测序。双端测序（paired-end read）同时读取片段两端，提供片段两端位置和插入长度（insert length）约束，有助于比对、isoform 分配和融合基因检测（fusion detection）。

再看计算层面，从 FASTQ 到 count 矩阵通常包括质量控制（quality control）、去接头（adapter trimming）、比对（alignment）或伪比对（pseudoalignment）、特征分配（feature assignment）与样本级 QC。传统流程（如 STAR + featureCounts）把 reads 定位到基因组或转录组上再计数；Salmon、Kallisto 一类方法则通过 $k$-mer 和等价类（equivalence class）配合概率模型直接估计 transcript 丰度，速度快得多。需要记住：最终每个基因的 count 是 read/fragment 分配的结果，而不是仪器直接读出的“基因分子数”。

### 为什么用负二项分布

原始 count 受测序深度和样本组成的影响。TPM 便于在同一样本内比较相对丰度，但差异表达通常直接对 count 建模。如果每个 read 独立地来自某基因，count 应服从 Poisson 分布，方差等于均值；但生物学重复之间还有真实的个体差异，实际方差明显大于均值（过度离散，overdispersion）。因此常用负二项分布（negative binomial，NB）：

$$
Y_{gi}\sim\operatorname{NB}(\mu_{gi},\alpha_g)
$$

其中 $Y_{gi}$ 是基因 $g$ 在样本 $i$ 中的 count，$\mu_{gi}$ 是其均值，由文库大小和设计矩阵决定；$\alpha_g$ 是基因 $g$ 的离散度（dispersion），表示超出 Poisson 的那部分变异。

在 DESeq2 的常用参数化下，

$$
\operatorname{Var}(Y_{gi})=\mu_{gi}+\alpha_g\mu_{gi}^2
$$

第一项 $\mu_{gi}$ 是 Poisson 的抽样噪声，第二项 $\alpha_g\mu_{gi}^2$ 是生物学变异；方差随均值非线性增加，Poisson 是 $\alpha_g=0$ 的特殊情形。直觉上，$\sqrt{\alpha_g}$ 近似等于生物学变异系数（biological coefficient of variation）：高表达基因的抽样噪声相对很小，方差主要由 $\alpha_g$ 决定。

### 负二项 GLM 与对比

负二项广义线性模型（GLM）常使用对数链接（log link）：

$$
\log\mu_{gi}=\log s_i+\mathbf x_i^\top\boldsymbol\beta_g
$$

$s_i$ 是样本特定的 size factor，作为偏移量（offset）进入模型，系数固定为 1，不被估计；$\mathbf x_i$ 是设计矩阵的第 $i$ 行，编码 condition、batch、pairing 等变量；$\boldsymbol\beta_g$ 是基因 $g$ 的系数向量。由于是对数链接，系数的含义是对数尺度上的倍数变化：例如 condition 系数为 1（以 $\log_2$ 计）表示表达加倍。

对比（contrast）明确了真正要检验的是哪个系数或系数组合，例如“处理 vs 对照”还是“处理在突变体中的效应减去在野生型中的效应”。没有指定 contrast 的“差异表达”不是一个完整的统计问题。

### 文库大小与组成偏差：size factor

为什么不直接用总 read 数来归一化？因为总 read 数不能解决组成偏差（composition bias）：少数极高表达的基因可能占据大量 reads，使其他基因的相对 count 普遍下降，看起来像“全体下调”。

DESeq2 默认的中位数比值法（median-of-ratios）分三步：

1. 对每个基因，计算它在所有样本中的几何均值，作为“伪参考样本”（pseudo-reference）；任一样本中为 0 的基因几何均值为 0，在这一步被跳过；
2. 对每个样本，计算每个基因的 count 与伪参考之比；
3. 取这些比值的中位数作为该样本的 size factor $s_i$。

> [!EXAMPLE]+ 小例子：中位数比值法为什么稳健
>
> 两个样本、四个基因，count 分别为 A：$(10,100,50,10)$，B：$(20,200,100,1000)$。前三个基因 B 恰好是 A 的两倍（测序深度差两倍），第四个基因在 B 中真实大幅上调。各基因几何均值为 $\sqrt{200}\approx14.1$、$\sqrt{20000}\approx141$、$\sqrt{5000}\approx70.7$、$\sqrt{10000}=100$。A 的比值为 $(0.707,0.707,0.707,0.1)$，中位数 $0.707$；B 的比值为 $(1.414,1.414,1.414,10)$，中位数 $1.414$。size factor 之比为 2，正确反映了测序深度。若改用总 count，A 为 170、B 为 1320，比例约 7.8，会把前三个基因都错判为在 B 中下调。

这种方法与 edgeR 的 TMM（trimmed mean of M-values）等方法一样，都依赖“大多数基因不发生强烈的同方向变化”这一稳健性假设。当存在全局 RNA 输出改变（例如 MYC 过表达导致几乎所有基因转录增加）时，该假设失效，通常需要 spike-in（如 ERCC 外加 RNA）或其他外部尺度。

### TPM 与原始 count

TPM（transcripts per million）先把 count 除以转录本长度，再把每个样本的总和缩放到 $10^6$，适合描述样本内部的相对组成，但不应直接作为 DESeq2 的表达矩阵。DESeq2 的标准输入是未预先按文库大小归一化的非负整数 count；原因在于 TPM 已经做过长度和总量归一化，失去了原始 count 的采样尺度，模型就无法正确区分“10 个 read 中的 5 个”和“10000 个 read 中的 5000 个”这两种不确定性完全不同的情形。

对 Salmon/Kallisto 的结果，应通过 tximport 汇总为基因水平的估计 count，并用 offset 处理有效转录本长度（effective transcript length）的差异，而不是把 TPM 取整后输入。

### 小样本下的离散度收缩

每组只有 3 个左右重复时，逐基因估计的 dispersion 很不稳定：偶然看起来很一致的基因会得到过小的 dispersion，从而产生假阳性。经验 Bayes 方法（DESeq2、edgeR 都使用）先拟合全体基因的“均值–离散度”趋势（mean–dispersion trend），再把每个基因的估计向该趋势收缩：数据信息多时更相信基因自身，信息少时更依赖趋势。这是在基因特异的变异和估计方差之间的折中，也就是“借用其他基因的信息”。limma-voom 对 log-CPM 数据做的方差收缩基于同样的思想。

## 多重检验

### 问题：成千上万次检验

对两万个基因各做一次检验，即使所有基因都没有变化，按 $p<0.05$ 也会期望出现约 1000 个假阳性。因此必须做多重检验校正。控制族错误率（FWER，至少一个假阳性的概率）的 Bonferroni 方法在基因组规模下过于保守；更常用的是 Benjamini–Hochberg（BH）方法，它控制错误发现率（false discovery rate，FDR）。

### BH 程序

把 $m$ 个 $p$ 值从小到大排序为 $p_{(1)}\le\dots\le p_{(m)}$，给定目标 FDR 水平 $q$，寻找最大的 $i$ 满足

$$
p_{(i)}\le\frac{i}{m}q
$$

然后拒绝排名前 $i$ 的全部假设（即使其中某些更小的 rank 没有单独满足不等式，也一并拒绝）。直观上，阈值 $iq/m$ 随 rank 线性升高：排名越靠后，允许的 $p$ 值越大。

### FDR 的确切含义

设 $V$ 为错误拒绝数、$R$ 为总拒绝数，则

$$
\mathrm{FDR}=E\left[\frac{V}{\max(R,1)}\right]
$$

也可简写为 $\mathrm{FDR}=E[V/\max(R,1)]$。分母取 $\max(R,1)$ 是为了在没有任何拒绝时定义为 0。需要准确理解：FDR 是在重复应用整个检验程序时，错误发现比例（false-discovery proportion）的期望；它不是某个特定基因为 null 的后验概率，也不保证当前这次结果中恰好有 $qR$ 个假阳性。

### BH adjusted p-value

为了不必为每个 $q$ 重新跑一遍程序，通常报告 BH 校正后的 $p$ 值：

$$
\tilde p_{(i)}
=\min_{j\ge i}\left(1,\frac{m}{j}p_{(j)}\right)
$$

即先算 $m\,p_{(j)}/j$，再从最大 rank 向前取累积最小值（cumulative minimum），并截断在 1。取累积最小值可以保证校正后的 $p$ 值随 rank 单调不减。这样，“$\tilde p_{(i)}\le q$”与“在 BH 水平 $q$ 下被拒绝”完全等价。

> [!EXAMPLE]+ 小例子：BH 手算
>
> $m=8$ 个排序后的 $p$ 值为 $0.001,0.008,0.039,0.041,0.042,0.06,0.074,0.205$，取 $q=0.05$。阈值 $iq/m$ 依次为 $0.00625,0.0125,0.01875,0.025,0.03125,0.0375,0.04375,0.05$。只有前两个 $p$ 值低于对应阈值，所以拒绝 2 个假设。
>
> 计算校正值：$m\,p_{(j)}/j$ 依次为 $0.008,0.032,0.104,0.082,0.0672,0.08,0.0846,0.205$；从后向前取累积最小值得到 $\tilde p=0.008,0.032,0.0672,0.0672,0.0672,0.08,0.0846,0.205$。小于 $0.05$ 的恰好也是前两个，与上面一致。注意第 3 个原始值 $0.104$ 被后面更小的 $0.0672$ 取代，这就是单调化的作用。

需要避免两种破坏错误率控制的做法：先观察结果、任意筛选基因，再只对剩下的检验做校正；或使用与 $p$ 值相关的统计量做过滤。独立过滤（independent filtering）是允许的，但所用的统计量（例如 DESeq2 中的 base mean）必须在 null 下与 $p$ 值近似独立。

### 实现

```python
def benjamini_hochberg(pvalues):
    """返回与输入顺序一致的 BH adjusted p-values。"""
    if any((not 0.0 <= p <= 1.0) for p in pvalues):
        raise ValueError("p-values must be finite values in [0, 1]")
    m = len(pvalues)
    if m == 0:
        return []

    # 保存原始索引，以便校正后恢复调用者的输入顺序。
    order = sorted(range(m), key=pvalues.__getitem__)
    adjusted = [0.0] * m
    running_min = 1.0
    # 从最大 p-value 向前取 cumulative minimum，保证单调性。
    for rank0 in range(m - 1, -1, -1):
        idx = order[rank0]
        rank = rank0 + 1
        running_min = min(running_min, pvalues[idx] * m / rank)
        adjusted[idx] = min(1.0, running_min)
    return adjusted
```

函数输出的顺序恢复为输入顺序，便于与基因名对应。输入检查中 `0.0 <= p <= 1.0` 对 `NaN` 返回 False，所以 `NaN` 会触发报错；实际使用时，`NaN`（例如 DESeq2 中被独立过滤或含 outlier 的基因）应在调用前单独标记并排除，并在结果中恢复其缺失位置。实现正确性的简单验证是：检查所有 adjusted p-value 位于 $[0,1]$，并且按原始 $p$ 值排序后单调不减；也可以与 R 的 `p.adjust(p, method = "BH")` 或 statsmodels 的 `multipletests(..., method="fdr_bh")` 对比。

### 效应量与证据强度要一起报告

log fold change 是效应量（effect size），adjusted p-value 是证据强度（evidence strength），两者回答不同的问题。大样本会使极小的效应也变得显著（统计上确定但生物学上可能无关紧要），小样本则会使大的效应估计很不稳定（低表达基因常出现夸张的 LFC）。因此两者与置信区间都应报告；火山图（volcano plot）正是把这两个维度同时画出来。

## 基因集富集分析（Gene Set Enrichment Analysis，GSEA）

### 动机：不再依赖阈值

知识表示一章中的过表示分析（ORA）先用阈值把基因分成“入选 / 未入选”两类，再做超几何检验。问题是阈值是任意的：$\mathrm{FDR}<0.05$ 与 $<0.1$ 会得到不同的候选集；而且一个通路中许多基因各自只有轻微但一致的变化时，可能没有一个基因单独过线，ORA 就完全看不到它。GSEA 则保留全部基因的排序，问的是：某个基因集的成员是否集中出现在排序列表的顶端或底端？

### 标准流程

两类表型（如处理 vs 对照）的标准 GSEA 流程如下：

1. 对每个基因计算它与表型的关联统计量。课程例子使用 $\log_2$ fold change，实际也可以用 signal-to-noise ratio、Wald 统计量等；
2. 按统计量从高到低排列全部基因；
3. 对给定基因集 $S$ 沿排序列表从头走到尾：遇到 $S$ 内的基因，按其关联强度加权上升；遇到集合外的基因，按常数下降；
4. 富集分数（enrichment score，ES）取这条累积曲线（running sum）距 0 的最大正偏离或最小负偏离；
5. 置换表型标签（phenotype label），重新计算完整排序与 ES，形成经验零分布（empirical null）；
6. 按基因集大小与零分布把 ES 归一化为 NES（normalized enrichment score），并在多个基因集之间控制 FDR。

### Running sum 的公式

设排序列表共有 $N$ 个基因，$g_j$ 是第 $j$ 位的基因，$r_j$ 是它的关联统计量；$N_H=|S|$ 是基因集大小，$p$ 是权重指数。到第 $i$ 位为止的命中与未命中累计量为

$$
P_{\mathrm{hit}}(i)=
\sum_{\substack{j\le i\\g_j\in S}}
\frac{|r_j|^p}{\sum_{g_k\in S}|r_k|^p},
\qquad
P_{\mathrm{miss}}(i)=
\sum_{\substack{j\le i\\g_j\notin S}}\frac{1}{N-N_H}
$$

两者都是从 0 增加到 1 的累积分布：$P_{\mathrm{hit}}$ 是集合内基因按权重的累积比例，$P_{\mathrm{miss}}$ 是集合外基因的累积比例。若集合成员随机散布，两者大致同步增长，差值在 0 附近；若成员集中在顶端，$P_{\mathrm{hit}}$ 先迅速上升，差值出现大的正峰。正向富集时

$$
ES(S)=\max_i\left(P_{\mathrm{hit}}(i)-P_{\mathrm{miss}}(i)\right)
$$

负向富集（成员集中在底端）则取绝对值更大的那个最小偏离。$p=0$ 时所有命中等权，ES 退化为类似 Kolmogorov–Smirnov 统计量的形式；原始 GSEA 论文推荐 $p=1$，让关联更强的基因贡献更大。在达到 ES 极值之前对分数贡献最大的那些集合成员构成前沿子集（leading-edge subset），它们通常是解释生物学含义时最值得关注的核心基因。

### 实现

下面的函数只计算观察到的 ES 与整条 running-sum 曲线：

```python
def gsea_enrichment_score(ranked, gene_set, weight=1.0):
    """计算一个 gene set 的 weighted running-sum ES。

    ranked 是已按关联统计量从高到低排列的 (gene, score) 序列。
    本函数只计算观察 ES；完整 GSEA 还需 label permutation、NES 和 FDR。
    """
    if weight < 0:
        raise ValueError("weight must be nonnegative")
    genes = [gene for gene, _ in ranked]
    if len(set(genes)) != len(genes):
        raise ValueError("ranked genes must be unique")

    members = set(genes) & set(gene_set)
    if not members or len(members) == len(genes):
        raise ValueError("gene set must hit between 1 and N-1 ranked genes")

    # 命中增量按 |score|^weight 归一化；未命中总下降量为 1。
    hit_norm = sum(
        abs(score) ** weight for gene, score in ranked if gene in members
    )
    if hit_norm == 0:
        raise ValueError("hit weights sum to zero")
    miss_step = 1.0 / (len(genes) - len(members))

    running, path = 0.0, []
    for gene, score in ranked:
        if gene in members:
            running += abs(score) ** weight / hit_norm
        else:
            running -= miss_step
        path.append(running)

    # 标准 ES 取绝对偏离更大的正峰或负谷，并保留其 0-based rank。
    max_index = max(range(len(path)), key=path.__getitem__)
    min_index = min(range(len(path)), key=path.__getitem__)
    peak_index = (
        max_index
        if abs(path[max_index]) >= abs(path[min_index])
        else min_index
    )
    return path[peak_index], path, peak_index
```

代码中 `members` 是基因集与排序列表的交集，对应公式里的 $S$（不在排序列表中的成员不计入 $N_H$）；`hit_norm` 是公式中命中权重的分母，`miss_step` 是 $1/(N-N_H)$。由于命中总增量和未命中总下降量都等于 1，曲线最后回到 0。返回的 `peak_index` 是极值所在的 0-based 位置：正向富集时，排在它之前（含）的成员即 leading-edge subset。

### 置换方式与解释限制

置换表型标签（phenotype-label permutation）保留了基因之间的相关结构，是更合理的 null，但需要足够的样本（每组样本太少时可区分的置换数很少）。样本很少时常改用基因集置换（gene-set permutation，即 per-gene permutation，随机抽取同样大小的基因集），后者对应的是不同的零假设（“这个集合比随机基因集更富集”），忽略了基因间相关，两者不能混为一谈。

解释 GSEA 结果时还要注意以下限制：

- GSEA 偏好协同的同向变化，可能漏掉同一 pathway 中一部分基因上调、另一部分下调的情形；
- 结果受基因集数据库（如 MSigDB 的不同集合）、集合大小范围、基因 ID 映射以及集合的预先定义所限制；
- 显著的基因集只说明其成员集中在排序的端部，并不证明 pathway 拓扑中的反应被激活。

## 表达模式聚类

### 动机：多条件下的共同响应

两条件分析通常逐基因估计效应与显著性。当条件、时间点或患者较多时，逐个比较的组合数迅速增加，更自然的问题变成“哪些基因有相似的表达模式”“样本可以分成几类”。这时可以对表达矩阵做聚类：

- **按行（基因）聚类**：寻找具有共同响应模式的基因模块（gene module），例如在时间序列中一起早期上调的基因；
- **按列（样本）聚类**：寻找样本亚型（subtype）或异常样本，例如肿瘤分子分型。

两者使用同一个矩阵，回答的问题却不同，标准化的方向也不同。按基因做 z-score（每行减均值、除标准差）会保留各基因的相对变化模式，使高表达和低表达基因可以比较形状，但同时移除了基因之间绝对表达量的差异。

### 常用算法

层次聚类（hierarchical clustering）、$k$-means 和自组织映射（self-organizing map，SOM）都可用于表达模式分析。层次聚类给出树状图，常与热图一起展示；$k$-means 需要预先指定簇数；SOM 把高维样本映射到一个保持邻域关系的低维网格上，相邻网格单元代表相似的模式，其结果依赖网格大小、初始化和训练顺序。

实践中，聚类前的预处理往往比算法名称更显著地改变结果：低表达过滤、高变异基因选择（variance selection）、对数变换、批次校正以及距离定义（Euclidean 距离关注绝对差异，$1-$Pearson 相关关注模式形状）都会决定最终的簇。

### 聚类结果的评价

聚类评价分为两个层面：

- **内部验证（internal validation）**：轮廓系数（silhouette）、簇内平方和（within-cluster sum of squares）以及重采样稳定性，只使用数据本身的几何结构；
- **外部验证（external validation）**：与已知亚型比较、做 GO/pathway 富集、在独立队列中复现、做实验功能验证。

共表达的基因可能参与同一过程，因此可以用簇做功能注释迁移（annotation transfer，即“与已知 DNA 修复基因共表达的未知基因可能也与 DNA 修复相关”）。但共同表达也可能来自细胞组成差异（例如样本中免疫细胞比例不同）、批次或共同的上游因素，不能直接推出基因之间相互调控或具有相同的分子功能。

## 实验设计

### 设计决定可识别性

再好的统计方法也无法从设计糟糕的实验中提取信息。几条基本原则：

- 生物学重复（不同个体、不同培养批）不能由技术重复（同一 RNA 测多次）替代，因为差异表达要推断的是生物学变异；
- 批次与条件完全混杂时，任何算法都无法区分二者；
- 配对设计、协变量、样本随机化和预先登记的过滤标准，通常比事后的各种选择更重要。

设计矩阵必须满秩（full rank）。若全部对照样本来自 batch 1、全部处理样本来自 batch 2，则 condition 列与 batch 列完全共线，模型无法分开估计两者的效应。ComBat 或其他批次校正方法也无法恢复实验设计本身从未提供的可识别性；在这种情况下强行校正，要么把真实的处理效应一起去掉，要么把批次效应当成处理效应。

> [!EXAMPLE]+ 例8：配对设计
>
> 同一患者治疗前后各取一个样本，模型 `~ patient + treatment` 用 patient 效应吸收个体基线差异，treatment 系数只反映同一个人治疗前后的变化。若忽略配对，个体差异进入残差并降低检验功效（power）；若把技术重复当作独立患者，则会伪造样本量，得到虚假的显著性。

## 差异表达分析检查表

差异表达分析应从样本级 QC 和设计矩阵开始，而不是直接调用检验函数。推荐的顺序是：

1. **样本级检查**：样本间相关性、PCA（通常在方差稳定变换后的数据上）、文库组成、低 count 基因、批次与异常样本（outlier）；
2. **建立设计**：包含 condition、batch、pairing 或交互项；
3. **指定对比**：需要解释的是预先指定 contrast 的 $\log_2$ fold change。

报告中至少应包含 base mean、$\log_2$ fold change、标准误、原始 $p$ 值、BH 校正后的 $p$ 值与置信区间；使用 LFC 收缩（LFC shrinkage，如 DESeq2 的 apeglm、ashr）时还要注明所用的估计器（estimator）。独立过滤、outlier 替换、样本剔除与批次处理也都应记录在案，以便他人复现。

> [!NOTE]+ 小结
>
> - qPCR 用 $2^{-\Delta\Delta Ct}$ 做相对定量，依赖效率相近与参考基因稳定；芯片受交叉杂交与动态范围限制。
> - RNA-seq 差异表达对原始 count 建负二项 GLM，方差 $\mu+\alpha\mu^2$；size factor 用中位数比值法抵抗组成偏差；不要输入 TPM。
> - BH 控制 FDR，adjusted p-value 通过累积最小值保证单调；效应量与置信区间同样要报告。
> - GSEA 用全排序和加权 running sum 避免阈值；置换方式决定 null 的含义。
> - 设计矩阵必须满秩，批次与条件完全混杂无法事后校正；生物学重复不可由技术重复替代。

<br>

<br>

# 全基因组关联研究（Genome-Wide Association Study, GWAS）

人群中的个体在基因组上存在大量差异（主要是单核苷酸多态性 SNP），同时在身高、血脂、是否患 2 型糖尿病等性状（phenotype）上也各不相同。一个自然的问题是：哪些遗传差异与哪些性状“一起变化”？GWAS 的做法很直接——在几十万到上千万个变异位点上逐个做统计检验，看某个等位基因（allele）的携带量是否与性状相关，然后把全基因组的检验结果画在一张图上，寻找显著偏离随机的位置。

核心想法可以用一句话概括：**GWAS 是“对每个 variant 做一次回归，再对几百万次检验做多重校正”**。它的优势是无需事先假设候选基因，缺点是只能给出统计关联（association），而且信号会被连锁不平衡（linkage disequilibrium, LD）“摊开”到一片区域。前面关于数据库、表达分析的章节处理的是“某个基因在做什么”；本章换到群体视角，问“哪段基因组和性状有关”；下一章（变异效应预测）则回到单个 variant，问“它是否真的破坏了分子功能、能否解释某位患者的疾病”。

## GWAS 回答什么、不回答什么

### 研究对象与输入输出

GWAS 研究的是：**群体中的遗传差异是否与 phenotype 共同变化**。涉及的对象有三类：

- **基因型剂量（genotype dosage）**：每个个体在每个 variant 上携带多少个效应等位基因（effect allele），取值 0、1、2；经过填补（imputation）的位点则是 0–2 之间的期望值。
- **表型（phenotype）**：可以是离散的（病例/对照）或连续的（如 LDL 胆固醇浓度）。
- **协变量（covariate）**：ancestry（祖源）、年龄、批次（batch）、亲缘关系等可能同时影响基因型与表型的因素。

输入通常是经过质量控制（QC）和 imputation 的 genotype matrix（行是个体，列是 variant）。输出是每个 variant 的效应估计 $\hat\beta$、标准误与 $p$ 值，进一步汇总成 Manhattan plot、QQ plot 和候选 locus（候选区域）列表。

### 解释边界与后续证据

GWAS 回答的是“某 allele 是否与性状相关”，**不直接回答“该 allele 是否因果致病”**。原因主要有两个：

1. **LD 使信号成片出现。** 相邻 variant 在人群中往往一起遗传，彼此高度相关。只要其中一个是真正的 causal variant，它周围许多 SNP 都会显示显著关联，因此同一 causal signal 周围会出现多个显著 SNP。
2. **群体结构（population structure）可能产生混杂。** 如果某个亚群体既在某 allele 上频率较高、又因为环境或其他遗传背景更容易患病，二者就会出现与生物机制无关的虚假关联。

因此，从 GWAS 峰到生物机制，还需要精细定位（fine mapping）、功能注释（functional annotation）、eQTL/共定位（colocalization）分析和实验验证，才能逐步把 variant、靶基因（target gene）与机制连接起来。

### 与个体诊断的区别

两者容易混淆，需要明确区分：

- GWAS **从群体中寻找 association**，典型效应很小（单个常见 variant 往往只把患病风险提高百分之几到几十），对单个个体的预测力有限。
- 个体诊断（如罕见病的外显子测序）则**结合 allele 频率、遗传模式、家系和 phenotype 来判断某个候选 variant** 是否能解释该患者的疾病，这正是下一章“变异效应与致病性预测”的主题。

<br>

## 研究设计：回归模型、协变量与质量控制

### 关联检验的回归形式

GWAS 在全基因组范围逐个检验 genotype 与 phenotype 的统计关联。对病例–对照（case–control）设计，第 $i$ 个个体的表型 $Y_i\in\{0,1\}$，可用逻辑回归（logistic regression）：

$$
\log\frac{P(Y_i=1)}{1-P(Y_i=1)}
=\beta_0+\beta_gG_i+\boldsymbol\beta_C^\top C_i
$$

各符号的含义：

- $G_i$：个体 $i$ 在当前 variant 上的 effect-allele dosage，编码为 0/1/2（加性模型，additive model），即每多一个效应等位基因，对数几率（log-odds）增加 $\beta_g$；
- $e^{\beta_g}$ 就是每个等位基因的比值比（odds ratio, OR），GWAS 汇总统计里报告的 OR 即由此而来；
- $C_i$：协变量向量，包括 ancestry principal component（祖源主成分）、年龄、性别、batch 等；$\boldsymbol\beta_C$ 是对应系数；
- 检验的零假设是 $H_0:\beta_g=0$，常用 Wald 检验或似然比检验得到 $p$ 值。

连续性状（如身高）常用线性回归（linear regression），形式相同，只是左边换成 $Y_i$ 本身，$\beta_g$ 解释为每个效应等位基因带来的平均表型变化。

两点实践细节：

- **样本相关时可使用混合模型（mixed model）。** 当样本中含有亲属或隐性亲缘（cryptic relatedness）时，普通回归会低估方差；线性混合模型用基因组亲缘矩阵（kinship / GRM）建模个体间的相关性，同时也能吸收部分群体结构。
- **imputed genotype 需要额外检查。** 填补得到的 dosage 有不确定性，需要检查 INFO score（填补质量）、allele frequency 和 dosage uncertainty；低 INFO 的位点效应估计不可靠。

### 质量控制（QC）

GWAS 检验的位点极多，任何微小的系统性技术偏差都会在几百万次检验中被放大成“显著信号”，所以 QC 非常关键。标准 QC 分两侧：

| 层面 | 检查项 | 目的 |
| --- | --- | --- |
| 样本侧 | 样本 call rate、杂合度、重复或亲缘关系、sex check、ancestry | 剔除低质量 DNA、样本污染或混样、重复样本，识别祖源离群者 |
| variant 侧 | call rate、次要等位基因频率（minor allele frequency, MAF）、Hardy–Weinberg equilibrium（HWE）、imputation quality | 剔除分型失败、极罕见（检验力不足）或分型错误的位点 |

HWE 偏离在对照组中常提示分型错误；在病例组中则可能是真实关联，因此通常主要在对照组中检查。

需要特别注意两点：**QC 阈值必须在分析前定义**（看了结果再调阈值等于“挑选”信号）；**病例和对照的技术处理若不一致**（例如在不同时间、不同平台、不同实验室分型）**会制造系统性关联**——两组之间任何技术差异都会和病例状态完全混杂在一起。

<br>

## 检验结果的可视化与多重检验校正

做完几百万次检验之后，需要回答两个问题：哪里有信号？整体结果是否可信？前者靠 Manhattan plot，后者靠 QQ plot；而“多显著才算显著”则由多重检验校正决定。

### Manhattan plot

- 横轴是按染色体排列的 genomic position（染色体 1 到 22，再到 X），交替着色；
- 纵轴通常为 $-\log_{10}p$，例如 $p=10^{-8}$ 对应高度 8；
- 高峰（形似曼哈顿天际线的“摩天楼”）表示某 locus 与性状有统计关联；
- 需要注意：峰中许多 SNP 可能只是与同一 causal variant 存在 linkage disequilibrium，**不能按显著点数理解为独立机制**。一个峰里有 50 个显著 SNP，很可能只对应一个独立信号；判断独立信号需要条件分析（conditional analysis）或 LD clumping。

### QQ plot

- 把观察到的 $p$ 值分位数与零假设（null，$p$ 值服从 $[0,1]$ 均匀分布）下的期望分位数比较，二者都取 $-\log_{10}$ 后作图；
- 如果绝大多数点落在对角线上、只有尾部上翘，说明只有少数真实信号；如果整条曲线从一开始就偏离对角线，说明存在总体膨胀（inflation），常用基因组膨胀系数 $\lambda_{GC}$ 量化；
- 它用于发现总体 inflation 或多基因（polygenic）信号——高度多基因的性状在大样本下本身就会让曲线整体抬高，不一定是混杂；
- 需要注意：**genomic inflation 不能只靠统一缩放掩盖 population stratification**。genomic control 把所有统计量除以 $\lambda_{GC}$，但群体分层对不同 variant 的影响随其在亚群间的频率差异而不同，统一缩放既可能残留偏差，也可能削弱真实的多基因信号；更好的做法是在模型中加入 ancestry PC 或使用 mixed model，并可借助 LD score regression 区分混杂与多基因性。

### 多重检验校正与 $5\times10^{-8}$ 阈值

若对每个 variant 都用 $\alpha=0.05$，一百万次独立检验会期望产生约五万个假阳性，因此必须校正。

- **Bonferroni 校正**：若独立检验数为 $m$，以 $\alpha/m$ 为单次检验阈值，控制族错误率（family-wise error rate, FWER），即“至少出现一个假阳性”的概率。
- **全基因组显著性阈值**：常见的 $5\times10^{-8}$ 大致等于 $0.05/10^6$，来自欧洲人群常见变异“有效独立检验数”约一百万的经验量级（HapMap 时代的估算，如 Pe'er 等 2008 年的工作）。由于 LD，几百万个 SNP 并不对应几百万次独立检验，所以有效检验数小于实际检验的 SNP 数。需要强调：它**不是所有 ancestry、测序研究和 variant class 的普适常数**。非洲祖源人群 LD 区块更短、独立检验更多；全基因组测序纳入大量稀有变异，有效检验数也更大，阈值通常需要更严格。
- **BH-FDR**：Benjamini–Hochberg 方法控制错误发现率（false discovery rate），即发现集合中的期望错误比例。它比 FWER 宽松，更常用于探索性或组学分析（如差异表达）。使用何种错误率取决于研究目标：要报告“确定的”关联位点用 FWER，要为后续实验筛选候选可用 FDR。

### 生物学分箱（biological binning）

单个稀有变异携带者太少，单点检验几乎没有检验力（power）。一个补救办法是把多个 variant 合并成一个检验单元：

- 按 gene、pathway、调控区域（regulatory region）或其他先验单元聚合 variant；
- 再进行负荷检验（burden test）、方差成分检验（variance-component test）或组合检验（combined test）；
- 这样可以减少检验数，并提高多个弱效应或稀有变异的 power。

但分箱也引入了新的假设与边界：

- bin 边界如何划定、bin 内效应方向是否一致、是否混入了大量中性（neutral）variant、bin 内 variant 之间的 LD，都成为新的建模假设。例如 burden test 假设 bin 内 variant 效应方向相同，若有保护性与致病性 variant 混在一起，信号会相互抵消；SKAT 这类方差成分检验则允许方向不一致。
- 它**不是把同一 bin 内单点 $p$ 值直接相加**——单点检验相关且分布不可加，直接相加没有合法的零分布。
- 置换检验、LD-aware 汇总统计方法、burden test 或 SKAT 等各自定义了不同的 null，结果之间不能直接互换解释。

<br>

## 从统计关联到生物机制

### 偏差来源与 lead SNP 的含义

GWAS 的基本结论只是：allele 与 trait 在给定模型和样本中存在 association。以下因素都可造成偏差：

- **群体分层（population stratification）**：亚群间 allele 频率与患病率同时不同；
- **隐性亲缘（cryptic relatedness）**：样本中未被识别的亲属让观测不独立；
- **批次效应（batch）**：分型平台或时间与病例状态相关；
- **选择偏倚（selection bias）**：样本招募方式与基因型或表型有关；
- **表型错分（phenotype misclassification）**：对照中混入未诊断的病例，会稀释效应或引入偏差。

即使关联可以重复，**显著性最强的 lead SNP 也常只是标签变异（tag variant）**，它和真正的 causal variant 处于 LD 中，本身未必有功能。精细定位需要：

- **可信集合（credible set）**：在给定 LD 结构下，以例如 95% 后验概率包含 causal variant 的最小 variant 集合；
- **跨 ancestry LD**：不同人群的 LD 结构不同，联合分析可以缩小可信集合；
- **functional annotation**：variant 是否落在编码区、启动子、增强子或开放染色质；
- **eQTL/colocalization**：GWAS 信号与某基因表达调控信号是否共享同一 causal variant；
- **染色质互作**（如 Hi-C）：远端调控元件究竟连接到哪个基因启动子；
- **实验扰动**：如 CRISPR 编辑、报告基因实验。

### 证据链

孟德尔随机化（Mendelian randomization）、colocalization 和中介分析（mediation analysis）各自有额外假设（例如孟德尔随机化要求工具变量只通过暴露影响结局，没有水平多效性），这些结论不能由 Manhattan peak 自动推出。最终的证据链应把每一环分开论证：

$$
\text{associated locus}
\rightarrow\text{candidate causal variant}
\rightarrow\text{target gene}
\rightarrow\text{molecular mechanism}
\rightarrow\text{phenotype}
$$

每一箭头都需要独立证据：从 locus 到候选 causal variant 靠 fine mapping；从 variant 到 target gene 靠 eQTL、colocalization 与染色质互作；从基因到机制、再到表型则要靠功能实验。最近的基因不一定是靶基因，这是常见误区。

“相关不等于因果”在 GWAS 中有三种具体表现：

1. **LD 造成 variant 混淆**：显著的 SNP 可能只是 causal variant 的“邻居”；
2. **群体结构造成个体混淆**：携带者与非携带者在祖源、环境上本来就不同；
3. **pleiotropy（多效性）或环境因素造成机制混淆**：同一 variant 可能通过另一条通路影响性状。

> [!NOTE]+ 小结
> - GWAS 对每个 variant 做回归（病例–对照用 logistic，连续性状用 linear），协变量至少包含 ancestry PC，相关样本用 mixed model。
> - QC 阈值要预先定义；病例与对照的技术处理不一致会制造系统性假关联。
> - Manhattan plot 找峰，QQ plot 查膨胀；一个峰内多个显著 SNP 通常源于 LD，不代表多个独立机制。
> - $5\times10^{-8}$ 是欧洲人群常见变异约一百万次有效独立检验的 Bonferroni 经验值，不是普适常数。
> - lead SNP 常只是 tag variant；从 locus 到机制需 fine mapping、功能注释、eQTL/colocalization 与实验逐环论证。

<br>

# 生物数据上的机器学习

前面的章节积累了各种各样的生物数据表示：表达矩阵、结构特征、化学指纹、蛋白质语言模型 embedding。它们最终都要落到同一个问题上——如何从这些特征中学到可靠的预测规则或发现内部结构。机器学习提供了通用工具，但生物数据有几个特殊难点：样本少、特征多、样本之间常有隐藏的依赖（同一患者、同源蛋白、相似骨架），而且标签往往不平衡。

本章的核心观点是：**模型好坏首先取决于数据拆分和评估方式，其次才是算法名称**。一个在随机拆分上 AUROC 0.95 的模型，换成按同源家族拆分后可能接近随机。因此本章依次讨论特征与任务（监督、非监督）、评估指标与泄漏控制、可解释性的边界。

## 任务类型、输入输出与训练边界

### 研究对象与任务类型

- 机器学习把生物对象转换成特征矩阵，再从样本中学习预测规则或内部结构。
- <strong>监督学习（supervised learning）</strong>使用标签预测类别（分类）或连续值（回归）。
- <strong>非监督学习（unsupervised learning）</strong>没有目标标签，但仍依赖距离、标准化、簇数和特征选择等假设——“非监督”不等于“无假设”。
- PCA、聚类和分类器都是通用工具，**不会自动理解 gene、protein 或 molecule 的生物语义**；生物意义来自特征构造和结果验证。

### 输入、输出与训练边界

- 输入是特征矩阵 $X\in\mathbb R^{n\times d}$（$n$ 个样本、$d$ 个特征）、可选的标签 $y$，以及定义独立样本单位的分组键（group key），例如患者 ID、序列聚类 ID 或化学骨架 ID。
- 输出可以是预测（prediction）、嵌入（embedding）、簇（cluster）、特征重要性（feature importance），以及带置信区间的评价指标。
- **预处理、特征选择和超参数搜索必须只在 training data 内完成。** 例如在全部样本上先挑出与标签最相关的 100 个基因，再做交叉验证，测试折的信息已经通过特征选择泄漏进模型，性能会被严重高估。

### 章节衔接与泄漏控制

本章收束表达矩阵、结构特征、化学指纹和蛋白 embedding 等不同输入。模型性能的关键不只是算法名称；拆分方式必须阻止同一患者、同源蛋白、相同 scaffold 或未来信息跨集合泄漏。

<br>

## 特征、降维、聚类与分类

### 课程中的典型任务

课程语境中的统计特征检测、聚类和分类可用于：

- 蛋白家族或功能分类；
- 样本亚型发现（如肿瘤分子亚型）；
- 变异效应预测；
- 药物反应预测；
- 细胞类型识别（如单细胞 RNA-seq）。

监督任务需要 label；非监督任务不等于没有假设，距离度量、标准化方式和簇数都编码了假设。比如用欧氏距离聚类表达数据时，高表达基因会主导距离，除非先做 log 变换或标准化。

### 降维：PCA、t-SNE 与 UMAP

- <strong>主成分分析（PCA）</strong>寻找最大方差的线性方向。它是确定性的、全局的，适合 QC（发现批次效应、离群样本）与压缩；GWAS 中用于 ancestry 校正的也正是基因型矩阵的 PCA。
- **t-SNE、UMAP** 是非线性方法，强调保持局部邻域关系。代价是：二维图中点间的全局距离和 cluster 面积通常不可定量解释——两个簇在图上离得远，不代表它们在原空间中更不相似；簇画得大，也不代表异质性更高。

一个常被忽视的事实：是否做 log transform、是否 scale feature、是否只选 highly-variable gene，往往比降维算法名更深地改变图形。报告降维图时应同时报告这些预处理选择。

### $k$-means 聚类

$k$-means 最小化簇内平方欧氏距离（within-cluster squared Euclidean distance）：

$$
\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|_2^2
$$

其中 $K$ 为预设簇数，$C_k$ 为第 $k$ 个簇的样本集合，$\mu_k$ 为该簇均值（质心）。算法交替执行“把每个点分配给最近质心”和“把质心更新为簇内均值”两步，每一步都不增加目标函数，所以必然收敛。

需要注意的性质：

- 由于目标是平方欧氏距离，它偏好近似球形、方差相近的簇，对细长或大小悬殊的簇效果差；
- 只保证收敛到局部最优，结果依赖初始化，实践中用多次随机初始化或 k-means++ 并取目标最小者。

### 层次聚类与 linkage

层次聚类（hierarchical clustering）自底向上逐步合并最近的两个簇，除了距离度量，还必须指定簇间距离的定义，即 linkage：

- **single linkage**（两簇最近点对距离）：容易产生链式效应（chaining），一串彼此相邻的点会把两个本应分开的簇连起来；
- **complete linkage**（两簇最远点对距离）：偏好紧致的簇；
- **average linkage**：取两簇所有点对距离的平均，介于二者之间。

需要注意：dendrogram（树状图）是数据处理、距离和 linkage 的共同产物，不自动等于生物 taxonomy。换一种 linkage 可能得到拓扑完全不同的树。

### Silhouette 系数：簇划分好不好

聚类后需要量化“分得清不清楚”。Silhouette coefficient 同时衡量簇内紧致性与到最近其他簇的分离度。对样本 $i$，令

- $a(i)$：它到同簇其他点的平均距离（越小越紧致）；
- $b(i)$：它到每个其他簇的平均距离中的最小值（即到“最近邻簇”的平均距离，越大越分离）。

则

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}\in[-1,1]
$$

为什么取值在 $[-1,1]$：分子 $b-a$ 的绝对值不超过 $\max\{a,b\}$。解释如下：

- 接近 1：$a\ll b$，分配清晰；
- 接近 0：$a\approx b$，样本位于两簇边界；
- 负值：$a>b$，它平均更靠近另一簇，可能被分错。

单点簇（簇中只有它自己）没有 $a(i)$，通常约定 $s(i)=0$。

平均 silhouette 可辅助选择簇数（对不同 $K$ 计算，取较大者），但它偏好凸的、分离良好的簇，不能替代稳定性分析、外部 annotation 或生物验证；不同距离度量下的分数也不能直接比较。

下面是一个教学实现，逐行对应上面的定义：

```python
import numpy as np

def silhouette_samples(X, labels):
    """以 Euclidean distance 计算每个样本的 silhouette coefficient。"""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or labels.shape != (len(X),):
        raise ValueError("X must be (n, d) and labels must be (n,)")
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(X):
        raise ValueError("need 2..n-1 clusters")

    # 教学实现显式构造 O(n^2) 距离矩阵；大数据应分块或调用成熟库。
    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    result = np.zeros(len(X), dtype=float)

    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False  # a(i) 不包含样本到自身的零距离。
        if not np.any(same):
            result[i] = 0.0  # 单点簇的标准软件约定。
            continue
        a_i = distances[i, same].mean()
        b_i = min(
            distances[i, labels == other].mean()
            for other in unique
            if other != labels[i]
        )
        denominator = max(a_i, b_i)
        # 所有相关距离都为 0 时没有可定义的分离方向，约定为 0。
        result[i] = 0.0 if denominator == 0 else (b_i - a_i) / denominator
    return result
```

代码要点：簇数必须在 2 到 $n-1$ 之间（只有 1 簇时没有 $b(i)$，$n$ 簇时每个点都是单点簇）；计算 $a(i)$ 时排除自身；分母为 0 时（所有相关点重合）约定返回 0。这些约定与 scikit-learn 的 `silhouette_samples` 一致。

### 分类与 baseline

分类至少应建立简单 baseline：多数类预测（majority class）、逻辑回归（logistic regression）或最近邻（nearest-neighbor）。复杂模型若只胜过很弱的 baseline，不能说明表示真正捕获了可泛化的生物规律。例如在蛋白功能预测中，基于序列相似性的最近邻（BLAST 最佳命中的注释）往往就是一个很强的 baseline，深度模型必须明显超过它才有意义。

<br>

## 模型评估：指标、拆分与泄漏

### 类别不平衡下的指标选择

类别不平衡时，accuracy 可能失真：若 99% 的 variant 是良性的，全预测“良性”就有 99% 准确率。此时应报告：

- 混淆矩阵（confusion matrix）；
- 精确率（precision）、召回率/灵敏度（recall/sensitivity）、特异度（specificity）、Matthews 相关系数（MCC）；
- 明确积分方式的 average precision（AP）或 PR-AUC（梯形积分与阶梯式 AP 数值不同，必须说明）。

PR 曲线的随机基线约等于正例比例（prevalence）：随机打分时 precision 恒约等于 prevalence。因此不同 prevalence 数据集上的 PR-AUC 不能直接横向比较。

超参数选择只能在 validation 集上完成，test 集只用于最终一次评估。

### 混淆矩阵与指标公式

令 TP、FP、TN、FN 为混淆矩阵四项（真阳性、假阳性、真阴性、假阴性）：

$$
\mathrm{precision}=\frac{TP}{TP+FP},\quad
\mathrm{recall}=\frac{TP}{TP+FN},\quad
\mathrm{specificity}=\frac{TN}{TN+FP}
$$

precision 回答“预测为阳性的里有多少真是阳性”，recall 回答“真阳性里找回了多少”，specificity 回答“真阴性里有多少被正确排除”。F1 是 precision $P$ 与 recall $R$ 的调和平均：

$$
\mathrm{F1}=\frac{2PR}{P+R}
$$

调和平均的特点是被较小者主导：只要 $P$ 或 $R$ 之一很低，F1 就低。

### AUROC、precision 与校准

- **AUROC** 等于随机抽一对正负样本时正例得分更高的概率，再加上二者得分相等概率的一半（即 Mann–Whitney U 统计量的归一化形式）。它不受 prevalence 直接影响，但正因如此，在极不平衡任务上仍可能显得乐观：大量容易排除的阴性样本会抬高 AUROC，而实际部署时排在前面的预测中假阳性仍可能很多。
- **precision 无法脱离部署 prevalence 解释**：同一模型在 prevalence 1% 与 50% 的人群中 precision 截然不同。
- <strong>校准（calibration）</strong>检验预测概率与实际频率是否一致：模型说“70% 概率阳性”的样本中，是否大约 70% 真是阳性。

概率模型还可用可靠性图（reliability diagram）、Brier score（预测概率与 0/1 结果之差的均方）、calibration intercept 与 slope 评估。Platt scaling 或 isotonic regression 等事后校准方法必须只在训练/验证数据上拟合，不能在 test 上拟合。

### 阈值与置信区间

分类阈值应依据漏诊/误报的代价、预设的最低 sensitivity，或决策曲线（decision-curve）的 net benefit 预先选定，**不能看过 test 结果后再调**。AUROC、AP 和 sensitivity 等指标应通过按 patient、protein family 或其他真正独立单位的 bootstrap 给出置信区间（confidence interval）——若按单个样本 bootstrap，而同一患者有多个样本，区间会过窄。

### 常见数据泄漏

数据泄漏（data leakage）指测试信息以某种方式进入训练过程，使评估过于乐观。常见形式：

- 同一患者的多个样本跨越训练/测试集；
- 同源蛋白跨越训练/测试集（模型只是记住了家族）；
- 预处理（标准化、特征选择、填补）在全数据上 fit；
- 标签由未来信息派生；
- batch 与 label 重合（模型学到的是批次而不是生物信号）。

### 分组交叉验证与嵌套交叉验证

交叉验证必须按真正的独立单位分组：

- protein function 任务按序列聚类（sequence cluster，例如按一定序列一致性阈值聚类）分组；
- 临床任务按 patient 分组；
- 面向未来发现的任务按时间切分。

嵌套交叉验证（nested cross-validation）把调参放在内层循环，外层只用于估计泛化性能，避免把同一个 test fold 反复用于模型选择——那样得到的性能是“选出来的最好值”，而不是对新数据的无偏估计。

<br>

## 可解释性的边界

模型训练好后，研究者常问“哪些特征最重要”，并希望由此得到生物学洞见。需要谨慎的是：

- **feature importance 不等于因果效应。** 重要性衡量的是特征对模型预测的贡献，而不是干预该特征对生物系统的影响。
- **共线特征会分摊或替代重要性。** 两个高度相关的基因，模型可能只用其中一个，另一个的重要性就接近零；或者二者平分重要性，各自都显得不重要。
- **post-hoc explanation 解释的是模型行为，而不是证明生物机制。** 候选机制仍需独立数据与干预实验验证。

SHAP、permutation importance 等方法还依赖特征依赖性（feature dependence）的假设。对高度相关的 gene expression feature，单独打乱一个 gene 会制造训练分布之外的样本（例如两个总是同时高表达的基因被拆开），模型在这类“不存在的样本”上的反应并不反映真实生物学。稳定性分析（不同随机种子、不同子样本下重要性是否一致）、外部队列和机制实验，比单张 explanation plot 更有说服力。

> [!NOTE]+ 小结
> - 预处理、特征选择、调参都只能在训练数据内完成；拆分要按患者、序列聚类、scaffold 或时间分组。
> - PCA 适合 QC 与压缩；t-SNE/UMAP 的全局距离和簇面积不可定量解读。
> - $k$-means 偏好球形簇且只到局部最优；层次聚类结果依赖 linkage；silhouette 只是辅助指标。
> - 不平衡数据报告混淆矩阵、MCC、PR-AUC 和校准；PR-AUC 基线等于 prevalence。
> - 阈值预先设定，置信区间按独立单位 bootstrap；特征重要性不等于因果。

<br>

# 变异效应与致病性预测

一个人的外显子组测序通常能找到上万个与参考基因组不同的位点，其中大部分无害。临床遗传学要做的是在这些 variant 中找出能解释患者疾病的那一个（或一对）。仅靠实验逐个验证不现实，于是出现了一系列计算方法：先注释每个 variant 改变了什么（是否改变蛋白序列），再用进化保守性、蛋白结构、全基因组注释等信息预测它是否“有害”，最后结合人群频率、遗传模式和患者表型进行排序。

本章的核心观点是：**“预测有害”与“临床致病”是不同层次的结论**。SIFT、PolyPhen-2、FATHMM、CADD、DANN 等工具回答的是分子功能或进化选择层面的问题，而临床致病性还需要频率、遗传模式、家系共分离、功能实验和表型匹配。本章承接前面的序列保守性、Profile HMM、蛋白结构环境和生物本体（ontology）等内容；与上一章 GWAS 的区别在于，GWAS 在群体中寻找统计 association，本章则判断具体 variant 是否可能影响分子功能、能否解释某位患者的疾病。

## 从基因组坐标到患者：证据层级

### 证据层级

变异分析把基因组坐标上的 REF/ALT（参考/替代等位基因）逐步连接到 transcript、protein、gene、disease，最后到患者 phenotype。每一层回答不同问题：

- <strong>后果注释（consequence annotation）</strong>判断变异改了什么（同义、错义、无义、剪接位点等）；
- <strong>有害性模型（deleteriousness model）</strong>判断这种改变是否可能损害分子功能或受到负选择；
- <strong>临床致病性（pathogenicity）</strong>还需要频率、遗传模式、家系共分离、功能实验和表型匹配。

### 输入、输出与模型差异

- 输入是固定了 genome build、等位基因标准化（allele normalization）、transcript 和患者背景之后的 variant；
- 输出可能包括 transcript consequence、蛋白替换（如 p.Arg117His）、deleteriousness score、群体频率和患者级候选排序；
- SIFT、PolyPhen-2、FATHMM、CADD、DANN 的训练目标与分数尺度各不相同；
- 不能把多个工具的标签机械投票当作独立的临床证据（原因见后文“集成与评估”）。

<br>

## 四个层次的问题与坐标规范

### 四个不同的问题

测序得到 variant 后，至少有四个不同的问题：

1. variant 是否改变 transcript、剪接位点（splice site）或 protein sequence；
2. 这种改变是否影响分子功能；
3. 该改变是否降低个体适合度（fitness），即产生 deleterious effect；
4. 在给定遗传模式、外显率（penetrance）和患者表型下，它是否能解释疾病。

对应关系：variant consequence annotation 回答第一项；SIFT、PolyPhen-2、FATHMM 和 CADD 主要为第二、三项提供分数；家系共分离、群体频率、功能实验和表型匹配才进入第四项。因此 `damaging`、`deleterious` 与 `pathogenic` 三个词不能互换。计算分数可用于排序，或按 ACMG/AMP 变异解读指南作为 supporting（支持性）级别的计算证据，但不能单独完成临床分类。

### 分析前必须固定的坐标规范

同一个变异可以有多种写法，必须先统一：

- **genome build**（如 GRCh37 与 GRCh38 坐标不同）；
- **reference/alternate allele** 与 **left normalization**（插入缺失在重复序列中可有多种等价位置，统一左对齐才能比较）；
- **transcript release** 与 **strand**。

一个 genomic variant 可能在不同 transcript 上分别成为 synonymous、missense、splice-region 或 intronic variant，所以注释必须说明用的是哪个 transcript。蛋白坐标还需记录 isoform；否则同一个 `p.Arg117His` 可能无法唯一映射回基因组。

<br>

## SIFT：基于同源序列的替换容忍度

### 思路与流程

SIFT（Sorting Intolerant From Tolerant）的直觉是：**如果一个位置在进化中始终只出现少数几种氨基酸，那么换成没出现过的氨基酸很可能有害**。它只依赖序列同源信息，流程为：

1. 检索与 query protein 相关的同源序列；
2. 构建多序列比对（MSA）；
3. 估计每个位置上 20 种氨基酸的允许程度（概率）；
4. 用该位置最常见氨基酸的概率归一化，得到 scaled probability，即 SIFT score。

### 分数与阈值

- SIFT score 范围为 0 到 1；
- 2009 年 Nature Protocols 通常把 $\leq0.05$ 解释为 predicted damaging（预测有害），$>0.05$ 为 tolerated；
- 低分表示该 substitution 在当前同源序列集合与 alignment 中很少被容许，**不是 95% 的致病概率**——0.05 只是经验截断点。

### 置信度来源

SIFT 的可靠性取决于 MSA 质量：

- alignment 太浅（同源序列太少）会使估计不稳定；
- 序列彼此过于相近，会把许多位置误判为高度保守（并非真保守，只是还没来得及变）；
- 该 protocol 用 median conservation（中位保守度，以信息量衡量序列多样性）判断这一点；
- median information $>3.25$ 时给出 low-confidence warning（低置信警告）。信息量最大约为 $\log_2 20\approx4.32$ 比特，中位值过高说明比对中的序列几乎一致。

### 与 BLOSUM 的区别

SIFT 与 BLOSUM 的目标不同。BLOSUM 汇总许多蛋白家族中的平均替换倾向（substitution tendency），适合给比对打分，对所有蛋白、所有位置一视同仁；SIFT 使用 query-specific MSA，估计**这个蛋白的这个位置**对替换的容忍度。SIFT 不直接使用三维结构、组织特异表达或疾病表型。

<br>

## PolyPhen-2：序列、结构与注释特征的朴素贝叶斯分类

### 特征选择与分类器

PolyPhen-2 在 SIFT 式保守性信息之外，加入了结构和注释信息：

- 同时使用 sequence、structure 和 annotation feature；
- 2010 年方法先构造 32 个候选特征；
- 再以贪心特征选择（greedy selection）选出 8 个序列特征与 3 个结构特征，共 11 个；
- 最后用朴素贝叶斯分类器（naive Bayes classifier）区分 damaging 与 non-damaging 的 missense variant。

最终特征主要描述 wild-type 与 mutant residue 在 MSA 中的 profile（例如 PSIC 分数及其差值）、替换在多远的同源序列中出现过，以及结构环境（如溶剂可及性、是否处于埋藏区域）。CpG context、Pfam annotation 等属于候选信息，不能全部视为最终 11 个分类特征。

### 两套训练集：HumDiv 与 HumVar

- **HumDiv**：damaging 集合来自引起 Mendelian disease 的 allele（UniProt 中已知影响分子功能的致病突变），negative 集合是接近跨物种固定的差异（人与近缘哺乳动物同源蛋白之间的差异）；适合评估罕见变异、复杂性状研究中的候选筛选。
- **HumVar**：以人类 disease mutation 对比人群中的常见 nsSNP（非同义 SNP），更适合孟德尔病诊断中区分致病突变与轻微有害的常见多态性。
- 两个模型的输出不可脱离训练集直接比较。

### 操作点与标签边界

- PolyPhen-2 v2.2 对 HumDiv 使用 5% 与 10% 假阳性率（false-positive-rate）操作点（operating point）；
- 对 HumVar 使用 10% 与 20%；
- 这些操作点划分出 `benign`、`possibly damaging` 和 `probably damaging` 三个标签。它们**不是通用概率阈值**，也不是 ACMG/AMP 意义上的 clinical significance。

<br>

## FATHMM：用 profile HMM 比较野生型与突变型

### 思路

FATHMM（Functional Analysis through Hidden Markov Models）把前面 HMM 章节中的 profile HMM 直接用于变异预测：用 profile HMM 表示同源序列或保守蛋白结构域（protein domain）。对 variant 所在的 match state，比较 wild-type 与 mutant amino acid 的发射概率（emission probability）。

- <strong>未加权版本（unweighted）</strong>主要反映进化容忍度；
- <strong>人类加权版本（human-weighted）</strong>再加入 HGMD 致病突变与 UniProt 推定中性 variant 在该 domain 中的相对频率——如果某个 domain 里致病突变特别多，该 domain 中的替换就更倾向于判为有害。

### 分数公式

若 $P_w$、$P_m$ 分别为 wild-type 与 mutant residue 在该 match state 的 HMM emission probability，原论文的未加权分数为

$$
s_{\mathrm{unweighted}}
=\log\frac{P_m/(1-P_m)}{P_w/(1-P_w)}
$$

分子分母各是一个“几率”（odds）：$P/(1-P)$。整体是突变型几率与野生型几率之比的对数。负值表示 mutant residue 的 HMM odds 低于 wild type，即突变型在该位置更“不被进化接受”。

### 阈值与疾病特异版本

2013 年论文（Shihab 等，在 SwissVar 数据上取灵敏度与特异度同时最大处）报告 unweighted 与 human-weighted model 的经验 threshold 分别为 $-3.0$ 和 $-1.5$，human-weighted score 通常以 $<-1.5$ 判为 damaging。阈值依 model 与 version 而异。

疾病特异（disease-specific）FATHMM 又对 17 类 disease concept 使用不同权重，目标是把“可能影响功能”进一步缩小到“与当前疾病类别更相关”。

### 与其他章节的联系与局限

FATHMM 把本章与 profile HMM、domain annotation 和 ontology 三部分连接起来。其性能仍受 domain 覆盖度（不在任何已知 domain 中的位置无法打分或依赖整蛋白比对）、训练数据库与测试集的重叠（独立评估曾发现加权版本在训练/测试重叠时可能过拟合），以及 pathogenic/neutral 标签质量的影响。

<br>

## CADD：全基因组有害性排名

### 训练思路：观察到的 vs 模拟的变异

前面三种方法主要针对错义变异。CADD（Combined Annotation Dependent Depletion）面向全基因组的 SNV 和短 indel，不限于 missense variant。它的巧妙之处在于绕开了“致病标签少且有偏”的问题，改为区分两类 variant：

- **模拟的 de novo variant**：按突变模型在基因组上随机生成，尚未经历自然选择，其中包含正常比例的有害变异；
- **人类与黑猩猩分化后在人群中固定或近固定的 derived allele**：它们通常已经通过纯化选择（purifying selection）的筛选，有害变异被大量清除。

如果某类注释特征（如高度保守、位于编码区）在“观察到的”集合中显著少于“模拟的”集合，说明带有这种特征的变异被选择清除了，即更可能有害——这就是名字里“depletion（耗竭）”的含义。

### 模型与标签含义

2014 年模型（CADD v1.0，训练矩阵共约 2940 万个 variant，observed 与 simulated 各约 1470 万）将 conservation、regulatory annotation、transcript consequence、chromatin、protein score 等 63 类 annotation 输入线性 SVM（linear-kernel SVM）；自 v1.1 起改用 logistic regression，并更新扩展了 annotation 集合（依据 CADD 官网 release notes）。训练标签不是临床上的 pathogenic/benign，模型学习的是 observed 与 simulated variant 的 annotation 差异。因此 CADD 更接近全基因组的 deleteriousness ranking（有害性排名），**不能解释为给定疾病下的致病概率**。

### PHRED 式标准化分数

Raw C-score 是模型的原始输出，尺度随注释集合变化而无直观含义。PHRED-like scaled C-score 按该 variant 在所有可能 substitution（v1.0 中为 GRCh37 参考基因组上约 86 亿种可能的单碱基替换）中的排名计算：

$$
C_{\mathrm{scaled}}=-10\log_{10}\left(\frac{\operatorname{rank}}{N}\right)
$$

其中 rank 为从最有害开始的 1-based 名次，$N$ 为所有可能 substitution 的总数。代入可得：rank/$N=10^{-2}$ 时 $C=20$，rank/$N=10^{-3}$ 时 $C=30$。因此 CADD 20 表示约位于全基因组可能 SNV 的前 1%，CADD 30 表示前 0.1%。它们是排名阈值（rank threshold），不是 1% 或 0.1% 的错误概率——虽然借用了测序质量 PHRED 分数的形式，含义完全不同。

reference build 和 CADD release 会改变 annotation 与分数，因此报告时必须注明版本；2019 年的更新（v1.4，Rentzsch 等，Nucleic Acids Research）支持 GRCh38，并继续扩展 feature 与接口。

```python
import math

def cadd_phred(rank: int, total: int) -> float:
    """把 1-based 全局排名转换为 PHRED-like relative rank。"""
    if not 1 <= rank <= total:
        raise ValueError("rank must be in [1, total]")
    # rank/total 是至少达到该名次的比例，不是错误概率。
    return -10.0 * math.log10(rank / total)
```

例如 `cadd_phred(1, 100)` 返回 20.0，`cadd_phred(1, 1000)` 返回 30.0；rank 越小（越靠前）分数越高，rank = total 时分数为 0。

<br>

## DANN 与非编码调控变异

### DANN

DANN 使用与 CADD 相近的大规模 annotation 和训练设计，把线性模型换成深度神经网络（deep neural network），以捕捉特征之间的非线性交互，输出 variant 的 deleteriousness score。模型可组合 conservation、蛋白后果和 regulatory feature，但高分仍不是临床 pathogenic probability。需要特别注意：DANN 与 CADD 共享部分训练设计和输入 annotation，**二者一致不构成两份独立证据**。

### 非编码变异可能影响的层级

大多数 GWAS 信号和许多罕见病候选落在非编码区，它们的解释尤其依赖细胞类型。一个非编码 variant 可能改变：

- 转录因子结合基序（transcription-factor binding motif）；
- 增强子/启动子（enhancer/promoter）活性；
- 染色质可及性（chromatin accessibility）；
- 剪接调控元件（splicing regulatory element）；
- RNA 稳定性。

### 证据边界

- 仅仅 motif score 改变，不能证明对应的 TF 在相关组织中表达、实际结合，或影响了目标基因；
- 应把序列 motif、表观基因组证据（epigenomic evidence）、QTL、三维染色质联系与扰动实验（perturbation experiment）分层记录，每一层只说明它能说明的事。

<br>

## 多工具集成与 benchmark 评估

### 为什么不能简单投票

不同方法共享 conservation、MSA、domain 和已有的 pathogenic 数据库，输出并非独立证据。把 SIFT、PolyPhen-2、FATHMM 和 CADD 的多数票相加，不会自动得到校准的概率；四个工具一致说“有害”，很大程度上是因为它们看的是同一份保守性信息。

Dong 等（2015）比较 18 种方法时发现，性能随 benchmark 和评价指标变化，部分集成方法（ensemble method，如 MetaSVM、MetaLR）优于单项方法；同一研究也显示，缺失分数（missing score）、训练/测试重叠、common benign 与 rare benign 的定义会显著影响结果。

### 致病性 benchmark 的常见偏差

- disease variant 与工具训练数据库重复（循环论证，circularity）；
- benign 集合由 common allele 构成，使 allele frequency 成为过强的捷径——模型只要学会“常见即良性”就能得高分；
- 同一 protein 或同源家族跨越 training/test split；
- 只保留所有工具都有分数的 variant，形成完整病例选择偏倚（complete-case selection bias）；
- 用总体 accuracy 掩盖 rare pathogenic variant 下的类别不平衡；
- 在同一 benchmark 上既选择 cutoff 又报告性能。

### 评估应报告什么

评价至少报告覆盖度（coverage）、缺失率（missingness）、AUROC、PR-AUC、sensitivity、specificity、MCC 和 calibration（指标定义见上一章）。按 gene、protein family 或发现时间（discovery time）分组切分可减少泄漏。临床场景还应按 ancestry、variant class、遗传模式（inheritance mode）和疾病领域（disease area）分层报告。

<br>

## 候选变异的过滤与优先级排序

### KGGSeq 三层框架

一个 exome 经过初步过滤后常保留数百个稀有候选 variant，排序需要合并 allele frequency、遗传模式、variant effect、基因约束（gene constraint）、phenotype 和家系信息。KGGSeq 2012 年的框架把这些信息分为三层：

1. **genetic level（遗传层）**：遗传模式（inheritance model）、受累者共享的基因组区域与家系约束；
2. **variant-gene level（变异–基因层）**：群体频率、序列后果、保守性、有害性分数与基因特征；
3. **knowledge level（知识层）**：已知疾病基因、通路（pathway）、相互作用和文献证据。

2017 年的 KGGSeq 扩展进一步处理全基因组注释、压缩 genotype 存取和大规模下游分析。

### 频率阈值与家系模型

- **频率阈值**必须与疾病患病率（prevalence）、外显率、遗传模式和 ancestry 相容。隐性病允许健康携带者存在，因此候选 allele 在人群中的频率可以相对高一些；显性高外显率病的候选 allele 通常要求更低的人群频率——如果健康人中经常出现该 allele，它就不可能是高外显率显性致病变异。
- **家系模型**：在 trio（患者与父母）中要区分新发（de novo）、复合杂合（compound heterozygous）、纯合隐性（homozygous recessive）和 X 连锁（X-linked）模型；
- **数据质量**：同时检查 read 层面质量与父母的 genotype quality，避免把父母处的漏检误判为 de novo。

### 三类数据库：gnomAD、ClinVar、OMIM

三类数据库回答不同的问题：

| 数据库 | 内容 | 用法与注意事项 |
| --- | --- | --- |
| **gnomAD** | 多个群体的大规模 allele frequency 与 gene constraint | 频率高于疾病模型允许上限可以反驳候选；但 ancestry 覆盖不均、测序可及性和低复杂度区域（low-complexity region）会影响“未观察到”的解释 |
| **ClinVar** | 提交者对 variant clinical significance 的断言，记录 review status 与冲突 | 数据库收录不等于专家共识，必须检查适用疾病、提交日期、星级和证据 |
| **OMIM** | 人类 gene、phenotype 与孟德尔关系的整理 | 适合建立 gene–disease 候选；不提供完整人群频率，也不提供逐 variant 的统一临床分类 |

数据库角色不能互换。典型过滤链是：

1. 先用 gnomAD 约束频率；
2. 再用 consequence/deleteriousness 排序；
3. 以 OMIM/HPO 检查 gene–disease 关系与 phenotype 相容性；
4. 最后核对 ClinVar 和原始病例/功能证据。

任何数据库都应记录 release 版本与访问日期。

<br>

## 表型语义匹配：HPO 与 Phrank

### 为什么需要表型语义

通过频率和有害性过滤后，往往仍有几十个候选基因。此时患者的临床表现是最有力的排序依据：如果患者有“小头畸形”和“癫痫”，而某候选基因已知导致的疾病恰好有这两个表现，它的优先级就应提高。困难在于表型描述的粒度不同——医生写“局灶性癫痫”，知识库写“癫痫”——这正是前面知识表示章节中本体（ontology）要解决的问题。

人类表型本体（Human Phenotype Ontology, HPO）把患者表现编码为有向无环图（DAG），越往下越具体。稀有且具体的 phenotype（如某种特定的骨骼畸形）比常见的上位词（如“发育异常”）携带更多信息。

### Phrank 分数

Phrank（Jagadeesh 等，Stanford）先分别取患者与疾病 phenotype set 的祖先闭包（ancestor closure，即所有 term 及其全部祖先），再对二者共有 term 的条件信息量（conditional information content）求和：

$$
\operatorname{Phrank}(\Phi_A,\Phi_B)
=\sum_{\phi\in \operatorname{anc}(\Phi_A)\cap\operatorname{anc}(\Phi_B)}
-\log_2\frac{|G_\phi|}{|G_{\operatorname{pa}(\phi)}|}
$$

- $\Phi_A$、$\Phi_B$：两个 phenotype term 集合（如患者表型与某疾病表型）；
- $\operatorname{anc}(\cdot)$：祖先闭包；
- $|G_\phi|$：与 term $\phi$ 关联的 gene 数，$|G_{\operatorname{pa}(\phi)}|$：与其父节点集合关联的 gene 数。

直觉：比值 $|G_\phi|/|G_{\operatorname{pa}(\phi)}|$ 是“已知基因落在父 term 下时，再细化到 $\phi$ 的条件概率”。细化得越窄（只有少数基因导致该具体表型），$-\log_2$ 越大，匹配上这个 term 贡献的信息越多。由于先取了祖先闭包，部分匹配（患者 term 与疾病 term 只在某个共同祖先上相交）也能得到相应的分数。

gene score 可取该 gene 所致各 disease score 的最大值。原论文在 169 个真实已诊断病例上评价，并指出基于知识库合成 phenotype 的 benchmark 会高估性能（合成病例的表型直接取自知识库，天然与知识库匹配）。

### Phrank 的边界与 PhenomeNET Variant Predictor

- Phrank 计算的是 phenotype compatibility（表型相容性），不判断某个 variant 是否破坏蛋白；完整的排序应同时满足 variant-level 证据与 gene/disease-level 表型证据。
- **PhenomeNET Variant Predictor** 用 ontology reasoning 整合人类与模式生物（model organism）的 phenotype，再用随机森林（random forest）合并语义相似度、variant 致病性、genotype 和遗传模式；目标是支持那些尚未建立明确人类 gene–disease 联系的候选基因——如果小鼠敲除该基因产生了与患者相似的表型，也能提供证据。

### 报告规范

报告需保留 HPO release、患者 term、否定表型（negated phenotype，明确不存在的表型）、发病年龄（onset）、知识库版本及候选 variant 过滤规则。缺失的 phenotype（没检查或没记录）与明确观察为阴性的 phenotype 含义不同；ontology annotation 的更新也会改变排序。

> [!NOTE]+ 小结
> - 区分四层问题：改变了什么、是否影响功能、是否有害、能否解释患者疾病；damaging、deleterious、pathogenic 不可互换。
> - SIFT 用 query-specific MSA（≤0.05 预测有害，median information >3.25 低置信）；PolyPhen-2 用 11 个特征的朴素贝叶斯，HumDiv/HumVar 操作点不同；FATHMM 比较 HMM 发射几率（阈值 −3.0 / −1.5）。
> - CADD 区分观察到的与模拟的变异，scaled 20/30 表示前 1%/0.1% 的排名，不是致病概率；DANN 与 CADD 不是独立证据。
> - 工具共享信息源，多数投票不等于校准概率；benchmark 需警惕循环、频率捷径和完整病例偏倚。
> - 候选排序综合 gnomAD 频率、有害性、OMIM/HPO 表型相容性（如 Phrank）和 ClinVar，并记录所有版本。

<br>

# 化学信息学（Cheminformatics）

药物发现需要在动辄上亿个化合物中找出可能结合靶点、具有理想性质的少数分子。实验筛选昂贵，于是需要用计算方法表示、比较和预测小分子。化学信息学处理的核心问题是：**如何把分子变成计算机能比较和学习的对象**，并在此基础上做相似性搜索、构效关系建模（QSAR）和虚拟筛选。

本章把前面的三条主线——字符串（SMILES 是分子的线性字符串）、图（分子是原子和键构成的图）和机器学习（QSAR 就是分子上的监督学习）——放到药物分子上。贯穿全章的提醒是：分子表示的每一个参数选择（标准化规则、指纹半径、位长）都会改变结果的含义。

## 分子的表示、任务与解释边界

### 研究对象与表示

- 化学信息学把小分子视为带原子标签和键类型的图；
- SMILES 与 InChI 是线性标识（字符串）；
- 指纹（fingerprint）把局部子结构压缩成稀疏集合或向量；
- 三维构象（conformer）进一步加入空间坐标；
- 同一化合物可能有不同的盐型（salt form）、互变异构体（tautomer）、质子化态（protonation）和立体异构体（stereoisomer），因此“两个分子是否相同”首先取决于标准化规则。

### 三类常见任务

| 任务 | 输入 | 输出 |
| --- | --- | --- |
| 相似性搜索（similarity search） | query molecule 和化合物数据库 | 按 Tanimoto 等相似度排序的近邻 |
| QSAR（定量构效关系） | 分子特征与实验 endpoint（如 IC50） | 活性或性质预测 |
| 分子对接（docking） | 受体结构与配体（ligand） | 候选结合构象（pose）和近似排序 |

### 解释边界

- docking score 不是严格的结合自由能；fingerprint 相似也不保证活性相似；
- Morgan fingerprint 的生成参数决定了 Tanimoto 值的含义；
- QSAR 的 scaffold/time split 决定了泛化任务的难度；
- assay 条件和单位决定了标签之间是否可比较。

<br>

## 分子表示：SMILES、分子图与指纹

### SMILES、InChI 与 canonicalization

SMILES 是分子图的线性编码，例如乙醇可写成 `CCO` 或 `OCC`——同一分子可有多个非 canonical SMILES。分子图以原子为节点、键为边。

SMILES 中的环闭合（ring closure，用数字标记，如苯 `c1ccccc1`）、分支（branch，用括号）、芳香原子（aromatic atom，小写字母）、形式电荷（formal charge，如 `[NH4+]`）和立体化学（stereochemistry，如 `@`、`/`、`\`）都有特定语法。

canonicalization（规范化）是软件定义的确定性排序规则，使同一分子总是输出同一个 SMILES；它**不是分子的物理属性**——不同软件（甚至同一软件的不同版本）可能给出不同的 canonical SMILES。InChI 提供分层的标准标识（连接性、氢原子、电荷、立体化学等分层）。两者都仍需事先决定盐、溶剂合物（solvate）、tautomer 与 protonation 的标准化规则。

### 指纹：ECFP 与 RDKit Morgan

fingerprint 把局部子结构映射为稀疏标识或 bit/count vector。ECFP（Extended-Connectivity Fingerprint）是 Rogers 与 Hahn（2010）描述的一类扩展连通性指纹；RDKit 的 Morgan fingerprint 是基于 Morgan 式迭代原子标识的可配置实现。二者思想相同（ECFP4 大致对应 RDKit 半径 2 的 Morgan 指纹，因为 ECFP 的数字表示直径），但不应在不说明参数时当作完全同义词。

### Morgan 指纹的生成过程

Morgan fingerprint 的生成可以理解为“逐层扩大每个原子看到的邻域”：

1. **初始化**：每个原子从 initial invariant（初始不变量，如元素、连接数、形式电荷、是否在环中）得到一个整数标识；
2. **迭代**：第 $r$ 轮中，每个原子把自己上一轮的标识与所有邻居的（键类型，邻居标识）一起哈希，得到新标识；这个新标识代表以该原子为中心、半径为 $r$ 的子结构；
3. **收集**：所有轮次、所有原子的标识构成该分子的特征集合；
4. **折叠（可选）**：对位长取模，映射到固定长度的 bit vector（如 2048 位）。

需要注意：

- 固定长度 hashed bit/count fingerprint 会发生碰撞（collision），即不同子结构落到同一位；未折叠的稀疏标识（sparse identifier）fingerprint 不存在这种固定 bit 位碰撞；
- 结果必须记录 atom/feature invariant、bond type、chirality（是否考虑手性）、radius、count 还是 binary、折叠长度和软件版本。

下面的简化实现演示了这一过程：

```python
import hashlib

def _stable_identifier(value):
    """把可序列化描述稳定映射到 64-bit 整数，避免 Python hash 随机化。"""
    digest = hashlib.blake2b(repr(value).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def morgan_fingerprint(atom_labels, bonds, radius=2, n_bits=2048):
    """从显式分子图构造简化 binary Morgan fingerprint。

    atom_labels 可包含元素、formal charge、aromaticity 等 invariant；
    bonds 为 (u, v, bond_label)。本函数不负责解析或标准化 SMILES。
    """
    if radius < 0 or n_bits <= 0:
        raise ValueError("radius must be nonnegative and n_bits positive")
    neighbors = [[] for _ in atom_labels]
    for left, right, bond_label in bonds:
        if not (0 <= left < len(atom_labels) and 0 <= right < len(atom_labels)):
            raise ValueError("bond endpoint is outside atom range")
        neighbors[left].append((right, bond_label))
        neighbors[right].append((left, bond_label))

    identifiers = [_stable_identifier(label) for label in atom_labels]
    bits = {identifier % n_bits for identifier in identifiers}
    for shell in range(1, radius + 1):
        updated = []
        for atom, identifier in enumerate(identifiers):
            environment = tuple(sorted(
                (str(bond_label), identifiers[neighbor])
                for neighbor, bond_label in neighbors[atom]
            ))
            updated.append(_stable_identifier(
                (shell, identifier, environment)
            ))
        identifiers = updated  # 同步更新，避免依赖原子遍历顺序。
        bits.update(identifier % n_bits for identifier in identifiers)
    return bits
```

代码要点：

- 用 `blake2b` 而不是 Python 内置 `hash`，因为后者对字符串的哈希在每次进程启动时随机化，会让同一分子在不同运行中得到不同指纹；
- 邻居环境 `environment` 先排序再哈希，保证结果与原子编号顺序无关（图同构不变）；
- 每轮先算完全部新标识再统一替换（同步更新），否则先更新的原子会影响后更新的原子；
- `identifier % n_bits` 就是折叠步骤，返回的是“被置 1 的位”的集合。

该实现只用于说明“逐层扩张局部环境，再折叠到 bit vector”的思想，不等同于 RDKit 的 Morgan/ECFP。成熟实现还定义了原子 invariant、键类型、chirality、重复环境去重（不同中心原子覆盖相同子结构时只保留一个）与 feature invariant（按药效团角色而非元素标记原子，即 FCFP）；同一分子只有在这些参数和标准化过程一致时，指纹才能相互比较。

### Tanimoto 相似度

两枚 binary fingerprint $A,B$（视为置 1 位的集合）的 Tanimoto 相似度为

$$
T(A,B)=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}
$$

分母 $|A|+|B|-|A\cap B|$ 正是 $|A\cup B|$（容斥原理），所以对二值指纹而言 Tanimoto 系数与集合的 Jaccard 系数相同，取值在 0（无共同位）到 1（完全相同）之间。

```python
def tanimoto(bits_a: set[int], bits_b: set[int]) -> float:
    """计算两枚 binary fingerprint 的 Tanimoto/Jaccard 相似度。"""
    union = bits_a | bits_b
    # 两个空集合的 1.0 是显式约定，调用方可按任务改成 NaN。
    return len(bits_a & bits_b) / len(union) if union else 1.0
```

当两枚 fingerprint 都为空时，集合公式为 $0/0$，代码返回 1.0 是显式的软件约定，不是公式推导的结果。

### 相似度的使用误区

- 相似度依赖 fingerprint 定义、半径、bit 长度与标准化规则；盐、tautomer、stereochemistry 和 protonation 处理不统一，会使同一化学实体看起来不同。
- Tanimoto 只比较 bit 重叠。值 0.7 的含义随 fingerprint 类型与化学空间而变（例如 Morgan 半径 2 下 0.7 已相当相似，而在子结构键类指纹上可能只是一般相似），不能作为普适的结构相似阈值。
- <strong>活性悬崖（activity cliff）</strong>指结构非常相似、活性却差异巨大的分子对（如一个甲基的差别使活性下降百倍）。它说明“相似结构有相似活性”的相似性原理（similarity principle）只是经验规律。

<br>

## 定量构效关系（QSAR）

### 流程

QSAR（Quantitative Structure–Activity Relationship）从结构特征预测活性或性质，本质上就是分子上的监督学习。典型流程是：分子标准化（molecule standardization）→ 描述符/指纹（descriptor/fingerprint）→ scaffold split → 模型训练 → 适用域（applicability domain）与不确定性评估。

### 活性标签：pIC50 及其含义

连续活性常转换为对数尺度：

$$
pIC_{50}=-\log_{10}(IC_{50}/\mathrm{mol\,L^{-1}})
$$

换算前必须统一为 mol/L。例如 $IC_{50}=10\,\mathrm{nM}=10^{-8}\,\mathrm{mol/L}$，则 $pIC_{50}=8$；数值越大活性越强。取对数的原因是活性跨越多个数量级，且测量误差近似在对数尺度上均匀。

需要理解 IC50 的含义：它是在指定 assay 条件下达到 50% 抑制所需的浓度，受底物/酶浓度、孵育时间、读出方式（readout）和作用机制影响。它不等于解离常数 $K_d$，也不普遍等于抑制常数 $K_i$。只有在特定的竞争性抑制与稳态假设下，才能用 Cheng–Prusoff 关系连接 IC50 与 $K_i$（对竞争性抑制，$K_i=IC_{50}/(1+[S]/K_m)$）。因此不同实验室、不同底物浓度下测得的 IC50 不能直接混用。

### 数据整理

QSAR 数据集要统一化学实体、salt form、stereochemistry、tautomer、protonation、单位、靶蛋白构建体（target construct）、assay 形式与 endpoint。

- **删失值（censored activity）**：`>`、`<` 等标注（如“IC50 > 10 µM”，表示在最高测试浓度下仍未达到 50% 抑制）不是普通精确值；直接把阈值当作测量值会引入偏差，应使用区间标签、删失回归（censored regression），或至少做敏感性分析（sensitivity analysis）；
- **重复测量**：同一分子多次测量时，要报告聚合规则（均值、中位数、几何平均）和实验变异。

### 拆分方式

随机拆分会把共享同一骨架（scaffold，如 Bemis–Murcko 骨架）的近邻分到训练集与测试集两侧，模型只需“插值”就能预测准确，从而显著高估对新骨架的泛化能力。更严格的 scaffold split 或时间切分更接近真实的药物发现场景——药物化学家需要的是对尚未合成的新骨架的预测。

### 适用域与不确定性

适用域（applicability domain）描述 query 分子是否位于训练集覆盖的化学空间内。最近邻相似度（nearest-neighbor similarity）、集成模型分歧（ensemble disagreement）与保形预测（conformal prediction）都可以提供不确定性线索；模型对 out-of-domain 分子给出的高置信预测不应直接作为筛选依据。

模型评价除 scaffold/time split 外，还应报告测试分子到训练集的 nearest-neighbor similarity、applicability domain 与 scaffold-level bootstrap，从而区分“同一骨架附近的插值（interpolation）”与“真正的新骨架外推（extrapolation）”。

<br>

## 虚拟筛选与分子对接

### 两类方法

虚拟筛选（virtual screening）在实验之前用计算方法从大库中挑出值得测试的分子：

- <strong>基于配体（ligand-based）</strong>的方法依赖与已知活性分子的相似性（如上文的 Tanimoto 搜索、药效团匹配），不需要靶点结构；
- <strong>基于结构（structure-based）</strong>的 docking 在受体结合口袋中搜索 ligand 的 pose，并用打分函数（scoring function）排序。

docking score 是近似的相对排序，不是严格的结合自由能，也不能替代实验 assay。打分函数为了速度大量简化了溶剂化、熵和柔性效应。

### 影响 docking 结果的因素与验证

docking 同时包含两个子问题：构象搜索（pose search）与打分（scoring），任何一个失败都会导致错误结论。蛋白质子化状态、结合位点水分子（binding-site water）、金属离子、侧链柔性和配体 tautomer 都会改变结果。

常见验证方法有两种：

- **redocking**：把共晶结构中的已知 ligand 拿出来重新对接，检查预测 pose 与晶体 pose 的 RMSD（通常 < 2 Å 视为成功），验证 pose search；
- **active–decoy benchmark**：混入已知活性分子与物化性质相近的诱饵分子（decoy，如 DUD-E 数据集），检查活性分子是否在排序前列富集（enrichment），验证打分排序能力。

需要注意：redocking 成功仍不保证对新 scaffold 的排序准确——它只说明在已知结合模式附近能找回正确 pose。

> [!NOTE]+ 小结
> - 分子是带标签的图；SMILES/InChI 是线性标识，canonicalization 是软件约定，比较前必须统一盐、tautomer、质子化与立体化学。
> - Morgan/ECFP 指纹逐层哈希原子邻域再折叠，折叠会碰撞；必须记录半径、位长、invariant 与软件版本。
> - Tanimoto = 交集/并集；0.7 等阈值不普适，activity cliff 说明相似性原理只是经验规律。
> - QSAR 标签用 pIC50（先统一为 mol/L），IC50 依赖 assay 条件，不等于 $K_d$ 或 $K_i$；删失值需特殊处理；scaffold/time split 才能评估新骨架泛化。
> - docking 分为 pose search 与 scoring，分数只是近似排序；用 redocking RMSD 与 active–decoy 富集验证。

<br>

<br>

# 药物遗传学与药物基因组学（Pharmacogenomics）

同一种药、同一剂量，用在两个人身上效果可能截然不同：有人几乎无效，有人出现严重出血或致命的过敏反应。造成差异的因素有年龄、肝肾功能、合并用药，也有相当一部分来自基因。药物遗传学研究的就是这部分：基因型怎样改变药物暴露（血药浓度有多高、维持多久）、疗效和不良反应，并把这种知识转化成处方时可以执行的建议。Stanford 的 Russ Altman 长期从事这一方向的工作，Stanford 维护的 PharmGKB 知识库就是药物基因组学证据的主要汇总平台之一。

核心想法可以用一条证据链概括：一个变异先改变某个蛋白的分子功能（例如一个代谢酶活性下降），再改变药代或药效表型（药物浓度升高、靶点反应改变），进而改变临床结局（出血、血栓、过敏），最后才可能成为指南里一条“遇到这种基因型应该怎样用药”的建议。链条上每一环都需要证据，统计关联只是起点。

本章把前面几章串起来：全基因组关联研究提供发现关联的统计框架，变异效应与致病性预测提供“变异改变了什么分子功能”的解释，化学信息学提供药物本身的结构与靶点信息。药物遗传学把这三者落到具体的药和病人身上。

## 研究对象：药代动力学、药效动力学与相关基因

### 药物遗传学与药物基因组学的区别

药物遗传学（pharmacogenetics）关注特定基因变异如何影响药物反应，通常从一个或少数几个候选基因出发；药物基因组学（pharmacogenomics）通常覆盖全基因组尺度，例如用 GWAS 或全基因组测序寻找影响药物反应的位点。两个词在实际使用中经常混用，缩写都写作 PGx。所研究的表型包括三类：药代动力学指标（如血药浓度、清除率）、疗效（如是否达到治疗目标）与不良反应（如出血、超敏反应）。

### 药代动力学与药效动力学

药物从进入人体到产生效果分为两段，基因变异可以作用于任意一段：

- **药代动力学（pharmacokinetics, PK）** 描述机体怎样处理药物，即 ADME 四个过程：吸收（absorption）、分布（distribution）、代谢（metabolism）、排泄（excretion）。PK 回答的问题是“给了这个剂量，体内的药物（或活性代谢物）浓度会有多高”。
- **药效动力学（pharmacodynamics, PD）** 描述药物怎样影响机体，即药物浓度怎样作用于靶点并转化为临床表型。PD 回答的问题是“这个浓度下，靶点和病人会有什么反应”。

一个直观的划分：代谢酶和转运体的变异主要改变 PK（同样剂量下浓度不同），药物靶点和受体的变异主要改变 PD（同样浓度下反应不同），而 HLA 变异改变的是免疫系统是否把药物（或药物–多肽复合物）识别为“异物”，属于另一类机制。

### 主要的基因类别

- **CYP 酶（细胞色素 P450）**：肝脏中负责大量药物代谢的酶家族，如 CYP2D6、CYP2C19、CYP2C9。功能下降可使原形药物蓄积（毒性风险升高），也可使需要活化的前药（prodrug）无法转化为活性形式（疗效下降）。
- **转运体**：负责药物进出细胞，例如 SLCO1B1 影响他汀类药物进入肝细胞，其功能下降与肌病风险相关。
- **受体与药物靶点**：例如华法林的靶点 VKORC1，其表达水平影响维持剂量。
- **HLA（人类白细胞抗原）**：部分 HLA 等位基因与特定药物的严重超敏反应强相关。

这些变异可以改变剂量需求，也可以改变毒性风险。

### 分析的输入与输出

药物遗传学分析的输入可能是单个基因型（genotype）、定相单倍型（phased haplotype）、star allele、拷贝数（copy number），也可能是全基因组数据；同时还需要结合具体药物、剂量、适应证和患者的临床变量。输出不是简单的“安全/危险”二分，而是代谢表型（metabolizer phenotype）、风险的变化幅度，或者特定 guideline 下的用药建议（例如“换用替代药”“起始剂量下调”）。

### 临床检测的目标与决策

临床上做药物基因检测，目标可能是：解释一次已经发生的异常药物反应、选择替代药物、调整起始剂量，或者降低严重药物不良反应（adverse drug reaction, ADR）的风险。

是否检测并不是“有关联就测”。需要考虑适应证、效应量、检测周转时间（结果能否在开药前返回）、是否有可替代的药物，以及相应的指南。存在 gene–drug association 不表示所有患者在所有用药场景都应常规筛查。检测的组织方式也有两种：

- **预先检测（pre-emptive panel）**：在需要之前一次性检测一组药物基因，结果存入病历，以后开任何相关药物时都能直接用；
- **反应性检测（reactive test）**：在开具某个药物时才针对性检测。

两者在成本、覆盖范围和结果返回时机上不同。反应性检测的常见问题是结果返回太慢，医生已经按常规方案开药；预先检测前期成本较高，但能覆盖未来多种处方。

## 基因型怎样表示：从 SNP 剂量到 star allele 与代谢表型

### 为什么不能只用 0/1/2

GWAS 中最常见的编码是把一个双等位 SNP 的效应等位基因个数记为 0、1、2。但很多药物基因的结构远比单个 SNP 复杂。以 CYP2D6 为例，它存在拷贝数变异（copy-number variation, CNV，例如整个基因缺失或重复多份）、与相邻假基因 CYP2D7 重组形成的杂合等位基因（hybrid allele），以及由多个变异共同定义的 star allele。简单的 SNP dosage 不能完整表示代谢表型。

### Star allele 与 diplotype

药物基因学界用 star allele 命名法描述一个基因的单倍型：`*1` 通常表示参考（正常功能）单倍型，`*2`、`*3`、`*4` 等是由一个或多个特定变异组合定义的单倍型，每个 star allele 被赋予一个功能分类（正常、降低、无功能、增加）。一个人的两条染色体各带一个 star allele，组合起来称为二倍型（diplotype），例如 CYP2C19 `*1/*2`。

从测得的变异推导 diplotype，需要知道变异位于哪条染色体上（phasing）；未定相（unphased）数据可能对应多种 diplotype。

### 从 diplotype 到表型：activity score

对部分 CYP 基因，CPIC 等共识采用活性分数（activity score）：给每个 star allele 赋一个数值，两个等位基因相加，再按区间映射到代谢表型。以 CYP2D6 为例（2019 年共识）：无功能等位基因（如 `*4`）记 0，功能降低的 `*41` 记 0.5、`*10` 记 0.25，正常功能（如 `*1`、`*2`）记 1；基因重复时数值按拷贝数相乘。总分为 0 对应差代谢者，$0<\text{AS}<1.25$（实际取值 0.25–1）对应中间代谢者，$1.25\le\text{AS}\le2.25$ 对应正常代谢者，大于 2.25 对应超快代谢者。

这里要注意两点：star allele/diplotype 到功能表型的映射是 gene-specific 的，不同基因规则不同；而且映射会随共识更新（例如 CYP2D6 `*10` 的分值就在 2019 年被下调）。因此报告结果时必须写明所用的映射版本。

### 表型命名的适用范围

poor / intermediate / normal / rapid / ultrarapid metabolizer（差、中间、正常、快、超快代谢者）这套命名适用于部分 CYP 基因，例如 rapid metabolizer 用于 CYP2C19（如 `*1/*17`）。它不是 HLA、转运体和受体的统一分类：HLA 结果是“携带/不携带某个风险等位基因”，转运体常用“功能正常/降低/差”描述。

## 遗传关联分析

### 回归模型

发现 gene–drug 关联的基本工具与 GWAS 相同：广义线性模型。对明确的双等位 SNP，把效应等位基因的剂量编码为 $G\in\{0,1,2\}$，表型 $Y$（如稳定维持剂量、是否发生 ADR）满足

$$
g(E[Y])=\beta_0+\beta_gG+\beta^\top C
$$

其中 $g$ 是连接函数（link function）：连续表型（如对数剂量）用恒等连接，二值表型（如是否出血）用 logit 连接。$\beta_g$ 是每多一个效应等位基因带来的效应；在 logit 连接下，$e^{\beta_g}$ 就是每个等位基因的比值比（odds ratio）。$C$ 是协变量向量，包含 ancestry principal components（祖源主成分，用于校正人群分层）、年龄、性别、合并用药等；$\beta$ 是对应系数。

这个模型的偏差来源包括：人群分层（population stratification，祖源同时影响等位基因频率和用药反应，产生假关联）、稀有变异、小样本（药物反应队列通常远小于疾病 GWAS 队列），以及表型定义不稳（例如“疗效不佳”的判定标准在不同研究中不一致）。

### 复杂基因型的编码

多等位的 HLA、定相单倍型（phased haplotype）、star allele、结构变异（structural variant）与 CNV 需要用类别变量、单倍型或 activity score 表示。强制把它们压成单个 SNP dosage 会丢失功能信息：例如 CYP2D6 基因重复的个体在 SNP 芯片上可能看起来和普通杂合子一样，但实际代谢能力大不相同。

### 全基因组检验的额外问题

在全基因组尺度做检验，还要控制样本间的亲缘关系（relatedness）、人群结构（population structure）与基因型填补质量（imputation quality）。稀有变异单点检验的统计功效（power）很低，常见做法是按基因把多个稀有变异聚合起来，做 burden 检验（把变异计数加总成一个分数）或 SKAT 类检验（允许不同变异效应方向不同）。但要意识到，聚合规则本身也构成模型假设：哪些变异算“有害”、权重怎么给，都会影响结果。

### 关联不等于可行动

统计显著的 association 不等于临床可行动性（actionability）。从关联到处方建议，还需要独立复制、足够的效应量、机制证据、风险收益权衡、指南级别和实施环境。两个关键资源分工不同：

- **PharmGKB**（Stanford 维护；2025 年起与 CPIC、PharmCAT 一同整合进 ClinPGx 平台，原 pharmgkb.org 链接重定向到 clinpgx.org）汇总并分级 gene–drug 证据，回答“这个关联的证据有多强”；
- **CPIC**（Clinical Pharmacogenetics Implementation Consortium）发布用药指南（并入 ClinPGx 后指南仍沿用 CPIC 名称），主要回答“基因结果已经可用时应如何解释和用药”，通常不负责决定是否订购检测。

从数据到处方的完整证据链为

$$
\text{variant}
\rightarrow\text{molecular consequence}
\rightarrow\text{PK/PD phenotype}
\rightarrow\text{clinical outcome}
\rightarrow\text{actionable guideline}
$$

每一步都可能受祖源、合并用药（如强 CYP 抑制剂会让正常代谢者表现得像差代谢者，称为 phenoconversion）、肝肾功能、年龄和适应证调节。因此 GWAS significance 不能直接替代剂量建议：一个 $p<5\times10^{-8}$ 的位点可能效应很小，也可能在某个适应证下无临床意义。

## 三个临床实例

下面三个例子分别代表三种不同的遗传机制：前药活化失败、免疫介导的超敏反应、多基因加临床变量共同决定剂量。

### CYP2C19 与氯吡格雷（clopidogrel）

氯吡格雷是抗血小板药，常用于冠脉支架术后和部分卒中患者。它是前药，本身没有活性，需要经肝脏 CYP2C19 等酶活化。`*2`、`*3` 是常见的无功能等位基因，`*17` 是功能增加的等位基因。

在某些适应证下（例如急性冠脉综合征行 PCI），中间或差代谢者形成的活性代谢物较少，血小板抑制不足，缺血事件风险可能升高。CPIC 2022 年更新扩展了适应证，并提高了对中间代谢者建议的强度：在无禁忌时考虑换用替代的 P2Y12 抑制剂（如普拉格雷 prasugrel、替格瑞洛 ticagrelor）；该指南不支持对中间代谢者改用每日 150 mg 的加倍剂量。需要注意替代药自身的禁忌，例如普拉格雷禁用于有卒中/TIA 史的患者。

完整流程是：由检测到的 star allele 推导 diplotype，再映射为 phenotype，最后按当前 CPIC guideline、适应证和替代药禁忌给出建议。该结论不能推广成“所有患者都必须检测”。

### HLA-B*57:01 与阿巴卡韦（abacavir）

阿巴卡韦是抗 HIV 药物。携带 HLA-B*57:01 的患者用药后发生超敏反应的风险显著升高，因此在开始用药前筛查该等位基因，阳性者避免使用阿巴卡韦。

这个例子的结构与 CYP 完全不同：HLA-B*57:01 表示特定 HLA allele 对超敏反应风险的影响，属于类别型的免疫风险（categorical immune risk）——携带即高风险，与“携带一个还是两个”关系不大；而 CYP activity score 是连续的代谢模型，等位基因功能可以相加。

### 华法林（warfarin）

华法林是抗凝药，治疗窗窄：剂量过高导致出血，过低导致血栓，不同患者的稳定剂量可以相差十倍以上。华法林剂量同时涉及：

- **CYP2C9**：代谢活性更强的 S-华法林。`*2`、`*3` 等功能降低的等位基因使清除变慢，需要更低剂量；
- **VKORC1**：华法林的靶点。启动子变异 −1639G>A（rs9923231）降低表达，携带者对华法林更敏感，需要更低剂量；
- **CYP4F2**：`*3` 携带者需要略高的剂量（CPIC 2017 建议可选地增加 5–10%）；
- 还有祖源、年龄、体表面积、合并用药（如胺碘酮）等临床变量。

CPIC 2017 指南推荐用已发表的药物基因学剂量算法（如 IWPC 算法与 Gage 算法，后者可在 warfarindosing.org 使用）计算剂量。常见遗传变异与已知非遗传因素合计约解释 50% 的剂量变异；不同基因的贡献依祖源而异，例如在非裔人群中，CYP2C9 的 `*5`、`*6`、`*8`、`*11` 及 rs12777823 等变异更重要（CPIC 2017 建议：携带上述 CYP2C9 等位基因者在算法剂量基础上下调 15–30%，rs12777823 A/G 或 A/A 者下调 10–25%），只测欧洲人群常见位点会低估风险；若非裔患者只有 CYP2C9 `*2`、`*3` 结果，该指南不建议做基因型指导剂量。Altman 与 Klein 等人参与的国际华法林药物基因组学联盟（IWPC）用多国数据建立了这类算法。

### 三个例子共同说明的问题

这几个例子共同说明：PGx 不能统一简化成单 SNP 的 0/1/2 dosage。检测覆盖哪些等位基因、是否做了 phasing、是否检测 CNV、是否能识别稀有等位基因（rare allele），以及所依据的 guideline 版本，都属于结果的一部分，必须随结果一起报告。

> [!NOTE]+ 小结
> - PK 描述机体处理药物，PD 描述药物作用于机体；CYP、转运体、受体、HLA 分别作用于不同环节。
> - 药物基因的基因型常需用 star allele、diplotype、activity score 或类别变量表示，映射规则因基因而异并随共识更新。
> - 关联分析用 $g(E[Y])=\beta_0+\beta_gG+\beta^\top C$，但要控制人群分层、亲缘关系和稀有变异功效问题。
> - 统计关联到用药建议要经过分子后果、PK/PD 表型、临床结局和指南；PharmGKB 分级证据，CPIC 解释已有结果。
> - 氯吡格雷（前药活化）、阿巴卡韦（免疫风险）、华法林（多基因加临床变量）代表三种不同机制。

<br>

# 网络生物学（Network Biology）

细胞里的分子并不单独工作：转录因子调控靶基因，蛋白质结合成复合物，代谢物通过酶促反应彼此转化。很多问题天然是“关系”问题，例如：一个新发现的疾病基因和哪些已知通路相连？在一张蛋白互作图上，哪些蛋白构成功能模块？已知几个疾病基因，下一个最值得实验验证的候选是谁？网络生物学把这些问题写成图论问题：基因、蛋白质、代谢物或疾病表示为节点，调控、物理结合、代谢反应、共表达或文献关联表示为边。

核心想法是“关系里有信息”：功能相关的分子在网络中往往相邻或处于同一稠密区域，因此可以用图结构做模块发现（社区检测）和候选排序（网络传播）。但要时刻记住，网络指标是给定图上的数学量，图本身是不完整、有偏的测量结果。

本章承接药物遗传学和知识表示：药物–靶点、基因–疾病关系都可以组织成图；下一章的文献计算则是网络中一类重要边（文献关联）的来源。

## 网络的组成：节点、边与边的语义

### 输入与输出

网络分析的输入通常是带类型、方向、权重和来源的边列表（edge list），以及种子基因（seed gene）或节点注释（annotation）。输出可以是：节点中心性（centrality）、社区（community）、传播排名或候选连接（candidate link）。

### 不同的边代表不同的关系

基因调控、蛋白互作、代谢和信号网络都可表示为图 $G=(V,E)$，$V$ 是节点集合，$E$ 是边集合。但边可能有方向、符号（激活/抑制）、权重、时间和证据来源。常见的几类边语义差别很大：

- **蛋白互作（PPI）** 常是无向的物理关系；
- **转录调控** 具有方向（谁调控谁）和符号（激活还是抑制）；
- **共表达** 只表示统计相关，两个基因同升同降，未必有直接作用；
- **文献关联** 只表示两个实体在文本中被一起提及或被描述存在关系。

即使同样叫 protein interaction，也可能来自 yeast two-hybrid（检测二元直接结合）、affinity purification（亲和纯化质谱，检测的是复合物共存，不一定直接接触）、co-expression 或文本挖掘，它们测量的关系不同。STRING 等数据库会把多种证据整合为一个综合置信分，BioGRID 则按实验方法分别记录相互作用；使用时应保留证据类型。

因此，有向的调控边（directed regulatory edge）、无向的物理相互作用（undirected physical interaction）与带符号的激活/抑制边（signed activation/inhibition）不应在不标注的情况下混入同一个邻接矩阵（adjacency matrix）。把不同来源的边不加区分地合并为无权图，会丢失机制和证据。

## 节点层面的网络指标

### 常见量

| 指标 | 含义 | 直觉 |
|---|---|---|
| 度（degree） | 局部连接数 | 这个节点有多少邻居 |
| 介数（betweenness） | 所有节点对之间的最短路径中经过该节点的比例 | 是否处于“交通要道” |
| 聚类系数（clustering coefficient） | 邻居之间彼此相连的程度 | 邻居是否抱团 |
| 社区（community） | 内部连接较密的节点组 | 功能模块的候选 |
| PageRank / random walk | 在网络上传播重要性 | 被“重要节点”指向的节点也重要 |

### 解释时的三个陷阱

- **度分布受确认偏倚（ascertainment bias）影响**：热门基因被研究得更多，于是在数据库里有更多已知相互作用，degree 往往更高。高 degree 可能反映的是“研究得多”，而不是“生物学上更中心”。
- **介数依赖最短路径假设**：betweenness 假设信息沿最短路径传播，但生物信号未必沿唯一的最短路径传播。
- **中心性不等于必需性**：centrality 是图上的数学量，不自动等于基因必需性（essentiality）或生物重要性。

## 社区检测与模块度

### 动机

蛋白复合物、信号通路在网络上表现为“内部连得密、对外连得疏”的节点组。社区检测（community detection）要把节点划分成这样的组。问题是“密”需要一个参照：一个高 degree 节点本来就和很多节点相连，它和某组节点之间有边并不稀奇。模块度（modularity）的思路是：把实际的社区内连边，和“保持每个节点 degree 不变、但随机连边”时的期望社区内连边相比较，多出来的部分才算社区结构。

### Newman–Girvan 模块度

对无向、无权图，设 $A_{ij}$ 为邻接矩阵元素（有边为 1，否则为 0），$k_i=\sum_j A_{ij}$ 为节点 $i$ 的度，$m$ 为边数（故 $\sum_i k_i=2m$），$c_i$ 为节点 $i$ 所属社区，$1[\cdot]$ 为指示函数。community detection 常优化

$$
Q=\frac{1}{2m}\sum_{i,j}
\left(A_{ij}-\frac{k_ik_j}{2m}\right)1[c_i=c_j]
$$

第二项 $k_ik_j/(2m)$ 是保持度分布的零模型（degree-preserving null）下 $i$ 与 $j$ 之间的期望边数：节点 $i$ 有 $k_i$ 个“边端点”，每个端点随机连到图中 $2m$ 个端点之一，连到 $j$ 的概率约为 $k_j/(2m)$。于是括号内是“实际减期望”，只对同社区的节点对求和，最后除以 $2m$ 归一化。$Q$ 越大，说明社区内的边比随机期望多得越多；把所有节点放进一个社区时 $Q=0$。

推广到其他图：加权图应以节点强度（strength，即相连边权之和）和总权重替代 degree 与 $m$；有向图还需分别使用入度与出度，期望项变为 $k_i^{\text{out}}k_j^{\text{in}}/m$ 的形式。

> [!EXAMPLE]+ 例1 两个三角形由一条边相连
> 节点 0、1、2 构成一个三角形，节点 3、4、5 构成另一个三角形，再用边 (2,3) 相连，共 $m=7$ 条边。若划分为 $\{0,1,2\}$ 与 $\{3,4,5\}$，每个社区内部有 3 条边，度之和为 $2+2+3=7$。用等价形式 $Q=\sum_c\left[\frac{L_c}{m}-\left(\frac{d_c}{2m}\right)^2\right]$（$L_c$ 为社区内边数，$d_c$ 为社区内度之和）得
>
> $$
> Q=2\left[\frac{3}{7}-\left(\frac{7}{14}\right)^2\right]=\frac{5}{14}\approx 0.357
> $$
>
> 若把 6 个节点全部放进一个社区，则 $Q=0$。用下面的 `modularity` 函数也能得到同样的数值。

### 模块度的局限

- **分辨率极限（resolution limit）**：在大图中，优化 $Q$ 可能把本应分开的小社区合并，因为小社区的期望项太小、合并后 $Q$ 反而更高。代码中的 `resolution` 参数（常记作 $\gamma$）乘在期望项上，调大它倾向于得到更多、更小的社区。
- **近简并最优解**：存在大量 $Q$ 值几乎相同、但划分差别明显的解。
- **高 $Q$ 不证明生物真实性**：即使在随机图上也能找到 $Q$ 不低的划分。

因此应报告所用的 resolution、随机种子、多次运行的一致性，以及独立的功能验证。

### 模块度计算代码

```python
import numpy as np

def modularity(adjacency, communities, resolution=1.0):
    """计算无向加权图在给定 community assignment 下的 modularity。"""
    A = np.asarray(adjacency, dtype=float)
    communities = np.asarray(communities)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.allclose(A, A.T) or np.any(A < 0):
        raise ValueError("this implementation requires an undirected graph")
    if communities.shape != (len(A),):
        raise ValueError("one community label is required per node")

    strength = A.sum(axis=1)  # 加权度（strength）k_i
    twice_total_weight = strength.sum()  # 2m
    if twice_total_weight == 0:
        return 0.0
    same = communities[:, None] == communities[None, :]  # 1[c_i = c_j]
    expected = np.outer(strength, strength) / twice_total_weight  # k_i k_j / 2m
    score = (
        (A - resolution * expected) * same
    ).sum() / twice_total_weight
    return float(score)
```

该函数评价一个已有的划分（partition），不负责搜索最优划分。搜索最优划分是 NP 困难问题，实际使用启发式算法：

- **Louvain**：反复把单个节点移动到能提高 $Q$ 的邻居社区，直到不再改进；再把每个社区收缩成一个超节点（supernode），在收缩后的图上重复。
- **Leiden**：在 Louvain 基础上增加细化步骤，进一步改善社区连通性（Louvain 可能产生内部不连通的社区）。

两者都是启发式优化，结果依赖随机顺序。应以多个随机种子检查近简并解，而不是把一次最高 $Q$ 当作唯一结构。

## 随机游走重启与网络传播

### 动机

社区检测回答“网络里有哪些模块”，而疾病基因优先级排序回答另一个问题：已知若干疾病基因（seed），网络中哪些其他基因与它们“最近”？只看直接邻居太局部，只看最短路径又太脆弱。随机游走重启（Random Walk with Restart, RWR）的想法是：让一个游走者从 seed 出发，每一步以概率 $1-r$ 沿边走到随机邻居，以概率 $r$ 跳回 seed。长期来看，游走者停留在各节点的概率就是该节点与 seed 集合的“网络接近度”。离 seed 近、且有多条路径相连的节点得分高。

### 迭代公式

令 $W$ 为 $n\times n$ 的列归一化邻接矩阵（每列之和为 1，$W_{ij}$ 是从 $j$ 走到 $i$ 的概率），$p_t$ 为第 $t$ 步时游走者位于各节点的概率向量，$p_0$ 为 seed 上的初始分布（seed 节点非零、和为 1），$r\in(0,1]$ 为重启概率。迭代

$$
p_{t+1}=(1-r)Wp_t+rp_0
$$

第一项是“继续走一步”，第二项是“跳回 seed”。

### 稳态解的推导

设迭代收敛到 $p^{\ast}$，则 $p^{\ast}=(1-r)Pp^{\ast}+rp_0$（这里把 $W$ 记作 $P$，强调它是与列向量约定一致的列随机转移矩阵 column-stochastic transition matrix）。移项得 $\left[I-(1-r)P\right]p^{\ast}=rp_0$。由于 $P$ 列随机，其谱半径为 1，$(1-r)P$ 的谱半径不超过 $1-r<1$（要求 $r>0$），故 $I-(1-r)P$ 可逆，收敛解满足

$$
p^{\ast}=r\left[I-(1-r)P\right]^{-1}p_0
$$

其中 $0<r\le1$。把逆矩阵展开成 Neumann 级数 $\sum_{k\ge0}(1-r)^kP^k$ 可得到直观解释：$p^{\ast}$ 是所有长度 $k$ 的游走分布 $P^kp_0$ 的加权平均，权重 $r(1-r)^k$ 随长度几何衰减。由此可见 restart $r$ 越大越局部（长路径权重小），越小传播越远。

实际大图不显式求逆（$O(n^3)$ 且破坏稀疏性），而以 power iteration 更新，稀疏矩阵乘法每轮约 $O(|E|)$；误差每轮至少按因子 $1-r$ 缩小。

> [!EXAMPLE]+ 例2 路径图上的 RWR
> 四个节点连成路径 0–1–2–3，seed 为节点 0。$r=0.3$ 时稳态得分约为 $(0.4240, 0.3543, 0.1642, 0.0575)$；$r=0.7$ 时约为 $(0.7338, 0.2255, 0.0354, 0.0053)$。重启概率越大，得分越集中在 seed 附近，远端节点 3 的得分从 0.0575 降到 0.0053。

### 实现

```python
import numpy as np

def random_walk_restart(
    W, seed, restart=0.3, tol=1e-10, max_iter=10_000
):
    """在列向量约定下以 power iteration 计算 RWR stationary score。"""
    W = np.asarray(W, dtype=float)
    p0 = np.asarray(seed, dtype=float)

    # 邻接权重与 seed 都代表概率质量，不能含负数。
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    if not np.isfinite(W).all() or np.any(W < 0):
        raise ValueError("W must be finite and nonnegative")
    if (
        p0.shape != (W.shape[0],)
        or not np.isfinite(p0).all()
        or np.any(p0 < 0)
        or p0.sum() <= 0
    ):
        raise ValueError("seed must be nonnegative with positive sum")
    if not 0 < restart <= 1:
        raise ValueError("restart must be in (0, 1]")
    if tol <= 0 or max_iter <= 0:
        raise ValueError("tol and max_iter must be positive")

    # seed 先归一化为 restart distribution。
    p0 = p0 / p0.sum()
    colsum = W.sum(axis=0)
    P = np.zeros_like(W)
    non_dangling = colsum > 0
    P[:, non_dangling] = W[:, non_dangling] / colsum[non_dangling]
    # 无出边节点把全部概率质量送回 seed，避免概率泄漏。
    P[:, ~non_dangling] = p0[:, None]

    p = p0.copy()
    for _ in range(max_iter):
        nxt = (1 - restart) * (P @ p) + restart * p0
        # L1 距离直接衡量两轮 probability mass 的变化。
        if np.linalg.norm(nxt - p, ord=1) < tol:
            return nxt
        p = nxt
    raise RuntimeError("RWR did not converge")
```

函数接受未归一化的权重矩阵 `W`，在内部按列归一化得到 `P`。对于没有出边的节点（悬挂节点，dangling node），如果不处理，概率质量会在每一步漏掉；这里把它们的整列设为 $p_0$，相当于游走者走到死胡同时直接回到 seed。

### 用途与偏差

该方法可做疾病基因优先级排序或功能注释传播。但结果会偏向高 degree 节点（游走者更容易经过连接多的节点），而高 degree 部分来自研究热度。因此网络的研究偏倚（study bias）、缺失边与数据库整合误差都必须评估。

## 富集分析、连接预测与零模型

### 模块的富集分析

发现模块后再做 GO enrichment（检验模块中是否某个 GO 功能注释过度出现）会涉及双重选择和依赖检验：模块是用数据挑出来的，再用同一网络（甚至同一共表达数据）去检验它，显著性会被高估。例如共表达模块富集某个 GO term，可能只是重复使用了构图数据。

### 连接预测的时间泄漏

连接预测（link prediction）要防止时间泄漏。如果用当前数据库随机藏掉一些边作为测试集，模型可能借助后来才发现的共同邻居“作弊”——那些共同邻居本身就是在发现被藏的边之后才加入数据库的。更严格的方法是按发现时间（discovery time）切分：用某年之前的网络训练，预测之后才被报道的边。

另一个问题是负样本：未观察到的边只是未标注（unlabeled），不等于确认不存在，因此负采样（negative sampling）方式会影响评价结果。

### 零模型：置换要保留什么结构

判断一个网络统计量是否显著，需要和零模型（null model）比较。置换必须保留与待检验统计量相关的结构，而不是随意打乱所有边：

- 无向 PPI 图可使用保持度的重连（degree-preserving rewiring）：反复随机选两条边 $(a,b)$、$(c,d)$，交换为 $(a,d)$、$(c,b)$，每个节点的度不变；
- 调控网络需分别保持入度与出度；
- 药物–靶点二部图还要保持节点类型（边只能连在药物与靶点之间）；
- 带时间或符号的网络需保留相应属性。

### 疾病基因排序的评价

疾病基因排序应与以下基线和方案比较，以确认性能不是由研究热度或数据库泄漏造成：

- **degree-matched baseline**：只按 degree 排序（或与 degree 匹配的随机基因）能达到多少性能；
- **leave-one-seed-out validation**：每次隐藏一个已知疾病基因，看它在其余 seed 传播下的排名；
- **time split**：用较早的网络与 seed 预测之后才被发现的疾病基因。

网络算法输出的是给定图与 seed 下的 ranking，不等于新的实验因果证据。结果需要适当零模型、独立数据和实验验证。

> [!NOTE]+ 小结
> - 网络边有方向、符号、权重和来源，PPI、调控、共表达、文献边语义不同，不应无标注地合并。
> - degree、betweenness 等中心性是图上的数学量，受研究热度影响，不等于 essentiality。
> - 模块度比较社区内实际边与保持度零模型期望；存在 resolution limit 和近简并解，Louvain/Leiden 需多种子检查。
> - RWR 的稳态解为 $p^{\ast}=r[I-(1-r)P]^{-1}p_0$，$r$ 越大越局部，实际用 power iteration。
> - 零模型要保持相关结构；评价要用 degree-matched baseline、leave-one-seed-out 和 time split。

<br>

# 文献计算（Biomedical Text Mining）

大量生物医学知识最初只以自然语言形式存在于论文里：PubMed 收录的文献已达数千万篇，任何人都不可能读完。数据库里的许多基因–疾病、药物–靶点关系，最初也是人工从论文里摘录的。文献计算要做的是把论文中的自由文本转换为可检索、可计算的实体和关系，以便扩大筛选范围、构建知识图谱、为下游分析提供先验。

核心处理链是三步：先识别文本中的 gene、protein、disease、drug 等实体提及（mention）；再把不同写法链接到稳定的数据库 ID；最后从句子或篇章中抽取实体之间的关系及其证据。难点在于语言的歧义：字符串相同不保证实体相同，同一实体也可能有多个缩写和历史名称。

本章连接数据检索、知识表示和网络生物学：文献产生的边最终会进入网络或知识图谱，所以必须保留出处和句子级证据，并与实验测得的 interaction 区分；进入知识图谱前还要统一本体（ontology）、物种（organism）和标识符命名空间（identifier namespace）。

## 输入、输出与解释边界

文献计算的输入是检索 query、摘要或全文；输出是 PMID 集合、规范化实体、关系边、证据句和置信度。

自动抽取适合扩大筛选范围（先用机器从大量文献中召回候选，再由人工核查），但受命名歧义、否定、实验语境和发表偏倚（publication bias）影响，不能把模型抽取出的关系直接当成数据库事实。

## 处理流程

[2018 年 Stanford Daily 的课程报道](https://stanforddaily.com/2018/11/06/classy-classes-cs-274-serves-as-a-rigorous-introduction-to-bioinformatics/)提到，课程内容包括以自然语言处理调查论文并辅助蛋白分类。典型流程包括：

1. PubMed 检索与去重；
2. 分句（sentence splitting）与分词（tokenization）；
3. 基因/蛋白/化学物质的命名实体识别（named entity recognition, NER）；
4. 实体规范化（entity normalization）到稳定 ID，例如基因到 NCBI Gene ID，疾病到 MeSH 或 MONDO；
5. 关系抽取（relation extraction）；
6. 证据句（evidence sentence）与来源信息（provenance）保存。

PubTator 等公开系统就是按这一流程对 PubMed 做预先标注的例子。

同名、缩写和物种上下文是主要困难。例如同一个基因符号在人和小鼠中对应不同的 ID。即使抽到“蛋白 A 与疾病 B 相关”，还需标记这句话是肯定、否定还是推测，实验对象是什么（细胞系、小鼠还是病人），关系方向如何（A 导致 B，还是 B 中 A 升高）。知识图谱中的边应保留来源句、PMID、方法、置信度和实体对。

### NER 与规范化是两个问题

NER 确定提及在文本中的跨度（mention span），即“哪几个字符是一个基因名”；规范化把这个提及链接到数据库 ID，即“它是哪个基因”。两者可以分别出错。

字符串完全匹配做规范化有两类错误：会漏掉别名（同一基因有多个历史名称），也会把 `CAT` 这类普通英文单词误连到 gene（CAT 也是过氧化氢酶基因的符号）。更好的规范化器（normalizer）可结合文内缩写定义（如 “catalase (CAT)” 这种括号定义）、物种、邻近实体与候选先验（candidate prior，某个名字通常指哪个 ID）。

### 句子级与篇章级关系抽取

句子级（sentence-level）方法只看同一句里的实体对，会遗漏跨句证据（例如上一句引入基因，下一句用“it”描述其作用）；篇章级（document-level）模型覆盖更广，但负样本构造更难——一篇文章里两两实体组合很多，哪些算“确实没有关系”难以界定。

### 分阶段评价

评价应分别报告实体提及/实体层面的 precision 与 recall、规范化准确率（normalization accuracy）和关系 F1，不能只给端到端的单一分数。只有一个总分时，无法知道错误出在哪一环。

### 文献数据本身的偏倚

文献数据还带发表偏倚：阳性结果、热门基因和常见疾病更容易发表。数据库可能重复收录同一实验。因此文本出现次数不等于效应量（effect size），缺乏文献也不证明不存在关系。

## 文献检索

文档检索（document retrieval）与信息抽取（information extraction）应分开评价。检索阶段要保存 query、数据库及其版本、检索日期、字段限制（如只检索标题/摘要）、去重规则与纳入排除标准，并报告 precision@k（前 $k$ 篇中相关文献的比例）、召回率（recall）或人工抽样误差。只评价抽取模型而不评价检索召回，会把“论文未被找到”误当作“论文没有关系”。

不同检索方式各有取舍：Boolean query（如 `CYP2C19 AND clopidogrel`）具有高可解释性，却难覆盖新别名；dense retrieval（把 query 和文献编码为向量后按相似度检索）能找到语义近似的文献，但可能牺牲精确控制。系统综述式任务通常组合受控词表（controlled vocabulary，如 MeSH）、关键词、引文追踪（citation chasing）与人工复核。

## 标注规范与数据划分

训练和评价抽取模型需要人工标注的语料。关系标签（relation label）至少应区分肯定、否定、推测、比较、实验对象、物种、方向和证据类型（evidence type）。标注指南（annotation guideline）要事先定义：

- 实体边界采用精确匹配（exact span）还是重叠即算匹配（overlap span）；
- 嵌套实体（nested entity，如 “BRCA1 mutation carriers” 中的 “BRCA1”）如何处理；
- 缩写与共指（coreference）是否合并为同一实体；
- 同句中出现但未声明关系的实体对是否算负样本；
- 跨句和跨段证据的最大范围；
- 标注者分歧（annotator disagreement）的处理与仲裁（adjudication）流程。

数据划分同样重要：随机按句切分会让同一论文的近重复句同时出现在训练和测试中，高估性能。应按 PMID、研究或发表时间拆分。评价指标上，规范化可报告 top-1 accuracy 与 recall@k（正确 ID 是否在前 $k$ 个候选中），关系抽取则同时给出 pipeline 条件（使用自动识别的实体）与 gold-entity 条件（使用人工标注的实体）下的结果，二者之差反映了上游 NER 错误传递造成的损失。

## 证据管理与解释

知识图谱中的每条边应保留原句、PMID、section（摘要、结果还是讨论）、entity ID、model version、confidence 与 extraction date。需要注意两点：

- 模型置信度是预测校准量，表示“模型认为这句话表达了该关系的概率”，不是证据强度；
- 同一实验被综述（review）或数据库重复引用，不能当作独立复制。

### 与其他章节的关系

蛋白语言模型处理的是生物序列 token（氨基酸），文献语言模型（如 BioBERT、PubMedBERT）处理的是自然语言。两者都使用 Transformer，不代表二者语义或评价标准相同。

放到全课程的视角下：序列分析提供候选与演化证据，结构分析提供空间机制，功能数据（包括文献）提供细胞和临床上下文。三类证据可以相互校验。

> [!NOTE]+ 小结
> - 处理链：检索 → 分句分词 → NER → 规范化 → 关系抽取 → 证据保存。
> - NER 定跨度，规范化定 ID；字符串匹配既漏别名又误连普通词。
> - 检索与抽取分开评价；关系抽取需区分肯定、否定、推测、物种和方向。
> - 按 PMID 或时间划分数据；分别报告各阶段指标和 pipeline/gold-entity 结果。
> - 置信度不等于证据强度，出现次数不等于效应量，缺乏文献不等于没有关系。

<br>

# 综合实践：课程项目与可复现分析

前面各章分别介绍了序列、结构、功能和网络的表示与算法。真实的生物信息学工作要把它们组成一条完整的分析流程，而这时最常见的失败往往不是某个算法写错，而是：数据版本记不清、ID 映射悄悄丢了一半基因、交叉验证时同一个病人出现在训练和测试两边、几个月后无法重现图表。综合实践关注的就是怎样把分散的表示、算法和统计检验组成可复现的分析。

组织原则是：重点不是把所有工具串在一起，而是从明确的生物问题出发，每一步都规定输入、输出、假设、测试和失败条件。下面的序列、结构与功能三个项目分别覆盖离散动态规划（DP）、三维几何和多源数据整合。

关于复现与正确性，有两条基本要求。第一，项目结果应能由命令行和固定环境从原始 accession 重建；Notebook 可用于探索，但最终 workflow 需要记录参数、版本、随机种子、checksum 和中间数据。第二，正确性同时包括算法输出、数据语义和统计解释三个层面：代码运行完成不代表分析成立。

## 三个项目方向

[2018 年 Stanford Daily 课程报道](https://stanforddaily.com/2018/11/06/classy-classes-cs-274-serves-as-a-rigorous-introduction-to-bioinformatics/)记载，当年课程设置三个较大型 Python 项目，主题为序列分析、三维结构和功能分析，另有四个概念作业。以下实践按这三个方向整理，不对应原作业题面。

### 序列项目：双序列比对

**目标。** 实现 global（全局，Needleman–Wunsch）、local（局部，Smith–Waterman）和 semi-global（半全局，两端空位不罚分）alignment，支持 linear gap 与 affine gap，并比较参数对结果的影响。

**输入与输出。** 输入包括多条 DNA/protein FASTA、替换矩阵和参数文件；输出包括 score、alignment、类似 CIGAR 的操作串（CIGAR-like operation，如 `5M1I3M` 表示 5 个匹配、1 个插入、3 个匹配）与运行统计。DNA 可使用简单的 match/mismatch 评分，蛋白质使用 BLOSUM62。

**验证。** 测试覆盖存在多个最优解的情形、空序列和错误输入；随机序列用于校准分数（判断一个比对得分是否显著高于随机），Biopython 用于交叉验证。实验记录序列长度增加时的时间和内存（标准 DP 为 $O(nm)$），并用偏离主对角线的已知 alignment 检查带状比对（banded alignment，只计算主对角线附近一条带内的 DP 格子）何时遗漏最优解：当最优路径走出带宽时，带状比对得到的就不再是全局最优。

### 结构项目：结构叠合

**目标。** 从 mmCIF/PDB 读取 $C_\alpha$ 坐标，实现 Kabsch 叠合，并比较 RMSD、distance-map score 和 TM-score。

**输入与输出。** 输入要记录 accession、assembly（生物组装体）、chain、model 与 atom selection，同时处理缺失残基和 alternate location（同一原子的多个可选构象）。输出包括变换矩阵（transformation matrix）、RMSD、coverage（参与叠合的残基比例）、逐残基距离、叠合后坐标和离群残基图；报告中要注明残基对应关系（correspondence）的来源（来自序列比对还是结构比对）。

**验证。** 测试包括已知的旋转/平移（叠合后应恢复原坐标、RMSD 为 0）、反射（reflection，Kabsch 需检查行列式符号，避免给出镜像而非旋转）、共线点（此时旋转不唯一）、缺失残基和两个结构域（domain）的相对转动（整体 RMSD 很大，但各结构域单独叠合很好）。RMSD 需要与 coverage 和逐残基误差一起报告，因为只叠合一小部分残基就能得到很低的 RMSD。

### 功能项目：多源数据整合

**目标。** 整合 UniProt、GO 和 bulk RNA-seq 数据，完成差异表达、GO enrichment 及分类模型比较。

**输入与实现。** 输入固定版本的 UniProt accession、GO annotation、count matrix 与 sample metadata。实现分四步：

1. 先审计标识符映射（identifier mapping），分别记录一对多（one-to-many）、无法映射（unmapped）、已废弃（obsolete）和异构体（isoform）映射的数量；
2. 再构建设计矩阵、完成差异表达；
3. 在预先定义的背景基因集（background）上进行 enrichment，并做 BH（Benjamini–Hochberg）多重检验校正；
4. 训练一个线性 baseline 与一个非线性模型并比较。

**验证。** 病人（patient）、同源簇（homology cluster）或时间单位不得跨训练/测试划分；预处理（标准化、特征选择等）只在训练折（training fold）上拟合；BH 的检验总数和过滤过程需要留档（先过滤低表达基因会改变检验总数，从而改变校正结果）。敏感性分析包括移除高 degree 节点和更换 GO release，检查结论是否稳健。整个报告应能在空环境中由一条 workflow command 重建。

## 测试

算法测试应覆盖：可手算的 toy case、空序列、全 mismatch、重复字符、平局时的取舍（tie-breaking）、数值下溢（如 HMM 中概率连乘）、`-inf`（对数空间中概率为 0）、坐标反射和格式错误的记录（malformed record）。除单元测试（unit test）外，还需要与成熟库做基于性质的对照（property-based test，如“比对得分对交换两条序列不变”），并记录运行时间和内存随输入规模的变化。

课程项目强调面向对象（OOP）与动态规划实现。类应负责维护清晰的数据不变量（例如“序列只含合法字母”），而不是把所有函数机械地包进对象。DP 代码则必须在 docstring 或邻近注释中说明：状态（每个格子代表什么）、递推、初始化、traceback 起点与停止条件。即使最终实现仍有错误，完整注释也能暴露推理链并获得部分评分；缺少注释的偶然正确输出不能证明算法被正确理解。

## 可复现的项目结构

一个推荐的目录结构如下：

```text
project/
├── README.md
├── pyproject.toml
├── environment.yml
├── src/
├── tests/
├── notebooks/
├── data/
│   ├── raw/
│   └── processed/
├── results/
└── metadata/
    ├── accessions.tsv
    └── checksums.sha256
```

源码（`src/`）、测试（`tests/`）、探索性 notebook、原始与处理后数据、结果和元数据各自分开。README 应写明问题、数据许可、下载命令、版本、运行入口和预期产物。原始数据库不宜直接提交到仓库（体积大，且可能有许可限制）；应保存 accession、query、日期、checksum 与处理脚本，使任何人都能重新下载并验证得到的是同一份数据。

## 交付物

完整项目的交付内容包括：

- 可执行的 Snakemake/Nextflow workflow；
- lock file 或 container digest（固定确切的依赖版本或镜像）；
- 数据字典（data dictionary）与来源清单（provenance manifest）；
- unit、integration 和 property-based test。

随机种子、无法消除的非确定性（nondeterminism，如 GPU 运算顺序）、图表、指标和失败报告均由流程记录。Model card 或分析说明列出适用边界、已知偏倚（known bias）与未解决问题。

Notebook 用于探索，不作为唯一执行入口。最终结果由脚本从原始 accession 重建；手工复制、界面点击和未记录的 spreadsheet edit 不属于可复现流程。

单次课程作业至少应达到一个更小的交付基线：PEP 8 风格、模块与函数 docstring、关键逻辑的 inline comment、`argparse`/`sys.argv` 命令行入口、环境文件和最小测试。需要区分两个层级：Conda 环境隔离解释器与依赖，lock file 或 container 再固定解析后的确切版本；二者解决的问题不同，前者防止项目之间互相干扰，后者保证换一台机器、过一段时间仍装出同样的版本。

> [!NOTE]+ 小结
> - 从明确的生物问题出发，每一步规定输入、输出、假设、测试和失败条件。
> - 序列项目验证多最优解、空序列和带状比对的边界；结构项目报告 RMSD 时同时给 coverage 和逐残基误差；功能项目先审计 ID 映射、防止数据泄漏。
> - DP 代码要注明状态、递推、初始化和 traceback；测试覆盖边界情形并与成熟库对照。
> - 结果应由一条 workflow 命令从原始 accession 重建；Notebook 只用于探索。

<br>

<br>

# 复习提要：算法、统计与证据链

前面各章分别讲了序列、演化、结构、表达、变异、化学与网络。复习时最容易出现的问题是“每章都背得出名词，但串不起来”：知道 BLAST 有 E-value、GSEA 有 FDR、AlphaFold 有 pLDDT，却说不清它们在统计意义上分别回答什么问题，也说不清一个分数在什么条件下可以跨数据集比较。

本章把全课内容按两个维度重新组织。第一个维度是**计算结构**：许多看上去无关的方法，背后是同一种算法范式（例如序列比对、HMM 和 RNA 折叠都是动态规划）。第二个维度是**统计与证据链**：任何方法的输出都要经过“表示—目标—近似—校准—解释”这条链才能变成结论。最后按主题列出核心公式、复杂度和易错点，便于考前逐项自查。

<br>

## 用算法结构把全课串起来

### 七类计算结构

课程中的算法可以归入下面几类计算结构。识别出一个新问题属于哪一类，往往就能直接联想到可用的技巧和它的代价。

| 计算结构 | 课程中的代表 | 共同特点 |
| --- | --- | --- |
| **动态规划（dynamic programming）** | 序列比对、HMM、RNA folding、树似然 | 把最优解（或总概率）分解为子问题，按顺序填表；精确但通常是多项式级别的二次或三次代价 |
| **索引与启发式搜索（indexing and heuristic search）** | BLAST、$k$-mer、结构 motif | 先用预建索引快速筛出候选，再在少量候选上做精细计算；以灵敏度换速度 |
| **图算法（graph algorithms）** | 组装、ontology、分子图、网络传播 | 对象之间的关系显式化为顶点与边，问题转化为路径、遍历、传播或社区划分 |
| **交替与连续优化（alternating and continuous optimization）** | STRUCTAL 的对应/叠合、能量最小化、模型训练 | 固定一部分变量优化另一部分，或沿梯度下降；通常只保证局部最优 |
| **概率推断（probabilistic inference）** | HMM、系统发育、表达 count、关联 | 写出生成模型或似然，再做极大似然、后验或假设检验 |
| **表示学习（representation learning）** | 蛋白质语言模型与结构预测 | 用大规模数据学习向量表示，下游任务在表示上进行 |
| **统计校准（statistical calibration）** | E-value、经验 null、bootstrap、FDR、留出集评估 | 把原始分数翻译成“在零模型下出现的概率”或“在新数据上的表现” |

需要注意，同一个方法可以同时落在多类里。HMM 的 Forward/Viterbi 是动态规划，参数学习是概率推断；BLAST 的 seed 阶段是索引搜索，E-value 是统计校准。

### 分析任何方法时的统一检查清单

无论是课程里讲过的方法，还是考试或项目中遇到的新方法，都可以用下面七个问题来拆解。它们对应一条从数据到结论的链条，任何一环说不清，最后的结论就站不住。

1. **输入表示**：决定哪些信息被保留、哪些信息已经丢失。例如 contact map 丢掉了距离大小，gene set 丢掉了通路方向。
2. **目标函数**：决定算法实际优化的量。Nussinov 最大化配对数而不是最小化自由能，两者的最优结构可以不同。
3. **计算近似**：说明相对精确算法舍弃了什么。BLAST 相对 Smith–Waterman 舍弃了“没有高分 seed 的同源”；progressive MSA 相对精确多维 DP 舍弃了全局最优保证。
4. **分数解释**：只有给出统计校准后，才可能跨数据集解释。原始比对分数依赖矩阵、gap 参数和数据库大小，不能直接比较。
5. **零模型（null model）**：明确“随机”或“背景”具体指什么。打乱序列、打乱标签、degree-preserving 随机网络、全基因组背景，各自回答不同问题。
6. **证据层级**：区分相似性、关联、机制和因果。序列相似不等于功能相同，GWAS 关联不等于 causal variant，预测有害不等于临床致病。
7. **可复现性**：记录数据版本、软件版本和参数。同一个命令换一个 database release 可能得到不同结果。

### 按期末串讲的依赖链自查

按课程期末串讲复习时，可以沿下面的依赖链逐项检查是否真正贯通。后一环依赖前一环的概念：没有理解测量过程，就无法判断表达数据的噪声；没有理解比对，就无法理解 MSA、系统发育和共进化结构预测。

1. **生物学对象与测量**：复制、转录、翻译、表观遗传；RT-qPCR、microarray、RNA-seq。
2. **序列**：global/local/affine alignment，BLAST 九步，MSA，1D motif，HMM。
3. **演化**：distance、molecular clock、parsimony、likelihood。
4. **结构**：torsion、RMSD 与 correspondence，STRUCTAL，force field，CASP、Rosetta、AF1/AF2。
5. **功能统计**：GWAS、GSEA、clustering/silhouette、GO、AUROC 与 AP。
6. **临床解释**：deleteriousness、population/clinical database、phenotype、PK/PD 与 pharmacogenomics。

这不是另一套章节顺序，而是从考试问题回到前文原理的索引。每一项都应能回答四个问题：**输入是什么、优化或检验什么、输出如何校准、哪些结论不能由该输出推出**。如果某一项只能说出名称而答不出这四个问题，就是需要回到对应章节重读的地方。

### 核心算法复杂度速查

下表汇总课程中主要算法的时间与空间代价（$m,n$ 为序列长度，$L$ 为 RNA 长度，$N$ 为 HMM 状态数，$T$ 为观测长度，$M$ 为 profile 长度，$s$ 为字符状态数）。这些是标准实现的量级，具体常数和优化版本见各章。

| 算法 | 时间 | 空间 | 备注 |
| --- | --- | --- | --- |
| Needleman–Wunsch / Smith–Waterman | $O(mn)$ | $O(mn)$ | 只要分数时可用 $O(\min(m,n))$ 空间；Hirschberg 分治可在线性空间内恢复比对 |
| Gotoh 仿射 gap | $O(mn)$ | $O(mn)$ | 三个矩阵，常数约为线性 gap 的 3 倍 |
| 精确多序列 DP（$k$ 条序列，长度 $n$） | $O(2^k n^k)$ | $O(n^k)$ | 指数级，因此实际使用 progressive 启发式 |
| KMP 精确匹配 | $O(n+m)$ | $O(m)$ | 失败函数预处理 $O(m)$ |
| FM-index 计数查询（模式长 $m$） | $O(m)$ 次 rank 查询 | 压缩后接近文本熵 | 定位每个命中还需 suffix array 采样 |
| de Bruijn 图 Eulerian 路径 | $O(\lvert E\rvert)$ | $O(\lvert E\rvert)$ | Hierholzer 算法；真实组装还要处理错误与重复 |
| Forward / Viterbi | $O(N^2T)$ | $O(NT)$ | Forward 只要总概率时可只存一列 |
| Profile HMM 搜索 | $O(MT)$ | $O(MT)$ | 每个 profile 位置只有 M/I/D 三状态，转移稀疏 |
| Nussinov | $O(L^3)$ | $O(L^2)$ | 枚举分叉点 $k$ 带来第三重循环 |
| Zuker / McCaskill | $O(L^3)$ | $O(L^2)$ | 需限制 internal loop 大小（常用 30），否则 internal loop 项为 $O(L^4)$ |
| UPGMA / Neighbor Joining（$n$ 个类群） | $O(n^3)$ | $O(n^2)$ | 每轮扫描距离矩阵，共 $n-2$ 轮 |
| Felsenstein pruning（$n$ 类群、$k$ 位点） | $O(nks^2)$ | $O(nks)$ | 对 DNA $s=4$；位点独立假设使各列可分开计算 |
| Kabsch（$n$ 个对应点） | $O(n)$ | $O(n)$ | 求 $3\times3$ 协方差矩阵后做常数规模 SVD |
| Random Walk with Restart（power iteration） | 每轮 $O(\lvert E\rvert)$ | $O(\lvert V\rvert+\lvert E\rvert)$ | 收敛轮数随 restart 概率增大而减少 |

<br>

## 表示与数据

同一个生物对象可以用多种方式表示，而表示一旦选定，能回答的问题也就被限定了。这一节的复习重点是“分数的含义”和“实体的身份”：很多错误不是算法写错，而是把两种含义不同的分数混用，或者把不同版本坐标下的同一名字当成同一实体。

### 表示选择决定可回答的问题

- 同一生物对象可以表示为序列、图、坐标或概率分布。例如一个蛋白质可以是氨基酸字符串、残基接触图、$C_\alpha$ 坐标集合，或是 AlphaFold 输出的 distogram（残基对距离的概率分布）。
- 各种表示保留不同信息，也对应不同可回答问题。字符串适合比对和搜索，坐标适合叠合和对接，图适合传播与社区检测，概率分布适合表达不确定性。

### 两种质量分数不能混用

FASTQ 和 SAM 文件中都有“质量”，但含义完全不同：

- **FASTQ base quality** 表示碱基判读误差，即测序仪对“这一位是 A 还是 C”的把握，Phred 刻度为 $Q=-10\log_{10}P(\text{base call 错误})$。
- **SAM mapping quality（MAPQ）** 表示定位不确定性，即“这条 read 放在这个基因组位置是错的”的概率，同样采用 Phred 刻度，但事件不同。

两者不能混用。一条每个碱基质量都很高的 read，如果落在重复序列中，MAPQ 可以接近 0；反之，一条有几个低质量碱基的 read 也可能唯一地比对到某处。

### 版本与实体

- Genome build（如 GRCh37 与 GRCh38）、坐标约定（0-based 半开区间还是 1-based 闭区间）、isoform 和 database release 都参与结果定义。同一个“chr7:117,559,590”在不同 build 下指向不同碱基。
- 实体解析错误会继续传播到表达、结构和功能分析。基因符号被 spreadsheet 自动转成日期、旧 symbol 与新 symbol 混用、isoform 选错导致残基编号偏移，这些错误在下游往往无法察觉。

<br>

## 序列算法

序列章节的核心是两类问题：**精确查找**（模式在不在文本中、在哪里）和**近似比对**（两条或多条序列怎样对齐才最能反映同源关系）。前者依赖巧妙的索引结构，后者依赖动态规划和打分系统；数据库搜索则把两者结合起来，再加上统计校准。

### 精确索引与组装

- **KMP** 利用模式串的前后缀：预先计算模式串每个前缀的“最长真前缀兼后缀”长度，失配时不回退文本指针，因此总时间 $O(n+m)$。
- **FM-index** 利用 BWT 和 rank 查询：BWT 把文本的轮转按字典序排序后取最后一列，相同上下文的字符聚在一起，既易压缩又支持 backward search；每扩展一个模式字符只需两次 rank 查询。BWA、Bowtie 等短读比对器都基于此。
- **$k$-mer index** 利用固定长度词：把文本切成所有长度为 $k$ 的子串建立哈希表，查询时先找共享 $k$-mer。BLAST 的 seed 阶段也是这一思想。
- **de Bruijn 图** 将组装写成 Eulerian traversal：以 $(k-1)$-mer 为顶点、$k$-mer 为边，基因组对应一条经过每条边恰好一次的路径。相比把 read 作为顶点、要求经过每个顶点一次的 Hamiltonian 路径（NP-hard），Eulerian 路径可在线性时间内求得。

### 双序列比对

全局比对（Needleman–Wunsch）的标准递推为

$$
F_{i,j}=\max\begin{cases}
F_{i-1,j-1}+s(x_i,y_j)\\
F_{i-1,j}+g\\
F_{i,j-1}+g
\end{cases}
$$

其中 $F_{i,j}$ 是前缀 $x_1\cdots x_i$ 与 $y_1\cdots y_j$ 的最优比对分数，$s(x_i,y_j)$ 是替换矩阵（如 BLOSUM62）中的得分，$g$ 是线性 gap 分数，与双序列比对一章一致写成负数（每个 gap 字符加 $g$），边界为 $F_{i,0}=ig$、$F_{0,j}=jg$。局部比对（Smith–Waterman）在 $\max$ 中再加一项 $0$，并把边界设为 0，允许比对从任意位置重新开始。

- **三种比对的区别**：全局、局部和半全局比对的区别集中在初始化、终点和 traceback 停止条件。全局比对第一行第一列按 gap 罚分累加、终点在右下角；局部比对边界为 0、终点取全表最大值、traceback 遇到 0 停止；半全局比对不罚端部 gap，终点取最后一行或最后一列的最大值。
- **仿射 gap**：用多个状态区分 gap-open 与 gap-extension。Gotoh 算法维护三个矩阵（匹配状态、$x$ 中 gap、$y$ 中 gap），仍是 $O(mn)$。
- **罚分约定**：必须注明长度为 $k$ 的 gap 罚分是 $g_o+kg_e$ 还是 $g_o+(k-1)g_e$ 的约定。两种写法只差一个 $g_e$，但会改变表格中的具体数值，抄错约定是考试中最常见的失分点。
- **Pair HMM**：把相同三状态 DP 解释为隐路径概率。Match 状态发射一对字符，两个 Insert 状态各发射一条序列中的字符；Viterbi 给出最可能比对，Forward 给出对所有比对求和的概率，后者可用于计算每对残基的后验对齐概率。

> [!WARNING]+ 双序列比对易错点
> - 局部比对的 traceback 必须在遇到 0 时停止，而不是走到左上角。
> - 替换矩阵分数是对数几率，期望得分必须为负，局部比对才不会退化为全长比对。
> - 仿射 gap 中 open 与 extend 的约定、以及是否允许 $x$ 中 gap 紧接 $y$ 中 gap，都会影响结果，答题时应先写明。

### 数据库搜索与多序列比对

- **BLAST 用 seed 筛选候选**：先找 query 与数据库序列之间得分超过阈值 $T$ 的短词命中，再向两侧做无 gap 延伸，得分足够高时再做带 gap 的延伸。代价是可能漏掉没有高分短词的远缘同源。
- **E-value 校准随机高分**：在随机序列模型下，得分不低于 $S$ 的局部比对的期望个数为 Karlin–Altschul 公式

$$
E=Kmn e^{-\lambda S}
$$

其中 $m,n$ 分别为 query 长度和数据库总长度，$K,\lambda$ 由打分系统与背景频率决定。换算为 bit score 后不再依赖 $K,\lambda$：

$$
S'=\frac{\lambda S-\ln K}{\ln 2},
\qquad E=mn\,2^{-S'}
$$

这意味着 E-value 随数据库变大而线性增大：同一比对在更大的数据库里显著性更低。当 $E$ 较小时，至少出现一次此类命中的概率 $P=1-e^{-E}\approx E$。

- **Progressive MSA 依次合并 sequence/profile**：先算两两距离、建 guide tree，再按树从叶到根逐步把序列或 profile 合并。其根本局限是“once a gap, always a gap”——早期合并引入的错误不会在后续步骤中被修正。
- **缓解早期错误**：可通过 consistency information（如 T-Coffee 利用三元组一致性改写打分，使合并时参考所有序列的信息）和 iterative refinement（如 MUSCLE、MAFFT 的迭代模式，反复把比对拆成两组再重新对齐）来缓解。

<br>

## 概率与演化

这一部分的共同语言是概率模型：motif 用位置特异的分布描述，HMM 用隐状态链描述序列的生成，系统发育用替换过程描述性状在树上的演化，RNA 结构则在能量模型下讨论结构的概率分布。复习时应特别注意每个算法“求和”还是“求最大”，以及模型的独立性假设。

### Motif 与 HMM

- **PWM 假设 motif 各位置条件独立**：位置 $i$ 上字符 $a$ 的得分通常为 $\log\frac{p_i(a)}{q(a)}$，$q$ 为背景频率，整个窗口的得分是各位置得分之和。位置之间的依赖（如成对共变）无法表示。
- **OOPS/ZOOPS/TCM 决定每条序列允许多少次出现**：OOPS（one occurrence per sequence）要求每条序列恰有一次，ZOOPS（zero or one）允许零次或一次，TCM（two-component mixture）允许任意多次。选错模型会使 MEME 等方法的似然和 motif 估计产生偏差。
- **Forward 计算总似然**：$P(x)=\sum_{\pi}P(x,\pi)$，递推中用求和。
- **Viterbi 求最可能路径**：$\pi^*=\arg\max_\pi P(x,\pi)$，递推中用最大值并记录回溯指针。
- **Baum–Welch 用期望计数更新参数**：E 步用 Forward–Backward 计算每个转移和发射在当前参数下的期望使用次数，M 步把期望计数归一化为新参数；它是 EM 算法，只保证似然单调不减并收敛到局部极大。
- **Profile HMM 的 delete state 不发射字符**：因此必须沿 profile 坐标在同一观测行传播，即从 $D_{j-1}$ 或 $M_{j-1}$ 到 $D_j$ 时，观测指针 $i$ 不前进。实现时如果像 match 一样把 $i$ 加一，就会算错。

> [!TIP]+ 数值稳定性
> 长序列上概率连乘会下溢。Viterbi 直接在对数空间做加法即可；Forward 需要 log-sum-exp 或逐列缩放（scaling），不能简单地把求和改成对数后相加。

### 系统发育

- **UPGMA 假设 molecular clock**：每次合并距离最小的两簇，新簇到其他簇的距离取按簇大小加权的平均。只有在所有谱系演化速率相同（树是 ultrametric）时才能恢复正确拓扑。
- **Neighbor Joining 使用 $Q$ 准则恢复 additive tree**：

$$
Q(i,j)=(n-2)d(i,j)-\sum_kd(i,k)-\sum_kd(j,k)
$$

每轮合并 $Q$ 最小的一对。减去每个类群到其他所有类群的总距离，修正了“演化快的类群看起来离谁都远”的偏差，因此不需要分子钟假设；只要距离是 additive 的，NJ 就能恢复正确树。

- **Felsenstein pruning 在树上递推祖先状态似然**：对每个位点，从叶到根计算条件似然 $L_v(a)=\prod_{c\in\text{children}(v)}\sum_b P_{ab}(t_c)L_c(b)$（$P_{ab}(t_c)$ 为沿长度 $t_c$ 的分支由状态 $a$ 变为 $b$ 的概率；叶节点观察到 $x$ 时 $L_v(a)=1[a=x]$），根处再按平稳分布加权求和，所有位点的对数似然相加。它把指数级的祖先状态枚举变成多项式时间。
- **Bootstrap support 与 Bayesian clade posterior 来自不同不确定性框架**：bootstrap 对 alignment 的列有放回重抽样，反映“数据再采一次，这个 clade 还会出现多少次”的频率派稳定性；Bayesian posterior 是在先验与模型下该 clade 为真的后验概率。二者数值不可直接互换，posterior 在模型设定错误时常常偏高。

### RNA 结构

- **Nussinov 最大化配对数**：

$$
N_{i,j}=\max\begin{cases}
N_{i+1,j}\\
N_{i,j-1}\\
N_{i+1,j-1}+\delta(i,j)\\
\max_{i<k<j}\,[N_{i,k}+N_{k+1,j}]
\end{cases}
$$

其中 $N_{i,j}$ 是子序列 $i..j$ 的最大配对数，初值 $N_{i,i}=0$；与 RNA 一章一致，当 $i,j$ 可合法配对且满足最小环长时 $\delta(i,j)=1$，否则该候选项取 $-\infty$（不可选）。四项依次对应 $i$ 不配对、$j$ 不配对、$i$ 与 $j$ 配对、在 $k$ 处分叉。它只数配对、不考虑堆积能，因此预测往往不符合实际。

- **Zuker/Turner 最小化 loop-dependent 自由能**：自由能按 loop 分解（stacking、hairpin、bulge、internal、multiloop），参数来自 Turner 实验测定的最近邻模型，堆积的碱基对贡献负自由能。
- **McCaskill 汇总结构 ensemble**：用相同的 loop 分解计算 partition function $Z=\sum_s e^{-\Delta G(s)/RT}$，进一步得到每对碱基的配对概率。当 MFE 结构与次优结构能量相近时，配对概率比单个 MFE 更能反映不确定性。
- 以上方法都不能处理 pseudoknot（交叉配对），这是嵌套结构 DP 的固有限制。

<br>

## 结构与动力学

结构部分围绕“三维坐标”展开：怎样比较两个结构、怎样在结构中找模式、怎样从序列预测结构，以及怎样在力场下研究结构的运动。复习时要分清每种表示丢失了什么信息，以及每个分数（RMSD、TM-score、pLDDT、PAE、能量）回答的是什么问题。

### 坐标、距离与叠合

- 坐标保留完整空间信息，但依赖参考系：同一结构平移或旋转后坐标全变。
- 距离矩阵消除刚体变换但不能区分镜像：$d_{ij}=\lVert x_i-x_j\rVert$ 对旋转、平移和反射都不变，所以一个结构与其镜像有相同距离矩阵。
- contact map 还会丢失距离大小：只记录 $d_{ij}$ 是否低于阈值（如 8 Å）。
- Classical MDS 可从 Euclidean distance matrix 恢复到刚体变换和反射等价的坐标：对平方距离矩阵做双中心化得到 Gram 矩阵 $B=-\tfrac12 JD^{\circ2}J$（$D^{\circ2}$ 为逐元素平方的距离矩阵，$J=I-\tfrac1n\mathbf 1\mathbf 1^\top$ 为中心化矩阵），取前三个特征向量乘以对应特征值的平方根即为坐标。
- Kabsch 在对应已知时求最优刚体变换：两组点先各自中心化，计算协方差矩阵 $H=\sum_i p_iq_i^\top$，做 SVD $H=U\Sigma V^\top$，在列向量约定 $q_i\approx Rp_i+t$ 下最优旋转 $R=V\,\mathrm{diag}(1,1,d)\,U^\top$，其中 $d=\det(UV^\top)\in\{\pm1\}$ 用于排除反射（结构叠合一章使用行向量约定，那里的旋转是此处 $R$ 的转置 $UDV^\top$）。
- 对应未知时，STRUCTAL 类方法在顺序约束 DP 与 Kabsch 之间迭代：固定叠合后用残基间距离构造打分矩阵做 DP 得到对应，固定对应后用 Kabsch 重新叠合，重复至收敛。这是交替优化，结果依赖初始化，只能保证局部最优。
- RMSD 对长度和离群点敏感，TM-score 使用长度归一化和饱和距离权重：

$$
\mathrm{RMSD}=\sqrt{\frac1n\sum_{i=1}^n\lVert Rp_i+t-q_i\rVert^2},\qquad
\text{TM-score}=\max\frac{1}{L_{\text{norm}}}\sum_i\frac{1}{1+(d_i/d_0)^2}
$$

其中求和只对比对上的残基对进行，$\max$ 取遍所有叠合方式，$d_0(L)=1.24\sqrt[3]{L-15}-1.8$ Å 中的 $L$ 取归一化长度 $L_{\text{norm}}$（通常为参考结构长度）。单个大偏差在 RMSD 中按平方放大，在 TM-score 中贡献趋于 0；TM-score $>0.5$ 常作为相同 fold 的经验信号。

### 一维与三维模式

- 1D motif 以 consensus、regex 或 PWM 表示：consensus 最简单但无法表达位置变异，regex（如 PROSITE 模式）可以表达允许字符集合与可变长度，PWM 进一步给出概率权重。
- 3D motif 以残基标签、距离、角度和手性约束表示：例如丝氨酸蛋白酶的 Ser–His–Asp 催化三联体可以写成三个残基类型加上原子间距离约束；手性约束用于区分镜像排列。
- 几何命中要相对完整扫描过程的 null 校准：在整个 PDB 上扫描时，即使随机也会出现若干满足宽松约束的命中，显著性必须相对“同样扫描所有候选残基组合”的背景计算。
- FEATURE 微环境分数、DALI Z-score 和 PWM/BLAST 分数都遵循“特征表示、背景模型、扫描空间、显著性”这条统计链。FEATURE 以位点周围同心球壳中的理化属性为特征、以非位点为背景；DALI 以距离矩阵相似度为特征、以随机结构对的分数分布计算 Z-score；PWM 与 BLAST 以序列得分为特征、以背景频率或随机序列为零模型。

### 结构预测

- **同源建模使用模板与序列比对**：找到序列相似的已知结构作为模板，按比对把坐标迁移过来，再建 loop 和侧链（如 MODELLER 的空间约束满足法）。精度主要由模板与目标的序列一致性决定。
- **threading 评价序列和结构环境的兼容性**：把目标序列“穿”到已知 fold 上，用残基与结构环境（埋藏程度、二级结构、接触）之间的统计势打分，用于序列相似性很低的远缘情形。
- **共进化方法使用 MSA 耦合**：同一蛋白质中空间接触的两个位置倾向于协同突变。直接统计互信息会混入间接耦合，DCA 等全局统计模型（Potts 模型）区分直接与间接耦合，得到更准的接触预测。
- **AlphaFold1** 从 MSA 预测 distogram 与 torsion，再优化学习到的势：神经网络输出残基对距离分布和二面角分布，转成可微的势函数后用梯度下降求结构。
- **AlphaFold2** 联合 MSA、pair representation、模板与 structure module 端到端预测坐标：Evoformer 在 MSA 表示和 pair 表示之间反复交换信息，structure module 以每个残基的刚体框架（frame）直接输出三维坐标，并通过 recycling 迭代细化。
- **置信度解读**：局部 pLDDT 高而结构域间 PAE 高，表示各结构域局部可信但相对取向不确定。pLDDT 是每个残基的局部置信度（0–100），PAE 是“以残基 $i$ 的框架对齐时残基 $j$ 的预期位置误差”，因此 PAE 矩阵的非对角块反映结构域间的相对位置可信度。
- **语言模型分数的含义**：蛋白质语言模型分数学到的是序列分布，不是折叠自由能。高 pseudo-likelihood 表示“像自然序列”，可与稳定性或功能相关，但不能当作热力学量。

### 能量、轨迹与 ensemble

- 能量最小化寻找附近局部极小值：梯度下降或共轭梯度只会把结构推到最近的能量谷，不会跨越能垒，也不等于找到天然态。
- MD 轨迹是给定模型和初态的时间演化：数值积分牛顿方程（常用 velocity Verlet 积分器），时间步长约 1–2 fs，受最快的键振动限制。
- 平衡 ensemble 定义状态概率：在正则系综中状态 $s$ 的概率正比于 $e^{-E(s)/k_BT}$；只有轨迹充分采样时，轨迹上的时间平均才等于系综平均。
- Force field、积分器、thermostat、barostat、约束和采样共同限定可解释的统计量。例如没有 barostat 就不是 NPT 系综；约束了含氢键长才能使用 2 fs 步长；采样不足时，自由能差和稀有事件频率都不可信。

<br>

## 功能与系统

功能与系统部分处理的是“高通量数据产生之后怎么解释”：一张差异表达基因列表、一个 GWAS 信号峰、一个候选变异、一个化合物或一张网络，都需要借助知识库和统计模型才能转化为生物学结论。这里的共同主题是**证据边界**：每一种分析能说明什么、不能说明什么。

### 知识表示与转录统计

- **GO 的三层概念**：GO term 定义概念，annotation 连接 gene product 与 term，evidence code 记录证据类型（如 EXP 为实验证据，IEA 为电子自动推断）。term 之间以 is_a、part_of 等关系构成有向无环图，注释沿 is_a 与 part_of 向上传递到祖先 term（true path rule）。
- **Pathway 与 gene set 的区别**：Pathway 含关系和方向（谁激活谁、谁磷酸化谁），gene set 只是无序成员集合。把 pathway 压成 gene set 做富集分析时，方向和拓扑信息都丢失了。
- **Over-representation analysis 检验阈值化列表**：先按阈值选出“显著基因”，再用超几何检验（Fisher 精确检验）问某个集合在其中是否过多。结果依赖阈值和背景基因集合的选择。
- **GSEA 检查预定义集合是否集中在全基因排序两端**：不设阈值，沿全部基因的排序计算加权 Kolmogorov–Smirnov 式的 running sum，峰值为 enrichment score，显著性用表型标签置换得到。
- **RNA-seq count 的方差结构**：count 具有均值相关方差，适合 negative-binomial model，$\operatorname{Var}(K)=\mu+\alpha\mu^2$，$\alpha$ 为 dispersion。Poisson 模型只有 $\mu$ 一项，会低估生物学重复间的变异。
- **TPM 已经过长度和总量归一化**：$\mathrm{TPM}_i=10^6\cdot\frac{c_i/\ell_i}{\sum_j c_j/\ell_j}$。因此 TPM 适合样本内比较不同基因，但不应作为 DESeq2/edgeR 的输入，后者需要原始 count。
- **Design matrix 不满秩时参数不可识别**：例如批次与处理完全混杂（所有处理组样本都在批次 1），则无法区分处理效应与批次效应，任何软件都无法“校正”出来。
- **三种量的区别**：FDR 描述检验程序的错误发现比例，effect size 描述差异大小，confidence interval 描述估计不确定性。一个基因可以 FDR 极小但 fold change 只有 1.1 倍。

Benjamini–Hochberg 程序：把 $m$ 个 p 值从小到大排序为 $p_{(1)}\le\dots\le p_{(m)}$，找最大的 $k$ 使 $p_{(k)}\le \frac{k}{m}q$，拒绝前 $k$ 个假设，则在独立（或正相关）条件下 FDR $\le q$。

### 关联、聚类与证据边界

- **GWAS 的检验对象**：GWAS 对离散或连续 phenotype 做全基因组关联检验，离散性状常用 logistic 回归或卡方检验，连续性状用线性回归，并加入主成分等协变量控制群体分层。常用全基因组显著性阈值为 $5\times10^{-8}$，大致对应约一百万个独立检验的 Bonferroni 校正。
- **Manhattan peak 受 LD 影响**：连锁不平衡使邻近 SNP 的检验统计量高度相关，一个峰中的多个显著 SNP 往往只反映同一个信号，峰顶 SNP 也不等于独立 causal variant，需要 fine-mapping 和功能实验。
- **多重检验处理的边界**：Bonferroni、FDR 和 gene/pathway binning 控制或重构多重检验问题，但不能把 association 升格为 mechanism。把 SNP 汇总到基因或通路只是改变了检验单位，结论仍是关联。
- **表达聚类需同时说明距离、标准化、算法和簇数**：Euclidean 距离与 $1-$Pearson 相关距离会聚出不同的簇；不做 z-score 标准化时高表达基因主导距离。
- **silhouette 是内部几何指标，不是生物学真实性**：

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}\in[-1,1]
$$

$a(i)$ 是点 $i$ 到同簇其他点的平均距离，$b(i)$ 是到最近的其他簇的平均距离。高 silhouette 只说明簇在所选距离下分得开。

- **评价指标**：AUROC 衡量随机正例得分高于随机负例的概率，对类别不平衡不敏感；AP（average precision）在正例稀少时更能反映前排预测的质量，因此在变异或药物靶点等极不平衡任务中应同时报告。

### 变异解释

- 变异预测需区分 **molecular consequence**（对转录本的影响，如 missense、stop-gained、splice）、**deleteriousness**（对分子功能或适应度的损害程度）和 **clinical pathogenicity**（是否导致某种疾病，按 ACMG/AMP 标准判定）。三者逐级需要更多证据。
- **SIFT 使用 query-specific conservation**：为每个 query 蛋白搜索同源序列并构建比对，计算该位置出现替换后氨基酸的归一化概率，低于 0.05 预测为 damaging。
- **PolyPhen-2 合并序列与结构特征**：用比对保守性、结构位置（埋藏、二级结构等）和其他注释特征训练朴素贝叶斯分类器。
- **FATHMM 使用 profile HMM 与 domain weight**：比较野生型与突变型氨基酸在 profile HMM 中的概率变化，并按蛋白结构域对疾病突变的倾向加权（weighted 版本）。
- **CADD 与 DANN 提供全基因组 deleteriousness 排序**：CADD 用“人类谱系中固定下来的变异”与“模拟的新生变异”训练分类器，输出 raw score 并转换为 PHRED 尺度（PHRED 20 表示位于全部可能 SNV 的前 1%）；DANN 用同样的训练数据换成深度神经网络。
- **患者级优先级**：还要加入 gnomAD population frequency（常见变异不太可能导致罕见病）、ClinVar assertion、OMIM gene–disease relation、inheritance（显性/隐性/新发）、segregation（是否在家系中与疾病共分离）和 HPO phenotype similarity（如 Phrank、Exomiser 的表型匹配）。

### 数据划分、化学、网络与临床

- **数据划分防泄漏**：按序列簇、患者和时间划分数据，分别控制同源泄漏（测试集蛋白与训练集高度同源）、个体重复（同一患者的多份样本分到两边）和未来信息泄漏（用未来才有的标注预测过去）。随机划分往往使性能虚高。
- **Morgan fingerprint 迭代聚合原子邻域**：每轮把原子标识与其邻居标识哈希成新标识，半径 $r$ 轮后覆盖直径 $2r$ 的子结构（ECFP4 对应半径 2），再折叠到固定长度 bit vector。
- **Tanimoto 衡量特定指纹下的 bit overlap**：$T(A,B)=\frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}$，换一种指纹或参数，数值含义就变了。
- **QSAR 建立结构与性质模型**：以分子描述符或指纹为特征回归活性或性质，前提是相似结构有相似性质；activity cliff 是其反例。
- **docking 搜索结合构象并近似排序**：在受体口袋中搜索配体的位置、取向和构象，用经验打分函数近似结合自由能。打分适合排序与筛选，不是精确的亲和力。
- **网络 modularity 比较簇内权重与 degree-preserving null**：

$$
Q=\frac{1}{2m}\sum_{i,j}\left[A_{ij}-\frac{k_ik_j}{2m}\right]\delta(c_i,c_j)
$$

$k_ik_j/2m$ 是保持度分布的随机图中 $i,j$ 之间的期望边数。Louvain 算法贪心地最大化 $Q$。

- **Random Walk with Restart 从 seed 向网络传播分数**：$p_{t+1}=(1-r)Pp_t+rp_0$，$P$ 为 column-stochastic 转移矩阵，$p_0$ 为 seed 分布。modularity 回答“网络如何分块”，RWR 回答“哪些节点靠近 seed”，二者回答不同问题。
- **药物基因组学的证据阶梯**：从 association 到 guideline 还需要分子功能、PK/PD、临床结局和风险收益证据。例如 CYP2C19 功能缺失等位基因与氯吡格雷活化减少的关联，经过机制研究、临床结局研究后才进入 CPIC 用药指南。

> [!NOTE]+ 小结
> - 全课算法可归为动态规划、索引搜索、图算法、交替优化、概率推断、表示学习和统计校准七类。
> - 任何分数都要沿“表示、目标、近似、校准、零模型”解释，原始分数不能跨评分系统比较。
> - 求和与求最大（Forward 对 Viterbi、McCaskill 对 Zuker）回答不同问题，一个给概率，一个给单一最优解。
> - 相似性、关联、机制、因果是递进的证据层级，GWAS、富集、变异预测和网络传播都停留在前两层。
> - 版本、坐标约定、数据划分和参数是结果定义的一部分，必须随结果一起记录。

<br>

# 必读论文与方法来源

课程各章的算法几乎都能追溯到一两篇奠基论文。读原始论文的价值在于看清方法最初要解决的问题和它的假设：例如 Smith–Waterman 原文只有三页，核心就是在 Needleman–Wunsch 递推中加入“0”；Karlin–Altschul 的论文则解释了为什么 BLAST 的 E-value 与数据库长度成正比。

本节优先收录课程 Readings 中可以核实出版信息的原始论文。教材节选、讲义和扫描页不冒充原始研究论文；同一方法同时出现原始论文与后续 protocol 时分别列出。每条后附卷号与页码（已通过 Crossref 与 PubMed 记录核对作者、年份和期刊）以及一两句说明：这篇解决什么问题、核心方法是什么、为什么重要。

<br>

## 课程背景

1. Cohen J. *Computer science and bioinformatics*. Communications of the ACM, 2005. [DOI](https://doi.org/10.1145/1047671.1047672)
   - 48:72–78。面向计算机科学读者的综述，说明分子生物学中的序列、结构与网络问题如何对应到字符串算法、优化和机器学习，可作为全课的导读。

<br>

## 序列分析：比对、打分与统计

这一组论文覆盖双序列比对的三种 DP（全局、局部、仿射 gap）、数据库搜索与其统计理论、替换矩阵、profile HMM，以及两种主流 MSA 程序。

1. Needleman SB, Wunsch CD. *A general method applicable to the search for similarities in the amino acid sequence of two proteins*. Journal of Molecular Biology, 1970. [DOI](https://doi.org/10.1016/0022-2836(70)90057-4)
   - 48:443–453。首次把两条蛋白质序列的最优全局比对写成动态规划问题，在二维矩阵上逐格累加得分。它奠定了此后所有比对算法的递推框架。
2. Smith TF, Waterman MS. *Identification of common molecular subsequences*. Journal of Molecular Biology, 1981. [DOI](https://doi.org/10.1016/0022-2836(81)90087-5)
   - 147:195–197。在递推中加入下界 0，得到寻找最优局部比对的算法。局部比对适合只共享一个结构域或 motif 的序列，是 BLAST 所近似的目标。
3. Gotoh O. *An improved algorithm for matching biological sequences*. Journal of Molecular Biology, 1982. [DOI](https://doi.org/10.1016/0022-2836(82)90398-9)
   - 162:705–708。对仿射 gap 罚分（open + extend）给出 $O(mn)$ 的算法，用三个矩阵记录比对末端的状态，避免了一般 gap 函数所需的三次时间。
4. Altschul SF et al. *Basic local alignment search tool*. Journal of Molecular Biology, 1990. [DOI](https://doi.org/10.1016/S0022-2836(05)80360-2)
   - 215:403–410，作者 Altschul、Gish、Miller、Myers、Lipman。提出 BLAST：用高分短词作 seed、向两侧延伸，并以极值分布给出显著性，使数据库规模的相似性搜索在速度上可行，是生物信息学引用最多的论文之一。
5. Karlin S, Altschul SF. *Methods for assessing the statistical significance of molecular sequence features by using general scoring schemes*. PNAS, 1990. [DOI](https://doi.org/10.1073/pnas.87.6.2264)
   - 87:2264–2268。证明在随机序列模型下，无 gap 局部比对的最高分服从极值（Gumbel）分布，给出 $K,\lambda$ 的计算方法，是 E-value 公式 $E=Kmn e^{-\lambda S}$ 的理论来源。
6. Henikoff S, Henikoff JG. *Amino acid substitution matrices from protein blocks*. PNAS, 1992. [DOI](https://doi.org/10.1073/pnas.89.22.10915)
   - 89:10915–10919。从 BLOCKS 数据库的无 gap 保守片段中统计氨基酸替换频率，按序列一致性阈值聚类后得到 BLOSUM 系列矩阵；BLOSUM62 至今是蛋白质搜索的默认矩阵。
7. Eddy SR. *Profile hidden Markov models*. Bioinformatics, 1998. [DOI](https://doi.org/10.1093/bioinformatics/14.9.755)
   - 14:755–763。综述 profile HMM 的 match/insert/delete 结构、参数估计与数据库搜索，说明它如何把 position-specific 打分和 gap 罚分统一到概率框架中，是 HMMER 与 Pfam 的方法基础。
8. Durbin R et al. *Biological Sequence Analysis*. Cambridge University Press, 1998. [出版社](https://doi.org/10.1017/CBO9780511790492)
   - 作者 Durbin、Eddy、Krogh、Mitchison。课程概率序列模型部分的标准教材，系统讲解比对 DP、Pair HMM、profile HMM、系统发育和 RNA 的随机文法（SCFG）。
9. Edgar RC. *MUSCLE: multiple sequence alignment with high accuracy and high throughput*. Nucleic Acids Research, 2004. [DOI](https://doi.org/10.1093/nar/gkh340)
   - 32:1792–1797。用 $k$-mer 距离快速建 guide tree，progressive 比对后再做 tree-dependent restricted partitioning 的迭代细化，在当时同时提高了速度和精度。
10. Katoh K, Standley DM. *MAFFT multiple sequence alignment software version 7*. Molecular Biology and Evolution, 2013. [DOI](https://doi.org/10.1093/molbev/mst010)
    - 30:772–780（完整标题副题为 improvements in performance and usability）。MAFFT 以 FFT 加速同源片段检测，提供从快速 progressive 到 consistency 型迭代（L-INS-i 等）的多种策略；v7 论文总结了新增功能与大规模比对能力。

<br>

## 索引与组装

这一组论文从数据压缩出发，到压缩全文索引，再到组装和短读比对，展示了“同一个数据结构如何从计算机科学进入基因组学”。

1. Burrows M, Wheeler DJ. *A block-sorting lossless data compression algorithm*. Digital Equipment Corporation, 1994. [报告](https://www.hpl.hp.com/techreports/Compaq-DEC/SRC-RR-124.pdf)
   - DEC Systems Research Center 技术报告 SRC-RR-124。提出 Burrows–Wheeler 变换：对字符串所有轮转排序后取最后一列，可逆且便于压缩。原意是压缩（bzip2 的基础），后来成为短读比对索引的核心。
2. Ferragina P, Manzini G. *Opportunistic data structures with applications*. FOCS, 2000. [DOI](https://doi.org/10.1109/SFCS.2000.892127)
   - 41st IEEE Symposium on Foundations of Computer Science, pp. 390–398。在 BWT 上加入 rank 结构，得到既压缩又能做 backward search 的 FM-index，查询时间与模式长度成正比而与文本长度近乎无关。
3. Pevzner PA, Tang H, Waterman MS. *An Eulerian path approach to DNA fragment assembly*. PNAS, 2001. [DOI](https://doi.org/10.1073/pnas.171285098)
   - 98:9748–9753。把片段组装从 overlap 图上的 Hamiltonian 路径问题转化为 de Bruijn 图上的 Eulerian 路径问题，并讨论重复序列造成的歧义，是短读组装器（如 Velvet、SPAdes）的理论起点。
4. Li H, Durbin R. *Fast and accurate short read alignment with Burrows–Wheeler transform*. Bioinformatics, 2009. [DOI](https://doi.org/10.1093/bioinformatics/btp324)
   - 25:1754–1760。即 BWA：在 FM-index 上做允许错配和 gap 的回溯搜索，使人类基因组规模的短读比对在内存和速度上都可行，并输出 SAM 格式与 mapping quality。

<br>

## RNA 结构与系统发育

1. Nussinov R et al. *Algorithms for loop matchings*. SIAM Journal on Applied Mathematics, 1978. [DOI](https://doi.org/10.1137/0135006)
   - 35:68–82，作者 Nussinov、Pieczenik、Griggs、Kleitman。用动态规划求 RNA 序列的最大配对数二级结构，是嵌套结构 DP 的原型。
2. Zuker M, Stiegler P. *Optimal computer folding of large RNA sequences using thermodynamics and auxiliary information*. Nucleic Acids Research, 1981. [DOI](https://doi.org/10.1093/nar/9.1.133)
   - 9:133–148。把目标从配对数换成基于 stacking 与 loop 的自由能，求最小自由能结构，并允许加入实验约束；mfold 与 ViennaRNA 的 MFE 算法都源于此。
3. Saitou N, Nei M. *The neighbor-joining method: a new method for reconstructing phylogenetic trees*. Molecular Biology and Evolution, 1987. [DOI](https://doi.org/10.1093/oxfordjournals.molbev.a040454)
   - 4:406–425。提出 Neighbor Joining：每步合并使总树长最小的一对类群（等价于 $Q$ 准则），不需要分子钟假设，速度快，至今是距离法建树的标准方法。
4. Felsenstein J. *Confidence limits on phylogenies: an approach using the bootstrap*. Evolution, 1985. [DOI](https://doi.org/10.1111/j.1558-5646.1985.tb00420.x)
   - 39:783–791。把 Efron 的 bootstrap 引入系统发育：对比对列重抽样、重复建树，统计每个 clade 出现的比例作为支持度。这是系统发育树分支支持值最常用的来源。
5. Felsenstein J. *Evolutionary trees from DNA sequences: a maximum likelihood approach*. Journal of Molecular Evolution, 1981. [DOI](https://doi.org/10.1007/BF01734359)
   - 17:368–376。在显式的核苷酸替换模型下计算树的似然，并给出在树上从叶到根递推的 pruning 算法，奠定了最大似然系统发育（RAxML、IQ-TREE）的基础。

<br>

## 蛋白质结构：叠合、比对、预测与功能位点

这一组论文覆盖结构比较（刚体叠合、DALI、TM-align）、结构分类评估、二级结构定义、结构预测（同源建模、片段组装、共进化、深度学习）和三维功能位点识别。

1. Kabsch W. *A solution for the best rotation to relate two sets of vectors*. Acta Crystallographica A, 1976. [DOI](https://doi.org/10.1107/S0567739476001873)
   - 32:922–923。给出两组对应点之间最小化均方偏差的最优旋转的闭式解，即 Kabsch 算法，是计算 RMSD 的标准方法。
2. Arun KS, Huang TS, Blostein SD. *Least-squares fitting of two 3-D point sets*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1987. [DOI](https://doi.org/10.1109/TPAMI.1987.4767965)
   - PAMI-9:698–700。在计算机视觉背景下用 SVD 求解同一问题，并讨论了 SVD 给出反射（行列式为 $-1$）的退化情形，与 Kabsch 算法中的符号修正对应。
3. Holm L, Sander C. *Protein structure comparison by alignment of distance matrices*. Journal of Molecular Biology, 1993. [DOI](https://doi.org/10.1006/jmbi.1993.1489)
   - 233:123–138。提出 DALI：不直接叠合坐标，而是比较两个蛋白的残基内部距离矩阵中相似的子矩阵，用 Monte Carlo 组合成比对，可发现拓扑相似但序列无关的结构。
4. Holm L, Sander C. *Mapping the protein universe*. Science, 1996. [DOI](https://doi.org/10.1126/science.273.5275.595)
   - 273:595–602。用 DALI 对当时已知结构做全体两两比较，构建 fold 空间的分类图，展示了“fold 数量有限、远缘同源大量存在”的格局。
5. Šali A, Blundell TL. *Comparative protein modelling by satisfaction of spatial restraints*. Journal of Molecular Biology, 1993. [DOI](https://doi.org/10.1006/jmbi.1993.1626)
   - 234:779–815。即 MODELLER 的方法：从模板结构中提取距离与二面角的空间约束，结合立体化学约束，以优化约束满足程度来构建目标蛋白模型，是同源建模的经典方法。
6. Simons KT et al. *Assembly of protein tertiary structures from fragments with similar local sequences using simulated annealing and Bayesian scoring functions*. Journal of Molecular Biology, 1997. [DOI](https://doi.org/10.1006/jmbi.1997.0959)
   - 268:209–225，作者 Simons、Kooperberg、Huang、Baker。Rosetta 从头预测的起源：从已知结构中取局部序列相似的短片段（9 残基），用模拟退火拼装，并以贝叶斯推导的打分函数评价。
7. Rohl CA et al. *Protein structure prediction using Rosetta*. Methods in Enzymology, 2004. [DOI](https://doi.org/10.1016/S0076-6879(04)83004-0)
   - 383:66–93，作者 Rohl、Strauss、Misura、Baker。系统介绍 Rosetta 的片段组装、低分辨率与全原子打分、比较建模等流程，是使用 Rosetta 的方法综述。
8. Zhang Y, Skolnick J. *TM-align: a protein structure alignment algorithm based on the TM-score*. Nucleic Acids Research, 2005. [DOI](https://doi.org/10.1093/nar/gki524)
   - 33:2302–2309。以 TM-score 为目标，在 DP 求对应与 Kabsch 叠合之间迭代，速度快且对长度不敏感，广泛用于结构比对与预测评估。
9. Marks DS et al. *Protein 3D structure computed from evolutionary sequence variation*. PLoS ONE, 2011. [DOI](https://doi.org/10.1371/journal.pone.0028766)
   - 6:e28766。用最大熵（Potts）模型从 MSA 中推断直接耦合（EVfold），把预测的接触作为约束折叠出三维结构，证明仅凭序列共进化就能得到正确 fold，是 AlphaFold 时代之前共进化方法的代表。
10. Jumper J et al. *Highly accurate protein structure prediction with AlphaFold*. Nature, 2021. [DOI](https://doi.org/10.1038/s41586-021-03819-2)
    - 596:583–589（2021 年 7 月在线）。AlphaFold2 的方法论文：Evoformer 联合处理 MSA 与 pair 表示，structure module 端到端输出原子坐标并给出 pLDDT，在 CASP14 中达到接近实验精度。
11. Baek M et al. *Accurate prediction of protein structures and interactions using a three-track neural network*. Science, 2021. [DOI](https://doi.org/10.1126/science.abj8754)
    - 373:871–876。RoseTTAFold：在一维序列、二维距离、三维坐标三条通路之间交换信息，精度接近 AlphaFold2，并可预测蛋白复合物。
12. Kabsch W, Sander C. *Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features*. Biopolymers, 1983. [DOI](https://doi.org/10.1002/bip.360221211)
    - 22:2577–2637。即 DSSP：根据骨架氢键能量的模式把每个残基标为螺旋、折叠、转角等，提供了从坐标定义二级结构的标准。
13. Gerstein M, Levitt M. *Comprehensive assessment of automatic structural alignment against a manual standard, the SCOP classification of proteins*. Protein Science, 1998. [DOI](https://doi.org/10.1002/pro.5560070226)
    - 7:445–456。用 SCOP 人工分类作为金标准，评估 STRUCTAL 等自动结构比对方法的打分能否区分同一超家族与无关结构，并讨论了结构比对分数的统计显著性。STRUCTAL 本身出自 Levitt 与 Gerstein，这篇是其分数显著性与评估的主要文献。
14. Fetrow JS, Skolnick J. *Method for prediction of protein function from sequence using the sequence-to-structure-to-function paradigm with application to glutaredoxins/thioredoxins and T1 ribonucleases*. Journal of Molecular Biology, 1998. [DOI](https://doi.org/10.1006/jmbi.1998.1993)
    - 281:949–968。提出 fuzzy functional form（FFF）：以活性位点残基的几何约束描述功能，在预测结构上搜索这些位点来推断功能，说明低精度模型也能用于功能注释。
15. Wallace AC, Borkakoti N, Thornton JM. *TESS: a geometric hashing algorithm for deriving 3D coordinate templates for searching structural databases*. Protein Science, 1997. [DOI](https://doi.org/10.1002/pro.5560061104)
    - 6:2308–2323。用几何哈希把三维原子模板（如催化三联体）的搜索变成哈希表查找，可在全 PDB 中快速检索特定功能位点的空间排列。
16. Senior AW et al. *Improved protein structure prediction using potentials from deep learning*. Nature, 2020. [DOI](https://doi.org/10.1038/s41586-019-1923-7)
    - 577:706–710（2020 年 1 月在线）。AlphaFold1：神经网络从 MSA 特征预测残基对距离分布（distogram）和二面角，构造可微势函数后用梯度下降折叠，在 CASP13 中排名第一。
17. Bagley SC, Altman RB. *Characterizing the microenvironment surrounding protein sites*. Protein Science, 1995. [DOI](https://doi.org/10.1002/pro.5560040404)
    - 4:622–635。FEATURE 方法的原始论文：以位点为中心的同心球壳内统计原子类型、残基类型、化学基团等理化属性，并与非位点背景比较，找出显著不同的特征，是本课程讲者实验室的代表工作。

<br>

## 蛋白质语言模型

1. Rives A et al. *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*. PNAS, 2021. [DOI](https://doi.org/10.1073/pnas.2016239118)
   - 118:e2016239118。ESM-1b 论文：在约 2.5 亿条 UniParc 序列上用 masked language modeling 训练 Transformer，表示中自发出现二级结构、接触和突变效应等信息。
2. Elnaggar A et al. *ProtTrans: Toward understanding the language of life through self-supervised learning*. IEEE TPAMI, 2022. [DOI](https://doi.org/10.1109/TPAMI.2021.3095381)
   - 44:7112–7127。系统比较多种 NLP 架构（ProtBERT、ProtT5 等）在 UniRef 与 BFD 等大规模序列库上的蛋白质预训练，并评估下游的二级结构和亚细胞定位预测。
3. Lin Z et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 2023. [DOI](https://doi.org/10.1126/science.ade2574)
   - 379:1123–1130。ESM-2 与 ESMFold：不用 MSA，仅凭单序列语言模型表示直接预测原子级结构，速度比 AlphaFold2 快一个数量级以上，并构建了六亿多个宏基因组蛋白的结构图谱。
4. Rao R et al. *Evaluating protein transfer learning with TAPE*. NeurIPS, 2019. [论文](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html)
   - Advances in Neural Information Processing Systems 32。提出 TAPE 基准，包含二级结构、接触预测、远缘同源、荧光和稳定性五个任务，并按序列同源性划分测试集，强调预训练表示的评估必须控制同源泄漏。

<br>

## 表达分析与知识表示

1. Ashburner M et al. *Gene ontology: tool for the unification of biology*. Nature Genetics, 2000. [DOI](https://doi.org/10.1038/75556)
   - 25:25–29。宣告 Gene Ontology 项目：用 molecular function、biological process、cellular component 三个受控词表统一不同物种数据库的基因功能描述。
2. Stevens R, Goble CA, Bechhofer S. *Ontology-based knowledge representation for bioinformatics*. Briefings in Bioinformatics, 2000. [DOI](https://doi.org/10.1093/bib/1.4.398)
   - 1:398–414。介绍 ontology 的概念、描述逻辑等表示语言，以及它在生物信息学数据整合中的作用，适合作为知识表示部分的入门综述。
3. The Gene Ontology Consortium. *The Gene Ontology resource: enriching a GOld mine*. Nucleic Acids Research, 2021. [DOI](https://doi.org/10.1093/nar/gkaa1113)
   - 49:D325–D334。GO 资源的更新报告，介绍 term 与注释规模、GO-CAM 因果活动模型以及注释质量控制。
4. Benjamini Y, Hochberg Y. *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society B, 1995. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
   - 57:289–300。定义 FDR 并给出 BH 逐步程序，在检验数很大时比控制 family-wise error 的 Bonferroni 校正有更高检出力，是高通量组学多重检验的标准。
5. Love MI, Huber W, Anders S. *Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2*. Genome Biology, 2014. [DOI](https://doi.org/10.1186/s13059-014-0550-8)
   - 15:550。基于负二项 GLM 做差异表达，用经验贝叶斯把各基因的 dispersion 与 fold change 向全局趋势收缩，提高小样本下估计的稳定性。
6. Subramanian A et al. *Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles*. PNAS, 2005. [DOI](https://doi.org/10.1073/pnas.0506580102)
   - 102:15545–15550。提出 GSEA 及 MSigDB：不设阈值，检验预定义基因集合在全基因排序中是否集中于两端，能发现多个基因协同的小幅变化。
7. Bolstad BM et al. *A comparison of normalization methods for high density oligonucleotide array data based on variance and bias*. Bioinformatics, 2003. [DOI](https://doi.org/10.1093/bioinformatics/19.2.185)
   - 19:185–193。比较 Affymetrix 芯片的多种归一化方法，提出并推荐 quantile normalization（把各芯片强度分布强制为相同），后成为 RMA 流程的一部分。
8. Robinson MD, Oshlack A. *A scaling normalization method for differential expression analysis of RNA-seq data*. Genome Biology, 2010. [DOI](https://doi.org/10.1186/gb-2010-11-3-r25)
   - 11:R25。提出 TMM（trimmed mean of M-values）归一化：指出少数高表达基因会扭曲按总 read 数缩放的结果，用截尾后的对数倍数均值估计样本间缩放因子，是 edgeR 的默认方法。
9. Wellcome Trust Case Control Consortium. *Genome-wide association study of 14,000 cases of seven common diseases and 3,000 shared controls*. Nature, 2007. [DOI](https://doi.org/10.1038/nature05911)
   - 447:661–678。早期大规模 GWAS 的里程碑：七种常见病各约 2000 例、共享约 3000 例对照，示范了共享对照设计、基因型质量控制和极严格显著性阈值（文中以 $5\times10^{-7}$ 作为强关联标准）的做法。

<br>

## 变异效应预测与候选优先级

这一组论文沿时间线展示了变异解释方法的演进：从基于保守性的 SIFT、结合结构的 PolyPhen，到 HMM 型的 FATHMM、全基因组的 CADD/DANN，再到把多种分数集成、并加入表型相似度的患者级优先级框架。

1. Ng PC, Henikoff S. *Predicting deleterious amino acid substitutions*. Genome Research, 2001. [DOI](https://doi.org/10.1101/gr.176601)
   - 11:863–874。SIFT 的原始论文：基于“功能重要位置在同源序列中保守”的假设，用 query 的同源比对估计每个位置可容忍的氨基酸，从而预测替换是否有害。
2. Ramensky V, Bork P, Sunyaev S. *Human non-synonymous SNPs: server and survey*. Nucleic Acids Research, 2002. [DOI](https://doi.org/10.1093/nar/gkf493)
   - 30:3894–3900。PolyPhen 的原始论文：结合序列保守性（PSIC 分数）与结构信息的规则判断 nsSNP 是否可能损害蛋白功能，并对当时已知的人类 nsSNP 做了全面调查。
3. Kumar P, Henikoff S, Ng PC. *Predicting the effects of coding non-synonymous variants on protein function using the SIFT algorithm*. Nature Protocols, 2009. [DOI](https://doi.org/10.1038/nprot.2009.86)
   - 4:1073–1081。SIFT 的使用 protocol：说明如何提交序列或基因组坐标、选择同源数据库、解读 SIFT score 与 0.05 阈值，以及中位保守度等质量指标。
4. Adzhubei IA et al. *A method and server for predicting damaging missense mutations*. Nature Methods, 2010. [DOI](https://doi.org/10.1038/nmeth0410-248)
   - 7:248–249。PolyPhen-2：用序列、系统发育和结构特征训练朴素贝叶斯分类器，提供 HumDiv 与 HumVar 两个训练模型，分别适用于复杂病研究与孟德尔病诊断场景。
5. Li MX et al. *A comprehensive framework for prioritizing variants in exome sequencing studies of Mendelian diseases*. Nucleic Acids Research, 2012. [DOI](https://doi.org/10.1093/nar/gkr1257)
   - 40:e53。提出 KGGSeq 的三级过滤与优先级框架（遗传层面、变异功能层面、基因与通路知识层面），在概念验证病例中把全外显子变异缩小到极少数候选。
6. Shihab HA et al. *Predicting the functional, molecular, and phenotypic consequences of amino acid substitutions using hidden Markov models*. Human Mutation, 2013. [DOI](https://doi.org/10.1002/humu.22225)
   - 34:57–65。FATHMM：用同源序列与 Pfam 结构域的 profile HMM 计算替换前后的概率变化，并以结构域对疾病突变的倾向加权，提高了对致病 missense 的区分能力。
7. Kircher M et al. *A general framework for estimating the relative pathogenicity of human genetic variants*. Nature Genetics, 2014. [DOI](https://doi.org/10.1038/ng.2892)
   - 46:310–315。CADD 原始论文：以人类谱系中已固定的变异（近似良性）和模拟的新生变异（含有害变异）作训练集，整合 60 多种注释训练线性 SVM（两类各约 1470 万个变异），为全部约 86 亿个可能的人类 SNV 预计算分数，并支持短 indel。
8. Shihab HA et al. *Ranking non-synonymous single nucleotide polymorphisms based on disease concepts*. Human Genomics, 2014. [DOI](https://doi.org/10.1186/1479-7364-8-11)
   - 8:11。在 FATHMM 中加入疾病特异的加权方案，在 17 个疾病概念/类别上减少了假阳性候选，使排序针对具体疾病而不只是一般性的有害程度。
9. Dong C et al. *Comparison and integration of deleteriousness prediction methods for nonsynonymous SNVs in whole exome sequencing studies*. Human Molecular Genetics, 2015. [DOI](https://doi.org/10.1093/hmg/ddu733)
   - 24:2125–2137。系统评估 18 种 deleteriousness 分数（11 种功能预测、3 种保守性、4 种集成分数），并据此训练 MetaSVM 与 MetaLR 两个集成模型，结论是集成分数优于任何单一方法；这些分数随后收录进 dbNSFP。
10. Li M et al. *Robust and rapid algorithms facilitate large-scale whole genome sequencing downstream analysis in an integrative framework*. Nucleic Acids Research, 2017. [DOI](https://doi.org/10.1093/nar/gkx019)
    - 45:e75（在线编号 gkx019）。KGGSeq 的后续版本：提出 gap-filled 基因特征注释、bit-block 基因型编码和分段快速读取三种算法，作为全基因组测序下游分析的并行计算框架基础。
11. Boudellioua I et al. *Semantic prioritization of novel causative genomic variants*. PLoS Computational Biology, 2017. [DOI](https://doi.org/10.1371/journal.pcbi.1005500)
    - 13:e1005500。PhenomeNET Variant Predictor：用 ontology 语义相似度比较患者表型与人类疾病、模式动物表型，结合变异有害性预测候选致病变异，尤其能处理尚无已知疾病关联的基因。
12. Jagadeesh KA et al. *Phrank measures phenotype sets similarity to greatly improve Mendelian diagnostic disease prioritization*. Genetics in Medicine, 2019 (online 2018). [DOI](https://doi.org/10.1038/s41436-018-0072-y)
    - 21:464–470。Phrank 在 HPO 有向无环图上用信息量（information content）度量两组表型的相似度，不依赖专有疾病表型注释，在真实病例中显著提高了致病基因的排序。作者来自 Stanford Bejerano 实验室。
13. Rentzsch P et al. *CADD: predicting the deleteriousness of variants throughout the human genome*. Nucleic Acids Research, 2019. [DOI](https://doi.org/10.1093/nar/gky1016)
    - 47:D886–D894。CADD 的更新综述：介绍最新版本 1.4 对 GRCh38 的支持、网站查询与 API，并重申训练设计（模拟新生变异对人–黑猩猩分化后固定的变异）和分数的使用方式。
14. Quang D, Chen Y, Xie X. *DANN: a deep learning approach for annotating the pathogenicity of genetic variants*. Bioinformatics, 2015. [DOI](https://doi.org/10.1093/bioinformatics/btu703)
    - 31:761–763。使用与 CADD 相同的训练数据和特征，把线性 SVM 换成深度神经网络以捕获特征间的非线性关系，分类性能优于 CADD。

<br>

## 化学信息学、网络与药物基因组学

1. Rogers D, Hahn M. *Extended-connectivity fingerprints*. Journal of Chemical Information and Modeling, 2010. [DOI](https://doi.org/10.1021/ci100050t)
   - 50:742–754。系统描述 ECFP（Morgan 类）指纹：以原子不变量为初始标识，按半径迭代哈希邻域并去除重复子结构，是相似性搜索与 QSAR 中最常用的圆形指纹。
2. Gaulton A et al. *ChEMBL: a large-scale bioactivity database for drug discovery*. Nucleic Acids Research, 2012. [DOI](https://doi.org/10.1093/nar/gkr777)
   - 40:D1100–D1107。介绍从文献人工抽取的化合物–靶点生物活性数据库 ChEMBL，是训练 QSAR 与靶点预测模型的主要公共数据源。
3. Barabási AL, Oltvai ZN. *Network biology: understanding the cell's functional organization*. Nature Reviews Genetics, 2004. [DOI](https://doi.org/10.1038/nrg1272)
   - 5:101–113。综述细胞网络的无标度度分布、hub、模块和 motif 等拓扑特征及其生物学含义，是网络生物学的经典入门。
4. Cowen L et al. *Network propagation: a universal amplifier of genetic associations*. Nature Reviews Genetics, 2017. [DOI](https://doi.org/10.1038/nrg.2017.38)
   - 18:551–562。综述以 random walk with restart、热扩散为代表的网络传播方法，说明它如何把弱的遗传关联信号沿相互作用网络放大，用于疾病基因发现与分层。
5. Whirl-Carrillo M et al. *An evidence-based framework for evaluating pharmacogenomics knowledge for personalized medicine*. Clinical Pharmacology & Therapeutics, 2021. [DOI](https://doi.org/10.1002/cpt.2350)
   - 110:563–572。PharmGKB 的临床注释证据评分体系：为基因–药物关联的每条证据打分并汇总为 1A–4 级证据等级，使“关联到指南”的证据阶梯可量化。PharmGKB 由 Altman 实验室参与创建。
6. Blondel VD et al. *Fast unfolding of communities in large networks*. Journal of Statistical Mechanics, 2008. [DOI](https://doi.org/10.1088/1742-5468/2008/10/P10008)
   - 2008:P10008。Louvain 算法：交替进行局部节点移动（贪心提升 modularity）和社区聚合，可在百万节点规模的网络上快速完成社区检测。
7. Szklarczyk D et al. *The STRING database in 2021: customizable protein–protein networks, and functional characterization of user-uploaded gene/measurement sets*. Nucleic Acids Research, 2021. [DOI](https://doi.org/10.1093/nar/gkaa1074)
   - 49:D605–D612。STRING 数据库的年度更新：整合实验、共表达、文本挖掘等多种证据通道的蛋白相互作用并给出置信分数，新增用户自定义网络与基因集功能富集。
8. Trott O, Olson AJ. *AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading*. Journal of Computational Chemistry, 2010. [DOI](https://doi.org/10.1002/jcc.21334)
   - 31:455–461。AutoDock Vina：结合经验与知识型项的打分函数、迭代局部搜索的全局优化和多线程，比 AutoDock 4 快约两个数量级且结合模式预测更准。

<br>

## 分子动力学

1. Alder BJ, Wainwright TE. *Phase transition for a hard sphere system*. Journal of Chemical Physics, 1957. [DOI](https://doi.org/10.1063/1.1743957)
   - 27:1208–1209。最早的分子动力学模拟之一：通过数值求解硬球系统的运动方程观察到相变，展示了用计算机直接积分粒子运动研究统计力学的可能。
2. Karplus M, McCammon JA. *Molecular dynamics simulations of biomolecules*. Nature Structural Biology, 2002. [DOI](https://doi.org/10.1038/nsb0902-646)
   - 9:646–652。由生物分子 MD 先驱撰写的综述，回顾从 1977 年第一次蛋白质模拟以来的方法发展，以及 MD 在构象变化、配体结合和结构精修中的应用。
3. Hollingsworth SA, Dror RO. *Molecular dynamics simulation for all*. Neuron, 2018. [DOI](https://doi.org/10.1016/j.neuron.2018.08.011)
   - 99:1129–1143。面向实验生物学家的 MD 入门综述：说明 MD 能回答和不能回答的问题、力场与时间尺度的限制，以及如何解读模拟结果，课程以此作为动力学部分的导读。

<br>

## 文献计算与降维可视化

1. Lee J et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. Bioinformatics, 2020. [DOI](https://doi.org/10.1093/bioinformatics/btz682)
   - 36:1234–1240。在 BERT 基础上用 PubMed 摘要与 PMC 全文继续预训练，在生物医学命名实体识别、关系抽取和问答任务上明显优于通用 BERT。
2. van der Maaten L, Hinton G. *Visualizing data using t-SNE*. Journal of Machine Learning Research, 2008. [论文](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
   - 第 9 卷。提出 t-SNE：在高维用高斯核、低维用 Student-t 核定义邻近概率并最小化 KL 散度，重尾分布缓解了“拥挤问题”。它保留局部邻域，簇间距离和簇大小不可直接解读。
3. McInnes L et al. *UMAP: Uniform Manifold Approximation and Projection*. Journal of Open Source Software, 2018. [DOI](https://doi.org/10.21105/joss.00861)
   - 3:861。UMAP 软件论文：基于模糊单纯复形构造高维近邻图并优化低维嵌入，比 t-SNE 更快、可嵌入新数据，在单细胞数据可视化中广泛使用；与 t-SNE 一样，全局距离解释需谨慎。

<br>

# 常用工具与软件

论文给出方法，工具把方法落到可运行的代码上。选工具时容易犯的错误是“按名气选”：用 ViennaRNA 去分析 RNA-seq，把 TPM 喂给 DESeq2，或者只写“用 BLAST 搜了一下”而不记录数据库版本。下面按主题列出课程涉及的工具，每条说明用途以及**必须一起记录或检查的东西**，因为同一工具在不同参数或数据版本下给出的结果并不相同。

<br>

## 序列分析工具

- [Biopython](https://biopython.org/)：提供 FASTA/GenBank、alignment 与 PDB 的 Python I/O，适合把序列读写、格式转换和简单比对嵌入自己的脚本。需要注意，输入输出格式和 alphabet validation 仍需显式检查，例如 DNA 文件中混入 `U` 或蛋白质序列中出现非标准字符时，解析器未必报错。
- [BLAST](https://blast.ncbi.nlm.nih.gov/)：局部相似性检索，对应课程中 seed–延伸–E-value 的流程。报告结果时应记录程序类型（blastn、blastp、blastx 等）、数据库 release、替换矩阵、gap 参数、low-complexity masking 设置与 E-value 阈值，因为这些都会改变命中及其显著性。
- [HMMER](http://hmmer.org/)：profile HMM 搜索与建模（hmmbuild 建模、hmmsearch/hmmscan 搜索、jackhmmer 迭代搜索），对远缘同源比 BLAST 更灵敏。应保存所用 profile、报告与纳入阈值（E-value 或 bit score cutoff）以及 target database 版本。
- [MAFFT](https://mafft.cbrc.jp/alignment/software/)：多序列比对。应按序列数量与预期 indel 选择策略：几百条以内、追求精度时可用 L-INS-i 等迭代 consistency 模式，数千条以上用 FFT-NS-2 等快速 progressive 模式。

## 系统发育、RNA 与转录组工具

- [IQ-TREE](http://www.iqtree.org/) / [RAxML-NG](https://github.com/amkozlov/raxml-ng)：maximum-likelihood phylogeny，在课程讲过的 Felsenstein 似然框架上做大规模树搜索。应保存输入 alignment、model selection 结果（如 ModelFinder 选出的替换模型）、random seed 与 bootstrap 设置（标准 bootstrap 还是 ultrafast bootstrap、重复次数）。
- [ViennaRNA](https://www.tbi.univie.ac.at/RNA/)：RNA 二级结构预测，提供 MFE（Zuker 型）与 partition function（McCaskill 型）计算，例如 RNAfold。它不是 RNA-seq 分析工具，名称中的 RNA 指的是结构而不是表达。
- [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/) / [MultiQC](https://multiqc.info/)：RNA-seq reads 的质量控制与批量 QC 汇总。FastQC 对单个文件给出碱基质量、GC 含量、接头污染等报告，MultiQC 把多个样本和多个工具的报告汇总到一页，便于发现离群样本。
- [STAR](https://github.com/alexdobin/STAR) / [Salmon](https://salmon.readthedocs.io/) / [kallisto](https://pachterlab.github.io/kallisto/)：alignment 或 transcript quantification。STAR 做剪接感知的基因组比对，输出 BAM；Salmon 与 kallisto 做 pseudo-alignment / selective alignment，直接估计转录本丰度，速度快得多。
- [DESeq2](https://bioconductor.org/packages/DESeq2/) / [edgeR](https://bioconductor.org/packages/edgeR/) / [limma-voom](https://bioconductor.org/packages/limma/)：差异表达分析。三者的输入尺度与统计模型不同：DESeq2 与 edgeR 直接对原始 count 建负二项模型（归一化分别为 median-of-ratios 与 TMM），limma-voom 把 count 转成 log-CPM 并估计均值–方差关系作为精度权重，再用线性模型。三者都不应以 TPM 作为输入。

## 结构分析工具

- [RCSB PDB](https://www.rcsb.org/)：提供实验结构、validation 报告（分辨率、R-free、Ramachandran 离群等）与 biological assembly 元数据。需要注意 asymmetric unit 不一定等于生物学功能单元。
- [Bio.PDB](https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ)：Biopython 中的 Python 结构解析模块，按 Structure–Model–Chain–Residue–Atom 层级访问坐标。生产代码优先处理 mmCIF，因为大型结构已无法用 PDB 格式表示，且 mmCIF 保留了更完整的元数据。
- [DALI](http://ekhidna2.biocenter.helsinki.fi/dali/) / [TM-align](https://zhanggroup.org/TM-align/)：对应关系未知时的结构比对，分别基于距离矩阵比较和 TM-score 优化。应同时报告 score（DALI Z-score 或 TM-score）、RMSD、aligned length 与归一化约定（TM-score 按哪条链长度归一化）。
- [FEATURE](https://simtk.org/projects/feature)：以局部理化微环境进行 3D functional-site 建模，即课程中同心球壳统计属性的方法。模型定义与背景样本决定 score 含义：同一位点在不同背景集合下得分不同。
- [MDAnalysis](https://www.mdanalysis.org/)：MD trajectory 的 Python 分析库，支持原子 selection、轨迹 alignment 与 RMSD、RMSF、距离等分析，可读取 GROMACS、AMBER、CHARMM 等多种格式。
- [OpenMM](https://openmm.org/) / [GROMACS](https://www.gromacs.org/)：分子动力学模拟引擎，OpenMM 以 Python API 和 GPU 加速见长，GROMACS 以高性能命令行流程见长。必须记录 force field、water model、integrator、constraint、ensemble（NVT/NPT 及对应 thermostat/barostat）与 timestep，否则模拟结果不可复现也无法解释。

## 变异解释工具

- [Ensembl VEP](https://www.ensembl.org/info/docs/tools/vep/index.html) / [ANNOVAR](https://annovar.openbioinformatics.org/)：variant consequence 预测与数据库注释（所在基因、转录本、missense/splice 等后果，并附加频率与预测分数）。应固定 genome build、transcript set（如 Ensembl canonical、MANE Select 或 RefSeq）、cache 和插件版本；同一变异在不同转录本上可能有不同后果。
- [dbNSFP](https://sites.google.com/site/jpopgen/dbNSFP)：汇总人类 nsSNV 与 splice-site 的多种 prediction score（SIFT、PolyPhen-2、CADD、MetaSVM 等）。不同分数的方向（有的越低越有害，如 SIFT；有的越高越有害，如 CADD）、缺失值和版本不能混用。
- [CADD](https://cadd.gs.washington.edu/)：全基因组 SNV/短 indel 的 deleteriousness ranking。应记录使用的是 raw 还是 PHRED score、model release 与 reference build（GRCh37 与 GRCh38 分数不同）。
- [Exomiser](https://exomiser.com/)：结合 inheritance 模式、variant score 与 HPO phenotype 相似度的候选基因/变异排序。输出仍需家系和临床证据复核，排名第一不等于诊断。
- [gnomAD](https://gnomad.broadinstitute.org/)：群体 allele frequency、coverage 与 gene constraint（如 LOEUF、pLI）。使用时应检查 ancestry 分组、release 版本和 callable region：在测序覆盖差的区域，“频率为 0”可能只是没有测到。
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)：variant 的 clinical assertion、condition、submitter 与 review status（星级）。不同提交者的冲突记录不能静默合并，应保留冲突并参考 review status。
- [OMIM](https://www.omim.org/)：Mendelian gene–phenotype 知识库。使用时遵守许可（批量下载需申请）并核对 phenotype mapping 的类型（例如是否确认了分子机制）。

## 化学信息学、网络与文本挖掘工具

- [RDKit](https://www.rdkit.org/)：分子 standardization、descriptor、fingerprint 与通用 cheminformatics 功能。Morgan 参数必须完整保存（半径、bit 数、是否使用 chirality 与 feature invariants），否则 Tanimoto 数值无法复现。
- [NetworkX](https://networkx.org/)：适合中小图原型和算法验证。大规模传播通常改用 SciPy sparse 矩阵运算或专用图框架，因为纯 Python 的逐节点遍历在百万边规模上过慢。
- [Cytoscape](https://cytoscape.org/)：交互网络可视化与 annotation，可通过插件导入 STRING 网络、做富集与布局。
- [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/)：NCBI 提供的生物医学实体（基因、疾病、化学物、变异等）与关系辅助标注。自动结果仍需证据复核，实体归一化错误和关系误抽取都很常见。

## 工作流与环境管理

- [Snakemake](https://snakemake.readthedocs.io/) / [Nextflow](https://www.nextflow.io/)：以 DAG 管理步骤之间的 dependency、计算 resource 与增量重跑：只有输入或代码改变的步骤才重新执行。
- Conda lock、container digest 或 Nix/Guix：固定环境，把解释器和所有依赖精确到版本或哈希。仅写“使用 Python 3”不足以复现。

工具选择应从输入、输出、假设、规模和许可证出发，而不是按知名度。论文引用、软件版本、参数与 database release 应同时记录；只链接项目主页不能复现实验。

> [!NOTE]+ 小结
> - 每个工具都对应前文某种方法，使用前先确认它的输入尺度和统计假设与数据相符。
> - 搜索与注释工具的结果依赖数据库 release、genome build 和转录本集合，必须随结果记录。
> - 不同分数的方向、尺度和缺失值约定不同，合并前要逐项核对。
> - 工作流引擎加锁定环境，才能让结果从原始数据重新生成。
