---
title: 斯坦福CS274：计算分子生物学的表示与算法
date: 2026-09-21
categories:
- 上斯坦福
tags:
- CompBio
desc: Stanford CS274 详细课程笔记：从序列比对、HMM 与系统发育，到蛋白质结构、分子动力学、转录组、化学信息学和网络生物学。
hidden: true
---

Stanford CS274 的全名是 *Representations and Algorithms for Computational Molecular Biology*，与 BIOE 214、BMDS 214、GENE 214 交叉挂牌，由 Russ Altman 主讲。它是一门研究生计算分子生物学导论，先修课为 CS106B，课程包含较多 Python 编程。内容围绕生物对象的表示展开，依次讨论序列比对、概率模型、系统发育、结构计算、分子动力学、组学分析和网络生物学，同时考察算法复杂度、统计假设与生物学解释。

- [Stanford Bulletin：CS274](https://bulletin.stanford.edu/courses/1410302)

<br>

> [!INFO]+ 版本范围
>
> 当前 Stanford Bulletin 明确列出：双序列与多序列的序列/结构比对、系统发育树、隐马尔可夫模型、蛋白质结构计算、蛋白质结构预测、蛋白质语言模型、分子动力学与能量最小化、三维生物数据统计、数据整合、知识表示与受控术语、转录分析、化学信息学、药物遗传学和网络生物学。
>
> 2020 年课程 staff 页面还列出动态规划、结构叠合、距离信息、1D/3D motif、统计特征检测以及 RNA 序列与结构。当前目录、历史课程说明和补充材料在文中分别注明。

<br>

# 总览

- **序列部分**
  - 字符串索引与精确匹配；
  - 双序列比对、数据库搜索与多序列比对；
  - motif、HMM 与序列家族模型；
  - 系统发育与 RNA 结构。
- **结构部分**
  - 三维坐标、距离与接触表示；
  - 结构叠合、对应搜索与三维 motif；
  - 结构预测、能量函数与分子动力学；
  - 轨迹对齐、降维、聚类与状态模型。
- **功能与系统部分**
  - 表达矩阵、受控术语与本体；
  - GWAS、变异效应与患者级优先级；
  - 化学结构、药物遗传学与生物网络；
  - 文献抽取、证据整合与可复现工作流。

各部分都按下列逻辑链记录：

$$
\text{生物问题}
\rightarrow\text{表示}
\rightarrow\text{目标函数}
\rightarrow\text{算法}
\rightarrow\text{统计校准}
\rightarrow\text{生物学解释}
$$

- **分数校准**
  - BLAST 分数需要结合数据库大小和零模型解释；
  - 预测结构要同时查看局部置信度与相对位置置信度；
  - 差异表达依赖实验设计和多重检验。
- **实现要求**
  - 能够实现 Needleman–Wunsch、Smith–Waterman、Viterbi、Kabsch 等基础算法；
  - 能够说明成熟工具的输入、目标函数、近似、输出和解释边界。

课程复习材料给出的另一条主线是从测量走向决策：

$$
\text{DNA/RNA/protein 与实验观测}
\rightarrow\text{序列、结构和表达表示}
\rightarrow\text{比对、预测与统计检验}
\rightarrow\text{功能、疾病和药物解释}
$$

- **从测量到计算对象**
  - 回答数据如何产生、怎样编码、表示保留了什么信息。
- **从模型分数到证据**
  - 回答分数能够支持哪一级结论；
  - GWAS 关联信号、变异 deleteriousness、患者致病性与药物处方分属不同证据层级；
  - GSEA 的 gene-set enrichment 不等于 pathway 已被因果激活。


核心章节之间的依赖为：

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

## 数学预备

课程反复使用四类数学对象。

$$
\log\frac{P(x\mid M_1)}{P(x\mid M_0)}
$$

- **log-odds：比较目标模型与背景模型的相对证据**
  - 当模型把观测分解为条件独立因子时，联合概率写成各因子的乘积，取对数后变成分数之和；
  - 这解释了替换矩阵、PWM、HMM 路径分数和许多分类器为何采用加和形式。
- **动态规划：把指数枚举压缩为子问题复用**
  - 问题需要具有 optimal substructure 和 overlapping subproblems；
  - 状态必须包含足够历史，使未来决策不再依赖更早细节；
  - 在线性 gap 评分下，每个网格位置只需保存一个最优值；
  - 仿射 gap 必须区分 match、gap-in-$x$ 和 gap-in-$y$ 三类末端状态。
- **最大化与求和：回答不同的概率问题**
  - Viterbi 的 max 找单条最可能路径；
  - Forward 的 sum 汇总所有路径；
  - 最可能解释的概率可能很低，不能用 Viterbi 路径概率代替总似然。
- **生成模型与判别模型：建模目标不同**
  - HMM、系统发育替换模型和 negative-binomial count model 描述数据如何产生；
  - logistic regression 等判别模型直接描述标签条件概率；
  - 前者可表达机制假设，后者通常更直接服务预测。

## 复杂度

除 Big-O 外，还要问输入中的哪个维度增长。长度分别为 $n,m$ 的 pairwise alignment 是 $O(nm)$；含 $K$ 个状态、长度为 $T$ 的稠密 HMM 是 $O(TK^2)$。对 $s$ 条长度约为 $n$ 的序列，精确 $s$ 维 DP 有 $O(n^s)$ 个格点，每格最多检查 $2^s-1$ 个非空移动，朴素时间为 $O((2^s-1)n^s)$，空间为 $O(n^s)$。数据库搜索还要区分一次建索引与每次 query 的成本。

空间复杂度同样重要。$10^5\times10^5$ 的 DP 表即使每格只占 4 bytes，也约需 40 GB，即约 37.3 GiB。生物规模数据通常需要 banded alignment、linear-space traceback、稀疏图或分块计算。

## 数值计算

概率动态规划不仅要写出递推，还要处理浮点下溢。Forward–backward 可在每个时间步使用 scaling factor，或全程在 log-space 以 log-sum-exp 计算。比较 likelihood 时还要说明是否按序列长度归一化，因为不同长度序列的原始 log-likelihood 通常不可直接比较。

## Python 实现基础

课程编程把 Python 用作表示和验证算法的工具，重点不是语法本身，而是把状态、边界条件与输入输出明确写进程序。NumPy 数组要求同质 dtype，数值内核应优先使用 broadcasting、切片和线性代数操作，避免在大数组上逐元素执行 Python `for` loop。向量化不会改变渐近复杂度，但可把循环移到底层编译代码。

NumPy 的整数索引会消去一个维度，切片则通常保留该维度。例如二维数组 `a` 中，`a[0, 0]` 是 scalar，`a[:, 0]` 的 shape 为 `(n,)`，`a[0:1, 0:1]` 的 shape 为 `(1, 1)`。动态规划表若把行列向量 shape 混淆，broadcasting 可能生成数值合理但语义错误的结果。

可执行脚本通过 `sys.argv` 或 `argparse` 接收输入，不应把文件路径与参数硬编码在 notebook。类方法的第一个参数 `self` 指当前实例；默认参数若使用可变对象会在调用间共享状态。`/` 总是浮点除法，`//` 是向下取整的 floor division，对负数并非简单截断。

递归、memoization 与 bottom-up DP 表达同一依赖图的不同求值方式。递归实现先写 base case，再保证每次调用都接近 base case；memoization 只计算可达状态，bottom-up 则按拓扑顺序填表。DFS 使用 stack 或递归，BFS 使用 queue，二者的访问顺序和最短路保证不同。

Conda 或其他隔离环境负责固定解释器与依赖，`PYTHONPATH` 只影响模块查找，不能替代环境管理。提交代码应遵守 PEP 8，并为函数写明参数、返回值、shape、单位和异常条件。行内注释应解释状态定义、边界和不变量，而不是逐字翻译代码。

<br>

# 生物表示

## 概念：生物表示

- **研究对象**
  - 计算方法不能直接处理“生命活动”本身，只能处理被编码后的对象；
  - DNA 可表示为带方向的字符串，基因可表示为参考基因组上的区间，蛋白质可表示为氨基酸序列或原子坐标，表达实验可表示为样本与基因组成的矩阵；
  - 每一种表示都保留一部分信息，同时丢弃其余信息。例如序列字符串保留残基顺序，却不包含细胞类型、化学修饰和空间构象。
- **本章任务与输入输出**
  - 建立后续算法共同使用的基本约定，包括字母表、链方向、互补关系、坐标系统、变异类型和实验观测层级；
  - 输入通常是实验数据或数据库记录，输出是经过规范化的序列、区间、类别或数值张量；
  - 若混淆 $5'$/$3'$、正负链、0-based/1-based 坐标或 transcript isoform，错误会继续传播到比对、变异注释和结构映射。
- **章节衔接**
  - 生物表示回答“对象在计算机中是什么”；
  - 下一章的数据格式回答“这些对象如何存储和交换”；
  - 字符串、图、概率模型和坐标算法回答“在这种表示上能计算什么”。

## 中心法则

$$
x=x_1x_2\cdots x_n,\qquad x_i\in\Sigma
$$

- **第一层抽象：有限字母表上的字符串**
  - DNA 由四种碱基组成；
  - RNA 以 U 代替 T；
  - 蛋白质通常以 20 种标准氨基酸表示。
  - DNA 的 $\Sigma=\{A,C,G,T\}$，蛋白质的字母表更大；
  - 字符串表示丢弃细胞环境、空间构象、修饰和实验误差，却使索引、匹配、动态规划和概率建模成为可能。
- **核酸化学与链方向**
  - 核苷酸由含氮碱基、五碳糖和磷酸组成；
  - 相邻核苷酸以磷酸二酯键连接，使链具有 $5'$ 到 $3'$ 的方向；
  - A–T 形成两个氢键，G–C 形成三个氢键；
  - 双螺旋稳定性还受到 base stacking、离子与溶剂影响，不能只按氢键数解释；
  - 常见 B-DNA 和 A-DNA 为右手双螺旋，Z-DNA 为左手，因此“DNA 一定右手”并不成立。
- **互补、转录与翻译**
  - DNA 双链具有方向性和互补性，正向链按 $5'\rightarrow3'$ 书写；
  - 反向互补序列先按 $A\leftrightarrow T$、$C\leftrightarrow G$ 取互补，再反转；
  - 转录产生 RNA；
  - 翻译以三联体 codon 读取开放阅读框；
  - 标准遗传密码中，61 个 sense codon 编码 20 种标准氨基酸，另有 3 个 stop codon；
  - 线粒体等遗传密码存在例外。
- **复制、细胞分裂与重组**
  - DNA replication 从 origin 形成 replication bubble 和两个 replication fork；
  - DNA polymerase 只能沿 $5'\rightarrow3'$ 合成；
  - leading strand 近似连续，lagging strand 以 Okazaki fragment 不连续合成；
  - Mitosis 通常保持体细胞染色体组；
  - meiosis 经一次复制和两次分裂形成单倍体，并通过 independent assortment 与 homologous recombination 产生组合多样性；
  - 重组既产生遗传多样性，也使相邻位点的关联随物理距离衰减，这是 linkage disequilibrium 与 GWAS 定位分辨率的生物学基础。
- **中心法则的扩展与限制**
  - DNA 可复制，DNA 转录为 RNA，RNA 经 reverse transcription 可生成 DNA；
  - 这不表示蛋白质序列能按模板逆向编码回核酸；
  - 真核 pre-mRNA 经历 $5'$ cap、splicing 和 poly(A) tail 加工；
  - Alternative splicing 使同一 gene 产生多个 transcript isoform；
  - 翻译后，phosphorylation、glycosylation、acetylation、ubiquitination 等 post-translational modification 和 membrane trafficking 继续决定定位、稳定性与功能。
- **扩展字母表与翻译上下文**
  - 核酸可含 IUPAC 歧义码；
  - 蛋白质可含 B、Z、J、X、U、O 与 stop symbol；
  - 翻译依赖 reading frame、strand、起始位点和 genetic-code table；
  - “字符串到蛋白质”的映射必须同时记录物种、细胞器和坐标约定。
- **坐标约定**
  - 计算时必须保留方向；
  - genomic interval 常用 0-based half-open 或 1-based closed 坐标，两者混用会产生 off-by-one error；
  - 正链上的区间 $[i,j)$ 长度是 $j-i$；
  - 负链 feature 的生物阅读方向与参考坐标递增方向相反。

## 变异

- **按尺度分类**
  - Point-level 变化包括 substitution、small insertion 和 deletion；
  - chromosomal alteration 包括大段 deletion、duplication、inversion 与 translocation；
  - copy-number variation 表示片段拷贝数改变；
  - 短串联重复扩增中，trinucleotide repeat 可在代际传递时继续扩张，并因是否落在 coding region 而产生不同后果。
- **比对中的方向解释**
  - 一个 gap 只是假设某处发生 indel；
  - 无法单凭两条现存序列判定是 insertion 还是 deletion；
  - 需要 outgroup 或祖先模型确定方向。

蛋白质变异的影响依赖位置环境。同样的 amino-acid substitution 位于活性位点、疏水核心、无序区或表面时，后果可能完全不同，因此还需要结构与功能层面的表示。

## 测量

算法处理的是实验观测值，而非未经测量误差影响的生物状态：

- sequencing read 是被采样并带错误的分子片段；
- RT-PCR 先把 RNA reverse-transcribe 为 cDNA，再以 gene-specific primer 扩增目标；实时 qPCR 以 fluorescence cycle curve 做相对或绝对定量；
- microarray intensity 是杂交和扫描产生的连续信号；
- RNA-seq count 同时受真实丰度、建库和测序深度影响；
- PDB coordinate 是实验密度与结构模型共同产生的估计；
- interaction edge 依赖 assay、阈值和数据库筛选。

因此，表示至少有三层：真实状态、观测过程、文件中的编码。把文件值直接等同于真实状态，会把技术噪声误当生物差异。

表达测量的发展体现了表示粒度的改变。RT-PCR 针对少量预选 target，microarray 依赖预先设计的 probe grid 和 hybridization intensity，RNA-seq 则把 cDNA 随机片段化并从单端或双端读取。Paired-end 数据同时给出两个 read 及 insert-size 信息；比对或 pseudoalignment 后再按 gene/transcript 计数。读长约 250 bp 只是特定平台配置，不是 RNA-seq 的定义。

## 表观遗传与调控

表观遗传调控改变可遗传或可维持的表达状态，而不改变 DNA 碱基序列。DNA methylation、histone modification 与 chromatin accessibility 影响转录机器能否接近 promoter 和 enhancer；约 22 nt 的 miRNA 可通过与目标 RNA 配对调节稳定性或翻译。表观遗传标记既可能参与调控，也可能是细胞类型、发育状态或疾病过程的结果，观察到差异不能直接推出因果方向。

## 表示层次

- **序列**：保留残基顺序，适合比对、motif、语言模型。
- **特征向量**：用长度、GC 含量、$k$-mer 频率、理化性质表示，适合分类与聚类。
- **图**：原子为节点、化学键为边；或基因为节点、相互作用为边。
- **三维坐标**：每个原子的 $(x,y,z)$，适合结构叠合、距离图和动力学。
- **概率模型**：把家族或过程表示成分布，不只保存一个共识序列。
- **本体与知识图谱**：表示实体、关系、证据和层级约束。

> [!WARNING]+ 表示决定可回答的问题
>
> FASTA 序列无法表达原子坐标；单个 PDB 构象不等于溶液中的完整构象分布；基因表达矩阵显示共变，却不自动给出调控方向。算法输出的上限由输入表示决定。

<br>

## 基础代码

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

对长度为 $n$ 的序列枚举全部 $k$-mer 需要 $O(n)$ 次窗口访问；若复制每个子串，实际字符处理量为 $O(nk)$。rolling hash 或编码为整数可避免重复复制。

<br>

# 数据与数据库

## 概念：数据与数据库

- **研究对象**
  - 生物信息分析通常不从手工输入的一条序列开始，而是从公共数据库、测序文件或实验平台导出的记录开始；
  - 数据库记录包含数据本身、标识符、版本、物种、实验方法和注释证据，文件格式规定这些信息怎样编码；
  - 数据库记录是某一时间点的整理结果，不是永远不变的生物学真值。
- **本章任务与输入输出**
  - 第一，正确解析 FASTA、FASTQ、VCF、SAM、mmCIF 等文件，形成后续算法需要的对象；
  - 第二，记录数据来源，使结果能在数据库更新后复现；
  - 典型输入是 accession、检索条件或原始文件；
  - 输出除序列、坐标和表达矩阵外，还应包括 release、下载日期、checksum、解析器版本和未映射记录。
- **章节衔接与解释边界**
  - 后续的序列比对、结构计算和组学分析都默认输入已经通过本章的格式检查与版本固定；
  - 文件能被程序读入只说明语法合法，不说明物种、参考组装、坐标或生物学含义正确。

## 文件格式

- **FASTA**：标识行以 `>` 开头，随后是序列；简单，但元数据结构松散。
- **FASTQ**：每条 read 包含标识、序列、分隔符和 Phred 质量值。
- **SAM/BAM/CRAM**：测序 reads 对参考基因组的比对；BAM 为二进制，CRAM 进一步依赖参考压缩。
- **VCF**：基因组变异及样本 genotype。
- **PDB/mmCIF**：大分子三维结构、链、残基、原子坐标、占有率与实验信息。
- **SDF**：可保存分子连接、二维或三维坐标及属性字段。
- **SMILES**：分子图的线性字符串表示。
- **GFF/GTF**：基因组区间与基因结构注释。
- **表达矩阵**：行通常为基因，列为样本，元素是 count 或归一化表达。

FASTQ 的 Phred 分数定义为 $Q=-10\log_{10}P_e$。因此 $Q=30$ 对应估计错误概率 $10^{-3}$。质量值是概率编码，不应直接当作线性置信度。

FASTQ 还存在 Phred+33 等字符编码约定。解析时字符 `I` 不是质量等级本身，而要先转换为整数。假设各碱基错误独立，一条 read 全部正确的概率是 $\prod_i(1-P_{e,i})$；但实际错误常受测序周期、局部序列和仪器影响，并不严格独立。

SAM 的核心字段包括 query name、bitwise FLAG、reference、position、mapping quality、CIGAR 和 sequence。CIGAR 的 `M` 表示 alignment match，可同时包含 sequence match 与 mismatch；要区分二者需使用 `=`/`X`，或结合 read、reference 与 MD tag。mapping quality 不是 base quality；前者描述 read 被放到该位置的不确定性，后者描述碱基本身的测量错误。

VCF 中 `REF`/`ALT` 必须相对于特定 reference assembly 解释。相同变异可能有不同左对齐表示；比较前应 normalize，并明确 GRCh37 或 GRCh38。genotype `0/1` 与 `1|0` 的差异在于后者带 phase。

PDB 固定列格式在大结构和复杂标识上受限，mmCIF 是当前标准。代码不应假设 chain ID 永远单字符，也不应默认 residue number 连续。插入码、非标准残基、缺失坐标与 biological assembly 都需要显式处理。

## 数据库

- [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)：核酸序列与注释。
- [UniProt](https://www.uniprot.org/)：蛋白质序列与功能；Swiss-Prot 偏人工审校，TrEMBL 偏自动注释。
- [RCSB PDB](https://www.rcsb.org/)：PDB archive 中的实验测定结构；RCSB.org 另提供对计算预测模型资源的访问。
- [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/)：转录组和功能基因组数据。
- [Gene Ontology](https://geneontology.org/)：分子功能、生物过程、细胞组分。
- [ChEMBL](https://www.ebi.ac.uk/chembl/) 与 [PubChem](https://pubchem.ncbi.nlm.nih.gov/)：化合物、靶点与活性。
- [PharmGKB](https://www.pharmgkb.org/)：基因型、药物和表型知识。

数据库记录不是无条件真值。需要检查 accession、版本、物种、组装版本、实验方法、证据码、发布日期和许可证。训练/测试拆分还需避免同源序列、同一患者或近重复结构跨集合泄漏。

## 数据谱系

可复现分析不能只保存下载文件名。最小 provenance 包含：

```yaml
resource: UniProtKB/Swiss-Prot
release: "2026_03"
query: "organism_id:9606 AND reviewed:true"
retrieved_at: "2026-09-21"
sha256: "..."
software:
  parser: "biopython==1.x"
```

stable identifier 也不是永久不变：记录会被合并、拆分或撤销。可靠 pipeline 应保留原 ID、映射表、目标 namespace、版本和未映射原因，而不是静默丢弃。

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

数据库批量下载还要遵守 API rate limit、认证与许可证，并优先保存 release identifier。`ETag`、`Last-Modified` 可能缺失，checksum 只能证明本地内容是否改变，不能替代数据库版本。生产流程还应重试瞬时网络错误，但不能把 404、权限拒绝或 schema 改变静默当作空数据。

accession 标识数据库记录，version 才标识其具体序列或注释状态。可复现检索应保存 database release、完整 query、返回的 `accession.version`、检索日期和 checksum；仅保存 gene symbol 或网页 URL，无法抵抗记录合并、拆分和更新。

> [!EXAMPLE]+ 例1：同名不是同一实体
>
> `PTEN` 可指 gene symbol、编码蛋白或论文中的概念；Ensembl gene、transcript 与 protein ID 位于不同层。若表达矩阵按 gene 汇总、结构数据按 protein isoform 记录，直接用 symbol join 会制造多对多重复。正确做法是先声明分析单位，再通过带版本的关系表映射。

<br>

# 字符串计算

## 概念：字符串计算

- **核心问题**
  - 将 DNA、RNA 或蛋白质抽象成有限字母表上的字符串后，首先要解决定位和计数；
  - 给定长文本 $T$ 与较短模式 $P$，需要回答模式出现在哪里；
  - 给定大量 reads，还需要统计局部词、建立索引或根据重叠恢复原序列；
  - 本章暂不考虑复杂替换和长 gap，重点是精确字符串结构。
- **输入、输出与方法**
  - 输入是文本、模式、$k$ 值或 read 集合；
  - 输出是匹配位置、前缀表、$k$-mer 频数、后缀索引或 de Bruijn 图；
  - KMP 利用模式内部的前后缀避免重复比较；
  - FM-index 以压缩索引支持大文本检索；
  - $k$-mer 用局部词换取速度和可扩展性。
- **章节衔接与解释边界**
  - 精确索引先找 seed，双序列比对再允许 mismatch 与 gap；
  - $k$-mer overlap 先构建组装图，图算法再选择路径；
  - “相同子串”只表示字面相同，不等于进化同源或功能相同。

## 精确匹配

朴素匹配在长度 $n$ 的文本中寻找长度 $m$ 的模式，最坏为 $O(nm)$。KMP 用前缀函数记录模式自身的边界，时间为 $O(n+m)$。后缀树可在线性时间建索引并快速查询，但常数和内存较大；后缀数组更紧凑，配合二分查找可在 $O(m\log n)$ 查询。FM-index 基于 Burrows–Wheeler Transform，以压缩空间支持 backward search，是短读段比对器的重要基础。

KMP 的 prefix function $\pi[i]$ 表示模式前缀 `pattern[:i+1]` 的最长 proper prefix 与 suffix 的公共长度。发生 mismatch 时，不必把文本指针回退，只把模式指针退到 $\pi[j-1]$。这利用了已经匹配部分的自相似。

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

BWT 先在文本末尾加入唯一终止符 `\$`，再将全部循环旋转排序并取最后一列。终止符保证变换可逆并唯一标识文本结尾。实际 FM-index 不显式保存所有旋转，而通过 suffix array 构建。采用半开区间 $[l,r)$，若 $C[c]$ 表示字典序小于字符 $c$ 的字符数，$\operatorname{Occ}(c,i)$ 表示 `BWT[0:i]` 中 $c$ 的次数，则模式从右向左加入字符 $c$ 时，suffix-array interval 更新为

$$
l'=C[c]+\operatorname{Occ}(c,l),\qquad
r'=C[c]+\operatorname{Occ}(c,r)
$$

区间非空即存在匹配。rank 查询通过 checkpoint 或 wavelet tree 加速。允许 mismatch/indel 时，搜索会分支，因此实际 aligner 结合 seed、剪枝和 DP。

## k-mer

$k$-mer 把长序列切成局部词。其用途包括：

- 无比对相似度；
- de Bruijn 图组装；
- minimizer 索引；
- 物种分类；
- 序列模型输入。

若两个序列的 $k$-mer 集合分别为 $A,B$，Jaccard 相似度为

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

增大 $k$ 会降低随机碰撞，却更怕测序错误和变异。MinHash 用少量哈希最小值近似 Jaccard；minimizer 在滑动窗口内只保留代表 $k$-mer，进一步减少索引。

上述 Jaccard 把 $k$-mer 当集合元素，忽略同一 $k$-mer 的 multiplicity。若 coverage 或 copy number 重要，应使用 multiset/weighted Jaccard，或直接比较 count vector。

若 DNA 背景近似均匀，一个指定 $k$-mer 随机出现概率约为 $4^{-k}$。但真实基因组有 GC bias 和重复序列，因此不能直接用此近似设显著性阈值。canonical $k$-mer 通常取一个 $k$-mer 与其 reverse complement 的字典序较小者，从而合并双链方向。

counting $k$-mer 时还需考虑：

- exact hash table 准确但内存高；
- Bloom filter 节省空间但有假阳性；
- Count–Min Sketch 给频数上界近似；
- rolling 2-bit encoding 可令窗口更新为 $O(1)$；
- 测序错误常形成低频 $k$-mer，可按 abundance 过滤，但会误删真实 rare variant。

## 组装

de Bruijn 图把 $k$-mer 的长度 $k-1$ 前后缀作为节点，把 $k$-mer 作为边。这样，overlap graph 中的序列排序问题转化为 de Bruijn 图上的 Eulerian path。重复序列造成分叉，测序错误造成短小 tips，覆盖不均造成断裂。

例如 reads 中观察到 `ATG`、`TGC`、`GCA`。取 $k=3$ 后得到边 `AT→TG`、`TG→GC`、`GC→CA`，Eulerian path 拼成 `ATGCA`。若同时存在 `TGA`，节点 `TG` 出现分叉；仅靠图拓扑不能判断分支来自真实变异、重复还是错误。

有向图存在 Eulerian path 还要求所有非零度节点在忽略方向后位于同一连通部分，并且至多一个节点满足出度比入度大 1、至多一个节点满足入度比出度大 1，其余节点入度等于出度。Hierholzer algorithm 为 $O(|E|)$。但 assembly 的困难不在于跑 Eulerian path，而在于 noisy reads 产生的巨大多重图、repeat 跨度、paired-end 约束和多倍体 variation。

## 编辑距离

Levenshtein edit distance 以 insertion、deletion 与 substitution 的最小次数定义字符串差异；longest common subsequence 则最大化保持相对顺序的公共子序列长度。二者都可用 $O(nm)$ 动态规划求解，分别体现最小代价与最大得分视角，是从精确字符串匹配进入生物序列比对的桥梁。

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

精确索引用于定位相同短词。存在 substitution 与 gap 时，需要把匹配改写成带代价的最优路径；BLAST 则先用 seed 定位候选，再进行局部延伸。

<br>

# 双序列比对

## 概念：双序列比对

- **研究对象与解释边界**
  - 比对依据 substitution、insertion 和 deletion 模型建立演化或功能上的对应关系，字符串排齐只是结果的可视化；
  - 相似性是可计算量；
  - 同源性表示共同祖先关系，通常不写成“百分之多少同源”。
- **输入、输出与任务选择**
  - 输入是两条序列以及替换分数、gap-open、gap-extension 和边界约定；
  - 输出是最优分数、一条或多条 alignment，以及每列对应关系；
  - Global alignment 解释两条完整序列；
  - local alignment 寻找高分局部区域；
  - semi-global alignment 允许指定末端不受罚；
  - 选择哪一种由生物问题决定，不是实现细节。
- **算法主线与章节衔接**
  - 动态规划把所有可能 alignment 表示成网格路径，并利用最优子结构避免指数枚举；
  - BLAST 用启发式方法缩小局部 DP 的搜索范围；
  - 多序列比对把字符对打分推广为 profile 列打分；
  - Pair HMM 把相同状态结构解释为生成概率。

给定替换分数 $s(a,b)$ 与 gap 罚分 $g$，全局比对适合全长相关序列，局部比对适合共享局部片段的序列。

## 路径表示与评分

一个 alignment column 只有三种合法情况：$(x_i,y_j)$、$(x_i,-)$、$(-,y_j)$，不能是 $(-,-)$。因此比对等价于从网格左上角走到右下角：对角步消耗两个字符，向下或向右只消耗一个。每条路径是一种 alignment，路径权重是各列分数之和。动态规划在每个格点只保留到达该点的最佳前缀，因为 additive score 使未来与更早路径细节无关。

Gap 不能免费。若 match 为正、mismatch 为负而 gap 为 0，算法可以在两条序列中任意插入 gap，绕开多数 mismatch，只累加偶然 match，得到缺乏生物意义的高分。Gap-open 与 extension penalty 分别表达“启动一次 indel 事件”和“延长同一事件”的代价；全局与局部比对的核心差异则在边界初始化、终点和 traceback 停止条件。

## Needleman–Wunsch

全局比对递推为

$$
F_{i,j}=\max\begin{cases}
F_{i-1,j-1}+s(x_i,y_j)\\
F_{i-1,j}+g\\
F_{i,j-1}+g
\end{cases}
$$

边界为 $F_{i,0}=ig$、$F_{0,j}=jg$。填表为 $O(nm)$ 时间；一次 traceback 最多走 $n+m$ 步，为 $O(n+m)$ 时间。保存全表与回溯指针需 $O(nm)$ 空间；只求分数可滚动到 $O(\min(n,m))$。Hirschberg 分治可在线性空间恢复比对。

最优子结构可用反证法说明：若通往 $(i,j)$ 的最优路径最后一步来自 $(i-1,j-1)$，但此前缀不是 $(i-1,j-1)$ 的最优路径，把它替换成更优前缀即可提高整条路径分数，与“原路径最优”矛盾。来自另两个方向时同理。

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

当多个候选同分时，代码中 dictionary insertion order 决定优先选 `D`、`U` 或 `L`。不同 tie-breaking 可能返回不同 alignment，但最优分数相同。科学报告应说明是只需一个 optimum，还是需要枚举、计数或采样多个 optimum。

> [!EXAMPLE]+ 例2：全局比对
>
> 对 `GATTACA` 与 `GCATGCU` 使用 match $+1$、mismatch $-1$、gap $-1$，一种最优解为：
>
> ```text
> G-ATTACA
> GCA-TGCU
> ```
>
> 该答案依赖参数。提高 gap-open penalty 后，算法会倾向于选择 mismatch 而不是多个短 gap。所得 alignment 是当前评分模型下的最优解。

## Smith–Waterman

局部比对把 0 加入候选：

$$
H_{i,j}=\max\left(0,H_{i-1,j-1}+s(x_i,y_j),H_{i-1,j}+g,H_{i,j-1}+g\right)
$$

traceback 从全表最大值开始，在 0 处停止。0 相当于允许丢弃负贡献前缀。该算法保证最优局部比对，但长序列数据库上的 $O(nm)$ 成本催生了 BLAST 等启发式方法。

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

> [!EXAMPLE]+ Smith–Waterman 手算
>
> 对 `TGTTACGG` 与 `GTTGACTA` 使用 match $+3$、mismatch $-3$、linear gap $-2$，DP 表的最高分为 13。一条最优局部比对是：
>
> ```text
> GTT-AC
> GTTGAC
> ```
>
> 五个 match 得 $15$，一个 gap 扣 $2$。回溯从分数 13 的格点开始，在首次到达 0 时停止，因此不会强制消费两端未参与局部同源区的字符。

全局与局部之间还有 semi-global/overlap alignment。若令 $x$ 为必须完整消费的短 read、$y$ 为两端免费的 reference，则初始化 $F_{0,j}=0$，终点取 $\max_jF_{n,j}$，traceback 到第 0 行停止。其他 overlap 变体才可能从最后一行或最后一列选择终点。算法主体相同，差异集中在初始化、终点与停止条件。

局部 alignment score 的统计分布与全局不同。在满足一定条件时，随机序列最大局部得分近似极值分布，这正是 BLAST E-value 理论的基础。

## 仿射 gap

连续插入/删除通常比多次独立 indel 更合理。仿射罚分写作

$$
G(k)=g_o+(k-1)g_e
$$

其中 $g_o$ 是 gap-open，$g_e$ 是 gap-extension。Gotoh 算法用 match、在 $x$ 中开 gap、在 $y$ 中开 gap 三张 DP 表，仍保持 $O(nm)$ 时间。

令 $M_{i,j}$ 表示最后一列为字符对，$I^x_{i,j}$ 表示最后一列为 $(x_i,-)$，$I^y_{i,j}$ 表示最后一列为 $(-,y_j)$。一种约定下：

$$
M_{i,j}=\max(M_{i-1,j-1},I^x_{i-1,j-1},I^y_{i-1,j-1})+s(x_i,y_j)
$$

$$
I^x_{i,j}=\max(M_{i-1,j}-g_o,\ I^x_{i-1,j}-g_e)
$$

$$
I^y_{i,j}=\max(M_{i,j-1}-g_o,\ I^y_{i,j-1}-g_e)
$$

若定义 $G(k)=g_o+(k-1)g_e$，从 $M$ 开 gap 时扣 $g_o$，延长时扣 $g_e$。有些软件定义 $G(k)=g_o+kg_e$，会在开 gap 的第一格同时扣 extension。参数数值不能脱离具体约定比较。

初始化时 $M_{0,0}=0$，不可能状态设为 $-\infty$；第一列只允许 $I^x$ 链，第一行只允许 $I^y$ 链。若错误地把所有边界设为 0，就会把 global alignment 变成 free-end-gap 版本。

上述递推采用“不允许两个相反方向 gap 紧邻”的规范化 alignment。常见生物序列实现禁止这种等价于插入后立即删除的冗余列组合；若评分体系允许这种路径并需枚举全部形式，则要显式加入 $I^x\leftrightarrow I^y$ 转移。

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

## Pair HMM 解释

Gotoh 的 $M,X,Y$ 三张表也可解释为 pair HMM 的三个隐状态。$M$ 同时发射 $(x_i,y_j)$，$X$ 发射 $(x_i,-)$，$Y$ 发射 $(-,y_j)$。把相关序列 pair-HMM 与独立背景模型做 log-odds，可得到：

- $M$ 的 emission log-odds 对应 substitution score；
- $M\rightarrow X/Y$ 的 transition log-odds 对应 gap-open；
- $X\rightarrow X$、$Y\rightarrow Y$ 对应 gap-extension；
- begin/end transition 与允许的入口、出口共同决定 global、local 或 semi-global 边界。

Gotoh 使用 `max`，因此求得最可能 alignment path，即 Viterbi alignment。若把每个状态转移的 `max` 改为 log-sum-exp，则得到所有 alignment 路径的总 likelihood。两条序列可能存在许多近似等价 alignment，Viterbi 只保留其中一条；posterior alignment 可进一步计算某对残基同源对应的边缘概率。

分数到概率参数并非逐项取指数即可完成。Transition row 必须归一化，emission 也必须相对明确的 null model；常数缩放与终止概率会影响概率模型，却可能不改变最优路径。因此“仿射 gap 像 HMM”是结构对应关系，不表示任意经验打分参数天然构成合法概率分布。

## 打分矩阵

DNA 可用 match/mismatch；蛋白质需利用替换频率。PAM 从近缘序列估计演化替换再外推，BLOSUM 从保守区块直接统计。矩阵元素本质是 log-odds：

最简单的 DNA identity matrix 在 A/T/C/G 相同配对上取 1，不同配对上取 0。它适合演示 DP 填表，却没有惩罚 mismatch，也没有表达 transition、transversion 或背景碱基频率差异。实际比对通常使用正 match、负 mismatch，并与 gap penalty 联合设定。

$$
s(a,b)=c\log\frac{p_{ab}}{q_aq_b}
$$

其中 $c>0$ 决定分数单位与取整尺度，不应预先把它等同于后文 Karlin–Altschul 统计参数 $\lambda$。正分表示该配对在相关序列中比随机背景更常见。BLOSUM62 是常用默认值，不代表所有演化距离都最优。

PAM1 近似每 100 个残基发生 1 个 accepted point mutation，再通过矩阵幂外推较远距离；编号越大代表距离越远。BLOSUM 从保守 block 聚类统计，编号越低通常代表把相似度较低的序列合并、面向更远关系。两套编号方向相反。

替换矩阵与 gap penalty 必须联合校准。若平均随机 alignment score 非负，局部比对可以通过无限延长随机区域提高分数，Karlin–Altschul 的标准条件不再成立。

> [!EXAMPLE]+ 例3：比对分数不等于概率
>
> 同一对序列使用不同矩阵、gap 参数或局部/全局目标，会得到不同最优比对。原始分数只有在指定评分系统后才有意义；跨搜索比较通常使用 bit score 和 E-value。

## 正确性检查

实现比对时可使用以下 invariant：

- `dp[i][j]` 只依赖已计算的左、上、左上；
- traceback 每步至少减少 $i$ 或 $j$，必然终止；
- 重算 traceback alignment 的列分数应等于终点 DP 值；
- alignment 去掉 gap 后应恢复原序列；
- 交换两序列时，若评分对称，最优分数应相同；
- 将所有分数乘正常数不会改变最优路径；
- local score 不应为负。

成熟库与手写实现结果不同时，应先核对 end-gap、gap convention、matrix alphabet、unknown residue 和 tie-breaking，而不是立即判断其中一个错误。

<br>

<br>

# 数据库搜索

## 概念：数据库搜索

- **核心问题与计算取舍**
  - 数据库搜索是“一条 query 对大量候选”的相似性检索；
  - 对数据库中的每条序列完整运行 Smith–Waterman 虽然精确，但成本通常不可接受；
  - BLAST 先用短 seed 找可能相关区域，只在候选附近延伸，以牺牲部分敏感度换取速度，并未另行定义一种相似性。
- **输入与输出**
  - 输入包括 query、数据库版本、替换矩阵、gap 参数、masking 和显著性阈值；
  - 输出是若干局部命中，每个命中带 alignment、bit score、E-value、identity 和 coverage；
  - 原始分数描述当前打分体系下的相似程度；
  - E-value 回答给定搜索空间中随机出现同等高分结果的期望数量。
- **章节衔接与解释边界**
  - 本章承接局部比对，并新增“搜索空间”和“零模型”；
  - 命中序列可进入 MSA、profile 或结构模板搜索；
  - 低 E-value 支持非随机相似，却不能单独证明功能相同、正交同源或相同结构域架构。

## BLAST

BLAST 采用 seed-and-extend。把课程中的执行链完整展开，可写成：

1. 对 query 做 low-complexity masking；
2. 从 query 枚举长度为 $k$ 的 word；
3. 对蛋白质 query，用替换矩阵生成分数超过阈值 $T$ 的邻居词；
4. 建立 query word lookup table，某些实现或模式也可预建数据库索引；
5. 在数据库中寻找 word hit；
6. 先做 ungapped extension，形成 high-scoring segment pair（HSP）；
7. 用 Karlin–Altschul 零模型筛除不显著 HSP；
8. 合并或连接相容的邻近 HSP，并做 gapped extension；
9. 报告局部比对、bit score、E-value、identity 与 coverage。

最后的 gapped alignment 可视为只在 seed 支持区域附近执行受限的 Smith–Waterman 式搜索。不同 BLAST 版本的具体阶段、two-hit 规则和统计修正并不完全相同，九步链用于理解启发式筛选，不应误当所有实现逐行一致的规范。

原始 BLAST 对蛋白 query 枚举超过阈值 $T$ 的相似 word，而不是只找完全相同 word。现代实现有 two-hit heuristic：同一对角线上短距离内出现两个 seed 后才延伸，从而减少随机 seed 的开销。ungapped extension 的分数从峰值下降超过阈值 $X$ 时停止；通过筛选后再做 gapped extension。

Karlin–Altschul 理论给出

$$
E=Kmn e^{-\lambda S}
$$

$m,n$ 是经过 edge correction 的有效 query 长度与有效数据库搜索空间，并不总等于原始字符数；$S$ 是原始分数，$K,\lambda$ 由评分体系决定。E-value 是随机数据库搜索中分数至少这么高的期望命中数，不是“结果错误的概率”。数据库越大，同一原始分数的 E-value 越差。

bit score 把不同评分体系标准化：

$$
S'=\frac{\lambda S-\ln K}{\ln 2},
\qquad E=mn\,2^{-S'}
$$

$E=0.01$ 表示每次搜索的随机高分命中期望数为 0.01；在 Poisson 近似下，至少出现一个随机命中的概率约为 $1-e^{-0.01}\approx0.00995$，不是“该结果为假的概率”，也不校正数据库 annotation error。

identity、positive、coverage 和 E-value 回答不同问题。短片段可以有 100% identity 却覆盖极少；长 alignment 可能 identity 不高但统计上极显著。判断 domain-level homology 时通常要求 query/subject coverage 与 domain boundary 一并检查。

## 敏感度

- seed 越短，敏感度越高、候选越多；
- low-complexity masking 防止重复区域支配结果；
- PSI-BLAST 从迭代命中构建 position-specific scoring matrix；
- profile HMM 通常比单一 query 更能检出远缘同源；
- reciprocal best hit 只是 ortholog 的启发式证据，不处理复杂复制与丢失历史。

实际工作中应记录 BLAST 版本、数据库版本、E-value 阈值、覆盖率、identity、masking 和过滤规则。

`blastn`、`blastp`、`blastx`、`tblastn` 与 `tblastx` 的 query/database 字母表及翻译方向不同，搜索空间也不同。蛋白搜索还可采用 composition-based score adjustment，减少氨基酸组成偏斜造成的虚假高分。报告中还应记录程序类型、矩阵、gap 参数与有效搜索空间。

PSI-BLAST 从第一轮命中构建 PSSM，再重复搜索。远缘 homolog 可逐轮加入，但一次 false positive 也可能污染 profile 并发生 profile drift。每轮纳入阈值通常应比最终报告阈值更严格，并人工检查 domain architecture。

> [!EXAMPLE]+ 例4：两个显著命中
>
> 命中 A 覆盖 query 的 95%，identity 32%；命中 B identity 80%，却只覆盖一个 20-residue low-complexity segment。若只按 identity 排序会选 B；综合 coverage、masking、domain 与 E-value 后，A 往往更能支持全长同源。

<br>

<br>

# 多序列比对

## 概念：多序列比对

- **研究对象**
  - 多序列比对为三条以上同源序列建立共同列坐标；
  - 某一列中的残基被解释为来自同一祖先位置，gap 表示相对该坐标系发生的插入或缺失；
  - MSA 不是把若干 pairwise alignment 简单并排，因为每次插入 gap 都必须同步更新已有序列的列结构。
- **输入、输出与计算近似**
  - 输入是一组序列，常伴随 pairwise distance 或 guide tree；
  - 输出是带 gap 的矩形 alignment、列分布和不确定区域；
  - MSA 支持保守位点识别、PWM/PSSM、Profile HMM、系统发育、共进化和结构预测；
  - 精确多维 DP 随序列数指数增长，因此实际算法通常先合并近邻 sequence/profile，再用 refinement 修正早期决策。
- **解释边界**
  - Guide tree 只规定渐进合并顺序，不自动等于物种树；
  - MSA 的列对应不是无误差标签；
  - 下游建树、功能位点和结构约束都应检查不稳定列对结论的影响。

## 难点

多序列比对同时安排多个序列的列对应。对 $k$ 条长度约为 $n$ 的序列，朴素精确 DP 时间为 $O((2^k-1)n^k)$、空间为 $O(n^k)$；常用工具因此采用 progressive alignment：

1. 计算两两距离；
2. 建 guide tree；
3. 按树从近到远合并序列或 profile；
4. 可再迭代优化。

Clustal 系列体现 progressive 思路，MUSCLE 加入快速距离估计和 refinement，MAFFT 利用 FFT 与不同策略平衡速度和精度。早期错误会被 profile 固化，因此 guide tree 不是最终系统发育树。

profile–profile alignment 不再比较两个字符，而是比较两列分布。若列 $i,j$ 的 residue frequency 分别为 $p_i(a)$、$q_j(b)$，期望 substitution score 可写为

$$
s(i,j)=\sum_{a,b}p_i(a)q_j(b)S(a,b)
$$

还需对 gap-rich column、sequence weight 和 pseudocount 处理。sequence weighting 防止某个过度采样物种支配 profile。

- **课程复习材料中的 synthetic sequence**
  - progressive MSA 被概括为“pairwise alignment、聚类、synthetic sequence、重新比对”；
  - synthetic sequence 更准确地指由已对齐序列形成的 profile 或 consensus-like 表示，并非伪造一条具有真实生物来源的序列；
  - 只取单一 consensus 会丢失列内变异；
  - 实际 profile–sequence 或 profile–profile DP 保留每列 residue frequency、gap frequency 和权重；
  - Henikoff weighting、基于 guide tree 的 weighting 等方法用于降低近重复序列的支配作用。

progressive alignment 的详细链条是：

1. 用 $k$-mer 或快速 pairwise score 得到距离矩阵；
2. 用 UPGMA/NJ 建 guide tree；
3. 叶节点是 sequence profile；
4. 内节点按 pairwise DP 对齐两个 profile；
5. 已对齐 block 作为整体传播到更高节点；
6. refinement 拆开部分 alignment 后重新对齐，尝试修复早期错误。

若有已知结构或可靠 motif，可加入 constraint。对于全局同源但有长插入的序列，MAFFT L-INS-i 一类局部 pairwise 信息可能更准确，但成本更高。

下面的教学实现完成一次 profile–profile DP，并按给定 guide order 渐进合并。它直接优化未加权 sum-of-pairs；成熟软件还会加入 sequence weight、位置特异 gap、局部 alignment 与 refinement。

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

`progressive_msa` 的输入顺序承担了 guide tree 的作用，因此不能用于方法评价。完整流程应先由 pairwise distance 构建 guide tree，再按内部节点调用 `profile_profile_align`；同一个 UPGMA tree 可用于指导 MSA，但 guide tree 是计算顺序，不应直接报告为物种树。

`progressive_msa_with_upgma` 把这条链实际连通：pairwise distance 决定下一次合并的 profile，profile–profile DP 决定列对应，合并后的簇再按成员数更新距离。初始距离可以来自快速 $k$-mer、编辑距离或 pairwise alignment；若来自未对齐序列，不能直接使用要求同源列的 p-distance。最终 alignment 改变后可重新估计距离和 guide tree，这就是 iterative refinement 的一个来源。

## 评价

sum-of-pairs 分数对每一列中的所有序列对求和，但与真实演化模型并不等价。benchmark 常看参考结构同源位点能否恢复。含大量 indel、低复杂度或无序区时，单一确定比对会隐藏不确定性。

多序列比对的下游用途包括保守位点、profile、系统发育、共进化、结构预测和功能注释。任何下游结论都应考虑比对误差传播。

需要区分作为优化目标的 sum-of-pairs 总分，与相对参考比对计算的 SP benchmark score。后者通常是参考 alignment 中 residue pair 被预测结果恢复的比例，部分 benchmark 还另报 precision，必须说明分母。TC 要求整列完全正确，更严格。真实数据通常没有 ground truth，可用结构 alignment、模拟序列或算法间稳定性间接评估。

Progressive alignment 之外，consistency-based 方法利用多个 pairwise alignment 间的传递支持，iterative refinement 则反复拆开并重新对齐 profile。对不稳定区应保留 alignment confidence 或比较多个合理 MSA，而不是把单一输出视作无误差同源关系。

下游建树前通常应 trim 极不可靠列，但过度 trimming 会删除快速演化信号。分析应报告原始/过滤长度和工具参数。

<br>

# Motif

## 概念：Motif

- **研究对象**
  - Motif 是比完整基因或结构域更短、在一组对象中反复出现的局部模式；
  - 序列 motif 可以表示转录因子结合位点、剪接信号或蛋白质功能片段；
  - 它允许部分位置高度保守，另一些位置可变；
  - 共识字符串只记录最常见字符，PWM 记录每个位置的完整字符分布。
- **两类计算任务**
  - 发现任务输入一组序列和背景模型，输出 motif 模型及每条序列中的候选位置；
  - 扫描任务输入已有 PWM 和新序列，输出各窗口的 log-odds 分数；
  - OOPS、ZOOPS、TCM 分别规定每条序列恰好一次、至多一次或可多次出现，因此对应不同统计模型。
- **章节衔接**
  - Motif 把 MSA 中“整条序列的共同列”收缩为局部模式；
  - PWM 假设 motif 内各位置独立；
  - HMM 可加入状态转移和可变长度；
  - 三维结构 motif 允许序列上相距很远的残基在空间中共同形成位点。

## 表示

motif 是在一组序列或结构中反复出现、与功能或约束相关的局部模式；它通常短于完整结构域，但长度不是定义条件。共识字符串会丢失每个位点允许的替换，position weight matrix（PWM）保留位置特异分布：

$$
W_{i,b}=\log\frac{p_{i,b}}{q_b}
$$

长度 $L$ 的窗口 $x$ 得分为 $\sum_{i=1}^{L}W_{i,x_i}$。伪计数避免未观察碱基产生 $-\infty$。

PWM 假设 motif 内各位置条件独立。若 position 2 与 5 有协同约束，独立 PWM 无法表达；dinucleotide model、Bayesian network 或神经网络可以加入依赖，但需要更多数据。

information content 衡量 motif 分布相对背景的差异。一般背景 $q_b$ 下：

$$
IC_i=D_{\mathrm{KL}}(p_i\|q)
=\sum_b p_{i,b}\log_2\frac{p_{i,b}}{q_b}
$$

只有 DNA 均匀背景 $q_b=1/4$ 时才化为 $IC_i=2-H_i$，并约定 $0\log0=0$。sequence logo 的字母高度通常结合 $IC_i$ 与频率，保守位置更高。

## 发现

- **枚举**：小 motif 可枚举候选，代价随长度指数增长。
- **EM**：把 motif 位置当隐变量，在期望位置和 PWM 参数间迭代。
- **Gibbs sampling**：逐条序列重采样 motif 位置。
- **富集检验**：比较正集和背景集中的出现率。

EM 只保证收敛到局部最优；初始化、背景模型和序列组成偏差都会改变结果。扫描整基因组还会引入海量多重检验。

以“每条序列恰有一个 motif”为例，隐变量 $z_{n,j}$ 表示第 $n$ 条序列的 motif 从位置 $j$ 开始。E-step 根据当前 PWM 计算 posterior responsibility：

$$
\gamma_{n,j}
=P(z_n=j\mid x_n,\theta)
\propto P(z_n=j)P(x_{n,j:j+L}\mid\theta_{\text{motif}})
P(x_{n,\text{background}}\mid\theta_{\text{bg}})
$$

M-step 用 $\gamma_{n,j}$ 加权统计每个位置的碱基频率并重新估计 PWM。多次随机初始化后比较 held-out likelihood，可降低局部最优风险。

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

`scan_pwm` 计算的是给定模型下的窗口分数，不含全基因组多重扫描校正。`motif_em_oops` 只实现 OOPS occurrence model；ZOOPS 还需要“本序列无 motif”的隐状态，TCM 则允许多个位置同时出现。初始化在第一批窗口上可能造成局部最优，实际应以不同 seed 重启并比较 held-out likelihood。

Motif EM 与后文 Baum–Welch 使用同一原则：E-step 对隐变量求 posterior，M-step 用期望计数更新参数。区别在隐变量结构。OOPS motif 的隐变量是每条序列的单一起点，候选数随序列长度增长；HMM 的隐变量是受 transition 约束的整条状态路径，forward–backward 用动态规划汇总指数多路径。

Gibbs sampling 每次暂时移除一条序列，用其余序列构建 PWM，再按 posterior 从被移除序列中采样新位置。输出是一组 posterior sample；还需检查 burn-in、mixing 和多链一致性。

Motif discovery 还必须声明 occurrence model：OOPS 假设每条序列恰有一次，ZOOPS 允许零次或一次，TCM 允许多次。背景模型也可从零阶频率扩展为一阶或更高阶 Markov model，以控制局部组成和重复序列造成的假 motif。

1D motif 主要约束残基在线性序列中的顺序，可连续，也可含 wildcard、gap 或可变间隔；3D motif 则约束指定原子或残基的空间关系，序列顺序与间隔可以很弱。

功能相关的 1D motif 在折叠蛋白中通常对应某种 3D 微环境，但逆命题不成立。来自序列上相距很远的残基可在折叠后形成 active site，不同序列和 fold 也可能通过 convergent geometry 形成相似功能位点，因此清晰的 3D motif 未必存在可识别的线性 consensus。

<br>

# 隐马尔可夫模型

## 概念：隐马尔可夫模型

- **研究对象**
  - 隐马尔可夫模型描述一类序列过程：能够观察到字符序列，真正决定字符分布的状态不可直接观察；
  - DNA 碱基可被观察，“GC-rich 区域”或“AT-rich 区域”是需要推断的隐状态；
  - 蛋白质家族中某个残基来自 Match、Insert 还是 Delete 也属于隐路径。
- **模型组成、输入与输出**
  - HMM 由初始分布、状态转移概率和发射概率组成；
  - 给定模型与观测序列，可以计算整条序列的 likelihood、找一条最可能状态路径，或计算每个位置属于各状态的 posterior；
  - 给定未标注序列，还可用 Baum–Welch 估计参数；
  - 输入是观测序列与模型参数，输出分别对应 evaluation、decoding 和 learning 三类问题。
- **与比对的关系**
  - HMM 与序列比对使用相同的动态规划思想，但目标不同；
  - Viterbi 用 `max` 保留最可能路径，Forward 用求和汇总所有路径；
  - Pair HMM 把 alignment 看作隐路径；
  - Profile HMM 把 MSA 的保守列建成 Match 状态，并用 Insert/Delete 描述 indel。
- **章节衔接**
  - HMM 承接 MSA 的列分布与 Motif 的位置特异概率，并把独立位置推广为有状态转移的生成模型；
  - 系统发育同样在隐祖先状态上求和；
  - RNA covariance model 加入成对位置；
  - FATHMM 把 Profile HMM 用于变异效应评分。

## 模型

HMM 由隐状态 $z_t$、转移概率 $a_{ij}=P(z_t=j\mid z_{t-1}=i)$ 和发射概率 $e_j(x_t)=P(x_t\mid z_t=j)$ 构成。联合概率为

$$
P(x,z)=P(z_1)e_{z_1}(x_1)\prod_{t=2}^{T}a_{z_{t-1},z_t}e_{z_t}(x_t)
$$

三个基本问题：

- **Evaluation**：序列在模型下的概率，forward 算法；
- **Decoding**：最可能状态路径，Viterbi 算法；
- **Learning**：未知参数估计，Baum–Welch/EM。

三者均利用 Markov 条件独立性避免指数路径枚举。Forward 与 Viterbi 在稠密转移下为 $O(TK^2)$；Baum–Welch 的每次 EM 迭代为 $O(TK^2)$，若迭代 $I$ 次则总计 $O(ITK^2)$。

一阶 HMM 假设当前隐状态只依赖上一状态，当前观测只依赖当前隐状态：

$$
P(z_t\mid z_{1:t-1})=P(z_t\mid z_{t-1}),\qquad
P(x_t\mid z_{1:t},x_{1:t-1})=P(x_t\mid z_t)
$$

一阶 Markov 假设是为获得可计算模型而采用的近似。更长的记忆可以通过扩展状态表示，代价是状态数和参数量增加。

## Forward

定义 $\alpha_t(j)=P(x_{1:t},z_t=j)$：

$$
\alpha_1(j)=\pi_j e_j(x_1)
$$

$$
\alpha_t(j)=e_j(x_t)\sum_i\alpha_{t-1}(i)a_{ij}
$$

最终 likelihood 为 $P(x)=\sum_j\alpha_T(j)$。forward 汇总所有可能状态路径，因此可以比较模型、训练参数或计算 posterior。

## Backward

定义 $\beta_t(i)=P(x_{t+1:T}\mid z_t=i)$：

$$
\beta_T(i)=1,\qquad
\beta_t(i)=\sum_j a_{ij}e_j(x_{t+1})\beta_{t+1}(j)
$$

forward 与 backward 结合得到单点 posterior：

$$
P(z_t=i\mid x)=\frac{\alpha_t(i)\beta_t(i)}{P(x)}
$$

posterior decoding 在每个位点选 marginal probability 最大状态；其拼接路径不一定是合法转移路径。Viterbi 则保证整条路径合法并最大化 joint probability，两者不应混称。

## Viterbi

定义 $\delta_t(j)$ 为以状态 $j$ 结束的最佳路径 log-score：

$$
\delta_t(j)=\log e_j(x_t)+\max_i
\left[\delta_{t-1}(i)+\log a_{ij}\right]
$$

除分数外还要保存 $\psi_t(j)=\arg\max_i(\cdots)$，否则无法 traceback。稀疏 transition graph 可把 $K^2$ 降为实际边数 $|E|$，复杂度为 $O(T|E|)$。

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

概率连乘会下溢，故实现应采用 log-space；forward 中的求和需使用 `logsumexp`，Viterbi 则把求和替换为最大值。示例字典必须覆盖所有允许的初始、转移与发射组合；不允许组合应显式记为 `-math.inf`。

## Baum–Welch

参数未知且训练序列没有 state label 时，Baum–Welch 使用 EM。E-step 计算期望转移计数

$$
\xi_t(i,j)=P(z_t=i,z_{t+1}=j\mid x)
=\frac{\alpha_t(i)a_{ij}e_j(x_{t+1})\beta_{t+1}(j)}{P(x)}
$$

以及状态 posterior $\gamma_t(i)$。M-step 归一化期望计数：

$$
\hat a_{ij}
=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}
{\sum_{t=1}^{T-1}\gamma_t(i)}
$$

发射概率按状态 $i$ 占据时观察到各符号的期望次数归一化。每次迭代不会降低数据 likelihood，但可能落入局部最优，并可能把某些概率估为 0；初始化、pseudocount 与多次重启很重要。

下面实现一次带 scaling 的 Baum–Welch 更新。输入参数使用普通概率而非 log probability，便于直接看到期望计数如何进入 M-step。

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

对多条独立序列训练时，E-step 分别计算每条序列的 $\gamma,\xi$，M-step 再汇总期望计数；不能把多条序列直接首尾拼接，否则会人为加入跨序列转移。反复调用 `baum_welch_step` 即形成 EM 迭代，但停止条件应比较总 log-likelihood 或参数变化，并用独立 validation sequence 防止过拟合。

训练应以 validation likelihood、参数变化或最大迭代次数停止，并记录随机初始化。训练 likelihood 单调上升不代表 validation likelihood 改善，也不能保证参数可识别。

## Profile HMM

序列家族的 profile HMM 通常为每个保守列建立 Match 状态，并提供 Insert、Delete 状态。它同时表示位置特异替换和 indel，比 PWM 更完整。HMMER 用 profile HMM 搜索蛋白质家族；Pfam 用它组织 domain。

Match $M_k$ 发射第 $k$ 个共识位置的 residue distribution；Insert $I_k$ 发射背景式 residue，并可自循环产生任意长度插入；Delete $D_k$ 不发射字符，只跳过共识位置。Delete 是 silent state，forward/Viterbi 实现必须按拓扑顺序传播，不能把它当作消耗一个观测的普通状态。

Profile HMM 的 DP 有两个坐标：query 前缀长度 $i$ 与 profile 位置 $k$。$M_{i,k}$ 同时消费 query 字符和 profile 列，$I_{i,k}$ 只消费 query 字符，$D_{i,k}$ 只推进 profile。下面给出固定转移参数的最小 Viterbi 实现；`match_emit[k]` 与 `insert_emit[k]` 均为 log emission probability。

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

代码把 begin/end 简化为虚拟 $M_0$ 和“在第 $L$ 列取最大值”，不含 HMMER 的 local-entry、local-exit 与 null model。其关键仍是 delete state 的更新顺序：同一 query 行内从小 $k$ 向大 $k$ 传播，确保 silent transition 不会错误消费字符。

构建 profile 时，常先由 MSA 决定哪些列属于 match，再用 sequence weighting、effective sequence number 和 Dirichlet mixture prior 平滑发射。最终搜索分数通常是 family model 相对 null background 的 log-odds，而不是单纯 $P(x\mid\text{profile})$。

`match_emissions_from_msa` 只估计 Match emission。完整训练还要把 match column 之间的 gap run 映射为 Delete，把非 match column 中的 residue 映射为 Insert，再统计 M/I/D transition 与 Insert emission；sequence weighting 与 prior 应在计数归一化之前加入。这样形成的 profile HMM 才把 MSA 的列分布与 indel pattern 同时编码。

HMMER 的 acceleration pipeline 先用便宜 filter 排除明显不匹配序列，再做更完整推断。这与 BLAST 的 seed-and-extend 一样体现共同工程原则：先用高召回低成本阶段缩小候选，再在候选上做昂贵精确计算。

> [!INFO]+ HMM 与比对
>
> 当替换分数和 gap 罚分可以解释为 pair-HMM 路径的 log-odds 时，Needleman–Wunsch 对应该模型的 Viterbi 解；将 max-product 改为 sum-product 可汇总全部 alignment 路径。两种递推都是对隐变量路径的动态规划。

<br>

> [!EXAMPLE]+ 例5：基因区段识别
>
> 两状态 HMM 可令 $z_t\in\{\text{GC-rich},\text{AT-rich}\}$，每个状态发射不同碱基分布，较高 self-transition 表示区段具有连续性。Viterbi 给出一条最可能 segmentation；posterior 则显示边界附近不确定性。若真实 GC 组成连续变化，两个离散状态只是粗略近似。

<br>

<br>

# 系统发育

## 概念：系统发育

- **研究对象**
  - 系统发育分析试图从现存序列推断共同祖先关系；
  - 树的叶是观测到的物种、基因或蛋白，内部节点表示未直接观测的祖先；
  - 拓扑表示分支关系，枝长可表示预期替换数或时间；
  - 有根树带演化方向，无根树只表示相对分裂关系。
- **输入、输出与方法分支**
  - 输入通常是 MSA，也可以是由 MSA 计算的 pairwise distance matrix；
  - 输出是树拓扑、枝长和分支支持度；
  - 距离法先把每对序列压缩成一个距离，再用 UPGMA 或 Neighbor Joining 建树；
  - 字符法直接使用每个 alignment column，以 parsimony、likelihood 或 Bayesian posterior 比较树。
- **章节衔接与解释边界**
  - 系统发育依赖前一章的列对应，也反过来影响协变与功能保守性的解释；
  - Guide tree 主要服务 MSA 的计算顺序；
  - phylogenetic tree 试图回答演化历史；
  - 两者即使拓扑相同，含义也不同。

## 距离法

UPGMA 每次合并距离最近的簇，并按簇大小更新平均距离。它假设 molecular clock，即所有叶到根距离相等；违反该假设会产生错误拓扑。Neighbor Joining 每轮最小化 $Q$-criterion 选择一对节点并更新距离；它不要求 ultrametric molecular clock，输出通常为 unrooted additive tree。

对当前 $n$ 个节点，Neighbor Joining 使用

$$
Q(i,j)=(n-2)d(i,j)-\sum_kd(i,k)-\sum_kd(j,k)
$$

选择最小的一对。新内部节点 $u$ 到其余节点 $k$ 的距离为

$$
d(u,k)=\frac{d(i,k)+d(j,k)-d(i,j)}{2}
$$

而 $i$ 侧分支长度为

$$
\delta(i,u)=\frac12d(i,j)
+\frac{\sum_kd(i,k)-\sum_kd(j,k)}{2(n-2)}.
$$

对 aligned DNA，最简单的 p-distance 是 mismatch column 比例，但多次 substitution 可能发生在同一位置，使观察距离低估真实变化。Jukes–Cantor 模型假设四种碱基平衡频率相同且替换率相同，修正距离为

$$
d_{\mathrm{JC}}=-\frac{3}{4}\ln\left(1-\frac{4}{3}p\right)
$$

当 $p\rightarrow0.75^{-}$ 时距离发散；有限样本若得到 $p\ge0.75$，JC69 距离没有有限实数估计，表示数据严重 saturation 或模型不适配。更复杂的 K80、HKY、GTR 区分 transition/transversion 与不同 equilibrium frequency。

UPGMA 合并簇 $A,B$ 后，与簇 $C$ 的距离按大小加权：

$$
d(A\cup B,C)=
\frac{|A|d(A,C)+|B|d(B,C)}{|A|+|B|}
$$

它得到 rooted ultrametric tree。Neighbor Joining 得到 unrooted tree；root 需要 outgroup、midpoint 或其他时间信息。

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

这两个函数假设输入距离完整且无缺失。非加性距离、抽样噪声或模型失配可能使 NJ 估计出负 branch length；代码保留该结果以暴露数据与树模型的冲突，生产软件可按明确规则截断或重新拟合。

## 字符法

- **Maximum parsimony**：寻找需要最少替换的树，直观但会受 long-branch attraction 影响。
- **Maximum likelihood**：指定替换模型，在树与分支长度下计算数据似然。
- **Bayesian inference**：结合树的先验，采样后验分布。

character-based 方法直接使用每个 taxon 在每个位点的状态，而不是先把整对序列压缩成一个距离。character 可以是 A/C/G/T、氨基酸，也可以是形态学或功能性状的 binary state。Binary coding 必须说明 0/1 是 absence/presence 还是 ancestral/derived；后者若无 outgroup 通常不能确定。

对固定二叉树和一个 character，Fitch algorithm 自底向上维护每个内部节点可取的最优状态集合。两个子集合有交集时取交集，不增加代价；无交集时取并集，并把变化数加一。它在线性时间内给出该拓扑的 unordered parsimony score，但寻找全局最优树仍需在巨大的 topology space 中搜索。

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

对 MSA 的总 parsimony score 是各列分数之和。Gap、歧义字符与加权替换需要显式约定；Fitch 的单位代价模型不区分 transition/transversion，也不估计 branch length。

Felsenstein pruning algorithm 在树上动态规划，把祖先状态的指数枚举降为每个位点 $O(|V||\Sigma|^2)$；对 $L$ 个独立位点和 $R$ 个离散速率类别，总复杂度约为 $O(LR|V||\Sigma|^2)$。

对于叶节点 $v$，若观察到碱基 $x$，conditional likelihood 为 $L_v(a)=1[a=x]$。内部节点 $v$ 有子节点 $u,w$ 时：

$$
L_v(a)=
\left(\sum_bP_{ab}(t_u)L_u(b)\right)
\left(\sum_cP_{ac}(t_w)L_w(c)\right)
$$

根处乘 equilibrium frequency 并求和得到该位点 likelihood。若各列独立，总 log-likelihood 是各列 log-likelihood 之和。树拓扑数量随 taxa 超指数增长，maximum likelihood 软件使用 NNI、SPR 等局部搜索而非穷举。

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

教学实现直接在概率空间递归，深树上会下溢。实际程序通常在每个内部节点缩放 conditional-likelihood vector 并累计 log scaling factor，或改用 log-space；多位点计算还会缓存相同 site pattern。

叶节点若为缺失、gap 或 IUPAC 歧义字符，应对所有兼容状态置 1，而不是强制 one-hot。真实位点演化速率通常不同，可用离散 Gamma 类别和 invariant-site component 近似。没有 outgroup 或 clock 信息时，unrooted likelihood tree 本身不提供演化方向。

parsimony 不需要完整替换概率，但会把所有变化视作同成本。快速演化长枝可能因平行变化被错误聚在一起，即 long-branch attraction。likelihood 也并非自动正确；若 substitution model 严重失配，仍会产生高置信错误树。

Maximum parsimony 的总分由发生变化的 informative column 决定；高度可变列可能贡献大量变化并主导拓扑，却也最容易出现 saturation 与 homoplasy。加权 parsimony 可以改变不同变化的代价，但权重本身仍是模型假设。Maximum likelihood 显式整合 branch length 与 substitution probability，因此能表达同一位点发生多次未观测替换的可能性，代价是模型选择和树空间搜索更复杂。

Bayesian phylogeny 的目标是

$$
P(T,\ell,\theta\mid X)
\propto P(X\mid T,\ell,\theta)
P(T,\ell,\theta)
$$

- **变量与计算**
  - $T$ 是拓扑，$\ell$ 是 branch length，$\theta$ 是替换模型参数；
  - Likelihood 仍由 pruning algorithm 计算；
  - MCMC 在拓扑与连续参数空间中提议 NNI/SPR、branch-length 和模型参数更新。
- **两类分支支持度**
  - Posterior clade probability 是给定模型、先验与数据后该 clade 的后验质量；
  - bootstrap support 是重复重采样 alignment column 时算法恢复该 clade 的频率；
  - 两者数值相似时也不能互换解释。
- **MCMC 诊断**
  - 检查多链混合、effective sample size、burn-in 与 posterior predictive model fit。

## bootstrap

从比对列有放回重采样，重复建树，某分支出现比例是 bootstrap support。它衡量数据重采样下的稳定性，不是该 clade 为真的后验概率。

标准 bootstrap 流程是：

1. 原 MSA 有 $L$ 列；
2. 有放回抽取 $L$ 列形成 replicate；
3. 每个 replicate 独立建树；
4. 将 replicate split 映射回主树；
5. 报告每个 split 的出现频率。

因为抽样单位是 alignment column，它默认列近似独立同分布。连锁、重组和结构相关会削弱该假设。

> [!WARNING]+ 基因树与物种树
>
> 基因复制、水平转移、不完全谱系排序和重组会使 gene tree 不等于 species tree。建树前还应检查 recombination、alignment quality 和模型适配。

<br>

<br>

# RNA

## 概念：RNA

- **研究对象**
  - RNA 既是序列，也是能够通过分子内碱基配对折叠的结构对象；
  - 二级结构记录哪些位置形成 A–U、G–C 或 G–U 配对，通常先忽略精确三维坐标；
  - 许多 RNA 的催化、识别和调控功能依赖这种配对结构，单看一级序列会漏掉重要约束。
- **输入、输出与方法层次**
  - 单序列折叠输入 RNA 序列；
  - 输出可以是配对集合、dot-bracket、minimum free energy 结构或 base-pair probability；
  - Nussinov 用最大配对数展示区间 DP；
  - 热力学方法进一步区分 stacking、hairpin、bulge、internal loop 和 multiloop；
  - 多序列分析利用补偿突变识别在序列变化后仍被保留的配对。
- **章节衔接与边界**
  - RNA 章节连接字符串 DP、概率 ensemble 和系统发育背景；
  - Covariance model 可以看成同时描述共识序列与成对结构约束的 Profile HMM 扩展；
  - 伪结会破坏普通非交叉区间 DP 的分解条件。

## 二级结构

RNA 的功能不仅由序列决定，还依赖碱基配对。二级结构常允许 A–U、G–C 和 G–U wobble。Nussinov 算法最大化配对数：

$$
N_{i,j}=\max\begin{cases}
N_{i+1,j}\\
N_{i,j-1}\\
N_{i+1,j-1}+\delta(i,j)\\
\max_{k}(N_{i,k}+N_{k+1,j})
\end{cases}
$$

其中当 $i,j$ 可合法配对且满足最小 loop 长度时 $\delta(i,j)=1$，否则该候选项取 $-\infty$。经典实现为 $O(n^3)$ 时间、$O(n^2)$ 空间。它忽略 stacking energy 和 loop 类型；实际自由能预测使用 nearest-neighbor 参数与更细的 loop 分解。

递推中的四项分别表示：

- $i$ 不配对；
- $j$ 不配对；
- $i,j$ 合法配对；
- 在 $k$ 处分割成两个独立子结构。

区间长度从短到长填表，保证右侧子问题已计算。traceback 需记录选择，才能恢复 dot-bracket structure，例如 `((...))`。

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

该代码只最大化配对数，用于展示区间 DP 与 traceback，不应替代 ViennaRNA 等 thermodynamic implementation。条件 `j - i > min_loop` 表示配对碱基之间至少有 `min_loop` 个位置；默认 3 即要求 $j-i-1\ge3$。

Nussinov 的“最大配对数”会把所有合法 pair 当作等价。Turner nearest-neighbor model 改以 stacking、hairpin、bulge、internal loop 和 multiloop 的实验自由能参数最小化 $\Delta G$。minimum free energy structure 仍只是单一最低点；partition function

$$
Z=\sum_s e^{-\Delta G(s)/(RT)}
$$

该式适用于 $\Delta G$ 以每摩尔能量表示；若使用单分子能量，分母应为 $k_BT$。它汇总所有结构 $s$，可计算 base-pair probability 和 ensemble uncertainty。Zuker/Turner DP 寻找 minimum-free-energy structure；McCaskill partition-function algorithm 则对全部允许结构求和。可靠报告应同时给出 MFE、ensemble diversity 与高概率配对。

Zuker 型 DP 不能只把 Nussinov 的 $+1$ 改成任意 pair energy。Nearest-neighbor 能量依赖 loop 类型与相邻碱基对，因此需要额外状态，例如“区间任意最优结构” $W(i,j)$ 与“$i,j$ 已配对” $V(i,j)$。$V$ 再区分 hairpin、stack、internal/bulge loop 和 multiloop。McCaskill 算法保留相同结构分解，但把最小化替换为 Boltzmann weight 求和。

下面是从 Nussinov 过渡到能量最小化的教学版本。它区分 GC、AU 和 GU 的 pair energy，但没有 stacking 与 loop penalty，因此不是完整 Turner/Zuker 实现。

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

这段代码只说明目标函数和 DP 状态如何改变。真实 folding 需要经实验标定的参数集、温度、盐条件和 loop-specific recurrence；生产分析应调用 ViennaRNA 等经过验证的软件。

## 协变

若结构中的一对碱基在演化中发生补偿突变，例如 G–C 变为 A–U，说明配对约束可能被保留。Covariance model 把 profile HMM 的线性依赖扩展到成对位置，可用于结构化 RNA 家族搜索，但计算更昂贵。

单纯 mutual information 会把 phylogenetic correlation 和间接相关也算入协变。可靠证据需要足够多样的同源序列、正确 alignment，并与共同祖先背景比较。

伪结使配对弧交叉，打破许多上下文无关式动态规划的分解假设，因此通常被排除或用近似方法处理。

<br>

# 蛋白质结构

## 概念：蛋白质结构

- **研究对象**
  - 蛋白质结构是原子在三维空间中的组织方式；
  - 氨基酸序列给出共价连接顺序，结构加入 backbone torsion、侧链方向、残基接触、结构域排布和多聚体界面；
  - 相同序列可因配体、修饰或环境处于不同构象；
  - 一个坐标文件只是某种实验条件下的结构模型，不是蛋白质全部状态。
- **输入与表示**
  - 计算输入通常来自 PDB/mmCIF，包括原子坐标、链、残基编号、占有率、实验方法和缺失信息；
  - 常见输出表示包括 Cartesian coordinate、distance matrix、contact map、torsion angle 和 secondary-structure label；
  - 这些表示分别适合叠合、检索、统计与预测，但保存的信息不同。
- **章节衔接**
  - 本章先说明结构对象本身以及实验坐标的可信边界；
  - 结构叠合比较两个构象；
  - 结构模式寻找局部功能几何；
  - 结构预测从序列估计坐标；
  - 分子力学在给定势能模型下研究坐标如何变化。

## 层次

- **一级结构**：氨基酸序列。
- **二级结构**：$\alpha$-helix、$\beta$-sheet、turn 等局部构象。
- **三级结构**：单条链的整体三维折叠。
- **四级结构**：多条链的复合物。

蛋白质主链重复单元包含 N、$C_\alpha$、羰基 C 和 O。肽键的 $\omega$ 二面角因部分双键性质而近似平面且多为 trans；主链构象主要由 $\phi$、$\psi$ 描述，侧链构象还涉及 $\chi$ 二面角。Ramachandran plot 展示 $\phi,\psi$ 因立体位阻和局部相互作用形成的允许区域。

在固定 bond length、bond angle，并把 $\omega$ 近似固定为 trans 的理想化主链模型中，长度为 $N$ 的链主要由约 $2N-2$ 个内部 $\phi/\psi$ torsion 自由度描述；末端缺少完整的二面角定义。这个计数是简化表示，不包括侧链 $\chi$、cis peptide、键角涨落和闭环约束。

20 种氨基酸侧链可按 hydrophobic、polar、charged、aromatic 等性质理解，但分类不是绝对边界。疏水效应推动非极性侧链埋入核心；hydrogen bond、salt bridge、van der Waals、disulfide bond 与溶剂共同稳定结构。结构稳定性是 folded ensemble 与 unfolded ensemble 的自由能差，不是“内部相互作用越多越稳定”的简单计数。

二面角由四个连续原子定义。由于 bond length 和 bond angle 变化较小，用 internal coordinate 表示可显著减少自由度；但最终与实验结构比较通常仍转成 Cartesian coordinate。

secondary structure 常由 DSSP 根据主链 hydrogen-bond geometry 赋值。helix、sheet 等标签是从连续几何离散化的结果，边界残基可能随结构或算法改变。

## 坐标与距离

平移和旋转会改变坐标，却不改变内部结构。距离矩阵

$$
D_{ij}=\|\mathbf{x}_i-\mathbf{x}_j\|_2
$$

天然具有刚体变换不变性。完整且精确的 Euclidean distance matrix 通常可把坐标确定到平移、旋转和反射，因此不能区分镜像构象。contact map 再把距离阈值化，还会丢失距离大小、方向和局部几何。

距离信息存在三个常见层级：完整距离矩阵保存所有 $D_{ij}$；distogram 把每个距离离散为区间分布；contact map 只保存 $1[D_{ij}<c]$。从坐标到这些表示是逐步有损的。给定平方距离矩阵 $D^{\circ2}$，classical multidimensional scaling 先双中心化：

$$
B=-\frac12JD^{\circ2}J,\qquad
J=I-\frac1n\mathbf1\mathbf1^\top
$$

若 $B$ 是半正定矩阵，其前三个正特征值和特征向量可恢复一组三维坐标。负特征值表示距离带噪声或并非严格 Euclidean；截断到三维还会产生近似误差。

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

`coordinates_from_distances` 返回的方向、符号与原坐标不必相同，Kabsch 对齐后才可比较；镜像仍无法由距离矩阵排除。Contact cutoff 还依原子定义，$C_\alpha$、$C_\beta$ 与最小重原子距离不能混用。

PDB 结构需检查：

- X-ray、NMR、cryo-EM 或计算预测来源；
- resolution、R-free 或局部分辨率；
- 缺失残基、alternate location、occupancy；
- biological assembly 与 asymmetric unit；
- ligand、金属、水和突变。

## 实验来源

X-ray crystallography 测量衍射强度，并经相位求解、电子密度重建和模型精修得到原子模型；resolution 数值越小通常表示可分辨细节越高。cryo-EM 从大量粒子图像重建 ensemble-averaged Coulomb-potential map，异质性和局部分辨率可能显著变化。NMR 条目常提交一组满足实验 restraint 的 conformer；这些 conformer 不能直接当作带真实群体权重的平衡 ensemble。B-factor 可反映位移、不确定性和模型因素，不能直接等同于动态性。

坐标文件是 fitted model，不是原始实验数据。评价时还应看 geometry outlier、clashscore、Ramachandran outlier、ligand density fit 和 validation report。

结构质量必须区分 precision、accuracy 与 conformational heterogeneity。X-ray 的 occupancy、B-factor 和 R-free，cryo-EM 的局部分辨率与 map–model fit，NMR 的 restraint violation 约束的是不同误差来源。PDB biological assembly 也可能来自作者注释或计算推断，不必然等于溶液中的真实寡聚状态。

<br>

# 结构叠合

## 概念：结构叠合

- **研究对象**
  - 结构叠合把两个三维对象放到同一坐标系中比较；
  - 对应关系已知时，需要寻找旋转 $R$ 和平移 $t$，使对应点距离平方和最小；
  - Kabsch 给出该刚体最小二乘问题的闭式解；
  - 平移和旋转是观察坐标系的差异，不应计入蛋白内部结构差异。
- **输入、输出与评价**
  - 输入是两组 $C_\alpha$ 或原子坐标以及 residue correspondence；
  - 输出是 $R,t$、叠合后的坐标和 RMSD；
  - RMSD 必须与 aligned length、coverage 和 atom selection 一起解释；
  - 只叠合三个局部点可以得到很低 RMSD，却不能说明两条完整蛋白相似。
- **已知对应与未知对应**
  - correspondence 未知时，结构比对还要决定哪些残基配对；
  - STRUCTAL、DALI、TM-align 等方法在几何对应和刚体叠合之间搜索或迭代；
  - Kabsch 是结构比对中的一个子问题，不是完整的未知对应结构搜索算法。
- **章节衔接**
  - 本章使用蛋白质结构章定义的坐标与距离表示，并把序列比对中的“对应关系”推广到三维；
  - 叠合与 residue mapping 将用于结构 motif 搜索、模板建模和 MD 轨迹对齐。

## Kabsch

给定已知对应的两组点 $P,Q\in\mathbb{R}^{n\times3}$，本文始终采用“每一行是一个点”的 row-vector convention：

$$
\min_{R,t}\frac1n\left\|PR+\mathbf1t-Q\right\|_F^2,
\qquad R^\top R=I,\quad\det R=1
$$

设质心为 $\bar p,\bar q$，中心化坐标为 $X=P-\bar p$、$Y=Q-\bar q$，计算

$$
H=X^\top Y=U\Sigma V^\top
$$

再令

$$
D=\operatorname{diag}\left(1,1,\det(UV^\top)\right)
$$

最优旋转和平移为

$$
R=UDV^\top,\qquad t=\bar q-\bar pR
$$

因为

$$
\|XR-Y\|_F^2
=\|X\|_F^2+\|Y\|_F^2-2\operatorname{tr}(R^\top H)
$$

所以该解最大化 $\operatorname{tr}(R^\top H)$。若使用 column vector，并写 $\mathbf q=R_{\mathrm{col}}\mathbf p+t$，则 $R_{\mathrm{col}}=R^\top=VDU^\top$。若省略 determinant 修正，SVD 可能返回镜像；镜像能降低 RMSD，却不是合法 rotation。

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

固定三维时，中心化与构建协方差为 $O(n)$，$3\times3$ SVD 为常数时间。一般 $d$ 维时为 $O(nd^2+d^3)$；保存坐标需 $O(nd)$ 空间，流式累计质心与协方差可降低额外空间。

代码采用 row-vector convention：`P @ R + t`。许多教材使用 column vector 并写成 $R\mathbf p+t$，此时矩阵公式转置。实现中最常见错误正是混用这两套约定。测试应构造已知 rotation/translation，确认恢复坐标而不只检查某个真实样本 RMSD。

Arun、Huang 与 Blostein 1987 年的推导同样通过 $3\times3$ cross-covariance 的 SVD 求最小二乘 rotation 和 translation。其问题写成带噪声的 $\mathbf p_i'=R\mathbf p_i+t+\mathbf n_i$，采用 column-vector convention；与上式比较时需要转置旋转矩阵。Kabsch 与 Arun 算法在无权、已知对应的刚体最小二乘问题上给出同类闭式解，文献中的矩阵排列和 determinant correction 写法可能不同。

若每个点有置信度 $w_i$，可用 weighted centroid 和 weighted covariance。存在 outlier 时，普通 least squares 会被少量大偏差主导，可迭代剔除或采用 robust loss，但应避免只保留“看起来最像”的 core 后仍报告全结构结论。

当点集共线、共面或包含重复点时，最优 RMSD 仍可定义，但 rotation 可能不唯一。生产代码还应检查输入 shape、有限值、点数、权重非负性与退化 singular value。

## 对应未知

序列比对已知时，可按对应残基叠合；结构比对通常连对应关系也未知。典型方法在“建立对应”和“刚体叠合”之间迭代，或比较局部距离模式。DALI 比较 residue–residue distance matrix，TM-align 优化对蛋白长度更稳健的 TM-score。

结构比对同时优化两个相互依赖的问题：给定 correspondence 时可闭式求最小 RMSD；给定空间取向时可按距离相似度寻找 correspondence。若只追求最小 RMSD，算法会倾向保留很短的局部 core；若只追求 aligned length，又会纳入大量低质量对应。因此必须联合报告 RMSD 与 correspondence/coverage，或使用同时奖励长度、惩罚距离和 gap 的目标。

STRUCTAL 交替执行这两个步骤：在当前相对取向下，根据 $C_\alpha$ 间距离构造 similarity matrix，并用保持序列顺序的动态规划求 correspondence；再对当前对应做最小二乘刚体叠合，更新取向，直至对应不再改变。每个子问题有明确解，不代表交替过程一定找到全局最优，结果会受初始取向和 score 设计影响。它把 Needleman–Wunsch 式 DP 与 Kabsch/Procrustes 叠合连接在同一迭代框架中。

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

这段代码展示交替优化，不是 STRUCTAL、DALI 或 TM-align 的复刻。其 distance score、global end-gap 和单一起点都很简化；真实结构比对会使用饱和距离权重、局部片段初始化、coverage normalization 和多起点搜索。若初始取向很差，第一次 DP 可形成错误对应，之后 Kabsch 会在错误 basin 中收敛。

DALI 把结构转为 intramolecular distance matrix，寻找具有相似距离 pattern 的 submatrix。距离模式对整体刚体变换不变，也允许在没有初始序列对应的情况下比较内部几何。原始方法先构造局部相似片段，再用 Monte Carlo 优化组合片段的 alignment；数据库搜索报告的 Z-score 依据背景分数分布衡量显著性，不等同于 RMSD。Holm 与 Sander 1996 年的 all-against-all 分析进一步用 DALI 组织已知结构空间，说明序列相似度很低时仍可发现共享 fold。

对给定对应关系，TM-score 在刚体叠合上取最大值：

$$
\operatorname{TM\text{-}score}
=\max_{\text{superposition}}
\frac{1}{L_{\text{norm}}}
\sum_{i=1}^{L_{\text{ali}}}
\frac{1}{1+[d_i/d_0(L_{\text{norm}})]^2}
$$

常用 $d_0(L)=1.24\sqrt[3]{L-15}-1.8\ \mathrm{\mathring A}$，短链实现还会设置下限。TM-align 交替更新对应关系和叠合，以启发式方式寻找高分解，不保证全局最优。归一化到某一条链时结果可非对称；按两条链分别归一化或使用平均长度会得到不同数值。远离 core 的大距离项贡献会饱和，因此 TM-score 比 RMSD 不易被少数 outlier 支配。

RMSD 对局部 outlier 敏感，并随蛋白长度变化。对典型单结构域蛋白，TM-score $>0.5$ 常作为相同 SCOP/CATH fold 的经验信号，随机无关结构平均水平约为 0.17；它不是适用于短链、多结构域和重复蛋白的硬定理。

结构相似不自动证明 sequence homology；物理约束可导致 convergent fold。反过来，明确同源的蛋白也可能因 domain motion 产生较高 global RMSD。可分别报告 domain-level alignment 和 hinge motion。

结构比较还必须报告 atom selection、residue correspondence、aligned coverage、归一化长度和是否允许 domain-wise alignment。低 RMSD 可能只来自很短的局部 core，高 TM-score 也不能单独证明演化同源。

<br>

# 结构模式

## 概念：结构模式

- **研究对象**
  - 结构模式研究“局部空间安排是否反复出现”；
  - 一维 motif 要求关键残基按序列顺序出现；
  - 三维 motif 要求若干残基或原子满足距离、角度、手性和理化环境约束；
  - 催化残基可在一级序列上相距很远，却在折叠后靠近，因此三维搜索能发现线性 consensus 看不到的功能位点。
- **输入、输出与方法**
  - 输入是几何模板或功能位点模型，以及待扫描的结构或结构库；
  - 输出是候选残基 assignment、几何误差和统计显著性；
  - TESS/FFF 更接近显式模板匹配；
  - FEATURE 把查询点周围的微环境转成多维属性向量。
- **章节衔接与解释边界**
  - 结构模式承接 Motif 和结构叠合；
  - 搜索阶段解决“哪里像模板”；
  - 统计阶段解决“扫描许多位置后，最好分数是否仍异常”；
  - 命中只表示局部结构与已知位点相容，不等于功能已经得到实验确认；
  - 只报告最低 RMSD 而没有背景分布，无法区分真正保守位点和数据库规模带来的偶然近似。

## 1D 与 3D

1D sequence motif 主要约束残基在线性序列中的顺序，可连续，也可含 wildcard、gap 或可变间隔。3D motif 约束关键原子或残基的空间关系，例如 catalytic triad 的成员可在序列上相距很远。

三维模式搜索需要：

- 原子或残基类型标签；
- 点间距离及容差；
- 可选角度、手性和溶剂可及性；
- 对旋转和平移不变；
- 对缺失原子和构象噪声鲁棒。

可以把 motif 表示成小型标记图，把候选结构表示成距离图，再进行 subgraph matching。一般子图同构判定是 NP-complete，带标签、距离区间与缺失点的近似匹配还会扩大搜索空间；不过生物 motif 通常很小。实际系统先用距离签名或几何哈希生成候选，再做精确叠合验证。

geometric hashing 通常由三个非共线且有确定顺序的点建立局部 frame；交换基点顺序可能改变 frame 手性。其余点转为 frame-relative coordinate 并写入 hash table，候选结构枚举兼容 frame 后投票，再以原子类型、手性、RMSD 与完整几何约束验证。

课程中出现的三类早期 3D functional-site 方法使用不同表示：

- **FFF（Fuzzy Functional Form）**：从实验知识和已知结构手工或半手工定义关键残基身份、构象与几何约束，并允许一定容差，使较低分辨率模型也能被筛查；
- **TESS**：从已知 active site 推导三维坐标模板，使用 geometric hashing 在结构库中寻找满足模板的候选；
- **FEATURE**：不要求每个关键原子一一对应，而把查询点周围的同心壳层表示为理化与结构属性向量，学习某类微环境相对背景的统计偏好。

FFF/TESS 更接近显式几何 template，FEATURE 更接近 local-environment classifier。三者都依赖正例定义和背景库；命中表示局部环境与已知功能位点相容，不等于蛋白已被实验确认具有该功能。

下面以 distance template 表示 3D motif。模板只指定每个位置允许的残基类型及位置间目标距离，因此天然不受整体旋转和平移影响。若功能依赖手性或原子方向，还需增加 signed volume、角度或局部 frame 约束。

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

`motif_types` 的每一项可以是集合，例如 `{"D", "E"}` 表示该位置允许酸性残基。枚举复杂度是各位置候选数的乘积；geometric hashing 的作用正是用局部 frame 和量化坐标先索引兼容组合，避免对整库执行完整笛卡尔积。

功能 motif 搜索应区分：

- residue identity 是否必须完全相同；
- 使用 $C_\alpha$、side-chain centroid 还是具体 functional atom；
- distance tolerance 是否考虑结构分辨率；
- 是否允许某个位置缺失；
- 是否要求相同 chirality；
- 命中是否位于可接近 pocket。

## 统计

扫描许多结构和位置会产生 selection bias。结构相似是否显著，取决于背景数据库、蛋白长度、残基组成和搜索空间，不能只报告最低 RMSD。

motif 命中评价应使用独立背景集或 decoy database 估计 false-positive rate。大量 motif、蛋白和局部 frame 的扫描会产生 multiple testing 与数据库冗余，因此需要同时报告覆盖、经验 $p$ 值、FDR 和最佳几何分数。

## 统计特征检测

课程中的 statistical feature detection 不是某一种固定检验，而是共同问题：先把序列窗口、三维微环境或网络局部结构映射为 feature score，再问该分数相对背景是否异常。PWM log-odds 用背景碱基分布作 null；BLAST E-value 使用随机序列极值分布；DALI Z-score 使用结构比较背景；FEATURE 则比较功能位点与非位点微环境的属性分布。

三维 FEATURE 类表示可把查询点周围空间分成同心 shell，在每个 shell 中统计 atom type、charge、hydrophobicity、secondary structure 与 solvent accessibility。若 $x_j$ 是某项属性，最简单的独立 Gaussian log-likelihood ratio 为

$$
\log\Lambda(x)
=\sum_j\log
\frac{\mathcal N(x_j;\mu_{1j},\sigma_{1j}^2)}
     {\mathcal N(x_j;\mu_{0j},\sigma_{0j}^2)}
$$

其中 1 表示功能位点，0 表示背景微环境。各 feature 实际高度相关，独立模型的分数不能直接解释为 posterior probability。Null set 还必须匹配蛋白类别、溶剂暴露、残基组成和结构质量，否则分类器只会识别数据来源。

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

经验 $p$ 值只要求背景抽样过程合理，Z-score 还隐含均值和方差足以概括 null。若背景明显偏态或存在极值尾部，应直接报告经验分位数，或为最大扫描分数拟合合适的极值模型。对同一结构扫描许多位置时，null replicate 必须重复整个扫描过程，不能把单个位置分数当作最大值的背景。

<br>

# 结构预测

## 概念：结构预测

- **研究对象与解释边界**
  - 结构预测从氨基酸序列推断蛋白质可能采用的三维构象；
  - 输出可以是坐标，也可以是二级结构、contact map、distogram 或 torsion distribution 等 intermediate representation；
  - 预测模型给出的是在训练数据、模板和输入 MSA 条件下最受支持的结构；
  - 它不是实验观测，也不是完整热力学 ensemble。
- **输入、输出与方法分支**
  - 输入通常包括目标序列、同源序列 MSA 和可用模板；
  - 输出包括原子坐标及 pLDDT、PAE、pTM 等置信度；
  - Homology modeling 依赖相似模板；
  - threading 判断序列是否适合已知 fold；
  - Rosetta 通过片段与能量搜索构象；
  - AlphaFold 学习序列、MSA、pair representation 与坐标之间的关系。
- **章节衔接**
  - 本章整合数据库搜索、MSA、结构叠合和距离表示；
  - 蛋白质语言模型讨论不显式构建 MSA 时怎样从大规模序列学习统计规律；
  - 分子力学从给定坐标出发计算力和时间演化，回答的是不同问题。

## 经典路线

- **CASP 的设计**
  - Critical Assessment of protein Structure Prediction（CASP）以尚未公开实验结构的 target 进行盲测；
  - 方法在提交截止后才与真实结构比较；
  - 盲测把“模型能否在已知结构上拟合”与“能否预测未来 target”分开。
- **数据规律与物理约束**
  - MSA、模板和训练结构提供数据规律；
  - 几何、packing 与能量限制可实现构象；
  - 结构预测因此同时是统计学习问题与物理约束问题。
- **评价边界**
  - 评价从单纯 RMSD 扩展到 GDT、TM-score、局部质量和不同 target category；
  - CASP 是周期性 benchmark，不是某个预测算法；
  - 排名还依赖 target 难度、domain 划分和评价指标。

- **二级结构预测**：从局部窗口到神经网络，输出 helix/sheet/coil。
- **同源建模**：找模板、对齐、复制骨架、建 loop/side chain、优化与验证。
- **threading/fold recognition**：把序列穿到已知 fold 上，用环境特异势评价。
- **ab initio**：在巨大构象空间中结合能量函数与搜索。

同源建模误差可来自模板选择、target–template alignment、插入缺失区、side-chain packing、构象状态和后续优化；哪一项占主导取决于模板相似度与覆盖范围，不存在固定的误差大小顺序。模板相似度高也不能自动保证活性状态、复合体界面或无序区正确。

约 30% target–template sequence identity 常被当作同源建模从相对可靠区进入 twilight zone 的经验界线，但它不是硬阈值。Alignment length、结构域架构、profile significance、保守 motif 和模板质量会共同改变可信度：短片段 35% identity 可能没有意义，覆盖完整结构域的 profile hit 即使 identity 更低也可能可靠。

同源建模的完整流程为：

1. 用 profile search 找模板；
2. 检查模板覆盖、resolution、ligand、oligomeric state；
3. 构建 target–template alignment；
4. 从模板复制 conserved backbone；
5. 对 insertion/deletion 区域做 loop modeling；
6. 安放 side chain rotamer；
7. 局部优化并做 stereochemical validation；
8. 用独立证据评价，而不是只看建模 objective。

threading 在模板 sequence identity 很低时，把 target residue 放入模板的结构环境。评分可包含 burial、secondary structure compatibility、pairwise contact 与 gap。若 pairwise contact score 依赖相距很远的 alignment position，标准一维 DP 不再严格可分，常需近似或迭代。

## Rosetta

- **原始 fragment assembly**
  - Simons 等 1997 年的方法把短片段装配、simulated annealing 与 Bayesian scoring 结合；
  - 片段库由与目标序列局部窗口相容的已知结构片段组成；
  - Monte Carlo 搜索反复替换局部 backbone torsion，在构象空间中寻找低能区域。
- **后续两阶段流程**
  - Rohl 等 2004 年总结的低分辨率阶段使用残基级统计势和紧致性、疏水作用、二级结构配对等项；
  - 高分辨率阶段加入全原子 packing、氢键、范德华作用与 side-chain rotamer 优化。

按课程串讲展开，流程从 target sequence 的 BLAST/profile search 与 PSSM 开始，在 PDB 中为每个重叠的 9-residue window 选择序列 profile 与局部二级结构相容的 fragment。一次 Monte Carlo move 以候选 9-mer 的 backbone torsion 替换目标窗口；Metropolis 准则以

$$
P_{\mathrm{accept}}=\min\left(1,e^{-\Delta E/(k_BT)}\right)
$$

接受降能 move，并以温度相关概率接受升能 move，从而跳出局部 basin。Simulated annealing 逐步降低有效温度。大量独立 trajectory 得到 decoy ensemble 后，以 score 与结构相似性聚类；大而低能的 cluster 及其中心结构通常比孤立的最低分 decoy 更可信。9-mer 是课程强调的经典片段长度，不表示所有 Rosetta protocol 只使用 9-mer。

方法依赖两项近似。第一，短序列片段的局部构象可由已知结构库覆盖；第二，所用能量函数的低值区域与 native-like basin 对应。搜索不足会停在错误 basin，能量函数误差则会把非天然构象排在前面。多个独立 trajectory 形成的低能结构应出现 convergence；单条最低能 decoy 缺少这种支持。Rosetta score 由经验与物理启发项加权构成，不应直接解释为实验折叠自由能。

## 共进化

多序列比对中的成对相关可能反映三维接触，但需要去除共同祖先和间接相关。Direct Coupling Analysis 用全局 Potts model 分离直接耦合：

$$
P(x)\propto\exp\left(\sum_i h_i(x_i)+\sum_{i<j}J_{ij}(x_i,x_j)\right)
$$

有效序列数和 MSA 多样性决定信号质量。

观察到的 covariance 可能来自共同祖先，也可能是两个位点都与第三个位点相关。DCA 拟合全局模型，估计在模型条件下不能由其他位点简单解释的统计耦合；这种 direct coupling 不等于直接物理接触，也不能彻底消除系统发育、MSA 错配和共同功能选择。Potts 参数存在 gauge redundancy，接触排序常转到 zero-sum gauge，再计算 coupling matrix 的 Frobenius norm 并施加 average-product correction。

参数量约为 $O(L^2q^2)$。稠密 mean-field inverse-covariance 的直接矩阵求逆可达 $O((Lq)^3)$；pseudo-likelihood 每次目标或梯度评估约为 $O(M_{\mathrm{eff}}L^2q)$，还要乘优化迭代数。

contact prediction 可把高 coupling pair 作为 folding restraint。序列数据库增长后，更深的 MSA 能提供更多共同演化约束，即使新增序列本身没有结构标签。

## AlphaFold

第一代 AlphaFold 在 CASP13 中从 MSA 和序列特征预测 residue-pair distance distribution 与 backbone torsion distribution。它把 distogram 的负对数概率拟合为可微、protein-specific potential，并加入 torsion potential；结构以 $\phi,\psi$ 参数化，再用 L-BFGS 等方法从多个初始化最小化总势。这里的“能量”主要来自网络预测分布，不是经典 force field 的实验自由能。

AlphaFold2 在 CASP14 中改变了问题分解方式：MSA representation、pair representation、模板信息与 structure module 端到端联合训练，并通过 recycling 反复细化。AF1 是“预测距离/角度后优化势能”，AF2 则在网络内部迭代更新表示与坐标；二者不能只概括成同一 CNN 的版本升级。AlphaFold2 显著提高单体结构精度，但输出仍需结合：

- **pLDDT**：局部残基置信度；
- **PAE**：残基对相对位置误差；
- 多聚体界面、配体、构象状态与无序区；
- 训练数据时间切分和模板泄漏。

- **Evoformer**
  - 在 MSA representation 与 residue-pair representation 之间交换信息；
  - triangle multiplicative update/attention 沿残基三角形传播 pair information；
  - 这种更新帮助网络学习几何一致性，但不在每一步严格保证 pair representation 是合法 Euclidean distance matrix。
- **Structure module 与 recycling**
  - structure module 使用 invariant point attention；
  - 标量输出对整体刚体变换保持不变；
  - 预测坐标随整体刚体变换等变；
  - recycling 将预测坐标与内部表示重新输入网络。

pLDDT 是模型对每个残基 lDDT-$C_\alpha$ 分数的预测期望，范围为 0–100；它评价局部距离模式，不依赖全局叠合。PAE$(i,j)$ 以 Å 为单位，表示“若以 residue $j$ 的主链 frame 对齐预测与真实结构，residue $i$ 的位置误差”的预测期望。PAE 一般不对称。两个 domain 可各自具有高 pLDDT，却因相对 orientation 不确定而具有高 inter-domain PAE。

多聚体预测还需查看 interface predicted TM-score、界面 PAE、物种配对 MSA 与 biological plausibility。pTM 汇总全局拓扑置信度，AlphaFold-Multimer 的 ipTM 更强调链间相对排布；二者都是模型置信度，不是实验结合证据。模型可能把非生理接触预测成界面，也可能漏掉 ligand、membrane 或 PTM 才稳定的状态。

预测结构不是实验观测，也不等于完整动力学 ensemble。高 pLDDT 不能证明蛋白功能或特定相互作用；低 pLDDT 可来自无序、柔性、缺少结合伙伴或预测失败，不能仅凭低分宣布 intrinsically disordered。还应检查不同随机种子、MSA/template 设置和模型间的一致性。

> [!EXAMPLE]+ 例6：置信度的两种尺度
>
> 某双结构域蛋白的两个 domain 内 pLDDT 都为 90，但 domain 间 PAE 为 25 Å。合理结论是“两个局部 fold 较可信，二者相对朝向不确定”，而不是“完整蛋白结构高置信”。

<br>

# 蛋白质语言模型

## 概念：蛋白质语言模型

- **研究对象**
  - 蛋白质语言模型把氨基酸序列当作 token 序列，在大量未标注蛋白上学习哪些残基组合在自然序列中常见；
  - 它不预先写入替换矩阵、motif 或 family profile；
  - 模型通过 masked-token prediction 或 autoregressive prediction 学习上下文条件分布。
- **输入、输出与用途**
  - 输入是一条或多条蛋白质序列；
  - 输出可包括每个残基的 embedding、整条序列的 embedding、条件概率或生成序列；
  - Embedding 可作为功能分类、定位、结构预测和变异效应模型的特征；
  - mutation log-odds 衡量某替换是否符合模型学到的序列分布，但不直接等于折叠自由能或临床致病概率。
- **与 Profile HMM 的关系**
  - 两者都是序列分布模型；
  - Profile HMM 针对一个家族，状态语义明确且参数较少；
  - 语言模型跨大量家族学习高维上下文，表达能力更强，也更依赖训练数据和严格的数据划分；
  - 解释任何分数前都要固定模型、tokenizer、序列处理和 pooling 方法。

## 训练目标

蛋白质语言模型把氨基酸当 token。对随机 mask 集合 $M$，masked language model 最小化 masked-token cross-entropy：

$$
\mathcal L_{\mathrm{MLM}}
=-\mathbb E_M\sum_{i\in M}
\log P_\theta(x_i\mid\widetilde x_M)
$$

其中 $\widetilde x_M$ 是遮盖或扰动后的输入。MLM objective 不是规范化的整序列 likelihood。embedding 可用于远缘同源、功能、变异效应、结构和生成。模型从大规模自然序列中吸收进化统计，不需要为每个任务重新手工设计特征。

masked language model 随机遮盖部分 residue 并预测原字符；autoregressive model 按给定顺序分解

$$
P(x)=\prod_{i=1}^{L}P(x_i\mid x_{<i})
$$

前者可双向利用上下文，后者天然给出序列 likelihood 并便于生成。二者学到的是训练语料分布与 objective 共同决定的表示，不是显式物理能量。

对长度 $L$、hidden dimension $d$ 的标准 Transformer layer，attention 主要计算量为 $O(L^2d)$，attention matrix 显存为 $O(L^2)$，完整 layer 还包含 $O(Ld^2)$ 的 projection/MLP 成本。只有把 $d$ 当常数时才简称 $O(L^2)$。长蛋白、multi-chain 或大量序列输入会成为瓶颈。

## 使用

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

不能通用地假设 `1:-1` 恰好去除一个 BOS 和一个 EOS；special-token mask 与 attention mask 才能同时排除特殊 token 和 padding。mean pooling 会丢失位点信息；长序列截断、同源泄漏和物种偏差也会显著影响结果。

常见使用层次包括：

- residue embedding：secondary structure、binding site、variant effect；
- sequence embedding：family、localization、function；
- pair representation/attention：contact 或结构；
- conditional generation：按 family、structure 或 function 设计序列。

linear probe 可测试 frozen representation 中是否已有可线性读出的信息；full fine-tuning 容量更高，但更易过拟合。比较方法时应保持下游 split 和 label budget 一致。

## 解释边界

语言模型的高 likelihood 表示序列符合训练分布，不直接等于稳定、可表达或有目标功能。zero-shot mutation score 常用突变前后 log-likelihood 差，但应以 DMS 数据校准。

若 masked model 每次只遮一个位置，pseudo-log-likelihood 可写为

$$
\operatorname{PLL}(x)=\sum_i\log P(x_i\mid x_{\setminus i})
$$

精确 PLL 通常需要 $L$ 次 forward pass。单点突变更常使用 mutation-site masked marginal：

$$
\Delta s=
\log P_\theta(x_i^{\mathrm{mut}}\mid x_{\setminus i})
-\log P_\theta(x_i^{\mathrm{wt}}\mid x_{\setminus i})
$$

它不是完整 mutant PLL 与 wild-type PLL 之差。多突变时必须说明 simultaneous masking、逐位相加还是重新计算完整 PLL。不同 protein、长度或模型间的绝对分数不可直接混用。

模型评估必须控制同源泄漏。近缘序列、重复数据库记录和相同家族跨 split 会使随机拆分结果严重乐观；应报告 identity threshold、时间切分和按 family 汇总的结果。attention weight 也不自动等于接触概率或因果解释。

<br>

# 分子力学

## 概念：分子力学

- **研究对象与近似**
  - 分子力学把蛋白质、配体、水和离子表示成一组带类型、质量、电荷与化学键的原子；
  - 参数化势能函数 $E(\mathbf r)$ 近似给定坐标 $\mathbf r$ 的能量；
  - 它不显式求解电子波函数，而是假设成键方式固定，再用经验和量子化学拟合的参数描述键长、键角、二面角、范德华和静电作用。

$$
\mathbf F_i=-\nabla_iE(\mathbf r)
$$

- **力、最小化与动力学**
  - 势能函数首先解决怎样比较相邻构象，并用上式计算每个原子的力；
  - 能量最小化沿力的方向消除 clash，寻找起始构象附近的局部极小值；
  - 分子动力学把力代入 Newton 方程，生成位置和速度随时间变化的轨迹；
  - 输入不仅是 PDB 坐标，还包括 atom type、bond topology、protonation、force-field 参数、水模型、边界条件和温度压力设置；
  - 输出是势能、力、优化坐标或轨迹。
- **解释边界**
  - 势能 $E(\mathbf r)$ 不是实验折叠自由能 $\Delta G$；
  - 前者评价一个微观构象，后者比较包含大量构象和溶剂状态的宏观 ensemble，同时含焓与熵；
  - 一次最小化不能预测真实折叠路径；
  - MD 能否估计平衡性质还取决于力场、ensemble 和采样是否充分。
- **章节衔接**
  - 本章从实验或预测结构提供的初始坐标出发；
  - 结构预测中的 learned potential 用数据约束生成坐标；
  - 经典 force field 为给定原子体系计算能量和力；
  - MD 轨迹随后进入三维统计章节做对齐、降维、聚类和状态建模。

## 势能函数

热力学稳定性由自由能而非单一势能决定：

$$
\Delta G=\Delta H-T\Delta S
$$

$\Delta H$ 汇总相互作用能与溶剂等焓贡献，$\Delta S$ 包含蛋白、溶剂和离子的构象与平移转动熵。某个构象的 force-field potential energy 较低，不足以证明其宏观态自由能最低；后者还取决于该状态占据的相空间体积。

经典 force field 把势能分为 bonded 与 non-bonded 项：

$$
E=E_{\text{bond}}+E_{\text{angle}}+E_{\text{dihedral}}
+E_{\text{vdW}}+E_{\text{electrostatic}}
$$

常见形式包括谐振子键长/键角、周期 torsion、Lennard–Jones 势和 Coulomb 势：

$$
E_{\mathrm{LJ}}(r)=4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12}
-\left(\frac{\sigma}{r}\right)^6\right]
$$

force field 是参数化近似，不显式求电子波函数，通常不能自然处理成键/断键。

课程所称 Levitt 势可按同一框架理解为 bond、angle、dihedral、Coulomb 与 van der Waals 项的加和。bond/angle 维持局部几何，dihedral 控制旋转偏好，Coulomb 表示带电或部分带电原子的静电作用，van der Waals 项包含短程排斥与中程吸引，并强烈约束原子 packing。“vdW 主导 packing”描述其对近接几何的直接作用，不表示总折叠自由能在所有体系中都由 vdW 单独主导。

一个示意性的 additive fixed-charge force field 可写为

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

AMBER、CHARMM、OPLS 等 force-field family 的 atom type、partial charge 和参数是成套拟合的，不应随意混合。water model 也是参数体系的一部分。protonation state、tautomer 和 metal coordination 设错时，长模拟不能自动修正起点化学错误。

该式仍省略 improper、Urey–Bradley、CMAP、约束、polarization 与特殊 pair rule。具体 force field 的 $1/2$ 系数、torsion convention、combination rule、1–2/1–3 exclusion 和 1–4 scaling 各不相同。非键求和也并非简单覆盖所有 $i<j$；显式溶剂固定电荷模型通常取 $\epsilon_r=1$，由水与离子显式产生屏蔽。

## 能量最小化

目标是找局部极小值 $\nabla E(\mathbf{x})=0$。steepest descent 稳健但后期慢；conjugate gradient、L-BFGS 利用更多曲率信息。最小化只能消除坏接触并进入附近 basin，不给出热平衡分布，也不能证明全局最低能量。

steepest descent 更新 $\mathbf x_{k+1}=\mathbf x_k-\eta_k\nabla E(\mathbf x_k)$。步长过大会震荡，过小则慢。L-BFGS 不显式保存完整 Hessian，而用有限历史的梯度和位置差近似 inverse Hessian，适合高维坐标。

收敛标准可看最大 force、RMS force 或能量变化。仅能量变化小可能只是步长太小，不代表 gradient 足够接近 0。

## 分子动力学

MD 数值积分 Newton 方程：

$$
m_i\frac{d^2\mathbf{r}_i}{dt^2}=-\nabla_iE(\mathbf{r})
$$

Velocity Verlet 以小时间步推进位置和速度。含氢共价键的高频伸缩振动限制 all-atom MD 的时间步：未约束时常用约 1 fs，约束 X–H 键后常用约 2 fs，hydrogen-mass repartitioning 等方案可使用约 4 fs，但会改变部分动力学时间尺度。约束算法、thermostat、barostat、周期边界和长程静电处理共同决定 ensemble。

Velocity Verlet 的核心为

$$
\mathbf r(t+\Delta t)=\mathbf r(t)+\mathbf v(t)\Delta t
+\frac{\mathbf a(t)}{2}\Delta t^2
$$

计算新位置上的 force 后，

$$
\mathbf v(t+\Delta t)=\mathbf v(t)
+\frac{\mathbf a(t)+\mathbf a(t+\Delta t)}{2}\Delta t
$$

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

每步最昂贵部分通常是 non-bonded force。直接全对计算为 $O(N^2)$；cutoff、neighbor list 将短程部分降至近线性，Particle Mesh Ewald 以网格 FFT 处理长程静电，典型为 $O(N\log N)$。

上述两式只表示无约束、确定性的基本积分器；SHAKE/RATTLE、Langevin thermostat 或 Trotter-split thermostat/barostat 还包含约束投影、随机或耦合子步。NVE ensemble 固定 $N,V,E$，但有限时间步数值轨迹通常只近似守恒能量，并可能更准确地守恒 shadow Hamiltonian。NVT/NPT 固定的是 ensemble 参数及其分布，不是让瞬时温度或压力每一步都等于目标值。

典型流程为 prepare structure、protonation、solvation、ions、minimization、equilibration、production、trajectory analysis。RMSD、RMSF、radius of gyration 和 contact occupancy 都只是轨迹摘要。

RMSF 对原子 $i$ 可写为

$$
\operatorname{RMSF}_i=
\sqrt{\left\langle\|\mathbf r_i(t)-\langle\mathbf r_i\rangle_t\|^2\right\rangle_t}
$$

计算前通常要把每帧对齐到共同 reference，去除整体 translation/rotation。若蛋白存在真实 domain motion，对全蛋白对齐与对单 domain 对齐会产生不同 RMSF 解释。

自由能与状态概率满足 $\Delta G_{A\rightarrow B}=-k_BT\ln(P_B/P_A)$。MD 的统计目标是从目标 ensemble 估计状态概率与期望。以时间平均近似 ensemble average 需要遍历性和充分采样。

> [!WARNING]+ 采样瓶颈
>
> 轨迹很长不等于独立样本很多。相关时间、稀有跃迁和初始构象会限制有效样本量。Replica exchange、metadynamics 等增强采样引入额外假设，并需要正确 reweight。

<br>

> [!EXAMPLE]+ 例7：轨迹帧不是独立样本
>
> 一条 1 μs 轨迹每 1 ps 保存一帧，共有 $10^6$ 帧。若 $\tau_{\mathrm{int}}=\int_0^\infty\rho(t)\,dt=10$ ns，则 $N_{\mathrm{eff}}\approx T/(2\tau_{\mathrm{int}})\approx50$，而不是一百万。若软件把二倍积分纳入其“相关时间”定义，则会报告约 100；必须注明约定。

## 平衡诊断

ensemble 由 Hamiltonian、边界条件、thermostat/barostat、约束与采样权重共同定义。单条长轨迹不能自动证明已达到平衡；应检查 equilibration、多个独立初态、状态往返、observable 的相关时间和 replicate 间一致性。增强采样得到的 biased trajectory 必须用相应权重重建无偏平衡概率。

<br>

<br>

# 三维统计

## 概念：三维统计

- **研究对象与预处理**
  - 三维统计处理一组带几何等价关系的坐标观测，例如多个同源蛋白、NMR conformer 或 MD trajectory frame，而非单个结构文件；
  - 原始坐标会随整体平移和旋转改变；
  - 周期二面角在 $-\pi/\pi$ 处不连续；
  - 同源亚基和对称原子还可能互换；
  - 若不先处理这些等价关系，统计模型会把坐标系差异误当作生物变化。
- **输入、输出与方法**
  - 输入可以是对齐后的坐标矩阵、距离、contact 或 torsion feature；
  - 输出包括平均形状、主成分、慢过程、构象簇和状态转移矩阵；
  - PCA 找最大方差方向；
  - tICA 找时间相关较慢的方向；
  - Markov state model 将连续轨迹离散为状态并估计给定 lag time 下的转移概率。
- **章节衔接与统计边界**
  - 本章承接结构叠合与分子动力学：前者去除刚体自由度，后者产生有时间相关的样本；
  - 轨迹帧通常不独立，不能把百万帧当作百万个生物重复；
  - 有效样本量、trajectory replicate、对称性和距离定义共同决定统计解释。

## 不变量与对齐

三维生物数据通常存在旋转、平移、尺度、对称和缺失。统计前应决定：

- 先对齐坐标还是直接使用距离/角度不变量；
- 对称等价原子如何匹配；
- ensemble 中不同构象如何加权；
- measurement uncertainty 如何传播。

## 形状分析

Procrustes analysis 在去平移、旋转，有时去尺度后比较形状。设原始轨迹矩阵为 $X_{\mathrm{raw}}\in\mathbb R^{N\times d}$，每行是一帧；先去除刚体运动，再按列中心化：

$$
X=X_{\mathrm{raw}}-\mathbf1\bar x^\top,\qquad
C=\frac{1}{N-1}X^\top X
$$

若未中心化，$X^\top X/(N-1)$ 是二阶矩而非样本协方差。显式构造协方差约为 $O(Nd^2)$，特征分解为 $O(d^3)$；$d$ 很大时通常直接对中心化轨迹做 economy/randomized SVD。最大方差方向不一定是最慢动力学过程，也不一定有功能意义。

对中心化特征 $x_t$，tICA 定义

$$
C_0=\mathbb E[x_tx_t^\top],\qquad
C_\tau=\mathbb E[x_tx_{t+\tau}^\top]
$$

并求解 generalized eigenproblem $C_\tau v_k=\lambda_k C_0v_k$。tICA 最大化归一化 time-lagged autocorrelation，名称中的 independent 不保证统计独立。非平衡拼接轨迹还需要 reweighting，否则均值和相关矩阵会偏向初始分布。

Markov state model 把构象聚类为离散状态。采用 row-stochastic convention 时：

$$
T_{ij}(\tau)=P(s_{t+\tau}=j\mid s_t=i),\qquad
\sum_jT_{ij}=1
$$

stationary distribution 满足 $\pi^\top T=\pi^\top$。若 $\lambda_i(\tau)$ 是非平凡特征值，则 implied timescale 为

$$
t_i(\tau)=-\frac{\tau}{\ln|\lambda_i(\tau)|}
$$

lag-time plateau 只是必要诊断；Chapman–Kolmogorov test 还应比较独立估计的 $T(k\tau)$ 与 $T(\tau)^k$。状态离散过粗、采样不足或轨迹非平衡都会破坏 Markov 近似。

## 密度与聚类

结构聚类依赖距离定义：backbone RMSD、TM-score、torsion distance 或 contact distance 会产生不同簇。DBSCAN 能识别非球形簇和噪声，层次聚类便于观察多尺度关系。聚类数与阈值需要稳定性分析，不应只看一张图。

三维数据还经常具有 nested sampling：同一患者多个细胞、同一蛋白多帧、同一结构多条链。把所有 observation 当独立样本会低估方差。bootstrap 应在真正独立的最高层级重采样，例如 trajectory replicate，而不是逐帧。

二面角是周期变量，直接做 Euclidean PCA 会在 $-\pi/\pi$ 边界制造假距离，常改用 $(\sin\theta,\cos\theta)$。具有对称侧链、同源亚基或可交换 ligand 时，还要先最小化 symmetry-equivalent permutation 下的距离，否则同一构象可能被错误分成多个簇。

<br>

# 知识表示

## 概念：知识表示

- **研究对象**
  - 知识表示研究怎样让基因、蛋白质、疾病和功能术语在不同数据库中保持一致含义；
  - 自由文本中的同义词、缩写和上下位概念不能仅靠字符串相等处理；
  - 本体用稳定标识符、定义和关系组织概念；
  - annotation 把具体 gene product 与本体 term 连接，并记录证据来源。
- **三类计算任务**
  - **语义规范化与推理**：例如沿 Gene Ontology 的允许关系传播注释；
  - **富集分析**：输入候选基因集、背景全集和本体版本，输出 term 命中数、fold enrichment、$p$ 值与 FDR；
  - **数据整合**：处理 gene、transcript、protein 和 phenotype 之间的一对多映射与冲突证据。
- **解释边界与章节衔接**
  - Term 不是 annotation，pathway 也不是普通 gene set；
  - 把 pathway 降成无序成员集合后可以做富集，却丢失方向、反应和拓扑；
  - 本章为转录组结果解释、HPO 表型匹配、网络模块注释和跨数据库 join 提供语义基础。

## 受控术语

自由文本存在同义词、缩写和粒度不一致。本体用唯一标识符、定义和关系规范语义。Gene Ontology 包含：

- Molecular Function；
- Biological Process；
- Cellular Component。

GO 是有向无环图而非简单树，一个 term 可有多个 parent。`is_a`、`part_of` 等关系语义不同。基因注释还带 evidence code，用于记录实验、系统发育、计算推断、作者陈述或自动注释等证据来源。evidence code 表示证据类型，不是通用于所有场景的统一质量等级。

本体由 class、relation、axiom 与 identifier 构成。DAG 中具体 term 可能同时继承多个上位概念。ontology term 与数据库 annotation 必须区分：前者定义概念，后者断言某 gene product 与概念之间存在带证据的关系。

Stevens、Goble 与 Bechhofer 沿用 Gruber 的定义，把 ontology 解释为概念化的显式规范，并结合生物信息学说明其用途。受控词表主要统一名称，taxonomy 增加层级，本体还可给出关系约束和可供推理机使用的 axiom。建模粒度取决于用途：用于检索的轻量本体未必足以支持自动一致性检查，用于严格推理的表示则需要声明 domain、range、互斥和等价关系。

annotation 还可能带 qualifier、isoform、reference 和 annotation extension。`NOT` 表示有证据明确否定某项 GO 关联，不是“没有显著结果”或“缺少注释”。正注释通常沿 GO 允许的关系向较一般 ancestor 传播；`NOT` 则约束该 term 及更具体 descendant。富集分析应排除 `NOT`，不能把被明确否定的功能当作阳性。

## Pathway 与 gene set

Gene set 是基因标识符的集合，本身不编码顺序、方向或机制。Pathway 通常还包含反应、调控方向、复合物、细胞区室和上下游关系；把 pathway 降成 gene set 后才能使用许多富集算法，但边方向、拓扑与基因间逻辑随之丢失。GO term 对应的 annotation gene set、人工整理 pathway、共表达 module 与实验 signature 都可作为 gene set，其生物语义并不相同。

## 注释传播

若一个基因注释到具体 term，通常也隐含由 GO relation rules 允许的某些 ancestor term；不能把所有图上可达 ancestor 都机械视作自动成立。计算 enrichment 时必须使用与实验可检测基因一致的 background universe，并处理 GO 层级造成的相关检验。

超几何检验：

$$
P(X\ge k)=\sum_{i=k}^{\min(K,n)}
\frac{\binom{K}{i}\binom{N-K}{n-i}}{\binom{N}{n}}
$$

$N$ 是背景基因数，$K$ 是带某 term 的背景基因数，$n$ 是候选集大小，$k$ 是命中数。候选集必须先与 background 取交集并在同一 ID 空间去重；$K,k$ 应按相同 GO 版本、物种、evidence/qualifier 过滤与传播规则计算。

超几何检验的 null 是从固定 background 随机抽取 $n$ 个基因。background 若误用全基因组，而实验平台实际只能检测其中一部分，富集结果会偏。GO term 因共享基因与 ancestor propagation 而相关；BH 在独立或特定正相关条件下仍可控制 FDR，但显著 term 不能解释成彼此独立的过程。若需要任意依赖下的保守保证，可考虑 Benjamini–Yekutieli、置换或层级检验。

fold enrichment 为

$$
\operatorname{FE}=\frac{k/n}{K/N}
$$

小 term 可能有很高 FE 却只包含两三个 gene，因此还要报告 gene count、FDR、term size 和具体命中基因。

## 推理与版本

GO term 会新增、合并、obsolete 或重新定义，因此分析应固定 ontology release、annotation release、物种、evidence-code filter 与 qualifier rule。富集结果至少保存 background、ID 映射前后数量、unmapped/duplicate ID、term size、命中基因、原始 $p$ 值、校正方法和 GO version。

若候选基因由表达阈值或模型筛选产生，简单超几何 null 未必反映上游选择过程。可使用表达量匹配置换、rank-based GSEA 或模拟做敏感性分析。显著 GO term 是当前 gene set 的统计摘要，不等于该过程被因果激活。

## 数据整合

数据库 join 不是普通字符串 join。gene symbol 会变更且可能歧义；同一基因可有多个 transcript/protein isoform；坐标依赖 genome build。可靠整合应保留：

- namespace 与 stable identifier；
- 数据库版本和映射日期；
- one-to-many 映射；
- provenance 与 evidence；
- 缺失机制；
- 冲突规则。

数据整合可分成三个不同问题：

1. **schema alignment**：字段、单位和粒度如何对应；
2. **entity resolution**：哪些记录指向同一实体；
3. **evidence reconciliation**：冲突断言如何保留、评级或汇总。

早期整合把不同模态 feature 拼接后统一建模；晚期整合分别建模再组合 prediction；中间整合学习 shared latent representation。缺失模态常不是随机缺失，例如热门蛋白更可能拥有结构和功能实验，直接删除缺失项会引入 selection bias。

<br>

# 转录分析

## 概念：转录分析

- **研究对象与测量边界**
  - 转录分析研究不同样本中 RNA 丰度如何变化；
  - 数据通常整理为基因或 transcript 乘样本的矩阵；
  - 矩阵元素不直接等于细胞内真实分子数；
  - RT-qPCR、microarray 和 RNA-seq 通过不同实验链获得信号，分别受扩增效率、杂交、测序深度和组成偏差影响。
- **输入、输出与模型**
  - 差异表达输入原始 count matrix、样本 metadata 和 design matrix；
  - 输出是指定 contrast 下的 log fold change、standard error、$p$ 值、adjusted $p$ 值与置信区间；
  - 负二项模型处理 RNA-seq 中均值随方差变化的问题；
  - size factor 处理样本测序尺度；
  - batch、配对和交互项由实验设计编码。
- **章节衔接与解释边界**
  - 本章从表达测量进入统计推断，再把差异基因或全基因排序交给 GO enrichment、GSEA、聚类和网络分析；
  - TPM 适合描述样本内部相对组成，不是 DESeq2 所需的原始 count；
  - 显著性不能替代 effect size 与实验重复。

## RT-PCR 与 qPCR

Reverse-transcription PCR 先以 RNA 为模板合成 cDNA，再用 gene-specific primer 扩增目标片段。End-point RT-PCR 主要观察终点产物；real-time quantitative PCR 在扩增过程中记录 fluorescence，并以 threshold cycle（Ct/Cq）近似反映起始模板量。理想扩增效率为 2 时，相差一个 cycle 约对应两倍起始量；实际必须验证 primer specificity 与 amplification efficiency。

相对定量常用 $\Delta Ct=Ct_{\mathrm{target}}-Ct_{\mathrm{reference}}$，两条件比较再用 $\Delta\Delta Ct$，fold change 为 $2^{-\Delta\Delta Ct}$。该公式依赖 target 与 reference 扩增效率接近，reference gene 在条件间稳定。RT-qPCR 通量低但适合验证预选基因；它与 RNA-seq 是独立测量链，不应期待数值尺度直接相同。

## 微阵列

微阵列以 probe hybridization 测量相对丰度。处理包括背景校正、normalization、probe summarization、批次检查和差异表达。交叉杂交与动态范围是固有限制。

two-color array 比较两个 fluorescent channel，需要处理 dye bias；单通道芯片通常用多个 probe 表示 probe set。RMA 包含背景校正、quantile normalization 和 median-polish summarization。quantile normalization 强制各样本经验分布相同；若处理真的引起全局表达分布变化，该假设可能不成立。

probe sequence 可能映射多个 transcript，旧芯片 annotation 也会过时。重分析时应更新 probe mapping，同时保存原平台和 annotation version。

## RNA-seq

RNA-seq 从 reads 计数出发。原始 count 受测序深度和组成影响；TPM 便于同一样本内比较相对丰度，但差异表达通常直接对 count 建模。常用 negative binomial：

$$
Y_{gi}\sim\operatorname{NB}(\mu_{gi},\alpha_g)
$$

其中 $\mu_{gi}$ 由 library size 和设计矩阵决定，$\alpha_g$ 表示 overdispersion。

DESeq2 常用参数化下

$$
\operatorname{Var}(Y_{gi})=\mu_{gi}+\alpha_g\mu_{gi}^2
$$

方差随均值非线性增加；Poisson 是 $\alpha_g=0$ 的特殊情形。

从 FASTQ 到 count matrix 的链条通常是 quality control、adapter trimming、alignment 或 pseudoalignment、feature assignment 与 sample QC。传统 pipeline 把 reads 定位到 genome/transcriptome；Salmon、Kallisto 一类方法通过 $k$-mer/equivalence class 和概率模型估计 transcript abundance。

建库层面的链条是 RNA 提取、富集或去除 rRNA、reverse transcription、cDNA fragmentation/size selection、adapter ligation 与 sequencing。Paired-end read 提供片段两端和 insert-length 约束，有助于比对、isoform assignment 和 fusion detection；最终的 per-gene count 是 read/fragment assignment 的结果，不是仪器直接读出的“基因分子数”。

negative-binomial GLM 常使用 log link：

$$
\log\mu_{gi}=\log s_i+\mathbf x_i^\top\boldsymbol\beta_g
$$

$s_i$ 是 sample-specific size-factor offset，$\mathbf x_i$ 编码 condition、batch、pairing 等变量。contrast 明确真正检验的系数组合。没有 contrast 的“差异表达”不是完整统计问题。

总 read 数不能解决 composition bias：少数极高表达基因可能占据大量 reads，使其他基因相对 count 普遍下降。DESeq2 默认对每个基因计算跨样本几何均值作为伪参考，再取每个样本非零 count/伪参考比值的中位数作为 size factor。它与 TMM 等方法都依赖“大多数基因不发生强烈同方向变化”的稳健假设；存在全局 RNA output 改变时，通常需要 spike-in 或其他外部尺度。

TPM 先除 transcript length，再令样本总量相同，适合描述样本内部相对组成，不应直接作为 DESeq2 expression matrix。DESeq2 标准输入是未预先按 library size 归一化的非负 count；Salmon/Kallisto 结果应通过 tximport 汇总为 estimated counts，并用 offset 处理 effective transcript length。TPM 已进行长度和总量归一化，失去了原始 count 的采样尺度。

小样本下逐基因 dispersion 很不稳定。经验 Bayes 方法向全局 mean–dispersion trend 收缩，在 gene-specific variation 和估计方差间折中。

## 多重检验

成千上万个基因同时检验时，$p<0.05$ 会产生大量假阳性。Benjamini–Hochberg 控制 FDR。对排序后的 $p_{(i)}$，寻找最大 $i$ 满足

$$
p_{(i)}\le\frac{i}{m}q
$$

再拒绝前 $i$ 个假设。若 $V$ 为错误拒绝数、$R$ 为总拒绝数，则 $\mathrm{FDR}=E[V/\max(R,1)]$。它是重复应用整个检验程序时 false-discovery proportion 的期望，不是某个基因为 null 的后验概率，也不保证当前结果恰有 $qR$ 个假阳性。

BH adjusted p-value 可写为

$$
\tilde p_{(i)}
=\min_{j\ge i}\left(1,\frac{m}{j}p_{(j)}\right)
$$

从最大 rank 向前取 cumulative minimum 可保证单调。先观察结果、任意筛选 gene，再只校正剩余检验会破坏错误率控制；independent filtering 必须使用在 null 下与 p-value 近似独立的统计量。

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

输出顺序恢复为输入顺序。`NaN` 应在调用前单独标记和排除，并在结果中恢复缺失位置。实现验证可检查 adjusted p-value 位于 $[0,1]$，并且按 raw p-value 排序后单调不减。

log fold change 是 effect size，adjusted p-value 是 evidence strength。大样本会使极小 effect 显著，小样本会使大 effect 不稳定；两者与置信区间都应报告。

## Gene Set Enrichment Analysis

Over-representation analysis 先用阈值把基因分成“入选/未入选”，GSEA 则保留全基因排序。标准两类表型流程为：

1. 对每个基因计算与 phenotype 的关联统计量，课程例子使用 $\log_2$ fold change，实际也可用 signal-to-noise、Wald statistic 等；
2. 按统计量从高到低排列全部基因；
3. 对给定 gene set $S$ 沿排序列表行走，遇到 $S$ 内基因按其关联强度加权上升，遇到集合外基因按常数下降；
4. enrichment score（ES）取 running sum 距 0 的最大正偏离或最小负偏离；
5. 置换 phenotype label，重算完整排序与 ES，形成经验 null；
6. 按 gene-set size 与 null 分布归一化为 NES，并在多个 gene set 间控制 FDR。

设排序共有 $N$ 个基因，$N_H=|S|$，权重指数为 $p$，则命中与未命中的累计量可写为

$$
P_{\mathrm{hit}}(i)=
\sum_{\substack{j\le i\\g_j\in S}}
\frac{|r_j|^p}{\sum_{g_k\in S}|r_k|^p},
\qquad
P_{\mathrm{miss}}(i)=
\sum_{\substack{j\le i\\g_j\notin S}}\frac{1}{N-N_H}
$$

$$
ES(S)=\max_i\left(P_{\mathrm{hit}}(i)-P_{\mathrm{miss}}(i)\right)
$$

负向 enrichment 则取绝对值更大的最小偏离。达到 ES 极值之前对分数贡献最大的 set members 构成 leading-edge subset。

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

Phenotype-label permutation 保留基因间相关结构，但需要足够样本；样本很少时常改用 gene-set/per-gene permutation，后者对应不同 null，不能混写。GSEA 偏好协同的同向变化，可能漏掉同一 pathway 中一部分上调、一部分下调的情形。结果受 gene-set database、集合大小、基因 ID 映射与预先定义的集合限制；显著集合说明其成员集中在排序端部，不证明 pathway 拓扑中的反应被激活。

## 表达聚类

两条件分析通常逐基因估计 effect 与显著性；条件、时间点或患者较多时，还可对表达矩阵聚类。按行聚类寻找具有共同响应模式的 gene module，按列聚类寻找 sample subtype 或异常样本。两者使用同一矩阵却回答不同问题，标准化方向也不同：按 gene 做 z-score 会保留相对模式，但移除基因间绝对表达量差异。

Hierarchical clustering、$k$-means 和 self-organizing map（SOM）都可用于表达模式。SOM 把高维样本映射到保持邻域关系的低维网格，结果依赖网格大小、初始化和训练顺序。聚类前的低表达过滤、variance selection、log transform、batch correction 与距离定义通常比算法名称更显著地改变结果。

聚类评价分为两层。Silhouette、within-cluster sum of squares 和重采样稳定性是 internal validation，只使用数据几何；已知 subtype、GO/pathway enrichment、独立 cohort 与实验功能是 external validation。共表达基因可能参与同一过程，因此 cluster 可用于 annotation transfer，但共同表达也可能来自 cell composition、batch 或共同上游因素，不能直接推出相互调控或相同分子功能。

## 实验设计

生物重复不能由技术重复替代。批次与条件完全混杂时，任何算法都无法区分二者。配对设计、协变量、样本随机化和预注册过滤标准通常比事后选择更重要。

design matrix 必须 full rank。若全部 control 来自 batch 1、全部 treatment 来自 batch 2，则 condition 与 batch 完全共线。ComBat 或其他 batch correction 无法恢复实验设计从未提供的可识别性。

> [!EXAMPLE]+ 例8：配对设计
>
> 同一患者治疗前后各取一个样本，模型 `~ patient + treatment` 用 patient effect 吸收个体基线差异。若忽略配对，个体差异进入残差并降低 power；若把技术重复当独立患者，则会伪造样本量。

## 分析检查表

差异表达应从 sample-level QC 和 design matrix 开始，而不是直接调用检验函数。先检查样本相关性、PCA、library composition、低计数 gene、batch 与 outlier，再建立包含 condition、batch、pairing 或 interaction 的设计。需要解释的是预先指定 contrast 的 log2 fold change。

报告至少包含 base mean、log2 fold change、standard error、原始 $p$ 值、BH-adjusted $p$ 值与置信区间；使用 LFC shrinkage 时还要注明 estimator。独立过滤、outlier replacement、样本剔除与批次处理都应记录。

<br>

# 全基因组关联研究

## 概念：全基因组关联研究

- **研究对象与输入输出**
  - GWAS 研究群体中的遗传差异是否与 phenotype 共同变化；
  - 研究对象包括 genotype dosage、离散或连续 phenotype，以及 ancestry、年龄、batch 和亲缘关系等协变量；
  - 输入通常来自经过 QC 和 imputation 的 genotype matrix；
  - 输出是每个 variant 的效应估计、标准误与 $p$ 值，并进一步汇总成 Manhattan plot、QQ plot 和候选 locus。
- **解释边界与后续证据**
  - GWAS 回答“某 allele 是否与性状相关”，不直接回答“该 allele 是否因果致病”；
  - Linkage disequilibrium 会使同一 causal signal 周围出现多个显著 SNP；
  - population structure 也可能产生混杂；
  - 后续还需 fine mapping、功能注释、eQTL/colocalization 和实验验证，才能逐步连接 variant、target gene 与机制。
- **与个体诊断的区别**
  - GWAS 从群体中寻找 association；
  - 个体诊断结合频率、遗传模式、家系和 phenotype 判断某个候选 variant。

## 研究设计

GWAS 在全基因组范围检验 genotype 与 phenotype 的统计关联。Case–control 设计可用 logistic regression：

$$
\log\frac{P(Y_i=1)}{1-P(Y_i=1)}
=\beta_0+\beta_gG_i+\boldsymbol\beta_C^\top C_i
$$

连续性状常用 linear regression。$G_i$ 可编码 effect-allele dosage 0/1/2，协变量 $C_i$ 包括 ancestry principal component、年龄、性别、batch 等。样本相关时可使用 mixed model；imputed genotype 还需检查 INFO score、allele frequency 和 dosage uncertainty。

标准 QC 包括样本 call rate、杂合度、重复或亲缘关系、sex check、ancestry；variant 侧包括 call rate、minor allele frequency、Hardy–Weinberg equilibrium 和 imputation quality。QC 阈值必须在分析前定义，且病例和对照的技术处理若不一致会制造系统性关联。

## 检验、图形与多重校正

- **Manhattan plot**
  - 横轴是按染色体排列的 genomic position；
  - 纵轴通常为 $-\log_{10}p$；
  - 高峰表示某 locus 与性状有统计关联；
  - 峰中许多 SNP 可能只是与同一 causal variant 存在 linkage disequilibrium，不能按显著点数理解为独立机制。
- **QQ plot**
  - 将观察到的 $p$ 值分位数与 null 比较；
  - 用于发现总体 inflation 或 polygenic signal；
  - genomic inflation 不能只靠统一缩放掩盖 population stratification。

若独立检验数为 $m$，Bonferroni 以 $\alpha/m$ 控制 family-wise error rate。常见全基因组阈值 $5\times10^{-8}$ 来自欧洲人群常见变异有效独立检验数的经验量级，不是所有 ancestry、测序研究和 variant class 的普适常数。BH-FDR 控制发现集合中的期望错误比例，更常用于探索性或组学分析；使用何种错误率取决于研究目标。

- **Biological binning**
  - 按 gene、pathway、regulatory region 或其他先验单元聚合 variant；
  - 再进行 burden、variance-component 或 combined test；
  - 它可以减少检验数，并提高多个弱效应或稀有变异的 power。
- **新增假设与边界**
  - bin boundary、效应方向、neutral variant 混入和 LD 都成为新假设；
  - 它不是把同一 bin 内单点 $p$ 值直接相加；
  - 置换、LD-aware summary method、burden test 或 SKAT 等各自定义不同 null。

## 从关联到机制

GWAS 的基本结论是 allele 与 trait 在给定模型和样本中的 association。Population stratification、cryptic relatedness、batch、selection bias 与 phenotype misclassification 都可造成偏差。即使关联可重复，lead SNP 也常只是 tag variant。精细定位需要 credible set、跨 ancestry LD、functional annotation、eQTL/colocalization、染色质互作和实验扰动。

Mendelian randomization、colocalization 和 mediation analysis 各有额外假设，不能由 Manhattan peak 自动推出。最终证据链应区分：

$$
\text{associated locus}
\rightarrow\text{candidate causal variant}
\rightarrow\text{target gene}
\rightarrow\text{molecular mechanism}
\rightarrow\text{phenotype}
$$

“相关不等于因果”在 GWAS 中具体表现为 LD 造成 variant 混淆、群体结构造成个体混淆，以及 pleiotropy 或环境因素造成机制混淆。

<br>

# 机器学习

## 概念：机器学习

- **研究对象与任务类型**
  - 机器学习把生物对象转换成特征矩阵，再从样本中学习预测规则或内部结构；
  - 监督学习使用标签预测类别或连续值；
  - 非监督学习没有目标标签，但仍依赖距离、标准化、簇数和特征选择等假设；
  - PCA、聚类和分类器是通用工具，不会自动理解 gene、protein 或 molecule 的生物语义。
- **输入、输出与训练边界**
  - 输入是 $X\in\mathbb R^{n\times d}$、可选标签 $y$ 和定义独立样本单位的 group key；
  - 输出可以是 prediction、embedding、cluster、feature importance 和带置信区间的评价指标；
  - 预处理、特征选择和超参数搜索必须只在 training data 内完成。
- **章节衔接与泄漏控制**
  - 本章收束表达矩阵、结构特征、化学指纹和蛋白 embedding 等不同输入；
  - 模型性能的关键不只是算法名称；
  - 拆分方式必须阻止同一患者、同源蛋白、相同 scaffold 或未来信息跨集合泄漏。

## 特征与任务

课程语境中的统计特征检测、聚类和分类可用于：

- 蛋白家族或功能分类；
- 样本亚型发现；
- 变异效应预测；
- 药物反应预测；
- 细胞类型识别。

监督任务需要 label，非监督任务不等于没有假设；距离、标准化和簇数都编码了假设。

PCA 寻找最大方差线性方向，适合 QC 与压缩；t-SNE、UMAP 强调局部邻域，二维点间全局距离和 cluster 面积通常不可定量解释。是否 log transform、scale feature、选择 highly-variable gene，往往比降维算法名更深地改变图形。

$k$-means 最小化 within-cluster squared Euclidean distance：

$$
\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|_2^2
$$

它偏好近似球形、方差相近的簇，并只保证收敛到局部最优。hierarchical clustering 还必须指定 linkage：single 容易 chaining，complete 偏好紧致簇，average 取两簇所有点对距离的平均。dendrogram 是数据处理、距离和 linkage 的共同产物，不自动等于生物 taxonomy。

Silhouette coefficient 同时衡量簇内紧致性与最近其他簇的分离度。对样本 $i$，令 $a(i)$ 为它到同簇其他点的平均距离，$b(i)$ 为它到每个其他簇平均距离中的最小值，则

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}\in[-1,1]
$$

接近 1 表示分配清晰，接近 0 表示位于边界，负值表示它平均更靠近另一簇。单点簇通常约定 $s(i)=0$。平均 silhouette 可辅助选择簇数，但它偏好凸、分离良好的簇，不能替代稳定性、外部 annotation 或生物验证；不同距离度量的分数也不能直接比较。

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

分类至少应建立 majority class、logistic regression 或 nearest-neighbor 等简单 baseline。复杂模型若只胜过很弱的 baseline，不能说明表示真正捕获了可泛化生物规律。

## 评估

类别不平衡时，accuracy 可能失真，应报告 confusion matrix、precision、recall/sensitivity、specificity、MCC，以及明确积分方式的 average precision 或 PR-AUC。PR 曲线基线约等于正例 prevalence，因此不同 prevalence 数据集的 PR-AUC 不能直接横向比较。超参数选择只能在 validation 完成，test 只用于最终一次评估。

常见的数据泄漏包括：

- 同一患者的多个样本跨 split；
- 同源蛋白跨 split；
- 预处理在全数据上 fit；
- 标签由未来信息派生；
- batch 与 label 重合。

交叉验证必须按真正独立单位分组。protein function 任务应按 sequence cluster 分组，临床任务按 patient 分组，未来发现任务按时间切分。nested cross-validation 把调参放在内层，外层只估计泛化，避免把 test fold 反复用于模型选择。

AUROC 等于随机抽一对正负样本时正例得分更高的概率，再加上得分相等概率的一半；在极不平衡任务上仍可能显得乐观。precision 无法脱离部署 prevalence 解释。calibration 则检验预测概率与实际频率是否一致。

令 TP、FP、TN、FN 为 confusion matrix 四项：

$$
\mathrm{precision}=\frac{TP}{TP+FP},\quad
\mathrm{recall}=\frac{TP}{TP+FN},\quad
\mathrm{specificity}=\frac{TN}{TN+FP}
$$

$$
\mathrm{F1}=\frac{2PR}{P+R}
$$

概率模型还可用 reliability diagram、Brier score、calibration intercept 与 slope。Platt scaling 或 isotonic regression 必须只在训练/验证数据拟合。

分类阈值应依据漏诊/误报代价、最低 sensitivity 或 decision-curve net benefit 预先选定，不能看过 test 后再调。AUROC、AP 和 sensitivity 等应通过按 patient、protein family 或其他真正独立单位 bootstrap 给出 confidence interval。

## 可解释

feature importance 不等于因果效应。共线特征会分摊或替代重要性；post-hoc explanation 解释模型行为，而不是证明生物机制。候选机制仍需独立数据与干预实验。

SHAP、permutation importance 等还依赖 feature dependence 假设。对高度相关的 gene expression feature，单独打乱一个 gene 会制造训练分布外样本。稳定性分析、外部队列和机制实验比单张 explanation plot 更有说服力。

<br>

# 变异效应与致病性预测

## 概念：变异效应与致病性预测

- **证据层级**
  - 变异分析把基因组坐标上的 REF/ALT 逐步连接到 transcript、protein、gene、disease 和患者 phenotype；
  - consequence annotation 判断变异改了什么；
  - deleteriousness model 判断这种改变是否可能损害分子功能；
  - 临床致病性还需要频率、遗传模式、共分离、功能实验和表型匹配。
- **输入、输出与模型差异**
  - 输入是固定 genome build、allele normalization、transcript 和患者背景后的 variant；
  - 输出可能包括 transcript consequence、蛋白替换、deleteriousness score、population frequency 和患者级候选排序；
  - SIFT、PolyPhen-2、FATHMM、CADD、DANN 的训练目标与分数尺度不同；
  - 不能把多个标签机械投票当作独立临床证据。
- **章节衔接**
  - 本章承接序列保守性、Profile HMM、结构环境和本体表型；
  - GWAS 在群体中寻找 association；
  - 本章判断具体 variant 是否可能影响分子功能并解释某位患者的疾病。

## 问题层次

测序得到 variant 后，至少有四个不同问题：

1. variant 是否改变 transcript、splice site 或 protein sequence；
2. 这种改变是否影响分子功能；
3. 该改变是否降低个体适合度或产生 deleterious effect；
4. 在给定遗传模式、外显率和患者表型下，它是否解释疾病。

Variant consequence annotation 回答第一项，SIFT、PolyPhen-2、FATHMM 和 CADD 主要为第二、三项提供分数，家系共分离、群体频率、功能实验和表型匹配才进入第四项。`damaging`、`deleterious` 与 `pathogenic` 不能互换。计算分数可用于排序或提供 supporting evidence，不能单独完成临床分类。

分析前必须固定 genome build、reference/alternate allele、left normalization、transcript release 和 strand。一个 genomic variant 可能在不同 transcript 上分别成为 synonymous、missense、splice-region 或 intronic variant。蛋白坐标还需记录 isoform；否则同一个 `p.Arg117His` 可能无法唯一映射回基因组。

## SIFT

SIFT 只依赖序列同源信息。流程为：

1. 检索与 query protein 相关的序列；
2. 构建多序列比对；
3. 估计各位置 20 种氨基酸的允许程度；
4. 用最常见氨基酸概率归一化，得到 scaled probability。

- **分数与阈值**
  - SIFT score 范围为 0 到 1；
  - 2009 年 Nature Protocols 通常把 $\leq0.05$ 解释为 predicted damaging；
  - 低分表示该 substitution 在当前同源序列集合与 alignment 中很少被容许，不是 95% 的致病概率。
- **置信度来源**
  - Alignment 太浅会使估计不稳定；
  - 序列过于相近会把许多位置误判为高度保守；
  - 该 protocol 用 median conservation 衡量序列多样性；
  - median information $>3.25$ 时给出 low-confidence warning。

SIFT 与 BLOSUM 的目标不同。BLOSUM 汇总许多家族中的平均 substitution tendency，适合比对打分；SIFT 使用 query-specific MSA，估计特定位置对替换的容忍度。SIFT 不直接使用三维结构、组织特异表达或疾病表型。

## PolyPhen-2

- **特征选择与分类器**
  - PolyPhen-2 同时使用 sequence、structure 和 annotation feature；
  - 2010 年方法先构造 32 个候选特征；
  - 再以 greedy selection 选出 8 个序列特征与 3 个结构特征；
  - 最后用 naive Bayes classifier 区分 damaging 与 non-damaging missense variant。
- **最终特征的含义**
  - 主要描述 wild-type 与 mutant residue 在 MSA 中的 profile、替换出现的同源距离及结构环境；
  - CpG context、Pfam annotation 等属于候选信息，不能全部视为最终 11 个分类特征。

- **训练任务**
  - HumDiv 的 damaging 集合来自引起 Mendelian disease 的 allele，negative 接近跨物种固定差异；
  - HumVar 以人类 disease mutation 对人群中的常见 nsSNP；
  - 两个模型输出不可脱离训练集直接比较。
- **操作点与标签边界**
  - PolyPhen-2 v2.2 对 HumDiv 使用 5% 与 10% false-positive-rate operating point；
  - 对 HumVar 使用 10% 与 20%；
  - `benign`、`possibly damaging` 和 `probably damaging` 不是通用概率阈值，也不是 ACMG/AMP clinical significance。

## FATHMM

FATHMM 用 profile HMM 表示同源序列或保守 protein domain。对 variant 所在 match state，模型比较 wild-type 与 mutant amino acid 的发射概率。未加权版本主要反映进化容忍度；human-weighted 版本再加入 HGMD disease mutation 与 UniProt putatively neutral variant 在 domain 中的相对频率。

若 $P_w$、$P_m$ 分别为 wild-type 与 mutant residue 的 HMM emission probability，原论文的未加权分数为

$$
s_{\mathrm{unweighted}}
=\log\frac{P_m/(1-P_m)}{P_w/(1-P_w)}
$$

负值表示 mutant residue 的 HMM odds 低于 wild type。2013 年论文报告 unweighted 与 human-weighted model 的经验 threshold 分别为 $-3.0$ 和 $-1.5$，human-weighted score 通常以 $<-1.5$ 判为 damaging。阈值依 model 与 version 而异。Disease-specific FATHMM 又对 17 类 disease concept 使用不同权重，其目标是把“可能影响功能”缩小到“与当前疾病类别更相关”。

FATHMM 把本章与 profile HMM、domain annotation 和 ontology 三部分连接起来。性能仍受 domain coverage、训练数据库重叠和 pathogenic/neutral label 质量影响。

## CADD

CADD 面向全基因组 SNV 和短 indel，不限于 missense variant。原始模型把两类 variant 区分开：

- 模拟的 de novo variant，尚未经历自然选择；
- 人类与黑猩猩分化后在人群中固定或近固定的 derived allele，通常已经通过 purifying selection。

2014 年模型将 conservation、regulatory annotation、transcript consequence、chromatin、protein score 等 63 类 annotation 输入 linear SVM。训练标签不是 clinical pathogenic/benign；模型学习的是 observed 与 simulated variant 的 annotation 差异。CADD 因而更接近 genome-wide deleteriousness ranking，不能解释为给定疾病下的致病概率。

Raw C-score 是模型分数，PHRED-like scaled C-score 按所有可能 substitution 的排名计算：

$$
C_{\mathrm{scaled}}=-10\log_{10}\left(\frac{\operatorname{rank}}{N}\right)
$$

因此 CADD 20 表示约位于全基因组可能 SNV 的前 1%，CADD 30 表示前 0.1%。它们是 rank threshold，不是 1% 或 0.1% 的 error probability。Reference build 和 CADD release 会改变 annotation 与分数；2019 年更新支持 GRCh38，并继续扩展 feature 与接口。

```python
import math

def cadd_phred(rank: int, total: int) -> float:
    """把 1-based 全局排名转换为 PHRED-like relative rank。"""
    if not 1 <= rank <= total:
        raise ValueError("rank must be in [1, total]")
    # rank/total 是至少达到该名次的比例，不是错误概率。
    return -10.0 * math.log10(rank / total)
```

## DANN 与调控特征

DANN 使用与 CADD 相近的大规模 annotation，将线性模型替换为深度神经网络，输出 variant 的 deleteriousness score。模型可组合 conservation、蛋白后果和 regulatory feature，但高分仍不是临床 pathogenic probability。DANN 与 CADD 共享部分训练设计和输入 annotation，二者一致不构成两份独立证据。

- **可能受影响的层级**
  - Noncoding variant 的解释尤其依赖细胞类型；
  - Variant 可能改变 transcription-factor binding motif、enhancer/promoter activity、chromatin accessibility、splicing regulatory element 或 RNA stability。
- **证据边界**
  - 只有 motif score 改变，不能证明对应 TF 在相关组织表达、实际结合或影响目标基因；
  - 应把 sequence motif、epigenomic evidence、QTL、三维染色质联系与 perturbation experiment 分层记录。

## 集成与评估

不同方法共享 conservation、MSA、domain 和既有 pathogenic database，输出并非独立证据。把 SIFT、PolyPhen-2、FATHMM 和 CADD 多数票相加，不会自动得到校准概率。Dong 等比较 18 种方法时发现性能随 benchmark 和评价指标变化，部分 ensemble method 优于单项方法；同一研究也显示 missing score、training/test overlap、common benign 与 rare benign 的定义会显著影响结果。

致病性 benchmark 常见偏差包括：

- disease variant 与训练数据库重复；
- benign set 由 common allele 构成，使 allele frequency 成为过强捷径；
- 同一 protein 或同源家族跨 training/test split；
- 只保留所有工具均有分数的 variant，形成 complete-case selection bias；
- 用总体 accuracy 掩盖 rare pathogenic variant 下的类别不平衡；
- 在同一 benchmark 上选择 cutoff 并报告性能。

评价至少报告 coverage、missingness、AUROC、PR-AUC、sensitivity、specificity、MCC 和 calibration。按 gene、protein family 或 discovery time 分组切分可减少泄漏。临床场景还应按 ancestry、variant class、inheritance mode 和 disease area 分层。

## 过滤与优先级

一个 exome 常保留数百个稀有候选 variant，排序需要合并 allele frequency、inheritance、variant effect、gene constraint、phenotype 和家系信息。KGGSeq 2012 年框架分为三层：

1. genetic level：inheritance model、受累者共享区域与家系约束；
2. variant-gene level：群体频率、sequence consequence、conservation、deleteriousness 与 gene feature；
3. knowledge level：disease gene、pathway、interaction 和 literature evidence。

- **频率阈值**
  - 必须与 disease prevalence、penetrance、遗传模式和 ancestry 相容；
  - 隐性病允许健康携带者；
  - 显性高外显率病的候选 allele 通常要求更低 population frequency。
- **家系模型与数据质量**
  - Trio 中要区分 de novo、compound heterozygous、homozygous recessive 和 X-linked model；
  - 同时检查 read-level 与 parental genotype quality。
- **框架扩展**
  - 2017 年 KGGSeq 扩展进一步处理 whole-genome annotation、压缩 genotype access 和大规模 downstream analysis。

三类数据库回答不同问题：

- **gnomAD** 汇总多个群体的大规模 allele frequency 与 gene constraint；频率高于疾病模型允许上限可反驳候选，但 ancestry 覆盖不均、测序可及性和 low-complexity region 会影响“未观察到”的解释；
- **ClinVar** 汇总提交者对 variant clinical significance 的断言，并记录 review status 与冲突；数据库收录不等于专家共识，必须检查条件、日期、星级和证据；
- **OMIM** 整理人类 gene、phenotype 与 Mendelian relationship，适合建立 gene–disease 候选，不提供完整人群频率或逐 variant 的统一临床分类。

数据库角色不能互换。典型过滤链先用 gnomAD 约束频率，再用 consequence/deleteriousness 排序，以 OMIM/HPO 检查 gene–disease 与 phenotype compatibility，最后核对 ClinVar 和原始病例/功能证据。任何数据库都应记录 release 与访问日期。

## 表型语义

Human Phenotype Ontology 将患者表现编码为 DAG。稀有且具体的 phenotype 比常见上位词携带更多信息。Phrank 先取患者与疾病 phenotype set 的 ancestor closure，再对共有 term 的 conditional information content 求和：

$$
\operatorname{Phrank}(\Phi_A,\Phi_B)
=\sum_{\phi\in \operatorname{anc}(\Phi_A)\cap\operatorname{anc}(\Phi_B)}
-\log_2\frac{|G_\phi|}{|G_{\operatorname{pa}(\phi)}|}
$$

$|G_\phi|$ 是与 term $\phi$ 关联的 gene 数，$|G_{\operatorname{pa}(\phi)}|$ 是与其 parent set 关联的 gene 数。Gene score 可取该 gene 所致各 disease score 的最大值。原论文在 169 个真实已诊断病例上评价，并指出基于知识库合成 phenotype 的 benchmark 会高估性能。

- **Phrank 的边界**
  - Phrank 计算 phenotype compatibility，不判断某个 variant 是否破坏蛋白；
  - 完整排序应同时满足 variant-level evidence 与 gene/disease-level phenotype evidence。
- **PhenomeNET Variant Predictor**
  - 用 ontology reasoning 整合人类与 model-organism phenotype；
  - 再由 random forest 合并 semantic similarity、variant pathogenicity、genotype 和 inheritance mode；
  - 目标是支持尚未建立明确人类 gene-disease link 的候选。

报告需保留 HPO release、patient term、negated phenotype、onset、知识库版本及候选 variant 过滤规则。缺失 phenotype 与明确观察为阴性的 phenotype 含义不同；ontology annotation 变化也会改变排序。

<br>

# 化学信息学

## 概念：化学信息学

- **研究对象与表示**
  - 化学信息学把小分子视为带原子标签和键类型的图；
  - SMILES 与 InChI 是线性标识；
  - fingerprint 把局部子结构压缩成稀疏集合或向量；
  - 三维 conformer 加入空间坐标；
  - 同一化合物可能有不同盐型、tautomer、protonation 和 stereoisomer，因此“分子是否相同”先取决于标准化规则。
- **三类常见任务**
  - **相似性搜索**：输入 query molecule 和数据库，输出 Tanimoto 等表示相关的近邻；
  - **QSAR**：输入分子特征与实验 endpoint，输出活性或性质预测；
  - **docking**：输入受体结构与 ligand，输出候选 pose 和近似排序。
- **解释边界与章节衔接**
  - Docking score 不是严格结合自由能，fingerprint 相似不保证活性相似；
  - 本章把字符串、图和机器学习三条主线放到药物分子上；
  - Morgan fingerprint 的生成参数决定 Tanimoto 的含义；
  - QSAR 的 scaffold/time split 决定泛化难度；
  - assay 条件和单位决定标签是否可比较。

## 分子表示

SMILES 是图的线性编码，同一分子可有多个非 canonical SMILES。分子图以原子为节点、键为边；fingerprint 把局部子结构映射为稀疏标识或 bit/count vector。ECFP 是 Rogers–Hahn 描述的一类 extended-connectivity fingerprint；RDKit Morgan fingerprint 是基于 Morgan 式迭代原子标识的可配置实现，二者不应在不说明参数时当作完全同义词。

SMILES 中 ring closure、branch、aromatic atom、formal charge 和 stereochemistry 都有特定语法。canonicalization 是软件定义的确定性排序，不是分子的物理属性。InChI 提供分层标准标识；两者都仍需决定 salt、solvate、tautomer 与 protonation 的标准化规则。

Morgan fingerprint 从每个原子的 initial invariant 出发，按半径逐轮聚合邻居标识并哈希。固定长度 hashed bit/count fingerprint 会发生 collision；未折叠 sparse identifier fingerprint 不存在同一种固定 bit 位碰撞。结果必须记录 atom/feature invariant、bond type、chirality、radius、count/binary、folding size 和软件版本。

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

该实现用于说明“逐层扩张局部环境再折叠到 bit vector”，不等同于 RDKit 的 Morgan/ECFP。成熟实现还定义原子 invariant、键类型、chirality、重复环境去重与 feature invariant；同一分子只有在这些参数和标准化过程一致时才能比较。

两枚 binary fingerprint $A,B$ 的 Tanimoto 相似度为

$$
T(A,B)=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}
$$

```python
def tanimoto(bits_a: set[int], bits_b: set[int]) -> float:
    """计算两枚 binary fingerprint 的 Tanimoto/Jaccard 相似度。"""
    union = bits_a | bits_b
    # 两个空集合的 1.0 是显式约定，调用方可按任务改成 NaN。
    return len(bits_a & bits_b) / len(union) if union else 1.0
```

当两枚 fingerprint 都为空时，集合公式为 $0/0$，代码返回 1.0 是显式软件约定，不是公式推导结果。相似度依赖 fingerprint 定义、半径、bit 长度与标准化规则。盐、tautomer、stereochemistry 和 protonation 不统一，会使同一实体看起来不同。

Tanimoto 只比较 bit overlap。值 0.7 的含义随 fingerprint 与 chemical space 改变，不能作为普适结构相似阈值。activity cliff 指结构非常相似却活性差异巨大，说明 similarity principle 只是经验规律。

## QSAR

QSAR 从结构特征预测活性或性质。典型流程是 molecule standardization、descriptor/fingerprint、scaffold split、模型训练、applicability domain 和不确定性评估。

连续活性常转成 $pIC_{50}=-\log_{10}(IC_{50}/\mathrm{mol\,L^{-1}})$，换算前必须统一为 mol/L。IC50 是指定 assay 条件下达到 50% 抑制的浓度，受 substrate/enzyme concentration、incubation、readout 和 mechanism 影响；它不等于 $K_d$，也不普遍等于 $K_i$。只有在特定竞争抑制与稳态假设下，才能用 Cheng–Prusoff relation 连接 IC50 与 $K_i$。

## 数据整理

QSAR 数据集要统一化学实体、salt form、stereochemistry、tautomer、protonation、单位、target construct、assay format 与 endpoint。`>`、`<` 等 censored activity 不是普通精确值；直接取阈值会引入偏差，应使用区间标签、censored regression 或至少做 sensitivity analysis。重复测量也要报告聚合规则和实验变异。

随机拆分会把相同 scaffold 的近邻分到训练与测试，显著高估新骨架泛化。更严格的 scaffold split 或时间切分更接近药物发现。

applicability domain 描述 query 是否位于训练 chemical space。nearest-neighbor similarity、ensemble disagreement 与 conformal prediction 可提供不确定性线索；模型对 out-of-domain 分子的高置信 prediction 不应直接作为筛选依据。

模型评价除 scaffold/time split 外，还应报告测试分子到训练集的 nearest-neighbor similarity、applicability domain 与 scaffold-level bootstrap，从而区分同一骨架附近的 interpolation 与真正的新骨架 extrapolation。

## 虚拟筛选

ligand-based 方法依赖已知活性分子相似性；structure-based docking 搜索 ligand pose 并用 scoring function 排序。Docking score 是近似相对排序，不是严格结合自由能，也不能替代 assay。

docking 同时包含 pose search 与 scoring。protein protonation、binding-site water、metal、side-chain flexibility 和 ligand tautomer 都会改变结果。常见验证包括 redocking 已知 ligand 检查 pose RMSD，以及 active–decoy benchmark 检查 enrichment。redocking 成功仍不保证对新 scaffold 的排序准确。

<br>

# 药物遗传学

## 概念：药物遗传学

- **研究对象**
  - 药物遗传学研究基因型怎样改变药物暴露、疗效和不良反应；
  - 药代动力学描述机体怎样吸收、分布、代谢和排泄药物；
  - 药效动力学描述药物浓度怎样影响靶点和临床表型；
  - CYP、转运体、受体和 HLA 变异可作用于不同环节。
- **输入与输出**
  - 输入可能是单个 genotype、phased haplotype、star allele、copy number 或全基因组数据；
  - 同时需要结合药物、剂量、适应证和患者临床变量；
  - 输出不是简单的“安全/危险”，而是 metabolizer phenotype、风险变化或特定 guideline 下的用药建议。
- **证据链与章节衔接**
  - 本章连接 GWAS、变异解释与化学信息学；
  - 遗传关联要经过分子后果、PK/PD 表型和临床结局，才能成为可行动建议；
  - CPIC 回答已有基因结果怎样用于处方，不等于决定所有患者是否接受检测。

## 问题

Pharmacogenetics 关注特定基因变异如何影响药物反应；pharmacogenomics 通常覆盖全基因组尺度。表型包括药代动力学、疗效与不良反应。

ADME 过程涉及 absorption、distribution、metabolism、excretion。CYP 酶、转运体、HLA 等基因的变异可改变剂量需求或毒性风险。

临床检测的目标可能是解释异常药物反应、选择替代药、调整起始剂量或降低严重 adverse drug reaction 风险。是否检测取决于适应证、效应量、检测周转时间、可替代方案和 guideline；存在 gene–drug association 不表示所有患者在所有用药场景都应常规筛查。Pre-emptive panel 与处方时 reactive test 的成本、覆盖和返回时机也不同。

- **PK 与 PD**
  - pharmacokinetics 描述机体怎样处理药物；
  - pharmacodynamics 描述药物怎样影响机体。
- **复杂基因型表示**
  - CYP2D6 等基因还存在 copy-number variation、hybrid allele 和 star allele；
  - 简单 SNP dosage 不能完整表示代谢表型；
  - star allele/diplotype 到功能表型的映射是 gene-specific，且会随共识更新。
- **表型命名边界**
  - poor/intermediate/normal/rapid/ultrarapid metabolizer 适用于部分 CYP 基因；
  - 它不是 HLA、转运体和受体的统一分类。

## 关联

对明确的双等位 SNP，可把 effect-allele dosage 编码为 0、1、2：

$$
g(E[Y])=\beta_0+\beta_gG+\beta^\top C
$$

$C$ 包含 ancestry principal components、年龄、性别、合并用药等。population stratification、稀有变异、小样本和 phenotype 定义不稳都会造成偏差。

多等位 HLA、phased haplotype、star allele、structural variant 与 copy-number variation 需要类别变量、单倍型或 activity-score 表示；强制压成单个 SNP dosage 会丢失功能信息。

association 不等于临床可行动性。需要独立复制、效应量、机制证据、风险收益、指南级别和实施环境。PharmGKB 汇总并分级 gene–drug 证据；CPIC 主要回答“基因结果已经可用时应如何解释和用药”，通常不负责决定是否订购检测。

从数据到处方的证据链为

$$
\text{variant}
\rightarrow\text{molecular consequence}
\rightarrow\text{PK/PD phenotype}
\rightarrow\text{clinical outcome}
\rightarrow\text{actionable guideline}
$$

每一步都可能受 ancestry、合并用药、肝肾功能、年龄和 indication 调节。GWAS significance 不能直接替代 dosing recommendation。

全基因组检验还要控制 relatedness、population structure 与 imputation quality。rare variant 单点 power 很低，常按 gene 聚合做 burden 或 SKAT 类检验，但聚合规则同样构成模型假设。

## 临床实例

### CYP2C19 与 clopidogrel

Clopidogrel 是前药，CYP2C19 参与活化。某些适应证下，中间或差代谢者形成的活性代谢物较少，缺血事件风险可能升高。完整流程是由检测到的 star allele 推导 diplotype，再映射 phenotype，最后按当前 CPIC guideline、适应证和替代药禁忌给建议。该结论不能推广成“所有患者都必须检测”。

### HLA-B*57:01 与 abacavir

HLA-B*57:01 表示特定 HLA allele 对超敏反应风险的影响，属于 categorical immune risk，与 CYP activity score 的连续代谢模型不同。

### Warfarin

Warfarin dose 同时涉及 CYP2C9、VKORC1、CYP4F2、ancestry、年龄和临床变量。多个例子共同说明：PGx 不能统一简化成单 SNP 的 0/1/2 dosage，检测覆盖、phasing、CNV、rare allele 与 guideline version 都属于结果。

<br>

# 网络生物学

## 概念：网络生物学

- **研究对象与边语义**
  - 网络生物学把基因、蛋白质、代谢物或疾病表示为节点，把调控、物理结合、代谢反应、共表达或文献关联表示为边；
  - PPI 常是无向物理关系；
  - 转录调控具有方向和符号；
  - 共表达只表示统计相关；
  - 将不同来源的边不加区分地合并为无权图会丢失机制和证据。
- **输入、输出与方法**
  - 输入通常是带类型、方向、权重和来源的 edge list，以及 seed gene 或节点 annotation；
  - 输出可以是 centrality、community、传播排名或候选 link；
  - Modularity 比较社区内实际权重与 degree-preserving null；
  - Random Walk with Restart 从已知 seed 向邻近节点传播分数；
  - 二者分别回答模块发现和候选优先级。
- **解释边界**
  - 网络指标是给定图上的数学量，不自动等于生物重要性；
  - 热门基因更容易被研究，degree 因而带 ascertainment bias；
  - 共表达模块富集某 GO term 也可能重复使用构图数据；
  - 结果需要适当零模型、独立数据和实验验证。

## 图模型

基因调控、蛋白互作、代谢和信号网络可表示为 $G=(V,E)$。边可能有方向、符号、权重、时间和证据来源。把所有来源压成无权图会丢失大量语义。

protein interaction 可能来自 yeast two-hybrid、affinity purification、co-expression 或文本挖掘，它们测量的关系不同。directed regulatory edge、undirected physical interaction 与 signed activation/inhibition 不应在不标注的情况下混入同一 adjacency matrix。

常见量：

- degree：局部连接数；
- betweenness：最短路径经过比例；
- clustering coefficient：邻居间连接程度；
- community：内部连接较密的节点组；
- PageRank/random walk：在网络上传播重要性。

degree distribution 受 ascertainment bias 影响：热门基因被研究更多，数据库 degree 往往更高。betweenness 依赖 shortest-path 假设，生物信号未必沿唯一最短路径传播。centrality 是图上的数学量，不自动等于 essentiality。

对无向、无权图，community detection 常优化 Newman–Girvan modularity：

$$
Q=\frac{1}{2m}\sum_{i,j}
\left(A_{ij}-\frac{k_ik_j}{2m}\right)1[c_i=c_j]
$$

第二项是 degree-preserving null expectation。加权图应以 strength 和总权重替代 degree 与 $m$；有向图还需分别使用入度与出度。modularity 存在 resolution limit，也有大量近简并最优解；高 $Q$ 不证明模块具有生物真实性。应报告 resolution、随机种子、多次运行一致性与独立功能验证。

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

    strength = A.sum(axis=1)
    twice_total_weight = strength.sum()
    if twice_total_weight == 0:
        return 0.0
    same = communities[:, None] == communities[None, :]
    expected = np.outer(strength, strength) / twice_total_weight
    score = (
        (A - resolution * expected) * same
    ).sum() / twice_total_weight
    return float(score)
```

该函数评价一个已有 partition，不负责搜索最优 partition。Louvain 反复把节点移动到可提高 $Q$ 的社区，再把社区收缩成 supernode；Leiden 进一步改善社区连通性。两者都是启发式优化，应以多个随机种子检查近简并解，而不是把一次最高 $Q$ 当作唯一结构。

## 随机游走重启

令 $W$ 为列归一化邻接矩阵，$p_0$ 为 seed，迭代

$$
p_{t+1}=(1-r)Wp_t+rp_0
$$

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

该方法可做疾病基因优先级或功能传播。结果会偏向高 degree 节点；网络的 study bias、缺失边与数据库整合误差必须评估。

收敛解满足

$$
p^{\ast}=r\left[I-(1-r)P\right]^{-1}p_0
$$

其中 $0<r\le1$，$P$ 是与列向量约定一致的 column-stochastic transition matrix。实际大图不显式求逆，而以 power iteration 更新，每轮约 $O(|E|)$。restart $r$ 越大越局部，越小传播越远。

## 富集与模块

发现模块后再做 GO enrichment 会涉及双重选择和依赖检验。网络置换应保持与零假设相关的结构，例如 degree-preserving rewiring，而不是随意打乱所有边。

link prediction 还要防时间泄漏。用当前数据库随机藏边时，模型可能借助后来才发现的共同邻居；更严格方法按 discovery time 切分。未观察到的边只是 unlabeled，不等于确认不存在，因此 negative sampling 会影响评价。

## 零模型

置换必须保留与待检验统计量相关的结构。无向 PPI 图可使用 degree-preserving rewiring；调控网络需分别保持 in/out degree；drug–target 二部图还要保持节点类型；带时间或符号的网络需保留相应属性。

疾病基因排序应与 degree-matched baseline、leave-one-seed-out validation 和 time split 比较，以确认性能不是由研究热度或数据库泄漏造成。网络算法输出的是给定图与 seed 下的 ranking，不等于新的实验因果证据。

<br>

# 文献计算

## 概念：文献计算

- **处理链**
  - 文献计算把论文中的自由文本转换为可检索的实体和关系；
  - 系统先识别 gene、protein、disease、drug 等 mention；
  - 再把不同写法链接到稳定数据库 ID；
  - 最后从句子或篇章中抽取关系及其证据；
  - 字符串相同不保证实体相同，同一实体也可能有多个缩写和历史名称。
- **输入、输出与解释边界**
  - 输入是检索 query、摘要或全文；
  - 输出是 PMID 集合、规范化实体、关系边、证据句和置信度；
  - 自动抽取适合扩大筛选范围，但受命名歧义、否定、实验语境和 publication bias 影响；
  - 不能把模型关系直接当成数据库事实。
- **章节衔接**
  - 本章连接数据检索、知识表示和网络生物学；
  - 文献产生的边必须保留出处和句子级证据，并与实验测得的 interaction 区分；
  - 进入知识图谱前还要统一 ontology、organism 和 identifier namespace。

## 流程

2018 年课程报道提到以自然语言处理调查论文并辅助蛋白分类。典型流程包括：

1. PubMed 检索与去重；
2. sentence splitting、tokenization；
3. gene/protein/chemical named entity recognition；
4. entity normalization 到稳定 ID；
5. relation extraction；
6. evidence sentence 与 provenance 保存。

同名、缩写和物种上下文是主要困难。“蛋白 A 与疾病 B 相关”还需标记肯定、否定、推测、实验对象和关系方向。知识图谱中的边应保留来源句、PMID、方法、置信度和实体对。

NER 与 normalization 是两个问题：前者确定 mention span，后者链接数据库 ID。字符串完全匹配会漏掉别名，也会把 `CAT` 等普通词误连到 gene。normalizer 可结合文内 abbreviation definition、species、邻近 entity 与 candidate prior。

relation extraction 的 sentence-level 方法会遗漏跨句证据；document-level model 覆盖更广，但 negative construction 更难。评价应分别报告 mention/entity precision、recall、normalization accuracy 和 relation F1，不能只给端到端单一分数。

文献数据还带 publication bias：阳性、热门基因和常见疾病更容易发表。数据库可能重复收录同一实验。文本出现次数不等于 effect size，缺乏文献也不证明不存在关系。

## 检索

document retrieval 与 information extraction 应分开评价。检索阶段保存 query、database/version、date、field restriction、deduplication rule 与纳入排除标准，并报告 precision@k、recall 或人工抽样误差。只评价抽取模型而不评价检索召回，会把“论文未被找到”误当作“论文没有关系”。

Boolean query 具有高可解释性，却难覆盖新别名；dense retrieval 能找语义近似文献，但可能牺牲精确控制。系统综述式任务通常组合 controlled vocabulary、关键词、citation chasing 与人工复核。

## 标注

relation label 至少区分肯定、否定、推测、比较、实验对象、物种、方向和 evidence type。annotation guideline 要定义：

- exact 还是 overlap span；
- nested entity 如何处理；
- abbreviation 与 coreference 是否合并；
- 同句未声明关系的 entity pair 是否算 negative；
- 跨句和跨段证据的最大范围；
- annotator disagreement 与 adjudication。

随机按句切分会让同一论文的近重复句跨入训练和测试。应按 PMID、研究或发表时间拆分；normalization 可报告 top-1 accuracy 与 recall@k，relation 则同时给 pipeline 与 gold-entity 条件下结果。

## 证据

知识图谱中的每条边应保留原句、PMID、section、entity ID、model version、confidence 与 extraction date。模型置信度是预测校准量，不是证据强度；同一实验被 review 或数据库重复引用，也不能当作独立复制。

蛋白语言模型处理的是生物序列 token；文献语言模型处理的是自然语言。两者都使用 Transformer，不代表二者语义或评价标准相同。

序列分析提供候选与演化证据，结构分析提供空间机制，功能数据提供细胞和临床上下文。三类证据可以相互校验。

<br>

# 综合实践

## 概念：综合实践

- **组织原则**
  - 综合实践把分散的表示、算法和统计检验组成可复现分析；
  - 重点不是把所有工具串在一起，而是从明确生物问题出发；
  - 每一步都要规定输入、输出、假设、测试和失败条件；
  - 序列、结构与功能项目分别覆盖离散 DP、三维几何和多源数据整合。
- **复现与正确性**
  - 项目结果应能由命令行和固定环境从原始 accession 重建；
  - Notebook 可用于探索；
  - 最终 workflow 需要记录参数、版本、随机种子、checksum 和中间数据；
  - 正确性同时包括算法输出、数据语义和统计解释；
  - 代码运行完成不代表分析成立。

## 项目结构

[2018 年 Stanford Daily 课程报道](https://stanforddaily.com/2018/11/06/classy-classes-cs-274-serves-as-a-rigorous-introduction-to-bioinformatics/)记载，当年课程设置三个较大型 Python 项目，主题为序列分析、三维结构和功能分析，另有四个概念作业。以下实践按这三个方向整理，不对应原作业题面。

### 序列项目

**目标。** 实现 global、local 和 semi-global alignment，支持 linear gap 与 affine gap，并比较参数对结果的影响。

**输入与输出。** 输入包括多条 DNA/protein FASTA、替换矩阵和参数文件；输出包括 score、alignment、CIGAR-like operation 与运行统计。DNA 可使用简单 match/mismatch 评分，蛋白质使用 BLOSUM62。

**验证。** 测试覆盖多个最优解、空序列和错误输入；随机序列用于校准分数，Biopython 用于交叉验证。实验记录序列长度增加时的时间和内存，并用偏离主对角线的已知 alignment 检查 banded alignment 何时遗漏最优解。

### 结构项目

**目标。** 从 mmCIF/PDB 读取 $C_\alpha$ 坐标，实现 Kabsch 叠合，并比较 RMSD、distance-map score 和 TM-score。

**输入与输出。** 输入记录 accession、assembly、chain、model 与 atom selection，同时处理缺失残基和 alternate location。输出包括 transformation matrix、RMSD、coverage、逐残基距离、叠合坐标和离群残基图；报告中注明 correspondence 的来源。

**验证。** 测试包括已知 rotation/translation、reflection、共线点、缺失残基和两个 domain 的相对转动。RMSD 需要与 coverage 和逐残基误差一起报告。

### 功能项目

**目标。** 整合 UniProt、GO 和 bulk RNA-seq 数据，完成差异表达、GO enrichment 及分类模型比较。

**输入与实现。** 输入固定版本的 UniProt accession、GO annotation、count matrix 与 sample metadata。先审计 identifier mapping，分别记录 one-to-many、unmapped、obsolete 和 isoform mapping；再构建设计矩阵、完成差异表达，在预先定义的 background 上进行 enrichment 和 BH 校正，并训练一个线性 baseline 与一个非线性模型。

**验证。** Patient、homology cluster 或时间单位不得跨数据划分；preprocessing 只在 training fold 上拟合；BH 的检验总数和过滤过程需要留档。敏感性分析包括移除高 degree 节点和更换 GO release。整个报告应能在空环境中由一条 workflow command 重建。

## 测试

算法测试包括可手算的 toy case、空序列、全 mismatch、重复字符、tie-breaking、数值下溢、`-inf`、坐标 reflection 和 malformed record。除 unit test 外，还需要与成熟库进行 property-based 对照，并记录运行时间和内存随输入规模的变化。

课程项目强调 OOP 与动态规划实现。类应负责清晰的数据不变量，而不是把所有函数机械包进对象；DP 代码则必须在 docstring 或邻近注释中说明状态、递推、初始化、traceback 起点与停止条件。即使最终实现仍有错误，完整注释也能暴露推理链并获得部分评分；缺少注释的偶然正确输出不能证明算法被正确理解。

## 可复现

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

README 应写明问题、数据许可、下载命令、版本、运行入口和预期产物。原始数据库不宜直接提交；应保存 accession、query、日期、checksum 与处理脚本。

## 交付物

交付内容包括可执行的 Snakemake/Nextflow workflow、lock file 或 container digest、data dictionary、provenance manifest，以及 unit、integration 和 property-based test。Random seed、无法消除的 nondeterminism、figures、metrics 和 failure report 均由流程记录。Model card 或分析说明列出适用边界、known bias 与未解决问题。

Notebook 用于探索，不作为唯一执行入口。最终结果由脚本从原始 accession 重建；手工复制、界面点击和未记录的 spreadsheet edit 不属于可复现流程。

单次课程作业至少应达到更小的交付基线：PEP 8 风格、模块与函数 docstring、关键逻辑的 inline comment、`argparse`/`sys.argv` 命令行入口、环境文件和最小测试。Conda 环境隔离解释器与依赖，lock file 或 container 再固定解析后的确切版本；二者解决的层级不同。

<br>

# 复习提要

## 算法结构

课程中的算法可归入几类计算结构：

- **动态规划**：序列比对、HMM、RNA folding、树似然；
- **索引与启发式搜索**：BLAST、$k$-mer、结构 motif；
- **图算法**：组装、ontology、分子图、网络传播；
- **交替与连续优化**：STRUCTAL 的对应/叠合、能量最小化、模型训练；
- **概率推断**：HMM、系统发育、表达 count、关联；
- **表示学习**：蛋白质语言模型与结构预测；
- **统计校准**：E-value、经验 null、bootstrap、FDR、留出集评估。

- **分析任何方法时统一检查**
  - **输入表示**：决定哪些信息被保留、哪些信息已经丢失；
  - **目标函数**：决定算法实际优化的量；
  - **计算近似**：说明相对精确算法舍弃了什么；
  - **分数解释**：只有给出统计校准后，才可能跨数据集解释；
  - **零模型**：明确“随机”或“背景”具体指什么；
  - **证据层级**：区分相似性、关联、机制和因果；
  - **可复现性**：记录数据版本、软件版本和参数。

按课程期末串讲复习时，可沿下面的依赖链检查是否真正贯通：

1. 生物学对象与测量：复制、转录、翻译、表观遗传，RT-qPCR、microarray、RNA-seq；
2. 序列：global/local/affine alignment，BLAST 九步，MSA，1D motif，HMM；
3. 演化：distance、molecular clock、parsimony、likelihood；
4. 结构：torsion、RMSD 与 correspondence，STRUCTAL，force field，CASP、Rosetta、AF1/AF2；
5. 功能统计：GWAS、GSEA、clustering/silhouette、GO、AUROC 与 AP；
6. 临床解释：deleteriousness、population/clinical database、phenotype、PK/PD 与 pharmacogenomics。

这不是另一套章节顺序，而是从考试问题回到前文原理的索引。每一项都应能回答输入是什么、优化或检验什么、输出如何校准、哪些结论不能由该输出推出。

## 表示与数据

- **表示选择**
  - 同一生物对象可以表示为序列、图、坐标或概率分布；
  - 各种表示保留不同信息，也对应不同可回答问题。
- **质量分数**
  - FASTQ base quality 表示碱基判读误差；
  - SAM mapping quality 表示定位不确定性；
  - 两者不能混用。
- **版本与实体**
  - Genome build、坐标约定、isoform 和 database release 都参与结果定义；
  - 实体解析错误会继续传播到表达、结构和功能分析。

## 序列算法

- **精确索引与组装**
  - KMP 利用模式串的前后缀；
  - FM-index 利用 BWT 和 rank 查询；
  - $k$-mer index 利用固定长度词；
  - de Bruijn 图将组装写成 Eulerian traversal。
- **双序列比对**
  - 全局、局部和半全局比对的区别集中在初始化、终点和 traceback 停止条件；
  - 仿射 gap 用多个状态区分 gap-open 与 gap-extension；
  - 必须注明 $g_o+kg_e$ 或 $g_o+(k-1)g_e$ 的约定；
  - Pair HMM 把相同三状态 DP 解释为隐路径概率。
- **数据库搜索与多序列比对**
  - BLAST 用 seed 筛选候选；
  - E-value 校准随机高分；
  - Progressive MSA 依次合并 sequence/profile；
  - 早期错误可通过 consistency information 和 iterative refinement 缓解。

## 概率与演化

- **Motif 与 HMM**
  - PWM 假设 motif 各位置条件独立；
  - OOPS/ZOOPS/TCM 决定每条序列允许多少次出现；
  - Forward 计算总似然；
  - Viterbi 求最可能路径；
  - Baum–Welch 用期望计数更新参数；
  - Profile HMM 的 delete state 不发射字符，必须沿 profile 坐标在同一观测行传播。
- **系统发育**
  - UPGMA 假设 molecular clock；
  - Neighbor Joining 使用 $Q$ 准则恢复 additive tree；
  - Felsenstein pruning 在树上递推祖先状态似然；
  - Bootstrap support 与 Bayesian clade posterior 来自不同不确定性框架。
- **RNA 结构**
  - Nussinov 最大化配对数；
  - Zuker/Turner 最小化 loop-dependent 自由能；
  - McCaskill 汇总结构 ensemble。

## 结构与动力学

- **坐标、距离与叠合**
  - 坐标保留完整空间信息；
  - 距离矩阵消除刚体变换但不能区分镜像；
  - contact map 还会丢失距离大小；
  - Classical MDS 可从 Euclidean distance matrix 恢复到刚体变换和反射等价的坐标；
  - Kabsch 在对应已知时求最优刚体变换；
  - 对应未知时，STRUCTAL 类方法在顺序约束 DP 与 Kabsch 之间迭代；
  - RMSD 对长度和离群点敏感，TM-score 使用长度归一化和饱和距离权重。
- **一维与三维模式**
  - 1D motif 以 consensus、regex 或 PWM 表示；
  - 3D motif 以残基标签、距离、角度和手性约束表示；
  - 几何命中要相对完整扫描过程的 null 校准；
  - FEATURE 微环境分数、DALI Z-score 和 PWM/BLAST 分数都遵循“特征表示、背景模型、扫描空间、显著性”这条统计链。
- **结构预测**
  - 同源建模使用模板与序列比对；
  - threading 评价序列和结构环境的兼容性；
  - 共进化方法使用 MSA 耦合；
  - AlphaFold1 从 MSA 预测 distogram 与 torsion，再优化学习到的势；
  - AlphaFold2 联合 MSA、pair representation、模板与 structure module 端到端预测坐标；
  - 局部 pLDDT 高而结构域间 PAE 高，表示各结构域局部可信但相对取向不确定；
  - 蛋白质语言模型分数学到的是序列分布，不是折叠自由能。
- **能量、轨迹与 ensemble**
  - 能量最小化寻找附近局部极小值；
  - MD 轨迹是给定模型和初态的时间演化；
  - 平衡 ensemble 定义状态概率；
  - Force field、积分器、thermostat、barostat、约束和采样共同限定可解释的统计量。

## 功能与系统

- **知识表示与转录统计**
  - GO term 定义概念，annotation 连接 gene product 与 term，evidence code 记录证据类型；
  - Pathway 含关系和方向，gene set 只是无序成员集合；
  - Over-representation analysis 检验阈值化列表；
  - GSEA 检查预定义集合是否集中在全基因排序两端；
  - RNA-seq count 具有均值相关方差，适合 negative-binomial model；
  - TPM 已经过长度和总量归一化；
  - Design matrix 不满秩时参数不可识别；
  - FDR 描述检验程序的错误发现比例，effect size 描述差异大小，confidence interval 描述估计不确定性。
- **关联、聚类与证据边界**
  - GWAS 对离散或连续 phenotype 做全基因组关联检验；
  - Manhattan peak 受 LD 影响，不等于独立 causal variant；
  - Bonferroni、FDR 和 gene/pathway binning 控制或重构多重检验问题，但不能把 association 升格为 mechanism；
  - 表达聚类需同时说明距离、标准化、算法和簇数；
  - silhouette 是内部几何指标，不是生物学真实性。
- **变异解释**
  - 变异预测需区分 molecular consequence、deleteriousness 和 clinical pathogenicity；
  - SIFT 使用 query-specific conservation；
  - PolyPhen-2 合并序列与结构特征；
  - FATHMM 使用 profile HMM 与 domain weight；
  - CADD 与 DANN 提供全基因组 deleteriousness 排序；
  - 患者级优先级还要加入 gnomAD population frequency、ClinVar assertion、OMIM gene–disease relation、inheritance、segregation 和 HPO phenotype similarity。
- **数据划分、化学、网络与临床**
  - 按序列簇、患者和时间划分数据，分别控制同源泄漏、个体重复和未来信息泄漏；
  - Morgan fingerprint 迭代聚合原子邻域；
  - Tanimoto 衡量特定指纹下的 bit overlap；
  - QSAR 建立结构与性质模型；
  - docking 搜索结合构象并近似排序；
  - 网络 modularity 比较簇内权重与 degree-preserving null；
  - Random Walk with Restart 从 seed 向网络传播分数，二者回答不同问题；
  - 药物基因组学从 association 到 guideline 还需要分子功能、PK/PD、临床结局和风险收益证据。

<br>

# 论文

本节优先收录课程 Readings 中可以核实出版信息的原始论文。教材节选、讲义和扫描页不冒充原始研究论文；同一方法同时出现原始论文与后续 protocol 时分别列出。

## 课程背景

1. Cohen J. *Computer science and bioinformatics*. Communications of the ACM, 2005. [DOI](https://doi.org/10.1145/1047671.1047672)

## 序列论文

1. Needleman SB, Wunsch CD. *A general method applicable to the search for similarities in the amino acid sequence of two proteins*. Journal of Molecular Biology, 1970. [DOI](https://doi.org/10.1016/0022-2836(70)90057-4)
2. Smith TF, Waterman MS. *Identification of common molecular subsequences*. Journal of Molecular Biology, 1981. [DOI](https://doi.org/10.1016/0022-2836(81)90087-5)
3. Gotoh O. *An improved algorithm for matching biological sequences*. Journal of Molecular Biology, 1982. [DOI](https://doi.org/10.1016/0022-2836(82)90398-9)
4. Altschul SF et al. *Basic local alignment search tool*. Journal of Molecular Biology, 1990. [DOI](https://doi.org/10.1016/S0022-2836(05)80360-2)
5. Karlin S, Altschul SF. *Methods for assessing the statistical significance of molecular sequence features by using general scoring schemes*. PNAS, 1990. [DOI](https://doi.org/10.1073/pnas.87.6.2264)
6. Henikoff S, Henikoff JG. *Amino acid substitution matrices from protein blocks*. PNAS, 1992. [DOI](https://doi.org/10.1073/pnas.89.22.10915)
7. Eddy SR. *Profile hidden Markov models*. Bioinformatics, 1998. [DOI](https://doi.org/10.1093/bioinformatics/14.9.755)
8. Durbin R et al. *Biological Sequence Analysis*. Cambridge University Press, 1998. [出版社](https://doi.org/10.1017/CBO9780511790492)
9. Edgar RC. *MUSCLE: multiple sequence alignment with high accuracy and high throughput*. Nucleic Acids Research, 2004. [DOI](https://doi.org/10.1093/nar/gkh340)
10. Katoh K, Standley DM. *MAFFT multiple sequence alignment software version 7*. Molecular Biology and Evolution, 2013. [DOI](https://doi.org/10.1093/molbev/mst010)

## 索引与组装

1. Burrows M, Wheeler DJ. *A block-sorting lossless data compression algorithm*. Digital Equipment Corporation, 1994. [报告](https://www.hpl.hp.com/techreports/Compaq-DEC/SRC-RR-124.pdf)
2. Ferragina P, Manzini G. *Opportunistic data structures with applications*. FOCS, 2000. [DOI](https://doi.org/10.1109/SFCS.2000.892127)
3. Pevzner PA, Tang H, Waterman MS. *An Eulerian path approach to DNA fragment assembly*. PNAS, 2001. [DOI](https://doi.org/10.1073/pnas.171285098)
4. Li H, Durbin R. *Fast and accurate short read alignment with Burrows–Wheeler transform*. Bioinformatics, 2009. [DOI](https://doi.org/10.1093/bioinformatics/btp324)

## RNA 与系统发育

1. Nussinov R et al. *Algorithms for loop matchings*. SIAM Journal on Applied Mathematics, 1978. [DOI](https://doi.org/10.1137/0135006)
2. Zuker M, Stiegler P. *Optimal computer folding of large RNA sequences using thermodynamics and auxiliary information*. Nucleic Acids Research, 1981. [DOI](https://doi.org/10.1093/nar/9.1.133)
3. Saitou N, Nei M. *The neighbor-joining method: a new method for reconstructing phylogenetic trees*. Molecular Biology and Evolution, 1987. [DOI](https://doi.org/10.1093/oxfordjournals.molbev.a040454)
4. Felsenstein J. *Confidence limits on phylogenies: an approach using the bootstrap*. Evolution, 1985. [DOI](https://doi.org/10.1111/j.1558-5646.1985.tb00420.x)
5. Felsenstein J. *Evolutionary trees from DNA sequences: a maximum likelihood approach*. Journal of Molecular Evolution, 1981. [DOI](https://doi.org/10.1007/BF01734359)

## 结构

1. Kabsch W. *A solution for the best rotation to relate two sets of vectors*. Acta Crystallographica A, 1976. [DOI](https://doi.org/10.1107/S0567739476001873)
2. Arun KS, Huang TS, Blostein SD. *Least-squares fitting of two 3-D point sets*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1987. [DOI](https://doi.org/10.1109/TPAMI.1987.4767965)
3. Holm L, Sander C. *Protein structure comparison by alignment of distance matrices*. Journal of Molecular Biology, 1993. [DOI](https://doi.org/10.1006/jmbi.1993.1489)
4. Holm L, Sander C. *Mapping the protein universe*. Science, 1996. [DOI](https://doi.org/10.1126/science.273.5275.595)
5. Šali A, Blundell TL. *Comparative protein modelling by satisfaction of spatial restraints*. Journal of Molecular Biology, 1993. [DOI](https://doi.org/10.1006/jmbi.1993.1626)
6. Simons KT et al. *Assembly of protein tertiary structures from fragments with similar local sequences using simulated annealing and Bayesian scoring functions*. Journal of Molecular Biology, 1997. [DOI](https://doi.org/10.1006/jmbi.1997.0959)
7. Rohl CA et al. *Protein structure prediction using Rosetta*. Methods in Enzymology, 2004. [DOI](https://doi.org/10.1016/S0076-6879(04)83004-0)
8. Zhang Y, Skolnick J. *TM-align: a protein structure alignment algorithm based on the TM-score*. Nucleic Acids Research, 2005. [DOI](https://doi.org/10.1093/nar/gki524)
9. Marks DS et al. *Protein 3D structure computed from evolutionary sequence variation*. PLoS ONE, 2011. [DOI](https://doi.org/10.1371/journal.pone.0028766)
10. Jumper J et al. *Highly accurate protein structure prediction with AlphaFold*. Nature, 2021. [DOI](https://doi.org/10.1038/s41586-021-03819-2)
11. Baek M et al. *Accurate prediction of protein structures and interactions using a three-track neural network*. Science, 2021. [DOI](https://doi.org/10.1126/science.abj8754)
12. Kabsch W, Sander C. *Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features*. Biopolymers, 1983. [DOI](https://doi.org/10.1002/bip.360221211)
13. Gerstein M, Levitt M. *Comprehensive assessment of automatic structural alignment against a manual standard, the SCOP classification of proteins*. Protein Science, 1998. [DOI](https://doi.org/10.1002/pro.5560070226)
14. Fetrow JS, Skolnick J. *Method for prediction of protein function from sequence using the sequence-to-structure-to-function paradigm with application to glutaredoxins/thioredoxins and T1 ribonucleases*. Journal of Molecular Biology, 1998. [DOI](https://doi.org/10.1006/jmbi.1998.1993)
15. Wallace AC, Borkakoti N, Thornton JM. *TESS: a geometric hashing algorithm for deriving 3D coordinate templates for searching structural databases*. Protein Science, 1997. [DOI](https://doi.org/10.1002/pro.5560061104)
16. Senior AW et al. *Improved protein structure prediction using potentials from deep learning*. Nature, 2020. [DOI](https://doi.org/10.1038/s41586-019-1923-7)
17. Bagley SC, Altman RB. *Characterizing the microenvironment surrounding protein sites*. Protein Science, 1995. [DOI](https://doi.org/10.1002/pro.5560040404)

## 语言模型

1. Rives A et al. *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*. PNAS, 2021. [DOI](https://doi.org/10.1073/pnas.2016239118)
2. Elnaggar A et al. *ProtTrans: Toward understanding the language of life through self-supervised learning*. IEEE TPAMI, 2022. [DOI](https://doi.org/10.1109/TPAMI.2021.3095381)
3. Lin Z et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 2023. [DOI](https://doi.org/10.1126/science.ade2574)
4. Rao R et al. *Evaluating protein transfer learning with TAPE*. NeurIPS, 2019. [论文](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html)

## 表达与知识

1. Ashburner M et al. *Gene ontology: tool for the unification of biology*. Nature Genetics, 2000. [DOI](https://doi.org/10.1038/75556)
2. Stevens R, Goble CA, Bechhofer S. *Ontology-based knowledge representation for bioinformatics*. Briefings in Bioinformatics, 2000. [DOI](https://doi.org/10.1093/bib/1.4.398)
3. The Gene Ontology Consortium. *The Gene Ontology resource: enriching a GOld mine*. Nucleic Acids Research, 2021. [DOI](https://doi.org/10.1093/nar/gkaa1113)
4. Benjamini Y, Hochberg Y. *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society B, 1995. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
5. Love MI, Huber W, Anders S. *Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2*. Genome Biology, 2014. [DOI](https://doi.org/10.1186/s13059-014-0550-8)
6. Subramanian A et al. *Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles*. PNAS, 2005. [DOI](https://doi.org/10.1073/pnas.0506580102)
7. Bolstad BM et al. *A comparison of normalization methods for high density oligonucleotide array data based on variance and bias*. Bioinformatics, 2003. [DOI](https://doi.org/10.1093/bioinformatics/19.2.185)
8. Robinson MD, Oshlack A. *A scaling normalization method for differential expression analysis of RNA-seq data*. Genome Biology, 2010. [DOI](https://doi.org/10.1186/gb-2010-11-3-r25)
9. Wellcome Trust Case Control Consortium. *Genome-wide association study of 14,000 cases of seven common diseases and 3,000 shared controls*. Nature, 2007. [DOI](https://doi.org/10.1038/nature05911)

## 变异效应与优先级

1. Ng PC, Henikoff S. *Predicting deleterious amino acid substitutions*. Genome Research, 2001. [DOI](https://doi.org/10.1101/gr.176601)
2. Ramensky V, Bork P, Sunyaev S. *Human non-synonymous SNPs: server and survey*. Nucleic Acids Research, 2002. [DOI](https://doi.org/10.1093/nar/gkf493)
3. Kumar P, Henikoff S, Ng PC. *Predicting the effects of coding non-synonymous variants on protein function using the SIFT algorithm*. Nature Protocols, 2009. [DOI](https://doi.org/10.1038/nprot.2009.86)
4. Adzhubei IA et al. *A method and server for predicting damaging missense mutations*. Nature Methods, 2010. [DOI](https://doi.org/10.1038/nmeth0410-248)
5. Li MX et al. *A comprehensive framework for prioritizing variants in exome sequencing studies of Mendelian diseases*. Nucleic Acids Research, 2012. [DOI](https://doi.org/10.1093/nar/gkr1257)
6. Shihab HA et al. *Predicting the functional, molecular, and phenotypic consequences of amino acid substitutions using hidden Markov models*. Human Mutation, 2013. [DOI](https://doi.org/10.1002/humu.22225)
7. Kircher M et al. *A general framework for estimating the relative pathogenicity of human genetic variants*. Nature Genetics, 2014. [DOI](https://doi.org/10.1038/ng.2892)
8. Shihab HA et al. *Ranking non-synonymous single nucleotide polymorphisms based on disease concepts*. Human Genomics, 2014. [DOI](https://doi.org/10.1186/1479-7364-8-11)
9. Dong C et al. *Comparison and integration of deleteriousness prediction methods for nonsynonymous SNVs in whole exome sequencing studies*. Human Molecular Genetics, 2015. [DOI](https://doi.org/10.1093/hmg/ddu733)
10. Li M et al. *Robust and rapid algorithms facilitate large-scale whole genome sequencing downstream analysis in an integrative framework*. Nucleic Acids Research, 2017. [DOI](https://doi.org/10.1093/nar/gkx019)
11. Boudellioua I et al. *Semantic prioritization of novel causative genomic variants*. PLoS Computational Biology, 2017. [DOI](https://doi.org/10.1371/journal.pcbi.1005500)
12. Jagadeesh KA et al. *Phrank measures phenotype sets similarity to greatly improve Mendelian diagnostic disease prioritization*. Genetics in Medicine, 2019 (online 2018). [DOI](https://doi.org/10.1038/s41436-018-0072-y)
13. Rentzsch P et al. *CADD: predicting the deleteriousness of variants throughout the human genome*. Nucleic Acids Research, 2019. [DOI](https://doi.org/10.1093/nar/gky1016)
14. Quang D, Chen Y, Xie X. *DANN: a deep learning approach for annotating the pathogenicity of genetic variants*. Bioinformatics, 2015. [DOI](https://doi.org/10.1093/bioinformatics/btu703)

## 化学与网络

1. Rogers D, Hahn M. *Extended-connectivity fingerprints*. Journal of Chemical Information and Modeling, 2010. [DOI](https://doi.org/10.1021/ci100050t)
2. Gaulton A et al. *ChEMBL: a large-scale bioactivity database for drug discovery*. Nucleic Acids Research, 2012. [DOI](https://doi.org/10.1093/nar/gkr777)
3. Barabási AL, Oltvai ZN. *Network biology: understanding the cell's functional organization*. Nature Reviews Genetics, 2004. [DOI](https://doi.org/10.1038/nrg1272)
4. Cowen L et al. *Network propagation: a universal amplifier of genetic associations*. Nature Reviews Genetics, 2017. [DOI](https://doi.org/10.1038/nrg.2017.38)
5. Whirl-Carrillo M et al. *An evidence-based framework for evaluating pharmacogenomics knowledge for personalized medicine*. Clinical Pharmacology & Therapeutics, 2021. [DOI](https://doi.org/10.1002/cpt.2350)
6. Blondel VD et al. *Fast unfolding of communities in large networks*. Journal of Statistical Mechanics, 2008. [DOI](https://doi.org/10.1088/1742-5468/2008/10/P10008)
7. Szklarczyk D et al. *The STRING database in 2021: customizable protein–protein networks, and functional characterization of user-uploaded gene/measurement sets*. Nucleic Acids Research, 2021. [DOI](https://doi.org/10.1093/nar/gkaa1074)
8. Trott O, Olson AJ. *AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading*. Journal of Computational Chemistry, 2010. [DOI](https://doi.org/10.1002/jcc.21334)

## 动力学

1. Alder BJ, Wainwright TE. *Phase transition for a hard sphere system*. Journal of Chemical Physics, 1957. [DOI](https://doi.org/10.1063/1.1743957)
2. Karplus M, McCammon JA. *Molecular dynamics simulations of biomolecules*. Nature Structural Biology, 2002. [DOI](https://doi.org/10.1038/nsb0902-646)
3. Hollingsworth SA, Dror RO. *Molecular dynamics simulation for all*. Neuron, 2018. [DOI](https://doi.org/10.1016/j.neuron.2018.08.011)

## 文献计算与降维

1. Lee J et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. Bioinformatics, 2020. [DOI](https://doi.org/10.1093/bioinformatics/btz682)
2. van der Maaten L, Hinton G. *Visualizing data using t-SNE*. Journal of Machine Learning Research, 2008. [论文](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
3. McInnes L et al. *UMAP: Uniform Manifold Approximation and Projection*. Journal of Open Source Software, 2018. [DOI](https://doi.org/10.21105/joss.00861)

<br>

# 工具

## 序列工具

- [Biopython](https://biopython.org/)：FASTA/GenBank、alignment 与 PDB 的 Python I/O；输入输出格式和 alphabet validation 仍需显式检查。
- [BLAST](https://blast.ncbi.nlm.nih.gov/)：局部相似性检索；记录程序类型、数据库 release、matrix、gap、masking 与 E-value。
- [HMMER](http://hmmer.org/)：profile HMM 搜索与建模；保存 profile、threshold 与 target database。
- [MAFFT](https://mafft.cbrc.jp/alignment/software/)：多序列比对；按序列数量与预期 indel 选择策略。

## 系统发育与 RNA

- [IQ-TREE](http://www.iqtree.org/) / [RAxML-NG](https://github.com/amkozlov/raxml-ng)：maximum-likelihood phylogeny；保存 alignment、model selection、seed 与 bootstrap 设置。
- [ViennaRNA](https://www.tbi.univie.ac.at/RNA/)：RNA 二级结构、MFE 与 partition function，不是 RNA-seq 分析工具。
- [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/) / [MultiQC](https://multiqc.info/)：RNA-seq reads 与批量 QC。
- [STAR](https://github.com/alexdobin/STAR) / [Salmon](https://salmon.readthedocs.io/) / [kallisto](https://pachterlab.github.io/kallisto/)：alignment 或 transcript quantification。
- [DESeq2](https://bioconductor.org/packages/DESeq2/) / [edgeR](https://bioconductor.org/packages/edgeR/) / [limma-voom](https://bioconductor.org/packages/limma/)：差异表达；三者的输入尺度与统计模型不同。

## 结构工具

- [RCSB PDB](https://www.rcsb.org/)：实验结构、validation 与 assembly 元数据。
- [Bio.PDB](https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ)：Python 结构解析；生产代码优先处理 mmCIF。
- [DALI](http://ekhidna2.biocenter.helsinki.fi/dali/) / [TM-align](https://zhanggroup.org/TM-align/)：对应关系未知时的结构比对；同时报告 score、RMSD、aligned length 与归一化约定。
- [FEATURE](https://simtk.org/projects/feature)：以局部理化微环境进行 3D functional-site 建模；模型定义与背景样本决定 score 含义。
- [MDAnalysis](https://www.mdanalysis.org/)：trajectory selection、alignment 与分析。
- [OpenMM](https://openmm.org/) / [GROMACS](https://www.gromacs.org/)：分子动力学；必须记录 force field、water、integrator、constraint、ensemble 与 timestep。

## 变异工具

- [Ensembl VEP](https://www.ensembl.org/info/docs/tools/vep/index.html) / [ANNOVAR](https://annovar.openbioinformatics.org/)：variant consequence 与数据库注释；固定 genome build、transcript set、cache 和插件版本。
- [dbNSFP](https://sites.google.com/site/jpopgen/dbNSFP)：汇总人类 nsSNV 与 splice-site prediction score；不同分数的方向、缺失值和版本不能混用。
- [CADD](https://cadd.gs.washington.edu/)：全基因组 SNV/短 indel deleteriousness ranking；记录 raw/PHRED score、model release 与 reference build。
- [Exomiser](https://exomiser.com/)：结合 inheritance、variant score 与 HPO phenotype 的候选排序；输出仍需家系和临床证据复核。
- [gnomAD](https://gnomad.broadinstitute.org/)：群体 allele frequency、coverage 与 gene constraint；检查 ancestry、release 和 callable region。
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)：variant clinical assertion、condition、submitter 与 review status；冲突记录不能静默合并。
- [OMIM](https://www.omim.org/)：Mendelian gene–phenotype 知识；使用时遵守许可并核对 phenotype mapping。

## 化学、网络与文本

- [RDKit](https://www.rdkit.org/)：standardization、descriptor、fingerprint 与 cheminformatics；Morgan 参数必须完整保存。
- [NetworkX](https://networkx.org/)：中小图原型；大规模传播通常改用 SciPy sparse 或专用图框架。
- [Cytoscape](https://cytoscape.org/)：交互网络可视化与 annotation。
- [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/)：生物医学实体与关系辅助标注；自动结果仍需证据复核。

## 工作流

- [Snakemake](https://snakemake.readthedocs.io/) / [Nextflow](https://www.nextflow.io/)：以 DAG 管理 dependency、resource 与重跑。
- Conda lock、container digest 或 Nix/Guix：固定环境。仅写“使用 Python 3”不足以复现。

工具选择应从输入、输出、假设、规模和许可证出发，而不是按知名度。论文引用、软件版本、参数与 database release 应同时记录；只链接项目主页不能复现实验。
