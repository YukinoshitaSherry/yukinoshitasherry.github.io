---
title: Rebuttal技巧
date: 2025-07-29
categories: 
    - 学CS/SE
tags: 
    - 科研技巧
desc: 写于2025.7 NIPS 人生首次rebuttal,2025~2026持续补完中。虽然人生首投稿NIPS估计是寄了，但经验可以复用，持续努力吧。
---

### 目标

- Rebuttal 的真正听众是 **AC / Editor**，而非只是某位审稿人；形式上是回复审稿意见，实质上是向 AC 呈现一份清晰、可信的材料。
- 在有限字数内体面回应，争取涨分或至少不被进一步扣分；用证据与逻辑说话，而非情绪对抗。
- 集中火力处理核心矛盾；该追问时追问，该申诉时申诉，为 AC 节省阅读成本。

<br>

### 心法

#### 听众与目标

不必以“说服某位审稿人”为唯一目标。尤其面对立场固化的反对者，更现实的做法是**管理好呈现给 AC 的信息**：客观、有据地说明批评是否成立、是否基于片面理解。沉着、专业的语气本身也会提高整体可信度。

#### 引导而非争辩

直接断言“我们是对的”容易触发**逆反心理**（Reactance）：对方感到选择自由受威胁，反而更坚持原立场。更有效的方式是**引**而非**推**——把结论拆成对方难以否认的一串小前提，让 AC 沿逻辑链自行到达终点；捍卫该结论，即是在捍卫其自身的判断。

> [!EXAMPLE]+ 创新性质疑：简单拼接 vs 逻辑链回应
>
> **审稿意见：**“你们的方法不就是方法 A 和方法 B 拼了一下吗？创新性有限。”
>
> **不宜：** “我们完全不同意。我们的方法并非简单拼接，创新性显著。”结论先行，易将 AC 推入“裁决对错”的对立框架。
>
> **宜：**先致谢，再分步陈述：
>
> 1. **前提 A（共识）：** 工作受 A、B 启发；A 在场景 1 有效，B 在场景 2 有效。
> 2. **前提 B（问题）：** 两场景同时出现时，A 与 B 存在具体冲突（如梯度冲突、信息丢失）；简单相加无法解决，甚至可能恶化。
> 3. **前提 C（贡献）：** 为解决该冲突，设计了“协同桥接模块”，目标即化解 A、B 之间的内在矛盾。
> 4. **前提 D（证据）：** 补充实验（如 Table 4）显示：A+B 拼接性能下降；加入该模块后提升至 85%。
>
> 结论可不必明写——AC 会自然得出：增益来自针对核心冲突的新设计，而非 A、B 的简单组合。**不要塞给 AC 一个待评判的观点，而是一份可自行推导的证明。**


<br>

### 步骤

开个共享文档给核心作者们开权限，把原始的review内容复制进去，把weakness等针对性一条条摘取出来。
在下面依次写草稿。

#### 检查
- openreview上会有字数限制，看各个会议的要求。如果限制5000字符等，事先统计好回答的字符数，然后按Response(1/n,2/n...,n/n)切分，同一个weakness下尽量放一起。部分不允许叠楼的，全部问题要都回复；但先放主要的放全，其他次要的暂时略写；可以引导reviewer回复，说due to字数限制，如果您xxx，回复后继续补充/讨论。
- 字少一点、用简单的话、层次（主次）分明，太长没人会看。尽量分点。
- 做到对于每一条review都有明确的回应，不要漏点，不要对某些东西含糊。
- 格式主要是markdown、可以插入latex公式,不能插入链接、图片。所以流程图等等用箭头象形一下。
- 回复的时候，可以先对每个weakness or comment做个简单的总结（短句即可，加粗显示，或者用引用形式：> xxx），然后回复，回复的内容可以突出重点，如果需要可以加粗，斜体，或者代码什么的，不要是纯文字。
- 记得先点击preview检查一下再提交。
- **有涨分马上感谢！**

<br>

### 态度与沟通

审稿压力大、时间紧，部分审稿人水平也参差不齐。即便意见离谱、敷衍或明显误解，仍应保持礼貌、理智，不卑不亢地讨论。

- 用事实与证据说话

反驳以事实与引用为先。证据充分时，宜同时给审稿人台阶，例如：“该工作发表不久，我们亦近期才注意到”；“该现象初看意外，细究后可以理解”。

- 笼统但严重的问题

若审稿人指出一个**严重但不具体**的问题，先聚焦这一点，其余可暂缓回复。礼貌追问细节，并**一次性**把论文中相关工作的总结写出来，减少来回轮次。

> 您说的这个问题很严重，但我们觉得里面可能存在误解。【解释】这个问题很重要，非常期待您的进一步回复。
>
> 【此处接 2–4 句话：概括论文中与此相关的设定、实验或结论】

- 缺少对比：列清已做与未做，请对方指明
> 您提到我们没有和最近的结果作对比，能否更具体一些？我们目前和 X / Y / Z 做了对比。我们没有和 A / B 作对比，原因是 A 的代码未公开、B 的数据集未发布。您指的是哪一个？非常感谢！这个问题很重要，非常期待您的进一步回复。

- 引导区分主要问题与次要问题
可从意见**篇幅、用词严重程度**等判断审稿人心中何者为主、何为次。能清晰区分主次的审稿人，往往更认真；若只是罗列意见，可通过回复**引导（或倒逼）其区分**，使审稿更客观。

**显式区分**（当判断对方把主次搞反时）：
> 根据您的意见，我们感觉您可能认为 A 是主要问题，但 A 并不是主要问题：【1–2 句说明】。您提出的 B 很有意思、很有启发，我们目前的工作确实在这方面有欠缺；但目前尚无工作能完全解决，我们的工作将这一问题向前推进了一步，要彻底解决可能需要另起一项研究。您的意见非常有见地，感谢！

**不显式猜主次**（更稳妥）：
> 您提了两点问题。其中我们认为 A 不是问题：【展开】。B 确实值得讨论，您的意见很有意思……

面对强硬反对，补充**新证据**往往比正面反驳更有效；可表述为“关于您提到的问题，我们补充了实验，发现一个值得注意的现象……”，给对方修正判断的台阶。锚定效应与认知失调会使对方下意识维护初始印象，硬刚只会加强防御。

<br>

### 审稿人类型

实战中常见三类审稿人，策略应有所区分。

#### 支持

已看懂工作并提炼了亮点。策略是**巩固与强化**：感谢之外，补充论据支撑其正面评价，使其成为更有理有据的盟友。

> 非常感谢 Reviewer 1 的洞见。您指出的 XX 优点确实是本文核心；此外，这一点还进一步体现在……

#### 中立

无偏见但有具体困惑（实验设计、论述跳跃等）。这是**主战场**：耐心、尊重地逐条回应；能补实验则补，能澄清则澄清。目标是从“仍有疑问”到“疑虑已清”。

#### 反对

反对立场往往较强。除上文“引导而非争辩”外，可用以下战术给AC写信（回应对象是 AC）：

| 战术 | 做法 |
| :--- | :--- |
| **礼貌重构问题** | 不说“不相关”，而说：“该问题涉及 XX 的更广范畴；本文聚焦于 YY，原因是……”将讨论拉回论文主线 |
| **借力其他审稿人** | 观点与其他审稿人一致时：“关于创新性，我们同意 Reviewer 1 的看法……这也回应了 Reviewer 2 在此处的疑虑。” |
| **点到为止** | 对明显偏见或外行评论：事实性简短回应后即收束。长篇辩解会抬高该问题权重，也显得心虚 |

<br>

### 策略

问题一般分为三类。

#### 好问题，但没时间做，没办法解决 

【最好还是尽可能补充实验，下面是没办法时候全靠论述尽可能挽救一点的方法！】

审稿人提出的问题合理，但在 rebuttal 窗口内无法补全大实验、换数据集或重做方法。常见情形：要求额外 baseline、更大规模验证、新设定下的泛化测试、需长期训练或重新收集数据的实验等。

**不宜**只写“感谢建议，留作 future work”——显得回避，且未说明该问题与本文核心贡献的关系。**也不宜**空泛承诺“camera-ready 一定补上”却不交代为何 rebuttal 期间做不了、现有材料能支撑什么判断。

可行思路是：**承认问题有价值 → 说明边界或客观限制 → 用已有结果、分析或文献作部分回应 → 明确修订版与后续工作如何衔接。**

| 情形 | 回应要点 |
| :--- | :--- |
| **超出本文范围** | 说明论文聚焦的核心问题；该建议重要，但属于另一条线或扩展设定，不宜在本文强行塞入 |
| **需大量算力/数据/时间** | 如实说明 rebuttal 周期内无法完成完整实验；若有可能，给出**轻量替代**（子集、简化设定、已有 checkpoint 的补充分析） |
| **领域内尚无定论** | 承认开放性；强调本文已推进到哪一步；不把“没人解决”当作“本文必须解决” |
| **确有局限但非致命** | 在 Limitation / Discussion 中写清；说明不影响对主结论的支持力度 |

**轻量替代**值得优先考虑：有时不能跑完全部实验，但可以跑一个**小但有说服力**的版本——例如在子集上补一个 baseline、用已有消融说明趋势、补充理论或定性分析。有结果就贴数字；实在没有，也要说明“现有 Table X / Fig Y 已部分覆盖该关切”。

完整回复可参考：

> 非常感谢您提出 XXX。这确实是值得深入的方向。本文当前聚焦于 YYY，因此正文未展开 ZZZ；在 rebuttal 期间，受限于【时间 / 算力 / 数据可得性】，我们未能完成您建议的完整实验。现有结果中，【Table A / 消融 B】表明……，说明在本文设定下主结论仍成立。我们将在修订版的 Discussion / Limitation 中明确讨论该点，并在后续工作中系统研究 ZZZ。

若问题重要但确实与主贡献无关，可礼貌**重构问题**（见“审稿人类型 → 消极反对者”），把篇幅留给致命伤与硬伤；若审稿人将其标为 major weakness，则不宜一笔带过，至少需解释为何不削弱核心 claim。

要点：**承认问题、说清边界与限制、用已有证据兜底；future work 是收尾，不是全文答案。**

#### 好问题，可以加实验解决 

加实验，把结果给 reviewer。

#### 表述不清，影响理解

审稿人常指出概念、符号或方法流程不够清楚。表面是写作问题，实则可能让审稿人**误解方法本身**，甚至把表达缺陷当成设计缺陷。此类意见宜按硬伤处理，不宜只承诺“终稿改进写作”。

仅说“我们将在最终版本中改进写作”的问题在于：未说明**哪里不清楚、打算怎么改、改后能消除何种理解障碍**，审稿人难以相信问题会被真正解决。

回应时应写明**改哪里、怎么改、解决什么障碍**，可从四方面展开：

| 方面 | 回应要点 | 示例表述 |
| :--- | :--- | :--- |
| 概念定义 | 在正文补充正式定义，必要时与已有概念区分 | 在 Section X 补充 XXX 的定义，并说明其与 Y 的区别 |
| 符号统一 | 消除歧义，全文一致 | 统一符号 Y 的用法，避免与 Z 在不同小节含义混淆 |
| 流程图示 | 方法步骤可视化 | 增加流程图或伪代码，标明输入、输出及每一步操作 |
| 实现细节 | 实验可复现性 | 在实验部分补充关键超参数、实现步骤与具体设置 |

完整回复可组织为：

> 感谢您的指出。我们同意当前版本在 XXX 部分表述不够清晰，可能导致读者误解。修订版中将：（一）在 Section X 补充 XXX 的正式定义；（二）统一符号 Y 的使用，避免与 Z 混淆；（三）增加方法流程图或伪代码，说明输入、输出与各步操作；（四）在实验部分进一步说明 XXX 的实现细节。上述修改将使方法描述更完整，便于读者理解具体执行过程。

要点：**不要只说会改写；要指出具体位置、修改方式，以及每项修改所消除的理解障碍。**

#### 坏问题，理解错误

说服对方错在何处，列举 reference 作为证据；语气保持尊重，见上文“态度与沟通”。

eg: scAgents > Weak baselines for some tasks: Comparisons for ATAC and CITE-seq tasks include only simple models like linear regression and random forest, rather than more recent approaches.

Thank you for your insightful comment. Indeed, our comparison for the ATAC and CITE-seq tasks relies on traditional machine learning methods. There are two main reasons for this choice, as our task focuses on predicting responses to unseen perturbations in scATAC-seq and multiomics CITE-seq settings.
First, most current perturbation prediction models are designed for scRNA-seq gene expression prediction11,12 and cannot be directly applied to scATAC-seq or CITE-seq due to substantial modality differences. For example, GERAS1 is tailored for gene node prediction and cannot handle peak features in ATAC-seq or protein features in CITE-seq. scGen2 relies on variational autoencoders with normal or negative binomial assumptions that are mismatched with the distributions observed in ATAC peaks or CITE proteins. scGPT3 is not compatible either, as it cannot process peak or protein tokens absent from its pretrained vocabulary. Similarly, CellOT4 struggles with the high dimensionality of ATAC-seq and does not support prediction under unseen perturbations.
Second, the limited number of existing multimodal perturbation prediction models are not suitable for our scenario of unseen peak prediction. For example, the recent preprint EpiAgent5 proposes an approach similar to scGPT, using cis-regulatory elements (CREs) as tokens to predict perturbation outcomes for scATAC-seq. However, it focuses on CRE token prediction rather than peak-level prediction as in our model. GET 13 also claims to support in silico perturbation prediction, but it is restricted to bulk ATAC-seq knockout experiments and cannot be extended to drug or cytokine perturbations.
For CITE-seq, OT-based method6 predicts one modality from another (e.g., using scRNA-seq to predict CITE-seq), emphasizing cross-modality translation rather than concurrent prediction, which differs from our objective. Additionally, most multimodal generative methods focus on integration rather than perturbation generalization. For instance, totalVI7, MultiVI8, and GLUE15 are developed for data harmonization under shared conditions and cannot generate data under novel perturbations. Biolord14 supports counterfactual predictions in single-modality ATAC or CITE-seq data, but lacks a covariate encoder necessary for generalizing to unseen perturbations.
Lastly, we would like to emphasize that linear regression and random forest are not necessarily naive baselines. Recent studies demonstrate that these classical models can outperform many newly proposed deep learning approaches in scRNA-seq tasks9,10. As such, our use of these models provides meaningful benchmarks and a solid foundation for evaluating the performance of our agent-based methods.


Lotfollahi, M., Wolf, F.A. & Theis, F.J. scGen predicts single-cell perturbation responses. Nat Methods 16, 715–721 (2019).
Cui, H. et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nat Methods 21, 1470–1480 (2024).
Bunne, C. et al. Learning single-cell perturbation responses using neural optimal transport. Nat Methods 20, 1759–1768 (2023).
Chen, X. et al. EpiAgent: Foundation model for single-cell epigenomic data. bioRxiv (2024).
Ryu, J. et al. Cross-modality matching and prediction of perturbation responses with labeled Gromov-Wasserstein optimal transport. arXiv preprint arXiv:2405.00838 (2024).
Gayoso, A. et al. Joint probabilistic modeling of single-cell multi-omic data with totalVI. Nat Methods 18, 272–282 (2021).
Ashuach, T. et al. MultiVI: deep generative model for the integration of multimodal data. Nat Methods 20, 1222–1231 (2023).
Ahlmann-Eltze, C., Huber, W. & Anders, S. Deep learning-based predictions of gene perturbation effects do not yet outperform simple linear baselines. bioRxiv (2024).
Adduri, A. et al. Predicting cellular responses to perturbation across diverse contexts with STATE. bioRxiv (2025).
Li, L. et al. A systematic comparison of single-cell perturbation response prediction models. bioRxiv (2024).
Li, C. et al. Benchmarking AI models for in silico gene perturbation of cells. bioRxiv (2024).
Fu, X. et al. A foundation model of transcription across human cell types. Nature 637, 965–973 (2025).
Piran, Z. et al. Disentanglement of single-cell data with biolord. Nat Biotechnol 42, 1678–1683 (2024).
Cao, Z.-J. & Gao, G. Multi-omics single-cell data integration and regulatory inference with graph-linked embedding. Nat Biotechnol 40, 1458–1466 (2022).

<br>

### 向 AC 申诉

通过 **author-editor confidential comment** 向 AC / SAC / PC 申诉，仅在部分场景值得投入时间。

| 情形 | 建议 |
| :--- | :--- |
| 两名审稿人偏正面（3–4 分 / 满分 5），第三名意见离谱（苛刻、笼统、有明显错误） | 可申诉；同时仍按上文方式体面回复该审稿人，引导其给出更细致意见。即便对方不回复或继续敷衍，AC 看到作者的克制与有理有据，通常会加分 |
| 三名审稿人意见均不佳（≤ 2.5 分） | 正常写 rebuttal 即可，**不必申诉**，基本难有作用；为 AC 节省时间，日后也可能再次遇到同一 AC |

<br>

### 语料

可复用的句式与段落结构（已脱敏）。**按审稿人分开写**，每位审稿人一条完整回复；不单独另写 Global Response。多位审稿人问同一问题时，**把同一段文段复制到各自回复里**（可略改开头致谢），保证每条 thread 自洽、AC 不必跳读。

#### 单条回复结构

对每位审稿人：**致谢并复述其肯定** → **逐条回应 W/C**（与原文 weakness / comment 编号对应）→ **收尾**。若多人质疑 novelty，可在相应审稿人处用编号重述贡献，文段可复用。

> Dear Reviewer [X],
>
> We are deeply grateful for your valuable time and for recognizing that [概括其正面评价]. Below are our detailed responses.
>
> We sincerely thank the reviewer for the detailed and constructive critique. Your comments on attribution evaluation and positioning helped us strengthen both the evidence and the narrative. We address each point below.
>
> **A revised manuscript has been uploaded with added content marked in [color].**


此后每一句话用`We thank the reviewer for`开头。

#### 引用正面评价

巩固支持者时，直接引用**该审稿人**的原话，比泛泛感谢更有力：

> We are deeply grateful for your valuable time and for recognizing that "[审稿人原话概括]" and that [另一条肯定]. Below are our detailed responses.

若需在回复中顺带提及他人正面评价（如借力其他审稿人），只引一句即可，不必集中汇总所有人好评。

#### 共性问题

多位审稿人提同一问题（如开销、可复现性、与某类方法的区别）时，**在同一段落模板写好，分别贴进各审稿人的对应条目下**。每位审稿人处仍写完整论证与表格，不要写「见 Global Response」或「详见 Reviewer Y 的回复」。

效率分析、能力对比表、prompt 附录等**同一套表 / 同一套说明**可在各回复中重复引用编号（如 Tab. R-E1），正文段落宜复制粘贴，仅调整「Thank you for raising…」等开头。

> Thank you — this is a **fair point** [该审稿人若也在 Limitations 中暗示过，可写 we also flagged in our Limitations]. We now report [新增分析]. The key takeaway: [一两句结论]. (See Tab. R-XX.)

#### 承认批评合理

先认同，再说明已采取的行动；若 Limitation 里写过，可点明「我们亦注意到这一点」。

> Thank you — this is a **fair point** that we flagged in our own Limitations, and we now measure it directly.

> We fully agree that [审稿人的具体要求，如：单一指标应交叉验证]. We therefore [具体补救措施].

#### 效率质疑

回应开销时，除绝对 token/延迟外，强调**自适应机制**与**单位收益成本**（如 tokens per accuracy point）：

> We emphasize up front that **[方法名]'s overhead is bounded and adaptive**, not a fixed N× blow-up: [机制说明，如：多数样本在 round 0–1 即停止，远低于上限].
>
> Most importantly, to isolate whether our gains come from *structure* rather than *budget*, we report **[单位收益指标]** (lower = more efficient). [方法] achieves the **best (lowest) [该指标]**, indicating that improvements stem from [设计层面] rather than from spending a larger inference budget.

#### 可复现性

【承诺 + 正文落地】

> We commit to **open-sourcing the full code, all prompts, and configuration files** upon acceptance. In the revision we add (i) [模板/规则清单], and (ii) a **hyper-parameter table** ([关键超参枚举]).

机制黑箱质疑时，用**决策规则 + JSON schema / 伪代码**回应，而非口头描述：

> This is done **without any gradient**, purely by [结构化流程，如：基于执行反馈的 LLM 自反思]. Given [输入], the agent emits a structured verdict following a fixed decision rule:
> - **[条件 A]** → *[失败类型]* → **[操作 1]**.
> - **[条件 B]** → *[失败类型]* → **[操作 2]**.

#### 创新点

【列能力对比表】

与既有工作的区分，用表格比长段文字更清晰；列维度按领域自定（结构、更新机制、是否用反馈、是否连接级/单元级编辑等）：

> Thank you. We add a **capability-comparison table** and make the distinction explicit. Prior methods largely (i) [共性局限 1], or (ii) [共性局限 2]. **[本文方法]** is different in that [核心差异，一两句].

个人回复中贴表后，其他审稿人处**复用同表、同段说明**，不必写「见 Reviewer X」：

> We add a **capability-comparison table** (Tab. R-XX) and make the distinction explicit. Prior methods largely (i) [共性局限 1], or (ii) [共性局限 2]. **[本文方法]** is different in that [核心差异，一两句].


> [!example]+ 2026EMNLP，老师的一个建议
> “我感觉这个回复没有回答清楚，尤其第二段感觉有点偏了，这里需要突出算法的贡献，目前只是说每个模块有用，感觉是把审稿人的说法论证了一下。回复时不宜强行声称提出了全新的学习算法，而应当：
> - 承认框架不依赖额外训练或监督式故障定位；
> - 澄清贡献并不在新优化算法，而在于任务建模、模块化推理机制、可解释的错误定位与修正流程；
> - 强调其与普通 prompt chaining / agent pipeline 的实质区别；
> - 用实验结果证明，这种设计不是简单工程堆叠，而是带来稳定、可复现且可分析的性能收益

#### 机制质疑

【规格说明 + 细粒度消融】

重要机制问题宜**两段式**回应：

> This is a crucial point and we address it in two ways.
>
> **(a) Full specification.** [完整规则、阈值、选择流程]. **We add the full prompts, the JSON schema, and [参数] to the Appendix.**
>
> **(b) Direct validation via fine-grained ablation.** To validate the mechanism itself (not just whole modules), we ablate each [操作/组件]: [Tab. R-XX]. This directly quantifies each operation's contribution.

#### 定义混淆与过度宣称

逐条澄清符号、阈值、缩放因子；承认原文表述不当并已改写：

> Thank you for the careful reading. We clarify each point:
> - **[符号/阈值]** The stopping threshold is **ε = [value]**; the reported **[另一数值] is the converged [指标名], not ε**. We agree the original text conflated the two and have rewritten it.
> - **Leakage / 过度宣称.** We agree that [原表述的问题]. We therefore (i) **soften [位置]** from "[原表述]" to "[更准确表述]," and (ii) add a **held-out evaluation** where [设定].

#### 实验协议

需要解释时，语气尊重、给理由、仍回应对方关切的精神：

> We would like to **respectfully clarify** why a *fully* shared backbone is not the fairest option here. The backbone choices are **dictated by each benchmark's evaluation ecosystem**: [各基准的官方/常用设定]. **Forcing every baseline onto one shared model would run them outside their tuned regime**, introducing *unfair* comparisons rather than removing them. Within each benchmark we already fix a common backbone for all methods. To honor the *spirit* of your request, we additionally report **[折中方案，如：同一 backbone 下的补充结果]**.



#### 失败归因

补充分析「方法何时、为何失败」，显得诚实且指向改进方向：

> When [方法] fails, we categorize the cause into [Stage-I / II / III 等], over a sample of failed cases: [Tab. R-XX]. This localizes where the remaining errors come from and points to concrete future improvements.

#### 同一文段在多处的用法

同一问题被多人提起时：

1. **首选**：在各审稿人对应条目下粘贴**同一段完整回答**（含表或核心数字），保证单条 thread 可读。
2. **可裁剪**：第二位审稿人处可删重复长表，但须保留**结论句 + 表号**，并写一两句摘要，仍不指向 Global 或其他审稿人的回复正文。
3. **不宜**：`Please see Global Response (1)`、`Details in Reviewer X, WY`（迫使 AC 跳读）。

若前文已在本条回复中写过该点，后文可写：

> As detailed above (Tab. R-XX), [两三句摘要].


#### 笔误


> **We also thank you for catching the [Table X] anomaly** ([具体矛盾]); this is a **table-typesetting error** that we have identified and corrected in the revision.


#### 收尾与涨分请求

语气礼貌，不施压；与正文「有涨分马上感谢」一致：

> Thank you again for your constructive [suggestions / feedback]! **If our response[s] and the added experiments address your concerns, we would be grateful if you would kindly consider raising your score.** We welcome any further [questions / discussion].
>
> We are grateful for your careful reading and hope the added attribution audit and clarified positioning address your main concerns. We would be grateful if you would consider raising your score.
>
> Best regards,【可有可无的署名部分】
> Authors of [Paper]

#### 重述贡献

当该审稿人质疑 novelty 时，在**其回复**中用编号重述，文段可在多位审稿人处复用,节约字符数：

> Global Response 
> Existing [领域] treat [对象] as [旧范式], which is brittle when [局限]. **[方法名]** reframes [核心视角], and makes three distinct contributions:
> 1. [贡献 1]
> 2. [贡献 2]
> 3. [贡献 3]

