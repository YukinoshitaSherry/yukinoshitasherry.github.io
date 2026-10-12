---
title: 斯坦福CS329H：Machine Learning from Human Preferences
date: 2026-09-23
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS329H 详细课程笔记：从选择数据、随机效用与 Bradley–Terry，到 RLHF、DPO、辅助博弈、对决赌博机与偏好聚合。
---

- **全名**
  - *Machine Learning from Human Preferences*（从人类偏好学习的机器学习）。
- **主讲**
  - Sanmi Koyejo。
- **课程组**
  - Sang T. Truong、Alex Nam、Justin Hartenstein、Lily Chen。
- **客座**
  - Andy Haupt、Erdem Bıyık、Modibo Camara、Alex Nam、Cassidy Laidlaw、Stewart Slocum、Nivasini Ananthakrishnan。
- **形式**
  - 研究生课，3 学分；
  - 2026 秋季，周一与周三 15:00–16:20（太平洋时间），Bishop Auditorium；
  - 首讲 2026-09-23，项目截止 2026-12-03；
  - 成绩：项目 50%，三次闭卷笔试各 10%，两次口试各 10%。课程主页把项目再细分为分析预案（pre-analysis plan，11 月 4 日截止，占项目 30%）、终稿（manuscript，占 60%）与代码（占 10%）三部分，终稿与代码 12 月 3 日截止；笔试每次 15 分钟，口试每次 12 分钟。
- **先修**
  - 机器学习，建议 CS221、CS229，亦列 CS230、CS224N；
  - 概率与统计，如 CS109；
  - 线性代数与微积分，如 MATH 51、CME 100；
  - Python。
- **入口**
  - [课程主页](https://web.stanford.edu/class/cs329h/)
  - [Stanford Bulletin：CS329H](https://bulletin.stanford.edu/courses/2254281)
  - [教材 *Machine Learning from Human Preferences*](https://mlhp.stanford.edu/)（Truong、Haupt、Koyejo；页眉日期 2026-09-23）
  - 答疑与事务在 Ed；作业在 Gradescope。

这门课回答的问题是：当“好坏”无法事先写成一个损失函数，只能靠人在具体结果之间指出“哪个更好”时，机器学习该怎样建模、怎样学习、怎样提问、怎样行动，以及多人意见不一致时怎样合并。全文的主线是先把“更喜欢”写成一个概率模型（随机效用与 Bradley–Terry），再用这个概率去做估计、征询、策略优化与聚合。

<br>

> [!INFO]+ 版本范围
>
> 正文按 2026 年 9 月课程主页的五单元、十八讲组织。Foundation 讲选择数据、Bradley–Terry、随机效用与因子模型；Learning 讲极大似然、贝叶斯与微调、Fisher 信息、RLHF、DPO 与主动征询；Action 讲辅助博弈、采集函数、Thompson 采样、对决赌博机与偏好贝叶斯优化；Inversion 讲风格控制、代理目标与有界理性；Aggregation 讲 Arrow、Gibbard–Satterthwaite、公平与信息聚合。
>
> 公式来自这些题目上的标准模型：Luce、Bradley–Terry、McFadden 条件 logit、Plackett–Luce、Christiano / Ouyang 的人类反馈强化学习、Rafailov 等人的 DPO、Hadfield-Menell 等人的合作逆强化学习（课程现称 assistance game）、Yue 等人的对决赌博机、Chu–Ghahramani 与 González 等人的偏好高斯过程。2026 年课堂幻灯片在季度进行中才挂出，因此下文不把某次课上的例题编号写成既成事实。
>
> 2023 秋季以论文报告为主，题目含逆强化学习、metric elicitation 与 RLHF。2024 年公开幻灯片亦单列 metric elicitation。2026 年大纲把征询收进实验设计、主动征询与聚合，不再单列该标题。Autumn 2023 阅读中的 Train、McFadden、Luce 仍是 Foundation 的经典来源。后续学期若改讲次名称，选择公理、Bradley–Terry 似然、KL 约束下的最优策略与 DPO 代换通常仍在。
>
> 文末「教材」按站点 2026-09-23 的侧栏写：引言、第 1–5 章、结语、附录。书中第 1.8 节有一句把一切独立同分布噪声都说成满足 IIA，同节 Theorem 1 只等价到 Gumbel，后文按定理。（换言之：按 Theorem 1，在独立同分布加性噪声的随机效用模型里，与 IIA 对应的是 Gumbel 噪声；高斯噪声同样独立同分布，给出的 probit 一般不满足 IIA。）

<br>


# 课程总览：五个单元与一条主线

人类偏好数据有一个共同的形状：人看到若干选项，挑出一个，或者在两者之间指出更好的那个。课程把围绕这种数据的问题分成五个单元，顺序大致是“先建模，再学习，再行动，再反推，最后合并多人”。先看清整张地图，后面每一章的位置才清楚：前两单元把“更喜欢”写成概率并估计参数，后三单元拿这个概率去决定问什么、做什么、以及怎样在多人之间取舍。

## 五个单元各讲什么

- **基础（foundation）**：这一单元建立描述选择的语言与模型。
  - 选择集（choice set）、显示偏好（revealed preference）与陈述偏好（stated preference）；
  - Bradley–Terry（BT）成对模型、可识别性与比较图；
  - 随机效用（random utility model, RUM）、Gumbel 噪声与 logit、IIA、嵌套 logit、Thurstone probit、Plackett–Luce；
  - 条件 logit、低秩因子与混合 logit。
- **学习（learning）**：这一单元讨论怎样从数据中估计这些模型，以及它们怎样接入语言模型训练。
  - 潜变量被积掉之后的观测似然与牛顿法；
  - 最大后验（maximum a posteriori, MAP）、Laplace 近似、监督微调（supervised fine-tuning, SFT）；
  - Fisher 信息与序贯实验设计；
  - 人类反馈强化学习（reinforcement learning from human feedback, RLHF）；
  - 直接偏好优化（direct preference optimization, DPO）；
  - 对语言模型补全对的主动征询。
- **行动（action）**：这一单元让系统在不确定人的偏好时主动做决策、主动提问。
  - 辅助博弈（assistance game）与情境赌博机（contextual bandit）；
  - 采集函数（acquisition function）与 Thompson 采样；
  - 对决赌博机（dueling bandit）的遗憾、以及偏好贝叶斯优化（preferential Bayesian optimization）。
- **反演（inversion）**：这一单元关注“从行为反推效用”时会出什么错。
  - 从行为反推效用时，风格特征与代理奖励（surrogate）会把目标带偏；
  - 有界理性（bounded rationality）下的 softmax 选择，以及配分函数不可计算时仍能用的成对模型。
- **聚合（aggregation）**：这一单元处理多个人偏好不一致时怎样合并。
  - Arrow 不可能定理与 Gibbard–Satterthwaite 定理；
  - 噪声与异质性、孔多塞陪审团定理，以及按样本量加权的合并似然。

## 记录逻辑

整份笔记按下面这条链条推进：先有选择数据，用随机效用解释它，写出观测似然并估计；然后用信息量决定下一次征询什么，用学到的偏好去优化策略；最后处理多人的合并。

$$
\text{选择数据}
\rightarrow\text{随机效用}
\rightarrow\text{观测似然}
\rightarrow\text{信息与征询}
\rightarrow\text{策略优化}
\rightarrow\text{多人聚合}
$$

## 贯穿全课的同一条梯度

许多看似不同的模型，训练时的梯度形状其实一样，记住这一点能省去大量重复推导。

- 条件 logit、最大熵逆强化学习与奖励模型，梯度都是「选中方案的特征，减去模型预测的期望特征」。直觉上，模型把概率质量从“它以为会被选的”挪向“实际被选的”，两者一致时梯度为零。
- 成对 BT 是这项差的特例：抬高胜者，压低败者，权重为当前预测还不确定的程度。预测已经很确定（概率接近 1）且确实猜对的样本，几乎不再推动参数。
- DPO 把奖励换成策略相对参考模型的对数概率比，梯度仍是这一对方向，权重为 $\sigma(-u)$。这里 $u$ 是胜者与败者隐式奖励之差（DPO 一章给出精确定义），$\sigma(-u)$ 就是模型当前判错的概率。

## 各类数字分别回答什么问题

后文会反复出现几类数字，它们回答的问题不同，混用是常见错误。

- 成对准确率会被容易分出胜负的样本拉高；对数损失才评价概率。
- 效用水平可以全体加一个常数，选择概率不变。报告的是差，或一个被钉住的参照项。
- $p(1-p)$ 在 $p=1/2$ 最大。预测已经接近 0 或 1 的比较，几乎不再提供 Fisher 信息。这是主动征询“问最拿不准的对”的数学依据。
- 代理奖励上升，不等于人的偏好被更好地满足。KL 离开参考策略越远，这两者越容易分开。
- 合并全体标注者的 BT，估计的是比较次数加权的平均。它不自动代表少数群体，也不是防策略投票规则。

## 主线：前半建模，后半使用

- **前半：把「更喜欢」写成概率。** 效用差、噪声分布、菜单，三者决定选择概率。
- **后半：用这个概率去问、去优化、去合并。** 问信息量大的对；用 KL 约束优化策略；多人时先声明合并规则放弃了哪条公理。

五条子链分别对应后文的几个大主题：

$$
\begin{aligned}
\text{Gumbel 噪声}
&\rightarrow\text{softmax / Bradley–Terry}
\rightarrow\text{凹的对数似然},\\
\text{参考策略作先验}
&\rightarrow\text{KL 约束的最优策略}
\rightarrow\text{DPO 消去 }Z(x),\\
\text{Fisher 信息}
&\rightarrow\text{下一次比较}
\rightarrow\text{主动征询},\\
\text{隐藏目标 }\theta
&\rightarrow\text{人与系统共同行动}
\rightarrow\text{辅助博弈},\\
\text{序数偏好剖面}
&\rightarrow\text{Arrow / Gibbard–Satterthwaite}
\rightarrow\text{聚合必有所放弃}.
\end{aligned}
$$

逐行读：第一行说独立同分布 Gumbel 噪声让选择概率变成 softmax，成对时即 BT，且对数似然是凹的，便于优化；第二行说把参考策略当先验、加 KL 约束后最优策略有闭式，DPO 借此消去难算的配分函数 $Z(x)$；第三行说 Fisher 信息告诉系统下一次比较该问哪一对；第四行说人知道而系统不知道的目标参数 $\theta$ 让双方的行动构成辅助博弈；第五行说只用序数偏好做聚合时，Arrow 与 Gibbard–Satterthwaite 定理保证必须放弃某些理想性质。

> [!NOTE]+ 概念链
>
> - 观测到的是选项，效用是潜变量。似然是潜变量落在「选中项最大」这一区域的概率。
> - 独立同分布 Gumbel 噪声把该积分收成 softmax。成对情形就是 Bradley–Terry，也就是 logistic。
> - 只给物品截距时，模型只能内插比较图里连通的物品。要预测新物品，效用必须是特征的函数。
> - 奖励模型先拟合 $r_\phi$，再用强化学习优化策略。DPO 在 KL 约束的最优解里把 $r$ 解出来代回 BT，于是只训练策略。
> - 辅助博弈里人知道奖励参数，系统不知道。人的动作同时改变环境和系统的信念。情境赌博机把反馈当成与信念无关的抽样。
> - 风格（长度、格式、语气）若与内容纠缠在特征里，反演得到的效用会奖励风格。代理模型被优化得越狠，这种缝隙越大。
> - 有唯一正确答案且误差独立时，多数票随人数变准。偏好是价值判断时，不存在同时满足 Arrow 四条要求的社会排序。

<br>


# 导论：为什么用比较作监督

很多目标写不成事先给定的标量损失：一句回复是否有帮助、一条机器人轨迹是否安全、医疗筛查里漏报和误报哪一种更贵。要求人直接给出“这句话值 0.73 分”既困难又不稳定，但人通常能在两个具体结果之间指出哪一个更好。这门课后面的估计、征询、策略和投票，用的监督信号都是这种比较。

核心想法一句话：把人的每一次选择看成“在一个菜单里挑出效用最高的那项”，而效用带有随机噪声，于是选择变成了一个可以写似然、可以做统计推断的概率事件。本章先固定菜单、选择、效用这些基本概念与记号，再列出人类反馈的几种形式，最后说明课程要完成的三类任务；下一章从最简单的成对比较模型 Bradley–Terry 开始真正建模。

## 菜单、选择与上下文

- **菜单（choice set）** $C$：一次决策里可用的有限选项集合。
- **选择数据**：二元组 $(C, y)$，或带上下文的三元组 $(x, C, y)$，其中 $y\in C$ 是被选中的项。
- **成对比较**是 $|C|=2$ 的特例；**排序**是对 $|C|$ 个元素给出的一个全序，可以看成一连串“从剩下的项里再挑一个”的选择。

上下文 $x$ 可以是用户、提示词（prompt）、状态或候选的属性。语言模型里常见 $x$ 为提示，$y$ 为补全（completion）；机器人里 $y$ 可以是一段轨迹。同一个菜单在不同上下文下可能得到不同选择，这正是模型需要条件化在 $x$ 上的原因。

## 效用、奖励与系统效用

效用（utility）$U$ 是决策者对选项的潜在评分，观测不到，只能从选择中推断。奖励（reward）在本课与效用指同一类潜在分数：选择模型的文献里叫 utility，强化学习的文献里叫 reward。

随机效用模型把效用拆成两部分：$U = V + \varepsilon$。系统效用（systematic utility）$V$ 是 $U$ 里不含噪声的部分，通常由参数或特征决定；$\varepsilon$ 是随机噪声，刻画人的不一致、未建模的因素和注意力波动。选择概率就是“选中项的 $U$ 最大”这一事件的概率。

<br>

## 记号

下表是全文统一使用的符号。

| 符号 | 含义 |
| :--- | :--- |
| $C$ | 一次选择的菜单 |
| $y\in C$ | 被选中的项 |
| $i\succ j$ | $i$ 优于 $j$ |
| $V_i$ 或 $s_i$ | 系统效用 / BT 分数 |
| $\varepsilon$ | 随机效用噪声 |
| $x_{j}$ | 选项 $j$ 的特征向量 |
| $w$ | 线性效用的参数 |
| $r_\phi(x,y)$ | 奖励模型 |
| $\pi_\theta(y\mid x)$ | 策略，即条件语言模型 |
| $\pi_{\mathrm{ref}}$ | 参考策略，通常是 SFT 模型，训练时冻结 |
| $\beta$ | KL 约束或 Boltzmann 理性的温度系数，$\beta>0$。注意两种写法：KL 约束中 $\pi^\star\propto\pi_{\mathrm{ref}}e^{r/\beta}$，$\beta$ 是温度；有界理性一章写 $P\propto e^{\beta Q}$，$\beta$ 是理性系数（即温度的倒数），$\beta\to\infty$ 表示完全理性 |
| $\sigma(z)$ | logistic，$\sigma(z)=(1+e^{-z})^{-1}$ |

需要注意 $x$ 有两种用法：作上下文时是 $x$，作选项 $j$ 的特征向量时写成带下标的 $x_j$。

### 平移不变性与“钉住”参照项

选择只依赖效用之间的比较，所以所有 $V_i$ 加上同一常数，softmax 不变：

$$
\frac{e^{V_i+c}}{\sum_{k} e^{V_k+c}}=\frac{e^{c}e^{V_i}}{e^{c}\sum_{k} e^{V_k}}=\frac{e^{V_i}}{\sum_{k} e^{V_k}}.
$$

这意味着 BT 分数 $\{s_i\}$ 只确定到一个加法常数，数据本身无法区分 $s$ 与 $s+c$。估计时必须人为消除这个自由度，常见两种做法：钉住 $s_1=0$（把第一个物品当参照），或加高斯先验（先验把分数拉向 0，等价于选定了一个“平均为零附近”的解）。不处理的话，牛顿法里的 Hessian 会是奇异的。

<br>

## 人类反馈的几种形式

人给出的监督不只有成对点击。笔记后面的似然都从下面几种观测改写而来：不同的反馈形式对应不同的“菜单”和不同的模型。

| 观测 | 符号 | 进入的模型 |
| :--- | :--- | :--- |
| 成对比较 | $y_w\succ y_l$ | Bradley–Terry、奖励模型、DPO |
| 全序或前 $k$ 名 | $y_1\succ\cdots\succ y_k$ | Plackett–Luce |
| 菜单中的一项 | $y\in C$ | 多项 logit、条件 logit |
| 示范轨迹 | $\tau$ | 行为克隆、最大熵逆强化学习 |
| 绝对评分 | 整数或实数 | 有序 logit；与别人的量表不能直接相减 |

其中 $y_w$ 是胜者（winner），$y_l$ 是败者（loser）。

### 为什么默认用比较而不是评分

绝对评分带有每个人自己的零点和刻度。甲习惯给 4 分表示“还行”，乙只在非常满意时才给 4 分，所以甲的 $4$ 分与乙的 $4$ 分不是同一个效用。成对比较只要求同一个人在同一次里分出高下：设某人的评分是 $a\cdot U + b$（$a>0$），比较两项时看的是差 $a(U_i-U_j)$，零点 $b$ 在相减时消失，符号也不受刻度 $a$ 影响。课程因此把比较当作默认监督，把评分当成需要再建模刻度的观测（例如有序 logit 里给每个人单独的阈值）。

### 示范是菜单极大的选择

示范是 $|C|$ 极大时的选择：动作空间是所有字符串或所有轨迹，人只给出一个选中的元素，没有显式的败者。SFT 把这个元素的概率拉高；逆强化学习还要求一个「未选中的动作概率更低」的模型，否则败者从未出现在损失里。这一点在讲 softmax 的配分函数时会再出现：要让“没选的更差”，模型必须对整个动作空间归一化。

<br>

## 三类任务：预测、征询、优化

三条任务共用同一个选择概率 $P(y\mid x,C)$，但输入、输出和评价方式不同。

- **预测**：输入 $(x,C)$，输出 $P(y\mid x,C)$。评价是留出（held-out）对数损失。
- **征询**：输入当前后验与候选查询，输出下一次送给人的对。评价是达到同一留出损失所用的查询数，查询越少说明问得越聪明。
- **优化**：输入偏好或由偏好拟合的 $r$，输出策略 $\pi$。评价是人的留出比较、相对 $\pi_{\mathrm{ref}}$ 的 KL，以及长度一类风格统计（用来发现策略是否靠“写得更长”骗取奖励）。

预测模型可以直接当优化用的奖励，也可以只用来决定问哪一对。两条用法对校准（calibration）的要求不同：若模型排序对了而概率贴近 0 和 1（过度自信），优化仍能挑出胜者，因为它只需要知道谁更好；征询则会以为没有信息可问，因为 $p(1-p)\approx 0$ 让每个候选对的信息量都显得很小。所以用于征询的模型必须校准良好。

> [!NOTE]+ 小结
>
> - 选择数据是 $(x,C,y)$：上下文、菜单、选中项；成对比较与排序都是特例。
> - 效用是潜变量，$U=V+\varepsilon$；奖励与效用在本课是同一概念。
> - 选择概率对效用的整体平移不变，估计时要钉住参照项或加先验。
> - 比较消去了个人评分的零点与刻度，因此是默认监督；示范是菜单极大、没有显式败者的选择。
> - 同一个选择概率服务于预测、征询、优化三类任务，征询对校准最敏感。

<br>


# 选择数据与 Bradley–Terry 模型

拿到一批“谁比谁好”的比较之后，第一件事是给每个选项一个分数，并能回答“下一次 $i$ 和 $j$ 比较，$i$ 赢的概率多大”。体育排名（谁更强）、聊天机器人竞技场（哪个模型的回答更受欢迎）、奖励模型训练（哪个补全更好）都是这个问题。Bradley–Terry（BT）模型给出的答案最简单：每个物品一个实数分数，胜率只取决于分数差，并通过 logistic 函数转成概率。

本章先区分数据从哪里来（显示偏好与陈述偏好）以及数据覆盖了哪些对（比较图），再给出 BT 的定义、似然、梯度与 Hessian，用一个三物品的例子手算牛顿法，讨论泛化与评价，最后处理极大似然的存在性、Hunter 的 MM 算法与平局。下一章会说明 BT 其实是随机效用模型在 Gumbel 噪声下的成对特例。

## 数据从哪里来：显示偏好与陈述偏好

显示偏好（revealed preference）是真实做出的选择：点击、购买、接受补全。陈述偏好（stated preference）是假设问题上的回答：标注界面里点选哪一段更好。两者都能进同一个似然，但噪声水平不同，偏差来源也不同：

- 陈述数据缺少真实后果，标注者没有为选择付出代价，可能敷衍或迎合；
- 显示数据则混有习惯与界面位置，例如排在第一位的结果天然更容易被点（位置偏差），用户也可能只是习惯性接受默认选项。

实际使用时，两类数据往往需要不同的噪声尺度，或在模型里显式加入位置等干扰因素。

## 数据覆盖了哪些对：比较图

比较图（comparison graph）以物品为顶点，至少被比较过一次的无序对为边。它决定了哪些分数差能从数据中确定。

BT 的效用差只在连通分支内部确定。两个分支之间没有路径时，各自可以加不同的常数，似然完全相同。例如 $\{A,B\}$ 互相比较过、$\{C,D\}$ 互相比较过，但两组之间从未比较，那么把 $C,D$ 的分数同时加 10 不会改变任何一条观测的概率，数据对“$A$ 与 $C$ 谁强”毫无信息。因此每个连通分支都需要各自钉住一个参照项。后面讨论存在性时会看到，仅有无向图连通还不够，还需要看“谁赢了谁”的有向图。

<br>

## Bradley–Terry 模型的定义

物品 $i$ 有实数分数 $s_i$，不同比较之间相互独立，单次比较满足

$$
P(i\succ j)
=
\frac{e^{s_i}}{e^{s_i}+e^{s_j}}
=
\sigma(s_i-s_j).
$$

第二个等号是分子分母同除以 $e^{s_i}$：$\frac{1}{1+e^{-(s_i-s_j)}}=\sigma(s_i-s_j)$。所以 BT 就是以分数差为输入的 logistic 回归。分数差为 0 时胜率 $1/2$，差为 1 时约 $0.73$，差为 2 时约 $0.88$。

由此 $P(j\succ i)=1-P(i\succ j)$（因为 $\sigma(-z)=1-\sigma(z)$），且 $P(i\succ i)$ 无定义：模型不含平局。Davidson（1970）给平局加了额外参数（见本章“平局”一节）；基础课先把平局从样本里拿掉，或拆成半次胜利。

### 对数似然

记 $w_{ij}$ 为 $i$ 胜过 $j$ 的次数，$n_{ij}=w_{ij}+w_{ji}$ 为 $i$ 与 $j$ 的比较次数。由于比较独立，似然是每次比较概率的乘积，取对数后对所有无序对求和：

$$
\ell(s)
=
\sum_{i<j}
\Big(
w_{ij}\log\sigma(s_i-s_j)
+
w_{ji}\log\sigma(s_j-s_i)
\Big).
$$

这和二项分布的对数似然只差一个与 $s$ 无关的组合数：同一对比较 $n_{ij}$ 次、$i$ 赢 $w_{ij}$ 次。

### 梯度：实际胜场减去期望胜场

对 $s_i$ 的梯度是胜利次数减去期望胜利次数：

$$
\frac{\partial\ell}{\partial s_i}
=
\sum_{j\neq i}
\big(
w_{ij}-n_{ij}\,\sigma(s_i-s_j)
\big).
$$

$n_{ij}\sigma(s_i-s_j)$ 正是模型预测的 $i$ 对 $j$ 的期望胜场。梯度为零的条件因此是：每个物品的实际总胜场等于模型预测的总胜场。这就是总览里“选中减期望”那条梯度在成对情形的样子。逐步推导见本章“梯度与 Hessian 的推导”一节。

### Hessian 与凹性

单对 $(i,j)$ 对 Hessian 的贡献：令 $p=\sigma(s_i-s_j)$，$c=n_{ij}p(1-p)$，则

$$
\frac{\partial^2\ell}{\partial s_i^2}
+=
-c,\qquad
\frac{\partial^2\ell}{\partial s_j^2}
+=
-c,\qquad
\frac{\partial^2\ell}{\partial s_i\partial s_j}
+=
c.
$$

这里 “$+=$” 表示把这一对的贡献累加到 Hessian 对应元素上。每一对贡献的是 $-c\,(e_i-e_j)(e_i-e_j)^{\top}$ 这样一个半负定的秩一矩阵（$e_i$ 为第 $i$ 个单位向量），所以总 Hessian 是比较图上以 $c$ 为边权的图拉普拉斯矩阵取负号。

$\ell$ 关于 $s$ 是凹的。$-H$ 作为拉普拉斯矩阵，其零空间恰好由每个连通分支上的常向量张成；全体分数同加一个常数的方向 $\mathbf{1}$ 永远在零空间里，这就是平移不变性。钉住一个参照分数之后，若比较图连通，Hessian 在自由坐标上负定，$\ell$ 严格凹，极大点至多一个；再加上数据不是完全分离（精确条件是下文 Ford 的强连通），极大点存在，极大似然唯一。

### 完全分离

某个物品从未失败，则把它的分数送到 $+\infty$ 时似然一直上升，极大似然不存在有限解。逻辑回归在线性可分时同样如此：训练集能被一条超平面完美分开，权重的范数就会无限增大。高斯先验或 $L_2$ 惩罚使解回到有限值，因为惩罚项随分数增大而增长，最终压过似然的增益。

<br>

## 例子：三件物品的手算

三件物品 $A,B,C$，钉住 $s_A=0$。比较次数：

| 对 | $A$ 胜 | $B$ 胜 | $C$ 胜 | 经验胜率 |
| :--- | ---: | ---: | ---: | ---: |
| $A$ vs $B$ | 3 | 1 |  | $0.75$ |
| $A$ vs $C$ | 4 |  | 1 | $0.80$ |
| $B$ vs $C$ |  | 3 | 1 | $0.75$ |

经验胜率指每一对中前一个物品的胜率。总胜场 $W_A=3+4=7$，$W_B=1+3=4$，$W_C=1+1=2$。

### 初始点的损失

共 13 次比较。分数全为 0 时每次 $p=1/2$，负对数似然为 $13\log 2=9.0109$。这是“完全不知道谁强”时的基线损失。

### 初始点的梯度

在 $s=0$，$p=1/2$，每一对的期望胜场都是比较次数的一半。按“实际胜场减期望胜场”：

- $s_B$：实际 $1+3=4$，期望 $4/2+4/2=4$，梯度 $0$；
- $s_C$：实际 $1+1=2$，期望 $5/2+4/2=4.5$，梯度 $-2.5$。

所以自由参数 $(s_B,s_C)$ 上对数似然的梯度为 $(0,\ -2.5)$。（$s_A$ 的梯度是 $7-4.5=2.5$，三者之和为 0，这是平移不变性的体现：沿 $\mathbf{1}$ 方向移动不改变似然。）$C$ 赢得比预期少，梯度要求降低 $s_C$；$B$ 恰好符合预期，暂时不动。

### 观测信息矩阵

观测信息（observed information，负 Hessian）为

$$
I
=
\begin{pmatrix}
2 & -1 \\
-1 & 2.25
\end{pmatrix}.
$$

计算：$A$–$B$ 的 $n p(1-p)=4\cdot 1/4=1$，$A$–$C$ 为 $5/4=1.25$，$B$–$C$ 为 $1$。$s_B$ 出现在两条与 $B$ 有关的边（$A$–$B$ 与 $B$–$C$）里，对角元 $1+1=2$；$s_C$ 的对角元 $1.25+1=2.25$；交叉项来自 $B$–$C$，为 $-1$。$A$ 已被钉住，$A$ 所在的行和列被删掉，因此矩阵是 $2\times 2$ 且可逆。

### 牛顿法一步

牛顿法最大化 $\ell$：$s\leftarrow s-H^{-1}\nabla\ell$，其中 $H=-I$，即 $s\leftarrow s+I^{-1}\nabla\ell$。$\det I=2\times 2.25-1=3.5$，

$$
I^{-1}\nabla\ell=\frac{1}{3.5}\begin{pmatrix}2.25 & 1\\ 1 & 2\end{pmatrix}\begin{pmatrix}0\\-2.5\end{pmatrix}=\begin{pmatrix}-2.5/3.5\\-5/3.5\end{pmatrix}.
$$

一步之后

$$
(s_B,s_C)=(-0.7143,\ -1.4286).
$$

值得注意的是 $s_B$ 也下降了，尽管它自己的梯度为 0：$s_C$ 要下降，而 $B$ 对 $C$ 赢了 3 比 1，耦合项把 $B$ 一起往下带，以保持 $B$ 与 $C$ 之间的差距。

### 收敛结果

继续迭代到梯度消失（第二步约为 $(-0.8294,-1.6589)$，第三步已与极限相差不到 $10^{-3}$）：

$$
\hat s_B=-0.8361,\qquad \hat s_C=-1.6723,
$$

$$
\hat P(A\succ B)=0.6977,\quad
\hat P(A\succ C)=0.8419,\quad
\hat P(B\succ C)=0.6977.
$$

负对数似然降到 $7.0854$，比基线 $9.0109$ 低约 $1.93$。

### 传递约束把概率拉离经验胜率

经验 logit 不满足传递：$\mathrm{logit}(0.75)+\mathrm{logit}(0.75)=\log 9=2.197$，而 $\mathrm{logit}(0.80)=\log 4=1.386$。（$\mathrm{logit}(p)=\log\frac{p}{1-p}$，$\mathrm{logit}(0.75)=\log 3$，$\mathrm{logit}(0.8)=\log 4$。）BT 强制

$$
(s_A-s_B)+(s_B-s_C)=s_A-s_C,
$$

所以三对概率不能同时等于经验胜率。拟合结果是 $\mathrm{logit}\hat P(A\succ B)=\mathrm{logit}\hat P(B\succ C)=0.8361$，两者之和等于 $\mathrm{logit}\hat P(A\succ C)$（$2\times0.8361\approx1.6723$）。传递约束把 $A$ 对 $B$ 的胜率从 $0.75$ 拉向 $0.70$，把 $A$ 对 $C$ 从 $0.80$ 拉向 $0.84$。

这意味着 BT 不是在逐对拟合胜率，而是在找一条一维“实力轴”，让所有对的胜率整体最接近数据。各对之间的信息因此互相借用：$A$ 对 $C$ 只比了 5 次，但 $A$–$B$ 与 $B$–$C$ 的结果也在约束它。若数据中存在真正的不传递（如石头剪刀布式的循环偏好），单一分数的 BT 无法表达，需要换模型或换目标（对决赌博机一章的“循环”一节专门讨论这种情形）。

<br>

## 泛化与评价

### 用对数损失而不是准确率

同一分布上的留出评价用对数损失

$$
-\log \hat P(y\mid C),
$$

成对时就是 $-\log\sigma(\hat s_i-\hat s_j)$（$i$ 为实际胜者）。准确率 $\mathbb{I}[\hat P>1/2]$ 对 $\hat P=0.51$ 与 $\hat P=0.99$ 一视同仁，容易对里的样本会主导准确率。例如实力悬殊的对几乎总能猜对，它们会把准确率抬得很高，却无法告诉读者模型在势均力敌的对上概率给得准不准；对数损失则会重罚“自信地猜错”，也会奖励在难对上给出接近 $1/2$ 的诚实概率。

### 新物品：从截距到特征

只有物品截距 $\{s_i\}$ 时，新物品没有参数，无法预测。把分数写成特征的线性函数 $s_i=w^{\top}x_i$ 之后，新物品只要有特征向量 $x_i$ 就能算概率，$P(i\succ j)=\sigma\big(w^{\top}(x_i-x_j)\big)$。这是后文因子模型一章（紧接随机效用一章之后）要解决的外推。奖励模型正是这种做法的神经网络版本：用网络从（提示，补全）中算出分数，从而能给从未见过的补全打分。

### 不连通时的预测是任意的

比较图不连通时，留出集若跨分支，似然对那个相对常数没有信息，预测会随钉扎方式任意变化：钉住哪个物品、先验多强，都会改变跨分支的预测，而训练损失完全一样。拟合前先看连通分支，对跨分支的预测要么拒绝，要么明确说明它来自先验而非数据。

<br>

## 代码：牛顿法拟合 BT

下面的实现最小化负对数似然（NLL），所以梯度和 Hessian 都是上文对数似然版本的相反数。`comparisons` 中每条记录 `(i, j, wins_i, wins_j)` 表示物品 `i` 与 `j` 的一组比较及各自的胜场。

```python
import numpy as np


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def bradley_terry_nll_grad_hess(scores, comparisons):
    """scores: (K,)。comparisons: (i, j, wins_i, wins_j)。返回 NLL、其梯度与 Hessian。"""
    s = np.asarray(scores, dtype=float)
    k = s.shape[0]
    nll = 0.0
    grad = np.zeros(k)
    hess = np.zeros((k, k))
    for i, j, wi, wj in comparisons:
        p = float(sigmoid(s[i] - s[j]))
        nll -= wi * np.log(p) + wj * np.log(1.0 - p)
        g_i = -(wi * (1.0 - p) - wj * p)
        grad[i] += g_i
        grad[j] -= g_i
        weight = (wi + wj) * p * (1.0 - p)
        hess[i, i] += weight
        hess[j, j] += weight
        hess[i, j] -= weight
        hess[j, i] -= weight
    return nll, grad, hess


def fit_bradley_terry(comparisons, k, anchor=0, steps=25):
    """钉住 scores[anchor]=0，牛顿法最小化 NLL。"""
    s = np.zeros(k)
    free = [i for i in range(k) if i != anchor]
    for _ in range(steps):
        _, grad, hess = bradley_terry_nll_grad_hess(s, comparisons)
        h_free = hess[np.ix_(free, free)]
        g_free = grad[free]
        s[free] = s[free] - np.linalg.solve(h_free, g_free)
    return s


comparisons = [(0, 1, 3, 1), (0, 2, 4, 1), (1, 2, 3, 1)]
scores = fit_bradley_terry(comparisons, k=3, anchor=0)
# scores ≈ [0, -0.8361, -1.6723]
```

几处实现细节：

- `g_i = -(wi * (1.0 - p) - wj * p)` 就是 $-(w_{ij}-n_{ij}p)$，即下一节第 2 步分子的相反数；`grad[j] -= g_i` 对应 $\partial\delta/\partial s_j=-1$。
- `np.clip` 只防止极端分数把 `exp` 溢出；在本例的有限解附近它不起作用。
- Hessian 用的是观测权重 $n p(1-p)$，与解析二阶导一致。对 BT 这类 logistic 模型，观测信息与期望（Fisher）信息相同，因为二阶导不依赖于观测到的胜负。
- 参照项不进入 `free`，否则 $H$ 有全 1 零空间，`np.linalg.solve` 会因矩阵奇异而失败。
- 循环固定跑 `steps=25` 步，没有收敛判据；本例 3 步左右就已收敛。若数据完全分离，牛顿法会让分数不断发散，应先检查下文的强连通条件或加 $L_2$ 惩罚（在 NLL 上加 $\tfrac{\lambda}{2}\|s\|^2$，梯度加 $\lambda s$，Hessian 加 $\lambda I$）。

<br>

## 梯度与 Hessian 的推导

只看一对 $(i,j)$。单对的对数似然 $\ell=w_{ij}\log p+w_{ji}\log(1-p)$，$p=\sigma(\delta)$，$\delta=s_i-s_j$。推导用到 logistic 函数的一个关键性质 $\sigma'(\delta)=\sigma(\delta)\big(1-\sigma(\delta)\big)=p(1-p)$。

1. 对 $p$ 求导：$w_{ij}/p-w_{ji}/(1-p)$。
2. 对 $\delta$ 求导时，由链式法则乘 $dp/d\delta=p(1-p)$。第一项变成 $w_{ij}(1-p)$，第二项变成 $w_{ji}p$，分子变成
   $$
   w_{ij}(1-p)-w_{ji}\,p=w_{ij}-n_{ij}p.
   $$
   最后一个等号是展开后合并：$w_{ij}-w_{ij}p-w_{ji}p=w_{ij}-(w_{ij}+w_{ji})p$。
3. $\partial\delta/\partial s_i=1$，$\partial\delta/\partial s_j=-1$。所以 $i$ 得到 $w_{ij}-n_{ij}p$，$j$ 得到相反数。
4. 把所有含 $i$ 的对加起来，就是 $\partial\ell/\partial s_i=\sum_{j\neq i}(w_{ij}-n_{ij}p_{ij})$，其中 $p_{ij}=\sigma(s_i-s_j)$。

二阶导只多一步。$w_{ij}-n_{ij}p$ 再对 $\delta$ 求导，$w_{ij}$ 是常数，得到 $-n_{ij}p(1-p)$。再乘上 $\partial\delta/\partial s_i\cdot\partial\delta/\partial s_i=1$ 或 $\partial\delta/\partial s_i\cdot\partial\delta/\partial s_j=-1$：因此 $s_i$（以及 $s_j$）的对角元累加这个负数，$s_i$ 与 $s_j$ 的交叉元累加其相反数。负对数似然的 Hessian 就是这些权重的正号版本，牛顿法解的是该正定矩阵（参照坐标删除之后）。

这个结果的含义：权重 $n_{ij}p(1-p)$ 在 $p=1/2$ 时最大。势均力敌的对对曲率贡献最多，也就是对分数估计提供的信息最多；一边倒的对几乎不提供信息。这正是后文 Fisher 信息与主动征询的出发点。

<br>

## 极大似然解的存在性：有向胜利图必须强连通

前面说过“数据不能完全分离”，这里给出精确条件。Zermelo（1929）最早研究这个模型，并在同类条件下证明了解的存在与唯一；Ford（1957）独立重新提出该模型，给出了有限极大似然的图条件。作有向图：只要 $i$ 至少赢过 $j$ 一次，就连一条 $i\to j$。似然在实数上达到唯一最大值（差一个加法常数）当且仅当这张图强连通：任意两点都能沿有向边互达。

等价的说法是：把物品任意分成两个非空组，每组都至少有一个物品赢过另一组的某个物品。直觉上，如果某一组从未赢过另一组，数据就无法说明“落后多少”，只能说明“落后”，而落后得越多似然越大。

### 强连通失败时分数发散

强连通失败时，可以把落后的一组分数一起送向 $-\infty$，似然不下降：组内比较的分数差不变，跨组比较的概率都向“领先组获胜”的方向趋近 1。极端情形：只有 $A$ 赢 $B$ 三次，没有反向胜利。负对数似然是 $-3\log\sigma(s_A-s_B)$，差越大损失越小，有限极大点不存在。前文 13 次比较里每一对都有双向胜利，有向图强连通，所以牛顿法停在有限解。

### 弱连通不够

弱连通不够。$A\to B\to C$ 且没有返回边时，图是弱连通的（忽略方向后连通），分数差仍然发散：$C$ 从未赢过，$A$ 从未输过。拟合前画的是有向胜利图，不只是无向比较图。两种图的作用不同：无向图连通保证「有路径把常数钉住」这一必要结构里的比较发生过，对应 Hessian 在自由坐标上可逆；有向强连通才保证胜利没有把某一侧推到无穷，对应极大点真的存在。

实践中，若数据不满足强连通（小样本或新物品常见），常用做法是加高斯先验 / $L_2$ 惩罚，或给每对加入少量伪计数；但需要清楚此时的分数差部分来自先验。

<br>

## MM 算法：每步有闭式的小步更新

牛顿法每步要解一个 $K\times K$ 的线性方程组，物品数 $K$ 很大时代价高。Hunter（2004）的 MM 算法（minorization–maximization）每步有闭式，极限与牛顿法相同。“MM”的思路是：每一步构造一个处处不高于目标、且在当前点与目标相切的下界函数（minorizer），然后最大化这个容易处理的下界；目标函数因此每步单调不降。

### 更新公式

$W_i=\sum_{j\neq i}w_{ij}$ 为 $i$ 的总胜场。更新

$$
s_i^{+}
=
\log W_i
-
\log\sum_{j\neq i}
\frac{n_{ij}}{e^{s_i}+e^{s_j}},
$$

再减去参照项，使 $s_A$ 保持为 0。

### 更新从哪里来

记 $\gamma_i=e^{s_i}>0$，对数似然改写为

$$
\ell(\gamma)=\sum_{i}W_i\log\gamma_i-\sum_{i<j}n_{ij}\log(\gamma_i+\gamma_j).
$$

难处在第二项把 $\gamma_i$ 和 $\gamma_j$ 耦合在一起。利用凹函数 $\log$ 的切线不等式 $-\log x\ge 1-\log y-x/y$（等号在 $x=y$ 成立），取 $x=\gamma_i+\gamma_j$，$y=\gamma_i^{(k)}+\gamma_j^{(k)}$（当前迭代值），得到下界

$$
\ell(\gamma)\ \ge\ \sum_{i}W_i\log\gamma_i-\sum_{i<j}n_{ij}\frac{\gamma_i+\gamma_j}{\gamma_i^{(k)}+\gamma_j^{(k)}}+\text{const}.
$$

右边对各 $\gamma_i$ 完全分离，对 $\gamma_i$ 求导置零：$W_i/\gamma_i=\sum_{j\neq i}n_{ij}/(\gamma_i^{(k)}+\gamma_j^{(k)})$，于是 $\gamma_i^{+}=W_i\big/\sum_{j\neq i}\frac{n_{ij}}{\gamma_i^{(k)}+\gamma_j^{(k)}}$，取对数即上面的更新式。由于 $\gamma$ 全体乘一个常数时更新结果也乘同一常数，每步之后减去参照项（即把 $\gamma$ 归一化）不影响迭代。

### 在例子上走一步

沿用前文数据，$W=(7,4,2)$，$n_{AB}=4$，$n_{AC}=5$，$n_{BC}=4$。从 $s=0$ 出发，$e^{s_i}+e^{s_j}=2$。

$$
\begin{aligned}
s_A^{+}&=\log 7-\log\big(4/2+5/2\big)=\log(7/4.5)=0.4418,\\
s_B^{+}&=\log 4-\log\big(4/2+4/2\big)=\log 1=0,\\
s_C^{+}&=\log 2-\log(4.5)=-0.8109.
\end{aligned}
$$

减去 $s_A^{+}$：$(0,\ -0.4418,\ -1.2527)$。继续迭代，第 15 步到 $(0,\ -0.8361,\ -1.6722)$，与牛顿法的极大似然一致（再迭代几步末位变为 $-1.6723$）。

### 与牛顿法比较

MM 步长短、每步不必解线性方程，每步代价只与比较对数成正比，适合大规模稀疏比较；牛顿法在凹目标上通常几步就到（本例 3 步对 15 步），但每步要解方程组。两者都要先满足强连通。最直接的失败是某物品从未赢过：$\log W_i$ 里会出现 $W_i=0$，更新无定义，这和分数发散是同一件事。需要补充的是，$W_i>0$ 对所有物品成立并不足以保证强连通（例如两组各自内部有胜有负，但一组从未赢过另一组），这时 MM 每步都有定义，但两组的分数差会随迭代不断拉大而不收敛，所以仍要先检查有向胜利图。

<br>

## 平局：Davidson 模型

基础 BT 把平局的概率定为 0。而在体育比赛、或者标注界面提供“差不多”选项时，平局很常见。Davidson（1970）加一个参数 $\theta\ge 0$：

$$
\begin{aligned}
P(i\succ j)
&=
\frac{e^{s_i}}{e^{s_i}+e^{s_j}+\theta e^{(s_i+s_j)/2}},\\[4pt]
P(i\sim j)
&=
\frac{\theta e^{(s_i+s_j)/2}}{e^{s_i}+e^{s_j}+\theta e^{(s_i+s_j)/2}}.
\end{aligned}
$$

$P(j\succ i)$ 把第一式分子换成 $e^{s_j}$，三者之和为 1。平局项 $e^{(s_i+s_j)/2}=\sqrt{e^{s_i}e^{s_j}}$ 是两者强度的几何平均（Davidson 原文写作 $\nu\sqrt{\pi_i\pi_j}$，$\pi_i=e^{s_i}$）。

如何理解：$\theta=0$ 时分母回到两项，平局概率为 0，模型就是 BT。分子分母同除以 $e^{(s_i+s_j)/2}$ 可得平局概率为 $\theta/\big(2\cosh(\delta/2)+\theta\big)$，$\delta=s_i-s_j$，它在 $\delta=0$ 时最大、随 $|\delta|$ 增大而减小。所以 $\theta$ 大时，分数接近的对更容易被判成平局，悬殊的对仍主要分出胜负。另外，在不平局的条件下，$P(i\succ j\mid\text{非平局})=\sigma(s_i-s_j)$，与 BT 完全相同。

把平局随机拆成半个胜利，等于在 $\theta=0$ 的模型里掺入噪声；数据里平局很多时，应把 $\theta$ 放进似然一起估计。

<br>

## 代码：MM 算法

```python
def mm_bradley_terry(comparisons, k, anchor=0, steps=30):
    """Hunter MM。comparisons: (i, j, wins_i, wins_j)。返回钉住参照后的分数。"""
    wins = np.zeros(k)
    n_pair = {}
    for i, j, wi, wj in comparisons:
        wins[i] += wi
        wins[j] += wj
        n_pair[(i, j)] = n_pair.get((i, j), 0.0) + wi + wj
        n_pair[(j, i)] = n_pair[(i, j)]
    s = np.zeros(k)
    for _ in range(steps):
        new = np.zeros(k)
        for i in range(k):
            total = 0.0
            for j in range(k):
                if i == j:
                    continue
                nij = n_pair.get((i, j), 0.0)
                total += nij / (np.exp(s[i]) + np.exp(s[j]))
            new[i] = np.log(wins[i]) - np.log(total)
        s = new - new[anchor]
    return s
```

代码与公式一一对应：`wins` 是 $W_i$，`n_pair` 是对称的 $n_{ij}$，`total` 是 $\sum_{j\neq i}n_{ij}/(e^{s_i}+e^{s_j})$，最后一行减去参照项。每一轮用的是上一轮的全部分数（同步更新）；也可以逐个分数轮流更新（循环版本），每一步同样是在对一个下界做最大化。

修正说明：原代码建立 `n_pair` 时写的是 `n_pair[(i, j)] = n_pair[(j, i)] = wi + wj`，同一对物品若在 `comparisons` 里出现多条记录，后一条会覆盖前一条而不是累加；这里改为累加，与 `wins` 的累加方式一致（对本例每对只有一条记录的数据，两种写法结果相同）。

`wins[i]=0` 的物品使 `log` 无定义，调用方应先把该物品划出连通分支，而不是加一个很小的伪胜场假装存在有限 MLE。如上一节所说，所有 `wins[i]>0` 仍不保证强连通，调用前最好直接检查有向胜利图。

> [!NOTE]+ 小结
>
> - BT：$P(i\succ j)=\sigma(s_i-s_j)$，分数只确定到加法常数，每个连通分支需钉住一个参照项。
> - 梯度是“实际胜场减期望胜场”，Hessian 是以 $n_{ij}p(1-p)$ 为权的图拉普拉斯取负，对数似然凹。
> - 有限 MLE 存在且唯一当且仅当有向胜利图强连通（Zermelo / Ford）；仅无向连通不够，不满足时用先验或 $L_2$ 惩罚。
> - 传递约束使拟合概率偏离逐对经验胜率；评价用对数损失，不用准确率。
> - 牛顿法几步收敛但每步解方程组；Hunter 的 MM 每步闭式、单调上升，极限相同。
> - 平局用 Davidson 模型：多一个参数 $\theta$，$\theta=0$ 退化为 BT。

<br>

<br>

# 随机效用模型（Random Utility Models）

前一章的 Bradley–Terry 模型直接假设「$i$ 胜 $j$ 的概率是 $\sigma(s_i-s_j)$」，但没有回答这条公式从哪里来、为什么是 sigmoid 而不是别的曲线、菜单里不止两项时应当怎样推广。随机效用模型给出一个统一的生成故事：决策者心里对每个选项有一个效用，这个效用由「可以用参数描述的系统部分」与「建模者看不到的随机扰动」两部分组成；决策者总是选效用最大的那一项。观测者只看到最后选了谁，看不到效用本身。

核心想法一句话：**选择概率的形状完全由噪声的分布决定**。噪声取 Gumbel 分布得到 softmax（多项 logit），菜单只有两项时退化为 Bradley–Terry；噪声取正态分布得到 probit（Thurstone 模型）；把同一个想法逐位应用到排序上得到 Plackett–Luce；把潜在效用切成等级得到有序 logit。本章依次推导这些模型，并讨论它们共同的性质（Luce 公理与 IIA）及其局限（红巴士 / 蓝巴士）。下一章把系统部分 $V$ 写成特征或潜在因子的函数，使模型能泛化到新物品与新用户。

形式上，菜单（choice set）$C$ 中选项 $i$ 的效用为

$$
U_i=V_i+\varepsilon_i.
$$

其中 $V_i\in\mathbb{R}$ 是系统部分（systematic utility），是待估参数的函数（例如自由截距 $s_i$，或特征的线性组合 $w^{\top}x_i$）；$\varepsilon_i$ 是随机扰动，代表建模者观测不到的口味、情绪、注意力等因素。决策者选 $U$ 最大的项。观测不到 $U$，只观测 $y=\arg\max_i U_i$，即

$$
y=\arg\max_{i\in C} U_i .
$$

因此可识别的只有「谁最大」这一序信息：给所有 $U_i$ 同加一个常数，或同乘一个正数，$\arg\max$ 都不变。前者意味着只有 $V$ 的差有意义，后者意味着 $V$ 的尺度要靠固定噪声的方差来钉住（Gumbel 标准化、probit 中令差的方差为 1，都是这种钉扎）。噪声 $\varepsilon$ 的分布决定选择概率的形状。

<br>

## 从 Gumbel 噪声推导多项 logit

为什么选 Gumbel？因为它是「最大值」的稳定分布：独立 Gumbel 变量的最大值仍是 Gumbel，这使「某一项是最大值」的概率能够写成闭式。下面完整推导。

### Gumbel 分布

标准 Gumbel（Type I extreme value，第一类极值）分布，众数为 0，其 CDF 与密度为

$$
F(\varepsilon)=\exp(-e^{-\varepsilon}),\qquad
f(\varepsilon)=e^{-\varepsilon}\exp(-e^{-\varepsilon}).
$$

其均值是欧拉常数 $\gamma\approx 0.577$（方差为 $\pi^2/6$），因此 $E[U_i]=V_i+\gamma$。全体选项共享这个常数时，均值之差等于 $V$ 之差。选择概率只依赖 $V$ 的差，所以这个 $\gamma$ 永远不会出现在最终公式里。

### 写出「$i$ 是最大值」的概率

假设各 $\varepsilon_j$ 独立同分布。先固定 $\varepsilon_i=\varepsilon$：此时「$i$ 胜过 $j$」等价于 $V_j+\varepsilon_j<V_i+\varepsilon$，即 $\varepsilon_j<V_i-V_j+\varepsilon$，概率为 $F(V_i-V_j+\varepsilon)$。由独立性，$i$ 同时胜过所有其他项的条件概率是这些 CDF 的乘积；再对 $\varepsilon$ 的密度积分去掉条件：

$$
P(y=i)
=
\int_{-\infty}^{\infty}
\prod_{j\neq i} F(V_i-V_j+\varepsilon)\, f(\varepsilon)\,d\varepsilon.
$$

### 代入 CDF，合并指数

把 $F(t)=\exp(-e^{-t})$ 代入，每一项为 $\exp\big(-e^{-\varepsilon}e^{-V_i}e^{V_j}\big)$，乘积变成指数上的求和：

$$
\prod_{j\neq i} F(V_i-V_j+\varepsilon)
=
\exp\Big(-e^{-\varepsilon}e^{-V_i}\sum_{j\neq i}e^{V_j}\Big).
$$

再乘 $f(\varepsilon)=e^{-\varepsilon}\exp(-e^{-\varepsilon})$。密度里的 $\exp(-e^{-\varepsilon})$ 可以并入上式的指数：指数里多出的 $-e^{-\varepsilon}\cdot 1$ 中，把 $1$ 写成 $e^{-V_i}e^{V_i}$，它恰好补上求和里缺的 $k=i$ 那一项，得到

$$
e^{-\varepsilon}
\exp\Big(-e^{-\varepsilon}e^{-V_i}\sum_{k}e^{V_k}\Big).
$$

### 换元积分

令 $\lambda=e^{-V_i}\sum_k e^{V_k}$（与 $\varepsilon$ 无关的正常数），换元 $u=e^{-\varepsilon}$，$du=-e^{-\varepsilon}d\varepsilon$。$\varepsilon$ 从 $-\infty$ 到 $\infty$ 时积分从 $u=\infty$ 到 $0$，负号把积分上下限翻回来：

$$
\int_{-\infty}^{\infty} e^{-\varepsilon}\exp(-\lambda e^{-\varepsilon})\,d\varepsilon
=\int_0^{\infty} e^{-\lambda u}\,du=\frac1\lambda .
$$

结果是 $1/\lambda$。于是

$$
P(y=i)
=
\frac{e^{V_i}}{\sum_{k\in C}e^{V_k}}.
$$

这就是多项 logit（multinomial logit, MNL），机器学习里叫 softmax。$|C|=2$ 时分母两项，$P(y=i)=e^{V_i}/(e^{V_i}+e^{V_j})=\sigma(V_i-V_j)$，即 Bradley–Terry。可以看到，前一章的 BT 不是凭空选的 sigmoid，而是「独立 Gumbel 噪声 + 取最大」的必然结果。

### 对数似然的梯度

设第 $n$ 次观测的菜单为 $C_n$，选项 $j$ 的特征为 $x_{n,j}\in\mathbb{R}^d$，所有选项共享参数 $w\in\mathbb{R}^d$，$V_i=w^{\top}x_i$。对数似然 $\ell=\sum_n\log P(y_n\mid x_n)$，其中 $\log P(y_n)=w^{\top}x_{n,y_n}-\log\sum_{j}e^{w^{\top}x_{n,j}}$。第一项对 $w$ 求导得 $x_{n,y_n}$；第二项（log-sum-exp）求导得 softmax 加权的特征平均。于是

$$
\nabla_w\ell
=
\sum_n
\Big(
x_{n,y_n}
-
\sum_{j\in C_n}P(j\mid x_n)\,x_{n,j}
\Big).
$$

括号内是选中项的特征减去模型下的期望特征。预测已经把概率集中在 $y_n$ 上时，期望特征几乎就是 $x_{n,y_n}$，该项梯度接近 0；预测错得越离谱，这一项把 $w$ 往「更偏好选中项特征」的方向推得越用力。在极大似然点，全体样本上「观测到的特征平均 = 模型预测的特征平均」，这是指数族矩匹配的一个实例。

<br>

## Luce 选择公理与无关方案独立性（IIA）

上一节从噪声分布出发得到 softmax。另一条路线完全不谈噪声，只对选择概率本身提出一条行为公理，结果得到同一个模型。这说明 softmax 不只是计算上的便利，它对应一条可检验的行为假设。

### Luce 公理

Luce 选择公理（Luce's choice axiom，1959）：对任意含 $i,j$ 的菜单 $S$，

$$
\frac{P(i\mid S)}{P(j\mid S)}
=
\frac{P(i\mid\{i,j\})}{P(j\mid\{i,j\})},
$$

只要这些概率都为正。直白地说：$i$ 与 $j$ 的「相对吸引力」是这两项之间的事，菜单上还摆了什么不影响它。

在有限菜单上，该公理等价于存在分数 $s_i$ 使选择概率为 softmax：取 $e^{s_i}\propto P(i\mid\{i,k\})/P(k\mid\{i,k\})$（以某个参照项 $k$ 为基准），公理保证这些比值在任何菜单里保持一致，归一化后就是 $e^{s_i}/\sum_{j\in S}e^{s_j}$；反过来，softmax 的比值 $e^{s_i-s_j}$ 显然与菜单无关。因此多项 logit、Luce 模型与独立 Gumbel 随机效用，在正概率的有限选择上是同一族分布的三种写法。（反方向「满足 Luce 公理的独立同分布随机效用模型必须是 Gumbel」需要额外条件，经典结论见 Yellott 1977。）

### IIA

无关方案独立性（independence of irrelevant alternatives, IIA）是上式的直接推论：两项的概率比不随菜单里其余项改变。加入或删去第三个选项，$i$ 与 $j$ 的份额按同一比例放缩。

需要注意区分两种 IIA。这里的 IIA 是个人选择概率的性质。后文社会选择部分 Arrow 定理中的 IIA 是社会排序只依赖每个人对这两项的名次，两者不是同一条公理；logit 满足前者，并不因此给出一种满足后者的投票规则。

<br>

## IIA 的反例：红巴士 / 蓝巴士与嵌套 logit

IIA 让模型极其简洁（每个选项一个参数），但它也意味着模型无法表达「两个选项很像、互为替代品」。下面的经典例子说明问题，并给出一种修正。

### 红巴士 / 蓝巴士

标准反例是交通方式。汽车与红巴士的系统效用都是 0。菜单只有这两项时，softmax 给出 $P(\mathrm{car})=1/2$。

加入一辆系统效用同样为 0 的蓝巴士，它除颜色外与红巴士完全相同。三项 logit 给出每一项 $1/3$，汽车概率从 $1/2$ 降到 $1/3$。汽车与红巴士的概率比仍是 1，IIA 成立。但行为上两辆巴士是替代品：坐巴士的人不在乎颜色，加入蓝巴士应当主要分走红巴士的份额，合理的结果是汽车约 $1/2$、两辆巴士各约 $1/4$，汽车概率应留在 $1/2$ 附近。独立 Gumbel 做不到这一点，因为独立噪声意味着三张「独立的票」，相似选项会被重复计数。

在偏好学习里，同样的问题出现在「同一回答的两个几乎相同的改写」被放进同一菜单时：多项 logit 会把它们当作两个独立的竞争者，压低其他回答的概率。

### 嵌套 logit

嵌套 logit（nested logit）把相关选项放进同一巢（nest），允许巢内噪声相关。选择被分解为两步：先选巢，再在巢内选项。巢 $B_k$ 内

$$
P(i\mid B_k)
=
\frac{e^{V_i/\lambda_k}}{\sum_{j\in B_k}e^{V_j/\lambda_k}},
\qquad
\mathrm{IV}_k
=
\log\sum_{j\in B_k}e^{V_j/\lambda_k}.
$$

$\mathrm{IV}_k$ 称为包容值（inclusive value，也叫 log-sum），衡量这个巢整体有多大吸引力。巢本身的系统效用为 $\lambda_k\mathrm{IV}_k$，选巢的概率是对各巢的 $e^{\lambda_k\mathrm{IV}_k}$ 做 softmax，最终 $P(i)=P(B_k)\,P(i\mid B_k)$。参数 $\lambda_k$ 粗略地衡量巢内噪声的「独立程度」：$\lambda_k$ 越大越独立，$1-\lambda_k$ 常被当作巢内相关程度的度量（严格地说，巢内两项效用噪声的相关系数为 $1-\lambda_k^2$）。与随机效用相容时 $\lambda_k\in(0,1]$。$\lambda_k=1$ 退回普通 logit；$\lambda_k\to 0$ 表示巢内完全相关，巢内选项如同同一个选项。

### 数值演示

汽车单独成巢，$\lambda=1$，$V=0$，包容值 $\mathrm{IV}\_{\mathrm{car}}=0$，巢权重 $e^0=1$。红、蓝巴士成巢，$V=0$。

$\lambda=1$ 时 $\mathrm{IV}\_{\mathrm{bus}}=\log 2$（即 $\log(e^0+e^0)$），巢效用为 $\log 2$，巢权重为 2，

$$
P(\mathrm{car})=\frac{1}{1+2}=\frac13,\qquad P(\mathrm{red})=\frac13.
$$

这与普通三项 logit 完全相同，验证了 $\lambda=1$ 时的退化。

$\lambda=1/2$ 时 $\mathrm{IV}\_{\mathrm{bus}}=\log(e^{0/0.5}+e^{0/0.5})=\log 2$（$V=0$ 时除以 $\lambda$ 不改变指数），巢效用为 $\tfrac12\log 2$，巢权重为 $e^{\frac12\log 2}=\sqrt{2}$，

$$
P(\mathrm{car})=\frac{1}{1+\sqrt{2}}=\sqrt{2}-1\approx 0.4142,
$$

两辆巴士在巢内平分剩余的份额：

$$
P(\mathrm{red})=\frac12\big(1-P(\mathrm{car})\big)\approx 0.2929.
$$

$\lambda$ 越小，汽车概率越靠近 $1/2$：$\lambda\to 0$ 时巢权重 $e^{\lambda\log 2}\to 1$，汽车与巴士巢各占一半。巢内相关正是在削弱「每辆巴士各算一个独立选项」。代价是需要事先指定哪些选项同巢；巢结构未知时，下一章的混合 logit 提供了更灵活的替代模式。

<br>

## 正态噪声：Thurstone 模型与 probit

Gumbel 的好处是闭式，但它本身没有特别的心理学依据。更早的 Thurstone 比较判断律（law of comparative judgment，1927）假设感知的「判别过程」带正态噪声，这是另一种自然的选择，代价是失去闭式。

Thurstone 用正态噪声。Case V（最简单的情形）：$\varepsilon_i$ 独立，$N(0,\sigma^2)$，所有选项方差相同。两个独立正态之差仍是正态，差 $\varepsilon_i-\varepsilon_j$ 服从 $N(0,2\sigma^2)$，于是 $P(U_i>U_j)=P(\varepsilon_j-\varepsilon_i<V_i-V_j)$ 为

$$
P(i\succ j)
=
\Phi\Big(\frac{V_i-V_j}{\sigma\sqrt{2}}\Big).
$$

其中 $\Phi$ 是标准正态 CDF。$V$ 与 $\sigma$ 不能分别识别（只有比值 $(V_i-V_j)/\sigma$ 出现），若把差的方差定成 1，公式写成 $\Phi(V_i-V_j)$。

二元情形有闭式。$|C|>2$ 时，多元 probit（multinomial probit）的 $P(y=i)$ 是「$i$ 与其他每一项的效用差都为正」的概率，即一个 $|C|-1$ 维正态在正象限（矩形）上的积分，没有 softmax 那样的闭式，用模拟（如 GHK 模拟器）或数值积分。作为补偿，probit 允许噪声有任意协方差，自然地表达红巴士 / 蓝巴士式的相关，不受 IIA 约束。

logit 与 probit 在二元、概率远离 0 和 1 时曲线接近（把 logit 的效用尺度除以约 1.6–1.8 后，两条曲线在中间区域几乎重合）。尾部不同：logit 尾更重，同样大的效用差下，概率更不容易贴近 0 或 1。因此在大量「一边倒」的比较数据上，两者对极端效用差的估计会有可见差别。

<br>

## 二元情形：BT 的第二种推导

$|C|=2$ 时不必做 $|C|-1$ 维积分，也不必套用上面的多项公式，可以直接看噪声之差的分布。两个独立标准 Gumbel 的差服从标准 logistic 分布：密度 $e^{-z}/(1+e^{-z})^2$，分布函数就是 $\sigma(z)=1/(1+e^{-z})$。于是

$$
P(U_i>U_j)
=
P(\varepsilon_j-\varepsilon_i<V_i-V_j)
=
\sigma(V_i-V_j).
$$

这是 BT 的第二份来源：第一份是多项积分在 $|C|=2$ 时的特例，第二份是噪声之差的分布函数。两者一致，并不意外。也由此可见，二元比较只依赖噪声**差**的分布：任何差为 logistic 的噪声都给出 BT。正态噪声的差仍是正态，所以 Thurstone 得到的是 $\Phi$ 而不是 $\sigma$。

数值：$\sigma=1$、$V_i-V_j=1$。logit 概率 $\sigma(1)=0.7311$。probit 取每个噪声方差为 1，把差的标准差设为 $\sqrt{2}$，临界点是 $1/\sqrt{2}\approx 0.7071$，

$$
\Phi(1/\sqrt{2})\approx 0.7602.
$$

同样的效用差，probit 比 logit 更接近 1。需要注意，这一比较部分来自尺度不同：标准 logistic 差的标准差是 $\pi/\sqrt{3}\approx 1.81$，大于这里正态差的 $\sqrt2\approx1.41$，噪声更大，概率自然更接近 1/2。效用差继续加大时，logit 更慢贴近 1，因为 logistic 尾比正态重。实际比较两种模型时，应当先对齐尺度，再比较形状。

<br>

## 排序数据：Plackett–Luce 模型

很多偏好数据不是「选一个」，而是「排一列」：标注者把 $m$ 个模型回答从好到坏排好。Plackett–Luce 模型把排序看成一串逐位的选择：先从全部 $m$ 项中按 softmax 选出第一名，再从剩下的项中选出第二名，依此类推。由 Gumbel 随机效用推导，这恰好是「按 $U$ 从大到小排序」的分布。

全序 $y_1\succ y_2\succ\cdots\succ y_m$（$y_k$ 为第 $k$ 名的物品，$s_i$ 为物品分数）的概率是逐位 softmax 的乘积

$$
P(y)
=
\prod_{k=1}^{m-1}
\frac{e^{s_{y_k}}}{\sum_{t=k}^{m}e^{s_{y_t}}}.
$$

第 $k$ 个因子的分母只对「还没被排上的物品」$y_k,\dots,y_m$ 求和。最后一名只剩自己，因子为 1，所以连乘只到 $m-1$。$m=2$ 时就是 BT。

数值：$(e^{s_A},e^{s_B},e^{s_C})=(3,2,1)$。$A\succ B\succ C$：第一位从总权重 6 中选 $A$，概率 $3/6$；第二位从剩下的 $B,C$（权重和 3）中选 $B$，概率 $2/3$；最后一位因子为 1。

$$
P(A\succ B\succ C)
=
\frac{3}{6}\cdot\frac{2}{3}\cdot 1
=
\frac13,
\qquad
P(A\succ C\succ B)
=
\frac{3}{6}\cdot\frac{1}{3}
=
\frac16.
$$

同一分数下的成对边缘（只看两项的 BT 概率）：

$$
P(A\succ B)=\frac{3}{5},\qquad
P(A\succ C)=\frac{3}{4},\qquad
P(B\succ C)=\frac{2}{3}.
$$

菜单 $\{A,B,C\}$ 上 $P(A)/P(B)=(3/6)/(2/6)=3/2$，成对菜单上 $P(A\succ B)/P(B\succ A)=3/2$（即 $(3/5)/(2/5)$）。Luce 的概率比在两个菜单上相同，这正是 IIA 在排序模型中的体现。实际意义在于：一条长度为 $m$ 的排序可以拆成 $m-1$ 次独立的「选一个」，对数似然是逐位 log-softmax 之和，估计方法与多项 logit 完全相同；只看前 $k$ 名（top-$k$）时，只取前 $k$ 个因子即可。

有序列表的另一种模型是 ordered logit（有序 logit）：潜在 $U=w^{\top}x+\varepsilon$，用切点把实线分成有序等级。它拟合的是等级，不是任意两两对决。等级有天然顺序（如 1–5 星评分）时用它；任意物品对决用 BT 或 Plackett–Luce。下一节展开有序 logit。

<br>

## 有序等级：有序 logit

当反馈是「给一个回答打 1–5 分」而不是「两个回答比一比」时，数据是单个对象的有序等级。直接把等级当连续值回归会假设相邻等级间距相等；当作无序类别又丢掉了顺序。有序 logit（ordered logit）的想法是：存在一个连续的潜在效用，标注者看它落在哪个区间就报哪个等级。

有序 logit 用切点（cutpoints）$\tau_1<\tau_2<\cdots$ 把一条潜在直线切成等级：$U=w^{\top}x+\varepsilon$ 落在 $(\tau_{j-1},\tau_j]$ 时报告等级 $j$。$\varepsilon$ 为 logistic 时，$P(y\le j)=P(w^{\top}x+\varepsilon\le\tau_j)$，即

$$
P(y\le j\mid x)=\sigma(\tau_j-w^{\top}x),
\qquad
P(y=j)=P(y\le j)-P(y\le j-1).
$$

最低等级的下累积为 0（相当于 $\tau=-\infty$），最高等级的上累积为 1（相当于 $\tau=+\infty$）。切点本身要单调，否则概率为负；实践中常把切点参数化为第一个切点加上一串正增量（如 $\exp$ 或 softplus 后累加）。

数值：$w^{\top}x=0$，切点 $\tau=(-1,\ 1)$，三等。例中等级从 0 编号，于是 $P(y\le 0)=\sigma(\tau_1-w^{\top}x)$，$P(y\le 1)=\sigma(\tau_2-w^{\top}x)$。

$$
\begin{aligned}
P(y=0)&=\sigma(-1)=0.2689,\\
P(y=1)&=\sigma(1)-\sigma(-1)=0.4621,\\
P(y=2)&=1-\sigma(1)=0.2689.
\end{aligned}
$$

三个概率之和为 1，且因为切点关于 0 对称，两端等级概率相同。

$w^{\top}x$ 增加 1，相当于切点都左移 1（$\sigma(\tau_j-(w^{\top}x+1))=\sigma((\tau_j-1)-w^{\top}x)$），质量从低等级流向高等级。

需要注意的坑：这里的 $y$ 是同一个人的等级，不是两个物品谁赢。把两件物品的评分相减再送进 BT，等于另外假设了评分差的含义（例如 5 分与 4 分之差等同于 2 分与 1 分之差）；有序 logit 不自动提供这个差。此外，不同标注者的切点可能不同（有人习惯打高分），这是下一章「个人截距」要处理的异质性。

> [!NOTE]+ 小结
> - 随机效用 $U_i=V_i+\varepsilon_i$ 加「取最大」：只有 $V$ 的差可识别，尺度由噪声方差钉住。
> - 独立 Gumbel 噪声 → softmax（多项 logit）；两项时 → Bradley–Terry，等价于 logistic 噪声差。
> - Luce 公理 ⇔ softmax ⇒ IIA；IIA 无法处理相似选项（红巴士 / 蓝巴士），嵌套 logit 用巢内相关修正。
> - 正态噪声 → probit：二元有闭式，多项需要高维积分，logit 尾更重。
> - 排序用 Plackett–Luce（逐位 softmax），有序等级用有序 logit（切点）。

<br>


# 因子模型：用特征与潜在因子表示效用

Bradley–Terry 一章的 BT 与上一章的多项 logit 都把每个物品的系统效用当成一个自由截距 $s_i$。这有两个根本限制。第一，自由截距只能内插已经比过的物品：一个新的回答、一个新的商品从未出现在数据里，它的 $s$ 无从谈起。第二，所有人共享同一组 $s$，无法表达「有人偏爱简洁、有人偏爱详尽」这类口味差异。要预测新菜单，或允许不同人的权衡不同，$V$ 得写成特征或潜在因子的函数。

核心想法：把效用分解为「可共享的参数 × 可观测或可学习的表示」。本章依次介绍条件 logit（显式特征）、低秩因子模型（学到的嵌入）、混合 logit 与潜类模型（人与人之间的系数异质）、Rasch 模型（最简单的双因子特例），然后讨论哪些参数能被数据识别，最后用一个手算的牛顿法例子把估计过程走一遍。下一章在此基础上讨论如何估计和评价这些预测模型。

## 条件 logit：用特征写效用

McFadden 条件 logit（conditional logit）：选项 $j$ 在上下文 $n$ 的系统效用 $V_{nj}=w^{\top}x_{nj}$，选择概率为 softmax：

$$
P(y_n=j\mid C_n)=\frac{e^{w^{\top}x_{nj}}}{\sum_{k\in C_n}e^{w^{\top}x_{nk}}}.
$$

$x_{nj}\in\mathbb{R}^d$ 是该选项的属性（价格、长度、是否包含代码），可以随决策 $n$ 变化（同一回答在不同问题下的长度不同）。$w\in\mathbb{R}^d$ 在选项之间共享，所以新选项只要能写出 $x$，就有 $V$。例如 $w$ 中「长度」的系数为正，就意味着模型认为标注者整体偏好更长的回答。

物品截距是特征的特例：第 $i$ 个物品的 $x$ 为独热向量。此时 $w^{\top}x_i=w_i$，模型退回 Bradley–Terry 一章的自由分数 $\{s_i\}$，新物品没有对应坐标。

## 低秩因子模型：学到的嵌入

显式特征需要人工设计。低秩因子把人 $u$ 与物品 $i$ 都嵌入同一 $d$ 维空间，让数据自己学出表示：

$$
V_{ui}
=
\alpha_u+\beta_i+p_u^{\top}q_i.
$$

其中 $\alpha_u\in\mathbb{R}$ 是这个人「容易给出高分」的倾向，$\beta_i\in\mathbb{R}$ 是物品的流行程度（全体用户共有的吸引力），$p_u,q_i\in\mathbb{R}^d$ 分别是人与物品的嵌入，内积 $p_u^{\top}q_i$ 是匹配程度。这与推荐系统中的矩阵分解、双塔模型是同一结构。成对概率仍是 $\sigma(V_{ui}-V_{uj})$；注意在同一个人的成对比较中 $\alpha_u$ 被消掉了，它只在等级、评分类数据（或跨人比较）中才可识别。

$d$ 远小于人数与物品数时，参数量从 $O(|\mathcal{U}|\,|\mathcal{I}|)$（每个人对每个物品一个自由效用）降到 $O(d(|\mathcal{U}|+|\mathcal{I}|))$。低秩正是泛化的来源：未被这个人比较过、但被其他人比较过的物品，可以通过 $q_i$ 得到预测。$p_u$ 完全没有数据的新用户则还需要 $\alpha_u,p_u$ 的先验或内容特征（冷启动问题）。

<br>

## 偏好异质性：混合 logit 与潜类模型

低秩模型给每个人一组参数；另一种思路是把「人」看成从某个总体中随机抽出的，系数服从一个分布。这样既能描述人群中口味的分布，又能顺带打破 IIA。

### 混合 logit

混合 logit（mixed logit，又称随机系数 logit）把系数本身当成随机的：每个决策者有自己的 $\beta$，给定 $\beta$ 时是普通 logit，再对 $\beta$ 的分布积分：

$$
P_n(i)
=
\int
\frac{e^{\beta^{\top}x_{ni}}}{\sum_j e^{\beta^{\top}x_{nj}}}
f(\beta\mid\theta)\,d\beta.
$$

$f$ 常取正态，均值与协方差由 $\theta$ 参数化。积分一般没有闭式，需要模拟（见下一章）。

为什么它能打破 IIA？人与人的 $w$ 不同，IIA 只在每个人内部成立，在总体概率上不再成立：总体是许多 logit 的混合，两项概率比是不同人的比值的加权平均，权重随菜单变化。因此替代模式可以比单一 logit 丰富：在红巴士 / 蓝巴士例子中，偏爱巴士的那群人会在两辆巴士之间分流，偏爱汽车的人基本不受影响。McFadden 与 Train（2000）证明，在温和条件下混合 logit 可以以任意精度逼近任意基于随机效用的选择概率。这是它作为因子模型的理论地位：它是随机效用模型族的「万能逼近器」。

### 离散潜类模型

离散潜类（latent class）模型是混合 logit 的有限混合：把连续分布 $f(\beta)$ 换成 $C$ 个离散的点，每个点是一类人：

$$
P(i\mid x)
=
\sum_{c=1}^{C}\pi_c\,
\mathrm{softmax}(w_c^{\top}x)_i,
\qquad
\pi_c>0,\ \sum_c\pi_c=1.
$$

$\pi_c$ 是第 $c$ 类在总体中的比例，$w_c$ 是该类的系数。类 $c$ 内部仍是 IIA；类之间的 $\pi_c$ 与 $w_c$ 不同，总体可以出现红巴士 / 蓝巴士式的替代。类成员身份是潜变量：数据只记录选择，不记录每个人属于哪一类，因此估计时要用 EM（见下一章）。潜类模型的好处是可解释：每一类可以读成一种「标注者人格」（如「偏好安全」型与「偏好有用」型）。

<br>

## Rasch 模型：最简单的双因子 logit

Rasch 模型来自心理测量与项目反应理论（item response theory, IRT），本是用来分析考试作答的；它恰好是 BT 的一个变体，可以帮助理解「人」与「物品」两组因子如何同时被估计。

Rasch 模型是只含两个一维因子的 logit：人 $p$ 有能力 $\theta_p\in\mathbb{R}$，题 $i$ 有难度 $b_i\in\mathbb{R}$，$X_{pi}\in\{0,1\}$ 表示人 $p$ 是否答对题 $i$：

$$
P(X_{pi}=1)=\sigma(\theta_p-b_i).
$$

把「人赢过这道题」看成一次比较，它就是 BT，$s$ 的坐标分成两组（人一组、题一组），比较只发生在组间。与 BT 一样有平移不可识别性：能力与难度同时加一个常数，概率不变。常用的钉扎是 $\sum_i b_i=0$，或令某一题 $b_1=0$。

数值：$\theta=0$，$b_1=-1$（容易题），$b_2=1$（难题）。

$$
P(X_{1}=1)=\sigma(1)=0.7311,\qquad
P(X_{2}=1)=\sigma(-1)=0.2689.
$$

能力比难度高 1 个单位，答对概率约 73%；低 1 个单位，约 27%，两者关于 1/2 对称。

人与题都换成向量时，$V_{pi}=p_p^{\top}q_i$，就是上一节的低秩因子（对应多维 IRT）；Rasch 是秩为 1 且人、题各只有一个标量的情形（加上难度作为物品截距）。

<br>

## 参数识别：数据能告诉什么

引入特征与因子后，模型的参数变多，随之而来的问题是：哪些参数能被数据唯一确定？如果两组不同参数给出完全相同的选择概率，数据就无法区分它们，报告这些参数的绝对值没有意义。根源在于上一章已经指出的事实：softmax 只依赖效用差。

**特征的平移。** 条件 logit 里，全体选项的特征向量都加上同一个向量 $c$，则每个 $V$ 增加 $w^{\top}c$，softmax 不变。数据能识别的是选项之间的特征差，不是特征的绝对水平。同理，一个在同一菜单内对所有选项取相同值的特征（例如提问者的年龄），其系数完全不可识别，除非让它与选项属性交互。

**物品截距。** 物品截距（alternative-specific constants）把这种不可识别写进设计矩阵：第 $j$ 个选项有自己的常数项 $\alpha_j$。这些常数同时加任意数都不改变概率，必须删掉一个常数，或加上 $\sum_j \alpha_j=0$。删掉的那一项是参照选项，其他 $\alpha_j$ 的含义是「相对参照项的剩余效用」，即特征没有解释的那部分偏好。

**缺乏菜单内变化的特征。** 只观测价格、且价格的变化只来自每个菜单里所有选项的价格一起上涨同一金额时，价格系数也不被识别：同一菜单内选项之间的价格差分为 0，softmax 看不到它。要识别价格系数，数据中必须存在同一菜单内价格不同的选项。

实际含义：报表上的 $w$ 必须附带参照项和特征的差分方式（以及噪声尺度的标准化），否则两个实验的系数不能比较。

<br>

## 数值例子：用牛顿法估计一维 logit

为了看清估计过程的每一步，下面把条件 logit 收成一个二元逻辑回归，便于把牛顿步算完。特征差固定为 $x=1$，四次独立比较里优势方胜 3 次、负 1 次。模型 $P=\sigma(w)$。

负对数似然 $J(w)=-\sum_n[y_n\log p_n+(1-y_n)\log(1-p_n)]$，其中 $p_n=\sigma(wx_n)$。利用 $\sigma'=\sigma(1-\sigma)$，$J$ 在 $w$ 处的导数与二阶导为

$$
\frac{\partial J}{\partial w}
=
\sum_n(p_n-y_n),
\qquad
\frac{\partial^2 J}{\partial w^2}
=
\sum_n p_n(1-p_n).
$$

（一般情形要乘 $x_n$ 与 $x_n^2$，这里 $x_n=1$。）二阶导恒正，$J$ 是凸的，极小点唯一；牛顿法 $w\leftarrow w-J'(w)/J''(w)$ 在本例中几步即收敛（一般情形下纯牛顿步从远处起步可能越过极小点，实践中常配合步长搜索）。

**第 1 步。** $w=0$ 时 $p=1/2$，$J=4\log 2=2.7726$，梯度 $3(1/2-1)+(1/2-0)=-1$，二阶导 $4\cdot 1/4=1$。牛顿步 $w\leftarrow 0-(-1)/1=1$。

**第 2 步。** $w=1$ 时 $p=\sigma(1)=0.7311$，梯度 $4p-3=-0.0758$，二阶导 $4p(1-p)=0.7864$，

$$
w\leftarrow 1-\frac{-0.0758}{0.7864}=1.0963.
$$

**收敛。** 再两步到 $w=\log 3=1.0986$（牛顿法在最优点附近二次收敛，每步有效数字约翻倍），因为四次独立同分布 Bernoulli 的极大似然就是经验胜率 $3/4$，而 $\sigma(\log 3)=3/4$。负对数似然从 $2.7726$ 降到 $2.2493$（$=-3\log\tfrac34-\log\tfrac14$）。

这个例子之所以有一元闭式，是因为所有比较的特征差相同。若两次比较的特征差分别是 $1$ 和 $2$，经验胜率不再对应单一的 $\sigma(w)$，就要用「特征减期望特征」的向量梯度迭代求解，没有这个一元闭式。

> [!NOTE]+ 小结
> - 条件 logit $V_{nj}=w^{\top}x_{nj}$ 让新选项只要有特征就能预测；物品截距是独热特征的特例。
> - 低秩因子 $\alpha_u+\beta_i+p_u^{\top}q_i$ 把参数量从 $O(|\mathcal{U}||\mathcal{I}|)$ 降到 $O(d(|\mathcal{U}|+|\mathcal{I}|))$，新用户需要先验或内容特征。
> - 混合 logit 与潜类模型表达人与人的异质，总体上不再满足 IIA；混合 logit 可逼近任意随机效用模型。
> - Rasch 是人、题各一个标量的 BT；识别只针对差，需要钉扎与参照项。

<br>


# 选择预测：估计、评价与模拟

前两章讨论的是模型「长什么样」；本章关心如何从数据中把参数估出来、如何评价预测好坏、以及在没有闭式的混合模型里怎样用模拟计算似然。

先区分两种建模立场。归约式（reduced form）直接回归 $P(y\mid x,C)$，把它当成一个普通的分类问题。结构式（structural form）先写 $V$ 与 $\varepsilon$，再推出 $P$。独立 Gumbel 加线性 $V$ 时，两者的条件分布都是 softmax，数值上是同一个逻辑回归，在训练分布上的预测完全一样。差别在菜单改变之后：结构式用同一个 $w$ 在新菜单上重算 softmax，新选项只要有特征就能参与；一个只在旧菜单的类别标签上训练、把「选项身份」当固定类别的分类器，没有新选项的类别头。偏好学习常常需要对从未见过的回答打分，这正是结构式视角的价值。

<br>

## 条件 logit 的极大似然估计

条件 logit 的对数似然关于 $w$ 凹（log-softmax 是线性函数减 log-sum-exp，后者凸），所以任何局部极大都是全局极大。梯度是「特征减去期望特征」，Hessian 是负的加权协方差：

$$
\nabla^2_w\ell=-\sum_n \mathrm{Cov}_{P(\cdot\mid x_n)}\big(x_{n,j}\big),
$$

在菜单内特征有变化的方向上严格负定，这些方向对应有信息的特征对比；菜单内无变化的方向 Hessian 为零，对应上一章所说的不可识别方向。牛顿法或 L-BFGS 都适用。标准做法：

1. 用 $w=0$ 起步，此时每个菜单上的概率均匀，梯度等于「选中特征减菜单内特征平均」。
2. 每步计算 softmax 时减去该菜单的最大 $V$，避免溢出（log-sum-exp 技巧，见下方代码）。
3. 若物品截距完全分离（某物品总是赢，MLE 在无穷远），改用 $w$ 的 $L_2$ 惩罚，或停止在验证对数损失最低处。
4. 报告自由参数的差，而不是绝对水平。

### 混合 logit 的模拟极大似然

混合 logit 的似然含一个没有闭式的积分，模拟极大似然（simulated maximum likelihood, SML）用蒙特卡洛平均代替积分：从 $f(\beta\mid\theta)$ 抽 $R$ 个 $\beta^{(r)}$，用

$$
\hat P_n(i)
=
\frac1R\sum_{r=1}^{R}
\mathrm{softmax}(\beta^{(r)\top}x_n)_i
$$

代入对数似然。$\hat P_n$ 是 $P_n$ 的无偏估计，但 $\log\hat P_n$ 不是 $\log P_n$ 的无偏估计（log 是凹函数，由 Jensen 不等式会系统性偏低），所以 $R$ 固定且用独立新抽样时，模拟误差会留在极限里：样本量再大，估计也收敛不到真值。实现上，用同一套随机种子的共同随机数（common random numbers：抽一次标准正态，再通过 $\beta^{(r)}=\mu+L z^{(r)}$ 变换，优化过程中固定 $z^{(r)}$ 不变），目标对 $\theta$ 光滑，优化稳定。$R$ 随样本增大而增大时，模拟误差才不影响一致性。更细的结论见 Train《Discrete Choice Methods with Simulation》第 10.5.1 节：$R$ 固定时 SML 不一致；$R$ 随 $N$ 增长但慢于 $\sqrt N$ 时一致但不渐近正态；$R$ 增长快于 $\sqrt N$ 时一致、渐近正态且有效，与精确 MLE 渐近等价。

<br>

## 潜类模型的 EM 算法

潜类模型是缺了类标签的混合。观测对数似然是「对数里套求和」，没有普通 logit 那么直接的凹性，局部极大存在，结果依赖初始化（实践中多起点重跑）。EM 把类标签当缺失数据：若知道每个样本属于哪一类，问题就分解成 $C$ 个独立的条件 logit；不知道，就先按当前参数算出「属于每一类的后验概率」，再用它加权。

E 步：当前参数下，样本 $n$ 属于类 $c$ 的责任度（responsibility），由贝叶斯公式

$$
\gamma_{nc}
=
\frac{\pi_c\,P(y_n\mid x_n,w_c)}{\sum_{c'}\pi_{c'}\,P(y_n\mid x_n,w_{c'})}.
$$

M 步：$\pi_c=\frac1N\sum_n\gamma_{nc}$。每个 $w_c$ 是以 $\gamma_{nc}$ 为权重的条件 logit 极大似然，本身还要用牛顿法，没有更短的闭式（加权后对数似然仍凹，所以这一子问题是好解的）。EM 每轮保证观测对数似然不降。

数值：两类先验 $\pi=(1/2,1/2)$，菜单两项。类 0 的 logit 差为 $2$，选第一项的概率 $\sigma(2)=0.8808$。类 1 的 logit 差为 $-2$，概率 $0.1192$。观测到选择第一项时，先验相等而相互抵消：

$$
\gamma_{0}
=
\frac{0.8808}{0.8808+0.1192}
=
0.8808.
$$

这一次观测把责任度几乎全部分给「本来就预测第一项」的类。M 步会用 $0.8808$ 与 $0.1192$ 给两个条件 logit 加权。同一个人若有多次选择，应把这些选择的似然连乘后再算一次责任度（同一人的类别不变），单次观测的责任度会被多次观测迅速拉向 0 或 1。

<br>

## 实现：数值稳定的条件 logit 负对数似然与梯度

下面的代码实现上一节的估计目标，可直接交给 L-BFGS 等优化器。`log_softmax` 先减去最大值再求指数，防止 $e^{V}$ 溢出；返回的是负对数似然及其梯度。

```python
import numpy as np


def log_softmax(v):
    v = np.asarray(v, dtype=float)
    shift = np.max(v)
    log_z = shift + np.log(np.sum(np.exp(v - shift)))
    return v - log_z


def conditional_logit_nll_grad(w, menus):
    """menus: (X, y)，X 形状 (m, d)，y 是选中行。w 形状 (d,)。"""
    w = np.asarray(w, dtype=float)
    nll = 0.0
    grad = np.zeros_like(w)
    for X, y in menus:
        log_p = log_softmax(X @ w)
        nll -= log_p[y]
        p = np.exp(log_p)
        expected = p @ X
        grad -= X[y] - expected
    return nll, grad
```

梯度 `X[y] - expected` 与随机效用一章的对数似然梯度公式反号，因为这里最小化的是负对数似然。`expected = p @ X` 就是菜单内的期望特征 $\sum_j P(j)x_j$。原文代码省略了 `import numpy as np`，这里补上以便直接运行；返回的梯度已与数值差分核对一致。

<br>

## 评价预测：恰当评分规则与校准

估出模型之后，需要衡量它的概率预测好不好。偏好数据天然带噪（同一对回答，不同标注者会给出不同答案），所以真正要评价的是概率是否准确，而不仅是「猜对谁赢」。

对数损失（log loss，即负对数似然）是该条件分布的严格恰当评分规则（strictly proper scoring rule）：报告真实概率 $p$ 时期望损失最低，且只有报告真实概率才最低，因此模型没有动机故意夸大或保守。成对情形下还有 Brier 分数

$$
(y-p)^2,
$$

$y\in\{0,1\}$。它同样在 $p$ 等于真实概率时期望最小，但对尾部误报的惩罚轻于对数损失：Brier 有界（最大为 1），对数损失无界。$p=0.01$ 而 $y=1$ 时，对数损失是 $-\log 0.01=4.605$，Brier 是 $0.99^2=0.980$。准确率对这两个预测都记 0，只要阈值是 $1/2$：无论模型错得多自信，准确率都只记一次「错」，看不出信心的差别。

校准曲线（calibration curve）把预测 $p$ 分箱，看箱内经验胜率是否等于箱的平均 $p$。需要注意一个常见误解：BT 在强连通数据上的极大似然，对训练过的那些对是在经验胜率与传递约束之间的折中，不是每对都校准到经验胜率。前文 Bradley–Terry 一章的例子中，$A$ 对 $B$ 的经验胜率是 $0.75$、拟合概率是 $0.698$，差来自传递约束（所有对的概率必须由同一组分数生成），不是优化没做完。

<br>

## 模拟似然的手算例子

为了看清混合 logit 的模拟似然在算什么，以及它能从数据中学到什么、学不到什么，下面用最小的例子手算。

### 单次观测：混合与退化无法区分

混合 logit 的一次模拟似然可以手算。两个等概率的系数 $\beta\in\{-2,\ 2\}$（相当于 $R=2$ 次抽样恰好抽到这两个点，或真分布就是这两点），菜单两项，特征差为 $1$，观测到选第一项。

$$
\begin{aligned}
P(\beta=-2)&=\sigma(-2)=0.1192,\\
P(\beta=2)&=\sigma(2)=0.8808,\\
\hat P&=\tfrac12(0.1192+0.8808)=0.5.
\end{aligned}
$$

这里 $P(\beta=\pm2)$ 指给定该系数时选第一项的概率。对数似然贡献是 $\log 0.5=-\log 2$。若错误地用均值 $\beta=0$ 的单一 logit，概率也是 $1/2$，这一次观测分不出混合与退化。再加一个「特征差为 $1$、选了第二项」的人：混合模型给第二项的概率同样是 $0.5$（对称），单一 logit 在 $\beta=0$ 也是 $0.5$。可以看到，每人只有一次选择时，混合分布对边缘选择概率的影响可能完全被平均掉；要识别异质性，通常需要同一个人的多次选择（面板数据）。

### 估计混合权重

换成估计混合权重。固定两个点 $\beta=\pm 2$，只估计 $\pi=P(\beta=2)$。两人分别选了第一项与第二项。每个人的似然是两类的加权平均，两人相乘：

$$
L(\pi)
=
\big(\pi\cdot 0.8808+(1-\pi)\cdot 0.1192\big)
\big(\pi\cdot 0.1192+(1-\pi)\cdot 0.8808\big).
$$

$\pi=1/2$ 时两个因子都是 $0.5$，$L=0.25$。$\pi=0.9$ 时第一项概率为 $0.9\cdot 0.8808+0.1\cdot 0.1192=0.8046$，第二项概率为 $0.1954$，$L=0.157$。均匀混合更好：记第一个因子为 $a$，则第二个因子为 $1-a$，$L=a(1-a)$ 在 $a=1/2$ 即 $\pi=1/2$ 时最大，这与「一人选第一项、一人选第二项」的对称数据相符。

最后要区分模拟与潜类推断：模拟抽的是 $\beta$，被积掉；人属于哪一类并不出现在报表里，除非用潜类 EM 把责任度算出来。

> [!NOTE]+ 小结
> - 结构式与归约式在训练菜单上数值相同，区别在于能否对新菜单、新选项重算 softmax。
> - 条件 logit 对数似然凹，用牛顿法或 L-BFGS；softmax 要减最大值，完全分离时加 $L_2$ 惩罚或早停。
> - 混合 logit 用模拟极大似然，固定 $R$ 有偏，需共同随机数并让 $R$ 随样本增长；潜类模型用 EM，存在局部极大。
> - 对数损失与 Brier 都是恰当评分规则，前者对自信的错误惩罚更重；BT 拟合不保证每对都校准到经验胜率。

<br>

<br>

# 贝叶斯估计与微调：先验、MAP 与 Laplace 近似

前面的极大似然（MLE）有三种典型的失灵场景。第一种是完全分离：某件物品在数据里从没输过，似然会沿着“把它的分数推向 $+\infty$”的方向一直上升，MLE 不存在有限解。第二种是样本很少：只比较过一两次的物品，MLE 给出的分数虽然有限，却极不稳定。第三种出现在语言模型上：微调时不希望模型离监督微调（SFT）得到的模型太远，否则它会丢掉原有的语言能力，或者钻奖励模型的空子。三种情况的共同解法是引入先验（prior）：它把参数“拴”在一个有限、合理的区域里。

核心想法一句话：**极大似然只问“数据最支持哪个参数”，贝叶斯估计再加一句“参数本来更可能在哪里”**。对 Bradley–Terry 分数而言，先验是一个以 0 为中心的高斯分布；对语言模型而言，先验就是参考模型 $\pi_{\mathrm{ref}}$ 本身。本章先在分数模型上讲 MAP 与 Laplace 近似，再把同一套思路搬到语言模型的 KL 约束微调上，得到 RLHF 与 DPO 都要用到的最优策略闭式解。Laplace 近似给出的后验方差，会在后面的实验设计一章里用来决定下一次该问哪一对。

<br>

## 从极大似然到 MAP：先验的作用

### 对数后验

记数据为 $\mathcal{D}$，参数为 $w$（例如 BT 模型的各物品分数，或特征权重），对数似然为 $\ell(w)=\log p(\mathcal{D}\mid w)$。由贝叶斯公式 $p(w\mid\mathcal{D})=p(\mathcal{D}\mid w)p(w)/p(\mathcal{D})$，取对数得到对数后验

$$
\log p(w\mid\mathcal{D})
=
\ell(w)+\log p(w)+c.
$$

其中 $c=-\log p(\mathcal{D})$ 与 $w$ 无关，优化时可以忽略。后验众数（maximum a posteriori, MAP）估计就是让上式最大的 $w$：它在似然之外多了一项 $\log p(w)$，惩罚那些“先验上不太可能”的参数。

### 高斯先验等价于 $L_2$ 正则

取各向同性高斯先验 $w\sim N(0,\tau^2 I)$，$\tau>0$ 是先验标准差。它的对数密度是 $\log p(w)=-\|w\|^2/(2\tau^2)+\text{常数}$。代入并把“最大化”改写为“最小化负数”，MAP 最小化

$$
-\ell(w)+\frac{1}{2\tau^2}\|w\|^2.
$$

这就是带 $L_2$ 正则（权重衰减）的极大似然，正则强度 $\lambda=1/\tau^2$。两个极端情形帮助理解：

- $\tau\to\infty$ 时先验变得无限平，惩罚项消失，MAP 回到 MLE。
- 完全分离时 MLE 发散（似然沿某个方向单调上升），但只要 $\tau$ 有限，惩罚项 $\|w\|^2/(2\tau^2)$ 随 $\|w\|$ 二次增长，而负对数似然最多下降到 0，因此目标函数是强凸、有下界的，任何有限 $\tau$ 都把解拉回有限值。

实际使用时，$\tau$ 可以看成“认为分数差大概在几个单位以内”的先验判断，也可以在留出比较数据上用对数损失挑选。

### Laplace 近似：把后验配成正态

MAP 只给出一个点，无法反映参数有多不确定。Laplace 近似（Laplace approximation）的做法是把对数后验在 MAP 点 $\hat w$ 处做二阶泰勒展开：

$$
\log p(w\mid\mathcal{D})
\approx
\log p(\hat w\mid\mathcal{D})
+\underbrace{\nabla\log p(\hat w\mid\mathcal{D})^{\top}(w-\hat w)}_{=0\ (\hat w\ \text{是极大点})}
+\frac12(w-\hat w)^{\top}H(w-\hat w).
$$

一阶项因为 $\hat w$ 是极大点而消失，剩下的是关于 $w$ 的二次型，指数化后正好是高斯密度的形状。于是后验近似为 $N(\hat w,\Sigma)$，$\Sigma=(-H)^{-1}$。在高斯先验下，对数后验在 MAP 处的 Hessian 为 $H=\nabla^2\ell-I/\tau^2$，所以协方差

$$
\Sigma
\approx
\big(-\nabla^2\ell(\hat w)+I/\tau^2\big)^{-1}.
$$

$-\nabla^2\ell$ 称为观测信息（observed information），先验贡献的 $I/\tau^2$ 称为先验精度。直观上，信息矩阵越“大”，后验越窄。对 BT 模型来说，某件物品参与的比较次数越少，信息矩阵对应的对角元越小，后验方差越大。这正是下一章（实验设计）的出发点：用这个方差决定还要问哪一对。

需要注意，Laplace 近似假设后验在 MAP 附近近似对称、单峰。数据很少或完全分离时，真实后验可能明显偏斜，此时 Laplace 给出的方差只是粗略参考。

<br>

## 监督微调与 KL 约束下的最优策略

语言模型的“参数估计”同样可以放进先验 + 似然的框架。本节先说明监督微调（supervised fine-tuning, SFT）在优化什么、缺什么，再推导 KL 约束下的奖励最大化问题的闭式解。这个闭式解说明：参考模型扮演先验，奖励扮演似然，最优策略就是“后验”。后面的 RLHF 与 DPO 两章都建立在这个结论上。

### SFT：在示范上做极大似然

SFT 的数据是示范对 $(x,y)$：$x$ 是提示（prompt），$y=(y_1,\dots,y_T)$ 是人写的或挑选出的回答，$y_t$ 是第 $t$ 个词元（token），$y_{<t}$ 是它之前的所有词元。语言模型 $\pi_\theta$ 是自回归的，序列概率按链式法则分解为

$$
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}),
$$

SFT 最大化该和，等价于最小化下一词元交叉熵（next-token cross-entropy）。它是在示范数据上做极大似然，把示范 $(x,y)$ 的条件概率拉高，学习目标是模仿示范的动作，不使用成对偏好。

从选择模型的角度看，示范可以理解成“完全理性”的选择：在 Boltzmann 理性模型 $p(y)\propto\exp(r(y)/\beta)$（$\beta$ 为温度）里，示范对应温度 $\beta\to 0$、也就是理性系数 $1/\beta\to\infty$ 的最优行为，人总是选出奖励最高的回答。原文写作“$\beta\to\infty$ 的最优行为”，那是有界理性一章 $P\propto e^{\beta Q}$ 的写法（$\beta$ 为理性系数，$\beta=\infty$ 即完全理性），两种说法含义相同；本节紧接着的 KL 约束问题里 $\beta$ 是温度（$e^{r/\beta}$），为避免混淆，这里按温度写成 $\beta\to 0$。

这种理解也暴露了 SFT 的局限：示范里的次优句会被同样拉高，因为 SFT 没有“败者”项。它只知道“这句话出现了”，不知道“哪句话比哪句话差”。

### KL 约束的奖励最大化

假设已经有一个奖励函数 $r(x,y)$（例如从成对偏好里拟合出来的），希望策略 $\pi$ 多产生高奖励的回答，同时不要离参考模型 $\pi_{\mathrm{ref}}$（通常就是 SFT 模型）太远。把“不要太远”量化为 KL 散度，得到 KL 约束的奖励最大化

$$
\max_{\pi}
\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}
[r(x,y)]
-
\beta\,
\mathbb{E}_{x}
\big[
D_{\mathrm{KL}}(\pi(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x))
\big]
$$

其中 $\beta>0$ 控制 KL 惩罚的强度。这个问题在策略类不受限制时（即 $\pi(\cdot\mid x)$ 可以是任意分布）有闭式最优解

$$
\pi^\star(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\big(r(x,y)/\beta\big),
$$

$$
Z(x)
=
\sum_{y}
\pi_{\mathrm{ref}}(y\mid x)
\exp\big(r(x,y)/\beta\big).
$$

$Z(x)$ 称为配分函数（partition function），只依赖提示 $x$。注意对语言模型而言，求和遍历所有可能的回答字符串，数量是天文数字，所以 $Z(x)$ 实际上算不出来。这一点决定了后面 RLHF 要用强化学习近似求解，而 DPO 要设法把 $Z(x)$ 消掉。

### 最优策略的推导

**方法一：拉格朗日乘子。** 目标对不同的 $x$ 是可分的，所以可以对每个 $x$ 单独求解。固定 $x$ 后省略它，记 $\pi(y)=\pi(y\mid x)$，最大化 $\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log(\pi(y)/\pi_{\mathrm{ref}}(y))$，约束 $\sum_y\pi(y)=1$。引入乘子 $\lambda$，对 $\pi(y)$ 求导并令其为 0：KL 项 $\pi\log(\pi/\pi_{\mathrm{ref}})$ 的导数是 $\log(\pi/\pi_{\mathrm{ref}})+1$，所以

$$
r(y)-\beta\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}-\beta-\lambda
=
0,
$$

移项得 $\log(\pi(y)/\pi_{\mathrm{ref}}(y))=(r(y)-\beta-\lambda)/\beta$，即 $\pi(y)=\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}\cdot e^{-(\beta+\lambda)/\beta}$。后一个因子与 $y$ 无关，所以解出 $\pi(y)\propto\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}$。由 $\sum_y\pi(y)=1$ 确定比例常数，归一化常数就是 $Z(x)$。目标函数关于 $\pi$ 是凹的（负熵项严格凹），所以这个驻点就是全局最优。

**方法二：配成 KL（DPO 论文附录 A.1 的写法）。** 把目标除以 $-\beta$ 变成最小化，再利用 $\pi^\star$ 的定义 $\log\pi_{\mathrm{ref}}+r/\beta=\log\pi^\star+\log Z$：

$$
\begin{aligned}
&\min_\pi\ \mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}-\frac{r(y)}{\beta}\Big]\\
=\ &\min_\pi\ \mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y)}{\pi^\star(y)}\Big]-\log Z(x)\\
=\ &\min_\pi\ D_{\mathrm{KL}}(\pi\,\|\,\pi^\star)-\log Z(x).
\end{aligned}
$$

$\log Z(x)$ 与 $\pi$ 无关，而 KL 散度非负、当且仅当 $\pi=\pi^\star$ 时为零（Gibbs 不等式），所以最优解就是 $\pi^\star$。这个写法还顺带给出最优值：最优目标等于 $\beta\log Z(x)$。

### 参考模型即先验，$\beta$ 即温度

把 $\pi^\star\propto\pi_{\mathrm{ref}}\cdot e^{r/\beta}$ 和贝叶斯公式“后验 ∝ 先验 × 似然”对照，可以看到 $\pi_{\mathrm{ref}}$ 扮演回答（completions）上的先验，$e^{r/\beta}$ 扮演似然，$\beta$ 是温度。$r$ 越高的 $y$，后验质量越大。

$\beta$ 的方向需要特别留意：$\beta$ 越大，$r/\beta$ 越平，KL 惩罚越重，先验越强，策略越不敢离开参考模型（$\beta\to\infty$ 时 $\pi^\star\to\pi_{\mathrm{ref}}$）；$\beta$ 越小，奖励的作用越强（$\beta\to 0$ 时 $\pi^\star$ 集中到奖励最大的回答上）。原文此处写成“$\beta$ 越小，先验越强”，与公式方向相反，这里已改正。

### 反解奖励

对 $\pi^\star$ 的表达式取对数并解出 $r$：

$$
r(x,y)
=
\beta\log\frac{\pi^\star(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta\log Z(x).
$$

这说明奖励与最优策略一一对应（差一个只依赖 $x$ 的量）：知道了最优策略，就知道了奖励。$Z(x)$ 只依赖 $x$，在成对比较的奖励差 $r(x,y_w)-r(x,y_l)$ 里会被减掉。这是 DPO 一章的起点。

<br>

## 例子：三物品 BT 的 Laplace 近似

用一个具体数字例子把 Laplace 近似走一遍。沿用前文选择数据一章的三物品 BT 例子：物品 $A,B,C$，钉住 $s_A=0$，比较次数为 $A$ 对 $B$ 3 胜 1 负、$A$ 对 $C$ 4 胜 1 负、$B$ 对 $C$ 3 胜 1 负，极大似然解为 $\hat s=(0,\ -0.8361,\ -1.6723)$。

### 观测信息与后验协方差

在自由坐标 $(s_B,s_C)$ 上，观测信息由每对的 $n_{ij}\hat p_{ij}(1-\hat p_{ij})$ 拼成：$A$–$B$ 与 $B$–$C$ 各为 $4\times 0.2109=0.8437$，$A$–$C$ 为 $5\times 0.1331=0.6656$。$s_B$ 参与 $A$–$B$ 与 $B$–$C$，对角元 $0.8437+0.8437=1.6875$；$s_C$ 参与 $A$–$C$ 与 $B$–$C$，对角元 $0.6656+0.8437=1.5093$；交叉项来自 $B$–$C$，为 $-0.8437$：

$$
\mathcal{I}
=
\begin{pmatrix}
1.6875 & -0.8437 \\
-0.8437 & 1.5093
\end{pmatrix},
\qquad
\log\det\mathcal{I}=0.607.
$$

无信息先验的极限（$\tau\to\infty$）里，后验协方差就是 $\mathcal{I}^{-1}$：

$$
\Sigma
\approx
\begin{pmatrix}
0.822 & 0.460 \\
0.460 & 0.920
\end{pmatrix}.
$$

可以手工核对：$\det\mathcal{I}=1.6875\times1.5093-0.8437^2\approx1.835$，$\log 1.835\approx0.607$；二阶矩阵求逆是“对角互换、非对角变号、除以行列式”，$1.5093/1.835\approx0.822$，$0.8437/1.835\approx0.460$，$1.6875/1.835\approx0.920$。

### 怎样读这个协方差

- $s_C$ 的边际标准差约为 $\sqrt{0.920}=0.959$，$s_B$ 约为 $\sqrt{0.822}=0.907$。两个分数的不确定度都接近 1 个单位，和估计值本身同一量级，说明 13 次比较远不足以把分数钉死。$s_C$ 更不确定，因为 $C$ 与 $A$ 的比较结果更悬殊，提供的信息更少（$p(1-p)$ 更小）。
- 两者正相关（协方差 $0.460>0$）：把 $s_B$ 调高时，为了维持 $B$ 对 $C$ 的胜率，$s_C$ 往往也要调高。数据真正约束的是分数差，所以两个分数会“一起漂”。
- $A$ 被钉在 0，所以这不是三个分数的联合协方差。它是“以 $A$ 为参照”的相对不确定度；换一个参照物品，矩阵会变。

### 加入高斯先验

高斯先验 $N(0,\tau^2)$（只放在自由坐标上）把 $\mathcal{I}$ 的对角加上 $1/\tau^2$。$\tau=1$ 时对角变为 $2.6875$ 与 $2.5093$，方差下降，分离数据也会留在有限值。

但要注意顺序：MAP 本身会离开纯 MLE，协方差必须在 MAP 处重算信息，不能把 MLE 的 $\mathcal{I}$ 与先验精度相加之后还把均值留在 MLE。下面的数字（按本例数据用牛顿法另行计算，不是原课程给出的结果）说明差别有多大：

| 做法 | 均值 $(s_B,s_C)$ | 方差 $(s_B,s_C)$ | 协方差 |
| :--- | :--- | ---: | ---: |
| 错误：MLE 的 $\mathcal{I}$ 加 $I$，均值留在 MLE | $(-0.836,\ -1.672)$ | $(0.416,\ 0.446)$ | $0.140$ |
| 正确：先求 MAP，再在 MAP 处算信息加 $I$ | $(-0.290,\ -0.884)$ | $(0.383,\ 0.376)$ | $0.119$ |

$\tau=1$ 相对于只有 13 次比较的数据是一个相当强的先验，MAP 把分数明显拉向 0；在 MAP 处分数更接近，$p(1-p)$ 更大，信息也随之改变。

<br>

## 词元级的 SFT 损失：一个手算例子

前面说 SFT 的损失是词元对数概率之和、而且没有败者项。用一个两个词元的回答把这两点算清楚。

设模型给两个词元的条件概率为 $1/2$ 与 $1/4$。序列的负对数似然是逐词元相加：

$$
-\log\pi(y\mid x)
=
-\log\frac12-\log\frac14
=
\log 2+\log 4
=
2.0794.
$$

作为参照，均匀猜测两个词元、词表大小为 $4$ 时，每个词元贡献 $\log 4$，合计 $2.7726$。本例比均匀猜测低 $0.693$（即 $\log 2$），全部来自第一个词元：它的概率是 $1/2$ 而不是 $1/4$；第二个词元的概率恰好等于均匀猜测，没有贡献任何改进。

示范损失没有败者项。若第二个词元是人并不喜欢、只是写在示范里的字，SFT 仍把它的 $-\log\pi$ 压低，也就是继续抬高这个字的概率。成对数据才能在同一提示下把败者的对数概率压低、从而推高败者那一侧的损失；这正是 RLHF 与 DPO 引入偏好数据的原因。

> [!NOTE]+ 小结
>
> - 对数后验 = 对数似然 + 对数先验；高斯先验 $N(0,\tau^2I)$ 下 MAP 就是 $L_2$ 正则的 MLE，完全分离时也有有限解。
> - Laplace 近似用 MAP 处的负 Hessian（观测信息 + 先验精度）的逆作为后验协方差；必须在 MAP 而不是 MLE 处计算信息。
> - KL 约束奖励最大化的最优策略是 $\pi^\star\propto\pi_{\mathrm{ref}}e^{r/\beta}$：参考模型是先验，$\beta$ 是温度，$\beta$ 越大越贴近参考模型。
> - 反解得到 $r=\beta\log(\pi^\star/\pi_{\mathrm{ref}})+\beta\log Z(x)$，$Z(x)$ 在成对差中抵消，这是 DPO 的起点。
> - SFT 是示范上的极大似然，没有败者项，会把示范里的次优内容一并拉高。

<br>


# 实验设计：用 Fisher 信息挑选下一次比较

一次人类比较的标注成本很高：要请标注员读两段回答、做出判断，还可能需要专家。预算有限时，问哪一对就成了关键问题。均匀抽样会把预算花在结果已经几乎确定的对上，例如让一个明显很强的模型和一个明显很弱的模型反复对比，人几乎总是选强的那个，模型从中学不到新东西。

核心想法一句话：**下一对按当前参数下的 Fisher 信息来挑**，优先问“模型最拿不准、而且答案最能缩小参数不确定性”的那一对。上一章的 Laplace 近似表明，后验协方差约等于信息矩阵的逆；所以“让后验变窄”就等价于“让信息矩阵变大”。本章先推导单次比较的 Fisher 信息，再介绍把矩阵变成标量的 D/A/E 最优准则，用三物品例子演示序贯选择，最后讨论问题换成“征询评价指标”时的情形。

<br>

## 单次比较的 Fisher 信息

### 模型设定

一次成对观测 $y\in\{0,1\}$（$y=1$ 表示第一个选项胜），胜率 $p=\sigma(w^{\top}x)$，其中 $\sigma$ 是 logistic 函数，$w\in\mathbb{R}^d$ 是参数，$x\in\mathbb{R}^d$ 是两个选项的特征差。物品截距模型是特例：$x=e_i-e_j$，$w$ 就是各物品分数 $s$。

### 得分函数的推导

一次比较的对数密度 $\ell=y\log p+(1-y)\log(1-p)$，$p=\sigma(z)$，$z=w^{\top}x$。对 $w$ 求导用链式法则分三步：

1. $\partial\ell/\partial p=y/p-(1-y)/(1-p)$。
2. 乘 $\partial p/\partial z=p(1-p)$（logistic 函数的导数），通分后得到 $y(1-p)-(1-y)p=y-p$。
3. 乘 $\partial z/\partial w=x$，得分函数（score）是 $(y-p)x$。

得分函数的形式很直观：“观测减预测”乘以特征方向。模型完全猜中时 $y-p$ 接近 0，梯度很小；猜错时梯度大。

### 从得分到 Fisher 信息

在真实参数下 $E[y]=p$，所以 $E[y-p]=0$，得分的期望为 0。Fisher 信息定义为得分的方差（外积的期望）：

$$
E[(y-p)^2]\,xx^{\top}
=
p(1-p)\,xx^{\top},
$$

因为 $\mathrm{Var}(y)=p(1-p)$，而 $p$ 是常数，所以 $\mathrm{Var}(y-p)=p(1-p)$，即 $E[(y-p)^2]=p(1-p)$。这就是该次比较的 Fisher 信息：

$$
\mathcal{I}(w;x)
=
p(1-p)\,x x^{\top}.
$$

logistic 模型里，Fisher 信息与观测信息（负 Hessian）相同，因为 $\nabla^2\ell=-p(1-p)xx^{\top}$ 不含 $y$。独立比较的信息相加：问了 $n$ 次同一对，信息就是 $n$ 倍。

### 什么样的对信息最多

信息由两部分相乘：标量 $p(1-p)$ 与方向 $xx^{\top}$。

- **标量部分。** $p(1-p)$ 在 $p=1/2$ 时取最大值 $1/4$，在 $p=0.9$ 时等于 $0.09$，不到峰值的四成（$0.09/0.25=0.36$）。直观解释：预测已经很有把握的对，标签多半是模型已经知道的那个，梯度 $(y-p)x$ 的方差很小，观测不出什么新东西。
- **方向部分。** 物品截距模型里 $x=e_i-e_j$，$\|x\|^2=2$，信息落在分数差 $s_i-s_j$ 这一个方向上，标量大小与 $p(1-p)$ 成正比。所以最有信息的对是当前分数最接近的对，同时这对必须落在后验仍不确定的方向上。

需要警惕一个常见的简化：全局“分数最近”只在先验很平、且 $x$ 的范数都一样时才与矩阵准则一致。如果某个方向已经被大量比较钉住，即使那对分数很接近，再问一次也只是锦上添花。

<br>

## 最优设计准则：D、A、E

信息矩阵是 $d\times d$ 的，要比较两个设计的优劣，需要把它压成一个标量。设计 $\xi$ 是各对比较的次数（或比例），$\mathcal{I}(\xi)$ 是该设计下的总信息。三种常见标量准则：

- **$D$ 最优（D-optimal）**：最大化 $\log\det\mathcal{I}(\xi)$。由于 Laplace 后验协方差 $\Sigma\approx\mathcal{I}^{-1}$，$\det\Sigma=1/\det\mathcal{I}$，而高斯置信椭球的体积正比于 $\sqrt{\det\Sigma}$，所以它等价于最小化 Laplace 后验体积。$D$ 最优的一个好性质是对参数的线性重参数化不敏感。
- **$A$ 最优（A-optimal）**：最小化 $\mathrm{tr}\,\mathcal{I}(\xi)^{-1}$，即最小化各参数后验方差之和（$\Sigma$ 对角元之和）。
- **$E$ 最优（E-optimal）**：最大化 $\mathcal{I}(\xi)$ 的最小特征值，照顾最不确定的方向，是一种“最坏情况”准则。

三者的关系可以用特征值 $\lambda_1,\dots,\lambda_d$ 记：$D$ 看 $\sum\log\lambda_k$，$A$ 看 $\sum 1/\lambda_k$，$E$ 看 $\min\lambda_k$。

### 序贯局部设计

在 BT 模型里，信息里的 $p$ 依赖未知参数 $w$，所以没法事先算出“最好的设计”。实际做法是序贯局部设计（sequential local design）：

1. 在当前估计 $\hat w$ 处，计算每对候选把 $\mathcal{I}$ 增加之后的 $\log\det$；
2. 选增加最多的一对；
3. 请人比较，观测标签；
4. 更新 $\hat w$，回到第 1 步，重复。

因为 $p$ 依赖未知的 $w$，这是局部最优，不是事先固定的全局最优设计。这里有一个常见坑：$w$ 估计很差时，算法会认为某些对“分数接近”而在错误的边界附近过度采样；所以前几轮仍要保留覆盖比较图的探索，例如先保证比较图连通、每个物品都被比较过若干次。

### Kiefer–Wolfowitz 等价定理

对线性模型（信息矩阵不依赖参数），有一个检验 $D$ 最优的经典判据。Kiefer–Wolfowitz 等价定理说：把设计 $\xi$ 写成设计空间上的概率测度、$M(\xi)$ 为对应的单位观测平均信息矩阵，则 $\xi$ 是 $D$ 最优的，当且仅当设计空间里每个 $x$ 都满足 $x^{\top}M(\xi)^{-1}x\le d$，$d$ 是参数个数，并且设计支撑上的点等号成立。$x^{\top}M^{-1}x$ 正比于在 $x$ 处的预测方差，所以该定理同时说明 $D$ 最优与“最小化最大预测方差”（G 最优）等价。原文写作 $x^{\top}\mathcal{I}(\xi)^{-1}x\le d$，若 $\mathcal{I}$ 指总信息（随样本数增长），右边应改为 $d/N$；这里改写为按比例归一化的 $M(\xi)$，上界才是 $d$。

BT 的 $p$ 依赖未知的 $s$，信息矩阵是非线性的，等价定理只能在当前 $\hat s$ 处局部使用。这就是序贯设计每轮都重估 $p$ 的原因。

<br>

## 例子：三物品时下一对该问谁

沿用三物品的极大似然 $\hat s=(0,\ -0.8361,\ -1.6723)$。先看最简单的标量准则：各对的 $p(1-p)$。

| 对 | $\hat P$ | $p(1-p)$ |
| :--- | ---: | ---: |
| $A$ vs $B$ | $0.6977$ | $0.2109$ |
| $A$ vs $C$ | $0.8419$ | $0.1331$ |
| $B$ vs $C$ | $0.6977$ | $0.2109$ |

$A$ 与 $C$ 分得最开，再比一次的边际信息最低。

再用矩阵准则核对。钉住 $s_A$ 后，现有样本的观测信息（自由坐标 $s_B,s_C$）满足 $\log\det\mathcal{I}=0.607$（矩阵见上一章 Laplace 例子）。再加一次比较、仍用当前 $p$ 线性化：加 $A$–$B$ 只在 $s_B$ 对角加 $0.2109$；加 $A$–$C$ 只在 $s_C$ 对角加 $0.1331$；加 $B$–$C$ 在两个对角各加 $0.2109$、交叉项各减 $0.2109$。重新计算行列式：

| 新增的对 | 新的 $\log\det\mathcal{I}$ | 增量 |
| :--- | ---: | ---: |
| $A$ vs $B$ | $0.767$ | $0.160$ |
| $B$ vs $C$ | $0.767$ | $0.160$ |
| $A$ vs $C$ | $0.723$ | $0.116$ |

矩阵准则与 $p(1-p)$ 给出同一顺序。$A$–$B$ 与 $B$–$C$ 在这个对称解上增量相同（两对的 $\hat P$ 都是 $0.6977$）。若后验显示 $s_C$ 更不确定，交叉项会把两者分开，$A$ 最优与 $D$ 最优也可能选出不同的对。

一个有用的恒等式可以解释增量：秩一更新 $\mathcal{I}+c\,xx^{\top}$ 的行列式满足 $\log\det(\mathcal{I}+c\,xx^{\top})=\log\det\mathcal{I}+\log(1+c\,x^{\top}\mathcal{I}^{-1}x)$。所以 $D$ 最优的增量由 $c=p(1-p)$ 与当前后验在 $x$ 方向上的方差 $x^{\top}\Sigma x$ 共同决定，正对应前面说的“分数接近”与“方向仍不确定”两个条件。代入本例核对：$A$–$B$ 的 $x=e_B$，$x^{\top}\Sigma x=0.822$，增量 $\log(1+0.2109\times0.822)=0.160$；$B$–$C$ 的 $x=e_B-e_C$，$x^{\top}\Sigma x=0.822+0.920-2\times0.460=0.822$，增量同为 $0.160$；$A$–$C$ 的 $x=e_C$，增量 $\log(1+0.1331\times0.920)=0.116$。

<br>

## 代码

下面两个函数分别计算单对的标量信息和锚定后的观测信息矩阵。它们复用选择数据一章里定义的 `sigmoid` 与 `bradley_terry_nll_grad_hess`。

```python
def contrast_information(scores, i, j):
    """成对概率的 Bernoulli 方差 p(1-p)。物品截距下与该对比的 Fisher 标量成正比。"""
    p = float(sigmoid(scores[i] - scores[j]))
    return p * (1.0 - p)


def anchored_information(scores, comparisons, anchor=0):
    """观测信息 -Hessian(ll)，已删去参照坐标。"""
    _, _, hess_nll = bradley_terry_nll_grad_hess(scores, comparisons)
    free = [i for i in range(scores.shape[0]) if i != anchor]
    return hess_nll[np.ix_(free, free)]
```

负对数似然的 Hessian 就是观测信息，所以 `anchored_information` 直接取 NLL 的 Hessian 并删去参照坐标（否则矩阵有全 1 方向的零空间，不可逆）。用它实现序贯设计时：`comparisons` 里每加一次胜利计数，信息矩阵按新的 $n_{ij}$ 重算；序贯准则还要把 $p$ 用更新后的分数重估，不能一直沿用加样本之前的 $p$。例如，对每个候选对 $(i,j)$，把 `comparisons` 里该对的计数临时加一，算 `np.linalg.slogdet(anchored_information(...))`，取增量最大者即可。

<br>

## 指标征询：当连目标权重都未知时

前面的设计问题假设模型已经写好，只是参数 $w$ 未知。另一类问题更上游：连“用什么指标评价分类器”都要问人。2023 与 2024 年的大纲把“问人要哪一个评价指标”单独称为指标征询（metric elicitation）。2026 年没有这一讲的标题，问题本身仍是实验设计：未知的是线性权，查询的是成对比较。

### 线性性能指标

二分类里，一个线性性能指标可以写成

$$
\phi(h)=1-(a_1\,\mathrm{FP}(h)+a_2\,\mathrm{FN}(h)),
\qquad
a_1,a_2>0.
$$

$h$ 是分类器，$\mathrm{FP}(h)$ 与 $\mathrm{FN}(h)$ 是误报率和漏报率（混淆矩阵里的两个错误格）。$a_1,a_2$ 表达“误报和漏报各有多糟”，例如医疗筛查里漏报通常比误报代价大得多。

### 用成对比较做二分搜索

人不知道自己的 $(a_1,a_2)$，很难直接报出数字，但能在两个分类器（两张混淆矩阵）里指出更喜欢哪一个。Hiranandani、Boodaghians、Mehta 与 Koyejo 的论文（*Performance Metric Elicitation from Pairwise Classifier Comparisons*，AISTATS 2019）利用了一个几何事实：所有可行混淆矩阵构成一个凸集，线性指标的最优分类器落在这个凸集的边界上。于是可以把可行混淆矩阵的边界摊成一条曲线（用一个角度参数化），在曲线上对人做二分搜索式的成对查询。无人噪声时，找到 $\epsilon$-最优分类器的查询数是 $O(\log(1/\epsilon))$。人对近邻查询随机回答时（两个分类器很接近，人难以分辨），用带噪声的二分搜索，阶仍是对数，常数变大。

### 与 Fisher 设计的异同

这与 Fisher 设计问的不是同一件事：

- Fisher 设计假定 $w$ 的模型已经写好，选择最能缩小 $w$ 后验的对；
- 指标征询假定连目标里的权 $(a_1,a_2)$ 都未给定，查询的是分类器而不是物品。

共同点是两边都用成对比较，都用二分或信息增益减少查询。还有一个共同的识别问题：混淆矩阵上的线性权，和 BT 的物品分数，估计之前都要钉住尺度：$(a_1,a_2)$ 同乘正数不改变分类器的排序，正如 BT 分数整体平移不改变胜率。

> [!NOTE]+ 小结
>
> - 单次比较的得分是 $(y-p)x$，Fisher 信息是 $p(1-p)xx^{\top}$；$p$ 接近 $1/2$ 的对信息最多。
> - $D$/$A$/$E$ 最优分别看 $\log\det$、迹的逆、最小特征值；$D$ 最优等价于最小化 Laplace 后验体积。
> - BT 的信息依赖未知参数，只能做序贯局部设计：每轮在当前估计处挑增量最大的对，观测后重估 $p$；前期要保留探索。
> - 指标征询用成对比较分类器来二分搜索未知的指标权重，查询数 $O(\log(1/\epsilon))$。

<br>


# 人类反馈强化学习（RLHF）

贝叶斯估计与微调一章给出了 KL 约束奖励最大化的最优策略 $\pi^\star\propto\pi_{\mathrm{ref}}e^{r/\beta}$，但留下两个实际问题。第一，奖励 $r(x,y)$ 从哪里来？语言模型的动作是整段字符串，“这段回答好不好”手写不出 $r(x,y)$。第二，$Z(x)$ 要对所有字符串求和，闭式解没法直接算。

人类反馈强化学习（reinforcement learning from human feedback, RLHF）的回答是分两步走：先用成对偏好拟合奖励模型（就是前文的 Bradley–Terry 模型，只不过分数由神经网络给出），再在靠近参考策略的约束下用强化学习优化 $\pi$，用采样近似代替算不出的 $Z(x)$。下一章的 DPO 则说明，第二步其实可以绕开。

<br>

## 三阶段流水线

Christiano 等人（2017，*Deep Reinforcement Learning from Human Preferences*）在 Atari 与 MuJoCo 的轨迹片段上收集成对偏好，拟合奖励，再做深度强化学习，证明了“人只需比较、不需写奖励”这条路可行。Ouyang 等人（2022，InstructGPT）把同一结构用在语言模型上，形成三段：

1. **监督微调。** 在示范上做 SFT，得到 $\pi_{\mathrm{ref}}$。
2. **奖励建模。** 对同一提示的两个补全收集 $y_w\succ y_l$（$y_w$ 为胜者，$y_l$ 为败者），用 BT 拟合奖励模型 $r_\phi$。
3. **强化学习。** 以 $r_\phi$ 减 KL 惩罚为回报，用策略梯度（InstructGPT 用 PPO）更新 $\pi_\theta$。$\pi_{\mathrm{ref}}$ 冻结，只作为 KL 的参照。

InstructGPT 论文里的若干实现细节可作参考：标注员一次对同一提示的 $K=4$ 到 $9$ 个回答排序，产生 $\binom{K}{2}$ 个成对比较，同一提示的所有比较放在同一个批次元素里训练，损失除以 $\binom{K}{2}$；奖励模型只用 6B 规模（175B 训练不稳定）；进入 RL 之前，用一个偏置把奖励模型归一化，使标注员示范的平均奖励为 0；KL 惩罚是逐词元加入的。论文还提出 PPO-ptx 变体，在目标里再加上 $\gamma$ 倍的预训练数据对数似然，以减轻在公开 NLP 基准上的性能退化（“对齐税”）；$\gamma=0$ 时就是普通 PPO。

<br>

## 奖励模型：成对 Bradley–Terry 损失

奖励模型 $r_\phi(x,y)$ 通常是把 SFT 模型的最后一层换成输出标量的“奖励头”。它的训练目标就是成对负对数似然，和前文 BT 模型的似然完全相同，只是分数 $s_i$ 换成了 $r_\phi(x,y)$：

$$
\mathcal{L}_{\mathrm{RM}}(\phi)
=
-\mathbb{E}
\big[
\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)
\big].
$$

### 奖励的水平不可识别

损失只依赖奖励差。$r_\phi$ 整体加一个只依赖 $x$ 的函数 $f(x)$，不改变差，因此不改变损失。这与 BT 模型需要钉住一个物品分数是同一个识别问题。实现里常把奖励头的输出在一个批次内减均值，或对参数做 $L_2$，以钉住水平；InstructGPT 用示范均值为 0 的偏置做同样的事。

### 单对的梯度与数值例子

记一对的奖励差 $r_w-r_l=\delta$，单样本损失 $-\log\sigma(\delta)$。由 $\frac{d}{d\delta}\log\sigma(\delta)=1-\sigma(\delta)$，对 $\delta$ 的导数是 $\sigma(\delta)-1$。

取 $\delta=0.5$：$\sigma(0.5)=0.6225$，损失 $0.4741$，导数 $-0.3775$。导数为负，梯度下降会增加 $\delta$：提高胜者奖励、降低败者奖励，步的幅度是 $1-\sigma(\delta)$，也就是“模型认为败者会赢的概率”。$\delta$ 已经很大时 $\sigma(\delta)\approx 1$，导数接近 0，这条比较不再改奖励头。这和逻辑回归的行为一样：已经分对、而且分得很开的样本几乎不贡献梯度。

### 线性奖励头的几何

若 $r=\phi^{\top}h(x,y)$ 且 $\phi$ 是最后一层，$h$ 是冻结的隐藏向量，则由链式法则 $\partial\delta/\partial\phi=h_w-h_l$，$\phi$ 的梯度是 $(\sigma(\delta)-1)h_w+(1-\sigma(\delta))h_l=(\sigma(\delta)-1)(h_w-h_l)$。胜者与败者的隐藏向量若几乎平行（例如两个回答只差几个词），差向量 $h_w-h_l$ 很短，这一对提供的方向很短，和 BT 里 $x=e_i-e_j$ 范数固定是同一几何：梯度方向是两个选项的“特征差”。

### 奖励尺度与 $\beta$ 的耦合

批次里把 $r$ 减均值，不改变任何 $\delta$，因此不改变 $\mathcal{L}_{\mathrm{RM}}$。但到了 PPO 阶段情况不同：减去的均值会改变优势的零点；再除以标准差会改变 $r$ 相对 $\beta$ KL 的尺度。$\beta$ 是在这个尺度下选定的。中途开始做白化（whitening）等于换了 $\beta$，需要重新看留出比较。换句话说，奖励模型的“单位”在 RM 训练里无所谓，在 RL 阶段却直接决定了奖励与 KL 惩罚谁占上风。

<br>

## KL 正则的策略目标与逐词元奖励

第三阶段的序列级目标就是贝叶斯估计一章的 KL 约束问题，只是把真实奖励换成了学到的 $r_\phi$，把任意分布换成了参数化策略 $\pi_\theta$：

$$
\mathbb{E}[r_\phi(x,y)]
-
\beta\,
D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)).
$$

KL 项的作用是防止过优化（over-optimization）：$r_\phi$ 只在人类比较过的那类回答附近准确，策略一旦跑到分布外，就可能找到奖励模型的漏洞（reward hacking），得到很高的 $r_\phi$ 而真实质量下降。

### 序列 KL 的逐词元分解

序列概率是词元概率之积，所以对数比也是逐词元相加。自回归分解下

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=
\mathbb{E}_{\pi_\theta}
\Big[
\sum_t
\log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}
\Big].
$$

因此，把 $-\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ 加进每个词元的奖励（$r_\phi$ 只在最后一个词元给出），在 $\pi_\theta$ 下的期望与序列 KL 一致。这样语言模型生成就变成一个标准的强化学习问题：状态是提示加已生成的前缀，动作是下一个词元，每步有一个 KL 形式的即时奖励，终点有 $r_\phi$。这是实现细节，目标函数仍是上面的序列式；单条样本上的 $\log(\pi_\theta/\pi_{\mathrm{ref}})$ 可能为负，只有期望才是非负的 KL。

<br>

## 近端策略优化（PPO）

有了每步奖励，就可以用策略梯度优化 $\pi_\theta$。朴素策略梯度的问题是步长难调：一步走太大，策略可能突然把某个动作的概率推得极高，之后采到的数据全变了，训练崩溃。近端策略优化（proximal policy optimization, PPO，Schulman 等人 2017）的思路是：每轮用旧策略采一批数据，在这批数据上做多步更新，但通过“裁剪”让新策略不会离旧策略太远。

### 概率比与裁剪目标

PPO 不直接对新策略做无约束梯度步。令概率比

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},
$$

$s_t$ 是状态（提示加前缀），$a_t$ 是动作（下一个词元），$\pi_{\theta_{\mathrm{old}}}$ 是采样这批数据的策略。$\rho_t A_t$ 是重要性加权后的策略梯度代理目标：在 $\theta=\theta_{\mathrm{old}}$ 处，它的梯度正是普通策略梯度。优势（advantage）$A_t$ 衡量“这个动作比该状态下的平均水平好多少”，由旧策略下的回报估计。裁剪目标为

$$
\mathbb{E}
\big[
\min\big(
\rho_t A_t,\ 
\mathrm{clip}(\rho_t,\ 1-\epsilon,\ 1+\epsilon)\,A_t
\big)
\big].
$$

$\epsilon$ 常取 $0.1$ 或 $0.2$，属于算法超参数（PPO 原论文在连续控制基准上以 $0.2$ 表现最好）。取 $\min$ 的意思是“取两者中更悲观的那个”：

- $A_t>0$（好动作）时，$\rho_t$ 超过 $1+\epsilon$ 之后目标不再增加，避免一步把该动作的概率抬得过高。
- $A_t<0$（差动作）时，$\rho_t$ 低于 $1-\epsilon$ 之后目标不再下降，避免一步把该动作的概率压得过低。

### 裁剪的四种情形

令 $\epsilon=0.2$。裁剪目标对 $\rho$ 的依赖分四种，数字都取 $A=\pm 1$。

| $A$ | $\rho$ | $\rho A$ | $\mathrm{clip}(\rho)A$ | $\min$ | 对 $\rho$ 的梯度 |
| ---: | ---: | ---: | ---: | ---: | :--- |
| $+1$ | $1.1$ | $1.1$ | $1.1$ | $1.1$ | $+1$，继续抬高 |
| $+1$ | $1.3$ | $1.3$ | $1.2$ | $1.2$ | $0$，停在 $1+\epsilon$ |
| $-1$ | $0.9$ | $-0.9$ | $-0.9$ | $-0.9$ | $-1$，继续压低 |
| $-1$ | $0.5$ | $-0.5$ | $-0.8$ | $-0.8$ | $0$，停在 $1-\epsilon$ |

（梯度一列描述的是上升方向：目标对 $\rho$ 的导数为 $+1$ 时梯度上升会增大 $\rho$，为 $-1$ 时会减小 $\rho$。）

读表的要点：$A>0$ 且 $\rho>1+\epsilon$ 时，目标取常数 $(1+\epsilon)A$（$A=1$ 时即 $1+\epsilon$），不再奖励更大的概率比。$A<0$ 且 $\rho<1-\epsilon$ 时，目标取常数 $(1-\epsilon)A$，不再奖励更小的概率比。

裁剪是单侧的。反方向——把差动作的概率越抬越高（$A<0$ 且 $\rho>1+\epsilon$）——不会被这张裁剪挡住：此时 $\rho A<\mathrm{clip}(\rho)A$，$\min$ 取未裁剪的 $\rho A$，梯度保持完整，要靠负的优势本身把 $\rho$ 拉回来。同理，好动作的概率被意外压低（$A>0$ 且 $\rho<1-\epsilon$）时也不裁剪。这正是“悲观下界”的含义：只取消那些会让目标显得过于乐观的改进，从不阻止纠错。

### KL 惩罚与裁剪管的是两件事

KL 惩罚与 PPO 裁剪管的是两件不同的事，初学时容易混淆：

| | KL 惩罚 | PPO 裁剪 |
| :--- | :--- | :--- |
| 参照对象 | SFT 参考模型 $\pi_{\mathrm{ref}}$（固定） | 上一轮采样策略 $\pi_{\theta_{\mathrm{old}}}$（每轮更新） |
| 限制什么 | 最终策略能离示范多远 | 单次更新的步长 |
| 属于哪里 | 目标函数本身 | 优化算法 |

只用裁剪、令 $\beta=0$，每一步都很小，但许多小步累积起来，策略仍可能为了代理奖励离开 $\pi_{\mathrm{ref}}$ 很远。

优势估计、价值函数基线与广义优势估计影响方差，不改变“奖励减 $\beta$ KL”这个目标本身。奖励模型 $r_\phi$ 在 RL 阶段通常冻结；若边优化边更新 $r_\phi$，比较数据的分布会跟着策略变，原来的 BT 拟合不再对应当前状态分布。

### 广义优势估计

优势 $A_t$ 需要估计。用价值函数 $V(s)$（另一个网络，估计从状态 $s$ 出发的期望回报）可以构造时序差分误差 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，其中 $r_t$ 是第 $t$ 步的即时奖励（含 KL 项），$\gamma$ 是折扣因子（语言模型里常取 1）。广义优势估计（generalized advantage estimation, GAE）把时序差分按 $\lambda\in[0,1]$ 混合：

$$
A_t
=
\sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda$ 在偏差与方差之间折中：$\lambda=0$ 时 $A_t=\delta_t$，只看一步，方差小、偏差大（完全依赖 $V$ 的准确性）；$\lambda=1$ 时求和望远镜式抵消，回到蒙特卡洛回报减基线，无偏但方差大。它估计的是同一条轨迹回报的优势，不替换奖励里的 $-\beta\log(\pi/\pi_{\mathrm{ref}})$。注意这里的 $\delta_t$ 是时序差分误差，与上文奖励模型里的奖励差 $\delta$ 是两个不同的量。

<br>

## 不训练策略的替代：best-of-$n$ 抽样

PPO 实现复杂、超参数多。一个简单得多的基线是：不必先跑 PPO。从 $\pi_{\mathrm{ref}}$ 抽 $n$ 个补全，用冻结的 $r_\phi$ 打分并取最大，称为 best-of-$n$（也叫拒绝采样，rejection sampling）。它不改模型参数，只多花推理算力。

### best-of-$n$ 与参考模型的 KL

这个过程诱导出一个策略 $\pi_{\mathrm{BoN}}$。回答 $y$ 被输出，需要它出现在 $n$ 个样本中、而且奖励在其中最大。按“$n$ 个样本中任一个是 $y$”的并集界，诱导出的策略满足

$$
\pi_{\mathrm{BoN}}(y)
=
n\,\pi_{\mathrm{ref}}(y)\,
P\big(r(y)\text{ 为 }n\text{ 个样本中的最大}\big)
\le
n\,\pi_{\mathrm{ref}}(y),
$$

因为那个概率不超过 1。（这里的 $P(\cdot)$ 指其余 $n-1$ 个样本的奖励都不超过 $r(y)$ 的概率。中间的等号只是近似：$n$ 个位置中任一个抽到 $y$ 都计一次，$y$ 被重复抽到时会重复计数，所以严格说是并集界意义下的“$\le$”。奖励互不相同时精确式为 $\pi_{\mathrm{BoN}}(y)=F(y)^n-F_-(y)^n$，$F(y)$、$F_-(y)$ 分别是 $\pi_{\mathrm{ref}}$ 下奖励 $\le r(y)$、$<r(y)$ 的概率，它不超过 $n\,\pi_{\mathrm{ref}}(y)F(y)^{n-1}$；$\pi_{\mathrm{ref}}(y)$ 很小时两者几乎相等。无论哪种情形，最右边的上界都成立。）于是密度比处处不超过 $n$：

$$
D_{\mathrm{KL}}(\pi_{\mathrm{BoN}}\|\pi_{\mathrm{ref}})
=
\mathbb{E}\Big[\log\frac{\pi_{\mathrm{BoN}}}{\pi_{\mathrm{ref}}}\Big]
\le
\log n.
$$

$n=4$ 时 KL 不超过 $\log 4=1.386$。上界来自密度比，不依赖 $r$ 的具体形状。文献中常用更精细的公式 $\log n-(n-1)/n$（$n=4$ 时约 $0.636$），它通常在“各回答奖励互不相同、单个回答的概率可以忽略”的假设下推出；Beirami 等人（2024）证明一般情形下它是真实 KL 的上界，而不一定相等。

### 与 PPO 的对比

PPO 用 $\beta$ 把 KL 停在某个预算里；best-of-$n$ 用样本量直接限制 KL，KL 只随 $n$ 对数增长，因此是一种非常“保守”的优化。代价是推理成本随 $n$ 线性增长。$n$ 很大时，argmax 会走到 $r_\phi$ 的漏洞上，和加大 PPO 步数是同一种过优化。实际中 best-of-$n$ 常用来生成高奖励数据再做 SFT，或者作为评估奖励模型质量的手段。

> [!NOTE]+ 小结
>
> - RLHF 三段：SFT 得到 $\pi_{\mathrm{ref}}$；用 BT 损失在成对偏好上训练 $r_\phi$；以 $r_\phi$ 减 $\beta$ KL 为目标做 PPO。
> - 奖励模型只识别奖励差；平移不影响 RM 损失，但白化等尺度变化会改变 RL 阶段奖励与 $\beta$ KL 的相对大小。
> - 序列 KL 可拆成逐词元的 $\log(\pi_\theta/\pi_{\mathrm{ref}})$，作为每步奖励加入。
> - PPO 裁剪以旧策略为参照、限制单步；KL 以 SFT 模型为参照、限制总偏移；裁剪只挡“过于乐观”的方向。
> - best-of-$n$ 不训练策略，KL 不超过 $\log n$；$n$ 过大同样会过优化。

<br>


# 直接偏好优化（DPO）

RLHF 要训练奖励模型、价值网络，再跑不稳定的 PPO 循环，工程上很重。直接偏好优化（direct preference optimization, DPO，Rafailov 等人 2023）的观察是：贝叶斯估计一章已经证明，KL 约束问题的最优策略与奖励一一对应；既然如此，不妨直接把“奖励”用“策略”表示出来，代入 BT 似然，用成对偏好数据直接训练策略。

核心想法一句话：**语言模型本身就是一个（隐式的）奖励模型**。策略类够大时，KL 约束问题的解可以直接用成对似然训练，奖励网络和 PPO 循环都不单独保留。训练过程就是一个分类损失，和训练奖励模型一样简单。

<br>

## 从最优策略到 DPO 损失

### 代入 Bradley–Terry

把贝叶斯估计一章的反解

$$
r(x,y)
=
\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta\log Z(x)
$$

代入 Bradley–Terry。关键在于 BT 只看奖励差，$Z(x)$ 在 $y_w$ 与 $y_l$ 中相同，相减时消失：

$$
\begin{aligned}
P(y_w\succ y_l\mid x)
&=
\sigma\big(r(x,y_w)-r(x,y_l)\big) \\
&=
\sigma\Big(
\beta\log\frac{\pi(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\Big).
\end{aligned}
$$

这一步正好解决了 RLHF 遇到的“$Z(x)$ 算不出”的困难：偏好概率完全由策略与参考模型的对数概率表示，不再需要配分函数。

### DPO 损失

Rafailov 等人（2023）把待求策略 $\pi_\theta$ 放进这个概率，在偏好数据上最小化负对数似然

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
-\mathbb{E}
\Big[
\log\sigma\big(
\beta\,
\Delta_\theta
-
\beta\,
\Delta_{\mathrm{ref}}
\big)
\Big],
$$

其中

$$
\Delta_\theta
=
\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x),
\qquad
\Delta_{\mathrm{ref}}
=
\log\pi_{\mathrm{ref}}(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x).
$$

$\Delta_\theta$ 是当前策略对胜者相对败者的偏好程度（对数几率），$\Delta_{\mathrm{ref}}$ 是参考模型原本的偏好程度。损失鼓励策略“比参考模型更偏好胜者”。$\pi_{\mathrm{ref}}$ 停止梯度。$\log\pi(y\mid x)$ 仍是词元对数概率之和。

与 RLHF 的流水线对比，DPO 只需要两个模型的前向计算（$\pi_\theta$ 与冻结的 $\pi_{\mathrm{ref}}$），不需要在训练中从策略采样，也不需要奖励模型和价值网络。

### 适用条件与理论保证

DPO 论文用“等价类”说明这个代换没有损失一般性：两个奖励若只差一个只依赖 $x$ 的函数，则它们在 BT（以及 Plackett–Luce）下给出相同的偏好分布（论文 Lemma 1），在 KL 约束问题下给出相同的最优策略（Lemma 2）；并且每个这样的等价类都可以写成 $\beta\log(\pi/\pi_{\mathrm{ref}})$ 的形式（Theorem 1）。因此，用策略参数化奖励并不会漏掉任何可表示的偏好。

但这个保证有前提。该代换在“策略类包含上述 $\pi^\star$”时与 KL 约束问题对齐。神经网络只是一个受限策略类时，DPO 是这个重参数化之下的 BT 极大似然，不再保证达到无约束策略类上的 $\pi^\star$。参考模型的支撑还要盖住数据里出现的 $y$；$\pi_{\mathrm{ref}}(y\mid x)=0$ 的补全使对数比无定义。实践中这意味着：偏好数据最好由 $\pi_{\mathrm{ref}}$（或与之接近的模型）生成；如果直接拿别的模型生成的偏好数据，通常先在胜者上做一轮 SFT，让参考模型覆盖这些回答。

<br>

## DPO 的梯度

令 $u=\beta(\Delta_\theta-\Delta_{\mathrm{ref}})$，它就是隐式奖励差（见下文隐式奖励一节）。单样本损失 $-\log\sigma(u)$ 对 $u$ 的导数是 $-(1-\sigma(u))=-\sigma(-u)$。再由 $\partial u/\partial\theta=\beta(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x))$（$\Delta_{\mathrm{ref}}$ 不含 $\theta$），于是

$$
\nabla_\theta\mathcal{L}
=
-\mathbb{E}
\big[
\beta\,\sigma(-u)
\big(
\nabla_\theta\log\pi_\theta(y_w\mid x)
-
\nabla_\theta\log\pi_\theta(y_l\mid x)
\big)
\big].
$$

逐项解读：

- 括号里的方向：梯度下降会增加胜者的对数概率、降低败者的对数概率。
- 权重 $\sigma(-u)$ 是“隐式奖励模型把这对排错的概率”。它在 $u$ 很大时接近 0：模型已经以很大的奖励差把胜者排在前面，这条样本几乎不再推动参数。$u$ 很负时权重接近 1：模型把败者排得更高，更新最大。

这个权重很重要。DPO 论文指出，去掉权重的朴素版本（单纯抬高胜者、压低败者，类似 unlikelihood 训练）会让语言模型退化。权重配合 $\Delta_{\mathrm{ref}}$ 让更新“适可而止”：已经比参考模型更偏好胜者的样本，推力自动减小。

<br>

## 数值例子

取 $\beta=0.1$ 只为把算术做完，不是理论最优温度（DPO 论文的情感生成实验扫描了 $\beta\in\{0.05,0.1,1,5\}$）。参考模型对两个补全的概率都是 $0.2$。当前策略给胜者 $0.3$、败者 $0.1$。

$$
\begin{aligned}
u
&=
0.1\big(\log(0.3/0.2)-\log(0.1/0.2)\big) \\
&=
0.1\log 3
=
0.10986.
\end{aligned}
$$

第二个等号用了 $\log 1.5-\log 0.5=\log 3$。代入 sigmoid：

$$
\sigma(u)=0.5274,\qquad
-\log\sigma(u)=0.6397,\qquad
\sigma(-u)=0.4726.
$$

作为参照，策略与参考模型给出相同概率时 $u=0$，损失为 $\log 2=0.6931$。本例 $0.6397<\log 2$，说明策略已经比参考模型更偏好胜者；权重 $0.47$，仍有推一把的余地。

若把策略概率对调，胜者 $0.1$、败者 $0.3$，则 $u=-0.10986$，损失为 $0.7496>\log 2$，权重 $\sigma(0.10986)=0.5274$。方向反了的策略，损失高于“完全贴住参考模型”，梯度权重也更大。

<br>

## 代码

下面的函数对单个偏好对计算 $u$、损失与梯度权重，并用上面的数值例子做检查。`sigmoid` 来自选择数据一章的定义。

```python
def dpo_terms(logp_w, logp_l, logp_ref_w, logp_ref_l, beta):
    """返回 u、损失 -log σ(u)、以及梯度权重 σ(-u)。对数概率都是标量。"""
    u = beta * ((logp_w - logp_ref_w) - (logp_l - logp_ref_l))
    loss = np.logaddexp(0.0, -u)
    weight = float(sigmoid(-u))
    return float(u), float(loss), weight


u, loss, weight = dpo_terms(
    np.log(0.3), np.log(0.1), np.log(0.2), np.log(0.2), beta=0.1
)
# u ≈ 0.1099, loss ≈ 0.6397, weight ≈ 0.4726
```

实现上的几点说明：

- `logaddexp(0, -u)` 等于 $\log(1+e^{-u})=-\log\sigma(u)$，在 $u$ 很负时也不会下溢（直接算 `-np.log(sigmoid(u))` 时，$\sigma(u)$ 可能下溢为 0，得到无穷大）。
- 批量损失是样本均值。
- 反向传播时，胜者与败者的词元梯度按 `weight * beta` 缩放，符号相反。

在真实的语言模型训练里，四个对数概率 `logp_w` 等都是把回答部分各词元的对数概率求和得到的（提示部分的词元不计入），`logp_ref_*` 在冻结的参考模型上用 `no_grad` 计算。

<br>

## 隐式奖励

### 定义与数值

代换定义的隐式奖励是 $r(x,y)=\beta\log(\pi_\theta(y\mid x)/\pi_{\mathrm{ref}}(y\mid x))$，整体再加 $\beta\log Z(x)$ 也不影响比较。沿用 $\beta=0.1$、$\pi_{\mathrm{ref}}=0.2$、$\pi(y_w)=0.3$、$\pi(y_l)=0.1$：

$$
\begin{aligned}
r(y_w)&=0.1\log(1.5)=0.04055,\\
r(y_l)&=0.1\log(0.5)=-0.06931,\\
r(y_w)-r(y_l)&=0.10986.
\end{aligned}
$$

差正好等于损失里的 $u$。DPO 的 $\sigma(u)$ 就是用这份隐式奖励做的 BT 概率，所以 DPO 在形式上就是“用策略参数化的奖励模型”的 BT 训练。

### 怎么用、要注意什么

训练结束后若要一个可复用的标量奖励（例如给新回答打分、做 best-of-$n$），用这个对数比即可，不必再拟合单独的 $r_\phi$。但对数比依赖 $\pi_{\mathrm{ref}}$：换一个参考模型，同一 $\pi_\theta$ 的隐式奖励会变。另外，隐式奖励只在同一提示下的比较里有意义，因为不同 $x$ 的 $\beta\log Z(x)$ 不同且未知。

<br>

## 实现上的两个坑

**坑一：sigmoid 加在序列上，不是词元上。** 词元实现里 $\log\pi(y\mid x)=\sum_t\log\pi(y_t\mid x,y_{<t})$。$u$ 是两条序列的和之差，不是每个词元各自做一次 sigmoid。把 sigmoid 放进每个词元再平均，得到的是另一个目标，不再对应 BT 似然，也不再与 KL 约束问题对齐。

**坑二：参考模型必须截断梯度。** $\pi_{\mathrm{ref}}$ 的对数概率要在计算图里截断。若参考模型与策略共享初始权重但没有截断，梯度会同时把 $\pi_{\mathrm{ref}}(y_w)$ 拉低（因为 $-\Delta_{\mathrm{ref}}$ 出现在损失里，降低 $\pi_{\mathrm{ref}}(y_w)$ 也能增大 $u$），对数比被参考模型的移动放大，不再是冻结先验下的 BT 似然。

> [!NOTE]+ 小结
>
> - 把 $r=\beta\log(\pi/\pi_{\mathrm{ref}})+\beta\log Z(x)$ 代入 BT，$Z(x)$ 抵消，得到只含策略与参考模型对数概率的 DPO 损失。
> - DPO 梯度抬高胜者、压低败者，权重 $\sigma(-u)$ 是隐式奖励排错的概率；去掉权重会导致退化。
> - 理论对齐要求策略类包含 $\pi^\star$，且参考模型覆盖数据中的回答。
> - 隐式奖励 $\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ 的差就是 $u$，依赖参考模型的选择。
> - 实现时 sigmoid 作用在序列对数概率和之差上，参考模型的对数概率要停止梯度。

<br>

<br>

# 主动征询：该把哪一对送给人标注（Active Elicitation）

人类偏好标注是 RLHF 流水线里最贵的一环：一次比较往往需要几十秒到几分钟的阅读时间，而模型可以在同一时间里生成成千上万对候选补全。若随机挑选比较对，大量预算会浪费在“模型早就知道答案”的对上，例如一段明显通顺的回答对一段乱码。主动征询（active elicitation）要回答的问题是：在已经有一个（不完美的）偏好模型时，下一次应当把哪一对送给人，才能让每一次标注带来最多的信息。

核心想法一句话：<strong>优先询问模型“还没有想清楚”的对，但要区分“没想清楚”是因为参数还没学好，还是因为这件事本来就是随机的。</strong>前者可以靠数据消除，后者不能。前面实验设计一章用 Fisher 信息与 $\log\det$ 准则讨论了“设计哪些比较”，那里假设参数维度有限、比较对可以任意构造；本章把同样的思想换成只依赖预测分布的采集函数（acquisition function），并落到语言模型补全对上。下一章的辅助博弈则进一步放弃“人只是标注器”的假设。

## 两种主动学习设定与两种不确定性

### 池式与流式

主动学习按样本到达方式分两种：

- **池式（pool-based）**：存在一个未标注集合（池），算法每次从中挑一个点送去标注，可以在整个池上比较打分后再决定。
- **流式（stream-based）**：样本依次到来，算法当场决定标或不标；当时不标的样本就被丢弃，之后无法回头。

语言模型的场景天然是池式的：对一个提示，可以先从当前策略采样若干补全放进池里，再在池中所有补全对上计算采集分数，决定哪一对送给人。

### 认知不确定与偶然不确定

预测的不确定性有两个来源，主动征询必须把它们分开：

- **认知不确定（epistemic uncertainty）**：来自参数还没被数据钉死。多做几次比较，后验变窄，这部分不确定会下降。
- **偶然不确定（aleatoric uncertainty）**：来自观测噪声本身。例如同一个标注者面对两段完全相同的文字，只能随机点一个；无论再收集多少数据，这次选择的结果都不可预测。

以 Bradley–Terry 式的逻辑模型 $p=\sigma(w^{\top}x)$ 为例（$x\in\mathbb{R}^d$ 是比较对的特征差，$w\in\mathbb{R}^d$ 是参数，$p\in(0,1)$ 是第一项获胜的概率）：

- 若 $w$ 已知，且恰好使 $p=1/2$，那么标签的 Bernoulli 方差 $p(1-p)=1/4$ **全部是偶然的**，标注它不会改变对 $w$ 的认识。
- 若 $w$ 的后验很宽，即使后验预测（对 $w$ 积分后的）$p$ 接近 $1/2$，其中也有一部分不确定是**认知的**：不同的 $w$ 可能分别给出 $0.1$ 和 $0.9$，平均起来才接近一半。这种对才值得问。

这一区分正是后面 BALD 准则与朴素熵准则的分界线。

<br>

## 采集函数：熵、间隔、最小置信与 BALD

### 只看预测分布的三种准则

记模型对标签 $y$ 的预测分布为 $p(y\mid x)$（$y$ 取有限个类别；在成对比较里 $y\in\{0,1\}$ 表示哪一项获胜）。最简单的一类采集函数只用这个分布本身：

- **熵（entropy）**：$\alpha(x)=-\sum_y p(y\mid x)\log p(y\mid x)$，越大表示越不确定，选最大者。
- **间隔（margin）**：$p_{(1)}-p_{(2)}$，即最大与次大预测概率之差，越小越不确定，选最小者。二元时间隔为 $|2p-1|$。
- **最小置信（least confidence）**：$1-\max_y p(y\mid x)$，越大越不确定，选最大者。

数值上，自然对数下 $p=0.6$ 的熵为 $0.673$，$p=0.7$ 的熵为 $0.611$，$p=0.8$ 的熵为 $0.500$。三个准则都会选 $0.6$（详细算术见下文“算术例子”一节）。换对数底只是把熵整体乘一个正常数，排序不变，因此用 $\log_2$ 还是 $\ln$ 不影响选择。

### 委员会查询

委员会查询（query by committee, QBC；Seung、Opper 与 Sompolinsky 1992 提出）不用单一模型，而是维护多组参数（委员会成员，例如多个随机种子训练的模型或后验样本），用成员之间的分歧衡量不确定性。常见的两种分歧度量：

- **投票熵（vote entropy）**：每个成员给出一个硬标签，统计投票频率得到一个分布，再取它的熵。
- **平均分布的熵减去各成员熵的平均**：先把各成员的预测分布平均再取熵，减去每个成员自身熵的平均值。这正是下面的 BALD 在有限委员会上的蒙特卡洛形式。

### BALD：标签与参数的互信息

贝叶斯主动学习分歧（Bayesian active learning by disagreement, BALD；Houlsby 等人 2011）选择的是标签 $y$ 与参数 $w$ 之间的互信息 $I(y;w\mid x,\mathcal{D})$，其中 $\mathcal{D}$ 是已有数据，$p(w\mid\mathcal{D})$ 是参数后验：

$$
\alpha(x)
=
H[y\mid x]
-
\mathbb{E}_{w\sim p(w\mid\mathcal{D})}
\big[
H[y\mid x,w]
\big].
$$

两项分别是：

- 第一项 $H[y\mid x]$ 是后验预测分布 $p(y\mid x,\mathcal{D})=\mathbb{E}_w[p(y\mid x,w)]$ 的熵，大表示总体预测不确定；
- 第二项是“假如已经知道 $w$，标签还剩多少不确定”，在后验上取平均，即偶然部分（噪声）。

两者之差就是认知部分。差大，表示“不确定主要来自还不知道 $w$”，此时观测标签能显著缩窄后验。由互信息的对称性，这个量也等于“观测 $y$ 之后参数后验熵的期望下降量”，这是 BALD 被称为信息增益采集的原因。

**这意味着什么**：纯熵准则在噪声本质就大的区域（例如两段几乎一样好的回答）也会反复采样，因为那里的预测永远接近一半；BALD 会避开那些怎么标都帮不上参数的点。在偏好标注里，这一差别很实际：标注者之间本就意见不一的“平局对”会持续吸走熵准则的预算。

<br>

## 用在语言模型的补全对上

### 点估计下的主动策略

对固定提示 $x$ 与两个补全 $y_1,y_2$，BT 模型给出 $y_1$ 获胜的概率 $p=\sigma(r(x,y_1)-r(x,y_2))$，即

$$
p=\sigma\big(r(x,y_1)-r(x,y_2)\big),
$$

其中 $r(x,y)$ 是标量奖励，$\sigma$ 是逻辑函数。二元熵在 $p=1/2$ 处最大，也就是奖励差为 0 时。若只有一个点估计奖励模型 $r_\phi$，熵准则（以及间隔、最小置信）就化为一条简单规则：**送出奖励最接近的那一对**。

### 点估计的问题与后验的作用

点估计会过度自信：训练好的 $r_\phi$ 往往把奖励差的绝对值估得偏大，于是所有对的熵整体变小，算法以为没有什么可问的；而剩下“奖励差接近 0”的对，很可能只是某一个点估计碰巧打成平手。

改进做法是给奖励模型一个近似后验，常见的有：

- **深集成（deep ensembles）**：多个独立初始化训练的奖励头；
- **掉线近似（MC dropout）**：推理时保留 dropout，多次前向当作后验样本；
- **Laplace 近似**：在 MAP 处用 Hessian（或其对角、最后一层近似）构造高斯后验，见前面贝叶斯估计一章。

有了后验，就能得到奖励差 $r(x,y_1)-r(x,y_2)$ 的分布，再计算 BALD。这样问的是“参数之间还不同意谁赢”的对，而不是“某一个点估计碰巧打成平手”的对。

### 课程案例与已发表的设置

课程公开大纲把本讲标成“面向大语言模型的主动征询与案例”，案例材料随后挂出。与这个题目直接对应、且已经发表的设置有：

- 轨迹片段的成对比较（Christiano et al., 2017）：用奖励模型集成的分歧（各成员预测的方差）挑选要问人的片段对；
- 机器人奖励的批量主动偏好查询（Bıyık & Sadigh, 2018）：一次生成一批查询，兼顾信息量与批内多样性。

语言模型上的做法是同一准则作用于补全对：先从当前策略对每个提示采样若干个 $y$，构成候选池，再按互信息挑选送标的对。

<br>

## 算术例子：三个准则在二元情形下等价

三个候选对的预测胜率分别是 $0.6$、$0.7$、$0.8$。自然对数下的熵：

$$
\begin{aligned}
H(0.6)&=-0.6\log 0.6-0.4\log 0.4=0.673,\\
H(0.7)&=-0.7\log 0.7-0.3\log 0.3=0.611,\\
H(0.8)&=-0.8\log 0.8-0.2\log 0.2=0.500.
\end{aligned}
$$

熵准则选 $0.6$。另外两个准则：

| 预测胜率 $p$ | 熵 $H(p)$ | 间隔 $\lvert 2p-1\rvert$ | 最小置信 $1-\max(p,1-p)$ |
| ---: | ---: | ---: | ---: |
| $0.6$ | $0.673$ | $0.2$ | $0.4$ |
| $0.7$ | $0.611$ | $0.4$ | $0.3$ |
| $0.8$ | $0.500$ | $0.6$ | $0.2$ |

间隔最小的是 $0.6$，最小置信最大的也是 $0.6$。原因在于：二元 BT 上这三个准则都是 $|p-1/2|$ 的单调函数（熵与最小置信是它的递减函数，间隔 $|2p-1|=2|p-1/2|$ 是它的递增函数，取最小即等价；原笔记写作“都是 $|p-1/2|$ 的递减函数”，对间隔不成立），所以三者选出的对完全一致。它们只有在多类或委员会投票上才会分开：多类时“最大类概率”“前两类之差”“整个分布的熵”看的是分布的不同侧面。

### 投票熵的例子

三人委员会对 $x_1$ 的硬投票是 $2$ 对 $1$。投票分布 $(2/3,\ 1/3)$ 的熵为

$$
-\tfrac23\log\tfrac23-\tfrac13\log\tfrac13=0.637.
$$

若三人都投同一类，熵为 $0$，这一查询的投票分歧为 $0$。需要注意，投票熵用的是硬标签，看不到成员内部的置信差：一个成员给 $0.51$、另一个给 $0.99$，二者都投成同一类，投票熵为 $0$，但实际上二者对这一点的看法差距很大。BALD 用的是软概率，能捕捉到这种差异。

<br>

## 分歧例子：熵准则与 BALD 何时分道扬镳

设后验只有两个等权样本（可以理解为两个集成成员），下面三对比较说明 BALD 如何把认知不确定与偶然不确定分开。

**第一对**：两个样本给出的胜率分别是 $0.2$ 与 $0.8$。预测边缘是二者平均 $0.5$，熵为 $\log 2=0.693$。已知参数后的熵：$H(0.2)=0.500$，同样 $H(0.8)=0.500$，期望仍是 $0.500$。于是

$$
\mathrm{BALD}=0.693-0.500=0.193.
$$

**第二对**：两个样本都给出 $0.5$。边缘熵仍是 $0.693$，条件熵也是 $0.693$，$\mathrm{BALD}=0$。熵准则会去问第二对，因为它的预测和第一对一样“最不确定”（甚至打平时可能优先选它）；BALD 不去问，因为这里的不确定完全来自噪声，标签到了也不改变参数。

**第三对**：两个样本都给出 $0.6$。边缘熵 $0.673$，条件熵也是 $0.673$，$\mathrm{BALD}=0$。点估计的奖励差不为 0，但后验没有分歧。

把三对放在一起：

| 对 | 样本胜率 | 边缘熵 | 条件熵均值 | BALD |
| :--- | :--- | ---: | ---: | ---: |
| 第一对 | $0.2,\ 0.8$ | $0.693$ | $0.500$ | $0.193$ |
| 第二对 | $0.5,\ 0.5$ | $0.693$ | $0.693$ | $0$ |
| 第三对 | $0.6,\ 0.6$ | $0.673$ | $0.673$ | $0$ |

主动查询若只有一个 $r_\phi$，会把第一对和第二对都看成“接近一半”，而忽略第三对；有后验时只保留第一对。这就是“用后验算 BALD”比“用点估计算熵”更省标注的直观原因。

<br>

## 代码：二元 BALD

```python
def bernoulli_entropy(p):
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def bald_binary(probabilities):
    """probabilities: 后验样本给出的 P(y=1)。返回 BALD。"""
    ps = np.asarray(probabilities, dtype=float)
    p_bar = float(ps.mean())
    return bernoulli_entropy(p_bar) - float(np.mean([bernoulli_entropy(p) for p in ps]))
```

`bernoulli_entropy` 中的 `np.clip` 防止 $p=0$ 或 $1$ 时出现 $0\cdot\log 0$ 的数值问题。`bald_binary` 先平均得到边缘概率 `p_bar`，再用“边缘熵减条件熵均值”的定义计算。

`bald_binary([0.2, 0.8])` 约为 $0.193$；`bald_binary([0.5, 0.5])` 为 $0$。常见坑：样本必须来自参数后验（集成成员、dropout 样本、Laplace 抽样），把同一个点估计复制两遍会得到 BALD 恒为 0，此时 BALD 退化为“什么都不问”。

> [!NOTE]+ 小结
> - 主动征询在池中挑选最有信息的比较对；语言模型补全天然是池式设定。
> - 熵、间隔、最小置信在二元 BT 上都是 $|p-1/2|$ 的单调函数，排序一致；点估计下它们都等于“送奖励最接近的对”。
> - BALD = 边缘熵 − 条件熵均值，只计认知不确定，会避开纯噪声的平局对。
> - BALD 需要真正的后验样本；点估计复制多份会让它恒为 0。

<br>


# 辅助博弈：人不只是标注器（Assistance Games）

前面各章的比较都把人当成一个**标注器**：系统提问，人给出一个标签，标注本身不改变环境，也不考虑系统会怎样理解这个标签。现实中的协作并不是这样的。设想一个家用机器人在学习主人喜欢的咖啡口味：主人自己动手冲咖啡时，这个动作既是在完成任务（干预环境），又是在向机器人展示偏好（传递信息）；如果主人知道机器人在看，他甚至可能刻意把关键步骤做得夸张一些，好让机器人看懂。辅助博弈（assistance game）就是把这种“人的动作同时是信息和干预，奖励参数只有人知道”的情形写成一个可分析的博弈。

核心想法：<strong>人和系统共享同一个目标，但只有人知道目标的参数；系统必须一边行动一边推断参数，人则可能为了让系统学会而“教学”。</strong>这与上一章的主动征询互为补充：主动征询里系统决定问什么，人被动回答；辅助博弈里人也是主动的策略方。

## 合作逆强化学习的形式化

Hadfield-Menell、Russell、Abbeel 与 Dragan（2016）提出的合作逆强化学习（cooperative inverse reinforcement learning, CIRL），课程现在称为辅助博弈。它是一个两人、共同收益、信息不对称的部分可观测博弈，要素如下：

- 状态 $s$，人的动作 $a_H$，系统（机器人）的动作 $a_R$；
- 隐藏参数 $\theta\sim p(\theta)$，进入双方共享的奖励 $R(s,a_H,a_R;\theta)$；
- 人观测到 $\theta$；系统只观测状态与双方的动作，维持关于 $\theta$ 的信念 $b(\theta)$；
- 双方最大化**同一个**期望累计奖励。

“同一个奖励”是关键：系统不是在最大化自己的目标、再顺便模仿人，而是真正以人的 $\theta$ 为目标，只是不知道 $\theta$ 是多少。

### 从系统视角看：一个 POMDP

系统若把人的策略当成固定的（已知的函数 $\pi_H(a_H\mid s,\theta)$），自己面对的就是以 $\theta$ 为隐藏状态的部分可观测马尔可夫决策过程（POMDP）：每观测到一次人的动作，就用贝叶斯公式更新 $b(\theta)$，再按信念选择行动。Hadfield-Menell 等人证明，最优联合策略可以通过把 CIRL 约化为一个（规模更大的）POMDP 来求解。

### 从人视角看：教学动机

人若知道系统在学习，就会选择能让系统信念朝正确方向更新的动作，即使该动作的当下奖励较低。这种动作是**教学**，不是独立同分布的最优示范。CIRL 原文的一个结论正是：在 CIRL 的最优解里，人的最优策略一般不同于“只顾自己做到最好”的专家示范（demonstration-by-expert），前者会更有信息量。

用普通逆强化学习的似然去解释教学动作，会把“故意做给对方看”误读成“这件事本身奖励很高”。例如人为了示意“不要碰那个杯子”而绕了一个大圈，普通 IRL 可能得出“人喜欢绕圈”的结论。辅助博弈把人的策略写成依赖系统信念的均衡策略，教学的代价由未来合作的收益来补偿，从而能正确解释这类动作。

<br>

## 与情境赌博机的对比

### 情境赌博机的设定

情境赌博机（contextual bandit）：每轮观测上下文 $x_t$，选择动作 $a_t$，得到奖励 $r_t(x_t,a_t)$ 或一次偏好反馈。性能用遗憾（regret）衡量：

$$
\mathrm{Regret}(T)
=
\mathbb{E}
\Big[
\sum_{t=1}^{T}
\big(
r(x_t,a_t^\star)-r(x_t,a_t)
\big)
\Big],
$$

其中 $T$ 是总轮数，$a_t^\star$ 是该上下文上奖励最大的动作。这里反馈的分布是在算法的信念**之外**定义的：人即使存在，也只是一个把 $(x,a)$ 映到数字的预言机（oracle），不会因为算法学到了什么而改变回答方式。

### 辅助设定多出来的部分

辅助设定里，同一轮可以出现人的动作。该动作依赖 $\theta$，也依赖人对系统信念的预期，因此依赖学习算法本身。把这串动作当成情境赌博机的独立奖励抽样，会丢掉“人为了被看懂而偏离短视最优”的部分。

两个可以算清楚的极端情形：

- **人永不行动**，只在被问到时返回偏好：问题退回偏好赌博机，主动征询一章的采集准则直接适用。
- **人短视地最大化当前 $R$**，不考虑系统是否学会：人的动作就是 $\theta$ 的含噪观测，系统的推断是被动的贝叶斯更新，没有均衡里的教学动机。

课程要区分的是两者之间的中间情形：双方对同一 $\theta$ 的累计奖励负责，“传递信息”与“完成任务”在同一个目标里权衡。

<br>

## 信念更新的数值例子与教学动机

### 短视的人：纯贝叶斯更新

先把短视的人写成可计算的更新。设 $\theta\in\{0,1\}$，先验各 $1/2$。人知道 $\theta$，动作 $a\in\{0,1\}$，动作与 $\theta$ 匹配时即时奖励为 $1$，否则为 $0$。人按 Boltzmann 理性（温度参数 $\beta=2$，见后面有界理性一章）选择动作：两个动作的效用差是 $1$，所以

$$
\frac{e^{2\cdot 1}}{e^{2\cdot 1}+e^{2\cdot 0}}=\sigma(2),
$$

即

$$
P(a=\theta)=\sigma(2)=0.8808,\qquad P(a\neq\theta)=0.1192.
$$

观测到 $a=1$。由贝叶斯公式，先验相等时相互抵消，后验为

$$
P(\theta=1\mid a=1)
=
\frac{0.8808}{0.8808+0.1192}
=
0.8808.
$$

假设系统下一轮必须交付一项，交付正确得 $+1$、错误得 $-1$。选 $\theta=1$ 对应项的期望奖励是

$$
0.8808\cdot(+1)+0.1192\cdot(-1)=0.7616.
$$

选 $\theta=0$ 则是 $-0.7616$，所以系统交付 $\theta=1$ 的项。这一步只用了贝叶斯公式，人的策略被当成不依赖系统信念的固定噪声。

### 教学动机：人的目标含有未来项

教学动机出现在人的目标含有未来项时。设人可以发一个“信号”，成本 $0.1$；信号被系统信任后，系统交付正确项，奖励为 $1$。比较两种选择：

| 人的选择 | 当下 | 未来 | 合计 |
| :--- | ---: | ---: | ---: |
| 发信号 | $-0.1$ | $+1$ | $0.9$ |
| 不发信号（系统保持先验，交付正确与错误各半） | $0$ | $\tfrac12(+1)+\tfrac12(-1)=0$ | $0$ |

人发信号的净收益是 $0.9$；不发信号时系统保持先验，期望奖励是 $0$。

- 短视的人只看即时匹配，不会为了 $0.1$ 的成本去发一个当下奖励为 $0$ 的信号。
- 把未来的 $0.9$ 写进人的回报之后，发信号成为更好的动作。

如果逆强化学习假设人只最大化即时 $R$、且 $\beta\to\infty$（完全理性），这个信号的似然是 $0$，因为它不是短视最优的动作，模型根本无法解释它。辅助博弈则把信号解释成对累计共享奖励的最优反应。

**实际意义**：在 RLHF 里，标注者也常常有“教学”行为，例如刻意选择一个“虽然不完美但方向对”的回答来纠正模型。把这类反馈当作 i.i.d. 的偏好样本，会系统地误读其含义。

> [!NOTE]+ 小结
> - 辅助博弈（CIRL）：人与系统共享奖励 $R(s,a_H,a_R;\theta)$，只有人知道 $\theta$。
> - 固定人的策略时，系统面对以 $\theta$ 为隐藏状态的 POMDP，做贝叶斯更新。
> - 人若考虑未来合作收益，会产生“教学”动作；普通 IRL 会把它误读为高奖励行为。
> - 情境赌博机把人当成固定预言机；辅助设定中人的动作依赖学习算法本身。

<br>


# 采集采样：Thompson 采样与偏好反馈（Acquisition by Sampling）

主动征询一章的采集函数关心的是“学得快”：挑信息最多的点，不在乎被挑出来的选项本身好不好。但在线场景中，每一次展示都有代价，例如推荐系统给用户展示一个差的商品、聊天产品给用户一个差的回答。这时要同时兼顾**利用（exploitation）**——尽量选当前看起来最好的——和**探索（exploration）**——偶尔尝试不确定的选项以免错过更好的。这就是多臂赌博机（multi-armed bandit）问题，前一章已经用遗憾来衡量它的表现。

Thompson 采样（Thompson sampling；Thompson 1933）用一个非常简单的随机化规则解决这一权衡：<strong>按“某个选项是最优的后验概率”来选择它。</strong>实现上不需要算这个概率，只要从后验里抽一个“可能的世界”，然后在这个世界里贪心地行动即可。不确定的选项抽样波动大，偶尔会被抽到最高，于是自然地被探索。本章先讲标量奖励下的 Thompson 采样，再推广到偏好反馈，最后用一段带随机种子的完整轨迹展示它如何“用少量损失换信息”。

## 标量奖励下的 Thompson 采样

### 算法

每个臂（选项）的平均奖励都有一个后验分布。每一轮：

1. 从每个臂的后验中各抽一个均值 $\tilde\mu_k$；
2. 选择抽样值最大的臂；
3. 观测奖励；
4. 更新该臂的后验（其他臂不变）。

### Beta–Bernoulli 共轭

奖励为 Bernoulli（成功 $1$ / 失败 $0$）、先验为 Beta 时，共轭更新是封闭的：

- 先验 $\mathrm{Beta}(1,1)$ 就是 $[0,1]$ 上的均匀分布；
- 每观测一次成功，第一个参数 $\alpha$ 加 $1$；每观测一次失败，第二个参数 $\beta$ 加 $1$；
- 后验均值是 $(\alpha)/(\alpha+\beta)$。

（这里的 $\alpha,\beta$ 是 Beta 分布的参数，与后面 Boltzmann 温度 $\beta$、RUCB 的置信参数 $\alpha$ 无关。）

### 数值例子：探索与利用

- 臂 $A$ 已有 3 次成功、1 次失败，后验 $\mathrm{Beta}(4,2)$，均值 $4/6=0.667$。
- 臂 $B$ 有 1 次成功、3 次失败，后验 $\mathrm{Beta}(2,4)$，均值 $0.333$。

若一次抽样得到 $\tilde\mu_A=0.50$、$\tilde\mu_B=0.60$，规则选择 $B$。$B$ 的后验均值更低，这一轮仍可能被抽中，这就是**探索**：$B$ 只有 4 次观测，后验还很宽，真实均值高于 $A$ 的可能性并未排除。若抽样改成 $\tilde\mu_A=0.70$、$\tilde\mu_B=0.40$，则选择 $A$，这一轮是**利用**。随着观测增多，后验变窄，抽样越来越集中在均值附近，探索自动减少。

### 线性情境赌博机

线性情境赌博机：第 $t$ 轮每个动作 $a$ 有特征向量 $x_{t,a}\in\mathbb{R}^d$，奖励为 $x_{t,a}^{\top}w$ 加噪声，未知参数 $w$ 的后验为正态。每轮先抽 $\tilde w$，再选

$$
a=\arg\max_a x_{t,a}^{\top}\tilde w.
$$

后验方差大的方向上，抽样会偶尔把对应的臂推到最大，从而被尝试（Agrawal 与 Goyal 2013 给出了这一算法的遗憾界）。

<br>

## 推广到偏好反馈

偏好反馈没有标量奖励：每次只观测“两者中哪个更好”。Thompson 式的做法仍然是先抽一个效用参数 $\tilde w$（例如线性效用 $u(x)=x^{\top}w$ 的后验样本），再按 $\tilde w$ 行动：

- **要推荐一项**：选 $\tilde w$ 下效用最高的项。
- **要再问一对**：选 $\tilde w$ 下的最优与次优，或者当前记录的最优与这次抽样的最优。

第二种做法使查询集中在“抽样世界里会改变冠军”的对上：如果某个抽样世界认为另一项才是冠军，就让它与当前冠军正面比较。后验变窄之后，各个抽样世界彼此相似，查询与推荐都会稳定在同一项附近。

### 为什么偏好问题少用期望改进

贝叶斯优化中另一常用的采集函数是期望改进（expected improvement, EI），它依赖“比当前最优好多少”这样一个标量。偏好观测只提供序，不提供差多少；改进量必须借助潜在效用的高斯过程或线性模型才能定义。所以偏好问题里 Thompson 采样与互信息准则比直接搬用期望改进更常见（下一章的偏好贝叶斯优化中可以看到，研究者为此专门设计了 Copeland 意义下的期望改进）。

<br>

## 代码：Bernoulli 与偏好版 Thompson 采样

```python
def thompson_bernoulli(successes, failures, rng):
    """successes、failures 不含先验伪计数。先验 Beta(1, 1)。返回抽样均值最大的臂。"""
    alpha = np.asarray(successes, dtype=float) + 1.0
    beta = np.asarray(failures, dtype=float) + 1.0
    draw = rng.beta(alpha, beta)
    return int(np.argmax(draw)), draw


def thompson_preference_pair(samples_w, features):
    """samples_w: (S, d)。features: (K, d)。用第一个样本的效用选出最优与次优。"""
    values = features @ samples_w[0]
    order = np.argsort(-values)
    return int(order[0]), int(order[1])
```

`thompson_bernoulli` 中加上的 `1.0` 正是 $\mathrm{Beta}(1,1)$ 先验的伪计数；`rng.beta` 对每个臂各抽一次。`thompson_preference_pair` 中 `samples_w` 是 $S$ 个后验样本（形状 $(S,d)$），`features` 是 $K$ 个候选的特征（形状 $(K,d)$），函数取第一个样本下的效用排序，返回最优与次优的下标。

`rng` 由调用方固定种子，结果才能复现。偏好版本只展示“按一个后验样本排序”这一步；完整循环还要观测人的选择并更新 $w$ 的后验——线性效用配正态先验时，这就是贝叶斯逻辑回归的更新（没有闭式解，需要 Laplace 近似等方法，见下文）。

<br>

## 一段完整的 Thompson 采样轨迹

### 六轮记录

两臂真实均值为 $A$：$0.8$，$B$：$0.3$，先验 $\mathrm{Beta}(1,1)$，`numpy` 生成器种子 $0$。每轮先用 `rng.beta` 为两臂各抽一个均值，再用 `rng.random()` 与所选臂的真实均值比较，生成 Bernoulli 奖励。前六轮的抽样与奖励：

| 轮 | 抽样 $(A,B)$ | 选择 | 奖励 | 之后的成功 / 失败 |
| ---: | :--- | :---: | ---: | :--- |
| 1 | $0.702,\ 0.713$ | $B$ | $0$ | $A$ 空，$B$ 为 $0/1$ |
| 2 | $0.003,\ 0.122$ | $B$ | $0$ | $B$ 为 $0/2$ |
| 3 | $0.644,\ 0.377$ | $A$ | $1$ | $A$ 为 $1/0$ |
| 4 | $0.839,\ 0.128$ | $A$ | $1$ | $A$ 为 $2/0$ |
| 5 | $0.818,\ 0.330$ | $A$ | $0$ | $A$ 为 $2/1$ |
| 6 | $0.569,\ 0.161$ | $A$ | $1$ | $A$ 为 $3/1$，$B$ 为 $0/2$ |

下面的脚本可以复现这张表（与上面的 `thompson_bernoulli` 逻辑一致）：

```python
import numpy as np

rng = np.random.default_rng(0)
mu = [0.8, 0.3]                      # 真实均值：A, B
s = np.zeros(2); f = np.zeros(2)     # 成功、失败计数（不含先验）
for t in range(6):
    draw = rng.beta(s + 1, f + 1)    # 每臂从 Beta(s+1, f+1) 抽一次
    a = int(np.argmax(draw))
    r = int(rng.random() < mu[a])    # Bernoulli 奖励
    s[a] += r; f[a] += 1 - r
    print(t + 1, draw.round(3), "AB"[a], r, s, f)
```

### 解读

第 1 轮两臂后验都是均匀分布，$B$ 的后验均值只有 $0.5$（与 $A$ 相同），但抽样 $0.713$ 恰好高于 $A$ 的 $0.702$，于是去拉了较差的臂。两次失败之后，$B$ 的后验变成 $\mathrm{Beta}(1,3)$，均值是 $1/4$，大部分概率质量集中在低值区，后续抽样基本落到 $A$ 下面。

这六轮的累计奖励是 $3$。若事先知道答案、每轮都选 $A$，期望奖励是 $0.8$ 每轮，六轮期望 $4.8$。Thompson 用前两轮的 $0$ 换来了“$B$ 较差”这个信息；第 3–6 轮它都选了 $A$，这几轮的期望遗憾为 $0$（遗憾按真实均值计算，与实际拿到的奖励无关）。（第 5 轮 $A$ 的失败是 Bernoulli 噪声，与探索无关。）

### 线性与偏好版本的后验更新

线性情境版本只多一个矩阵。设先验 $w\sim N(\mu_0,\Lambda_0^{-1})$，观测噪声方差 $\sigma^2$，后验精度为

$$
\Lambda_t= \Lambda_0+\sum x_{a}x_{a}^{\top}/\sigma^2,
$$

求和遍历已选动作的特征。均值由同样的秩一更新得到：$\mu_t=\Lambda_t^{-1}\big(\Lambda_0\mu_0+\sum x_a r/\sigma^2\big)$，每来一个观测，精度加一个秩一矩阵 $x_ax_a^{\top}/\sigma^2$。决策时抽 $\tilde w\sim N(\mu_t,\Lambda_t^{-1})$，再取 $\arg\max_a x_a^{\top}\tilde w$。

偏好版本把奖励 $x^{\top}w$ 换成 $\sigma(w^{\top}(x_i-x_j))$ 的似然（$x_i,x_j$ 是被比较两项的特征）。逻辑似然与正态先验不共轭，精度矩阵不再有闭式。常用 Laplace 近似：在 MAP 估计处，每个观测给精度加上观测信息 $p(1-p)xx^{\top}$，其中此处的 $x$ 指特征差 $x_i-x_j$，$p$ 是 MAP 下的预测胜率。$p(1-p)$ 在 $p=1/2$ 时最大，再次说明“接近平手的比较信息最多”。

> [!NOTE]+ 小结
> - Thompson 采样：从后验抽一个世界，在该世界里贪心行动；不确定的臂因抽样波动而被探索。
> - Beta–Bernoulli 共轭：成功加 $\alpha$、失败加 $\beta$，均值 $\alpha/(\alpha+\beta)$。
> - 偏好版本：抽效用参数 $\tilde w$，推荐 $\tilde w$ 下最优项，或让抽样冠军与当前冠军对决。
> - 偏好似然无共轭，后验用 Laplace 近似，观测信息为 $p(1-p)xx^{\top}$。

<br>


# 对决赌博机与偏好贝叶斯优化（Dueling Bandits）

上一章的 Thompson 采样仍然假设每次能观测到一个标量奖励（或至少对一个效用模型做推断）。在很多实际系统里，最可靠的反馈却只有“相对比较”：搜索引擎把两个排序算法的结果交错展示，看用户点了哪一边（interleaving）；聊天产品把两个模型的回答并排给用户，让其选一个。用户很难说出“这个回答值 7.3 分”，却很容易说出“左边更好”。对决赌博机（dueling bandits）就是在线学习版本的成对比较：每轮选两个臂“对决”，只观测胜负。

核心想法：<strong>把单臂的奖励换成两臂之间的胜率，遗憾也用胜率来定义；算法既要通过比较找出最优臂，又要尽量少把差臂展示给用户。</strong>本章从定义与遗憾出发，用 BT 分数算一个例子，再讨论“最优臂”何时存在（Condorcet、Copeland、Borda），梳理几类代表算法，最后推广到连续输入上的偏好贝叶斯优化。

## 问题定义与遗憾

### 胜率的参数化

非情境对决赌博机：臂集合 $\mathcal{B}=\{b_1,\ldots,b_K\}$。每轮选两臂，观测一次比较结果。Yue、Broder、Kleinberg 与 Joachims（2012）把胜率记为

$$
P(b_i\succ b_j)
=
\frac12+\epsilon(b_i,b_j),
\qquad
\epsilon\in\Big(-\frac12,\frac12\Big),
$$

并且要求反对称 $\epsilon(b_i,b_j)=-\epsilon(b_j,b_i)$ 与 $\epsilon(b_i,b_i)=0$（自己对自己胜率一半）。$\epsilon>0$ 当且仅当 $i$ 以大于一半的概率击败 $j$。

这一参数化与 BT 相容：若每个臂有分数 $s_i$，则

$$
\epsilon(b_i,b_j)=\sigma(s_i-s_j)-1/2 .
$$

需要注意，$\epsilon$ 是概率减去一半，是一个总体量，不是单次观测的 $\pm 1$；单次对决只给出一个 Bernoulli 结果。

### 瞬时遗憾与累积遗憾

瞬时遗憾取所选两臂各自相对最优臂 $b^\star$ 的劣势之和：

$$
r_t
=
\big(P(b^\star\succ b_{t,1})-\tfrac12\big)
+
\big(P(b^\star\succ b_{t,2})-\tfrac12\big)
=
\epsilon(b^\star,b_{t,1})+\epsilon(b^\star,b_{t,2}).
$$

其中 $b_{t,1},b_{t,2}$ 是第 $t$ 轮选出的两臂。累积遗憾 $R_T=\sum_{t=1}^{T}r_t$。（后续文献也有其他写法：例如 Zoghi 等人（2014）取两项的平均 $(\epsilon(b^\star,b_{t,1})+\epsilon(b^\star,b_{t,2}))/2$，只差一个常数因子；还有只计较小一项的“弱遗憾”变体。结构相同。）

两臂都是 $b^\star$ 时 $r_t=0$，但这一轮自己打自己，结果是纯硬币，**没有任何信息**。于是算法必须在“多打几场有信息的对”与“少把差臂亮给用户”之间权衡——这正是探索–利用权衡在对决设定中的具体形式。

<br>

## 例子：三臂 BT 的遗憾与信息量

三个臂的 BT 分数 $(s_1,s_2,s_3)=(0,\ -0.5,\ -1.5)$，最优臂是 $1$。两两胜率与 $\epsilon$：

$$
\begin{aligned}
P(1\succ 2)&=\sigma(0.5)=0.6225, & \epsilon_{12}&=0.1225,\\
P(1\succ 3)&=\sigma(1.5)=0.8176, & \epsilon_{13}&=0.3176,\\
P(2\succ 3)&=\sigma(1.0)=0.7311, & \epsilon_{23}&=0.2311.
\end{aligned}
$$

若本轮亮出 $(2,3)$，两臂都不是最优臂，各自贡献相对臂 $1$ 的劣势：

$$
r_t=\epsilon_{12}+\epsilon_{13}=0.1225+0.3176=0.4401.
$$

若亮出 $(1,2)$，臂 $1$ 的劣势为 $0$，所以 $r_t=0+\epsilon_{12}=0.1225$。

比较两种选择：亮出最优与次优，遗憾更小；同时 $P(1\succ 2)=0.6225$ 比 $P(2\succ 3)=0.7311$ 更接近 $1/2$，单次比较的 Fisher 信息 $p(1-p)$ 更大（$0.235$ 对 $0.197$）。亮出 $(2,3)$ 既累积更多遗憾，又完全不观测最优臂。这个例子说明，好的对决算法应当让最优臂的候选“上场”，再给它配一个有竞争力的对手。

<br>

## 最优臂的定义：Condorcet、Copeland 与 Borda

### Condorcet 胜者

上面的遗憾公式用到了 $b^\star$，但一般的胜率矩阵不一定有一个“最优臂”。最常用的定义是 **Condorcet 胜者**：存在一臂，对其他每一臂的胜率都严格大于 $1/2$。

- 分数模型（如 BT）总有 Condorcet 胜者，因为分数最高的臂对所有其他臂都有 $\epsilon>0$（并列最高时除外）。
- 一般的 $\epsilon(b_i,b_j)$ 可以出现循环：$A$ 赢 $B$、$B$ 赢 $C$、$C$ 赢 $A$，此时没有 Condorcet 胜者。

上面的遗憾定义与“找到 $b^\star$”都以 Condorcet 胜者存在为前提。存在循环时，目标要改成：

- **Copeland 分数**：一个臂以超过一半的胜率击败的对手数，取最大者为 Copeland 胜者（总是存在，可能并列）；
- **Borda 分数**：一个臂对所有对手的胜率之和（等价于对随机对手的平均胜率）。

否则最优臂没有定义。这些概念与后面社会选择一章中的投票规则是同一回事，只是“选民”换成了随机比较。

### 代表算法

下面几类决策规则只保留算法结构，常数与理论界以原论文为准：

- **Interleaved Filter（Yue et al., 2012）**：维持一个候选集和一个当前候选最优臂，让它与其余候选反复对决；用置信界把“显著差于当前候选最优”的臂删去，若发现某臂显著强于当前候选则换它当候选。查询始终集中在候选集内部。
- **RUCB（relative upper confidence bound，Zoghi et al., 2014）**：给每一对的胜率一个上置信界（见下文“置信界”一节），把在这些乐观的界下不输给任何对手（对所有 $j$ 都有 $u_{cj}\ge 1/2$）的臂作为候选冠军 $c$，再以 $c$ 为基准做一次普通 UCB，选对它的上置信界 $u_{jc}$ 最大的臂作为挑战者。（原笔记概括为“看起来能打赢最多对手的臂”，按 Zoghi 等人原文的候选规则改为上述表述。）
- **情境对决（Dudík et al., 2015）**：让胜率依赖上下文，比较的对象从臂变成策略（从上下文到臂的映射）。原文的关键贡献是提出 **von Neumann 胜者**：一个随机化策略，对任何其他策略的胜率都不低于一半；它在 Condorcet 胜者不存在时依然有定义。原笔记把这项工作概括为“$\epsilon$ 依赖上下文，把线性赌博机的置信界搬到成对差上”；前半句成立，但 Dudík 等人的算法基于策略空间上的 EXP4 式对抗学习与分类预言机约化，并不使用线性置信界。把线性赌博机的置信界搬到成对特征差上的做法，属于后续的线性情境对决工作（如 Bengs、Saha 与 Hüllermeier 2022 在线性随机传递模型下的算法）。
- **Dueling posterior sampling（Novoseller et al., 2020）**：用于偏好反馈下的强化学习，后验同时包含环境转移与效用。每轮抽一套转移和效用，解出该样本下的最优策略，掷出两条轨迹，用人对两条轨迹的偏好更新两个后验。有限地平线、离散状态时，转移的 Dirichlet–多项共轭更新与贝叶斯线性效用（基于轨迹特征）可以分开写。2024 年公开课幻灯片用这个算法说明“偏好上的后验采样”。

<br>

## 连续输入：偏好贝叶斯优化

### 高斯过程上的成对观测

偏好贝叶斯优化（preferential Bayesian optimization）把臂从有限集合换成连续输入 $x\in\mathcal{X}$（例如一组超参数、一个提示模板的连续嵌入、一套机器人控制参数），在潜在效用函数 $f:\mathcal{X}\to\mathbb{R}$ 上放高斯过程（GP）先验。观测仍是对决。Chu 与 Ghahramani（2005）用 probit 观测模型：

$$
P(x\succ x'\mid f)
=
\Phi\Big(\frac{f(x)-f(x')}{\sqrt{2}\,\sigma}\Big),
$$

其中 $\Phi$ 是标准正态累积分布函数，$\sigma$ 为噪声尺度。$\sqrt{2}$ 的来源是：假设 $x$ 与 $x'$ 的效用各自加了独立的 $N(0,\sigma^2)$ 噪声，二者之差的噪声方差为 $2\sigma^2$。由于似然不是高斯的，后验也不再是高斯，需要用 Laplace 近似或期望传播（expectation propagation）给出近似的均值与协方差。

### 采集：对决 Thompson 采样

González 等人（2017）的偏好贝叶斯优化（PBO）在这类 GP 偏好后验上设计采集函数，提出了纯探索（pure exploration）、Copeland 期望改进（Copeland expected improvement）与对决 Thompson 采样（dueling-Thompson sampling）三种；在他们的实验里对决 Thompson 采样表现最好。它就是连续域上的对决 Thompson 采样：

- 原笔记给出的简化形式：从近似后验抽一条函数 $\tilde f$，令下一对的一个点为 $\tilde f$ 的最大值，另一个点为当前后验均值的最大值（抽样冠军对当前冠军，与采集采样一章“当前记录最优与这次抽样的最优”对应）。
- González 等人原文的具体做法不同：他们不是在单点效用 $f(x)$ 上放 GP，而是在“对” $[x,x']$ 所在的乘积空间上放 GP，并用 logistic 函数（而非 probit）把它压成胜率 $\pi_f([x,x'])$。第一步从后验抽一个样本 $\tilde f$，经逻辑函数压缩后对对手 $x'$ 积分，得到抽样的 soft-Copeland 分数，取其最大点作为第一个点 $x_{\mathrm{next}}$；第二步固定 $x_{\mathrm{next}}$，取使 $\sigma(f([x_{\mathrm{next}},x']))$ 的后验方差最大的 $x'$，即“最拿不准它能否打败第一个点”的挑战者。这里用的是潜在函数 $\sigma(f)$ 的方差（认知不确定），而不是对决结果 $y$ 的 Bernoulli 方差，后者在两点效用相近时永远接近最大、不会随数据下降——与主动征询一章 BALD 和熵准则的区别是同一个道理。

两种形式的共同点是：一个点来自后验样本的最优，另一个点负责提供对比与信息。

### 与有限维模型的关系

线性 BT 是该模型的退化情形：核取线性核 $k(x,x')=x^{\top}x'$（此时 $f(x)=w^{\top}x$），且观测函数用 logistic 而不是 probit。实验设计一章的 $\log\det$ 准则，对应的是参数有限维时的特殊情况。

<br>

## 代码：对决的 $\epsilon$ 与瞬时遗憾

```python
def duel_epsilon(scores, i, j):
    return float(sigmoid(scores[i] - scores[j]) - 0.5)


def instantaneous_regret(scores, star, i, j):
    return duel_epsilon(scores, star, i) + duel_epsilon(scores, star, j)


scores = np.array([0.0, -0.5, -1.5])
# instantaneous_regret(scores, 0, 1, 2) ≈ 0.4401
# instantaneous_regret(scores, 0, 0, 1) ≈ 0.1225
```

`duel_epsilon` 实现 $\epsilon(b_i,b_j)=\sigma(s_i-s_j)-1/2$（`sigmoid` 为前文定义的逻辑函数）；`instantaneous_regret` 实现两臂相对 `star` 的劣势之和。下标从 0 开始，所以 `star=0` 对应正文的臂 $1$，两行注释分别对应例子中亮出 $(2,3)$ 与 $(1,2)$ 的结果。

`star` 必须真是 `scores` 的 argmax。若调用方传入的 $\epsilon$ 表存在循环，这个函数仍会输出数字，但该数字不再是“相对 Condorcet 胜者”的遗憾。

<br>

## 置信界：RUCB 如何逐对淘汰

### 上置信界

RUCB 给每一对的胜率一个上置信界。记 $t$ 为已经进行的轮数，$n_{ij}$ 为 $i$ 与 $j$ 的比较次数，$\hat p_{ij}$ 为 $i$ 胜 $j$ 的频率。取

$$
u_{ij}
=
\hat p_{ij}
+
\sqrt{\frac{\alpha\log t}{n_{ij}}}.
$$

$\alpha$（$>1/2$）是控制置信水平的超参数。第二项是 UCB 式的探索奖励：比较次数 $n_{ij}$ 越少，界越宽；$\log t$ 随时间缓慢增大，保证每对都不会被永久忽视。

淘汰规则：若 $u_{ji}<1/2$，即使在乐观估计下 $j$ 仍打不赢 $i$，于是 $j$ 不可能是 Condorcet 胜者，可以从“可能的最优”里去掉。

### 数值例子

$t=300$，$\alpha=1$，$n_{12}=100$，其中臂 $1$ 胜了 $90$ 次。于是 $\hat p_{21}=0.10$。计算置信半径：$\log 300\approx 5.704$，$(\log t)/n=0.05704$，平方根为 $0.2388$。

$$
u_{21}=0.10+0.2388=0.3388<1/2.
$$

臂 $2$ 被判定为显著差于臂 $1$。

若臂 $1$ 只胜了 $70$ 次，则 $\hat p_{21}=0.30$，$u_{21}=0.30+0.2388$，即 $u_{21}=0.539>1/2$，臂 $2$ 还留在候选里，需要更多比较。

需要注意，界的宽度只依赖次数和 $t$，不依赖 BT 的传递性约束；它逐对消除，允许数据里暂时出现循环。这是 RUCB 相对于“先拟合 BT 再取最大分数”的稳健之处：不假设模型结构，也就不会被错误的结构误导。

<br>

## 循环：没有 Condorcet 胜者的情形

### 剪刀石头布式的胜率

三臂上可以没有 Condorcet 胜者。设胜率

$$
P(1\succ 2)=0.6,\quad
P(2\succ 3)=0.6,\quad
P(3\succ 1)=0.6.
$$

每个臂都赢一个、输一个，就像剪刀石头布。

- Copeland 分数是“胜率超过 $1/2$ 的对手数”，三臂都是 $1$，打平。
- Borda 分数是对所有对手的胜率之和，三臂都是 $0.6+0.4=1.0$，也打平。

遗憾公式里的 $b^\star$ 没有定义。

### BT 无法表示循环

有分数 $s_i$ 的 BT 模型写不出这个循环：$P(1\succ 2)>1/2$ 要求 $s_1>s_2$，依此类推会得到 $s_1>s_2>s_3>s_1$，这不可能。观测到循环时，有两种处理：

- 换 Copeland / Borda（或 von Neumann 胜者）当目标；
- 承认噪声很大，认为循环是有限样本造成的，会随着比较次数增加而消失。

用 BT 强行拟合，会把三个分数压到彼此接近，对数损失可以还不错，但“存在唯一最优臂”这一结论是模型强加上去的，而不是数据给出的。

> [!NOTE]+ 小结
> - 对决赌博机：每轮选两臂对决，$P(b_i\succ b_j)=\tfrac12+\epsilon(b_i,b_j)$，BT 下 $\epsilon=\sigma(s_i-s_j)-\tfrac12$。
> - 瞬时遗憾 $=\epsilon(b^\star,b_{t,1})+\epsilon(b^\star,b_{t,2})$；自打自零遗憾但零信息。
> - 最优臂需要 Condorcet 胜者存在；循环时改用 Copeland、Borda 或 von Neumann 胜者。
> - RUCB 用逐对上置信界淘汰，不依赖传递性；偏好贝叶斯优化把同样思路搬到连续输入与 GP 上。

<br>


# 风格代理：长度、格式与奖励过优化（Style as a Proxy）

从人类比较里反推出来的奖励分数，会吃进一切与胜负一起变化的东西：长度、格式（列表、标题、加粗）、语气（礼貌、自信）。这些特征在训练数据里确实与“被偏好”相关——更长的回答平均来说可能更完整——但它们不是人真正在意的目标本身。一旦拿这个分数去优化策略，优化器会发现“加长”是提升分数最便宜的方式，于是放大的是这些表面规律。公开研究也反复观察到这一点，例如 Singhal 等人（2023）发现 RLHF 带来的奖励提升有相当一部分可以由长度增加解释，AlpacaEval 等自动评测也因此引入了长度控制版本（Dubois 等人 2024）。

核心想法：<strong>风格特征是质量的代理（proxy），在被当作优化目标时会与真实质量脱钩；需要把风格从奖励里拆出来、或在推理时单独控制、或限制优化强度。</strong>本章依次讨论：风格如何混入奖励，如何拆开（以及拆开时的可识别性问题），推理时的转向向量，以及奖励过优化。

## 风格如何混入奖励

### 一个构造的长度偏差例子

下面的例子是为了算术构造的，系数不是实测回归。记内容质量为 $q$，词元数为 $n$，代理奖励为

$$
r=q+0.1\,n.
$$

- 简洁回答 $q=5$、$n=20$，则 $r=5+2$，即 $r=7$。
- 冗长回答 $q=1$、$n=200$，则 $r=1+20$，即 $r=21$。

BT 会稳定地让冗长者获胜，随后的 PPO 或 DPO 会加长输出。补救办法是把特征写成 $(q,\ n)$ 两维并单独估计各自权重，或在奖励里减去在训练集上拟合的长度项；这样简洁回答的名次才会回来。

关键在于：长度一旦进入 $r$，优化器没有别的渠道知道它是混淆因素（confounder）。奖励模型只输出一个标量，看不出这个标量里哪部分来自内容、哪部分来自长度。

### 多属性奖励

更一般地，多属性奖励写成

$$
r=w^{\top}\phi(x,y),
$$

其中 $\phi(x,y)$ 是属性向量，它的坐标可以是内容奖励头的输出、长度、是否使用列表、礼貌分类器的对数几率等；$w$ 是各属性的权重。$w$ 有两种来源：

- 由另一次小规模征询估计：这仍然是偏好学习，只是在低维属性空间上；
- 由设计者钉住（例如规定长度权重为 0）：这是把已经决定的价值权衡直接写进目标。

估计 $w$ 时，属性之间要有足够的独立变化，否则内容与长度共线，$w$ 不可识别。这和 BT 中比较图不连通导致分数不可识别是同一类问题：数据没有提供区分两个方向的信息。

<br>

## 拆开风格与内容，以及可识别性

### 减去长度项后名次对调

沿用 $r=q+0.1n$。减去长度项 $0.1n$ 之后，简洁回答的分数是 $q=5$，冗长回答是 $q=1$。名次从冗长领先 $21-7=14$，变成简洁领先 $5-1=4$。BT 的输入若改成这个差，胜者对调。

### 只有两类回答时权重不可识别

但“减去 $0.1n$”的前提是已经知道长度权重是 $0.1$。如果只观测这两类回答，权重本身是不可识别的。数据里的特征向量只有 $(q,n)=(5,20)$ 与 $(1,200)$，设一般的线性奖励 $r=w_qq+w_nn$，奖励差为

$$
r_{\mathrm{short}}-r_{\mathrm{long}}
=
4w_q-180w_n,
$$

这只是一个数。BT 的似然只依赖奖励差，记 $c=4w_q-180w_n$，那么满足 $4w_q-180w_n=c$ 的 $(w_q,w_n)$ 有无穷多（一条直线），似然只依赖 $c$。

要单独报告“长度权重是 $0.1$”，需要：

- 第三类回答，使所有回答的 $(q,n)$ 不落在同一条直线上（例如一个高质量且冗长的回答），从而提供第二个独立方向；或者
- 由设计者把其中一个权重钉死（例如规定 $w_q=1$）。

**实际意义**：长度去偏的数据集需要包含“长而好”“短而差”这类打破相关性的样本，否则任何去偏方法都只是在做假设。

<br>

## 推理时的转向向量

### 定义

推理时的风格控制可以不改模型权重。在某一层收集两种风格（例如简洁 / 冗长，正式 / 随意）的隐藏状态 $h$，转向向量（steering vector）定义为两类的均值差：

$$
v
=
\mathbb{E}[h\mid \mathrm{style}_{+}]
-
\mathbb{E}[h\mid \mathrm{style}_{-}].
$$

生成时把 $\alpha v$ 加到该层的隐藏状态上，$\alpha$ 控制强度（$\alpha<0$ 则推向反方向风格）。这类方法在文献中称为激活加法（activation addition）或对比激活加法（contrastive activation addition）。

### 能做什么、不能做什么

这改变的是条件分布的一个局部方向，而不是在成对数据上重新估计一个 $r_\phi$。它适合在已经训练好的策略上调节格式或语气；它不修正奖励模型里已经学到的长度偏好，除非负风格的语料恰好把该方向分离出来。换言之，转向是对**策略**的事后修补，奖励模型的偏差仍在，下一轮用它训练时偏差会再次出现。

<br>

## 奖励过优化与 Goodhart 规律

### 现象

代理奖励 $r_\phi$ 只在训练比较覆盖的分布附近近似人的判断。策略优化把概率质量移向 $r_\phi$ 的高处，往往也就移出了训练分布。Gao、Schulman 与 Hilton（2023）用一个大型“金标准”奖励模型代替人类、训练不同规模的代理奖励模型来研究这一问题，报告的定性现象是：相对参考策略的 KL 增大时，代理奖励继续上升，而用更强评价测得的真实偏好（金标准分数）先升后降。后一段就是奖励过优化（reward overoptimization）。他们还发现，代理奖励模型越大，金标准分数能达到的峰值越高、过优化越轻，并且 PPO 与 best-of-$n$ 采样的“金标准分数–KL”曲线形状不同。

### Goodhart 规律的具体形式

Goodhart 规律（“一个指标一旦成为目标，就不再是好指标”）在这里的具体形式是：一旦 $r_\phi$ 成为训练目标，策略会利用 $r_\phi$ 的漏洞——长度、套话、重复高分短语——这些特征在人的比较里只是与偏好相关，并不是目标本身。

### KL 正则的作用与取值

$\beta$ KL 正则项把策略限制在 $\pi_{\mathrm{ref}}$ 附近，也就是限制在奖励模型见过的分布附近。$\beta$ 不是越大越好：过大时策略几乎停在 SFT 模型，偏好数据里真正的内容改进也用不上。（Gao 等人还观察到，在他们的设置下 KL 惩罚的效果近似于提前停止：它改变的是走到曲线哪一点，而不是曲线本身。）

验证时要同时看三类量：

1. 代理奖励 $r_\phi$；
2. 人的留出比较（或更强的评价者）；
3. 平均长度这类风格统计。

只有第一项上升、后两项变差（真实偏好下降、长度暴涨），就是过优化，而不是训练尚未收敛；此时应当回退检查点、增大 $\beta$，或修正奖励模型。

> [!NOTE]+ 小结
> - 奖励模型会吸收与胜负相关的风格特征（长度、格式、语气），优化时会被放大。
> - 多属性奖励 $r=w^{\top}\phi$ 可拆开风格，但属性共线时 $w$ 不可识别，需要打破相关性的数据或钉死权重。
> - 转向向量在推理时调节风格，不修正奖励模型本身的偏差。
> - 过优化：代理奖励升、真实偏好先升后降；需同时监控代理奖励、留出人类比较与风格统计。

<br>


# 有界理性：人不是完美的效用最大化者（Bounded Rationality）

所有从人类行为反推目标的方法——BT 拟合、逆强化学习、RLHF——都需要一个“人如何根据目标做选择”的模型。最简单的假设是人总是选效用 $U$ 最大的项。但这个假设经不起数据检验：人会犯错、会疲劳、会在两个差不多的选项间随便挑一个。更严重的是，它在数学上也是脆弱的：若假设完全最优，任何一次次优示范的似然都是 0，反演得不到有限的奖励估计。

核心想法：<strong>用一个带“理性程度”参数的随机选择模型代替“总是选最优”，让次优选择有正概率；同时意识到，反演出的效用依赖于这个选择模型，而选择模型本身是一个假设。</strong>本章从 Boltzmann 理性出发，讨论温度与效用尺度的不可识别、最大熵逆强化学习的特征匹配、可计算性的约束，最后用一个按方面剔除的例子展示 softmax 模型（IIA 性质）在何处失败。

## Boltzmann 理性

### 从 Simon 到 Luce

Simon 提出的有界理性（bounded rationality）指：决策者在计算能力与信息有限时，采用“够用”（satisficing）的规则，而不是在完整菜单上最大化。这一思想很难直接写成可估计的模型。可计算的参数形式是 Luce 的 softmax 选择规则，在机器人与 IRL 文献中也叫 Boltzmann 理性（Boltzmann rationality）：

$$
P(a\mid s)
=
\frac{e^{\beta Q(s,a)}}{\sum_{a'}e^{\beta Q(s,a')}}.
$$

其中 $Q(s,a)$ 是在状态 $s$ 下采取动作 $a$ 的价值，$\beta\ge 0$ 是理性程度（逆温度）。两个极限：

- $\beta\to\infty$ 时，概率质量集中在 $Q$ 最大的动作上，成为最优策略；
- $\beta\to 0$ 时，趋向均匀分布，选择几乎与 $Q$ 无关。

两个动作时它化为 $\sigma(\beta\,\Delta Q)$，与 BT 模型完全一致——BT 就是成对比较下的 Boltzmann 理性。

### 博弈中的推广

量化反应均衡（quantal response equilibrium, QRE；McKelvey 与 Palfrey 1995）把同一个 softmax 用在博弈的每个参与者上：每人都按自己期望收益的 softmax 选择，均衡是彼此“最佳软反应”的不动点。$\beta\to\infty$ 时它趋向纳什均衡。

<br>

## 例子：温度与效用尺度不可分

### 误设温度的后果

设真实 $\beta=1$，两个动作 $Q=(2,\ 1)$。第一个动作被选中的概率为

$$
P(a_1)=\sigma(1)=0.7311.
$$

若反演模型误用 $\beta=5$，为了拟合观测到的 $0.7311$，必须满足 $\sigma(5\Delta Q)=0.7311$，于是效用差为

$$
\Delta Q
=
\frac{\mathrm{logit}(0.7311)}{5}
=
0.2.
$$

（$\mathrm{logit}(0.7311)=1$。）尺度被温度吸收：同样的选择概率，既可以是“差距大但很随机”（$\Delta Q=1,\beta=1$），也可以是“差距小但很确定”（$\Delta Q=0.2,\beta=5$）。只观测选择、不单独识别 $\beta$ 时，报告的 $Q$ 差只确定到一个正的乘性常数。实践中通常固定 $\beta=1$ 或固定效用的尺度，二者只需选一个。

### 完全理性假设下似然为 0

若模型取 $\beta=\infty$，要求示范动作是唯一最优的。只要数据里出现过一次 $a_2$，似然就是 0，极大似然不存在。给演示数据加 Boltzmann 噪声，似然才对次优动作有正概率，估计才有意义。这正是前面辅助博弈一章中“教学信号的似然为 0”问题的同一根源。

<br>

## 最大熵逆强化学习：特征匹配

### 模型与梯度

最大熵逆强化学习（maximum entropy IRL；Ziebart et al., 2008）把 Boltzmann 理性搬到整条轨迹上，在轨迹空间写

$$
P(\tau)\propto e^{R(\tau)},
$$

线性奖励 $R(\tau)=w^{\top}\mu(\tau)$，其中 $\mu(\tau)\in\mathbb{R}^d$ 为轨迹 $\tau$ 的特征计数（沿轨迹累加的状态特征），$w\in\mathbb{R}^d$ 为奖励权重。对数似然 $\log P(\tau_{\mathrm{demo}})=w^{\top}\mu(\tau_{\mathrm{demo}})-\log Z(w)$ 对 $w$ 求导，$\log Z$ 的梯度正是模型下的特征期望，于是梯度为

$$
\mu(\tau_{\mathrm{demo}})
-
\mathbb{E}_{\tau\sim\pi_w}[\mu(\tau)].
$$

这里再次出现“观测特征减去模型期望特征”的结构（与 BT、Plackett–Luce、逻辑回归的梯度相同）。梯度为 0 时，模型下的期望特征恰好等于示范特征，这就是特征匹配（feature matching）。

### 计算代价与尺度

内层期望 $\mathbb{E}_{\tau\sim\pi_w}[\mu(\tau)]$ 由当前 $w$ 下的软最优策略给出（用软值迭代计算状态访问频率），所以外层每走一步梯度，都要解或近似一个软强化学习问题，代价很高。奖励的尺度与 softmax 温度同样绑在一起（上一节的不可识别性），实践中固定温度或固定 $\|w\|$。

<br>

## 梯度步的数值演示

### 两条轨迹的例子

两条轨迹，特征 $\mu(\tau_1)=(1,0)$，$\mu(\tau_2)=(0,1)$，示范是 $\tau_1$。

**第一步**：$w=(0,0)$ 时两条轨迹奖励相同，策略均匀，$E[\mu]=(0.5,0.5)$，梯度

$$
(1,0)-(0.5,0.5)=(0.5,-0.5).
$$

步长 $1$ 之后 $w=(0.5,-0.5)$。两条轨迹的奖励是 $0.5$ 与 $-0.5$，差为 $1$，所以

$$
P(\tau_1)=\sigma(1)=0.7311,\qquad E[\mu]=(0.7311,\ 0.2689).
$$

**第二步**：新梯度是 $(1,0)-(0.7311,0.2689)=(0.2689,\ -0.2689)$。再走一步，$w\approx(0.769,\ -0.769)$，奖励差约 $1.538$，$P(\tau_1)\approx 0.823$（即 $\sigma(1.538)$）。

| 步 | $w$ | $P(\tau_1)$ | $E[\mu]$ | 梯度 |
| ---: | :--- | ---: | :--- | :--- |
| 0 | $(0,0)$ | $0.5$ | $(0.5,0.5)$ | $(0.5,-0.5)$ |
| 1 | $(0.5,-0.5)$ | $0.7311$ | $(0.7311,0.2689)$ | $(0.2689,-0.2689)$ |
| 2 | $\approx(0.769,-0.769)$ | $\approx 0.823$ | | |

梯度的范数每步下降，示范特征与模型期望逐渐靠近。注意示范只有一条且只有两条候选时，$P(\tau_1)$ 会不断趋向 $1$、$w$ 发散——这是单示范极大似然的常见现象，实践中靠正则或多条示范避免。

**解读**：$w$ 的正负方向就是“示范有、另一条没有”的特征坐标；若再乘温度，同一步的 $w$ 会整体放大，概率不变的条件是温度一起改。

### 与可行性表述的关系

Ng 与 Russell（2000）把同一匹配写成可行性问题：专家策略最优的条件是一组关于奖励的线性不等式。他们在有限状态情形下给出了用转移矩阵表示的充要条件；在大状态空间、从采样轨迹做 IRL 的部分，已经把奖励写成基函数的线性组合 $R=\sum_i\alpha_i\phi_i$，要求专家策略在初始状态的价值不低于一组其他策略的价值，这些约束对 $\alpha$ 是线性的。用特征期望 $\mu$ 的记号写成下式，是后来 Abbeel 与 Ng（2004）学徒学习的写法：线性奖励下，专家策略最优当且仅当对每个其他策略 $\pi$，

$$
w^{\top}(\mu_E-\mu_\pi)\ge 0.
$$

这里 $\mu_E,\mu_\pi$ 分别是专家与策略 $\pi$ 的期望特征计数。满足这组不等式的 $w$ 构成一个锥（$w=0$ 永远满足，正倍数也满足），所以“专家最优”并不挑出唯一的奖励。Ng 与 Russell 用最大化间隔等启发式在其中挑选；最大熵逆强化学习换了一种方式消除歧义：在“期望特征等于示范特征”的约束下取熵最大的轨迹分布，$w$ 正是这一约束的拉格朗日乘子，因此梯度才是上面的特征差。（原笔记写作“在这个锥里取使轨迹分布熵最大的 $w$”，严格说最大熵 IRL 的解不必落在该锥内，因为它允许示范次优；此处按原论文的约束优化表述修正。）

<br>

## 可计算性：全菜单 softmax 的代价

### 配分函数的问题

全菜单 softmax 要求对所有动作求和。动作是任意长度字符串（语言模型的回答）时，这个和就是配分函数 $Z(x)=\sum_y e^{\beta r(x,y)}$，候选数随长度指数增长，不能枚举。

三条把计算留在可行范围的路线：

- **成对 BT**：只在数据里出现的两个 $y$ 上归一化，不需要对全体求和。
- **DPO**：最优策略的闭式里含有 $Z(x)$，但它只依赖提示 $x$，在两个回答的奖励差中相减消失，训练时不必估计 $Z$。
- **Plackett–Luce**：若只观测到全排序的前 $k$ 位，乘积就在这 $k$ 步停下，每一步只在被展示的候选上归一化，不必列出其余字符串。

### 人的真实决策过程

人的真实过程也常常不是全菜单的随机效用模型（RUM）。Tversky（1972）的按方面剔除（elimination by aspects）描述的是：先选一个属性，用它删掉一批不满足的选项，再用下一个属性继续删，直到剩下一个。这种过程可以违反无关选项独立性（IIA），也可以产生顺序效应。用 logit 去拟合它，得到的 $w$ 是“最佳 logit 近似”，不一定是人在内部最大化的那组权重。

文献中两个方向的经验：

- Shah 等人（2019）讨论过：与其把某种偏差形式（例如近视、损失厌恶）写死在模型里，不如检验这种偏差能否从数据里学出来；他们用可微规划器从示范中学习人的规划过程，发现这样做可以胜过假设了错误偏差的模型，但可微近似本身带来的精度损失抵消了大部分收益，因此结论是目前需要在数据驱动的学习与合理的偏差假设之间折中。
- Bobu 等人（2020）指出，Boltzmann 模型把每条轨迹当作独立选项，在连续轨迹空间里会出问题：加入几条与已有轨迹几乎相同的轨迹，会不合理地稀释其他选项的概率（与下文红巴士/蓝巴士是同一类 IIA 问题）。他们提出考虑轨迹在特征空间中相似性的 LESS（Limiting Errors due to Similar Selections）模型，在用户研究中更好地解释了人的选择，在有真值的仿真中奖励推断更准确，在 7 自由度机械臂上单条示范的推断也更稳健；但把 12 名用户的示范合并做批量推断时，LESS 的稳健性反而低于 Boltzmann，作者把这一点列为尚待厘清的权衡。（原笔记把这项工作概括为“过强的复杂噪声假设不如更简单的选择模型稳健”，与原文主旨不符，此处按论文内容修正。）

两条都说明：反演结果依赖于选择模型，而选择模型是假设。

<br>

## 剔除规则的例子：IIA 在何处失败

下面是一个为了把概率算完而写死的剔除规则，用来看 IIA 在何处失败，不是 Tversky 论文里的拟合权重。

选项是汽车、红巴士、蓝巴士。规则：以 $1/2$ 的概率只保留“私人交通”，并在其中均匀抽取；若这一步被跳过（另外 $1/2$ 的概率），则在当前菜单上均匀抽取。汽车是唯一的私人交通。

**菜单 $\{汽车, 红巴士\}$**：

$$
P(\mathrm{car})=1/2+1/2\cdot 1/2=3/4,\qquad P(\mathrm{red})=1/4.
$$

第一项 $1/2$ 来自“只保留私人交通”分支（必然选汽车），第二项来自“均匀抽取”分支（两选一）。概率比 $P(\mathrm{car})/P(\mathrm{red})$ 是 $3$。

**菜单加上蓝巴士**：

$$
P(\mathrm{car})=1/2+1/2\cdot 1/3=2/3,\qquad P(\mathrm{red})=1/2\cdot 1/3=1/6.
$$

概率比是 $4$。

同一规则、同一对选项，菜单一变，概率比从 $3$ 变成 $4$。softmax（以及任何满足 IIA 的模型）在任何菜单上都保持这个比不变，因为 $P(\mathrm{car})/P(\mathrm{red})=e^{\beta(U_{\mathrm{car}}-U_{\mathrm{red}})}$ 与其他选项无关。直观上，蓝巴士主要抢的是红巴士的份额，而 softmax 认为它按比例从所有选项那里抢。用 logit 去拟合这组选择，得到的 $w$ 是两个菜单之间的折中，换菜单后预测会偏。

> [!NOTE]+ 小结
> - Boltzmann 理性 $P(a\mid s)\propto e^{\beta Q(s,a)}$ 让次优选择有正概率；两选项时即 BT。
> - 温度 $\beta$ 与效用尺度不可同时识别，需固定其一；$\beta=\infty$ 时一次次优示范即令似然为 0。
> - 最大熵 IRL 的梯度是“示范特征 − 模型期望特征”，每步需解一个软 RL 问题。
> - 全菜单 softmax 的配分函数不可算；BT、DPO、截断 Plackett–Luce 绕开了它。
> - 人的真实过程（如按方面剔除）可违反 IIA；反演结果依赖于所假设的选择模型。

<br>

<br>

# 社会选择理论：Arrow、Gibbard–Satterthwaite 与常见投票规则

前面各章几乎都默认“存在一个偏好”：一个 Bradley–Terry 分数向量、一个奖励函数、一个要对齐的目标。可现实中的偏好数据来自许多标注者，他们的排序互相冲突。把许多人的排序合成一个“集体排序”或“集体选择”，正是社会选择理论（social choice theory）研究的问题。它对偏好学习的意义在于：任何把多人比较合并成一个奖励的做法，本质上都是一条投票规则，因而逃不开这门学科几条著名的不可能定理。

本章的核心直觉是：**合并多人偏好没有“免费的”最好办法**。只要选项不少于三个，几条看似都很合理的要求（对任何偏好都要有输出、全体一致时尊重全体、不被无关选项干扰、没有独裁者、不能被谎报操纵）就不能同时满足。因此，任何聚合方案都必须明确放弃其中某一条，而“放弃哪一条”就是方案设计者的价值选择。

先固定三个基本对象：

- **偏好剖面（preference profile）**：所有投票人的排序放在一起，即一个列表 $(\succ_1,\ldots,\succ_n)$，其中 $\succ_i$ 是第 $i$ 个投票人在选项集合上的严格全序。
- **社会排序函数（social welfare function）** $F$：把一个剖面映成一个社会排序（一条完整的全序）。
- **社会选择函数（social choice function）** $f$：把一个剖面映成一个获胜项（只要一个赢家，不要完整排序）。

Arrow 定理讲的是社会排序函数 $F$，Gibbard–Satterthwaite 定理讲的是社会选择函数 $f$。两者结论相似，但对象不同，这一点在后面会反复用到。

<br>

## 一个最小的悖论：Condorcet 循环

### 三人三选项的例子

最简单的冲突只需要三名投票人、三个选项：

| 投票人 | 排序 |
| :--- | :--- |
| 1 | $A\succ B\succ C$ |
| 2 | $B\succ C\succ A$ |
| 3 | $C\succ A\succ B$ |

**成对多数（pairwise majority）** 的做法是：对每一对选项，看多数人把谁排在前面。

- $A$ 对 $B$：投票人 1、3 把 $A$ 放在 $B$ 前，$A$ 以 2 比 1 赢。
- $B$ 对 $C$：投票人 1、2 把 $B$ 放在 $C$ 前，$B$ 以 2 比 1 赢。
- $C$ 对 $A$：投票人 2、3 把 $C$ 放在 $A$ 前，$C$ 以 2 比 1 赢。

于是得到 $A\succ B\succ C\succ A$：多数关系形成循环。每个人自己的排序都是传递的，但多数关系不传递。这称为 **Condorcet 悖论**。一个在成对多数下击败所有其他选项的选项称为 **Condorcet 胜者（Condorcet winner）**；这个剖面没有 Condorcet 胜者。

这个现象对偏好学习有直接含义：即使每个标注者都是“理性的”（排序传递），把他们的成对比较合并之后，数据本身可能就是循环的。Bradley–Terry 模型只能表示传递的分数，遇到这种数据只能把三者拟合成近似相等的分数。

### Borda 计数在同一剖面上的结果

**Borda 计数（Borda count）** 不看成对多数，而是给名次打分：三个选项时，第一名 2 分、第二名 1 分、第三名 0 分，再把所有人给的分加起来。

$$
A:\ 2+0+1=3,\qquad
B:\ 1+2+0=3,\qquad
C:\ 0+1+2=3.
$$

（以 $A$ 为例：投票人 1 把它放第一得 2 分，投票人 2 放第三得 0 分，投票人 3 放第二得 1 分。）三个选项打成平局。这个剖面是循环对称的（把 $A\to B\to C\to A$ 改名后，得到的仍是同一组排序，只是投票人换了编号），因此任何同时满足匿名与中立的规则都只能打平。

Borda 满足**帕累托**性质：如果全体都把 $A$ 放在 $B$ 前，那么每个投票人给 $A$ 的分都严格高于给 $B$ 的分，加起来 $A$ 的分数一定更高。但它不满足下面要讲的 Arrow 的**无关方案独立性（IIA）**：社会对 $A$ 与 $B$ 的比较会受它们与 $C$ 的相对名次影响。

> [!EXAMPLE]+ 例1 Borda 违反 IIA 的一对剖面
>
> 两名投票人、三个选项。剖面 P1：投票人 1 是 $A\succ B\succ C$，投票人 2 是 $B\succ C\succ A$。Borda 分为 $A=2+0=2$，$B=1+2=3$，社会把 $B$ 放在 $A$ 前。
>
> 剖面 P2：投票人 1 改成 $A\succ C\succ B$，投票人 2 改成 $B\succ A\succ C$。两人关于 $A$ 与 $B$ 的相对名次都没变（投票人 1 仍是 $A$ 在前，投票人 2 仍是 $B$ 在前），只是 $C$ 的位置挪了。Borda 分变成 $A=2+1=3$，$B=0+2=2$，社会改为把 $A$ 放在 $B$ 前。
>
> 关于 $A$、$B$ 的个人信息完全相同，社会结论却翻转了，这正是违反 IIA。原因在于 Borda 不仅看“谁在前”，还看“隔了几个名次”：P1 中投票人 2 的 $B$ 与 $A$ 之间隔着 $C$，P2 中隔着 $C$ 的变成了投票人 1。后文介绍教材时会看到，Borda 满足的是一条放宽后的独立性 IIA'，恰好允许这种“距离”起作用。

<br>

## Arrow 不可能定理

### 定理陈述

Condorcet 循环说明成对多数会失败，Borda 说明换一种计分方法会违反 IIA。很自然的问题是：有没有某种聪明的社会排序函数能同时避开这些问题？Arrow（1951）证明了答案是否定的。

**Arrow 不可能定理。** 选项不少于 3 个时，不存在定义在全部严格排序剖面上的社会排序函数，同时满足以下四条：

1. **无限制域（unrestricted domain）**：每个人的任意严格排序都允许，规则对任何剖面都要给出社会排序。
2. **弱帕累托（weak Pareto）**：若所有人都把 $a$ 排在 $b$ 前面，社会也把 $a$ 排在 $b$ 前面。
3. **无关方案独立性（independence of irrelevant alternatives, IIA）**：社会关于 $a$ 与 $b$ 的名次，只依赖每个人关于 $a$ 与 $b$ 的名次，不依赖他们如何排列其他选项。
4. **非独裁（non-dictatorship）**：不存在这样一个人，其排序总是等于社会排序。

换言之，满足前三条的规则必然是独裁的。每一条单独看都很温和，矛盾在于组合。

### 每条公理在偏好学习里的对应

- 放弃**无限制域**，就是对偏好的形状做假设，例如后面会讲的单峰偏好，或者“所有人的偏好都来自同一个 Bradley–Terry 分数加噪声”。
- 放弃 **IIA**，就是允许名次距离或第三方选项影响两两比较，Borda 走的就是这条路。
- **弱帕累托**几乎没有人愿意放弃：全体标注者都说回复 $a$ 比 $b$ 好，奖励模型却偏好 $b$，显然说不过去。
- **非独裁**在实践中会以隐蔽形式被破坏：如果某个标注群体提供了绝大多数比较，合并模型就几乎等于这个群体的偏好（见下一章“公平聚合”的加权问题）。

### 两种 IIA 不是一回事

“IIA”这个缩写在课程里出现过两次，含义完全不同，容易混淆：

- **Luce 的 IIA**（随机效用一章）说的是**同一个人**在不同菜单上的选择概率比：$p(j\mid\mathcal{S})/p(k\mid\mathcal{S})$ 不因菜单里多了一个选项而变化。
- **Arrow 的 IIA** 说的是**社会名次**对剖面里“无关项”的不敏感：社会对 $a$、$b$ 的排序不因个人对 $c$ 的排法变化而变化。

logit 模型满足前者，但这并不因此提供一种满足后者的投票规则。把多人的选择数据塞进一个 logit / Bradley–Terry 模型，模型内部的成对概率比满足 Luce IIA，可作为“投票规则”的整个拟合过程仍然要服从 Arrow 定理。

<br>

## 防策略性与 Gibbard–Satterthwaite 定理

### 动机：投票人会说谎

Arrow 定理假设每个人如实报告偏好。但如果投票人知道规则，就可能为了让结果对自己更有利而谎报排序。标注场景里也一样：标注者若知道自己的比较会改变模型的奖励，就有动机夸大自己的偏好。一条规则若使任何人都无法通过谎报获益，称为**防策略的（strategy-proof）**。

### 定理陈述

**Gibbard–Satterthwaite 定理**（Gibbard 1973，Satterthwaite 1975）。设选项不少于 3 个，社会选择函数定义在全部严格排序上，且每个选项都在某个剖面上当选（**满射**，surjective）。若它是防策略的（没有人能靠谎报排序使结果按自己的真实偏好变好），则它是独裁的。

与 Arrow 对照：Arrow 讲的是输出全序的 $F$、要求 IIA；Gibbard–Satterthwaite 讲的是输出一个赢家的 $f$、要求防策略。两者都把“非独裁”逼到了死角。满射这一条用来排除“永远选同一个选项”这种平凡但防策略的规则。

### 定理的边界

定理否定的是一个**特定组合**，而不是一切合并方式。以下几种情况不在它的适用范围内：

- **两个选项的多数票**是防策略的，因为“选项不少于 3 个”的假设不满足。只有两个选项时，谎报只能把票投给自己更不喜欢的那个，不可能获益。
- **认可投票（approval voting）**：每人勾选一个“认可集合”，得票最多者当选。它的选票不是一个严格全序，所以不在这条定理的定义域里。
- **随机化社会选择**（按某种概率抽出赢家）同样需要另述（Gibbard 1977 对随机规则另有刻画）。

这些边界说明：定理否定的是“全域、确定、满射、防策略、非独裁”这一组合，不是否定任何形式的合并。

### 对偏好学习的直接后果

把所有标注者的比较倒进一个 BT 模型，得到的是在 logit 噪声假设下的**基数合并**，即按比较次数加权的平均效用。它不是一个定义在严格排序上的防策略社会选择函数。标注者若知道自己的比较会移动系统奖励，Gibbard–Satterthwaite 说明不能指望一个确定的非独裁规则让所有人都愿意报出真实全序。

实践上要靠**任务设计**降低操纵空间，而不是靠一个满足全部公理的投票公式。常见手段包括：

- 只问成对的具体例子，而不是让标注者报完整排序或抽象的价值立场；
- 随机抽问，让单个标注者难以预测自己的回答会落在哪一对上；
- 不展示系统当前的偏好或分数，使标注者无法据此“对冲”。

<br>

## 常见投票规则在同一张票上的分歧

### 七人剖面

Condorcet 循环里所有规则都打平，看不出差别。下面七名投票人的剖面专门用来把三种计票分开：

| 人数 | 排序 |
| ---: | :--- |
| 3 | $A\succ B\succ C$ |
| 2 | $B\succ C\succ A$ |
| 2 | $C\succ B\succ A$ |

### 相对多数

**相对多数（plurality）** 只数第一名：$A$ 得 3，$B$ 得 2，$C$ 得 2，$A$ 当选。注意 $A$ 虽然第一名最多，却有 4 个人把它放在最后。

### 成对多数与 Condorcet 胜者

逐对比较：

- $B$ 对 $A$：后两组（$2+2=4$ 人）都把 $B$ 放在 $A$ 前，$B$ 以 $4$ 比 $3$ 赢。
- $B$ 对 $C$：第一组 3 人和第二组 2 人把 $B$ 放在 $C$ 前，$B$ 以 $5$ 比 $2$ 赢。
- $C$ 对 $A$：后两组都把 $C$ 放在 $A$ 前，$C$ 以 $4$ 比 $3$ 赢。

$B$ 击败所有对手，是 Condorcet 胜者。

### Borda

第一名 2 分、第二名 1 分：

- $A$：只有第一组把它放第一，$3\times 2=6$，其余为 0，共 6 分。
- $C$：第二组放第二、第三组放第一，$2\times 1+2\times 2=6$。
- $B$：第一组放第二、第二组放第一、第三组放第二，$3\times 1+2\times 2+2\times 1=9$。

$B$ 当选。

### Copeland

**Copeland 分数**是在成对多数中击败的对手数：$B$ 击败两人，得 2；$C$ 击败 $A$，得 1；$A$ 得 0。Copeland 胜者是 $B$。只要 Condorcet 胜者存在，它的 Copeland 分数就是最大的 $m-1$，所以 Copeland 是“Condorcet 一致”的规则。

### 补充：Kemeny 规则

另一条常被提到的规则是 **Kemeny（Kemeny–Young）规则**：在所有可能的全序中，找出与各投票人排序“总不一致对数”（Kendall tau 距离之和）最小的那一个作为社会排序。在这个剖面上，排序 $B\succ C\succ A$ 与三条成对多数（$B\succ A$、$B\succ C$、$C\succ A$）全部一致，因此它就是 Kemeny 排序：穷举六个全序，$B\succ C\succ A$ 的加权 Kendall 距离为 $8$（第一组 3 人各差 2 对、第三组 2 人各差 1 对），是唯一最小值；次优的 $B\succ A\succ C$ 为 $9$，相对多数的赢家打头的 $A\succ B\succ C$ 为 $10$。Kemeny 规则也是 Condorcet 一致的；它还可以解释成 Condorcet 式“存在正确排序、每人带噪声报告”模型（Mallows 模型）下的极大似然排序，这与下一章的噪声模型是同一种思路。代价是一般情况下计算它是 NP 困难的。

### 这意味着什么

同一张票，$A$ 与 $B$ 可以分别当选：相对多数选 $A$，成对多数、Borda、Copeland、Kemeny 都选 $B$。在偏好数据上报告“人更喜欢 $A$”之前，必须写明用的是第一名、成对多数还是 Borda。对语言模型评测而言，“被选为最佳回复的次数”（类似相对多数）与“平均两两胜率”（类似 Borda）可以给出不同的冠军。

<br>

## 操纵的具体例子与绕开不可能定理的两条退路

### 相对多数如何被改票操纵

Gibbard–Satterthwaite 是存在性结论，下面这张票把“可操纵”具体化：

| 人数 | 排序 |
| ---: | :--- |
| 5 | $A\succ B\succ C$ |
| 4 | $B\succ C\succ A$ |
| 3 | $C\succ B\succ A$ |

如实投票时，第一名票数为 $A$ 5、$B$ 4、$C$ 3，$A$ 以 5 票当选。三名把 $C$ 放在第一、$B$ 放在第二的投票人更喜欢 $B$ 而不是 $A$。他们若改报 $B$ 为第一，则 $B$ 得 $4+3=7$，$A$ 仍得 5，$B$ 当选。谎报使结果按他们的真实偏好变好（从最差的 $A$ 变成次好的 $B$），所以这张相对多数不是防策略的。这就是日常所说的“策略性投票”或“弃保”。

Gibbard–Satterthwaite 说的是：只要选项不少于三个、规则又要满射并且对所有排序都有定义，这种可操纵性就无法在保持非独裁的同时消失。换一种规则（例如 Borda）只能改变**谁**在**哪种剖面**上能操纵，不能彻底消除操纵。

### 退路一：只有两个选项（May 定理）

两个选项时，May（1952）的刻画把多数票定下来。三条性质是：

- **匿名（anonymity）**：谁投票不影响规则，把投票人的身份打乱，结果不变。
- **中立（neutrality）**：把两个选项的名字对调，结果也对调。
- **正响应（positive responsiveness）**：有人把票改向 $a$ 时，$a$ 不会从当选变成落选；原来是平局时，会变成 $a$ 当选。

这三条一起等价于简单多数。也就是说，两个选项时多数票是唯一“公平且敏感”的规则，同时也是防策略的。选项数一到三个，Arrow 的前提成立，这条刻画不再有非独裁的推广。这也是成对比较数据看上去“没问题”的原因：每一次比较只涉及两个选项；问题出在把许多对拼成一个全局排序的时候。

### 退路二：单峰偏好（Black 中位投票人定理）

另一种退路是限制定义域，即放弃“无限制域”。**单峰偏好（single-peaked preferences）**：选项排在一条线上（例如政策从左到右、回复从短到长），每人有一个最优点（峰），离最优点越远越差。

Black（1948）证明：中位投票人的最优点能在成对多数下击败任何其他点。

> [!EXAMPLE]+ 例2 五个最优点的中位数
>
> 五名投票人的最优点为 $1,2,2,3,4$，排序后中位（第 3 个）是 $2$。
>
> 若有人提议改到 $3$：最优点在 $1$ 与 $2$ 的三人离 $2$ 更近（或就在 $2$），反对改动；最优点在 $3$ 与 $4$ 的两人赞成。$3$ 对 $2$ 是 2 比 3，$2$ 仍获胜。向左改到 $1$ 同理会输。

直觉上，中位数左右两侧各有不超过一半的人，任何偏离中位数的提议都会被“另一侧加中位本身”的多数否决。定义域收成单峰之后，中位数是 Condorcet 胜者，循环不再出现；代价是定理的“无限制域”不再满足。教材在“聚合”一章还会给出单峰域上防策略的广义中位数规则。

> [!NOTE]+ 小结
>
> - 偏好剖面、社会排序函数、社会选择函数是三个基本对象；多数关系可以循环（Condorcet 悖论）。
> - Arrow：至少三选项时，无限制域、弱帕累托、IIA、非独裁不能同时成立；Borda 违反的是 IIA。
> - Gibbard–Satterthwaite：至少三选项、全域、满射、防策略的确定规则必为独裁；两选项、认可投票、随机规则不在其范围内。
> - 同一张票上相对多数、成对多数、Borda、Copeland 可以选出不同赢家，报告结论必须写明计票方式。
> - 绕开不可能定理的常见办法是缩小问题：只比较两个选项（May 定理），或假设偏好单峰（Black 定理）。
> - 把多人比较合并进一个 BT 模型是一种按比较次数加权的基数合并，不是防策略规则，需要靠任务设计抑制操纵。

<br>


# 公平聚合：噪声、异质性与分组评估

上一章从公理角度说明“合并偏好没有完美规则”。本章换一个统计角度：当一批标注出现分歧（例如 70% 的人偏好 $a$、30% 偏好 $b$）时，这个分歧到底意味着什么？答案决定了应该怎样合并，以及合并之后谁的偏好被牺牲了。

有两种根本不同的解释模型：

- **噪声模型**：存在一个正确排序，每个人独立地以概率 $p$ 把它报对。分歧来自误差，多问几个人就能逼近真相。这是 Condorcet 陪审团定理的世界。
- **异质性模型**：每个人有自己的偏好参数 $w$，没有“报对”这回事。分歧是真实的价值差异，多问几个人只会更精确地测出各群体的比例。

同一个 70% 对 30% 的分裂，两种模型都解释得了。区分它们要靠**重复标注**：同一人在同一对上反复标注仍然波动，支持噪声；同一人稳定、人与人之间不同，支持异质。因子模型与选择预测章节里的潜类（latent class）与混合 logit（mixed logit）拟合的是后者。本章先看噪声模型下多数票的保证，再看异质性模型下合并带来的偏斜，最后用具体数字和分组公平指标收尾。

<br>

## Condorcet 陪审团定理：噪声模型下多数票为什么可靠

### 定理

**孔多塞陪审团定理（Condorcet jury theorem）。** 设 $n$ 为奇数（避免平局），投票人独立，每人正确的概率都是 $p>1/2$，则多数正确的概率随 $n$ 增大而增大，且当 $n\to\infty$ 时趋向 1。

直觉来自大数定律：每个人是一次“偏向正确”的独立硬币，正确票的比例会集中在 $p>1/2$ 附近，多数自然站在正确一边。

### $n=3$ 的计算

$p=0.6$、$n=3$ 时，多数正确等价于“三人全对”或“恰好两人对”：

$$
\begin{aligned}
P(\text{多数正确})
&=
p^3+3p^2(1-p) \\
&=
0.216+3\times 0.36\times 0.4 \\
&=
0.648.
\end{aligned}
$$

第一项是三人全对 $0.6^3=0.216$；第二项中 $3=\binom{3}{2}$ 是“哪两人对”的选法，$p^2(1-p)=0.36\times 0.4$。$0.648>0.6$，多问两个人已经提高了正确率。

### 假设失效的情形

- $p<1/2$ 时同样的公式会把多数推向错误：人越多，多数越确定地错。
- $p$ 在人群里不相同、误差相关（例如同一篇错误的标注说明被所有人读到）时，定理的独立同质假设不成立，人数增加不一定变准。相关误差相当于“有效人数”远小于名义人数。

### 适用范围

这条定理只在“存在正确选项”时给多数票背书。事实性问题（哪条回复的数学答案对）适合它；风格与价值问题（回复该简洁还是详尽、该多保守）没有外部正确排序，增加标注者只会更精确地估计各群体的比例，不会收敛到一个客观排序。

<br>

## 合并训练中的隐性加权

### 合并 BT 的似然天然按样本量加权

合并 BT 的对数似然是各条比较的和：

$$
\ell(s)=\sum_{\text{比较 }(i,j,y)}\Big[y\log\sigma(s_i-s_j)+(1-y)\log\sigma(s_j-s_i)\Big].
$$

一个群体若提供了 80% 的比较（例如 10 人中的 8 人），梯度里就有 80% 的项来自他们。少数群体的稳定偏好会被拟合到残差里，看起来像“噪声”。

后果是：留出准确率可以是 0.8，同时在少数群体的对上只有 0.2。总体对数似然并不惩罚这种倾斜，因为少数样本的权重小。这正是上一章“非独裁”在实践中被隐蔽破坏的方式：多数群体不必是独裁者，只要占据足够多的样本，合并模型就近似等于他们的偏好。

### 应该报告什么

报告时至少分开三组数字：

1. 全体对数损失；
2. 各群体自己的对数损失；
3. 群体之间对同一对的分歧率。

分歧率高而群体内重复标注稳定，说明该用混合系数（潜类 / 混合 logit）或分开的奖励头（每个群体一个奖励），而不是加噪声温度。加温度只会让模型对所有对都“更不确定”，不能把少数群体的偏好方向表示出来。近年关于多元对齐（pluralistic alignment）的工作讨论的正是这一类方案：保留多个偏好分布或按群体条件化，而不是把它们平均成一个。

### 策略层面的分组公平指标

评价已经训练好的分类或排序策略时，常用的分组公平定义与偏好群体是同一件事的两种切法。设预测为 $\hat Y\in\{0,1\}$、真实标签为 $Y$、组别为 $G$：

- **人口统计均等（demographic parity）**：预测为正的比例在两组相同，即 $P(\hat Y=1\mid G=g)$ 与 $g$ 无关。
- **机会均等（equalized odds）**：在真实标签的每一类内部，预测为正的比例在两组相同，即 $P(\hat Y=1\mid Y=y,G=g)$ 与 $g$ 无关（$y=1$ 时是真正例率相等，$y=0$ 时是假正例率相等）。

它们约束的是**策略输出**，不是标注合并。一个在多数标注上 BT 似然很高的奖励，优化出来的策略仍可能在某一组上违反机会均等。分组指标要在策略上单独计算，不能从训练损失读出。

<br>

## 用数字看偏斜：短回答与长回答、陪审团扩容、分组误差

### 例：8 比 2 的长度偏好

十个人各比较一次“短回答”与“长回答”。八人稳定偏好短，两人稳定偏好长。合并样本里短回答的胜率是 $0.8$，BT 的极大似然分数差就是这个胜率的 logit：

$$
s_{\mathrm{short}}-s_{\mathrm{long}}=\mathrm{logit}(0.8)=\log 4=1.3863.
$$

（$\mathrm{logit}(0.8)=\log\frac{0.8}{0.2}=\log 4$。）

用这个差做预测，短回答永远判赢。于是：

- 在多数人的标签上准确率是 $1$；
- 在少数人的标签上准确率是 $0$；
- 全体准确率是 $0.8$。

对数损失方面，模型给短回答的概率是 $0.8$。多数人的每条比较损失 $-\log 0.8\approx 0.223$；少数人的每条比较损失 $-\log 0.2=1.609$。若按全体平均，少数人的 $1.609$ 只占两成权重，平均值 $0.8\times 0.223+0.2\times 1.609\approx 0.500$ 看起来并不糟糕。

把两组分开写，才会看到差距：按分类错误率是多数组 $0$、少数组 $1$（原笔记写作“$0$ 与 $1.609$ 同时存在”，其中的 $0$ 对应多数组的错误率；严格按对数损失，多数组是 $0.223$ 而不是 $0$）；按对数损失是多数组 $0.223$、少数组 $1.609$，相差约 7 倍。前者被照顾得很好，后者被系统性地判错。

### 例：陪审团从 3 人扩到 5 人

$p=0.6$ 的陪审团从 $n=3$ 增到 $n=5$，多数正确即至少三人正确，按二项分布逐项相加：

$$
\binom{5}{3}0.6^3 0.4^2
+
\binom{5}{4}0.6^4 0.4
+
0.6^5
=
0.6826.
$$

三项分别为 $10\times 0.216\times 0.16=0.3456$、$5\times 0.1296\times 0.4=0.2592$、$0.07776$，合计约 $0.6826$，高于 $n=3$ 时的 $0.648$。增加的是“有正确答案且误差独立”时的正确率，不是价值分歧下的代表性。上一个长度偏好的例子里，再加多少人，合并模型仍然判短回答永远赢。

### 例：满足人口统计均等，却违反机会均等

分组误差可以与人口统计均等同时发生。设每组 10 个正例、10 个负例。

| | 正例判正 | 负例判正 | 真正例率 | 假正例率 | 预测为正比例 | 准确率 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 甲组 | 8 | 2 | $0.8$ | $0.2$ | $10/20=0.5$ | $0.8$ |
| 乙组 | 5 | 5 | $0.5$ | $0.5$ | $10/20=0.5$ | $0.5$ |

- 甲组：8 个正例被判正、2 个负例被判正，真正例率 $0.8$，假正例率 $0.2$；判对的是 8 个正例加 8 个负例，准确率 $16/20=0.8$。
- 乙组：5 个正例被判正、5 个负例被判正，两个率都是 $0.5$，准确率 $(5+5)/20=0.5$，相当于抛硬币。

两组的预测为正的比例都是 $10/20=0.5$，人口统计均等成立。真正例率 $0.8\neq 0.5$，机会均等不成立。只看整体准确率 $(0.8+0.5)/2=0.65$，这组差距不出现。

这个例子说明：满足一个公平指标并不意味着满足另一个，也不意味着各组被同样好地服务。报告时要明确用的是哪一个指标，并把分组数字列出来。

> [!NOTE]+ 小结
>
> - 70/30 的分歧可以是噪声，也可以是异质；用同一人的重复标注区分。
> - 陪审团定理只在存在正确答案、误差独立且 $p>1/2$ 时成立：$p=0.6$ 时 $n=3$ 为 $0.648$，$n=5$ 为 $0.6826$。
> - 合并 BT 按比较次数加权，少数群体的稳定偏好会被当作残差；需要分别报告全体损失、分组损失与群体间分歧率。
> - 人口统计均等与机会均等约束策略输出，要在策略上单独计算，二者可以一个成立一个不成立。

<br>


# 课程项目实践：要求、检查清单与预分析计划

前面各章给出了模型、算法与定理，本章转向“怎样把它们用到一个真实项目里”。CS329H 的成绩有一半来自项目，而偏好学习项目最常见的问题不是模型不够复杂，而是实验设计含糊：没说清观测是什么、没钉住参照、只报告代理奖励、把多人分歧揉成一个分数。下面先列出项目的硬性要求，再给出一份按前文各章整理的检查清单，最后说明预分析计划该写什么。

<br>

## 项目要求与评分

- **形式与权重**：项目占课程成绩的一半，可个人、两人或三人完成；三人组需要附贡献说明。
- **预分析计划**：2026-11-04 前提交，占项目分的 30%。内容包括动机、至少三篇文献、方法与分析、时间与分工；NeurIPS 版式 3–4 页（不含参考文献）。课程主页列出的细分是：概述与动机 6%、文献综述 6%、方法与分析 6%、时间与分工 2%、格式 10%。
- **终稿**：2026-12-03 提交终稿与代码。终稿占项目分的 60%，NeurIPS 版式至多 8 页（不含参考文献）；主页细分为引言与背景 10%、方法 15%、结果与讨论 20%、结论与展望 5%、格式 10%。
- **代码**：占 10%（文档 5%、可运行 5%）。要求有 README、固定依赖版本与随机种子，并能复现文中全部图表。
- **迟交**：两组截止合计 3 天迟交额度。

**生成式工具政策。** 项目可以使用任何生成式工具，但科学标准由提交者负责。使用时，提交须另附三段各约 150 词的说明：

1. 如何处理抄袭、不实与偏差（诚信说明）；
2. 工具如何改变了自己的分析（使用反思）；
3. 社会、伦理与误用风险（影响说明；主页原文还列出环境影响，以及署名、数据与合规的处理）。

缺任何一段，课程主页写明扣除课程总成绩的 10%。

**选题方向。** 大纲列出的方向包括：偏好学习的新应用、选择模型的比较、异质性与聚合、主动查询、RLHF 的变体、具体场景里的辅助博弈、小数据下的偏好模型、反演与偏差校正。

<br>

## 项目检查清单

清单的每一条都对应前文某一章里的一个常见坑。动手做实验和写报告时逐条核对：

1. **写清观测类型。** 写明菜单是什么、观测是选择、成对还是排序，是显示偏好（revealed，从行为推断）还是陈述偏好（stated，直接询问）。不同观测对应不同似然，混用会让估计失去意义。
2. **处理识别问题。** 效用要钉住一个参照（例如令某一物品分数为 0），或写明先验方差（用贝叶斯先验固定尺度）。比较图不连通时，不要解释跨连通分支的分数差，因为数据对它没有任何约束。
3. **同时报告对数损失与准确率。** 准确率只在预测概率远离 $1/2$ 的对上会看起来很好；对接近五五开的对，准确率接近抛硬币，而对数损失能反映概率是否校准。
4. **说明能否外推。** 有新用户或新物品时，用特征或低秩因子；只有截距的模型（每个物品一个独立分数）无法给没见过的物品打分，要在表格里标成不可外推。
5. **策略优化要多指标并报。** 同时报告代理奖励、留出成对胜率、相对 $\pi_{\mathrm{ref}}$ 的 KL，以及长度等风格均值。代理单独上升而留出胜率下降，按过优化（reward over-optimization）处理。
6. **多人标注要报告一致性。** 标注者多于一人时，给出群体内一致性与群体间分歧。分歧稳定就不要只用合并 BT（见“公平聚合”一章）。
7. **主动查询要有对照。** 报告选择准则（熵、BALD 或 $\log\det$）以及与随机查询的留出对数损失差；没有随机基线，无法判断主动查询是否真的有用。
8. **可复现。** 代码从干净环境按 README 跑通，种子固定，外部代码保留许可与修改说明。

<br>

## 预分析计划该写什么

预分析计划（pre-analysis plan）的作用是在看到结果之前把分析方案定下来，防止事后挑选对自己有利的指标或设置。做法是：在跑实验之前把下面几段写死，终稿再填数字。

- **观测**：菜单、比较还是排序、显示还是陈述、标注者是否分组。
- **参数**：参照项钉在哪里；若用 $\pi_{\mathrm{ref}}$，它是哪一个冻结模型；$\beta$ 的候选值。
- **查询**：若主动抽样，写下熵、BALD 或 $\log\det$ 中的哪一个，以及对照的随机查询。
- **成功标准**：留出对数损失、分组对数损失、KL、平均长度。只写代理奖励不算成功标准。
- **失败时如何解释**：事先写好每种失败对应回查哪一部分。
  - 强连通不成立（MLE 发散），回到比较图；
  - 隐式奖励与留出人评背离，回到长度和格式等风格变量；
  - 群体间分歧高于群体内重复误差，回到分组似然。

> [!NOTE]+ 小结
>
> - 项目占总成绩一半：预分析计划 30%（11 月 4 日）、终稿 60% 与代码 10%（12 月 3 日），共 3 天迟交额度。
> - 使用生成式工具须附三段约 150 词的说明，缺一段扣总成绩 10%。
> - 检查清单覆盖观测类型、识别、损失与准确率、外推、过优化、群体分歧、主动查询基线与可复现性。
> - 预分析计划先写死观测、参数、查询、成功标准与失败解释，终稿只填数字。

<br>


# 配套教材《Machine Learning from Human Preferences》各章要点

课程配套的在线教材把前面讲过的内容按“基础 → 学习 → 行动 → 反演 → 聚合”的顺序重新组织，并补充了不少课上没有展开的例子和定理。本章按教材的章节顺序摘出要点，重点放在课上没有覆盖、或与课上写法有出入的地方；与前文重复的推导只给出教材里的编号和例子。

书在 [mlhp.stanford.edu](https://mlhp.stanford.edu/)，作者 Sang T. Truong、Andreas Haupt、Sanmi Koyejo，页眉日期 2026-09-23。侧栏是引言、五章、结语、附录（核对时侧栏另列有 Teaching Materials 一项）：

- 引言，[首页](https://mlhp.stanford.edu/)
- 1 Foundations，[chap1.html](https://mlhp.stanford.edu/src/chap1.html)
- 2 Learning，[chap2.html](https://mlhp.stanford.edu/src/chap2.html)
- 3 Action，[chap3.html](https://mlhp.stanford.edu/src/chap3.html)
- 4 Inversion，[chap4.html](https://mlhp.stanford.edu/src/chap4.html)
- 5 Aggregation，[chap5.html](https://mlhp.stanford.edu/src/chap5.html)
- 结语 Conclusion，附录 Appendices（Mathematical Notation、Glossary），链在侧栏

十分制季度（十周）的课时建议写在引言页：第 1 章约一周，第 2、3 章各约三周，第 4、5 章各约一周半。Bradley–Terry 似然、Gumbel、Luce、DPO 损失和 Arrow 这几条，前面已经算过，下面直接用书上的编号和例子。

<br>

## 引言：从预训练到后训练

### 为什么预训练不够

书里的例子跨推荐、检索、机器人轨迹、排序，后半一直用语言模型对齐作主线。

预训练（pretraining）最小化下一词交叉熵，即式 (1.1)：

$$
\mathcal{L}\_{\mathrm{pretrain}}
=
-\mathbb{E}\_{x\sim\mathcal{D}}
\left[\sum_{t=1}^{T}\log\pi_\theta(x_t\mid x_{<t})\right].
$$

这里 $\mathcal{D}$ 是语料分布，$x=(x_1,\ldots,x_T)$ 是一段长度为 $T$ 的文本，$\pi_\theta(x_t\mid x_{<t})$ 是模型在前文条件下给第 $t$ 个词元的概率。

为什么这个目标会“忠实复制语料”？因为对数评分（logarithmic scoring rule）$S(p,y)=\log p(y)$ 是**严格恰当的（strictly proper）**：真实分布为 $q$ 时，期望得分 $\mathbb{E}\_{y\sim q}[S(p,y)]$ 只在预报 $p=q$ 时最大。（期望得分等于 $-H(q)-\mathrm{KL}(q\|p)$，KL 非负且仅在 $p=q$ 时为 0。）交叉熵因此把模型概率拉向语料里的频率。这既是优点也是问题：有害续写只要在语料里出现过，这项损失同样会抬高它的概率。预训练学到的是“人们写了什么”，而不是“人们想要什么”。

### 后训练：换数据而不是换目标形式

后训练（posttraining，书里也叫 alignment）换的是数据：同一提示 $x$ 下，标注者指出 $(y_1,y_2)$ 里哪条更好。

- **RLHF** 先拟合奖励 $r(x,y)$，再在靠近原模型的约束（KL 惩罚）下把策略往奖励上推。
- **DPO** 跳过单独的奖励网络，损失就是 (1.2)，和前文 DPO 一章的 $u=\beta(\Delta_\theta-\Delta_{\mathrm{ref}})$ 是同一个式子。把隐式奖励定义成 $r^\star(x,y)=\beta\log(\pi_\theta(y\mid x)/\pi_{\mathrm{ref}}(y\mid x))$ 之后，(1.3) 就是 Bradley–Terry。

### 三物品的数值例子

书里用 $V_A=2$，$V_B=1$，$V_C=0$ 算了一遍。成对概率只看分差：$p(A\succ B)=\sigma(1)\approx 0.731$，$p(A\succ C)=\sigma(2)\approx 0.881$。全排序 $A\succ B\succ C$ 的 Plackett–Luce 概率是“先从三个里选出 $A$，再从剩下两个里选出 $B$”：

$$
\frac{e^{2}}{e^{2}+e+1}\cdot\frac{e}{e+1}\approx 0.665\times 0.731\approx 0.486.
$$

$p(A\succ B)$ 只依赖 $V_A-V_B$，去掉 $C$ 这个概率不变，这就是这条模型里的 IIA（Luce 意义下）。前文用的效用是 $(0,1,2)$，与这里的 $(2,1,0)$ 只差标号与一个常数平移；全体加一个常数，所有概率相同。

<br>

## 第 1 章 基础（Foundations）：数据形式、Rasch 模型与噪声的含义

### 两种数据形式

**逐项反应（item-wise）** 记 $Y_{ij}\in\{0,1\}$：用户 $i$ 接受或拒绝物品 $j$。$N$ 个人、$M$ 件物品排成 $N\times M$ 的**反应矩阵（response matrix）**，即式 (1.5)。购买、跳过一首歌、向右滑、标注者是否标出某条内容，都是这种矩阵。

**成对比较（pairwise）** 是两件物品里谁赢，语言模型后训练用的是这一种。

### Rasch 模型及其与 Bradley–Terry 的关系

Rasch 模型（源自心理测量，人–题作答）把潜在分数写成胃口加吸引力（appetite + appeal）：

$$
H_{ij}=U_i+V_j,
\qquad
P(Y_{ij}=1)=\sigma(U_i+V_j).
$$

$U_i$ 是用户 $i$ 的“胃口”（多爱接受东西），$V_j$ 是物品 $j$ 的“吸引力”。按 $U_i$ 和 $V_j$ 把矩阵行列排好之后，图案很规整：胃口大的人接受大多数物品，热门物品被大多数人接受，接受和拒绝的分界大致沿 $U_i+V_j=0$ 的对角线。

同一人比较 $j$ 与 $k$，$U_i$ 在差里消失：

$$
P(j\succ k\mid i)=\sigma(V_j-V_k).
$$

这就是 Bradley–Terry。也就是说，加法结构下“谁更挑剔”对成对比较没有影响，成对数据也因此测不出 $U_i$。

### 多维因子：用户不再消掉

$K$ 维因子模型中用户向量不再消掉，式 (1.38)：

$$
P(j\succ k\mid i)
=
\sigma\big(U_i^{\top}(V_j-V_k)+(Z_j-Z_k)\big).
$$

其中 $U_i\in\mathbb{R}^K$ 是用户口味向量，$V_j\in\mathbb{R}^K$ 是物品属性向量，$Z_j$ 是物品的公共截距。

- $K=1$ 且 $Z_j=0$ 时，点积可以重新参数化成 Rasch 形式。
- $K>1$ 时，恐怖片观众和喜剧观众可以对同一对影片给出相反名次，$U_i$ 必须留在概率里。

$N\times M$ 的矩阵用 $N\times K$ 的用户向量和 $M\times K$ 的物品向量近似，$K\ll\min(N,M)$，没见过的人–物组合才能外推。这是低秩假设带来的泛化能力。

### 噪声项到底代表什么

随机效用写成 $\tilde H_{ij}=f(U_i,V_j)+\varepsilon_{ij}$，即 (1.22)；只有物品时是 $\tilde H_j=V_j+\varepsilon_j$，即 (1.23)。另一套写法把 $H_{ij}$ 固定，随机性放在观测上：$Y_{ij}\sim\mathrm{Bernoulli}(\sigma(H_{ij}))$。加法结构下，两种写法的成对概率可以一样。但 $\varepsilon$ 代表什么，结论就不一样：

- **异质（heterogeneity）**：$\varepsilon$ 是不同决策者之间的差异。$j$ 被选得更多，说明偏好 $j$ 的人更多。
- **有界理性（bounded rationality）**：$\varepsilon$ 是同一个人的决策误差。被选次数高，不能单独说明 $j$ 更好。
- **设计者信念（designer belief）**：$\varepsilon$ 是设计者对 $V$ 的不确定。数据更新先验，结论带着这个先验。

这三种解释与“公平聚合”一章的噪声 / 异质之分是一回事：同一组数据，解释不同，该采取的合并方式也不同。

> [!WARNING]+ IIA 与噪声
>
> 第 1.8 节有一句：独立同分布噪声的随机效用都满足 IIA（Independence of Irrelevant Alternatives）。同节 Theorem 1 写的是：IIA 当且仅当噪声为独立同分布的 Gumbel，分布函数 $F(x)=e^{-e^{-x}}$。两句互相矛盾。反例：标准正态也是独立同分布，得到的是 probit。红巴士 / 蓝巴士里，probit 不保持 $j$ 与 $k$ 的概率比。后文按 Theorem 1。Luce 的式子是 (1.24)：菜单 $\mathcal{S}$ 上 $p(j\mid\mathcal{S})/p(k\mid\mathcal{S})$ 不因多了一个 $\ell$ 而变。

### 识别与罗生门效应

**识别（identification）** 问题是参数里的平移：全体 $V_j$ 加常数 $c$，softmax 不变，(1.36)；一个人的 $H_{ij}$ 加 $c$，成对概率不变，(1.37)。选择数据只定差，不定绝对水平。DPO 的处理是把冻结的 $\pi_{\mathrm{ref}}$ 当作零点，隐式奖励是相对参考策略的对数比。

**罗生门效应（Rashomon effect，Breiman 2001）** 是另一件事：平移已经钉死，结构不同的模型仍可以在同一批比较上损失一样低。书中例子：五个物品、一百次成对比较，单一 Bradley–Terry、两组混合、嵌套 logit 都可以到约 90% 准确率。Rashomon set 收的是损失不超过最优 $(1+\epsilon)$ 倍的所有模型（Semenova, Rudin, Parr 2022）。对奖励学习而言，Skalse 等人（2022）指出，训练比较上拟合相当的奖励，优化出来的策略可以不同。这意味着“训练损失低”不足以选定奖励模型，还要看它们在下游策略上的差异。

<br>

## 第 2 章 学习（Learning）：Elo 与正则化

比较可以一次到齐，也可以一场一场来。到齐时做极大似然或贝叶斯估计；一场一场来时，Elo 只改刚比过的两件物品，是一种在线算法。

### Elo 是 Bradley–Terry 的随机梯度上升

Elo 是 Bradley–Terry 对数似然的随机梯度上升（stochastic gradient ascent）。设当前胜率 $p=\sigma(V_j-V_{j'})$，结果 $y\in\{0,1\}$（$j$ 胜为 1）。单场对数似然 $y\log p+(1-y)\log(1-p)$ 对 $V_j$ 的梯度 (2.14) 正是预测误差 $y-p$，于是更新 (2.15) 为

$$
V_j\leftarrow V_j+\eta(y-p),
\qquad
V_{j'}\leftarrow V_{j'}-\eta(y-p).
$$

棋界惯例里，学习率 $\eta$ 叫 $K$-factor，sigmoid 的自变量是分差除以 400（严格说国际象棋 Elo 用的是 $1/(1+10^{-\Delta/400})$，即以 10 为底；书中例子按自然底数的 $\sigma$ 计算，下面照书上数字）。

> [!EXAMPLE]+ 例3 一场 Elo 更新
>
> $V_A=1500$，$V_B=1700$，$K=32$。分差 $-200$，除以 400 得 $-0.5$：
>
> $$
> p=\sigma(-0.5)=\frac{1}{1+e^{0.5}}\approx 0.3775.
> $$
>
> 若 $A$ 胜（爆冷），$y-p\approx 0.6225$，两边各动 $32\times 0.6225\approx 19.92$ 分，$V_A\approx 1519.9$，$V_B\approx 1680.1$。
>
> 若 $B$ 胜（符合预期），$y-p\approx -0.3775$，变动 $32\times 0.3775\approx 12.08$ 分。
>
> 预测误差大，步子大；结果越出乎意料，分数调整越多。

Elo 的局限：$K$ 人人相同，旧比赛不折扣，也没有置信区间。Glicko 给每个人另存一个不确定度，闲置越久越大，更新幅度跟着放大。

### 正则化：$L_2$ 与早停

$L_2$ 目标 (2.16) 是对数似然减去 $(\lambda/2)\|V\|_2^2$。这是先验 $V_j\sim N(0,1/\lambda)$ 下的 MAP（最大后验）估计，$\lambda$ 是先验精度（方差的倒数）。梯度比极大似然多一项 $-\lambda V_m$，即 (2.17)。它的另一个作用是：即使比较图不强连通，正则化目标仍有有限的最优解。

早停（early stopping）不写 $\lambda$：验证集上的对数似然开始变差就停。它要留出一折比较。交叉验证切开的也是**比较**，不是物品。若按物品切，测试折里会出现训练里没见过的物品，其截距从未被估计过，测出来的误差没有意义。

<br>

## 第 3 章 行动（Action）：遗憾、胜者概念与采集函数

第 2 章的实验设计挑**信息量大**的查询；第 3 章挑**当下期望奖励高**的动作，探索来自 Thompson 采样：从后验抽一个参数，再按这个参数贪心。这是“学习”与“行动”的区别。

效用若随时间变，本章的遗憾上界用不上。旧比较要折扣，或只留滑动窗口。固定 $K$ 的 Elo 不折扣，当前分数里还留着全部历史。

### 强遗憾与弱遗憾

对决赌博机（dueling bandits）每轮亮出两臂 $b_1,b_2$ 让人比较。设 $b^\star$ 是 Condorcet 胜者，定义 $\epsilon(b^\star,b)=P(b^\star\succ b)-1/2\ge 0$，衡量臂 $b$ 比最优臂差多少。

**强遗憾（strong regret）** 对较差的那一臂收费：

$$
R_T=\sum_{t=1}^{T}\max\big\{\epsilon(b^\star,b_1^{(t)}),\epsilon(b^\star,b_2^{(t)})\big\}.
$$

**弱遗憾（weak regret）** 对较好的那一臂收费，把 $\max$ 换成 $\min$。弱遗憾只要求两臂里有一个是好的，适合“用户只用到较好那个”的场景。前文对决优化一章用的瞬时遗憾是两臂 $\epsilon$ 之和，等于强、弱两者相加。

Interleaved Filter（Yue et al., 2012）对两种遗憾都有次线性上界，在时限 $T$ 内以至少 $1-1/T$ 的概率返回 Condorcet 胜者。返回错了，遗憾可以到 $O(T)$；但这件事的概率只有 $1/T$，期望贡献仍是常数量级。

### 事后遗憾与贝叶斯遗憾

事后遗憾（ex-post regret）在真实均值 $\mu$ 给定时计算，(3.39)：

$$
\mathrm{Regret}(\mu)=T\max_i\mu_i-\sum_{t=1}^{T}\mathbb{E}[\mu_{I_t}].
$$

$I_t$ 是第 $t$ 轮选的臂，期望对算法的随机性取。贝叶斯遗憾（Bayesian regret）再对 $\mu$ 的先验取期望，(3.40)。事后遗憾只依赖 $\mu$；贝叶斯遗憾还依赖先验，Thompson 采样的分析通常用后者。

### 三种“最好”：Condorcet、Borda 与 von Neumann 胜者

**Condorcet 胜者**对每个其他臂的胜率都大于 $1/2$。偏好循环时可以没有。书里的偏好矩阵用 $\Delta_{jk}=P(j\succ k)-1/2$，若没有一行对所有列都为正，就没有 Condorcet 胜者。

**Borda 胜者**是 $\arg\max_j\sum_{i\neq j}P(j\succ i)$，即平均胜率最高的臂。它可以有多个并列，但循环时仍然存在。它和 Condorcet 胜者可以不是同一个臂：一种披萨在两两比较里都过半，另一种在全体投票里票更多（与“社会选择”一章七人剖面里相对多数和 Condorcet 分歧是同一类现象）。

**von Neumann 胜者**是把偏好矩阵看成混合策略零和博弈时，对最坏对手最稳的那个混合策略。它总是存在。Feel-good Thompson Sampling（Zhang 2021）对着这个目标，遗憾 (3.44) 写成最优奖励减去本轮两臂真实奖励的平均。

要点：算法用哪一种胜者，遗憾就在优化哪一种“最好”。选算法之前先选胜者概念。

### qEUBO：批量查询的采集函数

回答无噪声时，一次查询可以给出 $q\ge 2$ 个点让人挑最好的。qEUBO（Astudillo et al., 2023）是这 $q$ 个点上潜在函数最大值的后验期望，(3.53)：

$$
\mathrm{qEUBO}_n(X)
=
\mathbb{E}_n\big[\max\{g(x_1),\ldots,g(x_q)\}\big].
$$

这里 $X=(x_1,\ldots,x_q)$ 是一次查询的 $q$ 个候选，$g$ 是潜在效用函数，$\mathbb{E}_n$ 是对前 $n$ 次回答后的后验取期望。

无噪声时，最大化 qEUBO 的查询也最大化一步贝叶斯价值 $V_n$，(3.56)。$V_n$ 里面还有一层对下一步后验的优化，直接算很贵；qEUBO 只要一个期望最大值，便宜得多。回答有噪声之后，这个包含关系不再成立，qEUBO 只是近似。

### GRPO：组内相对优势

GRPO（Group Relative Policy Optimization，Shao et al., 2024）不用 PPO 的价值网络（critic）。同一条提示抽出 $G$ 个回复，优势是组内奖励的 $z$ 分数，(3.74)：

$$
\hat A_i=\frac{r_i-\mathrm{mean}(\{r_j\}\_{j=1}^{G})}{\mathrm{std}(\{r_j\}\_{j=1}^{G})}.
$$

组内均值起到基线（baseline）的作用，替代了价值网络的估计。更新仍是截断的重要性比 $\rho_i=\pi_\theta(y_i\mid x)/\pi_{\theta_{\mathrm{old}}}(y_i\mid x)$，再减去对 $\pi_{\mathrm{ref}}$ 的 KL，(3.75)。

常见误解：$\hat A_i>0$ 并不意味着回复“好”，只意味着比同组平均好。组里分数都高时，只有最高的几条 $\hat A_i>0$；组里分数都低时，中等回复的优势也可以为正。

<br>

## 第 4 章 反演（Inversion）：从行为反推偏好的陷阱

### 情境变量不能丢

前面各章都把观察到的选择当作偏好的直接读数。反演一章强调这一步并不安全。把行为 $B$、心理状态 $M$（真正的偏好）、情境 $C$（时间、疲劳、语言等）分开写，(4.1) 是

$$
P(B\mid M,C)\neq P(B\mid M).
$$

丢掉 $C$ 之后得到的是另一个条件分布，把情境的影响错算到偏好头上。例如评语均长 800 词和 400 词，差别可以只是日程空不空。主动学习若按长度抽样，查询和助手都会偏向已经有时间写长评的人；短评的人得到更差的助手，之后留下的比较更少，形成反馈循环。

### 四个阶段，各有一个选择

- **征询（elicitation）**：决定问谁。
- **学习**：决定哪一种模型结构合法；例如 IIA 把随情境变化的替代关系排除在外。
- **聚合**：决定谁的比较权重大；DPO 默认按样本量加权。
- **决策（decision）**：决定何时照个人偏好做，何时覆盖它。

### 自由协助与不自由协助

**自由协助（liberal assistance）** 照“不多管闲事”的偏好做：偏好的对象是自己的结果（non-nosy）。**不自由协助（illiberal assistance）** 执行关于他人的偏好（nosy），或在以下情况覆盖个人偏好：伤害第三人、偏好形成时条件很差、存在集体行动问题。覆盖要写明理由，并留申诉渠道。语言模型照做请求，是自由协助；拒绝有害请求，是不自由协助。

### 显示偏好对不上真实偏好的四类情形

显示偏好（revealed preference）把“选了 $x$ 而不是 $y$”读成“偏好 $x$”。书里举了四类对不上的情形：

1. 习惯还在，偏好已经变了；
2. 疲劳和分心使选择偏离；
3. 同一偏好在不同情境里表现为不同行为；
4. 选择是策略性的。

例子：智能食品柜看见每次都拿 Doritos，记成偏好 Doritos；人可能只是手边有、习惯伸手。医生下午的处方随疲劳变化，把全天决策同等对待，会把疲劳写进“偏好”。

RLHF 里的对应：标注做久了质量下降；点击和观看时长不是满意度；同一标注者随心情和前后例子改口；标注者若觉得某种模式对自己有利，会改标法（对应 Gibbard–Satterthwaite 意义下的策略性报告）。

<br>

## 第 5 章 聚合（Aggregation）：Borda、DPO 与绕开不可能定理的办法

Arrow 之外，书把 IIA 放宽成 IIA'。Borda 这时满足无限制域、帕累托、非独裁和这条较弱的独立性。再加采样假设，DPO 最大化的是加权 Borda 分。本章还给出单峰域、可分域上的正面结果，Sen 的自由悖论，以及 Community Notes、情境完整性与拍卖等应用。

### Borda 分数与 IIA'

选项 $y$ 的 Borda 分是它在各投票人排序中压过的其他选项个数之和，(5.1)：

$$
\mathrm{Borda}(y)
=
\sum_{i=1}^{n}\big|\{y'\neq y:y\succ_i y'\}\big|
=
\sum_{y'\neq y}\big|\{i:y\succ_i y'\}\big|.
$$

第一个等号按投票人数：投票人 $i$ 把 $y$ 排在多少个选项之前；第二个等号交换求和顺序，按对手数：每个对手 $y'$ 有多少人把 $y$ 放在它前面。两种写法计数的是同一组 $(i,y')$。$m$ 个选项时，名次分 $m-1,\ldots,0$ 就是压过的选项数，与“社会选择”一章的“第一名 2 分、第二名 1 分”一致。

**IIA'（modified IIA）**：固定两个选项 $y,y'$。两个剖面里，若每个投票人对 $y$ 与 $y'$ 的相对名次相同，并且两者之间隔着的选项个数也相同，社会选择不能在一个剖面上选 $y$、在另一个上选 $y'$。也就是说，名次相距多远可以影响结果。Borda 满足 IIA'，但仍违反原来的 IIA（“社会选择”一章例 1 中，两个剖面里隔着的选项个数不同，所以 IIA' 不约束它们）。

**定理（书中）**：Borda 计数满足无限制域、帕累托、非独裁与 IIA'。放宽 IIA 到 IIA' 正是 Borda 绕开 Arrow 的方式。

### DPO 在优化加权 Borda 分

比较从参考策略抽出时，$\pi_{\mathrm{ref}}$ 加权的 Borda 胜者是 (5.2)：

$$
y^\star
=
\arg\max_y
\mathbb{E}\_{y'\sim\pi_{\mathrm{ref}}(\cdot\mid x)}
\big[\mathbf{1}[y\succ y'\mid x]\big].
$$

这是“对参考策略随机抽出的对手的胜率”，即把普通 Borda 中“对每个对手等权”换成“按 $\pi_{\mathrm{ref}}$ 加权”。DPO 最优策略与 $\pi_{\mathrm{ref}}$ 的密度比，正比于这个分数：对参考分布抽出的对手胜率越高，策略给它的概率相对 $\pi_{\mathrm{ref}}$ 越大。

书中列出的三条前提是：回复从参考策略抽出、成对按参考策略均匀采样、Bradley–Terry 设定正确。三条里任何一条不成立，这个对应就断了：

- 抽对不是按 $\pi_{\mathrm{ref}}$ 来的，更长或更有争议的回复被抽得更多，计票规则就变了。
- 标注池之间偏好不同，Borda 分跟着谁在标走。
- 偏好不传递，或随上下文变，Bradley–Terry 设错，目标对不上任何一个 Borda 胜者。

这给“DPO 在优化什么”一个社会选择的答案：它实现的是一种特定的计分投票，因而继承了 Borda 的性质（满足帕累托、违反 IIA、可被策略性标注操纵）。

### 单峰域：中位数与幻影峰

选项在一条线上，每人一个最爱点（ideal point），离它越远越差，是单峰偏好（single-peaked preferences）。Black（1948）：中位投票人的峰是多数决的 Condorcet 胜者。书中例子：五人的峰为 $\{2,4,5,7,9\}$，纯中位数是 $5$。

Moulin（1980）进一步给出防策略的规则族：单峰域上，把真实峰再并上若干固定的**幻影峰（phantom peaks）**，取整体中位数，得到的**广义中位数**同时防策略（strategy-proof）、满足帕累托。幻影峰相当于规则设计者预先放置的“虚拟投票人”，可以把结果往某个默认值拉。书中习题（5.17 节）的设定：峰为 $\{1,3,5,7,9\}$，再加幻影峰 $2$ 与 $8$；书里没有给出答案，按定义计算，七个数 $\{1,2,3,5,7,8,9\}$ 的中位数仍是 $5$。

为什么防策略？峰在中位数左侧的人谎报一个更靠右的峰，只会把中位数往右推得更远离自己；谎报更靠左的峰则不改变中位数。右侧同理。域不再单峰时，Gibbard–Satterthwaite 把防策略推回独裁。

### 可分偏好与按委员会投票

可分偏好（separable preferences）：把物件 $x$ 加进任何一个集合 $A$ 是否变好，只看 $x$ 单独是否好过空集，(5.3)。用书中写法，记 $G(\succ)=\{x:\{x\}\succ\emptyset\}$ 为“好物件”集合，则 $A\cup\{x\}\succ A$ 当且仅当 $x\in G(\succ)$。

应用到对齐：若有用、无害、诚实（helpful, harmless, honest）三个议题可分，就可以按议题分别聚合。但现实往往不可分：一条很有用的回复必然带一点伤害，两个议题绑在一起，分议题投票会把这种依赖丢掉。

可分域上，满射、防策略、可分三条刻画的是**按委员会投票（voting by committees）**：对每个物件 $x$ 预先规定一族“获胜联盟” $W_x$，把 $x$ 放进自己最优集合的那些投票人若构成获胜联盟，$x$ 就入选。这条规则一般达不到帕累托。

### Sen 的帕累托自由人悖论

Sen（1970）的帕累托自由人（Paretian liberal）有三条：

- **minimal liberalism**：每人在自己的私事上至少对一对选项说了算；
- **Pareto**：全体都偏好 $x$ 甚于 $y$ 时社会也如此；
- **unrestricted domain**：任何偏好剖面都允许。

三条不能一起成立。

> [!EXAMPLE]+ 例4 拘谨者与随便者
>
> 两人、一本书。$a$ 是拘谨者（Prude）读，$b$ 是随便者（Lewd）读，$c$ 是没人读。拘谨者的偏好是 $c\succ a\succ b$（最好谁都别读，非要读就自己读，以免随便者读了更堕落）；随便者的偏好是 $a\succ b\succ c$（最想让拘谨者读，其次自己读，最差是没人读）。
>
> - 私事上，拘谨者决定自己读不读：在 $a$ 与 $c$ 之间他说了算，使 $c$ 压过 $a$。
> - 随便者决定自己读不读：在 $b$ 与 $c$ 之间他说了算，使 $b$ 压过 $c$。
> - 两人都觉得 $a$ 好于 $b$，帕累托使 $a$ 压过 $b$。
>
> $c$ 压 $a$、$a$ 压 $b$、$b$ 压 $c$，接成循环，没有一致的社会选择。

问题根源在于两人的偏好都落在**别人**的结果上（拘谨者在乎随便者读什么，反之亦然），这正是上文第 4 章（反演）所说的 nosy preference。内容审核里，一方要表达、一方要环境安全，偏好落在别人的结果上，也是 nosy preference，这也是审核规则难以同时满足“尊重个人”和“尊重全体一致”的原因。

### Community Notes：用因子模型分离质量与立场

X（原 Twitter）的 Community Notes 用因子模型把“注释写得好”和“评分人与注释立场相近”拆开，(5.4)：

$$
u=\mu+\alpha+\beta+p^{\top}q+\varepsilon.
$$

- $\mu$ 是总截距；
- $\alpha$ 是评分人偏严还是偏宽；
- $\beta$ 是注释本身的质量；
- $p^{\top}q$ 是评分人（$p$）与注释（$q$）在立场上的对齐；
- $\varepsilon$ 是噪声。

入选看的是控制对齐之后 $\beta$ 是否过阈值。极化注释会被盟友打高、被对手打低；朴素平均把立场混进质量，内积项把这一部分从 $\beta$ 里分开。只有跨立场都得到好评的注释，才会有高 $\beta$。形式和带因子的 Rasch 相同，多出来的是：要估计并据以决策的那一项是 $\beta$，不是票数。

### 情境完整性与诚实报价机制

**Nissenbaum 的情境完整性（contextual integrity）**：信息流动符合该场景的规范，隐私才算保住。五个参数是发送者（sender）、信息主体（subject）、接收者（recipient）、数据类型（data type）、传递原则（transmission principle）。心率交给教练用来调训练，和交给广告网络，数据类型相同，传递原则不同。用户点过同意，流动仍可以违反这个场景的规范。对偏好数据而言，标注数据被挪作他用同样要按这五个参数审视。

**维克里拍卖（Vickrey auction，第二价格密封拍卖）** 里，赢的人付第二高的报价。自己的报价只决定赢或输，不决定付多少，如实报价是占优策略（dominant strategy），机制是 DSIC（dominant-strategy incentive compatible）。例如报价 $(2,6,4,1)$，第二位以 6 胜出，支付 4。

为什么如实报价占优：设真实估值为 $v$，其他人的最高报价为 $b$。报高于 $v$ 只会在 $b\in(v,\text{报价})$ 时多赢，而此时付 $b>v$，亏损；报低于 $v$ 只会在 $b\in(\text{报价},v)$ 时少赢，丢掉 $v-b>0$ 的收益。

第一价格拍卖里，赢的人付自己的报价，均衡里会把报价压到真实价值下面，称为 bid shading。配置仍可能给估价最高的人，但要靠策略推理，不是 DSIC。这与 Gibbard–Satterthwaite 形成对照：引入金钱转移（拟线性效用）之后，防策略不再意味着独裁。

<br>

## 附录：记号约定

附录里的符号和正文一致：

- $N$ 是投票人集合，$A$ 或 $Y$ 是选项集合，$\mathcal{L}(A)$ 是 $A$ 上严格全序的集合；
- $F$ 输出社会全序（社会排序函数），$f$ 输出一个赢家（社会选择函数）；
- $\sigma$ 是 logistic 函数；
- $\pi_{\mathrm{ref}}$ 是冻结的参考策略。

> [!NOTE]+ 小结
>
> - 教材五章：基础、学习、行动、反演、聚合，十周季度中分别约 1、3、3、1.5、1.5 周。
> - 引言：对数评分严格恰当，所以预训练复制语料；后训练换数据，DPO 的隐式奖励代入后就是 BT。
> - 基础：Rasch 的 $U_i$ 在成对比较中消掉得到 BT，$K>1$ 因子不消；噪声的三种解释导致不同结论；教材 1.8 节关于 IIA 的一句与 Theorem 1 矛盾，以 Theorem 1 为准。
> - 学习与行动：Elo 是 BT 的 SGD；强 / 弱遗憾、三种胜者概念、qEUBO、GRPO 的组内 $z$ 分数优势。
> - 聚合：Borda 满足 IIA'；DPO 在三条前提下优化 $\pi_{\mathrm{ref}}$ 加权 Borda 分；单峰域广义中位数防策略；Sen 悖论源于 nosy preference。

<br>


# 复习：公式速查、综合例题与讲次对照

复习部分把全课最常用的公式和数字结论集中放在一起，方便考前或做项目时查找；随后用一组三物品比较把估计、信息与损失串成一道综合题；最后给出各讲与笔记章节的对照表。每条公式的推导都在对应章节，这里只保留结论和最容易记错的细节。

<br>

## 公式与结论速查

### 选择模型与估计

- **Bradley–Terry**：$P(i\succ j)=\sigma(s_i-s_j)$。对数似然的梯度是“胜场减去期望胜场”。参照分数要钉死（如 $s_A=0$），否则平移不可识别，Hessian 奇异。
- **Gumbel**：独立标准 Gumbel 噪声推出 softmax。Gumbel 均值里的欧拉常数 $\gamma$ 在效用差里消失。
- **Luce**：正概率下，菜单上的概率比等于成对概率比（选择公理），等价于 softmax。
- **嵌套 logit**：相似度参数 $\lambda\in(0,1]$。$\lambda=1/2$、两辆效用为 0 的巴士对一辆效用为 0 的汽车时，$P(\mathrm{car})=\sqrt{2}-1$（巴士巢的包含值为 $\lambda\log 2$，故 $P(\mathrm{car})=1/(\sqrt2+1)=\sqrt2-1\approx 0.414$，介于 IIA 给出的 $1/3$ 与“两巴士视为一辆”的 $1/2$ 之间）。
- **条件 logit 梯度**：选中项的特征减去期望特征。
- **存在性**：BT 的有限 MLE 当且仅当胜利有向图强连通。某物品胜场 $W_i=0$ 时 MM 更新无定义（分数会被推向 $-\infty$）。
- **Rasch**：$H_{ij}=U_i+V_j$。同一人的成对比较里 $U_i$ 消掉，得到 Bradley–Terry；$K$ 维点积不消掉。
- **Elo**：$\sigma(-0.5)\approx 0.3775$。$K=32$ 时爆冷约 $+19.92$ 分，预期结果约 $12.08$ 分。

### 信息与主动查询

- **Fisher 信息**：一次比较的信息是 $p(1-p)xx^{\top}$（$x$ 为两物品的特征差）。$p=1/2$ 时标量因子最大，为 $1/4$。
- **BALD**：边缘熵减去后验样本的平均熵。两个后验样本给出 $0.2$ 与 $0.8$ 时约为 $0.193$（以自然对数计：$\ln 2-H(0.2)\approx 0.693-0.500$）；两人都给出 $0.5$ 时为 $0$，因为没有分歧，回答无法区分它们。
- **qEUBO**：无噪声时，最大化查询内潜在值最大值的后验期望，也最大化一步贝叶斯价值。

### 策略优化

- **KL 正则最优策略**：$\pi^\star\propto\pi_{\mathrm{ref}}e^{r/\beta}$。
- **DPO**：$u=\beta(\Delta_\theta-\Delta_{\mathrm{ref}})$，损失 $-\log\sigma(u)$，梯度权重 $\sigma(-u)$（模型越把这一对判错，权重越大）。
- **best-of-$n$**：诱导密度不超过 $n$ 倍参考密度，故 KL 不超过 $\log n$。
- **GRPO**：优势是组内奖励的 $z$ 分数，更新是截断重要性比减去对参考策略的 KL。
- **Boltzmann 理性**：$\beta$ 与效用尺度不可同时从选择里识别。$\beta=\infty$ 时完全理性，给次优动作零概率。

### 赌博机

- **对决遗憾**：两臂相对 Condorcet 胜者的 $\epsilon$ 之和。没有 Condorcet 胜者时先换目标（Borda 或 von Neumann 胜者）。
- **强弱遗憾**：强遗憾取两臂 $\epsilon$ 的最大，弱遗憾取最小。两者之和才是两臂都计入。

### 社会选择与聚合

- **Arrow**：至少三选项时，全域、弱帕累托、IIA、非独裁不能同时成立。
- **相对多数**：七人票里 $A$ 靠第一名当选，$B$ 是 Condorcet 胜者且 Borda 分为 $9$。
- **陪审团**：$p=0.6$、$n=3$ 时多数正确概率为 $0.648$，$n=5$ 时为 $0.6826$，且要求独立与共同正确答案。
- **反演**：$P(B\mid M,C)\neq P(B\mid M)$。行为把心理状态和情境绑在一起。
- **Borda–DPO**：比较从 $\pi_{\mathrm{ref}}$ 抽出且 Bradley–Terry 设对时，DPO 最优策略与参考策略的密度比正比于加权 Borda 分。
- **单峰**：峰 $\{2,4,5,7,9\}$ 的纯中位数是 $5$。

<br>

## 综合例题：三物品比较从估计到 DPO

用同一组三物品比较把估计、信息与损失串起来。数据仍是选择数据一章的那组：

| 对 | 胜 | 负 |
| :--- | ---: | ---: |
| $A$ 对 $B$ | $A$ 胜 3 次 | $B$ 胜 1 次 |
| $A$ 对 $C$ | $A$ 胜 4 次 | $C$ 胜 1 次 |
| $B$ 对 $C$ | $B$ 胜 3 次 | $C$ 胜 1 次 |

1. **存在性。** 胜利有向图每一对都有双向边（每对双方都至少赢过一次），因而强连通，MLE 有限。
2. **估计。** 钉住 $s_A=0$，牛顿法或 MM 都得到 $(s_B,s_C)=(-0.8361,\ -1.6723)$。负对数似然为 $7.085$。可以注意到 $s_C\approx 2s_B$，三物品分数几乎等间距。
3. **下一次该问哪一对。** 拟合概率 $\hat P(A\succ C)=0.842$（即 $\sigma(1.6723)$），其 $p(1-p)=0.133$，低于另外两对（分差都约 $0.836$，$\hat p\approx 0.698$）的 $0.211$。按 $\log\det$ 准则，再标一次时 $A$ 对 $C$ 的信息增量最小：模型对这一对已经很有把握，问它学到的最少。
4. **不确定度。** 两个自由坐标的 Laplace 标准差约为 $0.91$ 与 $0.96$，与分数本身同一量级。把 $s_C$ 的点估计当成“已经知道 $A$ 远好于 $C$”会高估把握；只有 13 次比较，区间仍然很宽。
5. **与 DPO 的尺度衔接。** 前文 DPO 例子里隐式奖励差就是 $u=0.10986$，已经包含 $\beta$。BT 概率是 $\sigma(u)$，不必再把 $u$ 除以 $\beta$。若把估计好的分数 $s$ 乘上另一个 $\beta$ 再送进 $\sigma(\beta(s_i-s_j))$，等于改了尺度；那一步与 DPO 的代换不是同一个 $\beta$。常见错误正是把两处 $\beta$ 混为一谈，导致概率被重复放缩。

<br>

## 讲次与笔记章节对照

| 讲 | 题目 | 笔记位置 |
| :--- | :--- | :--- |
| 01 | 引言 | 导论 |
| 02 | 选择数据、泛化、Bradley–Terry | 选择数据 |
| 03 | 随机效用与公理 | 随机效用 |
| 04 | 因子模型 | 因子模型 |
| 05 | 预测选择、潜变量极大似然 | 选择预测 |
| 06 | 贝叶斯估计与微调 | 贝叶斯估计 |
| 07 | Fisher 信息与实验设计 | 实验设计 |
| 08 | RLHF | 人类反馈 |
| 09 | DPO | 直接偏好 |
| 10 | 语言模型的主动征询 | 主动征询 |
| 11 | 辅助博弈与情境赌博机 | 辅助博弈 |
| 12 | 采集函数与 Thompson 采样 | 采集采样 |
| 13 | 对决赌博机与偏好贝叶斯优化 | 对决优化 |
| 14 | 风格控制与代理 | 风格代理 |
| 15 | 有界理性与可计算选择 | 有界理性 |
| 16 | Arrow 与 Gibbard–Satterthwaite | 社会选择 |
| 17 | 公平与信息聚合 | 公平聚合 |
| 18 | 小结与实践 | 实践、复习 |

表中“笔记位置”指本笔记中讨论该主题的章（按主题名称对应）。按 2026 秋季课表，第 1 讲是 9 月 23 日，第 18 讲是 11 月 30 日；12 月 2 日因 NeurIPS 停课。

<br>


# 参考文献

以下文献按主题分组。课程主页与历年课表给出阅读清单；选择模型部分是前几章的经典来源；偏好优化与强化学习部分对应 RLHF、DPO 与逆强化学习；最后一组覆盖主动征询、对决赌博机与社会选择。

## 课程

1. Stanford CS329H. *Machine Learning from Human Preferences*. 2026 秋季主页。[https://web.stanford.edu/class/cs329h/](https://web.stanford.edu/class/cs329h/)
2. Stanford Bulletin. *CS329H*. [https://bulletin.stanford.edu/courses/2254281](https://bulletin.stanford.edu/courses/2254281)
3. Stanford CS329H. *Autumn 2023* 课表与阅读。[https://web.stanford.edu/class/cs329h/fall2023/](https://web.stanford.edu/class/cs329h/fall2023/)

## 选择模型

1. Luce RD. *Individual Choice Behavior*. Wiley, 1959.
2. Bradley RA, Terry ME. *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*. Biometrika, 1952.
3. Thurstone LL. *A Law of Comparative Judgment*. Psychological Review, 1927.
4. Plackett RL. *The Analysis of Permutations*. Applied Statistics, 1975.
5. McFadden D. *Conditional Logit Analysis of Qualitative Choice Behavior*. In Frontiers in Econometrics, 1974.
6. McFadden D, Train K. *Mixed MNL Models for Discrete Response*. Journal of Applied Econometrics, 2000.
7. Train K. *Discrete Choice Methods with Simulation*. Cambridge University Press.
8. Rasch G. *Probabilistic Models for Some Intelligence and Attainment Tests*. 1960.（人–题的 logistic 与 BT 同形）
9. Ford LR Jr. *Solution of a Ranking Problem from Binary Comparisons*. American Mathematical Monthly, 1957.
10. Hunter DR. *MM Algorithms for Generalized Bradley–Terry Models*. The Annals of Statistics, 2004.
11. Davidson RR. *On Extending the Bradley–Terry Model to Accommodate Ties in Paired Comparison Experiments*. Journal of the American Statistical Association, 1970.

## 偏好优化与强化学习

1. Christiano P et al. *Deep Reinforcement Learning from Human Preferences*. NeurIPS, 2017.
2. Ziegler DM et al. *Fine-Tuning Language Models from Human Preferences*. arXiv, 2019.
3. Ouyang L et al. *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS, 2022.
4. Rafailov R et al. *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*. NeurIPS, 2023.
5. Schulman J et al. *Proximal Policy Optimization Algorithms*. arXiv, 2017.
6. Gao L, Schulman J, Hilton J. *Scaling Laws for Reward Model Overoptimization*. ICML, 2023.
7. Ziebart BD et al. *Maximum Entropy Inverse Reinforcement Learning*. AAAI, 2008.
8. Hadfield-Menell D, Russell SJ, Abbeel P, Dragan A. *Cooperative Inverse Reinforcement Learning*. NeurIPS, 2016.
9. Ng AY, Russell SJ. *Algorithms for Inverse Reinforcement Learning*. ICML, 2000.

## 征询、赌博机与社会选择

1. Yue Y, Broder J, Kleinberg R, Joachims T. *The K-armed Dueling Bandits Problem*. Journal of Computer and System Sciences, 2012.
2. Zoghi M et al. *Relative Upper Confidence Bound for the K-Armed Dueling Bandit Problem*. ICML, 2014.
3. Dudík M et al. *Contextual Dueling Bandits*. COLT, 2015.
4. Novoseller E et al. *Dueling Posterior Sampling for Preference-Based Reinforcement Learning*. UAI, 2020.
5. Chu W, Ghahramani Z. *Preference Learning with Gaussian Processes*. ICML, 2005.
6. González J et al. *Preferential Bayesian Optimization*. ICML, 2017.
7. Houlsby N et al. *Bayesian Active Learning for Classification and Preference Learning*. arXiv, 2011.（BALD）
8. Arrow KJ. *Social Choice and Individual Values*. Wiley, 1951.
9. Gibbard A. *Manipulation of Voting Schemes: A General Result*. Econometrica, 1973.
10. Satterthwaite MA. *Strategy-proofness and Arrow's Conditions*. Journal of Economic Theory, 1975.
11. May KO. *A Set of Independent Necessary and Sufficient Conditions for Simple Majority Decision*. Econometrica, 1952.
12. Black D. *On the Rationale of Group Decision-making*. Journal of Political Economy, 1948.
13. Hiranandani G, Boodaghians S, Mehta R, Koyejo S. *Performance Metric Elicitation from Pairwise Classifier Comparisons*. AISTATS, 2019.
14. Tversky A. *Elimination by Aspects: A Theory of Choice*. Psychological Review, 1972.
15. Bıyık E, Sadigh D. *Batch Active Preference-Based Learning of Reward Functions*. CoRL, 2018.

<br>

# 工具

前文的数值例子大多只需要数组运算即可复现；涉及语言模型时才需要深度学习框架。下面按用途列出，并附上各自最容易出错的地方。

## 数值计算

- [NumPy](https://numpy.org/)：BT 牛顿法、信息矩阵与 DPO 标量损失都可以按上文直接跑。
- 逻辑回归与牛顿法的形状约定与 CS229 笔记相同：样本在第 0 维，参照类别或参照物品要钉住（否则 Hessian 奇异，牛顿步无法求解）。

## 偏好学习

- PyTorch：用于奖励头与 DPO 的词元对数概率。参考模型要用 `torch.no_grad()` 或 `detach`，否则 $\Delta_{\mathrm{ref}}$ 会把梯度送进 $\pi_{\mathrm{ref}}$，参考策略不再冻结，DPO 的零点跟着漂移。
- 高斯过程偏好：后验是非高斯的（似然是 probit 或 logistic），Laplace 或期望传播（expectation propagation）是标配近似；直接对分类似然做共轭更新是错的，因为它与高斯先验不共轭。

<br>
