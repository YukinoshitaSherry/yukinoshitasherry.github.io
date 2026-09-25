# 教材

书在 [mlhp.stanford.edu](https://mlhp.stanford.edu/)，Sang T. Truong、Andreas Haupt、Sanmi Koyejo，页眉日期 2026-09-23。侧栏八块，不是 `chap1` 到 `chap8` 八个编号章。

- 引言，[首页](https://mlhp.stanford.edu/)
- 1 Foundations，[chap1.html](https://mlhp.stanford.edu/src/chap1.html)
- 2 Learning，[chap2.html](https://mlhp.stanford.edu/src/chap2.html)
- 3 Action，[chap3.html](https://mlhp.stanford.edu/src/chap3.html)
- 4 Inversion，[chap4.html](https://mlhp.stanford.edu/src/chap4.html)
- 5 Aggregation，[chap5.html](https://mlhp.stanford.edu/src/chap5.html)
- 结语 Conclusion，附录 Appendices（Mathematical Notation、Glossary），链在侧栏

十分制季度的课时写在引言页：第 1 章约一周，第 2、3 章各约三周，第 4、5 章各约一周半。前文已经推过的 Bradley–Terry 似然、Gumbel、Luce、DPO 损失和 Arrow 四条，这里只补书上的编号、数字和还没写开的定义。

<br>

## 引言

偏好学习出现的地方，书里列了五处：recommender systems（推荐）、information retrieval（检索）、robotics（机器人轨迹）、language model alignment（语言模型对齐）、ranking（排序）。贯穿例子是语言模型。

预训练（pretraining）最小化下一词交叉熵，(1.1)：

$$
\mathcal{L}_{\mathrm{pretrain}}
=
-\mathbb{E}_{x\sim\mathcal{D}}
\left[\sum_{t=1}^{T}\log\pi_\theta(x_t\mid x_{<t})\right].
$$

对数评分（logarithmic scoring rule）$S(p,y)=\log p(y)$ 是严格恰当的（strictly proper）。真实分布为 $q$ 时，期望得分 $\mathbb{E}_{y\sim q}[S(p,y)]$ 只在预报 $p=q$ 时最大。交叉熵因此把语料里的频率拉向模型概率。能续写，不表示会停在有害续写前面。

后训练（posttraining，书里也叫 alignment）换数据：同一提示 $x$ 下，标注者指出 $(y_1,y_2)$ 里哪条更好。RLHF 先拟合奖励 $r(x,y)$，再在靠近原模型的约束下把策略往奖励上推。DPO 跳过单独的奖励网络，损失就是 (1.2)，和前文的 $u=\beta(\Delta_\theta-\Delta_{\mathrm{ref}})$ 是同一个式子。把隐式奖励定义成 $r^\star(x,y)=\beta\log(\pi_\theta(y\mid x)/\pi_{\mathrm{ref}}(y\mid x))$ 之后，(1.3) 就是 Bradley–Terry。

书里用 $V_A=2$，$V_B=1$，$V_C=0$ 算了一遍。$\sigma(1)\approx 0.731$，$\sigma(2)\approx 0.881$。全排序 $A\succ B\succ C$ 的 Plackett–Luce 概率是

$$
\frac{e^{2}}{e^{2}+e+1}\cdot\frac{e}{e+1}\approx 0.665\times 0.731\approx 0.486.
$$

$p(A\succ B)$ 只依赖 $V_A-V_B$，去掉 $C$ 这个概率不变。这就是这条模型里的 IIA。前文用的效用是 $(0,1,2)$，全体加一个常数，概率相同。

<br>

## 基础

逐项反应（item-wise）记 $Y_{ij}\in\{0,1\}$：用户 $i$ 接受或拒绝物品 $j$。$N$ 个人、$M$ 件物品排成反应矩阵（response matrix），(1.5)。购买、跳过一首歌、向右滑、标注者是否标出某条内容，都是这种矩阵。成对比较（pairwise）是两件物品里谁赢，语言模型后训练用的是这一种。

### Rasch

Rasch 把潜在分数写成胃口加吸引力（appetite + appeal）：

$$
H_{ij}=U_i+V_j,
\qquad
P(Y_{ij}=1)=\sigma(U_i+V_j).
$$

按 $U_i$ 和 $V_j$ 把矩阵排好之后，胃口大的人接受大多数物品，热门物品被大多数人接受，接受和拒绝的分界大致沿 $U_i+V_j$ 的对角线。

同一人比较 $j$ 与 $k$，$U_i$ 在差里消失：

$$
P(j\succ k\mid i)=\sigma(V_j-V_k).
$$

这就是 Bradley–Terry。$K$ 维因子不再消掉用户，(1.38)：

$$
P(j\succ k\mid i)
=
\sigma\big(U_i^{\top}(V_j-V_k)+(Z_j-Z_k)\big).
$$

$K=1$ 且 $Z_j=0$ 时，点积可以重新参数化成 Rasch。$K>1$ 时，恐怖片观众和喜剧观众可以对同一对影片给出相反名次，$U_i$ 必须留在概率里。$N\times M$ 的矩阵用 $N\times K$ 的用户向量和 $M\times K$ 的物品向量近似，$K\ll\min(N,M)$，没见过的人–物组合才能外推。

### 噪声

随机效用写成 $\tilde H_{ij}=f(U_i,V_j)+\varepsilon_{ij}$，(1.22)；只有物品时是 $\tilde H_j=V_j+\varepsilon_j$，(1.23)。另一套写法把 $H_{ij}$ 固定，随机性放在 $Y_{ij}\sim\mathrm{Bernoulli}(\sigma(H_{ij}))$。加法结构下，两种写法的成对概率可以一样。$\varepsilon$ 代表什么，结论不一样。

- 异质（heterogeneity）：$\varepsilon$ 是不同决策者。$j$ 被选得更多，偏好 $j$ 的人更多。
- 有界理性（bounded rationality）：$\varepsilon$ 是同一个人的决策误差。次数高，不能单独说 $j$ 更好。
- 设计者信念（designer belief）：$\varepsilon$ 是设计者对 $V$ 的不确定。数据更新先验，结论带着这个先验。

> [!WARNING]+ IIA 与噪声
>
> 第 1.8 节有一句：独立同分布噪声的随机效用都满足 IIA（Independence of Irrelevant Alternatives）。同节 Theorem 1 写的是：IIA 当且仅当噪声为独立同分布的 Gumbel，分布函数 $F(x)=e^{-e^{-x}}$。标准正态也是独立同分布，得到的是 probit。红巴士 / 蓝巴士里，probit 不保持 $j$ 与 $k$ 的概率比。后文按 Theorem 1。Luce 的式子是 (1.24)：菜单 $\mathcal{S}$ 上 $p(j\mid\mathcal{S})/p(k\mid\mathcal{S})$ 不因多了一个 $\ell$ 而变。

### 识别

识别（identification）是参数里的平移。全体 $V_j$ 加常数 $c$，softmax 不变，(1.36)。一个人的 $H_{ij}$ 加 $c$，成对概率不变，(1.37)。选择数据只定差。DPO 把冻结的 $\pi_{\mathrm{ref}}$ 当作零点，隐式奖励是相对参考策略的对数比。

罗生门（Rashomon effect，Breiman 2001）是另一件事：平移已经钉死，结构不同的模型仍可以在同一批比较上损失一样低。五个物品、一百次成对比较，单一 Bradley–Terry、两组混合、嵌套 logit 都可以到约 90% 准确率。Rashomon set 收的是损失不超过最优 $(1+\epsilon)$ 倍的模型（Semenova, Rudin, Parr 2022）。Skalse 等人（2022）指出，训练比较上拟合相当的奖励，优化出来的策略可以不同。

<br>

## 学习

比较可以一次到齐，也可以一场一场来。到齐时做极大似然或贝叶斯；一场一场来时，Elo 只改刚比过的两件物品。

### Elo

Elo 是 Bradley–Terry 对数似然的随机梯度上升（stochastic gradient ascent）。当前胜率 $p=\sigma(V_j-V_{j'})$，结果 $y\in\{0,1\}$。梯度 (2.14) 是预测误差，更新 (2.15) 为

$$
V_j\leftarrow V_j+\eta(y-p),
\qquad
V_{j'}\leftarrow V_{j'}-\eta(y-p).
$$

棋谱里 $\eta$ 叫 $K$-factor，sigmoid 的自变量是分差除以 400。$V_A=1500$，$V_B=1700$，$K=32$：

$$
p=\sigma(-0.5)=\frac{1}{1+e^{0.5}}\approx 0.3775.
$$

$A$ 胜，$y-p\approx 0.6225$，两边各动 $32\times 0.6225\approx 19.92$ 分，$V_A\approx 1519.9$，$V_B\approx 1680.1$。$B$ 胜，变动 $32\times 0.3775\approx 12.08$ 分。预测误差大，步子大。$K$ 人人相同，旧比赛不折扣，也没有区间。Glicko 给每个人另存不确定度，闲置越久越大，更新跟着放大。

### 正则

$L_2$ 目标 (2.16) 是对数似然减去 $(\lambda/2)\|V\|_2^2$。这是先验 $V_j\sim N(0,1/\lambda)$ 下的 MAP，$\lambda$ 是先验精度。梯度比极大似然多一项 $-\lambda V_m$，(2.17)。

早停（early stopping）不写 $\lambda$：验证集上的对数似然开始变差就停。它要留出一折比较。交叉验证切开的也是比较，不是物品。按物品切，测试折里会出现训练里没见过的截距。

<br>

## 行动

第 2 章的实验设计挑信息量大的查询。第 3 章挑当下期望奖励高的动作，探索来自 Thompson 采样：从后验抽一个参数，再按这个参数贪心。效用若随时间变，本章的遗憾上界用不上。旧比较要折扣，或只留滑动窗口。固定 $K$ 的 Elo 不折扣，当前分数里还留着全部历史。

### 遗憾

$b^\star$ 是 Condorcet 胜者，本轮亮出 $b_1,b_2$，$\epsilon(b^\star,b)=P(b^\star\succ b)-1/2$。

强遗憾（strong regret）对较差的那一臂收费：

$$
R_T=\sum_{t=1}^{T}\max\big\{\epsilon(b^\star,b_1^{(t)}),\epsilon(b^\star,b_2^{(t)})\big\}.
$$

弱遗憾（weak regret）对较好的那一臂收费，把 $\max$ 换成 $\min$。前文的瞬时遗憾是两臂 $\epsilon$ 之和，等于两者相加。Interleaved Filter（Yue et al., 2012）对两种都有次线性上界，时限 $T$ 内以至少 $1-1/T$ 的概率返回 Condorcet 胜者。返回错了，遗憾可以到 $O(T)$。

事后遗憾（ex-post regret）在真实均值给定时算，(3.39)：

$$
\mathrm{Regret}(\mu)=T\max_i\mu_i-\sum_{t=1}^{T}\mathbb{E}[\mu_{I_t}].
$$

贝叶斯遗憾（Bayesian regret）再对先验取期望，(3.40)。事后遗憾只依赖 $\mu$。

### 胜者

Condorcet 胜者对每个其他臂的胜率都大于 $1/2$。循环时可以没有。书里的偏好矩阵用 $\Delta_{jk}=P(j\succ k)-1/2$，没有一行对所有列都为正，就没有 Condorcet 胜者。

Borda 胜者是 $\arg\max_j\sum_{i\neq j}P(j\succ i)$，平均胜率最高。并列可以有多个，循环时仍然存在。它和 Condorcet 胜者可以不是同一个臂：一种披萨在两两比较里过半，另一种在全体投票里票更多。

von Neumann 胜者是混合策略零和博弈里对最坏对手最稳的那个。Feel-good Thompson Sampling（Zhang 2021）对着这个目标，遗憾 (3.44) 写成最优奖励减去本轮两臂真实奖励的平均。算法用哪一种胜者，遗憾就在优化哪一种「最好」。

### 采集

回答无噪声时，一次查询可以给出 $q\ge 2$ 个点。qEUBO（Astudillo et al., 2023）是这 $q$ 个点上潜在函数最大值的后验期望，(3.53)：

$$
\mathrm{qEUBO}_n(X)
=
\mathbb{E}_n\big[\max\{g(x_1),\ldots,g(x_q)\}\big].
$$

无噪声时，最大化 qEUBO 的查询也最大化一步贝叶斯价值 $V_n$，(3.56)。$V_n$ 里面还有一层对下一步后验的优化。回答有噪声之后，这个包含不再成立。

### 组内相对

GRPO（Group Relative Policy Optimization，Shao et al., 2024）不用 PPO 的价值网络（critic）。同一条提示抽出 $G$ 个回复，优势是组内奖励的 $z$ 分数，(3.74)：

$$
\hat A_i=\frac{r_i-\mathrm{mean}(\{r_j\}_{j=1}^{G})}{\mathrm{std}(\{r_j\}_{j=1}^{G})}.
$$

更新仍是截断的重要性比 $\rho_i=\pi_\theta(y_i\mid x)/\pi_{\theta_{\mathrm{old}}}(y_i\mid x)$，再减去对 $\pi_{\mathrm{ref}}$ 的 KL，(3.75)。组里分数都高时，只有最高的几条 $\hat A_i>0$。组里分数都低时，中等回复的优势也可以为正。

<br>

## 反演

行为 $B$、心理状态 $M$、情境 $C$（时间、疲劳、语言）分开写。(4.1) 是

$$
P(B\mid M,C)\neq P(B\mid M).
$$

丢掉 $C$ 之后是另一个条件分布。评语均长 800 词和 400 词，可以是日程空不空。主动学习按长度抽样，查询和助手都会偏向已经有时间写长评的人；短评的人得到更差的助手，之后留下的比较更少。

四个阶段各有一个选择。征询（elicitation）决定问谁。学习决定哪一种结构合法，IIA 把随情境变化的替代排除在外。聚合决定谁的比较权重大，DPO 默认按样本量加权。决策（decision）决定何时照个人偏好做，何时覆盖。

自由协助（liberal assistance）照非多管闲事的偏好做：偏好的对象是自己的结果（non-nosy）。不自由协助（illiberal assistance）执行关于他人的偏好（nosy），或在伤害第三人、偏好形成时条件很差、存在集体行动问题时覆盖。覆盖要写明理由，并留申诉。语言模型照做请求，是自由协助。拒绝有害请求，是不自由协助。

显示偏好（revealed preference）把「选了 $x$ 而不是 $y$」读成「偏好 $x$」。书里举了四类对不上的情形：习惯还在、偏好已经变了；疲劳和分心使选择偏离；同一偏好在不同情境里行为不同；选择是策略性的。智能食品柜看见每次都拿 Doritos，记成偏好 Doritos；人可能只是手边有、习惯伸手。医生下午的处方随疲劳变，把全天决策同等对待，会把疲劳写进「偏好」。RLHF 里对应的是：标注做久了质量下降；点击和观看时长不是满意度；同一标注者随心情和前后例子改口；标注者若觉得某种模式对自己有利，会改标法。

<br>

## 聚合

Arrow 之外，书把 IIA 放宽成 IIA'。Borda 这时满足无限制域、帕累托、非独裁和这条较弱的独立性。再加采样假设，DPO 最大化的是加权 Borda 分。

### Borda

$y$ 的 Borda 分是随机一位投票人把它排在多少个其他选项之前，(5.1)，也等于有多少人把它排在某个 $y'$ 之前：

$$
\mathrm{Borda}(y)
=
\sum_{i=1}^{n}\big|\{y'\neq y:y\succ_i y'\}\big|
=
\sum_{y'\neq y}\big|\{i:y\succ_i y'\}\big|.
$$

$m$ 个选项时，名次分 $m-1,\ldots,0$ 就是压过的人数。

IIA'（modified IIA）：两个剖面里，若每个投票人对 $y$ 与 $y'$ 的相对名次相同，并且两者之间隔着的选项个数也相同，社会选择不能在这两个剖面上把赢家从 $y$ 换成 $y'$。名次有多远可以影响结果。Borda 仍违反原来的 IIA。

### 与 DPO

比较从 $\pi_{\mathrm{ref}}$ 抽出时，加权 Borda 胜者 (5.2) 是对参考策略抽出的对手期望胜率最高的 $y$：

$$
y^\star
=
\arg\max_y
\mathbb{E}_{y'\sim\pi_{\mathrm{ref}}(\cdot\mid x)}
\big[\mathbf{1}[y\succ y'\mid x]\big].
$$

DPO 最优策略与 $\pi_{\mathrm{ref}}$ 的密度比，正比于这个分数。质量堆在「对参考分布抽出的回复胜得最多」的那条上。

三条里任何一条不成立，这个对应就断了。抽对不是按 $\pi_{\mathrm{ref}}$ 来的，更长或更有争议的回复被抽得更多，计票规则就变了。标注池之间偏好不同，Borda 分跟着谁在标走。偏好不传递，或随上下文变，Bradley–Terry 设错，目标对不上任何一个 Borda 胜者。

### 单峰

选项在一条线上，每人一个最爱点（ideal point），离它越远越差，是单峰偏好（single-peaked preferences）。Black（1948）：中位投票人的峰是多数决的 Condorcet 胜者。五人的峰为 $\{2,4,5,7,9\}$，纯中位数是 $5$。Moulin（1980）：单峰域上，真实峰再并上若干固定的幻影峰（phantom peaks），得到的广义中位数同时防策略（strategy-proof）、满足帕累托。域不再单峰时，Gibbard–Satterthwaite 把防策略推回独裁。

### 可分

可分偏好（separable preferences）：把物件 $x$ 加进任何一个集合 $A$ 是否变好，只看 $x$ 单独是否好过空集，(5.3)。有用、无害、诚实（helpful, harmless, honest）可分时，可以按议题分别聚合。一条很有用的回复必然带一点伤害，两个议题绑在一起，分议题投票会把这种依赖丢掉。可分域上，满射、防策略、可分三条刻画按委员会投票（voting by committees）。这条规则一般达不到帕累托。

### 自由

Sen（1970）的帕累托自由人（Paretian liberal）有三条：minimal liberalism，每人在自己的私事上至少对一对选项说了算；Pareto，全体都偏好 $x$ 甚于 $y$ 时社会也如此；unrestricted domain，任何偏好剖面都允许。三条不能一起成立。

两人、一本书。$a$ 是拘谨者（Prude）读，$b$ 是随便者（Lewd）读，$c$ 是没人读。拘谨者 $c\succ a\succ b$，随便者 $a\succ b\succ c$。私事上，拘谨者使 $c$ 压过 $a$，随便者使 $b$ 压过 $c$。两人都觉得 $a$ 好于 $b$，帕累托使 $a$ 压过 $b$。$c$ 压 $a$、$a$ 压 $b$、$b$ 压 $c$，接成循环。内容审核里，一方要表达、一方要环境安全，偏好落在别人的结果上，是 nosy preference。

### 社区注释

Community Notes 把评分写成因子模型，(5.4)：

$$
u=\mu+\alpha+\beta+p^{\top}q+\varepsilon.
$$

$\mu$ 是总截距，$\alpha$ 是评分人偏严还是偏宽，$\beta$ 是注释本身的质量，$p^{\top}q$ 是评分人与注释在立场上的对齐。入选看的是控制对齐之后 $\beta$ 是否过阈值。极化注释会被盟友打高、被对手打低；朴素平均把立场混进质量，内积项把这一部分从 $\beta$ 里分开。形式和带因子的 Rasch 相同，多出来的是要估计的那一项是 $\beta$，不是票数。

### 情境与报价

Nissenbaum 的情境完整性（contextual integrity）：信息流动符合该场景的规范，隐私才算保住。五个参数是发送者（sender）、信息主体（subject）、接收者（recipient）、数据类型（data type）、传递原则（transmission principle）。心率交给教练用来调训练，和交给广告网络，数据类型相同，传递原则不同。用户点过同意，流动仍可以违反这个场景的规范。

维克里拍卖（Vickrey auction）里，赢的人付第二高的报价。自己的报价只决定赢或输，不决定付多少，如实报价是占优策略（dominant strategy），机制是 DSIC（dominant-strategy incentive compatible）。第一价格拍卖里，赢的人付自己的报价，均衡里会把报价压到真实价值下面，称为 bid shading。配置仍可能给估价最高的人，但要靠策略，不是 DSIC。

<br>

## 记号

附录里的符号和正文一致。$N$ 是投票人，$A$ 或 $Y$ 是选项，$\mathcal{L}(A)$ 是严格全序的集合。$F$ 输出社会全序，$f$ 输出一个赢家。$\sigma$ 是 logistic。$\pi_{\mathrm{ref}}$ 是冻结的参考策略。

<br>

