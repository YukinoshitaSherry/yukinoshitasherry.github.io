---
title: 斯坦福CS329H：Machine Learning from Human Preferences
date: 2026-09-23
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS329H 详细课程笔记：从选择数据、随机效用与 Bradley–Terry，到 RLHF、DPO、辅助博弈、对决赌博机与偏好聚合。
hidden: true
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
  - 成绩：项目 50%，三次闭卷笔试各 10%，两次口试各 10%。
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

<br>

> [!INFO]+ 版本范围
>
> 正文按 2026 年 9 月课程主页的五单元、十八讲组织。Foundation 讲选择数据、Bradley–Terry、随机效用与因子模型；Learning 讲极大似然、贝叶斯与微调、Fisher 信息、RLHF、DPO 与主动征询；Action 讲辅助博弈、采集函数、Thompson 采样、对决赌博机与偏好贝叶斯优化；Inversion 讲风格控制、代理目标与有界理性；Aggregation 讲 Arrow、Gibbard–Satterthwaite、公平与信息聚合。
>
> 公式来自这些题目上的标准模型：Luce、Bradley–Terry、McFadden 条件 logit、Plackett–Luce、Christiano / Ouyang 的人类反馈强化学习、Rafailov 等人的 DPO、Hadfield-Menell 等人的合作逆强化学习（课程现称 assistance game）、Yue 等人的对决赌博机、Chu–Ghahramani 与 González 等人的偏好高斯过程。2026 年课堂幻灯片在季度进行中才挂出，因此下文不把某次课上的例题编号写成既成事实。
>
> 2023 秋季以论文报告为主，题目含逆强化学习、metric elicitation 与 RLHF。2024 年公开幻灯片亦单列 metric elicitation。2026 年大纲把征询收进实验设计、主动征询与聚合，不再单列该标题。Autumn 2023 阅读中的 Train、McFadden、Luce 仍是 Foundation 的经典来源。后续学期若改讲次名称，选择公理、Bradley–Terry 似然、KL 约束下的最优策略与 DPO 代换通常仍在。
>
> 文末「教材」按站点 2026-09-23 的侧栏写：引言、第 1–5 章、结语、附录。书中第 1.8 节有一句把一切独立同分布噪声都说成满足 IIA，同节 Theorem 1 只等价到 Gumbel，后文按定理。

<br>


# 总览

- **基础（foundation）**
  - 选择集、显示偏好与陈述偏好；
  - Bradley–Terry（BT）成对模型、可识别性与比较图；
  - 随机效用（random utility model, RUM）、Gumbel 噪声与 logit、IIA、嵌套 logit、Thurstone probit、Plackett–Luce；
  - 条件 logit、低秩因子与混合 logit。
- **学习（learning）**
  - 潜变量被积掉之后的观测似然与牛顿法；
  - 最大后验（maximum a posteriori, MAP）、Laplace 近似、监督微调（supervised fine-tuning, SFT）；
  - Fisher 信息与序贯实验设计；
  - 人类反馈强化学习（reinforcement learning from human feedback, RLHF）；
  - 直接偏好优化（direct preference optimization, DPO）；
  - 对语言模型补全对的主动征询。
- **行动（action）**
  - 辅助博弈（assistance game）与情境赌博机（contextual bandit）；
  - 采集函数（acquisition function）与 Thompson 采样；
  - 对决赌博机（dueling bandit）的遗憾、以及偏好贝叶斯优化（preferential Bayesian optimization）。
- **反演（inversion）**
  - 从行为反推效用时，风格特征与代理奖励（surrogate）会把目标带偏；
  - 有界理性（bounded rationality）下的 softmax 选择，以及配分函数不可计算时仍能用的成对模型。
- **聚合（aggregation）**
  - Arrow 不可能定理与 Gibbard–Satterthwaite 定理；
  - 噪声与异质性、孔多塞陪审团定理，以及按样本量加权的合并似然。

- **记录逻辑**

$$
\text{选择数据}
\rightarrow\text{随机效用}
\rightarrow\text{观测似然}
\rightarrow\text{信息与征询}
\rightarrow\text{策略优化}
\rightarrow\text{多人聚合}
$$

- **同一条梯度**
  - 条件 logit、最大熵逆强化学习与奖励模型，梯度都是「选中方案的特征，减去模型预测的期望特征」。
  - 成对 BT 是这项差的特例：抬高胜者，压低败者，权重为当前预测还不确定的程度。
  - DPO 把奖励换成策略相对参考模型的对数概率比，梯度仍是这一对方向，权重为 $\sigma(-u)$。
- **数字各自回答的问题**
  - 成对准确率会被容易分出胜负的样本拉高；对数损失才评价概率。
  - 效用水平可以全体加一个常数，选择概率不变。报告的是差，或一个被钉住的参照项。
  - $p(1-p)$ 在 $p=1/2$ 最大。预测已经接近 0 或 1 的比较，几乎不再提供 Fisher 信息。
  - 代理奖励上升，不等于人的偏好被更好地满足。KL 离开参考策略越远，这两者越容易分开。
  - 合并全体标注者的 BT，估计的是比较次数加权的平均。它不自动代表少数群体，也不是防策略投票规则。
- **主线**
  - **前半：把「更喜欢」写成概率**
    - 效用差、噪声分布、菜单，三者决定选择概率。
  - **后半：用这个概率去问、去优化、去合并**
    - 问信息量大的对；用 KL 约束优化策略；多人时先声明合并规则放弃了哪条公理。

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


# 导论

句子是否有帮助、轨迹是否安全、漏报和误报哪一种更贵，常常写不成一个事先给定的标量损失。人仍能在两个具体结果之间指出哪一个更好。后面的估计、征询、策略和投票，监督都是这种比较。

菜单（choice set）$C$ 是一次决策里可用的有限选项。选择数据是三元组 $(C, y)$ 或带上下文的 $(x, C, y)$，$y\in C$。成对比较是 $|C|=2$。排序是 $|C|$ 个元素的一个全序。

上下文 $x$ 可以是用户、提示词（prompt）、状态或候选的属性。语言模型里常见 $x$ 为提示，$y$ 为补全（completion）。机器人里 $y$ 可以是一段轨迹。

效用（utility）$U$ 是决策者对选项的潜在评分。奖励（reward）在本课与效用指同一类潜在分数：选择模型里的 utility，强化学习里的 reward。系统效用 $V$ 是 $U$ 里不含噪声的部分。

<br>

## 记号

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
| $\beta$ | KL 约束或 Boltzmann 理性的温度系数，$\beta>0$ |
| $\sigma(z)$ | logistic，$\sigma(z)=(1+e^{-z})^{-1}$ |

平移不变：所有 $V_i$ 加上同一常数，softmax 不变。BT 分数 $\{s_i\}$ 只确定到一个加法常数。估计时钉住 $s_1=0$，或加高斯先验。

<br>

## 反馈

人给出的监督不只有成对点击。笔记后面的似然都从下面几种观测改写而来。

| 观测 | 符号 | 进入的模型 |
| :--- | :--- | :--- |
| 成对比较 | $y_w\succ y_l$ | Bradley–Terry、奖励模型、DPO |
| 全序或前 $k$ 名 | $y_1\succ\cdots\succ y_k$ | Plackett–Luce |
| 菜单中的一项 | $y\in C$ | 多项 logit、条件 logit |
| 示范轨迹 | $\tau$ | 行为克隆、最大熵逆强化学习 |
| 绝对评分 | 整数或实数 | 有序 logit；与别人的量表不能直接相减 |

绝对评分带有每个人自己的零点和刻度。甲的 $4$ 分与乙的 $4$ 分不是同一个效用。成对比较只要求同一个人在同一次里分出高下，零点在相减时消失。课程因此把比较当作默认监督，把评分当成需要再建模刻度的观测。

示范是 $|C|$ 极大时的选择：动作空间是所有字符串或所有轨迹，人只给出一个选中的元素，没有显式的败者。SFT 把这个元素的概率拉高；逆强化学习还要求一个「未选中的动作概率更低」的模型，否则败者从未出现在损失里。

<br>

## 任务

三条任务共用选择概率，输入输出不同。

- **预测**：输入 $(x,C)$，输出 $P(y\mid x,C)$。评价是留出对数损失。
- **征询**：输入当前后验与候选查询，输出下一次送给人的对。评价是达到同一留出损失所用的查询数。
- **优化**：输入偏好或由偏好拟合的 $r$，输出策略 $\pi$。评价是人的留出比较、相对 $\pi_{\mathrm{ref}}$ 的 KL，以及长度一类风格统计。

预测模型可以直接当优化用的奖励，也可以只用来决定问哪一对。两条用法对校准的要求不同：排序对了而概率贴近 0 和 1，优化仍能挑出胜者，征询则会以为没有信息可问。

<br>


# 选择数据

显示偏好（revealed preference）是真实做出的选择：点击、购买、接受补全。陈述偏好（stated preference）是假设问题上的回答：标注界面里点选哪一段更好。两者都能进同一个似然，但噪声水平不同。陈述数据缺少真实后果，显示数据则混有习惯与界面位置。

比较图：物品为顶点，至少被比较过一次的无序对为边。BT 的效用差只在连通分支内部确定。两个分支之间没有路径时，各自可以加不同的常数，似然完全相同。

<br>

## 定义

Bradley–Terry 模型：物品 $i$ 有实数分数 $s_i$，独立比较满足

$$
P(i\succ j)
=
\frac{e^{s_i}}{e^{s_i}+e^{s_j}}
=
\sigma(s_i-s_j).
$$

由此 $P(j\succ i)=1-P(i\succ j)$，且 $P(i\succ i)$ 无定义：模型不含平局。Davidson（1970）给平局加了额外参数；基础课先把平局从样本里拿掉，或拆成半次胜利。

$n_{ij}=w_{ij}+w_{ji}$ 为 $i$ 与 $j$ 的比较次数，$w_{ij}$ 为 $i$ 胜出的次数。对数似然

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

对 $s_i$ 的梯度是胜利次数减去期望胜利次数：

$$
\frac{\partial\ell}{\partial s_i}
=
\sum_{j\neq i}
\big(
w_{ij}-n_{ij}\,\sigma(s_i-s_j)
\big).
$$

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

$\ell$ 关于 $s$ 是凹的。钉住一个参照分数之后，若比较图连通且数据不是完全分离，Hessian 在自由坐标上负定，极大似然唯一。

完全分离：某个物品从未失败，则把它的分数送到 $+\infty$ 时似然一直上升，极大似然不存在有限解。逻辑回归在线性可分时同样如此。高斯先验或 $L_2$ 惩罚使解回到有限值。

<br>

## 例子

三件物品 $A,B,C$，钉住 $s_A=0$。比较次数：

| 对 | $A$ 胜 | $B$ 胜 | $C$ 胜 | 经验胜率 |
| :--- | ---: | ---: | ---: | ---: |
| $A$ vs $B$ | 3 | 1 |  | $0.75$ |
| $A$ vs $C$ | 4 |  | 1 | $0.80$ |
| $B$ vs $C$ |  | 3 | 1 | $0.75$ |

共 13 次比较。分数全为 0 时每次 $p=1/2$，负对数似然为 $13\log 2=9.0109$。

在 $s=0$，$p=1/2$，自由参数 $(s_B,s_C)$ 上对数似然的梯度为 $(0,\ -2.5)$。观测信息（负 Hessian）为

$$
I
=
\begin{pmatrix}
2 & -1 \\
-1 & 2.25
\end{pmatrix}.
$$

计算：$A$–$B$ 的 $n p(1-p)=4\cdot 1/4=1$，$A$–$C$ 为 $5/4=1.25$，$B$–$C$ 为 $1$。$s_B$ 出现在前两条与 $B$ 有关的边里，对角元 $1+1=2$；$s_C$ 的对角元 $1.25+1=2.25$；交叉项来自 $B$–$C$，为 $-1$。

牛顿法最大化 $\ell$：$s\leftarrow s-H^{-1}\nabla\ell$，其中 $H=-I$。一步之后

$$
(s_B,s_C)=(-0.7143,\ -1.4286).
$$

继续迭代到梯度消失：

$$
\hat s_B=-0.8361,\qquad \hat s_C=-1.6723,
$$

$$
\hat P(A\succ B)=0.6977,\quad
\hat P(A\succ C)=0.8419,\quad
\hat P(B\succ C)=0.6977.
$$

负对数似然降到 $7.0854$。

经验 logit 不满足传递：$\mathrm{logit}(0.75)+\mathrm{logit}(0.75)=\log 9=2.197$，而 $\mathrm{logit}(0.80)=\log 4=1.386$。BT 强制

$$
(s_A-s_B)+(s_B-s_C)=s_A-s_C,
$$

所以三对概率不能同时等于经验胜率。拟合结果是 $\mathrm{logit}\hat P(A\succ B)=\mathrm{logit}\hat P(B\succ C)=0.8361$，两者之和等于 $\mathrm{logit}\hat P(A\succ C)$。传递约束把 $A$ 对 $B$ 的胜率从 $0.75$ 拉向 $0.70$，把 $A$ 对 $C$ 从 $0.80$ 拉向 $0.84$。

<br>

## 泛化

同一分布上的留出评价用对数损失

$$
-\log \hat P(y\mid C),
$$

成对时就是 $-\log\sigma(\hat s_i-\hat s_j)$。准确率 $\mathbb{I}[\hat P>1/2]$ 对 $\hat P=0.51$ 与 $\hat P=0.99$ 一视同仁，容易对里的样本会主导准确率。

只有物品截距 $\{s_i\}$ 时，新物品没有参数，无法预测。把分数写成特征的线性函数 $s_i=w^{\top}x_i$ 之后，新物品只要有 $x_i$ 就能算概率。这是下一章因子模型要解决的外推。

比较图不连通时，留出集若跨分支，似然对那个相对常数没有信息，预测会随钉扎方式任意变化。拟合前先看连通分支。

<br>

## 代码

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

`np.clip` 只防止极端分数把 `exp` 溢出；在本例的有限解附近它不起作用。Hessian 用的是观测权重 $n p(1-p)$，与解析二阶导一致。参照项不进入 `free`，否则 $H$ 有全 1 零空间。

<br>

## 梯度

单对的对数似然 $\ell=w_{ij}\log p+w_{ji}\log(1-p)$，$p=\sigma(\delta)$，$\delta=s_i-s_j$。

1. 对 $p$ 求导：$w_{ij}/p-w_{ji}/(1-p)$。
2. 对 $\delta$ 求导时乘 $p(1-p)$。分子变成
   $$
   w_{ij}(1-p)-w_{ji}\,p=w_{ij}-n_{ij}p.
   $$
3. $\partial\delta/\partial s_i=1$，$\partial\delta/\partial s_j=-1$。所以 $i$ 得到 $w_{ij}-n_{ij}p$，$j$ 得到相反数。
4. 把所有含 $i$ 的对加起来，就是 $\partial\ell/\partial s_i=\sum_{j\neq i}(w_{ij}-n_{ij}p_{ij})$。

二阶导只多一步。$w_{ij}-n_{ij}p$ 再对 $\delta$ 求导，得到 $-n_{ij}p(1-p)$。因此 $s_i$ 的对角元累加这个负数，$s_i$ 与 $s_j$ 的交叉元累加其相反数。负对数似然的 Hessian 就是这些权重的正号版本，牛顿法解的是该正定矩阵（参照坐标删除之后）。

<br>

## 存在

Ford（1957）给出有限极大似然的图条件。作有向图：只要 $i$ 至少赢过 $j$ 一次，就连一条 $i\to j$。似然在实数上达到唯一最大值（差一个加法常数）当且仅当这张图强连通：任意两点都能沿有向边互达。

强连通失败时，可以把落后的一组分数一起送向 $-\infty$，似然不下降。极端情形：只有 $A$ 赢 $B$ 三次，没有反向胜利。负对数似然是 $-3\log\sigma(s_A-s_B)$，差越大损失越小，有限极大点不存在。前文 13 次比较里每一对都有双向胜利，有向图强连通，所以牛顿法停在有限解。

弱连通不够。$A\to B\to C$ 且没有返回边时，图是弱连通的，分数差仍然发散。拟合前画的是有向胜利图，不只是无向比较图。无向图连通保证「有路径把常数钉住」这一必要结构里的比较发生过；有向强连通才保证胜利没有把某一侧推到无穷。

<br>

## 小步

Hunter（2004）的 MM 算法（minorization–maximization）每步有闭式，极限与牛顿法相同。$W_i=\sum_{j\neq i}w_{ij}$ 为 $i$ 的总胜场。更新

$$
s_i^{+}
=
\log W_i
-
\log\sum_{j\neq i}
\frac{n_{ij}}{e^{s_i}+e^{s_j}},
$$

再减去参照项，使 $s_A$ 保持为 0。

沿用前文数据，$W=(7,4,2)$，$n_{AB}=4$，$n_{AC}=5$，$n_{BC}=4$。从 $s=0$ 出发，$e^{s_i}+e^{s_j}=2$。

$$
\begin{aligned}
s_A^{+}&=\log 7-\log\big(4/2+5/2\big)=\log(7/4.5)=0.4418,\\
s_B^{+}&=\log 4-\log\big(4/2+4/2\big)=\log 1=0,\\
s_C^{+}&=\log 2-\log(4.5)=-0.8109.
\end{aligned}
$$

减去 $s_A^{+}$：$(0,\ -0.4418,\ -1.2527)$。继续迭代，第 15 步到 $(0,\ -0.8361,\ -1.6722)$，与牛顿法的极大似然一致。MM 步长短、每步不必解线性方程；牛顿法在凹目标上通常几步就到。两者都要先满足强连通，否则 $\log W_i$ 里会出现 $W_i=0$，更新无定义，这和分数发散是同一件事。

<br>

## 平局

基础 BT 把平局的概率定为 0。Davidson（1970）加一个参数 $\theta\ge 0$：

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

$\theta=0$ 时分母回到两项，平局概率为 0，模型就是 BT。$\theta$ 大时，分数接近的对更容易被判成平局。把平局随机拆成半个胜利，等于在 $\theta=0$ 的模型里掺入噪声；数据里平局很多时，应把 $\theta$ 放进似然一起估计。

<br>

## 代码：MM

```python
def mm_bradley_terry(comparisons, k, anchor=0, steps=30):
    """Hunter MM。comparisons: (i, j, wins_i, wins_j)。返回钉住参照后的分数。"""
    wins = np.zeros(k)
    n_pair = {}
    for i, j, wi, wj in comparisons:
        wins[i] += wi
        wins[j] += wj
        n_pair[(i, j)] = n_pair[(j, i)] = wi + wj
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

`wins[i]=0` 的物品使 `log` 无定义，调用方应先把该物品划出连通分支，而不是加一个很小的伪胜场假装存在有限 MLE。

<br>


# 随机效用

随机效用模型：选项 $i$ 的效用

$$
U_i=V_i+\varepsilon_i.
$$

决策者选 $U$ 最大的项。观测不到 $U$，只观测 $y=\arg\max_i U_i$。系统部分 $V_i$ 是参数的函数，噪声 $\varepsilon$ 的分布决定选择概率的形状。

<br>

## 推导

标准 Gumbel（Type I extreme value）分布，众数为 0：

$$
F(\varepsilon)=\exp(-e^{-\varepsilon}),\qquad
f(\varepsilon)=e^{-\varepsilon}\exp(-e^{-\varepsilon}).
$$

其均值是欧拉常数 $\gamma\approx 0.577$，因此 $E[U_i]=V_i+\gamma$。全体选项共享这个常数时，均值之差等于 $V$ 之差。选择概率只依赖 $V$ 的差。

独立同分布时，

$$
P(y=i)
=
\int_{-\infty}^{\infty}
\prod_{j\neq i} F(V_i-V_j+\varepsilon)\, f(\varepsilon)\,d\varepsilon.
$$

把 CDF 代入，乘积为

$$
\prod_{j\neq i} F(V_i-V_j+\varepsilon)
=
\exp\Big(-e^{-\varepsilon}e^{-V_i}\sum_{j\neq i}e^{V_j}\Big).
$$

再乘 $f(\varepsilon)$，指数里的 $1$ 写成 $e^{-V_i}e^{V_i}$，得到

$$
e^{-\varepsilon}
\exp\Big(-e^{-\varepsilon}e^{-V_i}\sum_{k}e^{V_k}\Big).
$$

令 $\lambda=e^{-V_i}\sum_k e^{V_k}$，换元 $u=e^{-\varepsilon}$，$du=-e^{-\varepsilon}d\varepsilon$，积分从 $u=\infty$ 到 $0$，结果是 $1/\lambda$。于是

$$
P(y=i)
=
\frac{e^{V_i}}{\sum_{k\in C}e^{V_k}}.
$$

这就是多项 logit（multinomial logit），也叫 softmax。$|C|=2$ 时分母两项，即 Bradley–Terry。

对数似然对共享参数 $w$（$V_i=w^{\top}x_i$）的梯度：

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

括号内是选中项的特征减去模型下的期望特征。预测已经把概率集中在 $y_n$ 上时，该项梯度接近 0。

<br>

## 公理

Luce 选择公理（1959）：对任意含 $i,j$ 的菜单 $S$，

$$
\frac{P(i\mid S)}{P(j\mid S)}
=
\frac{P(i\mid\{i,j\})}{P(j\mid\{i,j\})},
$$

只要这些概率都为正。在有限菜单上，该公理等价于存在分数使选择概率为 softmax。因此多项 logit、Luce 模型与独立 Gumbel 随机效用，在正概率的有限选择上是同一族分布的三种写法。

无关方案独立性（independence of irrelevant alternatives, IIA）是上式的直接推论：两项的概率比不随菜单里其余项改变。这里的 IIA 是个人选择概率的性质。后文 Arrow 的 IIA 是社会排序只依赖每个人对这两项的名次，两者不是同一条公理。

<br>

## 例子

标准反例是交通方式。汽车与红巴士的系统效用都是 0。菜单只有这两项时，softmax 给出 $P(\mathrm{car})=1/2$。

加入一辆系统效用同样为 0 的蓝巴士。三项 logit 给出每一项 $1/3$，汽车概率从 $1/2$ 降到 $1/3$。汽车与红巴士的概率比仍是 1，IIA 成立。行为上两辆巴士是替代品，加入蓝巴士应当主要分走红巴士的份额，汽车概率应留在 $1/2$ 附近。独立 Gumbel 做不到这一点，因为独立噪声意味着三张「独立的票」，相似选项会被重复计数。

嵌套 logit（nested logit）把相关选项放进同一巢。巢 $B_k$ 内

$$
P(i\mid B_k)
=
\frac{e^{V_i/\lambda_k}}{\sum_{j\in B_k}e^{V_j/\lambda_k}},
\qquad
\mathrm{IV}_k
=
\log\sum_{j\in B_k}e^{V_j/\lambda_k}.
$$

巢本身的系统效用为 $\lambda_k\mathrm{IV}_k$。与随机效用相容时 $\lambda_k\in(0,1]$。$\lambda_k=1$ 退回普通 logit；$\lambda_k\to 0$ 表示巢内完全相关。

汽车单独成巢，$\lambda=1$，$V=0$，包容值 $\mathrm{IV}_{\mathrm{car}}=0$。红、蓝巴士成巢，$V=0$。

$\lambda=1$ 时 $\mathrm{IV}_{\mathrm{bus}}=\log 2$，巢效用为 $\log 2$，

$$
P(\mathrm{car})=\frac{1}{1+2}=\frac13,\qquad P(\mathrm{red})=\frac13.
$$

$\lambda=1/2$ 时 $\mathrm{IV}_{\mathrm{bus}}=\log 2$，巢效用为 $\tfrac12\log 2$，巢权重为 $\sqrt{2}$，

$$
P(\mathrm{car})=\frac{1}{1+\sqrt{2}}=\sqrt{2}-1\approx 0.4142,
$$

$$
P(\mathrm{red})=\frac12\big(1-P(\mathrm{car})\big)\approx 0.2929.
$$

$\lambda$ 越小，汽车概率越靠近 $1/2$。巢内相关正是在削弱「每辆巴士各算一个独立选项」。

<br>

## probit

Thurstone（1927）用正态噪声。Case V：$\varepsilon_i$ 独立，$N(0,\sigma^2)$。差 $\varepsilon_i-\varepsilon_j$ 服从 $N(0,2\sigma^2)$，

$$
P(i\succ j)
=
\Phi\Big(\frac{V_i-V_j}{\sigma\sqrt{2}}\Big).
$$

若把差的方差定成 1，公式写成 $\Phi(V_i-V_j)$。二元情形有闭式。$|C|>2$ 时，多元 probit 是一个 $|C|-1$ 维正态矩形上的积分，没有 softmax 那样的闭式，用模拟或数值积分。

logit 与 probit 在二元、概率远离 0 和 1 时曲线接近。尾部不同：logit 尾更重，同样大的效用差下，概率更不容易贴近 0 或 1。

<br>

## 排序

Plackett–Luce：全序 $y_1\succ y_2\succ\cdots\succ y_m$ 的概率是逐位 softmax 的乘积

$$
P(y)
=
\prod_{k=1}^{m-1}
\frac{e^{s_{y_k}}}{\sum_{t=k}^{m}e^{s_{y_t}}}.
$$

最后一名的因子为 1。$m=2$ 时就是 BT。

数值：$(e^{s_A},e^{s_B},e^{s_C})=(3,2,1)$。

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

同一分数下的成对边缘：

$$
P(A\succ B)=\frac{3}{5},\qquad
P(A\succ C)=\frac{3}{4},\qquad
P(B\succ C)=\frac{2}{3}.
$$

菜单 $\{A,B,C\}$ 上 $P(A)/P(B)=(3/6)/(2/6)=3/2$，成对菜单上 $P(A\succ B)/P(B\succ A)=3/2$。Luce 的概率比在两个菜单上相同。

有序列表的另一种模型是 ordered logit：潜在 $U=w^{\top}x+\varepsilon$，用切点把实线分成有序等级。它拟合的是等级，不是任意两两对决。等级有天然顺序时用它；任意物品对决用 BT 或 Plackett–Luce。

<br>

## 二元

$|C|=2$ 时不必做 $|C|-1$ 维积分。两个独立标准 Gumbel 的差服从标准 logistic：密度 $e^{-z}/(1+e^{-z})^2$，分布函数就是 $\sigma(z)$。于是

$$
P(U_i>U_j)
=
P(\varepsilon_j-\varepsilon_i<V_i-V_j)
=
\sigma(V_i-V_j).
$$

这是 BT 的第二份来源：第一份是多项积分在 $|C|=2$ 时的特例，第二份是噪声之差的分布函数。正态噪声的差仍是正态，所以 Thurstone 得到的是 $\Phi$ 而不是 $\sigma$。

数值：$\sigma=1$、$V_i-V_j=1$。logit 概率 $\sigma(1)=0.7311$。probit 把差的标准差设为 $\sqrt{2}$，临界点是 $1/\sqrt{2}\approx 0.7071$，

$$
\Phi(1/\sqrt{2})\approx 0.7602.
$$

同样的效用差，probit 比 logit 更接近 1。效用差继续加大时，logit 更慢贴近 1，因为 logistic 尾比正态重。

<br>

## 等级

有序 logit 用切点 $\tau_1<\tau_2<\cdots$ 把一条潜在直线切成等级。$\varepsilon$ 为 logistic 时

$$
P(y\le j\mid x)=\sigma(\tau_j-w^{\top}x),
\qquad
P(y=j)=P(y\le j)-P(y\le j-1).
$$

最低等级的下累积为 0，最高等级的上累积为 1。切点本身要单调，否则概率为负。

数值：$w^{\top}x=0$，切点 $\tau=(-1,\ 1)$，三等。

$$
\begin{aligned}
P(y=0)&=\sigma(-1)=0.2689,\\
P(y=1)&=\sigma(1)-\sigma(-1)=0.4621,\\
P(y=2)&=1-\sigma(1)=0.2689.
\end{aligned}
$$

$w^{\top}x$ 增加 1，相当于切点都左移 1，质量从低等级流向高等级。这里的 $y$ 是同一个人的等级，不是两个物品谁赢。把两件物品的评分相减再送进 BT，等于另外假设了评分差的含义；有序 logit 不自动提供这个差。

<br>


# 因子模型

自由截距只能内插已经比过的物品。要预测新菜单，或允许不同人的权衡不同，$V$ 得写成特征或潜在因子的函数。

McFadden 条件 logit：选项 $j$ 在上下文 $n$ 的系统效用 $V_{nj}=w^{\top}x_{nj}$，选择概率为 softmax。$x_{nj}$ 是该选项的属性（价格、长度、是否包含代码），可以随决策 $n$ 变化。$w$ 在选项之间共享，所以新选项只要能写出 $x$，就有 $V$。

物品截距是特征的特例：第 $i$ 个物品的 $x$ 为独热向量。此时模型退回上一章的 $\{s_i\}$，新物品没有对应坐标。

低秩因子把人 $u$ 与物品 $i$ 都嵌入同一 $d$ 维：

$$
V_{ui}
=
\alpha_u+\beta_i+p_u^{\top}q_i.
$$

$\alpha_u$ 是这个人「容易给出高分」的倾向，$\beta_i$ 是物品的流行程度，$p_u^{\top}q_i$ 是匹配。成对概率仍是 $\sigma(V_{ui}-V_{uj})$。$d$ 远小于人数与物品数时，参数量从 $O(|\mathcal{U}|\,|\mathcal{I}|)$ 降到 $O(d(|\mathcal{U}|+|\mathcal{I}|))$。未被这个人比较过、但被其他人比较过的物品，可以通过 $q_i$ 得到预测。$p_u$ 完全没有数据的新用户则还需要 $\alpha_u,p_u$ 的先验或内容特征。

<br>

## 异质

混合 logit（mixed logit）把系数本身当成随机的：

$$
P_n(i)
=
\int
\frac{e^{\beta^{\top}x_{ni}}}{\sum_j e^{\beta^{\top}x_{nj}}}
f(\beta\mid\theta)\,d\beta.
$$

$f$ 常取正态，均值与协方差由 $\theta$ 参数化。积分一般没有闭式。人与人的 $w$ 不同，IIA 在总体概率上不再成立：总体是许多 logit 的混合，替代模式可以比单一 logit 丰富。McFadden 与 Train（2000）证明，在温和条件下混合 logit 可以以任意精度逼近任意基于随机效用的选择概率。这是它作为因子模型的理论地位。

离散潜类是混合 logit 的有限混合：

$$
P(i\mid x)
=
\sum_{c=1}^{C}\pi_c\,
\mathrm{softmax}(w_c^{\top}x)_i,
\qquad
\pi_c>0,\ \sum_c\pi_c=1.
$$

类 $c$ 内部仍是 IIA；类之间的 $\pi_c$ 与 $w_c$ 不同，总体可以出现红巴士 / 蓝巴士式的替代。类成员身份是潜变量。

<br>

## Rasch

Rasch 模型是只含两个一维因子的 logit：人 $p$ 有能力 $\theta_p$，题 $i$ 有难度 $b_i$，

$$
P(X_{pi}=1)=\sigma(\theta_p-b_i).
$$

把「人赢过这道题」看成一次比较，它就是 BT，$s$ 的坐标分成两组。能力与难度同时加一个常数，概率不变。常用的钉扎是 $\sum_i b_i=0$，或令某一题 $b_1=0$。

数值：$\theta=0$，$b_1=-1$，$b_2=1$。

$$
P(X_{1}=1)=\sigma(1)=0.7311,\qquad
P(X_{2}=1)=\sigma(-1)=0.2689.
$$

人与题都换成向量时，$V_{pi}=p_p^{\top}q_i$，就是上一节的低秩因子；Rasch 是秩为 1 且人、题各只有一个标量的情形。

<br>

## 识别

条件 logit 里，全体选项的特征向量都加上同一个向量 $c$，则每个 $V$ 增加 $w^{\top}c$，softmax 不变。数据能识别的是选项之间的特征差，不是特征的绝对水平。

物品截距（alternative-specific constants）把这种不可识别写进设计矩阵：第 $j$ 个选项有自己的常数项。这些常数之和加任意数都不改变概率，必须删掉一个常数，或加上 $\sum_j \alpha_j=0$。删掉的那一项是参照选项，其他 $\alpha_j$ 的含义是「相对参照项的剩余效用」。

只观测价格、且每个菜单里的价格一起上涨同一金额时，价格系数也不被识别：差分为 0。报表上的 $w$ 必须附带参照项和特征的差分方式，否则两个实验的系数不能比较。

<br>

## 数值

下面把条件 logit 收成一个二元逻辑回归，便于把牛顿步算完。特征差固定为 $x=1$，四次独立比较里优势方胜 3 次、负 1 次。模型 $P=\sigma(w)$。负对数似然在 $w$ 处的导数与二阶导为

$$
\frac{\partial J}{\partial w}
=
\sum_n(p_n-y_n),
\qquad
\frac{\partial^2 J}{\partial w^2}
=
\sum_n p_n(1-p_n).
$$

$w=0$ 时 $p=1/2$，$J=4\log 2=2.7726$，梯度 $3(1/2-1)+(1/2-0)=-1$，二阶导 $4\cdot 1/4=1$。牛顿步 $w\leftarrow 0-(-1)/1=1$。

$w=1$ 时 $p=\sigma(1)=0.7311$，梯度 $-0.0758$，二阶导 $0.7864$，

$$
w\leftarrow 1-\frac{-0.0758}{0.7864}=1.0963.
$$

再两步到 $w=\log 3=1.0986$，因为四次独立同分布 Bernoulli 的极大似然就是经验胜率 $3/4$，而 $\sigma(\log 3)=3/4$。负对数似然从 $2.7726$ 降到 $2.2493$。

若两次比较的特征差分别是 $1$ 和 $2$，经验胜率不再对应单一的 $\sigma(w)$，就要用「特征减期望特征」的向量梯度，没有这个一元闭式。

<br>


# 选择预测

归约式（reduced form）直接回归 $P(y\mid x,C)$。结构式（structural form）先写 $V$ 与 $\varepsilon$，再推出 $P$。独立 Gumbel 加线性 $V$ 时，两者的条件分布都是 softmax，数值上是同一个逻辑回归。差别在菜单改变之后：结构式用同一个 $w$ 在新菜单上重算 softmax；一个只在旧菜单的类别标签上训练、把「选项身份」当固定类别的分类器，没有新选项的类别头。

<br>

## 估计

条件 logit 的对数似然关于 $w$ 凹，梯度是「特征减去期望特征」，Hessian 是加权协方差，负定方向对应有信息的特征对比。牛顿法或 L-BFGS 都适用。标准做法：

1. 用 $w=0$ 起步，此时每个菜单上的概率均匀，梯度等于「选中特征减菜单内特征平均」。
2. 每步计算 softmax 时减去该菜单的最大 $V$，避免溢出。
3. 若物品截距完全分离，改用 $w$ 的 $L_2$ 惩罚，或停止在验证对数损失最低处。
4. 报告自由参数的差，而不是绝对水平。

混合 logit 的模拟极大似然：从 $f(\beta\mid\theta)$ 抽 $R$ 个 $\beta^{(r)}$，用

$$
\hat P_n(i)
=
\frac1R\sum_{r=1}^{R}
\mathrm{softmax}(\beta^{(r)\top}x_n)_i
$$

代入对数似然。$R$ 固定且用独立新抽样时，模拟误差会留在极限里；用同一套随机种子的共同随机数，目标对 $\theta$ 光滑，优化稳定。$R$ 随样本增大而增大时，模拟误差才不影响一致性。

<br>

## 潜类

潜类模型是缺了类标签的混合。观测对数似然没有普通 logit 那么直接的凹性，局部极大存在。EM 把类标签当缺失数据。

E 步：当前参数下，样本 $n$ 属于类 $c$ 的责任度

$$
\gamma_{nc}
=
\frac{\pi_c\,P(y_n\mid x_n,w_c)}{\sum_{c'}\pi_{c'}\,P(y_n\mid x_n,w_{c'})}.
$$

M 步：$\pi_c=\frac1N\sum_n\gamma_{nc}$。每个 $w_c$ 是以 $\gamma_{nc}$ 为权重的条件 logit 极大似然，本身还要用牛顿法，没有更短的闭式。

数值：两类先验 $\pi=(1/2,1/2)$，菜单两项。类 0 的 logit 差为 $2$，选第一项的概率 $\sigma(2)=0.8808$。类 1 的 logit 差为 $-2$，概率 $0.1192$。观测到选择第一项时

$$
\gamma_{0}
=
\frac{0.8808}{0.8808+0.1192}
=
0.8808.
$$

这一次观测把责任度几乎全部分给「本来就预测第一项」的类。M 步会用 $0.8808$ 与 $0.1192$ 给两个条件 logit 加权。

<br>

## 代码

```python
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

梯度 `X[y] - expected` 与上一章的公式反号，因为这里最小化的是负对数似然。

<br>

## 评分

对数损失是该条件分布的严格恰当评分规则：报告真实概率 $p$ 时期望损失最低。成对情形下还有 Brier 分数

$$
(y-p)^2,
$$

$y\in\{0,1\}$。它同样在 $p$ 等于真实概率时期望最小，但对尾部误报的惩罚轻于对数损失。$p=0.01$ 而 $y=1$ 时，对数损失是 $-\log 0.01=4.605$，Brier 是 $0.99^2=0.980$。准确率对这两个预测都记 0，只要阈值是 $1/2$。

校准曲线把预测 $p$ 分箱，看箱内经验胜率是否等于箱的平均 $p$。BT 在强连通数据上的极大似然，对训练过的那些对是在经验胜率与传递约束之间的折中，不是每对都校准到经验胜率。前文 $A$ 对 $B$ 的经验胜率是 $0.75$、拟合概率是 $0.698$，差来自传递约束，不是优化没做完。

<br>

## 模拟

混合 logit 的一次模拟似然可以手算。两个等概率的系数 $\beta\in\{-2,\ 2\}$，菜单两项，特征差为 $1$，观测到选第一项。

$$
\begin{aligned}
P(\beta=-2)&=\sigma(-2)=0.1192,\\
P(\beta=2)&=\sigma(2)=0.8808,\\
\hat P&=\tfrac12(0.1192+0.8808)=0.5.
\end{aligned}
$$

对数似然贡献是 $\log 0.5=-\log 2$。若错误地用均值 $\beta=0$ 的单一 logit，概率也是 $1/2$，这一次观测分不出混合与退化。再加一个「特征差为 $1$、选了第二项」的人：混合模型给第二项的概率同样是 $0.5$（对称），单一 logit 在 $\beta=0$ 也是 $0.5$。

换成估计混合权重。固定两个点 $\beta=\pm 2$，只估计 $\pi=P(\beta=2)$。两人分别选了第一项与第二项。似然

$$
L(\pi)
=
\big(\pi\cdot 0.8808+(1-\pi)\cdot 0.1192\big)
\big(\pi\cdot 0.1192+(1-\pi)\cdot 0.8808\big).
$$

$\pi=1/2$ 时 $L=0.25$。$\pi=0.9$ 时第一项概率为 $0.9\cdot 0.8808+0.1\cdot 0.1192=0.8046$，第二项概率为 $0.1954$，$L=0.157$。均匀混合更好。模拟抽的是 $\beta$，被积掉；人属于哪一类并不出现在报表里，除非用潜类 EM 把责任度算出来。

<br>


# 贝叶斯估计

比较把某一侧推到无穷、样本很少、或者微调不能离开 SFT 太远时，先验把参数留在有限区域里。

对数后验

$$
\log p(w\mid\mathcal{D})
=
\ell(w)+\log p(w)+c.
$$

高斯先验 $w\sim N(0,\tau^2 I)$ 时，MAP 最小化

$$
-\ell(w)+\frac{1}{2\tau^2}\|w\|^2.
$$

这就是 $L_2$ 正则的极大似然。$\tau\to\infty$ 时惩罚消失，MAP 回到 MLE；完全分离时 MLE 发散，任何有限 $\tau$ 都把解拉回有限值。

Laplace 近似把后验配成正态。对数后验在 MAP 处的 Hessian 为 $\nabla^2\ell-I/\tau^2$，协方差

$$
\Sigma
\approx
\big(-\nabla^2\ell(\hat w)+I/\tau^2\big)^{-1}.
$$

$-\nabla^2\ell$ 是观测信息。比较次数少的物品，对应对角元小，后验方差大。下一章用这个方差决定还要问哪一对。

<br>

## 微调

监督微调把示范 $(x,y)$ 的条件概率拉高。自回归分解

$$
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}),
$$

SFT 最大化该和，等价于最小化下一词元交叉熵。它模仿示范的动作，不使用成对偏好。示范本身可以看成 $\beta\to\infty$ 的最优行为；示范里的次优句会被同样拉高，因为 SFT 没有「败者」项。

KL 约束的奖励最大化

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

的最优解（策略类不受限制时）为

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

推导：对每个 $x$ 单独最大化 $\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log(\pi(y)/\pi_{\mathrm{ref}}(y))$，约束 $\sum_y\pi(y)=1$。对 $\pi(y)$ 求导并令其为 0，

$$
r(y)-\beta\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}-\beta-\lambda
=
0,
$$

解出 $\pi(y)\propto\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}$。归一化常数就是 $Z(x)$。

因此 $\pi_{\mathrm{ref}}$ 扮演 completions 上的先验，$\beta$ 是温度。$r$ 越高的 $y$，后验质量越大；$\beta$ 越小，先验越强，策略越不敢离开参考模型。

反解奖励：

$$
r(x,y)
=
\beta\log\frac{\pi^\star(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta\log Z(x).
$$

$Z(x)$ 只依赖 $x$，在成对比较的奖励差里会被减掉。这是 DPO 一章的起点。

<br>

## Laplace

把前文三物品 BT 的极大似然放进 Laplace 近似。自由坐标 $(s_B,s_C)$ 上的观测信息

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

无信息先验的极限里，后验协方差是 $\mathcal{I}^{-1}$：

$$
\Sigma
\approx
\begin{pmatrix}
0.822 & 0.460 \\
0.460 & 0.920
\end{pmatrix}.
$$

$s_C$ 的边际标准差约为 $\sqrt{0.920}=0.959$，$s_B$ 约为 $\sqrt{0.822}=0.907$。两者正相关：把 $s_B$ 调高时，为了维持 $B$ 对 $C$ 的胜率，$s_C$ 往往也要调高。$A$ 被钉在 0，所以这不是三个分数的联合协方差。

高斯先验 $N(0,\tau^2)$ 把 $\mathcal{I}$ 的对角加上 $1/\tau^2$。$\tau=1$ 时对角变为 $2.6875$ 与 $2.5093$，方差下降，分离数据也会留在有限值。MAP 本身会离开纯 MLE，协方差必须在 MAP 处重算信息，不能把 MLE 的 $\mathcal{I}$ 与先验精度相加之后还把均值留在 MLE。

<br>

## 词元

SFT 的损失是词元对数概率之和。两个词元的条件概率为 $1/2$ 与 $1/4$ 时

$$
-\log\pi(y\mid x)
=
-\log\frac12-\log\frac14
=
\log 2+\log 4
=
2.0794.
$$

均匀猜测两个词元、词表大小为 $4$ 时，每个词元贡献 $\log 4$，合计 $2.7726$。本例比均匀猜测低 $0.693$，全部来自第一个词元：它的概率是 $1/2$ 而不是 $1/4$。

示范损失没有败者项。若第二个词元是人并不喜欢、只是写在示范里的字，SFT 仍把它的 $-\log\pi$ 压低。成对数据才能在同一提示下把败者的对数概率推高损失。

<br>


# 实验设计

一次比较的标注成本高。下一对按当前参数下的 Fisher 信息来挑，均匀抽样会把预算花在结果已经几乎确定的对上。

一次成对观测 $y\in\{0,1\}$，$p=\sigma(w^{\top}x)$，$x$ 是两个选项的特征差。得分函数（score）为 $(y-p)x$。$E[y-p]=0$，$Var(y-p)=p(1-p)$，所以 Fisher 信息

$$
\mathcal{I}(w;x)
=
p(1-p)\,x x^{\top}.
$$

独立比较的信息相加。$p(1-p)$ 在 $p=1/2$ 时等于 $1/4$，在 $p=0.9$ 时等于 $0.09$，不到峰值的四成。预测已经很有把握的对，标签多半是模型已经知道的那个，梯度的方差很小。

物品截距模型里，$x=e_i-e_j$，$\|x\|^2=2$，标量信息与 $p(1-p)$ 成正比。最有信息的对是当前分数最接近的对，同时这对必须落在后验仍不确定的方向上。全局「分数最近」只在先验很平、且 $x$ 的范数都一样时才与矩阵准则一致。

<br>

## 准则

设计 $\xi$ 是各对比较的次数（或比例）。三种常见标量：

- $D$ 最优：最大化 $\log\det\mathcal{I}(\xi)$，等价于最小化 Laplace 后验体积。
- $A$ 最优：最小化 $\mathrm{tr}\,\mathcal{I}(\xi)^{-1}$，即最小化各参数后验方差之和。
- $E$ 最优：最大化 $\mathcal{I}(\xi)$ 的最小特征值，照顾最不确定的方向。

序贯局部设计：在当前 $\hat w$ 处计算每对候选把 $\mathcal{I}$ 增加之后的 $\log\det$，选增加最多的一对，观测标签，更新 $\hat w$，重复。因为 $p$ 依赖未知的 $w$，这是局部最优，不是事先固定的全局最优设计。$w$ 估计很差时，会在错误的边界附近过度采样；所以前几轮仍要保留覆盖比较图的探索。

<br>

## 例子

沿用三物品的极大似然 $\hat s=(0,\ -0.8361,\ -1.6723)$。各对的 $p(1-p)$：

| 对 | $\hat P$ | $p(1-p)$ |
| :--- | ---: | ---: |
| $A$ vs $B$ | $0.6977$ | $0.2109$ |
| $A$ vs $C$ | $0.8419$ | $0.1331$ |
| $B$ vs $C$ | $0.6977$ | $0.2109$ |

$A$ 与 $C$ 分得最开，再比一次的边际信息最低。

钉住 $s_A$ 后，现有样本的观测信息（自由坐标 $s_B,s_C$）满足 $\log\det\mathcal{I}=0.607$。再加一次比较、仍用当前 $p$ 线性化：

| 新增的对 | 新的 $\log\det\mathcal{I}$ | 增量 |
| :--- | ---: | ---: |
| $A$ vs $B$ | $0.767$ | $0.160$ |
| $B$ vs $C$ | $0.767$ | $0.160$ |
| $A$ vs $C$ | $0.723$ | $0.116$ |

矩阵准则与 $p(1-p)$ 给出同一顺序。$A$–$B$ 与 $B$–$C$ 在这个对称解上增量相同。若后验显示 $s_C$ 更不确定，交叉项会把两者分开，$A$ 最优与 $D$ 最优也可能选出不同的对。

<br>

## 代码

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

负对数似然的 Hessian 就是观测信息。`comparisons` 里每加一次胜利计数，信息矩阵按新的 $n_{ij}$ 重算；序贯准则还要把 $p$ 用更新后的分数重估，不能一直沿用加样本之前的 $p$。

<br>

## 得分

一次比较的对数密度 $\ell=y\log p+(1-y)\log(1-p)$，$p=\sigma(z)$，$z=w^{\top}x$。

1. $\partial\ell/\partial p=y/p-(1-y)/(1-p)$。
2. 乘 $\partial p/\partial z=p(1-p)$，得到 $y-p$。
3. 乘 $\partial z/\partial w=x$，得分函数是 $(y-p)x$。

$E[y-p]=0$，所以得分的期望为 0。方差是

$$
E[(y-p)^2]\,xx^{\top}
=
p(1-p)\,xx^{\top},
$$

因为 $\mathrm{Var}(y)=p(1-p)$。这就是该次比较的 Fisher 信息。$x$ 换成 $e_i-e_j$ 时，信息落在分数差这一个方向上，范数平方为 2，标量大小由 $p(1-p)$ 决定。

线性模型里，Kiefer–Wolfowitz 等价定理说：设计 $\xi$ 是 $D$ 最优的，当且仅当设计空间里每个 $x$ 都满足 $x^{\top}\mathcal{I}(\xi)^{-1}x\le d$，$d$ 是参数个数，并且设计支撑上的点等号成立。BT 的 $p$ 依赖未知的 $s$，信息矩阵是非线性的，等价定理只能在当前 $\hat s$ 处局部使用。这就是序贯设计每轮都重估 $p$ 的原因。

<br>

## 指标

2023 与 2024 年的大纲把「问人要哪一个评价指标」单独称为 metric elicitation。2026 年没有这一讲的标题，问题本身仍是实验设计：未知的是线性权，查询的是成对比较。

二分类里，一个线性性能指标可以写成

$$
\phi(h)=1-(a_1\,\mathrm{FP}(h)+a_2\,\mathrm{FN}(h)),
\qquad
a_1,a_2>0.
$$

$\mathrm{FP}$ 与 $\mathrm{FN}$ 是误报和漏报。人不知道自己的 $(a_1,a_2)$，但能在两个分类器（两张混淆矩阵）里指出更喜欢哪一个。Hiranandani 等人把可行混淆矩阵的边界摊成一条曲线，再对人做二分搜索式的成对查询。无人噪声时，找到 $\epsilon$-最优分类器的查询数是 $O(\log(1/\epsilon))$。人对近邻查询随机回答时，用带噪声的二分搜索，阶仍是对数，常数变大。

这与 Fisher 设计问的不是同一件事。Fisher 设计假定 $w$ 的模型已经写好，选择最能缩小 $w$ 后验的对。指标征询假定连目标里的权 $(a_1,a_2)$ 都未给定，查询的是分类器而不是物品。两边都用成对比较，都用二分或信息增益减少查询。混淆矩阵上的线性权，和 BT 的物品分数，估计之前都要钉住尺度：$(a_1,a_2)$ 同乘正数不改变分类器的排序。

<br>


# 人类反馈

语言模型的动作是字符串，手写不出 $r(x,y)$。成对偏好先拟合奖励，再在靠近参考策略的约束下优化 $\pi$。

Christiano 等人（2017）在轨迹片段上收集成对偏好，拟合奖励，再做深度强化学习。Ouyang 等人（2022）把同一结构用在语言模型上，形成三段：

1. 在示范上做 SFT，得到 $\pi_{\mathrm{ref}}$。
2. 对同一提示的两个补全收集 $y_w\succ y_l$，用 BT 拟合 $r_\phi$。
3. 以 $r_\phi$ 减 KL 惩罚为回报，用策略梯度更新 $\pi_\theta$。$\pi_{\mathrm{ref}}$ 冻结。

<br>

## 奖励

奖励模型的训练目标就是成对负对数似然：

$$
\mathcal{L}_{\mathrm{RM}}(\phi)
=
-\mathbb{E}
\big[
\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)
\big].
$$

$r_\phi$ 整体加一个只依赖 $x$ 的函数，不改变差，因此不改变损失。实现里常把奖励头的输出在一个批次内减均值，或对参数做 $L_2$，以钉住水平。

策略优化的序列目标

$$
\mathbb{E}[r_\phi(x,y)]
-
\beta\,
D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)).
$$

自回归分解下

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=
\mathbb{E}_{\pi_\theta}
\Big[
\sum_t
\log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}
\Big].
$$

把 $-\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ 加进每个词元的奖励，在 $\pi_\theta$ 下的期望与序列 KL 一致。这是实现细节，目标函数仍是上面的序列式。

<br>

## PPO

近端策略优化（proximal policy optimization, PPO）不直接对新策略做无约束梯度步。令概率比

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},
$$

优势（advantage）$A_t$ 由旧策略下的回报估计。裁剪目标为

$$
\mathbb{E}
\big[
\min\big(
\rho_t A_t,\ 
\mathrm{clip}(\rho_t,\ 1-\epsilon,\ 1+\epsilon)\,A_t
\big)
\big].
$$

$\epsilon$ 常取 $0.1$ 或 $0.2$，属于算法超参数。$A_t>0$ 时，$\rho_t$ 超过 $1+\epsilon$ 之后目标不再增加，避免一步把该动作的概率抬得过高。$A_t<0$ 时，$\rho_t$ 低于 $1-\epsilon$ 之后目标不再下降。

KL 惩罚与 PPO 裁剪管的是两件不同的事。KL 相对的是 SFT 参考模型，限制最终策略能离示范多远。裁剪相对的是上一轮采样策略，限制单次更新的步长。只用裁剪、令 $\beta=0$，策略仍可能为了代理奖励离开 $\pi_{\mathrm{ref}}$ 很远。

优势估计、价值函数基线与广义优势估计影响方差，不改变「奖励减 $\beta$ KL」这个目标本身。奖励模型 $r_\phi$ 在 RL 阶段通常冻结；若边优化边更新 $r_\phi$，比较数据的分布会跟着策略变，原来的 BT 拟合不再对应当前状态分布。

<br>

## 梯度

奖励模型 $r_w-r_l=\delta$ 时，单样本损失 $-\log\sigma(\delta)$。对 $\delta$ 的导数是 $\sigma(\delta)-1$。

$\delta=0.5$ 时 $\sigma(0.5)=0.6225$，损失 $0.4741$，导数 $-0.3775$。梯度下降会增加 $\delta$：提高胜者奖励、降低败者奖励，步的幅度是 $1-\sigma(\delta)$。$\delta$ 已经很大时 $\sigma(\delta)\approx 1$，导数接近 0，这条比较不再改奖励头。

若 $r=\phi^{\top}h(x,y)$ 且 $\phi$ 是最后一层，$h$ 是冻结的隐藏向量，则 $\phi$ 的梯度是 $(\sigma(\delta)-1)h_w+(1-\sigma(\delta))h_l$。胜者与败者的隐藏向量若几乎平行，这一对提供的方向很短，和 BT 里 $x=e_i-e_j$ 范数固定是同一几何。

批次里把 $r$ 减均值，不改变任何 $\delta$，因此不改变 $\mathcal{L}_{\mathrm{RM}}$。在 PPO 里，减去的均值会改变优势的零点；再除以标准差会改变 $r$ 相对 $\beta$ KL 的尺度。$\beta$ 是在这个尺度下选定的。中途开始做白化（whitening）等于换了 $\beta$，需要重新看留出比较。

<br>

## 抽样

不必先跑 PPO。从 $\pi_{\mathrm{ref}}$ 抽 $n$ 个补全，用冻结的 $r_\phi$ 取最大，称为 best-of-$n$。诱导出的策略满足

$$
\pi_{\mathrm{BoN}}(y)
=
n\,\pi_{\mathrm{ref}}(y)\,
P\big(r(y)\text{ 为 }n\text{ 个样本中的最大}\big)
\le
n\,\pi_{\mathrm{ref}}(y),
$$

因为那个概率不超过 1。于是

$$
D_{\mathrm{KL}}(\pi_{\mathrm{BoN}}\|\pi_{\mathrm{ref}})
=
\mathbb{E}\Big[\log\frac{\pi_{\mathrm{BoN}}}{\pi_{\mathrm{ref}}}\Big]
\le
\log n.
$$

$n=4$ 时 KL 不超过 $\log 4=1.386$。上界来自密度比，不依赖 $r$ 的具体形状。PPO 用 $\beta$ 把 KL 停在某个预算里；best-of-$n$ 用样本量直接限制 KL。$n$ 很大时，argmax 会走到 $r_\phi$ 的漏洞上，和加大 PPO 步数是同一种过优化。

<br>

## 裁剪

令 $\epsilon=0.2$。裁剪目标对 $\rho$ 的依赖分四种，数字都取 $A=\pm 1$。

| $A$ | $\rho$ | $\rho A$ | $\mathrm{clip}(\rho)A$ | $\min$ | 对 $\rho$ 的梯度 |
| ---: | ---: | ---: | ---: | ---: | :--- |
| $+1$ | $1.1$ | $1.1$ | $1.1$ | $1.1$ | $+1$，继续抬高 |
| $+1$ | $1.3$ | $1.3$ | $1.2$ | $1.2$ | $0$，停在 $1+\epsilon$ |
| $-1$ | $0.9$ | $-0.9$ | $-0.9$ | $-0.9$ | $-1$，继续压低 |
| $-1$ | $0.5$ | $-0.5$ | $-0.8$ | $-0.8$ | $0$，停在 $1-\epsilon$ |

$A>0$ 且 $\rho>1+\epsilon$ 时，目标取常数 $1+\epsilon$，不再奖励更大的概率比。$A<0$ 且 $\rho<1-\epsilon$ 时，目标取常数 $(1-\epsilon)A$，不再奖励更小的概率比。反方向——把差动作的概率越抬越高——不会被这张裁剪挡住，要靠负的优势本身把 $\rho$ 拉回来。

广义优势估计把时序差分 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ 按 $\lambda$ 混合：

$$
A_t
=
\sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda=0$ 时 $A_t=\delta_t$，方差小、偏差大；$\lambda=1$ 时回到蒙特卡洛回报减基线。它估计的是同一条轨迹回报的优势，不替换奖励里的 $-\beta\log(\pi/\pi_{\mathrm{ref}})$。

<br>


# 直接偏好

策略类够大时，KL 约束问题的解可以直接用成对似然训练。奖励网络和 PPO 循环都不单独保留。

把贝叶斯一章的反解

$$
r(x,y)
=
\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta\log Z(x)
$$

代入 Bradley–Terry。$Z(x)$ 在 $y_w$ 与 $y_l$ 中相同，

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

Rafailov 等人（2023）把待求策略 $\pi_\theta$ 放进这个概率，最小化负对数似然

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

$\pi_{\mathrm{ref}}$ 停止梯度。$\log\pi(y\mid x)$ 仍是词元对数概率之和。

该代换在「策略类包含上述 $\pi^\star$」时与 KL 约束问题对齐。神经网络只是一个受限策略类时，DPO 是这个重参数化之下的 BT 极大似然，不再保证达到无约束策略类上的 $\pi^\star$。参考模型的支撑还要盖住数据里出现的 $y$；$\pi_{\mathrm{ref}}(y\mid x)=0$ 的补全使对数比无定义。

<br>

## 梯度

令 $u=\beta(\Delta_\theta-\Delta_{\mathrm{ref}})$。单样本损失 $-\log\sigma(u)$ 对 $u$ 的导数是 $-\sigma(-u)$。于是

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

梯度下降会增加胜者的对数概率、降低败者的对数概率。权重 $\sigma(-u)$ 在 $u$ 很大时接近 0：模型已经以很大的奖励差把胜者排在前面，这条样本几乎不再推动参数。$u$ 很负时权重接近 1：模型把败者排得更高，更新最大。

<br>

## 例子

取 $\beta=0.1$ 只为把算术做完，不是理论最优温度。参考模型对两个补全的概率都是 $0.2$。当前策略给胜者 $0.3$、败者 $0.1$。

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

$$
\sigma(u)=0.5274,\qquad
-\log\sigma(u)=0.6397,\qquad
\sigma(-u)=0.4726.
$$

策略与参考模型给出相同概率时 $u=0$，损失为 $\log 2=0.6931$。本例 $0.6397<\log 2$，权重 $0.47$，仍有推一把的余地。

若把策略概率对调，胜者 $0.1$、败者 $0.3$，则 $u=-0.10986$，损失为 $0.7496>\log 2$，权重 $\sigma(0.10986)=0.5274$。方向反了的策略，损失高于「完全贴住参考模型」。

<br>

## 代码

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

`logaddexp(0, -u)` 等于 $\log(1+e^{-u})=-\log\sigma(u)$，在 $u$ 很负时也不会下溢。批量损失是样本均值。胜者与败者的词元梯度按 `weight * beta` 缩放，符号相反。

<br>

## 隐式

代换定义的隐式奖励是 $r(x,y)=\beta\log(\pi_\theta(y\mid x)/\pi_{\mathrm{ref}}(y\mid x))$，整体再加 $\beta\log Z(x)$ 也不影响比较。沿用 $\beta=0.1$、$\pi_{\mathrm{ref}}=0.2$、$\pi(y_w)=0.3$、$\pi(y_l)=0.1$：

$$
\begin{aligned}
r(y_w)&=0.1\log(1.5)=0.04055,\\
r(y_l)&=0.1\log(0.5)=-0.06931,\\
r(y_w)-r(y_l)&=0.10986.
\end{aligned}
$$

差正好等于损失里的 $u$。DPO 的 $\sigma(u)$ 就是用这份隐式奖励做的 BT 概率。训练结束后若要一个可复用的标量奖励，用这个对数比即可，不必再拟合单独的 $r_\phi$。对数比依赖 $\pi_{\mathrm{ref}}$：换一个参考模型，同一 $\pi_\theta$ 的隐式奖励会变。

词元实现里 $\log\pi(y\mid x)=\sum_t\log\pi(y_t\mid x,y_{<t})$。$u$ 是两条序列的和之差，不是每个词元各自做一次 sigmoid。把 sigmoid 放进每个词元再平均，得到的是另一个目标。

$\pi_{\mathrm{ref}}$ 的对数概率要在计算图里截断。若参考模型与策略共享初始权重但没有截断，梯度会同时把 $\pi_{\mathrm{ref}}(y_w)$ 拉低，对数比被参考模型的移动放大，不再是冻结先验下的 BT 似然。

<br>


# 主动征询

池式主动学习（pool-based）：存在一个未标注集合，算法每次从中选一个点送去标注。流式（stream-based）：样本依次到来，当时不标则丢弃。语言模型的补全可以先采样进池，再决定哪一对送给人，这是池式。

认知不确定（epistemic uncertainty）来自参数还没被数据钉死，更多比较能把它降低。偶然不确定（aleatoric uncertainty）来自噪声本身，例如同一个人在完全相同的两段文字上随机点击，加数据也降不下去。$p=\sigma(w^{\top}x)$ 在 $w$ 已知且等于 $1/2$ 时，Bernoulli 方差全是偶然的；若 $w$ 的后验很宽，即使后验预测 $p$ 接近 $1/2$，其中一部分是认知的。

<br>

## 采集

记预测分布 $p(y\mid x)$。三种只看这个分布的准则：

- 熵：$\alpha(x)=-\sum_y p(y\mid x)\log p(y\mid x)$，越大越不确定。
- 间隔：$p_{(1)}-p_{(2)}$，越小越不确定。二元时间隔为 $|2p-1|$。
- 最小置信：$1-\max_y p(y\mid x)$，越大越不确定。

自然对数下，$p=0.6$ 的熵为 $0.673$，$p=0.7$ 的熵为 $0.611$，$p=0.8$ 的熵为 $0.500$。三者都会选 $0.6$。换对数底只乘一个常数，排序不变。

委员会（query by committee）用多组参数。投票熵是硬投票分布的熵。平均分布的熵减去各成员熵的平均，就是下面的 BALD 在有限委员会上的蒙特卡洛形式。

贝叶斯主动学习分歧（Bayesian active learning by disagreement, BALD）选的是标签与参数之间的互信息：

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

第一项大，表示预测不确定；第二项是已知 $w$ 之后还剩下的噪声。两者之差大，表示「不确定主要来自还不知道 $w$」。纯熵准则在噪声本质就大的区域也会采样；BALD 会避开那些怎么标都帮不上参数的点。

<br>

## 语言模型

对固定提示 $x$ 与两个补全 $y_1,y_2$，BT 概率 $p=\sigma(r(x,y_1)-r(x,y_2))$。熵最大发生在奖励差为 0。只看点估计 $r_\phi$ 时，主动策略就是送出奖励最接近的对。

点估计会过度自信：差的绝对值被高估，熵全体变小，算法以为没有什么可问的。用后验（深集成、掉线近似或 Laplace）计算奖励差的分布，再算 BALD，问的是「参数还不同意谁赢」的对，而不是「某一个点估计碰巧打成平手」的对。

公开大纲把本讲标成主动征询与案例，案例材料随后挂出。与这个题目直接对应、且已经发表的设置有：轨迹片段的成对比较（Christiano et al., 2017），以及机器人奖励的批量主动偏好查询（Bıyık & Sadigh, 2018）。语言模型上的做法是同一准则作用于补全对：先从当前策略采样若干 $y$，再按互信息挑选送标的对。

<br>

## 算术

三个候选对的预测胜率是 $0.6$、$0.7$、$0.8$。自然对数下的熵：

$$
\begin{aligned}
H(0.6)&=-0.6\log 0.6-0.4\log 0.4=0.673,\\
H(0.7)&=-0.7\log 0.7-0.3\log 0.3=0.611,\\
H(0.8)&=-0.8\log 0.8-0.2\log 0.2=0.500.
\end{aligned}
$$

熵准则选 $0.6$。间隔 $|2p-1|$ 分别是 $0.2$、$0.4$、$0.6$，最小间隔也是 $0.6$。最小置信 $1-\max(p,1-p)$ 分别是 $0.4$、$0.3$、$0.2$，最大者也是 $0.6$。二元 BT 上这三个准则单调等价：都是 $|p-1/2|$ 的递减函数。它们在多类或委员会投票上才分开。

三人委员会对 $x_1$ 的硬投票是 $2$ 对 $1$。投票分布 $(2/3,\ 1/3)$ 的熵为 $0.637$。若三人都投同一类，熵为 $0$，这一查询的投票分歧为 $0$。投票熵用的是硬标签，不看成员把 $0.51$ 和 $0.99$ 都投成同一类时的置信差。

<br>

## 分歧

两个等权后验样本。第一对：两人给出的胜率是 $0.2$ 与 $0.8$。预测边缘是 $0.5$，熵 $\log 2=0.693$。已知参数后的熵是 $H(0.2)=0.500$，期望仍是 $0.500$。

$$
\mathrm{BALD}=0.693-0.500=0.193.
$$

第二对：两人都给出 $0.5$。边缘熵仍是 $0.693$，条件熵也是 $0.693$，$\mathrm{BALD}=0$。熵准则会去问第二对，因为它最不确定；BALD 不去问，因为不确定完全来自噪声，标签到了也不改参数。

第三对：两人都给出 $0.6$。边缘熵 $0.673$，条件熵也是 $0.673$，$\mathrm{BALD}=0$。点估计的奖励差不为 0，但后验没有分歧。主动查询若只有一个 $r_\phi$，会把第一对和第二对都看成「接近一半」而忽略第三对；有后验时只保留第一对。

<br>

## 代码

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

`bald_binary([0.2, 0.8])` 约为 $0.193$；`bald_binary([0.5, 0.5])` 为 $0$。样本要来自参数后验，把同一个点估计复制两遍会得到 BALD 恒为 0。

<br>


# 辅助博弈

前面的比较把人当成标注器，标注本身不改变环境。辅助博弈里人的动作同时是信息和干预，奖励参数只有人知道。

Hadfield-Menell、Russell、Abbeel 与 Dragan（2016）的合作逆强化学习（cooperative inverse reinforcement learning），课程现在称为辅助博弈（assistance game）。要素：

- 状态 $s$，人的动作 $a_H$，系统的动作 $a_R$；
- 隐藏参数 $\theta\sim p(\theta)$，进入共享奖励 $R(s,a_H,a_R;\theta)$；
- 人观测到 $\theta$，系统只观测状态与动作，维持信念 $b(\theta)$；
- 双方最大化同一个期望累计奖励。

系统若把人的策略当成固定的，自己面对的是以 $\theta$ 为隐藏状态的部分可观测马尔可夫决策过程（POMDP）。人若知道系统在学习，就会选择让信念更新的动作，即使该动作的当下奖励较低。这种动作是教学，不是独立同分布的最优示范。

用普通逆强化学习的似然去解释教学动作，会把「故意做给人看」当成「这件事本身奖励很高」。辅助博弈把人的策略写成依赖系统信念的均衡策略，教学的代价由未来合作收益补偿。

<br>

## 赌博机

情境赌博机：每轮观测上下文 $x_t$，选择动作 $a_t$，得到奖励 $r_t(x_t,a_t)$ 或一次偏好。遗憾

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

$a_t^\star$ 是该上下文上奖励最大的动作。反馈分布在算法的信念之外定义：人即使存在，也只是一个把 $(x,a)$ 映到数字的预言机。

辅助设定里，同一轮可以出现人的动作。该动作依赖 $\theta$ 和人对系统信念的预期，因此依赖学习算法本身。把这串动作当成情境赌博机的独立奖励抽样，会丢掉「人为了被看懂而偏离短视最优」的部分。

两个可以算清楚的极端：

- 人永不行动，只在被问到时返回偏好，则问题退回偏好赌博机，采集函数一章的准则适用。
- 人短视地最大化当前 $R$、不考虑系统是否学会，则人的动作是 $\theta$ 的含噪观测，系统的推断是被动的贝叶斯更新，没有均衡里的教学动机。

课程要分开的是中间情形：双方对同一 $\theta$ 的累计奖励负责，信息与任务完成在一个目标里权衡。

<br>

## 信念

把短视的人写成可计算的更新。$\theta\in\{0,1\}$，先验各 $1/2$。人知道 $\theta$，动作 $a\in\{0,1\}$，匹配时即时奖励 $1$，否则 $0$，Boltzmann 温度 $\beta=2$：

$$
P(a=\theta)=\sigma(2)=0.8808,\qquad P(a\neq\theta)=0.1192.
$$

观测到 $a=1$。后验

$$
P(\theta=1\mid a=1)
=
\frac{0.8808}{0.8808+0.1192}
=
0.8808.
$$

系统下一轮若必须交付一项，选 $\theta=1$ 的期望奖励是

$$
0.8808\cdot(+1)+0.1192\cdot(-1)=0.7616.
$$

选 $\theta=0$ 则是 $-0.7616$。这一步只用了贝叶斯公式，人的策略被当成不依赖系统信念的固定噪声。

教学动机出现在人的目标含有未来项时。设发信号成本 $0.1$，信号被信任后系统交付正确项的奖励为 $1$。人发信号的净收益是 $0.9$；不发信号、系统保持先验时期望奖励是 $0$。短视的人只看即时匹配，不会为了 $0.1$ 的成本去发一个当下奖励为 $0$ 的信号。把未来 $0.9$ 写进人的回报之后，发信号成为更好的动作。逆强化学习若假设人只最大化即时 $R$、且 $\beta\to\infty$，这个信号的似然是 $0$，因为它不是短视最优。辅助博弈把信号解释成对累计共享奖励的最优反应。

<br>


# 采集采样

Thompson 采样（Thompson sampling）：每个臂的均值有后验。每轮从各臂后验抽一个均值，选择抽样值最大的臂，观测奖励，更新该臂后验。

Bernoulli 奖励、Beta 先验时，共轭更新封闭。先验 $\mathrm{Beta}(1,1)$ 为均匀分布。成功加到第一个参数，失败加到第二个。后验均值是 $(\alpha)/(\alpha+\beta)$。

数值：臂 $A$ 已有 3 次成功、1 次失败，后验 $\mathrm{Beta}(4,2)$，均值 $4/6=0.667$。臂 $B$ 有 1 次成功、3 次失败，后验 $\mathrm{Beta}(2,4)$，均值 $0.333$。若一次抽样得到 $\tilde\mu_A=0.50$、$\tilde\mu_B=0.60$，规则选择 $B$。$B$ 的均值更低，这一轮仍可能被抽中，这就是探索。抽样改成 $\tilde\mu_A=0.70$、$\tilde\mu_B=0.40$ 时选择 $A$，这一轮是利用。

线性情境赌博机：奖励 $x_{t,a}^{\top}w$，$w$ 的后验为正态。每轮抽 $\tilde w$，选 $a=\arg\max_a x_{t,a}^{\top}\tilde w$。后验方差大的臂，抽样会偶尔把它们推到最大，从而被尝试。

<br>

## 偏好

偏好反馈没有标量奖励。Thompson 式的做法是抽一个效用参数 $\tilde w$，再按 $\tilde w$ 行动：

- 要推荐一项：选 $\tilde w$ 下最优的项。
- 要再问一对：选 $\tilde w$ 下最优与次优，或当前记录最优与这次抽样的最优。

第二项使查询集中在「抽样世界里会改变冠军」的对上。后验变窄之后，抽样世界彼此相似，查询与推荐都稳定在同一项附近。

期望改进（expected improvement）依赖「比当前最优好多少」的标量。偏好观测只提供序，改进量需要借潜在效用的高斯过程或线性模型才能定义。所以偏好问题里 Thompson 采样与互信息比直接搬用期望改进更常见。

<br>

## 代码

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

`rng` 由调用方固定种子，结果才能复现。偏好版本只展示「按一个后验样本排序」这一步；完整循环还要观测人的选择并更新 $w$ 的后验，线性效用配正态先验时就是贝叶斯逻辑回归的更新。

<br>

## 序列

两臂真实均值 $0.8$ 与 $0.3$，先验 $\mathrm{Beta}(1,1)$，`numpy` 生成器种子 $0$。前六轮的抽样与奖励：

| 轮 | 抽样 $(A,B)$ | 选择 | 奖励 | 之后的成功 / 失败 |
| ---: | :--- | :---: | ---: | :--- |
| 1 | $0.702,\ 0.713$ | $B$ | $0$ | $A$ 空，$B$ 为 $0/1$ |
| 2 | $0.003,\ 0.122$ | $B$ | $0$ | $B$ 为 $0/2$ |
| 3 | $0.644,\ 0.377$ | $A$ | $1$ | $A$ 为 $1/0$ |
| 4 | $0.839,\ 0.128$ | $A$ | $1$ | $A$ 为 $2/0$ |
| 5 | $0.818,\ 0.330$ | $A$ | $0$ | $A$ 为 $2/1$ |
| 6 | $0.569,\ 0.161$ | $A$ | $1$ | $A$ 为 $3/1$，$B$ 为 $0/2$ |

第 1 轮 $B$ 的后验均值只有 $0.5$，抽样 $0.713$ 仍高于 $A$ 的 $0.702$，于是去拉了较差的臂。两次失败之后，$B$ 的后验 $\mathrm{Beta}(1,3)$ 均值是 $1/4$，后续抽样落到 $A$ 下面。这六轮的累计奖励是 $3$。每轮都选 $A$ 的期望奖励是 $0.8$，六轮期望 $4.8$；Thompson 用前两轮的 $0$ 换来了「$B$ 较差」这个信息。

线性情境版本只多一个矩阵：后验精度 $\Lambda_t= \Lambda_0+\sum x_{a}x_{a}^{\top}/\sigma^2$，均值由同样的秩一更新得到。抽 $\tilde w\sim N(\mu_t,\Lambda_t^{-1})$ 再取 $\arg\max_a x_a^{\top}\tilde w$。偏好版本把奖励 $x^{\top}w$ 换成 $\sigma(w^{\top}(x_i-x_j))$ 的似然，精度矩阵不再有共轭闭式，要用 Laplace 在 MAP 处加观测信息 $p(1-p)xx^{\top}$。

<br>


# 对决优化

非情境对决赌博机：臂集合 $\mathcal{B}=\{b_1,\ldots,b_K\}$。每轮选两臂，观测一次比较。Yue、Broder、Kleinberg 与 Joachims（2012）记

$$
P(b_i\succ b_j)
=
\frac12+\epsilon(b_i,b_j),
\qquad
\epsilon\in\Big(-\frac12,\frac12\Big),
$$

并且 $\epsilon(b_i,b_j)=-\epsilon(b_j,b_i)$，$\epsilon(b_i,b_i)=0$。$\epsilon>0$ 当且仅当 $i$ 以大于一半的概率击败 $j$。这与 BT 相容：$\epsilon(b_i,b_j)=\sigma(s_i-s_j)-1/2$。$\epsilon$ 是概率减去一半，不是单次观测的 $\pm 1$。

瞬时遗憾取两臂各自相对最优臂 $b^\star$ 的劣势：

$$
r_t
=
\big(P(b^\star\succ b_{t,1})-\tfrac12\big)
+
\big(P(b^\star\succ b_{t,2})-\tfrac12\big)
=
\epsilon(b^\star,b_{t,1})+\epsilon(b^\star,b_{t,2}).
$$

累积遗憾 $R_T=\sum_{t=1}^{T}r_t$。两臂都是 $b^\star$ 时 $r_t=0$，但这一轮没有信息。算法必须在「多打几场有信息的对」与「少把差臂亮给用户」之间权衡。

<br>

## 例子

三个臂的 BT 分数 $(s_1,s_2,s_3)=(0,\ -0.5,\ -1.5)$，最优臂是 $1$。

$$
\begin{aligned}
P(1\succ 2)&=\sigma(0.5)=0.6225, & \epsilon_{12}&=0.1225,\\
P(1\succ 3)&=\sigma(1.5)=0.8176, & \epsilon_{13}&=0.3176,\\
P(2\succ 3)&=\sigma(1.0)=0.7311, & \epsilon_{23}&=0.2311.
\end{aligned}
$$

本轮亮出 $(2,3)$：

$$
r_t=\epsilon_{12}+\epsilon_{13}=0.1225+0.3176=0.4401.
$$

亮出 $(1,2)$ 时 $r_t=0+\epsilon_{12}=0.1225$。亮出最优与次优，遗憾更小，同时 $P(1\succ 2)$ 更接近 $1/2$，Fisher 信息更大。亮出 $(2,3)$ 既累积更多遗憾，又完全不观测最优臂。

<br>

## 规则

Condorcet 胜者：存在一臂，对其他每一臂的胜率都严格大于 $1/2$。分数模型 $s_i$ 总有 Condorcet 胜者，因为分数最高的臂对所有人 $\epsilon>0$（并列最高除外）。一般的 $\epsilon(b_i,b_j)$ 可以循环：$A$ 赢 $B$、$B$ 赢 $C$、$C$ 赢 $A$，此时没有 Condorcet 胜者。下面的遗憾定义与「找到 $b^\star$」都以存在该胜者为前提。存在循环时，目标要改成 Copeland 分数（击败的臂数）或 Borda 分数（胜率之和），否则最优臂没有定义。

若干决策规则只保留结构，常数以论文为准：

- Interleaved filter（Yue et al., 2012）：维持候选集，用置信界把「显著差于当前候选最优」的臂删去，查询集中在候选内部。
- RUCB（relative upper confidence bound，Zoghi et al., 2014）：给每一对的胜率一个上置信界，选择在这些界下看起来能打赢最多对手的臂，再配一个挑战者。
- 情境对决（Dudík et al., 2015）：$\epsilon$ 依赖上下文，把线性赌博机的置信界搬到成对差上。
- Dueling posterior sampling（Novoseller et al., 2020）：后验同时含转移与效用。每轮抽一套转移和效用，解出该样本下的最优策略，掷出两条轨迹，用偏好更新两个后验。有限地平线、离散状态时，转移的 Dirichlet–多项共轭更新与贝叶斯线性效用可以分开写。2024 年公开课幻灯片用这个算法说明「偏好上的后验采样」。

<br>

## 函数

偏好贝叶斯优化把臂从有限集合换成连续输入 $x\in\mathcal{X}$，潜在函数 $f$ 上放高斯过程先验。观测仍是对决。Chu 与 Ghahramani（2005）用 probit 观测模型

$$
P(x\succ x'\mid f)
=
\Phi\Big(\frac{f(x)-f(x')}{\sqrt{2}\,\sigma}\Big),
$$

$\sigma$ 为噪声尺度。后验不再是高斯，Laplace 近似或期望传播给出均值与协方差。González 等人（2017）的偏好贝叶斯优化在这个后验上做采集：从近似后验抽一条函数 $\tilde f$，令下一对的一个点为 $\tilde f$ 的最大值，另一个点为当前后验均值的最大值。这就是连续域上的对决 Thompson 采样。

线性 BT 是该模型的退化：核是线性核，且观测用 logistic 而不是 probit。实验设计一章的 $\log\det$ 准则，对应的是参数有限维时的特殊情况。

<br>

## 代码

```python
def duel_epsilon(scores, i, j):
    return float(sigmoid(scores[i] - scores[j]) - 0.5)


def instantaneous_regret(scores, star, i, j):
    return duel_epsilon(scores, star, i) + duel_epsilon(scores, star, j)


scores = np.array([0.0, -0.5, -1.5])
# instantaneous_regret(scores, 0, 1, 2) ≈ 0.4401
# instantaneous_regret(scores, 0, 0, 1) ≈ 0.1225
```

`star` 必须真是 `scores` 的argmax。若调用方传入的 $\epsilon$ 表存在循环，这个函数仍会输出数字，但该数字不再是「相对 Condorcet 胜者」的遗憾。

<br>

## 置信

RUCB 给每一对的胜率一个上置信界。$t$ 为已经进行的轮数，$n_{ij}$ 为该对的比较次数，$\hat p_{ij}$ 为 $i$ 胜 $j$ 的频率。取

$$
u_{ij}
=
\hat p_{ij}
+
\sqrt{\frac{\alpha\log t}{n_{ij}}}.
$$

$\alpha$ 是置信水平的超参数。若 $u_{ji}<1/2$，则在这个界下 $j$ 仍打不赢 $i$，$j$ 可以从「可能的最优」里去掉。

数值：$t=300$，$\alpha=1$，$n_{12}=100$，$1$ 胜了 $90$ 次。$\log 300\approx 5.704$，$(\log t)/n=0.05704$，平方根 $0.2388$。

$$
u_{21}=0.10+0.2388=0.3388<1/2.
$$

臂 $2$ 被判定为显著差于臂 $1$。若 $1$ 只胜了 $70$ 次，$\hat p_{21}=0.30$，$u_{21}=0.539>1/2$，$2$ 还留在候选里。界的宽度只依赖次数和 $t$，不依赖 BT 的传递约束；它逐对消除，允许数据里暂时出现循环。

<br>

## 循环

三臂上可以没有 Condorcet 胜者。设胜率

$$
P(1\succ 2)=0.6,\quad
P(2\succ 3)=0.6,\quad
P(3\succ 1)=0.6.
$$

每个臂都赢一个、输一个。Copeland 分数是「胜率超过 $1/2$ 的对手数」，三臂都是 $1$，打平。Borda 分数是对所有对手的胜率之和，三臂都是 $0.6+0.4=1.0$，也打平。遗憾公式里的 $b^\star$ 没有定义。

有分数 $s_i$ 的 BT 模型写不出这个循环，因为 $s_1>s_2>s_3>s_1$ 不可能。观测到循环时，要么换 Copeland / Borda 当目标，要么承认噪声很大、循环会随着次数增加消失。用 BT 强行拟合，会把三个分数压到彼此接近，对数损失可以还不错，但「唯一最优臂」是模型加上去的。

<br>


# 风格代理

从比较里反推的分数会吃进一切和胜负一起变的东西：长度、格式、语气。拿这个分数去优化，放大的是这些规律。

构造例子，系数是为了算术，不是实测回归。内容质量 $q$，词元数 $n$，代理

$$
r=q+0.1\,n.
$$

简洁回答 $q=5$、$n=20$，则 $r=7$。冗长回答 $q=1$、$n=200$，则 $r=21$。BT 会稳定地让冗长者获胜，随后的 PPO 或 DPO 会加长输出。把特征改成 $(q,\ n)$ 并单独估计权重，或在奖励里减去在训练集上拟合的长度项，简洁回答的名次才会回来。关键是长度进入了 $r$ 的时候，优化器没有别的渠道知道它是混淆因素。

多属性奖励写成 $r=w^{\top}\phi(x,y)$。$\phi$ 的坐标可以是内容奖励头、长度、是否使用列表、礼貌分类器的对数几率。$w$ 可以由另一次小规模征询估计，也可以由设计者钉住。前者仍是偏好学习；后者是把已经决定的权衡写进目标。估计 $w$ 时，属性之间要有足够的独立变化，否则内容与长度共线，$w$ 不可识别，和 BT 的不连通比较图是同一类问题。

<br>

## 拆开

沿用 $r=q+0.1n$。减去长度项之后，简洁回答的分数是 $q=5$，冗长回答是 $q=1$。名次从冗长领先 $21-7=14$，变成简洁领先 $4$。BT 的输入若改成这个差，胜者对调。

只观测这两类回答时，权重本身不可识别。数据里的特征向量只有 $(q,n)=(5,20)$ 与 $(1,200)$。奖励差

$$
r_{\mathrm{short}}-r_{\mathrm{long}}
=
4w_q-180w_n
$$

是一个数。满足 $4w_q-180w_n=c$ 的 $(w_q,w_n)$ 有无穷多，似然只依赖 $c$。要单独报告「长度权重是 $0.1$」，需要第三类回答，使 $(q,n)$ 不落在同一条直线上，或者由设计者把其中一个权重钉死。

<br>

## 转向

推理时的风格控制可以不改权重。在某一层收集两种风格的隐藏状态，转向向量

$$
v
=
\mathbb{E}[h\mid \mathrm{style}_{+}]
-
\mathbb{E}[h\mid \mathrm{style}_{-}].
$$

生成时把 $\alpha v$ 加到该层。$\alpha$ 控制强度。这改变的是条件分布的局部方向，不是一个在成对数据上重新估计的 $r_\phi$。它适合在已经训练好的策略上调节格式或语气；它不修正奖励模型里已经学到的长度偏好，除非负风格的语料恰好把该方向分离出来。

<br>

## 过优化

代理 $r_\phi$ 只在训练比较覆盖的分布附近近似人的判断。策略优化把质量移向 $r_\phi$ 的高处。Gao、Schulman 与 Hilton（2023）报告的定性现象是：相对参考策略的 KL 增大时，代理奖励继续上升，而用更强评价测到的真实偏好先升后降。后一段就是奖励过优化（reward overoptimization）。

Goodhart 规律在这里的具体形式：一旦 $r_\phi$ 成为训练目标，策略会利用 $r_\phi$ 的漏洞（长度、套话、重复高分短语），这些漏洞在人的比较里只是相关，不是目标本身。

$\beta$ KL 把策略限制在 $\pi_{\mathrm{ref}}$ 附近，也就是限制在奖励模型见过的分布附近。$\beta$ 不是越大越好：过大时策略几乎停在 SFT，偏好数据里真正的内容改进也用不了。验证时同时看代理奖励、人的留出比较、以及平均长度这类风格统计。只有第一项上升、后两项变差，就是过优化，而不是训练尚未收敛。

<br>


# 有界理性

人并不总是选 $U$ 最大的项。反演若假设完全最优，次优示范的似然是 0，奖励没有有限估计。

Simon 的有界理性：决策者在计算与信息有限时采用够用（satisficing）规则，而不是在完整菜单上最大化。可计算的参数形式是 Luce 的 softmax 选择，也叫 Boltzmann 理性：

$$
P(a\mid s)
=
\frac{e^{\beta Q(s,a)}}{\sum_{a'}e^{\beta Q(s,a')}}.
$$

$\beta\to\infty$ 时质量集中在 $Q$ 最大的动作，成为最优策略。$\beta\to 0$ 时趋向均匀，选择几乎与 $Q$ 无关。量化反应均衡（quantal response equilibrium）把同一 softmax 用在博弈的每个参与者上，均衡是彼此最佳软反应的不动点。

<br>

## 例子

真实 $\beta=1$，两个动作 $Q=(2,\ 1)$。

$$
P(a_1)=\sigma(1)=0.7311.
$$

反演模型误用 $\beta=5$，为了拟合 $0.7311$，必须把效用差设成

$$
\Delta Q
=
\frac{\mathrm{logit}(0.7311)}{5}
=
0.2.
$$

尺度被温度吸收：同样的选择概率，既可以是「差距大但很随机」，也可以是「差距小但很确定」。只观测选择、不单独识别 $\beta$ 时，报告的 $Q$ 差只确定到一个正的乘性常数。

若模型取 $\beta=\infty$，要求示范动作唯一最优。只要数据里出现过一次 $a_2$，似然就是 0，极大似然不存在。给演示数据加 Boltzmann 噪声，似然才对次优动作有正概率。

<br>

## 匹配

最大熵逆强化学习（Ziebart et al., 2008）在轨迹空间写 $P(\tau)\propto e^{R(\tau)}$，线性奖励 $R(\tau)=w^{\top}\mu(\tau)$，$\mu$ 为特征计数。对数似然对 $w$ 的梯度为

$$
\mu(\tau_{\mathrm{demo}})
-
\mathbb{E}_{\tau\sim\pi_w}[\mu(\tau)].
$$

再次出现「观测特征减去模型期望特征」。内层期望由当前 $w$ 的软最优策略给出，所以每一步外层梯度都要解或近似一个软强化学习问题。奖励的尺度与 softmax 温度同样绑在一起，实践中固定温度或固定 $\|w\|$。

<br>

## 步

两条轨迹，特征 $\mu(\tau_1)=(1,0)$，$\mu(\tau_2)=(0,1)$，示范是 $\tau_1$。$w=(0,0)$ 时策略均匀，$E[\mu]=(0.5,0.5)$，梯度

$$
(1,0)-(0.5,0.5)=(0.5,-0.5).
$$

步长 $1$ 之后 $w=(0.5,-0.5)$。两条轨迹的奖励是 $0.5$ 与 $-0.5$，

$$
P(\tau_1)=\sigma(1)=0.7311,\qquad E[\mu]=(0.7311,\ 0.2689).
$$

新梯度是 $(0.2689,\ -0.2689)$。再走一步，$w\approx(0.769,\ -0.769)$，$P(\tau_1)\approx 0.823$。梯度的范数每步下降，示范特征与模型期望靠近。$w$ 的正负方向就是「示范有、另一条没有」的坐标；尺度若再乘温度，同一步的 $w$ 会整体放大，概率不变的条件是温度一起改。

Ng 与 Russell（2000）把同一匹配写成可行性：线性奖励下，专家策略最优当且仅当对每个其他策略 $\pi$，

$$
w^{\top}(\mu_E-\mu_\pi)\ge 0.
$$

满足这组不等式的 $w$ 是一个锥，专家最优并不挑出唯一奖励。最大熵逆强化学习在这个锥里取使轨迹分布熵最大的那个 $w$，梯度才是上面的特征差。

<br>

## 可计算

全菜单 softmax 要求对所有动作求和。动作是任意长度字符串时，这个和就是配分函数 $Z(x)$，不能枚举。

三条把计算留在可行范围的路线：

- 成对 BT 只在数据里出现的两个 $y$ 上归一化，不求和。
- DPO 的 $Z(x)$ 在奖励差中相减消失，训练时不估计 $Z$。
- Plackett–Luce 若只观测到全排序的前 $k$ 位，乘积就在这 $k$ 步停下，不必列出其余字符串。

人的真实过程也常常不是全菜单 RUM。Tversky 的按方面剔除（elimination by aspects）先用一个属性删掉一批选项，再用下一个属性。这种过程可以违反 IIA，也可以产生顺序效应。用 logit 去拟合它，得到的 $w$ 是最佳 logit 近似，不一定是人在内部最大化的那组权重。Shah 等人（2019）讨论过：与其把某种偏差形式写死，不如去检验这种偏差能否从数据里学出来。Bobu 等人（2020）给出相反方向的经验：奖励推断里，过强的「人是复杂噪声最优」假设反而不如更简单的选择模型稳健。两条都说明，反演结果依赖于选择模型，而选择模型是假设。

<br>

## 剔除

下面是一个为了把概率算完而写死的剔除规则，用来看 IIA 在何处失败，不是 Tversky 论文里的拟合权重。选项是汽车、红巴士、蓝巴士。规则：以 $1/2$ 的概率只保留「私人交通」并在其中均匀抽取；若这一步被跳过，则在当前菜单上均匀抽取。汽车是唯一的私人交通。

菜单 $\{汽车, 红巴士\}$：

$$
P(\mathrm{car})=1/2+1/2\cdot 1/2=3/4,\qquad P(\mathrm{red})=1/4.
$$

概率比是 $3$。菜单加上蓝巴士：

$$
P(\mathrm{car})=1/2+1/2\cdot 1/3=2/3,\qquad P(\mathrm{red})=1/2\cdot 1/3=1/6.
$$

概率比是 $4$。同一规则、同一对选项，菜单一变，概率比从 $3$ 变成 $4$。softmax 在任何菜单上都保持这个比。用 logit 去拟合这组选择，得到的 $w$ 是折中，换菜单后预测会偏。

<br>


# 社会选择

偏好剖面是所有投票人的排序。社会排序函数把剖面映到一个社会排序。社会选择函数把剖面映到一个获胜项。

<br>

## 例子

三名投票人，三选项：

| 投票人 | 排序 |
| :--- | :--- |
| 1 | $A\succ B\succ C$ |
| 2 | $B\succ C\succ A$ |
| 3 | $C\succ A\succ B$ |

成对多数：$A$ 赢 $B$（投票人 1、3），$B$ 赢 $C$（1、2），$C$ 赢 $A$（2、3）。多数关系循环，没有 Condorcet 胜者。

Borda 计数：第一名 2 分，第二名 1 分，第三名 0 分。

$$
A:\ 2+0+1=3,\qquad
B:\ 1+2+0=3,\qquad
C:\ 0+1+2=3.
$$

Borda 打成平局。它满足帕累托（全体把 $A$ 放在 $B$ 前时，$A$ 的分数一定更高），但社会对 $A$ 与 $B$ 的比较会受它们与 $C$ 的相对名次影响，因此不满足下面的 Arrow IIA。

<br>

## Arrow

Arrow 不可能定理（1951）。选项不少于 3 个时，不存在定义在全部严格排序剖面上的社会排序函数，同时满足：

1. 无限制域（unrestricted domain）：每个人的任意严格排序都允许。
2. 弱帕累托：若所有人都把 $a$ 排在 $b$ 前面，社会也把 $a$ 排在 $b$ 前面。
3. 无关方案独立性：社会关于 $a$ 与 $b$ 的名次，只依赖每个人关于 $a$ 与 $b$ 的名次，不依赖他们如何排列其他选项。
4. 非独裁：不存在这样一个人，其排序总是等于社会排序。

Luce 的 IIA 说的是同一个人在不同菜单上的选择概率比。Arrow 的 IIA 说的是社会名次对剖面里「无关项」的不敏感。logit 模型满足前者，不因此提供一种满足后者的投票规则。

<br>

## 防策略

Gibbard–Satterthwaite 定理（Gibbard 1973，Satterthwaite 1975）。选项不少于 3 个，社会选择函数定义在全部严格排序上，且每个选项都在某个剖面上当选（满射）。若它是防策略的（strategy-proof：没有人能靠谎报排序使结果按自己的真实偏好变好），则它是独裁的。

只在两个选项上的多数票是防策略的，因为定理的选项数假设不满足。认可投票（approval voting）的选票不是一个严格全序，也不在这条定理的定义域里。随机化社会选择同样需要另述。这些边界说明定理否定的是「全域、确定、满射、防策略、非独裁」这一组合，不是否定任何形式的合并。

对偏好学习的直接后果：把所有标注者的比较倒进一个 BT 模型，得到的是在 logit 噪声假设下的基数合并，即比较次数加权的平均效用。它不是一个定义在严格排序上的防策略社会选择函数。标注者若知道自己的比较会移动系统奖励，Gibbard–Satterthwaite 说明不能指望一个确定的非独裁规则让所有人都愿意报出真实全序。实践上要靠任务设计（成对具体例子、随机抽问、不展示系统当前偏好）降低操纵空间，而不是靠一个满足全部公理的投票公式。

<br>

## 规则

七名投票人，用来把三种计票分开：

| 人数 | 排序 |
| ---: | :--- |
| 3 | $A\succ B\succ C$ |
| 2 | $B\succ C\succ A$ |
| 2 | $C\succ B\succ A$ |

相对多数（plurality）只数第一名：$A$ 得 3，$B$ 得 2，$C$ 得 2，$A$ 当选。

成对多数：$B$ 对 $A$ 是 $4$ 比 $3$（后两组都把 $B$ 放在 $A$ 前），$B$ 对 $C$ 是 $5$ 比 $2$，$C$ 对 $A$ 是 $4$ 比 $3$。$B$ 是 Condorcet 胜者。

Borda，第一名 2 分、第二名 1 分：$A$ 得 $3\times 2=6$，$C$ 得 $2\times 1+2\times 2=6$，$B$ 得 $3\times 1+2\times 2+2\times 1=9$。$B$ 当选。

Copeland 分数是击败的对手数：$B$ 击败两人，得 2；$C$ 击败 $A$，得 1；$A$ 得 0。

同一张票，$A$ 与 $B$ 可以分别当选。报告「人更喜欢 $A$」之前要写明用的是第一名、成对多数还是 Borda。

<br>

## 操纵

另一张票说明相对多数可以被改票：

| 人数 | 排序 |
| ---: | :--- |
| 5 | $A\succ B\succ C$ |
| 4 | $B\succ C\succ A$ |
| 3 | $C\succ B\succ A$ |

如实投票时 $A$ 以 5 票当选。三名把 $C$ 放在第一、$B$ 放在第二的投票人更喜欢 $B$ 而不是 $A$。他们若改报 $B$ 为第一，则 $B$ 得 $4+3=7$，$A$ 仍得 5，$B$ 当选。谎报使结果按他们的真实偏好变好，所以这张相对多数不是防策略的。Gibbard–Satterthwaite 说的是：只要选项不少于三个、规则又要满射并且对所有排序都有定义，这种可操纵性就无法在保持非独裁的同时消失。

两个选项时，May（1952）的刻画把多数票定下来：匿名（谁投票不影响规则）、中立（把两个选项的名字对调，结果也对调）、正响应（有人把票改向 $a$ 时，$a$ 不会从当选变成落选，平局会变成 $a$ 当选）。这三条一起等价于简单多数。选项数一到三个，Arrow 的前提成立，这条刻画不再有非独裁的推广。

单峰偏好是另一种退路。选项排在一条线上，每人有一个最优点，离最优点越远越差。Black（1948）证明中位投票人的最优点能在成对多数下击败任何其他点。五个最优点 $1,2,2,3,4$，中位是 $2$。改到 $3$：最优点在 $1$ 与 $2$ 的三人反对，最优点在 $3$ 与 $4$ 的两人赞成，$2$ 仍获胜。定义域收成单峰之后，中位数是 Condorcet 胜者；定理的「无限制域」不再满足。

<br>


# 公平聚合

噪声模型：存在一个正确排序，每个人独立地以概率 $p$ 把它报对。异质性模型：每个人有自己的 $w$，没有「报对」这回事。同一个 70% 对 30% 的分裂，两种模型都解释得了。区分它们要靠重复：同一人在同一对上反复标注仍波动，支持噪声；同一人稳定、人与人之间不同，支持异质。潜类与混合 logit 拟合的是后者。

<br>

## 陪审团

孔多塞陪审团定理。$n$ 为奇数，投票人独立，每人正确概率 $p>1/2$，则多数正确的概率随 $n$ 增大，且趋向 1。

$p=0.6$、$n=3$：

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

$0.648>0.6$，多问两个人已经提高正确率。$p<1/2$ 时同样的公式会把多数推向错误。$p$ 在人群里不相同、误差相关（同一篇错误说明被所有人读到）时，定理的假设不成立，人数增加不一定变准。

这条定理只在「存在正确选项」时给多数票背书。风格与价值问题没有外部正确排序，增加标注者只会更精确地估计各群体的比例，不会收敛到一个客观排序。

<br>

## 加权

合并 BT 的对数似然是各条比较的和。一个 8 人群体若提供了 80% 的比较，梯度里就有 80% 的项来自他们。少数群体的稳定偏好会被拟合到残差里。留出准确率可以是 0.8，同时在少数群体的对上只有 0.2：总体对数似然并不惩罚这种倾斜，因为少数样本的权重小。

报告时至少分开三组数字：全体对数损失、各群体自己的对数损失、以及群体之间对同一对的分歧率。分歧率高而群体内重复标注稳定，说明该用混合系数或分开的奖励头，而不是加噪声温度。

评价已经训练好的分类或排序策略时，常用的分组定义与偏好群体是同一件事的两种切法：

- 人口统计均等（demographic parity）：预测为正的比例在两组相同。
- 机会均等（equalized odds）：在真实标签的每一类内部，预测为正的比例在两组相同。

它们约束的是策略输出，不是标注合并。一个在多数标注上 BT 似然很高的奖励，优化出来的策略仍可能在某一组上违反机会均等。分组指标要在策略上单独计算，不能从训练损失读出。

<br>

## 数字

十个人各比较一次「短回答」与「长回答」。八人稳定偏好短，两人稳定偏好长。合并样本里短回答的胜率是 $0.8$，

$$
s_{\mathrm{short}}-s_{\mathrm{long}}=\mathrm{logit}(0.8)=\log 4=1.3863.
$$

用这个差做预测，短回答永远判赢。在多数人的标签上准确率是 $1$，在少数人的标签上准确率是 $0$，全体准确率是 $0.8$。对数损失若按全体平均，少数人的 $-\log 0.2=1.609$ 只占两成权重。把两组的对数损失分开写，才会看到 $0$ 与 $1.609$ 同时存在。

$p=0.6$ 的陪审团从 $n=3$ 增到 $n=5$：至少三人正确的概率是

$$
\binom{5}{3}0.6^3 0.4^2
+
\binom{5}{4}0.6^4 0.4
+
0.6^5
=
0.6826.
$$

高于 $n=3$ 时的 $0.648$。增加的是「有正确答案且误差独立」时的正确率，不是价值分歧下的代表性。

分组误差可以与人口统计均等同时发生。每组 10 个正例、10 个负例。甲组：8 个正例被判正、2 个负例被判正，真正例率 $0.8$，假正例率 $0.2$，准确率 $0.8$。乙组：5 个正例被判正、5 个负例被判正，两个率都是 $0.5$，准确率 $0.5$。两组的预测为正的比例都是 $10/20=0.5$，人口统计均等成立。真正例率 $0.8\neq 0.5$，机会均等不成立。只看整体准确率 $(0.8+0.5)/2=0.65$，这组差距不出现。

<br>


# 实践

项目占课程成绩的一半，可个人、两人或三人；三人需要贡献说明。2026-11-04 前提交预分析计划，占项目分的 30%：动机、至少三篇文献、方法与分析、时间与分工、NeurIPS 版式 3–4 页（不含参考文献）。2026-12-03 提交终稿与代码。终稿占项目分的 60%，NeurIPS 版式至多 8 页（不含参考文献）。代码占 10%：README、依赖版本与随机种子固定，并能复现文中图表。两组截止合计 3 天迟交额度。

使用生成式工具时，提交须另附三段各约 150 词的说明：如何处理抄袭、不实与偏差；工具如何改变了自己的分析；社会、伦理与误用风险。缺任何一段，课程主页写明扣除课程总成绩的 10%。

大纲列出的方向包括：偏好学习的新应用、选择模型的比较、异质性与聚合、主动查询、RLHF 的变体、具体场景里的辅助博弈、小数据下的偏好模型、反演与偏差校正。

<br>

## 清单

1. 写明菜单是什么、观测是选择、成对还是排序，显示偏好还是陈述偏好。
2. 效用钉住一个参照，或写明先验方差。比较图不连通就不要解释跨支的分数差。
3. 同时报告对数损失与准确率。准确率只在预测概率远离 $1/2$ 的对上会看起来很好。
4. 有新用户或新物品时，用特征或低秩因子；只有截距的模型在表格里标成不可外推。
5. 策略优化同时报告代理奖励、留出成对胜率、相对 $\pi_{\mathrm{ref}}$ 的 KL，以及长度等风格均值。代理单独上升而留出胜率下降，按过优化处理。
6. 标注者多于一人时，给出群体内一致性与群体间分歧。分歧稳定就不要只用合并 BT。
7. 主动查询报告选择准则（熵、BALD 或 $\log\det$）以及与随机查询的留出对数损失差。
8. 代码从干净环境按 README 跑通，种子固定，外部代码保留许可与修改说明。

<br>

## 预分析

预分析在跑实验之前把下面几段写死，终稿再填数字。

- **观测**：菜单、比较还是排序、显示还是陈述、标注者是否分组。
- **参数**：参照项钉在哪里；若用 $\pi_{\mathrm{ref}}$，它是哪一个冻结模型；$\beta$ 的候选值。
- **查询**：若主动抽样，写下熵、BALD 或 $\log\det$ 中的哪一个，以及对照的随机查询。
- **成功**：留出对数损失、分组对数损失、KL、平均长度。只写代理奖励不算成功标准。
- **失败时如何解释**：强连通不成立，回到比较图；隐式奖励与留出人评背离，回到长度和格式；群体间分歧高于群体内重复误差，回到分组似然。

<br>


# 教材

书在 [mlhp.stanford.edu](https://mlhp.stanford.edu/)，Sang T. Truong、Andreas Haupt、Sanmi Koyejo，页眉日期 2026-09-23。侧栏是引言、五章、结语、附录。

- 引言，[首页](https://mlhp.stanford.edu/)
- 1 Foundations，[chap1.html](https://mlhp.stanford.edu/src/chap1.html)
- 2 Learning，[chap2.html](https://mlhp.stanford.edu/src/chap2.html)
- 3 Action，[chap3.html](https://mlhp.stanford.edu/src/chap3.html)
- 4 Inversion，[chap4.html](https://mlhp.stanford.edu/src/chap4.html)
- 5 Aggregation，[chap5.html](https://mlhp.stanford.edu/src/chap5.html)
- 结语 Conclusion，附录 Appendices（Mathematical Notation、Glossary），链在侧栏

十分制季度的课时写在引言页：第 1 章约一周，第 2、3 章各约三周，第 4、5 章各约一周半。Bradley–Terry 似然、Gumbel、Luce、DPO 损失和 Arrow 四条，前面已经算过，下面用书上的编号和例子。

<br>

## 引言

书里的例子跨推荐、检索、机器人轨迹、排序，后半一直用语言模型对齐。

预训练（pretraining）最小化下一词交叉熵，(1.1)：

$$
\mathcal{L}_{\mathrm{pretrain}}
=
-\mathbb{E}_{x\sim\mathcal{D}}
\left[\sum_{t=1}^{T}\log\pi_\theta(x_t\mid x_{<t})\right].
$$

对数评分（logarithmic scoring rule）$S(p,y)=\log p(y)$ 是严格恰当的（strictly proper）。真实分布为 $q$ 时，期望得分 $\mathbb{E}_{y\sim q}[S(p,y)]$ 只在预报 $p=q$ 时最大。交叉熵因此把语料里的频率拉向模型概率。有害续写只要在语料里出现过，这项损失同样会抬高它的概率。

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

### Borda 与 IIA'

选项 \(y\) 的 Borda 分是随机一位投票人把它排在多少个其他选项之前，(5.1)：

\[
\mathrm{Borda}(y)
=
\sum_{i=1}^{n}\big|\{y'\neq y:y\succ_i y'\}\big|
=
\sum_{y'\neq y}\big|\{i:y\succ_i y'\}\big|.
\]

$m$ 个选项时，名次分 $m-1,\ldots,0$ 就是压过的人数。

IIA'（modified IIA）：两个剖面里，若每个投票人对 $y$ 与 $y'$ 的相对名次相同，并且两者之间隔着的选项个数也相同，社会选择不能在这两个剖面上把赢家从 $y$ 换成 $y'$。名次有多远可以影响结果。Borda 仍违反原来的 IIA。

### DPO 与 Borda

比较从参考策略抽出时，\(\pi_{\mathrm{ref}}\) 加权的 Borda 胜者是 (5.2)：

\[
y^\star
=
\arg\max_y
\mathbb{E}_{y'\sim\pi_{\mathrm{ref}}(\cdot\mid x)}
\big[\mathbf{1}[y\succ y'\mid x]\big].
\]

DPO 最优策略与 $\pi_{\mathrm{ref}}$ 的密度比，正比于这个分数：对参考分布抽出的对手胜率越高，策略给它的概率相对 $\pi_{\mathrm{ref}}$ 越大。

三条里任何一条不成立，这个对应就断了。抽对不是按 $\pi_{\mathrm{ref}}$ 来的，更长或更有争议的回复被抽得更多，计票规则就变了。标注池之间偏好不同，Borda 分跟着谁在标走。偏好不传递，或随上下文变，Bradley–Terry 设错，目标对不上任何一个 Borda 胜者。

### 单峰

选项在一条线上，每人一个最爱点（ideal point），离它越远越差，是单峰偏好（single-peaked preferences）。Black（1948）：中位投票人的峰是多数决的 Condorcet 胜者。五人的峰为 $\{2,4,5,7,9\}$，纯中位数是 $5$。Moulin（1980）：单峰域上，真实峰再并上若干固定的幻影峰（phantom peaks），得到的广义中位数同时防策略（strategy-proof）、满足帕累托。域不再单峰时，Gibbard–Satterthwaite 把防策略推回独裁。

### 可分

可分偏好（separable preferences）：把物件 $x$ 加进任何一个集合 $A$ 是否变好，只看 $x$ 单独是否好过空集，(5.3)。有用、无害、诚实（helpful, harmless, honest）可分时，可以按议题分别聚合。一条很有用的回复必然带一点伤害，两个议题绑在一起，分议题投票会把这种依赖丢掉。可分域上，满射、防策略、可分三条刻画按委员会投票（voting by committees）。这条规则一般达不到帕累托。

### 自由

Sen（1970）的帕累托自由人（Paretian liberal）有三条：minimal liberalism，每人在自己的私事上至少对一对选项说了算；Pareto，全体都偏好 $x$ 甚于 $y$ 时社会也如此；unrestricted domain，任何偏好剖面都允许。三条不能一起成立。

两人、一本书。$a$ 是拘谨者（Prude）读，$b$ 是随便者（Lewd）读，$c$ 是没人读。拘谨者 $c\succ a\succ b$，随便者 $a\succ b\succ c$。私事上，拘谨者使 $c$ 压过 $a$，随便者使 $b$ 压过 $c$。两人都觉得 $a$ 好于 $b$，帕累托使 $a$ 压过 $b$。$c$ 压 $a$、$a$ 压 $b$、$b$ 压 $c$，接成循环。内容审核里，一方要表达、一方要环境安全，偏好落在别人的结果上，是 nosy preference。

### 社区注释

Community Notes 用因子模型把「注释放得好」和「评分人与注释立场相近」拆开，(5.4)：

\[
u=\mu+\alpha+\beta+p^{\top}q+\varepsilon.
\]

$\mu$ 是总截距，$\alpha$ 是评分人偏严还是偏宽，$\beta$ 是注释本身的质量，$p^{\top}q$ 是评分人与注释在立场上的对齐。入选看的是控制对齐之后 $\beta$ 是否过阈值。极化注释会被盟友打高、被对手打低；朴素平均把立场混进质量，内积项把这一部分从 $\beta$ 里分开。形式和带因子的 Rasch 相同，多出来的是要估计的那一项是 $\beta$，不是票数。

### 情境与报价

Nissenbaum 的情境完整性（contextual integrity）：信息流动符合该场景的规范，隐私才算保住。五个参数是发送者（sender）、信息主体（subject）、接收者（recipient）、数据类型（data type）、传递原则（transmission principle）。心率交给教练用来调训练，和交给广告网络，数据类型相同，传递原则不同。用户点过同意，流动仍可以违反这个场景的规范。

维克里拍卖（Vickrey auction）里，赢的人付第二高的报价。自己的报价只决定赢或输，不决定付多少，如实报价是占优策略（dominant strategy），机制是 DSIC（dominant-strategy incentive compatible）。第一价格拍卖里，赢的人付自己的报价，均衡里会把报价压到真实价值下面，称为 bid shading。配置仍可能给估价最高的人，但要靠策略，不是 DSIC。

<br>

## 记号

附录里的符号和正文一致。$N$ 是投票人，$A$ 或 $Y$ 是选项，$\mathcal{L}(A)$ 是严格全序的集合。$F$ 输出社会全序，$f$ 输出一个赢家。$\sigma$ 是 logistic。$\pi_{\mathrm{ref}}$ 是冻结的参考策略。

<br>


# 复习

## 公式

- **Bradley–Terry**：$P(i\succ j)=\sigma(s_i-s_j)$。梯度是胜场减去期望胜场。参照分数钉死，否则 Hessian 奇异。
- **Gumbel**：独立标准 Gumbel 推出 softmax。均值里的 $\gamma$ 在差里消失。
- **Luce**：正概率下，菜单上的概率比等于成对概率比，等价于 softmax。
- **嵌套 logit**：$\lambda\in(0,1]$。$\lambda=1/2$、两辆效用为 0 的巴士对一辆效用为 0 的汽车时，$P(\mathrm{car})=\sqrt{2}-1$。
- **条件 logit 梯度**：选中特征减期望特征。
- **信息**：$p(1-p)xx^{\top}$。$p=1/2$ 时标量因子为 $1/4$。
- **KL 最优策略**：$\pi^\star\propto\pi_{\mathrm{ref}}e^{r/\beta}$。
- **DPO**：$u=\beta(\Delta_\theta-\Delta_{\mathrm{ref}})$，损失 $-\log\sigma(u)$，权重 $\sigma(-u)$。
- **对决遗憾**：两臂相对 Condorcet 胜者的 $\epsilon$ 之和。没有胜者时先换目标。
- **Boltzmann**：$\beta$ 与效用尺度不可同时从选择里识别。$\beta=\infty$ 给次优动作零概率。
- **Arrow**：至少三选项时，全域、弱帕累托、IIA、非独裁不能同时成立。
- **陪审团**：$p=0.6$、$n=3$ 时多数正确概率为 $0.648$，$n=5$ 时为 $0.6826$，且要求独立与共同正确答案。
- **存在性**：BT 的有限 MLE 对应胜利有向图强连通。$W_i=0$ 时 MM 更新无定义。
- **best-of-$n$**：诱导密度不超过 $n$ 倍参考密度，故 KL 不超过 $\log n$。
- **BALD**：边缘熵减后验样本的平均熵。两人给出 $0.2$ 与 $0.8$ 时约为 $0.193$；两人都给出 $0.5$ 时为 $0$。
- **相对多数**：七人票里 $A$ 靠第一名当选，$B$ 是 Condorcet 胜者且 Borda 分为 $9$。
- **Rasch**：$H_{ij}=U_i+V_j$。同一人的成对比较里 $U_i$ 消掉，得到 Bradley–Terry。$K$ 维点积不消掉。
- **Elo**：$\sigma(-0.5)\approx 0.3775$。$K=32$ 时爆冷约 $+19.92$ 分，预期结果约 $12.08$ 分。
- **强弱遗憾**：强遗憾取两臂 $\epsilon$ 的最大，弱遗憾取最小。两者之和才是两臂都计入。
- **qEUBO**：无噪声时，最大化查询内潜在值的后验期望，也最大化一步贝叶斯价值。
- **GRPO**：优势是组内奖励的 $z$ 分数，更新是截断重要性比减去对参考策略的 KL。
- **反演**：$P(B\mid M,C)\neq P(B\mid M)$。行为把心理状态和情境绑在一起。
- **Borda–DPO**：比较从 $\pi_{\mathrm{ref}}$ 抽出且 Bradley–Terry 设对时，密度比正比于加权 Borda 分。
- **单峰**：峰 $\{2,4,5,7,9\}$ 的纯中位数是 $5$。

<br>

## 综合

用同一组三物品比较把估计、信息与损失串起来。数据仍是 $A$ 胜 $B$ 三次、$B$ 胜 $A$ 一次，$A$ 胜 $C$ 四次、$C$ 胜 $A$ 一次，$B$ 胜 $C$ 三次、$C$ 胜 $B$ 一次。

1. 胜利有向图每一对都有双向边，强连通，MLE 有限。
2. 钉住 $s_A=0$，牛顿法或 MM 都得到 $(s_B,s_C)=(-0.8361,\ -1.6723)$。负对数似然 $7.085$。
3. $\hat P(A\succ C)=0.842$ 的 $p(1-p)=0.133$，低于另外两对的 $0.211$。再标一次时 $A$ 对 $C$ 的 $\log\det$ 增量最小。
4. 自由坐标的 Laplace 标准差约为 $0.91$ 与 $0.96$。把 $s_C$ 的不确定当成「已经知道 $A$ 远好于 $C$」会高估把握。
5. 前文 DPO 例子里隐式奖励差就是 $u=0.10986$，已经包含 $\beta$。BT 概率是 $\sigma(u)$，不必再把 $u$ 除以 $\beta$。把估计好的分数 $s$ 乘上另一个 $\beta$ 再送进 $\sigma(\beta(s_i-s_j))$，等于改了尺度；那一步与 DPO 的代换不是同一个 $\beta$。

<br>

## 讲次

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

<br>


# 参考文献

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

## 数值

- [NumPy](https://numpy.org/)：BT 牛顿法、信息矩阵与 DPO 标量损失都可以按上文直接跑。
- 逻辑回归与牛顿法的形状约定与 CS229 笔记相同：样本在第 0 维，参照类别或参照物品要钉住。

## 偏好学习

- PyTorch：奖励头与 DPO 的词元对数概率。参考模型用 `torch.no_grad()` 或 `detach`，否则 $\Delta_{\mathrm{ref}}$ 会把梯度送进 $\pi_{\mathrm{ref}}$。
- 高斯过程偏好：后验是非高斯的，Laplace 或期望传播是标配近似；直接对分类似然做共轭更新是错的。

<br>
