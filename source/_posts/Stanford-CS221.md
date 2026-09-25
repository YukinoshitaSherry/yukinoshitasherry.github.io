---
title: 斯坦福CS221：Artificial Intelligence
date: 2026-09-22
categories:
- 上斯坦福
tags:
- AI
desc: Stanford CS221 详细课程笔记：从反射模型与搜索，到 MDP、博弈、因子图、贝叶斯网与逻辑。
hidden: true
---

- **全名**
  - *Artificial Intelligence: Principles and Techniques*。
- **主讲**
  - 2025 Autumn：Percy Liang 等；
  - 2026 Spring：Moses Charikar、Zachary Robertson；
  - 往年还有 Sanmi Koyejo、Dorsa Sadigh 等，名单随学期变化。
- **形式**
  - 教师讲座、Problem Session 与 homework party；
  - 每周作业：书面推导加 Python，围绕一条应用；
  - 闭卷考试；可选项目或 AI Product Deep Dive 为额外加分。
- **先修**
  - 编程：CS106A、CS106B；
  - 离散数学：CS103；
  - 概率：CS109；
  - 线性代数：MATH51；
  - 建议：CS161、CS107。
- **入口**
  - [Stanford Bulletin：CS221](https://bulletin.stanford.edu/courses/1057301)
  - [2026 Spring 课表](https://stanford-cs221.github.io/spring2026/)
  - [2025 Autumn 课表](https://stanford-cs221.github.io/autumn2025/)
  - [模块讲义（Spring 2025）](https://stanford-cs221.github.io/spring2025/modules/)
  - [2026 Spring 作业规范](https://stanford-cs221.github.io/spring2026/homework.html)
  - [2025 Spring 作业题面（与 2026 Spring 八份作业对齐）](https://stanford-cs221.github.io/spring2025/)

<br>

> [!INFO]+ 版本范围
>
> Explore Courses / Bulletin 把 CS221 写成：在信息不完整（因而需要概率）与计算有限（因而需要算法）的条件下做决策的数学。列出的主题包括搜索（search）、约束满足（constraint satisfaction）、博弈（game playing）、马尔可夫决策过程（Markov decision processes）、图模型（graphical models）、机器学习（machine learning）与逻辑（logic）。
>
> 算法与记号以 Percy Liang 一系的模块讲义为准（反射 / 状态 / 变量 / 逻辑四条建模轴，以及建模-推断-学习）。正文按 2026 Spring 课表组织：学习、搜索、MDP、博弈、因子图与束搜索、贝叶斯网、逻辑。2025 Autumn 另讲语言模型、AI 与社会、AI 供应链，按课堂扩展补入。后续学期若微调讲座题目，四条建模轴与 UCS / A* / 值迭代 / minimax / 变量消去通常仍在。

<br>


# 总览

- **反射模型（reflex-based models）**
  - 线性预测器（linear predictor）、特征（feature）、损失（loss）与随机梯度下降（SGD）；
  - 铰链（hinge）、逻辑（logistic）与平方损失；
  - 非线性特征与浅层网络；推断是一次前向计算。
- **状态模型（state-based models）**
  - 搜索：后继、代价、回溯、DFS / BFS / UCS / A*、启发式（heuristic）与松弛（relaxation）；
  - MDP：转移、回报、策略、值迭代、Q 学习；
  - 博弈：expectimax、minimax、α-β 剪枝、估值函数。
- **变量模型（variable-based models）**
  - 因子图（factor graph）与约束满足（CSP）；
  - 回溯、弧一致（arc consistency）、束搜索（beam search）、变量消去（variable elimination）；
  - 贝叶斯网（Bayesian network）、条件独立、精确推断与粒子滤波（particle filter）。
- **逻辑模型（logic-based models）**
  - 命题逻辑的语法、语义、蕴含与消解（resolution）；
  - 一阶逻辑（first-order logic）与从语言到逻辑。
- **课堂扩展**
  - 语言模型（language models）；
  - AI 与社会、供应链与评估协议。

- **记录逻辑**

$$
\text{任务}
\rightarrow\text{模型族}
\rightarrow\text{推断问题}
\rightarrow\text{学习或指定参数}
\rightarrow\text{复杂度与近似}
$$

- **分数校准**
  - 训练损失下降不等于搜索代价最优，也不等于策略回报最大；
  - UCS 展开的结点数不是墙钟时间的全部；启发式必须先满足一致性再谈加速；
  - 贝叶斯网的边缘概率与 MAP 赋值回答不同问题。
- **实现要求**
  - 能够从零实现 SGD、UCS / A*、值迭代、minimax 与 α-β、回溯加 AC-3、变量消去；
  - 能够说明优先队列键、一致启发式、探索率与粒子权的实际语义。
- **主线**
  - **前半：从一次动作到一条路径**
    - 反射模型输出单个标签；搜索输出动作序列。
  - **后半：随机、对手与分解**
    - 后继不确定则改 MDP；对手存在则改博弈；状态爆炸则改变量分解。

$$
\begin{aligned}
\text{特征与损失}
&\rightarrow\text{SGD}
\rightarrow\text{反射预测},\\
\text{后继与代价}
&\rightarrow\text{UCS / A*}
\rightarrow\text{最小代价路径},\\
\text{转移与回报}
&\rightarrow\text{值迭代 / Q 学习}
\rightarrow\text{策略},\\
\text{因子}
&\rightarrow\text{回溯 / 变量消去}
\rightarrow\text{赋值或边缘}.
\end{aligned}
$$

> [!NOTE]+ 概念链
>
> - 建模-推断-学习把每个题目拆成三步，而不是先写代码。
> - 反射：一次前向，$f(x)$ 直接出标签。
> - 搜索：确定性后继，求最小代价路径。
> - MDP：后继随机，求最大期望回报的策略。
> - 博弈：后继由对手或机会决定，求博弈值。
> - 因子图 / CSP：变量赋值，求最大权或可行解。
> - 贝叶斯网：带概率的因子，求边缘或 MAP。
> - 逻辑：公式与证明，求蕴含。
> - 语言模型与社会专题处理部署，不改前面的推断定义。

<br>

> [!INFO]+ 材料分工
>
> - 模块讲义：定义、递推、正确性证明、启发式条件；作业核对此类公式与状态表示。
> - 课堂：情感分类、路径规划、Mountain Car、Pac-Man、排课、车辆跟踪、语言到逻辑。
> - 换模型族与换启发式 / 探索率是两件事，不宜绑在同一次实验里。

<br>


## 数学预备

课程反复使用六类对象。

- **预测器与损失**
  - 反射模型是 $f_w:\mathcal{X}\rightarrow\mathcal{Y}$；
  - 损失 $\mathrm{Loss}(x,y,w)$ 把一次预测变成标量；
  - 经验风险是训练集上损失的平均。
- **梯度**
  - SGD 每步用一个样本的 $\nabla_w\mathrm{Loss}$；
  - 铰链损失在间隔大于 $1$ 时梯度为 $0$。
- **图上的路径**
  - 搜索状态是图的顶点，动作是边；
  - $\mathrm{FutureCost}(s)$ 是从 $s$ 到终点的最小剩余代价；
  - $\mathrm{Cost}$ 越小越好，与 MDP 的回报 $R$ 符号相反；
  - 优先队列的键必须与已证的不变量一致，否则 UCS 不再正确。
- **期望与策略**
  - MDP 的值是对转移与策略的期望回报；
  - 贝尔曼方程把期望拆成一步奖励加后继值；
  - 确定、$R=-\mathrm{Cost}$、$\gamma=1$ 时 $V_{\mathrm{opt}}=-\mathrm{FutureCost}$。
- **因子与边缘**
  - 联合分布或权分解为若干因子的乘积；
  - 边缘化是对部分变量求和；MAP 是取最大。
- **蕴含**
  - $\mathrm{KB}\models\phi$ 表示所有满足知识库的赋值都满足 $\phi$；
  - 消解是完备的证明规则，不是启发式搜索。

## 复杂度

- **时间：先问哪一个维度增长**
  - 线性模型一步 SGD：$O(d)$；
  - UCS / A*：与展开的状态数及优先队列操作有关，最坏指数；
  - 值迭代：每轮 $O(|S|^{2}|A|)$（稠密转移）或 $O(|S||A|b)$（分支因子 $b$）；
  - minimax：深度 $d$、分支 $b$ 时 $O(b^{d})$；α-β 最好约 $O(b^{d/2})$；
  - 变量消去：指数于最大中间因子的宽度（induced width）；
  - HMM 前向：$O(T|Z|^{2})$；粒子滤波：$O(TK)$，另加样本方差。
- **空间**
  - 反射模型只存 $w$；
  - UCS 存 frontier 与 visited；
  - 值迭代存 $|S|$ 或 $|S||A|$；
  - 贝叶斯网精确推断的中间表可以比输入因子大几个数量级。

## 数值计算

- 优先队列键相等时必须有确定的打破规则，否则自动评分会因结点展开次序失败。
- 折扣 $\gamma=1$ 且无吸收态时，值迭代可以发散。
- 概率与权在 log 域相加，避免下溢。
- α-β 的窗口初始化必须是真正的 $\pm\infty$，不能用一个“很大的数”冒充。
- 粒子滤波权必须归一化；有效样本量过低时再重采样。

## Python 实现基础

- 状态必须可哈希：元组、冻结集合，不要用可变 list 当字典键。
- 后继函数返回的是新状态，不要原地改当前状态。
- 作业自动评分对浮点与结点计数都严格，先写小图对拍。

```python
import heapq
import numpy as np
from collections import defaultdict

def reconstruct_actions(prev, end):
    """prev[s] = (parent, action)。返回从起点到 end 的动作序列。"""
    acts = []
    s = end
    while True:
        p, a = prev[s]
        if p is None:
            break
        acts.append(a)
        s = p
    acts.reverse()
    return acts
```

<br>


# 建模框架

## 概念：框架

- **问题**
  - 把真实任务写成模型、推断问题与可选的学习问题；
  - 同一任务可以落在反射、状态、变量或逻辑四条轴上。
- **范围**
  - 固定课程的建模-推断-学习（modeling-inference-learning）用语；
  - 输入是任务描述，输出是模型族与要算的量。
- **章节衔接**
  - 本章不引入具体算法；
  - 下一章用反射模型把三步走完一遍。

推断（inference）是“给定模型，回答问题”。学习（learning）是“给定数据，填模型里的未知量”。建模（modeling）是选择状态、变量或公式，使推断在计算上可行、在任务上够用。

> [!NOTE]+ 建模-推断-学习
>
> 是 CS221 贯穿全书的拆题法。来源是把 AI 写成“在有限计算下做决策”，而不是先选一个库函数。相对直接调求解器：先写清问的是路径、策略、赋值还是蕴含，再选算法。相对 CS229 的“假设类加损失”：这里同一损失可以配完全不同的推断（一次前向或一次搜索）。作业失败常见原因是模型写错，而不是 SGD 步长。

<br>

四条轴：

| 轴 | 对象 | 推断问什么 | 典型算法 |
| :--- | :--- | :--- | :--- |
| 反射 | 输入 $x$ | $f(x)$ | 前向、$\arg\max$ 分数 |
| 状态 | 状态与动作 | 路径或策略 | UCS、值迭代、minimax |
| 变量 | 变量与因子 | 赋值或边缘 | 回溯、变量消去 |
| 逻辑 | 公式 | 蕴含 | 模型检测、消解 |

同一任务可以落在不同轴上，问的量随之改变。以“给一组课程排时段”为例：

- 反射：把（课程、教室、学期）编成 $\phi$，一次输出时段标签。快，不保证教室不冲突。
- 状态：状态是“已排课程集合加当前课”，动作为选时段，代价为冲突惩罚。推断是 UCS。状态数随课程数指数增长。
- 变量：每门课一个变量，冲突写成因子。推断是回溯或 VE。变量数等于课程数。
- 逻辑：把“先修必须更早”“同一教师不同时”写成公式，问 KB 是否蕴含某条排法合法。输出是是 / 否或反模型，不是一张最优课表。

选轴的判据是：输出是单个标签、一条路径、一组赋值，还是一句必然为真的话。计算预算不够时，在同一轴上改近似（束搜索、粒子、截断深度），而不是把问题假装成反射。

> [!EXAMPLE]+ 例0 送货四轴
>
> 从仓库出发送完一组包裹再回去。反射：当前坐标进网络，输出下一步方向，不保证送完。搜索：状态 $(\mathrm{loc},S)$，UCS 给最小代价路径。MDP：路况随机，要策略而不是一条路径。变量：每件包裹的送达顺序或时段是变量，车容量是因子。逻辑：问“是否必然存在一条不超时的路线”，不是求那条路线。四句题面用词相近，模型族不同。

<br>

> [!NOTE]+ 四条轴
>
> 课程用轴来限制“智能体”这个词的外延。来源是把 AI 按表示切分，而不是按应用切分。相对“先选神经网络”：网络只覆盖反射轴的 $\phi$，搜索与消解不会因为换了网络而自动出现。相对 CS229：那里默认推断是一次 $h_{\theta}(x)$；这里推断本身可以是 NP 难的搜索或消去。

<br>


# 反射模型

## 概念：反射

- **问题**
  - 输入 $x$（邮件、评论文本、图像），输出单个动作或标签 $y$；
  - 不在输出时搜索未来状态。
- **范围**
  - 线性预测器、损失、SGD、特征模板与浅层网络；
  - 输入 $(x,y)$ 训练集，输出 $w$ 与预测。
- **章节衔接**
  - 上一章给出三步拆法；
  - 本章推断几乎免费，难点在损失与特征；
  - 下一章输出变成动作序列，推断不再是一次点乘。

讲义把智能体写成函数 $f$。反射模型（reflex-based model）在看到 $x$ 的当下就给出 $y$，不展开后继状态。建模是选 $\phi$ 与损失；推断是一次点乘或一次前向；学习是 SGD。

线性预测器（linear predictor）先算分数（score）再做决策：

$$
s=w\cdot\phi(x),\qquad
f_w(x)=\mathrm{sign}(s).
$$

回归去掉 $\mathrm{sign}$，直接输出 $s$。多类时对每个标签 $y$ 有一份特征 $\phi(x,y)$，

$$
f_w(x)=\arg\max_{y}w\cdot\phi(x,y).
$$

$\phi(x)\in\mathbb{R}^{d}$ 必须在训练与测试用同一套定义。$w_j$ 大表示第 $j$ 维特征把分数往正类推。

> [!NOTE]+ 反射模型
>
> 当前输入一经给出，输出由一次计算确定，不展开后继。来源是感知机与线性分类，课程把它当作最简单的智能体。相对状态模型：不能处理“这一步的后果要几步之后才结算”的任务。相对直接用深度网：线性加特征已经能做情感分类作业；网络只是把 $\phi$ 改成可学习的。

<br>

## 损失

真正关心的指标往往是零一损失（zero-one loss）$\mathbf{1}\{f_w(x)\ne y\}$，对 $w$ 几乎处处梯度为 $0$，不能直接 SGD。课程改用代理损失（surrogate loss）。经验风险（empirical risk）

$$
\mathrm{TrainLoss}(w)=\frac1n\sum_{i=1}^{n}\mathrm{Loss}(x^{(i)},y^{(i)},w).
$$

标签 $y\in\{-1,+1\}$，分数 $s=w\cdot\phi(x)$ 时：

| 损失 | 公式 | 梯度 $\nabla_w$ | 何时用 |
| :--- | :--- | :--- | :--- |
| 零一 | $\mathbf{1}\{ys\le 0\}$ | 几乎处处 $0$ | 报告准确率 |
| 感知机 | $\max(0,-ys)$ | $ys<0$ 时 $-y\phi$ | 只改错分 |
| 铰链 | $\max(0,1-ys)$ | $ys<1$ 时 $-y\phi$ | 还要间隔 |
| 逻辑 | $\log(1+e^{-ys})$ | $-\frac{y}{1+e^{ys}}\phi$ | 要概率 |
| 平方 | $(s-y)^{2}$ | $2(s-y)\phi$ | 回归 |

零一、感知机、铰链、逻辑都是 $ys$ 的单调降函数。铰链与逻辑是零一的凸上界：$\mathrm{Loss}_{\mathrm{0-1}}\le\mathrm{Loss}_{\mathrm{hinge}}$，$\mathrm{Loss}_{\mathrm{0-1}}\le\mathrm{Loss}_{\mathrm{logistic}}/\log 2$（差一个正常数）。最小化上界并不自动最小化零一，但给出可优化的凸目标。

> [!INFO]+ 铰链梯度
>
> $s=w\cdot\phi(x)$，$\mathrm{Loss}=\max(0,1-ys)$。在 $ys>1$ 处损失为 $0$，梯度为 $0$。在 $ys<1$ 处 $\mathrm{Loss}=1-ys$，
>
> $$
> \nabla_w\mathrm{Loss}=-y\,\phi(x).
> $$
>
> $ys=1$ 处不可微，次梯度可取 $0$ 或 $-y\phi$。更新 $w\leftarrow w+\eta y\phi(x)$ 只发生在间隔不足的点上。
>
> 感性理解：已经离开间隔带的点不再拉 $w$。零一损失只看对错；铰链还要求分数至少为 $1$。逻辑损失对所有点都给一点梯度，间隔很大时梯度按 $e^{-ys}$ 变小，但不是精确的 $0$。

<br>

> [!INFO]+ 逻辑损失梯度
>
> $\mathrm{Loss}=\log(1+e^{-ys})$。令 $u=-ys$，则
>
> $$
> \frac{\partial\mathrm{Loss}}{\partial s}
> =
> \frac{-y\,e^{-ys}}{1+e^{-ys}}
> =
> -\frac{y}{1+e^{ys}}.
> $$
>
> 再乘 $\partial s/\partial w=\phi(x)$。$ys\rightarrow+\infty$ 时该项 $\rightarrow 0$；$ys\rightarrow-\infty$ 时该项 $\rightarrow -y$，与铰链在错分侧同阶。
>
> 概率读法：$P(y=1\mid x)=\sigma(s)$，$\sigma(z)=(1+e^{-z})^{-1}$。逻辑损失等于 Bernoulli 负对数似然（差一个与 $w$ 无关的写法）。需要校准概率时用它，不需要时铰链通常更稀疏。

<br>

> [!NOTE]+ 损失
>
> 把一次预测变成可优化的标量。零一损失是目标指标，不可直接 SGD。铰链与逻辑都是零一的凸上界，来源是统计学习里用代理损失换可微性。相对平方损失套在 $\pm 1$ 标签上：分类更关心间隔，而不是分数离 $\pm 1$ 有多近。平方损失在 $|s|$ 很大且符号已对时仍给大梯度，会把已经分对的点继续往远处推。

<br>

## SGD

批量梯度下降每步用全体样本：

$$
w\leftarrow w-\eta\nabla_w\mathrm{TrainLoss}(w)
=
w-\frac{\eta}n\sum_{i=1}^{n}\nabla_w\mathrm{Loss}(x^{(i)},y^{(i)},w).
$$

随机梯度下降（SGD）每次抽一个样本（或一小批）

$$
w\leftarrow w-\eta_t\nabla_w\mathrm{Loss}(x^{(t)},y^{(t)},w).
$$

$\eta_t$ 是步长（step size / learning rate）。铰链与逻辑对 $w$ 凸，适当衰减的 $\eta_t$（例如 $\eta_t=\eta_0/\sqrt{t}$）使 $\mathrm{TrainLoss}$ 收敛到全局最小的邻域。网络损失非凸，只保证到驻点附近。

感知机更新是铰链在 $\eta=1$、且只在 $ys\le 0$ 时更新的特例。线性可分时感知机有限步到达某个分隔超平面，该平面不必最大间隔；不可分时振荡。

> [!NOTE]+ SGD
>
> 用单个样本梯度代替全训练集梯度。来源是 Robbins-Monro 随机逼近，不是为深度学习新造的。相对批量梯度：$n$ 很大时一步便宜，并能在读完数据前就动 $w$。代价是噪声，需要衰减 $\eta$ 或小批量。作业里步长过大则 `nan`，过小则几十轮损失不动。

<br>

正则（regularization）：在损失上加 $\frac\lambda2\|w\|_{2}^{2}$，梯度多一项 $\lambda w$。$L_2$ 把权拉向 $0$，降低有效容量。$\lambda$ 用验证集选，不能看训练损失。通常不惩罚截距维。情感分类里 $\lambda$ 过大会把稀有强极性词的权压没。

过拟合：训练零一下降、验证零一上升。对策是加 $\lambda$、早停、减特征，或加数据。课程不把双重下降当作作业要求。

多类铰链（对每个错误标签）

$$
\mathrm{Loss}(x,y,w)=\sum_{y'\ne y}\max\big(0,1+w\cdot\phi(x,y')-w\cdot\phi(x,y)\big).
$$

对某个 $y'$，若分数还没比正确类低出间隔 $1$，梯度把 $w$ 沿 $\phi(x,y')-\phi(x,y)$ 推开。结构化感知机是它在 $y$ 为整条路径时的特例，推断改成一次搜索。作业情感分类若是正负两类，不必上这一式。

> [!INFO]+ 感知机是铰链的特例
>
> 铰链在 $ys<1$ 时更新，感知机在 $ys<0$ 时更新。把铰链的间隔从 $1$ 改成 $0$，并取 $\eta=1$，即感知机。线性可分时感知机有限步到达某个分隔超平面，该平面不必最大间隔；间隔越大的铰链解对特征扰动更稳。不可分时感知机振荡，铰链加 $L_2$ 仍有唯一最小点。
>
> 感性理解：感知机只在已经分错时动手。铰链在“对了但太险”时也动手。逻辑损失对分得很对的点仍留一点梯度。三者都在优化 $ys$ 的上界，不是直接数错分数。

<br>

## 特征

特征模板（feature template）先写模式，再在数据上展开成具体维。情感分类常见：

- 词袋：$\phi_{\mathrm{good}}(x)=\mathrm{count}(\mathrm{good})$；
- 二元语法：$\phi_{\mathrm{not\_good}}(x)=\mathrm{count}(\mathrm{not}\ \mathrm{good})$；
- 否定范围：从 not / n't 到标点，把极性词取反；
- 截距：$\phi_0=1$，吸收正负类先验。

词表必须在训练集上冻结。测试出现的新词进 `UNK` 桶，不能在测试时给 $w$ 加维。大小写、标点、HTML 是否归一化，训练与测试必须一致。

> [!EXAMPLE]+ 例1 两个词的情感
>
> 评论只含 good / bad。$\phi=(\mathbf{1}\{\mathrm{good}\},\mathbf{1}\{\mathrm{bad}\},1)$。初值 $w=0$，铰链在所有点上都更新。若干步后 $w_1>0$、$w_2<0$。把 “not good” 仍编码成 good，则正权重被错误触发。这是特征问题，不是步长问题。加上二元语法或否定范围后，$w_{\mathrm{not\_good}}$ 应为负。

<br>

## 网络

线性模型的决策面是 $\phi$ 空间中的超平面。XOR 四点在原始坐标上不可分，必须换 $\phi$ 或加隐层。浅层网络

$$
h=g(Wx+b),\qquad
s=v\cdot h.
$$

$g$ 常用 $\mathrm{ReLU}(z)=\max(0,z)$ 或 $\sigma$。反向传播：$\partial\mathrm{Loss}/\partial s$ 先到 $v$，再经 $g'$ 到 $W$。课程只要求会把链式法则接到损失上。初始化不能全零，否则对称神经元收到同一梯度。

```python
def hinge_grad(w, phi, y):
    """y in {+1, -1}。返回梯度向量。"""
    if y * np.dot(w, phi) >= 1.0:
        return np.zeros_like(w)
    return -y * phi

def logistic_grad(w, phi, y):
    s = float(np.dot(w, phi))
    return -(y / (1.0 + np.exp(y * s))) * phi

def sgd_hinge(examples, dim, eta=0.1, num_epochs=5, l2=0.0):
    w = np.zeros(dim)
    for _ in range(num_epochs):
        for phi, y in examples:
            w -= eta * (hinge_grad(w, phi, y) + l2 * w)
    return w

def predict(w, phi):
    return 1 if np.dot(w, phi) >= 0 else -1
```

> [!WARNING]+ 标签约定
>
> 讲义与情感作业用 $y\in\{-1,+1\}$。若数据是 $\{0,1\}$，必须先映射，否则铰链的 $ys$ 符号错，梯度整支反号。多类 $\arg\max_y w\cdot\phi(x,y)$ 不要再套 $\mathrm{sign}$。

<br>

> [!NOTE]+ 对照：反射与搜索
>
> 反射一次算完 $f(x)$。搜索要返回整条动作序列，并比较整条路径的代价。情感分类用反射足够；迷宫或路径规划用反射会在岔路处贪心，全局代价可以任意差。

<br>

> [!NOTE]+ 衔接
>
> 反射模型的推断是点乘。下一章把输出改成路径：必须定义状态、后继与代价，再用图搜索而不是 SGD 来做推断。

<br>


# 搜索

## 概念：搜索

- **问题**
  - 从起始状态出发，找一条到终点的动作序列，使路径代价最小；
  - 后继确定：$\mathrm{Succ}(s,a)$ 只有一个。
- **范围**
  - 写出搜索问题定义，并给出回溯、DFS / BFS / UCS 与 A*；
  - 输入后继与代价，输出路径、代价与展开结点数。
- **章节衔接**
  - 上一章输出单个标签；
  - 本章输出动作序列，推断是图搜索；
  - 下一章后继变成分布，路径不再够用。

搜索问题（search problem）由输入 $x$ 构造：

- $\mathrm{Start}(x)$：起始状态；
- $\mathrm{Actions}(s)$、$\mathrm{Succ}(s,a)$、$\mathrm{Cost}(s,a)\ge 0$；
- $\mathrm{IsEnd}(s)$。

一条动作序列 $a_1,\ldots,a_T$ 诱导状态 $s_0=\mathrm{Start}$，$s_t=\mathrm{Succ}(s_{t-1},a_t)$。目标是在 $\mathrm{IsEnd}(s_T)$ 的序列中最小化 $\sum_t\mathrm{Cost}(s_{t-1},a_t)$。

状态必须包含未来决策所需的全部信息（Markov）。路径规划只送货时，状态是位置；还要送完一组包裹再回去，状态是 $(\mathrm{loc},S)$，$S$ 为未送集合。8 数码只存棋盘。漏字段会导致“看起来合法、全局不可行”的路径。

> [!NOTE]+ 搜索问题
>
> 把任务写成加权图上的最短路。来源是路径规划与谜题，课程把它当作第一条状态轴。相对反射：动作的后果会被后续动作补偿或放大，必须展开序列。相对 MDP：后继没有随机性，谈的是一条确定路径的代价，不是期望。

<br>

最小剩余代价（future cost）

$$
\mathrm{FutureCost}(s)
=
\begin{cases}
0 & \mathrm{IsEnd}(s)\\
\min_{a}\big[\mathrm{Cost}(s,a)+\mathrm{FutureCost}(\mathrm{Succ}(s,a))\big] & \text{否则.}
\end{cases}
$$

从起点看，$\mathrm{FutureCost}(\mathrm{Start})$ 就是最优路径代价。已付代价记为 $\mathrm{PastCost}(s)$。任意前缀满足 $\mathrm{PastCost}(s)+\mathrm{FutureCost}(s)$ 不小于最优总代价。

> [!INFO]+ FutureCost 递推
>
> 终点剩余为 $0$。非终点必须走一步 $a$，付出 $\mathrm{Cost}(s,a)$，然后从后继再付最优剩余。对 $a$ 取最小即最优。这是动态规划，不是新的搜索启发式。
>
> 若图有环且某条环代价为负，沿环无限转可使代价趋向 $-\infty$，该递推没有有限解。UCS / A* 要求边代价非负。零代价边允许，但不能有负边。
>
> 感性理解：从终点往回填表，每个格子写“从这里走出去还要至少花多少”。搜索算法从起点往前走，用已付代价去逼近这张表。

<br>

> [!EXAMPLE]+ 例2 送货状态
>
> 地图三点 $A,B,C$，边代价 $1$。要从 $A$ 出发经过 $B$ 与 $C$（顺序不限）回到 $A$。若状态只存位置，搜索会以为到过 $B$ 一次即可，可能漏掉 $C$。正确状态是 $(p,S)$，$S\subseteq\{B,C\}$。起点 $(A,\{B,C\})$，终点 $(A,\emptyset)$。状态数 $3\cdot 2^{2}=12$，不是 $3$。

<br>

## 树搜索

先分清搜索树与状态图。状态图的结点是状态，边是 $(\mathrm{Succ},\mathrm{Cost})$。搜索树的结点是从起点出发的路径：同一状态经两条路径到达，在树里是两个结点。树搜索可以处理负代价（只要加深度上限或无负环），图搜索用 `visited` 丢掉重复状态，省时间，但必须在“第一次见到即最优”时才能丢。

回溯（backtracking）递归选动作，走到终点时记录整条路径代价，再撤回。完整展开搜索树。时间 $O(b^{d})$，空间 $O(d)$。边代价任意；有负环时必须另加深度或代价下限，否则递归不终止。

```python
def backtracking(s, is_end, successors, path=None, cost=0.0, best=None):
    if path is None:
        path = []
    if best is None:
        best = [np.inf, None]
    if is_end(s):
        if cost < best[0]:
            best[0] = cost
            best[1] = list(path)
        return best
    for act, sp, c in successors(s):
        path.append(act)
        backtracking(sp, is_end, successors, path, cost + c, best)
        path.pop()
    return best
```

DFS：栈或递归，先走深。有环必须记住当前路径上的状态，否则无穷递归。不保证最短。图搜索版若在入栈时标记 visited，会剪掉“先到但不优”的路径：单位代价时碰巧对，边权不等时可以错。

BFS：队列按层扩展。边代价全为 $1$ 时，第一次出队的终点即最少步数。边权不等时“层数最少”可以不是代价最小。空间 $O(b^{d})$，浅层宽时内存先爆。

迭代加深（iterative deepening, ID-DFS）：深度上限 $0,1,2,\ldots$ 反复 DFS。空间同 DFS，单位代价下最优性同 BFS。同一浅层结点会被多次展开，常数更大，内存紧且边权为单位时可用。

| 算法 | 最优对象 | 条件 | 时间（最坏） | 空间 |
| :--- | :--- | :--- | :--- | :--- |
| 回溯 | 任意代价路径 | 无负环或有深度帽 | $O(b^{d})$ | $O(d)$ |
| DFS | 不保证 | 有环须记路径 | $O(b^{d})$ | $O(d)$ |
| BFS | 最少步 | 边权全为 $1$ | $O(b^{d})$ | $O(b^{d})$ |
| ID-DFS | 最少步 | 边权全为 $1$ | $O(b^{d})$ | $O(d)$ |
| UCS | 最小代价 | 边权 $\ge 0$ | 见下文 | frontier |
| A* | 最小代价 | 一致 $h$ | 随 $h$ 降 | frontier |

> [!NOTE]+ 回溯与 BFS
>
> 回溯枚举完整路径；BFS 按层扩单位代价图。来源是穷尽搜索，不是带启发的。相对 UCS：二者都不看边权差异。边权不等时 BFS 的“层数最少”可以不是代价最小。相对动态规划：树搜索可以把同一状态经不同路径展开多次。

<br>

## UCS

一致代价搜索（uniform cost search）维护优先队列（frontier），键是从起点已付代价 $g(s)=\mathrm{PastCost}(s)$。每次弹出键最小的状态并展开后继。弹出的第一个终点即最优路径。

图搜索实现另维护 `best[s]`：到 $s$ 已知的最小 $g$。同一状态可以多次入队（lazy deletion），弹出时若 $g$ 已不是 `best[s]` 则丢弃。真正的展开发生在弹出时，不是入队时。作业统计的展开结点数按弹出计。

不变量：边权非负时，第一次弹出 $s$，$g(s)$ 已是从起点到 $s$ 的最小代价。此后不必再打开 $s$。这与 Dijkstra 相同，只是课程从搜索问题而不是从显式图出发。

```python
def uniform_cost_search(start, is_end, successors):
    """successors(s) -> iterable of (action, next_state, cost)."""
    pq = [(0.0, start)]
    best = {start: 0.0}
    prev = {start: (None, None)}
    expanded = 0
    while pq:
        cost, s = heapq.heappop(pq)
        if cost > best.get(s, np.inf):
            continue
        expanded += 1
        if is_end(s):
            return reconstruct_actions(prev, s), cost, expanded
        for act, sp, c in successors(s):
            if c < 0:
                raise ValueError("UCS 要求非负代价")
            nc = cost + c
            if nc < best.get(sp, np.inf):
                best[sp] = nc
                prev[sp] = (s, act)
                heapq.heappush(pq, (nc, sp))
    return None, np.inf, expanded
```

> [!INFO]+ UCS 弹出即最优
>
> 边权 $\ge 0$。对弹出次序做归纳。起点以 $g=0$ 弹出，显然最优。设此前弹出的状态都已拿到最优 $g$。令 $s$ 是下一个弹出、键为 $g$ 的状态。若存在更短路径 $P$ 到达 $s$，看 $P$ 上第一个尚未弹出的点 $u$。$u$ 的前驱已弹出，故 $u$ 已在队列里，且
>
> $$
> g(u)\le\mathrm{PastCost}_{P}(u)\le\mathrm{PastCost}_{P}(s)<g(s).
> $$
>
> 非负边权给出第二个不等式。优先队列会先弹出 $u$，与“下一个弹出的是 $s$”矛盾。因此 $g(s)$ 最优。终点第一次弹出时即可停，不必清空队列。
>
> 感性理解：优先队列总是先结算“已经确定不会更便宜”的点。非负边权保证后面加上去的边不会让旧点突然变便宜。负边权破坏这件事：一条尚未走完的弯路可以在最后一步用负代价反超，弹出不再等于最优。零代价边允许，循环零环不会让 $g$ 下降，但可能让队列里堆很多等价路径，状态哈希必须挡住。

<br>

> [!NOTE]+ UCS
>
> 按已付代价从小到大结算状态。来源是 Dijkstra 最短路，课程把它写成搜索问题而不是邻接表算法。相对 BFS：边权不必为 $1$，键是代价和而不是步数。相对 A*：启发式恒为 $0$。相对动态规划：UCS 只保证碰到的状态最优，不必预先列出全部状态。

<br>

复杂度：设最优代价为 $C^{\ast}$，最小正边权为 $\varepsilon$，分支因子为 $b$。树 UCS 最坏展开约 $O(b^{1+C^{\ast}/\varepsilon})$ 个结点。图 UCS 至多展开每个状态一次，时间由 $|S|$、边数与堆操作决定。作业图通常稀疏，展开结点数随启发式（下一节）变化更明显。

> [!EXAMPLE]+ 例3 四结点加权图
>
> 状态 $S,A,B,G$。边 $S\xrightarrow{1}A$，$S\xrightarrow{5}B$，$A\xrightarrow{1}B$，$A\xrightarrow{6}G$，$B\xrightarrow{1}G$。UCS 弹出次序：
>
> 1. $S$，$g=0$。入队 $A$ 键 $1$、$B$ 键 $5$。
> 2. $A$，$g=1$。入队 $B$ 键 $2$、$G$ 键 $7$。旧的 $B$ 键 $5$ 仍在堆里，弹出时会被丢掉。
> 3. $B$，$g=2$。入队 $G$ 键 $3$。
> 4. $G$，$g=3$。停。路径 $S\rightarrow A\rightarrow B\rightarrow G$。
>
> BFS 按步数会先走 $S\rightarrow A\rightarrow G$（两步，代价 $7$），不是最优。这就是边权不等时不能用 BFS 冒充 UCS。

<br>

## A*

A* 的优先队列键是 $f(s)=g(s)+h(s)$。$g$ 是已付代价，$h$ 是启发式（heuristic），估计从 $s$ 到终点的剩余。$h\equiv 0$ 时退回 UCS。

可采纳（admissible）：$h(s)\le\mathrm{FutureCost}(s)$ 对一切 $s$。一致（consistent / monotone）：

$$
h(s)\le\mathrm{Cost}(s,a)+h(\mathrm{Succ}(s,a)),\qquad h(\mathrm{end})=0.
$$

一致是“启发式版三角不等式”。一致蕴含可采纳。一致时 A* 弹出 $s$ 仍有 $g(s)$ 最优，第一次弹出终点即可停，图搜索不必重新打开已弹出状态。只可采纳、不一致时，树 A* 仍最优（不丢结点），图 A* 必须允许同一状态在更小 $g$ 时再次入队。

```python
def astar(start, is_end, successors, h):
    pq = [(h(start), 0.0, start)]
    best = {start: 0.0}
    prev = {start: (None, None)}
    expanded = 0
    while pq:
        f, g, s = heapq.heappop(pq)
        if g > best.get(s, np.inf):
            continue
        expanded += 1
        if is_end(s):
            return reconstruct_actions(prev, s), g, expanded
        for act, sp, c in successors(s):
            ng = g + c
            if ng < best.get(sp, np.inf):
                best[sp] = ng
                prev[sp] = (s, act)
                heapq.heappush(pq, (ng + h(sp), ng, sp))
    return None, np.inf, expanded
```

> [!INFO]+ 一致推出可采纳
>
> 对从 $s$ 出发的一条最优剩余路径 $s=s_0\rightarrow\cdots\rightarrow s_T$（$s_T$ 为终点）逐边写一致性：
>
> $$
> h(s_t)\le c_t+h(s_{t+1}).
> $$
>
> 从 $t=0$ 接到 $t=T-1$：
>
> $$
> h(s)\le c_0+c_1+\cdots+c_{T-1}+h(s_T).
> $$
>
> $h(s_T)=0$，右边正好是 $\mathrm{FutureCost}(s)$。故一致 $\Rightarrow$ 可采纳。逆命题不成立：可采纳只约束每个点与终点，中间可以起伏。
>
> 感性理解：一致性说启发式沿真实边下降得不够快，不会“跳过”真实代价。可采纳只比较当前点与终点，中间可以先低估再高估。UCS 式的弹出不变量需要更强的一致性：否则队列键 $g+h$ 的次序不再与真实 $g$ 的次序同向。

<br>

> [!INFO]+ 只可采纳时图 A* 可能早停错
>
> 反例。边 $S\xrightarrow{1}A\xrightarrow{1}G$，$S\xrightarrow{3}G$。取 $h(S)=2$，$h(A)=0$，$h(G)=0$。
>
> - 可采纳：$\mathrm{FutureCost}(S)=2$，$h(S)=2$；$\mathrm{FutureCost}(A)=1$，$h(A)=0$。
> - 不一致：$h(S)=2\not\le 1+h(A)=1$。
>
> 键：$S$ 的 $f=0+2=2$。展开后 $A$ 的 $f=1+0=1$，$G$（直达）的 $f=3+0=3$。接着弹出 $A$，再弹出经 $A$ 的 $G$，代价 $2$，仍碰巧正确。把 $h(A)$ 改成 $0$、$h(S)=2$ 并加入旁路后，若实现“弹出即关闭、禁止再打开”，一条先到达但 $g$ 较大的路径可能把 $A$ 关掉，更短的 $g$ 再也进不来。作业若只保证可采纳，必须允许 decrease-key 或 lazy 再入队；若要求弹出即最优，必须证明一致。
>
> 感性理解：不一致的 $h$ 会让算法以为某条已经很贵的前缀“看起来更有希望”，先把它下面的点关掉。可采纳保证第一次弹出终点时，不存在更短的完整路径（树搜索）；图搜索多了“关掉中间点”这一步，需要一致性来撑住。

<br>

> [!NOTE]+ A*
>
> 在 UCS 的键上加剩余估计 $h$，少展开明显偏离终点的点。来源是 Nilsson 等人的启发式搜索。相对 UCS：$h=0$ 退回 UCS；$h$ 越紧（越大但仍可采纳），展开越少。相对贪心只看 $h$：贪心不把已付代价算进去，可以在单位代价图上也不最优。$h$ 不可采纳时，第一次弹出的终点不必最优。加权 A* 用 $g+\lambda h$（$\lambda>1$）进一步少展开，放弃最优性，课程作业一般不准用。

<br>

启发式常由松弛（relaxation）得到：丢掉原问题的一部分约束，再精确求解松弛问题。松弛后的最优代价自动可采纳：原问题的每条合法路径仍是松弛问题的合法路径，故松弛最优 $\le$ 原最优。若松弛后的代价仍满足三角不等式，则一致。

常见构造：

- 路径规划丢掉障碍或单行，启发式变成直线距离或无视拥堵的最短路。欧氏距离在四向网格上可采纳（真实走的是折线），在只准四向时不是最紧；曼哈顿距离 $|x_G-x|+|y_G-y|$ 在四向、单位代价、无障碍时恰好等于 $\mathrm{FutureCost}$，有障碍时仍可采纳且一致。
- 8 数码允许棋子重叠或瞬移：曼哈顿距离（各瓦到目标位置的曼哈顿之和）可采纳；线性冲突（linear conflict）在曼哈顿上再加 $2$，仍可采纳，支配曼哈顿。
- 送货问题丢掉“必须送完再回去”中的一部分包裹：剩余包裹的最小生成树或最近邻下界。

两个可采纳启发式 $h_1,h_2$，点态最大 $h(s)=\max(h_1(s),h_2(s))$ 仍可采纳；二者都一致则最大也一致。$h$ 越大（仍可采纳）展开越少，这叫支配（domination）。

> [!EXAMPLE]+ 例4 网格与改边权
>
> $3\times 3$ 网格，四向移动代价 $1$，从 $(0,0)$ 到 $(2,2)$。UCS 与 BFS 都给出代价 $4$。曼哈顿 $h=|2-x|+|2-y|$ 一致：向目标走一步，$h$ 减 $1$，代价加 $1$；走回头 $h$ 加 $1$，代价仍加 $1$。A* 不展开明显远离终点的格子，展开数小于 UCS。
>
> 边权不等时必须离开“步数”直觉。另作一小图：边 $S\xrightarrow{10}G$，$S\xrightarrow{1}A\xrightarrow{1}B\xrightarrow{1}G$。BFS（按步）先走到直达 $G$，代价 $10$。UCS 走 $S\rightarrow A\rightarrow B\rightarrow G$，代价 $3$。单位曼哈顿也不能再当启发式：它假设每步代价 $1$，会把直达边估得过低。可采纳的改法是用最小边权缩放，或在丢掉部分边之后再求松弛最短路。

<br>

## 动态规划

FutureCost 递推本身就是动态规划。状态图无环（或已按拓扑逆序）时，按该序填表，每个状态算一次，时间与边数成正比。有环且边权非负时，递推的不动点仍存在，但简单递归会循环；改 UCS / A*，或先检测环。

```python
def future_cost(s, is_end, successors, memo=None):
    if memo is None:
        memo = {}
    if s in memo:
        return memo[s]
    if is_end(s):
        memo[s] = 0.0
        return 0.0
    best = np.inf
    for act, sp, c in successors(s):
        best = min(best, c + future_cost(sp, is_end, successors, memo))
    memo[s] = best
    return best
```

该写法默认后继图无环。送货状态 $(p,S)$ 若规定取包裹后 $|S|$ 严格下降，则自然无环，表大小为 $|\mathrm{Loc}|\cdot 2^{|P|}$，指数在包裹个数上，不在地图边上。这是作业里“状态设计决定复杂度”的典型来源。

从起点做 UCS 与从终点填 FutureCost，在确定、非负、无折扣时给出同一最优值。差别是：UCS 不必列举全部状态；DP 一次给出所有状态的剩余代价，适合后面还要从许多起点查询。

> [!NOTE]+ 动态规划与搜索
>
> 同一套 FutureCost 方程，可以用记忆化递归、拓扑填表或 UCS 求解。来源是最短路的最优子结构，不是新的启发式。相对树回溯：同一状态只算一次。相对值迭代：这里没有随机后继，也没有折扣，方程里是 $\min$ 加代价，不是 $\max$ 加期望回报。

<br>

## 结构化感知机

搜索到此假定 $\mathrm{Cost}$ 已知。结构化感知机（structured perceptron）在路径上学习代价。输入 $x$，真实路径（或标签结构）为 $y$，特征 $\phi(x,y)$。当前 $w$ 下的推断

$$
\hat y=\arg\min_{y'}w\cdot\phi(x,y'),
$$

等价于把边代价写成特征的线性函数，再跑 UCS / A*。更新与反射感知机同形：

$$
w\leftarrow w+\phi(x,y)-\phi(x,\hat y).
$$

$\hat y=y$ 时不更新。推断必须（近似）最优，否则 $\hat y$ 不是当前 $w$ 下的最小代价结构，更新方向可以指向更差的 $w$。模块讲义标为可选，考试可能只问更新式与“推断是一次搜索”。

> [!NOTE]+ 结构化感知机
>
> 把感知机从单个标签推广到整条结构（路径或赋值）。来源是结构化预测，不是另写一套搜索。相对反射感知机：推断不再是点乘符号，而是一次最小代价推断。相对独立调每条边的手工代价：用对错路径的特征差来改 $w$。相对下一章 MDP：这里没有随机后继，损失仍是感知机式的对错，不是期望回报。

<br>

## 实现注意

- 状态表示错误（漏掉必须记住的信息）比算法写错更常见。漏字段的搜索问题定义了另一张图，UCS 会在那张错图上给出“正确”最短路。
- `visited` / `best` 在 UCS 与一致 A* 里按“弹出时结算”处理。DFS 若在入栈时标记，会剪掉合法但后到的更优路径。
- 状态必须可哈希。可变 `list` 当键会导致同一格子被当成不同对象，展开数爆炸。
- 后继函数必须返回新对象，不能原地改当前状态再入队。
- 作业按展开结点数计分时，优先队列键相等必须有确定打破规则（再比状态元组、再比动作名），并与参考实现一致。
- 启发式不可采纳时，自动评分会在最优代价上失败，不是展开数差一点。

> [!WARNING]+ 负代价与零环
>
> UCS / A* 的弹出不变量依赖边权 $\ge 0$。出现负边时改其他算法，或先证明图无负环并把问题改成 DP。全零环不会让 $g$ 下降，但会让不记 `best` 的树搜索无限走。图搜索用 `best` 挡住即可。

<br>

> [!NOTE]+ 对照：树搜索、UCS 与 A*
>
> 树回溯保证看见所有路径，指数在深度上。UCS 在非负边上按 $g$ 结算，第一次弹出终点即最优。A* 用一致 $h$ 少展开，不改变最优性。三者问的都是一条确定路径。贪心只看 $h$、BFS 只看步数，问的是另一个目标。

<br>

> [!NOTE]+ 衔接
>
> 搜索假定 $\mathrm{Succ}(s,a)$ 唯一。下一步动作若按概率落到若干后继，最小代价路径不再是决策对象：骰子把智能体打到路径外之后，预先写死的动作序列无定义。必须改成对每个状态规定动作，并对期望回报取最优。这就是下一章的策略与贝尔曼方程。

<br>


# MDP

## 概念：MDP

- **问题**
  - 动作的后继按 $T(s,a,s')$ 随机；
  - 目标是策略 $\pi$，使期望折扣回报最大。
- **范围**
  - 写出贝尔曼方程、策略评估、值迭代与 Q 学习；
  - 输入模型或轨迹，输出 $V$、$Q$ 或 $\pi$。
- **章节衔接**
  - 上一章后继确定；
  - 本章后继随机，路径换成策略；
  - 下一章随机来自对手而不是自然。

马尔可夫决策过程（Markov decision process, MDP）是五元组 $(S,A,T,R,\gamma)$：

- $S$：状态集；$A$：动作集，或每个 $s$ 一份 $A(s)$；
- $T(s,a,s')=\Pr(s_{t+1}=s'\mid s_t=s,a_t=a)$，后继分布；
- $R(s,a,s')$：该步回报（reward），有的讲义写成 $R(s,a)$；
- $\gamma\in[0,1)$：折扣（discount）。$\gamma=1$ 只在几乎必然被吸收时才有有限值。

马尔可夫：下一状态只依赖当前 $(s,a)$，不依赖更早历史。策略 $\pi$ 对每个状态给一个动作（确定性）或一个分布 $\pi(a\mid s)$。搜索是 $T$ 退化成确定后继、回报取代价相反数、且只要一条路径的特例。

> [!NOTE]+ MDP
>
> 在随机后继下最大化期望回报。来源是最优控制与 Bellman，课程把它写成搜索的概率版。相对搜索：必须对每个状态规定动作，因为骰子可能把智能体打到路径外。相对监督学习：没有现成的 $a^{\ast}$，数据分布还依赖正在执行的 $\pi$。相对 CS229 的同一套方程：记号在 CS221 里从搜索的 $\mathrm{Cost}$ 切到 $R$，值是“越大越好”，不要把 UCS 的最小代价直接抄进贝尔曼。

<br>

从 $s_0=s$ 出发、一直按 $\pi$ 走，折扣回报的期望是策略值（value）

$$
V_{\pi}(s)=\mathbb{E}_{\pi}\Big[\sum_{t=0}^{\infty}\gamma^{t}R(s_t,a_t,s_{t+1})\ \Big|\ s_0=s\Big].
$$

动作值 $Q_{\pi}(s,a)$ 先强制走 $a$，此后再按 $\pi$。最优值 $V_{\mathrm{opt}}(s)=\max_{\pi}V_{\pi}(s)$，$Q_{\mathrm{opt}}(s,a)=\max_{\pi}Q_{\pi}(s,a)$。确定性最优策略 $\pi_{\mathrm{opt}}(s)=\arg\max_a Q_{\mathrm{opt}}(s,a)$。有限 MDP、$\gamma<1$ 时最优值唯一，最优策略可以不止一个。

把回报拆成“今晚”与“明天以后”，用全期望定律：

$$
V_{\pi}(s)=\sum_{a}\pi(a\mid s)\sum_{s'}T(s,a,s')\big[R(s,a,s')+\gamma V_{\pi}(s')\big].
$$

这是贝尔曼期望方程（Bellman expectation equation）。最优值满足贝尔曼最优方程（Bellman optimality equation）

$$
V_{\mathrm{opt}}(s)=\max_{a}\sum_{s'}T(s,a,s')\big[R(s,a,s')+\gamma V_{\mathrm{opt}}(s')\big].
$$

$Q$ 的对应式更常用于学习：

$$
Q_{\mathrm{opt}}(s,a)=\sum_{s'}T(s,a,s')\Big[R(s,a,s')+\gamma\max_{a'}Q_{\mathrm{opt}}(s',a')\Big],
$$

且 $V_{\mathrm{opt}}(s)=\max_a Q_{\mathrm{opt}}(s,a)$。终点（或吸收且 $R=0$）上 $V=0$、$Q=0$。

> [!INFO]+ 贝尔曼方程从哪来
>
> 令 $G_t=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k}$。则 $G_t=R_t+\gamma G_{t+1}$。两边在 $s_t=s$、按 $\pi$ 行动下取期望：
>
> $$
> V_{\pi}(s)=\mathbb{E}[R_t+\gamma V_{\pi}(s_{t+1})\mid s_t=s].
> $$
>
> 把期望按 $a\sim\pi(\cdot\mid s)$、$s'\sim T(s,a,\cdot)$ 展开，即期望方程。最优方程把对 $\pi$ 的平均换成 $\max_a$：在 $s$ 选使右端最大的动作，此后假设继续最优。这不是新的物理假设，是最优子结构。
>
> $\gamma<1$ 时贝尔曼最优算子 $(T_{\mathrm{opt}}V)(s)=\max_a\sum_{s'}T[R+\gamma V(s')]$ 是压缩映射：
>
> $$
> \|T_{\mathrm{opt}}V-T_{\mathrm{opt}}W\|_{\infty}\le\gamma\|V-W\|_{\infty}.
> $$
>
> 证明：$\lvert\max_a Q^{V}(s,a)-\max_a Q^{W}(s,a)\rvert\le\max_a\lvert Q^{V}-Q^{W}\rvert$，而 $\lvert Q^{V}(s,a)-Q^{W}(s,a)\rvert=\lvert\sum_{s'}T\cdot\gamma(V-W)\rvert\le\gamma\|V-W\|_{\infty}$。Banach 不动点给出唯一 $V_{\mathrm{opt}}$，值迭代从任意初值收敛。
>
> 感性理解：今天的值 $=$ 今晚的期望奖励 $+$ $\gamma$ 乘“明天再按同一规则走下去的值”。$\gamma$ 接近 $1$ 则远期奖励几乎同等计入，压缩变弱，迭代更慢。没有折扣且可以无限游荡时，值可以是 $+\infty$ 或 $-\infty$。

<br>

确定后继、$\gamma=1$、回报取代价相反数时，$V_{\mathrm{opt}}(s)=-\mathrm{FutureCost}(s)$。符号相反是因为搜索最小化代价，MDP 最大化回报。有随机性时预先写死的路径在偏离后无定义，必须改存一张策略表。

## 策略评估

固定 $\pi$ 之后，期望方程是 $|S|$ 元线性组。令 $P_{\pi}(s,s')=\sum_a\pi(a\mid s)T(s,a,s')$，$r_{\pi}(s)=\sum_a\pi(a\mid s)\sum_{s'}T(s,a,s')R(s,a,s')$，则

$$
V_{\pi}=r_{\pi}+\gamma P_{\pi}V_{\pi},
\qquad
V_{\pi}=(I-\gamma P_{\pi})^{-1}r_{\pi}.
$$

$\gamma<1$ 时 $I-\gamma P_{\pi}$ 可逆。规模大时改迭代：$V\leftarrow r_{\pi}+\gamma P_{\pi}V$，直到 $\max_s\lvert\Delta V(s)\rvert$ 小于容差。

```python
def policy_evaluation(states, pi, trans, gamma=0.95, tol=1e-8):
    V = {s: 0.0 for s in states}
    for _ in range(10000):
        delta = 0.0
        Vn = {}
        for s in states:
            a = pi[s]
            Vn[s] = sum(p * (r + gamma * V[s2]) for p, s2, r in trans[s][a])
            delta = max(delta, abs(Vn[s] - V[s]))
        V = Vn
        if delta < tol:
            break
    return V
```

## 值迭代与策略迭代

值迭代（value iteration）从任意 $V_0$（常用 $0$）反复套最优算子

$$
V_{k+1}(s)=\max_{a}\sum_{s'}T(s,a,s')\big[R(s,a,s')+\gamma V_k(s')\big].
$$

$\|V_{k+1}-V_{\mathrm{opt}}\|_{\infty}\le\gamma\|V_k-V_{\mathrm{opt}}\|_{\infty}$，故误差按 $\gamma^{k}$ 收缩。收敛后对 $Q$ 贪心得到 $\pi$。同步更新用一张旧表算整张新表；原地更新也能收敛，作业对拍时以同步为准。

```python
def q_value(s, a, V, trans, gamma):
    return sum(p * (r + gamma * V[s2]) for p, s2, r in trans[s][a])

def value_iteration(states, actions, trans, gamma=0.95, tol=1e-8):
    """trans[s][a] = list of (prob, s2, reward)."""
    V = {s: 0.0 for s in states}
    for _ in range(10000):
        Vn = {}
        delta = 0.0
        for s in states:
            acts = actions(s)
            Vn[s] = max(q_value(s, a, V, trans, gamma) for a in acts) if acts else 0.0
            delta = max(delta, abs(Vn[s] - V[s]))
        V = Vn
        if delta < tol:
            break
    return V

def greedy_policy(states, actions, V, trans, gamma):
    pi = {}
    for s in states:
        acts = actions(s)
        pi[s] = max(acts, key=lambda a: q_value(s, a, V, trans, gamma)) if acts else None
    return pi
```

策略迭代（policy iteration）：任取 $\pi_0$，交替 (1) 策略评估得到 $V_{\pi}$；(2) 贪心改进 $\pi'(s)=\arg\max_a Q_{V}(s,a)$。有限 MDP 上 $\pi$ 单调改进（$V_{\pi'}\ge V_{\pi}$ 点态），有限步到最优。单轮比值迭代贵（要解线性组或内层迭代），轮数通常更少。截断策略迭代只做少数次评估 sweep，介于二者之间。

> [!NOTE]+ 值迭代
>
> 反复套贝尔曼最优算子。来源是动态规划，不是梯度下降。相对 UCS：展开的是整张状态表，而不是一条路径。相对 Q 学习：必须知道 $T$ 与 $R$。$\gamma$ 接近 $1$ 时压缩变慢，需要更多轮。相对策略迭代：值迭代不显式存 $\pi$，每轮对所有动作算期望。

<br>

> [!EXAMPLE]+ 例5 两动作单状态
>
> 状态 $s$ 与吸收终点 $t$，$V(t)=0$，$\gamma=0.9$。动作 $A$：概率 $1$ 到 $t$，回报 $2$，故 $Q(s,A)=2$。动作 $B$：概率 $0.5$ 留在 $s$、回报 $0$，概率 $0.5$ 到 $t$、回报 $4$，
>
> $$
> Q(s,B)=0.5\big(0+\gamma V(s)\big)+0.5\cdot 4=2+0.45\,V(s).
> $$
>
> 最优方程 $V=\max(2,\ 2+0.45V)$。若取右边，$0.55V=2$，$V=40/11\approx 3.636$，且大于 $2$，故最优是 $B$。值迭代从 $V_0=0$：$V_1=2$，$V_2=2.9$，$V_3=3.305$，按 $\gamma$ 靠近 $40/11$。若误把 $B$ 的期望写成“先加 $4$ 再乘 $0.5$”而漏掉留在 $s$ 的分支，会得到错误的 $Q$。

<br>

## 无模型：TD 与 Q 学习

$T,R$ 未知时不能套期望，只能用轨迹 $(s,a,r,s')$。蒙特卡洛（Monte Carlo）等整条回合结束，用实际折扣回报当 $V(s)$ 的目标，无偏、方差大。TD(0)（temporal difference）用自举目标 $r+\gamma V(s')$：

$$
V(s)\leftarrow V(s)+\alpha\big(r+\gamma V(s')-V(s)\big).
$$

偏差来自当前 $V$ 不准确；方差更小，不需要完整回合。表格 TD 在 visition 充分且 $\alpha$ 满足 Robbins-Monro 时收敛到 $V_{\pi}$。函数逼近下 TD 可以发散，课程作业用表格。

Q 学习直接估 $Q_{\mathrm{opt}}$：

$$
Q(s,a)\leftarrow(1-\alpha)Q(s,a)+\alpha\Big(r+\gamma\max_{a'}Q(s',a')\Big).
$$

off-policy：产生数据的行为策略（behavior policy）可以探索，更新目标仍是最优贝尔曼。SARSA 把 $\max_{a'}$ 换成回合里实际采取的 $a'$，估的是行为策略自己的 $Q_{\pi}$，on-policy。

探索：$\varepsilon$-贪心（$\varepsilon$-greedy）以 $\varepsilon$ 均匀乱走，以 $1-\varepsilon$ 取 $\arg\max Q$；或对未访 $(s,a)$ 给乐观初值。$\varepsilon=0$ 且初值不乐观时，从未试过的动作永远看不到正确目标。Mountain Car 作业里，若 $\varepsilon=0$，小车到不了坡顶，更新永远看不到正回报。

```python
def epsilon_greedy(Q, s, actions, eps, rng):
    acts = actions(s)
    if rng.random() < eps:
        return acts[rng.integers(len(acts))]
    return max(acts, key=lambda a: Q[(s, a)])

def q_learning_episode(Q, start, is_end, actions, step, eps, alpha, gamma, rng):
    """step(s, a) -> (s2, r)。Q 是 dict[(s, a)]。"""
    s = start
    while not is_end(s):
        a = epsilon_greedy(Q, s, actions, eps, rng)
        s2, r = step(s, a)
        target = r if is_end(s2) else r + gamma * max(Q[(s2, ap)] for ap in actions(s2))
        Q[(s, a)] += alpha * (target - Q[(s, a)])
        s = s2
```

> [!INFO]+ Q 学习是随机近似
>
> 贝尔曼最优方程里对 $s'$ 的期望被一次样本 $(s,a,r,s')$ 代替，再向该目标走步长 $\alpha$。$\alpha_t$ 满足 $\sum_t\alpha_t=\infty$、$\sum_t\alpha_t^{2}<\infty$，且每个 $(s,a)$ 被访问无穷次时，表格 Q 学习几乎必然收敛到 $Q_{\mathrm{opt}}$。函数逼近下该保证一般消失。
>
> SARSA 的目标是 $r+\gamma Q(s',a')$，$a'$ 来自行为策略，收敛到 $Q_{\pi}$ 而不是 $Q_{\mathrm{opt}}$。若 $\varepsilon$ 缓慢降到 $0$，二者极限可以重合。作业若按最优回报评分，用 Q 学习；若按“实际执行的 $\varepsilon$-贪心”评分，二者数字不同。
>
> 感性理解：每一次转移都是对“若此处最优，下一步值该是多少”的一次带噪测量。$\max_{a'}$ 用的是当前估计，早期很不准，所以必须持续探索，否则从未走过的动作永远停在初值。

<br>

> [!EXAMPLE]+ 例6 火山格子
>
> 终点风景回报 $+20$，火山 $-50$，每步 $-0.1$，$\gamma=0.99$，动作以 $0.8$ 按意图走、$0.1$ 滑向两侧。值迭代的箭头会绕开火山：贴着悬崖的近路在滑行下期望为负。把滑行概率改成 $0$，策略贴近悬崖。模型错了则最优方程解的是另一盘棋。
>
> 同一格子上跑 Q 学习：$\varepsilon=0$ 时智能体若一开始不敢靠近风景，就学不到 $+20$。$\alpha$ 过大则 $Q$ 振荡，$+20$ 与 $-50$ 交替污染邻格。对拍应先在 $T$ 已知时跑值迭代，把 $Q$ 表当参考。

<br>

## 实现注意

- 转移概率对每个 $(s,a)$ 必须加总为 $1$。漏掉滑行分支等于改写物理。
- $R$ 写在 $(s,a,s')$ 上还是写在进入终点时，必须与作业约定一致，否则 $V$ 差一个常数或差一步。
- $\gamma=1$ 且存在可以无限转的环时，值迭代可以发散。先确认吸收，或改 $\gamma<1$。
- 表格 $Q$ 的键是 `(s, a)`。状态不可哈希则整张表废掉。
- 探索衰减：先大 $\varepsilon$ 再降。用训练回报选 $\varepsilon$ 会过拟合该随机种子。

> [!NOTE]+ 对照：搜索与 MDP
>
> 搜索要一条路径；MDP 要一张策略表。确定后继且无折扣时，值迭代算出的 $V$ 与 $\mathrm{FutureCost}$ 差一个符号。有随机性时“预先写死的路径”会在偏离后无定义。UCS 的弹出不变量在期望代价上不成立：必须对整张 $S$ 迭代，或对样本做随机近似。

<br>

> [!NOTE]+ 衔接
>
> MDP 的随机来自自然，对后继取期望。下一章随机来自对手：不能再对对手的动作取期望（除非对手策略已知），最坏情形改成 $\min$。算法从值迭代变成 minimax / expectimax。

<br>


# 博弈

## 概念：博弈

- **问题**
  - 后继由对手或机会决定；
  - 问的是博弈值与当前玩家的动作，不是单智能体路径。
- **范围**
  - expectimax、minimax、expectiminimax、α-β 与估值函数；
  - 输入博弈树，输出根值与动作。
- **章节衔接**
  - 上一章只有一个决策者；
  - 本章出现 min 玩家或随机节点；
  - 下一章不再沿时间展开状态，而把问题拆成变量。

零和回合制博弈把状态树分成三类节点：max（当前智能体）、min（对手）、chance（骰子或已知随机策略）。叶子给回报，从当前玩家视角越大越好。折扣 $\gamma$ 可有可无；Pac-Man 作业常按步数截断深度，叶子改估值。状态仍须 Markov：位置、豆、是否惊吓、谁走，漏字段等于换了一盘棋。

> [!NOTE]+ 博弈
>
> 在对手或机会存在时计算根节点的值。来源是 von Neumann 的对抗搜索，课程把它写成 MDP 的对手版。相对 MDP：不能对后继取期望（除非对手策略已知），最坏情形取 $\min$。相对搜索：一条预先写死的路径会在对手偏离时崩溃。相对反射：不能看一眼棋盘就输出动作而不展开对手回应。

<br>

## expectimax 与 minimax

固定对手（或自然）策略 $\pi_{\mathrm{opp}}$ 时，随机节点的分布已知，用 expectimax：

$$
V_{\mathrm{ex}}(s)=
\begin{cases}
\mathrm{eval}(s) & s\text{ 为叶}\\
\max_{a}V_{\mathrm{ex}}(\mathrm{Succ}(s,a)) & s\text{ 为 max}\\
\sum_{s'}T(s,s')V_{\mathrm{ex}}(s') & s\text{ 为 chance.}
\end{cases}
$$

这与 MDP 值迭代同形，只是动作层与机会层分开写。最坏对手用 minimax：chance 层换成 min 层，

$$
V_{\mathrm{mm}}(s)=\max_{a}\min_{a_{\mathrm{opp}}}V_{\mathrm{mm}}(\mathrm{Succ}(s,a,a_{\mathrm{opp}})).
$$

双方都最优时根值是博弈值（game value）。二人零和有限完美信息博弈存在该值（可随机化，纯策略在无同时行动的树上已够）。

混合：幽灵按已知分布走、另有随机果实，用 expectiminimax，三类节点按类型取值。选错节点类型等于解另一盘游戏。

```python
def expectimax(state, depth, node, actions, succ, eval_fn, chances):
    """node in {"max", "min", "chance"}。chances(state) -> (s2, p) 列表。"""
    if depth == 0 or state.terminal:
        return eval_fn(state)
    if node == "max":
        return max(expectimax(succ(state, a), depth - 1, "chance", actions, succ, eval_fn, chances)
                   for a in actions(state))
    if node == "min":
        return min(expectimax(succ(state, a), depth - 1, "max", actions, succ, eval_fn, chances)
                   for a in actions(state))
    return sum(p * expectimax(s2, depth - 1, "max", actions, succ, eval_fn, chances)
               for s2, p in chances(state))
```

实现里“下一层节点类型”必须按作业的回合顺序写死：Pac-Man 走完是否所有幽灵各走一步、机会节点插在哪，都会改树。

> [!INFO]+ minimax 在算什么
>
> 设叶子回报确定。从叶子向上：min 层取子节点最小，max 层取最大。根值等于：max 先走、此后双方都最优时的回报。这不是对历史对局的统计，而是对整棵树的定义。证明用结构归纳：叶显然；min 节点若子树值正确，取最小即该玩家能逼对方接受的值。
>
> 若对手并非最优，按 minimax 走仍有下界（不会比博弈值更差，前提是估值等于真回报），但可能放过更高回报。expectimax 在对手近随机时平均更高，在对手刻意针对时可以任意差。Pac-Man 里把跟踪幽灵写成均匀随机，最优动作表会整张偏掉：智能体会从“本会被堵住的巷口”穿过。
>
> 感性理解：max 问当前玩家能逼出的最好结果，min 问对手能逼出的最坏结果，chance 问骰子的平均。三者不能互换。只看一层贪心（只比下一步叶子）会忽略对手两步之后的反击。

<br>

> [!EXAMPLE]+ 例7 一层 expectimax
>
> 根 max，两个动作。左动作进 chance：叶子 $10$ 与 $0$ 各 $1/2$，期望 $5$。右动作进 chance：叶子 $4$ 与 $6$ 各 $1/2$，期望 $5$。根值 $5$，两动作并列。若右动作其实是 min 节点（对手选 $4$ 或 $6$），则右值变成 $4$，根选左，值仍 $5$。同一棵形状、节点类型不同，动作表不同。自动评分会核对根值与所选动作，不只核对“看起来合理”。

<br>

## α-β

α-β 剪枝在 minimax 递推里携带窗口 $[\alpha,\beta]$：

- $\alpha$：max 在通往根的路径上已保证的下界（已经能逼到至少这么好）；
- $\beta$：min 已保证的上界（已经能逼到至多这么好）。

某分支的值不可能落入 $(\alpha,\beta)$ 从而影响根值时剪掉。正确性：被剪掉的子树即使展开，根的 minimax 值不变。剪枝不改变值，只改变展开量。expectimax 的 chance 层一般不能 α-β：期望会被未展开的叶子拉动，除非有额外界。

最好情形（每层子节点按从好到坏排）大约 $O(b^{d/2})$ 个叶子，有效分支因子约 $\sqrt{b}$；最坏仍 $O(b^{d})$。实现必须在递推里传递并更新 $\alpha,\beta$，不能只在根做一次比较。窗口初始化必须是真正的 $\pm\infty$。

```python
def alphabeta(state, depth, alpha, beta, is_max, actions, succ, eval_fn):
    if depth == 0 or state.terminal:
        return eval_fn(state)
    if is_max:
        v = -np.inf
        for a in actions(state):
            v = max(v, alphabeta(succ(state, a), depth - 1, alpha, beta, False, actions, succ, eval_fn))
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return v
    v = np.inf
    for a in actions(state):
        v = min(v, alphabeta(succ(state, a), depth - 1, alpha, beta, True, actions, succ, eval_fn))
        beta = min(beta, v)
        if alpha >= beta:
            break
    return v
```

`alpha >= beta` 用闭区间（fail-hard）。并列时是否剪，会影响展开集合，必须与参考实现一致。若还要返回最优动作，只能在根层记录使 $v$ 更新的 $a$，不要在剪枝后的残缺 $v$ 上乱选。

> [!NOTE]+ α-β 剪枝
>
> 在 minimax 递推中携带当前窗口，剪掉不能更新根值的分支。来源是对博弈树的分支定界，不是近似。相对完整 minimax：值相同，展开更少。相对随便截断深度：截断改变的是叶子定义，必须另配估值函数（evaluation function）。相对 A*：二者都用界来少展开，A* 的界来自 $h$，α-β 的界来自兄弟节点已经算出的值。

<br>

> [!EXAMPLE]+ 例8 三层 α-β
>
> 根 max，两子为 min。左 min 的叶子从左到右 $3,12,8$；右 min 的叶子 $2,4,6$。约定从左向右展开。
>
> 1. 左 min 见 $3$，当前 $v=3$，$\beta=3$。再看 $12$、$8$，$\min$ 仍为 $3$。左子树值 $3$。根更新 $\alpha=3$。
> 2. 右 min 先看 $2$，当前 $v=2$，把 $\beta$ 降为 $2$。此时 $\alpha=3\ge\beta=2$，其余叶子 $4,6$ 不必看：min 已经能把该子树压到 $\le 2$，max 不会选它。
> 3. 根值 $\max(3,2)=3$。
>
> 把最左叶子 $3$ 改成 $10$。左 min 得 $\min(10,12,8)=8$，根 $\alpha=8$。右 min 仍先见 $2$，$\beta=2$，$8\ge 2$，同样剪掉 $4,6$。根值 $\max(8,2)=8$。完整展开右子树也是 $\min(2,4,6)=2$，值不变，只是展开集合在“左值为 $3$ 还是 $8$”时都剪右，但左子树的比较次数不同。自动评分会核对展开过的叶子集合，不单核对根值。
>
> 若右子树先展开（先看到 $2$），根 $\alpha$ 仍为 $-\infty$ 时还不能剪，剪枝位置随次序变。作业若指定动作顺序，不要按“看起来好”重排，除非题目允许启发式排序。

<br>

## 估值函数

深度有限时叶子不是终点，用估值函数 $\hat V(s)$ 代替真实回报。α-β 仍然精确计算这棵截断树上的 minimax 值；不精确的是树本身。估值必须与终点回报同号、同尺度，否则 max / min 的比较无意义：若终点吃豆是 $+10$、估值把距离写成 $+1000$，智能体会宁愿停在“离豆近”而不是真的吃豆。

Pac-Man 常用特征：距离最近食物、距离幽灵、惊吓剩余步数、剩余豆数、是否在死巷。线性估值 $\hat V=w\cdot\phi(s)$ 与反射模型同形，但这里的 $w$ 多半手调，不是 SGD。调估值是建模，不是调 α-β。

深度奇数或偶数会改变“谁最后走”，估值若偏爱某一侧，会出现水平效应（horizon effect）：智能体把不可避免的损失推到深度之外，估值看不见。对策是增加深度、静止搜索（quiescence），或把必吃着法走完再估值。课程作业通常只要求固定深度加一组特征。

> [!NOTE]+ 估值函数
>
> 把截断点当成叶子，用可计算的分数代替整棵子树。来源是象棋程序的评价，不是新的博弈定义。相对真实回报：改变的是叶子，不改变 minimax / α-β 的递推。相对反射：$\phi(s)$ 可以很像情感分类的特征，但推断仍是一层层 $\max/\min$，不是一次 $\mathrm{sign}(w\cdot\phi)$。

<br>

## 实现注意

- 状态拷贝：走一步必须返回新状态。原地改棋盘再递归，回溯时盘面已坏。
- 节点类型与“深度减一”的粒度：有的实现一步算 Pac-Man 加全部幽灵，深度按回合减；有的按半步减。与作业约定不一致则展开数全错。
- α-β 窗口必须沿递归传递。在子调用外更新却传入旧窗口，等于没剪。
- 并列动作：根层应有确定打破规则，否则动作集合与参考实现不一致。
- expectimax 不要对 chance 层做 α-β，除非另给概率界。

> [!NOTE]+ 对照：expectimax 与 minimax
>
> 对手策略已知且随机，用 expectimax。对手对抗，用 minimax。混合用 expectiminimax。选错节点类型等于解另一盘游戏。α-β 只对纯 minimax 合法。估值错了则整棵截断树在解另一盘游戏。

<br>

> [!NOTE]+ 衔接
>
> 状态轴沿时间展开，状态数随步数指数增长：棋盘、豆、谁走，复制一份就翻一倍。许多作业其实变量个数固定（时段、教室、位置坐标），不必把整条轨迹压进一个状态。下一章把问题拆成变量上的因子，推断改为求赋值或边缘。

<br>


# 因子图

## 概念：因子图

- **问题**
  - 一组变量 $X_1,\ldots,X_n$，每个有有限定义域；
  - 权 $w(x)=\prod_j f_j(x_{C_j})$，求最大权赋值或可行赋值。
- **范围**
  - CSP、回溯、弧一致、束搜索、变量消去；
  - 输入因子，输出赋值或分割函数。
- **章节衔接**
  - 上一章的状态随时间增长；
  - 本章变量个数固定，耦合写在因子里；
  - 下一章因子改成条件概率。

因子图（factor graph）是二部图：一侧是变量节点，一侧是因子节点，边表示因子依赖的变量。联合权

$$
w(x)=\prod_{j}f_j(x_{C_j}),\qquad x=(x_1,\ldots,x_n).
$$

推断问两件事之一：最大权赋值（MAP）$\arg\max_x w(x)$；或分割函数 $Z=\sum_x w(x)$ 以及边缘。约束满足（constraint satisfaction problem, CSP）是因子取值 $\{0,1\}$ 的特例：权为 $0$ 表示违反约束，可行赋值即 $w(x)=1$。排课作业：课程时段为变量，教室冲突、先修、教师不重叠为因子。

马尔可夫网（Markov network）把权解释为 $P(x)=w(x)/Z$。推断算法与因子图相同，只是输出要归一化成概率。无向图上的条件独立按割集读：给定隔开 $A$ 与 $B$ 的变量后二者独立。对撞结构不会出现，解释掉是有向网特有的。

> [!NOTE]+ 因子图
>
> 把联合权分解成局部因子的乘积。来源是约束满足与无向图模型，课程用它接住“变量轴”。相对状态模型：不需要把整个时间线压进一个状态；排 $n$ 门课是 $n$ 个变量，不是 $n$ 步路径。相对贝叶斯网：这里先不谈概率，只谈权与约束。相对反射：一次赋值要同时满足许多局部因子，不能对每个变量独立 $\mathrm{sign}$。

<br>

## 回溯与前向检查

回溯选一个未赋值变量，按定义域试值，与已有因子矛盾则撤回。变量序与值序影响展开量，不改变正确性：

- 最少剩余值（minimum remaining values, MRV）：先选定义域最短的变量，早失败；
- 度启发式：MRV 并列时先选约束最多的；
- 最少约束值（least constraining value, LCV）：先试删掉邻居最少选项的值，留给后面更大空间。

前向检查（forward checking）：刚给 $X_i=v$ 后，只检查与 $X_i$ 有约束的未赋值邻居，删掉与 $v$ 不一致的值。比纯回溯早发现失败，比弧一致弱：不在邻居之间继续传播。

```python
def backtrack(assign, domains, variables, consistent, neighbors):
    if len(assign) == len(variables):
        return dict(assign)
    x = min((v for v in variables if v not in assign),
            key=lambda v: (len(domains[v]), -len(neighbors[v])))
    for val in list(domains[x]):
        if all(consistent(x, val, y, assign[y]) for y in assign if y in neighbors[x]):
            assign[x] = val
            saved = {y: list(domains[y]) for y in neighbors[x] if y not in assign}
            ok = True
            for y in saved:
                domains[y] = [u for u in domains[y] if consistent(x, val, y, u)]
                if not domains[y]:
                    ok = False
                    break
            if ok:
                res = backtrack(assign, domains, variables, consistent, neighbors)
                if res is not None:
                    return res
            for y, dom in saved.items():
                domains[y] = dom
            del assign[x]
    return None
```

## 弧一致

弧一致（arc consistency）：对有序对 $(X_i,X_j)$，删掉 $X_i$ 中没有任何 $X_j$ 取值能满足约束的值。AC-3 把所有弧入队，某侧定义域缩小后再把指向它的弧入队，直到不动点。时间 $O(ed^{3})$（$e$ 条约束，$d$ 为定义域大小），多项式，不给出全局解。

回溯前做 AC-3，失败则整支剪掉。MAC（maintaining arc consistency）在每次赋值后对剩余图再跑 AC-3，比前向检查剪得更狠，常数更大。

```python
from collections import deque

def revise(domains, xi, xj, consistent):
    changed = False
    keep = []
    for vi in domains[xi]:
        if any(consistent(xi, vi, xj, vj) for vj in domains[xj]):
            keep.append(vi)
        else:
            changed = True
    domains[xi] = keep
    return changed

def ac3(variables, neighbors, domains, consistent):
    q = deque((i, j) for i in variables for j in neighbors[i])
    while q:
        xi, xj = q.popleft()
        if revise(domains, xi, xj, consistent):
            if not domains[xi]:
                return False
            for xk in neighbors[xi]:
                if xk != xj:
                    q.append((xk, xi))
    return True
```

> [!NOTE]+ 弧一致
>
> 删掉局部看起来已经不可能的值。来源是约束传播，不是搜索。相对纯回溯：许多失败在赋值前就被发现。相对前向检查：AC-3 会沿约束链传播，不限于刚赋值变量的邻居。相对变量消去：AC-3 多项式时间，但不给出全局解。定义域被删空则无解；定义域非空仍可能无解（例如奇数环图着色，局部都还有颜色）。

<br>

> [!EXAMPLE]+ 例9 双变量不等
>
> $X,Y\in\{1,2\}$，约束 $X\ne Y$。AC-3 不删任何值：每个值在对面都能找到搭档。前向检查在 $X=1$ 之后把 $Y$ 的 $1$ 删掉，剩下 $2$。再加 $Z\in\{1\}$ 与 $X\ne Z$、$Y\ne Z$：$Z$ 只能取 $1$，AC-3 会删掉 $X=1$ 与 $Y=1$，剩下 $X=Y=2$，再被 $X\ne Y$ 删空，报告无解。纯回溯要走到第三层才失败。

<br>

## 束搜索

束搜索（beam search）按变量序一层层扩展部分赋值，每层只保留权最高的 $K$ 个前缀。$K=1$ 是贪心；$K=\infty$ 是完整宽度优先。不保证最优，也不保证找到可行解。Spring 2026 单独讲它，是因为排课规模上完整回溯不可行。

```python
def beam_search(variables, domains, score, K):
    """score(assign) 越大越好。返回至多 K 个完整赋值。"""
    beam = [({}, 0.0)]
    for x in variables:
        nxt = []
        for assign, sc in beam:
            for val in domains[x]:
                a2 = dict(assign)
                a2[x] = val
                nxt.append((a2, score(a2)))
        nxt.sort(key=lambda t: t[1], reverse=True)
        beam = nxt[:K]
    return beam
```

部分赋值的 `score` 只能用已经触及的因子。用未赋值变量上的“乐观上界”当分数，属于分支定界，不是普通束搜索。作业必须报 $K$ 与是否找到可行解，不能只报分数。

> [!NOTE]+ 束搜索
>
> 在赋值的层上做截断的宽度优先搜索。来源是语音与翻译解码，不是 CSP 的完备算法。相对回溯：时间可控，可能漏掉唯一可行解。相对动态规划：不存整张表，只存 $K$ 条前缀。相对 A*：束宽 $K$ 是硬截断，没有可采纳保证。

<br>

## 变量消去

变量消去（variable elimination, VE）求

$$
Z=\sum_{x_1}\cdots\sum_{x_n}\prod_j f_j(x_{C_j})
$$

或 MAP $\max_x\prod_j f_j$。每次挑一个变量，把含它的因子乘起来，再对该变量求和（或取最大），得到一个新因子。消去顺序决定中间因子大小。树宽（treewidth）是所有消去顺序下最大中间宽度的最小值；树宽小则精确推断可行。

> [!INFO]+ 变量消去的一次步
>
> 设要消去 $X_i$。令 $F_i$ 为所有依赖 $X_i$ 的因子，其余因子不动。新因子
>
> $$
> f_{\mathrm{new}}(x_{S})=\sum_{x_i}\prod_{f\in F_i}f(\cdot),
> $$
>
> 其中 $S$ 是 $F_i$ 里除 $X_i$ 外的变量。联合乘积的值不变，只是 $X_i$ 被积掉。MAP 把 $\sum$ 换成 $\max$，并记录 $\arg\max$ 以便回填。证据 $X_e=e$ 在开始时把因子里与证据冲突的行置零（或删掉）。
>
> 感性理解：先把“还连着 $X_i$ 的几张表”合成一张，再把 $X_i$ 这列收掉。合成时出现的列数就是代价。顺序差一张表可以大到无法存。链状图按端点往里消，中间永远是两个变量；从中间先消，立刻出现两端的耦合。

<br>

复杂度：$O(n|\mathcal{D}|^{w+1})$，其中 $w$ 是该消去顺序下的最大中间宽度。找最优顺序 NP-难；常用最小度（先消邻居最少）或最小缺边（先消补全后少加边）启发式。乘积在 log 域改成加法，避免下溢；MAP 同样。

```python
def eliminate(factors, var, domains, maximize=False):
    """factors: list of (var_tuple, table dict)。消去 var，返回新因子列表。"""
    involved = [f for f in factors if var in f[0]]
    rest = [f for f in factors if var not in f[0]]
    others = []
    for vs, _ in involved:
        for v in vs:
            if v != var and v not in others:
                others.append(v)
    table = {}
    for assignment in _product(others, domains):
        acc = -np.inf if maximize else 0.0
        for val in domains[var]:
            full = dict(assignment)
            full[var] = val
            prod = 1.0
            for vs, tab in involved:
                prod *= tab[tuple(full[v] for v in vs)]
            if maximize:
                acc = max(acc, prod)
            else:
                acc += prod
        table[tuple(assignment[v] for v in others)] = acc
    rest.append((tuple(others), table))
    return rest
```

`_product` 枚举 `others` 上的赋值。实现细节不重要，重要的是：中间表的键是 `others`，大小 $|\mathcal{D}|^{|\mathrm{others}|}$。

> [!EXAMPLE]+ 例10 三变量链的数值消去
>
> 二值变量 $X,Y,Z$。因子
>
> $$
> f_{XY}(0,0)=2,\ f_{XY}(0,1)=1,\ f_{XY}(1,0)=1,\ f_{XY}(1,1)=3;
> $$
>
> $$
> f_{YZ}(0,0)=1,\ f_{YZ}(0,1)=4,\ f_{YZ}(1,0)=2,\ f_{YZ}(1,1)=1.
> $$
>
> 先消 $Y$。对每组 $(x,z)$ 把 $y$ 加起来：
>
> $$
> f_{XZ}(x,z)=\sum_{y}f_{XY}(x,y)f_{YZ}(y,z).
> $$
>
> $f_{XZ}(0,0)=2\cdot 1+1\cdot 2=4$，$f_{XZ}(0,1)=2\cdot 4+1\cdot 1=9$，$f_{XZ}(1,0)=1\cdot 1+3\cdot 2=7$，$f_{XZ}(1,1)=1\cdot 4+3\cdot 1=7$。再消 $X$：$g(z)=\sum_x f_{XZ}(x,z)$，$g(0)=4+7=11$，$g(1)=9+7=16$。分割函数 $\sum_z g(z)=27$。
>
> MAP 把每一步的求和换成最大。四条赋值的权是 $2\cdot 1=2$、$2\cdot 4=8$、$1\cdot 2=2$、$3\cdot 1=3$（按 $(x,y,z)=(0,0,0),(0,0,1),(0,1,0),(1,1,1)$ 等枚举），最大权为 $8$，赋值 $(X,Y,Z)=(0,0,1)$。边缘 $g(1)=16>g(0)$ 与该 MAP 的 $Z$ 同侧，只是巧合；换一组因子可以使边缘众数与 MAP 的分量不一致。把边缘 $\arg\max$ 当成 MAP 会错。

<br>

> [!NOTE]+ 对照：回溯、束搜索与 VE
>
> 回溯与 AC-3 找可行赋值，完备（若跑完）。束搜索有时间帽，不完备。VE 算 $Z$ 或 MAP，代价由树宽决定，与“好不好找一组可行解”不是同一复杂度。树宽高时精确 VE 不可行，改束搜索或采样，必须在报告里写这是近似。

<br>

> [!NOTE]+ 衔接
>
> 因子图的权还没有概率语义。下一章把因子写成条件概率表，推断问边缘 $P(Q\mid E=e)$，并引入条件独立来少算。算法骨架（消去、采样）仍在，数字的含义变成概率。

<br>


# 贝叶斯网

## 概念：贝叶斯网

- **问题**
  - 联合分布 $P(X_1,\ldots,X_n)$ 按有向无环图分解；
  - 问条件边缘、MAP 或最可能解释。
- **范围**
  - 写出链式法则、d-分离、精确推断与粒子滤波；
  - 输入网与证据，输出概率或粒子集。
- **章节衔接**
  - 上一章因子是任意非负权；
  - 本章因子是 $P(X_i\mid\mathrm{Pa}(X_i))$，积分为 $1$；
  - 下一章离开数值权，改用真假公式。

任意联合分布都有链式法则

$$
P(x_1,\ldots,x_n)=P(x_1)P(x_2\mid x_1)\cdots P(x_n\mid x_1,\ldots,x_{n-1}).
$$

贝叶斯网（Bayesian network）用有向无环图（DAG）把每项的条件集缩成父节点：

$$
P(x_1,\ldots,x_n)=\prod_{i=1}^{n}P(x_i\mid x_{\mathrm{Pa}(i)}).
$$

每张条件概率表（conditional probability table, CPT）是一个因子，且对父配置求和为 $1$。没有边不表示独立，独立由 d-分离读出。边是分解假设，不是自动的因果证明。

> [!NOTE]+ 贝叶斯网
>
> 用有向无环图把联合分布分解成局部 CPT。来源是 Pearl 的信念网，课程把它当作带概率的因子图。相对完整联合表：$n$ 个二值变量从 $2^{n}$ 降到各 CPT 大小之和。相对马尔可夫网：CPT 已经归一化，但询问 $P(Q\mid e)$ 仍要在证据下再归一化一次。相对因果口号：干预与 $do$ 演算不在本课考试范围，作业只问观测推断。

<br>

> [!EXAMPLE]+ 例11 报警网
>
> 变量：盗窃 $B$、地震 $E$、报警 $A$。边 $B\rightarrow A\leftarrow E$。CPT：$P(B=1)=0.01$，$P(E=1)=0.01$，$P(A=1\mid B,E)$ 在 $(1,1),(1,0),(0,1),(0,0)$ 上分别为 $0.95,0.94,0.29,0.001$。联合
>
> $$
> P(b,e,a)=P(b)P(e)P(a\mid b,e).
> $$
>
> $P(A=1)=\sum_{b,e}P(b)P(e)P(A=1\mid b,e)$。数值上主项来自“无盗窃无地震仍误报”与“只有盗窃”，不是来自同时发生的两因。CPT 未归一（某行加总不为 $1$）则整张网不是分布。

<br>

## 条件独立

d-分离（d-separation）：一条无向路径被证据开通或关闭，由三种局部结构决定。两组变量的所有路径都被证据关闭，则二者条件独立，联合里对应的因子化成立。

- 串行 $A\rightarrow B\rightarrow C$（链）：观察 $B$ 阻断；未观察则开通。
- 分叉 $A\leftarrow B\rightarrow C$（共同原因）：观察 $B$ 阻断；未观察则开通。
- 对撞 $A\rightarrow B\leftarrow C$（共同结果）：未观察 $B$ 时阻断；观察 $B$ 或其后裔反而开通。

图上的 d-分离是条件独立的充分条件；在忠实（faithful）分布上还是必要条件。作业按图读独立，不要用“看起来相关”代替。

> [!INFO]+ 对撞为何开通
>
> 未观察时，$A$ 与 $C$ 可以独立（两因一果）。一旦知道 $B$，关于 $A$ 的信息会改变对 $C$ 的信念：解释掉（explaining away）。这是贝叶斯网相对无向图多出来的结构。马尔可夫网没有对撞，给定中间变量总是阻断。
>
> 报警网：$B$ 与 $E$ 无边，先验独立。听到报警后，$P(E=1\mid A=1,B=1)<P(E=1\mid A=1)$：已经用盗窃解释了报警，地震的后验下降。
>
> 感性理解：引擎响与油灯亮可以事先独立。都知道车开不动之后，若确认是引擎，油灯的嫌疑下降。证据把两条因连起来了。

<br>

## 精确推断

询问 $P(Q\mid E=e)$：

1. 把每个 CPT 当作因子；
2. 证据代入：与 $E=e$ 冲突的行置零；
3. 对其余隐变量做变量消去，得到 $f(q)$；
4. 归一化 $P(q\mid e)=f(q)/\sum_{q'}f(q')$。

枚举（enumeration）是 VE 的笨实现：对隐变量的全部赋值累加 $\prod_i\mathrm{CPT}_i$，复杂度 $O(n|\mathcal{D}|^{n-|E|})$，不管图结构。VE 利用因子局部性，复杂度回到树宽。MAP / MPE 把求和换成最大化，求最可能的完整赋值；边缘众数（对每个变量单独 $\arg\max P(X_i\mid e)$）可以不是 MAP 赋值。

车辆跟踪若把 $z_{1:T}$ 整条轨迹当成一张静态网做精确推断，变量数随 $T$ 增长，树宽随轨迹变长。必须改用 HMM 的前向递推或粒子滤波。

Gibbs 采样：固定证据，轮流从满条件 $P(X_i\mid X_{-i},e)$ 抽样。平稳分布是真实后验。混合慢时相邻样本不独立，不能把连续状态当独立样本报误差条。燃烧期与间隔抽样是实现细节，作业通常只要求写出满条件正比于“含 $X_i$ 的那些 CPT 之积”。

> [!NOTE]+ 精确推断
>
> 证据下的边缘是因子图 VE 加一次归一化。来源是同一套消去，不是新的概率公理。相对 CSP：输出是数字 $P(q\mid e)\in[0,1]$，不是“有没有赋值”。相对 MAP：边缘与最可能完整赋值回答不同问题。相对 Gibbs：精确法无方差，树宽高时不可用。

<br>

## HMM 与前向

隐马尔可夫模型（hidden Markov model, HMM）是贝叶斯网的时间链：

$$
P(z_{1:T},x_{1:T})=P(z_1)\prod_{t=2}^{T}P(z_t\mid z_{t-1})\prod_{t=1}^{T}P(x_t\mid z_t).
$$

车辆跟踪作业：$z$ 是位置（或离散化位姿），$x$ 是传感器。滤波问 $P(z_t\mid x_{1:t})$，平滑问 $P(z_t\mid x_{1:T})$。

前向信息 $\alpha_t(z)=P(z_t=z,x_{1:t})$ 满足

$$
\alpha_1(z)=P(z_1=z)P(x_1\mid z),
\qquad
\alpha_{t}(z)=P(x_t\mid z)\sum_{z'}\alpha_{t-1}(z')P(z_t=z\mid z_{t-1}=z').
$$

归一化 $\alpha_t$ 即滤波分布。离散 $|Z|$ 小时这是精确的，时间 $O(T|Z|^{2})$，不随树宽爆炸。$|Z|$ 是连续网格或高维位姿时，前向表存不下，改粒子滤波。

> [!INFO]+ 前向递推
>
> $\alpha_t(z)$ 把“走到 $z$ 并且看到 $x_{1:t}$”的概率拆成：先以任意 $z'$ 结束上一步（$\alpha_{t-1}$），再乘转移，再乘本步似然。这是 VE 按时间顺序消 $z_{t-1}$ 的特化，不是新公式。
>
> 感性理解：每个格子存“有多少质量经过这里并且没跟传感器打架”。下一步把质量按动力学推移，再按新观测打折。归一化之后才是后验。漏掉归一化，后续乘似然会下溢到 $0$。

<br>

## 粒子滤波

粒子滤波维护一组加权粒子 $\{z^{(k)},w^{(k)}\}_{k=1}^{K}$：

1. 按转移核从上一时刻粒子前进一步（提议）；
2. 用观测似然 $P(x_t\mid z^{(k)})$ 乘权；
3. 归一化；有效样本量 $\mathrm{ESS}=1/\sum_k (w^{(k)})^{2}$ 过低则重采样。

```python
def particle_filter_step(particles, weights, transition, likelihood, rng):
    nxt = np.array([transition(p, rng) for p in particles])
    weights = weights * np.array([likelihood(p) for p in nxt])
    if weights.sum() <= 0:
        weights = np.ones(len(nxt)) / len(nxt)
        return nxt, weights
    weights = weights / weights.sum()
    ess = 1.0 / np.sum(weights ** 2)
    if ess < 0.5 * len(particles):
        idx = rng.choice(len(nxt), size=len(nxt), p=weights)
        nxt = nxt[idx]
        weights = np.ones(len(nxt)) / len(nxt)
    return nxt, weights
```

> [!INFO]+ 粒子权为何乘似然
>
> 重要性采样用转移核 $P(z_t\mid z_{t-1})$ 当提议。目标是滤波后验 $P(z_t\mid x_{1:t})\propto P(x_t\mid z_t)P(z_t\mid x_{1:t-1})$。前一步的粒子已经近似 $P(z_{t-1}\mid x_{1:t-1})$，推移后近似 $P(z_t\mid x_{1:t-1})$，故权乘 $P(x_t\mid z_t)$。
>
> 重采样按权有放回地复制粒子，权复位成均匀。ESS 低表示少数粒子占了大部分权，不重采样则下一步几乎只在更新那几颗。过频重采样增加方差。所有权变成 $0$（似然模型与粒子完全不相容）时必须重启，不能除以 $0$。
>
> 感性理解：粒子先按动力学撒到下一时刻，再看谁更像当前传感器。像的权变大，不像的权变小。权都集中到少数粒子时，重采样把计算预算重新摊开。观测模型写错，权会把粒子赶到错误位置。

<br>

> [!NOTE]+ 粒子滤波
>
> 用有限样本近似随时间变化的后验。来源是序贯蒙特卡洛，不是卡尔曼滤波的非线性补丁。相对精确前向：时间长度不再进入树宽，但有样本方差。相对卡尔曼：不要求线性高斯。粒子数不足则退化，不是调学习率能修的。相对 Gibbs：粒子滤波沿时间走，Gibbs 在一张静态网上扫变量。

<br>

> [!EXAMPLE]+ 例12 一维跟踪
>
> 位置在 $\{1,\ldots,10\}$，转移以 $0.8$ 留在邻格、以 $0.1$ 向两侧（端点反射）。观测是带噪距离，似然在真位置附近呈峰。$20$ 个粒子从均匀出发。连续两步观测都偏向 $8$ 附近后，粒子应集中到右侧。
>
> 若似然写成均匀，权不更新，滤波等于开环仿真。若转移写成“永远停在原地”，粒子到不了 $8$。若 $K=3$ 且从不重采样，两步之后权会堆在一颗粒子上，后验方差被严重低估。精确前向在 $|Z|=10$ 上可当对拍：粒子直方图应接近 $\alpha_t$ 的归一化。

<br>

> [!NOTE]+ 对照：CSP 与贝叶斯网
>
> CSP 问有没有赋值、或最大权赋值。贝叶斯网问概率。同一张图，零一因子变成 CPT 之后，算法骨架（消去、采样）仍在，输出的数字含义不同。把边缘概率当成“最可能的完整赋值”会错。HMM 前向是 VE 的时间特化；粒子滤波是前向在大 $|Z|$ 上的样本版。

<br>

> [!NOTE]+ 衔接
>
> 变量轴处理有限定义域上的数。下一章把“真与假”写成公式，推断问的是蕴含，不再对定义域求和。概率 $0.99$ 不是蕴含；逻辑作业若出现反模型，不能用“通常成立”搪塞。

<br>


# 逻辑

## 概念：逻辑

- **问题**
  - 用公式表示世界，问 $\mathrm{KB}\models\phi$ 是否对所有模型成立；
  - 作业 From Language to Logic 把自然语言写成一阶公式。
- **范围**
  - 命题逻辑的语法语义、模型检测、消解；
  - 一阶逻辑的量词、合一与还原。
- **章节衔接**
  - 上一章输出是概率或赋值；
  - 本章输出是证明或反模型；
  - 其后课堂扩展回到统计语言模型，二者回答不同问题。

命题符号 $P,Q,\ldots$，连接词 $\neg,\wedge,\vee,\rightarrow$。赋值 / 模型（model）给每个符号真假。公式在给定赋值下的真值按真值表递归定义。知识库 KB 是一组公式（视为合取）。蕴含（entailment）

$$
\mathrm{KB}\models\phi
$$

当且仅当每个使 KB 为真的赋值也使 $\phi$ 为真。等价判定：$\mathrm{KB}\wedge\neg\phi$ 不可满足。可满足（satisfiable）指存在至少一个模型；有效（valid）指在所有赋值下为真。

> [!NOTE]+ 逻辑蕴含
>
> 知识库的每个模型都满足查询。来源是经典证明论与模型论，不是概率大于阈值。相对贝叶斯网：没有“有多可能”，只有“是否必然”。相对检索：公式可以推出从未写过的句子，只要它在所有模型里为真。相对 CSP：CSP 找一个满足赋值；蕴含要求所有满足 KB 的赋值都满足 $\phi$。

<br>

> [!EXAMPLE]+ 例13 三个符号的真值表
>
> $\mathrm{KB}=\{P\vee Q,\ \neg P\vee R\}$，问是否 $\mathrm{KB}\models Q\vee R$。枚举 $(P,Q,R)$ 共 $8$ 行。使 KB 为真的行：$(0,1,0)$ 给出 $Q\vee R=1$；$(0,1,1)$ 为 $1$；$(1,0,1)$ 为 $1$；$(1,1,1)$ 为 $1$。没有 KB 真而 $Q\vee R$ 假的行，故蕴含成立。若查询改成 $R$，则 $(0,1,0)$ 是反模型：KB 真而 $R$ 假。
>
> 模型检测就是把这张表走完。$n$ 个符号 $2^{n}$ 行，符号少时够用。

<br>

## CNF 与消解

合取范式（conjunctive normal form, CNF）：公式写成子句的合取，子句是文字（命题或其否定）的析取。化 CNF 的步骤：

1. 消去 $\rightarrow$：$A\rightarrow B$ 改成 $\neg A\vee B$；
2. 把 $\neg$ 内移（德摩根），消去双否定；
3. 分配 $\vee$ 过 $\wedge$，直到顶层只有 $\wedge$。

分配会使公式指数变长，这是代价，不是可选步骤。子句常写成文字集合；$\{ \}$ 表示空子句，不可满足。

消解（resolution）取含互补文字 $P$ 与 $\neg P$ 的两子句，合并其余文字并去掉互补对。推出空子句则不可满足。命题消解是反驳完备的：不可满足的 CNF 一定能推出空子句。它不是证明完备到“任何重言式都能用几步消解直接写出”，标准用法是反驳：要证 $\mathrm{KB}\models\phi$，对 $\mathrm{KB}\wedge\neg\phi$ 的 CNF 做消解。

> [!INFO]+ 消解一步
>
> 子句 $C\vee P$ 与 $D\vee\neg P$ 推出 $C\vee D$。语义上：任一赋值使 $P$ 为真则须满足 $D$，为假则须满足 $C$，故 $C\vee D$ 在两子句同时真时为真。重复直到空子句或无法再产生新子句（命题情形子句集合有限，会停）。
>
> 因子（factoring）：同一子句里重复文字合成一个，否则永远消不干净。消解不是把两个子句的所有文字无条件并起来；必须先选一对互补文字。
>
> 感性理解：两句话分别说“$P$ 或左边”“非 $P$ 或右边”，合在一起至少一边成立。空子句表示两边都被抽空，知识库加查询的否定走不通。

<br>

> [!EXAMPLE]+ 例14 消解反驳
>
> KB：$P\vee Q$，$\neg Q\vee R$，$\neg R$。查询 $P$。把 $\neg P$ 加入。子句：$(P\vee Q)$，$(\neg Q\vee R)$，$(\neg R)$，$(\neg P)$。
>
> 1. $P\vee Q$ 与 $\neg P$ 消解得 $Q$；
> 2. $Q$ 与 $\neg Q\vee R$ 得 $R$；
> 3. $R$ 与 $\neg R$ 得空子句。
>
> 故 $\mathrm{KB}\models P$。若漏加 $\neg P$，只从 KB 消解，得不到空子句（KB 本身可满足），不能据此否定蕴含。

<br>

## 一阶

一阶逻辑（first-order logic, FOL）加对象、谓词、函数与量词 $\forall,\exists$。项（term）是常量、变量或函数作用在项上。原子是谓词作用在项上。赋值变成论域上的解释（interpretation）：论域非空，常量映射到对象，谓词映射到关系。蕴含仍是“所有解释、所有变量赋值”。

合一（unification）找使两文字相同的最一般替换（most general unifier, MGU）。$\mathrm{Knows}(\mathrm{John},x)$ 与 $\mathrm{Knows}(\mathrm{John},\mathrm{Jane})$ 的 MGU 是 $\{x/\mathrm{Jane}\}$。出现检查（occur check）：$x$ 与 $f(x)$ 不能合一，否则会得到无限项。一阶消解先合一互补文字，再把替换应用到其余文字。

Skolem 化：把 $\exists$ 换成新函数，依赖所有外层 $\forall$ 变量。$\forall x\,\exists y\,R(x,y)$ 变成 $\forall x\,R(x,f(x))$，不能换成同一个常量 $c$，否则语义变成“所有 $x$ 共享同一个 $y$”。$\exists y\forall x$ 才能 Skolem 成常量。量词外移必须先改成前束形，改名避免捕获。

From Language to Logic：把“每门课有一位教师”写成 $\forall c\,\mathrm{Course}(c)\rightarrow\exists t\,\mathrm{Teaches}(t,c)$。量词范围写错是最常见失败，不是合一实现错。存在量词不要先换成某个具体常量，除非做 Skolem 化并记录依赖。

> [!NOTE]+ 一阶逻辑
>
> 在对象与关系上写全称 / 存在命题。来源是谓词演算，课程用它接自然语言作业。相对命题逻辑：同一模式不必为每个常元复制一份子句。相对语言模型：公式的真假可检查，句子的流畅不能代替模型论语义。相对贝叶斯网：没有 CPT，失败模式是反模型，不是低概率。

<br>

> [!INFO]+ Skolem 依赖
>
> $\forall x\,\exists y$ 的 $y$ 可以随 $x$ 变，必须引入一元函数 $f(x)$。$\exists y\forall x$ 的 $y$ 先被选定，再用全称，Skolem 常量即可。把二者写反，得到的 CNF 证明的是另一句话。
>
> 感性理解：每个学生都有一张学生证，证号可以不同，不能先造一个全局常量 `ID0` 塞给所有人。存在一个校长管理所有系，才是常量。

<br>

> [!EXAMPLE]+ 例15 排课语句
>
> “CS221 与 CS229 不能同一时段”写成 $\neg\mathrm{SameSlot}(\mathrm{CS221},\mathrm{CS229})$，或用时段函数 $\mathrm{Slot}(\mathrm{CS221})\ne\mathrm{Slot}(\mathrm{CS229})$。若写成 $\exists t\,\neg\mathrm{At}(\mathrm{CS221},t)\vee\neg\mathrm{At}(\mathrm{CS229},t)$，语义变成“存在某个时段使至少一门不在”，与意图不符。意图是
>
> $$
> \forall t\,\neg\big(\mathrm{At}(\mathrm{CS221},t)\wedge\mathrm{At}(\mathrm{CS229},t)\big).
> $$
>
> “每门课恰好一位教师”需要两个方向：存在一位，以及若两位都教则是同一人。只写 $\exists$ 不禁止两位。自然语言里的“一位”常被漏写成存在而不写唯一。

<br>

## 实现注意

- 先手写小论域的反模型，再交给求解器。量词范围错误无法靠求解器“报个错”发现：求解器证的是写下去的那句。
- 合一时出现检查不能省。作业若允许函数嵌套，漏检查会接受非法替换。
- 命题消解的子句用集合，不要保留重复文字。
- $\rightarrow$ 的优先级低于 $\vee$；漏括号会把 $P\vee Q\rightarrow R$ 读成 $P\vee(Q\rightarrow R)$。

> [!NOTE]+ 对照：概率与逻辑
>
> 贝叶斯网给出 $P(\phi\mid e)$。逻辑给出 $\mathrm{KB}\models\phi$。把 $0.99$ 当成蕴含，或把“通常为真”写成 $\forall$，都会在作业里得到反模型。需要程度就留在概率轴；需要必然就留在逻辑轴。CSP 的可行赋值是逻辑可满足的特例（约束写成子句）；蕴含比“找到一个模型”更强。

<br>

> [!NOTE]+ 衔接
>
> 核心四条轴到此结束。2025 Autumn 用语言模型处理同一语言接口，但推断是 next-token 似然，不是消解。社会专题处理评价指标与供应链，不改前面的定义。流畅的句子可以不满足 KB；满足 KB 的公式可以读起来不像人话。

<br>


# 语言模型

## 概念：语言模型

- **问题**
  - 对 token 序列定义 $P(x_1,\ldots,x_T)=\prod_t P(x_t\mid x_{<t})$；
  - 2025 Autumn 新增讲座，不是 2026 Spring 核心作业。
- **范围**
  - 写出自回归分解、困惑度与提示；
  - 输入文本，输出下一 token 分布或生成序列。
- **章节衔接**
  - 上一章用公式保证蕴含；
  - 本章用数据拟合条件分布；
  - 流畅句子不必满足 KB。

自回归（autoregressive）语言模型把联合分布拆成逐步预测

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1}).
$$

这与贝叶斯网的链式法则相同，父集是全部前缀。$n$-gram 把父集截成最近 $n-1$ 个 token；神经模型（含 Transformer）用网络近似完整条件。训练最小化平均负对数似然

$$
J=\frac1N\sum_{i=1}^{N}\sum_{t=1}^{T_i}-\log P_{\theta}(x_t^{(i)}\mid x_{<t}^{(i)}).
$$

困惑度（perplexity）是 $\exp(J)$（按 token 平均）。它是似然的单调变换，不是准确率，也不能代替“是否满足 KB”。

提示（prompt）是推理期条件：固定前缀，对后续 token 取样或取 $\arg\max$。提示不是另一次梯度更新。温度 $\tau$ 把 logits 除以 $\tau$ 再 softmax：$\tau\rightarrow 0$ 近贪心，$\tau$ 大则更平。束搜索在 token 层保留 $K$ 条前缀，与排课作业的束搜索同形，同样不保证最优完整序列。

> [!NOTE]+ 语言模型
>
> 用 next-token 条件分布定义文本上的概率。来源是经典 $n$-gram 与当代 Transformer 语言模型。相对一阶逻辑：可以生成从未写过的通顺句子，但不提供模型论保证。相对贝叶斯网：变量是 token，因子由网络计算，树宽不再是讨论对象。相对反射：生成是一条序列，每步条件依赖已写出的前缀，但推断仍是局部 $\arg\max$ 或采样，不是 UCS。

<br>

> [!EXAMPLE]+ 例16 两个 token 的玩具
>
> 词表 $\{a,b\}$。设 $P(x_1=a)=0.6$，$P(x_2=a\mid x_1=a)=0.9$，$P(x_2=a\mid x_1=b)=0.2$。则 $P(ab)=0.6\cdot 0.1=0.06$，$P(ba)=0.4\cdot 0.2=0.08$。贪心先取 $a$，再取 $a$，得到 $aa$，联合 $0.54$，是最大。若把第二步温度开得很大，样本会经常落到 $ab$，联合更小。困惑度按 $-\log P$ 平均，不看“读起来像不像句子”。

<br>

与课程前半的接口：

- 检索加生成（RAG）仍要分开报命中率与忠实度。检索是变量 / 反射轴上的最近邻，生成是自回归，二者的错误会叠加。
- 把 Pac-Man 策略写成提示不会自动得到 minimax。语言模型没有展开对手树。
- 逻辑作业若改用语言模型翻译句子，必须再做符号检验。流畅的 $\forall$ 位置可以是错的。
- 情感分类仍是反射：把整句送进冻结 LM 取向量再训线性头，推断仍是一次前向，不是消解。

幻觉（hallucination）是高似然但与证据或世界冲突的续写。降低温度或加 RAG 可以减少一部分，不能变成蕴含。课程不把 RLHF 细节当作考试核心。

> [!NOTE]+ 衔接
>
> 语言模型回答的是“下文在数据分布上有多可能”。社会专题问的是系统部署之后对谁造成伤害、指标有没有把伤害藏起来。二者都不改 UCS、贝尔曼或消解的定义。

<br>


# 社会

## 概念：社会

- **问题**
  - 平均指标高仍可能在分组上系统失败；
  - 2025 Autumn 有 AI & Society 作业与供应链讲座。
- **范围**
  - 数据、标注、评价、部署与供给链；
  - 输入系统描述，输出伤害与指标，而不是新的推断公式。
- **章节衔接**
  - 前面各章优化训练损失、路径代价、回报或边缘似然；
  - 本章这些标量都不自动等于社会代价。

先写伤害，再选指标。伤害随作业变化：错分的评论被压、路径规划牺牲某类街区、Q 学习策略在少数起始状态下冲进火山、跟踪系统的误报触发错误处置。总体准确率、总体回报、总体边缘似然都可以把分组失败藏起来。

供应链（supply chain）：算力、数据许可、标注劳动、评测集污染、API 依赖。模型卡只写平均准确率，不足以支持部署。情感分类作业的词表会放大语料里的刻板相关；这是特征问题，不是 SGD 实现错。删掉敏感词之后，邮政编码、用词长度、表情符号仍可作为代理特征。

分组指标至少报：各组样本量、各组准确率或回报、最差组、以及总体。样本量极不均衡时，总体被多数组拖着走。公平性定义彼此不兼容（校准与均等机会一般不能同时精确成立），作业要求写清选了哪一条，而不是写“已经公平”。

> [!NOTE]+ AI 与社会
>
> 把系统放回数据来源与使用情境。来源是课程的 Embedded Ethics 与 2025 Autumn 专题，不是正则项。相对只报 $\mathrm{TrainLoss}$：同一铰链损失可以在不同群体上间隔分布完全不同。相对“删掉敏感词”：代理特征仍在。相对逻辑：社会伤害不是反模型，不能靠消解证没。

<br>

> [!WARNING]+ 指标与伤害
>
> 先写清伤害，再选指标。总体准确率、总体回报、总体边缘似然都可以把分组失败藏起来。用测试集调启发式或 $\varepsilon$，再把同一测试当最终分数，评价本身已经泄漏。供应链上的评测集污染（试题进入预训练语料）会让困惑度或准确率虚高，不改变模型在真实分布上的行为。

<br>

> [!EXAMPLE]+ 例17 情感分组
>
> 总体准确率 $0.90$。按方言或身份相关用词切开，其中一组 $0.62$，样本占 $8\%$。铰链损失仍在下降，因为多数组间隔在变大。加 $L_2$ 不会自动抬起该组。必须另报该组混淆矩阵，并检查词表是否把该组的极性词大量送进 `UNK`。

<br>

> [!NOTE]+ 衔接
>
> 社会专题不引入新的最优方程。下一章按作业与考试把算法对回模型族，并把“要报告的量”写进同一张表。

<br>


# 作业

## 概念：作业

- **问题**
  - 2026 Spring 八份周作业围绕应用，书面加编程；
  - 2026 Spring 作业页 zip 未在公开站挂出，题面与同主讲的 [2025 Spring](https://stanford-cs221.github.io/spring2025/) 对齐；2025 Autumn 用贝叶斯网与 AI & Society 替换排课与车辆跟踪。
- **范围**
  - 概括每题问什么，用 callout 给出可对拍的解答与实现要点；
  - 不照抄题面长文。数字与公式以 2025 Spring 公开页为准，学期微调时先对 zip 里的 `*.tex`。
- **章节衔接**
  - 前面各章给定义与递推；
  - 本章把定义接到 `submission.py` 与书面 PDF。

| 编号 | 作业 | 轴 | 公开题面 |
| :--- | :--- | :--- | :--- |
| HW1 | Foundations | 预备 | [Spr25](https://stanford-cs221.github.io/spring2025/assignments/foundations/index.html) |
| HW2 | Sentiment | 反射 | [Spr25](https://stanford-cs221.github.io/spring2025/assignments/sentiment/index.html) |
| HW3 | Route | 搜索 | [Spr25](https://stanford-cs221.github.io/spring2025/assignments/route/index.html) |
| HW4 | Mountain Car | MDP | [Spr25](https://stanford-cs221.github.io/spring2025/assignments/mountaincar/index.html) |
| HW5 | Pac-Man | 博弈 | [Spr25](https://stanford-cs221.github.io/spring2025/assignments/pacman/index.html) |
| HW6 | Scheduling | 因子 / CSP | [Spr25](https://stanford-cs221.github.io/spring2025/assignments/scheduling/index.html) |
| HW7 | Car tracking | 贝叶斯 / HMM | [Spr25](https://stanford-cs221.github.io/spring2025/assignments/car/index.html) |
| HW8 | Logic | 逻辑 | [Spr25](https://stanford-cs221.github.io/spring2025/assignments/logic/index.html) |
| Aut HW6 | Bayesian | 贝叶斯 | [Aut25](https://stanford-cs221.github.io/autumn2025/assignments/hw6_bayesian/index.html) |
| Aut HW8 | Society | 评价 | [Aut25](https://stanford-cs221.github.io/autumn2025/assignments/hw8_society/index.html) |

课表入口：[2026 Spring](https://stanford-cs221.github.io/spring2026/)、[作业规范](https://stanford-cs221.github.io/spring2026/homework.html)。提交只改 `submission.py` 与排版 PDF。自动评分只显示部分用例。迟交共 7 天、单份最多 2 天。

> [!NOTE]+ 读法（先看这个）
>
> 每份作业只做三件事：**建模**（状态 / 特征 / 公式是什么）、**推断**（算哪一个数）、**对拍**（用最小例子验算）。callout 的顺序固定为：题在问什么、符号含义、先用具体数字走一遍、再写一般式、最后把要交的答案圈出来。
>
> 卡住时从后往前找：数字对不上，多半是符号看错（$y$ 是 $\pm 1$ 还是 $0/1$；$V$ 是旧表还是新表；XOR 要不要等于 $1$）。不要从中间公式开始背。
>
> 书面题要交的是推导，不是只写最终小数。编程题先在纸上写出 `startState` / `isEnd` / 更新式，再写代码。
>
> 每份作业先盯死一个最容易看错的符号：
>
> | 作业 | 先盯这个 |
> | :--- | :--- |
> | HW1 | $w_i$ 是权重不是位置；$P(A\mid B)$ 分子是交集 |
> | HW2 | $y=\pm 1$，不是 $0/1$；铰链在 $ys\ge 1$ 时梯度为 $0$ |
> | HW3 | 代价里的 $x$ 是出发格横坐标；途经点状态必须带集合 $S$ |
> | HW4 | 第 $i$ 轮的 $Q$ 只能用 $V_{i-1}$；终点 $V$ 永远是 $0$ |
> | HW5 | $d$ 只在最后一个幽灵走完后减一 |
> | HW6 | XOR 为 $1$ 才合法；回溯不调用冲突叶子 |
> | HW7 | 先 `elapseTime` 再 `observe`；`pdf` 可以大于 $1$ |
> | HW8 | “当且仅当”用 $\leftrightarrow$；量词左右顺序不能换 |

<br>

> [!WARNING]+ 对拍与荣誉守则
>
> 公开页与当年 zip 可能改数字。先用小例子手算，再跑 `grader.py`，最后交 Gradescope。笔记里的解答用于理解，不能整段贴进作业；课程禁止对照往年官方解答或他人代码。生成式工具若使用必须交 transcript，且不得用来“检查”已写答案。

<br>

## Foundations

书面：优化、算子次序、期望递推、MLE、条件概率、梯度、复杂度、伦理。编程：六个短函数（字典序首词、欧氏距离、句子变异、稀疏点乘与累加、非单例词）。

> [!EXAMPLE]+ HW1.1a 加权二次
>
> **题在问什么**：数轴上有若干点 $x_i$，每个点有正权重 $w_i$。找一个位置 $\theta$，让“到各点的加权平方距离和”最小。
>
> **先算一个数**。两点 $x_1=0,x_2=4$，权重 $w_1=1,w_2=3$。
>
> $$
> f(\theta)=1\cdot(\theta-0)^{2}+3\cdot(\theta-4)^{2}=\theta^{2}+3(\theta^{2}-8\theta+16)=4\theta^{2}-24\theta+48.
> $$
>
> 求导 $f'=8\theta-24=0$，得 $\theta=3$。验算：加权平均 $(1\cdot 0+3\cdot 4)/(1+3)=12/4=3$。$f''=8>0$，是谷底不是山峰。
>
> **一般式**。$f(\theta)=\sum_i w_i(\theta-x_i)^{2}$。
>
> $$
> f'(\theta)=2\sum_i w_i(\theta-x_i)=2\Big(\theta\sum_i w_i-\sum_i w_i x_i\Big)=0
> \implies
> \theta^{\ast}=\frac{\sum_i w_i x_i}{\sum_i w_i}.
> $$
>
> $f''=2\sum_i w_i$。全体 $w_i>0$ 则 $f''>0$，驻点是唯一最小。
>
> **$w_i$ 变负会怎样**。取 $n=1,w_1=-1$，$f(\theta)=-(\theta-x_1)^{2}$。$\theta$ 拉得越远，$f$ 越小（趋向 $-\infty$），没有最小点。权重为负时，问题不再是“靠近这些点”。
>
> **要交的答案**：$\theta^{\ast}$ 的式子、二阶导说明是最小、$w_i<0$ 时可能无下界。

<br>

> [!EXAMPLE]+ HW1.1b 交换 min 与求和
>
> **题在问什么**。$f$ 先加再对一个共用的 $s$ 取最小。$g$ 先对每个坐标各自取最小再加。问 $f$ 与 $g$ 谁大。
>
> **符号**。$s\in[-1,1]$ 表示在 $-1$ 与 $1$ 之间选一个数（含两端）。$\min$ 表示在允许范围内把式子压到尽可能小。
>
> **先算一个数**。取 $x=(3,-1)$。
>
> - $f$：只能选一个 $s$。$f= \min_s(s\cdot 3+s\cdot(-1))=\min_s(2s)$。$s=-1$ 时得 $-2$。
> - $g$：第一个坐标选 $s_1=-1$ 得 $-3$；第二个选 $s_2=+1$ 得 $-1$。和为 $-4$。
>
> $-2>-4$，所以这个例子里 $f>g$。
>
> **为什么永远 $f\ge g$**。两式都在最小化 $\sum_i s_i x_i$。$g$ 允许每个 $s_i$ 不同；$f$ 强迫它们相等。选择更少，最小值只会更大或持平。
>
> 写成闭式：$f=-|\sum_i x_i|$，$g=-\sum_i|x_i|$。因为 $|\sum x_i|\le\sum|x_i|$，取负号后不等式翻转，$f\ge g$。$x$ 全同号时等号。
>
> **要交的答案**：$f\ge g$（对一切 $x$），加上上面的可行集或三角不等式证明。

<br>

> [!EXAMPLE]+ HW1.1c 骰子期望
>
> **题在问什么**。反复掷公平骰子，看到 $1$ 就停。看到 $3$ 加 $a$ 分，看到 $6$ 减 $b$ 分。问停下来时得分的**期望**（平均会是多少）。
>
> **为何能写成自己等于自己**。无论第一面是什么（除了 $1$），游戏规则与开始时完全一样，所以“再玩下去”的期望还是同一个 $V$。这叫一步递推，不是循环定义错误。
>
> | 第一面 | 概率 | 立刻发生 | 之后的期望 |
> | :--- | :--- | :--- | :--- |
> | $1$ | $1/6$ | 得 $0$ 并停止 | $0$ |
> | $2$ 或 $4$ 或 $5$ | $3/6=1/2$ | 得 $0$ | 还剩一个完整游戏 $=V$ |
> | $3$ | $1/6$ | 得 $a$ | 再加 $V$ |
> | $6$ | $1/6$ | 得 $-b$ | 再加 $V$ |
>
> $$
> V=\frac16\cdot 0+\frac12\cdot V+\frac16(a+V)+\frac16(-b+V).
> $$
>
> 把含 $V$ 的项收在一起：$\frac12+\frac16+\frac16=\frac56$。不含 $V$ 的项：$(a-b)/6$。
>
> $$
> V=\frac56 V+\frac{a-b}{6}
> \implies
> \frac16 V=\frac{a-b}{6}
> \implies
> V=a-b.
> $$
>
> **小检查**。若 $a=b$，每出现一次 $3$ 的赚会被一次 $6$ 抵消，期望应为 $0$，与 $V=a-a=0$ 一致。
>
> **要交的答案**：上面的递推、$V=a-b$、一两句说明“未停止则期望仍是 $V$”。

<br>

> [!EXAMPLE]+ HW1.1d 硬币 MLE
>
> **题在问什么**。一枚硬币正面概率是 $p$。掷了五次，结果是反、正、正、正、正。问哪个 $p$ 让这串结果**最像会发生**。
>
> **似然（likelihood）**就是“在这个 $p$ 下，正好掷出这串”的概率：
>
> $$
> L(p)=p^{4}(1-p).
> $$
>
> **先代入三个数感受一下**。
>
> - $p=0.5$：$L=0.5^{4}\cdot 0.5=0.03125$。
> - $p=0.8$：$L=0.8^{4}\cdot 0.2=0.08192$。
> - $p=0.9$：$L=0.9^{4}\cdot 0.1=0.06561$。
>
> $0.8$ 比 $0.5$ 和 $0.9$ 都大。正式求最大：$\log$ 只拉伸纵轴、不改峰的位置，所以改最大化 $\ell=\log L=4\log p+\log(1-p)$。
>
> $$
> \ell'(p)=\frac4p-\frac{1}{1-p}=0
> \implies
> 4(1-p)=p
> \implies
> p=\frac45.
> $$
>
> $\ell''(p)=-4/p^{2}-1/(1-p)^{2}<0$，峰是最大不是最小。$p$ 贴到 $0$ 或 $1$ 时 $L\rightarrow 0$，小于 $L(4/5)$。
>
> **一句话解释**：$5$ 次里 $4$ 次正面，频率就是 $4/5$。Bernoulli 的最大似然估计（maximum likelihood estimate, MLE）永远等于样本频率。
>
> **要交的答案**：$p=4/5$、求导与二阶导、一句“等于正面频率”。

<br>

> [!INFO]+ HW1.1e 条件概率
>
> **题在问什么**。事件 $A$ 与 $B$ 满足三条：$P(A\mid B)=P(B\mid A)$，并集概率是 $1/3$，交集概率严格大于 $0$。要证 $P(A)>1/6$。
>
> **符号**。$P(A\mid B)$ 读“已经发生 $B$ 的前提下，$A$ 再发生的概率”，定义是 $P(A\cap B)/P(B)$。并集 $A\cup B$ 是“至少一个发生”，交集 $A\cap B$ 是“两个都发生”。
>
> **先用数字感受**。若 $P(A)=P(B)=0.2$、$P(A\cap B)=0.07$，则 $P(A\mid B)=0.07/0.2=0.35=P(B\mid A)$，并集 $0.2+0.2-0.07=0.33=1/3$。此时 $0.2>1/6\approx 0.167$。
>
> **一般证明只走三步**。
>
> 1. $P(A\cap B)/P(B)=P(A\cap B)/P(A)$。交集大于 $0$，两边可约，得 $P(A)=P(B)$。
> 2. $P(A\cup B)=P(A)+P(B)-P(A\cap B)=2P(A)-P(A\cap B)=1/3$。
> 3. 减去一个正数会变小，所以 $1/3=2P(A)-P(A\cap B)<2P(A)$，即 $P(A)>1/6$。
>
> **要交的答案**：上面三步。不必假设 $A$ 与 $B$ 互斥。

<br>

> [!INFO]+ HW1.1f 梯度
>
> **题在问什么**。$w$ 是 $d$ 维列向量。函数是“很多平方差”再加一项 $L_2$ 正则。要写出 $\nabla f(w)$ 这个 $d$ 维向量。
>
> **先当所有东西都是标量**（$d=1$）。$f(w)=\sum_i\sum_j(a_i w-b_j w)^{2}+(\lambda/2)w^{2}$。令 $u_{ij}=(a_i-b_j)w$，则 $u_{ij}^{2}$ 对 $w$ 的导数是 $2u_{ij}(a_i-b_j)$。正则项导数是 $\lambda w$。
>
> **向量时只换记号**。内积 $(a_i-b_j)^{\top}w$ 代替 $(a_i-b_j)w$。对向量 $w$ 求导，$u_{ij}$ 的梯度是向量 $a_i-b_j$（不再是标量）。
>
> $$
> \nabla f(w)=2\sum_{i=1}^{n}\sum_{j=1}^{n}\big((a_i-b_j)^{\top}w\big)(a_i-b_j)+\lambda w.
> $$
>
> 不要把 $\|\cdot\|_{2}^{2}$ 拆成 $\sum_k w_k^{2}$ 再对每个 $k$ 写一遍。$d=1$ 时上式退回刚才的普通导数，可用来对拍。
>
> **要交的答案**：上面这一行。保持紧凑，不必展开成坐标。

<br>

> [!EXAMPLE]+ HW1.2 复杂度
>
> **第一问：三个矩形有多少种放法**。$n\times n$ 格点。轴对齐矩形由左、右、上、下四个边界决定。
>
> 先只放**一个**矩形：左边界从 $n$ 个 $x$ 里选，右边界也从 $n$ 个 $x$ 里选（允许重合，面积可以为 $0$），共 $n^{2}$。上下同理又是 $n^{2}$。一个矩形 $O(n^{4})$。三个矩形各自独立选位置，可以完全重叠，所以 $(n^{4})^{3}=O(n^{12})$。只问渐近，重不重叠、谁先谁后都不改指数。
>
> **第二问：带负代价的格子最短路**。$n$ 行 $2n$ 列，从左上 $(1,1)$ 走到右下 $(n,2n)$，每步只能向下或向右。格子代价 $c(i,j)$ **可以为负**。
>
> **最小例子** $n=1$，只有一行两列。唯一路径 $(1,1)\rightarrow(1,2)$，代价 $c(1,1)+c(1,2)$。
>
> **再大一点** $2\times 4$。走到格子 $(i,j)$ 的最小路径和记为 $F(i,j)$（含该格自身）。只能从上方或左方进来：
>
> $$
> F(1,1)=c(1,1),\qquad
> F(i,j)=c(i,j)+\min\big(F(i-1,j),\ F(i,j-1)\big).
> $$
>
> 没有上方就把 $F(i-1,j)$ 当成 $+\infty$。格子共 $n\cdot 2n=O(n^{2})$ 个，每格只看两个邻居，总时间 $O(n^{2})$。
>
> 不能用普通 BFS（步代价不是全 $1$）。不能只跑要求非负边的 Dijkstra。这张图没有回头边，是有向无环图（directed acyclic graph, DAG），动态规划即可。
>
> **要交的答案**：$O(n^{12})$；上面的 $F$ 递推与 $O(n^{2})$。

<br>

> [!NOTE]+ HW1.3 伦理与编程
>
> **书面怎么写**。打开 NeurIPS 伦理清单，每则情景抄出**一条**并划线，再写 2 到 5 句：机制是什么、伤害落在谁身上。不是写感想。
>
> | 情景 | 可对上的一条（示例） | 机制一句话 |
> | :--- | :--- | :--- |
> | 贷款模型主要看地点，且对黑人与男性假阳性更高 | 偏见 / 分配伤害 | 地点代理了群体，拒贷集中落在农村与特定群体 |
> | 文体学给匿名文本找作者，也可用于代码 | 隐私 | 去匿名把“不想被认出”的人重新绑回身份 |
> | 未授权刮名人脸训练识别 | 同意与版权 | 人像与照片未经许可进入训练集 |
> | iNaturalist 植物识别，用户同意且只在应用内 | 可论证无清单项 | 必须写清同意范围：训练用途、是否外传 |
>
> **编程六个短函数**。禁用 numpy。`find_alphabetically_first_word` 按字典序取最小词。`euclidean_distance` 是 $\sqrt{\sum(a_i-b_i)^{2}}$。`mutate_sentences` 把相邻词对当成边，重接出新句子。稀疏向量用 `dict`：点乘只遍历较短的那个；`increment` 是 $v\leftarrow v+\mathrm{scale}\cdot u$。`find_nonsingleton_words` 返回出现次数至少为 $2$ 的词集合。

<br>

## Sentiment

先把四个符号钉死，再动手：

- $y=+1$ 表示正评，$y=-1$ 表示负评（不是 $0/1$）。
- $\phi(x)$ 是词袋：某个词出现一次，对应维就是 $1$，没出现就是 $0$。
- 分数 $s=w\cdot\phi$：只把“出现了的词”的权加起来。
- 铰链损失 $\max(0,1-ys)$：若 $ys\ge 1$（分对且间隔够），损失为 $0$，**不改** $w$；否则 $w\leftarrow w+\eta y\phi$。$\eta=0.1$ 是步长。

词序固定为 $[\mathrm{pretty},\mathrm{good},\mathrm{bad},\mathrm{plot},\mathrm{not},\mathrm{scenery}]$。$w$ 初值 $[0,0,0,0,0,0]$。间隔恰好等于 $1$ 时题目规定梯度取 $0$。

> [!EXAMPLE]+ HW2.1a 四步 SGD
>
> **题在问什么**。四条短评按顺序各做一次随机梯度下降（stochastic gradient descent, SGD）。每次只改“这句话里出现过的词”的权。要交的是六维向量，顺序固定为 pretty, good, bad, plot, not, scenery。
>
> **每一步只做三件事**：算分数 $s=w\cdot\phi$；看 $ys$ 是否小于 $1$；小于 $1$ 才更新 $w\leftarrow w+\eta y\phi$。
>
> **步 1** `not good`，$y=-1$。$w$ 全 $0$，所以 $s=0$，$ys=0<1$，要更新。$y=-1$ 表示把出现的词的权各减 $0.1$：
>
> $$
> w\leftarrow [0,0,0,0,0,0]+0.1\cdot(-1)\cdot[0,1,0,0,1,0]=[0,\ -0.1,\ 0,\ 0,\ -0.1,\ 0].
> $$
>
> **步 2** `pretty bad`，$y=-1$。$s=0$（pretty 与 bad 当前权都是 $0$），$ys=0<1$。pretty 与 bad 各减 $0.1$：
>
> $w=[-0.1,\ -0.1,\ -0.1,\ 0,\ -0.1,\ 0]$。
>
> **步 3** `good plot`，$y=+1$。$s=w_{\mathrm{good}}=-0.1$，$ys=-0.1<1$。这次 $y=+1$，出现的词各**加** $0.1$：good 从 $-0.1$ 回到 $0$，plot 变成 $0.1$。
>
> $w=[-0.1,\ 0,\ -0.1,\ 0.1,\ -0.1,\ 0]$。
>
> **步 4** `pretty scenery`，$y=+1$。$s=w_{\mathrm{pretty}}=-0.1$，$ys=-0.1<1$。pretty 加 $0.1$ 回到 $0$，scenery 变成 $0.1$。
>
> **要交的答案**：$[0,\ 0,\ -0.1,\ 0.1,\ -0.1,\ 0.1]$。
>
> 检查：bad 与 not 只在负评里出现过，权为负；plot 与 scenery 只在正评里出现过，权为正。

<br>

> [!EXAMPLE]+ HW2.1b 词袋不可分
>
> **题在问什么**。四句话：`bad` 负，`good` 正，`not bad` 正，`not good` 负。只用“每个词出现几次”当特征，能否找到一组权让四句全对？不能的话，再补一个特征把它救回来。
>
> **线性分类器在干什么**。预测符号是 $\mathrm{sign}(w\cdot\phi)$。要全对，分数必须与标签同号（分数为 $0$ 也算错）。
>
> **先列加法表**。
>
> | 句子 | 要满足 |
> | :--- | :--- |
> | `good` | $w_{\mathrm{good}}>0$ |
> | `not good` | $w_{\mathrm{not}}+w_{\mathrm{good}}<0$，即 $w_{\mathrm{not}}<-w_{\mathrm{good}}<0$ |
> | `bad` | $w_{\mathrm{bad}}<0$ |
> | `not bad` | $w_{\mathrm{not}}+w_{\mathrm{bad}}>0$，即 $w_{\mathrm{not}}>-w_{\mathrm{bad}}>0$ |
>
> 第二行强迫 $w_{\mathrm{not}}<0$，第四行强迫 $w_{\mathrm{not}}>0$。同一个数不能既负又正。这与 SGD、铰链、步长都无关，是特征本身的线性不可分，形状与异或（XOR）相同。
>
> **补一个特征即可**。例如 $\phi_{\mathrm{not\_good}}(x)=\mathbf{1}\{\text{同时出现 not 与 good}\}$，或把 `not good` 整块当成单独一维。加上去之后，`not` 不再需要同时扮演“翻转 good”和“翻转 bad”两个角色。
>
> **要交的答案**：上面的矛盾证明，加一个具体新特征。

<br>

> [!INFO]+ HW2.2 平方损失加 sigmoid
>
> **题在问什么**。现在不预测正负，而是预测 $[0,1]$ 里的评分。预测值是 $p=\sigma(w\cdot\phi)$，$\sigma(z)=(1+e^{-z})^{-1}$ 把任意实数压进 $(0,1)$。损失是平方 $(p-y)^{2}$。要写出损失、对 $w$ 的梯度，以及 $y=1$ 时梯度能不能变成精确的 $0$。
>
> **损失**。$\mathrm{Loss}(x,y,w)=\big(\sigma(w\cdot\phi(x))-y\big)^{2}$。
>
> **梯度**。令 $z=w\cdot\phi$，$p=\sigma(z)$。链式法则：平方对 $p$ 是 $2(p-y)$，$\sigma$ 对 $z$ 是 $p(1-p)$，$z$ 对 $w$ 是 $\phi$。
>
> $$
> \nabla_w\mathrm{Loss}=2(p-y)\,p(1-p)\,\phi(x).
> $$
>
> **先代入一个数**。设 $\phi=(1)$，$w=2$，则 $z=2$，$p=\sigma(2)\approx 0.88$。若 $y=1$，梯度模长约为 $2(0.88-1)\cdot 0.88\cdot 0.12\approx 0.025$。已经很小。
>
> **$y=1$ 时模长**等于 $2p(1-p)^{2}\|\phi\|$。$p$ 越靠近 $1$，$(1-p)^{2}$ 越小。要把模长压到任意小，须 $w\cdot\phi\rightarrow+\infty$（$\phi$ 固定且非零）。$\sigma$ 永远到不了 $1$，所以梯度永远不是精确的 $0$。这就是消失梯度（vanishing gradient）：已经很对的点几乎不再迈步。
>
> **要交的答案**：损失式、梯度式、$w\cdot\phi\rightarrow+\infty$ 使模长任意小、但不能精确为零。

<br>

> [!EXAMPLE]+ HW2.4 毒性两组损失
>
> **题在问什么**。评论特征 $\phi=(1,d,t)$：$1$ 是偏置，$d=1$ 表示出现身份词，$t=1$ 表示出现“脏词”。标签 $y=+1$ 有毒，$y=-1$ 无毒。两组按 $(y,d)$ 切成四块。比较两个固定分类器：D 只看身份词，T 只看脏词。
>
> **先看分类器在干什么**（分数 $>0$ 预测有毒）。
>
> - D 的权 $w=(-0.1,1,0)$，分数 $-0.1+d$。$d=1$ 得 $0.9>0$；$d=0$ 得 $-0.1<0$。故 D 判有毒**当且仅当**出现身份词。
> - T 的权 $w=(-0.1,0,1)$，同理判有毒当且仅当 $t=1$。
>
> **四组人数先加起来**（表里 $t=0$ 与 $t=1$ 两行相加）：
>
> | 组 $(y,d)$ | 人数 |
> | :--- | :--- |
> | $(1,1)$ 有毒且有身份词 | $27+63=90$ |
> | $(1,0)$ 有毒且无身份词 | $3+7=10$ |
> | $(-1,1)$ 无毒且有身份词 | $7+3=10$ |
> | $(-1,0)$ 无毒且无身份词 | $63+27=90$ |
>
> **D 的零一损失**。D 的预测完全由 $d$ 决定，与 $t$ 无关：有身份词一律判有毒。于是 $(1,1)$ 全对、$(-1,1)$ 全错、$(1,0)$ 全错、$(-1,0)$ 全对。
>
> |  | $y=1$ | $y=-1$ |
> | :--- | :--- | :--- |
> | $d=1$ | $0$ | $1$ |
> | $d=0$ | $1$ | $0$ |
>
> 平均 $(90\cdot 0+10\cdot 1+10\cdot 1+90\cdot 0)/200=0.1$。最大组损失 $1$（两个小组成绩是满分错）。
>
> **T 的零一损失**。T 只看 $t$。在 $(y=1,d=1)$ 这 $90$ 条里，$t=0$ 的 $27$ 条会被判无毒，是错的，组损失 $27/90=0.3$。其余三组同样都是 $0.3$。
>
> |  | $y=1$ | $y=-1$ |
> | :--- | :--- | :--- |
> | $d=1$ | $0.3$ | $0.3$ |
> | $d=0$ | $0.3$ | $0.3$ |
>
> 平均 $0.3$。最大组损失 $0.3$。
>
> **谁更好**。D 平均更好（$0.1<0.3$），T 最大组更好（$0.3<1$）。
>
> **$\lambda$ 新损失** $\lambda\cdot\mathrm{avg}+(1-\lambda)\cdot\mathrm{max}$。
>
> - $\lambda=1$ 只压平均：对齐功利式总体福利，偏向 D 这种“多数对、少数组全灭”的分类器。
> - $\lambda=0$ 只压最差组：对齐最弱势优先，偏向 T。
> - 第二张表里 $(y=1,d=1)$ 只有 $1$ 条，最大组损失会被一条样本绑架。可对组损失加样本量下限、或用平滑后再取最大。
> - 平台同等关心平均与最差组，取 $\lambda=1/2$，并写清权衡：总体误拦少一点，还是不让某一身份组被系统性误杀。
>
> **要交的答案**：两个“当且仅当”句子、两张表、两个平均、两个最大、$\lambda$ 与伦理框架的对应。

<br>

> [!EXAMPLE]+ HW2.5 $k$-means
>
> **题在问什么**。四个二维点其实都在 $x$ 轴上：$\phi=[0,0],\ [4,0],\ [6,0],\ [11,0]$。做 $2$-means，两套初值各跑到收敛，报告每步的划分、中心、总损失。
>
> **一轮 $k$-means 只做两件事**。分配：每个点去离它最近的中心。更新：中心改成该簇点的均值。划分不再变就停。
>
> **初值 A** $\mu_1=[0,0]$，$\mu_2=[11,0]$。距离就是横坐标差的绝对值。
>
> 1. $0$ 归 $1$；$4$ 距 $0$ 为 $4$、距 $11$ 为 $7$，归 $1$；$6$ 距 $0$ 为 $6$、距 $11$ 为 $5$，归 $2$；$11$ 归 $2$。簇 $\{[0,0],[4,0]\}$、$\{[6,0],[11,0]\}$。
> 2. 更新：$\mu_1=[2,0]$，$\mu_2=[8.5,0]$。
> 3. 再分配：$4$ 距 $2$ 为 $2$、距 $8.5$ 为 $4.5$，仍归 $1$；$6$ 距 $2$ 为 $4$、距 $8.5$ 为 $2.5$，仍归 $2$。划分不变，停止。
>
> 损失 $(0-2)^{2}+(4-2)^{2}+(6-8.5)^{2}+(11-8.5)^{2}=4+4+6.25+6.25=20.5$。
>
> **初值 B** $\mu_1=[0,0]$，$\mu_2=[4,0]$。
>
> 1. $0$ 归 $1$；$4,6,11$ 都距 $4$ 更近，归 $2$。
> 2. 更新：$\mu_1=[0,0]$，$\mu_2=[7,0]$。
> 3. $4$ 距 $0$ 为 $4$、距 $7$ 为 $3$，仍归 $2$。划分不变。
>
> 损失 $0+(4-7)^{2}+(6-7)^{2}+(11-7)^{2}=9+1+16=26$。
>
> 同一套数据，初值不同，损失 $20.5$ 与 $26$ 都可能停住。$k$-means 不保证全局最小。
>
> **缩放**。全体坐标乘同一个非零常数 $c$，每段距离同乘 $|c|$，谁更近不会换，划分不变。只缩放某几维会改相对远近。反例：点 $(0,0),(1,0),(0,1)$，只把第 $2$ 维乘 $100$，$(0,1)$ 会从“靠近 $(0,0)$”变成“离谁都远”，划分可变。
>
> **实现**。预计算 $\|x\|^{2}$，用 $\|x-\mu\|^{2}=\|x\|^{2}+\|\mu\|^{2}-2x\cdot\mu$，不要每步开方。
>
> **要交的答案**：两套初值的逐步划分与中心、损失 $20.5$ 与 $26$、缩放两问。

<br>

> [!NOTE]+ HW2 编程
>
> **词特征**。`extractWordFeatures` 按空白切开。`great` 与 `Great` 是两个词。返回 `dict`，键是词，值是出现次数。
>
> **学习**。`learnPredictor` 用已有的 `dotProduct` / `increment` 做铰链 SGD，不要手改字典。每一轮打印训练误差与验证误差。满分线：训练 $<4\%$，验证 $<30\%$。对不上时先用 `generateDataset` 造线性可分的假数据自测。
>
> **字符 $n$-gram**。先去掉空格与制表符，再滑窗。验证误差通常在 $n=3$ 到 $5$ 最低：太短没有极性，太长稀疏。词特征会把未见过的长词当成全新维；字符块 `good` 在 `unbelievably good` 里仍能触发。书面要写清选中的 $n$，以及这一句反例。

<br>

## Route

网格：从 $(0,0)$ 到 $(m,n)$，$m,n\ge 0$。动作东 / 西 / 南 / 北，代价 $1+\max(x,0)$，其中 $x$ 是**出发格**的横坐标。

> [!EXAMPLE]+ HW3.1a 最小代价
>
> **题在问什么**。从格子 $(0,0)$ 走到 $(m,n)$。每走一步，代价是 $1+\max(x,0)$，这里的 $x$ 是**出发前**所在格子的横坐标。越靠东（$x$ 越大），无论朝哪走都越贵。
>
> **先走一遍** $m=2,n=1$，终点是 $(2,1)$。
>
> 路线甲：北、东、东。
>
> 1. 从 $(0,0)$ 北到 $(0,1)$，代价 $1+\max(0,0)=1$。
> 2. 从 $(0,1)$ 东到 $(1,1)$，代价 $1+0=1$。
> 3. 从 $(1,1)$ 东到 $(2,1)$，代价 $1+1=2$。
>
> 总和 $1+1+2=4$。
>
> 路线乙：东、东、北。
>
> 1. 从 $(0,0)$ 东到 $(1,0)$，代价 $1$。
> 2. 从 $(1,0)$ 东到 $(2,0)$，代价 $2$。
> 3. 从 $(2,0)$ 北到 $(2,1)$，代价 $1+\max(2,0)=3$。
>
> 总和 $1+2+3=6$，更贵。贵在最后一步北走时 $x$ 已经是 $2$。
>
> **一般结论**。不绕路时必须恰好 $m$ 次东、$n$ 次北。那 $m$ 次东必然分别从 $x=0,1,\ldots,m-1$ 出发，这部分代价固定为 $\sum_{x=0}^{m-1}(1+x)=m(m+1)/2$。北走的代价取决于当时的 $x$，越早北走越便宜，最便宜是全部在 $x=0$ 时北走，每步只付 $1$。故
>
> $$
> C^{\ast}=n+\frac{m(m+1)}{2}.
> $$
>
> 西或南只增加步数，代价又不会变负，不能更优。最优路径只有一条：先 $n$ 次北，再 $m$ 次东。
>
> **要交的答案**：上面的式子、一条最优路径、写明唯一。

<br>

> [!EXAMPLE]+ HW3.1b UCS 判断
>
> **先记住一条不变量**。一致代价搜索（uniform cost search, UCS）按已走路程 $g$ 从小到大弹出。边权非负时，一个点**第一次弹出**，$g$ 就是从起点到它的最小代价。终点第一次弹出，算法就停，后面还有多少没看过的格子都无所谓。
>
> **对本网格（三句）**
>
> 1. “格子无限，所以 UCS 永不停止”：**假**。终点 $(m,n)$ 第一次弹出就停。无限只表示外面还有格子，不表示算法走不到终点。
> 2. “只会看矩形 $0\le x\le m,\ 0\le y\le n$ 里面”：**假**。弹出只保证 $g\le C^{\ast}$，不保证坐标不越界。取 $m=2,n=0$，最优是东东，代价 $1+2=3$。先往西走到 $(-1,0)$，这一步代价 $1+\max(0,0)=1<3$，该点会被弹出。
> 3. “只会弹出 $g\le C^{\ast}$ 的点”：**真**。终点弹出时 $g=C^{\ast}$，此前弹出的点 $g$ 更小或相等。
>
> **对任意图（再三句，与上面网格无关）**
>
> 1. 图上加一条边，任意两点最短路代价只会变短或不变。**真**。多一条路不会让原路变贵。
> 2. 把某个动作代价改到足够小（甚至改成负数），UCS 返回的路径就一定含这个动作。**假**。两层原因：UCS 的正确性要求边权非负，改成负数后弹出即最优不再成立；这条边也可能根本不在任何从 $s$ 到 $t$ 的路上。
> 3. 每条边代价都加 $1$，最小代价那条**路径**不变。**假**。步数多的原最优路径会多挨几次 $+1$，步数少的次优路径可能反超。例如原路 $A$ 三步各 $1$（总 $3$），路 $B$ 一步代价 $4$；各加 $1$ 后变成 $6$ 对 $5$，$B$ 反而更短。
>
> **要交的答案**：六句真 / 假，各配一句理由。

<br>

> [!NOTE]+ HW3.2 到 HW3.4 建模
>
> **普通最短路（HW3.2）**。状态就是当前地点。`isEnd(s)` 当该地点带有 `endTag`（任意一家餐馆，而不是某一栋楼）。动作是走到相邻命名地点，代价是那段路的米数。先在纸上写出 `startState` / `isEnd` / `actionSuccessorsAndCosts` 三件套，再写代码。
>
> **无序途经点（HW3.3）**。除了人在哪，还要记住哪些标签还没满足。状态写成 $(\mathrm{loc},S)$，$S$ 是尚未满足的途经标签子集。到达某地时，用该地标签把 $S$ 里对应的位清掉。终点：地点带 `endTag` **并且** $S=\emptyset$。同一地点可以一次清掉多个标签。
>
> 最小例子：地点 $A,B,C$，要从 $A$ 到带 `end` 的 $C$，途经 `{food,gym}`。若 $B$ 同时有这两个标签，走到 $B$ 时 $S$ 一次清空。状态数上界是 $n\cdot 2^{k}$（$n$ 个地点，$k$ 个途经标签）。漏掉 $S$、只记地点，搜索会以为“路过一次就算完成”，路径会错。
>
> **A* 归约（HW3.4）**。不要重写搜索器。新代价
>
> $$
> \mathrm{Cost}'(s,a)=\mathrm{Cost}(s,a)+h(\mathrm{Succ}(s,a))-h(s),
> $$
>
> 再把这个新问题丢给现成 UCS。直线启发：到**任一**终点标签地点的欧氏距离下界，先预计算最近终点，评估时才快。无途经点启发：在原图上假装没有 $S$，跑一遍到终点标签的最短路，可采纳且通常更紧。
>
> **书面**。导航把车流赶到小路：用户自己堵在曾经的捷径上；非用户（居民、公交）承担噪声与路口占用。对策是改代价（容量、时间、避开居民区），不是只换启发式。途经点是双用途技术：可以插入休息点或卫生间，也可以被平台用来在上一单还没结束时接下单，让司机停不下来。

<br>

## Mountain Car

先把这条链画在纸上，再动手：

```
-2 ===== -1 ===== 0 ===== +1 ===== +2
终     非终    起点    非终     终
```

- 终点只有 $\pm 2$，**任何一轮**都有 $V_i(-2)=V_i(+2)=0$。
- 非终点动作只有两个。$a_1$：$80\%$ 向左一格、$20\%$ 向右一格。$a_2$：$70\%$ 向左、$30\%$ 向右。
- 回报看**走进去的那个格子**：$s'=-2$ 得 $10$，$s'=+2$ 得 $50$，进其它格子得 $-5$。
- $\gamma=1$，所以 $Q$ 就是“立即回报加后继的 $V$”的加权平均。$V$ 是两个 $Q$ 里较大的那个。
- 第 $i$ 轮算 $Q$ 时，后继必须用**上一轮**的 $V_{i-1}$，不能用本轮刚算出来的新值，也不能把上一轮的 $V_i(s)$ 直接抄到 $V_{i+1}(s)$。

> [!EXAMPLE]+ HW4.1 两轮值迭代
>
> **题在问什么**。从全零的 $V_0$ 做两轮值迭代，交出 $5+5=10$ 个数，再用 $V_2$ 读出三个非终点的贪心策略。
>
> **公式就这一行**（$\gamma=1$）：
>
> $$
> Q_i(s,a)=\sum_{s'}T(s'\mid s,a)\big[R(s,a,s')+V_{i-1}(s')\big],\qquad
> V_i(s)=\max_a Q_i(s,a).
> $$
>
> **第 1 轮**。$V_0$ 全是 $0$，所以 $Q$ 退化成“立即回报的期望”。
>
> - 在 $s=-1$：$a_1$ 有 $0.8$ 走进 $-2$（得 $10$），$0.2$ 走进 $0$（得 $-5$）。$Q(a_1)=0.8\cdot 10+0.2\cdot(-5)=7$。$a_2$ 同理 $0.7\cdot 10+0.3\cdot(-5)=5.5$。较大者 $V_1(-1)=7$。
> - 在 $s=0$：无论 $a_1$ 还是 $a_2$，都只能走进 $\pm 1$，回报都是 $-5$。$V_1(0)=-5$。
> - 在 $s=+1$：$a_1$ 有 $0.2$ 走进 $+2$（得 $50$），$Q(a_1)=0.8\cdot(-5)+0.2\cdot 50=6$。$a_2$ 有 $0.3$ 走进 $+2$，$Q(a_2)=0.7\cdot(-5)+0.3\cdot 50=11.5$。故 $V_1(+1)=11.5$。
>
> **第 2 轮**。现在后继不再是 $0$。走进终点时后继值仍是 $0$；走进非终点时要加上 $V_1$。
>
> - 在 $s=-1$：$a_1$ 仍有 $0.8$ 进 $-2$ 得 $10+0$；$0.2$ 进 $0$ 得 $-5+V_1(0)=-5+(-5)=-10$。$Q(a_1)=0.8\cdot 10+0.2\cdot(-10)=6$。$a_2$ 同理 $0.7\cdot 10+0.3\cdot(-10)=4$。故 $V_2(-1)=6$。
> - 在 $s=0$：向左走到 $-1$ 时带上 $V_1(-1)=7$，向右走到 $+1$ 时带上 $V_1(+1)=11.5$。
>
> $$
> Q(a_1)=0.8(-5+7)+0.2(-5+11.5)=0.8\cdot 2+0.2\cdot 6.5=1.6+1.3=2.9,
> $$
>
> $$
> Q(a_2)=0.7\cdot 2+0.3\cdot 6.5=1.4+1.95=3.35.
> $$
>
> 故 $V_2(0)=3.35$。
> - 在 $s=+1$：向左走进 $0$ 时带上 $V_1(0)=-5$，**不能**把 $V_1(+1)=11.5$ 再抄一遍。向右走进 $+2$ 时后继值是 $0$。
>
> $$
> Q(a_1)=0.8(-5-5)+0.2\cdot 50=-8+10=6,
> $$
>
> $$
> Q(a_2)=0.7(-5-5)+0.3\cdot 50=-7+15=8.
> $$
>
> 故 $V_2(+1)=8$。
>
> **由 $V_2$ 读策略**：谁的 $Q$ 大选谁。$\pi(-1)=a_1$（$6>4$），$\pi(0)=a_2$（$3.35>2.9$），$\pi(+1)=a_2$（$8>6$）。右侧更愿选“更常向右”的 $a_2$，去换那 $50$ 分。
>
> **要交的答案**（十个数加三个动作）：
>
> $$
> V_1=(-2,-1,0,+1,+2)=(0,\ 7,\ -5,\ 11.5,\ 0),
> $$
>
> $$
> V_2=(0,\ 6,\ 3.35,\ 8,\ 0),\qquad
> \pi(-1)=a_1,\ \pi(0)=a_2,\ \pi(+1)=a_2.
> $$

<br>

> [!INFO]+ HW4.2 把 $\gamma<1$ 化成 $\gamma'=1$
>
> **题在问什么**。原来每走一步，未来值要乘折扣 $\gamma<1$。手里的求解器只会解 $\gamma'=1$ 的 MDP。需要造一个新 MDP，让它在原状态上的最优值与原来相同。
>
> **想法**。$\gamma<1$ 可以读成：每一步有概率 $1-\gamma$ “游戏被强制结束”。结束走到一个新吸收点 $o$，$V'(o)=0$，之后不再有回报。
>
> **新转移与新回报**（$s,s'$ 都是原状态）：
>
> $$
> T'(s'\mid s,a)=\gamma\,T(s'\mid s,a),\qquad T'(o\mid s,a)=1-\gamma,
> $$
>
> $$
> R'(s,a,s')=\frac{R(s,a,s')}{\gamma},\qquad R'(s,a,o)=0.
> $$
>
> $o$ 上停住：$T'(o\mid o,\cdot)=1$、$R'(o,\cdot,o)=0$。
>
> **为什么 $R$ 要除以 $\gamma$**。新方程里走到 $s'$ 的概率已经带了一个 $\gamma$。若回报不除回来，一步回报会被额外乘一次 $\gamma$，值会整体错一层。代入：
>
> $$
> V'(s)=\max_a\sum_{s'}\gamma T\Big(\frac{R}{\gamma}+V'(s')\Big)
> =\max_a\sum_{s'}T\big[R+\gamma V'(s')\big].
> $$
>
> 右端正是原来的贝尔曼方程，所以在原状态上 $V'=V$。
>
> **要交的答案**：$T'$、$R'$、以及上面这三行验证。

<br>

> [!NOTE]+ HW4.3 到 HW4.5 实现
>
> **为何不能对连续小车直接值迭代**。连续状态有无穷个格子，$V(s)$ 存不下。先离散化，再用环境的 `.step` 做蒙特卡洛，估 $\hat T$ 与 $\hat R$，每隔若干步跑一次值迭代（`ModelBasedMonteCarlo`）。覆盖不足时，估出来的 $T$ 是另一盘棋，值迭代会在错模型上算到“最优”。
>
> **表格 Q 学习**。更新就这一行：
>
> $$
> Q(s,a)\leftarrow Q(s,a)+\alpha\big(r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big).
> $$
>
> 动作用 $\varepsilon$-贪心：以 $\varepsilon$ 随机走，否则走当前 $Q$ 最大的动作。
>
> **函数逼近**。不再给每个格子存一个 $Q$，而是 $Q(s,a)=w\cdot\phi(s,a)$。本作业 $\phi$ 是傅里叶特征：对缩放后的坐标做所有不超过 $m$ 的整数组合，再取 $\cos(\pi\sum m_i d_i s_i)$。更新的是 $w$，不是格子。二维 Mountain Car 格子少，表格通常更稳；状态很高维、或探索时间不够时，函数逼近能把附近格子的经验借过来。
>
> **安全探索**。`max_speed` 在物理积分**之后**把速度截断，智能体看到的已经是截断后的动力学。若最优轨迹本来就不超速，去掉截断曲线看起来差不多。动作层约束则不同：下一步速度不得过阈值，没有合法动作就返回 `None`，可行集变了，学到的策略会真的不同。
>
> 书面两套 MDP。A：状态含行人位置，动作含“不管红灯”，回报只看是否尽快到达，探索完全随机，可能撞人。B：把碰撞写成大负回报，`Actions(s)` 直接去掉会撞人的动作，探索只在安全动作里 $\varepsilon$-贪心。

<br>

## Pac-Man

一层深度不是“走一步”，而是“Pac-Man 走一次，然后每个幽灵各走一次”。$d$ 只在**最后一个**幽灵走完后减一。终点用 `getScore`，深度用尽还没结束则用 `evaluationFunction`。

> [!INFO]+ HW5.1 minimax 递推
>
> **题在问什么**。写出 $V(s,d)$。$a_0$ 是 Pac-Man（取最大），$a_1,\ldots,a_n$ 是幽灵（取最小）。先判断棋局是否结束，再判断深度是否用尽，最后才分支。
>
> **先看 $1$ 个幽灵、$d=1$**。树只有两层：Pac-Man 选一步，幽灵再选一步，然后 $d$ 变成 $0$，调用估值。Pac-Man 会选“幽灵随后能把它压到的值”里最大的那个。这就是普通的一层 minimax。
>
> **两个幽灵时**。顺序是 Pac-Man、幽灵 1、幽灵 2。幽灵 1 走完 $d$ 还不变；幽灵 2 走完才 $d\leftarrow d-1$。漏掉这一点会少折一层或早停。
>
> $$
> V(s,d)=
> \begin{cases}
> \mathrm{Utility}(s) & \mathrm{IsEnd}(s)\\
> \mathrm{Eval}(s) & d=0\\
> \max_{a}V(\mathrm{Succ}(s,a),d) & \mathrm{Player}(s)=a_0\\
> \min_{a}V(\mathrm{Succ}(s,a),d) & \mathrm{Player}(s)=a_i,\ 1\le i<n\\
> \min_{a}V(\mathrm{Succ}(s,a),d-1) & \mathrm{Player}(s)=a_n.
> \end{cases}
> $$
>
> 实现用 `agentIndex`：`0` 取 max，其余取 min，下标超过 $n$ 后回到 $0$ 并且 $d\leftarrow d-1$。一层深度对应树高 $n+1$。
>
> **对拍根值**（写错递推时这组数对不上）：
>
> | 图 | $d=1$ | $d=2$ | $d=3$ | $d=4$ |
> | :--- | :--- | :--- | :--- | :--- |
> | `minimaxClassic` | $9$ | $8$ | $7$ | $-492$ |
> | `mediumClassic` | $9$ | $18$ | $27$ | $36$ |
>
> α-β 必须得到同一组值，只是少展开一些节点。
>
> **要交的答案**：上面的分段式；编程对上这组根值。

<br>

> [!INFO]+ HW5.3 expectimax
>
> **题在问什么**。幽灵不再联合使坏，而是在合法动作里均匀随机。Pac-Man 仍然取最大；幽灵层改成期望。
>
> **先看一个数**。幽灵有两个合法动作，后继值是 $10$ 与 $-4$。minimax 会取 $-4$；expectimax 取 $(10-4)/2=3$。Pac-Man 若还有另一条稳得 $0$ 的路，minimax 会选 $0$，expectimax 会赌期望 $3$ 的那条。
>
> 把 HW5.1 里每个 $\min_a$ 换成
>
> $$
> \frac{1}{|\mathrm{Actions}(s)|}\sum_{a}V(\mathrm{Succ}(s,a),\ d\text{ 或 }d-1).
> $$
>
> $d$ 仍只在最后一个幽灵走完后减一。随机幽灵不会专门堵门，智能体会赌逃生吃豆。`trappedClassic` 深度 $3$：失败大约 $-502$，成功大约 $531$ 或 $532$（并列打破规则不同会差 $1$ 分）。估值函数必须与 `getScore` 同一尺度，否则深度用尽时的数与终局不可比。
>
> **要交的答案**：期望式；编程对上 `trappedClassic` 的根值符号与量级。

<br>

## Scheduling

> [!EXAMPLE]+ HW6.1 电灯 CSP 与 XOR 链
>
> **电灯在问什么**。$n$ 盏灯开始全灭，$m$ 个按钮。按钮 $j$ 控制集合 $T_j$ 里的灯，按下就翻转。同一盏灯被按奇数次才亮。要造一个 CSP：$m$ 个变量、$n$ 条约束。
>
> 变量 $B_j\in\{0,1\}$：按钮 $j$ 按不按。灯 $i$ 的约束：控制它的那些按钮里，按下的个数是奇数。
>
> $$
> \bigoplus_{j:\,i\in T_j}B_j=1.
> $$
>
> 一条约束最多依赖全部 $m$ 个按钮，所以允许 $m$ 元约束。
>
> **XOR 链：先列真值表**。三个变量 $X_1,X_2,X_3\in\{0,1\}$。因子 $t_1=X_1\oplus X_2$、$t_2=X_2\oplus X_3$，值为 $1$ 才合法。$\oplus$ 为 $1$ 当且仅当两边不同。
>
> | $X_1$ | $X_2$ | $X_3$ | $t_1$ | $t_2$ | 合法？ |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | $0$ | $0$ | $0$ | $0$ | $0$ | 否 |
> | $0$ | $0$ | $1$ | $0$ | $1$ | 否 |
> | $0$ | $1$ | $0$ | $1$ | $1$ | **是** |
> | $0$ | $1$ | $1$ | $1$ | $0$ | 否 |
> | $1$ | $0$ | $0$ | $1$ | $0$ | 否 |
> | $1$ | $0$ | $1$ | $1$ | $1$ | **是** |
> | $1$ | $1$ | $0$ | $0$ | $1$ | 否 |
> | $1$ | $1$ | $1$ | $0$ | $0$ | 否 |
>
> 一致赋值只有 $(0,1,0)$ 与 $(1,0,1)$。一句话：$X_1=X_3\ne X_2$。
>
> **回溯数的是“调用次数”**。只有新赋值与**已经完整出现**的因子不冲突，才继续递归。冲突的叶子不调用。要找全部解。
>
> 官方示例序 $X_1,X_2,X_3$ 会立刻触发 $t_1$，失败分支早砍，共 $7$ 次。本题要的是序 $X_3,X_1,X_2$。$t_1$ 与 $t_2$ 都含 $X_2$，所以赋完 $X_3$ 与 $X_1$ 时还没有任何因子完整，要等到赋 $X_2$ 才检查。
>
> 1. $\{[01],[01],[01]\}$ 根
> 2. $X_3=0$：$\{[01],[01],0\}$
> 3. $X_1=0$：$\{0,[01],0\}$
> 4. $X_2=1$：$\{0,1,0\}$ 合法。$X_2=0$ 使两因子为 $0$，**不**调用
> 5. $X_1=1$（仍 $X_3=0$）：$\{1,[01],0\}$。随后 $X_2$ 两值都不合法，不调用
> 6. $X_3=1$：$\{[01],[01],1\}$
> 7. $X_1=0$：$\{0,[01],1\}$。$X_2$ 两值都不合法
> 8. $X_1=1$：$\{1,[01],1\}$
> 9. $X_2=0$：$\{1,0,1\}$ 合法
>
> 共 **9** 次。次数随变量序变，因为因子完整出现的早晚不同。
>
> **加上 AC-3 会跳过哪几次**。赋 $X_3=0$ 后，$t_2$ 逼 $X_2=1$，再经 $t_1$ 逼 $X_1=0$，于是第 5 次（$X_3=0$ 且 $X_1=1$）根本不会被调用。赋 $X_3=1$ 后对称跳过第 7 次。共跳过 **2** 次。
>
> **$n$ 皇后（编程对拍）**。变量 $i$ 的域是第 $i$ 行的列号，大小为 $n$。二元约束：列不等、主对角不等、副对角不等。$n=8$ 必须恰好 $92$ 个解、朴素 $2057$ 次 `backtrack`；最约束变量（most constrained variable, MCV）后 $1361$ 次。域更大或约束更少，次数会对不上。
>
> **要交的答案**：电灯的变量与 XOR 约束、两个一致赋值、9 次调用列表、跳过 2 次的说明。

<br>

> [!NOTE]+ HW6.3 课表
>
> **先在纸上定变量**。常见取法：每门已请求的课对应一个变量，域是“某个已注册季度”或“不选”。`or` 请求把若干课绑成至多选一门。
>
> **四类约束怎么落**。
>
> 1. 每季学分：用 `create_sum_variable` 把该季各课学分配成一个和变量，夹在 `minUnits` 与 `maxUnits` 之间。
> 2. 先修：若请求写了 `B after A`，或公报里 $A$ 是 $B$ 的官方先修，则 $B$ 所在季度必须严格晚于 $A$。
> 3. 同一课至多出现在一个季度。
> 4. `in Aut2018,Sum2019` 把域直接裁成这些季度。
>
> 权重来自 `request` 的 `weight`，因子输出正实数，求解器找乘积最大的赋值。先在 `profile*.txt` 小档案上对拍（学分数、先修是否被违反），再跑完整公报。漏掉“不选”这个域值，会出现无解。
>
> **要交的答案**：`submission.py` 里通过小档案与隐藏用例；书面无需再抄公式。

<br>

## Car

先把故事钉死。自己的车位置 $a_t$ 已知。他车位置 $C$ 未知。测距 $D_t$ 是 $\|a_t-C\|$ 再加高斯噪声。作业要维护一张格子上的信念 $b(c)=P(C=c\mid\text{已见到的测距})$。

> [!INFO]+ HW7.1 发射
>
> **题在问什么**。他车静止。每来一个新测距 $d_t$，把旧信念乘上“若车在这个格子，测到 $d_t$ 有多像”，再除以总和。这叫发射更新（emission update）。
>
> **贝叶斯网**。$C$ 指向 $D_1,D_2,D_3$。$a_t$ 是已知参数，可以不画，或画成已观察节点。观察的是 $D$ 与 $a$，不观察 $C$。
>
> $$
> P(c,d_{1:3})=p(c)\prod_{t=1}^{3}p(d_t\mid c).
> $$
>
> **先用三个格子走一遍**。旧信念 $b=(0.2,\ 0.5,\ 0.3)$。新测距下三个格子的密度是 $(0.1,\ 0.8,\ 0.1)$。
>
> 1. 未归一化：逐格相乘 $(0.02,\ 0.40,\ 0.03)$。
> 2. 总和 $Z=0.45$。
> 3. 新信念 $(0.02,\ 0.40,\ 0.03)/0.45\approx(0.044,\ 0.889,\ 0.067)$。
>
> 中间那个格子“又常见、又像这次测距”，后验被抬起来。这就是热力图变亮的原因。
>
> **一般式**。$b_{t-1}(c)=P(C=c\mid d_{1:t-1})$，$p(d_t\mid c)=\mathrm{pdf}(\|a_t-c\|_{2},\sigma,d_t)$。
>
> $$
> b_t(c)\leftarrow b_{t-1}(c)\,p(d_t\mid c),\qquad
> b_t(c)\leftarrow b_t(c)\Big/\sum_{c'}b_t(c').
> $$
>
> `util.pdf` 是密度，可以大于 $1$，乘法仍然合法。$a_t$ 已知，不要把它放进求和变量。
>
> **要交的答案**：网的画法、联合式、上面两行更新。

<br>

> [!INFO]+ HW7.2 转移
>
> **题在问什么**。他车现在会动。信念先按转移概率摊开（时间流逝），再乘新测距（发射）。这是隐马尔可夫模型（hidden Markov model, HMM）的前向一步。
>
> **网**。链 $C_1\rightarrow C_2\rightarrow C_3$，每一时刻再有 $C_t\rightarrow D_t$。$a_t$ 仍是已知参数。
>
> $$
> P(c_{1:3},d_{1:3})=p(c_1)p(d_1\mid c_1)\prod_{t=2}^{3}p(c_t\mid c_{t-1})p(d_t\mid c_t).
> $$
>
> **先用三个格子走一遍**。旧信念 $b=(1,0,0)$，车确定在格子 $1$。转移：留在原地 $0.5$，向右一格 $0.5$（格子 $3$ 的向右当作留下）。
>
> $$
> b'(1)=1\cdot 0.5=0.5,\qquad
> b'(2)=1\cdot 0.5=0.5,\qquad
> b'(3)=0.
> $$
>
> 热力图从“一个亮点”变成“向可能开走的方向抹开”。若 $T$ 每一行加总为 $1$，则 $\sum b'=1$，这一步不必再除。
>
> **一般式**。对每个新格子 $c'$：
>
> $$
> b'(c')=\sum_{c}b(c)\,T(c\rightarrow c').
> $$
>
> 完整一轮顺序固定：先 `elapseTime`，再 `observe`。反了就会用“还没走”的位置去解释新测距。
>
> **多车**。彼此独立则各维护一张 $b$。测距若不能标明来自哪一辆，后验要在 $K$ 辆车的排列上求和，状态数乘 $K!$ 量级。
>
> **要交的答案**：网、联合式、`elapseTime` 的求和式、注明先推移再发射。

<br>

## Logic

> [!EXAMPLE]+ HW8.1 与 HW8.2 翻译
>
> **题在问什么**。把英语一句一句换成公式。命题逻辑用大写原子；一阶逻辑还要写 $\forall$ / $\exists$。代码里 `And` / `Or` 只收两个参数，三个以上用 `AndList` / `OrList`。变量写成 `'$x'`，谓词首字母大写，常量全小写。
>
> **命题：英语连接词对照**
>
> | 英语 | 公式 | 代码 |
> | :--- | :--- | :--- |
> | 夏天并且在加州，那么不下雨 | $\mathrm{Summer}\wedge\mathrm{CA}\rightarrow\neg\mathrm{Rain}$ | `Implies(And(Summer,CA), Not(Rain))` |
> | 湿当且仅当下雨或喷灌 | $\mathrm{Wet}\leftrightarrow(\mathrm{Rain}\vee\mathrm{Sprinkler})$ | `Equiv(Wet, Or(Rain,Sprinkler))` |
> | 白天或夜晚，但不能同时 | $\mathrm{Day}\oplus\mathrm{Night}$ | `And(Or(Day,Night), Not(And(Day,Night)))` |
>
> 第三句最容易写成普通 $\vee$。普通 $\vee$ 允许“又是白天又是夜晚”，grader 会给这个赋值当反模型。
>
> **一阶：逐词**
>
> - “每一个人有一个父母”：先限定人，再存在一个父母。$\forall x\,\mathrm{Person}(x)\rightarrow\exists y\,\mathrm{Parent}(y,x)$。题面写明不必强迫父母也是 Person。
> - “至少一个人没有子女”：存在一个人，并且不存在他的子女。$\exists x\,\mathrm{Person}(x)\wedge\neg\exists y\,\mathrm{Child}(y,x)$。
> - 父亲的定义：$\mathrm{Father}(x,y)\leftrightarrow\mathrm{Male}(x)\wedge\mathrm{Parent}(x,y)$。
> - 孙女的定义：$\mathrm{Granddaughter}(x,y)\leftrightarrow\mathrm{Female}(x)\wedge\exists z\,(\mathrm{Child}(x,z)\wedge\mathrm{Child}(z,y))$。中间那一代 $z$ 必须写出来。题面允许自己是自己的孩子。
>
> **量词位置会改意思**。$\forall x\exists y$ 是“每个人可以有自己的父母”；写成 $\exists y\forall x$ 变成“全世界共用同一个父母”。grader 给出的反模型就是这样不相等的一个赋值。
>
> **要交的答案**：`submission.py` 里七个公式；书面不必再抄，但上面的对照可用来对拍反模型。

<br>

> [!EXAMPLE]+ HW8.3 说谎者
>
> **题在问什么**。四个人：Mark、John、Nicole、Susan。每人说了一句话。已知恰好一人说真话，恰好一人撞了服务器。把六条事实写成公式，推理谁真、谁撞。
>
> **记号**。$T(X)$：这个人说真话。$C(X)$：这个人撞了服务器。
>
> **先把口语换成双条件**。说真话则这句话成立；说谎则这句话的否定成立。两者合在一起就是 $\leftrightarrow$。
>
> | 谁 | 原话 | 公式 |
> | :--- | :--- | :--- |
> | Mark | 不是我撞的 | $T(M)\leftrightarrow\neg C(M)$ |
> | John | 是 Nicole 撞的 | $T(J)\leftrightarrow C(N)$ |
> | Nicole | 不是我，是 Susan | $T(N)\leftrightarrow\big(\neg C(N)\wedge C(S)\big)$ |
> | Susan | Nicole 在说谎 | $T(S)\leftrightarrow\neg T(N)$ |
>
> 另外两条：恰好一个 $T$，恰好一个 $C$。六条按 `3a-0` 到 `3a-5` 分开对拍。
>
> **纸上穷举：假设唯一真话者是谁**。
>
> 1. 只有 Mark 真。他的话成立 $\Rightarrow\neg C(M)$。John 假 $\Rightarrow$ 不是 Nicole 撞的 $\Rightarrow\neg C(N)$。Susan 假 $\Rightarrow$ “Nicole 说谎”为假 $\Rightarrow T(N)$。已经有 Mark 与 Nicole 两个真话者，与“恰好一人真”矛盾。
> 2. 只有 John 真。他的话成立 $\Rightarrow C(N)$。Mark 假 $\Rightarrow$ “不是我”为假 $\Rightarrow C(M)$。Mark 与 Nicole 都撞了，与“恰好一人撞”矛盾。
> 3. 只有 Nicole 真。她的话成立 $\Rightarrow\neg C(N)$ 且 $C(S)$。Mark 假 $\Rightarrow C(M)$。又是两人撞机，矛盾。
> 4. 只有 Susan 真。她的话只要 $\neg T(N)$，已经满足。Mark 假 $\Rightarrow C(M)$。John 假 $\Rightarrow\neg C(N)$。Nicole 假：在 $\neg C(N)$ 已经成立时，整句要假只能让 $C(S)$ 也假。于是撞机的只有 Mark。四个条件全部相容。
>
> **要交的答案**：六条公式；跑 `3a-run` 应得到 Susan 说真话、Mark 撞服务器。纸上的穷举用来理解，不必整段贴进 PDF。

<br>

> [!NOTE]+ HW8.4 奇偶与可解释性
>
> **题在问什么**。用六条约束描述“后继、奇偶、更大”，查询已写好：每个数都存在一个比它大的偶数。`Larger` 只是一个谓词名字，**不要**偷用“大于”的日常性质（例如自己不能比自己大），题面没写就不能用。
>
> | 约束 | 公式 | 一句话 |
> | :--- | :--- | :--- |
> | 恰好一个后继，且不是自己 | $\forall x\,\exists! y\,\mathrm{Succ}(x,y)\wedge y\ne x$ | 每个 $x$ 指向别人 |
> | 奇偶恰好一个 | $\forall x\,(\mathrm{Odd}(x)\oplus\mathrm{Even}(x))$ | 不能两空也不能两真 |
> | 偶的后继是奇 | $\forall x\,\mathrm{Even}(x)\rightarrow\mathrm{Odd}(\mathrm{succ}(x))$ | |
> | 奇的后继是偶 | $\forall x\,\mathrm{Odd}(x)\rightarrow\mathrm{Even}(\mathrm{succ}(x))$ | |
> | 后继比自己大 | $\forall x\,\mathrm{Larger}(\mathrm{succ}(x),x)$ | |
> | 更大可传递 | $\forall x,y,z\,\mathrm{Larger}(x,y)\wedge\mathrm{Larger}(y,z)\rightarrow\mathrm{Larger}(x,z)$ | |
>
> 查询（已给，不要改）：$\forall x\,\exists y\,\mathrm{Even}(y)\wedge\mathrm{Larger}(y,x)$。
>
> **可解释性书面**。系统除了 Yes / No，还会吐出一条消解树。那是推导步骤，不是“因为 Nicole 人品不好”这种因果故事。知识库写错，解释会认真地为错模型辩护。
>
> **可靠与完备**。可靠（sound）：推出来的都是真的。完备（complete）：真的都能推出来。
>
> - 大模型会幻觉，推出假句，不可靠；也推不出大量本该成立的句，不完备。
> - 安全关键场景（假结论会伤人）选可靠但不一定完备的系统。
> - 旁边另有验证器时，可选完备但不可靠的系统，先多捕候选再过滤。
>
> **要交的答案**：六条公式按 `4a-0` 到 `4a-5` 对拍；书面写清消解树是什么、以及可靠 / 完备的选择。

<br>

## Autumn 增补

2025 Autumn 用两份作业替换排课与车辆跟踪。2026 Spring 不交这两份，社会题拆进各周书面伦理。

> [!NOTE]+ Aut HW6 系统发生树
>
> 树节点是物种或基因，边是条件概率表（conditional probability table, CPT）。`forward_sampling` 必须按拓扑序：先采样没有父节点的变量，再采样子节点。联合概率是一路上 CPT 取值的乘积。拒绝采样：先按先验采样整棵树，观测对不上就扔掉，留下的频率估后验。MLE / EM 在隐藏祖先状态时出现。卡住时先在 $3$ 个节点的树上手算一个样本。

<br>

> [!NOTE]+ Aut HW8 Society
>
> 按一个具体产品写：数据从哪来、对用户是否透明、训练数据有没有版权与劳动问题、分组指标在哪一组掉下去。PDF 里禁止嵌超链接，URL 必须粘贴成可见文本。2026 Spring 把同类问题拆进贷款、导航、毒性、自动驾驶等书面小节，不再单开这一份。

<br>

> [!NOTE]+ 衔接
>
> 作业按应用组织。下一章把同一张表对回考试：问的是损失梯度还是弹出不变量，而不是“情感”或“小车”。

<br>

# 实践

## 概念：建议

- **问题**
  - 作业按应用组织，考试按模型族组织，二者必须能互译；
  - 状态表示与因子设计比调步长更容易失分。
- **范围**
  - 给出选题、对拍与常见失败；
  - 输入题面，输出模型族与要报告的量。

| 作业 | 模型族 | 推断 | 必须报告 |
| :--- | :--- | :--- | :--- |
| Foundations | 预备 | 手算 + 短函数 | 导数、期望、大 O |
| Sentiment | 反射 | SGD | 训练 / 验证零一、特征维 |
| Route | 搜索 | UCS / A* | 代价、展开数、$h$ 是否一致 |
| Mountain Car | MDP / RL | Q 学习 | 回报曲线、$\varepsilon$、$\alpha$ |
| Pac-Man | 博弈 | minimax / expectimax | 根值、展开叶子、节点类型 |
| Scheduling | CSP / 因子图 | 回溯、束搜索 | 是否可行、$K$、违反约束 |
| Car tracking | 贝叶斯网 | 粒子滤波 | 后验 / ESS、粒子数 |
| Logic | 逻辑 | 翻译与检验 | 公式、小论域反模型 |
| Society | 评价 | 分组指标 | 伤害、各组样本量与分数 |

对拍顺序：先手写 $2$ 到 $4$ 个状态的例子，再跑实现，最后上作业图。搜索先对拍最优代价，再对拍展开数。博弈先对拍无剪枝的根值，再开 α-β 对拍叶子集合。粒子滤波在小 $|Z|$ 上对拍前向。逻辑先造反模型。

五类最常见实现失败：状态漏字段、启发式不可采纳、Q 学习不探索、CPT 未归一化、量词范围写错。优先队列与粒子权的并列处理必须与自动评分一致。

> [!WARNING]+ 项目失败模式
>
> 选题把四条轴叠在一次实验里，中期仍无对拍基线，用测试调 $h$ 或 $\varepsilon$，不固定 seed。范围宜先固定模型族与评价指标，再换启发或网络。把语言模型当万能推断，再在报告里写“已经 minimax”，属于模型族选错，不是调参失败。

<br>

## 选题

可选项目或 AI Product Deep Dive 若存在，先写三句话：任务、模型族、要报告的量。能落在单一轴上再做。路径规划不要同时上 Q 学习与 α-β。需要语言接口时，把翻译与符号推断分开评。

> [!NOTE]+ 衔接
>
> 协议写清之后，复习只保留各轴的定义、递推与正确性条件。

<br>


# 复习

闭卷通常覆盖：损失与 SGD、FutureCost 与 UCS / A* 的条件、贝尔曼与 Q 学习、minimax 与 α-β、因子与 VE、d-分离、消解。编程覆盖可哈希状态、优先队列键、α-β 窗口、粒子权归一化。

## 公式

- 反射：分数 $s=w\cdot\phi(x)$。铰链 $\max(0,1-ys)$，在 $ys<1$ 时 $\nabla_w=-y\phi(x)$。逻辑损失梯度 $-\frac{y}{1+e^{ys}}\phi$。SGD 加 $L_2$ 时多一项 $\lambda w$。
- 搜索：$\mathrm{FutureCost}(s)=\min_a[\mathrm{Cost}(s,a)+\mathrm{FutureCost}(\mathrm{Succ})]$。UCS 键 $g$，非负边弹出即最优。A* 键 $g+h$。一致：$h(s)\le\mathrm{Cost}(s,a)+h(\mathrm{Succ})$，蕴含可采纳。
- MDP：$V_{\mathrm{opt}}(s)=\max_a\sum_{s'}T(s,a,s')[R+\gamma V_{\mathrm{opt}}(s')]$。$Q$ 学习：$Q\leftarrow(1-\alpha)Q+\alpha(r+\gamma\max_{a'}Q(s',a'))$。
- 博弈：max / min / chance 三层。α-β：$\alpha$ 下界、$\beta$ 上界，$\alpha\ge\beta$ 剪枝，不改变根值。
- 因子：$w(x)=\prod_j f_j$。VE 一次步 $f_{\mathrm{new}}(x_S)=\sum_{x_i}\prod_{f\in F_i}f$。代价 $|\mathcal{D}|^{w+1}$。
- 贝叶斯网：$P(x)=\prod_i P(x_i\mid x_{\mathrm{Pa}(i)})$。对撞观察后开通。HMM 前向 $\alpha_t(z)=P(x_t\mid z)\sum_{z'}\alpha_{t-1}(z')T(z'\rightarrow z)$。
- 逻辑：$\mathrm{KB}\models\phi$ 当且仅当 $\mathrm{KB}\wedge\neg\phi$ 不可满足。消解 $(C\vee P),(D\vee\neg P)\vdash C\vee D$。$\forall x\exists y$ 的 Skolem 是 $f(x)$。

## 选算法

| 题面 | 适用 | 易错项 |
| :--- | :--- | :--- |
| 一步输出标签 | 反射 + SGD | UCS |
| 确定后继、最小代价路径 | UCS / A* | 值迭代（除非要整张表） |
| 随机后继、要策略 | 值迭代或 Q 学习 | 一条预先路径 |
| 对手对抗 | minimax + α-β | expectimax |
| 对手随机已知 | expectimax | α-β |
| 固定变量、零一约束 | CSP 回溯 + AC-3 | FutureCost |
| 要 $Z$ 或边缘 | VE | 只跑束搜索当精确 |
| 时间链、大 $|Z|$ | 粒子滤波 | 整段轨迹上 VE |
| 是否必然 | 消解 / 模型检测 | 语言模型似然 |

## 正确性条件

- UCS / 一致 A*：边权 $\ge 0$；弹出时标记。
- 可采纳不足以让图 A* 禁止再打开；要弹出即最优须一致。
- 值迭代：$\gamma<1$ 或几乎必然吸收。
- Q 学习：每个 $(s,a)$ 无穷次访问，表格，$\alpha$ 衰减。
- α-β：只对纯 minimax；窗口沿递归传。
- AC-3 非空定义域 $\not\Rightarrow$ 可满足。
- VE 顺序差则中间因子爆炸，值仍对（若算得完）。
- 粒子权必须归一化；似然全零要重启。
- 消解必须先加入 $\neg\phi$。

> [!NOTE]+ 三条失败轴
>
> - 拟合或求解失败：模型族选错、状态漏字段、启发式不一致、探索为 $0$、CPT 未归一。
> - 评价失败：用展开结点数当唯一分数、用训练回报选 $\varepsilon$、把边缘当成 MAP。
> - 部署失败：分组指标未报、语言流畅代替蕴含、跟踪系统的误报代价未写。

<br>


# 参考文献

## 课程

1. Stanford CS221. *Artificial Intelligence: Principles and Techniques*，2026 Spring。[https://stanford-cs221.github.io/spring2026/](https://stanford-cs221.github.io/spring2026/)
2. Stanford CS221. *Autumn 2025*。[https://stanford-cs221.github.io/autumn2025/](https://stanford-cs221.github.io/autumn2025/)
3. Stanford CS221. *Spring 2025* 模块。[https://stanford-cs221.github.io/spring2025/modules/](https://stanford-cs221.github.io/spring2025/modules/)
4. Stanford Explore Courses. *CS 221*。

## 讲义与教材

5. Percy Liang 等. CS221 模块讲义（反射、搜索、MDP、博弈、因子图、贝叶斯网、逻辑）。
6. Russell S, Norvig P. *Artificial Intelligence: A Modern Approach*。（综合参考；记号与课程不完全相同）
7. Sutton R, Barto A. *Reinforcement Learning: An Introduction*。（MDP 与强化学习）
8. Koller D, Friedman N. *Probabilistic Graphical Models*。（因子图与贝叶斯网；CS228 教材）
9. Tsang E. *Foundations of Constraint Satisfaction*。（CSP）

## 工具

- 作业自动评分对结点计数与浮点都严格，先在小图上对拍 UCS / A* / α-β。
- NumPy 用于 SGD 与粒子滤波；搜索与博弈用哈希状态加堆。
- 逻辑作业先手写小模型再交给求解器，量词范围错误无法靠求解器发现。
