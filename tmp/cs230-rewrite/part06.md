# 课堂扩展

Coursera 五门课之外，CS230 课堂还讲了四个专题：

1. **对抗与生成**：前面的训练都是对权重 $W$ 求梯度；这里转而对输入 $x$ 求梯度，既可以构造让模型出错的输入，也可以生成新数据。
2. **强化学习**：没有固定的标签 $y$，数据由智能体与环境交互产生，监督信号是奖励。
3. **LLM 应用**：骨干已经预训练好，工作重心变成如何构造输入（提示、检索、工具）以及如何评估。
4. **内部分析**：检查模型依据什么做判断，用来决定下一步的实验。

<br>

## 对抗与生成（Adversarial and Generative Models）

对抗样本在输入上加很小的扰动，分类器就会以高置信度出错。生成模型学习 $P(x)$ 或 $P(x\mid y)$，用来采样，而不是只输出标签。二者都对 $x$ 求梯度，不再只更新 $W$。内容来自 2025 Autumn Lecture 4；Coursera C1–C5 没有对应编程作业。

- 需要能写出 FGSM / PGD 与对抗训练，并能对上 VAE、GAN、扩散三条谱系。
- 强化学习进一步去掉固定标签 $y$，改用回报。

### 对抗样本

> [!INFO]+ FGSM
>
> FGSM（Goodfellow et al., 2015）沿 $\mathrm{sign}(\nabla_x J)$ 对输入走一小步，得到扰动很小、分类却错的 $x_{\mathrm{adv}}$。PGD 是同一方向上的多步投影。对抗训练把 $x_{\mathrm{adv}}$ 加入训练集。只报告 FGSM 准确率会高估稳健性。

<br>

#### 现象

> [!EXAMPLE]+ 例1 熊猫与长臂猿
>
> Goodfellow 等的经典例子：一张被 GoogLeNet 以 $57.7\%$ 置信度识别为熊猫的图像，加上 $\varepsilon=0.007$ 倍的符号梯度噪声后，人眼看不出任何区别，模型却以 $99.3\%$ 的置信度判为长臂猿。

对抗样本（adversarial example）对安全敏感的系统是现实威胁：人脸门禁、自动驾驶的路标识别、内容审核都可能被精心构造的输入欺骗。

#### 为什么高维线性模型脆弱

线性分类器 $w^{\top}x$ 在高维中对沿 $w$ 方向的小扰动敏感。取扰动 $\eta=\varepsilon\,\mathrm{sign}(w)$，每个分量的绝对值都不超过 $\varepsilon$（$\|\eta\|_\infty=\varepsilon$），但

$$
w^{\top}(x+\eta)=w^{\top}x+\varepsilon\|w\|_1.
$$

若 $w$ 有 $n$ 维、每维平均大小为 $m$，输出改变约 $\varepsilon mn$：**每个像素只改一点点，但 $n$ 个像素的改变同方向累加**，维度越高，越容易跨过决策面。神经网络中大量使用 ReLU 这类分段线性的单元，局部表现得很像线性模型，所以同样脆弱。

#### FGSM 与 PGD

把 $w$ 换成损失对输入的梯度，就得到快速梯度符号法（fast gradient sign method, FGSM）：

$$
x_{\mathrm{adv}}=x+\varepsilon\,\mathrm{sign}(\nabla_x J(\theta,x,y)).
$$

$\varepsilon$ 很小（图像上常对应每个像素几级灰度）即可跨过决策面。注意这里固定参数 $\theta$，对**输入**求梯度并沿**上升**方向走，与训练时沿下降方向更新 $\theta$ 正好相反。迭代 FGSM / PGD（projected gradient descent）在 $\ell_{\infty}$ 球内多步投影：

$$
x^{k+1}=\Pi_{\mathcal{B}\_{\varepsilon}(x)}\big(x^{k}+\alpha\,\mathrm{sign}(\nabla_x J)\big).
$$

投影 $\Pi$ 把每步结果裁回 $[x-\varepsilon,x+\varepsilon]$ 内，保证扰动始终足够小。

- 定向攻击（targeted attack）：想让模型判成指定类别 $y_{\mathrm{target}}$ 时，改为沿 $-\nabla_x J(\theta,x,y_{\mathrm{target}})$ 下降。
- 白盒攻击能拿到梯度；黑盒攻击拿不到时，可以在替代模型上构造对抗样本，它们往往能迁移到目标模型。

```python
def fgsm(x, grad_x, eps):
    """grad_x = dJ/dx，与 x 同形。像素若在 [0,1] 则裁剪。"""
    x_adv = x + eps * np.sign(grad_x)
    return np.clip(x_adv, 0.0, 1.0)
```

#### 防御

- 对抗训练：把 $x_{\mathrm{adv}}$ 加入训练集，最小化最坏扰动下的损失，是 min-max：

$$
\min_{\theta}\ \mathbb{E}_{(x,y)}\Big[\max_{\|\eta\|_\infty\le\varepsilon}J(\theta,x+\eta,y)\Big].
$$

- 代价：训练变慢；对攻击范数外的扰动不一定迁移。
- 防御不能只靠把输入模糊一下：自适应攻击会把防御编进梯度。
- 评估必须用强攻击（PGD、AutoAttack），不能只报 FGSM 准确率。

### 生成模型

> [!INFO]+ VAE、GAN、扩散
>
> 生成模型学习 $P(x)$，用来采样，不是只输出标签。VAE（2014）最大化 $\log p(x)$ 的 ELBO，重参数化让采样可微。GAN（2014）让生成器与判别器对打，采样快，训练不稳，似然不好算。扩散模型先加噪再学去噪，训练稳，采样步数多。课堂要能对上谱系，不必推完 SDE。

<br>

#### 监督与无监督

之前的问题都是**监督学习**（supervised learning）：数据集以“输入 + 标签”成对给出。生成模型只拿到大量样本（比如一堆卧室照片），没有标签，要学的是数据本身的分布 $P(x)$，这属于**无监督学习**（unsupervised learning）。学会之后，从 $P(x)$ 中采样就得到训练集中不存在的新样本。

#### 四条谱系

- **自回归**：$P(x)=\prod_t P(x_t\mid x_{<t})$，像素 RNN、GPT 属此类。似然可精确计算，采样是串行的（前一部分的语言模型就是这一类）。
- **VAE**：编码器 $q(z\mid x)$，解码器 $p(x\mid z)$，最大化 ELBO（evidence lower bound）
  $$
  \mathbb{E}\_{q}[\log p(x\mid z)]-\mathrm{KL}(q(z\mid x)\|p(z)).
  $$
  重参数化 $z=\mu+\sigma\odot\varepsilon$，$\varepsilon\sim\mathcal{N}(0,I)$，使采样路径可微。
- **GAN**：生成器 $G(z)$ 与判别器 $D(x)$。原目标
  $$
  \min_G\max_D \mathbb{E}\_{x}[\log D(x)]+\mathbb{E}\_{z}[\log(1-D(G(z)))].
  $$
  训练不稳定：模式崩溃、梯度消失。WGAN 等用不同距离缓解。似然不可直接得。
- **扩散**：逐步加噪到高斯，再学逐步去噪。训练稳定，采样步数多。课堂以直觉与应用为主，不要求推导全部 SDE。

- 条件生成（text-to-image、语音 vocoder）把 $y$ 注入归一化层或注意力。
- 评估用 FID、IS、人工偏好，不要只用训练损失。

#### VAE

> [!INFO]+ VAE 的 ELBO
>
> - $\log p(x)=\log\int p(x\mid z)p(z)\,\mathrm{d}z$。
> - 引入 $q(z\mid x)$ 后
>
> $$
> \log p(x)=\mathrm{KL}(q\|p(z\mid x))+\mathbb{E}\_{q}[\log p(x\mid z)]-\mathrm{KL}(q(z\mid x)\|p(z)).
> $$
>
> - KL 非负，故 ELBO 是 $\log p(x)$ 的下界。
> - 最大化 ELBO = 重构 − 正则。写成要最小化的损失，就是 $\mathbb{E}\_{q}[-\log p(x\mid z)]+\beta\,\mathrm{KL}(q(z\mid x)\|p(z))$，$\beta=1$ 时与 ELBO 只差一个符号。定义、不对称性与另外三种用法见 [信息论基础 · 相对熵](/信息论熵/#kl)。
> - $q=\mathcal{N}(\mu(x),\mathrm{diag}(\sigma^{2}(x)))$、$p(z)=\mathcal{N}(0,I)$ 时，KL 有闭式
>
> $$
> \frac{1}{2}\sum_j\big(\mu_j^{2}+\sigma_j^{2}-\log\sigma_j^{2}-1\big).
> $$
>
> - 重参数化使 $\partial/\partial\mu$、$\partial/\partial\sigma$ 能穿过采样。
> - KL 过强：后验坍缩到先验，解码器忽略 $z$（posterior collapse）。

<br>

两项的含义：重构项要求从 $z$ 解码回来的样本像原图；KL 项要求各样本的 $q(z\mid x)$ 都接近标准正态，使隐空间连续、规整。训练好后，从 $\mathcal{N}(0,I)$ 采样 $z$ 送入解码器即可生成新样本。

```python
def vae_elbo(x, mu, logvar, decoder_logpx, kl_weight=1.0):
    """返回要最大化的 ELBO；decoder_logpx 是 log p(x|z) 的像素和。"""
    recon = decoder_logpx.mean()
    kl = 0.5 * np.mean(np.sum(mu ** 2 + np.exp(logvar) - logvar - 1.0, axis=1))
    return recon - kl_weight * kl

def vae_loss(x, mu, logvar, decoder_logpx, kl_weight=1.0):
    """优化器最小化负 ELBO。"""
    return -vae_elbo(x, mu, logvar, decoder_logpx, kl_weight)

def reparameterize(mu, logvar, rng):
    eps = rng.normal(size=mu.shape)
    return mu + np.exp(0.5 * logvar) * eps

def pgd_linf(x, loss_grad_fn, eps, alpha, steps):
    """loss_grad_fn(x_adv) 返回 dJ/dx。"""
    x_adv = x.copy()
    for _ in range(steps):
        g = loss_grad_fn(x_adv)
        x_adv = np.clip(x_adv + alpha * np.sign(g), x - eps, x + eps)
        x_adv = np.clip(x_adv, 0.0, 1.0)
    return x_adv
```

#### GAN

GAN（generative adversarial network）由两个网络组成：

- **生成器** $G$（generator）：输入随机噪声 $z$，输出一张假图 $G(z)$，目标是以假乱真；
- **判别器** $D$（discriminator）：输入一张图，输出它是真图的概率，目标是分辨真假。

两者以竞争的方式学习：生成器学到越来越精妙的“作假技术”，判别器成长为越来越精准的“鉴定师”，切磋到最后，生成器可能画出足以乱真的图像。

> [!EXAMPLE]+ 例2 DCGAN 生成卧室（鱼书 8.5.2 节）
>
> DCGAN（Deep Convolutional GAN）用大量卧室照片训练后，可以从零生成新的卧室图像。这些图像看上去像真照片，但训练集中并不存在，是谁都没有见过的新图。DCGAN 的生成器和判别器都是卷积网络：生成器用转置卷积把噪声向量逐步放大成图像，判别器是普通的分类 CNN。

**最优判别器**：固定 $G$，对每个 $x$ 最大化 $p_{\mathrm{data}}(x)\log D(x)+p_g(x)\log(1-D(x))$。对 $D$ 求导令其为零，得

$$
D^{\ast}(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}.
$$

代回目标，$G$ 最小化的是 $-\log 4+2\,\mathrm{JS}(p_{\mathrm{data}}\|p_g)$。JS 散度非负，当且仅当 $p_g=p_{\mathrm{data}}$ 时取到全局最小值 $-\log 4$，此时 $D^{\ast}\equiv\frac{1}{2}$：判别器完全分不出真假。

**非饱和损失**：训练初期生成器很差，$D(G(z))\approx 0$，此时 $\log(1-D(G(z)))$ 在 $0$ 附近非常平坦，梯度几乎为零，生成器学不动。改为最大化 $\log D(G(z))$（等价于最小化 $-\log D(G(z))$），它在 $D(G(z))\approx0$ 处梯度很大，最优点不变，但早期学得快。

- GAN 训练交替：固定 $G$ 升 $D$，固定 $D$ 降 $\log(1-D(G(z)))$，或升 $\log D(G(z))$（非饱和启发式）。
- 判别器过强：生成器梯度消失。
- 判别器过弱：$G$ 收不到有用信号。
- 模式崩溃（mode collapse）：生成器只覆盖数据分布里的少数几种模式，输出几乎一样的图；判别器在这些点上仍可能被骗，多样性丢失。
- WGAN-GP 用 1-Lipschitz 约束替代 JS 散度，训练更稳，不是作业必实现。

#### 扩散模型

- 扩散前向：$q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)$。
- 闭式：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$，其中 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s\le t}\alpha_s$。任意 $t$ 的加噪样本可以一步得到，训练时随机选 $t$ 即可。
- 训练预测 $\varepsilon$ 或 $x_0$；采样从 $x_T\sim\mathcal{N}(0,I)$ 逐步去噪。常用损失为 $\mathbb{E}_{x_0,\varepsilon,t}\big\|\varepsilon-\varepsilon_\theta(x_t,t)\big\|_2^{2}$。
- 课堂要求掌握噪声预测的直觉，以及似然可算但采样步多、GAN 采样快但不稳的对比。

#### 生成评估

> [!NOTE]+ 生成评估（generative evaluation）
>
> Fréchet Inception Distance（FID）把真实图像与生成图像映射到 Inception 特征，分别拟合高斯，再比较均值与协方差：
>
> $$
> \mathrm{FID}
> =\|\mu_r-\mu_g\|_2^{2}
> +\operatorname{tr}\left(
> \Sigma_r+\Sigma_g
> -2(\Sigma_r\Sigma_g)^{1/2}\right).
> $$
>
> FID 越低越好，同时反映样本质量与多样性，但依赖特征提取器、样本数和预处理协议。
>
> Inception Score（IS）看单张图的类别分布是否尖锐，以及全体样本的边缘类别分布是否多样：
>
> $$
> \mathrm{IS}
> =\exp\left(
> \mathbb{E}\_{x}
> D_{\mathrm{KL}}\big(p(y\mid x)\|p(y)\big)\right).
> $$
>
> IS 越高越好，但不直接与真实数据比较，也会受 Inception 类别空间限制。两者都不能代替人工偏好与任务指标。

> [!NOTE]+ 对照：对 $x$ 的两种梯度
>
> - FGSM：沿 $\nabla_x J$ 上升一小步，制造错分类。
> - 生成器：沿判别器损失下降，制造更像真的 $x$。
> - 二者都对输入空间求梯度，不再只更新 $W$。
> - 评估协议必须与攻击或生成过程分开写清，否则数字不可比。

<br>

> [!WARNING]+ 评估
>
> - 只报 FGSM 准确率会高估稳健性。
> - 必须用 PGD 或 AutoAttack 等强攻击。
> - 生成模型不要只用训练损失；FID、IS 与人工偏好正交。

<br>

> [!NOTE]+ 衔接
>
> 对抗与生成仍假设标签或判别信号存在。强化学习去掉固定的 $(x,y)$：环境给奖励，网络近似价值 $Q$ 或策略 $\pi$。策略章里的偏差、方差和分布不匹配，在这里分别变成函数近似偏差、回报方差，以及仿真环境与真实环境不一致。

<br>

## 强化学习（Reinforcement Learning）

> [!INFO]+ DQN 与 PPO
>
> 强化学习没有固定的 $(x,y)$，数据由当前策略与环境交互产生。DQN（2015）用网络拟合 $Q$，经验回放降低样本相关，目标网络稳定 TD 目标。PPO 直接优化策略，并用裁剪限制一次更新的幅度。探索不足时学习曲线长期为零，不一定是实现写错。

<br>

### 基本框架

人学骑自行车不是靠老师逐个动作纠正，而是在不断尝试中自己摸索。让计算机在试错中自主学习，就是**强化学习**（reinforcement learning）。它与监督学习的区别在于：监督学习的每个输入都有“老师”给出的正确答案；强化学习只有事后的、往往延迟的奖励。

基本框架：**智能体**（agent）根据**环境**（environment）的状态选择**动作**（action），动作改变环境，智能体获得**奖励**（reward）。目标是学到一个行动方针（策略），使得到的奖励总和最大。

奖励往往不是即时明确的。在《超级马里奥兄弟》中，向右走一步能得多少奖励并不确定，要从游戏得分（硬币、消灭的敌人）或游戏结束这类明确指标反推出“预期奖励”。这也是强化学习比监督学习难的原因：一个动作的好坏可能要很多步之后才显现（信用分配问题，credit assignment）。

强化学习里没有固定的 $(x,y)$。智能体与环境交互，观察到状态 $s$、采取动作 $a$、得到奖励 $r$，目标是最大化期望回报 $G_t=\sum_{k=0}^{\infty}\gamma^{k} r_{t+k}$。深度网络用来近似策略 $\pi(a\mid s)$ 或价值 $Q(s,a)$。对应 2025 Autumn Lecture 5。

### 回报与价值

**折扣因子** $\gamma\in[0,1)$ 让越远的奖励越不重要：一方面未来不确定，另一方面保证无限长的和收敛。若每步奖励都是 1，$G=\sum_k\gamma^{k}=\frac{1}{1-\gamma}$，$\gamma=0.99$ 时约为 100，可以理解为智能体大约“看” 100 步远。

> [!EXAMPLE]+ 例3 折扣回报
>
> 取 $\gamma=0.9$。
>
> - 奖励序列 $(0,0,1)$：$G_0=0+0.9\cdot0+0.81\cdot1=0.81$。同样的奖励晚两步拿到，价值打了八一折。
> - 奖励序列 $(1,1,1,10)$：$G_0=1+0.9+0.81+0.729\cdot10=10$。
> - 一个立刻拿到 $+2$ 的选择，与一个 3 步后拿到 $+10$ 的选择相比，后者价值 $0.9^{3}\cdot 10=7.29>2$；若 $\gamma=0.5$，后者只值 $1.25<2$。$\gamma$ 决定智能体有多“远视”。

- MDP（Markov decision process）：$(S,A,P,R,\gamma)$。马尔可夫性指下一状态只依赖当前状态和动作，与更早的历史无关。
- 价值函数（value function）$V^{\pi}(s)=\mathbb{E}\_{\pi}[G_t\mid s_t=s]$：在状态 $s$ 按策略 $\pi$ 行动的期望回报。
- $Q^{\pi}(s,a)=\mathbb{E}\_{\pi}[G_t\mid s_t=s,a_t=a]$：在 $s$ 先做动作 $a$，之后按 $\pi$ 行动的期望回报。
- Bellman：回报可以拆成“这一步的奖励 + 折扣后的下一状态价值”，

$$
V^{\pi}(s)=\mathbb{E}\_{a\sim\pi,s'}[r+\gamma V^{\pi}(s')].
$$

- 最优 $Q^{\ast}$ 满足 $Q^{\ast}(s,a)=\mathbb{E}[r+\gamma\max_{a'}Q^{\ast}(s',a')]$。知道 $Q^{\ast}$ 后，最优策略就是每个状态选 $\arg\max_a Q^{\ast}(s,a)$。

### DQN

#### 从 Q 表到 Q 网络

状态和动作都很少时，可以用一张表存所有 $Q(s,a)$，用 Bellman 方程反复更新（Q 学习）。但游戏画面的状态数是天文数字，表存不下，于是用神经网络近似：输入状态 $s$，输出每个动作的 $Q$ 值。

- Deep Q-Network：网络 $Q(s,a;\theta)$ 回归 TD 目标。
- 时序差分目标（temporal-difference target）与 TD error 为

$$
y_t=r_t+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a';\theta^{-}),
\qquad
\delta_t=y_t-Q(s_t,a_t;\theta).
$$

$d_t=1$ 表示终止，后续价值必须置零。DQN 最小化 $\delta_t^{2}$ 或 Huber loss。目标网络参数 $\theta^{-}$ 暂时固定，避免预测值与目标同时快速移动。这相当于一个回归问题，只是“标签” $y_t$ 由网络自己估计出来。

- 目标网络（target network） $\theta^{-}$ 缓慢更新。TD 目标里用下一状态的 $Q$ 估计当前 $Q$，称为 bootstrapping；两边若用同一套正在更新的权重，目标会跟着漂。
- 经验回放（experience replay）：把轨迹存进缓冲区再随机抽样，打散相邻状态之间的相关，也提高样本利用率。
- 动作离散时可行；连续动作常用策略梯度（policy gradient）。

> [!EXAMPLE]+ 例4 DQN 玩 Atari（鱼书 8.5.4 节）
>
> 以往让计算机学玩游戏，要先把游戏状态（角色位置等）人工提取出来交给模型。DQN 的输入**只有游戏画面**：把连续 4 帧画面（灰度化、缩小到 $84\times84$，叠成 $84\times84\times4$，4 帧可以体现运动方向）送入 CNN，输出游戏手柄每个动作（摇杆移动、按钮按下与否）的 $Q$ 值。
>
> 因此不必针对每个游戏改设置：同一个网络结构学会了《吃豆人》等许多 Atari 游戏，并在很多游戏中超过了人类水平。

DQN 算法要点：

1. 用 $\varepsilon$-greedy 从 $Q(s,\cdot;\theta)$ 选动作（以概率 $\varepsilon$ 随机探索，否则选 $Q$ 最大的动作），与环境交互，把 $(s,a,r,s',\mathrm{done})$ 写入 replay buffer。
2. 均匀采样一批转移。
3. 目标 $y=r+\gamma\max_{a'}Q(s',a';\theta^{-})$（终止状态无后续项）。
4. 对 $\theta$ 做 MSE 或 Huber 回归。
5. 每隔 $C$ 步 $\theta^{-}\leftarrow\theta$。

- 没有 replay：样本强相关。
- 没有目标网络：$y$ 与 $Q$ 同时移动。
- 二者都是发散的常见原因。
- 连续动作不能对 $a$ 做 $\max$，改用 DDPG/PPO 等策略方法。

```python
def discounted_returns(rewards, gamma=0.99):
    """rewards: 一条轨迹上的 r_0,...,r_{T-1}。"""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return np.array(out[::-1])

def reinforce_loss(log_probs, returns):
    """最大化期望回报等价于最小化 -log pi * G。"""
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return -np.sum(log_probs * returns)

def dqn_td_target(r, done, q_next, gamma=0.99):
    """r, done, q_next: (batch,)。目标网络给出 q_next = max_a Q(s',a; theta-)。"""
    return r + gamma * (1.0 - done) * q_next
```

### 策略梯度

DQN 先学价值再导出策略；策略梯度直接参数化策略 $\pi_\theta(a\mid s)$（例如网络输出动作上的 softmax），沿提高期望回报的方向更新 $\theta$。

REINFORCE：

$$
\nabla_{\theta}J=\mathbb{E}\_{\pi_{\theta}}\big[\nabla_{\theta}\log\pi_{\theta}(a\mid s)\,G_t\big].
$$

直观含义：回报高的轨迹，提高其中各动作的概率；回报低的，降低其概率。形式上就像把 $G_t$ 当作样本权重的交叉熵。

- 轨迹概率 $P(\tau)=\rho(s_0)\prod_t \pi(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$。
- $\nabla\log P(\tau)=\sum_t\nabla\log\pi(a_t\mid s_t)$；转移 $P$ 不出现在梯度里。
- 因此可以在未知模型上学习（model-free）。
- 基线 $b(s)$（如 $V(s)$）不改变无偏性，但降低方差，形成 actor-critic。
- 优势 $\hat{A}_t=G_t-V(s_t)$ 或 GAE 进一步降方差：衡量这个动作比该状态下的平均表现好多少。
- PPO 用裁剪的概率比限制一次更新幅度：

$$
L^{\mathrm{CLIP}}=\mathbb{E}\big[\min\big(\rho\hat{A},\,\mathrm{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat{A}\big)\big],
$$

$$
\rho=\frac{\pi_{\theta}(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}.
$$

- PPO 的 clip 把 $\rho$ 限制在 $[1-\varepsilon,1+\varepsilon]$，防止一次更新毁掉策略。
- 另一种限制是在目标里加 $\beta D_{\mathrm{KL}}(\pi_{\theta}\|\pi_{\mathrm{ref}})$，使新策略靠近参考策略（reference policy）。RLHF 与 KL 正则强化学习（KL-regularized RL）用的是这一项。
- $\varepsilon$ 常取 $0.1$-$0.2$。

### 探索与奖励设计

- 探索：$\varepsilon$-greedy、熵奖励、噪声网络。
- 奖励稀疏（只有终局 $+1$）时，先塑造稠密奖励或用课程学习。
- 曲线长期为零：不能只归因于算法写错，也可能是没探索到。
- 仿真到真实：像素 vs 状态向量、延迟、摩擦都会造成不匹配。
- 先在状态可完整观测的玩具环境验证实现，再上像素输入。
- 对应策略章：先保证能拟合训练。

> [!WARNING]+ 奖励设计
>
> - 智能体会利用奖励漏洞（reward hacking）。
> - 稀疏奖励使探索困难。
> - 函数近似引入偏差；蒙特卡洛回报方差大。
> - 仿真到真实是分布不匹配。
> - 项目若做 RL：先在可重置、有明确成功条件的环境验证学习曲线，再谈迁移。

<br>

> [!EXAMPLE]+ 例5 AlphaGo
>
> AlphaGo 击败围棋世界冠军，内部同时用到深度学习和强化学习：先学习约 3000 万个职业棋手的棋谱局面（监督学习，得到初始策略），再不停地自我对弈积累经验（强化学习），并用价值网络评估局面、配合蒙特卡洛树搜索选点。AlphaGo 与 DQN 都出自 Google DeepMind。

### RLHF

> [!NOTE]+ RLHF 流程（reinforcement learning from human feedback）
>
> 2025 Autumn Lecture 5 把语言模型训练分成以下阶段。
>
> 1. 下一词元预测（next-token prediction）：预训练语言模型（pretrained language model）最大化语料的对数似然。
> 2. 监督微调（supervised fine-tuning, SFT）：使用人工示范（human-written demonstrations）学习有帮助的回答格式。
> 3. 偏好数据（preference data）：对同一提示的多个回答做人类排序（human ranking）。
> 4. 奖励模型（reward model, RM）：学习标量分数 $r_{\phi}(x,y)$，使偏好回答得分更高。常见成对偏好损失（pairwise preference loss）为
>
> $$
> \mathcal{L}\_{\mathrm{RM}}
> =-\log\sigma\big(r_{\phi}(x,y_{\mathrm{chosen}})
> -r_{\phi}(x,y_{\mathrm{rejected}})\big).
> $$
>
> 5. 强化学习微调（reinforcement-learning fine-tuning）：策略生成回答，奖励模型打分；PPO 或相关策略方法提高奖励，同时用 KL penalty 限制策略离开参考模型。
>
> $$
> \max_{\theta}
> \mathbb{E}\_{y\sim\pi_{\theta}(\cdot\mid x)}
> \left[r_{\phi}(x,y)
> -\beta\log\frac{\pi_{\theta}(y\mid x)}
> {\pi_{\mathrm{ref}}(y\mid x)}\right].
> $$
>
> PPO 的旧策略（old policy）$\pi_{\theta_{\mathrm{old}}}$ 用于构造裁剪概率比；RLHF 的参考策略（reference policy）$\pi_{\mathrm{ref}}$ 用于 KL 约束。二者角色不同，不能混成同一个分布。
>
> 奖励黑客（reward hacking）指策略利用奖励模型漏洞得到高分，却没有真正提高人类偏好。KL 太弱会漂移，太强则几乎不离开 SFT 模型。

在 RLHF 中，“状态”是提示加已生成的词，“动作”是下一个词，“奖励”只在整段回答结束时由奖励模型给出：这正是上文说的稀疏、延迟奖励。

> [!NOTE]+ 对照：监督、对抗、RL
>
> - 监督：$\nabla_\theta J(\theta,x,y)$，标签固定。
> - 对抗：$\nabla_x J$，标签仍在，更新的是输入。
> - RL：$\nabla_\theta\log\pi\cdot G$，标签换成回报，数据由策略自己产生。
> - 策略自己产生数据，因此探索失败与分布漂移比监督学习更早出现。

<br>

> [!NOTE]+ 衔接
>
> 强化学习的数据由当前策略与环境交互产生。LLM 应用层相反：骨干已经预训练，调整的是提示、检索与工具，也就是条件 $x$ 怎么拼。评估仍按任务拆开，不能只报一个笼统的主观质量分数。

<br>

## LLM 应用（LLM Applications）

### 应用层

骨干已经是预训练好的 $P(y\mid x)$ 时，产品效果往往取决于提示、检索、工具和评估，而不是再训一个网络。对应 2025 Autumn Lecture 8。

- 预训练模型是条件分布 $P(y\mid x)$。
- 应用层控制 $x$ 的构造：$x=$ 系统指令 + 检索片段 + 对话历史 + 用户问题。
- 上下文窗口（context window）有限，必须做选择与压缩。

可调的手段按代价从低到高排列：

1. **零样本提示**（zero-shot）：只写指令；
2. **少样本提示**（few-shot）：在上下文中放几个示范；
3. **检索增强**（RAG）：把相关文档放进上下文；
4. **工具调用**：让模型调用搜索、计算器、数据库；
5. **微调**（LoRA 等）：更新权重。

选择哪一种，应由误差分析决定，而不是默认从最贵的开始。

### 提示

- 少样本提示：把 $k$ 个输入-输出示范放进上下文，不更新权重。
- 示范必须与测试分布同格式，否则学到的是表面模板。
- 温度、top-$p$ 属于解码超参数，应在 dev 上选，并在报告中固定。

### RAG

> [!INFO]+ RAG 与 LoRA
>
> RAG（Lewis et al., 2020）先检索 $k$ 段文档，再在这些证据上生成，不更新骨干权重。检索库可以更新事实；检索失败时生成仍可能流畅地编造，因此召回率要单独测。LoRA（2021）在线性层上加低秩 $\Delta W=BA$，适合改风格和格式；事实过时时应改检索库。

<br>

模型的知识截止于训练数据，私有文档也不在其中，直接问容易编造（幻觉）。RAG 把“记住事实”的负担交给可更新的检索库。

- RAG（retrieval-augmented generation）：检索器返回 $k$ 段文档，生成器只在这些证据上回答。
- 检索失败时生成器仍会流畅地编造。
- 必须单独测检索召回，不能只看最终 BLEU 或人工分。

RAG 流水线可以写成与模型无关的步骤：

1. 文档切块：按标题或固定 token 数，块过大则噪声多，过小则上下文碎。
2. 嵌入并建索引；查询时对问题向量取 $k$ 近邻（与词嵌入一节的余弦相似度相同）。
3. 组装提示：系统规则 + 引用块（带编号）+ 用户问题 + “仅依据编号证据回答”。
4. 生成；解析引用是否指向真实块号。
5. 分别记录检索召回（金标准块是否进入 top-$k$）与生成正确率。

```python
def retrieve(query_vec, doc_vecs, k=5):
    """doc_vecs: (n_doc, d)，已 L2 归一化时内积即余弦。"""
    scores = doc_vecs @ query_vec
    idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

def build_rag_prompt(question, chunks, idx):
    evidence = "\n".join(f"[{i}] {chunks[j]}" for i, j in enumerate(idx, 1))
    return (
        "只根据下列证据回答。若证据不足，明确写出不知道。\n"
        f"{evidence}\n问题：{question}\n回答："
    )
```

### 工具与安全

- 工具调用把输出约束为可解析动作（搜索、计算器、数据库）：模型输出 JSON 或特定标记，程序执行后再把结果追加进上下文。
- 失败模式：字段缺失、类型错误、重复调用、对不可用工具幻想。
- 应用层用 schema 校验，而不是再提示“请一定输出合法 JSON”。
- 提示注入（prompt injection）：不可信文档里的“忽略以上指令”应被视为数据。
- 系统消息与工具权限隔离。
- 解析失败、权限过宽与注入攻击（指令藏在检索文档里）是系统风险，不能靠再调一点解码温度解决。

### LoRA

- LoRA（low-rank adaptation）：线性层上加低秩 $\Delta W=BA$，$B\in\mathbb{R}^{d\times r}$，$A\in\mathbb{R}^{r\times k}$，$r\ll \min(d,k)$。原权重 $W$ 冻结，前向变为 $h=(W+BA)x$。
- 可训参数从 $dk$ 降到 $r(d+k)$，远小于全量微调，适合项目算力。例如 $d=k=4096$、$r=8$ 时，从约 1678 万降到约 6.6 万，约为 $0.4\%$。
- 仅当误差表显示风格或格式占主导、且有足够私有数据时才值得做。
- 误差来自事实过时：更新检索库比 LoRA 便宜。

### 评估

- 评估应分任务：闭卷事实、开卷引用、数学、代码、拒答。
- 只报一个笼统的主观质量分数，无法判断下一步该改检索、提示还是工具。
- 评估协议先写再测。
- 闭卷 QA：exact match / F1。
- 开卷：检查引用是否支撑答案。
- 代码：单测通过率。
- 安全：红队集。
- 不要用同一聊天记录既调提示又报最终分（这等于在测试集上调参）。
- 项目若做 LLM 应用，基线应包括：无检索零样本（zero-shot）、少样本（few-shot）、简单 RAG。
- 误差分析标注：检索错 / 证据够但推理错 / 格式错 / 拒答不当。这与项目策略中“对出错样本分类计数”的误差分析方法相同。

> [!NOTE]+ 对照：改权重还是改上下文
>
> - 零样本 / 少样本 / RAG：不更新 $W$，改 $x$。
> - LoRA：更新低秩 $\Delta W$，适合风格与格式。
> - 事实过时优先改检索库，不是再训骨干。
> - 误差表决定下一步改检索、提示还是微调，对应策略章的正交化。

<br>

> [!WARNING]+ 注入与评估
>
> - 检索文档是数据，不是指令。
> - 工具权限过宽与解析失败属于系统风险。
> - 检索召回与生成正确率必须分开记。

<br>

> [!NOTE]+ 衔接
>
> 应用层调整的是条件 $x$。内部分析检查模型依据哪些特征做判断，以及某个子群是否被总体准确率掩盖。这些量用来决定下一实验，而不是写完报告再补说明。

<br>

## 内部分析

### 目的

可解释性要回答网络依据哪些特征做判断，以及失败能否定位到某一层、某类样本。工具包括学习曲线、切片指标、显著性和 Grad-CAM。这些量用来决定下一实验，不是写完报告再补的说明。对应 Lecture 10。

可用工具：

- **学习曲线**：训练与 dev 损失、指标同图，区分欠拟合、过拟合、优化发散。
- **混淆矩阵（confusion matrix）与切片**：按光照、语言、设备切片，避免总体准确率掩盖子群失败。
- **显著性 / Grad-CAM**：$\partial y_c/\partial x$ 或特征图层的加权，检查模型是否盯着物体而非水印。
- 显著图可被输入变换愚弄，只能当假设生成器。
- **最近邻嵌入**：把验证样本嵌到表示空间，看错例是否与另一类混叠。
- **消融（ablation）**：去掉增强、正则或某一数据源，量化贡献。

- 注意力权重、某神经元的最大化图都不是因果证明。
- 可解释性服务于误差分析与安全审查：决定下一周改数据还是改模型。

### 显著性与遮挡

- **显著图**（saliency map）：类别分数 $y_c$（softmax 之前的分数）对输入像素的梯度 $\left|\partial y_c/\partial x\right|$，绝对值大的像素表示“稍微改动就会显著影响该类分数”，可以粗略分割出物体。
- **遮挡敏感度**（occlusion sensitivity）：用灰色方块依次遮住图像的各个位置，记录每次的类别概率，画成热力图。遮住某处概率骤降，说明模型依赖该区域。不需要梯度，但要做很多次前向。
- **类别模型可视化**（class model visualization）：固定权重，从零图开始对输入做梯度上升 $\arg\max_x\, y_c(x)-\lambda\|x\|_2^{2}$，得到“模型心目中最像第 $c$ 类”的图像。正则项防止图像变成无意义的高频噪声。
- **中间层可视化**：对某个卷积通道，找出让它激活最大的训练图像块，或用反卷积（deconvolution）把激活投影回像素空间。卷积一节中“第 1 层看边缘、第 3 层看纹理、第 5 层看物体部件”的结论就来自这类研究。

### Grad-CAM

> [!INFO]+ Grad-CAM
>
> Grad-CAM（Selvaraju et al., 2017）用 $\partial y_c/\partial A^{k}$ 对卷积特征图加权，得到该类别的热力图。它只提供假设：模型可能依据哪些区域。热力图对输入变换不稳定，需要再用遮挡或切片核对。

<br>

最后一个卷积层的特征图既保留了空间位置，又已经包含高级语义。Grad-CAM 对卷积特征图 $A^{k}$ 与类别分数 $y_c$：

$$
\alpha_k=\frac{1}{Z}\sum_{i,j}\frac{\partial y_c}{\partial A^{k}\_{ij}},\qquad
L_{\mathrm{CAM}}=\mathrm{ReLU}\Big(\sum_k \alpha_k A^{k}\Big).
$$

$\alpha_k$ 是梯度在空间上的全局平均（$Z$ 为空间位置数），表示第 $k$ 个通道对类别 $c$ 有多重要。

- 热力图上采样到原图尺寸。
- ReLU 丢掉对类别有负贡献的区域。
- 热力图落在角落水印或床头文字而不是病灶：模型可能在抄捷径（shortcut）。
- 显著性图对输入变换不稳定，只能提出假设。
- 再用切片指标或遮挡实验验证：把高显著区域涂掉，若 $y_c$ 不变，则解释不可信。

### 切片与学习曲线

```python
def confusion_and_slices(y_true, y_pred, slice_ids):
    """slice_ids 如设备型号。返回总体混淆与各切片准确率。"""
    k = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = {}
    for s in np.unique(slice_ids):
        m = slice_ids == s
        acc[str(s)] = float((y_true[m] == y_pred[m]).mean()) if m.any() else None
    return cm, acc
```

学习曲线诊断：

- 训练损失不降：学习率、bug、标签错、初始化爆炸。
- 训练损失降、训练指标差：损失与指标不对齐（例如用 Acc 看交叉熵任务的阈值问题）。
- 训练好、dev 差且同分布：方差。
- 训练-训练 dev 好、真实 dev 差：不匹配。
- 损失出现 NaN：溢出、未裁剪、学习率过大、除零。

- 内部分析服务于误差分析：决定下一周改数据还是改模型。
- 期中若问“下一步做什么”，答案必须引用上述某一种曲线或误差表，而不是报层名。

> [!NOTE]+ 对照：解释与因果
>
> - Grad-CAM 与显著性只生成假设。
> - 遮挡、切片、消融才能检验假设。
> - 对抗章已经说明：输入梯度可被利用，也可被解释图愚弄。
> - 期中问“下一步做什么”时，必须引用曲线或误差表，而不是报层名。

<br>

> [!NOTE]+ 衔接
>
> 内部分析给出可引用的曲线与误差表。下一章先把 Coursera 编程作业对回公式，再进入自选项目。Coursera 作业数据集不能直接充当项目。

<br>

