---
title: AWS GPU 从零上手：EC2 与 SageMaker
date: 2026-11-15
categories:
  - 学CS/SE
tags:
  - AWS
  - GPU
  - SSH
  - Linux
desc: 做 CS230 课程项目时遇到的。一份 AWS GPU 官方教程的中英双语精读版。从账号、额度、配额讲到 g4dn.xlarge 是什么、AMI、EBS、安全组、SSH 密钥、uv、tmux、scp、Jupyter 隧道和停机计费，每个名词都有解释。后半部分记录 EC2 配额一直是 0 之后改走 SageMaker Studio 的全过程，包括共享空间、SSH 远程连接和 30 分钟自动关机；以及配额批下来后搬回 EC2、全组每人一把密钥共用一台机器的做法（公钥名单、安全组、IP 变化、共用规矩）。
---

本文主要参考：

- 一份 AWS Setup Instructions 教程（EC2 + g4dn.xlarge + neural-style 示例）
- <a href="https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html">AWS account plans and promotional credits</a>
- <a href="https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html">Amazon EC2 On-Demand quotas</a>
- <a href="https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html">EC2 accelerated computing specifications</a>
- <a href="https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html">AWS Deep Learning Base GPU AMI for Ubuntu 24.04</a>
- <a href="https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-to-linux-instance.html">AWS SSH and file-transfer instructions</a>
- <a href="https://docs.astral.sh/uv/getting-started/installation/">uv installation</a>，<a href="https://docs.astral.sh/uv/concepts/projects/sync/">uv locking and syncing</a>
- <a href="https://www.tensorflow.org/install/pip">TensorFlow installation and GPU troubleshooting</a>
- <a href="https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter-connect.html">AWS Jupyter SSH tunnel instructions</a>
- <a href="https://github.com/tmux/tmux/wiki/Getting-Started">tmux getting started</a>
- <a href="https://docs.aws.amazon.com/sagemaker/latest/dg/remote-access.html">SageMaker AI：Connect your local IDE to SageMaker spaces</a>

相关旧笔记：[Remote-SSH窗口关闭/网络断开解决方法](../remotessh持续实验)（nohup / tmux 细节），[rsync](../rsync)，[AutoDL使用教程](../autodl)。

【阅读方式：每一节先写中文，节末折叠卡片里是同一节的完整英文版，点开即可。命令只写一遍，中英共用。】

<br>

## 0 先看全局：这一套东西在干嘛

项目要训练神经网络，自己笔记本没有像样的 GPU，所以去 AWS 租一台带 GPU 的云服务器。整个流程就是：

1. 有一个 AWS 账号，有抵扣金（credits）的话先兑换。
2. 申请 GPU 配额（quota）。新账号默认能开的 GPU 数量是 0，不申请根本开不了。
3. 在 EC2 里用一个预装了显卡驱动的系统镜像（AMI）启动一台 g4dn.xlarge。
4. 用 SSH 密钥从自己电脑登录上去，装 Python 环境，确认程序真的能用上 GPU。
5. 训练用 tmux 挂着跑，断网也不怕；文件用 scp 来回拷；想用 Jupyter 就走 SSH 隧道。
6. **用完一定要停机**，否则一直按小时扣钱。

教程用的例子是 neural-style（风格迁移），跑通它就说明整条链路没问题，之后换成自己的项目代码即可。

> [!INFO]+ 云服务器 / Cloud server
> 就是别人机房里的一台电脑，你通过网络远程使用，按使用时长付钱。AWS（Amazon Web Services）是亚马逊的云平台，EC2 是其中"租虚拟机"的那项服务。
> A computer in someone else's data center that you use over the network and pay for by the hour. AWS is Amazon's cloud platform; EC2 is its service for renting virtual machines.

> [!NOTE]- English version
> **0 The big picture.** The project trains neural networks, and a laptop has no usable GPU, so we rent a GPU server on AWS. The whole flow is:
> 1. Have an AWS account, and redeem any credits you have.
> 2. Request a GPU quota. A new account may run zero GPU instances by default, so without a request you cannot start one at all.
> 3. In EC2, launch a g4dn.xlarge from a machine image (AMI) that already contains the NVIDIA driver.
> 4. Log in from your computer with an SSH key, install the Python environment, and confirm that the program really uses the GPU.
> 5. Run training inside tmux so a dropped connection does not kill it; copy files with scp; reach Jupyter through an SSH tunnel.
> 6. **Always stop the instance when you are done**, otherwise it keeps charging by the hour.
>
> The tutorial example is neural-style (style transfer). If it runs end to end, the setup works, and you can swap in your own project code.

<br>

## 1 账号、抵扣金和预算

### 1.1 账号与 credits

- 注册或登录 <a href="https://aws.amazon.com/">AWS 账号</a>。
- 如果学校或公司发 GPU credits（抵扣金），按他们的说明申请。
- **兑换前先看账号计划（account plan）**：免费计划（Free plan）不接受其他促销抵扣金。如果拿到的是促销码，账号要切到付费计划（Paid plan）才能兑换。
- 在 **Billing and Cost Management → Credits** 里兑换，并看清余额、过期时间、覆盖哪些服务。
- 定期看账单，设一个预算提醒。

> [!WARNING]+ credits 不等于全免费
> 抵扣金用完或过期之后照常扣钱；不在覆盖范围内的服务也照常扣钱。**预算提醒（budget alert）只会发邮件，不会帮你关机。**

> [!INFO]+ Credits / 抵扣金
> AWS 账户里的一笔"代金券"余额，账单先从这里扣。它有有效期和适用服务范围，两者都要在 Credits 页面确认。
> A voucher balance in the AWS account; bills are paid from it first. It has an expiry date and a list of covered services, and both are shown on the Credits page.

> [!INFO]+ Budget / 预算提醒
> AWS Budgets 里设一个金额阈值（比如 50 美元），花费接近或超过时给你发邮件。它只是报警器，不是断路器。
> A spending threshold in AWS Budgets (for example 50 USD). When spending gets close to or above it, AWS emails you. It is an alarm, not a circuit breaker.

### 1.2 我们组的实际情况

我们用的不是个人账号，而是分配给我们的"AWS 项目"账号：从 `settings.aws.com/projects/...` 这个门户用学校邮箱登录（底层是 IAM Identity Center，也就是常说的 SSO），点 "AWS 管理控制台" 进去。这个角色看不到账单页面，所以 credits 余额要找项目负责人确认。

> [!INFO]+ IAM / SSO
> IAM（Identity and Access Management）管"谁能在 AWS 里做什么"。SSO（单点登录，AWS 里叫 IAM Identity Center）让你用学校账号登录，然后扮演一个预先配好权限的角色。
> IAM controls who can do what in AWS. SSO (IAM Identity Center) lets you sign in with your school account and then act as a role with preset permissions.

> [!NOTE]- English version
> **1.1 Account and credits.** Create or sign in to your AWS account. If your school or employer provides GPU credits, request them as they describe. Before redeeming a code, check the account plan: a Free plan account does not accept other promotional credits, so a promotional code needs the Paid plan. Redeem it under Billing and Cost Management → Credits and check the balance, expiry date and covered services. Check the bill regularly and set a budget alert.
>
> Credits are not the same as free usage. After they run out or expire, normal charges apply, and services outside their coverage are charged as usual. A budget alert only sends email; it does not stop anything.
>
> **1.2 What our team actually had.** We did not use a personal account but an "AWS project" account assigned to us: sign in at the `settings.aws.com/projects/...` portal with the school email (IAM Identity Center underneath, i.e. SSO), then click "AWS Management Console". This role cannot open the billing pages, so the credit balance has to be confirmed with the project owner.

<br>

## 2 GPU 配额（quota）

### 2.1 为什么要申请

AWS 给每个账号的每个区域设了上限，防止新账号一上来就开一堆贵机器。GPU 机器归在 **Running On-Demand G and VT instances** 这一项里，**单位是 vCPU，不是台数**。新账号这一项通常是 0。

### 2.2 怎么申请

1. 右上角区域选 **US West (Oregon)，即 us-west-2**（教程的选择；我们的项目账号固定在 us-east-2）。
2. 打开 **Service Quotas → Amazon EC2**。
3. 搜 **Running On-Demand G and VT instances**。
4. 申请总额 **至少 4 vCPU**，够开一台 g4dn.xlarge（前提是没有别的实例占用这项配额）。

申请说明写清楚用途和机型即可，比如：

```text
Training neural networks for a deep learning project. I need one g4dn.xlarge instance in us-west-2, which requires 4 vCPUs.
```

> [!WARNING]+ 配额批了也不代表一定有机器
> 配额只是"允许你用"，既不送钱，也不给你预留 GPU。机房那一刻没有空闲 T4，照样启动失败（见第 9 节 Insufficient capacity）。

> [!INFO]+ Region / 区域，Availability Zone / 可用区
> 区域是一个地理位置的一组机房，比如 us-west-2 在俄勒冈。**配额、机器、硬盘都按区域分开**，切错区域就像进错了仓库，什么都看不到。可用区（AZ）是区域里互相独立的机房，比如 us-west-2a、us-west-2b。
> A region is a group of data centers in one place, e.g. us-west-2 in Oregon. Quotas, instances and disks are all per region; in the wrong region you see nothing of yours. An Availability Zone is an independent data center inside a region, e.g. us-west-2a.

> [!INFO]+ vCPU
> virtual CPU，虚拟 CPU。一般等于物理核上的一个超线程，所以 4 vCPU 大约是 2 个物理核。GPU 配额用 vCPU 计数：g4dn.xlarge 有 4 个 vCPU，所以至少要 4。
> A virtual CPU, usually one hyper-thread of a physical core, so 4 vCPUs is about 2 physical cores. The GPU quota is counted in vCPUs: a g4dn.xlarge has 4, so you need at least 4.

> [!INFO]+ On-Demand 与 Spot
> On-Demand（按需）：随开随用，按秒计费，不会被收回，价格固定。Spot（竞价）：用机房的空闲机器，便宜很多，但 AWS 需要时会提前 2 分钟通知后收回。教程用 On-Demand，省心。"G and VT" 里的 G 是带 GPU 的图形/计算机型，VT 是视频转码机型，它们共用这一项配额。
> On-Demand: start any time, billed per second, never taken away, fixed price. Spot: spare capacity, much cheaper, but AWS can reclaim it after a 2-minute notice. The tutorial uses On-Demand. In "G and VT", G is the GPU family and VT is the video-transcoding family; they share this quota.

### 2.3 我们组踩的坑

我们 2026-11-15 在 us-east-2 申请了 8 vCPU，变成了一个支持工单，状态一直是"工单已打开 / 尚未分配"，同一天之内没有任何进展。**配额审批是人工的，要提前好几天申请。** 等待期间我们发现 SageMaker 那边已经有 GPU 配额，就改走 SageMaker（第 10 节）。

> [!NOTE]- English version
> **2.1 Why a quota is needed.** AWS caps each account per region so a new account cannot start many expensive machines at once. GPU instances fall under **Running On-Demand G and VT instances**, and the unit is **vCPUs, not machines**. For a new account it is usually 0.
>
> **2.2 How to request it.** Select US West (Oregon), us-west-2 (the tutorial's choice; our project account is fixed to us-east-2). Open Service Quotas → Amazon EC2, find Running On-Demand G and VT instances, and request a total of at least 4 vCPUs, enough for one g4dn.xlarge if nothing else uses this quota. The request text above is only an example.
>
> An approved quota only permits usage. It gives no credits and reserves no GPU; if the data center has no free T4 at that moment, the launch still fails (see Insufficient capacity in section 9).
>
> **2.3 What happened to us.** On 2026-11-15 we requested 8 vCPUs in us-east-2. It became a support case that stayed "open / unassigned" for the whole day. Quota approval is done by people, so request it several days ahead. While waiting we found that SageMaker already had GPU quota and switched to it (section 10).

<br>

## 3 命令约定

- 标 **local computer** 的命令在自己电脑上跑；标 **AWS instance** 的命令在 SSH 登上去之后跑。
- `PUBLIC_DNS` 换成你机器当前的公网 DNS 名。
- 示例文件名和本地路径按自己电脑改。
- 行尾的反斜杠 `\` 表示命令没写完，下一行接着。
- 本地命令按 macOS / Linux / WSL 写。Windows 原生 PowerShell 也能用 OpenSSH，权限设置不同，见 4.1 的 Windows 卡片。

> [!INFO]+ WSL
> Windows Subsystem for Linux，在 Windows 里跑一个真正的 Linux 环境，Linux 命令可以原样用。Windows 的下载文件夹在 WSL 里通常是 `/mnt/c/Users/你的用户名/Downloads`。
> A real Linux environment inside Windows, so Linux commands work as written. The Windows Downloads folder is usually `/mnt/c/Users/YOUR_WINDOWS_USER/Downloads` in WSL.

> [!NOTE]- English version
> **3 Command conventions.** Commands labeled local computer run on your laptop or desktop; commands labeled AWS instance run after you connect over SSH. Replace `PUBLIC_DNS` with the instance's current public DNS name, and adapt file names and local paths. A trailing backslash continues the command on the next line. Local examples use macOS, Linux or WSL; native Windows PowerShell also has OpenSSH, but key permissions are set differently (see the Windows card in 4.1).

<br>

## 4 启动 GPU 实例

AWS 控制台里选 us-west-2，打开 **EC2 → Launch instance**，按下表填：

| 设置 / Setting | 填什么 / Value | 为什么 / Why |
| :--- | :--- | :--- |
| Name | GPU Project Instance | 只是个标签，方便认 |
| Amazon Machine Image | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)，选最新日期版本 | 自带 NVIDIA 驱动，省掉最难装的那一步 |
| Image publisher | Amazon | 防止选到别人上传的同名镜像 |
| Architecture | x86_64 | 和镜像、驱动、TensorFlow 轮子匹配 |
| Instance type | g4dn.xlarge | 1 块 T4，入门够用且便宜 |
| Purchase option | On-Demand | 不会被中途收回 |
| Key pair | 新建 RSA 密钥对，名字 gpu-key，下载 .pem | 登录凭证，只能下载这一次 |
| Network | 默认 VPC + 带 internet gateway 的公有子网 | 机器要能上网、能被你连上 |
| Auto-assign public IP | Enable | 没公网 IP 就连不上 |
| Security group | 入站只开 SSH，TCP 22，来源 My IP | 只有你的 IP 能敲门 |
| Storage | 根卷 EBS 保持默认大小或更大，能选 gp3 就选 gp3 | 系统、环境、权重都放这里 |

g4dn.xlarge：4 vCPU，16 GiB 内存，1 块 NVIDIA T4（16 GiB 显存）。镜像只提供驱动，Python 环境后面自己装。

> [!INFO]+ g4dn.xlarge 这个名字怎么读
> AWS 的机型名都是"**家族 + 代数 + 附加属性 . 大小**"：
> - **g**：加速计算里的 GPU 家族（graphics），带 NVIDIA 显卡。对比：p 是更贵的训练卡家族（A100/H100），c 是计算型 CPU，m 是通用型，r 是大内存，t 是可突发的小机器。
> - **4**：第 4 代。g4dn 配 T4，g5 配 A10G，g6 配 L4，代数越新卡越新。
> - **d**：带本地 NVMe 固态盘（instance store），g4dn.xlarge 有 125 GB，**停机就清空**。
> - **n**：网络增强（最高 25 Gbps）。
> - **xlarge**：大小档位。large 一般 2 vCPU，xlarge 4，2xlarge 8，4xlarge 16，依次翻倍，内存也跟着翻倍。
>
> How to read the name: family + generation + attributes . size. **g** is the GPU family (p is the pricier training family, c compute, m general purpose, r memory, t small burstable). **4** is the 4th generation (g4dn has T4, g5 A10G, g6 L4). **d** means local NVMe instance storage (125 GB on g4dn.xlarge, wiped on stop). **n** means enhanced networking (up to 25 Gbps). **xlarge** is the size: large ≈ 2 vCPUs, xlarge 4, 2xlarge 8, 4xlarge 16, with memory doubling too.

常见单卡 GPU 机型对比（按需价格是 us-east-1 / us-west-2 的大约数，随时变，以官网为准）：

| 机型 | GPU | 显存 | vCPU | 内存 | 本地 NVMe | 约每小时 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| g4dn.xlarge | T4 | 16 GiB | 4 | 16 GiB | 125 GB | 0.53 美元 |
| g5.xlarge | A10G | 24 GiB | 4 | 16 GiB | 250 GB | 1.01 美元 |
| g6.xlarge | L4 | 24 GiB | 4 | 16 GiB | 250 GB | 0.80 美元 |
| g5.2xlarge | A10G | 24 GiB | 8 | 32 GiB | 450 GB | 1.21 美元 |

> [!INFO]+ 显存 vs 内存 / GPU memory vs system memory
> 显存在显卡上，模型参数、激活值、batch 都要放得下，决定能训多大的模型和多大的 batch。内存在主板上，决定数据预处理、DataLoader 能放多少东西。T4 的 16 GiB 显存够一般规模的 CNN；我们的推荐模型要对 92 万个商品打分，选了 24 GiB 的 A10G。
> GPU memory sits on the card and must hold parameters, activations and the batch, so it limits model and batch size. System memory limits data preprocessing and loaders. A T4's 16 GiB suits moderate CNNs; our recommender scores 920k items, so we chose the 24 GiB A10G.

> [!INFO]+ AMI / 系统镜像
> Amazon Machine Image，相当于"装好系统和软件的硬盘模板"，每台新机器都从它复制出根盘。**Deep Learning Base OSS Nvidia Driver GPU AMI** 拆开读：Deep Learning（AWS 官方深度学习镜像）+ Base（基础版：有驱动、CUDA、cuDNN、NCCL、Docker，但**不预装** PyTorch/TensorFlow）+ OSS Nvidia Driver（用 NVIDIA 开源内核模块版驱动）+ GPU + Ubuntu 24.04。选 publisher 是 Amazon、架构 x86_64 的那个。官方镜像页面还给了一条查当前区域最新 AMI ID 的命令。
> A template disk with the OS and software installed; every new instance copies its root disk from it. Read the name as: official Deep Learning image, Base edition (driver, CUDA, cuDNN, NCCL, Docker, but no PyTorch/TensorFlow preinstalled), open-source NVIDIA kernel-module driver, GPU, Ubuntu 24.04. Pick the one published by Amazon for x86_64. The official image page also gives a query for the latest AMI ID in your region.

> [!INFO]+ x86_64 与 arm64
> CPU 指令集。x86_64 是 Intel/AMD；arm64 是 AWS 自研的 Graviton。g4dn 是 Intel CPU，镜像和 Python 包都要选 x86_64 版本，选错了要么启动不了，要么装不上包。
> CPU instruction sets: x86_64 is Intel/AMD, arm64 is AWS Graviton. g4dn uses Intel CPUs, so the image and packages must be x86_64.

> [!INFO]+ EBS、gp3、instance store
> **EBS**（Elastic Block Store）是通过网络挂上来的云硬盘，和机器分开存在：**停机不丢**，按容量每月收费（停机也收）。**gp3** 是 EBS 的通用 SSD 类型，比老的 gp2 便宜，性能（3000 IOPS、125 MB/s 起）不随容量绑死。**instance store**（g4dn 名字里的 d）是插在宿主机上的本地 NVMe 盘，快，但**停机、终止就清空**，只适合放缓存和临时文件。
> EBS is a network-attached cloud disk that lives separately from the instance: it survives a stop and is billed per GB-month even while stopped. gp3 is the general-purpose SSD type, cheaper than gp2, with a performance baseline (3000 IOPS, 125 MB/s) independent of size. Instance store (the d in g4dn) is local NVMe on the host: fast, but wiped on stop or terminate, so use it only for caches and scratch files.

> [!INFO]+ VPC、子网、internet gateway、公网 IP、公网 DNS
> **VPC**：你在 AWS 里的一张私有虚拟网络。**子网**：VPC 里的一段 IP 地址，属于某个可用区。**公有子网**：路由表里有一条"去往 0.0.0.0/0 的流量走 internet gateway"，所以能上互联网。**internet gateway**：VPC 通往互联网的大门。**公网 IP**：外面能找到你的地址；**公网 DNS** 是它的域名形式，长得像 `ec2-1-2-3-4.us-west-2.compute.amazonaws.com`。**停机再开机，自动分配的公网 IP 一般会变**，所以 `PUBLIC_DNS` 每次都要重新看。
> A VPC is your private virtual network in AWS; a subnet is an IP range in one AZ; a public subnet routes 0.0.0.0/0 to an internet gateway, the VPC's door to the internet. The public IP is how the outside reaches you, and the public DNS is its name form. After a stop and start the auto-assigned public IP usually changes, so look up `PUBLIC_DNS` again each time.

> [!INFO]+ Security group / 安全组
> 挂在实例上的虚拟防火墙，只写"允许"规则，默认全拒。它是**有状态的**：你允许了入站 SSH，回给你的流量自动放行，不用再写出站规则。"My IP" 会填成你此刻的公网 IP 加 `/32`，意思是只允许这一个地址。端口 22 是 SSH 的默认端口。
> A virtual firewall attached to the instance with allow rules only (everything else is denied). It is stateful: once inbound SSH is allowed, the replies are allowed automatically. "My IP" fills in your current public IP with `/32`, meaning exactly that address. Port 22 is the default SSH port.

### 4.1 选对镜像，检查后启动

- 按上面的 AMI 名字搜索，确认 publisher 是 Amazon、架构是 x86_64。
- SSH 只允许你当前的公网 IP。**8888 端口保持关闭**，Jupyter 走 SSH 隧道（第 8 节）。
- 换了网络（比如从宿舍到图书馆），公网 IP 会变，要回安全组更新 SSH 来源。
- 账号没有默认 VPC 或公有子网的话，按账号管理员的网络说明来。
- 看一眼预估费用，启动 **一台**。
- 等实例变成 **Running** 并且状态检查（status checks）通过，打开实例详情 → **Connect → SSH client**，那里有当前的公网 DNS 和建议的 SSH 命令。
- **.pem 私钥自己收好，AWS 不会让你再下载一次同一把私钥。** 丢了只能换新密钥。

> [!INFO]+ Key pair / 密钥对，.pem
> 一对数学上配对的密钥：**公钥**放在服务器上（AWS 第一次开机时把它写进 `ubuntu` 用户的 `~/.ssh/authorized_keys`），**私钥**在你手里（就是那个 .pem 文件）。登录时服务器出题，只有持有私钥的人能做出正确的签名，所以不用密码。RSA 是一种密钥算法。私钥等于钥匙本身，不能发给任何人，也不能传到 GitHub。
> A mathematically matched pair: the public key goes on the server (AWS writes it into the `ubuntu` user's `~/.ssh/authorized_keys` on first boot) and the private key stays with you (the .pem file). At login the server issues a challenge that only the private-key holder can sign, so no password is needed. RSA is the key algorithm. The private key is the key itself: never share it or push it to GitHub.

> [!NOTE]- English version
> **4 Launch the GPU instance.** In the console, select us-west-2, open EC2 and choose Launch instance, then fill in the table above. The "Why" column explains each choice: the Deep Learning Base AMI already has the NVIDIA driver; publisher Amazon avoids look-alike images; x86_64 matches the driver and TensorFlow wheels; g4dn.xlarge gives one T4 at a low price; On-Demand is never reclaimed; the RSA key pair named gpu-key is your login credential and can be downloaded only once; a public subnet with an internet gateway plus an auto-assigned public IP makes the instance reachable; the security group allows only SSH on TCP 22 from My IP; the root EBS volume stays at default size or larger, gp3 where offered. A g4dn.xlarge has 4 vCPUs, 16 GiB of memory and one NVIDIA T4 with 16 GiB. The image provides only the driver; you install the Python environment yourself.
>
> The cards above explain the instance name, GPU vs system memory, AMI, x86_64 vs arm64, EBS / gp3 / instance store, VPC / subnet / internet gateway / public IP / public DNS, security groups, and key pairs.
>
> **4.1 Select the official image, check access, launch.** Search for the AMI name and confirm publisher Amazon and x86_64. Allow SSH only from your current public IP and keep port 8888 closed, since Jupyter goes through an SSH tunnel (section 8). When your network or public IP changes, update the SSH source in the security group. Without a default VPC or public subnet, follow your administrator's networking instructions. Review the estimated charges and launch one instance. Wait for Running and passed status checks, then open the instance details, Connect → SSH client, for the current public DNS and the suggested SSH command. Keep the .pem file private; AWS never offers the same private key for download again, and a lost key means creating a new one.

<br>

## 5 用 SSH 连上去

### 5.1 在自己电脑上（local computer）

假设下载的私钥叫 `gpu-key.pem`：

```bash
mkdir -p "$HOME/.ssh"

mv "$HOME/Downloads/gpu-key.pem" "$HOME/.ssh/gpu-key.pem"

chmod 700 "$HOME/.ssh"
chmod 400 "$HOME/.ssh/gpu-key.pem"

ssh -i "$HOME/.ssh/gpu-key.pem" ubuntu@PUBLIC_DNS
```

- 浏览器存到别处了就改 Downloads 路径；WSL 里 Windows 的下载目录是 `/mnt/c/Users/YOUR_WINDOWS_USER/Downloads`。
- 这个 Ubuntu 镜像的登录用户名是 **ubuntu**。
- 第一次连会问要不要信任服务器指纹（fingerprint），先核对再输入 yes。

> [!INFO]+ chmod 700 / 400 是在干嘛
> Linux 权限三位数字分别对应"自己 / 同组 / 其他人"，4 读、2 写、1 执行。`700` 让 `.ssh` 目录只有自己能进；`400` 让私钥只有自己能读。**OpenSSH 发现私钥别人也能读，会直接拒绝使用**（报 UNPROTECTED PRIVATE KEY FILE），这是故意的安全设计。
> The three digits are owner / group / others, with 4 read, 2 write, 1 execute. `700` lets only you enter `.ssh`; `400` lets only you read the key. OpenSSH refuses a private key that others can read (UNPROTECTED PRIVATE KEY FILE), by design.

> [!INFO]+ Windows 原生 PowerShell 怎么设私钥权限
> Windows 没有 chmod，用 icacls 去掉继承权限，只给自己读：
> `icacls $env:USERPROFILE\.ssh\gpu-key.pem /inheritance:r /grant:r "$($env:USERNAME):(R)"`
> 然后同样 `ssh -i $env:USERPROFILE\.ssh\gpu-key.pem ubuntu@PUBLIC_DNS`。
> Windows has no chmod; use icacls to remove inherited permissions and grant only yourself read access (command above), then run the same ssh command.

> [!INFO]+ 服务器指纹 / Host key fingerprint
> 服务器也有自己的一对密钥，指纹是它公钥的摘要。第一次连接时核对指纹，确认你连的真是自己那台机器，而不是中间被人冒充（中间人攻击）。信任之后指纹记进 `~/.ssh/known_hosts`，以后变了 ssh 会大声报警。AWS 文档写了怎么从控制台查到它。
> The server has its own key pair; the fingerprint is a digest of its public key. Checking it on first connect proves you reached your own machine and not an impostor in the middle. Once trusted it is stored in `~/.ssh/known_hosts`, and ssh warns loudly if it changes later. The AWS docs explain how to read it from the console.

> [!EXAMPLE]+ 小技巧：写进 ~/.ssh/config
> 每次敲一长串很烦，可以在 `~/.ssh/config` 里写一个别名，之后 `ssh gpu` 就行。停机重开公网 DNS 变了，只改 HostName 这一行。
> ```
> Host gpu
>   HostName PUBLIC_DNS
>   User ubuntu
>   IdentityFile ~/.ssh/gpu-key.pem
> ```
> Tip: put an alias in `~/.ssh/config` and connect with `ssh gpu`. When the public DNS changes after a restart, edit only the HostName line.

### 5.2 在 AWS 机器上（AWS instance）

现在终端里敲的命令都在远程机器上执行了。先确认驱动认得 GPU，再看磁盘：

```bash
nvidia-smi
df -h /
```

工作空间留出大约 **20 GiB** 空闲，给包下载、环境、缓存和模型权重。这只是估算，大项目要更多。

装后面要用的命令行工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl tmux
```

> [!INFO]+ nvidia-smi 里看什么
> NVIDIA System Management Interface。能列出 GPU（Tesla T4）、显存占用、利用率、在跑的进程，就说明**驱动**正常。右上角的 "CUDA Version" 是这个驱动**最高支持**的 CUDA 版本，不代表你的环境里装了它。
> nvidia-smi lists the GPU (Tesla T4), memory use, utilization and processes, which proves the driver works. The "CUDA Version" at the top right is the highest CUDA the driver supports, not what your environment has installed.

> [!INFO]+ df -h、apt-get、sudo
> `df -h /` 看根分区还剩多少空间（h 是人类可读的单位）。`apt-get` 是 Ubuntu 的系统软件包管理器，`update` 刷新软件列表，`install -y` 安装且自动回答 yes。`sudo` 用管理员权限执行。
> `df -h /` shows free space on the root filesystem in readable units. `apt-get` is Ubuntu's system package manager (`update` refreshes the list, `install -y` installs and answers yes). `sudo` runs a command as administrator.

### 5.3 装 uv

uv 用来管理 Python 版本和项目环境。下载官方安装脚本再执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o uv-install.sh

sh uv-install.sh
rm uv-install.sh

. "$HOME/.local/bin/env"

uv --version
```

可以先打开 `uv-install.sh` 看一眼再运行。还是找不到 uv 的话，重新 SSH 登录一次再试 `uv --version`。

> [!INFO]+ uv
> Astral 用 Rust 写的 Python 包和项目管理器，一个工具顶 pip + venv + pyenv，速度快很多。它能自己下载指定版本的 Python（这里要 3.12），按锁文件精确装依赖。`. "$HOME/.local/bin/env"` 是把 uv 加进当前终端的 PATH，不然刚装完找不到命令。
> A fast Python package and project manager written in Rust by Astral, replacing pip + venv + pyenv. It can download a specific Python (3.12 here) and install exactly what the lockfile says. Sourcing `$HOME/.local/bin/env` adds uv to the current shell's PATH.

### 5.4 之后重新连接

再跑一遍 SSH 命令，用机器**当前的**地址。停机再开机，AWS 通常会换一个新的公网 IPv4，`PUBLIC_DNS` 要跟着更新。

> [!NOTE]- English version
> **5.1 On your local computer.** With the key named `gpu-key.pem`, run the commands above: create `~/.ssh`, move the key there, set `chmod 700` on the folder and `chmod 400` on the key, and connect as `ubuntu@PUBLIC_DNS`. Adjust the Downloads path if needed (in WSL it is usually `/mnt/c/Users/YOUR_WINDOWS_USER/Downloads`). The login name for this Ubuntu image is ubuntu. Verify the server fingerprint before accepting the first connection. The cards explain the permission digits (OpenSSH refuses a key others can read), the Windows icacls equivalent, the host fingerprint, and an `~/.ssh/config` alias.
>
> **5.2 On the AWS instance.** Your terminal now runs commands on the remote machine. Check that the driver sees the GPU with `nvidia-smi` and check free space with `df -h /`. Keep roughly 20 GiB free for downloads, the environment, caches and weights; larger projects need more. Install git, curl and tmux with apt-get.
>
> **5.3 Install uv.** uv manages Python and the project environment. Download the official installer, optionally read it, run it, remove it, source `$HOME/.local/bin/env`, and check `uv --version`. If uv is still not found, reconnect over SSH and try again.
>
> **5.4 Reconnect later.** Run the SSH command again with the instance's current address. After a stop and start AWS usually assigns a new public IPv4, so update `PUBLIC_DNS`.

<br>

## 6 准备并测试 neural-style

这一节都在 AWS 机器上跑。用的是固定的仓库版本和它的依赖锁文件，Python 3.12，TensorFlow 2.19.0。

### 6.1 克隆仓库并装依赖

```bash
cd "$HOME"

git clone https://github.com/anishathalye/neural-style.git

cd neural-style

git checkout --detach d0a114879c1c69086e5f812f3432be634e29db5e

uv sync --locked --python 3.12 --no-dev

uv pip install --python .venv/bin/python \
  'tensorflow[and-cuda]==2.19.0'

uv pip check --python .venv/bin/python
```

- 出现 detached HEAD 提示是正常的。
- 全程都用 `.venv/bin/python`，保证每条命令用的是同一个环境。
- **如果重新跑了 `uv sync`，要把 CUDA 那条安装（以及可选的 Jupyter 安装）再跑一遍**，因为 sync 会删掉锁文件之外单独装的包。

> [!INFO]+ git checkout --detach 一个 commit 号
> 直接切到某一次提交，而不是某个分支的最新版本，这样谁在什么时候跑，代码都一模一样，结果才可复现。"detached HEAD" 的意思就是"你现在不在任何分支上，只是停在一个具体的提交"，只读地用没有任何问题。
> Checking out a specific commit instead of a branch tip pins the exact code, so results are reproducible no matter who runs it or when. "Detached HEAD" just means you are on a commit, not a branch, which is fine for read-only use.

> [!INFO]+ uv sync --locked --no-dev 和"为什么要重装 CUDA 包"
> `uv sync` 让 `.venv` **精确等于**锁文件 `uv.lock` 描述的状态：缺的装上，**多的删掉**。`--locked` 要求锁文件和项目配置一致，不一致就报错而不是偷偷改锁文件；`--no-dev` 不装开发依赖。`tensorflow[and-cuda]` 是后来用 `uv pip install` 额外加的，不在锁文件里，所以下一次 sync 会把它当"多余的包"删掉，于是要重装。`uv pip check` 检查已装包之间有没有版本冲突。
> `uv sync` makes `.venv` exactly match `uv.lock`: it installs what is missing and removes what is extra. `--locked` errors out if the lockfile and project config disagree instead of rewriting the lock; `--no-dev` skips dev dependencies. `tensorflow[and-cuda]` is added afterwards with `uv pip install` and is not in the lock, so the next sync removes it as extra, hence the reinstall. `uv pip check` verifies that installed packages have compatible versions.

> [!INFO]+ 驱动和 CUDA 库是两层东西
> **驱动**（driver）在操作系统内核层，由 AMI 装好，`nvidia-smi` 检查的就是它。**CUDA / cuDNN 运行库**在用户层，TensorFlow 要靠它们调用 GPU。`tensorflow[and-cuda]` 这个 extra 会从 pip 装好对应版本的 NVIDIA 库，所以不用自己在系统里装 CUDA Toolkit。两层都对，Python 才能用上 GPU。
> The driver lives in the OS kernel layer and comes with the AMI; `nvidia-smi` checks it. The CUDA / cuDNN runtime libraries live in user space and are what TensorFlow calls. The `and-cuda` extra installs matching NVIDIA libraries from pip, so no system CUDA Toolkit is needed. Both layers must be right for Python to use the GPU.

### 6.2 确认 TensorFlow 看得见 GPU

```bash
.venv/bin/python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)

assert gpus, "No GPU detected; fix the setup before training."
PY
```

**GPU 列表不为空才继续。** 只过了 `nvidia-smi` 不能证明 Python 环境能用 GPU（原因见上面"两层"卡片）。

> [!INFO]+ <<'PY' 是什么写法
> heredoc：把到下一行 `PY` 为止的内容当作标准输入喂给 `python -`（减号表示从标准输入读代码）。`PY` 加了单引号，shell 不会去展开里面的 `$` 之类的符号，代码原样传进去。
> A heredoc feeds everything up to the closing `PY` line to `python -` as stdin (`-` means read code from stdin). Quoting `'PY'` stops the shell from expanding `$` and similar symbols inside.

### 6.3 下载预训练权重并校验

```bash
curl -fL --retry 3 \
  -o imagenet-vgg-verydeep-19.mat \
  https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

sha256sum imagenet-vgg-verydeep-19.mat
```

文件约 535 MB。把打印出来的 SHA256 和下面这个对比，不一样就重新下载：

```text
abdb57167f82a2a1fbab1e1c16ad9373411883f262a1a37ee5db2e6fb0044695
```

> [!INFO]+ SHA256 校验
> 对文件内容算出一个 64 位十六进制的"指纹"，文件哪怕只错一个比特，指纹就完全不同。所以对上了就说明下载完整、没被篡改。`curl -fL --retry 3`：`-f` 遇到 HTTP 错误就失败而不是把错误页存下来，`-L` 跟随跳转，`--retry 3` 失败重试 3 次。
> SHA256 computes a 64-hex-digit fingerprint of the file; a single changed bit gives a completely different value, so a match means the download is complete and untampered. In curl, `-f` fails on HTTP errors instead of saving the error page, `-L` follows redirects, `--retry 3` retries three times.

### 6.4 跑一个最小检查

```bash
.venv/bin/python neural_style.py \
  --content examples/1-content.jpg \
  --styles examples/1-style.jpg \
  --output smoke.jpg \
  --iterations 1 \
  --width 256
```

成功会生成 `smoke.jpg`，说明模型能跑通。迭代 1 次只验证流程，不能拿来判断画质。**不要用网上老的 `scipy==1.1` 降级办法**，它和这个仓库现在的依赖冲突。

> [!NOTE]- English version
> **6 Prepare and test neural-style** (all on the AWS instance). The guide pins a repository revision and its lockfile with Python 3.12 and TensorFlow 2.19.0.
>
> **6.1** Clone the repository, check out the pinned commit (a detached HEAD message is expected), run `uv sync --locked --python 3.12 --no-dev`, add `tensorflow[and-cuda]==2.19.0` with `uv pip install`, and run `uv pip check`. Use `.venv/bin/python` everywhere so every command uses the same environment. If you rerun `uv sync`, repeat the CUDA install (and the optional Jupyter install), because sync removes packages that are not in the lockfile. The cards explain pinned commits, what `uv sync --locked --no-dev` does, and why the driver and the CUDA libraries are two separate layers.
>
> **6.2** Run the TensorFlow check above and continue only if the GPU list is not empty; passing `nvidia-smi` alone does not prove that the Python environment can use the GPU. The heredoc card explains the `<<'PY'` syntax.
>
> **6.3** Download the VGG-19 weights (about 535 MB) and compare `sha256sum` with the expected value; if it differs, download again. The card explains checksums and the curl flags.
>
> **6.4** Run the one-iteration check; a `smoke.jpg` file means the model executes, though one iteration says nothing about image quality. Do not use the old `scipy==1.1` workaround; it conflicts with the current requirements.

<br>

## 7 用 tmux 跑完整模型

tmux 能让终端会话在 SSH 断开后继续活着。**但停机、重启、终止实例之后，里面的任务一样会没。**

```bash
tmux new -s train

cd "$HOME/neural-style"

.venv/bin/python neural_style.py \
  --content examples/1-content.jpg \
  --styles examples/1-style.jpg \
  --output output.jpg \
  --iterations 1000
```

跑完 `output.jpg` 在 `~/neural-style` 里。

**分离（detach）**：按 `Ctrl+b`，松开两个键，再按 `d`。之后可以放心关掉 SSH，任务继续跑。

重新 SSH 登录后，列出会话并回到它：

```bash
tmux ls

tmux attach -t train
```

需要中断正在跑的命令就按 `Ctrl+c`；任务结束后在空闲的 tmux 里输入 `exit` 关掉会话。更多 tmux 操作见旧笔记 [Remote-SSH窗口关闭/网络断开解决方法](../remotessh持续实验)。

> [!INFO]+ 为什么 SSH 一断任务就死，tmux 为什么能救
> SSH 断开时，服务器会给这个登录会话里的进程发 **SIGHUP**（挂断信号），默认处理方式就是退出，所以直接在 SSH 里跑的训练会跟着死。tmux 有一个独立的后台服务进程，你的训练是它的子进程，不属于那个 SSH 会话，SSH 断了它收不到 SIGHUP。你再连上来，只是重新"接上"了那个一直在跑的终端画面。nohup 是另一种办法：让进程忽略 SIGHUP，但没法再接回画面。
> When SSH drops, the server sends SIGHUP to the processes of that login session, and their default reaction is to exit. tmux runs its own background server; your job is its child, outside the SSH session, so it never gets that SIGHUP. Reconnecting just reattaches the still-running terminal. nohup instead makes a process ignore SIGHUP, but you cannot reattach to it.

> [!NOTE]- English version
> **7 Run the full model with tmux.** tmux keeps a terminal session alive when SSH disconnects, but it does not keep a job alive if you stop, reboot or terminate the instance. Start a session with `tmux new -s train`, go to `~/neural-style`, and run the 1000-iteration example; `output.jpg` appears in `~/neural-style`. To detach, press Ctrl+b, release both keys, then press d; you can then close SSH while the job continues. After reconnecting, `tmux ls` lists sessions and `tmux attach -t train` returns to yours. Ctrl+c interrupts a running command; type `exit` to close an idle tmux shell when the job is done. The card explains why a dropped SSH connection kills a job (SIGHUP) and why tmux avoids it.

<br>

## 8 在电脑和 AWS 之间拷文件

这些命令在**自己电脑**上跑：另开一个终端，或者先退出 SSH。`PUBLIC_DNS` 换成当前地址，本地路径按自己的改。

下载生成的图片：

```bash
scp -i "$HOME/.ssh/gpu-key.pem" \
  ubuntu@PUBLIC_DNS:~/neural-style/output.jpg \
  "$HOME/Downloads/"
```

下载一个文件夹：

```bash
scp -i "$HOME/.ssh/gpu-key.pem" -r \
  ubuntu@PUBLIC_DNS:~/neural-style/examples \
  "$HOME/Downloads/"
```

上传一个文件：

```bash
scp -i "$HOME/.ssh/gpu-key.pem" \
  "$HOME/Downloads/my-image.jpg" \
  ubuntu@PUBLIC_DNS:~/neural-style/
```

上传一个文件夹：

```bash
scp -i "$HOME/.ssh/gpu-key.pem" -r \
  ./my-project \
  ubuntu@PUBLIC_DNS:~/
```

下载整个项目用 `scp -r` 远程文件夹即可；项目里有很大的 `.venv`、数据集或权重时，只拷需要的部分。**代码和结果另外留一份备份**（代码推 GitHub 最省心）。

> [!INFO]+ scp 和 rsync
> scp（secure copy）走 SSH 通道拷文件，`-i` 指定私钥，`-r` 递归拷文件夹，写法是"从哪 到哪"，远程路径写成 `用户@地址:路径`。大文件或要断点续传、只同步改动过的文件时，rsync 更合适（见旧笔记 [rsync](../rsync)）。
> scp copies over SSH: `-i` picks the key, `-r` copies folders, and the order is source then destination, with remote paths written `user@host:path`. For big transfers, resuming, or syncing only changed files, rsync is better.

> [!NOTE]- English version
> **8 Copy files between your computer and AWS.** Run these on your local computer, in another terminal or after leaving SSH, with the current `PUBLIC_DNS` and your own local paths. The four commands download an image, download a folder, upload a file and upload a folder. To download a whole project use `scp -r` on its remote folder, but copy only what you need when it contains a large `.venv`, dataset or weights. Keep a separate backup of code and results (pushing code to GitHub is the easiest). The card compares scp with rsync.

<br>

## 9 （可选）通过 SSH 隧道用 Jupyter

在 AWS 上跑 JupyterLab，在自己电脑浏览器里通过 localhost 打开。**安全组里 8888 保持关闭**，连接由 22 端口上的 SSH 承载。

### 9.1 在 AWS 机器上启动 Jupyter

和模型用同一个环境，notebook 的内核才能用到 TensorFlow 和 CUDA：

```bash
cd "$HOME/neural-style"

uv pip install --python .venv/bin/python \
  'tensorflow[and-cuda]==2.19.0' \
  jupyterlab \
  ipykernel

.venv/bin/python -m ipykernel install --user \
  --name neural-style-env \
  --display-name "neural style"

.venv/bin/python -m jupyterlab \
  --no-browser \
  --ip=127.0.0.1 \
  --port=8888 \
  --ServerApp.port_retries=0
```

这个终端保持打开，或者在 tmux 里启动 Jupyter。**默认的 token 认证别关**，Jupyter 会打印一个带 token 的 localhost 链接，token 不要外传。

### 9.2 在自己电脑上建隧道

另开一个本地终端：

```bash
ssh -i "$HOME/.ssh/gpu-key.pem" -N \
  -L 127.0.0.1:8888:127.0.0.1:8888 \
  ubuntu@PUBLIC_DNS
```

用 Jupyter 期间这个隧道一直开着。浏览器打开 `http://localhost:8888/lab`，粘贴 AWS 终端里打印的 token（或者直接打开 Jupyter 打印的那个带 token 的链接），选 **neural style** 内核。

### 9.3 检查 notebook 内核

做 GPU 工作前先跑这一格：

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(gpus)

assert gpus, "The selected notebook kernel cannot see a GPU."
```

### 9.4 本地 8888 被占用了

本地换成 8889：

```bash
ssh -i "$HOME/.ssh/gpu-key.pem" -N \
  -L 127.0.0.1:8889:127.0.0.1:8888 \
  ubuntu@PUBLIC_DNS
```

然后打开 `http://localhost:8889/lab`。远程那边还是 8888。

### 9.5 结束 Jupyter

1. 保存 notebook，关掉内核。
2. 在 Jupyter 的终端按 `Ctrl+c` 停服务（问就确认）。
3. 在隧道终端按 `Ctrl+c` 关隧道。
4. 所有任务都结束后，停掉 EC2 实例。

> [!INFO]+ SSH 本地端口转发 -L 的原理
> `-L 127.0.0.1:8888:127.0.0.1:8888` 读作"把**我电脑**的 127.0.0.1:8888 转发到**远程机器看到的** 127.0.0.1:8888"。浏览器访问本地 8888，数据先进加密的 SSH 连接，到了远程机器再交给它本机的 8888（Jupyter）。Jupyter 只监听远程的 127.0.0.1，公网根本访问不到，所以 8888 不用对外开放；`-N` 表示只建隧道、不执行远程命令。token 是 Jupyter 自己的登录口令，挡住同一台机器上的其他人。
> `-L 127.0.0.1:8888:127.0.0.1:8888` means: forward my computer's 127.0.0.1:8888 to 127.0.0.1:8888 as seen by the remote machine. Browser traffic to local 8888 enters the encrypted SSH connection and is delivered to Jupyter on the remote side. Jupyter listens only on the remote loopback, so it is unreachable from the internet and port 8888 never needs opening. `-N` builds the tunnel without running a remote command. The token is Jupyter's own password.

> [!INFO]+ ipykernel 内核注册
> Jupyter 用"内核"决定 notebook 里的代码交给哪个 Python 执行。`ipykernel install --user --name ...` 把 `.venv` 这个环境注册成一个名叫 "neural style" 的内核，选了它，notebook 才和命令行用的是同一套包。
> A kernel decides which Python runs the notebook code. Registering `.venv` as "neural style" makes the notebook use the same packages as the command line.

> [!NOTE]- English version
> **9 Optional: Jupyter through an SSH tunnel.** JupyterLab runs on AWS and opens through localhost on your computer. Keep port 8888 closed in the security group; SSH on port 22 carries the connection.
>
> **9.1** On the instance, install JupyterLab and ipykernel into the same `.venv` (with the TensorFlow CUDA extra), register the kernel "neural style", and start JupyterLab on 127.0.0.1:8888 without a browser. Keep that terminal open or run it in tmux. Leave token authentication on and keep the printed token private.
>
> **9.2** In another local terminal, open the tunnel shown above and keep it running. Browse to `http://localhost:8888/lab`, paste the token (or open the printed localhost link), and select the neural style kernel.
>
> **9.3** Run the GPU check cell before any GPU work.
>
> **9.4** If local port 8888 is taken, forward local 8889 instead and open `http://localhost:8889/lab`; the remote port stays 8888.
>
> **9.5** To finish: save the notebook and shut down its kernels, stop Jupyter with Ctrl+c (confirm if asked), stop the tunnel with Ctrl+c, and stop the EC2 instance once all jobs are done. The cards explain local port forwarding and kernel registration.

<br>

## 10 用完停机（最重要）

先保存工作、备份重要结果。EC2 控制台里：

1. 选中你的实例。
2. **Instance state → Stop instance**。
3. 等状态变成 **Stopped**。

**关掉 SSH、分离 tmux、关掉 Jupyter，都不会停止 EC2 的计算费用。**

停机（Stop）会：

- 保留 EBS 硬盘上的数据；
- 结束所有在跑的进程；
- 清空内存和临时的 instance store 数据；
- 重新开机后，自动分配的公网 IP 通常会变。

**停机期间 EBS 硬盘照样收费**；保留着不用的 Elastic IP 也收费。

**Terminate（终止）** 只在彻底不用、并且已经备份好一切之后再点。终止是永久的，设置了"随实例删除"的硬盘会一起被删。

> [!INFO]+ Stop、Terminate、Elastic IP 分别意味着什么
> Stop 像关机：机器还是你的，硬盘还在，只是不再按小时收 GPU 的钱。Terminate 像退租：机器没了，默认根盘也一起删。Elastic IP 是一个固定公网 IP，好处是重开机地址不变，代价是不用的时候也按小时收费。
> Stop is like shutting down: the instance and disk remain and hourly compute charges end. Terminate is like ending the lease: the instance is gone and the root disk is deleted by default. An Elastic IP is a fixed public address that survives restarts but is charged even when idle.

> [!EXAMPLE]+ 我怎么防止自己忘记关机
> 教程里没写，但很实用：EC2 可以建一个 CloudWatch 告警，"CPU 利用率连续 N 分钟低于某个值就执行 Stop this instance"，相当于闲置自动关机。我们组在 SageMaker 上写了一个看门狗脚本，连续 30 分钟没有训练进程、GPU 利用率也低于 5%，就自动关掉机器（第 11.6 节）。
> Not in the tutorial but useful: a CloudWatch alarm on EC2 can run "Stop this instance" when CPU utilization stays below a threshold for N minutes, i.e. an idle auto-stop. On SageMaker our team wrote a watchdog that stops the machine after 30 minutes with no training process and GPU utilization under 5% (section 11.6).

> [!NOTE]- English version
> **10 Stop the instance when finished (the most important step).** Save your work and back up important results. In the EC2 console select the instance, choose Instance state → Stop instance, and wait for Stopped. Closing SSH, detaching tmux or shutting down Jupyter does not stop EC2 compute charges. Stopping keeps the EBS disks, ends running processes, erases RAM and instance-store data, and usually changes the auto-assigned public IP on restart. EBS storage is still charged while stopped, and retained Elastic IPs are charged too. Use Terminate only when you are completely done and have backed up everything; it is permanent, and volumes set to delete on termination go with it. The cards explain Stop / Terminate / Elastic IP and two ways to avoid forgetting: a CloudWatch idle alarm on EC2, and our 30-minute watchdog on SageMaker.

<br>

## 11 我们组实际走的路：SageMaker Studio

EC2 的 GPU 配额一直是 0，但我们在 Service Quotas 里翻 SageMaker 的配额时发现：项目账号里每种 g5 机型都已经有 1 台的 Studio 配额。原来"可以直接用"指的是这个。

### 11.1 SageMaker 和 EC2 的区别

| | EC2 | SageMaker Studio |
| :--- | :--- | :--- |
| 是什么 | 一台"裸"虚拟机 | 网页版 JupyterLab 工作室，背后也是 AWS 的机器 |
| 怎么用 | 自己 SSH 登录，自己装环境 | 浏览器打开就有终端、文件栏、notebook，环境预装 |
| 开关机 | 自己 Start / Stop | 点 Run space / Stop space |
| 价格 | 同样的卡更便宜 | 贵 20% 到 40%，是托管服务的溢价 |
| 配额 | 单独一套（我们是 0） | 单独一套（我们每种 g5 各 1 台） |

结论：不是 SageMaker 更好，而是当时只有它能用。

> [!INFO]+ SageMaker 的四层结构
> 从大到小：**域（Domain）**，整个工作环境，建一次，免费；**用户配置文件（User profile）**，域里的一个"住户"；**空间（Space）**，你的"房间"，自带一块 EBS 硬盘，文件都在 `/home/sagemaker-user`，**停机也保留**；**应用（App）+ 实例**，点 Run space 时分配一台机器（比如 ml.g5.2xlarge）把硬盘挂上去，**从这一刻开始按小时收费**。SageMaker 的机型名前面多一个 `ml.`，硬件和 EC2 同名机型一样。
> From big to small: Domain (the whole environment, created once, free); User profile (a resident); Space (your room with its own EBS disk, files in `/home/sagemaker-user`, kept while stopped); App plus instance (a machine such as ml.g5.2xlarge attached when you click Run space, billed from that moment). SageMaker instance names carry an `ml.` prefix but the hardware matches the EC2 type.

### 11.2 搭起来的步骤

1. 控制台 → **Amazon SageMaker AI → 域 → 创建域 → 针对单个用户设置（快速设置）**。会出现一个 `ValidationException ... servicecatalog:ListAcceptedPortfolioShares` 报错，可以忽略，只影响项目模板功能，域照样建好。
2. 域 → 用户配置文件 → **打开 Studio**（会弹新标签页）。
3. Studio 左侧 **JupyterLab → Create JupyterLab space**，名字 `gpu-shared`，Sharing 选 **Share with my domain**（全组可用）。
4. 机型选 **ml.g5.2xlarge**（1 块 A10G 24 GiB，8 vCPU，32 GiB 内存），硬盘填大小。
5. 点 **Run space** 才开始计费，等 Running 后 **Open JupyterLab**，File → New → Terminal。

> [!WARNING]+ 三个只有踩过才知道的坑
> 1. **私有 / 共享建好后改不了**。我们先建了私有空间，想给组员用只能新建一个共享的，再把私有的删掉。
> 2. **网页上改的机型和硬盘，不点 Run 就离开，不会保存**。刷新一下就变回 ml.t3.medium / 5 GB。想提前存好，要用命令行 `aws sagemaker update-space`。
> 3. **共享空间的设置页没有 Idle Shutdown（空闲自动关机）**，私有空间才有，最短还只能设 60 分钟。所以我们自己写了 30 分钟看门狗。
> Three lessons: the private / shared setting cannot be changed after creation; instance type and disk size edited in the web page are lost unless you click Run (use `aws sagemaker update-space` to persist them); shared spaces have no Idle Shutdown field (private ones do, minimum 60 minutes), so we wrote a 30-minute watchdog.

### 11.3 空间里的东西会不会丢

| 情况 | 主目录 `/home/sagemaker-user` 里的文件 |
| :--- | :--- |
| Stop / Run、自动关机、换机型（CPU 换 GPU） | 都在 |
| 删除空间 | 全没 |
| 系统环境里 `pip install` 的包、正在跑的程序 | 重启后没了（把环境建在主目录里就不会丢） |

所以数据只下载一次；下载数据用便宜的 CPU 机器（比如 ml.t3.large），训练时停机换成 GPU 机型，硬盘跟着空间走。

### 11.4 从自己电脑 SSH 进 SageMaker 空间

原理和 EC2 不一样：EC2 是你主动连机器的 22 端口；SageMaker 用的是 **AWS Systems Manager Session Manager**，机器上的代理主动向 AWS 建立出站连接，你的电脑调用 `StartSession` 拿到一个会话，再由 Session Manager 插件把 SSH 塞进这条加密通道。**不需要公网 IP，也不需要开任何入站端口。**

一次性准备（Windows）：

```powershell
winget install --id Amazon.AWSCLI -e
winget install --id Amazon.SessionManagerPlugin -e
aws login --profile myproj --region us-east-2
aws sts get-caller-identity --profile myproj
```

- 我们是从"AWS 项目"门户登录的，标准的 `aws sso login` 会报 `User not authorized for this Identity Center instance`，拿不到账户权限。改用新版 AWS CLI 的 **`aws login`**：它把浏览器里的控制台登录换成命令行临时凭证，会自动续期。
- 空间要打开 **Remote Access**（`update-space` 里设 `RemoteAccess: ENABLED`）。
- `~/.ssh/config` 里加一个 Host，`ProxyCommand` 指向一个调用 `aws sagemaker start-session` 再交给 `session-manager-plugin` 的脚本（AWS 文档 "Connect from the terminal via SSH CLI" 里有现成的 PowerShell 版本），之后 `ssh sm-gpu` 即可。空间必须在运行。

> [!INFO]+ AWS CLI、profile、临时凭证
> AWS CLI 是命令行版的控制台，`aws <服务> <操作>`。profile 是一组命名的登录配置，`--profile myproj` 表示用这一组。`aws login` / SSO 拿到的都是几小时有效的**临时凭证**，过期重新登录，比长期的 Access Key 安全得多。
> The AWS CLI is the console on the command line (`aws <service> <operation>`). A profile is a named set of login settings. `aws login` and SSO give temporary credentials valid for hours, far safer than long-lived access keys.

### 11.5 代码怎么在本机和 AWS 之间同步

- **代码走 GitHub**：开工 `git pull`，收工 `git commit` + `git push`。我们给共享空间单独生成了一个 **fine-grained token**：只能访问这一个仓库，权限只有 Contents 读写，90 天过期。共享机器上组员都能用到它，所以权限越小越好。
- **数据、特征、权重留在空间硬盘**，太大，不进 Git。
- 指标、配置、曲线、日志很小，跑完提交，组员都看得到。

> [!INFO]+ Fine-grained personal access token
> GitHub 的细粒度个人访问令牌，可以限定"哪个仓库、哪种权限、多久过期"。在 Settings → Developer settings → Personal access tokens → Fine-grained tokens 生成，只显示一次。在共享机器上用它代替你的账号密码，泄露了影响也只限这个仓库，到期自动失效。
> A GitHub token restricted to chosen repositories, permissions and an expiry date. It is shown only once. On a shared machine it replaces your account login, so a leak affects only that repository and expires by itself.

### 11.6 30 分钟自动关机

看门狗每 60 秒检查一次：有没有训练 / 编码 / 预处理 / 下载进程在跑，GPU 利用率是否超过 5%。**连续 30 分钟都没有**，就调用 SageMaker 的 `DeleteApp` 把机器停掉（硬盘数据保留）。每次 Run space 后都要先启动它：

```bash
nohup python code/aws/sm_autostop.py > ~/autostop.log 2>&1 &
tail -3 ~/autostop.log
```

随时可以在本机查有没有机器在收费：

```powershell
aws sagemaker list-apps --domain-id-equals <域ID> --profile myproj --region us-east-2 --query "Apps[?Status=='InService']"
```

返回 `[]` 就是没有。

### 11.7 钱都花在哪

| 项目 | 大约 | 什么时候收 |
| :--- | :--- | :--- |
| ml.g5.2xlarge | 1.5 到 1.7 美元 / 小时 | 运行时 |
| ml.t3.large（CPU） | 约 0.1 美元 / 小时 | 运行时 |
| 空间硬盘 | 每 GB 每月约 0.11 到 0.14 美元，100 GB 约 11 到 14 美元 / 月 | **一直收，停机也收** |
| 域、用户配置文件 | 免费 | |

硬盘只能调大不能调小，按真实需要设。最大的浪费永远是忘了关 GPU。

> [!NOTE]- English version
> **11 What our team actually used: SageMaker Studio.** EC2 GPU quota stayed at 0, but in Service Quotas we found that the project account already had a Studio quota of one machine for every g5 size, which is what "it can be used directly" meant.
>
> **11.1 SageMaker vs EC2.** EC2 is a bare virtual machine you SSH into and set up yourself; SageMaker Studio is a web JupyterLab with terminal, file browser and notebooks preinstalled, started with Run space / Stop space. The same GPU costs about 20 to 40% more on SageMaker, and the two services have separate quotas (ours: EC2 0, SageMaker one of each g5). SageMaker was not better, it was simply the only option available. The card explains the four layers: Domain, User profile, Space (EBS home directory kept while stopped) and App plus instance (billed while running), and the `ml.` prefix.
>
> **11.2 Setup.** Create a domain with single-user quick setup (ignore the servicecatalog ValidationException), open Studio from the user profile, create a JupyterLab space `gpu-shared` with Share with my domain, choose ml.g5.2xlarge (one A10G 24 GiB, 8 vCPUs, 32 GiB), set the disk size, and click Run space (billing starts), then Open JupyterLab and a terminal. Lessons: sharing cannot be changed after creation; instance type and disk edited in the page are lost without Run, so persist them with `aws sagemaker update-space`; shared spaces have no Idle Shutdown field, so we wrote a 30-minute watchdog.
>
> **11.3 What survives.** Files in `/home/sagemaker-user` survive Stop / Run, auto-stop and instance changes (CPU to GPU); deleting the space deletes them; packages installed into the system environment and running processes are gone after a restart (keep environments in the home directory). So data is downloaded once, on a cheap CPU instance such as ml.t3.large, and training later switches the same space to a GPU type.
>
> **11.4 SSH into a SageMaker space.** Unlike EC2, nothing connects to port 22: an agent on the machine opens an outbound connection to AWS Systems Manager, your computer calls `StartSession`, and the Session Manager plugin carries SSH through that encrypted channel, with no public IP and no inbound port. On Windows install the AWS CLI and the Session Manager plugin, then sign in with `aws login --profile myproj` (from the projects portal, `aws sso login` fails with "User not authorized for this Identity Center instance"; `aws login` turns the browser console session into auto-refreshing temporary credentials). Turn on Remote Access for the space and add an `~/.ssh/config` Host whose ProxyCommand calls `aws sagemaker start-session` and hands the result to `session-manager-plugin` (the AWS doc "Connect from the terminal via SSH CLI" has a PowerShell version); then `ssh sm-gpu` works while the space runs. The card explains the AWS CLI, profiles and temporary credentials.
>
> **11.5 Keeping code in sync.** Code goes through GitHub: `git pull` before work, commit and push after. The shared space uses a fine-grained token limited to one repository, Contents read and write, expiring in 90 days, because everyone on the shared machine can use it. Data, features and weights stay on the space disk; small metrics, configs, curves and logs are committed after each run.
>
> **11.6 Thirty-minute auto-stop.** Every 60 seconds the watchdog checks for training, encoding, preprocessing or download processes and for GPU utilization above 5%. After 30 minutes with none, it calls SageMaker `DeleteApp` to stop the machine (the disk is kept). Start it after every Run space with the command above; from your computer, `list-apps` filtered to InService returns `[]` when nothing is billing.
>
> **11.7 Where the money goes.** ml.g5.2xlarge costs about 1.5 to 1.7 USD per hour while running, ml.t3.large about 0.1 USD per hour, the space disk about 0.11 to 0.14 USD per GB-month (100 GB is about 11 to 14 USD per month) all the time, even while stopped, and domains and profiles are free. Disks can grow but never shrink, so size them to real needs. The biggest waste is always a GPU left running.

<br>

## 12 后来拿到了 EC2：一台机器，全组用 SSH 共用

两天后 EC2 的 GPU 配额批下来了（us-east-2，8 vCPU），我们就从 SageMaker 搬到了一台 **g5.2xlarge**（1 块 A10G 24 GiB，8 vCPU，32 GiB 内存）。同样的卡，EC2 约 1.2 美元 / 小时，比 SageMaker 便宜两三成。训练放在 tmux 里无人值守地跑，开机自启一个 30 分钟空闲自动停机的看门狗（判断逻辑和 11.6 一样；停机方式是在机器里执行 `sudo shutdown -h now`，前提是启动实例时把 **Shutdown behavior 设为 Stop**，默认就是 Stop。如果设成了 Terminate，系统关机就等于删除实例连同硬盘，一定要核对）。

问题随之而来：**一台机器，三个人，怎么都能登录，又不互相发私钥？**

### 12.1 原理：两把锁

登录 EC2 要同时过两关：

| 关卡 | 管什么 | 放在哪 |
| :--- | :--- | :--- |
| 安全组（防火墙） | **从哪来**：只有名单上的 IP 能连 22 端口 | AWS 控制台 / CLI |
| `authorized_keys`（公钥名单） | **你是谁**：只有私钥和名单上某个公钥配对的人能登录 | 机器上 `~ubuntu/.ssh/authorized_keys`，一行一个公钥 |

所以给组员开权限 = 把她的**公钥**加进 `authorized_keys` + 把她的 **IP** 加进安全组。私钥从头到尾只在她自己电脑上。

> [!INFO]+ authorized_keys
> 服务器上的一个纯文本文件，每行一个公钥，格式是 `算法 公钥内容 备注`，比如 `ssh-ed25519 AAAA... alice-cs230`。登录时服务器拿你声称的公钥出题，你用对应私钥签名，对得上就放行。AWS 启动实例时会把创建时选的那把密钥对的公钥写进第一行；之后再加人就是往后追加行，删人就是删掉那一行。行尾备注写上名字和日期，以后好查。
> A plain-text file on the server with one public key per line (`algorithm key comment`). At login the server challenges you with a listed public key and you sign with the matching private key. AWS writes the launch key pair's public key as the first line; adding a person appends a line, removing them deletes it. Put a name and date in the comment.

> [!WARNING]+ 为什么不能把 .pem 发给组员
> 最省事的做法是把创建实例时下载的 `.pem` 发到群里，大家都用它登录。**不要这样做**：私钥一旦发出去就收不回来，聊天记录、网盘、邮件里都会留一份；谁泄露的也查不出来；有人退出项目也没法单独收回他的权限，只能换掉所有人的钥匙。每人一对密钥，公钥交上来，私钥自己留着，以上问题都没有。
> Sending the launch `.pem` to the group chat is the shortcut to avoid: a shared private key cannot be taken back, copies stay in chats and drives, leaks cannot be traced, and one person cannot be removed without re-keying everyone. One key pair per person, public half handed in, private half kept.

### 12.2 组员：生成自己的密钥（只做一次）

Mac 的终端、Linux、Windows 的 PowerShell 都自带 OpenSSH，命令一样：

```bash
ssh-keygen -t ed25519 -C "yourname-cs230" -f ~/.ssh/cs230_ec2_me
```

- 提示输入密码（passphrase）时可以直接回车不设，也可以设一个只有自己知道的。**输入时屏幕上什么都不显示，是正常的。**
- 生成两个文件：`~/.ssh/cs230_ec2_me` 是**私钥，留在自己电脑上**；`~/.ssh/cs230_ec2_me.pub` 是公钥，可以发。
- 把 `.pub` 的内容发给负责人（一行，以 `ssh-ed25519` 开头），再附上自己的**公网 IP**：

```bash
cat ~/.ssh/cs230_ec2_me.pub
curl https://checkip.amazonaws.com
```

`curl` 那行返回的一串数字就是公网 IP；也可以用浏览器打开同一个网址。

> [!INFO]+ ed25519、-C、-f
> `-t ed25519` 选算法：比老的 RSA 更短更快，安全性相当，现在是 OpenSSH 推荐的默认。`-C` 是写在公钥末尾的备注，只为好认，不影响安全。`-f` 指定文件名；不写的话默认是 `~/.ssh/id_ed25519`，容易和别的项目的钥匙混在一起，所以给这个项目单独起名。
> `-t ed25519` picks the algorithm (shorter and faster than RSA, OpenSSH's recommended default). `-C` is a comment at the end of the public key, for recognition only. `-f` sets the file name; the default `~/.ssh/id_ed25519` mixes with other projects, so give this one its own name.

> [!INFO]+ 公网 IP 和内网 IP
> 电脑"设置 → 网络"里看到的 `192.168.x.x`、`10.x.x.x` 是**内网 IP**，只在你家路由器或学校网络里有意义。AWS 安全组看到的是你出了路由器之后的**公网 IP**，只能从外面的网站查（比如上面那个 AWS 自己的地址）。发错成内网 IP 是最常见的错误之一。
> The `192.168.x.x` / `10.x.x.x` address in your network settings is private and only meaningful inside your router or campus network. The security group sees your public IP, which only an outside site can tell you. Sending the private one is a classic mistake.

### 12.3 负责人：给组员开权限

在负责人自己的电脑上（已经能登录实例，AWS CLI 已登录）。`INSTANCE_IP` 是实例当前的公网地址，`TEAMMATE_IP` 是组员发来的 IP，`SG_ID` 是实例的安全组 ID：

```bash
# 1) 把组员的公钥追加到实例上（已存在就不重复加），行尾备注名字和日期
KEY="$(awk '{print $1" "$2}' alice.pub) alice-added-$(date +%F)"
ssh -i ~/.ssh/my-key.pem ubuntu@INSTANCE_IP \
  "grep -qF '$(awk '{print $2}' alice.pub)' ~/.ssh/authorized_keys || echo '$KEY' >> ~/.ssh/authorized_keys"

# 2) 安全组放行组员的 IP（/32 = 只放这一个地址）
aws ec2 authorize-security-group-ingress --profile myproj --region us-east-2 \
  --group-id SG_ID \
  --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=TEAMMATE_IP/32,Description=alice-2026-11-17}]"
```

我们把这两步写成了一个小脚本 `add_teammate_key.sh <公钥文件> <IP> <名字>`，脚本里先检查文件确实是公钥（以 `ssh-ed25519` / `ssh-rsa` 开头），里面如果出现 `PRIVATE KEY` 字样就拒绝，防止组员把私钥误发过来。

收回权限：删掉实例上 `~/.ssh/authorized_keys` 里带她名字的那一行，再删掉安全组里对应的规则（控制台或 `aws ec2 revoke-security-group-ingress`）。

> [!INFO]+ 安全组规则的 Description
> 每条入站规则可以写一段说明。写上"谁、哪天"，过一个月再看规则列表时，才知道哪个 IP 是谁的、还要不要留。不写的话只剩一排数字。
> Each inbound rule has a description field. "Who, which day" is what lets you tell, a month later, whose IP each rule is and whether it can go.

### 12.4 组员：登录和看进度

```bash
ssh -i ~/.ssh/cs230_ec2_me ubuntu@INSTANCE_IP
```

- Windows PowerShell 里路径也可以写成 `$HOME\.ssh\cs230_ec2_me`。`-i` 后面跟的是**私钥**（没有 `.pub` 的那个）。
- 第一次会问 `Are you sure you want to continue connecting (yes/no)?`，输入 `yes` 回车（指纹的含义见 5.1 的卡片）。
- 看到 `ubuntu@ip-...` 开头的提示符就是进去了。只想看看进度的话：

```bash
tail -5 ~/CS-230/runs/logs/<日志文件>.log   # 训练日志最后几行
nvidia-smi                                  # GPU 在不在忙
tmux attach -r -t cs230                     # 只读方式看正在跑的 tmux 会话
exit                                        # 退出登录，训练不受影响
```

`tmux attach -r` 里的 `-r` 是只读：能看不能输入，误按键盘也不会影响训练。从 tmux 里出来：先按 `Ctrl-b`，松开，再按 `d`。

### 12.5 IP 会变：最常见的"连不上"

**现象**：昨天还好好的，今天 `ssh: connect to host ... port 22: Operation timed out`。

**原因**：安全组只认名单上的 IP，而你的公网 IP 经常变：

- 换了网络：宿舍 → 图书馆 → 家里 → 手机热点，每个网络出口不同。
- 同一个网络也会变：学校和宽带运营商会从一段地址池里轮流分配，我们组有人半小时内就从 `x.x.x.22` 变成了 `x.x.x.77`。
- 开着 VPN：出口是 VPN 服务器的 IP，换节点就换 IP。

**办法**：

1. 先排除实例没开机：控制台里实例是 Running 吗？负责人自己能连上吗？
2. 组员重新运行 `curl https://checkip.amazonaws.com`，把新 IP 发给负责人加上。
3. 同一个网络反复变，可以放行整段地址，比如 `x.x.x.0/24`（256 个地址）。登录仍然需要私钥，风险不大，但范围变宽了，由负责人决定。**永远不要放行 `0.0.0.0/0`**（全世界），那样 22 端口会被扫描器整天敲门。
4. 关掉 VPN 再查 IP、再连。

**另一种 IP 变化**：实例本身停机再开机，它的公网 IP 也会变（见 5.4），所有人都要换成新地址。想固定，可以给实例绑一个 Elastic IP；注意 2024 年 2 月起 AWS 对每个公网 IPv4 地址按小时收费（约 0.005 美元 / 小时），绑着但实例停机时也收。

> [!INFO]+ CIDR：/32 和 /24
> `a.b.c.d/32` 表示前 32 位都固定，也就是**只有这一个地址**；`a.b.c.0/24` 表示前 24 位固定、最后一段 0 到 255 都算，共 256 个地址；`0.0.0.0/0` 表示所有地址。斜杠后面的数字越小，范围越大。
> `/32` fixes all 32 bits, i.e. exactly one address; `/24` fixes the first 24 bits, so the last number may be 0 to 255 (256 addresses); `/0` means every address. The smaller the number after the slash, the wider the range.

### 12.6 共用一台机器的规矩

> [!WARNING]+ 大家都是同一个 ubuntu 用户
> 所有人登录进来都是 `ubuntu`，权限完全一样：能看到、也能删掉、杀掉别人的文件和进程。所以靠约定：
> 1. **只有负责人开机、关机**。GPU 按小时收费，开机前在组里说一声；关机用 **Stop，不要 Terminate**（Terminate 会连硬盘一起删）。
> 2. **训练在 tmux 里跑**，其他人用 `tmux attach -r` 只读查看，不要在训练窗口里按 `Ctrl-c`。
> 3. **不要另外占用 GPU**。一块卡上同时跑两个训练，两个都会变慢，计时和显存也会互相干扰。
> 4. **不删别人的目录**，自己的试验放在自己名字的子目录里。
> 5. **有人退出项目，当天收回**他的公钥和 IP 规则。
> 6. 自动停机看门狗一直开着：没有训练进程、GPU 空闲 30 分钟就停机，忘了关也最多多花半小时的钱。
> Everyone logs in as the same `ubuntu` user with identical rights, so the rules are social: only the lead starts and stops the instance (Stop, never Terminate); training runs in tmux and others watch read-only; nobody starts a second GPU job; nobody deletes others' directories; remove a leaving member's key and IP rule the same day; the 30-minute idle watchdog stays on.

> [!NOTE]- English version
> **12 We got EC2 after all: one machine, shared over SSH.** Two days later the EC2 GPU quota was approved (us-east-2, 8 vCPUs) and we moved from SageMaker to a g5.2xlarge (one A10G 24 GiB, 8 vCPUs, 32 GiB), about 1.2 USD per hour, 20 to 30% cheaper than SageMaker for the same GPU. Training runs unattended in tmux, and a 30-minute idle watchdog (same logic as 11.6) starts at boot; it runs `sudo shutdown -h now`, which only stops the instance if its shutdown behavior is Stop (the default). With Terminate, an OS shutdown deletes the instance and its disk, so check it. The question: how can three people log in without passing private keys around?
>
> **12.1 Two locks.** The security group decides where you may connect from (listed IPs on port 22); `authorized_keys` on the machine decides who you are (only someone holding a private key that matches a listed public key). Giving a teammate access means adding her public key to `authorized_keys` and her IP to the security group; her private key never leaves her computer. The cards explain `authorized_keys` and why the launch `.pem` must not be shared.
>
> **12.2 Teammate: make your own key, once.** `ssh-keygen -t ed25519 -C "yourname-cs230" -f ~/.ssh/cs230_ec2_me` works in macOS Terminal, Linux and Windows PowerShell. A passphrase is optional and is typed invisibly. Keep the private file; send the one-line `.pub` content and your public IP from `curl https://checkip.amazonaws.com`. The cards explain ed25519, `-C`, `-f`, and public vs private IP addresses.
>
> **12.3 Lead: grant access.** Append the teammate's public key to `~/.ssh/authorized_keys` on the instance with a name and date comment, then allow `TEAMMATE_IP/32` on port 22 in the security group with a description. Our script checks the file really is a public key and refuses anything containing `PRIVATE KEY`. To revoke, delete her line from `authorized_keys` and the rule from the security group.
>
> **12.4 Teammate: log in and watch.** `ssh -i ~/.ssh/cs230_ec2_me ubuntu@INSTANCE_IP` (on Windows the path may be `$HOME\.ssh\cs230_ec2_me`; `-i` takes the private key). Answer `yes` on first connect. Then `tail` the log, `nvidia-smi`, or `tmux attach -r -t cs230` to watch read-only (leave with Ctrl-b then d), and `exit` to log out.
>
> **12.5 Changing IPs, the usual "cannot connect".** A timeout usually means your public IP changed: a different network, an ISP or campus address pool, or a VPN. Check the instance is running and the lead can connect, then send the new IP. If it keeps changing within one network, the lead may allow the whole `/24`; never allow `0.0.0.0/0`. The instance's own IP also changes after stop and start; an Elastic IP fixes it, but every public IPv4 address has been billed hourly since February 2024, even while the instance is stopped. The card explains `/32`, `/24` and `/0`.
>
> **12.6 House rules for a shared machine.** Everyone is the same `ubuntu` user, so: only the lead starts and stops the instance, and only with Stop; training runs in tmux and others attach read-only; no second GPU job; no deleting others' directories; revoke a leaving member the same day; keep the idle watchdog on.

<br>

## 13 排错表

| 问题 / Problem | 检查什么 / What to check |
| :--- | :--- |
| vCPU limit exceeded | 看 us-west-2 里 Running On-Demand G and VT instances 的**已应用**配额值，一台 g4dn.xlarge 需要 4 个空闲 vCPU。Check the applied quota in us-west-2; one g4dn.xlarge needs 4 free vCPUs. |
| Insufficient capacity | 换同区域的另一个可用区，或者过一会再试，配额批了不代表有货。Try another AZ in the same region or retry later; quota does not guarantee capacity. |
| SSH 超时 / SSH times out | 实例状态、状态检查、当前公网地址、公有子网路由、安全组是否允许你**当前** IP 的入站 SSH。Instance state, status checks, current address, public subnet route, inbound SSH from your current IP. |
| Permission denied (publickey) | 用户名用 ubuntu，私钥要配对；macOS / Linux / WSL 检查 `chmod 400`。组员：公钥是否已加进 `authorized_keys`，`-i` 后面是不是误用了 `.pub`。Use ubuntu and the matching key; check `chmod 400`. Teammates: is your public key in `authorized_keys`, and did you pass the private key (not `.pub`) to `-i`? |
| 昨天能连、今天超时 / Worked yesterday, times out today | 多半是你的公网 IP 变了（换网络、地址池轮换、VPN），`curl https://checkip.amazonaws.com` 查新 IP 交给负责人加进安全组（12.5）。Your public IP probably changed; send the new one to the lead (12.5). |
| `Load key ...: error in libcrypto`（Windows） | 私钥被存成了 Windows 换行（CRLF），在 Git Bash 里 `sed -i 's/$//' 私钥文件`。The key file got CRLF line endings; strip them. |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | 同一个 IP 换了一台新实例，指纹变了。确认是自己的新机器后 `ssh-keygen -R INSTANCE_IP` 删掉旧记录。A new instance reuses the address; after confirming, remove the old entry with `ssh-keygen -R`. |
| nvidia-smi 失败 / fails | 确认机型带 GPU、用的是指定的 NVIDIA 驱动 AMI。先修驱动，再动 Python 包。Confirm a GPU instance and the intended driver AMI; fix the driver before touching Python packages. |
| TensorFlow 看不到 GPU | 用 `.venv/bin/python` 跑检查，重装 `tensorflow[and-cuda]`，再看 TensorFlow 官方 GPU 排错。Run the check with `.venv/bin/python`, reinstall the CUDA extra, then see TensorFlow's troubleshooting. |
| Import / 依赖错误 | 检查固定的仓库版本和 Python 3.12 环境，不要把 SciPy 降到 1.1。Check the pinned revision and Python 3.12; do not downgrade SciPy to 1.1. |
| No space left on device | `df -h /`，清包缓存，必要时扩容 EBS 和文件系统。Run `df -h /`, clear caches, expand EBS and the filesystem. |
| Jupyter 打不开 | 服务和隧道都在跑、端口对得上、token 正确。Server and tunnel running, ports match, correct token. |
| （SageMaker）设置没保存 | 网页改了没点 Run；用 `aws sagemaker update-space` 写进空间。Edited without Run; persist with `update-space`. |
| （SageMaker）`aws sso login` 后 Forbidden | 项目门户账号用 `aws login --profile myproj`。Projects-portal users should use `aws login`. |

找人求助时附上：区域、AMI 名字、机型、完整报错、`nvidia-smi` 和 TensorFlow GPU 检查的输出。**永远不要贴私钥、AWS Access Key、密码、Jupyter token、GitHub token。**

> [!NOTE]- English version
> **13 Troubleshooting.** The table is bilingual. When asking someone for help, include the region, AMI name, instance type, the exact error, and the output of `nvidia-smi` and the TensorFlow GPU check. Never post private keys, AWS access keys, passwords, Jupyter tokens or GitHub tokens.

<br>

## 14 一页纸清单 / One-page checklist

- [ ] credits 兑换了，余额和过期时间看过了，预算提醒设了 / Credits redeemed, balance and expiry checked, budget alert set
- [ ] 区域没选错，GPU 配额已应用 ≥ 4 vCPU / Right region, applied GPU quota ≥ 4 vCPUs
- [ ] AMI 是 Amazon 发布的 x86_64 Deep Learning Base 版 / AMI is Amazon's x86_64 Deep Learning Base
- [ ] 安全组只开 22，来源 My IP；8888 没开 / Only port 22 from My IP; 8888 closed
- [ ] .pem 权限 400，备份在安全的地方，没进 Git / .pem at 400, backed up safely, not in Git
- [ ] `nvidia-smi` 正常，TensorFlow GPU 列表非空 / `nvidia-smi` works, TensorFlow sees the GPU
- [ ] 训练在 tmux 里 / Training runs in tmux
- [ ] 结果拷回本地或推到 GitHub / Results copied back or pushed
- [ ] （多人共用）每人一对密钥，只收公钥；安全组规则写了名字和日期；没有 `0.0.0.0/0` / (Shared) one key pair per person, public keys only; rules carry name and date; no `0.0.0.0/0`
- [ ] （多人共用）Shutdown behavior 是 Stop，空闲看门狗在运行 / (Shared) shutdown behavior is Stop, idle watchdog running
- [ ] **实例已 Stop，控制台显示 Stopped** / **Instance stopped and shown as Stopped**
