---
title: PyTorch小土堆课程笔记
date: 2023-09-19
categories:
- 学CS/SE
tags:
- 编程语言
desc: 本文以 B 站小土堆 PyTorch 教程为参考，整理 PyTorch 入门要点。
---

- 参考：
    - <a href="https://github.com/xiaotudui/pytorch-tutorial">小土堆学Pytorch</a>
- 其他笔记：
    - <a href="https://blog.csdn.net/Ever_____/article/details/136596213">【Pytorch入门】小土堆PyTorch入门教程完整学习笔记-CSDN</a>
    - <a href="https://zhuanlan.zhihu.com/p/634866046">Pytorch tutorial（小土堆）学习笔记-知乎</a>

<br>

配套视频：[PyTorch深度学习快速入门教程（绝对通俗易懂！）](https://www.bilibili.com/video/BV1hE411t7RN/)（B 站 `BV1hE411t7RN`，约 33 讲）。官方仓库代码均在 [`src/`](https://github.com/xiaotudui/pytorch-tutorial/tree/master/src)。下列笔记按视频顺序整理，源码以仓库为准。

| 模块 | 对应分 P | 仓库文件 |
| :--- | :--- | :--- |
| 环境与查阅 | P1–P5 | — |
| Dataset | P6–P7 | `read_data.py` |
| TensorBoard | P8–P9 | `P8_Tensorboard.py` |
| Transforms / CIFAR-10 | P10–P14 | `P9_transforms.py`、`P10_dataset_transform.py` |
| DataLoader | P15 | `dataloader.py` |
| `nn.Module` 与卷积 | P16–P18 | `nn_module.py`、`nn_conv.py`、`nn_conv2d.py` |
| 池化 / 激活 / 线性 | P19–P21 | `nn_maxpool.py`、`nn_relu.py`、`nn_linear.py` |
| Sequential | P22 | `nn_seq.py` |
| 损失与优化 | P23–P24 | `nn_loss.py`、`nn_loss_network.py`、`nn_optim.py` |
| 预训练与保存 | P25–P26 | `model_pretrained.py`、`model_save.py`、`model_load.py` |
| 完整训练 | P27–P29 | `model.py`、`train.py`、`train-cpu.py` |
| GPU 与验证 | P30–P32 | `train_gpu_1.py`、`train_gpu_2.py`、`test.py` |

> [!NOTE]+ 文件名与分 P
>
> 仓库里的 `P8_Tensorboard.py`、`P9_transforms.py`、`P10_dataset_transform.py` 与视频分 P **不是**严格同号：视频 P8–P9 对应 TensorBoard，P10–P13 对应 Transforms，P14 对应 CIFAR 内置集。后半脚本不再带 `Pxx` 前缀。

### 整体脉络：整门课在做哪一件事

整门课其实只讲了一件事：**用 CIFAR-10（10 类、32×32 的小彩图）训练一个能分类的卷积网络，并学会把它保存、搬上 GPU、拿新图片验证**。前面的每一 P 都是这条流水线上的一个零件。先把全图记住，后面每学一个零件就知道它装在哪里。

```text
 ┌─────────────────────────────────────────────────────────────────────┐
 │ 0. 准备工作：装环境、学会查文档        → 「环境」「查阅」  P1–P5    │
 └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ 1. 数据集 Dataset    │──▶│ 2. 变换 transforms   │──▶│ 3. 打包 DataLoader   │
 │ 按下标取一条(图,标签)│   │ PIL 图 → Tensor      │   │ 单条 → batch         │
 │ 本文「数据集」P6–P7  │   │ 归一化/裁剪 P10–P14  │   │ [64,3,32,32]   P15   │
 └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
            ▲                                                     │
            │ 随时用 TensorBoard 看图、看曲线                      ▼
            │ 本文「看板」P8–P9                    ┌──────────────────────────┐
            │                                      │ 4. 搭模型 nn.Module       │
            │                                      │ 卷积/池化/激活/线性/      │
            │                                      │ Sequential     P16–P22    │
            │                                      └──────────────────────────┘
            │                                                     │ 前向：outputs = model(imgs)
            │                                                     ▼
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
 │ 6. 优化器 optim      │◀──│ 反向传播 backward()  │◀──│ 5. 损失 loss             │
 │ step() 更新参数 P24  │   │ 算出每个参数的梯度   │   │ CrossEntropyLoss P23     │
 └──────────────────────┘   └──────────────────────┘   └──────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │ 7. 训练循环：for epoch → for batch → 前向→损失→清梯度→反传→更新     │
 │    每轮在测试集上算 loss / 准确率，写进 TensorBoard        P27–P29  │
 └─────────────────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ 8. 保存/加载模型     │──▶│ 9. 搬到 GPU          │──▶│ 10. 验证 test.py     │
 │ 预训练 VGG、         │   │ .cuda() / .to(device)│   │ 拿一张新图片,        │
 │ torch.save/load      │   │ 模型+数据+损失都要搬 │   │ 预测它属于哪一类     │
 │          P25–P26     │   │          P30–P31     │   │          P32         │
 └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

按「数据从哪来 → 在哪变形 → 去哪计算」再串一遍，每一步数据长什么样：

| 步骤 | 做什么 | 数据的样子（以 CIFAR-10 为例） | 本文位置 | 分 P |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 装环境、学 `dir` / `help` | — | 「环境」「查阅」 | P1–P5 |
| 1 | `Dataset[i]` 取一条样本 | `(PIL 图 32×32, 整数标签 3)` | 「数据集」 | P6–P7 |
| — | 用 TensorBoard 检查数据和曲线 | 事件文件 `logs/events.out.tfevents.*` | 「看板」 | P8–P9 |
| 2 | `transforms` 把图变成张量 | `Tensor [3, 32, 32]`，数值 0–1 | Transforms 一节 | P10–P14 |
| 3 | `DataLoader` 组 batch | `imgs [64, 3, 32, 32]`，`targets [64]` | DataLoader 一节 | P15 |
| 4 | `nn.Module` 前向计算 | `[64, 3, 32, 32] → [64, 10]`（10 类的分数） | 网络各节 | P16–P22 |
| 5 | 损失函数比较预测与标签 | `[64, 10]` + `[64]` → 一个标量 loss | 损失一节 | P23 |
| 6 | 反向传播 + 优化器更新 | 每个参数得到 `.grad`，再被 `step()` 改写 | 优化器一节 | P24 |
| 7 | 训练循环 | 重复 1–6，记录每轮 loss / 准确率 | 完整训练 | P27–P29 |
| 8 | 保存 / 加载 | `.pth` 文件 | 模型保存 | P25–P26 |
| 9 | GPU | 张量 `device='cuda:0'` | GPU 训练 | P30–P31 |
| 10 | 验证新图片 | `[1, 3, 32, 32] → argmax → "dog"` | 模型验证 | P32 |

> [!TIP]+ 怎么用这张图
>
> - 学到任何一个 API 时，先问一句「它在第几步？输入、输出分别是什么形状？」。
> - 报错时也按这张图定位：`FileNotFoundError` 多半在第 1 步；`TypeError ... found PIL` 多半是第 2 步漏了 `ToTensor`；`shapes cannot be multiplied` 是第 4 步层与层尺寸没对上；`Expected all tensors to be on the same device` 是第 9 步漏搬了东西。
> - 本文这一块（环境、查阅、数据集、看板）只覆盖第 0、1 步和 TensorBoard，后面的步骤在后续各节展开。

<br>

## 环境搭建：装好 PyTorch、选好编辑器（P1–P3、P5）

> [!NOTE]+ 这一节要解决什么
>
> 目的：得到一个「能 `import torch`、知道自己有没有 GPU」的 Python 环境，并选一个写代码的工具。它是整条流水线（数据 → 变换 → 打包 → 模型 → 损失 → 优化 → 训练循环 → 保存/验证）开工前的第 0 步；下一节「查阅」教的是在这个环境里怎么查 API。

对应视频 P1–P3、P5。目标是得到一个能 `import torch`、必要时能用 GPU 的解释器，并选好写代码的界面。

几个先要分清的名词：

| 名词 | 是什么 | 类比 |
| :--- | :--- | :--- |
| 解释器（interpreter） | 某一个 `python.exe`，代码最终由它执行 | 一台具体的机器 |
| 环境（environment） | 一个解释器 + 它装过的包的集合，conda 可以建很多个互不干扰的环境 | 一个独立的工具间 |
| CUDA | NVIDIA 显卡的通用计算平台；PyTorch 的 GPU 版靠它在显卡上算 | 显卡的“驱动接口” |
| 驱动（driver） | 装在系统里的显卡驱动，决定最高能支持哪个 CUDA 版本 | 地基，越新兼容越广 |

**搭环境的 5 步：**

1. 装 Anaconda 或 Miniconda（P1）：得到 `conda` 命令，用来建独立环境。
2. 新建并进入环境：`conda create -n pytorch python=3.10`，再 `conda activate pytorch`。之后装的包只进这个环境。
3. 到官网生成安装命令并执行（见下方「安装」）。有 NVIDIA 显卡就选 CUDA 版，没有就选 CPU 版。
4. 用下面的检查脚本确认 `import torch` 正常、`cuda.is_available()` 符合预期。
5. 在 PyCharm / Jupyter 里把解释器指到这个环境（P2–P3、P5），否则 IDE 用的可能是另一个没装 torch 的 Python。

```bash
conda create -n pytorch python=3.10   # 新建名为 pytorch 的环境（名字随意）
conda activate pytorch                # 进入环境，命令行前缀会变成 (pytorch)
conda env list                        # 列出所有环境，带 * 的是当前环境
python -c "import torch; print(torch.__version__)"   # 命令行快速检查
```

### 安装 PyTorch 并检查 GPU

到 [pytorch.org](https://pytorch.org/) 按操作系统、包管理器（conda / pip）、CUDA 版本生成命令。直接执行无 CUDA 标记的 `pip install torch` 时，默认源经常装到 CPU 版或与本机 CUDA 不匹配的构建。

> [!NOTE]+ 补充：不同系统上 `pip install torch` 的默认结果
>
> - Windows / macOS：PyPI 上的默认 wheel 是 **CPU 版**（macOS 可以用 Apple 芯片的 `mps` 加速，但不是 CUDA）。
> - Linux：PyPI 默认 wheel 一般**自带某个 CUDA 版本**的运行库，但不一定和本机驱动匹配。
> - 因此最稳的做法始终是复制官网给出的、带 `--index-url https://download.pytorch.org/whl/cu1xx` 的命令。
> - 官网的 CUDA 版本（如 cu118、cu121、cu124）只要求**驱动足够新**，不要求本机另外装同版本的 CUDA Toolkit；`nvidia-smi` 右上角的 `CUDA Version` 是驱动支持的**最高**版本，选不超过它的即可。

常见检查：

```python
import torch
print(torch.__version__)  # 版本号，例如 2.6.0+cpu（+cpu 表示 CPU 版）或 2.6.0+cu124（CUDA 12.4 版）
print(torch.cuda.is_available())  # True = 这个解释器能用 GPU；纯 CPU 机器上输出 False
print(torch.cuda.device_count())  # 可用 GPU 数量；纯 CPU 机器上输出 0
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # 第 0 块显卡的型号，如 NVIDIA GeForce RTX 3060
```

在本机 CPU 版 PyTorch 2.6 上实际输出：

```text
2.6.0+cpu
False
0
```

- `import torch` 无报错：CPU 版至少装好
- `torch.cuda.is_available()` 为 `True`：当前这个解释器能调 CUDA

`False` 的常见原因（视频 FAQ）：

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| 能 import，CUDA 为 False | 装的是 CPU wheel | 卸载后按官网 CUDA 命令重装 |
| 有 NVIDIA 显卡仍为 False | 驱动过旧，或 PyTorch 编译所用 CUDA 与驱动不兼容 | 升级驱动，或改选官网给出的 CUDA 11.x / 12.x 组合 |
| 同一台机器有的环境 True、有的 False | PyCharm / Jupyter 没用到装过 GPU 版的那个 conda 环境 | 在 IDE 里把 Interpreter 指到该环境 |

**排查 `False` 的 4 步：**

1. 看版本后缀：`print(torch.__version__)` 带 `+cpu` 就是 CPU 版，后面不用查了，直接重装 GPU 版。
2. 看驱动：命令行执行 `nvidia-smi`。命令不存在或报错，说明驱动没装好；能显示表格就记下右上角的 `CUDA Version`。
3. 看对得上不：`torch.version.cuda`（如 `'12.4'`）不能高于 `nvidia-smi` 显示的版本，高了就升级驱动或换低一档的 cu 版本。
4. 看是不是同一个 Python：`import sys; print(sys.executable)` 打印当前解释器路径，确认它就是装了 GPU 版的那个 conda 环境。

**常见报错与排查：**

| 报错 | 原因 | 修法 |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'torch'` | 当前解释器里没装 torch；最常见是装在 A 环境、IDE 用的是 B 环境 | `print(sys.executable)` 核对路径；在 IDE 里切换解释器，或在当前环境重装 |
| `OSError: [WinError 126] 找不到指定的模块 ... fbgemm.dll` 之类 | Windows 缺 VC++ 运行库，或安装不完整 | 装最新 Microsoft Visual C++ Redistributable，再重装 torch |
| `'conda' 不是内部或外部命令` | Anaconda 没加入 PATH | 用开始菜单里的 Anaconda Prompt，或把 conda 加入 PATH（见下方「环境变量」） |
| `pip install` 下载极慢 / 超时 | 网络问题 | 换国内镜像（`-i` 指定），但 GPU 版要确认镜像里确有对应 cu 版本的 wheel |

本地已下载的 conda 包可用：

```bash
conda install --use-local 包名.tar.bz2
```

> [!NOTE]+ 补充：离线安装本地包
>
> `--use-local` 的含义是「把本机 conda-build 产出的包目录也当作一个频道」。如果手里只是一个下载好的 `.tar.bz2` / `.conda` 文件，更直接的写法是给出文件路径：`conda install ./包名.tar.bz2`。这种方式**不会自动安装依赖**，缺什么依赖需要再手动装。

> [!NOTE]+ 环境变量
>
> 将 Python、Scripts、CUDA 的 `bin` 加入 PATH 后，命令行不必先 `cd` 到该目录即可调用 `python`、`pip`、`nvcc`。视频中的 `nb_conda` 用于在 Notebook 里切换 conda 环境；较新的 Jupyter 直接在界面选择 kernel 即可，该扩展已非必需。

让 Jupyter 能选到某个 conda 环境的现代做法（不需要 `nb_conda`）：

```bash
conda activate pytorch                                    # 先进入目标环境
pip install ipykernel                                     # 在该环境装 kernel 支持
python -m ipykernel install --user --name pytorch         # 注册成名为 pytorch 的 kernel
jupyter notebook                                          # 启动后在 Kernel → Change kernel 里选 pytorch
```

### 编辑器：PyCharm 与 Jupyter 怎么选

对比 PyCharm 与 Jupyter：

| | PyCharm | Jupyter |
| :--- | :--- | :--- |
| 适合 | 多文件项目、跳转源码、断点 | 逐步试 API、看中间 tensor |
| 运行 | 整文件 / 配置好的运行项 | 按单元格，左侧数字表示已跑完，`*` 表示仍在跑 |
| 多行试验 | Python Console 里 Shift+Enter 换行 | 单元格内直接换行 |
| 跳源码 | Ctrl 点击符号；Alt+7 结构视图；Ctrl+P 看参数 | `对象??` 显示源码 |

视频里的用法是两者混用：**在 PyCharm 的 Python Console 或 Jupyter 里试一行 API、看 shape，确认无误再写进 `.py` 文件**。Python Console 的好处是右侧变量面板能直接看到每个变量的类型和形状，等价于随手 `print(type(x), x.shape)`。

PyCharm 里给项目指定 conda 环境的步骤：

1. `File → Settings → Project: 项目名 → Python Interpreter`。
2. 点右上角齿轮 / `Add Interpreter` → 选 `Conda Environment` → `Existing environment`。
3. 指到该环境的 `python.exe`（Windows 上通常是 `Anaconda安装目录\envs\pytorch\python.exe`）。
4. 确定后，下方包列表里应能看到 `torch`；新开的 Python Console 里 `import torch` 不报错即成功。

相对路径以**当前工作目录**为基准，不是以 `.py` 文件所在目录为基准（除非先改 cwd）。例如脚本在 `Project/train/A.py`，图片在 `Project/image/B.jpg`，若 cwd 为 `train/`，则图片相对路径为 `../image/B.jpg`。

当前工作目录（current working directory，cwd）随启动方式变化：

| 启动方式 | cwd 通常是 |
| :--- | :--- |
| 命令行 `python train/A.py` | 敲命令时所在的目录（如 `Project/`） |
| PyCharm 右键 Run | 运行配置里的 `Working directory`，默认是该脚本所在目录 |
| PyCharm Python Console | 项目根目录 |
| Jupyter | `.ipynb` 文件所在目录 |

同一行 `Image.open("dataset/train/ants/0013035.jpg")` 换一种启动方式就可能 `FileNotFoundError`。排查时先打印 cwd，或者干脆用脚本自身位置拼出绝对路径：

```python
import os
print(os.getcwd())                                   # 当前工作目录，相对路径从这里算起
here = os.path.dirname(os.path.abspath(__file__))    # 本脚本所在目录（Jupyter 里没有 __file__，不能这样用）
img_path = os.path.join(here, "..", "image", "B.jpg")  # 不管从哪启动，都指向 Project/image/B.jpg
print(os.path.exists(img_path))                      # 先确认文件存在，再交给 Image.open
```

课程后续代码大量依赖「能打开源码查看 `__init__` / `forward`」，因此至少保证有一种方式能打开 PyTorch 源文件。

> [!TIP]+ 小结
>
> - 用 conda 建独立环境，装 PyTorch 一律复制官网命令；有卡选 CUDA 版，没卡选 CPU 版。
> - 三行自检：`torch.__version__`（看 `+cpu` / `+cu124`）、`torch.cuda.is_available()`、`sys.executable`。
> - `is_available()` 为 `False`：先看是不是 CPU 版，再看驱动版本，最后看 IDE 用的是不是同一个解释器。
> - 相对路径从 cwd 算起，不是从脚本位置算起；路径报错先 `print(os.getcwd())`。
> - PyCharm 适合写工程和跳源码，Jupyter / Python Console 适合一行行试 API、看 shape。

<br>

## 两大法宝：dir 与 help 查 API（P4）

> [!NOTE]+ 这一节要解决什么
>
> 目的：遇到不认识的模块、函数时，能自己查出「里面有什么」「怎么用」「输入输出是什么」。它不属于训练流水线的某一步，而是贯穿全程的工具：上一节装好了环境，这一节学会在环境里查东西，下一节「数据集」马上就用 `help(Dataset)` 去看要实现哪些方法。

对应视频 P4。文档和源码比记忆 API 列表更重要：课程后半所有层、损失、优化器都按同一套办法查。

### dir：打开工具箱看有什么

`dir(obj)` 列出对象的属性与方法名，相当于打开工具箱。不传参数时列出当前作用域中的名字。

```python
import torch

dir(torch)  # 返回字符串列表，几百个名字：'cuda'、'nn'、'tensor'、'zeros' ...
dir(torch.cuda)  # 进一层：'is_available'、'device_count'、'get_device_name' ...
dir(torch.cuda.is_available)  # 注意：后面不加括号，查的是函数对象本身
```

`dir(torch.cuda.is_available())` 会先**调用**函数（返回 `True`/`False`），再对布尔值做 `dir`，看到的就不是该函数的属性了。

`dir` 的结果很长，可以用列表推导筛一下：

```python
import torch
names = [n for n in dir(torch.cuda) if not n.startswith("_")]  # 去掉下划线开头的内部名字
print(names[:5])                                                # 只看前 5 个
print([n for n in dir(torch) if "conv" in n.lower()])            # 找名字里带 conv 的，如 'conv1d'、'conv2d' ...
```

双下划线包裹的名字是特殊方法（magic / dunder），例如 `__init__`、`__call__`、`__getitem__`、`__len__`，用来实现构造、当函数调用、下标访问、`len()` 等语法。自己写类时按需要去实现它们（下一节的自定义 Dataset 就要实现 `__getitem__` 和 `__len__`）；但不要去改别人对象（如 `torch` 内部对象）上的 dunder，也不要拿 `__xxx__` 当普通变量名。

> [!WARNING]+ 易错：dunder 不是“不能碰”
>
> Python 语法上并**不禁止**定义或重写 dunder 方法，它们是语法预留的「钩子」：类里实现了 `__getitem__`，对象就能写 `obj[0]`；实现了 `__call__`，对象就能写 `obj(x)`。容易误以为 dunder 一律不能动，其实自己的类该写就要写，只是不要随手改库里对象的。

几个后面一定会碰到的 dunder 与它们背后的语法：

| 写法 | Python 实际调用 | 课程里在哪出现 |
| :--- | :--- | :--- |
| `MyData(root, "ants")` | `MyData.__init__(self, root, "ants")` | 自定义 Dataset |
| `dataset[0]` | `dataset.__getitem__(0)` | 自定义 Dataset |
| `len(dataset)` | `dataset.__len__()` | 自定义 Dataset、DataLoader |
| `ants + bees` | `ants.__add__(bees)` → 得到 `ConcatDataset` | 拼接数据集 |
| `trans(img)` / `model(x)` | `trans.__call__(img)` / `model.__call__(x)`（内部再调 `forward`） | transforms、nn.Module |

### help：读说明书

`help(obj)` 打印文档字符串，相当于说明书。

```python
import torch
import torch.utils.tensorboard   # tensorboard 子模块不会随 import torch 自动导入，必须单独 import

help(torch.cuda.is_available)
help(torch.utils.data.Dataset)
help(torch.utils.tensorboard.SummaryWriter.add_image)
```

> [!WARNING]+ 易错：子模块要单独 import
>
> 如果只写了 `import torch`、没有 `import torch.utils.tensorboard`，最后一行会报 `AttributeError: module 'torch.utils' has no attribute 'tensorboard'`：`torch` 启动时会自动导入 `torch.utils.data`，但**不会**自动导入 `torch.utils.tensorboard`（已在 PyTorch 2.6 上实测）。写成 `from torch.utils.tensorboard import SummaryWriter` 之后 `help(SummaryWriter.add_image)` 也可以。

读一份 `help` 输出的 4 步（以 `help(SummaryWriter.add_image)` 为例）：

1. **看签名**：`add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')`。没有默认值的（`tag`、`img_tensor`）必须传，有默认值的可以省。
2. **看 Args**：每个参数要什么类型。这里写着 `img_tensor (torch.Tensor, numpy.ndarray, or string/blobname)`。
3. **看 Shape**：输入形状。这里写着默认 `(3, H, W)`，HWC 要配合 `dataformats` 参数。
4. **看 Examples**：文档末尾通常有可直接复制运行的小例子，先跑通例子再改成自己的数据。

不想看整段输出时，也可以只看签名：

```python
import inspect
from torch.utils.tensorboard import SummaryWriter
print(inspect.signature(SummaryWriter.add_image))
# (self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

在 IPython / Jupyter 中：

- `对象?`：文档
- `对象??`：尽量显示源码

PyCharm：`Ctrl` 点击跳进定义；`Alt+7` 结构视图；`Ctrl+P` 弹出当前调用的参数列表。看 transforms / `nn.Conv2d` 时，先看 `__init__` 要哪些构造参数，再看 `forward` / `__call__` 运行时接受何种形状的输入。

> [!EXAMPLE]+ 查阅路径
>
> 把 `torch` 看成多层工具箱：`dir(torch)` 看到分区（如 `cuda`），`dir(torch.cuda)` 看到工具（如 `is_available`），`help(torch.cuda.is_available)` 看到该工具的说明。后续 `Dataset`、`SummaryWriter.add_image`、`nn.Conv2d` 均按同一方式查阅。
>
> 经验规则：先确认**输入类型与形状**，再确认**输出类型与形状**；未知时 `print(type(x))`、`print(x.shape)`，或在 `__getitem__` / `forward` 里打断点。

除了 `dir` / `help`，还有两个随手可用的「探针」：

| 工具 | 回答的问题 | 例子 |
| :--- | :--- | :--- |
| `type(x)` | 它是什么类型？ | `<class 'torch.Tensor'>`、`<class 'numpy.ndarray'>` |
| `x.shape` / `x.size` | 多大？（Tensor / ndarray 用 `.shape`；PIL 图用 `.size`，顺序是 `(W, H)`） | `torch.Size([3, 512, 768])` |
| `x.dtype` | 元素是什么数值类型？ | `torch.float32`、`uint8` |
| 官方文档网页 | 参数含义、公式、示例 | `pytorch.org/docs/stable/` 里搜类名 |

> [!TIP]+ 小结
>
> - `dir(x)` 看「有什么」，`help(x)` 看「怎么用」；查函数时**不要加括号**，否则查的是返回值。
> - `help` 输出按「签名 → Args → Shape → Examples」的顺序读。
> - `torch.utils.tensorboard` 需要显式 import 后才能查。
> - dunder 方法是语法钩子：`[]` 对应 `__getitem__`、`len()` 对应 `__len__`、`()` 对应 `__call__`。
> - 任何时候拿不准，就 `print(type(x), x.shape)`。

<br>

## 读数据：自定义 Dataset（P6–P7）

> [!NOTE]+ 这一节要解决什么
>
> 目的：让 PyTorch 能「按编号取出第 i 张图片和它的标签」。这是训练流水线的第 1 步（**数据** → 变换 → 打包 → 模型 → …）；上一节学的 `help` 在这里第一次派上用场，下一节「看板」用 TensorBoard 把取出的图片画出来检查，再往后 Transforms 把 PIL 图变成张量、DataLoader 把单条样本打成 batch。

对应视频 P6–P7。数据从磁盘到网络分成两步：

- **Dataset**：把样本编成 `0, 1, 2, …`，按编号取出「一条数据 + 标签」
- **DataLoader**：把 Dataset 打成 mini-batch，并可打乱、多进程加载

Dataset 按索引提供单条样本；DataLoader 负责组 batch、打乱与多进程读取。本节只实现 Dataset；P15 再讲打包。

用一个比喻记住两者分工：

```text
磁盘上的图片文件
      │  Dataset：图书管理员，知道第 i 本书在哪个架子，一次只拿一本
      ▼
dataset[0] → (img, label)   dataset[1] → (img, label)   ...
      │  DataLoader：搬运工，一次搬 batch_size 本，可以先洗牌再搬
      ▼
(imgs [B, C, H, W], labels [B])  →  送进模型
```

### Dataset 接口：必须实现哪几个方法

```python
from torch.utils.data import Dataset
help(Dataset)
```

`Dataset` 是抽象类。自定义类要继承它，并实现下面三个方法（严格必需的只有 `__getitem__`，但 `__len__` 实际上也总要写，见下方提示）：

| 方法 | 作用 | 支撑的语法 |
| :--- | :--- | :--- |
| `__init__` | 记录根目录、类别名、文件列表等**全局**信息 | `MyData(...)` |
| `__getitem__(idx)` | 按索引读**一条**样本 | `dataset[0]` |
| `__len__` | 返回样本总数 | `len(dataset)` |

> [!WARNING]+ 易错：`__len__` 能不能省
>
> 按官方文档，**唯一强制**要求子类覆盖的是 `__getitem__`；`__len__` 是「可选，但很多 Sampler 和 DataLoader 的默认设置都依赖它」。没写 `__len__` 时，`len(dataset)` 会报 `TypeError: object of type 'NoLen' has no len()`（已实测），`DataLoader(..., shuffle=True)` 也无法工作。所以写代码时**三个都实现**。

`__init__` 里只做「列清单」，不宜把全部图片读进内存。真正读盘发生在 `__getitem__`，DataLoader 取第 `i` 张时才打开那张文件。

算一下为什么不宜全读进内存：一张缩放到 256×256 的 RGB 图转成 float32 张量占

$$
3 \times 256 \times 256 \times 4\ \text{字节} = 786432\ \text{字节} \approx 0.75\ \text{MB}
$$

蚂蚁/蜜蜂训练集共 245 张，全读进来约 $245 \times 0.75 \approx 184$ MB，还能接受；但 ImageNet 有 128 万张，同样算法要 900 多 GB，根本放不下。所以惯例是 `__init__` 只存**文件名列表**（几 KB），`__getitem__` 用到哪张读哪张。

**写一个自定义 Dataset 的 4 步：**

1. **继承**：`class MyData(Dataset):`，这样 `+` 拼接、DataLoader 等功能都能直接用。
2. **`__init__` 列清单**：存根目录、子目录、`os.listdir` 得到的文件名列表（必要时 `sort()`），以及要用的 `transform`。
3. **`__getitem__(idx)` 取一条**：用 `idx` 从清单里取文件名 → 拼完整路径 → `Image.open` 读图 → 找到对应标签 →（可选）做 transform → `return img, label`。
4. **`__len__` 报数量**：`return len(文件名列表)`。

### self 与路径拼接

`self.xxx` 是实例属性，在 `__init__` 里赋值后，`__getitem__` / `__len__` 都能用。`os.path.join` 按操作系统选择 `/` 或 `\`，避免手写斜杠在 Windows / Linux 间出错。`os.listdir(path)` 返回该目录下文件名列表（无序，需要标签文件与图像一一对应时要 `sort()`）。

几个路径函数的输入输出（以 Windows 为例）：

```python
import os
root_dir = "dataset/train"
label_dir = "ants"
path = os.path.join(root_dir, label_dir)   # 'dataset/train\\ants'：自动补上分隔符
names = os.listdir(path)                   # ['0013035.jpg', '1030023514_aad5c608f9.jpg', ...]，只有文件名，不含目录
full = os.path.join(path, names[0])        # 'dataset/train\\ants\\0013035.jpg'：可以直接交给 Image.open
print(os.path.exists(full))                # True 才说明路径拼对了
```

注意 `os.listdir` 返回的**只是文件名**，必须再和目录 `join` 一次才是能打开的路径；直接 `Image.open(names[0])` 会在 cwd 下找这个文件，报 `FileNotFoundError`。

视频用的蚂蚁/蜜蜂目录（文件夹名即标签）：

```text
dataset/train/
    ants/   0013035.jpg ...
    bees/   16838648_415acd9e3f.jpg ...
dataset/val/
    ants/
    bees/
```

这份数据来自 PyTorch 官方迁移学习教程的 `hymenoptera_data`（膜翅目昆虫）：训练集约 124 张蚂蚁、121 张蜜蜂，验证集各约 70–80 张。图片尺寸**各不相同**（如 500×375、768×512），所以后面必须 `Resize` 成统一大小才能组成 batch。

### 视频版：文件夹名就是标签

视频演示用的蚂蚁/蜜蜂数据集：根目录下按类别建文件夹，**文件夹名即标签**。`os.path.join` 按操作系统自动选择路径分隔符。

```python
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):  # 创建实例时只调用一次，例如 MyData("dataset/train", "ants")
        self.root_dir = root_dir  # "dataset/train"：存成实例属性，其他方法用 self.root_dir 访问
        self.label_dir = label_dir  # "ants"：既是子文件夹名，也直接当作标签
        self.path = os.path.join(self.root_dir, self.label_dir)  # "dataset/train\\ants"
        self.img_path = os.listdir(self.path)  # 文件名列表 ['0013035.jpg', ...]，只是“清单”，没读任何图片

    def __getitem__(self, idx):  # dataset[idx] 时被调用，每次只处理一张
        img_name = self.img_path[idx]  # 第 idx 个文件名，如 '0013035.jpg'
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 拼出完整路径
        img = Image.open(img_item_path)  # 真正读盘：得到 PIL 图像（JpegImageFile），size=(W, H)
        label = self.label_dir  # 标签是字符串 "ants"（之后训练时要换成整数，见下文）
        return img, label  # 返回元组 (PIL图, str)

    def __len__(self):  # len(dataset) 时被调用
        return len(self.img_path)  # 有多少文件就有多少样本


root_dir = "dataset/train"  # 相对路径：从 cwd 算起，cwd 不对就会 FileNotFoundError
ants_dataset = MyData(root_dir, "ants")  # 蚂蚁子集，下标 0..123
bees_dataset = MyData(root_dir, "bees")  # 蜜蜂子集，下标 0..120
img, label = ants_dataset[0]  # 等价于 MyData.__getitem__(ants_dataset, 0)，元组拆包
print(label)  # ants
# img.show()

train_dataset = ants_dataset + bees_dataset  # Dataset 支持相加拼接；结果类型是 ConcatDataset
print(len(ants_dataset), len(bees_dataset), len(train_dataset))  # 完整数据集上为 124 121 245
```

执行 `ants_dataset[0]` 时的顺序：

1. 解释器调用 `MyData.__getitem__(ants_dataset, 0)`
2. `img_name = self.img_path[0]`
3. `os.path.join` 拼出完整路径
4. `Image.open` 得到 PIL 图像；`label` 取文件夹名 `"ants"`
5. 返回 `(img, label)` 元组

`img.show()` 可弹出系统看图窗口。`ants_dataset + bees_dataset` 得到 `ConcatDataset`：下标先走 ants，再走 bees，长度相加，顺序不变。

`ConcatDataset` 如何把一个总下标换算成「哪个子集的第几个」：它内部存了累计长度 `cumulative_sizes`。以 ants 124 张、bees 121 张为例：

| 总下标 `i` | 比较 `cumulative_sizes = [124, 245]` | 实际取的是 |
| :--- | :--- | :--- |
| 0 | $0 \lt 124$ | `ants_dataset[0]` |
| 123 | $123 \lt 124$ | `ants_dataset[123]`（最后一只蚂蚁） |
| 124 | $124 \ge 124$，落在第 2 段 | `bees_dataset[124 - 124] = bees_dataset[0]` |
| 130 | 落在第 2 段 | `bees_dataset[130 - 124] = bees_dataset[6]` |
| 245 | 超出总长 245 | 报 `IndexError` |

在用生成图片造的小数据集（ants 3 张、bees 2 张）上实测：`type(train_dataset).__name__` 为 `ConcatDataset`，`len` 为 `3 2 5`，`cumulative_sizes` 为 `[3, 5]`，`train_dataset[3][1]` 为 `'bees'`；`ants_dataset[10]` 报 `IndexError: list index out of range`。

> [!WARNING]+ 返回类型
>
> 此处返回的是 PIL `Image`，还不能直接进卷积。后面要用 `transforms.ToTensor()` 转成 `[C, H, W]` 的 FloatTensor。两个 Dataset 相加要求两边 `__getitem__` 返回结构一致（都是 `(img, label)` 或都是 dict）。

`ToTensor` 做了两件事：把 `(H, W, C)` 的 0–255 整数换成 `(C, H, W)` 的 0–1 浮点数，换算是逐像素除以 255：

$$
x_{\text{tensor}} = \frac{x_{\text{pixel}}}{255}, \quad 0 \to 0.0,\ 128 \to 0.502,\ 255 \to 1.0
$$

实测一张 768×512 的图：`ToTensor()(img).shape` 为 `torch.Size([3, 512, 768])`，`dtype` 为 `torch.float32`，数值都落在 `[0, 1]`。如果忘了转换，直接把这个 Dataset 交给 DataLoader，取 batch 时会报：

```text
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.JpegImagePlugin.JpegImageFile'>
```

另外，训练时损失函数要的是**整数类别编号**（如 ants→0、bees→1），而不是字符串 `"ants"`。视频版为了直观返回了字符串，正式训练前要换掉，见下面的「可运行模板」。

### 仓库版：图像与标签分目录

官方 `read_data.py` 把图像目录与标签文件目录分开，并对文件名排序以保证一一对应，再接入 `transforms`：

目录结构是这样的（每张图有一个同名 `.txt`，里面只有一行类别名）：

```text
dataset/train/
    ants_image/   0013035.jpg   1030023514_aad5c608f9.jpg ...
    ants_label/   0013035.txt   1030023514_aad5c608f9.txt ...   ← 每个 txt 内容就是 "ants"
    bees_image/   ...
    bees_label/   ...
```

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter("logs")  # 事件文件写到 ./logs，最后用 TensorBoard 查看

class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform):  # 比视频版多了 image_dir 和 transform
        self.root_dir = root_dir  # "dataset/train"
        self.image_dir = image_dir  # "ants_image"：存图片的子目录
        self.label_dir = label_dir  # "ants_label"：存标签 txt 的子目录
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)  # ['0013035.jpg', ...]
        self.label_list = os.listdir(self.label_path)  # ['0013035.txt', ...]
        self.transform = transform  # 把“怎么预处理”交给外部传入，Dataset 本身不写死
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()  # 排序后第 i 个图片名
        self.label_list.sort()  # 与第 i 个标签名同主名，下标就对齐了

    def __getitem__(self, idx):
        img_name = self.image_list[idx]  # 'xxx.jpg'
        label_name = self.label_list[idx]  # 'xxx.txt'，排过序所以和上一行是同一个样本
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)  # PIL 图，尺寸各不相同

        with open(label_item_path, 'r') as f:  # with 块结束时自动关闭文件
            label = f.readline()  # 读第一行，得到 'ants\n'（若文件末尾有换行，会带着 \n）

        # img = np.array(img)
        img = self.transform(img)  # Resize((256,256)) + ToTensor 后：Tensor [3, 256, 256]
        sample = {'img': img, 'label': label}  # 返回字典而不是元组
        return sample

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)  # 图与标签数量不等就立刻报 AssertionError
        return len(self.image_list)

if __name__ == '__main__':  # Windows 下 num_workers>0 必须有这一行
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])  # 先统一尺寸，再转张量
    root_dir = "dataset/train"
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    train_dataset = ants_dataset + bees_dataset  # ConcatDataset，下标 0..123 是蚂蚁，124.. 是蜜蜂

    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)  # 这里只是创建，脚本没有真正遍历它

    writer.add_image('error', train_dataset[119]['img'])  # 第 119 个样本（仍是蚂蚁）的 [3,256,256] 张量，默认 CHW 正好匹配
    writer.close()  # 必须关闭，事件才完整落盘
```

与视频版的差别：图像与标签分目录存放，文件名排序后按同一下标对齐；`__getitem__` 返回 `{'img': ..., 'label': ...}`，因此 TensorBoard 里写 `train_dataset[119]['img']`。标签 `'error'` 只是事件名，不代表程序出错。`num_workers>0` 时，Windows 上入口代码必须放在 `if __name__ == '__main__':` 里，否则多进程会重复 import。视频里文件夹常叫 `ants` / `bees`，仓库示例用 `ants_image` 与 `ants_label` 分开，路径需与本地目录一致。

`__len__` 里的 `assert` 保证图像数与标签文件数相等，不相等时尽早报错，避免静默错位。

用生成的图片和标签文件实测这一套（ants 3 张、bees 2 张）：`train_dataset[4]['img'].shape` 为 `torch.Size([3, 256, 256])`，`train_dataset[4]['label']` 为 `'bees\n'`；交给 `DataLoader(batch_size=2)` 取出的一个 batch 中，`batch['img'].shape` 为 `torch.Size([2, 3, 256, 256])`，`batch['label']` 为 `['ants\n', 'ants\n']`。可以看到：

- 返回 dict 时，DataLoader 会**按键分别组 batch**：图片叠成 `[B, 3, 256, 256]`，字符串标签收集成长度为 B 的列表。
- `readline()` 读出的标签带着换行符 `\n`，比较或映射前应写成 `f.readline().strip()`。

> [!EXAMPLE]+ 可运行模板：自造小数据集 + 整数标签 + ToTensor
>
> 不需要下载任何数据，先生成几张随机图片，再用一个「视频版 + 改进」的 Dataset 读出来，最后交给 DataLoader 验证能组 batch。
>
> ```python
> import os
> import numpy as np
> from PIL import Image
> from torch.utils.data import Dataset, DataLoader
> from torchvision import transforms
>
> # 第 0 步：造一个迷你数据集（真实项目里这一步换成自己的图片目录）
> for name, n in [("ants", 3), ("bees", 2)]:
>     os.makedirs(f"toy/train/{name}", exist_ok=True)              # 建 toy/train/ants、toy/train/bees
>     for i in range(n):
>         arr = np.random.randint(0, 256, (60, 80, 3), dtype=np.uint8)  # 随机像素，HWC = 60×80×3
>         Image.fromarray(arr).save(f"toy/train/{name}/{i}.jpg")    # 存成 jpg
>
>
> class FolderData(Dataset):
>     def __init__(self, root_dir, label_dir, transform=None):
>         self.root_dir = root_dir
>         self.label_dir = label_dir
>         self.path = os.path.join(root_dir, label_dir)
>         self.img_names = sorted(os.listdir(self.path))         # 排序，保证每次运行顺序一致
>         self.transform = transform                              # 外部传入的预处理
>         self.label = 0 if label_dir == "ants" else 1            # 字符串类别 → 整数编号
>
>     def __getitem__(self, idx):
>         img_path = os.path.join(self.path, self.img_names[idx])
>         img = Image.open(img_path).convert("RGB")              # 统一成 3 通道（防灰度图/RGBA 图混入）
>         if self.transform is not None:
>             img = self.transform(img)                           # PIL → Tensor [3, 60, 80]
>         return img, self.label                                  # (Tensor, int)
>
>     def __len__(self):
>         return len(self.img_names)
>
>
> tf = transforms.ToTensor()
> ants = FolderData("toy/train", "ants", tf)
> bees = FolderData("toy/train", "bees", tf)
> train_set = ants + bees                                         # ConcatDataset
> print(len(ants), len(bees), len(train_set))                     # 3 2 5
> img, label = train_set[0]
> print(type(img), img.shape, img.dtype, label)                   # <class 'torch.Tensor'> torch.Size([3, 60, 80]) torch.float32 0
> print(train_set[3][1])                                          # 1（下标 3 落在 bees 段）
> loader = DataLoader(train_set, batch_size=2, shuffle=True)      # 打乱后两张一组
> imgs, labels = next(iter(loader))                               # 取第一个 batch
> print(imgs.shape, labels)                                       # torch.Size([2, 3, 60, 80]) tensor([1, 0])（顺序随机）
> ```

**常见报错与排查：**

| 报错 | 原因 | 修法 |
| :--- | :--- | :--- |
| `FileNotFoundError: [Errno 2] No such file or directory: 'dataset/train/ants_image/0013035.jpg'` | cwd 不是项目根目录；或文件夹名与代码不一致（`ants` vs `ants_image`、`data` vs `dataset`） | `print(os.getcwd())`、`os.path.exists(path)` 逐级检查；改成绝对路径 |
| `NotADirectoryError` / 读到 `Thumbs.db`、`.DS_Store` | `os.listdir` 把系统隐藏文件也列进来了 | 过滤扩展名：`[n for n in names if n.lower().endswith((".jpg", ".png"))]` |
| `IndexError: list index out of range` | 下标 ≥ 样本数，例如数据集只有 100 张却取 `[119]` | 先 `print(len(dataset))` |
| `TypeError: default_collate: batch must contain tensors ... found <class 'PIL...'>` | `__getitem__` 返回 PIL 图，没做 `ToTensor` | 在 transform 里加 `transforms.ToTensor()` |
| `RuntimeError: stack expects each tensor to be equal size` | 图片尺寸不同，无法叠成 batch | transform 里先 `Resize((256, 256))` |
| `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | Windows + `num_workers>0`，入口代码没放进 `if __name__ == '__main__':` | 把创建 DataLoader 和遍历的代码放进 main 保护块，或先用 `num_workers=0` |
| 标签和图片对不上 | 两个 `listdir` 没排序，或图片/标签文件主名不一致 | 两边都 `sort()`；更稳的是由图片名推出标签名：`img_name.replace(".jpg", ".txt")` |

> [!TIP]+ 小结
>
> - Dataset 只负责「按下标给一条样本」；组 batch、打乱交给 DataLoader（P15）。
> - 写法 4 步：继承 `Dataset` → `__init__` 列清单 → `__getitem__` 读一条并返回 → `__len__` 报总数。
> - `__init__` 只存文件名，读图放在 `__getitem__`，否则大数据集内存放不下。
> - `ants + bees` 得到 `ConcatDataset`，下标按累计长度分段；两边返回结构必须一致。
> - 能进网络的样本应是 `(Tensor [C, H, W], 整数标签)`：PIL 图要 `ToTensor`，字符串标签要映射成 0/1/…。

<br>

## 可视化看板：TensorBoard（P8–P9）

> [!NOTE]+ 这一节要解决什么
>
> 目的：把训练过程中的数字（loss、准确率）画成曲线、把图片显示出来，方便检查「数据对不对」「训练有没有在收敛」。它不是流水线中的某一步，而是挂在旁边的仪表盘：上一节读出的图片可以先在这里看一眼；下一节 Transforms 会用它对比变换前后的图像；到完整训练（P27–P29）时用它画 loss 曲线。

对应视频 P8–P9。TensorBoard 把标量、图像、计算图写入事件文件，浏览器中查看训练曲线与样本。先安装：

```bash
pip install tensorboard
```

工作原理分两半，写和看是**两个独立的程序**：

```text
 Python 训练脚本                          命令行 / 终端
 ┌───────────────────────────┐           ┌──────────────────────────────┐
 │ writer = SummaryWriter()  │  写文件   │ tensorboard --logdir=logs    │
 │ writer.add_scalar(...)    │ ───────▶  │ 读 logs/ 下的事件文件         │
 │ writer.add_image(...)     │ logs/     │ 在 localhost:6006 开网页      │
 │ writer.close()            │ events... │                              │
 └───────────────────────────┘           └──────────────────────────────┘
```

**第一次用 TensorBoard 的 5 步：**

1. 安装：`pip install tensorboard`（装在和 torch 同一个环境里）。
2. 在脚本里创建 `SummaryWriter("logs")`，指定事件文件目录。
3. 用 `add_scalar` / `add_image` 写数据，每次带上 `tag`（图名）和 `global_step`（横轴 / 第几步）。
4. 写完 `writer.close()`，确认 `logs/` 下出现 `events.out.tfevents.*` 文件。
5. 在终端里**先 cd 到 logs 的上一级目录**，执行 `tensorboard --logdir=logs`，浏览器打开 `http://localhost:6006/`。

### SummaryWriter：创建写日志的对象

```python
from torch.utils.tensorboard import SummaryWriter
# help(SummaryWriter)
writer = SummaryWriter("logs")  # 事件文件写到 ./logs
```

主要构造参数：

| 参数 | 含义 |
| :--- | :--- |
| `log_dir` | 事件文件目录；省略时默认 `runs/当前时间_主机名` |
| `flush_secs` | 多久把缓冲刷到磁盘 |

补充实测细节（PyTorch 2.6）：完整签名是 `SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')`。

- `flush_secs` 默认 **120 秒**：程序还在跑时，TensorBoard 网页里的数据可能滞后约 2 分钟，这是正常的。
- 不传 `log_dir` 时实测目录名形如 `runs\Oct15_19-48-16_主机名`，每次运行一个新目录，天然不会叠图。
- `comment="_lr0.01"` 会追加到默认目录名后面，方便区分实验（只在不传 `log_dir` 时生效）。
- 每次 `SummaryWriter(...)` 都会在目录里新建一个 `events.out.tfevents.<时间戳>.<主机名>.<进程号>.<序号>` 文件；序号是同一进程内的计数，第一个 writer 为 `0`，同一进程再建第二个就是 `1`。

`writer` 用完必须 `close()`，否则最后几条可能没落盘。

### 标量：画 loss 曲线

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")  # 打开（或新建）logs 目录
for i in range(100):  # i = 0..99，充当训练步数
    writer.add_scalar("y=2x", 2 * i, i)  # tag, Y, X
writer.close()  # 刷盘并关闭
```

`add_scalar(tag, scalar_value, global_step)`：

- `tag`：图标题。相同 tag 画在同一张图
- `scalar_value`：纵轴，如 loss、准确率
- `global_step`：横轴，通常用训练步数

到真实训练时，这三个参数一般这样填（P27–P29 会用到）：

```python
# 伪代码：训练循环里记录 loss
total_train_step = 0                                   # 全局步数计数器，跨 epoch 累加
for epoch in range(10):
    for imgs, targets in train_loader:
        ...                                            # 前向、算 loss、反传、更新
        total_train_step += 1
        if total_train_step % 100 == 0:                # 每 100 步记一次，避免事件文件太大
            writer.add_scalar("train_loss", loss.item(), total_train_step)  # .item() 把单元素张量变成 Python 数
```

`tag` 里可以用 `/` 分组，例如 `"loss/train"` 和 `"loss/test"` 会在网页里归到同一个 `loss` 折叠面板下。

在事件文件所在目录启动：

```bash
tensorboard --logdir=logs
```

默认端口 `6006`，浏览器打开终端打印的 URL（多为 `http://localhost:6006/`）。冲突时：

```bash
tensorboard --logdir=logs --port=6007
```

在 Jupyter 里也可以不开终端，直接在单元格中内嵌显示：

```text
%load_ext tensorboard
%tensorboard --logdir logs
```

同一 `tag` 反复跑实验、又不删旧事件文件时，曲线会叠在一起（例如先写 `y=x` 再写 `y=2x` 仍用标题 `"y=x"`）。处理：删掉 `logs` 里的 `events.out.tfevents.*`，Ctrl+C 停掉旧 tensorboard，再启动。

更省事的办法是**每个实验用一个子目录**：`SummaryWriter("logs/exp1")`、`SummaryWriter("logs/exp2")`，然后 `tensorboard --logdir=logs`。TensorBoard 会把每个子目录当作一次 run，用不同颜色画在同一张图里，左侧可以勾选显示哪几条，不会再混成一条乱线。

### 图像：add_image 与 HWC / CHW

`add_image` 要求 `img_tensor` 为 `torch.Tensor` 或 `numpy.ndarray`。PIL 的 `Image.open` 得到的不是这两种，必须先转换。

> [!WARNING]+ 易错：文档里的 string 不是图片路径
>
> `help(add_image)` 里写的是 `torch.Tensor, numpy.ndarray, or string/blobname`，这里的 string 指旧版 Caffe2 的 **blob 名称**，**不是图片文件路径**，很容易看错。实测把一个 png 路径直接传给 `add_image`，会报 `AssertionError: size of input tensor and input format are different. tensor shape: (1,), input_format: CHW`。实际可用的只有 Tensor 和 ndarray 两种：PIL 图要么 `np.array(img)`（配合 `dataformats='HWC'`），要么 `transforms.ToTensor()(img)`（默认 CHW）。

| 来源 | `type` | 典型 `shape` |
| :--- | :--- | :--- |
| `PIL.Image.open` | `PIL.Image.Image` | 无 `.shape`，有 `.size = (W, H)` |
| `np.array(PIL图)` | `numpy.ndarray` | `(H, W, C)` 即 HWC |
| `cv2.imread` | `numpy.ndarray` | `(H, W, C)`，通道为 BGR |
| `transforms.ToTensor()` | `torch.Tensor` | `(C, H, W)` 即 CHW |

用一张宽 768、高 512 的 RGB 图把三种表示对照一遍（实测）：

| 表示 | 打印 | 尺寸怎么读 |
| :--- | :--- | :--- |
| PIL | `img.size == (768, 512)`，`img.mode == 'RGB'` | 先宽后高 |
| ndarray | `arr.shape == (512, 768, 3)`，`dtype uint8`，值 0–255 | 高、宽、通道 |
| Tensor | `t.shape == torch.Size([3, 512, 768])`，`float32`，值 0–1 | 通道、高、宽 |

记忆方法：PIL 是「看图软件视角」（宽 × 高），numpy 是「矩阵视角」（行 = 高、列 = 宽、最后放通道），PyTorch 把通道提到最前面，因为卷积是按通道操作的。

`add_image` **默认按 CHW**。HWC 必须声明 `dataformats='HWC'`，否则会按 `(3, H, W)` 去解释 `(H, W, 3)`，直接报错或花屏。

实测：把 `(512, 768, 3)` 的 ndarray 不加 `dataformats` 传进去，报的是

```text
TypeError: Cannot handle this data type: (1, 1, 512), |u1
```

报错信息里的 `512` 其实是图片的高度，被误当成了「通道数」，这就是 HWC / CHW 弄反的典型症状。

```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants_image/0013035.jpg")  # 相对路径，cwd 必须是项目根目录
print(type(img))            # <class 'PIL.JpegImagePlugin.JpegImageFile'>，是 PIL.Image.Image 的子类
img_array = np.array(img)  # PIL → ndarray，HWC，uint8
print(type(img_array), img_array.shape)  # ndarray, 例如 (512, 768, 3)
writer.add_image("Img Test", img_array, 1, dataformats="HWC")  # 第 1 步；声明 HWC
writer.close()
```

> [!WARNING]+ 易错：判断是不是 PIL 图
>
> `Image.open` 返回的具体类随文件格式变化：JPEG 是 `JpegImageFile`，PNG 是 `PngImageFile`，它们都是 `PIL.Image.Image` 的**子类**。所以 `type(img) == Image.Image` 的结果是 `False`；要判断是不是 PIL 图，用 `isinstance(img, Image.Image)`。

`global_step` 可用来在同一 tag 下切换多张图（拖动滑条）。批量图像用 `writer.add_images`（复数，要求第一维是 batch）；计算图用 `writer.add_graph(model, input)`。

几个图像相关的形状要求（实测）：

| 调用 | 期望输入 | 例子 |
| :--- | :--- | :--- |
| `add_image(tag, t)` | 3 维 `[C, H, W]` | `[3, 256, 256]` |
| `add_image(tag, arr, dataformats="HWC")` | 3 维 `(H, W, C)` | `(512, 768, 3)` |
| `add_images(tag, batch)` | 4 维 `[N, C, H, W]` | `[2, 3, 256, 256]` |
| 把 4 维 batch 误传给 `add_image` | — | 报 `AssertionError: size of input tensor and input format are different. tensor shape: (2, 3, 256, 256), input_format: CHW` |

用 `global_step` 在同一个 tag 下放多张图，网页里图片上方会出现一个滑条：

```python
from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter("logs")
for step in range(3):
    fake = torch.rand(3, 64, 64)                 # 随机生成一张 [3, 64, 64] 的“图”，值在 0–1
    writer.add_image("random", fake, step)       # 同一个 tag，不同 step → 滑条切换
writer.add_images("batch", torch.rand(8, 3, 32, 32), 0)  # 一次写 8 张，网页里拼成网格显示
writer.close()
```

官方 `P8_Tensorboard.py`：

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/6240329_72c01e663e.jpg"  # 注意目录叫 data，前文是 dataset，按本地实际改
img_PIL = Image.open(image_path)  # PIL 图
img_array = np.array(img_PIL)  # 转 ndarray，HWC
print(type(img_array))  # <class 'numpy.ndarray'>
print(img_array.shape)  # (H, W, 3)

writer.add_image("train", img_array, 1, dataformats='HWC')  # tag="train"，第 1 步
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)  # 纵轴实际是 3i，tag 只是名字

writer.close()
```

批量图像用 `writer.add_images`（复数），计算图用 `writer.add_graph(model, input)`。仓库这份脚本把 `y=2x` 写成了 `3*i`，只是演示横纵轴，标题与公式不必一致；正式记曲线时应让 tag 与含义对应，避免旧事件叠图。

**常见报错与排查：**

| 现象 / 报错 | 原因 | 修法 |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'tensorboard'` | 当前环境没装 tensorboard | 在同一环境 `pip install tensorboard` |
| 调 `add_image` 时报 `ModuleNotFoundError: No module named 'PIL'` | `add_image` 内部用 Pillow 编码图片（文档注明 requires the pillow package） | `pip install pillow` |
| 网页显示 `No dashboards are active for the current data set` | `--logdir` 指错目录（终端 cwd 不对），或事件文件还没刷盘 | 终端先 cd 到 logs 的上一级，或写绝对路径；确认已 `close()` |
| 刚写的数据网页里看不到 | `flush_secs` 默认 120 秒才刷盘 | 等一会儿、`writer.flush()`，或 `close()` 后刷新网页 |
| 曲线来回折返、像乱麻 | 同一 tag、同一目录里有多次运行的事件文件 | 删旧事件文件，或每次实验用新子目录 |
| `TypeError: Cannot handle this data type: (1, 1, 512), \|u1` | HWC 的 ndarray 没写 `dataformats="HWC"` | 加上 `dataformats="HWC"`，或改用 `ToTensor()` 得到 CHW |
| `AssertionError: size of input tensor and input format are different` | 传入维度和 `dataformats` 不符：4 维 batch 用了 `add_image`，或传了路径字符串 | batch 用 `add_images`；路径先 `Image.open` 再转换 |
| `TensorBoard could not bind to port 6006` | 旧的 tensorboard 进程还在占用端口 | Ctrl+C 关掉旧进程，或 `--port=6007` |

> [!TIP]+ 小结
>
> - 写和看分开：脚本里 `SummaryWriter` 写事件文件，终端里 `tensorboard --logdir=logs` 读并开网页。
> - `add_scalar(tag, y, step)` 画曲线；`add_image(tag, img, step)` 放图，默认 CHW，ndarray 要加 `dataformats="HWC"`；batch 用 `add_images`。
> - PIL `.size` 是 `(W, H)`，numpy 是 `(H, W, C)`，PyTorch 是 `(C, H, W)`。
> - 用完 `close()`；不想叠图就每个实验一个子目录，或删旧 `events.out.tfevents.*`。
> - 路径字符串不能直接传给 `add_image`，先读成图再转成 ndarray / Tensor。

<br>

## 图像预处理：transforms（P10–P13）

> [!NOTE]+ 这一节要解决什么
>
> 目的：把 Dataset 读出来的 PIL 图片变成网络能吃的 Tensor，并顺手做缩放、裁剪、归一化。
>
> 位置：数据 → **变换** → 打包 → 模型 → 损失 → 优化 → 训练循环 → 保存/验证。
>
> 衔接：上一节的 `MyData.__getitem__` 返回的还是 PIL `Image`；这一节学会 transforms 之后，下一节的内置数据集只需传一个 `transform=` 参数，就能直接吐出 Tensor。

对应视频 P10–P13。`torchvision.transforms` 是视觉预处理工具箱（不用于文本）：缩放、裁剪、翻转、归一化、PIL → Tensor 等。用法模式都一样：先用类**构造模具**得到实例，再把图像**当参数调用**该实例。

用任何一个 transform 都是固定的 3 步：

1. **看文档/源码**：确认它要求的输入类型（PIL？Tensor？）、输出类型，以及 `__init__` 需要哪些参数（如 `Resize` 要 `size`，`Normalize` 要 `mean`、`std`）。
2. **构造实例**：`tool = transforms.Xxx(参数)`。这一步只是“设定好工具”，还没碰图片。
3. **调用实例**：`out = tool(img)`。真正的计算发生在这一步，然后 `print(type(out))` 确认类型对不对。

### 为什么实例能像函数一样调用：`__call__`

`transforms.ToTensor()` 返回的是**实例**。对该实例写 `tensor_trans(img)` 能跑，是因为类实现了 `__call__`：`obj(x)` 等价于 `obj.__call__(x)`。这与 `nn.Module`、以及后面几乎所有 transform 相同。

```python
class Person:
    def __call__(self, name):  # 定义了 __call__，实例就可以写成 person(...) 的形式调用
        print("__call__ Hello " + name)

    def hello(self, name):  # 普通方法，必须用 实例.hello(...) 调用
        print("Hello " + name)

person = Person()  # 构造实例：此时 __call__ 还没被执行
person.hello("li4")   # 必须写方法名  -> 输出 Hello li4
person("zhang3")      # 走 __call__，不需要点号  -> 输出 __call__ Hello zhang3
```

实际运行还可以验证两点：

```python
person.__call__("wang5")              # 与 person("wang5") 完全等价 -> __call__ Hello wang5
print(callable(person))               # True：实现了 __call__ 的对象是“可调用的”
print(callable(transforms.ToTensor()))  # True：transform 实例同理
```

> [!EXAMPLE]+ 容易混的两个括号
>
> `transforms.ToTensor()` 里的 `()` 是**构造**（调用类，走 `__init__`），得到工具；`tensor_trans(img)` 里的 `()` 是**使用**（调用实例，走 `__call__`），得到结果。所以也可以写成一行 `transforms.ToTensor()(img)`：第一对括号造工具，第二对括号用工具。

```python
from PIL import Image
from torchvision import transforms

img = Image.open("dataset/train/bees/16838648_415acd9e3f.jpg")  # PIL.Image，路径相对于“运行目录”
tensor_trans = transforms.ToTensor()  # 第 2 步：构造实例（工具）
tensor_img = tensor_trans(img)  # 第 3 步：调用实例，PIL -> Tensor
print(type(tensor_img), tensor_img.shape)  # torch.Tensor, [C, H, W]
```

没有蜜蜂图片时，可以用 numpy 自己造一张图跑通同样的类型链（下面是实际运行结果）：

```python
import numpy as np
from PIL import Image
from torchvision import transforms

arr = np.zeros((300, 400, 3), dtype=np.uint8)  # 高 300、宽 400、3 通道，HWC 排列，全黑
arr[0, 0] = [204, 102, 0]                      # 把左上角像素设成 R=204, G=102, B=0
img = Image.fromarray(arr)                     # ndarray -> PIL
print(type(img), img.size, img.mode)           # <class 'PIL.Image.Image'> (400, 300) RGB  注意 PIL 的 size 是 (宽, 高)

t = transforms.ToTensor()(img)                 # PIL -> Tensor
print(type(t), t.dtype, t.shape)               # <class 'torch.Tensor'> torch.float32 torch.Size([3, 300, 400])  变成 [C, H, W]
print(t[:, 0, 0])                              # tensor([0.8000, 0.4000, 0.0000])  左上角像素三个通道
print(t.min().item(), t.max().item())          # 0.0 0.800000011920929  全部落在 [0, 1]
```

`ToTensor` 同时解决两件事：

1. 类型：PIL / `uint8` ndarray → `torch.FloatTensor`
2. 数值：对 **uint8** 的 `[0, 255]` 除以 255，落到 `[0.0, 1.0]`；通道从 HWC 改为 CHW。若 ndarray 已经是 float，则**不再**除 255

手算上面左上角像素：$204/255=0.8$，$102/255=0.4$，$0/255=0$，与打印的 `tensor([0.8000, 0.4000, 0.0000])` 一致。形状变化：PIL `size=(400, 300)`（宽, 高）→ ndarray `(300, 400, 3)`（H, W, C）→ Tensor `[3, 300, 400]`（C, H, W）。

“float 不再除 255”的实测：

```python
a = np.array(img)                              # PIL -> ndarray
print(a.dtype, a.shape)                        # uint8 (300, 400, 3)
print(transforms.ToTensor()(a)[:, 0, 0])       # tensor([0.8000, 0.4000, 0.0000])  uint8 会除以 255
af = a.astype(np.float32)                      # 先转成 float32
print(transforms.ToTensor()(af)[:, 0, 0])      # tensor([204., 102.,   0.])  float 不再除 255，数值还是 0~255
print(transforms.ToTensor()(a.astype(np.float64)).dtype)  # torch.float64  float64 也原样保留，不会变成 float32
```

所以若自己把图片读成 float 数组再交给 `ToTensor`，需要先手动 `/ 255.0`，否则数值范围错了，网络照样能跑但效果会很差（这种错误不会报错，最难查）。

OpenCV 的 `cv2.imread` 得到 `numpy.ndarray`（BGR），也可直接交给 `ToTensor`，但颜色通道顺序与 PIL 的 RGB 不同。单通道 PIL（`mode='L'`）得到 `[1, H, W]`，不是 3 通道。

实测其它模式：`img.convert("L")` 后 `ToTensor` 得到 `torch.Size([1, 300, 400])`，左上角灰度 $\approx (0.299\times204+0.587\times102+0.114\times0)/255 \approx 121/255 \approx 0.4745$；`img.convert("RGBA")`（png 常见）得到 `[4, 300, 400]`，多出一个透明度通道。后面的 `Conv2d(3, ...)` 只接受 3 通道，所以读 png 时常写 `Image.open(path).convert("RGB")`。OpenCV 的 BGR 可以用 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 转回 RGB 再进 `ToTensor`。

> [!NOTE]+ Tensor
>
> Tensor 是神经网络专用的多维数组，除了存数值，还可携带梯度、设备（CPU/GPU）等信息。`nn.Conv2d`、损失函数和优化器都要求 Tensor；PIL 需先经 `ToTensor` 转换。

> [!WARNING]+ 易错：`np.array` 不等于 `ToTensor`
>
> `np.array(img)` 得到的是 numpy 数组，不是 Tensor，不能直接喂给 `nn.Conv2d`。若走 numpy 路线，还需要 `torch.from_numpy(a).permute(2, 0, 1).float() / 255`（转 Tensor、HWC→CHW、转 float、缩放）才等价于 `ToTensor`。`np.array` 版本只适合 `writer.add_image(..., dataformats='HWC')` 这类接受 ndarray 的场景。

### 常用 transform：Normalize / Resize / Compose / RandomCrop

课程强调每用一个 transform 都先确认：**输入类型 → 输出类型**、`__init__` 要哪些参数。

| transform | `__init__` 参数 | 输入类型 | 输出类型 |
| :--- | :--- | :--- | :--- |
| `ToTensor()` | 无 | PIL / ndarray | Tensor（float） |
| `Normalize(mean, std)` | 每通道均值、标准差列表 | float Tensor `[C, H, W]` | Tensor |
| `Resize(size)` | `(H, W)` 或一个 int | PIL 或 Tensor | 与输入同类型 |
| `RandomCrop(size)` | int 或 `(H, W)` | PIL 或 Tensor | 与输入同类型 |
| `Compose([...])` | transform 实例的列表 | 第一个 transform 的输入 | 最后一个 transform 的输出 |

**Normalize** `[mean, …], [std, …]`：按通道

$$
\text{out}[c] = \frac{\text{in}[c] - \mathrm{mean}[c]}{\mathrm{std}[c]}
$$

三通道都用 `0.5` 时，原 `[0, 1]` 被拉到 `[-1, 1]`。例如某像素 `0.8`：$(0.8-0.5)/0.5=0.6$。可用 `print(img_tensor[0][0][0])` 与归一化后对比。

接着上面造的那张图手算一遍（每个通道各算各的）：

| 通道 | ToTensor 后 | mean=0.5, std=0.5 | ImageNet 均值方差 |
| :--- | :--- | :--- | :--- |
| R | 0.8 | $(0.8-0.5)/0.5=0.6$ | $(0.8-0.485)/0.229\approx1.3755$ |
| G | 0.4 | $(0.4-0.5)/0.5=-0.2$ | $(0.4-0.456)/0.224=-0.25$ |
| B | 0.0 | $(0-0.5)/0.5=-1.0$ | $(0-0.406)/0.225\approx-1.8044$ |

```python
n = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(t)  # 输入必须是 float Tensor
print(n[:, 0, 0])                    # tensor([ 0.6000, -0.2000, -1.0000])  与手算一致
print(n.min().item(), n.max().item())  # -1.0 0.6000000238418579  0 被映射到 -1，最大的 0.8 映射到 0.6
n2 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t)  # 用预训练模型时常见的 ImageNet 统计量
print(n2[:, 0, 0])                   # tensor([ 1.3755, -0.2500, -1.8044])
```

为什么要归一化：让各通道数值以 0 为中心、尺度相近，梯度更新更稳定。用 `0.5/0.5` 只是课程演示；用 torchvision 预训练模型（后面“预训练”一节）时应换成 ImageNet 的那组数。注意 `Normalize` 不改形状，也不保证结果一定落在 `[-1, 1]`（ImageNet 那组就超出了）。

**Resize**：

- `Resize((H, W))`：强制拉到该高宽，可能变形
- `Resize(512)`：短边缩到 512，长边按比例，图像仍是 PIL（若输入是 PIL）

实测（原图宽 400、高 300）：

```python
r1 = transforms.Resize((512, 512))(img)  # 元组 = (H, W)，强制拉伸
print(type(r1).__name__, r1.size)        # Image (512, 512)
r2 = transforms.Resize(512)(img)         # 单个 int = 短边缩放到 512
print(type(r2).__name__, r2.size)        # Image (682, 512)  宽 400*512/300≈682.7 取整为 682，高 512
r3 = transforms.Resize(512)(t)           # 新版本 Resize 也能吃 Tensor
print(type(r3).__name__, r3.shape)       # Tensor torch.Size([3, 512, 682])  Tensor 的 shape 是 [C, H, W]
```

易错点：`Resize((H, W))` 里是“先高后宽”，而 PIL 打印 `img.size` 是“先宽后高”，两者顺序相反。

类型链常见为：PIL → Resize → PIL → ToTensor → Tensor。

**Compose**：参数是 **transform 实例的列表**，按顺序执行。`Compose([Resize(512), ToTensor()])` 等价于先 Resize 再 ToTensor。

`Compose` 的执行过程拆开就是一个 for 循环：

1. 拿到输入 `img`（PIL）；
2. 交给列表第 1 个 transform：`img = Resize(512)(img)`，仍是 PIL，尺寸变成 `(682, 512)`；
3. 把上一步结果交给第 2 个：`img = ToTensor()(img)`，变成 Tensor `[3, 512, 682]`；
4. 列表走完，返回最后的结果。

因此前一个的**输出类型**必须是后一个的**输入类型**。顺序写反（如 `[ToTensor(), ToTensor()]`，或 `Normalize` 放在 `ToTensor` 前面）就会报下文的类型错误。

**RandomCrop**：在画面中随机位置切出 `size`。`RandomCrop(512)` 切正方形；`RandomCrop((50, 50))` 切指定高宽。裁剪边长大于原图会报错。循环多次并 `add_image(..., global_step=i)` 可在 TensorBoard 里来回看不同裁剪。

实测：对 400×300 的图，`RandomCrop(256)` 得到 PIL `size=(256, 256)`；`RandomCrop((50, 80))` 得到 `size=(80, 50)`（同样是传入 (H, W)、打印出 (W, H)）；`RandomCrop(512)` 报 `ValueError: Required crop size (512, 512) is larger than input image size (300, 400)`。蜜蜂原图比 512 大所以视频能跑；换成 CIFAR 的 32×32 小图就必须把 size 调小，或加参数 `pad_if_needed=True` 先补边。

关注输入输出类型：未知就 `print(type(x))` / `print(x.size)`（PIL）/ `print(x.shape)`（Tensor）。参数不确定时 Ctrl+P。

下面的官方脚本按 6 步走：建 writer → 读图 → ToTensor → Normalize → Resize/Compose → RandomCrop 循环。行尾注释标出了每一步的类型变化：

```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")  # 事件文件写到 ./logs，之后 tensorboard --logdir=logs 查看
img = Image.open("dataset/train/bees/16838648_415acd9e3f.jpg")  # PIL.Image

trans_totensor = transforms.ToTensor()  # 工具：PIL -> Tensor
img_tensor = trans_totensor(img)  # Tensor [3, H, W]，值域 [0, 1]
writer.add_image("ToTensor", img_tensor)  # CHW Tensor，不需要 dataformats

trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 三个通道的 mean、std
img_norm = trans_norm(img_tensor)  # Tensor -> Tensor，值域 [0, 1] -> [-1, 1]
writer.add_image("Normalize", img_norm)  # 负数会被截断显示，所以图看起来颜色变了

trans_resize = transforms.Resize((512, 512))  # 强制拉到 512x512
img_resize = trans_totensor(trans_resize(img))  # PIL -> Resize -> PIL -> ToTensor -> Tensor [3, 512, 512]
writer.add_image("Resize", img_resize, 0)  # global_step=0

trans_compose = transforms.Compose([  # 把两步串成一个工具
    transforms.Resize(512),  # 第 1 步：短边缩到 512，输出 PIL
    transforms.ToTensor()  # 第 2 步：PIL -> Tensor
])
img_resize_2 = trans_compose(img)  # 一次调用走完两步，Tensor [3, 512, 长边按比例]
writer.add_image("Resize", img_resize_2, 1)  # 同一 tag、step=1，在滑条上与上一张对比

trans_random = transforms.RandomCrop(512)  # 随机切 512x512
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])  # PIL -> 裁剪 -> PIL -> Tensor
for i in range(10):  # 每次调用随机位置都不同
    writer.add_image("RandomCrop", trans_compose_2(img), i)  # step=i，共 10 张

writer.close()  # 关闭才会把缓冲写完
```

> [!WARNING]+ 仓库 `P9_transforms.py`
>
> 该类 `__getitem__` 里写的是 `img = transform(img)`，没有用 `self.transform`。能跑是因为文件后部定义了同名全局变量 `transform`，属于笔误。自定义 Dataset 应写 `self.transform(img)`，并先判断 `if self.transform is not None`。

### 常见报错与排查

以下报错信息均为实际运行得到：

| 报错 | 原因 | 修法 |
| :--- | :--- | :--- |
| `TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>` | 对已经是 Tensor 的东西又做了一次 `ToTensor`（常见于 Compose 里写了两次，或 Dataset 里转了一次、外面又转一次） | 只保留一次 `ToTensor` |
| `TypeError: img should be Tensor Image. Got <class 'PIL.Image.Image'>` | 把 PIL 直接交给 `Normalize` | 先 `ToTensor` 再 `Normalize` |
| `TypeError: Input tensor should be a float tensor. Got torch.uint8.` | 交给 `Normalize` 的是 uint8 Tensor（如 `torchvision.io.read_image` 读出来的） | 先 `.float() / 255` 或用 `ToTensor` |
| `RuntimeError: output with shape [1, 300, 400] doesn't match the broadcast shape [3, 300, 400]` | 灰度图只有 1 个通道，却给了 3 个通道的 mean/std | `img.convert("RGB")`，或改成 `Normalize([0.5], [0.5])` |
| `ValueError: Required crop size (512, 512) is larger than input image size (300, 400)` | 裁剪尺寸大于原图 | 调小 size 或 `pad_if_needed=True` |
| `TypeError: 'ToTensor' object is not iterable` | `Compose(ToTensor())` 没写成列表 | 写成 `Compose([ToTensor()])` |

> [!TIP]+ 小结
>
> - transform 都是“先构造实例、再调用实例”，能调用是因为类实现了 `__call__`。
> - `ToTensor`：PIL/uint8 → float Tensor，HWC → CHW，除以 255；float 数组不会再除。
> - `Normalize` 按通道算 $(x-\text{mean})/\text{std}$，输入必须是 float Tensor，通道数要和 mean/std 个数对上。
> - `Resize((H, W))` 强拉，`Resize(int)` 按短边等比缩放；PIL 的 `size` 是 (W, H)，Tensor 的 `shape` 是 (C, H, W)。
> - `Compose` 按列表顺序执行，前一个的输出类型必须是后一个的输入类型；不确定就 `print(type(x))`。

<br>

## 内置数据集：torchvision.datasets（P14）

> [!NOTE]+ 这一节要解决什么
>
> 目的：不手写 Dataset，直接用 torchvision 自带的标准数据集（CIFAR-10 等），并通过 `transform=` 参数把上一节的变换挂进去。
>
> 位置：**数据** + 变换 → 打包 → 模型 → ……
>
> 衔接：它和前面自己写的 `MyData` 遵守同一个约定（`dataset[i]` 返回 `(img, target)`）；下一节的 DataLoader 正是靠这个约定把样本一批批打包。

对应视频 P14。`torchvision.datasets` 提供 CIFAR、MNIST 等。不必手写 `MyData`，但返回约定仍是「下标 → (图像, 标签)」。

CIFAR-10：60000 张 `32×32` RGB，10 类每类 6000 张；训练 50000、测试 10000。类别名为

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

对应编号 `0 … 9`。

即 airplane=0、automobile=1、bird=2、cat=3、deer=4、dog=5、frog=6、horse=7、ship=8、truck=9。标签存的是整数编号，`dataset.classes[编号]` 才得到名字。

### 下载与参数

```python
import torchvision

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)  # 训练集 50000 张，约 170MB 压缩包
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)  # 测试集 10000 张，与训练集共用同一个压缩包
```

| 参数 | 含义 |
| :--- | :--- |
| `root` | 存放目录 |
| `train` | `True` 训练集，`False` 测试集（必须大写） |
| `download` | `True` 时若本地没有就下载；已存在则校验，不重复下 |
| `transform` | 对**图像**做的变换；`None` 时 `__getitem__` 返回 PIL |
| `target_transform` | 对**标签**做的变换（如把 int 转成 one-hot），默认 `None`，课程里用不到 |

第一次用内置数据集的 4 步：

1. **选数据集类**：如 `torchvision.datasets.CIFAR10`，看文档确认它的参数和返回值。
2. **确定 root 与 train**：`root` 是相对于运行目录的路径；训练集、测试集各建一个实例。
3. **先不加 transform 看一眼**：`print(test_set[0])`，确认是 `(PIL, int)`。
4. **再加 transform**：传 `transform=ToTensor()`（或 Compose），之后 `test_set[i]` 拿到的就是 Tensor。

> [!EXAMPLE]+ 下载太慢 / 不想下载时
>
> 下载慢时，可以把控制台打印的下载链接（`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`）复制到浏览器或下载工具里下好，把压缩包放进 `root` 目录，再运行代码（`download=True` 会先校验本地文件，存在就直接解压）。只想把代码跑通、不在乎图片内容时，可以用 `torchvision.datasets.FakeData` 生成假数据，它的返回约定和 CIFAR10 一样，见下文。

不设 `transform` 时：

```python
print(test_set[0])  # 下标访问 -> 调用 test_set.__getitem__(0)
# (<PIL.Image.Image image mode=RGB size=32x32 ...>, 3)  测试集第 0 张：target=3，对应 cat
print(test_set.classes)  # ['airplane', 'automobile', ..., 'truck']，长度 10

img, target = test_set[0]  # 元组解包：第 0 个是图，第 1 个是标签
print(type(img), img.size)          # PIL, (32, 32)
print(target)                       # int
print(test_set.classes[target])     # 类别字符串
# img.show()
```

> [!WARNING]+ 易错：`test_set[0]` 与 `train_set[0]` 别记混
>
> 按官方 CIFAR-10 数据文件，**测试集**第 0 张（`test_set[0]`）的标签是 `3`（cat）；**训练集**第 0 张（`train_set[0]`）的标签是 `6`（frog）。两者常被记混；类别编号与名字的对应见上面的列表（3 是 cat，6 是 frog）。

用 `FakeData` 不下载也能实际跑通同样的代码（假图是随机噪声，标签是随机整数，下面是实测输出）：

```python
import torchvision
from torchvision import transforms

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
fake = torchvision.datasets.FakeData(size=100, image_size=(3, 32, 32), num_classes=10)  # 100 张 3x32x32 假图，10 类
print(len(fake))                     # 100  -> 调用 __len__
print(fake[0])                       # (<PIL.Image.Image image mode=RGB size=32x32 at 0x...>, 7)  与 CIFAR10 同结构
img, target = fake[0]
print(type(img), img.size, img.mode) # <class 'PIL.Image.Image'> (32, 32) RGB
print(type(target), target, classes[target])  # <class 'int'> 7 horse

fake_t = torchvision.datasets.FakeData(size=10000, image_size=(3, 32, 32), num_classes=10,
                                       transform=transforms.ToTensor())  # 模拟 CIFAR 测试集：10000 张 + ToTensor
img, target = fake_t[0]
print(type(img), img.shape, img.dtype)  # <class 'torch.Tensor'> torch.Size([3, 32, 32]) torch.float32
```

`FakeData` 每个下标用固定随机种子生成，同一个 `fake_t[0]` 取两次内容相同（实测 `torch.equal` 为 `True`），行为像真实数据集。

PIL 不能直接进卷积，也不能直接 `add_image`（除非先 ToTensor / ndarray）。因此训练前几乎总会：

```python
dataset_transform = torchvision.transforms.Compose([  # 即使只有一步也习惯写成 Compose，方便以后往里加 Normalize 等
    torchvision.transforms.ToTensor()  # PIL [32x32] -> Tensor [3, 32, 32]
])
```

`transform=` 传入后，变换发生在该数据集的 `__getitem__` 内部，外部仍写 `img, target = test_set[i]`，只是 `img` 已是 Tensor。

拆开 `test_set[i]` 内部发生的事（与 CIFAR10 源码一致）：

1. 从内存数组 `self.data[i]`（形状 `(32, 32, 3)` 的 uint8）和 `self.targets[i]`（int）取出原始数据；
2. `Image.fromarray(...)` 转成 PIL，保证和其他数据集行为一致；
3. 若 `self.transform is not None`，执行 `img = self.transform(img)`；
4. 若 `self.target_transform is not None`，对标签做变换；
5. `return img, target`。

官方 `P10_dataset_transform.py`：

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)  # 取出来的 img 是 Tensor
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)  # 同上

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
#
# print(test_set[0])

writer = SummaryWriter("p10")  # 事件写到 ./p10
for i in range(10):  # 只看测试集前 10 张
    img, target = test_set[i]  # img: Tensor [3, 32, 32]，target: int
    writer.add_image("test_set", img, i)  # 同一 tag，step 0~9

writer.close()
```

循环里 `writer.add_image("test_set", img, i)` 用 `global_step=i` 切换前 10 张。此时 `img` 已是 CHW Tensor，不必写 `dataformats`。

打开 `CIFAR10` 源码中的 `__getitem__`，能确认返回的是 `(img, target)`，后面 DataLoader 才会解包成 `imgs, targets = data`。

### 常见报错与排查

- `RuntimeError: Dataset not found or corrupted. You can use download=True to download it`：`root` 下没有数据且 `download=False`，或路径写错（相对路径是相对于**运行目录**，PyCharm 里看 Run Configuration 的 Working directory）。
- 下载中途断掉后反复报校验失败：删掉 `root` 下不完整的 `cifar-10-python.tar.gz` 再重下。
- 交给 DataLoader 时报 `TypeError: default_collate: batch must contain tensors, ... found <class 'PIL.Image.Image'>`（见下一节）：忘了传 `transform=ToTensor()`，拿到的还是 PIL。
- `train=true`：`NameError: name 'true' is not defined`，Python 的布尔值必须大写 `True`。

> [!TIP]+ 小结
>
> - 内置数据集和自定义 Dataset 约定相同：`dataset[i]` → `(img, target)`，`len(dataset)` → 样本数。
> - 不传 `transform` 得到 PIL；传 `transform=ToTensor()` 得到 `[3, 32, 32]` 的 float Tensor，变换发生在 `__getitem__` 内部。
> - 标签是 0–9 的整数，`dataset.classes[target]` 查名字。
> - 不想下载时用 `FakeData(size, image_size, num_classes, transform)` 代替，接口一致。

<br>

## 打包成批：DataLoader（P15）

> [!NOTE]+ 这一节要解决什么
>
> 目的：把 Dataset “一次一张”变成“一次一批（batch）”，并负责洗牌、多进程读取。
>
> 位置：数据 → 变换 → **打包** → 模型 → 损失 → 优化 → 训练循环 → ……
>
> 衔接：上一节得到的是单张 `[3, 32, 32]`；这一节输出 `[64, 3, 32, 32]` 的四维张量，正好是下一节起 `nn.Module` / `Conv2d` 要求的输入形状 `[N, C, H, W]`。

对应视频 P15。Dataset 一次给一张；网络训练按 batch 算梯度。`DataLoader` 负责堆叠、打乱、多进程预取。

打个比方：Dataset 是一副扑克牌（能按编号抽一张），DataLoader 是发牌的人：每轮（epoch）是否先洗牌（`shuffle`）、每次发几张（`batch_size`）、最后剩下不够一手的牌要不要（`drop_last`）。

```python
torch.utils.data.DataLoader(
    dataset,  # 必填：要打包的数据集
    batch_size=1,  # 默认一次只取 1 张，训练时一般改成 64 等
    shuffle=False,  # 不打乱；签名里的默认值是 None，效果与 False 相同
    num_workers=0,  # 默认主进程读取
    drop_last=False  # 默认保留最后不满的一批
)
```

| 参数 | 含义 |
| :--- | :--- |
| `dataset` | 实现了 `__getitem__` / `__len__` 的数据集 |
| `batch_size` | 每批样本数 |
| `shuffle` | 每个 epoch 是否重新洗牌；训练常 `True`，对照实验可 `False` |
| `num_workers` | 读数据的子进程数；`0` 表示在主进程读。Windows 调试先用 `0` |
| `drop_last` | 最后一批不足 `batch_size` 时是否丢掉 |

默认 `collate_fn` 把一个 batch 里各样本的第 0 个返回值 stack 成 `imgs`，第 1 个 stack 成 `targets`。因此 CIFAR 单张 `[3, 32, 32]`、`batch_size=4` 时：

```python
imgs.shape   # torch.Size([4, 3, 32, 32])
targets      # tensor([4, 9, 3, 9])  四个类别编号
```

`batch_size=64` 则为 `[64, 3, 32, 32]`。

### 一个 batch 是怎么拼出来的（collate 拆步骤）

以 `batch_size=4` 为例，DataLoader 每产出一个 batch，内部做了 4 步：

1. **定下标**：采样器（sampler）给出 4 个下标；`shuffle=False` 时是 `[0, 1, 2, 3]`，`shuffle=True` 时是随机的 4 个。
2. **逐个取样本**：对每个下标调用 `dataset[i]`，得到 4 个元组 `(img, target)`，每个 `img` 形状 `[3, 32, 32]`，`target` 是 int。
3. **按位置分组**：把 4 个元组的第 0 项放在一起（4 张图），第 1 项放在一起（4 个 int）。
4. **堆叠**：图用 `torch.stack` 在最前面新增一维 → `[4, 3, 32, 32]`；int 变成 `tensor([...])`，形状 `[4]`，dtype 为 `int64`。

用 FakeData 手动模拟一遍，并与 PyTorch 自带的 `default_collate` 对比（实测输出）：

```python
import torch
from torch.utils.data import default_collate

samples = [fake_t[i] for i in range(4)]            # 第 1、2 步：取下标 0~3 的 4 个样本
print([s[0].shape for s in samples])               # 4 个 torch.Size([3, 32, 32])
print([s[1] for s in samples])                     # [7, 9, 4, 3]  4 个 int

imgs = torch.stack([s[0] for s in samples])        # 第 3、4 步：图在第 0 维堆叠
targets = torch.tensor([s[1] for s in samples])    # 标签变成一维 Tensor
print(imgs.shape, targets)                         # torch.Size([4, 3, 32, 32]) tensor([7, 9, 4, 3])

b = default_collate(samples)                       # DataLoader 默认用的就是这个函数
print(type(b).__name__, len(b))                    # list 2  注意：得到的是长度为 2 的 list，不是 tuple
print(b[0].shape, b[1], b[1].dtype)                # torch.Size([4, 3, 32, 32]) tensor([7, 9, 4, 3]) torch.int64
print(torch.equal(b[0], imgs))                     # True  与手动 stack 结果完全相同
```

所以 `imgs, targets = data` 能解包，是因为 `data` 是一个两项的 list：`[图像批, 标签批]`。

测试集 10000 张、`batch_size=64`：`10000 = 156×64 + 16`。

- `drop_last=False`：157 批，最后一批 16 张，形状 `[16, 3, 32, 32]`
- `drop_last=True`：156 批，全部 `[64, 3, 32, 32]`，后面若把 `Linear` 的 `in_features` 按 batch=64 写死，就不会因最后一批形状变化而报错

批数的通用公式（$N$ 为样本数，$B$ 为 batch_size）：`drop_last=False` 时向上取整，`drop_last=True` 时向下取整：

$$
\text{批数}_{\text{保留}} = \lceil N/B \rceil, \qquad \text{批数}_{\text{丢弃}} = \lfloor N/B \rfloor
$$

代入 $N=10000, B=64$：$10000/64=156.25$，向上取整 157、向下取整 156；余数 $10000-156\times64=16$。`len(loader)` 返回的就是批数。用 `FakeData(size=10000)` 实测：

| batch_size | drop_last | `len(loader)` | 最后一批张数 |
| :--- | :--- | :--- | :--- |
| 64 | False | 157 | 16 |
| 64 | True | 156 | 64 |
| 4 | False | 2500 | 4（正好整除，两种设置结果一样） |

> [!EXAMPLE]+ 为什么 Linear 会和 batch 绑在一起
>
> 正常写法 `torch.flatten(imgs, start_dim=1)` 保留 batch 维，`[64, 3, 32, 32] -> [64, 3072]`，`Linear(3072, 10)` 与 batch 大小无关。课程后面“线性层”一节为了演示，把**整批**拉成一行：`torch.reshape(imgs, (1, 1, 1, -1))` 得到 `[1, 1, 1, 196608]`（$64\times3\times32\times32=196608$），于是写了 `Linear(196608, 10)`。这种写法下最后一批只有 16 张，长度变成 $16\times3\times32\times32=49152$，实测报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x49152 and 196608x10)`，所以那里必须 `drop_last=True`。

单张展示用 `add_image`，一整批用 `add_images`。

`add_images` 默认要求 `[N, C, H, W]`（`dataformats='NCHW'`），正好是 DataLoader 给出的 `imgs`，TensorBoard 会把 64 张拼成一张网格图显示。

官方 `dataloader.py`：

```python
import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())  # 位置参数 "./dataset" 就是 root

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)  # 156 批，每批 64 张

# 测试数据集中第一张图片及target
img, target = test_data[0]  # 直接下标访问 Dataset，不经过 DataLoader
print(img.shape)  # torch.Size([3, 32, 32])  单张没有 batch 维
print(target)  # 一个 int

writer = SummaryWriter("dataloader")
for epoch in range(2):  # 把整个测试集扫 2 遍
    step = 0  # 每个 epoch 重新从 0 计数
    for data in test_loader:  # 每次循环取一个 batch，共 156 次
        imgs, targets = data  # imgs: [64, 3, 32, 32]，targets: [64]
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)  # 注意是 add_images（复数）
        step = step + 1

writer.close()
```

外层 `for epoch` 控制扫几遍数据；内层 `for data in loader` 取出一个 batch。`shuffle=True` 时两个 epoch 在 TensorBoard 同一滑条位置上的图像不同。`step` 按 batch 递增，作为 `add_images` 的 `global_step`。

用 FakeData 验证 shuffle（固定 `torch.manual_seed(0)` 后，两个 epoch 第一个 batch 的前 6 个标签）：epoch 0 为 `tensor([7, 4, 1, 8, 9, 9])`，epoch 1 为 `tensor([9, 7, 0, 1, 2, 9])`，确实每轮重新洗牌。`shuffle=False` 时每轮顺序都一样。

> [!NOTE]+ 迭代约定
>
> `for data in test_loader` 里的 `data` 就是 `__getitem__` 返回值按 batch 堆好的结果。CIFAR 为 `(imgs, targets)`；若 Dataset 返回 dict，则 `data` 是 dict，键对应的 value 带了 batch 维。也可写成 `for imgs, targets in test_loader`，与先取 `data` 再解包等价。

实测 dict 版本：`__getitem__` 返回 `{"img": Tensor[3, 32, 32], "label": int}` 时，`batch_size=4` 取出的 `data` 是 dict，`data["img"].shape` 为 `torch.Size([4, 3, 32, 32])`，`data["label"]` 为 `tensor([0, 1, 2, 3])`。

### 常见报错与排查

| 报错 | 原因 | 修法 |
| :--- | :--- | :--- |
| `TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>` | Dataset 没有 `transform=ToTensor()`，返回的是 PIL，无法 stack | 给数据集加 `ToTensor` |
| `RuntimeError: stack expects each tensor to be equal size, but got [3, 32, 32] at entry 0 and [3, 33, 32] at entry 1` | 同一批里图片尺寸不同（自己的数据集常见） | transform 里加 `Resize((H, W))` 统一尺寸 |
| Windows 下先报 `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`，再报 `RuntimeError: DataLoader worker (pid ...) exited unexpectedly`（部分环境是 `BrokenPipeError`） | `num_workers>0` 但入口代码没放进 `if __name__ == '__main__':` | 调试先用 `num_workers=0`，或加 main 保护 |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` 只在最后一批出现 | 最后一批张数不足，形状变了 | `drop_last=True`，或 flatten 时保留 batch 维 |

> [!TIP]+ 小结
>
> - DataLoader 把 `batch_size` 个 `(img, target)` 用 `default_collate` 堆成 `[imgs, targets]`，形状 `[B, 3, 32, 32]` 和 `[B]`。
> - 批数：`drop_last=False` 是 $\lceil N/B \rceil$，`True` 是 $\lfloor N/B \rfloor$；10000/64 → 157 或 156，余 16。
> - `shuffle=True` 每个 epoch 重新洗牌；Windows 调试时 `num_workers=0`。
> - 单张用 `add_image`，一批用 `add_images`（NCHW）。

<br>

## 搭网络的骨架：nn.Module（P16）

> [!NOTE]+ 这一节要解决什么
>
> 目的：学会写一个最小的神经网络类，理解“定义层”和“定义计算”分别写在哪、调用 `model(x)` 时内部发生了什么。
>
> 位置：数据 → 变换 → 打包 → **模型** → 损失 → 优化 → 训练循环 → ……
>
> 衔接：上一节 DataLoader 给出 `[64, 3, 32, 32]` 的 `imgs`；这一节的骨架就是接收 `imgs` 的容器；下一节起往骨架里放卷积、池化、线性等具体层。

对应视频 P16。神经网络写在 `torch.nn`。自定义模型必须继承 `nn.Module`，在 `__init__` 里**构造并注册**层，在 `forward` 里写数据怎么流。

写一个自定义网络固定 4 步：

1. **继承**：`class Xxx(nn.Module):`，这样才能用到父类提供的 `__call__`、`parameters()`、`to(device)`、`train()/eval()` 等功能。
2. **初始化父类**：`__init__` 第一行写 `super().__init__()`，让父类先建好存放参数和子模块的“登记簿”。
3. **注册层**：在 `__init__` 里用 `self.xxx = nn.层(...)` 创建层，自动登记进登记簿。
4. **写 forward**：`def forward(self, x):` 描述输入如何依次经过各层，最后 `return` 输出。之后用 `model = Xxx()` 和 `output = model(x)` 使用。

`torch.nn` 与 `torch.nn.functional`（常 `import torch.nn.functional as F`）的关系：后者是函数（每次传入 weight）；前者把 weight 存成模块参数，适合搭网络。课程后半用 `nn.Conv2d` 等层，手算卷积时用 `F.conv2d`。

| | `nn.Conv2d`（类） | `F.conv2d`（函数） |
| :--- | :--- | :--- |
| 用法 | `conv = nn.Conv2d(1, 20, 5)` 先构造，再 `conv(x)` | `F.conv2d(x, weight, bias)` 一次调用 |
| 权重 | 自己创建并保存为 `conv.weight`，会被优化器更新 | 需要自己传进去 |
| 适合 | 搭网络（有可学习参数的层） | 手算、没有参数的操作（如 `F.relu`） |

官方文档里的典型写法：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):  # 第 1 步：继承 nn.Module
    def __init__(self):
        super().__init__()  # 第 2 步：初始化父类，必须在给 self 挂层之前
        self.conv1 = nn.Conv2d(1, 20, 5)  # 第 3 步：输入 1 通道 -> 输出 20 通道，5x5 卷积核
        self.conv2 = nn.Conv2d(20, 20, 5)  # 输入通道必须等于上一层输出通道 20

    def forward(self, x):  # 第 4 步：x 形状 [N, 1, H, W]
        x = F.relu(self.conv1(x))  # [1, 1, 28, 28] -> [1, 20, 24, 24]，relu 不改形状
        return F.relu(self.conv2(x))  # [1, 20, 24, 24] -> [1, 20, 20, 20]
```

对上面的 `Model` 实际运行（输入 `torch.rand(1, 1, 28, 28)`）：

```python
m = Model()
print(m)
# Model(
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
# )
for name, p in m.named_parameters():       # 遍历所有登记过的参数
    print(name, tuple(p.shape), p.numel())
# conv1.weight (20, 1, 5, 5) 500
# conv1.bias (20,) 20
# conv2.weight (20, 20, 5, 5) 10000
# conv2.bias (20,) 20
print(sum(p.numel() for p in m.parameters()))  # 10540
x = torch.rand(1, 1, 28, 28)                # [N=1, C=1, H=28, W=28]
print(m.conv1(x).shape, m(x).shape)         # torch.Size([1, 20, 24, 24]) torch.Size([1, 20, 20, 20])
```

两个数可以手算验证：

- **输出尺寸**（下一节“卷积”会细讲）：$H_{out} = \lfloor (H_{in} + 2p - k)/s \rfloor + 1$。这里 $p=0, s=1, k=5$：$28-5+1=24$，$24-5+1=20$。
- **参数量**：卷积权重 = 输出通道 × 输入通道 × $k \times k$，偏置 = 输出通道。conv1：$20\times1\times5\times5=500$，加 20 个偏置 = 520；conv2：$20\times20\times5\times5=10000$，加 20 = 10020；合计 $520+10020=10540$，与打印一致。

必须遵守：

1. `__init__` 里调用 `super().__init__()`（或 `super(Tudui, self).__init__()`），它负责建好存放参数和子层的内部字典；漏写时，新版 PyTorch 在给 `self.xxx` 赋层时直接报 `AttributeError`（见下文实测）
2. 方法名必须是 `forward`，拼错则调用实例时不会走到自定义计算
3. 层写成 `self.conv1 = ...`；若只写成局部变量 `conv1 = ...`，则不算子模块，不会出现在 `parameters()` 里

### 为什么 `super().__init__()` 必须写

`nn.Module.__init__` 会在实例上建好一批内部字典：`_parameters`（存参数）、`_modules`（存子层）、`_buffers`、各种 `_xxx_hooks` 等。之后每次写 `self.conv1 = nn.Conv2d(...)`，`nn.Module` 重写过的 `__setattr__` 会发现“赋的是一个 Module”，于是把它放进 `self._modules["conv1"]`；`parameters()` 就是遍历这些字典。不调用父类 `__init__`，这些字典根本不存在。实测忘写时的表现：

| 情况 | 实际结果 |
| :--- | :--- |
| 忘写 super，`__init__` 里 `self.conv1 = nn.Conv2d(...)` | 构造时立刻报 `AttributeError: cannot assign module before Module.__init__() call` |
| 忘写 super，`__init__` 里 `self.w = nn.Parameter(...)` | 报 `AttributeError: cannot assign parameters before Module.__init__() call` |
| 忘写 super，`__init__` 里只有 `pass`（如 Tudui） | 构造成功，但调用 `model(x)` 时报 `AttributeError: 'NoSuper3' object has no attribute '_backward_hooks'` |

> [!WARNING]+ 易错：忘写 `super().__init__()` 的表现
>
> 新版 PyTorch 中忘写 `super().__init__()` 不会“静默地不注册参数”，而是直接报上表的 `AttributeError`（赋值层时报，或调用时报）。看到 `cannot assign module before Module.__init__() call`，第一反应就是检查 `super().__init__()` 是否漏写、是否写在了 `self.xxx = ...` 之后。之所以必须写，是因为参数和子层要靠父类建好的字典登记，`model.parameters()` 和优化器才找得到它们。

### forward 拼错与局部变量的实测

- 写成 `def foward(self, x)`：调用 `model(x)` 报 `NotImplementedError: Module [Typo] is missing the required "forward" function`。父类自带一个占位 `forward`，没被覆盖就会报这个错。
- `__init__` 里写局部变量 `conv1 = nn.Conv2d(1, 20, 5)`、另外写 `self.conv2 = ...`：`named_parameters()` 只有 `['conv2.weight', 'conv2.bias']`，`conv1` 在 `__init__` 结束后就丢了，`forward` 里也用不了。
- 用 Python 普通列表 `self.layers = [nn.Linear(2, 2), nn.Linear(2, 2)]` 存层：`parameters()` 长度为 `0`，列表里的层不会被登记。需要改用 `nn.ModuleList([...])` 或 `nn.Sequential(...)`（后面“串联”一节）。

实例 `tudui(x)` 并没有手写 `__call__`，但父类 `nn.Module` 实现了 `__call__`，内部会调用 `forward`。因此应使用 `tudui(x)`，而不是直接 `tudui.forward(x)`（后者会跳过 hook）。

> [!WARNING]+ 易错：直接调 `forward` 跳过的是什么
>
> 直接调 `forward` 跳过的是 **hook**（`register_forward_pre_hook`、`register_forward_hook`、反向 hook 等，由 `__call__` 负责执行）。`train()/eval()` 只是设置每个模块的 `self.training` 标志，`Dropout`、`BatchNorm` 是在**自己的 forward 里**读这个标志，所以直接调 `forward` 时 train/eval 照样生效（实测 `nn.Dropout(0.5)` 在 `eval()` 后 `forward(torch.ones(4))` 输出 `tensor([1., 1., 1., 1.])`，`train()` 后输出 `tensor([2., 0., 0., 2.])`）。容易误以为 train/eval 也会失效，其实不会；但为了 hook 正常工作，始终写 `model(x)`。

hook 被跳过的实测：给 `tudui` 注册一个 `register_forward_hook` 打印输出后，`tudui(x)` 依次打印 `in forward` 和 `hook fired, out = tensor(2.)`；`tudui.forward(x)` 只打印 `in forward`，hook 没有执行。

官方 `nn_module.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from torch import nn


class Tudui(nn.Module):  # 继承 nn.Module
    def __init__(self):
        super().__init__()  # 这里没有可学习的层，但仍要初始化父类

    def forward(self, input):  # 参数名叫 input 只是习惯，叫 x 也行
        output = input + 1  # 最简单的“网络”：每个元素加 1
        return output


tudui = Tudui()  # 构造：执行 __init__
x = torch.tensor(1.0)  # 0 维 Tensor（标量），shape 为 torch.Size([])
output = tudui(x)  # 调用实例：Module.__call__ -> forward
print(output)  # tensor(2.)
```

输入 `1.0` 时输出 `tensor(2.)`。可用 PyCharm 的 Step into My Code 跟进：

1. `tudui = Tudui()` → `__init__` → `super().__init__()`
2. `tudui(x)` → `Module.__call__` → `Tudui.forward`
3. `input + 1` 后返回

把第 2 步展开，`Module.__call__`（源码里是 `__call__ = _wrapped_call_impl`，它再转调 `_call_impl`）大致做了这些事：

1. 依次执行已注册的 forward **pre-hook**（可在进 forward 前修改输入）；
2. 调用 `self.forward(*args, **kwargs)`，也就是 `Tudui.forward(tudui, x)`；
3. 依次执行已注册的 forward **hook**（可查看或修改输出）；
4. 如有反向 hook，给输出挂上钩子；
5. 把结果返回给 `output`。

没有注册任何 hook 时，1、3、4 什么都不做，所以看起来就像“直接调用了 forward”。实测 `Tudui.__call__ is nn.Module.__call__` 为 `True`：`Tudui` 自己没有定义 `__call__`，用的是继承来的那个。这和本节开头 `Person` 的例子是同一个机制，只是 `__call__` 由父类写好了。

这是后面所有 `Conv2d` / `Sequential` 网络的最小骨架。

### 常见报错与排查

| 报错 | 原因 | 修法 |
| :--- | :--- | :--- |
| `AttributeError: cannot assign module before Module.__init__() call` | 漏写 `super().__init__()`，或写在 `self.xxx = 层` 之后 | `__init__` 第一行写 `super().__init__()` |
| `NotImplementedError: Module [Xxx] is missing the required "forward" function` | `forward` 拼错（`foward`、`Forward`）或缩进错了没写进类里 | 改成 `def forward(self, x):` |
| `RuntimeError: Given groups=1, weight of size [20, 1, 5, 5], expected input[1, 3, 28, 28] to have 1 channels, but got 3 channels instead` | 输入通道数和 `Conv2d` 的 `in_channels` 不一致 | 让 `in_channels` 等于输入的 C（CIFAR 是 3） |
| `RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [28, 28]` | 输入少了维度 | 补维度：`x.reshape(1, 1, 28, 28)` 或 `x.unsqueeze(0)` |
| `TypeError: Model.__init__() takes 1 positional argument but 2 were given` | 把数据传给了构造函数：`Model(x)` | 先 `m = Model()`，再 `m(x)` |

> [!TIP]+ 小结
>
> - 自定义网络 4 步：继承 `nn.Module` → `super().__init__()` → `self.层 = ...` 注册 → 写 `forward`。
> - `model(x)` 走父类的 `__call__`，它负责执行 hook 并调用 `forward`；不要直接调 `forward`。
> - 漏写 `super().__init__()` 会报 `cannot assign module before Module.__init__() call`；`forward` 拼错会报 `missing the required "forward" function`。
> - 只有挂在 `self` 上的层（或 `nn.ModuleList`/`nn.Sequential` 里的层）才会出现在 `parameters()` 里、被优化器更新。
> - 卷积参数量 = $C_{out}\times C_{in}\times k\times k + C_{out}$；输出边长 = $\lfloor (H+2p-k)/s\rfloor+1$（向下取整）。

<br>

## 卷积：Conv2d 怎么算、输出尺寸怎么定（P17–P18）

> [!NOTE]+ 这一节要解决什么
>
> 目的：弄懂“卷积核在图上滑动、相乘求和”到底算的是什么，以及输出的 shape 和参数量怎么提前算出来。
>
> 位置：训练流程 数据 → 变换 → 打包 → **模型** → 损失 → 优化 → 训练循环 → 保存/验证 里的“模型”一环。上一节用 `nn.Module` 搭好了骨架（空壳），这一节开始往骨架里放第一种真正干活的层；后面的池化、激活、全连接都是同样的思路：**先弄懂单层的输入输出 shape，再串起来**。

对应视频 P17（运算）与 P18（层）。图像在 PyTorch 里默认 **NCHW**：`N` batch，`C` 通道，`H` 高，`W` 宽。

举例：一批 64 张 CIFAR-10 彩色图，`ToTensor()` 之后再经 DataLoader 打包，形状就是 `[64, 3, 32, 32]`，即 N=64、C=3（RGB）、H=W=32。后面所有层的 shape 推导都从这个四元组出发。

### 先用函数手算：F.conv2d（P17）

二维卷积：卷积核在输入上滑动，对应位置相乘再求和。`F.conv2d` 的主要参数：

```python
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```

| 参数 | 形状 / 含义 |
| :--- | :--- |
| `input` | `(N, C_in, H, W)` |
| `weight`（卷积核） | `(C_out, C_in/groups, kH, kW)` |
| `bias` | `(C_out,)`，可选 |
| `stride` | 每次滑动格数，整数或 `(sH, sW)` |
| `padding` | 四周补零圈数；`'same'` 保持尺寸但要求 stride=1 |
| `dilation` | 核元素间距，空洞卷积，默认 1 |

几个名词先对上号：

- **卷积核 / kernel / weight**：一个小矩阵（如 `3×3`），里面的数就是要学的参数。`F.conv2d` 里需要手动传；`nn.Conv2d` 里由层自己随机初始化并在训练中更新。
- **步长 stride**：核每次往右（或往下）挪几格。stride 越大，输出越小。
- **填充 padding**：在输入四周补几圈 0，让边缘像素也能被核“扫到”，也用来控制输出尺寸。
- **偏置 bias**：每个输出通道一个数，算完乘加之后整体再加上它。

输出空间尺寸（`dilation=1` 时与课程公式一致）：

$$
n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - f}{s} \right\rfloor + 1
$$

公式里的符号：$n_{\text{in}}$ 输入边长，$p$ padding，$f$ 卷积核边长（filter），$s$ stride，$\lfloor\cdot\rfloor$ 向下取整。高和宽分别代入，各算各的。官方文档的完整版本带 dilation $d$：

$$
n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - d\,(f-1) - 1}{s} \right\rfloor + 1
$$

$d=1$ 时 $d(f-1)+1=f$，就退化成上面的课程公式。

代入数字练手（每一行都已在 PyTorch 2.6 里跑过确认）：

| 输入边长 | 核 f | stride s | padding p | 代入 | 输出边长 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 5 | 3 | 1 | 0 | $(5+0-3)/1+1$ | 3 |
| 5 | 3 | 2 | 0 | $(5+0-3)/2+1$ | 2 |
| 5 | 3 | 1 | 1 | $(5+2-3)/1+1$ | 5 |
| 32 | 3 | 1 | 0 | $(32+0-3)/1+1$ | 30 |
| 32 | 5 | 1 | 2 | $(32+4-5)/1+1$ | 32 |
| 32 | 5 | 2 | 2 | $\lfloor (32+4-5)/2 \rfloor+1=\lfloor 15.5\rfloor+1$ | 16 |

最后一行说明了“向下取整”的作用：除不尽时，最右/最下那一点放不下一个完整窗口，直接丢掉。

通道关系：

- 输入通道数 = 单个卷积核的通道数
- 输出通道数 = 卷积核个数（`out_channels`）

每个核与输入**所有通道**做相关后再相加，得到 **1** 个输出通道；`out_channels` 个核就得到相应数量的特征图。

用全 1 的张量可以直观验证“所有通道相加”：输入 `torch.ones(1, 3, 5, 5)`，核 `torch.ones(2, 3, 3, 3)`（2 个核，每个核 3 通道），输出 `[1, 2, 3, 3]`，每个位置的值是 27 = 3 个通道 × 9 个格子 × 1×1。如果核的通道数和输入对不上（例如输入 3 通道、核写成 `(2, 1, 3, 3)`），会报：

```text
RuntimeError: Given groups=1, weight of size [2, 1, 3, 3], expected input[1, 3, 5, 5] to have 1 channels, but got 3 channels instead
```

读法：weight 的第 2 维（1）就是它期望的输入通道数，而实际输入第 2 维是 3。

> [!EXAMPLE]+ 先手算一个更小的例子（4×4 输入，2×2 核）
>
> 输入 $x$ 与核 $k$：
>
> ```text
> x = [[1, 0, 2, 1],        k = [[1, 0],
>      [3, 1, 0, 2],             [1, 2]]
>      [0, 2, 1, 1],
>      [1, 0, 3, 0]]
> ```
>
> **第 1 步：先算输出尺寸。** stride=1、padding=0：$(4+0-2)/1+1=3$，输出 `3×3`。
>
> **第 2 步：核放在左上角，对应位置相乘再求和。** 左上 `2×2` 区域是 `[[1, 0], [3, 1]]`：$1\cdot1+0\cdot0+3\cdot1+1\cdot2=6$。
>
> **第 3 步：向右挪 1 格（stride=1），重复。** 区域 `[[0, 2], [1, 0]]`：$0\cdot1+2\cdot0+1\cdot1+0\cdot2=1$；再挪 1 格，区域 `[[2, 1], [0, 2]]`：$2+0+0+4=6$。
>
> **第 4 步：一行走完后往下挪 1 格，从最左边重新开始。** 9 个位置全部算完：
>
> ```text
> (0,0) 1·1+0·0+3·1+1·2 = 6    (0,1) 0·1+2·0+1·1+0·2 = 1    (0,2) 2·1+1·0+0·1+2·2 = 6
> (1,0) 3·1+1·0+0·1+2·2 = 7    (1,1) 1·1+0·0+2·1+1·2 = 5    (1,2) 0·1+2·0+1·1+1·2 = 3
> (2,0) 0·1+2·0+1·1+0·2 = 1    (2,1) 2·1+1·0+0·1+3·2 = 8    (2,2) 1·1+1·0+3·1+0·2 = 4
> ```
>
> **stride=2：** 尺寸 $\lfloor(4-2)/2\rfloor+1=2$。核只停在第 0、2 行与第 0、2 列，相当于从上面 9 个结果里挑 (0,0)、(0,2)、(2,0)、(2,2)：`[[6, 6], [1, 4]]`。
>
> **padding=1：** 四周补一圈 0 变成 `6×6`，尺寸 $(4+2-2)/1+1=5$。左上角窗口是 `[[0, 0], [0, 1]]`（三个补的 0 加上输入里的 $x_{00}=1$），结果 $0+0+0+1\cdot2=2$。补零后输出的中间 `3×3` 恰好等于不补零时的结果，多出来的一圈来自“核压在边缘、只盖住部分原图”的位置。
>
> **加 bias：** 传 `bias=torch.tensor([10.])`，每个位置都加 10，stride=1 结果变成 `[[16, 11, 16], [17, 15, 13], [11, 18, 14]]`。

用代码验证上面的手算（实际运行输出写在注释里）：

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1., 0, 2, 1],                # 写 1. 让整个张量是 float32
                  [3, 1, 0, 2],
                  [0, 2, 1, 1],
                  [1, 0, 3, 0]]).reshape(1, 1, 4, 4)   # [4, 4] -> [1, 1, 4, 4]，N=1, C=1
k = torch.tensor([[1., 0],
                  [1, 2]]).reshape(1, 1, 2, 2)         # [2, 2] -> [1, 1, 2, 2]，C_out=1, C_in=1

print(F.conv2d(x, k))             # [1,1,4,4] -> [1,1,3,3]
# [[6., 1., 6.],
#  [7., 5., 3.],
#  [1., 8., 4.]]
print(F.conv2d(x, k, stride=2))   # [1,1,4,4] -> [1,1,2,2]
# [[6., 6.],
#  [1., 4.]]
print(F.conv2d(x, k, padding=1))  # [1,1,4,4] -> 补零成 6×6 -> [1,1,5,5]
# [[2., 1., 4., 4., 1.],
#  [6., 6., 1., 6., 3.],
#  [0., 7., 5., 3., 3.],
#  [2., 1., 8., 4., 1.],
#  [0., 1., 0., 3., 0.]]
print(F.conv2d(x, k, bias=torch.tensor([10.])))  # 每个位置 +10
# [[16., 11., 16.],
#  [17., 15., 13.],
#  [11., 18., 14.]]
```

官方 `nn_conv.py` 用 `5×5` 输入与 `3×3` 核对 `stride`、`padding` 做数值验算：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆

import torch
import torch.nn.functional as F  # 函数式接口：不建层，直接调用，核要自己传

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))  # [5, 5] -> [1, 1, 5, 5]：补上 N=1、C=1
kernel = torch.reshape(kernel, (1, 1, 3, 3))  # [3, 3] -> [1, 1, 3, 3]：C_out=1、C_in=1

print(input.shape)  # torch.Size([1, 1, 5, 5])
print(kernel.shape)  # torch.Size([1, 1, 3, 3])

output = F.conv2d(input, kernel, stride=1)  # (5-3)/1+1=3 -> [1, 1, 3, 3]
print(output)

output2 = F.conv2d(input, kernel, stride=2)  # floor((5-3)/2)+1=2 -> [1, 1, 2, 2]
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)  # (5+2-3)/1+1=5 -> [1, 1, 5, 5]
print(output3)
```

`input` 与 `kernel` 需 reshape 成 `(N, C, H, W)` 与 `(C_out, C_in, kH, kW)`。缺维时常见报错是 `weight should have at least three dimensions`。`padding=1` 表示上下左右各补一圈 0，输入从 `5×5` 变成 `7×7`。

补充：上面那条报错对应的是**核**缺维（input 已 reshape、kernel 还是 `[3, 3]`）。如果是**输入**没 reshape、还是 `[5, 5]`，报错信息不同：

```text
RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [5, 5]
```

“3D (unbatched)” 的意思是：PyTorch 也接受不带 batch 维的 `[C, H, W]`，当作一张图处理；但 `[H, W]` 这种二维的不行，必须至少补出通道维。

> [!WARNING]+ 易错：卷积的数据类型
>
> 官方 `nn_conv.py` 没有指定 dtype，`torch.tensor([[1, 2, 0, 3, 1], ...])` 这样只写整数时默认是整型 `int64`。实测 PyTorch 2.6（CPU）：整型输入配整型核直接调用 `F.conv2d` **可以运行**，输出也是整型张量 `[[10, 12, 12], [18, 16, 16], [13, 9, 3]]`。真正出错的是**类型混用**：把整型输入喂给 `nn.Conv2d`（它的 weight / bias 是 float32），报 `RuntimeError: Input type (__int64) and bias type (float) should be the same`；GPU 上的卷积实现一般也不支持整型。所以做卷积一律用 `dtype=torch.float32`（或写成 `1.` 这样的小数字面量），后面池化示例已使用 `float32`。

左上角 `3×3` 与核的点积（stride=1 的第一个输出）：

$$
\begin{aligned}
&1\cdot1 + 2\cdot2 + 0\cdot1 \\
+\; &0\cdot0 + 1\cdot1 + 2\cdot0 \\
+\; &1\cdot2 + 2\cdot1 + 1\cdot0
= 10
\end{aligned}
$$

再算两个，确认规律：

- stride=1 的第一行第二个输出：窗口右移 1 格，区域是 `[[2, 0, 3], [1, 2, 3], [2, 1, 0]]`，乘核 `[[1, 2, 1], [0, 1, 0], [2, 1, 0]]` 得 $2+0+3+0+2+0+4+1+0=12$。
- stride=2 的第一行第二个输出：窗口直接右移 2 格，区域 `[[0, 3, 1], [2, 3, 1], [1, 0, 0]]`，得 $0+6+1+0+3+0+2+0+0=12$，所以 stride=2 的结果就是 stride=1 结果里隔一个取一个：`(0,0)、(0,2)、(2,0)、(2,2)` → `[[10, 12], [13, 3]]`。

运行结果（与视频一致）：

```text
stride=1, padding=0  →  3×3
[[10, 12, 12],
 [18, 16, 16],
 [13,  9,  3]]

stride=2, padding=0  →  2×2
[[10, 12],
 [13,  3]]

stride=1, padding=1  →  5×5（尺寸回到输入）
[[ 1,  3,  4, 10,  8],
 [ 5, 10, 12, 12,  6],
 [ 7, 18, 16, 16,  8],
 [11, 13,  9,  3,  4],
 [14, 13,  9,  7,  4]]
```

尺寸验算：$(5+0-3)/1+1=3$，$(5+0-3)/2+1=2$，$(5+2-3)/1+1=5$。

padding=1 的结果同样满足“中间 `3×3` 等于不补零的结果”：第 1–3 行、第 1–3 列正好是 `[[10, 12, 12], [18, 16, 16], [13, 9, 3]]`。

### 再用层：nn.Conv2d（P18）

网络里用 `nn.Conv2d`（层里带可学习的 weight / bias），而不是每次手传 kernel。完整签名：

```python
torch.nn.Conv2d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, dilation=1, groups=1,
    bias=True, padding_mode='zeros'
)
```

`padding_mode` 可为 `'zeros'` / `'reflect'` / `'replicate'` / `'circular'`。保持特征图边长常用 `padding = (kernel_size - 1) // 2`（奇数核）。

`F.conv2d` 和 `nn.Conv2d` 的区别：

| | `F.conv2d` | `nn.Conv2d` |
| :--- | :--- | :--- |
| 核从哪来 | 调用时自己传 `weight` | 构造时自动创建并随机初始化 |
| 能否训练 | 自己传的张量需要自己管理 | `weight`、`bias` 是 `nn.Parameter`，会出现在 `model.parameters()` 里交给优化器 |
| 需要写什么 | 核的形状 `(C_out, C_in, kH, kW)` | 只写 `in_channels`、`out_channels`、`kernel_size`，形状自动推出来 |
| 用途 | 理解原理、手算验证 | 搭网络 |

**参数量怎么算。** 一个 `Conv2d(in_channels, out_channels, kernel_size=k)` 有：

$$
\text{参数量} = \underbrace{C_{\text{out}} \times C_{\text{in}} \times k \times k}_{\text{weight}} + \underbrace{C_{\text{out}}}_{\text{bias}}
$$

代入 `Conv2d(3, 6, 3)`：weight 形状 `[6, 3, 3, 3]`，$6\times3\times3\times3=162$；bias 形状 `[6]`，共 $162+6=168$。注意参数量**和输入图片大小无关**：同一个核在整张图上滑动共用，32×32 还是 224×224 都是 168 个参数。

```python
from torch import nn

c = nn.Conv2d(3, 6, 3)
print(c.weight.shape)   # torch.Size([6, 3, 3, 3])  = (C_out, C_in, kH, kW)
print(c.bias.shape)     # torch.Size([6])
print(sum(p.numel() for p in c.parameters()))  # 168
```

CIFAR 输入 `[64, 3, 32, 32]`，`Conv2d(3, 6, 3)` 后为 `[64, 6, 30, 30]`，因为 $(32+0-3)/1+1=30$，6 个核 → 6 通道。`add_images` 只接受 3 通道，仓库里用 `reshape` 把 6 通道拆成更多张 3 通道图：`[64, 6, 30, 30] → [128, 3, 30, 30]`（`-1` 让 batch 维自动算）。这只是为了能显示，**不是**训练时的标准操作，通道被拆开后语义已乱。

`-1` 是怎么算出 128 的：元素总数 $64\times6\times30\times30$ 不变，除以已知的 $3\times30\times30$ 得 $64\times6/3=128$。

跑官方 `nn_conv2d.py` 的步骤：

1. **准备数据**：CIFAR10 测试集 + `ToTensor()`，`DataLoader(batch_size=64)` 打包，每批 `imgs` 是 `[64, 3, 32, 32]`。
2. **定义模型**：继承 `nn.Module`，在 `__init__` 里建一个 `Conv2d(3, 6, 3)`，在 `forward` 里调用它。
3. **实例化**：`tudui = Tudui()`，此时卷积核已随机初始化好（没有训练，所以输出图看起来像“随机滤镜”）。
4. **循环取批次，前向一次**：`output = tudui(imgs)`，形状 `[64, 6, 30, 30]`。
5. **写进 TensorBoard**：输入直接 `add_images`；输出先 reshape 成 3 通道再写。
6. **关掉 writer**，然后 `tensorboard --logdir=../logs` 打开对比。

官方 `nn_conv2d.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),  # 测试集 10000 张，每张 [3, 32, 32]
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)  # 每批 imgs: [64, 3, 32, 32]，最后一批 10000 % 64 = 16 张

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)  # weight [6, 3, 3, 3]，共 168 个参数

    def forward(self, x):
        x = self.conv1(x)  # [64, 3, 32, 32] -> [64, 6, 30, 30]
        return x

tudui = Tudui()

writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)  # 直接调用实例，内部会走 forward
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]

    output = torch.reshape(output, (-1, 3, 30, 30))  # [64, 6, 30, 30] -> [128, 3, 30, 30]，仅为显示
    writer.add_images("output", output, step)

    step = step + 1

writer.close()
```

> [!WARNING]+ 常见报错与排查（卷积）
>
> | 报错 | 原因 | 修法 |
> | :--- | :--- | :--- |
> | `Given groups=1, weight of size [6, 3, 3, 3], expected input[1, 1, 32, 32] to have 3 channels, but got 1 channels instead` | 输入通道数 ≠ `in_channels`（如灰度图喂给 `in_channels=3` 的层） | 改 `in_channels`，或把图转成 RGB（`Image.convert("RGB")`） |
> | `Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [32, 32]` | 输入只有 `[H, W]` | `reshape` / `unsqueeze` 补出通道维和 batch 维 |
> | `weight should have at least three dimensions` | `F.conv2d` 的核没 reshape 成 4 维 | `kernel.reshape(C_out, C_in, kH, kW)` |
> | `Input type (__int64) and bias type (float) should be the same` | 整型输入喂给 `nn.Conv2d` | 构造时 `dtype=torch.float32`，或 `.float()` |
> | `padding='same' is not supported for strided convolutions` | `padding='same'` 搭配了 `stride=2` | 改用整数 padding，按公式自己算 |
>
> 排查顺序：先 `print(x.shape, x.dtype)`，再对照层的 `weight.shape`，第 2 维就是它要的输入通道数。

> [!TIP]+ 小结
>
> - 卷积 = 核在图上滑动，对应位置相乘求和（再加 bias）；一个核把**所有输入通道**合成**一个**输出通道。
> - 输出边长 $\lfloor (n+2p-f)/s \rfloor + 1$，H、W 各算一遍；stride=1 时 `padding=(k-1)//2` 保持尺寸。
> - weight 形状 `(C_out, C_in, kH, kW)`；参数量 $C_{\text{out}} C_{\text{in}} k^2 + C_{\text{out}}$，与图片大小无关。
> - 输入必须是 `[N, C, H, W]`（或单张 `[C, H, W]`），类型用 float32。
> - `F.conv2d` 用来理解和手算，`nn.Conv2d` 用来搭网络。

<br>

## 池化：MaxPool2d 与 ceil_mode 手算（P19）

> [!NOTE]+ 这一节要解决什么
>
> 目的：在不加任何参数的前提下把特征图“缩小”，减少后续计算量，同时保留每块区域里最显著的特征。
>
> 位置：仍在“模型”一环。卷积负责提取特征（通道变多），池化负责压缩尺寸（边长变小）；CIFAR-10 网络里两者交替出现。下一节的激活函数则负责给网络加入非线性。

对应视频 P19。最大池化在窗口内取最大值，用一个数代表一块区域，降低分辨率、减少计算。特点：

1. 无可学习参数（没有 weight 要更新）
2. 通道数不变：输入 3 通道，输出仍 3 通道
3. 对微小平移相对稳健

和卷积对比着记：

| | 卷积 `Conv2d` | 最大池化 `MaxPool2d` |
| :--- | :--- | :--- |
| 窗口里做什么 | 和核相乘再求和 | 取最大值 |
| 可学习参数 | 有（weight、bias） | 无，`sum(p.numel() for p in nn.MaxPool2d(2).parameters())` 为 0 |
| 通道 | 可以变（由 `out_channels` 决定） | 不变，每个通道各自池化 |
| 默认 stride | 1 | 等于 `kernel_size` |
| 尺寸公式 | $\lfloor (n+2p-f)/s \rfloor + 1$ | 相同；`ceil_mode=True` 时把向下取整换成向上取整 |

```python
torch.nn.MaxPool2d(
    kernel_size, stride=None, padding=0,
    dilation=1, return_indices=False, ceil_mode=False
)
```

| 参数 | 含义 |
| :--- | :--- |
| `kernel_size` | 窗口边长 |
| `stride` | 默认等于 `kernel_size`（与卷积默认 stride=1 不同） |
| `ceil_mode` | `False`（floor）丢掉不足一窗的边缘；`True`（ceil）保留不完整窗口 |
| `return_indices` | 为 True 时还返回最大值位置，供 `MaxUnpool2d` 使用 |

`stride=None` 的意思就是“和 `kernel_size` 一样”，所以 `MaxPool2d(2)` 的窗口 `2×2`、每次挪 2 格，窗口之间不重叠，边长正好减半。

官方 `nn_maxpool.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)  # imgs: [64, 3, 32, 32]

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)  # 窗口 3×3，stride 默认 3

    def forward(self, input):
        output = self.maxpool1(input)  # [64, 3, 32, 32] -> [64, 3, 10, 10]，floor((32-3)/3)+1=10
        return output

tudui = Tudui()

writer = SummaryWriter("../logs_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)  # 通道仍是 3，可以直接 add_images
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
```

在 TensorBoard 里看到的输出图像是“马赛克”效果：每 `3×3` 块只留一个最亮的值，边长从 32 变成 10。

手算示例（`ceil_mode=True`，核 `3×3`）：

```python
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)  # 直接给 float32，和后面的卷积层类型一致
input = torch.reshape(input, (-1, 1, 5, 5))  # [5, 5] -> [1, 1, 5, 5]

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)  # stride 默认 = 3

    def forward(self, input):
        return self.maxpool1(input)  # [1, 1, 5, 5] -> [1, 1, 2, 2]

print(Tudui()(input))  # tensor([[[[2., 3.], [5., 1.]]]])
```

注：实测 PyTorch 2.6 CPU 上整型张量做 `MaxPool2d` 也能跑（`ceil_mode=False` 得到 `tensor([[[[2]]]])`），但后续要和卷积等层配合，统一用 float32 最省心。

`5×5`、核 `3×3`、stride 默认 3：

- `ceil_mode=True`：能划出 2 个窗口方向，输出 `2×2`，约为 `[[2, 3], [5, 1]]`（左上窗最大 2，右上 3，左下 5，右下不完整窗最大 1）
- `ceil_mode=False`：右、下不足 3 的边丢掉，只剩左上一个窗，输出 `1×1` 的 `[[2.]]`

> [!EXAMPLE]+ ceil_mode 一步步手算
>
> **第 1 步：套尺寸公式。** $n=5, f=3, s=3, p=0$，$(5-3)/3 = 0.67$。
>
> - `ceil_mode=False`：$\lfloor 0.67 \rfloor + 1 = 1$，输出 `1×1`。
> - `ceil_mode=True`：$\lceil 0.67 \rceil + 1 = 2$，输出 `2×2`。
>
> **第 2 步：把 5×5 按 stride=3 切块。** 行、列都切成 `[0, 1, 2]` 和 `[3, 4]` 两段（第二段只有 2 格，不满 3）：
>
> ```text
>  1 2 0 | 3 1
>  0 1 2 | 3 1
>  1 2 1 | 0 0
> -------+-----
>  5 2 3 | 1 1
>  2 1 0 | 1 1
> ```
>
> **第 3 步：每块取最大值。** 左上块 max = 2；右上块 `[[3,1],[3,1],[0,0]]` max = 3；左下块 `[[5,2,3],[2,1,0]]` max = 5；右下块 `[[1,1],[1,1]]` max = 1。
>
> **第 4 步：决定保不保留不完整的块。** `ceil_mode=True` 保留 → `[[2, 3], [5, 1]]`；`ceil_mode=False` 只保留完整的左上块 → `[[2]]`。
>
> 用 `return_indices=True` 还能看到最大值在展平后 25 个格子里的下标：`[[1, 3], [15, 18]]`，即 (0,1) 处的 2、(0,3) 处的 3（第一个 3，并列时取先遇到的）、(3,0) 处的 5、(3,3) 处的 1。

再对比一个常见尺寸：`7×7` 输入、`MaxPool2d(2)`，`ceil_mode=False` 得 `3×3`（$\lfloor 5/2 \rfloor + 1$），`ceil_mode=True` 得 `4×4`（$\lceil 5/2 \rceil + 1$）。PyTorch 对 ceil 还有一条附加规则：最后一个窗口的起点必须落在输入（或左侧 padding）内，纯粹落在外面的窗口不算，所以不会出现“全是空格”的窗口。

CIFAR 上池化后仍是 3 通道，因此 `add_images` 不必像卷积那样 `reshape` 通道。

> [!TIP]+ 小结
>
> - 最大池化：窗口内取最大值；**没有参数**，**通道不变**，只缩小 H、W。
> - `stride` 默认等于 `kernel_size`，所以 `MaxPool2d(2)` 让边长减半。
> - 尺寸公式和卷积一样；`ceil_mode=False` 向下取整（丢边），`True` 向上取整（保留不完整窗口）。
> - 5×5 + 核 3：floor 得 `[[2]]`，ceil 得 `[[2, 3], [5, 1]]`。

<br>

## 非线性激活：ReLU、Sigmoid 与 inplace（P20）

> [!NOTE]+ 这一节要解决什么
>
> 目的：理解为什么层与层之间要插一个“激活函数”，以及 ReLU / Sigmoid 对数值做了什么。
>
> 位置：仍在“模型”一环。卷积和全连接本质上都是“乘加”（线性运算），激活函数是夹在它们之间的“弯折”；下一节的全连接层就是另一种线性运算。

对应视频 P20。任意多层线性变换复合后仍是一次线性变换：$W_2(W_1 x)= (W_2 W_1)x$。ReLU、Sigmoid 等激活打断这种复合，网络才能拟合更复杂的函数。

- ReLU：$\mathrm{ReLU}(x)=\max(0,x)$，负值置零，正值原样。计算便宜，是卷积网络里最常用的
- Sigmoid：$\sigma(x)=1/(1+e^{-x})$，值域 `(0, 1)`。CIFAR 像素已在 `[0, 1]` 时，Sigmoid 会把两端往 0.5 附近压，图像发灰、对比变弱
- `inplace=True`：在原 tensor 上改，省内存，但会丢掉改前的值，不利于某些需要输入本身的反向实现；默认 `False`

用具体数字看两种激活（输入就是下面小例子里的 `[1, -0.5, -1, 3]`，已实际运行）：

| 输入 $x$ | ReLU | Sigmoid |
| :--- | :--- | :--- |
| 1.0 | 1.0 | 0.7311 |
| -0.5 | 0 | 0.3775 |
| -1.0 | 0 | 0.2689 |
| 3.0 | 3.0 | 0.9526 |

Sigmoid 手算一个：$\sigma(1)=1/(1+e^{-1})=1/(1+0.3679)=0.7311$。像素值 0、0.5、1 经 Sigmoid 后变成 0.5、0.6225、0.7311，原本 0 到 1 的跨度被压到 0.5 到 0.73，这就是“图像发灰”的原因。

**inplace 到底改了什么**（实际运行）：

```python
import torch
from torch import nn

x = torch.tensor([1.0, -0.5, -1.0, 3.0])

y = x.clone()                    # 复制一份，避免改到 x
out = nn.ReLU()(y)               # 默认 inplace=False：返回新张量
print(y)                         # tensor([ 1.0000, -0.5000, -1.0000,  3.0000])  原值还在

y = x.clone()
out = nn.ReLU(inplace=True)(y)   # 直接在 y 的内存上改
print(y)                         # tensor([1., 0., 0., 3.])  原值被覆盖
print(out is y)                  # True：返回的就是 y 本身
```

inplace 带来的两个真实报错：

- 对“需要梯度的叶子张量”（如 `torch.tensor([1., -1.], requires_grad=True)`）直接 inplace：`RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.`
- 先 `s = torch.sigmoid(a)`，再 `nn.ReLU(inplace=True)(s)`，然后反向：`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation ...`。原因是 Sigmoid 的反向需要用到它自己的输出 $\sigma(x)$（$\sigma'=\sigma(1-\sigma)$），而这个值已被 ReLU 原地改掉了。

所以初学阶段保持默认 `inplace=False` 即可；常见的 `Conv2d → ReLU(inplace=True)` 组合之所以安全，是因为卷积的反向不需要它自己的输出。

小例子（ReLU）：

```python
import torch
from torch import nn
from torch.nn import ReLU

x = torch.tensor([[1.0, -0.5],
                  [-1.0, 3.0]])
x = torch.reshape(x, (-1, 1, 2, 2))  # 补成 NCHW
                                     # [2, 2] -> [1, 1, 2, 2]

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()  # 无参数：sum(p.numel() for p in ReLU().parameters()) == 0

    def forward(self, input):
        return self.relu1(input)  # 形状不变：[1, 1, 2, 2] -> [1, 1, 2, 2]

print(Tudui()(x))
# 负值变 0：[[1, 0], [0, 3]]
```

`reshape` 成 `(-1, 1, 2, 2)` 是为了和图像张量一样带上 batch、通道维，`-1` 自动填 batch=1。

其实 ReLU / Sigmoid 是逐元素运算，任意形状都能直接算，`reshape` 只是为了和后面图像数据的习惯一致。

官方 `nn_relu.py`（前向里实际走的是 Sigmoid，便于在 TensorBoard 看图像变化）：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])  # 含小数 -0.5，所以整个张量自动是 float32

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)  # torch.Size([1, 1, 2, 2])

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)  # 逐元素 1/(1+e^-x)，形状不变 [64, 3, 32, 32]
        return output

tudui = Tudui()

writer = SummaryWriter("../logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
```

把 `forward` 改成 `self.relu1(input)` 则负像素被截断，图像上暗部更死。课程网络后半没有显式加 ReLU，结构以卷积+池化+全连接演示尺寸为主；实际分类网络通常在卷积后接激活。

补一句：`ToTensor()` 之后的像素都在 `[0, 1]`，没有负数，所以对原图直接做 ReLU，输出和输入**完全一样**，TensorBoard 里看不出区别；这也是官方脚本改用 Sigmoid 演示的原因。只有经过 `Normalize`（会产生负值）或卷积之后，ReLU 才有可见效果。

> [!TIP]+ 小结
>
> - 没有激活函数，多层线性层叠起来仍等价于一层；激活负责引入非线性。
> - ReLU：负数变 0，正数不变；Sigmoid：压到 `(0, 1)`，$\sigma(0)=0.5$。
> - 激活层没有参数、不改变形状，逐元素计算。
> - `inplace=True` 会覆盖原张量，可能导致反向报错；入门阶段用默认值。

<br>

## 线性层：nn.Linear、flatten 与 in_features 怎么算（P21）

> [!NOTE]+ 这一节要解决什么
>
> 目的：把卷积/池化得到的特征图“压平”成向量，再用全连接层映射到类别数（CIFAR-10 是 10）。关键难点是 `in_features` 填多少。
>
> 位置：模型的最后几层。上一节的激活夹在层之间；这一节之后，下一节把卷积、池化、展平、全连接用 `Sequential` 串成完整模型。

对应视频 P21。全连接层 `nn.Linear(in_features, out_features)` 做 $y = xW^T + b$，要求输入最后一维等于 `in_features`。卷积输出是 `[N, C, H, W]`，必须先展平。

拆开看 $y = xW^T + b$：

- `W` 形状 `[out_features, in_features]`，`b` 形状 `[out_features]`；
- 输入 `x` 形状 `[N, in_features]`，输出 `[N, out_features]`；更高维的输入只看**最后一维**，前面的维度原样保留；
- 参数量 $=\text{in} \times \text{out} + \text{out}$。

手算一个（已运行验证）：`Linear(3, 2)`，令 $W=\begin{bmatrix}1&0&2\\0&1&-1\end{bmatrix}$，$b=[0.5, 0]$，输入 $x=[1,2,3]$：

$$
y_0 = 1\cdot1+0\cdot2+2\cdot3+0.5 = 7.5,\qquad y_1 = 0\cdot1+1\cdot2+(-1)\cdot3+0 = -1
$$

代码输出 `[[7.5, -1.0]]`。每个输出就是“输入向量和 W 的一行做点积再加偏置”，所以叫全连接：每个输出都连着全部输入。

两种展平：

```python
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])  # 形状 [2, 2, 2]，可以把第 0 维看成 batch=2
torch.flatten(t)                 # 从 dim0 到最后 → 长度 8
torch.flatten(t, start_dim=1)    # 保留 batch → [2, 4]
                                 # 结果 [[1, 2, 3, 4], [5, 6, 7, 8]]：每个样本单独压平
```

- `torch.flatten(imgs)` 默认 `start_dim=0`，会把 **batch 也展进去**：`[64, 3, 32, 32] → [196608]`，于是 `Linear(196608, 10)` 输出 `[10]`，相当于把整个 batch 当成一条样本
- 正规写法是 `torch.flatten(imgs, start_dim=1)` → `[64, 3072]`，再 `Linear(3072, 10)` → `[64, 10]`
- `torch.reshape(imgs, (1, 1, 1, -1))` 同样得到长度 196608 的一条向量，课程用来迁就 `Linear(196608, 10)`

`64×3×32×32 = 196608`。最后一批若不足 64，展平长度就不是 196608，会和 `in_features` 对不上，所以该示例 `drop_last=True`。

实测各种展平的形状（`imgs = torch.rand(64, 3, 32, 32)`）：

| 写法 | 结果形状 | 再过 `Linear` |
| :--- | :--- | :--- |
| `torch.flatten(imgs)` | `[196608]` | `Linear(196608, 10)` → `[10]` |
| `torch.reshape(imgs, (1, 1, 1, -1))` | `[1, 1, 1, 196608]` | `Linear(196608, 10)` → `[1, 1, 1, 10]` |
| `torch.flatten(imgs, start_dim=1)` | `[64, 3072]` | `Linear(3072, 10)` → `[64, 10]` |
| `nn.Flatten()(imgs)` | `[64, 3072]` | 同上，可以放进 `Sequential` |

另外，`Linear(196608, 10)` 有 $196608\times10+10=1966090$ 个参数，而 `Linear(3072, 10)` 只有 30730 个；前者把“64 张图拼在一起”当作一个输入，既浪费又没有意义，只是课程用来演示 flatten 的效果。

不加 `drop_last=True` 时，CIFAR10 测试集 10000 张，$10000 = 156\times64 + 16$，最后一批 16 张展平成 $16\times3072=49152$，实测报错：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x49152 and 196608x10)
```

读法：`mat1` 是输入（1 行 49152 列），`mat2` 是 $W^T$（196608 行 10 列），矩阵乘法要求 mat1 的列数 = mat2 的行数，即**输入最后一维 = in_features**。

> [!EXAMPLE]+ in_features 怎么确定：3 种方法
>
> **方法 1：用公式逐层推。** 例如下一节的网络，32 → 卷积保持 32 → 池化 16 → 16 → 8 → 8 → 4，最后通道 64，$64\times4\times4=1024$。
>
> **方法 2：喂一个假输入，直接打印展平后的 shape（最推荐）。** 先只搭到 `Flatten` 为止：
>
> ```python
> import torch
> from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten
>
> feat = Sequential(
>     Conv2d(3, 32, 5, padding=2), MaxPool2d(2),
>     Conv2d(32, 32, 5, padding=2), MaxPool2d(2),
>     Conv2d(32, 64, 5, padding=2), MaxPool2d(2),
>     Flatten(),
> )
> print(feat(torch.ones(1, 3, 32, 32)).shape)   # torch.Size([1, 1024]) -> in_features = 1024
> ```
>
> **方法 3：逐层打印。** 遍历 `Sequential` 的每个子层，见下一节“逐层打印 shape”。
>
> 无论哪种，看的都是**去掉 batch 维之后**的长度。

官方 `nn_linear.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)  # 丢掉最后不满 64 张的那一批

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)  # 196608 = 64*3*32*32；参数量 1966090

    def forward(self, input):
        output = self.linear1(input)  # [196608] -> [10]
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    output = torch.flatten(imgs)  # start_dim=0，连 batch 一起压平
    print(output.shape)  # torch.Size([196608])
    output = tudui(output)
    print(output.shape)  # torch.Size([10])
```

> [!WARNING]+ 常见报错与排查（线性层）
>
> - `mat1 and mat2 shapes cannot be multiplied (AxB and CxD)`：B（输入最后一维）≠ C（`in_features`）。按上面的方法 2 打印展平后的 shape，把 `in_features` 改成 B。
> - 忘了展平，直接把 `[64, 64, 4, 4]` 送进 `Linear(1024, 64)`：Linear 只看最后一维 4，同样报 `mat1 and mat2 ...`。
> - 换了输入尺寸（比如 28×28 的图喂给为 32×32 设计的网络）：展平长度变成 $64\times3\times3=576$，报 `(64x576 and 1024x64)`。

> [!TIP]+ 小结
>
> - `Linear(in, out)`：$y=xW^T+b$，只作用在最后一维；参数量 $\text{in}\times\text{out}+\text{out}$。
> - 卷积输出接 Linear 前要展平，**保留 batch**：`torch.flatten(x, 1)` 或 `nn.Flatten()`。
> - `torch.flatten(x)` 默认连 batch 一起压平，是课程演示用法，不是正规写法。
> - `in_features` 最省事的确定方法：喂一个 `torch.ones(1, 3, 32, 32)` 看展平后的长度。

<br>

## 串联：用 Sequential 搭 CIFAR-10 模型（P22）

> [!NOTE]+ 这一节要解决什么
>
> 目的：把前面学的卷积、池化、展平、全连接按顺序串成一个完整的 CIFAR-10 分类模型，并能自己核对每一层的 shape 和参数量。
>
> 位置：“模型”一环的收尾。模型输出 `[N, 10]` 的分数（logits）之后，下一节用损失函数把它和真实标签比较，算出 loss 并反向传播。

对应视频 P22。后面 CIFAR-10 分类网络的结构为：

`Conv 3→32 (k=5, p=2)` → `MaxPool 2` → `Conv 32→32` → `MaxPool 2` → `Conv 32→64` → `MaxPool 2` → `Flatten` → `Linear 1024→64` → `Linear 64→10`

`kernel=5` 时 `padding=(5-1)/2=2`，空间边长不变。`MaxPool2d(2)` 默认 stride=2，边长减半。

**为什么是 padding=2：由公式反推。** 网络结构图上卷积前后都是 32×32，已知 $n_{\text{in}}=n_{\text{out}}=32$，$f=5$，stride 取 1，代入：

$$
32 = \frac{32 + 2p - 5}{1} + 1 \;\Rightarrow\; 32 = 28 + 2p \;\Rightarrow\; p = 2
$$

如果假设 stride=2，则 $32 = (32+2p-5)/2+1$，得 $p = 17.5$，不是整数也不合理，所以 stride 只能是 1。一般规律：stride=1 时想保持尺寸，需要 $2p = f-1$，即 $p=(f-1)/2$，核为奇数才能整除。也可以写 `padding='same'`（实测 `Conv2d(3, 32, 5, padding='same')` 输出 `[1, 32, 32, 32]`），但只限 stride=1。

如果忘了写 padding（即 padding=0），每次卷积边长减 4：32 → 28 → 池化 14 → 10 → 5 → 1 → 池化时报错：

```text
RuntimeError: Given input size: (64x1x1). Calculated output size: (64x0x0). Output size is too small
```

尺寸核对（输入 `32×32`）：

| 层 | 计算 | 输出 |
| :--- | :--- | :--- |
| Conv2d(3, 32, 5, padding=2) | $(32+4-5)/1+1=32$ | `[N, 32, 32, 32]` |
| MaxPool2d(2) | 32/2 | `[N, 32, 16, 16]` |
| Conv2d(32, 32, 5, padding=2) | 保持 | `[N, 32, 16, 16]` |
| MaxPool2d(2) | | `[N, 32, 8, 8]` |
| Conv2d(32, 64, 5, padding=2) | 保持 | `[N, 64, 8, 8]` |
| MaxPool2d(2) | | `[N, 64, 4, 4]` |
| Flatten | $64×4×4=1024$ | `[N, 1024]` |
| Linear(1024, 64) → Linear(64, 10) | | `[N, 10]` |

**逐层参数量**（实际运行 `sum(p.numel() for p in layer.parameters())` 得到）：

| 序号 | 层 | weight 形状 | 计算 | 参数量 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Conv2d(3, 32, 5) | `[32, 3, 5, 5]` | $32\times3\times25+32$ | 2432 |
| 1 | MaxPool2d(2) | — | 无参数 | 0 |
| 2 | Conv2d(32, 32, 5) | `[32, 32, 5, 5]` | $32\times32\times25+32$ | 25632 |
| 3 | MaxPool2d(2) | — | | 0 |
| 4 | Conv2d(32, 64, 5) | `[64, 32, 5, 5]` | $64\times32\times25+64$ | 51264 |
| 5 | MaxPool2d(2) | — | | 0 |
| 6 | Flatten | — | | 0 |
| 7 | Linear(1024, 64) | `[64, 1024]` | $1024\times64+64$ | 65600 |
| 8 | Linear(64, 10) | `[10, 64]` | $64\times10+10$ | 650 |
| | **合计** | | | **145578** |

可以看到参数主要集中在后两个卷积和第一个全连接层；池化、展平都是 0。

这里的 `Flatten` 是 `nn.Flatten()`，默认 `start_dim=1`，**保留 batch**，与上一节手写 `torch.flatten(imgs)` 不同。用 `torch.ones((64, 3, 32, 32))` 走一遍，确认输出为 `[64, 10]`；对不上就回头查哪一层的 padding / 通道写错。

不使用 Sequential 时每一层都要起名，并在 `forward` 里依次调用，便于中途 `print(x.shape)`：

```python
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import torch

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)  # in=3, out=32, k=5
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)  # in 必须等于上一层的 out=32
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)  # 1024 = 64*4*4
        self.linear2 = Linear(64, 10)  # 10 = CIFAR-10 类别数

    def forward(self, x):
        x = self.conv1(x)  # [N, 3, 32, 32] -> [N, 32, 32, 32]
        x = self.maxpool1(x)  # -> [N, 32, 16, 16]
        x = self.conv2(x)  # -> [N, 32, 16, 16]
        x = self.maxpool2(x)  # -> [N, 32, 8, 8]
        x = self.conv3(x)  # -> [N, 64, 8, 8]
        x = self.maxpool3(x)  # -> [N, 64, 4, 4]
        x = self.flatten(x)  # -> [N, 1024]
        x = self.linear1(x)  # -> [N, 64]
        x = self.linear2(x)  # -> [N, 10]
        return x
```

`nn.Sequential` 按添加顺序执行子模块，`forward` 只需一行。`print(tudui)` 会打印整棵模块树。`writer.add_graph(tudui, input)` 在 TensorBoard 的 GRAPHS 里展开 `Tudui` 可看数据流。

用 Sequential 搭模型的步骤：

1. **画出（或看懂）结构图**：每层的输入输出通道、边长写下来。
2. **由公式反推每层参数**：通道数直接抄，padding 按 $p=(f-1)/2$ 反推，`in_features` 按展平长度算。
3. **按顺序写进 `Sequential(...)`**，注意相邻两层“上一层的 out = 下一层的 in”。
4. **用假数据走一遍**：`torch.ones((64, 3, 32, 32))`，看输出是不是 `[64, 10]`。
5. **对不上就逐层打印 shape**（见下方），找到第一个和预期不符的层。

官方 `nn_seq.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # [N, 3, 32, 32] -> [N, 32, 32, 32]
            MaxPool2d(2),  # -> [N, 32, 16, 16]
            Conv2d(32, 32, 5, padding=2),  # -> [N, 32, 16, 16]
            MaxPool2d(2),  # -> [N, 32, 8, 8]
            Conv2d(32, 64, 5, padding=2),  # -> [N, 64, 8, 8]
            MaxPool2d(2),  # -> [N, 64, 4, 4]
            Flatten(),  # -> [N, 1024]
            Linear(1024, 64),  # -> [N, 64]
            Linear(64, 10)  # -> [N, 10]
        )

    def forward(self, x):
        x = self.model1(x)  # 依次经过上面 9 层
        return x

tudui = Tudui()
print(tudui)  # 打印模块树，见下方输出
input = torch.ones((64, 3, 32, 32))  # 假数据：只为核对 shape，不需要真实图片
output = tudui(input)
print(output.shape)  # torch.Size([64, 10])

writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input)  # 需要一个样例输入来追踪计算图
writer.close()
```

`print(tudui)` 的实际输出（括号里的数字就是 `Sequential` 内的下标，可以用 `tudui.model1[0]` 取出第 0 层）：

```text
Tudui(
  (model1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=1024, out_features=64, bias=True)
    (8): Linear(in_features=64, out_features=10, bias=True)
  )
)
```

**逐层打印 shape 和参数量**：`Sequential` 可以像列表一样遍历，手动让数据一层一层过（接上面的 `tudui`）：

```python
x = torch.ones((64, 3, 32, 32))
for i, layer in enumerate(tudui.model1):           # 按顺序取出每一层
    x = layer(x)                                   # 手动前向一层
    n = sum(p.numel() for p in layer.parameters()) # 这一层的参数个数
    print(i, layer.__class__.__name__, tuple(x.shape), n)
print("total", sum(p.numel() for p in tudui.parameters()))
# 实际输出：
# 0 Conv2d (64, 32, 32, 32) 2432
# 1 MaxPool2d (64, 32, 16, 16) 0
# 2 Conv2d (64, 32, 16, 16) 25632
# 3 MaxPool2d (64, 32, 8, 8) 0
# 4 Conv2d (64, 64, 8, 8) 51264
# 5 MaxPool2d (64, 64, 4, 4) 0
# 6 Flatten (64, 1024) 0
# 7 Linear (64, 64) 65600
# 8 Linear (64, 10) 650
# total 145578
```

等价的非 Sequential 写法是把每一层赋给 `self.conv1`、`self.maxpool1` … 再在 `forward` 里依次调用，便于插入调试打印，但更冗长。

> [!WARNING]+ 常见报错与排查（串联）
>
> - `Output size is too small`：某层输入已经比核还小，多半是 padding 漏写或池化次数太多。逐层打印 shape 找到边长掉到 1 的那一层。
> - `mat1 and mat2 shapes cannot be multiplied (64x576 and 1024x64)`：输入图不是 32×32（这里是 28×28），展平长度变了。要么把输入 resize 到 32，要么把 `Linear` 的 `in_features` 改成实际值。
> - 喂了单张图 `[3, 32, 32]`（没有 batch 维）：卷积能把它当成 unbatched 处理，但到了 `Flatten`（start_dim=1）会把 `[64, 4, 4]` 压成 `[64, 16]`，报 `(64x16 and 1024x64)`。修法：`img.unsqueeze(0)` 或 `reshape(1, 3, 32, 32)`。
> - `Given groups=1, weight of size [64, 32, 5, 5], expected input[...] to have 32 channels ...`：相邻两层通道没对上，检查“上一层 out = 下一层 in”。

> [!TIP]+ 小结
>
> - `Sequential` 按顺序执行子层，`forward` 一行搞定；`print(model)` 看结构，`model.model1[i]` 取单层。
> - padding 由“输入输出边长相等”反推：stride=1 时 $p=(f-1)/2$，k=5 → p=2。
> - 每层 shape 都要能手推，`Flatten` 后的长度就是第一个 `Linear` 的 `in_features`（这里 1024）。
> - 课程 CIFAR-10 模型共 145578 个参数；用 `torch.ones((64, 3, 32, 32))` 核对输出 `[64, 10]`。

<br>

## 损失函数与反向传播：L1 / MSE / CrossEntropy（P23）

> [!NOTE]+ 这一节要解决什么
>
> 目的：用一个数（loss）衡量“模型输出和真实答案差多远”，再通过 `backward()` 算出每个参数该往哪个方向调。
>
> 位置：数据 → 变换 → 打包 → 模型 → **损失** → 优化 → 训练循环。上一节模型已能输出 `[N, 10]`；这一节算出 loss 和梯度 `.grad`；下一节的优化器（optimizer）读 `.grad` 去真正修改权重。

对应视频 P23。损失函数度量预测与标签的差距，越小越好。得到标量 `loss` 后：

1. `loss.backward()` 沿计算图把梯度写进各参数的 `.grad`
2. 优化器再用 `.grad` 改权重

选哪种损失要看任务，并且**输入形状必须符合该损失的文档**。

| 任务 | 常用损失 | 模型输出 | 标签 |
| :--- | :--- | :--- | :--- |
| 回归（预测连续值） | `L1Loss`、`MSELoss` | 与标签同形状的 float | float，形状相同 |
| 多分类（如 CIFAR-10） | `CrossEntropyLoss` | `[N, C]` 的 logits（float） | `[N]` 的类别下标（int64） |

### 回归：L1Loss 与 MSELoss

- `L1Loss`：$\sum |x_i - y_i|$ 或平均。`reduction='mean'`（默认）取平均，`'sum'` 求和，`'none'` 不降维
- `MSELoss`：平方误差 $\mathrm{mean}((x-y)^2)$

官方 `nn_loss.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)  # 模型预测；必须是浮点
targets = torch.tensor([1, 2, 5], dtype=torch.float32)  # 真实值

inputs = torch.reshape(inputs, (1, 1, 1, 3))  # 模仿 [N, C, H, W]；L1/MSE 其实只要求两者同形状
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')  # 求和而不是平均
result = loss(inputs, targets)  # |1-1|+|2-2|+|3-5| = 2

loss_mse = nn.MSELoss()  # 默认 reduction='mean'
result_mse = loss_mse(inputs, targets)  # (0+0+4)/3 = 1.3333

print(result)  # tensor(2.)
print(result_mse)  # tensor(1.3333)


x = torch.tensor([0.1, 0.2, 0.3])  # 3 个类别的 logits
y = torch.tensor([1])  # 真实类别下标 1，dtype 自动是 int64
x = torch.reshape(x, (1, 3))  # [3] -> [1, 3]，即 N=1、C=3
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)  # tensor(1.1019)
```

`inputs=[1,2,3]`，`targets=[1,2,5]`：

- L1、`reduction='sum'`：$|3-5|=2$
- MSE 默认 mean：$((0)^2+(0)^2+(2)^2)/3 = 4/3 \approx 1.333$

三种 `reduction` 实测对照：

| 写法 | 结果 | 手算 |
| :--- | :--- | :--- |
| `L1Loss(reduction='sum')` | `tensor(2.)` | $0+0+2$ |
| `L1Loss()`（mean） | `tensor(0.6667)` | $2/3$ |
| `L1Loss(reduction='none')` | `tensor([[[[0., 0., 2.]]]])` | 逐元素，不降维 |
| `MSELoss()`（mean） | `tensor(1.3333)` | $4/3$ |
| `MSELoss(reduction='sum')` | `tensor(4.)` | $0+0+4$ |

L1 与 MSE 的区别：MSE 把误差平方，误差 2 记作 4、误差 10 记作 100，对大误差惩罚更重；L1 对所有误差一视同仁。整型张量直接算会报错：`L1Loss` 报 `RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long`，`MSELoss` 报 `RuntimeError: "mse_cpu" not implemented for 'Long'`，所以官方代码特意写了 `dtype=torch.float32`。

### 分类：CrossEntropyLoss 手算

交叉熵示例：logits $x=[0.1,0.2,0.3]$，真实类 $y=1$。`CrossEntropyLoss` 内部先 softmax 再取 $-\log p_y$，不必在网络末尾再接 Softmax。

$$
p_i = \frac{e^{x_i}}{\sum_j e^{x_j}},\quad
\mathcal{L} = -\log p_{y}
$$

$e^{0.1}\approx 1.105$，$e^{0.2}\approx 1.221$，$e^{0.3}\approx 1.350$，和 $\approx 3.676$，$p_1\approx 0.332$，$-\log 0.332 \approx 1.10$。真实类对应的 logit 相对越大，损失越小。

把手算拆成 4 步（数值与 `torch.softmax`、`CrossEntropyLoss` 实测一致）：

1. **取指数**：$e^{0.1}=1.1052$，$e^{0.2}=1.2214$，$e^{0.3}=1.3499$。
2. **求和**：$1.1052+1.2214+1.3499=3.6764$。
3. **归一化得到概率（softmax）**：$p=[0.3006, 0.3322, 0.3672]$，三者之和为 1。
4. **取真实类的概率求负对数**：$y=1$，$-\ln 0.3322 = 1.1019$。等价写法 $\mathcal{L} = -x_y + \ln\sum_j e^{x_j} = -0.2 + \ln 3.6764 = 1.1019$。

**batch 里有多个样本时**：默认 `reduction='mean'`，对每个样本分别算再取平均。例如再加一个样本 logits $[2.0, 0.5, -1.0]$、真实类 0：$e^{2}=7.389$，$e^{0.5}=1.6487$，$e^{-1}=0.3679$，和 $9.4057$，$p_0=0.7856$，损失 $-\ln 0.7856=0.2413$。两个样本平均 $(1.1019+0.2413)/2=0.6716$：

```python
import torch
from torch import nn

xb = torch.tensor([[0.1, 0.2, 0.3],      # 样本 0 的 logits，真实类 1
                   [2.0, 0.5, -1.0]])    # 样本 1 的 logits，真实类 0；xb: [N=2, C=3]
yb = torch.tensor([1, 0])                # [N=2]，int64 类别下标
print(nn.CrossEntropyLoss()(xb, yb))                    # tensor(0.6716)  平均
print(nn.CrossEntropyLoss(reduction='none')(xb, yb))    # tensor([1.1019, 0.2413])  每个样本一个
```

第二个样本真实类的 logit（2.0）明显最大，损失就小（0.24）；第一个样本真实类的 logit 并不突出，损失就大（1.10）。另外，一个随机初始化、什么都没学的 10 分类模型，每类概率约 0.1，损失约 $\ln 10 \approx 2.303$；后面实测随机初始化的 CIFAR 模型 loss 约 2.25–2.30，正是这个量级，可以当作“代码没写错”的一个检查点。

### 分类：CrossEntropyLoss 的形状与类型要求

`nn.CrossEntropyLoss` 用于多分类。约定：

- 网络输出 `outputs` 形状 `[N, C]`（**未经过** softmax 的 logits）
- 标签 `targets` 形状 `[N]`，值为类别下标 `0 … C-1`（不是 one-hot）

CIFAR-10 最后一层 `Linear(64, 10)` 正是 10 类 logits。`N=1` 时 `outputs` 为 `[1, 10]`，`targets` 为 `[1]` 的 LongTensor，二者匹配。

不符合要求时的实测结果（logits `x` 为 `[1, 3]`）：

| 传入的 target | 结果 |
| :--- | :--- |
| `torch.tensor([1])`（int64，`[N]`） | 正常，`1.1019` |
| `torch.tensor([1.0])`（float 下标） | `RuntimeError: expected scalar type Long but found Float` |
| `torch.tensor([1], dtype=torch.int32)` | `RuntimeError: expected scalar type Long but found Int` |
| `torch.tensor([3])`（只有 3 类，下标越界） | `IndexError: Target 3 is out of bounds.` |
| input 是 `[2, 3]`、target 是 `[1]` | `ValueError: Expected input batch_size (2) to match target batch_size (1).` |
| input 没 reshape、还是 `[3]`，target `[1]` | `RuntimeError: size mismatch (got input: [3], target: [1])` |

修法：标签用 `.long()` 转 int64；检查标签范围是不是 `0 … C-1`（有的数据集从 1 开始编号）；input 和 target 的第 0 维要相同。DataLoader 从 CIFAR10 / FakeData 读出来的 `targets` 本身就是 `torch.int64` 的 `[N]`，直接用即可。

> [!NOTE]+ 补充：target 也可以是概率
>
> PyTorch 1.10 起，`CrossEntropyLoss` 还接受与 input 同形状的 **float 概率** 作为 target（例如 `[[0., 1., 0.]]`），实测结果同样是 `1.1019`。这用于标签平滑、知识蒸馏等场景。入门阶段按课程约定用整数下标即可；看到 float 型 target 不报错时，要确认它确实是“每行和为 1 的概率”而不是写错的下标。

官方 `nn_loss_network.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)  # 每次 1 张：imgs [1, 3, 32, 32]，targets [1]

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)  # [1, 3, 32, 32] -> [1, 10]
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)  # [1, 10] 的 logits，带 grad_fn
    result_loss = loss(outputs, targets)  # 标量（0 维张量），grad_fn=<NllLossBackward0>
    print("ok")
```

视频里接下来会在循环里加一行 `result_loss.backward()`，然后看梯度。

在 `result_loss.backward()` 处打断点，可在调试器里看 `tudui.model1[0].weight.grad`：未 backward 时为 `None`，backward 后为与 weight 同形状的梯度。此处还没有 `optimizer.step()`，权重数值不变。

> [!EXAMPLE]+ backward 后 grad 在哪里看（不用调试器也能看）
>
> 用随机图代替 CIFAR（避免下载），实际运行：
>
> ```python
> import torch
> from torch import nn
>
> torch.manual_seed(0)
> tudui = Tudui()                              # 上面定义的模型
> imgs = torch.rand(1, 3, 32, 32)              # 一张假图
> targets = torch.tensor([3])                  # 假标签：第 3 类
> loss = nn.CrossEntropyLoss()
>
> result_loss = loss(tudui(imgs), targets)
> print(result_loss)                           # tensor(2.2545, grad_fn=<NllLossBackward0>)
> print(tudui.model1[0].weight.grad)           # None：还没 backward
>
> w0 = tudui.model1[0].weight.detach().clone() # 备份权重，稍后对比
> result_loss.backward()                       # 第 1 步：反向传播，填充 .grad
>
> g = tudui.model1[0].weight.grad
> print(g.shape)                               # torch.Size([32, 3, 5, 5])，与 weight 同形状
> print(torch.equal(w0, tudui.model1[0].weight))  # True：只算了梯度，权重没变
>
> for name, p in tudui.named_parameters():     # 第 2 步：遍历所有参数看梯度
>     print(name, p.grad.shape)                # model1.0.weight [32, 3, 5, 5]、model1.0.bias [32] ...
> ```
>
> 查看位置总结：
>
> - 单层：`tudui.model1[0].weight.grad`、`tudui.model1[0].bias.grad`（下标就是 `print(tudui)` 里括号中的数字）；
> - 全部：`for name, p in tudui.named_parameters(): print(name, p.grad)`；
> - 调试器（PyCharm / VS Code）：在 backward 下一行打断点，展开 `tudui → model1 → _modules → '0' → weight → grad`。
>
> 一个可以手算核对的梯度：交叉熵对 logits 的梯度是 $p - \text{onehot}(y)$。最后一层 `Linear(64, 10)` 的 bias 梯度正好等于它（N=1 时），实测 `tudui.model1[-1].bias.grad` 与 `softmax(outputs) - onehot` 完全一致：第 3 类为 $-0.8951$（负数，说明要把这一类的分数调高），其余 9 类约为 $+0.09 \sim +0.11$。

关于 `backward` 的两个坑（都已实测）：

- **同一个 loss 不能 backward 两次**：第二次报 `RuntimeError: Trying to backward through the graph a second time ...`。每个 batch 都要重新前向、重新算 loss。
- **梯度会累加**：不清零的话，再前向 + backward 一次，`.grad` 变成两次之和（实测正好是 2 倍）。所以训练循环里每步都要先 `optimizer.zero_grad()`，这是下一节优化器的内容。

> [!WARNING]+ 与 Softmax 重复
>
> `CrossEntropyLoss` 内部已含 log-softmax。`forward` 末尾若再接 `Softmax`，等于做了两次，梯度会变差。需要概率时在推理阶段单独 `softmax`。

实测对比：logits `[[2.0, 0.5, -1.0]]`、真实类 0，直接算 loss 为 0.2413；先 softmax 再送进去，loss 变成 0.7017。softmax 后的值都在 0 到 1 之间，再做一次 softmax 后各类概率被拉得很接近，模型再怎么“确信”，loss 也降不下去。另外 `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`，实测 `nn.NLLLoss()(torch.log_softmax(logits, 1), t)` 同样是 0.2413。

> [!TIP]+ 小结
>
> - 回归用 `L1Loss` / `MSELoss`：输入和标签同形状、都是 float；`reduction` 控制 mean / sum / none。
> - 分类用 `CrossEntropyLoss`：输入 `[N, C]` 的 logits（不要先 softmax），标签 `[N]` 的 int64 下标 `0 … C-1`。
> - 交叉熵手算：softmax 得概率，取真实类概率的 $-\ln$；随机初始化的 10 分类模型 loss 约 $\ln 10 \approx 2.3$。
> - `loss.backward()` 后梯度在 `参数.grad`（与参数同形状），backward 前是 `None`；权重要等 `optimizer.step()` 才会改。
> - 同一个图不能 backward 两次；梯度会累加，训练时每步先清零。

<br>

## 优化器：按梯度更新参数（torch.optim，P24）

> [!NOTE]+ 这一节要解决什么
>
> 上一节的损失函数（loss）给出了“错了多少”，`loss.backward()` 给出了“每个参数该往哪边改”（梯度，gradient）。优化器（optimizer）负责最后一步：**真正去改参数**。在整条流程 数据 → 变换 → 打包 → 模型 → 损失 → **优化** → 训练循环 → 保存/验证 中，它处在“损失”和“训练循环”之间。学会这一节，就能写出最小的训练循环；下一节开始换用现成的预训练网络，再往后是保存模型与完整训练脚本。

对应视频 P24。`torch.optim` 按梯度改参数。SGD 示意：$w \leftarrow w - \mathrm{lr}\cdot \partial L/\partial w$。

### 先把公式读懂：lr 是什么

公式里每个符号的含义：

| 符号 | 含义 | 在代码里 |
| :--- | :--- | :--- |
| $w$ | 某个可学习参数（卷积核里的一个数、Linear 的一个权重） | `tudui.parameters()` 里的张量 |
| $\partial L/\partial w$ | 损失对这个参数的梯度：$w$ 增大一点，loss 变大多少 | `w.grad`，由 `loss.backward()` 填进去 |
| $\mathrm{lr}$ | 学习率（learning rate）：沿梯度反方向走**多大一步** | `SGD(..., lr=0.01)` |
| $\leftarrow$ | 原地更新 $w$ | `optimizer.step()` 做的事 |

为什么是“减”：梯度指向 loss **增大最快**的方向，要让 loss 变小就往反方向走。

手算一步：设 $w=1$，loss $L = 2w$，则 $\partial L/\partial w = 2$；取 $\mathrm{lr}=0.1$：

$$
w_{\text{new}} = 1 - 0.1 \times 2 = 0.8
$$

学习率的大小直接决定能不能训好。用一个最简单的问题实际跑过：最小化 $L=(w-3)^2$（最优解 $w=3$），从 $w=0$ 出发，各走 10 步 SGD：

| lr | 10 步后的 $w$ | 现象 |
| :--- | :--- | :--- |
| 0.01 | 0.5488 | 太小：方向对，但走得太慢 |
| 0.1 | 2.6779 | 合适：稳步接近 3 |
| 0.5 | 3.0 | 这个问题里恰好一步到位（特例） |
| 1.1 | -15.5752 | 太大：每步都“跨过头”，越跳越远，发散 |

原因可以手算：梯度是 $2(w-3)$，更新后 $w-3$ 变成更新前的 $(1-2\,\mathrm{lr})$ 倍。lr=0.01 时倍数 0.98，10 步后距离 $3\times 0.98^{10}\approx 2.45$，所以 $w\approx 0.55$；lr=1.1 时倍数 $-1.2$，绝对值大于 1，距离越来越大。真实网络里 loss 变成 `nan` 或忽高忽低，常见原因之一就是 lr 太大。

标准三步，顺序固定：

1. `optimizer.zero_grad()`：清空上一轮累积的梯度
2. `loss.backward()`：计算当前 batch 的梯度
3. `optimizer.step()`：按学习率更新参数

每一步具体动了什么：

| 步骤 | 读什么 | 写什么 | 不做会怎样 |
| :--- | :--- | :--- | :--- |
| `zero_grad()` | — | 把每个参数的 `.grad` 清掉（PyTorch 2.x 默认 `set_to_none=True`，直接设为 `None`） | 新梯度叠加到旧梯度上 |
| `backward()` | 计算图（前向时记录的 `grad_fn`） | 每个 `requires_grad=True` 的参数的 `.grad` | `.grad` 为空，`step()` 什么都不改 |
| `step()` | 每个参数的 `.grad` 和 `lr` | 参数本身：`w -= lr * w.grad` | 梯度算了但参数不变，模型学不到东西 |

PyTorch 默认把梯度**累加**到 `.grad`。一个 batch 用完必须清零，否则下一个 batch 的梯度会叠加上去。视频里强调：先 zero_grad，再算 loss / backward，再 step（loss 也可以在 zero_grad 前算，但不能在 backward 之后还不 zero 就进入下一轮）。

> [!EXAMPLE]+ 实际跑一下：忘记 zero_grad 会怎样
>
> ```python
> import torch
>
> w = torch.tensor([1.0], requires_grad=True)  # 一个可学习参数，初值 1
> x = torch.tensor([2.0])                      # 输入，固定为 2
> for i in range(3):
>     loss = (w * x).sum()   # L = w*x，所以 dL/dw = x = 2，每轮都一样
>     loss.backward()        # 没有清零，直接 backward
>     print(i, w.grad)
> # 0 tensor([2.])
> # 1 tensor([4.])   <- 2 + 2，被累加了
> # 2 tensor([6.])   <- 2 + 2 + 2
>
> w.grad = None                                # 手动清掉，重新来
> opt = torch.optim.SGD([w], lr=0.1)
> for i in range(3):
>     opt.zero_grad()        # 1. 清零
>     loss = (w * x).sum()
>     loss.backward()        # 2. 算梯度：每轮都是 2
>     opt.step()             # 3. w = w - 0.1 * 2
>     print(i, w.grad, w.data)
> # 0 tensor([2.]) tensor([0.8000])
> # 1 tensor([2.]) tensor([0.6000])
> # 2 tensor([2.]) tensor([0.4000])
> ```
>
> 不清零时，第 3 轮的梯度是 6 而不是 2：相当于把“前几个 batch 的梯度”也算进来了，步子越来越大、方向也被旧数据带偏，loss 曲线会乱跳甚至发散。代码不会报错，所以这是最难发现的 bug 之一。
>
> 累加这个默认行为也有正当用途：显存不够时，可以连续 `backward()` 几个小 batch 再 `step()` 一次（梯度累积，gradient accumulation），效果近似一个大 batch。

两个循环职责不同：

| 循环 | 含义 |
| :--- | :--- |
| `for data in dataloader` | 处理**一个 batch** |
| `for epoch in range(E)` | 把**整份数据扫 E 遍** |

只跑内层一轮，loss 往往几乎不动；需要外层 epoch。课程示例 `batch_size=1` 扫测试集 20 轮，仅用于看 `running_loss` 下降，速度很慢，看几个 epoch 即可停。

用数字感受两个循环：CIFAR-10 测试集 10000 张、`batch_size=1` 时，内层循环一轮要跑 10000 次（每次更新一次参数），外层 20 个 epoch 就是 20 万次更新。若改成 `batch_size=64`，内层一轮只跑 $\lceil 10000/64 \rceil = 157$ 次。

写一个最小训练循环的 5 步（下面官方代码就是按这个顺序写的）：

1. 准备 `DataLoader`：决定每次喂多少张图（batch_size）。
2. 实例化网络 `tudui = Tudui()`。
3. 选损失函数 `nn.CrossEntropyLoss()`（分类任务）。
4. 建优化器：`torch.optim.SGD(tudui.parameters(), lr=0.01)`，告诉它“要改哪些参数、步长多大”。
5. 两层循环：外层 epoch，内层 batch；内层里依次 前向 → 算 loss → zero_grad → backward → step。

官方 `nn_optim.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),  # 测试集 10000 张，每张 -> [3, 32, 32]
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)  # 每次只取 1 张：imgs [1, 3, 32, 32]

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),   # [1, 3, 32, 32] -> [1, 32, 32, 32]
            MaxPool2d(2),                  # -> [1, 32, 16, 16]
            Conv2d(32, 32, 5, padding=2),  # -> [1, 32, 16, 16]
            MaxPool2d(2),                  # -> [1, 32, 8, 8]
            Conv2d(32, 64, 5, padding=2),  # -> [1, 64, 8, 8]
            MaxPool2d(2),                  # -> [1, 64, 4, 4]
            Flatten(),                     # -> [1, 1024]，64*4*4=1024
            Linear(1024, 64),              # -> [1, 64]
            Linear(64, 10)                 # -> [1, 10]，10 个类别的得分
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()  # 损失函数对象（注意名字叫 loss，下面的结果叫 result_loss）
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)  # 把网络全部可学习参数交给 SGD，步长 0.01
for epoch in range(20):  # 外层：整份数据扫 20 遍
    running_loss = 0.0  # 每个 epoch 开始时把累计 loss 归零
    for data in dataloader:  # 内层：每次一个 batch
        imgs, targets = data  # imgs [1, 3, 32, 32]，targets [1]
        outputs = tudui(imgs)  # 前向：[1, 10]
        result_loss = loss(outputs, targets)  # 标量张量，带 grad_fn
        optim.zero_grad()  # 第 1 步：清空旧梯度
        result_loss.backward()  # 第 2 步：反向传播，填好每个参数的 .grad
        optim.step()  # 第 3 步：w = w - 0.01 * w.grad
        running_loss = running_loss + result_loss  # 累计本 epoch 的 loss（建议改成 += result_loss.item()）
    print(running_loss)  # 每个 epoch 打印一次，应逐渐变小
```

`running_loss` 是该 epoch 所有 batch 的 loss 之和。此处写的是 `running_loss + result_loss`（张量相加），打印会带 `grad_fn`，计算图也会被挂在 `running_loss` 上，轮数多时浪费内存。正式代码应使用 `running_loss += result_loss.item()`。`tudui.parameters()` 把 `requires_grad=True` 的参数交给优化器。学习率 `1e-2` 即 $1\times 10^{-2}=0.01$。常用还有 `Adam`；课程用 SGD 是为了步骤最直白。文件里 `from torch.optim.lr_scheduler import StepLR` 未使用，可忽略。

补充几点：

- `tudui.parameters()` 是一个生成器，里面是 Tudui 全部 10 个参数张量（5 个层，每层 weight + bias），共 145578 个数。可用 `sum(p.numel() for p in tudui.parameters())` 验证。
- 换优化器只改一行：`torch.optim.Adam(tudui.parameters(), lr=1e-3)`。Adam 会为每个参数自适应调整步长，常用 lr 比 SGD 小（`1e-3` 左右）；循环里的三步写法完全不变。
- `StepLR` 是学习率调度器（scheduler），作用是每隔若干 epoch 把 lr 乘一个系数，例如 `StepLR(optim, step_size=5, gamma=0.1)` 表示每 5 个 epoch lr 乘以 0.1；用法是在每个 epoch 结尾调用一次 `scheduler.step()`。
- 优化器自己也有状态：`optim.state_dict()` 里有 `param_groups`（`lr`、`momentum`、`weight_decay` 等超参数）和 `state`（Adam 的动量等）。断点续训时要连它一起保存。

> [!WARNING]+ 常见报错与排查
>
> - **loss 不降、不报错**：检查是不是漏了 `zero_grad()`、`step()`，或者优化器建在了**另一个**模型实例的参数上（例如先 `optim = SGD(tudui.parameters())`，后来又 `tudui = Tudui()` 重新实例化）。
> - **loss 变成 `nan` 或越来越大**：lr 太大，先除以 10 试试。
> - `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`：对一个没有计算图的张量调用了 `backward()`。常见原因是前向写在了 `with torch.no_grad():` 里，或者 loss 被 `.item()` / `.detach()` 过。
> - `RuntimeError: Trying to backward through the graph a second time`：同一次前向的 loss 调用了两次 `backward()`。每个 batch 应该重新前向一次再 backward。

> [!TIP]+ 小结
>
> - SGD 更新：$w \leftarrow w - \mathrm{lr}\cdot \partial L/\partial w$；lr 太小学得慢，太大会发散。
> - 每个 batch 固定三步：`zero_grad()` → `backward()` → `step()`；漏掉 `zero_grad` 梯度会累加（2 → 4 → 6），不报错但训练会乱。
> - 内层循环 = 一个 batch 更新一次；外层循环 = 整份数据扫一遍。
> - 累计 loss 用 `.item()` 取数值，别把带计算图的张量一直加下去。

<br>

## 预训练模型：拿现成的 VGG16 改分类头（torchvision.models，P25）

> [!NOTE]+ 这一节要解决什么
>
> 前面的 Tudui 是自己从零搭的小网络。实际工作中更常见的是拿别人设计好、甚至在大数据集上训练好的网络（预训练模型，pretrained model），只改最后的输出层去适配自己的任务，这叫迁移学习（transfer learning）。它替换的是流程里“模型”这一环；损失、优化、训练循环的写法都不变。下一节讲怎么把这种模型存下来、再读回来。

对应视频 P25。`torchvision.models` 提供 VGG、ResNet 等现成结构。ImageNet 预训练权重在 1000 类上训练过，迁到 CIFAR-10 时要改最后的分类头。

几个名词先对齐：

| 名词 | 含义 |
| :--- | :--- |
| 结构（architecture） | 有哪些层、每层多少通道，由 Python 代码定义 |
| 权重（weights） | 每层里具体的数值，训练得到，存在 `.pth` 文件里 |
| 随机初始化 | 只有结构，权重是随机数，相当于没学过 |
| 预训练 | 结构 + 在 ImageNet（约 128 万张图、1000 类）上训练好的权重 |
| 分类头（head） | 网络最后输出类别得分的那一层，输出维度 = 类别数 |

视频中的 API：

```python
vgg16_false = torchvision.models.vgg16(pretrained=False)  # 仅结构，随机初始化
vgg16_true = torchvision.models.vgg16(pretrained=True)    # 结构 + ImageNet 权重
```

> [!WARNING]+ 新版 API
>
> 较新的 torchvision 用 `weights` 替代 `pretrained`，例如 `vgg16(weights=None)`、`vgg16(weights='DEFAULT')`。权重默认缓存于用户目录下的 `.cache/torch/hub/checkpoints`，可用 `torch.hub.set_dir(...)` 改路径。下载失败时可把 `.pth` 手动放到该目录。

补充：在当前测试的 torchvision 里 `pretrained=False` 仍能运行，只是会打印 `UserWarning` 提示改用 `weights`；更早的写法在未来版本可能被移除。VGG16 的 ImageNet 权重约 528 MB，第一次下载较慢。只想看结构、算 shape 时，用 `weights=None` 即可，不会下载任何东西，下面的实验都是这样做的。

`print(vgg16_true)` 能看到两段：`features`（卷积）与 `classifier`（若干 `Linear` + Dropout）。原 `classifier[6]` 是 `Linear(4096, 1000)`。

实际打印 `vgg16(weights=None)` 的结构，按三段理解：

```text
VGG(
  (features): Sequential(...)          # 13 个卷积 + ReLU + 5 个 MaxPool，[N,3,224,224] -> [N,512,7,7]
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))   # 不管输入多大，都输出 7x7：把输入划成 7x7 个区域分别取平均（输入小于 7x7 时，相邻区域会重复用到同一个像素）
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)   # 25088 = 512*7*7
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)    # 分类头：1000 类
  )
)
```

`forward` 的顺序是 `features` → `avgpool` → `flatten` → `classifier`。实测 shape：输入 `[2, 3, 224, 224]`，`features` 输出 `[2, 512, 7, 7]`，最终输出 `[2, 1000]`。整个 VGG16 共 138,357,544 个参数，其中光 `classifier[0]` 就有 $25088\times 4096+4096 \approx 1.03$ 亿个。

`classifier[6]` 这种写法能用，是因为 `nn.Sequential` 支持按下标取子层，下标就是打印出来括号里的数字 `(0)`…`(6)`。

迁到 10 类的两种改法：

1. **追加**：`vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))`，1000 维后再映到 10 维。若写成 `vgg16_true.add_module(...)` 会加在顶层，与 `classifier` 并列，前向**不会**自动经过这一层
2. **替换**：`vgg16_false.classifier[6] = nn.Linear(4096, 10)`，直接改最后一层输出维

两种改法逐步对比（均为 `weights=None` 实测）：

| | 追加 `classifier.add_module` | 替换 `classifier[6] = ...` |
| :--- | :--- | :--- |
| 改完后 classifier 最后两层 | `(6) Linear(4096,1000)` → `(add_linear) Linear(1000,10)` | `(6) Linear(4096,10)` |
| 输出 shape | `[2, 10]` | `[2, 10]` |
| 新增参数 | $1000\times 10+10=10010$ | 删掉 $4096\times 1000+1000=4097000$，新增 $4096\times 10+10=40970$ |
| 特点 | 保留了预训练的 1000 类头，多一层 | 结构更干净，迁移学习里更常用 |

> [!EXAMPLE]+ 实测：add_module 加错位置
>
> ```python
> import torch, torchvision
> from torch import nn
>
> x = torch.rand(2, 3, 224, 224)                       # 2 张假图
>
> v = torchvision.models.vgg16(weights=None)           # 不下载权重，只要结构
> v.classifier.add_module('add_linear', nn.Linear(1000, 10))  # 加在 classifier 内部
> print(v(x).shape)                                    # torch.Size([2, 10])  正确
>
> v2 = torchvision.models.vgg16(weights=None)
> v2.add_module('add_linear', nn.Linear(1000, 10))     # 加在顶层
> print([n for n, _ in v2.named_children()])           # ['features', 'avgpool', 'classifier', 'add_linear']
> print(v2(x).shape)                                   # torch.Size([2, 1000])  新层没被用到！
>
> v3 = torchvision.models.vgg16(weights=None)
> v3.classifier[6] = nn.Linear(4096, 10)               # 替换最后一层
> print(v3(x).shape)                                   # torch.Size([2, 10])
> ```
>
> 原因：VGG 的 `forward` 是写死的（只调用 `features`、`avgpool`、`classifier`），顶层多挂一个模块只是“注册”了它，`forward` 里没人调用它。而 `classifier` 是 `nn.Sequential`，它的 `forward` 会按顺序跑完内部**所有**子层，所以加在里面才生效。

> [!NOTE]+ 输入尺寸
>
> ImageNet 预训练的 VGG 按约 `224×224` 的输入训练。较新 torchvision 在 `features` 后有 `AdaptiveAvgPool2d((7, 7))`，因此 `32×32` 也能算出 `512×7×7` 并接到 `classifier`，不一定报错；但感受野与预训练分布仍按大图设计，直接拿 CIFAR 原图微调效果通常较差。要贴近预训练设定需 `Resize(224)`。`model.py` 里的小网络才是为 32×32 设计的。

实测佐证：输入 `[2, 3, 32, 32]` 时，`features` 经过 5 次 2 倍下采样，输出只剩 `[2, 512, 1, 1]`（$32/2^5=1$），`AdaptiveAvgPool2d((7,7))` 把这 1 个像素复制成 7×7，最终输出仍是 `[2, 1000]`。能跑，但每个通道只剩一个数，空间信息几乎没有了。

官方 `model_pretrained.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)  # 新写法 weights=None：随机权重
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 新写法 weights='DEFAULT'：会下载约 528MB 权重

print(vgg16_true)  # 看 features / avgpool / classifier 三段

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),  # CIFAR-10 只有 10 类
                                          download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))  # 改法 1：在 classifier 末尾追加 1000 -> 10
print(vgg16_true)  # classifier 末尾多出 (add_linear)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 改法 2：把下标 6 的 Linear(4096,1000) 换成 Linear(4096,10)
print(vgg16_false)  # (6) 变为 out_features=10
```

`print(vgg16_true)` 可看到 `features`（卷积）与 `classifier`（全连接）两段。`add_module` 加在 `vgg16_true` 顶层会成为与 `classifier` 并列的模块；加在 `vgg16_true.classifier` 上才会接在 1000 维输出之后。

拿预训练模型做迁移学习的完整 4 步（课程只演示到第 2 步）：

1. 加载结构与权重：`vgg16(weights='DEFAULT')`。
2. 改分类头为 10 类：`model.classifier[6] = nn.Linear(4096, 10)`。新层是随机初始化的，必须训练。
3. （可选）冻结前面的层：`for p in model.features.parameters(): p.requires_grad = False`，只训练分类器，速度快、不易过拟合；数据多时也可以全部一起微调（fine-tune）。
4. 预处理要和预训练一致：`Resize(224)`、`ToTensor()`，再加上 ImageNet 的 `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`。之后的训练循环与 Tudui 完全相同。

> [!WARNING]+ 常见报错与排查
>
> - `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x4096 and 1000x10)`：新层的 `in_features` 写错。替换 `classifier[6]` 时输入应是 4096（上一层的输出），只有在 1000 维之后追加时才写 1000。
> - 输出仍是 `[N, 1000]`：`add_module` 加到了顶层，见上面的实测。
> - 下载卡住 / `URLError`：手动下载 `vgg16-397923af.pth` 放到 `~/.cache/torch/hub/checkpoints/`，或者先用 `weights=None` 调通代码。

> [!TIP]+ 小结
>
> - `torchvision.models.vgg16(weights=None)` 只要结构，`weights='DEFAULT'` 才带 ImageNet 权重（旧写法 `pretrained=True/False`）。
> - VGG16 = `features`（卷积）+ `avgpool` + `classifier`（3 个 Linear），分类头是 `classifier[6]`：`Linear(4096, 1000)`。
> - 改 10 类：替换 `classifier[6] = nn.Linear(4096, 10)`，或在 `classifier` 内部 `add_module` 追加 `Linear(1000, 10)`。
> - 加在顶层的模块不会被 `forward` 调用；改完一定用假输入看一下输出 shape。

<br>

## 模型持久化：保存与加载（torch.save / torch.load，P26）

> [!NOTE]+ 这一节要解决什么
>
> 训练好的参数只存在内存里，程序一关就没了。持久化（serialization）就是把模型写到 `.pth` 文件、以后再读回来，对应流程最后的“保存/验证”。上一节的预训练权重本质上也是别人用这种方式存下来的文件。下一节的完整训练脚本会在每个 epoch 结尾存一次盘。

对应视频 P26。两种保存方式：

| 方式 | 调用 | 内容 | 加载 |
| :--- | :--- | :--- | :--- |
| 1 | `torch.save(model, path)` | 结构 + 参数（pickle 整个模块） | `torch.load(path)` |
| 2（官方推荐） | `torch.save(model.state_dict(), path)` | 仅 `OrderedDict` 参数 | 先建同结构模型，再 `load_state_dict` |

### state_dict 里到底有什么

`state_dict()` 返回一个有序字典 `OrderedDict`：**键**是参数的名字（按模块层级用 `.` 连接），**值**是参数张量。只有带参数（或 buffer，如 BN 的 running mean）的层才会出现，ReLU、MaxPool、Dropout 没有参数，不会出现。

实测：

```python
import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)   # 属性名 conv1 决定了键名前缀
    def forward(self, x):
        return self.conv1(x)

sd = Tudui().state_dict()
print(type(sd))                              # <class 'collections.OrderedDict'>
print(list(sd.keys()))                       # ['conv1.weight', 'conv1.bias']
print([tuple(v.shape) for v in sd.values()]) # [(64, 3, 3, 3), (64,)]
# weight: [out_channels, in_channels, kH, kW] = 64*3*3*3 = 1728 个数；bias: 64 个数
```

VGG16 的 `state_dict` 有 32 个键：13 个卷积 + 3 个 Linear，各有 weight 和 bias（$16\times 2=32$），开头是 `features.0.weight`，结尾是 `classifier.6.weight`、`classifier.6.bias`。键名里的数字就是 `Sequential` 里的下标，`features.1`（ReLU）不存在，因为它没有参数。存成文件约 553 MB（1.38 亿个 float32，每个 4 字节）。

方式 2 不绑定类的定义位置，换设备、改代码结构时更好控。`torch.load("vgg16_method2.pth")` 得到的是字典，直接 `print` 不是网络，必须先 `vgg16 = torchvision.models.vgg16(pretrained=False)` 再 `load_state_dict`。

> [!NOTE]+ 易错：方式 2 并不明显更省空间
>
> 两种方式存的参数一样多，体积差别只是方式 1 多存的那点结构信息（类名、模块路径等）。实测上面的单层 Tudui：方式 1 为 9530 字节，方式 2 为 8570 字节；对 VGG16 这种 500 多 MB 的模型，差别几乎可以忽略。选方式 2 的真正理由是**不依赖类定义、更安全、更灵活**，而不是省空间。

方式 2 加载的 3 步：

1. 用代码构造一个**结构相同**的空模型：`vgg16 = torchvision.models.vgg16(weights=None)`。
2. 把文件读成字典：`sd = torch.load("vgg16_method2.pth")`。
3. 把字典灌进模型：`vgg16.load_state_dict(sd)`，成功时返回 `<All keys matched successfully>`。

方式 1 的陷阱：pickle 会记录**类的定义位置**。在 `model_save.py` 里定义 `Tudui` 并保存整模型，到另一个文件 `torch.load` 时，若当前进程找不到 `Tudui`，会报 `AttributeError: Can't get attribute 'Tudui'`。Jupyter 同一 kernel 里刚定义过该类时有时能加载成功，换成 PyCharm 分文件则会报错。处理：

- `from model_save import Tudui`（加载前把类引进当前模块）
- 或把类定义复制到加载文件

方式 2 无此问题，因为结构由加载代码自己构造。

为什么会这样：pickle 保存一个对象时，并不保存类的代码，只保存“这个对象属于 `__main__.Tudui` 类”这样一个**名字引用**，加上它的属性（各层和参数）。加载时 pickle 按名字去找类，找不到就报错。实测：在 `model_save.py` 里存、在不含 `Tudui` 定义的 `model_load.py` 里读（已加 `weights_only=False`），报错为：

```text
AttributeError: Can't get attribute 'Tudui' on <module '__main__' from '...\model_load.py'>
```

报错里的 `__main__` 说明 pickle 在找“当前运行的主脚本里的 Tudui”。这也意味着：即便能加载，改了类的代码（例如改了 `forward`）后，加载出来的模型会用**新的** `forward`。

> [!WARNING]+ `torch.load` 默认值
>
> PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`，加载方式 1 的完整模块可能直接失败。加载课程里的 `.pth` 时可写 `torch.load(path, map_location=..., weights_only=False)`（仅限信任的文件）。方式 2 的 `state_dict` 更符合该默认。

实测（PyTorch 2.6）：用默认参数 `torch.load("m1.pth")` 加载方式 1 的文件，报 `_pickle.UnpicklingError: Weights only load failed...`；加上 `weights_only=False` 后成功得到 `Tudui` 对象。方式 2 的文件用默认参数直接加载成功。“仅限信任的文件”是因为 pickle 反序列化可以执行任意代码，来路不明的 `.pth` 用 `weights_only=False` 打开有安全风险。

官方 `model_save.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)  # 随机初始化的 VGG16（新写法 weights=None）
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")  # 整个对象 pickle 下来

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")  # 只存 OrderedDict{参数名: 张量}

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")  # 文件里记的是“model_save.Tudui / __main__.Tudui”这个名字，不是类的代码
```

官方 `model_load.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from model_save import *  # 把 Tudui 类引进来；注意这也会把 model_save.py 顶层代码重新执行一遍（再存一次文件）
# 方式1-》保存方式1，加载模型
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")  # PyTorch 2.6+ 需加 weights_only=False
# print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)  # 第 1 步：先造一个同结构的空模型
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  # 第 2、3 步：读字典，灌进模型
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱1
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load('tudui_method1.pth')  # 能成功全靠上面的 from model_save import *；同样需 weights_only=False
print(model)
```

`from model_save import *` 正是为了把 `Tudui` 引进当前模块，方式 1 才能反序列化成功。与上一节相同，新版本 `torch.load` 加载完整模块时可能需要 `weights_only=False`。

GPU 上保存、CPU 上加载时指定：

```python
model = torch.load("tudui_29_gpu.pth", map_location=torch.device("cpu"))  # 把存档里 cuda 上的张量映射到 cpu；方式 1 文件在 2.6+ 还需 weights_only=False
```

不加 `map_location` 时，在没有 GPU 的机器上加载 GPU 存档会报 `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`。

断点续训时推荐把模型、优化器、epoch 一起存成一个字典（方式 2 的扩展）：

```python
# 保存
torch.save({
    "epoch": i,                                  # 当前训练到第几轮
    "model": tudui.state_dict(),                 # 模型参数
    "optimizer": optimizer.state_dict(),         # 优化器状态（lr、动量等）
}, "ckpt.pth")

# 加载
ckpt = torch.load("ckpt.pth", map_location="cpu")
tudui = Tudui()                                  # 先造结构
tudui.load_state_dict(ckpt["model"])
optimizer = torch.optim.SGD(tudui.parameters(), lr=1e-2)
optimizer.load_state_dict(ckpt["optimizer"])
start_epoch = ckpt["epoch"] + 1                  # 从下一轮接着训
```

> [!WARNING]+ 常见报错与排查
>
> - `AttributeError: Can't get attribute 'Tudui' on <module '__main__' ...>`：方式 1 加载时找不到类，先 `from model_save import Tudui`，或改用方式 2。
> - `_pickle.UnpicklingError: Weights only load failed`：PyTorch 2.6+ 默认 `weights_only=True`，加载方式 1 文件时传 `weights_only=False`（仅限可信文件）。
> - `RuntimeError: Error(s) in loading state_dict ... size mismatch for conv1.weight: copying a param with shape torch.Size([64, 3, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 3, 3, 3])`：构造的模型结构和存档不一致（实测把 64 通道改成 32 通道即触发）。
> - `Missing key(s) in state_dict: "fc.weight" ... Unexpected key(s) in state_dict: "0.weight" ...`：层的**名字**对不上（例如存的时候用 `Sequential` 下标、加载的模型用属性名 `fc`）。确认结构一致；只想加载能对上的部分可用 `load_state_dict(sd, strict=False)`，它会返回缺失和多余的键列表。
> - 方式 2 用 `model = torch.load("xxx_method2.pth")` 后调用 `model(x)` 报 `'collections.OrderedDict' object is not callable`：读出来的是字典，不是网络，要按上面 3 步灌进模型。

> [!TIP]+ 小结
>
> - 方式 1 `torch.save(model)` 存整个对象，加载时必须能找到类定义；方式 2 `torch.save(model.state_dict())` 只存参数字典，官方推荐。
> - `state_dict` 是 `OrderedDict{"层名.weight": 张量, ...}`，没有参数的层（ReLU、池化）不出现。
> - 方式 2 加载 3 步：造同结构模型 → `torch.load` 读字典 → `load_state_dict`。
> - PyTorch 2.6+ 加载方式 1 文件要 `weights_only=False`；跨设备加载加 `map_location`。

<br>

## 完整训练套路：从数据到存盘（train.py，P27–P29）

> [!NOTE]+ 这一节要解决什么
>
> 把前面所有零件串成一个能真正跑完的训练脚本：Dataset/DataLoader（数据与打包）+ Tudui（模型）+ CrossEntropyLoss（损失）+ SGD（优化）+ 训练/测试循环 + `torch.save`（保存）+ TensorBoard（记录）。这是整条流程的总装，之后无论换什么数据集、什么网络，骨架都是这一套；后续的 GPU 训练只是在这个脚本上把模型和数据 `.cuda()` / `.to(device)`。

对应视频 P27–P29。完整套路：准备数据 → 建网 → 损失与优化器 → 按 epoch 训练 → 测试集评估 → 存模型 → TensorBoard 记录。

网络单独放在 `model.py`，训练脚本 `from model import *`。若 import 失败，把该目录标成 Sources Root（PyCharm）。`if __name__ == '__main__'` 里用全 1 张量检查输出是否为 `[64, 10]`，避免训练跑起来才发现 `Linear` 输入维写错。

`if __name__ == '__main__'` 的作用：直接运行 `python model.py` 时 `__name__` 等于 `'__main__'`，下面的测试代码会执行；被 `train.py` 用 `from model import *` 导入时 `__name__` 等于 `'model'`，测试代码**不会**执行，不会在训练开始时多打印一行。

`nn.Conv2d(3, 32, 5, 1, 2)` 五个位置参数依次是 `in_channels, out_channels, kernel_size, stride, padding`，与前面 `Conv2d(3, 32, 5, padding=2)` 等价。

为什么 `padding=2` 能让尺寸不变，代入卷积输出尺寸公式算一遍（$H_{in}=32$，$k=5$，$s=1$，$p=2$）：

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1 = \left\lfloor \frac{32 + 4 - 5}{1} \right\rfloor + 1 = 32
$$

所以每个卷积不改变宽高，只有 `MaxPool2d(2)` 让宽高减半：$32 \to 16 \to 8 \to 4$，最后 `Flatten` 得到 $64\times 4\times 4 = 1024$，这就是 `nn.Linear(64*4*4, 64)` 的来历。

官方 `model.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from torch import nn

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),   # [64, 3, 32, 32] -> [64, 32, 32, 32]，参数 3*32*25+32 = 2432
            nn.MaxPool2d(2),             # -> [64, 32, 16, 16]
            nn.Conv2d(32, 32, 5, 1, 2),  # -> [64, 32, 16, 16]，参数 32*32*25+32 = 25632
            nn.MaxPool2d(2),             # -> [64, 32, 8, 8]
            nn.Conv2d(32, 64, 5, 1, 2),  # -> [64, 64, 8, 8]，参数 32*64*25+64 = 51264
            nn.MaxPool2d(2),             # -> [64, 64, 4, 4]
            nn.Flatten(),                # -> [64, 1024]
            nn.Linear(64*4*4, 64),       # -> [64, 64]，参数 1024*64+64 = 65600
            nn.Linear(64, 10)            # -> [64, 10]，参数 64*10+10 = 650
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':  # 只有直接运行 model.py 时才执行，被 import 时跳过
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))  # 假装一个 batch：64 张 3 通道 32x32 的图
    output = tudui(input)
    print(output.shape)  # torch.Size([64, 10])，实测通过
```

参数量合计：$2432+25632+51264+65600+650=145578$，与 `sum(p.numel() for p in tudui.parameters())` 实测一致。对比 VGG16 的 1.38 亿，这个网络小了近 1000 倍，CPU 上也能训。

### 训练脚本要用到的几个小工具

`loss.item()` 把只含一个元素的张量转成 Python `float`。直接 `print(loss)` 会带 `grad_fn`，写入 TensorBoard 也需要纯数值。

```python
tensor = torch.tensor([3.14])
value = tensor.item()  # 3.14，类型 float
```

为什么一定要 `.item()`，实测对比：

```python
l = loss_fn(outputs, targets)
print(l)              # tensor(2.2627, grad_fn=<NllLossBackward0>)  <- 张量，还挂着计算图
print(l.item())       # 2.2626771926879883                           <- Python float
print(type(l.item())) # <class 'float'>

r = 0.0
r = r + l             # 不加 .item()：r 变成 tensor(2.2627, grad_fn=<AddBackward0>)
                      # 每加一次都把这个 batch 的计算图挂在 r 上，图无法释放，内存只涨不降
```

三条理由：① 打印干净；② 断开计算图，避免累加时内存泄漏；③ 在 GPU 上时 `.item()` 会把数值拷回 CPU，`writer.add_scalar`、`format` 等 Python 代码才能直接用。注意 `.item()` 只能用于**只有一个元素**的张量，对 `[64, 10]` 的 outputs 调用会报 `RuntimeError: a Tensor with 640 elements cannot be converted to Scalar`。

`with torch.no_grad():` 临时关闭梯度。测试集不需要 backward，不建图能省显存、加快速度。`with` 结束自动恢复。训练阶段的 `loss.backward()` 若包在 `no_grad` 内，则没有梯度，优化器无法更新。

实测 `no_grad` 内外的区别：

```python
with torch.no_grad():
    y = tudui(torch.ones(1, 3, 32, 32))
print(y.requires_grad, y.grad_fn)   # False None              <- 没建计算图
y2 = tudui(torch.ones(1, 3, 32, 32))
print(y2.requires_grad, y2.grad_fn) # True <AddmmBackward0 ...> <- 正常建图

with torch.no_grad():
    l = loss_fn(tudui(imgs), targets)
    l.backward()
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

分类正确数：`outputs` 为 `[N, 10]`，`argmax(1)` 表示在 **dim=1（类别维）** 取最大下标，得到 `[N]` 的预测编号。`argmax(0)` 会在 batch 维上取，结果不是「每张图的类别」。

```python
import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])
print(outputs.argmax(1))          # tensor([1, 1])
targets = torch.tensor([0, 1])
print(outputs.argmax(1) == targets)          # tensor([False, True])
print((outputs.argmax(1) == targets).sum())  # tensor(1)
```

> [!EXAMPLE]+ 手算准确率：4 张图、3 个类别
>
> 设一个 batch 有 4 张图，网络输出每类得分如下，真实标签 `targets = [0, 2, 0, 1]`：
>
> | 图 | 类 0 | 类 1 | 类 2 | argmax(1)（每行最大的列） | 真实 | 对吗 |
> | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
> | 0 | **2.0** | 0.5 | 0.1 | 0 | 0 | 对 |
> | 1 | 0.2 | 0.1 | **3.0** | 2 | 2 | 对 |
> | 2 | 1.0 | **1.5** | 0.3 | 1 | 0 | 错 |
> | 3 | 0.0 | **0.2** | 0.1 | 1 | 1 | 对 |
>
> 计算 4 步：
>
> 1. `outputs.argmax(1)` → `tensor([0, 2, 1, 1])`：每张图选得分最高的类。
> 2. `== targets` → `tensor([True, True, False, True])`：逐个比较。
> 3. `.sum()` → `tensor(3)`：True 当 1、False 当 0 相加，即答对 3 张。
> 4. 除以图片总数：$3/4 = 0.75$，准确率 75%。
>
> 对照 `outputs.argmax(0)` → `tensor([0, 2, 1])`：这是“每一列（每个类）哪张图得分最高”，长度是类别数 3 而不是图片数 4，含义完全不对。得分不需要先过 softmax，因为 softmax 不改变大小顺序，argmax 结果相同。

`total_accuracy / test_data_size` 得到准确率。`total_accuracy` 是张量，打印时会显示 `tensor(...)`，可再 `.item()`。

实测：`tensor(20) / 200` 得到 `tensor(0.1000)`，再 `.item()` 得到 `0.10000000149011612`（float32 精度带来的尾数，正常现象）。整数张量做 `/` 会自动得到浮点结果，不会像某些语言那样整除成 0。

`model.train()` / `model.eval()`：

| | `train()` | `eval()` |
| :--- | :--- | :--- |
| Dropout | 按概率置零 | 关闭，全部神经元保留 |
| BatchNorm | 用当前 batch 统计，并更新 running mean/var | 使用训练期累计的统计量 |

本课程的 Tudui 没有这两类层，不切换也能跑；有 BN / Dropout 时必须在训练循环开头 `train()`、测试循环开头 `eval()`。

实测 Dropout（p=0.5）对全 1 输入的表现：`train()` 模式输出 `[2., 0., 2., 0., 2., 2., 0., 2.]`（随机一半置 0，剩下的乘 $1/(1-p)=2$ 保持期望不变）；`eval()` 模式输出全是 `1.`。要点：

- `train()` / `eval()` 只是把模型及所有子模块的 `self.training` 标志设为 True / False，**不会**开启或关闭梯度计算；关梯度要靠 `torch.no_grad()`。两者职责不同，测试时通常一起用。
- 反过来，`torch.no_grad()` 也不会让 Dropout 停下来。
- 上一节 VGG16 的 `classifier` 里就有两个 Dropout，用它训练时这两个开关必须写。

| | 管什么 | 不管什么 |
| :--- | :--- | :--- |
| `model.eval()` | Dropout、BN 的行为 | 是否建计算图 |
| `torch.no_grad()` | 是否建计算图（省内存、提速） | Dropout、BN 的行为 |

### 每个 epoch 里做什么：循环结构

每个 epoch 内部顺序：

1. `tudui.train()`，遍历 `train_dataloader`：前向 → 损失 → `zero_grad` → `backward` → `step`；每 100 步打印并 `add_scalar("train_loss", ...)`
2. `tudui.eval()` + `no_grad`，遍历测试集，累加 loss 与正确个数
3. `torch.save(tudui, "tudui_{}.pth".format(i))` 按轮次存盘（方式 1）

`total_train_step` 跨 epoch 累加，作为 TensorBoard 横轴；`total_test_step` 每个 epoch 加一。`total_test_loss` 累加的是各 batch 的 **mean** loss（`CrossEntropyLoss` 默认 `reduction='mean'`），不是「全测试集逐样本平均」；最后一批较小时代入横轴时只作趋势，不宜当精确指标。准确率用 `total_accuracy / test_data_size`，分母是整集长度，与 `drop_last=False` 的官方 `train.py` 一致。

用 CIFAR-10 的数字算一下这些计数器：训练集 50000 张、`batch_size=64`，每个 epoch 有 $\lceil 50000/64 \rceil = 782$ 个 batch（最后一个只有 $50000-781\times 64=16$ 张），所以每个 epoch 大约打印 7 次 loss，10 个 epoch 后 `total_train_step = 7820`。测试集 10000 张有 157 个 batch，`total_test_loss` 是 157 个 batch 平均 loss 之和；若每个 batch 约 2.3，总和约 361。想要“平均每张图的 loss”，可以写 `total_test_loss / len(test_dataloader)`（近似）或在累加时用 `loss.item() * imgs.size(0)` 再除以 `test_data_size`（精确）。

### 官方 train.py 拆成 10 步

先看整体，再逐行读代码：

| 步 | 做什么 | 对应代码 | 为什么 |
| :--- | :--- | :--- | :--- |
| 1 | 准备数据集 | `CIFAR10(train=True/False)` | 训练集用来学，测试集用来检查有没有学会 |
| 2 | 看数据集大小 | `len(train_data)` | 算准确率要用测试集总数做分母 |
| 3 | 打包成 batch | `DataLoader(..., batch_size=64)` | 一次喂 64 张，比逐张快，梯度也更稳 |
| 4 | 创建网络 | `tudui = Tudui()` | 参数随机初始化 |
| 5 | 损失函数 | `nn.CrossEntropyLoss()` | 衡量输出得分和真实类别的差距 |
| 6 | 优化器 | `SGD(tudui.parameters(), lr=1e-2)` | 负责按梯度改参数 |
| 7 | 计数器与 TensorBoard | `total_train_step`、`SummaryWriter` | 画曲线需要横轴和写日志的对象 |
| 8 | 训练阶段（每个 epoch） | `train()` + 前向/loss/三步 | 真正更新参数 |
| 9 | 测试阶段（每个 epoch） | `eval()` + `no_grad` + 累加 loss/正确数 | 不更新参数，只评估 |
| 10 | 记录与保存 | `add_scalar`、`torch.save`、`writer.close()` | 留下曲线和每轮的模型文件 |

官方 `train.py`（依赖同目录 `model.py`）：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆

import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *  # 导入 Tudui；model.py 里也 import 了 torch，所以这里能直接用 torch
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),  # 【步骤 1】训练集 50000 张
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),  # 测试集 10000 张
                                         download=True)

# length 长度
train_data_size = len(train_data)  # 【步骤 2】50000
test_data_size = len(test_data)  # 10000，后面算准确率的分母
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)  # 【步骤 3】每次 imgs [64, 3, 32, 32]、targets [64]；训练集一般建议加 shuffle=True
test_dataloader = DataLoader(test_data, batch_size=64)  # 测试集不需要打乱

# 创建网络模型
tudui = Tudui()  # 【步骤 4】

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 【步骤 5】输入 [64, 10] 的得分 + [64] 的类别编号，输出一个标量（batch 平均）

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2  # 【步骤 6】单独写成变量，方便调参
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0  # 【步骤 7】跨 epoch 累加，每个 batch +1，做 train_loss 曲线的横轴
# 记录测试的次数
total_test_step = 0  # 每个 epoch +1，做 test 曲线的横轴
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")  # 日志目录，之后 tensorboard --logdir=../logs_train 查看

for i in range(epoch):  # i = 0, 1, ..., 9
    print("-------第 {} 轮训练开始-------".format(i+1))  # 显示时从 1 开始数

    # 训练步骤开始
    tudui.train()  # 【步骤 8】切到训练模式（本网络无 Dropout/BN，写上是好习惯）
    for data in train_dataloader:
        imgs, targets = data  # 8.1 取一个 batch：imgs [64, 3, 32, 32]，targets [64]
        outputs = tudui(imgs)  # 8.2 前向：[64, 10]
        loss = loss_fn(outputs, targets)  # 8.3 算损失：标量张量

        # 优化器优化模型
        optimizer.zero_grad()  # 8.4 清空旧梯度
        loss.backward()  # 8.5 反向传播求梯度
        optimizer.step()  # 8.6 更新参数

        total_train_step = total_train_step + 1  # 8.7 计数
        if total_train_step % 100 == 0:  # 每 100 个 batch 汇报一次，避免刷屏
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))  # .item() 取 float
            writer.add_scalar("train_loss", loss.item(), total_train_step)  # (曲线名, y 值, x 值)

    # 测试步骤开始
    tudui.eval()  # 【步骤 9】切到评估模式
    total_test_loss = 0  # 每个 epoch 重新累计
    total_accuracy = 0  # 本 epoch 测试集上答对的总张数
    with torch.no_grad():  # 不建计算图：省内存、提速，也保证不会误更新
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)  # [64, 10]（最后一批 [16, 10]）
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()  # 累加每个 batch 的平均 loss
            accuracy = (outputs.argmax(1) == targets).sum()  # 本 batch 答对几张，tensor(整数)
            total_accuracy = total_accuracy + accuracy  # 累加后仍是张量

    print("整体测试集上的Loss: {}".format(total_test_loss))  # 【步骤 10】记录
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))  # 打印形如 tensor(0.1000)，可加 .item()
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)  # 横轴是 epoch 序号
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))  # 方式 1 保存，生成 tudui_0.pth ... tudui_9.pth；推荐改成 tudui.state_dict()
    print("模型已保存")

writer.close()  # 把缓冲区的日志写盘并关闭
```

`train-cpu.py` 把 `Tudui` 写在同一文件里，不依赖 `from model import *`，其余逻辑相同。10 个 epoch、CIFAR-10 全量训练在 CPU 上会较慢；接上 GPU 版后主要加速的是卷积与矩阵乘，DataLoader 读盘仍在 CPU。

> [!NOTE]+ 字符串格式化
>
> `"训练数据集的长度为：{}".format(train_data_size)` 把 `{}` 替换为参数。训练脚本里轮次、步数、loss 均用此写法打印。

等价的 f-string 写法更短：`print(f"训练次数：{total_train_step}, Loss: {loss.item():.4f}")`，其中 `:.4f` 表示保留 4 位小数。

### 不下载数据，用 FakeData 把训练循环跑通

想先确认脚本逻辑没问题，可以把 CIFAR10 换成 `torchvision.datasets.FakeData`：它按需随机生成指定尺寸的图片和标签，接口和 CIFAR10 一样返回 `(PIL 图, int 标签)`，经过 `ToTensor` 后同样是 `[3, 32, 32]`。下面这段已实际运行（CPU，几秒钟）：

```python
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
torch.manual_seed(0)                                  # 固定随机种子，结果可复现

# Tudui 与 model.py 相同，此处省略定义
tf = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.FakeData(size=640, image_size=(3, 32, 32), num_classes=10,
                                           transform=tf, random_offset=0)     # 640 张假训练图
test_data = torchvision.datasets.FakeData(size=200, image_size=(3, 32, 32), num_classes=10,
                                          transform=tf, random_offset=1000)  # 200 张假测试图，偏移避免与训练集重复
img, t = train_data[0]
print(img.shape, type(t))                             # torch.Size([3, 32, 32]) <class 'int'>

train_dl = DataLoader(train_data, batch_size=64, shuffle=True)   # 640/64 = 10 个 batch
test_dl = DataLoader(test_data, batch_size=64)                   # ceil(200/64) = 4 个 batch

tudui = Tudui()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tudui.parameters(), lr=1e-2)

step = 0
for ep in range(3):
    tudui.train()
    for imgs, targets in train_dl:
        loss = loss_fn(tudui(imgs), targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        step += 1
        if step % 5 == 0:
            print("step", step, "loss", round(loss.item(), 4))
    tudui.eval(); total_test_loss = 0; total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_dl:
            outputs = tudui(imgs)
            total_test_loss += loss_fn(outputs, targets).item()
            total_accuracy += (outputs.argmax(1) == targets).sum()
    print("ep", ep, "test_loss", round(total_test_loss, 4),
          "acc", (total_accuracy / len(test_data)).item())
```

实际输出：

```text
torch.Size([3, 32, 32]) <class 'int'>
step 5 loss 2.3001
step 10 loss 2.3186
ep 0 test_loss 9.1976 acc 0.10000000149011612
step 15 loss 2.2997
step 20 loss 2.293
ep 1 test_loss 9.1958 acc 0.11500000208616257
step 25 loss 2.3015
step 30 loss 2.2963
ep 2 test_loss 9.1942 acc 0.09000000357627869
```

怎么读这段输出：

- loss 一直在 2.30 附近：10 类分类“完全瞎猜”时的交叉熵是 $-\ln(1/10)=\ln 10\approx 2.3026$，说明网络还处在随机猜测水平。
- test_loss 约 9.19：4 个测试 batch 的平均 loss 之和，$4\times 2.30\approx 9.2$，正好印证前面说的“`total_test_loss` 是各 batch mean 之和”。
- 准确率在 10% 上下：FakeData 的标签是随机的，图片和标签之间没有规律可学，10% 就是瞎猜的水平。这段代码的意义只是**验证流程能跑通、shape 都对**；换成真的 CIFAR-10，几个 epoch 后准确率会明显上升。

> [!WARNING]+ 常见报错与排查
>
> - `FileNotFoundError` / 下载失败：`root="../data"` 是相对**运行目录**的路径，不是相对脚本文件。在 PyCharm 和命令行里运行目录可能不同，可改成绝对路径。
> - `ModuleNotFoundError: No module named 'model'`：`train.py` 和 `model.py` 不在同一目录，或运行目录不对；PyCharm 里把目录标为 Sources Root。
> - `RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x4096 and 1024x64)`：输入图片不是 32×32（例如做了 `Resize`），Flatten 后的长度变了。先用 `model.py` 里的全 1 张量测试。
> - `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`：`targets` 的 shape 是 `[64, 1]` 而不是 `[64]`，用 `targets.squeeze(1)` 或 `targets.view(-1)` 修正。
> - `IndexError: Target 10 is out of bounds.`：标签取值必须在 `0 ~ 类别数-1`，10 类时只能是 0–9；检查数据标签或输出层的类别数。
> - 准确率一直约 10%、loss 一直约 2.30：检查 lr、是否漏了 `optimizer.step()`；若用的是 FakeData 则属正常。

> [!TIP]+ 小结
>
> - 训练脚本 10 步：数据集 → 长度 → DataLoader → 网络 → 损失 → 优化器 → 计数器/TensorBoard → 训练阶段 → 测试阶段 → 记录与保存。
> - 训练阶段 `train()` + 三步更新；测试阶段 `eval()` + `torch.no_grad()`，只累加 loss 和正确数，不更新参数。
> - 准确率 = `(outputs.argmax(1) == targets).sum()` 累加后除以测试集总数；`argmax(1)` 在类别维取最大。
> - 凡是要打印、累加、写 TensorBoard 的 loss，都先 `.item()`。
> - 10 类随机猜测时 loss 约 $\ln 10\approx 2.30$、准确率约 10%，可作为判断“有没有在学”的基准线。

<br>

## 用 GPU 训练：.cuda() 与 .to(device)（P30–P31）

> [!NOTE]+ 这一节要解决什么
>
> 上一节的 `train.py` 已经能在 CPU 上把整条流程（数据 → 变换 → 打包 → 模型 → 损失 → 优化 → 训练循环 → 保存）跑通，但 CIFAR-10 跑 10 个 epoch 很慢。这一节只做一件事：**把计算搬到 GPU 上**，训练逻辑一行不改，只是在几个固定位置加上“搬家”代码。下一节“验证”会反过来处理：GPU 上训练出的权重，怎么在（可能没有 GPU 的）机器上加载并预测一张图。

对应视频 P30–P31。网络、损失、输入、标签必须在同一设备，否则 `Expected all tensors to be on the same device`。优化器不必单独 `.cuda()`，它持有的是参数引用，参数搬家后优化器仍指向它们。

先弄清两个概念：

- **设备（device）**：张量的数据实际存放、计算发生的地方。`cpu` 是内存 + CPU；`cuda` / `cuda:0` 是第 0 块 NVIDIA 显卡的显存。每个张量都有 `.device` 属性，可以 `print(x.device)` 查看。
- **为什么必须同设备**：一次运算（如 `Conv2d` 里的卷积）要同时读输入和权重。一个在内存、一个在显存，PyTorch 不会偷偷帮忙拷贝，而是直接报错。

需要搬的对象：

- 模型：`tudui.cuda()` / `tudui.to(device)`
- 损失函数：带了张量（如 `nn.CrossEntropyLoss(weight=...)` 的类别权重 `weight`）时必须搬，不带时搬不搬都行；课程统一搬一次

> [!WARNING]+ 易错：loss 里的 weight 是 buffer，不是可学习参数
>
> **缓冲（buffer）** 按定义就是**不参与学习**的张量（不会被优化器更新），例如 `nn.CrossEntropyLoss(weight=...)` 里的类别权重 `weight` 就注册成 buffer。实测：`nn.CrossEntropyLoss(weight=torch.ones(10))` 的 `named_buffers()` 只有 `['weight']`，`parameters()` 为空；不带 `weight` 的 `nn.CrossEntropyLoss()` 连 buffer 都没有。所以 **loss 里带了张量（如 `weight`）时必须搬，否则会和 GPU 上的输出撞设备；不带时搬不搬都行**，课程统一写上搬一次，不吃亏。

- **每个 batch** 的 `imgs`、`targets`（DataLoader 默认在 CPU 上取出）

把“搬不搬”整理成一张表：

| 对象 | 要不要搬 | 搬几次 | 原因 |
| :--- | :--- | :--- | :--- |
| 模型 `tudui` | 要 | 训练开始前 1 次 | 权重（parameters）和 buffer 要和输入同设备 |
| 损失函数 `loss_fn` | 带张量（如 `weight`）时必须；否则可选 | 1 次 | 内部张量要和 `outputs` 同设备 |
| `imgs`、`targets` | 要 | **每个 batch 都搬**（训练和测试循环里都要写） | DataLoader 每次吐出的都是 CPU 上的新张量 |
| 优化器 `optimizer` | 不要 | — | 它只保存参数的引用，参数搬了它自然跟着 |
| `loss.item()`、`accuracy` 等结果 | 不要 | — | `.item()` 已经变成 Python 数字，没有设备之分 |

> [!EXAMPLE]+ 为什么优化器不用搬：实测
>
> `model.to(...)` 不是“新建一套参数”，而是**就地替换每个参数里的数据**，参数对象本身还是同一个。在 CPU 上用 `model.to(torch.float64)`（同样走 `.to` 的逻辑）验证：
>
> ```python
> opt = torch.optim.SGD(m.parameters(), lr=0.01)
> p0 = next(m.parameters())
> m.to(torch.float64)                                    # 原地改参数数据
> print(next(m.parameters()) is p0)                      # True：还是同一个参数对象
> print(opt.param_groups[0]["params"][0] is p0)          # True：优化器手里的引用仍然有效
> ```
>
> 官方文档的建议是：**先把模型 `.to(device)`，再创建优化器**。像 Adam 这种带内部状态（动量等）的优化器，如果已经 `step()` 过再搬模型，状态张量可能还留在原设备上。课程代码的顺序（先 `tudui.to(device)`，后 `optimizer = ...`）正好符合。

### 方式 1：.cuda()（P30）

方式 1：`.cuda()`。仓库里用 `torch.cuda.is_available()` 包一层，无 GPU 时仍可在 CPU 跑。

改成 GPU 版只需 3 步：

1. **模型**：创建后 `tudui = tudui.cuda()`。
2. **损失函数**：创建后 `loss_fn = loss_fn.cuda()`。
3. **数据**：训练循环**和**测试循环里，每次取出 `imgs, targets` 后都 `.cuda()` 一次。漏掉测试循环是最常见的错误：训练正常，一到测试就报设备不一致。

官方 `train_gpu_1.py` 相对 CPU 版的差异：

```python
tudui = Tudui()
if torch.cuda.is_available():  # 有可用 NVIDIA GPU 且装的是 CUDA 版 PyTorch 才为 True
    tudui = tudui.cuda()  # 第 1 步：模型搬到 GPU（默认 cuda:0）

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()  # 第 2 步：损失函数搬到 GPU

# 训练 / 测试循环内：
imgs, targets = data  # DataLoader 吐出的两个张量都在 CPU 上
if torch.cuda.is_available():
    imgs = imgs.cuda()  # 第 3 步：[64, 3, 32, 32] 搬到 GPU，必须赋值回 imgs
    targets = targets.cuda()  # [64] 的标签也要搬，CrossEntropyLoss 要求它和 outputs 同设备
```

其余与 `train.py` 相同（网络定义内联在该文件中）。

> [!NOTE]+ 本机没有 GPU 时，这段代码会怎样
>
> 在只有 CPU 的机器上（本笔记的代码都是这样跑通的），`torch.cuda.is_available()` 返回 `False`，所有 `if` 分支都跳过，代码退化成 CPU 版，照样能跑。`is_available()` 为 `False` 的常见原因：
>
> 1. 机器没有 NVIDIA 显卡（Apple、AMD、核显）；
> 2. 装的是 CPU 版 PyTorch：`print(torch.__version__)` 显示 `2.6.0+cpu` 这种带 `+cpu` 后缀；
> 3. 显卡驱动太旧，与 PyTorch 自带的 CUDA 版本不匹配。
>
> 有 NVIDIA 显卡时，可以在终端运行 `nvidia-smi` 看显卡型号、驱动版本、显存占用。没有显卡可以用 Google Colab（视频 P30 演示过）：菜单“修改 → 笔记本设置 → 硬件加速器”选 GPU。

### 方式 2：.to(device)（P31，更常用）

方式 2（更常用）：先定义 `device`，再 `.to(device)`。换 CPU / 多卡时只改一处。`torch.device("cuda")` 等价于当前默认 GPU（一般是 `cuda:0`）；指定卡用 `"cuda:1"`。无 GPU 的写法：

同样拆成 3 步：

1. **定义设备**：文件开头写一次 `device = ...`，以后所有“搬家”都引用这个变量。
2. **模型、损失**：各 `.to(device)` 一次。
3. **数据**：训练、测试循环里每个 batch `.to(device)`。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 第 1 步：有 GPU 用 cuda，否则 cpu；本机输出 device(type='cpu')
# 或视频中的写法（假定一定有 GPU）：
# device = torch.device("cuda")

tudui = tudui.to(device)  # 第 2 步：Module 的 .to 原地生效，返回的就是 tudui 自己
loss_fn = loss_fn.to(device)
imgs = imgs.to(device)  # 第 3 步：Tensor 的 .to 返回新张量，不赋值等于白搬
targets = targets.to(device)
```

`.cuda()` 只能去 NVIDIA GPU；`.to(device)` 还能去 `cpu`、`mps`（Apple）等，因此更通用。仓库 `train_gpu_2.py` 写死 `"cuda"`，没有 GPU 时会在 `.to(device)` 处报错，需改成上面的三元表达式。

补充两点：

- `torch.device("cuda")` 这一行本身在无 GPU 机器上**不报错**，它只是一个“地址标签”；真正去连 GPU 的是第一次 `.to(device)`。CPU 版 PyTorch 下报的是 `AssertionError: Torch not compiled with CUDA enabled`，有 CUDA 版但无驱动时报 `RuntimeError: Found no NVIDIA driver on your system`。
- 三元表达式 `"cuda" if 条件 else "cpu"` 是 Python 语法：条件为真取前者，否则取后者。也可以直接传字符串：`tudui.to("cuda")`、`imgs.to("cpu")`。

两种方式对比：

| | `.cuda()` | `.to(device)` |
| :--- | :--- | :--- |
| 能去哪里 | 只能 NVIDIA GPU（`.cuda(1)` 指定第 1 号卡） | `cpu`、`cuda`、`cuda:1`、`mps` 等任意设备 |
| 无 GPU 时 | 每处都要包 `if torch.cuda.is_available()` | 只在定义 `device` 时判断一次 |
| 改设备 | 改很多处 | 改 `device` 那一行 |
| 额外能力 | 无 | 还能改数据类型，如 `x.to(torch.float16)` |
| 回到 CPU | `.cpu()` | `.to("cpu")` |

官方 `train_gpu_2.py`：

阅读时只需盯住 6 处标了“← 设备”的行，其余与 `train.py` 完全一样。

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

# 定义训练的设备
device = torch.device("cuda")  # ← 设备 ①：写死 cuda；无 GPU 机器请改成 torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)  # 50000
test_data_size = len(test_data)  # 10000
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)  # 50000/64 -> 782 个 batch（最后一个只有 16 张）
test_dataloader = DataLoader(test_data, batch_size=64)  # 10000/64 -> 157 个 batch

# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # [N, 3, 32, 32] -> [N, 32, 32, 32]
            nn.MaxPool2d(2),  # -> [N, 32, 16, 16]
            nn.Conv2d(32, 32, 5, 1, 2),  # -> [N, 32, 16, 16]
            nn.MaxPool2d(2),  # -> [N, 32, 8, 8]
            nn.Conv2d(32, 64, 5, 1, 2),  # -> [N, 64, 8, 8]
            nn.MaxPool2d(2),  # -> [N, 64, 4, 4]
            nn.Flatten(),  # -> [N, 1024]
            nn.Linear(64*4*4, 64),  # -> [N, 64]
            nn.Linear(64, 10)  # -> [N, 10]
        )

    def forward(self, x):
        x = self.model(x)
        return x
tudui = Tudui()
tudui = tudui.to(device)  # ← 设备 ②：模型

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)  # ← 设备 ③：损失函数
# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # 优化器不搬；在模型搬完之后创建

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)  # ← 设备 ④：训练 batch 的图片
        targets = targets.to(device)  # ← 设备 ⑤：训练 batch 的标签
        outputs = tudui(imgs)  # 在 GPU 上前向：[64, 3, 32, 32] -> [64, 10]
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))  # .item() 把 GPU 上的标量拷回 CPU 变成 float
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)  # ← 设备 ⑥：测试循环同样要搬，最容易漏
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()  # 这是 GPU 上的 0 维张量
            total_accuracy = total_accuracy + accuracy  # 累加后 total_accuracy 也变成 GPU 张量

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))  # 打印出来形如 tensor(0.5432, device='cuda:0')
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))  # 整模型保存，里面的权重张量带着 cuda 设备信息
    print("模型已保存")

writer.close()
```

> [!NOTE]+ 赋值
>
> 对 Tensor，必须写 `imgs = imgs.to(device)`，因为 `to` 可能返回新对象。对 `nn.Module`，`model.to(device)` 是原地移动，有无赋值通常都能工作；课程代码仍写成赋值，两种设备迁移风格保持一致。

本机实测（CPU）：

```python
m = Tudui()
print(m.to("cpu") is m)                  # True：Module.to 返回自己，原地修改
t = torch.rand(2)
print(t.to("cpu") is t)                  # True：已经在目标设备上，直接返回原张量
print(t.to(torch.float64) is t)          # False：真的要变（换设备/换类型）时，返回一个新张量
```

这就是“可能返回新对象”里“可能”二字的含义：已经在目标设备上时什么都不做；需要搬时生成一个新张量，**变量 `imgs` 指向的仍是 CPU 上那个张量**。所以只写 `imgs.to(device)` 而不赋值，后面用的仍是 CPU 上的旧 `imgs`，于是报设备不一致。

> [!EXAMPLE]+ 计时对比：GPU 到底快多少（视频 P30 的做法）
>
> 视频里用 `time.time()` 记录开始时间，每 100 步打印一次耗时，对比 CPU 和 GPU。下面是一个独立的小基准：同一个 Tudui，固定一个 `[64, 3, 32, 32]` 的 batch，跑 50 步“前向 + 反向 + 更新”。
>
> ```python
> import time
> import torch
> from torch import nn
>
> def make_model():
>     return nn.Sequential(
>         nn.Conv2d(3, 32, 5, 1, 2), nn.MaxPool2d(2),
>         nn.Conv2d(32, 32, 5, 1, 2), nn.MaxPool2d(2),
>         nn.Conv2d(32, 64, 5, 1, 2), nn.MaxPool2d(2),
>         nn.Flatten(), nn.Linear(1024, 64), nn.Linear(64, 10))
>
> def bench(device, steps=50):
>     model = make_model().to(device)                    # 模型搬到目标设备
>     loss_fn = nn.CrossEntropyLoss().to(device)
>     opt = torch.optim.SGD(model.parameters(), lr=1e-2)
>     imgs = torch.rand(64, 3, 32, 32, device=device)    # 直接在目标设备上造一个 batch
>     targets = torch.randint(0, 10, (64,), device=device)
>     for _ in range(5):                                 # 预热：前几步含初始化开销，不计时
>         loss = loss_fn(model(imgs), targets); opt.zero_grad(); loss.backward(); opt.step()
>     if device.type == "cuda":
>         torch.cuda.synchronize()                       # GPU 是异步的，计时前先等它把活干完
>     start = time.time()
>     for _ in range(steps):
>         loss = loss_fn(model(imgs), targets); opt.zero_grad(); loss.backward(); opt.step()
>     if device.type == "cuda":
>         torch.cuda.synchronize()                       # 计时结束前再等一次
>     return time.time() - start
>
> print("cpu  50 steps: %.3f s" % bench(torch.device("cpu")))   # 本机：cpu  50 steps: 1.399 s
> if torch.cuda.is_available():                                  # 有 GPU 才测 GPU
>     print("cuda 50 steps: %.3f s" % bench(torch.device("cuda")))
> else:
>     print("no GPU, skip cuda benchmark")                       # 本机走这一支
> ```
>
> 要点：
>
> 1. **GPU 计算是异步的**：Python 发出命令就往下走，不等 GPU 算完。不加 `torch.cuda.synchronize()` 量出来的只是“发命令”的时间，会偏小很多。
> 2. **要预热**：第一次用 GPU 有初始化、选算法的开销，前几步不计入。
> 3. 小模型、小 batch 时 GPU 优势不明显，甚至可能更慢（拷贝开销占比大）；模型越大、batch 越大，差距越明显。
> 4. 全量训练时 DataLoader 读图、`ToTensor` 仍在 CPU 上做，如果 GPU 很快但总耗时没降多少，瓶颈可能在数据加载，可以试 `num_workers`。

> [!WARNING]+ 常见报错与排查（设备相关）
>
> | 报错信息（节选） | 原因 | 修法 |
> | :--- | :--- | :--- |
> | `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` | 模型在 GPU、输入在 CPU（或反过来）；最常见是测试循环漏了 `.to(device)`，或写了 `imgs.to(device)` 却没赋值 | 检查模型、loss、`imgs`、`targets` 四样是否都搬了，并且是 `x = x.to(device)` |
> | `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same` | 同上，卷积层报错时常见这个措辞（`FloatTensor` 是 CPU，`cuda.FloatTensor` 是 GPU） | 同上 |
> | `AssertionError: Torch not compiled with CUDA enabled` | 装的是 CPU 版 PyTorch，却调用了 `.cuda()` / `.to("cuda")` | 用 `torch.cuda.is_available()` 判断；或重装 CUDA 版 PyTorch |
> | `torch.OutOfMemoryError: CUDA out of memory`（较早版本显示为 `RuntimeError: CUDA out of memory`；该异常类是 `RuntimeError` 的子类） | 显存不够：batch 太大，或测试时没加 `torch.no_grad()` 导致保存了计算图 | 减小 `batch_size`；测试套上 `no_grad()`；`nvidia-smi` 看是否有别的程序占显存 |
> | `TypeError: can't convert cuda:0 device type tensor to numpy` | 对 GPU 张量直接 `.numpy()` | 先 `.cpu()` 再 `.numpy()`；只要一个数用 `.item()` |
>
> 本机没有 GPU，第一条用 `meta` 设备（一种只记形状、不存数据的“假设备”）模拟，报错是同一类：`nn.Linear(3, 2)(torch.rand(1, 3, device="meta"))` 得到 `RuntimeError: Tensor on device cpu is not on the expected device meta!`。

> [!TIP]+ 小结
>
> - 要搬的只有四样：**模型、损失函数、每个 batch 的 imgs 和 targets**；优化器不搬，但最好在模型搬完后创建。
> - 推荐 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` + `.to(device)`，一份代码 CPU/GPU 通吃。
> - `Module.to` 原地生效；`Tensor.to` 返回新张量，**必须赋值**。
> - 测试循环和训练循环一样要搬数据，漏掉就报 `Expected all tensors to be on the same device`。
> - GPU 计时要 `torch.cuda.synchronize()` 并预热，否则数字不可信。

<br>

## 验证：用训练好的模型预测一张图（P32）

> [!NOTE]+ 这一节要解决什么
>
> 训练循环保存了 `tudui_{i}.pth`，这一节把它真正**用起来**：拿一张模型从没见过的图（比如网上下载的飞机图），走一遍“读图 → 预处理 → 加载模型 → 推理 → 下标转类名”，看模型认为它是哪一类。这是整条流程（数据 → 变换 → 打包 → 模型 → 损失 → 优化 → 训练循环 → 保存/验证）的最后一环。核心原则只有一条：**预测时对图片做的处理，必须和训练时一模一样**。上一节的 GPU 知识在这里用于“在哪台设备上加载权重”。

对应视频 P32（P33 是完结篇，介绍如何阅读开源项目）。用训练得到的权重对**单张图**做推理，预处理必须与训练一致。

1. `Image.open` 读图
2. `convert('RGB')`：PNG 可能是 `RGBA`，四通道进 `Conv2d(in_channels=3)` 会报错
3. `Resize((32, 32))`：训练 CIFAR 用的就是 32×32
4. `ToTensor()`：变成 `[0,1]` 的 Tensor，与训练时一致
5. `reshape(1, 3, 32, 32)` 补 batch 维；缺少这维时，`Flatten` 会把第 0 维（通道）当成 batch，最后在 `Linear` 处报形状错误（见下方实测）
6. 加载模型（与保存方式对应）
7. `model.eval()` + `torch.no_grad()`
8. `output.argmax(1)` 得到类别下标
9. 用类名列表把下标映射成类名

把每一步“为什么”展开说：

**第 1 步 读图。** `Image.open(path)` 返回 PIL 图片对象，`print(image)` 会显示 `mode`（通道格式）和 `size`。注意 PIL 的 `size` 是 **(宽, 高)**，而 `Resize((32, 32))` 的参数和张量的形状都是 **(高, 宽)**。路径写错会报 `FileNotFoundError: [Errno 2] No such file or directory`；相对路径 `../imgs/airplane.png` 是相对于**运行脚本时所在的目录**，不是脚本文件所在目录。

**第 2 步 `convert('RGB')`，为什么需要。** PNG 格式支持透明度，常见的截图、网上下载的 PNG 是 **RGBA**：R、G、B 三个颜色通道 + 1 个 A（alpha，透明度）通道，共 4 个通道。而模型第一层是 `Conv2d(3, 32, 5, 1, 2)`，权重形状 `[32, 3, 5, 5]`，只接受 3 通道。实测把 RGBA 图直接送进去：

```text
RuntimeError: Given groups=1, weight of size [32, 3, 5, 5], expected input[1, 4, 32, 32] to have 3 channels, but got 4 channels instead
```

`convert('RGB')` 丢掉 alpha 通道，变成 3 通道。对本来就是 RGB 的 JPG 图调用它也没有副作用，所以**无论什么图都加上这一句**最省心。灰度图（mode 为 `L`，1 通道）同样会被转成 3 通道。

**第 3 步 `Resize((32, 32))`。** 网络里 `Linear(64*4*4, 64)` 的 `64*4*4` 是按 32×32 的输入算出来的（32 → 16 → 8 → 4，三次池化每次减半）。输入如果是 200×150，最后展平就不是 1024，`Linear` 处报 `mat1 and mat2 shapes cannot be multiplied`。所以必须缩放到训练时的尺寸。

**第 4 步 `ToTensor()`。** PIL 图 `[H, W, C]`、取值 0–255 的整数 → 张量 `[C, H, W]`、取值 `[0, 1]` 的 float，和训练时 `transform=torchvision.transforms.ToTensor()` 完全一致。如果训练时还用了 `Normalize`，这里也必须加上同样参数的 `Normalize`。

**第 5 步 `reshape` 补 batch 维。** 模型是按 `[N, C, H, W]` 一批一批训练的，单张图 `[3, 32, 32]` 需要变成 `[1, 3, 32, 32]`（N=1，一批只有 1 张）。写法：`torch.reshape(image, (1, 3, 32, 32))` 或更通用的 `image.unsqueeze(0)`（在第 0 维插入长度为 1 的维度），两者结果相同。

> [!NOTE]+ 不补 batch 维到底在哪里报错（实测）
>
> 新版 PyTorch 的 `Conv2d` / `MaxPool2d` 其实**接受**不带 batch 的 3 维输入 `[C, H, W]`（当作“单张图”处理），所以前面几层都不报错：`Conv2d(3, 32, 5, 1, 2)` 把 `[3, 32, 32]` 变成 `[32, 32, 32]`，三层下来得到 `[64, 4, 4]`。问题出在 `nn.Flatten()`：它默认 `start_dim=1`，**把第 0 维当成 batch**，于是 `[64, 4, 4]` 被展成 `[64, 16]`（“64 张图、每张 16 个数”）——也就是“把通道误当成 batch”。最终在 `Linear` 处报：
>
> ```text
> RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x16 and 1024x64)
> ```
>
> 看到 `(64x16 and 1024x64)` 这种左边第一个数恰好等于通道数的报错，基本就是忘了补 batch 维。

**第 6 步 加载模型。** 两种保存方式对应两种加载（详见“持久化”一节）：

- 方式 1 整模型 `torch.save(tudui, ...)`：`model = torch.load(path, map_location=..., weights_only=False)`，当前文件里必须有 `Tudui` 类的定义（或 `from model import *`）。
- 方式 2 只存参数 `torch.save(tudui.state_dict(), ...)`：先 `model = Tudui()`，再 `model.load_state_dict(torch.load(path, map_location=...))`。

**第 7 步 `eval()` + `no_grad()`。** `model.eval()` 切换到测试模式，影响 `Dropout`、`BatchNorm` 这类训练/测试行为不同的层（Tudui 里没有，但养成习惯）；`torch.no_grad()` 不记录计算图，省内存、更快，推理本来也不需要反向传播。

**第 8 步 `argmax(1)`。** 输出 `[1, 10]` 是 10 个类别的分数（logits），在第 1 维（类别维）上取最大值的位置，得到形如 `tensor([0])` 的下标张量；`.item()` 取出 Python 整数 `0`。

**第 9 步 映射类名。** 用一个按 CIFAR-10 顺序排好的列表 `classes[idx]` 把下标翻译成人能看懂的名字（见下方对照表）。也可以从数据集对象取：`train_data.classes`。

GPU 上训练、CPU 上加载时必须 `map_location=torch.device('cpu')`，否则会试图把存储里的 CUDA 张量还原到 GPU。反过来 CPU 权重放到 GPU 可在 load 后 `model.to('cuda')`，输入也要 `.to('cuda')`。

把“在哪训练 / 在哪加载”四种组合列清楚：

| 训练设备 | 加载设备 | 怎么写 |
| :--- | :--- | :--- |
| CPU | CPU | `torch.load(path)` 直接加载 |
| CPU | GPU | 正常 `torch.load(path)`，之后 `model = model.to("cuda")`，输入 `image = image.to("cuda")` |
| GPU | GPU | 直接加载即可（权重会回到保存时的 `cuda:0`），输入也要 `.to("cuda")` |
| GPU | CPU（无 GPU 机器） | **必须** `torch.load(path, map_location=torch.device("cpu"))` |

最后一种如果忘了 `map_location`，在无 GPU 机器上报（报错信息取自 PyTorch 源码）：

```text
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```

最省心的通用写法：**不管在哪训练，一律先映射到 CPU，再按当前设备搬过去**：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(path, map_location="cpu"))  # 先全部落到 CPU，任何机器都不会报错
model = model.to(device)                                       # 再按本机情况决定去不去 GPU
image = image.to(device)                                       # 输入跟着模型走
```

官方 `test.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../imgs/airplane.png"  # 相对“运行目录”的路径，找不到会 FileNotFoundError
image = Image.open(image_path)  # 第 1 步：读图，得到 PIL 对象
print(image)  # 形如 <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=...>
image = image.convert('RGB')  # 第 2 步：RGBA(4 通道) -> RGB(3 通道)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])  # 第 3、4 步：缩放到 32×32，再转 [0,1] 张量

image = transform(image)
print(image.shape)  # torch.Size([3, 32, 32])

class Tudui(nn.Module):  # 整模型加载需要这个类定义，必须与训练时一致
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu'))  # 第 6 步：GPU 训练的整模型映射到 CPU；PyTorch ≥ 2.6 还需加 weights_only=False
print(model)
image = torch.reshape(image, (1, 3, 32, 32))  # 第 5 步：[3, 32, 32] -> [1, 3, 32, 32]
model.eval()  # 第 7 步：测试模式
with torch.no_grad():  # 不建计算图
    output = model(image)  # [1, 3, 32, 32] -> [1, 10]
print(output)  # 10 个类别的分数（logits），不是概率

print(output.argmax(1))  # 第 8 步：形如 tensor([0])，0 对应 airplane
```

> [!WARNING]+ PyTorch 2.6 起 `torch.load` 整模型会报错
>
> PyTorch 2.6 把 `torch.load` 的 `weights_only` 默认值从 `False` 改成了 `True`（只允许加载张量、字典等“安全”内容，防止恶意 pickle 文件执行代码）。上面 `test.py` 加载的是**整个模型对象**，在本机 PyTorch 2.6 上实测报：
>
> ```text
> _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
> (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. ...
> ```
>
> 修法二选一：
>
> 1. 文件是自己训练出来的、可信：`torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu'), weights_only=False)`；
> 2. 更推荐：训练时改存 `state_dict`，加载时 `Tudui()` + `load_state_dict(...)`，默认的 `weights_only=True` 就能正常工作（实测输出 `<All keys matched successfully>`）。

下面是一份可以在纯 CPU 机器上直接运行的完整验证脚本，按上面的 9 步逐行标注。没有现成的 airplane.png，就先用 PIL 造一张 RGBA 的 PNG；模型用随机权重演示流程，所以预测结果没有意义，重点看每一步的形状：

```python
import torch
import torchvision
from PIL import Image
from torch import nn

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]          # CIFAR-10 下标 -> 类名

class Tudui(nn.Module):                                      # 结构必须与训练时一模一样
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.Linear(64, 10))
    def forward(self, x):
        return self.model(x)

# 第 0 步（仅演示用）：造一张 RGBA 的 PNG，模拟截图/网上下载的图
Image.new("RGBA", (200, 150), (30, 144, 255, 255)).save("demo.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open("demo.png")               # 第 1 步：读图
print(image.mode, image.size)                # RGBA (200, 150)  注意 PIL 的 size 是 (宽, 高)
image = image.convert("RGB")                 # 第 2 步：丢掉 alpha 通道，4 通道 -> 3 通道
print(image.mode)                            # RGB
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)), # 第 3 步：缩放到训练时的 32×32（(高, 宽)）
    torchvision.transforms.ToTensor(),       # 第 4 步：PIL -> [3, 32, 32] float，值 [0,1]
])
image = transform(image)
print(image.shape)                           # torch.Size([3, 32, 32])
image = torch.reshape(image, (1, 3, 32, 32)) # 第 5 步：补 batch 维 -> [1, 3, 32, 32]
print(image.shape)                           # torch.Size([1, 3, 32, 32])

torch.manual_seed(0)                         # 固定随机初始化，演示结果可复现
model = Tudui()                              # 第 6 步：建模型 + 加载权重
torch.save(model.state_dict(), "demo_sd.pth")                     # 演示：先存一份随机权重
model.load_state_dict(torch.load("demo_sd.pth", map_location="cpu"))  # 实际使用时换成训练得到的 .pth
model = model.to(device)
image = image.to(device)                     # 输入与模型同设备

model.eval()                                 # 第 7 步：eval 模式 + 关闭梯度
with torch.no_grad():
    output = model(image)
print(output.shape)                          # torch.Size([1, 10])
idx = output.argmax(1)                       # 第 8 步：取分数最大的下标
print(idx, idx.item())                       # tensor([3]) 3（随机权重，结果无意义）
print(classes[idx.item()])                   # 第 9 步：下标 -> 类名，cat
```

CIFAR-10 下标与类别：

| 下标 | 类别 |
| :--- | :--- |
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

`output` 形状 `[1, 10]`，`argmax(1)` 得到形如 `tensor([0])` 表示 airplane。仓库示例图是 `airplane.png`，训练充分时该下标应为 0。

如果想看“有多确定”，可以把 logits 变成概率：`torch.softmax(output, dim=1)` 得到 10 个和为 1 的数，最大的那个就是模型对预测类别的置信度。`argmax` 的结果不受 softmax 影响（softmax 不改变大小顺序）。另外，训练轮数少时（例如只训了 1 轮，测试正确率 30% 左右）预测错是正常的；视频里用训练 30 轮的 `tudui_29_gpu.pth`，飞机图才被认对。

方式 1 保存整模型时，加载侧必须能找到 `Tudui` 类，因此 `test.py` 把网络结构又写了一遍。`map_location` 写在 `torch.load` 里，与后面 `image` 是否 `.cuda()` 无关：先把权重映射到 CPU，再按当前设备决定是否把 model 和 image 搬到 GPU。较新 PyTorch 加载完整模块时可能还需 `weights_only=False`。

（补充：本机 PyTorch 2.6 实测，加载完整模块**一定**需要 `weights_only=False`，见上面的 WARNING。）

命令行传参（视频提及）形式为 `python test.py --image dog.png`，本仓库脚本把路径写死在文件里。

如果想实现命令行传参，可以用 Python 自带的 `argparse`：

```python
import argparse
parser = argparse.ArgumentParser()                                  # 创建参数解析器
parser.add_argument("--image", default="../imgs/airplane.png")      # 声明一个 --image 参数，不传时用默认值
args = parser.parse_args()                                          # 解析命令行
image = Image.open(args.image)                                      # 用传进来的路径读图
```

运行 `python test.py --image dog.png` 时 `args.image == "dog.png"`；直接 `python test.py` 则用默认路径。

> [!WARNING]+ 常见报错与排查（验证相关）
>
> | 报错信息（节选） | 原因 | 修法 |
> | :--- | :--- | :--- |
> | `expected input[1, 4, 32, 32] to have 3 channels, but got 4 channels instead` | PNG 是 RGBA，没 `convert('RGB')` | 读图后加 `image = image.convert('RGB')` |
> | `mat1 and mat2 shapes cannot be multiplied (64x16 and 1024x64)` | 忘了补 batch 维 | `torch.reshape(image, (1, 3, 32, 32))` 或 `image.unsqueeze(0)` |
> | `mat1 and mat2 shapes cannot be multiplied (1xXXXX and 1024x64)` | 没 `Resize((32, 32))`，展平后长度不是 1024 | 预处理加 `Resize((32, 32))` |
> | `Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False` | GPU 训练的权重在无 GPU 机器上加载 | `torch.load(..., map_location=torch.device('cpu'))` |
> | `Weights only load failed` | PyTorch ≥ 2.6 加载整模型 | 加 `weights_only=False`，或改用 `state_dict` |
> | `AttributeError: Can't get attribute 'Tudui' on <module '__main__'>` | 整模型加载时当前文件里没有 `Tudui` 类 | 把类定义复制过来，或 `from model import *` |
> | `Expected all tensors to be on the same device` | 模型搬到了 GPU，图片还在 CPU | `image = image.to(device)` |
> | 不报错但每张图都预测成同一类 | 训练轮数太少，或预处理与训练不一致（如训练用了 `Normalize` 而预测没用） | 多训几轮；逐项核对 transform |

> [!TIP]+ 小结
>
> - 预测时的预处理必须和训练一致：`convert('RGB')` → `Resize((32, 32))` → `ToTensor()`（训练有 `Normalize` 就也要有）。
> - 单张图要补 batch 维：`[3, 32, 32]` → `[1, 3, 32, 32]`。
> - 推理三件套：`model.eval()`、`with torch.no_grad():`、`output.argmax(1)`，最后用类名列表翻译下标。
> - GPU 训练的权重拿到 CPU 上用：`map_location=torch.device('cpu')`；通用做法是“先映射到 CPU，再 `.to(device)`”。
> - PyTorch ≥ 2.6 整模型加载要 `weights_only=False`；优先保存/加载 `state_dict`。

<br>

## 复习：完整知识清单与最小训练模板

> [!NOTE]+ 这一节要解决什么
>
> 前面各节是按视频顺序一块块学的，这一节把它们串成一条线：先用 7 步回顾“一条样本从磁盘到类别编号”的路径，再给一张**可以对照查找的速查表**（每个环节用到的 API、关键参数、形状变化、易错点），最后给一份**最小可运行的训练模板**（逐行注释，CPU 上实测跑通），以后写新的训练脚本可以直接从它改起。

一条样本从磁盘到类别编号的路径：

1. `Dataset` / `torchvision.datasets` 读出图像与标签（`__getitem__` / `__len__`）
2. `transforms`：`ToTensor`、`Normalize`、`Resize`、`Compose`、`RandomCrop`
3. `DataLoader` 堆成 batch `[N, C, H, W]`（`shuffle` / `drop_last` / `num_workers`）
4. `nn.Module`：`Conv2d` → `MaxPool2d` → 激活 → `Flatten` → `Linear`，输出 `[N, 10]` logits
5. `CrossEntropyLoss`；`zero_grad` → `backward` → `step`；外层 `epoch`
6. 测试：`eval()` + `no_grad()`，`argmax(1)` 累加正确个数
7. 保存优先 `state_dict`；GPU 权重用 `map_location`；推理前 `Resize` 到训练输入尺寸，并 `convert('RGB')`

### 速查表：每个环节用什么、注意什么

| 环节 | 核心 API | 关键参数 / 用法 | 形状或类型变化 | 易错点 |
| :--- | :--- | :--- | :--- | :--- |
| 查文档 | `dir()`、`help()` | `dir(torch.nn)` 列出内容；`help(nn.Conv2d)` 看说明 | — | 先看 `__init__` 参数和 `forward` 输入形状 |
| 自定义数据集 | `torch.utils.data.Dataset` | 继承后实现 `__init__`、`__getitem__(idx)`、`__len__` | `idx` → `(img, label)` | `__getitem__` 要返回单个样本，不是 batch |
| 内置数据集 | `torchvision.datasets.CIFAR10` | `root`、`train=True/False`、`transform`、`download=True` | 50000 训练 / 10000 测试，每张 3×32×32 | 不写 `transform` 时拿到的是 PIL 图，不能直接进模型 |
| 可视化 | `SummaryWriter` | `add_scalar(tag, y, x)`、`add_image(tag, img, step, dataformats=...)`；`tensorboard --logdir=logs` | — | 忘了 `writer.close()`；numpy 图要写 `dataformats="HWC"` |
| 变换 | `transforms.ToTensor`、`Normalize`、`Resize`、`Compose`、`RandomCrop` | `Normalize(mean, std)`：$x' = (x-\text{mean})/\text{std}$；`Resize((h, w))` | PIL `[H, W, C]` 0–255 → Tensor `[C, H, W]` 0–1 | `Compose` 里顺序有讲究：`Normalize` 要放在 `ToTensor` 之后 |
| 打包 | `DataLoader` | `batch_size`、`shuffle`、`drop_last`、`num_workers` | `[C, H, W]` × N → `[N, C, H, W]`；标签 → `[N]` | Windows 下 `num_workers>0` 需放在 `if __name__ == "__main__":` 里 |
| 骨架 | `nn.Module` | `super().__init__()`；在 `forward` 里写前向计算 | — | 调用用 `model(x)`，不要直接 `model.forward(x)` |
| 卷积 | `nn.Conv2d` | `in_channels, out_channels, kernel_size, stride=1, padding=0` | `[N, C_in, H, W]` → `[N, C_out, H', W']` | `in_channels` 必须等于输入通道数 |
| 池化 | `nn.MaxPool2d` | `kernel_size`；`stride` 默认等于 `kernel_size`；`ceil_mode` | 高宽缩小为约 1/k，通道不变 | 不改变通道数 |
| 非线性 | `nn.ReLU`、`nn.Sigmoid` | `inplace=False`（默认） | 形状不变 | — |
| 线性 | `nn.Flatten`、`nn.Linear` | `Linear(in_features, out_features)` | `[N, 64, 4, 4]` → `[N, 1024]` → `[N, 64]` | `in_features` 要等于展平后的长度 |
| 串联 | `nn.Sequential` | 按顺序把层装进去 | — | 不确定形状时用 `torch.ones(64, 3, 32, 32)` 喂一下看输出 |
| 损失 | `nn.CrossEntropyLoss`、`L1Loss`、`MSELoss` | 输入 logits `[N, C]` + 标签 `[N]`（类别下标，`long`） | → 标量 | 不要先手动 softmax；标签不是 one-hot |
| 优化 | `torch.optim.SGD` / `Adam` | `params=model.parameters()`、`lr` | — | 三步顺序：`zero_grad` → `backward` → `step` |
| 预训练模型 | `torchvision.models.vgg16` | `weights=...`（旧版 `pretrained=True`）；改 `classifier` 适配 10 类 | — | `add_module` 追加 / 直接替换 `classifier[6]` |
| 保存/加载 | `torch.save`、`torch.load`、`load_state_dict` | 方式 1 整模型；方式 2 `state_dict`（推荐） | — | 整模型加载需要类定义，PyTorch ≥ 2.6 需 `weights_only=False` |
| 训练循环 | `model.train()`、`model.eval()`、`torch.no_grad()` | 外层 epoch，内层 batch | — | `.item()` 取数再累加；测试时别忘 `no_grad` |
| 设备 | `torch.device`、`.to(device)`、`.cuda()` | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | — | 模型、loss、imgs、targets 都要搬；张量 `.to` 必须赋值 |
| 验证 | `Image.open`、`convert('RGB')` | `map_location=torch.device('cpu')` | `[3, 32, 32]` → `[1, 3, 32, 32]` → `[1, 10]` → 下标 | 预处理与训练一致 |

形状口算：卷积 $n_\text{out}=\lfloor(n+2p-f)/s\rfloor+1$；`MaxPool2d(k)` 默认 stride=$k$；CIFAR 这条网最后是 `64×4×4=1024`。

把 Tudui 每一层用公式算一遍（$n$ 输入边长，$p$ 填充，$f$ 卷积核大小，$s$ 步长）：

$$
\text{Conv2d}(3,32,5,1,2):\ \lfloor(32+2\times2-5)/1\rfloor+1=32
$$

$$
\text{MaxPool2d}(2):\ \lfloor(32-2)/2\rfloor+1=16
$$

同理第二组得到 16 → 8，第三组得到 8 → 4，所以 `Flatten` 后长度是 $64\times4\times4=1024$。`padding=2` 正是为了让 5×5 卷积“不缩小”：$2p - f + 1 = 0$，即 $p=(f-1)/2=2$。

参数量（以后估模型大小用）：卷积层参数 = $C_\text{out}\times C_\text{in}\times f\times f + C_\text{out}$（最后一项是偏置），线性层 = $\text{in}\times\text{out}+\text{out}$：

| 层 | 计算 | 参数量 |
| :--- | :--- | :--- |
| `Conv2d(3, 32, 5)` | $32\times3\times5\times5+32$ | 2432 |
| `Conv2d(32, 32, 5)` | $32\times32\times5\times5+32$ | 25632 |
| `Conv2d(32, 64, 5)` | $64\times32\times5\times5+64$ | 51264 |
| `Linear(1024, 64)` | $1024\times64+64$ | 65600 |
| `Linear(64, 10)` | $64\times10+10$ | 650 |
| 合计 | | 145578 |

可以用 `sum(p.numel() for p in model.parameters())` 验证，结果正是 145578。池化、`Flatten` 没有参数。

查阅习惯贯穿始终：`dir` / `help` / 跳进源码看 `__init__` 与 `forward` 的参数约定，再对照官方文档核对形状。

### 训练一个模型的标准 8 步

1. **定设备**：`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
2. **准备数据集**：`Dataset` + `transform`，训练集、测试集各一个。
3. **打包**：`DataLoader`，训练集 `shuffle=True`。
4. **建模型**：继承 `nn.Module`，`.to(device)`。
5. **损失与优化器**：`CrossEntropyLoss().to(device)`、`optim.SGD(model.parameters(), lr=...)`。
6. **训练循环**：`model.train()` → 每个 batch：搬数据 → 前向 → 算 loss → `zero_grad` → `backward` → `step`。
7. **测试循环**：`model.eval()` + `no_grad()` → 累加 loss 和正确个数 → 算正确率。
8. **保存与使用**：`torch.save(model.state_dict(), ...)`；预测时建同结构模型 → `load_state_dict` → 预处理单张图 → `argmax`。

### 最小可运行训练模板

下面的脚本把 8 步全部写齐，用 `torchvision.datasets.FakeData` 生成形状与 CIFAR-10 一样的随机图片（3×32×32、10 类），不需要下载任何数据，CPU 上几秒钟跑完。换成真数据只需改第 2 步的两行。因为标签是随机的，正确率在 10% 左右是正常现象，模板的目的是验证流程能跑通。

```python
# ===== 0. 导入 =====
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

torch.manual_seed(0)                                   # 固定随机种子，结果可复现

# ===== 1. 设备：有 GPU 用 GPU，没有就用 CPU =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)                               # 本机输出 device: cpu

# ===== 2. 数据：FakeData 代替 CIFAR10（形状一样：3×32×32，10 类）=====
transform = torchvision.transforms.ToTensor()          # PIL -> [3, 32, 32] 的 float Tensor，值在 [0,1]
train_data = torchvision.datasets.FakeData(size=512, image_size=(3, 32, 32), num_classes=10,
                                           transform=transform, random_offset=0)
test_data = torchvision.datasets.FakeData(size=128, image_size=(3, 32, 32), num_classes=10,
                                          transform=transform, random_offset=10000)
# 换成真数据只需改这两行：torchvision.datasets.CIFAR10(root="./data", train=True/False, transform=transform, download=True)

# ===== 3. 打包：DataLoader 把单张样本堆成 batch =====
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)    # 训练集打乱；512/64 = 8 个 batch
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)     # 测试集不必打乱；128/64 = 2 个 batch

# ===== 4. 模型 =====
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),     # [N, 3, 32, 32] -> [N, 32, 32, 32]
            nn.MaxPool2d(2),               # -> [N, 32, 16, 16]
            nn.Conv2d(32, 32, 5, 1, 2),    # -> [N, 32, 16, 16]
            nn.MaxPool2d(2),               # -> [N, 32, 8, 8]
            nn.Conv2d(32, 64, 5, 1, 2),    # -> [N, 64, 8, 8]
            nn.MaxPool2d(2),               # -> [N, 64, 4, 4]
            nn.Flatten(),                  # -> [N, 1024]
            nn.Linear(64 * 4 * 4, 64),     # -> [N, 64]
            nn.Linear(64, 10),             # -> [N, 10]，10 个类别的 logits
        )

    def forward(self, x):
        return self.model(x)

model = Tudui().to(device)                             # 模型搬到 device（Module 的 .to 原地生效）

# ===== 5. 损失与优化器 =====
loss_fn = nn.CrossEntropyLoss().to(device)             # 无参数的 loss 搬不搬都行，统一写上不吃亏
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)  # 优化器本身不用 .to；习惯上在 model.to 之后再创建

# ===== 6. 训练 + 测试循环 =====
epochs = 2
for epoch in range(epochs):
    model.train()                                      # 训练模式（影响 Dropout/BatchNorm）
    for imgs, targets in train_loader:
        imgs = imgs.to(device)                         # Tensor 的 .to 返回新对象，必须赋值回去
        targets = targets.to(device)
        outputs = model(imgs)                          # [64, 3, 32, 32] -> [64, 10]
        loss = loss_fn(outputs, targets)               # 标量
        optimizer.zero_grad()                          # ① 清空上一步的梯度
        loss.backward()                                # ② 反向传播算梯度
        optimizer.step()                               # ③ 按梯度更新参数

    model.eval()                                       # 测试模式
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():                              # 不建计算图，省内存、更快
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            total_loss += loss_fn(outputs, targets).item()                    # .item() 取出 Python float
            total_correct += (outputs.argmax(1) == targets).sum().item()      # 预测对的个数
    acc = total_correct / len(test_data)
    print(f"epoch {epoch + 1}: test loss {total_loss:.4f}, acc {acc:.4f}")
    # 本机输出：
    # epoch 1: test loss 4.6206, acc 0.0781
    # epoch 2: test loss 4.6186, acc 0.0781

# ===== 7. 保存（推荐只存参数）=====
torch.save(model.state_dict(), "tudui_template.pth")

# ===== 8. 加载 + 单张预测 =====
new_model = Tudui()                                                    # 先建同结构的空模型
state = torch.load("tudui_template.pth", map_location=torch.device("cpu"))  # 不管在哪训练的，先映射到 CPU
new_model.load_state_dict(state)
new_model = new_model.to(device)
new_model.eval()
img, label = test_data[0]                              # img: [3, 32, 32]
with torch.no_grad():
    pred = new_model(img.unsqueeze(0).to(device))      # 补 batch 维 -> [1, 3, 32, 32]，输出 [1, 10]
print("pred:", pred.argmax(1).item(), "label:", label) # 本机输出 pred: 0 label: 9（随机标签，猜错正常）
```

读输出的方法：`test loss` 是 2 个测试 batch 的 loss **之和**。10 类均匀乱猜时单个 batch 的交叉熵约为 $-\ln(1/10)=\ln 10\approx2.303$，两个 batch 加起来约 $4.61$，与输出的 4.6206 吻合，说明模型此时等于在乱猜（标签本来就是随机的）。换成真正的 CIFAR-10 后，loss 应逐轮下降、正确率逐轮上升。

> [!INFO]+ 进阶
>
> 官方 README 将目标检测入门当作本教程的后续：[BV19Z31z8ENH](https://www.bilibili.com/video/BV19Z31z8ENH/)。代码仍以本仓库 `src/` 为准。

> [!TIP]+ 小结
>
> - 整条流程：数据集 → 变换 → DataLoader → 模型 → 损失 → 优化器 → 训练/测试循环 → 保存 → 加载预测，每一环都在速查表里有对应行。
> - 形状是排错的第一线索：卷积公式 $n_\text{out}=\lfloor(n+2p-f)/s\rfloor+1$，Tudui 是 32 → 16 → 8 → 4，展平 1024。
> - 训练三步 `zero_grad` → `backward` → `step`；测试三件套 `eval()`、`no_grad()`、`argmax(1)`。
> - 设备一律 `.to(device)`；保存一律 `state_dict`；加载一律先 `map_location="cpu"`。
> - 写新项目时，从“最小可运行训练模板”复制一份，替换数据集和模型两处即可。
