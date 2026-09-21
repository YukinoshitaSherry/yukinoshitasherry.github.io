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

配套视频：[PyTorch深度学习快速入门教程（绝对通俗易懂！）](https://www.bilibili.com/video/BV1hE411t7RN/)（B 站 `BV1hE411t7RN`，约 33 讲，2021 年完结）。官方仓库代码均在 [`src/`](https://github.com/xiaotudui/pytorch-tutorial/tree/master/src)。下列笔记按视频顺序整理，源码以仓库为准。

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
| GPU 与验证 | P30–P33 | `train_gpu_1.py`、`train_gpu_2.py`、`test.py` |

> [!NOTE]+ 文件名与分 P
>
> 仓库里的 `P8_Tensorboard.py`、`P9_transforms.py`、`P10_dataset_transform.py` 与视频分 P **不是**严格同号：视频 P8–P9 对应 TensorBoard，P10–P13 对应 Transforms，P14 对应 CIFAR 内置集。后半脚本不再带 `Pxx` 前缀。

<br>

## 环境

对应视频 P1–P3、P5。目标是得到一个能 `import torch`、必要时能用 GPU 的解释器，并选好写代码的界面。

### 安装

到 [pytorch.org](https://pytorch.org/) 按操作系统、包管理器（conda / pip）、CUDA 版本生成命令。直接执行无 CUDA 标记的 `pip install torch` 时，默认源经常装到 CPU 版或与本机 CUDA 不匹配的构建。

常见检查：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

- `import torch` 无报错：CPU 版至少装好
- `torch.cuda.is_available()` 为 `True`：当前这个解释器能调 CUDA

`False` 的常见原因（视频 FAQ）：

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| 能 import，CUDA 为 False | 装的是 CPU wheel | 卸载后按官网 CUDA 命令重装 |
| 有 NVIDIA 显卡仍为 False | 驱动过旧，或 PyTorch 编译所用 CUDA 与驱动不兼容 | 升级驱动，或改选官网给出的 CUDA 11.x / 12.x 组合 |
| 同一台机器有的环境 True、有的 False | PyCharm / Jupyter 没用到装过 GPU 版的那个 conda 环境 | 在 IDE 里把 Interpreter 指到该环境 |

本地已下载的 conda 包可用：

```bash
conda install --use-local 包名.tar.bz2
```

> [!NOTE]+ 环境变量
>
> 将 Python、Scripts、CUDA 的 `bin` 加入 PATH 后，命令行不必先 `cd` 到该目录即可调用 `python`、`pip`、`nvcc`。视频中的 `nb_conda` 用于在 Notebook 里切换 conda 环境；较新的 Jupyter 直接在界面选择 kernel 即可，该扩展已非必需。

### 编辑器

对比 PyCharm 与 Jupyter：

| | PyCharm | Jupyter |
| :--- | :--- | :--- |
| 适合 | 多文件项目、跳转源码、断点 | 逐步试 API、看中间 tensor |
| 运行 | 整文件 / 配置好的运行项 | 按单元格，左侧数字表示已跑完，`*` 表示仍在跑 |
| 多行试验 | Python Console 里 Shift+Enter 换行 | 单元格内直接换行 |
| 跳源码 | Ctrl 点击符号；Alt+7 结构视图；Ctrl+P 看参数 | `对象??` 显示源码 |

相对路径以**当前工作目录**为基准，不是以 `.py` 文件所在目录为基准（除非先改 cwd）。例如脚本在 `Project/train/A.py`，图片在 `Project/image/B.jpg`，若 cwd 为 `train/`，则图片相对路径为 `../image/B.jpg`。

课程后续代码大量依赖「能打开源码查看 `__init__` / `forward`」，因此至少保证有一种方式能打开 PyTorch 源文件。

<br>

## 查阅

对应视频 P4。文档和源码比记忆 API 列表更重要：课程后半所有层、损失、优化器都按同一套办法查。

### dir

`dir(obj)` 列出对象的属性与方法名，相当于打开工具箱。不传参数时列出当前作用域中的名字。

```python
import torch

dir(torch)
dir(torch.cuda)
dir(torch.cuda.is_available)  # 注意：后面不加括号，查的是函数对象本身
```

`dir(torch.cuda.is_available())` 会先**调用**函数（返回 `True`/`False`），再对布尔值做 `dir`，看到的就不是该函数的属性了。

双下划线包裹的名字是特殊方法（magic / dunder），例如 `__init__`、`__call__`、`__getitem__`、`__len__`，用来实现构造、当函数调用、下标访问、`len()` 等语法。不允许当普通字段随意改。

### help

`help(obj)` 打印文档字符串，相当于说明书。

```python
help(torch.cuda.is_available)
help(torch.utils.data.Dataset)
help(torch.utils.tensorboard.SummaryWriter.add_image)
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

<br>

## 数据集

对应视频 P6–P7。数据从磁盘到网络分成两步：

- **Dataset**：把样本编成 `0, 1, 2, …`，按编号取出「一条数据 + 标签」
- **DataLoader**：把 Dataset 打成 mini-batch，并可打乱、多进程加载

Dataset 按索引提供单条样本；DataLoader 负责组 batch、打乱与多进程读取。本节只实现 Dataset；P15 再讲打包。

### 接口

```python
from torch.utils.data import Dataset
help(Dataset)
```

`Dataset` 是抽象类。自定义类必须继承它，并实现：

| 方法 | 作用 | 支撑的语法 |
| :--- | :--- | :--- |
| `__init__` | 记录根目录、类别名、文件列表等**全局**信息 | `MyData(...)` |
| `__getitem__(idx)` | 按索引读**一条**样本 | `dataset[0]` |
| `__len__` | 返回样本总数 | `len(dataset)` |

`__init__` 里只做「列清单」，不宜把全部图片读进内存。真正读盘发生在 `__getitem__`，DataLoader 取第 `i` 张时才打开那张文件。

### self 与路径

`self.xxx` 是实例属性，在 `__init__` 里赋值后，`__getitem__` / `__len__` 都能用。`os.path.join` 按操作系统选择 `/` 或 `\`，避免手写斜杠在 Windows / Linux 间出错。`os.listdir(path)` 返回该目录下文件名列表（无序，需要标签文件与图像一一对应时要 `sort()`）。

视频用的蚂蚁/蜜蜂目录（文件夹名即标签）：

```text
dataset/train/
    ants/   0013035.jpg ...
    bees/   16838648_415acd9e3f.jpg ...
dataset/val/
    ants/
    bees/
```

### 视频版

视频演示用的蚂蚁/蜜蜂数据集：根目录下按类别建文件夹，**文件夹名即标签**。`os.path.join` 按操作系统自动选择路径分隔符。

```python
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ants_dataset = MyData(root_dir, "ants")
bees_dataset = MyData(root_dir, "bees")
img, label = ants_dataset[0]
print(label)
# img.show()

train_dataset = ants_dataset + bees_dataset  # Dataset 支持相加拼接
print(len(ants_dataset), len(bees_dataset), len(train_dataset))
```

执行 `ants_dataset[0]` 时的顺序：

1. 解释器调用 `MyData.__getitem__(ants_dataset, 0)`
2. `img_name = self.img_path[0]`
3. `os.path.join` 拼出完整路径
4. `Image.open` 得到 PIL 图像；`label` 取文件夹名 `"ants"`
5. 返回 `(img, label)` 元组

`img.show()` 可弹出系统看图窗口。`ants_dataset + bees_dataset` 得到 `ConcatDataset`：下标先走 ants，再走 bees，长度相加，顺序不变。

> [!WARNING]+ 返回类型
>
> 此处返回的是 PIL `Image`，还不能直接进卷积。后面要用 `transforms.ToTensor()` 转成 `[C, H, W]` 的 FloatTensor。两个 Dataset 相加要求两边 `__getitem__` 返回结构一致（都是 `(img, label)` 或都是 dict）。

### 仓库版

官方 `read_data.py` 把图像目录与标签文件目录分开，并对文件名排序以保证一一对应，再接入 `transforms`：

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter("logs")

class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)

        with open(label_item_path, 'r') as f:
            label = f.readline()

        # img = np.array(img)
        img = self.transform(img)
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    root_dir = "dataset/train"
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    train_dataset = ants_dataset + bees_dataset

    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    writer.add_image('error', train_dataset[119]['img'])
    writer.close()
```

与视频版的差别：图像与标签分目录存放，文件名排序后按同一下标对齐；`__getitem__` 返回 `{'img': ..., 'label': ...}`，因此 TensorBoard 里写 `train_dataset[119]['img']`。标签 `'error'` 只是事件名，不代表程序出错。`num_workers>0` 时，Windows 上入口代码必须放在 `if __name__ == '__main__':` 里，否则多进程会重复 import。视频里文件夹常叫 `ants` / `bees`，仓库示例用 `ants_image` 与 `ants_label` 分开，路径需与本地目录一致。

`__len__` 里的 `assert` 保证图像数与标签文件数相等，不相等时尽早报错，避免静默错位。

<br>

## 看板

对应视频 P8–P9。TensorBoard 把标量、图像、计算图写入事件文件，浏览器中查看训练曲线与样本。先安装：

```bash
pip install tensorboard
```

### SummaryWriter

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

`writer` 用完必须 `close()`，否则最后几条可能没落盘。

### 标量

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)  # tag, Y, X
writer.close()
```

`add_scalar(tag, scalar_value, global_step)`：

- `tag`：图标题。相同 tag 画在同一张图
- `scalar_value`：纵轴，如 loss、准确率
- `global_step`：横轴，通常用训练步数

在事件文件所在目录启动：

```bash
tensorboard --logdir=logs
```

默认端口 `6006`，浏览器打开终端打印的 URL（多为 `http://localhost:6006/`）。冲突时：

```bash
tensorboard --logdir=logs --port=6007
```

同一 `tag` 反复跑实验、又不删旧事件文件时，曲线会叠在一起（例如先写 `y=x` 再写 `y=2x` 仍用标题 `"y=x"`）。处理：删掉 `logs` 里的 `events.out.tfevents.*`，Ctrl+C 停掉旧 tensorboard，再启动。

### 图像

`add_image` 要求 `img_tensor` 为 `torch.Tensor`、`numpy.ndarray` 或路径字符串。PIL 的 `Image.open` 得到的不是这三种，必须转换。

| 来源 | `type` | 典型 `shape` |
| :--- | :--- | :--- |
| `PIL.Image.open` | `PIL.Image.Image` | 无 `.shape`，有 `.size = (W, H)` |
| `np.array(PIL图)` | `numpy.ndarray` | `(H, W, C)` 即 HWC |
| `cv2.imread` | `numpy.ndarray` | `(H, W, C)`，通道为 BGR |
| `transforms.ToTensor()` | `torch.Tensor` | `(C, H, W)` 即 CHW |

`add_image` **默认按 CHW**。HWC 必须声明 `dataformats='HWC'`，否则会按 `(3, H, W)` 去解释 `(H, W, 3)`，直接报错或花屏。

```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants_image/0013035.jpg")
print(type(img))            # PIL.Image.Image
img_array = np.array(img)
print(type(img_array), img_array.shape)  # ndarray, 例如 (512, 768, 3)
writer.add_image("Img Test", img_array, 1, dataformats="HWC")
writer.close()
```

`global_step` 可用来在同一 tag 下切换多张图（拖动滑条）。批量图像用 `writer.add_images`（复数，要求第一维是 batch）；计算图用 `writer.add_graph(model, input)`。

官方 `P8_Tensorboard.py`：

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/6240329_72c01e663e.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()
```

批量图像用 `writer.add_images`（复数），计算图用 `writer.add_graph(model, input)`。仓库这份脚本把 `y=2x` 写成了 `3*i`，只是演示横纵轴，标题与公式不必一致；正式记曲线时应让 tag 与含义对应，避免旧事件叠图。

<br>

## 变换

对应视频 P10–P13。`torchvision.transforms` 是视觉预处理工具箱（不用于文本）：缩放、裁剪、翻转、归一化、PIL → Tensor 等。用法模式都一样：先用类**构造模具**得到实例，再把图像**当参数调用**该实例。

### 可调用对象

`transforms.ToTensor()` 返回的是**实例**。对该实例写 `tensor_trans(img)` 能跑，是因为类实现了 `__call__`：`obj(x)` 等价于 `obj.__call__(x)`。这与 `nn.Module`、以及后面几乎所有 transform 相同。

```python
class Person:
    def __call__(self, name):
        print("__call__ Hello " + name)

    def hello(self, name):
        print("Hello " + name)

person = Person()
person.hello("li4")   # 必须写方法名
person("zhang3")      # 走 __call__，不需要点号
```

```python
from PIL import Image
from torchvision import transforms

img = Image.open("dataset/train/bees/16838648_415acd9e3f.jpg")
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(type(tensor_img), tensor_img.shape)  # torch.Tensor, [C, H, W]
```

`ToTensor` 同时解决两件事：

1. 类型：PIL / `uint8` ndarray → `torch.FloatTensor`
2. 数值：对 **uint8** 的 `[0, 255]` 除以 255，落到 `[0.0, 1.0]`；通道从 HWC 改为 CHW。若 ndarray 已经是 float，则**不再**除 255

OpenCV 的 `cv2.imread` 得到 `numpy.ndarray`（BGR），也可直接交给 `ToTensor`，但颜色通道顺序与 PIL 的 RGB 不同。单通道 PIL（`mode='L'`）得到 `[1, H, W]`，不是 3 通道。

> [!NOTE]+ Tensor
>
> Tensor 是神经网络专用的多维数组，除了存数值，还可携带梯度、设备（CPU/GPU）等信息。`nn.Conv2d`、损失函数和优化器都要求 Tensor；PIL 需先经 `ToTensor`（或 `np.array`）转换。

### 常用算子

课程强调每用一个 transform 都先确认：**输入类型 → 输出类型**、`__init__` 要哪些参数。

**Normalize** `[mean, …], [std, …]`：按通道

$$
\text{out}[c] = \frac{\text{in}[c] - \mathrm{mean}[c]}{\mathrm{std}[c]}
$$

三通道都用 `0.5` 时，原 `[0, 1]` 被拉到 `[-1, 1]`。例如某像素 `0.8`：$(0.8-0.5)/0.5=0.6$。可用 `print(img_tensor[0][0][0])` 与归一化后对比。

**Resize**：

- `Resize((H, W))`：强制拉到该高宽，可能变形
- `Resize(512)`：短边缩到 512，长边按比例，图像仍是 PIL（若输入是 PIL）

类型链常见为：PIL → Resize → PIL → ToTensor → Tensor。

**Compose**：参数是 **transform 实例的列表**，按顺序执行。`Compose([Resize(512), ToTensor()])` 等价于先 Resize 再 ToTensor。

**RandomCrop**：在画面中随机位置切出 `size`。`RandomCrop(512)` 切正方形；`RandomCrop((50, 50))` 切指定高宽。裁剪边长大于原图会报错。循环多次并 `add_image(..., global_step=i)` 可在 TensorBoard 里来回看不同裁剪。

关注输入输出类型：未知就 `print(type(x))` / `print(x.size)`（PIL）/ `print(x.shape)`（Tensor）。参数不确定时 Ctrl+P。

```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/bees/16838648_415acd9e3f.jpg")

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

trans_resize = transforms.Resize((512, 512))
img_resize = trans_totensor(trans_resize(img))
writer.add_image("Resize", img_resize, 0)

trans_compose = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor()
])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    writer.add_image("RandomCrop", trans_compose_2(img), i)

writer.close()
```

> [!WARNING]+ 仓库 `P9_transforms.py`
>
> 该类 `__getitem__` 里写的是 `img = transform(img)`，没有用 `self.transform`。能跑是因为文件后部定义了同名全局变量 `transform`，属于笔误。自定义 Dataset 应写 `self.transform(img)`，并先判断 `if self.transform is not None`。

<br>

## 内置集

对应视频 P14。`torchvision.datasets` 提供 CIFAR、MNIST 等。不必手写 `MyData`，但返回约定仍是「下标 → (图像, 标签)」。

CIFAR-10：60000 张 `32×32` RGB，10 类每类 6000 张；训练 50000、测试 10000。类别名为

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

对应编号 `0 … 9`。

### 下载

```python
import torchvision

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)
```

| 参数 | 含义 |
| :--- | :--- |
| `root` | 存放目录 |
| `train` | `True` 训练集，`False` 测试集（必须大写） |
| `download` | `True` 时若本地没有就下载；已存在则校验，不重复下 |
| `transform` | 对**图像**做的变换；`None` 时 `__getitem__` 返回 PIL |

不设 `transform` 时：

```python
print(test_set[0])
# (PIL.Image.Image image mode=RGB size=32x32, 6)  视频示例：target=6 对应 frog
print(test_set.classes)

img, target = test_set[0]
print(type(img), img.size)          # PIL, (32, 32)
print(target)                       # int
print(test_set.classes[target])     # 类别字符串
# img.show()
```

PIL 不能直接进卷积，也不能直接 `add_image`（除非先 ToTensor / ndarray）。因此训练前几乎总会：

```python
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
```

`transform=` 传入后，变换发生在该数据集的 `__getitem__` 内部，外部仍写 `img, target = test_set[i]`，只是 `img` 已是 Tensor。

官方 `P10_dataset_transform.py`：

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

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

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
```

循环里 `writer.add_image("test_set", img, i)` 用 `global_step=i` 切换前 10 张。此时 `img` 已是 CHW Tensor，不必写 `dataformats`。

打开 `CIFAR10` 源码中的 `__getitem__`，能确认返回的是 `(img, target)`，后面 DataLoader 才会解包成 `imgs, targets = data`。

<br>

## 打包

对应视频 P15。Dataset 一次给一张；网络训练按 batch 算梯度。`DataLoader` 负责堆叠、打乱、多进程预取。

```python
torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=False
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

测试集 10000 张、`batch_size=64`：`10000 = 156×64 + 16`。

- `drop_last=False`：157 批，最后一批 16 张，形状 `[16, 3, 32, 32]`
- `drop_last=True`：156 批，全部 `[64, 3, 32, 32]`，后面若把 `Linear` 的 `in_features` 按 batch=64 写死，就不会因最后一批形状变化而报错

单张展示用 `add_image`，一整批用 `add_images`。

官方 `dataloader.py`：

```python
import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()
```

外层 `for epoch` 控制扫几遍数据；内层 `for data in loader` 取出一个 batch。`shuffle=True` 时两个 epoch 在 TensorBoard 同一滑条位置上的图像不同。`step` 按 batch 递增，作为 `add_images` 的 `global_step`。

> [!NOTE]+ 迭代约定
>
> `for data in test_loader` 里的 `data` 就是 `__getitem__` 返回值按 batch 堆好的结果。CIFAR 为 `(imgs, targets)`；若 Dataset 返回 dict，则 `data` 是 dict，键对应的 value 带了 batch 维。也可写成 `for imgs, targets in test_loader`，与先取 `data` 再解包等价。

<br>

## 骨架

对应视频 P16。神经网络写在 `torch.nn`。自定义模型必须继承 `nn.Module`，在 `__init__` 里**构造并注册**层，在 `forward` 里写数据怎么流。

`torch.nn` 与 `torch.nn.functional`（常 `import torch.nn.functional as F`）的关系：后者是函数（每次传入 weight）；前者把 weight 存成模块参数，适合搭网络。课程后半用 `nn.Conv2d` 等层，手算卷积时用 `F.conv2d`。

官方文档里的典型写法：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

必须遵守：

1. `__init__` 里调用 `super().__init__()`（或 `super(Tudui, self).__init__()`），否则参数不会注册到 `model.parameters()`，优化器更新不到
2. 方法名必须是 `forward`，拼错则调用实例时不会走到自定义计算
3. 层写成 `self.conv1 = ...`；若只写成局部变量 `conv1 = ...`，则不算子模块，不会出现在 `parameters()` 里

实例 `tudui(x)` 并没有手写 `__call__`，但父类 `nn.Module` 实现了 `__call__`，内部会调用 `forward`。因此应使用 `tudui(x)`，而不是直接 `tudui.forward(x)`（后者会跳过 hook、`train/eval` 相关逻辑）。

官方 `nn_module.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
```

输入 `1.0` 时输出 `tensor(2.)`。可用 PyCharm 的 Step into My Code 跟进：

1. `tudui = Tudui()` → `__init__` → `super().__init__()`
2. `tudui(x)` → `Module.__call__` → `Tudui.forward`
3. `input + 1` 后返回

这是后面所有 `Conv2d` / `Sequential` 网络的最小骨架。

<br>

## 卷积

对应视频 P17（运算）与 P18（层）。图像在 PyTorch 里默认 **NCHW**：`N` batch，`C` 通道，`H` 高，`W` 宽。

### 运算

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

输出空间尺寸（`dilation=1` 时与课程公式一致）：

$$
n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - f}{s} \right\rfloor + 1
$$

通道关系：

- 输入通道数 = 单个卷积核的通道数
- 输出通道数 = 卷积核个数（`out_channels`）

每个核与输入**所有通道**做相关后再相加，得到 **1** 个输出通道；`out_channels` 个核就得到相应数量的特征图。

官方 `nn_conv.py` 用 `5×5` 输入与 `3×3` 核对 `stride`、`padding` 做数值验算：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆

import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
```

`input` 与 `kernel` 需 reshape 成 `(N, C, H, W)` 与 `(C_out, C_in, kH, kW)`。缺维时常见报错是 `weight should have at least three dimensions`。`padding=1` 表示上下左右各补一圈 0，输入从 `5×5` 变成 `7×7`。

> [!WARNING]+ 浮点类型
>
> `F.conv2d` / `nn.Conv2d` 在较新的 PyTorch 中要求浮点（或复数）。官方 `nn_conv.py` 未指定 dtype，默认整型，可能报 `expected scalar type Float`。补救：构造时加 `dtype=torch.float32`。后面池化示例已使用 `float32`。

左上角 `3×3` 与核的点积（stride=1 的第一个输出）：

$$
\begin{aligned}
&1\cdot1 + 2\cdot2 + 0\cdot1 \\
+\; &0\cdot0 + 1\cdot1 + 2\cdot0 \\
+\; &1\cdot2 + 2\cdot1 + 1\cdot0
= 10
\end{aligned}
$$

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

### 层

网络里用 `nn.Conv2d`（层里带可学习的 weight / bias），而不是每次手传 kernel。完整签名：

```python
torch.nn.Conv2d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, dilation=1, groups=1,
    bias=True, padding_mode='zeros'
)
```

`padding_mode` 可为 `'zeros'` / `'reflect'` / `'replicate'` / `'circular'`。保持特征图边长常用 `padding = (kernel_size - 1) // 2`（奇数核）。

CIFAR 输入 `[64, 3, 32, 32]`，`Conv2d(3, 6, 3)` 后为 `[64, 6, 30, 30]`，因为 $(32+0-3)/1+1=30$，6 个核 → 6 通道。`add_images` 只接受 3 通道，仓库里用 `reshape` 把 6 通道拆成更多张 3 通道图：`[64, 6, 30, 30] → [128, 3, 30, 30]`（`-1` 让 batch 维自动算）。这只是为了能显示，**不是**训练时的标准操作，通道被拆开后语义已乱。

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

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1

writer.close()
```

<br>

## 池化

对应视频 P19。最大池化在窗口内取最大值，用一个数代表一块区域，降低分辨率、减少计算。特点：

1. 无可学习参数（没有 weight 要更新）
2. 通道数不变：输入 3 通道，输出仍 3 通道
3. 对微小平移相对稳健

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

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()

writer = SummaryWriter("../logs_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
```

手算示例（`ceil_mode=True`，核 `3×3`）：

```python
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        return self.maxpool1(input)

print(Tudui()(input))
```

`5×5`、核 `3×3`、stride 默认 3：

- `ceil_mode=True`：能划出 2 个窗口方向，输出 `2×2`，约为 `[[2, 3], [5, 1]]`（左上窗最大 2，右上 3，左下 5，右下不完整窗最大 1）
- `ceil_mode=False`：右、下不足 3 的边丢掉，只剩左上一个窗，输出 `1×1` 的 `[[2.]]`

CIFAR 上池化后仍是 3 通道，因此 `add_images` 不必像卷积那样 `reshape` 通道。

<br>

## 非线性

对应视频 P20。任意多层线性变换复合后仍是一次线性变换：$W_2(W_1 x)= (W_2 W_1)x$。ReLU、Sigmoid 等激活打断这种复合，网络才能拟合更复杂的函数。

- ReLU：$\mathrm{ReLU}(x)=\max(0,x)$，负值置零，正值原样。计算便宜，是卷积网络里最常用的
- Sigmoid：$\sigma(x)=1/(1+e^{-x})$，值域 `(0, 1)`。CIFAR 像素已在 `[0, 1]` 时，Sigmoid 会把两端往 0.5 附近压，图像发灰、对比变弱
- `inplace=True`：在原 tensor 上改，省内存，但会丢掉改前的值，不利于某些需要输入本身的反向实现；默认 `False`

小例子（ReLU）：

```python
import torch
from torch import nn
from torch.nn import ReLU

x = torch.tensor([[1.0, -0.5],
                  [-1.0, 3.0]])
x = torch.reshape(x, (-1, 1, 2, 2))  # 补成 NCHW

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        return self.relu1(input)

print(Tudui()(x))
# 负值变 0：[[1, 0], [0, 3]]
```

`reshape` 成 `(-1, 1, 2, 2)` 是为了和图像张量一样带上 batch、通道维，`-1` 自动填 batch=1。

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
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
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

<br>

## 线性

对应视频 P21。全连接层 `nn.Linear(in_features, out_features)` 做 $y = xW^T + b$，要求输入最后一维等于 `in_features`。卷积输出是 `[N, C, H, W]`，必须先展平。

两种展平：

```python
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
torch.flatten(t)                 # 从 dim0 到最后 → 长度 8
torch.flatten(t, start_dim=1)    # 保留 batch → [2, 4]
```

- `torch.flatten(imgs)` 默认 `start_dim=0`，会把 **batch 也展进去**：`[64, 3, 32, 32] → [196608]`，于是 `Linear(196608, 10)` 输出 `[10]`，相当于把整个 batch 当成一条样本
- 正规写法是 `torch.flatten(imgs, start_dim=1)` → `[64, 3072]`，再 `Linear(3072, 10)` → `[64, 10]`
- `torch.reshape(imgs, (1, 1, 1, -1))` 同样得到长度 196608 的一条向量，课程用来迁就 `Linear(196608, 10)`

`64×3×32×32 = 196608`。最后一批若不足 64，展平长度就不是 196608，会和 `in_features` 对不上，所以该示例 `drop_last=True`。

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

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
```

<br>

## 串联

对应视频 P22。后面 CIFAR-10 分类网络的结构为：

`Conv 3→32 (k=5, p=2)` → `MaxPool 2` → `Conv 32→32` → `MaxPool 2` → `Conv 32→64` → `MaxPool 2` → `Flatten` → `Linear 1024→64` → `Linear 64→10`

`kernel=5` 时 `padding=(5-1)/2=2`，空间边长不变。`MaxPool2d(2)` 默认 stride=2，边长减半。

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

这里的 `Flatten` 是 `nn.Flatten()`，默认 `start_dim=1`，**保留 batch**，与上一节手写 `torch.flatten(imgs)` 不同。用 `torch.ones((64, 3, 32, 32))` 走一遍，确认输出为 `[64, 10]`；对不上就回头查哪一层的 padding / 通道写错。

不使用 Sequential 时每一层都要起名，并在 `forward` 里依次调用，便于中途 `print(x.shape)`：

```python
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import torch

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
```

`nn.Sequential` 按添加顺序执行子模块，`forward` 只需一行。`print(tudui)` 会打印整棵模块树。`writer.add_graph(tudui, input)` 在 TensorBoard 的 GRAPHS 里展开 `Tudui` 可看数据流。

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
        x = self.model1(x)
        return x

tudui = Tudui()
print(tudui)
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input)
writer.close()
```

等价的非 Sequential 写法是把每一层赋给 `self.conv1`、`self.maxpool1` … 再在 `forward` 里依次调用，便于插入调试打印，但更冗长。

<br>

## 损失

对应视频 P23。损失函数度量预测与标签的差距，越小越好。得到标量 `loss` 后：

1. `loss.backward()` 沿计算图把梯度写进各参数的 `.grad`
2. 优化器再用 `.grad` 改权重

选哪种损失要看任务，并且**输入形状必须符合该损失的文档**。

### 回归

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

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)


x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
```

`inputs=[1,2,3]`，`targets=[1,2,5]`：

- L1、`reduction='sum'`：$|3-5|=2$
- MSE 默认 mean：$((0)^2+(0)^2+(2)^2)/3 = 4/3 \approx 1.333$

交叉熵示例：logits $x=[0.1,0.2,0.3]$，真实类 $y=1$。`CrossEntropyLoss` 内部先 softmax 再取 $-\log p_y$，不必在网络末尾再接 Softmax。

$$
p_i = \frac{e^{x_i}}{\sum_j e^{x_j}},\quad
\mathcal{L} = -\log p_{y}
$$

$e^{0.1}\approx 1.105$，$e^{0.2}\approx 1.221$，$e^{0.3}\approx 1.350$，和 $\approx 3.676$，$p_1\approx 0.332$，$-\log 0.332 \approx 1.10$。真实类对应的 logit 相对越大，损失越小。

### 分类

`nn.CrossEntropyLoss` 用于多分类。约定：

- 网络输出 `outputs` 形状 `[N, C]`（**未经过** softmax 的 logits）
- 标签 `targets` 形状 `[N]`，值为类别下标 `0 … C-1`（不是 one-hot）

CIFAR-10 最后一层 `Linear(64, 10)` 正是 10 类 logits。`N=1` 时 `outputs` 为 `[1, 10]`，`targets` 为 `[1]` 的 LongTensor，二者匹配。

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

dataloader = DataLoader(dataset, batch_size=1)

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
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, targets)
    print("ok")
```

在 `result_loss.backward()` 处打断点，可在调试器里看 `tudui.model1[0].weight.grad`：未 backward 时为 `None`，backward 后为与 weight 同形状的梯度。此处还没有 `optimizer.step()`，权重数值不变。

> [!WARNING]+ 与 Softmax 重复
>
> `CrossEntropyLoss` 内部已含 log-softmax。`forward` 末尾若再接 `Softmax`，等于做了两次，梯度会变差。需要概率时在推理阶段单独 `softmax`。

<br>

## 优化器

对应视频 P24。`torch.optim` 按梯度改参数。SGD 示意：$w \leftarrow w - \mathrm{lr}\cdot \partial L/\partial w$。

标准三步，顺序固定：

1. `optimizer.zero_grad()`：清空上一轮累积的梯度
2. `loss.backward()`：计算当前 batch 的梯度
3. `optimizer.step()`：按学习率更新参数

PyTorch 默认把梯度**累加**到 `.grad`。一个 batch 用完必须清零，否则下一个 batch 的梯度会叠加上去。视频里强调：先 zero_grad，再算 loss / backward，再 step（loss 也可以在 zero_grad 前算，但不能在 backward 之后还不 zero 就进入下一轮）。

两个循环职责不同：

| 循环 | 含义 |
| :--- | :--- |
| `for data in dataloader` | 处理**一个 batch** |
| `for epoch in range(E)` | 把**整份数据扫 E 遍** |

只跑内层一轮，loss 往往几乎不动；需要外层 epoch。课程示例 `batch_size=1` 扫测试集 20 轮，仅用于看 `running_loss` 下降，速度很慢，看几个 epoch 即可停。

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

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

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
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
```

`running_loss` 是该 epoch 所有 batch 的 loss 之和。此处写的是 `running_loss + result_loss`（张量相加），打印会带 `grad_fn`，计算图也会被挂在 `running_loss` 上，轮数多时浪费内存。正式代码应使用 `running_loss += result_loss.item()`。`tudui.parameters()` 把 `requires_grad=True` 的参数交给优化器。学习率 `1e-2` 即 $1\times 10^{-2}=0.01$。常用还有 `Adam`；课程用 SGD 是为了步骤最直白。文件里 `from torch.optim.lr_scheduler import StepLR` 未使用，可忽略。

<br>

## 预训练

对应视频 P25。`torchvision.models` 提供 VGG、ResNet 等现成结构。ImageNet 预训练权重在 1000 类上训练过，迁到 CIFAR-10 时要改最后的分类头。

视频中的 API：

```python
vgg16_false = torchvision.models.vgg16(pretrained=False)  # 仅结构，随机初始化
vgg16_true = torchvision.models.vgg16(pretrained=True)    # 结构 + ImageNet 权重
```

> [!WARNING]+ 新版 API
>
> 较新的 torchvision 用 `weights` 替代 `pretrained`，例如 `vgg16(weights=None)`、`vgg16(weights='DEFAULT')`。权重默认缓存于用户目录下的 `.cache/torch/hub/checkpoints`，可用 `torch.hub.set_dir(...)` 改路径。下载失败时可把 `.pth` 手动放到该目录。

`print(vgg16_true)` 能看到两段：`features`（卷积）与 `classifier`（若干 `Linear` + Dropout）。原 `classifier[6]` 是 `Linear(4096, 1000)`。

迁到 10 类的两种改法：

1. **追加**：`vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))`，1000 维后再映到 10 维。若写成 `vgg16_true.add_module(...)` 会加在顶层，与 `classifier` 并列，前向**不会**自动经过这一层
2. **替换**：`vgg16_false.classifier[6] = nn.Linear(4096, 10)`，直接改最后一层输出维

> [!NOTE]+ 输入尺寸
>
> ImageNet 预训练的 VGG 按约 `224×224` 的输入训练。较新 torchvision 在 `features` 后有 `AdaptiveAvgPool2d((7, 7))`，因此 `32×32` 也能算出 `512×7×7` 并接到 `classifier`，不一定报错；但感受野与预训练分布仍按大图设计，直接拿 CIFAR 原图微调效果通常较差。要贴近预训练设定需 `Resize(224)`。`model.py` 里的小网络才是为 32×32 设计的。

官方 `model_pretrained.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
```

`print(vgg16_true)` 可看到 `features`（卷积）与 `classifier`（全连接）两段。`add_module` 加在 `vgg16_true` 顶层会成为与 `classifier` 并列的模块；加在 `vgg16_true.classifier` 上才会接在 1000 维输出之后。

<br>

## 持久化

对应视频 P26。两种保存方式：

| 方式 | 调用 | 内容 | 加载 |
| :--- | :--- | :--- | :--- |
| 1 | `torch.save(model, path)` | 结构 + 参数（pickle 整个模块） | `torch.load(path)` |
| 2（官方推荐） | `torch.save(model.state_dict(), path)` | 仅 `OrderedDict` 参数 | 先建同结构模型，再 `load_state_dict` |

方式 2 文件更小、不绑定类的定义位置，换设备、改代码结构时更好控。`torch.load("vgg16_method2.pth")` 得到的是字典，直接 `print` 不是网络，必须先 `vgg16 = torchvision.models.vgg16(pretrained=False)` 再 `load_state_dict`。

方式 1 的陷阱：pickle 会记录**类的定义位置**。在 `model_save.py` 里定义 `Tudui` 并保存整模型，到另一个文件 `torch.load` 时，若当前进程找不到 `Tudui`，会报 `AttributeError: Can't get attribute 'Tudui'`。Jupyter 同一 kernel 里刚定义过该类时有时能加载成功，换成 PyCharm 分文件则会报错。处理：

- `from model_save import Tudui`（加载前把类引进当前模块）
- 或把类定义复制到加载文件

方式 2 无此问题，因为结构由加载代码自己构造。

> [!WARNING]+ `torch.load` 默认值
>
> PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`，加载方式 1 的完整模块可能直接失败。加载课程里的 `.pth` 时可写 `torch.load(path, map_location=..., weights_only=False)`（仅限信任的文件）。方式 2 的 `state_dict` 更符合该默认。

官方 `model_save.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")
```

官方 `model_load.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from model_save import *
# 方式1-》保存方式1，加载模型
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
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

model = torch.load('tudui_method1.pth')
print(model)
```

`from model_save import *` 正是为了把 `Tudui` 引进当前模块，方式 1 才能反序列化成功。与上一节相同，新版本 `torch.load` 加载完整模块时可能需要 `weights_only=False`。

GPU 上保存、CPU 上加载时指定：

```python
model = torch.load("tudui_29_gpu.pth", map_location=torch.device("cpu"))
```

<br>

## 训练

对应视频 P27–P29。完整套路：准备数据 → 建网 → 损失与优化器 → 按 epoch 训练 → 测试集评估 → 存模型 → TensorBoard 记录。

网络单独放在 `model.py`，训练脚本 `from model import *`。若 import 失败，把该目录标成 Sources Root（PyCharm）。`if __name__ == '__main__'` 里用全 1 张量检查输出是否为 `[64, 10]`，避免训练跑起来才发现 `Linear` 输入维写错。

`nn.Conv2d(3, 32, 5, 1, 2)` 五个位置参数依次是 `in_channels, out_channels, kernel_size, stride, padding`，与前面 `Conv2d(3, 32, 5, padding=2)` 等价。

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


if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)
```

### 辅助

`loss.item()` 把只含一个元素的张量转成 Python `float`。直接 `print(loss)` 会带 `grad_fn`，写入 TensorBoard 也需要纯数值。

```python
tensor = torch.tensor([3.14])
value = tensor.item()  # 3.14，类型 float
```

`with torch.no_grad():` 临时关闭梯度。测试集不需要 backward，不建图能省显存、加快速度。`with` 结束自动恢复。训练阶段的 `loss.backward()` 若包在 `no_grad` 内，则没有梯度，优化器无法更新。

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

`total_accuracy / test_data_size` 得到准确率。`total_accuracy` 是张量，打印时会显示 `tensor(...)`，可再 `.item()`。

`model.train()` / `model.eval()`：

| | `train()` | `eval()` |
| :--- | :--- | :--- |
| Dropout | 按概率置零 | 关闭，全部神经元保留 |
| BatchNorm | 用当前 batch 统计，并更新 running mean/var | 使用训练期累计的统计量 |

本课程的 Tudui 没有这两类层，不切换也能跑；有 BN / Dropout 时必须在训练循环开头 `train()`、测试循环开头 `eval()`。

### 循环

每个 epoch 内部顺序：

1. `tudui.train()`，遍历 `train_dataloader`：前向 → 损失 → `zero_grad` → `backward` → `step`；每 100 步打印并 `add_scalar("train_loss", ...)`
2. `tudui.eval()` + `no_grad`，遍历测试集，累加 loss 与正确个数
3. `torch.save(tudui, "tudui_{}.pth".format(i))` 按轮次存盘（方式 1）

`total_train_step` 跨 epoch 累加，作为 TensorBoard 横轴；`total_test_step` 每个 epoch 加一。`total_test_loss` 累加的是各 batch 的 **mean** loss（`CrossEntropyLoss` 默认 `reduction='mean'`），不是「全测试集逐样本平均」；最后一批较小时代入横轴时只作趋势，不宜当精确指标。准确率用 `total_accuracy / test_data_size`，分母是整集长度，与 `drop_last=False` 的官方 `train.py` 一致。

官方 `train.py`（依赖同目录 `model.py`）：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆

import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

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
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
```

`train-cpu.py` 把 `Tudui` 写在同一文件里，不依赖 `from model import *`，其余逻辑相同。10 个 epoch、CIFAR-10 全量训练在 CPU 上会较慢；接上 GPU 版后主要加速的是卷积与矩阵乘，DataLoader 读盘仍在 CPU。

> [!NOTE]+ 字符串格式化
>
> `"训练数据集的长度为：{}".format(train_data_size)` 把 `{}` 替换为参数。训练脚本里轮次、步数、loss 均用此写法打印。

<br>

## 设备

对应视频 P30–P31。网络、损失、输入、标签必须在同一设备，否则 `Expected all tensors to be on the same device`。优化器不必单独 `.cuda()`，它持有的是参数引用，参数搬家后优化器仍指向它们。

需要搬的对象：

- 模型：`tudui.cuda()` / `tudui.to(device)`
- 损失函数（若内部有可学习缓冲，如某些带 weight 的 loss）：同样搬一次
- **每个 batch** 的 `imgs`、`targets`（DataLoader 默认在 CPU 上取出）

### cuda

方式 1：`.cuda()`。仓库里用 `torch.cuda.is_available()` 包一层，无 GPU 时仍可在 CPU 跑。

官方 `train_gpu_1.py` 相对 CPU 版的差异：

```python
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 训练 / 测试循环内：
imgs, targets = data
if torch.cuda.is_available():
    imgs = imgs.cuda()
    targets = targets.cuda()
```

其余与 `train.py` 相同（网络定义内联在该文件中）。

### to

方式 2（更常用）：先定义 `device`，再 `.to(device)`。换 CPU / 多卡时只改一处。`torch.device("cuda")` 等价于当前默认 GPU（一般是 `cuda:0`）；指定卡用 `"cuda:1"`。无 GPU 的写法：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 或视频中的写法（假定一定有 GPU）：
# device = torch.device("cuda")

tudui = tudui.to(device)
loss_fn = loss_fn.to(device)
imgs = imgs.to(device)
targets = targets.to(device)
```

`.cuda()` 只能去 NVIDIA GPU；`.to(device)` 还能去 `cpu`、`mps`（Apple）等，因此更通用。仓库 `train_gpu_2.py` 写死 `"cuda"`，没有 GPU 时会在 `.to(device)` 处报错，需改成上面的三元表达式。

官方 `train_gpu_2.py`：

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
device = torch.device("cuda")

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Tudui(nn.Module):
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
tudui = Tudui()
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

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
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
```

> [!NOTE]+ 赋值
>
> 对 Tensor，必须写 `imgs = imgs.to(device)`，因为 `to` 可能返回新对象。对 `nn.Module`，`model.to(device)` 是原地移动，有无赋值通常都能工作；课程代码仍写成赋值，两种设备迁移风格保持一致。

<br>

## 验证

对应视频 P32–P33。用训练得到的权重对**单张图**做推理，预处理必须与训练一致。

1. `Image.open` 读图
2. `convert('RGB')`：PNG 可能是 `RGBA`，四通道进 `Conv2d(in_channels=3)` 会报错
3. `Resize((32, 32))` + `ToTensor()`：训练 CIFAR 就是 32×32、`[0,1]` Tensor
4. `reshape(1, 3, 32, 32)` 补 batch 维；缺少这维会把通道/高宽误当成 batch
5. `model.eval()` + `torch.no_grad()`
6. `output.argmax(1)` 得到类别下标

GPU 上训练、CPU 上加载时必须 `map_location=torch.device('cpu')`，否则会试图把存储里的 CUDA 张量还原到 GPU。反过来 CPU 权重放到 GPU 可在 load 后 `model.to('cuda')`，输入也要 `.to('cuda')`。

官方 `test.py`：

```python
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../imgs/airplane.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Tudui(nn.Module):
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

model = torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
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

方式 1 保存整模型时，加载侧必须能找到 `Tudui` 类，因此 `test.py` 把网络结构又写了一遍。`map_location` 写在 `torch.load` 里，与后面 `image` 是否 `.cuda()` 无关：先把权重映射到 CPU，再按当前设备决定是否把 model 和 image 搬到 GPU。较新 PyTorch 加载完整模块时可能还需 `weights_only=False`。

命令行传参（视频提及）形式为 `python test.py --image dog.png`，本仓库脚本把路径写死在文件里。

<br>

## 复习

一条样本从磁盘到类别编号的路径：

1. `Dataset` / `torchvision.datasets` 读出图像与标签（`__getitem__` / `__len__`）
2. `transforms`：`ToTensor`、`Normalize`、`Resize`、`Compose`、`RandomCrop`
3. `DataLoader` 堆成 batch `[N, C, H, W]`（`shuffle` / `drop_last` / `num_workers`）
4. `nn.Module`：`Conv2d` → `MaxPool2d` → 激活 → `Flatten` → `Linear`，输出 `[N, 10]` logits
5. `CrossEntropyLoss`；`zero_grad` → `backward` → `step`；外层 `epoch`
6. 测试：`eval()` + `no_grad()`，`argmax(1)` 累加正确个数
7. 保存优先 `state_dict`；GPU 权重用 `map_location`；推理前 `Resize` 到训练输入尺寸，并 `convert('RGB')`

形状口算：卷积 $n_\text{out}=\lfloor(n+2p-f)/s\rfloor+1$；`MaxPool2d(k)` 默认 stride=$k$；CIFAR 这条网最后是 `64×4×4=1024`。

查阅习惯贯穿始终：`dir` / `help` / 跳进源码看 `__init__` 与 `forward` 的参数约定，再对照官方文档核对形状。

> [!INFO]+ 进阶
>
> 官方 README 将目标检测入门当作本教程的后续：[BV19Z31z8ENH](https://www.bilibili.com/video/BV19Z31z8ENH/)。代码仍以本仓库 `src/` 为准。
