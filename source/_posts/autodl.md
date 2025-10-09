---
title: AutoDL使用教程
date: 2025-04-14
categories:
- 学CS/SE
tags:
- Tools
- GPU
desc: 算力租赁平台AutoDL使用教程
---

## 前期准备

### 实例选择

1.登录并充值。

2.容器实例-租用新实例。

3.选一种显卡，配版本环境。

<Img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/20250526102918919.png" style="width: 90%;">
<br>

#### 显卡

| 显卡型号                          | 显存规格           | 显存带宽      | 核心架构         | CUDA核心数量 | 功耗 (W) | 价格范围 (人民币)            | 适用场景            |
| ----------------------------- | -------------- | --------- | ------------ | -------- | ------ | --------------------- | --------------- |
| **NVIDIA A800**               | 80GB HBM2e     | 1.935TB/s | Ampere       | 未公开      | 300    | 约8.5-13万              | 高性能计算、深度学习训练    |
| **NVIDIA A100**               | 80GB/40GB HBM2 | 2TB/s     | Ampere       | 10816    | 300    | 单卡约16万元以上             | 深度学习训练、高性能计算    |
| **NVIDIA H100**               | 80GB HBM3      | 3TB/s     | Hopper       | 未公开      | 700    | 单卡约30万元以上             | 大型语言模型训练、AI推理   |
| **NVIDIA RTX 3090**           | 24GB GDDR6X    | 936GB/s   | Ampere       | 10496    | 350    | 约11999元起              | 高性能AI训练、游戏      |
| **NVIDIA RTX 5090**           | 24GB GDDR6X    | 1TB/s     | Ada Lovelace | 未公开      | 450    | 约2.1万                 | 高性能AI训练、游戏      |
| **NVIDIA RTX 5080**           | 16GB GDDR7     | 700GB/s   | Ada Lovelace | 未公开      | 320    | 约8299元                | 中等规模AI模型训练      |
| **NVIDIA RTX 4090**           | 24GB GDDR6X    | 1TB/s     | Ada Lovelace | 16384    | 450    | 约2.1万                 | 高性能AI训练、游戏      |
| **NVIDIA RTX 5070 Ti**        | 12GB GDDR6X    | 672GB/s   | Ada Lovelace | 未公开      | 285    | 约4000-5000元           | AI推理、轻量训练       |
| **NVIDIA RTX 4070 Ti SUPER**  | 16GB GDDR6X    | 608GB/s   | Ada Lovelace | 未公开      | 235    | 约3500-4500元           | AI推理、轻量训练       |
| **NVIDIA RTX 4060 Ti (16GB)** | 16GB GDDR6     | 256GB/s   | Ada Lovelace | 未公开      | 115    | 约2500-3000元           | 轻量AI推理、本地AI绘图   |
| **NVIDIA A40**                | 24GB GDDR6     | 544GB/s   | Ampere       | 未公开      | 180    | 约1.5-2万               | 数据中心、专业应用       |
| **NVIDIA RTX A6000**          | 48GB GDDR6     | 912GB/s   | Ampere       | 未公开      | 300    | 约3-4万                 | 深度学习训练、专业图形处理   |
| **NVIDIA V100**               | 32GB HBM2      | 900GB/s   | Volta        | 5120     | 300    | 单卡约8-10万元             | 深度学习训练、科学计算     |
| **AMD Radeon VII**            | 32GB HBM2      | 1TB/s     | Vega         | 未公开      | 300    | 已停产，二手市场价格约1000-2000元 | 轻量级AI训练、图形处理    |
| **AMD Instinct MI25**         | 16GB HBM2      | 1TB/s     | Vega         | 未公开      | 300    | 已停产，二手市场价格约1500-2500元 | AI训练、科学计算       |
| **AMD Instinct MI200**        | 128GB HBM2     | 3.2TB/s   | CDNA 2       | 未公开      | 560    | 单卡约10-15万元            | 大规模AI模型训练、高性能计算 |
| **AMD Radeon RX 7900 XTX**    | 24GB GDDR6     | 1.6TB/s   | RDNA 3       | 6144     | 355    | 约8000元左右              | 高性能AI训练、游戏      |
| **AMD Radeon RX 7900 XT**     | 20GB GDDR6     | 1.3TB/s   | RDNA 3       | 5376     | 315    | 约5000元左右              | 高性能AI训练、游戏      |
| **AMD Radeon RX 6900 XT**     | 16GB GDDR6     | 780GB/s   | RDNA 2       | 4608     | 300    | 约4000元左右              | 高性能AI训练、游戏      |
| **AMD Radeon Pro W6800**      | 32GB GDDR6     | 512GB/s   | RDNA 2       | 未公开      | 200    | 约10000元左右             | 专业图形处理、轻量AI训练   |


<br>

4.创建后自动开机，不用记得及时关掉省钱。

5.如果关机时间段内显卡被其他用户占用则这个实例不能用了，一天可以免费克隆3次去其他型号。
15天不用会清空。

<br>

### 文件准备

1.无卡模式开机
这个转态0.1r/h，没有连GPU, 但是可以保持连接状态传输文件。

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/20250526102154064.png" style="width: 90%;">
<br>

2.打开FileZilla，文件-站点管理器
<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/20250526103040730.png" style="width: 50%;">
<br>

3.每个新实例类似上图配置。
- 协议：SFTP
- 主机：region....com (@后面的东西)
- 用户：root
- 密码：复制粘贴
- 端口：ssh -p后面，root前面的数字

4. 连接以后，找到autodl-tmp文件夹，往里面拖拽需要的文件。
可能比较慢，等待一段时间。



## 连接使用

### Jupyter 

- 开机后直接点击JupyterLab,进入类似Jupyter Notebook的界面。
- autodl-tmp文件夹里有之前传上的数据
- 不被清空实例的情况下，关机后里面的东西都在，但记得及时下载。




<br>


### 本地VSCode

#### 连接

1.Ctrl+Shift+P → Remote-SSH:Connect to Host...

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/20250526112133880.png" style="width: 50%;">
<br>


2.选择已有的 或者 + Add New SSH Host

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/20250526112133880.png" style="width: 50%;">
<br>

输入完整账号，然后输入密码连接。

3.可能会开一个新窗口，选Linux,然后打开文件夹。


#### 上传

1. **文件传输**：
   - 使用AutoDL的文件上传功能，将整个项目文件夹上传到AutoDL实例中
   - 或者使用scp命令从本地上传,打开本地powershell：
```bash
scp -r /path/to/local/FileName username@autodl-instance-ip:/root/
```

2. **环境配置**：
```bash
# 创建并激活虚拟环境
conda create -n FileName python=3.9
conda activate FileName
   
# 安装依赖
pip install -r requirements.txt
```

3. **运行Jupyter Notebook**：
```bash
# 启动Jupyter Notebook服务
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

4. **访问Notebook**：
- 在AutoDL控制台找到实例的SSH端口（通常是22）
- 使用端口转发访问Jupyter Notebook：
```bash
ssh -L 8888:localhost:8888 username@autodl-instance-ip
```
- 在浏览器中访问：`http://localhost:8888`

5. **运行代码**：
- 在Jupyter Notebook中打开 `.ipynb`文件
- 按顺序运行所有单元格

6. **数据准备**：
- 确保数据文件已上传到正确位置
- 修改代码中的 `path` 相关变量指向数据文件所在目录

7. **注意事项**：
- 确保AutoDL实例有足够的GPU内存（建议至少8GB）
- 如果遇到内存不足，可以调整代码中的 `selected_regions` 数量
- 建议使用 `nvidia-smi` 命令监控GPU使用情况

8. **保存结果**：
- 代码会自动将结果保存在 `result_path` 相关路径指定的目录中
- 可以使用scp命令将结果下载到本地：

```bash
scp -r username@autodl-instance-ip:/root/FileName/results /path/to/local/directory
```