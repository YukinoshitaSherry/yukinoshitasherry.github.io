---
title: Remote-SSH窗口关闭/网络断开解决方法
date: 2025-07-01
categories: 
    - 学CS/SE
tags: 
    - SSH
    - Linux
desc: nohup、screen、tmux、Ctrl+r使用指南、NVIDIA指令、VSCode误删文件找回方法
---


## nohup

**简介** ：`nohup` 是 "no hang up" 的缩写，用于确保命令在终端关闭后仍能继续运行。

**基本用法** ：

```bash
nohup python train.py > output.log 2>&1 &
```

- `nohup` ：使命令忽略终端挂起信号。
- `python train.py` ：待执行的程序。
- `> output.log` ：将标准输出重定向到 `output.log`。
- `2>&1` ：将标准错误重定向到标准输出。
- `&` ：将命令置于后台运行。

**查看进程** ：使用 `ps` 或 `jobs` 查看运行中的进程。

**注意事项** ：`nohup` 适合简单任务，终止进程时需手动查找 PID。

<br>

## tmux

**简介** ：`tmux` 是终端复用工具，允许用户在一个终端窗口中运行多个会话、窗口和窗格。

### 安装

```bash
# Ubuntu/Debian
sudo apt-get install tmux

# CentOS/RHEL
sudo yum install tmux

# macOS
brew install tmux
```

### 启动与连接

```bash
tmux new -s mysession    # 创建新会话 自己用：prompt
tmux attach -t mysession # 重新连接已存在的会话
```

### 基本操作

带tmux的都在tmux外输入，快捷键都在tmux内按。

- **分离会话** ：按 `Ctrl + b` 后输入 `d`，会话将继续在后台运行。
- **查看会话** ：`tmux ls` 列出所有活动会话。
- **关闭会话** ：`tmux kill-session -t mysession` 彻底关闭会话及其内部命令。
- **查看历史会话**：按 `Ctrl + b` 后输入 `[`，按方向键查看历史记录，想不看了按`q`。在tmux内，输入指令时无法看历史记录，看历史记录时不能输入指令。

### 窗口与窗格管理

- **列出所有窗口**：`tmux list-windows`或`tmux ls-w`。
- **创建新窗口** ：按 `Ctrl + b` 后输入 `c`。
- **查看当前会话所有窗口**：`Ctrl + b` 然后 按 `w`。有10个以上窗口则用这个切换，按方向键选择，然后回车进入对应窗口。
- **切换窗口** ：切换到窗口2 `tmux select-window -t 2`，或`Ctrl + b` 然后 按 窗口编号 (0~9)。

**优势** ：适合复杂实验环境，支持多任务并行处理，可随时恢复会话。

<br>

## screen

**简介** ：`screen` 是终端复用工具，可在一个终端窗口里开多个虚拟终端。

### 基本用法

```bash
screen -S session_name    # 创建会话
screen -ls                # 查看所有会话
screen -r session_name    # 重新连接会话
```

- **分离会话** ：按 `Ctrl + a` 后输入 `d`。
- **关闭会话** ：输入 `exit`。

### 优势场景

- **远程工作不断线** ：断网后可重新连接，任务继续运行。
- **多任务并行管理** ：一个窗口开多个会话，轻松切换。
- **后台运行任务** ：关闭终端后任务仍继续执行。

### 三者对比

| 工具     | 优势                                       | 劣势                                       |
|----------|--------------------------------------------|--------------------------------------------|
| nohup    | 简单易用，适合短期任务                     | 无法实时查看输出，终止进程需查找 PID       |
| tmux     | 支持多会话、多窗口、多窗格，可实时查看     | 学习曲线较陡                               |
| screen   | 功能类似 tmux，兼容性好                   | 使用场景相对 tmux 较少                     |

总结:
- **短期任务** ：使用 `nohup` 快速启动并后台运行。
- **复杂多任务** ：使用 `tmux` 创建多会话、多窗口进行管理。
- **兼容性需求** ：`screen` 是不错的选择。

根据具体需求选择合适的工具，可有效提升远程实验的效率与便捷性。

<br>

## Ctrl+R 历史命令搜索

在终端中按 `Ctrl + R` 可以进入反向搜索模式，用于快速查找之前输入的命令。

**使用方法** ：

- **启动搜索** ：按 `Ctrl + R`，终端会显示 `(reverse-i-search):` 提示符。
- **输入关键词** ：输入命令的部分内容，终端会实时匹配历史命令。
- **选择命令** ：使用 `Ctrl + R` 继续向前搜索，`Ctrl + S` 向后搜索（需要在终端中启用）。
- **执行命令** ：找到目标命令后，按 `Enter` 执行，或按 `Esc` 取消搜索并编辑命令。

**优势** ：快速查找和重复执行历史命令，提高工作效率。

<br>

## NVIDIA 相关指令

**GPU 状态查看** ：

```bash
nvidia-smi                    # 查看GPU使用情况
watch -n 1 nvidia-smi         # 每秒刷新GPU状态
nvidia-smi -l 1               # 每秒刷新GPU状态（另一种方式）
```

**GPU 进程管理** ：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv  # 查看GPU进程
kill -9 <PID>                 # 终止占用GPU的进程
```

**CUDA 版本查看** ：

```bash
nvcc --version                # 查看CUDA编译器版本
cat /usr/local/cuda/version.txt  # 查看CUDA版本（如果存在）
```

**常用监控命令** ：

```bash
nvidia-smi dmon               # 监控GPU性能指标
nvidia-smi topo -m            # 查看GPU拓扑结构
```

**高级监控工具** ：

**nvitop** - 交互式GPU监控工具（推荐）

```bash
# 安装
pip install nvitop

# 使用
nvitop                      # 交互式监控界面，类似 htop
nvitop -m                   # 监控模式，显示所有GPU
```

**nvtop** - 类似 htop 的GPU监控工具

```bash
# Ubuntu/Debian 安装
sudo apt install nvtop

# 或从源码编译安装
git clone https://github.com/Syllo/nvtop.git
mkdir -p nvtop/build && cd nvtop/build
cmake .. -DNVML_RETRIEVE_HEADER_ONLINE=yes
make
sudo make install

# 使用
nvtop                       # 交互式监控界面
```

**watch 命令** - 定期执行并显示命令输出

```bash
watch -n 1 nvidia-smi       # 每秒刷新显示 nvidia-smi 输出
watch -n 2 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv'  # 自定义查询，每2秒刷新
```

**对比** ：

- **nvidia-smi** ：基础工具，所有系统都有，功能简单。
- **watch + nvidia-smi** ：适合简单监控需求，无需额外安装。
- **nvitop** ：Python工具，安装方便，界面友好，功能丰富，推荐使用。
- **nvtop** ：类似 htop，需要编译安装，功能强大。

<br>

## VSCode 误删文件找回

### 方法一：使用 Git 恢复

适用于已版本控制的文件：

```bash
# 查看删除的文件
git status

# 恢复特定文件
git checkout -- <file_path>

# 恢复所有删除的文件
git checkout -- .
```

### 方法二：使用 VSCode 时间线功能

1. 在 VSCode 中右键点击被删除文件所在的文件夹。
2. 选择 "Open Timeline"（打开时间线）。
3. 在时间线中找到文件的历史版本。
4. 点击历史版本，选择 "Restore"（恢复）。

### 方法三：从本地历史记录恢复

VSCode 会在本地保存文件历史：

1. 右键点击被删除文件所在的文件夹。
2. 选择 "Open Folder in File Explorer"。
3. 在 `.vscode` 目录下查找本地历史记录（如果有配置）。
4. 或使用 `Ctrl + Shift + P`，输入 "Local History" 查找相关功能。

### 方法四：从系统回收站恢复

如果文件系统支持，检查系统回收站：

- **Linux** ：检查 `~/.local/share/Trash/` 目录。
- **Windows** ：检查回收站。
- **macOS** ：检查废纸篓。

### 预防措施

- 使用 Git 进行版本控制。
- 定期提交代码。
- 在 VSCode 中启用自动保存和本地历史记录功能。
<br>

