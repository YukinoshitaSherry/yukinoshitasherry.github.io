---
title: Mac电脑初学与装机记录
date: 2025-12-18
categories: 
    - 学CS/SE
tags: 
    - Mac
desc: 被重装搞怕后由win换mac，借机熟悉一下这个操作系统。直观感受：原来人除了一指禅以外是还可以用两根和三根手指操作触控板的啊。
---

型号：MacBook Pro A2442 16G/1T 14寸 

## mac相对win优缺点

### 优点

**系统体验：**
- **系统稳定性好** - macOS基于Unix，系统崩溃和蓝屏情况较少【笔者从0开始学用mac的根本原因。】
    - **系统更新及时** - 苹果统一推送更新，无需担心版本碎片化

**开发环境：**
- **Unix命令行** - 原生支持bash/zsh，开发环境友好
- **包管理方便** - Homebrew包管理器简单易用，类似Linux的apt，可实现一键安装
- **终端体验好** - Terminal和iTerm2功能强大
- **环境配置友好** - 相比Windows，macOS的文件管理和环境配置更接近Linux
- **文件系统统一** - 存储不分盘（不像Windows的C盘、D盘,到时候C盘炸了），文件管理更简洁


**硬件与生态：**
- **(14寸)硬件轻** - 便于牛马携带办公
- **续航久** - M系列芯片MacBook续航表现突出
- **静音运行** - 风扇噪音小，适合安静环境
- **Apple生态联动** - 与苹果全家桶协作
- **Retina显示屏** - 高分辨率屏幕显示清晰

**安全性：**
- **广告与病毒较少** - macOS病毒和恶意软件相对较少
- **权限管理严格** - 应用权限控制更细致
- **隐私保护** - 系统级隐私保护机制完善【但公司电脑有监控软件，毫无隐私。】

### 缺点

**价格与硬件：**
- **价格昂贵** - 同配置下价格通常高于Windows笔记本
- **硬件升级困难** - 内存和硬盘多为焊接，无法自行升级
- **维修成本高** - 官方维修费用昂贵

**软件兼容性：**
- **部分专业软件缺失** - 某些Windows专用软件无法使用，部分企业级软件仅支持Windows
- **软件价格较高** - Mac平台软件通常比Windows版本贵

**使用习惯：**
- **学习成本** - 从Windows转过来需要适应新的操作习惯
    - **右键菜单** - 需要双指点击，不如Windows直观【搞懂右键差异后笔者大彻大悟。】
    - **快捷键差异** - Cmd键位置与Windows的Ctrl不同
- **文件管理** - Finder功能不如Windows资源管理器丰富

<br>

## 操作

### 快捷键

#### 基础操作
和Win比就是把Ctrl换Cmd
- `Cmd + C` / `Cmd + V` / `Cmd + X` - 复制/粘贴/剪切
- `Cmd + Z` - 撤销
- `Cmd + Shift + Z` - 重做
- `Cmd + A` - 全选
- `Cmd + F` - 查找
- `Cmd + S` - 保存
- `Cmd + W` - 关闭当前窗口
- `Cmd + Q` - 退出应用程序
- `Cmd + P` - 打印
- `Cmd + Tab` - 切换应用程序
- `Cmd + ` (反引号) - 切换同一应用的不同窗口

#### 窗口管理
- `Cmd + M` - 最小化窗口
- `Cmd + H` - 隐藏当前应用
- `Cmd + Option + H` - 隐藏其他应用
- `Cmd + Control + F` - 全屏/退出全屏
- `Cmd + Option + Esc` - 强制退出应用
- `F11` / `F12` - 显示桌面（取决于设置）

#### Finder快捷键
- `Cmd + N` - 新建Finder窗口
- `Cmd + Shift + N` - 新建文件夹
- `Cmd + Delete` - 移到废纸篓
- `Cmd + Shift + Delete` - 清空废纸篓
- `Cmd + Option + Delete` - 立即清空废纸篓（不确认）
- `Cmd + I` - 显示简介
- `Cmd + D` - 复制文件
- `Cmd + Enter` - 重命名文件
- `Space` - 快速预览（Quick Look）
- `Cmd + Y` - 预览文件
- `Cmd + 1/2/3/4` - 切换视图模式（图标/列表/分栏/封面流）

#### 文本编辑
- `Cmd + B` / `Cmd + I` / `Cmd + U` - 加粗/斜体/下划线
- `Cmd + Left/Right` - 移动到行首/行尾
- `Cmd + Up/Down` - 移动到文档开头/结尾
- `Option + Left/Right` - 按词移动
- `Cmd + Shift + Left/Right` - 选择到行首/行尾
- `Option + Delete` - 删除前一个词
- `Cmd + Delete` - 删除到行首

#### 截图
- `Cmd + Shift + 3` - 全屏截图
- `Cmd + Shift + 4` - 选择区域截图
- `Cmd + Shift + 4 + Space` - 窗口截图
- `Cmd + Shift + 5` - 打开截图工具（macOS Mojave+）

#### 系统
- `Cmd + Space` - 打开Spotlight搜索
- `Cmd + Option + Space` - 打开Finder搜索
- `Control + Cmd + Space` - 打开表情符号面板
- `Cmd + Option + D` - 显示/隐藏Dock
- `Cmd + Control + Power` - 立即锁定屏幕
- `Cmd + Option + Power` - 睡眠
- `Control + Cmd + Q` - 锁定屏幕

#### 终端
- `Cmd + K` - 清屏
- `Cmd + T` - 新建标签页 或 `Cmd + blank` 搜索terminal后enter
- `Cmd + W` - 关闭标签页
- `Cmd + N` - 新建窗口
- `Cmd + D` - 垂直分割
- `Cmd + Shift + D` - 水平分割


<br>

### 手势

#### 设置位置
- **系统设置** → **触控板**（或 **系统偏好设置** → **触控板**）
- 可以查看和自定义各种手势操作
- 支持预览每个手势的效果

#### 常用手势

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/3b1c09a97b4475cadbf60199877f3cd3_720.jpg"/>

**单指操作：**
- **单击** - 点击/选择
- **双击** - 打开文件/应用
- **用力点按（Force Touch）** - 预览文件、查看定义等（支持Force Touch的Mac）

**双指操作：**
- **双指点击** - 右键菜单
    - 压缩成.zip: 双指右键-压缩
- **双指上下滑动** - 滚动页面
- **双指左右滑动** - 前进/后退（浏览器）
- **双指捏合/展开** - 缩放（网页、图片）
- **双指旋转** - 旋转图片
- **双指从右边缘向左滑动** - 显示通知中心
- **双指在触控板上左右滑动** - 切换全屏应用

**三指操作：**
- **三指左右滑动** - 切换桌面/全屏应用
- **三指向上滑动** - 打开调度中心（Mission Control）
- **三指向下滑动** - 显示当前应用的所有窗口
- **三指点击** - 查找/词典（可在设置中开启）

**四指操作：**
- **四指向上滑动** - 打开调度中心
- **四指向下滑动** - 显示当前应用的所有窗口
- **四指左右滑动** - 切换桌面/全屏应用
- **四指捏合** - 打开启动台（Launchpad）
- **四指展开** - 显示桌面

#### 自定义手势
在 **系统设置** → **触控板** 中：
- 可以开启/关闭特定手势
- 可以调整滚动方向（自然滚动）
- 可以调整点击力度（Force Touch）
- 可以设置辅助点按（右键）的方式

#### 手势技巧
- 如果手势不灵敏，可以在设置中调整触控板跟踪速度
- 某些手势需要先启用才能使用
- 不同macOS版本的手势可能略有差异




<br>
<br>

## 安装

### homebrew

#### 安装
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安装完成后，根据提示将Homebrew添加到PATH（Apple Silicon Mac）：
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

#### 常用命令
- `brew install <package>` - 安装软件包
- `brew uninstall <package>` - 卸载软件包
- `brew update` - 更新Homebrew
- `brew upgrade` - 升级所有已安装的软件包
- `brew upgrade <package>` - 升级指定软件包
- `brew list` - 列出已安装的软件包
- `brew search <keyword>` - 搜索软件包
- `brew info <package>` - 查看软件包信息
- `brew doctor` - 检查Homebrew环境
- `brew cleanup` - 清理旧版本和缓存
- `brew services list` - 查看服务列表
- `brew services start <service>` - 启动服务
- `brew services stop <service>` - 停止服务

#### 常用软件包
- `brew install git` - Git版本控制
- `brew install python` - Python
- `brew install wget` - 下载工具
- `brew install tree` - 目录树显示
- `brew install htop` - 系统监控
- `brew install --cask <app>` - 安装GUI应用（如：`brew install --cask google-chrome`）

<br>

### zsh

#### 检查版本
```bash
zsh --version
```

macOS Catalina及以后版本默认使用zsh，无需安装。

#### 切换默认shell
```bash
chsh -s /bin/zsh
```

#### 配置文件
- `~/.zshrc` - zsh配置文件（每次启动终端时加载）
- `~/.zprofile` - 登录时加载（适合设置PATH等环境变量）

#### 常用配置
编辑 `~/.zshrc` 文件：
```bash
# 设置别名
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# 设置PATH
export PATH="/usr/local/bin:$PATH"

# 历史记录配置
HISTFILE=~/.zsh_history
HISTSIZE=10000
SAVEHIST=10000
setopt SHARE_HISTORY  # 多个终端共享历史记录
```

#### Oh My Zsh（可选，推荐）
增强zsh功能的框架：

**安装：**
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

**常用插件：**
编辑 `~/.zshrc` 中的 `plugins` 行：
```bash
plugins=(
  git
  zsh-autosuggestions  # 自动建议（需安装）
  zsh-syntax-highlighting  # 语法高亮（需安装）
  zsh-history-substring-search  # 历史记录搜索（需安装）
)
```

**安装插件：**
```bash
# 方式1：使用git clone（推荐）
# 自动建议
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# 语法高亮
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# 历史记录搜索（根据历史记录推荐）
git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search

# 方式2：使用oh-my-zsh插件管理器（如果已安装）
# 注意：需要先安装zinit或其他插件管理器
```

**历史记录搜索插件配置：**
在 `~/.zshrc` 中添加（需在plugins配置之后）：
```bash
# 历史记录搜索绑定键
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down
# 或使用方向键
bindkey '↑' history-substring-search-up
bindkey '↓' history-substring-search-down
```

**应用配置：**
```bash
source ~/.zshrc
```

<br>

### miniconda

#### 安装
**方式1：使用Homebrew（推荐）**
```bash
brew install --cask miniconda
```

**方式2：手动安装**
```bash
# 下载安装包（Apple Silicon Mac）
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Intel Mac使用
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# 运行安装脚本
bash Miniconda3-latest-MacOSX-arm64.sh

# 按照提示完成安装，最后选择yes初始化conda
```

#### 初始化
**Homebrew安装方式：**
安装完成后，conda可能不会自动初始化，需要手动初始化：
```bash
# 初始化conda（首次运行）
conda init zsh

# 重启终端或执行
source ~/.zshrc
```

**手动安装方式：**
安装脚本会询问是否初始化，选择yes即可。如果没有初始化，可以手动执行：
```bash
# 初始化conda
conda init zsh

# 重启终端或执行
source ~/.zshrc
```

**验证安装：**
```bash
conda --version
which conda  # 查看conda路径
```

#### 常用命令
- `conda --version` - 查看版本
- `conda update conda` - 更新conda
- `conda create -n <env_name> python=3.9` - 创建虚拟环境
- `conda activate <env_name>` - 激活环境
- `conda deactivate` - 退出环境
- `conda env list` - 查看所有环境
- `conda remove -n <env_name> --all` - 删除环境
- `conda install <package>` - 安装包
- `conda list` - 查看当前环境已安装的包
- `conda search <package>` - 搜索包
- `conda update <package>` - 更新包

#### 配置国内镜像源（可选，加速下载）
```bash
# 添加清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

<br>

### jupyter

#### 安装
**方式1：使用Homebrew（推荐，Mac原生方式）**
```bash
# 安装JupyterLab
brew install --cask jupyterlab

# 或安装Jupyter Notebook
brew install jupyter
```

**方式2：使用miniconda**
```bash
# 激活miniconda环境
conda activate <env_name>

# 安装jupyter
conda install jupyter

# 或安装jupyterlab（更现代的界面）
conda install jupyterlab
```

**方式3：使用pip**
```bash
pip install jupyter
# 或
pip install jupyterlab
```

#### 启动
```bash
# 启动Jupyter Notebook
jupyter notebook

# 或启动JupyterLab
jupyter lab
```

启动后会自动在浏览器中打开，默认地址为 `http://localhost:8888`

#### 常用配置

**生成配置文件：**
```bash
jupyter notebook --generate-config
# 或
jupyter lab --generate-config
```

**设置密码（可选）：**
```bash
jupyter notebook password
# 或
jupyter lab password
```

**设置启动目录：**
编辑 `~/.jupyter/jupyter_notebook_config.py`：
```python
c.NotebookApp.notebook_dir = '/path/to/your/notebooks'
```

#### 常用扩展插件（JupyterLab）
```bash
# 安装扩展管理器
conda install -c conda-forge jupyterlab-git

# 或使用pip
pip install jupyterlab-git
```

#### 常用快捷键
- `Shift + Enter` - 运行当前单元格并移动到下一个
- `Ctrl + Enter` - 运行当前单元格
- `A` - 在上方插入单元格
- `B` - 在下方插入单元格
- `DD` - 删除单元格
- `M` - 将单元格转为Markdown
- `Y` - 将单元格转为代码
- `Cmd + S` - 保存

#### 内核管理
```bash
# 查看已安装的内核
jupyter kernelspec list

# 安装Python内核到指定环境
python -m ipykernel install --user --name <env_name> --display-name "Python (<env_name>)"

# 删除内核
jupyter kernelspec uninstall <kernel_name>
```

<br>

