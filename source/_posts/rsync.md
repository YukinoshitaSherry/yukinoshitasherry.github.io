---
title: Remote-SSH跨服务器传输与挂载、环境迁移
date: 2025-07-02
categories: 
    - 学CS/SE
tags: 
    - SSH
    - Linux
desc: Remote-SSH 跨服务器免密rsync传输、scp传输、conda环境迁移、sshfs挂载、unar解压使用指南
---

# 免密登录（SSH Key Authentication）

## 生成 SSH 密钥对

```bash
ssh-keygen -t ed25519 -C "your-key-comment"
# 或使用 RSA：
ssh-keygen -t rsa -b 4096 -C "your-key-comment"
```

* 默认保存至 `~/.ssh/id_ed25519` 和 `~/.ssh/id_ed25519.pub`
* 推荐添加 passphrase（增强安全性）并使用 `ssh-agent` 缓存解密密码 ([youtube.com][1], [brandonchecketts.com][2])

## 分发公钥（多种方法）

| 方法              | 描述                                                                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ssh-copy-id** | 最简单：`ssh-copy-id user@host`，首次需要输入一次远程密码                                                                                                                            |
| **手动复制粘贴**      | 在本地执行 `cat ~/.ssh/id_ed25519.pub`，复制公钥内容，登录远程后粘贴到 `~/.ssh/authorized_keys`                                                                                          |
| **管道+ssh**      | 一行命令完成：<br>`cat ~/.ssh/id_ed25519.pub \\| ssh user@host "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"` |

* 注意追加方式，不要覆盖已有公钥
* 粘贴时确保没有换行或空格乱序，权限为 `.ssh` 700、`authorized_keys` 600

## 确保 sshd 配置支持公钥登录

编辑 `/etc/ssh/sshd_config`，确保：

```text
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
# Security hardening
PasswordAuthentication no
ChallengeResponseAuthentication no
```

设置后重启服务：

```bash
sudo systemctl restart sshd
```

网络建议禁止 root 登录，改用普通用户登录并通过 `sudo` 提权 ([digitalocean.com][3], [secopsolution.com][4])。

## 多服务器互通免密

* 本地 ➜ A、B：把公钥复制到 A(/home/yzy/.ssh/) 和 B(/root/.ssh/) 的 `authorized_keys` 即可
* A ➜ B：在 A 上生成 key，复制 A 的公钥到 B 的 `/root/.ssh/authorized_keys`
* B ➜ A：同理

登录命令需要**用户名一致**：如 `ssh yzy@A` 或 `ssh root@B`，并根据 `-v` 输出确认公钥被尝试使用

### 调试建议

```bash
ssh -v user@host
# 看是否成功尝试 id_ed25519，以及失败原因（权限、公钥不匹配等）
```

## 安全最佳实践

* 每人/设备用独立密钥对，定期（如每 2 年）轮换&#x20;
* 密钥加密保护 + `ssh-agent` 使用 ([brandonchecketts.com][2])
* 避免 root 远程登录，设 least privilege，启用 MFA/2FA&#x20;
* 配合 jump hosts 管理免密连接 ([reddit.com][5])


<br>

# rsync 用法汇总

`rsync` 是用于高效同步文件的强力工具 。

## 常见用法

* `-v`: verbose 输出
* `-a`: archive 模式（递归 + 保留权限）
* `-z`: 压缩
* `-P`: `--partial` + `--progress`（常用组合） ([my.cloudfanatic.net][6], [reddit.com][7])
* `--delete`: 目标中删除源不存在文件
* `-e "ssh -p PORT"`: 指定 SSH shell 和端口
* `--exclude=PATTERN`, `--include=PATTERN`: 过滤文件

## 本地与远程同步

| 场景             | 命令示例                                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| 本地 ➜ 本地        | `rsync -avh src/ dest/`                                                                                  |
| 本地 ➜ 远程主机      | `rsync -avzhe ssh local_dir/ user@host:/remote/path/` ([my.cloudfanatic.net][6], [geeksforgeeks.org][8]) |
| 远程 ➜ 本地        | `rsync -avzhe ssh user@host:/remote/path/ local_dir/`                                                    |
| 远程A ➜ 本地 ➜ 远程B | 效率高、保证两端免密登录                                                                                             |
| 远程A ➜ 远程B 直接跳转 | `ssh A "rsync -avz /src/ root@B:/dst/"` 或使用 `-J jump` 配合跳板                                               |

## 使用注意

* 数据量大时增量同步，节省带宽&#x20;
* 末尾 `/` 表示“目录内容”，不加表示复制目录本身&#x20;
* 可以设置 `--bwlimit`, `--dry-run`, `--checksum` 等高级选项 ([tecmint.com][9])

## 实战示例

1. 从 A ➜ B（在 A 上执行）：

   ```bash
   rsync -avz --progress /disk/.../data/ root@B:/data/dataset/
   ```

2. 从 B ➜ A（在 B 上执行）：

   ```bash
   rsync -avz --progress -e "ssh -p 1122" yzy@A:/disk/.../data/ /data/dataset/
   ```

3. 排除 `.git` 和 `.tmp` 文件：

   ```bash
   rsync -avz --progress --exclude='.git/' --exclude='*.tmp' src/ dest/
   ```

4. 使用跳板 host：

   ```bash
   rsync -avz -e "ssh -J jumpuser@jumphost" src/ user@target:/path/
   ```

<br>

# SCP

`scp` 是基于 SSH 的简单安全文件拷贝命令，适合快速传输文件/目录。

## 常用选项说明

* `-r`: 递归复制目录
* `-P PORT`: 指定 SSH 端口（注意是大写 P）
* `-v`: 显示调试信息
* `-C`: 启用压缩传输
* `-i identity_file`: 指定 SSH 私钥文件

## 常见使用场景

| 场景            | 命令示例                                                           |
| ------------- | -------------------------------------------------------------- |
| 本地 ➜ 远程       | `scp file.txt user@host:/remote/path/`                         |
| 远程 ➜ 本地       | `scp user@host:/remote/file.txt ./local_dir/`                  |
| 复制目录（递归）      | `scp -r ./mydir user@host:/remote/path/`                       |
| 指定端口          | `scp -P 2222 file.txt user@host:/remote/path/`                 |
| 使用密钥          | `scp -i ~/.ssh/id_ed25519 file.txt user@host:/remote/path/`    |
| A ➜ 本地 ➜ B 中转 | `scp userA@hostA:/path/file userB@hostB:/target/path/`（需要跳板支持） |

## 注意事项

* 相比 `rsync`，`scp` 不支持断点续传，不适合大文件或不稳定网络
* 权限与时间戳默认不保留，适合简单传输场景
* 推荐搭配 `-C` 压缩加速传输
* 远程地址格式固定：`user@host:/path/to/file`

## 实战示例

1. 将文件上传到远程服务器：

   ```bash
   scp ./model.pt root@192.168.1.10:/data/checkpoints/
   ```

2. 下载整个文件夹：

   ```bash
   scp -r root@192.168.1.10:/data/logs/ ./local_logs/
   ```

3. 上传大文件，使用密钥与端口：

   ```bash
   scp -P 2222 -i ~/.ssh/id_ed25519 -C ./large.tar.gz user@host:/backup/
   ```



<br>

# Conda 环境迁移

## 概述

Conda 是跨平台的包管理和环境管理工具，可管理 Python 包及依赖，支持独立环境隔离。  
环境迁移用于在不同机器或服务器间复制环境，保证开发/实验环境一致性。

迁移场景举例：
- 开发机器 → 服务器（Linux）
- Windows → Linux/macOS
- 备份/共享项目环境


## 步骤
### 导出环境

假设当前环境名为 `myenv`：

```bash
# 激活环境
conda activate myenv

# 导出环境到 YAML 文件
conda env export > myenv.yml
```

跨平台兼容建议

```bash
# 不导出 build 信息，减少依赖冲突
conda env export --no-builds > myenv.yml
```

> tips：
>
> * YAML 文件包含 conda 包、pip 包及 Python 版本。
> * 若环境中有 pip 包，YAML 文件会自动生成 `- pip:` 部分。


### 复制环境文件

将 `myenv.yml` 拷贝到目标机器：

```bash
# Linux/macOS
scp myenv.yml user@target_server:/path/to/destination/

# Windows (PowerShell)
scp myenv.yml user@target_server:C:\path\to\destination\
```

> 注意：
>
> * 确保目标路径有写权限。
> * 若网络不通，可使用 U 盘或网盘传输。


### 在目标机器上创建环境

```bash
# 使用 YAML 创建环境
conda env create -f myenv.yml

# 或指定新环境名
conda env create -f myenv.yml -n newenvname

# 查看环境列表
conda env list
```

> 建议：
>
> * 创建完成后，先激活环境测试基础包：

```bash
conda activate myenv
python -c "import numpy, pandas; print('Packages OK')"
```


### 激活环境

```bash
conda activate myenv   # 或 newenvname
```

> tip：在 Linux/macOS 上可加 `conda init bash/zsh` 确保 shell 自动识别 `conda activate`。



### 仅迁移包列表
（可选，适合快速迁移）

```bash
# 导出包列表
conda list --export > package-list.txt

# 创建环境
conda create --name newenv --file package-list.txt
```

> 注意：
>
> * 只适合同一操作系统。
> * 不保留 pip 包信息和 Python 版本。

---

### pip 包处理

YAML 文件中 pip 部分示例：

```yaml
- pip:
  - package1==1.0.0
  - package2>=2.0.0
```

在目标机器：

```bash
# 安装 pip 包
pip install -r requirements.txt
```

> tip：
>
> * 如果 YAML 丢失 pip 部分，可在源环境执行：

```bash
pip freeze > requirements.txt
```

然后在目标机器安装。





## 常见问题及解决方案

| 问题        | 解决方法                                                              |
| --------- | ----------------------------------------------------------------- |
| 平台不兼容     | 使用 `--no-builds` 导出，必要时手动调整依赖                                     |
| pip 包缺失   | 使用 `pip freeze > requirements.txt` 重新安装                           |
| 依赖冲突      | 先创建空环境：`conda create -n newenv python=3.x`，再安装包                   |
| 环境名重复     | 用 `-n newenvname` 指定新名字                                           |
| YAML 文件过大 | 可删除不必要的包，或只导出必要依赖：`conda env export --from-history > minimal.yml` |


## 高级技巧

### 一条命令迁移环境（跨平台示例）

假设两台 Linux 机器可通过 SSH 访问：

```bash
conda activate myenv \
&& conda env export --no-builds > myenv.yml \
&& scp myenv.yml user@target_server:/tmp/ \
&& ssh user@target_server "conda env create -f /tmp/myenv.yml"
```

> Windows 用户：
>
> * 使用 PowerShell 或 Git Bash 调整 scp/ssh 路径。
> * 建议将路径使用 `/` 或 `\\` 双斜杠。


### 跨操作系统迁移注意事项

1. **去掉 build 信息**：

```bash
conda env export --no-builds > myenv.yml
```

2. **确保 Python 版本一致**：

```yaml
dependencies:
  - python=3.10
```

3. **手动处理特殊包**：

   * 系统依赖不同（如 `opencv`, `pycairo`）可能需手动安装：

```bash
conda install -c conda-forge opencv
```

4. **Windows ↔ Linux/macOS**：

   * 有些包可能不支持目标平台，需要替换等效包或用 pip 安装。




## 总结

流程图

```
源环境 (myenv)
    │
    ├─ conda env export --no-builds → myenv.yml
    │
    ├─ scp / U盘 / 网盘 → 目标机器
    │
目标环境
    │
    ├─ conda env create -f myenv.yml -n newenvname
    │
    └─ conda activate newenvname → 测试运行
```


* Conda 环境迁移核心是 YAML 导出/导入。
* 跨平台迁移推荐去掉 build 信息，保证 Python 版本一致。
* pip 包需单独处理，确保目标机器可用。
* 遇到依赖冲突可先创建空环境，再逐步安装包。
* 一条命令迁移可大幅提高效率。
* 最佳实践：最小化环境 + 测试运行。



<br>

# SSHFS 挂载

## 概述

SSHFS（SSH Filesystem）是基于 SSH 协议的文件系统，允许将远程服务器的目录挂载到本地，实现透明访问远程文件。相比传统的文件传输工具，SSHFS 提供了更直观的文件操作体验。

## 安装 SSHFS

### Linux/macOS

```bash
# Ubuntu/Debian
sudo apt-get install sshfs

# CentOS/RHEL/Fedora
sudo yum install fuse-sshfs
# 或
sudo dnf install fuse-sshfs

# macOS (使用 Homebrew)
brew install sshfs
```

### Windows

Windows 用户可以使用以下工具：

* **WinFsp + SSHFS-Win**：安装 WinFsp 后下载 SSHFS-Win
* **WSL2**：在 WSL2 环境中使用 Linux 版本的 SSHFS
* **第三方工具**：如 ExpanDrive、Mountain Duck 等

## 基本用法

### 挂载远程目录

```bash
# 基本语法
sshfs user@host:/remote/path /local/mount/point

# 示例：挂载远程服务器的 /data 目录到本地的 /mnt/remote
sshfs user@192.168.1.10:/data /mnt/remote

# 指定端口
sshfs -p 2222 user@host:/remote/path /local/mount/point

# 使用密钥文件
sshfs -o IdentityFile=~/.ssh/id_ed25519 user@host:/remote/path /local/mount/point
```

### 常用挂载选项

| 选项                    | 描述                                                                 |
| --------------------- | ------------------------------------------------------------------ |
| `-o reconnect`        | 网络断开时自动重连                                                           |
| `-o ServerAliveInterval=60` | 每60秒发送保活信号，防止连接超时                                           |
| `-o compression=yes`  | 启用压缩传输                                                             |
| `-o cache=yes`        | 启用本地缓存，提高访问速度                                                   |
| `-o allow_other`      | 允许其他用户访问挂载点（需要 root 权限）                                        |
| `-o uid=1000,gid=1000` | 指定本地用户和组 ID，解决权限问题                                           |

### 完整挂载示例

```bash
# 创建挂载点
sudo mkdir -p /mnt/remote_server

# 挂载远程目录（推荐配置）
sshfs -o reconnect,ServerAliveInterval=60,compression=yes,cache=yes \
      user@192.168.1.10:/home/user/data /mnt/remote_server

# 验证挂载
df -h /mnt/remote_server
ls -la /mnt/remote_server
```

## 高级配置

### 自动挂载（fstab）

编辑 `/etc/fstab` 文件，添加自动挂载配置：

```bash
# 格式：sshfs#user@host:/remote/path /local/mount/point fuse _netdev,user,idmap=user,transform_symlinks,identity_file=/home/user/.ssh/id_ed25519,allow_other,default_permissions,reconnect,ServerAliveInterval=60,ServerAliveCountMax=3 0 0

sshfs#user@192.168.1.10:/data /mnt/remote_server fuse _netdev,user,idmap=user,transform_symlinks,identity_file=/home/user/.ssh/id_ed25519,allow_other,default_permissions,reconnect,ServerAliveInterval=60,ServerAliveCountMax=3 0 0
```

### 性能优化

```bash
# 启用缓存和压缩
sshfs -o cache=yes,compression=yes,large_read,big_writes \
      user@host:/remote/path /local/mount/point

# 调整缓存大小
sshfs -o cache=yes,cache_timeout=3600,attr_timeout=3600 \
      user@host:/remote/path /local/mount/point
```

## 卸载和故障排除

### 卸载挂载点

```bash
# 正常卸载
fusermount -u /mnt/remote_server
# 或
umount /mnt/remote_server

# 强制卸载（网络断开时）
sudo umount -f /mnt/remote_server
```

### 常见问题解决

| 问题                    | 解决方法                                                                 |
| --------------------- | -------------------------------------------------------------------- |
| 权限被拒绝               | 使用 `-o allow_other` 或调整 `uid/gid` 参数                                    |
| 连接超时                | 添加 `ServerAliveInterval` 和 `ServerAliveCountMax` 选项                    |
| 网络断开后无法重连           | 使用 `-o reconnect` 选项，或配置自动重连脚本                                      |
| 大文件传输慢              | 启用压缩 `-o compression=yes` 和缓存 `-o cache=yes`                          |
| 符号链接问题              | 使用 `-o transform_symlinks` 选项                                           |

### 调试命令

```bash
# 查看挂载状态
mount | grep sshfs

# 查看详细挂载信息
sshfs -o debug user@host:/remote/path /local/mount/point

# 检查网络连接
ssh -v user@host
```

## 实战示例

### 挂载远程数据集目录

```bash
# 挂载远程服务器的数据集目录
sshfs -o reconnect,ServerAliveInterval=60,compression=yes \
      user@research-server:/datasets /mnt/datasets

# 在 Python 中直接访问
import pandas as pd
df = pd.read_csv('/mnt/datasets/experiment_data.csv')
```

### 多服务器挂载

```bash
# 挂载多个远程服务器
sshfs user@server1:/data /mnt/server1_data
sshfs user@server2:/logs /mnt/server2_logs
sshfs user@server3:/backup /mnt/server3_backup

# 统一管理
mkdir -p /mnt/remote_data/{server1,server2,server3}
sshfs user@server1:/data /mnt/remote_data/server1
sshfs user@server2:/data /mnt/remote_data/server2
sshfs user@server3:/data /mnt/remote_data/server3
```

### 自动化脚本

```bash
#!/bin/bash
# mount_remote.sh

REMOTE_HOST="user@192.168.1.10"
REMOTE_PATH="/data"
LOCAL_MOUNT="/mnt/remote_data"

# 检查挂载点是否存在
if [ ! -d "$LOCAL_MOUNT" ]; then
    sudo mkdir -p "$LOCAL_MOUNT"
fi

# 挂载远程目录
sshfs -o reconnect,ServerAliveInterval=60,compression=yes,cache=yes \
      "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_MOUNT"

echo "Remote directory mounted at $LOCAL_MOUNT"
```

<br>

# unar 解压使用指南

## 概述

`unar` 是一个跨平台的解压缩工具，支持多种压缩格式，包括 ZIP、RAR、7z、TAR、GZIP 等。相比传统的解压工具，`unar` 具有更好的编码处理能力和更简洁的语法。

## 安装 unar

### Linux

```bash
# Ubuntu/Debian
sudo apt-get install unar

# CentOS/RHEL/Fedora
sudo yum install unar
# 或
sudo dnf install unar

# Arch Linux
sudo pacman -S unar
```

### macOS

```bash
# 使用 Homebrew
brew install unar

# 或使用 MacPorts
sudo port install unar
```

### Windows

* 从 [GitHub releases](https://github.com/TheUnarchiver/unar/releases) 下载预编译版本
* 或使用 Chocolatey：`choco install unar`

## 基本用法

### 解压文件

```bash
# 基本语法
unar archive_file

# 解压 ZIP 文件
unar data.zip

# 解压 RAR 文件
unar archive.rar

# 解压 7z 文件
unar data.7z

# 解压 TAR.GZ 文件
unar archive.tar.gz
```

### 指定输出目录

```bash
# 解压到指定目录
unar -o /path/to/output data.zip

# 解压到当前目录的子文件夹
unar -o ./extracted_data archive.rar
```

### 常用选项

| 选项        | 描述                           |
| --------- | ---------------------------- |
| `-o PATH` | 指定输出目录                       |
| `-f`      | 强制覆盖已存在的文件                  |
| `-q`      | 静默模式，不显示进度信息               |
| `-D`      | 不创建目录，直接解压到当前目录            |
| `-e`      | 使用文件扩展名作为输出目录名              |
| `-r`      | 递归解压子目录中的压缩文件              |

## 支持的格式

### 压缩格式

* **ZIP**：`.zip`
* **RAR**：`.rar`, `.r00`, `.r01` 等
* **7-Zip**：`.7z`
* **TAR**：`.tar`
* **GZIP**：`.gz`, `.tgz`
* **BZIP2**：`.bz2`, `.tbz2`
* **LZMA**：`.xz`, `.txz`
* **LZIP**：`.lz`
* **LZOP**：`.lzo`

### 编码支持

`unar` 的一大优势是自动检测和处理各种字符编码：

```bash
# 自动检测编码（默认）
unar chinese_files.zip

# 指定编码
unar -e GBK chinese_files.zip
unar -e UTF-8 japanese_files.rar
```

## 高级用法

### 批量解压

```bash
# 解压当前目录下所有 ZIP 文件
for file in *.zip; do
    unar "$file"
done

# 解压所有压缩文件到指定目录
for file in *.zip *.rar *.7z; do
    if [ -f "$file" ]; then
        unar -o ./extracted "$file"
    fi
done
```

### 处理损坏文件

```bash
# 尝试修复损坏的压缩文件
unar -f damaged.zip

# 跳过损坏的文件继续解压
unar -f -q archive.zip
```

### 查看压缩文件内容

```bash
# 列出压缩文件内容（不解压）
unar -l archive.zip

# 详细列表
unar -l -v archive.rar
```

## 与其他工具对比

### unar vs unzip

```bash
# unzip 基本用法
unzip archive.zip

# unar 基本用法
unar archive.zip

# unar 优势：更好的编码处理
unar chinese_files.zip  # 自动处理中文文件名
unzip chinese_files.zip # 可能显示乱码
```

### unar vs tar

```bash
# tar 解压
tar -xzf archive.tar.gz

# unar 解压
unar archive.tar.gz

# unar 优势：统一接口
unar archive.zip archive.rar archive.7z  # 相同命令处理不同格式
```

## 实战示例

### 解压数据集

```bash
# 解压大型数据集
unar -o ./datasets -q large_dataset.zip

# 检查解压结果
ls -la ./datasets/
du -sh ./datasets/
```

### 处理中文文件名

```bash
# 解压包含中文文件名的压缩包
unar -o ./chinese_data chinese_archive.zip

# 验证文件名正确性
ls -la ./chinese_data/
```

### 批量处理实验数据

```bash
#!/bin/bash
# batch_extract.sh

INPUT_DIR="./raw_data"
OUTPUT_DIR="./extracted_data"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 批量解压
for file in "$INPUT_DIR"/*.{zip,rar,7z}; do
    if [ -f "$file" ]; then
        echo "Extracting: $(basename "$file")"
        unar -o "$OUTPUT_DIR" -q "$file"
    fi
done

echo "Batch extraction completed!"
```

### 自动化脚本

```bash
#!/bin/bash
# auto_extract.sh

# 监控目录中的新压缩文件
WATCH_DIR="./downloads"
EXTRACT_DIR="./extracted"

inotifywait -m -e moved_to -e create "$WATCH_DIR" | while read path action file; do
    if [[ "$file" =~ \.(zip|rar|7z|tar\.gz)$ ]]; then
        echo "New archive detected: $file"
        unar -o "$EXTRACT_DIR" "$WATCH_DIR/$file"
    fi
done
```

## 故障排除

### 常见问题

| 问题                    | 解决方法                                                                 |
| --------------------- | -------------------------------------------------------------------- |
| 权限被拒绝               | 使用 `sudo unar` 或检查文件权限                                               |
| 编码问题                | 使用 `-e` 选项指定正确的编码（如 `-e GBK`, `-e UTF-8`）                        |
| 磁盘空间不足             | 检查可用空间：`df -h`，清理不需要的文件                                           |
| 文件损坏                | 使用 `-f` 选项强制解压，或尝试其他解压工具                                        |
| 文件名过长              | 使用 `-D` 选项不创建目录，或重命名文件                                            |

### 调试命令

```bash
# 查看 unar 版本和支持的格式
unar --version

# 详细输出模式
unar -v archive.zip

# 测试压缩文件完整性
unar -t archive.zip
```




