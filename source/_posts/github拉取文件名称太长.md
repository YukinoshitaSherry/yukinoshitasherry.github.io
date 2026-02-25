---
title: GitHub上传超时问题
date: 2026-02-25
categories:
- 学CS/SE
tags:
- Git&Github
desc: Git Clone 后 Checkout 失败（Filename too long）问题排查与解决指南。如果 clone 成功但 checkout 失败，优先怀疑路径长度问题，而不是网络或权限问题。
---

【Git Clone 后 Checkout 失败（Filename too long）问题排查与解决指南。如果 clone 成功但 checkout 失败，优先怀疑路径长度问题，而不是网络或权限问题。】

### 问题现象

在 Windows 环境下执行 `git clone` 后出现类似报错：

```text
Filename too long
fatal: cannot create directory
warning: Clone succeeded, but checkout failed.
```

说明：

- 仓库对象已下载完成（clone 成功）
- 但工作区文件无法写入（checkout 失败）
- 根本原因是 Windows 默认路径长度限制（MAX_PATH = 260）

<br>

### 问题原理

Windows 传统 Win32 API 默认最大路径长度为：260 characters


而部分仓库（尤其包含以下内容）会产生极长路径：

- Python venv / conda env
- node_modules
- 深层嵌套日志目录
- 自动生成的缓存文件

示例路径结构：

```text

repo/agent/sessions/skills/xlsx/env/lib/python3.12/site-packages/numpy/.../src/common/pythoncapi-compat

````

路径层级叠加后超过限制，Git checkout 创建文件时失败。



该问题本质是：

Windows 文件系统限制 ≠ Git 限制

Git 支持长路径，但 Windows 默认不支持。

只要：

* Windows 长路径启用
* Git longpaths 开启

即可永久解决。

<br>

### 标准解决方案

#### 1. 启用 Windows 长路径支持

管理员权限 PowerShell：

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
````

然后重启系统。

验证是否开启：

```powershell
reg query HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled
```

返回：

```powershell
LongPathsEnabled    REG_DWORD    0x1
```



#### 2. 配置 Git 支持长路径

管理员终端执行：

```bash
git config --system core.longpaths true
```

或仅当前用户：

```bash
git config --global core.longpaths true
```

验证：

```bash
git config --show-origin core.longpaths
```



#### 3. 恢复 checkout

如果仓库已经 clone 但 checkout 失败：

```bash
cd <repo>
git restore --source=HEAD :/
```

或

```bash
git checkout .
```



### 进阶方案
（大型仓库推荐）

#### 1. 浅克隆（避免历史文件）

```bash
git clone --depth 1 <repo>
```

适用于：

* 只需要最新代码
* 不关心历史 commit

<br>

#### 2. 部分克隆（Partial Clone）

```bash
git clone --filter=blob:none <repo>
```

特点：

* 不下载大文件
* 按需拉取内容

<br>

#### 3. 稀疏检出（Sparse Checkout）

只拉指定目录：

```bash
git clone --filter=blob:none --sparse <repo>
cd <repo>
git sparse-checkout init --cone
git sparse-checkout set trajectories
```

适用于：

* 数据仓库
* 多模块仓库
* 只需某子目录

<br>

### 诊断命令

查看当前失败状态：

```bash
git status
```

查看最长路径文件：

```bash
git ls-tree -r HEAD --name-only | awk '{ print length, $0 }' | sort -n | tail
```

Windows PowerShell 版本：

```powershell
git ls-tree -r HEAD --name-only | % { $_.Length.ToString().PadLeft(5) + " " + $_ } | sort
```

<br>


### 快速临时解决方法（无需修改系统）

如果只是临时使用仓库：

直接 clone 到极短路径：

```bash
git clone <repo> C:\r
```

或

```bash
git clone <repo> D:\r
```

路径越短，可用字符越多。

<br>

### 仓库维护建议

（供仓库作者参考）

如果你是仓库维护者，应避免提交：

* venv/
* node_modules/
* **pycache**/
* logs/
* checkpoints/

推荐 `.gitignore`：

```
venv/
env/
node_modules/
*.log
__pycache__/
.cache/
```

<br>




