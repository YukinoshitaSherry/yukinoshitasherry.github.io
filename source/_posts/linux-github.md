非常好，你的目标是：

> **在 Windows 中用 GitHub Desktop + VSCode 开发，但因为文件名限制（比如 `|`、`:`），只能在 Linux 环境下 checkout 项目。**

解决方案是：

> **通过 WSL2 安装 Linux（如 Ubuntu），在 WSL2 中 clone 仓库和开发；然后通过 VSCode + GitHub Desktop 实现图形界面开发体验。**

---

## ✅ 一次性完整教程：用 WSL2 完整开发、使用 GitHub Desktop、克隆并开发非法文件名的仓库

---

## 🧩 Part 1：准备环境（只需设置一次）

### ✅ 步骤 1：安装 WSL2 + Ubuntu（如果已安装可跳过）

1. 打开 PowerShell（管理员身份），执行：

   ```powershell
   wsl --install
   ```

2. 安装完成后 **重启电脑**

3. 第一次打开 WSL（输入 `wsl`）时，它会自动安装 Ubuntu 并要求你设置用户名密码。设置即可。

   ```bash
   wsl
   ```

---

### ✅ 步骤 2：配置 WSL2 默认环境

```powershell
wsl --set-default-version 2
```

确认默认是 WSL2，不是旧版的 WSL1。

---

### ✅ 步骤 3：在 WSL 中安装 Git（Ubuntu 默认已装）

进入 WSL 后检查是否已有 git：

```bash
git --version
```

如果没装，执行：

```bash
sudo apt update
sudo apt install git -y
```

---

### ✅ 步骤 4：在 VSCode 安装插件 Remote - WSL

打开 VSCode：

1. 进入插件市场，搜索：`Remote - WSL`
2. 安装（由微软官方开发）

这样就可以从 VSCode 中直接打开 Linux（WSL）内的项目。

---

## 🧩 Part 2：在 WSL 中 clone 项目

### ✅ 步骤 5：在 Ubuntu WSL 中 clone 项目（不会再报错）

打开 WSL 终端，执行：

```bash
cd ~
mkdir -p GitProjects
cd GitProjects

git clone https://github.com/e9la/DConanInfoSearch.git
cd DConanInfoSearch
```

💡 此时不会再报错！即使文件名包含 `|` `:` 也能成功 clone 和 checkout！

---

### ✅ 步骤 6：用 VSCode 打开这个目录

在 Ubuntu WSL 终端中执行：

```bash
code .
```

💡 如果第一次执行会提示你安装 VSCode Server，确认即可。

现在你已经在 WSL 中使用 VSCode 开发这个项目 ✅

---

## 🧩 Part 3：让 GitHub Desktop 同步你 WSL 项目

GitHub Desktop 不支持直接操作 WSL，但可以通过软链接或远程方式进行同步。你有两种方案：

---

### ✅ 方法 A（推荐）：让 GitHub Desktop 用 WSL Git Repo

1. 在 GitHub Desktop 选择 **Add Local Repository**

2. 选路径：

   ```
   \\wsl$\Ubuntu\home\你的用户名\GitProjects\DConanInfoSearch
   ```

   这是 GitHub Desktop 识别的 WSL 路径。

3. 点击添加 ✅

💡 此时你能用 GitHub Desktop 看到完整的 WSL Git 项目、提交、推送、分支管理。

---

### ✅ 方法 B：将 WSL 项目复制到 Windows 项目区（仅部分文件合法）

> ⚠️ **此方法会损失非法文件**，一般只用于导出文件，不推荐作为开发主路径。

---

## ✅ 最终结果

| 工具             | 路径                                             | 状态        |
| -------------- | ---------------------------------------------- | --------- |
| VSCode         | Remote WSL 打开 `~/GitProjects/DConanInfoSearch` | ✅ 全功能编辑开发 |
| GitHub Desktop | 添加路径 `\\wsl$\Ubuntu\home\你的用户名\...`            | ✅ 可用图形化操作 |
| Git CLI        | 在 WSL 中运行 Git 提交推送                             | ✅ 完全兼容    |

---

## 🔧 可选增强：配置 Git 账户信息（推荐）

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 🎁 Bonus：文件共享说明

* Windows 访问 WSL 项目目录：

  ```
  \\wsl$\Ubuntu\home\你的用户名\GitProjects
  ```

* WSL 访问 Windows 文件目录：

  ```bash
  cd /mnt/c/Users/你的用户名/Downloads
  ```

---

## ✅ 总结

你现在可以：

✅ 成功 clone 包含非法文件名的 GitHub 项目
✅ 使用 VSCode 全功能开发
✅ 用 GitHub Desktop 可视化提交
✅ 无需更改任何文件名！

---

如果你想我帮你写一个 `.bashrc` 自动打开路径、Git 别名或远程同步脚本，也可以说一声。是否需要我继续帮你优化 WSL 使用体验？
