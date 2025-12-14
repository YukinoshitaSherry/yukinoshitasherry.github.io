---
title: Git&Github相关知识
date: 2023-04-02
categories:
- 学CS/SE
tags:
- Git&Github
desc: ZJU朋辈辅学技能拾遗笔记，主要参考了授课人鹤翔万里的ppt
---

## 介绍与用法

参考ZJU鹤翔万里学长的朋辈辅学：<a href="https://www.bilibili.com/video/BV1og4y1u7XU/?vd_source=fa4dcf78649ce6604c2727b4c64e76dc">朋辈辅学-b站视频</a>

<br>
{% pdf /pdf/Git&GitHub.pdf %}
<br>

## 提交 log 写法

写 GitHub 提交 log 遵循以下原则：
简洁性 ：保持提交 log 简洁明了，通常不超过 50 个字符。
描述性 ：清楚地描述所做更改的内容和目的。
语法正确 ：使用正确的语法和拼写，避免语法错误和拼写错误。
一致性 ：在整个项目中保持提交 log 格式的统一。

### feat
feat: 新功能
表示添加了一个新功能。例如：“feat (user module): add user authentication feature”（在用户模块中添加用户认证功能）

### fix
fix: 修复问题
一般用于修复 bug。例如：“fix (login page): resolve login button click issue”（修复登录页面登录按钮点击问题）

### docks
docs: 文档更新
用于修改文档相关的文件。例如：“docs: update README with setup instructions”（更新 README 添加设置说明）


### style
style: 代码风格调整
主要指对代码格式、缩进、空格等不影响功能的修改。例如：“style (user.js): fix code indentation”（在 user.js 中修正代码缩进）

### refactor
refactor: 代码重构
改进代码结构和性能，没有添加新功能或修复 bug。例如：“refactor (app.js): optimize code structure for better performance”（重构 app.js 优化代码结构以提升性能）

### test
test: 测试相关
添加或修改测试代码。例如：“test (user service): add unit tests for user registration”（为用户服务添加用户注册单元测试）

### chore
chore: 其他更改
通常指日常维护任务，如更新依赖或构建工具。例如：“chore (dependencies): update npm packages”（更新 npm 依赖包）

<br>

## 仓库关联本地文件夹



### 核心概念
- **本地仓库 (Local Repository)**：电脑上的项目文件夹，通过 `git init` 初始化后，其变化可由Git管理。
- **远程仓库 (Remote Repository)**：在GitHub（或其它平台）上创建的在线存储空间，用于备份和共享代码。
- **关联 (Link)**：将本地仓库与远程仓库建立对应关系，使本地代码能推送 (`push`) 到远程，或拉取 (`pull`) 远程更新。
- **首次提交 (First Commit)**：必须先将本地文件提交到本地仓库，生成提交记录，才能推送到远程。

### 准备工作
1. **安装Git**：前往 [Git官网](https://git-scm.com/) 下载并安装。
2. **注册GitHub账号**。
3. **在GitHub上创建一个新的空仓库**（New Repository）：
   - 记下仓库的HTTPS或SSH地址（如 `https://github.com/你的用户名/仓库名.git`）。
   - **关键**：创建时**不要勾选** “Add a README file”、“.gitignore”或“License”，以确保仓库完全为空。

### 方法一：使用Git命令行


在**你的本地项目文件夹中**打开终端（Git Bash、PowerShell等）。
随便哪里打开再cd到目标项目文件夹路径也行。

1. **初始化本地Git仓库**
   ```bash
   git init
   ```
   **作用**：在当前文件夹创建隐藏的 `.git` 子目录，使其成为Git可管理的仓库。

2. **将远程仓库添加为“源” (origin)**
   ```bash
   git remote add origin <你的GitHub仓库URL>
   ```
   **示例**：
   ```bash
   git remote add origin https://github.com/YukinoshitaSherry/obsidian_vaults.git
   ```
   **作用**：为远程仓库起一个别名（通常叫 `origin`），方便后续引用。

3. **检查并重命名本地主分支（如有必要）**
   ```bash
   git branch -M main
   ```
   **作用**：将本地默认分支名改为 `main`，与GitHub的现代默认设置保持一致。如果你的本地分支已叫 `main`，此命令安全无害。

4. **将本地所有文件添加到暂存区**
   ```bash
   git add .
   ```
   **作用**：将文件夹内所有**新文件和修改过的文件**标记为待提交状态。使用前建议用 `git status` 命令预览哪些文件将被添加。

5. **创建首次提交**
   ```bash
   git commit -m "Initial commit"
   ```
   **作用**：将暂存区的内容正式保存到本地仓库，并附上描述信息。引号内的信息可自定义，建议说明此次提交的目的。

6. **推送到远程GitHub仓库**
   ```bash
   git push -u origin main
   ```
   **作用**：
   - 将本地 `main` 分支的提交推送到远程 `origin` 仓库。
   - `-u` 参数建立追踪关联，之后在该分支只需使用 `git push` 即可推送。

7. **验证与后续操作::
- **验证**：刷新你的GitHub仓库页面，应能看到所有文件。
- **查看远程关联**：`git remote -v`
- **日常同步**：
  ```bash
  git add .                    # 添加更改
  git commit -m "更新描述"     # 提交更改
  git push                     # 推送至远程
  ```

<br>

### 方法二：使用GitHub Desktop图形化操作

通过图形界面（GitHub Desktop）来管理这个仓库，可以按以下步骤操作。

1. **打开GitHub Desktop**。
2. 点击菜单栏 `File` -> `Add local repository...`（或初始界面上的相同选项）。
3. 在弹出的窗口中：
   - **Local Path**：点击 `Choose...`，导航并选择你的本地项目文件夹（例如 `C:\Users\lenovo_yzy\Documents\Obsidian Vault`）。
   - 点击 `Add Repository`。
4. 此时，GitHub Desktop会识别到这是一个尚未与远程关联的本地仓库。在软件界面顶部，你会看到提示 **“Publish repository”** 或 **“Push origin”**。
5. 点击该提示按钮，或在菜单栏选择 `Repository` -> `Push`。
6. 在弹出的窗口中，确认远程仓库的地址（通常是之前通过命令行 `git remote add` 添加的地址），然后点击 `Push` 按钮。


关联成功后，未来使用GitHub Desktop同步变更将非常直观：
1. 打开GitHub Desktop，软件会自动检测到已更改的文件。
2. 在左侧面板勾选你想提交的文件，或在底部输入框勾选 `Select all` 选择全部。
3. 在右下角的 `Summary` 输入框填写本次提交的描述。
4. 点击下方的 `Commit to main` 按钮提交到本地仓库。
5. 提交后，点击窗口右上角的 `Push origin` 按钮，将本地提交推送到GitHub远程仓库。

<br>


