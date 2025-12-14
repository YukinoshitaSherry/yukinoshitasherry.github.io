---
title: GitHub上传超时问题
date: 2023-04-03
categories:
- 学CS/SE
tags:
- Git&Github
desc: GitHub在科学上网环境下上传大量文件容易出现超时问题，配置proxy/改https为ssh方法解决。
---

GitHub在科学上网环境下上传大量文件容易出现超时问题，配置proxy/改https为ssh方法解决。更推荐更为稳定的ssh方法。

<br>

## 方法一：配置Git使用代理



### 查找本地代理地址和端口

当Git或GitHub Desktop因网络超时失败时，配置代理通常是解决方案。首先，你需要找到正确的代理地址（格式通常为 `http://127.0.0.1:端口号` 或 `socks5://127.0.0.1:端口号`）。

1. 检查代理客户端
代理地址和端口由你**主动开启**的代理软件决定。

| 常见代理软件 | 默认/常见端口 | 查看方法 |
| :--- | :--- | :--- |
| **Clash** 及其衍生版 | **`7890`** (混合端口) | 1. 点击系统托盘猫图标。<br>2. 打开 **“Dashboard” (面板)** 或 **“设置”**。<br>3. 在 **“端口”** 或 **“General”** 设置中查找 **`Mixed Port`** 或 **`Port`**。 |
| **V2RayN** | **`10809`** (HTTP) | 1. 点击系统托盘V字图标。<br>2. 进入 **“参数设置”** > **“Core:基础设置”**。<br>3. 查看 **“本地监听端口”**（http端口）。 |
| **Shadowsocks** | **`1080`** (SOCKS5) | 查看任务栏纸飞机图标，右键通常可看到服务器和端口信息。 |
| **Qv2ray** | **`1088`** (HTTP) | 在主界面，查看 **“首选项”** > **“入站设置”** 中的HTTP监听端口。 |

**核心**：找到软件的 **“HTTP代理”端口**（或混合端口），地址通常是 `127.0.0.1` 或 `localhost`。

`本机的clash verge是7897端口`。


2. 检查操作系统网络设置
如果代理软件设置了系统代理，可以从这里查看:

- **Windows**：
    - 打开 **设置 > 网络和Internet > 代理**。
    - 在 **“手动设置代理”** 下查看地址和端口。
- **macOS**：
    - 打开 **系统设置 > 网络 > 选中网络 > 详细信息 > 代理**。
    - 查看 **“网页代理(HTTP)”** 的配置。

3. 命令行查询
打开终端（Git Bash/PowerShell/CMD）并运行：

**在 Git Bash 或 PowerShell 中：**
```bash
echo $http_proxy
echo $https_proxy
```
**在 CMD 中：**
```cmd
echo %http_proxy%
echo %https_proxy%
```
如果有设置，会返回类似 `http://127.0.0.1:7897` 的结果。


找到地址后，必须测试其是否能连通GitHub:
```bash
curl -x http://127.0.0.1:7897 https://github.com
```
- 如果返回大段HTML代码，**代理可用**。
- 如果报错（如 `Connection refused`），说明代理未运行、端口错误或配置有误。

### 配置代理
确认有效代理地址（例如 `http://127.0.0.1:7897`）后，为Git配置：

```bash
# 设置全局代理
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# 推送后验证
git push -u origin main
```

**完成后，建议取消代理，避免影响不需要代理的网络：**
```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```
<br>

## 方法二：改https为ssh

SSH方式通常比HTTPS更稳定，特别是在网络环境复杂的情况下。
将远程仓库地址从HTTPS改为SSH协议，可以避免代理配置的复杂性。
但是SSH比HTTPS慢。

### 检查当前远程地址

首先查看当前仓库使用的远程地址：

```bash
git remote -v
```

如果显示的是 `https://github.com/用户名/仓库名.git`，则需要改为SSH格式。

### 生成SSH密钥（如果还没有）

1. 检查是否已有SSH密钥：
```bash
ls -al ~/.ssh
```
如果看到 `id_rsa` 和 `id_rsa.pub`（或 `id_ed25519` 和 `id_ed25519.pub`），说明已有密钥，可跳过此步骤。

2. 生成新的SSH密钥：
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
按提示操作（可直接按回车使用默认路径和空密码）。

3. 查看公钥内容：
```bash
cat ~/.ssh/id_ed25519.pub
```
**复制全部输出内容**（以 `ssh-ed25519` 开头的一整行）。

### 添加SSH密钥到GitHub

1. 登录GitHub，进入 **Settings > SSH and GPG keys**。
2. 点击 **New SSH key**。
3. 填写 **Title**（如"我的电脑"），将复制的公钥内容粘贴到 **Key** 字段。
4. 点击 **Add SSH key** 保存。



### 从HTTPS改为SSH
将远程地址从HTTPS改为SSH：
```bash
# 查看当前远程地址
git remote -v

# 修改为SSH格式（替换为你的用户名和仓库名）
git remote set-url origin git@github.com:用户名/仓库名.git

# 验证修改
git remote -v
```


测试是否能通过SSH连接到GitHub：
```bash
ssh -T git@github.com
```

如果看到 `Hi 用户名! You've successfully authenticated...`，说明配置成功。


配置完成后，尝试推送：
```bash
git push -u origin main
```



<br>


## GitHub Desktop遇到网络超时的解决方法
GitHub Desktop的底层也是Git，因此其网络问题根源与命令行相同。以下是针对其图形界面的专用排障步骤。

### 在GitHub Desktop内配置代理
GitHub Desktop**没有内置的代理设置菜单**，但它会**继承系统代理设置**或**Git的全局配置**。
1.  **最佳方法**：按照第一部分，通过命令行**为Git配置全局代理**（`git config --global http.proxy ...`）。GitHub Desktop会自动使用此配置。
2.  **系统级代理**：确保你的代理软件已开启 **“设置系统代理”** 选项。GitHub Desktop会读取系统设置。

### 在GitHub Desktop中切换远程仓库协议
将远程仓库地址从HTTPS切换为更稳定的SSH协议，这是解决GitHub Desktop网络问题的根本方法。

1.  **生成并添加SSH密钥到GitHub**（如果未做过）：
    - 打开Git Bash，运行：`ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`，一路按回车。
    - 运行 `cat ~/.ssh/id_rsa.pub`，**复制全部输出内容**。
    - 登录GitHub，进入 **Settings > SSH and GPG keys > New SSH key**，粘贴并保存。
2.  **在GitHub Desktop中修改远程仓库URL**：
    - 打开你的仓库，点击顶部菜单 **Repository > Repository settings...**。
    - 在 **“Remote”** 区域，你会看到当前的远程仓库地址（`https://github.com/...`）。
    - 将其替换为对应的SSH地址：`git@github.com:YukinoshitaSherry/obsidian_vaults.git`。
    - 点击 **Save**。
3.  **重新尝试推送**（`Push origin`）。

<br>

## 其他通用排查步骤
- **关闭并重启GitHub Desktop**：修改代理或Git配置后，重启软件使之生效。
- **检查防火墙/安全软件**：暂时禁用可能阻止GitHub Desktop连接的安全软件进行测试。
- **使用热点网络**：切换到手机4G/5G热点网络，可快速判断是否为当前局域网问题。
