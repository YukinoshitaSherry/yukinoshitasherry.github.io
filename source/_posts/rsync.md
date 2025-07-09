---
title: Remote-SSH跨服务器传输与挂载
date: 2025-07-02
categories: 
    - 学CS/SE
tags: 
    - SSH
    - Linux
desc: Remote-SSH 跨服务器免密rsync传输、scp传输、sshfs挂载、unar解压使用指南
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



