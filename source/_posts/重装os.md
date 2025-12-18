---
title: 联想电脑Windows11重装操作系统修复参考
date: 2025-12-12
categories: 
    - 学CS/SE
tags: 
    - 修电脑    
    - Windows
desc: 喜报！只能重装！联想我恨你。各种踩坑损失惨重的一集，浪费了我宝贵的两天时间，还好chocolately救我狗命，可惜只能救一点但不多。
---

【特别鸣谢：ZJUEVA社团志愿者对于本次重装OS的帮助orz】

### 背景

蓝屏许多次之后，2025.12.11凌晨4点左右，随着某一次系统更新蓝屏、企图和之前一样使用系统还原点之后，电脑的扬声器、麦克风坏了。

#### 故障排查步骤

**1. 检查系统日志**
```powershell
# 打开事件查看器
eventvwr.msc

# 查看Windows日志 -> 系统，查找错误和警告
# 重点关注：
# - 系统错误（蓝屏代码）
# - 驱动程序错误
# - 硬件故障记录
```

让AI写脚本整理错误并判断。-> 需要排查硬件、驱动问题。

**2. 检查硬件**
- 打开设备管理器：`Win + X` -> 设备管理器
- 检查是否有黄色感叹号的设备
- 重点检查：
  - 声音、视频和游戏控制器
  - 音频输入和输出
  - 系统设备

**3. 检查驱动状态**
```powershell
# 查看音频驱动
Get-PnpDevice | Where-Object {$_.Class -eq "AudioEndpoint" -or $_.Class -eq "Sound"}

# 查看所有问题设备
Get-PnpDevice | Where-Object {$_.Status -ne "OK"}
```

**4. 重装驱动**
- **方法1：通过设备管理器**
  1. 设备管理器 -> 找到问题设备
  2. 右键 -> 卸载设备（勾选"删除此设备的驱动程序软件"）
  3. 扫描检测硬件改动，或重启电脑
  4. 系统会自动重新安装驱动

- **方法2：通过联想官网**
  1. 访问联想官网 -> 服务与支持
  2. 输入电脑型号（如：ThinkPad X1 Carbon）
  3. 下载对应型号的音频驱动
  4. 运行安装程序

- **方法3：使用驱动管理软件**
  - 联想电脑管家
  - Driver Booster（需谨慎使用）
  - 驱动精灵（不推荐，可能捆绑软件）

**5. 系统还原尝试**
```powershell
# 查看系统还原点
rstrui.exe
```
发现系统更新功能整个损坏。powershell也坏了。
安装win11镜像也不行。


**6. 最终判断**
以上方法都无法解决问题，推测是系统更新一半残余卡bug导致。

**重装系统。**


<br>

### 备份

重装系统只格式化C盘，不影响其他盘。
可以把内容备份去其他盘/拷硬盘U盘里。

#### Chocolatey

Windows包管理器。
导出已安装软件列表神器，能留下版本直接迁移。
- 导出为JSON格式，包含软件名称和版本信息
- 支持一键批量安装，自动恢复所有软件及版本
- 比手动安装更快速、准确

**重要提示：**
Chocolatey只能导出通过它安装的软件。如果软件不是通过Chocolatey安装的，需要先通过Chocolatey安装一次（即使已存在），这样导出时才能包含。

**将已安装软件导入Chocolatey管理：**
```powershell
# 方法1：对于已安装的软件，使用choco install（即使已存在也会被Chocolatey识别）
choco install git -y
choco install vscode -y
choco install obsidian -y
# ... 列出所有需要管理的软件

# 方法2：如果软件已存在，Chocolatey会检测到并询问是否继续
# 选择"是"即可，Chocolatey会记录该软件

# 方法3：查看Chocolatey仓库中是否有对应的包名
choco search <软件名>

# 注意：某些软件在Chocolatey中的包名可能与软件名不同
# 例如：Visual Studio Code的包名是vscode，不是visual-studio-code
```

**安装Chocolatey（如果未安装）：**
```powershell
# 以管理员身份运行PowerShell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

**导出已安装软件列表（JSON格式，带版本）：**
```powershell
# 导出Chocolatey安装的软件列表为JSON格式（包含版本信息）
choco export -o your_path\choco-packages.json

# 或者导出为文本格式（可选，用于查看）
choco list --local-only > your_path\choco-packages.txt
```

**备份Chocolatey配置：**
```powershell
# 备份Chocolatey配置文件
Copy-Item "$env:ChocolateyInstall\config\chocolatey.config" -Destination "your_path\chocolatey-config.xml"
```

**备份到C盘以外的地方：**
将导出的JSON文件和配置文件拷贝/移动到D盘或其他盘。

<br>

#### 用户数据



**备份C盘用户数据：**
```powershell
# 备份用户主目录（推荐使用robocopy，支持断点续传）
# 直接手动拷贝也行
robocopy "C:\Users\<用户名>" "your_backup_path\Users\<用户名>" /E /Z /R:3 /W:5 /MT:8 /LOG:"your_backup_path\backup-log.txt"

# 参数说明：
# /E - 复制所有子目录，包括空目录
# /Z - 支持断点续传
# /R:3 - 失败重试3次
# /W:5 - 重试等待5秒
# /MT:8 - 使用8个线程加速
# /LOG - 记录日志
```

**重要目录备份清单：**
```powershell
# .ssh目录（SSH密钥）
robocopy "C:\Users\<用户名>\.ssh" "your_backup_path\.ssh" /E /Z

# .vscode目录（VS Code配置和插件）
robocopy "C:\Users\<用户名>\.vscode" "your_backup_path\.vscode" /E /Z

# VS Code扩展列表
code --list-extensions > "your_backup_path\vscode-extensions.txt"

# Obsidian配置和数据
robocopy "C:\Users\<用户名>\Documents\Obsidian" "your_backup_path\Obsidian" /E /Z
# 或如果使用自定义位置
robocopy "D:\Obsidian" "your_backup_path\Obsidian" /E /Z

# Zotero数据
robocopy "C:\Users\<用户名>\Zotero" "your_backup_path\Zotero" /E /Z
# Zotero数据通常在：C:\Users\<用户名>\Zotero

# 桌面文件(如果在C盘)
robocopy "C:\Users\<用户名>\Desktop" "your_backup_path\Desktop" /E /Z

# 文档
robocopy "C:\Users\<用户名>\Documents" "your_backup_path\Documents" /E /Z

# 下载
robocopy "C:\Users\<用户名>\Downloads" "your_backup_path\Downloads" /E /Z

# 图片
robocopy "C:\Users\<用户名>\Pictures" "your_backup_path\Pictures" /E /Z

```


**浏览器书签和配置：**
```powershell
# Chrome
robocopy "C:\Users\<用户名>\AppData\Local\Google\Chrome\User Data" "your_backup_path\Chrome\User Data" /E /Z

# Edge
robocopy "C:\Users\<用户名>\AppData\Local\Microsoft\Edge\User Data" "your_backup_path\Edge\User Data" /E /Z



**Git配置：**
```powershell
# .gitconfig
Copy-Item "C:\Users\<用户名>\.gitconfig" -Destination "your_backup_path\.gitconfig"
```
<br>


#### 桌面

**将桌面位置改为D盘（重装前操作）：**

1. **在D盘创建桌面文件夹（可选）**
   ```powershell
   # 创建D盘桌面目录（也可以直接在属性中选择时创建）
   New-Item -ItemType Directory -Path "D:\Desktop" -Force
   ```
   注意：如果文件夹不存在，在修改属性时Windows会自动创建。

2. **修改桌面属性指向D盘**
   - 打开文件资源管理器
   - 右键点击左侧"桌面" -> 属性
   - 切换到"位置"选项卡
   - 点击"移动"按钮
   - 选择 `D:\Desktop` 文件夹（如果不存在，可以在这里新建）
   - 点击"确定"
   - **系统会自动询问是否移动现有文件，选择"是"即可**（Windows会自动将所有桌面文件移动到新位置）

3. **验证桌面位置**
   ```powershell
   # 检查桌面实际位置
   (Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders").Desktop
   # 应该显示：D:\Desktop
   ```

**优点：**
- 重装系统时桌面文件不会丢失（因为不在C盘）
- 减少C盘占用空间
- 便于备份和管理 

<br>

#### 其他重要数据


**环境变量和PATH：**
```powershell
# 导出环境变量
[System.Environment]::GetEnvironmentVariable("Path", "User") | Out-File "your_backup_path\user-path.txt"
[System.Environment]::GetEnvironmentVariable("Path", "Machine") | Out-File "your_backup_path\system-path.txt"
```

**其他配置文件：**
```powershell
# PowerShell配置文件
Copy-Item "$PROFILE" -Destination "your_backup_path\PowerShell-profile.ps1" -ErrorAction SilentlyContinue

# Windows Terminal配置
robocopy "C:\Users\<用户名>\AppData\Local\Packages\Microsoft.WindowsTerminal_*\LocalState" "your_backup_path\WindowsTerminal" /E /Z
```
<br>

#### WSL-Ubuntu安装到D盘

**重要说明：**
- WSL默认安装在C盘，占用空间较大
- 将WSL安装到D盘可以节省C盘空间，重装系统时数据不会丢失
- 如果WSL已经在C盘，可以导出后迁移到D盘

##### 方法1：全新安装WSL到D盘

1. **启用WSL功能（如果未启用）**
   ```powershell
   # 以管理员身份运行PowerShell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   
   # 重启计算机
   Restart-Computer
   ```

2. **下载WSL2内核更新包**
   - 访问：https://aka.ms/wsl2kernel
   - 下载并安装WSL2 Linux内核更新包

3. **设置WSL默认版本为WSL2**
   ```powershell
   wsl --set-default-version 2
   ```

4. **下载Ubuntu安装包**
   - 访问Microsoft Store或直接下载Ubuntu的.appx安装包
   - 将下载的`.appx`文件重命名为`.zip`
   - 解压到D盘，例如：`D:\Ubuntu`

5. **安装Ubuntu到D盘**
   ```powershell
   # 进入解压后的目录
   cd D:\Ubuntu
   
   # 运行ubuntu.exe完成安装
   .\ubuntu.exe
   ```

6. **设置默认用户**
   - 首次运行会要求创建用户名和密码
   - 完成后Ubuntu会安装在D盘

##### 方法2：迁移已安装的WSL到D盘

如果WSL已经在C盘，可以导出后迁移：

1. **导出当前WSL分发版**
   ```powershell
   # 以管理员身份运行PowerShell
   # 查看已安装的WSL分发版
   wsl --list --verbose
   
   # 导出Ubuntu（替换为你的分发版名称）
   wsl --export Ubuntu D:\Ubuntu\ubuntu-backup.tar
   ```

2. **注销当前WSL分发版**
   ```powershell
   # 注销Ubuntu（会删除C盘中的WSL数据）
   wsl --unregister Ubuntu
   ```

3. **导入到D盘**
   ```powershell
   # 在D盘创建Ubuntu目录
   New-Item -ItemType Directory -Path "D:\Ubuntu" -Force
   
   # 导入Ubuntu到D盘
   wsl --import Ubuntu D:\Ubuntu D:\Ubuntu\ubuntu-backup.tar --version 2
   ```

4. **设置默认用户**
   ```powershell
   # 进入Ubuntu安装目录
   cd D:\Ubuntu
   
   # 设置默认用户（替换为你的用户名）
   ubuntu.exe config --default-user <你的用户名>
   
   # 或者使用wsl命令
   wsl -d Ubuntu -u <你的用户名>
   ```

5. **验证安装位置**
   ```powershell
   # 查看WSL分发版信息
   wsl --list --verbose
   
   # 进入Ubuntu验证
   wsl -d Ubuntu
   
   # 在Ubuntu中查看挂载点
   df -h
   ```

**备份WSL数据（重装前）：**

如果WSL已安装在D盘，数据会自动保留。如果担心，可以额外备份：

```powershell
# 导出WSL分发版（备份）
wsl --export Ubuntu D:\Backup\ubuntu-backup.tar

# 或者直接备份整个D:\Ubuntu目录
robocopy "D:\Ubuntu" "your_backup_path\Ubuntu" /E /Z
```

<br><br>

### 重装

#### 准备Windows 11安装介质

**方法1：使用Media Creation Tool（推荐）**
1. 访问微软官网下载Media Creation Tool
2. 运行工具，选择"为另一台电脑创建安装介质"
3. 选择语言、版本和体系结构（64位）
4. 选择"USB闪存驱动器"或"ISO文件"
5. 如果选择USB，插入至少8GB的U盘
6. 等待下载和创建完成

**方法2：下载ISO文件**
1. 访问微软官网下载Windows 11 ISO
2. 使用Rufus等工具制作启动U盘
- 下载Rufus
- 插入U盘
- 选择ISO文件
- 选择U盘
- 点击"开始"


#### 进入BIOS/UEFI设置

**联想电脑进入BIOS方法：**
- 开机时按 `F2` 或 `Fn + F2`
- 或按 `F12` 进入启动菜单

**BIOS设置：**
1. 找到"Boot"或"启动"选项
2. 将U盘设置为第一启动项
3. 确保UEFI模式已启用
4. 保存并退出（通常是 `F10`）

#### 开始安装

**安装步骤：**
1. 从U盘启动后，选择语言、时间和键盘输入法
2. 点击"现在安装"
3. 输入产品密钥（如果有，也可以稍后激活）
4. 选择Windows版本（通常选择Windows 11 Pro或Home，我的是Home）
5. 接受许可条款
6. 选择安装类型：
   - **升级**：保留文件、设置和应用程序（不推荐，可能保留问题）
   - **自定义**：仅安装Windows（推荐，全新安装）

**自定义安装（全新安装）：**
1. 选择要安装Windows的驱动器
2. 如果看到多个分区，建议：
   - 删除所有分区（注意：会丢失所有数据！）
   - 创建新分区
   - 选择主分区进行安装
3. 点击"下一步"开始安装
4. 等待安装完成（约30min）

#### 首次设置

**重要说明：**

**会改变的内容：**
- 机器名字（计算机名）
- 用户账号名
- 用户密码
- 系统设置和注册表
- 已安装的软件（需要重新安装）
- C盘的所有数据（格式化后清空）
- 用户配置文件路径（`C:\Users\<新用户名>`）

**不会改变的内容：**

**硬件标识（物理特性，无法改变）：**
- **MAC地址** - 网卡、WiFi、蓝牙的物理地址（硬件层面，重装系统不会改变）
- **硬件ID** - CPU、主板、硬盘等硬件的唯一标识
- **BIOS/UEFI设置** - 除非手动修改，否则保持不变
- **硬件驱动** - 虽然需要重新安装，但硬件本身不变

**备份在其他盘的数据（重装后可直接使用或恢复）：**
- **SSH密钥** - 如果备份在D盘或其他盘（如`D:\Backup\.ssh`），恢复后可直接使用
- **Git配置** - `.gitconfig`文件，恢复后配置不变
- **VS Code配置** - 如果备份了`.vscode`目录，恢复后配置和插件列表不变
- **Obsidian数据** - 如果数据在D盘，重装后直接可用
- **Zotero数据** - 如果数据在D盘，重装后直接可用
- **桌面文件** - 如果桌面已指向D盘，重装后文件不变
- **其他D盘/E盘数据** - 所有非C盘的数据都不会被格式化

**其他不变内容：**
- **软件许可证** - 如果绑定硬件或账户，重装后仍可使用
- **Microsoft账户授权** - 如果之前已激活，重装后登录同一账户可自动激活
- **产品密钥** - Windows产品密钥（如果绑定硬件，重装后可能自动激活）
- **浏览器书签和密码** - 如果使用云同步（Chrome/Edge账户），登录后自动恢复
- **OneDrive数据** - 如果使用OneDrive同步，登录后自动下载

**注意事项：**
- SSH密钥、配置文件等需要从备份恢复到新用户目录才能使用
- 软件需要重新安装，但配置和数据可以从备份恢复
- 如果使用云同步服务，登录账户后很多内容会自动恢复

**OOBE（开箱即用体验）设置：**

1. **选择区域**

2. **键盘布局**

3. **网络连接** (建议跳过)
   - 可以连接WiFi，也可以点击"我没有Internet连接"跳过
   - 跳过网络连接可以避免强制登录Microsoft账户
   - 稍后可以在设置中配置网络

4. **创建用户账户**
   - 如果跳过了网络连接，可以直接创建本地账户
   - 输入用户名（可以与之前不同）
   - 如果连接了网络，可能需要登录Microsoft账户或创建新的Microsoft账户

5. **设置密码和PIN**

6. **隐私设置**
   - 关闭不需要的选项：
     - 位置服务
     - 查找我的设备
     - 诊断数据（可选）
     - 广告ID
     - 语音识别
   - 根据个人需求选择

7. **等待系统完成初始化**


**设置完成后：**
- 系统会创建新的用户配置文件（路径：`C:\Users\<新用户名>`）
- 旧的用户数据不会自动恢复，需要手动从备份恢复
    - 但如果C盘足够大会有Windows.old文件夹，里面保留了旧C盘的用户文件夹，别的C盘东西都没了。
    - C盘不够大就无了。
- SSH密钥、配置文件等需要从备份位置恢复到新用户目录
- Chocolatey可以通过JSON文件一键恢复所有软件（包含版本），**需要重新输账户密码，之前务必存好。**
- 没装进chocolately的手动自己再装一下。



<br>

### 恢复

#### 恢复桌面位置到D盘

**重装后恢复桌面指向D盘：**

1. **检查D盘桌面文件夹是否存在**
   ```powershell
   # 检查D盘桌面文件夹
   Test-Path "D:\Desktop"
   ```

2. **如果D盘桌面文件夹不存在，创建它**
   ```powershell
   # 如果之前备份了桌面，先恢复
   if (Test-Path "your_backup_path\Desktop") {
       robocopy "your_backup_path\Desktop" "D:\Desktop" /E /Z
   } else {
       # 创建新的桌面文件夹
       New-Item -ItemType Directory -Path "D:\Desktop" -Force
   }
   ```

3. **修改桌面属性指向D盘**
   - 打开文件资源管理器
   - 右键点击左侧"桌面" -> 属性
   - 切换到"位置"选项卡
   - 点击"移动"按钮
   - 选择 `D:\Desktop` 文件夹
   - 点击"确定"
   - 如果D盘桌面文件夹已有文件，系统会询问是否合并，选择"是"

4. **验证桌面位置**
   ```powershell
   # 检查桌面实际位置
   (Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders").Desktop
   # 应该显示：D:\Desktop
   
   # 或者使用注册表查看
   reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders" /v Desktop
   ```

5. **刷新桌面**


**注意事项：**
- 如果重装前桌面已经指向D盘，重装后只需要重新设置属性即可
- 桌面文件会保留在D盘，不会因为重装系统而丢失
- 有一些软件快捷方式会失效，装好后重新发送到桌面即可

<br>

#### 恢复其他用户数据

**恢复重要目录：**
```powershell
# 恢复.ssh目录
robocopy "your_backup_path\.ssh" "C:\Users\<用户名>\.ssh" /E /Z

# 恢复.vscode目录
robocopy "your_backup_path\.vscode" "C:\Users\<用户名>\.vscode" /E /Z

# 恢复文档
robocopy "your_backup_path\Documents" "C:\Users\<用户名>\Documents" /E /Z

# 恢复下载
robocopy "your_backup_path\Downloads" "C:\Users\<用户名>\Downloads" /E /Z

# 恢复图片
robocopy "your_backup_path\Pictures" "C:\Users\<用户名>\Pictures" /E /Z
```

<br>

#### 恢复Chocolatey和软件

**重新安装Chocolatey：**
```powershell
# 以管理员身份运行PowerShell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

**批量安装软件（一键安装）：**
```powershell
# 从JSON文件一键安装所有软件（包含版本信息）
choco install your_backup_path\choco-packages.json -y

# 或者使用import命令（推荐）
choco import your_backup_path\choco-packages.json -y

# 查看JSON文件内容（可选）
Get-Content "your_backup_path\choco-packages.json" | ConvertFrom-Json
```

**注意事项：**
- JSON文件包含软件名称和版本信息，可以精确恢复到之前的版本
- 如果某些软件安装失败，可以单独安装：
  ```powershell
  choco install <package_name> -y
  ```
- 安装过程中可能需要输入账户密码，请提前准备好

<br>

#### 恢复VS Code配置和插件

**恢复VS Code配置：**
```powershell
# 配置已通过.vscode目录恢复，验证一下
Test-Path "C:\Users\<用户名>\.vscode"
```

**恢复VS Code扩展：**
```powershell
# 读取扩展列表并安装
if (Test-Path "your_backup_path\vscode-extensions.txt") {
    Get-Content "your_backup_path\vscode-extensions.txt" | ForEach-Object {
        code --install-extension $_
    }
}
```

【更方便的方法是VSCode登录账号，sync】

#### 恢复Obsidian和Zotero

**恢复Obsidian数据：**
```powershell
# 如果Obsidian数据在Documents目录
robocopy "your_backup_path\Obsidian" "C:\Users\<用户名>\Documents\Obsidian" /E /Z

# 或恢复到自定义位置
robocopy "your_backup_path\Obsidian" "D:\Obsidian" /E /Z
```
【更方便的方法是Obsidian登录账号，sync】


**恢复Zotero数据：**
```powershell
# Zotero数据目录通常在用户目录下
robocopy "your_backup_path\Zotero" "C:\Users\<用户名>\Zotero" /E /Z
```

【更方便的方法是Zotero+坚果云登录账号，sync，参考：[Zotero+坚果云文献管理配置](2411Zotero.md)】


#### 恢复浏览器配置

**恢复Chrome（关闭Chrome后操作）：**
```powershell
robocopy "your_backup_path\Chrome\User Data" "C:\Users\<用户名>\AppData\Local\Google\Chrome\User Data" /E /Z
```

**恢复Edge（关闭Edge后操作）：**
```powershell
robocopy "your_backup_path\Edge\User Data" "C:\Users\<用户名>\AppData\Local\Microsoft\Edge\User Data" /E /Z
```

#### 恢复Git配置

**恢复Git配置：**
```powershell
Copy-Item "your_backup_path\.gitconfig" -Destination "C:\Users\<用户名>\.gitconfig"
```
<br>

#### 验证恢复

**检查关键软件和数据：**
```powershell
# 检查Git
git --version

# 检查Python
python --version

# 检查VS Code
code --version

# 检查SSH密钥
Test-Path "C:\Users\<用户名>\.ssh\id_rsa"

# 检查桌面位置
(Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders").Desktop
```

<br>