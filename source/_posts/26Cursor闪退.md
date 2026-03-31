---
title: Cursor在Windows电脑闪退处理参考
date: 2026-03-30
categories: 
    - 学CS/SE
tags: 
    - 修电脑    
    - Windows
    - Agent
desc: Cursor突然闪退的错误排查与处理。
---



### 问题现象

某天 Cursor 突然无法启动：

* 双击图标后闪退
* **完全没有报错窗口**
* 任务管理器中闪现一个 `Cursor.exe` 然后立即消失
* 前一天晚上仍能正常使用



<br>

### 初步尝试（全部无效）

依次尝试了以下方式，均无效果：

#### 卸载 + 重装 Cursor

* 无改善

#### 管理员运行 Cursor

* 仍旧闪退

#### 禁用杀毒软件（火绒）

* 问题依旧

#### 终端运行 `Cursor.exe --verbose`

* 无有效报错输出（Electron 未成功启动）



<br>

### WinEvent 日志分析

使用 PowerShell 搜集应用崩溃记录：

```powershell
Get-WinEvent -LogName Application |
    Where-Object {
        ($_.Message -match "cursor" -or $_.Message -match "electron" -or $_.Message -match "chrome") -and
        ($_.Id -eq 1000 -or $_.Id -eq 1001)
    } |
    Select-Object TimeCreated, Id, LevelDisplayName, Message |
    Format-List
```

得到了多条 **RADAR_PRE_LEAK_64**（内存泄露预判断）相关记录：

```
事件名称: RADAR_PRE_LEAK_64
P1: Cursor.exe
P2: 2.4.31.0
...
```

说明 Electron 在初始化阶段就崩掉，但 **没有明确报错源头**。



<br>

### 系统底层修复（仍无效）

因为 Windows 事件查看器中出现了大量如下错误：

* ESENT 权限错误
* WMI 损坏
* MSDTC 组件缺失
* ActivationContext 错误

尝试执行系统修复：

#### 修复性能计数器

```powershell
lodctr /R
```

#### 修复 WMI

```powershell
winmgmt /verifyrepository
winmgmt /salvagerepository
winmgmt /resetrepository
```

#### 系统文件修复

```powershell
sfc /scannow
DISM /Online /Cleanup-Image /RestoreHealth
```

> 结果：全部执行成功，但 Cursor **仍然闪退**。



<br>

### 重装 WebView2（Electron UI 依赖）

下载安装：

[https://developer.microsoft.com/en-us/microsoft-edge/webview2/](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)

→ 重启
→ Cursor 还是闪退

---

<br>

### 定位问题：Cursor 配置损坏

Cursor 的配置分两部分：

| 位置                      | 说明                                    |
| ----------------------- | ------------------------------------- |
| `%LocalAppData%\Cursor` | Electron 缓存、GPU 缓存、Crashpad、IndexedDB |
| `%AppData%\Cursor`      | 用户配置、工作区、设置文件（最关键）                    |

先尝试**隔离 LocalAppData**：

```powershell
Rename-Item "$env:LocalAppData\Cursor" "$env:LocalAppData\Cursor_backup_full"
```

结果：**仍然闪退**
说明不是 LocalAppData 的问题。



<br>

### 关键突破：隔离 Roaming 配置（AppData\Roaming）

执行：

```powershell
$roamingCursor = "$env:AppData\Cursor"
Rename-Item $roamingCursor "${roamingCursor}_backup"
```

执行后 Cursor 可正常启动。

结论：
**Cursor 的 Roaming 配置（设置/状态文件）损坏**
尤其可能是：

* `User/` 下的某些状态
* `Preferences`
* `keybindings.json`
* `Local Storage` / `WebStorage` 某个 corrupted SQLite



<br>

### 恢复旧配置（安全分步骤）

由于直接全部复制回去可能再次导致闪退——因此采用“逐步恢复 + 可回退”的方式。

#### 创建恢复前备份（用于回滚）

```powershell
$old = "$env:AppData\Cursor_backup"      
$new = "$env:AppData\Cursor"
$backup = "$env:AppData\Cursor_restore_backup"

Copy-Item $new $backup -Recurse -Force
```

---

#### 优先恢复安全文件夹

```powershell
$foldersToRestore = @("Workspaces", "User", "WebStorage", "Local Storage")

foreach ($folder in $foldersToRestore) {
    if (Test-Path "$old\$folder") {
        Copy-Item "$old\$folder" "$new" -Recurse -Force
    }
}
```



#### 恢复核心设置文件

```powershell
$settingsFiles = @("Settings.json","Preferences","keybindings.json")

foreach ($file in $settingsFiles) {
    if (Test-Path "$old\$file") {
        Copy-Item "$old\$file" "$new" -Force
    }
}
```

---

<br>

### 回退机制（闪退时用）

如果 Cursor 恢复后再次闪退：

```powershell
Remove-Item "$env:AppData\Cursor" -Recurse -Force
Copy-Item "$env:AppData\Cursor_restore_backup" "$env:AppData\Cursor" -Recurse -Force
```

确保任何恢复操作**都是安全可逆的**。


<br>

### 最终结果

恢复以下后仍能正常启动：

* Workspaces
* User
* Local Storage
* WebStorage
* Settings.json / Preferences / keybindings.json

说明导致闪退的并非你的自定义环境，而是 Roaming 下某些运行时状态缓存（SQLite、IndexedDB、Session Storage 等）。



## 总结

本次Cursor 的崩溃原因是：

> **%AppData%\Cursor 下某些运行时状态文件损坏触发 Electron 初始化崩溃**
> （通常是 Local Storage / WebStorage / IndexedDB）

通过 **隔离 Roaming 配置 + 分步骤恢复 + 必要时回退**
成功保留了全部重要配置，不再闪退。



## 未来如何避免

#### 定期备份以下目录：

```
%AppData%\Cursor\User
%AppData%\Cursor\Workspaces
%AppData%\Cursor\Settings.json
```


#### 避免备份缓存（避免未来恢复后再损坏）

不要恢复：

* Cache
* CachedProfilesData
* Code Cache
* GPUCache
* Crashpad

它们都可自动重建。
