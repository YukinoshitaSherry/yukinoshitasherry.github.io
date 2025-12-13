---
title: Zotero+坚果云文献管理配置
date: 2024-11-17
categories: 
    - 学CS/SE
tags: 
    - 文献管理
desc: 利用Zotero管理论文，但zotero的免费托管内存不够，配合支持WebDAV的坚果云可以方便地满足上云与设备迁移需要。
---

本文主要参考：<a href="https://zhuanlan.zhihu.com/p/26564079081">知乎：Zotero7.0+坚果云，电脑手机平板无缝切换</a>


### 背景

Zotero的同步分为二部分：
1、数据同步：条目信息，注释、链接、标签等，附件文件除外的所有内容。
2、文件同步：附件，包括快照、pdf。

数据同步没有大小限制，永远免费；文件同步只免费300M，超过需要付费。
但Zotero支持用WebDAV方式上传,坚果云是国内最好的支持此方式的网盘。

#### WebDAV

基于Web的分布式编写和版本控制（**Web**-based **D**istributed **A**uthoring and **V**ersioning）是超文本传输协议（HTTP）的扩展，有利于用户间协同编辑和管理存储在万维网服务器文档。
WebDAV由互联网工程任务组的工作组在RFC 4918中定义。它的设计目标包括提供对文件的创建、编辑、删除和读取等基本操作的支持，同时解决多用户协作中的同步和冲突问题。 具体来说，WebDAV 增强了 HTTP 协议，它在GET、POST、HEAD等几个HTTP标准方法以外添加了一些新的方法与头信息，使应用程序可对Web Server直接读写，并支持写文件锁定(Locking)及解锁(Unlock)、支持文件的版本控制，使得用户能够在远程服务器上执行文件操作。这些扩展包括 PROPFIND、PROPPATCH、MKCOL、COPY、MOVE 等方法，用于支持元数据管理、目录操作和文件移动等功能。

具体参考：<a href="https://cloud.tencent.com/developer/article/2483609"> https://cloud.tencent.com/developer/article/2483609 </a> & <a href="https://developer.aliyun.com/article/1463361"> https://developer.aliyun.com/article/1463361 </a>

> "目前在国内支持webdav的网盘非常的少，用的比较多的也几乎就坚果云一家，为什么很多国内的网盘都不支持webdav，这项技术很难吗？当然不是，对于国内的大部分网盘，如果想要支持webdav基本上都是没有技术障碍的，但是如果支持了webdav，那么大家都可以使用一些第三方软件来管理这些网盘，也会大大减少网盘应用的安装量，一些“附加服务”也无法添加到专有应用上，这对网盘服务商的是不利的，所以很多网盘产品，特别是大容量网盘产品，一般都是不支持webdav的。" —— 少数派@之晓青年《WebDAV是什么，有哪些支持webdav的网盘和工具？》(https://sspai.com/post/60540)


<br>


### 操作

#### 数据同步

先开启数据同步：
打开Zotero客户端-点击`编辑`-`设置`-`同步`

<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/b49ac292140835813b91be11da82efdf.png" style="width:20%">

开通Zotero网盘：在这个设置界面，点击`创建帐户`。
或直接进入<a href="https://www.zotero.org/">Zotero官网</a>，注册Zotero帐号。

数据同步下，填写**Zotero网盘帐号和密码**，然后点启用同步。

#### 文件同步

去<a href="https://www.jianguoyun.com">坚果云官网</a>，注册坚果云帐号，登录。
找到右上角用户名-点击`帐户信息`-点击`安全选项`-点击`添加应用`，添加名为“Zotero”的应用。
完成后如下：
<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251214020537373.png" style="width:100%">

回到Zotero客户端，点击`编辑`-`设置`-`同步`，在文件同步处，`我的文库`附件同步方式前打勾，同步方式改为`WebDAV`。
<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251214015735590.png" style="width:80%">

网址、用户名、密码是**坚果云授权的存放文件的地址、用户和应用密码**。
注意密码不是坚果云的账户密码，是**坚果云官网安全选项**里面那个，上面安全选项界面图，点击右下角应用名称Zotero那一栏的`显示密码`即可查看、复制。

填写完后，点击`验证服务器`，提示`“文件同步设定成功”`即可。

#### 手动同步

在Zotero客户端点击右上角的双循环箭头。
<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251214021055447.png" style="width:100%">


<br>

若重装操作系统/迁移设备重复填写以上内容即可。

<br>

### 存储迁移

为了防止再一次重装后数据没了，**不要装在默认的C盘**，自己换个路径。
<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/20251214021637734.png" style="width:80%">
事先建好一个空文件夹，关掉Zotero，把原先Data directory的默认路径下的东西全部复制粘贴到这个新文件夹下，重启Zotero即可。

已链接附件根目录不用设置 ，因为添加附件使用链接方式，移动端不能浏览。

<br>
