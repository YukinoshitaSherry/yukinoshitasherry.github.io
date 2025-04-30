---
title: Git&Github相关知识
date: 2023-04-02
categories:
- 学CS/SE
tags:
- Git&Github
desc: ZJU朋辈辅学技能拾遗笔记，主要参考了授课人鹤翔万里的ppt
---

<a href="https://www.bilibili.com/video/BV1og4y1u7XU/?vd_source=fa4dcf78649ce6604c2727b4c64e76dc">朋辈辅学-b站视频</a>

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





