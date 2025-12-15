---
title: Callouts模板语法
date: 2024-10-19
categories:
- 学CS/SE
tags:
- 可视化
desc: Callout 是 Obsidian 的特有语法，源于 markdown的引用。其他地方也可以看见这种卡片。
---

【博客框架不支持渲染Callouts格式，自己魔改了框架】


Callout 是 Obsidian 的特有语法，源于 markdown的引用。其他地方也可以看见这种卡片。

**需要注意，Callout 的名称中不能包含空格。**
### 自定义标题


- +型 +号后面有空格才行 (默认展开，可以点击∨后折叠，变成和-型一样)
```callouts
> [!INFO]+ 请输入标题
> 正文
```
> [!INFO]+ 请输入标题
> 正文

如果需要换行：
```callouts
> [!INFO]+ 请输入标题
> 正文1
> 
> 正文2
```
> [!INFO]+ 请输入标题
> 正文 1
> 
> 正文 2



- -型(默认是折叠的，点击 > 后会展开，变成和+型一样。)
```callouts
>[!INFO]- 请输入标题
>正文
```
>[!INFO]- 请输入标题
> 正文


### 自带格式类型
#### 笔记
- 样例
- info
```callouts
> [!INFO] 
> 这里是callout的info模块 
> 支持**markdown** 和 [[Internal link|wikilinks]].
```
> [!INFO] 
> 这里是callout的info模块 
> 支持**markdown** 和 [[Internal link|wikilinks]].

其它只要把INFO改掉就行,改成对应的大写会出现各种颜色和图标
- note
```callouts
> [!NOTE]
> 这里是callout的note模块
```
> [!NOTE]
> 这里是callout的note模块


#### 摘要
- abstract
```callouts
> [!ABSTRACT]
> 这里是callous的abstract模块
```
> [!ABSTRACT]
> 这里是callous的abstract模块

- summary
```callouts
> [!SUMMARY]
> 这里是callouts的summary模块
```
> [!SUMMARY]
> 这里是callouts的summary模块

- tldr
```callouts
> [!TLDR]
> 这里是callouts的tldr模块
```
> [!TLDR]
> 这里是callouts的tldr模块


#### 重点与提示
- tip
```callouts
> [!TIP]
> 这里是callouts的tip模块
```
> [!TIP]
> 这里是callouts的tip模块

- hint
```callouts
> [!HINT]
> 这里是callouts的hint模块
```
> [!HINT]
> 这里是callouts的hint模块

- important
```callouts
> [!IMPORTANT]
> 这里是callouts的important模块
```
> [!IMPORTANT]
> 这里是callouts的important模块

#### 警告与报错
- warning
```callouts
> [!WARNING]
> 这里是callouts的warning模块
```
> [!WARNING]
> 这里是callouts的warning模块

- caution
```callouts
> [!CAUTION]
> 这里是callouts的caution模块
```
> [!CAUTION]
> 这里是callouts的caution模块

- attention
```callouts
> [!ATTENTION]
> 这里是callouts的attention模块
```
> [!ATTENTION]
> 这里是callouts的attention模块

- failure
```callouts
> [!FAILURE]
> 这里是callouts的failure模块
```
> [!FAILURE]
> 这里是callouts的failure模块

- fail
```callouts
> [!FAIL]
> 这里是callouts的fail模块
```
> [!FAIL]
> 这里是callouts的fail模块

- missing
```callouts
> [!MISSING]
> 这里是callouts的missing模块
```
> [!MISSING]
> 这里是callouts的missing模块

- danger
```callouts
> [!DANGER]
> 这里是callouts的danger模块
```
> [!DANGER]
> 这里是callouts的danger模块

- error
```callouts
> [!ERROR]
> 这里是callouts的error模块
```
> [!ERROR]
> 这里是callouts的error模块

- bug
```callouts
> [!BUG]
> 这里是callouts的bug模块
```
> [!BUG]
> 这里是callouts的bug模块


#### 引用
- example
```callouts
> [!EXAMPLE]
> 这里是callouts的example模块
```
> [!EXAMPLE]
> 这里是callouts的example模块

- quote
```callouts
> [!QUOTE]
> 这里是callouts的quote模块
```
> [!QUOTE]
> 这里是callouts的quote模块

- cite
```callouts
> [!CITE]
> 这里是callouts的cite模块
```
> [!CITE]
> 这里是callouts的cite模块




#### 问题与帮助
- question
```callouts
> [!QUESTION]
> 这里是callouts的question模块
```
> [!QUESTION]
> 这里是callouts的question模块

- help
```callouts
> [!HELP]
> 这里是callouts的help模块
```
> [!HELP]
> 这里是callouts的help模块

- faq
```callouts
> [!FAQ]
> 这里是callouts的faq模块
```
> [!FAQ]
> 这里是callouts的faq模块



#### 代办与已办
- todo
```callouts
> [!TODO]
> 这里是callouts的todo模块
```
> [!TODO]
> 这里是callouts的todo模块

- success
```callouts
> [!SUCCESS]
> 这里是callouts的success模块
```
> [!SUCCESS]
> 这里是callouts的success模块

- check
```callouts
> [!CHECK]
> 这里是callouts的check模块
```
> [!CHECK]
> 这里是callouts的check模块

- done
```callouts
> [!DONE]
> 这里是callouts的done模块

```
> [!DONE]
> 这里是callouts的done模块



### 自定义格式类型

自定义Callouts 需要控制两个变量, 颜色和图标。
如果想要获得一个名为 custom-type 的模板，则创建一个 css 文件：
```css
.callout[data-callout="custom-type"] {
    --callout-color: 0, 0, 0;
    --callout-icon: lucide-alert-circle;
}
```

图标代码可以从 [Lucide](https://link.zhihu.com/?target=https%3A//lucide.dev/) 找到，也可以使用自己找的 svg 图标。
格式：
```css
--callout-icon: '<svg>...custom svg...</svg>';
```

使用：
```callouts
> [!custom-type] Title
> Contents
```

同样可以基于 css 语法调整宽度高度等等。


<br>

