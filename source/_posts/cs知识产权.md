---
title: 美国计算机知识产权与信息政策
date: 2026-09-21
categories:
- 思万物
tags:
- 法律
- 科技
desc: MIT 15.628J、6.4590 与 Stanford CS202 美国法笔记：专利、版权、商业秘密、商标、平台责任、隐私及 2025–2026 生成式 AI 诉讼。
hidden: true
---

本文以美国法为主。基础材料是 MIT 15.628J / 6.903J *Patents, Copyrights, and the Law of Intellectual Property*（Spring 2013）、MIT 6.4590 *Foundations of Information Policy*（Fall 2024）与 Stanford CS202 *Law for Computer Science Professionals*。MIT 给 doctrine 和信息政策框架；Stanford CS202 用技术公司真实争议把 software patent、API、商业秘密、CFAA 和雇佣合同串起来。2013 年之后再补 *Alice*、*Google v. Oracle*、*Warhol*、*Authors Guild v. Google* 与 2025–2026 训练数据诉讼。

- [15.628J](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/)
- [6.4590](https://ocw.mit.edu/courses/6-4590-foundations-of-information-policy-fall-2024/)
- [Stanford CS202（ExploreCourses）](https://explorecourses.stanford.edu/search?q=CS+202)

<br>

> [!INFO]+ 章节
>
> `#` 按问题顺序，不按课表周次，也不是四种权利平铺。激励 → 体系：为何给排他权，四件工具怎么分工。版权 → 专利 → 秘密 → 商标：私权本体，计算机里最常碰到的在前；版权按独创、思想、软件、接口、合理、间接拆构成要件，专利按资格到救济走授权与诉讼。归属：权利在谁手里。信息政策：言论、平台、搜查、隐私、跨境，私权管不到的公法规制。人工智能 → 智能体：把前面的规则套到训练、输出、作者、发明人，再扩到带工具的连续行动。未决 → 资料：尚未被最高法院钉住的问题，以及课程、法条、判决出处。

<br>

> [!NOTE]+ CS / AI 问题映射
>
> GitHub 代码能否用于训练 → copyright、许可证、fair use。Agent 自动爬网站 → CFAA、合同、版权、隐私。API 实现与 API 本身的区别 → § 102(b)、*Lotus*、*Google v. Oracle*。实习期间写出的 agent framework 归谁 → assignment、work made for hire、*Stanford v. Roche*。Open-weight model 能否商用 → 模型许可证、权重和代码的权利分层。研究代码能否公开到 GitHub → 雇佣合同、商业秘密、专利 novelty。NDA 与论文发表冲突 → 保密义务、发明转让、学校与公司之间的 publication review。

# 激励

宪法写得很窄：

*To promote the Progress of Science and useful Arts, by securing for limited Times to Authors and Inventors the exclusive Right to their respective Writings and Discoveries.*（U.S. Const. art. I, § 8, cl. 8）

Authors / Inventors，不是作品、不是发明物。limited Times，不是永久。目的是 progress。自然权利叙事从一开始就没写进去。

Trademark 不在这条里。Lanham Act 走反不正当竞争，管消费者会不会认错来源。Trade secret 更接近侵权和合同，联邦层 2016 年才有 Defend Trade Secrets Act。

课上真正反复出现的，不是“要不要保护创新”。是：给了排他权之后，后人还能不能站上去。专利 claim 写太上位，锁死方向。版权若盖到功能，兼容和反向工程一起没。商业秘密不公开，社会拿不到披露。每一种保护都在买激励、付后续创新的账。

Arrow 的 information paradox 很直白。买方要知道 invention 是什么才肯付钱；知道了，往往不必再买。没有排他或保密，事前投资会偏少。反过来，排他过宽，复制成本已经接近零的东西被法律重新变贵。Patent 用公开换 20 年左右的独占。Copyright 用表达换复制禁令，到期进 public domain。Trade secret 不公开、无固定期限，但挡不住独立发现和合法 reverse engineering。三条路，不是强度排序。

<br>

# 体系

| | Patent | Copyright | Trade Secret | Trademark |
| :--- | :--- | :--- | :--- | :--- |
| 管什么 | 有用的技术方案 | original expression | 因保密而值钱的信息 | 来源识别 |
| 门槛 | useful / novel / non-obvious / eligible / 充分披露 | 独立创作 + 一点点 creativity，固定在有形介质 | 秘密 + 合理保密措施 + 经济价值 | 商业使用、能识别来源、还没变成通用名称 |
| 能挡谁 | 独立发明人也挡 | 只挡 copying | 只挡不正当获取和使用 | 挡混淆、有时挡淡化 |
| 多久 | 申请日起约 20 年 | 自然人 life+70；雇佣作品发表起 95 年 | 还是秘密就还在 | 一直用、一直续 |
| 换来什么 | 向公众披露 | 表达最终进公共领域 | 无强制披露 | 降低搜寻成本 |
| 软件里常见 | 写得够具体的系统 / 方法 claim | 代码、文档、非功能性结构 | 未开源权重、数据、过滤、系统提示 | 产品名、logo、模型名 |

实验室里四样会抢同一块资产。算法思想拿去申请，论文一发 novelty 可能没了，§ 101 还不一定过。代码版权自动有，开源等于按许可证丢掉一部分。权重和数据最像商业秘密，权重一公开就薄。产品名走商标。常见组合其实很土：论文换名声，数据当秘密，名字注册商标，专利只打真正有技术改进的那一层。

开源权重几乎是主动放弃商业秘密。Llama 那种 community license 是合同，不是 GPL 那种 copyright copyleft。对没点同意的下游下载者，约束力比想象中弱。

<br>

> [!NOTE]+ 四种不是强度排序
>
> 专利挡独立发明，但要公开、有期限、§ 101 可能过不去。版权自动获得，只挡 copying。商业秘密不公开，一泄露或被合法反向就没。商标可以续很久，不管创意。实验室常见组合很土：论文换名声，数据当秘密，名字注册商标，专利只打真正有技术改进的那一层。

<br>

# 版权

17 U.S.C. § 102(a)：*original works of authorship fixed in any tangible medium of expression*。

§ 102(b) 同一条里立刻收回去：不保护 idea、procedure、process、system、method of operation、concept、principle、discovery。CS 相关的判决，最后几乎都回到这句。

> [!EXAMPLE]+ 同一句在软件里怎么拆
>
> 排序算法的思想 → idea，§ 102(b)。某段具体 quicksort 实现 → expression。菜单如何组织功能 → *Lotus* 写成 method of operation。API 声明代码 → *Google v. Oracle* 假定可版权，按 fair use 放走，copyrightability 本身没钉。训练学到的统计规律更接近 idea / 事实；复制训练语料是另一根 § 106。

§ 106 给的是一束权利，不是“这作品归某人”。复制、制备演绎作品、发行、公开表演、公开展示。后面争的常常不是“有没有版权”，是碰到了哪一根。训练制作复制件，走复制权。输出长得很像，可能走复制，也可能走演绎。把角色图当成产品功能公开展示，还要看 display。

版权因创作产生，不必登记。美国 1989 年加入伯尔尼之后，notice 也不是成立条件。但要在联邦法院起诉，须登记（*Fourth Estate v. Wall-Street.com*, 2019）。statutory damages 和律师费，一般也要侵权开始前登过记。赔偿数字在 AI 诉讼里不是细节：每部作品通常 $750–$30,000，故意侵权可到 $150,000。语料规模一乘，生存威胁先于法理。

期限：1978 年之后自然人创作，life+70。雇佣作品、匿名、化名：发表起 95 年或创作起 120 年，取短。1998 年 Copyright Term Extension Act 再延 20 年，*Eldred v. Ashcroft*（2003）维持。limited Times 还在，只是很长。

发行权有 first sale / 用尽。正版实体书买来可以再卖。*Kirtsaeng v. John Wiley*（2013）把这层用到合法进口的教材上。数字复制不是同一件事：多出来的那份拷贝，不是“卖掉手里那一本”。*Capitol Records v. ReDigi*（2d Cir. 2018）对二手数字音乐很硬。*Bartz* 里 Anthropic 后来改买二手书、拆掉、扫描，物理上消灭了那一本，硬盘上留下一份新的。Alsup 把“买来拆了做数字图书馆”和“拿这份拷贝去训练”分开写，训练可以 fair use，图书馆那一层当时还留下事实争议。First sale 救的是那本纸书的流转，不是扫描件。

<br>

## 独创

*Feist Publications v. Rural Telephone Service*, 499 U.S. 340 (1991)。Rural 是堪萨斯一家电话公司，白页按姓氏排列。Feist 编区域号簿，Rural 不授权，Feist 照抄，连几个虚构条目都抄进去了——access 因此很好证。

最高法院拒绝 sweat of the brow。事实不受保护。汇编要有 original selection, coordination, or arrangement。按字母排电话号码，创造性为零。O'Connor 写 originality 只要求独立创作加 a modicum of creativity，门槛极低，但不是零。

所以：

- 测量、新闻事件、判决原文、股票价格：不受版权
- 对事实的选择、编排、注释：可能受保护，范围很窄
- 小说、插画、大量代码：核心区

*Thomson Reuters v. Ross* 后来打的就是这层。Westlaw headnotes 不是判决本身，是编辑对判决的归纳。Delaware 地区法院 2025 年认定这层有 originality。法律意见公共，headnotes 不一定公共。数据工作里“我花了很多时间”从来不够。

<br>

## 思想

*Baker v. Selden*, 101 U.S. 99 (1879)。Selden 写书讲一套记账法，附空白表格。Baker 用了类似表格。版权保护书里的说明文字，不保护记账方法。方法要排他，走专利。

Learned Hand 在 *Nichols v. Universal Pictures*（2d Cir. 1930）里把这层写成 abstractions test。从具体对白往上抽：情节、角色类型、主题、idea。抽到某一层，版权退出。Hand 的原话是没人能精确划出那条边界，以后也不会有人能。软件、模型、prompt，后来都在这把梯子上爬。

Merger：某种 idea 只有极少几种写法，表达和思想合并，版权退。否则等于用版权垄断思想。Scènes à faire：西部片的酒吧、代码里的常规循环和错误处理，保护很弱。独立创作出相似作品，不侵权——这是和专利最关键的差别。Patent 挡独立发明。Copyright 只挡 copying。

Plagiarism 是学术伦理。Infringement 是法定权利。抄了没版权的事实，可能是抄袭，不是侵权。反过来，改写了仍可能侵权。两套词不要混着用。

*Selle v. Gibb*, 741 F.2d 896 (7th Cir. 1984)。Bee Gees《How Deep Is Your Love》。侵权要 copying。没有 access 的证据，只靠两首歌听起来像，不够。后来软件和模型输出争 substantial similarity，仍要先过 copying 这一关：有没有接触、是不是独立生成了相似的东西。

Substantial similarity 本身也不是一个测试。普通观察者会不会觉得这是抄的；更复杂的作品会拆 extrinsic（可客观比较的结构、保护范围）和 intrinsic（普通人的整体印象）。滤掉 idea、公有领域、常规写法之后再比。模型输出争的常常是：这段是 memorization，还是同一主题下的独立生成。NYT 展示过近乎逐字的文章，那是前者。某个角色的姿势、配色“有点像”，可能到不了。风格几乎不受保护，可识别角色、独特情节、大段原文很强。分析输出侵权，先把对象放到这根尺子上，不要停在“感觉像”。

<br>

## 软件

字面抄源码最好认。难的是结构、接口、菜单、API。15.628J 把 *Baker* 和 *Lotus* 放进同一周，就是把“感觉像抄”拆开。

*Whelan v. Jaslow*（3d Cir. 1986）曾经把程序的 structure, sequence, organization 扩得很宽，几乎把功能目的当成 expression。*Computer Associates v. Altai*, 982 F.2d 693 (2d Cir. 1992）把它压回去。Abstraction–filtration–comparison：

1. 把程序抽象到不同层：源码、目标码、模块、算法、功能
2. 滤掉 idea、公有领域、效率强制的写法、外部因素（硬件、协议、兼容需求）
3. 只比较剩下的 protectable expression

滤完之后能主张的，往往比权利人想的窄。效率强制很关键：某种排序在某种数据结构下只能那样写，merger 直接上来。

*Lotus v. Borland*, 49 F.3d 807 (1st Cir. 1995)。Borland 的 Quattro Pro 做了 1-2-3 兼容模式，菜单命令层级几乎照搬，好让会计不用重新学。第一巡回认定菜单是 method of operation，§ 102(b) 不管。用户已经学会的操作方式，不能用版权锁死。最高法院 4–4，维持，没有意见。所以这还是第一巡回的法，不是全国终局，但影响很大。CLI、快捷键、配置格式、很多“兼容层”，分析时都还在用这条直觉。

反向工程。*Sega v. Accolade*, 977 F.2d 1510 (9th Cir. 1992)。Accolade 要给 Genesis 做未经授权的游戏，反汇编中间复制了整份代码。第九巡回：为了读到不受保护的功能、做出兼容，中间整份复制可以是 fair use。功能本身不受版权。为了看见功能而做的拷贝，和把表达拿去替代原作市场，不是一类用途。Agent 读网页、安全研究反编译、为了接 API 而扒文档，经常要回到这里。注意：DMCA § 1201 反规避是另一条诉因，fair use 过了，绕过技术保护措施仍可能挂。

<br>

## 接口

*Google v. Oracle*, 593 U.S. 1 (2021)。Google 为 Android 复制了约 11,500 行 Java SE API 的 declaring code，涉及 37 个 package。Implementing code 自己重写。地区法院是 William Alsup——自己会写代码，庭审里让律师当场讲 Java。Alsup 原判：API 的声明代码不可版权。联邦巡回两次反过来，认可版权。最高法院 6–2，Breyer 写意见，假定可版权，直接在 fair use 上给 Google。

理由大致是：重新实现一套移动平台，用途不同；复制声明是为了让程序员把已经会的技能带走；Android 并不在同一市场上替代 Java SE 许可。Thomas 异议，认为这就是拿了 Java 的心脏。

Copyrightability of APIs 被绕开了。接口到底能不能被版权盖住，仍不是一句终局。Fair use 是本案的出口，不是“API 无版权”。

同一个 Alsup，2025 年写 *Bartz v. Anthropic*。对“复制别人的结构、换一种用途”有多宽容，这个法庭已经有过一次。Oracle 地区阶段的两次意见（API 不可版权；fair use）都被联邦巡回推翻，最后是最高法院用 fair use 把 Google 放走。读 *Bartz* 时，Alsup 的脾气和这段前史都在。

<br>

## 合理

17 U.S.C. § 107。四个因素：

1. purpose and character（商业性、是否 transformative）
2. nature of the copyrighted work（事实性 vs 创造性；已发表 vs 未发表）
3. amount and substantiality（数量，以及有没有取走 heart）
4. effect upon the potential market

不是四项打分、三比一就赢。第一项和第四项通常最重，但互相拖。

*Campbell v. Acuff-Rose*, 510 U.S. 569 (1994)。2 Live Crew 讽拟 Roy Orbison 的《Oh, Pretty Woman》。商业性不再一票否决。Parody 必须借用足够的原作，公众才能听出在讽什么，第三因素因此让步。Transformative 从此变成第一因素的中心词，也被用滥。

*Andy Warhol Foundation v. Goldsmith*, 598 U.S. 508 (2023) 把这个词收紧。Goldsmith 给 Prince 拍了照片，Vanity Fair 1984 年授权 Warhol 做一次丝网。Prince 去世后，Vanity Fair 的母公司 Condé Nast 又去授权 Warhol 基金会那张橙色 Prince，当杂志封面插图。最高法院看的不是“Warhol 的艺术是不是变革性的”，是这一次商业许可：和 Goldsmith 照片的授权市场，目的实质相同。Sotomayor 写 majority。Kagan 异议很长。

<span style="background:#ffe58f">Transformative 必须落在被诉的那一次使用上。新技术、新中间物、新美学，都不自动等于新用途。</span>

读训练数据判决时，这件案子是背景。不能因为 LLM 在技术上把 token 变成权重，就宣布第一因素结束。Alsup 在 *Bartz* 里把训练写得很满；Chhabria 在 *Kadrey* 里把第四因素拉回来。两个人都读过 *Warhol*。

课堂复印。*Cambridge University Press v. Becker*（N.D. Ga. 2012 及后续 Eleventh Circuit）是 15.628J Session 5 用来练 § 107 的样本：大学电子课件摘多少算合理。和生成模型无关，但把“数量”“市场”“教育目的”怎么映射到四个因素，写得很慢。值得当例题，不要当 AI 训练的类比。

互联网平台上的版权，6.4590 Week 7 才会和 15.628J 的课堂复印分开。*A&M Records v. Napster*（9th Cir. 2001）集中式索引，间接责任好证。*Grokster* 改成去中心化之后，改打 inducement。*Perfect 10 v. Amazon*（9th Cir. 2007）缩略图检索，fair use 方向更接近后来的 Google Books：用途是检索，展示是碎片。Agent 浏览缓存、搜索引擎快照，分析时更靠近这一支，而不是预训练吃整本书。

<br>

## 间接

直接侵权之外三条：

**Contributory**：知道 + 实质性帮助。

**Vicarious**：有权且有能力控制 + 直接经济利益。

**Inducement**：以促进侵权为目的而散布工具。

*Sony v. Universal*（1984），Betamax。设备具有 substantial noninfringing uses，制造商不因用户可能用来录像而承担间接责任。*MGM v. Grokster*, 545 U.S. 913 (2005)。P2P 软件也可以有大量非侵权用途，但 Grokster 的广告、内部文件表明目的就是接替 Napster。Inducement 成立。能力中立不够，还要看怎么推。

平台版权责任大量走 DMCA § 512：notice-and-takedown，符合条件的服务提供者有安全港。§ 1201 反规避是独立的。§ 1202 管版权管理信息（CMI）移除。*Kadrey* 里对 Llama 训练的 CMI 主张，随 fair use 成立一起倒下——没有底层侵权，移除 CMI 就不构成“为了隐匿侵权”。

Grokster 那套结构，后面会套到模型、agent、爬虫工具上。模型能吐出侵权图，不等于提供模型的人一定诱导。把“生成某个角色”做成卖点、首页展示、广告词，诱导会好证得多。*Disney v. Midjourney* 的诉状就是往这走。

<br>

# 专利

四条腿：Constitution → Congress（Title 35）→ USPTO → 法院。专利上诉集中到 Federal Circuit，再上最高法院。Design patent 管外观，plant patent 管植物。CS 日常几乎都是 utility patent。

AIA（2011，主要条款 2013 年生效）之后，美国是 first inventor to file。15.628J 同时发了旧 § 102 和新 § 102，读案例先看发明日还是申请日。欧洲基本没有美国这种一年 grace period。自己开会、自己预印本、自己把 demo 挂出去，在美国可能还能救，在 EPO 常常已经公开。

授权不是终点。AIA 同时造了 PTAB 上的 inter partes review，用 prior art 打 § 102 / § 103，比地区法院便宜、快。NPE 诉讼策略因此改过一次。15.628J 那年 IPR 刚开始，现在已经是专利实务的日常。

<br>

## 外观

*Apple v. Samsung* 不是一件单纯的“圆角矩形归谁”案。长期诉讼同时出现 utility patents、design patents 和 trade dress；Stanford CS202 用它讲 smartphone design / patent，最高法院真正处理的是 design-patent damages。

陪审团认定 Samsung 手机侵犯 Apple 关于圆角正面、边框和图标网格的外观设计专利，并把 Samsung 涉案手机的全部利润计入 § 289。最高法院在 *Samsung Electronics Co. v. Apple Inc.*, 580 U.S. 53 (2016) 一致裁定：多组件产品里的 “article of manufacture” 不一定是消费者买到的整机，也可以是其中一个组件，即使该组件没有单独出售。法院撤销发回，但没有决定本案应把整部手机、屏幕还是正面外壳当作 article，也没有给出完整识别测试。

> [!EXAMPLE]+ 外观专利赔偿
>
> Utility patent 通常围绕 lost profits / reasonable royalty。Design patent 的 35 U.S.C. § 289 可以追相关 article of manufacture 的 total profit。争点因此不是只看“抄了多少设计”，还要先认定利润对应的 article 是整机还是组件。复杂产品把这一步放大。

<br>

## 资格

35 U.S.C. § 101：*any new and useful process, machine, manufacture, or composition of matter*。司法例外三条：laws of nature、natural phenomena、abstract ideas。例外是法官造的，成文法没写。

*Diamond v. Diehr*, 450 U.S. 175 (1981)。用 Arrhenius 方程 + 计算机控制橡胶硫化时间。Claim 作为整体是工业过程，不是“数学公式本身”。软件专利的开门案例。开门不等于什么软件都能进。

*State Street Bank*, 149 F.3d 1368 (Fed. Cir. 1998)。*useful, concrete and tangible result*。商业方法一度大开，金融专利、“用计算机做 X”的 claim 涌进来。

*Bilski v. Kappos*, 561 U.S. 593 (2010)。对冲商品价格风险的方法，abstract idea。Machine-or-transformation 是有用线索，不是唯一标准。*State Street* 的宽松口径被关上。15.628J Session 11 读到这里，Short Exercise 2 就是 *Bilski*。

Session 12 是生物 / 医疗那条司法例外，不是课后补丁。

*Mayo Collaborative Services v. Prometheus*, 566 U.S. 66 (2012)。代谢物浓度和药效的关系是自然定律。加上“告诉医生去测量、调整剂量”这类常规步骤，不够。Breyer 写。医疗诊断专利被削了一层。

*Association for Molecular Pathology v. Myriad Genetics*, 569 U.S. 576 (2013)。分离出的自然 DNA 片段不可专利；cDNA 可以，因为自然界不存在。自然产物 vs 人为干预。和软件里 abstract idea vs 具体应用是同一类问题：自然里已有的，加常规步骤，过不了。*Myriad* 判在 2013 年 6 月，已经贴学期末，OCW readings 仍收了进去。

课后来不及收的是 *Alice*。

*Alice Corp. v. CLS Bank*, 573 U.S. 208 (2014)。中间清算风险的商业方法，放到计算机上。两步：

```text
Step 1  claim 是否指向司法例外（尤其是 abstract idea）？
        ↓ Yes
Step 2  其余部分有没有 inventive concept，
        把 claim 转成对该 idea 的可专利应用？
        只把 idea 放到通用计算机上，不够。
```

> [!INFO]+ Alice 两步
>
> Step 1 问 claim 是不是指向 abstract idea / 自然定律 / 自然现象。Step 2 问剩下的部分有没有 inventive concept。只把 idea 放到通用计算机上，不够。*Enfish*（自引用数据表）、*McRO*（自动口型规则）说明：改善计算机本身功能的软件，有可能在 Step 1 就不是 abstract idea。和 novelty、obviousness 不是一层过滤器。

Federal Circuit 后来在 *Enfish*（2016）里承认：改善计算机本身功能的软件——那件是自引用的数据表结构——可以在 Step 1 就不是 abstract idea。*McRO*（2016）对自动口型动画的规则，也过了。实践中 § 101 变成软件专利最不稳定的过滤器。USPTO 多次出 eligibility guidance，口径来回拉。立法改 § 101 从 2010 年代吵到现在，没落地。

算法能不能专利，不是 yes/no。纯数学、纯商业方法、“用计算机做 X”，§ 101 很容易挂。具体技术环境里的具体改进——减少延迟、改存储结构、改训练时的硬件调度——仍可能过，但 claim 必须写出那一层，不能停在 “a neural network configured to…”。

PHOSITA 在 ML 里也不好定。普通水平是读过 Vaswani 2017、会调现成框架，还是会手推 kernel？Claim 写“一种 transformer，其特征在于多了某层残差”，KSR 下很像已知元件的可预见组合。真正难被 § 103 打穿的，往往是效果出乎意料，或者解决了长期有人试、一直没做成的具体工程问题。商业成功可以当次要考虑，营销成功不算。

<br>

## 新颖

§ 102。Prior art 不是“以前有人做过类似的”。是法定范畴里、在法定时间点之前、以法定方式向公众公开的信息。公开方式：printed publication、public use、on sale、otherwise available to the public。一篇没人看过但可检索的论文，可能已经是 printed publication。内部演示不一定是。

*Pfaff v. Wells*, 525 U.S. 55 (1998)。发明已经 ready for patenting，且进入商业销售或要约，即使实物还没做出来，也可能毁掉新颖性。软件和模型：早期商业演示、API 预售、把 checkpoint 卖给客户，都可能踩。实验室默认“先发论文再考虑专利”，在 first-to-file 世界里是在跟时间对赌。

15.628J 的 *Structural Rubber v. Park Rubber* 是练 prior art 阅读的作业，不需要当规则记，需要当方法记：日期、公开方式、是不是同一发明。

<br>

## 显而易见

§ 103。没有一份 prior art 完整公开这项发明，仍可能对 PHOSITA 显而易见而无效。

*Graham v. John Deere*, 383 U.S. 1 (1966)：prior art 范围与内容；与 claim 的差异；相关领域普通技术水平；次要考虑（商业成功、长期未解决问题、他人失败）。

*KSR v. Teleflex*, 550 U.S. 398 (2007) 打破刚性的 teaching-suggestion-motivation。PHOSITA 有普通创造力。已知元件的常规组合可以显而易见。对 ML 专利尤其紧：attention、residual、某种 tokenizer，拼起来若只是可预见的组合，§ 103 很容易打穿。能站住的往往是非显而易见的技术效果，不是模块清单。

次要考虑不是装饰。长期有人试、一直没做成、商业突然成功，可以往回推“其实没那么显然”。反过来，商业成功也可能只是营销。

<br>

## 权利要求

§ 112。Specification 必须 enable 本领域技术人员实施，写出 written description，claim 要明确。Patent 的社会契约在披露。写得很玄、实施不了，无效。写得很具体、claim 却想盖住所有神经网络，也无效。

*O'Reilly v. Morse*, 56 U.S. 62 (1853)。Morse 的 Claim 8 试图垄断一切用电磁力远距离通信的手段，远超出他实际披露的电报装置。上位 claim 没有对应披露。今天 “a neural network configured to diagnose…” 这类写到天边的 claim，影子还在。

权利边界是 claim，不是说明书故事。说明书、附图、实施例用来解释 claim。Independent claim 最宽，dependent claim 往里加 limitation，范围变窄、有效性通常更好证。

字面侵权：被控产品满足全部 limitations（all-elements rule  /  all-limitations rule）。少一个就不字面侵权。Doctrine of equivalents 防止改个无关紧要的零件就绕开。*Warner-Jenkinson*（1997）收到逐元件对等。Prosecution history estoppel：审查时为了避开 prior art 而缩小的范围，后面不能用等价理论要回来。所以申请过程中怎么改 claim、怎么争辩，诉讼里都会被翻出来。

*Markman v. Westview Instruments*, 517 U.S. 370 (1996)。Claim construction 是法官的工作，不是陪审团的。专利诉讼前半场经常在 Markman hearing 里已经定胜负。同一个词在说明书里出现过几次、有没有定义、审查时怎么用，都会进 claim construction。

15.628J 的 patent-search assignment：拿一份刚授权的美国专利，只盯 Claim 1，四小时内找可能破坏 § 102 / § 103 的 prior art，写 2–3 页。日期必须按 prior-art 定义来，不能拿申请日之后的文献打新颖性。换成一件 CV / LLM / agent 专利做同一件事，会发现摘要写得很未来，真正卡住的是某句很具体的系统步骤。摘要几乎没有法律意义。

<br>

## 侵权

抗辩首先是无效：§ 101 / 102 / 103 / 112。然后是不侵权。还有 inequitable conduct（申请时对 USPTO 隐瞒关键 prior art，门槛高）、laches 等。

*Roche v. Bolar*, 733 F.2d 858 (Fed. Cir. 1984)。仿制药实验使用仍是侵权。国会随后用 Hatch-Waxman 开出 Bolar exemption。实验使用例外在美国极窄。“做研究所以不侵权”不是美国专利的默认。版权的 fair use、专利的 experimental use，两套，不要串。

*Bowman v. Monsanto*, 569 U.S. 847 (2013)。农民买了抗除草剂大豆，种一代可以。把收获再种，不行。专利用尽管的是售出的那一件，不是复制下一代。数字世界里的对应很硬：买到一份权重，不等于买到复制、再训练、再分发的权利。许可证写什么，才是什么。

<br>

## 救济

损害赔偿：lost profits 或 reasonable royalty。*Georgia-Pacific* 列出合理许可费的一长串因素，实务里是谈判和专家证言的战场。故意侵权可能倍计。

禁令曾经几乎自动。*eBay v. MercExchange*, 547 U.S. 388 (2006) 废掉这个惯例，改回普通法四要素：不可挽回损害、法律救济不足、衡平法权衡、公共利益。NPE 拿着专利要挟整个产业的能力被削了一刀。经营实体仍比较容易拿到禁令，纯粹许可商难得多。15.628J Short Exercise 3 就是拆这四个要素。

对生成模型公司，版权侧的 statutory damages 乘语料规模，往往比专利 royalty 更先构成生存威胁。*Bartz* 后来能和解到这个量级，背景是这个乘法，不只是 Alsup 那句 *spectacularly so*。

<br>

# 秘密

州法走 Uniform Trade Secrets Act。联邦 2016 年 DTSA，结构相近：信息因不为公众所知而有独立经济价值，权利人采取了合理保密措施。

Misappropriation：不正当获取、披露、使用。独立发现不构成。合法获得实物之后的 reverse engineering，一般不构成。Employee mobility 是主战场。前雇员脑子里带走的，哪些是一般技能，哪些是雇主的秘密。

*Wexler v. Greenberg*, 399 Pa. 569 (1960)。化学配方那类。只把“在这家公司做过”当成秘密，雇佣市场会停。一般技能、经验、记忆，切出去。

*Waymo v. Uber* 是 Stanford CS202 明列的 employee movement 案。Waymo 主张前工程师 Anthony Levandowski 离职前下载逾 14,000 份自动驾驶与 LiDAR 文件，随后创办 Otto；Uber 收购 Otto，并让 Levandowski 负责自动驾驶项目。民事案在 2018 年陪审审理第五天和解：Waymo 获得 Uber 0.34% 股权，当时约值 2.45 亿美元；Uber 同意确保其硬件和软件不纳入 Waymo 机密信息。和解没有逐份裁定哪些文件构成商业秘密，也不是法院认定 Uber 实际使用了全部文件。

Levandowski 的刑事案是另一条程序。他不是 Waymo 对 Uber 民事案的被告；联邦检方后来单独起诉。2020 年他对一项 trade-secret theft 认罪，涉及一份 “Chauffeur Weekly Update”，被判 18 个月监禁。CS202 目录写 “civil and criminal trade secret theft”，指的是这组相连事实，不是说一份判决同时处理民刑责任。

离职带 disk、repo、数据，是 trade-secret 诉讼的标准事实模式。模型实验室对应的是未公开权重、训练数据、配比、过滤规则、内部评测集、系统提示。已发论文的算法、公开 benchmark、开源框架里的常规实现，通常不构成。

合理保密措施要真做：权限、NDA、访问日志、不把生产权重扔进公共网盘。措施可以不完美，完全没有就很难主张。Open-weight 发布，权重上的商业秘密基本结束。还能主张的是没开源的数据、后训练、产品层。

反向：模型抽取、蒸馏、爬内部 API，可能同时撞版权、合同和商业秘密。三套事实可以是同一串 HTTP 请求。

<br>

# 商标

Lanham Act。不管创意聪不聪明，管消费者会不会以为来源有关联。可以无限续展，前提是还在用，还没变成 generic。Aspirin、Thermos、Escalator 都是教训。功能性设计走专利，不能用商标把功能锁成永久。

混淆是多因素：标志相似度、商品接近程度、实际混淆证据、被告意图、消费者注意程度。*Lois Sportswear v. Levi Strauss*, 799 F.2d 867 (2d Cir. 1986)，后袋缝线那种商业外观。15.628J 拿来当入门，不是商标法全貌。

生成模型吐出带 logo、角色外形、包装的图，可能同时走版权（角色、美术）和商标（来源混淆、假冒）。*Disney v. Midjourney* 主诉是版权，商标逻辑在旁边：用户会不会以为这是迪士尼的官方生成器。模型名、公司名自身的商标冲突是另一层，和训练无关，和起名有关。

<br>

# 归属

## 职务

职务发明靠 assignment agreement，不是靠“觉得应该归组里”。美国版权有 statutory work made for hire，专利没有同等自动归属。*Stanford v. Roche*（2011）：Bayh-Dole 并不自动把联邦资助发明的权利归大学，还是要靠发明人的书面转让。斯坦福当年那份表格的措辞，和 Roche 先前拿到的转让，谁在先，谁赢。实验室入职文件不是仪式。

共同发明：对 claim 有实质性贡献的自然人才是 joint inventor。漏列、错列可以毁专利。共同作者是另一套。Copyright 里 jointly authored work 原则上每个人可以单独许可，要结算。专利共有规则不同，未经其他共有人同意的许可，在不同法域后果不一样。论文作者列表和发明人列表对齐不了是常态，不是疏忽一句能混过去。

## 流动

Non-compete。加州长期基本不可执行，人才流动本身被当成创新政策。联邦 FTC 2024 年试图全国禁止，随后在法院受阻。实际约束更多靠 NDA、invention assignment、customer non-solicit。去相邻实验室继续做同一条产品线，合同问题通常不是“竞业”四个字，是带走了什么。

## 开源

GPL、MIT、Apache 2.0 管的是软件作品。模型许可是另一张纸。GPL 的 copyleft 以源码衍生为前提。权重、LoRA、checkpoint 算不算 derivative work，几乎没有判决。研究里把“模型开源”理解成“和 Linux 一样”，权利处分会读错。Creative Commons 不管专利，也不自动覆盖训练。CC BY-SA 的传播条件，和把语料送进闭源模型，中间那层法院还没系统回答。

<br>

# 信息政策

Internet 不是一件产品，是网络、市场、社交空间、信息系统和政治机构叠在一起。Lessig 那句 code is law 是分析起点：架构先决定什么行为容易发生，法律再把责任和成本压回架构。

技术设计改变用户能做什么。用户行为改变法律要不要出手。法律再逼架构改。加密默认开，执法换一种理由要后门。平台把转发做成一键，诽谤和版权的间接责任理论全部重写。推荐算法把“展示谁的内容”变成平台自己的编辑动作，§ 230 还管不管，变成 2020 年代的问题。

政策分析不满足于找法条得到答案。基本顺序是：

技术事实 → 利益相关方 → 现有规则 → 备选方案 → 方案会怎样改架构 → 社会后果 → 建议。

建议要落到谁执行、用什么工具、失败了会怎样。

<br>

## 言论

*Reno v. ACLU*, 521 U.S. 844 (1997)。通讯端正法里过宽的不雅言论条款，按广播那套管互联网。最高法院按印刷品给 First Amendment 保护，条款推翻。同一部 Communications Decency Act 活下来的是 § 230。

互联网从一开始就不是“广播，所以政府可以严格管内容”。这是后面所有平台争论的底板。

<br>

## 平台

47 U.S.C. § 230(c)(1)：交互式计算机服务不因其用户的内容而被当成 publisher 或 speaker。§ 230(c)(2) 给善意审核一个安全港。*Zeran v. AOL*（4th Cir. 1997）把这层读得很厚：就算平台已经收到通知，知道某条帖子是诽谤，仍可能受保护。平台治理因此主要是私人治理：ToS、推荐、审核团队。

政府若强迫平台删或留某些合法言论，会撞第一修正案。政府若把平台当成发布者追责，会撞 § 230。两边同时存在，所以争论永远是：这是私人审核，还是国家作用。

*Gonzalez v. Google*（2023）本来可能重写推荐算法还受不受 230 保护。最高法院没接这道题，跟着 *Twitter v. Taamneh* 把相关责任问题先绕开。推荐是不是平台自己的 speech，仍悬着。

生成式 AI 把结构拧弯。模型输出在多大程度上是用户的 speech，多大程度上是公司的 speech，直接决定 230 能不能用。把 ChatGPT 的回答当成“用户发布的内容”，和把推荐流里的推文当成用户内容，不是同一件事。用户只打了一句提示，输出由模型参数和公司的安全策略共同决定。2024 年以后的诉讼卡在这点上，没有 *Reno* 级别的终局。

深度伪造、虚假信息，6.4590 放在 Week 8 内容审核和后来的 AI 周，不放在纯宪法课。技术能力先出现，归责规则后补。Copyright Office Part 1 建议给个人形象单独立法，不硬塞版权法。中国走标识和禁止删除标识。两边都承认：版权不是处理“这张脸像谁”的主工具。

<br>

## 搜查

Fourth Amendment 管政府搜查扣押，不管公司隐私政策。两套问题并成一句“隐私”，后面会写乱。

第三人原则。*Smith v. Maryland*（1979）：电话号码交给电话公司，政府用 pen register 调取，不构成 search。数字时代几乎所有东西都交给第三方。*United States v. Jones*（2012），车底装 GPS 长期追踪。*Carpenter v. United States*, 585 U.S. 296 (2018)，长期 CSLI（基站位置记录）要令状。Roberts 写：数字时代第三人原则不能无限延伸。6.4590 Week 4 专门卡在位置隐私和第三人原则上：交给手机公司的位置，还是不是 *Smith* 那种“自愿交给第三方”。多数意见认定调取 7 天的 CSLI 已经构成 search；更短的窗口有没有例外，故意没划。agent 连续上报坐标，会落在这条光谱的哪一侧，还没有同级别的判决。

位置数据、云端对话日志、agent 的浏览记录，落在这条光谱上。公司依用户协议收集，是合同、州隐私法、消费者法。政府向公司调取，是第四修正案和 Stored Communications Act。同一份 ChatGPT 日志，两次法律评价。

CFAA 管未经授权的访问。*Van Buren*（2021）把“违反使用政策就算黑客”收窄。警察查了一次他有权访问、但不该查的数据库，不构成 CFAA 意义上的超授权。

*hiQ Labs v. LinkedIn* 是 Stanford CS202 明列的 scraping / CFAA 案。hiQ 抓取任何浏览器都能看到的公开职业资料，用于员工流失与技能分析。LinkedIn 发 cease-and-desist，并用技术措施阻断。第九巡回 2022 年在 *Van Buren* 发回后再次维持 preliminary injunction：hiQ 至少提出了严重问题，公开资料没有 CFAA 所要求的那道授权 gate，访问它一般不像 “without authorization”。

这不是对所有 scraping 的终局豁免。判决阶段只是 preliminary injunction，也没有解决合同、trespass、隐私、版权等全部请求。案件后来以 consent judgment 收尾：hiQ 同意 50 万美元判决、停止被禁止的抓取并删除相关数据和工具。该和解不推翻第九巡回关于公开页面的 CFAA 分析，也不产生先例；同时，hiQ 使用假账号进入 password-protected pages 的行为与纯公开页面抓取不能混为一谈。

所以 Agent 抓网页至少要分三层：公开且无登录；违反网站条款但没有绕过访问门；伪造账号、绕过登录或技术屏障。CFAA、合同、版权、隐私可能给出四个不同答案。robots.txt 也不自动等于 CFAA 的授权门。漏洞报告同样可能既是研究、又是 ToS 违约、又涉及未经授权访问。

<br>

## 隐私

美国没有统一联邦隐私法典。HIPAA、FERPA、COPPA、加州 CCPA/CPRA、州生物识别法，拼起来。欧盟 GDPR 是权利框架：合法基础、目的限制、最小化、跨境、自动化决策。中国是网安法 + 数安法 + 个保法。训练把“处理”放大成“写进参数”，删除权和遗忘权在技术上变得很别扭。隐私法和版权法在这里碰到同一个硬点：模型不是文件夹，删一条语料不等于模型里没了。

<br>

## 跨境

User 在一国，server 在另一国，公司在第三国，weights / logs / fine-tune data 多处。Which law applies 不是学期论文题目。GDPR 的效果原则、中国对境内提供服务的属地抓手、美国长臂，会叠在同一个请求上。出口管制另算：先进芯片、某些封闭权重。6.4590 把 trade policy 和 international affairs 单列，因为架构决策经常在布鲁塞尔、华盛顿、北京同时被改写。同一套开源发布，三个市场的合规动作不一样。

<br>

# 人工智能

先拆对象。

“某模型抄了我的东西”至少四件：

1. 训练时制作复制件
2. 把盗版库当永久图书馆（存，不一定训）
3. 输出与作品 substantial similarity（复述、角色、逐页替代）
4. 把角色或商标当成产品功能来卖

四件不共用抗辩。Fair use 可能盖住 1，盖不住 2。3 要过 copying + similarity。4 更像 Grokster 加直接侵权。

分析顺序可以很慢：

Copied or produced what？idea / 事实 / expression / 代码 / 整本书 / 权重 / 输出 / 角色 / CMI。

哪一根 § 106。谁是 actor：实验室、云厂商、开源发布者、下游应用、用户、agent。抗辩是 fair use、许可、安全港，还是根本没有适格作者。

美国没有国会层面的 AI 训练例外。战场在 § 107。到 2026 年秋，地区法院已经给出三条不能互相化约的线。都不是最高法院。第三巡回听过 *Ross* 的辩论，还没下判。

<br>

> [!WARNING]+ 2025 年北加州两份意见不是规则
>
> *Bartz*、*Kadrey* 都是 N.D. Cal. summary judgment，且都写了本案记录。Chhabria 自己说这份裁定不等于 Meta 对所有权利人都合法。*Ross* 是 Delaware、非生成式、直接竞品，结论相反。引的时候把法院、用途、来源、有没有输出主张写全。

<br>

训练诉讼出现之前，最像的先例是整本扫描。*Authors Guild v. Google*, 804 F.3d 202 (2d Cir. 2015)。Google 未经许可扫描数千万册，做全文检索和 snippet。Leval 写：用途 highly transformative；整本复制对检索是必要的；snippet 不构成原作的实质性替代；商业动机不自动击败 fair use。作者主张的“检索许可市场”，法院不认。*HathiTrust* 那条图书馆扫描线更早一点，结构相近。Alsup 在 *Bartz* 里把训练写成更强的 transform，用的就是这条家族的语言。差别也硬：Google Books 给用户看的是检索和碎片，不生成新的书；LLM 把书压进参数之后，可以写出无数二手文本。Chhabria 咬的就是这点。

<br>

## 训练

### 竞争

*Thomson Reuters v. Ross Intelligence*（D. Del. 2025）。Ross 不是生成式模型，是法律检索工具，想做 Westlaw 的竞品。Vendor 用 Westlaw headnotes 做 Bulk Memos 当训练数据。法官是第三巡回的 Stephanos Bibas，designation 到 Delaware 坐地区法院——所以这意见写得不像普通 district court。

Headnotes 有 originality。复制成立。Fair use 不成立：商业、非变革性、直接做同一功能的竞品。第二、第三因素偏向 Ross（headnotes 保护窄，最终用户看不见那些 memo），压不过第一和第四。Bibas 自己写 *the questions here are hard*，准许中间上诉。第三巡回 2026 年 6 月 11 日开庭辩论，Ross 被反复追问：用途和 Westlaw 对用户来说是不是同一件事。至 2026 年 9 月尚未下判。这是目前唯一走进上诉审的训练 / 合理使用案件。

生成式 / 非生成式、是否直接替代原产品，会把第一因素和第四因素一起拖走。不能把所有 training 当成一类用途。Ross 更像是用别人的表达去造一个功能相同的检索器。*Bartz* 里 Claude 不是在卖书。

<br>

### 来源

*Bartz v. Anthropic*（N.D. Cal. June 23, 2025, Alsup, J.）。Andrea Bartz、Charles Graeber、Kirk Wallace Johnson。Anthropic 先从 LibGen 一类影子库下了数百万册，后来改策略，买二手书、拆掉、扫描。原告打的是训练和中央图书馆，没有主张 Claude 输出大段复述。

Alsup 切开两件事。合法获得的书拿去训练，和把盗版书存进永久中央图书馆，不是一次使用。

训练这一支按 § 107 写得很满。第一因素：*transformative—spectacularly so*。把 token 压进参数、学的是统计规律，不是把书拿去给用户读。类比是阅读和记忆：为了每次想起一本书、每次用新的方式写新东西就付一次钱，unthinkable。商业性在，但没压过变革性。第二因素偏向作者（小说等创造性作品），这项一向权重低。第三因素：整本复制，但对训练是 reasonably necessary，整本不自动致命——*Campbell*、*Sega* 都出现过这种结构。第四因素：保护的是替代原作市场，不是任何一种竞争。原告没主张 Claude 会逐页替代，Anthropic 有防复述。作者主张的“训练许可市场”，法院不认为是 Copyright Act 保证他们独占的市场。如果 fair use 成立，就不能靠“本来可以收训练费”反过来把第四因素赢回去。这点 Chhabria 两天后会骂。

盗版库这一支不是 fair use。后续即使不拿来训练，也不能洗白。没有“打算用于变革性用途，所以先盗版没关系”。Alsup 对任何后续 fair use 能否正当化盗版下载，写得很怀疑。买来拆掉扫描的那一层，当时还留下事实争议，没有在 summary judgment 上和训练一起送走。

盗版库这一支后来按 15 亿美元和解。作品清单约 482,000 部，单部粗算 3,000 美元量级，销毁盗版数据集。2026 年 7 月 20 日终局批准，律师费被压到约 6.8%（不是常见的 20% 开口）。和解管过去的复制和库，不管输出。法定赔偿上限按件相乘，理论暴露可以到公司存亡。和解因此不是对“训练是否 fair use”的终局承认，是对盗版库风险的定价。Anthropic 曾想上诉 fair use 和集体认证，没等上诉就结了。

Alsup 也是 *Oracle v. Google* 的地区法官。对“复制结构、换用途”已经宽容过一次。读 *Bartz* 时这不是八卦。

> [!EXAMPLE]+ Alsup 切开的两件事
>
> 合法获得的书 → 训练 LLM：fair use，*transformative—spectacularly so*。盗版书 → 永久中央图书馆：不是 fair use，后续不拿来训练也洗不白。买来拆掉扫描的数字库，当时还留下事实争议。15 亿和解主要给后两层定了价，没把第一层送进上诉。

<br>

### 市场

*Kadrey v. Meta*（N.D. Cal. June 25, 2025, Chhabria, J.）。Richard Kadrey、Sarah Silverman、Junot Díaz 等十三位作者。Llama，Books3 一类影子库，还涉及 torrent。隔 *Bartz* 两天，同一个北加州。

训练在本案记录上被认定为 fair use，即使作品来自盗版库。和 Alsup 的分歧主要两处。盗版来源是否单独致命：Chhabria 认为训练本身仍可 fair，来源问题另说。第四因素：他认为这才是最重要的因素，Alsup 把训练比成教孩子写作，inapt。LLM 能以极低成本制造海量二手作品。Market dilution——用原作把模型喂强，再让模型淹没原作市场——理论上说得通。书籍之所以是好训练数据，恰恰支持“训练会让模型更会冲淡这些书的市场”。

但原告几乎没做这支证据。Chhabria 写得很不客气。大意是：这份裁定只说明这十三位作者打错了论点、没把对的论点做成记录，*不*等于 Meta 对所有权利人都合法。

Torrent seeding（把盗版书再分发出去）相关主张仍活着。2025 年底原告还在改诉状。DMCA CMI 主张随 fair use 一起倒下。Chhabria 2026 年 7 月拒绝向第九巡回提交中间上诉。训练问题目前没有西海岸的上诉审。

同一联邦地区、隔两天、两个法官。上诉审之前，不能把“美国允许训练”写成规则。Chhabria 等于给后面的原告留了图纸：去做稀释的证据，不要只喊许可费。

<br>

## 输出

*In re OpenAI Copyright Infringement Litigation*（S.D.N.Y. MDL 25-md-03143）。作者、新闻机构等并案。训练复制，以及 ChatGPT 输出的摘要、大纲。纽约时报那支很早就展示过近乎逐字的文章复述——这是输出轨，不是“训练是否变革性”轨。2025 年 10 月法院认为原告已经主张了陪审团可能认为实质性相似的输出，fair use 留到后面。2026 年初强制开出数千万级 output logs。事实还在 discovery。这是目前最接近双轨的大案。

*Disney v. Midjourney*（C.D. Cal.，2025 年起诉）。训练复制，以及服务持续生成、展示迪士尼 / 卢卡斯 / 漫威角色，并把这种能力当卖点。Willful、首页展示、没做过滤，诉状按诱导和直接侵权一起写。角色比“某种风格”好主张得多。风格几乎不受版权保护。可识别的角色、服装、场景，保护很强。

许可市场曾经被做出来过。2025 年 12 月 Disney 与 OpenAI 宣布：三年许可，Sora / ChatGPT Images 使用 200 多个迪士尼、皮克斯、漫威、星战角色，不含演员肖像和声音；Disney 拟投资 10 亿美元。2026 年 3 月 OpenAI 关停 Sora 消费端，交易取消，Reuters 报道钱没交割。对第四因素仍是信号——有人愿意给角色生成定价——但不能写成已经稳定存在、正在收款的市场。Midjourney 那支诉讼还在。

更早还有 *Andersen v. Stability*、Getty 那批图像案。Stable Diffusion 输出里冒出 Getty 水印，是对 memorization / 来源的直观证据，比“统计学习不是复制”的抽象辩论好懂。图像和文字的 fair use 叙事不必同一套。

<br>

## 作者

美国版权要求 human authorship。不是政策偏好，是现行解释。

*Thaler v. Perlmutter*。Stephen Thaler 的 Creativity Machine 生成《A Recent Entrance to Paradise》，铁轨伸进光里那种图。申请表上作者填机器，自己只当权利人。版权局拒登。D.D.C. Howell 法官：human authorship 是 bedrock。Work made for hire 也以存在可保护作品为前提，机器“受雇”产生不了第一步。D.C. Cir. 2025 年 3 月维持。Thaler 在行政程序里放弃了“自己因创造并操作机器而成为作者”，所以法院不处理人类辅助到什么程度才够。最高法院 2026 年 3 月 2 日拒签 certiorari。

纯机器、申请人自己也这么主张的案子，到此结束。人类辅助到哪一条线，没有被这件案子回答。

Copyright Office *Part 2: Copyrightability*（2025-01-29）：不建议改法。纯生成材料不能登记。人类对输出的可感知选择、编排、修改可以保护。Prompt 本身通常够不上，因为提示者并不控制表达细节。反复改 prompt、挑图、修图、拼成作品，保护的是人做的那一层。Part 1（2024）走数字副本 / 深度伪造，建议单独立法，不硬塞版权法。

<br>

## 发明人

*Thaler v. Vidal*, 43 F.4th 1207 (Fed. Cir. 2022)。专利发明人必须是 natural person。AIA 之后 § 100(f) 把 inventor 定义成 individual(s)，宣誓用 himself / herself。DABUS 不能列名。Federal Circuit 明确没回答：人类借助 AI 完成的发明能不能专利。

USPTO 2024 年 2 月曾用 *Pannu* 共同发明标准去筛人类有没有 significant contribution。2025 年 11 月整份指导废止。Pannu 只适用于多个自然人之间。AI 不是人，谈不上共同发明人。替代口径回到传统 conception：发明人要对完整且可实施的发明形成确定、持久的观念。AI 被写成实验设备、软件、数据库。可以用别人的服务和想法，那些来源不因此成为发明人。列发明人时不得把模型写上去。

和版权只是表面对称。版权问这件作品有没有人的表达。专利问这件 claim 有没有人完成 conception。一个人用模型扫几千个候选、自己选出并验证其中一种，和一个人按一次生成键得到一张图，不是一层评价。实验室用代码助手写实现，conception 仍要落到具体人。漏列在团队工具普及之后更容易发生。

<br>

## 比较

美国主线到此为止。其他法域解决的是同一批技术事实，但工具不同：欧盟用 TDM 例外、权利保留和 GPAI 透明度；中国以合法来源、内容责任和标识为主；日本、新加坡另设较明确的数据分析例外。

### 欧盟

欧盟不走事后 fair use。走的是例外，加上退出。

DSM Directive (EU) 2019/790。第 3 条：科研机构 TDM。第 4 条：更广的文本与数据挖掘，包括商业，但权利人可按第 4(3) 条以适当方式保留权利，实践中常被理解成机器可读 opt-out。robots.txt、网站条款、专门协议，哪些算“适当”，仍在磨。欧盟把选择权放在权利人事前，美国把判断权放在法官事后。

AI Act（Regulation 2024/1689）把版权义务接到通用模型上。第 53 条对 GPAI 提供者：技术文档；向下游系统提供者披露必要信息；建立遵守欧盟版权法的政策，包括用先进技术识别并尊重第 4(3) 条保留；按 AI Office 模板公布足够详细的训练内容摘要。开源 GPAI 在文档义务上有条件豁免，系统风险模型不行。罚则可到全球年营业额 3% 或 1500 万欧元，以高者计。

GPAI 义务 2025-08-02 开始适用。AI Office 对 GPAI 的执法权 2026-08-02。2025 年 7 月的训练内容摘要模板不要求倾倒数据集，要求覆盖预训练到后训练的主要数据、抓取来源、显著域名。2025 年 8 月之前投放市场的模型，摘要可延到 2027-08-02。行为准则是证明合规的路径，不是改写实体法。2025–2026 年间还有简化高风险期限的政治讨论，附件 III / 附件 I 的日期被拉动过，引的时候看最新文本。

同一套权重，在欧盟要处理 opt-out 和摘要，在美国要准备 fair use 诉讼记录。合规动作不是翻译关系。

<br>

### 中国

生成式服务的监管是内容治理驱动。知识产权是其中一条，不是唯一主轴。先看清适用范围。

《互联网信息服务深度合成管理规定》，2023-01-10 施行。深度合成覆盖文本、图像、音频、视频、虚拟场景。第 16 条隐式标识。第 17 条对可能导致混淆误认的服务做显著显式标识。禁止删除、篡改、隐匿。

《生成式人工智能服务管理暂行办法》，2023-08-15 施行。向境内公众提供生成文本、图片、音频、视频。未向境内公众提供的研发和内部应用，不适用。第七条训练数据：合法来源；不得侵害知识产权；个人信息需同意或其他法定情形；提高数据质量。提供者对输出承担网络信息内容生产者责任，对个人信息承担处理者责任。总则第四条是义务清单：价值观、反歧视、知识产权和商业秘密、人格权、透明度。不是抗辩清单。

《人工智能生成合成内容标识办法》，2025-09-01 施行。把显式标识和文件元数据隐式标识（属性、服务提供者、内容编号）写成可执行规则。传播平台有核验和提示义务。标识不是版权登记。诉讼里会和“这是不是作品、谁是作者、平台尽没尽到注意义务”缠在一起。去掉标识再传播，李某案那种事实，以后会更容易被写成恶意。

中国没有 TDM 例外，也没有美国式一般 fair use。著作权法合理使用是封闭列举。《暂行办法》第七条把训练阶段的版权风险写成监管义务，但何种训练构成侵害，仍要回到复制、信息网络传播、合理使用，外加正在长出来的判决。立法和司法目前不同步。监管可以先要求“合法来源”，法院还在逐案问合法指什么。

<br>

中国作者资格的早期样本是北京互联网法院 2023 年李某诉刘某（(2023)京0491民初11279号）。李某用 Stable Diffusion，反复改提示词和参数，生成图片并署名发布；刘某去除署名后使用。法院认定其中有人的独创性智力投入，构成美术作品，赔偿 500 元。美国 Copyright Office 通常认为 prompt 本身控制不足；北京互联网法院则把提示词和参数迭代写进独创性。保护的仍是人的安排与选择，不是模型的人格。

杭州、广州等地的奥特曼生成图从输出侵权切入。输出与受保护角色实质性相似时，生成服务提供者可能承担直接或帮助侵权；通知、过滤、标识、是否把角色生成当卖点，会影响注意义务。国内训练数据案件仍在形成，尚无可稳定对标 *Bartz* 的上级法院规则。

### 其他

日本著作权法 30 条之 4（2018 年改正，2019 年施行）。不以享受作品中表达的思想感情为目的的利用，必要范围内可以，包括数据分析。不当损害权利人利益则不适用。没有美国式一般 fair use，这条是专门例外。文化厅 2024 年整理：生成式 AI 训练原则上可落入非享受目的；过拟合到能复现表达、把作品当检索结果展示、绕过 robots.txt 或抓有偿分析用数据库，可能构成不当损害。日本给训练开的门比美国立法宽，比 Alsup 的修辞窄——复现和绕过技术措施，明确留在门外。

新加坡 Copyright Act 2021 第 243–244 条 computational data analysis，明文覆盖机器学习，含商业训练，前提是合法访问。合同和机器人协议能不能排除该例外，2026 年 8 月 MinLaw / IPOS 仍在征求意见。合法访问四个字，会变成付费墙、API 条款、scraping 的主战场。

英国 2024–2025 年讨论过带 opt-out 的透明 TDM 例外，后来相关报告回到维持现状。商业训练在英国仍更接近要许可。三个普通法地区，三种脾气。

<br>

# 智能体

Agent 把一次生成扩成带工具的连续行动：浏览、点击、填表、调 API、读写邮箱、提交 PR。版权、合同、CFAA、隐私会在同一条轨迹上被点亮。预训练吃 Common Crawl 是一件事。推理时对当前网页做即时复制，是另一件。

感知：读网页、读代码、读邮件。复制件有没有产生，有没有绕过技术措施。

规划：系统提示、记忆、检索库自己的权利归属。

行动：HTTP、点击、git push。ToS、CFAA、平台规则。

输出：代码、文案、图片、PR。作者、侵权、商业秘密、许可证。

RAG 把某段 GPL 源码贴进 PR，是普通的复制发行，不需要先回答权重是不是衍生作品。把客户代码丢进第三方托管模型，可能直接违反 NDA。代码助手写出来的实现，conception 仍要落到人，发明人漏列风险变大。

robots.txt、付费墙、登录后的页面，在日本 2024 年意见和欧盟 opt-out 实践里，已经被当成权利人有没有保留的信号。在美国它们更多是合同和 CFAA。Copyright 上能不能单独成立，还要看复制的量和用途。*Sega* 的中间复制叙事，和预训练吃整本书的叙事，技术事实不同，硬套会错。

平台治理。Agent 以用户名义发帖、下单、注册。§ 230、消费者保护、广告法都会问这句话是谁说的。技术上用户点了允许，法律上不一定等于用户发表。证券、医疗、法律咨询这些已有职业规范的领域，agent 输出会先撞行业法，再撞版权。

<br>

# 未决

还没有被最高法院钉住、但已经可以当问题来写，而不是当立场来喊的，大概是这些。

训练通用模型，究竟是一种用途上的 transform，还是大规模复制的中间步骤。*Warhol* 要求看具体市场用途。Alsup 把技术过程写得很变革。Chhabria 把市场稀释留给下一家证据更好的原告。Bibas 证明：直接做竞品时，训练两个字救不了。

盗版来源是独立侵权，还是污染整个 fair use。两名北加州法官已经投下不同的票。15 亿和解给盗版库定了价，没给训练定规则。

许可市场一旦密集存在，未许可训练还能否声称没有受保护的市场。新闻社、图库、影视角色都在试着做交易。Disney–OpenAI 宣布过、没交割就停了，说明定价意愿在，市场还没稳。

输出。Memorization、角色、逐页替代、风格模仿，保护强度递减。风格几乎不受版权保护，不意味着角色可以随便吐。

Agent 瞬时复制和预训练复制是否共用同一套 fair use。技术事实不同。

开源权重把商业秘密拿走之后，版权和合同还剩下什么。模型许可如果只是合同，对非缔约下载者约束力有限。

分析这些问题，先写清复制了什么、存了什么、用户看没看到、绕没绕过技术措施。再写作者、实验室、下游、用户、平台、监管各自要什么。最后才讨论例外、强制许可、透明度、标识、选择退出、责任保险。每一种都会反过来改模型：过滤、记忆、工具权限、日志。结论可以硬，不能从立场直接跳到口号。

<br>

# 资料

15.628J：[主页](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/) · [Syllabus](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/syllabus/) · [Readings](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/readings/) · [Assignments](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/assignments/)

6.4590：[OCW](https://ocw.mit.edu/courses/6-4590-foundations-of-information-policy-fall-2024/) · [课网](https://internetpolicy.mit.edu/6.4590/) · [About](https://internetpolicy.mit.edu/6.4590/about-the-class/)

Stanford CS202：[ExploreCourses 正式课程描述](https://explorecourses.stanford.edu/search?q=CS+202) · [Fall 2006 旧课程归档](http://web.stanford.edu/class/cs202/)。五件现代案例来自前者，不来自 2006 归档。

17 U.S.C. §§ 102, 106, 107, 512, 1201, 1202。35 U.S.C. §§ 101–103, 112。47 U.S.C. § 230。DSM 2019/790 arts. 3–4。Regulation (EU) 2024/1689 art. 53。

USPTO inventorship guidance：2024-02 发，2025-11 废，改回 conception。Copyright Office *Copyright and Artificial Intelligence* Part 1 Digital Replicas（2024-07）；Part 2 Copyrightability（2025-01-29）。

*Authors Guild v. Google*, 804 F.3d 202 (2d Cir. 2015)。*Bartz* 终局批准 2026-07-20（N.D. Cal. 4:24-cv-05417）。*Ross* 第三巡回 No. 25-2153，2026-06-11 辩论。*Thaler v. Perlmutter* cert. denied 2026-03-02。Disney–OpenAI Sora 许可 2025-12 宣布、2026-03 随 Sora 关停取消。

《生成式人工智能服务管理暂行办法》（2023-08-15）。《互联网信息服务深度合成管理规定》（2023-01-10）。《人工智能生成合成内容标识办法》（2025-09-01）。日本著作权法 30 条之 4。Singapore Copyright Act 2021 ss. 243–244。

15.628J 的案例仍值得按 session 过：先建立四种 IP 和 claim 意识，再读软件专利史，再把 *Alice* 以后和 2025–2026 训练数据判决当成同一条时间线上的续集。6.4590 不必按周重做。架构、§ 230、*Carpenter*、跨境那几周的读物，当政策写作样本足够。
