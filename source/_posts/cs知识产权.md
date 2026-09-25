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

# 导论

- **材料范围**：三门课程提供知识产权教义、信息政策与技术公司争议三条主线，2013 年后的判例补足其时间缺口。
- **阅读路线**：先理解排他权和四类知识产权，再处理归属与信息政策，最后把规则用于人工智能和智能体。
- **问题索引**：从训练、抓取、API、职务成果、开放权重、论文公开和 NDA 等具体问题反查制度。
- **判例方法**：区分法源、裁判规则、程序阶段、诉因、救济和证明标准。

## 材料范围

本文以美国法为主。基础材料是 MIT 15.628J / 6.903J *Patents, Copyrights, and the Law of Intellectual Property*（Spring 2013）、MIT 6.4590 *Foundations of Information Policy*（Fall 2024）与 Stanford CS202 *Law for Computer Science Professionals*。MIT 给 doctrine 和信息政策框架；Stanford CS202 用技术公司真实争议把 software patent、API、商业秘密、CFAA 和雇佣合同串起来。2013 年之后再补 *Alice*、*Google v. Oracle*、*Warhol*、*Authors Guild v. Google* 与 2025–2026 训练数据诉讼。

- [15.628J](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/)
- [6.4590](https://ocw.mit.edu/courses/6-4590-foundations-of-information-policy-fall-2024/)
- [Stanford CS202（ExploreCourses）](https://explorecourses.stanford.edu/search?q=CS+202)

<br>

## 阅读路线

> [!INFO]+ 章节
>
> `#` 按问题顺序排，不是课表。激励和体系先交代排他权的目的，以及专利、版权、商业秘密、商标各自管什么。版权、专利、秘密、商标是私权本体，计算机里最常碰到的放前面。版权按独创、思想、软件、接口、合理、间接拆构成要件；专利按资格写到救济。归属处理职务发明、跳槽和开源。信息政策转公法：言论、平台、搜查、隐私、跨境。人工智能和智能体把前面的规则套到训练、输出、作者、发明人，以及带工具的连续行动。未决和资料收录尚无最高法院终局规则的问题，以及课程、法条、判决出处。

<br>

## 问题索引

> [!NOTE]+ CS / AI 问题映射
>
> - **GitHub 代码能否用于训练** → copyright、许可证、fair use。
> - **Agent 自动爬网站** → CFAA、合同、版权、隐私。
> - **API 实现与 API 本身的区别** → § 102(b)、*Lotus*、*Google v. Oracle*。
> - **实习期间写出的 agent framework 归谁** → assignment、work made for hire、*Stanford v. Roche*。
> - **Open-weight model 能否商用** → 模型许可证、权重和代码的权利分层。
> - **研究代码能否公开到 GitHub** → 雇佣合同、商业秘密、专利 novelty。
> - **NDA 与论文发表冲突** → 保密义务、发明转让、学校与公司之间的 publication review。

<br>

## 判例方法

> [!INFO]+ 判例怎么读
>
> **Statute（成文法）**是国会或州议会通过的文字，例如 17 U.S.C. § 107。**Regulation（行政法规）**由行政机关依据授权制定。**Case law（判例法）**是法院在具体案件里解释、适用规则形成的法律。
>
> **Holding（裁判规则）**是法院为解决本案争点所必需的结论；**dicta（附带意见）**是不影响本案结果的讨论，可能有说服力，但通常不具同等约束力。**Precedent（先例）**的约束范围取决于法院层级和辖区：美国最高法院约束全国的联邦法问题；联邦巡回上诉法院约束本巡回；地区法院意见原则上不约束其他法官。
>
> **Procedural posture（程序阶段）**会改变一句话的分量。驳回动议只问诉状是否足以继续；summary judgment（简易判决）是在关键事实没有真正争议时依法裁判；preliminary injunction（初步禁令）只是诉讼期间的临时救济；settlement（和解）通常不等于法院认定任何一方实体上胜诉。
>
> **Claim（诉因）**是原告据以请求救济的法律主张，不等于专利里的 claim（权利要求）。每个诉因由若干 **elements（构成要件）**组成。原告通常负担证明责任；被告可以否认要件，也可以提出 **affirmative defense（积极抗辩）**，例如 fair use。**Remedy（救济）**是责任成立后法院给什么，包括 damages（损害赔偿）、injunction（禁令）、declaratory judgment（确认判决）。

<br>

> [!NOTE]+ 引用怎么看
>
> `17 U.S.C. § 107` 表示《美国法典》第 17 编第 107 节；`§§` 表示多个条文。`593 U.S. 1 (2021)` 表示判决收在 United States Reports 第 593 卷第 1 页，年份 2021。`9th Cir.` 是第九巡回上诉法院；`N.D. Cal.` 是加州北区联邦地区法院；`D. Del.` 是特拉华联邦地区法院；`S.D.N.Y.` 是纽约南区联邦地区法院。
>
> 联邦案件通常从 district court（地区法院）开始，上诉到所属 circuit court（巡回上诉法院），最后才可能由 Supreme Court（最高法院）选择受理。`cert. denied` 只表示最高法院拒绝受理，不表示最高法院赞成下级法院理由。

<br>

> [!NOTE]+ 证明到什么程度
>
> 本文多数是 civil case（民事案件）。普通民事事实通常按 **preponderance of the evidence（优势证据）**证明，即某事实更可能为真。专利已经授权后，无效主张通常须达到 **clear and convincing evidence（清楚且有说服力的证据）**。刑事商业秘密盗窃等则适用更高的 **beyond a reasonable doubt（排除合理怀疑）**。同一组事实进入民事案和刑事案，诉讼主体、构成要件和证明标准都可能不同。

<br>

# 激励

本章按“为什么授权—权利边界在哪里—制度以什么换什么”的顺序展开：

- **宪法基础**：知识产权条款授权什么，其目的和范围是什么。
- **排他边界**：排他权能禁止哪些行为，又保留哪些公共使用空间。
- **制度交换**：专利、版权和商业秘密分别以披露、期限或保密换取什么。

## 宪法基础

宪法知识产权条款的范围很窄：

*To promote the Progress of Science and useful Arts, by securing for limited Times to Authors and Inventors the exclusive Right to their respective Writings and Discoveries.*（U.S. Const. art. I, § 8, cl. 8）

条文把有限期排他权授予 Authors 和 Inventors，权利分别对应其 Writings 和 Discoveries。它不是对作者或发明人身份本身的保护。期限是 limited Times，目的条款是 progress of Science and useful Arts。劳动应得、自然权利这类叙事，条文里没有。

Trademark 不在这条里，其联邦立法权主要来自 Commerce Clause（商业条款）；Lanham Act 延续反不正当竞争传统，核心问题是消费者会不会认错来源。Trade secret 更接近侵权和合同，长期以州法为主。联邦层面 1996 年 Economic Espionage Act 已规定刑事商业秘密犯罪，2016 年 Defend Trade Secrets Act 又增加联邦民事诉权。

## 排他边界

这里的 **exclusive right（排他权）**不是“国家确认某人对某物拥有绝对控制”。它只允许权利人在法定范围内排除特定行为。例如版权人可以禁止复制受保护的表达，但不能禁止别人独立想到同一个事实或思想；专利权人可以排除他人实施 claim 覆盖的发明，即使对方独立研发。权利范围、期限、例外、救济都由法律限定。

知识产权有时被口语化地称为 monopoly，但两者不能完全等同。专利确实赋予排除权，却不保证专利权人能实际生产产品：产品还可能落入别人的更宽专利、药品监管或出口管制。版权也不是对作品主题的垄断，只保护可识别的表达。

课上反复碰到的，是排他权给出去之后，后来的人还能否在同一方向上继续做。专利 claim 如果写到抽象功能层，覆盖面会把整条技术路线圈进去，后来者即使自己做出实现，也要避开或交许可。版权如果把程序的功能、操作方式也当成 expression，兼容实现、以及为兼容而做的反向工程，空间都会被挤掉；*Lotus*、*Sega* 后来就是在拆这件事。商业秘密没有强制披露，社会拿不到说明书那一层交换。四种制度都在激励创作和留下后续使用空间之间做取舍，方式并不一样。

## 制度交换

Arrow 的 information paradox：买方要知道 invention 是什么才肯付钱；一旦知道了，往往不必再买。没有排他或保密，事前投资会偏少。反过来，排他范围过宽时，复制成本已经接近零的东西会被法律重新变贵。Patent 用向公众披露换大约 20 年的独占。Copyright 保护的是 expression，给的是复制等禁令，到期进入 public domain（公共领域）：版权已经届满、不成立或依法不受保护，公众无需取得版权许可即可使用。Trade secret 不公开、也没有固定期限，但挡不住独立发现和合法 reverse engineering。这是三套不同的制度，不能按“谁更强”排序。

<br>

# 体系

本章把四种制度放在同一张图里，但不把它们当作同一种保护：

- **四权分工**：先区分各制度的保护对象、成立门槛、可对抗主体和期限。
- **叠加保护**：再看同一个模型产品的技术、代码、数据和品牌如何分别落入不同制度。
- **资产策略**：最后讨论公开论文、开源代码、发布权重和申请专利对权利状态的影响。

## 四权分工

四种制度回答的不是同一个问题：

1. **Patent**：某项技术方案是否满足授权条件；授权后，权利人可在美国法定范围内排除他人制造、使用、销售、许诺销售或进口落入 claim 的发明。
2. **Copyright**：是否存在人的原创表达，被告是否复制了受保护部分。
3. **Trade secret**：信息是否仍属秘密，被告取得或使用的手段是否不正当。
4. **Trademark**：某个标志是否识别商品来源，被告的使用是否导致消费者混淆。

同一个产品可以同时受四种制度保护。例如模型推理系统的具体技术改进可能申请专利；源代码受版权保护；未公开的训练配方可构成商业秘密；产品名和 logo 走商标。四种权利并不互相替代。

| | Patent | Copyright | Trade Secret | Trademark |
| :--- | :--- | :--- | :--- | :--- |
| 管什么 | 有用的技术方案 | original expression | 因保密而值钱的信息 | 来源识别 |
| 门槛 | useful / novel / non-obvious / eligible / 充分披露 | 独立创作 + 一点点 creativity，固定在有形介质 | 秘密 + 合理保密措施 + 经济价值 | 商业使用、能识别来源、还没变成通用名称 |
| 能挡谁 | 独立发明人也挡 | 只挡 copying | 只挡不正当获取和使用 | 挡混淆、有时挡淡化 |
| 多久 | 通常从最早美国非临时申请日起算 20 年；另有 PTA / PTE | 自然人 life+70；雇佣、匿名、化名作品为发表后 95 年或创作后 120 年，取较早者 | 满足法定条件且仍属秘密期间 | 持续使用并按期续展 |
| 换来什么 | 向公众披露 | 表达最终进公共领域 | 无强制披露 | 降低搜寻成本 |
| 软件里常见 | 写得够具体的系统 / 方法 claim | 代码、文档、非功能性结构 | 未开源权重、数据、过滤、系统提示 | 产品名、logo、模型名 |

## 叠加保护

上表比较的是制度；放到具体产品中，还要按资产逐项识别保护对象，不能把产品整体只归入其中一栏。

## 资产策略

实验室里这四样经常叠在同一块资产上。算法思想若拿去申请专利，论文一发 novelty 可能没了，§ 101 也不一定过。代码的版权因创作自动产生，开源则是按许可证把其中一部分权利放掉。权重和数据最接近商业秘密；权重一旦公开，秘密性很难再主张。产品名走商标。常见做法是：论文换引用和职位，数据继续当秘密，名字去注册商标，专利只覆盖真正有技术改进、claim 也写得动的那一层。

开源权重通常意味着放弃权重本身的商业秘密保护。Llama community license 同时包含版权许可和额外使用限制；具体限制属于许可条件还是合同义务、能否约束下游，取决于权重受哪些权利保护、条款性质和合同是否成立，不能直接等同于 GPL 的 copyright copyleft。

<br>

> [!NOTE]+ 四种制度不好比强度
>
> 专利可以挡住独立发明，但必须公开、有期限，软件 claim 还可能过不了 § 101。版权自动产生，只挡住 copying。商业秘密不公开，一旦泄露或被合法反向工程就难以再主张。商标可以续展很久，保护的不是创意本身。实验室里常见的处理仍是：论文换引用，数据当秘密，名字注册商标，专利只覆盖有技术改进的那一层。

<br>

# 版权

本章从版权是否成立开始，依次处理保护范围、软件与接口、合理使用和平台责任：

- **成立与权利**：保护对象、归属、侵权、抗辩、救济，以及权利束、登记、期限和发行用尽。
- **独创性**：独立创作和最低创造性，尤其是事实汇编。
- **保护边界**：思想—表达二分、merger、标准场景和实质性相似。
- **软件与接口**：程序对象分层、抽象—过滤—比较、兼容与 API。
- **合理使用**：四因素、具体用途、教育、搜索和训练类比。
- **间接责任**：帮助、控制、诱导、双用途工具和平台安全港。

## 成立与权利

### 五步分析

17 U.S.C. § 102(a)：*original works of authorship fixed in any tangible medium of expression*。

版权问题按五步拆：

1. **保护与权属**：是否存在已固定的 original expression，原告是否拥有有效版权。
2. **事实复制**：被告是接触并复制了原作，还是独立创作出相似内容。
3. **违法挪用**：先滤除思想、事实、标准场景和公有领域材料，再判断剩余表达是否达到 substantial similarity，并确认触及哪一项 § 106 权利。
4. **抗辩或限制**：fair use、first sale、许可、§ 117、DMCA safe harbor 等是否适用。
5. **救济**：实际损失、侵权利润、法定赔偿、律师费、禁令。

**Fixation（固定）**是表达被记录在足以稳定保存和再现的介质上。写入磁盘、录音、保存图片都算；转瞬即逝且没有被记录的即兴表达可能不算。版权保护自动发生在原创表达固定时，登记主要影响诉讼资格和救济，不是版权出生的时点。

侵权通常先证明两件事：被告 **actually copied（实际复制）**，而复制的内容又与原作中的可保护表达构成 **substantial similarity（实质性相似）**。实际复制可由直接证据证明，也常由 access（接触机会）加相似性推断。即使证明复制，还要滤掉事实、思想、标准场景和公有领域材料。

§ 102(b) 紧接着把 idea、procedure、process、system、method of operation、concept、principle、discovery 排除出去。软件、API、模型相关判决，争点经常落回这一句。

> [!EXAMPLE]+ 同一句在软件里怎么拆
>
> 排序算法的思想 → idea，§ 102(b)。某段具体 quicksort 实现 → expression。菜单如何组织功能 → *Lotus* 写成 method of operation。API 声明代码 → *Google v. Oracle* 假定可版权，按 fair use 放走，copyrightability 本身没钉。训练学到的统计规律更接近 idea / 事实；复制训练语料是另一根 § 106。

### 权利类型

§ 106 列的是一束专有权利：复制、制备演绎作品、发行、公开表演、公开展示，以及录音作品通过数字音频传输公开表演的特定权利。争点常常不是“有没有版权”，而是被诉行为碰到了哪一项。训练阶段制作复制件，走复制权。输出与原作很像，可能走复制，也可能走演绎。把角色图当成产品功能公开展示，还要看 display。

**Derivative work（演绎作品）**是基于原作改编、翻译、影视化或以其他形式重铸后的作品。新作品可以包含新作者自己的表达，但这不消除对原作受保护表达的利用。**Distribution（发行）**通常要求向公众转移副本的所有权或占有；**display（展示）**与 **performance（表演）**则分别覆盖图像展示和作品表演/播放。

### 登记与期限

版权因创作产生，不必登记。美国 1989 年加入伯尔尼之后，notice 也不是成立条件。美国作品通常须完成登记后才能在联邦法院起诉（*Fourth Estate v. Wall-Street.com*, 2019）；外国作品涉及不同规则。法定赔偿和律师费另受 § 412 限制：原则上须在侵权开始前登记，但作品首次发表后三个月内登记仍可保留相关救济；未及时登记并不当然消灭实际损失和可归因于侵权的被告利润请求。§ 504(c) 的幅度是每部作品 $750–$30,000，法院可在此间酌定；故意侵权上限 $150,000，不知情可低至 $200。“每部作品”的认定还会受 compilation、collective work 等规则影响，不能把每个文件机械地各算一件，但大型语料中的作品数量仍会显著放大赔偿风险。

期限：1978 年之后自然人创作，life+70。雇佣作品、匿名、化名：发表起 95 年或创作起 120 年，取短。1998 年 Copyright Term Extension Act 再延 20 年，*Eldred v. Ashcroft*（2003）维持。limited Times 还在，只是很长。

### 发行用尽

发行权有 first sale / 用尽。正版实体书买来可以再卖。*Kirtsaeng v. John Wiley*（2013）把这层用到合法进口的教材上。数字复制不是同一件事：多出来的那份拷贝，不是“卖掉手里那一本”。*Capitol Records v. ReDigi*（2d Cir. 2018）因此没有把 first sale 扩到二手数字音乐转售。*Bartz* 里 Anthropic 后来购买二手书、拆装、扫描，并销毁纸本。Alsup 认定这种不增加副本数量的数字替代属于 fair use；这是 fair use 结论，不是 first sale 自动覆盖扫描件。

<br>

## 独创

### 最低门槛

Originality 只包括两个要求：**independent creation（独立创作）**和极低程度的 **creativity（创造性）**。它不要求新颖、好看、有价值，也不要求以前没人做过。两人完全独立拍出近似照片，各自都可能有版权；这和专利的 novelty 不同。

### 事实汇编

*Feist Publications v. Rural Telephone Service*, 499 U.S. 340 (1991)。Rural 是堪萨斯一家电话公司，白页按姓氏排列。Feist 编区域号簿，Rural 不授权，Feist 照抄，连几个虚构条目都抄进去了——access 因此很好证。

最高法院拒绝 sweat of the brow。事实不受保护。汇编要有 original selection, coordination, or arrangement。按字母排电话号码，创造性为零。O'Connor 写 originality 只要求独立创作加 a modicum of creativity，门槛极低，但不是零。

所以：

- 测量、新闻事件、判决原文、股票价格：不受版权
- 对事实的选择、编排、注释：可能受保护，范围很窄
- 小说、插画、大量代码：核心区

*Thomson Reuters v. Ross* 后来打的就是这层。Westlaw headnotes 不是判决本身，是编辑对判决的归纳。Delaware 地区法院 2025 年认定这层有 originality。法律意见公共，headnotes 不一定公共。数据工作里，花费大量时间本身从来不够。

<br>

## 思想

### 思想—表达二分

**Idea–expression dichotomy（思想—表达二分）**是版权边界。思想、事实、方法、系统留给公众；作者把它写成的具体文字、画面、代码才可能受保护。不是先把作品整体标为“有版权”再禁止一切相似，而是侵权比较时逐层滤掉不受保护部分。

*Baker v. Selden*, 101 U.S. 99 (1879)。Selden 写书讲一套记账法，附空白表格。Baker 用了类似表格。版权保护书里的说明文字，不保护记账方法。方法要排他，走专利。

Learned Hand 在 *Nichols v. Universal Pictures*（2d Cir. 1930）里写成 abstractions test。从具体对白往上抽：情节、角色类型、主题、idea。抽到某一层，版权不再管。Hand 说没人能精确划出那条边界，以后也不会有人能。软件、模型、prompt 的保护范围争论，后来也是在不同抽象层之间移动。

### 合并、标准场景与抄袭

Merger：某种 idea 只有极少几种写法，表达和思想合并，版权让开；否则等于用版权把思想垄断掉。Scènes à faire：西部片的酒吧、代码里的常规循环和错误处理，保护很弱。独立创作出相似作品，不构成侵权。这一点和专利不同：专利可以挡住独立发明，版权只挡住 copying。

Plagiarism 是学术伦理。Infringement 是法定权利。抄了没有版权的事实，可能构成学术抄袭，却不一定构成侵权；改写了仍可能侵权。两套标准需要分开。

### 复制与实质性相似

*Selle v. Gibb*, 741 F.2d 896 (7th Cir. 1984)。Bee Gees《How Deep Is Your Love》。侵权要 copying。没有 access 的证据，只靠两首歌听起来像，不够。后来软件和模型输出争 substantial similarity，仍要先过 copying 这一关：有没有接触、是不是独立生成了相似的东西。

Substantial similarity 也不是单一测试。一种问普通观察者会不会觉得是抄的；第九巡回等法院对复杂作品会拆 extrinsic（可客观比较的结构、保护范围）和 intrinsic（普通人的整体印象），但这不是全国统一措辞。滤掉 idea、公有领域、常规写法之后再比。模型输出常争的是：这段属于 memorization，还是同一主题下的独立生成。NYT 诉状展示过近乎逐字的文章输出，可作为 memorization 主张的证据，但并非终局侵权认定。某个角色的姿势、配色“有点像”，可能到不了。抽象风格本身通常不能被版权垄断；但所谓风格化输出仍可能复制可识别角色、独特情节、构图、台词或大段原文。分析输出侵权，需要先把对象放到这根尺子上，不能停在观感相似。

<br>

## 软件

### 对象分层

计算机程序在 § 101 里定义为使计算机产生一定结果的语句或指令，作为 literary work 受保护。§ 117 允许合法副本所有人为运行程序制作必要拷贝、以及备份；这管的是自己用这份程序，不是把代码送进训练集。字面抄源码最好认。难的是结构、接口、菜单、API。15.628J 把 *Baker* 和 *Lotus* 放进同一周，是为了把“看起来像抄”拆成思想、操作方式和表达。

软件同时包含多层对象：源码的逐字文本、模块组织、数据结构、接口名称、运行结果和底层功能。越靠近具体源码，版权保护越强；越靠近算法、协议、兼容要求和操作方法，越容易落入 § 102(b)、merger 或 scènes à faire。Abstraction–filtration–comparison 的目的，就是防止法院把“程序整体相似”直接当成侵权。

### 抽象—过滤—比较

*Whelan v. Jaslow*（3d Cir. 1986）曾经把程序的 structure, sequence, organization 扩得很宽，几乎把功能目的当成 expression。*Computer Associates v. Altai*, 982 F.2d 693 (2d Cir. 1992）把它压回去。Abstraction–filtration–comparison：

1. 把程序抽象到不同层：源码、目标码、模块、算法、功能
2. 滤掉 idea、公有领域、效率强制的写法、外部因素（硬件、协议、兼容需求）
3. 只比较剩下的 protectable expression

滤完之后能主张的，往往比权利人想的窄。效率强制很关键：某种排序在某种数据结构下只能那样写，merger 直接上来。

### 兼容与反向工程

*Lotus v. Borland*, 49 F.3d 807 (1st Cir. 1995)。Borland 的 Quattro Pro 做了 1-2-3 兼容模式，菜单命令层级几乎照搬，好让会计不用重新学。第一巡回认定菜单是 method of operation，§ 102(b) 不保护。若把用户已经学会的操作方式纳入版权，竞争者实现兼容就要取得许可。最高法院 4–4，维持，没有意见。所以这还是第一巡回的法，不是全国终局，但影响很大。CLI、快捷键、配置格式、很多“兼容层”，分析时都还在用这条思路。

反向工程。*Sega v. Accolade*, 977 F.2d 1510 (9th Cir. 1992)。Accolade 要给 Genesis 做未经授权的游戏，反汇编中间复制了整份代码。第九巡回：为了读到不受保护的功能、做出兼容，中间整份复制可以是 fair use。功能本身不受版权。为了看见功能而做的拷贝，和把表达拿去替代原作市场，不是一类用途。Agent 读网页、安全研究反编译、为了接 API 而获取文档，经常要回到这里。DMCA § 1201 的反规避责任是另一条诉因；即使复制属于 fair use，绕过技术保护措施仍可能承担独立责任。

<br>

## 接口

### 声明与实现

API 至少要区分三层：**declaring code** 告诉开发者名称、参数和调用方式；**implementing code** 真正完成运算；**organizational structure** 决定 package、class、method 怎么分组。兼容实现可能复制第一层和部分组织，却完全重写第二层。*Google v. Oracle* 处理的正是这种分层，不是把整个 Java 实现搬进 Android。

### *Google v. Oracle*

*Google v. Oracle*, 593 U.S. 1 (2021)。Google 为 Android 复制了约 11,500 行 Java SE API 的 declaring code，涉及 37 个 package，implementing code 则自行重写。地区法院法官 William Alsup 在审理中详细区分了 Java 的声明与实现。Alsup 原判 API 声明代码不可版权，联邦巡回两次推翻相关结论。最高法院以 6–2 裁判，多数意见由 Breyer 撰写；法院假定声明代码可以受版权保护，直接从 fair use 判 Google 胜诉。

理由大致是：重新实现移动设备平台形成了不同的使用环境；声明代码与程序员已经掌握的调用知识紧密相连，复制它有助于开发者把技能带到新平台；结合 Java SE 的既有市场、Android 的移动平台用途及潜在市场证据，第四因素倾向 Google。这不是认定 Android 与 Java SE 完全不存在任何市场关系。Thomas 异议，认为 Google 取走了 Java 的核心并损害 Oracle 的许可市场。

最高法院没有裁定 API 声明代码是否受版权保护，而是假定其可以受保护，再从 fair use 裁判。因此本案不能概括成“API 无版权”；接口声明的 copyrightability 仍没有全国统一的终局规则。

Alsup 在 2025 年又审理 *Bartz v. Anthropic*。在 *Oracle* 的地区法院阶段，他先后作出 API 声明代码不可版权、以及复制属于 fair use 的意见，两次均被联邦巡回推翻；最高法院最后从 fair use 判 Google 胜诉。比较 *Bartz* 时，这段程序史有助于理解地区法院怎样处理“复制结构、改变用途”的论证。

<br>

## 合理使用

### 四因素

17 U.S.C. § 107。四个因素：

1. purpose and character（商业性、是否 transformative）
2. nature of the copyrighted work（事实性 vs 创造性；已发表 vs 未发表）
3. amount and substantiality（数量，以及有没有取走 heart）
4. effect upon the potential market

Fair use 是积极抗辩，不是事先发放的许可证，也没有固定百分比规则。法院要对具体使用逐案衡量：

- 第一因素问被诉使用的目的、商业性，以及是否加入新目的或新意义。
- 第二因素问原作偏事实还是创作、是否已发表；小说通常比事实资料更靠近保护核心。
- 第三因素同时看数量和质量。复制比例小，也可能取走作品的 heart；复制整部也不一定自动败诉。
- 第四因素看该使用及同类使用若广泛发生，会不会替代原作或权利人依法应控制的许可市场。

Transformative 不等于技术上“变换了格式”。OCR、tokenization、embedding 都是技术过程；法律仍问被诉使用在市场和表达利用上做什么。*Warhol* 后尤其要把用途写具体。

不是四项打分、三比一就赢，法条也没有规定固定权重。判例经常重点讨论第一项和第四项，但法院仍须结合具体事实整体衡量，二者也会互相牵扯。

### 用途界定

*Campbell v. Acuff-Rose*, 510 U.S. 569 (1994)。2 Live Crew 讽拟 Roy Orbison 的《Oh, Pretty Woman》。商业性不再一票否决。Parody 必须借用足够的原作，公众才能听出在讽什么，第三因素因此让步。Transformative 从此变成第一因素的中心词。后来几乎任何新用途都会被写成 transformative，这个词被拉得很宽。

*Andy Warhol Foundation v. Goldsmith*, 598 U.S. 508 (2023) 收紧了 transformative 的使用。Goldsmith 给 Prince 拍了照片，Vanity Fair 1984 年授权 Warhol 制作一次丝网作品。Prince 去世后，Condé Nast 又向 Warhol 基金会取得橙色 Prince 图像的杂志封面许可。最高法院分析的是这一次商业许可，而不是抽象评价 Warhol 的艺术是否具有新意义；该用途与 Goldsmith 照片的授权市场实质相同。多数意见由 Sotomayor 撰写；Kagan 的异议重点反对多数意见对用途和艺术转换的界定。

<span style="background:#ffe58f">Transformative 必须落在被诉的那一次使用上。新技术、新中间物、新美学，都不自动等于新用途。</span>

读训练数据判决时，这件案子是背景。不能因为 LLM 在技术上把 token 变成权重，就宣布第一因素结束。Alsup 在 *Bartz* 里把训练的四个因素逐项写完；Chhabria 在 *Kadrey* 里把第四因素拉回来。两份意见都引用了 *Warhol*。

### 教育、搜索与缓存

课堂复印。*Cambridge University Press v. Becker*（N.D. Ga. 2012 及后续 Eleventh Circuit）是 15.628J Session 5 用来练 § 107 的样本：大学电子课件摘多少算合理。和生成模型无关，但把“数量”“市场”“教育目的”怎么映射到四个因素，写得很细。适合当 § 107 的练习样本，不宜直接拿来类比 AI 训练。

互联网平台上的版权，6.4590 Week 7 才会和 15.628J 的课堂复印分开。*A&M Records v. Napster*（9th Cir. 2001）集中式索引，间接责任好证。*Grokster* 改成去中心化之后，改打 inducement。*Perfect 10 v. Amazon*（9th Cir. 2007）缩略图检索，fair use 方向更接近后来的 Google Books：用途是检索，展示是碎片。Agent 浏览缓存、搜索引擎快照，分析时更靠近这一支，而不是预训练使用整本书。

<br>

## 间接责任

### 三条责任路径

直接侵权之外三条：

**Contributory**：知道 + 实质性帮助。

**Vicarious**：有权且有能力控制 + 直接经济利益。

**Inducement**：以促进侵权为目的而散布工具。

三条责任的功能不同。Contributory infringement 类似“知道侵权还提供关键帮助”；vicarious liability 即使不知道，也可能因能控制行为且直接获利而成立；inducement 看产品提供者的主观目的和推广证据。它们都以某个直接侵权行为为基础。仅证明模型“有能力”生成侵权内容，通常还不够说明平台应负哪一种责任。

### 双用途工具

*Sony v. Universal*（1984），Betamax。设备具有 substantial noninfringing uses，制造商不因用户可能用来录像而承担间接责任。*MGM v. Grokster*, 545 U.S. 913 (2005)。P2P 软件也可以有大量非侵权用途，但 Grokster 的广告、内部文件表明目的就是接替 Napster。Inducement 成立。仅有 dual-use 能力不够，还要看散布者是否以促成侵权为目的去推广。

### 平台安全港与技术措施

平台版权责任大量走 DMCA § 512：notice-and-takedown。安全港分成接入、缓存、存储、检索四类，各有条件；不符合条件只表示回到普通间接责任，并不等于平台自动侵权。§ 1201 反规避是独立诉因。§ 1202 管版权管理信息（CMI）移除；责任通常还要求行为人知道其在移除或更改 CMI，并知道或有合理理由知道此举将诱发、促成、便利或掩盖侵权。*Kadrey* 里对 Llama 训练的 CMI 主张，是因该案 summary-judgment 记录中的训练复制被认定为 fair use，无法据此证明移除是为了掩盖该项侵权；这个结论不能扩张成所有 fair-use 案件中的 § 1202 请求都会消失。

Notice-and-takedown 是权利人发送合格侵权通知，服务商及时删除或断开访问，用户可提交 counter-notice，随后由权利人决定是否起诉的程序。CMI（copyright management information）包括作者、作品名、版权人、使用条款等与作品权利管理有关的信息；不是所有元数据都自动属于 § 1202 的 CMI。

Safe harbor（安全港）的作用通常是限制符合条件的服务商承担金钱责任和部分禁令，并不表示底层内容或用户行为已经被认定为合法。

Grokster 的结构后面会被用于分析模型、agent、爬虫工具。模型能够生成侵权图，不等于提供模型的人一定构成诱导。若把“生成某个角色”做成卖点、放在首页、写进广告词，证明促成侵权的目的会容易得多。*Disney v. Midjourney* 的诉状就是按这条写的。

<br>

# 专利

本章按一项专利从申请、有效性到侵权和救济的顺序展开：

- **授权框架**：制度层级、四项法定门槛、申请、期限和授权后复审。
- **外观设计**：保护对象、图示边界和组件利润。
- **适格性**：司法例外、*Alice/Mayo* 两步与软件算法。
- **新颖性与显而易见性**：既有公开、单一文献、文献组合和 PHOSITA。
- **权利要求**：披露要求、claim 结构、解释与逐项比对。
- **侵权与救济**：行为类型、抗辩、用尽、赔偿和禁令。

## 授权框架

### 制度层级

四层：Constitution → Congress（Title 35）→ USPTO → 法院。专利上诉集中到 Federal Circuit，再上最高法院。Design patent 管外观，plant patent 管植物。CS 里日常碰到的几乎都是 utility patent。

### 四项门槛

专利不是“有一个想法就登记”。申请人向 USPTO 提交 specification（说明书）、drawings（附图）和 claims（权利要求）；审查员用 prior art 审查。Utility patent 至少要分别过：

1. **§ 101 subject-matter eligibility**：对象是否属于可专利主题，而非抽象思想、自然规律或自然现象本身。
2. **§ 102 novelty**：是否已有一份 prior-art reference 把 claim 的每个要素公开出来。
3. **§ 103 non-obviousness**：即使没有单篇文献全公开，多份现有技术的组合对 PHOSITA 是否显而易见。
4. **§ 112 disclosure**：说明书是否真正教会本领域技术人员实施，是否表明发明人掌握了所主张范围，claim 是否明确。

这四关不能混。某算法可以是新的，却仍因 abstract idea 过不了 § 101；也可以属于可专利主题，却早已被论文公开而过不了 § 102。

### 申请、期限与授权后程序

专利申请通过后才 **issue / grant（授权）**。申请公开不等于授权。授权也不等于有效性永远确定；被告可在诉讼中主张无效，第三方也可在 PTAB 发起 IPR。专利有 **presumption of validity（有效性推定）**，但仍可被推翻。

实用专利期限原则上从相关申请链中最早的美国非临时申请日起算 20 年，provisional application 和外国优先权日通常不计入这一期限起点。PTA（patent term adjustment）补偿 USPTO 的部分审查延迟；PTE（patent term extension）主要补偿药品等受监管产品在上市审批中消耗的期限。专利权通常到授权后才能实际主张，所以“期限起算”和“可以起诉侵权的时间”也不是一回事。

AIA（2011，主要条款 2013 年生效）之后，美国是 first inventor to file，但申请人仍须是真正发明人或其权利继受人，并不是任何最先提交的人都能取得专利。15.628J 同时发了旧 § 102 和新 § 102，读案例先看发明日还是申请日。EPO 原则上实行绝对新颖性，仅保留很狭窄的例外，不存在美国式的一般一年宽限期。美国的一年 grace period 也主要保护发明人本人或来源于发明人的特定披露，不是所有公开行为都自动获得一年保险。

授权不是终点。AIA 同时造了 PTAB 上的 inter partes review，用 prior art 打 § 102 / § 103，比地区法院便宜、快。NPE 诉讼策略因此改过一次。15.628J 那年 IPR 刚开始，现在已经是专利实务的日常。

PTAB 是 USPTO 内的 Patent Trial and Appeal Board。IPR 是授权后的行政审理，第三方主要用专利或印刷出版物主张 claim 因 § 102 或 § 103 无效。NPE（non-practicing entity）指主要持有、许可或主张专利而不亲自生产相关产品的实体；它是商业模式描述，不等于专利无效或诉讼当然滥用。

<br>

## 外观设计

### 保护对象

Utility patent 保护技术功能，design patent 保护产品的装饰性外观。外观专利图中的实线通常界定被主张的设计，虚线多用于说明环境、不属于主张范围。它也不是版权或 trade dress 的别称：授权门槛、侵权比较、期限和赔偿规则都不同。

外观专利侵权通常依 *Egyptian Goddess v. Swisa*（Fed. Cir. 2008）的 ordinary observer test：在熟悉现有设计的普通观察者眼中，被控设计与专利图示的整体视觉印象是否实质相同。不能只抽出“圆角”“边框”等单一概念比较。2015 年 5 月 13 日以后提交的美国外观专利，期限通常是自授权日起 15 年。

*Apple v. Samsung* 不是一件单纯的“圆角矩形归谁”案。长期诉讼同时出现 utility patents、design patents 和 trade dress；Stanford CS202 用它讲 smartphone design / patent，最高法院真正处理的是 design-patent damages。

### 组件利润

陪审团认定 Samsung 手机侵犯 Apple 关于圆角正面、边框和图标网格的外观设计专利，并把 Samsung 涉案手机的全部利润计入 § 289。最高法院在 *Samsung Electronics Co. v. Apple Inc.*, 580 U.S. 53 (2016) 一致裁定：多组件产品里的 “article of manufacture” 不一定是消费者买到的整机，也可以是其中一个组件，即使该组件没有单独出售。法院撤销发回，但没有决定本案应把整部手机、屏幕还是正面外壳当作 article，也没有给出完整识别测试。

> [!EXAMPLE]+ 外观专利赔偿
>
> Utility patent 通常围绕 lost profits / reasonable royalty。Design patent 的 35 U.S.C. § 289 可以追相关 article of manufacture 的 total profit。争点因此不是只看“抄了多少设计”，还要先认定利润对应的 article 是整机还是组件。复杂产品把这一步放大。

<br>

## 适格性

### 判断框架

35 U.S.C. § 101：*any new and useful process, machine, manufacture, or composition of matter*。司法例外三条：laws of nature、natural phenomena、abstract ideas。例外是法官造的，成文法没写。

Subject-matter eligibility 问的是“这一类东西能不能用专利法保护”，不是“它有没有新意”。*Alice/Mayo* 两步中，Step 1 先确定 claim 是否 directed to 司法例外；Step 2 再看单个要素及其有序组合是否加入足以把例外转成具体应用的 inventive concept。Inventive concept 也不是把 § 102/103 的新颖性审查原样搬进来。

### 案例演进

*Diamond v. Diehr*, 450 U.S. 175 (1981)。用 Arrhenius 方程 + 计算机控制橡胶硫化时间。Claim 作为整体是工业过程，不是“数学公式本身”。软件相关专利常从这里算起。这只说明含计算机步骤的工业过程仍可能是专利对象，并不表示任意软件 claim 都能过 § 101。

*State Street Bank*, 149 F.3d 1368 (Fed. Cir. 1998)。*useful, concrete and tangible result*。商业方法一度大开，金融专利、“用计算机做 X”的 claim 涌进来。

*Bilski v. Kappos*, 561 U.S. 593 (2010)。对冲商品价格风险的方法，abstract idea。Machine-or-transformation 是有用线索，不是唯一标准。*State Street* 的宽松口径被关上。15.628J Session 11 读到这里，Short Exercise 2 就是 *Bilski*。


*Mayo Collaborative Services v. Prometheus*, 566 U.S. 66 (2012)。代谢物浓度和药效的关系是自然定律。加上“告诉医生去测量、调整剂量”这类常规步骤，不够。多数意见由 Breyer 撰写。医疗诊断专利的空间被收窄。

*Association for Molecular Pathology v. Myriad Genetics*, 569 U.S. 576 (2013)。仅从人体分离出的自然 DNA 片段不可专利；通常不含内含子的 cDNA 因并非自然状态下的同一分子而可通过 § 101，但仍须满足新颖性、非显而易见性和披露等其他条件。若一段极短 cDNA 与自然序列完全相同，法院也没有把它一概纳入保护。问题因此不是“只要人为处理就可以”，而是主张对象与自然存在物是否具有法律要求的差异。*Myriad* 判在 2013 年 6 月，已经贴学期末，OCW readings 仍收了进去。

课程结束后，最高法院又在 *Alice* 中形成了当前的软件专利资格框架。

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

### 软件与算法

算法能否取得专利不能一概而论。纯数学、纯商业方法、“用计算机做 X”，容易因 § 101 被认定为 abstract idea。具体技术环境里的具体改进——减少延迟、改存储结构、改训练时的硬件调度——仍可能通过，但 claim 必须写出那一层，不能停在 “a neural network configured to…”。

PHOSITA 在 ML 里也不好定。普通水平是读过 Vaswani 2017、会调现成框架，还是会设计 kernel？Claim 写“一种 transformer，其特征在于多了某层残差”，按 *KSR* 容易被认定为已知元件的可预见组合。更可能通过 § 103 的，是产生意外技术效果，或者解决长期存在但一直未解决的具体工程问题。商业成功可以成为次要考虑，单纯营销成功则不足。

<br>

## 新颖性

新颖性判断可以依次检查：

- **关键日期**：先确定申请链、优先权和可能适用的 grace period。
- **公开方式**：再判断材料是否属于 printed publication、public use、on sale 或其他向公众可得的信息。
- **全要素披露**：最后检查一份材料是否直接或必然披露 claim 的全部 limitations 及其组合关系。

§ 102。Prior art 不是“以前有人做过类似的”。是法定范畴里、在法定时间点之前、以法定方式向公众公开的信息。公开方式：printed publication、public use、on sale、otherwise available to the public。一篇没人看过但可检索的论文，可能已经是 printed publication。内部演示不一定是。

Anticipation（预见/丧失新颖性）要求一份 prior-art reference 直接或必然公开 claim 的全部 limitations，而且组合关系也要对应。不能从论文 A 拿一半、论文 B 拿一半，再称 § 102；多文献组合通常转入 § 103。美国 AIA 给发明人自己的部分公开保留有限一年 grace period，但规则有细节，不能把它理解成所有国家通用的一年保险。

*Pfaff v. Wells*, 525 U.S. 55 (1998)。发明已经 ready for patenting，且进入商业销售或销售要约，即使实物还没做出来，也可能触发 on-sale bar。AIA 后 *Helsinn v. Teva*（2019）又确认，销售细节未向公众完整披露也不当然避开该规则。软件和模型中的早期商业演示、API 预售、向客户出售 checkpoint 都需要检查，但普通商务接洽并不自动等于法定销售或销售要约。实验室默认“先发论文再考虑专利”，在 first-to-file 世界里是在跟时间对赌。

15.628J 的 *Structural Rubber v. Park Rubber* 是练 prior art 阅读的作业，不需要当规则记，需要当方法记：日期、公开方式、是不是同一发明。

<br>

## 显而易见

显而易见性与新颖性的区别在于，它允许在有理由组合时综合多份现有技术：

- **普通技术水平**：界定 PHOSITA 的教育、经验和领域背景。
- **组合理由**：说明为什么当时会组合这些文献，以及是否有合理成功预期。
- **客观辅助因素**：再用商业成功、长期需求、他人失败等证据校验事后偏见。

§ 103。没有一份 prior art 完整公开这项发明，仍可能对 PHOSITA 显而易见而无效。

PHOSITA 是 **person having ordinary skill in the art（本领域普通技术人员）**的法律假想人，不是实际陪审员，也不是天才研究员。法院会看教育背景、经验、技术复杂度和行业速度。§ 103 允许组合多份现有技术，但需要解释为什么普通技术人员当时有动机这样组合、是否有合理成功预期；不能在知道发明答案后倒推，形成 hindsight bias。

Secondary considerations / objective indicia（客观辅助因素）包括商业成功、长期未解决需求、他人失败、行业赞誉、模仿等。它们必须与 claim 的技术特征有 nexus（联系），否则销量可能只是营销造成，证明力很低。

*Graham v. John Deere*, 383 U.S. 1 (1966)：prior art 范围与内容；与 claim 的差异；相关领域普通技术水平；次要考虑（商业成功、长期未解决问题、他人失败）。

*KSR v. Teleflex*, 550 U.S. 398 (2007) 否定刚性的 teaching-suggestion-motivation 要求。PHOSITA 被假定具有普通创造力。已知元件的常规组合可以显而易见。对 ML 专利尤其如此：attention、residual、某种 tokenizer 的组合若只有可预见效果，可能因 § 103 无效。关键是组合产生的非显而易见技术效果，不是模块清单。

次要考虑不是形式项目。长期未解决需求、他人反复失败和与发明相关的商业成功，可以反证该方案并非显而易见。若成功来自广告、定价或不在 claim 内的功能，证明力则有限。

<br>

## 权利要求

### 披露要求

§ 112 中几个要求需要分别判断：

- **Written description**：说明书是否表明申请人在申请时已经掌握所主张的发明。
- **Enablement**：是否足以使 PHOSITA 在不过度试验的情况下实施 claim 的完整范围。*Amgen v. Sanofi*（2023）说明，主张很宽的功能性属概念，却只教会少量实施例，可能无法支持整个范围。
- **Definiteness**：结合说明书和审查历史，claim 是否以合理确定性告知本领域人员权利边界；这是 *Nautilus v. Biosig*（2014）的标准。
- **Means-plus-function**：§ 112(f) 允许用 “means for …” 表述功能，但范围会联系到说明书披露的对应结构及其等同物。软件 claim 若只写功能、说明书却没有算法或结构，范围可能很窄，甚至不明确而无效。

Patent 的社会契约在披露。说明书既解释 claim，也承担 written description 和 enablement 的独立义务。写得很玄、实施不了，无效；只披露少量具体实现，却试图用 claim 覆盖所有神经网络，也可能无效。

*O'Reilly v. Morse*, 56 U.S. 62 (1853)。Morse 的 Claim 8 试图垄断一切用电磁力远距离通信的手段，远超出他实际披露的电报装置。上位 claim 没有对应披露。今天 “a neural network configured to diagnose…” 这类写得极宽、说明书却撑不住的 claim，问题结构还在。

### 范围结构

权利边界是 claim，不是说明书故事。说明书、附图、实施例用来解释 claim。Independent claim 最宽，dependent claim 往里加 limitation，范围变窄、有效性通常更好证。

Claim 可以把产品写成装置、方法、系统或计算机可读介质。每个 claim 是一组 limitations 的组合。Independent claim 自己写齐必要要素；dependent claim 引用前一项，再加限制，因此法律范围一定更窄。说明书中的漂亮功能如果没有写进 claim，通常不能单靠摘要或背景段落拿来主张侵权。

### 解释与比对

Claim construction（权利要求解释）先决定词是什么意思。Intrinsic evidence 的顺序通常是 claim 本身、specification、prosecution history；字典、专家等 extrinsic evidence 次之。解释完成后，才把被控产品逐项映射到 limitations。

字面侵权：被控产品满足全部 limitations（all-elements rule  /  all-limitations rule）。少一个就不字面侵权。Doctrine of equivalents 防止改个无关紧要的零件就绕开。*Warner-Jenkinson*（1997）收到逐元件对等。Prosecution history estoppel：审查时为了避开 prior art 而缩小的范围，后面不能用等价理论要回来。所以申请过程中怎么改 claim、怎么争辩，诉讼里都会被翻出来。

*Markman v. Westview Instruments*, 517 U.S. 370 (1996)。Claim construction 是法官的工作，不是陪审团的。专利诉讼前半场经常在 Markman hearing 里已经定胜负。同一个词在说明书里出现过几次、有没有定义、审查时怎么用，都会进 claim construction。

15.628J 的 patent-search assignment：拿一份刚授权的美国专利，只盯 Claim 1，四小时内找可能破坏 § 102 / § 103 的 prior art，写 2–3 页。日期必须按 prior-art 定义来，不能拿申请日之后的文献打新颖性。换成一件 CV / LLM / agent 专利做同一件事，会发现摘要写得很未来，真正卡住的是某句很具体的系统步骤。摘要几乎没有法律意义。

<br>

## 侵权

### 抗辩结构

被告首先可以主张专利因 § 101 / 102 / 103 / 112 无效，也可以在承认专利可能有效的前提下，主张产品没有满足全部 limitations。其他抗辩包括许可、exhaustion、以及 inequitable conduct。后者不是一般的“漏交材料”，而要依 *Therasense v. Becton*（Fed. Cir. 2011）证明通常具有 but-for materiality 的重要遗漏或虚假陈述，并有明确欺骗 USPTO 的意图；成立后涉及专利不可执行。*SCA Hygiene v. First Quality*（2017）以后，laches 不能阻止 § 286 六年追溯期内的专利损害赔偿，但 equitable estoppel 等其他理论仍可能适用，迟延也可能影响部分衡平救济。

### 行为类型

Direct infringement 的常见结构：

- **产品/系统 claim**：制造、使用、销售、许诺销售或进口被 claim 覆盖的产品。
- **方法 claim**：原则上必须有人实施全部步骤；多主体分别执行时会出现 divided infringement。
- **induced infringement**：知道专利、并有意促使他人直接侵权。
- **contributory infringement**：提供专门用于侵权、且没有 substantial non-infringing use 的关键部件。

专利侵权通常不要求被告抄袭，也不要求事先看过专利。独立研发不是不侵权抗辩；但知情与否会影响诱导、故意侵权和增强赔偿。

### 例外与用尽

*Roche v. Bolar*, 733 F.2d 858 (Fed. Cir. 1984)。仿制药实验使用仍是侵权。国会随后用 Hatch-Waxman 开出 Bolar exemption。实验使用例外在美国极窄。“做研究所以不侵权”不是美国专利的默认。版权的 fair use、专利的 experimental use 是两套规则，不能串用。

*Bowman v. Monsanto*, 569 U.S. 847 (2013)。农民买了抗除草剂大豆，种一代可以。把收获再种，不行。专利用尽管的是售出的那一件，不是复制下一代。对应到数字物：买到一份权重，不等于买到复制、再训练、再分发的权利，要看许可证怎么写。

<br>

## 救济

救济分成两个时间方向：

- **过去损失**：lost profits、reasonable royalty，以及严重故意侵权时可能适用的增强赔偿。
- **未来行为**：是否按 *eBay* 四要素签发禁令。

35 U.S.C. § 284 要求赔偿至少达到 reasonable royalty。权利人若能证明没有侵权时本可取得的销售，还可请求 lost profits。*Georgia-Pacific* 列出合理许可费的一长串因素，实务里主要靠假想谈判和专家证言。依 *Halo Electronics v. Pulse*（2016），法院对 egregious willful infringement 可酌情将赔偿提高到最高三倍，不是认定故意后自动倍计。

Lost profits 试图还原“没有侵权时，专利权人本来能赚多少”；常用的 *Panduit* 框架检查需求、可接受的非侵权替代品、生产营销能力和利润额。Reasonable royalty 假设侵权开始前双方作为自愿许可人与被许可人谈判，会同意什么费率。禁令和赔偿解决不同问题：赔偿处理过去损失，禁令限制未来行为；法院不发禁令时，还可能对判决后的继续实施确定 ongoing royalty。特殊案件还可能依 § 285 判给律师费。

禁令曾经几乎自动。*eBay v. MercExchange*, 547 U.S. 388 (2006) 废掉这个惯例，改回普通法四要素：不可挽回损害、法律救济不足、衡平法权衡、公共利益。NPE 靠禁令要挟许可的空间缩小。经营实体仍比较容易拿到禁令，纯粹许可商难得多。15.628J Short Exercise 3 就是拆这四个要素。

对生成模型公司，版权侧的 statutory damages 按作品件数相乘，往往比专利 royalty 更早变成难以承受的敞口。*Bartz* 后来能和解到这个量级，背景是这个乘法，而不只是 Alsup 那句 *spectacularly so*。

<br>

# 商业秘密

本章按商业秘密从成立、被不当取得到人员流动和救济的顺序展开：

- **成立要件**：秘密性、因秘密而产生的经济价值、合理保密措施。
- **不当取得**：区分 improper means、独立发现、合法反向工程和一般技能。
- **保密措施**：NDA、权限、日志、离职回收与公开权重。
- **人员流动**：以前雇员携带文件和知识为中心的民事、刑事路径。
- **救济与竞合**：禁令、损害赔偿，以及版权、合同和商业秘密请求的并行。

## 成立要件

州法走 Uniform Trade Secrets Act。联邦 2016 年 DTSA，结构相近：信息因不为公众所知而有独立经济价值，权利人采取了合理保密措施。

Trade secret 要持续满足三项核心条件：

1. 信息没有被一般公众或能从其使用中获利的人普遍知晓、也不容易通过正当手段查明。
2. 信息正因为秘密而有独立经济价值。
3. 权利人采取了在情境下合理的保密措施。

对象可以是公式、源代码、客户名单、训练数据、负样本、系统提示或失败实验。法律不要求绝对秘密，也不要求军事级安保；但至少要能说明谁可访问、是否有 NDA、仓库权限、日志、离职回收和对外披露控制。公开论文、GitHub 提交或无保密限制的演示，会削弱秘密性。

## 不当取得

Misappropriation：不正当获取、披露、使用。独立发现不构成。合法获得实物之后的 reverse engineering，一般不构成。最常见的争点是 employee mobility：前雇员脑子里带走的，哪些是一般技能，哪些是雇主的秘密。

**Improper means（不正当手段）**包括盗窃、贿赂、虚假陈述、违反保密义务、电子入侵等。正常观察公开产品、独立研发、合法 reverse engineering 原则上是正当手段。员工记住的一般技能和经验不能都变成前雇主的永久财产；诉讼要具体指出 secret 是什么，不能只说“整套业务知识”。

## 保密措施

NDA（non-disclosure agreement）是保密协议。它能约定当事人如何使用、披露信息，但合同写了 confidential 不会自动让所有内容满足 trade secret 的法定条件。反过来，没有 NDA 也不必然排除商业秘密；法院会综合看信息性质、关系和实际保密措施。

## 人员流动

*Wexler v. Greenberg*, 399 Pa. 569 (1960)。化学配方那类。只把“在这家公司做过”当成秘密，雇佣市场会停。一般技能、经验、记忆，切出去。

*Waymo v. Uber* 是 Stanford CS202 明列的 employee movement 案。Waymo 主张前工程师 Anthony Levandowski 离职前下载逾 14,000 份自动驾驶与 LiDAR 文件，随后创办 Otto；Uber 收购 Otto，并让 Levandowski 负责自动驾驶项目。民事案在 2018 年陪审审理第五天和解：Waymo 获得 Uber 0.34% 股权，当时约值 2.45 亿美元；Uber 同意确保其硬件和软件不纳入 Waymo 机密信息。和解没有逐份裁定哪些文件构成商业秘密，也不是法院认定 Uber 实际使用了全部文件。

Levandowski 的刑事案是另一条程序。他不是 Waymo 对 Uber 民事案的被告；联邦检方后来单独起诉。2020 年他对一项 trade-secret theft 认罪，涉及一份 “Chauffeur Weekly Update”，被判 18 个月监禁。CS202 目录写 “civil and criminal trade secret theft”，指的是这组相连事实，不是说一份判决同时处理民刑责任。

离职带 disk、repo、数据，是 trade-secret 诉讼的标准事实模式。模型实验室对应的是未公开权重、训练数据、配比、过滤规则、内部评测集、系统提示。已发论文的算法、公开 benchmark、开源框架里的常规实现，通常不构成。

合理保密措施必须实际执行：权限、NDA、访问日志、生产权重的存储控制。措施不必完美，但完全没有就很难主张。向公众无保密限制地发布 open-weight model 后，公开权重本身的秘密性基本结束；只向负有保密义务的有限对象提供受控访问，则不一定立即灭失。未公开的数据、后训练流程和产品层信息仍可能分别构成秘密。

## 救济与竞合

DTSA 是联邦民事诉因，不取代州法，适用于与州际或对外商业中使用、拟使用的产品或服务有关的商业秘密，许多案件会同时主张州法和联邦法。诉讼时效通常是发现或应当发现 misappropriation 后三年；持续的不当行为原则上视为一个请求。救济可包括禁令、实际损失、不当得利、合理许可费；故意、恶意 misappropriation 还可能有倍计赔偿和律师费。极端情况下 DTSA 有 ex parte seizure，但门槛很高，不是普通取证工具。DTSA 还规定特定举报披露的免责，并要求雇主在相关保密协议中通知该免责；遗漏通知会限制部分救济。

模型抽取、蒸馏、访问内部 API，可能同时涉及版权、合同和商业秘密。同一串 HTTP 请求可以同时构成多个诉因的事实基础。

<br>

# 商标

本章沿“标志是否受保护—被告使用是否造成混淆—功能性是否排除保护—模型场景如何适用”的顺序展开：

- **来源识别**：商标保护的对象、使用要求、注册效果和续展。
- **显著性**：从 fanciful 到 generic 的强弱序列。
- **混淆与淡化**：侵权要件、多因素测试和成名商标保护。
- **功能性**：trade dress 不能把功能设计永久私有化。
- **模型场景**：输出内容与模型、公司命名分别触发什么问题。

## 来源识别

Lanham Act。不管创意聪不聪明，问消费者会不会以为来源有关联。可以无限续展，前提是还在用，还没变成 generic。Aspirin、Escalator 是美国法中常见的通用化案例；Thermos 的法律与市场历史更复杂，适合作为通用化风险而非全球统一结论。功能性设计走专利，不能用商标把功能锁成永久。

Trademark 保护的是 **source identifier（来源识别）**。词、logo、包装、声音、颜色都可能成为标志，但要能让消费者把它和某一商业来源联系起来。显著性从强到弱通常分：

- **fanciful**：臆造词，例如原本不存在的词；
- **arbitrary**：常见词用在无关商品上；
- **suggestive**：需要联想才能想到商品特点；
- **descriptive**：直接描述特点，通常要证明 secondary meaning（消费者已把它认作来源）；
- **generic**：商品通用名称，不能取得商标权。

## 注册与显著性

美国权利以 use in commerce 为基础，联邦注册会带来程序和全国范围上的优势，但注册本身不能挽救 generic 或 functional 标志。

## 混淆与淡化

商标侵权可以按顺序判断：原告是否拥有有效标志及优先权；被告是否在商业中把相关符号作商标性使用；这种使用是否可能造成消费者混淆；最后再看 descriptive fair use、nominative fair use 等抗辩和救济。只是为了描述产品属性而使用普通词，与把该词当品牌使用，不是同一行为。

混淆是多因素：标志相似度、商品接近程度、实际混淆证据、被告意图、消费者注意程度。*Lois Sportswear v. Levi Strauss*, 799 F.2d 867 (2d Cir. 1986)，处理的是牛仔裤后袋缝线构成的商业外观。15.628J 用它作为入门案例，不代表商标法只有混淆一条。成名商标另有淡化（blurring / tarnishment），不必证明来源混淆。生成带 logo 的图，可能同时涉及混淆和淡化。

## 功能性

Likelihood of confusion 不是只比两个 logo 像不像。还要看商品/服务是否接近、销售渠道、购买者注意程度、实际混淆、标志强度和被告意图。Trade dress（商业外观）保护产品或包装的整体来源识别，但必须 nonfunctional；若某设计主要影响成本、质量或用途，不能借商标取得永久排他。

## 模型场景

生成模型输出带 logo、角色外形、包装的图，可能同时涉及版权（角色、美术）和商标（来源混淆、假冒）。*Disney v. Midjourney* 主诉是版权，商标问题另看用户会不会误以为这是迪士尼的官方生成器。模型名、公司名自身的商标冲突是另一层，和训练无关，和命名有关。

<br>

# 归属

本章先把“权利是否成立”和“权利归谁”分开，再处理实验室里最常见的三类归属问题：

- **权利分层**：作者、发明人、受让人和署名者并非同一法律身份。
- **职务成果**：版权的 work made for hire、专利转让和共同权利。
- **人员流动**：竞业、保密、发明转让与禁止招揽分别限制什么。
- **开源许可**：软件、权重和合同义务的边界，以及“开放”的不同层级。

## 权利分层

“作品/发明受不受保护”和“权利归谁”是两道题。先有作者或发明人，再看雇佣关系、合同、书面转让和法定规则是否把权利移给学校或公司。论文署名、Git commit、项目负责人身份都不能自动代替法律上的 authorship、inventorship 或 assignment。

## 职务成果

### 版权归属

权属不靠口头默契。美国版权：雇员在职务范围内创作，走 statutory work made for hire，雇主视为作者。委托作品只有落入 § 101 列举的九类、且有书面约定，才是 WMFH。计算机程序本身不在这九类里。不属于普通法雇员的外包人员、实习人员或访问学生写的代码，通常需要 § 204 的书面转让，不能假定自动归实验室。

### 专利归属

专利没有同等的自动归属。*Stanford v. Roche*（2011）：Bayh-Dole 并不自动把联邦资助发明的权利归大学，还是要靠发明人的书面转让。斯坦福当年那份表格的措辞（“agree to assign” 还是 “hereby assign”），和 Roche 先前拿到的转让，谁在先，谁赢。实验室入职时的 assignment 文本，措辞和签署时间会直接决定权属。

### 共同权利

共同发明：对至少一项 claim 的 conception 作出实质性贡献的自然人才是 joint inventor。漏列、错列会影响权属、诉讼资格和专利有效性，但不一定当然毁掉专利；35 U.S.C. § 256 允许在符合条件时更正 inventorship，能否更正以及是否伴随欺骗行为会影响后果。共同作者是另一套规则。Copyright 里 jointly authored work 还要求各方有意把贡献合并成统一整体；原则上每个人可以发放非独占许可，但要向其他共同作者结算。美国专利默认相反：35 U.S.C. § 262，各共有人可自行实施、许可，原则上不必向其他共有人结算，除非另有约定。论文作者列表和发明人列表对不齐很常见，但专利仍须逐项 claim 识别真实贡献者。

## 人员流动与合同

人员流动中的合同条款要按限制对象区分：

- **Non-compete**：限制离职后为竞争者工作。
- **NDA**：限制披露或使用约定的机密信息。
- **Invention assignment**：将约定范围内的职务发明或作品权利转让给雇主。
- **Non-solicit**：限制招揽客户或员工。

Non-compete。加州 Business and Professions Code § 16600 长期对竞业限制采取非常严格的立场，人才流动本身被当成创新政策。FTC 2024 年发布全国性 non-compete rule，但联邦地区法院撤销该规则并阻止其全国生效，因此现行可执行性仍主要回到州法。NDA、invention assignment 和 non-solicit 也不是当然有效；尤其在加州，customer non-solicit 同样可能受 § 16600 限制。去相邻实验室继续做同一条产品线，真正会打的往往是带走的文件、数据和 NDA，而不是竞业条款本身。

Non-compete 禁止离职后去竞争者工作；NDA 禁止披露约定的机密；invention assignment 转让职务发明；non-solicit 限制带走客户或员工。四类条款的对象不同。NDA 也不能把公开信息、一般技能或依法受保护的举报永久变成秘密。

## 开源许可

### 权利对象

GPL、MIT、Apache 2.0 管的是软件作品。模型许可是另一份法律文本。GPL 义务通常在复制、修改或传播 covered work，或者传播基于它形成的作品时触发；不能简化为“只要源码衍生就触发”。权重、LoRA、checkpoint 算不算 derivative work，至少要先问其中是否包含或改编了可识别的受保护表达，以及基于哪一部作品，目前几乎没有判决。研究中把“模型开源”直接理解成“和 Linux 一样”，容易读错权利范围。Creative Commons 主要处理版权而不自动提供专利许可；它是否已经授权训练，要看训练触及哪些版权行为、许可覆盖范围、具体版本及是否满足条件。模型或输出是否触发署名、相同方式共享等传播条件，仍未被法院系统回答。

### 转让、许可与合同

Assignment（转让）把权利所有权移给受让人；license（许可）只允许被许可人在约定范围内实施特定行为；合同义务还要求当事人之间形成合同关系。许可证不是把版权“送掉”，而是权利人预先允许特定行为并附条件。MIT 主要要求保留版权和许可声明；Apache 2.0 另含明确专利许可和终止条款；GPL 要求分发 derivative work 时按 GPL 提供对应源代码。违反条款究竟构成版权侵权还是合同违约，要看它是许可 condition（条件）还是独立 covenant（承诺），以及合同是否成立。

### 开放层级

“Open source”在 OSI 语境有明确标准。只允许研究、限制用户规模、禁止竞争或禁止某些用途的 model license，即使权重可下载，也通常只是 source-available / open-weight，不等于开源软件许可。

<br>

# 信息政策

本章从政策分析方法进入六类公法和监管问题：

- **政策方法**：从技术事实、利益相关方和现行规则推导可执行建议。
- **言论**：国家行为、言论类型和审查强度。
- **平台**：§ 230、推荐与生成、政府干预和合成内容。
- **搜查与访问**：第四修正案、第三人数据、CFAA 和网页抓取。
- **隐私**：法域结构、核心概念与匿名化。
- **跨境**：连接点、实体法、法院管辖、承认执行和数据流动。

## 政策方法

Internet 不是一件产品，是网络、市场、社交空间、信息系统和政治机构叠在一起。Lessig 那句 code is law 是分析起点：架构先决定什么行为容易发生，法律再把责任和成本压回架构。

信息政策比知识产权更宽。它不只问“谁拥有内容”，还问政府能否限制言论、平台是否为用户内容负责、执法机关如何取得数据、公司如何收集个人信息，以及不同国家的规则怎样同时适用。同一行为可能同时落入私法和公法：平台删帖涉及合同和言论治理；政府命令平台删帖才会直接出现 state action 与第一修正案问题。

技术设计改变用户能做什么。用户行为改变法律要不要出手。法律再逼架构改。加密默认开，执法换一种理由要后门。平台把转发做成一键，诽谤和版权的间接责任理论全部重写。推荐算法把“展示谁的内容”变成平台自己的编辑动作，§ 230 还管不管，变成 2020 年代的问题。

政策分析不满足于找法条得到答案。基本顺序是：

- **技术事实**：系统实际收集、复制、生成或传输了什么。
- **利益相关方**：用户、平台、权利人、政府与公众分别承担什么成本。
- **现有规则**：私法、公法和行业监管如何同时适用。
- **备选方案**：立法、执法、合同、架构和市场机制各能解决什么。
- **架构反馈**：方案会怎样改变权限、日志、过滤、推荐和数据保存。
- **社会后果与建议**：比较表达自由、隐私、创新、竞争和执行成本，再提出可操作选择。

建议要落到谁执行、用什么工具、失败了会怎样。

<br>

## 言论

First Amendment 主要限制政府，不直接要求私人平台提供发言空间。分析顺序先问有没有 **state action（国家行为）**，再问受限制的是何种 speech、规则是按内容还是内容中立、适用什么审查强度。私人公司依据 ToS 删帖，通常不是第一修正案诉讼；政府胁迫平台删帖，问题才不同。

Content-based restriction（基于内容的限制）通常适用 strict scrutiny：政府须证明迫切利益，手段还要 narrowly tailored。诈骗、真正威胁、部分诽谤等类别保护较弱；“网上内容令人不适”本身不是把互联网降到广播监管强度的理由。

*Reno v. ACLU*, 521 U.S. 844 (1997)。《通信规范法》（Communications Decency Act）中针对 indecent、patently offensive 网络表达的条款过于模糊、宽泛。最高法院拒绝把互联网等同于受特殊监管的广播媒介，给予其高度第一修正案保护，并判相关条款无效；不是整部法律都被推翻，同一法律中的 § 230 保留下来。

互联网从一开始就没有被当成“广播，所以政府可以严格管内容”。后面关于平台内容管制的讨论，起点仍是互联网表达受到高度第一修正案保护，而不是机械套用广播监管标准。

<br>

## 平台

### 第 230 条

47 U.S.C. § 230(c)(1)：交互式计算机服务不因其用户的内容而被当成 publisher 或 speaker。§ 230(c)(2) 给善意审核一个安全港。*Zeran v. AOL*（4th Cir. 1997）对这项保护作了较宽解释：即使平台已经收到通知、知道某条帖子涉嫌诽谤，仍可能受保护。平台治理因此主要通过私人规则运行：ToS、推荐和审核团队。

§ 230(c)(1) 常拆成三问：被告是否是 interactive computer service；原告是否要把它当 publisher / speaker 追责；相关信息是否由 another information content provider 提供。平台若自己负责创作违法内容的实质部分，不能把那部分算成纯用户内容。§ 230 也有明确例外，例如联邦刑法、知识产权和部分性交易法律；知识产权例外是否包括州知识产权法，各巡回处理并不完全一致。它不是互联网公司的普遍免责条款。

§ 230 和 DMCA § 512 不同。§ 230 主要处理用户内容引起的州法责任，如诽谤；DMCA § 512 专门处理版权，并要求 notice-and-takedown 等条件。收到通知后 § 230 仍可能保护平台，不代表版权通知也可以不处理。

### 推荐与生成

政府若强迫平台删除或保留某些合法言论，可能违反第一修正案。政府若把平台当成用户内容的发布者追责，又可能受 § 230 限制。具体争议往往落在某一审核动作属于私人决定还是 state action。

*Gonzalez v. Google*（2023）本来可能重写推荐算法还受不受 230 保护。最高法院没接这道题，跟着 *Twitter v. Taamneh* 把相关责任问题先绕开。推荐是不是平台自己的 speech，仍悬着。

*Moody v. NetChoice*（2024）进一步说明，平台对信息流的选择、排序和呈现可能构成受第一修正案保护的编辑判断，但法院必须针对具体产品功能和法律适用范围分析，不能把“整个平台”抽象地统一归类。它解决的是州政府限制平台内容审核的宪法问题，不直接回答 § 230 是否保护生成式输出。

生成式 AI 让 publisher / speaker 的划分不好套。模型输出在多大程度上是用户的 speech，多大程度上是公司的 speech，直接决定 230 能不能用。把 ChatGPT 的回答当成“用户发布的内容”，和把推荐流里的推文当成用户内容，并不是同一件事。用户只打了一句提示，输出由模型参数和公司的安全策略共同决定。2024 年以后的诉讼卡在这点上，还没有 *Reno* 级别的终局。

### 合成内容

深度伪造、虚假信息，6.4590 放在 Week 8 内容审核和后来的 AI 周，不放在纯宪法课。技术能力先出现，归责规则随后形成。Copyright Office Part 1 建议为个人形象单独立法，而不是强行纳入版权法。中国采用标识义务，并禁止删除、篡改、隐匿标识。两种路径都说明：版权不是处理“这张脸像谁”的主要工具。

<br>

## 搜查与访问

### 政府搜查

Fourth Amendment 管政府搜查扣押，不管公司隐私政策。两套问题并成一句“隐私”，后面容易混。

第四修正案先问政府行为是否构成 **search or seizure**。传统上看是否侵入人身、房屋、文件、财物，后来又发展出 reasonable expectation of privacy。若构成搜查，通常需要基于 probable cause 的 warrant；但 consent、紧急情况、逮捕附带搜查等例外很多。“公司持有数据”并不自动回答政府调取是否需要令状。

### 第三人数据

第三人原则。*Smith v. Maryland*（1979）：电话号码交给电话公司，政府用 pen register 调取，不构成 search。数字时代大量数据都会交给第三方。*United States v. Jones*（2012）涉及在车底安装 GPS 长期追踪；多数意见以政府为安装设备而物理侵入受保护财产为基础，长期监控理论主要见于协同意见。*Carpenter v. United States*, 585 U.S. 296 (2018) 中，政府实际取得约 127 天的 CSLI（基站位置记录）；最高法院认定取得至少七天的历史 CSLI 构成 search，通常需要令状。Roberts 写：数字时代第三人原则不能无限延伸。6.4590 Week 4 的问题是，自动交给手机公司的位置是否仍像 *Smith* 中的号码一样属于自愿交付。多数意见没有回答更短时间窗口，也没有为所有第三方数字记录建立一般规则。Agent 连续上报坐标会落在这条光谱的哪一侧，目前没有同级别判决。

位置数据、云端对话日志、agent 的浏览记录，落在这条光谱上。公司依用户协议收集，是合同、州隐私法、消费者法。政府向公司调取，是第四修正案和 Stored Communications Act。同一份 ChatGPT 日志，两次法律评价。

### 访问边界

CFAA 管未经授权的访问。*Van Buren*（2021）把“违反使用政策就算黑客”收窄。警察查了一次他有权访问、但不该查的数据库，不构成 CFAA 意义上的超授权。

CFAA 的核心词是 **without authorization** 和 **exceeds authorized access**。*Van Buren* 采用 gates-up-or-down 的理解：账号本来有权进入某个数据库，仅仅为了不当目的查询，通常不等于越过技术上的访问边界。伪造账号、盗取凭据、绕过登录或继续突破明确技术屏障，事实更接近未经授权。违反 ToS 可以构成合同问题，但不能自动升级成联邦计算机犯罪。

*hiQ Labs v. LinkedIn* 是 Stanford CS202 明列的 scraping / CFAA 案。hiQ 抓取任何浏览器都能看到的公开职业资料，用于员工流失与技能分析。LinkedIn 发 cease-and-desist，并用技术措施阻断。第九巡回 2022 年在 *Van Buren* 发回后再次维持 preliminary injunction：hiQ 至少提出了严重问题，公开资料没有 CFAA 所要求的那道授权 gate，访问它一般不像 “without authorization”。

这不是对所有 scraping 的终局豁免。判决阶段只是 preliminary injunction，也没有解决合同、trespass、隐私、版权等全部请求。案件后来以 consent judgment 收尾：hiQ 同意 50 万美元判决、停止被禁止的抓取并删除相关数据和工具。该和解不推翻第九巡回关于公开页面的 CFAA 分析，也不产生先例；同时，hiQ 使用假账号进入 password-protected pages 的行为与纯公开页面抓取不能混为一谈。

所以 Agent 抓网页至少要分三层：

- **公开且无登录**：通常缺少 CFAA 所要求的授权 gate，但仍可能涉及合同、版权、隐私和服务器资源问题。
- **违反网站条款但没有绕过访问门**：可能构成合同争议，不会仅因用途不当就自动升级为 CFAA 犯罪。
- **伪造账号、绕过登录或技术屏障**：更接近 without authorization，需要结合凭据、技术措施和收到通知后的行为分析。

CFAA、合同、版权、隐私可能给出四个不同答案。robots.txt 也不自动等于 CFAA 的授权门。漏洞报告同样可能既是研究、又是 ToS 违约、又涉及未经授权访问。

<br>

## 隐私

### 法域结构

美国没有统一联邦隐私法典。FTC Act § 5、HIPAA、FERPA、COPPA、加州 CCPA/CPRA、Illinois BIPA 和其他州法共同组成分行业、分州的结构。HIPAA 只直接约束 covered entities、business associates 等法定主体，并不是所有健康信息的一般隐私法。欧盟 GDPR 是综合权利框架：合法基础、目的限制、最小化、跨境、自动化决策。中国主要涉及网安法、数安法和个保法。训练把“处理”从保存文件扩展到参数更新，删除权和遗忘权因而出现技术困难：模型不是文件夹，删除一条原始语料不等于参数和输出能力中不再保留其影响；但也不能预设个人数据一定能够从参数中识别或恢复，仍须检查具体模型和处理事实。

### 核心概念

几个概念需要分开：

- **personal data / personal information**：能直接或间接关联自然人的信息，不只姓名和身份证号。
- **controller / processor**：前者决定处理目的和方式，后者受托处理；义务不同。
- **legal basis**：GDPR 下处理必须有同意、合同必要、法定义务、legitimate interests 等合法基础之一。
- **purpose limitation**：为某目的收集的数据，不能无条件改作完全不同用途。
- **data minimization**：只处理完成目的所必要的数据。
- **deletion / erasure**：删除原始记录相对直接；数据是否已被模型参数记忆、是否能从输出恢复，是另一技术问题。

### 匿名化

匿名化与去标识也不是一回事。真正匿名化后无法合理重新识别的数据可能不再是个人数据；仅删除姓名、换成编号，若仍可结合其他数据还原个人，通常只是 pseudonymization。

<br>

## 跨境

跨境问题应先列连接点，再按三项法律问题和一项数据合规问题依次判断：

- **连接点**：用户、服务器、公司、资产、权重、日志和微调数据分别位于哪里。
- **实体法**：哪个国家或地区的法律调整具体行为。
- **法院管辖**：哪个法院能对被告及争议行使审判权。
- **承认执行**：取得判决后，能否在资产、数据或主体所在地实际执行。
- **数据流动**：跨境传输是否另需标准合同、充分性决定或安全评估。

### 连接点

用户在一国，服务器在另一国，公司在第三国，weights / logs / fine-tune data 分布多处。适用法会直接决定同一套发布能否上线。GDPR 的域外适用、中国针对向境内提供服务的规则、美国法院的长臂管辖，可能同时出现在一个请求中。出口管制另算：先进芯片、部分封闭权重。6.4590 把 trade policy 和 international affairs 单列，因为架构决策经常同时受到布鲁塞尔、华盛顿、北京的规则影响。同一套开源发布，在三个市场需要的合规动作不同。

### 跨境三问

- **第一问：实体法**
  - 哪个国家或地区的实体法适用于被诉行为。
  - 知识产权具有属地性，应分别定位复制、实施、销售、进口、展示或混淆发生地。
  - GDPR 域外适用、向境内用户提供服务等规则还会根据用户、经营活动和目标市场建立连接。
- **第二问：法院管辖**
  - 哪个法院可以受理争议并对被告作出有约束力的判决。
  - Personal jurisdiction（属人管辖）问被告与法院地是否有足够联系。
  - Subject-matter jurisdiction（事项管辖）问该法院是否有权审理这一类案件。
- **第三问：承认执行**
  - 一地法院取得的判决能否在另一地被承认。
  - 即使胜诉，也要确认资产、数据和相关主体位于何处，以及当地是否提供实际执行手段。

Personal jurisdiction（属人管辖）问法院能否对这个被告行使审判权，通常要求被告与该地有足够联系。它和 subject-matter jurisdiction（事项管辖）不同：后者问这个法院是否有权审理这一类案件。知识产权还具有属地性，美国专利和商标法原则上调整与美国有足够联系的行为，不会因为公司在美国就当然覆盖全球全部行为。

### 数据流动

数据传输还可能单独需要标准合同条款、充分性决定或安全评估。服务器放在哪里只是连接点之一，不会自动排除用户所在地和公司经营所在地的法律。

<br>

# 人工智能

本章把前述制度用于 AI，但仍以具体行为而不是“模型”作为分析单位：

- **分析框架**：拆行为、对象、主体、权利、抗辩、证据和程序阶段。
- **训练**：分别处理竞争用途、副本来源和市场损害。
- **输出**：判断复制、实质性相似、文本复述、角色与图像。
- **作者与发明人**：区分人的表达控制和对 claim 的 conception。
- **比较法**：比较 fair use、TDM 例外、合法访问、退出和透明度。

## 分析框架

### 行为拆分

先分开被诉的行为。

AI 不是独立的法律部门。训练、保存语料、提供模型、生成输出和分发输出是不同 actor 在不同时间做的行为。每个行为都要重新确定对象、权利、责任人和抗辩，不能用“模型学过作品”直接推出“每次输出都侵权”，也不能用“输出没有复述”反推训练阶段没有发生复制。

“模型构成侵权”至少要拆成四件：

1. 训练时制作复制件
2. 把盗版库当永久图书馆（存，不一定训）
3. 输出与作品 substantial similarity（复述、角色、逐页替代）
4. 把角色或商标当成产品功能来卖

四件事的构成要件和证据不能混用；fair use、许可等抗辩可能在不同行为中重复出现，但必须分别适用。训练复制要按 § 107 衡量。永久保存也不是当然排除 fair use，要看副本来源、是否增加副本数量和图书馆用途；*Bartz* 对影子库下载与购书后的等量数字替代给出了相反结论。输出侵权要过 copying + substantial similarity。把角色生成作为产品卖点，则还可能涉及直接侵权和 *Grokster* inducement。

### 对象与主体

分析需要按对象拆，不能从“模型”一次跳到结论：

Copied or produced what？idea / 事实 / expression / 代码 / 整本书 / 权重 / 输出 / 角色 / CMI。

随后确定该行为可能触及 § 106 中的哪一项专有权。再看谁是 actor：实验室、云厂商、开源发布者、下游应用、用户、agent。抗辩是 fair use、许可、安全港，还是根本没有适格作者。

### 证据清单

可以固定成一张检查表：

1. **行为**：下载、缓存、tokenize、训练、保存、微调、部署、输出、公开展示、分发。
2. **对象**：事实、思想、受保护表达、代码、角色、logo、个人数据、商业秘密。
3. **主体**：数据提供者、模型开发者、云厂商、应用开发者、用户。
4. **权利或义务**：§ 106 哪一项、合同、CFAA、隐私法、商标、商业秘密。
5. **抗辩**：许可、fair use、safe harbor、独立创作、无实质性相似。
6. **证据**：数据来源记录、训练管线、输出测试、过滤措施、通知后处置、市场替代证据。
7. **程序阶段**：诉状、取证、summary judgment、审判、和解。

### 判例基线

美国没有国会层面的 AI 训练例外。争点落在 § 107。到 2026 年秋，地区法院已经给出三条不能互相化约的线。都不是最高法院。第三巡回听过 *Ross* 的辩论，还没下判。

<br>

> [!WARNING]+ 2025 年北加州两份意见不是规则
>
> *Bartz*、*Kadrey* 都是 N.D. Cal. summary judgment，且都写了本案记录。Chhabria 自己说这份裁定不等于 Meta 对所有权利人都合法。*Ross* 是 Delaware、非生成式、直接竞品，结论相反。引用时需写明法院、用途、来源、有没有输出主张。

<br>

训练诉讼出现之前，最接近的先例是整本扫描。*Authors Guild v. Google*, 804 F.3d 202 (2d Cir. 2015)。Google 未经许可扫描数千万册，做全文检索和 snippet。Leval 写：用途 highly transformative；整本复制对检索是必要的；snippet 不构成原作的实质性替代；商业动机不自动击败 fair use。作者主张的“检索许可市场”，法院不认。*HathiTrust* 那条图书馆扫描线更早一点，结构相近。Alsup 在 *Bartz* 里把训练写成更强的 transform，用的就是这条家族的语言。差别在于：Google Books 给用户看的是检索和碎片，不生成新的书；LLM 把书压进参数之后，可以写出大量二手文本。Chhabria 在 *Kadrey* 里抓的就是这一点。

<br>

## 训练

训练阶段最容易混淆四个问题：

- **获取副本**：抓取或下载语料时是否制作了未经许可的复制件。
- **长期存储**：把数据保存在中央库是否具有与训练分离的用途。
- **训练使用**：把合法或非法来源的副本送入训练，是否构成 fair use。
- **参数状态**：模型参数能否固定、再现受保护表达，从而构成法律意义上的 copy。

模型参数本身是不是作品的 copy，不能只凭“信息进了权重”回答。

### 竞争

*Ross* 的重要性不在于模型是否生成文本，而在于训练材料和最终产品处于同一法律检索市场。原告把编辑 headnotes 授权给法律研究，Ross 用其制作功能相近的检索工具。第一因素的 purpose 和第四因素的 market substitution 因而连在一起。

*Thomson Reuters v. Ross Intelligence*（D. Del. 2025）。Ross 不是生成式模型，而是法律检索工具，目标是与 Westlaw 竞争。数据供应商使用 Westlaw headnotes 制作 Bulk Memos，供 Ross 训练。Stephanos Bibas 是第三巡回法官，经 designation（指定）在特拉华地区法院审理本案；这份意见在先例层级上仍是地区法院判决。

Headnotes 有 originality。复制成立。Fair use 不成立：商业、非变革性、直接做同一功能的竞品。第二、第三因素偏向 Ross（headnotes 保护窄，最终用户看不见那些 memo），压不过第一和第四。Bibas 自己写 *the questions here are hard*，准许中间上诉。第三巡回 2026 年 6 月 11 日开庭辩论，Ross 被反复追问：用途和 Westlaw 对用户来说是不是同一件事。至 2026 年 9 月尚未下判。这是目前唯一走进上诉审的训练 / 合理使用案件。

生成式或非生成式本身不是决定标准；最终产品是否直接替代原产品，会同时影响第一因素的 purpose 和第四因素的 market harm。不能把所有 training 当成一类用途。Ross 用别人的表达制作功能相同的检索器；*Bartz* 中 Claude 并不直接销售书籍副本。

<br>

### 来源

来源问题问的是“副本怎么得到”，用途问题问的是“得到后拿来做什么”。购买纸书、从授权数据库下载、从影子库下载，会形成不同的复制链条。后续训练可能有自己的 fair-use 分析，但不会自动消除此前下载盗版副本的责任。

*Bartz v. Anthropic*（N.D. Cal. June 23, 2025, Alsup, J.）。原告包括 Andrea Bartz、Charles Graeber、Kirk Wallace Johnson。Anthropic 先从 LibGen 一类影子库下载数百万册，后来改为购买二手书、拆装、扫描。原告主张训练复制和中央图书馆侵权，没有主张 Claude 输出大段复述。

Alsup 把三个行为分开：用书训练模型；购买纸书后销毁原件、制作等量数字替代件；从影子库下载并永久保存盗版副本。前两项被认定为 fair use，但理由不同；第三项不是 fair use。

训练这一支按 § 107 四个因素逐项写完。第一因素：*transformative—spectacularly so*。把 token 压进参数、学的是统计规律，不是把书拿去给用户读。类比是阅读和记忆：为了每次想起一本书、每次用新的方式写新东西就付一次钱，unthinkable。商业性在，但没压过变革性。第二因素偏向作者（小说等创造性作品），这项一向权重低。第三因素：整本复制，但对训练是 reasonably necessary，整本不自动致命——*Campbell*、*Sega* 都出现过这种结构。第四因素：保护的是替代原作市场，不是任何一种竞争。原告没主张 Claude 会逐页替代，Anthropic 有防复述。作者主张的“训练许可市场”，法院不认为是 Copyright Act 保证他们独占的市场。如果 fair use 成立，就不能靠“本来可以收训练费”反过来把第四因素赢回去。Chhabria 两天后在 *Kadrey* 里明确反对这种对第四因素的处理。

盗版库这一支不是 fair use。后续即使不用于训练，也不能使此前未经许可下载和永久保存的行为变成合法。与此不同，法院已认定购买纸书后销毁原件、制作不增加副本数量的数字替代件属于 fair use。

盗版库这一支后来按 15 亿美元和解。作品清单约 482,000 部，单部粗算 3,000 美元量级，销毁盗版数据集。Alsup 2025 年 9 月初步批准后，年底转 inactive，案随机分给 Araceli Martínez-Olguín。2026 年 7 月 20 日由 Martínez-Olguín 终局批准，律师费压到约 6.8%（原告曾从 20% 降到 12.5% 开口）。和解覆盖从 LibGen 等影子库下载的作品，不管输出，也不等于承认或推翻训练 fair use。法定赔偿上限按件相乘，理论暴露可以到公司存亡。Anthropic 曾想上诉 fair use 和集体认证，没等上诉就结了。

2025 年 6 月的 summary judgment 由 Alsup 作出；他在 *Oracle v. Google* 地区法院阶段处理过接口复制和 fair use。2026 年终局和解令则由 Martínez-Olguín 签署，二者不能混写成同一位法官的终局处理。

> [!EXAMPLE]+ Alsup 区分的三项行为
>
> 书籍 → 训练 LLM：fair use，*transformative—spectacularly so*。合法购书 → 销毁纸本并制作等量数字替代件：fair use，理由是更换存储格式且不增加副本总数。影子库下载 → 永久中央图书馆：不是 fair use。15 亿美元和解量化了特定作品清单所涉的既往盗版下载请求，但不构成法院对训练合法性或训练许可价格的判定。

<br>

### 市场

第四因素不只看模型会不会逐字吐书。至少有三类市场理论：

- **输出替代**：输出直接替代原作或其衍生市场。
- **训练许可**：未经许可训练侵占已经存在或传统上可能发展的训练许可市场。
- **市场稀释**：模型批量生成同类作品，造成 market dilution，压低原作者作品需求。

前两类在既有 fair-use 案例里较常见。第三类是 *Kadrey* 强调、但原告证据不足的一条。证明它需要经济和市场证据，不能只说模型“会竞争”。

*Kadrey v. Meta*（N.D. Cal. June 25, 2025, Chhabria, J.）。Richard Kadrey、Sarah Silverman、Junot Díaz 等十三位作者。Llama，Books3 一类影子库，还涉及 torrent。隔 *Bartz* 两天，同一个北加州。

训练在本案记录上被认定为 fair use，即使作品来自盗版库。和 Alsup 的分歧主要有两处。关于盗版来源，Chhabria 认为训练本身仍可构成 fair use，来源问题另行分析。关于第四因素，他认为市场影响最重要，并批评 Alsup 把训练类比成教孩子写作并不准确。LLM 能以很低的边际成本制造大量同类作品。Market dilution 理论认为，使用原作提高模型能力后，模型生成的大量同类作品会稀释原作市场；书籍具有训练价值，反而可能支持训练与市场稀释之间的因果联系。

但原告几乎没有为 market dilution 提交证据。Chhabria 明确限定判决范围：裁定只说明这十三位作者提出的理论和证据不足，*不*等于 Meta 对所有权利人的训练都合法。

Torrent seeding（把盗版书再分发出去）相关主张仍活着。2025 年底原告还在改诉状。DMCA CMI 请求在该案记录上失败：训练复制被认定为 fair use，且原告未建立移除 CMI 与掩盖其他上传侵权之间所需的证据联系。Chhabria 2026 年 7 月 8 日拒绝向第九巡回提交中间上诉，理由是预计终局判决后可以把问题一并上诉；这是一项程序判断，不保证具体裁判日期。训练问题目前没有西海岸的上诉审。

同一联邦地区、相隔两天的两份意见采用了不同路径。上诉法院作出判断之前，不能把“美国允许训练”写成统一规则。*Kadrey* 表明，原告若主张第四因素，需要提交市场稀释的实际证据，而不是只主张应当存在训练许可费。

<br>

## 输出

### 复制判断

输出分析与训练分析分开。先证明特定输出复制了原作，再比较 protectable expression。能提示出事实摘要，不等于能提示出受保护表达；能生成同类风格，也不等于复制具体作品。反过来，长段逐字复现、稳定出现独特角色设定或训练图水印，会提高 copying 与 substantial similarity 的证明力。

Memorization 是技术现象，infringement 是法律结论。某段文本被模型记住仍要问它是否受保护、输出是否由被告或用户实施、是否公开或分发、有没有 fair use。风格通常处于思想、技法或一般审美层，不等于角色、具体构图和台词也不受保护。

### 文本输出

*In re OpenAI Copyright Infringement Litigation*（S.D.N.Y. MDL 25-md-03143，Stein / Wang）。作者、新闻机构等案件合并审理，同时涉及训练复制和 ChatGPT 输出的摘要、大纲。纽约时报一案较早展示过近乎逐字的文章复述，这是输出问题，不等于已经回答训练是否具有 transformative purpose。2025 年 10 月，法院认为原告已经充分主张陪审团可能认定实质性相似的输出，把 fair use 留给后续程序。2025 年 11 月起，法院要求提供约 2000 万条去标识的 ChatGPT 对话样本；2026 年又要求提供其他规模更大的对话样本集合，包括约 7800 万和 1000 万条的集合。案件仍处于 discovery（证据开示）阶段。

### 图像与角色

*Disney v. Midjourney*（C.D. Cal.，2025 年起诉）。训练复制，以及服务持续生成、展示迪士尼 / 卢卡斯 / 漫威角色，并把这种能力当卖点。Willful、首页展示、没做过滤，诉状按诱导和直接侵权一起写。充分刻画且具有稳定、独特表达的角色，通常比抽象“风格”更容易主张；但仍须识别具体受保护元素，不能由“可识别”直接跳到必然侵权。

许可市场曾经被宣布过。2025 年 12 月 Disney 与 OpenAI 宣布：三年许可，Sora / ChatGPT Images 使用 200 多个迪士尼、皮克斯、漫威、星战角色，不含演员肖像和声音；Disney 拟投资 10 亿美元。当时文本写明尚待最终协议和交割条件。2026 年 3 月 OpenAI 关停 Sora 应用，交易取消，Reuters 等报道钱没交割。对第四因素仍是信号——有人愿意给角色生成定价——但不能写成已经稳定存在、正在收款的市场。Midjourney 那支诉讼还在。

更早还有 *Andersen v. Stability*、Getty 那批图像案。Stable Diffusion 输出中出现畸变的 Getty 水印，可以作为调查训练来源、特定输出相似性或来源混淆的相关线索，但单凭水印还不能证明模型保存了完整原图，也不能直接完成 memorization 或侵权的法律证明。图像和文字的 fair use 叙事不必同一套。

<br>

## 作者

### 人类控制

美国版权要求 human authorship。这是现行解释，不是政策偏好本身。

作者资格问的是谁对最终可感知表达作出原创控制。使用相机、画笔或软件不妨碍人成为作者，因为人仍决定表达。生成模型的难点在于 prompt 往往只描述目标，具体像素或措辞由系统决定。人的后期修改、选择、编排可以单独受保护，但不会因此把未修改的机器生成部分全部变成人的作品。

### 机器生成

*Thaler v. Perlmutter*。Stephen Thaler 的 Creativity Machine 生成《A Recent Entrance to Paradise》，铁轨伸进光里那种图。申请表上作者填机器，自己只当权利人。版权局拒登。D.D.C. Howell 法官：human authorship 是 bedrock。Work made for hire 也以存在可保护作品为前提，机器“受雇”产生不了第一步。D.C. Cir. 2025 年 3 月维持。Thaler 在行政程序里放弃了“自己因创造并操作机器而成为作者”，所以法院不处理人类辅助到什么程度才够。最高法院 2026 年 3 月 2 日拒签 certiorari。

就 Thaler 这名当事人把机器列为唯一作者的行政记录而言，诉讼已经结束；最高法院拒绝 certiorari 不表示认可下级法院理由，也不是最高法院对 human authorship 作出的实体判决。人类辅助到哪一条线，没有被这件案子回答。另有 *Allen v. Perlmutter*：申请人主张用大量 prompt 迭代生成图像，版权局仍拒登，争的就是人的控制到什么程度才够。至 2026 年秋仍在地区法院阶段。

### 登记政策

Copyright Office *Part 2: Copyrightability*（2025-01-29）：不建议修改现行作者资格规则。纯生成材料不能登记。人类对输出的可感知选择、编排、修改可以保护。Prompt 通常不足以使提示者成为输出表达的作者，因为提示者并不控制具体表达细节；这不等于提示词文本本身必然没有版权。反复修改 prompt、选择输出、修图、编排成作品，可能保护的是人实际贡献的那一层。登记时应披露并排除不主张保护的 AI-generated material。Part 1（2024-07）讨论数字副本和深度伪造，建议单独立法，而不是纳入版权法。

<br>

## 发明人

### 构思标准

专利里的 inventor 不是提出课题、出钱或管理项目的人，而是对至少一项 claim 的 conception 作出贡献的自然人。Reduction to practice（把方案做出来）可以是证据，但只按指令执行实验的人不一定是发明人。反过来，没写代码的人如果形成了 claim 中完整、可实施的技术构思，也可能是发明人。

### AI 辅助

*Thaler v. Vidal*, 43 F.4th 1207 (Fed. Cir. 2022)。专利发明人必须是 natural person。AIA 之后 § 100(f) 把 inventor 定义成 individual(s)，宣誓用 himself / herself。DABUS 不能列名。Federal Circuit 明确没回答：人类借助 AI 完成的发明能不能专利。

USPTO 2024 年 2 月 13 日曾用 *Pannu* 共同发明标准去筛人类有没有 significant contribution。2025 年 11 月 26 日整份指导废止，改发修订稿。Pannu 只适用于多个自然人之间。AI 不是人，谈不上共同发明人。单人借助 AI 时，口径回到传统 conception：发明人要对完整且可实施的发明形成确定、持久的观念。AI 被写成实验设备、软件、数据库。可以用别人的服务和想法，那些来源不因此成为发明人。列发明人时不得把模型写上去。多人共同发明时，Pannu 仍用来筛人与人之间的贡献，不用来筛人和模型。

### 版权对照

版权问这件作品有没有人的表达。专利问这件 claim 有没有人完成 conception。看起来对称，判断对象并不一样。一个人用模型扫几千个候选、自己选出并验证其中一种，和一个人按一次生成键得到一张图，评价层次不同。实验室用代码助手写实现，conception 仍要落到具体人。漏列在团队工具普及之后更容易发生。

<br>

## 比较法

### 比较维度

美国主线到此为止。其他法域解决的是同一批技术事实，但工具不同：欧盟用 TDM 例外、权利保留和 GPAI 透明度；中国以合法来源、内容责任和标识为主；日本、新加坡另设较明确的数据分析例外。

比较法不能只比“允许/不允许训练”，至少要同时比较：

- **例外结构**：开放式 fair use，还是封闭列举的法定例外。
- **访问前提**：是否要求 lawful access。
- **权利保留**：权利人能否 opt out，以及如何表达保留。
- **用途区分**：是否区分科研与商业训练。
- **透明度**：是否要求训练内容摘要或其他文档。
- **输出责任**：模型或平台对违法、有害、侵权输出承担什么责任。

一个法域的例外也不能直接当作另一个法域的抗辩。

Lawful access（合法访问）通常表示使用者有权取得材料，但具体范围依各法域而变：是否已支付订阅费、是否越过登录或技术措施、访问条款怎样限制使用、robots.txt 是否构成有效保留，都可能影响结论。它不是“网页能打开就一定合法”，也不等于“违反任何网站条款就一定不合法”。

### 欧盟

欧盟不走事后 fair use。走的是例外，加上退出。

TDM 是 text and data mining：用自动化技术分析文本和数据以发现模式、趋势和相关性。DSM 第 3 条主要给研究组织和文化遗产机构，权利人不能一般性 opt out；第 4 条适用范围更广，包括商业主体，但前提是合法访问，权利人还可以适当保留权利。两条的主体、目的和退出机制不同。

DSM Directive (EU) 2019/790。第 3 条：科研机构 TDM。第 4 条：更广的文本与数据挖掘，包括商业，但权利人可按第 4(3) 条以适当方式保留权利，实践中常被理解成机器可读 opt-out。robots.txt、网站条款、专门协议，哪些算“适当”，仍在磨。欧盟把选择权放在权利人事前，美国把判断权放在法官事后。

AI Act（Regulation 2024/1689）把版权义务接到通用模型上。第 53 条对 GPAI 提供者：技术文档；向下游系统提供者披露必要信息；建立遵守欧盟版权法的政策，包括用先进技术识别并尊重第 4(3) 条保留；按 AI Office 模板公布足够详细的训练内容摘要。开源 GPAI 在文档义务上有条件豁免，系统风险模型不行。罚则可到全球年营业额 3% 或 1500 万欧元，以高者计。

GPAI 义务 2025-08-02 开始适用。AI Office 对 GPAI 的执法权 2026-08-02。2025 年 7 月的训练内容摘要模板不要求倾倒数据集，要求覆盖预训练到后训练的主要数据、抓取来源、显著域名。2025 年 8 月之前投放市场的模型，摘要可延到 2027-08-02。行为准则是证明合规的路径，不是改写实体法。2025–2026 年间还有简化高风险期限的政治讨论，附件 III / 附件 I 的日期被拉动过，引用时需核对最新文本。

同一套权重，在欧盟要处理 opt-out 和摘要，在美国要准备 fair use 诉讼记录。两边的合规动作不能互相翻译。

<br>

### 中国

生成式服务的监管是内容治理驱动。知识产权是其中一条，不是唯一主轴。先看清适用范围。

三份规则也不在同一层面。《深度合成规定》先管深度合成服务和标识；《生成式人工智能服务管理暂行办法》管向境内公众提供生成式服务；《标识办法》再把生成、传播、应用分发和用户声明串成标识链。它们是行政监管义务，不直接替代法院按著作权法判断作品、复制、实质性相似和合理使用。

《互联网信息服务深度合成管理规定》，2023-01-10 施行。深度合成覆盖文本、图像、音频、视频、虚拟场景。第 16 条隐式标识。第 17 条对可能导致混淆误认的服务做显著显式标识。禁止删除、篡改、隐匿。

《生成式人工智能服务管理暂行办法》，2023-08-15 施行。向境内公众提供生成文本、图片、音频、视频。未向境内公众提供的研发和内部应用，不适用。第七条训练数据：合法来源；不得侵害知识产权；个人信息需同意或其他法定情形；提高数据质量。提供者对输出承担网络信息内容生产者责任，对个人信息承担处理者责任。总则第四条是义务清单：价值观、反歧视、知识产权和商业秘密、人格权、透明度。不是抗辩清单。

《人工智能生成合成内容标识办法》，2025-09-01 施行。把显式标识和文件元数据隐式标识（属性、服务提供者、内容编号）写成可执行规则。同日施行的强制性国家标准 GB 45438-2025《网络安全技术 人工智能生成合成内容标识方法》补技术细节。传播平台有核验和提示义务。标识不是版权登记。诉讼里会和“这是不是作品、谁是作者、平台尽没尽到注意义务”缠在一起。去掉标识再传播，李某案那种事实，以后会更容易被写成恶意。

中国没有 TDM 例外，也没有美国式一般 fair use。著作权法合理使用是封闭列举。《暂行办法》第七条把训练阶段的版权风险写成监管义务，但何种训练构成侵害，仍要回到复制、信息网络传播、合理使用，外加正在长出来的判决。立法和司法目前不同步。监管可以先要求“合法来源”，法院还在逐案问合法指什么。

<br>

中国作者资格的早期样本是北京互联网法院 2023 年李某诉刘某（(2023)京0491民初11279号）。李某用 Stable Diffusion，反复改提示词和参数，生成图片并署名发布；刘某去除署名后使用。法院认定其中有人的独创性智力投入，构成美术作品，赔偿 500 元。美国 Copyright Office 通常认为 prompt 本身控制不足；北京互联网法院则把提示词和参数迭代写进独创性。保护的仍是人的安排与选择，不是模型的人格。

杭州、广州等地的奥特曼生成图从输出侵权切入。输出与受保护角色实质性相似时，生成服务提供者可能承担直接或帮助侵权；通知、过滤、标识、是否把角色生成当卖点，会影响注意义务。国内训练数据案件仍在形成，尚无可稳定对标 *Bartz* 的上级法院规则。

### 其他

日本和新加坡采用的是专门数据分析例外，不是美国式开放 fair use。专门例外的好处是训练者较容易预判，代价是每个法定前提都要满足；lawful access、非享受目的、是否损害权利人利益，会成为决定性词语。

日本著作权法 30 条之 4（2018 年改正，2019 年施行）。不以享受作品中表达的思想感情为目的的利用，必要范围内可以，包括数据分析。不当损害权利人利益则不适用。没有美国式一般 fair use，这条是专门例外。文化厅 2024 年整理：生成式 AI 训练原则上可落入非享受目的；过拟合到能复现表达、把作品当检索结果展示、绕过 robots.txt 或抓有偿分析用数据库，可能构成不当损害。日本制定法给训练开的口子比美国成文法宽，但比 Alsup 在 *Bartz* 里的表述窄：复现表达、绕过技术措施，明确不在例外里。

新加坡 Copyright Act 2021 第 243–244 条 computational data analysis，明文覆盖机器学习，含商业训练，前提是合法访问。第 187 条原则上使试图排除该例外的合同条款无效。2026 年 MinLaw / IPOS 的咨询进一步讨论访问范围条款、技术措施和 robots.txt 应怎样影响 lawful access，而不是重新决定合同能否整体排除该例外。合法访问会成为付费墙、API 条款和 scraping 的主要争点。

英国 2024–2025 年讨论过带 opt-out 的透明 TDM 例外，后来相关报告回到维持现状。商业训练在英国仍更接近要许可。英国、美国、新加坡对商业训练的态度并不一致。

<br>

# 智能体

本章以一条 Agent 轨迹为单位组织责任：

- **行为链**：把预训练、推理时读取和工具执行分开。
- **分层责任**：依次分析感知、规划、行动和输出。
- **控制证据**：查明授权、实际访问、越界和结果去向。
- **工具边界**：RAG、代码助手、网页抓取分别涉及什么既有规则。
- **平台归责**：Agent 代表用户发帖、交易或提供专业服务时，行为归于谁。

## 行为链

Agent 把一次生成扩成带工具的连续行动：浏览、点击、填表、调 API、读写邮箱、提交 PR。版权、合同、CFAA、隐私可能在同一条轨迹上同时出现。预训练使用 Common Crawl 是一件事。推理时对当前网页做即时复制，是另一件。

责任分析要按动作日志而不是按“agent 很自主”来做。模型提出计划、用户授予权限、工具执行 HTTP 请求、平台接收提交，可能分别由不同主体控制。Autonomy 会影响可预见性和控制能力，但不会自动产生一种新的“智能体责任”。

## 分层责任

- **感知**
  - 读网页、读代码、读邮件。
  - 检查复制件有没有产生，以及是否绕过技术措施。
- **规划**
  - 检查系统提示、记忆和检索库各自的权利归属。
- **行动**
  - 检查 HTTP、点击、git push 等外部动作。
  - 对照 ToS、CFAA 和平台规则判断权限边界。
- **输出**
  - 检查代码、文案、图片和 PR。
  - 分别判断作者、侵权、商业秘密和许可证问题。

## 控制证据

每层至少问四件事：

- **授权**：谁授权了动作，授权范围到哪里。
- **访问**：系统实际访问了什么数据、页面、仓库或账户。
- **边界**：是否越过技术措施、账号权限或合同限制。
- **去向**：产生的副本和对外输出去了哪里，谁能够取得。

对高风险工具，再看最小权限、确认点、日志、撤销和人工复核。技术控制会进入过失、诱导、保密措施和合规义务的证据。

## 工具边界

RAG 把某段 GPL 源码贴进 PR，至少发生了复制；是否构成 GPL 意义上的传播或 conveying，要看 PR 是仅限同一法律主体内部，还是提供给外部协作者、客户或公众。GPL 的对应源代码等义务通常在传播触发后分析，不需要先回答模型权重是不是衍生作品。把客户代码丢进第三方托管模型，可能直接违反 NDA。代码助手参与实现并不自动造成发明人漏列；仍须逐项 claim 判断哪些自然人完成了 conception，并保留提示、输出、选择和实验记录。

robots.txt、付费墙、登录后的页面，在日本 2024 年意见和欧盟 opt-out 实践里，已经被当成权利人有没有保留的信号。在美国它们更多是合同和 CFAA。Copyright 上能不能单独成立，还要看复制的量和用途。*Sega* 的中间复制，和预训练使用整本书，技术事实不同，不宜直接类比。

## 平台归责

平台治理。Agent 以用户名义发帖、下单、注册。§ 230、消费者保护、广告法都需要判断表达或交易应归属于谁。技术上用户授予权限，不一定等于每一项输出都由用户发表。证券、医疗、法律咨询这些已有职业规范的领域，agent 输出会先受行业法约束，再另行分析版权。

<br>

# 未决问题

本章不是列出结论，而是保存尚待上诉法院、最高法院或立法机关回答的争点：

- **训练用途**：用途上的 transform 与大规模中间复制如何协调。
- **盗版来源**：来源违法应独立分析，还是影响训练的 fair use。
- **许可市场**：正在形成的训练和角色许可市场如何进入第四因素。
- **输出边界**：记忆、角色、逐页替代和风格模仿的保护梯度。
- **人类贡献**：AI 辅助创作达到何种程度才有作者。
- **智能体复制**：推理时即时复制能否沿用预训练分析。
- **开放权重**：商业秘密消失后，版权和合同能限制到什么范围。
- **分析顺序**：在政策设计前先确定行为、主体、主张和证据。

以下问题尚无最高法院终局规则，但已有足够判决和诉讼材料形成明确争点。

## 训练用途

训练通用模型，究竟是一种用途上的 transform，还是大规模复制的中间步骤。*Warhol* 要求看具体市场用途。Alsup 把技术过程写成高度变革。Chhabria 把市场稀释留给下一家证据更好的原告。Bibas 说明：直接做功能相同的竞品时，仅主张“这是训练”过不了第一、第四因素。

## 盗版来源

盗版来源应作为独立复制行为分析，还是会进一步影响训练用途的 fair use。两名北加州法官给出了不同处理。15 亿美元和解量化了特定作品清单所涉的既往影子库下载请求，没有为训练行为建立一般规则或许可价格。

## 许可市场

许可市场一旦密集存在，未许可训练还能否声称没有受保护的市场。新闻社、图库、影视角色都在试着做交易。Disney–OpenAI 宣布过、没交割就停了，说明定价意愿在，市场还没稳。

## 输出边界

输出争议不能简单排成统一的线性强弱表。逐字或逐页替代通常最接近复制受保护表达；角色要看是否得到充分刻画以及输出取走了哪些具体元素；抽象风格本身通常不受版权垄断，但风格化输出仍可能复制角色、构图、服装、场景或台词。

## 人类贡献

人类辅助到哪一条线才构成作者，*Thaler* 没有回答。*Allen v. Perlmutter* 还在地区法院。

## 智能体复制

Agent 瞬时复制和预训练复制是否适用同一套 fair use。技术事实不同。

## 开放权重

权重公开后，公开部分不再具有商业秘密的秘密性，但版权和合同是否仍能限制使用。模型许可如果只产生合同义务，对未与发布者形成合同关系的下游下载者约束力有限。

## 分析顺序

- **行为与对象**
  - 先写清复制了什么、存了什么、用户是否看见内容、是否绕过技术措施。
- **主体与主张**
  - 再写作者、实验室、下游、用户、平台和监管机关分别主张什么。
- **规则与方案**
  - 最后才讨论例外、强制许可、透明度、标识、选择退出和责任保险。
- **架构反馈**
  - 每一种方案都会反过来改变模型的过滤、记忆、工具权限和日志设计。

<br>

# 资料

本章按资料用途整理，便于回查：

- **权威层级**：区分具有约束力的法源、说服性判决、行政材料与诉讼事实材料。
- **课程**：三门课程及其页面。
- **法源**：美国法条、欧盟指令和 AI Act。
- **判例进度**：AI 作者、发明人、训练和输出案件的程序状态。
- **比较法材料**：中国、日本和新加坡的制定法与技术标准。
- **阅读建议**：如何把旧课程与 2025–2026 年案件接成一条线。

## 权威层级

资料不能只按主题堆在一起，还要先辨认法律效力：

1. **宪法、制定法与有效行政法规**：提供直接规则，但仍可能需要法院解释。
2. **最高法院判决**：对联邦法问题具有全国约束力。
3. **巡回上诉法院判决**：约束本巡回内的联邦地区法院；其他巡回通常只把它作为说服性材料。
4. **地区法院意见**：原则上不约束其他法官，即使同属一个地区；summary judgment 也须按该案记录理解。
5. **行政机关指导、报告与技术标准**：权重取决于法定授权、程序和具体用途，不等同于判决或制定法。
6. **诉状、专家报告、和解与新闻报道**：可证明当事人提出了什么、案件如何结束或市场发生了什么，不能单独当作法院已经认定的实体规则。
7. **课程与二手资料**：适合建立框架和寻找原始材料，引用结论时仍应回到法条、判决和正式文件。

## 课程

15.628J：[主页](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/) · [Syllabus](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/syllabus/) · [Readings](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/readings/) · [Assignments](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/assignments/)

6.4590：[OCW](https://ocw.mit.edu/courses/6-4590-foundations-of-information-policy-fall-2024/) · [课网](https://internetpolicy.mit.edu/6.4590/) · [About](https://internetpolicy.mit.edu/6.4590/about-the-class/)

Stanford CS202：[ExploreCourses 正式课程描述](https://explorecourses.stanford.edu/search?q=CS+202) · [Fall 2006 旧课程归档](http://web.stanford.edu/class/cs202/)。五件现代案例来自前者，不来自 2006 归档。

## 法源

17 U.S.C. §§ 101, 102, 106, 107, 109, 117, 201, 204, 412, 504, 512, 1201, 1202。35 U.S.C. §§ 101–103, 112, 154, 256, 262, 271, 284–286。18 U.S.C. § 1836。15 U.S.C. §§ 1114, 1125。47 U.S.C. § 230。DSM 2019/790 arts. 3–4。Regulation (EU) 2024/1689 art. 53。

## 判例进度

USPTO inventorship guidance：2024-02-13 发，2025-11-26 废，改回 conception。Copyright Office *Copyright and Artificial Intelligence* Part 1 Digital Replicas（2024-07）；Part 2 Copyrightability（2025-01-29）。

*Authors Guild v. Google*, 804 F.3d 202 (2d Cir. 2015)。*Bartz* summary judgment 2025-06-23（Alsup）；终局批准 2026-07-20（Martínez-Olguín，N.D. Cal. 4:24-cv-05417）。*Kadrey* 拒绝中间上诉 2026-07-08。*Ross* 第三巡回 No. 25-2153，2026-06-11 辩论，至 2026-09 未下判。*Thaler v. Perlmutter* cert. denied 2026-03-02。Disney–OpenAI Sora 许可 2025-12 宣布、2026-03 随 Sora 应用关停取消。

## 比较法材料

《生成式人工智能服务管理暂行办法》（2023-08-15）。《互联网信息服务深度合成管理规定》（2023-01-10）。《人工智能生成合成内容标识办法》（2025-09-01）。GB 45438-2025。日本著作权法 30 条之 4。Singapore Copyright Act 2021 ss. 243–244。

## 阅读建议

15.628J 的案例仍值得按 session 过：先建立四种 IP 和 claim 意识，再读软件专利史，再把 *Alice* 以后和 2025–2026 训练数据判决当成同一条时间线上的续集。6.4590 不必按周重做。架构、§ 230、*Carpenter*、跨境那几周的读物，当政策写作样本足够。
