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
- **问题索引**：从训练、抓取（scraping）、API、职务成果、开放权重、论文公开和 NDA 等具体问题反查制度。
- **判例方法**：区分法源、裁判规则、程序阶段、诉因、救济和证明标准。
- **法院体系**：联邦制（federalism）、州与联邦两套法院、Federal Circuit 的事项管辖（subject-matter jurisdiction），以及选法院本身就是策略。
- **三分钟入门**：没有法律基础，先读这一节和文末的术语表。

## 三分钟入门

> [!TIP]+ 一句话版本
>
> 知识产权（IP）是法律人为造出来的“排他权”：在一定范围、一定期限内，允许你禁止别人做某些事。它不保护“想法”本身，只保护法律认可的几种成果。

### 四种权利，各用一个例子

| | 专利 Patent | 版权 Copyright | 商业秘密 Trade secret | 商标 Trademark |
| :--- | :--- | :--- | :--- | :--- |
| 生活里的例子 | iPhone 的滑动解锁 | 一本小说、一段代码、一张照片 | 可口可乐配方、未公开的模型权重 | “Google” 这个名字和 logo |
| 保护什么 | 一个有用的**技术方案** | 一段**具体的表达**（不是背后的想法） | 一条**因为保密才值钱**的信息 | 让顾客**认出是谁家的东西** |
| 怎么得到 | 向政府申请、审查通过 | 写下来、拍下来就自动有 | 自己保密就有 | 在生意中使用；注册更有利 |
| 能挡住谁 | 任何人，哪怕对方独立想出来 | 只挡**抄袭**的人 | 只挡**偷、骗、违约（breach of contract）**拿到的人 | 让顾客**认错来源**的人 |
| 多久 | 约 20 年 | 作者终身 + 70 年 | 只要一直没泄露 | 只要一直在用 |
| 代价 | 必须公开技术细节 | 几乎没有 | 一旦泄露就没了 | 名字变成通用词就没了 |

### 读这篇笔记前要知道的六件事

1. **同一个产品常常四种权利叠在一起。** 一个 AI 产品：算法改进可以申请专利，代码有版权，没公开的训练数据（training data）是商业秘密，产品名是商标。
2. **想法本身谁都不保护。** “做一个能预测比分的 app”这个点子，不受任何一种权利保护；只有做成具体代码、具体技术方案或签了合同，才有保护。
3. **“有权利”和“谁拥有”是两道题。** 你在公司写的代码有版权，但版权可能属于公司。
4. **美国法很多规则来自法院判决，不只是法条。** 所以笔记里大量是“某某诉某某”的案例。斜体的 *A v. B* 就是案名，括号里是法院和年份。
5. **不只亲手做的人要负责。** 自己复制、制造是 direct infringement（直接侵权）；帮助、诱导别人侵权，也可能承担 secondary liability（次级责任，又叫间接侵权）。
6. **AI 没有专门的法律。** 训练、生成、自动浏览网页，都是把上面这些老规则套到新行为上，所以结论常常还没定。

### 怎么读

- **只想建立概念**：读三分钟入门 → 激励 → 体系 → 每章开头的“核心问题”和加粗句子。
- **遇到具体问题**：查下面的“问题索引”，直接跳到对应章节。
- **看不懂某个词**：查文末“术语表”。
- **想看真实案子**：看各章里标为“例子”的方框，每个方框第一行先写它说明了什么。

## 材料范围

本文以美国法为主，三门课各管一条线：

- **MIT 15.628J / 6.903J** *Patents, Copyrights, and the Law of Intellectual Property*（Spring 2013）：知识产权的基本规则（doctrine）。
- **MIT 6.4590** *Foundations of Information Policy*（Fall 2024）：言论、平台、隐私、跨境等信息政策框架。
- **Stanford CS202** *Law for Computer Science Professionals*：用技术公司的真实争议，把软件专利、API、商业秘密、CFAA 和雇佣合同串起来。
- **课程之后的补充**：*Alice*、*Google v. Oracle*、*Warhol*、*Authors Guild v. Google*，以及 2025–2026 年的训练数据诉讼。

- [15.628J](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/)
- [6.4590](https://ocw.mit.edu/courses/6-4590-foundations-of-information-policy-fall-2024/)
- [Stanford CS202（ExploreCourses）](https://explorecourses.stanford.edu/search?q=CS+202)
- **CS202 Fall 2026 课件（David W. Hansen）**：美国法院体系、思想投递、专利 §§ 101–103 / 112 与 claim construction、版权入门，外加 DMCA、CFAA 与 AI 的 2025–2026 新案。文中 Blue Origin 海上回收、Apple 滑动解锁、Neonode、Tangle、Myst 字母等例子多出自这批材料。

<br>

## 阅读路线

> [!INFO]+ 章节
>
> 章节按问题顺序排，不按课表。
>
> 1. **为什么保护**：激励、体系——排他权的目的，四种权利各管什么。
> 2. **四种私权本体**：版权 → 专利 → 商业秘密 → 商标。计算机里最常碰到的放前面。
>    - 版权按“成立 → 独创 → 思想边界 → 软件 → 接口 → 合理使用 → 间接责任”拆。
>    - 专利按“授权 → 适格 → 实用 → 新颖 → 显而易见 → 权利要求 → 侵权 → 救济”拆。
> 3. **权利归谁**：归属——职务发明、跳槽、开源（open source）。
> 4. **公法与监管**：信息政策——言论、平台、搜查、隐私、跨境。
> 5. **套用到新技术**：人工智能、智能体——训练、输出、作者、发明人（inventor），以及带工具的连续行动。
> 6. **还没定的与出处**：未决问题、资料（含术语表）。

<br>

## 问题索引

> [!NOTE]+ CS / AI 问题映射
>
> - **GitHub 代码能否用于训练** → copyright、许可证（license）、fair use。
> - **Agent 自动爬网站** → CFAA、合同、版权、隐私。
> - **Agent 代用户在电商网站操作、下单** → *Amazon v. Perplexity*：CFAA 意义上谁在 access，服务条款能否约束 agent。
> - **模型能生成侵权内容，开发者要不要负责** → 次级侵权（secondary liability）：*Grokster*、*Cox v. Sony*——诱导，或专为侵权设计；三种权利的对照见体系一章。
> - **用户按专利方法使用我的 SDK** → 专利诱导侵权（induced infringement） § 271(b)、*Global-Tech*、*Akamai*。
> - **API 实现与 API 本身的区别** → § 102(b)、*Lotus*、*Google v. Oracle*。
> - **实习期间写出的 agent framework 归谁** → assignment、work made for hire、*Stanford v. Roche*。
> - **Open-weight model 能否商用** → 模型许可证、权重和代码的权利分层。
> - **研究代码能否公开到 GitHub** → 雇佣合同、商业秘密、专利 novelty。
> - **公开发布一个图像数据集（dataset）** → 复制、发行、展示；*EVOX v. Stanford*。
> - **抓取时绕过平台的反爬、加密措施** → DMCA § 1201，不以侵权为前提。
> - **给大公司发邮件推销产品点子** → *Desny* 默示合同；先签协议再披露。
> - **先发论文、先卖 beta，再申请专利** → § 102(b)(1) 宽限期（grace period）与 on-sale bar。
> - **给模型或产品起名** → 描述性商标、优先权；CHATGPT、OPENAI、io。
> - **NDA 与论文发表冲突** → 保密义务、发明转让、学校与公司之间的 publication review。

<br>

## 判例方法

> [!INFO]+ 判例怎么读
>
> **法从哪里来**
>
> - **Statute（成文法）**：国会或州议会通过的文字，例如 17 U.S.C. § 107。
> - **Regulation（行政法规）**：行政机关依据授权制定。
> - **Case law（判例法）**：法院在具体案件里解释、适用规则形成的法律。
>
> **判决里哪句话算数**
>
> - **Holding（裁判规则）**：解决本案争点所必需的结论，有约束力。
> - **Dicta（附带意见）**：不影响本案结果的讨论，可能有说服力，约束力弱。
> - **Precedent（先例）**的范围看法院层级：最高法院约束全国的联邦法问题；巡回上诉法院（court of appeals）约束本巡回；地区法院意见原则上不约束其他法官。
>
> **案子走到哪一步**（procedural posture 改变一句话的分量）
>
> - **驳回动议**：只问诉状（complaint）写的事实如果为真，能不能成立。
> - **Summary judgment（简易判决）**：关键事实没有真正争议时，法官直接依法裁判。
> - **Preliminary injunction（初步禁令）**：诉讼期间的临时救济，不是最终输赢。
> - **Settlement（和解）**：通常不等于法院认定谁对谁错。
>
> **原告（plaintiff）要证明什么**
>
> - **Claim（诉因）**：原告请求救济的法律主张。注意它不等于专利里的 claim（权利要求）。
> - **Elements（构成要件）**：每个诉因由几个要件组成，原告通常要逐一证明。
> - **Affirmative defense（积极抗辩）**：被告（defendant）不否认事实，但主张有法定理由免责，例如 fair use。
> - **Remedy（救济）**：责任成立后法院给什么——damages（赔偿）、injunction（禁令）、declaratory judgment（确认判决）。

<br>

> [!NOTE]+ 引用怎么看
>
> | 写法 | 意思 |
> | :--- | :--- |
> | `17 U.S.C. § 107` | 《美国法典》第 17 编第 107 节；`§§` 表示多个条文 |
> | U.S. Code 的“编”（Title） | 国会立法的官方汇编，共 53 编。本文常见：Title 15 商业与贸易（含商标）、17 版权、18 刑法（含 CFAA、商业秘密犯罪）、35 专利；另如 Title 11 破产、52 选举 |
> | `593 U.S. 1 (2021)` | 判决收在 United States Reports 第 593 卷第 1 页，2021 年（最高法院） |
> | `F.3d` / `F.4th` | 巡回上诉法院判决汇编 |
> | `9th Cir.` / `Fed. Cir.` | 第九巡回上诉法院 / 联邦巡回上诉法院（管专利） |
> | `N.D. Cal.` / `D. Del.` / `S.D.N.Y.` | 加州北区 / 特拉华 / 纽约南区联邦地区法院 |
> | `cert. denied` | 最高法院拒绝受理；**不**表示赞成下级法院理由 |
>
> 联邦案件的路径：district court（地区法院）→ circuit court（巡回上诉法院）→ Supreme Court（最高法院，自己选择受理哪些案件）。

<br>

> [!NOTE]+ 证明到什么程度
>
> 从低到高三档：
>
> 1. **Preponderance of the evidence（优势证据）**：某事实“更可能为真”，大约过半即可。普通民事案件用这一档；PTAB 的 IPR 也用。
> 2. **Clear and convincing evidence（清楚且有说服力）**：在法院主张已授权专利无效，要达到这一档。
> 3. **Beyond a reasonable doubt（排除合理怀疑）**：刑事案件，例如刑事商业秘密盗窃。
>
> 同一组事实进入民事案和刑事案，当事人、构成要件和证明标准都可能不同。

<br>

## 法院体系

### 联邦制

- **列举权力**：国会只有宪法列举的权力，加上 necessary and proper clause；其余权力保留给州。
- **与 IP 相关的三条**：
  - Art. I, § 8, cl. 8 知识产权条款 → 专利、版权。
  - Art. I, § 8, cl. 3 商业条款（Commerce Clause） → 联邦商标法。
  - Art. VI Supremacy Clause → 联邦法优先；1790 年专利法之后，授予专利从各州收归联邦。
- **结果**：专利、版权由联邦法统一；商业秘密、想法投递、合同、肖像权（right of publicity）等大量仍是州法。同一纠纷常同时出现联邦和州的诉因。

### 两套法院

- **规模**：州和地方法院每年受理 6000 万件以上（交通、家事、人身伤害、合同、普通刑事）；联邦地区法院 2025 年约 35 万件，约 80% 民事。
- **联邦层级**：district court → 13 个上诉法院（11 个按地域编号的巡回 + D.C. Cir. + Federal Circuit）→ 最高法院（petition for writ of certiorari，裁量受理）。
- **加州 4 个联邦地区**：N.D. Cal.（SF / Oakland / San Jose 三个办公地，硅谷案件主场）、E.D.、C.D.（LA，娱乐业案件多）、S.D.；都上诉到第九巡回。
- **Federal Circuit**：按事项而非地域管辖，全国专利上诉、PTAB 和 ITC 的上诉都归它。

> [!NOTE]+ 管辖怎样影响结果
>
> - *Oracle v. Google* 含专利请求，整案上诉去了 Federal Circuit；但版权问题仍须适用第九巡回法。Federal Circuit 两次推翻 Alsup，实际上是在“预测”第九巡回会怎么判。
> - Neonode 2020 年在 W.D. Tex. 起诉 Apple、Samsung；Federal Circuit 后来命令把 Apple 案移到 N.D. Cal.，理由是案件的 “center of gravity” 在北加州。原告挑法院、被告争移送，本身就是诉讼策略。

<br>

> [!NOTE]+ 为什么案子都在北加州和特拉华
>
> - **两个概念**：jurisdiction（管辖权）问这个法院能不能审这个被告、这类案件；venue（审判地）问在有管辖权的联邦法院里具体放在哪个地区。
> - **版权案**：依 28 U.S.C. § 1400(a)，可在被告“居住或可被找到”的地区起诉。Anthropic、Meta、OpenAI 总部都在湾区，*Bartz*、*Kadrey* 因此都在 N.D. Cal.。
> - **专利案更窄**：§ 1400(b) 加 *TC Heartland v. Kraft Foods*, 581 U.S. 258 (2017)：公司只在注册州“居住”，否则要去它有常设营业地并实施侵权的地区。大量公司在 Delaware 注册，D. Del. 因此是专利诉讼重镇，*Ross* 也在那里。
> - **不能随便挑**：一家只在 Montana 经营、和加州没有联系的出版社，不能被拖到旧金山。属人管辖（personal jurisdiction）要求被告与法院地有 minimum contacts（属于民事诉讼法，Civil Procedure）。
> - **不一定上陪审团（jury）**：关键事实没有争议时，法官用 summary judgment 依法裁判；*Bartz* 的 fair use 结论、*Kadrey* 的训练结论都是这样来的。

<br>

# 激励

> **核心问题**：国家为什么要让人“垄断”一个想法或作品？换来了什么，又付出了什么？

本章按“为什么授权—权利边界在哪里—制度以什么换什么”的顺序展开：

- **宪法基础**：知识产权（intellectual property）条款授权什么，其目的和范围是什么。
- **排他边界**：排他权能禁止哪些行为，又保留哪些公共使用空间。
- **制度交换**：专利、版权和商业秘密分别以披露、期限或保密换取什么。
- **资产价值**：无形资产怎样成为企业和国家的主要资产。
- **制度史**：1790 年两部法律、Jefferson 的转向与 Lincoln 的专利。

## 宪法基础

宪法知识产权条款的范围很窄：

*To promote the Progress of Science and useful Arts, by securing for limited Times to Authors and Inventors the exclusive Right to their respective Writings and Discoveries.*（U.S. Const. art. I, § 8, cl. 8）

- **给谁**：Authors 和 Inventors；对象分别是 Writings 和 Discoveries。
- **给什么**：exclusive Right（排他权），不是对作者身份本身的保护。
- **给多久**：limited Times，必须有期限。
- **为什么给**：to promote the Progress of Science and useful Arts。目的是促进进步，不是“劳动应得”或“自然权利”——这类说法条文里没有。
- **不在这一条里的两种**：
  - **商标**：联邦立法权来自 Commerce Clause（商业条款）。Lanham Act 延续反不正当竞争传统，核心是消费者会不会认错来源。
  - **商业秘密**：更接近侵权法（tort）和合同，长期以州法为主；联邦 1996 年 Economic Espionage Act 规定刑事犯罪，2016 年 Defend Trade Secrets Act 增加联邦民事诉权。

## 排他边界

- **排他权不是绝对控制**：只允许权利人在法定范围内**排除特定行为**。
  - 版权人能禁止复制受保护的表达，不能禁止别人独立想到同一个事实或思想。
  - 专利权人能排除他人实施 claim 覆盖的发明，即使对方是独立研发的。
  - 范围、期限、例外、救济都由法律限定。
- **也不完全是 monopoly（垄断）**：
  - 专利只给“排除别人”的权利，不保证自己能生产：产品可能落入别人更宽的专利，或受药品监管、出口管制。
  - 版权不垄断作品主题，只保护可识别的表达。
- **真正的难题是给后来者留多少空间**：
  - 专利 claim 写到抽象功能层，会把整条技术路线圈进去，后来者自己做出实现也要避开或交许可（license）。
  - 版权若把程序的功能、操作方式也当成表达，兼容实现和反向工程的空间就被挤掉；*Lotus*、*Sega* 拆的就是这件事。
  - 商业秘密不强制公开，社会拿不到“说明书（specification）”那一层交换。
- **结论**：四种制度都在“激励创作”和“留下后续使用空间”之间取舍，只是方式不同。

## 制度交换

- **问题**：Arrow 的 information paradox（信息悖论）——买方要先知道发明是什么才肯付钱；一旦知道了，往往就不必再买。没有排他或保密，没人愿意事先投钱研发。
- **反方向的风险**：排他范围过宽，本来复制成本接近零的东西，会被法律重新变贵。
- **三种交换方式**：
  - **专利**：向公众公开技术，换约 20 年的独占。
  - **版权**：只保护表达，给复制等禁止权；到期进入 public domain（公共领域）——期限届满、权利不成立或依法不受保护的东西，任何人都可以不经许可使用。
  - **商业秘密**：不公开、没有固定期限，但挡不住别人独立发现或合法反向工程（reverse engineering）。
- **结论**：这是三套不同的交易，不能按“谁更强”排序。

## 资产价值

- **定义**：WIPO 把 IP 称为 “creations of the mind”：发明、文学艺术作品、外观设计，以及商业中使用的符号、名称和图像。
- **为何要立法**：Jefferson（1813）说发明 “cannot, in nature, be a subject of property”。土地（real property）和动产（tangible personal property）天然可占有；想法不行，排他权只能靠法律造出来。
- **价值倒置**（Ocean Tomo）：1975 年 S&P 500 市值中 83% 是厂房设备等有形资产；2025 年约 92% 是无形资产。
- **宏观规模**（USPTO 局长 John Squires，2026-09-15）：
  - 2024 年 IP 密集型产业约 11.4 万亿美元，占美国 GDP 44%。
  - 直接雇用约 5000 万人，占劳动力 33%。
  - 人均年产出约 23 万美元，比非 IP 密集产业高 60%。
- **企业怎么写**：IPO 招股书的标准句式是同时依靠 patents、trademarks、trade secrets、copyrights、保密程序和合同。实际保护是组合，不是单一权利；四大类之外还有域名等其他权利。
- **不审质量**：PTO、Copyright Office 和法院只看法定要件，不评价商业或艺术质量。大量授权专利从未实施，称为 paper patent。

## 制度史

- **Washington 本人**：自认首先是农民，出于需要做过不少农具发明但没申请专利，例如便于脱粒的十五边形谷仓、把种子经管道送到犁沟的 “drill plow”。
- **1790 专利法**：
  - Washington 在 1790-01-08 首次国情咨文中请国会鼓励 “the introduction of new and useful inventions”；国会 1 月 22 日起草，4 月 10 日签署。
  - 此前各州（原殖民地）各自授予专利；此后依 Supremacy Clause 成为联邦专属事务。
  - 审查 “merit” 与 “originality”，期限 14 年，每份专利须总统签名。
  - 由国务卿、战争部长、司法部长共同授权，国务卿 Jefferson 主导，相当于首任专利审查官。
- **Jefferson 的转向**：原本视一切垄断为 “odious”，反对 IP 条款（茶叶垄断引发了独立战争）；1790 年后称专利法 “given a spring to invention beyond my conception”。他发明了转椅（据说在上面起草了《独立宣言》）、加密用的 wheel cipher，还画过通心粉压面机的图纸，却从未申请专利，Franklin、Washington 也一样。
- **1790 版权法**：1790-05-31 签署，只保护 maps、charts、books；全文 2 页，今天的版权法印出来 486 页。当时的 “Progress of Science” 指知识的创造与传播。
- **版权的目的**：*Aiken*（1975）说直接效果是保障作者劳动的 fair return，终极目标是借经济激励促进公共利益；*Mazer*（1954）说个人收益的激励是推进公共福利的最好方式。
- **Lincoln**：
  - 唯一持有专利的总统：No. 6,469（1849），“Buoying Vessels Over Shoals”，用可充气浮囊帮船过浅滩。他在律所里亲手削了模型，合伙人 Herndon 觉得不切实际但没说出口。Lincoln 从未商业化，是典型的 paper patent（这种情况并不少见）。
  - 做过专利诉讼律师，把专利法列为人类史上三大进步之一（另两个是发现美洲和印刷术）。
  - USPTO 入口刻着他的话：“The patent system added the fuel of interest to the fire of genius.”
  - 1863 年签署法案，设立 National Academy of Sciences。
- ***McCormick v. Manny***（1850s，收割机专利案）：
  - **背景**：McCormick 发明机械收割机，1834 年获专利，让农民收成翻倍以上。Manny 1853 年做出新型收割机，1855 年在巴黎博览会击败 McCormick 拿了头奖。
  - **起诉**：McCormick 的 1834 年基础专利按当时规则（授权后 17 年）已到期，只能拿改进专利起诉 Manny，要求停产并赔偿 40 万美元（约合今天 1500 万美元）。
  - **结果**：Manny 一审和最高法院（Supreme Court）都胜诉。
  - **Lincoln 与 Stanton**：Manny 请全国知名的 Edwin Stanton 主辩，Lincoln 只是当地律师，被 Stanton 冷落、不许参与庭审。Lincoln 认为自己 “contributed nothing”，退回了酬金支票（对方再寄来，他才收下）。五年后当选总统，他仍任命 Stanton 为战争部长；Stanton 成为内战中最重要的顾问之一，Lincoln 去世时守在床边，说 “Now he belongs to the ages.”
  - **启示**：基础专利到期后靠改进专利延续控制，这一策略今天仍常见；但改进专利只覆盖改进那一层，基础技术已进入公共领域。

<br>

# 体系

> **核心问题**：专利、版权、商业秘密、商标各管什么？同一个产品怎样同时受几种保护？

本章把四种制度放在同一张图里，但不把它们当作同一种保护：

- **四权分工**：先区分各制度的保护对象、成立门槛、可对抗主体和期限。
- **想法投递**：四权都不保护的“想法本身”，只能靠合同。
- **次级侵权对照**：帮助、诱导别人侵权，在三种权利里分别要件是什么。
- **叠加保护**：再看同一个模型产品的技术、代码、数据和品牌如何分别落入不同制度。
- **资产策略**：最后讨论公开论文、开源（open source）代码、发布权重和申请专利对权利状态的影响。

## 四权分工

四种制度回答的不是同一个问题：

1. **Patent**：某项技术方案是否满足授权条件；授权后，权利人可在美国法定范围内排除他人制造、使用、销售、许诺销售或进口落入 claim 的发明。
2. **Copyright**：是否存在人的原创表达，被告（defendant）是否复制了受保护部分。
3. **Trade secret**：信息是否仍属秘密，被告取得或使用的手段是否不正当。
4. **Trademark**：某个标志是否识别商品来源，被告的使用是否导致消费者混淆。

同一个产品可以同时受四种制度保护，彼此不互相替代：

- 模型推理系统的具体技术改进 → 可能申请专利；
- 源代码 → 版权；
- 未公开的训练配方 → 商业秘密；
- 产品名和 logo → 商标。

| | Patent | Copyright | Trade Secret | Trademark |
| :--- | :--- | :--- | :--- | :--- |
| 管什么 | 有用的技术方案 | original expression | 因保密而值钱的信息 | 来源识别 |
| 门槛 | useful / novel / non-obvious / eligible / 充分披露 | 独立创作 + 一点点 creativity，固定在有形介质 | 秘密 + 合理保密措施 + 经济价值 | 商业使用、能识别来源、还没变成通用名称（generic name） |
| 能挡谁 | 独立发明人（inventor）也挡 | 只挡 copying | 只挡不正当获取和使用 | 挡混淆、有时挡淡化（dilution） |
| 多久 | 通常从最早美国非临时申请日起算 20 年；另有 PTA / PTE | 自然人 life+70；雇佣、匿名、化名作品为发表后 95 年或创作后 120 年，取较早者 | 满足法定条件且仍属秘密期间 | 持续使用并按期续展 |
| 换来什么 | 向公众披露 | 表达最终进公共领域（public domain） | 无强制披露 | 降低搜寻成本 |
| 软件里常见 | 写得够具体的系统 / 方法 claim | 代码、文档、非功能性（non-functional）结构 | 未开源权重、数据、过滤、系统提示 | 产品名、logo、模型名 |

## 想法投递

- **原则**：“Generally speaking, ideas are as free as the air.”（*Desny v. Wilder*, Cal. 1956）想法本身不是财产：版权排除 idea，专利排除抽象思想（abstract idea），商业秘密要求保密。
- **合同例外**：*Desny* 在加州合同法里开了一个窄口子。
  - **明示合同**：双方事先约定为想法付费。
  - **默示合同**（implied-in-fact）三要件：
    1. 披露以付费为条件；
    2. 对方知道或应当知道这个条件，仍自愿接受披露；
    3. 想法有价值，且被对方实际使用。
  - **先说后谈不算**：披露发生在对方知道“要付费”之前，默示合同不成立。“The idea man who blurts out his idea without having first made his bargain has no one but himself to blame.”
- ***Norris v. NBA Properties*（S.D.N.Y. 2020）**：
  - Norris 给 NBA 总裁发邮件推销预测类 app “Predict a Pick 3”，被转给新技术部门，约好电话前先发了一页说明；电话没打，随后收到拒绝。
  - NBA 不久上线 “Pick'Em”，Norris 以违反默示合同起诉。
  - 法院驳回：Norris 主动披露，也没有指控 NBA 任何人同意受具体条款约束；“at most, Stanchak agreed to listen to his pitch”。纽约同样遵循 *Desny*。
- **公司对策**：Apple、Nest 等公布 unsolicited idea submission policy，声明拒收，或收到即不承担保密、付费义务。目的就是切断默示合同的第 2 要件。

> [!TIP]+ 对 CS 从业者
>
> - 先谈条件，再披露：NDA 或书面投递协议在前，demo 在后。
> - 想法越抽象越难保护。写成代码（版权）、具体技术方案（专利）或保密的实现细节（商业秘密），才落进四权。
> - 这是 Arrow 信息悖论的合同法版本：不说卖不出去，说了就没了。

## 次级侵权对照

**Direct infringement（直接侵权）**是自己实施了被禁止的行为；**secondary liability（次级责任 / 间接侵权）**是因为帮助、控制或诱导**别人**的直接侵权而负责。没有别人的直接侵权，就没有次级责任。

| | 版权 | 专利 | 商标 |
| :--- | :--- | :--- | :--- |
| 法律来源 | 全部来自判例 | 写进条文：35 U.S.C. § 271(b)、(c) | 来自判例（*Inwood*） |
| 帮助侵权（contributory） | 意图让服务被用于侵权；*Cox v. Sony*（2026）收窄为只能用诱导或“专为侵权设计”证明 | 提供专门用于侵权、没有大量合法用途的部件，且知道专利 | 故意诱导，或明知仍继续供货；*Tiffany v. eBay* 要求具体知情 |
| 诱导侵权（inducement） | *MGM v. Grokster*（2005）：以促成侵权为目的推广 | § 271(b)；*Global-Tech*（2011）：知道专利和侵权，故意视而不见也算 | 包含在 *Inwood* 的第一条路径里 |
| 替代责任（vicarious） | 有权有能力控制 + 直接经济利益 | 无对应概念 | 要求合伙或代理关系，很窄 |
| “双用途”保护 | *Sony v. Universal*（1984）：有大量合法用途的产品，厂商不仅因用户可能侵权而负责（借自专利法） | staple article 排除 | 无明确对应 |
| 平台安全港（safe harbor） | DMCA § 512（通知—删除） | 无 | 无；§ 230 也不适用于知识产权（intellectual property） |
| 对 AI 开发者 | 训练时的复制是直接侵权问题；用户生成侵权内容是次级问题 | 用户按专利方法使用模型或 SDK，开发者可能被诉诱导 | 用户生成带他人 logo 的图，看具体知情和继续提供 |

## 叠加保护

“四权分工”那张表比较的是制度；放到具体产品中，还要按资产逐项识别保护对象，不能把产品整体只归入其中一栏。

- **同名双权**：PageRank 既是专利（2019-09-24 到期，技术进入公共领域），也是 Google 的商标（持续使用即可续展）。同一个名字下的两种权利，寿命完全不同。许可经过见后文 Stanford 的 PageRank 段。

## 资产策略

实验室里四种权利常叠在同一块资产上，每个动作都会改变权利状态：

| 资产 | 最接近的权利 | 一个动作会怎样改变它 |
| :--- | :--- | :--- |
| 算法思想 | 专利 | 先发论文，novelty 可能没了；纯算法还可能过不了 § 101 |
| 代码 | 版权（创作即产生） | 开源 = 按许可证放掉其中一部分权利 |
| 权重、数据 | 商业秘密 | 一旦公开，秘密性基本无法再主张 |
| 产品名 | 商标 | 越早使用、注册，越有优先权 |

- **常见组合**：论文换引用和职位；数据继续当秘密；名字注册商标；专利只覆盖真正有技术改进、claim 也写得动的那一层。
- **开放权重（open-weight）**：意味着放弃权重本身的商业秘密保护。Llama community license 既有版权许可，又有额外使用限制；这些限制是许可条件还是合同义务、能否约束下游，要看权重受哪些权利保护、条款性质、合同是否成立，不能直接等同于 GPL 的 copyleft。

> [!NOTE]+ 四种制度不好比强度
>
> 专利能挡独立发明，但要公开、有期限；版权自动产生，只挡抄袭；商业秘密不公开，一旦泄露或被反向工程（reverse engineering）就难再主张；商标可续展很久，但保护的不是创意本身。各有代价，没有“最强”的一种。

<br>

# 版权

> **核心问题**：我写的代码、画的图、拍的照片受什么保护？别人怎样用才算侵权，怎样用不算？

本章从版权是否成立开始，依次处理保护范围、软件与接口、合理使用和平台责任：

- **成立与权利**：保护对象、归属、侵权、抗辩（defense）、救济，以及权利束、登记、期限和发行用尽（exhaustion）。
- **独创性**：独立创作和最低创造性，尤其是事实汇编。
- **保护边界**：思想—表达二分、merger、标准场景、实用物品、字母与语言，以及实质性相似。
- **软件与接口**：程序对象分层、抽象—过滤—比较、兼容与 API。
- **合理使用**：四因素、具体用途、教育、搜索和训练类比。
- **间接责任（次级侵权）**：帮助、控制、诱导、双用途工具和平台安全港。

## 成立与权利

### 五步分析

17 U.S.C. § 102(a)：*original works of authorship fixed in any tangible medium of expression*。

版权问题按五步拆：

1. **保护与权属**：是否存在已固定的 original expression，原告（plaintiff）是否拥有有效版权。
2. **事实复制**：被告（defendant）是接触并复制了原作，还是独立创作出相似内容。
3. **违法挪用**：先滤除思想、事实、标准场景和公有领域材料，再判断剩余表达是否达到 substantial similarity，并确认触及哪一项 § 106 权利。
4. **抗辩或限制**：fair use、first sale、许可（license）、§ 117、DMCA safe harbor 等是否适用。
5. **救济**：实际损失、侵权利润、法定赔偿（statutory damages）、律师费、禁令（injunction）。

- **Fixation（固定）**：表达被记录在足以稳定保存和再现的介质上。写入磁盘、录音、保存图片都算；没有被记录的即兴表达可能不算。
- **自动产生**：原创表达一固定，版权就有了。登记影响的是能否起诉和能拿到哪些赔偿，不是版权的出生时间。

> [!NOTE]+ 固定的边界
>
> § 101：作品须由作者或经其授权，固定在 “sufficiently permanent or stable” 的介质上，能被感知、复制或传播 “for more than a transitory duration”。
>
> - **RAM 副本**：通常已构成 copy，即使断电即消失（*MAI Systems v. Peak*, 9th Cir. 1993）。
> - **瞬时缓冲**：每个 bit 停留不超过 1.2 秒并随即被覆盖，只是 transitory，不是 copy（*Cartoon Network v. CSC Holdings*, 2d Cir. 2008）。
> - **会动的作品**：*Tangle v. Aritzia*（9th Cir. 2025-01-14）。Tangle 的可扭转雕塑由 17–18 节 90° 弯管连接而成，Aritzia 2023 年在橱窗里摆了 18 节的同类雕塑。地区法院（district court）以“没有固定”驳回；第九巡回推翻：电影、交响乐、舞蹈本来就在动，保护覆盖作品全部动态范围，而不只某一个姿势。否则 Calder、Tinguely、Rickey 的动态雕塑都会失去保护。
> - **对 AI 的意义**：推理时把网页或作品载入内存，接近 *MAI*；流式处理中极短暂的缓冲，接近 *Cartoon Network*。Agent 的即时复制先要过“是不是 copy”这一关，才谈得上 fair use。

- **侵权要证明两件事**：
  1. 被告 **actually copied（实际复制）**：可用直接证据，更常见的是 access（接触机会）+ 相似性推断。
  2. 复制的部分与原作中**可保护的表达**构成 **substantial similarity（实质性相似）**：先滤掉事实、思想、标准场景和公有领域材料。
- **§ 102(b) 的排除清单**：idea、procedure、process、system、method of operation、concept、principle、discovery。软件、API、模型的版权争议，常常落回这一句。

> [!EXAMPLE]+ 同一句在软件里怎么拆
>
> | 对象 | 归类 | 依据 |
> | :--- | :--- | :--- |
> | 排序算法的思想 | idea，不受保护 | § 102(b) |
> | 一段具体的 quicksort 实现 | expression，受保护 | — |
> | 菜单怎样组织功能 | method of operation | *Lotus* |
> | API 声明代码 | 假定可版权，按 fair use 放行；可版权性没定 | *Google v. Oracle* |
> | 训练学到的统计规律 | 更接近 idea / 事实 | — |
> | 复制训练语料 | 另一件事，落在 § 106 复制权 | — |

### 权利类型

§ 106 给的是“一束”专有权利，争点常常不是“有没有版权”，而是被诉行为碰到了哪一项：

| § 106 | 权利 | 意思 | AI 里的例子 |
| :--- | :--- | :--- | :--- |
| (1) | 复制 | 制作副本 | 训练时下载、复制语料 |
| (2) | 演绎 | 改编、翻译、影视化等重铸；新作者自己的表达不抵消对原作的利用 | 输出改写了原作 |
| (3) | 发行 | 向公众转移副本的所有权或占有 | 分发据称“含有作品”的模型 |
| (4) | 公开表演 | 表演、播放 | — |
| (5) | 公开展示 | 公开展示图像 | 把角色图当产品功能公开展示 |
| (6) | 数字音频传输 | 仅限录音作品 | 流媒体播放 |

### 作品类型

- **§ 102(a) 八类**：literary；musical（含歌词）；dramatic（含配乐）；pantomime 与 choreographic；pictorial, graphic, sculptural（PGS）；motion picture 与其他 audiovisual；sound recording；architectural。
- **一个产品，多部作品**：
  - 一首歌：曲谱 = musical work；歌词单独 = literary，配曲后 = musical；录音 = sound recording；MV = audiovisual；专辑封面 = pictorial。
  - 一部电影：剧本是 literary，表演固定在 motion picture 里。
  - 电子游戏、MV 都是 audiovisual work。
  - 所以 AI 音乐的授权要同时处理词曲（composition）和录音（master），权利人往往不是同一家。
- **§ 103 汇编与演绎**：
  - Compilation：对已有材料或数据作原创性的选择、协调、编排，例如名言库、地区服务指南、年度最佳短篇选、网站的图文组合、专题期刊。
  - Derivative work：基于已有作品的改编、翻译、改写，例如程序的新版本、网站改版、小说改电影、照片改画作、乐曲改编。
  - 新版权只覆盖新增的那部分，与原作版权分开；用受保护的原作汇编或演绎，都要许可。

### 侵权样本

- **Bratz 案**：Mattel 诉 MGA，2008 年陪审团（jury）判 Mattel 最多约 1 亿美元，只是它所请求 18 亿美元的约 5.5%。雕塑作品（玩偶造型）也能打出大额诉讼。
- **Miley Cyrus “Flowers”**：2024 年 9 月，Tempo Music 在 C.D. Cal. 起诉，主张该曲复制了 Bruno Mars “When I Was Your Man” 的音乐表达。音乐案的核心仍是 access + 受保护元素的实质性相似。
- **Fortnite 舞蹈表情**：多起诉讼主张游戏表情复制了名人舞步。难点在于简单舞步和社交舞通常算不上可登记的 choreographic work。

### 登记与期限

- **不登记也有版权**：美国 1989 年加入伯尔尼公约后，版权标记（©）也不是成立条件。
- **登记的作用**：
  - **能起诉**：美国作品通常要先完成登记才能在联邦法院起诉（*Fourth Estate v. Wall-Street.com*, 2019）；外国作品规则不同。
  - **能拿法定赔偿和律师费**（§ 412）：原则上要在侵权开始前登记；首次发表后三个月内登记也算。没及时登记，仍可请求实际损失和被告的侵权利润。
- **法定赔偿**（§ 504(c)）：每部作品 750–30,000 美元，法院酌定；故意侵权最高 150,000 美元，不知情最低 200 美元。
  - “每部作品”怎么数受 compilation、collective work 规则影响，不能把每个文件各算一件；但大型语料里作品数量多，仍会把赔偿放大到很高。
- **期限**：
  - 1978 年以后自然人创作：作者终身 + 70 年。
  - 雇佣作品、匿名、化名作品：发表起 95 年或创作起 120 年，取较早者。
  - 1998 年 Copyright Term Extension Act 再延 20 年，*Eldred v. Ashcroft*（2003）认定合宪：limited Times 还在，只是很长。

### 登记表

- **表格分类**：Form TX（文字作品，含计算机程序）、Form VA（视觉艺术）、Form SR（录音）。
- **关键栏位与对应规则**：
  - **作者与 work made for hire**：雇佣作品的“作者”填雇主；对应归属一章。
  - **国籍或住所、生卒年**：决定外国作品待遇和 life+70 的起算。
  - **创作完成年份 vs 首次发表日期**：两件事；发表日期影响 95/120 年期限、§ 412 的三个月窗口和交存要求。
  - **Claimant 与 transfer**：权利人和作者不同时，要写明权利怎么转过来。
  - **Previous registration**：是否已登记过早期版本。
  - **Derivative work or compilation**：6a 写所依据的已有材料，6b 写本次新增的材料。只就新增部分主张，正是 § 103 的结构；AI 生成内容按 Copyright Office 指引也在这里排除。
  - **Certification**：由作者、权利人或其代理人签名证明属实。
- **交存（deposit）**：未发表作品交一份完整副本；已发表作品交两份 best edition；首次在美国境外发表交一份；集体作品中的一篇交一份集体作品的 best edition。

### 发行用尽

- **First sale（首次销售 / 用尽）**：买来的正版实体书可以再卖，版权人管不着这一本的去向。
  - *Kirtsaeng v. John Wiley*（2013）：在国外合法制造、进口到美国的教材也适用。
- **数字副本不一样**：转卖数字文件会多出一份拷贝，不是“卖掉手里那一本”。*Capitol Records v. ReDigi*（2d Cir. 2018）因此不让 first sale 覆盖二手数字音乐转售。
- **和 *Bartz* 的区别**：Anthropic 买二手书、拆开扫描、销毁纸本。Alsup 认定这种不增加副本总数的数字替代是 fair use——这是 fair use 的结论，不是 first sale 自动覆盖了扫描件。

<br>

## 独创

### 最低门槛

- **Originality（独创性）只有两个要求**：
  1. **independent creation**：自己创作，不是抄的；
  2. 极低程度的 **creativity**（创造性）。
- **不要求**新颖、好看、有价值，也不要求以前没人做过。两人各自独立拍出近似照片，各自都可能有版权——这和专利（patent）的 novelty 完全不同。

- **门槛有多低**：*Feist* 说绝大多数作品都有 “some creative spark, ‘no matter how crude, humble or obvious’”；只有一小类作品的创造性 “utterly lacking or so trivial as to be virtually nonexistent”。
- **低于门槛的样本**：Cooperstown Bat 申请登记“美国国旗图案棒球棒”被拒，理由是只把熟悉的国旗换成三维形式，属于 “very slight variation”。
- **版权局不做什么**：不作审美判断，不比较作品，不考虑商业成功。被模仿、卖得好，都不是可版权性的证据。

### 事实汇编

> [!EXAMPLE]+ *Feist v. Rural Telephone*, 499 U.S. 340 (1991)：花力气不等于有版权
>
> - **事实**：Rural 是堪萨斯一家电话公司，白页按姓氏排列。Feist 编区域号簿，Rural 不授权，Feist 照抄，连 Rural 故意放的几个虚构条目都抄了——所以“接触过”很好证明。
> - **判决**：Rural 的白页不受版权保护。
>   - 拒绝 sweat of the brow（“流汗原则”）：辛苦收集本身不够。
>   - 事实不受保护；汇编要受保护，须有 original selection, coordination, or arrangement。
>   - 按字母排电话号码，创造性为零。O'Connor：只要求独立创作加 “a modicum of creativity”，门槛极低，但不是零。

| 对象 | 版权状态 |
| :--- | :--- |
| 测量数据、新闻事件、判决原文、股价 | 不受保护 |
| 对事实的选择、编排、注释 | 可能受保护，范围很窄 |
| 小说、插画、大量代码 | 保护的核心区 |

- **后续**：*Thomson Reuters v. Ross* 打的就是这一层。Westlaw 的 headnotes 不是判决本身，而是编辑对判决的归纳；D. Del. 2025 年认定它有 originality。判决公开，headnotes 不一定公开。
- **对数据工作**：花了大量时间整理数据，本身从来不是保护理由。

<br>

## 思想

### 思想—表达二分

- **规则**：思想、事实、方法、系统留给公众；作者把它写成的具体文字、画面、代码才可能受保护。
- **用法**：不是先把整部作品标为“有版权”再禁止一切相似，而是比较时逐层滤掉不受保护的部分。
- ***Baker v. Selden*, 101 U.S. 99 (1879)**：Selden 写书讲一套记账法，附空白表格；Baker 用了类似表格。版权保护书里的说明文字，不保护记账方法本身。方法想要排他，得去申请专利。
- **抽象层次测试**（Learned Hand，*Nichols v. Universal Pictures*, 2d Cir. 1930）：
  - 从具体对白往上抽象：情节 → 角色类型 → 主题 → idea。抽到某一层，版权就不再管。
  - Hand 承认没人能精确划出这条线。软件、模型、prompt 的保护范围之争，至今仍是在不同抽象层之间移动。

### 合并、标准场景与抄袭

- **Merger（合并）**：某种思想只有极少几种写法时，表达和思想“合并”，版权让开；否则等于用版权垄断了思想。
- **Scènes à faire（标准场景）**：西部片的酒吧、代码里的常规循环和错误处理，几乎人人都这么写，保护很弱。
- **独立创作不侵权**：版权只挡抄袭（copying）；专利连独立发明也挡。
- **抄袭 ≠ 侵权**：plagiarism 是学术伦理，infringement 是法定权利。
  - 抄了没有版权的事实，可能是学术抄袭，却不一定侵权。
  - 改写过了，仍可能侵权。

### 实用物品

- **规则**：有 “intrinsic utilitarian function” 的物品本身不受版权保护，例如椅子、花瓶、餐具。
- **可分离的部分**：物品上的 PGS 特征，只要 “capable of existing independently of the utilitarian aspects”，可以受保护。椅背雕花、银餐具上的浮雕可以；椅子和餐具本身的造型不行。
- ***Star Athletica v. Varsity Brands*（2017）**：
  - 不要求物理上能拆下来，只要能在观念上（imaginatively）把它看成独立的二维或三维艺术作品。
  - 啦啦队服上的条纹、V 形色块因此可受保护。
- **与其他制度的分工**：产品的功能性（functionality）外形走专利，装饰性外观走外观专利，来源识别走 trade dress，版权只管可分离的艺术部分。

### 不受保护的对象

- **未固定的**：没有录下来的演讲、讲课。
- **标题、名称、短语、口号**：交给商标（trademark）处理。
- **事实**：包括菜谱里的配料清单；菜谱的说明文字和编排可能受保护。
- **字体（typeface）**：一般不保护；生成字体的软件可以作为程序受保护。
- **颜色本身**：版权局不登记颜色。例外是给黑白片上色，须由人从大量色彩中作出众多选择，且变化不只是 trivial。
  - Google Deep Dream 把 *American Gothic* 自动转成 Munch 风格，颜色不是人加的，这部分没有人类作者。

### 字母与语言

> [!EXAMPLE]+ Myst 字母被拒登（Copyright Office Board of Appeals, 2004）
>
> - **事实**：游戏公司 Cyan 申请登记小说 *MYST: The Book of Ti'ana* 中虚构语言的一套字母和一套数字符号，作为 “2-dimensional artwork”。
> - **申请方理由**：
>   - 字符是新创的，不来自公共领域，不是 idea 或交流方法；只主张字形的艺术表达。
>   - 引 *Lotus* 地区法院意见：书也用来交流思想，照样受保护。
>   - 引 *Toro*：有 “meaningful pattern” 的系统可以受保护；还类比新的编程语言。
> - **驳回理由**：
>   - 一套用于书写、带读音或含义的字符，本质上是 system of communication，落入 § 102(b)，不论新旧。
>   - *Baker v. Selden*：记账表格虽有表头和划线，仍属系统本身。
>   - 国会 1976 年立法报告把 typeface 定义为一组按重复设计元素组成、用于排字的字母、数字和符号，明确不作为 PGS 保护；37 C.F.R. § 202.1(a) 规定 lettering 本身不可登记。
>   - 用 Cyrillic 写的小说受保护，Cyrillic 字母本身属于公共领域（*Boisson v. Banian*, 2d Cir. 2001：“the alphabet is in the public domain”）。
>   - *Toro* 那句话只是 dicta，且该案依赖已被 *Feist* 否定的 sweat of the brow；*Lotus* 第一巡回最终认定菜单是 method of operation。
> - **启示**：申请人一会儿说“不是交流系统”，一会儿说“是不在公共领域的语言”，自相矛盾。主张越靠近“别人能用它交流”，越会撞上 § 102(b)。

- **编程语言**：
  - 美国至今没有法院判决语言本身能否受版权保护；用语言写成的作品当然可以。
  - *SAS Institute v. World Programming*（CJEU 2012-05-02）：程序的功能、编程语言和数据文件格式都不是“表达”，不受欧盟软件指令保护。
- **Klingon**（*Paramount v. Axanar*, C.D. Cal.）：
  - Paramount 主张粉丝电影侵犯 *Star Trek* 版权，侵权清单里包括使用 Klingon 语。
  - Language Creation Society 提交部分用 Klingon 写的法庭之友意见（amicus brief）挑战这一主张，法院未准许提交。
  - Paramount 没有坚持语言本身的主张；Klingon 的发明者 Marc Okrand 也公开说语言不能被版权垄断。2017 年开庭前和解（settlement）。

### 复制与实质性相似

- **先过 copying 这一关**：*Selle v. Gibb*, 741 F.2d 896 (7th Cir. 1984)，Bee Gees《How Deep Is Your Love》。没有 access 的证据，只靠两首歌听起来像，不能认定抄袭。模型输出案也一样：先问有没有接触，是不是独立生成了相似的东西。
- **实质性相似不是单一测试**：
  - 常见问法：普通观察者会不会觉得是抄的。
  - 第九巡回等法院对复杂作品拆成两步：extrinsic（客观比较结构和保护范围）+ intrinsic（普通人的整体印象）。这不是全国统一措辞。
  - 无论哪种问法，都先滤掉 idea、公有领域和常规写法再比。
- **模型输出的常见争点**：这段是 memorization（记住了原文），还是同一主题下的独立生成？
  - NYT 诉状（complaint）展示过近乎逐字的文章输出，可作 memorization 的证据，但不是法院的侵权认定。
  - 角色姿势、配色“有点像”，可能达不到实质性相似。
  - 抽象风格本身通常不能被版权垄断；但“风格化输出”仍可能复制可识别的角色、独特情节、构图、台词或大段原文。
- **结论**：分析输出侵权，要把具体对象放到这根尺子上逐项比，不能停在“看着像”。

<br>

## 软件

### 对象分层

- **法律定位**：§ 101 把计算机程序定义为“使计算机产生一定结果的语句或指令”，作为 literary work（文字作品）保护。
- **§ 117**：合法副本的所有人可以为运行程序制作必要的拷贝和备份。它管的是自己用这份程序，不管把代码送进训练集。
- **难点在哪**：逐字抄源码最好认；难的是结构、接口、菜单、API。15.628J 把 *Baker* 和 *Lotus* 放在同一周，就是为了把“看起来像抄”拆成思想、操作方式和表达三层。
- **一个程序里的层次**：逐字源码 → 模块组织 → 数据结构 → 接口名称 → 运行结果 → 底层功能。
  - 越靠近具体源码，保护越强。
  - 越靠近算法、协议、兼容要求和操作方法，越容易落入 § 102(b)、merger 或 scènes à faire。
- **保护什么**：源码和二进制都受保护；保护的是程序员的原创表达，纯功能部分不受保护。
- **literal vs non-literal**：
  - literal：源代码、目标代码。
  - non-literal：structure, sequence and organization（SSO）、用户界面。
  - 非字面元素能否受保护，取决于它是 idea 的表达，还是 idea 本身。
- **屏幕显示**：通常与生成它的程序视为同一部作品。
- **HTML**：多由建站工具自动生成，一般不算可版权的“代码”；由人写、且有足够创作性时才可作为 literary work 保护。
- **§ 102(b) 的双向含义**：它不扩大也不缩小保护范围。表达不会只因“体现在某种操作方法里”就失去保护；Federal Circuit 在 *Oracle* 里正是这样读的，*Lotus* 则偏向另一端。

### 抽象—过滤—比较

- **背景**：*Whelan v. Jaslow*（3d Cir. 1986）曾把程序的 SSO 保护得很宽，几乎把功能目的也当成表达。*Computer Associates v. Altai*, 982 F.2d 693 (2d Cir. 1992) 把它压了回去。
- **Altai 三步法**（abstraction–filtration–comparison）：
  1. **抽象**：把程序分层——源码、目标码、模块、算法、功能。
  2. **过滤**：滤掉 idea、公有领域材料、为效率不得不这样写的部分、由外部因素决定的部分（硬件、协议、兼容需求）。
  3. **比较**：只比较剩下的可保护表达。
- **效果**：滤完之后能主张的，往往比权利人想的窄。效率限制尤其关键：某种排序在某种数据结构下只能那样写，merger 直接适用。
- **目的**：防止法院把“程序整体相似”直接当成侵权。

### 兼容与反向工程

> [!EXAMPLE]+ *Lotus v. Borland*, 49 F.3d 807 (1st Cir. 1995)：菜单是操作方法
>
> - **事实**：Borland 的 Quattro Pro 做了 Lotus 1-2-3 兼容模式，菜单命令层级几乎照搬，好让会计不用重新学。
> - **判决**：第一巡回认定菜单命令层级是 method of operation，§ 102(b) 不保护。若用户已经学会的操作方式受版权保护，竞争者要做兼容就得先拿许可。
> - **效力**：最高法院（Supreme Court） 4:4 维持，没有写意见。所以这是第一巡回的法，不是全国规则，但影响很大。CLI 命令、快捷键、配置格式、各种“兼容层”，分析时都还用这条思路。

> [!EXAMPLE]+ *Sega v. Accolade*, 977 F.2d 1510 (9th Cir. 1992)：为兼容而反向工程
>
> - **事实**：Accolade 要给 Sega Genesis 做未经授权（without authorization）的游戏，反汇编时中间复制了整份代码。
> - **判决**：为了读到不受保护的功能、做出兼容，中间整份复制可以是 fair use。功能本身不受版权；为了看见功能而做的拷贝，和拿表达去替代原作市场，不是同一种用途。
> - **今天的用处**：agent 读网页、安全研究反编译、为接 API 而获取文档，常要回到这里。
> - **注意**：DMCA § 1201 反规避（anti-circumvention）是另一条诉因。即使复制是 fair use，绕过技术保护措施（technological protection measure）仍可能单独负责。

<br>

## 接口

### 声明与实现

API 至少分三层：

| 层次 | 作用 | 兼容实现通常怎么做 |
| :--- | :--- | :--- |
| **declaring code**（声明代码） | 告诉开发者名称、参数和调用方式 | 照搬，否则程序员已有的代码跑不起来 |
| **organizational structure**（组织结构） | package、class、method 怎么分组 | 部分照搬 |
| **implementing code**（实现代码） | 真正完成运算 | 自己重写 |

*Google v. Oracle* 处理的正是这种分层，而不是把整个 Java 实现搬进 Android。

### 联邦巡回 2014：可版权性

> [!INFO]+ 事实与程序
>
> - **Java 的结构**：Java SE API 有 3 万多个 method，Sun 把它们组织成 166 个 package、约 3,000 个 class。调用时写 declaring code（例如 `java.lang.Math.max(int x, int y)` 的声明），由 implementing code 完成计算。
> - **谈判破裂**：Google 曾向 Sun 谈 Java SE 许可；Sun 要求 Android 与 Java 平台兼容，Google 不同意。Android 最终既用 Java 语言，又不与 Java 兼容，应用不能互通。
> - **复制范围**：37 个 package 的 SSO 和全部 declaring code（课件计 11,330 行，最高法院写约 11,500 行），implementing code 基本自写；另有 9 行 `rangeCheck` 逐字相同。
> - **起诉**：Oracle 2010 年 1 月以 74 亿美元收购 Sun，同年 8 月 13 日在 N.D. Cal. 起诉，主张专利和版权侵权；Ellison 称 Java 是 “the single most important software asset we have ever acquired”。Alsup 法官把案件拆成版权、专利、赔偿三段。
> - **2012 一审**：陪审团被告知假定 SSO 可版权，认定侵权，但在 fair use 上僵持；专利部分 Google 胜。Alsup 随后自行裁定 SSO 与 declaring code 不可版权。
> - **2014-05-09 Federal Circuit 推翻**，发回审 fair use；最高法院当时拒绝受理 Google 的复审申请。之后的 fair use 再审中陪审团判 Google 胜，Federal Circuit 2018 年再次推翻，这才有了 2021 年的最高法院判决。

- **SSO 可版权**：package-class-method 这种组织方法本身是 § 102(b) 的 method of operation；但 Sun 选择的具体命名和组织方式有 “unlimited options”，属原创表达。Google 自己也承认 SSO “original and creative”。
- **声明代码可版权**：
  - 同样功能可以有无数种写法；没有什么阻止 Google 自己写声明代码。Google 的 Java 专家称写声明代码是 “an art, not a science”。
  - 不能拆成一条条 short phrase 来否认保护：问题是这些短语“如何串在一起”是否有创作性。
  - 方法声明的选择与编排构成受保护的 compilation。
- **流行不等于失去保护**：作品成了行业标准，不会因此丧失版权（*Practice Management*, 9th Cir. 1997）。Google 本可自建 API 并说服程序员采用，它复制是为了利用已经熟悉 Java 的开发者群体，“that desire has nothing to do with copyrightability”。

### 最高法院 2021：合理使用

- ***Google v. Oracle*, 593 U.S. 1 (2021)**：6:2，Breyer 撰写多数意见。**假定**声明代码可以受版权保护，直接以 fair use 判 Google 胜诉。
- **理由**：
  - 为移动设备重新实现平台，形成了不同的使用环境。
  - 声明代码和程序员已经学会的调用方式紧密相连，复制它让开发者能把技能带到新平台。
  - 结合 Java SE 的既有市场、Android 的移动用途和潜在市场证据，第四因素倾向 Google。这不是说两者完全没有市场关系。
- **异议**：Thomas 认为 Google 取走了 Java 的核心，损害了 Oracle 的许可市场。
- **不能概括成“API 没有版权”**：最高法院没有回答可版权性，接口声明是否受保护至今没有全国统一的终局规则。

> [!TIP]+ 两级法院的分歧在哪
>
> - Federal Circuit 把兼容性、开发者锁定这些因素挪出可版权性，放进 fair use；最高法院接过这个结构，直接在 fair use 上判 Google 胜，回避了可版权性。
> - 对工程师的含义：复制接口能否过关，主要看用途和市场证据，而不是“接口天然免费”。
> - Alsup 在 *Oracle* 地区法院阶段两次判 Google 胜（不可版权、fair use），都被 Federal Circuit 推翻，最高法院最后从 fair use 支持了 Google。2025 年他又审 *Bartz v. Anthropic*。读 *Bartz* 时，这段经历有助于理解他怎样处理“复制结构、改变用途”的论证。

<br>

## 合理使用

### 四因素

Fair use（17 U.S.C. § 107）是**积极抗辩**：被告在诉讼里主张，不是事先发放的许可证，也没有“复制不超过 X%”这种固定规则。

| 因素 | 问什么 | 要点 |
| :--- | :--- | :--- |
| 1. purpose and character | 被诉使用的目的、是否商业、是否 transformative（加入新目的或新意义） | 判例重点之一 |
| 2. nature of the work | 原作偏事实还是偏创作、是否已发表 | 小说比事实资料更靠近保护核心；通常权重较低 |
| 3. amount and substantiality | 复制了多少、是否取走了作品的 heart | 比例小也可能取走核心；整部复制也不一定自动败诉 |
| 4. market effect | 这种使用若广泛发生，会不会替代原作或权利人应能控制的许可市场 | 判例重点之一 |

- **不是打分**：不是四项三比一就赢，法条也没规定权重。判例常重点讨论第一、四因素，但要结合具体事实整体衡量，两者也互相牵连。
- **transformative ≠ 技术变换**：OCR、tokenization、embedding 都是技术过程；法律问的是这次使用在市场和表达利用上做了什么。*Warhol* 之后尤其要把用途写具体。

### 用途界定

- ***Campbell v. Acuff-Rose*, 510 U.S. 569 (1994)**：2 Live Crew 讽拟 Roy Orbison 的《Oh, Pretty Woman》。
  - 商业性不再一票否决。
  - 讽拟（parody）必须借用足够多的原作，公众才听得出在讽刺什么，第三因素因此让步。
  - Transformative 从此成为第一因素的中心词，此后几乎任何新用途都被说成 transformative，这个词被拉得很宽。
- ***Andy Warhol Foundation v. Goldsmith*, 598 U.S. 508 (2023)**：把它收紧。
  - Goldsmith 给 Prince 拍了照片，Vanity Fair 1984 年获授权请 Warhol 据此做一次丝网作品。Prince 去世后，Condé Nast 又向 Warhol 基金会取得橙色 Prince 图像作杂志封面。
  - 最高法院只分析这一次商业许可，不抽象评价 Warhol 的艺术有没有新意义；这次用途和 Goldsmith 照片本来的授权市场实质相同。
  - Sotomayor 撰写多数意见；Kagan 异议，反对多数意见对用途和艺术转换的界定。

<span style="background:#ffe58f">Transformative 必须落在被诉的那一次使用上。新技术、新中间物、新美学，都不自动等于新用途。</span>

- **对训练案的意义**：不能因为 LLM 在技术上把 token 变成了权重，就宣布第一因素结束。Alsup 在 *Bartz* 里把四个因素逐项写完；Chhabria 在 *Kadrey* 里把第四因素拉回中心。两份意见都引了 *Warhol*。

### 教育、搜索与缓存

- **课堂复印**：*Cambridge University Press v. Becker*（N.D. Ga. 2012 及后续第十一巡回），大学电子课件摘录多少算合理。15.628J Session 5 用它练 § 107：把“数量”“市场”“教育目的”逐一映射到四因素，写得很细。适合当练习，不宜直接类比 AI 训练。
- **互联网平台**（6.4590 Week 7）：
  - *A&M Records v. Napster*（9th Cir. 2001）：集中式索引，间接责任好证。
  - *Grokster*：改成去中心化之后，原告改打诱导侵权（induced infringement）。
  - *Perfect 10 v. Amazon*（9th Cir. 2007）：缩略图检索，方向接近后来的 Google Books——用途是检索，展示的是碎片。
- **对 agent 的类比**：浏览缓存、搜索引擎快照更靠近这一支，而不是预训练使用整本书。

<br>

## 间接责任（次级侵权）

> [!INFO]+ 名词
>
> **Secondary liability**（次级责任），也叫 indirect / secondary infringement（间接侵权、次级侵权）：自己没有亲手实施侵权，但因为帮助、控制或诱导了**别人的**直接侵权（direct infringement）而负责。版权、专利、商标都有，但要件不同，见体系一章“次级侵权对照”。版权法条文里没有写次级责任，全部来自判例；专利法则写进了 35 U.S.C. § 271(b)、(c)。

### 三条责任路径

自己没有直接侵权，也可能因为别人的侵权负责。三条路径都以**某个直接侵权行为存在**为前提：

| 路径 | 要件 | 直白说法 |
| :--- | :--- | :--- |
| **Contributory**（帮助） | 知道 + 实质性帮助 | 明知别人侵权还提供关键帮助 |
| **Vicarious**（替代） | 有权且有能力控制 + 直接经济利益 | 即使不知道，也因能管、又从中获利而负责 |
| **Inducement**（诱导） | 以促进侵权为目的散布工具 | 看主观目的和推广证据 |

仅证明模型“有能力”生成侵权内容，通常不足以说明平台应负哪一种责任。

### 双用途工具

- ***Sony v. Universal*（1984，Betamax 录像机）**：设备有 substantial noninfringing uses（大量合法用途），厂商不因用户可能拿来侵权而负间接责任。
- ***MGM v. Grokster*, 545 U.S. 913 (2005)**：P2P 软件同样有大量合法用途，但 Grokster 的广告和内部文件表明目的就是接替 Napster，诱导侵权成立。只有“双用途”能力不够，还要看散布者是否以促成侵权为目的去推广。

> [!EXAMPLE]+ *Cox Communications v. Sony Music*, 607 U.S. ___ (2026-03-25)：只是“知道”不够
>
> - **事实**：Cox 是网络接入服务商（ISP）。Sony 等唱片公司发出大量侵权通知，主张 Cox 明知部分用户反复用 P2P 下载音乐，仍不断网。
> - **程序**：2019 年弗吉尼亚东区陪审团判赔约 10 亿美元；第四巡回 2024 年撤销 vicarious 部分、维持 contributory 部分，发回重审赔偿额。
> - **判决**：最高法院撤销第四巡回。服务商只有在**意图**让服务被用于侵权时才负 contributory liability，意图只能用两种方式证明：
>   1. 诱导侵权（*Grokster*）；
>   2. 提供专为侵权设计、缺乏实质性合法用途的服务（*Sony*）。
> - 仅仅知道用户在侵权、没有终止账户，不够。Cox 的用户协议禁止侵权，还有警告、暂停、终止流程，两条都不满足。
> - **意见**：Thomas 撰写多数意见；Sotomayor（Jackson 加入）只同意结果，认为多数意见把次级责任收得过窄。判决只处理 contributory，不涉及 vicarious。

- **套到生成模型**：模型能生成高度相似的内容，不等于开发者要为用户的输出负责。按 *Cox*，权利人要么证明开发者鼓励这种用途（广告、首页示例、把生成某个角色当卖点——正是 *Disney v. Midjourney* 诉状的写法），要么证明产品专为侵权而做。通用模型有大量合法用途，后一条很难证。
- **不管的部分**：开发者自己在训练、保存语料时的复制，是直接侵权问题，*Cox* 不处理这一段。

### 平台安全港与技术措施

- **§ 512 安全港**（notice-and-takedown，通知—删除）：
  - 流程：权利人发送合格的侵权通知 → 服务商及时删除或断开访问 → 用户可提交 counter-notice（反通知）→ 权利人决定是否起诉。
  - 分接入、缓存、存储、检索四类服务，各有条件。不符合条件只是回到普通间接责任，不等于平台自动侵权。
  - 作用是限制服务商的金钱责任和部分禁令，不表示用户内容已被认定合法。
- **§ 1201 反规避**：绕过技术保护措施本身就是独立诉因，见下一节新案。
- **§ 1202 版权管理信息（CMI）**：
  - CMI 指作者、作品名、版权人、使用条款等与权利管理有关的信息；不是所有元数据都算。
  - 责任通常要求行为人知道自己在移除或更改 CMI，并知道或有合理理由知道这会诱发、促成、便利或掩盖侵权。
  - *Kadrey* 中针对 Llama 训练的 CMI 主张失败，是因为该案记录里训练复制被认定为 fair use，无法证明移除 CMI 是为了掩盖该项侵权。这个结论不能推广成“所有 fair use 案件里 § 1202 主张都会消失”。

### DMCA 新案

- ***Doe v. GitHub*（9th Cir. 2026-09-16）**：匿名开发者起诉 GitHub、Microsoft、OpenAI，主张 Copilot、Codex 用其代码训练，输出时去掉了署名（attribution）、版权声明和许可证条款，违反 § 1202(b) 禁止 “remove” 或 “alter” CMI 的规定。核心难点：输出不是原样复制时，还算不算“移除”了原作的 CMI。
- ***Ted Entertainment v. Apple*（N.D. Cal. 2026）**：主张 Apple 绕过 YouTube 的技术保护措施抓取（scraping）数百万视频训练生成模型，违反 § 1201(a)(1)(A) 反规避条款。同类诉讼还针对 OpenAI、Amazon、NVIDIA、Meta、ByteDance、Snap。
- ***Yout v. RIAA*（2d Cir. 2026）**：Yout 请求确认其浏览器端下载工具不违反 DMCA；RIAA 主张它绕过 YouTube 的 “rolling cipher” 访问控制。Yout 自称是现代版录像机，只是录下公开可访问的媒体。
- ***Google v. SerpApi*（N.D. Cal. 2025）**：Google 主张 SerpApi 规避 SearchGuard 抓取搜索结果页上的受版权内容，违反 § 1201(a)(1)(A)；把抓取工具卖给客户，又构成 § 1201(a)(2) 的 trafficking。Reddit 在纽约起诉 SerpApi 和其客户 Perplexity，结构相同。

> [!TIP]+ 为什么原告转向 § 1201
>
> - § 1201 的责任不以侵权为前提：即使训练复制最后被认定为 fair use，绕过访问控制本身仍可独立成立。
> - 证明对象从“作品被怎样使用”变成“技术措施有没有被绕过”，更接近工程事实，也更难用变革性辩护。
> - 对数据团队：数据从哪里、用什么方式取得，和拿来训练什么，是两份法律风险。

<br>

# 专利

> **核心问题**：一个技术方案要满足什么条件才能拿到专利？专利拿到后，能挡住谁、能要多少钱？

本章按一项专利从申请、有效性到侵权和救济的顺序展开：

- **授权框架**：制度层级、四项法定门槛、申请、期限、专利文件结构、发明披露表、授权后复审与专利价值分布。
- **外观设计**：保护对象、图示边界和组件利润。
- **适格性（patent eligibility）**：司法例外、*Alice/Mayo* 两步与软件算法。
- **实用性**：极低门槛与“不可能运行”的例外。
- **新颖性与显而易见性**：既有公开、单一文献、文献组合和 PHOSITA。
- **权利要求**：披露要求、claim 结构、排除权与阻挡专利、解释与逐项比对。
- **侵权与救济**：行为类型、抗辩（defense）、用尽（exhaustion）、赔偿和禁令（injunction）。

## 授权框架

### 制度层级

- **四层来源**：宪法 → 国会（35 U.S.C.，专利法）→ USPTO（审查授权）→ 法院（解释、判定有效与侵权）。
- **上诉**：专利案上诉集中到 Federal Circuit，再到最高法院（Supreme Court）。
- **三种专利**：utility patent（实用专利，管功能）、design patent（外观设计）、plant patent（植物）。CS 里日常碰到的几乎都是 utility patent。

- **地域性**：美国专利只在美国有效，德国专利不会自动在美国得到保护。
  - 想在多国保护，要在各法域分别取得权利：常见路径是先提 PCT 国际申请，再进入各国国家阶段；或走 EPO，再在成员国生效。
  - 境外制造、再进口到美国，另按 § 271(a) 的进口行为和 § 271(g) 的方法所得产品处理。

- **PCT 的作用**（Patent Cooperation Treaty）：
  - 一份国际申请可同时指定 150 多个缔约国，先锁定申请日；没有“国际专利”，授权仍由各国分别决定。
  - 国际阶段出国际检索报告和书面意见，自优先权日起 18 个月国际公布。
  - 通常在自优先权日起 30 个月内进入各国国家阶段，再分别缴费、翻译、审查。
  - 价值在于推迟决定：用 30 个月看市场和检索结果，再决定在哪些国家花钱。

### 四项门槛

专利不是“有一个想法就登记”。申请人向 USPTO 提交 specification（说明书）、drawings（附图）和 claims（权利要求），审查员拿 prior art（现有技术）去比。Utility patent 要分别过四关：

1. **§ 101 subject-matter eligibility**：对象是否属于可专利主题，而非抽象思想、自然规律（law of nature）或自然现象本身。
2. **§ 102 novelty**：是否已有一份 prior-art reference 把 claim 的每个要素公开出来。
3. **§ 103 non-obviousness**：即使没有单篇文献全公开，多份现有技术的组合对 PHOSITA 是否显而易见。
4. **§ 112 disclosure**：说明书是否真正教会本领域技术人员实施，是否表明发明人掌握了所主张范围，claim 是否明确。

这四关不能混。某算法可以是新的，却仍因 abstract idea 过不了 § 101；也可以属于可专利主题，却早已被论文公开而过不了 § 102。

### 申请、期限与授权后程序

- **授权**（issue / grant）：申请审查通过后才授权。申请被公开不等于授权。
- **授权不等于永远有效**：被告（defendant）可在诉讼中主张无效，第三方可在 PTAB 发起 IPR。专利有 presumption of validity（有效性推定），但可以被推翻。

- **期限起算**：自相关申请链中最早的美国非临时申请日起 20 年。provisional application（临时申请）和外国优先权日通常不算起点。
- **两种延长**：
  - PTA（patent term adjustment）：补偿 USPTO 审查拖延的时间。
  - PTE（patent term extension）：主要补偿药品等受监管产品在上市审批中耗掉的时间。
- **起算 ≠ 可以起诉**：专利要授权后才能主张，所以期限从申请日算，起诉却只能在授权之后。

- **期限一览**：utility 20 年（自申请日）；plant 20 年；design 15 年（自授权日）。
- **为什么从申请日起算**：1995 年以前是授权后 17 年。申请人可以让申请秘密地拖很多年，等行业普遍采用后才授权、突然起诉，称 submarine patent。1995 年改为自申请日起算，再加上申请满 18 个月公开（1999 年立法，适用于 2000-11-29 以后的申请），潜艇就很难再藏。
- **PTA 实例**：Neonode 的 US 8,095,879 于 2002-12-10 申请、2012-01-10 才授权，封面注明期限延长 1,228 天。审查拖了近十年，PTA 把被 USPTO 耗掉的时间补回一部分。

### 专利文件结构

- **封面**（INID 代码）：
  - (54) 名称、(75) 发明人、(73) 受让人（assignee）。
  - (*) Notice：PTA 天数、terminal disclaimer。
  - (21) 申请号、(22) 申请日、(65) 在先公开数据。Neonode 2002-12 申请、2004-06 公开，正好 18 个月。
  - (56) References cited：审查中考虑过的 prior art，带 * 的是审查员引用的。
  - (57) 摘要。摘要几乎没有法律意义，界定权利的是 claim。
- **附图**：用编号标出每个部件，说明书和 claim 引用这些编号。
- **说明书**：technical field → background art → summary（要解决的技术问题与方案）→ brief description of drawings → detailed description / preferred embodiments。
- **Claims**：在说明书最后，以 “What is claimed is:” 或 “I claim:” 开头。

### 发明披露表

公司或大学的专利流程通常从 invention disclosure form 开始：发明人先填表，专利律师再判断能否、是否值得申请。表上每一栏都对应一条法律规则。

| 栏位 | 对应规则 |
| :--- | :--- |
| 首次构思、首次书面描述、首次草图、首个原型、首次测试的日期 | conception 与 reduction to practice 的证据 |
| 首次公开披露日期 | § 102(b)(1) 一年宽限期从哪天开始算 |
| 对外公开的情形、保密披露的对象及协议 | 是否已构成 prior art；商业秘密（trade secret）是否还在 |
| 已发表或计划发表的论文 | printed publication；先申请再发表 |
| 技术领域、现有技术的问题、与现有技术的区别 | § 103 的差异与 long-felt need |
| 详细描述发明，包括所有替代实施方式（alternative embodiments），附图 | § 112 的 written description 和 enablement；替代方案写得越全，claim 越能写宽 |
| 局限如何克服 | 改进方向，可能成为 dependent claim 或后续申请 |
| 发明的局限、尚需的测试与开发 | enablement；离 ready for patenting 还有多远 |
| 新概念 / 重大突破 / 现有技术改进 / 现有技术新用途 | claim 能写多宽 |
| 相关论文、专利和申请（无论是否本人的） | 检索起点；向 USPTO 的披露义务 |
| 用最宽泛的方式概括发明 | 独立 claim 的雏形 |
| 全部发明人、签名、见证人 | inventorship；构思日期的佐证 |

> [!WARNING]+ 不要和 IDS 混淆
>
> USPTO 程序里的 Information Disclosure Statement（IDS）是另一件事：申请人依 37 C.F.R. §§ 1.56、1.97 向审查员提交其知道的重要 prior art。故意隐瞒重要文献，可能构成 inequitable conduct，使整件专利不可执行（见侵权一节的 *Therasense*）。内部披露表是给自己律师的，IDS 是给审查员的。

- **AIA（America Invents Act, 2011）**：适用于 2013-03-16 及以后提交的申请。
  - 美国从 first to invent（先发明）改为 first inventor to file（发明人先申请）。但申请人仍须是真正的发明人或其权利继受人，不是谁先交表谁就拿专利。
  - 同时扩大了 prior art 的范围，强烈鼓励尽早申请。
  - 15.628J 同时发了旧 § 102 和新 § 102：读旧案例要先看适用发明日还是申请日。
- **宽限期不是全球通用**：
  - EPO（欧洲）原则上实行绝对新颖性，只有极窄的例外，没有美国式的一年宽限期。
  - 美国的一年宽限期主要保护发明人本人（或源自发明人）的披露，不是任何公开都自动有一年保险。

- **IPR 的由来**：AIA 同时在 PTAB 设立 inter partes review，用 prior art 打 § 102 / § 103，比地区法院（district court）便宜、快。NPE 的诉讼策略因此变了一次。15.628J 那年 IPR 刚开始，现在已是专利实务的日常。

- **名词**：
  - PTAB：USPTO 内的 Patent Trial and Appeal Board（专利审判与上诉委员会）。
  - IPR：授权后的行政审理，第三方用在先专利或印刷出版物主张 claim 因 § 102 或 § 103 无效。
  - NPE（non-practicing entity）：主要持有、许可（license）或主张专利、自己不生产相关产品的实体。它描述商业模式，不等于专利无效或诉讼滥用。

- **谁能发起**：任何人，不需要先被起诉。Neonode 案里 Google 没有被告，仍自己提了 IPR。
- **多快**：PTAB 通常 12–18 个月出决定；地区法院常在此期间中止诉讼（stay）。
- **能打什么**：只有 novelty 和 obviousness，且只能用在先专利和印刷出版物。public use、on sale、written description、enablement 都不能在 IPR 里提，只能回地区法院。
- **标准**：IPR 用 preponderance of the evidence；地区法院推翻已授权专利要 clear and convincing evidence。
- **上诉**：到 Federal Circuit。

### 专利价值分布

- **少数极值钱**：PageRank、Overture 竞价排序专利（见后文 Stanford 段）都直接换成了数亿美元的股票。
- **utility patent 覆盖面很广**（“anything under the sun”）：游戏与玩具（Monopoly 1935 年 US 2,026,082、魔方 1983 年 US 4,378,116）、生物制品（植物肉里含血红素辅因子的食品组合物）、机电装置等。
- **多数不值钱**：课件列举的 Nike 磁吸眼镜、可在裆部分离的“快换裤”（2019-09-27 届满，申请日起 20 年）、带牵绳的“鸟用尿布”、防进食面罩、自助踢屁股机，都拿到了授权。
- **含义**：授权只说明满足法定要件，不说明有市场。专利组合的价值集中在极少数被实施或被主张的 claim 上；大多数是 paper patent。NPE（non-practicing entity）指主要持有、许可或主张专利而不亲自生产相关产品的实体；它是商业模式描述，不等于专利无效或诉讼当然滥用。

<br>

## 外观设计

### 保护对象

- **保护对象**：utility patent 保护技术功能；design patent 保护产品的装饰性外观。
- **怎么读图**：外观专利图中的实线界定被主张的设计；虚线多用来说明环境，不属于保护范围。
- **不是版权（copyright）或 trade dress 的别名**：授权门槛、侵权比较方法、期限和赔偿规则都不同。

- **侵权测试**：*Egyptian Goddess v. Swisa*（Fed. Cir. 2008）的 ordinary observer test——在熟悉现有设计的普通观察者眼中，被控设计与专利图示的**整体视觉印象**是否实质相同。不能只抽出“圆角”“边框”单一概念来比。
- **期限**：2015-05-13 以后提交的美国外观专利，自授权日起 15 年。

- **课件样本**：
  - 自由女神像：Bartholdi 的外观专利 No. 11,023（1879）。
  - 鞋：Nike 的 D498,914（中底造型），与 Wal-Mart 鞋款逐图比对。
  - 手机：Apple 的 D593,087、D604,305（图标网格）、D618,677。
- **Plant patent**：保护无性繁殖的新植物品种，期限 20 年，例如 1942 年的一种玫瑰（Plant Pat. 550）、2020 年的一种天人菊（PP32,215）。

- ***Apple v. Samsung* 不只是“圆角矩形归谁”**：长期诉讼同时涉及 utility patent、design patent 和 trade dress；Stanford CS202 用它讲手机外观设计，最高法院真正处理的是外观专利的赔偿。

### 组件利润

- **问题**：陪审团认定 Samsung 手机侵犯 Apple 关于圆角正面、边框和图标网格的外观专利，并把涉案手机的**全部利润**计入 § 289。
- ***Samsung v. Apple*, 580 U.S. 53 (2016)**，一致判决：多组件产品里的 “article of manufacture” 不一定是整机，也可以是其中一个组件，即使该组件不单独出售。
- **没回答的**：本案到底应以整机、屏幕还是正面外壳为准，法院没有决定，也没给完整的识别测试，只是撤销发回。

- **结局**：发回后 N.D. Cal. 重审赔偿。2018 年 5 月陪审团判 Samsung 赔约 5.39 亿美元，其中约 5.33 亿出自三项外观专利、约 530 万出自两项实用专利；陪审团仍把整机当作 article of manufacture。2018 年 6 月双方和解（settlement），条款未公开，从 2011 年打起的诉讼结束。
- **课上的用法**：CS202 第一讲把 iPhone 发布前后的 Samsung 手机并排放，说明手机外观本身就是专利资产。
- **对生成式设计**：模型做出一款和已有产品很像的工业设计，也要过 ordinary observer test 和专利图示的范围，不是“看着像”就侵权。

> [!EXAMPLE]+ 外观专利赔偿
>
> Utility patent 通常围绕 lost profits / reasonable royalty。Design patent 的 35 U.S.C. § 289 可以追相关 article of manufacture 的 total profit。争点因此不是只看“抄了多少设计”，还要先认定利润对应的 article 是整机还是组件。复杂产品把这一步放大。

<br>

## 适格性

### 判断框架

35 U.S.C. § 101：*any new and useful process, machine, manufacture, or composition of matter*。司法例外三条：laws of nature、natural phenomena、abstract ideas。例外是法官造的，成文法没写。

- **四类对象**：
  - **process**：主要是工业或技术流程；软件和商业方法专利的争议都在这一类。
  - **machine**：机械装置。
  - **manufacture**：制造出来的物品。
  - **composition of matter**：化学组合物、混合物、新化合物。
- **例外的样子**：
  - 自然规律：E=mc²、万有引力、勾股定理。*Funk Bros. Seed*（1948）称其为 “part of the storehouse of knowledge”，“free to all … and reserved exclusively to none”。
  - 自然产物：普通老鼠不行；基因工程改造的老鼠可能可以，因为不是 “nature's handiwork”。
  - 抽象思想：“An idea itself is not patentable.”

- ***Diamond v. Chakrabarty*, 447 U.S. 303 (1980)**：经基因改造、能分解原油的细菌可以申请专利。Burger 引 1952 年立法报告：可专利主题包括 *anything under the sun that is made by man*。§ 101 范围很宽，但三条司法例外仍在：人想出来的抽象思想照样不行。

- **适格性 ≠ 新颖性**：subject-matter eligibility 问“这一类东西能不能用专利法保护”，不问“它有没有新意”。
- ***Alice/Mayo* 两步**：Step 1 看 claim 是否 directed to（指向）司法例外；Step 2 看其余要素单独和组合起来，有没有加入足以把例外变成具体应用的 inventive concept。Inventive concept 不是把 §§ 102/103 的审查原样搬进来。

### 案例演进

按时间顺序，§ 101 在软件和商业方法上的口径一放一收：

| 年份 | 案件 | 对象 | 结论 | 意义 |
| :--- | :--- | :--- | :--- | :--- |
| 1981 | *Diamond v. Diehr*, 450 U.S. 175 | 用 Arrhenius 方程 + 计算机控制橡胶硫化时间 | 适格 | claim 整体是工业过程，不是“公式本身”；软件相关专利常从这里算起。但不表示任意软件 claim 都能过 |
| 1998 | *State Street Bank*, 149 F.3d 1368 (Fed. Cir.) | 金融数据处理 | 适格：*useful, concrete and tangible result* | 商业方法大开，“用计算机做 X”的 claim 涌入 |
| 2010 | *Bilski v. Kappos*, 561 U.S. 593 | 对冲商品价格风险的方法 | 不适格，abstract idea | machine-or-transformation 是有用线索，不是唯一标准；*State Street* 的宽松口径关上。15.628J Session 11、Short Exercise 2 |
| 2012 | *Mayo v. Prometheus*, 566 U.S. 66 | 代谢物浓度与药效的关系 | 不适格，自然规律 | 加上“告诉医生去测、去调剂量”这类常规步骤不够；Breyer 撰写；诊断专利空间收窄 |
| 2013 | *AMP v. Myriad*, 569 U.S. 576 | 基因序列 | 分离的天然 DNA 不适格；cDNA 通常适格 | 见下 |
| 2014 | *Alice v. CLS Bank*, 573 U.S. 208 | 用计算机做中间清算 | 不适格 | 形成今天的两步框架，见下 |

- ***Myriad* 细节**：仅从人体分离出来的天然 DNA 片段不可专利；通常去掉内含子的 cDNA 不是自然状态下的同一分子，可以过 § 101，但仍要满足新颖性等其他条件。极短、与自然序列完全相同的 cDNA，法院没有一概纳入保护。关键不是“经过人为处理就行”，而是主张对象与自然物是否有法律要求的差异。*Myriad* 判在 2013 年 6 月，贴近 15.628J 学期末，OCW readings 仍收了进去。


***Alice* 两步**（中间清算风险的商业方法放到计算机上，不适格）：

```text
Step 1  claim 是否指向司法例外（尤其是 abstract idea）？
        ↓ Yes
Step 2  其余部分有没有 inventive concept，
        把 claim 转成对该 idea 的可专利应用？
        只把 idea 放到通用计算机上，不够。
```

- **别和 §§ 102/103 混**：§ 101 和新颖性、显而易见性不是同一层过滤器；一项 claim 可以新而不适格，也可以适格却不新。
- **之后的松动**：Federal Circuit 在 *Enfish*（2016，自引用数据表结构）中承认，改善计算机本身功能的软件可以在 Step 1 就不是 abstract idea；*McRO*（2016，自动口型动画规则）也过了。
- **现状**：§ 101 成了软件专利最不稳定的一关。USPTO 多次发 eligibility guidance，口径来回拉；国会修改 § 101 从 2010 年代吵到现在，没有落地。

### 软件与算法

- **算法能否专利，看写到哪一层**：
  - 容易被认定为 abstract idea：纯数学、纯商业方法、“用计算机做 X”。
  - 可能通过：具体技术环境里的具体改进——降低延迟、改存储结构、改训练时的硬件调度。
  - claim 必须写出那一层，不能停在 “a neural network configured to…”。

- **ML 专利的 § 103 难题**：
  - PHOSITA 的水平不好定：是读过 Vaswani 2017、会调现成框架的人，还是会写 kernel 的人？
  - “一种 transformer，其特征在于多加了一层残差”，按 *KSR* 容易被认定为已知元件的可预见组合。
  - 更可能通过的：产生意外的技术效果，或解决了长期存在、一直没解决的具体工程问题。商业成功可以作辅助证据，单纯营销成功不够。

<br>

## 实用性

- **来源**：§ 101 只给 “new and useful” 的发明，即 utility requirement。
- **门槛极低**：只要有某种用处。课件样本：
  - 小便时托住额头的支架（US 6,681,419，发明人 Eric Page，与 Larry Page 无关）。
  - 遮盖局部秃顶的梳发方法（US 4,022,227）。
- **operable utility**：PTO 以“不能运行”驳回的比例很低（< 0.05%），但会驳回 “an inherently unbelievable undertaking or involve implausible scientific principles” 的申请（*In re Brana*, Fed. Cir. 1995）。
  - Zinger（2022）：用时间机器和量子力学把无白血病骨髓“凭空出现在空盘子上”。驳回理由：时间旅行尚不存在，量子力学也变不出物质。
  - Swartz（2018）冷核聚变；Fregeau（1985）用磁场改变食物味道；Newman（1989）永动机；Houghton（1970）模仿鸟类扑翼的飞行器。
- **证明能用就行**：1969 年 Puharich 的“增强超感知能力的屏蔽笼”（铜衬 Faraday 笼）拿到了授权。反面教材是 2012 年荷兰人 Jarno Smeets 的“人力扑翼飞行”视频，后来承认是动画师做的假片。
- **含义**：实用性很少成为真正障碍；对软件和 AI 专利，难点几乎都在 § 101 适格性、§ 103 和 § 112，而不在“有没有用”。

<br>

## 新颖性

新颖性判断可以依次检查：

- **关键日期**：先确定申请链、优先权和可能适用的 grace period。
- **公开方式**：再判断材料是否属于 printed publication、public use、on sale 或其他向公众可得的信息。
- **全要素披露**：最后检查一份材料是否直接或必然披露 claim 的全部 limitations 及其组合关系。

- **Prior art 是什么**：不是“以前有人做过类似的”，而是法定范畴里、在法定时间点之前、以法定方式向公众公开的信息。
  - 方式：printed publication（印刷出版物）、public use、on sale、otherwise available to the public。
  - 一篇没人读过但能检索到的论文，可能已经是 printed publication；内部演示不一定是。
  - **美国和外国**的现有技术都算：专利、出版物、公开使用、销售要约，不论发生在哪国。
- **§ 102(a)(1) 原文**：“A person shall be entitled to a patent unless—the claimed invention was patented, described in a printed publication, or in public use, on sale, or otherwise available to the public before the effective filing date of the claimed invention.”
- **目的**：防止别人已发明、或已在公共领域（public domain）的东西被再次专利化。

- **Anticipation（预见）**：要求**一份**现有技术直接或必然公开 claim 的**全部** limitations，而且组合关系也对得上。
  - 不能从论文 A 拿一半、论文 B 拿一半来主张 § 102；多份文献组合属于 § 103。
- **宽限期**：AIA 给发明人自己的公开保留有限的一年宽限期，细节见下文，不能理解成全球通用的一年保险。

- **On-sale bar**：
  - *Pfaff v. Wells*, 525 U.S. 55 (1998)：发明已经 ready for patenting，并进入商业销售或销售要约，即使实物还没做出来，也可能触发。
  - *Helsinn v. Teva*（2019）：AIA 之后，销售细节没向公众完整披露，也不能当然避开（详见下文“宽限期与秘密销售”）。
  - 软件和模型里要检查：早期商业演示、API 预售、向客户出售 checkpoint。普通商务接洽不自动等于法定的销售或要约。
- **对实验室**：习惯“先发论文再考虑专利”，在先申请制下是在跟时间对赌。

- **练习方法**：15.628J 的 *Structural Rubber v. Park Rubber* 是读 prior art 的作业，记方法不记规则：日期、公开方式、是不是同一发明。

### 预见与侵权对称

- **对称原则**：“That which will infringe, if later, will anticipate, if earlier.”（*Peters v. Active Mfg.*, 1889）同一份 claim 映射表，产品在专利之后出现就是侵权，在之前出现就是预见。
- ***WPEM v. SOTI*（Fed. Cir. 2020）**：
  - WPEM 用 SOTI 用户手册第 11 版证明其技术包含 claim 全部要素。
  - SOTI 拿出早于 WPEM 申请日的第 10 版手册，同样功能都在：被控技术本身就是 prior art。
  - WPEM 带偏见撤诉，还被判付对方律师费（§ 285，并不常见）：第 11 版手册已清楚提示存在更早版本，起诉前调查不充分，主张属 frivolous。
- **教训**：主张侵权前，先查被告产品的历史版本、changelog、旧文档和 Wayback Machine。

### 宽限期与秘密销售

- **§ 102(b)(1) 宽限期**：
  - 发明人本人（或源自发明人）的公开，有绝对的 1 年宽限期，过期不补。
  - 第三人在申请日前公开，若发明人在此之前已经公开过、并在 1 年内申请，第三人的公开不破坏新颖性。
- **秘密销售也算 on sale**：
  - *Helsinn v. Teva*（2019）：Helsinn 在申请前约 2 年签了药品销售合同，经销商负有保密义务；仍触发 on-sale bar，须在合同后 1 年内申请。
  - *Celanese v. ITC*（Fed. Cir. 2024）：Celanese 从未公开 Ace-K 甜味剂的制法，但用该秘密方法生产的产品在申请前 1 年以上已经销售；方法专利因 on-sale bar 无效。
  - *NCS Multistage v. Nine Energy*（Fed. Cir. 2026-09-14）：秘密销售会启动 1 年宽限期的计时，但不能像公开披露那样挡住第三人的公开。没有发明人在先的公开，第三人的公开仍会使发明失去新颖性。
- **含义**：
  - 秘密方法要么永久当商业秘密，要么在开始卖产品后 1 年内申请；不能先秘密卖几年，再回头申请专利。
  - 发明人“抢先公开”可以对抗后来的第三人公开，秘密销售不行。公开有防御价值，但也立刻葬送了欧洲等绝对新颖性法域的申请机会。

> [!EXAMPLE]+ SpaceX v. Blue Origin：一篇 1998 年论文撤掉海上回收专利
>
> - **专利**：Blue Origin 的 US 8,678,321（2009-06-15 申请），Bezos 是第一发明人。主题是可重复使用火箭在海上平台垂直着陆，降低航天成本。
> - **Claim 1**：头朝前发射、点燃发动机 → 发射后翻转成尾朝前 → 在水域中放置着陆结构 → 带推力尾朝前垂直降落在该结构上。Claim 2、3 只加“从陆上发射”“着陆结构是浮动平台”。
> - **冲突**：SpaceX 计划的海上回收几乎都落在这项 claim 里，要么付许可费，要么面临禁令。
> - **IPR**：SpaceX 向 PTAB 提交 Ishijima 1998 年的 AIAA 论文，可公开检索，早于申请日，是 printed publication。论文描述的飞行序列包括上升、再入、滑翔并把姿态从头朝前改为尾朝前、在海上船只上带推力软着陆。
> - **结果**：PTAB 认定 “there is clearly identity between Ishijima and claims 1-3”，有合理可能被预见；Blue Origin 随后自请 adverse judgment，claims 1–13 全部撤销（IPR2014-01376）。
> - **启示**：
>   - Claim 写到功能层面越宽，越容易撞上某篇冷门论文。
>   - 学术论文是最常见、也最便宜的 prior art；IPR 让竞争者不用等被起诉就先动手。
>   - SpaceX 后来改用发射塔机械臂在空中“夹住”助推器；2026-09-28 的 Starship 第 14 次试飞为保护发射塔又改回了水上降落。

> [!EXAMPLE]+ 花两年做出来，发现别人早有专利
>
> - **来源说明**：课堂录音转写把这个装置记成“兼具日常和防身用途的机械装置”，具体名称听不清；课件第 88–91 页的图和专利文件明确是下面这个手机保护壳，以课件为准。
> - **事实**（CS202 第一讲）：几位德国发明人花了大约两年，做出一种手机“气囊”保护壳：检测到手机跌落时，壳的四角弹出弹性支脚，先于手机着地、吸收冲击。2019 年在德国申请专利。
> - **问题一：不新**。美国早有一件 US 7,059,182（Ragner，2006 年授权，“Active Impact Protection System”），描述的正是检测自由落体、向外伸出缓冲件。在先公开破坏 novelty，他们拿不到有效专利。
> - **问题二：可能侵权**。该专利的 Claim 10 是一种方法：检测便携设备处于危险的自由落体状态 → 从外壳向外伸出缓冲装置，使其先于设备其他部分接触地面并吸收冲击。在先专利若仍有效、又覆盖他们销售的地区，制造销售就可能侵权；独立研发不是专利侵权的抗辩（见侵权一节）。
> - **启示**：版权侵权要证明复制，专利不要求被告见过别人的发明。动手之前先查 Google Patents、Espacenet、USPTO，比做完再查便宜得多。

<br>

## 显而易见

显而易见性与新颖性的区别在于，它允许在有理由组合时综合多份现有技术：

- **普通技术水平**：界定 PHOSITA 的教育、经验和领域背景。
- **组合理由**：说明为什么当时会组合这些文献，以及是否有合理成功预期。
- **客观辅助因素**：再用商业成功、长期需求、他人失败等证据校验事后偏见。

§ 103。没有一份 prior art 完整公开这项发明，仍可能对 PHOSITA 显而易见而无效。

- **PHOSITA**：person having ordinary skill in the art（本领域普通技术人员），法律假想的人，不是陪审员，也不是天才研究员。法院看教育背景、经验、技术复杂度和行业发展速度。
- **组合要讲理由**：§ 103 允许组合多份现有技术，但要解释普通技术人员当时为什么有动机这样组合、是否有合理的成功预期；不能知道答案后倒推，即 hindsight bias（事后诸葛亮）。

- **客观辅助因素**（secondary considerations / objective indicia）：商业成功、长期未解决的需求、他人失败、行业赞誉、被模仿等。必须与 claim 的技术特征有 nexus（关联），否则销量可能只是营销的结果，证明力很低。

- ***Graham v. John Deere*, 383 U.S. 1 (1966)** 的四步：
  1. 现有技术的范围与内容；
  2. 现有技术与 claim 的差异；
  3. 相关领域的普通技术水平；
  4. 客观辅助因素。

源头是 *Hotchkiss v. Greenwood*（1851）：发明须体现超出 “an ordinary mechanic acquainted with the business” 的技巧与创造力。现有技术的显而易见变体属于公共领域。

- ***KSR v. Teleflex*, 550 U.S. 398 (2007)**：
  - 否定僵硬的 teaching-suggestion-motivation 要求：不必在文献里找到“明说可以组合”的句子。
  - PHOSITA 被假定有普通的创造力；已知元件的常规组合可以是显而易见的。
  - 对 ML 专利尤其要紧：attention、residual、某种 tokenizer 的组合若只有可预见的效果，可能因 § 103 无效。关键是组合产生的非显而易见的技术效果，不是模块清单。

- **辅助因素不是走过场**：长期需求、他人反复失败、与发明相关的商业成功，可以反证并非显而易见；成功若来自广告、定价或 claim 以外的功能，证明力有限。下面的 Apple 案就是这一点的完整示范。

### Apple v. Samsung：滑动解锁

> [!INFO]+ 专利与程序
>
> - **专利**：Apple 的 US 8,046,721，“Unlocking a device by performing gestures on an unlock image”。要解决的问题是触屏误触：口袋里误拨电话、误发邮件。
> - **Claim 7**：带触屏、存储器、处理器的便携设备；在预定位置检测到对 unlock image 的接触，随接触持续移动该图像，图像从起点移到预定解锁区域时解锁。
> - **Claim 8**：Claim 7 + 显示视觉提示，告诉用户解锁要往哪个方向拖。Apple 专家用 Galaxy Nexus 上移动的 chevron 动画对应这一要素。
> - **陪审团（2014-05）**：Samsung 侵犯 Claim 8，且 Claim 8 不显而易见；Apple 请求 22 亿美元，判 1.196 亿美元。
> - **合议庭（2016-02-26）**：三人合议庭推翻，认为 “no reasonable jury” 能认定其不显而易见。
> - **全院重审（2016-10-07）**：11 名法官中 8 人恢复陪审团裁决，3 人异议。
> - **最高法院**：2017-11-06 拒绝 Samsung 的 certiorari。

- **Prior art**：
  - Neonode N1（瑞典，2005）已有 slide-to-unlock。
  - Plaisant & Wallace（1992）比较墙挂式控制器上的六种触屏开关，其中一种是指示方向的 slider toggle。15 名本科生把它排在第 5，论文还说滑动比点击更复杂、更难实现。
- **争点只剩组合动机**：Apple 承认两份文献合起来覆盖 Claim 8 全部要素，争的是手机设计者会不会去组合它们。
  - Samsung 专家：组合是 “natural”、“very routine”。
  - Apple 专家：做手机解决误拨的人，不会去翻空调控制面板的论文；何况 Plaisant 说滑块不受欢迎，属于 **teaching away**。
- **全院多数的理由**：
  - 文献教导什么、技术人员有没有组合动机，是**事实问题**，归陪审团。
  - 上诉审只问有没有 substantial evidence 支持陪审团；即使也有证据支持 Samsung，结论也不变。
- **客观辅助因素**（多数称其常是 “most probative and cogent evidence”，用来防止 hindsight）：
  - **行业赞誉**：Jobs 发布会上演示滑动解锁时观众欢呼；Samsung 内部文件称赞该功能。
  - **复制**：Samsung 内部报告反复写“改进方向是让解锁机制像 iPhone 一样”，Samsung 也承认复制了该功能。
  - **商业成功**：必须与 claim 有 nexus，看的是滑动解锁这个功能，不是整部 iPhone。证据包括：消费者没有该功能就不太愿买或愿付更少；它是第一支 iPhone 电视广告里展示的第一个功能。
  - **长期需求**：iPhone 之前对误拨问题只有 “frustrating solutions”，连 Samsung 自己都没解决。

> [!TIP]+ 这件案子真正说明的
>
> - § 103 的胜负常常不在“文献里有没有”，而在谁来认定事实、上诉审查多宽。同一组证据，合议庭和全院得出相反结论。
> - 被告自己的内部文件是对 Samsung 最致命的证据。工程团队写“照着竞品改”的文档，诉讼里会被当成复制和不显而易见的证据。
> - 专利有地域性：Samsung 事后称该专利 “has been invalidated by courts around the world”，在美国照样有效。

<br>

## 权利要求

### 披露要求

§ 112 中几个要求需要分别判断：

- **Written description**：说明书是否表明申请人在申请时已经掌握所主张的发明。
- **Enablement**：是否足以使 PHOSITA 在不过度试验的情况下实施 claim 的完整范围。*Amgen v. Sanofi*（2023）说明，主张很宽的功能性属概念（functional genus），却只教会少量实施例，可能无法支持整个范围。
- **Definiteness**：结合说明书和审查历史，claim 是否以合理确定性告知本领域人员权利边界；这是 *Nautilus v. Biosig*（2014）的标准。
- **Means-plus-function**：§ 112(f) 允许用 “means for …” 表述功能，但范围会联系到说明书披露的对应结构及其等同物。软件 claim 若只写功能、说明书却没有算法或结构，范围可能很窄，甚至不明确而无效。
- **Best mode**：§ 112(a) 要求写出发明人设想的最佳实施方式。AIA 之后未披露 best mode 不再是无效或不可执行的理由（§ 282(b)(3)(A)），但仍是法定写作义务。
- **§ 112(b) 的写法要求**：说明书以一项或多项 claim 结尾，“particularly pointing out and distinctly claiming” 发明人认为是其发明的主题。

- **专利的社会契约在披露**：说明书既用来解释 claim，又独立承担 written description 和 enablement 义务。
  - 写得很玄、实施不了 → 无效。
  - 只披露少量具体实现，却想用 claim 覆盖所有神经网络 → 也可能无效。

- ***O'Reilly v. Morse*, 56 U.S. 62 (1853)**：Morse 的 Claim 8 想垄断一切用电磁力远距离通信的方式，远超出他实际披露的电报装置，上位 claim 没有对应的披露。今天 “a neural network configured to diagnose…” 这类写得极宽、说明书撑不住的 claim，问题结构完全一样。

### 范围结构

- **边界由 claim 划定**，不是由说明书讲的故事划定；说明书、附图、实施例用来解释 claim。

- **claim 的类型**：装置、方法、系统、计算机可读介质。每项 claim 是一组 limitations（限制条件）的组合。
- **独立与从属**：
  - independent claim（独立权利要求）自己写齐全部要素，范围最宽。
  - dependent claim（从属权利要求）引用前一项再加限制，范围一定更窄，但更不容易被无效。
- **没写进 claim 的不算**：说明书里的漂亮功能如果没写进 claim，通常不能单靠摘要或背景段落去主张侵权。

- **写法**：claim 界定 “metes and bounds”，像地契划出地界；每项 claim 只能是一句话。
- **PageRank 的例子**：
  - Claim 1：计算机实现的为一组链接文档打分的方法：取得文档 → 按链接它的文档的分数给每个文档赋分 → 按分数处理文档。
  - Claim 2：赋分时按链接数确定权重因子。Claim 3：权重因子取决于链接文档被访问的概率估计。每加一层限制，范围窄一层，也多一道防线。
- **同一发明，四种 claim 形式**（Apple '721）：
  - Claim 1 方法；Claim 7 设备；Claim 11 means-plus-function；Claim 12 计算机可读存储介质。
  - Claim 13–15 分别挂在 1、7、12 之下，只加“unlock image 是单一图像”。
  - 这样写是为了覆盖不同侵权人：执行方法的用户、卖设备的厂商、分发软件的开发者。

### 排除权与阻挡专利

- **负向权利**：专利不授予实施发明的权利，只授予在美国、在期限内排除他人实施的权利。
- **阻挡专利**（blocking patent）：

| 权利人 | Claim |
| :--- | :--- |
| Smith | 一种凳子 |
| Jones（改进） | 一种带靠背的凳子 |
| Brown（改进） | 一种带摇杆的凳子；一种带靠背和摇杆的凳子 |

- Jones 造带靠背的凳子，要 Smith 的许可；Brown 造带靠背和摇杆的凳子，要 Smith 和 Jones 两份许可。Smith 反过来也不能造带靠背的凳子而不找 Jones。
- 改进专利因此常成为交叉许可的筹码；拥有专利和能够合法上市是两个问题，上市前要做 freedom-to-operate 检索。

### 解释与比对

- **先解释，再比对**：claim construction 先决定词是什么意思，再把被控产品逐项映射到 limitations。
- **证据顺序**：intrinsic evidence（内部证据）——claim 本身、说明书、审查历史——优先；字典、专家等 extrinsic evidence（外部证据）其次。

- **字面侵权（literal infringement）**：被控产品满足**全部** limitations（all-elements rule）。少一个就不构成字面侵权。
- **等同原则**（doctrine of equivalents）：防止改一个无关紧要的零件就绕开。*Warner-Jenkinson*（1997）要求逐个元件比对等同，不能拿整体相似代替。
- **审查历史禁止反悔**（prosecution history estoppel）：审查时为避开 prior art 而缩小的范围，诉讼里不能再用等同原则要回来。所以申请时怎么改 claim、怎么争辩，诉讼里都会被翻出来。

- ***Markman v. Westview Instruments*, 517 U.S. 370 (1996)**：claim 解释由法官做，不交陪审团。专利诉讼的前半场常在 Markman hearing 上就定了胜负：一个词在说明书里出现几次、有没有定义、审查时怎么用，都会被拿来讨论。

- **规则来源**：Federal Circuit 1995 年全院判决 *Markman* 先确立 claim 解释是法律问题，最高法院 1996 年维持。
- **基准**：术语按本领域普通技术人员（POSITA）在发明时的 “ordinary and customary meaning” 理解。
- **说明书最重要**：*Phillips v. AWH*（Fed. Cir. 2005, en banc）引 *Vitronics*：说明书 “usually … is dispositive; it is the single best guide to the meaning of a disputed term”。
- **被告的两条路**：争取一种解释，要么让 claim 宽到被 prior art 覆盖而无效，要么窄到产品落在外面而不侵权。宽窄两头都要防，这就是 claim construction 成为核心战场的原因。

> [!EXAMPLE]+ Neonode：同一个限制，先救了专利，又杀了专利
>
> - **专利**：US 8,095,879，发明人 Magnus Goertz，手持设备的触屏用户界面。Claim 1 四个要素：
>   - [a] 触敏区域中提供某一功能的表示；
>   - [b] 该表示只有一种激活方式；
>   - [c] 多步操作激活：物体触碰该表示所在位置，然后沿触敏区域 “gliding … away” 离开触点；
>   - [d] 滑动过程中，该表示不被移动或复制（not relocated or duplicated）。
> - **起诉**：2020-06-06 在 W.D. Tex. 起诉 Apple、Samsung，主张 swipe-to-unlock 和滑动输入侵权。讽刺的是，Apple 和 Samsung 在 '721 案里都承认 Neonode 的滑动解锁在先。
> - **IPR**：Samsung、Apple 2020-11 提 IPR，德州案中止；Google 未被起诉，2021-06 也提了 IPR。
> - **对 Google 的 IPR（PTAB 2023-01-11，Neonode 胜）**：争点是 Robertson 文献里用笔 “flick right” 拨号算不算 “gliding”。
>   - 审查历史：Neonode 审查时把 claim 从任何 “movement” 缩成 “glide”；“glide”“swipe”都不在说明书里；提交 N2 手机视频示范；限制 [d] 是为区别现有的 drag-and-drop，按审查员建议加入的。PTAB 认为 claim 意在覆盖今天所说的 swipe。
>   - 字典：flick 是轻、快、突然的动作；glide 是平滑、连续的动作。
>   - 行业用法：Apple 的界面规范和 Android 4.0 都把 flick 与 swipe 分开。
>   - 专家：Robertson 系统靠更快、更短来识别 flick；如果 flick 和 glide 一样，它就无法区分 flick 与 drag-and-drop。
> - **对 Samsung/Apple 的 IPR（PTAB 2022-12-15，Neonode 胜）**：Hirayama 拖动拨号图标时，图标是否被移动或复制（限制 [d]）。PTAB 认为证据接近，但当时的行业惯例就是拖动时移动或复制图标，Samsung 专家不可信；未达优势证据。
> - **股价**：PTAB 驳回 Google 当天（2023-01-11），Neonode 股价涨近 50%；Federal Circuit 2024 年 7 月维持之后，同年 9 月的报道称其股价半年内已涨 500% 以上，分析师开始估算对 Samsung、Apple 的诉讼价值。专利诉讼的每个程序节点都直接反映在小公司的市值上。
> - **Federal Circuit（2024-07-18）**：两件都维持。Samsung 2025-09 和解（据报 4000 万美元以上），专利此时已转让（assignment）给 Aequitas 旗下公司。
> - **N.D. Cal.（2026-06-08，Chen 法官）**：Apple 在地区法院改打 § 112(a) written description——IPR 里法定不能提的理由。法院认定说明书对限制 [d] 只字未提，也不必然披露；Apple 以 clear and convincing evidence 证明，Claim 1 无效。它是唯一被主张的独立 claim，案件终结。Neonode 已上诉。
>
> 启示：
>
> - 为绕开 prior art 在审查中加的限制，会在三个地方回头咬人：prosecution history estoppel 限制等同侵权；claim 解释被收窄；说明书没写的限制还可能因 written description 无效。
> - “not relocated or duplicated” 这种否定式限制尤其危险：说明书很少专门写“某事没有发生”。
> - 被告的组合打法：IPR 打 §§ 102 / 103，地区法院保留 § 112。两个程序的标准和可用理由不同，专利要同时扛住两边。

> [!TIP]+ 15.628J 的检索作业
>
> - **任务**：拿一份刚授权的美国专利，只盯 Claim 1，四小时内找可能破坏 § 102 / § 103 的 prior art，写 2–3 页。
> - **规则**：日期必须按 prior art 的定义来，不能拿申请日之后的文献打新颖性。
> - **换成 AI 专利做一遍**：CV / LLM / agent 专利的摘要往往写得很未来，真正卡住的是某一句很具体的系统步骤。摘要几乎没有法律意义。

<br>

## 侵权

### 抗辩结构

被告通常同时走几条路：

| 抗辩 | 主张什么 | 要点 |
| :--- | :--- | :--- |
| 无效 | 专利不满足 § 101 / 102 / 103 / 112 | 法院要 clear and convincing evidence；IPR 只能打 102 / 103 |
| 不侵权 | 即使专利有效，产品没有满足全部 limitations | 常和 claim 解释一起打 |
| 许可、用尽 | 已获许可，或买的是专利权人售出的那一件 | 见下文 *Bowman* |
| inequitable conduct（不正当行为） | 申请人对 USPTO 有重大隐瞒或虚假陈述 | *Therasense v. Becton*（Fed. Cir. 2011）：通常要求 but-for materiality，加上明确的欺骗意图；成立则整件专利不可执行 |
| 迟延 | 权利人拖了太久才起诉 | *SCA Hygiene v. First Quality*（2017）：laches 不能挡 § 286 六年追溯期内的赔偿；equitable estoppel 等仍可能适用，迟延也可能影响部分衡平救济 |

### 直接侵权

35 U.S.C. § 271(a)：未经许可，在美国制造、使用、许诺销售、销售或进口专利发明。

- **产品 / 系统 claim**：制造、使用、销售、许诺销售或进口被 claim 覆盖的产品。
- **方法 claim**：原则上必须有人实施**全部**步骤。
- **不看你是否抄袭**：不要求被告抄袭，也不要求事先看过专利，独立研发不是抗辩。但是否知情，会影响次级侵权、故意侵权和增强赔偿。

### 间接侵权（次级侵权）

专利法把次级责任（secondary liability）直接写进了条文，两条路：

| | 诱导侵权（induced infringement） § 271(b) | 帮助侵权（contributory infringement） § 271(c) |
| :--- | :--- | :--- |
| 条文 | “actively induces infringement” | 销售、许诺销售或进口专利发明的关键部件 |
| 行为 | 指导、鼓励、推广他人去实施专利（说明书、广告、技术支持） | 提供一个部件，这个部件“especially made or adapted”用于侵权 |
| 主观要件 | 知道专利，并知道被诱导的行为构成侵权 | 知道该部件专门用于侵犯某项专利（*Aro Mfg. v. Convertible Top*, 1964） |
| 关键排除 | — | 如果是有 substantial non-infringing use 的通用商品（staple article），不构成 |
| 前提 | 必须存在某人的直接侵权（direct infringement） | 必须存在某人的直接侵权 |

- **“知道”的门槛**：
  - *Global-Tech v. SEB*, 563 U.S. 754 (2011)：诱导侵权要求知道专利、知道构成侵权；故意视而不见（willful blindness）也算知道。案中被告复制了对手的炸锅，请律师做检索时故意不告诉对方这是照着谁抄的。
  - *Commil v. Cisco*, 575 U.S. 632 (2015)：被告真诚相信专利**无效**，不能免除诱导责任；真诚相信**不侵权**则可以。
- **必须有直接侵权**：*Limelight v. Akamai*, 572 U.S. 915 (2014)：方法的各步骤分别由被告和它的客户完成、没有任何一方实施全部步骤时，不能直接判诱导侵权。
- **分工执行的方法**（divided infringement）：Federal Circuit 发回后（*Akamai v. Limelight*, 2015 en banc）放宽：一方指导或控制了另一方的步骤，或双方是共同企业，或一方把自己的利益或服务条件建立在对方执行某些步骤上，全部步骤可以归于这一方，构成直接侵权。
- **和版权的关系**：版权的两条重要规则是从专利法借来的。
  - *Sony v. Universal* 的“大量合法用途”标准，借自 § 271(c) 的 staple article 排除。
  - *Grokster* 的诱导侵权，借自 § 271(b)。
- **对软件和 AI**：
  - 云服务或 SDK 由客户执行部分步骤、自己执行其余步骤时，先过 *Akamai* 的“指导或控制”测试。
  - 文档、教程、示例代码教用户按专利方法使用产品，是诱导侵权最常见的证据。
  - 收到侵权通知后继续推广，“知道”这一要件就容易满足；所以专利警告函常常是为后续诱导和故意侵权主张做铺垫。

### 例外与用尽

- **实验使用例外极窄**：*Roche v. Bolar*, 733 F.2d 858 (Fed. Cir. 1984)，仿制药为申报做的实验仍是侵权；国会随后用 Hatch-Waxman 法开出专门的 Bolar exemption。
  - “做研究所以不侵权”不是美国专利法的默认规则。
  - 版权的 fair use 和专利的 experimental use 是两套规则，不能串用。

- **用尽只管售出的那一件**：*Bowman v. Monsanto*, 569 U.S. 847 (2013)，农民买了抗除草剂大豆，种一代可以；把收获的种子再种，是在“复制”专利发明，不行。
  - 对应到数字物：买到一份权重，不等于买到复制、再训练、再分发的权利，要看许可证怎么写。

<br>

## 救济

救济分成两个时间方向：

- **过去损失**：lost profits、reasonable royalty，以及严重故意侵权时可能适用的增强赔偿。
- **未来行为**：是否按 *eBay* 四要素签发禁令。

| 救济 | 解决什么 | 怎么算 |
| :--- | :--- | :--- |
| **reasonable royalty**（合理许可费） | 过去的损失，§ 284 的最低线 | 假设侵权开始前双方自愿谈判会同意的费率；*Georgia-Pacific* 列了一长串因素，实务靠假想谈判和专家证言 |
| **lost profits**（利润损失） | 过去的损失 | 还原“没有侵权时本可赚多少”；常用 *Panduit* 框架：有需求、没有可接受的替代品、有生产和销售能力、能算出利润 |
| **增强赔偿** | 严重故意侵权 | *Halo v. Pulse*（2016）：对 egregious willful infringement 可酌情提高到最多三倍，不是认定故意就自动翻倍 |
| **律师费** | 特殊案件 | § 285 |
| **禁令** | 未来的行为 | 按 *eBay* 四要素决定，见下 |
| **ongoing royalty** | 不发禁令时，判决后继续实施的对价 | 法院确定 |

- **禁令不再自动**：*eBay v. MercExchange*, 547 U.S. 388 (2006) 废掉“胜诉就发禁令”的惯例，回到普通法四要素：
  1. 不可挽回的损害；
  2. 金钱赔偿不足以弥补；
  3. 双方困难程度的权衡；
  4. 公共利益。
- **影响**：NPE 靠禁令威胁换许可的空间缩小；自己生产产品的权利人仍较容易拿到禁令。15.628J Short Exercise 3 就是拆这四个要素。

- **和版权赔偿的对比**：对生成模型公司，版权的法定赔偿（statutory damages）按作品件数相乘，往往比专利许可费更早变成难以承受的敞口。*Bartz* 能和解到 15 亿美元的量级，背景是这个乘法，而不只是 Alsup 那句 *spectacularly so*。

<br>

# 商业秘密

> **核心问题**：不公开的配方、代码、数据，凭什么受法律保护？员工跳槽带走的东西，哪些算偷？

本章按商业秘密从成立、被不当取得到人员流动和救济的顺序展开：

- **成立要件**：秘密性、因秘密而产生的经济价值、合理保密措施。
- **经典样本**：可口可乐、李施德林、巨无霸酱、Google 搜索算法与开放权重（open-weight）。
- **不当取得**：区分 improper means、独立发现、合法反向工程（reverse engineering）和一般技能。
- **保密措施**：NDA、权限、日志、离职回收与公开权重。
- **人员流动**：以前雇员携带文件和知识为中心的民事、刑事路径。
- **救济与竞合**：禁令（injunction）、损害赔偿（damages），以及版权（copyright）、合同和商业秘密请求的并行。

## 成立要件

- **法源**：州法多采 Uniform Trade Secrets Act（UTSA）；联邦 2016 年 Defend Trade Secrets Act（DTSA）结构相近。
- **三项条件**，而且要**持续**满足：
  1. **秘密性**：没有被公众或能从中获利的人普遍知道，也不容易用正当手段查到。
  2. **经济价值**：正因为是秘密才有独立的经济价值。
  3. **合理保密措施**：权利人采取了在当时情境下合理的保密措施。
- **可以是什么**：公式、源代码、客户名单、训练数据（training data）、负样本、系统提示，甚至失败的实验。
- **不要求绝对保密**，也不要求军事级安保；但至少要说得清谁能访问、有没有 NDA、仓库权限、日志、离职回收和对外披露控制。
- **会削弱秘密性的动作**：发表论文、提交到公开 GitHub、没有保密限制的演示。

## 经典样本

- **Coca-Cola 配方**：
  - 1886 年 Pemberton 发明，直到 1919 年为收购公司融资才第一次写成文字，抵押在纽约银行金库，1925 年还清贷款后移到亚特兰大银行保存 86 年，后来移入 World of Coca-Cola 的展示金库。
  - 2006 年一名员工和两名同伙想把配方卖给 Pepsi；Pepsi 通知 Coca-Cola，三人被捕。
  - 秘密可以无限期，前提是一直保住；竞争对手拒收、举报，也是行业规范的一部分。
- **Listerine 配方**：1880 年代 Lawrence 发明，把秘密配方许可（license）给 Lambert。配方后来公开，Warner-Lambert 起诉要求停付许可费，败诉：合同约定的是使用配方就付费，不以配方保密为条件。许可费义务能否随秘密消失而终止，取决于合同怎么写。
- **Big Mac 酱**：1968 年全国推出，1980 年代换成便宜配方后，原配方在公司内部“丢了”。2004 年回归的前高管 Fred Turner 在档案里找到记录，重新找回 36 年前的供应商。保密不等于没有内部记录：秘密也会被权利人自己弄丢。
- **Google 搜索算法**：公开排序考虑的因素（查询词、页面相关性与可用性、来源专业性、位置和设置）以及“权重随查询类型变化”的原则，但不公开具体算法和权重，且持续调整。公开一般原则、保留具体参数，是商业秘密与用户信任之间的常见折中。
- **开放权重**：模型免费并公开权重，意味着放弃对权重本身的商业秘密保护。OpenAI 等公司通常把权重当作 “secret sauce”。

## 不当取得

- **Misappropriation（盗用）**：以不正当手段获取，或违反义务披露、使用。
- **不正当手段**（improper means）：盗窃、贿赂、欺骗、违反保密义务、电子入侵等。
- **正当手段**：观察公开产品、独立研发、合法取得实物后反向工程。
- **最常见的争点：员工跳槽**。前雇员脑子里带走的，哪些是一般技能，哪些是雇主的秘密？
  - 一般技能和经验不能都变成前雇主的永久财产。
  - 诉讼要具体指出秘密是什么，不能只说“整套业务知识”。

## 保密措施

- **NDA**（non-disclosure agreement，保密协议）约定当事人怎样使用、披露信息。
  - 合同写了 “confidential”，不会自动让内容满足商业秘密的法定条件。
  - 没有 NDA 也不必然排除商业秘密；法院综合看信息性质、双方关系和实际保密措施。
- **措施必须真的执行**：权限、NDA、访问日志、生产权重的存储控制。不必完美，但完全没有就很难主张。
- **开放权重**：
  - 向公众无保密限制地发布后，公开权重本身的秘密性基本结束。
  - 只向负有保密义务的有限对象提供受控访问，不一定立即失去秘密性。
  - 未公开的数据、后训练流程和产品层信息仍可能各自构成秘密。

## 人员流动

- ***Wexler v. Greenberg*, 399 Pa. 569 (1960)**：化学配方类案件。若把“在这家公司做过”本身当成秘密，雇佣市场就会停摆；一般技能、经验、记忆要切出去。
- **标准事实模式**：离职时带走硬盘、代码仓库、数据。模型实验室对应的是未公开权重、训练数据、配比、过滤规则、内部评测集、系统提示；已发表论文的算法、公开 benchmark、开源（open source）框架里的常规实现，通常不算。

> [!EXAMPLE]+ *Waymo v. Uber*：同一组事实，民事与刑事两条线
>
> - **事实**：Waymo 主张前工程师 Anthony Levandowski 离职前下载了 14,000 多份自动驾驶和 LiDAR 文件，随后创办 Otto；Uber 收购 Otto，并让他负责自动驾驶项目。Stanford CS202 把它列为员工流动的代表案。
> - **民事案**：2018 年陪审团（jury）审理第五天和解（settlement）。Waymo 得到 Uber 0.34% 股权，当时约值 2.45 亿美元；Uber 同意确保其软硬件不含 Waymo 机密信息。和解没有逐份认定哪些文件是商业秘密，也不是法院认定 Uber 用了全部文件。
> - **刑事案**：Levandowski 不是民事案的被告（defendant），联邦检方另行起诉。2020 年他对一项商业秘密盗窃罪认罪，涉及一份 “Chauffeur Weekly Update”，被判 18 个月监禁。
> - **启示**：CS202 目录写 “civil and criminal trade secret theft”，指的是这组相连事实，不是一份判决同时处理了民刑责任。

## 救济与竞合

- **DTSA 的适用**：联邦民事诉因，不取代州法；适用于与州际或对外贸易中的产品、服务有关的商业秘密。许多案件同时主张州法和联邦法。
- **时效**：通常是发现或应当发现盗用后三年；持续的盗用原则上算一个请求。
- **救济**：禁令、实际损失、不当得利、合理许可费（reasonable royalty）；故意、恶意盗用可能有倍计赔偿和律师费。
- **ex parte seizure**（单方扣押）：极端情况下可用，门槛很高，不是普通取证工具。
- **举报人免责**：DTSA 对特定举报披露免责，并要求雇主在保密协议中告知；没告知会限制部分救济。
- **竞合**：模型抽取、蒸馏、访问内部 API，可能同时涉及版权、合同和商业秘密。同一串 HTTP 请求可以是多个诉因的事实基础。

<br>

# 商标

> **核心问题**：名字和 logo 为什么受保护？什么样的使用会让顾客认错来源？

本章沿“标志是否受保护—被告（defendant）使用是否造成混淆—功能性（functionality）是否排除保护—模型场景如何适用”的顺序展开：

- **来源识别**：商标保护的对象、使用要求、续展与通用化。
- **显著性与注册**：从 fanciful 到 generic 的强弱序列，以及注册的作用。
- **可注册性**：贬损、不道德与人名条款，以及言论自由的边界。
- **混淆与淡化**：侵权要件、多因素测试、成名商标保护、间接侵权（次级侵权）与戏仿（parody）。
- **功能性**：trade dress 不能把功能设计永久私有化。
- **模型场景**：输出内容与模型、公司命名分别触发什么问题，含 CHATGPT、OPENAI、io 的商标争议。

## 来源识别

- **保护对象**：source identifier（来源识别标志）。词、logo、包装、声音、颜色都可以，只要消费者能把它和某一商业来源联系起来。
- **问什么**：不问创意聪不聪明，只问消费者会不会以为来源有关联（Lanham Act）。
- **期限**：可以无限续展，前提是一直在用，并且没有变成商品的通用名称（generic name）。
- **通用化**（genericide）：Aspirin、Escalator 是美国法里常见的例子，名字变成了品类本身。Thermos 的法律和市场历史更复杂，适合当作风险提示，而不是全球统一结论。
- **和专利（patent）的分工**：功能性设计走专利，不能用商标把功能锁成永久。

## 显著性与注册

| 强度 | 类型 | 意思 | 能否保护 |
| :--- | :--- | :--- | :--- |
| 最强 | **fanciful**（臆造） | 原本不存在的词 | 直接受保护 |
| 强 | **arbitrary**（任意） | 常见词用在无关商品上，如用 Apple 卖电脑 | 直接受保护 |
| 中 | **suggestive**（暗示） | 要联想一下才想到商品特点 | 直接受保护 |
| 弱 | **descriptive**（描述） | 直接描述商品特点 | 要证明 secondary meaning：消费者已把它认作某家来源 |
| 无 | **generic**（通用） | 商品的通用名称 | 不能取得商标权 |

- **注册的作用**：美国的商标权以 use in commerce（在商业中使用）为基础；联邦注册带来程序和全国范围上的优势，但救不了通用或功能性标志。
- **标志会演变**：Stanford 早年用印第安人头像，后来改为红色 S 加红杉树。旧标志退出使用，权利也随之淡出。
- **Teddy Bear**：1902 年 Theodore Roosevelt 打猎时拒绝射杀被绑在树上的熊，漫画家画成 “Drawing the Line in Mississippi”。布鲁克林店主 Michtom 夫妇据此做出 “Teddy's Bear”，征得罗斯福同意使用他的名字，后来创立 Ideal Toy Company。至今泰迪熊相关专利众多、注册商标数以千计。继任者 Taft 的 “Billy Possum” 想复制这套打法，很快失败。名字能借来热度，但来源识别要靠市场长期接受。

## 可注册性

- **贬损条款被推翻**：*Matal v. Tam*（2017），亚裔乐队 The Slants 申请注册被拒，最高法院（Supreme Court）认定禁止 “disparaging” 商标违反第一修正案（First Amendment）。Washington Redskins 的注册曾因贬损被撤销，此后也失去依据。
- **不道德、丑闻条款被推翻**：*Iancu v. Brunetti*（2019），服装品牌 FUCT，理由相同：按观点歧视。
- **人名条款维持**：*Vidal v. Elster*（2024），申请人想为 T 恤注册 “TRUMP TOO SMALL”。§ 1052(c) 禁止未经本人同意注册在世者姓名，最高法院维持：该条不针对观点，且有长期历史传统。
- **区分**：拒绝注册不等于禁止使用；Elster 仍可以卖 T 恤，只是得不到联邦注册带来的优势。

## 混淆与淡化

- **侵权四步**：
  1. 原告（plaintiff）是否拥有有效标志，并且有优先权；
  2. 被告是否在商业中把相关符号**作为商标**使用；
  3. 这种使用是否可能造成消费者混淆；
  4. 抗辩（defense）和救济，例如 descriptive fair use（描述性使用）、nominative fair use（指名使用）。
- **描述 ≠ 当品牌**：为了描述产品属性使用一个普通词，和把这个词当品牌用，是两回事。
- **混淆的多因素**：不只是比两个 logo 像不像，还要看标志强度、商品或服务是否接近、销售渠道、购买者注意程度、实际混淆证据、被告意图。
  - *Lois Sportswear v. Levi Strauss*, 799 F.2d 867 (2d Cir. 1986)：牛仔裤后袋缝线构成的商业外观。15.628J 用它入门，不代表商标法只有混淆一条。
- **淡化**（dilution）：成名商标另有 blurring（弱化）和 tarnishment（丑化）保护，不必证明来源混淆。生成带 logo 的图，可能同时涉及混淆和淡化。

### 间接侵权（次级侵权）

商标法同样没有在条文里写次级责任（secondary liability），来自判例：

- **帮助侵权（contributory infringement）**（contributory trademark infringement）：*Inwood Laboratories v. Ives*, 456 U.S. 844 (1982)——
  - 故意诱导他人侵犯商标（intentionally induces）；或
  - 继续向明知或有理由知道正在侵权的人供货（continues to supply … to one whom it knows or has reason to know is engaging in trademark infringement）。
- **平台**：*Tiffany v. eBay*, 600 F.3d 93 (2d Cir. 2010)——eBay 一般性地知道站内有假冒 Tiffany，还不够；要对**具体**的侵权商品或卖家有知情，并且在收到通知后不处理。eBay 及时下架被举报商品，不承担帮助责任。
- **替代责任**（vicarious liability）：要求被告与直接侵权（direct infringement）人有实际或表面上的合伙、代理关系，能约束对方或与其共同控制侵权产品，比版权（copyright）的替代责任窄。
- **没有法定安全港（safe harbor）**：商标没有 DMCA § 512 那样的通知—删除安全港，§ 230 也明确不适用于知识产权（intellectual property）；平台靠 *Tiffany* 式的“收到具体通知就处理”降低风险。
- **对 AI**：模型生成带他人 logo 的图，开发者是否负次级商标责任，要看是否知道具体的侵权使用、是否继续提供服务，以及是否把生成某品牌的标志当卖点。

> [!EXAMPLE]+ 戏仿：*Jack Daniel's v. VIP Products*（2023）
>
> - **事实**：VIP 卖狗咬玩具 “Bad Spaniels”，瓶形和标签模仿 Jack Daniel's 威士忌，“Old No. 7” 变成 “Old No. 2”，“40% alc. by vol.” 变成 “43% poo by vol.”。
> - **判决**：被告把对方商标的仿作当作自己商品的来源标识时，不适用保护艺术表达的 *Rogers* 门槛测试，回到普通的混淆可能性分析；淡化中“非商业使用”的除外也不适用于作为商标的使用。
> - **含义**：戏仿可以是很好的言论，但一旦把它当品牌卖，就要像普通商标一样过混淆测试。

## 功能性

- **Trade dress（商业外观）**：保护产品或包装的整体样子，作为来源识别。
- **必须 nonfunctional**：若某设计主要影响成本、质量或用途，就是功能性的，不能借商标取得永久排他。功能该由专利保护，而专利有期限。

## 模型场景

- **输出层**：生成带 logo、角色外形、包装的图，可能同时涉及版权（角色、美术）和商标（来源混淆、假冒）。*Disney v. Midjourney* 主诉是版权；商标问题另看用户会不会误以为这是迪士尼官方的生成器。
- **命名层**：模型名、公司名本身的商标冲突是另一层，和训练无关，和起名有关。
- **CHATGPT（TTAB 2026-03-04）**：是否 “merely descriptive”，以及能否靠 acquired distinctiveness（后天获得显著性）注册。“GPT” 是技术名词，描述性风险正来自这里。
- **OPENAI（TTAB 2026-08-06）**：同样的问题。“open” 加 “AI” 越像对服务的描述，越要证明公众已把它认作某一家公司。
- ***OpenAI v. Open Artificial Intelligence*（N.D. Cal. 2025；9th Cir. 2026）**：争的是谁先使用 OPENAI 标志，即优先权。
- ***IYO v. IO Products*（9th Cir. 2025）**：OpenAI 收购 Jony Ive 的 IO Products 后，是否应被禁止在计划中的硬件产品上使用 “io”。
- **启示**：AI 公司的命名常取描述性或技术性词汇，注册难、冲突多。起名前做商标检索，比产品发布后改名便宜得多。

<br>

# 归属

> **核心问题**：作品或发明有了保护，到底归作者、发明人、学校还是公司？开源和许可证改变了什么？

本章先把“权利是否成立”和“权利归谁”分开，再处理实验室里最常见的三类归属问题：

- **权利分层**：作者、发明人、受让人（assignee）和署名者并非同一法律身份。
- **职务成果**：版权（copyright）的 work made for hire、专利（patent）转让、大学技术转移和共同权利。
- **人员流动**：竞业、保密、发明转让与禁止招揽分别限制什么。
- **开源许可**：软件、权重和合同义务的边界，以及“开放”的不同层级。

## 权利分层

- **两道题**：“作品或发明受不受保护”和“权利归谁”要分开回答。
- **顺序**：先确定作者或发明人，再看雇佣关系、合同、书面转让和法定规则，是否把权利移给了学校或公司。
- **不能互相代替**：论文署名、Git commit、项目负责人身份，都不等于法律上的 authorship（作者身份）、inventorship（发明人身份）或 assignment（转让）。

## 职务成果

### 版权归属

- **权属不靠口头默契。**
- **雇员**：在职务范围内创作，是法定的 work made for hire，雇主被视为作者。
- **委托作品**：只有落入 § 101 列举的九类、且有书面约定，才算 work made for hire。计算机程序本身不在这九类里。
- **外包人员、实习生、访问学生**：不属于普通法上的雇员时，他们写的代码通常要有 § 204 的书面转让，不能默认归实验室。

### 专利归属

- **没有自动归属**：*Stanford v. Roche*（2011）——Bayh-Dole 法不会自动把联邦资助的发明归大学，仍要靠发明人的书面转让。
- **措辞决定输赢**：斯坦福当年表格写的是 “agree to assign”（同意将来转让），Roche 拿到的是 “hereby assign”（现在即转让），后者在先生效，Roche 胜。
- **对实验室**：入职时签的 assignment 文本，措辞和签署时间会直接决定权属。

### 大学技术转移

> [!EXAMPLE]+ PageRank：Stanford 的另一端样本
>
> - **背景**：Larry Page 和 Sergey Brin 在斯坦福读博时做搜索引擎，用网页之间的链接结构给网页排序。
> - **专利**：US 6,285,999（1998 年 1 月申请，2001 年 9 月授权），权利人是 Stanford，发明人是 Page。
> - **许可**：Stanford 独占许可（exclusive license）给 Google，换来约 180 万股，2005 年前后全部卖出，进账约 3.36 亿美元，是 Stanford 当时最大的一笔许可收入。
> - **路径**：研究 → 发明披露 → 专利申请 → 许可 → 商业产品，中间几步由斯坦福 OTL（Office of Technology Licensing）负责。独占的范围以合同为准，常见限制是领域、地域和学校保留的研究使用权。
> - **专利价值 ≠ 业务价值**：Google 的数据、工程和规模都不在这份专利里；相关专利 2019 年前后已经到期。

- **同一时期的另一面**：Yahoo 旗下的 Overture 用竞价广告排序专利（US 6,269,361）起诉 Google，2004 年 8 月 Google 上市前和解（settlement），向 Yahoo 发行 270 万股。搜索公司的起点，一头靠大学专利许可，一头为别人的专利付了股票。

### 共同权利

| | 共同发明（专利） | 共同作品（版权） |
| :--- | :--- | :--- |
| 谁算 | 对至少一项 claim 的 conception 作出实质贡献的自然人 | 各方有意把贡献合并成 “inseparable or interdependent parts of a unitary whole”，且每人都有可版权的贡献 |
| 份额 | 每个共有人对整件专利享有不可分割的权利 | 默认均分，贡献不必相等 |
| 能否单独许可 | 可以自行实施、许可（35 U.S.C. § 262） | 可以发放非独占许可 |
| 要不要分钱 | 原则上不必向其他共有人结算 | 要向其他共同作者分享许可费 |
| 能否约定改变 | 可以 | 可以 |

- **漏列、错列发明人**：影响权属、诉讼资格和专利有效性，但不一定毁掉专利；35 U.S.C. § 256 允许在符合条件时更正，是否涉及欺骗会影响后果。
- **对不齐很正常**：论文作者列表和发明人列表常常不同，专利仍须逐项 claim 找出真正的贡献者。

## 人员流动与合同

| 条款 | 限制什么 | 加州的态度 |
| :--- | :--- | :--- |
| **Non-compete**（竞业限制） | 离职后为竞争者工作 | Business and Professions Code § 16600 基本不承认；人才流动被当成创新政策 |
| **NDA**（保密协议） | 披露或使用约定的机密信息 | 有效，但不能把公开信息、一般技能或依法受保护的举报变成秘密 |
| **Invention assignment**（发明转让） | 约定范围内的职务发明或作品归雇主 | 有效，范围受法定限制 |
| **Non-solicit**（禁止招揽） | 带走客户或员工 | 客户招揽限制同样可能受 § 16600 约束 |

- **联邦层面**：FTC 2024 年发布全国性竞业禁止规则，但被联邦地区法院（district court）撤销、阻止生效，可执行性仍主要回到州法。
- **实际会打什么**：去相邻实验室继续做同一条产品线，真正会打的往往是带走的文件、数据和 NDA，而不是竞业条款本身。

## 开源许可

### 权利对象

- **管的东西不同**：GPL、MIT、Apache 2.0 管的是软件作品；模型许可是另一份法律文本。
- **GPL 何时触发**：通常在复制、修改或传播 covered work，或传播基于它形成的作品时；不能简化成“只要源码衍生就触发”。
- **权重算不算衍生作品**：权重、LoRA、checkpoint 是否构成 derivative work，至少要先问其中是否包含或改编了可识别的受保护表达、基于哪一部作品。目前几乎没有判决。
- **常见误读**：把“模型开源”理解成“和 Linux 一样”，容易读错权利范围。
- **Creative Commons**：主要处理版权，不自动提供专利许可。是否已授权训练，要看训练触及哪些版权行为、许可覆盖范围、具体版本和是否满足条件。模型或输出是否触发署名、相同方式共享等条件，法院还没系统回答。

### 转让、许可与合同

| 概念 | 意思 |
| :--- | :--- |
| **Assignment**（转让） | 把权利的所有权移给受让人 |
| **License**（许可） | 只允许被许可人在约定范围内做特定的事，权利仍归原权利人 |
| **Contract**（合同） | 当事人之间形成了合同关系，才有合同义务 |

- **许可证不是把版权送掉**，而是权利人预先允许特定行为，并附上条件。
  - MIT：主要要求保留版权和许可声明。
  - Apache 2.0：另有明确的专利许可和终止条款。
  - GPL：分发衍生作品时要按 GPL 提供对应源代码。
- **违约（breach of contract）还是侵权（infringement）**：违反条款构成版权侵权还是合同违约，要看它是许可的 condition（条件），还是独立的 covenant（承诺），以及合同是否成立。

### 开放层级

- **“Open source”有明确标准**（OSI 定义）。
- 只允许研究、限制用户规模、禁止竞争或禁止某些用途的模型许可，即使权重可以下载，通常也只是 source-available / open-weight，不等于开源软件许可。

<br>

# 信息政策

> **核心问题**：政府、平台、公司在言论、搜查、隐私和跨境数据上各受什么约束？

本章从政策分析方法进入六类公法和监管问题：

- **政策方法**：从技术事实、利益相关方和现行规则推导可执行建议。
- **言论**：国家行为、言论类型和审查强度。
- **平台**：§ 230、推荐与生成、政府干预和合成内容。
- **搜查与访问**：第四修正案（Fourth Amendment）、第三人数据、CFAA、网页抓取和代理访问。
- **隐私**：法域结构、核心概念与匿名化。
- **跨境**：连接点、实体法、法院管辖、承认执行和数据流动。

## 政策方法

- **互联网不是一件产品**：它是网络、市场、社交空间、信息系统和政治机构叠在一起。
- **Code is law**（Lessig）：架构先决定什么行为容易发生，法律再把责任和成本压回架构。
- **比知识产权（intellectual property）更宽**：不只问“谁拥有内容”，还问——
  - 政府能否限制言论；
  - 平台是否为用户内容负责；
  - 执法机关怎样取得数据；
  - 公司怎样收集个人信息；
  - 不同国家的规则怎样同时适用。
- **私法和公法会叠在一起**：平台按自己的规则删帖，是合同和言论治理问题；政府命令平台删帖，才直接出现 state action 与第一修正案（First Amendment）问题。
- **技术、行为、法律互相推动**：
  - 加密默认开启 → 执法机关换一种理由要后门。
  - 转发做成一键 → 诽谤和版权的间接责任理论被重写。
  - 推荐算法把“展示谁的内容”变成平台自己的编辑动作 → § 230 还管不管，成了 2020 年代的问题。

> [!TIP]+ 政策分析的六步
>
> 1. **技术事实**：系统实际收集、复制、生成或传输了什么。
> 2. **利益相关方**：用户、平台、权利人、政府与公众分别承担什么成本。
> 3. **现有规则**：私法、公法和行业监管怎样同时适用。
> 4. **备选方案**：立法、执法、合同、架构和市场机制各能解决什么。
> 5. **架构反馈**：方案会怎样改变权限、日志、过滤、推荐和数据保存。
> 6. **后果与建议**：比较表达自由、隐私、创新、竞争和执行成本，再提出可操作的选择。
>
> 建议要落到：谁执行、用什么工具、失败了会怎样。

<br>

## 言论

- **第一修正案主要约束政府**，不直接要求私人平台给人发言空间。
- **分析顺序**：
  1. 有没有 **state action（国家行为）**？私人公司按服务条款删帖，通常不是第一修正案问题；政府胁迫平台删帖才不同。
  2. 受限制的是哪种言论？
  3. 规则是按内容限制，还是内容中立？
  4. 适用什么审查强度？
- **审查强度**：
  - content-based restriction（基于内容的限制）通常适用 **strict scrutiny**：政府要证明有迫切利益，手段还要 narrowly tailored（严格限缩）。
  - 诈骗、真正的威胁、部分诽谤等类别保护较弱。
  - “网上内容令人不适”本身，不是把互联网降到广播监管强度的理由。
- ***Reno v. ACLU*, 521 U.S. 844 (1997)**：
  - 《通信规范法》（Communications Decency Act）里针对 indecent、patently offensive 网络表达的条款过于模糊、宽泛，被判无效。
  - 最高法院（Supreme Court）拒绝把互联网等同于受特殊监管的广播，给它高度的第一修正案保护。
  - 不是整部法律都被推翻：同一部法律里的 § 230 保留了下来。
  - 此后关于平台内容管制的讨论，起点都是“互联网表达受高度保护”，而不是套用广播监管标准。

<br>

## 平台

### 第 230 条

- **条文**：
  - § 230(c)(1)：交互式计算机服务不因用户发布的内容而被当成 publisher（发布者）或 speaker（发言者）。
  - § 230(c)(2)：善意审核内容，也有一个安全港（safe harbor）。
- **宽解释**：*Zeran v. AOL*（4th Cir. 1997）——即使平台已收到通知、知道某条帖子涉嫌诽谤，仍可能受保护。所以平台治理主要靠私人规则：服务条款、推荐和审核团队。
- **(c)(1) 拆成三问**：
  1. 被告（defendant）是不是 interactive computer service？
  2. 原告（plaintiff）是不是要把它当 publisher / speaker 追责？
  3. 相关信息是不是由 another information content provider（另一个内容提供者）提供的？平台自己负责创作违法内容的实质部分，那部分就不算纯用户内容。
- **不是普遍免责**：联邦刑法、知识产权、部分性交易法律有明确例外；知识产权例外是否包括州知识产权法，各巡回不完全一致。

| | § 230 | DMCA § 512 |
| :--- | :--- | :--- |
| 管什么 | 用户内容引起的州法责任，如诽谤 | 只管版权 |
| 条件 | 基本没有通知义务 | 要求 notice-and-takedown 等条件 |
| 收到通知后 | 仍可能受保护 | 不处理就可能失去安全港 |

### 推荐与生成

- **两头受限**：政府强迫平台删除或保留某些合法言论，可能违反第一修正案；政府把平台当成用户内容的发布者追责，又可能受 § 230 限制。争议常落在某个审核动作是私人决定还是 state action。
- ***Gonzalez v. Google*（2023）**：本可能回答推荐算法还受不受 § 230 保护。最高法院没接这道题，跟着 *Twitter v. Taamneh* 把责任问题绕开。推荐是不是平台自己的言论，仍悬着。
- ***Moody v. NetChoice*（2024）**：平台对信息流的选择、排序和呈现，可能是受第一修正案保护的编辑判断；但要针对具体功能和法律适用范围分析，不能把“整个平台”统一归类。它处理的是州政府限制平台审核的宪法问题，不回答 § 230 是否保护生成式输出。
- **生成式 AI 的难题**：
  - 模型输出多大程度是用户的言论、多大程度是公司的言论，直接决定 § 230 能不能用。
  - 用户只打了一句提示，输出由模型参数和公司的安全策略共同决定。这和推荐流里展示别人的推文不是一回事。
  - 2024 年以后的诉讼卡在这里，还没有 *Reno* 级别的终局答案。

### 合成内容

- **课程位置**：深度伪造（deepfake）、虚假信息，6.4590 放在 Week 8 内容审核和后来的 AI 周，不放在纯宪法课。技术能力先出现，归责规则随后形成。
- **两种路径**：
  - Copyright Office Part 1 建议为个人形象单独立法，而不是强行纳入版权法。
  - 中国采用标识义务，并禁止删除、篡改、隐匿标识。
  - 两者都说明：版权不是处理“这张脸像谁”的主要工具。
- **为什么版权管不了**：声音和长相不是 “original works of authorship”。2023 年 4 月流行的 AI 仿 Drake 与 The Weeknd 歌曲 “Heart On My Sleeve”，难点正在这里：可能没有复制任何具体录音或词曲。
- **联邦层面**：
  - TAKE IT DOWN Act（2025 年签署）：把未经同意传播私密图像、深度伪造色情内容规定为犯罪。
  - NO FAKES Act（2026 年提案，尚未通过）：拟创设联邦 IP 权利，禁止未经授权制作个人声音和视觉形象的 digital replica。
- **州层面**：
  - Tennessee ELVIS Act（2024）：最早把声音明确纳入肖像权保护。
  - California AB 2602（2024）：合同使用表演者声音、形象的 AI 数字复制品时，必须具体写明用途。
  - California AB 1836（2024）：未经同意，不得在影视、游戏、有声书、录音等作品中商业使用已故表演者的数字复制品。
  - California SB 1050（2026-09-16 签署）：广告使用 AI 生成的 “synthetic performers” 时须清楚披露。
- **结构**：声音、形象走肖像权（right of publicity）和专门立法；版权只在复制了具体录音、画面或歌词时介入。

<br>

## 搜查与访问

### 政府搜查

- **两套问题别混**：第四修正案管政府搜查扣押，不管公司的隐私政策。
- **先问是不是 search or seizure**：
  - 传统标准：是否侵入人身、房屋、文件、财物。
  - 后来的标准：是否侵犯 reasonable expectation of privacy（合理的隐私期待）。
- **构成搜查时**：通常要有基于 probable cause（相当理由）的 warrant（令状）；但同意、紧急情况、逮捕附带搜查等例外很多。
- **“数据在公司手里”**不自动回答政府调取要不要令状。

### 第三人数据

| 案件 | 事实 | 结论 |
| :--- | :--- | :--- |
| *Smith v. Maryland*（1979） | 电话号码交给电话公司，政府用 pen register 记录 | 不是搜查——“第三人原则（third-party doctrine）”：交给第三方的信息没有隐私期待 |
| *United States v. Jones*（2012） | 在车底装 GPS 长期追踪 | 构成搜查；多数意见基于物理侵入财产，长期监控理论主要见于协同意见 |
| *Carpenter v. United States*, 585 U.S. 296 (2018) | 政府取得约 127 天的 CSLI（基站位置记录） | 取得至少七天的历史 CSLI 是搜查，通常需要令状 |

- ***Carpenter* 的意义**：Roberts 写，数字时代第三人原则不能无限延伸。6.4590 Week 4 的问题是：自动交给手机公司的位置，是否还像 *Smith* 的号码那样算“自愿交付”。
- **没回答的**：更短的时间窗口；所有第三方数字记录的一般规则。agent 连续上报坐标落在哪一侧，目前没有同级别判决。
- **同一份数据，两次评价**：位置数据、云端对话日志、agent 浏览记录——
  - 公司依用户协议收集 → 合同、州隐私法、消费者保护法。
  - 政府向公司调取 → 第四修正案和 Stored Communications Act。

### 访问边界

- **CFAA**（Computer Fraud and Abuse Act，18 U.S.C. § 1030）：
  - 1986 年通过，起点是刑法，管未经授权访问计算机（针对 “hackers”）。
  - 1994 年加入 § 1030(g) 民事诉权：满足一年内损失合计至少 5,000 美元等条件，受害方可直接起诉。CFAA 因此成了公司之间打 scraping 的常用工具。
- **两个核心词**：
  - **without authorization**（未经授权）：根本没有获得进入某个系统的许可（license），却进去了。
  - **exceeds authorized access**（超越授权）：有权进入系统，但取得了自己无权取得的那部分信息。例如有某网站的账户，不等于能进它后台的所有数据库。
- ***Van Buren v. United States*, 593 U.S. 374 (2021)**：
  - 佐治亚州警察 Nathan Van Buren 在 FBI 设的局里收了约 5,000 美元，用执法车牌数据库替人查车牌。他有权访问这个数据库，只是目的不当。
  - 最高法院 6:3（Barrett）撤销定罪：采用 gates-up-or-down（门开着还是关着）的理解。有权进入的数据库，只是出于不当目的查询，不算超越授权。
  - 伪造账号、盗取凭据、绕过登录、继续突破明确的技术屏障，更接近未经授权。
  - 违反服务条款（ToS）可以是合同问题，但不能自动升级成联邦计算机犯罪。agent 用真实用户的合法账户办事，最先碰到的就是这条线。

> [!EXAMPLE]+ *hiQ Labs v. LinkedIn*：公开页面抓取
>
> - **事实**：hiQ 抓取任何浏览器都能看到的公开职业资料，做员工流失和技能分析。LinkedIn 发停止函（cease-and-desist），并用技术手段阻断。
> - **第九巡回（2022）**：*Van Buren* 发回后再次维持初步禁令（preliminary injunction）。hiQ 至少提出了严重问题：公开资料没有 CFAA 所要求的那道授权“门”，访问它一般不像 “without authorization”。
> - **不是对所有 scraping 的豁免**：
>   - 判决只在初步禁令阶段，没解决合同、trespass、隐私、版权等其他请求。
>   - 案件以 consent judgment 收尾：hiQ 同意 50 万美元判决、停止被禁止的抓取并删除数据和工具。和解（settlement）不推翻第九巡回对公开页面的分析，也不产生先例（precedent）。
>   - hiQ 用假账号进入密码保护页面的行为，和纯公开页面抓取要分开看。

- **登录后的数据是另一端**：
  - *Meta v. Bright Data*（N.D. Cal. 2024-01，Chen 法官）：Meta 的条款只约束登录用户；Bright Data 在登出状态抓公开页面不违约（breach of contract），Meta 随后撤诉。
  - *LinkedIn v. ProAPIs*（N.D. Cal. 2025-10 起诉）：对方每天注册成百上千个假账号，登录后抓会员、公司和学校数据（包括只在密码墙后才有的数据），向客户收每月最高 1.5 万美元。2026 年以 consent judgment 结束：禁止用假账号、机器人或其他自动化手段访问、出售 LinkedIn 数据，删除已抓数据。

> [!TIP]+ Agent 抓网页的三层
>
> | 情形 | CFAA | 其他风险 |
> | :--- | :--- | :--- |
> | 公开、无需登录 | 通常没有授权“门”，难构成 | 合同、版权、隐私、服务器负载 |
> | 违反网站条款，但没绕过访问门 | 不会仅因用途不当就成为犯罪 | 合同争议 |
> | 伪造账号、绕过登录或技术屏障 | 更接近 without authorization | 结合凭据、技术措施、收到通知后的行为分析 |
>
> - CFAA、合同、版权、隐私可能给出四个不同的答案。
> - robots.txt 不自动等于 CFAA 的授权门。
> - 漏洞报告可能同时是研究、违反服务条款、未经授权访问。

### 代理访问

> [!EXAMPLE]+ *Amazon.com Services v. Perplexity AI*：第一宗 agentic browser 的 CFAA 判决
>
> - **产品**：Perplexity 的 Comet 浏览器带 AI Assistant，用户可以让它在 Amazon 上搜索、比价、加购物车、下单。传统浏览器由用户自己点，agentic browser 由软件替用户点。
> - **起诉**：Amazon 多次要求停止未果，2025 年 11 月在 N.D. Cal. 起诉，诉因是 CFAA 和加州 CDAFA（Penal Code § 502），主张 Comet 把自己伪装成普通 Chrome、进入用户受密码保护的 Amazon 账户。Amazon 还屏蔽了 Meta、Google、OpenAI 的 agent，但没有起诉它们。
> - **商业动机**：agent 不看广告，也不看推荐位；Amazon 卖的不只是商品，还有用户的注意力和交互数据。动机和 CFAA 的构成要件（elements）仍要分开看。
> - **地区法院（2026-03，Chesney 法官）**：发初步禁令，禁止 Perplexity 用 Assistant 访问 Amazon 受密码保护的账户，并销毁已取得的数据。
> - **第九巡回（2026-08-04，No. 26-1444）**：撤销禁令。
>   - 条文是 *whoever … intentionally accesses* 一台受保护计算机。记录显示 Perplexity 的服务器从不直接和 Amazon 通信：Assistant 在用户设备上截图，发回 Perplexity 服务器拿指令，再在用户浏览器里执行。按这个架构，access Amazon 的是用户，而用户本来有权进自己的账户。
>   - 援引 rule of lenity（刑事条文有歧义时作有利被告的解释），因为 CFAA 也是刑法。其余禁令要素也偏向 Perplexity。
> - **范围有限**：只针对本案记录上的 access。AI 公司对 agent 控制更强，或自己的服务器直接向目标网站发请求，结论可能反过来。它也没处理侵权法（tort law）请求，没限制 Amazon 用服务条款管自动化访问。
> - **下一轮**：多半打合同——服务条款能不能约束代用户操作的 agent；用户点了同意，开发者受不受约束。

<br>

## 隐私

### 法域结构

| 法域 | 结构 | 要点 |
| :--- | :--- | :--- |
| 美国 | 没有统一的联邦隐私法，分行业、分州 | FTC Act § 5、HIPAA、FERPA、COPPA、加州 CCPA/CPRA、Illinois BIPA 等。HIPAA 只约束 covered entities、business associates 等法定主体，不是所有健康信息的一般隐私法 |
| 欧盟 | GDPR，综合权利框架 | 合法基础、目的限制、最小化、跨境传输、自动化决策 |
| 中国 | 网络安全法、数据安全法、个人信息保护法 | — |

- **训练带来的新问题**：训练把“处理”从保存文件扩展到更新参数。
  - 模型不是文件夹：删除一条原始语料，不等于参数和输出能力里不再有它的影响。
  - 但也不能预设个人数据一定能从参数里识别或恢复，要看具体模型和处理事实。

### 核心概念

- **personal data / personal information**：能直接或间接关联到自然人的信息，不只是姓名和身份证号。
- **controller / processor**：前者决定处理的目的和方式，后者受托处理；义务不同。
- **legal basis**：GDPR 下处理必须有同意、合同必要、法定义务、legitimate interests（正当利益）等合法基础之一。
- **purpose limitation**：为某个目的收集的数据，不能无条件改作完全不同的用途。
- **data minimization**：只处理完成目的所必要的数据。
- **deletion / erasure**：删除原始记录相对直接；数据是否已被模型记住、能否从输出恢复，是另一个技术问题。

### 匿名化

- **匿名化（anonymization）**：处理后无法合理地重新识别个人，可能不再是个人数据。
- **假名化（pseudonymization）**：只删姓名、换成编号，但结合其他数据仍能还原个人，通常仍是个人数据。

<br>

## 跨境

跨境问题先列连接点，再依次回答三个法律问题和一个数据合规问题：

- **连接点**：用户、服务器、公司、资产、权重、日志和微调数据分别在哪里。
- **实体法**：哪个国家或地区的法律调整这个行为。
- **法院管辖**：哪个法院能审这个被告和这个争议。
- **承认执行**：拿到判决后，能否在资产、数据或主体所在地实际执行。
- **数据流动**：跨境传输是否还要标准合同、充分性决定或安全评估。

### 连接点

- **分散是常态**：用户在一国，服务器在另一国，公司在第三国，权重、日志、微调数据分布多处。适用哪国法，直接决定同一套发布能否上线。
- **可能同时出现**：GDPR 的域外适用、中国针对向境内提供服务的规则、美国法院的长臂管辖。
- **出口管制另算**：先进芯片、部分封闭权重。
- 6.4590 把 trade policy 和 international affairs 单列，因为架构决策常同时受布鲁塞尔、华盛顿、北京的规则影响。同一套开源（open source）发布，在三个市场需要的合规动作不同。

### 跨境三问

- **第一问：实体法**
  - 哪个国家或地区的实体法适用于被诉行为。
  - 知识产权有属地性：分别定位复制、实施、销售、进口、展示或混淆发生在哪里。美国专利和商标（trademark）法原则上只调整与美国有足够联系的行为，公司在美国不等于覆盖它在全球的全部行为。
  - GDPR 域外适用、向境内用户提供服务等规则，还会根据用户、经营活动和目标市场建立连接。
- **第二问：法院管辖**
  - Personal jurisdiction（属人管辖）：被告与法院地有没有足够联系，法院能否对这个被告行使审判权。
  - Subject-matter jurisdiction（事项管辖）：这个法院有没有权审理这一类案件。
- **第三问：承认执行**
  - 一地法院的判决能否在另一地被承认。
  - 即使胜诉，也要看资产、数据和相关主体在哪里，当地是否提供实际的执行手段。

### 数据流动

- 数据传输可能单独需要标准合同条款、充分性决定或安全评估。
- 服务器放在哪里只是连接点之一，不会自动排除用户所在地和公司经营所在地的法律。

<br>

# 人工智能

> **核心问题**：用作品训练模型、模型生成内容、AI 参与创作和发明，各自适用哪些旧规则，结论到哪一步了？

本章把前述制度用于 AI，但仍以具体行为而不是“模型”作为分析单位：

- **分析框架**：拆行为、对象、主体、权利、抗辩（defense）、证据和程序阶段。
- **训练**：分别处理竞争用途、副本来源和市场损害。
- **输出**：判断复制、实质性相似、文本复述、角色与图像。
- **作者与发明人**：区分人的表达控制和对 claim 的 conception。
- **比较法**：比较 fair use、TDM 例外、合法访问、退出和透明度。

## 分析框架

> [!INFO]+ AI 与知识产权的三个法律层次（CS202 第一讲的总结）
>
> | 层次 | 问什么 |
> | :--- | :--- |
> | **Copyright Law**（版权法） | AI 训练是否复制了受保护作品？是否属于 fair use？开发者是否诱导用户侵权？ |
> | **Computer Access Law**（计算机访问法，主要是 CFAA） | 爬虫（web scraping）或 AI agent 有没有访问权限？谁实际访问了系统？ |
> | **Patent Law**（专利法） | 技术是否有 novelty？是否侵犯已有专利？AI 能否当发明人？怎样做技术许可（license）？ |
>
> 教授认为最值得继续深入的两个方向：AI training 的 fair use（决定开发时能怎样用数据），以及 agentic AI 的访问授权与责任归属（决定能自主操作互联网的 agent 该怎样设计）。

### 行为拆分

- **AI 不是独立的法律部门**：训练、保存语料、提供模型、生成输出、分发输出，是不同主体（actor）在不同时间做的不同行为。
- **每个行为单独分析**：对象、权利、责任人和抗辩都要重新确定。
  - 不能从“模型学过作品”推出“每次输出都侵权”。
  - 也不能从“输出没有复述”反推“训练时没有复制”。
- **两类主张**（CS202 第一讲的分法）：
  - **直接侵权（direct infringement）**：开发者从网上复制、下载受保护的书来训练，复制本身落在 § 106，主要抗辩是 fair use。看开发者自己做了什么。
  - **间接侵权**：模型让用户能做出侵权作品，权利人主张开发者诱导了用户侵权。看产品怎样设计和推广，见版权一章的 *Cox v. Sony*。
  - 两类案子的诉讼费用和商业敞口都极大。

“模型构成侵权”至少要拆成四件事：

| 行为 | 要证明什么 | 典型抗辩或案例 |
| :--- | :--- | :--- |
| 1. 训练时制作复制件 | 复制事实 | § 107 fair use |
| 2. 把盗版库当永久图书馆（存，不一定训） | 复制和保存 | 看副本来源、是否增加副本数量、图书馆用途；*Bartz* 对影子库下载和购书后的等量数字替代给出相反结论 |
| 3. 输出与作品实质性相似（复述、角色、逐页替代） | copying + substantial similarity | 独立生成、不受保护元素 |
| 4. 把角色或商标（trademark）当成产品功能来卖 | 直接侵权，或 *Grokster* 式诱导 | *Cox* 的意图标准 |

- 四件事的要件和证据不能混用；fair use、许可等抗辩可能反复出现，但要分别适用。
- **原告（plaintiff）怎样对应 § 106**：
  - § 106(1) 复制权：未经许可下载、复制作品用于训练；并主张模型里存有作品的 “compressed copies”（压缩副本）。
  - § 106(2) 演绎权：每一个输出都是训练作品的未授权演绎作品（derivative work）。
  - § 106(3) 发行权：分发含有 “compressed copies” 的模型，等于分发作品副本。
- **哪些容易成立**：第一项最稳，复制事实通常不难证明，争点在 fair use。后两项要证明参数里真的固定了可识别的受保护表达，或输出与具体作品实质性相似；“每个输出都是演绎作品”在早期图像案中已被法院认为过宽。
- **直接与间接的抗辩**：
  - 直接侵权的主要抗辩是 fair use。Stability 曾说：“Anyone that believes this isn't fair use does not understand the technology and misunderstands the law.”
  - 诱导侵权（让用户生成侵权作品或模仿作品“风格”）这一类主张，已被 *Cox v. Sony*（2026）收窄。

### 对象与主体

- **复制或生成了什么**：idea / 事实 / 受保护表达 / 代码 / 整本书 / 权重 / 输出 / 角色 / CMI。
- **触及 § 106 的哪一项**：复制、演绎、发行、展示……
- **谁是行为人**：实验室、云厂商、开源（open source）发布者、下游应用、用户、agent。
- **抗辩是什么**：fair use、许可、安全港，还是根本没有适格的作者。

### 证据清单

1. **行为**：下载、缓存、tokenize、训练、保存、微调、部署、输出、公开展示、分发。
2. **对象**：事实、思想、受保护表达、代码、角色、logo、个人数据（personal data）、商业秘密（trade secret）。
3. **主体**：数据提供者、模型开发者、云厂商、应用开发者、用户。
4. **权利或义务**：§ 106 哪一项、合同、CFAA、隐私法、商标、商业秘密。
5. **抗辩**：许可、fair use、safe harbor、独立创作、无实质性相似。
6. **证据**：数据来源记录、训练管线、输出测试、过滤措施、通知后的处置、市场替代证据。
7. **程序阶段**：诉状（complaint）、取证、summary judgment、审判、和解（settlement）。

### 判例基线

- **没有立法例外**：美国国会没有为 AI 训练设专门例外，争点全落在 § 107。
- **三条线**：到 2026 年秋，三份地区法院（district court）判决（*Ross*、*Bartz*、*Kadrey*）给出了不能互相化约的思路，都不是最高法院（Supreme Court）。第三巡回听过 *Ross* 的辩论，还没下判。

> [!WARNING]+ 2025 年北加州两份意见不是规则
>
> - *Bartz*、*Kadrey* 都是 N.D. Cal. 的 summary judgment，都只依据本案记录。
> - Chhabria 自己说，他的裁定不等于 Meta 对所有权利人的训练都合法。
> - *Ross* 在 Delaware、非生成式、直接竞品，结论相反。
> - 引用时要写明：法院、用途、数据来源、有没有输出主张。

> [!INFO]+ 从产品发布到诉讼潮
>
> - **产品**：DALL-E（2021-01）→ DALL-E 2（2022-04）→ Midjourney 公测（2022-07-12）→ Stable Diffusion（2022-08-22）→ DeviantArt DreamUp（2022-11-11）→ ChatGPT（2022-11-30）。
> - **第一批诉讼**：
>   - *Andersen v. Stability AI*（N.D. Cal., 2023-01-13）：艺术家起诉 Stability、Midjourney、DeviantArt。
>   - *Getty Images v. Stability AI*（D. Del., 2023-02-03）。
> - **规模**：到 2026 年 8 月，美国在审的生成式 AI 版权诉讼超过 130 件，集中在 N.D. Cal. 和 S.D.N.Y.（OpenAI 多案合并的 MDL）。
> - **全球**：加拿大（新闻机构诉 OpenAI 等）、英国（Getty 诉 Stability 已上诉）、德国（GEMA 诉 OpenAI、Suno，慕尼黑）、法国（出版商诉 Meta）、荷兰、欧盟法院（*Like Company v. Google*）、日本（读卖、日经、朝日诉 Perplexity）、韩国（广播协会诉 Naver）、印度（ANI 诉 OpenAI）、巴西、中国（广州互联网法院等奥特曼案）。
> - **已出的 fair use 判决**：*Ross*（2025-03-11）、*Bartz*（2025-06-23）、*Kadrey*（2025-06-25）；下一份重要判决最可能是第三巡回的 *Ross*。

> [!EXAMPLE]+ 训练案之前的先例：*Authors Guild v. Google*, 804 F.3d 202 (2d Cir. 2015)
>
> - **事实**：Google 未经许可扫描数千万册图书，做全文检索和 snippet（片段）展示。
> - **判决**（Leval 法官）：fair use。
>   - 用途 highly transformative；
>   - 整本复制对检索是必要的；
>   - snippet 不构成对原作的实质替代；
>   - 商业动机不会自动击败 fair use；
>   - 作者主张的“检索许可市场”，法院不认。
> - **同一家族**：更早的 *HathiTrust* 图书馆扫描案结构相近。Alsup 在 *Bartz* 里把训练写成更强的 transformation，用的就是这一家族的语言。
> - **差别**：Google Books 给用户的是检索结果和碎片，不生成新的书；LLM 把书压进参数后，可以写出大量“二手”文本。Chhabria 在 *Kadrey* 里抓的正是这一点。

<br>

## 训练

训练阶段最容易混淆四个问题：

- **获取副本**：抓取或下载语料时，是否制作了未经许可的复制件。
- **长期存储**：把数据存进中央库，是否有和训练分开的用途。
- **训练使用**：把合法或非法来源的副本送进训练，是否构成 fair use。
- **参数状态**：模型参数能否固定、再现受保护表达，从而构成法律意义上的 copy。不能只凭“信息进了权重”回答。

### 竞争

> [!EXAMPLE]+ *Thomson Reuters v. Ross Intelligence*（D. Del. 2025）：直接竞品不是 fair use
>
> - **事实**：Ross 做的是法律检索工具，不是生成式模型，目标是和 Westlaw 竞争。数据供应商用 Westlaw 的 headnotes 制作 Bulk Memos，供 Ross 训练。
> - **法官**：Stephanos Bibas 是第三巡回法官，经指定（designation）在特拉华地区法院审理；这份意见在先例（precedent）层级上仍是地区法院判决。
> - **结论**：headnotes 有 originality，复制成立，fair use 不成立。
>   - 第一因素：商业性、非变革性，直接做同一功能的竞品。
>   - 第二、三因素偏向 Ross：headnotes 保护范围窄，最终用户看不到那些 memo。
>   - 第四因素：替代了 Westlaw 的市场。第一、四因素压过了第二、三因素。
> - **后续**：Bibas 自己写 *the questions here are hard*，准许中间上诉。第三巡回 2026-06-11 开庭辩论，追问 Ross：在用户眼里，它的用途和 Westlaw 是不是同一件事。至 2026 年 9 月尚未判决。这是目前唯一走进上诉审的训练 / fair use 案件。
> - **要点**：关键不在模型是否生成文本，而在训练材料和最终产品处于同一个法律检索市场。第一因素的 purpose 和第四因素的市场替代因此连在一起。

- **生成式与否不是标准**：最终产品是否直接替代原产品，会同时影响第一因素和第四因素。不能把所有训练当成一类用途：Ross 用别人的表达做出功能相同的检索器；*Bartz* 中 Claude 并不卖书的副本。

<br>

### 来源

- **两个问题**：来源问“副本怎么来的”，用途问“拿来做什么”。
- **不同的复制链条**：买纸书、从授权数据库下载、从影子库（shadow library，盗版电子书站）下载，各不相同。
- **后面合法不能洗白前面**：训练本身可能有自己的 fair use 分析，但不会消除此前下载盗版副本的责任。
- **取得方式本身的风险**：绕过 YouTube 等平台的技术措施抓取，会触发 DMCA § 1201，且不以侵权为前提（见版权一章“DMCA 新案”的 *Ted Entertainment v. Apple*）。

> [!EXAMPLE]+ 数据集作者也会被告：*EVOX Productions v. Stanford*（N.D. Cal. 2026）
>
> - **主张**：EVOX 称其商业汽车图库中的照片被收进 Stanford 研究者编制的两个数据集：ImageNet 和较小的 Stanford Cars dataset；两者都被放在 Stanford 网页上供其他研究者下载。
> - **被告（defendant）**：Stanford 大学；主持 ImageNet 的李飞飞没有被列为被告。ImageNet 及其年度竞赛是深度学习兴起的关键基础设施。
> - **和训练案的区别**：被诉行为不是“用作品训练模型”，而是编制并公开分发含有照片的数据集。这更接近复制、发行和公开展示，fair use 的论证不能直接借用 *Bartz* 那套“训练是高度变革性”的语言。
> - **要看的事实**：数据集分发的是图片本身还是 URL；是否限于研究用途、有无访问条款；Stanford 收到通知后怎样处理；同类数据集的下游使用者是否也有风险。
> - **对研究者**：发布数据集是一个独立的法律行为。数据卡里写清来源、许可和移除机制，与写清模型能力同样重要。

> [!EXAMPLE]+ *Bartz v. Anthropic*（N.D. Cal. 2025-06-23，Alsup 法官）：三个行为，三个结论
>
> - **当事人**：原告包括作者 Andrea Bartz、Charles Graeber、Kirk Wallace Johnson。
> - **事实**：Anthropic 先从 LibGen 一类影子库下载数百万册书，后来改为购买二手书、拆开扫描。原告主张训练复制和“中央图书馆”侵权，没有主张 Claude 输出大段复述。
>
> | 行为 | 结论 | 理由 |
> | :--- | :--- | :--- |
> | 用书训练 LLM | fair use | *transformative—spectacularly so* |
> | 买纸书 → 销毁原件 → 制作等量数字版 | fair use | 只是换存储格式，副本总数没变 |
> | 从影子库下载并永久保存 | **不是** fair use | 后来不用于训练，也不能让此前的下载和保存变合法 |
>
> - **训练这一支的四因素**：
>   1. **第一因素**：把 token 压进参数、学的是统计规律，不是把书拿给用户读。类比阅读和记忆——每次想起一本书、每次用新方式写新东西都要付钱，“unthinkable”。商业性存在，但压不过变革性。
>   2. **第二因素**：偏向作者（小说是创造性作品），但这一项向来权重低。
>   3. **第三因素**：整本复制，但对训练 reasonably necessary；整本复制不自动致命，*Campbell*、*Sega* 都有过这种结构。
>   4. **第四因素**：保护的是替代原作的市场，不是一切竞争。原告没主张 Claude 会逐页替代，Anthropic 也有防复述措施。“训练许可市场”不是版权法保证作者独占的市场；fair use 成立时，不能靠“本来可以收训练费”反过来赢回第四因素。Chhabria 两天后在 *Kadrey* 里明确反对这种处理。
> - **和解**（只针对盗版库这一支）：
>   - 金额 15 亿美元，作品清单约 482,000 部，每部粗算约 3,000 美元；销毁盗版数据集。
>   - Alsup 2025 年 9 月初步批准，年底他转为 inactive 状态，案件随机分给 Araceli Martínez-Olguín 法官，2026-07-20 由她终局批准；律师费压到约 6.8%（原告律师曾从 20% 降到 12.5% 开口）。
>   - 和解覆盖从影子库下载的作品，不涉及输出，也不等于承认或推翻训练的 fair use。法定赔偿（statutory damages）按件数相乘，理论敞口可以威胁公司存亡；Anthropic 曾想就 fair use 和集体诉讼认证上诉，没等上诉就和解了。
>   - 2025 年 6 月的 summary judgment 是 Alsup 作出的；2026 年的终局和解令是 Martínez-Olguín 签署的，不要混成同一位法官。
> - **时间巧合**：判决出在 2025-06-23，刚好在 CS202 Spring 2025 那一期结课之后约两周。
> - **启示**：15 亿美元量化的是特定作品清单的既往盗版下载请求，不是法院给训练合法性或训练许可定价。

<br>

### 市场

- **第四因素不只看会不会逐字吐书**。至少三种市场理论：
  - **输出替代**：输出直接替代原作或其衍生市场。
  - **训练许可**：未经许可的训练侵占了已有的、或通常可能发展出来的训练许可市场。
  - **市场稀释**（market dilution）：模型批量生成同类作品，压低原作者作品的需求。
- 前两种在既有 fair use 案例里较常见；第三种是 *Kadrey* 强调、但原告证据不足的一条。证明它需要经济和市场证据，不能只说模型“会竞争”。

> [!EXAMPLE]+ *Kadrey v. Meta*（N.D. Cal. 2025-06-25，Chhabria 法官）：同样的结论，不同的理由
>
> - **当事人**：Richard Kadrey、Sarah Silverman、Junot Díaz 等十三位作者。涉及 Llama、Books3 一类影子库，还涉及 torrent。和 *Bartz* 同一个地区法院，相隔两天。
> - **结论**：在本案记录上，训练是 fair use，即使作品来自盗版库。
> - **和 Alsup 的两处分歧**：
>   1. **盗版来源**：训练本身仍可构成 fair use，来源问题另行分析。
>   2. **第四因素最重要**：批评 Alsup 把训练类比成教孩子写作不准确。LLM 能以极低成本制造大量同类作品；用原作提高模型能力后，大量同类输出会稀释原作市场。书越有训练价值，反而越可能支持训练与市场稀释之间的因果联系。
> - **为什么原告还是输了**：几乎没有为市场稀释提交证据。Chhabria 明确说，裁定只表明这十三位作者的理论和证据不足，*不*等于 Meta 对所有权利人的训练都合法。
> - **还活着的部分**：torrent seeding（把盗版书再分发出去）的主张仍在审，2025 年底原告还在修改诉状。
> - **失败的部分**：DMCA CMI 请求——训练复制被认定为 fair use，原告也没建立移除 CMI 与掩盖其他上传侵权之间的证据联系。
> - **程序**：2026-07-08 拒绝向第九巡回提交中间上诉，理由是终局判决后可以一并上诉；这只是程序判断，不保证裁判日期。训练问题目前在西海岸没有上诉审。
> - **启示**：同一地区、相隔两天的两份意见路径不同。上诉法院判决之前，不能写成“美国允许训练”。原告若主张第四因素，要拿出市场稀释的实际证据，而不只是说“应该有训练许可费”。

<br>

## 输出

### 复制判断

- **和训练分开分析**：先证明特定输出复制了原作，再比较可保护表达。
  - 能提示出事实摘要 ≠ 能提示出受保护表达。
  - 能生成同类风格 ≠ 复制了具体作品。
  - 反过来，长段逐字复现、稳定出现独特的角色设定、出现训练图的水印，会提高 copying 和实质性相似的证明力。
- **Memorization 是技术现象，infringement 是法律结论**。模型记住了某段文本，仍要问：它受不受保护？输出是被告还是用户实施的？有没有公开或分发？有没有 fair use？
- **风格**通常属于思想、技法或一般审美层面；但这不等于角色、具体构图和台词也不受保护。

### 文本输出

> [!EXAMPLE]+ *In re OpenAI Copyright Infringement Litigation*（S.D.N.Y. MDL 25-md-03143）
>
> - **法官**：Stein（地区法官）、Wang（治安法官，管证据开示）。
> - **范围**：作者、新闻机构等案件合并审理（MDL），同时涉及训练复制和 ChatGPT 输出的摘要、大纲。
> - **输出问题**：纽约时报一案较早展示过近乎逐字的文章复述。这是输出问题，不等于回答了训练是否 transformative。
> - **2025-10**：法院认为原告已充分主张了陪审团（jury）可能认定实质性相似的输出，把 fair use 留给后续程序。
> - **证据开示**：2025-11 起要求提供约 2000 万条去标识的 ChatGPT 对话样本；2026 年又要求提供规模更大的对话样本集合，包括约 7800 万条和 1000 万条的集合。案件仍在 discovery（证据开示）阶段。

### 图像与角色

- ***Disney v. Midjourney***（C.D. Cal.，2025 年起诉）：
  - 主张：训练复制；服务持续生成、展示迪士尼、卢卡斯、漫威的角色，并把这种能力当卖点。
  - 诉状强调故意、首页展示、没做过滤，按诱导侵权（induced infringement）和直接侵权一起写。
  - 充分刻画、有稳定独特表达的角色，通常比抽象“风格”更容易主张；但仍须识别具体的受保护元素，不能从“可识别”直接跳到“必然侵权”。
- **角色许可市场的信号**：
  - 2025 年 12 月 Disney 与 OpenAI 宣布三年许可：Sora / ChatGPT Images 可使用 200 多个迪士尼、皮克斯、漫威、星战角色，不含演员肖像和声音；Disney 拟投资 10 亿美元。当时写明尚待最终协议和交割条件。
  - 2026 年 3 月 OpenAI 关停 Sora 应用，交易取消，Reuters 等报道钱没有交割。
  - 对第四因素仍是信号——有人愿意给角色生成定价——但不能写成已经稳定存在、正在收款的市场。Midjourney 那支诉讼还在。
- **更早的图像案**：*Andersen v. Stability*、Getty 案。Stable Diffusion 输出里出现畸变的 Getty 水印，可以作为调查训练来源、输出相似性或来源混淆的线索；但单凭水印不能证明模型保存了完整原图，也不能直接完成 memorization 或侵权的法律证明。图像和文字的 fair use 论证不必是同一套。

<br>

## 作者

### 人类控制

- **规则**：美国版权要求 human authorship（人类作者）。这是对现行法的解释，不是政策偏好。
- **问什么**：谁对最终可感知的表达作出了原创性的控制。
  - 用相机、画笔、软件不妨碍人成为作者，因为表达仍由人决定。
  - 生成模型的难点：prompt 往往只描述目标，具体像素或措辞由系统决定。
  - 人的后期修改、选择、编排可以单独受保护，但不会把没改过的机器生成部分也变成人的作品。

### 机器生成

> [!EXAMPLE]+ *Thaler v. Perlmutter*：机器当唯一作者
>
> - **事实**：Stephen Thaler 的 Creativity Machine 生成《A Recent Entrance to Paradise》（铁轨伸进一片光里的图）。申请表上作者填机器，Thaler 自己只当权利人。版权局拒绝登记。
> - **D.D.C.（Howell 法官）**：human authorship 是 “bedrock requirement”。Work made for hire 也以存在可保护的作品为前提，机器“受雇”产生不了第一步。
> - **D.C. Cir.（2025-03-18）**：维持。注意课件把这份判决标成 “Sixth Circuit”，实际是 D.C. Circuit。
> - **最高法院（2026-03-02）**：拒绝 certiorari。
> - **没回答的**：Thaler 在行政程序里放弃了“我因创造并操作机器而成为作者”的主张，所以法院没处理人类辅助到什么程度才够。拒绝 cert 也不是最高法院对 human authorship 的实体判决。

- **三个样本排成一条线**：
  - *A Recent Entrance to Paradise*（Thaler）：机器被列为唯一作者 → 不能登记。
  - *Zarya of the Dawn*（Kristina Kashtanova, 2022）：漫画的文字和人对图文的选择、编排可以登记；用 Midjourney 生成的单幅图像被排除，版权局撤销原登记后重新发放了范围更窄的登记。
  - *Théâtre D'opéra Spatial*（Jason Allen, 2022）：用 Midjourney 生成、经 600 多次 prompt 迭代和后期修改，获科罗拉多州博览会数字艺术奖；版权局拒绝登记其中 AI 生成部分。Allen 起诉，即 *Allen v. Perlmutter*，争的正是人的控制到什么程度才够，至 2026 年秋仍在地区法院。
- **规律**：人越是直接决定最终可感知的表达（写字、编排、手工修改），越有作者资格；只描述想要什么、由模型决定具体像素，越接近没有作者。猴子自拍（*Naruto v. Slater*, 9th Cir. 2018）是同一条原则在非 AI 场景的样子：非人类不是版权法上的作者。

### 登记政策

Copyright Office *Copyright and Artificial Intelligence*：

- **Part 2: Copyrightability（2025-01-29）**：
  - 不建议修改现行作者资格规则。
  - 纯生成材料不能登记；人对输出的可感知选择、编排、修改可以保护。
  - prompt 通常不足以让提示者成为输出的作者，因为提示者不控制具体表达；但这不等于提示词文本本身一定没有版权。
  - 反复修改 prompt、挑选输出、修图、编排成作品，保护的是人实际贡献的那一层。
  - 登记时应披露并排除不主张保护的 AI-generated material（对应登记表的 6a / 6b 栏）。
- **Part 1: Digital Replicas（2024-07）**：讨论数字复制品和深度伪造（deepfake），建议单独立法，不纳入版权法。

<br>

## 发明人

### 构思标准

- **发明人是谁**：对至少一项 claim 的 conception（构思）作出贡献的自然人。
- **不是谁**：提出课题、出钱或管理项目的人，不因此成为发明人；只按指令做实验的人也不一定是。
- **reduction to practice**（把方案做出来）可以作证据，但不是标准本身。没写代码的人，如果形成了 claim 中完整、可实施的技术构思，也可能是发明人。
- **课上的问题**：AI 自主提出一种新的机械结构或技术方案，inventor（发明人）该写谁？
  - AI 系统本身？——不行，必须是自然人（*Thaler v. Vidal*）。
  - AI 系统的开发者？
  - 使用 AI 的研究人员？
  - 设计问题、验证结果的人？
  - 答案不是“谁运行了 AI”，而是谁对 claim 作出了 conception 层面的贡献。
- **科研 agent 的流程**：人提出研究目标、约束与问题 → AI research agent 搜文献、生成候选设计、跑实验 → 人评估、修改，形成具体技术方案 → 可能的专利申请。发明人要逐项 claim 落到实际完成 conception 的人，不是按谁启动了 agent 来定。

### AI 辅助

> [!EXAMPLE]+ *Thaler v. Vidal*, 43 F.4th 1207 (Fed. Cir. 2022)：DABUS 不能当发明人
>
> - **判决**：专利发明人必须是 natural person。AIA 之后 § 100(f) 把 inventor 定义为 “individual(s)”，宣誓书用 himself / herself。
> - **没回答的**：人借助 AI 完成的发明能不能拿专利。
> - **最高法院**：2023-04-24 拒绝 certiorari。
> - **其他法域**：英国最高法院（2023-12）、EPO 都不接受 DABUS 作发明人；德国联邦最高法院（2024-06）允许把 Thaler 本人列为发明人，并注明他让 DABUS 生成了方案；只有不做实质审查的南非在 2021 年授权。

- **USPTO 指导的反复**：
  - 2024-02-13：用 *Pannu* 共同发明（joint invention）标准筛查人有没有 “significant contribution”。
  - 2025-11-26：整份指导废止，改发修订稿。理由是 *Pannu* 只适用于多个自然人之间，AI 不是人，谈不上共同发明人。
  - **现行口径**：单人借助 AI 时，回到传统 conception——发明人要对完整、可实施的发明形成确定、持久的观念。AI 被看作实验设备、软件或数据库；用了别人的服务和想法，那些来源不因此成为发明人。列发明人时不得写上模型。多人共同发明时，*Pannu* 仍用来比较人与人之间的贡献，不用来比较人和模型。
- **为什么这个问题会变大**：课件引 2026-06-11 NYT：Bezos 创办的 Prometheus 已融资 120 亿美元以上，目标是做 “artificial general engineer”，加快“发明循环”；他的说法是 “All societal wealth is driven by invention.” AI 承担越多设计工作，“哪个自然人完成了 conception”越难证明，也越决定专利能否成立。

### 版权对照

| | 版权 | 专利 |
| :--- | :--- | :--- |
| 问什么 | 这件作品有没有人的表达 | 这项 claim 有没有人完成 conception |
| 典型的“够” | 人写字、编排、手工修改 | 人从模型给出的几千个候选里选出一种，并验证、改成具体方案 |
| 典型的“不够” | 按一次生成键得到一张图 | 只是启动了 agent、提出了课题 |

- 实验室用代码助手写实现，conception 仍要落到具体的人。团队工具普及之后，漏列发明人更容易发生。

<br>

## 比较法

### 比较维度

- **同一批技术事实，不同工具**：
  - 美国：事后由法官按 fair use 判断。
  - 欧盟：TDM 例外 + 权利保留 + GPAI 透明度。
  - 中国：合法来源 + 内容责任 + 标识。
  - 日本、新加坡：较明确的专门数据分析例外。
- **不能只比“允许 / 不允许训练”**，至少同时比：
  - **例外结构**：开放式 fair use，还是封闭列举的法定例外。
  - **访问前提**：是否要求 lawful access（合法访问）。
  - **权利保留**：权利人能否 opt out（选择退出），怎样表达保留。
  - **用途区分**：是否区分科研训练与商业训练。
  - **透明度**：是否要求公开训练内容摘要或其他文档。
  - **输出责任**：模型或平台对违法、有害、侵权输出承担什么责任。
- **不能跨法域借用**：一个法域的例外，不能直接当另一个法域的抗辩。
- **lawful access 的含义因法域而变**：是否付了订阅费、是否越过登录或技术措施、访问条款怎样限制使用、robots.txt 是否构成有效保留，都会影响结论。它不是“网页能打开就合法”，也不是“违反任何网站条款就不合法”。

### 欧盟

- **思路**：不走事后 fair use，而是“例外 + 退出”。欧盟把选择权放在权利人**事前**，美国把判断权放在法官**事后**。
- **TDM**（text and data mining，文本与数据挖掘）：用自动化技术分析文本和数据，发现模式、趋势和相关性。
- **DSM Directive (EU) 2019/790**：

| | 第 3 条 | 第 4 条 |
| :--- | :--- | :--- |
| 谁可以用 | 研究组织、文化遗产机构 | 范围更广，包括商业主体 |
| 前提 | 合法访问 | 合法访问 |
| 权利人能否退出 | 不能一般性 opt out | 可以按第 4(3) 条“以适当方式”保留权利，实践中常理解为机器可读的 opt-out |

- **仍在磨的问题**：robots.txt、网站条款、专门协议，哪些算“适当方式”。
- **AI Act（Regulation 2024/1689）第 53 条**：把版权义务接到 GPAI（通用目的 AI 模型）提供者身上。
  1. 编制技术文档；
  2. 向下游系统提供者披露必要信息；
  3. 建立遵守欧盟版权法的政策，包括用先进技术识别并尊重第 4(3) 条的保留；
  4. 按 AI Office 模板公布“足够详细”的训练内容摘要。
  - 开源 GPAI 在文档义务上有条件豁免；有系统性风险的模型不豁免。
  - 罚款最高可达全球年营业额 3% 或 1500 万欧元，取高者。
- **时间表**：
  - 2025-08-02：GPAI 义务开始适用。
  - 2026-08-02：AI Office 对 GPAI 的执法权生效。
  - 2025 年 7 月的训练内容摘要模板：不要求倾倒数据集，但要覆盖从预训练到后训练的主要数据、抓取来源、显著域名。
  - 2025 年 8 月之前投放市场的模型，摘要可延到 2027-08-02。
  - 行为准则（code of practice）是证明合规的路径，不改写实体法。
  - 2025–2026 年有简化高风险规则期限的政治讨论，附件 III / 附件 I 的日期被调整过，引用时要核对最新文本。
- **启示**：同一套权重，在欧盟要处理 opt-out 和摘要，在美国要准备 fair use 诉讼记录。两边的合规动作不能互相翻译。

<br>

### 中国

- **主轴是内容治理**：知识产权（intellectual property）是其中一条，不是唯一主轴。先看清适用范围。
- **三份规则不在同一层面**，都是行政监管义务，不替代法院按著作权法判断作品、复制、实质性相似和合理使用：

| 规则 | 施行 | 管什么 | 要点 |
| :--- | :--- | :--- | :--- |
| 《互联网信息服务深度合成管理规定》 | 2023-01-10 | 深度合成服务（文本、图像、音频、视频、虚拟场景）与标识 | 第 16 条隐式标识；第 17 条对可能导致混淆误认的服务作显著显式标识；禁止删除、篡改、隐匿 |
| 《生成式人工智能服务管理暂行办法》 | 2023-08-15 | **向境内公众**提供生成文本、图片、音频、视频的服务；内部研发和不对公众提供的应用不适用 | 第七条训练数据（training data）：合法来源、不得侵害知识产权、个人信息需同意或其他法定情形、提高数据质量。提供者对输出承担网络信息内容生产者责任，对个人信息承担处理者责任。第四条是义务清单（价值观、反歧视、知识产权和商业秘密、人格权、透明度），不是抗辩清单 |
| 《人工智能生成合成内容标识办法》 | 2025-09-01 | 把生成、传播、应用分发和用户声明串成标识链 | 显式标识 + 文件元数据中的隐式标识（属性、服务提供者、内容编号）；同日施行强制性国家标准 GB 45438-2025 补技术细节；传播平台有核验和提示义务 |

- **标识不是版权登记**：但在诉讼里会和“是不是作品、谁是作者、平台有没有尽到注意义务”缠在一起。去掉标识再传播（如李某案那种事实），以后更容易被认定为恶意。
- **没有 TDM 例外，也没有一般 fair use**：著作权法的合理使用是封闭列举。《暂行办法》第七条把训练阶段的版权风险写成了监管义务，但什么训练构成侵权，仍要回到复制、信息网络传播、合理使用，以及正在形成的判决。监管先要求“合法来源”，法院还在逐案问“合法”指什么。

> [!EXAMPLE]+ 北京互联网法院 李某诉刘某（(2023)京0491民初11279号）：AI 图片有作者
>
> - **事实**：李某用 Stable Diffusion 反复改提示词和参数，生成图片并署名（attribution）发布；刘某去掉署名后使用。
> - **判决**：图片中有人的独创性智力投入，构成美术作品，赔偿 500 元。
> - **对比美国**：美国版权局通常认为 prompt 本身控制不足；北京互联网法院则把提示词和参数的迭代算进独创性。保护的仍是人的安排与选择，不是模型的人格。

- **输出侵权**：杭州、广州等地的奥特曼生成图案从输出切入。输出与受保护角色实质性相似时，生成服务提供者可能承担直接或帮助侵权（contributory infringement）；通知、过滤、标识、是否把角色生成当卖点，都会影响注意义务。
- **训练数据案件**仍在形成，尚无能稳定对标 *Bartz* 的上级法院规则。

### 其他

- **专门例外的取舍**：日本、新加坡用专门的数据分析例外，不是开放式 fair use。好处是训练者容易预判；代价是每个法定前提都要满足，lawful access、非享受目的、是否不当损害权利人利益会成为关键词。
- **日本：著作权法第 30 条之 4**（2018 年修改，2019 年施行）：
  - 不以“享受”作品所表达的思想感情为目的的利用（包括数据分析），在必要范围内可以；不当损害权利人利益的除外。
  - 文化厅 2024 年整理：生成式 AI 训练原则上属于非享受目的；但过拟合到能复现表达、把作品当检索结果展示、绕过 robots.txt、抓取有偿的分析用数据库，可能构成不当损害。
  - 给训练开的口子比美国成文法宽，但比 Alsup 在 *Bartz* 里的说法窄：复现表达、绕过技术措施，明确不在例外内。
- **新加坡：Copyright Act 2021 第 243–244 条**（computational data analysis，计算数据分析）：
  - 明文覆盖机器学习，包括商业训练，前提是合法访问。
  - 第 187 条原则上使排除这一例外的合同条款无效。
  - 2026 年 MinLaw / IPOS 咨询讨论的是访问范围条款、技术措施和 robots.txt 怎样影响 lawful access，而不是重新决定合同能否排除该例外。付费墙、API 条款和 scraping 会是主要争点。
- **英国**：2024–2025 年讨论过带 opt-out 的透明 TDM 例外，后来的报告回到维持现状；商业训练在英国仍更接近需要许可。英、美、新三地对商业训练的态度不一致。

<br>

# 智能体

> **核心问题**：AI 替人浏览、点击、下单、提交代码时，出了问题算谁的？

本章以一条 Agent 轨迹为单位组织责任：

- **行为链**：把预训练、推理时读取和工具执行分开。
- **分层责任**：依次分析感知、规划、行动和输出。
- **控制证据**：查明授权、实际访问、越界和结果去向。
- **工具边界**：RAG、代码助手、网页抓取（scraping）分别涉及什么既有规则。
- **平台归责**：Agent 代表用户发帖、交易或提供专业服务时，行为归于谁。

## 行为链

- **Agent 是什么**：把一次生成扩展成带工具的连续行动——浏览、点击、填表、调 API、读写邮箱、提交 PR。
- **多种规则同时出现**：版权（copyright）、合同、CFAA、隐私可能在同一条轨迹上叠加。
- **两种复制要分开**：预训练用 Common Crawl 是一件事；推理时对当前网页的即时复制是另一件（是否构成 copy，见版权一章“固定的边界”）。
- **按动作日志分析，不按“agent 很自主”分析**：模型提出计划、用户授予权限、工具发出 HTTP 请求、平台接收提交，可能分别由不同主体控制。
- **autonomy（自主性）**会影响可预见性和控制能力，但不会自动产生一种新的“智能体责任”。
- **新的复杂性**：传统软件的分析围绕开发者、平台和最终用户；agent 则是用户只给高层目标，由 agent 自己决定步骤、调用工具、访问资源。

> [!EXAMPLE]+ “帮我买最合适的商品”（CS202 第一讲）
>
> agent 可能自己完成：搜索 → 抓取商品信息 → 比价 → 登录账户 → 提交订单。此时要分别判断：
>
> 1. 用户究竟授权了哪些行为？
> 2. agent 是否超出了用户的授权？
> 3. 第三方网站是否允许这类访问？
> 4. agent 开发者是否要为某个具体行为负责？
> 5. agent 复制了受版权保护的内容，是否构成侵权（infringement）？
> 6. agent 自主提出了可专利（patentable）的技术方案，谁是发明人（inventor）？
>
> 这些问题不能用一句“用户授权了 AI”解决。

## 分层责任

| 层 | 典型动作 | 要检查什么 |
| :--- | :--- | :--- |
| 感知 | 读网页、读代码、读邮件 | 有没有产生复制件；是否绕过技术措施 |
| 规划 | 用系统提示、记忆、检索库做计划 | 这些材料各自的权利归属 |
| 行动 | HTTP 请求、点击、git push | 对照 ToS、CFAA 和平台规则判断权限边界 |
| 输出 | 代码、文案、图片、PR | 作者资格、侵权、商业秘密（trade secret）、许可证（license） |

## 控制证据

- **每层问四件事**：
  - **授权**：谁授权了这个动作，范围到哪里。
  - **访问**：系统实际访问了什么数据、页面、仓库或账户。
  - **边界**：是否越过技术措施、账号权限或合同限制。
  - **去向**：产生的副本和对外输出去了哪里，谁能拿到。
- **高风险工具再加**：最小权限、确认点、日志、撤销、人工复核。这些技术控制会成为过失、诱导、保密措施和合规义务的证据。

> [!EXAMPLE]+ 从 *Perplexity* 倒推的设计问题
>
> | 设计问题 | 对应的法律问题 |
> | :--- | :--- |
> | Agent 用谁的账户、谁的凭据 | 授权来自谁，范围到哪里 |
> | 请求从用户设备发出，还是从开发者服务器发出 | CFAA 意义上谁在 access |
> | 是否伪装 user-agent、绕过登录或反爬措施 | 是否越过技术 gate |
> | 是否违反目标网站的 ToS | 合同责任，开发者是否也受约束 |
> | 是否保存、回传第三方页面数据 | 复制、隐私、数据使用 |
> | 是否替用户付款下单 | 代理权限、交易确认 |
>
> 第九巡回看的正是第二行。用户同意让 agent 代办，只解决了用户和开发者之间的授权；第三方网站的条款、版权和隐私问题还在。法律上的授权、产品里的权限控制、模型实际能执行的动作，是三层，要分别设计。

## 工具边界

- **RAG 贴进 GPL 代码**：把某段 GPL 源码贴进 PR，至少发生了复制。是否构成 GPL 意义上的传播（conveying），要看 PR 只在同一法律主体内部，还是提供给外部协作者、客户或公众。对应源代码等义务在传播触发后才分析，不必先回答“模型权重是不是衍生作品（derivative work）”。
- **把客户代码丢进第三方托管模型**：可能直接违反 NDA。
- **代码助手与发明人**：参与实现不自动导致漏列发明人；仍要逐项 claim 判断哪些自然人完成了 conception，并保留提示、输出、选择和实验记录。
- **robots.txt、付费墙、登录页**：
  - 在日本 2024 年意见和欧盟 opt-out 实践里，被当成权利人是否保留权利的信号。
  - 在美国更多是合同和 CFAA 问题；能否单独构成版权问题，还要看复制的量和用途。
  - *Sega* 的中间复制和预训练使用整本书，技术事实不同，不宜直接类比。

## 平台归责

- **归属问题**：agent 以用户名义发帖、下单、注册时，§ 230、消费者保护、广告法都要判断这个表达或交易归谁。
- **授权 ≠ 用户发表了一切**：用户授予了权限，不等于每一项输出都算用户发表的。
- ***Perplexity* 的边界**：它把 CFAA 上的 access 归给用户，是刑法条文的措辞和那套“截图—指令—本地执行”架构一起决定的，不能推广成“agent 的一切行为都归用户”。
- **行业规范优先**：证券、医疗、法律咨询等已有职业规范的领域，agent 的输出先受行业法约束，再另行分析版权。

<br>

# 未决问题

> **核心问题**：哪些问题法院还没给出终局答案？现在能确定到什么程度？

本章不是列出结论，而是保存尚待上诉法院、最高法院（Supreme Court）或立法机关回答的争点：

- **训练用途**：用途上的 transform 与大规模中间复制如何协调。
- **盗版来源**：来源违法应独立分析，还是影响训练的 fair use。
- **许可（license）市场**：正在形成的训练和角色许可市场如何进入第四因素。
- **输出边界**：记忆、角色、逐页替代和风格模仿的保护梯度。
- **人类贡献**：AI 辅助创作达到何种程度才有作者。
- **智能体复制**：推理时即时复制能否沿用预训练分析。
- **开放权重（open-weight）**：商业秘密（trade secret）消失后，版权（copyright）和合同能限制到什么范围。
- **数据集（dataset）与反规避（anti-circumvention）**：数据集发布者的责任，以及 § 1201、§ 1202 能否绕开 fair use。
- **语言与接口**：编程语言、API 声明本身是否受版权保护。
- **分析顺序**：在政策设计前先确定行为、主体、主张和证据。

以下问题尚无最高法院终局规则，但已有足够判决和诉讼材料形成明确争点。

## 训练用途

- **问题**：训练通用模型，是一种用途上的 transform，还是大规模复制的中间步骤？
- **现有答案**：
  - *Warhol*：要看具体的市场用途。
  - Alsup（*Bartz*）：把技术过程写成高度变革。
  - Chhabria（*Kadrey*）：把市场稀释（market dilution）留给下一家证据更好的原告（plaintiff）。
  - Bibas（*Ross*）：直接做功能相同的竞品时，只说“这是训练”过不了第一、第四因素。

## 盗版来源

- **问题**：盗版来源是独立的复制行为，还是会影响训练用途本身的 fair use？
- **现有答案**：两位北加州法官处理不同（Alsup 分开判、盗版部分不是 fair use；Chhabria 认为训练仍可是 fair use，来源另论）。15 亿美元和解（settlement）只量化了特定作品清单的既往下载请求，没有为训练建立一般规则或许可价格。

## 许可市场

- **问题**：许可市场一旦大量存在，未经许可的训练还能不能说“没有受保护的市场”？
- **现状**：新闻社、图库、影视角色都在尝试交易。Disney–OpenAI 宣布过、没交割就停了，说明有定价意愿，但市场还不稳定。

## 输出边界

- 输出争议不能排成一条简单的强弱线：
  - 逐字或逐页替代：最接近复制受保护表达。
  - 角色：看是否充分刻画、输出取走了哪些具体元素。
  - 抽象风格：本身通常不受版权垄断；但风格化输出仍可能复制角色、构图、服装、场景或台词。

## 人类贡献

- 人的辅助到哪条线才构成作者，*Thaler* 没有回答。*Allen v. Perlmutter* 还在地区法院（district court）。

## 智能体复制

- agent 的即时复制和预训练复制，是否适用同一套 fair use？技术事实不同：前者可能只在内存里存在片刻（先过 *MAI* / *Cartoon Network* 的 copy 门槛），后者是整本长期保存。

## 开放权重

- 权重公开后，公开部分不再有商业秘密的秘密性；版权和合同还能限制到什么程度？
- 模型许可如果只产生合同义务，对没有和发布者形成合同关系的下游下载者，约束力有限。

## 数据集与反规避

- **数据集发布**：*EVOX v. Stanford* 把问题从“训练是否 fair use”前移到“编制并公开分发含作品的数据集”。学术用途、只分发 URL、访问限制分别能起多大作用，尚无上诉审答案。
- **§ 1201**：*Ted Entertainment v. Apple* 等案主张绕过平台技术措施抓取（scraping）。若成立，即使训练本身是 fair use，取得数据的方式仍要负责。
- **§ 1202**：*Doe v. GitHub* 问的是，输出不逐字复制时，能否算“移除”了 CMI。答案决定开源（open source）许可证里的署名（attribution）条款能否通过 DMCA 约束代码模型。

## 语言与接口

- 美国没有判决回答编程语言本身能否受版权保护；*Google v. Oracle* 也回避了 API 声明的可版权性。
- 欧盟 *SAS v. WPL* 已答“否”。
- 版权局（Copyright Office）对 Myst 字母的处理（交流系统落入 § 102(b)）提供了类比，但不是法院判决。

## 分析顺序

1. **行为与对象**：先写清复制了什么、存了什么、用户是否看见内容、是否绕过技术措施。
2. **主体与主张**：再写作者、实验室、下游、用户、平台和监管机关分别主张什么。
3. **规则与方案**：最后才讨论例外、强制许可、透明度、标识、选择退出（opt-out）和责任保险。
4. **架构反馈**：每一种方案都会反过来改变模型的过滤、记忆、工具权限和日志设计。

<br>

# 资料

本章按资料用途整理，便于回查：

- **权威层级**：区分具有约束力的法源、说服性判决、行政材料与诉讼事实材料。
- **课程**：三门课程及其页面。
- **法源**：美国法条、欧盟指令和 AI Act。
- **判例进度**：AI 作者、发明人、训练和输出案件的程序状态。
- **比较法材料**：中国、日本和新加坡的制定法与技术标准。
- **阅读建议**：如何把旧课程与 2025–2026 年案件接成一条线。
- **术语表**：约 50 个核心词的中文名和一句话解释。

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

- **15.628J**：[主页](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/) · [Syllabus](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/syllabus/) · [Readings](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/readings/) · [Assignments](https://ocw.mit.edu/courses/15-628j-patents-copyrights-and-the-law-of-intellectual-property-spring-2013/pages/assignments/)
- **6.4590**：[OCW](https://ocw.mit.edu/courses/6-4590-foundations-of-information-policy-fall-2024/) · [课网](https://internetpolicy.mit.edu/6.4590/) · [About](https://internetpolicy.mit.edu/6.4590/about-the-class/)
- **Stanford CS202**：[ExploreCourses 正式课程描述](https://explorecourses.stanford.edu/search?q=CS+202) · [Fall 2006 旧课程归档](http://web.stanford.edu/class/cs202/)。现代案例来自 ExploreCourses 和 Fall 2026 课件，不来自 2006 归档。
  - **课程范围**（syllabus）：四大 IP（专利、版权、商标、商业秘密）的基础与现状；DMCA 与 CFAA；AI 在各领域带来的新问题；IP 相关的雇佣与合同问题（许可、转让、NDA）。
  - **第 1 讲（2026-09-22）**：IP 总论与价值；四大 IP 各举例；训练诉讼的两类主张、*Bartz* 与 *Kadrey*、*Cox v. Sony*；DMCA 新案；CFAA、*Van Buren*、*hiQ*、*Amazon v. Perplexity*、*LinkedIn v. ProAPIs*；深度伪造立法；外观设计与 *Apple v. Samsung*；新颖性与属地性；DABUS / *Thaler*；PageRank 与 Overture。
  - **第 2 讲（2026-09-29）**：美国法院体系；思想投递（*Desny*、*Norris*）；专利 §§ 101–102：可专利对象、先申请制、期限、claim、阻挡专利、实用性、新颖性、宽限期与秘密销售、Blue Origin。
  - **第 3 讲（2026-10-06）**：Lincoln 的专利；§ 103 与 Apple 滑动解锁；专利文件结构与 § 112；claim construction 与 Neonode。
  - **第 4 讲（2026-10-13）**：版权入门：1790 年版权法、目的、独创性、固定、§ 106、§ 102 作品类型、§ 103 汇编与演绎、共同作品、实用物品、不受保护的对象、软件、*Oracle* 的可版权性、编程语言。
  - **课外材料**：总统与 IP 的历史；Lincoln 与收割机案（Spring 2025）；版权局驳回 Myst 字母登记的复议决定（2004）；Apple '721 与 Neonode '879 专利全文；课程项目视频（发明披露表、版权登记表）。
  - **待补**：邮件预告的专利侵权、专利统计、申请流程与 PCT，以及之后的版权侵权、DMCA、商标、商业秘密、合同等讲次。

## 法源

- **版权**：17 U.S.C. §§ 101, 102, 103, 106, 107, 109, 117, 201, 204, 408, 412, 504, 512, 1201, 1202；37 C.F.R. § 202.1(a)。
- **专利**：35 U.S.C. §§ 101–103（含 § 102(b)(1) 宽限期）, 112, 154, 256, 262, 271, 282, 284–286, 311–319（IPR）；37 C.F.R. §§ 1.56, 1.97（IDS）。
- **商标**：15 U.S.C. §§ 1052(a), (c), 1114, 1125。
- **商业秘密与计算机**：18 U.S.C. §§ 1030（CFAA）, 1836（DTSA）。
- **程序与平台**：28 U.S.C. § 1400（审判地）；47 U.S.C. § 230。
- **欧盟**：DSM Directive 2019/790 arts. 3–4；Regulation (EU) 2024/1689（AI Act）art. 53。

## 判例进度

- **行政指导**：
  - USPTO inventorship guidance：2024-02-13 发布，2025-11-26 废止，改回 conception 标准。
  - Copyright Office *Copyright and Artificial Intelligence*：Part 1 Digital Replicas（2024-07）；Part 2 Copyrightability（2025-01-29）。
- **AI 训练与输出**：
  - *Authors Guild v. Google*, 804 F.3d 202 (2d Cir. 2015)。
  - *Bartz*：summary judgment 2025-06-23（Alsup）；终局批准 2026-07-20（Martínez-Olguín，N.D. Cal. 4:24-cv-05417）。
  - *Kadrey*：拒绝中间上诉 2026-07-08。
  - *Ross*：第三巡回 No. 25-2153，2026-06-11 辩论，至 2026-09 未判。
  - Disney–OpenAI Sora 许可：2025-12 宣布，2026-03 随 Sora 应用关停取消。
  - *EVOX v. Stanford*（N.D. Cal. 2026）。
- **AI 作者与发明人**：*Thaler v. Perlmutter* cert. denied 2026-03-02；*Thaler v. Vidal* cert. denied 2023-04-24。
- **间接责任与 DMCA**：*Cox v. Sony*, 607 U.S. ___ (2026-03-25)，No. 24-171；*Doe v. GitHub*（9th Cir. 2026-09-16）；*Ted Entertainment v. Apple*（N.D. Cal. 2026）；*Yout v. RIAA*（2d Cir. 2026）；*Google v. SerpApi*（N.D. Cal. 2025）。
- **CFAA 与抓取**：*Van Buren*, 593 U.S. 374 (2021)；*Amazon v. Perplexity* 初步禁令 2026-03（Chesney，N.D. Cal.），第九巡回 No. 26-1444 于 2026-08-04 撤销；*LinkedIn v. ProAPIs* 2025-10 起诉，2026 consent judgment；*Meta v. Bright Data* 2024-01 简易判决，Meta 撤诉。
- **专利**：
  - *Apple v. Samsung*（'721）：陪审团 2014-05；合议庭 2016-02-26 推翻；全院 2016-10-07 恢复（8:3）；cert. denied 2017-11-06。
  - *Samsung v. Apple*（外观专利）：最高法院 2016；重审 2018-05 约 5.39 亿美元，2018-06 和解。
  - *SpaceX v. Blue Origin*：IPR2014-01376，claims 1–13 撤销。
  - Neonode '879：PTAB 2022-12-15（Samsung/Apple）、2023-01-11（Google）；Fed. Cir. 2024-07-18 维持；Samsung 2025-09 和解；N.D. Cal. 2026-06-08 依 § 112(a) 判无效，已上诉。
  - *Celanese v. ITC*（Fed. Cir. 2024）；*NCS Multistage v. Nine Energy*（Fed. Cir. 2026-09-14）；*WPEM v. SOTI*（Fed. Cir. 2020）。
  - *TC Heartland*, 581 U.S. 258 (2017)；*Diamond v. Chakrabarty*, 447 U.S. 303 (1980)。
- **版权**：*Tangle v. Aritzia*（9th Cir. 2025-01-14）；*Star Athletica*（2017）；*Oracle v. Google*（Fed. Cir. 2014-05-09）。
- **商标**：*Matal v. Tam*（2017）；*Iancu v. Brunetti*（2019）；*Vidal v. Elster*（2024）；*Jack Daniel's v. VIP*（2023）；CHATGPT（TTAB 2026-03-04）、OPENAI（TTAB 2026-08-06）。
- **合同与思想**：*Desny v. Wilder*（Cal. 1956）；*Norris v. NBA Properties*（S.D.N.Y. 2020）。

## 比较法材料

- **中国**：《互联网信息服务深度合成管理规定》（2023-01-10）；《生成式人工智能服务管理暂行办法》（2023-08-15）；《人工智能生成合成内容标识办法》（2025-09-01）；GB 45438-2025。
- **日本**：著作权法第 30 条之 4。
- **新加坡**：Copyright Act 2021 ss. 187, 243–244。

## 阅读建议

- **15.628J**：值得按 session 过。先建立四种 IP 和 claim 意识，再读软件专利史，再把 *Alice* 以后和 2025–2026 训练数据判决当成同一条时间线上的续集。
- **6.4590**：不必按周重做。架构、§ 230、*Carpenter*、跨境那几周的读物，当政策写作样本足够。
- **CS202**：按讲次读课件，再回到本文对应章节；案例方框比课件多了“说明什么”和“程序走到哪一步”。

## 术语表

大致按英文首字母排序。“见”指主要出现的章节。

| 术语 | 中文 | 一句话解释 | 见 |
| :--- | :--- | :--- | :--- |
| abstract idea | 抽象思想 | 不能拿去申请专利的纯想法、数学或商业方法 | 专利 · 适格性 |
| access（CFAA） | 访问 | 进入一台计算机或系统；未经授权的访问可能是联邦犯罪 | 信息政策 |
| affirmative defense | 积极抗辩 | 被告承认基本事实、但主张有法定理由免责，例如 fair use | 导论 |
| anticipation | 预见 | 一份在先文献已公开了 claim 的全部要素，专利因此不新 | 专利 · 新颖性 |
| assignment / assignee | 转让 / 受让人 | 把权利所有权转给别人；接收方叫受让人 | 归属 |
| CFAA | 计算机欺诈与滥用法 | 美国禁止未经授权访问计算机的联邦法，民事刑事都有 | 信息政策 |
| claim（专利） | 权利要求 | 专利文件末尾逐项编号的句子，划定专利到底保护什么 | 专利 · 权利要求 |
| claim（诉讼） | 诉因 | 原告据以要求赔偿或禁令的法律主张 | 导论 |
| claim construction | 权利要求解释 | 法官决定 claim 里每个词是什么意思，常决定胜负 | 专利 · 权利要求 |
| CMI | 版权管理信息 | 作者名、版权声明、许可条款等；故意去掉可能违反 DMCA | 版权 |
| conception | 构思 | 在脑中形成完整、可实施的发明方案；决定谁是发明人 | 专利、AI |
| contributory / vicarious / inducement | 帮助 / 替代 / 诱导侵权 | 三种次级责任：明知而实质帮助 / 能控制且直接获利 / 以促成侵权为目的推广 | 版权、专利、商标 |
| direct infringement | 直接侵权 | 自己实施了法律禁止的行为，如复制作品、制造专利产品 | 各章 |
| secondary liability / indirect infringement | 次级责任 / 间接侵权 | 自己没直接侵权，但因帮助、控制或诱导别人的直接侵权而负责 | 体系 · 次级侵权对照 |
| staple article | 通用商品 | 有大量合法用途的商品；提供它不构成专利帮助侵权，版权的 *Sony* 规则即借自这里 | 专利 · 侵权 |
| willful blindness | 故意视而不见 | 刻意回避去知道事实；在诱导侵权里等同于“知道” | 专利 · 侵权 |
| divided infringement | 分工侵权 | 方法专利的各步骤由不同主体完成，谁算直接侵权人 | 专利 · 侵权 |
| copy（版权） | 复制件 | 作品被固定下来、超过瞬间存在的副本，包括内存里的 | 版权 |
| derivative work | 演绎作品 | 基于原作改编、翻译、改写的新作品 | 版权 |
| design patent | 外观设计专利 | 保护产品的装饰性外观，而不是功能 | 专利 · 外观设计 |
| DMCA § 512 / § 1201 / § 1202 | 数字千年版权法 | 平台安全港 / 禁止绕过技术保护措施 / 禁止去除 CMI | 版权 |
| enablement / written description | 可实施性 / 书面描述 | 说明书必须教会同行实施，并证明发明人当时已掌握 | 专利 · 权利要求 |
| exclusive license | 独占许可 | 只授权给一家使用，权利人自己也可能不再许可他人 | 归属 |
| fair use | 合理使用 | 未经许可使用作品但不算侵权的法定抗辩，按四个因素判断 | 版权 · 合理使用 |
| fixation | 固定 | 作品被记录在稳定介质上；版权从这一刻自动产生 | 版权 |
| grace period | 宽限期 | 美国允许发明人自己公开后 1 年内仍申请专利 | 专利 · 新颖性 |
| holding / dicta | 裁判规则 / 附带意见 | 判决中有约束力的结论 / 顺带说的、约束力弱的话 | 导论 |
| idea–expression dichotomy | 思想—表达二分 | 版权只保护具体表达，不保护背后的思想、方法、系统 | 版权 · 思想 |
| injunction | 禁令 | 法院命令停止某种行为；preliminary 是诉讼期间的临时禁令 | 导论、专利 · 救济 |
| IPR / PTAB | 双方复审 / 专利审判上诉委员会 | 专利授权后，任何人可向 USPTO 的 PTAB 申请宣告其无效 | 专利 · 授权框架 |
| merger | 合并原则 | 一种思想几乎只有一种写法时，这种写法也不受版权保护 | 版权 · 思想 |
| NDA | 保密协议 | 约定不披露、不滥用对方机密信息的合同 | 商业秘密、归属 |
| non-compete | 竞业限制 | 禁止离职后去竞争对手工作；加州基本不承认 | 归属 |
| NPE | 非实施实体 | 自己不生产产品、主要靠许可或起诉赚钱的专利持有人 | 专利 |
| novelty / non-obviousness | 新颖性 / 非显而易见性 | 专利两道核心门槛：不能已被公开，也不能是同行显然能想到的 | 专利 |
| objective indicia | 客观辅助因素 | 商业成功、被抄袭、长期难题等，用来证明发明不显而易见 | 专利 · 显而易见 |
| on-sale bar | 销售限制 | 申请前一年以上已经销售或要约销售，就不能再取得专利 | 专利 · 新颖性 |
| open-weight | 开放权重 | 模型参数可以下载，但许可证可能限制用途，不等于开源 | 体系、归属 |
| PCT | 专利合作条约 | 一份国际申请先占申请日，之后再进各国分别审查 | 专利 · 授权框架 |
| PHOSITA / POSITA | 本领域普通技术人员 | 法律假想的“普通同行”，用来判断是否显而易见、能否实施 | 专利 |
| prior art | 现有技术 | 申请日之前已向公众公开的技术信息 | 专利 · 新颖性 |
| prosecution history | 审查历史 | 申请人和审查员往来的全部记录；诉讼中会被拿来解释 claim | 专利 · 权利要求 |
| public domain | 公共领域 | 不受或不再受版权、专利保护，任何人都能用 | 激励 |
| reverse engineering | 反向工程 | 拆解合法取得的产品来弄清原理；一般不构成盗用商业秘密 | 商业秘密、版权 |
| § 230 | 第 230 条 | 平台一般不为用户发布的内容承担发布者责任 | 信息政策 |
| secondary meaning | 第二含义 | 描述性词语经长期使用，被顾客认作某个品牌 | 商标 |
| statutory damages | 法定赔偿 | 不用证明实际损失，按每部作品 750–150,000 美元计 | 版权 |
| substantial similarity | 实质性相似 | 去掉不受保护的部分后，剩下的表达是否足够像 | 版权 |
| summary judgment | 简易判决 | 关键事实没有争议时，法官不经陪审团直接依法裁判 | 导论 |
| TDM | 文本与数据挖掘 | 欧盟、日本等用专门例外允许的自动化分析，含 AI 训练 | AI · 比较法 |
| trade dress | 商业外观 | 产品或包装的整体样子，作为来源识别受商标法保护 | 商标 |
| transformative | 变革性使用 | 使用作品时赋予新目的或新意义，是合理使用第一因素的核心 | 版权 · 合理使用 |
| USPTO / Copyright Office | 美国专利商标局 / 版权局 | 审查专利和商标 / 登记版权的政府机关 | 全文 |
| utility patent | 实用专利 | 最常见的专利，保护功能性的技术方案 | 专利 |
| work made for hire | 雇佣作品 | 员工职务内创作的作品，法律上雇主就是作者 | 归属 |
