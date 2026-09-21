---
title: 斯坦福SGSI课程-Ethics for Academic
date: 2026-09-11
categories:
- 上斯坦福
tags:
- 课程体验
desc: 2026 SGSI 四讲笔记。
hidden: true
---

# 课程

## 学习

AI × university × learning。

### 问题

课上真正在追的，不是 Can students use ChatGPT。

表面：*How should students use AI in education?*

往下：*If AI can increasingly do the work that students used to do, what is the purpose of education?*

再往下：*If AI changes how humans learn, work, and produce knowledge, how should universities themselves change?*

中间一层层拆开问：

- What happens to the learning process?
- What is the purpose of an assignment?
- What is the purpose of assessment?
- What should universities teach?
- What is the purpose of a university?
- How should educational institutions respond to a changing labor market and society?

作业能不能外包，只是入口。后面一直在问：学习过程还在不在、考核在测什么、大学该教什么、大学是什么、机构怎么回应劳动市场和社会。

继续参与时课上给过的切入点：

- *If AI makes information and cognitive work increasingly cheap, should universities move away from evaluating the final product and focus more on the learning process?*
- *And if so, does that fundamentally change what a university is supposed to provide?*
- *What is education for when the cost of producing answers approaches zero?*

AI research 常问 *How can we make AI agents more capable?* 这堂课反过来问：*If AI agents become capable of doing increasingly complex cognitive work, what should humans learn to do?* *How should humans and institutions change when machines become better at learning and doing?*

<br>

### 替代

第一层 distinction：AI 是在帮助学习，还是在替代学习。

以前完成任务：问题 → 想 → 试 → 失败 → 搜 / 读 → 再试 → 理解 → 交答案。

AI 之后可以：问题 → 问 AI → AI 出答案 → 交。

两边最后可能是同一篇 essay、同一份 code。

<span style="background:#ffe58f">The output can be the same while the learning process is completely different.</span>

完成任务 ≠ 发生学习。一个 assignment 经常叠两件事：

**Function A: Production**

要交出去的东西：essay, code, presentation, research paper, problem set, analysis。

**Function B: Learning**

过程里发生的：思考、犯错、查资料、形成判断、学习表达、学习解题、搭 mental model。

AI 很容易把 Function A 自动化。用不好，Function B 一起没了。

<br>

### 挣扎

<span style="background:#ffe58f">The process of struggling with a problem may itself be part of education.</span>

CS 初学者自己 debug：报错 → 读 error → 看不懂 → 搜 → 改 → 又报错 → 重新理解 → 才过。慢。

Paste 给 AI：解释 + 建议代码 → Done。快。

productivity：AI 更好。learning：不一定。真正长能力的经常不是“最后跑通”，是中间那段 confusion → failure → reasoning → correction。

不是必须故意受苦。是：困难全部外包之后，能力从哪来，不清楚。

<br>

### 工具

课上不是 Don't use AI.

<span style="background:#ffe58f">AI can be a learning tool if it is used to support rather than replace the student's cognitive work.</span>

还算人在做认知工作：

- I don't understand this concept. Explain it in another way.
- Here is my proof. Point out where the reasoning is wrong.
- Here is my code. Don't rewrite it. Help me identify the bug.
- I wrote this paragraph. Point out grammatical problems.

Human → does the thinking. AI → provides assistance.

风险更大：Give me the solution / Write my essay / Do my homework / Solve this problem / Write the entire code / Generate my research paper.

AI 做认知工作，人收件。教育价值不是同一件事。

<br>

### 对象

<span style="background:#ffe58f">There may not be one universally correct AI policy.</span>

合不适用，看这项活动在练什么。*The appropriate use of AI depends partly on what skill the educational activity is trying to develop.*

**CS beginner**：基础还没长出来，Let AI write everything 会把训练抽空。

**Experienced programmer**：coding assistant、debugger、文档、brainstorm。有底去判断 AI 对不对。

**English learner**：自己出观点、自己搭 argument 和 structure，最后 polish grammar。和 AI generate the whole essay 不是一类事。

同一条校规套三种人，会歪。

<br>

### 核验

AI 可以生成看起来很对的错答案。convincing 本身是问题的一部分。

以后要紧的不一定是 *Can you produce information?* 而是 *Can you evaluate information?*

- Is it correct?
- Where did this claim come from?
- Can I verify it?
- What assumptions does it make?
- Is there another interpretation?
- What evidence supports it?

paradox：AI 越强，获取信息越容易；判断信息是否正确可能越重要。

```text
AI capability ↑ → information production ↑ → abundance ↑
→ verification 更重要
→ domain knowledge 可能更重要，而不是更不重要
```

完全没有领域知识，甚至听不出 AI 何时在胡说。

Person A 有基础：AI 说“这个算法该这样”，能觉得 complexity 不对。Person B 没有基础：AI 说 O(n²)，好的。B 过不了这一关。

<span style="background:#ffe58f">AI does not necessarily eliminate the need for knowledge.</span>

改的是 knowledge 拿来干什么：过去 Knowledge → produce answer；现在 Knowledge → evaluate, guide, verify, and use AI effectively。

<br>

### 考核

如果 AI 可以完成传统 homework，homework 还是不是好的 assessment？

旧模型：作业 → 学生写 essay → 教授打分 → Grade = evidence of ability。

现在：作业 → 学生 + AI 出 essay → 打分 → ???

这个 grade 到底测了什么？可能混着：

- student's writing ability
- student's prompting ability
- student's ability to edit AI
- AI's capability
- student's ability to recognize errors

测的是哪一项，作业设计里要说清楚。否则 assessment 失效，不是学生单方面作弊的问题。

<br>

### 作业

institutional question：大学为什么布置这个作业。

过去默认 Students need to write a 10-page paper。

AI 时代不能只问 How do we stop students from using AI?

而应该问：<span style="background:#ffe58f">Why did we ask students to write this paper in the first place?</span>

目标不同，形式才有理由变：

- 学 argument：oral defense + iterative drafts + discussion
- 学 research：research diary + source evaluation + methodology
- 学 coding：live coding + code explanation + debugging session
- 学 critical thinking：compare competing AI answers and identify flaws

<span style="background:#ffe58f">AI may force universities to redesign assessment around learning objectives rather than traditional formats.</span>

<br>

### 使命

问题不再是 Can students use ChatGPT，而是：

*What should a university be in an age when information and cognitive labor are increasingly automated?*

传统：University → Knowledge → Professor → Student。教授掌握知识，再传给学生。互联网已经改过一次。AI 再改：Information → Internet + AI → Available almost everywhere。

大学不能只靠 We have information that you don't have。

如果教育 = information transfer，互联网和 AI 会削传统课堂优势。如果教育 = transformation of the learner，大学仍有位置。

<span style="background:#ffe58f">Education ≠ information transfer.</span>

```text
Information → Understanding → Reasoning → Judgment → Practice → Experience → Wisdom
```

AI 在第一层极强。后面几层靠做，不靠看一份漂亮答案。

大学需要明确：*What are we actually trying to accomplish through education?*

- transmit knowledge?
- develop critical thinking?
- prepare people for employment?
- create researchers?
- develop citizens?
- create communities?
- preserve culture?
- enable social mobility?
- develop creativity?

答案不同，AI policy 会不同。只禁模型，解决不了使命没想清楚。

<br>

### 价值

信息到处都有之后，大学的价值可能转到哪？

**Community**：people learning together。讨论、合作、冲突、反馈。

**Mentorship**：教授不只是 information provider，可以是 mentor / guide / intellectual partner。

**Deliberate learning environment**：有人逼着练困难的东西、给反馈、不许跳过 struggle。

**Social and intellectual development**：不只是 learn facts，而是 become a certain kind of thinker / person.

这些不好被模型替代。是机构还在的理由，不是怀旧。

Learning by doing 仍在：programming, writing, research, experimentation, collaboration, presenting, debating。会看漂亮答案 ≠ *Can I do it myself when the situation changes?*

<br>

### 社会

AI 不只是教育技术。还会碰到 jobs, companies, labor markets, productivity, inequality, privacy, intellectual property, organizational structure, government regulation。

如果 1 worker + AI ≈ 过去几个工人的产出，productivity ↑。关键问题：*Who receives the benefit?* company / shareholders / consumers / workers / highly skilled employees / society，并不均分。

就业：*If AI makes workers more productive, does that mean fewer workers are needed?*

**Substitution**：效率 ↑ → 要的人 ↓ → 部分岗位消失。

**Complementarity**：效率 ↑ → 成本 ↓ → 需求 ↑ → 工作量反增。

现实里两件事可以同时发生，落在不同任务、不同人身上。

<span style="background:#ffe58f">“AI will destroy jobs” 和 “AI will create jobs” 都过粗。Which jobs? Which tasks? Which workers? Who captures the gains?</span>

为什么不同公司、行业、国家结果不同？不只是技术。还取决于 institutional structure, regulation, incentives, labor market, education system, ownership, culture, economic structure。

Same technology → different institutions → different outcomes. 从 technology 走进 sociology / political economy。

<br>

### 政策

课上一直在拆：AI is dangerous, therefore ban it. / AI is useful, therefore let everyone use it.

还要问：

- **What?** AI 用来做什么
- **When?** 学习过程的哪个阶段
- **Why?** 使用目的是什么
- **Who?** 什么类型的学生
- **Where?** 什么课程、什么学科
- **How much?** AI 做多少，人做多少

Policy = Context + Purpose + Student + Task + Learning objective。

大学该有多大程度的统一政策？standardization vs. local autonomy。

University-wide：简单、清晰、学生好懂。代价：不同专业学习目标完全不同。

Department / course-specific：更贴技能。例如 CS 可以辅助 debugging、某些考试禁止；English 可以 grammar check、不能产生 argument；Research 可以 literature search、必须 disclose。更合理，也更碎、更难执行、学生更容易困惑。

<br>

### 词

| Concept | 意思 |
| :--- | :--- |
| Learning vs. Production | 完成任务 ≠ 发生学习 |
| Learning by Doing | 做本身可能就是学习 |
| AI as Assistance | 可以增强学习 |
| AI as Substitution | 也可以替代认知过程 |
| Verification | 越强，判断真假越重要 |
| Domain Knowledge | 判断 AI 仍可能需要基础 |
| Assessment | 到底在测什么 |
| Institutional Design | AI policy 也是制度设计 |
| Substitution vs. Complementarity | 可能替代劳动，也可能增强劳动 |
| Purpose of University | 最根本不是能不能用 AI，是大学为什么存在 |

<br>

## 中立

Public trust, institutional neutrality, academic freedom.

### 问题

起点：higher education 的 public trust 在掉。课上问为什么掉、大学能做什么。

不是讨论“大学有没有政治”。是使命、机构发言、政治介入、学术自由、观点多元、信任、legitimacy 缠在一起。

整堂课三问：

**Mission**　*What is the fundamental mission of a university?* education? knowledge? research? social service? social justice? some combination?

**Representation**　*Who should be represented in the university, and in what ways?* demographic / socioeconomic / international / viewpoint / intellectual / methodological。*What causes underrepresentation?* structural? cultural? individual? combination?

**Institutional Response**　*How should universities respond to political and social pressures?* speak? remain neutral? protect individuals? intervene? research the issue? educate students? 还是仅在使命被直接牵及时发机构声明？

中间还会碰到：

- What is a university fundamentally supposed to do?
- When should a university speak publicly?
- Should the university actively shape society, or primarily provide the intellectual infrastructure through which society can understand and debate its problems?
- Does speaking make the university more effective, or compromise its institutional role?
- Why is public trust declining?
- What can be measured is not necessarily the same as what matters.
- Who is represented? Who is underrepresented? Does representation matter? What kinds of diversity matter?
- Should universities seek demographic diversity? Should they also seek intellectual or political diversity?
- Is demographic diversity sufficient?
- What makes a good classroom environment for difficult discussions?
- What should universities do about self-censorship?

<br>

### 使命

*What is a university fundamentally supposed to do?*

课上出现过的答案：educate students；create / preserve / disseminate knowledge；conduct research；prepare people for society；contribute to society；promote social progress；cultivate intellectual communities；potentially address social problems。

全写进使命，就无法判断某次表态、某项政策算不算越界。Mission statement 是问责标准，不是海报。

Mission → institutional goals → institutional actions → accountability。

声称在创造并传播知识，就可以问：有没有在创造？有没有在传播？学生有没有在学？知识有没有在保存？政策是否服务于这些活动？使命含糊，评价和“该不该说话”都会漂。几乎任何活动都能被正当化。

斯坦福作例子。较宽的旧表述：teaching, research, education, scholarship, service / contribution to society, preparing people to serve society。较新、更短的表述：creating, disseminating, and preserving knowledge through research and education.

使命缩短之后课上问：

- Does a shorter mission make the university easier to understand?
- Does it make the university easier to hold accountable?
- Does it make institutional decisions easier to evaluate?
- Does it remove some of the university's broader social commitments?
- Does it make it harder to justify activities that are not directly related to research and education?

知识可有内在价值（值得保存，不论立刻有用），也可有工具价值（解题、技术、政策、生活）。使命往往两头都沾。

Create：research, discovery, experimentation, scholarship。Disseminate：teaching, publications, conferences, public communication。Preserve：libraries, archives, scholarship, institutional memory。

按这三项问，比只问排名具体：新知识有没有；分享得怎样；怎么保存；学生学得怎样。

<br>

### 发言

使命定了，才能问 When should a university speak publicly。

假如中心使命是 creating, preserving, and disseminating knowledge，一次有争议的政治声明就要问：

- Is the statement necessary for the university's mission?
- Is the issue directly related to academic freedom?
- Does the statement protect the university's ability to conduct research?
- Is the university acting as an institution or simply expressing the views of its members?
- Does speaking improve the university's ability to fulfill its mission?
- Or does institutional political activity distract from the mission?

两层不要混。

**Individual speech**：学生、教师、研究者自己的观点。外面听成某个人的看法。

**Institutional speech**：大学、校长办公室、官方声明。外面听成 This is what the university believes. / “斯坦福认为……”

<span style="background:#ffe58f">Institutional speech and individual speech are different.</span>

Academic freedom：探究、发表、教学，不受不正当压制。保护的是 inquiry，不是学校的政治立场。Why disagreement is necessary：人人必须同意机构，就没有探究空间。异议不是故障，是条件。

<br>

### 中立

Institutional neutrality：对有争议的公共政治议题，机构不轻易以大学名义站队。

<span style="background:#ffe58f">中立 ≠ 个人必须中立。机构不站队，是给成员留出不同意机构、也互相不同意的空间。</span>

正向：机构不背书每一个争议立场 → 多种观点仍可能 → 师生可以和机构、彼此争执 → academic freedom 更有空间。

反向：机构不说话 → 被读成冷漠，或默认现状 → 中立是不是一种政治选择？

<span style="background:#ffe58f">Is neutrality itself a political choice? 沉默也有社会后果，所以中立本身可被争。</span>

何时开口，判据是和使命的距离，不是热度。

与核心使命无关的全国 / 国际政治：倾向不说。理由：不是大学的角色去裁定每一个政治问题；声明未必推进教学或研究；可能不必要地把大学绑到一个政治立场上。

直接碰到本校运行、academic freedom、研究、教学、治理、学者能否做研究、学生能否学习、机构独立性：可能必须说。

The more directly an issue concerns the university's own mission and functioning, the stronger the justification for institutional action or speech.

两种大学想象并排放着。

**Civic / social institution**：contribute to public life；address social problems；promote social justice；respond to crises；defend values；participate in public debate。

**Intellectual institution**：create / preserve / disseminate knowledge；educate；provide a forum for inquiry；protect academic freedom。

中央张力：<span style="background:#ffe58f">Should the university actively shape society, or primarily provide the intellectual infrastructure through which society can understand and debate its problems?</span>

为什么有的大学仍想发声：protecting members；defending vulnerable populations；expressing institutional values；addressing injustice；responding to major social events；showing solidarity；maintaining relationships with students and faculty；fulfilling perceived civic responsibilities。

课上追问：Does this make the university more effective, or does it compromise its institutional role?

官方声明会制造 This is what the university believes。大学里人很多、意见不同。机构发言可能：让内部分歧更不可见；暗示某种观点代表整个共同体；让少数观点显得不合法；加强机构和某种政治认同的绑定。

极化过程可以是：大学发政治声明 → 被看成站队 → 不同意的人不信任 → 对 expertise 的信任下降 → 机构更要辩护立场 → 更多机构政治活动 → 极化更厚。课上不把信任下降收成单一原因。还可能有学费、教育价值疑虑、极化、意识形态同质观感、对专家 / 精英的不信任、对大学优先事项的分歧、更大的社会变化。

社会正义这一侧：大学有责任回应不公。另一侧：可以通过研究不平等、发表证据、教育、提供机会、分析政策来贡献，不必官方政治倡议。

<span style="background:#ffe58f">Social impact does not necessarily require institutional political advocacy.</span>

与其告诉社会该持何立场，不如生产让社会能更好决策的知识。Research as social contribution：teaching、public scholarship、evidence、innovation。

空喊 We value free expression / academic freedom 不够。要有政策、程序、规范、保护、日常做法。价值 → 规则 → 激励 → 个人行为。保护可以长什么样：反报复、程序透明、faculty governance、何时发声的标准、课堂讨论规范、严重问题的报告渠道。

<br>

### 信任

*Public trust in higher education has declined over time.* 重要问题是 Why?

**Political**：polarization；perception of ideological bias；institutional political statements。

**Institutional**：cost；governance；administrative decisions；perceived lack of accountability。

**Cultural**：changing attitudes toward experts；broader distrust of institutions；changing social values。

**Performance-related**：universities 是否真的在交付教育和社会价值。

信任还跟前后是否一致有关。说 We value academic freedom，别人看 Does the institution actually protect academic freedom。说 We value diversity，别人问 How does the institution behave when people disagree。

Trust depends partly on the consistency between institutional principles and institutional behavior.

排名（QS、U.S. News）给出好比较的数：research output, publications, citations, reputation, faculty, internationalization, resources。课上问：排名有没有抓住大学的 actual mission。

<span style="background:#ffe58f">What can be measured is not necessarily the same as what matters.</span>

可以优化 publication count、citations、grants、rankings。更宽的目标可能是 educating students、intellectual independence、serving society、useful knowledge、communities。Metrics 是 proxy。Proxy 会自己变成目标。

AI 还能生成文本、代码、草稿、文献综述，甚至参与研究产出。传统 academic productivity 更难读。课上问：

- What counts as original work?
- What counts as a genuine research contribution?
- How should publications be evaluated?
- Will publication counts remain meaningful?
- How should human intellectual contribution be measured?
- Could AI make quantitative academic metrics easier to manipulate?

产量上升 ≠ 探究变好。激励如果只认研究，课堂和指导被挤掉不是意外。

不同类型大学可以有不同使命：research universities；liberal arts colleges；community colleges；religious universities；public universities。用同一套排名去量，会错位。Diversity can exist between institutions, not only within institutions. Institutional pluralism：不同使命、教育模型、政治文化、宗教传统、智识传统、研究优先，学生可以选择。

<br>

### 多样

Who is represented? Who is underrepresented? Does representation matter? What kinds of diversity matter? Should universities seek demographic diversity? Should they also seek intellectual or political diversity?

**Demographic**：nationality, socioeconomic background, gender, race, 其他人口特征。

**Viewpoint**：political / ideological perspectives, intellectual traditions, competing theories.

可以 demographically diverse but intellectually homogeneous。于是问：Is demographic diversity sufficient?

Suppose Group X is underrepresented in field Y。不必开场只钉一种原因。

Structural：institutional rules, admissions, hiring, promotion, funding, incentives, barriers, organizational structure。

Cultural：social norms, family expectations, professional culture, attitudes toward the field。

Individual：preferences, interests, preparation, personal circumstances, career choices。

这些解释不必互斥。先看 pattern，并列假说，再看证据。若事先认定 “The problem is caused by institutional structures”，研究就只会盯结构；认定 “The problem is cultural”，会漏掉制度机制。同一原则：multiple causal mechanisms, alternative explanations, empirical evidence, competing methodological approaches。

Intellectual pluralism：同一问题可以用不同理论、学科、方法、假设去碰。不是每种解释同等有效。Different approaches should be allowed to compete through evidence and argument。

Formal freedom：规章允许说话。Substantive freedom：人敢说。成绩、推荐信、同辈、职业风险、belonging，都会变成自我审查。

<span style="background:#ffe58f">Formal freedom does not eliminate self-censorship.</span>

Reasons 课上点过：fear of social judgment / disagreement / damaging relationships / being perceived negatively / consequences from professors / peers / concern about belonging。课堂文化会放大或压住这一点。

公众不信任，机构能做的：

1. Clarify institutional mission：做什么，不宣称做什么
2. Clarify institutional neutrality：何时说话、何时沉默、什么算机构议题
3. Protect academic freedom：争议研究、难问题、异议
4. Increase intellectual diversity：viewpoint / methodological / competing explanations
5. Improve transparency：决策、政策、理由
6. Improve classroom discussion：respect, curiosity, listening, disagreement, intellectual humility

自我审查：规范说出口、反报复、区分 disagreement 和 harassment、鼓励提问、教师不因正当异议惩罚。同时避免把课堂做成 people are protected from hearing ideas they dislike。

<span style="background:#ffe58f">freedom to participate + freedom to disagree + basic respect</span>

<br>

### 讨论

What makes a good classroom environment for difficult discussions?

目标不是 everyone agrees，是 people can disagree productively。

Curiosity：不是 How do I prove this person wrong，而是 Why does this person believe this。还可以问 What evidence led you there / What experience shaped your view / What assumptions are you making / What would change your mind / Is there another interpretation。Curiosity turns disagreement into inquiry.

Open-mindedness 不是 Every opinion is equally correct，是 Being willing to seriously consider evidence or arguments that could change your mind。disagreement 和 open-mindedness 兼容；skepticism 和 openness 兼容；改主意不是智识失败。

Intellectual humility：承认不确定、可能错。

Listening 操作：Listen → Identify the argument → Restate it → Verify → Respond。Restatement：驳之前能复述对方论证，减少打稻草人。Reframing：换一种表述，看是不是同一分歧。

Respecting someone does not require agreeing with them. Listening does not mean accepting the argument. Being open-minded does not mean abandoning standards of evidence.

**Intellectual discomfort**：I strongly dislike / disagree with this idea。  
**Hostile environment**：人身攻击、威胁、无法参与。

严肃探究会让人不舒服。要保护的是人，不是保护人不碰到挑战。

好讨论：

- Curiosity: Why do you think this?
- Listening: Let me understand your argument.
- Restatement: If I understand correctly…
- Clarification: Is that what you mean?
- Evidence: What evidence supports that?
- Alternative: Could there be another explanation?
- Disagreement: I disagree because…
- Openness: What evidence would change my mind?

避开：贴政治身份、预设恶意、打断、攻人、把反对当敌意、把自己的文化规范当普遍、驳一个没人主张的立场、把不适当成该停的证明、认定只有一种解释。

跨文化：权威、提问、反对、参与、等级。有的系统里质疑教授很罕见；有的系统里挑战教授算正常 academic participation。不要把参与风格自动读成笨、不感兴趣、不开放。可能只是教育规范不同。

斯坦福国际学生多，对“什么叫尊重、能不能直接反对、政治意见能不能进课堂、什么算贡献”，预设差很远。规范需要说出口：允许反对、鼓励提问、批观念不批人、可以改主意、不确定可以。

Classroom design as institutional design：Values alone are insufficient. Institutions need structures that support those values.

方法上这堂课也在练：<span style="background:#ffe58f">Do not decide the answer before examining competing explanations.</span>

观察 pattern → 多种假说 → 证据 → 结构 / 文化 / 个人 → 比较 → 必要时改口。和学术自由是同一原则。

<span style="background:#ffe58f">The purpose of a university is not simply to produce agreement. It is to create conditions under which people can pursue truth, challenge assumptions, produce knowledge, and disagree without shutting down inquiry.</span>

<br>

## 课堂

Graduate education, inclusion, academic community.

### 问题

开场就很大：

- 大学究竟是什么？
- Graduate education 究竟应该培养什么？
- 希望自己所在的 academic community 是什么样的？

不只是如何完成学位、发表、找工作、提高个人竞争力。还在问：如何一起学、如何合作、如何支持其他学生、如何与 faculty / instructors / TAs 相处、学校和个人分别承担什么责任、希望建立怎样的学术文化。

好课堂不能只问 How much content did the instructor cover。还要问：

- 学生有没有参与？
- 有没有机会提出问题？
- 不同背景的学生是否能够参与？
- 是否愿意表达不同意见？
- 是否能够犯错？
- 是否能够从其他学生身上学习？
- 是否形成了 intellectual community？

Participation 到底是什么：*What does participation actually mean?*

University 的责任：*How much responsibility does an institution have for students?* 学术支持、经费、住房、心理、人身安全、职业、mentoring、community、文化适应、国际学生支持——大学该管哪些？如果学生遇到任何问题，大学是不是都应该负责？个人生活、家庭、财务、心理压力、社交、学术，介入到什么程度？

Inclusive education 不只是 Everyone is welcome。还要问：谁知道这个机会？谁知道资源在哪里？谁知道该向谁求助？谁敢提问？谁知道 office hours 干什么？谁认识 faculty？

Excellence vs inclusion 是不是必然冲突？Research vs teaching 激励偏向哪边？Support 会不会滑成代做决定？helping others 是额外负担还是共同体日常？

<br>

### 发生

<span style="background:#ffe58f">Graduate education is not simply something that happens to students.</span>

不是学校给、学生接。学生也在造这个 community。

传统模型：Professor → knowledge → students。讲、听、作业、考试、成绩。实际课堂同时还是 social / intellectual / collaborative space。学生之间的互动是学习的一部分。

<br>

### 参与

传统很容易把 participation 理解成举手、发言、回答、主动讲话。

有人需要更多时间思考；英语不是第一语言；不习惯大课堂发言；更倾向书面；小组里更积极；会先听；不确定自己的问题是否“足够好”。

Silence does not necessarily mean disengagement。没说话 ≠ 没听、没想、没兴趣、没理解。提问、听、小组、书面、课后、分享资源、帮同学，都算。

Inclusive classroom：不是降低标准。是减少和能力无关的进入壁垒。要分开：

- Access：可以进入这里
- Participation：实际上有能力在这里参与
- Belonging：觉得自己属于这里

Hidden advantages：知道 office hours 怎么用；会给教授写信；知道怎么找研究机会；知道隐规则；已有关系网；有经济缓冲，失败起得来。Merit 话语如果不管这些，会把结构差异说成个人不够努力。课上明确：不要把所有问题都归结为个人不够努力。Social network 本身就是一种隐性优势：认识谁、谁会介绍、谁提前知道机会。

<span style="background:#ffe58f">Resource existing ≠ resource accessible.</span>

A resource is not useful if students don't know it exists or cannot access it. fellowship, counseling, academic advising, teaching resources, research programs, student groups, mentoring programs, financial support——可以都在。学生仍可能不知道、不理解、不会申请、不知自己合资格、不知该找谁。Institution 要管 visibility + accessibility，不只 availability。

International students、不熟悉美国体系的人：office hours 是什么、怎么和 professor 沟通、怎样找 research、怎样问问题、怎样表达 disagreement、怎样建 academic network、哪些该自己解决、哪些可以向学校求助。Unwritten rules matter.

Hidden curriculum：怎样与教授建关系、怎样进 research group、怎样找 mentor、怎样用 office hours、怎样申请机会、怎样在讨论里表达不同意见。手册里没有。有人因过去经验已经知道，有人需要被明确告知。

Belonging：不只是 You are officially a Stanford student，而是 I feel that I belong here。被看见、被尊重、能够参与、可以把失败讲出来。If I don't see people like me in this field, can I imagine myself belonging here?

<br>

### 责任

课上没有 Everything is the university's responsibility，也没有 Everything is the student's responsibility。在讨论两者关系。

Individual：主动找资源、提问、找 mentor、管理时间、做决定、对自己的教育负责。

Institutional：提供资源、让资源可见、建立公平 access、创造支持性环境、降低不必要 barriers、让人知道在哪里可以获得帮助。

Student agency：做决定、探索、试错、提出自己的目标、找机会、建关系、主动塑造 graduate experience。学生不是被动容器。

<span style="background:#ffe58f">Support ≠ making decisions for students.</span>

I know what is best for you → paternalism。成人学生可以做决定，机构 / faculty / mentor 却代选“正确道路”。

较认的顺序：Ask → Listen → Understand → Offer resources / 选项 → Student chooses → follow up。

Mentoring 同构。先问 What are you trying to do / struggling with / need，再给选项，不先塞答案。

Faculty 晋升、经费、声誉主要认 research。教学、office hours、mentoring 若不计“工作”，课堂和共同体被挤掉不是意外。Research vs teaching：时间零和。制度怎么算，人就怎么走。

Teaching 本身是 intellectual work：设计、解释、诊断误解、给反馈。不是研究之余的服务。

Instructor：内容 + 把空间撑住（谁能说话、反对会不会被罚）。

TA：课、作业、情绪、准入的前线。需要培训和权责边界，不能默认会做研究就会教。学生常把 TA 当 What is the answer 的窗口；更好的角色是帮对方自己走到答案。

Teaching relationship 里的 respect：不是讨好，是把对方当能思考的人。

Expertise vs humility：有专业权威，同时承认不确定、会错、学生可能看到教师没看到的。

Academic disagreement 针对论证和证据，不针对人格。课上问的不是只 Who is correct，还有 What is interesting。Being willing to change your mind 是规范，不是软弱。

Academic community 是 collective。每个人都在影响 academic culture，包括沉默、排他的笑话、把机会藏起来。

Information sharing 是 community responsibility。不要只问 How can this help me，还问 Who else can I connect with。机会、经验、坑，独占会复制不平等。Social network 双面：支持与信息；也排斥、封闭、只在圈内传。

Small interventions：分享一个链接、告诉新同学某个资源、介绍一个人、分享申请经验、回答一个问题、告诉别人某个机会、改一句 syllabus。不必等整场改革才动手。Institutional change 仍然要动激励、资源、政策、谁被听见。

helping others 课上不把它理解成额外负担。很多帮助很小，但对刚进陌生体系的人可能很关键。分享信息、介绍、听 ≠ 替对方活、替对方决定。

<br>

### 研究

对“做研究”的再想：不只是产出。还有怎么合作、怎么带人、怎么处理失败、怎么对待竞争者。

Competition vs community：竞争可以出活；只剩竞争，信息与支持会停。

Excellence vs inclusion：课上不接受两者必然相反。壁垒和能力不是一回事。

Efficiency vs reflection：一直产出，没有停下来问还要不要这条路。

Track vs exploration：标准路径（资格考、论文、学术市场）清楚，但不是唯一合法人生。Graduate school 不一定只有一种成功：学术、工业、教学、公共、离开研究，都可以是完整选择。

Reflection / check-in：停下来问是不是还要走。不是自我批评比赛，是 agency 的一部分。

学生和 institution 的关系，至少三种：消费者（买服务）、被管理对象、共同治理的成员。三种给出三种责任分配。

Academic culture 不是固定的。改文化要现实主义：激励和权力还在，faculty 也是制度中的人，不是制度外的道德完人。

Authority vs relationship：评价权、经费、推荐信是权力；信任是关系。叠在同一个人身上。

好的大学社区，课上散落的特征：Access / Participation / Belonging / Agency / Support / Responsibility。能问、能反对、能找得到资源、失败起得来、指导不是运气、隐规则有人讲。

圈可以记：Individual → Classroom → Community → Institution → 再问个人能做什么。

Graduate education is not only about what one learns or produces. It is also about the kind of academic community being created, how people support one another, and how they take responsibility for their own experience and others'.

<br>

### 词

- academic community 学术共同体
- inclusive classroom 包容性课堂（不是降标准）
- participation 参与（≠ 举手）
- belonging 归属
- student agency 自主
- hidden curriculum 隐性课程
- hidden advantages 隐性优势
- access vs availability 可及 vs 存在
- paternalism 家长式
- faculty incentives 教师激励
- small intervention 小干预
- information sharing 作为共同体责任的分享
- student-centered education
- mutual respect
- intellectual community

张力（课上就是并排放的，不是要选边赢）：

- research vs teaching
- individual vs institution
- support vs agency
- excellence vs inclusion
- expertise vs humility
- track vs exploration
- competition vs community
- efficiency vs reflection
- correctness vs creativity
- authority vs relationship

<br>

## 指导

Mentorship.

### 问题

开场：*What does mentorship mean to you?*

看起来普通，实际很复杂：

- 什么样的关系可以被称为 mentoring？
- Mentor 和 mentee 之间究竟是什么关系？
- Mentorship 是单向的知识传递，还是一种更复杂的关系？
- Mentor 是否一定比 mentee 更资深、更有权力？
- Mentorship 是否一定发生在正式的 academic setting？
- 一个好的 mentor 到底应该做什么？
- 为什么有些人能够获得很多 mentorship，而另一些人很难获得？
- Mentorship 与 identity / power / social-cultural context / institution / career / personal life 怎么缠在一起？

不要收成：一个更聪明、更资深的人告诉年轻人该怎么做。看成一种长期形成的关系和实践。

课上还让人回头问：

*Who has been a mentor to you?*

- 为什么这个人产生影响？
- 具体做了什么？
- 是给了 advice？
- 还是相信对方？
- 还是给了机会？
- 还是让人看到一种可能的生活？
- 还是帮助理解一个 community？
- 还是困难时在？

不只问 Who mentored you，还问 *Who have you mentored?* 给 junior 建议、帮同学理解制度、分享 research experience、帮别人准备 application、讲自己的经历、鼓励别人继续——这些都可能已经是 mentoring。很多人没意识到。

找人之前先问需要什么，不是找最厉害的那个：

- 现在卡在什么？
- 需要哪类帮助？
- 希望发展什么能力？
- 要 career advice 还是 research advice？
- 要不要懂自己 context 的人？
- 要不要 role model？
- 要不要能带进某个 community 的人？

More mentoring is always better 不成立。还要问：What kind of mentoring? By whom? For whom? Toward what ends?

<br>

### 关系

传统：senior → junior，knowledge / advice / information → mentee。教授带博士、PI 带组里学生、教怎么发论文、申请、grant。

不只是 information transfer。还涉及：理解自己是谁；理解领域文化；进入陌生 community；弄懂 institution 怎么转；困难时的支持；看见原来没看见的可能性；人生 / 职业决定。

<span style="background:#ffe58f">Mentorship can shape a person's understanding of themselves and their possible futures.</span>

Relational：影响经常不来自正式 meeting。一次谈话、一封 email、一句 you can do this、在想放弃时有人在、介绍另一个人、让人第一次觉得某条路是可能的。难量化。

和个人生活分不开。真正要紧的对话有时不是 paper 做得怎么样，而是 What kind of life do you want / Why continue / Do you actually like this.

Mentor 改的常常不是下一步，是 worldview。从“只能做 A”到“B、C、D 也可能是选项”。identity-shaping。

人在学术环境里同时是 student、researcher、家人、伴侣、移民、家长、照护者……这些会影响时间、什么算 important、怎么定义 success。只看见 student / researcher，看不见 whole person，处境会读错。

Academic culture 很容易让人觉得必须一直工作才 committed：paper、实验、deadline、会议、grant、申请、教学、会。指导要能谈：工作意义是什么、愿意牺牲什么、什么是长期可持续的。不能把永远工作当成唯一正确生活方式。

Explicit rules vs implicit rules：手册写的是前者。后者包括怎么开口、怎么维持关系、什么能问、什么会显得不 committed。Mentor 经常是把隐性规则讲出来的人。Hidden curriculum 同样作用于“如何找导师、如何开口、如何维持关系”。

传统 success = publications + awards + prestigious positions + productivity。一个人可能更在乎 intellectual freedom、家庭、健康、community、教学、公共服务、钱、弹性。好的 mentor 不宣布唯一道路，而是帮对方问 What kind of life do you want.

职业上：academia / industry / entrepreneurship / government / nonprofit。别人都这么做 ≠ 适合自己。是自己的目标还是别人期待；喜欢工作本身还是已经投入太多所以不走。

<span style="background:#ffe58f">Past investment ≠ future obligation.</span>

年数、论文、training、进了某个 program，不能自动决定未来。Mentorship 的一件事就是让人停下来问：如果重新选，现在还会走这条路吗。

<br>

### 瞬间

<span style="background:#ffe58f">Mentoring moment：不必是数年正式关系。课后十分钟、senior student 一句关键建议、有人在自我怀疑时认真听，也可以长期转向。</span>

Mentorship can happen through moments, not only through formal relationships.

场合：office hours、coffee、午餐、走廊、课上、组会、conference、email、社交、学生组织、系里活动。不必每月固定 30 分钟才算。不要只找 official mentor。

I'm thinking about X. Could I get your perspective? 有时就够打开。

创造 moment：主动谈、把问题说具体、表达兴趣、要 perspective、分享经历、问对方经历、合适时 follow up、建长期联系。

过程更像：Encounter → Conversation → Trust → Relationship → Reflection → Growth → Independence → Mentoring others。不是找到 mentor → 问题解决。

<br>

### 判断

好的 mentor 往往不是 the person who tells you what to do，而是 the person who helps you figure out what you want to do.

<span style="background:#ffe58f">Advice ≠ Mentorship.</span>

Advice 有用。Mentoring 更靠 listening、questioning、reflection、context、relationship、帮对方长出判断。

会问的那种问题：What do you want? What matters? Why are you doing this? What are you worried about? What would success look like for you? What options haven't you considered? What kind of life do you want? What would you choose if you weren't worried about what others think?

不是代做决定。

Mentoring should be personalized。当年每天 12 小时、后辈也应当——personality、identity、family、culture、goals、钱、兴趣都可能不同。指导不是复制自己的 life trajectory。

目标是 autonomy，不是制造依附。Mentorship 的最终目标之一是让人能够离开 mentor。被指导的人后来成为指导者：generative cycle。不是所有权。

Care：把对方当人，不只当劳动力或论文产量。Care 也需要边界：尤其叠着评价、雇佣、推荐时。互惠不是互相算账，是关系能持续的条件。

<br>

### 权力

传统：mentor = senior，mentee = junior。knowledge / experience / institutional / professional / 有时 financial / evaluative 都不对称。

同一人可同时是 mentor、teacher、PI、evaluator、recommender、employer。Mentee 很难真说 I disagree with you。尤其还握着成绩、研究机会、经费、推荐信、作者、雇佣、签证相关支持、网络。

Good mentorship 要看见 power dynamics：不滥用、不把 personal preferences 强加、不要求畸形 loyalty、不把 mentoring 变成 control、不强迫按自己的路生活。

Mentor 是不是朋友：可以亲近，甚至变成长期友谊。同时是 supervisor / evaluator / employer 时，professional boundaries 和 conflicts of interest 更硬。

Mentoring is not necessarily one-directional. 新技术、新观点、新世代经验、新方向，mentor 也可以学。Mutual learning，不只是等级教学。

为什么有人很容易得到指导、有人很难：不只是够不够主动。前面可能有制度、文化、语言、代表性、信息、权力。Why don't you have a mentor → 不够主动。这句把结构抹掉。反过来把一切归因于结构、取消 agency，也不对。

非正式网络省信息，也复制 privilege、hierarchy、exclusion、既有文化。完全依赖 informal mentorship 可能复制原有不平等。正式项目可以扩大 access；真正有效仍靠 trust、care、respect。Formal program ≠ automatically good mentorship。

Social capital：经网络得到信息、机会、资源的能力，分布不均。Institutional culture 可以促进也可以阻碍：教授有没有时间、学生能不能碰到、指导算不算工作、机会是否均等。

<br>

### 网络

课上反对必须找到 THE ONE mentor。

可以同时有：

- Academic：研究、课、智识
- Career：工业 / 学术、决定、networking
- Personal：生活、价值、balance
- Peer：同学、senior students
- Community：机构怎么转、文化

更合理的是 mentoring network。一个人补不齐所有需求，不要理想化成一位教授解决全部问题。

Peer mentoring：距离近、更懂当下、权力差更小、更容易说真话。Mentor 不一定年长很多。

相似身份有时有助于 belonging。不必须相同：愿听、尊重目标、有 perspective，差异也可以有价值。Similarity can help, but difference can also be valuable.

Mentor：理解问题、发展能力、思考选择、advice、reflection。帮人理解系统。

Sponsor：用自己的位置和网络给对方创造机会。推荐、介绍、提名、拉进项目、在对方不在场的房间里说话。帮人获得 access。

<span style="background:#ffe58f">Mentor helps you understand the system. Sponsor may help you gain access within the system.</span>

很多职业机会并不完全公开竞聘：informal recommendation、私人网络、邀请、介绍、提名。只有 skills 没有 access，仍然受限。所以两种都重要。

<br>

### 结构

问题不只是这个教授是不是好 mentor。Institutional structures 是否让 mentoring 更容易发生。

<span style="background:#ffe58f">Mentorship 可以扩大机会、破信息壁垒；也可以复制既有规范、等级、主导文化。</span>

Positive：扩大机会、破信息壁垒、看见可能的生活。Negative：复制既有规范、等级、主导文化、把个人偏好当成义务。

Structure：institution、culture、hierarchy、norms、resources、access、networks。  
Agency：选择、开口、找人、建关系、改方向。

Mentorship 卡在中间：Institution shapes mentoring opportunities, while individuals also actively create mentoring relationships.

两端并放（课上就是这样问，不是填空选正确答案）：

| 问题 | 两端 |
| :--- | :--- |
| Mentorship 是什么 | Information transfer ↔ Relationship |
| Mentor 的角色 | Advice ↔ Facilitation |
| 关系结构 | Hierarchy ↔ Mutual learning |
| 目标 | Success according to institution ↔ Person-defined life |
| 机会来源 | Informal ↔ Formal |
| 影响因素 | Individual agency ↔ Institutional structure |
| 身份 | 职业角色 ↔ whole person |
| 规模 | 一位 mentor ↔ network |
| 相似 | 同质 ↔ 差异 |
| 支持 | mentoring ↔ sponsorship |
| 结果 | 依附 ↔ 自主 |
| 社会作用 | 赋权 ↔ 复制结构 |


# 杂项

本来不太想去了，签证也迟迟没下，报名后withdraw了，但是Anne主动给我发邮件问我要不要来，特别感动。后面也一直很supporting，鼓励我讲话(x) Catlin也很supporting，大家都很好。小班研讨课，一直需要讲话。算是我这种没有学过人文社科、没上文理学校的学生，一场为期4天的体验吧，据说美国文理学院本科就是这样的。
每天包早饭午饭，自己随便拿，比orientation吃的好，午饭是大家坐一起聊天。同学们都挺友好的，国际生也不少，虽然因为文化差异和英语不行，我感觉很难融入。
探讨了一些人文社科，有guest speaker，有一些insight但不多，因为最后一定会回到无力改变的制度问题、经济基础等等，不过能意识到、说出来已经很好了。一方面，让我感受到在这个AI时代还有很多人在乎这些东西，很难得；另一方面，确实需要我们这些有“privilege”的人，需要Stanford这样的机构，有人stand out。
还练习了口语，第一天阿巴阿巴，最后一天至少能讲了，虽然还是支离破碎。还练习了听力，虽然有点吃力，但是其实美国人讲话还可以，印度人讲话我真的没水平听懂、、、
总之AI时代，多在乎一下具体的人吧。