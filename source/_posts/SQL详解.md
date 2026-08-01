---
title: SQL详解
date: 2024-06-30
categories:
  - 上浙大
tags:
  - DB
  - SQL
desc: 数据库 SQL 专项：DDL/DML、JOIN 全家、嵌套与相关子查询、NOT EXISTS 全称量化、窗口与 CTE、视图触发器权限。
---


# 语法

## 总览

### SQL 语言分类

| 缩写 | 全称 | 作用 | 常见语句 |
| :--- | :--- | :--- | :--- |
| DDL | Data Definition Language（数据定义语言） | 定义/修改模式与对象结构 | `CREATE`、`ALTER`、`DROP` |
| DML | Data Manipulation Language（数据操纵语言） | 增删改查表中的数据 | `SELECT`、`INSERT`、`UPDATE`、`DELETE` |
| DCL | Data Control Language（数据控制语言） | 权限与安全控制 | `GRANT`、`REVOKE` |
| TCL | Transaction Control Language（事务控制语言） | 事务提交与回滚 | `COMMIT`、`ROLLBACK`、`SAVEPOINT` |

有的教材把 `SELECT` 单独称为 DQL（Data Query Language，数据查询语言）；本文仍按习惯归在 DML 一侧讨论。

### 定位

本文是 [数据库系统笔记](/数据库系统/) 的 SQL 独立册：例子密度按作业 + 面试组织，覆盖复杂嵌套与“全部 / 仅有”类语义。理论侧（关系代数、优化、事务）见主笔记。

参考：

- [菜鸟教程 SQL](https://www.runoob.com/sql/sql-tutorial.html)
- [菜鸟教程 JOIN](https://www.runoob.com/sql/sql-join.html)
- [牛客 SQL 篇](https://www.nowcoder.com/exam/oj?tab=SQL%E7%AF%87&topicId=199)

### 示例模式

全文默认使用下列模式：

```text
student(sid, name, age, gender, department)
club(cid, name, supervisor)
member(sid, cid, date)

instructor(ID, name, dept_name, salary)
department(dept_name, building, budget)
teaches(ID, course_id, sec_id, semester, year)
course(course_id, title, dept_name, credits)
takes(ID, course_id, sec_id, semester, year, grade)
```

> [!INFO]+ 执行顺序（理解用）
>
> `FROM` → `WHERE` → `GROUP BY` → `HAVING` → `SELECT` → `ORDER BY`。
> `WHERE` 不能使用 `SELECT` 列表别名；`HAVING` 可使用聚集结果。

<br>

## DDL

（Data Definition Language，数据定义语言：建表、改表、删表、建索引等，管的是“结构”不是行数据。）

### 建表

```sql
CREATE TABLE student (
    sid         INT,
    name        VARCHAR(50) NOT NULL,
    age         INT,
    gender      VARCHAR(10),
    department  VARCHAR(50),
    PRIMARY KEY (sid),
    CHECK (age IS NULL OR age >= 0)
);

CREATE TABLE club (
    cid         INT PRIMARY KEY,
    name        VARCHAR(50) NOT NULL,
    supervisor  VARCHAR(50)
);

CREATE TABLE member (
    sid   INT,
    cid   INT,
    date  DATE,
    PRIMARY KEY (sid, cid),
    FOREIGN KEY (sid) REFERENCES student(sid)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (cid) REFERENCES club(cid)
);
```

常见类型（几乎所有类型允许 `NULL`，除非声明 `NOT NULL`）：

| 类型 | 含义 |
| :--- | :--- |
| `CHAR(n)` | 定长字符串，长度固定为 $n$；不足时通常右侧补空格 |
| `VARCHAR(n)` | 变长字符串，最长 $n$；按实际长度存，更省空间 |
| `INT` | 整型（机器相关的有限整数子集） |
| `SMALLINT` | 小整型，取值范围通常小于 `INT` |
| `NUMERIC(p,d)` | 定点小数：一共 $p$ 位有效数字，其中小数点后 $d$ 位 |
| `REAL` / `DOUBLE` | 单精度 / 双精度浮点数（精度机器相关） |
| `FLOAT(n)` | 浮点数，精度至少 $n$ 位 |
| `DATE` | 日期（年-月-日） |
| `TIME` | 时间（时:分:秒） |
| `TIMESTAMP` | 日期时间戳（日期 + 时间） |
| `BLOB` | 二进制大对象，存图片、音视频等二进制数据 |
| `CLOB` | 字符大对象，存很长的文本 |

### 改表与删表

```sql
ALTER TABLE student ADD resume VARCHAR(256);
ALTER TABLE student DROP COLUMN resume;

DROP TABLE member;   -- 表结构删除
DELETE FROM member;  -- 仅清空数据，表仍在
TRUNCATE TABLE member; -- 产品相关：快速清空
```

更具体的例子：

```sql
-- 1) 给 student 增加一列，并设默认值
ALTER TABLE student ADD enrollment_year INT DEFAULT 2024;

-- 2) 修改列类型 / 是否可空（不同SQL方言略有差异）
ALTER TABLE student MODIFY age INT NOT NULL;          -- MySQL
-- ALTER TABLE student ALTER COLUMN age SET NOT NULL; -- PostgreSQL

-- 3) 改列名
ALTER TABLE student RENAME COLUMN resume TO cv;       -- 较新标准 / PG / MySQL 8+
-- ALTER TABLE student CHANGE resume cv VARCHAR(256); -- MySQL 旧写法

-- 4) 加约束：学号唯一（若尚未是主键）
ALTER TABLE student ADD CONSTRAINT uq_student_sid UNIQUE (sid);

-- 5) 加外键：member.cid 必须对应 club.cid
ALTER TABLE member
  ADD CONSTRAINT fk_member_club
  FOREIGN KEY (cid) REFERENCES club(cid)
  ON DELETE CASCADE;

-- 6) 删约束后再删列（有外键时往往要先拆依赖）
ALTER TABLE member DROP FOREIGN KEY fk_member_club;   -- MySQL
ALTER TABLE member DROP COLUMN date;

-- 7) 条件删除 vs 整表清空
DELETE FROM member WHERE cid = 10;     -- 只删参加俱乐部 10 的记录
DELETE FROM student WHERE age < 16;    -- 按条件删行，可回滚（在事务里）
TRUNCATE TABLE member;                 -- 整表清空，通常不可带 WHERE，速度更快

-- 8) 删表顺序：先删引用方，再删被引用方
DROP TABLE IF EXISTS member;           -- 先删有外键的表
DROP TABLE IF EXISTS club;
DROP TABLE IF EXISTS student;
```

> [!NOTE]+ `DELETE` 与 `TRUNCATE` / `DROP`
>
> - `DELETE`：逐行删，可加 `WHERE`，受事务与触发器影响。
> - `TRUNCATE`：快速清空数据，表结构保留；多数引擎记日志方式不同，通常不能按条件删。
> - `DROP TABLE`：表定义与数据一起去掉；依赖此外键的表需先处理，否则报错。

### 索引（语法级）

```sql
CREATE INDEX idx_student_dept ON student(department);
DROP INDEX idx_student_dept;  -- 语法因方言而异
```

物理意义与 B+ 树见主笔记索引章。参考：[菜鸟教程 CREATE INDEX](https://www.runoob.com/sql/sql-create-index.html)。

<br>

## DML 基础

（Data Manipulation Language，数据操纵语言：对表中已有数据进行插入、更新、删除；查询见后文各章。）

### 插入

```sql
-- 按表定义列的顺序插入全部列；列数、类型、顺序必须与表一致
INSERT INTO student VALUES (1, 'Alice', 20, 'Female', 'CS');

-- 指定列名：只写给出的列，其余列填默认值或 NULL（未声明 NOT NULL 时）
-- 推荐写法：改表加列后不容易因“顺序对不上”而写错
INSERT INTO student (sid, name, department) VALUES (2, 'Bob', 'Math');

-- 一次插入多行（多数方言支持）
INSERT INTO student (sid, name, age, gender, department) VALUES
    (3, 'Carol', 19, 'Female', 'EE'),
    (4, 'Dave', 21, 'Male', 'CS');

-- 用查询结果批量插入：SELECT 的列数、类型须与 INSERT 列列表对应
-- 与“手工 VALUES”的区别：数据来自他表/同表，可加 WHERE 过滤
INSERT INTO student (sid, name, age, gender, department)
SELECT sid + 1000, name, age, gender, department
FROM student
WHERE department = 'CS';

-- 仅当不存在时再插入（方言相关；示意“防重复”思路）
-- MySQL: INSERT IGNORE / ON DUPLICATE KEY UPDATE
-- 标准思路也可用：先 NOT EXISTS 判断，或依赖主键冲突失败
INSERT INTO club (cid, name, supervisor)
SELECT 99, 'Reading', 'JL SUN'
WHERE NOT EXISTS (SELECT 1 FROM club WHERE cid = 99);
```

> [!WARNING]+ `INSERT` 易错点
>
> - `VALUES` 不写列名时，顺序 = 建表时列顺序，中间插过列就会错位。
> - `INSERT ... SELECT` 不要写成 `VALUES (SELECT ...)`（多数引擎语法不对）。
> - 违反主键 / `UNIQUE` / `NOT NULL` / 外键时整句失败（除非用 `IGNORE` 等方言扩展）。

### 更新与 CASE

```sql
-- 基本更新：改满足条件的行；务必带 WHERE，否则整表都被改
UPDATE instructor
SET salary = salary * 1.05
WHERE dept_name = 'Comp. Sci.';

-- 一次改多列
UPDATE student
SET age = age + 1,
    department = 'CS'
WHERE sid = 2;

-- CASE：按条件分支赋值（分类涨薪），避免写多条 UPDATE
UPDATE instructor
SET salary = CASE
    WHEN salary <= 100000 THEN salary * 1.05   -- 低薪多涨
    ELSE salary * 1.03                          -- 其余少涨
END;
-- 上面未写 WHERE → 所有行都走 CASE；若只要某系，再加 WHERE dept_name = '...'

-- 用子查询决定新值（“调到本系平均薪”示意）
UPDATE instructor i
SET salary = (
    SELECT avg_sal FROM (
        SELECT dept_name, AVG(salary) AS avg_sal
        FROM instructor
        GROUP BY dept_name
    ) t
    WHERE t.dept_name = i.dept_name
)
WHERE dept_name = 'Finance';
-- 注意：有的引擎禁止 UPDATE 直接子查询同一张表，需再包一层派生表（如上）
```

> [!NOTE]+ `UPDATE` 与 `CASE`
>
> - `SET col = expr`：每行用表达式算新值；`CASE` 是表达式，不是独立语句。
> - `CASE WHEN ... THEN ... ELSE ... END`：类似编程里的 if-else；漏写 `ELSE` 时，未命中分支结果常为 `NULL`（会把原值改成空，危险）。
> - 更新前先用同样的 `WHERE` 做一次 `SELECT` 核对影响行数，是面试/生产里的稳妥习惯。

### 删除

```sql
-- 删一行：用主键（或能唯一锁定该行的条件）定位
DELETE FROM student WHERE sid = 1;

-- 确认只会命中一行可先查：
-- SELECT * FROM student WHERE sid = 1;

-- 联合主键表：条件要写全，否则可能删多行
DELETE FROM member WHERE sid = 1 AND cid = 10;

-- 条件删除：所有满足谓词的行都会删掉（可能是多行）
DELETE FROM member WHERE cid = 10;

DELETE FROM student WHERE age < 16;

-- 清空全表数据，但保留表结构（等价于无 WHERE 的 DELETE，但实现/日志可能不同）
DELETE FROM member;

-- 与 DDL 的对比（删数据 vs 拆表）见上文「改表与删表」：
--   DELETE FROM t;     -- DML：删行，可事务回滚（一般情况）
--   TRUNCATE TABLE t;  -- 快速清空，通常不可带 WHERE
--   DROP TABLE t;      -- DDL：表定义一并去掉
```

> [!EXAMPLE]+ 只删一行
>
> SQL 没有单独的“删第 N 行”语法，靠 `WHERE` 精确匹配。
> 稳妥做法：条件落在**主键 / UNIQUE** 上（如 `WHERE sid = 1`），保证最多一行。
> 若 `WHERE` 能匹配多行（如 `WHERE department = 'CS'`），会一次删掉所有命中行。

> [!WARNING]+ `DELETE` / `TRUNCATE` / `DROP` 区别
>
> | | `DELETE` | `TRUNCATE` | `DROP` |
> | :--- | :--- | :--- | :--- |
> | 类别 | DML | 多视为 DDL（产品相关） | DDL |
> | 能否 `WHERE` | 能 | 一般不能 | 不适用 |
> | 表结构 | 保留 | 保留 | 删除 |
> | 事务 | 通常可回滚 | 很多引擎不可或受限 | 提交后结构已无 |
> | 触发器 | 常逐行触发 | 往往不触发行级触发器 | 对象直接消失 |

<br>

## 单表查询

从一张表里把需要的行/列取出来。业务上对应：列表筛选、模糊搜索、排序分页、取 Top-K。

### SELECT 骨架

```sql
-- DISTINCT：去掉重复的 department（多重集 → 集合语义）
-- 为什么需要：报表只要“有哪些系”，不要每个学生占一行重复系名
SELECT DISTINCT department
FROM student
-- BETWEEN：闭区间 [18,22]，等价 age >= 18 AND age <= 22
WHERE age BETWEEN 18 AND 22
  -- LIKE：模式匹配；% 任意长度，_ 单个字符
  -- 用在哪：搜索框“姓以 A 开头”、商品名模糊查
  AND name LIKE 'A%'
-- 先按系名字典序；ASC 可省略，DESC 为降序
ORDER BY department ASC;
```

要点：

- `*` 表示全部列；`DISTINCT` 去重；默认 `ALL` 保留重复（多重集）。
- `SELECT` 列表可含表达式：`salary/12 AS monthly`（算月薪展示，不必改表）。
- 关键字大小写不敏感；字符串比较是否敏感依排序规则/方言。
- `LIKE`：`%` 任意串，`_` 单字符。参考：[菜鸟教程 LIKE](https://www.runoob.com/sql/sql-like.html)。

> [!INFO]+ 为何有 `DISTINCT` / `ORDER BY` / `LIMIT`
>
> | 功能 | 为什么需要 | 典型场景 |
> | :--- | :--- | :--- |
> | `DISTINCT` | 投影后天然会产生重复行 | 去重名单、枚举取值 |
> | `ORDER BY` | 集合无序，展示必须显式排序 | 排行榜、时间线 |
> | `LIMIT`/`TOP` | 只要前几条，省传输与阅读成本 | 首页 Top10、分页 |

### WHERE 常用谓词

```sql
-- IN：属于给定集合；比一长串 OR 好读，常接子查询
SELECT * FROM student WHERE department IN ('CS', 'EE');

-- 判空必须用 IS NULL / IS NOT NULL，不能用 = NULL（见后文三值逻辑）
SELECT * FROM student WHERE age IS NULL;
SELECT * FROM student WHERE age IS NOT NULL;

-- 比较 + 逻辑组合；<> 表示不等于（有的方言也写 !=）
SELECT * FROM instructor
WHERE salary > 50000 AND dept_name <> 'Finance';
```

三值逻辑：与 `NULL` 比较得 `UNKNOWN`；`WHERE` 只保留真。判断空必须用 `IS NULL`。

### ORDER BY 与限行

```sql
SELECT name, age
FROM student
-- 年龄从大到小；同龄再按姓名升序（稳定展示）
ORDER BY age DESC, name ASC
LIMIT 1;          -- MySQL / PG：只要 1 行 → Top-1
-- SQL Server: SELECT TOP 1 ...
-- 标准：OFFSET 0 ROWS FETCH NEXT 1 ROWS ONLY
```

“最大差距两人”类题：自连接 + 排序 + `LIMIT`（Quiz 套路）。

```sql
-- 自连接：同一张表当两个人 s1、s2 来配对
-- sid < 避免 (A,B) 与 (B,A) 重复，也避免自己和自己比
SELECT s1.name, s2.name, ABS(s1.age - s2.age) AS age_difference
FROM student s1, student s2
WHERE s1.sid < s2.sid
ORDER BY age_difference DESC
LIMIT 1;   -- 差距最大的那一对
```

<br>

## 连接

多表信息拼在一起看。为什么需要：范式拆表后，学生在一张表、选课在另一张，业务却要“谁选了什么课”。参考：[菜鸟教程 JOIN](https://www.runoob.com/sql/sql-join.html)。

### 笛卡尔积与旧式内连接

```sql
-- 旧式写法：FROM 多表 = 先笛卡尔积，再用 WHERE 当连接条件
-- 漏写连接条件会变成巨大叉乘，极危险
SELECT student.name, club.name
FROM student, member, club
WHERE student.sid = member.sid      -- 学生↔成员
  AND member.cid = club.cid         -- 成员↔社团
  AND student.department = 'CS'
  AND club.name = 'Dancing';
```

`FROM` 多表先理解为笛卡尔积，再用 `WHERE` 过滤——优化器会改写，但语义如此。

### INNER / LEFT / RIGHT / FULL

```sql
-- INNER JOIN：两边都匹配才保留
-- 用在哪：只要“确实参加了社团”的学生名单
SELECT s.name, c.name AS club_name
FROM student s
INNER JOIN member m ON s.sid = m.sid
INNER JOIN club c ON m.cid = c.cid;

-- LEFT JOIN：左表全保留，右表对不上则填 NULL
-- 为什么需要：要看“没参加任何社团的学生”，或做完整花名册
SELECT s.name, c.name AS club_name
FROM student s
LEFT JOIN member m ON s.sid = m.sid
LEFT JOIN club c ON m.cid = c.cid;
-- 找从未入社的学生：再加 WHERE m.sid IS NULL

-- RIGHT JOIN：右表全保留（少用，常改写成换左右的 LEFT）
SELECT * FROM student s
RIGHT JOIN member m ON s.sid = m.sid;

-- FULL OUTER JOIN：两侧未匹配都保留（MySQL 常需用 UNION 模拟）
SELECT * FROM student s
FULL OUTER JOIN member m ON s.sid = m.sid;
```

> [!EXAMPLE]+ 四种 JOIN 何时选
>
> | 类型 | 保留谁 | 典型问题 |
> | :--- | :--- | :--- |
> | INNER | 匹配上的 | 订单明细（必须有商品） |
> | LEFT | 左表全部 | 用户列表 + 可选的最近登录 |
> | RIGHT | 右表全部 | 少用 |
> | FULL | 两侧全部 | 对账：两边都有、仅左有、仅右有 |

### NATURAL / USING / ON

```sql
-- NATURAL：自动按全部同名列等值连接（省事但不安全）
SELECT * FROM instructor NATURAL JOIN department;

-- USING：只指定同名连接列，比 NATURAL 可控一点
SELECT * FROM teaches JOIN course USING (course_id);

-- ON：最明确，允许列名不同、可写复杂条件 —— 生产首选
SELECT * FROM teaches t
JOIN course c ON t.course_id = c.course_id;
```

> [!WARNING]+ NATURAL JOIN
>
> 按**全部同名列**等值连接。列名一不小心同名即误连。作业与生产更推荐显式 `ON`。

### 自连接

```sql
-- 同一张 emp 表：a 是员工，b 是其经理
-- 为什么需要：层级/推荐人/先修课都在一张表里，要“自己连自己”
SELECT a.name AS emp, b.name AS manager
FROM emp a
JOIN emp b ON a.manager_id = b.id;
```

<br>

## 集合运算

把两个查询结果当集合做并/交/差。为什么需要：两个条件筛出的名单要合并或对比，又不想写很绕的 OR/EXISTS。

```sql
-- UNION：并集并去重 —— “CS 系 或 女生”的姓名（人只出现一次）
SELECT name FROM student WHERE department = 'CS'
UNION
SELECT name FROM student WHERE gender = 'Female';

-- UNION ALL：并集保留重复 —— 更快；要计数/流水合并时常用
-- SELECT ... UNION ALL SELECT ...

-- INTERSECT：两边都出现（交）
-- EXCEPT / MINUS：在前不在后（差）
-- MySQL 旧版本可能没有，可用 EXISTS / NOT EXISTS 模拟
```

模式需相容：列数相同、类型相容。

> [!NOTE]+ `UNION` vs `UNION ALL`
>
> `UNION` 要排序/去重，贵；确定无重复或需要重复时用 `UNION ALL`。面试常问性能差异。

<br>

## 聚集与分组

把多行收成“每个组一行”的统计。为什么需要：人均、总数、每个系多少人——业务报表核心。

```sql
-- GROUP BY department：每个系一组
-- COUNT / AVG：组内统计；AS 起别名方便读
SELECT department, COUNT(*) AS cnt, AVG(age) AS avg_age
FROM student
GROUP BY department
-- HAVING：分组之后再过滤组（这里：至少 3 人的系）
-- 为什么不能写在 WHERE：WHERE 时组还没形成，COUNT 尚不存在
HAVING COUNT(*) >= 3;
```

规则：

- `SELECT` 中未进入聚集的列，必须出现在 `GROUP BY` 中（严格模式）。
- `WHERE` 在分组前过滤行；`HAVING` 在分组后过滤组。
- `COUNT(*)` 计行；`COUNT(col)` 忽略该列 NULL；`SUM`/`AVG` 忽略 NULL；输入全空时 `SUM`/`AVG` 常为 NULL，`COUNT` 为 0。

```sql
-- 系均薪高于全校均薪的系
SELECT dept_name, AVG(salary) AS avg_salary
FROM instructor
GROUP BY dept_name
HAVING AVG(salary) > (SELECT AVG(salary) FROM instructor);
```

“每个系参加社团的学生百分比”：

```sql
-- WHERE 先留下“参加过社团”的学生，再按系计数
-- 分母是全校人数（标量子查询）；面试常追问是否应改成本系人数
SELECT department,
       COUNT(DISTINCT sid) * 100.0 / (SELECT COUNT(*) FROM student) AS percentage
FROM student
WHERE sid IN (SELECT sid FROM member)
GROUP BY department;
```

> [!WARNING]+ `WHERE` 与 `HAVING`
>
> - 过滤**行**：`WHERE age > 18`
> - 过滤**组**：`HAVING COUNT(*) >= 3` 或 `HAVING AVG(salary) > 50000`
> - `WHERE` 里写 `COUNT(*)` → 直接报错

<br>

## 子查询

查询里再套查询。为什么需要：条件依赖“另一查询的结果”（高于平均、属于某集合、存在关联行），一层 SELECT 写不清。

### 标量

```sql
-- 子查询返回单个值，拿来比较
-- 用在哪：高于全校均薪的人、晚于某活动开始时间的订单
SELECT name, salary
FROM instructor
WHERE salary > (SELECT AVG(salary) FROM instructor);
```

标量子查询必须返回至多一行一列（多行会报错）。

### IN / NOT IN

```sql
-- IN：sid 落在“社团 1 的成员”集合里
SELECT name FROM student
WHERE sid IN (SELECT sid FROM member WHERE cid = 1);

-- NOT IN：不在任何成员名单里 → 从未入社（有 NULL 陷阱，见下）
SELECT name FROM student
WHERE sid NOT IN (SELECT sid FROM member);
```

> [!WARNING]+ NOT IN 与 NULL
>
> 子查询结果含 `NULL` 时，`NOT IN` 整体可能恒为未知，导致结果为空。更稳妥：`NOT EXISTS`。

### SOME / ALL / ANY

```sql
-- > SOME：至少大于集合中某一个（“比 Comp. Sci. 里某人高”）
SELECT name FROM instructor
WHERE salary > SOME (
    SELECT salary FROM instructor WHERE dept_name = 'Comp. Sci.'
);

-- > ALL：大于集合中每一个（“比 Comp. Sci. 全系都高” → 该系最高薪之上）
SELECT name FROM instructor
WHERE salary > ALL (
    SELECT salary FROM instructor WHERE dept_name = 'Comp. Sci.'
);
```

`= SOME` 等价 `IN`；`<> ALL` 等价 `NOT IN`（仍需注意 NULL）。`ANY` 与 `SOME` 同义。

### EXISTS / NOT EXISTS

```sql
-- EXISTS：只要子查询有一行就算真（不关心选出什么，常写 SELECT 1）
-- 用在哪：有没有成员、有没有未完成订单 —— 存在性判断；相关子查询利器
SELECT c.name
FROM club c
WHERE EXISTS (
    SELECT 1 FROM member m WHERE m.cid = c.cid
);
```

相关子查询：内层引用外层元组。逻辑上对外层每行执行一次（优化器可能改写为半连接）。

### FROM 中的子查询

```sql
-- 先算出“每系均薪”当临时表 t，再筛均薪 > 50000
-- 为什么需要：WHERE 不能直接引用 SELECT 列表别名；派生表可先算再滤
SELECT dept_name, avg_salary
FROM (
    SELECT dept_name, AVG(salary) AS avg_salary
    FROM instructor
    GROUP BY dept_name
) AS t
WHERE avg_salary > 50000;
```

标准 SQL 中，`FROM` 子查询一般不能引用同级其他表列，除非 `LATERAL`（方言支持不一）。

<br>

## 全称量化

自然语言“**所有** … 都 …”在 SQL 没有直写的 `FOR ALL`，常用双重 `NOT EXISTS`（不存在反例）。为什么需要：权限“必须学完所有先修”、社团“JL SUN 名下每个社都参加”——全称命题。

### 模式

> 不存在一个（反例），使得（该反例未被满足）。

### Quiz：JL SUN 监督的所有俱乐部的成员

找学生：对 JL SUN 的每个俱乐部，该生都是成员。

```sql
SELECT student.name
FROM student
WHERE NOT EXISTS (
    -- 外层 NOT EXISTS：不存在这样的“坏俱乐部”
    SELECT club.cid
    FROM club
    WHERE club.supervisor = 'JL SUN'
      AND NOT EXISTS (
          -- 内层：该生没有参加这个俱乐部 → 构成反例
          SELECT member.sid
          FROM member
          WHERE member.sid = student.sid
            AND member.cid = club.cid
      )
);
```

直觉：不存在“JL SUN 的俱乐部，且该生未参加”。

### Quiz：只有女生的俱乐部

```sql
SELECT club.name
FROM club
WHERE NOT EXISTS (
    -- 不存在“非女生成员” → 成员若存在则全是女生
    SELECT member.sid
    FROM member
    JOIN student ON member.sid = student.sid
    WHERE member.cid = club.cid
      AND student.gender <> 'Female'
);
```

空俱乐部是否算“只有女生”？按此写法会被选出；若需至少一人，再加 `EXISTS` 成员条件。

### 除法：选了某集合全部课程

```sql
-- 关系代数除法：选了 Comp. Sci. 系每一门课的学生
SELECT DISTINCT s.ID
FROM student s
WHERE NOT EXISTS (
    SELECT course_id
    FROM course
    WHERE dept_name = 'Comp. Sci.'
      AND NOT EXISTS (
          SELECT *
          FROM takes t
          WHERE t.ID = s.ID
            AND t.course_id = course.course_id
      )
);
```

等价计数法（集合无重复选课前提）：

```sql
-- 选中的 Comp. Sci. 课门数 = 该系课门数总数 → 门门都选了
SELECT t.ID
FROM takes t
JOIN course c ON t.course_id = c.course_id
WHERE c.dept_name = 'Comp. Sci.'
GROUP BY t.ID
HAVING COUNT(DISTINCT t.course_id) = (
    SELECT COUNT(*) FROM course WHERE dept_name = 'Comp. Sci.'
);
```

<br>

## WITH 与 CTE

CTE（Common Table Expression）给子查询起个临时名。为什么需要：复杂查询分层写、可读可复用；递归 CTE 还能走树/图。

```sql
-- 先定义“CSE 教师”临时结果，后面像用表一样查
WITH cse_instructors AS (
    SELECT ID, name, salary
    FROM instructor
    WHERE dept_name = 'Comp. Sci.'
)
SELECT name FROM cse_instructors WHERE salary > 80000;
```

递归 CTE（组织树、评论楼中楼、账单展开；方言需支持 `RECURSIVE`）：

```sql
-- 锚点：从 id=1 的员工出发
-- 递归臂：不断找“manager 是上一层的人” → 全部下属
WITH RECURSIVE subordinates AS (
    SELECT id, manager_id, name FROM emp WHERE id = 1
    UNION ALL
    SELECT e.id, e.manager_id, e.name
    FROM emp e
    JOIN subordinates s ON e.manager_id = s.id
)
SELECT * FROM subordinates;
```

> [!INFO]+ CTE 用在哪
>
> 多步报表（先过滤再聚合再排名）、递归组织架构、替代层层嵌套派生表。作用域只在本条语句。

<br>

## 窗口函数

在“不合并行”的前提下做排名、累计、组内平均。为什么需要：既要看每个员工明细，又要旁注“本部门第几名 / 部门均薪”——`GROUP BY` 会把多行收成一行，窗口不会。

```sql
SELECT name, dept_name, salary,
       -- 各部门内按薪降序排名（并列会占位，看 RANK/DENSE_RANK 区别）
       RANK() OVER (PARTITION BY dept_name ORDER BY salary DESC) AS rk,
       -- 每行旁挂“本部门平均薪”（行数不变）
       AVG(salary) OVER (PARTITION BY dept_name) AS dept_avg,
       -- 按薪排序的累计和（跑动合计；财务流水常用）
       SUM(salary) OVER (
           ORDER BY salary
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running
FROM instructor;
```

常用：`ROW_NUMBER`（唯一序号）、`RANK` / `DENSE_RANK`（排名）、`NTILE`（分桶）、`LAG`/`LEAD`（上下行）、`SUM/AVG/... OVER`。

与 `GROUP BY` 区别：窗口不合并行，只附加分析列。

> [!EXAMPLE]+ 窗口与分组
>
> - 只要“每系均薪”一张汇总表 → `GROUP BY`
> - 要“每个教师一行，外加本系均薪、本系名次” → 窗口
> - Top-K per group、连续登录、同比环比 → 窗口几乎是标配

<br>

## NULL 与三值逻辑

`NULL` 表示“未知/缺失”，不是 0、不是空串。为什么单独讲：一不小心 `= NULL`、`NOT IN (..., NULL)` 就会静默得到空结果，线上难查。

| 表达式 | 结果要点 |
| :--- | :--- |
| `1 = NULL` | UNKNOWN |
| `NULL = NULL` | UNKNOWN（不是真） |
| `age IN (1, NULL)` | 若 age≠1 则为 UNKNOWN 而非 FALSE |
| `WHERE` 子句 | 仅 TRUE 通过（UNKNOWN 丢掉） |
| 聚集 | 除 `COUNT(*)` 外多数忽略 NULL |

```sql
-- age = age 对 NULL 为 UNKNOWN → NULL 年龄的行被丢掉
SELECT * FROM student WHERE age = age;

-- 正确找“年龄未知”
SELECT * FROM student WHERE age IS NULL;

-- COALESCE：把 NULL 换成默认值（报表展示、避免算式被 NULL 污染）
SELECT name, COALESCE(age, 0) AS age_show FROM student;
```

<br>

## 视图

把常用查询存成“虚拟表”。为什么需要：简化复杂 SQL、按角色裁剪可见列（安全）、逻辑模式变化时少改应用（逻辑数据独立性）。

```sql
-- 只暴露 CS 学生的部分列；应用侧当表查即可
CREATE VIEW cs_student AS
SELECT sid, name, age
FROM student
WHERE department = 'CS';

SELECT * FROM cs_student WHERE age > 20;
```

可更新视图通常要求：单基表、无聚集/`DISTINCT`、无计算列作为关键路径等（标准与产品限制多）。

> [!NOTE]+ 视图不是备份
>
> 默认不存数据副本（物化视图除外）；基表变，视图结果变。授权时常 `GRANT SELECT ON 视图` 而不授基表。

<br>

## 完整性与触发器

完整性：不让坏数据进库。触发器：数据变更时自动跑一段逻辑。为什么需要：学分合计校验、涨薪超过 50% 记审计日志、删订单前清明细——放在应用层容易漏。

### 断言（多数 MySQL 不支持）

```sql
-- 全库级约束：每个学生 total_cred 必须等于及格课程学分之和
CREATE ASSERTION credits_constraint CHECK (
    NOT EXISTS (
        SELECT *
        FROM student S
        WHERE total_cred <> (
            SELECT SUM(credits)
            FROM takes NATURAL JOIN course
            WHERE takes.ID = S.ID
              AND grade IS NOT NULL
              AND grade <> 'F'
        )
    )
);
```

实践中多用表级 `CHECK`、外键、或触发器/应用校验代替断言。

### 触发器骨架

```sql
-- AFTER UPDATE：薪资改完后触发
-- 当涨幅 > 50% 时做处理（记日志 / 拒绝 / 告警）——防误操作或合规审计
CREATE TRIGGER bump_salary
AFTER UPDATE OF salary ON instructor
REFERENCING NEW ROW AS n OLD ROW AS o
FOR EACH ROW
WHEN (n.salary > o.salary * 1.5)
BEGIN
    -- 方言相关过程体
END;
```

`BEFORE`/`AFTER` × `INSERT`/`UPDATE`/`DELETE`；`OLD`/`NEW` 分别是改前/改后行。`BEFORE` 可改拟写入值；`AFTER` 适合级联写其他表。

<br>

## 权限

（属 DCL：Data Control Language；`GRANT` / `REVOKE` 控制谁能碰哪些对象。）

为什么需要：同一库里财务只能改薪资、教务只能读成绩——最小权限原则。

```sql
-- 角色：一批权限的打包，便于批量赋给同类用户
CREATE ROLE analyst;
GRANT SELECT ON student TO analyst;     -- 角色可读 student
GRANT analyst TO alice;                 -- alice 继承 analyst

-- 列级更新：bob 只能改 salary，且可转授（WITH GRANT OPTION）
GRANT UPDATE (salary) ON instructor TO bob WITH GRANT OPTION;

-- 回收；CASCADE 会连带收回转授出去的权限
REVOKE SELECT ON student FROM alice CASCADE;
```

特权直觉：读、插入、删除、更新；另有参照、触发器等产品扩展。

<br>


# 习题

## 面试题型清单

下列题目在语句旁用注释标出**考点**与**为何这么写**；先自己写再对照。

### 基础

1. 第二高薪（注意并列）：窗口 `DENSE_RANK` 或两次 `MAX` 子查询。
2. 连续登录 N 天：窗口差值 / 自连接日期。
3. 每个部门工资前三：`RANK() OVER (PARTITION BY ...)`。
4. 有经理的员工与无经理的员工：`LEFT JOIN` + `IS NULL`。

```sql
-- 考点：分组内 Top-K；DENSE_RANK 并列同名次且不跳号
SELECT dept_name, name, salary
FROM (
    SELECT dept_name, name, salary,
           DENSE_RANK() OVER (PARTITION BY dept_name ORDER BY salary DESC) AS rk
    FROM instructor
) t
WHERE rk <= 3;
```

### 中级嵌套

```sql
-- 考点：相关子查询；每人跟“自己所在系”的均薪比，不是全校
SELECT i.name, i.dept_name, i.salary
FROM instructor i
WHERE i.salary > (
    SELECT AVG(salary) FROM instructor i2
    WHERE i2.dept_name = i.dept_name
);
```

```sql
-- 考点：反存在；LEFT JOIN ... IS NULL 亦可
SELECT course_id, title
FROM course c
WHERE NOT EXISTS (
    SELECT 1 FROM takes t WHERE t.course_id = c.course_id
);
```

### 高级：相关 + 分组 + 连接

```sql
-- 至少选修了两门同系课程的学生
SELECT s.ID, s.name
FROM student s
WHERE EXISTS (
    SELECT 1
    FROM takes t1
    JOIN course c1 ON t1.course_id = c1.course_id
    JOIN takes t2 ON t1.ID = t2.ID AND t1.course_id < t2.course_id
    JOIN course c2 ON t2.course_id = c2.course_id
    WHERE t1.ID = s.ID
      AND c1.dept_name = c2.dept_name
);
```

```sql
-- 找出没有不合格成绩（无 F / 无空可按题意）且学分总和最高的学生
WITH passed AS (
    SELECT t.ID, SUM(c.credits) AS cred
    FROM takes t
    JOIN course c ON t.course_id = c.course_id
    WHERE t.grade IS NOT NULL AND t.grade <> 'F'
    GROUP BY t.ID
    HAVING SUM(CASE WHEN t.grade = 'F' THEN 1 ELSE 0 END) = 0
)
SELECT ID, cred FROM passed
WHERE cred = (SELECT MAX(cred) FROM passed);
```


<br>

## 补充例子

### 第二高薪

```sql
SELECT MAX(salary) AS second_high
FROM instructor
WHERE salary < (SELECT MAX(salary) FROM instructor);
```

```sql
SELECT DISTINCT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rk
    FROM instructor
) t
WHERE rk = 2;
```

### 没有下属的员工

```sql
SELECT e.name
FROM emp e
LEFT JOIN emp s ON s.manager_id = e.id
WHERE s.id IS NULL;
```

### 连续出现三次的数字（经典）

```sql
SELECT DISTINCT l1.num AS ConsecutiveNums
FROM logs l1
JOIN logs l2 ON l1.id = l2.id - 1 AND l1.num = l2.num
JOIN logs l3 ON l2.id = l3.id - 1 AND l2.num = l3.num;
```

### 部门工资最高的员工

```sql
SELECT d.name AS Department, e.name AS Employee, e.salary
FROM emp e
JOIN dept d ON e.dept_id = d.id
WHERE (e.dept_id, e.salary) IN (
    SELECT dept_id, MAX(salary) FROM emp GROUP BY dept_id
);
```

### 换座（奇数换到下一个，末行奇数不动）

```sql
SELECT
    CASE
        WHEN id % 2 = 1 AND id = (SELECT MAX(id) FROM seat) THEN id
        WHEN id % 2 = 1 THEN id + 1
        ELSE id - 1
    END AS id,
    student
FROM seat
ORDER BY id;
```

### 分数排名（不并列跳号用 RANK，密级用 DENSE_RANK）

```sql
SELECT score,
       DENSE_RANK() OVER (ORDER BY score DESC) AS `rank`
FROM scores;
```

### 每月首次登录

```sql
SELECT user_id, MIN(login_date) AS first_day
FROM logins
GROUP BY user_id, DATE_FORMAT(login_date, '%Y-%m');
```

### 三角连接：三人互相成为好友（对称边需规范）

```sql
SELECT a.person AS p1, b.person AS p2, c.person AS p3
FROM friendship a
JOIN friendship b ON a.friend = b.person
JOIN friendship c ON b.friend = c.person AND c.friend = a.person
WHERE a.person < b.person AND b.person < c.person;
```

### 用窗口算同比

```sql
SELECT ym, revenue,
       LAG(revenue) OVER (ORDER BY ym) AS prev_rev,
       revenue - LAG(revenue) OVER (ORDER BY ym) AS diff
FROM monthly_revenue;
```

<br>


## ZJU习题

### Quiz2 五题（完整可交卷版）

模式：`student(sid,name,age,gender,department)`，`club(cid,name,supervisor)`，`member(sid,cid,date)`。

#### (1) CS 且舞蹈社

```sql
SELECT student.name
FROM student
JOIN member ON student.sid = member.sid
JOIN club ON member.cid = club.cid
WHERE student.department = 'CS'
  AND club.name = 'Dancing';
```

#### (2) JL SUN 监督的全部俱乐部之成员

```sql
SELECT student.name
FROM student
WHERE NOT EXISTS (
    SELECT club.cid
    FROM club
    WHERE club.supervisor = 'JL SUN'
      AND NOT EXISTS (
          SELECT member.sid
          FROM member
          WHERE member.sid = student.sid
            AND member.cid = club.cid
      )
);
```

#### (3) 只有女生的俱乐部

```sql
SELECT club.name
FROM club
WHERE NOT EXISTS (
    SELECT member.sid
    FROM member
    JOIN student ON member.sid = student.sid
    WHERE member.cid = club.cid
      AND student.gender <> 'Female'
);
```

若要求俱乐部非空，追加：

```sql
AND EXISTS (
    SELECT 1 FROM member WHERE member.cid = club.cid
)
```

#### (4) 每系“参加过社团”的学生占全校比例

```sql
SELECT department,
       COUNT(DISTINCT sid) * 100.0 / (SELECT COUNT(*) FROM student) AS percentage
FROM student
WHERE sid IN (SELECT sid FROM member)
GROUP BY department;
```

分母若改为“本系人数”：

```sql
SELECT s.department,
       COUNT(DISTINCT s.sid) * 100.0 / d.cnt AS percentage
FROM student s
JOIN (SELECT department, COUNT(*) AS cnt FROM student GROUP BY department) d
  ON s.department = d.department
WHERE s.sid IN (SELECT sid FROM member)
GROUP BY s.department, d.cnt;
```

#### (5) 年龄差最大的两人

```sql
SELECT s1.name, s2.name, ABS(s1.age - s2.age) AS age_difference
FROM student s1
JOIN student s2 ON s1.sid < s2.sid
ORDER BY age_difference DESC
LIMIT 1;
```

<br>

## 408 向 SQL 句式

### 查询“从不 / 没有”

```sql
-- 没有选任何课的学生
SELECT name FROM student s
WHERE NOT EXISTS (SELECT 1 FROM takes t WHERE t.ID = s.ID);

-- 等价反连接
SELECT s.name
FROM student s
LEFT JOIN takes t ON s.ID = t.ID
WHERE t.ID IS NULL;
```

### 查询“至少 / 恰好 / 至多”

```sql
SELECT ID
FROM takes
GROUP BY ID
HAVING COUNT(DISTINCT course_id) >= 3;

SELECT ID
FROM takes
GROUP BY ID
HAVING COUNT(DISTINCT course_id) = 3;
```

### 查询“高于所在组平均”

```sql
SELECT *
FROM instructor i
WHERE salary > (
    SELECT AVG(salary) FROM instructor
    WHERE dept_name = i.dept_name
);
```

### 用 EXISTS 表达交与差

```sql
-- 既选了 A 课又选了 B 课
SELECT DISTINCT t1.ID
FROM takes t1
WHERE t1.course_id = 'CS-101'
  AND EXISTS (
      SELECT 1 FROM takes t2
      WHERE t2.ID = t1.ID AND t2.course_id = 'CS-190'
  );

-- 选了 A 没选 B
SELECT DISTINCT t1.ID
FROM takes t1
WHERE t1.course_id = 'CS-101'
  AND NOT EXISTS (
      SELECT 1 FROM takes t2
      WHERE t2.ID = t1.ID AND t2.course_id = 'CS-190'
  );
```

### 更新中的子查询

```sql
UPDATE instructor
SET salary = salary * 1.05
WHERE dept_name IN (
    SELECT dept_name FROM department WHERE budget > 100000
);
-- 注意：部分引擎禁止在 UPDATE 中直接子查询同表，需包一层派生表
```

### 删除“没有参照”的行

```sql
DELETE FROM course
WHERE NOT EXISTS (
    SELECT 1 FROM section WHERE section.course_id = course.course_id
);
```

<br>

## 复杂嵌套模板库

### 模板 A：全称量化（除法）

```sql
-- r ÷ s  : 选了 s 中全部课程的学生
SELECT DISTINCT r.sid
FROM takes r
WHERE NOT EXISTS (
    SELECT * FROM course_set s   -- s 为被除关系
    WHERE NOT EXISTS (
        SELECT * FROM takes t
        WHERE t.sid = r.sid AND t.course_id = s.course_id
    )
);
```

### 模板 B：双重否定“唯一性”

```sql
-- 只开设在一个校区的课程（示意）
SELECT cno
FROM offering
GROUP BY cno
HAVING COUNT(DISTINCT campus) = 1;
```

### 模板 C：排名后取区间

```sql
SELECT * FROM (
    SELECT name, salary,
           RANK() OVER (ORDER BY salary DESC) AS rk
    FROM instructor
) x
WHERE rk BETWEEN 2 AND 5;
```

### 模板 D：缺口补齐（日历 / 连续）

```sql
-- 找出断签：前一天登录过且当天未登录（示意）
SELECT a.user_id, a.dt AS prev_day
FROM checkins a
WHERE EXISTS (
    SELECT 1 FROM checkins b
    WHERE b.user_id = a.user_id AND b.dt = DATE_ADD(a.dt, INTERVAL 1 DAY)
) IS FALSE;
-- 实际连续问题多用窗口 row_number 与日期差值分组
```

```sql
SELECT user_id, MIN(dt) AS start_dt, MAX(dt) AS end_dt, COUNT(*) AS days
FROM (
    SELECT user_id, dt,
           DATE_SUB(dt, INTERVAL ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY dt) DAY) AS grp
    FROM checkins
) t
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

<br>

## 易错清单

1. `WHERE` 里写聚集 → 应放 `HAVING`。
2. `SELECT` 非聚集列未进 `GROUP BY`。
3. `NOT IN` 子查询含 NULL。
4. `COUNT(col)` 与 `COUNT(*)` 混淆。
5. 旧式逗号连接漏连接条件 → 意外笛卡尔积。
6. `UNION` 与 `UNION ALL` 性能与去重差异。
7. 相关子查询相关性列写错表别名。
8. 全称量化少写一层 `NOT EXISTS`。
9. `ORDER BY` 别名在部分方言/子查询中不可用。
10. 事务题与 SQL 题混淆隔离级别默认值（产品相关）。

<br>


