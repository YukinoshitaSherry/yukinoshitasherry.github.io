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

### 内容

本文是 [数据库系统笔记](/数据库系统/) 的 SQL 独立部分。理论侧（关系代数、优化、事务）见主笔记。

参考：

- [菜鸟教程 SQL](https://www.runoob.com/sql/sql-tutorial.html)
- [菜鸟教程 JOIN](https://www.runoob.com/sql/sql-join.html)
- [牛客 SQL 篇](https://www.nowcoder.com/exam/oj?tab=SQL%E7%AF%87&topicId=199)


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
    name        VARCHAR(50) NOT NULL,   -- 不允许空姓名
    age         INT,
    gender      VARCHAR(10),
    department  VARCHAR(50),
    PRIMARY KEY (sid),                  -- 主键：唯一标识一名学生
    CHECK (age IS NULL OR age >= 0)     -- 表级检查：年龄非负或未知
);

CREATE TABLE club (
    cid         INT PRIMARY KEY,        -- 列级写法，与 PRIMARY KEY (cid) 等价
    name        VARCHAR(50) NOT NULL,
    supervisor  VARCHAR(50)
);

-- 选课/入社关系表：多对多用“中间表 + 联合主键 + 外键”
CREATE TABLE member (
    sid   INT,
    cid   INT,
    date  DATE,
    PRIMARY KEY (sid, cid),             -- 联合主键：同一人同一社只记一次
    FOREIGN KEY (sid) REFERENCES student(sid)
        ON DELETE CASCADE               -- 学生删了，其入社记录跟着删
        ON UPDATE CASCADE,              -- 学号改了，外键同步改
    FOREIGN KEY (cid) REFERENCES club(cid)
);
```

#### 常见类型

（几乎所有类型允许 `NULL`，除非声明 `NOT NULL`）：

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


### 键与约束

建表不只是“有哪些列”，更要声明**谁唯一标识一行、表与表如何引用、哪些值非法**。下面按 SQL 里最常写的约束说明；理论侧（超键/候选键/闭包）见 [数据库系统笔记](/数据库系统/)。

#### 主键 PRIMARY KEY

（Primary Key）

- **是什么**：从表的候选键里选出的一个，用来**唯一标识**每一行。
- **规则**：唯一 + 非空（不能是 `NULL`）；一张表通常只有一个主键（可以是单列或联合多列）。
- **为什么需要**：更新/删除要精确定位一行（`WHERE sid = 1`）；JOIN 时有稳定连接列；很多引擎会据此建索引。
- **用在哪**：几乎每张实体表（学生、订单、商品）；中间表常用联合主键。

```sql
-- 单列主键（两种等价写法）
CREATE TABLE department (
    dept_name VARCHAR(20) PRIMARY KEY,
    building  VARCHAR(20),
    budget    NUMERIC(12,2)
);

-- 联合主键：多对多关系“边”上，两端一起才唯一
-- PRIMARY KEY (sid, cid) 见上 member 表
```

> [!NOTE]+ 主键 vs `UNIQUE`
>
> - 主键：唯一且非空，表级“官方身份证”，一般只有一个。
> - `UNIQUE`：唯一，但**允许 NULL**（多数引擎对 NULL 的唯一性处理有差异）；一张表可有多个唯一约束（如邮箱、手机号）。
> - 业务上“也能唯一标识、但不是主键”的列 → 用 `UNIQUE`（候选键落库的常见写法）。

#### 外键 FOREIGN KEY

（Foreign Key）

- **是什么**：本表某列（或列组）的值必须在**另一张表**的主键（或唯一键）中出现，或为 `NULL`（允许“暂不关联”时）。
- **为什么需要**：保证参照完整性——不能出现“选课记录指向不存在的学生”。
- **用在哪**：订单→用户、成员→学生/社团、`teaches.ID`→`instructor.ID`、`instructor.dept_name`→`department.dept_name`。

```sql
CREATE TABLE instructor (
    ID         CHAR(5),
    name       VARCHAR(20) NOT NULL,
    dept_name  VARCHAR(20),
    salary     NUMERIC(8,2),
    PRIMARY KEY (ID),
    FOREIGN KEY (dept_name) REFERENCES department(dept_name)
);

-- 插入失败例：department 里没有 'Astrology' 时，下面会报错
-- INSERT INTO instructor VALUES ('99999', 'Alice', 'Astrology', 90000);
```

父表（被引用）改删时，子表（带外键）怎么办——用参照动作：

| 动作 | 含义 | 典型场景 |
| :--- | :--- | :--- |
| `NO ACTION` / `RESTRICT` | 有子行引用则禁止删/改父行 | 默认、最安全 |
| `ON DELETE CASCADE` | 删父行时级联删子行 | 删用户顺带删其购物车 |
| `ON UPDATE CASCADE` | 改父键时子表外键跟着改 | 学号变更（少见） |
| `SET NULL` | 父行没了，子行外键置空 | 岗位撤销，员工“暂无部门” |
| `SET DEFAULT` | 置为默认值 | 产品相关 |

```sql
FOREIGN KEY (sid) REFERENCES student(sid)
    ON DELETE CASCADE
    ON UPDATE CASCADE;
```

##### CASCADE 级联（详解）

**Cascade** 原意是“瀑布式连带”：对**父表（被引用表）**做删除或更新时，数据库按外键定义**自动同步处理子表（引用表）**，不必手写第二条 `DELETE`/`UPDATE`。

为什么需要：没有级联时，父行若仍被子行引用，删除常被直接拒绝；若先手工删子再删父，应用层容易漏步骤，留下“指向已不存在主键”的脏数据或半截清理。

| 写法 | 父表操作 | 子表自动发生什么 |
| :--- | :--- | :--- |
| `ON DELETE CASCADE` | `DELETE` 掉被引用的主键行 | 所有引用该键的子行一并删除 |
| `ON UPDATE CASCADE` | `UPDATE` 改了被引用的主键值 | 子表外键列改成新值 |

```sql
-- 假设 member.sid 已声明 ON DELETE CASCADE / ON UPDATE CASCADE

-- 删学生 1 之前：member 里可能有 (1,10)、(1,20)
DELETE FROM student WHERE sid = 1;
-- 之后：上述 member 行自动消失；无需再 DELETE FROM member WHERE sid = 1;

-- 学号 1 改成 1001（少见，但语法上支持）
UPDATE student SET sid = 1001 WHERE sid = 1;
-- 之后：原引用 sid=1 的 member 行自动变成 sid=1001
```

对比（同一场景，**没有** CASCADE，默认 `NO ACTION`/`RESTRICT`）：

```sql
-- member 里还有 sid=1 时：
DELETE FROM student WHERE sid = 1;   -- 失败：有子行引用
-- 只能先：
DELETE FROM member WHERE sid = 1;
DELETE FROM student WHERE sid = 1;
```

> [!EXAMPLE]+ 何时用 / 何时慎用
>
> **适合 CASCADE**
>
> - 子行是父行的附属数据：用户→购物车、订单→订单明细、学生→入社记录。
> - 业务语义就是“父没了，子也没意义”。
>
> **慎用 CASCADE**
>
> - 子行有独立业务价值（如历史订单、审计流水）：误删父表会**连锁删光**子表，不可轻易恢复。
> - 多层外键链：`A ← B ← C` 都 CASCADE 时，删 A 可能拖垮一大片，需想清楚范围。
> - 更稳妥时改用 `ON DELETE SET NULL`（保留子行、断开关联）或禁止删除（`RESTRICT`），由应用显式归档。

> [!WARNING]+ 别和另外两种 “CASCADE” 搞混
>
> | 出现位置 | 含义 |
> | :--- | :--- |
> | 外键 `ON DELETE/UPDATE CASCADE` | 改删父键时**级联改删子表数据**（上文） |
> | `REVOKE ... CASCADE` | 收回权限时，**连带收回**由此转授出去的权限 |
> | 事务里的“级联回滚” | 并发控制概念：一事务中止导致读过其脏写的事务也回滚（见主笔记），**不是**外键语法 |
>
> 口语里说“加个 cascade”，在建表语境下几乎总是指外键的 `ON DELETE CASCADE`。

> [!WARNING]+ 外键方向
>
> 外键建在**引用方（多的一端 / 关系表）**。  
> `member.sid` → `student.sid`，不是反过来在 `student` 上写“有哪些社团”。
> 级联也写在子表外键定义上：是“子表声明：父没了我就跟着没”。

#### 其他常用约束

| 约束 | 作用 | 为什么需要 / 用在哪 |
| :--- | :--- | :--- |
| `NOT NULL` | 该列禁止空 | 姓名、主键列；必填业务字段 |
| `UNIQUE` | 取值不重复（主键以外） | 登录名、身份证号、邮箱 |
| `CHECK (条件)` | 行级/表级取值校验 | 年龄≥0、成绩 in ('A'..'F')、起止日期有序 |
| `DEFAULT 值` | 插入未给列时用默认 | 注册年份、状态='pending' |

```sql
CREATE TABLE takes (
    ID         CHAR(5),
    course_id  VARCHAR(8),
    sec_id     VARCHAR(8),
    semester   VARCHAR(6),
    year       NUMERIC(4,0),
    grade      VARCHAR(2) DEFAULT NULL,   -- 未出成绩可为空
    PRIMARY KEY (ID, course_id, sec_id, semester, year),
    FOREIGN KEY (ID) REFERENCES student(ID),
    FOREIGN KEY (course_id) REFERENCES course(course_id),
    CHECK (grade IS NULL OR grade IN ('A','B','C','D','F'))
);
```

#### 三类完整性（对应关系）

| 完整性 | SQL 里主要靠 |
| :--- | :--- |
| 实体完整性 | `PRIMARY KEY`（非空且唯一） |
| 参照完整性 | `FOREIGN KEY` + 参照动作 |
| 用户定义完整性 | `NOT NULL` / `UNIQUE` / `CHECK` / 触发器 / 断言 |

> [!EXAMPLE]+ 用本文模式串起来
>
> - `student.sid`、`club.cid`：各自主键。
> - `member(sid,cid)`：联合主键防重复入社；两个外键分别挂学生与社团。
> - 删学生且 `ON DELETE CASCADE`：其 `member` 行自动清掉，避免“幽灵选课”。
> - 删社团若**没有**级联且仍有成员：删除被拒绝 → 先清 `member` 或改策略。

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

先固定两张小表（数字少才好看清）。连接条件一律：`student.sid = member.sid`。

<div style="display: flex; gap: 1.5rem; margin: 10px 0; flex-wrap: wrap; align-items: flex-start;">
<div style="flex: 1; min-width: 200px;">

**表 `student`（左表）**

| sid | name |
| :---: | :--- |
| 1 | 小明 |
| 2 | 小红 |
| 3 | 小刚 |

</div>
<div style="flex: 1; min-width: 200px;">

**表 `member`（右表，谁进了哪个社）**

| sid | club |
| :---: | :--- |
| 1 | 舞蹈社 |
| 1 | 篮球社 |
| 2 | 舞蹈社 |
| 4 | 棋社 |

</div>
</div>

读表：小明进了两个社；小红进了舞蹈社；小刚**没进任何社**；`member` 里还有 sid=4（棋社），但 `student` 里**没有**这个人（脏数据或学生已删）。

#### INNER JOIN（只要两边对得上）

```sql
SELECT s.sid, s.name, m.club
FROM student s
INNER JOIN member m ON s.sid = m.sid;
```

结果只有“学生表里有、且确实入过社”的行：

| sid | name | club |
| :---: | :--- | :--- |
| 1 | 小明 | 舞蹈社 |
| 1 | 小明 | 篮球社 |
| 2 | 小红 | 舞蹈社 |

- 小刚没了（左有右无）
- sid=4 棋社没了（右有左无）
- 小明出现两行：一人多社，INNER 会**复制左行**

> [!INFO]+ 何时用 INNER
>
> 只要“有效关联”，不要光杆司令。例如：下过单的用户、选过课的学生。

#### LEFT JOIN（左表全留，右对不上就 NULL）

```sql
SELECT s.sid, s.name, m.club
FROM student s
LEFT JOIN member m ON s.sid = m.sid;
```

| sid | name | club |
| :---: | :--- | :--- |
| 1 | 小明 | 舞蹈社 |
| 1 | 小明 | 篮球社 |
| 2 | 小红 | 舞蹈社 |
| 3 | 小刚 | NULL |

- 小刚还在，`club` 为 `NULL` → 表示“这个学生没有匹配的入社记录”
- sid=4 棋社仍然不出现（右独有，LEFT 不管）

找“从未入社的学生”：

```sql
SELECT s.sid, s.name
FROM student s
LEFT JOIN member m ON s.sid = m.sid
WHERE m.sid IS NULL;   -- 右表关键列是 NULL → 没配上
```

结果：只有小刚。

> [!INFO]+ 何时用 LEFT
>
> 花名册要完整，右边信息可有可无。例如：全部员工 + 可选的部门名；全部学生 + 可选的社团。

#### RIGHT JOIN（右表全留，左对不上就 NULL）

```sql
SELECT s.sid, s.name, m.sid AS member_sid, m.club
FROM student s
RIGHT JOIN member m ON s.sid = m.sid;
```

| sid | name | member_sid | club |
| :---: | :--- | :---: | :--- |
| 1 | 小明 | 1 | 舞蹈社 |
| 1 | 小明 | 1 | 篮球社 |
| 2 | 小红 | 2 | 舞蹈社 |
| NULL | NULL | 4 | 棋社 |

- 棋社那行留下了，学生侧是 `NULL` → “入社记录在，学生表对不上”
- 小刚没了（左独有，RIGHT 不管）

`A RIGHT JOIN B` 等价于 `B LEFT JOIN A`（换左右即可），所以生产里少写 RIGHT，统一用 LEFT 更不易晕。

> [!INFO]+ 何时用 RIGHT
>
> 理论上要“以右表为主”。实务更常把主表放到左边，改写成 LEFT。

#### FULL OUTER JOIN（两边的独苗都留）

```sql
SELECT s.sid, s.name, m.sid AS member_sid, m.club
FROM student s
FULL OUTER JOIN member m ON s.sid = m.sid;
```

| sid | name | member_sid | club |
| :---: | :--- | :---: | :--- |
| 1 | 小明 | 1 | 舞蹈社 |
| 1 | 小明 | 1 | 篮球社 |
| 2 | 小红 | 2 | 舞蹈社 |
| 3 | 小刚 | NULL | NULL |
| NULL | NULL | 4 | 棋社 |

= INNER 的匹配行 ∪ 仅左有（小刚）∪ 仅右有（棋社）。

MySQL 长期没有 `FULL OUTER JOIN`，可用 UNION 模拟：

```sql
SELECT s.sid, s.name, m.club
FROM student s
LEFT JOIN member m ON s.sid = m.sid
UNION
SELECT s.sid, s.name, m.club
FROM student s
RIGHT JOIN member m ON s.sid = m.sid;
```

> [!INFO]+ 何时用 FULL
>
> 对账、比对两份名单：看两边都有、只有左、只有右。例如：系统用户表 vs 考勤打卡表。

#### 一眼对照

同一份数据，四种结果差在“独苗要不要”：

| | 小明/小红（两边都有） | 小刚（仅左） | 棋社 sid=4（仅右） |
| :--- | :--- | :--- | :--- |
| INNER | 有 | 丢 | 丢 |
| LEFT | 有 | 有，右列 NULL | 丢 |
| RIGHT | 有 | 丢 | 有，左列 NULL |
| FULL | 有 | 有，右列 NULL | 有，左列 NULL |

> [!EXAMPLE]+ 记法
>
> - INNER = 交集（配得上才要）
> - LEFT = 左表底册 + 能配上的右信息
> - RIGHT = 右表底册 + 能配上的左信息
> - FULL = 两边底册都要（对账）

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

模式需相容：列数相同、类型相容（都选 `name` 才能竖着接在一起）。


还是用一张小 `student`：

| name | department | gender |
| :--- | :--- | :--- |
| 小明 | CS | Male |
| 小红 | CS | Female |
| 小美 | Math | Female |
| 小刚 | EE | Male |

两个子查询各自会得到：

**查询 A：CS 系** → 小明、小红  

**查询 B：女生** → 小红、小美  

注意：**小红两边都有**。差别就出在“叠在一起时要不要去掉这份重复”。

### UNION
（合并后去重）

```sql
SELECT name FROM student WHERE department = 'CS'   -- A：小明、小红
UNION
SELECT name FROM student WHERE gender = 'Female'; -- B：小红、小美
```

结果（小红只留一次）：

| name |
| :--- |
| 小明 |
| 小红 |
| 小美 |

含义：名字出现在「CS 或 女生」里即可，人只报一次。引擎通常要**排序/哈希去重**，所以更慢一点。

### UNION ALL
（合并后不去重）

```sql
SELECT name FROM student WHERE department = 'CS'
UNION ALL
SELECT name FROM student WHERE gender = 'Female';
```

结果（小红出现两次）：

| name |
| :--- |
| 小明 |
| 小红 |
| 小红 |
| 小美 |

含义：把两段结果**直接竖着接起来**，不管有没有重复行。

> [!EXAMPLE]+ 一张图记住
>
> | | 小明（仅 A） | 小红（A 和 B 都有） | 小美（仅 B） | 小刚（都不沾） |
> | :--- | :--- | :--- | :--- | :--- |
> | `UNION` | 1 次 | **1 次** | 1 次 | 无 |
> | `UNION ALL` | 1 次 | **2 次** | 1 次 | 无 |

什么时候用 UNION ALL

| 场景 | 为什么用 ALL |
| :--- | :--- |
| 两段结果**本来就不会重复**（如 2023 流水 ∪ 2024 流水，按年切开） | 去重纯浪费，ALL 更快 |
| **就要保留重复**（两份名单叠在一起做后续 `COUNT`，重复代表“命中两次条件”） | `UNION` 会把信息弄丢 |
| 递归 CTE 里 `UNION ALL` 往下扩一层 | 标准写法，通常不在这里去重 |

```sql
-- 例：把两年订单流水拼成一张长表（订单号跨年不重复 → 用 ALL）
SELECT order_id, amount FROM orders_2023
UNION ALL
SELECT order_id, amount FROM orders_2024;

-- 若误用 UNION：引擎仍会全局去重，白干活，还可能更慢
```

和 OR 的关系：

「CS 或女生」用 `WHERE department='CS' OR gender='Female'` 在**同一张表**上也能写，且每人一行。  
`UNION` / `UNION ALL` 更适合：**两段查询结构不同**（不同表、不同列计算），再竖着合并。

```sql
-- 同表时 OR 往往更直接（结果无重复小红）
SELECT name FROM student
WHERE department = 'CS' OR gender = 'Female';
```

### INTERSECT / EXCEPT（顺带）

```sql
-- INTERSECT：两边都出现 → 只有小红（既是 CS 又是女生）
-- EXCEPT / MINUS：在 A 不在 B → 小明（CS 但不是女生）
-- MySQL 旧版本可能没有，可用 EXISTS / NOT EXISTS 模拟
```

> [!NOTE]+ 面试一句
>
> `UNION` = 并上再去重；`UNION ALL` = 只拼接、不去重、通常更快。需要重复或确定无重复时用 `ALL`。

<br>

## 聚集与分组

把多行收成「每个组一行」的统计（每系人数、均薪等）。

### `GROUP BY` 在干什么

**一句话**：按某列（或多列）的取值，把行**分堆**；每一堆用 `COUNT`/`AVG`/… 收成**一行**。

**student（小数据）**

| sid | name | department | age |
| :---: | :--- | :--- | ---: |
| 1 | 小明 | CS | 20 |
| 2 | 小红 | CS | 22 |
| 3 | 小刚 | EE | 19 |
| 4 | 小美 | EE | 21 |
| 5 | 小强 | Bio | 18 |

没有 `GROUP BY` 时，整表是一堆 5 行。写上 `GROUP BY department` 后，引擎先在脑子里分成三堆：

```text
CS 堆：小明、小红          → 之后收成 1 行
EE 堆：小刚、小美          → 之后收成 1 行
Bio 堆：小强               → 之后收成 1 行
```

```sql
SELECT department, COUNT(*) AS cnt, AVG(age) AS avg_age
FROM student
GROUP BY department;
```

| department | cnt | avg_age |
| :--- | ---: | ---: |
| CS | 2 | 21 |
| EE | 2 | 20 |
| Bio | 1 | 18 |

- `department` 能出现在结果里，是因为它是**分组键**（每堆里这个值相同）。
- `COUNT(*)` / `AVG(age)` 是对**这一堆里的行**算的。
- 不能写 `SELECT name, COUNT(*) ... GROUP BY department`：一堆里有多个 `name`，引擎不知道该留哪一个（严格模式报错）。

> [!INFO]+ `COUNT(*)` 里的 `*` 是什么
>
> - `SELECT *` 的 `*` = 所有**列**。
> - `COUNT(*)` 的 `*` = 按**行**计数（历史写法），**不是**把各列加起来。
> - `COUNT(age)` 只数 `age` 非空的行；`COUNT(*)` 行在就数，不管哪列是 NULL。

### 过滤行 vs 过滤组（同一张表走两遍）

执行顺序要记住这一段：

```text
FROM 取出行
  → WHERE     先扔掉一些「行」     （此时还没有组，也还没有 COUNT/AVG）
  → GROUP BY  把剩下的行分堆
  → 对每堆算 COUNT / AVG / …
  → HAVING    再扔掉一些「组」     （此时才有 COUNT/AVG，按组的统计量筛选）
  → SELECT 输出
```

| | `WHERE` | `HAVING` |
| :--- | :--- | :--- |
| 时机 | **分组之前** | **分组并算完聚集之后** |
| 对象 | 每一**行**（小明这一行要不要） | 每一**组**（CS 这一组要不要） |
| 能否用 `COUNT`/`AVG` | **不能**（组还没形成） | **能** |
| 典型条件 | `age >= 20`、`department = 'CS'` | `COUNT(*) >= 2`、`AVG(age) > 20` |

#### 只过滤行：`WHERE`

题意：只统计 **age ≥ 20** 的学生，再按系数人数。

```sql
SELECT department, COUNT(*) AS cnt
FROM student
WHERE age >= 20          -- 先按「行」筛：小刚19、小强18 被扔掉
GROUP BY department;
```

1. `WHERE` 后剩下：小明20、小红22、小美21（3 行）。
2. 分堆：CS={小明,小红}，EE={小美}；Bio 空堆，不出现。
3. 结果：

| department | cnt |
| :--- | ---: |
| CS | 2 |
| EE | 1 |

注意：小刚虽是 EE，但被 `WHERE` 提前丢掉，**不会**进 EE 的 `COUNT`。

#### 只过滤组：`HAVING`

题意：先按系统计**全部**学生，再只要「至少 2 人」的系。

```sql
SELECT department, COUNT(*) AS cnt
FROM student
GROUP BY department
HAVING COUNT(*) >= 2;    -- 按「组」筛：Bio 只有 1 人，整组扔掉
```

1. 不分 `WHERE`，5 行全进分组 → CS:2，EE:2，Bio:1。
2. `HAVING` 看的是每组的 `COUNT(*)`，不是某一行的 age。
3. 结果：

| department | cnt |
| :--- | ---: |
| CS | 2 |
| EE | 2 |

Bio 组被丢掉时，**小强整个人都不输出**——因为输出单位已经是「组」而不是「行」。

#### 两个一起用（对照）

```sql
-- ① 先丢掉年龄 < 20 的行
-- ② 再按系分组计数
-- ③ 只保留「筛完后仍至少 2 人」的系
SELECT department, COUNT(*) AS cnt
FROM student
WHERE age >= 20
GROUP BY department
HAVING COUNT(*) >= 2;
```

| 步骤 | 还剩什么 |
| :--- | :--- |
| `WHERE age >= 20` | 小明、小红、小美 |
| `GROUP BY` | CS:2，EE:1 |
| `HAVING COUNT(*) >= 2` | 只留 **CS:2**（EE 只剩 1 人，组被丢） |

若把 `HAVING COUNT(*) >= 2` 误写成 `WHERE COUNT(*) >= 2` → **报错**：`WHERE` 阶段没有组，没有 `COUNT`。

> [!WARNING]+ 口诀
>
> - 条件里是**一行上的列**（age、salary、sid）→ 多半 `WHERE`。
> - 条件里是**一组上的统计**（人数、均薪、总学分）→ 必须 `HAVING`（或先写成子查询/CTE 再在外层 `WHERE`）。

规则补充：

- `SELECT` 中未进入聚集的列，必须出现在 `GROUP BY` 中（严格模式）。
- `COUNT(*)` 计行；`COUNT(col)` 忽略该列 NULL；`SUM`/`AVG` 忽略 NULL；输入全空时 `SUM`/`AVG` 常为 NULL，`COUNT(*)` 为 0。

#### 例：系均薪高于全校均薪

**instructor（小数据）**

| name | dept_name | salary |
| :--- | :--- | ---: |
| 张三 | CS | 90 |
| 李四 | CS | 70 |
| 王五 | EE | 80 |
| 赵六 | EE | 100 |
| 钱七 | Bio | 60 |

```sql
-- 系均薪高于全校均薪的系
SELECT dept_name, AVG(salary) AS avg_salary
FROM instructor
GROUP BY dept_name
HAVING AVG(salary) > (SELECT AVG(salary) FROM instructor);
```

带入数值：

1. 全校均薪（标量子查询，不分系）：$(90+70+80+100+60)/5 = 80$。
2. 各组均薪：CS $(90+70)/2 = 80$；EE $(80+100)/2 = 90$；Bio $= 60$。
3. `HAVING AVG(salary) > 80` → 只留 **EE**（90）。CS 等于 80 不进；Bio 低于不进。

| dept_name | avg_salary |
| :--- | ---: |
| EE | 90 |

#### 例：每个系参加社团的学生百分比

**student**

| sid | name | department |
| :---: | :--- | :--- |
| 1 | 小明 | CS |
| 2 | 小红 | CS |
| 3 | 小刚 | EE |
| 4 | 小美 | EE |
| 5 | 小强 | Bio |

**member**（只列出 sid；一人可多社，故后面用 `DISTINCT`）

| sid | cid |
| :---: | :---: |
| 1 | 10 |
| 1 | 20 |
| 2 | 10 |
| 4 | 10 |

```sql
-- WHERE 先留下“参加过社团”的学生，再按系计数
-- 分母是全校人数（标量子查询）；面试常追问是否应改成本系人数
SELECT department,
       COUNT(DISTINCT sid) * 100.0 / (SELECT COUNT(*) FROM student) AS percentage
FROM student
WHERE sid IN (SELECT sid FROM member)
GROUP BY department;
```

带入数值：

1. `SELECT sid FROM member` → `{1, 2, 4}`（去重后；小明两社仍算一人）。
2. `WHERE sid IN (...)` 后剩下：小明(CS)、小红(CS)、小美(EE)。小刚、小强未入社被丢掉。
3. 分母：全校 `COUNT(*)` $= 5$。
4. 按系：CS 有 2 人 → $2 \times 100.0 / 5 = 40$；EE 有 1 人 → $1 \times 100.0 / 5 = 20$；Bio 无人入社 → **无行**。

| department | percentage |
| :--- | ---: |
| CS | 40 |
| EE | 20 |

若分母改成「本系人数」：CS 为 $2/2=100$，EE 为 $1/2=50$（题意不同，面试常追问这一点）。

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

自然语言「**所有**……都……」，SQL 没有 `FOR ALL`。标准改写：

$$
\text{对所有 } x,\ P(x)
\quad\Longleftrightarrow\quad
\text{不存在 } x,\ \text{使得 }\neg P(x)
$$

口语：**全部成立** ⟺ **找不到反例**。SQL 用 `NOT EXISTS` 写“找不到”。

### 积木：EXISTS 的真假（先死记）

```sql
EXISTS (子查询)      -- 子查询能取出 ≥1 行 → 整个为 TRUE
NOT EXISTS (子查询)  -- 子查询 0 行         → 整个为 TRUE
```

相关子查询：里层用到外层当前行（如 `student.sid`）。外层换一个人，里层就重跑一遍。

### 例 1：参加了 JL SUN 的「所有」俱乐部

#### 题目

找出：JL SUN 管的每一个社，这个学生都进了。  
（不是“至少进一个”，是“一个都不能少”。）

#### 小数据

**club**

| cid | name | supervisor |
| :---: | :--- | :--- |
| 10 | 舞蹈社 | JL SUN |
| 20 | 篮球社 | JL SUN |
| 30 | 棋社 | 别人 |

**member**

| sid | cid | 含义 |
| :---: | :---: | :--- |
| 1 | 10 | 小明 → 舞蹈 |
| 1 | 20 | 小明 → 篮球 |
| 2 | 10 | 小红 → 舞蹈（没进篮球） |

- JL SUN 的全集 = {10, 20}（棋社 30 不管）
- 小明两个都有 → **要**
- 小红缺 20 → **不要**

#### 先写成人话三层（不要急着看 SQL）

对**某一个学生 S**：

```text
第 0 层  正在考察学生 S

第 1 层  问：存不存在「坏俱乐部」？
         坏俱乐部 = （归 JL SUN 管）并且（S 没参加）
         若一个坏的都没有 → S 合格，选出来

第 2 层  怎么判断「S 没参加俱乐部 C」？
         去 member 里找 (S, C)
         找到了 → 参加了
         找不到 → 没参加
```

对应到双重否定：

```text
S 合格
  ⟺ 不存在坏俱乐部 C
  ⟺ 不存在 C，使得：C 是 JL SUN 的  且  S 没参加 C
  ⟺ 不存在 C，使得：C 是 JL SUN 的  且  member 里没有 (S,C)
```

#### SQL：每一层在干什么（对着缩进看）

```sql
SELECT student.name
FROM student
-- ========== 第 0 层：循环每个学生 ==========
-- 下面 WHERE 对「当前这个 student」为真才输出
WHERE NOT EXISTS (
    -- ========== 第 1 层：找「坏俱乐部」==========
    -- NOT EXISTS(...) 为真  ⟺  这个括号里 0 行  ⟺  没有坏俱乐部
    SELECT club.cid
    FROM club
    WHERE club.supervisor = 'JL SUN'   -- 只从 JL SUN 的社里找坏的
      AND NOT EXISTS (
          -- ========== 第 2 层：当前学生有没有进「当前这个 club」==========
          -- 内层 EXISTS 语义：member 里有没有 (student.sid, club.cid)
          -- NOT EXISTS 为真  ⟺  一行都没有  ⟺  「没参加」
          -- 于是：JL SUN 的社 + 没参加 → 这一行 club 被第 1 层选中（坏俱乐部）
          SELECT 1
          FROM member
          WHERE member.sid = student.sid   -- 钉死第 0 层的学生
            AND member.cid = club.cid      -- 钉死第 1 层的俱乐部
      )
);
```

> [!WARNING]+ 最容易晕的一点
>
> 第 2 层的 `NOT EXISTS`：**不是**最终答案，只负责定义「没参加」。  
> 真正决定「选不选这个学生」的，是第 1 层外面的那个 `NOT EXISTS`（有没有坏社）。

#### 真假跳变表（建议抄一遍）

固定学生 S、俱乐部 C：

| member 里有 (S,C)？ | 第 2 层 `NOT EXISTS` | 含义 | C 会不会进第 1 层结果（在 JL SUN 前提下） |
| :--- | :--- | :--- | :--- |
| 有 | **假** | 参加了 | 不会（AND 短路掉） |
| 没有 | **真** | 没参加 | **会**（这是坏俱乐部） |

再往外：

| 第 1 层能查出坏俱乐部吗 | 第 1 层外的 `NOT EXISTS` | 学生 S |
| :--- | :--- | :--- |
| 能（≥1 个坏社） | **假** | 丢掉 |
| 不能（0 个坏社） | **真** | **留下** |

#### 用小明走（应留下）

1. 第 0 层：S = 小明 (sid=1)。
2. 扫 JL SUN 的社。
3. C=10：member 有 (1,10) → 第 2 层 `NOT EXISTS`=假 → 10 不是坏社。
4. C=20：有 (1,20) → 同理不是坏社。
5. 第 1 层结果空 → 外层 `NOT EXISTS`=真 → **输出小明**。

#### 用小红走（应丢掉）

1. S = 小红 (sid=2)。
2. C=10：有 (2,10) → 不是坏社。
3. C=20：没有 (2,20) → 第 2 层 `NOT EXISTS`=真 → **20 是坏社**，第 1 层查到一行。
4. 外层 `NOT EXISTS`=假 → **不输出小红**。

> [!EXAMPLE]+ 嵌套括号（人话版）
>
> ```text
> 对每个学生 S：
>   若不存在俱乐部 C，使得
>        ( C.supervisor = 'JL SUN'
>          且  不存在 member 行 (S 进了 C) )
>   则选出 S
> ```

### 例 2：只有女生的俱乐部（单层，练手）

只要一层：反例 =「社里有个非女生」。

```sql
SELECT club.name
FROM club
-- 对每个俱乐部 C：
WHERE NOT EXISTS (
    -- 反例成员：属于 C，且性别不是女
    SELECT 1
    FROM member
    JOIN student ON member.sid = student.sid
    WHERE member.cid = club.cid
      AND student.gender <> 'Female'
);
-- 找不到反例 → 选出（注意：空社也「找不到反例」会被选出）
```

不要空社时再加：

```sql
AND EXISTS (SELECT 1 FROM member WHERE member.cid = club.cid)
```

### 例 3：选了 Comp. Sci.「全部」课（与例 1 同构）

把「JL SUN 的社」换成「CS 系的课」，结构一字不差：

```sql
SELECT DISTINCT s.ID
FROM student s
WHERE NOT EXISTS (                      -- 第 1 层：不存在「漏选的 CS 课」
    SELECT course_id
    FROM course
    WHERE dept_name = 'Comp. Sci.'      -- 全集
      AND NOT EXISTS (                  -- 第 2 层：该生没选这门 → 这门是坏课
          SELECT 1
          FROM takes t
          WHERE t.ID = s.ID
            AND t.course_id = course.course_id
      )
);
```

验算可用计数（同一门不重复计时）：

```sql
-- 选中的 CS 课门数 = CS 课总门数 → 全选了
SELECT t.ID
FROM takes t
JOIN course c ON t.course_id = c.course_id
WHERE c.dept_name = 'Comp. Sci.'
GROUP BY t.ID
HAVING COUNT(DISTINCT t.course_id) = (
    SELECT COUNT(*) FROM course WHERE dept_name = 'Comp. Sci.'
);
```

> [!NOTE]+ 认题
>
> | 题意 | 全集 | 反例 | 几层 NOT EXISTS |
> | :--- | :--- | :--- | :--- |
> | 所有 JL SUN 社都参加 | JL SUN 的 club | 某社没进 | 两层 |
> | 只有女生 | 该社成员 | 有个非女生 | 一层 |
> | 全部 CS 课都选了 | CS 的 course | 某门没选 | 两层 |
>
> 口诀：全称 → 找反例 → 外层 `NOT EXISTS`；反例若是「缺一条关联记录」→ 里层再 `NOT EXISTS`。

<br>

## WITH 与 CTE

### CTE 是什么

**CTE**（Common Table Expression，公用表表达式）= 给一段查询起临时名字，写在 `WITH` 里，后面像用表一样用。

复杂逻辑若全塞进一层层括号，阅读成本高；拆成「先算出中间表 → 再查」更清楚。作用域**只在本条 SQL**，不是永久建表。

### 读法：派生表 vs CTE（同一题两写法）

目标不变：找出 Comp. Sci. 系里 `salary > 80000` 的教师姓名。

**写法 A：派生表（括号嵌套，由内向外读）**

```sql
SELECT name                          -- ← 第 2 层：最终要的列
FROM (
    -- ========== 第 1 层（里层）：先缩小范围 ==========
    SELECT ID, name, salary
    FROM instructor
    WHERE dept_name = 'Comp. Sci.'
) AS t                              -- 里层结果临时叫 t
WHERE salary > 80000;               -- ← 第 2 层：再在 t 上过滤
```

执行顺序：① 算出括号里的表 t → ② 对 t 做外层 `WHERE`/`SELECT`。

**写法 B：CTE（先定义后使用，从上往下读）**

```sql
-- ========== 第 1 步：给中间结果起名（还没最终输出）==========
WITH cse_instructors AS (
    SELECT ID, name, salary
    FROM instructor
    WHERE dept_name = 'Comp. Sci.'
)
-- ========== 第 2 步：像查真表一样查这个名字 ==========
SELECT name
FROM cse_instructors
WHERE salary > 80000;
```

语义与写法 A **完全等价**；差别只在书写顺序：CTE 把「里层」提到语句最上面。

> [!EXAMPLE]+ 小数据走一遍
>
> 假设 `instructor` 有：
>
> | name | dept_name | salary |
> | :--- | :--- | ---: |
> | 张三 | Comp. Sci. | 90000 |
> | 李四 | Comp. Sci. | 70000 |
> | 王五 | Biology | 95000 |
>
> 1. 第 1 步 CTE → 临时表只有张三、李四（王五被 dept 滤掉）。
> 2. 第 2 步 `salary > 80000` → 只剩**张三**。

### 多个 CTE：流水线，不是洋葱括号

逗号分隔多个定义；**后面的可以引用前面的**，像传送带：

```sql
-- 传送带第 1 站：CS 教师
WITH cse AS (
    SELECT ID, name, salary
    FROM instructor
    WHERE dept_name = 'Comp. Sci.'
),
-- 传送带第 2 站：吃第 1 站的输出，再筛高薪
cse_high AS (
    SELECT * FROM cse WHERE salary > 80000
)
-- 最终消费者：只读第 2 站
SELECT name FROM cse_high;
```

| 步骤 | 名字 | 输入 | 输出（相对上例） |
| :--- | :--- | :--- | :--- |
| 1 | `cse` | `instructor` | 张三、李四 |
| 2 | `cse_high` | `cse` | 张三 |
| 3 | 主查询 | `cse_high` | 张三 |

### CTE vs 视图 vs 子查询

| | 存活多久 | 怎么读 |
| :--- | :--- | :--- |
| 子查询 / 派生表 | 仅本句 | 往往要由内向外抠括号 |
| `WITH` CTE | 仅本句 | 从上往下，先定义后用 |
| `VIEW` 视图 | 建完一直在库里 | 像真表，多语句可反复用 |

### 递归 CTE：锚点 + 递归臂

用于走**树/链表**：组织架构、评论回复、账单拆分。方言需支持 `WITH RECURSIVE`。

固定两段，用 `UNION ALL` 拼接：

```text
临时表 = 锚点（起点，只跑一次）
         ∪
         递归臂（用「已进临时表的行」再扩下一批，反复跑）
         ∪
         …
         直到某一轮 0 行新结果 → 停
```

#### 小组织树

**emp**

| id | name | manager_id |
| :---: | :--- | :---: |
| 1 | 老板 | NULL |
| 2 | 经理A | 1 |
| 3 | 经理B | 1 |
| 4 | 员工甲 | 2 |
| 5 | 员工乙 | 2 |

树形：`老板 → 经理A → 员工甲/乙`，以及 `老板 → 经理B`。

目标：列出 **id=1 老板的全部下属**（含间接）。

```sql
WITH RECURSIVE subordinates AS (

    -- ========== 锚点（只跑一次）：把起点放进结果 ==========
    SELECT id, manager_id, name, 0 AS lvl
    FROM emp
    WHERE id = 1

    UNION ALL

    -- ========== 递归臂（反复跑）==========
    -- 含义：在 emp 里找「上级已经在 subordinates 里」的人
    SELECT e.id, e.manager_id, e.name, s.lvl + 1
    FROM emp e
    JOIN subordinates s          -- 引用「正在定义的同一个 CTE」已算出的部分
      ON e.manager_id = s.id     -- 新人的老板 = 上一层已入选的人
)
SELECT * FROM subordinates;
```

#### 按轮次展开（对着 JOIN 条件看）

| 轮次 | `s`（已有） | `e.manager_id = s.id` 对上谁 | 本轮新增 |
| :--- | :--- | :--- | :--- |
| 锚点 | — | — | (1, 老板, lvl0) |
| 第 1 轮 | 只有老板 id=1 | 经理A、经理B 的 manager_id=1 | (2,A,1)、(3,B,1) |
| 第 2 轮 | 含 1、2、3 | 甲/乙 的 manager_id=2 | (4,甲,2)、(5,乙,2) |
| 第 3 轮 | 含全部 | 无人再挂在已有 id 下 | **空 → 停止** |

最终 `SELECT *` 得到上述全部累计行（含老板本人；只要下属可加 `WHERE lvl > 0`）。

> [!WARNING]+ 最容易晕的一点
>
> 递归臂里的 `JOIN subordinates` **不是**去查另一张永久表，而是引用**本 CTE 到目前为止已算出的行**。  
> 每一轮只拿「上一轮新进的行」去扩下一层（实现上可能优化，语义上按「越扩越大」理解即可）。

<br>

## 窗口函数

### 和 GROUP BY 差在哪

| | `GROUP BY` | 窗口 `OVER(...)` |
| :--- | :--- | :--- |
| 行数 | 多行收成**更少行** | **行数不变**，旁边多挂几列 |
| 要什么 | 「每系均薪」一张汇总表 | 「每个教师仍一行 + 本系均薪/名次」 |

同一批数据对比：

```sql
-- GROUP BY：CS 三行 → 收成 1 行
SELECT dept_name, AVG(salary) AS dept_avg
FROM instructor
GROUP BY dept_name;

-- 窗口：CS 仍是 3 行，每行多一列 dept_avg（值相同）
SELECT name, dept_name, salary,
       AVG(salary) OVER (PARTITION BY dept_name) AS dept_avg
FROM instructor;
```

### 语法：从外往里拆

```sql
函数(...) OVER (
    PARTITION BY 列    -- ① 先按谁切成互不干扰的「组」
    ORDER BY 列        -- ② 组内怎么排（排名、累计需要）
    窗口帧             -- ③ 累计从哪一行加到哪一行（可先忽略）
)
```

读一条带窗口的 `SELECT` 时：先看 `FROM` 得到明细行 → 再对每一行，按 `OVER` 规则算出旁注列。

### 小数据：分层算 RANK 与 AVG

**instructor（片段）**

| name | dept_name | salary |
| :--- | :--- | ---: |
| 张三 | CS | 90 |
| 李四 | CS | 70 |
| 王五 | CS | 90 |
| 赵六 | EE | 80 |

```sql
SELECT
    name,
    dept_name,
    salary,
    -- ========== 旁注列 ①：组内排名 ==========
    -- 对「当前行」：只看同 dept 的行，按 salary 降序排，给出名次
    RANK() OVER (
        PARTITION BY dept_name   -- 切开：CS 一组、EE 一组
        ORDER BY salary DESC     -- 组内：薪高在前
    ) AS rk,
    -- ========== 旁注列 ②：组内平均 ==========
    -- 对「当前行」：同 dept 所有 salary 求平均，抄到这一行（不删行）
    AVG(salary) OVER (
        PARTITION BY dept_name
    ) AS dept_avg
FROM instructor;
```

#### 只盯 CS 组（张三这一行怎么得到 rk、dept_avg）

1. `PARTITION BY dept_name` → 当前窗口成员 = {张三90, 李四70, 王五90}。
2. `ORDER BY salary DESC` → 次序：张三与王五并列最高，李四最低。
3. `RANK()`：并列都给 1，下一个空号到 3 → 张三 rk=1，王五 rk=1，李四 rk=**3**。
4. `AVG`：$(90+70+90)/3 \approx 83.3$，三行都抄同一值。

逻辑结果：

| name | dept | salary | rk | dept_avg |
| :--- | :--- | ---: | ---: | ---: |
| 张三 | CS | 90 | 1 | 83.3… |
| 王五 | CS | 90 | 1 | 83.3… |
| 李四 | CS | 70 | 3 | 83.3… |
| 赵六 | EE | 80 | 1 | 80 |

`DENSE_RANK` 不跳号（李四会是 2）。`ROW_NUMBER` 强制唯一序号（并列也拆开）。

### 累计和：窗口帧在干什么

```sql
SELECT name, salary,
       SUM(salary) OVER (
           ORDER BY salary
           -- 帧：从排序后的「第一行」一直加到「当前行」
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running
FROM instructor;
```

假设按 salary 排序后为：李四70 → 赵六80 → 张三90 → 王五90，则 `running` 依次为 70、150、240、330（具体并列次序依实现/排序稳定性）。

### 嵌套一层：每组 Top-K

窗口**不能**直接在同一层 `WHERE rk <= 1`（`WHERE` 早于窗口计算）。正确读法是「先开窗，再包一层」：

```sql
-- ========== 外层：只留 rk=1 ==========
SELECT name, dept_name, salary, rk
FROM (
    -- ========== 内层：先给每行算 rk ==========
    SELECT
        name, dept_name, salary,
        RANK() OVER (
            PARTITION BY dept_name
            ORDER BY salary DESC
        ) AS rk
    FROM instructor
) AS t
WHERE rk = 1;
-- CS 并列第一会留下张三、王五两行；只要一人可用 ROW_NUMBER
```

常用：`ROW_NUMBER`、`RANK`/`DENSE_RANK`、`LAG`/`LEAD`（上一行/下一行）、`SUM/AVG/... OVER`。

> [!EXAMPLE]+ 选型
>
> - 只要汇总表 → `GROUP BY`
> - 明细 + 旁注 → 窗口
> - 每组 Top-K → 内层窗口出 `rk`，外层再 `WHERE rk <= K`

<br>

## NULL 与三值逻辑

`NULL` = 未知/缺失，**不是** 0，**不是** `''`。

比较运算遇到 `NULL` → 结果是 **UNKNOWN**（第三种逻辑值，不是 TRUE/FALSE）。  
`WHERE` / `HAVING` **只保留结果为 TRUE 的行**；UNKNOWN 与 FALSE 一样被丢掉。

| 表达式 | 结果 | 进不进 `WHERE` |
| :--- | :--- | :--- |
| `1 = NULL` | UNKNOWN | 不进 |
| `NULL = NULL` | UNKNOWN | 不进 |
| `age = age`（该行 age 为 NULL） | UNKNOWN | 不进 |
| `age IS NULL` | TRUE | **进** |
| `age IS NOT NULL` | 视情况 | 非空才进 |

```sql
-- 错：= NULL 永远不是 TRUE，筛不出「空年龄」
-- SELECT * FROM student WHERE age = NULL;

-- 对：专门判断空 / 非空
SELECT * FROM student WHERE age IS NULL;
SELECT * FROM student WHERE age IS NOT NULL;

-- 展示时把空换成默认值（原表未改）
SELECT name, COALESCE(age, 0) AS age_show FROM student;
```

> [!EXAMPLE]+ 小数据
>
> | name | age |
> | :--- | ---: |
> | 小明 | 20 |
> | 小红 | NULL |
>
> `WHERE age = 20` → 只有小明。  
> `WHERE age <> 20` → **没有小红**（`NULL <> 20` 是 UNKNOWN）。  
> 要「年龄不是 20（含未知）」需另写逻辑，例如 `age IS NULL OR age <> 20`。

聚集：`COUNT(*)` 计行；`COUNT(age)` / `AVG(age)` **跳过** age 为 NULL 的行。

<br>

## 视图

视图 = 把一段 `SELECT` **存进库里的名字**，之后当表名用。

| | CTE | 视图 |
| :--- | :--- | :--- |
| 寿命 | 本条 SQL 结束即消失 | `CREATE` 后一直在 |
| 典型用途 | 单条语句内分层 | 多语句复用、授权裁剪 |

```sql
-- ========== 第 1 步：定义（执行一次，写入数据字典）==========
CREATE VIEW cs_student AS
SELECT sid, name, age
FROM student
WHERE department = 'CS';

-- ========== 第 2 步：查询（引擎内部展开成下面的等价式）==========
SELECT * FROM cs_student WHERE age > 20;

-- 大致等价于：
-- SELECT sid, name, age
-- FROM student
-- WHERE department = 'CS' AND age > 20;
```

用途：少写重复过滤条件；只把视图的 `SELECT` 权限授给某角色（看不到其它列/行）；基表改名/拆分时少改应用 SQL。

> [!NOTE]+ 视图不是备份
>
> 默认**不存**数据副本；基表变，再查视图结果就变。物化视图是另一话题。

<br>

## 完整性与触发器

声明式约束（主键、外键、`CHECK`）见 DDL「键与约束」。  
**触发器** = 对表发生 `INSERT`/`UPDATE`/`DELETE` 时，由引擎**自动**执行的一段过程。

### 时机：一层层发生什么

```text
客户端发出：UPDATE instructor SET salary = ... WHERE ...
        │
        ▼
  ① BEFORE 触发器（若有）—— 尚可改 NEW，或拒绝本次修改
        │
        ▼
  ② 真正写入基表
        │
        ▼
  ③ AFTER 触发器（若有）—— 记审计日志、级联写其它表等
```

### 示例：涨薪过大时做处理

```sql
CREATE TRIGGER bump_salary
AFTER UPDATE OF salary ON instructor  -- 只关心改了 salary 的 UPDATE
REFERENCING OLD ROW AS o              -- o = 改之前的那一行
            NEW ROW AS n              -- n = 改之后的那一行
FOR EACH ROW                          -- 每更新一行，触发一次
WHEN (n.salary > o.salary * 1.5)      -- 仅当涨幅超过 50% 才进入主体
BEGIN
    -- 主体：写审计表 / SIGNAL 报错 等（具体语法随方言）
END;
```

读法：先看 `AFTER UPDATE OF salary`（何时）→ 再看 `WHEN`（哪些行）→ 最后看 `BEGIN...END`（做什么）。

<br>

## 权限

（DCL）控制**谁**能对**哪个对象**做**什么操作**。

```sql
-- ① 建角色（权限的篮子，还不是某个具体登录名）
CREATE ROLE analyst;

-- ② 往篮子里放特权：可读 student
GRANT SELECT ON student TO analyst;

-- ③ 把篮子挂到用户：alice 拥有 analyst 里的全部特权
GRANT analyst TO alice;

-- ④ 列级特权 + 转授：bob 可改 instructor.salary，且可再授给别人
GRANT UPDATE (salary) ON instructor TO bob WITH GRANT OPTION;

-- ⑤ 收回：CASCADE 表示连带收回已转授出去的同权
REVOKE SELECT ON student FROM alice CASCADE;
```

层次可记：`特权 → 角色 → 用户`。原则：够用即可，避免长期 `WITH GRANT OPTION` 扩散。

<br>


## SQL 注入

（SQL Injection）

把不可信输入直接拼进 SQL 字符串时，攻击者可改写语句语义。权限管“谁能执行什么”，注入管的是“语句本身是否仍是开发者写的那一句”。面试与安全基线都会问。

### 定义

应用若用字符串拼接构造查询，例如把登录框里的用户名直接粘进：

```text
"SELECT * FROM users WHERE name = '" + 用户输入 + "'"
```

则输入不再只是“数据”，而可能变成 SQL 的一段语法（多一个引号、多一个 `OR`、多一句语句）。这就是 SQL 注入。

为什么会出现：早期图省事用拼接；模板引擎/ORM 用错；动态表名/排序字段白名单没做好。

### 经典形态

#### 登录绕过

意图：按姓名查用户。

```sql
-- 开发者心里的语句
SELECT * FROM users WHERE name = 'alice' AND pass = 'secret';
```

若 `name` 来自输入且拼成字符串，恶意输入形如：

```text
alice' OR '1'='1
```

拼出的效果等价于（示意）：

```sql
SELECT * FROM users WHERE name = 'alice' OR '1'='1' AND pass = '...';
-- 逻辑被改写：OR 恒真部分可能让 WHERE 整体放行（具体取决于 AND/OR 优先级与后续片段）
```

更常见的教材写法是密码处注入 `' OR '1'='1' --`，用注释吃掉后半句。要点只有一个：**输入改变了语句结构**。

#### 联合查询拖库

在原本只返回少量列的查询后，用 `UNION SELECT ...` 拼出额外结果集，试图读其他表。前提往往是：注入点在 `SELECT` 查询中、列数/类型可对齐、错误信息或页面回显可利用。

#### 盲注

页面不回显查询结果，但可通过真/假条件导致的页面差异、时间延迟等推断数据。防御思路与显式注入相同：不让输入变成语法。

> [!WARNING]+ 说明
>
> 上文只为理解机制与面试问答。对未授权系统做注入测试违法；实验请用自建库或专门靶场（如 DVWA、本地 Docker 练习环境）。

### 根因

| 错误做法 | 问题 |
| :--- | :--- |
| 字符串拼接 SQL | 数据与代码边界消失 |
| 仅过滤关键词（禁 `OR`/`SELECT`） | 易绕过，黑名单不可靠 |
| 前端校验当唯一防线 | 请求可被直接重放、绕过浏览器 |
| 数据库账号权限过大 | 一旦注入，破坏面从“读一行”扩大到“删库、读系统表” |

### 防护

#### 参数化查询 / 预编译（首选）

把 SQL 结构固定，用户输入只作为**绑定参数**，不再参与解析。

```sql
-- 语句形态固定，? 或 :name 只代表“值”的占位
SELECT * FROM users WHERE name = ? AND pass = ?;
```

应用侧（示意，非某一语言绑定）：

```text
prepare("SELECT * FROM users WHERE name = ? AND pass = ?")
bind(1, 用户输入的姓名)   -- 哪怕内容是 alice' OR '1'='1，也只当普通字符串比较
bind(2, 用户输入的密码)
execute
```

JDBC / PDO / MyBatis `#{}` / ORM 正确用法都属于这一路。注意：有的模板 `${}` 仍是拼接，用错等于没防。

#### 其他配套

| 手段 | 作用 |
| :--- | :--- |
| 最小权限 | 应用账号不要用 DBA；按需 `GRANT`，即使注入也难 `DROP DATABASE` |
| 输入校验 | 长度、类型、白名单（如排序字段只允许 `age`/`name` 几个字面量） |
| 避免直接拼动态标识符 | 表名、列名不能参数化时，必须白名单映射，禁止用户字符串原样进 SQL |
| 错误信息不回显 | 生产环境不把堆栈/SQL 原文返回给浏览器，降低探测信息 |
| WAF / 审计 | 辅助层，不能替代参数化 |

> [!EXAMPLE]+ 安全写法对照
>
> **危险（拼接）**
>
> ```text
> sql = "SELECT * FROM student WHERE sid = '" + sid + "'"
> ```
>
> **安全（参数化）**
>
> ```text
> sql = "SELECT * FROM student WHERE sid = ?"
> bind(sid)   -- sid 无论含引号还是 OR，都不会改变 WHERE 结构
> ```

### 面试要点

1. 定义：不可信输入进入 SQL 并改变语义。
2. 主因：拼接；主防：预编译/参数绑定。
3. 黑名单过滤不够；权限最小化能降损。
4. ORM 不等于自动安全，仍可能拼字符串或误用拼接占位符。

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


