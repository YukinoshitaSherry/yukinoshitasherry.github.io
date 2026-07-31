---
title: SQL详解
date: 2024-06-30
categories:
  - 上浙大
tags:
  - DB
  - SQL
desc: 数据库 SQL 专项：DDL/DML、JOIN 全家、嵌套与相关子查询、NOT EXISTS 全称量化、窗口与 CTE、视图触发器权限；对齐浙大作业与面试难度，外链菜鸟教程与牛客。
---

## 总览

### 定位

本文是 [数据库系统笔记](/数据库系统/) 的 SQL 独立册：例子密度按作业 + 面试组织，覆盖复杂嵌套与“全部 / 仅有”类语义。理论侧（关系代数、优化、事务）见主笔记。

外链：

- [菜鸟教程 SQL](https://www.runoob.com/sql/sql-tutorial.html)
- [菜鸟教程 JOIN](https://www.runoob.com/sql/sql-join.html)
- [牛客 SQL 篇](https://www.nowcoder.com/exam/oj?tab=SQL%E7%AF%87&topicId=199)

### 示例模式

全文默认使用下列模式（与浙大 Quiz 风格一致）：

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

常见类型：`CHAR(n)`、`VARCHAR(n)`、`INT`、`SMALLINT`、`NUMERIC(p,d)`、`REAL`/`DOUBLE`、`FLOAT(n)`、`DATE`/`TIME`/`TIMESTAMP`、`BLOB`/`CLOB`。几乎所有类型允许 `NULL`，除非声明 `NOT NULL`。

### 改表与删表

```sql
ALTER TABLE student ADD resume VARCHAR(256);
ALTER TABLE student DROP COLUMN resume;

DROP TABLE member;   -- 表结构删除
DELETE FROM member;  -- 仅清空数据，表仍在
TRUNCATE TABLE member; -- 产品相关：快速清空
```

### 索引（语法级）

```sql
CREATE INDEX idx_student_dept ON student(department);
DROP INDEX idx_student_dept;  -- 语法因方言而异
```

物理意义与 B+ 树见主笔记索引章。参考：[菜鸟教程 CREATE INDEX](https://www.runoob.com/sql/sql-create-index.html)。

<br>

## DML 基础

### 插入

```sql
INSERT INTO student VALUES (1, 'Alice', 20, 'Female', 'CS');
INSERT INTO student (sid, name, department) VALUES (2, 'Bob', 'Math');

INSERT INTO student (sid, name, age, gender, department)
SELECT sid + 1000, name, age, gender, department
FROM student
WHERE department = 'CS';
```

### 更新与 CASE

```sql
UPDATE instructor
SET salary = CASE
    WHEN salary <= 100000 THEN salary * 1.05
    ELSE salary * 1.03
END;
```

### 删除

```sql
DELETE FROM member WHERE cid = 10;
DELETE FROM student WHERE age < 16;
```

<br>

## 单表查询

### SELECT 骨架

```sql
SELECT DISTINCT department
FROM student
WHERE age BETWEEN 18 AND 22
  AND name LIKE 'A%'
ORDER BY department ASC;
```

要点：

- `*` 表示全部列；`DISTINCT` 去重；默认 `ALL` 保留重复（多重集）。
- `SELECT` 列表可含表达式：`salary/12 AS monthly`。
- 关键字大小写不敏感；字符串比较是否敏感依排序规则/方言。
- `LIKE`：`%` 任意串，`_` 单字符。参考：[菜鸟教程 LIKE](https://www.runoob.com/sql/sql-like.html)。

### WHERE 常用谓词

```sql
SELECT * FROM student WHERE department IN ('CS', 'EE');
SELECT * FROM student WHERE age IS NULL;
SELECT * FROM student WHERE age IS NOT NULL;
SELECT * FROM instructor WHERE salary > 50000 AND dept_name <> 'Finance';
```

三值逻辑：与 `NULL` 比较得 `UNKNOWN`；`WHERE` 只保留真。判断空必须用 `IS NULL` / `IS UNKNOWN`（后者较少用）。

### ORDER BY 与限行

```sql
SELECT name, age
FROM student
ORDER BY age DESC, name ASC
LIMIT 1;          -- MySQL / PG 常用
-- SQL Server: SELECT TOP 1 ...
-- 标准 OFFSET/FETCH 依方言
```

“最大差距两人”类题：自连接 + 排序 + `LIMIT`（Quiz 套路）。

```sql
SELECT s1.name, s2.name, ABS(s1.age - s2.age) AS age_difference
FROM student s1, student s2
WHERE s1.sid < s2.sid
ORDER BY age_difference DESC
LIMIT 1;
```

<br>

## 连接

参考图示与分类：[菜鸟教程 JOIN](https://www.runoob.com/sql/sql-join.html)。

### 笛卡尔积与旧式内连接

```sql
SELECT student.name, club.name
FROM student, member, club
WHERE student.sid = member.sid
  AND member.cid = club.cid
  AND student.department = 'CS'
  AND club.name = 'Dancing';
```

`FROM` 多表先理解为笛卡尔积，再用 `WHERE` 过滤——优化器会改写，但语义如此。

### INNER / LEFT / RIGHT / FULL

```sql
-- 内连接：仅匹配行
SELECT s.name, c.name AS club_name
FROM student s
INNER JOIN member m ON s.sid = m.sid
INNER JOIN club c ON m.cid = c.cid;

-- 左外：保留学生，无社团则 club 侧为空
SELECT s.name, c.name AS club_name
FROM student s
LEFT JOIN member m ON s.sid = m.sid
LEFT JOIN club c ON m.cid = c.cid;

-- 右外 / 全外：方言支持度不同（MySQL 长期无 FULL）
SELECT * FROM student s
RIGHT JOIN member m ON s.sid = m.sid;

SELECT * FROM student s
FULL OUTER JOIN member m ON s.sid = m.sid;
```

### NATURAL / USING / ON

```sql
SELECT * FROM instructor NATURAL JOIN department;
SELECT * FROM teaches JOIN course USING (course_id);
SELECT * FROM teaches t JOIN course c ON t.course_id = c.course_id;
```

> [!WARNING]+ NATURAL JOIN
>
> 按**全部同名列**等值连接。列名一不小心同名即误连。作业与生产更推荐显式 `ON`。

### 自连接

```sql
SELECT a.name AS emp, b.name AS manager
FROM emp a
JOIN emp b ON a.manager_id = b.id;
```

<br>

## 集合运算

```sql
SELECT name FROM student WHERE department = 'CS'
UNION
SELECT name FROM student WHERE gender = 'Female';

-- UNION 去重；UNION ALL 保留重复
-- INTERSECT / EXCEPT（MySQL 8+ 支持程度需核对；可用 EXISTS 模拟）
```

模式需相容：列数相同、类型相容。

<br>

## 聚集与分组

```sql
SELECT department, COUNT(*) AS cnt, AVG(age) AS avg_age
FROM student
GROUP BY department
HAVING COUNT(*) >= 3;
```

规则：

- `SELECT` 中未进入聚集的列，必须出现在 `GROUP BY` 中（严格模式）。
- `WHERE` 在分组前过滤行；`HAVING` 在分组后过滤组。
- `COUNT(*)` 计行（含全 NULL 行）；`COUNT(col)` 忽略该列 NULL；`SUM`/`AVG` 等忽略 NULL，输入全空时结果常为 NULL（`COUNT` 为 0）。

```sql
SELECT dept_name, AVG(salary) AS avg_salary
FROM instructor
GROUP BY dept_name
HAVING AVG(salary) > (SELECT AVG(salary) FROM instructor);
```

“每个系参加社团的学生百分比”：

```sql
SELECT department,
       COUNT(DISTINCT sid) * 100.0 / (SELECT COUNT(*) FROM student) AS percentage
FROM student
WHERE sid IN (SELECT sid FROM member)
GROUP BY department;
```

更严谨的分母可用“每系总人数”相关子查询或窗口，面试常追问分母是否按系。

<br>

## 子查询

### 标量

```sql
SELECT name, salary
FROM instructor
WHERE salary > (SELECT AVG(salary) FROM instructor);
```

标量子查询必须返回至多一行一列。

### IN / NOT IN

```sql
SELECT name FROM student
WHERE sid IN (SELECT sid FROM member WHERE cid = 1);

SELECT name FROM student
WHERE sid NOT IN (SELECT sid FROM member);
```

> [!WARNING]+ NOT IN 与 NULL
>
> 子查询结果含 `NULL` 时，`NOT IN` 整体可能恒为未知，导致结果为空。更稳妥：`NOT EXISTS`。

### SOME / ALL / ANY

```sql
-- 高于某一 Comp. Sci. 教师工资
SELECT name FROM instructor
WHERE salary > SOME (SELECT salary FROM instructor WHERE dept_name = 'Comp. Sci.');

-- 高于所有 Comp. Sci. 教师
SELECT name FROM instructor
WHERE salary > ALL (SELECT salary FROM instructor WHERE dept_name = 'Comp. Sci.');
```

`= SOME` 等价 `IN`；`<> ALL` 等价 `NOT IN`（仍需注意 NULL）。

### EXISTS / NOT EXISTS

```sql
SELECT c.name
FROM club c
WHERE EXISTS (
    SELECT 1 FROM member m WHERE m.cid = c.cid
);
```

相关子查询：内层引用外层元组。对每个外层行执行一次（逻辑上；优化器可能改写）。

### FROM 中的子查询

```sql
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

### 模式

自然语言“**所有** … 都 …”在 SQL 中常用双重 `NOT EXISTS`：

> 不存在一个（反例），使得（该反例未被满足）。

### Quiz：JL SUN 监督的所有俱乐部的成员

找学生：对 JL SUN 的每个俱乐部，该生都是成员。

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

直觉：不存在“JL SUN 的俱乐部，且该生未参加”的俱乐部。

### Quiz：只有女生的俱乐部

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

不存在“该俱乐部的非女生成员”。空俱乐部是否算“只有女生”？按此写法空俱乐部会被选出；若需至少一人，再加 `EXISTS` 成员条件。

### 除法：选了某集合全部课程

```sql
-- 选了 computer 系全部课程的学生
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

```sql
WITH cse_instructors AS (
    SELECT ID, name, salary
    FROM instructor
    WHERE dept_name = 'Comp. Sci.'
)
SELECT name FROM cse_instructors WHERE salary > 80000;
```

递归 CTE（树/图，方言支持）：

```sql
WITH RECURSIVE subordinates AS (
    SELECT id, manager_id, name FROM emp WHERE id = 1
    UNION ALL
    SELECT e.id, e.manager_id, e.name
    FROM emp e
    JOIN subordinates s ON e.manager_id = s.id
)
SELECT * FROM subordinates;
```

<br>

## 窗口函数

面试高频。参考各方言文档；概念通用。

```sql
SELECT name, dept_name, salary,
       RANK() OVER (PARTITION BY dept_name ORDER BY salary DESC) AS rk,
       AVG(salary) OVER (PARTITION BY dept_name) AS dept_avg,
       SUM(salary) OVER (ORDER BY salary
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running
FROM instructor;
```

常用：`ROW_NUMBER`、`RANK`、`DENSE_RANK`、`NTILE`、`LAG`/`LEAD`、`SUM/AVG/... OVER`。

与 `GROUP BY` 区别：窗口不合并行，只在结果集上附加分析列。

<br>

## NULL 与三值逻辑

| 表达式 | 结果要点 |
| :--- | :--- |
| `1 = NULL` | UNKNOWN |
| `NULL = NULL` | UNKNOWN |
| `age IN (1, NULL)` | 若 age≠1 则为 UNKNOWN 而非 FALSE |
| `WHERE` 子句 | 仅 TRUE 通过 |
| 聚集 | 除 `COUNT(*)` 外多数忽略 NULL |

```sql
SELECT * FROM student WHERE age = age;  -- age 为 NULL 的行进不来
SELECT * FROM student WHERE age IS NULL;
```

<br>

## 视图

```sql
CREATE VIEW cs_student AS
SELECT sid, name, age
FROM student
WHERE department = 'CS';

SELECT * FROM cs_student WHERE age > 20;
```

可更新视图通常要求：单基表、无聚集/`DISTINCT`、无计算列作为关键路径等（标准与产品限制多）。视图用于安全裁剪与逻辑独立性。

<br>

## 完整性与触发器

### 断言（多数 MySQL 不支持）

```sql
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

### 触发器骨架

```sql
CREATE TRIGGER bump_salary
AFTER UPDATE OF salary ON instructor
REFERENCING NEW ROW AS n OLD ROW AS o
FOR EACH ROW
WHEN (n.salary > o.salary * 1.5)
BEGIN
    -- 方言相关过程体：记日志 / 回滚 / 告警
END;
```

`BEFORE`/`AFTER` × `INSERT`/`UPDATE`/`DELETE`；`OLD`/`NEW` 行面向。

<br>

## 权限

```sql
CREATE ROLE analyst;
GRANT SELECT ON student TO analyst;
GRANT analyst TO alice;
GRANT UPDATE (salary) ON instructor TO bob WITH GRANT OPTION;
REVOKE SELECT ON student FROM alice CASCADE;
```

特权直觉：读、插入、删除、更新；另有参照、触发器等产品扩展。

<br>

## 面试题型清单

### 基础

1. 第二高薪（注意并列）：窗口 `DENSE_RANK` 或两次 `MAX` 子查询。
2. 连续登录 N 天：窗口差值 / 自连接日期。
3. 每个部门工资前三：`RANK() OVER (PARTITION BY ...)`。
4. 有经理的员工与无经理的员工：`LEFT JOIN` + `IS NULL`。

```sql
-- 各部门工资前三（含并列策略用 RANK 或 DENSE_RANK）
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
-- 工资高于本系平均
SELECT i.name, i.dept_name, i.salary
FROM instructor i
WHERE i.salary > (
    SELECT AVG(salary) FROM instructor i2
    WHERE i2.dept_name = i.dept_name
);
```

```sql
-- 从未被选修的课程
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

### 浙大风格核对

| 题意关键词 | 写法 |
| :--- | :--- |
| 所有 / 全部 | `NOT EXISTS` 双重否定或计数 = 全集大小 |
| 只有 / 仅含 | `NOT EXISTS` 反例 |
| 对于每个 | `GROUP BY` 或相关子查询 |
| 最大 / Top-K | 排序 `LIMIT` / 窗口 / `ALL` |
| 室友 / 同组配对 | 自连接或先投影键再连接 |

<br>

## 方言差异速记

| 点 | MySQL | PostgreSQL | SQL Server |
| :--- | :--- | :--- | :--- |
| 限行 | `LIMIT` | `LIMIT`/`FETCH` | `TOP` / `OFFSET FETCH` |
| `FULL JOIN` | 常用左+右模拟 | 支持 | 支持 |
| `INTERSECT` | 8+ | 支持 | 支持 |
| 窗口 | 8+ | 成熟 | 成熟 |
| 递归 CTE | 8+ | 支持 | 支持 |

面试写题前先问清引擎；算法课作业常按标准 SQL / PostgreSQL 口径。

<br>

## 练习入口

1. 把本文“全称量化”三例改写成 `JOIN` + `GROUP BY` + `HAVING` 版本，对照结果。
2. 牛客 SQL 篇按易→难刷：连接 → 聚合 → 子查询 → 窗口。
3. 用主笔记大学模式手写：系均薪、只在一校区开设的课、先修链（若有 `prereq` 表）等。

> [!NOTE]+ 与主笔记分工
>
> 主笔记讲“为什么”与代价/事务；本文讲“怎么写对、怎么写全”。两者交叉处（视图、索引语法、隔离）以主笔记概念为准，以本文语句为准。

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

## 浙大习题精讲

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

## 与主笔记交叉索引

| 主题 | 主笔记 | 本文 |
| :--- | :--- | :--- |
| 除法语义 | 关系代数·除法 | 模板 A / Quiz2(2) |
| 聚集与 HAVING | SQL 概要 | 聚集章 + Quiz2(4) |
| 事务隔离 | 事务章 | 仅语句级 `START TRANSACTION` 等 |
| 授权 | 权限摘要 | GRANT 节 |

<br>
