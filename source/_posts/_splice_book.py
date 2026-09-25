from pathlib import Path

main = Path(r"d:\Git\yukinoshitasherry.github.io\source\_posts\Stanford-CS329H.md")
sec = Path(r"d:\Git\yukinoshitasherry.github.io\source\_posts\_book_section.md")
text = main.read_text(encoding="utf-8")
new = sec.read_text(encoding="utf-8")
start = text.index("\n# 教材\n")
end = text.index("\n# 复习\n")
if not new.endswith("\n"):
    new += "\n"
text = text[: start + 1] + new + text[end + 1 :]
old = (
    "> 教材补注按 2026-09-23 站点侧栏的八块写在文末「教材」一节：引言、Foundations、Learning、Action、Inversion、Aggregation、结语、附录。"
    "该节补 Rasch 消参、识别与 Rashomon、Elo 更新、强弱遗憾、qEUBO、GRPO、反演概率、Borda–DPO、单峰中位数与 Community Notes。"
    "第 1.8 节里「一切独立同分布噪声都满足 IIA」与同节 Theorem 1 冲突，补注采用 Gumbel 等价。"
)
repl = (
    "> 文末「教材」按站点 2026-09-23 的侧栏写书上的公式：引言、第 1–5 章、结语、附录。"
    "第 1.8 节有一句把一切独立同分布噪声都说成满足 IIA；同节 Theorem 1 只等价到 Gumbel，后文按定理。"
)
if old not in text:
    raise SystemExit("callout not found")
text = text.replace(old, repl, 1)
needle = "3. Stanford CS329H. *Autumn 2023* 课表与阅读。[https://web.stanford.edu/class/cs329h/fall2023/](https://web.stanford.edu/class/cs329h/fall2023/)\n"
insert = needle + (
    "4. Truong ST, Haupt A, Koyejo S. *Machine Learning from Human Preferences*. "
    "[https://mlhp.stanford.edu/](https://mlhp.stanford.edu/)，2026-09-23。各章为 `src/chap1.html` 至 `src/chap5.html`。\n"
)
if needle not in text:
    raise SystemExit("bib not found")
bib = text.split("# 参考文献", 1)[1]
if "mlhp.stanford.edu/" not in bib:
    text = text.replace(needle, insert, 1)
main.write_text(text, encoding="utf-8")
print("lines", text.count("\n") + 1)
print("has mirror", "The site sidebar" in text)
print("has rasch", "appetite" in text)
