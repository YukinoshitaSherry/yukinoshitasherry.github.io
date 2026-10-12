import re, pathlib

root = pathlib.Path(__file__).parent
t = (root.parent.parent / "source" / "_posts" / "Stanford-CS230.md").read_text(encoding="utf-8")
L = t.splitlines()
fence = "`" * 3
print("dash lines", [i + 1 for i, l in enumerate(L) if l.strip() == "---"])
print("fences", sum(1 for l in L if l.strip().startswith(fence)))
body = re.sub(fence + r".*?" + fence, "", t, flags=re.S)
print("display math delimiters", body.count("$$"))
pron = [(i + 1, l[:70]) for i, l in enumerate(L) if re.search(r"我们|你们|你|(?<!自)我", l)]
print("pronouns", pron[:30])
print("H1", [l for l in L if l.startswith("# ")])
inf = False
for i, l in enumerate(L):
    if l.strip().startswith(fence):
        inf = not inf
    elif inf and l.startswith("> [!"):
        print("callout in fence", i + 1)
