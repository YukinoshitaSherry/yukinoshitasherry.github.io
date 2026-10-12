import re, pathlib

root = pathlib.Path(__file__).parent
lines = (root / "missing.txt").read_text(encoding="utf-8").splitlines()
cur = (root.parent.parent / "source" / "_posts" / "Stanford-CS230.md").read_text(encoding="utf-8").splitlines()


def norm(s):
    s = re.sub(r"^[>\s\-\*\d\.]+", "", s)
    return re.sub(r"\s+", "", s)


def grams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)}


windows = []
cn = [norm(l) for l in cur]
for i in range(len(cn)):
    windows.append((i + 1, grams("".join(cn[i:i + 3]))))

out = []
for l in lines:
    num, text = l.split(": ", 1)
    g = grams(norm(text))
    if not g:
        continue
    best, bi = 0.0, 0
    for i, w in windows:
        c = len(g & w) / len(g)
        if c > best:
            best, bi = c, i
    if best < 0.8:
        out.append(f"{num} [{best:.2f} ~L{bi}]: {text}")
(root / "missing2.txt").write_text("\n".join(out), encoding="utf-8")
print(len(out))
