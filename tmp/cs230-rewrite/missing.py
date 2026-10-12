import re, pathlib, difflib

root = pathlib.Path(__file__).parent.parent.parent / "source" / "_posts"
copy = (root / "Stanford-CS230 copy.md").read_text(encoding="utf-8").splitlines()
cur = (root / "Stanford-CS230.md").read_text(encoding="utf-8")


def norm(s):
    s = re.sub(r"^[>\s\-\*\d\.]+", "", s)
    s = re.sub(r"\s+", "", s)
    return s.strip("。；;，,：:")


cur_norm = norm(cur.replace("\n", ""))
cur_lines = [norm(l) for l in cur.splitlines()]
out = []
for i, l in enumerate(copy, 1):
    n = norm(l)
    if len(n) < 12 or n.startswith("<br") or n in ("$$",):
        continue
    if n in cur_norm:
        continue
    best = difflib.get_close_matches(n, cur_lines, n=1, cutoff=0.75)
    if best:
        continue
    out.append(f"{i}: {l}")
(pathlib.Path(__file__).parent / "missing.txt").write_text("\n".join(out), encoding="utf-8")
print(len(out))
