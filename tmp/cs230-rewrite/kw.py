import pathlib, sys

t = (pathlib.Path(__file__).parent.parent.parent / "source" / "_posts" / "Stanford-CS230.md").read_text(encoding="utf-8")
kws = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").split("\n")
res = []
for k in kws:
    k = k.strip()
    if k:
        res.append(f"{t.count(k)}\t{k}")
(pathlib.Path(__file__).parent / "kw_out.txt").write_text("\n".join(res), encoding="utf-8")
