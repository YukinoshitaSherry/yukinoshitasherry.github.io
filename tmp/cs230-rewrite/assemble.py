import sys, pathlib

root = pathlib.Path(__file__).parent
orig = (root / "orig.md").read_text(encoding="utf-8").splitlines(keepends=True)
n_parts = int(sys.argv[1])
orig_start = int(sys.argv[2])  # 1-based line in orig.md where untouched text resumes; 0 = none

out = []
for i in range(1, n_parts + 1):
    out.append((root / f"part{i:02d}.md").read_text(encoding="utf-8"))
if orig_start > 0:
    out.append("".join(orig[orig_start - 1:]))

target = root.parent.parent / "source" / "_posts" / "Stanford-CS230.md"
target.write_text("".join(out), encoding="utf-8")
print(len("".join(out).splitlines()), "lines written")
