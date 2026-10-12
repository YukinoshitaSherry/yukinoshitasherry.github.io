"""Usage: python -I assemble.py <COURSE>   e.g. CS229
Concatenates tmp/<COURSE>-rewrite/partNN.md in order into source/_posts/Stanford-<COURSE>.md,
then runs whole-file sanity checks."""
import re, sys, pathlib

sys.stdout.reconfigure(encoding="utf-8")
course = sys.argv[1]
root = pathlib.Path(__file__).resolve().parent.parent
d = root / f"{course}-rewrite"
parts = sorted(d.glob("part[0-9][0-9].md"))
text = ""
for p in parts:
    t = p.read_text(encoding="utf-8").strip("\n")
    text += t + "\n\n<br>\n\n" if p != parts[-1] else t + "\n"
text = re.sub(r"(\n<br>\n\n){2,}", "\n<br>\n\n", text)
target = root.parent / "source" / "_posts" / f"Stanford-{course}.md"
target.write_text(text, encoding="utf-8")
L = text.splitlines()
F = "`" * 3
print(course, "parts:", [p.name for p in parts], "lines:", len(L))
print("front matter ok:", L[0] == "---")
print("fences even:", sum(1 for l in L if l.strip().startswith(F)) % 2 == 0)
body = re.sub(F + r".*?" + F, "", text, flags=re.S)
print("$$ even:", body.count("$$") % 2 == 0)
inf = False
h1 = []
for i, l in enumerate(L, 1):
    if l.strip().startswith(F):
        inf = not inf
    elif not inf and l.startswith("# "):
        h1.append(l)
print("H1:", h1)
print("pronouns:", [(i + 1, l[:60]) for i, l in enumerate(L) if re.search(r"我们|你们|你(?!好)", l)][:20])
