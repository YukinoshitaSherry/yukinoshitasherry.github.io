"""Check that a rewritten part keeps the hard content of its original line range.

Usage: python -I check_part.py <orig.md> <start_line> <end_line> <part.md>

Reports, in priority order:
  1. code blocks whose (whitespace-normalized) body is missing from the part
  2. display-math blocks ($$...$$) missing from the part
  3. URLs missing
  4. inline-math fragments ($...$) missing
  5. numbers (with >=2 significant chars, e.g. 0.1345, 71.27) missing
  6. prose lines whose normalized text has no close match (informational: rephrasing is allowed,
     but the information must still be present somewhere in the part)
"""
import re, sys, difflib, pathlib

sys.stdout.reconfigure(encoding="utf-8")

orig_path, s, e, part_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
orig_lines = pathlib.Path(orig_path).read_text(encoding="utf-8").splitlines()[s - 1:e]
orig = "\n".join(orig_lines)
part = pathlib.Path(part_path).read_text(encoding="utf-8")
FENCE = "`" * 3


def ws(x):
    x = re.sub(r"^\s*>[ \t]?", "", x, flags=re.M)  # drop callout prefixes
    x = x.replace("\\_", "_").replace("\\*", "*")
    return re.sub(r"\s+", "", x)


part_ws = ws(part)

print("== 1. code blocks missing")
codes = re.findall(FENCE + r"[^\n]*\n(.*?)" + FENCE, re.sub(r"^\s*>[ \t]?", "", orig, flags=re.M), flags=re.S)
for c in codes:
    if ws(c) not in part_ws:
        # report first differing line for speed of fixing
        missing = [l for l in c.splitlines() if l.strip() and ws(l) not in part_ws]
        print("  CODE BLOCK starting:", c.strip().splitlines()[0][:80] if c.strip() else "", "| lines not found:", missing[:5])

body = re.sub(FENCE + r".*?" + FENCE, "", orig, flags=re.S)
print("== 2. display math missing")
for m in re.findall(r"\$\$(.*?)\$\$", body, flags=re.S):
    if ws(m) not in part_ws:
        print("  $$", ws(m)[:160])

print("== 3. urls missing")
for u in sorted(set(re.findall(r"https?://[^\s)\]>]+", orig))):
    if u not in part:
        print("  ", u)

body2 = re.sub(r"\$\$.*?\$\$", "", body, flags=re.S)
print("== 4. inline math missing")
seen = set()
for m in re.findall(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", body2):
    k = ws(m)
    if len(k) < 3 or k in seen:
        continue
    seen.add(k)
    if k not in part_ws:
        print("  $", m[:120])

print("== 5. numbers missing")
nums = set(re.findall(r"(?<![\w.])\d+\.\d+|(?<![\w.])\d{3,}(?![\w])", re.sub(r"https?://\S+", "", body2)))
for n in sorted(nums):
    if n not in part:
        print("  ", n)

print("== 6. prose lines without close match (verify info is kept, rephrasing OK)")


def norm(x):
    x = re.sub(r"^[>\s\-\*\d\.]+", "", x)
    x = re.sub(r"\s+", "", x)
    return x.strip("。；;，,：:")


part_lines = [norm(l) for l in part.splitlines()]
part_norm = norm(part.replace("\n", ""))
cnt = 0
in_f = False
for i, l in enumerate(orig_lines, s):
    if l.strip().startswith(FENCE):
        in_f = not in_f
        continue
    if in_f:
        continue
    n = norm(l)
    if len(n) < 12 or n.startswith("<br") or "$$" in n:
        continue
    if n in part_norm or difflib.get_close_matches(n, part_lines, n=1, cutoff=0.6):
        continue
    cnt += 1
    print(f"  {i}: {l[:150]}")
print("prose lines flagged:", cnt)
