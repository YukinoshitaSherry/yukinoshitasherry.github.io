/**
 * 公式占位符逻辑（无 Hexo 依赖），供 protect-math.js 与 callouts.js 共用。
 */
const PH = (i) => `MATHPH${i}XEND`;
const PH_RE = /MATHPH(\d+)XEND/g;

const normalizeTex = (tex) =>
  tex
    .replace(/\\\\\\\\/g, "\\\\")
    .replace(/(^|[^\\])\\([_*])/g, "$1$2");

const restoreMathFragment = (s) => s;

function protectSegment(text, store) {
  const codes = [];
  text = text.replace(/(`+)[^`\n]*?\1/g, (m) => {
    codes.push(m);
    return `CODEPH${codes.length - 1}XEND`;
  });

  const display = (open, close) => (m, body, offset, whole) => {
    const lineStart = whole.lastIndexOf("\n", offset) + 1;
    const inQuote = /^\s*>/.test(whole.slice(lineStart, offset));
    if (inQuote) body = body.replace(/\n\s*>[ \t]?/g, "\n");
    store.push(`${open}${normalizeTex(body)}${close}`);
    return PH(store.length - 1);
  };
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, display("$$", "$$"));
  text = text.replace(/(?<!\\)\\\[([\s\S]+?)(?<!\\)\\\]/g, display("\\[", "\\]"));

  text = text.replace(/(?<!\\)\\\(([^\n]+?)(?<!\\)\\\)/g, (m, body) => {
    store.push(`\\(${normalizeTex(body)}\\)`);
    return PH(store.length - 1);
  });

  text = text.replace(/(^|[^\\$])\$(?=\S)((?:\\.|[^$\\\n])+?)(?<=\S)\$(?!\d)/g, (m, pre, body) => {
    store.push(`$${normalizeTex(body)}$`);
    return pre + PH(store.length - 1);
  });

  return text.replace(/CODEPH(\d+)XEND/g, (m, i) => codes[+i]);
}

function protectMath(content, store) {
  const lines = content.split("\n");
  const out = [];
  let buf = [];
  let fence = null;
  const flush = () => {
    if (buf.length) out.push(protectSegment(buf.join("\n"), store));
    buf = [];
  };
  for (const line of lines) {
    const m = line.match(/^(?:\s*>)*\s*(`{3,}|~{3,})/);
    if (fence) {
      out.push(line);
      if (m && m[1][0] === fence[0] && m[1].length >= fence.length) fence = null;
    } else if (m) {
      flush();
      fence = m[1];
      out.push(line);
    } else {
      buf.push(line);
    }
  }
  flush();
  return out.join("\n");
}

function restoreMathInHtml(s, store) {
  if (!s || !store.length) return s;
  return s.replace(PH_RE, (m, i) => restoreMathFragment(store[+i]));
}

module.exports = {
  PH,
  PH_RE,
  protectSegment,
  protectMath,
  restoreMathInHtml,
};
