/**
 * marked 会把公式里的 \\ \{ \, \| 等“反斜杠 + 标点”当作 Markdown 转义吃掉一个反斜杠，
 * 导致 cases/aligned 不换行、集合花括号与间距丢失；_ 与 * 还可能被解析成斜体。
 * 渲染前把 $$...$$、\[...\]、\(...\)、$...$ 换成占位符，渲染后原样放回（HTML 转义），交给浏览器端 MathJax。
 * 旧文为迁就 marked 写的 \\\\、\_、\* 在这里还原为标准 LaTeX 的 \\、_、*。
 * 优先级须早于 callouts.js（默认 10），callout 正文中的公式才会被保护。
 */
const PH = (i) => `MATHPH${i}XEND`;
const PH_RE = /MATHPH(\d+)XEND/g;

const normalizeTex = (tex) =>
  tex
    .replace(/\\\\\\\\/g, "\\\\")
    .replace(/(^|[^\\])\\([_*])/g, "$1$2");

const escapeHtml = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

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

hexo.extend.filter.register("before_post_render", (data) => {
  if (!data || !data.content || !/\.(md|markdown)$/i.test(data.source || "")) return data;
  const store = [];
  data.content = protectMath(data.content, store);
  data._mathStore = store;
  return data;
}, 5);

hexo.extend.filter.register("after_post_render", (data) => {
  const store = data && data._mathStore;
  if (!store || !store.length) return data;
  const restore = (s) => s && s.replace(PH_RE, (m, i) => escapeHtml(store[+i]));
  data.content = restore(data.content);
  if (data.excerpt) data.excerpt = restore(data.excerpt);
  if (data.more) data.more = restore(data.more);
  delete data._mathStore;
  return data;
});
