/**
 * 把 Obsidian 风格的 callout 语法（> [!TYPE]+/- Title）转换成 HTML 结构。
 * - 在 markdown 渲染前运行，避免影响普通引用。
 * - 支持可折叠（+ 展开，- 折叠）和非折叠两种形式。
 */

const ICON_MAP = {
  note: '<i class="fa fa-info-circle" aria-hidden="true"></i>',
  info: '<i class="fa fa-info-circle" aria-hidden="true"></i>',
  hint: '<i class="fa fa-lightbulb-o" aria-hidden="true"></i>',
  important: '<i class="fa fa-info-circle" aria-hidden="true"></i>',
  question: '<i class="fa fa-question-circle" aria-hidden="true"></i>',
  help: '<i class="fa fa-question-circle" aria-hidden="true"></i>',
  tip: '<i class="fa fa-check-circle" aria-hidden="true"></i>',
  success: '<i class="fa fa-check-circle" aria-hidden="true"></i>',
  warning: '<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>',
  caution: '<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>',
  attention: '<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>',
  abstract: '<i class="fa fa-list-alt" aria-hidden="true"></i>',
  summary: '<i class="fa fa-list-alt" aria-hidden="true"></i>',
  tldr: '<i class="fa fa-list-alt" aria-hidden="true"></i>',
  example: '<i class="fa fa-list-alt" aria-hidden="true"></i>',
  quote: '<i class="fa fa-quote-left" aria-hidden="true"></i>',
  cite: '<i class="fa fa-quote-left" aria-hidden="true"></i>',
  failure: '<i class="fa fa-times-circle" aria-hidden="true"></i>',
  fail: '<i class="fa fa-times-circle" aria-hidden="true"></i>',
  missing: '<i class="fa fa-times-circle" aria-hidden="true"></i>',
  danger: '<i class="fa fa-times-circle" aria-hidden="true"></i>',
  error: '<i class="fa fa-times-circle" aria-hidden="true"></i>',
  bug: '<i class="fa fa-bug" aria-hidden="true"></i>',
};
const stripBlockquotePrefix = (line) => line.replace(/^\s*>\s?/, "");

const escapeHtml = (text) =>
  (text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

function renderBodyMarkdown(body) {
  const content = (body || "").trim();
  if (!content) return "";
  return hexo.render.renderSync({ text: content, engine: "markdown" });
}

function buildCallout(typeRaw, title, bodyMd, collapseMark) {
  const type = (typeRaw || "note").toLowerCase();
  const icon = ICON_MAP[type] || "💬";
  const header = escapeHtml(title || type);
  const body = renderBodyMarkdown(bodyMd);
  const collapsible = collapseMark === "+" || collapseMark === "-";
  const openAttr = collapseMark === "-" ? "" : " open";

  if (collapsible) {
    return `<details class="callout callout-${type} is-collapsible" data-callout="${type}"${openAttr}>
<summary><span class="callout-caret">▸</span><span class="callout-icon">${icon}</span><span class="callout-title-text">${header}</span></summary>
<div class="callout-content">${body}</div>
</details>`;
  }

  return `<div class="callout callout-${type}" data-callout="${type}">
<div class="callout-title"><span class="callout-icon">${icon}</span><span class="callout-title-text">${header}</span></div>
<div class="callout-content">${body}</div>
</div>`;
}

function transformCallouts(markdown) {
  const lines = (markdown || "").split("\n");
  const output = [];

  let inFence = false;
  let fenceMarker = "";

  for (let i = 0; i < lines.length; ) {
    const line = lines[i];

    // 处理代码块，避免误解析示例
    const fenceMatch = line.match(/^(\s*)(`{3,}|~{3,})(.*)$/);
    if (fenceMatch) {
      const marker = fenceMatch[2];
      if (!inFence) {
        inFence = true;
        fenceMarker = marker;
      } else if (marker === fenceMarker) {
        inFence = false;
        fenceMarker = "";
      }
      output.push(line);
      i += 1;
      continue;
    }

    if (inFence) {
      output.push(line);
      i += 1;
      continue;
    }

    const meta = line.match(/^\s*>\s*\[!([A-Za-z0-9_-]+)\]([+-])?\s*(.*)$/);
    if (!meta) {
      output.push(line);
      i += 1;
      continue;
    }

    // 收集连续的引用行
    const blockLines = [];
    let j = i;
    while (j < lines.length && /^\s*>/.test(lines[j])) {
      blockLines.push(lines[j]);
      j += 1;
    }

    const [, type, collapseMark, customTitle] = meta;
    const bodyLines = blockLines.slice(1).map(stripBlockquotePrefix);
    const bodyMd = bodyLines.join("\n");
    const html = buildCallout(type, customTitle && customTitle.trim(), bodyMd, collapseMark);

    output.push(html);
    i = j;
  }

  return output.join("\n");
}

hexo.extend.filter.register("before_post_render", (data) => {
  if (!data || !data.content) return data;
  data.content = transformCallouts(data.content);
  return data;
});

