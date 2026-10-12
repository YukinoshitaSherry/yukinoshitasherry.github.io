/**
 * marked 会把公式里的 \\ \{ \, \| 等“反斜杠 + 标点”当作 Markdown 转义吃掉一个反斜杠，
 * 导致 cases/aligned 不换行、集合花括号与间距丢失；_ 与 * 还可能被解析成斜体。
 * 渲染前把 $$...$$、\[...\]、\(...\)、$...$ 换成占位符，渲染后原样放回（不做 HTML 转义），交给浏览器端 MathJax。
 * 旧文为迁就 marked 写的 \\\\、\_、\* 在这里还原为标准 LaTeX 的 \\、_、*。
 * 优先级 5，须早于 callouts.js（11），callout 正文中的公式才会被保护。
 */
const { protectMath, restoreMathInHtml } = require("./protect-math-lib");

hexo.extend.filter.register("before_post_render", (data) => {
  if (!data || !data.content || !/\.(md|markdown)$/i.test(data.source || "")) return data;
  const store = data._mathStore || [];
  data._mathStore = store;
  data.content = protectMath(data.content, store);
  return data;
}, 5);

hexo.extend.filter.register("after_post_render", (data) => {
  const store = data && data._mathStore;
  if (!store || !store.length) return data;
  data.content = restoreMathInHtml(data.content, store);
  if (data.excerpt) data.excerpt = restoreMathInHtml(data.excerpt, store);
  if (data.more) data.more = restoreMathInHtml(data.more, store);
  delete data._mathStore;
  return data;
});
