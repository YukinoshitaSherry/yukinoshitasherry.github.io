/**
 * hexo-renderer-marked 7 每次渲染都对全局 marked 调用 marked.use({ renderer, tokenizer })，
 * tokenizer 包装层逐次叠加；url tokenizer 在 autolink 下返回 false 会穿过全部包装层，
 * 渲染次数过万（callouts.js 每个 callout 单独渲染一次）后栈溢出。
 * 每次渲染前把 marked 的全局配置恢复为默认值，包装层始终只有一层。
 */
const { marked } = require(require.resolve("marked", {
  paths: [require.resolve("hexo-renderer-marked")],
}));

hexo.extend.filter.register("marked:use", () => {
  marked.setOptions(marked.getDefaults());
}, 1);
