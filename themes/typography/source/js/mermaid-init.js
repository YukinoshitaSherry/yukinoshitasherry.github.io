/**
 * 将 ```mermaid 代码块渲染为 Mermaid SVG（蓝白主题）
 * 兼容：1) pre code.language-mermaid  2) hexo-prism 行号模式 figure.highlight td.code pre（语言常被标为 plaintext）
 */
(function () {
  'use strict';

  // 用于确认脚本是否真的加载到页面（你打开 Console 就能看到）
  try {
    window.__mermaidInitLoaded = true;
    console.log('[mermaid-init] script loaded');
  } catch (e) {
    // ignore
  }

  function stripLeadingLineNumbers(text) {
    // Prism/hexo-prism 可能把行号以纯文本方式塞进 code.textContent 前面，导致无法用 /^flowchart/ 判断
    // 这里尝试剔除形如： "1\n2\n3\n" 的前缀数字行。
    return String(text || '').replace(/^(?:\s*\d+\s*\n)+/m, '').trim();
  }

  /**
   * 避免误判：以 "graph" 开头的普通英文（如 graph theory）不应匹配；
   * Mermaid 合法写法为 graph TD / graph LR 等。
   */
  function isMermaidSource(text) {
    var t = stripLeadingLineNumbers(
      String(text || '')
        .replace(/\r\n/g, '\n')
        .replace(/^\uFEFF/, '')
    );
    if (!t) {
      return false;
    }
    if (/^flowchart\b/im.test(t)) {
      return true;
    }
    if (/^graph\s+(?:TD|TB|BT|LR|RL)\b/im.test(t)) {
      return true;
    }
    if (
      /^(sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|mindmap|timeline|block-beta)\b/im.test(
        t
      )
    ) {
      return true;
    }
    return false;
  }

  function normalizeMermaidSource(text) {
    var t = stripLeadingLineNumbers(
      String(text || '')
        .replace(/\r\n/g, '\n')
        .replace(/^\uFEFF/, '')
    );

    // 再兜底一次：如果 mermaid 关键字不在第一行，就从“首次出现关键字处”截取。
    // 处理场景：行号可能不是通过换行符分隔，而是以空格/其他形式拼在一起。
    var flowIdx = t.search(/\bflowchart\s+[A-Za-z]+\b/i);
    if (flowIdx !== -1) {
      return t.slice(flowIdx).trim();
    }

    var graphIdx = t.search(/\bgraph\s+(?:TD|TB|BT|LR|RL)\b/i);
    if (graphIdx !== -1) {
      return t.slice(graphIdx).trim();
    }

    return t;
  }

  function renderMermaidCharts() {
    if (typeof mermaid === 'undefined') {
      return;
    }

    var candidates = [];

    document.querySelectorAll('pre code.language-mermaid').forEach(function (code) {
      var pre = code.parentNode;
      if (!pre || pre.tagName !== 'PRE') {
        return;
      }
      var fig = pre.closest('figure.highlight');
      candidates.push({
        pre: pre,
        source: normalizeMermaidSource(code.textContent),
        replaceFigure: fig || null
      });
    });

    document.querySelectorAll('figure.highlight td.code pre').forEach(function (pre) {
      var src = normalizeMermaidSource(pre.textContent);
      if (!src || !isMermaidSource(src)) {
        return;
      }
      var seen = candidates.some(function (c) {
        return c.pre === pre;
      });
      if (!seen) {
        candidates.push({ pre: pre, source: src, replaceFigure: pre.closest('figure.highlight') });
      }
    });

    // 兜底：部分主题/高亮器不会给 mermaid 代码块打上 language-mermaid class，
    // 但只要代码文本本身符合 Mermaid 语法，就仍然尝试渲染。
    document.querySelectorAll('pre code').forEach(function (code) {
      var pre = code.parentNode;
      if (!pre || pre.tagName !== 'PRE') {
        return;
      }
      var src = normalizeMermaidSource(code.textContent);
      if (!src || !isMermaidSource(src)) {
        return;
      }
      var seen = candidates.some(function (c) {
        return c.pre === pre;
      });
      if (!seen) {
        candidates.push({ pre: pre, source: src, replaceFigure: pre.closest('figure.highlight') });
      }
    });

    if (!candidates.length) {
      return;
    }

    // 调试用：帮助确认脚本确实在跑、以及到底收集到了多少张图
    try {
      console.log('[mermaid-init] mermaid candidates:', candidates.length);
    } catch (e) {
      // ignore
    }

    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'loose',
      theme: 'base',
      themeVariables: {
        darkMode: false,
        background: '#ffffff',
        primaryColor: '#e3f2fd',
        primaryTextColor: '#0d47a1',
        primaryBorderColor: '#1976d2',
        secondaryColor: '#ffffff',
        secondaryTextColor: '#1565c0',
        secondaryBorderColor: '#90caf9',
        tertiaryColor: '#f5f9ff',
        tertiaryBorderColor: '#bbdefb',
        lineColor: '#42a5f5',
        textColor: '#0d47a1',
        mainBkg: '#e3f2fd',
        nodeBorder: '#1976d2',
        clusterBkg: '#fafdff',
        clusterBorder: '#bbdefb',
        edgeLabelBackground: '#ffffff',
        titleColor: '#0d47a1'
      },
      flowchart: {
        htmlLabels: true,
        curve: 'basis',
        padding: 10,
        nodeSpacing: 28,
        rankSpacing: 36
      },
      fontFamily: 'system-ui, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif'
    });

    // 逐图渲染：避免单个图的语法错误导致整页渲染中断
    var renderPromises = [];

    candidates.forEach(function (item, idx) {
      var wrap = document.createElement('div');
      wrap.className = 'mermaid-chart-wrap';

      // 给每个图一个唯一 id，供 mermaid.render 使用
      var graphId = 'mermaid-chart-' + idx;
      var graph = document.createElement('div');
      graph.className = 'mermaid';
      graph.id = graphId;

      // 先塞回原始文本，便于排查（渲染失败时也能看到源码）
      graph.textContent = item.source;
      wrap.appendChild(graph);

      if (item.replaceFigure) {
        item.replaceFigure.parentNode.replaceChild(wrap, item.replaceFigure);
      } else {
        item.pre.parentNode.replaceChild(wrap, item.pre);
      }

      if (typeof mermaid.render === 'function') {
        var p = mermaid
          .render(graphId, item.source)
          .then(function (result) {
            // mermaid.render 返回 { svg, bindFunctions }
            wrap.innerHTML = result && result.svg ? result.svg : '';
          })
          .catch(function (e) {
            console.error('Mermaid render failed (idx=' + idx + '):', e);
            try {
              console.error('Mermaid source (idx=' + idx + '):', item.source);
            } catch (err) {
              // ignore
            }
          });
        renderPromises.push(p);
      } else {
        // fallback：如果 render 不存在，仍尝试 run，但不保证逐图失败不会中断
        renderPromises.push(
          Promise.resolve().then(function () {
            if (typeof mermaid.run === 'function') {
              mermaid.run({ querySelector: '#' + graphId });
            }
          })
        );
      }
    });

    if (renderPromises.length && typeof Promise !== 'undefined' && Promise.allSettled) {
      Promise.allSettled(renderPromises).catch(function () {
        // 这里已经在单图 catch 里打印过错误
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderMermaidCharts);
  } else {
    renderMermaidCharts();
  }
})();
