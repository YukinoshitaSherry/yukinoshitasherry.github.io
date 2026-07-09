/**
 * 站点复制保护：
 * - 豁免分类文章：全文可复制
 * - 加密文章：全文不可复制（密码输入框除外）
 * - 其他页面：仅代码块可复制；主页导览仅分类链接可复制
 * 配置见站点 _config.yml → copyright.allow_copy_categories
 */
(function () {
  if (document.body.dataset.copyrightProtect !== 'true') return;

  var CODE_SELECTOR =
    'pre, code, figure.highlight, .highlight, .prism-show-language';
  var INTERACTIVE_SELECTOR =
    'input, textarea, select, [contenteditable="true"], #giscus-container, .gsc-comment-box, .hbe-input-field';
  var COPYABLE_CATEGORY_SELECTOR = '.custom-content a';

  function getElement(node) {
    return node && node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
  }

  function matchesSelector(el, selector) {
    if (!el || el.nodeType !== Node.ELEMENT_NODE || !el.matches) return false;
    try {
      return el.matches(selector);
    } catch (e) {
      return false;
    }
  }

  function isInside(el, selector) {
    var current = getElement(el);
    while (current && current !== document.body) {
      if (matchesSelector(current, selector)) return true;
      current = current.parentElement;
    }
    return false;
  }

  function isPostCopyAllowed() {
    return document.body.dataset.postCopyAllowed === 'true';
  }

  function isEncryptedPost() {
    return document.body.dataset.postEncrypted === 'true';
  }

  function isRangeFullyInsideAllowed(range, allowedSelector) {
    function nodeAllowed(node) {
      return isInside(node, allowedSelector);
    }

    var startOk = nodeAllowed(range.startContainer);
    var endOk = nodeAllowed(range.endContainer);
    if (!startOk || !endOk) return false;

    var walker = document.createTreeWalker(range.commonAncestorContainer, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        return range.intersectsNode(node) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });

    var textNode;
    while ((textNode = walker.nextNode())) {
      if (!nodeAllowed(textNode)) return false;
    }
    return true;
  }

  function isRangeCopyAllowed(range) {
    if (isEncryptedPost()) {
      return isRangeFullyInsideAllowed(range, INTERACTIVE_SELECTOR);
    }

    var allowedSelector =
      CODE_SELECTOR +
      ', ' +
      INTERACTIVE_SELECTOR +
      ', ' +
      COPYABLE_CATEGORY_SELECTOR;

    return isRangeFullyInsideAllowed(range, allowedSelector);
  }

  function shouldAllowCopyEvent() {
    if (isPostCopyAllowed()) return true;

    var selection = window.getSelection();
    if (!selection || selection.isCollapsed) return true;

    var range = selection.rangeCount ? selection.getRangeAt(0) : null;
    if (!range) return true;

    return isRangeCopyAllowed(range);
  }

  function notifyBlocked() {
    var tip = document.getElementById('copyright-copy-tip');
    if (!tip) {
      tip = document.createElement('div');
      tip.id = 'copyright-copy-tip';
      tip.setAttribute('role', 'status');
      tip.setAttribute('aria-live', 'polite');
      document.body.appendChild(tip);
    }
    tip.textContent = isEncryptedPost()
      ? '本站加密内容受版权保护，禁止复制'
      : '本站内容受版权保护，除代码块与分类链接外禁止复制';
    tip.classList.add('is-visible');
    clearTimeout(notifyBlocked._timer);
    notifyBlocked._timer = setTimeout(function () {
      tip.classList.remove('is-visible');
    }, 2200);
  }

  document.addEventListener(
    'copy',
    function (event) {
      if (shouldAllowCopyEvent(event)) return;
      event.preventDefault();
      notifyBlocked();
    },
    true
  );

  document.addEventListener(
    'cut',
    function (event) {
      if (shouldAllowCopyEvent(event)) return;
      event.preventDefault();
      notifyBlocked();
    },
    true
  );

  document.addEventListener(
    'contextmenu',
    function (event) {
      if (isPostCopyAllowed()) return;
      if (isEncryptedPost()) {
        if (isInside(event.target, INTERACTIVE_SELECTOR)) return;
        event.preventDefault();
        return;
      }
      if (
        isInside(event.target, CODE_SELECTOR) ||
        isInside(event.target, INTERACTIVE_SELECTOR) ||
        isInside(event.target, COPYABLE_CATEGORY_SELECTOR)
      ) {
        return;
      }
      event.preventDefault();
    },
    true
  );
})();
