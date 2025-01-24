//加密文档每次刷新都输密码

(function() {
  // 在页面加载时清除所有与加密相关的缓存
  window.onload = function() {
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('hexo-blog-encrypt:')) {
        localStorage.removeItem(key);
      }
    });
  };
  
  // 在页面刷新或离开前清除缓存
  window.onbeforeunload = function() {
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('hexo-blog-encrypt:')) {
        localStorage.removeItem(key);
      }
    });
  };
})(); 