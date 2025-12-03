/**
 * Hexo 脚本：确保每次 generate 后自动应用自定义 Prism 主题
 * 此脚本会在 hexo generate 完成后自动运行
 */

const fs = require('fs');
const path = require('path');

hexo.on('generateAfter', function() {
    // 1. 复制自定义CSS文件
    const customCssPath = path.join(hexo.base_dir, 'themes/typography/source/css/prism-custom.css');
    const targetCssPath = path.join(hexo.public_dir, 'css/prism.css');
    
    if (fs.existsSync(customCssPath)) {
        const customCss = fs.readFileSync(customCssPath, 'utf8');
        const targetDir = path.dirname(targetCssPath);
        if (!fs.existsSync(targetDir)) {
            fs.mkdirSync(targetDir, { recursive: true });
        }
        fs.writeFileSync(targetCssPath, customCss, 'utf8');
    }
    
    // 2. 复制语言标签JavaScript文件
    const customJsPath = path.join(hexo.base_dir, 'themes/typography/source/js/prism-language-label.js');
    const targetJsPath = path.join(hexo.public_dir, 'js/prism-language-label.js');
    
    if (fs.existsSync(customJsPath)) {
        const customJs = fs.readFileSync(customJsPath, 'utf8');
        const targetJsDir = path.dirname(targetJsPath);
        if (!fs.existsSync(targetJsDir)) {
            fs.mkdirSync(targetJsDir, { recursive: true });
        }
        fs.writeFileSync(targetJsPath, customJs, 'utf8');
    }
});
