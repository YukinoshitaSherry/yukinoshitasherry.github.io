/**
 * 为代码块添加语言标签
 * 自动从 class 中提取语言类型并显示在右上角
 */
(function() {
    'use strict';
    
    // 语言名称映射（将语言代码转换为显示名称）
    const languageMap = {
        'python': 'Python',
        'py': 'Python',
        'cpp': 'C++',
        'c++': 'C++',
        'c': 'C',
        'javascript': 'JavaScript',
        'js': 'JavaScript',
        'typescript': 'TypeScript',
        'ts': 'TypeScript',
        'java': 'Java',
        'go': 'Go',
        'rust': 'Rust',
        'php': 'PHP',
        'ruby': 'Ruby',
        'swift': 'Swift',
        'kotlin': 'Kotlin',
        'scala': 'Scala',
        'html': 'HTML',
        'css': 'CSS',
        'scss': 'SCSS',
        'sass': 'SASS',
        'less': 'LESS',
        'json': 'JSON',
        'xml': 'XML',
        'yaml': 'YAML',
        'yml': 'YAML',
        'toml': 'TOML',
        'markdown': 'Markdown',
        'md': 'Markdown',
        'bash': 'Bash',
        'sh': 'Shell',
        'shell': 'Shell',
        'zsh': 'Zsh',
        'powershell': 'PowerShell',
        'ps1': 'PowerShell',
        'sql': 'SQL',
        'mysql': 'MySQL',
        'postgresql': 'PostgreSQL',
        'dockerfile': 'Dockerfile',
        'docker': 'Docker',
        'makefile': 'Makefile',
        'cmake': 'CMake',
        'nginx': 'Nginx',
        'apache': 'Apache',
        'vim': 'Vim',
        'lua': 'Lua',
        'perl': 'Perl',
        'r': 'R',
        'matlab': 'MATLAB',
        'octave': 'Octave',
        'dart': 'Dart',
        'elixir': 'Elixir',
        'erlang': 'Erlang',
        'haskell': 'Haskell',
        'clojure': 'Clojure',
        'lisp': 'Lisp',
        'scheme': 'Scheme',
        'prolog': 'Prolog',
        'ocaml': 'OCaml',
        'fsharp': 'F#',
        'f#': 'F#',
        'vbnet': 'VB.NET',
        'vb': 'VB.NET',
        'csharp': 'C#',
        'c#': 'C#',
        'objectivec': 'Objective-C',
        'objc': 'Objective-C',
        'assembly': 'Assembly',
        'asm': 'Assembly',
        'tex': 'TeX',
        'latex': 'LaTeX',
        'graphql': 'GraphQL',
        'gql': 'GraphQL',
        'diff': 'Diff',
        'plaintext': 'Text',
        'text': 'Text'
    };
    
    /**
     * 从 class 中提取语言类型
     */
    function extractLanguage(preElement) {
        const classList = preElement.className.split(/\s+/);
        for (let className of classList) {
            if (className.startsWith('language-')) {
                const lang = className.replace('language-', '').toLowerCase();
                return languageMap[lang] || lang.toUpperCase();
            }
        }
        return null;
    }
    
    /**
     * 为所有代码块添加语言标签
     * 支持两种格式：
     * 1. figure.highlight (hexo-prism-plugin 格式)
     * 2. pre[class*="language-"] (标准 Prism 格式)
     */
    function addLanguageLabels() {
        // 处理 figure.highlight 格式（hexo-prism-plugin）
        const figureBlocks = document.querySelectorAll('figure.highlight');
        figureBlocks.forEach(function(figure) {
            const pre = figure.querySelector('pre');
            if (!pre) return;
            
            // 如果已经有标签，跳过
            if (pre.hasAttribute('data-language')) {
                return;
            }
            
            // 从 figure 的 class 中提取语言
            const classList = figure.className.split(/\s+/);
            let language = null;
            for (let className of classList) {
                if (className !== 'highlight') {
                    const lang = className.toLowerCase();
                    language = languageMap[lang] || lang.toUpperCase();
                    break;
                }
            }
            
            if (language) {
                // 同时给 figure 和 pre 设置 data-language
                figure.setAttribute('data-language', language);
                if (pre) {
                    pre.setAttribute('data-language', language);
                }
            }
        });
        
        // 处理标准 Prism 格式
        const codeBlocks = document.querySelectorAll('pre[class*="language-"]:not(figure.highlight pre)');
        codeBlocks.forEach(function(pre) {
            // 如果已经有标签，跳过
            if (pre.hasAttribute('data-language')) {
                return;
            }
            
            const language = extractLanguage(pre);
            if (language) {
                pre.setAttribute('data-language', language);
            }
        });
    }
    
    // 页面加载完成后执行
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addLanguageLabels);
    } else {
        addLanguageLabels();
    }
    
    // 如果使用了 Prism.js，在 Prism 高亮完成后也执行一次
    if (typeof Prism !== 'undefined') {
        const originalHighlight = Prism.highlightAll;
        Prism.highlightAll = function() {
            originalHighlight.apply(Prism, arguments);
            setTimeout(addLanguageLabels, 0);
        };
    }
})();

