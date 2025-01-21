document.addEventListener('DOMContentLoaded', function() {
    // 生成目录
    function generateTOC() {
      const container = document.querySelector('.toc-container');
      if (!container) return;
      
      // 如果是 About 页面,直接隐藏目录
      if (document.querySelector('.about-page')) {
        container.style.display = 'none';
        return;
      }
      
      // 在生成目录前先设置好位置
      container.style.position = 'fixed';
      container.style.left = '20px';
      container.style.top = '80px';
      container.style.opacity = '1';
      container.style.transform = 'none';
      
      const tocList = document.createElement('ul');
      tocList.className = 'toc-list';
      
      // Archive页面特殊处理
      if (document.querySelector('.archive')) {
        const years = document.querySelectorAll('.archive-year');
        if (years.length === 0) {
          container.style.display = 'none';
          return;
        }
        years.forEach(year => {
          const li = document.createElement('li');
          const link = document.createElement('a');
          link.textContent = year.textContent;
          link.href = '#' + year.textContent;
          link.className = 'toc-link';
          li.appendChild(link);
          tocList.appendChild(li);
        });
      } else {
        // 普通文章页面
        const headings = document.querySelectorAll('h1, h2, h3, h4');
        const validHeadings = Array.from(headings).filter(heading => {
          const text = heading.textContent;
          return !text.includes('明月守灯寻长梦') && 
                 !text.includes('梦长寻灯守月明') && 
                 !text.includes('秋月春风') &&
                 !text.includes('目录');
        });
        
        if (validHeadings.length === 0) {
          container.style.display = 'none';
          return;
        }

        let prevLevel = 0;
        let currentUL = tocList;
        const ulStack = [tocList];

        validHeadings.forEach(heading => {
          const level = parseInt(heading.tagName.charAt(1));
          const li = document.createElement('li');
          li.className = `level-${level}`;
          
          const link = document.createElement('a');
          link.textContent = heading.textContent;
          link.href = '#' + heading.id;
          link.className = 'toc-link';

          // 处理层级关系
          if (level > prevLevel) {
            // 创建新的子列表
            const ul = document.createElement('ul');
            ul.style.display = 'block'; // 默认展开
            if (ulStack[ulStack.length - 1].lastElementChild) {
              ulStack[ulStack.length - 1].lastElementChild.appendChild(ul);
              ulStack.push(ul);
              currentUL = ul;
              
              // 为父级添加折叠按钮
              const parentLi = ulStack[ulStack.length - 2].lastElementChild;
              const toggle = document.createElement('span');
              toggle.className = 'toc-toggle';
              toggle.innerHTML = '▼';
              parentLi.insertBefore(toggle, parentLi.firstChild);
            } else {
              currentUL.appendChild(ul);
              ulStack.push(ul);
              currentUL = ul;
            }
          } else if (level < prevLevel) {
            // 返回上级
            for (let i = 0; i < prevLevel - level; i++) {
              ulStack.pop();
            }
            currentUL = ulStack[ulStack.length - 1];
          }

          li.appendChild(link);
          currentUL.appendChild(li);
          prevLevel = level;
        });
      }

      container.innerHTML = '';
      container.appendChild(tocList);
    }
  
    // 折叠功能
    document.addEventListener('click', function(e) {
      if (e.target.classList.contains('toc-toggle')) {
        e.target.classList.toggle('collapsed');
        e.target.innerHTML = e.target.classList.contains('collapsed') ? '▶' : '▼';
        const ul = e.target.parentElement.querySelector('ul');
        if (ul) {
          ul.style.display = ul.style.display === 'none' ? 'block' : 'none';
        }
      }
    });
  
    // 滚动监听
    function updateActiveLink() {
      const scrollPos = window.scrollY;
      document.querySelectorAll('.toc-link').forEach(link => {
        const target = document.querySelector(link.hash);
        if (target) {
          const position = target.offsetTop;
          if (scrollPos >= position - 100) {
            document.querySelectorAll('.toc-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
          }
        }
      });
    }
  
    window.addEventListener('scroll', updateActiveLink);
    generateTOC();
  });