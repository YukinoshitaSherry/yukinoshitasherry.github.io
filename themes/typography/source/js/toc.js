document.addEventListener('DOMContentLoaded', function() {
    // 生成目录
    function generateTOC() {
      const container = document.querySelector('.toc-container');
      if (!container) return;
      
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
        // 过滤掉固定标题
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

        validHeadings.forEach(heading => {
          const level = parseInt(heading.tagName.charAt(1));
          const li = document.createElement('li');
          li.className = `level-${level}`;
          
          const link = document.createElement('a');
          link.textContent = heading.textContent;
          link.href = '#' + heading.id;
          link.className = 'toc-link';
          
          if (heading.nextElementSibling) {
            const toggle = document.createElement('span');
            toggle.className = 'toc-toggle';
            li.appendChild(toggle);
          }
          
          li.appendChild(link);
          tocList.appendChild(li);
        });
      }
      
      container.innerHTML = ''; 
      container.appendChild(tocList);
    }
  
    // 折叠功能
    document.addEventListener('click', function(e) {
      if (e.target.classList.contains('toc-toggle')) {
        e.target.classList.toggle('collapsed');
        const nextUl = e.target.parentElement.querySelector('ul');
        if (nextUl) {
          nextUl.style.display = nextUl.style.display === 'none' ? 'block' : 'none';
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