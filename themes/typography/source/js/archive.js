document.addEventListener('DOMContentLoaded', function() {
    // 为年份标题添加点击事件
    document.querySelectorAll('.archive-year').forEach(year => {
        year.addEventListener('click', function() {
            this.classList.toggle('collapsed');
            let content = this.parentElement.querySelector('.archive-year-content');
            if (content) {
                content.style.display = this.classList.contains('collapsed') ? 'none' : 'block';
            }
            
            // 更新目录状态
            updateToc(this.id, !this.classList.contains('collapsed'));
        });
    });

    // 为月份标题添加点击事件
    document.querySelectorAll('.archive-month').forEach(month => {
        month.addEventListener('click', function() {
            this.classList.toggle('collapsed');
            let content = this.parentElement.querySelector('.archive-month-content');
            if (content) {
                content.style.display = this.classList.contains('collapsed') ? 'none' : 'block';
            }
            
            // 更新目录状态
            updateToc(this.id, !this.classList.contains('collapsed'));
        });
    });

    // 生成目录
    generateToc();
});

function generateToc() {
    const toc = document.querySelector('.toc-content');
    if (!toc) return;

    const years = document.querySelectorAll('.archive-year');
    let tocHtml = '<ul class="toc-list">';

    years.forEach(year => {
        const yearId = year.id;
        const yearText = year.textContent;
        
        tocHtml += `
            <li>
                <a href="#${yearId}" class="toc-year">${yearText}</a>
                <ul class="toc-months">
        `;

        const months = year.parentElement.querySelectorAll('.archive-month');
        months.forEach(month => {
            const monthId = month.id;
            const monthText = month.textContent;
            tocHtml += `
                <li>
                    <a href="#${monthId}" class="toc-month">${monthText}</a>
                </li>
            `;
        });

        tocHtml += '</ul></li>';
    });

    tocHtml += '</ul>';
    toc.innerHTML = tocHtml;
}

function updateToc(id, isExpanded) {
    const tocItem = document.querySelector(`.toc-list a[href="#${id}"]`);
    if (tocItem) {
        tocItem.classList.toggle('collapsed', !isExpanded);
    }
} 