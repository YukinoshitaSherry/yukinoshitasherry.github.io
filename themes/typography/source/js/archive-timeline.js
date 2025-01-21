$(document).ready(function() {
    if(!$('.archive-timeline').length) return;
    
    const posts = $('.archive.animated.fadeInDown .post-container');
    const timeline = $('.timeline-content');
    let currentYear = '';
    let currentMonth = '';
    let timelineHtml = '';
    
    // 生成时间轴
    posts.each(function() {
        const date = $(this).find('.post-meta .date').text().trim();
        const [year, month] = date.split('-');
        
        if(year !== currentYear) {
            timelineHtml += `<div class="timeline-year">${year}</div>`;
            currentYear = year;
        }
        
        if(`${year}-${month}` !== currentMonth) {
            timelineHtml += `<div class="timeline-month" data-date="${year}-${month}">${month}月</div>`;
            currentMonth = `${year}-${month}`;
        }
    });
    
    timeline.html(timelineHtml);
    
    // 点击月份跳转
    $('.timeline-month').click(function() {
        const year = $(this).data('year');
        const month = $(this).data('month');
        
        // 滚动到对应的文章区域
        const target = $(`.archive-year:contains("${year}")`).next().find(`.archive-month:contains("${month}月")`);
        if(target.length) {
            $('html, body').animate({
                scrollTop: target.offset().top - 100
            }, 500);
        }
    });
    
    // 滚动时高亮当前月份
    $(window).scroll(function() {
        const scrollTop = $(window).scrollTop();
        $('.archive-month').each(function() {
            const monthTop = $(this).offset().top;
            if(monthTop - 150 <= scrollTop) {
                const month = $(this).text().replace('月', '');
                const year = $(this).closest('.year-container').prev('.archive-year').text();
                $('.timeline-month').removeClass('active');
                $(`.timeline-month[data-year="${year}"][data-month="${month}"]`).addClass('active');
            }
        });
    });
}); 