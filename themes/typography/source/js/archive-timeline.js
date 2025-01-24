$(document).ready(function() {
    if(!$('.archive-timeline').length) return;
    
    const posts = $('.archive.animated.fadeInDown .post-container');
    const timeline = $('.timeline-content');
    let currentYear = '';
    let currentMonth = '';
    let timelineHtml = '';
    
    // 从文档中获取所有年份和月份
    $('.year-section').each(function() {
        const year = $(this).find('.archive-year').text().trim();
        
        if(year !== currentYear) {
            timelineHtml += `<div class="timeline-year">${year}</div>`;
            currentYear = year;
        }
        
        $(this).find('.archive-month').each(function() {
            const month = $(this).text().replace('月','');
            timelineHtml += `<div class="timeline-month" data-year="${year}" data-month="${month}">${month}月</div>`;
        });
    });
    
    timeline.html(timelineHtml);
    
    // 点击月份跳转
    $('.timeline-month').click(function() {
        const year = $(this).data('year');
        const month = $(this).data('month');
        
        const target = $(`.archive-year:contains("${year}")`).parent().find(`.archive-month:contains("${month}月")`);
        if(target.length) {
            $('html, body').animate({
                scrollTop: target.offset().top - 100
            }, 500);
        }
    });

    // 点击年份跳转
    $('.timeline-year').click(function() {
        const year = $(this).text();
        const target = $(`.archive-year:contains("${year}")`);
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