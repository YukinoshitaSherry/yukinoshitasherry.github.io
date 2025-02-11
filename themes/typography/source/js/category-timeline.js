$(document).ready(function() {
    if(!$('.archive-timeline').length) return;
    
    const timeline = $('.timeline-content');
    let timelineHtml = '';
    
    // 从文档中获取所有年份和月份
    $('.year-section').each(function() {
        const year = $(this).find('.archive-year').text().trim();
        timelineHtml += `<div class="timeline-year">${year}</div>`;
        
        // 获取该年份下的所有月份
        $(this).find('.archive-month').each(function() {
            const month = $(this).text().trim();
            timelineHtml += `<div class="timeline-month" data-year="${year}" data-month="${month}">${month}</div>`;
        });
    });
    
    timeline.html(timelineHtml);
    
    // 点击月份跳转
    $('.timeline-month').click(function() {
        const year = $(this).data('year');
        const month = $(this).text();
        const target = $(`.archive-year:contains("${year}")`).parent()
                      .find(`.archive-month:contains("${month}")`);
        if(target.length) {
            $('html, body').animate({
                scrollTop: target.offset().top - 100
            }, 500);
        }
    });
}); 