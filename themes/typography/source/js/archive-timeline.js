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
        const targetDate = $(this).data('date');
        const targetPost = posts.filter(function() {
            return $(this).find('.post-meta .date').text().includes(targetDate);
        }).first();
        
        if(targetPost.length) {
            $('html, body').animate({
                scrollTop: targetPost.offset().top - 100
            }, 500);
            
            $('.timeline-month').removeClass('active');
            $(this).addClass('active');
        }
    });
    
    // 滚动时高亮当前月份
    $(window).scroll(function() {
        const scrollTop = $(window).scrollTop();
        
        posts.each(function() {
            const postTop = $(this).offset().top - 120;
            const postBottom = postTop + $(this).height();
            
            if(scrollTop >= postTop && scrollTop < postBottom) {
                const date = $(this).find('.post-meta .date').text().trim();
                const [year, month] = date.split('-');
                
                $('.timeline-month').removeClass('active');
                $(`.timeline-month[data-date="${year}-${month}"]`).addClass('active');
                return false;
            }
        });
    });
}); 