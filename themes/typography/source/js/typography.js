/**
 * 侧边栏动画
 */
var stage;
var siteNavShown = true;

function triggerSiteNav() {
    return;
    if (siteNavShown) {
        $('#site-nav').hide(300);
        siteNavShown = false;
    } else {
        $('#site-nav').show(300);
        siteNavShown = true;
    }
}
function updateSidebar() {
    const hasToc = $('.toc-container').children().length > 0 && 
                   $('.toc-container').css('display') !== 'none';
                   
    if (window.innerWidth <= 768 || window.innerHeight <= 600) {
        $('#side-bar').innerWidth($('#stage').width());
        $('#main-container').removeClass('col-sm-9');
    } else {
        var sidebarW = stage.width() - $('#main-container').outerWidth() + 
                      (window.innerWidth - stage.innerWidth()) / 2;
        $('#side-bar').outerWidth(sidebarW);
        $('#main-container').addClass('col-sm-9');
        
        // 只在无大纲时调整左边距
        if (!hasToc) {
            $('.main-container').css('margin-left', '20px');
        }
    }
}
$(document).ready(function () {
    stage = $('#stage');
    $(window).resize(function () {
        updateSidebar();
    });
    updateSidebar();
    
    // 移除主容器的invisible类，并且不添加任何动画
    $('#main-container').removeClass('invisible').css({
        'opacity': '1',
        'transform': 'none',
        'transition': 'none'
    });
    
    // 只给侧边栏添加动画
    $('#side-bar').removeClass('invisible').addClass('fadeInRight');
    
    // 移除目录容器的所有动画
    $('.toc-container').css({
        'opacity': '1',
        'transform': 'none',
        'transition': 'none',
        'animation': 'none',
        '-webkit-animation': 'none'
    });
    
    $('.site-title').click(function(e) {
        e.preventDefault();
        window.location.href = $(this).find('a:first').attr('href');
    });
});