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
    
    // 主容器只使用透明度动画
    $('#main-container').removeClass('invisible');
    
    // 侧边栏保持原有动画
    $('#side-bar').removeClass('invisible');
    
    // 大纲容器只使用透明度动画
    $('.toc-container').removeClass('invisible').css({
        'animation': 'fadeIn 0.8s ease-out forwards'
    });
    
    $('.site-title').click(function(e) {
        e.preventDefault();
        window.location.href = $(this).find('a:first').attr('href');
    });
});