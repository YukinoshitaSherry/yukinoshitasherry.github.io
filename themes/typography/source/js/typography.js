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
    // 只在PC端检查TOC状态
    const hasToc = window.innerWidth > 768 && 
                   $('.toc-container').children().length > 0 && 
                   $('.toc-container').css('display') !== 'none';
                   
    if (window.innerWidth <= 768) {
        $('#side-bar .site-title').hide(); // 移动端隐藏左侧标题
        $('#main-container').removeClass('col-sm-9').css('width', '100%');
        $('.main-container').css('margin-left', '0');
    } else {
        $('#side-bar .site-title').show(); // PC端显示左侧标题
        var sidebarW = stage.width() - $('#main-container').outerWidth() + 
                      (window.innerWidth - stage.innerWidth()) / 2;
        $('#side-bar').outerWidth(sidebarW);
        $('#main-container').addClass('col-sm-9');
        
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