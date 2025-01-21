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
    if (window.innerWidth <= 768 || window.innerHeight <= 600) {
        $('#side-bar').innerWidth($('#stage').width());
        $('#main-container').removeClass('col-sm-9');
        //$('#site-nav').hide();
        //siteNavShown = false;
    } else {
        //$('#site-nav').show();
        //siteNavShown = true;
        var sidebarW =
            stage.width() - $('#main-container').outerWidth() + (window.innerWidth - stage.innerWidth()) / 2;
        $('#side-bar').outerWidth(sidebarW);
        console.log("sidebarW=" + sidebarW);
        $('#main-container').addClass('col-sm-9');
    }
}
$(document).ready(function () {
    stage = $('#stage');
    $(window).resize(function () {
        updateSidebar();
    });
    updateSidebar();
    
    // 移除主容器和侧边栏的invisible类
    $('#main-container').removeClass('invisible');
    $('#side-bar').removeClass('invisible').addClass('fadeInRight'); // 先移除invisible再添加动画
    
    // 为大纲容器设置样式，确保没有动画
    $('.toc-container').css({
        'opacity': '1',
        'transform': 'none',
        'animation': 'none',
        '-webkit-animation': 'none'
    });
    
    $('.site-title').click(function(e) {
        e.preventDefault();
        window.location.href = $(this).find('a:first').attr('href');
    });
});