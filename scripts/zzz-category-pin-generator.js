'use strict';

/**
 * 覆盖 hexo-generator-category-enhance 的 category 生成器：
 * 同一分类内，front-matter 中 category_pin: true 的文章排在最前；
 * 置顶区与非置顶区各自按 date 降序；再交给 hexo-pagination 分页。
 * 不影响 archive、index、tag 等其它生成器。
 */
var pagination = require('hexo-pagination');

function sortCategoryPosts(posts) {
  var arr = posts.sort('-date').toArray();
  var pinned = [];
  var unpinned = [];
  for (var i = 0; i < arr.length; i++) {
    var p = arr[i];
    if (p.category_pin) {
      pinned.push(p);
    } else {
      unpinned.push(p);
    }
  }
  pinned.sort(function (a, b) {
    return b.date.valueOf() - a.date.valueOf();
  });
  unpinned.sort(function (a, b) {
    return b.date.valueOf() - a.date.valueOf();
  });
  return pinned.concat(unpinned);
}

hexo.extend.generator.register('category', function (locals) {
  var config = this.config;
  var perPage = config.category_generator.per_page;
  var paginationDir = config.pagination_dir || 'page';
  var categories = locals.categories;
  var categoryDir;

  var pages = categories.reduce(function (result, category) {
    if (!category.length) return result;

    var postsOrdered = sortCategoryPosts(category.posts);
    var data = pagination(category.path, postsOrdered, {
      perPage: perPage,
      layout: ['category', 'archive', 'index'],
      format: paginationDir + '/%d/',
      data: {
        category: category.name
      }
    });

    return result.concat(data);
  }, []);

  if (config.category_generator.enable_index_page) {
    categoryDir = config.category_dir;
    if (categoryDir[categoryDir.length - 1] !== '/') {
      categoryDir += '/';
    }

    pages.push({
      path: categoryDir,
      layout: ['category-index'],
      posts: locals.posts,
      data: {
        base: categoryDir,
        total: 1,
        current: 1,
        current_url: categoryDir,
        posts: locals.posts,
        prev: 0,
        prev_link: '',
        next: 0,
        next_link: '',
        categories: categories
      }
    });
  }

  return pages;
});
