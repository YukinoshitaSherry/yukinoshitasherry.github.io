const path = require('path');
const fs = require('fs');

hexo.extend.console.register('new', 'Create a new post', {
  arguments: [
    { name: 'title', desc: 'Post title' },
    { name: 'category', desc: 'Post category' }
  ]
}, function(args, callback) {
  const title = args._[0];
  let category = args._[1];
  
  // 验证标题
  if (!title) {
    console.error('请指定文章标题');
    return callback();
  }

  // 获取配置文件中定义的所有分类
  const validCategories = this.config.categories || [];
  
  // 如果没有提供分类或分类无效，提示用户选择
  if (!category || !validCategories.includes(category)) {
    console.log('可用的分类：');
    validCategories.forEach((cat, index) => {
      console.log(`${index + 1}. ${cat}`);
    });
    console.error('请使用有效的分类，例如：hexo new "文章标题" "分类名称"');
    return callback();
  }

  // 创建分类目录
  const categoryPath = path.join(this.source_dir, '_posts', category);
  if (!fs.existsSync(categoryPath)) {
    fs.mkdirSync(categoryPath, { recursive: true });
  }

  // 生成文章文件名
  const filename = path.join(category, `${title}.md`);
  
  // 创建文章
  hexo.post.create({
    title: title,
    path: filename,
    layout: 'post',
    date: new Date(),
    categories: [category]
  }).then(callback);
}); 