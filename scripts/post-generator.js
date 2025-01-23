/**
 * 新的文章生成器，会覆盖旧的
 * 将文章移动到正确的分类目录
 */

const path = require('path');
const fs = require('fs');

hexo.extend.generator.register('post', function(locals) {
  // 获取文章的分类
  locals.posts.forEach(post => {
    if (post.categories && post.categories.length > 0) {
      const category = post.categories.data[0].name;
      // 检查文章是否已经在正确的目录中
      const currentPath = post.full_source;
      const expectedDir = path.join(this.source_dir, '_posts', category);
      
      // 如果目录不存在，创建它
      if (!fs.existsSync(expectedDir)) {
        fs.mkdirSync(expectedDir, { recursive: true });
      }
      
      // 只有当文章不在正确目录时才移动
      if (!currentPath.includes(category)) {
        const newPath = path.join(expectedDir, path.basename(currentPath));
        // 确保不会覆盖现有文件
        if (!fs.existsSync(newPath)) {
          fs.renameSync(currentPath, newPath);
          post.full_source = newPath;
        }
      }
    }
  });
}); 