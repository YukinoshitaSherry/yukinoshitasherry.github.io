---
title: Single Cell常见可视化图表
date: 2025-05-28
categories:
  - 学AI/DS
tags:
  - AI4S
  - CompBio
  - SingleCell
  - 可视化
  - 科研技巧
desc: 理解与绘制：肘部图、PCA、UMAP、tSNE、热图、小提琴图、等高线图、密度图、火山图、山脊图、和弦图、GSEA富集图
---

传统生信分析较多使用R语言，本文使用Python。


**参考:**
- <a href="https://www.xiaohongshu.com/explore/66faa731000000002c02cd4a?xsec_token=ABj225oDnTR3oU8JE2y1HSqPeJ2m3lYkGnwa2iz0C-tQQ=&xsec_source=pc_collect">小红书：12种常见的文献数据图解读</a>
- <a href="https://zhuanlan.zhihu.com/p/488362896">知乎：单细胞marker基因可视化的补充</a>
- <a href="https://zhuanlan.zhihu.com/p/376439417">知乎：单细胞分析实录 展示marker基因的4种图形</a>

# 背景知识



# 常见图表

## ElbowPlot


## UMAP

### 作用 
UMAP 是一种基于流形学习的降维可视化方法，能将高维的单细胞转录组数据映射到二维或三维空间，便于观察细胞聚类和细胞类型间的连续性，同时保留数据的局部结构。

### 参数
- `n_neighbors`，用于指定每个点的邻居个数，较大的值会使embedding更关注全局结构，较小的值则使embedding更关注局部结构
- `min_dist`，用于控制嵌入空间中点之间的最小距离，较小的值会使点更紧密地聚集，较大的值会使点更分散
- `metric`，用于指定计算距离的度量方式，如'euclidean'、'cosine'等
- `n_components`，指定降维后的目标维度，通常为2或3。

### 读图
每个点代表一个细胞，不同颜色表示不同的细胞类型或聚类。距离相近的点代表细胞在特征上相似，可能属于同一细胞类型或具有相似的细胞状态。通过观察UMAP图，可以直观地了解细胞群体的结构和分布，发现潜在的细胞亚群和细胞类型间的过渡状态。

### 示例代码
```python
# 生成示例数据
np.random.seed(42)
data = np.random.rand(1000, 50)

# 应用UMAP降维
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', n_components=2)
embedding = reducer.fit_transform(data)

# 绘制UMAP图
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='viridis')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP of Single Cell Data')
plt.show()
```

### 其他适合领域

UMAP 除了在单细胞数据可视化中表现出色外，还广泛应用于其他高维数据的可视化领域，如图像识别、自然语言处理等。

## t-SNE

### 作用

t-SNE 是一种常用的非线性降维可视化技术，能够揭示数据的局部结构，并在降维过程中尽可能地保留数据的相似性，常用于单细胞数据中细胞类型或聚类的可视化。

### 参数

主要参数有`n_components`，目标维度，通常为 2；`perplexity`，困惑度参数，较大的困惑度会使邻居的数量更大，更适合发现全局结构，较小的困惑度则更关注局部结构；`early_exaggeration`，早期夸大因子，影响初始阶段数据的聚集程度；`learning_rate`，学习率，影响优化过程的步长，过小会导致收敛慢，过大可能导致不收敛；`n_iter`，最大迭代次数。

### 读图

图中每个点代表一个细胞，颜色可表示不同的细胞类型、聚类或其他特征。t-SNE 会将相似的细胞聚集在一起，不同的簇可能代表不同的细胞类型或状态。需要注意的是，t-SNE 的结果可能因参数设置和随机性而存在一定的变化。

### 示例代码

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
np.random.seed(42)
data = np.random.rand(1000, 50)

# 应用 t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=12, learning_rate=200, n_iter=1000)
embedding = tsne.fit_transform(data)

# 绘制 t-SNE 图
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='viridis')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE of Single Cell Data')
plt.show()
```

### 其他适合领域

t-SNE 在图像识别、文本分类等领域也有广泛应用，用于数据的可视化和探索性分析。

## 热图

### 作用

热图可以展示基因在不同细胞或细胞聚类中的表达水平，有助于识别差异表达基因和细胞类型的特异性基因。

### 参数

- `data`，输入数据，通常是基因表达矩阵
- `row_cluster`，是否对行进行聚类
- `col_cluster`，是否对列进行聚类
- `cmap`，颜色映射，如`'viridis'`、`'hot'`等
- `figsize`，图的大小
- `xticklabels`，是否显示 x 轴刻度标签
- `yticklabels`，是否显示 y 轴刻度标签。

### 读图

热图的行通常代表基因，列代表细胞或细胞聚类。颜色的深浅表示基因表达水平的高低，通常使用颜色条来指示表达水平的数值范围。通过观察热图，可以直观地看到哪些基因在特定的细胞或聚类中高表达或低表达。

### 示例代码

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
np.random.seed(42)
genes = [f'Gene_{i}' for i in range(50)]
cells = [f'Cell_{i}' for i in range(100)]
data = np.random.rand(50, 100)

# 绘制热图
plt.figure(figsize=(12, 8))
sns.heatmap(data, xticklabels=False, yticklabels=True, cmap='viridis')
plt.xlabel('Cells')
plt.ylabel('Genes')
plt.title('Heatmap of Gene Expression')
plt.show()
```

### 其他适合领域

热图在基因表达分析、蛋白质组学等领域也有广泛应用，用于展示数据的表达模式和聚类关系。

## 小提琴图

### 作用

小提琴图结合了箱线图和核密度估计，可以展示基因表达的分布特征，包括中位数、四分位数范围、分布形状等。

### 参数

- `x`，分组变量，通常是细胞类型或聚类
- `y`，数值变量，如基因表达水平
- `data`，输入数据框
- `inner`，指定内部箱线图的类型，如`'box'`、`'quartile'`等
- `bw`，带宽参数，影响核密度估计的平滑程度
- `cut`，控制密度曲线在数据范围外的延伸程度
- `scale`，控制小提琴图的宽度比例。

### 读图

小提琴图的宽度表示基因表达的密度，较宽的部分表示在该表达水平的细胞数量较多。内部的箱线图显示了数据的中位数、四分位数范围和异常值等统计信息。通过比较不同组间的小提琴图，可以了解基因在不同细胞类型或条件下的表达分布差异。

### 示例代码

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
np.random.seed(42)
data = pd.DataFrame({
    'Gene_1': np.random.normal(0, 1, 100),
    'Gene_2': np.random.normal(2, 1, 100),
    'Cell_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], 100)
})

# 绘制小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(x='Cell_Type', y='Gene_1', data=data, inner='quartile', palette='Set2')
plt.xlabel('Cell Type')
plt.ylabel('Gene Expression')
plt.title('Violin Plot of Gene Expression')
plt.show()
```

### 其他适合领域

小提琴图在基因表达分析、生物统计等领域有广泛应用，用于展示数据的分布特征和统计信息。

## 等高线图

### 作用

等高线图可以展示基因表达密度或某些特征在二维空间中的分布情况，有助于识别高密度区域和基因表达的空间模式。

### 参数

- `x` 和`y`，通常是降维后的坐标，如 UMAP 或 t-SNE 的第一、二个主成分；
- `z`，表示基因表达水平或其他特征值
- `levels`，指定等高线的数量或具体数值
- `cmap`，颜色映射
- `extend`，控制等高线图的扩展方式，如`'neither'`、`'both'`等
- `alpha`，透明度。

### 读图

等高线连接的是具有相同特征值的点，线条越密集表示该区域的变化越快。颜色的深浅结合等高线可以更直观地反映特征值的高低和分布趋势。

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 绘制等高线图
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
plt.colorbar(contour)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot')
plt.show()
```

### 其他适合领域

等高线图在地理信息系统、气象学等领域有广泛应用，用于展示地理数据、气象数据等的分布情况。

## 密度图

### 作用

密度图通过核密度估计展示基因表达或细胞分布的密度，能够突出高密度区域，帮助识别细胞聚集的位置和模式。

### 参数

- `values`，输入数据，如基因表达值
- `bw_method`，带宽计算方法，可指定为`'scott'`、`'silverman'`或一个标量
- `ind`，指定用于计算密度的网格点
- `kernel`，核函数类型，如`'gaussian'`、`'tophat'`等
- `weights`，可为数据点指定权重。

### 读图

密度图中曲线的峰值位置表示数据的高密度区域，曲线的形状反映了数据的分布特征。颜色或阴影的深浅可表示密度的高低。

### 示例代码

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# 绘制密度图
plt.figure(figsize=(8, 6))
sns.kdeplot(data, shade=True, color='blue')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Density Plot')
plt.show()
```

### 其他适合领域
密度图在统计学、概率论等领域有广泛应用，用于展示数据的概率密度分布。

## 火山图

### 作用

火山图用于展示基因表达的差异分析结果，横轴表示基因表达的对数值变化倍数，纵轴表示统计显著性（如 p 值的对数值），能够直观地识别差异表达基因。

### 参数

- `logfc`，基因表达的对数值变化倍数
- `pval`，统计显著性 p 值
- `alpha`，显著性阈值
- `is_log2`，是否使用 log2 转换。

### 读图

火山图中，横轴表示基因表达的变化倍数，纵轴表示统计显著性。通常，红色点表示上调的差异表达基因，蓝色点表示下调的差异表达基因，灰色点表示非差异表达基因。

### 示例代码

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 生成示例数据
np.random.seed(42)
logfc = np.random.normal(0, 1, 1000)
pval = np.random.uniform(0, 0.05, 1000)
pval[:500] = np.random.uniform(0.05, 1, 500)
data = pd.DataFrame({'logfc': logfc, 'pval': -np.log10(pval)})

# 绘制火山图
plt.figure(figsize=(10, 8))
sns.scatterplot(x='logfc', y='pval', data=data, palette='coolwarm', alpha=0.6)
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--')
plt.axvline(x=1, color='red', linestyle='--')
plt.axvline(x=-1, color='red', linestyle='--')
plt.xlabel('log2(Fold Change)')
plt.ylabel('-log10(p-value)')
plt.title('Volcano Plot')
plt.show()
```

### 其他适合领域

火山图在基因表达分析、蛋白质组学等领域有广泛应用，用于展示差异分析结果。

## 山脊图

### 作用

山脊图用于展示多个组之间的分布情况，每个组的分布以密度曲线的形式呈现，可以直观地比较不同组的分布特征。

### 参数

- `data`，输入数据
- `hue`，分组变量
- `palette`，颜色映射
- `bw_method`，带宽计算方法
- `common_norm`，是否对所有组使用相同的标准化。

### 读图

每个密度曲线代表一个组的分布情况，曲线的形状和位置反映了组内数据的分布特征。通过比较不同组的密度曲线，可以了解组间的分布差异。

### 示例代码

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 生成示例数据
np.random.seed(42)
data = pd.DataFrame({
    'Value': np.concatenate([np.random.normal(0, 1, 300), np.random.normal(2, 1, 300), np.random.normal(4, 1, 300)]),
    'Group': ['Group A'] * 300 + ['Group B'] * 300 + ['Group C'] * 300
})

# 绘制山脊图
plt.figure(figsize=(10, 6))
sns.kdeplot(data=data, x='Value', hue='Group', palette='Set2', fill=True, common_norm=False)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Ridgeline Plot')
plt.show()
```

### 其他适合领域

山脊图在生物统计、社会科学等领域有广泛应用，用于展示多组分布的比较。

## 和弦图

### 作用

和弦图用于展示细胞类型之间的相互关系、转换关系或基因共表达关系等，能够直观地呈现不同元素之间的联系强度和模式。

### 参数

- `matrix`，输入矩阵，表示元素之间的关系强度
- `labels`，元素的标签
- `colormap`，颜色映射
- `width`，控制弧的宽度
- `pad`，控制弧之间的间距
- `chordwidth`，控制和弦的宽度。

### 读图

和弦图由弧和连接弧的和弦组成，弧的长度表示元素自身的值，和弦的粗细表示元素之间关系的强度。颜色可用于区分不同的元素或关系类型。

### 示例代码

```python

```

### 其他适合领域

和弦图在社交网络分析、生态学等领域有广泛应用，用于展示元素之间的关系和交互。

## GSEA 富集图

### 作用

GSEA（Gene Set Enrichment Analysis）富集图用于分析基因集在特定生物学过程或通路中的富集情况，能够揭示基因表达变化的生物学意义。

### 参数
- `gene_sets`，基因集数据库，如`'c2.cp.kegg.v7.5.1.symbols.gmt'`
- `msigdb`，是否使用 MSigDB 数据库
- `figsize`，图的大小
- `format`，输出格式
- `no_plot`，是否显示图。

### 读图

GSEA 富集图通常包括富集分数曲线、基因集的分布位置以及显著性结果等。富集分数曲线显示基因集在排序列表中的富集程度，基因集的分布位置用点表示，显著性结果用 p 值和 FDR 校正值表示。

### 示例代码

```python
from gseapy import GSEA
# 运行 GSEA 分析
gsea_results = GSEA(data, cls=sample_labels, gene_sets='KEGG_2021_HUMAN', outdir='GSEA_Results', format='png', no_plot=False)

# 绘制 GSEA 富集图
gsea_results.plot_topTerms()
```
