---
title: 科研论文常见可视化图表
date: 2025-05-29
categories:
  - 学AI/DS
tags:
  - 科研技巧
  - 可视化
desc: 流程图构图、素材与配色工具&技巧，理解与绘制：柱状图、散点图、饼图、雷达图、箱线图、ROC曲线、桑基图、网络图
---

参考资料：


生信相关可视化图表参考：[Single Cell绘图](../SingleCell绘图)【肘部图、PCA、UMAP、tSNE、热图、小提琴图、等高线图、密度图、火山图、山脊图、和弦图、GSEA富集图】

# 配色

## 参考网站

## 常见搭配

## prompt

# 素材

## 参考网站

## ppt绘制

# 流程图

## 总体workflow

先把内容输入gpt，让gpt设计几版本供自己参考布局。
图是撑满的、方形结构。


## 模块详细架构

icon
例子


# 结果可视化图表

## 柱状图
**Bar Chart**

### 作用
柱状图用于比较不同类别或组之间的数值大小，能够直观地展示数据的分布和对比关系。

### 参数
- `x`：指定 x 轴的类别或组别，通常是离散的文本标签或分类变量。
- `height`：指定每个柱子的高度，表示对应的数值大小。
- `width`：指定柱子的宽度，默认值通常为 0.8，可通过调整该值改变柱子之间的间距和整体布局。
- `bottom`：指定柱子的起始位置，默认为 0，可用于堆叠柱状图等场景。
- `align`：指定柱子的对齐方式，默认为`'center'`，也可选择`'edge'`。
- `color`：指定柱子的颜色，可以是单一颜色或颜色列表，用于区分不同的类别或组。

### 示例代码
```python
import matplotlib.pyplot as plt

# 数据
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 35]

# 创建柱状图
plt.bar(categories, values, width=0.5, bottom=0, align='center', color=['blue', 'green', 'red', 'orange'])

# 添加标题和标签
plt.title('Bar Chart Example')
plt.xlabel('Categories')
plt.ylabel('Values')

# 显示图表
plt.show()
```

## 散点图
**Scatter Plot**

#### 作用
散点图用于展示两个变量之间的关系，可以揭示数据的分布模式、相关性以及异常值等。

#### 参数
- `x`：指定 x 轴的数值变量。
- `y`：指定 y 轴的数值变量。
- `s`：指定点的大小，默认值通常为 20，可以是单一数值或数组，用于表示第三个变量的大小。
- `c`：指定点的颜色，默认为`'blue'`，可以是单一颜色、颜色列表或数值数组（用于颜色映射）。
- `marker`：指定点的形状，默认为`'o'`，可选择如`','`、`'.'`、`'s'`、`'P'`等。
- `alpha`：指定点的透明度，默认为 1，可用于处理重叠点的可视化问题。

#### 示例代码
```python
import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 创建散点图
plt.scatter(x, y, s=100, c='red', marker='o', alpha=0.7)

# 添加标题和标签
plt.title('Scatter Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图表
plt.show()
```

## 饼图
**Pie Chart**

### 作用
饼图用于展示各部分占整体的比例关系，强调各分类在总量中的占比情况。

### 参数
- `x`：指定各部分的数值大小，决定了饼图中各扇形的面积占比。
- `labels`：指定各部分的标签，用于标识每个扇形代表的分类。
- `explode`：指定各扇形的偏移距离，默认为`None`，可用于突出显示某个部分。
- `colors`：指定各扇形的颜色，默认为 Matplotlib 的默认颜色循环，可以是颜色列表。
- `autopct`：指定在扇形内部显示的文本格式，如`'%1.1f%%'`用于显示百分比。
- `startangle`：指定饼图的起始角度，默认为 0，用于调整饼图的起始方向。

### 示例代码
```python
import matplotlib.pyplot as plt

# 数据
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]

# 创建饼图
plt.pie(sizes, labels=labels, explode=(0.1, 0, 0, 0), colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen'], autopct='%1.1f%%', startangle=90)

# 添加标题
plt.title('Pie Chart Example')

# 显示图表
plt.show()
```

## 雷达图
**Radar Chart**

### 作用
雷达图用于展示多维数据的各个特征值，能够直观地比较不同样本在多个变量上的表现，常用于性能评估、特征分析等场景。

### 参数
- `data`：指定每个样本在各个特征上的数值。
- `labels`：指定各个特征的标签，用于标注雷达图的轴。
- `radar_range`：指定每个特征的数值范围，默认通常为`[0, 100]`，可根据实际数据调整，以确保数据在合理范围内展示。
- `color`：指定雷达图填充的颜色，默认为`'blue'`，通过透明度和渐变效果展示不同样本的覆盖范围。
- `fill`：指定是否填充雷达图的区域，默认为`True`，用于突出样本的综合表现区域。
- `alpha`：指定填充区域的透明度，默认为 0.25，方便多个雷达图叠加时观察重叠区域。

### 示例代码
```python
import numpy as np
import matplotlib.pyplot as plt

# 数据
labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
data1 = np.array([80, 90, 70, 85, 75])
data2 = np.array([70, 85, 90, 65, 80])

# 计算雷达图的角度
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# 闭合数据
data1 = np.concatenate((data1, [data1[0]]))
data2 = np.concatenate((data2, [data2[0]]))
angles += angles[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, data1, color='red', alpha=0.25)
ax.fill(angles, data2, color='blue', alpha=0.25)
ax.plot(angles, data1, color='red', linewidth=2, label='Sample 1')
ax.plot(angles, data2, color='blue', linewidth=2, label='Sample 2')
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 添加标题
plt.title('Radar Chart Example')

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# 显示图表
plt.show()
```

## 箱线图
**Boxplot**

### 作用
箱线图用于展示数据的分布特征，包括中位数、四分位数、异常值等，能够快速识别数据的中心位置、离散程度以及异常值情况。

### 参数
- `data`：指定输入数据，可以是数组或数组列表，用于绘制一个或多个箱线图。
- `vert`：指定箱线图的方向，默认为`True`（垂直），设置为`False`时为水平方向。
- `widths`：指定箱体的宽度，默认为 0.5，对于水平箱线图则是箱体的高度。
- `patch_artist`：指定是否使用`PatchArtist`绘制箱体，默认为`False`，设置为`True`时可以自定义箱体的颜色和样式。
- `sym`：指定异常值的标记样式，默认为`'b+'`，可指定颜色和标记形状。
- `whis`：指定须的长度，默认为 1.5，表示须的范围是四分位距的 1.5 倍，超出此范围的值被视为异常值。

### 示例代码
```python
import matplotlib.pyplot as plt

# 数据
data = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]

# 创建箱线图
plt.boxplot(data, vert=True, patch_artist=True, widths=0.5, sym='r+', whis=1.5)

# 添加标题和标签
plt.title('Boxplot Example')
plt.xlabel('Groups')
plt.ylabel('Values')

# 显示图表
plt.show()
```

## ROC 曲线
**ROC Curve**

### 作用
ROC 曲线用于评估二分类模型的性能，通过绘制真正例率与假正例率之间的关系，能够直观地展示模型在不同阈值下的分类能力，帮助选择最优的分类阈值。

### 参数
- `fpr`：假正例率数组，表示在不同阈值下模型将负类错误分类为正类的比例。
- `tpr`：真正例率数组，表示在不同阈值下模型正确识别正类的比例。
- `roc_auc`：ROC 曲线下的面积（AUC），用于量化模型的整体性能，值越大表示模型性能越好。
- `label`：图例标签，默认为`None`，用于标识不同模型或条件下的 ROC 曲线。
- `color`：曲线的颜色，默认为`'blue'`，可通过指定颜色参数区分多条曲线。
- `lw`：线宽，默认为 2，可根据需要调整曲线的粗细以增强可视化效果。

### 示例代码
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 数据
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

# 计算 ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 创建 ROC 曲线图
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Example')
plt.legend(loc='lower right')

# 显示图表
plt.show()
```

## 桑基图
**Sankey Diagram**

### 作用
桑基图用于展示数据的流向和转换关系，能够直观地呈现不同节点之间的流量或转移情况，常用于能源流向、资金流动、用户转化等场景。

### 参数
- `data`：指定节点之间的流量数据，通常为包含源节点、目标节点和流量的列表或 DataFrame。
- `nodeWidth`：指定节点的宽度，默认通常为 20，可根据需要调整节点的大小以适应不同数据规模和布局。
- `nodePadding`：指定节点之间的间距，默认通常为 10，用于避免节点重叠，确保图表的可读性。
- `splinePadding`：指定流向曲线与节点之间的距离，默认通常为 8，用于调整曲线的形状和布局。
- `colors`：指定节点的颜色，默认为`'d3.schemeCategory10'`，可通过颜色映射或自定义颜色列表区分不同节点类别。
- `marginTop`、`marginRight`、`marginBottom`、`marginLeft`：指定图表的上、右、下、左外边距，默认通常为 0，可根据需要调整图表在容器中的位置和布局。

### 示例代码
```python
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.palettes import Category10

# 数据
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
links = [
    {'source': 'A', 'target': 'B', 'value': 5},
    {'source': 'A', 'target': 'C', 'value': 3},
    {'source': 'B', 'target': 'D', 'value': 4},
    {'source': 'B', 'target': 'E', 'value': 2},
    {'source': 'C', 'target': 'E', 'value': 1},
    {'source': 'C', 'target': 'F', 'value': 2},
    {'source': 'D', 'target': 'F', 'value': 3},
    {'source': 'E', 'target': 'F', 'value': 2}
]

# 创建桑基图
output_file('sankey_example.html')
p = figure(title='Sankey Diagram Example', x_range=nodes, y_range=nodes, plot_width=800, plot_height=600)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# 绘制节点
node_coords = [(i, 0) for i in range(len(nodes))]
node_source = ColumnDataSource({'x': [x for x, y in node_coords], 'y': [y for x, y in node_coords], 'names': nodes})
p.circle(x='x', y='y', size=20, source=node_source, color=Category10[10][:len(nodes)], legend_field='names')

# 绘制连接线
for link in links:
    source_idx = nodes.index(link['source'])
    target_idx = nodes.index(link['target'])
    p.quadratic_curve(x0=source_idx, y0=0, x1=target_idx, y1=0, x2=(source_idx + target_idx) / 2, y2=0.5, line_width=link['value'], color='gray')

# 添加标签
labels = LabelSet(x='x', y='y', text='names', level='glyph', source=node_source, text_align='center', text_baseline='middle')
p.add_layout(labels)

# 显示图表
show(p)
```

## 网络图
**Network Graph**

### 作用
网络图用于展示实体之间的关系和连接结构，能够直观地呈现复杂系统中的交互关系，如社交网络、分子相互作用网络等。

### 参数
- `nodes`：指定网络中的节点，可以是节点的列表或包含节点属性的列表。
- `edges`：指定节点之间的连接关系，通常为包含源节点和目标节点的列表或包含权重的列表。
- `node_size`：指定节点的大小，默认通常为 300，可根据节点的重要性和数据属性进行调整，如按度中心性大小设置节点尺寸，突出关键节点。
- `node_color`：指定节点的颜色，默认为`'skyblue'`，可以是单一颜色或颜色列表，用于区分不同类型的节点或根据节点属性映射颜色。
- `edge_color`：指定边的颜色，默认为`'black'`，可用于区分不同类型的关系或表示边的权重。
- `width`：指定边的宽度，默认为 1.0，可根据边的权重进行调整，反映关系的强弱。
- `edge_cmap`：指定边的颜色映射，默认为`None`，用于根据边的权重或其他属性进行颜色编码，增强关系的可视化效果。
- `style`：指定边的线条样式，默认为`'solid'`，可选择如`'dashed'`、`'dotted'`等，用于区分不同类型的边。

### 示例代码
```python
import networkx as nx
import matplotlib.pyplot as plt

# 创建网络图
G = nx.Graph()

# 添加节点
nodes = ['A', 'B', 'C', 'D', 'E']
G.add_nodes_from(nodes)

# 添加边
edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E')]
G.add_edges_from(edges)

# 绘制网络图
pos = nx.spring_layout(G)  # 使用 Fruchterman-Reingold 布局算法
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='black', linewidths=1, font_size=15)

# 添加标题
plt.title('Network Graph Example')

# 显示图表
plt.show()
```
