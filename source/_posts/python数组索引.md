---
title: Python数组索引使用方法
date: 2023-07-04
categories:
- 学CS/SE
tags:
- Python
desc: 注意避免与C、C++混淆！
---

在 Python 中，数组通常是指列表（`list`）或 NumPy 数组。列表是 Python 的内置数据结构，而 NumPy 数组是 NumPy 库中用于高效数值计算的数据结构。以下将分别详细介绍这两种数组的索引方式。

## Python 列表索引

### 正向索引
- **定义**：从列表的开头开始计数，索引值从 0 开始递增。
- **示例**
  ```python
  my_list = ['a', 'b', 'c', 'd', 'e']
  print(my_list[0])  # 输出 'a'，第一个元素
  print(my_list[2])  # 输出 'c'，第三个元素
  ```
- **特点**：正向索引是最直观的索引方式，适用于从列表开头按顺序访问元素。

### 负向索引
- **定义**：从列表的末尾开始计数，索引值从 -1 开始递减。
- **示例**
  ```python
  my_list = ['a', 'b', 'c', 'd', 'e']
  print(my_list[-1])  # 输出 'e'，最后一个元素
  print(my_list[-3])  # 输出 'c'，倒数第三个元素
  ```
- **特点**：负向索引非常方便地访问列表的末尾元素，尤其是当不确定列表长度时。

### 切片索引
- **定义**：通过指定一个范围来获取列表的一个子集。
- **语法**：`list[start:end:step]`
  - `start`：起始索引 **（包含）**，默认为 0。
  - `end`：结束索引 **（不包含）**，默认为列表长度。
  - `step`：步长，默认为 1。
- **示例**
  ```python
  my_list = ['a', 'b', 'c', 'd', 'e']
  print(my_list[1:4])  # 输出 ['b', 'c', 'd']，从索引 1 到索引 4（不包含 4）
  print(my_list[:3])   # 输出 ['a', 'b', 'c']，从开头到索引 3（不包含 3）
  print(my_list[2:])   # 输出 ['c', 'd', 'e']，从索引 2 到末尾
  print(my_list[::2])  # 输出 ['a', 'c', 'e']，每隔一个元素取一个
  print(my_list[::-1]) # 输出 ['e', 'd', 'c', 'b', 'a']，反转列表
  ```
- **特点**：切片索引非常强大，可以快速提取列表的子集，同时支持步长和反转操作。

## NumPy 数组索引

NumPy 是 Python 中用于科学计算的一个基础库，它提供了强大的数组对象。NumPy 数组的索引方式比 Python 列表更复杂，但也更强大。

### 一维数组索引
- **正向索引**：与 Python 列表类似，从 0 开始。
  ```python
  import numpy as np
  arr = np.array([1, 2, 3, 4, 5])
  print(arr[0])  # 输出 1
  print(arr[3])  # 输出 4
  ```
- **负向索引**：从 -1 开始，从数组末尾向前计数。
  ```python
  print(arr[-1])  # 输出 5
  print(arr[-3])  # 输出 3
  ```
- **切片索引**：语法与 Python 列表类似，但支持更复杂的操作。
  ```python
  print(arr[1:4])  # 输出 [2 3 4]
  print(arr[::2])  # 输出 [1 3 5]
  ```

### 多维数组索引
- **定义**：NumPy 数组可以是多维的，索引方式类似于多维数组的坐标系统。
    - [row_start:row_end:row_step, col_start:col_end:col_step]
- **示例**
  ```python
  arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  print(arr_2d[0, 1])  # 输出 2，第一行第二列的元素
  print(arr_2d[1, :])  # 输出 [4 5 6]，第二行的所有元素
  print(arr_2d[:, 2])  # 输出 [3 6 9]，第三列的所有元素
  ```
- **切片操作**
  ```python
  print(arr_2d[1:3, 1:3])  # 输出 [[5 6]
                           #       [8 9]]，从第二行到第三行，第二列到第三列
  print(arr_2d[::2, ::2])  # 输出 [[1 3]
                           #       [7 9]]，每隔一行和一列取一个元素
  ```

### 布尔索引
- **定义**：使用布尔数组作为索引，布尔数组的长度必须与数组的某个维度长度相同。
- **示例**
  ```python
  arr = np.array([1, 2, 3, 4, 5])
  bool_mask = arr > 3
  print(bool_mask)  # 输出 [False False False  True  True]
  print(arr[bool_mask])  # 输出 [4 5]，筛选出大于 3 的元素
  ```

### 花式索引（Fancy Indexing）
- **定义**：使用整数列表或数组作为索引，可以同时访问多个不连续的元素。
- **示例**
  ```python
  arr = np.array([10, 20, 30, 40, 50])
  indices = [1, 3]
  print(arr[indices])  # 输出 [20 40]，访问索引为 1 和 3 的元素
  ```

<br>


## 与 C/C++ 数组表示的区分

### 索引方式
1. **Python**
   - **索引从 0 开始**：无论是 Python 列表还是 NumPy 数组，索引都是从 0 开始的。
   - **支持负向索引**：可以使用负数索引从数组末尾开始计数，例如 `arr[-1]` 表示最后一个元素。
   - **支持切片操作**：可以使用切片语法（如 `arr[start:end:step]`）来提取子数组。

2. **C/C++**
   - **索引从 0 开始**：C/C++ 中的数组索引也是从 0 开始的。
   - **不支持负向索引**：C/C++ 中没有负向索引的概念，只能通过手动计算来访问数组末尾的元素。
   - **不支持切片操作**：C/C++ 中没有内置的切片操作，需要手动编写循环来提取子数组。

### 内存布局
1. **Python**
   - **动态内存分配**：Python 列表是动态数组，可以根据需要自动扩展和收缩。NumPy 数组也是动态分配内存的，但一旦创建，大小通常固定。
   - **连续内存**：NumPy 数组在内存中是连续存储的，这使得它在进行数值计算时效率很高。Python 列表的内存布局相对复杂，因为列表中的元素可以是任意类型的对象，每个元素可能存储在不同的内存位置。

2. **C/C++**
   - **静态或动态内存分配**：C/C++ 中的数组可以是静态分配（在栈上分配，大小在编译时确定）或动态分配（在堆上分配，大小在运行时确定）。
   - **连续内存**：C/C++ 中的数组在内存中是连续存储的，这使得数组操作非常高效，尤其是在进行指针操作时。

### 操作方式
1. **Python**
   - **高级操作**：Python 提供了丰富的内置方法和库（如 NumPy）来操作数组，例如排序、过滤、切片等。
   - **简洁语法**：Python 的数组操作语法简洁，例如 `arr.sort()`、`arr[::-1]` 等。

2. **C/C++**
   - **底层操作**：C/C++ 中的数组操作通常需要手动编写循环和指针操作，例如排序、查找等操作需要自己实现算法。
   - **指针操作**：C/C++ 中的数组操作常常依赖指针，例如通过指针偏移来访问数组元素。

### 示例对比

#### Python
```python
# 创建一个数组
arr = [1, 2, 3, 4, 5]

# 访问元素
print(arr[0])  # 输出 1
print(arr[-1]) # 输出 5

# 切片操作
print(arr[1:4])  # 输出 [2, 3, 4]
print(arr[::-1]) # 输出 [5, 4, 3, 2, 1]
```

#### C
```c
#include <stdio.h>

int main() {
    // 创建一个数组
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]); // 获取数组长度

    // 访问元素
    printf("%d\n", arr[0]);  // 输出 1
    printf("%d\n", arr[n - 1]);  // 输出 5

    // 手动实现切片操作（输出从索引 1 到索引 3 的元素）
    for (int i = 1; i < 4; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // 手动实现反转数组
    for (int i = n - 1; i >= 0; i--) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
```

#### C++
```cpp
#include <iostream>
#include <vector>

int main() {
    // 创建一个动态数组（vector）
    std::vector<int> arr = {1, 2, 3, 4, 5};

    // 访问元素
    std::cout << arr[0] << std::endl;  // 输出 1
    std::cout << arr[arr.size() - 1] << std::endl;  // 输出 5

    // 手动实现切片操作（输出从索引 1 到索引 3 的元素）
    for (int i = 1; i < 4; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // 手动实现反转数组
    for (int i = arr.size() - 1; i >= 0; i--) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### 总结
- **Python**：
  - **优点**：操作方便，语法简洁，支持高级操作（如切片、布尔索引等）。
  - **缺点**：底层实现相对复杂，性能可能不如 C/C++。
- **C/C++**：
  - **优点**：底层控制能力强，内存布局简单，性能高。
  - **缺点**：操作复杂，需要手动编写大量代码，容易出错。

在实际开发中，选择合适的语言和数据结构取决于具体需求。如果需要高效的数据处理和复杂的操作，Python 是一个不错的选择；如果需要高性能和底层控制，C/C++ 更适合。

<br>

## 应用：数据预处理

- 参考：<a href="https://zh.d2l.ai/chapter_preliminaries/pandas.html">动手学DL</a>

在数据预处理中，处理缺失值是一个常见的步骤。可以选择插值法或删除法来处理缺失值。
这里我们使用插值法。

### 使用 iloc 分割数据
iloc 是 pandas 中的一个非常强大的工具，用于基于位置的索引。它允许我们通过行号和列号来访问数据。
语法：data.iloc[row_start:row_end, col_start:col_end]
row_start 和 row_end：指定行的范围（包括起始行，不包括结束行）。
col_start 和 col_end：指定列的范围（包括起始列，不包括结束列）。
示例：
```Python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print("Inputs:\n", inputs)
print("Outputs:\n", outputs)
输出：
复制
Inputs:
    NumRooms Alley
0       NaN  Pave
1       2.0   NaN
2       4.0   NaN
3       NaN   NaN
Outputs:
0    127500
1    106000
2    178100
3    140000
```

对于数值型数据，我们可以用同一列的均值来填充缺失值。

```Python
inputs = inputs.fillna(inputs.mean())
print(inputs)

输出：
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
```

### 处理类别型缺失值  

对于类别型（离散）特征，把缺失值 `"NaN"` 当作一个独立类别，并用独热编码（one-hot）表示。 

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

输出：
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```

说明：
原列 Alley 只有两个取值 "Pave" 和 "NaN"，因此被拆成两列 Alley_Pave 和 Alley_nan。
缺失值所在行在 Alley_nan 上置 1，其余行在 Alley_Pave 上置 1。

### 转换为张量格式
所有列都已是数值型，可直接转成张量。以 MXNet 为例：
```Python
from mxnet import np

X = np.array(inputs.to_numpy(dtype=float))   # 特征张量
y = np.array(outputs.to_numpy(dtype=float))  # 标签张量
X, y
输出：
(array([[3., 1., 0.],
        [2., 0., 1.],
        [4., 0., 1.],
        [3., 0., 1.]], dtype=float64),
 array([127500., 106000., 178100., 140000.], dtype=float64))
```

至此，数据已准备好，可以送入深度学习模型训练。