---
title: Python创建对象的地址问题
date: 2023-07-05
categories:
- 学CS/SE
tags:
- Python
desc: 本文以CPython为标准。感觉相关知识很乱，比较安全的方法是：需要副本就用copy()，需要共享就直接赋值；修改时优先使用+=、-=等原地操作符；不确定时先复制再操作；关键位置用id()验证对象关系。
---

## 内存地址与对象标识

Python 中的 `id()` 函数返回对象在内存中的唯一标识符，通常对应对象的内存地址。在 CPython 实现中，`id()` 返回的是对象在内存中的实际地址值。每个对象在创建时都会被分配一个唯一的内存地址，这个地址在对象的生命周期内保持不变，直到对象被垃圾回收。

> [!NOTE]+ CPython 是什么
> CPython 是 Python 编程语言的官方参考实现，使用 C 语言编写(因此得名 CPython)。它是 Python 解释器的最常见实现，通常我们安装的 Python 就是 CPython。
> 用 C 语言编写，
> - **对象模型**：所有 Python 对象在底层都是 C 结构体（`PyObject`），`id()` 返回的就是指向这些结构体的指针值
> - **性能**：作为参考实现，CPython 在稳定性和兼容性方面表现优秀
> - **其他实现**：除了 CPython，还有 PyPy（使用 JIT 编译）、Jython（运行在 JVM 上）、IronPython（运行在 .NET 上）等，它们对 `id()` 的实现可能不同，但都保证返回值在对象生命周期内唯一且不变


<br>

`id()` 函数的底层实现（CPython 源码）：

```c
// CPython 中 id() 函数的实现（Python/bltinmodule.c）
static PyObject *
builtin_id(PyObject *self, PyObject *v)
{
    // 直接返回对象指针的整数值
    // 在 CPython 中，PyObject* 指针的值就是对象的内存地址
    return PyLong_FromVoidPtr(v);
}

// PyLong_FromVoidPtr 将指针转换为 Python 整数
PyObject *
PyLong_FromVoidPtr(void *p)
{
    // 将指针值转换为 Python 的 long 对象
    return PyLong_FromUnsignedLong((unsigned long)(uintptr_t)p);
}
```

在 CPython 中，所有对象都是 `PyObject` 结构体的实例，`id()` 函数直接返回指向该结构体的指针值，这个指针值就是对象在内存中的地址。其他 Python 实现（如 PyPy、Jython）可能使用不同的策略，但都保证 `id()` 返回的值在对象生命周期内唯一且不变。

```python
import numpy as np

X = np.array([1, 2, 3])
Y = np.array([4, 5, 6])

print(f'id(X): {id(X)}')
print(f'id(Y): {id(Y)}')
```

理解 `id()` 的返回值对于追踪对象的内存分配和引用关系至关重要。当两个变量引用同一个对象时，它们的 `id()` 值相同；当对象被重新分配时，`id()` 值会发生变化。

<br>

## 赋值操作的内存分配机制

### 普通赋值创建新对象

执行 `Y = Y + X` 时，Python 解释器会按照以下步骤操作：

1. 首先在堆内存中计算 `Y + X` 的结果，创建一个新的临时对象
2. 为新对象分配独立的内存空间
3. 将变量 `Y` 的引用从旧对象转移到新对象
4. 旧对象如果没有其他引用，会被 Python 的垃圾回收机制回收

```python
before = id(Y)
Y = Y + X  # 创建新对象
print(f'id(Y) == before: {id(Y) == before}')  # False
```

这种操作的内存开销包括：
- 新对象的堆内存分配
- 旧对象的垃圾回收开销
- 引用计数的维护成本

在机器学习场景中，当参数矩阵达到数百兆甚至更大规模时，频繁创建新对象会导致显著的内存压力和性能下降。此外，如果代码中存在对旧对象的其他引用，这些引用不会自动更新，可能导致数据不一致的问题。

### 原地操作的内存优化

原地操作（in-place operations）直接修改现有对象的内存内容，而不创建新对象。这避免了内存分配和垃圾回收的开销。

#### 切片赋值实现原地操作

使用切片赋值 `[:]` 可以将计算结果写入已分配的内存空间：

```python
Z = np.zeros_like(Y)
print(f'id(Z): {id(Z)}')
Z[:] = X + Y  # 原地赋值，修改 Z 指向的内存内容
print(f'id(Z): {id(Z)}')  # 地址不变，内存内容已更新
```

切片赋值 `Z[:]` 表示对 `Z` 的所有元素进行赋值，NumPy 会直接将右侧表达式的计算结果写入 `Z` 已分配的内存缓冲区，而不是创建新对象。

#### 增量赋值操作符

`+=` 操作符在 NumPy 数组上会调用 `__iadd__` 方法，该方法执行原地更新：

```python
before = id(X)
X += Y  # 调用 X.__iadd__(Y)，原地修改 X
print(f'id(X) == before: {id(X) == before}')  # True
```

`__iadd__` 方法会检查操作是否可以原地执行，如果可以，直接修改对象内容；如果不可行（例如类型不兼容），则回退到创建新对象的方式。

NumPy 数组的 `__iadd__` 方法底层实现（简化版）：

```python
# NumPy ndarray.__iadd__ 的简化实现逻辑
class ndarray:
    def __iadd__(self, other):
        # 检查是否可以原地操作
        if self.flags.writeable and np.can_cast(other.dtype, self.dtype):
            # 使用 ufunc 进行原地计算
            np.add(self, other, out=self)  # 将结果写入 self
            return self  # 返回 self，保持对象标识
        else:
            # 不可行时回退到 __add__
            return NotImplemented
```

实际 NumPy 实现中，`np.add(self, other, out=self)` 会调用底层的 C 函数，直接将计算结果写入 `self` 的内存缓冲区，避免创建新对象。


<br>

## 增量赋值与普通赋值的底层差异

### `+=` 操作符的语义

`arr += X` 在底层调用 `arr.__iadd__(X)` 方法。对于 NumPy 数组，该方法会：

1. 检查操作数类型和形状是否兼容
2. 如果兼容，直接修改 `arr` 的内存内容
3. 返回 `self` 引用，保持对象标识不变

```python
import numpy as np

arr = np.array([1, 2, 3])
before = id(arr)
arr += np.array([4, 5, 6])  # 调用 __iadd__，原地修改
print(f'id(arr) == before: {id(arr) == before}')  # True
```

Python 解释器处理 `+=` 的底层流程（简化版）：

```python
# Python 字节码层面的处理逻辑（伪代码）
def handle_inplace_add(obj, other):
    # 1. 首先尝试调用 __iadd__
    if hasattr(obj, '__iadd__'):
        result = obj.__iadd__(other)
        if result is not NotImplemented:
            return result  # 返回修改后的对象
    
    # 2. 如果 __iadd__ 不存在或返回 NotImplemented，回退到 __add__
    if hasattr(obj, '__add__'):
        result = obj.__add__(other)
        # 将结果赋值回原变量（这会改变引用）
        return result
    
    # 3. 都不存在则抛出 TypeError
    raise TypeError(f"unsupported operand type(s) for +=: {type(obj)} and {type(other)}")
```

NumPy 数组的 `__iadd__` 实现会调用底层的 ufunc（通用函数）：

```python
# NumPy 底层 C 扩展的简化逻辑
# 实际实现在 numpy/core/src/umath/loops.c 中
def array_iadd_impl(self, other):
    # 检查类型兼容性和可写性
    if not self.flags.writeable:
        return NotImplemented
    
    # 广播检查
    if not np.can_broadcast(self.shape, other.shape):
        return NotImplemented
    
    # 调用底层 C 函数进行原地计算
    # 实际调用: PyUFunc_GenericFunction(ufunc, args, kwargs)
    # 其中 out 参数指向 self，实现原地操作
    ufunc_add(self, other, out=self)
    return self
```

### `= +` 操作的语义

`arr = arr + X` 的执行流程：

1. 计算 `arr + X`，创建临时对象
2. 调用 `arr.__add__(X)` 方法，返回新对象
3. 将 `arr` 的引用绑定到新对象
4. 旧对象如果没有其他引用，等待垃圾回收

```python
arr = np.array([1, 2, 3])
before = id(arr)
arr = arr + np.array([4, 5, 6])  # 调用 __add__，创建新对象
print(f'id(arr) == before: {id(arr) == before}')  # False
```

Python 解释器处理 `+` 的底层流程（简化版）：

```python
# Python 字节码层面的处理逻辑（伪代码）
def handle_add(obj, other):
    # 1. 调用 __add__ 方法
    if hasattr(obj, '__add__'):
        result = obj.__add__(other)
        if result is not NotImplemented:
            return result  # 返回新对象
    
    # 2. 尝试反向调用 other.__radd__
    if hasattr(other, '__radd__'):
        result = other.__radd__(obj)
        if result is not NotImplemented:
            return result
    
    # 3. 都不存在则抛出 TypeError
    raise TypeError(f"unsupported operand type(s) for +: {type(obj)} and {type(other)}")
```

NumPy 数组的 `__add__` 实现会创建新数组：

```python
# NumPy ndarray.__add__ 的简化实现逻辑
class ndarray:
    def __add__(self, other):
        # 1. 广播检查
        other = np.asarray(other)
        result_shape = np.broadcast_shapes(self.shape, other.shape)
        
        # 2. 创建新数组（分配新内存）
        result = np.empty(result_shape, dtype=self.dtype)
        
        # 3. 执行计算，结果写入新数组
        np.add(self, other, out=result)  # out 参数指向新数组
        
        # 4. 返回新对象
        return result
```

实际 NumPy 实现中，`np.add(self, other, out=result)` 会调用底层的 C 函数，将计算结果写入新分配的内存缓冲区。

### 列表类型的特殊行为

Python 内置列表的 `+=` 操作通过 `list.__iadd__` 实现，该方法会调用 `list.extend()`，直接修改列表内容：

```python
# 列表的 += 是原地操作
lst = [1, 2, 3]
before = id(lst)
lst += [4, 5, 6]  # 调用 list.__iadd__，原地扩展
print(f'id(lst) == before: {id(lst) == before}')  # True
```

而 `lst = lst + [4, 5, 6]` 会创建新列表：

```python
# 列表的 = + 创建新对象
lst = [1, 2, 3]
before = id(lst)
lst = lst + [4, 5, 6]  # 调用 list.__add__，创建新列表
print(f'id(lst) == before: {id(lst) == before}')  # False
```

这种差异源于列表的 `__iadd__` 和 `__add__` 方法的不同实现策略。

Python 列表的底层实现（CPython 源码简化版）：

```python
# CPython 中 list 对象的 __iadd__ 实现（Objects/listobject.c 简化）
class list:
    def __iadd__(self, iterable):
        # 直接调用 extend，原地修改
        self.extend(iterable)
        return self  # 返回 self，保持对象标识
    
    def extend(self, iterable):
        # 底层实现会：
        # 1. 检查是否需要扩容
        # 2. 将 iterable 的元素追加到列表末尾
        # 3. 更新列表的长度和容量
        # 实际实现在 C 层面，直接操作 PyListObject 结构体
        pass

# CPython 中 list 对象的 __add__ 实现（简化）
class list:
    def __add__(self, other):
        # 创建新列表
        result = []
        # 复制 self 的所有元素
        result.extend(self)
        # 追加 other 的元素
        result.extend(other)
        return result  # 返回新对象
```

实际 CPython 实现中，`list.__iadd__` 在 `Objects/listobject.c` 的 `list_inplace_concat` 函数中实现，直接修改列表的内部数组；而 `list.__add__` 在 `list_concat` 函数中实现，会创建新的 `PyListObject` 结构体。

> [!EXAMPLE]+ 具体数据示例：`+=` vs `= +`
> 
> 以下示例展示两种操作在内存层面的具体差异：
> 
> ```python
> # 初始状态
> lst1 = [1, 2, 3]
> lst2 = [1, 2, 3]
> 
> print(f"初始 lst1 的 id: {id(lst1)}")  # 例如: 140234567890000
> print(f"初始 lst2 的 id: {id(lst2)}")  # 例如: 140234567890128
> 
> # 保存引用
> ref1 = lst1
> ref2 = lst2
> 
> # 使用 += (调用 __iadd__)
> lst1 += [4, 5]
> print(f"lst1 += [4, 5] 后:")
> print(f"  lst1 的 id: {id(lst1)}")      # 仍然是 140234567890000 (不变)
> print(f"  ref1 的 id: {id(ref1)}")      # 仍然是 140234567890000 (不变)
> print(f"  lst1 is ref1: {lst1 is ref1}")  # True (同一对象)
> print(f"  lst1 的内容: {lst1}")         # [1, 2, 3, 4, 5]
> print(f"  ref1 的内容: {ref1}")         # [1, 2, 3, 4, 5] (同步变化)
> 
> # 使用 = + (调用 __add__)
> lst2 = lst2 + [4, 5]
> print(f"\nlst2 = lst2 + [4, 5] 后:")
> print(f"  lst2 的 id: {id(lst2)}")      # 新地址，例如: 140234567890256 (改变)
> print(f"  ref2 的 id: {id(ref2)}")      # 仍然是 140234567890128 (旧对象)
> print(f"  lst2 is ref2: {lst2 is ref2}")  # False (不同对象)
> print(f"  lst2 的内容: {lst2}")         # [1, 2, 3, 4, 5]
> print(f"  ref2 的内容: {ref2}")         # [1, 2, 3] (旧内容，未变化)
> ```
> 
> **内存层面的差异**：
> - **`+=` 操作**：`list_inplace_concat` 函数直接修改 `lst1` 指向的 `PyListObject` 结构体中的 `ob_item` 数组指针，扩展数组容量（如果需要），然后追加新元素。对象的内存地址（`id`）保持不变。
> - **`= +` 操作**：`list_concat` 函数创建新的 `PyListObject` 结构体，分配新的内存空间，复制原列表的所有元素，再追加新元素。`lst2` 的引用被重新绑定到新对象，旧对象如果没有其他引用会被垃圾回收。

<br>

## 内存优化的实践策略

在深度学习训练循环中，参数更新操作可能每秒执行数千次。使用原地操作可以显著减少内存分配和垃圾回收的开销。

### 参数更新模式

不推荐的方式会创建新对象：

```python
# 创建新对象，内存开销大
Y = Y + X
```

推荐使用原地操作：

```python
# 原地更新，内存高效
Y += X
# 或使用切片赋值
Y[:] = Y + X
```

### 原地操作的使用场景

适合使用原地操作的场景：

1. **参数更新**：在梯度下降等优化算法中，参数需要频繁更新，且不需要保留历史值
2. **内存受限环境**：在 GPU 内存或系统内存有限的情况下，减少内存分配可以避免 OOM 错误
3. **保持引用一致性**：当多个变量引用同一对象时，原地操作可以确保所有引用看到相同的更新

需要注意的场景：

1. **需要保留原始值**：如果后续计算需要原始数据，应该先创建副本再操作
2. **自动微分框架**：某些框架（如 PyTorch）的自动微分需要追踪操作历史，原地操作可能破坏计算图

<br>

## 对象转换时的内存隔离

### NumPy 数组与框架张量的转换

深度学习框架（PyTorch、TensorFlow、MXNet 等）的张量与 NumPy 数组之间的转换会创建新的内存缓冲区，两者不共享内存。这种设计是为了避免数据竞争和确保计算的正确性。

```python
# 假设 X 是深度学习框架的张量
import torch
X_tensor = torch.tensor([1, 2, 3])

# 转换为 NumPy 数组
A = X_tensor.numpy()  # 或 X.asnumpy()（MXNet）
B = np.array(A)

print(f'type(A): {type(A)}')  # numpy.ndarray
print(f'type(B): {type(B)}')  # numpy.ndarray
```

内存隔离的原因：

1. **计算设备差异**：框架张量可能位于 GPU 内存，而 NumPy 数组位于 CPU 内存，物理上无法共享
2. **异步执行**：GPU 计算是异步的，如果共享内存，NumPy 操作可能读取到未完成的计算结果
3. **数据布局差异**：框架可能使用特定的内存布局（如行主序、列主序、stride 等），与 NumPy 的默认布局不同
4. **生命周期管理**：框架和 NumPy 使用不同的内存管理机制，共享内存会导致生命周期管理的复杂性

因此，修改转换后的 NumPy 数组不会影响原始张量：

```python
A = X_tensor.numpy()
A[0] = 999  # 修改 A 不会影响 X_tensor
print(X_tensor[0])  # 仍然是原始值
```

### 标量提取方法

将形状为 `(1,)` 的数组转换为 Python 标量有多种方法：

```python
a = np.array([3.5])

# 方法1：使用 item() 方法
scalar1 = a.item()  # 返回 Python 标量，类型为 float

# 方法2：使用类型转换函数
scalar2 = float(a)  # 调用 a.__float__()，返回 float
scalar3 = int(a)    # 调用 a.__int__()，返回 int

print(f'a: {a}')           # array([3.5])
print(f'a.item(): {scalar1}')  # 3.5 (float)
print(f'float(a): {scalar2}')  # 3.5 (float)
print(f'int(a): {scalar3}')    # 3 (int)
```

`item()` 方法会检查数组是否只包含一个元素，如果是则返回该元素的 Python 原生类型；如果数组包含多个元素，会抛出 `ValueError`。类型转换函数（`float()`、`int()`）在底层调用数组的相应魔术方法，也会进行类似的检查。

类型转换的底层实现（简化版）：

```python
# Python 内置函数 float() 的底层逻辑（简化）
def float(obj):
    # 1. 如果对象有 __float__ 方法，调用它
    if hasattr(obj, '__float__'):
        result = obj.__float__()
        if result is not NotImplemented:
            return result
    
    # 2. 尝试其他转换方式
    # ... 其他转换逻辑
    
    # 3. 都不行则抛出 TypeError
    raise TypeError(f"can't convert {type(obj)} to float")

# NumPy 数组的 __float__ 实现（简化）
class ndarray:
    def __float__(self):
        # 检查数组大小
        if self.size != 1:
            raise ValueError("只能转换大小为1的数组为标量")
        # 返回第一个元素的 Python 原生类型
        return float(self.flat[0])
    
    def __int__(self):
        if self.size != 1:
            raise ValueError("只能转换大小为1的数组为标量")
        return int(self.flat[0])
    
    def item(self):
        if self.size != 1:
            raise ValueError("只能提取大小为1的数组的元素")
        # 根据 dtype 返回相应的 Python 原生类型
        return self.flat[0].item()  # 调用元素的 item() 方法
```

实际 NumPy 实现中，这些方法在 C 层面实现，会直接访问数组的数据缓冲区，提取标量值并转换为相应的 Python 对象。

<br>

## 安全实践建议

由于 Python 的内存模型和引用语义容易导致混淆，以下是比较安全、不容易出错的做法：

### 明确区分需要副本还是引用

当需要独立的数据副本时，始终使用 `copy()` 方法：

```python
import numpy as np

arr = np.array([1, 2, 3])
# 需要独立副本时
arr_copy = arr.copy()  # 明确创建副本
arr_copy += 1
# arr 和 arr_copy 互不影响
```

当需要共享数据时，直接赋值即可：

```python
# 需要共享数据时
arr_ref = arr  # 直接赋值，共享引用
arr_ref += 1
# arr 和 arr_ref 指向同一对象，都会改变
```

### 优先使用原地操作符

对于需要修改数组内容的场景，优先使用 `+=`、`-=`、`*=` 等原地操作符：

```python
# 安全做法：使用原地操作
arr += other_arr  # 明确表示原地修改
arr *= 2          # 明确表示原地修改
```

避免使用可能产生歧义的写法：

```python
# 容易混淆：不清楚是否创建新对象
arr = arr + other_arr  # 创建新对象，改变引用
```

### 需要保留原始值时先复制

如果后续计算需要原始值，在操作前先创建副本：

```python
# 安全做法：先复制再操作
original = arr.copy()
arr += modification
# 此时 original 保留原始值，arr 是修改后的值
```

### 使用切片赋值进行批量更新

当需要将计算结果写回原数组时，使用切片赋值：

```python
# 安全做法：切片赋值保持对象标识
result = complex_calculation(arr)
arr[:] = result  # 原地更新，不改变 arr 的引用
```

### 转换时明确内存关系

在框架张量和 NumPy 数组之间转换时，明确它们不共享内存：

```python
import torch

# 安全做法：明确转换会复制数据
tensor = torch.tensor([1, 2, 3])
numpy_arr = tensor.numpy()  # 复制数据，不共享内存

# 修改 numpy_arr 不会影响 tensor
numpy_arr[0] = 999
print(tensor[0])  # 仍然是 1
```

### 使用 id() 验证对象关系

在关键位置使用 `id()` 验证对象关系是否符合预期：

```python
def safe_operation(arr, other):
    original_id = id(arr)
    
    # 执行操作
    arr += other
    
    # 验证对象标识是否保持
    assert id(arr) == original_id, "对象标识意外改变"
    return arr
```

### 避免在循环中重复创建对象

在循环中更新数组时，使用原地操作避免内存浪费：

```python
# 安全做法：循环中使用原地操作
arr = np.zeros(1000)
for i in range(100):
    arr += compute_update(i)  # 原地更新，不创建新对象

# 不安全做法：每次循环都创建新对象
arr = np.zeros(1000)
for i in range(100):
    arr = arr + compute_update(i)  # 创建新对象，内存浪费
```

<br>

## Python 与 C/C++ 的内存模型差异

Python 采用基于引用的对象模型，与 C/C++ 的值语义有根本性差异。

### C/C++ 的值语义

在 C/C++ 中，变量直接存储值，赋值操作复制值：

```cpp
int a = 10;
int b = a;  // b 是 a 的副本，独立存储
b = 20;     // 修改 b 不影响 a
```

### Python 的引用语义

在 Python 中，变量存储的是对象的引用，赋值操作复制引用：

```python
a = [1, 2, 3]
b = a  # b 和 a 指向同一个对象
b.append(4)
print(a)  # [1, 2, 3, 4]，a 也被修改
```

### NumPy 数组的引用行为

NumPy 数组作为对象，遵循 Python 的引用语义：

```python
arr1 = np.array([1, 2, 3])
arr2 = arr1  # arr2 和 arr1 引用同一对象
arr2 += 1
print(arr1)  # [2, 3, 4]，arr1 也被修改
```

如果需要独立副本，必须显式创建：

```python
arr1 = np.array([1, 2, 3])
arr2 = arr1.copy()  # 创建新对象，复制数据
arr2 += 1
print(arr1)  # [1, 2, 3]，arr1 不变
print(arr2)  # [2, 3, 4]，arr2 独立
```

### 视图与副本的区别

NumPy 还提供了视图（view）的概念，视图共享数据缓冲区但可能有不同的形状或步长：

```python
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]  # 创建视图，共享数据
view[0] = 999
print(arr)  # [1, 999, 3, 4, 5]，原数组也被修改

copy = arr[1:4].copy()  # 创建副本，独立数据
copy[0] = 888
print(arr)  # [1, 999, 3, 4, 5]，原数组不变
```
<br>
