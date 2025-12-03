---
title: Leetcode Hot100 题解整理
date: 2025-11-28
categories: 
    - 学CS/SE
tags: 
    - Leetcode
desc: 本文档整理了 LeetCode Hot 100 的所有题目，按照算法类型分类，每道题目包含题干、解题思路、复杂度分析和 Python、C++ 代码实现。
---


本文档整理了 LeetCode Hot 100 的所有题目，按照算法类型分类，每道题目包含题干、解题思路、复杂度分析和 Python、C++ 代码实现。


## 一、数组与哈希表

### 1. 两数之和 (Two Sum)

**题目描述：**

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** `target` 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

**示例：**
```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
```

**解题思路：**

使用哈希表（字典）存储已遍历的元素及其索引。对于每个元素，计算目标值与当前元素的差值，检查该差值是否已存在于哈希表中。如果存在，则找到了两个数；否则，将当前元素及其索引存入哈希表。

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度。每个元素最多被访问一次。
- 空间复杂度：O(n)，用于存储哈希表中的元素。

**Python 解答：**
```python
def twoSum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
```

**C++ 解答：**
```cpp
#include <vector>
#include <unordered_map>

std::vector<int> twoSum(std::vector<int>& nums, int target) {
    std::unordered_map<int, int> num_map;
    for (int i = 0; i < nums.size(); ++i) {
        int complement = target - nums[i];
        if (num_map.find(complement) != num_map.end()) {
            return {num_map[complement], i};
        }
        num_map[nums[i]] = i;
    }
    return {};
}
```

<br>

### 2. 三数之和 (3Sum)

**题目描述：**

给你一个整数数组 `nums`，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k`，同时还满足 `nums[i] + nums[j] + nums[k] == 0`。请你返回所有和为 `0` 且不重复的三元组。

**注意：** 答案中不可以包含重复的三元组。

**示例：**
```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
```

**解题思路：**

1. 首先对数组进行排序
2. 固定第一个数，使用双指针在剩余部分寻找另外两个数
3. 使用双指针法：左指针指向固定数的下一个位置，右指针指向数组末尾
4. 根据三数之和与 0 的关系移动指针
5. 注意跳过重复元素以避免重复解

**复杂度分析：**
- 时间复杂度：O(n²)，其中 n 是数组的长度。排序 O(n log n) + 双指针遍历 O(n²)
- 空间复杂度：O(1)，除了存储答案的空间外，只需要常数的额外空间

**Python 解答：**
```python
def threeSum(nums):
    nums.sort()
    res = []
    n = len(nums)
    
    for i in range(n - 2):
        # 跳过重复元素
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                # 跳过重复元素
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    
    return res
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {
    std::sort(nums.begin(), nums.end());
    std::vector<std::vector<int>> res;
    int n = nums.size();
    
    for (int i = 0; i < n - 2; ++i) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        
        int left = i + 1, right = n - 1;
        while (left < right) {
            int total = nums[i] + nums[left] + nums[right];
            if (total < 0) {
                ++left;
            } else if (total > 0) {
                --right;
            } else {
                res.push_back({nums[i], nums[left], nums[right]});
                while (left < right && nums[left] == nums[left + 1]) ++left;
                while (left < right && nums[right] == nums[right - 1]) --right;
                ++left;
                --right;
            }
        }
    }
    
    return res;
}
```

<br>

### 3. 盛最多水的容器 (Container With Most Water)

**题目描述：**

给定一个长度为 `n` 的整数数组 `height`。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])`。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**示例：**
```
输入：height = [1,8,6,2,5,4,8,3,7]
输出：49
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

**解题思路：**

使用双指针法：
1. 初始化左右指针分别指向数组的两端
2. 计算当前容器的容量：`min(height[left], height[right]) * (right - left)`
3. 更新最大容量
4. 移动较短的那一侧指针（因为移动较长的一侧不会增加容量）
5. 重复直到左右指针相遇

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度。双指针最多遍历整个数组一次
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def maxArea(height):
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        h = min(height[left], height[right])
        width = right - left
        max_area = max(max_area, h * width)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int maxArea(std::vector<int>& height) {
    int left = 0, right = height.size() - 1;
    int max_area = 0;
    
    while (left < right) {
        int h = std::min(height[left], height[right]);
        int width = right - left;
        max_area = std::max(max_area, h * width);
        
        if (height[left] < height[right]) {
            ++left;
        } else {
            --right;
        }
    }
    
    return max_area;
}
```

<br>

### 4. 无重复字符的最长子串 (Longest Substring Without Repeating Characters)

**题目描述：**

给定一个字符串 `s`，请你找出其中不含有重复字符的 **最长子串** 的长度。

**示例：**
```
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**解题思路：**

使用滑动窗口（双指针）+ 哈希集合：
1. 使用两个指针表示滑动窗口的左右边界
2. 使用哈希集合记录窗口内的字符
3. 右指针不断向右移动，如果遇到重复字符，移动左指针直到窗口内无重复字符
4. 在移动过程中更新最大长度

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是字符串的长度。每个字符最多被访问两次（左指针和右指针各一次）
- 空间复杂度：O(min(n, m))，其中 m 是字符集的大小。哈希集合最多存储 min(n, m) 个字符

**Python 解答：**
```python
def lengthOfLongestSubstring(s):
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # 如果遇到重复字符，移动左指针
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

**C++ 解答：**
```cpp
#include <string>
#include <unordered_set>

int lengthOfLongestSubstring(std::string s) {
    std::unordered_set<char> char_set;
    int left = 0, max_length = 0;
    
    for (int right = 0; right < s.size(); ++right) {
        while (char_set.find(s[right]) != char_set.end()) {
            char_set.erase(s[left]);
            ++left;
        }
        char_set.insert(s[right]);
        max_length = std::max(max_length, right - left + 1);
    }
    
    return max_length;
}
```

<br>

### 5. 最长回文子串 (Longest Palindromic Substring)

**题目描述：**

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。

**示例：**
```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

**解题思路：**

使用中心扩展法：
1. 回文串可能是奇数长度（以某个字符为中心）或偶数长度（以两个字符的中间为中心）
2. 对于每个可能的中心位置，向两边扩展，找到最长的回文子串
3. 比较所有找到的回文子串，返回最长的

**复杂度分析：**
- 时间复杂度：O(n²)，其中 n 是字符串的长度。对于每个中心位置，最多需要扩展 O(n) 次
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def longestPalindrome(s):
    def expandAroundCenter(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    start = end = 0
    
    for i in range(len(s)):
        # 奇数长度的回文串
        len1 = expandAroundCenter(i, i)
        # 偶数长度的回文串
        len2 = expandAroundCenter(i, i + 1)
        max_len = max(len1, len2)
        
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end + 1]
```

**C++ 解答：**
```cpp
#include <string>

std::string longestPalindrome(std::string s) {
    auto expandAroundCenter = [&](int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            --left;
            ++right;
        }
        return right - left - 1;
    };
    
    int start = 0, end = 0;
    
    for (int i = 0; i < s.size(); ++i) {
        int len1 = expandAroundCenter(i, i);
        int len2 = expandAroundCenter(i, i + 1);
        int max_len = std::max(len1, len2);
        
        if (max_len > end - start) {
            start = i - (max_len - 1) / 2;
            end = i + max_len / 2;
        }
    }
    
    return s.substr(start, end - start + 1);
}
```

<br>

### 6. 合并两个有序数组 (Merge Sorted Array)

**题目描述：**

给你两个按 **非递减顺序** 排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n`，分别表示 `nums1` 和 `nums2` 中元素的数目。

请你 **合并** `nums2` 到 `nums1` 中，使合并后的数组同样按 **非递减顺序** 排列。

**注意：** 最终，合并后数组不应由函数返回，而是存储在数组 `nums1` 中。为了应对这种情况，`nums1` 的初始长度为 `m + n`，其中前 `m` 个元素表示应合并的元素，后 `n` 个元素为 `0`，应忽略。`nums2` 的长度为 `n`。

**示例：**
```
输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
解释：需要合并 [1,2,3] 和 [2,5,6] 。
合并结果是 [1,2,2,3,5,6] ，其中斜体加粗标注的为 nums1 中的元素。
```

**解题思路：**

使用双指针从后往前合并：
1. 因为 `nums1` 后面有足够的空间，所以从后往前填充
2. 使用三个指针：`i` 指向 `nums1` 的有效元素末尾，`j` 指向 `nums2` 的末尾，`k` 指向合并后的数组末尾
3. 比较 `nums1[i]` 和 `nums2[j]`，将较大的元素放到 `nums1[k]`
4. 如果 `nums2` 还有剩余元素，需要全部复制到 `nums1` 的前面

**复杂度分析：**
- 时间复杂度：O(m + n)，其中 m 和 n 分别是两个数组的长度
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    
    # 如果 nums2 还有剩余元素
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1
```

**C++ 解答：**
```cpp
#include <vector>

void merge(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n) {
    int i = m - 1, j = n - 1, k = m + n - 1;
    
    while (i >= 0 && j >= 0) {
        if (nums1[i] > nums2[j]) {
            nums1[k] = nums1[i];
            --i;
        } else {
            nums1[k] = nums2[j];
            --j;
        }
        --k;
    }
    
    while (j >= 0) {
        nums1[k] = nums2[j];
        --j;
        --k;
    }
}
```

<br>

### 7. 移动零 (Move Zeroes)

**题目描述：**

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**注意** 必须在不复制数组的情况下原地对数组进行操作。

**示例：**
```
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
```

**解题思路：**

使用双指针：
1. 使用一个指针 `left` 指向当前应该放置非零元素的位置
2. 遍历数组，遇到非零元素就将其放到 `left` 位置，然后 `left++`
3. 遍历结束后，将 `left` 之后的所有位置置为 0

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def moveZeroes(nums):
    left = 0
    
    # 将所有非零元素移到前面
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
```

**C++ 解答：**
```cpp
#include <vector>

void moveZeroes(std::vector<int>& nums) {
    int left = 0;
    
    for (int right = 0; right < nums.size(); ++right) {
        if (nums[right] != 0) {
            std::swap(nums[left], nums[right]);
            ++left;
        }
    }
}
```

<br>

### 8. 找到所有数组中消失的数字 (Find All Numbers Disappeared in an Array)

**题目描述：**

给你一个含 `n` 个整数的数组 `nums`，其中 `nums[i]` 在区间 `[1, n]` 内。请你找出所有在 `[1, n]` 范围内但没有出现在 `nums` 中的数字，并以数组的形式返回结果。

**示例：**
```
输入：nums = [4,3,2,7,8,2,3,1]
输出：[5,6]
```

**解题思路：**

利用数组本身作为哈希表：
1. 对于每个数字 `nums[i]`，将 `nums[abs(nums[i]) - 1]` 标记为负数
2. 如果某个位置的数字已经是负数，说明该数字出现过
3. 最后遍历数组，如果某个位置的数字是正数，说明该位置对应的数字没有出现过

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度
- 空间复杂度：O(1)，除了返回数组外，只需要常数的额外空间

**Python 解答：**
```python
def findDisappearedNumbers(nums):
    # 标记出现过的数字
    for num in nums:
        index = abs(num) - 1
        if nums[index] > 0:
            nums[index] = -nums[index]
    
    # 找出未出现的数字
    result = []
    for i in range(len(nums)):
        if nums[i] > 0:
            result.append(i + 1)
    
    return result
```

**C++ 解答：**
```cpp
#include <vector>

std::vector<int> findDisappearedNumbers(std::vector<int>& nums) {
    for (int num : nums) {
        int index = abs(num) - 1;
        if (nums[index] > 0) {
            nums[index] = -nums[index];
        }
    }
    
    std::vector<int> result;
    for (int i = 0; i < nums.size(); ++i) {
        if (nums[i] > 0) {
            result.push_back(i + 1);
        }
    }
    
    return result;
}
```

<br>

### 9. 除自身以外数组的乘积 (Product of Array Except Self)

**题目描述：**

给你一个整数数组 `nums`，返回 **数组 `answer`**，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

题目数据 **保证** 数组 `nums` 之中任意元素的全部前缀元素和后缀的乘积都在 **32 位整数** 范围内。

请 **不要使用除法**，且在 `O(n)` 时间复杂度内完成此题。

**示例：**
```
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
```

**解题思路：**

使用左右乘积列表：
1. 创建两个数组：`left` 存储每个元素左侧所有元素的乘积，`right` 存储每个元素右侧所有元素的乘积
2. 对于位置 `i`，`answer[i] = left[i] * right[i]`
3. 可以优化空间复杂度：先计算 `left` 数组，然后从右往左遍历，用一个变量存储右侧乘积

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度
- 空间复杂度：O(1)，除了返回数组外，只需要常数的额外空间（优化后）

**Python 解答：**
```python
def productExceptSelf(nums):
    n = len(nums)
    answer = [1] * n
    
    # 计算左侧乘积
    for i in range(1, n):
        answer[i] = answer[i - 1] * nums[i - 1]
    
    # 计算右侧乘积并同时更新答案
    right = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= right
        right *= nums[i]
    
    return answer
```

**C++ 解答：**
```cpp
#include <vector>

std::vector<int> productExceptSelf(std::vector<int>& nums) {
    int n = nums.size();
    std::vector<int> answer(n, 1);
    
    // 计算左侧乘积
    for (int i = 1; i < n; ++i) {
        answer[i] = answer[i - 1] * nums[i - 1];
    }
    
    // 计算右侧乘积并同时更新答案
    int right = 1;
    for (int i = n - 1; i >= 0; --i) {
        answer[i] *= right;
        right *= nums[i];
    }
    
    return answer;
}
```

<br>

### 10. 旋转图像 (Rotate Image)

**题目描述：**

给定一个 `n × n` 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 **原地** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

**示例：**
```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```

**解题思路：**

方法一：先转置再翻转每一行
1. 先对矩阵进行转置（行列互换）
2. 然后翻转每一行

方法二：直接旋转
1. 对于位置 `(i, j)`，旋转后的位置是 `(j, n-1-i)`
2. 每次旋转四个位置：`(i, j) -> (j, n-1-i) -> (n-1-i, n-1-j) -> (n-1-j, i) -> (i, j)`

**复杂度分析：**
- 时间复杂度：O(n²)，其中 n 是矩阵的边长
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def rotate(matrix):
    n = len(matrix)
    
    # 转置矩阵
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # 翻转每一行
    for i in range(n):
        matrix[i].reverse()
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

void rotate(std::vector<std::vector<int>>& matrix) {
    int n = matrix.size();
    
    // 转置矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }
    
    // 翻转每一行
    for (int i = 0; i < n; ++i) {
        std::reverse(matrix[i].begin(), matrix[i].end());
    }
}
```


## 二、链表

### 11. 反转链表 (Reverse Linked List)

**题目描述：**

给你单链表的头节点 `head`，请你反转链表，并返回反转后的链表。

**示例：**
```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

**解题思路：**

使用迭代法：
1. 使用三个指针：`prev`（前一个节点）、`curr`（当前节点）、`next`（下一个节点）
2. 遍历链表，将当前节点的 `next` 指向前一个节点
3. 然后移动三个指针继续遍历

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是链表的长度
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head):
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    return prev
```

**C++ 解答：**
```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while (curr) {
        ListNode* next_temp = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next_temp;
    }
    
    return prev;
}
```

<br>

### 12. 合并两个有序链表 (Merge Two Sorted Lists)

**题目描述：**

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例：**
```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**解题思路：**

使用双指针（迭代法）：
1. 创建一个虚拟头节点 `dummy`，用于简化边界处理
2. 使用指针 `curr` 指向当前合并后的链表的末尾
3. 比较两个链表的当前节点，将较小的节点连接到 `curr` 后面
4. 移动指针继续比较，直到其中一个链表为空
5. 将剩余的链表连接到结果链表的末尾

**复杂度分析：**
- 时间复杂度：O(n + m)，其中 n 和 m 分别是两个链表的长度
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def mergeTwoLists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 else l2
    
    return dummy.next
```

**C++ 解答：**
```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            curr->next = l1;
            l1 = l1->next;
        } else {
            curr->next = l2;
            l2 = l2->next;
        }
        curr = curr->next;
    }
    
    curr->next = l1 ? l1 : l2;
    
    return dummy->next;
}
```

<br>

### 13. 环形链表 (Linked List Cycle)

**题目描述：**

给你一个链表的头节点 `head`，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：`pos` 不作为参数进行传递。仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 `true`。否则，返回 `false`。

**示例：**
```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

**解题思路：**

使用快慢指针（Floyd 判圈算法）：
1. 使用两个指针：`slow` 每次移动一步，`fast` 每次移动两步
2. 如果链表中存在环，快慢指针最终会相遇
3. 如果快指针到达链表末尾（`nullptr`），说明没有环

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是链表中节点的数量
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def hasCycle(head):
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True
```

**C++ 解答：**
```cpp
bool hasCycle(ListNode* head) {
    if (!head || !head->next) {
        return false;
    }
    
    ListNode* slow = head;
    ListNode* fast = head->next;
    
    while (slow != fast) {
        if (!fast || !fast->next) {
            return false;
        }
        slow = slow->next;
        fast = fast->next->next;
    }
    
    return true;
}
```

<br>

### 14. 环形链表 II (Linked List Cycle II)

**题目描述：**

给定一个链表的头节点 `head`，返回链表开始入环的第一个节点。如果链表无环，则返回 `null`。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 `pos` 是 `-1`，则在该链表中没有环。

**示例：**
```
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

**解题思路：**

使用快慢指针找到相遇点，然后找到环的入口：
1. 使用快慢指针找到相遇点
2. 设从起点到环入口的距离为 `a`，从环入口到相遇点的距离为 `b`，从相遇点到环入口的距离为 `c`
3. 当快慢指针相遇时，慢指针走了 `a + b`，快指针走了 `a + b + n(b + c)`
4. 由于快指针速度是慢指针的 2 倍：`2(a + b) = a + b + n(b + c)`，化简得 `a = (n-1)(b+c) + c`
5. 这意味着从起点到环入口的距离等于从相遇点到环入口的距离（加上整数倍的环长）
6. 因此，将一个指针重置到起点，两个指针同时移动，相遇点就是环的入口

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是链表中节点的数量
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def detectCycle(head):
    if not head or not head.next:
        return None
    
    slow = fast = head
    
    # 找到相遇点
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # 没有环
    
    # 找到环的入口
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

**C++ 解答：**
```cpp
ListNode* detectCycle(ListNode* head) {
    if (!head || !head->next) {
        return nullptr;
    }
    
    ListNode* slow = head;
    ListNode* fast = head;
    
    // 找到相遇点
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            break;
        }
    }
    
    if (!fast || !fast->next) {
        return nullptr;  // 没有环
    }
    
    // 找到环的入口
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    
    return slow;
}
```

<br>

### 15. 相交链表 (Intersection of Two Linked Lists)

**题目描述：**

给你两个单链表的头节点 `headA` 和 `headB`，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null`。

**示例：**
```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
```

**解题思路：**

使用双指针法：
1. 创建两个指针 `pA` 和 `pB`，分别指向两个链表的头节点
2. 同时移动两个指针，当其中一个指针到达链表末尾时，将其重置到另一个链表的头节点
3. 如果两个链表相交，两个指针最终会在相交节点相遇
4. 如果两个链表不相交，两个指针最终都会到达 `nullptr`

**复杂度分析：**
- 时间复杂度：O(m + n)，其中 m 和 n 分别是两个链表的长度
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def getIntersectionNode(headA, headB):
    if not headA or not headB:
        return None
    
    pA, pB = headA, headB
    
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    
    return pA
```

**C++ 解答：**
```cpp
ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
    if (!headA || !headB) {
        return nullptr;
    }
    
    ListNode* pA = headA;
    ListNode* pB = headB;
    
    while (pA != pB) {
        pA = pA ? pA->next : headB;
        pB = pB ? pB->next : headA;
    }
    
    return pA;
}
```

<br>

### 16. 删除链表的倒数第 N 个结点 (Remove Nth Node From End of List)

**题目描述：**

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

**示例：**
```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

**解题思路：**

使用双指针法：
1. 创建一个虚拟头节点 `dummy`，简化边界处理
2. 使用两个指针 `first` 和 `second`，`first` 先移动 `n + 1` 步
3. 然后同时移动 `first` 和 `second`，直到 `first` 到达链表末尾
4. 此时 `second` 指向倒数第 `n + 1` 个节点，删除其下一个节点即可

**复杂度分析：**
- 时间复杂度：O(L)，其中 L 是链表的长度
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0)
    dummy.next = head
    
    first = dummy
    second = dummy
    
    # first 先移动 n + 1 步
    for _ in range(n + 1):
        first = first.next
    
    # 同时移动 first 和 second
    while first:
        first = first.next
        second = second.next
    
    # 删除倒数第 n 个节点
    second.next = second.next.next
    
    return dummy.next
```

**C++ 解答：**
```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    
    ListNode* first = dummy;
    ListNode* second = dummy;
    
    // first 先移动 n + 1 步
    for (int i = 0; i <= n; ++i) {
        first = first->next;
    }
    
    // 同时移动 first 和 second
    while (first) {
        first = first->next;
        second = second->next;
    }
    
    // 删除倒数第 n 个节点
    second->next = second->next->next;
    
    return dummy->next;
}
```

<br>

### 17. 两数相加 (Add Two Numbers)

**题目描述：**

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**示例：**
```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

**解题思路：**

模拟加法过程：
1. 同时遍历两个链表，逐位相加
2. 使用变量 `carry` 记录进位
3. 对于每一位，计算 `sum = l1.val + l2.val + carry`
4. 新节点的值为 `sum % 10`，进位为 `sum // 10`
5. 如果其中一个链表遍历完了，继续处理另一个链表
6. 最后如果还有进位，需要添加一个新节点

**复杂度分析：**
- 时间复杂度：O(max(m, n))，其中 m 和 n 分别是两个链表的长度
- 空间复杂度：O(max(m, n))，新链表的长度最多为 max(m, n) + 1

**Python 解答：**
```python
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        curr.next = ListNode(total % 10)
        curr = curr.next
        
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next
```

**C++ 解答：**
```cpp
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    int carry = 0;
    
    while (l1 || l2 || carry) {
        int val1 = l1 ? l1->val : 0;
        int val2 = l2 ? l2->val : 0;
        
        int total = val1 + val2 + carry;
        carry = total / 10;
        curr->next = new ListNode(total % 10);
        curr = curr->next;
        
        l1 = l1 ? l1->next : nullptr;
        l2 = l2 ? l2->next : nullptr;
    }
    
    return dummy->next;
}
```

<br>

### 18. 合并 K 个升序链表 (Merge k Sorted Lists)

**题目描述：**

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

**示例：**
```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

**解题思路：**

方法一：分治法
1. 将 k 个链表两两合并，直到只剩下一个链表
2. 使用分治的思想，递归地合并链表

方法二：优先队列（最小堆）
1. 将所有链表的头节点放入优先队列
2. 每次取出最小的节点，连接到结果链表
3. 如果该节点还有下一个节点，将其放入优先队列
4. 重复直到优先队列为空

**复杂度分析：**
- 时间复杂度：O(n log k)，其中 n 是所有链表中节点的总数，k 是链表的数量
- 空间复杂度：O(1)（分治法）或 O(k)（优先队列）

**Python 解答（分治法）：**
```python
def mergeKLists(lists):
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    
    mid = len(lists) // 2
    left = mergeKLists(lists[:mid])
    right = mergeKLists(lists[mid:])
    
    return mergeTwoLists(left, right)

def mergeTwoLists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 else l2
    return dummy.next
```

**C++ 解答（优先队列）：**
```cpp
#include <vector>
#include <queue>

struct Compare {
    bool operator()(ListNode* a, ListNode* b) {
        return a->val > b->val;
    }
};

ListNode* mergeKLists(std::vector<ListNode*>& lists) {
    std::priority_queue<ListNode*, std::vector<ListNode*>, Compare> pq;
    
    for (ListNode* list : lists) {
        if (list) {
            pq.push(list);
        }
    }
    
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    
    while (!pq.empty()) {
        ListNode* node = pq.top();
        pq.pop();
        curr->next = node;
        curr = curr->next;
        
        if (node->next) {
            pq.push(node->next);
        }
    }
    
    return dummy->next;
}
```


## 三、字符串

### 19. 有效的括号 (Valid Parentheses)

**题目描述：**

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s`，判断字符串是否有效。

有效字符串需满足：
1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

**示例：**
```
输入：s = "()[]{}"
输出：true
```

**解题思路：**

使用栈：
1. 遍历字符串，遇到左括号就入栈
2. 遇到右括号时，检查栈顶是否是对应的左括号
3. 如果是，弹出栈顶；如果不是，返回 false
4. 最后检查栈是否为空

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是字符串的长度
- 空间复杂度：O(n)，最坏情况下栈中存储所有左括号

**Python 解答：**
```python
def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack
```

**C++ 解答：**
```cpp
#include <string>
#include <stack>
#include <unordered_map>

bool isValid(std::string s) {
    std::stack<char> stack;
    std::unordered_map<char, char> mapping = {
        {')', '('},
        {'}', '{'},
        {']', '['}
    };
    
    for (char c : s) {
        if (mapping.find(c) != mapping.end()) {
            if (stack.empty() || stack.top() != mapping[c]) {
                return false;
            }
            stack.pop();
        } else {
            stack.push(c);
        }
    }
    
    return stack.empty();
}
```

<br>

### 20. 字母异位词分组 (Group Anagrams)

**题目描述：**

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

**示例：**
```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

**解题思路：**

使用哈希表：
1. 对于每个字符串，将其排序后的结果作为键
2. 将具有相同键的字符串分组
3. 返回所有分组

**复杂度分析：**
- 时间复杂度：O(nk log k)，其中 n 是字符串数组的长度，k 是字符串的最大长度
- 空间复杂度：O(nk)，用于存储哈希表和结果

**Python 解答：**
```python
def groupAnagrams(strs):
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {
    std::unordered_map<std::string, std::vector<std::string>> groups;
    
    for (const std::string& s : strs) {
        std::string key = s;
        std::sort(key.begin(), key.end());
        groups[key].push_back(s);
    }
    
    std::vector<std::vector<std::string>> result;
    for (auto& pair : groups) {
        result.push_back(pair.second);
    }
    
    return result;
}
```


## 四、动态规划

### 21. 爬楼梯 (Climbing Stairs)

**题目描述：**

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶？

**示例：**
```
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
```

**解题思路：**

动态规划：
1. 定义 `dp[i]` 为到达第 i 阶的方法数
2. 状态转移方程：`dp[i] = dp[i-1] + dp[i-2]`
3. 初始条件：`dp[1] = 1, dp[2] = 2`
4. 可以优化空间复杂度，只使用两个变量

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是楼梯的阶数
- 空间复杂度：O(1)，只需要常数的额外空间（优化后）

**Python 解答：**
```python
def climbStairs(n):
    if n <= 2:
        return n
    
    a, b = 1, 2
    for i in range(3, n + 1):
        a, b = b, a + b
    
    return b
```

**C++ 解答：**
```cpp
int climbStairs(int n) {
    if (n <= 2) {
        return n;
    }
    
    int a = 1, b = 2;
    for (int i = 3; i <= n; ++i) {
        int temp = b;
        b = a + b;
        a = temp;
    }
    
    return b;
}
```

<br>

### 22. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)

**题目描述：**

给定一个数组 `prices`，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0`。

**示例：**
```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

**解题思路：**

一次遍历：
1. 维护两个变量：`min_price`（到目前为止的最低价格）和 `max_profit`（最大利润）
2. 遍历数组，对于每一天：
   - 更新最低价格：`min_price = min(min_price, prices[i])`
   - 更新最大利润：`max_profit = max(max_profit, prices[i] - min_price)`

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度
- 空间复杂度：O(1)，只需要常数的额外空间

**Python 解答：**
```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    
    return max_profit
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int maxProfit(std::vector<int>& prices) {
    int min_price = INT_MAX;
    int max_profit = 0;
    
    for (int price : prices) {
        min_price = std::min(min_price, price);
        max_profit = std::max(max_profit, price - min_price);
    }
    
    return max_profit;
}
```

<br>

### 23. 最大子数组和 (Maximum Subarray)

**题目描述：**

给你一个整数数组 `nums`，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组** 是数组中的一个连续部分。

**示例：**
```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

**解题思路：**

动态规划（Kadane 算法）：
1. 定义 `dp[i]` 为以第 i 个元素结尾的最大子数组和
2. 状态转移方程：`dp[i] = max(nums[i], dp[i-1] + nums[i])`
3. 可以优化空间复杂度，只使用一个变量

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度
- 空间复杂度：O(1)，只需要常数的额外空间（优化后）

**Python 解答：**
```python
def maxSubArray(nums):
    max_sum = current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int maxSubArray(std::vector<int>& nums) {
    int max_sum = nums[0];
    int current_sum = nums[0];
    
    for (int i = 1; i < nums.size(); ++i) {
        current_sum = std::max(nums[i], current_sum + nums[i]);
        max_sum = std::max(max_sum, current_sum);
    }
    
    return max_sum;
}
```

<br>

### 24. 打家劫舍 (House Robber)

**题目描述：**

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下**，一夜之内能够偷窃到的最高金额。

**示例：**
```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

**解题思路：**

动态规划：
1. 定义 `dp[i]` 为偷窃前 i 间房屋能获得的最大金额
2. 状态转移方程：`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
   - 不偷第 i 间：`dp[i-1]`
   - 偷第 i 间：`dp[i-2] + nums[i]`
3. 可以优化空间复杂度，只使用两个变量

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度
- 空间复杂度：O(1)，只需要常数的额外空间（优化后）

**Python 解答：**
```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current
    
    return prev1
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int rob(std::vector<int>& nums) {
    if (nums.empty()) {
        return 0;
    }
    if (nums.size() == 1) {
        return nums[0];
    }
    
    int prev2 = nums[0];
    int prev1 = std::max(nums[0], nums[1]);
    
    for (int i = 2; i < nums.size(); ++i) {
        int current = std::max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = current;
    }
    
    return prev1;
}
```

<br>

### 25. 完全平方数 (Perfect Squares)

**题目描述：**

给你一个整数 `n`，返回 **和为 `n` 的完全平方数的最少数量**。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

**示例：**
```
输入：n = 12
输出：3
解释：12 = 4 + 4 + 4
```

**解题思路：**

动态规划：
1. 定义 `dp[i]` 为组成数字 i 的完全平方数的最少数量
2. 状态转移方程：`dp[i] = min(dp[i], dp[i - j*j] + 1)`，其中 `j*j <= i`
3. 初始条件：`dp[0] = 0`

**复杂度分析：**
- 时间复杂度：O(n√n)，其中 n 是给定的整数
- 空间复杂度：O(n)，用于存储 dp 数组

**Python 解答：**
```python
def numSquares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    
    return dp[n]
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>
#include <climits>

int numSquares(int n) {
    std::vector<int> dp(n + 1, INT_MAX);
    dp[0] = 0;
    
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j * j <= i; ++j) {
            dp[i] = std::min(dp[i], dp[i - j * j] + 1);
        }
    }
    
    return dp[n];
}
```

<br>

### 26. 单词拆分 (Word Break)

**题目描述：**

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。如果可以利用字典中出现的单词拼接出 `s` 则返回 `true`。

**注意：** 不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

**示例：**
```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

**解题思路：**

动态规划：
1. 定义 `dp[i]` 表示字符串 `s` 的前 i 个字符是否可以被字典中的单词拼接
2. 状态转移方程：`dp[i] = dp[j] && s[j:i] in wordDict`，其中 `0 <= j < i`
3. 初始条件：`dp[0] = true`（空字符串可以被拼接）

**复杂度分析：**
- 时间复杂度：O(n²)，其中 n 是字符串的长度
- 空间复杂度：O(n)，用于存储 dp 数组

**Python 解答：**
```python
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]
```

**C++ 解答：**
```cpp
#include <string>
#include <vector>
#include <unordered_set>

bool wordBreak(std::string s, std::vector<std::string>& wordDict) {
    std::unordered_set<std::string> word_set(wordDict.begin(), wordDict.end());
    int n = s.length();
    std::vector<bool> dp(n + 1, false);
    dp[0] = true;
    
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (dp[j] && word_set.find(s.substr(j, i - j)) != word_set.end()) {
                dp[i] = true;
                break;
            }
        }
    }
    
    return dp[n];
}
```

<br>

### 27. 最长递增子序列 (Longest Increasing Subsequence)

**题目描述：**

给你一个整数数组 `nums`，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例：**
```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

**解题思路：**

动态规划 + 二分查找：
1. 方法一：动态规划
   - 定义 `dp[i]` 为以 `nums[i]` 结尾的最长递增子序列的长度
   - 状态转移方程：`dp[i] = max(dp[j]) + 1`，其中 `0 <= j < i` 且 `nums[j] < nums[i]`
   - 时间复杂度：O(n²)

2. 方法二：贪心 + 二分查找（优化）
   - 维护一个数组 `tails`，其中 `tails[i]` 表示长度为 i+1 的递增子序列的最小末尾元素
   - 使用二分查找找到第一个大于等于当前元素的位置
   - 时间复杂度：O(n log n)

**复杂度分析：**
- 时间复杂度：O(n log n)（优化方法），O(n²)（基础方法）
- 空间复杂度：O(n)

**Python 解答（优化方法）：**
```python
def lengthOfLIS(nums):
    tails = []
    
    for num in nums:
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)
```

**C++ 解答（优化方法）：**
```cpp
#include <vector>
#include <algorithm>

int lengthOfLIS(std::vector<int>& nums) {
    std::vector<int> tails;
    
    for (int num : nums) {
        auto it = std::lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }
    
    return tails.size();
}
```

<br>

### 28. 零钱兑换 (Coin Change)

**题目描述：**

给你一个整数数组 `coins`，表示不同面额的硬币；以及一个整数 `amount`，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数**。如果没有任何一种硬币组合能组成总金额，返回 `-1`。

你可以认为每种硬币的数量是无限的。

**示例：**
```
输入：coins = [1, 2, 5], amount = 11
输出：3
解释：11 = 5 + 5 + 1
```

**解题思路：**

动态规划：
1. 定义 `dp[i]` 为凑成金额 i 所需的最少硬币数
2. 状态转移方程：`dp[i] = min(dp[i], dp[i - coin] + 1)`，其中 `coin` 是硬币面额
3. 初始条件：`dp[0] = 0`，其他为 `inf`

**复杂度分析：**
- 时间复杂度：O(S × n)，其中 S 是金额，n 是硬币种类数
- 空间复杂度：O(S)，用于存储 dp 数组

**Python 解答：**
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>
#include <climits>

int coinChange(std::vector<int>& coins, int amount) {
    std::vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; ++i) {
        for (int coin : coins) {
            if (i >= coin && dp[i - coin] != INT_MAX) {
                dp[i] = std::min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    
    return dp[amount] == INT_MAX ? -1 : dp[amount];
}
```

<br>

### 29. 编辑距离 (Edit Distance)

**题目描述：**

给你两个单词 `word1` 和 `word2`，请返回将 `word1` 转换成 `word2` 所使用的最少操作数。

你可以对一个单词进行如下三种操作：
- 插入一个字符
- 删除一个字符
- 替换一个字符

**示例：**
```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**解题思路：**

动态规划：
1. 定义 `dp[i][j]` 为将 `word1` 的前 i 个字符转换为 `word2` 的前 j 个字符所需的最少操作数
2. 状态转移方程：
   - 如果 `word1[i-1] == word2[j-1]`：`dp[i][j] = dp[i-1][j-1]`
   - 否则：`dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1`
     - `dp[i-1][j]`：删除 word1[i-1]
     - `dp[i][j-1]`：在 word1 中插入 word2[j-1]
     - `dp[i-1][j-1]`：替换 word1[i-1] 为 word2[j-1]

**复杂度分析：**
- 时间复杂度：O(m × n)，其中 m 和 n 分别是两个字符串的长度
- 空间复杂度：O(m × n)，可以优化到 O(min(m, n))

**Python 解答：**
```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充 dp 数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n]
```

**C++ 解答：**
```cpp
#include <string>
#include <vector>
#include <algorithm>

int minDistance(std::string word1, std::string word2) {
    int m = word1.length(), n = word2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    
    // 初始化
    for (int i = 0; i <= m; ++i) {
        dp[i][0] = i;
    }
    for (int j = 0; j <= n; ++j) {
        dp[0][j] = j;
    }
    
    // 填充 dp 数组
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1;
            }
        }
    }
    
    return dp[m][n];
}
```

<br>

### 30. 最长公共子序列 (Longest Common Subsequence)

**题目描述：**

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列**，返回 `0`。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

**示例：**
```
输入：text1 = "abcde", text2 = "ace"
输出：3
解释：最长公共子序列是 "ace" ，它的长度为 3 。
```

**解题思路：**

动态规划：
1. 定义 `dp[i][j]` 为 `text1` 的前 i 个字符和 `text2` 的前 j 个字符的最长公共子序列长度
2. 状态转移方程：
   - 如果 `text1[i-1] == text2[j-1]`：`dp[i][j] = dp[i-1][j-1] + 1`
   - 否则：`dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

**复杂度分析：**
- 时间复杂度：O(m × n)，其中 m 和 n 分别是两个字符串的长度
- 空间复杂度：O(m × n)，可以优化到 O(min(m, n))

**Python 解答：**
```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

**C++ 解答：**
```cpp
#include <string>
#include <vector>
#include <algorithm>

int longestCommonSubsequence(std::string text1, std::string text2) {
    int m = text1.length(), n = text2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    return dp[m][n];
}
```


## 五、二叉树

### 31. 二叉树的最大深度 (Maximum Depth of Binary Tree)

**题目描述：**

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

**说明：** 叶子节点是指没有子节点的节点。

**示例：**
```
给定二叉树 [3,9,20,null,null,15,7]，
    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。
```

**解题思路：**

递归法：
1. 如果根节点为空，返回 0
2. 否则，返回 `max(左子树深度, 右子树深度) + 1`

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是二叉树的节点数
- 空间复杂度：O(h)，其中 h 是二叉树的高度（递归栈的深度）

**Python 解答：**
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root):
    if not root:
        return 0
    return max(maxDepth(root.left), maxDepth(root.right)) + 1
```

**C++ 解答：**
```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

int maxDepth(TreeNode* root) {
    if (!root) {
        return 0;
    }
    return std::max(maxDepth(root->left), maxDepth(root->right)) + 1;
}
```

<br>

### 32. 对称二叉树 (Symmetric Tree)

**题目描述：**

给你一个二叉树的根节点 `root`，检查它是否轴对称。

**示例：**
```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**解题思路：**

递归法：
1. 定义辅助函数 `isMirror(left, right)` 判断两个子树是否镜像对称
2. 两个子树镜像对称的条件：
   - 两个根节点的值相等
   - 左子树的左子树与右子树的右子树镜像对称
   - 左子树的右子树与右子树的左子树镜像对称

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是二叉树的节点数
- 空间复杂度：O(h)，其中 h 是二叉树的高度

**Python 解答：**
```python
def isSymmetric(root):
    def isMirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and 
                isMirror(left.left, right.right) and 
                isMirror(left.right, right.left))
    
    if not root:
        return True
    return isMirror(root.left, root.right)
```

**C++ 解答：**
```cpp
bool isSymmetric(TreeNode* root) {
    if (!root) {
        return true;
    }
    return isMirror(root->left, root->right);
}

bool isMirror(TreeNode* left, TreeNode* right) {
    if (!left && !right) {
        return true;
    }
    if (!left || !right) {
        return false;
    }
    return (left->val == right->val &&
            isMirror(left->left, right->right) &&
            isMirror(left->right, right->left));
}
```

<br>

### 33. 二叉树的层序遍历 (Binary Tree Level Order Traversal)

**题目描述：**

给你二叉树的根节点 `root`，返回其节点值的 **层序遍历**。（即逐层地，从左到右访问所有节点）。

**示例：**
```
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
```

**解题思路：**

使用队列（BFS）：
1. 将根节点入队
2. 当队列不为空时：
   - 记录当前层的节点数
   - 遍历当前层的所有节点，将它们的值加入结果，并将它们的子节点入队
   - 将当前层的结果加入最终结果

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是二叉树的节点数
- 空间复杂度：O(n)，用于存储队列

**Python 解答：**
```python
from collections import deque

def levelOrder(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

**C++ 解答：**
```cpp
#include <vector>
#include <queue>

std::vector<std::vector<int>> levelOrder(TreeNode* root) {
    std::vector<std::vector<int>> result;
    if (!root) {
        return result;
    }
    
    std::queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int level_size = q.size();
        std::vector<int> level;
        
        for (int i = 0; i < level_size; ++i) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            
            if (node->left) {
                q.push(node->left);
            }
            if (node->right) {
                q.push(node->right);
            }
        }
        
        result.push_back(level);
    }
    
    return result;
}
```

<br>

### 34. 将有序数组转换为二叉搜索树 (Convert Sorted Array to Binary Search Tree)

**题目描述：**

给你一个整数数组 `nums`，其中元素已经按 **升序** 排列，请你将其转换为一棵 **高度平衡** 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1」的二叉树。

**示例：**
```
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：
```

**解题思路：**

递归法：
1. 选择数组中间的元素作为根节点
2. 递归构建左子树（左半部分）和右子树（右半部分）
3. 这样可以保证树的高度平衡

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是数组的长度
- 空间复杂度：O(log n)，递归栈的深度

**Python 解答：**
```python
def sortedArrayToBST(nums):
    def build(left, right):
        if left > right:
            return None
        
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        root.left = build(left, mid - 1)
        root.right = build(mid + 1, right)
        return root
    
    return build(0, len(nums) - 1)
```

**C++ 解答：**
```cpp
TreeNode* sortedArrayToBST(std::vector<int>& nums) {
    return build(nums, 0, nums.size() - 1);
}

TreeNode* build(std::vector<int>& nums, int left, int right) {
    if (left > right) {
        return nullptr;
    }
    
    int mid = (left + right) / 2;
    TreeNode* root = new TreeNode(nums[mid]);
    root->left = build(nums, left, mid - 1);
    root->right = build(nums, mid + 1, right);
    return root;
}
```

<br>

### 35. 验证二叉搜索树 (Validate Binary Search Tree)

**题目描述：**

给你一个二叉树的根节点 `root`，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：
- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

**示例：**
```
输入：root = [2,1,3]
输出：true
```

**解题思路：**

中序遍历法：
1. 二叉搜索树的中序遍历结果是严格递增的
2. 进行中序遍历，检查当前节点的值是否大于前一个节点的值

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是二叉树的节点数
- 空间复杂度：O(h)，其中 h 是二叉树的高度

**Python 解答：**
```python
def isValidBST(root):
    prev = None
    
    def inorder(node):
        nonlocal prev
        if not node:
            return True
        
        if not inorder(node.left):
            return False
        
        if prev is not None and node.val <= prev:
            return False
        prev = node.val
        
        return inorder(node.right)
    
    return inorder(root)
```

**C++ 解答：**
```cpp
bool isValidBST(TreeNode* root) {
    TreeNode* prev = nullptr;
    return inorder(root, prev);
}

bool inorder(TreeNode* node, TreeNode*& prev) {
    if (!node) {
        return true;
    }
    
    if (!inorder(node->left, prev)) {
        return false;
    }
    
    if (prev && node->val <= prev->val) {
        return false;
    }
    prev = node;
    
    return inorder(node->right, prev);
}
```

<br>

### 36. 二叉树的最近公共祖先 (Lowest Common Ancestor of a Binary Tree)

**题目描述：**

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

**最近公共祖先** 的定义为："对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。"

**示例：**
```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
```

**解题思路：**

递归法：
1. 如果当前节点为空或等于 p 或 q，返回当前节点
2. 递归查找左子树和右子树
3. 如果左右子树都找到了节点，说明当前节点是最近公共祖先
4. 如果只有一边找到了，返回那一边的结果

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是二叉树的节点数
- 空间复杂度：O(h)，其中 h 是二叉树的高度

**Python 解答：**
```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    
    if left and right:
        return root
    return left if left else right
```

**C++ 解答：**
```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) {
        return root;
    }
    
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    
    if (left && right) {
        return root;
    }
    return left ? left : right;
}
```

<br>

### 37. 二叉树中的最大路径和 (Binary Tree Maximum Path Sum)

**题目描述：**

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 **至多出现一次**。该路径 **至少包含一个** 节点，且不一定经过根节点。

**路径和** 是路径中各节点值的总和。

给你一个二叉树的根节点 `root`，返回其 **最大路径和**。

**示例：**
```
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```

**解题思路：**

递归法：
1. 定义辅助函数 `maxGain(node)` 返回以 node 为起点的最大路径和
2. 对于每个节点，计算：
   - 经过该节点的最大路径和 = `node.val + left_gain + right_gain`
   - 返回给父节点的最大增益 = `node.val + max(left_gain, right_gain)`
3. 在递归过程中更新全局最大路径和

**复杂度分析：**
- 时间复杂度：O(n)，其中 n 是二叉树的节点数
- 空间复杂度：O(h)，其中 h 是二叉树的高度

**Python 解答：**
```python
def maxPathSum(root):
    max_sum = float('-inf')
    
    def maxGain(node):
        nonlocal max_sum
        if not node:
            return 0
        
        left_gain = max(maxGain(node.left), 0)
        right_gain = max(maxGain(node.right), 0)
        
        current_path = node.val + left_gain + right_gain
        max_sum = max(max_sum, current_path)
        
        return node.val + max(left_gain, right_gain)
    
    maxGain(root)
    return max_sum
```

**C++ 解答：**
```cpp
int maxPathSum(TreeNode* root) {
    int max_sum = INT_MIN;
    maxGain(root, max_sum);
    return max_sum;
}

int maxGain(TreeNode* node, int& max_sum) {
    if (!node) {
        return 0;
    }
    
    int left_gain = std::max(maxGain(node->left, max_sum), 0);
    int right_gain = std::max(maxGain(node->right, max_sum), 0);
    
    int current_path = node->val + left_gain + right_gain;
    max_sum = std::max(max_sum, current_path);
    
    return node->val + std::max(left_gain, right_gain);
}
```


## 二、回溯算法

### 38. 全排列 (Permutations)

**题目描述：**

给定一个不含重复数字的数组 `nums`，返回其 **所有可能的全排列**。你可以 **按任意顺序** 返回答案。

**示例：**
```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**解题思路：**

回溯算法：
1. 使用回溯法，维护一个当前排列 `path`
2. 对于每个位置，尝试所有未使用的数字
3. 选择一个数字后，递归处理下一个位置
4. 回溯时撤销选择，尝试其他可能性

**复杂度分析：**
- 时间复杂度：O(n × n!)，共有 n! 个排列，每个排列需要 O(n) 时间复制
- 空间复杂度：O(n)，递归栈深度为 n

**Python 解答：**
```python
def permute(nums):
    result = []
    
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                path.append(nums[i])
                backtrack(path, used)
                path.pop()
                used[i] = False
    
    backtrack([], [False] * len(nums))
    return result
```

**C++ 解答：**
```cpp
#include <vector>

std::vector<std::vector<int>> permute(std::vector<int>& nums) {
    std::vector<std::vector<int>> result;
    std::vector<int> path;
    std::vector<bool> used(nums.size(), false);
    
    std::function<void()> backtrack = [&]() {
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }
        
        for (int i = 0; i < nums.size(); ++i) {
            if (!used[i]) {
                used[i] = true;
                path.push_back(nums[i]);
                backtrack();
                path.pop_back();
                used[i] = false;
            }
        }
    };
    
    backtrack();
    return result;
}
```

<br>

### 39. 子集 (Subsets)

**题目描述：**

给你一个整数数组 `nums`，数组中的元素 **互不相同**。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

**示例：**
```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**解题思路：**

回溯算法：
1. 对于每个元素，有两种选择：包含或不包含
2. 从第一个元素开始，逐个决定是否加入当前子集
3. 当处理完所有元素时，将当前子集加入结果

**复杂度分析：**
- 时间复杂度：O(n × 2^n)，共有 2^n 个子集，每个子集需要 O(n) 时间复制
- 空间复杂度：O(n)，递归栈深度为 n

**Python 解答：**
```python
def subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

**C++ 解答：**
```cpp
#include <vector>

std::vector<std::vector<int>> subsets(std::vector<int>& nums) {
    std::vector<std::vector<int>> result;
    std::vector<int> path;
    
    std::function<void(int)> backtrack = [&](int start) {
        result.push_back(path);
        
        for (int i = start; i < nums.size(); ++i) {
            path.push_back(nums[i]);
            backtrack(i + 1);
            path.pop_back();
        }
    };
    
    backtrack(0);
    return result;
}
```

<br>

### 40. 组合总和 (Combination Sum)

**题目描述：**

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target`，找出 `candidates` 中可以使数字和为目标数 `target` 的 **所有** 不同组合，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取**。如果至少一个数字的被选数量不同，则两种组合是不同的。

**示例：**
```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7。注意 2 可以使用多次。
7 也是一个候选，7 = 7。
仅有这两种组合。
```

**解题思路：**

回溯算法：
1. 对数组排序，便于剪枝
2. 从第一个元素开始，尝试所有可能的组合
3. 如果当前和等于 target，加入结果
4. 如果当前和小于 target，继续递归
5. 剪枝：如果当前和大于 target，直接返回

**复杂度分析：**
- 时间复杂度：O(S)，其中 S 是所有可行解的长度之和
- 空间复杂度：O(target)，递归栈深度最多为 target

**Python 解答：**
```python
def combinationSum(candidates, target):
    result = []
    candidates.sort()
    
    def backtrack(start, path, current_sum):
        if current_sum == target:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            if current_sum + candidates[i] > target:
                break
            path.append(candidates[i])
            backtrack(i, path, current_sum + candidates[i])
            path.pop()
    
    backtrack(0, [], 0)
    return result
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

std::vector<std::vector<int>> combinationSum(std::vector<int>& candidates, int target) {
    std::vector<std::vector<int>> result;
    std::vector<int> path;
    std::sort(candidates.begin(), candidates.end());
    
    std::function<void(int, int)> backtrack = [&](int start, int current_sum) {
        if (current_sum == target) {
            result.push_back(path);
            return;
        }
        
        for (int i = start; i < candidates.size(); ++i) {
            if (current_sum + candidates[i] > target) {
                break;
            }
            path.push_back(candidates[i]);
            backtrack(i, current_sum + candidates[i]);
            path.pop_back();
        }
    };
    
    backtrack(0, 0);
    return result;
}
```

<br>

### 41. 单词搜索 (Word Search)

**题目描述：**

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word`。如果 `word` 存在于网格中，返回 `true`；否则，返回 `false`。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中"相邻"单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例：**
```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

**解题思路：**

回溯 + DFS：
1. 遍历网格，找到与单词首字母匹配的位置
2. 从该位置开始，使用 DFS 搜索
3. 对于每个位置，检查上下左右四个方向
4. 使用 visited 数组标记已访问的位置
5. 如果找到完整路径，返回 true

**复杂度分析：**
- 时间复杂度：O(m × n × 4^L)，其中 L 是单词长度
- 空间复杂度：O(L)，递归栈深度为 L

**Python 解答：**
```python
def exist(board, word):
    m, n = len(board), len(board[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    def dfs(i, j, index):
        if index == len(word):
            return True
        
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[index]:
            return False
        
        temp = board[i][j]
        board[i][j] = '#'
        
        for dx, dy in directions:
            if dfs(i + dx, j + dy, index + 1):
                return True
        
        board[i][j] = temp
        return False
    
    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    
    return False
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>

bool exist(std::vector<std::vector<char>>& board, std::string word) {
    int m = board.size(), n = board[0].size();
    std::vector<std::pair<int, int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    
    std::function<bool(int, int, int)> dfs = [&](int i, int j, int index) {
        if (index == word.length()) {
            return true;
        }
        
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != word[index]) {
            return false;
        }
        
        char temp = board[i][j];
        board[i][j] = '#';
        
        for (auto [dx, dy] : directions) {
            if (dfs(i + dx, j + dy, index + 1)) {
                return true;
            }
        }
        
        board[i][j] = temp;
        return false;
    };
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dfs(i, j, 0)) {
                return true;
            }
        }
    }
    
    return false;
}
```

<br>

### 42. 分割回文串 (Palindrome Partitioning)

**题目描述：**

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串**。返回 `s` 所有可能的分割方案。

**示例：**
```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

**解题思路：**

回溯算法：
1. 使用回溯法，尝试所有可能的分割方式
2. 对于每个位置，检查从当前位置到字符串末尾的所有子串
3. 如果子串是回文，加入当前路径，继续递归
4. 回溯时撤销选择

**复杂度分析：**
- 时间复杂度：O(N × 2^N)，最坏情况下所有子串都是回文
- 空间复杂度：O(N)，递归栈深度为 N

**Python 解答：**
```python
def partition(s):
    result = []
    
    def is_palindrome(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        
        for i in range(start, len(s)):
            if is_palindrome(start, i):
                path.append(s[start:i+1])
                backtrack(i + 1, path)
                path.pop()
    
    backtrack(0, [])
    return result
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>

std::vector<std::vector<std::string>> partition(std::string s) {
    std::vector<std::vector<std::string>> result;
    std::vector<std::string> path;
    
    auto isPalindrome = [&](int left, int right) {
        while (left < right) {
            if (s[left] != s[right]) {
                return false;
            }
            ++left;
            --right;
        }
        return true;
    };
    
    std::function<void(int)> backtrack = [&](int start) {
        if (start == s.length()) {
            result.push_back(path);
            return;
        }
        
        for (int i = start; i < s.length(); ++i) {
            if (isPalindrome(start, i)) {
                path.push_back(s.substr(start, i - start + 1));
                backtrack(i + 1);
                path.pop_back();
            }
        }
    };
    
    backtrack(0);
    return result;
}
```


## 三、贪心算法

### 43. 跳跃游戏 (Jump Game)

**题目描述：**

给定一个非负整数数组 `nums`，你最初位于数组的 **第一个下标**。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

**示例：**
```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

**解题思路：**

贪心算法：
1. 维护一个变量 `max_reach`，表示当前能到达的最远位置
2. 遍历数组，更新 `max_reach = max(max_reach, i + nums[i])`
3. 如果 `max_reach >= len(nums) - 1`，说明可以到达最后一个位置
4. 如果在某个位置 `i > max_reach`，说明无法继续前进

**复杂度分析：**
- 时间复杂度：O(n)，遍历数组一次
- 空间复杂度：O(1)

**Python 解答：**
```python
def canJump(nums):
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= len(nums) - 1:
            return True
    return True
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

bool canJump(std::vector<int>& nums) {
    int max_reach = 0;
    for (int i = 0; i < nums.size(); ++i) {
        if (i > max_reach) {
            return false;
        }
        max_reach = std::max(max_reach, i + nums[i]);
        if (max_reach >= nums.size() - 1) {
            return true;
        }
    }
    return true;
}
```

<br>

### 44. 跳跃游戏 II (Jump Game II)

**题目描述：**

给你一个非负整数数组 `nums`，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。假设你总是可以到达数组的最后一个位置。

**示例：**
```
输入：nums = [2,3,1,1,4]
输出：2
解释：跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

**解题思路：**

贪心算法：
1. 维护 `end` 表示当前跳跃能到达的最远位置
2. 维护 `max_pos` 表示在 `[start, end]` 范围内能到达的最远位置
3. 当到达 `end` 时，跳跃次数加1，更新 `end = max_pos`
4. 继续遍历直到到达最后一个位置

**复杂度分析：**
- 时间复杂度：O(n)，遍历数组一次
- 空间复杂度：O(1)

**Python 解答：**
```python
def jump(nums):
    jumps = 0
    end = 0
    max_pos = 0
    
    for i in range(len(nums) - 1):
        max_pos = max(max_pos, i + nums[i])
        if i == end:
            jumps += 1
            end = max_pos
    return jumps
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int jump(std::vector<int>& nums) {
    int jumps = 0;
    int end = 0;
    int max_pos = 0;
    
    for (int i = 0; i < nums.size() - 1; ++i) {
        max_pos = std::max(max_pos, i + nums[i]);
        if (i == end) {
            ++jumps;
            end = max_pos;
        }
    }
    return jumps;
}
```

<br>

### 45. 合并区间 (Merge Intervals)

**题目描述：**

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]`。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

**示例：**
```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6]。
```

**解题思路：**

排序 + 贪心：
1. 按区间的起始位置排序
2. 遍历区间，如果当前区间与结果中最后一个区间重叠，则合并
3. 否则，将当前区间加入结果

**复杂度分析：**
- 时间复杂度：O(n log n)，排序的时间复杂度
- 空间复杂度：O(1)，不考虑结果存储的空间

**Python 解答：**
```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]
    
    for interval in intervals[1:]:
        if interval[0] <= result[-1][1]:
            result[-1][1] = max(result[-1][1], interval[1])
        else:
            result.append(interval)
    
    return result
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

std::vector<std::vector<int>> merge(std::vector<std::vector<int>>& intervals) {
    std::sort(intervals.begin(), intervals.end());
    std::vector<std::vector<int>> result;
    result.push_back(intervals[0]);
    
    for (int i = 1; i < intervals.size(); ++i) {
        if (intervals[i][0] <= result.back()[1]) {
            result.back()[1] = std::max(result.back()[1], intervals[i][1]);
        } else {
            result.push_back(intervals[i]);
        }
    }
    
    return result;
}
```

<br>

### 46. 插入区间 (Insert Interval)

**题目描述：**

给你一个 **无重叠的**，按照区间起始端点排序的区间列表 `intervals`，其中 `intervals[i] = [starti, endi]` 表示第 `i` 个区间的开始和结束，并且 `intervals` 按照 `starti` 升序排列。同样给定一个区间 `newInterval = [start, end]` 表示另一个区间的开始和结束。

在 `intervals` 中插入区间 `newInterval`，使得 `intervals` 仍然按照区间起始端点排序，且区间之间不重叠（如果有必要的话，可以合并区间）。

返回插入后的 `intervals`。

**示例：**
```
输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]
```

**解题思路：**

1. 找到所有与新区间重叠的区间
2. 合并这些区间
3. 将合并后的区间插入到正确位置

**复杂度分析：**
- 时间复杂度：O(n)，遍历数组一次
- 空间复杂度：O(1)，不考虑结果存储的空间

**Python 解答：**
```python
def insert(intervals, newInterval):
    result = []
    i = 0
    
    while i < len(intervals) and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    
    while i < len(intervals) and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    
    result.append(newInterval)
    
    while i < len(intervals):
        result.append(intervals[i])
        i += 1
    
    return result
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

std::vector<std::vector<int>> insert(std::vector<std::vector<int>>& intervals, std::vector<int>& newInterval) {
    std::vector<std::vector<int>> result;
    int i = 0;
    
    while (i < intervals.size() && intervals[i][1] < newInterval[0]) {
        result.push_back(intervals[i]);
        ++i;
    }
    
    while (i < intervals.size() && intervals[i][0] <= newInterval[1]) {
        newInterval[0] = std::min(newInterval[0], intervals[i][0]);
        newInterval[1] = std::max(newInterval[1], intervals[i][1]);
        ++i;
    }
    
    result.push_back(newInterval);
    
    while (i < intervals.size()) {
        result.push_back(intervals[i]);
        ++i;
    }
    
    return result;
}
```

<br>

### 47. 螺旋矩阵 (Spiral Matrix)

**题目描述：**

给你一个 `m` 行 `n` 列的矩阵 `matrix`，请按照 **顺时针螺旋顺序**，返回矩阵中的所有元素。

**示例：**
```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**解题思路：**

模拟法：
1. 定义四个边界：top, bottom, left, right
2. 按照右、下、左、上的顺序遍历
3. 每完成一个方向，更新对应的边界
4. 当边界相遇时停止

**复杂度分析：**
- 时间复杂度：O(m × n)，需要遍历所有元素
- 空间复杂度：O(1)，不考虑结果存储的空间

**Python 解答：**
```python
def spiralOrder(matrix):
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    
    return result
```

**C++ 解答：**
```cpp
#include <vector>

std::vector<int> spiralOrder(std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return {};
    
    std::vector<int> result;
    int top = 0, bottom = matrix.size() - 1;
    int left = 0, right = matrix[0].size() - 1;
    
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; ++j) {
            result.push_back(matrix[top][j]);
        }
        ++top;
        
        for (int i = top; i <= bottom; ++i) {
            result.push_back(matrix[i][right]);
        }
        --right;
        
        if (top <= bottom) {
            for (int j = right; j >= left; --j) {
                result.push_back(matrix[bottom][j]);
            }
            --bottom;
        }
        
        if (left <= right) {
            for (int i = bottom; i >= top; --i) {
                result.push_back(matrix[i][left]);
            }
            ++left;
        }
    }
    
    return result;
}
```

<br>

### 48. 旋转图像 (Rotate Image)

**题目描述：**

给定一个 `n × n` 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 **原地** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

**示例：**
```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```

**解题思路：**

方法1：转置 + 翻转
1. 先转置矩阵（行列互换）
2. 再翻转每一行

方法2：四角交换
1. 对于每个位置 (i, j)，找到旋转后的位置
2. 一次交换四个位置的值

**复杂度分析：**
- 时间复杂度：O(n²)
- 空间复杂度：O(1)

**Python 解答：**
```python
def rotate(matrix):
    n = len(matrix)
    
    # 转置
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # 翻转每一行
    for i in range(n):
        matrix[i].reverse()
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

void rotate(std::vector<std::vector<int>>& matrix) {
    int n = matrix.size();
    
    // 转置
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }
    
    // 翻转每一行
    for (int i = 0; i < n; ++i) {
        std::reverse(matrix[i].begin(), matrix[i].end());
    }
}
```

<br>

### 49. 搜索二维矩阵 II (Search a 2D Matrix II)

**题目描述：**

编写一个高效的算法来搜索 `m x n` 矩阵 `matrix` 中的一个目标值 `target`。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

**示例：**
```
输入：matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], target = 5
输出：true
```

**解题思路：**

从右上角开始搜索：
1. 如果当前元素等于 target，返回 true
2. 如果当前元素大于 target，向左移动（排除当前列）
3. 如果当前元素小于 target，向下移动（排除当前行）

**复杂度分析：**
- 时间复杂度：O(m + n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def searchMatrix(matrix, target):
    if not matrix:
        return False
    
    m, n = len(matrix), len(matrix[0])
    i, j = 0, n - 1
    
    while i < m and j >= 0:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] > target:
            j -= 1
        else:
            i += 1
    
    return False
```

**C++ 解答：**
```cpp
#include <vector>

bool searchMatrix(std::vector<std::vector<int>>& matrix, int target) {
    if (matrix.empty()) return false;
    
    int m = matrix.size(), n = matrix[0].size();
    int i = 0, j = n - 1;
    
    while (i < m && j >= 0) {
        if (matrix[i][j] == target) {
            return true;
        } else if (matrix[i][j] > target) {
            --j;
        } else {
            ++i;
        }
    }
    
    return false;
}
```

<br>

### 50. 最长连续序列 (Longest Consecutive Sequence)

**题目描述：**

给定一个未排序的整数数组 `nums`，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

**示例：**
```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

**解题思路：**

哈希集合：
1. 将所有数字存入哈希集合
2. 对于每个数字，如果它是连续序列的起点（即 num-1 不在集合中），则从该数字开始计算连续序列的长度
3. 更新最长连续序列的长度

**复杂度分析：**
- 时间复杂度：O(n)，每个数字最多被访问两次
- 空间复杂度：O(n)

**Python 解答：**
```python
def longestConsecutive(nums):
    if not nums:
        return 0
    
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            longest = max(longest, current_length)
    
    return longest
```

**C++ 解答：**
```cpp
#include <vector>
#include <unordered_set>
#include <algorithm>

int longestConsecutive(std::vector<int>& nums) {
    if (nums.empty()) return 0;
    
    std::unordered_set<int> num_set(nums.begin(), nums.end());
    int longest = 0;
    
    for (int num : num_set) {
        if (num_set.find(num - 1) == num_set.end()) {
            int current_num = num;
            int current_length = 1;
            
            while (num_set.find(current_num + 1) != num_set.end()) {
                ++current_num;
                ++current_length;
            }
            
            longest = std::max(longest, current_length);
        }
    }
    
    return longest;
}
```


## 四、图论

### 51. 岛屿数量 (Number of Islands)

**题目描述：**

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

**示例：**
```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**解题思路：**

DFS 或 BFS：
1. 遍历网格，遇到 '1' 时，岛屿数量加1
2. 使用 DFS 或 BFS 将相邻的所有 '1' 标记为 '0'（沉岛）
3. 继续遍历直到所有位置都被访问

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(m × n)，递归栈深度

**Python 解答：**
```python
def numIslands(grid):
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    
    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        grid[i][j] = '0'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    
    return count
```

**C++ 解答：**
```cpp
#include <vector>

int numIslands(std::vector<std::vector<char>>& grid) {
    if (grid.empty()) return 0;
    
    int m = grid.size(), n = grid[0].size();
    int count = 0;
    
    std::function<void(int, int)> dfs = [&](int i, int j) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '0';
        dfs(i + 1, j);
        dfs(i - 1, j);
        dfs(i, j + 1);
        dfs(i, j - 1);
    };
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == '1') {
                ++count;
                dfs(i, j);
            }
        }
    }
    
    return count;
}
```

<br>

### 52. 课程表 (Course Schedule)

**题目描述：**

你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1`。

在选修某些课程之前需要一些先修课程。先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]`，表示如果要学习课程 `ai` 则 **必须** 先学习课程 `bi`。

例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0`，你需要先完成课程 `1`。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true`；否则，返回 `false`。

**示例：**
```
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。这是可能的。
```

**解题思路：**

拓扑排序（检测环）：
1. 构建有向图，计算每个节点的入度
2. 使用队列存储所有入度为 0 的节点
3. 每次从队列中取出一个节点，将其相邻节点的入度减1
4. 如果相邻节点的入度变为0，将其加入队列
5. 如果所有节点都被访问，说明没有环，返回 true

**复杂度分析：**
- 时间复杂度：O(V + E)，V 是节点数，E 是边数
- 空间复杂度：O(V + E)

**Python 解答：**
```python
def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    indegree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1
    
    queue = [i for i in range(numCourses) if indegree[i] == 0]
    count = 0
    
    while queue:
        node = queue.pop(0)
        count += 1
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return count == numCourses
```

**C++ 解答：**
```cpp
#include <vector>
#include <queue>

bool canFinish(int numCourses, std::vector<std::vector<int>>& prerequisites) {
    std::vector<std::vector<int>> graph(numCourses);
    std::vector<int> indegree(numCourses, 0);
    
    for (auto& edge : prerequisites) {
        graph[edge[1]].push_back(edge[0]);
        ++indegree[edge[0]];
    }
    
    std::queue<int> q;
    for (int i = 0; i < numCourses; ++i) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }
    
    int count = 0;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        ++count;
        for (int neighbor : graph[node]) {
            --indegree[neighbor];
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }
    
    return count == numCourses;
}
```

<br>

### 53. 克隆图 (Clone Graph)

**题目描述：**

给你无向 **连通** 图中一个节点的引用，请你返回该图的 **深拷贝**（克隆）。

图中的每个节点都包含它的值 `val`（`int`）和其邻居的列表（`List[Node]`）。

**示例：**
```
输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
```

**解题思路：**

DFS + 哈希表：
1. 使用哈希表存储已克隆的节点
2. 对于每个节点，如果已克隆，直接返回
3. 否则，创建新节点，递归克隆所有邻居

**复杂度分析：**
- 时间复杂度：O(V + E)
- 空间复杂度：O(V)

**Python 解答：**
```python
def cloneGraph(node):
    if not node:
        return None
    
    visited = {}
    
    def dfs(original):
        if original in visited:
            return visited[original]
        
        clone = Node(original.val)
        visited[original] = clone
        
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```

**C++ 解答：**
```cpp
#include <unordered_map>

Node* cloneGraph(Node* node) {
    if (!node) return nullptr;
    
    std::unordered_map<Node*, Node*> visited;
    
    std::function<Node*(Node*)> dfs = [&](Node* original) -> Node* {
        if (visited.find(original) != visited.end()) {
            return visited[original];
        }
        
        Node* clone = new Node(original->val);
        visited[original] = clone;
        
        for (Node* neighbor : original->neighbors) {
            clone->neighbors.push_back(dfs(neighbor));
        }
        
        return clone;
    };
    
    return dfs(node);
}
```

<br>

### 54. 单词接龙 (Word Ladder)

**题目描述：**

字典 `wordList` 中从单词 `beginWord` 和 `endWord` 的 **转换序列** 是一个按下述规格形成的序列：

- 序列中第一个单词是 `beginWord`。
- 序列中最后一个单词是 `endWord`。
- 每次转换只能改变一个字母。
- 转换过程中的中间单词必须是字典 `wordList` 中的单词。

给你两个单词 `beginWord` 和 `endWord` 和一个字典 `wordList`，找到从 `beginWord` 到 `endWord` 的 **最短转换序列** 中的 **单词数目**。如果不存在这样的转换序列，返回 `0`。

**示例：**
```
输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
输出：5
解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
```

**解题思路：**

BFS：
1. 将 beginWord 加入队列
2. 对于队列中的每个单词，尝试改变每个位置的字符
3. 如果改变后的单词在 wordList 中且未被访问，加入队列
4. 使用 BFS 找到最短路径

**复杂度分析：**
- 时间复杂度：O(M × N)，M 是单词长度，N 是字典大小
- 空间复杂度：O(N)

**Python 解答：**
```python
def ladderLength(beginWord, endWord, wordList):
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0
    
    queue = [(beginWord, 1)]
    visited = {beginWord}
    
    while queue:
        word, length = queue.pop(0)
        
        if word == endWord:
            return length
        
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word in wordSet and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))
    
    return 0
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>
#include <unordered_set>
#include <queue>

int ladderLength(std::string beginWord, std::string endWord, std::vector<std::string>& wordList) {
    std::unordered_set<std::string> wordSet(wordList.begin(), wordList.end());
    if (wordSet.find(endWord) == wordSet.end()) {
        return 0;
    }
    
    std::queue<std::pair<std::string, int>> q;
    q.push({beginWord, 1});
    std::unordered_set<std::string> visited;
    visited.insert(beginWord);
    
    while (!q.empty()) {
        auto [word, length] = q.front();
        q.pop();
        
        if (word == endWord) {
            return length;
        }
        
        for (int i = 0; i < word.length(); ++i) {
            std::string new_word = word;
            for (char c = 'a'; c <= 'z'; ++c) {
                new_word[i] = c;
                if (wordSet.find(new_word) != wordSet.end() && visited.find(new_word) == visited.end()) {
                    visited.insert(new_word);
                    q.push({new_word, length + 1});
                }
            }
        }
    }
    
    return 0;
}
```


## 五、堆与优先队列

### 55. 数组中的第K个最大元素 (Kth Largest Element in an Array)

**题目描述：**

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

**示例：**
```
输入：[3,2,1,5,6,4], k = 2
输出：5
```

**解题思路：**

方法1：快速选择（类似快排）
方法2：最小堆（维护大小为 k 的最小堆）

**复杂度分析：**
- 时间复杂度：O(n log k)（堆方法）或 O(n)（快速选择，平均情况）
- 空间复杂度：O(k)（堆方法）或 O(1)（快速选择）

**Python 解答：**
```python
import heapq

def findKthLargest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]
```

**C++ 解答：**
```cpp
#include <vector>
#include <queue>

int findKthLargest(std::vector<int>& nums, int k) {
    std::priority_queue<int, std::vector<int>, std::greater<int>> heap;
    
    for (int num : nums) {
        heap.push(num);
        if (heap.size() > k) {
            heap.pop();
        }
    }
    
    return heap.top();
}
```

<br>

### 56. 前 K 个高频元素 (Top K Frequent Elements)

**题目描述：**

给你一个整数数组 `nums` 和一个整数 `k`，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

**示例：**
```
输入：nums = [1,1,1,2,2,3], k = 2
输出：[1,2]
```

**解题思路：**

1. 统计每个元素的频率
2. 使用最小堆维护频率最高的 k 个元素
3. 返回堆中的元素

**复杂度分析：**
- 时间复杂度：O(n log k)
- 空间复杂度：O(n)

**Python 解答：**
```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    count = Counter(nums)
    heap = []
    
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]
```

**C++ 解答：**
```cpp
#include <vector>
#include <unordered_map>
#include <queue>

std::vector<int> topKFrequent(std::vector<int>& nums, int k) {
    std::unordered_map<int, int> count;
    for (int num : nums) {
        ++count[num];
    }
    
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, 
                       std::greater<std::pair<int, int>>> heap;
    
    for (auto& [num, freq] : count) {
        heap.push({freq, num});
        if (heap.size() > k) {
            heap.pop();
        }
    }
    
    std::vector<int> result;
    while (!heap.empty()) {
        result.push_back(heap.top().second);
        heap.pop();
    }
    
    return result;
}
```

<br>

### 57. 合并K个升序链表 (Merge k Sorted Lists)

**题目描述：**

给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。

**示例：**
```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
```

**解题思路：**

优先队列（最小堆）：
1. 将所有链表的头节点加入最小堆
2. 每次取出堆顶节点，加入结果链表
3. 如果该节点还有下一个节点，将下一个节点加入堆
4. 重复直到堆为空

**复杂度分析：**
- 时间复杂度：O(n log k)，n 是总节点数，k 是链表数
- 空间复杂度：O(k)

**Python 解答：**
```python
import heapq

def mergeKLists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))
    
    return dummy.next
```

**C++ 解答：**
```cpp
#include <vector>
#include <queue>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* mergeKLists(std::vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    std::priority_queue<ListNode*, std::vector<ListNode*>, decltype(cmp)> heap(cmp);
    
    for (ListNode* node : lists) {
        if (node) {
            heap.push(node);
        }
    }
    
    ListNode* dummy = new ListNode(0);
    ListNode* current = dummy;
    
    while (!heap.empty()) {
        ListNode* node = heap.top();
        heap.pop();
        current->next = node;
        current = current->next;
        if (node->next) {
            heap.push(node->next);
        }
    }
    
    return dummy->next;
}
```


## 六、栈与队列

### 58. 有效的括号 (Valid Parentheses)

**题目描述：**

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s`，判断字符串是否有效。

有效字符串需满足：
1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

**示例：**
```
输入：s = "()[]{}"
输出：true
```

**解题思路：**

使用栈：
1. 遇到左括号，入栈
2. 遇到右括号，检查栈顶是否匹配
3. 如果匹配，出栈；否则返回 false
4. 最后检查栈是否为空

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack
```

**C++ 解答：**
```cpp
#include <stack>
#include <unordered_map>

bool isValid(std::string s) {
    std::stack<char> stack;
    std::unordered_map<char, char> mapping = {{')', '('}, {'}', '{'}, {']', '['}};
    
    for (char c : s) {
        if (mapping.find(c) != mapping.end()) {
            if (stack.empty() || stack.top() != mapping[c]) {
                return false;
            }
            stack.pop();
        } else {
            stack.push(c);
        }
    }
    
    return stack.empty();
}
```

<br>

### 59. 最小栈 (Min Stack)

**题目描述：**

设计一个支持 `push`，`pop`，`top` 操作，并能在常数时间内检索到最小元素的栈。

实现 `MinStack` 类：
- `MinStack()` 初始化堆栈对象。
- `void push(int val)` 将元素val推入堆栈。
- `void pop()` 删除堆栈顶部的元素。
- `int top()` 获取堆栈顶部的元素。
- `int getMin()` 获取堆栈中的最小元素。

**示例：**
```
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]
```

**解题思路：**

使用辅助栈存储最小值：
1. 主栈存储所有元素
2. 辅助栈存储每个状态下的最小值
3. 每次 push 时，同时更新最小值栈
4. 每次 pop 时，同时弹出最小值栈

**复杂度分析：**
- 时间复杂度：O(1)，所有操作都是常数时间
- 空间复杂度：O(n)

**Python 解答：**
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.min_stack[-1]
```

**C++ 解答：**
```cpp
#include <stack>

class MinStack {
private:
    std::stack<int> stack;
    std::stack<int> min_stack;

public:
    MinStack() {}
    
    void push(int val) {
        stack.push(val);
        if (min_stack.empty() || val <= min_stack.top()) {
            min_stack.push(val);
        }
    }
    
    void pop() {
        if (stack.top() == min_stack.top()) {
            min_stack.pop();
        }
        stack.pop();
    }
    
    int top() {
        return stack.top();
    }
    
    int getMin() {
        return min_stack.top();
    }
};
```

<br>

### 60. 每日温度 (Daily Temperatures)

**题目描述：**

给定一个整数数组 `temperatures`，表示每天的温度，返回一个数组 `answer`，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 `0` 来代替。

**示例：**
```
输入：temperatures = [73,74,75,71,69,72,76,73]
输出：[1,1,4,2,1,1,0,0]
```

**解题思路：**

单调栈：
1. 使用栈存储温度的下标
2. 遍历数组，对于每个温度，如果当前温度大于栈顶温度，则找到了下一个更高温度
3. 计算天数差，更新结果数组
4. 将当前下标入栈

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def dailyTemperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    stack = []
    
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    
    return result
```

**C++ 解答：**
```cpp
#include <vector>
#include <stack>

std::vector<int> dailyTemperatures(std::vector<int>& temperatures) {
    int n = temperatures.size();
    std::vector<int> result(n, 0);
    std::stack<int> stack;
    
    for (int i = 0; i < n; ++i) {
        while (!stack.empty() && temperatures[i] > temperatures[stack.top()]) {
            int prev_index = stack.top();
            stack.pop();
            result[prev_index] = i - prev_index;
        }
        stack.push(i);
    }
    
    return result;
}
```

<br>

### 61. 柱状图中最大的矩形 (Largest Rectangle in Histogram)

**题目描述：**

给定 `n` 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

**示例：**
```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

**解题思路：**

单调栈：
1. 使用单调递增栈
2. 对于每个柱子，找到左右两边第一个比它矮的柱子
3. 计算以当前柱子为高的最大矩形面积
4. 更新最大面积

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, height * width)
    
    return max_area
```

**C++ 解答：**
```cpp
#include <vector>
#include <stack>
#include <algorithm>

int largestRectangleArea(std::vector<int>& heights) {
    std::stack<int> stack;
    int max_area = 0;
    
    for (int i = 0; i < heights.size(); ++i) {
        while (!stack.empty() && heights[stack.top()] > heights[i]) {
            int height = heights[stack.top()];
            stack.pop();
            int width = stack.empty() ? i : i - stack.top() - 1;
            max_area = std::max(max_area, height * width);
        }
        stack.push(i);
    }
    
    while (!stack.empty()) {
        int height = heights[stack.top()];
        stack.pop();
        int width = stack.empty() ? heights.size() : heights.size() - stack.top() - 1;
        max_area = std::max(max_area, height * width);
    }
    
    return max_area;
}
```

<br>

### 62. 接雨水 (Trapping Rain Water)

**题目描述：**

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

**示例：**
```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。
```

**解题思路：**

方法1：双指针
1. 使用左右指针，维护左右两边的最大高度
2. 对于每个位置，能接的雨水 = min(left_max, right_max) - height[i]
3. 移动较小的一边

方法2：单调栈
1. 使用单调递减栈
2. 当遇到更高的柱子时，计算能接的雨水

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)（双指针）或 O(n)（单调栈）

**Python 解答：**
```python
def trap(height):
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int trap(std::vector<int>& height) {
    if (height.empty()) return 0;
    
    int left = 0, right = height.size() - 1;
    int left_max = 0, right_max = 0;
    int water = 0;
    
    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= left_max) {
                left_max = height[left];
            } else {
                water += left_max - height[left];
            }
            ++left;
        } else {
            if (height[right] >= right_max) {
                right_max = height[right];
            } else {
                water += right_max - height[right];
            }
            --right;
        }
    }
    
    return water;
}
```


## 七、字符串处理

### 63. 最长公共前缀 (Longest Common Prefix)

**题目描述：**

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 `""`。

**示例：**
```
输入：strs = ["flower","flow","flight"]
输出："fl"
```

**解题思路：**

方法1：横向扫描
1. 以第一个字符串为基准
2. 逐个比较每个字符串的对应字符
3. 找到第一个不匹配的位置

方法2：纵向扫描
1. 同时比较所有字符串的同一位置
2. 找到第一个不匹配的位置

**复杂度分析：**
- 时间复杂度：O(S)，S 是所有字符串字符的总数
- 空间复杂度：O(1)

**Python 解答：**
```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    
    prefix = strs[0]
    for i in range(1, len(strs)):
        while not strs[i].startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>

std::string longestCommonPrefix(std::vector<std::string>& strs) {
    if (strs.empty()) return "";
    
    std::string prefix = strs[0];
    for (int i = 1; i < strs.size(); ++i) {
        while (strs[i].find(prefix) != 0) {
            prefix = prefix.substr(0, prefix.length() - 1);
            if (prefix.empty()) return "";
        }
    }
    return prefix;
}
```

<br>

### 64. 反转字符串 (Reverse String)

**题目描述：**

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 `s` 的形式给出。

不要给另外的数组分配额外的空间，你必须**原地修改输入数组**、使用 O(1) 的额外空间解决这一问题。

**示例：**
```
输入：s = ["h","e","l","l","o"]
输出：["o","l","l","e","h"]
```

**解题思路：**

双指针：
1. 使用左右指针
2. 交换左右指针指向的字符
3. 向中间移动指针

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def reverseString(s):
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
```

**C++ 解答：**
```cpp
#include <vector>

void reverseString(std::vector<char>& s) {
    int left = 0, right = s.size() - 1;
    while (left < right) {
        std::swap(s[left], s[right]);
        ++left;
        --right;
    }
}
```

<br>

### 65. 反转字符串中的单词 (Reverse Words in a String)

**题目描述：**

给你一个字符串 `s`，请你反转字符串中 **单词** 的顺序。

**单词** 是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的 **单词** 分隔开。

返回 **单词顺序颠倒且单词之间用单个空格连接的结果字符串**。

**注意：** 输入字符串 `s` 中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。

**示例：**
```
输入：s = "the sky is blue"
输出："blue is sky the"
```

**解题思路：**

1. 先反转整个字符串
2. 再反转每个单词
3. 处理多余空格

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def reverseWords(s):
    s = s.strip()
    words = s.split()
    return ' '.join(reversed(words))
```

**C++ 解答：**
```cpp
#include <string>
#include <algorithm>
#include <sstream>

std::string reverseWords(std::string s) {
    std::istringstream iss(s);
    std::vector<std::string> words;
    std::string word;
    
    while (iss >> word) {
        words.push_back(word);
    }
    
    std::reverse(words.begin(), words.end());
    
    std::string result;
    for (int i = 0; i < words.size(); ++i) {
        if (i > 0) result += " ";
        result += words[i];
    }
    
    return result;
}
```

<br>

### 66. 字符串相乘 (Multiply Strings)

**题目描述：**

给定两个以字符串形式表示的非负整数 `num1` 和 `num2`，返回 `num1` 和 `num2` 的乘积，它们的乘积也表示为字符串形式。

**注意：** 不能使用任何内置的 BigInteger 库或直接将输入转换为整数。

**示例：**
```
输入：num1 = "2", num2 = "3"
输出："6"
```

**解题思路：**

模拟竖式乘法：
1. 创建一个数组存储结果
2. 从右到左逐位相乘
3. 处理进位
4. 去除前导零

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(m + n)

**Python 解答：**
```python
def multiply(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"
    
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = int(num1[i]) * int(num2[j])
            p1, p2 = i + j, i + j + 1
            total = mul + result[p2]
            result[p2] = total % 10
            result[p1] += total // 10
    
    start = 0
    while start < len(result) and result[start] == 0:
        start += 1
    
    return ''.join(map(str, result[start:]))
```

**C++ 解答：**
```cpp
#include <string>
#include <vector>

std::string multiply(std::string num1, std::string num2) {
    if (num1 == "0" || num2 == "0") return "0";
    
    int m = num1.length(), n = num2.length();
    std::vector<int> result(m + n, 0);
    
    for (int i = m - 1; i >= 0; --i) {
        for (int j = n - 1; j >= 0; --j) {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int p1 = i + j, p2 = i + j + 1;
            int total = mul + result[p2];
            result[p2] = total % 10;
            result[p1] += total / 10;
        }
    }
    
    int start = 0;
    while (start < result.size() && result[start] == 0) {
        ++start;
    }
    
    std::string res;
    for (int i = start; i < result.size(); ++i) {
        res += (result[i] + '0');
    }
    
    return res;
}
```

<br>

### 67. 简化路径 (Simplify Path)

**题目描述：**

给你一个字符串 `path`，表示指向某一文件或目录的 Unix 风格 **绝对路径**（以 `'/'` 开头），请你将其转化为更加简洁的规范路径。

在 Unix 风格的文件系统中，一个点（`.`）表示当前目录本身；此外，两个点 （`..`） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。任意多个连续的斜杠（即，`'//'`）都被视为单个斜杠 `'/'`。 对于此问题，任何其他格式的点（例如，`'...'`）均被视为文件/目录名称。

返回 **简化后** 的 **规范路径**。

**示例：**
```
输入：path = "/home//foo/"
输出："/home/foo"
解释：在规范路径中，多个连续斜杠需要用一个斜杠替换，并且末尾的斜杠也需要被移除。
```

**解题思路：**

使用栈：
1. 按 '/' 分割路径
2. 遇到 '.' 或空字符串，跳过
3. 遇到 '..'，弹出栈顶
4. 其他情况，入栈
5. 最后用 '/' 连接栈中元素

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def simplifyPath(path):
    stack = []
    parts = path.split('/')
    
    for part in parts:
        if part == '..':
            if stack:
                stack.pop()
        elif part and part != '.':
            stack.append(part)
    
    return '/' + '/'.join(stack)
```

**C++ 解答：**
```cpp
#include <string>
#include <vector>
#include <sstream>

std::string simplifyPath(std::string path) {
    std::vector<std::string> stack;
    std::istringstream iss(path);
    std::string part;
    
    while (std::getline(iss, part, '/')) {
        if (part == "..") {
            if (!stack.empty()) {
                stack.pop_back();
            }
        } else if (part != "" && part != ".") {
            stack.push_back(part);
        }
    }
    
    std::string result;
    for (const std::string& s : stack) {
        result += "/" + s;
    }
    
    return result.empty() ? "/" : result;
}
```

<br>

### 68. 编辑距离 (Edit Distance)

**题目描述：**

给你两个单词 `word1` 和 `word2`，请返回将 `word1` 转换成 `word2` 所使用的最少操作数。

你可以对一个单词进行如下三种操作：
- 插入一个字符
- 删除一个字符
- 替换一个字符

**示例：**
```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**解题思路：**

动态规划：
1. `dp[i][j]` 表示 word1 的前 i 个字符转换为 word2 的前 j 个字符的最少操作数
2. 如果 `word1[i-1] == word2[j-1]`，`dp[i][j] = dp[i-1][j-1]`
3. 否则，`dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1`

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(m × n)

**Python 解答：**
```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n]
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>
#include <algorithm>

int minDistance(std::string word1, std::string word2) {
    int m = word1.length(), n = word2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    
    for (int i = 0; i <= m; ++i) {
        dp[i][0] = i;
    }
    for (int j = 0; j <= n; ++j) {
        dp[0][j] = j;
    }
    
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1;
            }
        }
    }
    
    return dp[m][n];
}
```


## 八、位运算

### 69. 只出现一次的数字 (Single Number)

**题目描述：**

给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

**说明：** 你的算法应该具有线性时间复杂度。你可以不使用额外空间来实现吗？

**示例：**
```
输入：[2,2,1]
输出：1
```

**解题思路：**

异或运算：
1. 任何数与0异或等于它本身
2. 任何数与自身异或等于0
3. 异或运算满足交换律和结合律
4. 将所有数字异或，结果就是只出现一次的数字

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
```

**C++ 解答：**
```cpp
#include <vector>

int singleNumber(std::vector<int>& nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```

<br>

### 70. 只出现一次的数字 II (Single Number II)

**题目描述：**

给你一个整数数组 `nums`，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次**。请你找出并返回那个只出现了一次的元素。

**示例：**
```
输入：nums = [2,2,3,2]
输出：3
```

**解题思路：**

位运算：
1. 统计每一位上1出现的次数
2. 如果某位上1出现的次数是3的倍数，说明只出现一次的数字在该位为0
3. 否则，只出现一次的数字在该位为1

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def singleNumber(nums):
    result = 0
    for i in range(32):
        count = 0
        for num in nums:
            count += (num >> i) & 1
        result |= (count % 3) << i
    return result if result < 2**31 else result - 2**32
```

**C++ 解答：**
```cpp
#include <vector>

int singleNumber(std::vector<int>& nums) {
    int result = 0;
    for (int i = 0; i < 32; ++i) {
        int count = 0;
        for (int num : nums) {
            count += (num >> i) & 1;
        }
        result |= (count % 3) << i;
    }
    return result;
}
```

<br>

### 71. 只出现一次的数字 III (Single Number III)

**题目描述：**

给定一个整数数组 `nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。找出只出现一次的那两个元素。你可以按 **任意顺序** 返回答案。

**示例：**
```
输入：nums = [1,2,1,3,2,5]
输出：[3,5]
解释：[5, 3] 也是有效的答案。
```

**解题思路：**

位运算：
1. 将所有数字异或，得到两个只出现一次数字的异或结果
2. 找到异或结果中任意一个为1的位（说明两个数字在该位不同）
3. 根据该位将数组分成两组，每组各包含一个只出现一次的数字
4. 分别对两组进行异或，得到两个结果

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def singleNumber(nums):
    xor_all = 0
    for num in nums:
        xor_all ^= num
    
    diff_bit = xor_all & (-xor_all)
    
    result = [0, 0]
    for num in nums:
        if num & diff_bit:
            result[0] ^= num
        else:
            result[1] ^= num
    
    return result
```

**C++ 解答：**
```cpp
#include <vector>

std::vector<int> singleNumber(std::vector<int>& nums) {
    int xor_all = 0;
    for (int num : nums) {
        xor_all ^= num;
    }
    
    int diff_bit = xor_all & (-xor_all);
    
    std::vector<int> result(2, 0);
    for (int num : nums) {
        if (num & diff_bit) {
            result[0] ^= num;
        } else {
            result[1] ^= num;
        }
    }
    
    return result;
}
```


## 九、其他重要题目

### 72. 缺失的第一个正数 (First Missing Positive)

**题目描述：**

给你一个未排序的整数数组 `nums`，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。

**示例：**
```
输入：nums = [1,2,0]
输出：3
```

**解题思路：**

原地哈希：
1. 将数组视为哈希表，将数字 i 放在位置 i-1
2. 遍历数组，如果 `nums[i]` 在 [1, n] 范围内，将其放到正确位置
3. 再次遍历数组，找到第一个 `nums[i] != i+1` 的位置

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def firstMissingPositive(nums):
    n = len(nums)
    
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int firstMissingPositive(std::vector<int>& nums) {
    int n = nums.size();
    
    for (int i = 0; i < n; ++i) {
        while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
            std::swap(nums[nums[i] - 1], nums[i]);
        }
    }
    
    for (int i = 0; i < n; ++i) {
        if (nums[i] != i + 1) {
            return i + 1;
        }
    }
    
    return n + 1;
}
```

<br>

### 73. 寻找重复数 (Find the Duplicate Number)

**题目描述：**

给定一个包含 `n + 1` 个整数的数组 `nums`，其数字都在 `[1, n]` 范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数**，返回 **这个重复的数**。

你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。

**示例：**
```
输入：nums = [1,3,4,2,2]
输出：2
```

**解题思路：**

快慢指针（Floyd判圈算法）：
1. 将数组视为链表，`nums[i]` 指向 `nums[nums[i]]`
2. 使用快慢指针找到环的入口
3. 环的入口就是重复的数字

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def findDuplicate(nums):
    slow = fast = nums[0]
    
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow
```

**C++ 解答：**
```cpp
#include <vector>

int findDuplicate(std::vector<int>& nums) {
    int slow = nums[0];
    int fast = nums[0];
    
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (slow != fast);
    
    slow = nums[0];
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[fast];
    }
    
    return slow;
}
```

<br>

### 74. 寻找两个正序数组的中位数 (Median of Two Sorted Arrays)

**题目描述：**

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数**。

算法的时间复杂度应该为 `O(log (m+n))`。

**示例：**
```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
```

**解题思路：**

二分查找：
1. 将问题转化为寻找第 k 小的元素
2. 每次比较两个数组的第 k/2 个元素
3. 排除较小的一半，继续查找

**复杂度分析：**
- 时间复杂度：O(log(m + n))
- 空间复杂度：O(1)

**Python 解答：**
```python
def findMedianSortedArrays(nums1, nums2):
    def getKth(k):
        i, j = 0, 0
        while True:
            if i == len(nums1):
                return nums2[j + k - 1]
            if j == len(nums2):
                return nums1[i + k - 1]
            if k == 1:
                return min(nums1[i], nums2[j])
            
            mid = k // 2
            idx1 = min(i + mid - 1, len(nums1) - 1)
            idx2 = min(j + mid - 1, len(nums2) - 1)
            
            if nums1[idx1] <= nums2[idx2]:
                k -= idx1 - i + 1
                i = idx1 + 1
            else:
                k -= idx2 - j + 1
                j = idx2 + 1
    
    m, n = len(nums1), len(nums2)
    if (m + n) % 2 == 1:
        return getKth((m + n) // 2 + 1)
    else:
        return (getKth((m + n) // 2) + getKth((m + n) // 2 + 1)) / 2.0
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
    int m = nums1.size(), n = nums2.size();
    
    auto getKth = [&](int k) {
        int i = 0, j = 0;
        while (true) {
            if (i == m) return nums2[j + k - 1];
            if (j == n) return nums1[i + k - 1];
            if (k == 1) return std::min(nums1[i], nums2[j]);
            
            int mid = k / 2;
            int idx1 = std::min(i + mid - 1, m - 1);
            int idx2 = std::min(j + mid - 1, n - 1);
            
            if (nums1[idx1] <= nums2[idx2]) {
                k -= idx1 - i + 1;
                i = idx1 + 1;
            } else {
                k -= idx2 - j + 1;
                j = idx2 + 1;
            }
        }
    };
    
    if ((m + n) % 2 == 1) {
        return getKth((m + n) / 2 + 1);
    } else {
        return (getKth((m + n) / 2) + getKth((m + n) / 2 + 1)) / 2.0;
    }
}
```

<br>

### 75. 正则表达式匹配 (Regular Expression Matching)

**题目描述：**

给你一个字符串 `s` 和一个字符规律 `p`，请你来实现一个支持 `'.'` 和 `'*'` 的正则表达式匹配。

- `'.'` 匹配任意单个字符
- `'*'` 匹配零个或多个前面的那一个元素

所谓匹配，是要涵盖 **整个** 字符串 `s` 的，而不是部分字符串。

**示例：**
```
输入：s = "aa", p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

**解题思路：**

动态规划：
1. `dp[i][j]` 表示 s 的前 i 个字符和 p 的前 j 个字符是否匹配
2. 如果 `p[j-1] == '*'`，可以选择匹配0次或多次
3. 如果 `p[j-1] == '.'` 或 `s[i-1] == p[j-1]`，匹配一个字符

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(m × n)

**Python 解答：**
```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2] or (dp[i-1][j] and (s[i-1] == p[j-2] or p[j-2] == '.'))
            else:
                dp[i][j] = dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.')
    
    return dp[m][n]
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>

bool isMatch(std::string s, std::string p) {
    int m = s.length(), n = p.length();
    std::vector<std::vector<bool>> dp(m + 1, std::vector<bool>(n + 1, false));
    dp[0][0] = true;
    
    for (int j = 2; j <= n; ++j) {
        if (p[j-1] == '*') {
            dp[0][j] = dp[0][j-2];
        }
    }
    
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (p[j-1] == '*') {
                dp[i][j] = dp[i][j-2] || (dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.'));
            } else {
                dp[i][j] = dp[i-1][j-1] && (s[i-1] == p[j-1] || p[j-1] == '.');
            }
        }
    }
    
    return dp[m][n];
}
```

<br>

### 76. 通配符匹配 (Wildcard Matching)

**题目描述：**

给定一个字符串 (`s`) 和一个字符模式 (`p`)，实现一个支持 `'?'` 和 `'*'` 的通配符匹配。

- `'?'` 可以匹配任何单个字符。
- `'*'` 可以匹配任意字符串（包括空字符串）。

两个字符串**完全匹配**才算匹配成功。

**示例：**
```
输入：s = "adceb", p = "*a*b"
输出：true
解释：第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".
```

**解题思路：**

动态规划：
1. `dp[i][j]` 表示 s 的前 i 个字符和 p 的前 j 个字符是否匹配
2. 如果 `p[j-1] == '*'`，可以匹配0个或多个字符
3. 如果 `p[j-1] == '?'` 或 `s[i-1] == p[j-1]`，匹配一个字符

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(m × n)

**Python 解答：**
```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '?')
    
    return dp[m][n]
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>

bool isMatch(std::string s, std::string p) {
    int m = s.length(), n = p.length();
    std::vector<std::vector<bool>> dp(m + 1, std::vector<bool>(n + 1, false));
    dp[0][0] = true;
    
    for (int j = 1; j <= n; ++j) {
        if (p[j-1] == '*') {
            dp[0][j] = dp[0][j-1];
        }
    }
    
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (p[j-1] == '*') {
                dp[i][j] = dp[i][j-1] || dp[i-1][j];
            } else {
                dp[i][j] = dp[i-1][j-1] && (s[i-1] == p[j-1] || p[j-1] == '?');
            }
        }
    }
    
    return dp[m][n];
}
```

<br>

### 77. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)

**题目描述：**

给定一个数组 `prices`，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0`。

**示例：**
```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5。
```

**解题思路：**

一次遍历：
1. 维护一个变量记录最低价格
2. 遍历数组，计算当前价格与最低价格的差值
3. 更新最大利润

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    
    return max_profit
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>
#include <climits>

int maxProfit(std::vector<int>& prices) {
    int min_price = INT_MAX;
    int max_profit = 0;
    
    for (int price : prices) {
        min_price = std::min(min_price, price);
        max_profit = std::max(max_profit, price - min_price);
    }
    
    return max_profit;
}
```

<br>

### 78. 买卖股票的最佳时机 II (Best Time to Buy and Sell Stock II)

**题目描述：**

给你一个整数数组 `prices`，其中 `prices[i]` 表示某支股票第 `i` 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 **最多** 只能持有 **一股** 股票。你也可以先购买，然后在 **同一天** 出售。

返回你能获得的 **最大** 利润。

**示例：**
```
输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4。
随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3。
总利润: 4 + 3 = 7。
```

**解题思路：**

贪心算法：
1. 只要后一天价格高于前一天，就进行交易
2. 累加所有正收益

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def maxProfit(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit
```

**C++ 解答：**
```cpp
#include <vector>

int maxProfit(std::vector<int>& prices) {
    int profit = 0;
    for (int i = 1; i < prices.size(); ++i) {
        if (prices[i] > prices[i-1]) {
            profit += prices[i] - prices[i-1];
        }
    }
    return profit;
}
```

<br>

### 79. 买卖股票的最佳时机 III (Best Time to Buy and Sell Stock III)

**题目描述：**

给定一个数组，它的第 `i` 个元素是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 **两笔** 交易。

**注意：** 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例：**
```
输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3。
随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3。
```

**解题思路：**

动态规划：
1. `dp[i][k][0/1]` 表示第 i 天，最多 k 次交易，持有/不持有股票的最大利润
2. 状态转移：
   - `dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])`
   - `dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])`

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def maxProfit(prices):
    buy1 = buy2 = float('-inf')
    sell1 = sell2 = 0
    
    for price in prices:
        buy1 = max(buy1, -price)
        sell1 = max(sell1, buy1 + price)
        buy2 = max(buy2, sell1 - price)
        sell2 = max(sell2, buy2 + price)
    
    return sell2
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>
#include <climits>

int maxProfit(std::vector<int>& prices) {
    int buy1 = INT_MIN, sell1 = 0;
    int buy2 = INT_MIN, sell2 = 0;
    
    for (int price : prices) {
        buy1 = std::max(buy1, -price);
        sell1 = std::max(sell1, buy1 + price);
        buy2 = std::max(buy2, sell1 - price);
        sell2 = std::max(sell2, buy2 + price);
    }
    
    return sell2;
}
```

<br>

### 80. 买卖股票的最佳时机 IV (Best Time to Buy and Sell Stock IV)

**题目描述：**

给定一个整数数组 `prices`，其中 `prices[i]` 是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 `k` 笔交易。

**注意：** 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例：**
```
输入：k = 2, prices = [2,4,1]
输出：2
解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2。
```

**解题思路：**

动态规划：
1. 如果 `k >= n/2`，相当于可以无限次交易，使用贪心算法
2. 否则，使用动态规划，类似第79题

**复杂度分析：**
- 时间复杂度：O(n × k)
- 空间复杂度：O(k)

**Python 解答：**
```python
def maxProfit(k, prices):
    n = len(prices)
    if k >= n // 2:
        profit = 0
        for i in range(1, n):
            if prices[i] > prices[i-1]:
                profit += prices[i] - prices[i-1]
        return profit
    
    buy = [float('-inf')] * (k + 1)
    sell = [0] * (k + 1)
    
    for price in prices:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j-1] - price)
            sell[j] = max(sell[j], buy[j] + price)
    
    return sell[k]
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>
#include <climits>

int maxProfit(int k, std::vector<int>& prices) {
    int n = prices.size();
    if (k >= n / 2) {
        int profit = 0;
        for (int i = 1; i < n; ++i) {
            if (prices[i] > prices[i-1]) {
                profit += prices[i] - prices[i-1];
            }
        }
        return profit;
    }
    
    std::vector<int> buy(k + 1, INT_MIN);
    std::vector<int> sell(k + 1, 0);
    
    for (int price : prices) {
        for (int j = 1; j <= k; ++j) {
            buy[j] = std::max(buy[j], sell[j-1] - price);
            sell[j] = std::max(sell[j], buy[j] + price);
        }
    }
    
    return sell[k];
}
```

<br>

### 81. 最大矩形 (Maximal Rectangle)

**题目描述：**

给定一个仅包含 `0` 和 `1`、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

**示例：**
```
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
```

**解题思路：**

将问题转化为柱状图中最大的矩形：
1. 对于每一行，计算以该行为底部的柱状图高度
2. 对每行应用"柱状图中最大的矩形"的算法

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def maximalRectangle(matrix):
    if not matrix:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0
    
    for i in range(m):
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        stack = []
        for j in range(n + 1):
            h = heights[j] if j < n else 0
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = j if not stack else j - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(j)
    
    return max_area
```

**C++ 解答：**
```cpp
#include <vector>
#include <stack>
#include <algorithm>

int maximalRectangle(std::vector<std::vector<char>>& matrix) {
    if (matrix.empty()) return 0;
    
    int m = matrix.size(), n = matrix[0].size();
    std::vector<int> heights(n, 0);
    int max_area = 0;
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            heights[j] = matrix[i][j] == '1' ? heights[j] + 1 : 0;
        }
        
        std::stack<int> stack;
        for (int j = 0; j <= n; ++j) {
            int h = j < n ? heights[j] : 0;
            while (!stack.empty() && heights[stack.top()] > h) {
                int height = heights[stack.top()];
                stack.pop();
                int width = stack.empty() ? j : j - stack.top() - 1;
                max_area = std::max(max_area, height * width);
            }
            stack.push(j);
        }
    }
    
    return max_area;
}
```

<br>

### 82. 分割等和子集 (Partition Equal Subset Sum)

**题目描述：**

给你一个 **只包含正整数** 的 **非空** 数组 `nums`。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**示例：**
```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11]。
```

**解题思路：**

0-1背包问题：
1. 如果总和为奇数，不可能分割
2. 目标是找到子集，使其和为总和的一半
3. 使用动态规划：`dp[i][j]` 表示前 i 个元素能否组成和为 j

**复杂度分析：**
- 时间复杂度：O(n × sum)
- 空间复杂度：O(sum)

**Python 解答：**
```python
def canPartition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]
```

**C++ 解答：**
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

bool canPartition(std::vector<int>& nums) {
    int total = std::accumulate(nums.begin(), nums.end(), 0);
    if (total % 2 != 0) return false;
    
    int target = total / 2;
    std::vector<bool> dp(target + 1, false);
    dp[0] = true;
    
    for (int num : nums) {
        for (int j = target; j >= num; --j) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    
    return dp[target];
}
```

<br>

### 83. 目标和 (Target Sum)

**题目描述：**

给你一个整数数组 `nums` 和一个整数 `target`。

向数组中的每个整数前添加 `'+'` 或 `'-'`，然后串联起所有整数，可以构造一个 **表达式**：

例如，`nums = [2, 1]`，可以在 `2` 之前添加 `'+'`，在 `1` 之前添加 `'-'`，然后串联起来得到表达式 `"+2-1"`。

返回可以通过上述方法构造的、运算结果等于 `target` 的不同 **表达式** 的数目。

**示例：**
```
输入：nums = [1,1,1,1,1], target = 3
输出：5
```

**解题思路：**

动态规划：
1. 设正数和为 P，负数和为 N，则 P - N = target，P + N = sum
2. 得到 P = (target + sum) / 2
3. 转化为0-1背包问题：找到和为 P 的子集数目

**复杂度分析：**
- 时间复杂度：O(n × sum)
- 空间复杂度：O(sum)

**Python 解答：**
```python
def findTargetSumWays(nums, target):
    total = sum(nums)
    if (total + target) % 2 != 0 or total < abs(target):
        return 0
    
    p = (total + target) // 2
    dp = [0] * (p + 1)
    dp[0] = 1
    
    for num in nums:
        for j in range(p, num - 1, -1):
            dp[j] += dp[j - num]
    
    return dp[p]
```

**C++ 解答：**
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

int findTargetSumWays(std::vector<int>& nums, int target) {
    int total = std::accumulate(nums.begin(), nums.end(), 0);
    if ((total + target) % 2 != 0 || total < std::abs(target)) {
        return 0;
    }
    
    int p = (total + target) / 2;
    std::vector<int> dp(p + 1, 0);
    dp[0] = 1;
    
    for (int num : nums) {
        for (int j = p; j >= num; --j) {
            dp[j] += dp[j - num];
        }
    }
    
    return dp[p];
}
```

<br>

### 84. 不同路径 (Unique Paths)

**题目描述：**

一个机器人位于一个 `m x n` 网格的左上角（起始点在下图中标记为 "Start"）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 "Finish"）。

问总共有多少条不同的路径？

**示例：**
```
输入：m = 3, n = 7
输出：28
```

**解题思路：**

动态规划：
1. `dp[i][j]` 表示到达位置 (i, j) 的路径数
2. `dp[i][j] = dp[i-1][j] + dp[i][j-1]`
3. 边界条件：第一行和第一列都是1

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def uniquePaths(m, n):
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    
    return dp[n-1]
```

**C++ 解答：**
```cpp
#include <vector>

int uniquePaths(int m, int n) {
    std::vector<int> dp(n, 1);
    
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            dp[j] += dp[j-1];
        }
    }
    
    return dp[n-1];
}
```

<br>

### 85. 不同路径 II (Unique Paths II)

**题目描述：**

一个机器人位于一个 `m x n` 网格的左上角（起始点在下图中标记为 "Start"）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 "Finish"）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。

**示例：**
```
输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
```

**解题思路：**

动态规划：
1. 如果 `obstacleGrid[i][j] == 1`，`dp[i][j] = 0`
2. 否则，`dp[i][j] = dp[i-1][j] + dp[i][j-1]`

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def uniquePathsWithObstacles(obstacleGrid):
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [0] * n
    dp[0] = 1 if obstacleGrid[0][0] == 0 else 0
    
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    
    return dp[n-1]
```

**C++ 解答：**
```cpp
#include <vector>

int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacleGrid) {
    int m = obstacleGrid.size(), n = obstacleGrid[0].size();
    std::vector<int> dp(n, 0);
    dp[0] = obstacleGrid[0][0] == 0 ? 1 : 0;
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (obstacleGrid[i][j] == 1) {
                dp[j] = 0;
            } else if (j > 0) {
                dp[j] += dp[j-1];
            }
        }
    }
    
    return dp[n-1];
}
```

<br>

### 86. 最小路径和 (Minimum Path Sum)

**题目描述：**

给定一个包含非负整数的 `m x n` 网格 `grid`，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：** 每次只能向下或者向右移动一步。

**示例：**
```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

**解题思路：**

动态规划：
1. `dp[i][j]` 表示到达位置 (i, j) 的最小路径和
2. `dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]`

**复杂度分析：**
- 时间复杂度：O(m × n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [float('inf')] * n
    dp[0] = grid[0][0]
    
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            if j > 0:
                dp[j] = min(dp[j], dp[j-1]) + grid[i][j]
            else:
                dp[j] = dp[j] + grid[i][j]
    
    return dp[n-1]
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>
#include <climits>

int minPathSum(std::vector<std::vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    std::vector<int> dp(n, INT_MAX);
    dp[0] = grid[0][0];
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == 0 && j == 0) continue;
            if (j > 0) {
                dp[j] = std::min(dp[j], dp[j-1]) + grid[i][j];
            } else {
                dp[j] = dp[j] + grid[i][j];
            }
        }
    }
    
    return dp[n-1];
}
```

<br>

### 87. 三角形最小路径和 (Triangle)

**题目描述：**

给定一个三角形 `triangle`，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点在这里指的是 **下标** 与 **上一层结点下标** 相同或者等于 **上一层结点下标 + 1** 的两个结点。也就是说，如果正位于当前行的下标 `i`，那么下一步可以移动到下一行的下标 `i` 或 `i + 1`。

**示例：**
```
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
```

**解题思路：**

动态规划（自底向上）：
1. 从倒数第二行开始，向上计算
2. `dp[j] = min(dp[j], dp[j+1]) + triangle[i][j]`

**复杂度分析：**
- 时间复杂度：O(n²)
- 空间复杂度：O(n)

**Python 解答：**
```python
def minimumTotal(triangle):
    n = len(triangle)
    dp = triangle[n-1][:]
    
    for i in range(n-2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = min(dp[j], dp[j+1]) + triangle[i][j]
    
    return dp[0]
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int minimumTotal(std::vector<std::vector<int>>& triangle) {
    int n = triangle.size();
    std::vector<int> dp = triangle[n-1];
    
    for (int i = n - 2; i >= 0; --i) {
        for (int j = 0; j < triangle[i].size(); ++j) {
            dp[j] = std::min(dp[j], dp[j+1]) + triangle[i][j];
        }
    }
    
    return dp[0];
}
```

<br>

### 88. 解码方法 (Decode Ways)

**题目描述：**

一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码**：

```
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
```

要 **解码** 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，`"11106"` 可以映射为：
- `"AAJF"`，将消息分组为 `(1 1 10 6)`
- `"KJF"`，将消息分组为 `(11 10 6)`

注意，消息不能分组为 `(1 11 06)`，因为 `"06"` 不能映射为 `"F"`，这是由于 `"6"` 和 `"06"` 在映射中并不等价。

给你一个只含数字的 **非空** 字符串 `s`，请计算并返回 **解码方法的总数**。

**示例：**
```
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6)。
```

**解题思路：**

动态规划：
1. `dp[i]` 表示前 i 个字符的解码方法数
2. 如果 `s[i-1] != '0'`，可以单独解码：`dp[i] += dp[i-1]`
3. 如果 `s[i-2:i]` 在 10-26 之间，可以组合解码：`dp[i] += dp[i-2]`

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def numDecodings(s):
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    prev2, prev1 = 1, 1
    
    for i in range(1, n):
        current = 0
        if s[i] != '0':
            current += prev1
        if 10 <= int(s[i-1:i+1]) <= 26:
            current += prev2
        prev2, prev1 = prev1, current
    
    return prev1
```

**C++ 解答：**
```cpp
#include <string>

int numDecodings(std::string s) {
    if (s.empty() || s[0] == '0') return 0;
    
    int n = s.length();
    int prev2 = 1, prev1 = 1;
    
    for (int i = 1; i < n; ++i) {
        int current = 0;
        if (s[i] != '0') {
            current += prev1;
        }
        int two_digit = std::stoi(s.substr(i-1, 2));
        if (two_digit >= 10 && two_digit <= 26) {
            current += prev2;
        }
        prev2 = prev1;
        prev1 = current;
    }
    
    return prev1;
}
```

<br>

### 89. 单词拆分 II (Word Break II)

**题目描述：**

给定一个字符串 `s` 和一个字符串字典 `wordDict`，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。以任意顺序 **返回所有这些可能的句子**。

**注意：** 词典中的同一个单词可能在分段中被重复使用多次。

**示例：**
```
输入：s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
输出：["cats and dog","cat sand dog"]
```

**解题思路：**

回溯 + 记忆化：
1. 使用回溯法尝试所有可能的分割
2. 使用记忆化避免重复计算
3. 如果当前子串在字典中，继续递归

**复杂度分析：**
- 时间复杂度：O(2^n)，最坏情况
- 空间复杂度：O(2^n)

**Python 解答：**
```python
def wordBreak(s, wordDict):
    wordSet = set(wordDict)
    memo = {}
    
    def backtrack(start):
        if start in memo:
            return memo[start]
        
        if start == len(s):
            return [""]
        
        result = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in wordSet:
                for sentence in backtrack(end):
                    result.append(word + ("" if not sentence else " " + sentence))
        
        memo[start] = result
        return result
    
    return backtrack(0)
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>

std::vector<std::string> wordBreak(std::string s, std::vector<std::string>& wordDict) {
    std::unordered_set<std::string> wordSet(wordDict.begin(), wordDict.end());
    std::unordered_map<int, std::vector<std::string>> memo;
    
    std::function<std::vector<std::string>(int)> backtrack = [&](int start) -> std::vector<std::string> {
        if (memo.find(start) != memo.end()) {
            return memo[start];
        }
        
        if (start == s.length()) {
            return {""};
        }
        
        std::vector<std::string> result;
        for (int end = start + 1; end <= s.length(); ++end) {
            std::string word = s.substr(start, end - start);
            if (wordSet.find(word) != wordSet.end()) {
                std::vector<std::string> sentences = backtrack(end);
                for (const std::string& sentence : sentences) {
                    result.push_back(word + (sentence.empty() ? "" : " " + sentence));
                }
            }
        }
        
        memo[start] = result;
        return result;
    };
    
    return backtrack(0);
}
```

<br>

### 90. 回文子串 (Palindromic Substrings)

**题目描述：**

给你一个字符串 `s`，请你统计并返回这个字符串中 **回文子串** 的数目。

回文字符串是正着读和倒着读一样的字符串。

子字符串是字符串中的由连续字符组成的一个序列。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

**示例：**
```
输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"
```

**解题思路：**

中心扩展法：
1. 对于每个可能的中心（单个字符或两个字符之间），向两边扩展
2. 统计所有回文子串的数量

**复杂度分析：**
- 时间复杂度：O(n²)
- 空间复杂度：O(1)

**Python 解答：**
```python
def countSubstrings(s):
    n = len(s)
    count = 0
    
    def expandAroundCenter(left, right):
        nonlocal count
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    
    for i in range(n):
        expandAroundCenter(i, i)
        expandAroundCenter(i, i + 1)
    
    return count
```

**C++ 解答：**
```cpp
#include <string>

int countSubstrings(std::string s) {
    int n = s.length();
    int count = 0;
    
    auto expandAroundCenter = [&](int left, int right) {
        while (left >= 0 && right < n && s[left] == s[right]) {
            ++count;
            --left;
            ++right;
        }
    };
    
    for (int i = 0; i < n; ++i) {
        expandAroundCenter(i, i);
        expandAroundCenter(i, i + 1);
    }
    
    return count;
}
```

<br>

### 91. 最长有效括号 (Longest Valid Parentheses)

**题目描述：**

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**示例：**
```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

**解题思路：**

动态规划：
1. `dp[i]` 表示以 s[i] 结尾的最长有效括号长度
2. 如果 `s[i] == ')'` 且 `s[i-1] == '('`，`dp[i] = dp[i-2] + 2`
3. 如果 `s[i] == ')'` 且 `s[i-1] == ')'` 且 `s[i-dp[i-1]-1] == '('`，`dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2`

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def longestValidParentheses(s):
    n = len(s)
    dp = [0] * n
    max_len = 0
    
    for i in range(1, n):
        if s[i] == ')':
            if s[i-1] == '(':
                dp[i] = (dp[i-2] if i >= 2 else 0) + 2
            elif i - dp[i-1] > 0 and s[i - dp[i-1] - 1] == '(':
                dp[i] = dp[i-1] + (dp[i - dp[i-1] - 2] if i - dp[i-1] >= 2 else 0) + 2
            max_len = max(max_len, dp[i])
    
    return max_len
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>
#include <algorithm>

int longestValidParentheses(std::string s) {
    int n = s.length();
    std::vector<int> dp(n, 0);
    int max_len = 0;
    
    for (int i = 1; i < n; ++i) {
        if (s[i] == ')') {
            if (s[i-1] == '(') {
                dp[i] = (i >= 2 ? dp[i-2] : 0) + 2;
            } else if (i - dp[i-1] > 0 && s[i - dp[i-1] - 1] == '(') {
                dp[i] = dp[i-1] + (i - dp[i-1] >= 2 ? dp[i - dp[i-1] - 2] : 0) + 2;
            }
            max_len = std::max(max_len, dp[i]);
        }
    }
    
    return max_len;
}
```

<br>

### 92. 戳气球 (Burst Balloons)

**题目描述：**

有 `n` 个气球，编号为 `0` 到 `n - 1`，每个气球上都标有一个数字，这些数字存在数组 `nums` 中。

现在要求你戳破所有的气球。戳破第 `i` 个气球，你可以获得 `nums[i - 1] * nums[i] * nums[i + 1]` 枚硬币。这里的 `i - 1` 和 `i + 1` 代表和 `i` 相邻的两个气球的序号。如果 `i - 1` 或 `i + 1` 超出了数组的边界，那么就当它是一个数字为 `1` 的气球。

求所能获得硬币的最大数量。

**示例：**
```
输入：nums = [3,1,5,8]
输出：167
解释：
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 15 + 120 + 24 + 8 = 167
```

**解题思路：**

区间DP：
1. 在数组两端添加1
2. `dp[i][j]` 表示戳破区间 (i, j) 内所有气球能获得的最大硬币数
3. 枚举最后一个戳破的气球 k，`dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])`

**复杂度分析：**
- 时间复杂度：O(n³)
- 空间复杂度：O(n²)

**Python 解答：**
```python
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):
                dp[i][j] = max(dp[i][j], 
                               dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
    
    return dp[0][n-1]
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int maxCoins(std::vector<int>& nums) {
    nums.insert(nums.begin(), 1);
    nums.push_back(1);
    int n = nums.size();
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    
    for (int length = 2; length < n; ++length) {
        for (int i = 0; i < n - length; ++i) {
            int j = i + length;
            for (int k = i + 1; k < j; ++k) {
                dp[i][j] = std::max(dp[i][j],
                                   dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]);
            }
        }
    }
    
    return dp[0][n-1];
}
```

<br>

### 93. 打家劫舍 II (House Robber II)

**题目描述：**

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 **围成一圈**，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **在不触动警报装置的情况下**，今晚能够偷窃到的最高金额。

**示例：**
```
输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
```

**解题思路：**

将问题分解为两个子问题：
1. 不偷第一间房：计算 nums[1:] 的最大值
2. 不偷最后一间房：计算 nums[:-1] 的最大值
3. 取两者的最大值

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def rob(nums):
    if len(nums) == 1:
        return nums[0]
    
    def robRange(start, end):
        prev2, prev1 = 0, 0
        for i in range(start, end):
            current = max(prev1, prev2 + nums[i])
            prev2, prev1 = prev1, current
        return prev1
    
    return max(robRange(0, len(nums) - 1), robRange(1, len(nums)))
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int rob(std::vector<int>& nums) {
    if (nums.size() == 1) return nums[0];
    
    auto robRange = [&](int start, int end) {
        int prev2 = 0, prev1 = 0;
        for (int i = start; i < end; ++i) {
            int current = std::max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = current;
        }
        return prev1;
    };
    
    return std::max(robRange(0, nums.size() - 1), robRange(1, nums.size()));
}
```

<br>

### 94. 打家劫舍 III (House Robber III)

**题目描述：**

小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 `root`。

除了 `root` 之外，每栋房子有且只有一个"父"房子与之相连。一番侦察之后，聪明的小偷意识到"这个地方的所有房屋的排列类似于一棵二叉树"。如果 **两个直接相连的房子在同一天晚上被打劫**，房屋将自动报警。

给定二叉树的 `root`。返回 **在不触动警报的情况下**，小偷能够盗取的最高金额。

**示例：**
```
输入：root = [3,2,3,null,3,null,1]
输出：7
解释：小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7
```

**解题思路：**

树形DP：
1. 对于每个节点，返回两个值：[不偷该节点的最大值, 偷该节点的最大值]
2. 如果偷当前节点，则不能偷子节点
3. 如果不偷当前节点，可以偷或不偷子节点

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(h)，h 是树的高度

**Python 解答：**
```python
def rob(root):
    def dfs(node):
        if not node:
            return [0, 0]
        
        left = dfs(node.left)
        right = dfs(node.right)
        
        rob_current = node.val + left[0] + right[0]
        not_rob_current = max(left) + max(right)
        
        return [not_rob_current, rob_current]
    
    return max(dfs(root))
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
};

std::vector<int> dfs(TreeNode* node) {
    if (!node) {
        return {0, 0};
    }
    
    std::vector<int> left = dfs(node->left);
    std::vector<int> right = dfs(node->right);
    
    int rob_current = node->val + left[0] + right[0];
    int not_rob_current = std::max(left[0], left[1]) + std::max(right[0], right[1]);
    
    return {not_rob_current, rob_current};
}

int rob(TreeNode* root) {
    std::vector<int> result = dfs(root);
    return std::max(result[0], result[1]);
}
```

<br>

### 95. 分割回文串 II (Palindrome Partitioning II)

**题目描述：**

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是回文。

返回符合要求的 **最少分割次数**。

**示例：**
```
输入：s = "aab"
输出：1
解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
```

**解题思路：**

动态规划：
1. 先用DP判断所有子串是否为回文
2. `dp[i]` 表示前 i 个字符的最少分割次数
3. 如果 `s[j:i+1]` 是回文，`dp[i] = min(dp[i], dp[j-1] + 1)`

**复杂度分析：**
- 时间复杂度：O(n²)
- 空间复杂度：O(n²)

**Python 解答：**
```python
def minCut(s):
    n = len(s)
    is_palindrome = [[False] * n for _ in range(n)]
    
    for i in range(n):
        is_palindrome[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2:
                    is_palindrome[i][j] = True
                else:
                    is_palindrome[i][j] = is_palindrome[i+1][j-1]
    
    dp = [0] * n
    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0
        else:
            dp[i] = i
            for j in range(1, i + 1):
                if is_palindrome[j][i]:
                    dp[i] = min(dp[i], dp[j-1] + 1)
    
    return dp[n-1]
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>
#include <algorithm>

int minCut(std::string s) {
    int n = s.length();
    std::vector<std::vector<bool>> is_palindrome(n, std::vector<bool>(n, false));
    
    for (int i = 0; i < n; ++i) {
        is_palindrome[i][i] = true;
    }
    
    for (int length = 2; length <= n; ++length) {
        for (int i = 0; i <= n - length; ++i) {
            int j = i + length - 1;
            if (s[i] == s[j]) {
                if (length == 2) {
                    is_palindrome[i][j] = true;
                } else {
                    is_palindrome[i][j] = is_palindrome[i+1][j-1];
                }
            }
        }
    }
    
    std::vector<int> dp(n);
    for (int i = 0; i < n; ++i) {
        if (is_palindrome[0][i]) {
            dp[i] = 0;
        } else {
            dp[i] = i;
            for (int j = 1; j <= i; ++j) {
                if (is_palindrome[j][i]) {
                    dp[i] = std::min(dp[i], dp[j-1] + 1);
                }
            }
        }
    }
    
    return dp[n-1];
}
```

<br>

### 96. 最长回文子序列 (Longest Palindromic Subsequence)

**题目描述：**

给你一个字符串 `s`，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

**示例：**
```
输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb"。
```

**解题思路：**

动态规划：
1. `dp[i][j]` 表示 s[i:j+1] 的最长回文子序列长度
2. 如果 `s[i] == s[j]`，`dp[i][j] = dp[i+1][j-1] + 2`
3. 否则，`dp[i][j] = max(dp[i+1][j], dp[i][j-1])`

**复杂度分析：**
- 时间复杂度：O(n²)
- 空间复杂度：O(n²)

**Python 解答：**
```python
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = 1
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    
    return dp[0][n-1]
```

**C++ 解答：**
```cpp
#include <vector>
#include <string>
#include <algorithm>

int longestPalindromeSubseq(std::string s) {
    int n = s.length();
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    
    for (int i = 0; i < n; ++i) {
        dp[i][i] = 1;
    }
    
    for (int length = 2; length <= n; ++length) {
        for (int i = 0; i <= n - length; ++i) {
            int j = i + length - 1;
            if (s[i] == s[j]) {
                dp[i][j] = dp[i+1][j-1] + 2;
            } else {
                dp[i][j] = std::max(dp[i+1][j], dp[i][j-1]);
            }
        }
    }
    
    return dp[0][n-1];
}
```

<br>

### 97. 俄罗斯套娃信封问题 (Russian Doll Envelopes)

**题目描述：**

给你一个二维整数数组 `envelopes`，其中 `envelopes[i] = [wi, hi]`，表示第 `i` 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 **最多能有多少个** 信封能组成一组"俄罗斯套娃"信封（即可以把一个信封放到另一个信封里面）。

**注意：** 不允许旋转信封。

**示例：**
```
输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
```

**解题思路：**

排序 + 最长递增子序列：
1. 按宽度升序排序，宽度相同时按高度降序排序
2. 对高度数组求最长递增子序列

**复杂度分析：**
- 时间复杂度：O(n log n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def maxEnvelopes(envelopes):
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    
    heights = [env[1] for env in envelopes]
    dp = []
    
    for h in heights:
        left, right = 0, len(dp)
        while left < right:
            mid = (left + right) // 2
            if dp[mid] < h:
                left = mid + 1
            else:
                right = mid
        
        if left == len(dp):
            dp.append(h)
        else:
            dp[left] = h
    
    return len(dp)
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int maxEnvelopes(std::vector<std::vector<int>>& envelopes) {
    std::sort(envelopes.begin(), envelopes.end(), 
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  return a[0] < b[0] || (a[0] == b[0] && a[1] > b[1]);
              });
    
    std::vector<int> dp;
    for (const auto& env : envelopes) {
        int h = env[1];
        auto it = std::lower_bound(dp.begin(), dp.end(), h);
        if (it == dp.end()) {
            dp.push_back(h);
        } else {
            *it = h;
        }
    }
    
    return dp.size();
}
```

<br>

### 98. 最长递增子序列 (Longest Increasing Subsequence)

**题目描述：**

给你一个整数数组 `nums`，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例：**
```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,18]，因此长度为 4。
```

**解题思路：**

动态规划 + 二分查找：
1. 使用数组 `dp` 存储长度为 i+1 的递增子序列的最小末尾元素
2. 对于每个元素，使用二分查找找到应该插入的位置
3. 如果元素大于所有已有元素，追加到数组末尾

**复杂度分析：**
- 时间复杂度：O(n log n)
- 空间复杂度：O(n)

**Python 解答：**
```python
def lengthOfLIS(nums):
    dp = []
    
    for num in nums:
        left, right = 0, len(dp)
        while left < right:
            mid = (left + right) // 2
            if dp[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(dp):
            dp.append(num)
        else:
            dp[left] = num
    
    return len(dp)
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int lengthOfLIS(std::vector<int>& nums) {
    std::vector<int> dp;
    
    for (int num : nums) {
        auto it = std::lower_bound(dp.begin(), dp.end(), num);
        if (it == dp.end()) {
            dp.push_back(num);
        } else {
            *it = num;
        }
    }
    
    return dp.size();
}
```

<br>

### 99. 最佳买卖股票时机含冷冻期 (Best Time to Buy and Sell Stock with Cooldown)

**题目描述：**

给定一个整数数组 `prices`，其中第 `prices[i]` 表示第 `i` 天股票的价格。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

- 卖出股票后，你无法在第二天买入股票（即冷冻期为 1 天）。

**注意：** 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例：**
```
输入：prices = [1,2,3,0,2]
输出：3
解释：对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

**解题思路：**

动态规划：
1. `dp[i][0]` 表示第 i 天持有股票的最大利润
2. `dp[i][1]` 表示第 i 天不持有股票（处于冷冻期）的最大利润
3. `dp[i][2]` 表示第 i 天不持有股票（不处于冷冻期）的最大利润

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def maxProfit(prices):
    hold = float('-inf')
    frozen = 0
    free = 0
    
    for price in prices:
        hold, frozen, free = max(hold, free - price), hold + price, max(free, frozen)
    
    return max(frozen, free)
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>
#include <climits>

int maxProfit(std::vector<int>& prices) {
    int hold = INT_MIN;
    int frozen = 0;
    int free = 0;
    
    for (int price : prices) {
        int new_hold = std::max(hold, free - price);
        int new_frozen = hold + price;
        int new_free = std::max(free, frozen);
        hold = new_hold;
        frozen = new_frozen;
        free = new_free;
    }
    
    return std::max(frozen, free);
}
```

<br>

### 100. 买卖股票的最佳时机含手续费 (Best Time to Buy and Sell Stock with Transaction Fee)

**题目描述：**

给定一个整数数组 `prices`，其中 `prices[i]` 表示第 `i` 天的股票价格；整数 `fee` 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

**注意：** 这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

**示例：**
```
输入：prices = [1, 3, 2, 8, 4, 9], fee = 2
输出：8
解释：能够达到的最大利润:  
在此处买入 prices[0] = 1
在此处卖出 prices[3] = 8
在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9
总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8
```

**解题思路：**

动态规划：
1. `hold` 表示持有股票的最大利润
2. `sold` 表示不持有股票的最大利润
3. 买入时扣除手续费，卖出时扣除手续费（或只在卖出时扣除）

**复杂度分析：**
- 时间复杂度：O(n)
- 空间复杂度：O(1)

**Python 解答：**
```python
def maxProfit(prices, fee):
    hold = -prices[0]
    sold = 0
    
    for i in range(1, len(prices)):
        hold = max(hold, sold - prices[i])
        sold = max(sold, hold + prices[i] - fee)
    
    return sold
```

**C++ 解答：**
```cpp
#include <vector>
#include <algorithm>

int maxProfit(std::vector<int>& prices, int fee) {
    int hold = -prices[0];
    int sold = 0;
    
    for (int i = 1; i < prices.size(); ++i) {
        hold = std::max(hold, sold - prices[i]);
        sold = std::max(sold, hold + prices[i] - fee);
    }
    
    return sold;
}
```

