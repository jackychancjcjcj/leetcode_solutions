* [二分查找](#1)
  * [模板一](#1.1)
* [位运算意义](#2)
* [回溯算法](#3)
  * [模板一](#3.1)
* [动态规划](#4)
  * [模板一](#4.1)
* [深度优先搜索](#5)
  * [模板一](#5.1)
* [广度优先搜索](#6)
  * [模板一](#6.1)
* [排序](#7)
  * [冒泡排序](#7.1)
  * [选择排序](#7.2)
  * [插入排序](#7.3)
  * [希尔排序](#7.4)
  * [归并排序](#7.5)
# <span id='1'>二分查找</span>
## <span id='1.1'>模板一</span>
核心思想就是维护两个指针，每次在中间找数。
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1
        left, right = 0, len(nums)-1
        nums.sort()
        while left <= right:
            mid = (left+right) // 2
            if nums[mid] > target:
                right = mid - 1
            if nums[mid] < target:
                left = mid + 1
            if nums[mid] == target:
                return mid
        return -1
```

# <span id='2'>位运算意义</span>

* 10 >> n 意味着 10 // 2^n
* 10 << n 意味着 10 * 2^n
* 10 & 1 意味着 10 % 2 是否有余数也就是判断奇偶性
* n & (n-1) == 0 意味着 n是2的次幂
# <span id='3'>回溯算法</span>
## <span id='3.1'>模板一</span>
当满足条件时，递归执行函数
```python
 def backtrack(conbination,nextdigit):
     if len(nextdigit) == 0:
         res.append(conbination)
     else:
         for letter in phone[nextdigit[0]]:
             backtrack(conbination + letter,nextdigit[1:])

 res = []
 backtrack('',digits)
```
# <span id='4'>动态规划</span>
## <span id='4.1'>模板一</span>
明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义
```python
# 初始化 base case
dp[0][0][...] = base
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)
```
# <span id='5'>深度优先搜索</span>
## <span id='5.1'>模板一</span>
对于每一个节点，往下遍历
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:

        res = []
        def dfs(root):
            if not root:
                return
            
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)
        
        dfs(root)
        return res
```
# <span id='6'>广度优先搜索</span>
## <span id='6.1'>模板一</span>
对于每一个节点，往右遍历
```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        que = collections.deque([(root, 1)])
        while que:
            node, depth = que.popleft()
            if not node.left and not node.right:
                return depth
            if node.left:
                que.append((node.left, depth + 1))
            if node.right:
                que.append((node.right, depth + 1))
        
        return 0
```
# <span id='7'>排序</span>
## <span id='7.1'>冒泡排序</span>
* 原理：每一轮都是前后比较交换位置。
* 最好时间复杂度o(n),最坏时间复杂度o(n^2)，稳定.
```python
for i in range(len(nums)):
    for j in range(0,len(nums)-i-1):
        if nums[j] > nums[j+1]:
            nums[j], nums[j+1] = nums[j+1], nums[j]
```
## <span id='7.2'>选择排序</span>
* 原理：每一轮找到数组中最小的值，交换到该轮数组首位。
* 最好时间复杂度o(n^2),最坏时间复杂度o(n^2)，不稳定.
```python
for i in range(len(nums)-1):
    min_index = i
    for j in range(i+1,len(nums)):
        if nums[min_index] > nums[j]:
            min_index = j
    nums[i], nums[min_index] = nums[min_index], nums[i]
```
## <span id='7.3'>插入排序</span>
* 原理：每一轮将当前数字与前面以排序数组比较插入。
* 最好时间复杂度o(n),最坏时间复杂度o(n^2)，稳定.
```python
for i in range(1,len(nums)):
    key = nums[i]
    j = i - 1
    while j >= 0 and key < nums[j]:
        nums[j+1] = nums[j]
        j -= 1
    nums[j+1] = key
```
## <span id='7.4'>希尔排序</span>
* 原理：将数组异步长gap分组，每组内插入排序，然后gap/2继续插入排序。
* 最好时间复杂度o(nlogn),最坏时间复杂度o(n^2)，不稳定.
```python
gap = int(len(nums)/2)
while gap > 0:
    for i in range(gap,len(nums)):
        key = nums[i]
        j = i
        while j-gap >= 0 and key < nums[j-gap]:
            nums[j] = nums[j-gap]
            j -= gap
        nums[j] = key
    gap = int(gap/2)
```
## <span id='7.5'>归并排序</span>
* 原理：主要是递归或者说分治，将数组切分到最细粒度，每个子递归有左右两块，两块分别是有序的，然后进行合并，总体看就是分治和合并的过程。
* 最好时间复杂度o(nlogn),最坏时间复杂度o(nlogn)，稳定，但空间复杂度上来了，o(n).
```python
def mergesort(tmp):
    n = len(tmp)
    if n <= 1:
        return tmp
    mid = n // 2
    l_arr = mergesort(tmp[:mid])
    r_arr = mergesort(tmp[mid:])
    l,r = 0,0
    res = []
    while l < len(l_arr) and r < len(r_arr):
        if l_arr[l] < r_arr[r]:
            res.append(l_arr[l])
            l += 1
        else:
            res.append(r_arr[r])
            r += 1                      
    res += l_arr[l:]
    res += r_arr[r:]
    return res

output = mergesort(nums)
```
## <span id='7.6'>快速排序</span>
* 原理：递归，找到一个标的，比枢纽大的放右边，比枢纽小的放左边，每一轮都能确定这个枢纽在数组中的位置，递归就是处理枢纽左子序列和右子序列。
* 最好时间复杂度o(nlogn),最坏时间复杂度o(n^2)，不稳定，空间复杂度o(logn).
```python
 def quicksort(nums,left,right):
     if left < right:
         l = left
         r = right
         key = nums[left]
         while r > l:
             while l < r and nums[r] > key:
                 r -= 1
             if l < r:
                 nums[l] = nums[r]
                 l += 1
             while l < r and nums[l] < key:
                 l += 1
             if l < r:
                 nums[r] = nums[l]
                 r -= 1
         nums[l] = key
         quicksort(nums,left,l-1)
         quicksort(nums,l+1,right)

 quicksort(nums,0,len(nums)-1)
```
