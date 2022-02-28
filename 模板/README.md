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

* 10 >> 1 意味着 10 // 2
* 10 & 1 意味着 10 % 2 是否有余数也就是判断奇偶性
* 
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
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        if not root.left and not root.right:
            return 1
        
        min_depth = 10**9
        if root.left:
            min_depth = min(self.minDepth(root.left), min_depth)
        if root.right:
            min_depth = min(self.minDepth(root.right), min_depth)
        
        return min_depth + 1
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
