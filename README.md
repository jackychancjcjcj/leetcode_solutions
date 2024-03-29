# leetcode_solutions
![Author](https://img.shields.io/badge/Author-CJ-red.svg "Author")
![LICENSE](https://img.shields.io/github/license/JoeyBling/hexo-theme-yilia-plus "LICENSE")
![Language](https://img.shields.io/badge/Language-python3.6-green.svg "Laguage")
![Last update](https://img.shields.io/badge/last%20update-01%20Dec%202021-brightgreen.svg?style=flat-square "Last update")
* [424.替换后的最长重复字符](#424)
* [408.滑动窗口中位数](#408)
* [643.子数组最大平均数I](#643-1)
* [888.公平的糖果棒交换](#888)
* [1208.尽可能使字符串相等](#1208)
* [1423.可获得的最大点数](#1423)
* [665.非递减数列](#665)
* [978.最长湍流子数组](#978)
* [992.K 个不同整数的子数组](#992)
* [567.字符串的排列](#567)
* [561.数组拆分I](#561-1)
* [485.最大连续1的个数](#485)
* [566.重塑矩阵](#566)
* [765.情侣牵手](#765)
* [995.K连续位的最小翻转次数](#995)
* [1004.最大连续1的个数III](#1004-3)
* [697.数组的度](#697)
* [1438.绝对差不超过限制的最长连续子数组](#1438)
* [766.托普利茨矩阵](#766)
* [1052.爱生气的书店老板](#1052)
* [832.翻转图像](#832)
* [867.转置矩阵](#867)
* [1178.猜字谜](#1178)
* [395.至少有K个重复字符的最长子串](#395)
* [896.单调数列](#896)
* [303.区域和检索 - 数组不可变](#303)
* [304.二维区域和检索 - 矩阵不可变](#304)
* [338.比特位计数](#338)
* [354.俄罗斯套娃信封问题](#354)
* [503.下一个更大元素 II](#503)
* [131.分割回文串](#131)
* [132.分割回文串 II](#132)
* [1047.删除字符串中的所有相邻重复项](#1047)
* [224.基本计算器](#224)
* [331.验证二叉树的前序序列化](#331)
* [227.基本计算器 II](#227)
* [705.设计哈希集合](#705)
* [706.设计哈希映射](#706)
* [54.螺旋矩阵](#54)
* [59.螺旋矩阵II](#59)
* [92.反转链表 II](#92)
* [191.位1的个数](#191)
* [150.逆波兰表达式求值](#150)
* [73.矩阵置零](#73)
* [341.扁平化嵌套列表迭代器](#341)
* [456.132模式](#456)
* [74.搜索二维矩阵](#74)
* [190.颠倒二进制位](#190)
* [90.子集 II](#90)
* [1006.笨阶乘](#1006)
* [面试题17.21.直方图的水量](#17.21)
* [80.删除有序数组中的重复项 II](#80)
* [81.搜索旋转排序数组 II](#81)
* [153.寻找旋转排序数组中的最小值](#153)
* [154.寻找旋转排序数组中的最小值 II](#154)
* [179.最大数](#179)
* [783.二叉搜索树节点最小距离](#783)
* [87. 扰乱字符串](#87)
* [27. 移除元素](#27)
* [377. 组合总和 Ⅳ](#377)
* [897. 递增顺序搜索树](#897)
* [1011. 在 D 天内送达包裹的能力](#1011)
* [938. 二叉搜索树的范围和](#938)
* [633. 平方数之和](#633)
* [403. 青蛙过河](#403)
* [872. 叶子相似的树](#872)
* [1269. 停在原地的方案数](#1269)
* [401. 二进制手表](#401)
* [剑指 Offer II 001. 整数除法](#JZ001)
* [剑指 Offer II 008. 和大于等于 target 的最短子数组](#JZ008)
* [剑指 Offer II 010. 和为 k 的子数组](#JZ010)
* [剑指 Offer II 014. 字符串中的变位词](#JZ014)
## <span id='424'>424.替换后的最长重复字符</span>
双指针法，动态窗口：
```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        num = [0]*26
        maxn = left = right = 0
        n = len(s)
        while right < n:
            num[ord(s[right])-ord('A')] += 1
            maxn = max(maxn,num[ord(s[right])-ord('A')])
            if right - left + 1 - maxn > k:
                num[ord(s[left])-ord('A')] -= 1
                left += 1
            right += 1
        return right - left
```
稍微优化，用defaultdict取代数组:
```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        import collections
        num = collections.defaultdict(int)
        maxn = left = right = 0
        n = len(s)
        while right < n:
            num[s[right]] += 1
            maxn = max(maxn,num[s[right]])
            if right - left + 1 - maxn > k:
                num[s[left]] -= 1
                left += 1
            right += 1
        return right - left
```
## <span id='408'>408.滑动窗口中位数</span>
解法1：
```python
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        # 数组+暴力
        median = lambda a: a[len(a)//2] if len(a)%2 else a[len(a)//2-1]/2 + a[len(a)//2]/2
        res = []
        for i in range(len(nums)-k+1):
            res.append(median(sorted(nums[i:i+k])))
        return res 
```
解法2：
```python
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        # 数组+二分
        import bisect
        median = lambda a:a[len(a)//2] if len(a)%2 else a[len(a)//2-1]/2 + a[len(a)//2]/2
        a = sorted(nums[:k])
        res = [median(a)]
        for i,j in zip(nums[:-k],nums[k:]):
            a.remove(i)
            a.insert(bisect.bisect_left(a,j),j)
            res.append(median(a))
        return res
```
解法3：
```python
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        # 数组+二分
        import bisect
        median = lambda a:a[len(a)//2] if len(a)%2 else a[len(a)//2-1]/2 + a[len(a)//2]/2
        a = sorted(nums[:k])
        res = [median(a)]
        for i,j in zip(nums[:-k],nums[k:]):
            a.remove(bisect.bisect_left(a,i))
            a.insert(bisect.bisect_left(a,j),j)
            res.append(median(a))
        return res
```
## <span id='643-1'>643.子数组最大平均数I</span>
暴力解法：
```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        Maxaverage = sum(nums[:k])/k
        for i in range(1,len(nums)-k+1):
            if nums[i-1] > nums[i+k-1]:
                pass
            else:
                Maxaverage = max(Maxaverage,sum(nums[i:k+i])/k)
        return Maxaverage
```
维护两个数组：
```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        Maxaverage = sum(nums[:k])
        total = sum(nums[:k])
        for i in range(k,len(nums)):
            total = total - nums[i-k] + nums[i]
            Maxaverage = max(Maxaverage,total)
        return Maxaverage/k
```
## <span id='888'>888.公平的糖果棒交换</span>
哈希表：
```python
class Solution:
    def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
        sumA,sumB = sum(A),sum(B)
        delta = (sumA-sumB)//2
        setA = set(A)
        for y in B:
            x = y + delta
            if x in setA:
                return [x,y]
```
## <span id='1208'>1208.尽可能使字符串相等</span>
队列解法：
```python
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        tmp = []
        from collections import deque
        tmp_2 = deque()
        total = 0
        output = -inf
        for i,j in zip(s,t):
            tmp.append(abs(ord(i)-ord(j)))
        for i in tmp:
            total += i
            tmp_2.append(i)
            if total > maxCost:
                total -= tmp_2.popleft()
            output = max(output,len(tmp_2))
        return output
```
## <span id='1423'>1423.可获得的最大点数</span>
滑动窗口：
```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        windowSize = n - k
        total = sum(cardPoints[:windowSize])
        minScore = total
        for i in range(windowSize,n):
            total = total - cardPoints[i-windowSize] + cardPoints[i]
            minScore = min(minScore,total)
        return sum(cardPoints) - minScore
```
## <span id='665'>665.非递减数列</span>
```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        N = len(nums)
        count = 0
        for i in range(1, N):
            if nums[i] < nums[i - 1]:
                count += 1
                if i == 1 or nums[i] >= nums[i - 2]:
                    nums[i - 1] = nums[i]
                else:
                    nums[i] = nums[i - 1]
            if count == 2:
                return False
        return True
```
## <span id='978'>978.最长湍流子数组</span>
两个数组，一个记录前面符号，一个记录现在符号：
```python
class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        n = len(arr)
        count = 2
        maxLen = -inf 
        symbol_last = 1
        if n == 1:
            return 1
        if arr[0] < arr[1]:
            symbol_last = 1
        elif arr[0] == arr[1]:
            symbol_last = 0
            count = 1
        else:
            symbol_last = -1
        if n == 2 and symbol_last == 0:
            return 1
        for i in range(2,n):
            if arr[i] > arr[i-1]:
                symbol_now = 1
            elif arr[i] == arr[i-1]:
                maxLen = max(maxLen,count)
                count = 1
                symbol_last = 0 
                continue
            else:
                symbol_now = -1
            if symbol_now * symbol_last == 1:
                maxLen = max(maxLen,count)
                count = 2
            else:
                count += 1
            symbol_last = symbol_now
        return max(maxLen,count)
```
## <span id='992'>992.K 个不同整数的子数组</span>
滑动窗口：
```python
class Solution:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        from collections import Counter
        left1 = left2 = right = 0
        num1,num2 = Counter(),Counter()
        tot1 = tot2 = 0
        res = 0
        for right,num in enumerate(A):
            if num1[num] == 0:
                tot1 += 1
            num1[num] += 1
            if num2[num] == 0:
                tot2 += 1
            num2[num] += 1
            while tot1 > K:
                num1[A[left1]] -= 1
                if num1[A[left1]] == 0:
                    tot1 -= 1
                left1 += 1
            while tot2 > K-1:
                num2[A[left2]] -= 1
                if num2[A[left2]] == 0:
                    tot2 -= 1
                left2 += 1
            res += left2 - left1
        return res
```
## <span id='567'>567.字符串的排列</span>
滑动窗口+列表：
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        t1 = collections.Counter(s1)
        N = len(s2)
        left = 0
        right = len(s1) - 1
        t2 = collections.Counter(s2[0:right])
        while right < N:
            t2[s2[right]] += 1
            if t1 == t2:
                return True
            t2[s2[left]] -= 1
            if t2[s2[left]] == 0:
                del t2[s2[left]]
            left += 1
            right += 1
        return False
```
## <span id='561-1'>561.数组拆分I</span>
```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        total = 0
        for i in range(len(nums)):
            if i % 2 == 0:
                total += nums[i]
        return total
```
## <span id='485'>485.最大连续1的个数</span>
```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        tmp = 0
        max_1 = -inf
        for i in range(len(nums)):
            if nums[i] == 1:
                tmp += 1
            else:
                max_1 = max(max_1,tmp)
                tmp = 0
        max_1 = max(max_1,tmp)
        return max_1
```
## <span id='566'>566.重塑矩阵</span>
```python
class Solution:
    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        m,n = len(nums),len(nums[0])
        if m*n != r*c :
            return nums
        output = []
        tmp = []
        count = 0
        for i in range(m):
            for j in range(n):
                count += 1
                if count < c:
                    tmp.append(nums[i][j])
                else:
                    tmp.append(nums[i][j])
                    count = 0
                    output.append(tmp)
                    tmp = []
        return output
```
## <span id='765'>765.情侣牵手</span>
贪心算法+异或位运算：
```python
class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        res = 0
        for i in range(0,len(row)-1,2):
            if row[i] == row[i+1]^1:
                continue
            for j in range(i+1,len(row)):
                if row[i] == row[j]^1:
                    row[i+1],row[j] = row[j],row[i+1]
            res += 1
        return res 
```
## <span id='995'>995.K连续位的最小翻转次数</span>
贪心+差分运算优化：
```python
class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        n = len(A)
        res = 0
        tmp = [0] * (n+1)
        ans = 0
        for i in range(n):
            res += tmp[i]
            if (A[i]+res) % 2 == 0: # 证明当前位置是偶数
                if i + K > n:
                    return -1
                res += 1
                tmp[i+K] -= 1 # 即使超出数组也要弄进去
                ans += 1 
        return ans
```
## <span id='1004-3'>1004.最大连续1的个数III</span>
滑动窗口：
```python 
class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        n = len(A)
        left = lsum = rsum = 0
        ans = 0
        for right in range(n):
            rsum += 1 - A[right]
            while lsum < rsum - K:
                lsum += 1 - A[left]
                left += 1
            ans = max(ans,right-left+1)
        return ans 
```
## <span id='697'>697.数组的度</span>
哈希表：
```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        tmp = defaultdict(list)
        for index,num in enumerate(nums):
            if not tmp[num]:
                tmp[num] = []
            tmp[num].append(index)
        max_len = -inf
        res = inf
        for values in tmp.values():
            if len(values) > max_len:
                max_len = len(values)
                res = values[-1] - values[0] + 1
            elif len(values) == max_len:
                res = min(res,values[-1] - values[0] + 1)
        return res
```
## <span id='1438'>1438.绝对差不超过限制的最长连续子数组</span>
滑动窗口+排序列表：
```python
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        from sortedcontainers import SortedList
        s = SortedList()
        left = right = res = 0
        while right < len(nums):
            s.add(nums[right])
            while s[-1] - s[0] > limit:
                s.remove(nums[left])
                left += 1
            res = max(res,right-left+1)
            right += 1
        return res
```
滑动窗口+单项队列：
```python
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        from collections import deque
        max_q, min_q = deque(),deque()
        left = right = res = 0
        while right < len(nums):
            while max_q and nums[right] > max_q[-1]:
                max_q.pop()
            while min_q and nums[right] < min_q[-1]:
                min_q.pop()
            max_q.append(nums[right])
            min_q.append(nums[right])
            while max_q and min_q and max_q[0] - min_q[0] > limit:
                if nums[left] == min_q[0]:
                    min_q.popleft()
                if nums[left] == max_q[0]:
                    max_q.popleft()
                left += 1
            res = max(res,right-left+1)
            right += 1
        return res
```
## <span id='766'>766.托普利茨矩阵</span>
暴力遍历：
```python
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        m,n = len(matrix),len(matrix[0])
        for i in range(m-1):
            column = 0
            row = i
            tmp = matrix[i][column]
            while row < m-1 and column < n-1:
                row += 1 
                column += 1
                if tmp != matrix[row][column]:
                    return False
        for i in range(1,n-1):
            row = 0
            column = i
            tmp = matrix[row][i]
            while row < m-1 and column < n-1:
                row += 1 
                column += 1
                if tmp != matrix[row][column]:
                    return False
        return True
```
切片（思路，i行的[:-1]是要等于i+1行的[1:]的）：
```python
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        for i in range(len(matrix)-1):
            if matrix[i][:-1] != matrix[i+1][1:]:
                return False
        return True
```
## <span id='1052'>1052.爱生气的书店老板</span>
滑动窗口：
```python
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], X: int) -> int:
        res = ans_2 = 0
        ans = -inf
        if X >= len(grumpy):
            return sum(customers)
        for i in range(X):
            if grumpy[i] == 1:
                res += customers[i]
            else:
                ans_2 += customers[i]
        ans = max(ans,res)
        for i in range(X,len(customers)):
            res -= grumpy[i-X] * customers[i-X]
            if grumpy[i] == 1:
                res += customers[i]
            else:
                ans_2 += customers[i]
            ans = max(ans,res)
        return ans_2 + ans
```
## <span id='832'>832.翻转图像</span>
暴力法：
```python
class Solution:
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        res = []
        for i in range(len(A)):
            tmp = []
            for j in range(len(A[0])):
                tmp.append(A[i][j]^1)
            res.append(tmp[::-1])
        return res
```
二分法查找，遍历减半：
```python
class Solution:
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        for row in A:
            for j in range((len(row) + 1) // 2):
                if row[j] == row[-1-j]:             # 采用Python化的符号索引
                    row[j] = row[-1-j] = 1 - row[j]    
        return A
```
## <span id='867'>867.转置矩阵</span>
暴力遍历：
```python
class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        m,n = len(matrix),len(matrix[0])
        res = [[0]*m for _ in range(n)]
        for i in range(m):
            for j in range(n):
                res[j][i] = matrix[i][j]
        return res
```
## <span id='1178'>1178.猜字谜</span>
状态压缩+子集：
```python
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        freq = collections.Counter()
        res = []
        for word in words:
            mask = 0
            for c in word:
                mask |= 1 << (ord(c)-ord('a'))
            freq[mask] += 1
        for puzzle in puzzles:
            total = 0
            for perm in self.subset(puzzle[1:]):
                mask = 1 << (ord(puzzle[0])-ord('a'))
                for c in perm:
                    mask |= 1 << (ord(c)-ord('a'))
                total += freq[mask]
            res.append(total)
        return res

    def subset(self,words):
        res = ['']
        for i in words:
            res = res + [i+word for word in res]
        return res
```
## <span id='395'>395.至少有K个重复字符的最长子串]</span>
递归：
```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        if len(s) < k:
            return 0
        for c in set(s):
            if s.count(c) < k:
                return max(self.longestSubstring(t,k) for t in s.split(c))
        return len(s)
```
## <span id='896'>896.单调数列</span>
```python
class Solution:
    def isMonotonic(self, A: List[int]) -> bool:
        if len(A) <= 1:
            return True
        last_diff = A[1] - A[0]
        for i in range(2,len(A)):
            now_diff = A[i] - A[i-1]
            if last_diff * now_diff < 0:
                return False
            if now_diff != 0:
                last_diff = now_diff
        return True
```
## <span id='303'>区域和检索 - 数组不可变</span>
前缀和:
```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = [0]
        self.res = self.nums
        for num in nums:
            self.res.append(self.res[-1]+num)

    def sumRange(self, i: int, j: int) -> int:
        return self.res[j+1] - self.res[i]
```
## <span id='304'>304.二维区域和检索 - 矩阵不可变</span>
一维前缀和：
```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m,n = len(matrix),(len(matrix[0]) if matrix else 0)
        _sum = [[0]*(n+1) for _ in range(m)]
        for i in range(m):
            for j in range(n):
                _sum[i][j+1] = _sum[i][j] + matrix[i][j]
        self.sum = _sum

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        total = 0
        for i in range(row1,row2+1):
            total += self.sum[i][col2+1] - self.sum[i][col1]
        return total
```
二维前缀和：
```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m,n = len(matrix),(len(matrix[0]) if matrix else 0)
        _sum = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                _sum[i+1][j+1] = _sum[i+1][j] + _sum[i][j+1] - _sum[i][j] + matrix[i][j]
        self.sum = _sum

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.sum[row2+1][col2+1] - self.sum[row1][col2+1] - self.sum[row2+1][col1] + self.sum[row1][col1]
```
## <span id='338'>338.比特位计数</span>
位与运行：
```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        def countOnes(x: int) -> int:
            ones = 0
            while x > 0:
                x &= (x - 1)
                ones += 1
            return ones
        
        bits = [countOnes(i) for i in range(num + 1)]
        return bits
```
位与运算+动态规划：
```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        bits = [0]
        highbit = 0
        for i in range(1,num+1):
            if (i & i-1) == 0:
                highbit = i
            bits.append(bits[i-highbit]+1)
        return bits
```
## <span id='354'>354.俄罗斯套娃信封问题</span>
动态规划：
```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        n = len(envelopes)
        envelopes.sort(key=lambda x:(x[0],-x[1]))
        res = [1]*n
        for i in range(n):
            for j in range(i):
                if envelopes[j][1] < envelopes[i][1]:
                    res[i] = max(res[i],res[j]+1)
        return max(res)
```
二分查找+动态规划：
```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        import bisect
        n = len(envelopes)
        envelopes.sort(key=lambda x:(x[0],-x[1]))
        res = [envelopes[0][1]]
        for i in range(n):
            if (num := envelopes[i][1]) > res[-1]:
                res.append(num)
            else:
                index = bisect.bisect_left(res,num)
                res[index] = num
        return len(res)
```
## <span id='503'>503.下一个更大元素 II</span>
单调栈+循环数组：
```python

class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ret = [-1] * n
        stk = list()

        for i in range(n * 2 - 1):
            while stk and nums[stk[-1]] < nums[i % n]:
                ret[stk.pop()] = nums[i % n]
            stk.append(i % n)
        return ret
```
## <span id='131'>131.分割回文串</span>
回溯：
```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        def cc(s, tmp):
            if not s:
                return res.append(tmp)
            for i in range(1,len(s)+1):
                if s[:i] == s[:i][::-1]:
                    cc(s[i:],tmp+[s[:i]])
        cc(s,[])
        return res
```
## <span id='132'>132.分割回文串 II</span>
动态规划：
```python
class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        g = [[True] * n for _ in range(n)]

        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                g[i][j] = (s[i] == s[j]) and g[i + 1][j - 1]

        f = [float("inf")] * n
        for i in range(n):
            if g[0][i]:
                f[i] = 0
            else:
                for j in range(i):
                    if g[j + 1][i]:
                        f[i] = min(f[i], f[j] + 1)
        
        return f[n - 1]

```
## <span id='1047'>1047.删除字符串中的所有相邻重复项</span>
栈：
```python
class Solution:
    def removeDuplicates(self, S: str) -> str:
        stk = list()
        for s in S:
            if stk and stk[-1] == s:
                stk.pop()
            else:
                stk.append(s)
        return ''.join(stk)
```
## <span id='224'>224.基本计算器</span>
栈：
```python
class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        num = 0
        stack = []
        sign = 1
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == "+":
                res += sign * num
                num = 0
                sign = 1
            elif c == "-":
                res += sign * num
                num = 0
                sign = -1
            elif c =="(":
                stack.append(res)
                stack.append(sign)
                sign = 1
                res = 0 
            elif c == ")":
                res += num * sign
                num = 0
                res = res * stack.pop() + stack.pop()
        res += num * sign
        return res
```
## <span id='331'>331.验证二叉树的前序序列化</span>
边，一个空节点消耗一条边，非空节点创造两条边：
```python
class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        edge = 1
        tmp = preorder.split(',')
        for i in tmp:
            edge -= 1
            if edge < 0:return False
            if i != '#':
                edge += 2
        return edge == 0
```
栈：
```python
class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        stack = []
        tmp = preorder.split(',')
        for i in tmp:
            while stack and stack[-1] == '#' and i == '#':
                stack.pop()
                if not stack:
                    return False
                stack.pop()
            stack.append(i)
        return len(stack) == 1 and stack[0] == '#'
```
## <span id='227'>227.基本计算器 II</span>
栈：
```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        pre_ops = '+'
        num = 0
        for index,i in enumerate(s):
            if i.isdigit():
                num = num*10 + int(i)
            if index == len(s)-1 or i in '+-*/':
                if pre_ops == '+':
                    stack.append(num)
                elif pre_ops == '-':
                    stack.append(-num)
                elif pre_ops == '*':
                    stack.append(stack.pop()*num)
                elif pre_ops == '/':
                    tmp = stack.pop()
                    if tmp < 0:
                        stack.append(int(tmp/num))
                    else:
                        stack.append(tmp//num)
                pre_ops = i
                num = 0
        return sum(stack)
```
## <span id='705'>705.设计哈希集合</span>
拉链数组：
```python
class MyHashSet:

    def __init__(self):
        self.buckets = 1000
        self.itemsPerBucket = 1001
        self.table = [[] for _ in range(self.buckets)]

    def hash(self, key):
        return key % self.buckets
    
    def pos(self, key):
        return key // self.buckets
    
    def add(self, key):
        hashkey = self.hash(key)
        if not self.table[hashkey]:
            self.table[hashkey] = [0] * self.itemsPerBucket
        self.table[hashkey][self.pos(key)] = 1
        
    def remove(self, key):
        hashkey = self.hash(key)
        if self.table[hashkey]:
            self.table[hashkey][self.pos(key)] = 0

    def contains(self, key):
        hashkey = self.hash(key)
        return (self.table[hashkey] != []) and (self.table[hashkey][self.pos(key)] == 1)
```
## <span id='706'>706.设计哈希映射</span>
拉链数组：
```python
class MyHashMap(object):

    def __init__(self):
        self.map = [[-1] * 1000 for _ in range(1001)]

    def put(self, key, value):
        row, col = key // 1000, key % 1000
        self.map[row][col] = value

    def get(self, key):
        row, col = key // 1000, key % 1000
        return self.map[row][col]

    def remove(self, key):
        row, col = key // 1000, key % 1000
        self.map[row][col] = -1

```
## <span id='54'>54.螺旋矩阵</span>
玄学：
```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        tmp = []
        while matrix:
            tmp += matrix.pop(0)
            matrix = list(zip(*matrix))[::-1]
        return tmp
```
## <span id='59'>59.螺旋矩阵II</span>
模拟：
```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        res = [[0]*n for _ in range(n)]
        row,col,index=0,0,0
        tmp = [[0,1],[1,0],[0,-1],[-1,0]]
        for i in range(n*n):
            res[row][col] += (i+1)
            r,c = tmp[index]
            tmp_r,tmp_c = row + r,col + c
            if tmp_r < 0 or tmp_r >= n or tmp_c < 0 or tmp_c >= n or res[tmp_r][tmp_c] > 0:
                index = (index+1)%4
                r,c = tmp[index]
            row,col = row+r,col+c
        return res
```
## <span id='92'>92.反转链表 II</span>
指针法：
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        count = 1
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        while pre.next and count < left:
            pre = pre.next
            count += 1
        cur = pre.next
        tail = cur
        while cur and count <= right:
            tmp = cur.next
            cur.next = pre.next
            pre.next = cur
            tail.next = tmp
            cur = tmp
            count += 1
        return dummy.next
```
## <span id='191'>191.位1的个数</span>
位运算：
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = sum(1 for i in range(32) if n & (1<<i))
        return res
```
位运算性质, n & n-1是将末位的1变为0：
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            n &= n-1
            res += 1

        return res
```
## <span id='150'>150. 逆波兰表达式求值</span>
栈：
```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        func = {
            '+':add,
            '-':sub,
            '*':mul,
            '/':lambda x,y:int(x/y)
        }

        stack = list()
        for token in tokens:
            try:
                value = int(token)
            except ValueError:
                num_1 = stack.pop()
                num_2 = stack.pop()
                value = func[token](num_2,num_1)
            stack.append(value)

        return stack[0]
```
## <span id='73'>73. 矩阵置零</span>
双指针：
```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        flag_col0 = any(matrix[i][0] == 0 for i in range(m))
        flag_row0 = any(matrix[0][j] == 0 for j in range(n))
        
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        if flag_col0:
            for i in range(m):
                matrix[i][0] = 0
        
        if flag_row0:
            for j in range(n):
                matrix[0][j] = 0
```
## <span id='341'>341.扁平化嵌套列表迭代器</span>
列表+递归：
```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.points = 0
        self.lists = []
        def gen(lists):
            for i in lists:
                if i.isInteger():
                    self.lists.append(i.getInteger())
                else:
                    gen(i.getList())
        gen(nestedList)

    def next(self) -> int:
        self.points += 1
        return self.lists[self.points-1]
        
    def hasNext(self) -> bool:
        if self.points >= len(self.lists):
             return False
        return True
```
栈+递归：
```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        from collections import deque
        self.q = deque()
        self.dfs(nestedList)
        
    def dfs(self, nestedList):
        for elem in nestedList:
            if elem.isInteger():
                self.q.append(elem.getInteger())
            else:
                self.dfs(elem.getList())    
    
    def next(self) -> int:
        return self.q.popleft()
    
    def hasNext(self) -> bool:
        return len(self.q)
```
## <span id='456'>456.132模式</span>
枚举3：
```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 3:return False
        min_left = nums[0]
        tmp = sorted(nums[2:])
        from bisect import bisect_right 
        for i in range(1,n-1):
            if nums[i] > min_left:
                index = bisect_right(tmp,min_left)
                if index < len(tmp) and nums[i] > tmp[index]:
                    return True
            min_left = min(min_left,nums[i])
            tmp.remove(nums[i+1])
        return False
```
## <span id='74'>74.搜索二维矩阵</span>
二分查找：
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool: 
        n,m = len(matrix),len(matrix[0])
        import bisect
        row0 = [row[0] for row in matrix]
        row = bisect.bisect_right(row0,target) - 1
        if row < 0:
            return False
        col0 = matrix[row]
        col = bisect.bisect_left(col0,target)
        if col >= m:
            return False
        if matrix[row][col] == target:
            return True
        else:
            return False
```
## <span id='190'>190.颠倒二进制位</span>
位运算：
```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range(32):
            res = (res<<1) | (n&1)
            n >>= 1
        return res
```
## <span id='90'>90.子集 II</span>
回溯：
```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        self.dfs(nums, 0, res, [])
        return res
    
    def dfs(self, nums, index, res, path):
        if path not in res:
            res.append(path)
        for i in range(index, len(nums)):
            self.dfs(nums, i + 1, res, path + [nums[i]])
```
## <span id='1006'>1006.笨阶乘</span>
暴力干：
```python
class Solution:
    def clumsy(self, N: int) -> int:
        tmp = 0
        if N <=4 :
            if  N==4:
                return 7
            elif N==3:
                return 6
            elif N==2:
                return 2
            else:
                return 1
        else:
            tmp += int(N*(N-1)/(N-2))+N-3
        for num in range(N-4,0,-4):
            if num < 3:
                if num == 2:
                    tmp += -num*(num-1)
                else:
                    tmp += -num
            else:
                tmp += -int(num*(num-1)/(num-2))+num-3
        return tmp
```
栈：
```python
class Solution:
    def clumsy(self, N: int) -> int:
        op = 0 
        stack = [N]
        for i in range(N-1,0,-1):
            if op == 0:
                stack.append(stack.pop()*i)
            elif op == 1:
                stack.append(int(stack.pop()/i))
            elif op == 2:
                stack.append(i)
            elif op == 3:
                stack.append(-i)
            op = (op+1)%4
        return sum(stack)
```
## <span id='17.21'>面试题17.21.直方图的水量</span>
动态规划：
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        n = len(height)
        left_max = [height[0]] + [0]*(n-1)
        for i in range(1,n):
            left_max[i] = max(left_max[i-1],height[i])
        right_max = [0]*(n-1)+[height[n-1]]
        for i in range(n-2,-1,-1):
            right_max[i] = max(right_max[i+1],height[i])
        return sum(min(right_max[i],left_max[i])-height[i] for i in range(n))
```
双指针：
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:return 0
        ans = 0
        left,right = 0,len(height)-1
        left_max,right_max = 0,0
        while left < right:
            left_max = max(left_max,height[left])
            right_max = max(right_max,height[right])
            if height[left] < height[right]:
                ans += left_max - height[left]
                left += 1
            else:
                ans += right_max - height[right]
                right -= 1
        return ans
```
## <span id='80'>80.删除有序数组中的重复项 II</span>
双指针：
```
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 1
        i = 2
        for j in range(2,n):
            if nums[j] != nums[i-2]:
                nums[i] = nums[j]
                i += 1
        return i
```
## <span id='81'>81.搜索旋转排序数组 II</span>
原创蛇皮解法：
```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        n = len(nums)
        if n == 1:
            return nums[0]==target
        index = 0
        for i in range(1,n):
            if nums[i] < nums[i-1]:
                index = i
                break
        if not index:
            if target in nums:
               return True
            else:
                return False

        import bisect
        if target > nums[-1]:
            i = bisect.bisect_right(nums[:index],target)
            if i == 0:
                return False
            elif nums[i-1] == target:
                return True
        elif target == nums[-1]:
            return True
        elif target < nums[-1]:
            i = bisect.bisect_right(nums[index:],target)
            if i == 0:
                return False
            elif nums[index:][i-1] == target:
                return True
        return False
```
## <span id='153'>153. 寻找旋转排序数组中的最小值</span>
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l,r = 0,len(nums)-1
        while l < r:
            mid = (l+r) >> 1
            if nums[mid] > nums[r]:
                l = mid + 1
            elif nums[mid] < nums[r]:
                r = mid
        return nums[l]
```
## <span id='154'>154. 寻找旋转排序数组中的最小值 II</span>
二分查找：
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        l,r = 0,len(nums)-1
        while l < r:
            mid = (l+r) >> 1
            if nums[mid] > nums[r]:
                l = mid + 1
            elif nums[mid] == nums[r]:
                # l += 1
                r -= 1
            elif nums[mid] < nums[r]:
                r = mid
        return nums[l]
```
## <span id='179'>179.最大数</span>
自定义排序函数：
```python
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        from functools import cmp_to_key
        n = len(nums)
        if n == 0:
            return ''
        elif n == 1:
            return str(nums[0])
        cmp_key = cmp_to_key(lambda a,b:int(b+a)-int(a+b))
        a = list(map(str,nums))
        a.sort(key=cmp_key)

        res = ''.join(a)
        if res[0] == '0':
            return '0'
        return res
```
## <span id='783'>783.二叉搜索树节点最小距离</span>
中序遍历：
```python
class Solution:
    def minDiffInBST(self, root: TreeNode) -> int:
        tmp = []
        def cc(root):
            if root:
                cc(root.left)
                tmp.append(root.val)
                cc(root.right)
        cc(root)
        return min([tmp[i+1]-tmp[i] for i in range(len(tmp)-1)])
```
## <span id='87'>87. 扰乱字符串</span>
递归：
```python
class Solution:
    @cache
    def isScramble(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False
        if s1 == s2:
            return True
        if sorted(s1) != sorted(s2):
            return False

        for i in range(1, len(s1)):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]) or \
                    (self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i:], s2[:-i])):
                return True
        return False
```
## <span id='27'>27. 移除元素</span>
双指针：
```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        a = b = 0
        while a < len(nums):
            if nums[a] != val:
                nums[b] = nums[a]
                b += 1
            a += 1
        return b
```
## <span id='377'>377. 组合总和 Ⅳ</span>
动态规划:
```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1] + [0]*target
        for i in range(1,target+1):
            for num in nums:
                if num <= i:
                    dp[i] += dp[i-num]
        return dp[target]
```
## <span id='897'>897. 递增顺序搜索树</span>
```python
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        def sort(root,ls):
            if not root:
                return 
            if root.left:
                sort(root.left,ls)
            ls.append(root.val)
            if root.right:
                sort(root.right,ls)
        res = []
        sort(root,res)
        ans = cur = TreeNode(None)
        for i in res:
            cur.val = i
            if i == res[-1]:
                break
            cur.right = TreeNode(None)
            cur = cur.right
        return ans
```
## <span id='1011'>1011. 在 D 天内送达包裹的能力</span>
二分查找:
```python
class Solution:
    def shipWithinDays(self, weights: List[int], D: int) -> int:
        left,right = max(weights),sum(weights)
        while left < right:
            need = 1
            cur = 0
            mid = (left+right) // 2
            for i in weights:
                if cur + i> mid:
                    need += 1
                    cur = 0
                cur += i
            if need <= D:
                right = mid
            else:
                left = mid + 1
        return left
```
## <span id='938'>938. 二叉搜索树的范围和</span>
广度优先：
```python
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        q = collections.deque([root])
        res = 0
        while q:
            node = q.popleft()
            if not node:
                continue
            if node.val > high:
                q.append(node.left)
            elif node.val < low:
                q.append(node.right)
            else:
                res += node.val
                q.append(node.left)
                q.append(node.right)
        return res
```
深度优先：
```python
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        if root.val > high:
            return self.rangeSumBST(root.left,low,high)
        if root.val < low:
            return self.rangeSumBST(root.right,low,high)
        return root.val + self.rangeSumBST(root.left,low,high) + self.rangeSumBST(root.right,low,high)
```
## <span id='633'>633. 平方数之和</span>
双指针:
```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        low = 0
        high = int(c**.5)
        while low <= high:
            res = low**2 + high**2
            if res == c:
                return True
            if res > c:
                high -= 1
            else:
                low += 1
        return False
```
## <span id='403'>403. 青蛙过河</span>
记忆化回溯 or dfs+记忆化：
```python
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        @lru_cache(None)
        def dfs(pos,step):
            if pos == stones[-1]:
                return True
            for d in [-1,0,1]:
                if step + d > 0 and pos + step + d in set(stones):
                    if dfs(pos+step+d,step+d):
                        return True
            return False
        pos,step = 0,0
        return dfs(pos,step)
```
## <span id='872'>872. 叶子相似的树</span>
dfs:
```python
class Solution:
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        def cc(root):
            if not root.left and not root.right:
                yield root.val
            if root.left:
                yield from cc(root.left)
            if root.right:
                yield from cc(root.right)
        res1 = list(cc(root1)) if root1 else list()
        res2 = list(cc(root2)) if root2 else list()
        return res1 == res2
```
## <span id='1269'>1269. 停在原地的方案数</span>
动态规划:
```python
class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        maxCols = min(arrLen-1,steps)
        mod = 10**9 + 7
        dp = [[0]*(maxCols+1) for _ in range(steps+1)]
        dp[0][0] = 1
        for i in range(1,steps+1):
            for j in range(0,maxCols+1):
                dp[i][j] = dp[i-1][j]
                if j - 1 >= 0:
                    dp[i][j] = (dp[i][j] + dp[i-1][j-1]) % mod
                if j + 1 <= maxCols:
                    dp[i][j] = (dp[i][j] + dp[i-1][j+1]) % mod               
        return dp[steps][0]
```
## <span id='401'>401. 二进制手表</span>
穷举:
```python
class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        tabel = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
        res = []
        for i in range(60):
            tabel[bin(i).count('1')].append(i)
        for h in range(12):
            key = turnedOn - bin(h).count('1')
            if key < 0 or key > 6:continue
            for minute in tabel[key]:
                res.append(str(h) + ':' + ('0' if minute < 10 else '') + str(minute))
        return res
```
## <span id='JZ001'>剑指 Offer II 001. 整数除法</span>
一般机考不会考这种，位运算复习，<<1 表示乘以2^0
```python
class Solution:
    def divide(self, a: int, b: int) -> int:
        
        flag = -1 if (a>0) ^ (b>0) else 1
        a,b = abs(a),abs(b)

        def cal(x,y):
            n = 1 
            while x > y << 1:
                y <<= 1
                n <<= 1
            return n,y
        res = 0
        while a>=b:
            t,z = cal(a,b)
            res += t
            a -= z
        res *= flag
        return res if res < 2**31 else res-1
```
## <span id='JZ008'>剑指 Offer II 008. 和大于等于 target 的最短子数组</span>
滑动窗口法，很经典
```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        total = 0
        start,end = 0,0
        length = len(nums)
        res = inf
        while end < length:
            total += nums[end]
            while total >= target:
                res = min(res,end-start+1)                
                total -= nums[start]
                start += 1
            end += 1
        return 0 if res==inf else res 
```
## <span id='JZ010'>剑指 Offer II 010. 和为 k 的子数组</span>
前缀和的方法，简单来说就是用一个dict来存累计
```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        tmp = {0:1}
        a = b = 0
        for i in nums:
            a += i
            b += tmp.get(a-k,0)
            tmp[a] = tmp.get(a,0)+1
        return b
```
## <span id='JZ014'>剑指 Offer II 014. 字符串中的变位词</span>
动态维护数组，并且用ord值存字母，比用排列组合省多了。
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        arr1, arr2, lg = [0] * 26, [0] * 26, len(s1)
        if lg > len(s2):
            return False

        for i in range(lg):
            arr1[ord(s1[i]) - ord('a')] += 1
            arr2[ord(s2[i]) - ord('a')] += 1

        for j in range(lg, len(s2)):
            if arr1 == arr2:
                return True
            arr2[ord(s2[j - lg]) - ord('a')] -= 1
            arr2[ord(s2[j]) - ord('a')] += 1
        return arr1 == arr2
```
