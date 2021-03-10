# leetcode_solutions
![Author](https://img.shields.io/badge/Author-CJ-red.svg "Author")
![LICENSE](https://img.shields.io/github/license/JoeyBling/hexo-theme-yilia-plus "LICENSE")
![Language](https://img.shields.io/badge/Language-python3.6-green.svg "Laguage")
![Last update](https://img.shields.io/badge/last%20update-10%20Mar%202021-brightgreen.svg?style=flat-square "Last update")
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
