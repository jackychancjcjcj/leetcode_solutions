# leetcode_solutions
[![Author](https://img.shields.io/badge/Author-CJ-red.svg "Author")]
[![LICENSE](https://img.shields.io/github/license/JoeyBling/hexo-theme-yilia-plus "LICENSE")]
[![Language](https://img.shields.io/badge/Language-python3.6-green.svg "Laguage")]
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
