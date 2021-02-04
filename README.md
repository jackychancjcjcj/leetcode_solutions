# leetcode_solutions
* [424.替换后的最长重复字符](#424)
* [408.滑动窗口中位数](#408)
* [643.子数组最大平均数I](#643-1)
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
