# leetcode_solutions
* [424.替换后的最长重复字符](#替换后的最长重复字符)
## 替换后的最长重复字符
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
