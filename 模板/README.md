* [二分查找](#1)
  * [模板一](#1.1)

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
