---
title: Python 04. [Algorithm 06] 배열
date: 2025-12-26 20:15:30 +0900
categories: [Study, Python, Algorithm]
tags: [study, python, algorithm]     # TAG names should always be lowercase
math: true
---

## 배열

배열은 값 또는 변수의 집합으로 구성된 구조로, 하나 이상의 인덱스 또는 키로 식별된다.

자료 구조
- 연속 (Contiguous): 메모리 공간 기반
- 연결 (Link): 포인터 기반

배열은 어느 위치에서나 $$O(1)$$에 조회가 가능하다는 장점이 있다.

> **메모리와 포인터** <br>
> 32비트 머신의 포인터는 32비트이며, 64비트 머신의 포인터는 64비트다. 포인터는 메모리 영역을 1바이트 단위로 가리키는 주소인데, 과거 32비트 머신은 메모리 주소를 0에서 $$2^{32}-1$$까지밖에 표현할 수 없었다. 따라서 메모리는 4GB 이상 인식할 수 없는 문제가 있었는데, 최근 64비트 시스템은 $$2^{64}$$, 약 16EB 크기의 가상 메모리를 가리킬 수 있다.

배열은 고정된 크기만큼의 연속된 메모리 할당이지만, 파이썬에서는 동적 배열을 제공한다. 미리 초깃값을 잡아 배열을 생성하고, 데이터가 추가되어 꽉 채워지면 늘려주고 복사한다. 더블링 (Doubling)이라고 하여, 2배씩 늘려주지만 각 언어마다 늘려주는 비율은 상이하다. 파이썬은 초반에는 2배씩 늘려가지만, 전체적으로는 약 1.125배이다. 

---

## 07 두 수의 합

Q. 덧셈하여 타겟을 만들 수 있는 배열의 두 숫자 인덱스를 리턴하라. [(Leet Code)](https://leetcode.com/problems/two-sum/)

---

### 풀이 1 브루트 포스로 계산

브루트 포스 (Brute-Force): 모든 조합을 더해서 일일이 확인해보는 무차별 대입 방식

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
  for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
      if nums[i] + nums[j] == target:
        return [i, j]
```

- 시간복잡도: 약 $$O(n^{2})$$ (지나치게 느림)

---

### 풀이 2 in을 이용한 탐색

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
  for i, n in enumerate(nums):
    complement = target - n

    if complement in nums[i + 1:]:
      return [nums.index(n), nums[i + 1:].index(complement) + (i + 1)]
```

- 시간복잡도: `in`은 $$O(n)$$, 전체는 $$O(n^{2})$$.
- 같은 시간복잡도여도 훨씬 빠르다.

---

### 풀이 3 첫 번째 수를 뺀 결과 키 조회

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
  nums_map = {}
  # 키와 값을 바꿔서 딕셔너리로 저장
  for i, num in enumerate(nums):
    nums_map[num] = i
  
  # 타겟에서 첫 번째 수를 뺀 결과를 키로 조회
  for i, num in enumerate(nums):
    if target - num in nums_map and i != nums_map[target - num]:
      return [i, nums_map[target - num]]
```

---

### 풀이 4 조회 구조 개선

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
  nums_map = {}
  # 하나의 for문으로
  for i, num in enumerate(nums):
    if target - num in nums_map:
      return [nums_map[target_num], i]
    nums_map[num] = i
```

---

### 풀이 5 투 포인터 이용

왼쪽 포인터와 오른쪽 포인터의 합이 타겟보다 크면 오른쪽 포인터를 왼쪽으로, 작다면 왼쪽 포인터를 오른쪽으로 옮기면서 값을 조정하는 방법. 제대로 풀이하려면 정렬이 필요하고, 정렬하면 인덱스가 섞여 풀이가 복잡해진다. 

---

## 08 빗물 트래핑

Q. 높이를 입력받아 얼마나 많은 물이 쌓일 수 있는지 계산하라. [(Leet Code)](https://leetcode.com/problems/trapping-rain-water)

---

### 풀이 1 투 포인터를 최대로 이동

```python
def trap(self, height: List[int]) -> int:
  if not height:
    return 0

  volume = 0
  left, right = 0, len(height) - 1
  left_max, right_max = height[left], height[right]

  while left < right:
    left_max, right_max = max(height[left], left_max), max(height[right], right_max)
    # 더 높은 쪽을 향해 투 포인터 이동
    if left_max <= right_max:
      volume += left_max - height[left]
      left += 1
    else:
      volume += right_max - height[right]
      right -= 1
  
  return volume
```

---

### 풀이 2 스택 쌓기

```python
def trap(self, height: List[int]) -> int:
    stack = []
    volume = 0

    for i in range(len(height)):
      # 변곡점을 만나는 경우
        while stack and height[i] > height[stack[-1]]:
            # 스택에서 꺼낸다
            top = stack.pop()

            if not len(stack):
                break

            # 이전과의 차이만큼 물 높이 처리
            distance = i - stack[-1] - 1
            waters = min(height[i], height[stack[-1]]) - height[top]

            volume += distance * waters

        stack.append(i)

    return volume
```

---

## 09 세 수의 합 

Q. 배열을 입력받아 합으로 0으로 만들 수 있는 3개의 element를 출력하라. [(Leet Code)](https://leetcode.com/problems/3sum/)

---

### 풀이 1 브루트 포스로 계산

- Timeout 가능성이 높다.

---

### 풀이 2 투 포인터로 합 계산

```python
def threeSum(self, nums: List[int]) -> List[List[int]]:
    results = []
    nums.sort()

    for i in range(len(nums) - 2):
        # 중복 건너뛰기
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # 간격을 좁혀가며 합 계산
        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum < 0:
                left += 1
            elif sum > 0:
                right -= 1
            else:
                # sum = 1인 경우이므로 정답 및 스킵 처리
                results.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right -1]:
                    right -= 1
                left += 1
                right -= 1    
    return results
```

> **투 포인터** <br>
> 범위를 좁혀나가기 위해서는 일반적으로 배열이 정렬되어 있는 것이 좋다. 슬라이딩 윈도우와 비슷한 점이 많다.

---

## 10 배열 파티션 I

Q. n개의 페어를 이용한 `min(a,b)`의 합으로 만들 수 있는 가장 큰 수를 출력하라. [(Leet Code)](https://leetcode.com/problems/array-partition-i)

- Input: `[1, 4, 3, 2]`
- Output: `4`
  - `n=2`, `min(1, 2)` + `min(3, 4)` = `4`

---

### 풀이 1 오름차순 풀이

```python
def arrayPairSum(self, nums: List[int]) -> int:
    sum = 0
    pair = []
    nums.sort()

    for n in nums:
        pair.append(n)
        if len(pair) == 2:
            sum += min(pair)
            pair = []

    return sum
```
