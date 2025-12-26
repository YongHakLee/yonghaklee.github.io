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

- 시간복잡도: `in`은 ($$O(n)$$), 전체는 $$O(n^{2})$$.
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



