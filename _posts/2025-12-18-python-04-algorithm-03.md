---
title: Python 04. [Algorithm 03] 리스트 (List)
date: 2025-12-18 10:37:32 +0900
categories: [Study, Python, Algorithm]
tags: [study, python, algorithm]     # TAG names should always be lowercase
math: true
---

## 리스트

리스트 (List)
: 순서대로 저장하는 시퀀스, 변경 가능한 목록 (Mutable List). 내부적으로는 동적 배열로 구현되어 있음.

---

## 리스트의 주요 연산 시간 복잡도

| 연산               | 시간 복잡도     | 설명                                                                                                                |
| ------------------ | --------------- | ------------------------------------------------------------------------------------------------------------------- |
| `len(a)`           | $$O(1)$$        | -                                                                                                                   |
| `a[i]`             | $$O(1)$$        | 인덱스 i의 요소 가져오기                                                                                            |
| `a[i:j]`           | $$O(k)$$        | 슬라이스 길이 k                                                                                                     |
| `elem in a`        | $$O(n)$$        | 최악의 경우 전체 탐색                                                                                               |
| `a.count(elem)`    | $$O(n)$$        | 요소 elem의 개수를 반환                                                                                             |
| `a.index(elem)`    | $$O(n)$$        | 요소 elem의 인덱스를 반환                                                                                           |
| `a.append(elem)`   | $$O(1)$$        | 리스트 끝에 요소 elem을 추가                                                                                        |
| `a.pop()`          | $$O(1)$$        | 마지막 요소를 추출. 스택의 연산.                                                                                    |
| `a.pop(0)`         | $$O(n)$$        | 첫 번째 요소를 추출. 전체 복사가 필요함. <br>큐의 연산을 주로 사용하면 리스트보다는<br> deque를 사용하는 것이 좋다. |
| `del a[i]`         | $$O(n)$$        | -                                                                                                                   |
| `a.sort()`         | $$O(n \log n)$$ | Timsort를 사용. 최선의 경우 $$O(n)$$                                                                                |
| `min(a)`, `max(a)` | $$O(n)$$        | -                                                                                                                   |
| `a.reverse()`      | $$O(n)$$        | -                                                                                                                   |

---

## 리스트의 활용 방법

```python
# 선언
a = list()
a = []

a = [1, 2, 3]
a.append(4)
print(a)  # [1, 2, 3, 4]

a.insert(3, 5)
print(a)  # [1, 2, 3, 5, 4]

# 다양한 자료형 혼합 가능
a.append('hello')
a.append(True)
print(a)  # [1, 2, 3, 5, 4, 'hello', True]

# Step
a[1:4:2]
# [2, 5]

del a[1]
print(a)  # [1, 3, 5, 4, 'hello', True]

a.remove(3)
print(a)  # [1, 5, 4, 'hello', True]

a.pop(3)
# 'hello'

print(a)  # [1, 5, 4, True]
```

- 존재하지 않는 인덱스를 조회하면 `IndexError`가 발생

## 리스트의 특징

파이썬은 모든 것이 객체이며, 리스트의 요소 또한 객체이다. 이들 객체에 대한 포인터 목록을 관리하는 형태로 되어 있어서 다양한 자료형을 요소로 삼을 수 있다. 하지만 각 자료형의 크기는 저마다 서로 다르기 때문에 이들을 연속된 메모리 공간에 할당하는 것은 불가능하다. 이처럼 파이썬은 강력한 기능을 위해 속도를 희생한 측면이 있다.

