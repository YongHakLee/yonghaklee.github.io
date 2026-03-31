---
title: Python 04. [Algorithm 07] 연결 리스트
date: 2026-03-29 15:12:12 +0900
categories: [Study, Python, Algorithm]
tags: [study, python, algorithm]     # TAG names should always be lowercase
math: true
---

## 연결 리스트 (Linked List)

---

연결 리스트는 데이터 요소의 선형 집합으로, 데이터의 순서가 메모리에 물리적인 순서대로 저장되지는 않는다.
컴퓨터과학에서 배열과 함께 가장 기본이 되는 대표적인 선형 자료구조 중 하나로, 다양한 추상 자료형 (Abstract Data Type, ADT) 구현의 기반이 된다. 

연결 리스트는 배열과는 달리 특정 인덱스에 접근하기 위해서는 전체를 순서대로 읽어야 하므로 상수 시간에 접근할 수 없다. 즉, 탐색은 느리지만 시작 또는 끝 지점에 아이템을 추가하거나 삭제, 추출하는 작업은 빠르다.

## 13 팰린드롬 연결 리스트 

---

[(Leet Code)](https://leetcode.com/problems/palindrome-linked-list)

Q. 연결 리스트가 팰린드롬 구조인지 판별하라.

### 풀이 1 리스트 변환

---

```python
def isPalindrome(self, head: ListNode) -> bool:
    q: List = []

    if not head:
        return True

    node = head
    # 리스트 변환
    while node is not None:
        q.append(node.val)
        node = node.next
    
    # 팰린드롬 판별
    while len(q) > 1:
        if q.pop(0) != q.pop():
            reutrn False

    return True
```

### 풀이 2 데크를 이용한 최적화

---

파이썬의 데크 (Deque)는 이중 연결 리스트 구조로 양쪽 방향 모두 추출하는 데 시간 복잡도 $$O(1)$$ 에 실행된다.

```python
def isPalindrome(self, head: ListNode) -> bool:
    q: Deque = collections.deque()

    if not head:
        return True

    node = head
    while node is not None:
        q.append(node.val)
        node = node.next
    
    # 팰린드롬 판별
    while len(q) > 1:
        if q.popleft() != q.pop():
            reutrn False

    return True
```

### 풀이 3 고 (Go)를 이용한 데크 구현

---

데크가 없고, 복잡함. 코딩 테스트에는 적합하지 않음.

### 풀이 4 런너를 이용한 우아한 풀이

---

생략.

## 14 두 정렬 리스트의 병합

---

[(Leet Code)](https://leetcode.com/problems/merge-two-sorted-lists)

Q. 정렬되어 있는 두 연결 리스트를 합쳐라.

### 풀이 1 재귀 구조로 연결

---

```python
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    if (not l1) or (l2 and l1.val > l2.val):
        l1, l2 = l2, l1
    if l1:
        l1.next = self.mergeTwoLists(l1.next, l2)
    return l1
```
