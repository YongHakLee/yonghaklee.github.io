---
title: Python 04. [Algorithm 02] 자료형
date: 2025-12-17 15:56:32 +0900
categories: [Study, Python, Algorithm]
tags: [study, python, algorithm]     # TAG names should always be lowercase
math: true
---

## 자료형 (Data Types)

리스트 (List)와 딕셔너리 (Dictionary)를 중심으로

---

## 파이썬 자료형 (Python Data Types)

- None (class None type)
- 숫자
  - 실수 (class float)
  - 정수형
    - 정수 (class int)
    - 불리언 (class bool) (논리 자료형)
- 집합형
  - 집합 (class set)
- 매핑
  - 딕셔너리 (class dict)
- 시퀀스
  - 가변
    - 리스트 (class list)
  - 불변
    - 튜플 (class tuple)
    - 문자열 (class str)
    - 바이트 (class bytes)

---

### 숫자 (Numbers)

`bool`은 엄밀히 따지면 논리 자료형인데 파이썬에서는 내부적으로 1(True)과 0(False)으로 처리되는 `int`의 서브 클래스다. `int`는 `object`의 하위 클래스이기도 하기 때문에 결국 다음과 같은 구조를 이룬다.

```text
bool < int < object
```

### 매핑 (Mapping)

키와 자료형으로 구성된 복합 자료형이며, 파이썬에 내장된 유일한 매핑 자료형은 딕셔너리이다.

### 집합 (Set)

`set`은 중복된 값을 갖지 않는 자료형이다.

- 빈 집합 선언

```python
a = set()
a
# set()
type(a)
# <class 'set'>
```

- 값이 포함된 집합

```python
a = {1, 2, 3}
```

`set`은 입력 순서가 유지되지 않으며, 중복된 값이 있을 경우 하나의 값만 유지한다.

---

## 시퀀스 (Sequence)

어떤 특정 대상의 순서 있는 나열

- 불변 (Immutable): `str`, `tuple`, `bytes`
- 가변 (Mutable): `list`

```python
a = 'abc'
id('abc')
# 111
id(a)
# 111
a = 'def'
id(a)
# 222
id('def')
# 222
a[1] = 'd'
# Traceback (most recent call last):
# ...
```

## 객체

| 클래스 | 불변여부 |
| ------ | -------- |
| bool   | 불변     |
| int    | 불변     |
| float  | 불변     |
| list   | 가변     |
| tuple  | 불변     |
| str    | 불변     |
| set    | 가변     |
| dict   | 가변     |

---

### 불변 객체

```python
10
a = 10
b = a
# id(10) = id(a) = id(b)
```

숫자와 문자는 모두 불변 객체이다. 값이 변하지 않기 때문에 `dict`의 키나 `set`의 값으로도 사용할 수 있다. `list`는 언제든 값이 변할 수 있기 때문에 `dict`의 키로 정하거나 `set`의 값으로는 추가할 수 없다.

### 개변 객체

`list`는 값이 바뀔 수 있으며, 다른 변수가 참조하고 있을 때 그 변수의 값 또한 변경된다.

```python
a = [1, 2, 3, 4, 5]
b = a
a[2] = 4
# a와 b는 여전히 같다.
```

### is와 ==

- `is`: 두 변수가 동일한 객체를 참조하는지 확인
- `==`: 두 변수가 동일한 값을 가지는지 확인

- `None`은 값 자체가 정의되어 있지 않아 `is`로 비교해야 한다.

```python
if a is None:
  pass
```

```python
a = [1, 2, 3, 4]
a == a
# True
a is a
# True
a == list(a)
# True
a is list(a) # 값은 동일, 별도의 객체로 복사되어 다른 ID를 가짐
# False
```

```python
a = [1, 2, 3, 4]
a == copy.deepcopy(a)
# True
a is copy.deepcopy(a) # 값은 같지만 ID는 다르다.
# False
```

---

### 속도

단순히 정수형의 덧셈 연산을 하는 경우에도 메모리에서 값을 꺼내 한 번 연산하면 끝인 원시 타입에 비해, 파이썬의 객체는 값을 꺼내는 데도 여러 단계의 부가 작업이 필요하다. (느리다) `Numpy` 같은 라이브러리는 내부적으로 리스트를 원시 타입 배열로 관리하여 속도를 높인다.
