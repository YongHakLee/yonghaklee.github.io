---
title: Python 04. [Algorithm 04] 딕셔너리 (Dictionary)
date: 2025-12-19 20:32:32 +0900
categories: [Study, Python, Algorithm]
tags: [study, python, algorithm]     # TAG names should always be lowercase
math: true
---

## 딕셔너리

키/값 구조이며, 입력 순서가 유지된다. 내부적으로는 해시 테이블 (Hash Table)로 구현되어 있다. 해시할 수만 있다면 숫자, 문자, 집합까지 불변 객체를 모두 키로 사용할 수 있다. 해시 테이블의 주요 연산과 시간 복잡도는 다음과 같다.

**딕셔너리의 주요 연산 시간 복잡도**

|      연산      | 시간 복잡도 | 설명  |
| :------------: | :---------: | :---: |
|    `len(a)`    |  $$O(1)$$   |   -   |
|    `a[key]`    |  $$O(1)$$   |   -   |
| `a[key]=value` |  $$O(1)$$   |   -   |
|   `key in a`   |  $$O(1)$$   |   -   |

파이썬 3.7부터는 딕셔너리의 입력 순서가 유지된다. 하지만 그 이하의 버전일 수도 있으므로 아래와 같은 모듈을 활용하자.
- `collections.OrderedDict()`: 입력 순서가 유지됨
- `collections.defaultdict()`: 조회 시 항상 디폴트 값을 생성해 키 오류를 방지함
- `collections.Counter()`: 요소의 값을 키로 하고 개수를 값 형태로 만들어 카운팅함

---

## 딕셔너리의 활용 방법

```python
# 선언
a = dict()
a = {}

a = {'key1':'value1', 'key2':'value2'}
print(a)
# {'key1':'value1', 'key2':'value2'}

a['key3'] = 'value3'
print(a)
# {'key1':'value1', 'key2':'value2', 'key3':'value3'}
```

리스트에서는 존재하지 않는 인덱스를 조회할 경우 `IndexError`가 발생하며, 딕셔너리에서는 존재하지 않는 키를 조회할 경우 `KeyError`가 발생한다. 다음과 같이 `try` 구문으로 예외 처리를 할 수 있다.

```python
try:
  print(a['key4'])
except KeyError:
  print('존재하지 않는 키')
```

키가 존재하는지 미리 확인해 이후 작업을 진행할 수도 있다.

```python
if 'key4' in a:
  print('존재하는 키')
else:
  print('존재하지 않는 키')
```

딕셔너리에 있는 키/값은 for 반복문으로도 조회가 가능하다. 다음과 같이 `items()` 메소드를 사용하면 된다.

```python
for k, v in a.items():
  print(k, v)

# key1 value1
# key2 value2
# key3 value3

del a['key1']
print(a)
# {'key2':'value2', 'key3':'value3'}
```

## 딕셔너리 모듈

### defaultdict 객체

`defaultdict` 객체는 존재하지 않는 키를 조회할 경우, 에러 메시지를 출력하는 대신 디폴트 값을 기준으로 해당 키에 대한 딕셔너리 아이템을 생성해준다. 

```python
a = collections.defaultdict(int)
a['A'] = 5
a['B'] = 4
a
# defaultdict(<class 'int'>, {'A':5, 'B':4})

a['C'] += 1
a
# defaultdict(<class 'int'>, {'A':5, 'B':4, 'C':1})
```

`KeyError`가 발생하지 않고 디폴트인 0을 기준으로 자동으로 `'C'`를 생성한 후 여기에 1을 더한다.

---

### Counter 객체

`Counter` 객체는 아이템에 대한 개수를 계산해 딕셔너리로 리턴하며, 다음과 같이 사용한다.

```python
a = [1, 2, 3, 4, 5, 5, 5, 6, 6]
b = collections.Counter(a)
b
# Counter({5: 3, 6: 2, 1: 1, 2: 1, 3: 1, 4: 1})
```

키에는 아이템의 값이, 값에는 해당 아이템의 개수가 들어간 딕셔너리를 생성한다. 실제로는 딕셔너리를 한 번 더 래핑 (Wrapping)한 `collections.Counter` 클래스를 갖는다.

```python
type(b)
# <class 'collections.Counter'>
```

- `most_common()`: `Counter` 객체에서 가장 빈도수가 높은 요소를 추출하는 방법 

```python
b.most_common(2)
# [(5, 3), (6, 2)]
```

---

### OrderedDict 객체

입력 그대로 순서가 유지된다.

```python
collections.OrderedDict({
  'banana': 3,
  'apple': 4,
  'pear': 1,
  'orange': 2
})
# OrderedDict([('banana', 3), ('apple', 4), ('pear', 1), ('orange', 2)])
```

