---
title: Python 03. Syntax
date: 2025-12-16 14:34:55 +0900
categories: [Study, Python]
tags: [study, python]     # TAG names should always be lowercase
---

> 리마인드가 필요한 주요 Syntax 정리
{: .prompt-info}


## Lambda Expression

정의
: 람다 표현식 (Lambda Expression)은 프로그래밍에서 사용되는 **익명 함수 (Anonymous Function)**를 지칭한다. 일반적인 함수(`def`)와 달리 이름을 지정하지 않고, 한 줄로 간결하게 함수를 작성할 때 사용한다.

구조
: `lambda 매개변수: 표현식`

특징
: 
`return` 키워드 없이도 표현식의 결과가 자동으로 반환되고, 주로 `map()`, `filter()`, `sort()`와 같은 함수 내에서 **일회성**으로 기능을 수행해야 할 때 유용함

**예시**

```python
add_lambda = lambda x, y: x + y
print(f"람다 함수 결과: {add_lambda(10, 20)})    # 출력: 30
```

`map()`은 리스트의 모든 요소에 동일한 규칙을 적용하여 새로운 리스트를 만들 때 사용한다.****

```python
prices_usd = [10.5, 20.0, 5.5, 100.0]
prices_krw = list(map(lambda price: int(price * 1400), prices_usd))

print(f"달러 가격: {prices_usd}")
print(f"원화 가격: {prices_krw}")

# 출력 결과
# 달러 가격: [10.5, 20.0, 5.5, 100.0]
# 원화 가격: [14700, 28000, 7700, 140000]
```

`filter()`는 조건이 참(True)인 요소만 남기고 나머지는 걸러낼 때 사용한다.

```python

users = [
    {"name": "Kim", "age": 25, "is_active": True},
    {"name": "Lee", "age": 15, "is_active": True},
    {"name": "Park", "age": 30, "is_active": False},
    {"name": "Choi", "age": 45, "is_active": True}
]

# 나이가 19 이상이고(and), 활성 상태인 유저만 필터링
target_users = list(filter(lambda u: u["age"] >= 19 and u["is_active"], users))

print("타겟 유저 목록:")
for user in target_users:
    print(user)

# 출력 결과
# 타겟 유저 목록:
# {'name': 'Kim', 'age': 25, 'is_active': True}
# {'name': 'Choi', 'age': 45, 'is_active': True}
```

---

## List Comprehension

```python
[n * 2 for n in range(1, 10 + 1) if n % 2 == 1]
# [2, 6, 10, 14, 18]
```

```python
a = {key: value for key, value in original.items()}
```

- 역할별로 줄 구분을 하면 가독성이 높아지고 이해하기 쉬워짐

```python
strls = [
  strl[i:i + 2].lower() for i in range(len(str1) - 1)
  if re.findall('[a-z]{2}', strl[i:i + 2].lower())
]
```

---

## Generator

- 루프의 반복 (Iteration) 동작을 제어할 수 있는 루틴 형태를 말한다. <br>
- `yield` 구문을 사용하여 제너레이터를 리턴할 수 있다. 
- `yield`는 제너레이터가 여기까지 실행 중이던 값을 내보낸다는 의미로, 값을 리턴 후 함수가 종료되지 않는다.

```python
>>> get_natual_number():
  n = 0
  while True:
    n += 1
    yield n

>>> get natural_number()
<generator object get_natural_number at 0x10d3139d0>
```

- 만약 다음 값을 생성하려면 `next()`로 추출하면 된다.

```python
g = get_natural_number()
for _ in range(0, 100):
  print(next(g))

# 1
# 2 
...
# 100
```

- 제너레이터는 여러 타입의 값을 하나의 함수에서 생성하는 것도 가능하다.

```python
def generator():
  yield 1
  yield 'string'
  yield True

g = generator()
g
<generator ...>
next(g)
# 1
next(g)
# 'string'
next(g)
# True
```

---

## range

- 제너레이터의 방식을 활용하는 대표적인 함수
- 생성할 숫자가 매우 많은 경우 생성 조건만 정해두고 나중에 필요할 때 생성해서 꺼내 쓸 수 있음

```python
a = [n for n in range(10000000)]
b = range(10000000)

sys.getsizeof(a)
# 8697464
sys.getsizeof(b)
# 48

b[999]
# 999
```

---

## enumerate

```python
a = [1, 2, 3, 4, 45, 2, 5]
for i, v in enumerate(a):
  print(i, v)

# 0 1
# 1 2
...
```

## // 나눗셈 연산자

- 정수형을 나눗셈할 때 동일한 정수형을 결과로 리턴하면서 내림 (Floor Division) 연산자의 역할
- 몫 (Quotient)을 구하는 연산자

```python
5 // 3
# 1
int(5/3)
# 1
```

- 나머지 (Remainder)를 구하는 모듈로 (Modulo) 연산자는 `%`이며 다음과 같이 사용할 수 있음

```python
5 % 3
# 2
```

- 몫과 나머지를 동시에 구하려면 `divmod()` 함수를 사용
  
```python
divmod(5, 3)
# (1, 2)
```

---

## print

- 리스트를 출력할 때는 `join()`으로 묶어서 처리함

```python
a = ['A', 'B']
print(' '.join(a))
# A B
```

---

## pass

- 먼저 목업 (mockup) 인터페이스부터 구현한 다음에 추후 구현을 진행하여 오류 방지

```python
class Myclass(object):
  def method_a(self):
    pass
  def method_b(self):
    prnit("Method B")

c = MyClass()
```

---

## locals

- `locals()`는 로컬 심볼 테이블 딕셔너리를 가져오는 메소드로, 업데이트 또한 가능

```python
import pprint
pprint.pprint(locals())

# {'nums': [2, 7, 11, 15],
# 'pprint': <module ...>}
# 'self': <__main__....>
# 'target': 9}

```

---

## Google Python Style Guide

- 함수의 기본 값으로 가변 객체 (Mutable Object)를 사용하지 않아야 함
  - 함수가 객체를 수정하면 기본값이 변경되기 때문
  - 기본값으로 `[]`나 `{}`를 사용하는 것은 지양해야 함
- 불변 객체 (Immutable Object)를 사용한다. `None`을 명시적으로 할당하는 것도 좋은 방법

```python
Yes: def foo(a, b=None):
    if b is None:
      b = []
Yes: def foo(a, b: Optional[Sequence] = None):
    if b is None:
      b = []
```

```python
Yes: if not users:
No: if len(users) == 0:

Yes: if foo == 0:
No: if foo is not None and not foo:

Yes: if i % 10 == 0:
No: if not i % 10:
```


