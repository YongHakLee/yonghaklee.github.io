---
title: Python 04. [Algorithm 05] 문자열 조작
date: 2025-12-20 17:42:31 +0900
categories: [Study, Python, Algorithm]
tags: [study, python, algorithm]     # TAG names should always be lowercase
math: true
---

## 문자열 조작

문자열 조작 (String Manipulation)이란 문자열을 변경하거나 분리하는 등의 여러 과정을 말한다.

---

## 01 유효한 팰린드롬

[Leet Code](https://leetcode.com/problems/valid-palindrome)

팰린드롬 (Palindrome)
: 앞뒤가 똑같은 단어나 문장으로, 뒤집어도 같은 말이 되는 단어 또는 문장이다.

Q. 주어진 문자열이 팰린드롬인지 확인하라. 대소문자를 구분하지 않으며, 영문자와 숫자만을 대상으로 한다.

---

### 풀이 1 리스트로 변환

- 전처리

```python
def isPalindrome(self, s: str) -> bool:
  strs = []
  for char in s:
    if char.isalnum():
      strs.append(char.lower())

  # 팰린드롬 여부 판별
  while len(strs) > 1:
    if strs.pop(0) != strs.pop():
      return False

  return True
```

---

### 풀이 2 데크 자료형을 이용한 최적화

```python
def isPalindrome(self, s: str) -> bool:
  # 자료형 데크로 선언
  strs: Deque = collections.deque()

  for char in s:
    if char.isalnum():
      strs.append(char.lower())

  while len(strs) > 1:
    if strs.popleft() != strs.pop():
      return False

  return True
```

- 리스트의 `pop(0)`: $$O(n)$$, `n`번 반복하면 $$O(n^2)$$
- 데크의 `popleft()`: $$O(1)$$, `n`번 반복하면 $$O(n)$$

---

### 풀이 3 슬라이싱 사용

```python
def isPalindrome(self, s: str) -> bool:
  s = s.lower()
  # 정규식으로 불필요한 문자 필터링
  s = re.sub('[^a-z0-9]', '', s)

  return s == s[::-1] # 슬라이싱
```

파이썬은 문자열을 배열이나 리스트처럼 자유롭게 슬라이싱할 수 있는 좋은 기능을 제공한다. '풀이 2'에 비해 약 2배 정도 더 속도를 높일 수 있다.

---

### 풀이 4 C 구현

C와 같은 컴파일 언어는 파이썬 같은 인터프리터 언어에 비해 더욱 좋은 성능을 낸다.

---

> **문자열 슬라이싱** <br>
> 내부적으로 매우 빠르게 동작한다. `[:]`는 사본을 리턴하며, 값을 참조하는 `a=b`와 다르게 값을 복사한다.

---

## 02 문자열 뒤집기

Q. 문자열을 뒤집는 함수를 작성하라. 입력값은 문자 배열이며, 리턴 없이 리스트 내부를 직접 조작하라. [(Leet Code)](https://leetcode.com/problems/reverse-string)

---

### 풀이 1 투 포인터를 이용한 스왑

```python
def reverseString(self, s, List[str]) -> None:
  left, right = 0, len(s) - 1
  while left < right:
    s[left], s[right] = s[right], s[left]
    left += 1
    right -= 1
```

---

### 풀이 2 파이썬다운 방식

```python
def reverseString(self, s, List[str]) -> None:
  s.reverse()
```

`s = s[::-1]` 도 가능하다. 하지만 변수 할당을 처리하는 데 다소 제약이 있는 경우 `s[:] = s[::-1]`과 같이 사용할 수 있다.

---

## 03 로그 파일 재정렬

Q. 로그를 재정렬하라. 기준은 다음과 같다. [(Leet Code)](https://leetcode.com/problems/reorder-data-in-log-files)

1. 로그의 가장 앞 부분은 식별자다.
2. 문자로 구성된 로그가 숫자 로그보다 앞에 온다.
3. 식별자는 순서에 영향을 끼치지 않지만, 문자가 동일할 경우 식별자 순으로 한다.
4. 숫자 로그는 입력 순서대로 한다.

- 입력
```python
logs = ["dig1 8 1 5 1", "let1 art can", ...]
```

- 출력
```python
logs = ["let1 art can", ...]
```

---

### 풀이 1 람다와 + 연산자를 이용

```python
def reorderLogFiles(self, logs: List[str]) -> List[str]:
  letters, digits = [], []
  for log in logs:
    if log.split()[1].isdigit():
      digits.append(log)
    else:
      letters.append(log)

  # 2개의 키를 람다 표현식으로 정렬
  letters.sort(key=lambda x: (x.split()[1:], x.split()[0]))
  return letters + digits
```

> **람다 표현식** <br>
> 식별자 없이 실행 가능한 함수를 말하며, 함수 선언 없이도 하나의 식으로 함수를 단순하게 표현할 수 있다. 만약 s가 `['2 A', '1 B', '4 C', '1 A']`라면 `sorted()`로 정렬한 결과는 다음과 같다. `['1 A', '1 B', '2 A', '4 C']`. 문자 순 정렬을 원하고, 문자가 동일하면 숫자순으로 정렬되는 형태를 구현하기 위해서는 `key`에 적절한 값을 할당해야 한다. `key=lambda x: (x.split()[1], x.split()[0])`.

---

## 04 가장 흔한 단어

Q. 금지된 단어를 제외한 가장 흔하게 등장하는 단어를 출력하라. 대소문자 구분을 하지 않으며, 구두점 (마침표, 쉼표 등) 또한 무시한다. [(Leet Code)](https://leetcode.com/problems/most-common-word)

---

### 풀이 1 리스트 컴프리헨션, Counter 객체 사용

- 전처리 (Preprocessing)
```python
words = [word for word in re.sub(r'[^\w]', ' ', paragraph)
        .lower().split()
          if word not in banned]
```
  - `\w`: 단어 문자 (Word Character)
  - `^`: not
  - 소문자, 구두점을 제외하고 `banned`를 제외한 단어 목록이 저장된다.

- Counter 객체 활용하기
```python
def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
    words = [word for word in re.sub(r'[^\w]', ' ', paragraph)
          .lower().split()
            if word not in banned]

    counts = collections.Counter(words)
    # 가장 흔하게 등장하는 단어의 첫 번째 인덱스 리턴
    return counts.most_common(1)[0][0]
```

---

## 05 그룹 애너그램

Q. 문자열 배열을 받아 애너그램 단위로 그룹핑하라. [(Leet Code)](https://leetcode.com/problems/group-anagrams)

애너그램
: 문자를 재배열하여 다른 뜻을 가진 단어로 바꾸는 것을 말한다.

---

### 풀이 1 정렬하여 딕셔너리에 추가

```python
def groupAnagrams(self, strs: [List[str]]) -> List[List[str]]:
  anagrams = collections.defaultdict(list)

  for word in strs:
    # 정렬하여 딕셔너리에 추가
    anagrams[''.join(sorted(word))].append(word)
  return list(anagrams.values())
```

---

### 여러 가지 정렬 방법

- `sorted(List)`

```python
a = [2, 5, 1, 9, 7]
sorted(a)
[1, 2, 5, 7, 9]

b = 'zbdaf'
sorted(b)
['a', 'b', 'd', 'f', 'z']

b = 'zbdaf'
"".join(sorted(b))
'abdfz'
```

- `sort()` 유의사항

```python
alist.sort() # sort()는 리스트 자체를 정렬
alist = blist.sort() # sort()는 None을 Return하므로 잘못된 구문
```

- `key` 옵션으로 정렬을 위한 키 또는 함수를 지정할 수 있다.

```python
# 길이 순서대로 정렬
c = ['ccc', 'aaaa', 'd', 'bb']
sorted(c, key=len)
['d', 'bb', 'ccc', 'aaaa']
```

```python
# 함수를 이용한 정렬
a = ['cde', 'cfc', 'abc']
sorted(a, key=labmda s: (s[0], s[-1]))
['abc', 'cfc', 'cde']
```

---

## 06 가장 긴 팰린드롬 부분 문자열

Q. 가장 긴 팰린드롬 부분 문자열을 출력하라. [(Leet Code)](https://leetcode.com/problems/longest-palindromic-substring/)

---

### 풀이 1 중앙을 중심으로 확장하는 풀이
```python
def longestPalindrome(self, s: str) -> str:
  # 팰린드롬 판별 및 투 포인터 확장
  def expand(left: int, right: int) -> str:
    while left >= 0 and right < len(s) and s[left] == s[right]:
      left -= 1
      right += 1
    return s[left + 1:right]

  # 해당 사항이 없을 때 빠르게 리턴
  if len(s) < 2 or s == s[::-1]:
    return s

  result = ''
  # 슬라이딩 윈도우 우측으로 이동
  for i in range(len(s) - 1):
    result = max(result, expand(i, i + 1), expand(i, i + 2), key=len)

  return result
  
```

