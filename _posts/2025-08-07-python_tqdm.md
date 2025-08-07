---
title: tqdm
date: 2025-08-07 14:34:55 +0900
categories: [Study, Python]
tags: [study, python, tqdm]     # TAG names should always be lowercase
---

## 1. tqdm

tqdm is a library to display progress bar in terminal.

---

## 2. Usage

### 기본 사용법

```python
from tqdm import tqdm

for i in tqdm(range(100)):
    time.sleep(0.1)
```

---

### 바 크기 조절
루프가 실행되는 도중에 터미널 창 크기를 바꾸면 줄바꿈이 일어나면서 출력이 깨져 보이는 현상이 발생

- 동적 너비 조절: `dynamic_ncols=True`

```python
from tqdm import tqdm
import time

for i in tqdm(range(100), 
              desc="동적 너비 조절 테스트", 
              dynamic_ncols=True): # 이 옵션을 추가!
    time.sleep(0.1)
```

- 고정 너비 조절: `ncols=100`

```python
from tqdm import tqdm
import time

for i in tqdm(range(100), 
              desc="고정 너비 조절 테스트", 
              ncols=100): # 이 옵션을 추가!
    time.sleep(0.1)
```

---
