# cs231n 강의 노트 한국어 대역 포스팅 — 설계

- 작성일: 2026-08-25
- 대상 저장소: `yonghaklee.github.io` (Jekyll + jekyll-theme-chirpy)

## 1. 목표

cs231n(Stanford CS231n: Convolutional Neural Networks for Visual Recognition) 강의
노트 12개 페이지를 이 블로그에 한국어 대역 포스트로 옮긴다. 독자가 **원문과 한국어를
나란히 대조하며** 읽을 수 있어야 한다. 한국어로 정착되지 않은 용어는 원어를 그대로
노출해, 번역이 오히려 이해를 방해하지 않게 한다.

### 대상 페이지

| # | 원문 URL slug | 원문 제목 |
| --- | --- | --- |
| 01 | `classification` | Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits |
| 02 | `linear-classify` | Linear Classification: Support Vector Machine, Softmax |
| 03 | `optimization-1` | Optimization: Stochastic Gradient Descent |
| 04 | `optimization-2` | Backpropagation, Intuitions |
| 05 | `neural-networks-1` | Neural Networks Part 1: Setting up the Architecture |
| 06 | `neural-networks-2` | Neural Networks Part 2: Setting up the Data and the Loss |
| 07 | `neural-networks-3` | Neural Networks Part 3: Learning and Evaluation |
| 08 | `neural-networks-case-study` | Putting it Together: Minimal Neural Network Case Study |
| 09 | `convolutional-networks` | Convolutional Neural Networks: Architectures, Convolution / Pooling Layers |
| 10 | `understanding-cnn` | Understanding and Visualizing Convolutional Neural Networks |
| 11 | `transfer-learning` | Transfer Learning and Fine-tuning Convolutional Neural Networks |
| 12 | `rnn` | Recurrent Neural Networks |

Module 1은 01–08, Module 2는 09–11, Appendix는 12에 해당한다.

### 원문 실측치

| 항목 | 수량 |
| --- | --- |
| 총 단어 수 | 56,986 |
| 이미지 | 73개 고유 (참조 75회), 7.6 MB |
| 코드 블록(`<pre>`) | 60 |
| 문단(`<p>`) | 645 (그중 106개는 `<p><a name="..."></a></p>` 형태의 빈 앵커 문단) |
| 리스트 항목(`<li>`) | 308 (그중 20개는 빈 항목) |
| **번역 대상 조각** | **843** |
| figcaption | 60 (그중 2개는 캡션이 비어 있어 번역 대상 58개) |
| 인라인 SVG | 3 |
| iframe | 1 (`convolutional-networks`의 합성곱 데모) |

특수 케이스가 거의 없고 12페이지의 HTML 구조가 일관적이라, 기계적 추출이 안정적으로
동작한다.

## 2. 산출물

### 파일 배치

```
_posts/2026-08-25-cs231n-NN-<원문 slug>.md          12개
assets/img/posts/cs231n/<원문 slug>/<원본 파일명>     73개
```

포스트 파일명은 저장소 관례(`python-04-algorithm-06`)를 따른다. 원문 slug를 파일명에
유지해 어떤 포스트가 어느 URL의 번역인지 파일명만으로 알 수 있게 한다. 이미지는
페이지별 하위 폴더로 나눠, 원본에서 같은 파일명이 여러 페이지에 쓰여도 충돌하지
않게 한다.

### 날짜

`_config.yml`에 `future` 설정이 없어 Jekyll 기본값 `future: false`가 적용된다. 미래
날짜 포스트는 빌드에서 제외되므로 **12개 모두 `2026-08-25`로 두고 시각만 5분씩
증가**시킨다 (01이 `09:00:00`, 12가 `09:55:00`). Chirpy는 날짜 내림차순 정렬이라
목록에서 12→01 순으로 보이며, 이는 기존 알고리즘 시리즈(01이 가장 오래된 날짜)와
동일한 배치다.

### 프론트매터

```yaml
---
title: "01. Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits"
description: 이미지 분류 문제, 데이터 기반 접근법, k-최근접 이웃 분류기, 하이퍼파라미터 튜닝을 위한 train/val/test 분할.
date: 2026-08-25 09:00:00 +0900
categories: [Computer Vision, cs231n]
tags: [study, computer vision, cs231n, deep learning]
math: true
image:
  path: /assets/img/posts/cs231n/classification/classify.png
  alt: The task of image classification.
---
```

- `title`: 원문 영어 제목을 그대로 쓰고 앞에 순번을 붙인다. 목록에서 원본 대응이
  바로 보이고, 영어 용어로도 검색에 잡힌다.
- `description`: 한국어 한 줄 요약. 목록 카드와 SEO 메타에 쓰인다.
- `categories`: `[Computer Vision, cs231n]` — 요청대로 신규 카테고리.
- `tags`: 저장소 관례대로 전부 소문자.
- `math: true`: 12개 전부에 넣는다. `understanding-cnn`, `transfer-learning`은 원문에
  수식이 없지만 통일해두면 나중에 내용을 보탤 때 걸리지 않는다.
- `image`: 해당 페이지의 첫 번째 그림을 미리보기로 지정한다. 그림이 없는
  `optimization-2`, `transfer-learning`은 생략한다.
- `toc`는 `_config.yml`의 posts 기본값에 이미 `true`로 있으므로 넣지 않는다.

## 3. 포스트 본문 포맷

포맷은 `_drafts/2019-08-08-text-and-typography.md`(Chirpy 기능 데모)에서 확인한
문법만 사용한다. 테마·CSS 수정 없이 순수 마크다운으로 성립한다.

### 3.1 구분 원칙

독자가 어디까지가 cs231n이고 어디부터가 추가한 내용인지 혼동하면, 원문 대조라는
목적 자체가 무너진다. 규칙은 하나다.

> **인용 블록 안 = 원문. 그 아래 일반 문단 = 번역. `.prompt-*` 박스와 `### 보충:`
> 섹션 = 추가한 내용.**

이 규칙을 각 포스트 상단 출처 블록에 명시해, 독자가 처음부터 알고 읽게 한다.

### 3.2 상단 출처 블록

각 포스트는 원문 정보로 시작한다. cs231n 강의 노트는 MIT 라이선스이며, 번역물임을
밝히는 것은 필수다.

```markdown
> **원문**: [Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits](https://cs231n.github.io/classification/)
> — CS231n: Convolutional Neural Networks for Visual Recognition (Stanford University, MIT License)
>
> 원문을 문단 단위로 인용하고 그 아래에 한국어 번역을 붙였다.
> 인용 블록이 원문, 그 아래 문단이 번역이며, `역주` 박스와 `보충` 섹션은
> 원문에 없는 추가 내용이다.
{: .prompt-info }
```

### 3.3 문단

원문을 `>` 인용 블록으로, 번역을 바로 아래 일반 문단으로 놓는다. Chirpy에서 인용
블록은 왼쪽 세로줄과 흐린 배경으로 렌더링되어, 스크롤하며 원문과 번역이 시각적으로
즉시 구분된다.

```markdown
> **Motivation**. In this section we will introduce the Image Classification
> problem, which is the task of assigning an input image one label from a fixed
> set of categories.

**동기(Motivation).** 이 절에서는 **이미지 분류(image classification)** 문제를
소개한다. 고정된 카테고리 집합에서 레이블 하나를 입력 이미지에 할당하는 작업이다.
```

리스트는 **항목별로 쪼개지 않고 리스트 전체**를 하나의 인용 블록에 넣은 뒤, 번역
리스트를 그 아래에 둔다. 항목마다 원문·번역을 번갈아 놓으면 리스트의 구조가 읽히지
않는다.

```markdown
> - **Viewpoint variation**. A single instance of an object can be oriented in
>   many ways with respect to the camera.
> - **Scale variation**. Visual classes often exhibit variation in their size.

- **시점 변화(viewpoint variation).** 같은 물체 하나도 카메라를 기준으로 여러
  방향으로 놓일 수 있다.
- **크기 변화(scale variation).** 시각적 클래스는 크기가 제각각인 경우가 많다.
```

원문 안의 `<blockquote>` 14개는 인용을 중첩(`>>`)해 원문 형태를 유지하고, 번역은
그 아래에 일반 인용(`>`)으로 둔다.

### 3.4 용어 표기 원칙

- 한국어로 정착된 용어는 번역하고 **첫 등장에만** 원어를 병기한다 —
  손실 함수(loss function), 역전파(backpropagation), 과적합(overfitting)
- 번역이 오히려 이해를 방해하는 용어는 **원어 그대로** 쓴다 —
  softmax, dropout, batch normalization, epoch, logit, mini-batch, hinge loss
- 수식 기호와 변수명은 손대지 않는다 — $$W$$, $$x_i$$, `score`
- 코드 안의 식별자와 주석은 **원문 그대로 유지**한다. 코드를 건드리면 원본과의
  대조가 깨진다. 코드 설명은 코드 밖 번역 문단이 담당한다

용어집을 스크래치패드의 `glossary.md`에 두고, 12개 포스트 전체에서 같은 원어가 같은
한국어로 나가도록 강제한다. 01번 포스트를 작성하며 초기 용어집을 만들고, 이후
포스트에서 새 용어가 나올 때마다 추가한다.

### 3.5 이미지와 캡션

```markdown
![The task in image classification](/assets/img/posts/cs231n/classification/classify.png){: width="800" height="333" }
_The task in Image Classification is to predict a single label (or a distribution
over labels as shown here to indicate our confidence) for a given image._

이미지 분류의 과제는 주어진 이미지에 대해 하나의 레이블을(혹은 그림처럼 확신도를
나타내는 레이블 분포를) 예측하는 것이다.
```

- `_이탤릭_` 캡션은 Chirpy가 그림 캡션으로 렌더링한다. 원문 캡션을 여기에 두고,
  번역은 그 아래 일반 문단으로 둔다. figcaption 60개 전부 번역 대상이다.
- `{: width= height= }`를 붙여 레이아웃 시프트(CLS)를 막는다. 값은 다운로드한
  이미지 파일에서 실제 크기를 읽어 채운다.
- 원문의 `fig figleft`(6개) 좌측 정렬은 재현하지 않고 전부 중앙 정렬로 통일한다.
  좁은 화면에서 플로트 정렬은 깨지기 쉽고, 원문 정보 자체는 손실되지 않는다.

### 3.6 코드

원문 그대로 두고 언어 표시만 붙인다.

````markdown
```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')
```
````

원문의 코드 블록 60개는 `language-python` 57개와 `language-plaintext` 3개로 나뉜다
(실측 확인). 감싸는 `<div>`의 클래스에서 언어를 그대로 읽어 쓴다.

### 3.7 수식

원문은 `\( ... \)`(인라인)과 `\[ ... \]`(디스플레이)를 쓴다. kramdown이 `\(`를
이스케이프 문법으로 해석해 백슬래시를 제거해버리므로, 그대로 두면 수식이 깨진다.
따라서 **`$$ ... $$`로 변환**한다. 디스플레이 수식은 앞뒤에 빈 줄을 둔다.

`assets/js/data/mathjax.js`의 설정상 `$..$`도 인라인으로 동작하지만, 이 저장소의
기존 포스트(`2025-12-26-python-04-algorithm-06.md`)가 인라인에도 `$$..$$`를 쓰고
있으므로 그 관례를 따른다. 문서 내 리터럴 `$` 문자가 우연히 수식으로 잡히는 사고도
피할 수 있다.

### 3.8 역주

이해가 막히는 대목 바로 아래에 `.prompt-tip` 박스로 단다.

```markdown
<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **역주.** '불만족(unhappiness)'은 비유적 표현이다. 실제로는 정답 클래스의 점수가
> 낮을수록 커지는 스칼라값이며, 아래에서 SVM 손실과 Softmax 손실 두 가지 형태로
> 구체화된다.
{: .prompt-tip }
<!-- markdownlint-restore -->
```

`markdownlint` 주석은 타이포그래피 데모가 prompt 블록에 쓰는 방식을 따른 것이다.

### 3.9 보충 예제

섹션이 끝나는 지점에 `### 보충: ...` 제목으로 넣는다.

````markdown
### 보충: SVM 손실 직접 계산해보기

원문의 손실 함수 정의를 숫자로 확인해보자.

```python
import numpy as np

scores = np.array([3.2, 5.1, -1.7])   # 원문 예시의 cat / car / frog 점수
correct_class = 0                      # 정답은 cat
margins = np.maximum(0, scores - scores[correct_class] + 1.0)
margins[correct_class] = 0
print(margins.sum())
```

```text
2.9000000000000004
```
````

**예제 코드는 반드시 실제로 실행해 출력을 확인한 뒤 싣는다.** 검증하지 않은 예제는
없느니만 못하다. 실행 환경은 `uv`로 스크래치패드에 만든 numpy 가상환경을 쓴다.

### 3.10 남용하지 않기

역주와 보충은 **이해가 실제로 막히는 곳에만** 넣는다. 페이지당 역주 3–6개, 보충 예제
0–2개를 기준으로 삼되, 원문이 이미 충분히 설명한 곳에는 넣지 않는다.
`transfer-learning`처럼 짧고 명료한 페이지는 역주가 한두 개거나 없을 수 있다. 원문
분량의 절반이 해설이 되면 그것은 번역물이 아니라 다른 글이다.

### 3.11 특수 항목

- **인라인 SVG 3개** (`optimization-2`의 역전파 회로 다이어그램): kramdown이 원시
  HTML을 통과시키므로 SVG를 그대로 삽입한다. 텍스트 라벨이 SVG 안에 영어로 남는데,
  이는 원문 그림 그대로이므로 의도에 부합한다.

  다만 이 SVG들은 `stroke="black"`과 검은색 기본 텍스트를 쓰므로 **Chirpy 다크
  모드에서 배경에 묻혀 보이지 않는다.** 각 SVG를 흰 배경 컨테이너로 감싸 두 테마
  모두에서 읽히게 한다. 인라인 스타일이라 테마 CSS를 건드리지 않는다.

  ```html
  <div style="background:#fff; padding:1rem; border-radius:8px; overflow-x:auto">
    <svg style="max-width: 420px" viewbox="0 0 420 220">...</svg>
  </div>
  ```
- **iframe 1개** (`convolutional-networks`의 합성곱 애니메이션 데모):
  `https://cs231n.github.io/assets/conv-demo/index.html`을 `<iframe>`으로 임베드하고,
  바로 아래에 원본 링크와 한국어 설명을 붙인다. 이 데모는 해당 절의 핵심 설명
  수단이라 링크만 걸면 내용이 비게 된다. 해당 URL은 `X-Frame-Options` 헤더가 없어
  임베드가 가능함을 확인했다.
- **페이지 내 앵커 링크 105개**: 원문 목차가 `#image-classification` 같은 슬러그로
  각 절을 가리키는데, 이 슬러그는 원문 `<h2 id="...">`와 일치한다. 영어 제목을 그대로
  유지하므로 kramdown이 동일한 id를 재생성해 링크가 그대로 동작한다. 원문의
  `<p><a name="intro"></a></p>` 형태의 빈 앵커 문단 106개는 어디에서도 참조되지
  않으므로 버린다. htmlproofer가 내부 해시 링크를 검사하므로 어긋나면 CI에서 잡힌다.
- **상대 링크**: 원문의 `/assignments/`, `/optimization-1/` 같은 사이트 내부 링크는
  절대 URL(`https://cs231n.github.io/...`)로 변환한다. 단, 이 시리즈 안에 대응
  포스트가 있는 링크는 해당 포스트의 사이트 내부 경로로 바꾼다.

## 4. 변환 파이프라인

기계가 잘하는 일(구조 보존)과 사람이 잘하는 일(번역)을 나누고, 그 경계에서 검증이
가능하도록 설계한다. 이미지 73개, 코드 60블록, 수식 540여 개를 손으로 옮기면 누락이
반드시 발생하고 확인할 방법도 없다.

스크립트 세 개는 스크래치패드에 두고 **저장소에 커밋하지 않는다.** 산출물은 포스트와
이미지이며, 일회성 저작 도구가 블로그 저장소에 남을 이유가 없다.

### 4.1 `fetch.py`

12개 페이지 HTML을 받아 캐시하고, 이미지 73개를 `assets/img/posts/cs231n/<slug>/`로
내려받는다. 원본 파일명을 유지해 원문과 대조할 때 어떤 그림인지 바로 찾을 수 있게
한다. 다운로드와 동시에 각 이미지의 실제 픽셀 크기를 읽어(`uv`로 설치한 Pillow)
`{: width= height= }` 속성에 채울 값을 만든다.

### 4.2 `convert.py`

각 페이지의 `<article class="post-content">` … `</article>` 구간을 순회하며 마크다운 스켈레톤을 만든다.

| 원문 요소 | 스켈레톤 출력 |
| --- | --- |
| `<p>` | `>` 인용 원문 + 그 아래 `<!-- KO -->` 슬롯 |
| `<ul>` / `<ol>` | 리스트 **전체**를 하나의 인용 블록으로 + 슬롯 하나 |
| `<li>` | 위 인용 블록 안의 `- ` / `1. ` 항목 |
| `<h2>` / `<h3>` / `<h4>` | `##` / `###` / `####`, 영문 제목 유지 |
| `<img>` | 로컬 경로 + `{: width= height= }`, 슬롯 없음 |
| `figcaption` | `_이탤릭 원문_` + `<!-- KO -->` 슬롯 |
| `<pre>` | 언어 표시 붙인 펜스 코드, 원문 그대로, 슬롯 없음 |
| `<a>` | 마크다운 링크, 상대 URL은 절대 URL로 변환 |
| `<strong>` / `<em>` / `<code>` | `**` / `*` / 백틱 |
| `<blockquote>` | 인용 중첩(`>>`)으로 원문 유지 + 슬롯 (내부 `<p>`는 한 번만 셈) |
| 인라인 SVG | 원시 HTML 그대로 통과 |
| `\(..\)` / `\[..\]` | `$$..$$` |
| HTML 엔티티 | 유니코드 문자로 디코드 |

번역 작업은 이 슬롯을 채우는 것으로 완결된다. 슬롯이 하나라도 남아 있으면 그 문단은
번역이 빠진 것이며, 검증 단계에서 걸린다.

### 4.3 `verify.py`

완성된 포스트를 원문과 대조한다. 다섯 가지를 검사한다.

1. `<!-- KO -->` 잔여 0개 — 번역 누락 검출
2. 원문 인용 조각 수 == 기대치 (전체 **843**) — 문단 누락 검출. 인용 블록 안의
   문단 하나를 1개로, 인용 블록 안의 `- ` / `1. ` 항목 하나를 1개로 센다.

   조각을 세는 규칙은 다음 세 가지다.
   - 태그를 걷어냈을 때 빈 문자열이 되는 `<p>` 106개(`<p><a name="intro"></a></p>`
     형태의 앵커 자리표시자)와 빈 `<li>` 20개는 **제외**한다. 번역할 내용이 없으며
     변환기가 출력에서 버린다
   - `<li>` 안에 중첩된 `<p>`는 **따로 세지 않는다**. `<li><p>…</p></li>`는 화면에
     조각 하나로 보이므로 둘로 세면 기대치가 영원히 맞지 않는다
   - 캡션이 비어 있는 `figcaption` 2개는 제외한다 (그래서 캡션 기대치는 60이 아니라
     **58**이다)

   페이지별 기대치는 아래와 같다. `groundtruth.py`가 변환기와 다른 경로로 센 값이며,
   변환기 출력·독립 기준선·완성 포스트 재계수 셋이 모두 일치함을 확인했다.

   | slug | 조각 수 | slug | 조각 수 |
   | --- | ---: | --- | ---: |
   | `classification` | 73 | `neural-networks-3` | 123 |
   | `linear-classify` | 83 | `neural-networks-case-study` | 50 |
   | `optimization-1` | 63 | `convolutional-networks` | 162 |
   | `optimization-2` | 57 | `understanding-cnn` | 20 |
   | `neural-networks-1` | 73 | `transfer-learning` | 19 |
   | `neural-networks-2` | 72 | `rnn` | 48 |
3. 이미지 참조 수 == 원문 `<img>` 수, **그리고 참조된 파일이 실제로 존재**
4. 코드 블록 수 == 원문 `<pre>` 수
5. `$$` 개수가 짝수 — 수식 구분자 깨짐 검출

**주의:** Chirpy의 prompt 박스도 마크다운 문법상 `>` 인용 블록이다. 이것을 원문
인용으로 잘못 세면 2번 검사가 무의미해진다. `{: .prompt-* }`로 끝나는 인용 블록은
원문 카운트에서 제외한다. 이 구분이 무너지면 검증 전체가 무너지므로, 01번 포스트에서
이 케이스를 반드시 확인한다.

## 5. 검증

### 5.1 렌더링 검증

`verify.py`가 통과해도 Jekyll이 실제로 렌더링하는지는 별개 문제다. 로컬에 bundler와
jekyll이 없다(루비 3.2.3, gem 3.4.20만 존재). 다음 순서로 처리한다.

1. `gem install --user-install bundler` → `bundle config set --local path vendor/bundle`
   → `bundle install`로 로컬 빌드를 시도한다. 이 저장소는 Chirpy 테마 소스 자체
   (`gemspec` 포함)이므로 네이티브 젬 컴파일이 필요하며, 실패할 수 있다.
2. 로컬 빌드에 실패하면 브랜치를 만들어 PR을 올린다. `.github/workflows/ci.yml`이
   `jekyll build`와 `htmlproofer`를 실행한다. htmlproofer는 `--disable-external`로
   내부 링크와 이미지 경로를 전부 검사하므로, 이미지 경로 오류는 여기서 확실히
   잡힌다.

확인 대상은 수식, 이미지와 캡션, 코드 하이라이팅, 인용 블록, prompt 박스, 인라인 SVG,
iframe이 의도대로 렌더링되는지다.

### 5.2 예제 코드 검증

`uv venv` + numpy 환경에서 보충 예제를 전부 실행하고, 실제 출력을 포스트에 싣는다.

## 6. 작업 순서

사용자 요청에 따라 **12개 포스트를 모두 작성한 뒤 일괄 테스트하고, 커밋·푸시한 다음
사용자가 배포된 사이트에서 확인**한다.

1. `fetch.py` / `convert.py` / `verify.py` 작성
2. 이미지 73개 전체 다운로드, 픽셀 크기 추출
3. `01-classification` 작성 — 변환기 출력 형태를 실물로 확인하고 초기 용어집을
   만든다. 여기서 포맷 문제를 잡아 나머지 11개에 같은 실수가 반복되지 않게 한다.
   (사용자 승인 게이트는 두지 않는다)
4. Module 1 나머지(02–08) → Module 2(09–11) → Appendix(12) 순으로 작성
5. `verify.py`를 12개 전부에 대해 통과시킨다
6. 렌더링 검증(5.1)
7. 커밋·푸시 후 사용자 확인

**감수할 위험:** 포맷 검토 게이트가 없으므로, 포맷 판단이 틀렸다면 12개 포스트 전부를
고쳐야 한다. `verify.py`와 렌더링 검증이 기계적 오류는 잡지만, 번역 톤이나 역주 밀도
같은 주관적 판단은 잡지 못한다. 3단계에서 01번을 기준으로 삼아 편차를 줄이고,
푸시 후 지적받은 사항은 12개에 일괄 반영한다.

### 커밋

저장소의 conventional commits 관례를 따른다.

- 이미지 일괄 커밋 1개 — `feat: add cs231n lecture note images`
- 포스트별 커밋 12개 — `feat: add cs231n 01 image classification notes (ko)`

포스트를 하나씩 커밋하면 나중에 특정 포스트만 되돌리기 쉽다.

## 7. 범위 밖

- 저장소 테마·CSS·레이아웃 수정. 순수 마크다운만으로 포맷이 성립하도록 설계했다
- cs231n 과제(assignment) 페이지, 강의 영상, 슬라이드 — 요청 목차에 없다
- 원문의 좌측 플로트 그림 정렬 재현 (3.5 참조)
- 원문 자체의 오류 수정이나 최신 내용 반영. 원문 시점 그대로 옮긴다
