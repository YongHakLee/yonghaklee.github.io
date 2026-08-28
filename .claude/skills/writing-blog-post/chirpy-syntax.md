# Chirpy 문법 레퍼런스

이 블로그에서 **쓰지 않는 것**: 커버 이미지(`image:`), `pin: true`, 본문 목차.
목차는 Chirpy 가 사이드바에 자동 생성한다.

원본 데모: `_drafts/2019-08-08-text-and-typography.md`

---

## prompt 박스

`tip` / `info` / `warning` / `danger` 네 가지. markdownlint 가 `{: ... }` 를 오해하므로
주석으로 감싼다.

```markdown
<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> 참고할 내용을 여기 쓴다.
{: .prompt-info }
<!-- markdownlint-restore -->
```

---

## 코드블록에 파일명 라벨 붙이기

````markdown
```sass
@import "colors/light-typography";
```
{: file='_sass/jekyll-theme-chirpy.scss'}
````

---

## 경로 강조

```markdown
설정은 `/etc/systemd/system/foo.service`{: .filepath} 에 둔다.
```

---

## 정의 리스트

용어를 정의할 때 쓴다. 용어 다음 줄에 `: ` 로 시작한다.

```markdown
`SetUID`
: 파일이 실행되는 동안 파일 소유자의 권한으로 실행되도록 한다.

`SetGID`
: 파일이 실행되는 동안 파일 소유 그룹의 권한으로 실행되도록 한다.
```

---

## 각주

```markdown
본문에서 각주를 단다[^fn-1].

[^fn-1]: 각주 내용. 문서 맨 아래에 모아 둔다.
```

---

## 이미지

이미지는 `/assets/img/posts/<시리즈>/` 아래에 둔다. `width`/`height` 를 지정해야
레이아웃이 밀리지 않는다.

기본 (캡션은 이미지 바로 다음 줄의 기울임):

```markdown
![CPU 사용률 그래프](/assets/img/posts/linux/systemd-status.png){: width="972" height="589" }
_systemctl status 출력_
```

폭 조절과 정렬 — `.w-75` `.w-50` (폭), `.normal` (왼쪽 정렬), `.left` `.right` (텍스트 흘림),
`.shadow` `.rounded-10` (그림자·둥근 모서리):

```markdown
![설명](/assets/img/posts/foo/bar.png){: width="972" height="589" .w-75 .normal}
```

다크/라이트 모드용 이미지 쌍:

```markdown
![라이트 모드](/assets/img/posts/foo/light.png){: .light .w-75 .shadow .rounded-10 w='1212' h='668' }
![다크 모드](/assets/img/posts/foo/dark.png){: .dark .w-75 .shadow .rounded-10 w='1212' h='668' }
```

여러 장 가로 배치 (cs231n 포스트가 쓰는 형태):

```html
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;align-items:start">
<img src="/assets/img/posts/foo/a.png" alt="설명" width="273" height="250" style="width:100%">
<img src="/assets/img/posts/foo/b.png" alt="설명" width="250" height="248" style="width:100%">
<img src="/assets/img/posts/foo/c.png" alt="설명" width="250" height="249" style="width:100%">
<em style="grid-column:1/-1">세 장에 대한 공통 캡션.</em>
</div>
```

---

## 수식

frontmatter 에 `math: true` 가 필요하다. MathJax 를 쓴다.

인라인은 `$$...$$` 로 감싼다. 블록은 빈 줄로 띄운다.

```markdown
$$a \ne 0$$ 일 때 $$ax^2 + bx + c = 0$$ 의 해는 다음과 같다.

$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$
```

번호를 붙이고 참조하려면:

```markdown
$$
\begin{equation}
  \sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6}
  \label{eq:series}
\end{equation}
$$

식 \eqref{eq:series} 에서 보듯이 ...
```

---

## mermaid

frontmatter 에 `mermaid: true` 가 필요하다.

````markdown
```mermaid
graph LR
  A[systemctl start] --> B[unit 로드]
  B --> C[ExecStart 실행]
```
````

---

## 유튜브 임베드

```markdown
{% include embed/youtube.html id='Balreaj8Yqs' %}
```

---

## 링크

CI 가 `htmlproofer` 로 링크와 앵커를 검사한다. 깨진 내부 링크는 빌드를 실패시킨다.

```markdown
[다른 포스트](/posts/파일명-슬러그/)
<https://example.com>
```
