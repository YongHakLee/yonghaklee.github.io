#!/usr/bin/env bash
# _posts/ 포스트의 기계적 관례를 검사한다.
# 사용법: .claude/skills/writing-blog-post/check-post.sh _posts/2026-08-28-foo.md
set -uo pipefail

file="${1:-}"
[ -n "$file" ] || { echo "usage: check-post.sh <post.md>" >&2; exit 2; }
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 2; }

root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
errors=0
err()  { printf 'ERROR: %s\n' "$*"; errors=$((errors + 1)); }
warn() { printf 'WARN:  %s\n' "$*"; }

base=$(basename "$file")
parent=$(basename "$(dirname "$(readlink -f "$file")")")

# --- 위치와 파일명 ---
[ "$parent" = "_posts" ] || err "파일이 _posts/ 안에 있어야 한다 (현재: $parent/)"

fname_date=""
if [[ "$base" =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2})-[A-Za-z0-9._-]+\.md$ ]]; then
  fname_date="${BASH_REMATCH[1]}"
else
  err "파일명은 YYYY-MM-DD-슬러그.md 형식이어야 한다 (현재: $base)"
fi

# --- frontmatter 분리 ---
if [ "$(head -n 1 "$file")" != "---" ]; then
  err "파일이 '---' frontmatter로 시작해야 한다"
fi
fm_end=$(awk 'NR>1 && $0 == "---" { print NR; exit }' "$file")
if [ -z "$fm_end" ]; then
  err "frontmatter 를 닫는 '---' 가 없다"
  fm_end=1
fi
fm=$(awk -v e="$fm_end" 'NR>1 && NR<e' "$file")
body=$(awk -v e="$fm_end" 'NR>e' "$file")

# 값에서 YAML 주석을 떼어낸다 (기존 포스트의 tags 뒤 주석에 대문자가 있다)
fm_get() { printf '%s\n' "$fm" | sed -n "s/^$1:[[:space:]]*//p" | head -n 1 | sed 's/[[:space:]]*#.*$//'; }

for key in title date categories tags; do
  printf '%s\n' "$fm" | grep -q "^$key:" || err "frontmatter 에 '$key' 가 없다"
done

# --- date ---
date_val=$(fm_get date)
if [ -n "$date_val" ]; then
  case "$date_val" in
    *+0900*) ;;
    *) err "date 에 +0900 오프셋이 필요하다 (현재: $date_val)" ;;
  esac
  date_day="${date_val%% *}"
  if [ -n "$fname_date" ] && [ "$date_day" != "$fname_date" ]; then
    err "date 의 날짜($date_day)가 파일명 날짜($fname_date)와 다르다"
  fi
fi

# --- tags 소문자 ---
tags_val=$(fm_get tags)
if printf '%s' "$tags_val" | grep -q '[A-Z]'; then
  err "tags 는 전부 소문자여야 한다 (현재: $tags_val)"
fi

# --- categories 2단계 이하 ---
cats_val=$(fm_get categories | sed 's/^\[//; s/\][[:space:]]*$//')
if [ -n "$cats_val" ]; then
  n=$(printf '%s' "$cats_val" | awk -F, '{ print NF }')
  [ "$n" -le 2 ] || err "categories 는 2단계 이하여야 한다 (현재 ${n}개: $cats_val)"
fi

# --- 코드펜스 짝수 ---
fences=$(printf '%s\n' "$body" | grep -c '^[[:space:]]*```') || true
if [ $((fences % 2)) -ne 0 ]; then
  err "코드펜스(\`\`\`) 개수가 홀수다 (${fences}개). 닫히지 않은 코드블록이 있다"
fi

# --- 코드블록 바깥 본문 ---
outside=$(printf '%s\n' "$body" | awk '/^[[:space:]]*```/ { f = !f; next } !f')
# 인라인 코드(`...`)를 뺀 본문. 셸의 `$$` 같은 것이 수식으로 오인되지 않게 한다.
outside_prose=$(printf '%s\n' "$outside" | sed 's/`[^`]*`//g')

if printf '%s\n' "$outside" | grep -q '^# '; then
  err "본문에 H1(# )이 있다. 제목은 frontmatter 의 title 이 담당한다"
fi

if printf '%s\n' "$outside_prose" | grep -q '\$\$'; then
  printf '%s\n' "$fm" | grep -q '^math:[[:space:]]*true' || warn "수식(\$\$)이 있는데 frontmatter 에 'math: true' 가 없다"
fi

if printf '%s\n' "$body" | grep -q '^[[:space:]]*```mermaid'; then
  printf '%s\n' "$fm" | grep -q '^mermaid:[[:space:]]*true' || warn "mermaid 블록이 있는데 frontmatter 에 'mermaid: true' 가 없다"
fi

# --- 참조 이미지 실존 여부 (CI htmlproofer 대비) ---
for p in $(printf '%s\n' "$body" | grep -o '/assets/img/[A-Za-z0-9._/-]*' | sort -u); do
  [ -f "$root$p" ] || warn "참조한 이미지가 없다: $p"
done

if [ "$errors" -gt 0 ]; then
  printf '\n%s: ERROR %d건\n' "$base" "$errors"
  exit 1
fi
printf '\n%s: OK\n' "$base"
