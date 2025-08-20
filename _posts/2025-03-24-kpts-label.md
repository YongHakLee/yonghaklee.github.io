---
title: Keypoint Detection Labeling with labelme
date: 2025-03-24 18:40:48 +0900
categories: [Project, Keypoint Detection]
tags: [project, manual, keypoint detection, labeling]     # TAG names should always be lowercase
---

## 01. 기본 설명

Keypoint Detection
: 이미지 내 특정 위치를 찾아 좌표를 추출하는 것이다.
우리는 특정 물체 이미지를 딥러닝 모델에 입력하면,
자동으로 특정 위치의 좌표를 추출할 수 있게 만들고 싶다.
딥러닝 모델이 자동으로 찾아내기 위해서는
딥러닝 모델이 학습할 수 있도록 레이블링을 해야 한다.
<br>

레이블링
: (이미지 처리에 대해서는)
모델에게 우리가 원하는 정답을 알려주기 위해 이미지에
특정 좌표를 찍거나 영역을 지정하는 등의 작업을 수행하고,
이를 파일로 저장하는 일련의 행위이다.
모델과 데이터, 원하는 정답의 종류에 따라 레이블링 또한 달라질 수 있다.
이 가이드는 우리가 사용하는 Keypoint Detection 모델을
학습시키기 위한 레이블링 방법을 설명한다.

## 02. labelme 준비

1. Anaconda를 설치한다.
2. Anaconda Prompt를 실행한다.
3. 가상환경을 생성한다.
```bash
(base) conda create -n env_name python
# env_name은 생성할 가상환경의 이름이다.
# python을 붙이면 가상환경을 생성할 때 Python이 같이 설치된다.
```
1. 가상환경을 활성화한다.
```bash
(base) conda activate env_name
```
1. lableme를 설치한다.
```bash
(env_name) pip install labelme==5.4.1
```
최신 버전은 오류가 발생하는 경우가 많기 때문에
`2025-03-27` 기준으로 `5.4.1` 버전을 사용하고 있다.
`pip install package_name==version_number` 형식으로
특정 버전을 설치할 수 있다.
1. labelme를 실행한다.
```bash
(env_name) labelme
```

.














