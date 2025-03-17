---
title: 원격 개발을 위한 기본 안내
date: 2025-03-12 17:57:40 +0900
categories: [Lab]
tags: [lab]     # TAG names should always be lowercase
---

## 들어가며

---

통계적 인공지능 연구실은 총 5대의 서버를 보유 및 활용하고 있다. 하지만 모든 사람이 모든 서버를 자유롭게 사용할 수 없다. 따라서 배정받은 서버를 고정적으로 사용하되, 특별한 사정이 있을 경우 상담하길 바란다. 서버를 사용하는 기간 동안에는 반드시 ***서버 사용과 관련된 시트([이용학 연구원](/posts/profile)에게 주소 문의)***에 사용 정보를 기록한다. 또한, 연구 이외 개인적인 목적으로 사용을 금지한다.<br><br>
아쉽게도 대부분의 설명은 윈도우 운영체제에 맞게 서술되어 있다. 맥을 사용하는 연구원이면서, 설명이 맞지 않을 경우 ***"큰 주제 맥"*** 등의 형식으로 구글에 검색하여 활용 방법을 익히면 된다.<br><br>
만약 안내에 따라 실습하는 도중 문제가 발생하거나 궁금한 것이 생긴다면, ChatGPT 등의 챗봇을 활용하여 문제를 해결하거나 궁금증을 해소하는 것도 좋은 방법이다. 

> 연구를 위해 원격 개발이 필요한 경우 반드시 [이용학 연구원](/posts/profile)에게 문의하세요.
{: .prompt-danger }

---

## 서버의 기본 정보

---

***서버 사용과 관련된 시트***를 기준으로 한다. 해당 시트는 지속적으로 업데이트할 계획이며, 서버를 활용하기 위해 기본적으로 알아야 할 정보들이 있다.

서버명
: 서버를 구분하기 위해 붙인 이름이다.

IP
: 클라이언트 PC(본인 PC)로 서버 PC에 원격으로 접속하기 위한 서버 PC의 IP Host 주소이다. (XXX.XXX.XXX.XXX)

SSH Port
: 서버에 원격으로 접속하는 방법 중 하나인 SSH로 접속할 때 필요한 포트 번호이다. SSH Port는 기본적으로 22를 사용한다.

SSH ID, PW
: SSH로 서버에 원격으로 접속했을 때, 로그인이 필요하다. 그때의 계정 이름과 비밀번호이다.

Jupyter Port
: Python 코드를 편집하고 실행하기 위한 유용한 툴인 Jupyter에 접속하기 위한 포트 번호이다. 웹 브라우저 (크롬, 엣지 등) 주소창에 `서버 IP:Port`를 입력하여 접속한다. 예를 들어, IP가 `123.123.123.123`인 서버의 `1234` Port에 Jupyter가 연결되어 있는 경우, 외부에서 해당 서버의 Jupyter에 접속하기 위해서는 웹 브라우저를 열고 주소창에 `123.123.123.123:1234`를 입력하면 된다. 접속했을 때 비밀번호를 입력하라고 나오는 경우 Jupyer PW를 입력하면 된다.

Ubuntu
: 서버의 운영 체제(OS)이며, 리눅스(Linux) 배포판 중 하나인 Ubuntu를 사용하고 있다. 서버에 설치된 Ubuntu 버전을 의미한다.

CUDA, cuDNN
: 그래픽카드(GPU)를 활용한 연산을 위해 필요한 툴이며, 이들의 버전을 의미한다. CUDA와 cuDNN 둘 다 버전이 있으며, CUDA 버전이 가장 중요한 것으로 알고 있으면 된다. 예를 들어, GPU 연산이 필요한 딥러닝 프레임워크 `pytorch`를 설치 할 때 CUDA버전을 고려해야 한다. 

사용목적
: 각 서버에 모두 GPU를 활용하기 위한 설정이 되어 있지만, 같은 서버에서 여러 명이 동시에 작업하는 경우 GPU 용량이 부족해질 수 있다. 따라서 겹치지 않도록 각 서버의 사용목적을 기록하고 있다.

---

## Linux와 Ubuntu

---

서버는 운영체제 중 하나인 리눅스(Linux) 배포판 중 하나인 Ubuntu를 사용하고 있다. 리눅스는 윈도우의 cmd와 같이 커맨드를 입력하여 명령하는 것에 최적화되어 있다. terminal을 통해 커맨드를 입력하며, SSH 방식으로 원격으로 서버에 접속하는 것은 서버에서 직접 terminal을 실행하는 것과 거의 동일하다. 명령어가 궁금할 경우 구글링 또는 ChatGPT 등의 챗봇을 활용하면 쉽게 해결할 수 있다.<br>

간단한 명령어 소개
: `pwd` : 현재 디렉토리 위치 확인
: `ls` : 현재 디렉토리 내 파일 목록 확인
: `mkdir 폴더이름` : 디렉토리 생성
: `cd 경로` : 디렉토리 이동
: `cd ..` : 상위 디렉토리로 이동

> 리눅스에서는 `Tab`키를 눌러 명령어를 자동으로 완성시킬 수 있다.
{: .prompt-tip }

---

## Anaconda

---

서버를 SSH 방식으로 원격 접속하여 로그인하면 터미널 커맨드에 `(base)`가 앞에 붙어 있는데, 이는 아나콘다의 가상환경 이름이다. 주로 파이썬을 활용한 원격 개발을 위해서 서버를 사용하기 때문에 아나콘다를 기본으로 설정해놓은 것이다. 아나콘다는 파이썬 패키지 관리를 편리하게 도와준다. 예를 들어, 다음과 같이 두 개의 딥러닝 모델을 개발하고자 하는데, 필요한 패키지 버전이 다르다고 해보자.

- A 모델 : OO패키지 4.4버전 필요
- B 모델 : OO패키지 2.1버전 필요

파이썬 패키지는 보통 하나의 버전만 사용할 수 있기 때문에, 가상환경 단위로 관리하지 않으면 매번 패키지를 삭제했다가 설치하는 과정을 거쳐야 할 것이다. 따라서 가상환경 C에는 OO패키지 4.4버전을 설치하여 A 모델을 개발하고, 가상환경 D에는 OO패키지 2.1버전을 설치하여 B 모델을 개발하면 된다.<br>

간단한 명령어 소개
: `conda create -n 가상환경이름` : 가상환경 생성
: `conda activate 가상환경이름` : 가상환경 활성화
: `conda deactivate` : 가상환경 비활성화
: `conda install 패키지이름` : 패키지 설치
: `conda remove 패키지이름` : 패키지 삭제
: `conda list` : 가상환경에 설치된 패키지 목록 확인
: `conda env list` : 가상환경 목록 확인
: `conda remove -n 가상환경이름 --all` : 가상환경 삭제

가상환경이 활성화되면 해당 가상환경 이름이 제일 앞 `()` 에 들어간다. 아나콘다를 처음 실행하면 `base` 가상환경이 활성화된다. SSH를 통해 서버에 원격접속할 경우 `(base)` 상태이다. 가상환경이 활성화된 상태에서 파이썬 패키지를 설치해야 해당 가상환경에 파이썬 패키지가 설치된다.

> `(base)` 가상환경 상태에서 파이썬 패키지를 절대 설치하지 말자.
{: .prompt-danger }

> 서버를 활용하기 전에 로컬 운영체제(본인 PC)에 아나콘다를 설치하여 연습해보는 것을 권장한다.
{: .prompt-warning }

> 파이썬 패키지 설치 및 제거는 아나콘다보다 pip을 더 많이 사용한다. `pip install 패키지이름` 명령어를 통해 패키지를 설치할 수 있고, `pip uninstall 패키지이름` 명령어를 통해 패키지를 제거할 수 있다.
{: .prompt-tip }

---

## 서버 접속 맛보기

---

SSH(Secure Shell)는 네트워크를 통해 다른 컴퓨터에 접속하기 위한 프로토콜이다. SSH는 주로 원격 서버에 안전하게 접근하고, 명령을 실행하고, 파일을 전송하는 데 사용된다. SSH는 보안이 중요시되는 환경에서 특히 유용하며, 데이터 전송 과정에서의 도청 및 변조를 방지하기 위해 암호화를 사용한다.<br>

SSH는 파일 전송 프로토콜인 SCP(Secure Copy)와 SFTP(Secure File Transfer Protocol)를 지원한다. 이를 통해 안전하게 파일을 업로드하거나 다운로드할 수 있다.

### [MobaXterm](https://mobaxterm.mobatek.net/)
SSH를 지원하는 프로그램이면서, 파일 조회와 전송 또한 편리하게 도와준다. (맛보기 또는 임시로 사용하길 권장하며, 웬만한 터미널 접속 및 작업은 VS Code로도 가능하다.) ([참고페이지](https://backendcode.tistory.com/270))

#### 사용법

1. 한글 인코딩 설정
- [Settings] → [Terminal] → [Default font settings]
  - Font chatset : DEFAULT (System)
  - Term charset : eucKR (Korean)
2. SSH 접속
- [Session] → [SSH]
  - Remote Host : 서버 IP
  - Port : 기본 22
  - Bookmark settings → Session name : 북마크 이름
  - 왼쪽 디렉토리 : 접속한 서버PC와 파일 주고받기 가능

---

## 서버의 저장공간

---

총 5대의 서버가 있는데, 각각의 서버에 장착된 저장공간을 사용한다면 용량이 충분하지 않다.
따라서 우리는 외부 저장장치인 NAS를 활용하여 모든 서버에서 NAS를 사용할 수 있도록 설정해놓았다.
쉽게, 모든 서버에 동일한 외장하드를 연결해놓았다고 이해해도 좋다.<br>
서버 저장공간의 가장 상위 디렉토리는 `/`이고, NAS 저장공간의 디렉토리는 `/mnt/nas`이다.
`ls /mnt/nas4` 명령어를 입력해보면 알 수 있는데,
각자 `/mnt/nas4/lyh`처럼 이니셜로 디렉토리를 만들어 사용하고 있다.

### [FileZilla](https://filezilla-project.org/)
서버를 통해 원격 개발을 주로 하는 경우, 파일 관리 또한 서버의 저장공간에서 하게 될 것이다.
그리고, 로컬 PC의 저장공간과 서버 PC의 저장공간에 있는 파일들을 서로 주고받아야 하는 경우가 많다.
이러한 작업을 처리하기 위한 방법은 여러 가지가 있지만,
파일 관리 프로그램인 [FileZilla](https://filezilla-project.org/)를 활용하는 것이 편리하다.

#### 사용법

1. FileZilla 다운로드 및 설치
2. FileZilla 실행 → 파일 → 사이트 관리자 → 새 사이트에 서버 정보 입력
   - 프로토콜: SFTP - SSH File Transfer Protocol
   - 호스트: 서버 IP
   - 포트: 22
   - 사용자: 서버 로그인 계정
   - 비밀번호: 서버 로그인 계정 비밀번호

> FileZilla로 서버 중 하나에 접속하여 `/mnt/nas4` 경로로 이동한 뒤, 본인 이니셜 이름의 폴더가 없으면 생성해보자.
{: .prompt-tip }

![FileZilla](/assets/img/lab/01_filezilla.png)

---

## [VS Code (Visual Studio Code)](https://code.visualstudio.com/)

---

IDE (Integrated Development Environment)
: 프로그래밍 작업을 위한 통합 개발 환경이다.
코드 작성, 디버깅, 테스트, 배포 등 개발 과정을 완전히 지원하는 소프트웨어이다.
주로 프로그래밍 언어를 지원하는 프로그램이며, 코드 작성 및 디버깅을 위한 다양한 기능을 제공한다.
터미널을 통한 원격 접속 및 제어, 원격 코드 작성 및 편집, 원격 파일 관리 등을 포함한다.

[VS Code (Visual Studio Code)](https://code.visualstudio.com/)
: 마이크로소프트에서 개발한 무료 소스 코드 편집기이다.
로컬 및 원격 개발을 위한 다양한 IDE가 존재하지만, VS Code 사용을 권장한다.
무료이면서 다양한 확장 프로그램을 지원하여 편리한 개발 환경을 제공한다.

### 원격 개발 세팅 (Remote SSH Setting)

VS Code에서 원격 개발을 하기 위해서는 다음과 같은 준비가 필요하다.

- VS Code 설치
- Extension에서 Remote - SSH 설치

![VS Code](/assets/img/lab/02_vscode_01.png)

- `F1` → `Remote-SSh: Connect to Host..` → `Configure SSH Hosts..`

![VS Code](/assets/img/lab/02_vscode_02.png)

- 여러 개 나올 경우 `~~\.ssh\config` 또는  `~~/.ssh/config` 선택
- 파일이 하나 열리는데, 아래 예시와 같이 입력하고 저장한다.

```text
# 서버명01
Host 서버명01
    HostName 서버명01의 IP
    User 계정명
    Port 22
    
# 서버명02
Host 서버명02
    HostName 서버명02의 IP
    User 계정명
    Port 22

# 서버명03
Host 서버명03
    HostName 서버명03의 IP
    User 계정명
    Port 22

# 서버명04
Host 서버명04
    HostName 서버명04의 IP
    User 계정명
    Port 22

# 서버명05
Host 서버명05
    HostName 서버명05의 IP
    User 계정명
    Port 22
```

- 이제 `F1`을 누르고, `Remote-SSH: Connect to Host..`를 누르면,
`서버명01 ~ 서버명05` 목록이 나오는데, 선택해서 접속하면 되고, os는 Linux로 선택한다.
위 파일 예시에서 IP, 계정명, 비밀번호는 [이용학 연구원](/posts/profile)에게 문의하면 된다.
어떤 서버를 사용할 것인지 또한 [이용학 연구원](/posts/profile)에게 문의하면 된다.
- 특정 directory에 접속해야 하면 open folder를 누르고 입력하면 된다.

---

## GPU & CUDA & cuDNN

우리가 서버를 사용하는 이유 중 하나는 GPU 사용 때문이다.<br>
서버의 GPU 정보 조회는 `nvidia-smi` 명령어를 사용한다.

```bash
nvidia-smi
```

```bash
Mon Mar 17 17:24:52 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080 Ti     Off |   00000000:09:00.0 Off |                  N/A |
|  0%   52C    P8             44W /  350W |      25MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3080 Ti     Off |   00000000:0A:00.0 Off |                  N/A |
|  0%   43C    P8             23W /  370W |      11MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2141      G   /usr/lib/xorg/Xorg                              9MiB |
|    0   N/A  N/A      2397      G   /usr/bin/gnome-shell                            6MiB |
|    1   N/A  N/A      2141      G   /usr/lib/xorg/Xorg                              4MiB |
+-----------------------------------------------------------------------------------------+
```

GPU 정보 읽기
: sailab03 서버의 GPU 정보이다.<br>
NVIDIA GeForce RTX 3080Ti 2개가 있다.<br>
위에서부터 가장 왼쪽에 있는 숫자로 0번, 1번 GPU이다.<br>
0번 GPU는 지금 25MiB를 사용중이다.<br>
만약 현재 0번 GPU의 사용량이 10000Mib 이상이라면<br>
GPU 사용이 필요한 작업을 진행할 경우 Out of Memory 에러가 발생할 확률이 높다.<br>
<br>

GPU 사용이 필요한 Python Script 실행
: Python Script를 실행할 때, terminal에서 `python script.py` 와 같이 입력한다. 그런데 서버에 GPU가 여러 개이면서, GPU가 필요한 작업이 포함된 script라면 특정 GPU만 사용하게 할 수 있다. `CUDA_VISIBLE_DEVICES=GPU번호 python script.py` 와 같이 입력하면 된다.<br>
`CUDA_VISIBLE_DEVICES=0 python script.py`: 0번 GPU만 사용하여 script.py를 실행한다.<br>
`CUDA_VISIBLE_DEVICES=0,1 python script.py`: 0번, 1번 GPU를 사용하여 script.py를 실행한다.<br>
<br>

CUDA 버전 확인
: GPU를 통한 연산에서 중요한 역할을 하는 CUDA Version 정보는 `nvcc -V` 명령어를 사용한다.

```bash
nvcc -V
```

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

- sailab02 서버의 CUDA Version 정보이다.
- 12.1 버전인 것을 확인할 수 있다.

> Pytorch와 같은 딥러닝 프레임워크는 CUDA 버전에 따라 설치 가능한 버전이 제한될 수 있으므로, CUDA 버전에 맞는 버전의 Pytorch를 설치해야 한다.
{: .prompt-warning }

---

