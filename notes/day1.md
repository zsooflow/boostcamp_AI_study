# [DAY 1] 파이썬/AI 개발환경 준비하기

### 1. File System & Terminal Basic

#### 컴퓨터 OS

- Operating System, 운영체제
- 우리의 프로그램이 동작할 수 있는 구동 환경
- Software가 Hardware와 연결이 되기 위한 기반
- Application은 OS에 dependent 할 수밖에 없음 
  *ex. exe 파일을 Mac OS에서 사용 불가*

#### 파일 시스템

* OS에서 파일을 저장하는 <b>트리구조</b>의 저장 체계

  ##### 파일과 디렉토리

  * 디렉토리 (Directory)
    - 폴더 또는 디렉토리로 불림
    - 파일과 다른 디렉토리를 포함할 수 있음
  * 파일 (File)
    * 컴퓨터에서 정보를 저장하는 논리적인 단위
    * 파일은 파일명과 확장자로 식별됨 (*ex. hello.py*)
    * 실행, 쓰기, 읽기 등을 할 수 있음

  ##### 절대 경로와 상대 경로

  * 경로

    * 컴퓨터 파일의 고유한 위치

  * 절대 경로

    * 루트 Directory부터 파일위치까지의 경로
      *ex. C:\user\docs\somefile.txt*

  * 상대 경로

    * 현재 있는 Directory로부터 타깃 파일까지의 경로

      *ex. ../../somefile.txt*

#### 터미널

- 마우스가 아닌, 키보드로 명령을 하여 입력 프로그램을 실행할 수 있음

![스크린샷 2021-01-24 오후 11.57.24](/Users/jisukim/Desktop/스크린샷 2021-01-24 오후 11.57.24.png)

* 터미널 또한 CLI 환경이라고 볼 수 있음

  ##### CLI (Command Line Interface)

  * GUI (Graphical User Interface) 와 달리, Text를 사용하여 컴퓨터에 명령을 입력하는 인터페이스 체계
  * Windows - CMD window, Windows Terminal, cmder
    Mac, Linux - Terminal
  * Console = Terminal = CMD창

* (Mac 기준) Spotlight (command + spacebar)에서 Terminal을 검색하여 사용하고 있음

  ##### 기본 명령어

  * 각 터미널에서는 프로그램을 작동하는 shell이 존재함

  ![스크린샷 2021-01-25 오전 12.02.47](/Users/jisukim/Desktop/스크린샷 2021-01-25 오전 12.02.47.png)

  * 이 외에도 mkdir (= make directory, 폴더 생성) 등의 명령어가 있음	 
  * [쉘 명렁어 관련 블로그](https://velog.io/@devmin/%EB%A6%AC%EB%88%85%EC%8A%A4-%EC%89%98-%EA%B8%B0%EB%B3%B8-%EB%AA%85%EB%A0%B9%EC%96%B4Basic-Shell-Commands) 참고

### 2. 파이썬 개요

#### About python

* 1991년 Gudio Van Rossum이 발표. (*cf. 1989년 크리스마스에 할 일이 없어서 개발했다고 함*)

* **플랫폼 독립적이며, Interpreter 언어**

  * 플랫폼(OS) 독립적 : OS에 상관없이 한 번 프로그램을 작성하면 사용 가능

  * Interpreter = 통역기를 사용하는 언어 : 소스코드를 바로 실행할 수 있게 지원

  * [참고] 컴파일러 vs 인터프리터

    <img src="/Users/jisukim/Desktop/스크린샷 2021-01-25 오전 12.15.46.png" alt="스크린샷 2021-01-25 오전 12.15.46" style="zoom:33%;" />

    * 예전에는 컴퓨터 속도 한계로 인터프리터 언어를 사용하지 못하였으나, 이제는 인터프리터 언어까지 편하게 사용할 수 있음

  * [참고] 프로그램의 동작 과정

    <img src="/Users/jisukim/Desktop/스크린샷 2021-01-25 오전 12.19.40.png" alt="스크린샷 2021-01-25 오전 12.19.40" style="zoom:33%;" />

    

* 객체 지향 / 동적 타이핑 언어

  * 객체 지향 언어 : 실행 순서가 아닌, 단위 모듈 (객체) 중심으로 프로그램을 작성
    하나의 객체는 어떤 목적을 달성하기 위한 method와 attribute를 가지고 있음
  * 동적 타이핑 언어 : 프로그램이 실행되는 시점에 프로그램이 사용해야 할 데이터 타입을 결정함

#### Why Python

* 쉽고 간단하며 다양함
* 이해하기 쉬운 문법 ("**사람의 시간이 기계의 시간보다 중요해짐**")
* 다양한 Library (특히 통계, 데이터 분석 관련 Library 발달)
* 널리 쓰이고 있고, 어디에든 쓸 수 있는 언어



### 3. 파이썬 코딩환경

#### 개발 환경

* 1) 운영 체제 (OS)
  * Windows, Linux, Mac
* 2) Python Interpreter
  * [Python](https://www.python.org/ )
  * [Anaconda](https://www.anaconda.com/ )
* 3) 코드 편집기 (Editor)
  * [VI editor](http://www.vim.org/ )
  * [Sublime Text](http://www.sublimetext.com/ )
  * [VS Code](https://code.visualstudio.com/ )
  * [PyCharm](https://www.jetbrains.com/pycharm/ )

