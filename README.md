# checkmate-ai
![체크메이트썸네일](https://i.imgur.com/rpwI0gG.png)

- 🌟 Starmix Organization GitHub : [https://github.com/Starmix-ajou](https://github.com/Starmix-ajou)
- ♟ checkmate URL : [https://checkmate.it.kr](https://checkmate.it.kr)
- 🖥 Project Manager 전용 뷰 : [https://manager.checkmate.it.kr](https://manager.checkmate.it.kr)

## 프로젝트 소개

**checkmate는 소규모 주니어 개발팀을 위한 프로젝트 관리 및 협업툴입니다.**

AI를 활용한 프로젝트 생성과 Sprint 구성, 회의록 자동 요약 기능을 제공하여 팀의 초기 기획부터 실행까지의 과정을 효율적으로 지원합니다. 
회의 내용을 실시간으로 정리할 수 있는 공동 편집 기능을 통해 주요 논의 사항을 요약하고, 이를 실행 가능한 액션 아이템(Task)으로 전환할 수 있습니다. 
Task는 Epic 단위로 구조화할 수 있으며, Gantt Chart, Kanban Board, Calendar를 통해 관리할 수 있습니다. 
또한 상세 Task 페이지의 댓글 기능을 통해 팀 내부 이해관계자 간의 원활한 소통이 가능하도록 하여, 개발 과정 전반에서 협업의 생산성을 높입니다.

## 팀원 구성

<div align="center">

| 박승연 |
| --- |
| <img src="https://github.com/user-attachments/assets/6161e664-bf97-452f-b6b0-6b384a643b7c" width="200" height="200"/> |

</div>
<br>

## 1. 개발 환경

### AI
![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![OpenAI API](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=openai&logoColor=white)

### 협업 툴
![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)
![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)

### 이슈 및 버전 관리
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)
![Jira](https://img.shields.io/badge/jira-%230A0FFF.svg?style=for-the-badge&logo=jira&logoColor=white)

### 테스트
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC.svg?style=for-the-badge&logo=pytest&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-158CBA.svg?style=for-the-badge&logo=pydantic&logoColor=white)

## 2. 채택한 개발 기술과 브랜치 전략

### Python

### FastAPI

### HuggingFace


### 브랜치 전략

- 브랜치 명: **Jira 태스크 ID 기반**
  - ex) `CM-123`
- **Rebase Merge** 방식으로 main 브랜치에 병합
- **최소 1명 이상의 approve** 필요

## 3. Challenge


## 4. 프로젝트 구조
``` bash
.env
.gitignore
├── .github               # GitHub 설정 및 워크플로우
├── .pytest_cache        
├── AI_hub/summary        # AI hub로부터 다운로드된 요약 학습 데이터
├── Korpora               # Korpora로부터 다운로드된 NER 학습 데이터
├── before-lang           # LangChain, LangGraph 적용 버전
├── mvp                   # 배포 프로젝트
├── tests                 # 테스트 코드
├── venv                  # 가상머신 관련 파일
├── docker-compose.yml    # 도커 DB 이미지 생성 파일 (Docker 빌드 이미지 실행 테스트용 at local) 
├── Dockerfile            # 도커 이미지 생성 파일
├── model_training.py     # 모델 학습 코드

```

## Fine-tuning
### 사용한 데이터셋
1. [Korpora: 네이버 x 창원대 NER 데이터](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/naver_changwon_ner.html)
    - Korpora에 업로드된 NER 데이터로, 총 90000개의 샘플이 존재함. 해당 데이터와 동일한 개체명 분류 기준을 채택하여 2번의 AI Hub 데이터에 대해 개체명을 부여함.
2. [AI Hub: 요약문 및 레포트 생성 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=582)
    - 8종류의 서로 다른 자연어 문서에 대한 데이터. 이 중에서 4번 paper(정책 토론회), 5번 minute(공공기관 부서 및 부처 회의록) 데이터를 사용함. Korpora와 유사하게 개체명을 gpt를 사용하여 부여하고, 5번 minute 데이터를 파인튜닝 용으로 사용함.


## 5. 구조적 장점

## 6. 테스트

### Pytest를 사용한 테스트 코드 작성


## 7. 개발 기간 및 작업 관리

### 개발 기간

- **2025-03-06 ~ (진행 중)**

### 작업 관리

- **협업 툴**: GitHub + Slack + Jira
- **회의**: 주 2회 팀 전체 회의 진행 + Google Docs로 회의록 공유
- **요청/QA 문서화**: Notion을 통해 요청 사항 정리 및 QA 문서로 재활용


## 작성된 AI Modules
1. feature_definition/ (기능 정의서 생성):
    - 사용자 입력 기능에 대한 feature_definition 생성
    - 사용자 수정 요청에 따른 feature definition 재생성
    - approved 시 벡터 임베딩 후 db에 저장

2. feature_specification/ (기능 명세서 생성):
    - feature_definition에 대한 feature_specification 생성
        - feature을 task, epic 단위로 구분
        - epic 별 task 분류
        - 개발 기간, 개발 난이도에 따라 task 별 개발 예상 시간 책정
        - task 별 담당자 배정
    - 사용자 수정 요청에 따른 feature_specification 재생성
    - approved 시 벡터 임베딩 후 db에 저장

3. meeting_summary/ (회의록 요약):
    - 회의록 텍스트의 내용을 단순 요약
    - 벡터 임베딩하여 db에 저장

4. meeting_action_item/ (회의록 액션 아이템 추출):
    - 회의록 텍스트에 대해 NER 분류 작업 수행
        - 작업 내용, 작업 담당자, 작업 마감일 추출
    - 사용자 요청에 따른 NER 객체 추가, 수정 및 삭제
    - 벡터 임베딩하여 db에 저장

5. search_rag/ (내용 검색):
    - 사용자가 찾고자 하는 내용과 유사한 내용을 db로부터 검색하여 반환
    - 2가지 종류의 검색 지원
        - 1) 벡터 값 간의 차이를 계산하여 유사도를 평가하는 단순 검색
        - 2) RAG와 같이 LLM이 개입하여 deep search와 내용 정리를 지원하는 강화 검색

6. fine-tuning_test_data, train_data/ (파인 튜닝):
    - gpt-3.5-turbo-16k-1160과 BAAI/bge-m3 모델에 대해 파인튜닝을 진행
    - train_data: 재학습에 사용한 공개/합성 데이터 저장
    - test_data: 파인튜닝 모델 성능 테스트로 사용한 공개/합성 데이터 & 테스트 결과 저장


## Used AI Models and Metrics by Modules 1-5
#### Input Embedding
- 1. feature_definition: x
- 2. feature_specification: x
- 3. meeting_summary: x
- 4. meeting_action_item: x
- 5. search_rag: BAAI/bge-m3
- on plan to use text_preprocessor.py when a .pdf file is given as input

### Output Creation
- 1. feature_definition: gpt-3.5-turbo-16k-1160 (fine-tuned)
    - used original model also for the tests
- 2. feature_specification: gpt-3.5-turbo-16k-1160 (fine-tuned)
    - used original model also for the tests
- 3. meeting_summary: gpt-3.5-turbo-16k-1160 (original)
- 4. meeting_action_item: gpt-4o {on test now}
- 5. search_rag: (metric) cosine similarity + (function) Atlas Vector Search

### Output Embedding
- use BAAI/bge-m3 in common


