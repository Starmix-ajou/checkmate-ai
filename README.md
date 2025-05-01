# checkmate-AI
- 2025-1학기 캡스톤 디자인 과목에서 개발하는 CheckMate 서비스가 지원하는 AI 기능들과 관련된 데이터, 코드를 정리함.

## AI architecture
- ver 1.0
![AI Architecture Design](./figure/AI%20아키텍쳐.png)

## AI Modules
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



## Fine-tuning
### 사용한 데이터셋
1. [Korpora: 네이버 x 창원대 NER 데이터](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/naver_changwon_ner.html)
    - Korpora에 업로드된 NER 데이터로, 총 90000개의 샘플이 존재함. 이 중 
2. [AI Hub: 요약문 및 레포트 생성 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=582)
    - 