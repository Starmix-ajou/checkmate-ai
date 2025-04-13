import re

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# BGE embedding 모델 로드
model = SentenceTransformer('BAAI/bge-m3')

def get_embedding(text: str):
    # 프롬프트 텍스트를 임베딩 벡터로 변환
    return model.encode(f"query: {text}")

# 검색 함수
def search(query: str, top_k: int = 3):
    query_vec = np.array([get_embedding(query)])
    D, I = index.search(query_vec, k=top_k)
    
    # (index, distance) 튜플 리스트 반환
    return [(docs[i], float(D[0][j])) for j, i in enumerate(I[0])]

# FAISS 벡터 인덱스 생성 및 검색
embedding_dim = 1024    # bge-m3 임베딩 차원

# 유클리드 거리 기반의 검색 (빠르다)
index = faiss.IndexFlatL2(embedding_dim)

# 전처리된 문장들을 docs 리스트로 로드
docs = []
with open('preprocessed_sentences.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    current_sentence = ""
    
    for line in lines:
        line = line.strip()
        if not line:  # 빈 줄 건너뛰기
            continue
            
        # 숫자로 시작하는 줄 처리 (예: "1. 텍스트")
        if re.match(r'^\d+\.', line):
            # 이전 문장이 있으면 저장
            if current_sentence:
                docs.append(current_sentence)
            # 번호를 제외한 실제 문장 부분만 저장
            current_sentence = re.sub(r'^\d+\.\s*', '', line)
        else:
            # 번호가 없는 일반 텍스트 줄
            if current_sentence:
                current_sentence += " " + line
            else:
                current_sentence = line
    
    # 마지막 문장 처리
    if current_sentence:
        docs.append(current_sentence)

print(f"총 {len(docs)}개의 문장이 로드되었습니다.")
print("\n처음 5개 문장 예시:")
for i, doc in enumerate(docs[:5], 1):
    print(f"{i}. {doc}")

# 문서 임베딩 벡터화해서 인덱스로 추가하기
doc_embeddings = np.array([get_embedding(doc) for doc in docs])
index.add(doc_embeddings)

app = FastAPI()

# 요청 데이터 모델
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    
@app.post("/rag/search")
def rag_search(request: QueryRequest):
    # 검색 함수 호출 (top_k 파라미터 전달)
    results = search(request.query, request.top_k)
    
    # 결과 반환 형식 설정하기
    return {
        "query": request.query,
        "results": [
            {"doc": doc, "score": score} for doc, score in results
        ]
    } 