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

# 예시: 문서 임베딩 벡터화해서 인덱스로 추가하기
docs = [
    "한글 문서 예시입니다.",
    "파이썬과 머신러닝에 관한 문서입니다.",
    "이것은 GPT와 RAG 시스템 설명입니다."
]
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