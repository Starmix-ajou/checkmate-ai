import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# BGE embedding 모델 로드
model = SentenceTransformer('BAAI/bge-m3')

def get_embedding(text: str, is_query: bool = True):
    """
    텍스트를 임베딩 벡터로 변환합니다.
    
    Args:
        text (str): 변환할 텍스트
        is_query (bool): 쿼리 텍스트인지 여부
        
    Returns:
        numpy.ndarray: 임베딩 벡터
    """
    if is_query:
        text = f"query: {text}"
    return model.encode(text)

def semantic_search(query: str, top_k: int = 3):
    """
    의미적 유사도 기반 검색을 수행합니다.
    
    Args:
        query (str): 검색 쿼리
        top_k (int): 반환할 결과 개수
        
    Returns:
        list: (문서, 유사도 점수) 튜플의 리스트
    """
    query_vec = np.array([get_embedding(query)])
    D, I = index.search(query_vec, k=top_k)
    
    return [(docs[i], float(D[0][j])) for j, i in enumerate(I[0])]

def exact_match_search(query: str, docs: list, top_k: int = 3):
    """
    정확한 문장 매칭 검색을 수행합니다.
    
    Args:
        query (str): 검색할 문장
        docs (list): 검색 대상 문서 리스트
        top_k (int): 반환할 결과 개수
        
    Returns:
        list: (문서, 유사도 점수) 튜플의 리스트
    """
    # 쿼리와 각 문서의 유사도 계산
    query_vec = get_embedding(query)
    doc_scores = []
    
    for doc in docs:
        doc_vec = get_embedding(doc, is_query=False)
        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        doc_scores.append((doc, float(similarity)))
    
    # 유사도 기준 상위 k개 결과 반환
    return sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_k]

def print_search_results(query: str, results: list, search_type: str):
    """
    검색 결과를 보기 좋게 출력합니다.
    """
    print(f"\n=== {search_type} 검색 결과 ===")
    print(f"쿼리: {query}")
    print("\n상위 결과:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. 유사도 점수: {score:.4f}")
        print(f"문서: {doc}")

# FAISS 벡터 인덱스 생성 및 검색
embedding_dim = 1024    # bge-m3 임베딩 차원
index = faiss.IndexFlatL2(embedding_dim)

# 전처리된 단락들을 docs 리스트로 로드
with open('preprocessed_paragraphs.txt', 'r', encoding='utf-8') as file:
    docs = [line.split('. ', 1)[1].strip() for line in file if line.strip()]

# 문서 임베딩 벡터화해서 인덱스로 추가하기
doc_embeddings = np.array([get_embedding(doc, is_query=False) for doc in docs])
index.add(doc_embeddings)

if __name__ == "__main__":
    # 테스트 쿼리 예시
    test_queries = [
        # 정확한 문장 매칭 테스트
        "AI 시스템의 윤리적 사용이 중요한 이슈가 되었습니다",
        # 의미적 검색 테스트
        "인공지능 윤리",
        # 일반적인 주제 검색 테스트
        "클라우드 컴퓨팅 기술",
        # 복합 주제 검색 테스트
        "AI와 보안"
    ]
    
    print("\n=== RAG 시스템 테스트 ===")
    
    for query in test_queries:
        # 의미적 검색 테스트
        semantic_results = semantic_search(query, top_k=2)
        print_search_results(query, semantic_results, "의미적")
        
        # 정확한 매칭 검색 테스트
        exact_results = exact_match_search(query, docs, top_k=2)
        print_search_results(query, exact_results, "정확한 매칭")
        
        print("\n" + "="*50)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    search_type: str = "semantic"  # "semantic" 또는 "exact"
    top_k: int = 3

@app.post("/rag/search")
def rag_search(request: QueryRequest):
    """
    RAG 검색을 수행합니다.
    
    Args:
        request (QueryRequest): 검색 요청 (쿼리, 검색 타입, top_k)
        
    Returns:
        dict: 검색 결과
    """
    if request.search_type == "semantic":
        results = semantic_search(request.query, request.top_k)
    else:  # exact match
        results = exact_match_search(request.query, docs, request.top_k)
    
    return {
        "query": request.query,
        "search_type": request.search_type,
        "results": [
            {"doc": doc, "score": score} for doc, score in results
        ]
    } 