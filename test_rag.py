import pytest
from fastapi.testclient import TestClient

from rag_query import app, docs, search

# TestClient 생성
client = TestClient(app)

def test_rag_search():
    """
    RAG 검색 API 테스트
    - 기본 검색 기능
    - top_k 파라미터 테스트
    - 빈 쿼리 처리
    """
    # 1. 기본 검색 테스트
    response = client.post(
        "/rag/search",
        json={"query": "파이썬", "top_k": 3}
    )
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) == 3
    
    # 검색 결과의 형식 확인
    for result in response.json()["results"]:
        assert "doc" in result
        assert "score" in result
        assert isinstance(result["score"], float)
        assert result["doc"] in docs  # 결과가 실제 문서 목록에 있는지 확인
    
    # 2. top_k 파라미터 테스트
    response = client.post(
        "/rag/search",
        json={"query": "파이썬", "top_k": 1}
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
    
    # 3. 빈 쿼리 테스트
    response = client.post(
        "/rag/search",
        json={"query": "", "top_k": 3}
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3

def test_search_function():
    """
    search 함수 직접 테스트
    - 검색 결과 형식 확인
    - top_k 파라미터 동작 확인
    """
    # 1. 기본 검색 테스트
    results = search("파이썬", top_k=3)
    assert len(results) == 3
    assert all(isinstance(result, tuple) for result in results)
    assert all(len(result) == 2 for result in results)  # (doc, score) 튜플 확인
    
    # 2. top_k=1 테스트
    results = search("파이썬", top_k=1)
    assert len(results) == 1
    
    # 3. 결과가 실제 문서 목록에 있는지 확인
    for doc, score in results:
        assert doc in docs
        assert isinstance(score, float)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 