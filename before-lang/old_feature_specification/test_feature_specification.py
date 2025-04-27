from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from FeatureSpecification import app

# TestClient 생성
client = TestClient(app)

class MockOpenAIResponse:
    """OpenAI API 응답을 모방하는 mock 클래스"""
    def __init__(self):
        self.choices = [
            type('Choice', (), {
                'message': type('Message', (), {
                    'content': "테스트용 명세서 내용"
                })()
            })()
        ]

@patch('openai.chat.completions.create')
def test_generate_specification(mock_openai):
    """
    기능 명세서 생성 API 테스트
    - OpenAI API 모의 처리
    - 명세서 생성
    - 잘못된 ID 처리
    """
    # OpenAI API 응답 모의 설정
    mock_response = type('Response', (), {
        'choices': [
            type('Choice', (), {
                'message': type('Message', (), {
                    'content': "테스트용 명세서 내용"
                })()
            })()
        ]
    })
    mock_openai.return_value = mock_response
    
    # 1. 명세서 생성 테스트
    spec_response = client.post(
        "/generate/specification",
        json={"feature_id": "valid_feature_id"}
    )
    print(f"Response: {spec_response.json()}")  # 디버깅을 위한 출력
    assert spec_response.status_code == 200
    assert "data" in spec_response.json()
    
    # 2. 잘못된 feature_id로 명세서 생성 시도
    wrong_response = client.post(
        "/generate/specification",
        json={"feature_id": "invalid_id"}
    )
    assert wrong_response.status_code == 404

    # TODO: MongoDB 연동 시 구현
    # # MongoDB에서 생성된 명세서 확인
    # spec_doc = specifications_collection.find_one({"feature_id": feature_id})
    # assert spec_doc is not None
    # assert "generated_text" in spec_doc
    # # 테스트 후 정리
    # specifications_collection.delete_one({"feature_id": feature_id})

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 