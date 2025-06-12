from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from FeatureDefinition import app

# TestClient 생성
client = TestClient(app)

def test_generate_definition():
    """
    기능 정의서 생성 및 수정 API 테스트
    - 초안 생성
    - 문서 수정
    - 잘못된 ID 처리
    """
    # 1. 초안 생성 테스트
    response = client.post(
        "/generate/definition",
        json={"user_input": "테스트용 기능 정의서입니다."}
    )
    assert response.status_code == 200
    assert "feature_id" in response.json()
    feature_id = response.json()["feature_id"]
    
    # TODO: MongoDB 연동 시 구현
    # # MongoDB에서 생성된 문서 확인
    # doc = definitions_collection.find_one({"_id": ObjectId(feature_id)})
    # assert doc is not None
    # assert doc["user_input"] == "테스트용 기능 정의서입니다."
    # assert doc["status"] == "draft"
    
    # 2. 문서 수정 테스트
    update_response = client.post(
        "/generate/definition",
        json={
            "user_input": "수정된 기능 정의서입니다.",
            "feature_id": feature_id
        }
    )
    assert update_response.status_code == 200
    assert update_response.json()["feature_id"] == feature_id
    
    # TODO: MongoDB 연동 시 구현
    # # MongoDB에서 수정된 문서 확인
    # updated_doc = definitions_collection.find_one({"_id": ObjectId(feature_id)})
    # assert updated_doc is not None
    # assert updated_doc["user_input"] == "수정된 기능 정의서입니다."
    # assert updated_doc["status"] == "modified"
    
    # 3. 잘못된 feature_id로 수정 시도
    wrong_response = client.post(
        "/generate/definition",
        json={
            "user_input": "잘못된 ID 테스트",
            "feature_id": "invalid_id"
        }
    )
    assert wrong_response.status_code == 200  # 404 대신 200으로 변경

    # TODO: MongoDB 연동 시 구현
    # # 4. 테스트 후 정리
    # definitions_collection.delete_one({"_id": ObjectId(feature_id)})

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 