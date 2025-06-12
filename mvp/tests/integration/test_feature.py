import pytest
from fastapi.testclient import TestClient
from feature_specification import create_feature_specification
from mongodb_setting import get_database
from serve import app


@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def test_db():
    db = get_database()
    yield db
    # 테스트 후 데이터 정리
    db.features.delete_many({})
    db.projects.delete_many({})

@pytest.mark.asyncio
async def test_feature_creation_and_retrieval(test_client, test_db):
    """기능 명세 생성 및 조회 통합 테스트"""
    # 테스트 프로젝트 생성
    project_data = {
        "_id": "test-project",
        "name": "테스트 프로젝트",
        "members": [
            {"id": "user1"},
            {"id": "user2"}
        ]
    }
    test_db.projects.insert_one(project_data)
    
    # 기능 명세 생성
    feature_data = {
        "projectId": "test-project",
        "name": "테스트 기능",
        "description": "테스트 기능 설명",
        "priority": "HIGH",
        "status": "TODO",
        "assignee": "user1"
    }
    
    response = test_client.post("/api/features", json=feature_data)
    assert response.status_code == 200
    feature_id = response.json()["id"]
    
    # 생성된 기능 명세 조회
    response = test_client.get(f"/api/features/{feature_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "테스트 기능"
    assert response.json()["description"] == "테스트 기능 설명"

@pytest.mark.asyncio
async def test_feature_update(test_client, test_db):
    """기능 명세 업데이트 통합 테스트"""
    # 테스트 프로젝트 생성
    project_data = {
        "_id": "test-project",
        "name": "테스트 프로젝트",
        "members": [
            {"id": "user1"},
            {"id": "user2"}
        ]
    }
    test_db.projects.insert_one(project_data)
    
    # 초기 기능 명세 생성
    feature_data = {
        "projectId": "test-project",
        "name": "테스트 기능",
        "description": "테스트 기능 설명",
        "priority": "HIGH",
        "status": "TODO",
        "assignee": "user1"
    }
    
    response = test_client.post("/api/features", json=feature_data)
    feature_id = response.json()["id"]
    
    # 기능 명세 업데이트
    update_data = {
        "name": "수정된 기능",
        "description": "수정된 설명",
        "priority": "MEDIUM",
        "status": "IN_PROGRESS"
    }
    
    response = test_client.put(f"/api/features/{feature_id}", json=update_data)
    assert response.status_code == 200
    
    # 업데이트된 기능 명세 확인
    response = test_client.get(f"/api/features/{feature_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "수정된 기능"
    assert response.json()["description"] == "수정된 설명"
    assert response.json()["priority"] == "MEDIUM"
    assert response.json()["status"] == "IN_PROGRESS"

@pytest.mark.asyncio
async def test_feature_deletion(test_client, test_db):
    """기능 명세 삭제 통합 테스트"""
    # 테스트 프로젝트 생성
    project_data = {
        "_id": "test-project",
        "name": "테스트 프로젝트",
        "members": [
            {"id": "user1"},
            {"id": "user2"}
        ]
    }
    test_db.projects.insert_one(project_data)
    
    # 기능 명세 생성
    feature_data = {
        "projectId": "test-project",
        "name": "테스트 기능",
        "description": "테스트 기능 설명",
        "priority": "HIGH",
        "status": "TODO",
        "assignee": "user1"
    }
    
    response = test_client.post("/api/features", json=feature_data)
    feature_id = response.json()["id"]
    
    # 기능 명세 삭제
    response = test_client.delete(f"/api/features/{feature_id}")
    assert response.status_code == 200
    
    # 삭제된 기능 명세 확인
    response = test_client.get(f"/api/features/{feature_id}")
    assert response.status_code == 404 