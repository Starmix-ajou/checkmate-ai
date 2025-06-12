import pytest
from create_sprint import create_sprint
from fastapi.testclient import TestClient
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
    db.sprints.delete_many({})
    db.projects.delete_many({})
    db.features.delete_many({})

@pytest.mark.asyncio
async def test_sprint_creation_and_retrieval(test_client, test_db):
    """스프린트 생성 및 조회 통합 테스트"""
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
    
    # 테스트 기능 생성
    feature_data = {
        "_id": "feature1",
        "projectId": "test-project",
        "name": "테스트 기능 1",
        "description": "테스트 기능 1 설명",
        "priority": "HIGH",
        "status": "TODO",
        "assignee": "user1"
    }
    test_db.features.insert_one(feature_data)
    
    # 스프린트 생성
    sprint_data = {
        "projectId": "test-project",
        "name": "테스트 스프린트",
        "startDate": "2024-03-20",
        "endDate": "2024-04-02",
        "features": ["feature1"]
    }
    
    response = test_client.post("/api/sprints", json=sprint_data)
    assert response.status_code == 200
    sprint_id = response.json()["id"]
    
    # 생성된 스프린트 조회
    response = test_client.get(f"/api/sprints/{sprint_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "테스트 스프린트"
    assert response.json()["features"] == ["feature1"]

@pytest.mark.asyncio
async def test_sprint_update(test_client, test_db):
    """스프린트 업데이트 통합 테스트"""
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
    
    # 테스트 기능들 생성
    feature1_data = {
        "_id": "feature1",
        "projectId": "test-project",
        "name": "테스트 기능 1",
        "description": "테스트 기능 1 설명",
        "priority": "HIGH",
        "status": "TODO",
        "assignee": "user1"
    }
    feature2_data = {
        "_id": "feature2",
        "projectId": "test-project",
        "name": "테스트 기능 2",
        "description": "테스트 기능 2 설명",
        "priority": "MEDIUM",
        "status": "TODO",
        "assignee": "user2"
    }
    test_db.features.insert_many([feature1_data, feature2_data])
    
    # 초기 스프린트 생성
    sprint_data = {
        "projectId": "test-project",
        "name": "테스트 스프린트",
        "startDate": "2024-03-20",
        "endDate": "2024-04-02",
        "features": ["feature1"]
    }
    
    response = test_client.post("/api/sprints", json=sprint_data)
    sprint_id = response.json()["id"]
    
    # 스프린트 업데이트
    update_data = {
        "name": "수정된 스프린트",
        "features": ["feature1", "feature2"]
    }
    
    response = test_client.put(f"/api/sprints/{sprint_id}", json=update_data)
    assert response.status_code == 200
    
    # 업데이트된 스프린트 확인
    response = test_client.get(f"/api/sprints/{sprint_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "수정된 스프린트"
    assert len(response.json()["features"]) == 2
    assert "feature1" in response.json()["features"]
    assert "feature2" in response.json()["features"]

@pytest.mark.asyncio
async def test_sprint_deletion(test_client, test_db):
    """스프린트 삭제 통합 테스트"""
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
    
    # 테스트 기능 생성
    feature_data = {
        "_id": "feature1",
        "projectId": "test-project",
        "name": "테스트 기능 1",
        "description": "테스트 기능 1 설명",
        "priority": "HIGH",
        "status": "TODO",
        "assignee": "user1"
    }
    test_db.features.insert_one(feature_data)
    
    # 스프린트 생성
    sprint_data = {
        "projectId": "test-project",
        "name": "테스트 스프린트",
        "startDate": "2024-03-20",
        "endDate": "2024-04-02",
        "features": ["feature1"]
    }
    
    response = test_client.post("/api/sprints", json=sprint_data)
    sprint_id = response.json()["id"]
    
    # 스프린트 삭제
    response = test_client.delete(f"/api/sprints/{sprint_id}")
    assert response.status_code == 200
    
    # 삭제된 스프린트 확인
    response = test_client.get(f"/api/sprints/{sprint_id}")
    assert response.status_code == 404 