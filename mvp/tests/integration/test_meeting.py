import pytest
from fastapi.testclient import TestClient
from meeting_analysis import analyze_meeting
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
    db.meetings.delete_many({})
    db.projects.delete_many({})
    db.users.delete_many({})

@pytest.mark.asyncio
async def test_meeting_creation_and_retrieval(test_client, test_db):
    """회의 생성 및 조회 통합 테스트"""
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
    
    # 테스트 사용자 생성
    user1_data = {
        "_id": "user1",
        "name": "홍길동",
        "profiles": [
            {
                "projectId": "test-project",
                "positions": ["BE"]
            }
        ]
    }
    user2_data = {
        "_id": "user2",
        "name": "김철수",
        "profiles": [
            {
                "projectId": "test-project",
                "positions": ["FE"]
            }
        ]
    }
    test_db.users.insert_many([user1_data, user2_data])
    
    # 회의 생성
    meeting_data = {
        "projectId": "test-project",
        "title": "테스트 회의",
        "content": "회의 내용 테스트",
        "date": "2024-03-20",
        "participants": ["user1", "user2"]
    }
    
    response = test_client.post("/api/meetings", json=meeting_data)
    assert response.status_code == 200
    meeting_id = response.json()["id"]
    
    # 생성된 회의 조회
    response = test_client.get(f"/api/meetings/{meeting_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "테스트 회의"
    assert response.json()["content"] == "회의 내용 테스트"
    assert len(response.json()["participants"]) == 2

@pytest.mark.asyncio
async def test_meeting_analysis(test_client, test_db):
    """회의 분석 통합 테스트"""
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
    
    # 테스트 사용자 생성
    user1_data = {
        "_id": "user1",
        "name": "홍길동",
        "profiles": [
            {
                "projectId": "test-project",
                "positions": ["BE"]
            }
        ]
    }
    user2_data = {
        "_id": "user2",
        "name": "김철수",
        "profiles": [
            {
                "projectId": "test-project",
                "positions": ["FE"]
            }
        ]
    }
    test_db.users.insert_many([user1_data, user2_data])
    
    # 회의 생성
    meeting_data = {
        "projectId": "test-project",
        "title": "테스트 회의",
        "content": "회의 내용 테스트",
        "date": "2024-03-20",
        "participants": ["user1", "user2"]
    }
    
    response = test_client.post("/api/meetings", json=meeting_data)
    meeting_id = response.json()["id"]
    
    # 회의 분석 요청
    response = test_client.get(f"/api/meetings/{meeting_id}/analyze")
    assert response.status_code == 200
    analysis_result = response.json()
    assert "summary" in analysis_result
    assert "action_items" in analysis_result
    assert "participants" in analysis_result

@pytest.mark.asyncio
async def test_meeting_update(test_client, test_db):
    """회의 업데이트 통합 테스트"""
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
    
    # 테스트 사용자 생성
    user1_data = {
        "_id": "user1",
        "name": "홍길동",
        "profiles": [
            {
                "projectId": "test-project",
                "positions": ["BE"]
            }
        ]
    }
    user2_data = {
        "_id": "user2",
        "name": "김철수",
        "profiles": [
            {
                "projectId": "test-project",
                "positions": ["FE"]
            }
        ]
    }
    test_db.users.insert_many([user1_data, user2_data])
    
    # 초기 회의 생성
    meeting_data = {
        "projectId": "test-project",
        "title": "테스트 회의",
        "content": "회의 내용 테스트",
        "date": "2024-03-20",
        "participants": ["user1"]
    }
    
    response = test_client.post("/api/meetings", json=meeting_data)
    meeting_id = response.json()["id"]
    
    # 회의 업데이트
    update_data = {
        "title": "수정된 회의",
        "content": "수정된 회의 내용",
        "participants": ["user1", "user2"]
    }
    
    response = test_client.put(f"/api/meetings/{meeting_id}", json=update_data)
    assert response.status_code == 200
    
    # 업데이트된 회의 확인
    response = test_client.get(f"/api/meetings/{meeting_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "수정된 회의"
    assert response.json()["content"] == "수정된 회의 내용"
    assert len(response.json()["participants"]) == 2

@pytest.mark.asyncio
async def test_meeting_deletion(test_client, test_db):
    """회의 삭제 통합 테스트"""
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
    
    # 테스트 사용자 생성
    user1_data = {
        "_id": "user1",
        "name": "홍길동",
        "profiles": [
            {
                "projectId": "test-project",
                "positions": ["BE"]
            }
        ]
    }
    test_db.users.insert_one(user1_data)
    
    # 회의 생성
    meeting_data = {
        "projectId": "test-project",
        "title": "테스트 회의",
        "content": "회의 내용 테스트",
        "date": "2024-03-20",
        "participants": ["user1"]
    }
    
    response = test_client.post("/api/meetings", json=meeting_data)
    meeting_id = response.json()["id"]
    
    # 회의 삭제
    response = test_client.delete(f"/api/meetings/{meeting_id}")
    assert response.status_code == 200
    
    # 삭제된 회의 확인
    response = test_client.get(f"/api/meetings/{meeting_id}")
    assert response.status_code == 404 