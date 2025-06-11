from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from serve import app


@pytest.fixture
def test_client():
    return TestClient(app)

# Feature Definition 테스트
def test_post_feature_definition(test_client):
    """기능 정의서 생성 엔드포인트 테스트"""
    with patch('serve.create_feature_definition', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = {
            "suggestion": {
                "features": ["기능1", "기능2"],
                "suggestions": [{"text": "제안1"}, {"text": "제안2"}]
            }
        }
        
        response = test_client.post(
            "/project/definition",
            json={
                "email": "test@example.com",
                "description": "테스트 설명",
                "definitionUrl": "http://example.com"
            }
        )
        assert response.status_code == 200
        assert response.json() == mock_create.return_value

def test_put_feature_definition(test_client):
    """기능 정의서 피드백 엔드포인트 테스트"""
    with patch('serve.update_feature_definition', new_callable=AsyncMock) as mock_update:
        mock_update.return_value = {
            "features": ["기능1", "기능2"],
            "isNextStep": True
        }
        
        response = test_client.put(
            "/project/definition",
            json={
                "email": "test@example.com",
                "feedback": "피드백 내용"
            }
        )
        assert response.status_code == 200
        assert response.json() == mock_update.return_value

# Feature Specification 테스트
def test_post_feature_specification(test_client):
    """기능 명세서 생성 엔드포인트 테스트"""
    with patch('serve.create_feature_specification', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = {
            "features": [
                {"id": "1", "name": "기능1"},
                {"id": "2", "name": "기능2"}
            ]
        }
        
        response = test_client.post(
            "/project/specification",
            json={"email": "test@example.com"}
        )
        assert response.status_code == 200
        assert response.json() == mock_create.return_value

def test_put_feature_specification(test_client):
    """기능 명세서 피드백 엔드포인트 테스트"""
    with patch('serve.update_feature_specification', new_callable=AsyncMock) as mock_update:
        mock_update.return_value = {
            "features": [
                {"id": "1", "name": "기능1"},
                {"id": "2", "name": "기능2"}
            ],
            "isNextStep": True
        }
        
        response = test_client.put(
            "/project/specification",
            json={
                "email": "test@example.com",
                "feedback": "피드백 내용",
                "createdFeatures": [{"name": "새기능"}],
                "modifiedFeatures": [{"id": "1", "name": "수정기능"}],
                "deletedFeatures": ["2"]
            }
        )
        assert response.status_code == 200
        assert response.json() == mock_update.return_value

# Sprint 테스트
def test_post_sprint(test_client):
    """스프린트 생성 엔드포인트 테스트"""
    with patch('serve.create_sprint', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = {
            "sprint": {"id": "1", "name": "스프린트1"},
            "epics": [{"id": "1", "name": "에픽1"}]
        }
        
        response = test_client.post(
            "/sprint",
            json={
                "projectId": "project1",
                "pendingTasksIds": ["task1", "task2"],
                "startDate": datetime.now().isoformat()
            }
        )
        assert response.status_code == 200
        assert response.json() == mock_create.return_value

# Meeting 테스트
def test_post_meeting(test_client):
    """회의록 분석 엔드포인트 테스트"""
    with patch('serve.analyze_meeting_document', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = {
            "summary": "회의 요약",
            "actionItems": [
                {"task": "작업1", "assignee": "홍길동"},
                {"task": "작업2", "assignee": "김철수"}
            ]
        }
        
        response = test_client.post(
            "/meeting",
            json={
                "title": "테스트 회의",
                "content": "회의 내용",
                "projectId": "project1"
            }
        )
        assert response.status_code == 200
        assert response.json() == mock_analyze.return_value

# 에러 처리 테스트
def test_error_handling(test_client):
    """에러 처리 테스트"""
    with patch('serve.create_feature_definition', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = Exception("테스트 에러")
        
        response = test_client.post(
            "/project/definition",
            json={
                "email": "test@example.com",
                "description": "테스트 설명"
            }
        )
        assert response.status_code == 500
        assert "테스트 에러" in response.json()["detail"]

# 서버 시작 이벤트 테스트
@pytest.mark.asyncio
async def test_lifespan():
    """서버 시작 이벤트 테스트"""
    from serve import lifespan
    
    with patch('serve.test_redis_connection', new_callable=AsyncMock) as mock_redis, \
         patch('serve.test_mongodb_connection', new_callable=AsyncMock) as mock_mongo:
        mock_redis.return_value = True
        mock_mongo.return_value = True
        
        # lifespan 컨텍스트 매니저 실행
        async with lifespan(app):
            mock_redis.assert_called_once()
            mock_mongo.assert_called_once()