from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from serve import app


@pytest.fixture
def test_client():
    return TestClient(app)

def test_get_project_members_endpoint(test_client):
    """프로젝트 멤버 조회 엔드포인트 테스트"""
    with patch('serve.get_project_members', new_callable=AsyncMock) as mock_get_members:
        mock_get_members.return_value = [
            ("홍길동", "BE"),
            ("김철수", "FE")
        ]
        
        response = test_client.get("/api/projects/test-project/members")
        assert response.status_code == 200
        assert response.json() == [
            {"name": "홍길동", "positions": "BE"},
            {"name": "김철수", "positions": "FE"}
        ]

def test_get_project_members_not_found(test_client):
    """존재하지 않는 프로젝트 멤버 조회 테스트"""
    with patch('serve.get_project_members', new_callable=AsyncMock) as mock_get_members:
        mock_get_members.side_effect = Exception("프로젝트를 찾을 수 없습니다")
        
        response = test_client.get("/api/projects/non-existent/members")
        assert response.status_code == 404
        assert response.json() == {"detail": "프로젝트를 찾을 수 없습니다"}

def test_analyze_meeting_endpoint(test_client):
    """회의 분석 엔드포인트 테스트"""
    with patch('serve.analyze_meeting', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = {
            "summary": "회의 요약",
            "action_items": ["액션 아이템 1", "액션 아이템 2"],
            "participants": ["홍길동", "김철수"]
        }
        
        response = test_client.get("/api/meetings/test-meeting/analyze")
        assert response.status_code == 200
        assert response.json() == {
            "summary": "회의 요약",
            "action_items": ["액션 아이템 1", "액션 아이템 2"],
            "participants": ["홍길동", "김철수"]
        }

def test_analyze_meeting_not_found(test_client):
    """존재하지 않는 회의 분석 테스트"""
    with patch('serve.analyze_meeting', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.side_effect = Exception("회의를 찾을 수 없습니다")
        
        response = test_client.get("/api/meetings/non-existent/analyze")
        assert response.status_code == 404
        assert response.json() == {"detail": "회의를 찾을 수 없습니다"} 