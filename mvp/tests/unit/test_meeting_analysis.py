from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from meeting_analysis import (analyze_meeting_document,
                              convert_action_items_to_tasks,
                              create_action_items_gpt, create_summary)


class FakeAsyncCursor:
    def __init__(self, data):
        self._data = data

    async def to_list(self, length=None):
        return self._data

@pytest.mark.asyncio
async def test_create_summary_success():
    """회의 요약 생성 성공 테스트"""
    title = "테스트 회의"
    content = """
    # 회의 안건
    1. 프로젝트 진행 상황
    2. 다음 단계 계획
    
    ## 프로젝트 진행 상황
    - 현재 80% 완료
    - 남은 작업: UI 개선
    
    ## 다음 단계 계획
    - 다음 주까지 UI 개선 완료
    - 테스트 진행
    """
    project_id = "test-project"
    
    mock_response = AsyncMock()
    mock_response.content = '{"summary": "# 테스트 회의\\n\\n## 프로젝트 진행 상황\\n- 현재 80% 완료\\n- 남은 작업: UI 개선\\n\\n## 다음 단계 계획\\n- 다음 주까지 UI 개선 완료\\n- 테스트 진행"}'
    
    with patch('meeting_analysis.get_project_members', new_callable=AsyncMock) as mock_get_members, \
         patch('meeting_analysis.ChatOpenAI') as mock_chat:
        
        mock_get_members.return_value = [
            {"id": "user1", "name": "홍길동"},
            {"id": "user2", "name": "김철수"}
        ]
        mock_chat.return_value.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await create_summary(title, content, project_id)
        assert isinstance(result, str)
        assert title in result
        assert "프로젝트 진행 상황" in result
        assert "다음 단계 계획" in result

@pytest.mark.asyncio
async def test_create_summary_empty_content():
    """빈 내용으로 회의 요약 생성 테스트"""
    with patch('meeting_analysis.get_project_members', new_callable=AsyncMock) as mock_get_members, \
         patch('meeting_analysis.ChatOpenAI') as mock_chat:
        
        mock_get_members.return_value = []
        mock_chat.return_value.ainvoke = AsyncMock(side_effect=Exception("GPT API 처리 중 오류 발생"))
        
        with pytest.raises(Exception):
            await create_summary("테스트 회의", "", "test-project")


@pytest.mark.asyncio
async def test_create_action_items_gpt_success():
    """GPT를 사용한 액션 아이템 생성 성공 테스트"""
    content = """
    회의 내용:
    - 김승연이 10월 1일까지 보고서를 제출하기로 함
    - 자료 정리는 다음 주까지 완료하기로 함
    """
    
    result = await create_action_items_gpt(content)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, dict) for item in result)
    assert all("description" in item for item in result)
    assert all("assignee" in item for item in result)
    assert all("endDate" in item for item in result)

@pytest.mark.asyncio
async def test_create_action_items_gpt_empty_content():
    """빈 내용으로 GPT 액션 아이템 생성 테스트"""
    with pytest.raises(Exception):
        await create_action_items_gpt("")

@pytest.mark.asyncio
async def test_convert_action_items_to_tasks_success():
    """액션 아이템을 태스크로 변환 성공 테스트"""
    action_items = [
        {
            "description": "보고서 제출하기",
            "assignee": "홍길동",
            "endDate": "2024-10-01"
        }
    ]
    project_id = "test-project"
    
    mock_epics = [
        {
            "_id": "epic1",
            "title": "문서 작업",
            "description": "보고서 작성 및 제출"
        }
    ]
    
    mock_users = [
        {
            "_id": "user1",
            "name": "홍길동",
            "profiles": [{"projectId": "test-project", "positions": ["BE"]}]
        }
    ]
    
    mock_response = AsyncMock()
    mock_response.content = '{"actionItems": [{"title": "보고서 제출", "description": "보고서 제출하기", "assigneeId": "user1", "endDate": "2024-10-01", "epicId": "epic1"}]}'
    
    with patch('meeting_analysis.get_epic_collection', new_callable=AsyncMock) as mock_epic_collection, \
         patch('meeting_analysis.get_project_members', new_callable=AsyncMock) as mock_get_members, \
         patch('meeting_analysis.get_user_collection', new_callable=AsyncMock) as mock_user_collection, \
         patch('meeting_analysis.get_project_collection', new_callable=AsyncMock) as mock_project_collection, \
         patch('meeting_analysis.ChatOpenAI') as mock_chat:
        
        # MongoDB 컬렉션 모의 설정
        mock_get_members.return_value = mock_users
        mock_user_collection.return_value.find_one = AsyncMock(return_value=mock_users[0])
        mock_project_collection.return_value.find_one = AsyncMock(return_value={"members": [{"id": "user1"}]})
        mock_chat.return_value.ainvoke = AsyncMock(return_value=mock_response)
        
        # MongoDB 컬렉션 모의 설정
        mock_epic_collection_instance = MagicMock()
        mock_epic_collection_instance.find.return_value = FakeAsyncCursor(mock_epics)
        mock_epic_collection.return_value = mock_epic_collection_instance
        
        result = await convert_action_items_to_tasks(action_items, project_id)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, dict) for item in result)
        assert all("title" in item for item in result)
        assert all("description" in item for item in result)
        assert all("assigneeId" in item for item in result)
        assert all("endDate" in item for item in result)
        assert all("epicId" in item for item in result)

@pytest.mark.asyncio
async def test_convert_action_items_to_tasks_empty_input():
    """빈 액션 아이템으로 태스크 변환 테스트"""
    with pytest.raises(AssertionError):
        await convert_action_items_to_tasks(None, "test-project")

@pytest.mark.asyncio
async def test_analyze_meeting_document_success():
    """회의 문서 분석 성공 테스트"""
    title = "테스트 회의"
    content = """
    # 회의 안건
    1. 프로젝트 진행 상황
    2. 다음 단계 계획
    """
    project_id = "test-project"
    
    with patch('meeting_analysis.create_summary', new_callable=AsyncMock) as mock_create_summary, \
         patch('meeting_analysis.create_action_items_gpt', new_callable=AsyncMock) as mock_create_action_items, \
         patch('meeting_analysis.convert_action_items_to_tasks', new_callable=AsyncMock) as mock_convert_tasks:
        
        mock_create_summary.return_value = "요약 내용"
        mock_create_action_items.return_value = [{"description": "테스트", "assignee": "홍길동", "endDate": "2024-10-01"}]
        mock_convert_tasks.return_value = [{"title": "테스트", "description": "테스트", "assigneeId": "user1", "endDate": "2024-10-01", "epicId": "epic1"}]
        
        result = await analyze_meeting_document(title, content, project_id)
        assert isinstance(result, dict)
        assert "summary" in result
        assert "actionItems" in result
        assert isinstance(result["actionItems"], list)

@pytest.mark.asyncio
async def test_analyze_meeting_document_empty_input():
    """빈 입력으로 회의 문서 분석 테스트"""
    with pytest.raises(Exception):
        await analyze_meeting_document("", "", "test-project")