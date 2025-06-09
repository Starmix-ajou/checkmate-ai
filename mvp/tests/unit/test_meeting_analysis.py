from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from meeting_analysis import (analyze_meeting, convert_action_items_to_tasks,
                              create_action_items_gpt, create_summary)


@pytest.mark.asyncio
async def test_analyze_meeting_success():
    """회의 분석 성공 테스트"""
    mock_meeting_data = {
        "_id": "test-meeting",
        "projectId": "test-project",
        "content": "회의 내용 테스트",
        "participants": ["user1", "user2"],
        "date": "2024-03-20"
    }
    
    mock_project_data = {
        "_id": "test-project",
        "name": "테스트 프로젝트",
        "members": [
            {"id": "user1"},
            {"id": "user2"}
        ]
    }
    
    mock_user_data = [
        {
            "_id": "user1",
            "name": "홍길동",
            "profiles": [
                {
                    "projectId": "test-project",
                    "positions": ["BE"]
                }
            ]
        },
        {
            "_id": "user2",
            "name": "김철수",
            "profiles": [
                {
                    "projectId": "test-project",
                    "positions": ["FE"]
                }
            ]
        }
    ]
    
    with patch('meeting_analysis.get_meeting_collection', new_callable=AsyncMock) as mock_meeting_collection, \
         patch('meeting_analysis.get_project_collection', new_callable=AsyncMock) as mock_project_collection, \
         patch('meeting_analysis.get_user_collection', new_callable=AsyncMock) as mock_user_collection:
        
        mock_meeting_collection.return_value.find_one = AsyncMock(return_value=mock_meeting_data)
        mock_project_collection.return_value.find_one = AsyncMock(return_value=mock_project_data)
        mock_user_collection.return_value.find_one = AsyncMock(side_effect=mock_user_data)
        
        result = await analyze_meeting("test-meeting")
        
        assert result is not None
        assert "summary" in result
        assert "action_items" in result
        assert "participants" in result

@pytest.mark.asyncio
async def test_analyze_meeting_not_found():
    """회의를 찾을 수 없는 경우 테스트"""
    with patch('meeting_analysis.get_meeting_collection', new_callable=AsyncMock) as mock_meeting_collection:
        mock_meeting_collection.return_value.find_one = AsyncMock(return_value=None)
        
        with pytest.raises(Exception, match="회의를 찾을 수 없습니다"):
            await analyze_meeting("non-existent-meeting")

@pytest.mark.asyncio
async def test_analyze_meeting_no_project():
    """프로젝트를 찾을 수 없는 경우 테스트"""
    mock_meeting_data = {
        "_id": "test-meeting",
        "projectId": "non-existent-project",
        "content": "회의 내용 테스트",
        "participants": ["user1"],
        "date": "2024-03-20"
    }
    
    with patch('meeting_analysis.get_meeting_collection', new_callable=AsyncMock) as mock_meeting_collection, \
         patch('meeting_analysis.get_project_collection', new_callable=AsyncMock) as mock_project_collection:
        
        mock_meeting_collection.return_value.find_one = AsyncMock(return_value=mock_meeting_data)
        mock_project_collection.return_value.find_one = AsyncMock(return_value=None)
        
        with pytest.raises(Exception, match="프로젝트를 찾을 수 없습니다"):
            await analyze_meeting("test-meeting") 