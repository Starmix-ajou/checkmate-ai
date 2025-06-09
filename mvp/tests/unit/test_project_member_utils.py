from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from project_member_utils import (get_project_members,
                                  map_memberName_to_memberId)


@pytest.mark.asyncio
async def test_get_project_members_success():
    """프로젝트 멤버 조회 성공 테스트"""
    mock_project_data = {
        "_id": "test-project",
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
                    "positions": ["BE", "FE"]
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
    
    with patch('project_member_utils.get_project_collection', new_callable=AsyncMock) as mock_project_collection, \
         patch('project_member_utils.get_user_collection', new_callable=AsyncMock) as mock_user_collection:
        
        mock_project_collection.return_value.find_one = AsyncMock(return_value=mock_project_data)
        mock_user_collection.return_value.find_one = AsyncMock(side_effect=mock_user_data)
        
        result = await get_project_members("test-project")
        
        assert len(result) == 2
        assert ("홍길동", "BE, FE") in result
        assert ("김철수", "FE") in result

@pytest.mark.asyncio
async def test_get_project_members_no_project():
    """프로젝트가 없는 경우 테스트"""
    with patch('project_member_utils.get_project_collection', new_callable=AsyncMock) as mock_project_collection:
        mock_project_collection.return_value.find_one = AsyncMock(return_value=None)
        
        with pytest.raises(Exception, match="프로젝트를 찾을 수 없습니다"):
            await get_project_members("non-existent-project")

@pytest.mark.asyncio
async def test_get_project_members_no_members():
    """멤버가 없는 경우 테스트"""
    mock_project_data = {
        "_id": "test-project",
        "members": []
    }
    
    with patch('project_member_utils.get_project_collection', new_callable=AsyncMock) as mock_project_collection:
        mock_project_collection.return_value.find_one = AsyncMock(return_value=mock_project_data)
        
        with pytest.raises(AssertionError, match="members가 없습니다"):
            await get_project_members("test-project")

@pytest.mark.asyncio
async def test_map_memberName_to_memberId_success():
    """멤버 이름으로 ID 매핑 성공 테스트"""
    mock_user_data = {
        "_id": "user1",
        "name": "홍길동"
    }
    
    with patch('project_member_utils.get_user_collection', new_callable=AsyncMock) as mock_user_collection:
        mock_user_collection.return_value.find_one = AsyncMock(return_value=mock_user_data)
        
        result = await map_memberName_to_memberId("홍길동", mock_user_collection.return_value)
        assert result == "user1"

@pytest.mark.asyncio
async def test_map_memberName_to_memberId_not_found():
    """멤버를 찾을 수 없는 경우 테스트"""
    with patch('project_member_utils.get_user_collection', new_callable=AsyncMock) as mock_user_collection:
        mock_user_collection.return_value.find_one = AsyncMock(return_value=None)
        
        with pytest.raises(Exception, match="사용자 정보를 찾을 수 없습니다"):
            await map_memberName_to_memberId("존재하지 않는 멤버", mock_user_collection.return_value) 