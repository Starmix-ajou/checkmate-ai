from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from create_sprint import (calculate_eff_mandays, calculate_percentiles,
                           create_sprint, create_task_from_epic,
                           create_task_from_feature, create_task_from_null)


@pytest.mark.asyncio
async def test_calculate_eff_mandays():
    """효율적인 작업일수 계산 테스트"""
    # 기본 케이스
    result = await calculate_eff_mandays(0.8, 2, 5, 8)
    assert result == 64  # 2 * 5 * 8 * 0.8 = 64
    
    # 효율성 100% 케이스
    result = await calculate_eff_mandays(1.0, 2, 5, 8)
    assert result == 80  # 2 * 5 * 8 * 1.0 = 80
    
    # 효율성 50% 케이스
    result = await calculate_eff_mandays(0.5, 2, 5, 8)
    assert result == 40  # 2 * 5 * 8 * 0.5 = 40
    
    # 실패 케이스: 잘못된 입력값
    with pytest.raises(Exception):
        await calculate_eff_mandays(-1, 2, 5, 8)  # 음수 효율성
    
    with pytest.raises(Exception):
        await calculate_eff_mandays(0.8, 0, 5, 8)  # 0명의 개발자

@pytest.mark.asyncio
async def test_calculate_percentiles():
    """우선순위 분위수 계산 테스트"""
    # 기본 케이스
    tasks = [
        {"priority": 100},
        {"priority": 200},
        {"priority": 300},
        {"priority": 400},
        {"priority": 500}
    ]
    
    result = await calculate_percentiles(tasks)
    
    # 각 task의 priority 값을 개별적으로 검사
    assert result[0]["priority"] == 50  # Low
    assert result[1]["priority"] == 50  # Low
    assert result[2]["priority"] == 150  # Medium
    assert result[3]["priority"] == 250  # High
    assert result[4]["priority"] == 250  # High
    
    # 실패 케이스: 빈 리스트
    with pytest.raises(Exception):
        await calculate_percentiles([])
    
    # 실패 케이스: priority가 없는 task
    with pytest.raises(Exception):
        await calculate_percentiles([{"title": "Task 1"}])

@pytest.mark.asyncio
async def test_create_task_from_feature():
    """기능으로부터 태스크 생성 테스트"""
    # 성공 케이스
    mock_feature = {
        "name": "로그인 기능",
        "useCase": "사용자 로그인",
        "input": "이메일, 비밀번호",
        "output": "로그인 성공/실패",
        "startDate": None,
        "endDate": None,
        "difficulty": 3,
        "expected_workhours": 10
    }
    
    mock_response = AsyncMock()
    mock_response.content = """
    {
        "tasks": [
            {
                "title": "로그인 API 구현",
                "description": "로그인 API 엔드포인트 구현",
                "assignee": "홍길동",
                "difficulty": 3,
                "startDate": "2024-03-01",
                "endDate": "2024-03-05",
                "expected_workhours": 10
            }
        ]
    }
    """
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    mock_collections = (
        MagicMock(),  # feature_collection
        MagicMock(),  # project_collection
        MagicMock(),  # epic_collection
        MagicMock(),  # task_collection
        MagicMock()   # user_collection
    )
    
    with patch('create_sprint.ChatOpenAI', return_value=mock_llm), \
         patch('create_sprint.init_collections', new_callable=AsyncMock) as mock_init_collections, \
         patch('create_sprint.get_project_members', new_callable=AsyncMock) as mock_get_members:
        
        mock_init_collections.return_value = mock_collections
        mock_collections[0].find_one = AsyncMock(return_value=mock_feature)
        mock_get_members.return_value = ["홍길동 (BE)", "김철수 (FE)"]
        
        result = await create_task_from_feature(
            epic_id="test-epic",
            feature_id="test-feature",
            project_id="test-project",
            workhours_per_day=8
        )
        assert isinstance(result, list)
    
    # 실패 케이스: feature를 찾을 수 없음
    pytest.raises(Exception)

@pytest.mark.asyncio
async def test_create_task_from_epic():
    """에픽으로부터 태스크 생성 테스트"""
    # 성공 케이스
    mock_epic = {
        "_id": "test-epic",
        "title": "로그인 기능 개발",
        "description": "사용자 로그인 기능 구현"
    }
    
    mock_task_data = [{
        "title": "로그인 API 구현",
        "description": None,
        "assignee": None,
        "difficulty": 3,
        "expected_workhours": 10
    }]
    
    mock_response = AsyncMock()
    mock_response.content = """
    {
        "epic_description": "사용자 로그인 기능 구현",
        "tasks": [
            {
                "title": "로그인 API 구현",
                "description": "로그인 API 엔드포인트 구현",
                "assignee": "홍길동",
                "difficulty": 3,
                "expected_workhours": 10
            }
        ]
    }
    """
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    mock_collections = (
        MagicMock(),  # feature_collection
        MagicMock(),  # project_collection
        MagicMock(),  # epic_collection
        MagicMock(),  # task_collection
        MagicMock()   # user_collection
    )
    
    with patch('create_sprint.ChatOpenAI', return_value=mock_llm), \
         patch('create_sprint.init_collections', new_callable=AsyncMock) as mock_init_collections, \
         patch('create_sprint.get_project_members', new_callable=AsyncMock) as mock_get_members:
        
        mock_init_collections.return_value = mock_collections
        mock_collections[2].find_one = AsyncMock(return_value=mock_epic)
        mock_get_members.return_value = ["홍길동 (BE)", "김철수 (FE)"]
        
        result = await create_task_from_epic(
            epic_id="test-epic",
            project_id="test-project",
            task_db_data=mock_task_data,
            workhours_per_day=8
        )
        assert isinstance(result, list)
    
    # 실패 케이스: epic을 찾을 수 없음
    pytest.raises(Exception)

@pytest.mark.asyncio
async def test_create_task_from_null():
    """null로부터 태스크 생성 테스트"""
    # 성공 케이스
    mock_epic = {
        "_id": "test-epic",
        "title": "로그인 기능 개발",
        "description": None
    }
    
    mock_project = {
        "_id": "test-project",
        "description": "사용자 인증 시스템 개발"
    }
    
    mock_response = AsyncMock()
    mock_response.content = """
    {
        "epic_description": "사용자 인증 시스템 개발",
        "tasks": [
            {
                "title": "로그인 API 구현",
                "description": "로그인 API 엔드포인트 구현",
                "assignee": "홍길동",
                "difficulty": 3,
                "expected_workhours": 10
            }
        ]
    }
    """
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    mock_collections = (
        MagicMock(),  # feature_collection
        MagicMock(),  # project_collection
        MagicMock(),  # epic_collection
        MagicMock(),  # task_collection
        MagicMock()   # user_collection
    )
    
    with patch('create_sprint.ChatOpenAI', return_value=mock_llm), \
         patch('create_sprint.init_collections', new_callable=AsyncMock) as mock_init_collections, \
         patch('create_sprint.get_project_members', new_callable=AsyncMock) as mock_get_members:
        
        mock_init_collections.return_value = mock_collections
        mock_collections[2].find_one = AsyncMock(return_value=mock_epic)
        mock_collections[1].find_one = AsyncMock(return_value=mock_project)
        mock_get_members.return_value = ["홍길동 (BE)", "김철수 (FE)"]
        
        result = await create_task_from_null(
            epic_id="test-epic",
            project_id="test-project",
            workhours_per_day=8
        )
        assert isinstance(result, list)
    
    # 실패 케이스: epic을 찾을 수 없음
    pytest.raises(Exception)

class FakeAsyncCursor:
    def __init__(self, data):
        self._data = data

    async def to_list(self, length=None):
        return self._data

@pytest.mark.asyncio
async def test_create_sprint():
    """스프린트 생성 테스트"""
    # 성공 케이스
    mock_tasks = [
        {
            "_id": "task1",
            "title": "로그인 API 구현",
            "description": "로그인 API 엔드포인트 구현",
            "assignee": "홍길동",
            "startDate": "2024-03-01",
            "endDate": "2024-03-05",
            "priority": 150
        }
    ]
    
    mock_epics = [
        {
            "_id": "epic1",
            "title": "로그인 기능",
            "description": "사용자 로그인 기능 구현",
            "projectId": "test-project"
        }
    ]
    
    mock_project = {
        "_id": "test-project",
        "startDate": "2024-03-01",
        "endDate": "2024-03-31",
        "members": ["user1", "user2"]
    }
    
    mock_users = [
        {
            "_id": "user1",
            "name": "홍길동",
            "profiles": [{"projectId": "test-project", "positions": ["BE"]}]
        }
    ]
    
    with patch('create_sprint.init_collections', new_callable=AsyncMock) as mock_init_collections, \
        patch('create_sprint.get_project_members', new_callable=AsyncMock) as mock_get_members:

        mock_collections = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        mock_collections[1].find_one = AsyncMock(return_value=mock_project)
        mock_collections[2].find.return_value = FakeAsyncCursor(mock_epics)
        mock_collections[3].find.return_value = FakeAsyncCursor(mock_tasks)
        mock_collections[4].find_one.return_value = mock_users[0]

        mock_init_collections.return_value = mock_collections
        mock_get_members.return_value = mock_users
        
        result = await create_sprint(
            project_id="test-project",
            pending_tasks_ids=["task1"],
            start_date="2024-03-01"
        )
        assert isinstance(result, dict)
    
    # 실패 케이스: 프로젝트를 찾을 수 없음
    pytest.raises(Exception)