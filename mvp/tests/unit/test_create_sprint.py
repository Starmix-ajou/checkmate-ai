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

def test_calculate_percentiles():
    """우선순위 분위수 계산 테스트"""
    tasks = [
        {"priority": 100},
        {"priority": 200},
        {"priority": 300},
        {"priority": 400},
        {"priority": 500}
    ]
    
    result = calculate_percentiles(tasks)
    
    # 우선순위가 재조정되었는지 확인
    priorities = [task["priority"] for task in result]
    assert all(priority in [50, 150, 250] for priority in priorities)
    
    # 분위수에 따른 재조정이 올바르게 되었는지 확인
    assert result[0]["priority"] == 50  # Low
    assert result[2]["priority"] == 150  # Medium
    assert result[4]["priority"] == 250  # High

@pytest.mark.asyncio
async def test_create_task_from_feature():
    """기능으로부터 태스크 생성 테스트"""
    mock_feature = {
        "name": "로그인 기능",
        "useCase": "사용자 로그인",
        "input": "이메일, 비밀번호",
        "output": "로그인 성공/실패",
        "startDate": "2024-03-01",
        "endDate": "2024-03-15",
        "expectedDays": 10
    }
    
    mock_response = {
        "content": """
        {
            "tasks": [
                {
                    "title": "로그인 API 구현",
                    "description": "로그인 API 엔드포인트 구현",
                    "assignee": "홍길동",
                    "startDate": "2024-03-01",
                    "endDate": "2024-03-05",
                    "difficulty": 3,
                    "expected_workhours": 40
                }
            ]
        }
        """
    }
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    with patch('create_sprint.ChatOpenAI', return_value=mock_llm), \
         patch('create_sprint.feature_collection.find_one', new_callable=AsyncMock) as mock_find_one, \
         patch('create_sprint.get_project_members', new_callable=AsyncMock) as mock_get_members:
        
        mock_find_one.return_value = mock_feature
        mock_get_members.return_value = ["홍길동 (BE)", "김철수 (FE)"]
        
        result = await create_task_from_feature(
            epic_id="test-epic",
            feature_id="test-feature",
            project_id="test-project",
            workhours_per_day=8
        )
        
        assert len(result) > 0
        assert "title" in result[0]
        assert "description" in result[0]
        assert "assignee" in result[0]
        assert "startDate" in result[0]
        assert "endDate" in result[0]
        assert "priority" in result[0]

@pytest.mark.asyncio
async def test_create_task_from_epic():
    """에픽으로부터 태스크 생성 테스트"""
    mock_epic = {
        "_id": "test-epic",
        "title": "로그인 기능 개발",
        "description": "사용자 로그인 기능 구현"
    }
    
    mock_task_data = [{
        "title": "로그인 API 구현",
        "description": None,
        "assignee": None,
        "startDate": None,
        "endDate": None,
        "priority": None
    }]
    
    mock_response = {
        "content": """
        {
            "tasks": [
                {
                    "title": "로그인 API 구현",
                    "description": "로그인 API 엔드포인트 구현",
                    "assignee": "홍길동",
                    "startDate": "2024-03-01",
                    "endDate": "2024-03-05",
                    "difficulty": 3,
                    "expected_workhours": 40
                }
            ]
        }
        """
    }
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    with patch('create_sprint.ChatOpenAI', return_value=mock_llm), \
         patch('create_sprint.epic_collection.find_one', new_callable=AsyncMock) as mock_find_one, \
         patch('create_sprint.get_project_members', new_callable=AsyncMock) as mock_get_members:
        
        mock_find_one.return_value = mock_epic
        mock_get_members.return_value = ["홍길동 (BE)", "김철수 (FE)"]
        
        result = await create_task_from_epic(
            epic_id="test-epic",
            project_id="test-project",
            task_db_data=mock_task_data,
            workhours_per_day=8
        )
        
        assert len(result) > 0
        assert all(field in result[0] for field in ["title", "description", "assignee", "startDate", "endDate", "priority"])

@pytest.mark.asyncio
async def test_create_sprint():
    """스프린트 생성 테스트"""
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
    
    with patch('create_sprint.task_collection.find', new_callable=AsyncMock) as mock_find, \
         patch('create_sprint.task_collection.update_many', new_callable=AsyncMock) as mock_update:
        
        mock_find.return_value.to_list = AsyncMock(return_value=mock_tasks)
        
        result = await create_sprint(
            project_id="test-project",
            pending_tasks_ids=["task1"],
            start_date="2024-03-01"
        )
        
        assert "sprint_id" in result
        assert "tasks" in result
        assert len(result["tasks"]) > 0
        mock_update.assert_called_once() 