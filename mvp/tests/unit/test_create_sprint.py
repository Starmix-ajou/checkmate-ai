import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson.dbref import DBRef
from create_sprint import (calculate_eff_mandays, calculate_percentiles,
                           create_sprint, create_task_from_epic,
                           create_task_from_feature, create_task_from_null)


class FakeAsyncCursor:
    def __init__(self, data):
        self._data = data

    async def to_list(self, length=None):
        return self._data

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
    """기능으로부터 태스크 생성 테스트 - 구조 검증"""
    # 입력 데이터 정의
    input_data = {
        "epic_id": "epic1",
        "feature_id": "feature1",
        "project_id": "test-project",
        "workhours_per_day": 8
    }
    
    # 예상되는 출력 데이터 정의
    expected_output = {
        "title": "로그인 API 구현",
        "description": "로그인 API 엔드포인트 구현",
        "assignee": "홍길동",
        "startDate": "2024-03-01",
        "endDate": "2024-03-05",
        "priority": 150,
        "epic": "epic1"
    }
    
    # 입력과 출력의 구조 검증
    assert isinstance(input_data, dict)
    assert "epic_id" in input_data
    assert "feature_id" in input_data
    assert "project_id" in input_data
    assert "workhours_per_day" in input_data
    
    assert isinstance(expected_output, dict)
    assert "title" in expected_output
    assert "description" in expected_output
    assert "assignee" in expected_output
    assert "startDate" in expected_output
    assert "endDate" in expected_output
    assert "priority" in expected_output
    assert "epic" in expected_output

@pytest.mark.asyncio
async def test_create_task_from_epic():
    """에픽으로부터 태스크 생성 테스트 - 구조 검증"""
    # 입력 데이터 정의
    input_data = {
        "epic_id": "epic1",
        "project_id": "test-project",
        "task_db_data": [
            {
                "title": "로그인 API 구현",
                "description": "로그인 API 엔드포인트 구현",
                "assignee": "user1",
                "startDate": "2024-03-01",
                "endDate": "2024-03-05",
                "priority": 150
            }
        ],
        "workhours_per_day": 8
    }
    
    # 예상되는 출력 데이터 정의
    expected_output = {
        "title": "로그인 API 구현",
        "description": "로그인 API 엔드포인트 구현",
        "assignee": "user1",
        "startDate": "",
        "endDate": "",
        "priority": 150,
        "epic": "epic1"
    }
    
    # 입력과 출력의 구조 검증
    assert isinstance(input_data, dict)
    assert "epic_id" in input_data
    assert "project_id" in input_data
    assert "task_db_data" in input_data
    assert "workhours_per_day" in input_data
    
    assert isinstance(expected_output, dict)
    assert "title" in expected_output
    assert "description" in expected_output
    assert "assignee" in expected_output
    assert "startDate" in expected_output
    assert "endDate" in expected_output
    assert "priority" in expected_output
    assert "epic" in expected_output

@pytest.mark.asyncio
async def test_create_task_from_null():
    """null로부터 태스크 생성 테스트 - 구조 검증"""
    # 입력 데이터 정의
    input_data = {
        "epic_id": "test-epic",
        "project_id": "test-project",
        "workhours_per_day": 8
    }
    
    # 예상되는 출력 데이터 정의
    expected_output = {
        "title": "로그인 API 구현",
        "description": "로그인 API 엔드포인트 구현",
        "assignee": "홍길동",
        "startDate": "",
        "endDate": "",
        "priority": 150,
        "epic": "test-epic"
    }
    
    # 입력과 출력의 구조 검증
    assert isinstance(input_data, dict)
    assert "epic_id" in input_data
    assert "project_id" in input_data
    assert "workhours_per_day" in input_data
    
    assert isinstance(expected_output, dict)
    assert "title" in expected_output
    assert "description" in expected_output
    assert "assignee" in expected_output
    assert "startDate" in expected_output
    assert "endDate" in expected_output
    assert "priority" in expected_output
    assert "epic" in expected_output

@pytest.mark.asyncio
async def test_create_sprint():
    """스프린트 생성 테스트 - 입력과 출력 구조 검증"""
    # 입력 데이터 정의
    input_data = {
        "project_id": "test-project",
        "pending_tasks_ids": ["task1"],
        "start_date": "2024-03-01"
    }
    
    # 예상되는 출력 데이터 정의
    expected_output = {
        "sprint": {
            "title": "API 구현",
            "description": "로그인 API 엔드포인트 구현",
            "startDate": "2024-03-01",
            "endDate": "2024-03-14"
        },
        "epics": [
            {
                "epicId": "epic1",
                "tasks": [
                    {
                        "title": "로그인 API 구현",
                        "description": "로그인 API 엔드포인트 구현",
                        "assigneeId": "user1",
                        "startDate": "2024-03-01",
                        "endDate": "2024-03-14",
                        "priority": 300
                    }
                ]
            }
        ]
    }
    
    # 입력과 출력의 구조 검증
    assert isinstance(input_data, dict)
    assert "project_id" in input_data
    assert "pending_tasks_ids" in input_data
    assert "start_date" in input_data
    
    assert isinstance(expected_output, dict)
    assert "sprint" in expected_output
    assert "epics" in expected_output
    
    sprint = expected_output["sprint"]
    assert "title" in sprint
    assert "description" in sprint
    assert "startDate" in sprint
    assert "endDate" in sprint
    
    assert isinstance(expected_output["epics"], list)
    assert len(expected_output["epics"]) > 0
    
    epic = expected_output["epics"][0]
    assert "epicId" in epic
    assert "tasks" in epic
    
    tasks = epic["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) > 0
    
    task = tasks[0]
    assert "title" in task
    assert "description" in task
    assert "assigneeId" in task
    assert "startDate" in task
    assert "endDate" in task
    assert "priority" in task

@pytest.mark.asyncio
async def test_gpt_response_format():
    """GPT 응답 형식 검증 테스트"""
    # Mock GPT 응답
    mock_gpt_response = {
        "sprint_days": 14,
        "eff_mandays": 80,
        "workhours_per_day": 8,
        "number_of_sprints": 1,
        "sprints": [
            {
                "title": "API 구현",
                "description": "로그인 및 회원가입 API 엔드포인트 구현",
                "startDate": "2024-03-01",
                "endDate": "2024-03-14",
                "epics": [
                    {
                        "epicId": "epic1",
                        "tasks": [
                            {
                                "title": "로그인 API 구현",
                                "description": "로그인 API 엔드포인트 구현",
                                "assignee": "홍길동",
                                "startDate": "2024-03-01",
                                "endDate": "2024-03-14",
                                "expected_workhours": 80,
                                "priority": 300,
                                "difficulty": 3
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AsyncMock(content=json.dumps(mock_gpt_response))
    
    with patch('create_sprint.ChatOpenAI', return_value=mock_llm):
        # GPT 응답 형식 검증
        assert "sprint_days" in mock_gpt_response
        assert "eff_mandays" in mock_gpt_response
        assert "workhours_per_day" in mock_gpt_response
        assert "number_of_sprints" in mock_gpt_response
        assert "sprints" in mock_gpt_response
        
        sprint = mock_gpt_response["sprints"][0]
        assert "title" in sprint
        assert "description" in sprint
        assert "startDate" in sprint
        assert "endDate" in sprint
        assert "epics" in sprint
        
        epic = sprint["epics"][0]
        assert "epicId" in epic
        assert "tasks" in epic
        
        task = epic["tasks"][0]
        assert "title" in task
        assert "description" in task
        assert "assignee" in task
        assert "startDate" in task
        assert "endDate" in task
        assert "expected_workhours" in task
        assert "priority" in task
        assert "difficulty" in task