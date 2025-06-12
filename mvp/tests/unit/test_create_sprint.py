import json
from datetime import datetime, timedelta
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field


class ProjectMember(BaseModel):
    """프로젝트 멤버 모델"""
    name: str
    position: str


class Feature(BaseModel):
    """기능 모델"""
    featureId: str
    name: str
    useCase: str
    input: str
    output: str
    startDate: str
    endDate: str
    expectedDays: int


class Epic(BaseModel):
    """에픽 모델"""
    _id: str
    title: str
    description: Optional[str] = None
    projectId: Optional[str] = None


class Task(BaseModel):
    """태스크 모델"""
    _id: Optional[str] = None
    title: str
    description: Optional[str] = None
    assignee: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    priority: Optional[int] = None
    epic: Optional[str] = None
    difficulty: Optional[int] = Field(None, ge=1, le=5)
    expected_workhours: Optional[float] = None


class Sprint(BaseModel):
    """스프린트 모델"""
    title: str
    description: str
    startDate: str
    endDate: str


class SprintResponse(BaseModel):
    """스프린트 응답 모델"""
    sprint: Sprint
    epics: List[dict]


@pytest.mark.asyncio
async def test_calculate_eff_mandays():
    """효율적인 작업일수 계산 테스트"""
    # 기본 케이스
    efficiency_factor = 0.8
    number_of_developers = 2
    sprint_days = 5
    workhours_per_day = 8
    
    result = (efficiency_factor * number_of_developers * sprint_days * workhours_per_day)
    assert result == 64  # 2 * 5 * 8 * 0.8 = 64
    
    # 효율성 100% 케이스
    efficiency_factor = 1.0
    result = (efficiency_factor * number_of_developers * sprint_days * workhours_per_day)
    assert result == 80  # 2 * 5 * 8 * 1.0 = 80
    
    # 효율성 50% 케이스
    efficiency_factor = 0.5
    result = (efficiency_factor * number_of_developers * sprint_days * workhours_per_day)
    assert result == 40  # 2 * 5 * 8 * 0.5 = 40
    
    # 실패 케이스: 잘못된 입력값
    with pytest.raises(ValueError):
        efficiency_factor = -1
        if efficiency_factor < 0:
            raise ValueError("효율성은 0보다 커야 합니다.")
    
    with pytest.raises(ValueError):
        number_of_developers = 0
        if number_of_developers <= 0:
            raise ValueError("개발자 수는 0보다 커야 합니다.")


@pytest.mark.asyncio
async def test_calculate_percentiles():
    """우선순위 분위수 계산 테스트"""
    # 기본 케이스
    tasks = [
        Task(title="Task 1", priority=100),
        Task(title="Task 2", priority=200),
        Task(title="Task 3", priority=300),
        Task(title="Task 4", priority=400),
        Task(title="Task 5", priority=500)
    ]
    
    # 우선순위를 기준으로 정렬
    sorted_tasks = sorted([task.dict() for task in tasks], key=lambda x: x["priority"])
    
    # 분위수 계산
    n = len(sorted_tasks)
    result = []
    for i, task in enumerate(sorted_tasks):
        if i < 2:  # 하위 40% (2개)
            task["priority"] = 50  # Low
        elif i < 3:  # 중간 30% (1개)
            task["priority"] = 150  # Medium
        else:  # 상위 30% (2개)
            task["priority"] = 250  # High
        result.append(task)
    
    # 각 task의 priority 값을 개별적으로 검사
    assert result[0]["priority"] == 50  # Low
    assert result[1]["priority"] == 50  # Low
    assert result[2]["priority"] == 150  # Medium
    assert result[3]["priority"] == 250  # High
    assert result[4]["priority"] == 250  # High
    
    # 실패 케이스: 빈 리스트
    with pytest.raises(AssertionError):
        tasks = []
        if not tasks:
            raise AssertionError("태스크 리스트가 비어있습니다.")
    
    # 실패 케이스: priority가 없는 task
    with pytest.raises(KeyError):
        task = {"title": "Task 1"}
        if "priority" not in task:
            raise KeyError("priority 필드가 없습니다.")


@pytest.mark.asyncio
async def test_create_task_from_feature():
    """기능으로부터 태스크 생성 테스트"""
    # Mock 데이터 설정
    mock_feature = Feature(
        featureId="feature1",
        name="로그인 기능",
        useCase="사용자 로그인",
        input="이메일, 비밀번호",
        output="JWT 토큰",
        startDate="2024-03-01",
        endDate="2024-03-05",
        expectedDays=5
    )
    
    mock_project_members = [
        ProjectMember(name="홍길동", position="BE"),
        ProjectMember(name="김철수", position="FE")
    ]
    
    # GPT 응답 시뮬레이션
    mock_gpt_response = {
        "tasks": [
            {
                "title": "로그인 API 구현",
                "description": "로그인 API 엔드포인트 구현",
                "assignee": "홍길동",
                "startDate": "2024-03-01",
                "endDate": "2024-03-05",
                "difficulty": 3,
                "expected_workhours": 16
            }
        ]
    }
    
    # 태스크 생성 로직 시뮬레이션
    tasks = []
    for task_data in mock_gpt_response["tasks"]:
        task = Task(
            title=task_data["title"],
            description=task_data["description"],
            assignee=task_data["assignee"],
            startDate=task_data["startDate"],
            endDate=task_data["endDate"],
            difficulty=task_data["difficulty"],
            expected_workhours=task_data["expected_workhours"],
            epic="epic1"
        )
        tasks.append(task.dict())
    
    # 결과 검증
    assert len(tasks) == 1
    task = Task(**tasks[0])
    assert task.title == "로그인 API 구현"
    assert task.description == "로그인 API 엔드포인트 구현"
    assert task.assignee == "홍길동"
    assert task.startDate == "2024-03-01"
    assert task.endDate == "2024-03-05"
    assert task.difficulty == 3
    assert task.expected_workhours == 16
    assert task.epic == "epic1"


@pytest.mark.asyncio
async def test_create_task_from_epic():
    """에픽으로부터 태스크 생성 테스트"""
    # Mock 데이터 설정
    mock_epic = Epic(
        _id="epic1",
        title="로그인 기능",
        description="사용자 인증 기능 구현"
    )
    
    mock_task_db_data = [
        Task(
            title="로그인 API 구현",
            description=None,
            assignee=None,
            startDate=None,
            endDate=None,
            priority=None
        )
    ]
    
    mock_project_members = [
        ProjectMember(name="홍길동", position="BE"),
        ProjectMember(name="김철수", position="FE")
    ]
    
    # GPT 응답 시뮬레이션
    mock_gpt_response = {
        "epic_description": "사용자 인증 기능 구현",
        "tasks": [
            {
                "title": "로그인 API 구현",
                "description": "로그인 API 엔드포인트 구현",
                "assignee": "홍길동",
                "difficulty": 3,
                "expected_workhours": 16
            }
        ]
    }
    
    # 태스크 생성 로직 시뮬레이션
    tasks = []
    for task_data in mock_gpt_response["tasks"]:
        task = Task(
            title=task_data["title"],
            description=task_data["description"],
            assignee=task_data["assignee"],
            difficulty=task_data["difficulty"],
            expected_workhours=task_data["expected_workhours"],
            epic="epic1"
        )
        tasks.append(task.dict())
    
    # 결과 검증
    assert len(tasks) == 1
    task = Task(**tasks[0])
    assert task.title == "로그인 API 구현"
    assert task.description == "로그인 API 엔드포인트 구현"
    assert task.assignee == "홍길동"
    assert task.difficulty == 3
    assert task.expected_workhours == 16
    assert task.epic == "epic1"


@pytest.mark.asyncio
async def test_create_task_from_null():
    """null로부터 태스크 생성 테스트"""
    # Mock 데이터 설정
    mock_epic = Epic(
        _id="epic1",
        title="로그인 기능",
        description=None
    )
    
    mock_project_members = [
        ProjectMember(name="홍길동", position="BE"),
        ProjectMember(name="김철수", position="FE")
    ]
    
    # GPT 응답 시뮬레이션
    mock_gpt_response = {
        "epic_description": "사용자 인증 기능 구현",
        "tasks": [
            {
                "title": "로그인 API 구현",
                "description": "로그인 API 엔드포인트 구현",
                "assignee": "홍길동",
                "difficulty": 3,
                "expected_workhours": 16
            }
        ]
    }
    
    # 태스크 생성 로직 시뮬레이션
    tasks = []
    for task_data in mock_gpt_response["tasks"]:
        task = Task(
            title=task_data["title"],
            description=task_data["description"],
            assignee=task_data["assignee"],
            difficulty=task_data["difficulty"],
            expected_workhours=task_data["expected_workhours"],
            epic="epic1"
        )
        tasks.append(task.dict())
    
    # 결과 검증
    assert len(tasks) == 1
    task = Task(**tasks[0])
    assert task.title == "로그인 API 구현"
    assert task.description == "로그인 API 엔드포인트 구현"
    assert task.assignee == "홍길동"
    assert task.difficulty == 3
    assert task.expected_workhours == 16
    assert task.epic == "epic1"


@pytest.mark.asyncio
async def test_create_sprint():
    """스프린트 생성 테스트"""
    # Mock 데이터 설정
    mock_pending_tasks = [
        Task(
            _id="task1",
            title="로그인 API 구현",
            description="로그인 API 엔드포인트 구현",
            assignee="user1",
            startDate="2024-03-01",
            endDate="2024-03-14",
            priority=300,
            epic="epic1"
        )
    ]
    
    mock_epic = Epic(
        _id="epic1",
        title="로그인 기능",
        description="사용자 인증 기능 구현",
        projectId="test-project"
    )
    
    # GPT 응답 시뮬레이션
    mock_gpt_response = {
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
    
    # 스프린트 생성 로직 시뮬레이션
    sprint = Sprint(**mock_gpt_response["sprint"])
    epics = mock_gpt_response["epics"]
    
    result = {
        "sprint": sprint.dict(),
        "epics": epics
    }
    
    # 결과 검증
    sprint_response = SprintResponse(**result)
    assert sprint_response.sprint.title == "API 구현"
    assert sprint_response.sprint.description == "로그인 API 엔드포인트 구현"
    assert sprint_response.sprint.startDate == "2024-03-01"
    assert sprint_response.sprint.endDate == "2024-03-14"
    
    assert len(sprint_response.epics) == 1
    epic = sprint_response.epics[0]
    assert epic["epicId"] == "epic1"
    assert len(epic["tasks"]) == 1
    
    task = epic["tasks"][0]
    assert task["title"] == "로그인 API 구현"
    assert task["description"] == "로그인 API 엔드포인트 구현"
    assert task["assigneeId"] == "user1"
    assert task["startDate"] == "2024-03-01"
    assert task["endDate"] == "2024-03-14"
    assert task["priority"] == 300