import asyncio
import datetime
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from feature_specification import calculate_priority
from gpt_utils import extract_json_from_gpt_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mongodb_setting import (get_epic_collection, get_feature_collection,
                             get_project_collection, get_task_collection,
                             get_user_collection)
from openai import AsyncOpenAI
from project_member_utils import get_project_members

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

def check_collection_initialized():
    collections = {
        "feature_collection": feature_collection,
        "project_collection": project_collection,
        "epic_collection": epic_collection,
        "task_collection": task_collection,
        "user_collection": user_collection
    }
    
    uninitialized_collections = []
    for name, collection in collections.items():
        if collection is None:
            uninitialized_collections.append(name)
    
    if len(uninitialized_collections) > 0:
        raise ValueError(f"다음의 collection들이 초기화되지 않았습니다: {uninitialized_collections}")
    
    logger.info("✅ 모든 collection이 정상적으로 초기화되었습니다.")
    return True

# db 초기화 함수
async def init_collections():
    global feature_collection, project_collection, epic_collection, task_collection, user_collection
    feature_collection = None
    project_collection = None
    epic_collection = None
    task_collection = None
    user_collection = None
    
    feature_collection = await get_feature_collection()
    project_collection = await get_project_collection()
    epic_collection = await get_epic_collection()
    task_collection = await get_task_collection()
    user_collection = await get_user_collection()
    
    if not check_collection_initialized():
        raise False
    return True

async def calculate_eff_mandays(efficiency_factor: float, number_of_developers: int, sprint_days: int, workhours_per_day: int) -> int:
    logger.info(f"🔍 개발자 수: {number_of_developers}명, 1일 개발 업무시간: {workhours_per_day}시간, 스프린트 주기: {sprint_days}일, 효율성 계수: {efficiency_factor}")
    mandays = number_of_developers * sprint_days * workhours_per_day
    logger.info(f"⚙️  Sprint별 작업 배정 시간: {mandays}시간")
    eff_mandays = round(mandays * efficiency_factor)
    logger.info(f"⚙️  Sprint별 효율적인 작업 배정 시간: {eff_mandays}시간")
    return eff_mandays

async def calculate_percentiles(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''
    "tasks": [
        {{
            "title": "string",
            "description": "string",
            "assignee": "string",
            "startDate": str(YYYY-MM-DD),
            "endDate": str(YYYY-MM-DD),
            "expected_workhours": int,
            "priority": int
        }},
        ...
    ]
    '''
    
    # task별 priority 값을 모아서 percentile 30, 70 추출
    priority_values = [task["priority"] for task in tasks if "priority" in task]
    p30 = np.percentile(priority_values, 30)
    p70 = np.percentile(priority_values, 70)
    
    # 각 task의 priority 값을 분위수에 따라 재조정
    for task in tasks:
        original_priority = task["priority"]
        if original_priority <= p30:
            task["priority"] = 50       # Low
        elif original_priority <= p70:
            task["priority"] = 150      # Medium
        else:
            task["priority"] = 250      # High
    logger.info(f"🔍 H:30, M:40, L:30 비율로 우선순위 재조정한 결과: {tasks}")
    
    return tasks


########## =================== Create Task ===================== ##########
'''
경우마다 서로 다른 context 정보를 사용해서 Task를 구성하게 됨.
1. create_task_from_feature: feature collection에 저장된 UseCase, input, output, priority, workhours, assignee, start & endDate 모두 사용
2. create_task_from_epic: epic title, description & task title, description, assignee, priority, expected_workhours 사용
3. create_task_from_null: project & epic의 description 사용
'''
async def create_task_from_feature(epic_id: str, feature_id: str, project_id: str, workhours_per_day: int) -> List[Dict[str, Any]]:
    logger.info(f"🔍 기존의 feature 정보로부터 task 정의 시작: {feature_id}")
    assert feature_id is not None, "feature로부터 정의된 epic에 대해 task를 정의하는 스텝이므로 feature_id가 존재해야 합니다."
    feature = await feature_collection.find_one({"featureId": feature_id})
    
    task_creation_from_feature_prompt = ChatPromptTemplate.from_template(
    """
    당신은 애자일 마스터입니다. 당신의 주요 언어는 한국어입니다. 당신의 업무는 규칙에 따라 주어진 epic에 대한 정보를 바탕으로 각 epic의 하위 task를 정의하는 것입니다.
    규칙은 다음과 같습니다.
    1. 반드시 하나 이상의 task를 생성해야 합니다. task는 {epic_title}과 {epic_description}를 참고하여 개발할 수 있는 구체적인 수준으로 정의해야 합니다.
    예를 들어 {epic_description}이 "알람 기능 개발"이라면 task의 title은 "알람 API response 정의", task의 description은 "알람 API에서 frontend가 backend에 전송할 response의 body의 내용을 정의"와 같이 구체적으로 작성되어야 합니다.
    2. {workhours_per_day}는 팀원들이 하루에 개발에 사용하는 시간입니다. {epic_expected_workhours} 이하의 값으로 task별 전체 개발 예상 시간을 산정하고, 이를 {workhours_per_day}로 나누어 expected_workhours를 task 별로 정의하세요.
    3. difficulty는 반드시 1 이상 5 이하의 정수여야 합니다. 절대 이 범위를 벗어나지 마세요.
    4. assignee는 반드시 {project_members}에 존재하는 멤버여야 합니다. 절대 이를 어겨선 안됩니다. 반환할 때는 FE, BE와 같은 포지션을 제외하고 이름만 반환하세요. assignee는 반드시 한 명이어야 합니다.
    5. "{epic_endDate} - {epic_startDate} >= expected_workhours" 조건을 만족하는지 검사하세요. 만약 만족하지 못한다면 startDate를 {epic_startDate}, endDate를 {epic_endDate}로 지정하세요.
    6. 만약 5번의 조건을 만족한다면 startDate가 {epic_startDate}보다 빠른지 검사하세요. 빠르다면 startDate를 {epic_startDate}로 지정하세요. 빠르지 않다면 그대로 유지하세요.
    7. 만약 5번의 조건을 만족한다면 endDate가 {epic_endDate}보다 늦은지 검사하세요. 늦다면 endDate를 {epic_endDate}로 지정하세요. 늦지 않다면 그대로 유지하세요.

    현재 프로젝트에 참여 중인 멤버들의 정보는 다음과 같습니다:
    {project_members}
    
    결과를 다음과 같은 형식으로 반환해 주세요.
    {{
        "tasks": [
            {{
                "title": "string",
                "description": "string",
                "assignee": "string",
                "startDate": "YYYY-MM-DD",
                "endDate": "YYYY-MM-DD",
                "difficulty": int,
                "expected_workhours": float
            }},
            ...
        ]
    }}
    """)
    project_members = await get_project_members(project_id)
    assert project_members is not None, "project_members가 존재하지 않습니다."
    
    messages = task_creation_from_feature_prompt.format_messages(
        project_members=project_members,
        epic_title=feature["name"],
        epic_description="사용 시나리오: "+feature["useCase"]+"\n"+"입력 데이터: "+feature["input"]+"\n"+"출력 데이터: "+feature["output"],
        epic_startDate=feature["startDate"],
        epic_endDate=feature["endDate"],
        epic_expected_workhours=feature["expectedDays"],
        workhours_per_day=workhours_per_day
    )
    
    # LLM Config
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.4,
    )
    response = await llm.ainvoke(messages)

    try:
        content = response.content
        try:
            gpt_result = extract_json_from_gpt_response(content)
        except Exception as e:
            logger.error(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True)
            raise Exception(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True) from e
        
    except Exception as e:
        logger.error(f"GPT API 처리 중 오류 발생: {e}", exc_info=True)
        raise Exception(f"GPT API 처리 중 오류 발생: {str(e)}", exc_info=True) from e
    
    task_to_store = []
    tasks = gpt_result["tasks"]
    logger.info("⚙️ gpt가 반환한 결과로부터 task 정보를 추출합니다.")
    for task in tasks:
        task_data = {
            "title": task["title"],
            "description": task["description"],
            "assignee": task["assignee"],
            "startDate": task["startDate"],
            "endDate": task["endDate"],
            "priority": calculate_priority(task["difficulty"], task["expected_workhours"]),
            "epic": epic_id
        }
        if task_data["startDate"] <= feature["startDate"]:
            logger.warning(f"⚠️ task {task['title']}의 startDate가 epic의 startDate보다 이전입니다. 이를 바탕으로 정의된 task의 startDate를 epic의 startDate로 조정합니다.")
            task_data["startDate"] = feature["startDate"]
        if task_data["endDate"] >= feature["endDate"]:
            logger.warning(f"⚠️ task {task['title']}의 endDate가 epic의 endDate보다 이후입니다. 이를 바탕으로 정의된 task의 endDate를 epic의 endDate로 조정합니다.")
            task_data["endDate"] = feature["endDate"]
        task_to_store.append(task_data)
    logger.info(f"🔍 epic {epic_id}에 속한 task 정의 완료: {task_to_store}")
    return task_to_store


async def create_task_from_epic(epic_id: str, project_id: str, task_db_data: List[Dict[str, Any]], workhours_per_day: int) -> List[Dict[str, Any]]:
    logger.info(f"🔍 기존의 epic과 task 정보로부터 task 정의 시작: {epic_id}")
    assert epic_id is not None, "epic에 _id가 없습니다."    # epic은 id가 없으면 안 됨
    assert len(task_db_data) > 0, "task_db_data가 매개변수로 전달되지 않음."
    try:
        epic = await epic_collection.find_one({"_id": epic_id})
    except Exception as e:
        logger.error(f"🚨 epic {epic_id} 조회 중 오류 발생: {e}", exc_info=True)
        raise e

    null_fields = []
    # task의 description, assignee, startDate, endDate, priority 중에 null인 필드가 있는지 확인
    for task in task_db_data:
        if task["description"] is None:
            null_fields.append("description")
        if task["assignee"] is None:
            null_fields.append("assignee")
        if task["startDate"] is None:
            null_fields.append("startDate")
        if task["endDate"] is None:
            null_fields.append("endDate")
        if task["priority"] is None:
            null_fields.append("priority")
    
    task_creation_from_epic_prompt = ChatPromptTemplate.from_template(
    """
    당신은 애자일 마스터입니다. 당신의 주요 언어는 한국어입니다. 당신의 업무는 규칙에 따라 주어진 epic과 epic의 하위 task에 대해 null인 필드의 값을 생성하는 것입니다.
    규칙은 다음과 같습니다.
    1. {epic_description}이 "null"인지 확인하세요. 만약 null이라면 {epic_title}으로부터 {epic_description}을 구성하세요. {epic_title}에 대해 예상되는 사용 시나리오, 입력 데이터, 출력 데이터를 내용으로 포함하세요.
    2. 1번을 마무리 했다면, {null_fields}에 "description", "assignee", "startDate", "endDate", "priority" 중에 어떤 값들이 존재하는지 확인하세요.
    3. 2번에서 확인한 내용별로 다음의 규칙을 지켜서 값을 생성하고 결과를 반환하세요.
    3-1. "description"이 확인된다면 {epic_description}과 task의 title을 참고하여 task의 "description"을 정의하세요.
    예를 들어 {epic_description}이 "알람 기능 개발"이고, task의 title이 "알람 API response 정의"라면, task의 description은 "알람 API에서 frontend가 backend에 전송할 response의 body의 내용을 정의"와 같이 구체적으로 작성되어야 합니다.
    만약 task의 title이 {epic_description}과 관련이 없다면, {epic_description}을 참고하여 task의 description의 생성과 함께 task의 title도 수정하세요.
    3-2. "assignee"가 확인된다면 {project_members}에 존재하는 멤버 중에서 적절한 멤버를 선택하여 task의 "assignee"를 정의하세요.
    assignee는 반드시 {project_members}에 존재하는 멤버여야 합니다. 절대 이를 어겨선 안됩니다. 반환할 때는 FE, BE와 같은 포지션을 제외하고 이름만 반환하세요. assignee는 반드시 한 명이어야 합니다.
    3-3. "priority"가 확인된다면 difficulty와 expected_workhours를 정의하세요.
    difficulty는 반드시 1 이상 5 이하의 정수여야 합니다. 절대 이 범위를 벗어나지 마세요.
    {workhours_per_day}는 팀원들이 하루 중 개발에 사용하는 시간이므로 이를 바탕으로 task 개발에 소요될 것으로 예상되는 시간을 산정한 다음, {workhours_per_day}로 나누어 expected_workhours를 정의하세요.
    4. 마지막으로 가장 중요한 규칙입니다. {null_fields}에 존재하지 않는 task의 모든 필드들은 {task_db_data}에 존재하는 값을 그대로 반환해야 합니다.
    다시 한 번 강조합니다. {null_fields}에 존재하지 않는 task의 모든 필드들은 2번과 3번의 과정과 관련없으므로 {task_db_data}에 존재하는 값을 그대로 반환해야 합니다.
    
    결과를 다음과 같은 형식으로 반환해 주세요.
    {{
        "epic_description": "string",
        "tasks": [
            {{
                "title": "string",
                "description": "string",
                "assignee": "string",
                "difficulty": int,
                "expected_workhours": float
            }},
            ...
        ]
    }}
    """)
    project_members = await get_project_members(project_id)
    assert project_members is not None, "project_members가 존재하지 않습니다."
    
    messages = task_creation_from_epic_prompt.format_messages(
        null_fields = null_fields,
        epic_title = epic["title"],
        epic_description = epic["description"] if epic["description"] is not None else "null",
        task = task_db_data,
        project_members = project_members,
        workhours_per_day = workhours_per_day
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.4,
    )
    response = await llm.ainvoke(messages)
    try:
        content = response.content
        try:
            gpt_result = extract_json_from_gpt_response(content)
        except Exception as e:
            logger.error(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True)
            raise Exception(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True) from e
    except Exception as e:
        logger.error(f"GPT API 처리 중 오류 발생: {e}", exc_info=True)
        raise Exception(f"GPT API 처리 중 오류 발생: {str(e)}", exc_info=True) from e
    
    task_to_store = []
    tasks = gpt_result["tasks"]
    logger.info("⚙️ gpt가 반환한 결과로부터 task 정보를 추출합니다.")
    for task in tasks:
        task_data = {
            "title": task["title"],
            "description": task["description"],
            "assignee": task["assignee"],
            "startDate": "",
            "endDate": "",
            "priority": calculate_priority(task["difficulty"], task["expected_workhours"]),
            "epic": epic_id
        }
        task_to_store.append(task_data)
    logger.info(f"🔍 epic {epic_id}에 속한 task 정의 완료: {task_to_store}")
    epic_description = gpt_result["epic_description"]
    if epic["description"] is None:
        epic["description"] = epic_description
        logger.info(f"🔍 epic {epic['title']}의 description이 공란인 관계로 새롭게 정의된 {epic_description}을 저장합니다.")
    return task_to_store


async def create_task_from_null(epic_id: str, project_id: str, workhours_per_day: int) -> List[Dict[str, Any]]:
    logger.info(f"🔍 null로부터 task 정의 시작: {epic_id}")
    task_creation_from_null_prompt = ChatPromptTemplate.from_template(
    """
    당신은 애자일 마스터입니다. 당신의 주요 언어는 한국어입니다. 당신의 업무는 규칙에 따라 주어진 epic에 대한 정보를 바탕으로 각 epic의 하위 task를 정의하는 것입니다.
    규칙은 다음과 같습니다.
    1. {epic_description}이 "null"이 아니라면 그대로 반환하고, "null"이라면 {project_description}을 참고해서 새롭게 정의한 description을 반환하세요.
    2. task는 {epic_description}을 수행하기 위한 아주 자세한 개발 단위를 정의해야 합니다.
    예를 들어 "epic_description"이 "알람 기능 개발"이라면 task의 title은 "알람 API response 정의", task의 description은 "알람 API에서 frontend가 backend에 전송할 response의 body의 내용을 정의"와 같이 구체적으로 작성되어야 합니다.
    3. task의 assignee는 {project_members}에 존재하는 멤버여야 합니다. 명심하세요. assignee는 반드시 한 명이어야 합니다. 반환할 때는 FE, BE와 같은 포지션을 제외하고 이름만 반환하세요.
    4. difficulty는 1 이상 5 이하의 정수여야 합니다. 절대 이 범위를 벗어나지 마세요.
    5. assignee는 반드시 {project_members}에 존재하는 멤버여야 합니다. 절대 이를 어겨선 안됩니다. 반환할 때는 FE, BE와 같은 포지션을 제외하고 이름만 반환하세요. assignee는 반드시 한 명이어야 합니다.
    6. {workhours_per_day}는 팀원들이 하루 중 개발에 사용하는 시간이므로 이를 바탕으로 task 개발에 소요될 것으로 예상되는 시간을 산정한 다음, {workhours_per_day}로 나누어 expected_workhours를 정의하세요.
    
    결과를 다음과 같은 형식으로 반환해 주세요.
    {{
        "epic_description": "string",
        "tasks": [
            {{
                "title": "string",
                "description": "string",
                "assignee": "string",
                "difficulty": int,
                "expected_workhours": float
            }},
            ...
        ]
    }}
    
    """)
    project_members = await get_project_members(project_id)
    assert project_members is not None, "project_members가 존재하지 않습니다."
    
    project = await project_collection.find_one({"_id": project_id})
    project_description = project["description"]
    logger.info(f"🔍 context로 전달할 project description: {project_description}")
    
    epic = await epic_collection.find_one({"_id": epic_id})
    epic_description = epic["description"]
    logger.info(f"🔍 context로 전달할 epic description: {epic_description}")
    
    messages = task_creation_from_null_prompt.format_messages(
        project_description = project_description,
        epic_description = epic_description if epic_description is not None else "null",
        project_members = project_members,
        workhours_per_day = workhours_per_day
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.4,
    )
    response = await llm.ainvoke(messages)
    try:
        content = response.content
        try:
            gpt_result = extract_json_from_gpt_response(content)
        except Exception as e:
            logger.error(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True)
            raise Exception(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True) from e
    except Exception as e:
        logger.error(f"GPT API 처리 중 오류 발생: {e}", exc_info=True)
        raise Exception(f"GPT API 처리 중 오류 발생: {str(e)}", exc_info=True) from e
    
    task_to_store = []
    tasks = gpt_result["tasks"]
    logger.info("⚙️ gpt가 반환한 결과로부터 task 정보를 추출합니다.")
    for task in tasks:
        task_data = {
            "title": task["title"],
            "description": task["description"],
            "assignee": task["assignee"],
            "startDate": "",
            "endDate": "",
            "priority": calculate_priority(task["difficulty"], task["expected_workhours"]),
            "epic": epic_id
        }
        task_to_store.append(task_data)
    logger.info(f"🔍 epic {epic_id}에 속한 task 정의 완료: {task_to_store}")
    epic_description = gpt_result["epic_description"]
    if epic["description"] is None:
        epic["description"] = epic_description
        logger.info(f"🔍 epic {epic['title']}의 description이 공란인 관계로 새롭게 정의된 {epic_description}을 저장합니다.")
    
    return task_to_store


########## =================== Create Sprint ===================== ##########
'''
Sprint 생성 POST API에 라우팅 되는 함수
다음의 과정을 거쳐서 Sprint를 생성한다.

1. 이번 Sprint에 포함되는 epic들을 projectId로 조회한다. 이때 조회된 epic들이 epic_id를 갖는지 검사한다.
2. projectId를 사용하여 프로젝트 멤버 정보("project_members")를 구성한다.
3. 전체 프로젝트 기간에 따라 sprint_days, workhours_per_day를 정의하고, 정의된 값들을 바탕으로 effective_mandays를 계산한다. (efficiency_factor를 1로 고정: 현재로서는 효율에 대한 coefficient를 고려하지 않음 << 수정된 내용)
4. 각 epic에 대한 task 정보("task_db_data")를 조회한다. 이떄 조회된 task들이 task_id를 갖는지 검사한다.
5. task_db_data에 존재하는 task들을 순회하며 title, description, assignee, priority, startDate, endDate가 null인지 검사한다. project_members과 task_db_data를 입력으로 하여 각 task의 필드 정보를 생성한다.
단, workhours_per_day 정보를 알고 있는 상태에서 expected_workdays를 정의하도록 한다. (!startDate, !endDate)
또한, priority 값 부여 함수가 의도대로 동작하는지 반드시 확인한다. "expected_workhours" ? "(endDate - startDate)"로 정의되는 개발 시간을 80%, 1-5 사이의 값으로 정의되는 개발 난이도를 20% 반영)
6. pendingTaskIds가 task_db_data에 모두 존재하는지 검사한다. 누락된 task는 task_id로 정보를 가져와서 task_db_data에 추가한다.
7. task_db_data를 epic 단위로 묶어서 tasks_by_epic을 정의하고 epic과 epic 내 task를 우선순위("priority") 내림차순 정렬한다.
8. 정렬된 tasks_by_epic을 바탕으로 각 epic의 총 우선순위를 계산하고, epic 단위로 sprint를 확장하면서 epic에 속한 task들의 expected_workhours의 합이 effective_mandays를 초과하지 않는지 검사한다.
9. 첫 번째 sprint에 들어갈 epic을 확정하고, 포함된 task들의 startDate, endDate를 expected_workhours를 바탕으로 정의한다.
이때 startDate 또는 endDate가 존재한다면 해당 값을 그대로 사용하고, 존재하지 않는다면 sprint 시작일을 startDate로 통일한다.
'''

async def create_sprint(project_id: str, pending_tasks_ids: Optional[List[str]], start_date: datetime) -> Dict[str, Any]:
    logger.info(f"🔍 스프린트 생성 시작: {project_id}")
    assert project_id is not None, "project_id가 존재하지 않습니다."
    
    # DB 콜렉션 인스턴스 생성 및 초기화
    initialize_db_collection = await init_collections()
    assert initialize_db_collection is True, "collection 호출 및 초기화에 실패하였습니다. 다시 시도하세요."
    
    ### 1단계: 이번 Sprint에 포함되는 epic들을 projectId로 조회한다. 이때 조회된 epic들이 epic_id를 갖는지 검사한다.
    try:
        epics = await epic_collection.find({"projectId": project_id}).to_list(length=None)  # 모든 epic은 projectId가 존재함
        logger.info(f"🔍 projectId: {project_id}로 조회되는 epic들: {epics}")
    except Exception as e:
        logger.error(f"🚨 epic collection 접근 중 오류 발생: {e}", exc_info=True)
        raise e
    logger.info("✅ MongoDB에서 epic 정보들 로드 완료")
    
    ### 2단계: projectId를 사용하여 프로젝트 멤버 정보("project_members")를 구성한다.
    project_members = await get_project_members(project_id)
    assert project_members is not None, "project_members가 존재하지 않습니다."
    
    ### 3단계: 전체 프로젝트 기간에 따라 sprint_days, workhours_per_day를 정의하고, 정의된 값들을 바탕으로 effective_mandays를 계산한다.
    # 프로젝트 기간 정보 추출
    try:
        project = await project_collection.find_one({"_id": project_id})
        logger.info("✅ 효율적인 작업일수 계산을 위해 프로젝트 정보를 조회합니다.")
    except Exception as e:
        logger.error(f"🚨 MongoDB에서 Project 정보 로드 중 오류 발생: {e}", exc_info=True)
        raise e
    try:
        logger.info(f"🔍 프로젝트 시작일: {project['startDate']}, 프로젝트 종료일: {project['endDate']}")
        project_start_date = project["startDate"]  # 이미 datetime 객체이므로 그대로 사용
        project_end_date = project["endDate"]      # 이미 datetime 객체이므로 그대로 사용
        project_days = (project_end_date - project_start_date).days
    except Exception as e:
        logger.error(f"🚨 프로젝트 기간 계산 중 오류 발생: {e}", exc_info=True)
        raise e
    
    # 프로젝트 기간에 따른 개발팀 1일 작업 시간 지정
    if project_days <= 90:
        logger.info("🔍 프로젝트 기간이 90일 이하입니다. 주 5일 근무, 1일 8시간 개발, 총 주차별 40시간 작업으로 계산합니다.")
        workhours_per_day = 8
        sprint_days = 14
    elif project_days <= 180 and project_days > 90:
        logger.info("🔍 프로젝트 기간이 180일 이하입니다. 주 5일 근무, 1일 6시간 개발, 총 주차별 30시간 작업으로 계산합니다.")
        workhours_per_day = 6
        sprint_days = 14
    elif project_days <= 270 and project_days > 180:
        logger.info("🔍 프로젝트 기간이 270일 이하입니다. 주 5일 근무, 1일 4시간 개발, 총 주차별 20시간 작업으로 계산합니다.")
        workhours_per_day = 4
        sprint_days = 21
    elif project_days <= 365 and project_days > 270:
        logger.info("🔍 프로젝트 기간이 365일 이하입니다. 주 5일 근무, 1일 2시간 개발, 총 주차별 10시간 작업으로 계산합니다.")
        workhours_per_day = 2
        sprint_days = 21
    else:
        logger.info("🔍 프로젝트 기간이 365일 초과입니다. 주 5일 근무, 1일 1시간 개발, 총 주차별 5시간 작업으로 계산합니다.")
        workhours_per_day = 1
        sprint_days = 28
    
    # 프로젝트의 effective mandays 계산
    efficiency_factor = 1.0
    number_of_developers = len(project_members)
    eff_mandays = await calculate_eff_mandays(efficiency_factor, number_of_developers, sprint_days, workhours_per_day)
    
    
    ### 4단계: 각 epic에 대한 task 정보("task_db_data")를 조회한다. 이떄 조회된 task들이 task_id를 갖는지 검사한다.
    ### 만약 featureId가 존재하는 epic이거나 task가 없는 epic이라면 task를 생성하는 로직을 추가로 수행한다.
    captured_tasks=[]
    for epic in epics:
        assert epic["_id"] is not None, "epic에 _id가 없습니다."    # epic은 id가 없으면 안 됨
        epic_id = epic["_id"]
        logger.info(f"🔍 현재 task를 정리 중인 epic: {epic['title']}\n그리고 해당 epic의 id: {epic_id}")
        # 불러온 epic에 딸린 task들의 정보를 점검
        try:
            task_db_data = await task_collection.find({"epic": epic_id}).to_list(length=None)
            logger.info(f'🔍 MongoDB: epic {epic["title"]}에 속한 task 정보: {task_db_data}')
        except Exception as e:
            logger.error(f"🚨 epic {epic['title']}의 task 로드 (MongoDB 사용) 중 오류 발생: {e}", exc_info=True)
            raise e
        # task 정의 상태에 따라 3가지 서로 다른 전략으로 epic 하위 task를 정의
        try:
            if len(task_db_data) == 0:  # 정의된 하위 task가 없는 epic은 task 정보를 생성해야 합니다.
                logger.info(f"❌ epic {epic['title']}의 task 정보가 없습니다. 새로운 task 정보를 구성합니다.")
                if "featureId" in epic and epic["featureId"] is not None:  # featureId가 존재하는 epic
                    logger.info(f"❌ - ✅ epic {epic['title']}에 featureId가 존재합니다. feature 정보로부터 새로운 task 정보를 생성합니다.")
                    feature_id = epic["featureId"]
                    task_defined_from_feature = await create_task_from_feature(epic_id, feature_id, project_id, workhours_per_day)
                    captured_tasks.extend(task_defined_from_feature)
                else:
                    logger.info(f"❌ - ❌ epic {epic['title']}의 featureId가 없습니다. epic 정보로부터 새로운 task 정보를 생성합니다.")
                    task_defined_from_null = await create_task_from_null(epic_id, project_id, workhours_per_day)
                    captured_tasks.extend(task_defined_from_null)
            else:   # 정의된 하위 task가 있는 epic은 기존 task 정보를 사용하되, null인 값을 채워 넣습니다.
                logger.info(f"✅ epic {epic['title']}의 task 정보가 이미 존재합니다. 기존 task 정보를 사용합니다.")
                task_defined_from_epic = await create_task_from_epic(epic_id, project_id, task_db_data, workhours_per_day)
                captured_tasks.extend(task_defined_from_epic)
            logger.info(f"🔍 epic {epic['title']}의 하위 task 정의 결과: {captured_tasks}")
        except Exception as e:
            logger.error(f"🚨 epic {epic['title']}의 하위 task 정의 과정에서 오류 발생: {e}", exc_info=True)
            raise e
        
        # epic의 총합 우선순위를 계산해서 prioritySum 필드로 기입하고, task를 우선순위 내림차순 정렬
        epic_priority_sum = 0
        for task in captured_tasks:
            epic_priority_sum += task["priority"]
        epic["prioritySum"] = epic_priority_sum
        logger.info(f"🔍 Epic {epic['title']}의 우선순위 총합: {epic_priority_sum}")
        captured_tasks.sort(key=lambda x: x["priority"], reverse=True)
        #logger.info(f"🔍⭐️ epic {epic_id}의 '정렬 전' task 개수: {len(tasks)}개")
        #tasks.sort(key=lambda x: x["priority"], reverse=True)
        #logger.info(f"🔍⭐️ epic {epic_id}의 '정렬 후' task 개수: {len(tasks)}개")
        logger.info(f"⚙️ epic {epic['title']}의 우선순위에 따른 tasks 정렬 결과: {captured_tasks}")
    #logger.info(f"✅ 모든 epic에 대한 task들 정의 결과: {tasks}")
    
    # epic 우선순위에 내림차순 정렬
    try:
        #logger.info(f"🔍⭐️ epic 우선순위에 따른 '정렬 전' epic 개수: {len(epics)}개")
        epics.sort(key=lambda x: x["prioritySum"], reverse=True)
        #logger.info(f"🔍⭐️ epic 우선순위에 따른 '정렬 후' epic 개수: {len(epics)}개")
        logger.info(f"⚙️ epic들의 우선순위에 따른 정렬 결과: {epics}")
    except Exception as e:
        logger.error(f"🚨 Epic 우선순위에 따른 정렬 중 오류 발생: {e}", exc_info=True)
        raise e

    ### 6단계: pendingTaskIds가 task_db_data에 모두 존재하는지 검사한다. 누락된 task는 task_id로 정보를 가져와서 task_db_data에 추가한다.
    # pendingTaskIds가 존재할 경우, Id를 하나씩 순회하면서 tasks에서 제외되어 있는 task를 추가하고, priority로 300을 부여하여 제일 앞에 위치시킨다.
    if pending_tasks_ids:
        logger.info(f"🔍 pendingTaskIds가 존재합니다. 이를 바탕으로 tasks에서 제외되어 있는 task를 추가하고, tasks의 제일 앞에 위치시킵니다.")
        for pending_task_id in pending_tasks_ids:
            captured_tasks_ids = [task["_id"] for task in captured_tasks]
            if pending_task_id not in captured_tasks_ids:
                logger.info(f"🔍 pendingTaskId: {pending_task_id}가 tasks에 존재하지 않습니다. 해당 id를 가진 task를 이번 sprint에 추가합니다.")
                try:
                    pending_task = await task_collection.find_one({"_id": pending_task_id})
                    assert pending_task is not None, f"pendingTaskId: {pending_task_id}로 task collection에서 조회되는 정보가 없습니다."
                    epic_id = pending_task["epic"]
                    assert epic_id is not None, f"pendingTaskId: {pending_task_id}에 epicId가 없습니다."
                    # pending_task의 모든 필드를 점검하여 null인 필드가 존재하는지 확인
                    for field, value in pending_task.items():
                        if value is None:
                            logger.info(f"🔍 pendingTaskId: {pending_task_id}의 {field} 필드가 null입니다.")
                            target_pending_task = await create_task_from_epic(epic_id, project_id, pending_task, workhours_per_day)
                            break
                    else:
                        logger.info(f"🔍 pendingTaskId: {pending_task_id}의 모든 필드가 존재하므로 그대로 pending_task를 이번 sprint에 추가합니다.")
                        target_pending_task = pending_task
                except Exception as e:
                    logger.error(f"🚨 pendingTaskId: {pending_task_id}로 task collection에서 조회되는 정보가 없습니다. {e}", exc_info=True)
                    raise e
                try:
                    target_pending_task["priority"] = 300
                    captured_tasks.insert(0, target_pending_task)
                except Exception as e:
                    logger.error(f"🚨 pendingTaskId: {pending_task_id}인 task를 맨 앞에 위치시키는 중 오류 발생: {e}", exc_info=True)
                    raise e
            else:
                logger.info(f"🔍 ✅ pendingTaskId: {pending_task_id}인 task가 이미 tasks에 존재합니다.")


    tasks_by_epic = []
    for epic in epics:
        epic_tasks = {
            "epicId": epic["_id"],
            "tasks": []
        }
        for task in tasks:
            if task["epic"] == epic["_id"]:
                epic_tasks["tasks"].append(task)
        tasks_by_epic.append(epic_tasks)
    assert len(tasks_by_epic) > 0, "tasks_by_epic 정의에 실패했습니다."
    
    logger.warning(f"❗️ tasks_by_epic (에픽 별로 정의된 태스크 목록입니다. 다음의 항목이 중복된 내용 없이 잘 구성되어 있는지 반드시 확인하세요): {tasks_by_epic}")
    
    ### Sprint 정의하기
    sprint_prompt = ChatPromptTemplate.from_template("""
    당신은 애자일 마스터입니다. 당신의 업무는 주어지는 Epic과 Epic별 Task의 정보를 바탕으로 적절한 Sprint Backlog를 생성하는 것입니다.
    명심하세요. 당신의 주요 언어는 한국어입니다.
    다음의 과정을 반드시 순서대로 진행하고 모두 완료해야 합니다.
    1. 현재 설정된 스프린트의 주기는 {sprint_days}일입니다. 날짜 {today}부터 {project_end_date}까지 프로젝트가 진행되므로 {sprint_days} 단위로 총 몇 개의 sprint가 구성될 수 있고, 각 sprint의 시작일과 종료일은 무엇인지 판단하세요.
    2. 각 스프린트에는 {epics}에 정의된 epic이 최소 하나 이상 포함되어야 합니다. 각 epic마다 "epicId" 필드가 존재하고, 각 epic에는 "tasks" 필드가 존재하므로 스프린트에 epic을 배정할 때 해당 epic의 모든 정보를 누락없이 포함하세요.
    3. {epics}는 priority가 높은 순서대로 정렬된 데이터이므로, 각 스프린트에 되도록 제공된 순서대로 epic을 추가하세요.
    4. epic에 포함된 task들의 priority를 점검하세요. 같은 epic에 포함된 task들의 priority는 서로 값이 30 이상씩 차이가 나야 합니다.
    만약 그렇지 않다면, task의 priority를 task가 존재하는 순서대로 300부터 50씩 감소하도록 조정하세요. 반드시 같은 epic에 속한 task들이 서로 같은 priority 값을 가지지 않도록 한 번 더 확인하세요.
    5. 각 epic의 "tasks" 필드에서 "expected_workhours" 필드를 찾아 그 값을 모두 합산하여 sprint별 총 작업량을 계산하세요.
    6. 계산된 총 작업량이 {eff_mandays}를 초과하는지 검사하세요. 만약 초과한다면 모든 task의 expected_workhours를 0.75배로 일괄되게 축소하세요.
    7. 0.75배로 조정된 "expected_workhours"의 합산이 {eff_mandays}를 초과하는지 검토하세요. 초과할 경우, 모든 task의 expected_workhours를 0.5배로 한 번 더 바꾸세요. 초과하지 않는 경우에는 바꿀 필요 없이 다음 단계로 넘어가세요.
    8. sprint_days, eff_mandays, workhours_per_day를 4~6번의 계산 과정에 사용한 값 그대로 반환하세요.
    9. {epics}안에 정의된 epicId는 반드시 그대로 반환하세요. 다시 한 번 말합니다, {epics}안에 정의된 epicId는 절대로 바꾸지 말고 필요한 곳에 그대로 반환하세요.
    10. 스프린트의 description은 해당 스프린트에 포함된 epic들의 성격을 정의할 수 있는 하나의 문장으로 작성하고, 스프린트의 title은 description을 요약하여 제목으로 정의하세요.
    
    결과를 다음과 같은 형식으로 반환하세요. 이때, 만약 startDate와 endDate가 정의되지 않은 task가 존재한다면, sprint와 동일한 시작일, 종료일을 적용하세요.
    반드시 tasks의 모든 field가 값을 가지는지 확인하세요. 또한 priority 값이 중복되는 task가 존재하지 않도록 하세요.
    {{
        "sprint_days": int,
        "eff_mandays": int,
        "workhours_per_day": int,
        "number_of_sprints": int
        "sprints": [
        {{
            "title": "string",
            "description": "string",
            "startDate": str(YYYY-MM-DD),
            "endDate": str(YYYY-MM-DD),
            "epics": [
            {{
                "epicId": "string",
                "tasks": [
                {{
                    "title": "string",
                    "description": "string",
                    "assignee": "string",
                    "startDate": str(YYYY-MM-DD),
                    "endDate": str(YYYY-MM-DD),
                    "expected_workhours": int,
                    "priority": int
                }},
                ...
                ]
            }},
            ...
            ]
        }},
        ...
        ]
    }}
    """)
    
    messages = sprint_prompt.format_messages(
        eff_mandays=eff_mandays,
        sprint_days=sprint_days,
        workhours_per_day=workhours_per_day,
        today=start_date,
        project_end_date=project_end_date,
        epics=tasks_by_epic,
    )
    
    # LLM Config
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.4,
    )
    response = await llm.ainvoke(messages)

    try:
        content = response.content
        try:
            gpt_result = extract_json_from_gpt_response(content)
        except Exception as e:
            logger.error(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True)
            raise Exception(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True) from e
    except Exception as e:
        logger.error(f"GPT API 처리 중 오류 발생: {e}", exc_info=True)
        raise e
    
    # GPT가 정의한 Sprint 정보 검토
    gpt_sprint_days = gpt_result["sprint_days"]
    gpt_workhours_per_day = gpt_result["workhours_per_day"]
    gpt_eff_mandays = gpt_result["eff_mandays"]
    number_of_sprints = gpt_result["number_of_sprints"]
    
    if gpt_sprint_days is None:
        logger.warning(f"⚠️ gpt_result로부터 sprint_days 정보를 추출할 수 없습니다. 기존에 책정된 스프린트 주기: {sprint_days}일을 사용합니다.")
    if gpt_workhours_per_day is None:
        logger.warning(f"⚠️ gpt_result로부터 workhours_per_day 정보를 추출할 수 없습니다. 기존에 책정된 1일 작업 가능 시간: {workhours_per_day}시간을 사용합니다.")
    if gpt_eff_mandays is None:
        logger.warning(f"⚠️ gpt_result로부터 eff_mandays 정보를 추출할 수 없습니다. 기존에 책정된 개발팀의 실제 작업 가능 시간: {eff_mandays}시간을 사용합니다.")
    if number_of_sprints is None:
        logger.warning(f"⚠️ gpt_result로부터 number_of_sprints 정보를 추출할 수 없습니다.")
    
    sprint_days = gpt_sprint_days if gpt_sprint_days is not None else sprint_days
    workhours_per_day = gpt_workhours_per_day if gpt_workhours_per_day is not None else workhours_per_day
    eff_mandays = gpt_eff_mandays if gpt_eff_mandays is not None else eff_mandays
    number_of_sprints = number_of_sprints if number_of_sprints is not None else 1
    
    logger.info(f"⚙️ sprint 한 주기: {sprint_days}일")
    logger.info(f"⚙️ 생성된 총 스프린트의 개수: {number_of_sprints}개")
    logger.info(f"⚙️ 평가된 개발팀의 실제 작업 가능 시간: {eff_mandays}시간")
    logger.info(f"⚙️ 평가된 개발팀의 1일 작업 가능 시간: {workhours_per_day}시간")
    
    
    # eff_mandays 내부에 sprint별로 포함된 task들의 '재조정된 기능별 예상 작업시간'의 총합이 들어오는지 확인
    sprints = gpt_result["sprints"]
    for sprint in sprints:
        assert sprint is not None, "sprint를 감지하지 못하였습니다."
        sum_of_workdays_per_sprint = 0
        epics = sprint["epics"]
        assert len(epics) > 0, "epic의 묶음(epics)을 감지하지 못하였습니다."
        for epic in epics:
            assert epic is not None, "epic을 감지하지 못하였습니다."
            tasks = epic["tasks"]
            assert len(tasks) > 0, "task의 묶음(tasks)을 감지하지 못하였습니다."
            for task in tasks:
                assert task is not None, "task을 감지하지 못하였습니다."
                sum_of_workdays_per_sprint += task["expected_workhours"]
        logger.info(f"⚙️ 스프린트 {sprint['title']}에 포함된 태스크들의 예상 작업 일수의 합: {sum_of_workdays_per_sprint}시간")
        #logger.info(f"⚙️ effective mandays: {eff_mandays}시간")
        if eff_mandays < sum_of_workdays_per_sprint:
            logger.warning(f"⚠️ 스프린트 {sprint['title']}에 포함된 태스크들의 예상 작업 일수의 합이 effective mandays를 초과합니다.")
    logger.info(f"✅ 생성된 모든 스프린트에 포함된 태스크들의 예상 작업 일수의 합이 effective mandays를 초과하지 않습니다.")
    
    name_to_id = {}
    user_collection = await get_user_collection()
    
    # DBRef에서 직접 ID 매핑 생성
    project_data = await project_collection.find_one({"_id": project_id})
    logger.info("🔍 프로젝트 멤버 name:id mapping 시작")
    for member_ref in project_data["members"]:
        try:
            user_id = member_ref.id
            user_info = await user_collection.find_one({"_id": user_id})
            if user_info is None:
                logger.warning(f"⚠️ 사용자 정보를 찾을 수 없습니다: {user_id}")
                continue
            
            name = user_info.get("name")
            if name is None:
                logger.warning(f"⚠️ 사용자 이름이 없습니다: {user_id}")
                continue
                
            # ObjectId를 문자열로 변환
            name_to_id[name] = str(user_id)
            logger.info(f"✅ 사용자 매핑 성공 - 이름: {name}, ID: {str(user_id)}")
        except Exception as e:
            logger.error(f"❌ 사용자 정보 처리 중 오류 발생: {str(e)}", exc_info=True)
            continue
    logger.info(f"📌 생성된 name_to_id 매핑: {name_to_id}")
    
    if not name_to_id:
        raise Exception("사용자 정보를 찾을 수 없습니다. 프로젝트 멤버 정보를 확인해주세요.")
    
    first_sprint = sprints[0]
    logger.info(f"📌 첫 번째 순서의 sprint만 추출 : {first_sprint}")
    
    ### Task 중복 구성 문제 해결하기 !!! ###
    first_sprint_epics = first_sprint["epics"]
    
    priority_list = []
    for epic in first_sprint_epics:
        priority_list.extend([task["priority"] for task in epic["tasks"]])
        logger.info(f"🔍 {epic['epicId']} 소속 tasks들의 priority 값 누적 목록: {priority_list}")
    priority_list = list(set(priority_list))    # set을 사용해서 중복되는 우선순위를 걷어내 보자.
    p30 = np.percentile(priority_list, 30)
    p70 = np.percentile(priority_list, 70)
    logger.info(f"----🔍 priority 목록의 30% 값: {p30}, 70% 값: {p70}----")
    
    for epic in first_sprint_epics:
        for task in epic["tasks"]:
            if task["priority"] <= p30:
                task["priority"] = 50
            elif task["priority"] <= p70:
                task["priority"] = 150
            else:
                task["priority"] = 250
            if task["assignee"] not in name_to_id:
                logger.warning(f"⚠️ 현재 매핑된 사용자 목록: {list(name_to_id.keys())}")
                raise Exception(f"⚠️ {task['title']}의 담당자인 {task['assignee']}가 매핑된 name_to_id에 존재하지 않습니다.")
            logger.info(f"✅ {task['title']}의 담당자인 {task['assignee']}가 매핑된 name_to_id에 존재합니다.")
            try:
                task["assignee"] = name_to_id[task["assignee"]]  # 이름을 ID로 변환
                logger.info(f"✅ name을 id로 변환하였습니다. 현재 task의 assignee의 정보: {task['assignee']}")
            except Exception as e:
                logger.error(f"🚨 name을 id로 변환하는 데에 실패했습니다: {e}", exc_info=True)
                raise e

    logger.info(f"👉👉👉 ❗️ 첫 번째 sprint 반환하기 전에 반드시 task && priority가 중복되는지 확인하세요: {first_sprint}")
    
    # API 응답 반환
    response = {
        "sprint":
        {
            "title": first_sprint["title"],
            "description": first_sprint["description"],
            "startDate": first_sprint["startDate"],
            "endDate": first_sprint["endDate"]
        },
        "epics": [
            {
                "epicId": epic["epicId"],
                "tasks": [
                    {
                        "title": task["title"],
                        "description": task["description"],
                        "assigneeId": task["assignee"],
                        "startDate": task["startDate"],
                        "endDate": task["endDate"],
                        "priority": task["priority"]
                    }
                    for task in epic["tasks"]
                ]
            }
            for epic in first_sprint["epics"]
        ]
    }
    logger.info(f"👉 API 응답 결과: {response}")
    return response
    
if __name__ == "__main__":
    asyncio.run(create_sprint())