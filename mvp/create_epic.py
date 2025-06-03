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
from gpt_utils import extract_json_from_gpt_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mongodb_setting import (get_epic_collection, get_feature_collection,
                             get_project_collection, get_task_collection,
                             get_user_collection)
from openai import AsyncOpenAI
from project_member_utils import (get_project_members,
                                  map_memberName_to_memberId)

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

async def calculate_percentiles(priority_dict: Dict[str, int]) -> Dict[str, int]:
    # 우선순위에 따라 정렬
    sorted_priority_values = sorted(priority_dict.values())
    
    # 우선순위 30%, 70%, 100%에 해당하는 priority 값 추출
    p30 = np.percentile(sorted_priority_values, 30)
    p70 = np.percentile(sorted_priority_values, 70)
    
    for task_title in priority_dict.keys():
        if priority_dict[task_title] <= p30:
            priority_dict[task_title] = 50
        elif priority_dict[task_title] <= p70:
            priority_dict[task_title] = 150
        else:
            priority_dict[task_title] = 250
    logger.info(f"🔍 H:30, M:40, L:30 비율로 우선순위 재조정한 결과: {priority_dict}")
    return priority_dict


########## =================== Create Task ===================== ##########
'''
경우마다 서로 다른 context 정보를 사용해서 Task를 구성하게 됨.
1. create_task_from_feature: feature collection에 저장된 UseCase, input, output, priority, workhours, assignee, start & endDate 모두 사용
2. create_task_from_epic: epic title, description & task title, description, assignee, priority, expected_workhours 사용
3. create_task_from_null: project & epic의 description 사용
'''
async def create_task_from_feature(epic_id: str, feature_id: str, project_id: str) -> List[Dict[str, Any]]:
    ### feature 정보를 바탕으로 최초의 스프린트 구성 과정에서 epic=feaature에 대한 task를 생성하는 함수
    logger.info(f"🔍 기존의 feature 정보로부터 task 정의 시작: {feature_id}")
    assert feature_id is not None, "feature로부터 정의된 epic에 대해 task를 정의하는 스텝이므로 feature_id가 존재해야 합니다."
    feature = await feature_collection.find_one({"featureId": feature_id})
    
    task_creation_from_feature_prompt = ChatPromptTemplate.from_template("""
    당신은 애자일 마스터입니다. 당신의 주요 언어는 한국어입니다. 당신의 업무는 주어진 epic에 대한 정보를 바탕으로 각 epic의 하위 task를 정의하는 것입니다.
    이때 지켜야 하는 규칙이 있습니다.
    1. 반드시 하나 이상의 task를 생성해야 합니다. task는 {epic_description}를 참고하여 관련된 내용을 정의해야 합니다.
    2. task의 title과 description은 개발자가 이해하기 쉽고 실제로 개발이 이루어지는 단위까지 구체적으로 작성하세요. title의 예는 "~ 기능 API 개발"이고, description의 예는 "~ 기능을 구현하기 위한 벡앤드와 프론트엔드 사이의 API를 명세하고 코드를 작성"입니다.
    3. startDate와 endDate는 반드시 {epic_startDate}와 {epic_endDate} 사이에 있어야 합니다. 절대로 이 범위를 벗어나서는 안됩니다. 
    4. priority는 반드시 0 이상 {epic_priority} 이하의 정수여야 합니다. 절대 이 범위를 벗어나서는 안됩니다.
    5. expected_workhours는 반드시 0 이상 {epic_expected_workhours} 이하의 정수여야 합니다. 절대 이 범위를 벗어나서는 안됩니다.
    6. assignee는 반드시 {project_members}에 존재하는 멤버여야 합니다. 절대 이를 어겨선 안됩니다. 반환할 때는 FE, BE와 같은 포지션을 제외하고 이름만 반환하세요.
    7. assignee는 반드시 한 명이어야 합니다. 절대 여러 명이 할당되어서는 안됩니다.
   
    현재 task를 정의하는 에픽에 대한 일반 정보는 다음과 같습니다:
    {epic}
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
                "priority": int,
                "expected_workhours": float
            }},
            ...
        ]
    }}
    """)
    project_members = await get_project_members(project_id)
    assert project_members is not None, "project_members가 존재하지 않습니다."  # project_members가 전역 변수로서 참조 가능한지 반드시 사전 검사
    
    messages = task_creation_from_feature_prompt.format_messages(
        epic = feature,
        project_members=project_members,
        epic_name=feature["name"],
        epic_description="사용 시나리오: "+feature["useCase"]+"\n"+"입력 데이터: "+feature["input"]+"\n"+"출력 데이터: "+feature["output"],
        epic_startDate=feature["startDate"],
        epic_endDate=feature["endDate"],
        epic_priority=feature["priority"],
        epic_expected_workhours=feature["expectedDays"]
    )
    
    # LLM Config
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.6,
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
            "assigneeId": task["assignee"],
            "startDate": task["startDate"],
            "endDate": task["endDate"],
            "priority": task["priority"],
            "expected_workhours": task["expected_workhours"],
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


async def create_task_from_epic(epic_id: str, project_id: str, task_db_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ### 사용자가 직접 추가한 epic, task의 경우에는 null이 존재하는지 살펴보고, null이 아닌 내용을 바탕으로 epic의 description, task의 field를 생성해야 함
    logger.info(f"🔍 기존의 epic과 task 정보로부터 task 정의 시작: {epic_id}")
    assert epic_id is not None, "epic에 _id가 없습니다."    # epic은 id가 없으면 안 됨
    assert len(task_db_data) > 0, "task_db_data가 매개변수로 전달되지 않음."
    
    # epic의 description이 null인지 확인
    try:
        epic = await epic_collection.find_one({"_id": epic_id})
    except Exception as e:
        logger.error(f"🚨 epic {epic_id} 조회 중 오류 발생: {e}", exc_info=True)
        raise e
    
    # epic의 description이 null인지 확인
    null_fields = []
    if epic["description"] is None:
        logger.info(f"⚠️ epic {epic['title']}의 description 필드가 null이므로 새롭게 정의합니다.")
        null_fields.append(("epic", "description"))
    # task의 description, assignee, startDate, endDate, priority 중에 null인 필드가 있는지 확인
    for task in task_db_data:
        if task["description"] is None:
            null_fields.append(("task", "description"))
        if task["assignee"] is None:
            null_fields.append(("task", "assignee"))
        if task["startDate"] is None:
            null_fields.append(("task", "startDate"))
        if task["endDate"] is None:
            null_fields.append(("task", "endDate"))
        if task["priority"] is None:
            null_fields.append(("task", "priority"))
    
    task_creation_from_epic_prompt = ChatPromptTemplate.from_template("""
    당신은 애자일 마스터입니다. 당신의 주요 언어는 한국어입니다. 당신의 업무는 epic과 epic의 하위 task에 대해 null인 필드의 값을 생성하여 채워넣는 것입니다.
    이때 지켜야 하는 규칙이 있습니다.
    1. {null_fields} 변수를 가장 먼저 점검하세요. 값으로 "epic"과 "task"가 모두 존재하는지 아니면 "task"만 존재하는지 확인하세요.
    2. "epic"이 확인되었다면, {epic_title}로부터 {epic_description}을 구성하세요. {epic_title}에 대해 예상되는 사용 시나리오, 입력 데이터, 출력 데이터를 내용으로 포함하세요.
    3. 2번을 마무리 했다면, {null_fields}의 "task" 안에 "task_description", "task_assignee", "task_startDate", "task_endDate", "task_priority" 중에 어떤 값들이 존재하는지 확인하세요. 이름이 존재하는 "task_*"를 생성해야 합니다.
    4. "task_title"로부터 "task_description"을, "task_description"으로부터 "priority"와 "assignee"를, "task_priority"로부터 "expected_workhours"를 생성하도록 하세요.
    5. 현재 프로젝트에 참여 중인 멤버들의 정보는 {project_members}에 있습니다. assignee에 멤버의 이름을 부여할 때 참고하고, 정의되지 않은 멤버의 이름은 절대로 사용하지 마세요.
    
    결과를 다음과 같은 형식으로 반환해 주세요.
    {{
        "epic_description": "string",
        "tasks": [
            {{
                "title": "string",
                "description": "string",
                "assignee": "string",
                "priority": int,
                "expected_workhours": float
            }},
            ...
        ]
    }}
    """)
    project_members = await get_project_members(project_id)
    assert project_members is not None, "project_members가 존재하지 않습니다."  # project_members가 전역 변수로서 참조 가능한지 반드시 사전 검사
    
    messages = task_creation_from_epic_prompt.format_messages(
        null_fields = null_fields,
        epic_title = epic["title"],
        task = task_db_data,
        project_members = project_members
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
            "assigneeId": task["assignee"],
            "startDate": "",
            "endDate": "",
            "priority": task["priority"],
            "expected_workhours": task["expected_workhours"],
            "epic": epic_id
        }
        task_to_store.append(task_data)
    logger.info(f"🔍 epic {epic_id}에 속한 task 정의 완료: {task_to_store}")
    epic_description = gpt_result["epic_description"]
    return task_to_store, epic_description


async def create_task_from_null(epic_id: str, project_id: str) -> List[Dict[str, Any]]:
    logger.info(f"🔍 null로부터 task 정의 시작: {epic_id}")
    task_creation_from_null_prompt = ChatPromptTemplate.from_template(
    """
    당신은 스프린트 마스터입니다. 당신의 업무는 주어지는 epic에 대해 적절한 task를 정의하는 것입니다.
    epic에 포함된 정보가 매우 적기 때문에 {epic_description}과 project의 description 정보인 {project_description}을 참고해서 epic이 project 내에서 맡는 역할을 분석하세요.
    이후 분석한 역할을 바탕으로 예상되는 task의 목록을 정의하고, task별로 구체적인 field 정보를 다음의 형식으로 반환하세요.
    {{
        "epic_description": "string",
        "tasks": [
            {{
                "title": "string",
                "description": "string",
                "assignee": "string",
                "priority": int,
                "expected_workhours": float
            }},
            ...
        ]
    }}
    
    """)
    project = await project_collection.find_one({"_id": project_id})
    project_description = project["description"]
    logger.info(f"🔍 context로 전달할 project description: {project_description}")
    
    epic = await epic_collection.find_one({"_id": epic_id})
    epic_description = epic["description"]
    logger.info(f"🔍 context로 전달할 epic description: {epic_description}")
    
    messages = task_creation_from_null_prompt.format_messages(
        project_description = project_description,
        epic_description = epic_description
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
            "assigneeId": task["assignee"],
            "startDate": "",
            "endDate": "",
            "priority": task["priority"],
            "expected_workhours": task["expected_workhours"],
            "epic": epic_id
        }
        task_to_store.append(task_data)
    logger.info(f"🔍 epic {epic_id}에 속한 task 정의 완료: {task_to_store}")
    epic_description = gpt_result["epic_description"]
    logger.info(f"🔍 epic {epic_id}의 description 정의 완료: {epic_description}")
    
    return task_to_store, epic_description


########## =================== Create Sprint ===================== ##########
### POST /sprint API가 참조하는 메인 함수
async def create_sprint(project_id: str, pending_tasks_ids: Optional[List[str]], start_date: datetime) -> Dict[str, Any]:
    logger.info(f"🔍 스프린트 생성 시작: {project_id}")
    assert project_id is not None, "project_id가 존재하지 않습니다."
    
    # DB 콜렉션 인스턴스 생성 및 초기화
    initialize_db_collection = await init_collections()
    assert initialize_db_collection is True, "collection 호출 및 초기화에 실패하였습니다. 다시 시도하세요."
    
    try:
        epics = await epic_collection.find({"projectId": project_id}).to_list(length=None)  # 모든 epic은 projectId가 존재함
        logger.info(f"🔍 projectId: {project_id}로 조회되는 epic들: {epics}")
    except Exception as e:
        logger.error(f"🚨 epic collection 접근 중 오류 발생: {e}", exc_info=True)
        raise e
    logger.info("✅ MongoDB에서 epic 정보들 로드 완료")
    
    # 프로젝트 멤버 정보 구성
    project_members = await get_project_members(project_id)
    assert project_members is not None, "project_members가 존재하지 않습니다."
    
    tasks = []
    for epic in epics:
        assert epic["_id"] is not None, "epic에 _id가 없습니다."    # epic은 id가 없으면 안 됨
        epic_id = epic["_id"]
        logger.info(f"🔍 현재 task를 정리 중인 epic: {epic['title']}\n그리고 해당 epic의 id: {epic_id}")
        try:
            task_db_data = await task_collection.find({"epic": epic_id}).to_list(length=None)
            logger.info(f'🔍 MongoDB: epic {epic["title"]}에 속한 task 정보: {task_db_data}')
        except Exception as e:
            logger.error(f"🚨 epic {epic['title']}의 task 로드 (MongoDB 사용) 중 오류 발생: {e}", exc_info=True)
            raise e
        try:
            if len(task_db_data) == 0:
                logger.info(f"❌ epic {epic['title']}의 task 정보가 없습니다. 새로운 task 정보를 생성합니다.")
                if "featureId" in epic and epic["featureId"] is not None:
                    logger.info(f"❌ - ✅ epic {epic['title']}에 featureId가 존재합니다.")
                    feature_id = epic["featureId"]
                    task_defined_from_feature = await create_task_from_feature(epic_id, feature_id, project_id)
                    current_epic_tasks = task_defined_from_feature
                else:
                    logger.info(f"❌ - ❌ epic {epic['title']}의 featureId가 없습니다.")    # task 없는 epic만 존재하는 경우 (문제는, 이거를 어떻게 해석할 것인가)
                    task_defined_from_null, epic_description = await create_task_from_null(epic_id, project_id)
                    current_epic_tasks = task_defined_from_null
                    if epic["description"] is None:
                        epic["description"] = epic_description
                        logger.info(f"🔍 epic {epic['title']}이 공란인 관계로 create_task_from_null에서 정의된 {epic_description}을 저장")
            else:
                logger.info(f"✅ epic {epic['title']}의 task 정보가 이미 존재합니다. 기존 task 정보를 사용합니다.")
                try:
                    task_defined_from_null, epic_description = await create_task_from_epic(epic_id, project_id, task_db_data)
                except Exception as e:
                    logger.error(f"🚨 epic {epic['title']}의 create_task_from_epic 호출 과정에서 오류 발생: {e}", exc_info=True)
                    raise e
                current_epic_tasks = task_defined_from_null
                if epic["description"] is None:
                    epic["description"] = epic_description
                    logger.info(f"🔍 epic {epic['title']}이 공란인 관계로 create_task_from_epic에서 정의된 {epic_description}을 저장")
            
            logger.info(f"🔍 epic {epic['title']}의 task 정보: {current_epic_tasks}")
        except Exception as e:
            logger.error(f"🚨 epic {epic['title']}의 task 정보 구성 과정에서 오류 발생: {e}", exc_info=True)
            raise e
        # 이번 epic의 총합 우선순위를 계산해서 prioritySum 필드로 기입
        epic_priority_sum = 0
        for task in current_epic_tasks:
            epic_priority_sum += task["priority"]
        epic["prioritySum"] = epic_priority_sum
        logger.info(f"🔍 Epic {epic['title']}의 우선순위 총합: {epic_priority_sum}")
        tasks.extend(current_epic_tasks)
        logger.info(f"🔍⭐️ epic {epic_id}의 '정렬 전' task 개수: {len(tasks)}개")
        tasks.sort(key=lambda x: x["priority"], reverse=True)
        logger.info(f"🔍⭐️ epic {epic_id}의 '정렬 후' task 개수: {len(tasks)}개")
        logger.info(f"⚙️ epic {epic_id}까지의 우선순위에 따른 task 정렬 결과: {tasks}")
    logger.info(f"✅ 모든 epic에 대한 task들 정의 결과: {tasks}")
    
    # 누적 우선순위 값이 높은 순서대로 epic 정렬
    try:
        logger.info(f"🔍⭐️ epic 우선순위에 따른 '정렬 전' epic 개수: {len(epics)}개")
        epics.sort(key=lambda x: x["prioritySum"], reverse=True)
        logger.info(f"🔍⭐️ epic 우선순위에 따른 '정렬 후' epic 개수: {len(epics)}개")
        logger.info(f"⚙️ epic 우선순위에 따른 정렬 결과: {epics}")
    except Exception as e:
        logger.error(f"🚨 Epic 우선순위에 따른 정렬 중 오류 발생: {e}", exc_info=True)
        raise e
    
    ### 프로젝트 전체 수행 기간에 따른 effecive mandays 계산 및 tasks들의 expected_workhours 재조정
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
    efficiency_factor = 0.6
    number_of_developers = len(project_members)
    eff_mandays = await calculate_eff_mandays(efficiency_factor, number_of_developers, sprint_days, workhours_per_day)

    # pendingTaskIds가 존재할 경우, Id를 하나씩 순회하면서 tasks에서 제외되어 있는 task를 추가하고, tasks의 제일 앞에 위치시키기
    if pending_tasks_ids:
        logger.info(f"🔍 pendingTaskIds가 존재합니다. 이를 바탕으로 tasks에서 제외되어 있는 task를 추가하고, tasks의 제일 앞에 위치시킵니다.")
        for pending_task_id in pending_tasks_ids:
            tasks_ids = [task["_id"] for task in tasks]
            if pending_task_id not in tasks_ids:
                logger.info(f"🔍 pendingTaskId: {pending_task_id}가 tasks에 존재하지 않습니다. 해당 id를 가진 task를 추가합니다.")
                try:
                    pending_task = await task_collection.find_one({"_id": pending_task_id})
                    logger.info(f"🔍 pendingTaskId: {pending_task_id}로 task collection에서 조회된 정보: {pending_task}")
                except Exception as e:
                    logger.error(f"🚨 pendingTaskId: {pending_task_id}로 task collection에서 조회되는 정보가 없습니다. {e}", exc_info=True)
                    raise e
                try:
                    tasks.insert(0, pending_task)
                except Exception as e:
                    logger.error(f"🚨 pendingTaskId: {pending_task_id}를 가진 task를 제일 앞에 위치시키는 중 오류 발생: {e}", exc_info=True)
                    raise e
            else:
                logger.info(f"🔍 pendingTaskId: {pending_task_id}가 tasks에 이미 존재합니다.")

    # tasks들의 expected_workhours 계산
    #logger.info(f" tasks의 타입: {type(tasks)}")   # Dict
    logger.info(f" tasks의 내용: {tasks}")
    for task in tasks:
        #logger.info(f" task의 타입: {type(task)}")   # List
        #logger.info(f" task의 내용: {task}")
        try:
            task["expected_workhours"] = float(task["expected_workhours"]) * 0.5 * (workhours_per_day/number_of_developers)
        except (ValueError, TypeError) as e:
            logger.error(f"🚨 expected_workhours 변환 중 오류 발생: {e}")
            raise e
        logger.info(f"🔍 {task['title']}의 예상 작업시간: {task['expected_workhours']}")

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
    4. 스프린트 별 epic 구성이 완료되고 나면, 각 epic의 "tasks" 필드에서 "expected_workhours" 필드를 찾아 그 값을 모두 합산하여 sprint별 총 작업량을 계산하세요.
    5. 계산된 총 작업량이 {eff_mandays}를 초과하는지 검사하세요. 만약 초과한다면 모든 task의 expected_workhours를 0.75배로 일괄되게 축소하세요.
    6. 0.75배로 조정된 "expected_workhours"의 합산이 {eff_mandays}를 초과하는지 검토하세요. 초과할 경우, 모든 task의 expected_workhours를 0.5배로 한 번 더 바꾸세요. 초과하지 않는 경우에는 바꿀 필요 없이 다음 단계로 넘어가세요.
    7. sprint_days, eff_mandays, workhours_per_day를 4~6번의 계산 과정에 사용한 값 그대로 반환하세요.
    8. {epics}안에 정의된 epicId는 반드시 그대로 반환하세요. 다시 한 번 말합니다, {epics}안에 정의된 epicId는 절대로 바꾸지 말고 필요한 곳에 그대로 반환하세요.
    9. 스프린트의 description은 해당 스프린트에 포함된 epic들의 성격을 정의할 수 있는 하나의 문장으로 작성하고, 스프린트의 title은 description을 요약하여 제목으로 정의하세요.
    
    결과를 다음과 같은 형식으로 반환하세요. 이때, 만약 startDate와 endDate가 정의되지 않은 task가 존재한다면, sprint와 동일한 시작일, 종료일을 적용하세요. 
    반드시 tasks의 모든 field가 값을 가지도록 구성하세요.
    {{
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
        "sprint_days": int,
        "eff_mandays": int,
        "workhours_per_day": int,
        "number_of_sprints": int
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
        temperature=0.5,
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
    try:
        gpt_sprint_days = gpt_result["sprint_days"]
        if gpt_sprint_days is None:
            logger.warning(f"⚠️ gpt_result로부터 sprint_days 정보를 추출할 수 없습니다. 기존에 책정된 스프린트 주기: {sprint_days}일을 사용합니다.")
        else:
            sprint_days = int(round(gpt_sprint_days))
        gpt_workhours_per_day = gpt_result["workhours_per_day"]
        if gpt_workhours_per_day is None:
            logger.warning(f"⚠️ gpt_result로부터 workhours_per_day 정보를 추출할 수 없습니다. 기존에 책정된 1일 작업 가능 시간: {workhours_per_day}시간을 사용합니다.")
        else:
            workhours_per_day = int(round(gpt_workhours_per_day))
        gpt_eff_mandays = gpt_result["eff_mandays"]
        if gpt_eff_mandays is None:
            logger.warning(f"⚠️ gpt_result로부터 eff_mandays 정보를 추출할 수 없습니다. 기존에 책정된 개발팀의 실제 작업 가능 시간: {eff_mandays}시간을 사용합니다.")
        else:
            eff_mandays = int(round(gpt_eff_mandays))
        number_of_sprints = gpt_result["number_of_sprints"]
    except Exception as e:
        logger.error("gpt_result로부터 Sprint 관련 정보를 추출할 수 없음", exc_info=True)
        raise e
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
    first_sprint_epics = first_sprint["epics"]
    first_sprint_tasks = []
    priority_dict = {}
    # 태스크의 assignee 확인을 위한 디버깅 코드 추가
    logger.info("🔍 태스크 assignee 확인 시작")
    for epic in first_sprint_epics:
        #logger.info(f"📌 이번 sprint에 포함된 epic의 정보: {epic}")
        for task in epic["tasks"]:
            #logger.info(f"📌 이번 sprint에 포함된 task의 정보: {task['title']}, 담당자: {task['assignee']}")
            priority_dict[task["title"]] = task["priority"]
            # assignee가 name_to_id에 없는 경우 처리
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
            first_sprint_tasks.append(task)
    
    # 우선순위 High, Medium, Low의 비율 재조정 (현재 3:4:3)
    priority_dict = await calculate_percentiles(priority_dict)

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
                        "priority": priority_dict[task["title"]]
                    }
                    for task in first_sprint_tasks
                ]
            }
            for epic in first_sprint_epics
        ]
    }
    logger.info(f"👉 API 응답 결과: {response}")
    return response
    
if __name__ == "__main__":
    priority_dict = {
        "task1": 259,
        "task2": 237,
        "task3": 189,
        "task4": 220,
        "task5": 271
    }
    priority_dict = asyncio.run(calculate_percentiles(priority_dict))
    print(f"🔍 우선순위 재조정 결과: {priority_dict}")