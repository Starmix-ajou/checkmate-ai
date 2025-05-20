import asyncio
import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from gpt_utils import extract_json_from_gpt_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mongodb_setting import (get_epic_collection, get_feature_collection,
                             get_project_collection, get_task_collection,
                             get_user_collection)
from openai import AsyncOpenAI
from redis_setting import load_from_redis, save_to_redis

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

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

async def calculate_eff_mandays(efficiency_factor: float, number_of_developers: int, sprint_days: int, workhours_per_day: int) -> float:
    logger.info(f"🔍 개발자 수: {number_of_developers}명, 1일 개발 업무시간: {workhours_per_day}시간, 스프린트 주기: {sprint_days}일, 효율성 계수: {efficiency_factor}")
    mandays = number_of_developers * sprint_days * workhours_per_day
    logger.info(f"⚙️  Sprint별 작업 배정 시간: {mandays}시간")
    eff_mandays = mandays * efficiency_factor
    logger.info(f"⚙️  Sprint별 효율적인 작업 배정 시간: {eff_mandays}시간")
    return eff_mandays

########## =================== Create Task ===================== ##########
### feature에 epicId가 추가되었으므로 epic별로 task를 정의
### 이때 task는 title, description, assignee, startDate, endDate, priority, expected_workhours, epicId를 포함해야 함.
async def create_task(project_id: str, epic_id: str) -> List[Dict[str, Any]]:
    logger.info(f"🔍 task 정의 시작: {epic_id}")
    try:
        feature = await feature_collection.find_one({"epicId": epic_id})
    except Exception as e:
        logger.error(f"MongoDB에서 epic(epicId: {epic_id}) 정보 로드 중 오류 발생: {e}", exc_info=True)
        raise e
    
    print(f"[DEBUG] epic_id: {epic_id}")
    print(f"[DEBUG] features keys: {feature['epicId']}")
    if feature is None:
        raise ValueError(f"Feature not found for epic_id={epic_id}")
    epic = feature
    
    task_creation_prompt = ChatPromptTemplate.from_template("""
    당신은 애자일 마스터입니다. 당신의 주요 언어는 한국어입니다. 당신의 업무는 주어진 epic에 대한 정보를 바탕으로 각 epic의 하위 task를 정의하는 것입니다.
    이때 지켜야 하는 규칙이 있습니다.
    1. 반드시 하나 이상의 task를 생성해야 합니다. task를 생성할 때 그 내용이 {epic_description}과 관련이 있어야 합니다.
    2. task의 이름을 자연어로 정의해 주세요. task는 {epic_name}과 유사한 방식으로 정의하세요.
    3. startDate와 endDate는 반드시 {epic_startDate}와 {epic_endDate} 사이에 있어야 합니다. 절대로 이 범위를 벗어나서는 안됩니다.
    4. priority는 반드시 0 이상 {epic_priority} 이하의 정수여야 합니다. 절대 이 범위를 벗어나서는 안됩니다.
    5. expected_workhours는 반드시 0 이상 {epic_expected_workhours} 이하의 정수여야 합니다. 절대 이 범위를 벗어나서는 안됩니다.
    6. assignee는 반드시 {project_members}에 존재하는 멤버여야 합니다. 절대 이를 어겨선 안됩니다. 반환할 때는 FE, BE와 같은 포지션을 제외하고 이름만 반환하세요.
    7. assignee는 반드시 한 명이어야 합니다. 절대 여러 명이 할당되어서는 안됩니다.
    8. {epic_id}는 반드시 절대로 바꾸지 말고 주어진 값을 그대로 "epic" 필드에 기입하세요.
    
    결과를 다음과 같은 형식으로 반환해 주세요.
    {{
        "tasks": [
            {{
                "title": "댓글 추가 API 개발",
                "description": "댓글을 추가하기 위한 벡앤드와 프론트엔드 사이의 API를 명세하고 코드를 작성",
                "assignee": "홍길동",
                "startDate": "2024-03-01",
                "endDate": "2024-03-03",
                "priority": 100,
                "expected_workhours": 1,
                "epic": "epic_id"
            }},
            ...
        ]
    }}
    
    현재 task를 정의하는 에픽에 대한 일반 정보:
    {epic}
    
    현재 epic의 id:
    {epic_id}
    
    현재 프로젝트 멤버 정보:
    {project_members}
    """)
    
    messages = task_creation_prompt.format_messages(
        epic=epic,
        project_members=project_members,
        epic_name=feature["name"],
        epic_description="사용 시나리오: "+feature["useCase"]+"\n"+"입력 데이터: "+feature["input"]+"\n"+"출력 데이터: "+feature["output"],
        epic_startDate=feature["startDate"],
        epic_endDate=feature["endDate"],
        epic_priority=feature["priority"],
        epic_expected_workhours=feature["expectedDays"],
        epic_id=epic_id
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
            "assignee": task["assignee"],
            "startDate": task["startDate"],
            "endDate": task["endDate"],
            "priority": task["priority"],
            "expected_workhours": task["expected_workhours"],
            "epic": task["epic"]
        }
        task_to_store.append(task_data)
    
    logger.info(f"🔍 epic {epic_id}에 속한 task 정의 완료: {task_to_store}")
    return task_to_store


########## =================== Create Sprint ===================== ##########
async def create_sprint(project_id: str, pending_tasks_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    logger.info(f"🔍 스프린트 생성 시작: {project_id}")
    await init_collections()
    
    try:
        epics = await epic_collection.find({"projectId": project_id}).to_list(length=None)
        logger.info(f"epic collection에 존재하는 epic 정보: {epics}")
    except Exception as e:
        logger.error(f"epic collection 접근 중 오류 발생: {e}", exc_info=True)
        raise e
    logger.info("✅ MongoDB에서 projectId가 일치하는 epic 정보들 로드 완료")
    
    ### ===== project_members를 "global"로 선언함 ===== ####
    global project_members
    project_members = []
    try:
        project_data = await project_collection.find_one({"_id": project_id})
        if not project_data:
            logger.error(f"projectId {project_id}에 해당하는 프로젝트를 찾을 수 없습니다.")
            raise Exception(f"projectId {project_id}에 해당하는 프로젝트를 찾을 수 없습니다.")
        
        logger.info(f"프로젝트 데이터: {project_data}")
        
        for member in project_data.get("members", []):
            try:
                name = member.get("name")
                profiles = member.get("profile", [])  # "profiles" -> "profile"로 수정
                for profile in profiles:
                    if profile.get("projectId") == project_data.get("_id"):  # "projectId" -> "_id"로 수정
                        logger.info(f">> projectId가 일치하는 profile이 존재함: {name}")
                        position = profile.get("position", "")  # "positions" -> "position"으로 수정
                        member_info = [name, position]
                        project_members.append(", ".join(str(item) for item in member_info))
                        logger.info(f"추가된 멤버: {name}, {position}")
            except Exception as e:
                logger.error(f"멤버 정보 처리 중 오류 발생: {str(e)}", exc_info=True)
                continue
    
    except Exception as e:
        logger.error(f"MongoDB에서 Project 정보 로드 중 오류 발생: {e}", exc_info=True)
        raise e
    
    logger.info(f"📌 project_members: {project_members}")
    assert len(project_members) > 0, "project_members가 비어있습니다."
    
    tasks = []
    for epic in epics:
        try:
            epic_id = epic["_id"]
            logger.info(f"🔍 현재 task를 정리 중인 epic의 id: {epic_id}")
        except Exception as e:
            logger.error(f"🚨 epic에 id가 선언되어 있지 않습니다.", exc_info=True)
            raise e
        try:
            task_db_data = await task_collection.find({"epic": epic_id}).to_list(length=None)
            logger.info(f'🔍 MongoDB: epic {epic_id}에 속한 task 정보: {task_db_data}')
        except Exception as e:
            logger.error(f"🚨 epic {epic_id}의 task 로드 (MongoDB 사용) 중 오류 발생: {e}", exc_info=True)
            raise e
        try:
            if len(task_db_data) == 0:
                logger.info(f"❌ epic {epic_id}의 task 정보가 없습니다. 새로운 task 정보를 생성합니다.")
                task_creation_result = await create_task(project_id, epic_id)  # 여기에서 epic collection에 들어있는 epic 정보들로부터 각 epic에 속한 task들을 정의
                current_epic_tasks = task_creation_result
            else:
                logger.info(f"✅ epic {epic_id}의 task 정보가 이미 존재합니다. 기존 task 정보를 사용합니다.")
                current_epic_tasks = task_db_data
            logger.info(f"🔍 epic {epic_id}의 task 정보: {current_epic_tasks}")
        except Exception as e:
            logger.error(f"🚨 epic {epic_id}의 task 정보 구성 과정에서 오류 발생: {e}", exc_info=True)
            raise e
        # 이번 epic의 총합 우선순위를 계산해서 prioritySum 필드로 기입
        epic_priority_sum = 0
        for task in current_epic_tasks:
            epic_priority_sum += task["priority"]
        epic["prioritySum"] = epic_priority_sum
        logger.info(f"🔍 Epic {epic['title']}의 우선순위 총합: {epic_priority_sum}")
        tasks.extend(current_epic_tasks)
        tasks.sort(key=lambda x: x["priority"], reverse=True)
        logger.info(f"⚙️ epic {epic_id}까지의 우선순위에 따른 task 정렬 결과: {tasks}")
    logger.info(f"✅ 모든 epic에 대한 task들 정의 결과: {tasks}")
    
    # 누적 우선순위 값이 높은 순서대로 epic 정렬
    try:
        epics.sort(key=lambda x: x["prioritySum"], reverse=True)
        logger.info(f"✅ Epic 우선순위에 따른 정렬 완료: {epics}")
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
        project_start_date = datetime.strptime(project["startDate"], "%Y-%m-%d %H:%M:%S")
        project_end_date = datetime.strptime(project["endDate"], "%Y-%m-%d %H:%M:%S")
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
    number_of_developers = len(project["members"])
    eff_mandays = await calculate_eff_mandays(efficiency_factor, number_of_developers, sprint_days, workhours_per_day)

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
    
    ### Sprint 정의하기
    sprint_prompt = ChatPromptTemplate.from_template("""
    당신은 애자일 마스터입니다. 당신의 업무는 주어지는 Epic과 Epic별 Task의 정보를 바탕으로 적절한 Sprint Backlog를 생성하는 것입니다.
    명심하세요. 당신의 주요 언어는 한국어입니다.
    다음의 과정을 반드시 순서대로 진행하고 모두 완료해야 합니다.
    1. 현재 설정된 스프린트의 주기는 {sprint_days}일입니다. {project_start_date}와 {project_end_date}를 사용해서 전체 스프린트의 개수와 각 스프린트의 시작일, 종료일을 먼저 구성하세요.
    2. 각 스프린트에는 {epics}로부터 정의된 epic들이 포함되어야 합니다. 각 epic마다 "epicId" 필드가 존재하고, 각 epic에는 "tasks" 필드가 존재합니다. 스프린트에 epic을 추가했다면 해당 epic의 모든 정보를 함께 포함하세요.
    3. {epics}는 priority가 높은 순서대로 이미 정렬된 데이터이므로, 각 스프린트에 해당 정보들을 정리할 때 되도록 순서대로 정리하세요. {epics}는 반환 형식에서 정의된 형식과 동일한 형식으로 정의되어 있음을 참고하세요.
    4. sprint의 구성이 완료되었다면 각 epic에서 "tasks" 필드 하위에 딕셔너리의 리스트로 정의된 모든 task의 "expected_workhours" 필드를 모두 합산하여 해당 스프린트의 총 작업량을 계산하세요.
    5. 계산된 총 작업량이 {eff_mandays}를 초과하는지 검사하세요. 만약 초과한다면 초과된 작업량을 줄이기 위해 각 task의 expected_workhours를 조정하세요.
    6. 한 번 더 조정된 작업량이 {eff_mandays}를 초과하지 않는지 검토하세요. 만약 초과한다면 초과된 작업량을 줄이기 위해 각 task의 expected_workhours를 한 번 더 조정하세요.
    7. sprint_days, eff_mandays, workhours_per_day를 계산에 사용한 값 그대로 반환하세요. 
    8. epicId는 반드시 절대로 바꾸지 마세요. 다시 한 번 말합니다, epicId는 절대로 바꾸지 말고 필요한 곳에 그대로 반환하세요.
    
    결과를 다음과 같은 형식으로 반환하세요.
    {{
        "sprints": [
        {{
            "title": "스프린트 1",
            "description": "스프린트 1은 댓글 관련 기능들을 개발하는 스프린트입니다.",
            "startDate": str(YYYY-MM-DD),
            "endDate": str(YYYY-MM-DD),
            "epics": [
            {{
                "epicId": "string",
                "tasks": [
                {{
                    "title": "댓글 추가 API 개발",
                    "description": "댓글을 추가하기 위한 벡앤드와 프론트엔드 사이의 API를 명세하고 코드를 작성",
                    "assignee": "Alicia",
                    "startDate": str(YYYY-MM-DD),
                    "endDate": str(YYYY-MM-DD),
                    "expected_workhours": 1,
                    "priority": 100
                }},
                ...
                ]
            }},
            ...
            ]
        }},
        ...
        ]
        "sprint_days": 14,
        "eff_mandays": 100,
        "workhours_per_day": 8,
        "number_of_sprints": 1
    }}
    """)
    
    messages = sprint_prompt.format_messages(
        eff_mandays=eff_mandays,
        sprint_days=sprint_days,
        project_days=project_days,
        workhours_per_day=workhours_per_day,
        project_start_date=project_start_date,
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
        sprint_days = gpt_result["sprint_days"]
        number_of_sprints = gpt_result["number_of_sprints"]
        workhours_per_day = gpt_result["workhours_per_day"]
        eff_mandays = gpt_result["eff_mandays"]
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
    
    # GPT를 통해 feature의 expected_days 재조정
    # adjust_prompt = ChatPromptTemplate.from_template("""
    # 당신은 프로젝트 일정 조정 전문가입니다. 현재 스프린트의 작업량이 개발팀의 실제 작업 가능 시간보다 많습니다.
    # 각 feature의 expected_days를 조정하여 전체 작업량을 줄여야 합니다.
        
    # 현재 스프린트 정보:
    # {sprints}
        
    # 현재 Epic 정보:
    # {epics}
        
    # 현재 Feature 정보:
    # {features}
        
    # 개발팀의 실제 작업 가능 시간(eff_mandays): {eff_mandays}
    # 현재 예상 작업 시간(total_sum_of_modified_expected_days): {total_sum_of_modified_expected_days}
        
    # 다음 사항을 고려하여 각 feature의 expected_days를 조정해주세요:
    # 1. 전체 작업량이 eff_mandays 이내가 되도록 조정
    # 2. 우선순위가 높은 feature는 가능한 한 원래 예상 시간을 유지
    # 3. 우선순위가 낮은 feature의 작업 시간을 우선적으로 줄임
    # 4. 각 feature의 expected_days는 최소 0.5일 이상 유지
        
    # 다음 형식으로 응답해주세요:
    # {{
    #     "features": [
    #         {{
    #             "featureId": "feature_id",
    #             "expected_days": 조정된_예상_작업_시간
    #         }},
    #         ...
    #     ]
    # }}
    # """)
        
    # messages = adjust_prompt.format_messages(
    #     sprints=sprints,
    #     epics=epics,
    #     features=features,
    #     eff_mandays=eff_mandays,
    #     total_sum_of_modified_expected_days=total_sum_of_modified_expected_days
    # )
        
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    # response = await llm.ainvoke(messages)
        
    # try:
    #     content = response.content
    #     try:
    #         adjusted_result = extract_json_from_gpt_response(content)
    #     except Exception as e:
    #         logger.error(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True)
    #         raise Exception(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True) from e
    # except Exception as e:
    #     logger.error(f"GPT API 처리 중 오류 발생: {e}", exc_info=True)
    #     raise e
            
    # # feature의 expected_days 업데이트
    # for adjusted_feature in adjusted_result["features"]:
    #     for feature in features:
    #         if feature["featureId"] == adjusted_feature["featureId"]:
    #             logger.info(f"✅ {feature['name']}의 예상 작업시간이 {feature['expected_days']}시간에서 {adjusted_feature['expected_days']}시간으로 조정되었습니다.")
    #             feature["expected_days"] = adjusted_feature["expected_days"]
    #     total_sum_of_modified_expected_days = sum(feature["expected_days"] for feature in features)
    #     # 조정된 작업량 확인
    #     if eff_mandays < total_sum_of_modified_expected_days:
    #         logger.error(f"⚠️ 작업량 조정 후에도 eff_mandays({eff_mandays})가 total_sum_of_modified_expected_days({total_sum_of_modified_expected_days})보다 작습니다.")
    #         raise Exception(f"⚠️ 작업량 조정 후에도 eff_mandays({eff_mandays})가 total_sum_of_modified_expected_days({total_sum_of_modified_expected_days})보다 작습니다.")
    
    name_to_id = {}
    user_collection = await get_user_collection()
    for member in project_members:
        try:
            # member가 문자열인 경우를 처리
            if isinstance(member, str):
                name = member.split(", ")[0]  # "이름, 포지션" 형식에서 이름만 추출
            else:
                name = member[0]  # 리스트인 경우 첫 번째 요소가 이름
            
            user_info = await user_collection.find_one({"name": name})
            if user_info is None:
                logger.warning(f"⚠️ 사용자 정보를 찾을 수 없습니다: {name}")
                continue
                
            id = user_info["_id"]
            name_to_id[name] = id
            logger.info(f"✅ 사용자 매핑 성공 - 이름: {name}, ID: {id}")
        except Exception as e:
            logger.error(f"❌ 사용자 정보 처리 중 오류 발생: {name} - {str(e)}", exc_info=True)
            continue
    
    if not name_to_id:
        raise Exception("사용자 정보를 찾을 수 없습니다. 프로젝트 멤버 정보를 확인해주세요.")
    
    first_sprint = sprints[0]
    logger.info(f"📌 첫 번째 순서의 sprint만 추출 : {first_sprint}")
    first_sprint_epics = first_sprint["epics"]
    first_sprint_tasks = []
    for epic in first_sprint_epics:
        logger.info(f"📌 첫 번째 순서의 sprint에 포함된 epic들의 Id: {epic['epicId']}")
        for task in epic["tasks"]:
            logger.info(f"📌 첫 번째 순서의 sprint에 포함된 epic의 task들: {task['title']}")
            # assignee가 name_to_id에 없는 경우 처리
            if task["assignee"] not in name_to_id:
                logger.warning(f"⚠️ 할당된 사용자를 찾을 수 없습니다: {task['assignee']}")
                continue
            first_sprint_tasks.append(task)
    
    # API 응답 반환
    response = {
        "sprint": [
            {
                "title": first_sprint["title"],
                "description": first_sprint["description"],
                "startDate": first_sprint["startDate"],
                "endDate": first_sprint["endDate"]
            }
        ],
        "epics": [
            {
                "epicId": epic["epicId"],
                "tasks": [
                    {
                        "title": task["title"],
                        "description": task["description"],
                        "assignee": name_to_id[task["assignee"]],
                        "startDate": task["startDate"],
                        "endDate": task["endDate"],
                        "priority": task["priority"]
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
    asyncio.run(create_sprint())
    
            
# PendingTaskId 검사
#for pending_task in pending_tasks_ids:
#    if pending_task in current_epic_tasks:
#        logger.info(f"👍 {pending_task}는 이미 sprint에 포함되어 있습니다.")
#        pass
#    else:
#        logger.info(f"👎 {pending_task}가 sprint에 포함되어 있지 않습니다.")
#        pass
#    try:
#        task_to_append = task_collection.find_one({"_id": pending_task})
#        logger.info(f"추가할 pending task의 정보를 DB에서 확인하였습니다: {task_to_append}")
#   except Exception as e:
#        logger.error(f"추가할 pending task의 정보를 DB에서 확인하는 중 오류 발생: {e}", exc_info=True)
#        raise e
#    try:
#        epic_to_append = epic_collection.find_one({"_id": task_to_append["epicId"]})
#        logger.info(f"추가할 pending task의 epic 정보를 DB에서 확인하였습니다: {epic_to_append}")
#   except Exception as e:
#        logger.error(f"추가할 pending task의 epic 정보를 DB에서 확인하는 중 오류 발생: {e}", exc_info=True)
#        raise e
#    if epic_to_append["epicId"] == epic_id:
#        logger.info(f"pending task가 속한 epic이 이미 sprint에 포함되어 있습니다.")
#        break
#    logger.info(f"pending task가 속한 epic을 추가해야 합니다.")
### --------- 여기에 epic, task 추가 로직 작성해야 됨 --------- ###