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
                             get_project_collection)
from openai import AsyncOpenAI
from redis_setting import load_from_redis, save_to_redis

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def init_collections():
    global feature_collection, project_collection, epic_collection
    feature_collection = None
    project_collection = None
    epic_collection = None
    feature_collection = await get_feature_collection()
    project_collection = await get_project_collection()
    epic_collection = await get_epic_collection()

async def create_epic(project_id: str) -> int:
    """
    DB에서 프로젝트 명세 정보를 조회하여 각 기능을 하나의 task로 변환하고, 이를 묶어서 epic을 정의합니다.
    
    Args:
        project_id (str): 개발 프로젝트의 ID (DB 조회 목적)
        pending_tasks_ids (List[str]): 이번 스프린트에서 제외되는 기능들의 ID 목록 (GPT API 호출의 입력에서 제외)
        
    Returns:
        Dict[str, Any]: epic 정의 정보
    """
    try:
        features = await feature_collection.find({"projectId": project_id}).to_list(length=None)
    except Exception as e:
        logger.error(f"MongoDB에서 Features 정보 로드 중 오류 발생: {e}", exc_info=True)
        raise e
    print(f"features로부터 epic 생성을 시작합니다.\nfeatures: {features}")
    
    epic_prompt = ChatPromptTemplate.from_template("""
    당신은 애자일 마스터입니다. 당신의 주요 언어는 한국어입니다. 당신의 업무는 비슷한 task들을 묶어서 epic을 정의하는 것입니다.
    이때 지켜야 하는 규칙이 있습니다. 
    1. 각 epic은 반드시 하나 이상의 task를 포함해야 합니다.
    2. epic의 이름을 자연어로 정의해 주세요. 이름은 epic이 포함하는 task들의 성격을 반영해야 합니다.
    3. 비기능과 관련된 task가 존재할 경우 비기능과 관련된 task를 묶어서 "nonFunctional" epic으로 정의해 주세요.
    4. 당신에게 주어지는 feature는 task와 1:1로 대응됩니다. 즉, features의 수만큼 tasks가 존재해야 합니다.
    5. 기능 Id, 기능 이름, 담당자 등 기능과 관련된 내용을 절대로 수정하거나 삭제하지 마세요.
    6. 모든 task는 소속된 epic이 존재해야 하고, 두 개 이상의 epic에 소속될 수 없습니다. 중복되는 task가 존재할 경우 더 적합한 epic을 평가한 후 소속 epic을 하나로 결정해 주세요.
    7. startDate와 endDate는 문자열(YYYY-MM-DD) 형식으로 반환하고, epic의 날짜들은 각 epic이 포함하는 task의 날짜들을 사용하여 정의해야 합니다.
    
    결과를 다음과 같은 형식으로 반환해 주세요.
    {{{{
        "number_of_epics": 정수
        "epics": [
            {{
                "epic_title": "epic의 이름",
                "epic_description": "epic에 대한 간략한 설명",
                "featureIds": ["id_013", "id_002", "id_010"],
                "epic_startDate": 문자열(YYYY-MM-DD). epic의 시작 날짜이며 포함하는 task 중에 가장 startDate가 빠른 task의 startDate와 같아야 합니다.
                "epic_endDate": 문자열(YYYY-MM-DD). epic의 종료 날짜이며 포함하는 task 중에 가장 endDate가 늦은 task의 endDate와 같아야 합니다.
            }},
            ...
        ]
    }}}}
    
    현재 기능 정보:
    {features}
    """)
    
    messages = epic_prompt.format_messages(
        features=features
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
    
    epic_to_store = []
    epics = gpt_result["epics"]
    logger.info("⚙️ gpt가 반환한 결과로부터 epic 정보를 추출합니다.")
    for epic in epics:
        epic_title = epic["epic_title"]
        epic_description = epic["epic_description"]
        feature_ids = epic["featureIds"]
        epic_startDate = epic["epic_startDate"]
        epic_endDate = epic["epic_endDate"]
        
        print(f"Epic Title: {epic_title}")
        print(f"Epic Description: {epic_description}")
        print(f"Feature Ids: {feature_ids}")
        print(f"Epic Start Date: {epic_startDate}")
        print(f"Epic End Date: {epic_endDate}")
        
        epic_data = {
            "epicTitle": epic_title,
            "epicDescription": epic_description,
            "featureIds": feature_ids,
            "epicStartDate": epic_startDate,
            "epicEndDate": epic_endDate
        }
        epic_to_store.append(epic_data)
    
    # Redis에 저장할 때 gpt_result 형식 유지
    redis_data = {
        "number_of_epics": gpt_result["number_of_epics"],
        "epics": epic_to_store
    }
    
    try:
        await save_to_redis(f"epic:{project_id}", redis_data)
    except Exception as e:
        logger.error(f"Redis에 Epic 데이터 저장 중 오류 발생: {e}", exc_info=True)
        raise e
    return redis_data


async def calculate_eff_mandays(efficiency_factor: float, number_of_developers: int, sprint_days: int, workhours_per_day: int) -> float:

    logger.info(f"개발자 수: {number_of_developers}명, 1일 개발 업무시간: {workhours_per_day}시간, 스프린트 주기: {sprint_days}일, 효율성 계수: {efficiency_factor}")
    mandays = number_of_developers * sprint_days * workhours_per_day
    logger.info(f"⚙️ Sprint별 작업 배정 시간: {mandays}시간")
    eff_mandays = mandays * efficiency_factor
    logger.info(f"⚙️ Sprint별 효율적인 작업 배정 시간: {eff_mandays}시간")
    
    return eff_mandays

### ======== Create Sprint ======== ###
async def create_sprint(project_id: str, pending_tasks_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    logger.info(f"🔍 스프린트 생성 시작: {project_id}")
    await init_collections()
    
    if pending_tasks_ids is not None and len(pending_tasks_ids) > 0:
        logger.info(f"🔍 이번 스프린트에서 제외되는 기능들: {pending_tasks_ids}")
        ### Pending Tasks를 features 목록 내에서 열외
    
    logger.info(f"🔍 이번 스프린트에서 제외되는 기능 없음")
    try:
        features = await feature_collection.find({"projectId": project_id}).to_list(length=None)
        logger.info("✅ Mongodb에서 projectId와 일치하는 기능들 조회 완료")
    except Exception as e:
        logger.error(f"MongoDB에서 Features 정보 로드 중 오류 발생: {e}", exc_info=True)
        raise e
    
    try:
        epics = await load_from_redis(f"epic:{project_id}")
        logger.info("✅ Redis에서 Epic 정보 로드 완료")
        # Redis에서 불러온 데이터가 리스트인 경우 gpt_result 형식으로 변환
        if isinstance(epics, list):
            epics = {
                "number_of_epics": len(epics),
                "epics": epics
            }
    except Exception as e:
        logger.error(f"Redis로부터 Epic 정보 로드 중 오류 발생: {e}", exc_info=True)
        raise e
    if not epics:   # epic 정보가 redis에 없는 경우
        # epic부터 정의합니다.
        logger.info("Epic이 아직 정의되지 않은 프로젝트입니다. Epic을 정의합니다.")
        epics = await create_epic(project_id)
        logger.info("✅ Epic 정의 완료")
        logger.info(f"새롭게 정의된 Epic의 수: {epics['number_of_epics']}")
    
    # 이제 Epic이 정의되어 있으므로 Sprint의 뼈대를 구성합니다.
    logger.info(f"이제 정의된 Epic을 기반으로 Sprint의 뼈대를 구성합니다. 정의되어 있는 Epic의 수: {epics['number_of_epics']}")
    
    # Epic 별 누적 우선순위 값 계산
    epics = epics["epics"]
    try:
        for epic in epics:
            priority_sum = 0
            target_features = epic["featureIds"]
            for feature in features:
                if feature["featureId"] in target_features:
                    logger.info(f"{feature['name']}가 {epic['epicTitle']}에 속합니다.")
                    priority_sum += feature["priority"]
            epic["prioritySum"] = priority_sum
            logger.info(f"{epic['epicTitle']}의 누적 우선순위 값: {priority_sum}")
    except Exception as e:
        logger.error(f"Epic 별 누적 우선순위 값 계산 중 오류 발생: {e}", exc_info=True)
        raise e
    
    # 누적 우선순위 값이 높은 순서대로 정렬
    try: 
        epics.sort(key=lambda x: x["prioritySum"], reverse=True)
        logger.info(f"우선순위가 높은 순서대로 정렬된 Epic 정보: {epics}")
    except Exception as e:
        logger.error(f"Epic 우선순위 오름차순 정렬 중 오류 발생: {e}", exc_info=True)
        raise e
    
    # 정렬된 Epic 정보를 Redis에 저장
    await save_to_redis(f"epic:{project_id}", epics)    # 이제 Epic들은 Redis에 누적 우선순위가 높은 순서대로 정렬되어 있음.
    logger.info(f"✅ Redis에 정렬된 epic 정보를 저장하였습니다.")

    #features = await load_from_redis(f"features:{project_id}")
    epics = await load_from_redis(f"epic:{project_id}")
    # 적절한 Sprint 주기 찾기
    # 사용하는 정보: 전체 프로젝트 기간, 각 task별 기간과 우선순위, 각 Epic별 누적 우선순위
    sprint_prompt = ChatPromptTemplate.from_template("""
    당신은 애자일 마스터입니다. 당신의 업무는 주어지는 Epic과 Epic별 Task의 정보를 바탕으로 적절한 Sprint Backlog를 생성하는 것입니다.
    당신의 주요 언어는 한국어입니다.
    다음은 주의사항입니다.
    1. Sprint의 startDate는 스프린트가 포함하는 모든 Epic들 중에 가장 시작 날짜가 빠른 Epic의 startDate 값과 같아야 합니다.
    2. Sprint의 endDate는 스프린트가 포함하는 모든 Epic들 중에 가장 종료 날짜가 늦은 Epic의 endDate 값과 같아야 합니다.
    3. 첫 번째 의 startDate는 프로젝트의 startDate보다 빠를 수 없습니다.
    4. 마지막 스프린트의 endDate는 프로젝트의 endDate보다 늦을 수 없습니다.
    5. 우선순위가 높은 에픽과 기능들이 앞 순서의 스프린트에 포함되도록 에픽을 배치하세요. 이를 위해 epic별 prioritySum과 task별 Priority를 참고하세요.
    6. 스프린트에 되도록 Epic의 모든 task가 포함되도록 하세요. 스프린트 배치의 기본 단위는 Task가 아닌 Epic이어야 합니다.
    
    결과를 다음과 같은 형식으로 반환해 주세요.
    {{{{
        "number_of_sprints": 정수. 프로젝트 기간에 포함되는 전체 스프린트의 개수
        "sprint_days": 정수. 하나의 스프린트가 진행되는 기간(일)
        "sprints": [
            {{
                "sprint_number": int,
                "sprint_title": "string",
                "sprint_description": "string",
                "sprint_startDate": str(YYYY-MM-DD),
                "sprint_endDate": str(YYYY-MM-DD),
                "epic_titles": ["epic_title_01", "epic_title_02", "epic_title_03"]
            }},
            ...
        ]
    }}}}
    
    현재 기능 정보:
    {features}
    
    현재 Epic 정보:
    {epics}
    """)
    
    messages = sprint_prompt.format_messages(
        features=features,
        epics=epics
    )
    
    # LLM Config
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,
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
        sprint_totalnum = gpt_result["number_of_sprints"]
    except Exception as e:
        logger.error("gpt_result로부터 field를 추출할 수 없음", exc_info=True)
        raise e
    logger.info(f"⚙️ sprint 한 주기: {sprint_days}일")
    logger.info(f"⚙️ 생성된 총 스프린트의 개수: {sprint_totalnum}개")
    
    sprints = gpt_result["sprints"]
    #sprint_to_store = []
    for sprint in sprints:
        sprint_number = sprint["sprint_number"]
        sprint_startDate = sprint["sprint_startDate"]
        sprint_endDate = sprint["sprint_endDate"]
        epic_titles = sprint["epic_titles"]
        logger.info(f"⚙️ Sprint {sprint_number}의 시작일: {sprint_startDate}, 종료일: {sprint_endDate}, 포함된 Epic: {epic_titles}")
        #sprint_data = {
        #    "sprintNumber": sprint_number,
        #    "sprintStartDate": sprint_startDate,
        #    "sprintEndDate": sprint_endDate,
        #    "epicTitles": epic_titles
        #}
    #try:
    #    await save_to_redis(f"sprint:{project_id}", sprint_to_store)
    #except Exception as e:
    #    logger.error(f"Redis에 Sprint 데이터 저장 중 오류 발생: {e}", exc_info=True)
    #    raise e
    
    try:
        project = await project_collection.find_one({"_id": project_id})
        logger.info("✅ 효율적인 작업일수 계산을 위해 프로젝트 정보를 조회합니다.")
    except Exception as e:
        logger.error(f"MongoDB에서 Project 정보 로드 중 오류 발생: {e}", exc_info=True)
        raise e
    
    #### sprint duration에 따른 expected_workday 수정 (일->시간)
    try:
        logger.info(f"프로젝트 시작일: {project['startDate']}, 프로젝트 종료일: {project['endDate']}")
        project_start_date = project["startDate"] if isinstance(project["startDate"], datetime) else datetime.strptime(project["startDate"], "%Y-%m-%d")
        project_end_date = project["endDate"] if isinstance(project["endDate"], datetime) else datetime.strptime(project["endDate"], "%Y-%m-%d")
        project_days = (project_end_date - project_start_date).days
    except Exception as e:
        logger.error(f"프로젝트 기간 계산 중 오류 발생: {e}", exc_info=True)
        raise e
    
    if project_days <= 90:
        logger.info("프로젝트 기간이 90일 이하입니다. 주 5일 근무, 1일 8시간 개발, 총 주차별 40시간 작업으로 계산합니다.")
        workhours_per_day = 8
    elif project_days <= 180 and project_days > 90:
        logger.info("프로젝트 기간이 180일 이하입니다. 주 5일 근무, 1일 6시간 개발, 총 주차별 30시간 작업으로 계산합니다.")
        workhours_per_day = 6
    elif project_days <= 270 and project_days > 180:
        logger.info("프로젝트 기간이 270일 이하입니다. 주 5일 근무, 1일 4시간 개발, 총 주차별 20시간 작업으로 계산합니다.")
        workhours_per_day = 4
    elif project_days <= 365 and project_days > 270:
        logger.info("프로젝트 기간이 365일 이하입니다. 주 5일 근무, 1일 2시간 개발, 총 주차별 10시간 작업으로 계산합니다.")
        workhours_per_day = 2
    else:
        logger.info("프로젝트 기간이 365일 초과입니다. 주 5일 근무, 1일 1시간 개발, 총 주차별 5시간 작업으로 계산합니다.")
        workhours_per_day = 1
    
    ### Sprint Capacity 계산
    efficiency_factor = 0.6
    number_of_developers = len(project["memberIds"])
    eff_mandays = await calculate_eff_mandays(efficiency_factor, number_of_developers, sprint_days, workhours_per_day)
    
    ### workhours_per_day를 기준으로 각 task의 예상 작업 시간 재조정
    #features = await load_from_redis(f"features:{project_id}")
    #epics = await load_from_redis(f"epic:{project_id}")
    
    print(f"현재 features: {features}")
    print(f"현재 epics: {epics}")
    
    for feature in features:
        feature["expected_days"] *= 0.5 * (workhours_per_day/2)
        logger.info(f"✅ {feature['name']}의 재조정된 예상 작업시간: {feature['expected_days']}")
    
    ### 최종적으로 구성된 eff_mandays 내부에 sprint별로 포함된 task들의 '재조정된 기능별 예상 작업시간'의 총합이 들어오는지 확인
    modified_feat_expected_days = 0
    for sprint in sprints:
        epic_titles = sprint["epic_titles"]
        for epic in epics:
            if epic["epicTitle"] in epic_titles:
                for feature in features:
                    if feature["featureId"] in epic["featureIds"]:
                        logger.info(f"✅ {feature['name']}이 {epic['epicTitle']}에 속합니다.")
                        modified_feat_expected_days += feature["expected_days"]
    if eff_mandays < modified_feat_expected_days:
        logger.error(f"⚠️ 최종적으로 구성된 eff_mandays 내부에 sprint별로 포함된 task들의 '재조정된 기능별 예상 작업시간'의 총합이 들어오지 않습니다. eff_mandays: {eff_mandays}, modified_feat_expected_days: {modified_feat_expected_days}")
        raise Exception(f"⚠️ 최종적으로 구성된 eff_mandays 내부에 sprint별로 포함된 task들의 '재조정된 기능별 예상 작업시간'의 총합이 들어오지 않습니다. eff_mandays: {eff_mandays}, modified_feat_expected_days: {modified_feat_expected_days}")
    
    # API 응답 반환
    response = [
        {
            "sprint": [
                {
                    "title": sprint["sprint_title"],
                    "description": sprint["sprint_description"],
                    "startDate": sprint["sprint_startDate"],
                    "endDate": sprint["sprint_endDate"]
                }
                for sprint in sprints
            ],
            "epics": [
                {
                    "title": epic["epicTitle"],
                    "description": epic["epicDescription"],
                    "featureIds": epic["featureIds"]
                }
                for epic in epics
            ],
        }
    ]
    logger.info(f"👉 API 응답 결과: {response}")
    return response
    
if __name__ == "__main__":
    asyncio.run(create_sprint())