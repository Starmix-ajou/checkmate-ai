import asyncio
import datetime
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mongodb_setting import get_feature_collection
from openai import AsyncOpenAI
from redis_setting import load_from_redis, save_to_redis

logger = logging.getLogger(__name__)
# 최상위 디렉토리의 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

API_ENDPOINT = "http://localhost:8000/project/specification"

project_members=[]
feature_collection = get_feature_collection()

def calculate_priority(time_assignment: Dict[str, Any], difficulty_assignment: Dict[str, Any]) -> Dict[str, Any]:
    """
    개발 예상 시간과 난이도를 기반으로 우선순위를 계산합니다.
    
    Args:
        time_assignment (Dict[str, Any]): 개발 예상 시간 데이터
        difficulty_assignment (Dict[str, Any]): 개발 난이도 데이터
        
    Returns:
        Dict[str, Any]: 우선순위가 계산된 데이터
    """
    priority_assignment = {}
    
    for feature_name in time_assignment.keys():
        # 시간과 난이도의 가중치 (시간이 더 중요하다고 가정)
        time_weight = 0.6
        difficulty_weight = 0.4
        
        # 정규화된 시간 점수 (시간이 짧을수록 점수가 높음)
        time_score = 1 - (time_assignment[feature_name]["expected_days"] / 30)  # 30일을 최대치로 가정
        
        # 정규화된 난이도 점수 (난이도가 낮을수록 점수가 높음)
        difficulty_score = 1 - ((difficulty_assignment[feature_name]["difficulty_level"] - 1) / 4)
        
        # 최종 우선순위 점수 계산
        priority_score = (time_score * time_weight) + (difficulty_score * difficulty_weight)
        
        # 1-100 범위로 변환 (점수가 높을수록 우선순위가 높음)
        priority_assignment[feature_name] = math.ceil(priority_score * 100)
    
    return priority_assignment

async def create_feature_specification(email: str) -> Dict[str, Any]:
    """
    Redis에서 프로젝트 정보를 조회하고 기능 명세서를 생성합니다.
    
    Args:
        email (str): 사용자 이메일
        
    Returns:
        Dict[str, Any]: 기능 명세서 데이터
    """
    # 변수 초기화
    stacks=[]
    project_members=[]
    
    # 프로젝트 정보 조회
    project_data = await load_from_redis(email)
    feature_data = await load_from_redis(f"feature:{email}")
    if not project_data:
        raise ValueError(f"Project for user {email} not found")

    if isinstance(project_data, str):
        project_data = json.loads(project_data)
    
    # 프로젝트 정보 추출
    projectId = project_data.get("projectId", "")
    print(f"프로젝트 아이디: {projectId}")
    for member in project_data.get("members", []):
        name = member.get("name")
        print(f"멤버 이름: {name}")
        profiles = member.get("profiles", [])
        print(f"멤버 프로필: {profiles}")
        for profile in profiles:
            if profile.get("projectId") == projectId:
                print(f"프로젝트 아이디 일치: {projectId}")
                stacks=profile.get("stacks", [])
                # positions 값이 'string'이 아닌 실제 역할(BE/FE)을 사용하도록 수정
                position = profile.get("positions", [])[0] if profile.get("positions") else ""
                if position == "string":
                    # 이전 프로필에서 BE/FE 값을 찾아서 사용
                    for prev_profile in profiles:
                        if prev_profile.get("projectId") != projectId and prev_profile.get("positions"):
                            prev_position = prev_profile.get("positions")[0]
                            if prev_position in ["Backend", "Frontend"]:
                                position = prev_position
                                break
                member_info = [
                    name, 
                    position,
                    ", ".join(profile.get("stacks", []))
                ]
                project_members.append(", ".join(str(item) for item in member_info))
    features = feature_data.get("features", [])
    print(f"프로젝트 멤버: {project_members}")
    
    # 필수 데이터 검증
    if not stacks:
        raise ValueError("프로젝트 기술 스택 정보가 없습니다.")
    if not project_members:
        raise ValueError("프로젝트 멤버 정보가 없습니다.")
    if not features:
        raise ValueError("프로젝트 기능 목록이 없습니다.")
    
    print("\n=== 프로젝트 정보 ===")
    print("스택:", stacks)
    print("멤버:", project_members)
    print("기능 목록:", features)
    print("=== 프로젝트 정보 끝 ===\n")
    
    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_template("""
    당신은 소프트웨어 기능 목록을 분석하여 기능 명세서를 작성하는 일을 도와주는 엔지니어입니다.
    다음 기능 정의서와 프로젝트 스택 정보, 프로젝트에 참여하는 멤버 정보를 분석하여 
    각 기능별로 상세 명세를 작성하고, 필요한 정보를 지정해주세요.
    절대 주석을 추가하지 마세요. 당신은 한글이 주언어입니다.
    
    프로젝트 개발에 사용되는 스택:
    {stacks}
    
    프로젝트 멤버별 [이름, 역할, 스택]를 융합한 리스트:
    {project_members}
    
    기능 정의서:
    {features}
    
    주의사항:
    1. 위 기능 정의서에 나열된 모든 기능에 대해 상세 명세를 작성해주세요.
    2. 새로운 기능을 추가하거나 기존 기능을 제외하지 마세요.
    3. 각 기능의 이름은 기능 정의서와 동일하게 사용하고 절대 임의로 바꾸지 마세요.
    4. 담당자 할당 시 각 멤버의 역할(BE/FE)을 고려해주세요.
    5. 각 기능의 고유 ID는 기능 정의서에 나열된 순서대로 부여해주세요. 형식은 "id_001", "id_002", "id_003", ...로 통일합니다.
    각 기능에 대해 다음 항목들을 JSON 형식으로 응답해주세요:
    {{{{
        "feature_name": {{{{
            "specification": {{{{
                "useCase": "기능의 사용 사례 설명",
                "input": "기능에 필요한 입력 데이터",
                "output": "기능의 출력 결과",
                "precondition": "기능 실행 전 만족해야 할 조건",
                "postcondition": "기능 실행 후 보장되는 조건"
            }}}},
            "stack": {{{{
                "required_stacks": ["필수 스택1", "필수 스택2", ...],
                "optional_stacks": ["선택 스택1", "선택 스택2", ...]
            }}}},
            "time": {{{{
                "expected_days": 정수
            }}}},
            "difficulty": {{{{
                "difficulty_level": 1-5
            }}}}
        }}}},
        ...
    }}}}
    """)
    
    # 프롬프트에 데이터 전달
    message = prompt.format_messages(
        stacks="\n".join(stacks),
        project_members="\n".join(project_members),
        features="\n".join(features)
    )
    
    # LLM 호출
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    response = await llm.ainvoke(message)
    
    # 응답 파싱
    try:
        content = response.content
        logger.info("\n=== GPT 원본 응답 ===")
        logger.info(content)
        logger.info("=== 응답 끝 ===\n")
        
        # JSON 블록 추출
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        else:
            # JSON 블록이 없는 경우 전체 내용에서 첫 번째 { 부터 마지막 } 까지 추출
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != 0:
                content = content[start:end]
        
        # 줄바꿈과 불필요한 공백 제거
        content = content.replace("\n", "").replace("  ", " ").strip()
        
        # 주석 제거 (# 이후의 텍스트 제거)
        content_lines = content.split("#")
        content = content_lines[0]
        for i in range(1, len(content_lines)):
            # #이 JSON 문자열 안에 있을 수 있으므로, 다음 { 또는 , 가 나오는 부분부터 다시 포함
            next_part = content_lines[i]
            json_continue = next_part.find("{")
            if json_continue == -1:
                json_continue = next_part.find(",")
            if json_continue != -1:
                content += next_part[json_continue:]
        
        logger.info("\n=== 정리된 JSON 문자열 ===")
        logger.info(content)
        logger.info("=== JSON 문자열 끝 ===\n")
        
        # JSON 파싱 시도
        try:
            result = json.loads(content)
            logger.info("\n=== 파싱된 결과 ===")
            logger.info(json.dumps(result, indent=2, ensure_ascii=False))
            logger.info("=== 결과 끝 ===\n")
        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시 문자열 내용 분석
            logger.error("\n=== JSON 파싱 실패 분석 ===")
            logger.error(f"파싱 실패 위치: {e.pos}")
            logger.error(f"문제의 문자: {content[e.pos-10:e.pos+10]}")  # 문제 지점 주변 문자열 출력
            logger.error(f"전체 에러: {str(e)}")
            logger.error("=== 분석 끝 ===\n")
            raise Exception(f"JSON 파싱 실패: {str(e)}") from e
        
        # 우선순위 계산
        time_assignment = {name: data["time"] for name, data in result.items()}
        difficulty_assignment = {name: data["difficulty"] for name, data in result.items()}
        priority_assignment = calculate_priority(time_assignment, difficulty_assignment)
        
        features_to_store = []
        for feature_name, data in result.items():
            feature = {
                "name": feature_name,
                "useCase": data["specification"]["useCase"],
                "input": data["specification"]["input"],
                "output": data["specification"]["output"],
                "precondition": data["specification"]["precondition"],
                "postcondition": data["specification"]["postcondition"],
                "stack": data["stack"]["required_stacks"],
                "priority": priority_assignment[feature_name],
                "relfeatIds": [],
                "embedding": [],
            }
            features_to_store.append(feature)
            
        # 기존 프로젝트 데이터에 features 추가
        feature_data = features_to_store
            
        # Redis에 저장
        try:
            await save_to_redis(f"feature:{email}", feature_data)
        except Exception as e:
            logger.error(f"feature_specification 초안 Redis 저장 실패: {str(e)}")
            raise e
        
        # API 응답 반환
        result = {
            "features": [
                {
                    "name": feature_name,
                    "useCase": data["specification"]["useCase"],
                    "input": data["specification"]["input"],
                    "output": data["specification"]["output"]
                }
                for feature_name, data in result.items()
            ]
        }
        return result
    except Exception as e:
        logger.error(f"GPT API 응답 처리 중 오류 발생: {str(e)}")
        raise Exception(f"GPT API 응답 처리 중 오류 발생: {str(e)}") from e



async def update_feature_specification(email: str, feedback: str) -> Dict[str, Any]:
    """
    사용자 피드백을 기반으로 기능 명세서를 업데이트합니다.
    
    Args:
        email (str): 사용자 이메일
        feedback (str): 사용자 피드백
        
    Returns:
        Dict[str, Any]: 업데이트된 기능 명세서 데이터
            - features: 업데이트된 기능 목록
            - isNextStep: 다음 단계 진행 여부 (0: 종료, 1: 계속)
    """
    
    raw_feature_specification = await load_from_redis(f"feature:{email}")
    project_data = await load_from_redis(email)
    if not raw_feature_specification:
        raise ValueError(f"Feature specification for user {email} not found")
    
    # Redis에서 가져온 데이터가 문자열인 경우 JSON 파싱
    if isinstance(raw_feature_specification, str):
        raw_feature_specification = json.loads(raw_feature_specification)
    current_features = raw_feature_specification
    
    # 피드백 분석 및 기능 업데이트
    update_prompt = ChatPromptTemplate.from_template("""
    당신은 사용자의 피드백을 분석하고 기능 명세서를 업데이트하는 전문가입니다.
    반드시 아래 형식의 JSON으로만 응답해주세요. 추가 설명이나 주석은 절대 포함하지 마세요.

    현재 기능 목록:
    {current_features}
    
    프로젝트 정보:
    {project_data}

    다음은 기능 명세 단계에서 받은 사용자의 피드백입니다:
    {feedback}
    이 피드백이 다음 중 어떤 유형인지 판단해주세요:

    1. 수정/삭제 요청:
       예시: "담당자를 다른 사람으로 변경해 주세요", "~기능 개발 우선순위를 낮추세요", "~기능을 삭제해주세요"

    2. 종료 요청:
       예시: "이대로 좋습니다", "더 이상 수정할 필요 없어요", "다음으로 넘어가죠"

    다음 형식으로 응답해주세요:
    {{
        "isNextStep": 0 또는 1,  # 0: 수정/삭제 요청, 1: 종료 요청
        "features": [
            {{
                "name": "기능명",
                "useCase": "사용 사례",
                "input": "입력 데이터",
                "output": "출력 결과",
                "precondition": "기능 실행 전 만족해야 할 조건",
                "postcondition": "기능 실행 후 보장되는 조건",
                "stack": {{
                    "required_stacks": ["필수 스택1", "필수 스택2"],
                    "optional_stacks": ["선택 스택1", "선택 스택2"]
                }},
                "time": {{
                    "expected_days": 정수
                }},
                "difficulty": {{
                    "difficulty_level": 정수
                }},
                "priority": 정수
            }}
        ]
    }}

    주의사항:
    0. 반드시 모든 내용을 한국어로 작성해주세요. 만약 한국어로 대체하기 어려운 단어가 있다면 영어를 사용해 주세요.
    1. 반드시 위 JSON 형식을 정확하게 따라주세요.
    2. 모든 문자열은 쌍따옴표로 감싸주세요.
    3. 객체의 마지막 항목에는 쉼표를 넣지 마세요.
    4. 수정된 기능만 포함하고, 수정되지 않은 기능은 제외해주세요.
    5. isNextStep은 사용자의 피드백이 종료 요청인 경우 1, 수정/삭제 요청인 경우 0으로 설정해주세요.
    6. 각 기능의 모든 필드를 포함해주세요.
    7. difficulty_level은 1에서 5 사이의 정수여야 합니다.
    8. expected_days는 양의 정수여야 합니다.
    9. 절대 주석을 추가하지 마세요.
    """)
    
    messages = update_prompt.format_messages(
        current_features=str(current_features),
        project_data=str(project_data),
        feedback=feedback
    )
    
    # LLM Config
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3
    )
    response = await llm.ainvoke(messages)
    
    # 응답 파싱
    try:
        content = response.content
        logger.info("\n=== GPT 원본 응답 ===")
        logger.info(content)
        logger.info("=== 응답 끝 ===\n")
        
        # JSON 블록 추출 전 content 정리
        content = content.strip()
        
        # JSON 블록 추출
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # 줄바꿈과 불필요한 공백 제거
        content = content.replace("\n", " ").replace("\r", " ")
        while "  " in content:
            content = content.replace("  ", " ")
        content = content.strip()
        
        # 주석 제거
        content_parts = []
        in_string = False
        comment_start = -1
        
        for i, char in enumerate(content):
            if char == '"' and (i == 0 or content[i-1] != '\\'):
                in_string = not in_string
            elif char == '#' and not in_string:
                if comment_start == -1:
                    comment_start = i
            elif char in '{[,' and comment_start != -1:
                content_parts.append(content[comment_start:i].strip())
                comment_start = -1
        
        if comment_start != -1:
            content_parts.append(content[comment_start:].strip())
        
        for part in content_parts:
            content = content.replace(part, '')
        
        logger.info("\n=== 정리된 JSON 문자열 ===")
        logger.info(content)
        logger.info("=== JSON 문자열 끝 ===\n")
        
        # JSON 파싱
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("\n=== JSON 파싱 실패 분석 ===")
            logger.error(f"파싱 실패 위치: {e.pos}")
            logger.error(f"문제의 문자: {content[max(0, e.pos-20):min(len(content), e.pos+20)]}")
            logger.error(f"전체 에러: {str(e)}")
            logger.error("=== 분석 끝 ===\n")
            raise

        # 응답 검증
        if not isinstance(result, dict):
            raise ValueError("응답이 객체 형식이 아닙니다.")
        
        if "isNextStep" not in result:
            raise ValueError("isNextStep 필드가 누락되었습니다.")
        
        if not isinstance(result["isNextStep"], int) or result["isNextStep"] not in [0, 1]:
            raise ValueError("isNextStep은 0 또는 1이어야 합니다.")
        
        if "features" not in result:
            raise ValueError("features 필드가 누락되었습니다.")
        
        if not isinstance(result["features"], list):
            raise ValueError("features는 배열이어야 합니다.")
        
        # 각 기능 검증
        for feature in result["features"]:
            required_fields = [
                "name", "useCase", "input", "output", "precondition", "postcondition",
                "stack", "time", "difficulty"
            ]
            for field in required_fields:
                if field not in feature:
                    raise ValueError(f"기능 '{feature.get('name', 'unknown')}'에 '{field}' 필드가 누락되었습니다.")
            
            if not isinstance(feature["stack"], dict) or "required_stacks" not in feature["stack"]:
                raise ValueError(f"기능 '{feature['name']}'의 stack 형식이 잘못되었습니다.")
            
            if not isinstance(feature["time"], dict) or "expected_days" not in feature["time"]:
                raise ValueError(f"기능 '{feature['name']}'의 time 형식이 잘못되었습니다.")
            
            if not isinstance(feature["difficulty"], dict) or "difficulty_level" not in feature["difficulty"]:
                raise ValueError(f"기능 '{feature['name']}'의 difficulty 형식이 잘못되었습니다.")
            
            if not isinstance(feature["time"]["expected_days"], int) or feature["time"]["expected_days"] <= 0:
                raise ValueError(f"기능 '{feature['name']}'의 expected_days는 양의 정수여야 합니다.")
            
            if not isinstance(feature["difficulty"]["difficulty_level"], int) or \
               not 1 <= feature["difficulty"]["difficulty_level"] <= 5:
                raise ValueError(f"기능 '{feature['name']}'의 difficulty_level은 1에서 5 사이의 정수여야 합니다.")

        logger.info("\n=== 검증된 결과 ===")
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("=== 결과 끝 ===\n")
        
    except Exception as e:
        logger.error(f"GPT API 응답 처리 중 오류 발생: {str(e)}")
        raise Exception(f"GPT API 응답 처리 중 오류 발생: {str(e)}") from e
    
    # 업데이트된 기능 정보를 기존 기능 리스트와 융합
    updated_map = {feature["name"]: feature for feature in result["features"]}
    merged_features = []
    
    # 기존 기능 리스트 순회
    for current_feature in current_features:
        feature_name = current_feature["name"]
        if feature_name in updated_map:
            # 업데이트된 기능이 있는 경우
            updated = updated_map[feature_name]
            merged_feature = current_feature.copy()
            
            # time이나 difficulty가 변경되었는지 확인
            time_changed = updated.get("time") != current_feature.get("time")
            difficulty_changed = updated.get("difficulty") != current_feature.get("difficulty")
            
            merged_feature.update({
                "useCase": updated.get("useCase", current_feature.get("useCase")),
                "input": updated.get("input", current_feature.get("input")),
                "output": updated.get("output", current_feature.get("output")),
                "precondition": updated.get("precondition", current_feature.get("precondition")),
                "postcondition": updated.get("postcondition", current_feature.get("postcondition")),
                "stack": updated.get("stack", {}).get("required_stacks", current_feature.get("stack")),
                "time": updated.get("time", current_feature.get("time")),
                "difficulty": updated.get("difficulty", current_feature.get("difficulty"))
            })
            
            # priority 처리
            if "priority" in updated:
                # GPT가 직접 priority를 지정한 경우
                merged_feature["priority"] = updated["priority"]
            elif time_changed or difficulty_changed:
                # time이나 difficulty가 변경된 경우 우선순위 재계산
                time_map = {feature_name: merged_feature["time"]}
                difficulty_map = {feature_name: merged_feature["difficulty"]}
                merged_feature["priority"] = calculate_priority(time_map, difficulty_map)[feature_name]
            else:
                # 변경사항이 없는 경우 기존 priority 유지
                merged_feature["priority"] = current_feature.get("priority")
            
            merged_features.append(merged_feature)
        else:
            # 업데이트되지 않은 기능은 그대로 유지
            merged_features.append(current_feature)
    
    # 업데이트된 기능 목록으로 교체
    logger.info("\n=== 업데이트된 feature_specification 데이터 ===")
    logger.info(json.dumps(merged_features, indent=2, ensure_ascii=False))
    logger.info("=== 데이터 끝 ===\n")
    
    # Redis에 저장
    try:
        await save_to_redis(f"feature:{email}", merged_features)
    except Exception as e:
        logger.error(f"업데이트된 feature_specification Redis 저장 실패: {str(e)}")
        raise e
    
    # 다음 단게로 넘어가는 경우, MongoDB에 Redis의 데이터를 옮겨서 저장
    if result["isNextStep"] == 1:
        try:
            feature_collection = await get_feature_collection()
            for feat in merged_features:
                feature_data = {
                    "projectId": project_data["projectId"],
                    "feature": feat,
                    "createdAt": datetime.datetime.utcnow()
                }
                try:
                    result = await feature_collection.insert_one(feature_data)
                    feat["featureId"] = str(result.inserted_id)
                    logger.info(f"{feat['name']} MongoDB 저장 성공 (ID: {feat['featureId']})")
                except Exception as e:
                    logger.error(f"{feat['name']} MongoDB 저장 실패: {str(e)}")
                    raise e
            logger.info("모든 feature MongoDB 저장 완료")
        except Exception as e:
            logger.error(f"feature_specification MongoDB 저장 실패: {str(e)}")
            raise e
    
    # API 응답 반환
    return {
        "features": [
            {
                "name": feature["name"],
                "useCase": feature["useCase"],
                "input": feature["input"],
                "output": feature["output"]
            }
            for feature in result["features"]
        ],
        "isNextStep": result["isNextStep"]
    }
