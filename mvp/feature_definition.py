import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import httpx
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from redis_setting import load_from_redis, save_to_redis

logger = logging.getLogger(__name__)

# 최상위 디렉토리의 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

API_ENDPOINT = "http://localhost:8000/project/definition"


async def create_feature_definition(email: str, description: str, definition_url: Optional[str] = None) -> Dict[str, Any]:
    """
    기능 정의서를 생성합니다.
    
    Args:
        email (str): 사용자 이메일
        description (str): 기능 정의서 텍스트
        definition_url (Optional[str]): 기능 정의서 URL
        
    Returns:
        Dict[str, Any]: 생성된 기능 목록
    """
    try:
        project_data = {
            "email": email,
            "description": description,
            "definitionUrl": definition_url
        }
        user_input = description  # user_input 변수 초기화
    except Exception as e:
        logger.error(f"프로젝트 데이터 처리 중 오류 발생: {str(e)}")
        raise Exception(f"프로젝트 데이터 처리 중 오류 발생: {str(e)}") from e
    
    
    # 사전 정의된 기능 정의서 존재 여부 확인
    if(project_data.get("definitionUrl")):
        logger.info("기능 정의서가 이미 존재합니다.")
        predefined_definition = project_data.get("definitionUrl")
        
        # GPT API 호출을 위한 프롬프트 정의
        create_feature_prompt = """
        당신의 역할은 주니어 개발팀의 입장에서 개발하려는 서비스에 필요할 것으로 예상되는 기능 목록을 정의하는 것입니다. 
        각 기능은 구현 가능한 작은 단위여야 하고, 반드시 중복되지 않아야 합니다.
        다음은 개발팀이 사전에 정의한 정의서가 존재하는 링크입니다. 링크: {predefined_definition}
        링크의 pdf 파일을 분석해서 이미 정의되어 있는 기능 목록들을 다음의 형식으로 반환해 주세요.
        이미 정의되어 있는 기능 목록들은 반드시 포함해야 합니다.
        {{
            "features": 
            [
                "로그인 기능",
                "회원가입 기능",
                "게시판 기능"
            ]
        }}
        
        추가로 다음의 형식으로 추가하면 좋을 것으로 예상되는 기능 목록을 제안해 주세요.
        {{
            "suggestions": [
                {{
                    "question": "이런 기능을 추가하시는 건 어떤가요?",
                    "answers": ["결제 기능", "주문 기능", "주문 조회 기능"]
                }}
            ]
        }}
        
        정보:
        {user_input}
        """
        
        # GPT API 호출
        completion = await openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 소프트웨어 요구사항 분석가입니다."
                },
                {
                    "role": "user",
                    "content": create_feature_prompt.format(user_input=user_input, predefined_definition=predefined_definition)
                }
            ]
        )
    else:
        logger.info("기능 정의서가 존재하지 않습니다.")
        
        # GPT API 호출을 위한 프롬프트 정의
        create_feature_prompt = """
        당신의 역할은 주니어 개발팀의 입장에서 개발하려는 서비스에 필요할 것으로 예상되는 기능 목록을 정의하는 것입니다. 
        각 기능은 구현 가능한 작은 단위여야 하고, 반드시 중복되지 않아야 합니다.
        다음 형식으로 추가하면 좋을 것으로 예상되는 기능 목록을 제안해 주세요:
        {{
            "suggestions": [
                {{
                    "question": "이런 기능을 추가하시는 건 어떤가요?",
                    "answers": ["결제 기능", "주문 기능", "주문 조회 기능"]
                }}
            ]
        }}
        
        정보:
        {user_input}
        """
        
        # GPT API 호출
        completion = await openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 소프트웨어 요구사항 분석가입니다."
                },
                {
                    "role": "user",
                    "content": create_feature_prompt.format(user_input=user_input)
                }
            ]
        )
    
    # GPT 응답에서 features 추출
    try:
        content = completion.choices[0].message.content
        logger.info(f"GPT API 원본 응답: {content}")
        
        # JSON 형식 정리
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        logger.info(f"정리된 JSON 문자열: {content}")
        feature_names = json.loads(content)
        logger.info(f"파싱된 features: {feature_names}")
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {str(e)}")
        logger.error(f"파싱 실패한 내용: {content}")
        raise Exception(f"GPT API 응답 파싱 중 오류 발생: {str(e)}") from e
    
    except Exception as e:
        logger.error(f"GPT API 응답 처리 중 오류 발생: {str(e)}")
        raise Exception(f"GPT API 응답 처리 중 오류 발생: {str(e)}") from e
        
    # features, suggestions 추출
    features = feature_names.get("features", [])
    suggestions = feature_names.get("suggestions", [])
    
    # 파싱된 결과 반환
    result = {
        "suggestion": {
            "features": features,
            "suggestions": suggestions
        }
    }
    logger.info(f"최종 반환 결과: {result}")
    
    # Redis에 저장할 데이터 구성 (features와 suggestions의 answers만 포함)
    all_features = features + [answer for suggestion in suggestions for answer in suggestion["answers"]]
    redis_data = {
        "email": email,
        "features": all_features
    }
    
    # Redis에 저장
    await save_to_redis(f"features:{email}", redis_data)
    logger.info(f"Redis에 데이터 저장 완료: {redis_data}")
    
    return result

async def update_feature_definition(email: str, feedback: str) -> Dict[str, Any]:
    """
    사용자 피드백을 기반으로 기능 정의서를 업데이트합니다.
    
    Args:
        email (str): 사용자 이메일
        feedback (str): 사용자 피드백
        
    Returns:
        Dict[str, Any]: 업데이트된 기능 정의서 데이터
            - features: 업데이트된 기능 목록
            - isNextStep: 다음 단계 진행 여부 (0: 종료, 1: 계속)
    """
    
    feature_data = await load_from_redis(f"features:{email}")
    if not feature_data:
        raise ValueError(f"Project information for user {email} not found")
    
    # 이미 딕셔너리인 경우 JSON 파싱 생략
    if isinstance(feature_data, str):
        feature_data = json.loads(feature_data)
    
    current_features = feature_data.get("features", [])
    
    # 1. 피드백 분석
    update_prompt = """
    당신은 사용자의 피드백을 분석하여 기능 정의 단계를 계속 진행할지 종료할지 판단하는 전문가입니다.

    다음은 기능 정의 단계에서 받은 사용자의 피드백입니다:
    {feedback}

    이 피드백이 다음 중 어떤 유형인지 판단해주세요:

    1. 수정/추가 요청:
       - 새로운 기능 추가 요청
       - 기존 기능 수정 요청
       - 기능 목록 변경 요청
       예시: "장바구니 기능 추가해주세요", "결제 기능도 필요해요"

    2. 종료 요청:
       - 기능 정의 완료 의사 표현
       - 더 이상의 수정이 필요 없다는 의견
       - 다음 단계로 넘어가고 싶다는 의견
       예시: "이대로 좋습니다", "더 이상 수정할 필요 없어요", "다음으로 넘어가죠"

    응답은 다음 중 하나로만 해주세요:
    - 수정/추가 요청인 경우: "continue"
    - 종료 요청인 경우: "end"
    """
    
    formatted_prompt = update_prompt.format(feedback=feedback)
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "당신은 소프트웨어 요구사항 분석가입니다."},
            {"role": "user", "content": formatted_prompt}
        ]
    )
    
    if "end" in completion.choices[0].message.content.lower():
        result = {
            "features": current_features,
            "isNextStep": 0
        }
        return result
    
    # 2. 기능 업데이트
    update_features_prompt = """
    현재 기능 정의서와 사용자 피드백을 기반으로 기능을 업데이트해주세요.

    현재 기능 목록:
    {current_features}

    사용자 피드백:
    {feedback}

    응답은 반드시 다음과 같은 JSON 형식으로만 작성해주세요:
    {{
        "features": [
            "기능명1",
            "기능명2",
            "기능명3"
        ]
    }}

    추가 설명이나 다른 텍스트는 포함하지 마세요.
    """
    
    formatted_update_prompt = update_features_prompt.format(
        current_features=current_features,
        feedback=feedback
    )
    update_response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "당신은 소프트웨어 요구사항 분석가입니다. JSON 형식으로만 응답해주세요."},
            {"role": "user", "content": formatted_update_prompt}
        ]
    )
    
    # 응답 파싱
    content = update_response.choices[0].message.content
    logger.info(f"GPT API 원본 응답: {content}")
    
    try:
        # 응답에서 JSON 부분만 추출
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # 줄바꿈과 불필요한 공백 제거
        content = content.replace("\n", "").replace("  ", " ").strip()
        logger.info(f"정리된 JSON 문자열: {content}")
        
        updated_features = json.loads(content)
        logger.info(f"파싱된 features: {updated_features}")
        
        if not isinstance(updated_features, dict) or "features" not in updated_features:
            raise ValueError("응답이 올바른 형식이 아닙니다. 'features' 키가 필요합니다.")
        
        if not isinstance(updated_features["features"], list):
            raise ValueError("'features'는 리스트 형식이어야 합니다.")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {str(e)}")
        logger.error(f"파싱 실패한 내용: {content}")
        raise Exception(f"GPT API 응답 파싱 중 오류 발생: {str(e)}") from e
    except Exception as e:
        logger.error(f"GPT API 응답 처리 중 오류 발생: {str(e)}")
        raise Exception(f"GPT API 응답 처리 중 오류 발생: {str(e)}") from e
    
    # Redis 업데이트
    # 업데이트 전 데이터 로깅
    logger.info(f"업데이트 전 Redis 데이터: {feature_data}")
    
    # 기능 목록 업데이트
    feature_data["features"] = updated_features["features"]
    
    # 업데이트할 데이터 로깅
    logger.info(f"업데이트할 Redis 데이터: {feature_data}")
    
    # Redis 업데이트
    redis_data = {
        "email": email,
        "features": updated_features["features"]
    }
    # Redis에 저장
    try:
        await save_to_redis(f"features:{email}", redis_data)
        logger.info(f"Redis에 데이터 저장 완료: {redis_data}")
    except Exception as e:
        logger.error(f"Redis 저장 중 오류 발생: {str(e)}")
        raise Exception(f"Redis 저장 중 오류 발생: {str(e)}") from e
    
    # 업데이트 확인
    updated_data = await load_from_redis(f"features:{email}")
    logger.info(f"업데이트 후 Redis 데이터: {updated_data}")
    
    # API 응답용 결과 반환
    result = {
        "features": updated_features["features"],
        "isNextStep": 1
    }
    
    return result
    