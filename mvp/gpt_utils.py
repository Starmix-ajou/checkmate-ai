import asyncio
import json
import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

async def safe_chat_completion(llm: ChatOpenAI, messages, retries=3):
    for i in range(retries):
        try:
            response = await llm.ainvoke(messages)
            return response
        except Exception as e:
            print(f"[{i+1}/{retries}] OpenAI API 오류: {e}")
            await asyncio.sleep(1)
    raise RuntimeError("ChatCompletion API 요청 실패")


def extract_json_from_gpt_response(content: str) -> List[Dict[str, Any]]:
    """
    GPT 응답에서 JSON 블록만 추출하고 파싱합니다.

    Args:
        content (str): GPT 모델이 반환한 전체 텍스트 응답

    Returns:
        List[Dict[str, Any]]: GPT의 응답을 최종적으로는 리스트로 반환합니다. 내부에 필드 구분을 위한 딕셔너리 구조가 있을 수 있습니다.

    Raises:
        ValueError: 유효한 JSON이 아닌 경우
    """
    if not content:
        raise ValueError("🚨 GPT 응답이 비어 있습니다.")

    logger.info(f"GPT 응답 원본: {content}")

    # 1. 코드 블록 제거 (```json 또는 ``` 로 감싼 경우)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    # 2. 응답이 그냥 문자열인 경우에도 JSON 부분만 추출
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        content = content[start:end]
    
    # 3. 개행 및 중복 공백 제거
    content = content.replace("\n", "").replace("\r", "")
    while "  " in content:
        content = content.replace("  ", " ")
    content = content.strip()
    
    # 4. 주석 처리된 부분을 제거하고, 안전하게 #이 문자열 안에 있는 경우는 제거하지 않음
    content = remove_comments_safe(content)

    logger.info(f"🔍 정리된 JSON 문자열: {content}")

    # 4. 파싱 시도
    try:
        parsed = json.loads(content)
        logger.info("✅ JSON 파싱 완료")
    except json.JSONDecodeError as e:
        error_pos = int(e.pos) if isinstance(e.pos, str) else e.pos
        error_context = content[max(0, error_pos-10):min(len(content), error_pos+10)]
        logger.error(f"❌ JSON 파싱 실패: 위치 {error_pos}, 문제 문자 주변: {error_context}")
        raise e
    
    return parsed

# 안전하게 #이 문자열 안에 있는 경우는 제거하지 않음
def remove_comments_safe(content: str) -> str:
    result = []
    in_string = False
    i = 0
    while i < len(content):
        char = content[i]
        if char == '"' and (i == 0 or content[i - 1] != '\\'):
            in_string = not in_string
        if char == '#' and not in_string:
            while i < len(content) and content[i] != '\n':
                i += 1
            continue
        result.append(char)
        i += 1
    return ''.join(result)