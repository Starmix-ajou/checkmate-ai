import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import BaseModel

# 기본 세팅
load_dotenv()
app = FastAPI()

# API Key Settings
client = OpenAI(
    api_key = os.environ.get('OPENAI_API_KEY')
)

text_data = """
2024년 3월 20일 프로젝트 회의록

참석자: 김철수, 이영희, 박민수

1. 프로젝트 진행 상황
- 현재 기능 개발 70% 완료
- UI/UX 디자인 리뷰 필요
- 테스트 케이스 작성 시작 예정

2. 다음 단계 계획
- 이영희: 로그인 페이지 디자인 수정 필요 (3월 25일까지)
- 김철수: 백엔드 API 문서화 작업 진행 (3월 30일까지)
- 박민수: 테스트 케이스 작성 및 실행 (4월 5일까지)

3. 이슈 사항
- 보안 취약점 발견, 긴급 패치 필요
- 모바일 버전 호환성 문제 해결 필요

다음 회의: 3월 27일 오후 2시
"""
payload = {
    "text": text_data
}

class MeetingNote(BaseModel):
    text: str


def generate_action_items(text: str):
    prompt = f"""
아래는 회의록입니다. 다음 회의록을 읽고 Action Item List를 생성하세요.

각 Action Item은 다음 항목을 포함해야 합니다:
- Task: 수행해야 할 작업
- Owner: 작업을 수행할 담당자
- Deadline: 작업 마감 기한 (YY-MM-DD 형식)

회의록:
\"\"\"
{text}
\"\"\"

응답을 JSON 리스트 형식으로만 작성하세요. 추가 설명, 말머리, 코드블록 마크다운을 포함하지 마세요.
예시:
[
    {{
        "tasK": "로그인 기능 구현",
        "owner": "홍길동",
        "deadline": "25-08-31"
    }},
    ...
]
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,    # 0에 가까울수록 보수적인 응답. 정보 요약, 정리 작업은 0~0.3 사이 추천
        max_tokens=800,     # 1 token = 영단어 하나, 한글 1-2글자. 
                            # 작업량이 많고 세부적인 정리가 필요하면 800~1500, 간단한 정보 요약이면 300~500
        
    )
  
    action_items_json = response.choices[0].message.content
    print("GPT가 생성한 응답: ", action_items_json)   
    try:
        action_items = json.loads(action_items_json)
        return action_items
    except json.JSONDecodeError:
        print("JSON 디코딩 오류 발생")
        print("JSON Decoding 결과: ", action_items_json)
        return []


# API endpoint
@app.post("/extract_action_items/")
async def extract_action_items(meeting_note: MeetingNote):
    action_items = generate_action_items(meeting_note.text)
    return {
        "message": "회의록 분석 완료",
        "action_items": action_items
    }
    
    
# 내부 간단 테스트
if __name__ == "__main__":
    meeting_note = MeetingNote(**payload)
    result = generate_action_items(meeting_note.text)
    print("테스트 결과: ", result)
        