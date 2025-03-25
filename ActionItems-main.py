from fastapi import FastAPI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

import json
from fastapi.testclient import TestClient

# 기본 세팅
load_dotenv()
app = FastAPI()

# API Key Settings
client = OpenAI(
    api_key = os.environ.get('API_KEY')
)

text_data = """

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
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,    # 0에 가까울수록 보수적인 응답. 정보 요약, 정리 작업은 0~0.3 사이 추천
        max_tokens=800,     # 1 token = 영단어 하나, 한글 1-2글자. 
                            # 작업량이 많고 세부적인 정리가 필요하면 800~1500, 간단한 정보 요약이면 300~500
        
    )
    ß
  
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
        