import os
from datetime import datetime
from typing import Optional

import openai
import torch
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

#from sentence_transformers import SentenceTransformer
# gpt는 사용자 입력을 그대로 받아도 되므로 임베딩을 생성하는 것은 오버헤드에 해당됨. 따라서 임베딩과 관련된 내용을 우선 각주 처리함.

# OPENAI API KEY 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다. "
        "1. .env.example 파일을 .env로 복사하고 "
        "2. .env 파일에 실제 API 키를 입력해주세요."
    )

# OpenAI 클라이언트 초기화
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# FastAPI application 초기화
app = FastAPI()

# MiniLM 모델 로드 (자연어 임베딩용)
#embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# MongoDB 연결 설정
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["checkmate"]
definitions_collection = db["feature_definitions"]

# 데이터 요청 모델 정의
class FeatureRequest(BaseModel):
    user_input: str
    feature_id: Optional[str] = None

# 기능 정의서 응답 모델
class FeatureDefinition(BaseModel):
    feature_id: str
    feature_name: str
    description: str
    purpose: str
    use_cases: list[str]
    inputs: list[str]
    outputs: list[str]
    preconditions: list[str]
    postconditions: list[str]
    ui_involved: list[str]
    related_features: list[str]
    priority: str
    notes: list[str]

class FeatureResponse(BaseModel):
    message: str
    feature_id: str
    definition: FeatureDefinition

# 기능 정의서 생성 및 수정
@app.post("/generate/definition", response_model=FeatureResponse)
async def generate_definition(request: FeatureRequest):
    """
    기능 정의서를 생성하거나 수정하는 API
    - feature_id가 없으면: 새로운 기능 정의서 생성
    - feature_id가 있으면: 기존 기능 정의서 수정
    """
    try:
        user_input = request.user_input
        feature_id = request.feature_id

        # 사용자 입력을 임베딩
        #embeddings = embedding_model.encode(user_input) 
        
        # GPT-4를 사용하여 기능 정의서 생성
        system_role = """
        당신은 애자일 소프트웨어 개발팀을 위해
        개발자 친화적이고 상세한 기능 정의서를 작성하는 똑똑한 AI 도우미입니다.
        사용자는 기능 이름, 간단한 설명, 목적, 예시 사용 사례와 같은 일부 정보만 자연어로 제공할 것입니다.
        당신의 임무는 이러한 입력을 기반으로 전체 기능 정의서를 자동으로 완성하는 것입니다.
        """
        system_content = """
        다음의 항목들을 반드시 포함해야 하며, 사용자 입력이 불완전하거나 모호하더라도 유추하여 채워야 합니다:
        1. 기능 ID: 고유 ID 부여. (예: "F001", "F002")
        2. 기능 이름: 사용자 입력을 요약하여 기능 이름 명시. (예: "회원 가입")
        3. 기능 설명: 기능 이름을 참고하여 기능의 역할 자연어로 설명. (예: "사용자가 이메일과 비밀번호를 입력해 계정을 생성할 수 있는 기능입니다.")\
            -> 3, 4, 5, 6, 7, 10(화면 구성이 필요한지 판단한 결과), 13을 융합
        4. 기능 목적: 이 기능이 왜 필요한지 자연어로 설명. (예: "사용자가 로그인 기능을 사용하기 위해 사전에 회원가입을 해야 합니다.")
        5. 사용 시나리오: 실제 사용 예시를 1~2개 구체적으로 작성. (예: "처음 방문한 사용자가 가입 버튼을 눌러 폼을 작성하고 가입한다.")
        6. 입력값: 기능 실행을 위해 사용자나 시스템으로부터 전달 받아야 하는 값. (예: "이메일", "비밀번호")
        7. 출력값: 기능 실행 후 출력되는 정보. (예: "가입 완료 메시지", "대시보드 페이지로 이동")
        # 8. 전제 조건: 기능 실행 전 필요한 조건. (예: "가입 폼이 화면에 표시되어 있어야 함")
        # 9. 후행 조건: 기능 실행 후 보장되는 결과. (예: "DB에 사용자 정보가 저장됨")
        10. 사용자 화면: 화면 출력에서 사용되는 경우, 어떤 화면인지 명시. (예: "웹 브라우저 회원가입 페이지")
        # 11. 관련 기능: 관련된 기능 추론. (예: "로그인", "비밀번호 재설정")
        12. 우선순위: High / Medium / Low 중 하나로 우선순위 추정. 사용자 입력이 있을 경우 사용자 입력을 우선적으로 반영.
        13. 참고사항: 그 외 고려사항, 제약 조건, 가정이 있을 경우 출력. (예: "이메일 중복 확인 기능이 함께 있어야 함")"""
        
        output_format = """출력은 JSON 형식으로 제공해주세요. 각 필드는 다음과 같은 형식을 가져야 합니다:
        {
        "feature_id": "F001",
        "feature_name": "기능 이름",
        "description": "기능 설명",
        "purpose": "기능 목적",
        "use_cases": ["사용 사례 1", "사용 사례 2"],
        "inputs": ["입력 1", "입력 2"],
        "outputs": ["출력 1", "출력 2"],
        "preconditions": ["전제 조건 1", "전제 조건 2"],
        "postconditions": ["후행 조건 1", "후행 조건 2"],
        "ui_involved": ["UI 1", "UI 2"],
        "related_features": ["관련 기능 1", "관련 기능 2"],
        "priority": "High",
        "notes": ["참고사항 1", "참고사항 2"]
        }"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "system", "content": system_content},
                    {"role": "system", "content": output_format},
                    {"role": "user", "content": user_input}
                ],
                response_format={ "type": "json_object" }  # JSON 형식으로 응답을 강제
            )
            
            generated_definition = response.choices[0].message.content
            
            # JSON 문자열을 파싱하여 FeatureDefinition 객체 생성
            try:
                import json
                definition_dict = json.loads(generated_definition)
                feature_definition = FeatureDefinition(**definition_dict)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"GPT 응답을 JSON으로 파싱하는 중 오류가 발생했습니다: {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"기능 정의서 생성 중 오류가 발생했습니다: {str(e)}"
                )
            
            if feature_id is None:
                # 새로운 기능 정의서 생성
                document = {
                    "user_input": user_input,
                    #"embeddings": embeddings.tolist(),
                    "definition": definition_dict,
                    "status": "draft",
                    "created_at": datetime.now()
                }
                result = definitions_collection.insert_one(document)
                new_feature_id = str(result.inserted_id)
                
                return FeatureResponse(
                    message=f"새로운 기능 정의서가 생성되었습니다. {new_feature_id}",
                    feature_id=new_feature_id,
                    definition=feature_definition
                )
            else:
                # 기존 기능 정의서 수정
                try:
                    object_id = ObjectId(feature_id)
                except:
                    return FeatureResponse(
                        message=f"기능 정의서가 수정되었습니다. {feature_id}",
                        feature_id=feature_id,
                        definition=feature_definition
                    )

                try:
                    result = definitions_collection.update_one(
                        {"_id": object_id},
                        {
                            "$set": {
                                "user_input": user_input,
                                #"embeddings": embeddings.tolist(),
                                "definition": definition_dict,
                                "status": "modified",
                                "updated_at": datetime.now()
                            }
                        }
                    )
                    if result.modified_count == 0:
                        raise HTTPException(status_code=404, detail=f"기능 정의서 {feature_id}가 존재하지 않음")
                    
                    return FeatureResponse(
                        message=f"기능 정의서가 수정되었습니다. {feature_id}",
                        feature_id=feature_id,
                        definition=feature_definition
                    )
                except Exception as e:
                    raise HTTPException(status_code=404, detail="Feature가 존재하지 않음")
            
        except openai.APIError as e:
            if "billing_not_active" in str(e):
                raise HTTPException(
                    status_code=402,
                    detail="OpenAI API 결제 계정이 활성화되지 않았습니다. OpenAI 웹사이트에서 결제 정보를 확인해주세요."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI API 오류가 발생했습니다: {str(e)}"
                )
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
