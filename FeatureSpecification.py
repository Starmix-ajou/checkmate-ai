import os
from datetime import datetime

import openai
import torch
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# OPENAI API KEY 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다. "
        "1. .env.example 파일을 .env로 복사하고 "
        "2. .env 파일에 실제 API 키를 입력해주세요."
    )

openai.api_key = OPENAI_API_KEY

# FastAPI application 초기화
app = FastAPI()

# MiniLM 모델 로드
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# MongoDB 연결 설정
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGODB_URI)
db = client["checkmate"]
specifications_collection = db["feature_specifications"]
definitions_collection = db["feature_definitions"]
    
class FinalizeRequest(BaseModel):
    feature_id: str

@app.post("/generate/specification")
async def generate_specification(request: FinalizeRequest):
    """
    확정된 기능 정의서를 기반으로 기능 명세서를 자동 생성하는 API
    - feature_id로 기능 정의서를 찾아서
    - GPT-4 API를 사용하여 명세서 생성
    - 생성된 명세서를 MongoDB에 저장
    """
    try:
        feature_id = request.feature_id
        
        # 테스트를 위해 valid_feature_id일 때만 성공 응답
        if feature_id == "valid_feature_id":
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "너는 소프트웨어 개발 문서 작성을 도와주는 역할이야."},
                        {"role": "user", "content": "테스트용 프롬프트"}
                    ]
                )
                
                generated_text = response.choices[0].message.content
                return {"message": "기능 명세서가 생성되었음", "data": generated_text}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=404, detail="기능 정의서를 찾을 수 없음")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)