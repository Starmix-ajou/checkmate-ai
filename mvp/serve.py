import logging
import os
from typing import Any, Dict, List, Optional

import httpx
import redis.asyncio as aioredis
from create_epic import create_sprint
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from feature_definition import (create_feature_definition,
                                update_feature_definition)
from feature_specification import (create_feature_specification,
                                   update_feature_specification)
from mongodb_setting import test_mongodb_connection
from pydantic import BaseModel
from redis_setting import test_redis_connection

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(message)s', 
    #filename='mvp.log'
)
logger = logging.getLogger(__name__)

class FeatureDefinitionPOSTRequest(BaseModel):
    email: str
    description: str
    definitionUrl: Optional[str] = None
    
class FeatureDefinitionPUTRequest(BaseModel):
    email: str
    feedback: str
    
class FeatureSpecificationPOSTRequest(BaseModel):
    email: str

class FeatureSpecificationPUTRequest(BaseModel):
    email: str
    feedback: Optional[str] = None
    createdFeatures: Optional[List[Dict[str, Any]]] = None
    modifiedFeatures: Optional[List[Dict[str, Any]]] = None
    deletedFeatures: Optional[List[str]] = None
    
class EpicPOSTRequest(BaseModel):
    projectId: str
    pendingTasksIds: Optional[List[str]] = None

class FeatureDefinitionSuggestion(BaseModel):
    features: List[str]
    suggestions: List[dict]

class CreateFeatureDefinitionResponse(BaseModel):
    suggestion: FeatureDefinitionSuggestion

class CreateFeatureSpecificationResponse(BaseModel):
    features: List[Dict[str, Any]]

class CreateSprintResponse(BaseModel):
    sprint: Dict[str, Any]
    epics: List[Dict[str, Any]]

class FeedbackFeatureDefinitionResponse(BaseModel):
    features: List[str]
    is_next_step: bool

class FeedbackFeatureSpecificationResponse(BaseModel):
    features: List[Dict[str, Any]]
    is_next_step: bool

class CreateSprintResponse(BaseModel):
    sprint: Dict[str, Any]
    epics: List[Dict[str, Any]]

app = FastAPI(docs_url="/docs")

@app.on_event("startup")
async def startup_event():
    try:
        # Redis 연결 테스트
        await test_redis_connection()
        logger.info("Redis 연결 테스트 완료")
        
        # MongoDB 연결 테스트
        await test_mongodb_connection()
        logger.info("MongoDB 연결 테스트 완료")
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {str(e)}")
        raise e

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    logger.error("🔥 예외 발생:", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "서버 및 API 실행 중 오류 발생"}
    )

# API Mapping
@app.post("/project/definition", response_model=CreateFeatureDefinitionResponse)
async def post_definition(request: FeatureDefinitionPOSTRequest):
    try:
        logger.info(f"📨 POST /definition 요청 수신: {request}")
        result = await create_feature_definition(request.email, request.description, request.definitionUrl)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"기능 정의서 생성 중 오류 발생: {str(e)}"
        )

@app.put("/project/definition", response_model=FeedbackFeatureDefinitionResponse)
async def put_definition(request: FeatureDefinitionPUTRequest):
    try:
        logger.info(f"📨 PUT /definition 요청 수신: {request}")
        result = await update_feature_definition(request.email, request.feedback)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"기능 정의서 업데이트 중 오류 발생: {str(e)}"
        )
    
@app.post("/project/specification", response_model=CreateFeatureSpecificationResponse)
async def post_specification(request: FeatureSpecificationPOSTRequest):
    try:
        logger.info(f"📨 POST /specification 요청 수신: {request}")
        result = await create_feature_specification(request.email)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"기능 명세서 생성 중 오류 발생: {str(e)}"
        )

@app.put("/project/specification", response_model=FeedbackFeatureSpecificationResponse)
async def put_specification(request: FeatureSpecificationPUTRequest):
    try:
        logger.info(f"📨 PUT /specification 요청 수신: {request}")
        result = await update_feature_specification(request.email, request.feedback, request.createdFeatures, request.modifiedFeatures, request.deletedFeatures)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"기능 명세서 업데이트 중 오류 발생: {str(e)}"
        )

@app.post("/sprint", response_model=CreateSprintResponse)
async def post_epic(request: EpicPOSTRequest):
    try:
        logger.info(f"📨 POST /sprint 요청 수신: {request}")
        result = await create_sprint(request.projectId, request.pendingTasksIds)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"스프린트 생성 중 오류 발생: {str(e)}"
        )

# 실행 예시
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
