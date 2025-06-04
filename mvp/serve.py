import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import redis.asyncio as aioredis
from create_epic import create_sprint
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from feature_definition import (create_feature_definition,
                                update_feature_definition)
from feature_specification import (create_feature_specification,
                                   update_feature_specification)
from meeting_analysis import (analyze_meeting_document,
                              convert_action_items_to_tasks)
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

### 요청 모델
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
    startDate: datetime

class MeetingPOSTRequest(BaseModel):
    meetingId: str
    title: str
    content: str
    projectId: str

class CreateActionItemPOSTRequest(BaseModel):
    actionItems: List[str]
    projectId: str


### 응답 모델
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
    isNextStep: bool

class FeedbackFeatureSpecificationResponse(BaseModel):
    features: List[Dict[str, Any]]
    isNextStep: bool

class CreateSprintResponse(BaseModel):
    sprint: Dict[str, Any]
    epics: List[Dict[str, Any]]
    
class CreateMeetingResponse(BaseModel):
    summary: str
    actionItems: List[str]
    
class CreateActionItemResponse(BaseModel):
    tasks: List[Dict[str, Any]]


app = FastAPI(docs_url="/docs")

@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Uvicorn 서버 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# 처리 시간 측정 CORS 설정
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    startTime = datetime.now()
    response = await call_next(request)
    logger.info(f"Processing Time (처리 소요 시간): {datetime.now() - startTime}")
    return response


# API Mapping
@app.post("/project/definition", response_model=CreateFeatureDefinitionResponse)
async def post_definition(request: FeatureDefinitionPOSTRequest):
    try:
        logger.info(f"📨 POST /definition 요청 수신: {request}")
        logger.info(f"📨 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        logger.info(f"📨 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        logger.info(f"📨 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        logger.info(f"📨 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        logger.info(f"📨 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = await create_sprint(request.projectId, request.pendingTasksIds, request.startDate)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"스프린트 생성 중 오류 발생: {str(e)}"
        )

@app.post("/meeting", response_model=CreateMeetingResponse)
async def post_meeting(request: MeetingPOSTRequest):
    try:
        #content = await file.read()
        #content = content.decode('utf-8')
        logger.info(f"📨 POST /meeting 요청 수신: {request}")
        logger.info(f"📨 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = await analyze_meeting_document(request.meetingId, request.title, request.content, request.projectId)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"회의록 요약 중 오류 발생: {str(e)}"
        )

@app.post("/meeting/action-items", response_model=CreateActionItemResponse)
async def post_action_items(request: CreateActionItemPOSTRequest):
    try:
        logger.info(f"📨 POST /meeting/action-items 요청 수신: {request}")
        logger.info(f"📨 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = await convert_action_items_to_tasks(request.actionItems, request.projectId)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"회의록 액션 아이템 생성 중 오류 발생: {str(e)}"
        )


# 실행 예시
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
