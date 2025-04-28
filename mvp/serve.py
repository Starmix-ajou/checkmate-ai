import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from .feature_definition import (create_feature_definition,
                                update_feature_definition)
from .feature_specification import (create_feature_specification,
                                   update_feature_specification)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class FeatureDefinition(BaseModel):
    name: str

class Feature(BaseModel):
    name: str
    useCase: str
    input: str
    output: str

class FeatureDefinitionPOSTRequest(BaseModel):
    email: str
    description: str
    definitionUrl: Optional[str] = None

class FeatureDefinitionPOSTResponse(BaseModel):
    suggestion: Dict[str, Any]

class FeatureDefinitionPUTRequest(BaseModel):
    email: str
    feedback: Optional[str] = None

class FeatureDefinitionPUTResponse(BaseModel):
    features: List[FeatureDefinition]
    isNextStep: bool

class FeatureSpecificationPOSTRequest(BaseModel):
    email: str

class FeatureSpecificationPOSTResponse(BaseModel):
    features: List[Feature]

class FeatureSpecificationPUTRequest(BaseModel):
    email: str
    feedback: str

class FeatureSpecificationPUTResponse(BaseModel):
    features: List[Feature]
    isNextStep: bool
    
    
API_KEY = "OPENAI_API_KEY"
app = FastAPI(docs_url="/docs")

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    logger.error("🔥 예외 발생:", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "서버 및 API 실행 중 오류 발생"}
    )

# API Mapping
@app.post("/project/definition", response_model=Dict[str, Any])
async def post_definition(request: FeatureDefinitionPOSTRequest):
    try:
        logger.info(f"📨 POST /definition 요청 수신: {request}")
        result = await create_feature_definition(request.email, request.description, request.definitionUrl)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"기능 정의서 생성 중 오류 발생: {str(e)}"
        )

@app.put("/project/definition", response_model=Dict[str, Any])
async def put_definition(request: FeatureDefinitionPUTRequest):
    try:
        logger.info(f"📨 PUT /definition 요청 수신: {request}")
        result = await update_feature_definition(request.email, request.feedback)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"기능 정의서 업데이트 중 오류 발생: {str(e)}"
        )
    
@app.post("/project/specification", response_model=Dict[str, Any])
async def post_specification(request: FeatureSpecificationPOSTRequest):
    try:
        logger.info(f"📨 POST /specification 요청 수신: {request}")
        result = await create_feature_specification(request.email)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"기능 명세서 생성 중 오류 발생: {str(e)}"
        )

@app.put("/project/specification", response_model=Dict[str, Any])
async def put_specification(request: FeatureSpecificationPUTRequest):
    try:
        logger.info(f"📨 PUT /specification 요청 수신: {request}")
        result = await update_feature_specification(request.email, request.feedback)
        logger.info(f"✅ 처리 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"🔥 예외 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"기능 명세서 업데이트 중 오류 발생: {str(e)}"
        )


# 실행 예시
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
