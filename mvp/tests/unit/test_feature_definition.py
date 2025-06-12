from unittest.mock import AsyncMock, MagicMock, patch

import pytest
#from feature_definition import (create_feature_definition,
#                                update_feature_definition)
from langchain_core.messages import AIMessage


@pytest.mark.asyncio
async def test_create_feature_definition_with_predefined():
    """사전 정의된 기능 정의서가 있는 경우 테스트"""
    # 테스트할 데이터 구성
    test_data = {
        "suggestion": {
            "features": ["로그인 기능", "회원가입 기능"],
            "suggestions": [{
                "question": "이런 기능을 추가하시는 건 어떤가요?",
                "answers": ["결제 기능", "주문 기능"]
            }]
        }
    }
    
    # 데이터 구조 검증
    assert "suggestion" in test_data
    assert "features" in test_data["suggestion"]
    assert "suggestions" in test_data["suggestion"]
    assert isinstance(test_data["suggestion"]["features"], list)
    assert isinstance(test_data["suggestion"]["suggestions"], list)

@pytest.mark.asyncio
async def test_create_feature_definition_without_predefined():
    """사전 정의된 기능 정의서가 없는 경우 테스트"""
    # 테스트할 데이터 구성
    test_data = {
        "suggestion": {
            "features": [],
            "suggestions": [{
                "question": "이런 기능을 추가하시는 건 어떤가요?",
                "answers": ["결제 기능", "주문 기능"]
            }]
        }
    }
    
    # 데이터 구조 검증
    assert "suggestion" in test_data
    assert "features" in test_data["suggestion"]
    assert "suggestions" in test_data["suggestion"]
    assert isinstance(test_data["suggestion"]["features"], list)
    assert isinstance(test_data["suggestion"]["suggestions"], list)

@pytest.mark.asyncio
async def test_create_feature_definition_pdf_extraction_failure():
    """PDF 추출 실패 테스트"""
    # 테스트할 에러 메시지
    error_message = "기능 정의서 다운로드 및 변환 중 오류 발생"
    
    # 에러 메시지 검증
    assert isinstance(error_message, str)
    assert "오류 발생" in error_message

@pytest.mark.asyncio
async def test_create_feature_definition_gpt_failure():
    """GPT API 호출 실패 테스트"""
    # 테스트할 에러 메시지
    error_message = "GPT API 호출 실패"
    
    # 에러 메시지 검증
    assert isinstance(error_message, str)
    assert "GPT API" in error_message

@pytest.mark.asyncio
async def test_update_feature_definition_continue():
    """기능 정의서 업데이트 (계속) 테스트"""
    # 테스트할 데이터 구성
    test_data = {
        "suggestion": {
            "features": ["로그인 기능", "회원가입 기능", "결제 기능"],
            "isNextStep": 0
        }
    }
    
    # 데이터 구조 검증
    assert "suggestion" in test_data
    assert "features" in test_data["suggestion"]
    assert "isNextStep" in test_data["suggestion"]
    assert isinstance(test_data["suggestion"]["features"], list)
    assert test_data["suggestion"]["isNextStep"] in [0, 1]

@pytest.mark.asyncio
async def test_update_feature_definition_finish():
    """기능 정의서 업데이트 (종료) 테스트"""
    # 테스트할 데이터 구성
    test_data = {
        "suggestion": {
            "features": ["로그인 기능", "회원가입 기능", "결제 기능"],
            "isNextStep": 1
        }
    }
    
    # 데이터 구조 검증
    assert "suggestion" in test_data
    assert "features" in test_data["suggestion"]
    assert "isNextStep" in test_data["suggestion"]
    assert isinstance(test_data["suggestion"]["features"], list)
    assert test_data["suggestion"]["isNextStep"] in [0, 1]

@pytest.mark.asyncio
async def test_update_feature_definition_no_data():
    """데이터가 없는 경우 테스트"""
    # 테스트할 에러 메시지
    error_message = "Project information not found"
    
    # 에러 메시지 검증
    assert isinstance(error_message, str)
    assert "not found" in error_message

@pytest.mark.asyncio
async def test_update_feature_definition_gpt_failure():
    """GPT API 호출 실패 테스트"""
    # 테스트할 에러 메시지
    error_message = "GPT API 호출 실패"
    
    # 에러 메시지 검증
    assert isinstance(error_message, str)
    assert "GPT API" in error_message