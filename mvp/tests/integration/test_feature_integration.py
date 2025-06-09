import os

import pytest
from feature_definition import (create_feature_definition,
                                update_feature_definition)
from redis_setting import load_from_redis


@pytest.mark.asyncio
async def test_feature_definition_workflow():
    """기능 정의 워크플로우 통합 테스트"""
    test_email = "test@example.com"
    test_description = "테스트 프로젝트입니다."
    
    try:
        # 1. 기능 정의서 생성
        result = await create_feature_definition(
            email=test_email,
            description=test_description
        )
        
        # 기본 검증
        assert "suggestion" in result
        assert "suggestions" in result["suggestion"]
        assert len(result["suggestion"]["suggestions"]) > 0
        
        # Redis에 저장된 데이터 검증
        saved_features = await load_from_redis(f"features:{test_email}")
        assert saved_features is not None
        assert len(saved_features) > 0
        
        # 2. 기능 정의 업데이트 - 계속 진행
        update_result = await update_feature_definition(
            email=test_email,
            feedback="새로운 기능을 추가해주세요"
        )
        assert "isNextStep" in update_result
        
        # 3. 기능 정의 업데이트 - 종료
        final_result = await update_feature_definition(
            email=test_email,
            feedback="이대로 좋습니다"
        )
        assert final_result["isNextStep"] == 1
        
    except Exception as e:
        pytest.fail(f"통합 테스트 실패: {str(e)}")

@pytest.mark.asyncio
async def test_feature_definition_with_pdf():
    """PDF가 있는 기능 정의 통합 테스트"""
    test_email = "test@example.com"
    test_description = "테스트 프로젝트입니다."
    test_pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        # PDF가 있는 기능 정의서 생성
        result = await create_feature_definition(
            email=test_email,
            description=test_description,
            definition_url=test_pdf_url
        )
        
        # 기본 검증
        assert "suggestion" in result
        assert "features" in result["suggestion"]
        assert "suggestions" in result["suggestion"]
        
        # Redis에 저장된 데이터 검증
        saved_features = await load_from_redis(f"features:{test_email}")
        assert saved_features is not None
        assert len(saved_features) > 0
        
    except Exception as e:
        pytest.fail(f"PDF 통합 테스트 실패: {str(e)}")

@pytest.mark.asyncio
async def test_feature_definition_error_handling():
    """에러 처리 통합 테스트"""
    test_email = "test@example.com"
    
    # 잘못된 이메일로 테스트
    with pytest.raises(Exception):
        await create_feature_definition(
            email="",
            description=""
        )
    
    # 잘못된 피드백으로 테스트
    with pytest.raises(Exception):
        await update_feature_definition(
            email=test_email,
            feedback=""
        ) 