from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from feature_definition import (create_feature_definition,
                                update_feature_definition)


@pytest.mark.asyncio
async def test_create_feature_definition_with_predefined():
    """사전 정의된 기능 정의서가 있는 경우 테스트"""
    mock_response = {
        "content": """
        {
            "features": ["로그인 기능", "회원가입 기능"],
            "suggestions": [{
                "question": "이런 기능을 추가하시는 건 어떤가요?",
                "answers": ["결제 기능", "주문 기능"]
            }]
        }
        """
    }
    
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.extract_pdf_text', new_callable=AsyncMock) as mock_extract_pdf, \
         patch('feature_definition.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        mock_extract_pdf.return_value = "테스트 PDF 내용"
        
        result = await create_feature_definition(
            email="test@example.com",
            description="테스트 프로젝트",
            definition_url="http://test.com/test.pdf"
        )
        
        assert "features" in result["suggestion"]
        assert "suggestions" in result["suggestion"]
        assert len(result["suggestion"]["features"]) == 2
        assert len(result["suggestion"]["suggestions"][0]["answers"]) == 2
        mock_save_redis.assert_called_once()

@pytest.mark.asyncio
async def test_create_feature_definition_without_predefined():
    """사전 정의된 기능 정의서가 없는 경우 테스트"""
    mock_response = {
        "content": """
        {
            "suggestions": [{
                "question": "이런 기능을 추가하시는 건 어떤가요?",
                "answers": ["로그인 기능", "회원가입 기능"]
            }]
        }
        """
    }
    
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        result = await create_feature_definition(
            email="test@example.com",
            description="테스트 프로젝트"
        )
        
        assert "suggestions" in result["suggestion"]
        assert len(result["suggestion"]["suggestions"][0]["answers"]) == 2
        mock_save_redis.assert_called_once()

@pytest.mark.asyncio
async def test_update_feature_definition_continue():
    """기능 정의 업데이트 - 계속 진행 테스트"""
    mock_response = {
        "content": """
        {
            "isNextStep": 0
        }
        """
    }
    
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        
        mock_load_redis.return_value = ["기존 기능1", "기존 기능2"]
        
        result = await update_feature_definition(
            email="test@example.com",
            feedback="새로운 기능을 추가해주세요"
        )
        
        assert result["isNextStep"] == 0

@pytest.mark.asyncio
async def test_update_feature_definition_finish():
    """기능 정의 업데이트 - 종료 테스트"""
    mock_response = {
        "content": """
        {
            "isNextStep": 1
        }
        """
    }
    
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        
        mock_load_redis.return_value = ["기존 기능1", "기존 기능2"]
        
        result = await update_feature_definition(
            email="test@example.com",
            feedback="이대로 좋습니다"
        )
        
        assert result["isNextStep"] == 1

@pytest.mark.asyncio
async def test_update_feature_definition_no_data():
    """기능 정의 업데이트 - 데이터 없음 테스트"""
    with patch('feature_definition.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        mock_load_redis.return_value = None
        
        with pytest.raises(ValueError):
            await update_feature_definition(
                email="test@example.com",
                feedback="테스트 피드백"
            ) 