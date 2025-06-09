from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from feature_definition import (create_feature_definition,
                                update_feature_definition)
from langchain_core.messages import AIMessage


@pytest.mark.asyncio
async def test_create_feature_definition_with_predefined():
    """사전 정의된 기능 정의서가 있는 경우 테스트"""
    mock_response = AIMessage(content="""
        {
            "features": ["로그인 기능", "회원가입 기능"],
            "suggestions": [{
                "question": "이런 기능을 추가하시는 건 어떤가요?",
                "answers": ["결제 기능", "주문 기능"]
            }]
        }
        """)
    
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
    mock_response = AIMessage(content="""
        {
            "suggestions": [{
                "question": "이런 기능을 추가하시는 건 어떤가요?",
                "answers": ["로그인 기능", "회원가입 기능"]
            }]
        }
        """)
    
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
async def test_create_feature_definition_pdf_extraction_failure():
    """PDF 텍스트 추출 실패 테스트"""
    with patch('feature_definition.extract_pdf_text', new_callable=AsyncMock) as mock_extract_pdf:
        mock_extract_pdf.side_effect = Exception("PDF 추출 실패")
        
        with pytest.raises(Exception, match="PDF 추출 실패"):
            await create_feature_definition(
                email="test@example.com",
                description="테스트 프로젝트",
                definition_url="http://test.com/test.pdf"
            )

@pytest.mark.asyncio
async def test_create_feature_definition_gpt_failure():
    """GPT API 호출 실패 테스트"""
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(side_effect=Exception("GPT API 호출 실패"))
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.extract_pdf_text', new_callable=AsyncMock) as mock_extract_pdf:
        
        mock_extract_pdf.return_value = "테스트 PDF 내용"
        
        with pytest.raises(Exception, match="GPT API 호출 실패"):
            await create_feature_definition(
                email="test@example.com",
                description="테스트 프로젝트",
                definition_url="http://test.com/test.pdf"
            )

@pytest.mark.asyncio
async def test_create_feature_definition_redis_failure():
    """Redis 저장 실패 테스트"""
    mock_response = AIMessage(content="""
        {
            "features": ["로그인 기능"],
            "suggestions": [{
                "question": "이런 기능을 추가하시는 건 어떤가요?",
                "answers": ["결제 기능"]
            }]
        }
        """)
    
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.extract_pdf_text', new_callable=AsyncMock) as mock_extract_pdf, \
         patch('feature_definition.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        mock_extract_pdf.return_value = "테스트 PDF 내용"
        mock_save_redis.side_effect = Exception("Redis 저장 실패")
        
        with pytest.raises(Exception, match="Redis 저장 실패"):
            await create_feature_definition(
                email="test@example.com",
                description="테스트 프로젝트",
                definition_url="http://test.com/test.pdf"
            )

@pytest.mark.asyncio
async def test_update_feature_definition_continue():
    """기능 정의 업데이트 - 계속 진행 테스트"""
    # 첫 번째 GPT 호출 (피드백 분석)
    mock_response1 = AIMessage(content="""
        {
            "isNextStep": 0,
            "features": ["기존 기능1", "기존 기능2", "새로운 기능"]
        }
        """)
    
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response1)
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.load_from_redis', new_callable=AsyncMock) as mock_load_redis, \
         patch('feature_definition.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        mock_load_redis.return_value = ["기존 기능1", "기존 기능2"]
        
        result = await update_feature_definition(
            email="test@example.com",
            feedback="새로운 기능을 추가해주세요"
        )
        
        assert result["isNextStep"] == 0
        assert "features" in result
        mock_save_redis.assert_called_once()

@pytest.mark.asyncio
async def test_update_feature_definition_finish():
    """기능 정의 업데이트 - 종료 테스트"""
    mock_response = AIMessage(content="""
        {
            "isNextStep": 1
        }
        """)
    
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

@pytest.mark.asyncio
async def test_update_feature_definition_redis_load_failure():
    """Redis 데이터 로드 실패 테스트"""
    with patch('feature_definition.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        mock_load_redis.side_effect = Exception("Redis 로드 실패")
        
        with pytest.raises(Exception, match="Redis에서 데이터 로드 중 오류 발생: Redis 로드 실패"):
            await update_feature_definition(
                email="test@example.com",
                feedback="테스트 피드백"
            )

@pytest.mark.asyncio
async def test_update_feature_definition_gpt_failure():
    """GPT API 호출 실패 테스트"""
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(side_effect=Exception("GPT API 호출 실패"))
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        
        mock_load_redis.return_value = ["기존 기능1", "기존 기능2"]
        
        with pytest.raises(Exception, match="GPT API 호출 실패"):
            await update_feature_definition(
                email="test@example.com",
                feedback="테스트 피드백"
            )

@pytest.mark.asyncio
async def test_update_feature_definition_redis_update_failure():
    """Redis 업데이트 실패 테스트"""
    mock_response1 = AIMessage(content="""
        {
            "isNextStep": 0,
            "features": ["기존 기능1", "기존 기능2", "새로운 기능"]
        }
        """)
    
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response1)
    
    with patch('feature_definition.ChatOpenAI', return_value=mock_llm), \
         patch('feature_definition.load_from_redis', new_callable=AsyncMock) as mock_load_redis, \
         patch('feature_definition.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        mock_load_redis.return_value = ["기존 기능1", "기존 기능2"]
        mock_save_redis.side_effect = Exception("Redis 업데이트 실패")
        
        with pytest.raises(Exception, match="Redis 업데이트 중 오류 발생: Redis 업데이트 실패"):
            await update_feature_definition(
                email="test@example.com",
                feedback="테스트 피드백"
            )