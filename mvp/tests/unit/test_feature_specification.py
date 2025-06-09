import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from feature_specification import (assign_featureId, calculate_priority,
                                   create_feature_specification,
                                   update_feature_specification)


def test_assign_featureId():
    """기능 ID 할당 테스트"""
    feature = {"name": "테스트 기능"}
    result = assign_featureId(feature)
    
    assert "_id" in result
    assert isinstance(result["_id"], str)
    assert len(result["_id"]) > 0

def test_calculate_priority():
    """우선순위 계산 테스트"""
    # 최소 우선순위 테스트
    assert calculate_priority(30, 5) == 1
    
    # 최대 우선순위 테스트
    assert calculate_priority(0, 1) == 300
    
    # 중간 우선순위 테스트
    priority = calculate_priority(15, 3)
    assert 1 <= priority <= 300

@pytest.mark.asyncio
async def test_create_feature_specification():
    """기능 명세서 생성 테스트"""
    mock_project_data = {
        "projectId": "test-project",
        "startDate": "2024-03-01",
        "endDate": "2024-03-31",
        "members": [
            {
                "name": "테스트 멤버",
                "profiles": [
                    {
                        "projectId": "test-project",
                        "positions": ["BE", "FE"]
                    }
                ]
            }
        ]
    }
    
    mock_feature_data = ["로그인 기능", "회원가입 기능"]
    
    mock_response = {
        "content": """
        {
            "features": [
                {
                    "name": "로그인 기능",
                    "useCase": "사용자 로그인",
                    "input": "이메일, 비밀번호",
                    "output": "로그인 성공/실패",
                    "precondition": "회원가입 완료",
                    "postcondition": "로그인 상태",
                    "startDate": "2024-03-01",
                    "endDate": "2024-03-15",
                    "difficulty": 2
                }
            ]
        }
        """
    }
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm), \
         patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        
        mock_load_redis.side_effect = [
            json.dumps(mock_project_data),
            json.dumps(mock_feature_data)
        ]
        
        result = await create_feature_specification("test@example.com")
        
        assert "features" in result
        assert len(result["features"]) > 0
        assert result["features"][0]["name"] == "로그인 기능"

@pytest.mark.asyncio
async def test_update_feature_specification():
    """기능 명세서 업데이트 테스트"""
    mock_response = {
        "content": """
        {
            "isNextStep": 1
        }
        """
    }
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm):
        result = await update_feature_specification(
            email="test@example.com",
            feedback="테스트 피드백",
            createdFeatures=[],
            modifiedFeatures=[],
            deletedFeatures=[]
        )
        
        assert "isNextStep" in result
        assert result["isNextStep"] == 1

@pytest.mark.asyncio
async def test_create_feature_specification_no_data():
    """데이터가 없는 경우 테스트"""
    with patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        mock_load_redis.return_value = None
        
        with pytest.raises(ValueError):
            await create_feature_specification("test@example.com") 