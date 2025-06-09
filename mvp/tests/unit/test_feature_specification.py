import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from feature_specification import (assign_featureId, calculate_priority,
                                   create_feature_specification,
                                   update_feature_specification)
from langchain_core.messages import AIMessage


def test_assign_featureId():
    """기능 ID 할당 테스트"""
    feature = {"name": "테스트 기능"}
    result = assign_featureId(feature)
    
    assert "_id" in result
    assert isinstance(result["_id"], str)
    assert len(result["_id"]) > 0

def test_assign_featureId_invalid_input():
    """잘못된 입력에 대한 기능 ID 할당 테스트"""
    with pytest.raises(TypeError):
        assign_featureId(None)
    
    with pytest.raises(TypeError):
        assign_featureId("잘못된 입력")

def test_calculate_priority():
    """우선순위 계산 테스트"""
    # 최소 우선순위 테스트
    assert calculate_priority(30, 5) == 1
    
    # 최대 우선순위 테스트
    assert calculate_priority(0, 1) == 300
    
    # 중간 우선순위 테스트
    priority = calculate_priority(15, 3)
    assert 1 <= priority <= 300

def test_calculate_priority_invalid_input():
    """잘못된 입력에 대한 우선순위 계산 테스트"""
    with pytest.raises(TypeError):
        calculate_priority("30", 5)
    
    with pytest.raises(TypeError):
        calculate_priority(30, "5")
    
    with pytest.raises(ValueError):
        calculate_priority(-1, 5)
    
    with pytest.raises(ValueError):
        calculate_priority(31, 5)
    
    with pytest.raises(ValueError):
        calculate_priority(30, 0)
    
    with pytest.raises(ValueError):
        calculate_priority(30, 6)

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
    
    mock_response = AIMessage(content="""
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
        """)
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm), \
         patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis, \
         patch('feature_specification.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        async def mock_load_redis_side_effect(key):
            if key == "test@example.com":
                return mock_project_data
            elif key == "features:test@example.com":
                return mock_feature_data
            return None
        
        mock_load_redis.side_effect = mock_load_redis_side_effect
        
        result = await create_feature_specification("test@example.com")
        
        assert "features" in result
        assert len(result["features"]) > 0
        #assert result["features"][0]["name"] == "로그인 기능"
        mock_save_redis.assert_called_once()

@pytest.mark.asyncio
async def test_create_feature_specification_no_data():
    """데이터가 없는 경우 테스트"""
    with patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        mock_load_redis.return_value = None
        
        with pytest.raises(ValueError):
            await create_feature_specification("test@example.com")

@pytest.mark.asyncio
async def test_create_feature_specification_redis_save_failure():
    """Redis 저장 실패 테스트"""
    mock_project_data = {
        "projectId": "test-project",
        "startDate": "2024-03-01",
        "endDate": "2024-03-31",
        "members": []
    }
    
    mock_feature_data = ["로그인 기능"]
    
    mock_response = AIMessage(content="""
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
        """)
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm), \
         patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis, \
         patch('feature_specification.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        mock_load_redis.side_effect = [
            mock_project_data,
            mock_feature_data
        ]
        mock_save_redis.side_effect = Exception("Redis 저장 실패")
        
        with pytest.raises(Exception):
            await create_feature_specification("test@example.com")

@pytest.mark.asyncio
async def test_create_feature_specification_gpt_failure():
    """GPT API 호출 실패 테스트"""
    mock_project_data = {
        "projectId": "test-project",
        "startDate": "2024-03-01",
        "endDate": "2024-03-31",
        "members": []
    }
    
    mock_feature_data = ["로그인 기능"]
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("GPT API 호출 실패"))
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm), \
         patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        
        mock_load_redis.side_effect = [
            json.dumps(mock_project_data),
            json.dumps(mock_feature_data)
        ]
        
        with pytest.raises(Exception):
            await create_feature_specification("test@example.com")

@pytest.mark.asyncio
async def test_update_feature_specification():
    """기능 명세서 업데이트 테스트"""
    mock_response = AIMessage(content="""
        {
            "isNextStep": 1,
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
                    "difficulty": 2,
                    "priority": 150
                }
            ]
        }
        """)
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    mock_project_data = {
        "projectId": "test-project",
        "startDate": "2024-03-01",
        "endDate": "2024-03-31",
        "members": []
    }
    
    mock_feature_data = [{
        "_id": "test-feature-id",
        "name": "로그인 기능",
        "useCase": "사용자 로그인",
        "input": "이메일, 비밀번호",
        "output": "로그인 성공/실패",
        "precondition": "회원가입 완료",
        "postcondition": "로그인 상태",
        "startDate": "2024-03-01",
        "endDate": "2024-03-15",
        "difficulty": 2,
        "priority": 150
    }]
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm), \
         patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis, \
         patch('feature_specification.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        async def mock_load_redis_side_effect(key):
            if key == "test@example.com":
                return mock_project_data
            elif key == "features:test@example.com":
                return mock_feature_data
            return None
        
        mock_load_redis.side_effect = mock_load_redis_side_effect
        
        result = await update_feature_specification(
            email="test@example.com",
            feedback="테스트 피드백",
            createdFeatures=[],
            modifiedFeatures=[],
            deletedFeatures=[]
        )
        
        assert "isNextStep" in result
        assert result["isNextStep"] == 1
        assert "features" in result
        mock_save_redis.assert_called_once()

@pytest.mark.asyncio
async def test_update_feature_specification_redis_load_failure():
    """Redis 로드 실패 테스트"""
    with patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        mock_load_redis.side_effect = Exception("Redis 로드 실패")
        
        with pytest.raises(Exception):
            await update_feature_specification(
                email="test@example.com",
                feedback="테스트 피드백",
                createdFeatures=[],
                modifiedFeatures=[],
                deletedFeatures=[]
            )

@pytest.mark.asyncio
async def test_update_feature_specification_redis_save_failure():
    """Redis 저장 실패 테스트"""
    mock_response = AIMessage(content="""
        {
            "isNextStep": 0,
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
                    "difficulty": 2,
                    "priority": 150
                }
            ]
        }
        """)
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    mock_project_data = {
        "projectId": "test-project",
        "startDate": "2024-03-01",
        "endDate": "2024-03-31",
        "members": []
    }
    
    mock_feature_data = [{
        "_id": "test-feature-id",
        "name": "로그인 기능",
        "useCase": "사용자 로그인",
        "input": "이메일, 비밀번호",
        "output": "로그인 성공/실패",
        "precondition": "회원가입 완료",
        "postcondition": "로그인 상태",
        "startDate": "2024-03-01",
        "endDate": "2024-03-15",
        "difficulty": 2,
        "priority": 150
    }]
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm), \
         patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis, \
         patch('feature_specification.save_to_redis', new_callable=AsyncMock) as mock_save_redis:
        
        mock_load_redis.side_effect = [
            mock_project_data,
            mock_feature_data
        ]
        mock_save_redis.side_effect = Exception("Redis 저장 실패")
        
        with pytest.raises(Exception):
            await update_feature_specification(
                email="test@example.com",
                feedback="테스트 피드백",
                createdFeatures=[],
                modifiedFeatures=[],
                deletedFeatures=[]
            )

@pytest.mark.asyncio
async def test_update_feature_specification_gpt_failure():
    """GPT API 호출 실패 테스트"""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("GPT API 호출 실패"))
    
    mock_project_data = {
        "projectId": "test-project",
        "startDate": "2024-03-01",
        "endDate": "2024-03-31",
        "members": []
    }
    
    mock_feature_data = [{
        "_id": "test-feature-id",
        "name": "로그인 기능",
        "useCase": "사용자 로그인",
        "input": "이메일, 비밀번호",
        "output": "로그인 성공/실패",
        "precondition": "회원가입 완료",
        "postcondition": "로그인 상태",
        "startDate": "2024-03-01",
        "endDate": "2024-03-15",
        "difficulty": 2,
        "priority": 150
    }]
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm), \
         patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis:
        
        mock_load_redis.side_effect = [
            mock_project_data,
            mock_feature_data
        ]
        
        with pytest.raises(Exception):
            await update_feature_specification(
                email="test@example.com",
                feedback="테스트 피드백",
                createdFeatures=[],
                modifiedFeatures=[],
                deletedFeatures=[]
            )

@pytest.mark.asyncio
async def test_update_feature_specification_mongodb_failure():
    """MongoDB 저장 실패 테스트"""
    mock_response = AIMessage(content="""
        {
            "isNextStep": 1,
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
                    "difficulty": 2,
                    "priority": 150
                }
            ]
        }
        """)
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    mock_collection = AsyncMock()
    mock_collection.insert_one = AsyncMock(side_effect=Exception("MongoDB 저장 실패"))
    
    mock_project_data = {
        "projectId": "test-project",
        "startDate": "2024-03-01",
        "endDate": "2024-03-31",
        "members": []
    }
    
    mock_feature_data = [{
        "_id": "test-feature-id",
        "name": "로그인 기능",
        "useCase": "사용자 로그인",
        "input": "이메일, 비밀번호",
        "output": "로그인 성공/실패",
        "precondition": "회원가입 완료",
        "postcondition": "로그인 상태",
        "startDate": "2024-03-01",
        "endDate": "2024-03-15",
        "difficulty": 2,
        "priority": 150
    }]
    
    with patch('feature_specification.ChatOpenAI', return_value=mock_llm), \
         patch('feature_specification.load_from_redis', new_callable=AsyncMock) as mock_load_redis, \
         patch('feature_specification.save_to_redis', new_callable=AsyncMock) as mock_save_redis, \
         patch('feature_specification.get_feature_collection', return_value=mock_collection):
        
        mock_load_redis.side_effect = [
            json.dumps(mock_project_data),
            mock_feature_data
        ]
        
        with pytest.raises(Exception):
            await update_feature_specification(
                email="test@example.com",
                feedback="테스트 피드백",
                createdFeatures=[],
                modifiedFeatures=[],
                deletedFeatures=[]
            ) 