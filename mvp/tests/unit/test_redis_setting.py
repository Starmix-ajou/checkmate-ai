import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis_setting import load_from_redis, redis_client, save_to_redis
from redis_setting import test_redis_connection as check_redis_connection


@pytest.fixture(autouse=True)
def mock_redis():
    with patch('redis_setting.redis_client', new_callable=AsyncMock) as mock_client:
        # ping 명령어 mock
        mock_client.ping = AsyncMock(return_value=True)
        # set/get 명령어 mock
        mock_client.set = AsyncMock()
        mock_client.get = AsyncMock()
        yield mock_client

@pytest.mark.asyncio
async def test_redis_connection(mock_redis):
    """Redis 연결 테스트"""
    result = await check_redis_connection()
    assert result is True
    mock_redis.ping.assert_called_once()

@pytest.mark.asyncio
async def test_save_to_redis_dict(mock_redis):
    """Redis에 딕셔너리 저장 테스트"""
    test_data = {"key": "value", "number": 123}
    await save_to_redis("test_key", test_data)
    mock_redis.set.assert_called_once_with("test_key", json.dumps(test_data, ensure_ascii=False))

@pytest.mark.asyncio
async def test_save_to_redis_list(mock_redis):
    """Redis에 리스트 저장 테스트"""
    test_data = [1, 2, 3, "test"]
    await save_to_redis("test_key", test_data)
    mock_redis.set.assert_called_once_with("test_key", json.dumps(test_data, ensure_ascii=False))

@pytest.mark.asyncio
async def test_save_to_redis_string(mock_redis):
    """Redis에 문자열 저장 테스트"""
    test_data = "test string"
    await save_to_redis("test_key", test_data)
    mock_redis.set.assert_called_once_with("test_key", json.dumps(test_data, ensure_ascii=False))

@pytest.mark.asyncio
async def test_save_to_redis_other_type(mock_redis):
    """Redis에 다른 타입 저장 테스트"""
    test_data = 123
    await save_to_redis("test_key", test_data)
    mock_redis.set.assert_called_once_with("test_key", test_data)

@pytest.mark.asyncio
async def test_load_from_redis_json(mock_redis):
    """Redis에서 JSON 데이터 로드 테스트"""
    test_data = {"key": "value", "number": 123}
    mock_redis.get.return_value = json.dumps(test_data)
    
    result = await load_from_redis("test_key")
    assert result == test_data
    mock_redis.get.assert_called_once_with("test_key")

@pytest.mark.asyncio
async def test_load_from_redis_none(mock_redis):
    """Redis에서 없는 데이터 로드 테스트"""
    mock_redis.get.return_value = None
    
    result = await load_from_redis("test_key")
    assert result is None
    mock_redis.get.assert_called_once_with("test_key")

@pytest.mark.asyncio
async def test_load_from_redis_error(mock_redis):
    """Redis 로드 에러 테스트"""
    mock_redis.get.side_effect = Exception("Redis error")
    
    with pytest.raises(Exception) as exc_info:
        await load_from_redis("test_key")
    assert str(exc_info.value) == "Redis error"
    mock_redis.get.assert_called_once_with("test_key") 