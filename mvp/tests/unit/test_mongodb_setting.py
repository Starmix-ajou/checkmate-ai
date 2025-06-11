from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mongodb_setting import (collection_is_initialized, db,
                             get_epic_collection, get_feature_collection,
                             get_project_collection, get_task_collection,
                             get_user_collection, init_collections,
                             mongo_client)
from mongodb_setting import test_mongodb_connection as check_mongodb_connection


@pytest.fixture(autouse=True)
def mock_mongo():
    with patch('mongodb_setting.mongo_client', new_callable=AsyncMock) as mock_client, \
         patch('mongodb_setting.db') as mock_db:
        mock_client.admin.command = AsyncMock(return_value={"ok": 1.0})
        mock_db.__getitem__ = AsyncMock()
        yield mock_client, mock_db

@pytest.mark.asyncio
async def test_mongodb_connection(mock_mongo):
    """MongoDB 연결 테스트"""
    mock_client, _ = mock_mongo
    result = await check_mongodb_connection()
    assert result is True
    mock_client.admin.command.assert_called_once_with('ping')

@pytest.mark.asyncio
async def test_get_project_collection(mock_mongo):
    """프로젝트 컬렉션 가져오기 테스트"""
    _, mock_db = mock_mongo
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    collection = await get_project_collection()
    assert collection is not None
    mock_db.__getitem__.assert_called_once_with('projects')

@pytest.mark.asyncio
async def test_get_user_collection(mock_mongo):
    """사용자 컬렉션 가져오기 테스트"""
    _, mock_db = mock_mongo
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    collection = await get_user_collection()
    assert collection is not None
    mock_db.__getitem__.assert_called_once_with('users')

@pytest.mark.asyncio
async def test_get_feature_collection(mock_mongo):
    """기능 컬렉션 가져오기 테스트"""
    _, mock_db = mock_mongo
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    collection = await get_feature_collection()
    assert collection is not None
    mock_db.__getitem__.assert_called_once_with('features')

@pytest.mark.asyncio
async def test_get_epic_collection(mock_mongo):
    """에픽 컬렉션 가져오기 테스트"""
    _, mock_db = mock_mongo
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    collection = await get_epic_collection()
    assert collection is not None
    mock_db.__getitem__.assert_called_once_with('epics')

@pytest.mark.asyncio
async def test_get_task_collection(mock_mongo):
    """태스크 컬렉션 가져오기 테스트"""
    _, mock_db = mock_mongo
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    collection = await get_task_collection()
    assert collection is not None
    mock_db.__getitem__.assert_called_once_with('tasks')

@pytest.mark.asyncio
async def test_init_collections(mock_mongo):
    """컬렉션 초기화 테스트"""
    _, mock_db = mock_mongo
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    collections = await init_collections()
    assert len(collections) == 5
    assert all(collection is not None for collection in collections)
    assert mock_db.__getitem__.call_count == 5

@pytest.mark.asyncio
async def test_collection_is_initialized(mock_mongo):
    """컬렉션 초기화 상태 확인 테스트"""
    _, mock_db = mock_mongo
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    collections = await init_collections()
    assert len(collections) == 5
    assert all(collection is not None for collection in collections)
    result = collection_is_initialized()
    assert result is True