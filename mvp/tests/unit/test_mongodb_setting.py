from unittest.mock import MagicMock, patch

import pytest

from mvp.mongodb_setting import (get_feature_collection,
                                 get_project_collection, get_user_collection)


@pytest.fixture
def mock_mongo_client():
    with patch('mongodb_setting.AsyncIOMotorClient') as mock:
        yield mock

def test_get_project_collection(mock_mongo_client):
    """프로젝트 컬렉션 가져오기 테스트"""
    mock_collection = MagicMock()
    mock_mongo_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection
    
    collection = get_project_collection()
    assert collection is not None
    mock_mongo_client.return_value.__getitem__.return_value.__getitem__.assert_called_once_with('projects')

def test_get_user_collection(mock_mongo_client):
    """사용자 컬렉션 가져오기 테스트"""
    mock_collection = MagicMock()
    mock_mongo_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection
    
    collection = get_user_collection()
    assert collection is not None
    mock_mongo_client.return_value.__getitem__.return_value.__getitem__.assert_called_once_with('users')

def test_get_feature_collection(mock_mongo_client):
    """기능 컬렉션 가져오기 테스트"""
    mock_collection = MagicMock()
    mock_mongo_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection
    
    collection = get_feature_collection()
    assert collection is not None
    mock_mongo_client.return_value.__getitem__.return_value.__getitem__.assert_called_once_with('features')