import os
from unittest.mock import patch

import mongomock
import pytest

#from dotenv import load_dotenv

# 환경 변수 로드
#load_dotenv()
os.environ["OPENAI_API_KEY"] = "sk-proj-1234567890"
#os.environ["DB_NAME"] = "test_db"
#os.environ["REDIS_HOST"] = "localhost"
#os.environ["REDIS_PORT"] = "6379"
#os.environ["REDIS_PASSWORD"] = "123456000"

#@pytest.fixture(scope="session")
#def test_env():
#    """테스트 환경 설정을 위한 fixture"""
#    return {
#        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
#        "MONGODB_URI": os.getenv("MONGODB_URI"),
#        "REDIS_URL": os.getenv("REDIS_URL")
#    }

@pytest.fixture(scope="function")
def mock_openai():
    """OpenAI API 모킹을 위한 fixture"""
    with patch("openai.ChatCompletion.create") as mock_create, \
         patch("openai.ChatCompletion.acreate") as mock_acreate:
        
        # 동기 호출 모킹
        mock_create.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a mock response"
                    }
                }
            ]
        }
        
        # 비동기 호출 모킹
        mock_acreate.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a mock response"
                    }
                }
            ]
        }
        
        yield mock_create, mock_acreate

#@pytest.fixture(autouse=True)
#def mock_mongodb_client():
#    """MongoDB 모킹을 위한 fixture"""
#    mock_client = mongomock.MongoClient()
#    with patch("mongodb_setting.mongo_client", mock_client):
#        with patch("mongodb_setting.db", mock_client["test_db"]):
#            yield