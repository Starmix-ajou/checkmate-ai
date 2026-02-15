import os
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(dotenv_path=".env")

# ✅ session scope 환경 변수 설정 
@pytest.fixture(scope="session", autouse=True)
def configure_test_env():
    required_vars = ["OPENAI_API_KEY", "MONGODB_URI", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "DB_NAME"]
    for var in required_vars:
        assert os.getenv(var), f"{var} is not set"

# ✅ function별 입력 정의
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


