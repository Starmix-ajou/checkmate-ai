import os

import pytest
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

@pytest.fixture(scope="session")
def test_env():
    """테스트 환경 설정을 위한 fixture"""
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "MONGODB_URI": os.getenv("MONGODB_URI"),
        "REDIS_URL": os.getenv("REDIS_URL")
    }

@pytest.fixture(scope="function")
def mock_gpt_response():
    """GPT 응답을 모킹하기 위한 fixture"""
    return {
        "choices": [
            {
                "message": {
                    "content": "테스트 응답입니다."
                }
            }
        ]
    } 