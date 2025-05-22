import asyncio
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# 최상위 디렉토리의 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# MongoDB 연결 설정
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME')

mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[DB_NAME]

# MOCK_MONGODB_URI = "mongodb://localhost:27017"
# MOCK_DB_NAME = "checkmate"

# mock_mongo_client = AsyncIOMotorClient(MOCK_MONGODB_URI)
# mock_db = mock_mongo_client[MOCK_DB_NAME]

async def test_mongodb_connection():
    #logger.info(f"MongoDB 연결 설정: uri={MONGODB_URI}, db={DB_NAME}")
    #logger.info(f"Mock MongoDB 연결 설정: uri={MOCK_MONGODB_URI}, db={MOCK_DB_NAME}")
    try:
        pong = await mongo_client.admin.command('ping')
        logger.info(f"MongoDB 연결 성공: {pong}")
        return True
    except Exception as e:
        logger.error(f"MongoDB 연결 실패: {e}")
        raise e
    
async def get_feature_collection():
    logger.info(f"🔍 get_feature_collection 호출 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return db['features']

async def get_epic_collection():
    logger.info(f"🔍 get_epic_collection 호출 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return db['epics']

async def get_task_collection():
    logger.info(f"🔍 get_task_collection 호출 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return db['tasks']

async def get_project_collection():
    logger.info(f"🔍 get_project_collection 호출 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return db['projects']

async def get_user_collection():
    logger.info(f"🔍 get_user_collection 호출 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return db['users']


if __name__ == "__main__":
    asyncio.run(test_mongodb_connection())