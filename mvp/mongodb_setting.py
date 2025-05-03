import asyncio
import logging
import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# 최상위 디렉토리의 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# MongoDB 연결 설정
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME')

logger.info(f"MongoDB 연결 설정: uri={MONGODB_URI}, db={DB_NAME}")

mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[DB_NAME]

async def test_mongodb_connection():
    try:
        pong = await mongo_client.admin.command('ping')
        print(f"MongoDB 연결 성공: {pong}")
        logger.info(f"MongoDB 연결 성공: {pong}")
        return True
    except Exception as e:
        print(f"MongoDB 연결 실패: {e}")
        logger.error(f"MongoDB 연결 실패: {e}")
        raise e
    
async def get_feature_collection():
    print("get_feature_collection 호출")
    return db['features']

async def get_epic_collection():
    print("get_epic_collection 호출")
    return db['epics']

async def get_task_collection():
    print("get_task_collection 호출")
    return db['tasks']

if __name__ == "__main__":
    print(MONGODB_URI)
    print(DB_NAME)
    asyncio.run(test_mongodb_connection())