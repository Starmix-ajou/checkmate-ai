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

async def test_mongodb_connection():
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

def collection_is_initialized():
    collections = {
        "feature_collection": feature_collection,
        "project_collection": project_collection,
        "epic_collection": epic_collection,
        "task_collection": task_collection,
        "user_collection": user_collection
    }
    
    uninitialized_collections = []
    for name, collection in collections.items():
        if collection is None:
            uninitialized_collections.append(name)
    
    if len(uninitialized_collections) > 0:
        raise ValueError(f"다음의 collection들이 초기화되지 않았습니다: {uninitialized_collections}")
    
    logger.info("✅ 모든 collection이 정상적으로 초기화되었습니다.")
    return True

# db 초기화 함수
async def init_collections():
    global feature_collection, project_collection, epic_collection, task_collection, user_collection
    feature_collection = None
    project_collection = None
    epic_collection = None
    task_collection = None
    user_collection = None
    
    feature_collection = await get_feature_collection()
    project_collection = await get_project_collection()
    epic_collection = await get_epic_collection()
    task_collection = await get_task_collection()
    user_collection = await get_user_collection()
    
    if not collection_is_initialized():
        raise False
    return feature_collection, project_collection, epic_collection, task_collection, user_collection

if __name__ == "__main__":
    asyncio.run(test_mongodb_connection())