import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Union

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Redis 연결 설정
REDIS_HOST = os.getenv('REDIS_HOST') or ("localhost")
REDIS_PORT = int(os.getenv('REDIS_PORT')) or (6379)
REDIS_PWD = os.getenv('REDIS_PASSWORD') or ("123456000")
if not isinstance(REDIS_HOST, str):
    raise ValueError(f"REDIS_HOST must be a string: {REDIS_HOST}")
if not isinstance(REDIS_PORT, int):
    raise ValueError(f"REDIS_PORT must be an integer: {REDIS_PORT}")
if not isinstance(REDIS_PWD, str):
    raise ValueError(f"REDIS_PWD must be a string: {REDIS_PWD}")

#logger.info(f"Redis 연결 설정: host={REDIS_HOST}, port={REDIS_PORT}, password={'*' * len(REDIS_PWD) if REDIS_PWD else None}")

redis_client = aioredis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PWD,
    decode_responses=True,
    socket_timeout=3600,
    socket_connect_timeout=3600,
)

async def test_redis_connection():
    try:
        pong = await redis_client.ping()
        print(f"Redis 연결 테스트 성공: {pong}")
        logger.info(f"Redis 연결 테스트 성공: {pong}")
        return True
    except Exception as e:
        print(f"Redis 연결 실패: {str(e)}")
        logger.error(f"Redis 연결 실패: {str(e)}")
        raise e

async def save_to_redis(key: str, data: Any):
    logger.info(f"🔍 Redis 데이터 저장 호출 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # 딕셔너리를 JSON 문자열로 직렬화
        if isinstance(data, dict) or isinstance(data, list) or isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        else:
            data = data
            
        await redis_client.set(key, data)
        logger.info(f"✅ Redis에 데이터 저장 성공: {key}")
    except Exception as e:
        logger.error(f"❌ Redis 저장 중 오류 발생: {str(e)}")
        raise e

async def load_from_redis(key: str) -> Any:
    logger.info(f"🔍 Redis 데이터 로드 호출 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        data = await redis_client.get(key)
        if data:
            # 이미 JSON으로 파싱된 데이터인지 확인
            if isinstance(data, dict) or isinstance(data, list) or isinstance(data, str):
                return json.loads(data)
            else:
                return data
        return None
    except Exception as e:
        logger.error(f"❌ Redis 로드 중 오류 발생: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(test_redis_connection())