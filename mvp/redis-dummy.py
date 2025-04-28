import asyncio
import json
import os

import redis.asyncio as aioredis
from dotenv import load_dotenv

# 1) Redis 서버에 연결
load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

async def initialize_data(r):
    """초기 더미 데이터를 생성합니다. 이미 데이터가 있다면 생성하지 않습니다."""
    # 키 존재 여부 확인
    exists = await r.exists("email:test@test.com")
    if exists:
        print("이미 초기 데이터가 존재합니다.")
        return
    
    # 더미 데이터 생성
    project_data = {
        "email": "test@test.com",
        "members": [
            {
                "name": "jamie",
                "role": "BE"
            },
            {
                "name": "jung",
                "role": "FE"
            }
        ],
        "stacks": ["python", "django", "react", "mysql", "docker", "aws"],
        "startDate": "2025-04-27",
        "endDate": "2025-07-10",
        "description": [
            "사용자 로그인 기능",
            "사용자 회원가입 기능",
            "상품 검색 기능",
            "상품 상세 정보 조회 기능",
            "장바구니 추가 및 관리 기능",
            "주문 및 결제 처리 기능",
            "주문 내역 조회 기능",
            "사용자 리뷰 작성 및 조회 기능",
            "상품 카테고리별 정렬 및 필터링 기능",
            "프로모션 및 할인 코드 적용 기능",
            "사용자 프로필 관리 기능",
            "배송 정보 입력 및 관리 기능",
            "위시리스트 추가 및 조회 기능",
            "재고 관리 기능",
            "고객 지원 및 문의 기능",
            "상품 추천 및 관련 상품 표시 기능",
            "이메일 알림 및 뉴스레터 구독 기능",
            "다국어 지원 기능",
            "반품 및 환불 처리 기능",
            "관리자 대시보드 기능"
        ]
    }

    # JSON 문자열로 직렬화하여 저장
    await r.set(f"email:{project_data['email']}", json.dumps(project_data, ensure_ascii=False))
    print("초기 데이터가 생성되었습니다.")

async def check_data(r):
    """Redis에 저장된 현재 데이터를 조회합니다."""
    # 저장된 값 조회
    raw = await r.get("email:test@test.com")
    if raw:
        data = json.loads(raw)
        print("\n현재 저장된 데이터:")
        print("- 이메일:", data.get("email"))
        print("- 프로젝트:", data.get("project"))
        print("- 기능 목록:")
        
        # description 또는 features 필드 확인
        features = data.get("features", []) or data.get("description", [])
        for feature in features:
            if isinstance(feature, dict):
                print(f"  * {feature.get('name')}: {feature.get('useCase')}")
            else:
                print(f"  * {feature}")
    else:
        print("저장된 데이터가 없습니다.")

    # 현재 DB에 있는 키 목록 출력
    keys = await r.keys("email:*")
    print("\n현재 DB의 email 키:", keys)

async def reset_data(r):
    """Redis의 모든 email: 키를 가진 데이터를 삭제합니다."""
    keys = await r.keys("email:*")
    if keys:
        await r.delete(*keys)
        print("기존 데이터가 모두 삭제되었습니다.")

async def main():
    # Redis 클라이언트 생성
    r = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    # 연결 확인
    try:
        pong = await r.ping()
        print("Redis 연결 성공:", pong)
    except Exception as e:
        print("Redis 연결 실패:", e)
        return
    
    # 데이터 초기화
    await reset_data(r)

    # 초기 데이터 생성 (없을 경우에만)
    await initialize_data(r)
    
    # 데이터 조회
    await check_data(r)
    
    # 연결 종료
    await r.aclose()

if __name__ == "__main__":
    asyncio.run(main())
