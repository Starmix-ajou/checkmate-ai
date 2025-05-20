import logging
from typing import List, Tuple

from motor.motor_asyncio import AsyncIOMotorCollection

logger = logging.getLogger(__name__)

async def get_project_members(
    project_id: str,
    project_collection: AsyncIOMotorCollection,
    user_collection: AsyncIOMotorCollection
) -> List[Tuple[str, str]]:
    """
    프로젝트의 멤버 정보를 가져옵니다.
    
    Args:
        project_id (str): 프로젝트 ID
        project_collection (AsyncIOMotorCollection): 프로젝트 컬렉션
        user_collection (AsyncIOMotorCollection): 사용자 컬렉션
        
    Returns:
        List[Tuple[str, str]]: [(멤버 이름, 포지션 문자열), ...] 형태의 리스트
        
    Raises:
        Exception: 프로젝트를 찾을 수 없거나 멤버 정보가 없는 경우
    """
    project_members = []
    try:
        project_data = await project_collection.find_one({"_id": project_id})
        if not project_data:
            logger.error(f"projectId {project_id}에 해당하는 프로젝트를 찾을 수 없습니다.")
            raise Exception(f"projectId {project_id}에 해당하는 프로젝트를 찾을 수 없습니다.")
        
        logger.info(f"프로젝트 데이터: {project_data}")
        
        members = project_data.get("members", [])
        assert len(members) > 0, "members가 없습니다."
        
        for member_ref in members:
            try:
                user_id = member_ref.id
                user_info = await user_collection.find_one({"_id": user_id})
                if not user_info:
                    logger.warning(f"⚠️ 사용자 정보를 찾을 수 없습니다: {user_id}")
                    continue
                
                name = user_info.get("name")
                assert name is not None, "name이 없습니다."
                profiles = user_info.get("profiles", [])
                assert len(profiles) > 0, "profile이 없습니다."
                for profile in profiles:
                    if profile.get("projectId") == project_id:
                        logger.info(f">> projectId가 일치하는 profile이 존재함: {name}")
                        positions = profile.get("positions", [])
                        assert len(positions) > 0, "position이 없습니다."
                        positions_str = ", ".join(positions)
                        member_info = (name, positions_str)
                        project_members.append(member_info)
                        logger.info(f"추가된 멤버: {name}, {positions}")
            except Exception as e:
                logger.error(f"멤버 정보 처리 중 오류 발생: {str(e)}", exc_info=True)
                continue
    except Exception as e:
        logger.error(f"MongoDB에서 Project 정보 로드 중 오류 발생: {e}", exc_info=True)
        raise e
    
    logger.info(f"📌 project_members: {project_members}")
    assert len(project_members) > 0, "project_members가 비어있습니다."
    
    return project_members 