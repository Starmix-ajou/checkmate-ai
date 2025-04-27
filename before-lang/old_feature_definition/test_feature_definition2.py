import asyncio
import json
import sys
from typing import Tuple

from FeatureDefinition import FeatureRequest, generate_definition

# 테스트 입력들
test_inputs = [
    {"name": "Test 1: Minimal", "content": "기능 이름: 회원 가입"},
    {"name": "Test 2: With description", "content": "기능 이름: 댓글 작성\n설명: 사용자가 게시글에 댓글을 남길 수 있어야 해요."},
    {"name": "Test 3: With purpose", "content": "기능 이름: 할 일 추가\n설명: 사용자가 해야 할 일을 등록할 수 있어야 해요.\n목적: 작업을 목록으로 정리해서 관리하기 위해 필요해요."},
    {"name": "Test 4: With use case", "content": "기능 이름: 이메일 알림 설정\n설명: 사용자가 특정 이벤트에 대해 이메일 알림을 받을 수 있어야 해요.\n목적: 중요한 이벤트를 놓치지 않도록 알림을 받아야 하기 때문이에요.\n시나리오: 사용자가 설정 페이지에 들어가서 이메일 알림을 켜면, 이후 특정 조건이 충족되면 이메일을 받아요."},
    {"name": "Test 5: Nearly complete", "content": "기능 이름: 비밀번호 재설정\n설명: 사용자가 비밀번호를 잊었을 경우 이메일을 통해 재설정할 수 있어야 해요.\n목적: 사용자 계정 접근이 불가능할 경우를 대비하기 위함이에요.\n시나리오: 로그인 화면에서 '비밀번호를 잊으셨나요?'를 클릭한 후 이메일을 입력하면, 인증 메일을 받아서 새 비밀번호를 설정할 수 있어요.\n입력값: 이메일 주소, 새 비밀번호\n출력값: 비밀번호 변경 성공 메시지\n전제 조건: 사용자가 이미 계정을 가지고 있어야 함"}
]

# 수정 테스트 입력들
mod_test_inputs = [
    {"name": "MOD 1: Minimal", "content": "기능 이름을 회원 가입에서 서비스 회원 등록으로 변경"},
    {"name": "MOD 2: With description", "content": "게시글에 남긴 댓글에 수정 댓글과 답글을 추가할 수 있어야 해요."},
    {"name": "MOD 3: With purpose", "content": "목록으로 정리할 때 목록을 가나다 순으로 정렬해야 더 보기 쉬울 것 같아요."},
    {"name": "MOD 4: With use case", "content": "사용자가 이메일을 받는 특정 조건은 D-Day가 되었을 때 입니다."},
    {"name": "MOD 5: Nearly complete", "content": "비밀번호 찾기/재설정 외에도 로그인 이메일을 추가하거나 변경할 수 있는 기능도 넣고 싶어요."}
]

async def test_creation(input_text: str) -> Tuple[bool, str, str]:
    """
    API를 호출하여 새로운 기능 정의서를 생성하고 결과를 확인합니다.
    """
    try:
        # API 요청 생성
        request = FeatureRequest(user_input=input_text)
        
        # API 호출
        response = await generate_definition(request)
        
        # 응답 검증
        if response and response.definition:
            return True, json.dumps(response.definition.model_dump(), ensure_ascii=False, indent=2), response.feature_id
        return False, "API 응답이 올바르지 않습니다.", ""
        
    except Exception as e:
        return False, f"테스트 실패: {str(e)}", ""

async def test_modification(input_text: str, feature_id: str) -> Tuple[bool, str]:
    """
    API를 호출하여 기존 기능 정의서를 수정하고 결과를 확인합니다.
    """
    try:
        # API 요청 생성
        request = FeatureRequest(user_input=input_text, feature_id=feature_id)
        
        # API 호출
        response = await generate_definition(request)
        
        # 응답 검증
        if response and response.definition:
            return True, json.dumps(response.definition.model_dump(), ensure_ascii=False, indent=2)
        return False, "API 응답이 올바르지 않습니다."
        
    except Exception as e:
        return False, f"테스트 실패: {str(e)}"

async def run_all_tests():
    """
    모든 테스트 케이스를 실행하고 결과를 출력합니다.
    """
    print("=== 기능 정의서 생성 테스트 시작 ===\n")
    
    for test, mod_test in zip(test_inputs, mod_test_inputs):
        print(f" {test['name']}")
        print(f"입력: {test['content']}\n")
        
        # 생성 테스트 실행
        print("1. 초안 생성 테스트")
        success, output, feature_id = await test_creation(test["content"])
        
        if success:
            print("PASS - 생성 API 호출 성공")
        else:
            print(f"FAIL - {output}")
        
        print("\n생성된 기능 정의서 초안:")
        print(output)
        
        # 수정 테스트 실행
        if success and feature_id:
            print("\n2. 초안 수정 테스트")
            print(f"수정 내용 입력: {mod_test['content']}\n")
            mod_success, mod_output = await test_modification(mod_test["content"], feature_id)
            
            if mod_success:
                print("PASS - 수정 API 호출 성공")
            else:
                print(f"FAIL - {mod_output}")
            
            print("\n수정된 기능 정의서:")
            print(mod_output)
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    # 명령줄 인수 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 테스트 모드 실행
        asyncio.run(run_all_tests())
    else:
        print("테스트를 실행하려면 --test 옵션을 사용하세요.")