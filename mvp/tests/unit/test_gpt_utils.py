import pytest
from gpt_utils import extract_json_from_gpt_response, remove_comments_safe


def test_extract_json_from_gpt_response_basic():
    """기본적인 JSON 추출 테스트"""
    content = '{"key": "value"}'
    result = extract_json_from_gpt_response(content)
    assert result == {"key": "value"}

def test_extract_json_from_gpt_response_with_code_block():
    """코드 블록이 있는 경우 테스트"""
    content = '```json\n{"key": "value"}\n```'
    result = extract_json_from_gpt_response(content)
    assert result == {"key": "value"}

def test_extract_json_from_gpt_response_with_comments():
    """주석이 있는 경우 테스트"""
    content = '# 주석\n{"key": "value"} # 인라인 주석'
    result = extract_json_from_gpt_response(content)
    assert result == {"key": "value"}

def test_extract_json_from_gpt_response_empty():
    """빈 응답 테스트"""
    with pytest.raises(ValueError):
        extract_json_from_gpt_response("")

def test_remove_comments_safe():
    """주석 제거 함수 테스트"""
    content = '# 주석\n{"key": "value"} # 인라인 주석'
    result = remove_comments_safe(content)
    assert result == '\n{"key": "value"} '

def test_remove_comments_safe_with_string():
    """문자열 내 # 기호가 있는 경우 테스트"""
    content = '{"key": "value#123"}'
    result = remove_comments_safe(content)
    assert result == '{"key": "value#123"}'

# 추가 테스트 케이스들
def test_extract_json_from_gpt_response_with_plain_code_block():
    """일반 코드 블록(```)이 있는 경우 테스트"""
    content = '```\n{"key": "value"}\n```'
    result = extract_json_from_gpt_response(content)
    assert result == {"key": "value"}

def test_extract_json_from_gpt_response_with_text_before_after():
    """JSON 앞뒤에 텍스트가 있는 경우 테스트"""
    content = '이것은 설명입니다. {"key": "value"} 이것은 추가 설명입니다.'
    result = extract_json_from_gpt_response(content)
    assert result == {"key": "value"}

def test_extract_json_from_gpt_response_with_multiple_json():
    """여러 JSON이 있는 경우 첫 번째 JSON만 추출하는지 테스트"""
    content = '{"key1": "value1"} {"key2": "value2"}'
    with pytest.raises(ValueError, match="GPT 응답 파싱 중 오류 발생"):
        extract_json_from_gpt_response(content)

def test_extract_json_from_gpt_response_with_nested_json():
    """중첩된 JSON 구조 테스트"""
    content = '{"key": {"nested": "value"}}'
    result = extract_json_from_gpt_response(content)
    assert result == {"key": {"nested": "value"}}

def test_extract_json_from_gpt_response_with_array():
    """배열이 포함된 JSON 테스트"""
    content = '{"items": [1, 2, 3]}'
    result = extract_json_from_gpt_response(content)
    assert result == {"items": [1, 2, 3]}

def test_extract_json_from_gpt_response_invalid_json():
    """잘못된 JSON 형식 테스트"""
    content = '{"key": "value"'  # 닫는 괄호 누락
    with pytest.raises(ValueError, match="GPT 응답 파싱 중 오류 발생"):
        extract_json_from_gpt_response(content)

def test_remove_comments_safe_with_multiple_comments():
    """여러 줄의 주석이 있는 경우 테스트"""
    content = '# 첫 번째 주석\n# 두 번째 주석\n{"key": "value"}'
    result = remove_comments_safe(content)
    assert result == '\n\n{"key": "value"}'

def test_remove_comments_safe_with_escaped_quotes():
    """이스케이프된 따옴표가 있는 경우 테스트"""
    content = '{"key": "value with \\"quotes\\""}'
    result = remove_comments_safe(content)
    assert result == '{"key": "value with \\"quotes\\""}'

def test_remove_comments_safe_with_multiline_string():
    """여러 줄 문자열이 있는 경우 테스트"""
    content = '{"key": "line1\\nline2"}'
    result = remove_comments_safe(content)
    assert result == '{"key": "line1\\nline2"}' 