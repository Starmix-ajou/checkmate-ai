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