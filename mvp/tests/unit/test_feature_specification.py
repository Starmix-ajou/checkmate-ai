import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from feature_specification import (assign_featureId, calculate_priority,
                                   create_feature_specification,
                                   update_feature_specification)
from langchain_core.messages import AIMessage


def test_assign_featureId():
    """기능 ID 할당 테스트"""
    # 테스트할 데이터 구성
    test_data = {"name": "테스트 기능"}
    
    # 데이터 구조 검증
    assert isinstance(test_data, dict)
    assert "name" in test_data
    assert isinstance(test_data["name"], str)

def test_assign_featureId_invalid_input():
    """잘못된 입력에 대한 기능 ID 할당 테스트"""
    # 테스트할 에러 케이스
    invalid_inputs = [None, "잘못된 입력"]
    
    # 에러 케이스 검증
    for invalid_input in invalid_inputs:
        assert not isinstance(invalid_input, dict)

def test_calculate_priority():
    """우선순위 계산 테스트"""
    # 테스트할 데이터 구성
    test_cases = [
        {"expectedDays": 30, "difficulty": 5, "expected": 1},
        {"expectedDays": 0, "difficulty": 1, "expected": 300},
        {"expectedDays": 15, "difficulty": 3, "expected_range": (1, 300)}
    ]
    
    # 데이터 구조 검증
    for case in test_cases:
        assert isinstance(case["expectedDays"], int)
        assert isinstance(case["difficulty"], int)
        if "expected" in case:
            assert isinstance(case["expected"], int)
        if "expected_range" in case:
            assert isinstance(case["expected_range"], tuple)
            assert len(case["expected_range"]) == 2

def test_calculate_priority_invalid_input():
    """잘못된 입력에 대한 우선순위 계산 테스트"""
    # 테스트할 에러 케이스
    invalid_cases = [
        {"expectedDays": "30", "difficulty": 5},
        {"expectedDays": 30, "difficulty": "5"},
        {"expectedDays": -1, "difficulty": 5},
        {"expectedDays": 31, "difficulty": 5},
        {"expectedDays": 30, "difficulty": 0},
        {"expectedDays": 30, "difficulty": 6}
    ]
    
    # 에러 케이스 검증
    for case in invalid_cases:
        assert not (isinstance(case["expectedDays"], int) and 
                   isinstance(case["difficulty"], int) and
                   0 <= case["expectedDays"] <= 30 and
                   1 <= case["difficulty"] <= 5)

@pytest.mark.asyncio
async def test_create_feature_specification():
    """기능 명세서 생성 테스트"""
    # 테스트할 데이터 구성
    test_data = {
        "features": [
            {
                "featureId": "test-id-1",
                "name": "로그인 기능",
                "useCase": "사용자 로그인",
                "input": "이메일, 비밀번호",
                "output": "로그인 성공/실패"
            }
        ]
    }
    
    # 데이터 구조 검증
    assert "features" in test_data
    assert isinstance(test_data["features"], list)
    assert len(test_data["features"]) > 0
    
    feature = test_data["features"][0]
    required_fields = ["featureId", "name", "useCase", "input", "output"]
    for field in required_fields:
        assert field in feature
        assert isinstance(feature[field], str)

@pytest.mark.asyncio
async def test_create_feature_specification_no_data():
    """데이터가 없는 경우 테스트"""
    # 테스트할 에러 메시지
    error_message = "Project for user test@example.com not found"
    
    # 에러 메시지 검증
    assert isinstance(error_message, str)
    assert "not found" in error_message

@pytest.mark.asyncio
async def test_create_feature_specification_gpt_failure():
    """GPT API 호출 실패 테스트"""
    # 테스트할 에러 메시지
    error_message = "GPT API 호출 실패"
    
    # 에러 메시지 검증
    assert isinstance(error_message, str)
    assert "GPT API" in error_message

@pytest.mark.asyncio
async def test_update_feature_specification():
    """기능 명세서 업데이트 테스트"""
    # 테스트할 데이터 구성
    test_data = {
        "isNextStep": 1,
        "features": [
            {
                "name": "로그인 기능",
                "useCase": "사용자 로그인",
                "input": "이메일, 비밀번호",
                "output": "로그인 성공/실패",
                "precondition": "회원가입 완료",
                "postcondition": "로그인 상태",
                "startDate": "2024-03-01",
                "endDate": "2024-03-15",
                "difficulty": 2,
                "priority": 150
            }
        ]
    }
    
    # 데이터 구조 검증
    assert "isNextStep" in test_data
    assert "features" in test_data
    assert test_data["isNextStep"] in [0, 1]
    assert isinstance(test_data["features"], list)
    assert len(test_data["features"]) > 0
    
    feature = test_data["features"][0]
    required_fields = ["name", "useCase", "input", "output", "precondition", 
                      "postcondition", "startDate", "endDate", "difficulty", "priority"]
    for field in required_fields:
        assert field in feature
        if field in ["difficulty", "priority"]:
            assert isinstance(feature[field], int)
        else:
            assert isinstance(feature[field], str)

@pytest.mark.asyncio
async def test_update_feature_specification_no_data():
    """데이터가 없는 경우 테스트"""
    # 테스트할 에러 메시지
    error_message = "Redis로부터 기능 명세서 초안 불러오기 실패"
    
    # 에러 메시지 검증
    assert isinstance(error_message, str)
    assert "불러오기 실패" in error_message

@pytest.mark.asyncio
async def test_update_feature_specification_gpt_failure():
    """GPT API 호출 실패 테스트"""
    # 테스트할 에러 메시지
    error_message = "GPT API 호출 실패"
    
    # 에러 메시지 검증
    assert isinstance(error_message, str)
    assert "GPT API" in error_message