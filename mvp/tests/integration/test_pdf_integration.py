import os

import aiohttp
import pytest
from read_pdf_util import clean_text, extract_pdf_text


@pytest.mark.asyncio
async def test_pdf_processing_integration():
    """PDF 처리 통합 테스트"""
    # 테스트용 PDF URL (실제 존재하는 PDF URL로 변경 필요)
    test_pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        # PDF 텍스트 추출
        extracted_text = await extract_pdf_text(test_pdf_url)
        
        # 기본 검증
        assert extracted_text is not None
        assert len(extracted_text) > 0
        
        # 텍스트 정리 검증
        cleaned_text = clean_text(extracted_text)
        assert cleaned_text is not None
        assert len(cleaned_text) > 0
        
        # 파일 저장 검증
        asset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "asset")
        text_files = [f for f in os.listdir(asset_dir) if f.endswith('.txt')]
        assert len(text_files) > 0
        
    except Exception as e:
        pytest.fail(f"통합 테스트 실패: {str(e)}")

@pytest.mark.asyncio
async def test_pdf_processing_with_invalid_url():
    """잘못된 URL로 PDF 처리 테스트"""
    invalid_url = "https://example.com/nonexistent.pdf"
    
    with pytest.raises(Exception):
        await extract_pdf_text(invalid_url)

@pytest.mark.asyncio
async def test_pdf_processing_with_large_file():
    """대용량 PDF 파일 처리 테스트"""
    # 대용량 PDF URL (실제 존재하는 대용량 PDF URL로 변경 필요)
    large_pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        extracted_text = await extract_pdf_text(large_pdf_url)
        assert extracted_text is not None
        assert len(extracted_text) > 0
    except Exception as e:
        pytest.fail(f"대용량 PDF 처리 테스트 실패: {str(e)}") 