import io
from unittest.mock import MagicMock, patch

import pytest
from read_pdf_util import clean_text


def test_clean_text_basic():
    """기본적인 텍스트 정리 테스트"""
    input_text = "  Hello   World  !  "
    expected = "Hello World!"
    assert clean_text(input_text) == expected

def test_clean_text_with_newlines():
    """줄바꿈이 있는 텍스트 정리 테스트"""
    input_text = "Hello\n\n\nWorld\n\n!"
    expected = "Hello\n\nWorld\n\n!"
    assert clean_text(input_text) == expected

def test_clean_text_with_parentheses():
    """괄호가 있는 텍스트 정리 테스트"""
    input_text = "( Hello ) World ( Test )"
    expected = "(Hello) World (Test)"
    assert clean_text(input_text) == expected

@pytest.mark.asyncio
async def test_extract_pdf_text_success():
    """PDF 텍스트 추출 성공 테스트"""
    mock_pdf_content = b'%PDF-1.4\n...'  # 유효한 PDF 헤더
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.content.read = MagicMock(return_value=mock_pdf_content)
    
    mock_pdf_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Test PDF Content"
    mock_pdf_reader.pages = [mock_page]
    
    with patch('aiohttp.ClientSession') as mock_session, \
         patch('PyPDF2.PdfReader', return_value=mock_pdf_reader), \
         patch('aiofiles.open', MagicMock()):
        
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
        
        from read_pdf_util import extract_pdf_text
        result = await extract_pdf_text("http://test.com/test.pdf")
        assert "Test PDF Content" in result

@pytest.mark.asyncio
async def test_extract_pdf_text_invalid_pdf():
    """잘못된 PDF 파일 테스트"""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.content.read = MagicMock(return_value=b'Invalid PDF content')
    
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
        
        from read_pdf_util import extract_pdf_text
        with pytest.raises(ValueError, match="유효한 PDF 파일이 아닙니다"):
            await extract_pdf_text("http://test.com/test.pdf")

@pytest.mark.asyncio
async def test_extract_pdf_text_download_failure():
    """PDF 다운로드 실패 테스트"""
    mock_response = MagicMock()
    mock_response.status = 404
    
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
        
        from read_pdf_util import extract_pdf_text
        with pytest.raises(Exception, match="기능 정의서 다운로드 실패"):
            await extract_pdf_text("http://test.com/test.pdf") 