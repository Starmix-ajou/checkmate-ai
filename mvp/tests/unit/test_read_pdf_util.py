import io
from unittest.mock import AsyncMock, MagicMock, patch

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
    expected = "Hello World!"
    assert clean_text(input_text) == expected

def test_clean_text_with_parentheses():
    """괄호가 있는 텍스트 정리 테스트"""
    input_text = "( Hello ) World ( Test )"
    expected = "(Hello) World (Test)"
    assert clean_text(input_text) == expected

def test_clean_text_sentence_end():
    """문장 끝 공백 제거 테스트"""
    input_text = "Hello World ! Test . End !"
    expected = "Hello World! Test. End!"
    assert clean_text(input_text) == expected

def test_clean_text_multiple_newlines():
    """여러 줄바꿈 처리 테스트"""
    input_text = "Hello\n\n\n\nWorld\n\n\n\n!"
    expected = "Hello World!"
    assert clean_text(input_text) == expected

@pytest.mark.asyncio
async def test_extract_pdf_text_success():
    """PDF 텍스트 추출 성공 테스트"""
    # 유효한 PDF 파일의 기본 구조
    mock_pdf_content = b'''%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test Content) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000056 00000 n
0000000111 00000 n
0000000212 00000 n
0000000256 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
364
%%EOF'''
    
    # PDF 리더 모의
    mock_pdf_reader = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Test Content"
    mock_pdf_reader.pages = [mock_page1]
    
    # HTTP 응답 모의
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value.status = 200
    mock_response.__aenter__.return_value.content.read = AsyncMock(return_value=mock_pdf_content)
    
    with patch('aiohttp.ClientSession.get', return_value=mock_response), \
         patch('PyPDF2.PdfReader', return_value=mock_pdf_reader), \
         patch('aiofiles.open', MagicMock()):
        
        from read_pdf_util import extract_pdf_text
        result = await extract_pdf_text("http://test.com/test.pdf")
        
        # 결과 형식 검증
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.endswith(".pdf") is False  # PDF 확장자가 결과에 포함되지 않아야 함

@pytest.mark.asyncio
async def test_extract_pdf_text_invalid_pdf():
    """잘못된 PDF 파일 테스트"""
    # HTTP 응답 모의
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value.status = 200
    mock_response.__aenter__.return_value.content.read = AsyncMock(return_value=b'Invalid PDF content')
    
    with patch('aiohttp.ClientSession.get', return_value=mock_response):
        from read_pdf_util import extract_pdf_text
        with pytest.raises(Exception, match="유효한 PDF 파일이 아닙니다"):
            await extract_pdf_text("http://test.com/test.pdf")

@pytest.mark.asyncio
async def test_extract_pdf_text_download_failure():
    """PDF 다운로드 실패 테스트"""
    # HTTP 응답 모의
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value.status = 404
    
    with patch('aiohttp.ClientSession.get', return_value=mock_response):
        from read_pdf_util import extract_pdf_text
        with pytest.raises(Exception, match="기능 정의서 다운로드 및 변환 중 오류 발생"):
            await extract_pdf_text("http://test.com/test.pdf")

@pytest.mark.asyncio
async def test_extract_pdf_text_empty_file():
    """빈 PDF 파일 테스트"""
    # HTTP 응답 모의
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value.status = 200
    mock_response.__aenter__.return_value.content.read = AsyncMock(return_value=b'')
    
    with patch('aiohttp.ClientSession.get', return_value=mock_response):
        from read_pdf_util import extract_pdf_text
        with pytest.raises(Exception, match="다운로드된 PDF 파일이 비어있습니다"):
            await extract_pdf_text("http://test.com/test.pdf")

@pytest.mark.asyncio
async def test_extract_pdf_text_save_failure():
    """텍스트 파일 저장 실패 테스트"""
    # 유효한 PDF 파일의 기본 구조
    mock_pdf_content = b'''%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test Content) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000056 00000 n
0000000111 00000 n
0000000212 00000 n
0000000256 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
364
%%EOF'''
    
    # PDF 리더 모의
    mock_pdf_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Test Content"
    mock_pdf_reader.pages = [mock_page]
    
    # HTTP 응답 모의
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value.status = 200
    mock_response.__aenter__.return_value.content.read = AsyncMock(return_value=mock_pdf_content)
    
    with patch('aiohttp.ClientSession.get', return_value=mock_response), \
         patch('PyPDF2.PdfReader', return_value=mock_pdf_reader), \
         patch('aiofiles.open', MagicMock(side_effect=Exception("파일 저장 실패"))):
        
        from read_pdf_util import extract_pdf_text
        with pytest.raises(Exception, match="기능 정의서 다운로드 및 변환 중 오류 발생"):
            await extract_pdf_text("http://test.com/test.pdf") 