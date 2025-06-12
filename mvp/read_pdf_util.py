import asyncio
import io
import logging
import os
import re

import aiofiles
import aiohttp
from PyPDF2 import PdfReader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    PDF에서 추출된 텍스트를 정리합니다.
    
    Args:
        text (str): 원본 텍스트
        
    Returns:
        str: 정리된 텍스트
    """
    # 연속된 공백을 하나로 통일
    text = re.sub(r'\s+', ' ', text)
    
    # 불필요한 공백 제거
    text = text.strip()
    
    # 문장 끝에 있는 공백 제거
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    # 괄호 안의 공백 정리
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # 줄바꿈 정리
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text

async def extract_pdf_text(predefined_definition: str) -> str:
    """
    PDF 파일에서 텍스트를 추출하는 함수를 테스트합니다.
    
    Args:
        pdf_path (str): PDF 파일 경로
        
    Returns:
        str: 추출된 텍스트
    """
    try:
        asset_dir=os.path.join(os.path.dirname(__file__), "asset")
        os.makedirs(asset_dir, exist_ok=True)
        
        filename=os.path.basename(predefined_definition)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(predefined_definition) as response:
                if response.status == 200:
                    logger.info(f"✅ 기능 정의서 URL 접근 성공")
                    # PDF 데이터를 바이트로 읽기
                    pdf_bytes = await response.content.read()
                    logger.info(f"✅ 기능 정의서 bytes로 읽기 성공 (크기: {len(pdf_bytes)} bytes)")
                    
                    if len(pdf_bytes) == 0:
                        raise ValueError("다운로드된 PDF 파일이 비어있습니다.")
                    
                    # PDF를 텍스트로 변환
                    try:
                        pdf_file = io.BytesIO(pdf_bytes)
                        logger.info(f"✅ io.BytesIO 생성 성공")
                        
                        # PDF 파일 유효성 검사
                        if not pdf_file.read(5).startswith(b'%PDF-'):
                            raise ValueError("유효한 PDF 파일이 아닙니다.")
                        pdf_file.seek(0)  # 파일 포인터를 다시 처음으로
                        
                        pdf_reader = PdfReader(pdf_file)
                        logger.info(f"✅ PdfReader 생성 성공 (페이지 수: {len(pdf_reader.pages)})")
                        
                        text_content = ""
                        for i, page in enumerate(pdf_reader.pages, 1):
                            page_text = page.extract_text()
                            if page_text:
                                # 각 페이지의 텍스트를 정리
                                cleaned_text = clean_text(page_text)
                                text_content += cleaned_text + "\n\n"
                            logger.info(f"✅ 페이지 {i} 텍스트 추출 및 정리 완료")
                        
                        # 전체 텍스트 정리
                        text_content = clean_text(text_content)
                        logger.info(f"✅ 전체 텍스트 추출 및 정리 완료 (길이: {len(text_content)} 문자)")
                        
                        # 텍스트 파일로 저장
                        text_filename = os.path.splitext(filename)[0] + ".txt"
                        text_file_path = os.path.join(asset_dir, text_filename)
                        
                        async with aiofiles.open(text_file_path, 'w', encoding='utf-8') as f:
                            await f.write(text_content)
                        logger.info(f"✅ 텍스트 파일 저장 성공: {text_file_path}")
                        return text_content
                        
                    except Exception as e:
                        logger.error(f"PDF 처리 중 오류 발생: {str(e)}", exc_info=True)
                        raise Exception(f"PDF 처리 중 오류 발생: {str(e)}") from e
                else:
                    logger.error(f"기능 정의서 다운로드 실패: HTTP {response.status}")
                    raise Exception(f"기능 정의서 다운로드 실패: HTTP {response.status}")
    except Exception as e:
        logger.error(f"기능 정의서 다운로드 및 변환 중 오류 발생: {str(e)}", exc_info=True)
        raise Exception(f"기능 정의서 다운로드 및 변환 중 오류 발생: {str(e)}") from e