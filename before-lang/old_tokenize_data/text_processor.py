import re
from typing import List, Union

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# NLTK 데이터 다운로드 (한국어 문장 분리를 위해)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리합니다.
    
    Args:
        text (str): 분리할 텍스트
        
    Returns:
        List[str]: 문장 리스트
    """
    # 기본적인 문장 분리 (마침표, 물음표, 느낌표로 끝나는 문장)
    sentences = re.split(r'[.!?]+', text)
    
    # 빈 문장 제거 및 공백 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def split_into_paragraphs(text: str) -> List[str]:
    """
    텍스트를 단락 단위로 분리합니다.
    각 단락은 제목과 내용을 포함하며, 빈 줄로 구분됩니다.
    
    Args:
        text (str): 분리할 텍스트
        
    Returns:
        List[str]: 단락 리스트
    """
    # 빈 줄을 기준으로 텍스트를 분리
    raw_paragraphs = text.split('\n\n')
    
    # 빈 단락 제거 및 공백 제거
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    
    # 제목과 내용을 포함하는 단락으로 처리
    processed_paragraphs = []
    current_title = None
    current_content = []
    
    for paragraph in raw_paragraphs:
        # 한 줄이고 다음에 빈 줄이 있다면 제목으로 간주
        if len(paragraph.split('\n')) == 1 and not paragraph.endswith('.'):
            if current_title and current_content:
                # 이전 단락 저장
                full_paragraph = f"{current_title} {' '.join(current_content)}"
                processed_paragraphs.append(full_paragraph)
            current_title = paragraph
            current_content = []
        else:
            current_content.append(paragraph)
    
    # 마지막 단락 처리
    if current_title and current_content:
        full_paragraph = f"{current_title} {' '.join(current_content)}"
        processed_paragraphs.append(full_paragraph)
    
    return processed_paragraphs

def process_text_file(file_path: str, split_by: str = 'sentence') -> List[str]:
    """
    텍스트 파일을 읽어서 문장 또는 단락으로 분리합니다.
    
    Args:
        file_path (str): 텍스트 파일 경로
        split_by (str): 분리 기준 ('sentence' 또는 'paragraph')
        
    Returns:
        List[str]: 분리된 텍스트 리스트
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
        if split_by == 'sentence':
            return split_into_sentences(text)
        elif split_by == 'paragraph':
            return split_into_paragraphs(text)
        else:
            raise ValueError("split_by는 'sentence' 또는 'paragraph'여야 합니다.")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        raise Exception(f"파일 처리 중 오류 발생: {str(e)}")

def save_preprocessed_text(texts: List[str], output_file: str = 'preprocessed_text.txt'):
    """
    전처리된 텍스트를 새로운 파일에 저장합니다.
    
    Args:
        texts (List[str]): 저장할 텍스트 리스트
        output_file (str): 출력 파일 경로
    """
    try:
        # 텍스트를 파일에 저장
        with open(output_file, 'w', encoding='utf-8') as file:
            for i, text in enumerate(texts, 1):
                file.write(f"{i}. {text}\n\n")
            
        print(f"전처리된 텍스트가 {output_file}에 성공적으로 저장되었습니다.")
        
    except Exception as e:
        raise Exception(f"파일 저장 중 오류 발생: {str(e)}")

# 사용 예시
if __name__ == "__main__":
    # 텍스트 파일 처리 예시
    file_path = "input.txt"  # 처리할 텍스트 파일 경로
    
    try:
        print("\n=== 텍스트 처리 결과 ===")
        
        # 문장 단위로 분리
        print("\n1. 문장 단위 분리:")
        sentences = process_text_file(file_path, split_by='sentence')
        print(f"총 문장 수: {len(sentences)}")
        print("\n처음 3개 문장:")
        for i, sentence in enumerate(sentences[:3], 1):
            print(f"{i}. {sentence}")
        
        # 단락 단위로 분리
        print("\n2. 단락 단위 분리:")
        paragraphs = process_text_file(file_path, split_by='paragraph')
        print(f"총 단락 수: {len(paragraphs)}")
        print("\n처음 2개 단락:")
        for i, paragraph in enumerate(paragraphs[:2], 1):
            print(f"\n단락 {i}:")
            print(paragraph)
        
        # 전처리된 텍스트 저장
        print("\n3. 전처리된 텍스트 저장:")
        save_preprocessed_text(sentences, 'preprocessed_sentences.txt')
        save_preprocessed_text(paragraphs, 'preprocessed_paragraphs.txt')
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}") 