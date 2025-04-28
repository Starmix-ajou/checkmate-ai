# 베이스 이미지
FROM  python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 소스 복사
COPY ./src /app

# 패키지 설치
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Fast API 서버 실행
CMD ["uvicorn", "integrated_serve.app", "--host", "0.0.0.0", "--port", "8000"]