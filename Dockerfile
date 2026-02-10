# 베이스 이미지: Kaniko의 안정성을 위해 docker.io를 명시합니다.
FROM docker.io/library/python:3.11-slim

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 의존성 설치 (캐시 레이어 최적화)
COPY ./mvp/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY ./mvp /app

# 포트 개방
EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
