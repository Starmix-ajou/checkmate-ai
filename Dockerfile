# 베이스 이미지
FROM python:3.11-slim

WORKDIR /app
# 소스 복사
COPY ./mvp /app/mvp
COPY ./requirements.txt /app/

# 패키지 설치
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# PYTHONPATH 설정
ENV PYTHONPATH=/app

CMD ["uvicorn", "mvp.serve:app", "--host", "0.0.0.0", "--port", "8000"]