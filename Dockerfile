# 베이스 이미지
FROM python:3.11-slim

WORKDIR /mvp
# 소스 복사
COPY ./mvp /mvp
COPY ./mvp/requirements.txt /mvp

# 패키지 설치
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# PYTHONPATH 설정
ENV PYTHONPATH=/mvp

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]