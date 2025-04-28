FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY ./mvp /app/mvp
CMD ["uvicorn", "mvp.serve:app", "--host", "0.0.0.0", "--port", "8000"]