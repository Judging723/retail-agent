FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY knowledge-pipeline ./knowledge-pipeline
COPY customer-service-app/backend/data/knowledge.json ./customer-service-app/backend/data/knowledge.json

RUN python -m pip install --upgrade pip && python -m pip install ./knowledge-pipeline

WORKDIR /workspace/knowledge-pipeline

CMD ["python", "scripts/index_milvus.py"]
