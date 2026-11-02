FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/customer-service-app/backend

COPY customer-service-app/backend/pyproject.toml ./pyproject.toml
COPY customer-service-app/backend/app ./app
COPY customer-service-app/backend/data ./data
COPY customer-service-app/backend/migrations ./migrations
COPY customer-service-app/backend/alembic.ini ./alembic.ini

RUN python -m pip install --upgrade pip && python -m pip install .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
