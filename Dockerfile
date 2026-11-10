# Use the same Python minor version as training (see model/python_env.yaml)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# 1) Dependencies first: this layer is cached as long as requirements don't change
COPY model/requirements-serving.txt model-requirements.txt
COPY app/requirements.txt app-requirements.txt
RUN pip install -r model-requirements.txt -r app-requirements.txt

# 2) Then the model and the code
COPY model/ ./model/
COPY app/ ./app/

# Don't run as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
