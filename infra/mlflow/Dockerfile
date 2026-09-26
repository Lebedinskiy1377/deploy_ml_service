FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt constraints.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY entrypoint.py /usr/local/bin/mlflow-entrypoint.py

EXPOSE 5000

# Server options come from MLFLOW_* environment variables (see docker-compose.yml).
ENTRYPOINT ["python", "/usr/local/bin/mlflow-entrypoint.py"]
CMD ["mlflow", "server"]
