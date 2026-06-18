FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/application

COPY requirements-train.txt setup.py ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-train.txt

COPY . .

ENTRYPOINT ["python", "-m", "src.models.train_model"]
