FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.13.0 boto3 psycopg2-binary

RUN mkdir -p /app/artifacts/mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0"]
