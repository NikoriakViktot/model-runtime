FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -U pip

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY mrm /app/mrm

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "mrm.app:app", "--host", "0.0.0.0", "--port", "8010"]
