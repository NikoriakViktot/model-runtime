FROM python:3.11-slim

# 1️⃣ Системні залежності (мінімум)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

# 2️⃣ Робоча директорія
WORKDIR /app

# 3️⃣ Встановлюємо Python залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Копіюємо код
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY control_plane ./control_plane

# 5️⃣ PYTHONPATH
ENV PYTHONPATH=/app

# 6️⃣ Запуск API
CMD ["uvicorn", "control_plane.main:app", "--host", "0.0.0.0", "--port", "8004"]
