# Використовуємо легкий базовий образ (без CUDA)
FROM python:3.10-slim

# Змінні середовища для оптимізації Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Встановлюємо curl для healthcheck (перевірки працездатності)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Спочатку копіюємо залежності (для кешування Docker шарів)
COPY requirements.txt .

# Встановлюємо бібліотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код додатку
COPY . .

# Відкриваємо порт
EXPOSE 8502

# Healthcheck: Docker буде знати, чи живий ваш Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health || exit 1

# Запуск
ENTRYPOINT ["streamlit", "run", "ui/app_2.py"]