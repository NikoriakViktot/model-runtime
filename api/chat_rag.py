import json
import requests
import logging
from typing import List

LLM_API_BASE = "http://litellm:4000"
logger = logging.getLogger(__name__)


def extract_triggers(prompt: str, model_name: str) -> List[str]:
    """
    Використовує ту саму модель, що і основний чат, для витягування слів.
    model_name: це model_alias, який повернув MRM (наприклад, 'qwen-7b-instruct')
    """
    system = (
        "Extract important keywords, entities, or topics from the user message. "
        "Return ONLY a JSON array of strings. Do not explain."
    )

    payload = {
        "model": model_name,  # <--- ВИКОРИСТОВУЄМО АКТИВНУ МОДЕЛЬ
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,  # Мінімум фантазії, максимум точності
        "max_tokens": 100,
        # Ми НЕ передаємо сюди extra_body з LoRA.
        # Це означає, що запит піде до чистої базової моделі, що ідеально для логіки.
    }

    try:
        r = requests.post(
            f"{LLM_API_BASE}/v1/chat/completions",
            json=payload,
            timeout=10
        )

        r.raise_for_status()

        content = r.json()["choices"][0]["message"]["content"]

        # Очистка від можливих markdown блоків ```json ... ```
        clean_content = content.replace("```json", "").replace("```", "").strip()

        return json.loads(clean_content)
    except Exception as e:
        logger.warning(f"Failed to extract triggers via {model_name}: {e}")
        # Fallback: повертаємо пустий список, щоб не ламати чат
        return []