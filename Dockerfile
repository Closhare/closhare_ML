# ============================ Dockerfile ============================
# 베이스 이미지는 슬림 + PIP 캐시 제거
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    C_FORCE_ROOT=1          # ↔ Celery root 실행 허용

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models \
 && gdown --folder --id 17tnfnHUU4UlTM4CFuom1A5cF8AeLHdWe -O models

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
