# ============================ Dockerfile ============================
# 베이스 이미지는 슬림 + PIP 캐시 제거
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    C_FORCE_ROOT=1          # ↔ Celery root 실행 허용

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 기본 실행은 FastAPI 서버.
# → docker-compose 에서 worker 서비스만 command 로 덮어씌움
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
