FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    C_FORCE_ROOT=1        

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* \
 && pip install gdown \
 && mkdir -p models \
 && gdown --folder --id 17tnfnHUU4UlTM4CFuom1A5cF8AeLHdWe -O models

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--http", "h11"]

