# ========================= main.py =========================

from fastapi import FastAPI, BackgroundTasks, HTTPException, status, Body, Request
from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from typing import List
import os, logging, requests
from io import BytesIO
from PIL import Image
from pinecone import Pinecone
from utils.onnx_clip import fclip
from utils.labels import fashion_tags_kr, fashion_tags_en
# from tasks import embed_and_tag, embed_and_tag_sync
from pydantic import BaseModel, Field, ConfigDict, HttpUrl
from typing import List, Optional
from celery.result import AsyncResult          # ⚡ NEW
from tasks import _encode_image_to_vec, _auto_tag, _upsert_to_index, _fetch_image
from tasks import embed_and_tag                # ⚡ NEW ─ Celery task 호출용
from fastapi import Request
import traceback  
import json
import re

app = FastAPI(title="CloShare ML API", version="2.0.0")  # ⚡ 버전업
logger = logging.getLogger("uvicorn.error")

pc = Pinecone(api_key=os.getenv("API_KEY"), environment="us-east-1")
index = pc.Index("closhare")
tag_index = pc.Index("closhare-tags")

# --------------------------- Routes ---------------------------
@app.get("/healthz")
def health():
    return {"status": "ok"}


# ----------- [1] 이미지 업로드 → 작업만 큐에 넣고 202 반환 ----------- ⚡
@app.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(request: Request):
    try:
        headers = dict(request.headers)
        raw_body = await request.body()
        print(f"[DEBUG] Headers: {headers}")
        print(f"[DEBUG] Raw Body: {raw_body}")
        data = json.loads(raw_body)
        print(f"Received data: {data}")
        img_url = data.get("imgUrl")
        tag_data = data.get("tags")
        product_id = data.get("productId")

        if not img_url or not tag_data or not product_id:
            raise HTTPException(status_code=400, detail="imgUrl, tags, or productId missing")

        # 여기에 비동기 태스크 큐 enqueue (예: Celery)
        task = embed_and_tag.delay(product_id, img_url, tag_data)

        return {"status": "queued", "task_id": task.id}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"task_enqueue_failed: {type(e).__name__} - {e}")
    
# ----------- [2] 작업 상태/결과 조회 -------------------------------- ⚡
@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    try:
        res = AsyncResult(task_id)
        if res.state == "SUCCESS":
            return {"status": "done", "result": res.result}
        elif res.state == "FAILURE":
            return {"status": "failed", "error": str(res.info)}
        else:
            return {"status": res.state.lower()}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": f"{type(e).__name__} - {e}"}

# ----------------------------- /tags --------------------------  # ✅ 
@app.post("/tags")
async def tags(request: Request):
    try:
        headers = dict(request.headers)
        raw_body = await request.body()
        print(f"[DEBUG] Headers: {headers}")
        print(f"[DEBUG] Raw Body: {raw_body}")
        data = json.loads(raw_body)
        print(f"Received data: {data}")
        img_url = data.get("img_url")

        if not img_url:
            raise HTTPException(status_code=400, detail="Missing img_url")

        resp = requests.get(img_url, timeout=10)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        vec = fclip.encode_images([image])[0]

        categories = ["Clothing Types", "Fashion Styles", "Materials", "Colors", "Seasons"]
        tag_results: List[str] = []
        for cat in categories:
            tk = 2 if cat == "Fashion Styles" else 1
            qr = tag_index.query(vector=vec.tolist(), top_k=tk, include_metadata=True, filter={"category": {"$eq": cat}})
            for m in qr.get("matches", []):
                raw = m["metadata"].get("tag")
                if raw.strip() in fashion_tags_kr:
                    tag_results.append(fashion_tags_kr[raw])

        return {"status": "success", "results": {"tags": tag_results}}

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"tagging_failed: {type(exc).__name__} - {exc}")


# ----------------------------- /search ------------------------ 
@app.post("/search")
async def search(request: Request):
    try:
        headers = dict(request.headers)
        raw_body = await request.body()
        print(f"[DEBUG] Headers: {headers}")
        print(f"[DEBUG] Raw Body: {raw_body}")
        data = json.loads(raw_body)
        print(f"Received data: {data}")
        query = data.get("query")
        top_k = data.get("top_k", 15)

        if not query:
            return {"status": "400", "code": "missing_query", "results": []}

        vec = fclip.encode_text([query])[0]
        qr = index.query(vector=vec.tolist(), top_k=top_k, include_metadata=False)
        ids = [int(m["id"]) for m in qr.get("matches", []) if m.get("id", "").isdigit()]
        print(f"Search query: {query}, Top K: {top_k}, Results: {ids}")

        return {"status": "200", "code": "ok", "results": ids}

    except Exception as exc:
        traceback.print_exc()
        return {"status": "500", "code": "ml_error", "error": f"{type(exc).__name__} - {exc}", "results": []}

# ------------------------ /search-by-image -------------------- 
@app.post("/search-by-image")
async def search_by_image(request: Request):
    try:
        headers = dict(request.headers)
        raw_body = await request.body()
        print(f"[DEBUG] Headers: {headers}")
        print(f"[DEBUG] Raw Body: {raw_body}")
        data = json.loads(raw_body)
        print(f"Received data: {data}")
        image_url = data.get("image_url")
        top_k = data.get("top_k", 15)

        if not image_url:
            return {"status": "400", "code": "missing_image_url", "results": []}

        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        vec = fclip.encode_images([image])[0]

        qr = index.query(vector=vec.tolist(), top_k=top_k, include_metadata=False)
        ids = [int(m["id"]) for m in qr.get("matches", []) if m.get("id", "").isdigit()]

        return {"status": "200", "code": "ok", "results": ids}

    except Exception as exc:
        traceback.print_exc()
        return {"status": "500", "code": "ml_error", "error": f"{type(exc).__name__} - {exc}", "results": []}

# ----------------------- /search-recommend -------------------- 
@app.post("/search-recommend")
async def search_recommend(request: Request):
    try:
        headers = dict(request.headers)
        raw_body = await request.body()
        print(f"[DEBUG] Headers: {headers}")
        print(f"[DEBUG] Raw Body: {raw_body}")
        data = json.loads(raw_body)
        print(f"Received data: {data}")
        query = data.get("query")  # 예: "봄/스웨터/캐주얼"
        top_k = data.get("top_k", 15)

        if not query:
            return {"status": "400", "code": "missing_query", "results": []}

        kw_kr = [k.strip() for k in query.split("/") if k.strip()]
        kw_en = [fashion_tags_en.get(k, "") for k in kw_kr]
        kw_en = [k for k in kw_en if k]

        if not kw_en:
            return {"status": "400", "code": "no_valid_tags", "results": []}

        query_text = " ".join(kw_en)
        vec = fclip.encode_text([query_text])[0]
        qr = index.query(vector=vec.tolist(), top_k=top_k, include_metadata=False)
        ids = [int(m["id"]) for m in qr.get("matches", []) if m.get("id", "").isdigit()]
        print(f"Recommend query: {query}, Top K: {top_k}, Results: {ids}")

        return {"status": "200", "code": "ok", "results": ids}

    except Exception as exc:
        traceback.print_exc()
        return {"status": "500", "code": "recommend_error", "error": f"{type(exc).__name__} - {exc}", "results": []}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("FashionCLIP model loaded successfully.")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
