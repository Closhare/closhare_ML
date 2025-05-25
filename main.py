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

app = FastAPI(title="CloShare ML API", version="2.0.0")  # ⚡ 버전업
logger = logging.getLogger("uvicorn.error")

pc = Pinecone(api_key=os.getenv("API_KEY"), environment="us-east-1")
index = pc.Index("closhare")
tag_index = pc.Index("closhare-tags")

# --------------------------- Schemas ---------------------------

# class TagItem(BaseModel):
#     tag: str
#     category: str

# class UploadReq(BaseModel):
#     product_id: int = Field(..., alias="productId")
#     img_url: HttpUrl = Field(..., alias="imgUrl")
#     tags: Optional[List[TagItem]] = None
    
#     class Config:
#          model_config = ConfigDict(populate_by_name=True) # ✅ 이게 없으면 alias만 보고 필드명을 무시함

# class SearchReq(BaseModel):
#     query: str
#     top_k: int = 15

# class SearchImgReq(BaseModel):
#     image_url: str
#     top_k: int = 15

# class TagReq(BaseModel):
#     img_url: str = Field(..., alias="img_url")

# class SearchRecReq(BaseModel):
#     query: str
#     top_k: int = 15

# --------------------------- Routes ---------------------------
@app.get("/healthz")
def health():
    return {"status": "ok"}

# ---------- sync upload (embed + tag + upsert) --------------- 
# @app.post("/upload", status_code=200)
# async def upload(req: UploadReq):
#     """
#     이미지 1장 업로드 → 임베딩 · 태깅 · Pinecone 업서트까지
#     모두 직접 처리해서 즉시 완료 결과 반환.
#     """
#     try:
#         # tag_payload = (
#         #     [t.dict(by_alias=True) for t in req.tags]
#         #     if req.tags else None
#         # )

#         print(f">>> SYNC TASK START for product_id={req.product_id}")
#         image = await _fetch_image(str(req.img_url)) 
#         vector = _encode_image_to_vec(image)

#         auto_tags = _auto_tag(vector)
#         print(f">>> AUTO TAGS: {auto_tags}")

#         metadata = {
#             "tag_inputs": [t.tag for t in req.tags] if req.tags else [],
#             "categories": [t.category for t in req.tags] if req.tags else [],
#             "imgUrl": str(req.img_url)
#         }

#         print(f">>> UPSERTING to Pinecone for product_id={req.product_id}")
#         _upsert_to_index(req.product_id, vector, metadata)

#         return {"status": "success", "tags": auto_tags}
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"ML 태깅 실패: {e}")

# ----------- [1] 이미지 업로드 → 작업만 큐에 넣고 202 반환 ----------- ⚡
@app.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(request: Request):
    body = await request.json()
    try:
        product_id = body.get("productId")
        img_url = body.get("imgUrl")
        tags = body.get("tags")

        if not product_id or not img_url:
            raise HTTPException(status_code=400, detail="Missing productId or imgUrl")

        tags_payload = tags if tags else None
        task = embed_and_tag.delay(product_id, str(img_url), tags_payload)
        return {"status": "queued", "task_id": task.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"task_enqueue_failed: {e}")

# ----------- [2] 작업 상태/결과 조회 -------------------------------- ⚡
@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Celery 작업 상태 확인:
    PENDING → 실행 대기
    STARTED → 실행 중
    SUCCESS → 완료(결과 포함)
    FAILURE → 실패(에러 포함)
    """
    res = AsyncResult(task_id)
    if res.state == "SUCCESS":
        return {"status": "done", "result": res.result}
    elif res.state == "FAILURE":
        return {"status": "failed", "error": str(res.info)}
    else:
        return {"status": res.state.lower()}

# ----------------------------- /tags --------------------------  # ✅
@app.post("/tags")
async def tags(request: Request):
    try:
        body = await request.json()
        img_url = body.get("img_url")

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
        logger.error(f"/tags failed: {exc}")
        raise HTTPException(status_code=500, detail="tagging_failed")

# ----------------------------- /search ------------------------ 
@app.post("/search")
async def search(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        top_k = body.get("top_k", 15)

        if not query:
            return {"status": "400", "code": "missing_query", "results": []}

        vec = fclip.encode_text([query])[0]
        qr = index.query(vector=vec.tolist(), top_k=top_k, include_metadata=False)
        ids = [int(m["id"]) for m in qr.get("matches", []) if m.get("id", "").isdigit()]

        return {"status": "200", "code": "ok", "results": ids}

    except Exception as exc:
        logger.error(f"/search error: {exc}")
        return {"status": "500", "code": "ml_error", "results": []}

# ------------------------ /search-by-image -------------------- 
@app.post("/search-by-image")
async def search_by_image(request: Request):
    try:
        body = await request.json()
        image_url = body.get("image_url")
        top_k = body.get("top_k", 15)

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
        logger.error(f"/search-by-image error: {exc}")
        return {"status": "500", "code": "ml_error", "results": []}

# ----------------------- /search-recommend -------------------- 
@app.post("/search-recommend")
async def search_recommend(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        top_k = body.get("top_k", 15)

        if not query:
            return {"status": "400", "code": "missing_query", "results": []}

        kw_kr = [k.strip() for k in query.split("/") if k.strip()]
        kw_en = [fashion_tags_en.get(k, "") for k in kw_kr]
        kw_en = [k for k in kw_en if k]

        if not kw_en:
            return {"status": "400", "code": "no_valid_tags", "results": []}

        query_text = " ".join(kw_en)
        logger.info(f"query_kr: {query}, query_en: {query_text}")

        vec = fclip.encode_text([query_text])[0]
        qr = index.query(vector=vec.tolist(), top_k=top_k, include_metadata=False)
        ids = [int(m["id"]) for m in qr.get("matches", []) if m.get("id", "").isdigit()]

        return {"status": "200", "code": "ok", "results": ids}

    except Exception as exc:
        logger.error(f"/search-recommend error: {exc}")
        return {"status": "500", "code": "recommend_error", "results": []}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("FashionCLIP model loaded successfully.")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
