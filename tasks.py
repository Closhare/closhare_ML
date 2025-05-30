# ========================= tasks.py =========================
"""Celery tasks for embedding + tagging + Pinecone upsert."""

import os
import uuid
from io import BytesIO
from typing import List, Dict, Tuple

import httpx
from PIL import Image
from celery import Celery
from pinecone import Pinecone
from dotenv import load_dotenv

from utils.onnx_clip import fclip
from utils.labels import fashion_tags_kr
import asyncio 
from typing import List, Optional
# from fastapi import logger

# ────────────────────────────────────────────────
# ─────────────── Celery & Pinecone ──────────────
# ────────────────────────────────────────────────
load_dotenv()

BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")  # ← fallback 도 localhost
celery_app = Celery("tasks", broker=BROKER_URL, backend=BROKER_URL)

load_dotenv()
api_key = os.getenv("API_KEY")
pc = Pinecone(api_key=api_key, environment="us-east-1")
index = pc.Index("closhare")  # 반드시 vector 차원이 맞아야 함

INDEX_NAME = "closhare"
TAG_INDEX_NAME = "closhare-tags"

index = pc.Index(INDEX_NAME)
tag_index = pc.Index(TAG_INDEX_NAME)

CATEGORIES = ["Clothing Types", "Fashion Styles", "Materials", "Colors", "Seasons"]

# ────────────────────────────────────────────────
# ────────────────  공통 유틸들  ───────────────────
# ────────────────────────────────────────────────
HTTP_TIMEOUT = 10          # 한 곳에 모아두면 조정 쉽다


async def _fetch_image(url: str) -> Image.Image:
    """URL → PIL Image (RGB). httpx 비동기 사용."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(str(url)) 
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")


def _encode_image_to_vec(image: Image.Image):
    """CLIP inference → 1-D numpy array."""
    return fclip.encode_images([image])[0]


def _auto_tag(vector, *, top_k_style: int = 2) -> List[str]:
    """Pinecone에서 유사 태그 추출 후 한글 매핑."""
    tags: List[str] = []
    for cat in CATEGORIES:
        k = top_k_style if cat == "Fashion Styles" else 1
        qr = tag_index.query(
            vector=vector.tolist(),
            top_k=k,
            include_metadata=True,
            filter={"category": {"$eq": cat}},
        )
        for m in qr.get("matches", []):
            raw = m["metadata"].get("tag")
            if raw in fashion_tags_kr:
                tags.append(fashion_tags_kr[raw])
    return tags


def _upsert_to_index(product_id: int, vector, metadata: Dict):
    """Pinecone upsert 래퍼."""
    index.upsert(
        [
            {
                "id": str(product_id),
                "values": vector.tolist(),
                "metadata": metadata,
            }
        ]
    )


# ────────────────────────────────────────────────
# ─────────────── Celery Tasks  ──────────────────
# ────────────────────────────────────────────────
# ─────────────────────── 헬퍼 함수들 ───────────────────────────────
HTTP_TIMEOUT = 10

async def _fetch_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")

def _encode_image_to_vec(image: Image.Image):
    return fclip.encode_images([image])[0]

def _auto_tag(vector, *, top_k_style: int = 2) -> List[str]:
    tags: List[str] = []
    for cat in CATEGORIES:
        k = top_k_style if cat == "Fashion Styles" else 1
        qr = tag_index.query(
            vector=vector.tolist(),
            top_k=k,
            include_metadata=True,
            filter={"category": {"$eq": cat}},
        )
        for m in qr.get("matches", []):
            raw = m["metadata"].get("tag")
            if raw in fashion_tags_kr:
                tags.append(fashion_tags_kr[raw])
    return tags

def _upsert(product_id: int, vector, metadata: Dict):
    index.upsert([{"id": str(product_id), "values": vector.tolist(), "metadata": metadata}])

# ─────────────────────── Celery Task (핵심) ───────────────────────
# @celery_app.task(
#     name="embed_and_tag",
#     bind=True,
#     autoretry_for=(httpx.HTTPError, Exception),
#     max_retries=3,
#     retry_backoff=True,
# )
# def embed_and_tag(self, product_id: int, img_url: str, tags: List[dict] | None = None):
#     try:
#         image = asyncio.run(_fetch_image(img_url))
#         vector = _encode_image_to_vec(image)
#         auto_tags = _auto_tag(vector)

#         metadata = {
#             "tags": auto_tags,
#             "imgUrl": img_url,
#             "categories": [t["category"] for t in (tags or [])],
#         }
#         resp = index.upsert([{"id": str(product_id), "values": vector.tolist(), "metadata": metadata}])

#         return {"status": "success", "tags": auto_tags}
#     except Exception as exc:
#         raise self.retry(exc=exc, countdown=5)
    
@celery_app.task(
    name="embed_only",
    bind=True,
    autoretry_for=(httpx.HTTPError, Exception),
    max_retries=3,
    retry_backoff=True,
)
def embed_only(product_id: int, img_url: str, tags: List[dict]):
    image = asyncio.run(_fetch_image(img_url))
    vector = _encode_image_to_vec(image)

    metadata = {
        "tags": [t["tag"] for t in tags],
        "imgUrl": img_url,
        "categories": [t["category"] for t in tags],
    }

    index.upsert([{
        "id": str(product_id),
        "values": vector.tolist(),
        "metadata": metadata,
    }])
    return {"status": "success", "tags": tags}