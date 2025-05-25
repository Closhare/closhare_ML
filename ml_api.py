from flask import Flask, request, jsonify
import torch
from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image
import numpy as np
import io
# import sqlite3
# import base64
# from cryptography.fernet import Fernet
from flask_cors import CORS
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import requests
from io import BytesIO
import json
import uuid

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()
api_key = os.getenv("API_KEY")
pc = Pinecone(api_key=api_key, environment="us-east-1")
index = pc.Index("closhare")  # 반드시 vector 차원이 맞아야 함
tag_index = pc.Index("closhare-tags")  # 태그용 인덱스

# Load the FashionCLIP model
fclip = FashionCLIP("fashion-clip")

# Fashion-related tags for similarity matching
# fashion_tags = {
#     "Clothing Types": ["T-shirt", "Hoodie", "Sweatshirt", "Jacket", "Coat", "Blazer", "Vest", "Sweater", "Cardigan", 
#                        "Jeans", "Shorts", "Skirt", "Dress", "Jumpsuit", "Overalls", 
#                        "Leggings", "Trousers", "Pants", "Shirt", "Blouse", "Tank top", "Crop top", 
#                        "Belt", "Scarf", "Tie", "Hat", "Cap", "Beanie", "Gloves", "Socks", "Tights", 
#                        "Underwear", "Lingerie", "Pajamas", "Sleepwear", "Swimwear", "Bikini", "Cover-up", "Rash guard",
#                        "Sneakers", "Running shoes", "Loafers", "Sandals", "Boots", "Heels",
#                        "Backpack", "Tote bag", "Clutch bag", "Crossbody bag", "Wallet", "Sunglasses"],
#     "Fashion Styles": ["Casual", "Streetwear", "Minimal", "Sporty", "Formal", "Business casual", "Chic", "Bohemian", 
#                        "Vintage", "Retro", "Preppy", "Grunge", "Gothic", "Punk", "Athleisure", 
#                        "Classic", "Eclectic", "Artsy", "Edgy", "Romantic", "Feminine",
#                        "Masculine", "Androgynous", "Avant-garde", "Luxury", "High fashion", "Urban", "Cozy", "Relaxed",
#                        "Sophisticated", "Elegant", "Playful", "Bold", "Subdued", "Timeless", "Transitional",
#                           "Resort", "Cruise", "Festival", "Beachwear", "Loungewear", "Activewear", "Athletic",
#                           "Outdoor", "Travel", "Weekend", "Date night", "Cocktail", "Evening", "Party", "Bridal",
#                           "Gala", "Red carpet", "Black tie", "Business formal", "Smart casual"],
#     "Materials": ["Cotton", "Denim", "Leather", "Silk", "Wool", "Cashmere", "Linen", "Polyester", "Nylon",
#                   "Acrylic", "Rayon", "Viscose", "Spandex", "Bamboo", "Tencel", "Modal", "Faux fur",
#                   "Satin", "Velvet", "Corduroy", "Canvas", "Taffeta", "Chiffon", "Organza",
#                   "Jersey", "Sweatshirt fabric", "Fleece", "Terry cloth", "Neoprene", "Mesh", "Sheer",
#                   "Transparent", "Ribbed", "Quilted", "Puff print", "Brocade", "Jacquard", "Tartan" ],
#     "Colors": ["Black", "White", "Gray", "Beige", "Red", "Blue", "Green", "Yellow", "Pink", "Purple", "Brown", 
#                "Orange", "Turquoise", "Navy", "Olive", "Burgundy", "Teal", "Coral", "Lavender", "Mustard", "Cream",
#                 "Maroon", "Charcoal", "Ivory", "Peach", "Mint", "Aqua", "Magenta", "Cyan", "Indigo", "Fuchsia",
#                 "Emerald", "Sapphire", "Ruby", "Amber", "Amber", "Copper", "Bronze", "Silver", "Gold", "Platinum",
#                 "Rose gold", "Champagne", "Slate", "Dusty rose", "Seafoam", "Lilac", "Berry", "Rust", "Terracotta" ], 
#     "Seasons": ["Spring", "Summer", "Fall", "Winter"],
# }

fashion_tags_kr = {
    # Clothing Types
    "T-shirt": "티셔츠",
    "Hoodie": "후디",
    "Sweatshirt": "스웨트셔츠",
    "Jacket": "자켓",
    "Coat": "코트",
    "Blazer": "블레이저",
    "Vest": "베스트",
    "Sweater": "스웨터",
    "Cardigan": "가디건",
    "Jeans": "청바지",
    "Shorts": "반바지",
    "Skirt": "치마",
    "Dress": "드레스",
    "Jumpsuit": "점프수트",
    "Overalls": "오버롤",
    "Leggings": "레깅스",
    "Trousers": "슬랙스",
    "Pants": "바지",
    "Shirt": "셔츠",
    "Blouse": "블라우스",
    "Tank top": "민소매",
    "Crop top": "크롭탑",
    "Belt": "벨트",
    "Scarf": "스카프",
    "Tie": "넥타이",
    "Hat": "모자",
    "Cap": "캡",
    "Beanie": "비니",
    "Gloves": "장갑",
    "Socks": "양말",
    "Tights": "타이츠",
    "Underwear": "속옷",
    "Lingerie": "란제리",
    "Pajamas": "파자마",
    "Sleepwear": "잠옷",
    "Swimwear": "수영복",
    "Bikini": "비키니",
    "Cover-up": "커버업",
    "Rash guard": "래시가드",
    "Sneakers": "스니커즈",
    "Running shoes": "러닝화",
    "Loafers": "로퍼",
    "Sandals": "샌들",
    "Boots": "부츠",
    "Heels": "하이힐",
    "Backpack": "백팩",
    "Tote bag": "토트백",
    "Clutch bag": "클러치백",
    "Crossbody bag": "크로스백",
    "Wallet": "지갑",
    "Sunglasses": "선글라스",

    # Fashion Styles
    "Casual": "캐주얼",
    "Streetwear": "스트리트웨어",
    "Minimal": "미니멀",
    "Sporty": "스포티",
    "Formal": "포멀",
    "Business casual": "비즈니스 캐주얼",
    "Chic": "시크",
    "Bohemian": "보헤미안",
    "Vintage": "빈티지",
    "Retro": "레트로",
    "Preppy": "프레피",
    "Grunge": "그런지",
    "Gothic": "고딕",
    "Punk": "펑크",
    "Athleisure": "애슬레저",
    "Classic": "클래식",
    "Eclectic": "에클레틱",
    "Artsy": "아트시",
    "Edgy": "엣지",
    "Romantic": "로맨틱",
    "Feminine": "페미닌",
    "Masculine": "매스큘린",
    "Androgynous": "앤드로지너스",
    "Avant-garde": "아방가르드",
    "Luxury": "럭셔리",
    "High fashion": "하이패션",
    "Urban": "어반",
    "Cozy": "코지",
    "Sophisticated": "세련된",
    "Elegant": "우아한",
    "Playful": "장난기 있는",
    "Bold": "대담한",
    "Subdued": "차분한",
    "Timeless": "타임리스",
    "Transitional": "트랜지셔널",
    "Resort": "리조트",
    "Cruise": "크루즈",
    "Festival": "페스티벌",
    "Beachwear": "비치웨어",
    "Activewear": "액티브웨어",
    "Athletic": "애슬레틱",
    "Outdoor": "아웃도어",
    "Travel": "여행용",
    "Bridal": "웨딩룩",

    # Materials
    "Cotton": "면",
    "Denim": "데님",
    "Leather": "가죽",
    "Silk": "실크",
    "Wool": "울",
    "Cashmere": "캐시미어",
    "Linen": "린넨",
    "Polyester": "폴리에스터",
    "Nylon": "나일론",
    "Acrylic": "아크릴",
    "Rayon": "레이온",
    "Viscose": "비스코스",
    "Spandex": "스판덱스",
    "Bamboo": "대나무 섬유",
    "Tencel": "텐셀",
    "Modal": "모달",
    "Faux fur": "페이크 퍼",
    "Satin": "새틴",
    "Velvet": "벨벳",
    "Corduroy": "코듀로이",
    "Canvas": "캔버스",
    "Taffeta": "태피터",
    "Chiffon": "쉬폰",
    "Organza": "오간자",
    "Jersey": "저지",
    "Sweatshirt fabric": "스웨트셔츠 원단",
    "Fleece": "플리스",
    "Terry cloth": "테리천",
    "Neoprene": "네오프렌",
    "Mesh": "메쉬",
    "Sheer": "시스루",
    "Transparent": "투명 소재",
    "Ribbed": "골지",
    "Quilted": "퀼팅",
    "Puff print": "퍼프 프린트",
    "Brocade": "브로케이드",
    "Jacquard": "자카드",
    "Tartan": "타탄 체크",

    # Colors
    "Black": "블랙",
    "White": "화이트",
    "Gray": "그레이",
    "Beige": "베이지",
    "Red": "레드",
    "Blue": "블루",
    "Green": "그린",
    "Yellow": "옐로우",
    "Pink": "핑크",
    "Purple": "퍼플",
    "Brown": "브라운",
    "Orange": "오렌지",
    "Turquoise": "터콰이즈",
    "Navy": "네이비",
    "Olive": "올리브",
    "Burgundy": "버건디",
    "Teal": "틸",
    "Coral": "코랄",
    "Lavender": "라벤더",
    "Mustard": "머스타드",
    "Cream": "크림",
    "Maroon": "마룬",
    "Charcoal": "차콜",
    "Ivory": "아이보리",
    "Peach": "피치",
    "Mint": "민트",
    "Aqua": "아쿠아",
    "Magenta": "마젠타",
    "Cyan": "시안",
    "Indigo": "인디고",
    "Fuchsia": "푸시아",
    "Emerald": "에메랄드",
    "Sapphire": "사파이어",
    "Ruby": "루비",
    "Amber": "앰버",
    "Copper": "코퍼",
    "Bronze": "브론즈",
    "Silver": "실버",
    "Gold": "골드",
    "Platinum": "플래티넘",
    "Rose gold": "로즈골드",
    "Champagne": "샴페인",
    "Slate": "슬레이트",
    "Dusty rose": "더스티 로즈",
    "Seafoam": "씨폼",
    "Lilac": "라일락",
    "Berry": "베리",
    "Rust": "러스티",
    "Terracotta": "테라코타",

    # Seasons
    "Spring": "봄",
    "Summer": "여름",
    "Fall": "가을",
    "Winter": "겨울"
}

fashion_tags_en = {v: k for k, v in fashion_tags_kr.items()}

fashion_categories = {
    "Clothing_type" : [
        "Tops", "Bottoms", "One-piece", "Underwear & Sleepwear", "Swimwear & Activewear", "Footwear", "Accessories", "Bags & Wallets"
        ],
    }

# # Convert fashion tags into embeddings
# text_labels = [f"A photo of {tag}" for category in fashion_tags.values() for tag in category]
# label_embeddings = fclip.encode_text(text_labels, batch_size=32)
# label_embeddings /= np.linalg.norm(label_embeddings, axis=-1, keepdims=True)  # Normalize


@app.route("/tags", methods=["POST"]) 
def tags():
    data = request.get_json()
    img_url = data.get("img_url")

    if not img_url:
        return jsonify({"error": "Missing 'imgUrl'"}), 400

    try:
        # 1. imgUrl에서 product_id 추출 (e.g., products/123/...)
        import re
        match = re.search(r"/products/(\d+)/", img_url)
        if not match:
            return jsonify({"error": "Could not extract product_id from imgUrl"}), 400
        product_id = int(match.group(1))

        # 2. 이미지 다운로드 및 전처리
        resp = requests.get(img_url)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")

        # 3. 이미지 임베딩 생성
        with torch.no_grad():
            vector = fclip.encode_images([image], batch_size=1)[0]
            vector = vector / np.linalg.norm(vector)

        # 4. 카테고리별 태그 추출
        categories = ["Clothing Types", "Fashion Styles", "Materials", "Colors", "Seasons"]
        tag_results = []

        for category in categories:
            top_k = 2 if category == "Fashion Styles" else 1

            qr = tag_index.query(
                vector=vector.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter={"category": {"$eq": category}}
            )

            matches = qr.get("matches", [])
            for match in matches:
                raw_tag = match["metadata"].get("tag")
                if raw_tag and raw_tag in fashion_tags_kr:
                    tag_results.append(fashion_tags_kr[raw_tag])

        # 5. 태그만 반환 (❌ Pinecone 업로드 안 함)
        return jsonify({
            "status": "success",
            "results": {
                "tags": tag_results
            }
        })

    except Exception as e:
        app.logger.error(f"[ERROR] Failed to process image: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500  

@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        img_url = data.get("imgUrl")
        tag_data = data.get("tags")
        product_id = data.get("productId")  # BE가 보내는 실제 DB ID

        if not img_url or not tag_data or not product_id:
            return jsonify({"status": "error", "message": "imgUrl, tags, or productId missing"}), 400

        # 1. 이미지 다운로드
        resp = requests.get(img_url)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")

        # 2. 벡터 생성
        with torch.no_grad():
            vector = fclip.encode_images([image], batch_size=1)[0]
            vector = vector / np.linalg.norm(vector)

        # 3. 메타데이터 업로드
        metadata = {
            "tags": [t["tag"] for t in tag_data],
            "categories": [t["category"] for t in tag_data],
            "imgUrl": img_url
        }

        index.upsert([
            {
                "id": str(product_id),
                "values": vector.tolist(),
                "metadata": metadata
            }
        ])

        return jsonify({"status": "success", "message": "업로드 완료"})

    except Exception as e:
        app.logger.error(f"[ML Upload Error] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# {
#   "status": "success",
#   "results": {
#     "product_id": "123",
#     "tags": ["Jacket", "Streetwear", "Denim", "Black", "Fall"]
#   }
# }


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    if not data:
        return jsonify({
            "status": "400",
            "code": "bad_request",
            "results": []
        }), 400

    query_text = data.get("query")
    top_k = data.get("top_k", 15)

    if not query_text:
        return jsonify({
            "status": "400",
            "code": "missing_query",
            "results": []
        }), 400

    if not isinstance(top_k, int) or top_k <= 0:
        return jsonify({
            "status": "400",
            "code": "invalid_top_k",
            "results": []
        }), 400

    try:
        with torch.no_grad():
            query_vector = fclip.encode_text([query_text], batch_size=1)[0]
            query_vector /= np.linalg.norm(query_vector)

        query_results = index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        # match["id"]가 실제 product_id (str이므로 int로 변환 시도)
        product_ids = []
        for m in query_results.get("matches", []):
            match_id = m.get("id")
            try:
                product_ids.append(int(match_id))  # str → int 변환
            except (ValueError, TypeError):
                continue  # 잘못된 id는 건너뜀

        return jsonify({
            "status": "200",
            "code": "ok",
            "results": product_ids
        })

    except Exception as e:
        print(f"[ERROR] Failed to process /search: {e}")
        return jsonify({
            "status": "500",
            "code": "error",
            "results": []
        }), 500

@app.route("/search-recommend", methods=["POST"])
def search_recommend():
    data = request.get_json()
    if not data:
        return jsonify({"status": "400", "code": "bad_request", "results": []}), 400

    query_kr = data.get("query")  # 예: "봄/스웨터/캐주얼"
    top_k = data.get("top_k", 15)

    if not query_kr:
        return jsonify({"status": "400", "code": "missing_query", "results": []}), 400

    # 슬래시(/)로 분할 후 영어 변환
    keywords_kr = query_kr.split("/")
    keywords_en = [fashion_tags_en.get(k.strip(), "") for k in keywords_kr]
    keywords_en = [k for k in keywords_en if k]  # 공백 제거

    if not keywords_en:
        return jsonify({"status": "400", "code": "no_valid_tags", "results": []}), 400

    # 병합된 쿼리 텍스트
    query_text = " ".join(keywords_en)

    try:
        with torch.no_grad():
            query_vector = fclip.encode_text([query_text], batch_size=1)[0]
            query_vector /= np.linalg.norm(query_vector)

        query_results = index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        product_ids = []
        for m in query_results.get("matches", []):
            match_id = m.get("id")
            try:
                product_ids.append(int(match_id))
            except (ValueError, TypeError):
                continue

        return jsonify({
            "status": "200",
            "code": "ok",
            "query_kr": query_kr,
            "query_en": query_text,
            "results": product_ids
        })

    except Exception as e:
        print(f"[ERROR] Failed to process /search-recommend: {e}")
        return jsonify({"status": "500", "code": "error", "results": []}), 500

@app.route("/search-by-image", methods=["POST"])
def search_by_image():
    data = request.get_json()
    image_url = data.get("image_url")
    top_k = data.get("top_k", 15)

    if not image_url:
        return jsonify({
            "status": "400",
            "code": "missing_image_url",
            "results": []
        }), 400

    if not isinstance(top_k, int) or top_k <= 0:
        return jsonify({
            "status": "400",
            "code": "invalid_top_k",
            "results": []
        }), 400

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return jsonify({
            "status": "400",
            "code": "image_fetch_error",
            "results": [],
            "message": f"Failed to fetch image: {e}"
        }), 400

    try:
        with torch.no_grad():
            query_vector = fclip.encode_images([image], batch_size=1)[0]
            query_vector /= np.linalg.norm(query_vector)

        query_results = index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        product_ids = []
        for m in query_results.get("matches", []):
            match_id = m.get("id")
            try:
                product_ids.append(int(match_id))  # id는 str로 저장됐으므로 int 변환
            except (ValueError, TypeError):
                continue

        return jsonify({
            "status": "200",
            "code": "ok",
            "results": product_ids
        })

    except Exception as e:
        print(f"[ERROR] Failed to process /search-by-image: {e}")
        return jsonify({
            "status": "500",
            "code": "error",
            "results": [],
            "message": str(e)
        }), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    print("FashionCLIP model loaded successfully.")
    app.run(host="0.0.0.0", port=5000, debug=True)


