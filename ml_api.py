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
    "Relaxed": "릴랙스드",
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
    "Loungewear": "라운지웨어",
    "Activewear": "액티브웨어",
    "Athletic": "애슬레틱",
    "Outdoor": "아웃도어",
    "Travel": "여행용",
    "Weekend": "주말 스타일",
    "Date night": "데이트룩",
    "Cocktail": "칵테일복",
    "Evening": "이브닝웨어",
    "Party": "파티룩",
    "Bridal": "웨딩룩",
    "Gala": "갈라복",
    "Red carpet": "레드카펫",
    "Black tie": "블랙타이",
    "Business formal": "비즈니스 포멀",
    "Smart casual": "스마트 캐주얼",

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


# # Convert fashion tags into embeddings
# text_labels = [f"A photo of {tag}" for category in fashion_tags.values() for tag in category]
# label_embeddings = fclip.encode_text(text_labels, batch_size=32)
# label_embeddings /= np.linalg.norm(label_embeddings, axis=-1, keepdims=True)  # Normalize


@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    product_id = data.get("product_id")
    img_url = data.get("imgUrl")

    if not product_id or not img_url:
        return jsonify({"error": "Missing 'product_id' or 'imgUrl'"}), 400

    try:
        # 이미지 다운로드 및 전처리
        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        with torch.no_grad():
            vector = fclip.encode_images([image], batch_size=1)[0]
            vector /= np.linalg.norm(vector)

        # 카테고리별 태그 추출
        categories = [
            "Clothing Types", "Fashion Styles",
            "Materials", "Colors", "Seasons"
        ]
        tag_results = []
        for category in categories:
            
            if category in ["Fashion Styles"]:
                k_num = 2
            else:
                k_num = 1
            
            query_result = tag_index.query(
                vector=vector.tolist(),
                top_k=k_num,
                include_metadata=True,
                filter={"category": {"$eq": category}}
            )
            matches = query_result.get("matches", [])
            if matches:
                tag = matches[0]["metadata"].get("tag")
                if tag:
                    tag_results.append(fashion_tags_kr[tag])

        # Pinecone 업로드 (태그 포함)
        index.upsert([
            {
                "id": product_id,
                "values": vector.tolist(),
                "metadata": {
                    "product_id": product_id,
                    "tags": tag_results  # List 형태로 저장
                }
            }
        ])

        print(f"[SUCCESS] Uploaded: {product_id} with tags: {tag_results}")
        return jsonify({
            "status": "success",
            "results": {
                "product_id": product_id,
                "tags": tag_results
            }
        })

    except Exception as e:
        print(f"[ERROR] Failed to process {product_id}: {e}")
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
    query_text = request.json.get("query")
    if not query_text:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        with torch.no_grad():
            query_vector = fclip.encode_text([query_text], batch_size=1)[0]
            query_vector /= np.linalg.norm(query_vector)

        query_results = index.query(
            vector=query_vector.tolist(),
            top_k=15,
            include_metadata=True
        )

        results = [
            {"product_id": m["metadata"].get("product_id", "")}
            for m in query_results.get("matches", [])
        ]
        return jsonify({"results": results})

    except Exception as e:
        print(f"[ERROR] Failed to process /search: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/search-by-image", methods=["POST"])
def search_by_image():
    image_url = request.json.get("image_url")
    if not image_url:
        return jsonify({"error": "Missing 'image_url' field"}), 400

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {e}"}), 400

    try:
        with torch.no_grad():
            query_vector = fclip.encode_images([image], batch_size=1)[0]
            query_vector /= np.linalg.norm(query_vector)

        query_results = index.query(
            vector=query_vector.tolist(),
            top_k=15,
            include_metadata=True
        )

        results = [
            {"product_id": m["metadata"].get("product_id", "")}
            for m in query_results.get("matches", [])
        ]
        return jsonify({"results": results})
    
    except Exception as e:
        print(f"[ERROR] Failed to process /search-by-image: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask server...")
    print("FashionCLIP model loaded successfully.")
    app.run(host="0.0.0.0", port=5000, debug=True)


