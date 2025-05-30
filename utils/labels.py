"""Shared tag dictionaries (extracted from original Flask code)."""

# Korean mapping (EN → KR)

fashion_tags_kr = {
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
  "Casual": "캐주얼",
  "Street Fashion" : "스트리트",
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
  "Aesthetic" : "예술적", 
  "Athleisure": "애슬레저",
  "Classic": "클래식",
  "Eclectic": "에클레틱",
  "Romantic": "로맨틱",
  "Feminine": "페미닌",
  "Masculine": "매스큘린",
  "Gender-less" : "젠더리스",
  "Avant-garde": "아방가르드",
  "Luxury": "럭셔리",
  "High fashion": "하이패션",
  "Urban": "어반",
  "Cozy": "코지",
  "Sophisticated": "세련된",
  "Elegant": "우아한",
  "Lovely" : "러블리",
  "Bold": "대담한",
  "Subdued": "차분한",
  "Timeless": "타임리스",
  "Transitional": "트랜지셔널",
  "Beachwear": "비치웨어",
  "Activewear": "액티브웨어",
  "Travel-clothing" : "여행룩", 
  "Bridal": "웨딩룩",
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
  "Spring": "봄",
  "Summer": "여름",
  "Fall": "가을",
  "Winter": "겨울"
}

# Reverse mapping (KR → EN)
fashion_tags_en = {v: k for k, v in fashion_tags_kr.items()}