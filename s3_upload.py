import requests

# presigned URL (예: AWS S3에서 발급 받은 PUT URL)
upload_url = "https://closhare.s3.ap-northeast-2.amazonaws.com/products/1748082955127/product_1748082955127_089c76b7.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250524T103555Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=AKIAWDKG3HGWLAKP7TMV%2F20250524%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Signature=a6b7e2d5bddfd8441811df83b4069e20f3596f2c298081c149f3c87a2c6b379e"
# "https://closhare.s3.ap-northeast-2.amazonaws.com/products/1748082955127/product_1748082955127_089c76b7.jpg"

# 업로드할 로컬 이미지 경로
image_path = "1_006.jpg"

# 이미지 바이너리 로드 및 업로드 요청
with open(image_path, "rb") as f:
    image_data = f.read()

    response = requests.put(
        upload_url,
        data=image_data,
        headers={"Content-Type": "image/jpeg"}
    )

# 결과 확인
if response.status_code == 200:
    print("✅ Upload successful.")
else:
    print(f"❌ Upload failed: {response.status_code} - {response.text}")
