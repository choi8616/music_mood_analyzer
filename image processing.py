import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import ClapModel, ClapProcessor

class BridgeRecommender:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {self.device}...")

        # 1. 이미지 캡션 모델 (BLIP) 로드 - "눈" 역할
        # 이미지를 보고 텍스트로 설명해줍니다.
        print("Loading BLIP (Image Captioning)...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        # 2. 오디오-텍스트 모델 (CLAP) 로드 - "번역기" 역할
        # 텍스트를 파트너의 오디오 벡터와 같은 공간의 숫자로 바꿉니다.
        # *주의*: 파트너가 사용하는 모델명과 정확히 일치해야 합니다 (보통 'laion/clap-htsat-unfused')
        print("Loading CLAP (Text Encoder)...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    def get_query_vector(self, image_path):
        """
        이미지 -> 텍스트 캡션 -> CLAP 텍스트 임베딩 (512 dim, Normalized)
        """
        # --- Step 1: Image to Text (Captioning) ---
        try:
            raw_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            return None

        # 이미지 전처리 및 캡션 생성
        inputs = self.blip_processor(raw_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # max_new_tokens=50: 너무 긴 문장은 필요 없음
            out = self.blip_model.generate(**inputs, max_new_tokens=50)
            
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        
        # 캡션에 '분위기' 키워드를 살짝 더해주면 추천 품질이 올라갑니다.
        # 예: "a photo of ~" 같은 건조한 문장보다는 감성 키워드가 음악 매칭에 유리
        enhanced_caption = f"{caption}, atmospheric, mood, cinematic"
        print(f"🤖 AI가 본 그림: '{caption}' (Query: {enhanced_caption})")

        # --- Step 2: Text to Vector (CLAP Embedding) ---
        # 파트너의 오디오 벡터와 매칭될 텍스트 임베딩 생성
        text_inputs = self.clap_processor(text=[enhanced_caption], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.clap_model.get_text_features(**text_inputs)
        
        # 정규화 (Normalization) - 코사인 유사도를 위해 필수
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        vector = text_features.cpu().numpy().flatten()
        return vector

# --- 실행 테스트 ---
# --- 실행 테스트 ---
if __name__ == "__main__":
    recommender = BridgeRecommender()
    
    # 1. 여기에 실제 이미지 파일 이름을 넣으세요!
    # (코드가 있는 폴더에 이미지가 같이 있어야 합니다)
    image_filename = "lake.jpg"  # <-- 본인 파일명으로 수정
    
    vector = recommender.get_query_vector(image_filename)
    
    if vector is not None:
        print("\n" + "="*30)
        print("🎉 성공! 벡터가 생성되었습니다.")
        print(f"1. 벡터 차원(길이): {vector.shape}")  # (512,) 가 나와야 정답
        print(f"2. 벡터 앞부분 5개: {vector[:5]}")     # 숫자가 보이면 성공
        print("="*30 + "\n")
