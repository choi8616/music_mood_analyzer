import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import ClapModel, ClapProcessor

class BridgeRecommender:
    def __init__(self):
        self.device = "cpu"  # 안정성 우선
        
        print(f"Running on {self.device}...")
        
        print("Loading BLIP...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
        print("Loading CLAP...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        
        print("✅ Models loaded!\n")
    
    def get_query_vector(self, image_path):
        try:
            # 이미지 로드
            raw_image = Image.open(image_path).convert('RGB')
            
            # BLIP으로 캡션 생성
            inputs = self.blip_processor(raw_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_new_tokens=50)
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # 캡션 강화
            enhanced_caption = f"{caption}, atmospheric, mood, cinematic"
            
            print(f"Caption: {caption}")
            print(f"Enhanced: {enhanced_caption}\n")
            
            # CLAP 텍스트 임베딩 생성
            text_inputs = self.clap_processor(
                text=[enhanced_caption],
                return_tensors="pt",
                padding=True
            )
            
            # 각 텐서를 device로 이동
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                text_features = self.clap_model.get_text_features(**text_inputs)
            
            # 안전하게 텐서 추출
            if isinstance(text_features, torch.Tensor):
                embeds = text_features
            elif hasattr(text_features, 'text_embeds'):
                embeds = text_features.text_embeds
            elif hasattr(text_features, 'pooler_output'):
                embeds = text_features.pooler_output
            elif hasattr(text_features, 'last_hidden_state'):
                embeds = text_features.last_hidden_state.mean(dim=1)
            else:
                embeds = text_features[0] if isinstance(text_features, (tuple, list)) else text_features
            
            # 정규화
            embeds = embeds / torch.norm(embeds, p=2, dim=-1, keepdim=True)
            
            # numpy로 변환
            vector = embeds.cpu().numpy().flatten()
            
            return vector
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    recommender = BridgeRecommender()
    
    vector = recommender.get_query_vector("lake.jpg")
    
    if vector is not None:
        print(f"✅ Query vector shape: {vector.shape}")
        print(f"✅ Vector norm: {(vector**2).sum()**0.5:.6f}")
    else:
        print("❌ Failed to generate query vector")