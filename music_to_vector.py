import os
import json
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor
import librosa
import warnings
warnings.filterwarnings('ignore')

class MusicDatabaseBuilder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {self.device}...")
        
        # 파트너와 동일한 CLAP 모델 사용 (필수!)
        print("Loading CLAP (Audio Encoder)...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        print("✅ CLAP 로드 완료\n")
    
    def audio_to_vector(self, audio_path, sample_rate=48000, duration=30):
        """
        음악 파일 -> CLAP 오디오 임베딩 (512 dim, Normalized)
        파트너의 텍스트 벡터와 동일한 공간에 매핑됩니다.
        """
        try:
            print(f"   📂 로딩: {os.path.basename(audio_path)}")
            
            # 오디오 로드 (CLAP은 48kHz 권장)
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            print(f"   ⏱️  원본 길이: {len(audio)/sr:.1f}초")
            
            # 하이라이트 부분만 추출 (중간 30초)
            max_length = sample_rate * duration
            if len(audio) > max_length:
                start = (len(audio) - max_length) // 2
                audio = audio[start:start + max_length]
                print(f"   ✂️  {duration}초로 자름 (중간 부분)")
            
            # CLAP 입력 형식으로 변환
            inputs = self.clap_processor(
                audio=audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                audio_features = self.clap_model.get_audio_features(**inputs)
            
            # Extract the tensor from the output object
            if hasattr(audio_features, 'pooler_output'):
                audio_features = audio_features.pooler_output
            else:
                raise ValueError("Unexpected output format from get_audio_features")
            
            # 정규화 (파트너 코드와 동일하게!)
            audio_features = audio_features / audio_features.norm(p=2, dim=-1, keepdim=True)
            
            vector = audio_features.cpu().numpy().flatten()
            print(f"   ✅ 벡터 생성 완료: {vector.shape}\n")
            
            return vector
            
        except Exception as e:
            print(f"   ❌ 오류: {e}\n")
            return None
    
    def build_database(self, music_folder="music", output_prefix="music_database"):
        """
        음악 폴더의 모든 파일을 벡터화하여 데이터베이스 생성
        """
        # 지원하는 오디오 포맷
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        
        # 음악 파일 찾기
        music_files = []
        for root, dirs, files in os.walk(music_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    music_files.append(os.path.join(root, file))
        
        if not music_files:
            print(f"⚠️  '{music_folder}' 폴더에 음악 파일이 없습니다!")
            print(f"💡 다음 명령어로 폴더를 만드세요: mkdir {music_folder}")
            print(f"💡 그리고 음악 파일(.mp3, .wav 등)을 넣어주세요.\n")
            return
        
        print(f"📁 {len(music_files)}개의 음악 파일 발견")
        print("="*50 + "\n")
        
        vectors = []
        metadata = []
        
        # 각 음악 파일 처리
        for idx, audio_path in enumerate(music_files, 1):
            print(f"[{idx}/{len(music_files)}] 처리 중...")
            
            vector = self.audio_to_vector(audio_path)
            
            if vector is not None:
                vectors.append(vector)
                
                # 메타데이터 생성 (나중에 수동으로 수정 가능)
                filename = os.path.basename(audio_path)
                title = os.path.splitext(filename)[0]  # 확장자 제거
                
                metadata.append({
                    "id": len(metadata),
                    "file_path": audio_path,
                    "title": title,
                    "artist": "Unknown",  # 🔧 수동 입력 필요
                    "mood": "Unknown",    # 🔧 수동 입력 필요
                    "genre": "Unknown"    # 🔧 수동 입력 필요
                })
        
        # 저장
        if vectors:
            print("="*50)
            print("💾 저장 중...\n")
            
            vectors_array = np.array(vectors)  # (N, 512)
            
            np.save(f"{output_prefix}.npy", vectors_array)
            print(f"✅ 벡터 저장: {output_prefix}.npy")
            print(f"   Shape: {vectors_array.shape}")
            
            with open(f"{output_prefix}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"✅ 메타데이터 저장: {output_prefix}_metadata.json")
            
            print(f"\n🎉 데이터베이스 생성 완료! 총 {len(vectors)}곡")
            print("\n📝 다음 단계:")
            print(f"   1. {output_prefix}_metadata.json 파일을 열어서")
            print(f"      artist, mood, genre를 수동으로 입력하세요")
            print(f"   2. search_engine.py를 실행하여 매칭 테스트를 해보세요")
        else:
            print("❌ 처리된 음악이 없습니다.")


if __name__ == "__main__":
    print("🎵 음악 데이터베이스 빌더")
    print("="*50 + "\n")
    
    # 실행
    builder = MusicDatabaseBuilder()
    
    # music 폴더의 모든 음악을 처리
    builder.build_database(music_folder="music")