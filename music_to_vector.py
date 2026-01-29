import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import ClapModel, ClapProcessor
import librosa
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class FMADatabaseBuilder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {self.device}...")
        
        print("Loading CLAP (Audio Encoder)...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        print("✅ CLAP 로드 완료\n")
    
    def load_fma_metadata(self, metadata_path="fma_metadata/tracks.csv"):
        """
        FMA 메타데이터 로드 및 파싱
        """
        print("📊 FMA 메타데이터 로딩...")
        
        try:
            # FMA CSV는 multi-level header를 가짐
            tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
            
            metadata_list = []
            
            for track_id in tracks.index:
                try:
                    # track_id를 6자리 문자열로 변환 (예: 2 -> "000002")
                    track_id_str = str(track_id).zfill(6)
                    
                    # 메타데이터 추출
                    title = str(tracks.loc[track_id, ('track', 'title')])
                    artist = str(tracks.loc[track_id, ('artist', 'name')])
                    genre_top = str(tracks.loc[track_id, ('track', 'genre_top')])
                    
                    # NaN 체크
                    if title == 'nan':
                        title = f"Track {track_id_str}"
                    if artist == 'nan':
                        artist = "Unknown Artist"
                    if genre_top == 'nan':
                        genre_top = "Unknown"
                    
                    metadata_list.append({
                        'track_id': track_id_str,
                        'title': title,
                        'artist': artist,
                        'genre': genre_top,
                    })
                    
                except Exception as e:
                    continue
            
            print(f"✅ {len(metadata_list)}곡의 메타데이터 로드 완료\n")
            return metadata_list
            
        except Exception as e:
            print(f"❌ 메타데이터 로드 실패: {e}")
            return []
    
    def audio_to_vector(self, audio_path, sample_rate=48000, duration=30):
        """
        음악 파일 -> CLAP 벡터
        """
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            
            # 하이라이트 부분만 추출 (중간 30초)
            max_length = sample_rate * duration
            if len(audio) > max_length:
                start = (len(audio) - max_length) // 2
                audio = audio[start:start + max_length]
            
            # CLAP 처리
            inputs = self.clap_processor(
                audios=[audio],
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clap_model.get_audio_features(**inputs)
            
            # 텐서 추출
            if hasattr(outputs, 'audio_embeds'):
                audio_embeds = outputs.audio_embeds
            else:
                audio_embeds = outputs
            
            # 정규화
            audio_embeds = audio_embeds / torch.norm(audio_embeds, p=2, dim=-1, keepdim=True)
            
            return audio_embeds.cpu().numpy().flatten()
            
        except Exception as e:
            return None
    
    def get_audio_path(self, track_id, fma_folder="fma_small"):
        """
        FMA 폴더 구조에 맞는 파일 경로 생성
        FMA 구조: fma_small/000/000002.mp3
        """
        subfolder = track_id[:3]  # 처음 3자리
        audio_path = os.path.join(fma_folder, subfolder, f"{track_id}.mp3")
        return audio_path
    
    def build_database(self, 
                      fma_folder="fma_small",
                      metadata_path="fma_metadata/tracks.csv",
                      output_prefix="music_database",
                      max_songs=100):
        """
        FMA 데이터셋에서 음악 데이터베이스 생성
        
        Args:
            fma_folder: FMA 오디오 파일 폴더 (fma_small 등)
            metadata_path: tracks.csv 경로
            output_prefix: 출력 파일 이름 prefix
            max_songs: 최대 처리할 곡 수
        """
        # 메타데이터 로드
        fma_metadata = self.load_fma_metadata(metadata_path)
        
        if not fma_metadata:
            print("❌ 메타데이터를 로드할 수 없습니다.")
            return
        
        vectors = []
        metadata = []
        
        print(f"🎵 최대 {max_songs}곡 처리 시작...")
        print(f"📁 오디오 폴더: {fma_folder}\n")
        
        # 진행상황 표시
        processed = 0
        for meta in tqdm(fma_metadata, desc="Processing tracks"):
            if processed >= max_songs:
                break
            
            track_id = meta['track_id']
            audio_path = self.get_audio_path(track_id, fma_folder)
            
            # 파일 존재 확인
            if not os.path.exists(audio_path):
                continue
            
            # 벡터 생성
            vector = self.audio_to_vector(audio_path)
            
            if vector is not None:
                vectors.append(vector)
                
                metadata.append({
                    "id": len(metadata),
                    "file_path": audio_path,
                    "title": meta['title'],
                    "artist": meta['artist'],
                    "mood": meta['genre'],  # 장르를 무드로 사용
                    "genre": meta['genre']
                })
                
                processed += 1
        
        # 저장
        if vectors:
            print(f"\n💾 저장 중...\n")
            
            vectors_array = np.array(vectors)
            np.save(f"{output_prefix}.npy", vectors_array)
            print(f"✅ 벡터 저장: {output_prefix}.npy")
            print(f"   Shape: {vectors_array.shape}")
            
            with open(f"{output_prefix}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"✅ 메타데이터 저장: {output_prefix}_metadata.json")
            
            print(f"\n🎉 데이터베이스 생성 완료! 총 {len(vectors)}곡")
        else:
            print("❌ 처리된 음악이 없습니다.")


if __name__ == "__main__":
    print("🎵 FMA 음악 데이터베이스 빌더")
    print("="*50 + "\n")
    
    builder = FMADatabaseBuilder()
    
    # FMA Small 데이터셋으로 100곡 처리
    builder.build_database(
        fma_folder="fma_small",  # 다운로드한 FMA 폴더
        metadata_path="fma_metadata/tracks.csv",  # 메타데이터 CSV
        output_prefix="music_database",
        max_songs=100  # 원하는 곡 수 (테스트: 30, 실전: 500+)
    )