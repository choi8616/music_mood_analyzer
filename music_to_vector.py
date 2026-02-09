import os
import json
import numpy as np
import pandas as pd
import shutil
import torch
from transformers import ClapModel, ClapProcessor
import librosa
import warnings
warnings.filterwarnings('ignore')

class MusisDBBuilder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.new_music_folder = "new_music"
        self.processed_music_folder = "processed_music"
        os.makedirs(self.new_music_folder, exist_ok=True)
        os.makedirs(self.processed_music_folder, exist_ok=True)

        print(f"Running Music Data Base Builder...")
        print(f"Running on {self.device}...")

        print("Loading CLAP...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        print("✅ CLAP Loading Completed!")
    
    def audio_to_vector(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=48000, mono=True)

            # 30초 초과 시 중간 부분 자르기
            if len(audio) > 48000 * 30:
                start = len(audio) // 2 - 48000 * 15
                audio = audio[start:start + 48000 * 30]

            inputs = self.clap_processor(
                audio=[audio],
                sampling_rate=48000,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # [중요] model(...) 대신 get_audio_features(...)를 사용해야 
                # 텍스트 입력(input_ids)이 없어도 에러가 나지 않습니다.
                outputs = self.clap_model.get_audio_features(**inputs)
            
            # [중요] outputs가 Tensor인지 객체인지 확인하여 처리
            # transformers 버전에 따라 리턴 타입이 다를 수 있어 안전장치 추가
            if isinstance(outputs, torch.Tensor):
                audio_embeds = outputs
            elif hasattr(outputs, 'pooler_output'):
                audio_embeds = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # 혹시라도 pooling 전 상태가 나오면 평균을 냄
                audio_embeds = outputs.last_hidden_state.mean(dim=1)
            else:
                # 튜플이나 다른 형태일 경우 첫 번째 요소 사용
                audio_embeds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            # 정규화 (Normalization)
            audio_embeds = audio_embeds / torch.norm(audio_embeds, p=2, dim=-1, keepdim=True)
            
            return audio_embeds.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            return None

    def build_from_folder(self):
        print(f"\n{'='*60}")
        print(f"📁 Scanning folders...")
        print(f"{'='*60}\n")

        # Step 1: new_music 폴더에서 음악 파일 찾기
        audio_files = []
        supported_formats = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')

        for file in os.listdir(self.new_music_folder):
            if file.endswith(supported_formats):
                full_path = os.path.join(self.new_music_folder, file)
                audio_files.append(full_path)

        # 파일명 순으로 정렬
        audio_files.sort()

        # Step 2: 현재 상태 출력
        existing_count = 0
        processed_count = len([f for f in os.listdir(self.processed_music_folder) 
                                if f.endswith(supported_formats)])

        if os.path.exists("music_database_metadata.json"):
            with open("music_database_metadata.json", 'r', encoding='utf-8') as f:
                existing_count = len(json.load(f))

        print(f"📊 Status:")
        print(f"   Already in database: {existing_count} songs")
        print(f"   In processed_music/: {processed_count} files")
        print(f"   In new_music/: {len(audio_files)} files")

        # 새 음악이 없으면 종료
        if not audio_files:
            print("\n✅ No new music to process!")
            print(f"💡 Add music files to '{self.new_music_folder}/' folder")
            print(f"💡 Supported formats: {supported_formats}")
            return

        # 처리할 파일 목록 출력
        print(f"\n🆕 New music to process:")
        for i, path in enumerate(audio_files, 1):
            filename = os.path.basename(path)
            print(f"   {i}. {filename}")

        print(f"\n{'='*60}")
        print("🎵 Starting vectorization...")
        print(f"{'='*60}\n")

        # Step 3: 기존 DB 로드
        existing_vectors = []
        existing_metadata = []

        if os.path.exists("music_database.npy"):
            existing_vectors = np.load("music_database.npy").tolist()
            print(f"📂 Loaded existing vectors: {len(existing_vectors)} songs")

        if os.path.exists("music_database_metadata.json"):
            with open("music_database_metadata.json", 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
            print(f"📂 Loaded existing metadata: {len(existing_metadata)} songs\n")

        # Step 4: 새 음악 벡터화
        new_vectors = []
        new_metadata = []
        successfully_processed = []  # 성공한 파일만 저장

        for i, audio_path in enumerate(audio_files, 1):
            filename = os.path.basename(audio_path)
            
            print(f"[{i}/{len(audio_files)}] Processing: {filename}")
            print("-" * 60)
            
            # 벡터화 시도
            vector = self.audio_to_vector(audio_path)
            
            if vector is not None:
                # 성공!
                new_vectors.append(vector)
                
                # 파일명에서 제목 추출 (.mp3 등 제거)
                title = os.path.splitext(filename)[0]
                
                # processed_music의 새 경로 (이동 후)
                new_path = os.path.join(self.processed_music_folder, filename)
                
                # 메타데이터 생성
                new_metadata.append({
                    "id": len(existing_metadata) + len(new_metadata),  # ID는 순차적으로
                    "file_path": new_path,  # 이동 후 경로
                    "title": title,
                    "mood": "Unknown",
                    "genre": "Unknown"
                })
                
                # 이동할 파일 목록에 추가
                successfully_processed.append((audio_path, new_path))
                
                print(f"✅ Success! Vector shape: {vector.shape}\n")
            else:
                # 실패 (에러 메시지는 audio_to_vector에서 출력됨)
                print(f"⚠️  Skipped due to error (file remains in new_music/)\n")

        # Step 5: DB 저장 및 파일 이동
        if new_vectors:
            print(f"{'='*60}")
            print("💾 Saving database and organizing files...")
            print(f"{'='*60}\n")
            
            # 기존 + 새로운 벡터 합치기
            all_vectors = existing_vectors + new_vectors
            all_metadata = existing_metadata + new_metadata
            
            # NumPy 배열로 변환
            vectors_array = np.array(all_vectors)
            
            # 벡터 저장 (.npy)
            np.save("music_database.npy", vectors_array)
            print(f"✅ Vectors saved: music_database.npy")
            print(f"   Previous: {len(existing_vectors)} songs")
            print(f"   Added: {len(new_vectors)} songs")
            print(f"   Total: {vectors_array.shape[0]} songs")
            print(f"   Vector dimension: {vectors_array.shape[1]}")
            print(f"   File size: {os.path.getsize('music_database.npy') / 1024:.1f} KB")
            
            # 메타데이터 저장 (.json)
            with open("music_database_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ Metadata saved: music_database_metadata.json")
            print(f"   Total songs: {len(all_metadata)}")
            
            # 파일 이동
            print(f"\n📦 Moving processed files to processed_music/...")
            for old_path, new_path in successfully_processed:
                try:
                    shutil.move(old_path, new_path)
                    print(f"   ✓ {os.path.basename(old_path)}")
                except Exception as e:
                    print(f"   ✗ Failed to move {os.path.basename(old_path)}: {e}")
            
            print(f"\n{'='*60}")
            print(f"🎉 Database updated successfully!")
            print(f"{'='*60}\n")
            
            print("📝 Summary:")
            print(f"   • new_music/: {len(os.listdir(self.new_music_folder))} files (should be 0)")
            print(f"   • processed_music/: {len([f for f in os.listdir(self.processed_music_folder) if f.endswith(supported_formats)])} files")
            print(f"   • Database: {len(all_metadata)} songs")
            
            print("\n💡 Next steps:")
            print("   1. Check: new_music/ should be empty")
            print("   2. Edit: music_database_metadata.json (add artist, mood, genre)")
            print("   3. Add more: Put new music in new_music/ folder")
            print("   4. Run again: python3 music_to_vector.py")
            
        else:
            print("\n❌ No songs were processed successfully.")
            print("💡 Check the error messages above and fix the files.")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎵 Music Database Builder")
    print("="*60)
    
    # 인스턴스 생성
    builder = MusisDBBuilder()
    
    # new_music 폴더 처리
    builder.build_from_folder()