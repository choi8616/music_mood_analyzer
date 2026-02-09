import argparse
import json
import os
import subprocess
from typing import List, Tuple

import numpy as np

from image_to_vector import BridgeRecommender


def load_database(vectors_path: str, metadata_path: str) -> Tuple[np.ndarray, List[dict]]:
    """음악 DB 로드 및 검증"""
    vectors = np.load(vectors_path)
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    if vectors.shape[0] != len(metadata):
        raise RuntimeError(
            f"❌ DB 불일치: 벡터={vectors.shape[0]}개, 메타데이터={len(metadata)}개"
        )
    
    # 벡터 정규화
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    
    print(f"✅ DB 로드: {len(metadata)}곡, 차원={vectors.shape[1]}\n")
    
    return vectors, metadata


def compute_similarity(query_vector: np.ndarray, db_vectors: np.ndarray) -> np.ndarray:
    """코사인 유사도 (정규화되어 있으면 dot product)"""
    return db_vectors @ query_vector


def get_top_k(similarities: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """유사도 높은 순 Top-K"""
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(int(i), float(similarities[i])) for i in top_indices]


def print_recommendations(results: List[Tuple[int, float]], metadata: List[dict]):
    """추천 결과 출력"""
    print("\n" + "="*60)
    print("🎵 추천 음악 Top 5")
    print("="*60 + "\n")
    
    for rank, (idx, score) in enumerate(results, 1):
        song = metadata[idx]
        
        print(f"{rank}. 🎵 {song.get('title', 'Unknown')}")
        print(f"   📁 {song.get('file_path', 'Unknown')}")
        print(f"   🎭 mood: {song.get('mood', 'Unknown')}")
        print(f"   🎸 genre: {song.get('genre', 'Unknown')}")
        print(f"   📊 similarity: {score:.4f}")
        print()


def open_file_macos(file_path: str):
    """macOS에서 파일 열기"""
    if not os.path.exists(file_path):
        print(f"⚠️  파일 없음: {file_path}")
        return
    
    try:
        subprocess.run(["open", file_path], check=True)
        print(f"▶️  재생: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"❌ 열기 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="이미지로 음악 추천")
    parser.add_argument("image", help="이미지 파일 경로")
    parser.add_argument("--topk", type=int, default=5, help="추천 개수")
    parser.add_argument("--vectors", default="music_database.npy")
    parser.add_argument("--meta", default="music_database_metadata.json")
    parser.add_argument("--play", action="store_true", help="1위 곡 자동 재생")
    
    args = parser.parse_args()
    
    # 1) 이미지 확인
    if not os.path.exists(args.image):
        print(f"❌ 이미지 없음: {args.image}")
        return
    
    print(f"\n🖼️  이미지: {args.image}\n")
    
    # 2) DB 로드
    db_vectors, metadata = load_database(args.vectors, args.meta)
    
    # 3) 쿼리 벡터 생성
    recommender = BridgeRecommender()
    query_vector = recommender.get_query_vector(args.image)
    
    if query_vector is None:
        print("❌ 쿼리 벡터 생성 실패")
        return
    
    print(f"✅ 쿼리 벡터 생성 완료: {query_vector.shape}\n")
    
    # 4) 유사도 계산
    similarities = compute_similarity(query_vector, db_vectors)
    
    # 5) Top-K 추출
    results = get_top_k(similarities, args.topk)
    
    # 6) 결과 출력
    print_recommendations(results, metadata)
    
    # 7) 자동 재생 (옵션)
    if args.play and results:
        top_idx, _ = results[0]
        top_file = metadata[top_idx]["file_path"]
        open_file_macos(top_file)

if __name__ == "__main__":
    main()