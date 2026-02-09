import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gradio as gr
import numpy as np
import json
import os
from typing import List, Tuple

from image_to_vector import BridgeRecommender


class MusicRecommenderApp:
    def __init__(self):
        # DB 로드
        self.vectors = np.load("music_database.npy")
        
        with open("music_database_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        # 벡터 정규화
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-12)
        
        # BridgeRecommender 로드
        self.recommender = BridgeRecommender()
        
        print(f"✅ DB 로드 완료: {len(self.metadata)}곡")
    
    def recommend(self, image, topk=5):
        """
        이미지로 음악 추천
        
        Args:
            image: PIL Image or numpy array
            topk: 추천 개수
        
        Returns:
            (caption, results_html, audio_path)
        """
        if image is None:
            return "이미지를 업로드해주세요!", "", None
        
        try:
            # 임시 파일로 저장 (BridgeRecommender가 파일 경로를 받으므로)
            temp_path = "temp_image.jpg"
            if hasattr(image, 'save'):
                image.save(temp_path)
            else:
                from PIL import Image
                Image.fromarray(image).save(temp_path)
            
            # 쿼리 벡터 생성
            query_vector = self.recommender.get_query_vector(temp_path)
            
            if query_vector is None:
                return "❌ 이미지 처리 실패", "", None
            
            # 유사도 계산
            similarities = self.vectors @ query_vector
            
            # Top-K
            top_indices = np.argsort(similarities)[::-1][:topk]
            
            # 결과 HTML 생성
            results_html = self._format_results(top_indices, similarities)
            
            # 1위 곡 오디오 파일 경로
            top_audio = self.metadata[int(top_indices[0])]["file_path"]
            
            # Caption 추출 (BLIP 출력 캡처를 위해 recommender 수정 필요, 일단 간단히)
            caption = f"✅ 추천 완료! Top {topk}곡을 찾았습니다."
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return caption, results_html, top_audio
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 오류 발생: {str(e)}\n{traceback.format_exc()}"
            return error_msg, "", None
    
    def _format_results(self, indices, similarities):
        """추천 결과를 HTML로 포맷"""
        html = "<div style='font-family: Arial; padding: 10px;'>"
        
        for rank, idx in enumerate(indices, 1):
            song = self.metadata[int(idx)]
            score = float(similarities[int(idx)])
            
            mood = song.get('mood', 'Unknown')
            genre = song.get('genre', 'Unknown')
            title = song.get('title', 'Unknown')
            
            # 각 곡마다 카드 형태로
            html += f"""
            <div style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            '>
                <div style='font-size: 20px; font-weight: bold;'>
                    #{rank} 🎵 {title}
                </div>
                <div style='margin-top: 8px; opacity: 0.9;'>
                    🎭 Mood: <b>{mood}</b> | 🎸 Genre: <b>{genre}</b>
                </div>
                <div style='margin-top: 5px; opacity: 0.8; font-size: 14px;'>
                    📊 Similarity: {score:.4f}
                </div>
            </div>
            """
        
        html += "</div>"
        return html


def create_interface():
    """Gradio 인터페이스 생성"""
    app = MusicRecommenderApp()
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎵 Music Mood Analyzer
            ### 이미지로 음악 추천받기
            
            이미지를 업로드하면 AI가 분위기를 분석해서 어울리는 음악을 추천해드립니다!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📸 이미지 업로드",
                    type="pil",
                    height=400
                )
                
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="추천 개수"
                )
                
                submit_btn = gr.Button(
                    "🎵 음악 추천받기",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                caption_output = gr.Textbox(
                    label="📝 상태",
                    interactive=False
                )
                
                results_output = gr.HTML(
                    label="🎯 추천 결과"
                )
                
                audio_output = gr.Audio(
                    label="🎧 1위 곡 미리듣기",
                    autoplay=False
                )
        
        # 이벤트 연결
        submit_btn.click(
            fn=lambda img, k: app.recommend(img, int(k)),
            inputs=[image_input, topk_slider],
            outputs=[caption_output, results_output, audio_output]
        )
        
        gr.Markdown(
            """
            ---
            💡 **사용 방법**
            1. 왼쪽에 이미지 드래그&드롭 또는 클릭해서 업로드
            2. 추천 개수 선택 (1~10)
            3. "음악 추천받기" 버튼 클릭
            4. 오른쪽에서 결과 확인 및 1위 곡 미리듣기!
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # 외부 접속 허용
        server_port=7860,
        share=False  # True로 하면 공개 링크 생성
    )