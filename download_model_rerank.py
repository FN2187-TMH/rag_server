# download_model.py (Bản đầy đủ, an toàn nhất để chạy trước khi thi)
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# Đường dẫn thư mục lưu trong project của bạn
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")


# 2. TẢI VÀ LƯU RERANKER MODEL
rerank_name = "BAAI/bge-reranker-base"
rerank_save_path = os.path.join(models_dir, "reranker")

print(f"\n--- Đang tải Reranker: {rerank_name} ---")
rerank_model = CrossEncoder(rerank_name)
print(f"--- Đang lưu Reranker vào: {rerank_save_path} ---")
rerank_model.save(rerank_save_path)

print("\n🎉 HOÀN THÀNH TẢI TẤT CẢ MODEL CHẠY OFFLINE!")