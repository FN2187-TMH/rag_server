# download_model.py
import os
from sentence_transformers import SentenceTransformer, CrossEncoder


# 2. TẢI VÀ LƯU RERANKER MODEL (Bổ sung phần này)
rerank_name = "BAAI/bge-reranker-base"
rerank_save_path = os.path.join(models_dir, "reranker")

print(f"\n--- Đang tải Reranker: {rerank_name} ---")
rerank_model = CrossEncoder(rerank_name)
print(f"--- Đang lưu Reranker vào: {rerank_save_path} ---")
rerank_model.save(rerank_save_path)

print("\n🎉 HOÀN THÀNH TẢI TẤT CẢ MODEL CHẠY OFFLINE!")