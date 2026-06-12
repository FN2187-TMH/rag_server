# download_model.py
from sentence_transformers import SentenceTransformer
import os

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
save_path = "./models/paraphrase"

print(f"--- Đang tải model {model_name} ---")
model = SentenceTransformer(model_name)

print(f"--- Đang lưu model vào {save_path} ---")
model.save(save_path)
print("--- HOÀN THÀNH! ---")
