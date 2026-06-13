# database_vector_rerank.py
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import os

class VectorDB:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. KHỞI TẠO CHROMA CLIENT ĐƯỜNG DẪN VẬT LÝ VĨNH VIỄN (ĐÃ SỬA LỖI THIẾU)
        db_path = os.path.join(current_dir, "chroma_db")
        self.client = chromadb.PersistentClient(path=db_path)

        # 2. CẤU HÌNH EMBEDDING LOCAL
        local_model_path = os.path.join(current_dir, "models", "paraphrase")
        model_source = local_model_path if os.path.exists(local_model_path) else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_source
        )

        # 3. CẤU HÌNH RERANKER LOCAL
        local_rerank_path = os.path.join(current_dir, "models", "reranker")
        rerank_source = local_rerank_path if os.path.exists(local_rerank_path) else "BAAI/bge-reranker-base"
        
        print(f"🧠 Khởi chạy Reranker từ nguồn: {rerank_source}")
        self.reranker = CrossEncoder(rerank_source, max_length=512)

        # 4. KHỞI TẠO COLLECTION NGAY KHI BẬT SERVER (ĐÃ SỬA LỖI THIẾU)
        self.collection = self.client.get_or_create_collection(
            name="rag_exam", 
            embedding_function=self.embedding_fn
        )
        print(f"📊 Hiện đang có {self.collection.count()} chunks sẵn sàng trong kho SSD.")

    def reset_db(self):
        """Xóa sạch bộ nhớ để nạp đề thi mới nếu cần"""
        try: 
            self.client.delete_collection(name="rag_exam")
        except Exception: 
            pass
        self.collection = self.client.get_or_create_collection(
            name="rag_exam", 
            embedding_function=self.embedding_fn
        )

    def add_chunks(self, chunks: list):
        """Đẩy dữ liệu băm nhỏ từ upload vào kho lưu trữ cứng"""
        self.reset_db()
        if not chunks: 
            return
        
        ids = [f"id_{i}" for i in range(len(chunks))]
        
        # Chia cụm (Batch) nhỏ khi nạp để tránh quá tải RAM nếu tài liệu quá dài
        batch_size = 128
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.collection.add(documents=batch_chunks, ids=batch_ids)
        print(f"💾 Đã nạp mới và lưu cứng vĩnh viễn {self.collection.count()} chunks vào SSD.")

    def query_context(self, question: str, n_results: int = 5) -> list:
        """Tìm kiếm ngữ cảnh kết hợp bộ lọc xếp hạng sâu Reranker"""
        try:
            # Nếu DB trống thì dừng lại luôn, đỡ tốn thời gian tính toán
            if self.collection.count() == 0:
                print("⚠️ [CẢNH BÁO DB] Vector DB local đang trống rỗng!")
                return []

            # Bước 1: Thu lưới rộng gấp 3 lần nhu cầu bằng Vector Search ngữ nghĩa
            vector_results = self.collection.query(query_texts=[question], n_results=15)
            candidates = vector_results['documents'][0] if vector_results['documents'] else []
            
            if not candidates: 
                return []
            
            # Bước 2: Ép cặp Reranker tính toán độ tương quan thực tế câu hỏi câu trả lời
            pairs = [[question, doc] for doc in candidates]
            scores = self.reranker.predict(pairs)
            
            # Bước 3: Sắp xếp giảm dần theo điểm số uy tín của Reranker
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Trả về số lượng mảnh bối cảnh sạch nhất
            return [item[0] for item in scored_candidates[:n_results]]
        except Exception as e:
            print(f"❌ Lỗi trong quá trình truy vấn RAG-Rerank: {e}")
            return []

# Khởi tạo một thực thể duy nhất dùng chung cho toàn bộ luồng services.py
vector_db = VectorDB()