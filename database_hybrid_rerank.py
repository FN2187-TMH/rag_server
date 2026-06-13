# database_hybrid_rerank.py
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re
import numpy as np
import os

class VectorDB:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. KHỞI TẠO CHROMA CLIENT VỚI CHẾ ĐỘ LƯU TRỮ VĨNH VIỄN (Đã sửa lỗi thiếu)
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
        
        # 4. ĐỊNH NGHĨA VÀ KHỞI TẠO COLLECTION TRƯỚC KHI LOAD DATA (Đã sắp xếp lại đúng trình tự)
        self.collection = self.client.get_or_create_collection(
            name="rag_exam", 
            embedding_function=self.embedding_fn
        )
        
        self.chunks = []
        self.bm25 = None
        
        # 5. TỰ ĐỘNG KHÔI PHỤC DỮ LIỆU ĐỂ BUILD LẠI BM25 KHI RESTART SERVER
        self._reload_existing_data()

    def _reload_existing_data(self):
        try:
            existing_data = self.collection.get()
            if existing_data and existing_data['documents']:
                # Sắp xếp lại theo ID để giữ đúng trình tự mạch văn bản quy định
                combined = list(zip(existing_data['ids'], existing_data['documents']))
                combined.sort(key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
                
                self.chunks = [item[1] for item in combined]
                if self.chunks:
                    tokenized_corpus = [self._preprocess(doc) for doc in self.chunks]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    print(f"✅ [HYBRID-RERANK] Đã nạp lại thành công {len(self.chunks)} chunks có sẵn từ SSD.")
        except Exception as e:
            print(f"ℹ️ Chưa có dữ liệu cũ trong kho lưu trữ cứng hoặc lỗi nạp: {e}")

    def _preprocess(self, text: str) -> list:
        if not text: return []
        text = text.lower()
        text = re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]', ' ', text)
        return text.split()

    def reset_db(self):
        """Xóa sạch bộ nhớ Chroma và RAM để chuẩn bị nạp đề mới"""
        try: 
            self.client.delete_collection(name="rag_exam")
        except Exception: 
            pass
        self.collection = self.client.get_or_create_collection(
            name="rag_exam", 
            embedding_function=self.embedding_fn
        )
        self.chunks, self.bm25 = [], None

    def add_chunks(self, chunks: list):
        """Lưu toàn bộ mảnh văn bản vào đĩa SSD và khởi tạo thực thể từ khóa BM25"""
        self.reset_db()
        if not chunks: return
        self.chunks = chunks
        ids = [f"id_{i}" for i in range(len(self.chunks))]
        
        # Ghi dữ liệu theo Batch nhỏ (128 cụm một lần) để tránh quá tải RAM cục bộ
        batch_size = 128
        for i in range(0, len(self.chunks), batch_size):
            batch_chunks = self.chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.collection.add(documents=batch_chunks, ids=batch_ids)
            
        print(f"💾 Đã lưu và đồng bộ {self.collection.count()} chunks vào Vector DB.")
        
        # Khởi dựng bộ chỉ mục từ khóa BM25
        tokenized_corpus = [self._preprocess(doc) for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query_context(self, question: str, n_results: int = 5) -> list:
        """Thực hiện Hybrid Search sâu kết hợp Reranker nội bộ"""
        if not self.chunks or self.collection.count() == 0: 
            return []

        # Bước 1.1: Quét diện rộng lấy Top 10 ứng viên bằng Semantic Vector Search
        try:
            vector_results = self.collection.query(query_texts=[question], n_results=10)
            vector_docs = vector_results['documents'][0] if vector_results['documents'] else []
        except Exception: 
            vector_docs = []

        # Bước 1.2: Quét diện rộng lấy Top 10 ứng viên bằng Keyword BM25 Search
        try:
            tokenized_query = self._preprocess(question)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[::-1][:10]
            bm25_docs = [self.chunks[i] for i in top_indices]
        except Exception: 
            bm25_docs = []

        # Bước 1.3: Trộn xen kẽ Interleaving để lọc bớt trùng lặp
        candidates = []
        seen = set()
        for i in range(max(len(bm25_docs), len(vector_docs))):
            if i < len(bm25_docs) and bm25_docs[i] not in seen:
                candidates.append(bm25_docs[i])
                seen.add(bm25_docs[i])
            if i < len(vector_docs) and vector_docs[i] not in seen:
                candidates.append(vector_docs[i])
                seen.add(vector_docs[i])

        if not candidates: 
            return []

        # Bước 2: Dùng Reranker chấm điểm tương quan phân phối xác suất ngữ nghĩa (Question, Candidate)
        try:
            pairs = [[question, doc] for doc in candidates]
            scores = self.reranker.predict(pairs)
            
            # Sắp xếp danh sách ứng viên giảm dần theo điểm số uy tín của Reranker
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Trả về các đoạn ngữ cảnh chất lượng nhất để đẩy vào Prompt
            return [item[0] for item in scored_candidates[:n_results]]
        except Exception as e:
            print(f"⚠️ Lỗi xử lý Reranker: {e}. Tự động fallback trả về tập thô.")
            return candidates[:n_results]

# Khởi tạo một instance duy nhất dùng chung toàn hệ thống khi file được nạp
vector_db = VectorDB()