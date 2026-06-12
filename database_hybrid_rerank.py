# services_hybrid_rerank.py
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
        db_path = os.path.join(current_dir, "chroma_db")
        self.client = chromadb.PersistentClient(path=db_path)

        # Cấu hình Embedding Local
        local_model_path = os.path.join(current_dir, "models", "paraphrase")
        model_source = local_model_path if os.path.exists(local_model_path) else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_source)
        self.collection = self.client.get_or_create_collection(name="rag_exam", embedding_function=self.embedding_fn)
        
        # Cấu hình Reranker Local (Tải sẵn tương tự như embedding hoặc chạy tự động nếu có mạng)
        # Bạn có thể dùng model: "BAAI/bge-reranker-base" cực kỳ mạnh về chấm điểm ngữ nghĩa tiếng Việt
        print("🎯 Đang khởi tạo bộ lọc xếp hạng Reranker...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        
        self.chunks = []
        self.bm25 = None
        self._reload_existing_data()

    def _reload_existing_data(self):
        try:
            existing_data = self.collection.get()
            if existing_data and existing_data['documents']:
                combined = list(zip(existing_data['ids'], existing_data['documents']))
                combined.sort(key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
                self.chunks = [item[1] for item in combined]
                if self.chunks:
                    tokenized_corpus = [self._preprocess(doc) for doc in self.chunks]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    print(f"✅ [RERANK MODE] Đã nạp lại {len(self.chunks)} chunks từ SSD.")
        except Exception as e:
            print(f"ℹ️ Chưa có dữ liệu cũ: {e}")

    def _preprocess(self, text: str) -> list:
        if not text: return []
        text = text.lower()
        text = re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]', ' ', text)
        return text.split()

    def reset_db(self):
        try: self.client.delete_collection(name="rag_exam")
        except Exception: pass
        self.collection = self.client.get_or_create_collection(name="rag_exam", embedding_function=self.embedding_fn)
        self.chunks, self.bm25 = [], None

    def add_chunks(self, chunks: list):
        self.reset_db()
        if not chunks: return
        self.chunks = chunks
        ids = [f"id_{i}" for i in range(len(self.chunks))]
        self.collection.add(documents=self.chunks, ids=ids)
        tokenized_corpus = [self._preprocess(doc) for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query_context(self, question: str, n_results: int = 5) -> list:
        if not self.chunks: return []

        # Bước 1: Thu thập diện rộng (Top 10 từ Vector và Top 10 từ BM25)
        try:
            vector_results = self.collection.query(query_texts=[question], n_results=10)
            vector_docs = vector_results['documents'][0] if vector_results['documents'] else []
        except Exception: vector_docs = []

        try:
            tokenized_query = self._preprocess(question)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[::-1][:10]
            bm25_docs = [self.chunks[i] for i in top_indices]
        except Exception: bm25_docs = []

        # Trộn Interleaving lấy tập thô (candidate chunks)
        candidates = []
        seen = set()
        for i in range(max(len(bm25_docs), len(vector_docs))):
            if i < len(bm25_docs) and bm25_docs[i] not in seen:
                candidates.append(bm25_docs[i]); seen.add(bm25_docs[i])
            if i < len(vector_docs) and vector_docs[i] not in seen:
                candidates.append(vector_docs[i]); seen.add(vector_docs[i])

        if not candidates: return []

        # Bước 2: Kích hoạt Reranker chấm điểm tương quan thực tế giữa (Câu hỏi, Chunks)
        pairs = [[question, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sắp xếp lại danh sách ứng viên dựa trên điểm số giảm dần của Reranker
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Trả về số lượng ngữ cảnh tinh khiết nhất theo yêu cầu câu lệnh
        final_docs = [item[0] for item in scored_candidates[:n_results]]
        return final_docs

vector_db = VectorDB()