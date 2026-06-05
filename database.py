import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import re
import numpy as np
from rank_bm25 import BM25Okapi
import os

class VectorDB:
    def __init__(self):
        # 1. Khởi tạo Chroma client chạy trên RAM
        self.client = chromadb.Client()

        # 2. Định nghĩa hàm Embedding sử dụng vietnamese-sbert local
        # Kiểm tra nếu đã có folder model local thì load từ đó, không thì mới tải
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(current_dir, "models", "vietnamese-sbert")
        
        model_source = local_model_path if os.path.exists(local_model_path) else "keepitreal/vietnamese-sbert"
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_source
        )

        # 3. Tạo hoặc lấy collection kèm theo hàm embedding xịn tiếng Việt
        self.collection = self.client.get_or_create_collection(
            name="rag_exam",
            embedding_function=self.embedding_fn
        )
        self.bm25 = None
        self.chunks = []

    def _preprocess(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.lower()
        # Giữ lại các ký tự tiếng Việt và số, loại bỏ ký tự đặc biệt gây nhiễu
        text = re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]', ' ', text)
        return text.split()

    def reset_db(self):
        """Xóa dữ liệu cũ nếu cần nạp tài liệu mới"""
        try:
            self.client.delete_collection(name="rag_exam")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="rag_exam",
            embedding_function=self.embedding_fn
        )
        self.chunks = []
        self.bm25 = None

    def add_chunks(self, chunks: List[str]):
        """Lưu các đoạn text vào Vector DB"""
        self.reset_db()
        if not chunks:
            return
            
        self.chunks = chunks
        ids = [f"id_{i}" for i in range(len(self.chunks))]
        
        self.collection.add(
            documents=self.chunks,
            ids=ids
        )
        
        # Khởi tạo BM25 để tìm kiếm từ khóa chính xác (Điều, Khoản, số liệu)
        tokenized_corpus = [self._preprocess(doc) for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query_context(self, question: str, n_results: int = 3) -> List[str]:
        """Tìm kiếm các đoạn văn bản liên quan nhất tới câu hỏi"""
        if not self.chunks:
            return []

        # 1. Vector Search (Semantic)
        try:
            vector_results = self.collection.query(
                query_texts=[question],
                n_results=n_results * 2
            )
            vector_docs = vector_results['documents'][0] if vector_results['documents'] else []
        except Exception:
            vector_docs = []

        # 2. BM25 Search (Keyword)
        try:
            tokenized_query = self._preprocess(question)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(bm25_scores)[::-1][:n_results*2]
            bm25_docs = [self.chunks[i] for i in top_n_indices]
        except Exception:
            bm25_docs = []

        # 3. Hybrid Merge (Interleaving)
        final_docs = []
        seen = set()
        max_len = max(len(bm25_docs), len(vector_docs))
        for i in range(max_len):
            if i < len(bm25_docs) and bm25_docs[i] not in seen:
                final_docs.append(bm25_docs[i])
                seen.add(bm25_docs[i])
            if i < len(vector_docs) and vector_docs[i] not in seen:
                final_docs.append(vector_docs[i])
                seen.add(vector_docs[i])
            if len(final_docs) >= n_results:
                break

        return final_docs[:n_results]

# Khởi tạo một instance dùng chung toàn hệ thống
vector_db = VectorDB()