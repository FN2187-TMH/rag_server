# services_vector_rerank.py
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import os

class VectorDB:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "chroma_db")
        self.client = chromadb.PersistentClient(path=db_path)

        local_model_path = os.path.join(current_dir, "models", "paraphrase")
        model_source = local_model_path if os.path.exists(local_model_path) else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_source)
        self.collection = self.client.get_or_create_collection(name="rag_exam", embedding_function=self.embedding_fn)
        
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

    def reset_db(self):
        try: self.client.delete_collection(name="rag_exam")
        except Exception: pass
        self.collection = self.client.get_or_create_collection(name="rag_exam", embedding_function=self.embedding_fn)

    def add_chunks(self, chunks: list):
        self.reset_db()
        if not chunks: return
        ids = [f"id_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids)

    def query_context(self, question: str, n_results: int = 5) -> list:
        try:
            # Thu rộng lưới Top 15 bằng Vector mã hóa ngữ nghĩa
            vector_results = self.collection.query(query_texts=[question], n_results=15)
            candidates = vector_results['documents'][0] if vector_results['documents'] else []
            
            if not candidates: return []
            
            # Dùng Reranker lọc tinh tế lại lấy Top chuẩn
            pairs = [[question, doc] for doc in candidates]
            scores = self.reranker.predict(pairs)
            
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            return [item[0] for item in scored_candidates[:n_results]]
        except Exception:
            return []

vector_db = VectorDB()