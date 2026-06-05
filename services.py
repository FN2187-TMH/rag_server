from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import vector_db

TEACHER_PROXY_URL = "http://192.168.50.218:8000/api/v1/proxy"
STUDENT_ID = "B22DCCN320"

client = OpenAI(
    base_url=TEACHER_PROXY_URL,
    api_key=STUDENT_ID
)

def process_and_store_document(text: str) -> int:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50, 
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    vector_db.add_chunks(chunks)
    return len(chunks)

def ask_llm_with_rag(question: str) -> dict:
    print("\n" + "="*50)
    print(f"[ASK] Nhận câu hỏi từ thầy:\n👉 {question}")
    
    # 1. Retrieve Context
    contexts = vector_db.query_context(question, n_results=3)
    
    print(f"\n[RETRIEVE] Tìm thấy {len(contexts)} đoạn văn bản liên quan nhất:")
    for i, ctx in enumerate(contexts):
        # In ra 150 ký tự đầu của mỗi chunk để kiểm tra xem tìm đúng ngữ cảnh không
        print(f"  📍 Chunk {i+1}: {ctx[:150]}...") 
        
    context_str = "\n---\n".join(contexts)
    
    # 2. Prompt Engineering
    system_prompt = (
        "Bạn là một trợ lý học tập. Dựa vào tài liệu cung cấp dưới đây, hãy trả lời câu hỏi trắc nghiệm.\n"
        "YÊU CẦU NGHIÊM NGẶT: Chỉ trả về duy nhất 1 ký tự đại diện cho đáp án đúng (Ví dụ: A hoặc B hoặc C hoặc D). "
        "Không giải thích, không viết thêm từ nào khác.\n\n"
        f"Tài liệu tham khảo:\n{context_str}"
    )
    
    # 3. Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.0
    )
    
    answer = response.choices[0].message.content.strip()
    answer = answer.replace(".", "").replace("Đáp án:", "").strip()[-1].upper()
    
    print(f"\n[LLM RESPONSE] 👉 ĐÁP ÁN CUỐI CÙNG TRẢ VỀ: {answer}")
    print("="*50)
    
    return {
        "answer": answer,
        "sources": contexts
    }