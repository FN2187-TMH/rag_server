from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import vector_db

TEACHER_PROXY_URL = "http://10.170.45.200:8000/api/v1/proxy"
STUDENT_ID = "B22DCCN320"

client = OpenAI(
    base_url=TEACHER_PROXY_URL,
    api_key=STUDENT_ID
)

def process_and_store_document(text: str) -> int:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    vector_db.add_chunks(chunks)
    return len(chunks)

def ask_llm_with_rag(question: str) -> dict:
    print("\n" + "="*50)
    print(f"[ASK] Nhận câu hỏi từ thầy:\n👉 {question}")
    
    # 1. Retrieve Context
    contexts = vector_db.query_context(question, n_results=5)
    
    print(f"\n[RETRIEVE] Tìm thấy {len(contexts)} đoạn văn bản liên quan nhất:")
    for i, ctx in enumerate(contexts):
        # In ra 150 ký tự đầu của mỗi chunk để kiểm tra xem tìm đúng ngữ cảnh không
        print(f"  📍 Chunk {i+1}: {ctx[:150]}...") 
        
    context_str = "\n---\n".join(contexts)
    
    # 2. Prompt Engineering
    system_prompt = (
        "Bạn là một chuyên gia giải đề thi. Nhiệm vụ của bạn là dựa vào tài liệu tham khảo được cung cấp "
        "để suy luận và trả lời câu hỏi trắc nghiệm từ người dùng.\n\n"
        "HƯỚNG DẪN ĐÁNH GIÁ:\n"
        "1. Đọc kỹ câu hỏi và các phương án lựa chọn (A, B, C, D).\n"
        "2. Đối chiếu chặt chẽ với tài liệu tham khảo. Nếu tài liệu không trực tiếp nhắc đến, hãy dùng logic "
        "để chọn phương án có khả năng đúng nhất.\n\n"
        "YÊU CẦU ĐỊNH DẠNG (CỰC KỲ NGHIÊM NGẶT):\n"
        "- CHỈ TRẢ VỀ DUY NHẤT 1 KÝ TỰ VIẾT HOA: 'A' hoặc 'B' hoặc 'C' hoặc 'D'.\n"
        "- Tuyệt đối KHÔNG viết thêm bất kỳ từ nào khác, không giải thích, không có dấu chấm, không xuống dòng.\n\n"
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