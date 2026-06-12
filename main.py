import random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import requests

import services

app = FastAPI(title="PTIT Offline RAG Competition - Student Server")
## 192.168.50.218:8000
## 192.168.50.168
TEACHER_BASE_URL = "http://10.170.45.200:8000/api/v1"
MY_STUDENT_ID = "B22DCCN320"      
MY_SERVER_URL = "http://10.170.45.64:5000"   

@app.on_event("startup")
def auto_register_to_teacher():
    url = f"{TEACHER_BASE_URL}/competition/register"
    headers = {
        "X-Student-ID": MY_STUDENT_ID, 
        "Content-Type": "application/json"
    }
    payload = {"server_url": MY_SERVER_URL}
    
    print("\n[STARTUP] --- ĐANG TỰ ĐỘNG ĐĂNG KÝ VỚI TEACHER SERVER ---")
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(f"[SUCCESS] Đăng ký thành công! Kịch bản phản hồi: {response.json()}\n")
        else:
            print(f"[WARNING] Thầy trả về lỗi (Code {response.status_code}): {response.text}\n")
    except Exception as e:
        print(f"[ERROR] Không thể kết nối đến Teacher Server: {e}\n")


class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str

class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int

class AskRequest(BaseModel):
    question: str

class EvaluateRequest(BaseModel):
    document_received: Optional[bool] = False

class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []


# @app.post("/upload", response_model=UploadResponse)
# async def upload_document(payload: UploadRequest):
#     print("\n" + "╔" + "═"*48 + "╗")
#     print(f"║ [RECEIVE] Nhận tài liệu thi từ thầy! ID: {payload.doc_id} ║")
#     print("╚" + "═"*48 + "╝")
#     try:
#         total_chunks = services.process_and_store_document(payload.text)
#         print(f"✅ Xử lý thành công: Cắt thành {total_chunks} chunks và nạp vào Vector DB.")
#         return UploadResponse(status="success", doc_id=payload.doc_id, chunks=total_chunks)
#     except Exception as e:
#         print(f"❌ Lỗi xử lý upload tài liệu: {e}")
#         return UploadResponse(status="fail", doc_id=payload.doc_id, chunks=0)

@app.post("/upload", response_model=UploadResponse)
async def upload_document(payload: UploadRequest):
    print("\n" + "╔" + "═"*52 + "╗")
    print(f"║ [RECEIVE] Nhận tài liệu thi từ thầy! ID: {payload.doc_id or 'None'} ║")
    print("╚" + "═"*52 + "╝")
    
    # 1. In ra màn hình tầm 150 kí tự đầu của document để kiểm tra dữ liệu trực quan
    preview_text = payload.text[:150].replace("\n", " ") # Thay thế xuống dòng thành khoảng trắng để log đẹp, không nát dòng
    print(f"📄 [PREVIEW 150 CHARS]: {preview_text}...")
    print(f"📊 Tổng số ký tự thô nhận được: {len(payload.text)}")

    # 2. Ghi sao lưu toàn bộ dữ liệu raw của document ra file txt vật lý đề phòng sự cố
    try:
        raw_file_name = f"raw_document_{payload.doc_id or 'latest'}.txt"
        with open(raw_file_name, "w", encoding="utf-8") as f:
            f.write(payload.text)
        print(f"💾 [BACKUP SUCCESS] Đã lưu dữ liệu thô vào file: {raw_file_name}")
    except Exception as file_err:
        print(f"⚠️ [FILE ERROR] Không thể ghi file sao lưu txt: {file_err}")

    # 3. Tiến hành đẩy sang services để băm nhỏ (Chunking) và Vector hóa lưu DB
    try:
        total_chunks = services.process_and_store_document(payload.text)
        print(f"✅ Xử lý thành công: Cắt thành {total_chunks} chunks và nạp vào Vector DB.")
        return UploadResponse(status="success", doc_id=payload.doc_id, chunks=total_chunks)
    except Exception as e:
        print(f"❌ Lỗi xử lý upload tài liệu tại services: {e}")
        return UploadResponse(status="fail", doc_id=payload.doc_id, chunks=0)

@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    try:
        # Gọi logic RAG có tích hợp sẵn log chi tiết câu hỏi/câu trả lời
        result = services.ask_llm_with_rag(payload.question)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        random_answer = random.choice(["A", "B", "C", "D"])
        print(f"💥 [CRASH LOG] Quá trình tính toán bị lỗi: {e}")
        print(f"🎲 Kích hoạt chế độ cứu điểm -> Chọn bừa đáp án: {random_answer}")
        return AskResponse(answer=random_answer, sources=[f"Lỗi hệ thống: {str(e)}"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)