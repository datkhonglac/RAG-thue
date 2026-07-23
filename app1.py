# app.py
# Ứng dụng RAG đơn giản để hỏi đáp tài liệu thuế bằng Streamlit.
# Kiến trúc:
# 1. Đọc PDF và chia văn bản thành các đoạn nhỏ.
# 2. Chuyển các đoạn thành vector embedding.
# 3. Khi người dùng hỏi, tìm các đoạn gần nghĩa nhất.
# 4. Gửi câu hỏi + ngữ cảnh tìm được cho mô hình OpenAI để tạo câu trả lời.

import hashlib
import os
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Đọc biến môi trường từ file .env nếu người dùng có sử dụng.
load_dotenv()


# -----------------------------
# CẤU HÌNH GIAO DIỆN
# -----------------------------
st.set_page_config(
    page_title="Trợ lý hỏi đáp tài liệu thuế",
    page_icon="📚",
    layout="wide",
)

st.title("Trợ lý hỏi đáp tài liệu thuế")
st.caption(
    "Tải một file PDF luật hoặc hướng dẫn thuế, sau đó đặt câu hỏi dựa trên nội dung tài liệu."
)


# -----------------------------
# HÀM XỬ LÝ PDF
# -----------------------------
def clean_text(text: str) -> str:
    """
    Làm sạch văn bản PDF:
    - Bỏ khoảng trắng dư thừa.
    - Ghép các dòng thành một chuỗi dễ chia đoạn hơn.
    """
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def split_text(
    text: str,
    page_number: int,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[Dict]:
    """
    Chia văn bản thành các đoạn nhỏ có chồng lấn.

    chunk_size:
        Số ký tự tối đa trong mỗi đoạn.

    overlap:
        Số ký tự được lặp lại giữa hai đoạn liên tiếp,
        giúp tránh mất ngữ cảnh tại ranh giới đoạn.
    """
    chunks = []

    if not text:
        return chunks

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "text": chunk,
                    "page": page_number,
                }
            )

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


@st.cache_data(show_spinner=False)
def read_pdf_and_create_chunks(file_bytes: bytes) -> List[Dict]:
    """
    Đọc toàn bộ PDF từ dữ liệu bytes và tạo danh sách các đoạn văn bản.
    Kết quả được cache để không phải đọc lại PDF ở mỗi lần người dùng hỏi.
    """
    reader = PdfReader(BytesIO(file_bytes))
    all_chunks = []

    for page_index, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        page_text = clean_text(page_text)

        page_chunks = split_text(
            text=page_text,
            page_number=page_index + 1,
        )
        all_chunks.extend(page_chunks)

    return all_chunks


# -----------------------------
# HÀM EMBEDDING VÀ TÌM KIẾM
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """
    Tải mô hình embedding đa ngôn ngữ.

    Mô hình này hỗ trợ tiếng Việt và chỉ được tải một lần.
    Lần chạy đầu tiên có thể mất thời gian vì phải tải model về máy.
    """
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_data(show_spinner=False)
def create_embeddings(texts: Tuple[str, ...]) -> np.ndarray:
    """
    Chuyển danh sách đoạn văn thành vector embedding.
    normalize_embeddings=True giúp có thể dùng phép nhân vô hướng
    để tính độ tương đồng cosine.
    """
    model = load_embedding_model()

    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)


def search_relevant_chunks(
    question: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Dict]:
    """
    Tìm top_k đoạn văn gần nghĩa nhất với câu hỏi.
    """
    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Vì các vector đã được chuẩn hóa, dot product tương đương cosine similarity.
    scores = embeddings @ question_embedding

    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        result = dict(chunks[index])
        result["score"] = float(scores[index])
        results.append(result)

    return results


# -----------------------------
# HÀM GỌI MÔ HÌNH NGÔN NGỮ
# -----------------------------
def build_context(retrieved_chunks: List[Dict]) -> str:
    """
    Ghép các đoạn tìm được thành phần ngữ cảnh gửi cho mô hình.
    """
    context_parts = []

    for index, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Nguồn {index} - Trang {item['page']}]\n{item['text']}"
        )

    return "\n\n".join(context_parts)


def ask_openai(
    api_key: str,
    model_name: str,
    question: str,
    retrieved_chunks: List[Dict],
) -> str:
    """
    Gửi câu hỏi và ngữ cảnh cho mô hình OpenAI.
    """
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    context = build_context(retrieved_chunks)

    system_prompt = """
Bạn là trợ lý hỏi đáp tài liệu thuế Việt Nam.

Yêu cầu:
1. Chỉ sử dụng thông tin có trong phần NGỮ CẢNH được cung cấp.
2. Không tự suy đoán điều khoản, mức thuế, thời hạn hoặc thủ tục nếu tài liệu không nêu rõ.
3. Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ rằng tài liệu chưa cung cấp đủ thông tin.
4. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc.
5. Không khẳng định đây là tư vấn pháp lý chính thức.
""".strip()

    user_prompt = f"""
CÂU HỎI:
{question}

NGỮ CẢNH TRÍCH TỪ PDF:
{context}

Hãy trả lời câu hỏi dựa trên ngữ cảnh trên.
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.choices[0].message.content.strip()


# -----------------------------
# KHỞI TẠO SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Cấu hình")

    uploaded_file = st.file_uploader(
        "Tải file PDF luật hoặc hướng dẫn thuế",
        type=["pdf"],
        accept_multiple_files=False,
    )
    
   # Tự động lấy Key từ két sắt và gán cố định tên mô hình
    api_key = st.secrets.get("GROQ_API_KEY", "")
    model_name = "llama-3.3-70b-versatile"

    top_k = st.slider(
        "Số đoạn tài liệu dùng làm ngữ cảnh",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.divider()
    st.markdown("### ✨ Tính năng nâng cao")
    expert_mode = st.toggle("🔍 Phân tích chuyên sâu ")

    if st.button("Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.info(
        "Ứng dụng chỉ trả lời dựa trên tài liệu đã tải lên. "
        "Kết quả không thay thế ý kiến tư vấn của cơ quan thuế hoặc chuyên gia pháp lý."
    )


# -----------------------------
# XỬ LÝ FILE ĐƯỢC TẢI LÊN
# -----------------------------
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    # Chỉ tạo lại chỉ mục khi người dùng tải một file mới.
    if current_hash != st.session_state.document_hash:
        with st.spinner("Đang đọc PDF và tạo chỉ mục tìm kiếm..."):
            chunks = read_pdf_and_create_chunks(file_bytes)

            if not chunks:
                st.error(
                    "Không trích xuất được văn bản từ PDF. "
                    "File có thể là bản scan ảnh và cần OCR trước."
                )
            else:
                texts = tuple(item["text"] for item in chunks)
                embeddings = create_embeddings(texts)

                st.session_state.document_hash = current_hash
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.messages = []

    if st.session_state.chunks:
        st.sidebar.success(
            f"Đã xử lý: {uploaded_file.name}\n\n"
            f"Số đoạn văn bản: {len(st.session_state.chunks)}"
        )
else:
    st.warning("Hãy tải một file PDF ở thanh bên để bắt đầu.")


# -----------------------------
# HIỂN THỊ LỊCH SỬ CHAT
# -----------------------------
# 1. Thêm lời chào mặc định nếu chưa có tin nhắn nào
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Chào bạn! Tôi là trợ lý AI chuyên tra cứu và phân tích tài liệu Thuế. Hãy tải tài liệu của bạn lên thanh bên trái và đặt câu hỏi cho tôi nhé!"
    })

# 2. Hiển thị tin nhắn với Avatar tùy chỉnh
for message in st.session_state.messages:
    # Đặt icon người dùng và icon cô giáo cho AI
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "👩‍🏫"
    
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("🔍 Xem các đoạn tài liệu được truy xuất"):
                for source in message["sources"]:
                    st.markdown(
                        f"**Trang {source['page']} — "
                        f"độ tương đồng {source['score']:.3f}**"
                    )
                    st.write(source["text"])
                    st.divider()


# -----------------------------
# KHUNG NHẬP CÂU HỎI
# -----------------------------
can_chat = (
    uploaded_file is not None
    and bool(st.session_state.chunks)
    and st.session_state.embeddings is not None
)

question = st.chat_input(
    "Nhập câu hỏi về nội dung tài liệu thuế...",
    disabled=not can_chat,
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    if not api_key:
        error_message = (
            "Bạn chưa nhập OpenAI API Key. "
            "Hãy nhập API Key trong thanh bên rồi gửi lại câu hỏi."
        )

        with st.chat_message("assistant", avatar="👩‍🏫"):
            st.error(error_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )
    else:
        with st.chat_message("assistant", avatar="👩‍🏫"):
            with st.spinner("Đang tìm kiếm trong tài liệu và tạo câu trả lời..."):
                try:
                    # Bổ sung yêu cầu phân tích vĩ mô khi bật chế độ chuyên gia.
                    if expert_mode:
                        st.info(
                        
                            "'Đang phân tích'"
                        )
                        question = (
                            question
                            + "\n\n(YÊU CẦU ẨN: Hãy phân tích thêm tác động "
                            "của chính sách thuế này dưới góc độ kinh tế vĩ mô "
                            "và sự vận hành của nền sản xuất trong nền kinh tế "
                            "tư bản chủ nghĩa. Trả lời sắc bén và sâu sắc)."
                        )

                    retrieved_chunks = search_relevant_chunks(
                        question=question,
                        chunks=st.session_state.chunks,
                        embeddings=st.session_state.embeddings,
                        top_k=top_k,
                    )

                    answer = ask_openai(
                        api_key=api_key,
                        model_name=model_name,
                        question=question,
                        retrieved_chunks=retrieved_chunks,
                    )

                    st.markdown(answer)

                    with st.expander(
                        "Xem các đoạn tài liệu được truy xuất"
                    ):
                        for source_item in retrieved_chunks:
                            st.markdown(
                                f"**Trang {source_item['page']} — "
                                f"độ tương đồng "
                                f"{source_item['score']:.3f}**"
                            )
                            st.write(source_item["text"])
                            st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks,
                        }
                    )

                except Exception as exc:
                    error_message = (
                        "Không thể tạo câu trả lời. "
                        f"Chi tiết lỗi: {exc}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )
