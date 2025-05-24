import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel, configure
import json

# 🔐 Cấu hình API key từ Streamlit secrets
configure(api_key=st.secrets["GEMINI_API_KEY"])

# ⚙️ Khởi tạo LLM Gemini
llm = GenerativeModel("models/gemini-2.0-flash-lite")

# 🧠 Load FAISS index từ thư mục đã lưu
#embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#vectorstore = FAISS.load_local("data_output/faiss_index", embeddings=embedding)
from langchain.vectorstores import FAISS
import pickle
import json

with open("data_output/faiss_index/index.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# Sau đó dùng vectorstore như bình thường...


# 📄 Load văn bản gốc (nếu muốn hiển thị)
with open("data_output/source_documents.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

# 💬 Giao diện Streamlit
st.title("📚 RAG Chatbot (Từ dữ liệu FAISS đã tạo trước)")
query = st.text_input("💬 Nhập câu hỏi của bạn:")

if query:
    # Tìm văn bản liên quan
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Gửi prompt đến Gemini
    prompt = f"Answer based on context:\n{context}\n\nQuestion:\n{query}"
    response = llm.generate_content(prompt)

    # Hiển thị kết quả
    st.markdown("### 🧠 Trả lời:")
    st.write(response.text)

    # Tuỳ chọn: hiển thị lại context
    with st.expander("📄 Ngữ cảnh đã dùng"):
        st.write(context)
