import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter

import os

# Đọc API key từ Streamlit secrets
GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("❌ Thiếu GEMINI_API_KEY trong Settings/Secrets")
    st.stop()

st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")
st.markdown("### 📁 Tải file văn bản (.txt)")
uploaded_file = st.file_uploader("Upload file", type=["txt"])

query = st.text_input("💬 Nhập câu hỏi:")

if uploaded_file and query:
    with st.spinner("📄 Đang xử lý file..."):
        text = uploaded_file.read().decode("utf-8")

        # Chia nhỏ văn bản
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_text(text)

        # Tạo embedding
        try:
            st.info("📡 Đang tạo FAISS vector DB...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = FAISS.from_texts(texts, embeddings)
        except Exception as e:
            st.error(f"❌ Lỗi tạo FAISS DB: {e}")
            st.stop()

        # Khởi tạo LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        # Tìm kiếm nội dung phù hợp nhất
        docs = vectordb.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"Dựa trên đoạn văn sau, hãy trả lời câu hỏi:\n\n{context}\n\nCâu hỏi: {query}"
        try:
            answer = llm.invoke(prompt)
            st.success("💡 Trả lời:")
            st.write(answer.content)
        except Exception as e:
            st.error(f"❌ Lỗi khi gọi Gemini API: {e}")
