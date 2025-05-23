import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader

# Cấu hình môi trường
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")

uploaded_file = st.file_uploader("📁 Tải file văn bản (.txt)", type="txt")

if uploaded_file:
    try:
        # Lưu file
        with open("data.txt", "wb") as f:
            f.write(uploaded_file.read())

        st.write("📄 Đang xử lý file...")

        # Tách văn bản
        loader = TextLoader("data.txt", encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)

        # Khởi tạo vector DB
        st.write("📡 Đang tạo FAISS vector DB...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embedding)

        # Khởi tạo mô hình Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)

        query = st.text_input("💬 Nhập câu hỏi:")
        if query:
            docs = db.similarity_search(query)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Trả lời câu hỏi sau dựa vào ngữ cảnh:\n\n{context}\n\nCâu hỏi: {query}"
            response = llm.invoke(prompt)
            st.write("🧠 Trả lời:", response.content)

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
