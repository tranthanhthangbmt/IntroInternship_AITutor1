import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Cấu hình API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# UI
st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")
uploaded_file = st.file_uploader("📁 Tải file văn bản (.txt)", type="txt")
query = st.text_input("💬 Nhập câu hỏi:")

if uploaded_file:
    try:
        st.write("📄 Đang xử lý file...")
        with open("data.txt", "wb") as f:
            f.write(uploaded_file.read())

        loader = TextLoader("data.txt", encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        st.write("📡 Đang tạo FAISS vector DB...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3),
            retriever=vectorstore.as_retriever()
        )

        if query:
            with st.spinner("⏳ Đang truy vấn..."):
                answer = qa_chain.run(query)
                st.success(answer)

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
