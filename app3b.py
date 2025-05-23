import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import os

st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")

uploaded_file = st.file_uploader("📁 Tải file văn bản (.txt)", type="txt")

if uploaded_file:
    st.info("📄 Đang xử lý file...")

    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.read())

    loader = TextLoader("temp.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    st.info("📡 Đang tạo FAISS vector DB...")

    try:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embedding)
        st.success("✅ FAISS DB created successfully!")
    except Exception as e:
        st.error(f"❌ Lỗi tạo FAISS DB: {e}")
