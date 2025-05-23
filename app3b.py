import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load biến môi trường từ file .env hoặc streamlit secrets
# This correctly loads from .env for local development and st.secrets for Streamlit Cloud.
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY chưa được cấu hình. Vui lòng cấu hình trong biến môi trường hoặc Streamlit Secrets.")
    st.stop()

# Set the environment variable for Langchain to use
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["STREAMLIT_WATCHER_TYPE"] = "none" # Often good for deployment

st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")

# Tải file
uploaded_file = st.file_uploader("📁 Tải file văn bản (.txt)", type="txt")
query = st.text_input("💬 Nhập câu hỏi:")

if uploaded_file:
    try:
        st.info("📄 Đang xử lý file...")

        # Save the uploaded file temporarily
        with open("data.txt", "wb") as f:
            f.write(uploaded_file.read())

        loader = TextLoader("data.txt", encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        st.info("📡 Đang tạo FAISS vector DB...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever()

        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        if query:
            st.info("🤖 Đang suy nghĩ...")
            response = qa_chain.run(query)
            st.success(response)

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
