import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# Đảm bảo khóa API tồn tại
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    st.error("🔑 GOOGLE_API_KEY chưa được thiết lập trong Settings > Secrets!")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Giao diện người dùng
st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")
uploaded_file = st.file_uploader("📁 Tải file văn bản (.txt)", type="txt")

query = st.text_input("💬 Nhập câu hỏi:")

# Xử lý khi có file
if uploaded_file and query:
    try:
        st.info("📄 Đang xử lý file...")
        with open("temp.txt", "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = TextLoader("temp.txt", encoding="utf-8")
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        st.info("📡 Đang tạo FAISS vector DB...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embedding)

        retriever = db.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

        st.success("✅ Trả lời:")
        result = llm.invoke(query)
        st.write(result.content)

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
