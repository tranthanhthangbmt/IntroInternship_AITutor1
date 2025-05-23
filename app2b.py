import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Cấu hình title
st.set_page_config(page_title="RAG Chatbot Gemini", page_icon="🤖")
st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")

# Tắt cảnh báo tokenizer song song
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Kiểm tra API key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Thiếu GEMINI_API_KEY trong Settings > Secrets.")
    st.stop()
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Load tài liệu và chia nhỏ
with st.spinner("📂 Đang tải dữ liệu..."):
    loader = TextLoader("example.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content) < 1000]
    texts = [doc.page_content for doc in chunks]

# Tạo embedding
with st.spinner("📡 Đang tạo FAISS vector DB..."):
    try:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_texts(texts, embedding)
    except Exception as e:
        st.error(f"❌ Lỗi tạo FAISS DB: {e}")
        st.stop()

# Khởi tạo LLM Gemini
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Giao diện hỏi đáp
query = st.text_input("Nhập câu hỏi:")
if query:
    with st.spinner("🤖 Đang trả lời..."):
        docs = vectordb.similarity_search(query)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Dựa trên văn bản sau, hãy trả lời câu hỏi:\n\n{context}\n\nCâu hỏi: {query}"
        try:
            answer = llm.invoke(prompt)
            st.markdown(f"**📌 Trả lời:** {answer.content}")
        except Exception as e:
            st.error(f"❌ Lỗi truy vấn Gemini: {e}")
