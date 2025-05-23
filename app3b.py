import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings  # <- Sửa lại đường dẫn đúng

# Cấu hình môi trường
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="RAG Chatbot Gemini", page_icon="🤖")
st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")

# Kiểm tra khóa API
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Thiếu GEMINI_API_KEY trong Settings > Secrets.")
    st.stop()
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Tải và xử lý văn bản
with st.spinner("📂 Đang tải dữ liệu..."):
    loader = TextLoader("example.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks if len(c.page_content) < 1000]

# Tạo vector embedding và FAISS
with st.spinner("📡 Đang tạo FAISS vector DB..."):
    try:
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = FAISS.from_texts(texts, embedding)
    except Exception as e:
        st.error(f"❌ Lỗi tạo FAISS DB: {e}")
        st.stop()

# Khởi tạo LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Giao diện nhập câu hỏi
query = st.text_input("Nhập câu hỏi:")
if query:
    with st.spinner("🤖 Đang trả lời..."):
        try:
            docs = vectordb.similarity_search(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"Dựa trên văn bản sau, hãy trả lời câu hỏi:\n\n{context}\n\nCâu hỏi: {query}"
            answer = llm.invoke(prompt)
            st.markdown(f"**📌 Trả lời:** {answer.content}")
        except Exception as e:
            st.error(f"❌ Lỗi truy vấn Gemini: {e}")
