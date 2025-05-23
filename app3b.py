import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")
st.info("Nhập câu hỏi sau khi upload file văn bản (.txt)")

# Lấy API key từ streamlit secrets
GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Thiếu API key trong settings. Vui lòng đặt GEMINI_API_KEY.")
    st.stop()

# Upload file
uploaded_file = st.file_uploader("📁 Tải file văn bản (.txt)", type=["txt"])

question = st.text_input("💬 Nhập câu hỏi:")

if uploaded_file and question:
    try:
        st.info("📄 Đang xử lý file...")
        docs = TextLoader(uploaded_file).load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        st.info("📡 Đang tạo FAISS vector DB...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embedding)

        retriever = vectordb.as_retriever()
        context_docs = retriever.get_relevant_documents(question)
        context = "\n".join(doc.page_content for doc in context_docs)

        prompt = f"Trả lời dựa vào nội dung sau:\n---\n{context}\n---\nCâu hỏi: {question}"

        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
        st.info("🤖 Đang tạo câu trả lời...")
        answer = llm.invoke(prompt)
        st.success(answer)

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
