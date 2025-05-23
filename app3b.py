import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

import os

st.set_page_config(page_title="RAG Chatbot - HuggingFace + Gemini LLM")

st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")
st.markdown("### 📁 Tải file văn bản (.txt)")
uploaded_file = st.file_uploader("Upload file", type=["txt"])

question = st.text_input("💬 Nhập câu hỏi:")

if uploaded_file and question:
    try:
        st.info("📄 Đang xử lý file...")
        text = uploaded_file.read().decode("utf-8")
        docs = [Document(page_content=text)]

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        st.info("📡 Đang tạo FAISS vector DB...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embedding)

        retriever = vectordb.as_retriever()

        st.info("💡 Đang gọi Gemini LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=st.secrets["GEMINI_API_KEY"]
        )

        retrieved_docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"Với ngữ cảnh sau:\n{context}\n\nTrả lời câu hỏi: {question}"
        response = llm.invoke(prompt)

        st.success("✅ Kết quả:")
        st.write(response.content)

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
