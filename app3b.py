import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")

uploaded_file = st.file_uploader("📁 Tải file văn bản (.txt)", type="txt")
question = st.text_input("💬 Nhập câu hỏi:")

if uploaded_file and question:
    try:
        # Load & chunk text
        raw_text = uploaded_file.read().decode("utf-8")
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents([Document(page_content=raw_text)])

        st.info("📡 Đang tạo FAISS vector DB...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)

        # Gemini
        api_key = st.secrets["GEMINI_API_KEY"]
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
        retriever = db.as_retriever()

        # Answer
        context_docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([d.page_content for d in context_docs])
        prompt = f"Trả lời câu hỏi sau dựa trên văn bản:\n---\n{context_text}\n---\nCâu hỏi: {question}"
        answer = llm.invoke(prompt)

        st.success("✅ Trả lời:")
        st.write(answer.content)

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
