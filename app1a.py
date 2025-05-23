import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.title("🤖 RAG Chatbot - HuggingFace + Gemini LLM")

# Load dữ liệu
loader = TextLoader("example.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
chunks = [c for c in chunks if len(c.page_content) < 1000]
texts = [doc.page_content for doc in chunks]

# Embedding bằng HuggingFace
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embedding = HuggingFaceEmbeddings()
vectordb = None

try:
    vectordb = FAISS.from_texts(texts, embedding)
except Exception as e:
    st.error(f"❌ Lỗi khi tạo FAISS vector DB: {e}")

# LLM Gemini chỉ để trả lời
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
llm = ChatGoogleGenerativeAI(model="gemini-pro")

query = st.text_input("Nhập câu hỏi:")
if query and vectordb:
    docs = vectordb.similarity_search(query)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Dựa trên văn bản sau, hãy trả lời câu hỏi:\n\n{context}\n\nCâu hỏi: {query}"
    answer = llm.invoke(prompt)
    st.markdown(f"**📌 Trả lời:** {answer.content}")
elif query:
    st.warning("⚠ Không thể truy vấn vì vector DB chưa được khởi tạo.")
