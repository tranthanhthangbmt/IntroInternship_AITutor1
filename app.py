import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from langchain.embeddings import HuggingFaceEmbeddings

# Lấy API key từ Streamlit Secrets và ép buộc cấu hình môi trường
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.title("🤖 Gemini-powered RAG Chatbot")

# Load văn bản và tách nhỏ
loader = TextLoader("example.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
chunks = [c for c in chunks if len(c.page_content) < 1000]

# Tạo danh sách văn bản
texts = [doc.page_content for doc in chunks]

# Embedding với kiểm tra toàn bộ
#embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#embedding = HuggingFaceEmbeddings()
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    #vectors = embedding.embed_documents(texts)
    #vectordb = FAISS.from_texts(texts, embedding)    
    vectordb = FAISS.from_texts(texts, embedding)
except Exception as e:
    st.error(f"❌ Lỗi khi tạo vector DB (Gemini Embedding): {e}")
    vectors = []
    vectordb = None

# Tạo LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Giao diện nhập và trả lời
query = st.text_input("Nhập câu hỏi:")
if query and vectordb:
    docs = vectordb.similarity_search(query)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Dựa trên văn bản sau, hãy trả lời câu hỏi:\n\n{context}\n\nCâu hỏi: {query}"
    answer = llm.invoke(prompt)
    st.markdown(f"**📌 Trả lời:** {answer.content}")
elif query:
    st.warning("⚠ Không thể truy vấn vì vector DB chưa được khởi tạo.")
