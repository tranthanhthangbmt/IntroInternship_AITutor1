
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("🤖 Gemini-powered RAG Chatbot")

# Load and split documents
loader = TextLoader("example.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
chunks = [c for c in chunks if len(c.page_content) < 1000]

# Gemini Embedding
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Try embedding and handle failure
texts = []
try:
    texts = [doc.page_content for doc in chunks]
    embedding.embed_documents(texts[:2])  # test only small batch
    vectordb = FAISS.from_texts(texts, embedding)
except Exception as e:
    st.error(f"Lỗi khi tạo vector database: {e}")
    vectordb = None

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

query = st.text_input("Nhập câu hỏi:")
if query and vectordb:
    docs = vectordb.similarity_search(query)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Dựa trên văn bản sau, hãy trả lời câu hỏi:\n\n{context}\n\nCâu hỏi: {query}"
    answer = llm.invoke(prompt)
    st.markdown(f"**📌 Trả lời:** {answer.content}")
elif query:
    st.warning("Vector DB chưa được khởi tạo vì lỗi embedding.")
