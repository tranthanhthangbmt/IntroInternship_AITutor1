import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
st.title("🤖 Gemini-powered RAG Chatbot")

# Load and split documents
loader = TextLoader("example.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
chunks = [c for c in chunks if len(c.page_content) < 1000]

# Embedding
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Thử embedding từng batch nhỏ, chỉ giữ cái thành công
texts = []
vectors = []
for i, chunk in enumerate(chunks):
    try:
        text = chunk.page_content
        vector = embedding.embed_documents([text])[0]
        texts.append(text)
        vectors.append(vector)
    except Exception as e:
        st.warning(f"Lỗi batch {i}: {e}")

vectordb = None
if vectors:
    vectordb = FAISS.__from(texts, vectors)

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
    st.warning("Không thể tạo vector DB vì tất cả các đoạn embedding đều thất bại.")
