import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Cấu hình API Gemini
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

st.title("💡 RAG với Gemini API")

@st.cache_resource
def load_chain():
    loader = TextLoader("data/data.txt")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

qa_chain = load_chain()

query = st.text_input("❓ Câu hỏi của bạn:")
if query:
    answer = qa_chain.run(query)
    st.write("📌 Trả lời:")
    st.write(answer)
