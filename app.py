import streamlit as st
#from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("🤖 Gemini-powered RAG Chatbot")

# Load and split documents
loader = TextLoader("example.txt")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Embedding with Gemini
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectordb = FAISS.from_documents(chunks, embedding)

# QA chain using Gemini
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

# Query from user
query = st.text_input("Nhập câu hỏi bạn muốn biết:")
if query:
    docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    st.markdown(f"**📌 Trả lời:** {answer}")
