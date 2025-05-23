
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
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

# Safe embedding function
def safe_embedding(chunks, embedding, batch_size=10):
    all_vectors = []
    texts = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch]
        try:
            embedding.embed_documents(batch_texts)  # Trigger to catch error early
            texts.extend(batch_texts)
        except Exception as e:
            st.error(f"Lỗi khi embedding batch {i // batch_size}: {e}")
    return texts

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
texts = safe_embedding(chunks, embedding)
vectordb = FAISS.from_texts(texts, embedding)

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

query = st.text_input("Nhập câu hỏi:")
if query:
    docs = vectordb.similarity_search(query)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Dựa trên văn bản sau, hãy trả lời câu hỏi:\n\n{context}\n\nCâu hỏi: {query}"
    answer = llm.invoke(prompt)
    st.markdown(f"**📌 Trả lời:** {answer.content}")
