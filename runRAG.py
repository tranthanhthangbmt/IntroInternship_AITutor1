import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel, configure

# ✅ Cấu hình API key Gemini từ secrets
configure(api_key=st.secrets["GEMINI_API_KEY"])

# ✅ Khởi tạo model Gemini
model = GenerativeModel("models/gemini-2.0-flash-lite")

# ✅ Load FAISS index đã lưu
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        "data_output/faiss_index",
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# ✅ UI Streamlit
st.title("📚 RAG Chatbot (dùng FAISS + Gemini)")
query = st.text_input("💬 Nhập câu hỏi của bạn:")

if query:
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the following question based on the context below:

    Context:
    {context}

    Question:
    {query}
    """

    response = model.generate_content(prompt)

    st.markdown("### 💡 Trả lời từ Gemini:")
    st.write(response.text)

    with st.expander("📄 Ngữ cảnh đã dùng"):
        st.write(context)
