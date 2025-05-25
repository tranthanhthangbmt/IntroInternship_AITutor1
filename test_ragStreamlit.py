import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel, configure

# ⚠️ Cấu hình API key Gemini (thay bằng key thực tế hoặc dùng dotenv)
configure(api_key="AIzaSyB23c7ttZ-RWiqj9O4dY82NutHsjz0N45s")

# Khởi tạo model Gemini
model = GenerativeModel("models/gemini-2.0-flash-lite")

# Load FAISS index
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
vectorstore = FAISS.load_local(
    "data_output/faiss_index",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Mini RAG Chatbot", page_icon="🤖")
st.title("🤖 Mini RAG Chatbot")
st.caption("Tìm kiếm ngữ cảnh bằng FAISS & trả lời với Gemini 2.0")

# Khởi tạo session state để lưu lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Nhập câu hỏi
query = st.text_input("❓ Nhập câu hỏi của bạn:")

if query:
    # Truy xuất ngữ cảnh liên quan
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Tạo prompt cho Gemini
    prompt = f"""Answer the following question based on the context below:

    Context:
    {context}

    Question:
    {query}
    """
    # Gửi prompt đến Gemini
    response = model.generate_content(prompt)
    answer = response.text.strip()

    # Lưu lịch sử Q&A
    st.session_state.chat_history.append({
        "question": query,
        "answer": answer
    })

# Hiển thị lịch sử hội thoại (mới nhất ở dưới cùng)
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("ai"):
        st.markdown(chat["answer"])
