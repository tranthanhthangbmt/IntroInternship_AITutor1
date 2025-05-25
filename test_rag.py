# ✅ Mới:
from langchain_community.vectorstores import FAISS
#!from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from google.generativeai import GenerativeModel, configure

# Cấu hình API key Gemini
configure(api_key="AIzaSyB23c7ttZ-RWiqj9O4dY82NutHsjz0N45s")  # ✅ Thay bằng API key thực

# Khởi tạo model Gemini
model = GenerativeModel("models/gemini-2.0-flash-lite")

# Load FAISS index đã lưu
#!embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
vectorstore = FAISS.load_local(
    "data_output/faiss_index",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

# Vòng lặp để người dùng nhập câu hỏi
print("🤖 Mini RAG Chatbot - nhập 'exit' để thoát")
while True:
    query = input("❓ Bạn hỏi gì: ").strip()
    if query.lower() == "exit":
        break

    # Truy xuất context liên quan
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Gửi đến Gemini
    prompt = f"""Answer the following question based on the context below:

    Context:
    {context}

    Question:
    {query}
    """
    response = model.generate_content(prompt)
    print("\n💡 Trả lời từ Gemini:\n", response.text)
    print("=" * 50)
