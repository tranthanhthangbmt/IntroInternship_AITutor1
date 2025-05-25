#run: streamlit run test_RAGStreamlit.py
import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel, configure

# ⚠️ Cấu hình API key Gemini (thay bằng key thực tế hoặc dùng dotenv)
configure(api_key="AIzaSyB23c7ttZ-RWiqj9O4dY82NutHsjz0N45s")

# Khởi tạo model Gemini
model = GenerativeModel("models/gemini-2.0-flash-lite")

# Load FAISS index
#embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
#embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


vectorstore = FAISS.load_local(
    "IntroInternshipRAG/faiss_index",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Tutor AI – Hỗ trợ Thực tập CNTT", page_icon="🎓")
# Sidebar – hiển thị logo và thông tin
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tranthanhthangbmt/AITutor_Gemini/main/LOGO_UDA_2023_VN_EN_chuan2.png", width=180)
    st.markdown("""
    ### 🎓 Tutor AI – Đại học Đông Á
    **Hỗ trợ sinh viên thực tập ngành CNTT**

    ---
    📍 *Mọi thắc mắc vui lòng nhập bên dưới để được giải đáp.*
    """)
    
st.title("🎓 Tutor AI - Hỗ trợ Thực tập CNTT")
#st.caption("Tìm kiếm ngữ cảnh bằng FAISS & trả lời với Gemini 2.0")
with st.chat_message("assistant"):
    st.markdown("""
    👋 **Xin chào!**  
    Tôi là **Tutor AI** – trợ lý ảo hỗ trợ sinh viên thực hiện **Thực tập Nhận Thức ngành Công nghệ Thông tin** tại Trường Đại học Đông Á.

    🎯 Nhiệm vụ của tôi:
    - Hướng dẫn bạn nắm rõ nội dung, yêu cầu và lịch trình thực tập
    - Tư vấn cách ghi nhật ký, viết báo cáo đúng chuẩn
    - Trả lời các câu hỏi liên quan đến: **mẫu biểu**, **đánh giá**, **báo cáo tuần**, **thái độ - kỹ năng nghề nghiệp**
    - Giải thích quy trình thực tập và giúp bạn định hướng nghề nghiệp

    ✏️ Hãy nhập câu hỏi bên dưới như:
    - “Cần nộp những biểu mẫu nào trong thực tập?”
    - “Mẫu nhật ký thực tập viết thế nào?”
    - “Bài toán thực tập là gì? Làm sao để chọn?”
    
    Tôi luôn sẵn sàng đồng hành cùng bạn trong suốt 10 tuần thực tập 🤝
    """)

# Khởi tạo session state để lưu lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Nhập câu hỏi
#query = st.text_input("❓ Nhập câu hỏi của bạn:")
query = st.chat_input("❓ Nhập câu hỏi của bạn:")


if query:
    # Truy xuất ngữ cảnh liên quan
    docs = vectorstore.similarity_search(query, k=6)
    context = "\n".join([doc.page_content for doc in docs])

    # Tạo prompt cho Gemini
    prompt = f"""
    Bạn là trợ lý AI đang hỗ trợ sinh viên thực tập CNTT.
    Hãy trả lời câu hỏi dưới đây một cách chi tiết, chính xác và dễ hiểu nhất, dựa trên các thông tin có trong tài liệu:

    Ngữ cảnh:
    {context}

    Câu hỏi:
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
