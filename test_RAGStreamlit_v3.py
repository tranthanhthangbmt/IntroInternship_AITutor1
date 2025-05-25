#run: streamlit run test_RAGStreamlit.py
import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel, configure

#audio
import edge_tts
from modules.audio_module import generate_and_encode_audio

def render_audio_block(text: str, autoplay=False):
    b64 = generate_and_encode_audio(text)
    autoplay_attr = "autoplay" if autoplay else ""
    st.markdown(f"""
    <audio controls {autoplay_attr}>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Trình duyệt của bạn không hỗ trợ phát âm thanh.
    </audio>
    """, unsafe_allow_html=True)

# ⚠️ Cấu hình API key Gemini (thay bằng key thực tế hoặc dùng dotenv)
#configure(api_key="AIzaSyB23c7ttZ-RWiqj9O4dY82NutHsjz0N45s")
import streamlit as st
from google.generativeai import configure

configure(api_key=st.secrets["GEMINI_API_KEY"])
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ Thiếu khóa API Gemini. Vui lòng khai báo trong Settings > Secrets.")
    st.stop()

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
    if "enable_audio_playback" not in st.session_state:
        st.session_state["enable_audio_playback"] = True  # mặc định bật
    
    st.session_state["enable_audio_playback"] = st.sidebar.checkbox(
        "🔊 Tự động phát âm thanh",
        value=st.session_state["enable_audio_playback"]
    )

    st.markdown("""
    ### 🎓 Tutor AI – Đại học Đông Á
    **Hỗ trợ sinh viên thực tập ngành CNTT**

    ---
    📍 *Mọi thắc mắc vui lòng nhập bên dưới để được giải đáp.*
    """)
    
st.title("🎓 Tutor AI - Hỗ trợ Thực tập CNTT")
#st.caption("Tìm kiếm ngữ cảnh bằng FAISS & trả lời với Gemini 2.0")
with st.chat_message("assistant"):
    intro_text = """
    **Xin chào!**
    
    Tôi là **Tutor AI** – trợ lý ảo đồng hành cùng sinh viên trong quá trình thực hiện **Thực tập Nhận Thức ngành Công nghệ Thông tin** tại Trường Đại học Đông Á.
    
    Trong suốt kỳ thực tập, tôi sẽ hỗ trợ bạn:
    - Nắm rõ nội dung, yêu cầu và lịch trình thực tập
    - Ghi nhật ký, viết báo cáo đúng chuẩn và đầy đủ
    - Hiểu rõ các mẫu biểu, quy trình đánh giá, kỹ năng nghề nghiệp cần có
    - Định hướng và triển khai bài toán thực tập hiệu quả
    
    Hãy nhập câu hỏi của bạn bên dưới. Tôi luôn sẵn sàng hỗ trợ!
    """


    
    # Hiển thị phần giới thiệu
    st.markdown(intro_text)
    
    # Nếu bật âm thanh, phát giới thiệu
    if st.session_state.get("enable_audio_playback", False):
        render_audio_block(intro_text, autoplay=True)

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
        if st.session_state.get("enable_audio_playback", False):
            render_audio_block(chat["answer"], autoplay=True)
