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
st.set_option("client.showErrorDetails", False)
# Sidebar – hiển thị logo và thông tin
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tranthanhthangbmt/AITutor_Gemini/main/LOGO_UDA_2023_VN_EN_chuan2.png", width=180)
    if "enable_audio_playback" not in st.session_state:
        st.session_state["enable_audio_playback"] = False  # mặc định bật
    
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
    Xin chào, tôi là **Tutor AI – Trợ lý ảo đồng hành cùng bạn trong kỳ Thực tập Nhận Thức. Tôi sẽ hỗ trợ bạn trong suốt quá trình thực tập với các vai trò:
    
    - Giải đáp về nội dung, yêu cầu và lịch trình thực tập
    - Hướng dẫn cách ghi **nhật ký**, viết **báo cáo**, sử dụng **mẫu biểu** đúng chuẩn
    - Cung cấp kiến thức nền tảng về **văn hóa doanh nghiệp CNTT**, kỹ năng làm việc chuyên nghiệp
    - Giới thiệu về **chuyển đổi số trong doanh nghiệp**, vai trò của **AI, dữ liệu và tự động hóa**
    - Gợi ý và hướng dẫn đề tài thực tế như: ứng dụng AI hỗ trợ nghiệp vụ, chatbot nội bộ, quản lý tài liệu số, phân tích dữ liệu khách hàng, hệ thống phản hồi thông minh...
    
    Hãy đặt câu hỏi bên dưới – tôi luôn sẵn sàng hỗ trợ bạn!
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
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Tạo prompt cho Gemini
    prompt = f"""
    Bạn là trợ lý AI dành cho sinh viên CNTT. 
    Hãy trả lời ngắn gọn, rõ ràng và thực tế, chỉ tập trung vào ý chính trong ngữ cảnh sau:
    
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
