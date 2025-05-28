#versions:
# test_RAGStreamlit_v3_10h56_27.5.2025.py sửa lại audio cho streamlit local
#test_RAGStreamlit_v3C_15h13_27.5.2025.py sử lại phầu tải file RAG từ Google Drive
#test_RAGStreamlit_v4_22h13_27.5.2025.py: huấn luyện tiếng Anh, hỏi bằng tiếng Việt
#test_RAGStreamlit_v4A_24h49_28.5.2025.py: thêm history messages của mỗi lần người dùng prompt
#test_RAGStreamlit_v4C_09h09_28.5.2025.py: thêm CrossEncoder/ms-marco-MiniLM-L-6-v2, chỉ có mỗi vi
#test_RAGStreamlit_v4D_09h25_28.5.2025.py: bổ xung thên en,vn trong rank
#---------------------------
#run: streamlit run test_RAGStreamlit_v4D_09h25_28.5.2025.py
#run2: streamlit run --server.fileWatcherType none test_RAGStreamlit_v4D_09h25_28.5.2025.py
#------------------

# import os
# os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel, configure

#audio
import edge_tts
#from modules.audio_module import generate_and_encode_audio
from modules.audio_module import render_audio_block

# Thêm thư viện và hàm rerank_with_crossencoder
from sentence_transformers import CrossEncoder

def shorten_passage(text, max_char=1200, overlap=200):
    if len(text) <= max_char:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_char, len(text))
        chunks.append(text[start:end].strip())
        start += max_char - overlap
    return chunks

def rerank_with_crossencoder(query, docs, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    model = CrossEncoder(model_name)
    scored_docs = []
    for doc in docs:
        parts = shorten_passage(doc.page_content)
        for part in parts:
            score = model.predict([(query, part)])[0]
            scored_docs.append((score, part))
    ranked = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    return ranked

# def rerank_with_crossencoder(query, docs, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
#     model = CrossEncoder(model_name)
#     scored_docs = []
#     for doc in docs:
#         score = model.predict([(query, doc.page_content)])[0]
#         scored_docs.append((score, doc.page_content))
#     ranked = sorted(scored_docs, key=lambda x: x[0], reverse=True)
#     return ranked

def summarize_chat_history(history, max_turns=3):
    if not history:
        return ""

    recent_turns = history[-max_turns:]
    summary = ""
    for turn in recent_turns:
        summary += f"Sinh viên: {turn['question']}\n"
        summary += f"Tutor: {turn['answer']}\n"
    return summary.strip()

# ⚠️ Cấu hình API key Gemini (thay bằng key thực tế hoặc dùng dotenv)
#configure(api_key="AIzaSyB23c7ttZ-RWiqj9O4dY82NutHsjz0N45s")

import streamlit as st
from google.generativeai import configure

configure(api_key=st.secrets["GEMINI_API_KEY"])
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ Thiếu khóa API Gemini. Vui lòng khai báo trong Settings > Secrets.")
    st.stop()

# import time

# long_text = "Xin chào, tôi là Tutor AI..." * 30  # tạo đoạn dài
# start = time.time()
# b64 = generate_and_encode_audio(long_text)
# print("Thời gian tạo âm:", time.time() - start)

# Khởi tạo model Gemini
model = GenerativeModel("models/gemini-2.0-flash-lite")

import os
import gdown

# Tạo đoạn mã Python tải thư mục FAISS từ Google Drive
# if not os.path.exists("IntroInternshipRAG_MiniLM_L3_withEbooks/faiss_index"):
#     gdown.download_folder(
#         url="https://drive.google.com/drive/folders/1GZF0Aas4n7m1kkd-MmoCGyR7oH77xXWE",
#         output="IntroInternshipRAG_MiniLM_L3_withEbooks",
#         quiet=False,
#         use_cookies=False
#     )

DATA_OUTPUT_FOLDER = "IntroInternshipRAG_MiniLM_L3_allRef"
if not os.path.exists(DATA_OUTPUT_FOLDER + "/faiss_index"):
    gdown.download_folder(
        url="https://drive.google.com/drive/folders/13tzW43t7MuqtOP8-xYy-7lRYgEgtNTVX",
        output=DATA_OUTPUT_FOLDER,
        quiet=False,
        use_cookies=False
    )

# Dùng mô hình nhỏ hơn
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Load FAISS index từ thư mục mới
#DATA_OUTPUT_FOLDER = "IntroInternshipRAG_MiniLM_L3_allRef"

vectorstore = FAISS.load_local(
    #"IntroInternshipRAG_MiniLM_L3/faiss_index",
    DATA_OUTPUT_FOLDER+ "/faiss_index",
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
    **Trợ lý ảo đồng hành cùng sinh viên năm 2 trong kỳ thực tập nhận thức ngành Công nghệ Thông tin tại doanh nghiệp**
    
    💡 Bạn sẽ được hỗ trợ về:
    - Nội dung, yêu cầu và lịch trình thực tập
    - Hướng dẫn ghi nhật ký, viết báo cáo và sử dụng biểu mẫu
    - Kiến thức về văn hóa doanh nghiệp CNTT, kỹ năng làm việc chuyên nghiệp
    - Tổng quan về chuyển đổi số, vai trò của AI, dữ liệu và tự động hóa
    - Gợi ý đề tài thực tế tại doanh nghiệp
    
    📍 Vui lòng nhập câu hỏi bên dưới để được hỗ trợ nhanh chóng, chính xác và dễ hiểu.
    
    ---
    #### ℹ️ Thông tin hệ thống
    - Nền tảng: Gemini + FAISS
    - Phiên bản: 1.0.0
    
    ---
    © 2025 Khoa Công nghệ Thông tin, Đại học Đông Á. Mọi quyền được bảo lưu.
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
    # Cách 1: Truy xuất ngữ cảnh liên quan
    # docs = vectorstore.similarity_search(query, k=8)
    # context = "\n".join([doc.page_content for doc in docs])
    
    #cách 2:----------------------------
    # from deep_translator import GoogleTranslator
    # from langdetect import detect

    # lang = detect(query)

    # if lang == "vi":
    #     en_query = GoogleTranslator(source='vi', target='en').translate(query)
    # else:
    #     en_query = query  # đã là tiếng Anh

    # # docs = vectorstore.similarity_search(en_query, k=8)
    # # context = "\n".join([doc.page_content for doc in docs])
    
    # #cách 3:----------------------------
    # docs_vi = vectorstore.similarity_search(query, k=4)
    # query_en = GoogleTranslator(source="vi", target="en").translate(query)
    # docs_en = vectorstore.similarity_search(query_en, k=4)

    # # Kết hợp loại bỏ trùng
    # context = "\n\n".join({doc.page_content for doc in docs_vi + docs_en})
    
    #cách 4:----------------------------
    from deep_translator import GoogleTranslator
    from langdetect import detect

    # 1. Phát hiện ngôn ngữ câu hỏi gốc
    lang = detect(query)

    # 2. Luôn tạo cả 2 bản: VI và EN
    if lang == "vi":
        query_vi = query
        query_en = GoogleTranslator(source="vi", target="en").translate(query)
    else:
        query_en = query
        query_vi = GoogleTranslator(source="en", target="vi").translate(query)

    # 3. Truy xuất context từ cả 2 câu hỏi
    docs_vi = vectorstore.similarity_search(query_vi, k=8) #4
    docs_en = vectorstore.similarity_search(query_en, k=8) #4
    # print("🟩 Context từ tiếng Việt:")
    # for doc in docs_vi:
    #     print(doc.page_content)

    # print("\n🟦 Context từ tiếng Anh:")
    # for doc in docs_en:
    #     print(doc.page_content)
    
    # 4. Ghép cả 2 context (không loại bỏ trùng)
    #context = "\n\n".join({doc.page_content for doc in docs_vi + docs_en})
    
    top_k_docs = docs_vi + docs_en
    #reranked = rerank_with_crossencoder(query_vi, top_k_docs)
    # Truy vấn tiếng Việt + tiếng Anh
    reranked_vi = rerank_with_crossencoder(query_vi, top_k_docs)
    reranked_en = rerank_with_crossencoder(query_en, top_k_docs)

    #Lấy riêng từng phần top 3 + top 3
    top_3_vi = sorted(reranked_vi, key=lambda x: x[0], reverse=True)[:3]
    top_3_en = sorted(reranked_en, key=lambda x: x[0], reverse=True)[:3]


    # Gộp và chọn top-k
    # combined = reranked_vi + reranked_en
    # combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # context = "\n\n".join([chunk for _, chunk in combined[:3]])
    
    combined = top_3_vi + top_3_en
    context = "\n\n".join([chunk for _, chunk in combined])

    # In log kiểm tra
    for i, (score, text) in enumerate(combined):
        print(f"🔥 TOP {i+1} – Score: {score:.3f}")
        #print(text[:300], "\n")
        print(text, "\n")

    # Tạo prompt cho Gemini
    #Tạo history_summary trước khi tạo prompt
    history_summary = summarize_chat_history(st.session_state.chat_history, max_turns=2)
    
    prompt = f"""
    # Vai trò: Bạn là một trợ lý AI thân thiện, đang hỗ trợ sinh viên năm 2 ngành CNTT trong kỳ thực tập.

    # Ngữ cảnh: Dưới đây là phần hội thoại gần đây giữa sinh viên và bạn:
        - {history_summary}

    # Yêu cầu đầu ra: Hãy trả lời câu hỏi dưới đây dựa trên thông tin tài liệu nếu có:
        - Giải thích rõ ràng, dễ hiểu
        - Trả lời bằng tiếng Việt
        - Nếu thông tin có cả tiếng Anh và tiếng Việt, bạn có thể kết hợp cả hai
        - Tránh từ chuyên môn nếu không cần thiết; nếu có, hãy giải thích thêm hoặc đưa ví dụ minh họa
        - Ưu tiên sử dụng thông tin từ tài liệu tham khảo nếu có liên quan
        - Nếu thông tin trong tài liệu không đủ hoặc không rõ, bạn có thể sử dụng kiến thức nền tảng từ bên ngoài để đưa ra câu trả lời phù hợp và chính xác
        - Đảm bảo câu trả lời không vượt quá 1000 ký tự
        - Tránh lặp lại lời chào hoặc mở đầu như "Chào bạn", "Rất vui được hỗ trợ..." – hãy đi thẳng vào nội dung chính.

    # Ngữ cảnh truy xuất từ tài liệu:
        - {context}

    # Câu hỏi của sinh viên:
        - {query}
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
            #render_audio_block(chat["answer"], autoplay=True)
            render_audio_block(chat["answer"], autoplay=st.session_state.get("enable_audio_playback", False))

