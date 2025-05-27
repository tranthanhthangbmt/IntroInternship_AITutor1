import asyncio
import base64
import streamlit as st
import edge_tts
import uuid
import os


def generate_audio_filename(text, voice="vi-VN-HoaiMyNeural"):
    """
    Tạo tên file âm thanh duy nhất từ đoạn text.
    """
    return f"audio_{hash(text + voice)}.mp3"


async def generate_audio_async(text, voice="vi-VN-HoaiMyNeural"):
    """
    Tạo file âm thanh từ văn bản sử dụng Microsoft Edge TTS.
    """
    filename = generate_audio_filename(text, voice)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)
    return filename


def play_audio(text, voice="vi-VN-HoaiMyNeural"):
    """
    Phát âm đoạn văn bản trong Streamlit.
    """
    if not text.strip():
        st.warning("⚠️ Không có nội dung để phát âm.")
        return

    filename = asyncio.run(generate_audio_async(text, voice))

    with open(filename, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay controls>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Trình duyệt của bạn không hỗ trợ audio.
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        # Chèn JS để chỉ phát audio cuối cùng
        js_code = """
        <script>
        document.addEventListener("DOMContentLoaded", function() {
            const audios = document.querySelectorAll("audio");
            audios.forEach((audio, index) => {
                if (index !== audios.length - 1) {
                    audio.pause();
                    audio.currentTime = 0;
                } else {
                    audio.play();
                }
            });
        });
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)

def generate_and_encode_audio(text, voice="vi-VN-HoaiMyNeural"):
    """
    Sinh file audio từ văn bản, encode base64 để nhúng HTML
    """
    if not text.strip():
        st.warning("⚠️ Không có nội dung hợp lệ để phát âm.")
        return ""

    async def _generate_audio(text, filename, voice):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(filename)
        except Exception as e:
            st.error(f"❌ Lỗi khi tạo audio: {e}")
            return False
        return True

    temp_filename = f"temp_{uuid.uuid4().hex}.mp3"
    success = asyncio.run(_generate_audio(text, temp_filename, voice))

    if not success or not os.path.exists(temp_filename):
        st.error("❌ Không thể tạo file âm thanh.")
        return ""

    try:
        with open(temp_filename, "rb") as f:
            audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode()
        return b64
    finally:
        os.remove(temp_filename)
# def generate_and_encode_audio(text, voice="vi-VN-HoaiMyNeural"):
#     """
#     Sinh file audio từ văn bản, encode base64 để nhúng HTML
#     """
#     async def _generate_audio(text, filename, voice):
#         communicate = edge_tts.Communicate(text, voice)
#         await communicate.save(filename)

#     temp_filename = f"temp_{uuid.uuid4().hex}.mp3"
#     asyncio.run(_generate_audio(text, temp_filename, voice))

#     with open(temp_filename, "rb") as f:
#         audio_bytes = f.read()
#         b64 = base64.b64encode(audio_bytes).decode()

#     os.remove(temp_filename)
#     return b64


# Gợi ý giọng đọc tiếng Việt:
# "vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"
# Xem thêm danh sách voice tại https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support
