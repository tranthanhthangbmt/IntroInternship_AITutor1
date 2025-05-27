import asyncio
import base64
import streamlit as st
import edge_tts
import uuid
import os
import textwrap
import concurrent.futures


def split_text(text, max_chars=300):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para.strip() + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(para) > max_chars:
                wrapped = textwrap.wrap(para, max_chars)
                chunks.extend(wrapped)
                current_chunk = ""
            else:
                current_chunk = para.strip() + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

async def generate_audio_async(text, filename, voice="vi-VN-HoaiMyNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)


def generate_and_encode_audio_parallel(text, voice="vi-VN-HoaiMyNeural"):
    if not text.strip():
        st.warning("⚠️ Không có nội dung hợp lệ để phát âm.")
        return ""

    parts = split_text(text)
    temp_paths = []

    async def synthesize_parts():
        tasks = []
        for i, part in enumerate(parts):
            temp_path = f"temp_{uuid.uuid4().hex}_{i}.mp3"
            temp_paths.append(temp_path)
            tasks.append(generate_audio_async(part, temp_path, voice))
        await asyncio.gather(*tasks)

    asyncio.run(synthesize_parts())

    combined_path = f"combined_{uuid.uuid4().hex}.mp3"
    with open(combined_path, "wb") as outfile:
        for path in temp_paths:
            with open(path, "rb") as infile:
                outfile.write(infile.read())

    with open(combined_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    for path in temp_paths:
        os.remove(path)
    os.remove(combined_path)

    return b64

def generate_and_encode_audio_stable(text, voice="vi-VN-HoaiMyNeural"):
    if not text.strip():
        st.warning("⚠️ Không có nội dung hợp lệ để phát âm.")
        return ""

    parts = split_text(text, max_chars=300)
    temp_paths = []

    # Tạo từng file âm thanh một cách tuần tự
    for i, part in enumerate(parts):
        temp_path = f"temp_{uuid.uuid4().hex}_{i}.mp3"
        temp_paths.append(temp_path)
        try:
            asyncio.run(generate_audio_async(part, temp_path, voice))
        except Exception as e:
            st.error(f"❌ Lỗi khi tạo phần âm thanh: {e}")
            return ""

    # Ghép file âm thanh
    combined_path = f"combined_{uuid.uuid4().hex}.mp3"
    with open(combined_path, "wb") as outfile:
        for path in temp_paths:
            with open(path, "rb") as infile:
                outfile.write(infile.read())

    # Encode base64
    with open(combined_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # Xoá file tạm
    for path in temp_paths:
        os.remove(path)
    os.remove(combined_path)

    return b64

def render_audio_block(text: str, autoplay=True):
    import time
    start = time.time()

    try:
        #b64 = generate_and_encode_audio_parallel(text)
        b64 = generate_and_encode_audio_stable(text)
    except Exception as e:
        st.error(f"⚠️ Lỗi phát âm thanh: {e}")
        return

    end = time.time()
    #st.caption(f"⏱️ Đã tạo âm thanh trong {round(end - start, 2)} giây")

    audio_id = f"audio_{int(time.time())}"
    audio_html = f"""
    <audio id="{audio_id}" controls {"autoplay" if autoplay else ""}>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Trình duyệt của bạn không hỗ trợ phát âm thanh.
    </audio>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const audio = document.getElementById("{audio_id}");
            if (audio) {{
                audio.play().catch(e => {{
                    console.warn("⚠️ Không thể tự phát audio:", e);
                }});
            }}
        }});
    </script>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
