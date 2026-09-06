import streamlit as st
import io
import asyncio
import tempfile
import os
import requests
import numpy as np
import subprocess
import json
import shutil
import re


from PIL import Image, ImageFilter
from deep_translator import GoogleTranslator
import edge_tts
import PyPDF2
from groq import Groq

st.title("AI Student Assistant")

st.sidebar.title("Tools")

tool = st.sidebar.radio("Choose a tool:", [" Chatbot", "📄PDF Summarize", "📸image editor", "🌐Translator", "🎙️ Extract Text from Audio, Video & YouTube", "🎙️ Audio to Text"])

if tool == " Chatbot":
    st.header("Chatbot")
    
    client = Groq(api_key=st.secrets["Groq_API_Key"])
    client = Groq(api_key="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")

    
    audio = st.audio_input("Record...")
    
    prompt = st.chat_input("Write your text here...")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt:
        st.session_state.messages.append({"role" : "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    messages = [{"role": "system", "content": "your are a helpful assistant"}]
                    for msg in st.session_state.messages:
                        messages.append({"role": msg['role'], "content": msg["content"]})
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7
                    )
                    answer = response.choices[0].message.content
                except Exception as e:
                    answer = e
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if audio:
        if st.button("send audio"):
            with st.spinner("converting audio to text..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio.read())
                    tmp_file_path = tmp_file.name
                try:
                    with open(tmp_file_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create (
                            model="whisper-large-v3-turbo",
                            file=os.path.basename(tmp_file_path),
                            response_format="text"
                        )

                finally:
                    os.unlink(tmp_file_path)
            if not transcript or len(transcript.strip()) < 2:
                st.session_state.messages.append({"role": "assistant", "content":transcript})
                with st.chat_message("user"):
                    st.write(f"{transcript}")
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            messages = [{"role": "system", "content": "your are a helpful assistant"}]
                            for msg in st.session_state.messages:
                                messages.append({"role": msg['role'], "content": msg["content"]})
                            response = client.chat.completions.create(
                                model="openai/gpt-oss-120b",
                                messages=messages,
                                max_tokens=500,
                                temperature=0.7
                            )
                            answer = response.choices[0].message.content
                        except Exception as e:
                            answer = e
                        

elif tool == "📄PDF Summarize":
    st.header("📄PDF Summarize")
    client = Groq(api_key=st.secrets["Groq_API_Key"])
    client = Groq(api_key="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")

    uploaded_file = st.file_uploader("أو ارفع ملف PDF", type=["pdf"])
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""
        if pdf_text.strip():
            st.info(f" تم استخراج النص من PDF ({len(pdf_reader.pages)} صفحة)")
            text_from_pdf = pdf_text
        else:
            st.warning(" لم يتم العثور على نص في الملف")
            text_from_pdf = ""
    else:
        text_from_pdf = ""
    text = st.text_area("اكتب النص هنا...", height=200, value=text_from_pdf)
    if st.button(" لخّص!"):
        if len(text.split()) < 10:
            st.warning(" النص قصير جداً!")
        else:
            with st.spinner(" جاري التلخيص..."):
                # كشف اللغة
                arabic_letters = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
                language = "العربية" if arabic_letters > 5 else "الإنجليزية"
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system","content": f"You are a helpful assistant for kids. Summarize the text in 3-4 simple sentences. You MUST respond in {language} only. Do NOT use any other language or characters."},
                        {"role": "user", "content": text}
                    ]
                )
                st.success(response.choices[0].message.content)

elif tool == "📸image editor":
    st.header("📸image editor")



    # ---------------- Filters ---------------- #

    def filter_vintage(img):
        arr = np.array(img, dtype=np.float32)
        arr[:,:,0] = np.clip(arr[:,:,0]*1.1+20,0,255)
        arr[:,:,1] = np.clip(arr[:,:,1]*0.9+10,0,255)
        arr[:,:,2] = np.clip(arr[:,:,2]*0.75,0,255)
        return Image.fromarray(arr.astype(np.uint8))

    def filter_bw(img):
        return img.convert("L").convert("RGB")

    def filter_sharp(img):
        return img.filter(ImageFilter.SHARPEN)

    def filter_blur(img):
        return img.filter(ImageFilter.BLUR)

    def filter_warm(img):
        arr=np.array(img,dtype=np.float32)
        arr[:,:,0]=np.clip(arr[:,:,0]*1.2+20,0,255)
        arr[:,:,1]=np.clip(arr[:,:,1]*1.1+10,0,255)
        arr[:,:,2]=np.clip(arr[:,:,2]*0.85,0,255)
        return Image.fromarray(arr.astype(np.uint8))

    def filter_gray(img):
        return img.convert("L").convert("RGB")

    def filter_edge(img):
        return img.filter(ImageFilter.FIND_EDGES)

    filters={
        "No Filter":None,
        "Vintage":filter_vintage,
        "Black & White":filter_bw,
        "Sharp":filter_sharp,
        "Blur":filter_blur,
        "Warm":filter_warm,
        "Grayscale":filter_gray,
        "Edge Detection":filter_edge
    }

    # ---------------- Upload ---------------- #

    
    

    person_file=st.file_uploader(
        "Upload Person Image",
        type=["jpg","jpeg","png"]
    )

    if person_file:

        

        person=Image.open(person_file).convert("RGBA")
        



        # Make sure person fits background
        

        choice=st.selectbox(
            "Choose Filter",
            list(filters.keys())
        )

        result=person.convert("RGB")

        if filters[choice]:
            result=filters[choice](result)

        col1,col2=st.columns(2)

        st.image(result,caption="Final Image")

        buffer=io.BytesIO()
        result.save(buffer,format="PNG")

        st.download_button(
            "⬇ Download Image",
            data=buffer.getvalue(),
            file_name="FinalImage.png",
            mime="image/png"
    )

elif tool == "🌐Translator":
    st.header("🌐Translator")
    

    lang = {
        "Arabic":"ar",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Korean": "ko"
        
    }

    col1, col2 = st.columns(2)

    with col1:
        source_lang = st.selectbox( "Source Language",list(lang.keys()),index=1)
    with col2:
        target_lang = st.selectbox( "Target Language",list(lang.keys()),index=0)

    text_input = st.text_area("Enter your text", height=150, placeholder="Write your text ... ")

    if st.button("Translate", use_container_width=True): 
        if text_input == "":
            st.warning("Enter your text first ... ")
        else:
            try:
                translted = GoogleTranslator(source=lang[ source_lang], target=lang[target_lang]) . translate(text_input)
                st. text_area("Trasnlated text", value=translted, height=100)
            except:
                st.error("There is an error for connection.please try again!.! ")

    st.markdown(" --- ")
    st.caption("By student !Ebr@him! ")

elif tool == "🎙️ Extract Text from Audio, Video & YouTube":
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    client = Groq(api_key="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")

    def transcribe_audio_file(path):
        with open(path, "rb") as f:
            return client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(os.path.basename(path), f),
                response_format="text"
            )

    def extract_audio_from_video(video_path):
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = audio_file.name
        audio_file.close()
        command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            raise RuntimeError("FFmpeg error: " + result.stderr[-1000:])
        return audio_path

    def parse_json3_sub(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lines = []
        for event in data.get("events", []):
            for seg in event.get("segs", []):
                word = seg.get("utf8", "").strip()
                if word and word != "\n":
                    lines.append(word)
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    def get_yt_transcript(url):
        tmp_dir = tempfile.mkdtemp()
        try:
            output_path = os.path.join(tmp_dir, "subs")
            for lang in ["ar", "en", None]:
                command = [
                    "yt-dlp", "--no-playlist", "--skip-download",
                    "--write-subs", "--write-auto-subs",
                    "--sub-format", "json3", "-o", output_path
                ]
                if lang:
                    command += ["--sub-lang", lang]
                command.append(url)
                subprocess.run(command, capture_output=True, text=True)
                for filename in os.listdir(tmp_dir):
                    if filename.endswith(".json3"):
                        text = parse_json3_sub(os.path.join(tmp_dir, filename))
                        if text:
                            return text
            raise ValueError("No subtitles found for this YouTube video.")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Audio / Video
    st.header("📁 Audio / Video")
    uploaded_file = st.file_uploader(
        "Upload your file",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / 1024**2
        if file_size_mb > 25:
            st.error("File size exceeds 25MB. Please upload a smaller file.")
        elif st.button("🎙️ Extract Text", key="file_btn"):
            temp_file_path = None
            audio_path = None
            try:
                with st.spinner("Extracting text..."):
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_file_path = temp_file.name
                    audio_extensions = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
                    if ext in audio_extensions:
                        audio_path = temp_file_path
                    else:
                        audio_path = extract_audio_from_video(temp_file_path)
                    transcript = transcribe_audio_file(audio_path)
                if not transcript or len(transcript.strip()) < 5:
                    st.error("No text was extracted.")
                else:
                    st.subheader("📄 Extracted Text")
                    st.text_area("Transcript", transcript, height=400)
                    st.download_button(
                        "⬇️ Download Text",
                        transcript,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                if audio_path and audio_path != temp_file_path and os.path.exists(audio_path):
                    os.unlink(audio_path)

    # YouTube
    st.divider()
    st.header("▶️ YouTube")
    st.caption("Enter a YouTube URL to extract its subtitles.")
    yt_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=example"
    )

    def is_youtube_url(url):
        pattern = r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+"
        return re.match(pattern, url.strip()) is not None

    if st.button("🎬 Extract YouTube Text", key="yt_btn"):
        if not yt_url.strip():
            st.warning("Please enter a YouTube URL.")
        elif not is_youtube_url(yt_url):
            st.error("Please enter a valid YouTube URL.")
        else:
            with st.spinner("Extracting YouTube subtitles..."):
                try:
                    transcript = get_yt_transcript(yt_url.strip())
                    if not transcript or len(transcript.strip()) < 5:
                        st.error("No text was extracted from the YouTube video.")
                    else:
                        st.subheader("📄 YouTube Transcript")
                        st.text_area("Transcript", transcript, height=400)
                        st.download_button(
                            "⬇️ Download Text",
                            transcript,
                            file_name="youtube_transcript.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

elif tool == "🎙️ audio to text":
    st.header("🎙️ Audio to Text")
    text = st.text_area("Write text here")

    lang = st.selectbox("Choose language", ["عربي", "English"])

    voices = {
        "عربي": {
            " زارية (امرأة - السعودية)": "ar-SA-ZariyahNeural",
            " شاكر (رجل - مصر)": "ar-EG-ShakirNeural",
            " سلمى (امرأة - مصر)": "ar-EG-SalmaNeural",
        },
        "English": {
            " غاي (رجل - أمريكا)": "en-US-GuyNeural",
            " جيني (امرأة - أمريكا)": "en-US-JennyNeural",
            " رايان (رجل - بريطانيا)": "en-GB-RyanNeural",
            " ليبي (امرأة - بريطانيا)": "en-GB-LibbyNeural",
        },
    }
    voice_label = st.selectbox("Choose Voice", list(voices[lang].keys()))

    voice_id = voices[lang][voice_label]

    async def generate_audio(text, voice):
        communicate = edge_tts.Communicate(text=text, voice=voice)

        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        return b"".join(audio_chunks)

    if st.button("Convert"):
        if text:
            audio_bytes = asyncio.run(generate_audio(text, voice_id))
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.warning("write your text first")
else:
    st.warning("Please select a tool from the sidebar.")


