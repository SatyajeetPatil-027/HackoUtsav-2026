"""
app.py — Steamroller End-to-End Fast Pipeline
Run: streamlit run app.py

Flow:
Upload stuttered audio -> Whisper ASR -> NLP cleaning -> local repair agents -> repaired audio output.
"""

import html
import time

import streamlit as st

st.set_page_config(page_title="Steamroller", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
  --bg:#f4fbff; --surface:#ffffff; --border:#cfe6ee;
  --accent:#08a6c8; --accent2:#13c6b8; --text:#0f172a;
  --green:#00896f; --radius:16px;
}
html, body, [data-testid="stAppViewContainer"] { background:var(--bg)!important; color:var(--text)!important; }
[data-testid="stSidebar"] { background:#ffffff!important; border-right:1px solid var(--border); }
#MainMenu, footer, header { visibility:hidden; }
.hero { text-align:center; padding:1.2rem 1rem 1.6rem; }
.hero h1 { margin:0; font-size:2.4rem; color:#102a43; font-weight:800; }
.hero p { color:#486581; max-width:700px; margin:.5rem auto 0; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.35rem; box-shadow:0 8px 22px rgba(15,23,42,.05); margin-bottom:1rem; }
.card-title { font-weight:800; color:#0f172a; letter-spacing:.04em; margin-bottom:.8rem; text-transform:uppercase; }
.text-block { background:#f8fdff; color:#0f172a; border:1px solid #d9eef5; border-left:5px solid var(--accent); border-radius:12px; padding:1rem; min-height:90px; line-height:1.7; font-size:1rem; word-break:break-word; }
.text-block.final { border-left-color:var(--green); background:#f5fffc; color:#052e2b; font-weight:600; }
.text-block.empty { color:#64748b; font-style:italic; font-weight:400; }
.stButton > button { background:linear-gradient(135deg,var(--accent),var(--accent2))!important; color:white!important; border:0!important; border-radius:12px!important; font-weight:800!important; padding:.75rem 1rem!important; }
.stDownloadButton > button { border-radius:12px!important; font-weight:700!important; }
[data-testid="stFileUploader"] section { background:#f8fdff!important; border:2px dashed #9bd8e8!important; border-radius:16px!important; }
.small-note { color:#64748b; font-size:.85rem; line-height:1.5; }
.status-pill { display:inline-block; background:#e7fbf8; color:#006b5f; border:1px solid #baf0e7; border-radius:999px; padding:.25rem .7rem; font-weight:700; font-size:.82rem; margin:.15rem .25rem .15rem 0; }
</style>
""", unsafe_allow_html=True)


def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def html_text(text: str) -> str:
    return html.escape(text or "")


with st.sidebar:
    st.markdown("## 🎙️ STEAMROLLER")
    st.caption("Fast stuttered audio → fluent audio")
    st.markdown("---")
    st.markdown("### Speed Settings")
    whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        index=0,
        help="tiny is fastest and usually keeps total processing under 1 minute for short clips."
    )
    language = st.selectbox("Language", ["Auto-detect", "English", "Hindi", "Spanish", "French", "German"], index=0)
    lang_code_map = {"Auto-detect": None, "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
    lang_code = lang_code_map[language]
    tts_backend = st.radio("Output voice", ["System voice (faster, less female)", "Google TTS"], index=0)
    st.markdown("---")
    st.markdown("<div class='small-note'>For speed, upload clips around 5–30 seconds. Longer audio will naturally take more time.</div>", unsafe_allow_html=True)
    if st.button("Clear Session", use_container_width=True):
        reset_session()
        st.rerun()

for key, default in {
    "raw_audio": None,
    "raw_text": "",
    "final_text": "",
    "tts_bytes": None,
    "tts_format": "audio/wav",
    "timings": {},
    "error": None,
    "filename": "audio.wav",
}.items():
    st.session_state.setdefault(key, default)

st.markdown("""
<div class='hero'>
  <h1>FLUENT FORCE — Speech Repair Assistant</h1>
  <p>Upload a stuttered audio file. The system transcribes it, repairs the speech text, and returns repaired fluent audio.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.error:
    st.error(st.session_state.error)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='card'><div class='card-title'>Voice Input</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag and drop your stuttered audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if uploaded:
        audio_bytes = uploaded.getvalue()
        st.session_state.raw_audio = audio_bytes
        st.session_state.filename = uploaded.name
        st.audio(audio_bytes)

    process = st.button("Repair Speech", disabled=not bool(st.session_state.raw_audio), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>Stuttered Text</div>", unsafe_allow_html=True)
    if st.session_state.raw_text:
        st.markdown(f"<div class='text-block'>{html_text(st.session_state.raw_text)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='text-block empty'>The transcription will appear here after processing.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><div class='card-title'>Repaired Output</div>", unsafe_allow_html=True)
    if st.session_state.final_text:
        st.markdown(f"<div class='text-block final'>{html_text(st.session_state.final_text)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='text-block empty'>The final repaired sentence will appear here.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>Repaired Audio</div>", unsafe_allow_html=True)
    if st.session_state.tts_bytes:
        st.audio(st.session_state.tts_bytes, format=st.session_state.tts_format)
        ext = "mp3" if st.session_state.tts_format == "audio/mp3" else "wav"
        st.download_button("Download Repaired Audio", data=st.session_state.tts_bytes, file_name=f"steamroller_repaired.{ext}", mime=st.session_state.tts_format, use_container_width=True)
    else:
        st.markdown("<div class='text-block empty'>Audio output will appear here.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if process:
    st.session_state.error = None
    st.session_state.raw_text = ""
    st.session_state.final_text = ""
    st.session_state.tts_bytes = None
    st.session_state.timings = {}

    overall_start = time.perf_counter()
    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("1/4 Transcribing audio with fast Whisper settings...")
        t0 = time.perf_counter()
        from audio_pipeline import transcribe_uploaded_file
        raw_text, _ = transcribe_uploaded_file(
            st.session_state.raw_audio,
            filename=st.session_state.filename,
            model_size=whisper_model,
            language=lang_code,
        )
        st.session_state.raw_text = raw_text
        st.session_state.timings["ASR"] = round(time.perf_counter() - t0, 2)
        progress.progress(35)

        status.info("2/4 Cleaning stutter patterns...")
        t0 = time.perf_counter()
        from nlp_cleaner import clean_asr_output
        report = clean_asr_output(raw_text)
        st.session_state.timings["NLP"] = round(time.perf_counter() - t0, 2)
        progress.progress(55)

        status.info("3/4 Repairing fluency with local agents...")
        t0 = time.perf_counter()
        from repair_agents import run_multi_agent_repair
        verdict = run_multi_agent_repair(report.cleaned, original_text=raw_text)
        st.session_state.final_text = verdict.final_text
        st.session_state.timings["Repair"] = round(time.perf_counter() - t0, 2)
        progress.progress(75)

        status.info("4/4 Generating repaired audio...")
        t0 = time.perf_counter()
        from tts_engine import synthesise_speech
        backend = "pyttsx3" if tts_backend.startswith("System") else "gtts"
        st.session_state.tts_bytes = synthesise_speech(
            st.session_state.final_text,
            backend=backend,
            lang=lang_code or "en",
            slow=False,
        )
        st.session_state.tts_format = "audio/wav" if backend == "pyttsx3" else "audio/mp3"
        st.session_state.timings["TTS"] = round(time.perf_counter() - t0, 2)
        st.session_state.timings["Total"] = round(time.perf_counter() - overall_start, 2)
        progress.progress(100)
        status.success(f"Repair complete in {st.session_state.timings['Total']} seconds.")
        time.sleep(0.5)
        st.rerun()
    except Exception as exc:
        st.session_state.error = f"Processing failed: {exc}"
        st.rerun()

if st.session_state.timings:
    st.markdown("<div class='card'><div class='card-title'>Processing Time</div>", unsafe_allow_html=True)
    st.markdown("".join(f"<span class='status-pill'>{k}: {v}s</span>" for k, v in st.session_state.timings.items()), unsafe_allow_html=True)
    st.markdown("<div class='small-note'>True voice cloning is not included because it needs a separate voice-cloning model and a speaker reference sample. This version uses fast system TTS and avoids the default female Google voice where possible.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
