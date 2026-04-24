"""
app.py — Steamroller: Stutter-to-Fluent Speech Pipeline
Run: streamlit run app.py
"""

import sys
import os
import time
import base64
import io
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Steamroller",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root theme ── */
:root {
    --bg:        #ffffff;
    --surface:   #f7f8fb;
    --surface2:  #eef1f6;
    --border:    #d7dce5;
    --accent:    #ff4d6d;
    --accent2:   #00d4ff;
    --accent3:   #7c3aed;
    --green:     #00a676;
    --yellow:    #ffb400;
    --text:      #111827;
    --muted:     #5b6770;
    --radius:    12px;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.05em; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0a1a2e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(124,58,237,0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(0,212,255,0.10) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    letter-spacing: 0.08em;
    color: var(--accent2);
    line-height: 1;
    margin: 0;
}
.hero-sub {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: var(--accent2);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Pipeline step cards ── */
.pipeline-track {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
    padding: 1rem 1.5rem;
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
}
.pipe-step {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 99px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    transition: all 0.3s;
}
.pipe-step.idle   { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }
.pipe-step.active { background: rgba(0,212,255,0.15); color: var(--accent2); border: 1px solid var(--accent2); box-shadow: 0 0 12px rgba(0,212,255,0.2); }
.pipe-step.done   { background: rgba(0,230,118,0.12); color: var(--green); border: 1px solid var(--green); }
.pipe-step.error  { background: rgba(255,77,109,0.12); color: var(--accent); border: 1px solid var(--accent); }
.pipe-arrow { color: var(--muted); font-size: 0.9rem; font-family: 'Bebas Neue', sans-serif; }

/* ── Section cards ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 0.06em;
    color: var(--accent2);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}

/* ── Text displays ── */
.text-block {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text);
    min-height: 60px;
    word-break: break-word;
}
.text-block.stutter { border-left: 3px solid var(--accent); color: #ffb3c1; }
.text-block.cleaned { border-left: 3px solid var(--yellow); color: #fff8e1; }
.text-block.repaired { border-left: 3px solid var(--green); color: #ccffe8; font-size: 1rem; font-family: 'DM Sans', sans-serif; font-weight: 500; }
.text-block.empty { color: var(--muted); font-style: italic; }

/* ── Agent cards ── */
.agent-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; margin: 1rem 0; }
.agent-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
}
.agent-card.winner { border-color: var(--green); background: rgba(0,230,118,0.05); }
.agent-name { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.05em; font-size: 1.1rem; margin-bottom: 0.5rem; }
.agent-conf { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.4rem; }
.conf-bar { height: 4px; background: var(--border); border-radius: 99px; margin-bottom: 0.6rem; }
.conf-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent3), var(--accent2)); }
.agent-text { font-size: 0.82rem; line-height: 1.5; color: var(--text); }
.agent-expl { font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; font-style: italic; }
.winner-badge { display: inline-block; font-size: 0.65rem; font-weight: 700; background: var(--green); color: #000; border-radius: 99px; padding: 0.1rem 0.5rem; margin-left: 0.5rem; vertical-align: middle; }

/* ── Metric pills ── */
.metrics-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.75rem 0; }
.metric-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    text-align: center;
    min-width: 100px;
}
.metric-val { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; color: var(--accent2); line-height: 1; }
.metric-lbl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent3) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.08em !important;
    font-size: 1.1rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,77,109,0.35) !important;
}

/* ── Sidebar ── */
.sidebar-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 0.1em;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sidebar-tagline { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Audio player ── */
.stAudio { margin-top: 0.5rem; }

/* ── Streamlit overrides ── */
.stSelectbox > div > div, .stSlider > div { color: var(--text) !important; }
label { color: var(--muted) !important; font-size: 0.82rem !important; }
.stAlert { border-radius: var(--radius) !important; }

/* ── Spinner ── */
.stSpinner > div { border-color: var(--accent2) transparent transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎙 STEAMROLLER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Stutter Speech → Fluent Speech</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### ⚙️ Pipeline Settings")

    # whisper_model = st.selectbox(
    #     "Whisper Model",
    #     ["tiny", "base", "small", "medium"],
    #     index=1,
    #     help="Larger = more accurate, slower",
    # )

    language = st.selectbox(
        "Language",
        ["Auto-detect", "English", "Hindi", "Spanish", "French", "German"],
        index=0,
    )
    lang_code_map = {
        "Auto-detect": None,
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
    }
    lang_code = lang_code_map[language]

    tts_speed = st.radio("TTS Speed", ["Normal", "Slow"], horizontal=True)
    tts_slow = tts_speed == "Slow"

    st.markdown("---")
    st.markdown("#### 📋 Pipeline Stages")
    st.markdown(
        """
<div style="font-size:0.8rem; color:#6b6b8a; line-height:2;">
🎤 Mic Input / Upload Audio File<br>
🧠 Whisper ASR Module<br>
🔍 Speech Cleaning using NLP <br>
📋 Multi-Repair Agents <br>
🔊 TTS Synthesis
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🗑️ Clear Session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── Session state defaults ───────────────────────────────────────────────────
defaults = {
    "stage": "idle",          # idle | recording | transcribed | cleaned | repaired | tts_done
    "raw_audio": None,
    "raw_text": "",
    "cleaned_text": "",
    "cleaning_report": None,
    "repair_verdict": None,
    "tts_bytes": None,
    "error": None,
    "input_mode": "upload",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helper: pipeline tracker ─────────────────────────────────────────────────
PIPE_STAGES = [
    ("", "Input"),
    ("", "Whisper"),
    ("", "NLP"),
    ("", "Agents"),
    ("", "TTS"),
]
STAGE_ORDER = ["idle", "recording", "transcribed", "cleaned", "repaired", "tts_done"]


def pipeline_tracker_html(current_stage: str) -> str:
    idx = STAGE_ORDER.index(current_stage) if current_stage in STAGE_ORDER else 0
    html = '<div class="pipeline-track">'
    for i, (icon, label) in enumerate(PIPE_STAGES):
        if i < idx:
            cls = "done"
        elif i == idx - 1 and idx > 0:
            cls = "done"
        else:
            cls = "idle"
        # Map stage index to pipe step
        if i < idx:
            cls = "done"
        elif i == idx and idx > 0:
            cls = "active"
        else:
            cls = "idle"

        html += f'<div class="pipe-step {cls}">{icon} {label}</div>'
        if i < len(PIPE_STAGES) - 1:
            html += '<span class="pipe-arrow">›</span>'
    html += "</div>"
    return html


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">STEAMROLLER</div>
    <div class="hero-sub">Stutter-to-Fluent Speech · Whisper ASR · Multi-Agent AI Repair · TTS Output</div>
</div>
""",
    unsafe_allow_html=True,
)

# Pipeline tracker
st.markdown(pipeline_tracker_html(st.session_state.stage), unsafe_allow_html=True)

# ── Error banner ─────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"⚠️ {st.session_state.error}")
    if st.button("Dismiss"):
        st.session_state.error = None
        st.rerun()

# ═════════════════════════════════════════════
# MAIN LAYOUT — 2 columns
# ═════════════════════════════════════════════
col_left, col_right = st.columns([1, 1], gap="large")

# ────────────────────────────────────────────
# LEFT COLUMN — Input + ASR + NLP
# ────────────────────────────────────────────
with col_left:

    # ── Input Mode ──────────────────────────
    st.markdown(
        '<div class="section-card"><div class="section-title">🎤 Audio Input</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Input method",
        ["Upload Audio File", "Record from Mic"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.input_mode = input_mode

    audio_bytes = None

    if input_mode == "Upload Audio File":
        uploaded = st.file_uploader(
            "Upload WAV / MP3 / M4A / OGG file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            label_visibility="visible",
        )
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes)
            st.session_state.raw_audio = audio_bytes
            st.session_state._upload_filename = uploaded.name
    else:
        st.info("🎙️ Recording happens in your browser. Upload the resulting audio above, or use `sounddevice` recording below.")

        duration = st.slider("Recording duration (seconds)", 3, 30, 8)
        if st.button("⏺  Start Recording", use_container_width=True):
            st.session_state.stage = "recording"
            with st.spinner(f"Recording for {duration}s… speak now!"):
                try:
                    from audio_pipeline import record_audio, numpy_to_wav_bytes
                    audio_np = record_audio(duration_seconds=duration)
                    wav_bytes = numpy_to_wav_bytes(audio_np)
                    st.session_state.raw_audio = wav_bytes
                    st.session_state.stage = "transcribed"  # will transcribe next
                    st.success("✅ Recording complete!")
                    st.rerun()
                except Exception as e:
                    st.session_state.error = f"Recording failed: {e}"
                    st.session_state.stage = "idle"
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────
# RIGHT COLUMN — Agent Repair + TTS
# ────────────────────────────────────────────
with col_right:

    verdict = st.session_state.repair_verdict

    # ── TTS Output ───────────────────────────
    st.markdown(
        '<div class="section-card"><div class="section-title">🔊 TTS Audio Output</div>',
        unsafe_allow_html=True,
    )

    final_text = verdict.final_text if verdict else ""

    tts_text = st.text_area(
        "Text to synthesise (edit if needed):",
        value=final_text,
        height=80,
        label_visibility="visible",
    )

    run_tts = st.button(
        "🔊 Synthesise Speech",
        disabled=not tts_text.strip(),
        use_container_width=True,
    )

    if run_tts:
        with st.spinner("Synthesising audio…"):
            try:
                from tts_engine import synthesise_speech
                audio_out = synthesise_speech(
                    tts_text,
                    backend="gtts",
                    lang=lang_code or "en",
                    slow=tts_slow,
                )
                st.session_state.tts_bytes = audio_out
                st.session_state.stage = "tts_done"
                st.rerun()
            except Exception as e:
                st.session_state.error = f"TTS failed: {e}"

    if st.session_state.tts_bytes:
        st.markdown("**▶ Repaired Audio:**")
        st.audio(st.session_state.tts_bytes, format="audio/mp3")
        st.download_button(
            "⬇️ Download Repaired Audio",
            data=st.session_state.tts_bytes,
            file_name="steamroller_output.mp3",
            mime="audio/mp3",
            use_container_width=True,
        )
    else:
        st.markdown(
            '<div class="text-block empty" style="text-align:center; padding:2rem;">Audio will appear here after synthesis…</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# BOTTOM — Before / After comparison
# ═════════════════════════════════════════════
if st.session_state.raw_text and (st.session_state.repair_verdict or st.session_state.cleaned_text):
    st.markdown("---")
    st.markdown(
        '<h2 style="font-family:Bebas Neue,sans-serif; color:var(--accent2); letter-spacing:0.06em;">📊 BEFORE / AFTER COMPARISON</h2>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔴 Original Stuttered Text**")
        st.markdown(
            f'<div class="text-block stutter">{st.session_state.raw_text}</div>',
            unsafe_allow_html=True,
        )
        words_before = len(st.session_state.raw_text.split())
        st.caption(f"{words_before} words")

    with c2:
        final = (
            st.session_state.repair_verdict.final_text
            if st.session_state.repair_verdict
            else st.session_state.cleaned_text
        )
        st.markdown("**🟢 Repaired Fluent Text**")
        st.markdown(
            f'<div class="text-block repaired">{final}</div>',
            unsafe_allow_html=True,
        )
        words_after = len(final.split())
        reduction = max(0, round((1 - words_after / max(words_before, 1)) * 100))
        st.caption(f"{words_after} words · {reduction}% reduction in stutter artifacts")
