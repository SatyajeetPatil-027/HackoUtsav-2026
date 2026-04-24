# 🚂 Steamroller

<div align="center">

**HackoUtsav-2026 · 24-Hour Hackathon Submission**

*An end-to-end intelligent audio processing system — transcription, NLP cleaning, self-healing repair agents, and voice cloning, unified under a single master orchestrator.*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-22c55e?style=for-the-badge)

</div>

---

## 🏆 What We Built in 24 Hours

> **Steamroller** is a fully functional, multi-agent audio intelligence platform built from scratch in 24 hours. It takes raw speech — live microphone or uploaded audio — and transforms it through a pipeline of specialized AI agents: transcription, NLP cleaning, self-repair, and voice synthesis. The result is clean, structured, speaker-cloned audio output, all accessible through an interactive Streamlit interface.

---

## 👥 Team Members

| Role | Name | Responsibility |
|------|------|----------------|
| 🧠 **NLP Processing** | Pranali Sawant | `nlp_cleaner.py` — transcript cleaning & sentence tokenization |
| 🔧 **Repair Agents** | Sanika Sathe | `repair_agents.py` / `repair_agents_runner.py` — audio & text quality correction |
| 🤖 **Master Agent & Voice Clone** | Satyajeet Patil | `master_agent.py`, `tts_engine.py` — orchestration & voice cloning |
| 🎨 **Streamlit UI** | Aditya Suryakar | `app.py` — full interactive frontend |

---

## 🧩 System Architecture

```
                        ┌─────────────────────────────────┐
                        │         master_agent.py          │
                        │     (Pipeline Orchestrator)      │
                        └────────────┬────────────────────┘
                                     │
            ┌────────────────────────┼───────────────────────┐
            ▼                        ▼                        ▼
   ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
   │ audio_pipeline  │───▶│   nlp_cleaner    │───▶│   repair_agents      │
   │     .py         │    │      .py         │    │     _runner.py       │
   │ (Whisper STT)   │    │ (NLTK Tokenizer) │    │ (Quality Correction) │
   └─────────────────┘    └──────────────────┘    └──────────┬───────────┘
                                                             │
                                                             ▼
                                                  ┌──────────────────┐
                                                  │  tts_engine.py   │
                                                  │ (Voice Cloning & │
                                                  │  TTS Synthesis)  │
                                                  └──────────────────┘
                                                             │
                                                             ▼
                                                  ┌──────────────────┐
                                                  │     app.py       │
                                                  │  (Streamlit UI)  │
                                                  └──────────────────┘
```

---

## ✅ Key Features & Achievements

### 🎙️ Voice Cloning — Fully Implemented
The `tts_engine.py` module delivers complete speaker voice cloning:
- Clones a target speaker's voice from a short reference audio sample (as little as 5–10 seconds)
- Synthesizes new, natural-sounding speech in the cloned voice
- Playback and download of cloned voice output directly from the Streamlit UI
- Supports multiple speaker profiles within a single session

### ⚡ Real-Time Audio Input
No file prep required — speak directly into the app:
- Live microphone capture via `sounddevice` and `soundfile`
- Audio is buffered in real time and handed to the Whisper pipeline instantly
- Works alongside file upload for pre-recorded `.wav` files
- Configurable sample rate and chunk duration

### 🚀 Optimized Output Speed
Multiple performance improvements applied during the hackathon:
- Parallel batch processing for handling multiple audio files simultaneously
- Whisper model is warm-cached on startup — no repeated load overhead
- NLTK tokenizer pre-initializes to eliminate per-file setup delays
- Progressive streaming output — transcript results appear in the UI as each file completes

### 🤖 Master Agent Orchestration
`master_agent.py` acts as the central brain:
- Coordinates all pipeline stages with automatic inter-agent communication
- Handles errors at each stage and triggers repair agents when quality thresholds are breached
- Single command to run the entire system end-to-end

### 🔧 Self-Healing Repair Agents
`repair_agents.py` continuously monitors output quality:
- Detects transcription anomalies and low-confidence segments
- Applies correction passes before text reaches NLP or TTS stages
- Can be triggered automatically by the master agent or run standalone

---

## 📁 Folder Structure

```
Steamroller/
│
├── __pycache__/                   # Python bytecode cache (auto-generated, do not commit)
│   ├── audio_pipeline.cpython-3...
│   ├── nlp_cleaner.cpython-3...
│   ├── repair_agents.cpython-3...
│   └── tts_engine.cpython-3...
│
├── app.py                         # 🎨 Streamlit UI — main entry point for the app
├── audio_pipeline.py              # 🎵 Stage 1 — Audio ingestion & Whisper transcription
├── master_agent.py                # 🤖 Orchestrator — coordinates all agents
├── nlp_cleaner.py                 # 🧠 Stage 2 — NLP cleaning & sentence tokenization
├── repair_agents.py               # 🔧 Stage 3 — Audio/text repair & quality correction
├── repair_agents_runner.py        # ▶️  Standalone runner for repair agents
├── tts_engine.py                  # 🔊 Stage 4 — TTS synthesis & voice cloning
├── requirements.txt               # 📦 All Python dependencies
└── README.md                      # 📄 This file
```

---

## 🔧 Prerequisites

### 1. Python 3.8+
```bash
python --version
```
Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. FFmpeg
Required by OpenAI Whisper for audio decoding.

| OS | Install |
|----|---------|
| **Windows** | Download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) → extract to `C:\ffmpeg` → add `C:\ffmpeg\bin` to system PATH |
| **macOS** | `brew install ffmpeg` |
| **Ubuntu/Debian** | `sudo apt install ffmpeg` |

```bash
# Verify
ffmpeg -version
```

### 3. Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option A — Full Pipeline (Recommended)

```bash
# 1. Navigate to project
cd Steamroller

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything via master agent
python master_agent.py
```

### Option B — Interactive UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Running Individual Agents

Each module can be run independently for testing or debugging:

```bash
# Stage 1 — Transcribe audio files
python audio_pipeline.py

# Stage 2 — Clean & tokenize transcripts
python nlp_cleaner.py

# Stage 3 — Run repair agents on output
python repair_agents_runner.py

# Stage 4 — TTS & voice cloning
python tts_engine.py
```

> ⚠️ Whisper runs on CPU by default — processing time scales with audio length and model size.

---

## ⚙️ Configuration & Customization

### Switch Whisper Model (Accuracy vs Speed)

Edit `audio_pipeline.py`:

```python
# Faster ◄──────────────────────────────────► More accurate
# "tiny"   "base"   "small"   "medium"   "large"
model = whisper.load_model("base")
```

### Real-Time Microphone Settings

Edit `app.py`:

```python
SAMPLE_RATE = 16000   # Hz — Whisper is optimized for 16kHz
CHUNK_DURATION = 5    # Seconds per recording chunk
```

---

## 📊 Data Flow Example

```
Input:  Raw speech from mic or .wav file
           │
           ▼  audio_pipeline.py  (OpenAI Whisper)
        Transcript: "Hello my name is John I work in software"
           │
           ▼  nlp_cleaner.py  (NLTK)
        Tokens: ["Hello my name is John.", "I work in software."]
           │
           ▼  repair_agents.py  (Quality check & correction)
        Cleaned: ["Hello, my name is John.", "I work in software."]
           │
           ▼  tts_engine.py  (Voice clone synthesis)
        Output: cloned_voice_output.wav  ✅
```

---

## 🔗 Tech Stack

| Library | Purpose |
|---------|---------|
| [openai-whisper](https://github.com/openai/whisper) | State-of-the-art speech-to-text |
| [nltk](https://www.nltk.org/) | Sentence tokenization |
| [streamlit](https://streamlit.io/) | Interactive web UI |
| [sounddevice](https://python-sounddevice.readthedocs.io/) | Real-time mic capture |
| [soundfile](https://pysoundfile.readthedocs.io/) | Audio file I/O |
| [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) | FFmpeg bindings |
| [torch](https://pytorch.org/) | Whisper model backend |

---

## 🎯 Hackathon Deliverables — Completion Checklist

| Feature | Status |
|---------|--------|
| Audio ingestion & Whisper transcription | ✅ Complete |
| NLP cleaning & sentence tokenization | ✅ Complete |
| Self-healing repair agents | ✅ Complete |
| Voice cloning & TTS synthesis | ✅ Complete |
| Real-time microphone input | ✅ Complete |
| Master agent orchestration | ✅ Complete |
| Streamlit interactive UI | ✅ Complete |
| Parallel batch processing & speed optimization | ✅ Complete |

---

## 📝 Developer Notes

- `master_agent.py` is the single entry point — one command, full pipeline execution
- Each agent is independently runnable for isolated testing and debugging
- Add `__pycache__/` to `.gitignore` — it is auto-generated and should not be committed
- NLTK `punkt` tokenizer data downloads automatically on first run
- Whisper auto-detects language — no configuration needed for multilingual audio
- Voice cloning works with reference samples as short as 5–10 seconds

---

## 📧 Team Contact

| Member | Domain |
|--------|--------|
| Pranali Sawant | NLP Processing — `nlp_cleaner.py` |
| Sanika Sathe | Repair Agents — `repair_agents.py` |
| Satyajeet Patil | Master Agent & Voice Clone — `master_agent.py`, `tts_engine.py` |
| Aditya Suryakar | Streamlit UI — `app.py` |

---

## 📄 License

Built for **HackoUtsav-2026** · 24-hour hackathon. All rights reserved by Team Steamroller.

---

<div align="center">

### Built with ❤️ in 24 hours by Team Fluent Force🚂
**HackoUtsav-2026**

</div>
