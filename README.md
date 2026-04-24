🚂 Steamroller
HackoUtsav-2026 Hackathon Project
Steamroller is an intelligent audio processing and analysis system that combines cutting-edge NLP, voice cloning, and interactive UI to deliver a comprehensive solution for audio transcription, tokenization, and user feedback.

👥 Team Members
RoleNameNLP ProcessingPranali SawantRepair AgentsSanika SatheFeedback & Voice CloneSatyajeetStreamlit UIAditya Suryakar

📋 Project Overview
Steamroller consists of four integrated branches:

NLP Processing - Processes audio transcripts into structured sentence tokens
Repair Agents - Handles audio correction and enhancement
Feedback & Voice Clone - Implements voice cloning and user feedback systems
Streamlit UI - Provides an interactive user interface for the entire pipeline

Core Pipeline: Audio to NLP Sentence Tokenizer
A two-stage Python pipeline that transforms raw audio into tokenized sentences:

Stage 1: Transcribes .wav audio files to text using OpenAI Whisper
Stage 2: Tokenizes the transcripts into sentences using NLTK


📁 Folder Structure
Steamroller/
│
├── audio/                        # Input: Place your .wav audio files here
│
├── transcripts/                  # Output: Whisper transcription results (.txt)
│
├── tokens/                       # Output: Sentence-tokenized transcripts (.txt)
│
├── audioToTextConvertor.py       # Stage 1 - Audio to text transcription
├── nlp_tokenizer.py              # Stage 2 - Sentence tokenization
├── app.py                        # Streamlit UI application
└── README.md                     # This file

🔧 Prerequisites
1. Python
Ensure Python 3.8+ is installed.

Download: https://www.python.org/downloads/
Verify installation:

bash  python --version
2. FFmpeg
Required by Whisper to read audio files.

Download: https://ffmpeg.org/download.html
Add to PATH: Ensure ffmpeg is in your system PATH
Verify installation:

bash  ffmpeg -version
3. Python Dependencies
Install required packages:
bashpip install openai-whisper nltk
For Streamlit UI:
bashpip install streamlit

🚀 How to Run
Step 1: Prepare Your Audio Files
Place all .wav audio files inside the audio/ folder:
Steamroller/
└── audio/
    ├── sample1.wav
    ├── sample2.wav
    └── ...
Step 2: Transcribe Audio to Text
Run the transcription script:
bashcd Steamroller
python audioToTextConvertor.py
What it does:

Loads all .wav files from the audio/ folder
Transcribes each using OpenAI Whisper (tiny model)
Saves transcripts as .txt files in the transcripts/ folder

⚠️ Note: This runs on CPU by default and may take several minutes for large audio files.
Example Output:
Transcribing: audio/sample1.wav
Saved: transcripts/sample1.txt

All audio files transcribed successfully
Step 3: Tokenize Transcripts into Sentences
Run the NLP tokenizer:
bashpython nlp_tokenizer.py
What it does:

Reads all .txt files from the transcripts/ folder
Splits each transcript into individual sentences using NLTK
Saves tokenized output to the tokens/ folder

Example Output:
Processed sentence tokens saved: tokens/sample1_tokens.txt
NLP processing completed successfully.
Step 4: Launch the Streamlit UI (Optional)
To use the interactive interface:
bashstreamlit run app.py

📊 Output Format
Input Audio File
audio/sample1.wav - Raw audio file
Transcription Output
transcripts/sample1.txt
Hello my name is John and I am from New York.
I have been working in software for five years.
Tokenized Output
tokens/sample1_tokens.txt
Hello my name is John and I am from New York.
I have been working in software for five years.
Each line represents one sentence token.

⚙️ Configuration & Customization
Use a More Accurate Whisper Model
The pipeline defaults to the tiny model for speed. For better accuracy (with slower processing), modify audioToTextConvertor.py:
pythonmodel = whisper.load_model("base")  # Options: "tiny", "base", "small", "medium", "large"
Model Performance Comparison
ModelRelative SpeedAccuracytiny🟢 FastestLowerbase🟡 FastMediumsmall🟠 ModerateGoodmedium🔴 SlowVery Goodlarge🔴🔴 Very SlowExcellent

📝 Notes

The transcripts/ and tokens/ folders are created automatically — no manual creation needed
Scripts use relative paths, so they work from any location on your system
NLTK's punkt tokenizer data is downloaded automatically on first run
Whisper supports multiple languages — transcripts will be detected automatically
Voice cloning features are handled by the Feedback & Voice Clone branch
The Streamlit UI provides a user-friendly interface for all operations


🔗 Dependencies Used

OpenAI Whisper - State-of-the-art speech recognition
NLTK - Natural Language Toolkit for sentence tokenization
Streamlit - Interactive web app framework


🎯 Project Goals

✅ Automated audio transcription with high accuracy
✅ Intelligent sentence tokenization for NLP tasks
✅ Voice cloning and feedback mechanisms
✅ Intuitive user interface for non-technical users
✅ Scalable architecture for batch processing


📧 Support & Questions
For issues or questions, please reach out to the respective team member for your area of interest:

Audio Processing: Pranali Sawant
System Repair/Enhancement: Sanika Sathe
Voice & Feedback: Satyajeet
UI/UX: Aditya Suryakar


📄 License
This project was created for HackoUtsav-2026 hackathon.

Happy Processing! 🚀
