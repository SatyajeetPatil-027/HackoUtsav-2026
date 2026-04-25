# STEAMROLLER — Fast End-to-End Speech Repair

## Flow
Upload stuttered audio -> Whisper ASR -> NLP cleaning -> local repair agents -> repaired audio output.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Speed Notes
- Default Whisper model is `tiny` for faster processing.
- Use short clips around 5–30 seconds for near/under-1-minute processing.
- The repair stage is local rule-based, not cloud LLM-based, so it is fast.

## Voice Note
This version does not do true voice cloning. True voice cloning requires a separate model and a reference speaker sample. The app uses local system TTS by default to avoid the default female Google TTS voice as much as possible.
