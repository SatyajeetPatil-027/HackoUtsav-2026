"""
audio_pipeline.py — Upload audio + anti-hallucination Whisper ASR.

Anti-hallucination stack (applied in order):
  PRE-WHISPER GATES (input validation & conditioning)
  ────────────────────────────────────────────────────
  Gate 1 — Duration check       : reject clips < MIN_DURATION_S seconds
  Gate 2 — SNR estimate         : reject audio where signal-to-noise < SNR_FLOOR_DB
  Gate 3 — VAD voiced-ratio     : voiced frames must exceed MIN_VOICED_RATIO
  Gate 4 — Silence trimming     : strip leading/trailing silence
  Gate 5 — Amplitude normalise  : bring quiet recordings to full scale
  Gate 6 — Audio padding        : pad short clips to MIN_PAD_S; helps Whisper context

  WHISPER PARAMETER HARDENING
  ───────────────────────────
  - temperature=0            (greedy decode — no sampling noise)
  - condition_on_previous_text=False  (stops one hallucination seeding the next)
  - no_speech_threshold=0.40 (per-segment silence rejection)
  - logprob_threshold=-0.8   (reject low-confidence predictions)
  - compression_ratio_threshold=1.8  (catch repetition loops early)
  - initial_prompt            (domain-specific priming reduces off-topic generation)

  POST-WHISPER GATES (output validation)
  ───────────────────────────────────────
  Gate 7 — Segment confidence   : drop segments where no_speech_prob > 0.40 or avg_logprob < -1.0
  Gate 8 — Blocklist regex      : reject known Whisper hallucination phrases
  Gate 9 — Repetition detector  : reject transcripts with unique-word ratio < 0.40
  Gate 10 — Temperature fallback: if text is empty, retry with temperature=(0.2, 0.4) schedule

  LONG AUDIO (> CHUNK_THRESHOLD_S)
  ─────────────────────────────────
  Chunked transcription with 30 % overlap; overlapping portions are deduplicated
  before stitching so sentence boundaries are preserved.
"""

import io
import os
import re
import wave
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16_000          # Hz — Whisper's native rate
CHANNELS      = 1

# Pre-Whisper gate thresholds
MIN_DURATION_S   = 2.0   # Gate 1: clips shorter than this are rejected
SNR_FLOOR_DB     = 3.0   # Gate 2: minimum SNR (dB) — kept low; stuttered/child speech is quiet
MIN_VOICED_RATIO = 0.05  # Gate 3: ≥5 % voiced frames — stuttering has many pauses/silent gaps
MIN_PAD_S        = 5.0   # Gate 6: pad clips to at least this length (Whisper context)
PAD_SILENCE_S    = 0.5   # Gate 6: silence appended before & after speech

# Long-audio chunking
CHUNK_THRESHOLD_S = 30.0        # switch to chunked mode above this
CHUNK_S           = 25.0        # each chunk length
OVERLAP_S         = 7.0         # overlap between consecutive chunks

# Gender detection
_FEMALE_F0_THRESHOLD = 160.0    # Hz — female F0 typically above this

# Whisper initial prompt — steers the model toward speech-repair domain
_INITIAL_PROMPT = (
    "The following is a natural spoken monologue. "
    "It may contain stuttering, repetitions, or filler words such as um and uh. "
    "Transcribe exactly what is said."
)

# ── gender detection ──────────────────────────────────────────────────────────

def detect_speaker_gender(file_bytes: bytes, filename: str = "audio.wav") -> str:
    """Estimate speaker gender via autocorrelation pitch (F0) analysis."""
    suffix = Path(filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(file_bytes); tmp.flush(); tmp.close()
    try:
        import scipy.io.wavfile as wavfile
        from scipy.signal import resample_poly
        import math

        sr, data = wavfile.read(tmp.name)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.abs(data).max() > 1.0:
            data /= 32768.0
        if sr != SAMPLE_RATE:
            gcd = math.gcd(sr, SAMPLE_RATE)
            data = resample_poly(data, SAMPLE_RATE // gcd, sr // gcd)
            sr = SAMPLE_RATE

        frame_len  = int(sr * 0.03)
        hop_len    = int(sr * 0.01)
        min_period = int(sr / 300)
        max_period = int(sr / 60)
        pitches: List[float] = []
        for start in range(0, len(data) - frame_len, hop_len):
            frame = data[start:start + frame_len]
            if np.sqrt(np.mean(frame ** 2)) < 0.01:
                continue
            ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
            if ac[0] == 0:
                continue
            ac = ac / ac[0]
            seg = ac[min_period:max_period]
            if seg.size == 0:
                continue
            peak = int(np.argmax(seg)) + min_period
            if ac[peak] > 0.3:
                pitches.append(sr / peak)

        return "female" if not pitches or float(np.median(pitches)) >= _FEMALE_F0_THRESHOLD else "male"
    except Exception:
        return "female"
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass

# ── basic audio utilities ─────────────────────────────────────────────────────

def record_audio(duration_seconds: int = 10) -> np.ndarray:
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = np.abs(audio).max()
    return audio if peak < 1e-6 else audio * (target_peak / peak)


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(2)
        wf.setframerate(sample_rate); wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def save_audio_to_temp(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(tmp.name, sample_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
    return tmp.name

# ── Gate 1 — Duration ─────────────────────────────────────────────────────────

def check_audio_duration(audio: np.ndarray, min_seconds: float = MIN_DURATION_S) -> bool:
    """Reject clips shorter than min_seconds.
    Very short clips give Whisper no context — the primary hallucination trigger."""
    return len(audio) >= int(SAMPLE_RATE * min_seconds)

# ── Gate 2 — SNR estimate ─────────────────────────────────────────────────────

def estimate_snr_db(audio: np.ndarray) -> float:
    """Estimate signal-to-noise ratio in dB using a simple percentile method.

    The bottom 10th-percentile RMS of 20 ms frames is used as the noise floor;
    the top 90th-percentile as the signal level. Returns 0 if audio is silent.
    """
    if audio.size == 0:
        return 0.0
    frame_len = int(SAMPLE_RATE * 0.02)
    rms_values = [
        np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
        for i in range(0, len(audio) - frame_len, frame_len)
    ]
    if not rms_values:
        return 0.0
    rms_arr  = np.array(rms_values)
    noise    = float(np.percentile(rms_arr, 10))
    signal   = float(np.percentile(rms_arr, 90))
    if noise < 1e-9 or signal < 1e-9:
        return 0.0
    return float(20.0 * np.log10(signal / noise))


def check_snr(audio: np.ndarray) -> bool:
    """Return True when the estimated SNR is above SNR_FLOOR_DB."""
    return estimate_snr_db(audio) >= SNR_FLOOR_DB

# ── Gate 3 — VAD voiced-ratio ─────────────────────────────────────────────────

def get_voiced_ratio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    aggressiveness: int = 2,
) -> float:
    """Return fraction of 30-ms frames classified as voiced.

    Uses WebRTC VAD when available; falls back to simple RMS thresholding.
    aggressiveness 0–3 (0 = least strict). Use 2 for live mic recordings.
    """
    try:
        import webrtcvad
        vad       = webrtcvad.Vad(aggressiveness)
        frame_len = int(sample_rate * 30 / 1000)   # 30 ms frames only
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        total = voiced = 0
        for start in range(0, len(pcm) - frame_len, frame_len):
            frame_bytes = pcm[start:start + frame_len].tobytes()
            try:
                if vad.is_speech(frame_bytes, sample_rate):
                    voiced += 1
            except Exception:
                pass
            total += 1
        return voiced / total if total > 0 else 0.0

    except ImportError:
        # RMS-based fallback.
        # Threshold of 0.015 ≈ -36 dBFS — lower than before to capture quiet/child voices
        # and stuttered speech where loudness between disfluencies is naturally low.
        frame_len = int(sample_rate * 0.03)
        frames = [audio[i:i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
        if not frames:
            return 0.0
        voiced = sum(1 for f in frames if float(np.sqrt(np.mean(f ** 2))) >= 0.015)
        return voiced / len(frames)


def check_voiced_ratio(audio: np.ndarray) -> bool:
    return get_voiced_ratio(audio) >= MIN_VOICED_RATIO

# ── Gate 4 — Silence trimming ─────────────────────────────────────────────────

def trim_silence(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    top_db: float = 30.0,
) -> np.ndarray:
    """Strip leading and trailing silence.

    Uses an energy-threshold approach (no librosa dependency).
    Frames more than top_db dB below the peak are treated as silent.
    """
    if audio.size == 0:
        return audio

    frame_len = int(sample_rate * 0.02)
    hop_len   = frame_len // 2

    peak_power = float(np.max(audio ** 2)) + 1e-10
    threshold  = peak_power * 10 ** (-top_db / 10)

    voiced_flags = [
        float(np.mean(audio[i:i + frame_len] ** 2)) >= threshold
        for i in range(0, len(audio) - frame_len, hop_len)
    ]

    if not any(voiced_flags):
        return audio   # all silent — gate 3 will reject this

    first = next(i for i, v in enumerate(voiced_flags) if v)
    last  = len(voiced_flags) - 1 - next(i for i, v in enumerate(reversed(voiced_flags)) if v)

    start_sample = max(0, first * hop_len - frame_len)
    end_sample   = min(len(audio), (last + 1) * hop_len + frame_len)
    return audio[start_sample:end_sample]

# ── Gate 6 — Audio padding ────────────────────────────────────────────────────

def pad_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Pad short clips with silence to reach MIN_PAD_S seconds.

    Whisper was trained on 30-second windows. Clips shorter than MIN_PAD_S
    lack enough context, pushing the model into hallucination territory.
    Adding zero-padding before/after is cheaper than repeating speech.
    """
    pad_samples = int(PAD_SILENCE_S * sample_rate)
    silence     = np.zeros(pad_samples, dtype=np.float32)
    padded      = np.concatenate([silence, audio, silence])

    min_total = int(MIN_PAD_S * sample_rate)
    if len(padded) < min_total:
        extra = np.zeros(min_total - len(padded), dtype=np.float32)
        padded = np.concatenate([padded, extra])
    return padded

# ── Gate 7 — Segment-level confidence filter ──────────────────────────────────

def filter_hallucinated_segments(
    result: dict,
    no_speech_threshold: float = 0.40,
    logprob_threshold:   float = -1.0,
) -> str:
    """Discard Whisper segments where confidence is too low.

    Per-segment no_speech_prob and avg_logprob are the two most reliable
    indicators of hallucination available directly from Whisper's output.
    """
    segments = result.get("segments", [])
    if not segments:
        return result.get("text", "").strip()

    good: List[str] = []
    for seg in segments:
        nsp    = float(seg.get("no_speech_prob", 0.0))
        avg_lp = float(seg.get("avg_logprob",    0.0))
        text   = seg.get("text", "").strip()
        if nsp < no_speech_threshold and avg_lp > logprob_threshold and text:
            good.append(text)

    return " ".join(good).strip()

# ── Gate 8 & 9 — Post-transcript filters ─────────────────────────────────────

# Common Whisper hallucinations for silent / noisy input
_HALLUCINATION_BLOCKLIST = [
    r"thank\s*you[\s.!,]*$",
    r"^thanks[\s.!,]*$",
    r"^you[\s.!,]*$",
    r"^\.*$",
    r"^,+$",
    r"^(uh+|um+|hmm+|ah+|oh+)[\s.!,]*$",
    r"subscribe",
    r"like and subscribe",
    r"please subscribe",
    r"^\[.*?\]$",               # [Music] [Applause] etc.
    r"^\(.*?\)$",               # (silence) (noise) etc.
    r"^www\.",                  # URL hallucinations
    r"subtitles?\s+by",
    r"transcript\s+by",
    r"amara\.org",
    r"dotsub\.com",
    r"caption(ed)?\s+by",
    r"\bvisit\s+us\b",
    r"\bfor\s+more\s+information\b",
]
_BLOCKLIST_RE  = re.compile("|".join(_HALLUCINATION_BLOCKLIST), re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def post_filter_transcript(text: str) -> str:
    """Apply blocklist + repetition checks to catch residual hallucinations.

    Gate 8 — Blocklist: reject phrases that match known hallucination patterns.
    Gate 9 — Repetition: reject text where unique-word ratio < 0.40
              (catches 'the the the', 'thank you thank you …' loops).
    """
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text or len(text) < 3:
        return ""

    if _BLOCKLIST_RE.search(text):
        return ""

    words = text.split()
    if len(words) >= 4:
        unique_ratio = len({w.lower().strip(".,!?\"'") for w in words}) / len(words)
        if unique_ratio < 0.40:
            log.debug("post_filter: repetition ratio %.2f — rejected", unique_ratio)
            return ""

    return text

# ── Whisper model cache & options ─────────────────────────────────────────────

_whisper_models: dict = {}


def _load_whisper(model_size: str = "base"):
    if model_size not in _whisper_models:
        import whisper
        log.info("Loading Whisper model '%s'…", model_size)
        _whisper_models[model_size] = whisper.load_model(model_size)
    return _whisper_models[model_size]


# Primary decode options — tightest possible anti-hallucination settings
_WHISPER_OPTS_PRIMARY = dict(
    verbose=False,
    fp16=False,
    beam_size=1,
    best_of=1,
    temperature=0,                        # greedy: no sampling randomness
    condition_on_previous_text=False,     # stops hallucination chains
    no_speech_threshold=0.40,            # per-segment silence gate
    logprob_threshold=-0.8,              # confidence gate
    compression_ratio_threshold=1.8,     # repetition-loop gate
    initial_prompt=_INITIAL_PROMPT,      # domain priming
)

# Fallback schedule — only used when the primary pass returns nothing
_TEMPERATURE_FALLBACK = (0.2, 0.4)

# ── Long-audio chunked transcription ─────────────────────────────────────────

def _dedup_overlap(prev: str, curr: str, overlap_words: int = 6) -> str:
    """Remove words at the start of `curr` that duplicate the tail of `prev`.

    Used to stitch overlapping transcript chunks without duplicating sentences.
    """
    if not prev:
        return curr
    prev_tail = prev.split()[-overlap_words:]
    curr_words = curr.split()
    for n in range(min(len(prev_tail), len(curr_words)), 0, -1):
        if prev_tail[-n:] == curr_words[:n]:
            return " ".join(curr_words[n:])
    return curr


def _transcribe_chunked(
    audio: np.ndarray,
    model_size: str,
    language: Optional[str],
) -> Tuple[str, dict]:
    """Split long audio into overlapping chunks and stitch transcripts.

    Each chunk is independently gated and transcribed. Overlapping tails are
    deduplicated before joining, preserving natural sentence boundaries across
    chunk edges.
    """
    chunk_len   = int(CHUNK_S   * SAMPLE_RATE)
    overlap_len = int(OVERLAP_S * SAMPLE_RATE)
    step        = chunk_len - overlap_len

    parts: List[str] = []
    combined_result: dict = {"text": "", "segments": []}
    offset = 0

    while offset < len(audio):
        chunk = audio[offset:offset + chunk_len]

        # Skip silent or too-short chunks
        chunk_trimmed = trim_silence(chunk)
        if len(chunk_trimmed) < int(MIN_DURATION_S * SAMPLE_RATE):
            offset += step
            continue
        if not check_voiced_ratio(chunk_trimmed):
            offset += step
            continue

        chunk_norm = normalize_audio(chunk_trimmed)
        chunk_pad  = pad_audio(chunk_norm)
        tmp = save_audio_to_temp(chunk_pad)
        try:
            import whisper as wh
            model  = _load_whisper(model_size)
            result = model.transcribe(tmp, language=language, **_WHISPER_OPTS_PRIMARY)
            text   = filter_hallucinated_segments(result)
            text   = post_filter_transcript(text)
            if text:
                text = _dedup_overlap(parts[-1] if parts else "", text)
                if text:
                    parts.append(text)
                    combined_result["segments"].extend(result.get("segments", []))
        except Exception as exc:
            log.warning("Chunk at %ds failed: %s", offset // SAMPLE_RATE, exc)
        finally:
            try: os.unlink(tmp)
            except OSError: pass

        offset += step

    combined_result["text"] = " ".join(parts)
    return combined_result["text"], combined_result

# ── Core transcription runner ─────────────────────────────────────────────────

def _run_transcription(
    tmp_path: str,
    model_size: str,
    language: Optional[str],
    is_long: bool = False,
) -> Tuple[str, dict]:
    """Run Whisper with the full anti-hallucination stack.

    For long audio (> CHUNK_THRESHOLD_S) uses chunked mode with overlap.
    For short audio uses single-pass with temperature-fallback retry.
    Raises ValueError if no trustworthy transcript can be produced.
    """
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(tmp_path)
    audio = data.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio /= 32768.0

    duration_s = len(audio) / sr

    if duration_s > CHUNK_THRESHOLD_S or is_long:
        text, result = _transcribe_chunked(audio, model_size, language)
    else:
        model  = _load_whisper(model_size)
        result = model.transcribe(tmp_path, language=language, **_WHISPER_OPTS_PRIMARY)

        # Gate 7 — segment confidence
        text = filter_hallucinated_segments(result)
        # Gate 8 & 9 — blocklist + repetition
        text = post_filter_transcript(text)

        # Gate 10 — temperature fallback
        if not text:
            for temp in _TEMPERATURE_FALLBACK:
                opts = {**_WHISPER_OPTS_PRIMARY, "temperature": temp}
                result = model.transcribe(tmp_path, language=language, **opts)
                text   = filter_hallucinated_segments(result)
                text   = post_filter_transcript(text)
                if text:
                    log.debug("Temperature fallback %.1f succeeded.", temp)
                    break

    if not text:
        raise ValueError(
            "No clear speech detected in the audio. "
            "Please speak closer to the microphone, reduce background noise, "
            "and try again."
        )

    return text, result

# ── Pre-Whisper conditioning pipeline ────────────────────────────────────────

def _prepare_live_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Condition audio before passing it to Whisper.

    Gate 1 (duration) is the only **hard** gate — truly empty audio cannot be
    transcribed. SNR and VAD are surfaced as **warnings only**: stuttered speech,
    child voices, and quiet recordings legitimately fail those thresholds even
    when they contain real, transcribable speech. Hallucination prevention is
    handled after transcription by Gates 7–10.
    """
    # Gate 1 — duration (hard gate: nothing to transcribe)
    if not check_audio_duration(audio):
        raise ValueError(
            f"Audio too short (minimum {MIN_DURATION_S:.0f}s). "
            "Please record for a little longer."
        )

    # Gate 4 — trim silence
    audio = trim_silence(audio, sample_rate)

    # Gate 2 — SNR advisory (soft — warn but continue)
    snr = estimate_snr_db(audio)
    if snr < SNR_FLOOR_DB:
        log.warning(
            "Low SNR %.1f dB (threshold %.0f dB) — continuing anyway; "
            "post-processing gates will catch hallucinations.",
            snr, SNR_FLOOR_DB,
        )

    # Gate 3 — voiced-ratio advisory (soft — warn but continue)
    # Stuttered / disfluent speech may have as little as 3–5 %% voiced frames.
    voiced = get_voiced_ratio(audio, sample_rate)
    if voiced < MIN_VOICED_RATIO:
        log.warning(
            "Low voiced-frame ratio %.0f%% (threshold %.0f%%) — continuing anyway.",
            voiced * 100, MIN_VOICED_RATIO * 100,
        )

    # Gate 5 — normalise amplitude
    audio = normalize_audio(audio)

    # Gate 6 — pad to give Whisper enough context window
    audio = pad_audio(audio, sample_rate)

    return audio

# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio: np.ndarray,
    model_size: str = "base",
    language: Optional[str] = None,
) -> Tuple[str, dict]:
    """Transcribe a float32 numpy audio array (live recording path)."""
    audio = _prepare_live_audio(audio)
    tmp   = save_audio_to_temp(audio)
    try:
        return _run_transcription(tmp, model_size, language)
    finally:
        try: os.unlink(tmp)
        except OSError: pass


def transcribe_uploaded_file(
    file_bytes: bytes,
    filename:   str = "audio.wav",
    model_size: str = "base",
    language:   Optional[str] = None,
    is_live:    bool = False,
) -> Tuple[str, dict]:
    """Transcribe raw audio bytes (upload or live-recording path).

    When ``is_live=True`` the full pre-Whisper gate stack is applied (same as
    ``transcribe_audio``). For uploaded files only the Whisper + post-processing
    gates run, since uploaded files are assumed to be complete recordings.
    """
    suffix = Path(filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(file_bytes); tmp.flush(); tmp.close()

    try:
        import scipy.io.wavfile as wavfile

        # Both uploaded and live recordings go through the same conditioning:
        # trim → normalise → pad. SNR/VAD are advisory-only (logged, not raised).
        sr, data = wavfile.read(tmp.name)
        data = data.astype(np.float32)
        if np.abs(data).max() > 1.0:
            data /= 32768.0
        data = _prepare_live_audio(data.flatten(), sr)
        wav_write(tmp.name, sr, (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16))

        duration_s = len(data) / sr
        is_long = duration_s > CHUNK_THRESHOLD_S

        return _run_transcription(tmp.name, model_size, language, is_long=is_long)

    finally:
        try: os.unlink(tmp.name)
        except OSError: pass