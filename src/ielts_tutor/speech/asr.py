"""Speech to text with word timings, on this machine (faster-whisper; no audio leaves it).

Whisper tidies speech: it drops fillers and smooths self-corrections, which would flatter
fluency and grammar (design doc 6.2). A disfluent initial prompt nudges it to keep them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

VERBATIM_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. I- I mean, er, you know."


class Word(BaseModel):
    text: str
    start: float
    end: float
    prob: float


class Transcript(BaseModel):
    text: str
    words: list[Word]
    duration: float  # seconds of audio
    language: str = "en"
    model: str = ""


@lru_cache(maxsize=2)
def _model(name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(audio: Path, model: str = "base.en") -> Transcript:
    segments, info = _model(model).transcribe(str(audio), word_timestamps=True, initial_prompt=VERBATIM_PROMPT,
                                              vad_filter=False, condition_on_previous_text=False)
    words = [Word(text=w.word.strip(), start=float(w.start), end=float(w.end), prob=float(w.probability))
             for s in segments for w in (s.words or []) if w.word.strip()]
    return Transcript(text=" ".join(w.text for w in words), words=words, duration=float(info.duration),
                      language=info.language, model=model)
