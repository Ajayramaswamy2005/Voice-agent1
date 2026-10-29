"""Small text helpers shared by the pipeline: language guess, ack detection, TTS phrase chunking."""

import re

from .personas import ACK_WORDS

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[\wऀ-ॿ'-]+")
# Romanised Hindi markers: enough of these and we treat the caller as speaking Hindi.
_HINGLISH = {
    "hai", "haan", "nahi", "nahin", "kya", "mera", "meri", "mujhe", "aap", "kab", "kaise", "kyun", "tha",
    "karna", "karo", "chahiye", "bhai", "ji", "accha", "theek", "kal", "aaj", "abhi", "wala", "wali",
}


def detect_language(text: str) -> str:
    """'hi' if the text is Devanagari or noticeably Hinglish, else 'en'."""
    if not text:
        return "en"
    letters = [ch for ch in text if ch.isalpha()]
    if letters and sum(bool(_DEVANAGARI.match(ch)) for ch in letters) / len(letters) > 0.3:
        return "hi"
    words = [w.lower() for w in _WORD.findall(text)]
    if words and sum(w in _HINGLISH for w in words) / len(words) >= 0.25:
        return "hi"
    return "en"


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def is_ack_only(text: str) -> bool:
    """True for things like 'mm-hm', 'haan ji', 'okay okay' that shouldn't interrupt the agent."""
    ws = words(text)
    return bool(ws) and len(ws) <= 3 and all(w.replace("-", "") in ACK_WORDS or w in ACK_WORDS for w in ws)


def same_words(a: str, b: str) -> bool:
    """Transcripts are the same utterance if their words match, ignoring case and punctuation."""
    return words(a) == words(b)


_SENTENCE_END = re.compile(r"[.!?।]+[\"')\]]*\s")
_CLAUSE_END = re.compile(r"[,;:—]\s")


class PhraseChunker:
    """Buffers streamed LLM tokens and emits speakable phrases for TTS.

    The first phrase is released early (at a clause boundary) to cut time-to-first-audio;
    later phrases wait for sentence ends so prosody stays natural.
    """

    def __init__(self, first_min_chars: int = 12, min_chars: int = 30):
        self.buf = ""
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self.emitted = 0

    def push(self, token: str) -> list[str]:
        self.buf += token
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            phrase, self.buf = self.buf[:cut].strip(), self.buf[cut:]
            if phrase:
                out.append(phrase)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        phrase, self.buf = self.buf.strip(), ""
        return [phrase] if phrase else []

    def _find_cut(self) -> int | None:
        min_len = self.first_min_chars if self.emitted == 0 else self.min_chars
        for m in _SENTENCE_END.finditer(self.buf):
            if m.end() >= min(min_len, 4):
                return m.end()
        if self.emitted == 0:
            for m in _CLAUSE_END.finditer(self.buf):
                if m.end() >= min_len:
                    return m.end()
        # Safety valve for long unpunctuated runs.
        if len(self.buf) > 160:
            sp = self.buf.rfind(" ", 0, 160)
            return sp if sp > 0 else 160
        return None
