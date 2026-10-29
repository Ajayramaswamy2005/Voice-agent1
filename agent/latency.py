"""Per-turn latency tracking.

All timestamps are time.perf_counter() seconds. "eou" (end of user utterance) is when the caller's last
word actually ended in the audio, mapped from Deepgram's audio-time to wall-clock, so the STT stage includes
Deepgram's endpointing delay rather than starting the clock when the transcript arrives.

The stages are measured along the critical path, so they add up exactly to the end-to-end number:

    stt        eou -> final transcript received
    llm_ttft   transcript -> first LLM token          (LLM is started speculatively, in parallel with turn detection)
    chunking   first token -> first phrase sent to TTS
    tts_ttfb   phrase sent -> first audio byte back
    turn_gate  audio ready -> released to the caller (waiting on the turn detector, if it was slower)
    playout    browser jitter buffer -> audible (reported by the client)
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ["stt", "llm_ttft", "chunking", "tts_ttfb", "turn_gate", "playout"]


@dataclass
class TurnMetrics:
    persona: str
    turn: int
    eou: float
    transcript: float
    language: str = "en"
    user_text: str = ""
    turn_decided: float | None = None
    turn_method: str = ""
    llm_first_token: float | None = None
    first_phrase: float | None = None
    first_audio: float | None = None
    released: float | None = None
    playout_ms: float | None = None
    path: str = "speech"  # "speech" | "filler" (tool call answered with a pre-rendered filler)
    tools: list[str] = field(default_factory=list)
    interrupted: bool = False
    speculative: bool = False  # LLM started from an interim transcript, before the final arrived
    model: str = ""  # model that served the first LLM round (primary, or fallback when rate-limited)

    def mark(self, attr: str, t: float | None = None) -> None:
        if getattr(self, attr) is None:
            setattr(self, attr, t if t is not None else time.perf_counter())

    def breakdown(self) -> dict[str, float] | None:
        if None in (self.llm_first_token, self.first_phrase, self.first_audio, self.released):
            return None
        # Walk the critical path in order. Work that finished early (e.g. the LLM, started speculatively
        # before the final transcript arrived) is hidden behind the previous stage and counts as 0 ms.
        # The filler path can emit its first phrase before the LLM's first token, so tok is clamped too.
        points = [self.eou, self.transcript, min(self.llm_first_token, self.first_phrase), self.first_phrase,
                  self.first_audio, self.released]
        for i in range(1, len(points)):
            points[i] = max(points[i], points[i - 1])
        durations = [(b - a) * 1000 for a, b in zip(points, points[1:])]
        out = dict(zip(STAGES[:5], durations))
        out["playout"] = self.playout_ms or 0.0
        out["e2e"] = sum(out[s] for s in STAGES)
        return {k: round(v, 1) for k, v in out.items()}

    def record(self) -> dict | None:
        b = self.breakdown()
        if b is None:
            return None
        return {
            "ts": time.time(),
            "persona": self.persona,
            "turn": self.turn,
            "language": self.language,
            "path": self.path,
            "tools": self.tools,
            "turn_method": self.turn_method,
            "interrupted": self.interrupted,
            "speculative": self.speculative,
            "model": self.model,
            "user_text": self.user_text[:200],
            **b,
        }


class LatencyLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(exist_ok=True)

    def write(self, rec: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def read(self, limit: int = 500) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]


def summarize(records: list[dict]) -> dict:
    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)

    out = {"n": len(records)}
    for key in STAGES + ["e2e"]:
        vals = [r[key] for r in records if key in r]
        out[key] = {"p50": pct(vals, 0.5), "p90": pct(vals, 0.9), "mean": round(sum(vals) / len(vals), 1) if vals else 0}
    out["under_700ms"] = round(sum(r["e2e"] < 700 for r in records) / len(records), 3) if records else 0
    return out
