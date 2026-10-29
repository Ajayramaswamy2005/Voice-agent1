"""Streaming speech-to-text over Deepgram's live WebSocket API."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

from .config import Config

log = logging.getLogger("stt")


@dataclass
class Transcript:
    text: str
    is_final: bool  # this segment's text won't change
    speech_final: bool  # endpoint detected (short silence after speech)
    start: float  # audio-time seconds
    end: float  # audio-time end of last word
    language: str | None
    received_at: float  # time.perf_counter()
    from_finalize: bool = False  # produced by our Finalize request (a pause our VAD detected)


@dataclass
class SpeechStarted:
    audio_time: float
    received_at: float


@dataclass
class UtteranceEnd:
    last_word_end: float
    received_at: float


class DeepgramSTT:
    def __init__(self, cfg: Config, on_event):
        self.cfg = cfg
        self.on_event = on_event  # async callable(Transcript | SpeechStarted | UtteranceEnd)
        self.ws = None
        self._tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        params = {
            "model": self.cfg.stt_model,
            "language": self.cfg.stt_language,
            "encoding": "linear16",
            "sample_rate": self.cfg.in_sample_rate,
            "channels": 1,
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
            "endpointing": self.cfg.stt_endpointing_ms,
            "utterance_end_ms": 1000,
            "vad_events": "true",
        }
        url = f"wss://{self.cfg.deepgram_host}/v1/listen?{urlencode(params)}"
        self.ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Token {self.cfg.deepgram_key}"}, max_size=None
        )
        self._tasks = [asyncio.create_task(self._recv_loop()), asyncio.create_task(self._keepalive())]

    async def send_audio(self, pcm16: bytes) -> None:
        if self.ws is not None:
            try:
                await self.ws.send(pcm16)
            except websockets.ConnectionClosed:
                log.warning("deepgram socket closed while sending audio")

    async def finalize(self) -> None:
        """Ask Deepgram to finalize the audio received so far right now, instead of waiting for endpointing."""
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({"type": "Finalize"}))
            except websockets.ConnectionClosed:
                pass

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass

    async def _keepalive(self) -> None:
        while True:
            await asyncio.sleep(5)
            try:
                await self.ws.send(json.dumps({"type": "KeepAlive"}))
            except websockets.ConnectionClosed:
                return

    async def _recv_loop(self) -> None:
        try:
            async for raw in self.ws:
                now = time.perf_counter()
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "Results":
                    alt = msg["channel"]["alternatives"][0]
                    text = alt.get("transcript", "")
                    ws = alt.get("words") or []
                    langs = alt.get("languages") or []
                    start = msg.get("start", 0.0)
                    end = ws[-1]["end"] if ws else start + msg.get("duration", 0.0)
                    await self.on_event(Transcript(
                        text=text, is_final=bool(msg.get("is_final")), speech_final=bool(msg.get("speech_final")),
                        start=start, end=end, language=langs[0] if langs else None, received_at=now,
                        from_finalize=bool(msg.get("from_finalize")),
                    ))
                elif kind == "SpeechStarted":
                    await self.on_event(SpeechStarted(audio_time=msg.get("timestamp", 0.0), received_at=now))
                elif kind == "UtteranceEnd":
                    await self.on_event(UtteranceEnd(last_word_end=msg.get("last_word_end", 0.0), received_at=now))
                elif kind == "Error" or "err_code" in msg:
                    log.error("deepgram error: %s", msg)
        except websockets.ConnectionClosed as e:
            log.info("deepgram closed: %s", e)
        except Exception:
            log.exception("deepgram receive loop crashed")
