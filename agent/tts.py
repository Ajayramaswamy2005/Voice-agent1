"""Streaming text-to-speech over Cartesia's WebSocket API, one context per agent utterance."""

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

from .config import Config

log = logging.getLogger("tts")


@dataclass
class Audio:
    pcm: bytes


@dataclass
class WordTimes:
    words: list[str]
    start: list[float]
    end: list[float]


class _Done:
    pass


DONE = _Done()


class TTSContext:
    """One continuous utterance. Text is pushed in phrases; audio streams back in order."""

    def __init__(self, tts: "CartesiaTTS", language: str, voice: str | None = None, sample_rate: int | None = None):
        self.tts = tts
        self.voice = voice or tts.cfg.cartesia_voice
        self.sample_rate = sample_rate or tts.cfg.out_sample_rate
        self.id = uuid.uuid4().hex
        self.language = language
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def push(self, text: str, final: bool = False) -> None:
        if self.closed:
            return
        await self.tts._send(self._request(text, cont=not final))
        if final:
            self.closed = True

    async def end(self) -> None:
        """No more text: flush whatever is buffered."""
        if not self.closed:
            await self.push("", final=True)

    async def cancel(self) -> None:
        self.closed = True
        self.tts._contexts.pop(self.id, None)
        try:
            await self.tts._send({"context_id": self.id, "cancel": True})
        except Exception:
            pass
        self.queue.put_nowait(DONE)

    async def events(self):
        while True:
            ev = await self.queue.get()
            if ev is DONE:
                return
            yield ev

    def _request(self, text: str, cont: bool) -> dict:
        cfg = self.tts.cfg
        return {
            "model_id": cfg.cartesia_model,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice},
            "language": self.language,
            "context_id": self.id,
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.sample_rate},
            "add_timestamps": True,
            "continue": cont,
            # We chunk text ourselves (PhraseChunker); server-side buffering would add up to 3s.
            "max_buffer_delay_ms": 0,
        }


class CartesiaTTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ws = None
        self._contexts: dict[str, TTSContext] = {}
        self._send_lock = asyncio.Lock()
        self._recv_task: asyncio.Task | None = None

    async def connect(self) -> None:
        url = f"wss://api.cartesia.ai/tts/websocket?{urlencode({'cartesia_version': self.cfg.cartesia_version})}"
        self.ws = await websockets.connect(
            url, additional_headers={"X-API-Key": self.cfg.cartesia_key}, max_size=None, open_timeout=10
        )
        self._recv_task = asyncio.create_task(self._recv_loop())

    def context(self, language: str, voice: str | None = None, sample_rate: int | None = None) -> TTSContext:
        ctx = TTSContext(self, language, voice, sample_rate)
        self._contexts[ctx.id] = ctx
        return ctx

    async def synthesize(self, text: str, language: str, voice: str | None = None,
                         sample_rate: int | None = None, timeout: float = 20.0) -> bytes:
        """Whole-clip synthesis (used to pre-render backchannels and fillers)."""
        ctx = self.context(language, voice, sample_rate)
        pcm = bytearray()
        try:
            async with asyncio.timeout(timeout):
                await ctx.push(text, final=True)
                async for ev in ctx.events():
                    if isinstance(ev, Audio):
                        pcm += ev.pcm
        except TimeoutError:
            log.warning("synthesis timed out after %.0fs: %r", timeout, text[:40])
            await ctx.cancel()
            return b""
        return bytes(pcm)

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
        if self.ws is not None:
            await self.ws.close()

    async def _send(self, msg: dict) -> None:
        async with self._send_lock:
            await self.ws.send(json.dumps(msg, ensure_ascii=False))

    async def _recv_loop(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                ctx = self._contexts.get(msg.get("context_id", ""))
                if ctx is None:
                    continue  # cancelled context: drop late audio
                kind = msg.get("type")
                if kind == "chunk" and msg.get("data"):
                    ctx.queue.put_nowait(Audio(base64.b64decode(msg["data"])))
                elif kind == "timestamps" and msg.get("word_timestamps"):
                    wt = msg["word_timestamps"]
                    ctx.queue.put_nowait(WordTimes(wt["words"], wt["start"], wt["end"]))
                elif kind == "error":
                    log.error("cartesia error: %s", msg)
                if msg.get("done"):
                    self._contexts.pop(ctx.id, None)
                    ctx.queue.put_nowait(DONE)
        except websockets.ConnectionClosed as e:
            log.warning("cartesia closed: %s", e)
        except Exception:
            log.exception("cartesia receive loop crashed")
        finally:
            for ctx in list(self._contexts.values()):
                ctx.queue.put_nowait(DONE)
            self._contexts.clear()
