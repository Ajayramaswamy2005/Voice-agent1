"""One phone call: the full-duplex turn-taking state machine.

Data flow:
    browser mic --pcm16--> DeepgramSTT --transcripts--> CallSession
    CallSession --(speculative)--> Response: Groq LLM --phrases--> Cartesia TTS --pcm--> browser speaker

Behaviours:
  * Fast endpointing: our own VAD (agent/vad.py) hears a pause 100 ms in and sends Deepgram `Finalize`, so
    the final transcript arrives ~200 ms after the caller stops instead of waiting for Deepgram's endpointing.
    If the latest interim transcript already covers the speech, the LLM starts on it right away
    (provisional); the final transcript confirms it (same words) or replaces it.
  * Semantic turn detection: in parallel a small model decides whether the caller is actually done. Audio is
    gated until it says yes (or the caller stays silent for max_incomplete_wait_ms), and never starts before
    min_turn_silence_ms of silence. If the caller resumes (VAD onset or new words), the unreleased response
    is thrown away silently.
  * Barge-in: caller speech while the agent talks ducks playback immediately (VAD) and, once the words are
    confirmed to be more than an acknowledgement ("haan", "okay"), stops playback, cancels LLM/TTS, and
    records only what the caller actually heard into history.
  * Backchannels: "mm-hm"/"हाँ" during the caller's pauses when they're mid-thought, or on phrase boundaries
    during long monologues.
  * Tool calls during speech: as soon as the LLM starts a tool call we play a pre-rendered filler
    ("let me check that") while the MCP tool runs, plus a second filler if the tool is slow.
"""

import asyncio
import collections
import json
import logging
import re
import struct
import time
from dataclasses import dataclass

from .clips import Clip, ClipBank
from .config import Config
from .latency import LatencyLog, TurnMetrics
from .llm import GroqLLM, ModelUsed, TextDelta, ToolCalls, ToolCallStarted, TurnDetector
from .mcp_hub import MCPHub
from .personas import Persona
from .stt import DeepgramSTT, SpeechStarted, Transcript, UtteranceEnd
from .text import PhraseChunker, detect_language, is_ack_only, same_words, words
from .tts import Audio, CartesiaTTS, TTSContext, WordTimes
from .vad import EnergyVAD

log = logging.getLogger("session")
_PSEUDO_CALL = re.compile(r"^\W*(calling|call|function|tool)\s*[:(]", re.IGNORECASE)


@dataclass
class _End:
    pass


class Response:
    """One agent turn: possibly several LLM rounds with tool calls, spoken as an ordered list of segments."""

    def __init__(self, session: "CallSession", user_text: str, metrics: TurnMetrics, seq: int,
                 provisional: bool = False, event: bool = False):
        self.s = session
        self.user_text = user_text
        self.event = event  # agent-initiated (a tool event, not caller speech): kept out of latency stats
        self.m = metrics
        self.seq = seq
        self.gate = asyncio.Event()  # audio may play: transcript confirmed AND turn detector says done
        # A provisional response was started from an interim transcript; it's confirmed when the final
        # transcript matches. Until then it may think (LLM) but not speak (TTS) or act (tools).
        self.confirmed = asyncio.Event()
        if not provisional:
            self.confirmed.set()
        self._turn_ok = False
        self.segments: asyncio.Queue = asyncio.Queue()  # Clip | TTSContext | _End
        self.new_messages: list[dict] = []  # tool-call/tool-result messages produced this turn
        self.spoken: list[tuple[float, str]] = []  # (end_ms within this response's audio, word)
        self.sent_ms = 0.0
        self.done = False
        self.interrupted = False
        self._contexts: list[TTSContext] = []
        self._clip_texts: set[str] = set()
        self._said: set[tuple[str, ...]] = set()  # phrases already spoken in this response (normalized)
        self._tasks: list[asyncio.Task] = []

    @property
    def released(self) -> bool:
        return self.gate.is_set()

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._produce()), asyncio.create_task(self._consume())]

    @property
    def provisional(self) -> bool:
        return not self.confirmed.is_set()

    def release(self) -> None:
        """The turn detector (or the hold timeout) says the caller is done."""
        self._turn_ok = True
        self._try_release()

    def confirm(self) -> None:
        """The final transcript matched the interim this response was started from."""
        self.confirmed.set()
        self._try_release()

    def _try_release(self) -> None:
        if self.done or self.gate.is_set() or not (self._turn_ok and self.confirmed.is_set()):
            return
        wait = self.m.eou + self.s.cfg.min_turn_silence_ms / 1000 - time.perf_counter()
        if wait > 0:
            asyncio.get_running_loop().call_later(wait, self._try_release)
            return
        self.gate.set()
        self.s.on_response_released(self)

    async def cancel(self) -> None:
        for t in self._tasks:
            t.cancel()
        for ctx in self._contexts:
            await ctx.cancel()
        self.done = True

    # Filler clips are left out of the text that goes into history: they're sound effects, and when the
    # LLM sees "Let me check that for you." in its own past turns it starts saying it too.
    def heard_text(self, played_ms: float) -> str:
        return " ".join(w for end, w in self.spoken if end <= played_ms + 80 and w not in self._clip_texts)

    def full_text(self) -> str:
        return " ".join(w for _, w in self.spoken if w not in self._clip_texts)

    def _repeat(self, phrase: str) -> bool:
        """After a tool call the model often restates what it said before it ("I'll check that order.")
        or echoes a filler we already played. Never say the same phrase twice in one reply."""
        key = tuple(words(phrase))
        if key in self._said:
            return True
        self._said.add(key)
        return False

    def _pseudo_tool_call(self, phrase: str) -> bool:
        """Some models occasionally write a tool call as text ("[Calling: track_order ...]") instead of
        making one. That must never be spoken."""
        p = phrase.strip()
        return ((p.startswith("[") and p.endswith("]")) or bool(_PSEUDO_CALL.match(p))
                or any(t["function"]["name"] in p for t in self.s.tools))

    # -- producer: LLM -> phrases -> TTS contexts / filler clips ----------------------------------------
    async def _produce(self) -> None:
        s = self.s
        messages = _recent(s.history) + [{"role": "user", "content": self.user_text}]
        model = None  # primary (with rate-limit fallback)
        try:
            for _round in range(4):
                chunker = PhraseChunker()
                ctx: TTSContext | None = None
                text = ""
                calls: list[dict] = []
                pseudo = False
                async for ev in s.llm.stream(messages + self.new_messages, s.tools, model=model):
                    if isinstance(ev, ModelUsed):
                        self.m.model = self.m.model or ev.model
                    elif isinstance(ev, TextDelta):
                        self.m.mark("llm_first_token")
                        text += ev.text
                        for phrase in chunker.push(ev.text):
                            if self._pseudo_tool_call(phrase):
                                pseudo = True
                                continue
                            if not self._repeat(phrase):
                                ctx = await self._speak(ctx, phrase)
                    elif isinstance(ev, ToolCallStarted):
                        self.m.mark("llm_first_token")
                        self.m.tools.append(ev.name)
                        asyncio.create_task(s.send_json({"type": "tool", "name": ev.name}))
                        if not text.strip() and not self.m.first_phrase:
                            self._queue_clip("filler")
                    elif isinstance(ev, ToolCalls):
                        calls = ev.calls
                for phrase in chunker.flush():
                    if self._pseudo_tool_call(phrase):
                        pseudo = True
                        continue
                    if not self._repeat(phrase):
                        ctx = await self._speak(ctx, phrase)
                if ctx is not None:
                    await ctx.end()
                if pseudo and not calls and model is None and s.cfg.llm_fallback_model:
                    # It meant to call a tool but wrote it out instead; redo this round on the fallback
                    # model, which reliably emits real tool calls.
                    log.warning("pseudo tool call in text, retrying on %s: %r", s.cfg.llm_fallback_model, text[:80])
                    model = s.cfg.llm_fallback_model
                    continue
                if not calls:
                    break
                # Side-effecting tools must never run on a speculative (unconfirmed) turn.
                await self.gate.wait()
                self.new_messages.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in calls
                    ],
                })
                results = await self._run_tools(calls)
                for c, out in zip(calls, results):
                    self.new_messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("response producer failed")
            sorry = "माफ़ कीजिए, क्या आप दोबारा बता सकते हैं?" if self.s.lang == "hi" else "Sorry, could you say that again?"
            ctx = await self._speak(None, sorry)
            await ctx.end()
        finally:
            self.segments.put_nowait(_End())

    async def _speak(self, ctx: TTSContext | None, phrase: str) -> TTSContext:
        if ctx is None:
            # Don't spend a TTS generation (Cartesia allows only a few concurrent) on unconfirmed text.
            await self.confirmed.wait()
            ctx = self.s.tts.context(detect_language(phrase))
            self._contexts.append(ctx)
            self.segments.put_nowait(ctx)
        self.m.mark("first_phrase")
        await ctx.push(phrase + " ")
        return ctx

    def _queue_clip(self, kind: str) -> None:
        clip = self.s.clips.pick(self.s.persona.key, kind, self.s.lang)
        if clip is None:
            return
        self._clip_texts.add(clip.text)
        self._said.add(tuple(words(clip.text)))
        if kind == "filler" and self.m.first_phrase is None:
            self.m.path = "filler"
        self.m.mark("first_phrase")
        self.segments.put_nowait(clip)

    async def _run_tools(self, calls: list[dict]) -> list[str]:
        jobs = [asyncio.create_task(self.s.hub.call(self.s.persona.mcp_server, c["name"], c["arguments"]))
                for c in calls]
        done, _ = await asyncio.wait(jobs, timeout=self.s.cfg.slow_tool_filler_s)
        if len(done) < len(jobs):
            self._queue_clip("slow_filler")
        return list(await asyncio.gather(*jobs))

    # -- consumer: segments -> audio frames to the caller (gated) -------------------------------------
    async def _consume(self) -> None:
        s = self.s
        try:
            while True:
                seg = await self.segments.get()
                if isinstance(seg, _End):
                    break
                if isinstance(seg, Clip):
                    self.m.mark("first_audio")
                    await self._emit(seg.pcm)
                    self.spoken.append((self.sent_ms, seg.text))
                    continue
                base_ms = self.sent_ms
                async for ev in seg.events():
                    if isinstance(ev, Audio):
                        self.m.mark("first_audio")
                        await self._emit(ev.pcm)
                    elif isinstance(ev, WordTimes):
                        self.spoken += [(base_ms + e * 1000, w) for w, e in zip(ev.words, ev.end)]
            await self.gate.wait()  # keep history ordered even if nothing was spoken
            self.done = True
            s.on_response_finished(self)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("response consumer failed")
            self.done = True

    async def _emit(self, pcm: bytes) -> None:
        if not self.gate.is_set():
            await self.gate.wait()
        self.m.mark("released")
        self.sent_ms += len(pcm) / self.s.bytes_per_ms_out
        await self.s.send_audio(self.seq, pcm)


class CallSession:
    def __init__(self, ws, cfg: Config, persona: Persona, hub: MCPHub, llm: GroqLLM, turn: TurnDetector,
                 tts: CartesiaTTS, clips: ClipBank, latency: LatencyLog, lang: str = "en"):
        self.ws = ws
        self.cfg = cfg
        self.persona = persona
        self.hub = hub
        self.llm = llm
        self.turn = turn
        self.tts = tts
        self.clips = clips
        self.latency = latency
        self.lang = lang
        hidden = persona.hidden_tools()
        self.tools = [t for t in hub.tools.get(persona.mcp_server, []) if t["function"]["name"] not in hidden]
        self._events_task: asyncio.Task | None = None
        self.bytes_per_ms_out = cfg.out_sample_rate * 2 / 1000

        self.history: list[dict] = [{"role": "system", "content": persona.system_prompt()}]
        self.stt = DeepgramSTT(cfg, self.on_stt)
        self.vad = EnergyVAD(cfg.in_sample_rate)
        self._interim: Transcript | None = None  # latest non-final transcript of the current segment

        self.user_segments: list[str] = []
        self.user_turn_started: float | None = None  # perf time of first word in this caller turn
        self.last_word_end = 0.0  # audio-time seconds
        self.response: Response | None = None
        self.pending: dict[int, TurnMetrics] = {}  # seq -> metrics awaiting playout report
        self.seq = 0
        self.turn_no = 0

        self._send_lock = asyncio.Lock()
        self._samples_in = 0
        # perf_counter - audio_time for recent frames; the min is the least-delayed frame. A sliding window
        # (2 s) keeps the mapping right even if the client's stream stalls or drifts.
        self._offsets: collections.deque[float] = collections.deque(maxlen=100)
        self._play_end = 0.0  # perf time when the caller's speaker buffer runs dry (estimate)
        self._last_backchannel = 0.0
        self._ducked_at: float | None = None
        self._speech_hint_at = 0.0  # last Deepgram SpeechStarted (perf time)
        self._hold_task: asyncio.Task | None = None

    # -- lifecycle -------------------------------------------------------------------------------------
    async def start(self) -> None:
        await self.stt.connect()
        greeting = self.clips.pick(self.persona.key, "greeting", self.lang)
        if greeting:
            self.history.append({"role": "assistant", "content": greeting.text})
            await self.send_json({"type": "agent_text", "text": greeting.text, "final": True})
            self.seq += 1
            await self.send_audio(self.seq, greeting.pcm)
        if self.persona.event_tool:
            self._events_task = asyncio.create_task(self._watch_events())

    async def close(self) -> None:
        if self.response:
            await self.response.cancel()
        if self._hold_task:
            self._hold_task.cancel()
        if self._events_task:
            self._events_task.cancel()
        await self.stt.close()

    # -- agent-initiated turns -------------------------------------------------------------------------
    async def _watch_events(self) -> None:
        """Long-poll the line's event tool (e.g. the browser agent: "approval needed", "finished") and say
        each event out loud, so the agent can speak first instead of only answering."""
        server, tool, after = self.persona.mcp_server, self.persona.event_tool, -1
        while True:
            try:
                ev = json.loads(await self.hub.call(server, tool, json.dumps({"after": after, "timeout_s": 20})))
            except (json.JSONDecodeError, TypeError):
                await asyncio.sleep(1)
                continue
            if "error" in ev:
                log.warning("event tool %s failed: %s", tool, ev["error"])
                await asyncio.sleep(2)
                continue
            after = int(ev.get("id", after))
            if ev.get("event"):
                await self.send_json({"type": "app_event", "text": ev["event"]})
                await self.notify(ev["event"])

    async def notify(self, text: str) -> None:
        """Start a turn the caller didn't ask for, at the next quiet moment: never over the caller's speech,
        an answer in progress, or audio still playing. It enters history as an "[update]" user message."""
        await self._inject_turn(f"[update] {text}")

    async def on_typed(self, text: str) -> None:
        """The caller typed instead of speaking: answer it like a spoken turn, at the next quiet moment."""
        text = text.strip()[:1000]
        if text:
            self.lang = detect_language(text)
            await self._inject_turn(text)

    async def _inject_turn(self, user_text: str) -> None:
        while ((self.response is not None and not self.response.done) or self.agent_speaking
               or self.vad.in_speech or self.user_segments):
            await asyncio.sleep(0.2)
        now = time.perf_counter()
        self.turn_no += 1
        self.seq += 1
        m = TurnMetrics(persona=self.persona.key, turn=self.turn_no, eou=now - self.cfg.min_turn_silence_ms / 1000,
                        transcript=now, language=self.lang, user_text=user_text)
        r = Response(self, user_text, m, self.seq, event=True)
        self.response = r
        r.start()
        r.release()

    # -- inbound ---------------------------------------------------------------------------------------
    async def on_audio(self, pcm: bytes) -> None:
        now = time.perf_counter()
        self._samples_in += len(pcm) // 2
        self._offsets.append(now - self._samples_in / self.cfg.in_sample_rate)
        await self.stt.send_audio(pcm)
        was_speaking = self.vad.in_speech
        speech_end = self.vad.update(pcm)
        if speech_end is not None:
            await self._on_pause(speech_end)
        elif self.vad.in_speech and not was_speaking:
            await self._maybe_duck()
            r = self.response
            if r is not None and not r.released:
                # Caller resumed after a pause: drop the unreleased answer now (~120 ms after they
                # start), rather than when the next interim transcript arrives (~500 ms).
                await self._drop_speculative()

    async def on_client_message(self, msg: dict) -> None:
        if msg.get("type") == "typed":
            asyncio.create_task(self.on_typed(str(msg.get("text", ""))))
            return
        if msg.get("type") == "playout":
            m = self.pending.pop(int(msg.get("seq", -1)), None)
            if m is not None:
                m.playout_ms = float(msg.get("delay_ms", 0))
                self._write_metrics(m)

    def audio_to_wall(self, audio_t: float) -> float:
        return audio_t + (min(self._offsets) if self._offsets else 0.0)

    @property
    def agent_speaking(self) -> bool:
        r = self.response
        return (r is not None and r.released and not r.done) or self._play_end > time.perf_counter()

    async def on_stt(self, ev) -> None:
        if isinstance(ev, SpeechStarted):
            self._speech_hint_at = time.perf_counter()
            await self._maybe_duck()
            return
        if isinstance(ev, UtteranceEnd):
            if self.user_segments and (self.response is None or self.response.done):
                await self._end_of_turn_candidate(ev.received_at)
            return
        assert isinstance(ev, Transcript)
        text = ev.text.strip()
        if text:
            await self.send_json({"type": "user_text", "text": text, "final": ev.is_final})

        if not ev.is_final:
            if not words(text):
                return
            self._interim = ev
            r = self.response
            if r is not None and not r.released:
                if not same_words(r.user_text, self._current_text()):
                    # Caller kept talking after a pause: drop the speculative response silently.
                    await self._drop_speculative()
            elif self.agent_speaking and not is_ack_only(text):
                await self._barge_in()
            self.user_turn_started = self.user_turn_started or ev.received_at
            return

        self._interim = None
        if text:
            if self.agent_speaking and is_ack_only(text):
                return  # "haan", "okay" while the agent talks: listening noise, not a turn
            if time.perf_counter() - self._last_backchannel < 1.2 and is_ack_only(text):
                return  # likely our own backchannel echoing back through the mic
            if self.agent_speaking:
                await self._barge_in()
            self.user_segments.append(text)
            r = self.response
            ends_turn = ev.speech_final or ev.from_finalize
            if (r is not None and not r.released and not ends_turn
                    and not same_words(r.user_text, " ".join(self.user_segments))):
                await self._drop_speculative()
            self.last_word_end = ev.end
            self.user_turn_started = self.user_turn_started or ev.received_at
            hint = ev.language or ""
            self.lang = "hi" if hint.startswith("hi") else detect_language(" ".join(self.user_segments))

        if (ev.speech_final or ev.from_finalize) and self.user_segments:
            await self._end_of_turn_candidate(ev.received_at)
        elif text:
            await self._maybe_backchannel(long_monologue=True)

    # -- turn taking ---------------------------------------------------------------------------------
    def _current_text(self) -> str:
        parts = self.user_segments + ([self._interim.text] if self._interim else [])
        return " ".join(p for p in parts if p)

    async def _on_pause(self, speech_end: float) -> None:
        """Our VAD heard a short pause. Finalize the transcript now and, if the latest interim already covers
        everything said, start thinking about the answer before the final transcript even arrives."""
        await self.stt.finalize()
        interim = self._interim
        if interim is None or self.agent_speaking:
            return
        if self.response is not None and not self.response.done:
            return
        if interim.end < speech_end - 0.35:
            return  # the interim lags the audio: its last words are missing, so don't guess
        text = self._current_text()
        if words(text):
            self._start_response(text, eou_audio=interim.end, transcript_at=time.perf_counter(), provisional=True)

    async def _end_of_turn_candidate(self, transcript_at: float) -> None:
        user_text = " ".join(self.user_segments)
        r = self.response
        if r is not None and not r.done and not r.released and r.provisional and same_words(r.user_text, user_text):
            # The final transcript matches what we already started on: keep the head start.
            r.user_text = user_text
            r.m.user_text = user_text
            r.m.eou = self.audio_to_wall(self.last_word_end)
            r.m.transcript = transcript_at
            r.confirm()
            return
        if r is not None and not r.done and r.user_text == user_text:
            return
        if r is not None and not r.released:
            await self._drop_speculative()
        self._start_response(user_text, eou_audio=self.last_word_end, transcript_at=transcript_at)

    def _start_response(self, user_text: str, eou_audio: float, transcript_at: float,
                        provisional: bool = False) -> None:
        self.turn_no += 1
        self.seq += 1
        m = TurnMetrics(persona=self.persona.key, turn=self.turn_no, eou=self.audio_to_wall(eou_audio),
                        transcript=transcript_at, language=self.lang, user_text=user_text, speculative=provisional)
        r = Response(self, user_text, m, self.seq, provisional=provisional)
        self.response = r
        r.start()  # the LLM runs now; audio waits on the gate
        asyncio.create_task(self._decide_turn(r))

    async def _decide_turn(self, r: Response) -> None:
        last_agent = next((m["content"] for m in reversed(self.history)
                           if m["role"] == "assistant" and m.get("content")), "")
        complete, method = await self.turn.is_complete(r.user_text, last_agent)
        if r is not self.response or r.done:
            return
        r.m.turn_decided = time.perf_counter()
        r.m.turn_method = method
        await self.send_json({"type": "turn", "complete": complete, "method": method})
        if complete:
            r.release()
            return
        await self._maybe_backchannel(long_monologue=False)
        self._hold_task = asyncio.create_task(self._release_after_hold(r))

    async def _release_after_hold(self, r: Response) -> None:
        await asyncio.sleep(self.cfg.max_incomplete_wait_ms / 1000)
        if r is self.response and not r.done:
            r.m.turn_method += "+timeout"
            r.release()

    async def _drop_speculative(self) -> None:
        r = self.response
        if r is None or r.released:
            return
        if self._hold_task:
            self._hold_task.cancel()
        await r.cancel()
        self.response = None

    def on_response_released(self, r: Response) -> None:
        self.history.append({"role": "user", "content": r.user_text})
        self.user_segments = []
        self.user_turn_started = None
        if r.event:
            return
        self.pending[r.seq] = r.m
        asyncio.create_task(self._metrics_fallback(r.seq))

    def on_response_finished(self, r: Response) -> None:
        self.history += r.new_messages
        text = r.full_text()
        if text:
            self.history.append({"role": "assistant", "content": text})
        asyncio.create_task(self.send_json({"type": "agent_text", "text": text, "final": True}))

    async def _barge_in(self) -> None:
        r = self.response
        if r is None or not r.released or r.interrupted:
            # Agent may still be audible from a finished response's buffered audio.
            if self._play_end > time.perf_counter():
                await self._clear_playback()
            return
        r.interrupted = True
        r.m.interrupted = True
        played_ms = r.sent_ms - max(0.0, self._play_end - time.perf_counter()) * 1000
        await r.cancel()
        await self._clear_playback()
        heard = r.heard_text(played_ms)
        # Keep tool results that completed (they happened), close any dangling tool calls.
        self.history += _close_tool_calls(r.new_messages)
        self.history.append({"role": "assistant", "content": (heard + " —") if heard else "—"})
        await self.send_json({"type": "barge_in", "heard": heard, "played_ms": round(played_ms)})
        log.info("barge-in after %.0fms; caller heard: %r", played_ms, heard)

    async def _clear_playback(self) -> None:
        self._play_end = time.perf_counter()
        await self.send_json({"type": "clear", "drop_below": self.seq + 1})
        self.seq += 1

    async def _maybe_duck(self) -> None:
        """Lower the agent's voice while the caller may be cutting in. Deepgram's SpeechStarted alone fires on
        any noise, including the agent's own voice leaking back through speakers, so both it and our VAD
        (120 ms of sound) must agree within a short window. The real stop still waits for confirmed words."""
        now = time.perf_counter()
        if not (self.agent_speaking and self.vad.in_speech and now - self._speech_hint_at < 1.5):
            return
        self._ducked_at = now
        await self.send_json({"type": "duck", "gain": self.cfg.duck_gain})
        asyncio.create_task(self._unduck_later())

    async def _unduck_later(self) -> None:
        ducked = self._ducked_at
        await asyncio.sleep(0.9)
        if self._ducked_at == ducked:
            await self.send_json({"type": "unduck"})

    # -- backchannels --------------------------------------------------------------------------------
    async def _maybe_backchannel(self, long_monologue: bool) -> None:
        now = time.perf_counter()
        if self.agent_speaking or self.user_turn_started is None:
            return
        if now - self._last_backchannel < self.cfg.backchannel_min_gap_s:
            return
        talked = now - self.user_turn_started
        if talked < (self.cfg.backchannel_min_user_speech_s * (2 if long_monologue else 1)):
            return
        clip = self.clips.pick(self.persona.key, "backchannel", self.lang)
        if clip is None:
            return
        self._last_backchannel = now
        await self.send_json({"type": "backchannel", "text": clip.text})
        await self.send_audio(0, clip.pcm)  # seq 0: side channel, mixed under the caller, never cleared

    # -- outbound ------------------------------------------------------------------------------------
    async def send_audio(self, seq: int, pcm: bytes) -> None:
        if seq != 0:
            now = time.perf_counter()
            self._play_end = max(self._play_end, now) + len(pcm) / self.bytes_per_ms_out / 1000
        async with self._send_lock:
            await self.ws.send_bytes(struct.pack("<I", seq) + pcm)

    async def send_json(self, msg: dict) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(msg, ensure_ascii=False))
            except Exception:
                pass

    # -- metrics -------------------------------------------------------------------------------------
    async def _metrics_fallback(self, seq: int) -> None:
        await asyncio.sleep(3)
        m = self.pending.pop(seq, None)
        if m is not None:
            self._write_metrics(m)

    def _write_metrics(self, m: TurnMetrics) -> None:
        rec = m.record()
        if rec is None:
            return
        self.latency.write(rec)
        log.info("turn %d latency: %s", m.turn, {k: rec[k] for k in ("stt", "llm_ttft", "chunking", "tts_ttfb",
                                                                     "turn_gate", "playout", "e2e")})
        asyncio.create_task(self.send_json({"type": "latency", **rec}))


def _recent(history: list[dict], max_messages: int = 16) -> list[dict]:
    """System prompt + the last few exchanges. Prompt length is time-to-first-token on every turn.
    The cut always starts at a user message so tool calls stay paired with their results."""
    system, rest = history[:1], history[1:]
    if len(rest) <= max_messages:
        return history
    tail = rest[-max_messages:]
    while tail and tail[0]["role"] != "user":
        tail = tail[1:]
    return system + tail


def _close_tool_calls(messages: list[dict]) -> list[dict]:
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    out = list(messages)
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if tc["id"] not in answered:
                out.append({"role": "tool", "tool_call_id": tc["id"],
                            "content": json.dumps({"error": "cancelled: caller interrupted"})})
    return out
