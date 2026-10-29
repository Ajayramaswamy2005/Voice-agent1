"""Turn-taking state machine tests with fake LLM / TTS / turn detector (no network)."""

import asyncio
import dataclasses
import json
import struct
import time

import pytest

from agent.clips import Clip, ClipBank
from agent.config import CONFIG
from agent.latency import LatencyLog
from agent.llm import TextDelta, ToolCalls, ToolCallStarted
from agent.personas import CLINIC
from agent.session import CallSession
from agent.stt import Transcript
from agent.tts import DONE, Audio, WordTimes

MS = 48  # bytes per ms at 24 kHz pcm16


class FakeWS:
    def __init__(self):
        self.audio: list[tuple[int, int]] = []  # (seq, nbytes)
        self.msgs: list[dict] = []

    async def send_bytes(self, b: bytes):
        self.audio.append((struct.unpack("<I", b[:4])[0], len(b) - 4))

    async def send_text(self, t: str):
        self.msgs.append(json.loads(t))

    def types(self):
        return [m["type"] for m in self.msgs]


class FakeCtx:
    def __init__(self, lang):
        self.lang = lang
        self.queue: asyncio.Queue = asyncio.Queue()
        self.cancelled = False
        self.t = 0.0

    async def push(self, text, final=False):
        # 300 ms of audio per word, streamed back after a short "TTFB"
        ws = text.split()
        await asyncio.sleep(0.01)
        for w in ws:
            self.queue.put_nowait(Audio(b"\0" * (300 * MS)))
            self.t += 0.3
            self.queue.put_nowait(WordTimes([w], [self.t - 0.3], [self.t]))

    async def end(self):
        self.queue.put_nowait(DONE)

    async def cancel(self):
        self.cancelled = True
        self.queue.put_nowait(DONE)

    async def events(self):
        while (ev := await self.queue.get()) is not DONE:
            yield ev


class FakeTTS:
    def __init__(self):
        self.contexts: list[FakeCtx] = []

    def context(self, lang):
        c = FakeCtx(lang)
        self.contexts.append(c)
        return c


class FakeLLM:
    def __init__(self, scripts):
        self.scripts = list(scripts)  # one list of events per call
        self.calls: list[list[dict]] = []
        self.models: list = []

    async def stream(self, messages, tools, model=None):
        self.models.append(model)
        self.calls.append(messages)
        for ev in self.scripts.pop(0) if self.scripts else [TextDelta("Okay.")]:
            await asyncio.sleep(0.005)
            yield ev


class FakeTurn:
    def __init__(self, answers):
        self.answers = list(answers)

    async def is_complete(self, user, agent):
        await asyncio.sleep(0.02)
        return (self.answers.pop(0) if self.answers else True), "model"


class FakeHub:
    def __init__(self):
        self.tools = {"clinic": [{"type": "function", "function": {"name": "check_availability"}}]}
        self.called: list[str] = []

    async def call(self, server, name, args):
        self.called.append(name)
        await asyncio.sleep(0.05)
        return json.dumps({"slots": ["10:00"]})


def clips():
    bank = ClipBank()
    for kind, text in [("filler", "Let me check that for you."), ("slow_filler", "Still looking."),
                       ("backchannel", "Mm-hm."), ("greeting", "Hi.")]:
        bank._clips[("clinic", kind, "en")] = [Clip(text, b"\0" * (500 * MS))]
    return bank


def make(tmp_path, llm_scripts, turn_answers=(), **cfg):
    cfg = dataclasses.replace(CONFIG, latency_log=tmp_path / "lat.jsonl", **cfg)
    ws, llm, hub = FakeWS(), FakeLLM(llm_scripts), FakeHub()
    s = CallSession(ws, cfg, CLINIC, hub, llm, FakeTurn(turn_answers), FakeTTS(), clips(),
                    LatencyLog(cfg.latency_log))
    return s, ws, llm, hub


def tr(text, final=True, speech_final=True, end=1.0):
    return Transcript(text=text, is_final=final, speech_final=speech_final, start=0.0, end=end, language="en",
                      received_at=time.perf_counter())


async def test_speculative_response_waits_for_turn_detector_then_hold_timeout(tmp_path):
    s, ws, llm, _ = make(tmp_path, [[TextDelta("Sure, which day?")]], turn_answers=[False],
                         max_incomplete_wait_ms=300)
    await s.on_stt(tr("I want to book an appointment"))
    await asyncio.sleep(0.15)
    assert llm.calls, "LLM should start speculatively"
    assert not ws.audio, "audio must be gated while the detector says the caller isn't done"
    await asyncio.sleep(0.4)
    assert ws.audio, "after max_incomplete_wait the agent answers anyway"
    assert s.history[-1]["role"] in ("user", "assistant")


async def test_caller_resuming_drops_speculative_response(tmp_path):
    s, ws, llm, _ = make(tmp_path, [[TextDelta("First answer.")], [TextDelta("Second answer.")]],
                         turn_answers=[False, True], max_incomplete_wait_ms=2000)
    await s.on_stt(tr("I need an appointment for my"))
    await asyncio.sleep(0.05)
    await s.on_stt(tr("son", final=False, speech_final=False))  # caller keeps going
    await s.on_stt(tr("son, he has a fever", end=2.5))
    await asyncio.sleep(0.2)
    assert len(llm.calls) == 2
    assert llm.calls[1][-1]["content"] == "I need an appointment for my son, he has a fever"
    users = [m["content"] for m in s.history if m["role"] == "user"]
    assert users == ["I need an appointment for my son, he has a fever"]
    assert ws.audio


async def test_barge_in_stops_playback_and_truncates_history(tmp_path):
    long = "We have slots at ten, ten thirty, eleven, eleven thirty and also in the afternoon."
    s, ws, llm, _ = make(tmp_path, [[TextDelta(long)]])
    await s.on_stt(tr("Any slots tomorrow?"))
    await asyncio.sleep(0.25)
    assert s.agent_speaking
    await s.on_stt(tr("okay", final=False, speech_final=False))  # acknowledgement: keep talking
    assert "clear" not in ws.types()
    await s.on_stt(tr("wait, I need Friday", final=False, speech_final=False))
    assert "clear" in ws.types()
    barge = next(m for m in ws.msgs if m["type"] == "barge_in")
    last = s.history[-1]
    assert last["role"] == "assistant" and last["content"].endswith("—")
    assert len(last["content"]) < len(long)  # only what the caller heard
    assert barge["heard"] in last["content"]
    assert not s.agent_speaking


async def test_tool_call_plays_filler_and_runs_tool_after_gate(tmp_path):
    s, ws, llm, hub = make(tmp_path, [
        [ToolCallStarted("check_availability"),
         ToolCalls([{"id": "c1", "name": "check_availability", "arguments": '{"date_str": "2026-10-29"}'}])],
        [TextDelta("Ten o'clock is free.")],
    ])
    await s.on_stt(tr("Is Dr Rao free tomorrow?"))
    await asyncio.sleep(0.4)
    assert hub.called == ["check_availability"]
    assert ws.audio[0][1] == 500 * MS, "first audio out is the pre-rendered filler"
    roles = [m["role"] for m in s.history]
    assert roles[-3:] == ["assistant", "tool", "assistant"]
    assert s.history[-3]["tool_calls"][0]["function"]["name"] == "check_availability"
    assert s.history[-1]["content"] == "Ten o'clock is free."  # filler played, but kept out of history
    assert s.response.m.path == "filler"


async def test_latency_record_written_on_playout_report(tmp_path):
    s, ws, llm, _ = make(tmp_path, [[TextDelta("Sure.")]])
    await s.on_stt(tr("Hello"))
    await asyncio.sleep(0.15)
    seq = s.response.seq
    await s.on_client_message({"type": "playout", "seq": seq, "delay_ms": 25})
    recs = s.latency.read()
    assert len(recs) == 1 and recs[0]["playout"] == 25
    assert recs[0]["e2e"] == pytest.approx(sum(recs[0][k] for k in
                                               ("stt", "llm_ttft", "chunking", "tts_ttfb", "turn_gate", "playout")),
                                           abs=1)


def interim(text, end):
    return Transcript(text=text, is_final=False, speech_final=False, start=0.0, end=end, language="en",
                      received_at=time.perf_counter())


def finalized(text, end):
    return Transcript(text=text, is_final=True, speech_final=False, start=0.0, end=end, language="en",
                      received_at=time.perf_counter(), from_finalize=True)


async def test_interim_speculation_confirmed_by_final_keeps_head_start(tmp_path):
    s, ws, llm, _ = make(tmp_path, [[TextDelta("Yes, ten and eleven are open.")]])
    await s.on_stt(interim("Any slots tomorrow", end=1.0))
    await s._on_pause(1.1)  # our VAD heard the pause; the interim covers the speech
    await asyncio.sleep(0.1)
    assert len(llm.calls) == 1, "LLM starts on the interim, before the final transcript"
    assert not s.tts.contexts and not ws.audio, "no TTS or audio until the final confirms the text"
    await s.on_stt(finalized("Any slots tomorrow?", end=1.0))
    await asyncio.sleep(0.15)
    assert len(llm.calls) == 1, "matching final keeps the speculative response"
    assert ws.audio and s.response.m.speculative
    assert [m["content"] for m in s.history if m["role"] == "user"] == ["Any slots tomorrow?"]


async def test_interim_speculation_discarded_when_final_differs(tmp_path):
    s, ws, llm, _ = make(tmp_path, [[TextDelta("Wrong.")], [TextDelta("Friday works.")]])
    await s.on_stt(interim("Any slots on", end=1.0))
    await s._on_pause(1.1)
    await asyncio.sleep(0.05)
    await s.on_stt(finalized("Any slots on Friday?", end=1.4))
    await asyncio.sleep(0.2)
    assert len(llm.calls) == 2
    assert llm.calls[1][-1]["content"] == "Any slots on Friday?"
    assert "Friday works." in s.history[-1]["content"]


async def test_no_speculation_when_interim_lags_the_audio(tmp_path):
    s, ws, llm, _ = make(tmp_path, [[TextDelta("Hm.")]])
    await s.on_stt(interim("Actually do you also have a", end=2.1))
    await s._on_pause(3.9)  # speech went on well past the interim's last word
    await asyncio.sleep(0.05)
    assert not llm.calls


async def test_pseudo_tool_call_text_is_never_spoken_and_retried_on_fallback(tmp_path):
    s, ws, llm, hub = make(tmp_path, [
        [TextDelta("[Calling: check_availability with date_str=2026-10-29]")],  # model wrote the call as text
        [ToolCallStarted("check_availability"),
         ToolCalls([{"id": "c1", "name": "check_availability", "arguments": '{"date_str": "2026-10-29"}'}])],
        [TextDelta("Ten is free.")],
    ])
    await s.on_stt(tr("Is anyone free tomorrow?"))
    await asyncio.sleep(0.4)
    assert llm.models[:2] == [None, CONFIG.llm_fallback_model]
    assert hub.called == ["check_availability"]
    spoken = " ".join(s.history[-1]["content"].split())
    assert "Calling" not in spoken and "Ten is free." in spoken


async def test_filler_clips_are_kept_out_of_history(tmp_path):
    s, ws, llm, hub = make(tmp_path, [
        [ToolCallStarted("check_availability"),
         ToolCalls([{"id": "c1", "name": "check_availability", "arguments": "{}"}])],
        [TextDelta("Ten is free.")],
    ])
    await s.on_stt(tr("Is anyone free tomorrow?"))
    await asyncio.sleep(0.4)
    assert ws.audio[0][1] == 500 * MS, "the filler still plays"
    assert s.history[-1]["content"] == "Ten is free."


async def test_restated_phrase_after_tool_call_is_not_spoken_twice(tmp_path):
    s, ws, llm, hub = make(tmp_path, [
        [TextDelta("I'll check that order now. "), ToolCallStarted("check_availability"),
         ToolCalls([{"id": "c1", "name": "check_availability", "arguments": "{}"}])],
        [TextDelta("I'll check that order now. It ships Friday.")],
    ])
    await s.on_stt(tr("Where is my order?"))
    await asyncio.sleep(0.4)
    assert s.history[-1]["content"].count("I'll check that order now.") == 1
    assert "It ships Friday." in s.history[-1]["content"]
