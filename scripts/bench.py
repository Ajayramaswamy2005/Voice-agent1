"""Scripted-caller benchmark: talks to a running agent over its WebSocket and records per-turn latency.

    uv run python -m agent.main          # terminal 1
    uv run python -m scripts.bench       # terminal 2  (--persona clinic|support|all, --lang en|hi|all)

Caller lines are synthesized once with Cartesia (16 kHz, cached in data/bench_audio/) and streamed in
real time, 20 ms per frame, exactly like the browser does. A "|" in a line inserts a mid-thought pause,
which exercises the semantic turn detector (the agent should *not* answer at the pause).

Playout is reported as 0 ms: there is no speaker here, so add your browser's playout (~20-60 ms) when
comparing with live calls.
"""

import argparse
import asyncio
import hashlib
import json
import struct
import time

import websockets

from agent.config import CONFIG, ROOT
from agent.latency import LatencyLog, summarize
from agent.tts import CartesiaTTS

AUDIO_DIR = ROOT / "data" / "bench_audio"
CALLER_VOICE = None  # None = agent's voice; set an id to use a different caller voice
PAUSE_S = 0.7
FRAME = 320 * 2  # 20 ms @ 16 kHz, 16-bit

SCRIPTS = {
    ("clinic", "en"): [
        "Hi, I'd like to book an appointment with a skin doctor.",
        "Tomorrow, if possible, | sometime in the morning.",
        "The first slot is fine. My name is Kavya Iyer, | and my number is nine eight four five zero one two three four five.",
        "Yes, please go ahead.",
        "Actually, do you also have a pediatrician available on Friday?",
    ],
    ("clinic", "hi"): [
        "नमस्ते, मुझे डॉक्टर से मिलना है, | बुखार है दो दिन से।",
        "कल सुबह का कोई टाइम मिलेगा?",
        "पहला वाला ठीक है। मेरा नाम रोहित वर्मा है, नंबर नौ आठ सात छह पांच चार तीन दो एक शून्य।",
        "हाँ, बुक कर दीजिए।",
    ],
    ("support", "en"): [
        "Hi, I want to check on my order, OD1002.",
        "Okay. I also want to return the headphones from order OD1001, | they stopped charging after two days.",
        "Yes, please start the return.",
        "Thanks, that's all.",
    ],
    ("support", "hi"): [
        "हेलो, मेरा ऑर्डर OD1005 अभी तक नहीं आया है।",
        "कब तक आ जाएगा? | मुझे वो बैग स्कूल के लिए चाहिए।",
        "ठीक है, इसको हाई प्रायोरिटी टिकट बना दीजिए, मेरा नंबर नौ नौ आठ आठ सात सात छह छह पांच पांच है।",
        "ठीक है, धन्यवाद।",
    ],
}


async def render_lines(tts: CartesiaTTS, lang: str, line: str) -> list[bytes]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for piece in (p.strip() for p in line.split("|")):
        key = hashlib.sha1(f"{CONFIG.cartesia_model}|{CALLER_VOICE}|{lang}|{piece}".encode()).hexdigest()[:16]
        path = AUDIO_DIR / f"{key}.pcm"
        for _ in range(4):
            if path.exists():
                break
            if tts.ws is None:  # connect lazily: cached lines need no TTS at all
                await tts.connect()
            pcm = await tts.synthesize(piece, lang, voice=CALLER_VOICE, sample_rate=16000)
            if pcm:
                path.write_bytes(pcm)
            else:
                await asyncio.sleep(2)
                await tts.close()
                await tts.connect()
        parts.append(path.read_bytes())
    return parts


class Caller:
    def __init__(self, ws):
        self.ws = ws
        self.last_audio = 0.0
        self.got_audio_since = False
        self.play_end = 0.0  # when the agent's audio would finish playing on a real speaker
        self.agent_done = False  # agent finished its whole reply (not just a "let me check" filler)
        self.clock0: float | None = None
        self.sent = 0  # bytes of caller audio sent; pacing is against one clock for the whole call
        self.turns_complete = 0
        self.latencies: list[dict] = []

    async def recv_loop(self) -> None:
        async for msg in self.ws:
            if isinstance(msg, bytes):
                seq = struct.unpack("<I", msg[:4])[0]
                if seq != 0:
                    if not self.got_audio_since:
                        await self.ws.send(json.dumps({"type": "playout", "seq": seq, "delay_ms": 0}))
                    self.got_audio_since = True
                    now = time.monotonic()
                    self.last_audio = now
                    self.play_end = max(self.play_end, now) + (len(msg) - 4) / 48000
                continue
            m = json.loads(msg)
            if m["type"] == "clear":
                self.play_end = time.monotonic()
            if m["type"] == "barge_in" or (m["type"] == "agent_text" and m.get("final")):
                self.agent_done = True
            if m["type"] == "turn":
                print(f"    [turn {'complete' if m['complete'] else 'NOT done'} via {m['method']}]")
            elif m["type"] == "tool":
                print(f"    [tool → {m['name']}]")
            elif m["type"] == "backchannel":
                print(f"    [backchannel {m['text']}]")
            elif m["type"] == "agent_text" and m.get("text"):
                print(f"  agent: {m['text']}")
            elif m["type"] == "latency":
                self.latencies.append(m)
                print(f"    [latency e2e {m['e2e']:.0f} ms = stt {m['stt']:.0f} + llm {m['llm_ttft']:.0f} + "
                      f"phrase {m['chunking']:.0f} + tts {m['tts_ttfb']:.0f} + gate {m['turn_gate']:.0f}  {m.get('model', '')}]")

    async def stream(self, pcm: bytes) -> None:
        if self.clock0 is None:
            self.clock0 = time.monotonic()
        for i in range(0, len(pcm), FRAME):
            await self.ws.send(pcm[i:i + FRAME])
            self.sent += FRAME
            # pace against one fixed clock so sleep overshoot (~15 ms on Windows) never accumulates
            await asyncio.sleep(max(0.0, self.clock0 + self.sent / 32000 - time.monotonic()))

    async def silence(self, seconds: float) -> None:
        await self.stream(b"\x00" * int(seconds * 32000 // FRAME * FRAME))

    async def wait_for_agent(self, timeout: float = 15.0) -> None:
        """Keep sending silence until the agent has spoken and gone quiet."""
        t0 = time.monotonic()
        self.got_audio_since = False
        self.agent_done = False
        while time.monotonic() - t0 < timeout:
            await self.silence(0.2)
            now = time.monotonic()
            # audio arrives faster than real time: wait until it would have finished *playing*
            if (self.got_audio_since and self.agent_done and now - self.last_audio > 1.0
                    and now > self.play_end + 0.4):
                return


async def run_call(persona: str, lang: str, tts: CartesiaTTS) -> list[dict]:
    lines = SCRIPTS[(persona, lang)]
    rendered = [await render_lines(tts, lang, line) for line in lines]
    url = f"ws://127.0.0.1:{CONFIG.port}/ws?persona={persona}&lang={lang}"
    async with websockets.connect(url, max_size=None) as ws:
        caller = Caller(ws)
        recv = asyncio.create_task(caller.recv_loop())
        print(f"\n=== {persona} / {lang} ===")
        await caller.wait_for_agent()  # greeting
        for line, parts in zip(lines, rendered):
            print(f"  caller: {line.replace(' | ', ' … ')}")
            for j, pcm in enumerate(parts):
                await caller.stream(pcm)
                if j < len(parts) - 1:
                    await caller.silence(PAUSE_S)
            await caller.wait_for_agent()
        await caller.silence(3.5)  # let the last playout/metrics land
        recv.cancel()
        return caller.latencies


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="all", choices=["clinic", "support", "all"])
    ap.add_argument("--lang", default="all", choices=["en", "hi", "all"])
    args = ap.parse_args()
    tts = CartesiaTTS(CONFIG)
    recs: list[dict] = []
    try:
        for persona, lang in SCRIPTS:
            if args.persona in ("all", persona) and args.lang in ("all", lang):
                recs += await run_call(persona, lang, tts)
    finally:
        if tts.ws is not None:
            await tts.close()
    s = summarize(recs)
    print(f"\n{s['n']} turns · e2e p50 {s['e2e']['p50']} ms · p90 {s['e2e']['p90']} ms · "
          f"{s['under_700ms'] * 100:.0f}% under 700 ms")
    for k in ("stt", "llm_ttft", "chunking", "tts_ttfb", "turn_gate"):
        print(f"  {k:<10} p50 {s[k]['p50']:>6} ms   p90 {s[k]['p90']:>6} ms")
    print(f"\nAll-time log: {LatencyLog(CONFIG.latency_log).path}  ·  dashboard: http://localhost:{CONFIG.port}/latency")


if __name__ == "__main__":
    asyncio.run(main())
