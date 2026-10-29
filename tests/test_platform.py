"""Phone bridge, line configs, browser-agent safety gate, agent-initiated turns (no network)."""

import array
import asyncio
import json
import math
import struct

from agent.personas import load_lines
from agent.phone import TwilioBridge, pcm24k_to_ulaw, ulaw_to_pcm16k, valid_signature
from mcp_servers.browser_agent import parse_reply, risk_reason
from tests.test_session import make


# -- phone ------------------------------------------------------------------------------------------
def _tone(rate: int, n: int, amp: int = 8000) -> bytes:
    return array.array("h", (int(amp * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n))).tobytes()


def test_ulaw_round_trip_keeps_the_signal():
    ulaw = pcm24k_to_ulaw(_tone(24000, 2400))  # 100 ms
    assert len(ulaw) == 800  # 8 kHz
    back = array.array("h")
    back.frombytes(ulaw_to_pcm16k(ulaw))
    assert len(back) == 1600  # 16 kHz
    ref = array.array("h")
    ref.frombytes(_tone(16000, 1600))
    # Resampling shifts phase slightly, so compare shape (correlation) rather than sample-by-sample.
    dot = sum(a * b for a, b in zip(back, ref))
    corr = dot / math.sqrt(sum(a * a for a in back) * sum(b * b for b in ref))
    assert corr > 0.97, f"correlation {corr:.3f}"


class FakeTwilioWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, t):
        self.sent.append(json.loads(t))


async def test_twilio_bridge_marks_clears_and_drops_interrupted_audio():
    ws = FakeTwilioWS()
    b = TwilioBridge(ws, "MZ1")
    await b.send_bytes(struct.pack("<I", 1) + _tone(24000, 480))
    assert [m["event"] for m in ws.sent] == ["mark", "media"]
    assert b.on_mark("1")["seq"] == 1
    await b.send_text(json.dumps({"type": "clear", "drop_below": 2}))
    assert ws.sent[-1] == {"event": "clear", "streamSid": "MZ1"}
    n = len(ws.sent)
    await b.send_bytes(struct.pack("<I", 1) + _tone(24000, 480))  # late audio from the interrupted reply
    assert len(ws.sent) == n
    await b.send_text(json.dumps({"type": "agent_text", "text": "hi"}))  # UI-only messages are dropped
    assert len(ws.sent) == n


def test_twilio_signature():
    # Example from Twilio's webhook security docs.
    params = {"CallSid": "CA1234567890ABCDE", "Caller": "+12349013030", "Digits": "1234", "From": "+12349013030",
              "To": "+18005551212"}
    url = "https://mycompany.com/myapp.php?foo=1&bar=2"
    assert valid_signature("12345", url, params, "0/KCTR6DLpKmkAf8muzZqo1nDgQ=")
    assert not valid_signature("12345", url, params, "wrong")


# -- lines ------------------------------------------------------------------------------------------
def test_lines_load_from_toml_and_filter(tmp_path):
    (tmp_path / "pizza.toml").write_text(
        'name = "Pizza"\nmcp_command = ["npx", "pizza-mcp"]\nprompt = "You take orders."\n'
        '[greeting]\nen = "Pizza place!"\n', encoding="utf-8")
    (tmp_path / "other.toml").write_text('prompt = "x"\n', encoding="utf-8")
    lines = load_lines(tmp_path, enabled="pizza")
    assert list(lines) == ["pizza"]
    p = lines["pizza"]
    assert p.mcp_command == ("npx", "pizza-mcp") and p.mcp_server == "pizza" and p.greeting["en"] == "Pizza place!"


def test_browser_line_hides_app_only_tools():
    p = load_lines()["browser"]
    assert {"wait_browser_event", "browser_details", "open_page"} <= p.hidden_tools()


# -- browser safety gate ----------------------------------------------------------------------------
def test_risky_actions_need_approval():
    submit = {"tag": "button", "type": "submit", "label": "Submit application"}
    assert risk_reason({"action": "click", "mark": 1}, submit, "jobs.example")
    assert risk_reason({"action": "click", "mark": 1}, {"tag": "a", "type": "", "label": "Delete account"}, "")
    assert risk_reason({"action": "type", "mark": 2, "text": "x"}, {"tag": "input", "type": "password",
                                                                  "label": "Password", "name": "pw"}, "")
    assert risk_reason({"action": "upload", "mark": 3}, {"tag": "input", "type": "file", "label": "Resume"}, "")
    assert risk_reason({"action": "navigate", "text": "https://evil.example/x"}, None, "jobs.example")


def test_ordinary_actions_run_without_asking():
    assert not risk_reason({"action": "type", "mark": 1, "text": "Ananya"},
                           {"tag": "input", "type": "text", "label": "First name", "name": "first_name"}, "")
    assert not risk_reason({"action": "click", "mark": 4}, {"tag": "input", "type": "checkbox",
                                                            "label": "I can work from the office"}, "")
    assert not risk_reason({"action": "scroll", "text": "down"}, None, "")
    assert not risk_reason({"action": "navigate", "text": "https://jobs.example/apply"}, None, "jobs.example")


def test_parse_reply_tolerates_fences_and_prose():
    r = parse_reply('Sure!\n```json\n{"observation": "form", "actions": [{"action": "scroll"}]}\n```')
    assert r["actions"][0]["action"] == "scroll"
    assert parse_reply('{"done": true, "summary": "ok"}')["actions"] == []


# -- agent-initiated turns --------------------------------------------------------------------------
async def test_notify_speaks_an_update_without_counting_it_as_a_latency_turn(tmp_path):
    from agent.llm import TextDelta

    s, ws, llm, _ = make(tmp_path, [[TextDelta("The form is filled. Should I submit it?")]])
    await s.notify("Approval needed: click Submit")
    await asyncio.sleep(0.6)
    assert llm.calls[-1][-1] == {"role": "user", "content": "[update] Approval needed: click Submit"}
    assert ws.audio, "the update is spoken"
    assert not s.pending, "event turns stay out of the latency stats"


async def test_notify_waits_for_the_caller_to_finish():
    from agent.llm import TextDelta
    import tempfile
    from pathlib import Path

    s, ws, llm, _ = make(Path(tempfile.mkdtemp()), [[TextDelta("Done.")]])
    s.user_segments = ["I want to"]  # caller is mid-turn
    task = asyncio.create_task(s.notify("Browser task finished"))
    await asyncio.sleep(0.3)
    assert not llm.calls
    s.user_segments = []
    await asyncio.wait_for(task, 1)
    assert llm.calls


# -- browser assistant: allowed sites, stricter gates, queue, plan review ---------------------------
def test_allowed_sites_include_subdomains_only():
    from mcp_servers.browser_agent import site_allowed

    allowed = ("linkedin.com", "naukri.com")
    assert site_allowed("www.linkedin.com", allowed) and site_allowed("naukri.com", allowed)
    assert not site_allowed("evil-linkedin.com", allowed)
    assert not site_allowed("google.com", allowed)
    assert site_allowed("anything.example", ())  # no list = any site


def test_actions_on_signed_in_sites_need_approval():
    for label in ("Send", "Reply all", "Post", "Connect", "Follow", "Share"):
        assert risk_reason({"action": "click", "mark": 1}, {"tag": "button", "type": "", "label": label}, ""), label
    assert risk_reason({"action": "type", "mark": 1, "text": "x"},
                       {"tag": "input", "type": "text", "label": "UPI ID", "name": "vpa"}, "")
    assert risk_reason({"action": "new_tab", "text": "https://other.example"}, None, "jobs.example")
    assert not risk_reason({"action": "new_tab", "text": "https://careers.jobs.example"}, None, "jobs.example")
    assert not risk_reason({"action": "read"}, None, "") and not risk_reason({"action": "note", "text": "x"}, None, "")


class _NoLLM:
    pass


async def test_tasks_queue_while_one_runs(tmp_path):
    from mcp_servers.browser_agent import BrowserAgent

    a = BrowserAgent(_NoLLM(), "m", tmp_path)
    a.state = "running"
    assert a.start("second task").startswith("queued")
    assert [q["goal"] for q in a.queue] == ["second task"]
    assert await a.stop() == "Nothing is running." and not a.queue  # stop also clears the queue


async def test_plan_review_can_be_edited(tmp_path):
    from mcp_servers.browser_agent import BrowserAgent, Pending

    a = BrowserAgent(_NoLLM(), "m", tmp_path)
    assert "no plan" in await a.revise_plan(steps=["x"])
    a.state, a.pending = "plan_review", Pending("start this plan", kind="plan")
    a.plan = [{"text": "old", "done": False}]
    assert await a.revise_plan(steps=["Open LinkedIn", " ", "Search jobs"]) == "plan updated"
    assert [p["text"] for p in a.plan] == ["Open LinkedIn", "Search jobs"]
    assert a.events[-1].startswith("Plan updated")
    assert await a.revise_plan(instructions="also check Naukri") == "re-planning"
    assert a.pending.kind == "replan" and a.pending.decided.is_set()


def test_audit_log_round_trip(tmp_path):
    from mcp_servers.browser_agent import BrowserAgent

    a = BrowserAgent(_NoLLM(), "m", tmp_path)
    a.goal = "g"
    a.audit("action", action={"action": "click", "mark": 3}, result="ok")
    a.audit("approval", description="click Send", approved=False)
    log = a.read_audit()
    assert [e["kind"] for e in log] == ["approval", "action"]  # newest first


def test_rate_limit_wait_is_parsed_from_groq_message():
    from mcp_servers.browser_agent import _retry_after_s

    assert _retry_after_s("Please try again in 12m22.616s. Need more tokens?") == 12 * 60 + 22.616
    assert _retry_after_s("try again in 3.5s") == 3.5
    assert _retry_after_s("try again in 1h2m") == 3720
    assert _retry_after_s("rate limited") is None


# -- ducking ----------------------------------------------------------------------------------------
async def test_noise_alone_does_not_duck_but_real_speech_does(tmp_path):
    import time

    from agent.stt import SpeechStarted

    s, ws, _, _ = make(tmp_path, [])
    s._play_end = time.perf_counter() + 5  # the agent is talking
    await s.on_stt(SpeechStarted(audio_time=0.0, received_at=time.perf_counter()))
    assert "duck" not in ws.types(), "Deepgram's hint alone (e.g. echo or a fan) must not lower the volume"
    s.vad.in_speech = True  # our VAD also hears 120 ms of sound
    await s._maybe_duck()
    duck = next(m for m in ws.msgs if m["type"] == "duck")
    assert duck["gain"] == 0.6
