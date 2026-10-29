import asyncio
from datetime import date

import pytest

from agent.latency import STAGES, TurnMetrics, summarize
from agent.llm import _TRAILING_INCOMPLETE
from agent.text import PhraseChunker, detect_language, is_ack_only


def test_detect_language():
    assert detect_language("I need an appointment tomorrow") == "en"
    assert detect_language("मुझे कल अपॉइंटमेंट चाहिए") == "hi"
    assert detect_language("mera order kab aayega, kya aap check kar sakte hai") == "hi"
    assert detect_language("") == "en"


def test_ack_only():
    for t in ["Mm-hm.", "okay", "haan ji", "हाँ", "yeah yeah", "Right."]:
        assert is_ack_only(t), t
    for t in ["wait", "no, that's wrong", "okay but I wanted Friday", "हाँ लेकिन कल नहीं"]:
        assert not is_ack_only(t), t


def test_chunker_releases_first_clause_early_then_sentences():
    c = PhraseChunker()
    out = []
    for tok in ["Sure", ", I can", " help with", " that. ", "Which", " day works", " best", " for you?", " "]:
        out += c.push(tok)
    out += c.flush()
    assert out[0] == "Sure, I can help with that."
    assert out[-1] == "Which day works best for you?"


def test_chunker_hindi_danda():
    c = PhraseChunker()
    out = c.push("जी, मैं चेक करती हूँ। ") + c.push("आपका ऑर्डर कल आएगा।") + c.flush()
    assert out == ["जी, मैं चेक करती हूँ।", "आपका ऑर्डर कल आएगा।"]


@pytest.mark.parametrize("text,incomplete", [
    ("I want to book an appointment for my", True),
    ("I want to book an appointment and", True),
    ("mujhe appointment chahiye kyunki", True),
    ("मेरा ऑर्डर और", True),
    ("I want to book an appointment.", False),
    ("Tomorrow morning works", False),
    ("हाँ, बुक कर दीजिए", False),
])
def test_trailing_heuristic(text, incomplete):
    assert bool(_TRAILING_INCOMPLETE.search(text)) == incomplete


def test_breakdown_sums_to_e2e():
    m = TurnMetrics(persona="clinic", turn=1, eou=10.0, transcript=10.25)
    m.llm_first_token, m.first_phrase, m.first_audio, m.released = 10.45, 10.50, 10.62, 10.62
    m.playout_ms = 30
    b = m.breakdown()
    assert b == {"stt": 250.0, "llm_ttft": 200.0, "chunking": 50.0, "tts_ttfb": 120.0, "turn_gate": 0.0,
                 "playout": 30.0, "e2e": 650.0}
    assert sum(b[s] for s in STAGES) == pytest.approx(b["e2e"])


def test_breakdown_turn_gate_when_detector_slower():
    m = TurnMetrics(persona="clinic", turn=1, eou=0.0, transcript=0.2)
    m.llm_first_token, m.first_phrase, m.first_audio, m.released = 0.4, 0.45, 0.55, 0.9
    assert m.breakdown()["turn_gate"] == pytest.approx(350.0)


def test_summarize():
    recs = [{**{s: 10.0 for s in STAGES}, "e2e": v} for v in (500, 600, 800)]
    s = summarize(recs)
    assert s["e2e"]["p50"] == 600 and s["under_700ms"] == pytest.approx(0.667, abs=1e-3)


async def test_clinic_server_booking_flow(tmp_path, monkeypatch):
    from mcp_servers import clinic

    monkeypatch.setattr(clinic, "DB_PATH", tmp_path / "clinic.db")
    monkeypatch.setattr(clinic, "LATENCY_S", (0, 0))
    clinic.seed(date(2026, 10, 26))  # a Monday
    avail = await clinic.check_availability("2026-10-27", specialty="dermatology")
    slot = avail["doctors"][0]["slots"][0]
    booked = await clinic.book_appointment("D2", f"2026-10-27 {slot}", "Kavya Iyer", "+91 98450 12345", "rash")
    assert booked["ok"]
    again = await clinic.book_appointment("D2", f"2026-10-27 {slot}", "Someone", "9999999999")
    assert not again["ok"]
    mine = await clinic.find_appointments("9845012345")
    assert mine[0]["id"] == booked["appointment_id"]
    assert (await clinic.cancel_appointment(booked["appointment_id"]))["ok"]
    assert (await clinic.check_availability("2026-11-01"))["note"].startswith("Clinic is closed")


async def test_support_server_refund_rules(tmp_path, monkeypatch):
    from mcp_servers import support

    monkeypatch.setattr(support, "DB_PATH", tmp_path / "support.db")
    monkeypatch.setattr(support, "LATENCY_S", (0, 0))
    support.seed()
    assert (await support.request_refund("od 1001", "stopped charging"))["ok"]
    assert "after delivery" in (await support.request_refund("OD1002", "changed mind"))["error"]
    assert "window" in (await support.request_refund("1004", "broken"))["error"]
    t = await support.create_ticket("9988776655", "delivery", "late", order_id="OD1005", priority="high")
    assert t["sla"] == "24 hours"
    assert (await support.ticket_status(t["ticket_id"]))["status"] == "open"
    assert (await support.track_order("OD1002"))["expected_delivery"]


def test_vad_detects_pause_after_speech():
    import array as _array
    import math as _math

    from agent.vad import EnergyVAD

    vad = EnergyVAD(16000, silence_ms=100)
    tone = _array.array("h", (int(8000 * _math.sin(i / 5)) for i in range(320))).tobytes()
    quiet = bytes(640)
    events = [vad.update(quiet) for _ in range(10)]
    events += [vad.update(tone) for _ in range(25)]  # 500 ms of "speech"
    events += [vad.update(quiet) for _ in range(10)]
    ends = [e for e in events if e is not None]
    assert len(ends) == 1 and ends[0] == pytest.approx(0.7, abs=0.02)  # speech ended at 200 + 500 ms


def test_vad_ignores_short_blips():
    from agent.vad import EnergyVAD

    vad = EnergyVAD(16000)
    click = bytes([0xFF, 0x3F] * 320)
    events = [vad.update(bytes(640)) for _ in range(5)] + [vad.update(click)] + [vad.update(bytes(640)) for _ in range(10)]
    assert all(e is None for e in events)


def test_breakdown_hides_llm_behind_stt_when_speculative():
    m = TurnMetrics(persona="clinic", turn=1, eou=0.0, transcript=0.40)
    m.llm_first_token, m.first_phrase, m.first_audio, m.released = 0.30, 0.32, 0.50, 0.50
    b = m.breakdown()
    assert b["llm_ttft"] == 0 and b["chunking"] == 0 and b["tts_ttfb"] == pytest.approx(100)
    assert b["e2e"] == pytest.approx(500)


def test_compact_schema():
    from agent.mcp_hub import compact_schema

    raw = {"properties": {"date_str": {"title": "Date Str", "type": "string"},
                          "doctor_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None,
                                        "title": "Doctor Id"}},
           "required": ["date_str"], "type": "object", "title": "check_availabilityArguments"}
    assert compact_schema(raw) == {"properties": {"date_str": {"type": "string"}, "doctor_id": {"type": "string"}},
                                   "required": ["date_str"], "type": "object"}


def test_recent_history_keeps_tool_pairs():
    from agent.session import _recent

    h = [{"role": "system", "content": "s"}]
    for i in range(10):
        h += [{"role": "user", "content": f"u{i}"},
              {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]},
              {"role": "tool", "tool_call_id": f"c{i}", "content": "{}"},
              {"role": "assistant", "content": f"a{i}"}]
    r = _recent(h)
    assert r[0]["role"] == "system" and r[1]["role"] == "user" and len(r) <= 17
    assert r[-1]["content"] == "a9"
