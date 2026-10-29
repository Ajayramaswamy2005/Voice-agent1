import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Config:
    deepgram_key: str = field(default_factory=lambda: _env("DEEPGRAM_API_KEY"))
    groq_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    cartesia_key: str = field(default_factory=lambda: _env("CARTESIA_API_KEY"))

    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen/qwen3.8-27b"))
    # Used immediately when the primary is rate-limited (Groq free tier: 8k tokens/min per model).
    llm_fallback_model: str = field(default_factory=lambda: _env("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b"))
    turn_model: str = field(default_factory=lambda: _env("TURN_MODEL", "qwen/qwen3.8-27b"))

    # Use the Deepgram region nearest your server: every transcript pays a round trip. From India,
    # api.in.deepgram.com is ~45 ms away vs ~270 ms for the global endpoint.
    deepgram_host: str = field(default_factory=lambda: _env("DEEPGRAM_HOST", "api.deepgram.com"))
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "nova-3"))
    stt_language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "multi"))
    # Short endpointing: we let the semantic turn detector decide, not silence.
    stt_endpointing_ms: int = field(default_factory=lambda: int(_env("STT_ENDPOINTING_MS", "100")))

    cartesia_model: str = field(default_factory=lambda: _env("CARTESIA_MODEL", "sonic-3.6"))
    cartesia_version: str = field(default_factory=lambda: _env("CARTESIA_VERSION", "2026-08-14"))
    cartesia_voice: str = field(
        default_factory=lambda: _env("CARTESIA_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b")
    )

    in_sample_rate: int = 16000
    out_sample_rate: int = 24000

    # Turn-taking knobs
    # Never start talking until the caller has been silent this long, so a comma pause ("Yes, | please
    # go ahead") isn't mistaken for a turn end. The LLM usually needs longer than this anyway.
    min_turn_silence_ms: int = 300
    max_incomplete_wait_ms: int = 1500  # detector said "not done": respond anyway after this much silence
    turn_detect_timeout_ms: int = 600  # detector slower than this => assume complete (it runs parallel to the LLM)
    backchannel_min_gap_s: float = 5.0
    backchannel_min_user_speech_s: float = 3.0
    slow_tool_filler_s: float = 2.5
    duck_gain: float = 0.6  # agent volume while the caller may be cutting in (a false alarm should barely show)

    # Browser agent (lines/browser.toml). Any OpenAI-compatible endpoint with an image-capable model works;
    # by default it's the Groq key and the same Qwen model, which accepts images.
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "qwen/qwen3.8-27b"))
    vision_base_url: str = field(default_factory=lambda: _env("VISION_BASE_URL"))
    vision_key: str = field(default_factory=lambda: _env("VISION_API_KEY"))
    browser_headless: bool = field(default_factory=lambda: _env("BROWSER_HEADLESS", "1") != "0")
    # A saved profile keeps your sign-ins between runs ("Show browser window" to log in once).
    browser_profile_dir: Path | None = field(
        default_factory=lambda: None if _env("BROWSER_PROFILE_DIR") == "none"
        else Path(_env("BROWSER_PROFILE_DIR") or ROOT / "data" / "browser" / "profile")
    )
    # Drive your own running Chrome instead: start it with --remote-debugging-port=9222, set http://localhost:9222
    browser_cdp_url: str = field(default_factory=lambda: _env("BROWSER_CDP_URL"))
    # Comma-separated sites the agent may use (subdomains included); empty = any site.
    browser_allowed_sites: tuple[str, ...] = field(
        default_factory=lambda: tuple(s.strip() for s in _env("BROWSER_ALLOWED_SITES").split(",") if s.strip())
    )
    # "auto": read the page structure, screenshot only when needed (faster, fewer tokens). "always": every step.
    browser_vision: str = field(default_factory=lambda: _env("BROWSER_VISION", "auto"))
    browser_start_url: str = field(
        default_factory=lambda: _env("BROWSER_START_URL", f"http://localhost:{_env('PORT', '8000')}/demo/job-application")
    )

    # Hosting: who may call, and how much of your API credit one visitor can spend.
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    access_code: str = field(default_factory=lambda: _env("ACCESS_CODE"))  # empty = anyone with the link
    max_calls: int = field(default_factory=lambda: int(_env("MAX_CALLS", "3")))  # concurrent calls
    max_call_s: int = field(default_factory=lambda: int(_env("MAX_CALL_SECONDS", "300")))
    # Phone calls (Twilio Media Streams). With an auth token set, webhook signatures are checked.
    twilio_auth_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    public_url: str = field(default_factory=lambda: _env("PUBLIC_URL").rstrip("/"))  # e.g. https://duplex.fly.dev

    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))
    latency_log: Path = ROOT / "data" / "latency.jsonl"

    def missing_keys(self) -> list[str]:
        return [
            name
            for name, val in [
                ("DEEPGRAM_API_KEY", self.deepgram_key),
                ("GROQ_API_KEY", self.groq_key),
                ("CARTESIA_API_KEY", self.cartesia_key),
            ]
            if not val
        ]


CONFIG = Config()
