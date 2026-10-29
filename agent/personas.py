import os
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

LINES_DIR = Path(__file__).resolve().parent.parent / "lines"

VOICE_RULES = """\
Live phone call; your words become speech.
- 1-2 short sentences. No lists, markdown or symbols. Say numbers as spoken ("ten thirty").
- Reply in the language of the caller's last message: English if they spoke English; Hindi in \
Devanagari only if they spoke Hindi or Hinglish.
- Facts about appointments, orders, tickets come only from tools. Call the tool immediately, without \
announcing it (a "let me check" sound plays for you).
- Confirm details once before booking, cancelling or refunding.
- A reply of yours ending in "—" was cut off by the caller; don't repeat it, answer them.
"""


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    mcp_server: str  # module under mcp_servers/
    prompt: str
    greeting: dict[str, str]
    fillers: dict[str, list[str]]  # spoken while a tool runs
    slow_fillers: dict[str, list[str]]  # spoken if the tool is still running after a while
    mcp_command: tuple[str, ...] = ()  # stdio command for an external MCP server; empty = mcp_servers.<mcp_server>
    event_tool: str = ""  # long-poll tool whose results the agent speaks unprompted (hidden from the LLM)
    app_tools: tuple[str, ...] = ()  # tools only the app calls (UI endpoints); hidden from the LLM
    mcp_env: tuple[tuple[str, str], ...] = ()  # extra environment for the server; values like "$NAME" read os.environ

    def hidden_tools(self) -> set[str]:
        return set(self.app_tools) | ({self.event_tool} if self.event_tool else set())

    def system_prompt(self) -> str:
        today = date.today()
        return f"{self.prompt}\n\nToday is {today:%A, %d %B %Y} ({today.isoformat()}).\n\n{VOICE_RULES}"


BACKCHANNELS = {
    "en": ["Mm-hm.", "Right.", "Okay."],
    "hi": ["हाँ।", "अच्छा।", "जी।"],
}

# Short acknowledgements from the *caller* while the agent talks: not a barge-in.
ACK_WORDS = {
    "mm", "mhm", "mmhm", "mmhmm", "uhhuh", "hmm", "uh-huh", "uh", "huh", "yeah", "yes", "yep", "ok", "okay", "right", "sure",
    "haan", "ha", "han", "ji", "accha", "acha", "achha", "theek", "thik",
    "हाँ", "हां", "हा", "जी", "अच्छा", "ठीक", "हम्म", "हम",
}

def load_lines(directory: Path = LINES_DIR, enabled: str = "") -> dict[str, Persona]:
    """One Persona per lines/*.toml. `enabled` (comma-separated keys, e.g. from ENABLED_LINES) limits which
    load; empty means all."""
    keep = {k.strip() for k in enabled.split(",") if k.strip()}
    out: dict[str, Persona] = {}
    found = [(p.stem, tomllib.loads(p.read_text(encoding="utf-8"))) for p in directory.glob("*.toml")]
    for key, d in sorted(found, key=lambda kd: (kd[1].get("order", 100), kd[0])):  # first = default line
        if keep and key not in keep:
            continue
        out[key] = Persona(
            key=key,
            name=d.get("name", key),
            mcp_server=d.get("mcp_server", key),
            prompt=d["prompt"],
            greeting=d.get("greeting", {}),
            fillers=d.get("fillers", {}),
            slow_fillers=d.get("slow_fillers", {}),
            mcp_command=tuple(d.get("mcp_command", ())),
            event_tool=d.get("event_tool", ""),
            app_tools=tuple(d.get("app_tools", ())),
            mcp_env=tuple((k, os.environ.get(v[1:], "") if str(v).startswith("$") else str(v))
                          for k, v in d.get("env", {}).items()),
        )
    return out


PERSONAS = load_lines(enabled=os.environ.get("ENABLED_LINES", ""))
CLINIC = load_lines()["clinic"]  # used by tests whatever ENABLED_LINES says
