"""Groq streaming chat completions with tool calls, plus the semantic end-of-turn classifier."""

import asyncio
import logging
import re
from dataclasses import dataclass

from groq import APIConnectionError, AsyncGroq, RateLimitError

from .config import Config

log = logging.getLogger("llm")


def reasoning_off(model: str) -> dict:
    """Reasoning models must not think before speaking: thinking tokens are pure latency on a call."""
    if "gpt-oss" in model:
        return {"reasoning_effort": "low", "include_reasoning": False}
    if "qwen3" in model:
        return {"reasoning_effort": "none"}
    return {}


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallStarted:
    name: str


@dataclass
class ModelUsed:
    model: str  # which model is actually serving this stream (primary or a fallback)


@dataclass
class ToolCalls:
    calls: list[dict]  # [{"id", "name", "arguments"(json str)}]


class GroqLLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # No SDK retries: on a phone call a 5 s backoff is worse than switching models.
        self.client = AsyncGroq(api_key=cfg.groq_key, max_retries=0)

    async def _open(self, messages: list[dict], tools: list[dict], model: str | None = None):
        if model:
            models = [model]
        else:
            models = [self.cfg.llm_model] + ([self.cfg.llm_fallback_model] if self.cfg.llm_fallback_model else [])
        # A dropped connection fails fast, so retrying the same model once is cheap; a rate limit means
        # that model's quota is spent, so go straight to the next one.
        for mi, m in enumerate(models):
            last_model = mi == len(models) - 1
            for attempt in range(2):
                try:
                    return m, await self._create(m, messages, tools)
                except RateLimitError:
                    if last_model:
                        raise
                    log.warning("%s rate-limited; falling back to %s", m, models[mi + 1])
                    break
                except APIConnectionError:
                    if attempt == 1 and last_model:
                        raise
                    log.warning("%s connection error (attempt %d)", m, attempt + 1)

    def _create(self, model: str, messages: list[dict], tools: list[dict]):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
            temperature=0.4,
            max_tokens=150,  # Groq reserves max_tokens against the per-minute quota
            stream=True,
            **reasoning_off(model),
        )

    async def stream(self, messages: list[dict], tools: list[dict], model: str | None = None):
        """Yields TextDelta / ToolCallStarted while streaming, then ToolCalls if the model wants tools."""
        used, stream = await self._open(messages, tools, model)
        yield ModelUsed(used)
        calls: dict[int, dict] = {}
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield TextDelta(delta.content)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        if not slot["name"]:
                            yield ToolCallStarted(tc.function.name)
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        finally:
            await stream.close()
        if calls:
            yield ToolCalls([calls[i] for i in sorted(calls)])


_TRAILING_INCOMPLETE = re.compile(
    r"(\b(and|but|or|so|because|the|a|an|to|of|for|with|my|is|um|uh|like|if|aur|ya|ki|ke|ka|toh|to|"
    r"matlab|lekin|par|kyunki|mera|meri|mujhe)|और|या|कि|के|का|की|तो|लेकिन|पर|क्योंकि|मेरा|मेरी|मुझे|,|\.\.\.)\s*$",
    re.IGNORECASE,
)

TURN_PROMPT = """\
You decide whether a caller on a phone line has finished their turn and expects a reply.
The transcript may be English, Hindi (Devanagari) or Hinglish, and may lack punctuation.
Answer with exactly one letter: Y if the caller's turn is complete (a full question, statement, answer, \
or a short reply like "yes"/"haan"/a number), N if they are mid-thought (trailing off, listing digits \
that look unfinished, a dangling clause, a filler like "um").

Agent's last line: {agent}
Caller so far: {user}
Complete?"""


class TurnDetector:
    """Semantic end-of-turn detection: 'is the user actually done?' rather than a fixed silence timeout.

    Fast path: an obvious dangling ending ("and", "ki", "...") => not done, no model call.
    Otherwise a fast model on Groq (reasoning off) answers Y/N with one token.
    """

    def __init__(self, cfg: Config, client: AsyncGroq):
        self.cfg = cfg
        self.client = client

    async def is_complete(self, user_text: str, agent_last: str) -> tuple[bool, str]:
        text = user_text.strip()
        if not text:
            return False, "empty"
        if _TRAILING_INCOMPLETE.search(text):
            return False, "heuristic"
        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.cfg.turn_model,
                    messages=[{"role": "user", "content": TURN_PROMPT.format(
                        agent=agent_last[-200:] or "(none)", user=text[-400:])}],
                    temperature=0,
                    max_tokens=2,
                    **reasoning_off(self.cfg.turn_model),
                ),
                timeout=self.cfg.turn_detect_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            return True, "timeout"
        except Exception as e:
            log.warning("turn detector failed: %s", e)
            return True, "error"
        answer = (resp.choices[0].message.content or "").strip().upper()
        return not answer.startswith("N"), "model"
