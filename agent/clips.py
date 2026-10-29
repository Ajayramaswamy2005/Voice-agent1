"""Pre-rendered audio clips (backchannels, tool fillers, greetings).

These are latency-critical: "mm-hm" or "let me check that" must start instantly, so they're synthesized
once at startup (and cached on disk) instead of on demand.
"""

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass

from .config import ROOT, Config
from .personas import BACKCHANNELS, PERSONAS
from .tts import CartesiaTTS

log = logging.getLogger("clips")
CACHE_DIR = ROOT / "data" / "clips"


@dataclass
class Clip:
    text: str
    pcm: bytes


class ClipBank:
    def __init__(self):
        self._clips: dict[tuple[str, str, str], list[Clip]] = {}  # (persona|"*", kind, lang) -> clips

    def pick(self, persona: str, kind: str, lang: str) -> Clip | None:
        options = self._clips.get((persona, kind, lang)) or self._clips.get((persona, kind, "en"))
        return random.choice(options) if options else None

    async def build(self, cfg: Config, tts: CartesiaTTS) -> None:
        jobs: list[tuple[tuple[str, str, str], str]] = []
        for lang, texts in BACKCHANNELS.items():
            jobs += [((p, "backchannel", lang), t) for p in PERSONAS for t in texts]
        for p in PERSONAS.values():
            for lang, text in p.greeting.items():
                jobs.append(((p.key, "greeting", lang), text))
            for lang, texts in p.fillers.items():
                jobs += [((p.key, "filler", lang), t) for t in texts]
            for lang, texts in p.slow_fillers.items():
                jobs += [((p.key, "slow_filler", lang), t) for t in texts]

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(1)  # Cartesia's free tier allows 2 concurrent generations; leave one spare

        async def render(key: tuple[str, str, str], text: str) -> None:
            lang = key[2]
            h = hashlib.sha1(f"{cfg.cartesia_model}|{cfg.cartesia_voice}|{lang}|{cfg.out_sample_rate}|{text}"
                             .encode()).hexdigest()[:16]
            path = CACHE_DIR / f"{h}.pcm"
            if path.exists():
                pcm = path.read_bytes()
            else:
                pcm = b""
                for attempt in range(5):  # retry rate-limit (429) rejections with backoff
                    async with sem:
                        pcm = await tts.synthesize(text, lang)
                    if pcm:
                        break
                    await asyncio.sleep(0.5 * 2 ** attempt)
                if pcm:
                    path.write_bytes(pcm)
            if pcm:
                self._clips.setdefault(key, []).append(Clip(text, pcm))

        await asyncio.gather(*(render(k, t) for k, t in jobs))
        log.info("clip bank ready: %d/%d clips", sum(len(v) for v in self._clips.values()), len(jobs))
