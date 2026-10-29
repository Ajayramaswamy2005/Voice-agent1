"""Phone calls over Twilio Media Streams.

Twilio dials our /twilio/voice webhook, we answer with TwiML that opens a WebSocket (/twilio/stream), and
audio flows as base64 μ-law 8 kHz in both directions. TwilioBridge makes that socket look like the browser
one, so CallSession runs unchanged:

    caller μ-law 8k --decode, ×2--> pcm16 16k --> CallSession --> pcm16 24k --÷3, encode--> μ-law 8k --> caller

Barge-in maps to Twilio's `clear` event (drop everything buffered). Playout is measured with `mark` events:
a mark sent just before a reply's first frame comes back when the caller starts hearing it.
"""

import array
import base64
import hashlib
import hmac
import json
import struct
import time
from xml.sax.saxutils import quoteattr

# -- G.711 μ-law ----------------------------------------------------------------------------------------
_BIAS, _CLIP = 0x84, 32635


def _ulaw_encode_sample(s: int) -> int:
    sign = 0x80 if s < 0 else 0
    s = min(abs(s), _CLIP) + _BIAS
    exp = max(0, s.bit_length() - 8)
    mant = (s >> (exp + 3)) & 0x0F
    return ~(sign | (exp << 4) | mant) & 0xFF


def _ulaw_decode_byte(b: int) -> int:
    b = ~b & 0xFF
    exp, mant = (b >> 4) & 0x07, b & 0x0F
    s = (((mant << 3) + _BIAS) << exp) - _BIAS
    return -s if b & 0x80 else s


_DECODE = [_ulaw_decode_byte(b) for b in range(256)]
_ENCODE = bytes(_ulaw_encode_sample(s - 65536 if s >= 32768 else s) for s in range(65536))  # index: s & 0xFFFF


def ulaw_to_pcm16k(data: bytes) -> bytes:
    """μ-law 8 kHz -> pcm16 16 kHz (linear interpolation between samples)."""
    src = [_DECODE[b] for b in data]
    out = array.array("h", bytes(4 * len(src)))
    for i, s in enumerate(src):
        nxt = src[i + 1] if i + 1 < len(src) else s
        out[2 * i] = s
        out[2 * i + 1] = (s + nxt) // 2
    return out.tobytes()


def pcm24k_to_ulaw(pcm: bytes) -> bytes:
    """pcm16 24 kHz -> μ-law 8 kHz (average each group of three: a crude low-pass before decimating)."""
    src = array.array("h")
    src.frombytes(pcm[: len(pcm) // 2 * 2])
    n = len(src) // 3
    return bytes(_ENCODE[((src[3 * i] + src[3 * i + 1] + src[3 * i + 2]) // 3) & 0xFFFF] for i in range(n))


# -- Twilio plumbing --------------------------------------------------------------------------------------
def twiml_stream(ws_url: str, line: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
        f"<Stream url={quoteattr(ws_url)}><Parameter name=\"line\" value={quoteattr(line)}/></Stream>"
        "</Connect></Response>"
    )


def valid_signature(auth_token: str, url: str, params: dict[str, str], signature: str) -> bool:
    """Twilio's X-Twilio-Signature: base64 HMAC-SHA1 of the URL followed by the sorted POST params."""
    payload = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature or "")


class TwilioBridge:
    """Duck-types the browser WebSocket for CallSession: send_bytes / send_text."""

    def __init__(self, ws, stream_sid: str):
        self.ws = ws
        self.sid = stream_sid
        self.drop_below = 0
        self._marked: set[int] = set()
        self._mark_sent: dict[str, float] = {}
        self._carry = b""  # odd tail of 24 kHz audio that didn't fill a group of three samples

    async def send_bytes(self, frame: bytes) -> None:
        seq = struct.unpack("<I", frame[:4])[0]
        if seq != 0 and seq < self.drop_below:
            return  # audio from a reply the caller already interrupted
        pcm = self._carry + frame[4:]
        usable = len(pcm) // 6 * 6
        self._carry = pcm[usable:]
        if seq != 0 and seq not in self._marked:
            self._marked.add(seq)
            self._mark_sent[str(seq)] = time.perf_counter()
            await self._send({"event": "mark", "streamSid": self.sid, "mark": {"name": str(seq)}})
        ulaw = pcm24k_to_ulaw(pcm[:usable])
        for i in range(0, len(ulaw), 1600):  # 200 ms per message: pre-rendered clips arrive as one long buffer
            payload = base64.b64encode(ulaw[i:i + 1600]).decode()
            await self._send({"event": "media", "streamSid": self.sid, "media": {"payload": payload}})

    async def send_text(self, text: str) -> None:
        msg = json.loads(text)
        if msg.get("type") == "clear":
            self.drop_below = int(msg["drop_below"])
            self._carry = b""
            await self._send({"event": "clear", "streamSid": self.sid})
        # Everything else (transcripts, latency, duck) is for the browser UI; a phone has no screen.

    def on_mark(self, name: str) -> dict | None:
        """A mark came back: the caller is now hearing reply `name`. Returns a playout report for the session."""
        sent = self._mark_sent.pop(name, None)
        if sent is None:
            return None
        return {"type": "playout", "seq": int(name), "delay_ms": (time.perf_counter() - sent) * 1000}

    async def _send(self, msg: dict) -> None:
        await self.ws.send_text(json.dumps(msg))
