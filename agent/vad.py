"""Tiny energy-based voice activity detector on the inbound 16 kHz PCM stream.

Deepgram's own endpointing returns a final transcript ~550-650 ms after the caller stops. We watch the
audio ourselves and send Deepgram `Finalize` as soon as a short pause starts, which brings the final in
~150 ms sooner, and gives us the moment to start speculating on the latest interim transcript.

Finalizing on every short pause is safe: it only closes the current transcript segment. Whether the caller
is *done* is still decided by the semantic turn detector.
"""

import array
import math


class EnergyVAD:
    def __init__(self, sample_rate: int = 16000, silence_ms: int = 100, min_speech_ms: int = 120):
        self.sr = sample_rate
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.floor_db = -60.0  # adaptive noise floor
        self.in_speech = False
        self.speech_ms = 0.0
        self.silence_run_ms = 0.0
        self.samples = 0
        self.last_speech_end = 0.0  # audio-time seconds of the last voiced frame's end

    def threshold(self) -> float:
        return max(self.floor_db + 15.0, -50.0)

    def update(self, pcm: bytes) -> float | None:
        """Feed one frame. Returns the audio-time where speech ended when a pause is detected, else None."""
        a = array.array("h", pcm)
        n = len(a)
        if n == 0:
            return None
        self.samples += n
        frame_ms = n * 1000 / self.sr
        energy = sum(x * x for x in a) / n
        db = 10 * math.log10(energy / (32768.0 ** 2)) if energy > 0 else -100.0

        voiced = db > self.threshold()
        if not voiced:
            # Track the noise floor quickly downward, slowly upward.
            rate = 0.2 if db < self.floor_db else 0.02
            self.floor_db = max(-90.0, self.floor_db + rate * (db - self.floor_db))

        if voiced:
            self.speech_ms += frame_ms
            self.silence_run_ms = 0.0
            self.last_speech_end = self.samples / self.sr
            if self.speech_ms >= self.min_speech_ms:
                self.in_speech = True
            return None

        self.silence_run_ms += frame_ms
        if self.in_speech and self.silence_run_ms >= self.silence_ms:
            self.in_speech = False
            self.speech_ms = 0.0
            return self.last_speech_end
        if not self.in_speech and self.silence_run_ms >= self.silence_ms:
            self.speech_ms = 0.0  # a blip too short to count as speech
        return None
