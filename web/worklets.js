// Audio worklets: mic capture (-> 16 kHz PCM16 frames) and a clearable playback buffer.

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0; // fractional read position into incoming samples
    this.carry = new Float32Array(0);
    this.frame = new Int16Array(320); // 20 ms at 16 kHz
    this.fill = 0;
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    const buf = new Float32Array(this.carry.length + ch.length);
    buf.set(this.carry);
    buf.set(ch, this.carry.length);
    while (this.pos + 1 < buf.length) {
      const i = Math.floor(this.pos), f = this.pos - i;
      const s = buf[i] * (1 - f) + buf[i + 1] * f;
      this.frame[this.fill++] = Math.max(-32768, Math.min(32767, s * 32768));
      if (this.fill === this.frame.length) {
        this.port.postMessage(this.frame.buffer.slice(0));
        this.fill = 0;
      }
      this.pos += this.ratio;
    }
    const keep = Math.floor(this.pos);
    this.carry = buf.slice(keep);
    this.pos -= keep;
    return true;
  }
}

class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.main = []; // [{seq, data: Float32Array, off}]
    this.side = []; // backchannels: mixed in, never cleared
    this.gain = 1;
    this.target = 1;
    this.startedSeq = -1;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "audio") (m.side ? this.side : this.main).push({ seq: m.seq, data: m.data, off: 0 });
      else if (m.type === "clear") this.main = [];
      else if (m.type === "gain") this.target = m.value;
    };
  }
  pull(queue, out, i) {
    while (queue.length) {
      const head = queue[0];
      if (head.off < head.data.length) return head;
      queue.shift();
    }
    return null;
  }
  process(_, outputs) {
    const out = outputs[0][0];
    let wasPlaying = this.main.length > 0;
    for (let i = 0; i < out.length; i++) {
      this.gain += (this.target - this.gain) * 0.002; // ~10 ms smoothing for duck/unduck
      let s = 0;
      const m = this.pull(this.main);
      if (m) {
        if (m.seq !== this.startedSeq) {
          this.startedSeq = m.seq;
          this.port.postMessage({ type: "started", seq: m.seq });
        }
        s += m.data[m.off++] * this.gain;
      }
      const b = this.pull(this.side);
      if (b) s += b.data[b.off++] * 0.8;
      out[i] = s;
    }
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out);
    if (wasPlaying && !this.pull(this.main)) this.port.postMessage({ type: "idle" });
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
registerProcessor("player-processor", PlayerProcessor);
