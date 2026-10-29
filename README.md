# Duplex: a full-duplex voice agent

A phone-style voice agent that handles conversation the way people do, instead of taking strict walkie-talkie turns.
It runs three jobs ("lines"), each with its tools exposed through **MCP**:

| Line | MCP server | Tools |
|---|---|---|
| **Sunrise Clinic**: appointment booking | `mcp_servers/clinic.py` | `list_doctors`, `check_availability`, `book_appointment`, `find_appointments`, `reschedule_appointment`, `cancel_appointment` |
| **NovaMart**: Hindi/English customer support | `mcp_servers/support.py` | `lookup_orders`, `track_order`, `request_refund`, `create_ticket`, `ticket_status`, `transfer_to_human` |
| **Kiran**: operates a web browser by looking at the screen | `mcp_servers/browser.py` | `start_browser_task`, `confirm_browser_action`, `stop_browser_task`, `browser_status` |

Lines are plain config files in `lines/`, so you can [add your own](#add-your-own-line) with any MCP server. People can use it from a [hosted link](#host-it-shareable-link) or [a phone number](#phone-calls-twilio).

Both lines understand English, Hindi and Hinglish (Deepgram `nova-3` with `language=multi`). The agent replies in the caller's language and switches mid-call.

## What makes it full-duplex

| Behaviour | How |
|---|---|
| **Barge-in** | Caller speech while the agent talks *ducks* playback instantly (Deepgram VAD event). Once the words are confirmed to be more than an acknowledgement (`okay`, `haan`, `mm-hm`), playback is flushed client-side, the LLM stream and TTS context are cancelled, and history records **only the words the caller actually heard** (from Cartesia word timestamps), ending in `—`, so the LLM knows it was cut off. |
| **Fast endpointing + speculation** | A server-side VAD hears a pause 100 ms in and sends Deepgram `Finalize`, so the final transcript lands ~200 ms after the caller stops. If the latest interim transcript already covers the speech, the LLM starts on it immediately; the final transcript then confirms it (same words) or replaces it. Unconfirmed text never reaches TTS and never runs tools. |
| **Semantic turn detection** | In parallel a fast model (one output token, reasoning off) answers "is the caller actually done?". Audio stays gated until it says yes, and never starts before 300 ms of silence, so a comma pause isn't a turn end. If the caller resumes (VAD onset ~120 ms, or new words), the unreleased answer is discarded silently. A regex fast-path catches dangling endings (`and…`, `ki…`, `क्योंकि…`) without a model call. A caller who goes quiet mid-thought still gets an answer after 1.5 s. |
| **Backchanneling** | Pre-rendered `Mm-hm` / `Right` / `हाँ` / `अच्छा` play on a side channel when the detector says the caller paused mid-thought, or at phrase boundaries in long monologues, rate-limited to one every 5 s. |
| **Tool calls during speech** | The moment the LLM *starts* emitting a tool call, a pre-rendered filler ("Let me check that", "एक सेकंड, मैं चेक करती हूँ") plays while the MCP tool runs. A second filler plays if the tool takes longer than 2.5 s. Side-effecting tools never run on an unconfirmed (speculative) turn. |
| **Latency budget** | Streaming everywhere: Deepgram live WS (nearest region) → Groq streaming → phrase chunker (first clause released early) → Cartesia WS with continuations and `max_buffer_delay_ms: 0`. Prompts are kept small (compacted MCP tool schemas, trimmed history) because prompt length is time-to-first-token. Every turn is timed along the critical path and charted at `/latency`. |

```
mic ─pcm16 16k─▶ browser ─WS─▶ Deepgram STT ──▶ CallSession ──▶ Groq LLM ─phrases─▶ Cartesia TTS
                                                    │  ▲                │ tool calls
                                    TurnDetector ◀──┘  │                ▼
                                    (8B, 1 token)      │            MCP client ─stdio─▶ clinic / support servers
speaker ◀─pcm 24k── browser ◀───────────────── gated audio + clear/duck control
```

## Run it

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), Chrome or Edge, and three API keys (each has free credits):
[Deepgram](https://console.deepgram.com), [Groq](https://console.groq.com), [Cartesia](https://play.cartesia.ai).

```powershell
uv sync
uv run playwright install chromium   # for the browser line
copy .env.example .env        # fill in DEEPGRAM_API_KEY, GROQ_API_KEY, CARTESIA_API_KEY (+ DEEPGRAM_HOST, see below)
uv run python -m agent.main   # http://localhost:8000 (landing)  ·  /app (call UI)  ·  /latency (dashboard)
```

The first start pre-renders about 30 short clips (greetings, fillers, backchannels) and caches them in `data/clips/`.
Databases are re-seeded on every start, so the demo data is always fresh relative to today's date.

Things to try:
- *Clinic*: "I need a skin doctor tomorrow morning", then interrupt while it lists slots: "wait, Friday instead".
- *Clinic*: "I want to book for my… … son", pausing mid-sentence. It should backchannel, not answer.
- *Support (Hindi)*: "मेरा ऑर्डर OD1005 अभी तक नहीं आया", then "can you make it a high priority ticket?"
- *Browser*: "Fill this form with my resume details", then answer "yes" when it asks before submitting.
- Demo identities: phone `9876543210` (Priya, orders OD1001/OD1002; also clinic appointment A1001), `9123456780`, `9988776655`.

Use headphones if your browser's echo cancellation is weak. Otherwise the agent can hear itself and barge in on itself.

## The browser line: an assistant that does work for you

Pick **Kiran** and say (or type in the panel) what you want done: *"Fill this form with my resume details"*, *"Find five Python backend jobs in Bengaluru posted this week and list them with links"*, *"Compare the 3 cheapest 1 TB SSDs on Amazon"*. It works a real Chromium browser and the right-hand panel shows what it sees, its plan, each step, and the result.

| Part | How |
|---|---|
| **Your sign-ins** | The browser keeps a saved profile (`data/browser/profile`, never committed). Press **Show browser window**, sign in to Gmail, LinkedIn, GitHub and so on yourself, then hide it. The agent reuses those sessions and never sees or types your passwords. To drive your own Chrome instead, start it with `--remote-debugging-port=9222` and set `BROWSER_CDP_URL=http://localhost:9222`. |
| **Reads and reports** | Besides clicking and typing it can `read` whole pages, `search` the web, open/switch/close tabs, and keep `note`s with sources. A task ends with a short spoken summary plus a full written **Result** in the panel (with a Copy button). |
| **Plans** | Every task gets a plan. Short ones start right away. Longer ones, or any task with *Show me the plan before it starts* ticked (`plan_first` by voice), wait for you: start it, edit the steps in the panel, or say *"skip step 2"*. Steps are ticked off as it works. |
| **Fast by default** | *Auto* vision mode reads the page structure (numbered element list + page text) on every step and sends a screenshot only when needed: a new page, after a failure, every 4 steps, when the model asks to `look`, and always before it may say *done*. That is several times fewer image tokens than one screenshot per step. `BROWSER_VISION=always` restores a screenshot every step. |
| **Type or talk** | Type tasks in the panel and keep working. They queue if one is running. Updates appear in the conversation, and as desktop notifications when the tab is in the background. During a call you can also type messages to the agent. |
| **GUI grounding** | *Set-of-marks*: every visible interactive element gets a numbered box drawn on the page, and the model answers with mark numbers. For targets without a mark it predicts **x/y pixel coordinates**, and the element under that point goes through the same checks. |
| **Action verification** | After each action the DOM is checked (value kept, option selected, checkbox flipped, page reacted), and the model confirms `previous_ok` on what it sees next. Failures are retried; after 3 it stops and says why. |
| **Safety** | Submit, send, post, reply, share, pay, book, delete, follow/connect, upload and similar clicks, Enter in a form, and sensitive fields (passwords, OTP, card, UPI, Aadhaar/PAN, bank) **pause for your yes/no**, by voice or the Approve button. `BROWSER_ALLOWED_SITES=linkedin.com,naukri.com` makes everything else off-limits: navigation is blocked, and clicks that lead elsewhere are undone. Every task, plan, action, approval and result is written to `data/browser/audit.jsonl` and shown under **Activity log**. |

Upload your own resume (PDF, .txt or .md) as "your details". Until you do, a sample resume and a demo job form (`/demo/job-application`) are used.

**Model and quota.** The vision model defaults to Groq `qwen/qwen3.8-27b` (it accepts images). `VISION_API_KEY` can hold a separate key, but Groq limits are per organization, so only a key from a *different* Groq account adds quota. On the free tier (8k tokens/min, 200k/day) a long task can take minutes and a few tasks use up the day. Any OpenAI-compatible vision endpoint works via `VISION_BASE_URL` / `VISION_API_KEY` / `VISION_MODEL`.

## Add your own line

Each file in `lines/` is one voice agent. Copy `lines/clinic.toml`, change the prompt, greeting and fillers, and point it at any MCP server:

```toml
name = "Pizza Palace: orders"
order = 4
mcp_command = ["npx", "-y", "@acme/pizza-mcp"]   # any stdio MCP server
prompt = "You are Rhea, taking pizza orders for delivery in Indiranagar..."

[env]                       # external servers don't inherit your environment, so your API keys stay private
PIZZA_API_KEY = "$PIZZA_API_KEY"

[greeting]
en = "Pizza Palace, this is Rhea. What can I get you?"

[fillers]
en = ["Let me check the menu."]
```

Restart, and it appears in the UI with its tools listed. Fillers and the greeting are pre-rendered at startup. `ENABLED_LINES=clinic,pizza` limits which lines are served.

## Host it (shareable link)

The `Dockerfile` serves the clinic and support lines on `0.0.0.0:8000`. Any host with WebSockets and HTTPS works (browsers only allow the microphone on HTTPS). Pick a region near your callers and set `DEEPGRAM_HOST` to match, since every turn pays those round trips.

```powershell
fly launch --no-deploy            # or Render / Railway: point them at the Dockerfile
fly secrets set DEEPGRAM_API_KEY=... GROQ_API_KEY=... CARTESIA_API_KEY=... DEEPGRAM_HOST=api.in.deepgram.com ACCESS_CODE=pick-one
fly deploy
```

Every visitor spends **your** API credit, so the app has limits: `MAX_CALLS` concurrent calls (default 3), `MAX_CALL_SECONDS` per call (default 300), and `ACCESS_CODE` to make the link invite-only (the UI asks for it). The Groq free tier is about one call per minute; use the Dev tier for a public link.

The browser line is off in the image on purpose: it would let anyone with the link drive a browser on your server. If you enable it (`ENABLED_LINES=clinic,support,browser`), set `ACCESS_CODE`. It is one shared browser, meant for one user at a time.

## Phone calls (Twilio)

The same agent answers a real phone number. `agent/phone.py` bridges Twilio Media Streams (μ-law 8 kHz) to the pipeline, maps barge-in to Twilio's `clear`, and measures playout with `mark` events.

1. Deploy (above) and set `PUBLIC_URL=https://your-app.fly.dev` and `TWILIO_AUTH_TOKEN` (enables webhook signature checks).
2. In the Twilio console, set your number's *A call comes in* webhook to `POST https://your-app.fly.dev/twilio/voice?line=clinic` (or `line=support`).
3. Call the number. For Indian numbers, Exotel and Plivo have similar WebSocket streaming. Only the bridge would change.

Phones have their own echo cancellation, so headphones don't matter, but there is no screen: the browser line needs the web UI.

## Measure latency and publish the chart

```powershell
uv run python -m scripts.bench                  # scripted EN + HI calls on both lines, prints p50/p90 per stage
uv run python -m scripts.export_report          # -> data/latency_report.html (standalone, shareable)
```

`scripts/bench.py` synthesizes the caller's lines and streams them in real time, including mid-thought pauses, so the numbers cover the whole pipeline. It reports playout as 0 ms because nothing plays to a speaker; live calls measure real playout in the browser.

Stages (they add up exactly to end-to-end):

| Stage | From → to |
|---|---|
| `stt` | caller's last word *ends in the audio* → final transcript (includes the endpointing wait) |
| `llm_ttft` | transcript → first LLM token |
| `chunking` | first token → first speakable phrase sent to TTS |
| `tts_ttfb` | phrase → first audio byte |
| `turn_gate` | audio ready → released (only when the turn detector is slower than the LLM+TTS path) |
| `playout` | first frame received in browser → audible (worklet + `outputLatency`) |

What moved the numbers, in order of impact:

1. **Deepgram region.** From India the global endpoint is ~270 ms per round trip; `DEEPGRAM_HOST=api.in.deepgram.com` is ~45 ms. STT went from ~440 ms to ~160–200 ms.
2. **VAD + `Finalize`** instead of waiting for Deepgram's endpointing: about −150 ms.
3. **Smaller prompts**: 1,263 → 992 tokens took median TTFT from ~345 to ~255 ms.
4. **Model choice**: `qwen/qwen3.8-27b` with reasoning off (~130–400 ms TTFT). `openai/gpt-oss-*` call tools reliably but their hidden reasoning costs 550–1,500 ms, so they're only the fallback.

Measure your own round trips (`curl -w "%{time_connect}"` to each API) and pick regions accordingly.

**Groq free tier.** 8,000 tokens/minute per model is roughly one call's worth of turns per minute; once it's spent the agent falls back to the slower model (watch for `rate-limited; falling back` in the log). The Dev tier removes this.

Other knobs: `STT_ENDPOINTING_MS`, `LLM_MODEL` / `LLM_FALLBACK_MODEL`, and `min_turn_silence_ms` / `turn_detect_timeout_ms` / `max_incomplete_wait_ms` in `agent/config.py`.

## Use the MCP servers elsewhere

They're ordinary stdio MCP servers, so you can add them to Claude Code or any other MCP client:

```powershell
claude mcp add clinic -- uv --directory C:\Users\ajayr\voice-agent run python -m mcp_servers.clinic
```

## Layout

```
agent/
  main.py        Starlette app, startup (MCP servers, clip bank), /ws call socket, limits, browser + Twilio routes
  session.py     the turn-taking state machine: speculative turns, barge-in, backchannels, fillers
  stt.py         Deepgram live WebSocket client
  tts.py         Cartesia WebSocket client (contexts, continuations, word timestamps)
  llm.py         Groq streaming + tool-call assembly; TurnDetector
  mcp_hub.py     MCP stdio client → OpenAI-style tool specs
  clips.py       pre-rendered fillers/backchannels/greetings
  vad.py         energy VAD on inbound audio: pause → Deepgram Finalize + speculation; onset → cancel
  latency.py     per-turn critical-path timing, JSONL log, p50/p90
  personas.py    loads lines/*.toml; voice rules, backchannels
  phone.py       Twilio Media Streams bridge (μ-law codec, clear, marks)
  text.py        language detection, ack detection, phrase chunker
lines/           one .toml per line: prompt, greeting, fillers, MCP server
mcp_servers/     clinic.py, support.py (MCP SDK v2 `MCPServer`, SQLite); browser.py + browser_agent.py (Playwright, set-of-marks)
web/             index.html + app.js (call UI, browser panel), worklets.js (mic + clearable player), latency.html, demo/ (sample form)
scripts/         bench.py, export_report.py
tests/           unit tests + state-machine tests with fake STT/LLM/TTS
```

## Tests

```powershell
uv run pytest
```
