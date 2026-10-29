"""Entry point: `uv run python -m agent.main`, then open http://localhost:8000"""

import asyncio
import base64
import json
import logging
import sys
from urllib.parse import parse_qsl

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from .clips import ClipBank
from .config import CONFIG, ROOT
from .latency import LatencyLog, summarize
from .llm import GroqLLM, TurnDetector
from .mcp_hub import MCPHub
from .personas import PERSONAS
from .phone import TwilioBridge, twiml_stream, ulaw_to_pcm16k, valid_signature
from .session import CallSession
from .tts import CartesiaTTS

log = logging.getLogger("main")
WEB = ROOT / "web"
RESUME_DIR = ROOT / "data" / "resume"
MAX_RESUME_BYTES = 5_000_000


class App:
    def __init__(self):
        self.cfg = CONFIG
        self.hub = MCPHub()
        self.llm = GroqLLM(self.cfg)
        self.turn = TurnDetector(self.cfg, self.llm.client)
        self.tts = CartesiaTTS(self.cfg)
        self.clips = ClipBank()
        self.latency = LatencyLog(self.cfg.latency_log)
        self.active_calls = 0

    async def start(self) -> None:
        await self.hub.start({p.mcp_server: p.mcp_command for p in PERSONAS.values()},
                             env={p.mcp_server: dict(p.mcp_env) for p in PERSONAS.values()})
        for attempt in range(4):  # TLS to Cartesia occasionally fails transiently
            try:
                log.info("connecting to Cartesia…")
                await asyncio.wait_for(self.tts.connect(), 15)
                break
            except (OSError, TimeoutError, asyncio.TimeoutError) as e:
                if attempt == 3:
                    raise
                log.warning("Cartesia connect failed (%r); retrying", e)
                await asyncio.sleep(1 + attempt)
        log.info("rendering clips…")
        await asyncio.wait_for(self.clips.build(self.cfg, self.tts), 120)

    def authorized(self, code: str | None) -> bool:
        return not self.cfg.access_code or code == self.cfg.access_code

    async def _run_call(self, ws, persona, lang: str, receive) -> None:
        """Shared by browser and phone calls: limits, session lifecycle. `receive` feeds the session."""
        if self.tts.ws is None or self.tts.ws.close_code is not None:
            await self.tts.connect()  # reconnect if Cartesia dropped an idle socket
        session = CallSession(ws, self.cfg, persona, self.hub, self.llm, self.turn, self.tts, self.clips,
                              self.latency, lang=lang)
        self.active_calls += 1
        await session.start()
        log.info("call started: persona=%s lang=%s (%d active)", persona.key, lang, self.active_calls)
        try:
            await asyncio.wait_for(receive(session), self.cfg.max_call_s)
        except (asyncio.TimeoutError, TimeoutError):
            log.info("call hit the %ds limit", self.cfg.max_call_s)
            await session.send_json({"type": "ended", "reason": f"Calls end after {self.cfg.max_call_s // 60} minutes."})
        except WebSocketDisconnect:
            pass
        finally:
            self.active_calls -= 1
            await session.close()
            log.info("call ended")

    async def ws_call(self, ws: WebSocket) -> None:
        await ws.accept()
        if not self.authorized(ws.query_params.get("code")):
            await ws.close(4401, "Wrong access code")
            return
        if self.active_calls >= self.cfg.max_calls:
            await ws.close(4429, "All lines are busy. Try again in a few minutes.")
            return
        default = next(iter(PERSONAS))
        persona = PERSONAS.get(ws.query_params.get("persona", default), PERSONAS[default])
        lang = "hi" if ws.query_params.get("lang") == "hi" else "en"

        async def receive(session: CallSession) -> None:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if msg.get("bytes"):
                    await session.on_audio(msg["bytes"])
                elif msg.get("text"):
                    await session.on_client_message(json.loads(msg["text"]))

        await self._run_call(ws, persona, lang, receive)
        try:
            await ws.close()
        except RuntimeError:
            pass  # already closed by the client

    async def twilio_stream(self, ws: WebSocket) -> None:
        """Twilio Media Streams: `connected`, `start` (with our custom parameters), then `media` frames."""
        await ws.accept()
        start = None
        while start is None:
            msg = json.loads(await ws.receive_text())
            if msg.get("event") == "start":
                start = msg["start"]
            elif msg.get("event") == "stop":
                return
        line = start.get("customParameters", {}).get("line", "")
        persona = PERSONAS.get(line) or PERSONAS[next(iter(PERSONAS))]
        if self.active_calls >= self.cfg.max_calls:
            await ws.close()
            return
        bridge = TwilioBridge(ws, start["streamSid"])

        async def receive(session: CallSession) -> None:
            while True:
                msg = json.loads(await ws.receive_text())
                ev = msg.get("event")
                if ev == "media" and msg["media"].get("track", "inbound") == "inbound":
                    await session.on_audio(ulaw_to_pcm16k(base64.b64decode(msg["media"]["payload"])))
                elif ev == "mark":
                    report = bridge.on_mark(msg["mark"]["name"])
                    if report:
                        await session.on_client_message(report)
                elif ev == "stop":
                    return

        await self._run_call(bridge, persona, "en", receive)


def build_app(app: App) -> Starlette:
    async def landing(_):
        return FileResponse(WEB / "landing.html")

    async def index(_):
        return FileResponse(WEB / "index.html")

    async def dashboard(_):
        return FileResponse(WEB / "latency.html")

    async def demo_form(_):
        return FileResponse(WEB / "demo" / "job-application.html")

    async def latency_api(_):
        recs = app.latency.read()
        return JSONResponse({"records": recs, "summary": summarize(recs)})

    async def personas_api(_):
        return JSONResponse([{"key": p.key, "name": p.name, "browser": p.mcp_server == "browser",
                              "tools": [t["function"]["name"] for t in app.hub.tools.get(p.mcp_server, [])
                                        if t["function"]["name"] not in p.hidden_tools()]}
                             for p in PERSONAS.values()])

    async def config_api(_):
        return JSONResponse({"access_code_required": bool(app.cfg.access_code),
                             "max_call_seconds": app.cfg.max_call_s})

    # -- browser agent (only when a line uses the browser MCP server) --------------------------------
    def browser_guard(request: Request) -> Response | None:
        if "browser" not in app.hub.sessions:
            return JSONResponse({"error": "The browser line is not enabled."}, status_code=404)
        if not app.authorized(request.headers.get("x-access-code") or request.query_params.get("code")):
            return JSONResponse({"error": "Wrong access code."}, status_code=401)
        return None

    async def browser_api(request: Request):
        if (denied := browser_guard(request)) is not None:
            return denied
        return Response(await app.hub.call("browser", "browser_details", "{}"), media_type="application/json")

    async def browser_screen(request: Request):
        if (denied := browser_guard(request)) is not None:
            return denied
        path = ROOT / "data" / "browser" / "screen.jpg"
        if not path.exists():
            return Response(status_code=204)
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    async def browser_approve(request: Request):
        if (denied := browser_guard(request)) is not None:
            return denied
        body = await request.json()
        args = json.dumps({"approve": bool(body.get("approve"))})
        return Response(await app.hub.call("browser", "confirm_browser_action", args), media_type="application/json")

    async def browser_open(request: Request):
        if (denied := browser_guard(request)) is not None:
            return denied
        body = await request.json()
        return Response(await app.hub.call("browser", "open_page", json.dumps({"url": str(body.get("url", ""))})),
                        media_type="application/json")

    async def browser_task(request: Request):
        """Typed instead of spoken: start a browser task from the UI."""
        if (denied := browser_guard(request)) is not None:
            return denied
        body = await request.json()
        args = json.dumps({"goal": str(body.get("goal", ""))[:500], "plan_first": bool(body.get("plan_first"))})
        return Response(await app.hub.call("browser", "queue_browser_task", args), media_type="application/json")

    async def browser_plan(request: Request):
        """Edit the plan under review: {"steps": [...]} replaces it, {"instructions": "..."} re-plans."""
        if (denied := browser_guard(request)) is not None:
            return denied
        body = await request.json()
        if body.get("steps"):
            steps = [str(s)[:300] for s in body["steps"]][:10]
            out = await app.hub.call("browser", "set_browser_plan", json.dumps({"steps": steps}))
        else:
            out = await app.hub.call("browser", "revise_browser_plan",
                                     json.dumps({"instructions": str(body.get("instructions", ""))[:500]}))
        return Response(out, media_type="application/json")

    async def browser_visible(request: Request):
        if (denied := browser_guard(request)) is not None:
            return denied
        body = await request.json()
        return Response(await app.hub.call("browser", "set_browser_visible",
                                           json.dumps({"visible": bool(body.get("visible"))})),
                        media_type="application/json")

    async def browser_stop(request: Request):
        if (denied := browser_guard(request)) is not None:
            return denied
        return Response(await app.hub.call("browser", "stop_browser_task", "{}"), media_type="application/json")

    async def browser_audit(request: Request):
        if (denied := browser_guard(request)) is not None:
            return denied
        return Response(await app.hub.call("browser", "browser_audit", json.dumps({"limit": 150})),
                        media_type="application/json")

    async def resume_api(request: Request):
        """GET: what's loaded. POST raw file bytes with ?name=cv.pdf (pdf, txt or md)."""
        if (denied := browser_guard(request)) is not None:
            return denied
        text_file = RESUME_DIR / "resume.txt"
        if request.method == "GET":
            if not text_file.exists():
                return JSONResponse({"loaded": False, "name": "Sample resume (Ananya Rao)"})
            meta = json.loads((RESUME_DIR / "meta.json").read_text(encoding="utf-8"))
            return JSONResponse({"loaded": True, "name": meta["name"]})
        name = request.query_params.get("name", "resume.txt")
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in ("pdf", "txt", "md"):
            return JSONResponse({"error": "Upload a PDF, .txt or .md file."}, status_code=415)
        data = await request.body()
        if not data or len(data) > MAX_RESUME_BYTES:
            return JSONResponse({"error": "The file is empty or larger than 5 MB."}, status_code=413)
        if ext == "pdf":
            text = await asyncio.to_thread(pdf_text, data)
            if not text.strip():
                return JSONResponse({"error": "No text found in that PDF. Is it a scan? Try a .txt version."},
                                    status_code=422)
        else:
            text = data.decode("utf-8", errors="replace")
        RESUME_DIR.mkdir(parents=True, exist_ok=True)
        for old in RESUME_DIR.glob("original.*"):
            old.unlink()
        (RESUME_DIR / f"original.{ext}").write_bytes(data)
        text_file.write_text(text, encoding="utf-8")
        (RESUME_DIR / "meta.json").write_text(json.dumps({"name": name}), encoding="utf-8")
        return JSONResponse({"loaded": True, "name": name, "chars": len(text)})

    # -- phone ----------------------------------------------------------------------------------------
    async def twilio_voice(request: Request):
        form = dict(parse_qsl((await request.body()).decode()))
        base = app.cfg.public_url or f"https://{request.headers.get('host', '')}"
        if app.cfg.twilio_auth_token:
            url = base + request.url.path + (f"?{request.url.query}" if request.url.query else "")
            if not valid_signature(app.cfg.twilio_auth_token, url, form, request.headers.get("x-twilio-signature", "")):
                return Response("bad signature", status_code=403)
        line = request.query_params.get("line", next(iter(PERSONAS)))
        ws_url = base.replace("https://", "wss://").replace("http://", "ws://") + "/twilio/stream"
        return Response(twiml_stream(ws_url, line), media_type="application/xml")

    return Starlette(routes=[
        Route("/", landing),
        Route("/app", index),
        Route("/latency", dashboard),
        Route("/demo/job-application", demo_form),
        Route("/api/latency", latency_api),
        Route("/api/personas", personas_api),
        Route("/api/config", config_api),
        Route("/api/browser", browser_api),
        Route("/api/browser/screen.jpg", browser_screen),
        Route("/api/browser/approve", browser_approve, methods=["POST"]),
        Route("/api/browser/open", browser_open, methods=["POST"]),
        Route("/api/browser/task", browser_task, methods=["POST"]),
        Route("/api/browser/plan", browser_plan, methods=["POST"]),
        Route("/api/browser/visible", browser_visible, methods=["POST"]),
        Route("/api/browser/stop", browser_stop, methods=["POST"]),
        Route("/api/browser/audit", browser_audit),
        Route("/api/resume", resume_api, methods=["GET", "POST"]),
        Route("/twilio/voice", twilio_voice, methods=["POST"]),
        WebSocketRoute("/ws", app.ws_call),
        WebSocketRoute("/twilio/stream", app.twilio_stream),
        Mount("/static", StaticFiles(directory=WEB)),
    ])


def pdf_text(data: bytes) -> str:
    import io

    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-8s %(message)s", datefmt="%H:%M:%S")
    missing = CONFIG.missing_keys()
    if missing:
        sys.exit(f"Missing API keys: {', '.join(missing)}. Copy .env.example to .env and fill them in.")
    if not PERSONAS:
        sys.exit("No lines to serve: check lines/*.toml and ENABLED_LINES.")
    app = App()
    try:
        await app.start()
        server = uvicorn.Server(uvicorn.Config(build_app(app), host=CONFIG.host, port=CONFIG.port,
                                               log_level="warning", proxy_headers=True, forwarded_allow_ips="*"))
        log.info("ready: http://localhost:%d  (latency dashboard: /latency)", CONFIG.port)
        await server.serve()
    except Exception:
        log.exception("fatal error")
        raise
    finally:
        # Close MCP subprocesses on every exit path; left open, asyncio.run() hangs on shutdown.
        await app.tts.close()
        await app.hub.close()


if __name__ == "__main__":
    asyncio.run(main())
