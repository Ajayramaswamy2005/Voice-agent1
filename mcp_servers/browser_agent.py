"""A browser assistant that does work for you: fills forms, researches, and reports back.

Each step:
  1. Observe. Set-of-marks numbers every visible interactive element (and draws the numbers on the page);
     the element list and a page-text excerpt always go to the model. A screenshot goes too when it's
     needed: a new page, after a failure, every few steps, when the model asks to look, and always before
     a task may count as done. Reading the page structure first is what keeps a task fast and cheap.
  2. Act. The model picks actions by mark number or, when nothing is marked, by predicted x/y pixel
     coordinates. It can also read whole pages, keep notes, search, and work across tabs.
  3. Gate. Risky actions (submit, send, post, pay, delete, upload, sensitive fields) pause for a human
     yes/no; sites off the allowed list are blocked outright.
  4. Verify. After each action the DOM is checked (did the value stick? did the page react?), and the
     next screenshot asks the model whether the previous step worked. Failures are retried, then reported.

Bigger tasks start with a plan the user can approve or edit. Tasks typed while one is running are queued.
Every action, approval and result is appended to an audit log.
"""

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus, urlparse

log = logging.getLogger("browser")

VIEWPORT = {"width": 1024, "height": 720}  # small screenshot = fewer image tokens per step
MAX_STEPS = 40
MAX_FAILS = 3
IMAGE_EVERY = 4  # in "auto" vision mode, re-check with a screenshot at least this often
MAX_QUOTA_WAIT_S = 15 * 60  # rate-limited for less than this: pause and carry on instead of failing
SEARCH_URL = "https://duckduckgo.com/?q="

# Numbers every visible interactive element, draws a box + label for each, returns their descriptions.
SET_OF_MARKS_JS = r"""
() => {
  document.getElementById('__som')?.remove();
  document.querySelectorAll('[data-som]').forEach(e => e.removeAttribute('data-som'));
  const sel = 'a[href],button,input,textarea,select,summary,[role=button],[role=link],[role=checkbox],' +
              '[role=radio],[role=combobox],[role=tab],[role=menuitem],[role=switch],[role=textbox],' +
              '[contenteditable=true]';
  const layer = document.createElement('div');
  layer.id = '__som';
  layer.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483647';
  const out = [];
  const W = innerWidth, H = innerHeight;
  const text = s => (s || '').replace(/\s+/g, ' ').trim().slice(0, 60);
  for (const el of document.querySelectorAll(sel)) {
    if (el.disabled || el.type === 'hidden') continue;
    const r = el.getBoundingClientRect();
    const file = el.type === 'file';  // often visually hidden behind a styled button; still usable
    if (!file) {
      if (r.width < 3 || r.height < 3 || r.bottom < 0 || r.right < 0 || r.top > H || r.left > W) continue;
      const st = getComputedStyle(el);
      if (st.visibility === 'hidden' || st.opacity === '0') continue;
      const hit = document.elementFromPoint(Math.min(W - 1, Math.max(0, r.x + r.width / 2)),
                                            Math.min(H - 1, Math.max(0, r.y + r.height / 2)));
      if (hit && !el.contains(hit) && !hit.contains(el) && hit.closest('label')?.control !== el) continue;
    }
    const id = out.length + 1;
    el.setAttribute('data-som', id);
    const label = text(el.getAttribute('aria-label') || el.labels?.[0]?.innerText || el.placeholder ||
                       el.innerText || el.value || el.title || el.name || el.getAttribute('href'));
    const item = {id, tag: el.tagName.toLowerCase(), type: el.type || el.getAttribute('role') || '', label,
                  name: el.name || '', x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2)};
    if (el.tagName === 'A') item.href = (el.getAttribute('href') || '').slice(0, 80);
    if (el.tagName === 'SELECT') item.options = [...el.options].slice(0, 12).map(o => text(o.text));
    if (el.type === 'checkbox' || el.type === 'radio') item.checked = el.checked;
    else if ('value' in el && el.tagName !== 'BUTTON' && !file) item.value = text(el.value);
    if (el.required) item.required = true;
    out.push(item);
    if (!file) {
      const b = document.createElement('div');
      b.style.cssText = `position:fixed;left:${r.x}px;top:${r.y}px;width:${r.width}px;height:${r.height}px;` +
                        'border:2px solid #2457f5;border-radius:3px;box-sizing:border-box';
      const t = document.createElement('span');
      t.textContent = id;
      t.style.cssText = 'position:absolute;left:-2px;top:-16px;background:#2457f5;color:#fff;font:600 11px/14px ' +
                        'system-ui;padding:0 3px;border-radius:3px';
      b.append(t);
      layer.append(b);
    }
  }
  document.documentElement.append(layer);
  return out;
}
"""
CLEAR_MARKS_JS = "() => document.getElementById('__som')?.remove()"
# Readable text of the page (main content first), for reading/research without a screenshot.
PAGE_TEXT_JS = r"""
(limit) => {
  const root = document.querySelector('main, [role=main], article') || document.body;
  const t = (root.innerText || '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  return t.slice(0, limit);
}
"""
# What's under a predicted coordinate, so coordinate clicks go through the same safety gate.
ELEMENT_AT_JS = r"""
([x, y]) => {
  const el = document.elementFromPoint(x, y)?.closest('a,button,input,textarea,select,[role],label') ||
             document.elementFromPoint(x, y);
  if (!el) return null;
  return {tag: el.tagName.toLowerCase(), type: el.type || el.getAttribute('role') || '',
          label: (el.getAttribute('aria-label') || el.innerText || el.value || '').trim().slice(0, 60)};
}
"""
PAGE_SIGNATURE_JS = "() => location.href + '|' + document.body.innerText.length + '|' + " \
                    "document.querySelectorAll('*').length + '|' + (document.activeElement?.outerHTML || '').slice(0, 80)"

# Clicking these changes something for real (or for other people): always ask first.
RISKY_LABEL = re.compile(
    r"\b(submit|send|apply|pay|purchase|buy|order|checkout|check out|confirm|delete|remove|sign ?up|register|"
    r"book|transfer|publish|post|place|finish|complete|reply|tweet|share|comment|message|follow|connect|"
    r"invite|accept|decline|unsubscribe|subscribe|cancel|withdraw|save changes|update|upload|archive|"
    r"merge|deploy|create (repository|account|project))\b", re.IGNORECASE)
SENSITIVE_FIELD = re.compile(
    r"password|passcode|\botp\b|one.time|cvv|cvc|card ?number|credit card|debit card|\bssn\b|social security|"
    r"aadhaar|\bpan\b|bank account|account number|ifsc|\bpin\b|upi", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are a browser assistant doing work for the user: filling forms, finding information, and getting tasks \
done on websites. You see the page as a list of numbered interactive elements plus a text excerpt, and \
sometimes a screenshot where the same numbers are drawn in blue boxes.

Reply with one JSON object only:
{"previous_ok": true|false|null, "observation": "what the page shows, short", "plan_done": [step numbers \
finished], "actions": [{"action": "...", "mark": 3, "text": "...", "why": "short"}], "done": false, \
"summary": "", "answer": ""}

Actions:
- {"action":"type","mark":N,"text":"..."}       replace a text field's value
- {"action":"select","mark":N,"text":"option"}  choose a dropdown option by its visible text
- {"action":"click","mark":N}                    click (also checkboxes, radios, links)
- {"action":"click","x":X,"y":Y}                 click at screenshot pixel coordinates, only when the target \
has no number
- {"action":"upload","mark":N}                   attach the user's resume file to a file input
- {"action":"scroll","text":"down"|"up"}
- {"action":"press","text":"Enter"|"Tab"|"Escape"}
- {"action":"navigate","text":"https://..."}     open a URL in this tab
- {"action":"search","text":"query"}             web search
- {"action":"new_tab","text":"https://..."}      open a URL in a new tab and switch to it
- {"action":"switch_tab","text":"2"}             switch to an open tab by number
- {"action":"close_tab"}                         close this tab
- {"action":"read"}                              get the full text of this page next step (for research)
- {"action":"note","text":"..."}                 save a finding for the final answer (facts, with source)
- {"action":"look"}                              ask for a screenshot next step when the text isn't enough
- {"action":"wait","text":"2"}                   wait some seconds for the page

Rules:
- previous_ok: does what you see now show that your previous actions worked? null on the first step.
- Up to 6 actions per reply. Put a click that submits, sends, posts or pays alone and last.
- Use only facts from the goal, the user's details and pages you saw. Never invent data. Leave a field empty \
if you don't know it, and say so.
- Never type passwords, OTPs, card, UPI or bank details. If a page needs a login, stop and tell the user to \
sign in using "Show browser window".
- For research, open several sources, use "note" for each finding with where it came from, then answer.
- Mark plan steps finished in plan_done as you complete them.
- Only when you can see the goal reached (e.g. a confirmation page, or you have the information), or you \
cannot go on, set "done": true and "actions": []. "summary": one short spoken sentence. "answer": the full \
result for the screen (findings, lists, links), plain text, empty if there is nothing to report. Never claim \
success you cannot see.
"""

PLAN_PROMPT = """\
Break the user's browser task into a short plan of 1 to 7 steps, each a plain sentence the user will read, \
like "Open LinkedIn Jobs and search for Python roles in Bengaluru". Simple tasks get 1 or 2 steps. Reply \
with one JSON object only: {"plan": ["...", "..."]}

Task: {goal}
Current page: {url}
{feedback}"""


@dataclass
class Step:
    n: int
    observation: str
    actions: list[dict]
    results: list[str] = field(default_factory=list)
    saw_image: bool = False


@dataclass
class Pending:
    description: str
    kind: str = "action"  # "action" (a risky step) | "plan" (start this plan?)
    decided: asyncio.Event = field(default_factory=asyncio.Event)
    approved: bool = False


def site_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """No list = every site. Otherwise the host or a subdomain of a listed site (example.com, *.example.com)."""
    if not allowed or not host:
        return True
    host = host.lower()
    return any(host == s or host.endswith("." + s) for s in allowed)


def risk_reason(act: dict, el: dict | None, page_host: str) -> str:
    """Why this action needs a human's OK, or "" if it's safe to just do."""
    kind = act.get("action")
    label = (el or {}).get("label", "") or ""
    what = f"“{label}”" if label else "that element"
    if kind == "click" and el and ((el.get("type") == "submit") or RISKY_LABEL.search(label)):
        return f"click {what}"
    if kind == "press" and str(act.get("text", "")).lower() == "enter" and el and el.get("tag") in ("input", "textarea"):
        return "press Enter, which may submit or send"
    if kind in ("type", "select") and el and SENSITIVE_FIELD.search(f"{label} {el.get('name', '')}"):
        return f"enter sensitive information into {what}"
    if kind == "upload":
        return f"upload your resume to {what}"
    if kind in ("navigate", "new_tab"):
        host = urlparse(str(act.get("text", ""))).hostname or ""
        if host and page_host and host != page_host and not host.endswith("." + page_host):
            return f"open another site, {host}"
    return ""


def parse_reply(text: str) -> dict:
    """Models sometimes wrap JSON in prose or code fences; take the outermost object."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:120]!r}")
    reply = json.loads(m.group(0))
    reply.setdefault("actions", [])
    return reply


class BrowserAgent:
    def __init__(self, llm_client, model: str, screen_dir: Path, headless: bool = True, start_url: str = "",
                 profile_dir: Path | None = None, cdp_url: str = "", allowed_sites: tuple[str, ...] = (),
                 vision_mode: str = "auto"):
        self.llm = llm_client
        self.model = model
        self.screen_dir = screen_dir
        self.headless = headless
        self.start_url = start_url
        self.profile_dir = profile_dir  # persistent profile: sign in once, sessions are kept
        self.cdp_url = cdp_url  # drive an already-running Chrome instead (chrome --remote-debugging-port=9222)
        self.allowed_sites = tuple(s.strip().lower() for s in allowed_sites if s.strip())
        self.vision_mode = vision_mode  # "auto" (screenshots when needed) | "always"
        self.audit_path = screen_dir / "audit.jsonl"
        self._pw = self._browser = self.context = self.page = None
        self._launch_lock = asyncio.Lock()

        self.state = "idle"  # idle | planning | plan_review | running | needs_approval | waiting_quota | done | failed | stopped
        self.goal = ""
        self.plan: list[dict] = []  # [{"text", "done"}]
        self.notes: list[str] = []
        self.summary = ""
        self.answer = ""
        self.steps: list[Step] = []
        self.pending: Pending | None = None
        self.queue: list[dict] = []  # [{"goal", "plan_first"}] waiting for the current task
        self.resume_text = ""
        self.resume_file: Path | None = None
        self.screen_version = 0
        self._task: asyncio.Task | None = None
        self._plan_feedback = ""
        self.events: list[str] = []  # spoken to the caller, in order; id = index + 1
        self._event_cond = asyncio.Condition()

    # -- browser lifecycle -------------------------------------------------------------------------
    async def ensure_page(self):
        async with self._launch_lock:
            if self.page is not None and not self.page.is_closed():
                return self.page
            if self.context is not None and self.context.pages:
                self.page = self.context.pages[-1]
                return self.page
            await self._launch()
            return self.page

    async def _launch(self) -> None:
        from playwright.async_api import async_playwright

        self._pw = self._pw or await async_playwright().start()
        if self.cdp_url:
            self._browser = await self._pw.chromium.connect_over_cdp(self.cdp_url)
            self.context = self._browser.contexts[0] if self._browser.contexts else await self._browser.new_context()
        elif self.profile_dir:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            self.context = await self._pw.chromium.launch_persistent_context(
                str(self.profile_dir), headless=self.headless, viewport=VIEWPORT, device_scale_factor=1)
        else:
            self._browser = await self._pw.chromium.launch(headless=self.headless)
            self.context = await self._browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        if self.start_url and self.page.url in ("", "about:blank"):
            try:
                await self.page.goto(self.start_url, wait_until="domcontentloaded", timeout=15000)
            except Exception as e:
                log.warning("could not open start page %s: %s", self.start_url, e)
        await self.snapshot()

    async def set_visible(self, visible: bool) -> str:
        """Show the real browser window (to sign in to sites yourself), or hide it again. The profile keeps
        the sessions either way."""
        if self.cdp_url:
            return "This is your own Chrome; it is always visible."
        if self.state in ("running", "needs_approval", "planning"):
            return "Wait for the current task to finish first."
        if visible == (not self.headless) and self.page is not None:
            return "visible" if visible else "hidden"
        url = self.page.url if self.page is not None and not self.page.is_closed() else ""
        await self._close_browser()
        self.headless = not visible
        await self.ensure_page()
        if url and url != "about:blank":
            await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await self.snapshot()
        return "visible" if visible else "hidden"

    async def open(self, url: str) -> str:
        page = await self.ensure_page()
        url = _with_scheme(url)
        host = urlparse(url).hostname or ""
        if not site_allowed(host, self.allowed_sites):
            raise ValueError(f"{host} is not on your allowed sites")
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await self.snapshot()
        return page.url

    async def _close_browser(self) -> None:
        try:
            if self.context is not None and not self.cdp_url:
                await self.context.close()
            if self._browser is not None and not self.cdp_url:
                await self._browser.close()
        except Exception:
            pass
        self.context = self._browser = self.page = None

    async def close(self) -> None:
        await self._close_browser()
        if self._pw:
            await self._pw.stop()

    async def observe(self, marks: bool = True) -> tuple[bytes, list[dict]]:
        """Screenshot (with set-of-marks drawn) and the element list. The screenshot is always taken (it's local
        and cheap) and kept on disk for the UI; whether the model sees it is decided per step."""
        page = self.page
        elements: list[dict] = []
        if marks:
            elements = await page.evaluate(SET_OF_MARKS_JS)
        try:
            img = await page.screenshot(type="jpeg", quality=70)
        finally:
            if marks:
                await page.evaluate(CLEAR_MARKS_JS)
        self._save_screen(img)
        return img, elements

    async def snapshot(self) -> bytes:
        img = await self.page.screenshot(type="jpeg", quality=70)
        self._save_screen(img)
        return img

    def _save_screen(self, img: bytes) -> None:
        self.screen_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.screen_dir / "screen.tmp"
        tmp.write_bytes(img)
        tmp.replace(self.screen_dir / "screen.jpg")
        self.screen_version += 1

    # -- events (the voice agent long-polls these and speaks them) and the audit log -----------------
    async def emit(self, text: str) -> None:
        async with self._event_cond:
            self.events.append(text)
            self._event_cond.notify_all()

    async def wait_event(self, after: int, timeout_s: float) -> dict:
        if after < 0:
            return {"id": len(self.events)}
        async with self._event_cond:
            try:
                await asyncio.wait_for(self._event_cond.wait_for(lambda: len(self.events) > after), timeout_s)
            except (asyncio.TimeoutError, TimeoutError):
                return {"id": after}
        return {"id": after + 1, "event": self.events[after]}

    def audit(self, kind: str, **data) -> None:
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "goal": self.goal,
               "url": self.page.url if self.page is not None and not self.page.is_closed() else "", **data}
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("audit log write failed: %s", e)

    def read_audit(self, limit: int = 100) -> list[dict]:
        if not self.audit_path.exists():
            return []
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # -- task control ------------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.state in ("planning", "plan_review", "running", "needs_approval", "waiting_quota")

    def start(self, goal: str, plan_first: bool = False) -> str:
        goal = goal.strip()
        if not goal:
            return "No task given."
        if self.busy:
            self.queue.append({"goal": goal, "plan_first": plan_first})
            return f"queued: {len(self.queue)} task(s) waiting after the current one"
        self._begin(goal, plan_first)
        return "started"

    def _begin(self, goal: str, plan_first: bool) -> None:
        self.goal, self.summary, self.answer = goal, "", ""
        self.steps, self.notes, self.plan, self.pending = [], [], [], None
        self._plan_feedback = ""
        self.state = "planning"
        self.audit("task_start", plan_first=plan_first)
        self._task = asyncio.create_task(self._run(plan_first))

    def decide(self, approve: bool) -> str:
        if self.pending is None:
            return "Nothing is waiting for approval."
        self.pending.approved = approve
        self.pending.decided.set()
        return "approved" if approve else "declined"

    async def revise_plan(self, steps: list[str] | None = None, instructions: str = "") -> str:
        """Edit the plan under review: replace it with `steps`, or re-plan following `instructions`."""
        if self.state != "plan_review" or self.pending is None:
            return "There is no plan waiting for review."
        if steps:
            self.plan = [{"text": s.strip(), "done": False} for s in steps if s.strip()][:10]
            self.audit("plan_edited", plan=[p["text"] for p in self.plan])
            await self.emit(f"Plan updated: {self._plan_text()}. Should I start?")
            return "plan updated"
        if instructions:
            self._plan_feedback = instructions
            self.pending.approved = False
            self.pending.kind = "replan"
            self.pending.decided.set()
            return "re-planning"
        return "Nothing to change."

    async def stop(self) -> str:
        self.queue.clear()
        if self._task and not self._task.done():
            self._task.cancel()
            self.state = "stopped"
            self.pending = None
            self.audit("task_stopped")
            return "stopped"
        return "Nothing is running."

    def status(self) -> dict:
        return {
            "state": self.state,
            "goal": self.goal,
            "url": self.page.url if self.page is not None and not self.page.is_closed() else "",
            "tabs": len(self.context.pages) if self.context is not None else 0,
            "plan": self.plan,
            "summary": self.summary,
            "answer": self.answer,
            "notes": self.notes[-20:],
            "waiting_for_approval": self.pending.description if self.pending else "",
            "approval_kind": self.pending.kind if self.pending else "",
            "queue": [q["goal"] for q in self.queue],
            "steps": [{"n": s.n, "observation": s.observation, "saw_image": s.saw_image,
                       "actions": [{k: a[k] for k in ("action", "mark", "text", "x", "y", "why") if k in a}
                                   for a in s.actions],
                       "results": s.results} for s in self.steps[-12:]],
            "screen_version": self.screen_version,
            "resume_loaded": bool(self.resume_text),
            "window_visible": not self.headless or bool(self.cdp_url),
            "allowed_sites": list(self.allowed_sites),
            "events": self.events[-8:],
            "event_count": len(self.events),
        }

    # -- the loop ------------------------------------------------------------------------------------
    async def _run(self, plan_first: bool) -> None:
        try:
            await self.ensure_page()
            if not await self._make_plan(plan_first):
                return
            self.state = "running"
            await self._work()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("browser task crashed")
            await self._finish("failed", f"Something broke: {e}")
        finally:
            if self.queue and not self.busy:
                nxt = self.queue.pop(0)
                self._begin(nxt["goal"], nxt["plan_first"])

    async def _make_plan(self, plan_first: bool) -> bool:
        """Plan the task. Short plans start right away; longer ones (or plan_first) wait for the user's OK,
        and can be edited or re-planned meanwhile. Returns False if the user declined."""
        while True:
            reply = await self._ask(PLAN_PROMPT.replace("{goal}", self.goal).replace("{url}", self.page.url)
                                    .replace("{feedback}", f"User's changes to the last plan: {self._plan_feedback}"
                                             if self._plan_feedback else ""), image=None, system=None)
            self.plan = [{"text": str(s), "done": False} for s in (reply.get("plan") or [self.goal])][:10]
            self.audit("plan", plan=[p["text"] for p in self.plan])
            if len(self.plan) <= 2 and not plan_first and not self._plan_feedback:
                return True
            self.state = "plan_review"
            self.pending = Pending("start this plan", kind="plan")
            await self.snapshot()
            await self.emit(f"Here's my plan: {self._plan_text()}. Should I start?")
            try:
                await asyncio.wait_for(self.pending.decided.wait(), timeout=900)
            except (asyncio.TimeoutError, TimeoutError):
                self.pending.approved = False
            pending, self.pending = self.pending, None
            if pending.kind == "replan":
                self.state = "planning"
                continue
            self.audit("plan_decision", approved=pending.approved)
            if not pending.approved:
                await self._finish("stopped", "Okay, I won't do that task.", speak="Okay, I won't start that.")
                return False
            return True

    async def _work(self) -> None:
        fails = 0
        last_results: list[str] = []
        read_text = ""
        force_image = True
        image_url_seen = ""
        last_image_step = 0
        for n in range(1, MAX_STEPS + 1):
            img, elements = await self.observe()
            url = self.page.url
            with_image = (self.vision_mode == "always" or force_image or url != image_url_seen
                          or n - last_image_step >= IMAGE_EVERY)
            if with_image:
                image_url_seen, last_image_step = url, n
            reply = await self._think(img if with_image else None, elements, last_results, read_text)
            read_text, force_image = "", False
            if reply.get("previous_ok") is False and self.steps:
                self.steps[-1].results.append("model: the page does not show this worked")
                fails += 1
                force_image = True
            for i in reply.get("plan_done") or []:
                if isinstance(i, int) and 1 <= i <= len(self.plan):
                    self.plan[i - 1]["done"] = True
            step = Step(n, str(reply.get("observation", ""))[:200], reply["actions"][:6], saw_image=with_image)
            self.steps.append(step)

            if not step.actions:
                # Finish only on a step with no actions, and only after looking at a screenshot: a reply that
                # claims success from text alone gets one more look first.
                if not with_image and reply.get("done"):
                    force_image = True
                    step.results.append("checking the screen before finishing")
                    continue
                if reply.get("done") or n > 1:
                    await self._finish("done", str(reply.get("summary") or "Finished."), answer=str(reply.get("answer") or ""))
                    return
                force_image = True
                continue

            last_results = []
            by_id = {e["id"]: e for e in elements}
            for act in step.actions:
                result = await self._act(act, by_id)
                step.results.append(result)
                self.audit("action", action={k: act[k] for k in ("action", "mark", "text", "x", "y") if k in act},
                           result=result)
                last_results.append(f"{act.get('action')} {act.get('mark', '')}: {result}")
                if act.get("action") == "read" and result == "ok":
                    read_text = await self.page.evaluate(PAGE_TEXT_JS, 8000)
                if act.get("action") == "look":
                    force_image = True
                if result.startswith("declined"):
                    what = result.removeprefix("declined: ")
                    await self._finish("done", f"Stopped before I {what}, as you asked.",
                                       speak=f"Okay, I did not {what}. I've stopped there.")
                    return
                if result.startswith(("failed", "blocked")):
                    fails += 1
                    force_image = True
                    break  # look again before doing more
            if fails >= MAX_FAILS:
                await self._finish("failed", f"Gave up after repeated problems. Last one: {last_results[-1]}")
                return
            await asyncio.sleep(0.3)  # let the page settle before the next look
        await self._finish("failed", f"Stopped after {MAX_STEPS} steps without finishing.")

    async def _finish(self, state: str, summary: str, answer: str = "", speak: str = "") -> None:
        self.state, self.summary = state, summary
        self.answer = answer or ("\n".join(f"- {n}" for n in self.notes) if self.notes else "")
        self.pending = None
        try:
            await self.snapshot()
        except Exception:
            pass
        self.audit("task_end", state=state, summary=summary, answer=self.answer[:2000])
        more = f" {len(self.queue)} more task(s) queued." if self.queue else ""
        label = {"done": "Task finished", "failed": "Task failed", "stopped": "Task stopped"}[state]
        await self.emit((speak or f"{label}: {summary}") + (" The details are on screen." if self.answer else "") + more)

    def _plan_text(self) -> str:
        return "; ".join(f"{i}. {p['text']}" for i, p in enumerate(self.plan, 1))

    async def _think(self, img: bytes | None, elements: list[dict], last_results: list[str], read_text: str) -> dict:
        history = "\n".join(
            f"step {s.n}: " + "; ".join(f"{a.get('action')} {a.get('mark', '')} {a.get('text', '')!s:.40} -> {r}"
                                        for a, r in zip(s.actions, s.results))
            for s in self.steps[-6:]
        ) or "(none)"
        tabs = self.context.pages if self.context is not None else [self.page]
        tab_list = "\n".join(f"{i}. {'(this tab) ' if p is self.page else ''}{p.url}" for i, p in enumerate(tabs, 1))
        page_text = await self.page.evaluate(PAGE_TEXT_JS, 1500)
        parts = [
            f"Goal: {self.goal}",
            "Plan:\n" + "\n".join(f"{i}. [{'x' if p['done'] else ' '}] {p['text']}" for i, p in enumerate(self.plan, 1)),
            f"User's details (resume/profile):\n{self.resume_text[:5000] or '(none uploaded)'}",
            f"Resume file for uploads: {'available' if self.resume_file else 'none'}",
            f"Allowed sites: {', '.join(self.allowed_sites) or 'any'}",
            f"Notes so far:\n" + ("\n".join(f"- {n}" for n in self.notes[-15:]) or "(none)"),
            f"Tabs:\n{tab_list}",
            f"Page: {self.page.url}  title: {await self.page.title()}",
            f"Previous steps:\n{history}",
            f"Verification of the last actions: {'; '.join(last_results) or '(none)'}",
        ]
        if read_text:
            parts.append(f"Full page text you asked to read:\n{read_text}")
        else:
            parts.append(f"Page text (start):\n{page_text}")
        parts.append("Elements:\n" + "\n".join(json.dumps(e, ensure_ascii=False) for e in elements[:80]))
        parts.append("Screenshot attached." if img else "No screenshot this step (use \"look\" if you need one).")
        return await self._ask("\n\n".join(parts), image=img, system=SYSTEM_PROMPT)

    async def _ask(self, user: str, image: bytes | None, system: str | None) -> dict:
        content: str | list = user
        if image is not None:
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}}]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        attempt = 0
        while attempt < 3:
            try:
                resp = await self.llm.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.1, max_tokens=900,
                    **({"reasoning_effort": "none"} if "qwen3" in self.model else {}),
                )
            except Exception as e:
                if getattr(e, "status_code", None) != 429:
                    raise
                # Out of quota. A background task can afford to wait a few minutes; longer than that, say so.
                wait_s = _retry_after_s(str(e))
                if wait_s is None or wait_s > MAX_QUOTA_WAIT_S:
                    raise RuntimeError("the model's rate limit is used up" +
                                       (f"; it frees up in about {round(wait_s / 60)} minutes" if wait_s else "")) from e
                prev_state, self.state = self.state, "waiting_quota"
                self.audit("quota_wait", seconds=round(wait_s))
                await self.emit(f"I've hit the model's rate limit, so I'm pausing for about "
                                f"{max(1, round(wait_s / 60))} minute{'s' if wait_s > 90 else ''}. I'll carry on by myself.")
                await asyncio.sleep(wait_s + 2)
                self.state = prev_state
                continue
            attempt += 1
            try:
                return parse_reply(resp.choices[0].message.content)
            except (ValueError, json.JSONDecodeError) as e:
                log.warning("unparseable reply (attempt %d): %s", attempt + 1, e)
        raise RuntimeError("the model kept replying without valid JSON")

    async def _act(self, act: dict, by_id: dict[int, dict]) -> str:
        page = self.page
        kind = act.get("action")
        mark = act.get("mark")
        el = by_id.get(int(mark)) if isinstance(mark, (int, float, str)) and str(mark).isdigit() else None
        if kind in ("type", "select", "upload") and el is None:
            return f"failed: no element numbered {mark}"
        if kind == "click" and el is None:
            if act.get("x") is None or act.get("y") is None:
                return f"failed: no element numbered {mark} and no coordinates"
            el = await page.evaluate(ELEMENT_AT_JS, [act["x"], act["y"]])

        # Allowed sites are a hard limit, not a question.
        if kind in ("navigate", "new_tab"):
            target = _with_scheme(str(act.get("text", "")))
            host = urlparse(target).hostname or ""
            if not site_allowed(host, self.allowed_sites):
                return f"blocked: {host} is not on your allowed sites"
        if kind == "search" and not site_allowed(urlparse(SEARCH_URL).hostname, self.allowed_sites):
            return "blocked: web search is not on your allowed sites"

        reason = risk_reason(act, el, urlparse(page.url).hostname or "")
        if reason and not await self._ask_approval(reason):
            return f"declined: {reason}"

        loc = page.locator(f'[data-som="{mark}"]') if el is not None and mark is not None else None
        before = await page.evaluate(PAGE_SIGNATURE_JS)
        try:
            if kind == "type":
                want = str(act.get("text", ""))
                await loc.fill(want, timeout=5000)
                got = await loc.input_value() if el.get("tag") in ("input", "textarea") else await loc.inner_text()
                return "ok" if got.strip() == want.strip() else f"failed: field shows {got[:40]!r}"
            if kind == "select":
                want = str(act.get("text", ""))
                try:
                    await loc.select_option(label=want, timeout=5000)
                except Exception:
                    await loc.select_option(value=want, timeout=5000)
                got = await loc.evaluate("e => e.options[e.selectedIndex]?.text || ''")
                return "ok" if got.strip().lower() == want.strip().lower() else f"failed: selected {got!r}"
            if kind == "upload":
                if not self.resume_file:
                    return "failed: no resume file uploaded"
                await loc.set_input_files(str(self.resume_file), timeout=5000)
                n = await loc.evaluate("e => e.files.length")
                return "ok" if n else "failed: no file attached"
            if kind == "click":
                if loc is not None:
                    checkable = el.get("type") in ("checkbox", "radio")
                    was = await loc.is_checked() if checkable else None
                    await loc.click(timeout=5000)
                    if checkable and el.get("type") == "checkbox" and (await loc.is_checked()) == was:
                        return "failed: checkbox did not change"
                else:
                    await page.mouse.click(float(act["x"]), float(act["y"]))
                await asyncio.sleep(0.4)
                if (blocked := await self._enforce_allowed()) is not None:
                    return blocked
                after = await page.evaluate(PAGE_SIGNATURE_JS)
                return "ok" if after != before else "ok (no visible change yet)"
            if kind == "scroll":
                await page.mouse.wheel(0, -500 if str(act.get("text")) == "up" else 500)
                return "ok"
            if kind == "press":
                await page.keyboard.press(str(act.get("text", "Enter")))
                await asyncio.sleep(0.3)
                return (await self._enforce_allowed()) or "ok"
            if kind == "navigate":
                await page.goto(_with_scheme(str(act.get("text"))), wait_until="domcontentloaded", timeout=20000)
                return "ok"
            if kind == "search":
                await page.goto(SEARCH_URL + quote_plus(str(act.get("text", ""))), wait_until="domcontentloaded",
                                timeout=20000)
                return "ok"
            if kind == "new_tab":
                self.page = await self.context.new_page()
                await self.page.goto(_with_scheme(str(act.get("text"))), wait_until="domcontentloaded", timeout=20000)
                return f"ok (tab {len(self.context.pages)})"
            if kind == "switch_tab":
                pages = self.context.pages
                i = int(str(act.get("text", "1")).strip() or 1)
                if not 1 <= i <= len(pages):
                    return f"failed: there are {len(pages)} tabs"
                self.page = pages[i - 1]
                await self.page.bring_to_front()
                return "ok"
            if kind == "close_tab":
                if len(self.context.pages) <= 1:
                    return "failed: this is the last tab"
                await page.close()
                self.page = self.context.pages[-1]
                return "ok"
            if kind == "read":
                return "ok"
            if kind == "note":
                text = str(act.get("text", "")).strip()
                if text:
                    self.notes.append(text[:400])
                return "ok"
            if kind == "look":
                return "ok"
            if kind == "wait":
                await asyncio.sleep(min(10.0, float(str(act.get("text") or 2))))
                return "ok"
        except Exception as e:
            return f"failed: {str(e).splitlines()[0][:120]}"
        return f"failed: unknown action {kind!r}"

    async def _enforce_allowed(self) -> str | None:
        """A click or Enter may have led off the allowed sites: go back and report it."""
        host = urlparse(self.page.url).hostname or ""
        if site_allowed(host, self.allowed_sites):
            return None
        try:
            await self.page.go_back(timeout=10000)
        except Exception:
            pass
        return f"blocked: that led to {host}, which is not on your allowed sites"

    async def _ask_approval(self, description: str) -> bool:
        """Human-in-the-loop gate: pause until the caller (by voice) or the UI approves or declines."""
        self.pending = Pending(description)
        self.state = "needs_approval"
        await self.snapshot()
        await self.emit(f"Approval needed: I'm about to {description}. Should I go ahead?")
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(self.pending.decided.wait(), timeout=300)
            approved = self.pending.approved
        except (asyncio.TimeoutError, TimeoutError):
            approved = False
        log.info("approval for %r: %s after %.0fs", description, approved, time.monotonic() - t0)
        self.audit("approval", description=description, approved=approved)
        self.pending = None
        self.state = "running"
        return approved


def _retry_after_s(message: str) -> float | None:
    """Groq's 429 text says e.g. "Please try again in 12m22.5s" or "in 3.2s"."""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", message)
    if not m or not any(m.groups()):
        return None
    h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mins * 60 + secs


def _with_scheme(url: str) -> str:
    url = url.strip()
    return url if urlparse(url).scheme else "https://" + url
