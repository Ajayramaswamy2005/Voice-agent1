"""MCP server: a browser assistant the voice agent can hand work to (see browser_agent.py).

Run standalone with `uv run python -m mcp_servers.browser` (stdio transport). Needs GROQ_API_KEY (or
VISION_* settings) and `uv run playwright install chromium`.
"""

import sys

from groq import AsyncGroq
from mcp.server.mcpserver import MCPServer

from agent.config import CONFIG, ROOT
from mcp_servers.browser_agent import BrowserAgent

RESUME_DIR = ROOT / "data" / "resume"
SAMPLE_RESUME = ROOT / "data" / "resume_sample.md"

mcp = MCPServer("browser", version="0.2.0")
agent = BrowserAgent(
    AsyncGroq(api_key=CONFIG.vision_key or CONFIG.groq_key, base_url=CONFIG.vision_base_url or None, max_retries=3),
    model=CONFIG.vision_model,
    screen_dir=ROOT / "data" / "browser",
    headless=CONFIG.browser_headless,
    start_url=CONFIG.browser_start_url,
    profile_dir=CONFIG.browser_profile_dir,
    cdp_url=CONFIG.browser_cdp_url,
    allowed_sites=CONFIG.browser_allowed_sites,
    vision_mode=CONFIG.browser_vision,
)


def load_resume() -> None:
    """The uploaded resume (data/resume/), or the sample. resume.txt holds the text, the original is kept for
    file-upload fields."""
    text_file = RESUME_DIR / "resume.txt"
    originals = [p for p in RESUME_DIR.glob("original.*")] if RESUME_DIR.exists() else []
    if text_file.exists():
        agent.resume_text = text_file.read_text(encoding="utf-8")
        agent.resume_file = originals[0] if originals else text_file
    elif SAMPLE_RESUME.exists():
        agent.resume_text = SAMPLE_RESUME.read_text(encoding="utf-8")
        agent.resume_file = SAMPLE_RESUME


# -- for the LLM ------------------------------------------------------------------------------------------
@mcp.tool()
async def start_browser_task(goal: str, url: str = "", plan_first: bool = False) -> dict:
    """Hand a task to the browser: fill a form, find information, compare things, get something done on a
    website. Returns at once; the plan, approval requests and the result arrive later as [update] messages.
    If a task is already running, this one is queued. url: optional page to open first. plan_first: true for
    a bigger multi-step task, so the caller can review the plan before it starts."""
    load_resume()
    if url and not agent.busy:
        await agent.open(url)
    return {"status": agent.start(goal, plan_first)}


@mcp.tool()
async def confirm_browser_action(approve: bool) -> dict:
    """Answer the browser's pending question (start this plan? / do this risky step?) with the caller's yes
    (true) or no (false)."""
    return {"status": agent.decide(approve)}


@mcp.tool()
async def revise_browser_plan(instructions: str) -> dict:
    """Change the plan that is waiting for review, following the caller's words ("skip step 2", "also check
    Naukri")."""
    return {"status": await agent.revise_plan(instructions=instructions)}


@mcp.tool()
async def stop_browser_task() -> dict:
    """Stop the current browser task and clear the queue."""
    return {"status": await agent.stop()}


@mcp.tool()
async def browser_status() -> dict:
    """What the browser is doing: state, plan, current page, recent steps, pending question, and the result
    (answer) of the last task."""
    st = agent.status()
    return {k: st[k] for k in ("state", "goal", "url", "plan", "summary", "waiting_for_approval", "queue")} | {
        "answer": st["answer"][:1500], "steps": st["steps"][-3:]}


# -- for the app, not the LLM (the session hides event_tool; the UI calls the rest over HTTP) -------------
@mcp.tool()
async def wait_browser_event(after: int = -1, timeout_s: float = 20) -> dict:
    """Long-poll: the next spoken update after event id `after`. after=-1 returns the current id."""
    return await agent.wait_event(after, min(timeout_s, 30))


@mcp.tool()
async def browser_details() -> dict:
    """Full status for the UI, including the step log."""
    return agent.status()


@mcp.tool()
async def open_page(url: str) -> dict:
    """Open a URL in the browser."""
    return {"url": await agent.open(url)}


@mcp.tool()
async def set_browser_plan(steps: list[str]) -> dict:
    """Replace the plan under review with these steps (edited in the UI)."""
    return {"status": await agent.revise_plan(steps=steps)}


@mcp.tool()
async def set_browser_visible(visible: bool) -> dict:
    """Show the browser window so the user can sign in to sites themselves, or hide it again."""
    return {"status": await agent.set_visible(visible)}


@mcp.tool()
async def browser_audit(limit: int = 100) -> dict:
    """The newest entries of the activity log: every task, plan, action, approval and result."""
    return {"entries": agent.read_audit(min(limit, 500))}


@mcp.tool()
async def queue_browser_task(goal: str, plan_first: bool = False) -> dict:
    """Start a task typed in the UI (queued if one is running)."""
    load_resume()
    return {"status": agent.start(goal, plan_first)}


if __name__ == "__main__":
    load_resume()
    if "--headed" in sys.argv:
        agent.headless = False
    mcp.run(transport="stdio")
