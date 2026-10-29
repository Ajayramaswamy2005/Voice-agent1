"""MCP server: customer support backend for "NovaMart", a fictional Indian e-commerce store.

Run standalone with `uv run python -m mcp_servers.support` (stdio transport).
"""

import asyncio
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from mcp.server.mcpserver import MCPServer

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "support.db"
LATENCY_S = (0.6, 1.4)

mcp = MCPServer("support", version="0.1.0")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def seed(today: date | None = None) -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    today = today or date.today()
    d = lambda n: (today - timedelta(days=n)).isoformat()  # noqa: E731
    orders = [
        ("OD1001", "9876543210", "Priya Nair", "Boat Rockerz wireless headphones", 1799, d(6), "delivered", "BlueDart", "Delivered to customer"),
        ("OD1002", "9876543210", "Priya Nair", "Cotton kurta set (M)", 1299, d(2), "shipped", "Delhivery", "Out for delivery, Bengaluru hub"),
        ("OD1003", "9123456780", "Arjun Singh", "Prestige pressure cooker 5L", 2499, d(1), "packed", "Ekart", "Awaiting pickup"),
        ("OD1004", "9123456780", "Arjun Singh", "Mi power bank 20000mAh", 1499, d(12), "delivered", "BlueDart", "Delivered, signed by Arjun"),
        ("OD1005", "9988776655", "Fatima Shaikh", "Kids school bag", 899, d(4), "delayed", "Delhivery", "Held at Pune sorting centre due to weather"),
    ]
    with _db() as c:
        c.executescript(
            """
            DROP TABLE IF EXISTS orders; DROP TABLE IF EXISTS tickets; DROP TABLE IF EXISTS refunds;
            CREATE TABLE orders (id TEXT PRIMARY KEY, phone TEXT, customer TEXT, item TEXT, amount INTEGER,
                ordered_on TEXT, status TEXT, courier TEXT, tracking_note TEXT);
            CREATE TABLE tickets (id TEXT PRIMARY KEY, phone TEXT, order_id TEXT, category TEXT, summary TEXT,
                priority TEXT, status TEXT, created_on TEXT);
            CREATE TABLE refunds (id TEXT PRIMARY KEY, order_id TEXT, reason TEXT, amount INTEGER, status TEXT);
            """
        )
        c.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", orders)
        c.execute(
            "INSERT INTO tickets VALUES ('T501','9988776655','OD1005','delivery','Order delayed beyond promised date',"
            "'normal','open',?)",
            (d(1),),
        )


async def _backend_delay() -> None:
    await asyncio.sleep(random.uniform(*LATENCY_S))


def _norm_phone(phone: str) -> str:
    return "".join(ch for ch in phone if ch.isdigit())[-10:]


def _norm_order(order_id: str) -> str:
    s = "".join(ch for ch in order_id.upper() if ch.isalnum())
    return s if s.startswith("OD") else f"OD{s}"


@mcp.tool()
async def lookup_orders(phone: str) -> list[dict]:
    """Recent orders for the caller's registered phone number."""
    await _backend_delay()
    with _db() as c:
        rows = c.execute(
            "SELECT id, item, amount, ordered_on, status FROM orders WHERE phone=? ORDER BY ordered_on DESC",
            (_norm_phone(phone),),
        ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
async def track_order(order_id: str) -> dict:
    """Shipping status, courier and latest tracking note for an order ID like OD1002."""
    await _backend_delay()
    with _db() as c:
        row = c.execute("SELECT * FROM orders WHERE id=?", (_norm_order(order_id),)).fetchone()
    if row is None:
        return {"error": "Order not found. Ask the caller to re-check the order ID."}
    out = dict(row)
    out.pop("phone")
    if out["status"] in ("shipped", "packed"):
        out["expected_delivery"] = (date.fromisoformat(out["ordered_on"]) + timedelta(days=4)).isoformat()
    return out


@mcp.tool()
async def request_refund(order_id: str, reason: str) -> dict:
    """Start a return/refund. Only delivered orders within 10 days are eligible."""
    await _backend_delay()
    oid = _norm_order(order_id)
    with _db() as c:
        row = c.execute("SELECT * FROM orders WHERE id=?", (oid,)).fetchone()
        if row is None:
            return {"ok": False, "error": "Order not found."}
        if row["status"] != "delivered":
            return {"ok": False, "error": f"Order is '{row['status']}'; refunds open after delivery. "
                    "Offer cancellation via a ticket instead."}
        age = (date.today() - date.fromisoformat(row["ordered_on"])).days
        if age > 10:
            return {"ok": False, "error": f"Return window (10 days) has passed: ordered {age} days ago."}
        n = c.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]
        rid = f"R{9001 + n}"
        c.execute("INSERT INTO refunds VALUES (?,?,?,?,'initiated')", (rid, oid, reason, row["amount"]))
    return {"ok": True, "refund_id": rid, "amount_inr": row["amount"],
            "note": "Pickup in 1-2 days; refund to original payment method 5-7 days after pickup."}


@mcp.tool()
async def create_ticket(phone: str, category: str, summary: str, order_id: str | None = None,
                        priority: str = "normal") -> dict:
    """Open a support ticket. category: delivery | payment | product | account | other.
    priority: normal | high."""
    await _backend_delay()
    with _db() as c:
        n = c.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        tid = f"T{501 + n}"
        c.execute(
            "INSERT INTO tickets VALUES (?,?,?,?,?,?,'open',?)",
            (tid, _norm_phone(phone), _norm_order(order_id) if order_id else None, category, summary,
             priority, date.today().isoformat()),
        )
    return {"ok": True, "ticket_id": tid, "sla": "24 hours" if priority == "high" else "48 hours"}


@mcp.tool()
async def ticket_status(ticket_id: str) -> dict:
    """Status of an existing support ticket, e.g. T501."""
    await _backend_delay()
    tid = ticket_id.upper().replace(" ", "")
    tid = tid if tid.startswith("T") else f"T{tid}"
    with _db() as c:
        row = c.execute("SELECT * FROM tickets WHERE id=?", (tid,)).fetchone()
    return dict(row) if row else {"error": "Ticket not found."}


@mcp.tool()
async def transfer_to_human(reason: str) -> dict:
    """Hand the call to a human agent when the caller asks for one or is very upset."""
    await asyncio.sleep(0.2)
    return {"ok": True, "queue_position": random.randint(1, 4), "estimated_wait_min": random.randint(2, 6)}


if __name__ == "__main__":
    import sys

    if "--reseed" in sys.argv or not DB_PATH.exists():
        seed()
    mcp.run(transport="stdio")
