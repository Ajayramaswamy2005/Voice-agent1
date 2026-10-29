"""MCP server: appointment booking for a small multi-specialty clinic.

Run standalone with `uv run python -m mcp_servers.clinic` (stdio transport), or
point any MCP client (Claude Desktop, Claude Code, the voice agent) at it.
"""

import asyncio
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from mcp.server.mcpserver import MCPServer

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clinic.db"
# Simulated backend latency so the agent's "let me check" behaviour is exercised.
LATENCY_S = (0.6, 1.4)

mcp = MCPServer("clinic", version="0.1.0")

DOCTORS = [
    ("D1", "Dr. Ananya Rao", "general physician", "en,hi"),
    ("D2", "Dr. Vikram Mehta", "dermatology", "en,hi"),
    ("D3", "Dr. Sarah Thomas", "pediatrics", "en"),
    ("D4", "Dr. Imran Qureshi", "orthopedics", "en,hi"),
]
SLOT_TIMES = ["10:00", "10:30", "11:00", "11:30", "12:00", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30"]


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def seed(today: date | None = None) -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    today = today or date.today()
    with _db() as c:
        c.executescript(
            """
            DROP TABLE IF EXISTS doctors; DROP TABLE IF EXISTS slots; DROP TABLE IF EXISTS appointments;
            CREATE TABLE doctors (id TEXT PRIMARY KEY, name TEXT, specialty TEXT, languages TEXT);
            CREATE TABLE slots (doctor_id TEXT, start TEXT, booked INTEGER DEFAULT 0, PRIMARY KEY (doctor_id, start));
            CREATE TABLE appointments (
                id TEXT PRIMARY KEY, doctor_id TEXT, start TEXT, patient_name TEXT, phone TEXT,
                reason TEXT, status TEXT DEFAULT 'booked');
            """
        )
        c.executemany("INSERT INTO doctors VALUES (?,?,?,?)", DOCTORS)
        rng = random.Random(7)  # deterministic "already booked" pattern
        for offset in range(0, 14):
            day = today + timedelta(days=offset)
            if day.weekday() == 6:  # closed Sundays
                continue
            for doc_id, *_ in DOCTORS:
                for t in SLOT_TIMES:
                    c.execute(
                        "INSERT INTO slots VALUES (?,?,?)",
                        (doc_id, f"{day.isoformat()} {t}", int(rng.random() < 0.45)),
                    )
        # One existing appointment so lookup/reschedule/cancel can be demoed.
        start = c.execute(
            "SELECT start FROM slots WHERE doctor_id='D1' AND booked=0 ORDER BY start LIMIT 1 OFFSET 3"
        ).fetchone()["start"]
        c.execute("UPDATE slots SET booked=1 WHERE doctor_id='D1' AND start=?", (start,))
        c.execute(
            "INSERT INTO appointments VALUES ('A1001','D1',?,'Rahul Sharma','9876543210','fever','booked')",
            (start,),
        )


async def _backend_delay() -> None:
    await asyncio.sleep(random.uniform(*LATENCY_S))


def _norm_phone(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    return digits[-10:]


@mcp.tool()
async def list_doctors(specialty: str | None = None) -> list[dict]:
    """List clinic doctors, optionally filtered by specialty (e.g. 'dermatology', 'pediatrics')."""
    await _backend_delay()
    with _db() as c:
        rows = c.execute("SELECT * FROM doctors").fetchall()
    out = [dict(r) for r in rows]
    if specialty:
        out = [d for d in out if specialty.lower() in d["specialty"]] or out
    return out


@mcp.tool()
async def check_availability(date_str: str, doctor_id: str | None = None, specialty: str | None = None) -> dict:
    """Free appointment slots on a date (YYYY-MM-DD). Filter by doctor_id or specialty.
    Returns at most 6 slots per doctor."""
    await _backend_delay()
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "date_str must be YYYY-MM-DD"}
    query = (
        "SELECT s.doctor_id, d.name, d.specialty, s.start FROM slots s JOIN doctors d ON d.id=s.doctor_id "
        "WHERE s.booked=0 AND s.start LIKE ?"
    )
    args: list = [f"{day.isoformat()}%"]
    if doctor_id:
        query += " AND s.doctor_id=?"
        args.append(doctor_id)
    elif specialty:
        query += " AND d.specialty LIKE ?"
        args.append(f"%{specialty.lower()}%")
    with _db() as c:
        rows = c.execute(query + " ORDER BY s.doctor_id, s.start", args).fetchall()
    by_doc: dict[str, dict] = {}
    for r in rows:
        entry = by_doc.setdefault(r["doctor_id"], {"doctor_id": r["doctor_id"], "doctor": r["name"],
                                                   "specialty": r["specialty"], "slots": []})
        if len(entry["slots"]) < 6:
            entry["slots"].append(r["start"][-5:])
    if day.weekday() == 6:
        return {"date": day.isoformat(), "note": "Clinic is closed on Sundays.", "doctors": []}
    return {"date": day.isoformat(), "doctors": list(by_doc.values())}


@mcp.tool()
async def book_appointment(doctor_id: str, start: str, patient_name: str, phone: str, reason: str = "") -> dict:
    """Book a slot. `start` is 'YYYY-MM-DD HH:MM'. Confirm details with the caller before calling."""
    await _backend_delay()
    with _db() as c:
        slot = c.execute("SELECT booked FROM slots WHERE doctor_id=? AND start=?", (doctor_id, start)).fetchone()
        if slot is None:
            return {"ok": False, "error": "No such slot."}
        if slot["booked"]:
            return {"ok": False, "error": "Slot already taken. Check availability again."}
        n = c.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
        appt_id = f"A{1001 + n}"
        c.execute("UPDATE slots SET booked=1 WHERE doctor_id=? AND start=?", (doctor_id, start))
        c.execute(
            "INSERT INTO appointments VALUES (?,?,?,?,?,?,'booked')",
            (appt_id, doctor_id, start, patient_name, _norm_phone(phone), reason),
        )
    return {"ok": True, "appointment_id": appt_id, "doctor_id": doctor_id, "start": start}


@mcp.tool()
async def find_appointments(phone: str) -> list[dict]:
    """Upcoming appointments for a caller's phone number."""
    await _backend_delay()
    with _db() as c:
        rows = c.execute(
            "SELECT a.id, a.start, a.status, a.patient_name, d.name AS doctor FROM appointments a "
            "JOIN doctors d ON d.id=a.doctor_id WHERE a.phone=? AND a.status='booked' ORDER BY a.start",
            (_norm_phone(phone),),
        ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
async def reschedule_appointment(appointment_id: str, new_start: str) -> dict:
    """Move an appointment to a new free slot with the same doctor. `new_start` is 'YYYY-MM-DD HH:MM'."""
    await _backend_delay()
    with _db() as c:
        appt = c.execute("SELECT * FROM appointments WHERE id=? AND status='booked'", (appointment_id,)).fetchone()
        if appt is None:
            return {"ok": False, "error": "Appointment not found."}
        slot = c.execute(
            "SELECT booked FROM slots WHERE doctor_id=? AND start=?", (appt["doctor_id"], new_start)
        ).fetchone()
        if slot is None or slot["booked"]:
            return {"ok": False, "error": "That slot is not available."}
        c.execute("UPDATE slots SET booked=0 WHERE doctor_id=? AND start=?", (appt["doctor_id"], appt["start"]))
        c.execute("UPDATE slots SET booked=1 WHERE doctor_id=? AND start=?", (appt["doctor_id"], new_start))
        c.execute("UPDATE appointments SET start=? WHERE id=?", (new_start, appointment_id))
    return {"ok": True, "appointment_id": appointment_id, "start": new_start}


@mcp.tool()
async def cancel_appointment(appointment_id: str) -> dict:
    """Cancel an appointment. Confirm with the caller first."""
    await _backend_delay()
    with _db() as c:
        appt = c.execute("SELECT * FROM appointments WHERE id=? AND status='booked'", (appointment_id,)).fetchone()
        if appt is None:
            return {"ok": False, "error": "Appointment not found."}
        c.execute("UPDATE appointments SET status='cancelled' WHERE id=?", (appointment_id,))
        c.execute("UPDATE slots SET booked=0 WHERE doctor_id=? AND start=?", (appt["doctor_id"], appt["start"]))
    return {"ok": True, "appointment_id": appointment_id, "status": "cancelled"}


if __name__ == "__main__":
    import sys

    if "--reseed" in sys.argv or not DB_PATH.exists():
        seed()
    mcp.run(transport="stdio")
