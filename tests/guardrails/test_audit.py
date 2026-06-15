from app.config import get_settings
from app.db.connection import connect, init_db
from app.guardrails.audit import record_events
from app.guardrails.types import GuardFinding, GuardVerdict


async def test_record_writes_one_row_per_finding(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    get_settings.cache_clear()
    await init_db()

    verdict = GuardVerdict(
        action="block",
        findings=[
            GuardFinding("harmful_intent", 0.9, "stalking"),
            GuardFinding("minor_target", 0.8, "minor"),
        ],
        reason="harmful_intent, minor_target",
    )
    await record_events(verdict, point="input", snippet_source="bad query", thread_id="t1")

    async with connect() as conn:
        rows = await (
            await conn.execute("SELECT category, decision FROM guard_events")
        ).fetchall()
    assert len(rows) == 2
    assert {r["decision"] for r in rows} == {"block"}
    get_settings.cache_clear()


async def test_no_findings_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t2.db"))
    get_settings.cache_clear()
    await init_db()
    await record_events(GuardVerdict(action="allow"), point="input", snippet_source="ok")
    async with connect() as conn:
        rows = await (await conn.execute("SELECT * FROM guard_events")).fetchall()
    assert rows == []
    get_settings.cache_clear()
