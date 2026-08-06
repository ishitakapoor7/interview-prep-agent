"""Session persistence.

One row per session holding the whole bundle and plan as JSON. They are always
read and written as a unit — nothing queries inside them — so a document row beats
a normalized schema here. WAL is on so a reader never blocks on a writer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict

from app.models import LessonPlan, Module, QuizQuestion, ResearchBundle, SourceDoc


class SessionStore:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        # Readers and the writer coexist instead of taking a whole-file lock.
        self._conn.execute("PRAGMA journal_mode=WAL")

    def init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    bundle_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def save(self, session_id: str, bundle: ResearchBundle, plan: LessonPlan) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions (id, bundle_json, plan_json) "
                "VALUES (?, ?, ?)",
                (session_id, json.dumps(asdict(bundle)), json.dumps(asdict(plan))),
            )
            self._conn.commit()

    def load(self, session_id: str) -> tuple[ResearchBundle, LessonPlan] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT bundle_json, plan_json FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return (
            _bundle_from_dict(json.loads(row["bundle_json"])),
            _plan_from_dict(json.loads(row["plan_json"])),
        )


def _bundle_from_dict(d: dict) -> ResearchBundle:
    return ResearchBundle(
        company=d["company"],
        role=d["role"],
        docs=[SourceDoc(**doc) for doc in d.get("docs", [])],
        gaps=d.get("gaps", []),
        reflection_queries=d.get("reflection_queries", []),
    )


def _plan_from_dict(d: dict) -> LessonPlan:
    return LessonPlan(
        company=d["company"],
        role=d["role"],
        gap_analysis=d.get("gap_analysis", ""),
        modules=[
            Module(
                number=m["number"],
                title=m["title"],
                content=m["content"],
                quiz=[QuizQuestion(**q) for q in m.get("quiz", [])],
            )
            for m in d.get("modules", [])
        ],
    )
