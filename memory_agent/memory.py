"""SQLite persistence. Network calls never hold a database transaction."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schemas import Fact, MemoryPlan, validate_user_id


class StorageError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.connection = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=3, isolation_level=None)
            self.connection.row_factory = sqlite3.Row
            # DELETE journal avoids retaining forgotten personal rows in WAL files.
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA secure_delete=ON")
            self._initialize()
        except (sqlite3.Error, OSError, StorageError):
            self.close()
            raise StorageError("Не удалось открыть локальное хранилище.") from None

    def _initialize(self) -> None:
        # DDL is transactional too. executescript() would implicitly commit and
        # could leave a half-migrated database after an error or interruption.
        with self._transaction():
            schema = self.connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'facts'"
            ).fetchone()
            if schema and "'occupation'" not in schema[0].casefold():
                self._migrate_facts()
            self.connection.execute("""CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('name', 'city', 'occupation', 'preference')),
                    value TEXT NOT NULL CHECK(length(value) BETWEEN 1 AND 500),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )""")
            self.connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_profile_slot
                ON facts(user_id, kind) WHERE kind IN ('name', 'city', 'occupation')""")
            self.connection.execute("CREATE INDEX IF NOT EXISTS facts_by_user ON facts(user_id)")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )""")
            self.connection.execute("CREATE INDEX IF NOT EXISTS events_by_user ON events(user_id)")

    def _migrate_facts(self) -> None:
        objects = self.connection.execute("""SELECT name, sql FROM sqlite_master
            WHERE tbl_name = 'facts' AND type IN ('index', 'trigger') AND sql IS NOT NULL""").fetchall()
        sequence = self.connection.execute("SELECT seq FROM sqlite_sequence WHERE name = 'facts'").fetchone()
        self.connection.execute("""CREATE TABLE facts_migrating (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('name', 'city', 'occupation', 'preference')),
            value TEXT NOT NULL CHECK(length(value) BETWEEN 1 AND 500),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        self.connection.execute("""INSERT INTO facts_migrating(id, user_id, kind, value, created_at, updated_at)
            SELECT id, user_id, kind, value, created_at, updated_at FROM facts""")
        self.connection.execute("DROP TABLE facts")
        self.connection.execute("ALTER TABLE facts_migrating RENAME TO facts")
        if sequence:
            self.connection.execute("UPDATE sqlite_sequence SET seq = max(seq, ?) WHERE name = 'facts'", (sequence[0],))
        for obj in objects:
            if obj["name"] != "one_profile_slot":
                self.connection.execute(obj["sql"])

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _transaction(self):
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield
            self.connection.commit()
        except (sqlite3.Error, StorageError):
            self.connection.rollback()
            raise StorageError("Не удалось сохранить данные. Повторите сообщение.") from None
        except BaseException:
            self.connection.rollback()
            raise

    def facts(self, user_id: str) -> list[Fact]:
        validate_user_id(user_id)
        try:
            rows = self.connection.execute(
                "SELECT id, kind, value, created_at, updated_at FROM facts WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
            return [Fact(**dict(row)) for row in rows]
        except sqlite3.Error:
            raise StorageError("Не удалось прочитать память.") from None

    def _event(self, user_id: str, event_type: str, payload: dict) -> None:
        self.connection.execute(
            "INSERT INTO events(timestamp, user_id, event_type, payload_json) VALUES (?, ?, ?, ?)",
            (utc_now(), user_id, event_type, json.dumps(payload, ensure_ascii=False)),
        )

    def log_best_effort(self, user_id: str, event_type: str, payload: dict) -> bool:
        validate_user_id(user_id)
        try:
            with self._transaction():
                self._event(user_id, event_type, payload)
            return True
        except StorageError:
            return False

    def commit_turn(self, user_id: str, plan: MemoryPlan, request: str, response: str) -> None:
        validate_user_id(user_id)
        with self._transaction():
            # Optimistic snapshot check: a concurrent writer must not be overwritten.
            if tuple(self.facts(user_id)) != plan.before:
                raise StorageError("Память изменилась в другом процессе. Повторите сообщение.")
            originals = {f.id: f for f in plan.before}
            final = {f.id: f for f in plan.after}
            now = utc_now()
            for fact_id in originals.keys() - final.keys():
                self.connection.execute("DELETE FROM facts WHERE id = ? AND user_id = ?", (fact_id, user_id))
            id_map = {}
            for fact in plan.after:
                if fact.id < 0:
                    cur = self.connection.execute(
                        "INSERT INTO facts(user_id, kind, value, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (user_id, fact.kind, fact.value, now, now),
                    )
                    id_map[fact.id] = cur.lastrowid
                elif fact != originals[fact.id]:
                    self.connection.execute(
                        "UPDATE facts SET value = ?, updated_at = ? WHERE id = ? AND user_id = ? AND kind = ?",
                        (fact.value, now, fact.id, user_id, fact.kind),
                    )
            self._event(user_id, "memory.read", {"fact_ids": [f.id for f in plan.before]})
            for op in plan.operations:
                payload = {**op, "fact_id": id_map.get(op["fact_id"], op["fact_id"])}
                self._event(user_id, "memory." + op["action"], payload)
            self._event(user_id, "turn", {"request": request, "response": response})

    def forget(self, user_id: str) -> None:
        validate_user_id(user_id)
        with self._transaction():
            self.connection.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
            self.connection.execute("DELETE FROM events WHERE user_id = ?", (user_id,))
        # Deliberately no new event: /forget leaves no personal record behind.
