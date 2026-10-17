import sqlite3

import pytest

from conftest import operation
from memory_agent.agent import Agent
from memory_agent.memory import MemoryStore, StorageError


# Exact original schema, deliberately independent of the current implementation.
OLD_SCHEMA = """
CREATE TABLE facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('name', 'city', 'preference')),
    value TEXT NOT NULL CHECK(length(value) BETWEEN 1 AND 500),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX one_profile_slot ON facts(user_id, kind) WHERE kind IN ('name', 'city');
CREATE INDEX facts_by_user ON facts(user_id);
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX events_by_user ON events(user_id);
"""


def make_old_db(path):
    with sqlite3.connect(path) as c:
        c.executescript(OLD_SCHEMA)
        c.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?)", [
            (3, "you", "name", "Сергей", "created", "updated"),
            (7, "you", "city", "Казань", "created", "updated"),
            (11, "you", "preference", "Любит чай", "created", "updated"),
            (19, "other", "name", "Анна", "created", "updated"),
            (100, "deleted", "name", "Removed", "created", "updated"),
        ])
        c.execute("DELETE FROM facts WHERE id = 100")
        c.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", [
            (21, "time", "you", "turn", '{"request":"old request","response":"old response"}'),
            (40, "time", "other", "turn", '{"request":"other request"}'),
        ])
        return c.execute("SELECT * FROM facts ORDER BY id").fetchall(), c.execute("SELECT * FROM events ORDER BY id").fetchall()


def test_exact_legacy_schema_migrates_preserving_rows_ids_indexes_and_isolation(tmp_path, llm):
    path = tmp_path / "old.sqlite3"
    facts, events = make_old_db(path)
    with MemoryStore(path) as store:
        assert [tuple(r) for r in store.connection.execute("SELECT * FROM facts ORDER BY id")] == facts
        assert [tuple(r) for r in store.connection.execute("SELECT * FROM events ORDER BY id")] == events
        indexes = {r[1] for r in store.connection.execute("PRAGMA index_list(facts)")}
        assert indexes == {"one_profile_slot", "facts_by_user"}
        assert store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        llm.operations = [operation("occupation", "Программист", "я программист")]
        assert Agent(store, llm).respond("you", "я программист") == "Готово"
        occupation = store.facts("you")[-1]
        assert occupation.id > 100
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute("INSERT INTO facts(user_id,kind,value,created_at,updated_at) VALUES ('you','occupation','Учитель','x','x')")
        all_facts = store.connection.execute("SELECT * FROM facts ORDER BY id").fetchall()
        all_events = store.connection.execute("SELECT * FROM events ORDER BY id").fetchall()
    with MemoryStore(path) as store:
        assert store.connection.execute("SELECT * FROM facts ORDER BY id").fetchall() == all_facts
        assert store.connection.execute("SELECT * FROM events ORDER BY id").fetchall() == all_events
        agent = Agent(store, llm)
        llm.operations = []
        agent.respond("other", "Привет")
        other_facts, other_history = store.facts("other"), agent.history("other")
        other_events = store.connection.execute("SELECT * FROM events WHERE user_id='other'").fetchall()
        agent.forget("you")
        assert store.facts("you") == [] and agent.history("you") == []
        assert store.connection.execute("SELECT count(*) FROM events WHERE user_id='you'").fetchone()[0] == 0
        assert store.facts("other") == other_facts and agent.history("other") == other_history
        assert store.connection.execute("SELECT * FROM events WHERE user_id='other'").fetchall() == other_events


def test_migration_preserves_extra_index_trigger_and_empty_table_sequence(tmp_path, seed):
    path = tmp_path / "empty.sqlite3"
    make_old_db(path)
    with sqlite3.connect(path) as c:
        c.execute("DELETE FROM facts")
        c.execute("CREATE INDEX custom_fact_value ON facts(value)")
        c.execute("CREATE TRIGGER custom_fact_guard BEFORE INSERT ON facts WHEN NEW.value='blocked' BEGIN SELECT RAISE(ABORT,'blocked'); END")
    with MemoryStore(path) as store:
        assert seed(store).id > 100
        assert store.connection.execute("SELECT name FROM sqlite_master WHERE name='custom_fact_value'").fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute("INSERT INTO facts(user_id,kind,value,created_at,updated_at) VALUES ('other','name','blocked','x','x')")


def test_migration_failure_rolls_back_schema_data_indexes_and_events(tmp_path, monkeypatch):
    path = tmp_path / "rollback.sqlite3"
    facts, events = make_old_db(path)
    with sqlite3.connect(path) as c:
        before_schema = c.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
        before_sequence = c.execute("SELECT * FROM sqlite_sequence").fetchall()
    original = MemoryStore._migrate_facts

    def fail_after_rebuild(self):
        original(self)
        raise sqlite3.OperationalError("synthetic failure")

    monkeypatch.setattr(MemoryStore, "_migrate_facts", fail_after_rebuild)
    with pytest.raises(StorageError):
        MemoryStore(path)
    with sqlite3.connect(path) as c:
        assert c.execute("SELECT * FROM facts ORDER BY id").fetchall() == facts
        assert c.execute("SELECT * FROM events ORDER BY id").fetchall() == events
        assert c.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall() == before_schema
        assert c.execute("SELECT * FROM sqlite_sequence").fetchall() == before_sequence
