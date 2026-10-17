import sqlite3

import pytest

from conftest import operation
from memory_agent.agent import SAVE_FAILED, UNAVAILABLE
from memory_agent.llm import ServiceUnavailable
from memory_agent.memory import StorageError


@pytest.mark.parametrize("failure", [ServiceUnavailable("timeout"), ServiceUnavailable("api_error", 401),
                                    ServiceUnavailable("api_error", 429), ServiceUnavailable("api_error", 503)])
@pytest.mark.parametrize("stage", ["extract", "reply"])
def test_service_failure_preserves_existing_state(agent, store, seed, llm, failure, stage):
    seed(store)
    agent.respond("sergey", "Привет")
    facts, history = store.facts("sergey"), agent.history("sergey")
    llm.operations = [operation(value="Иван", evidence="Меня зовут Иван")]
    if stage == "extract":
        llm.failure = failure
    else:
        llm.reply_failure = failure
    assert agent.respond("sergey", "Меня зовут Иван") == UNAVAILABLE
    assert store.facts("sergey") == facts
    assert agent.history("sergey") == history
    assert "Сергей" in agent.show_memory("sergey")


def test_failed_event_insert_rolls_back_facts_and_partial_events(agent, llm, store, seed):
    seed(store)
    before = store.facts("sergey")
    llm.operations = [operation(value="Иван", evidence="Меня зовут Иван")]
    store.connection.execute("""CREATE TRIGGER fail_turn BEFORE INSERT ON events
        WHEN NEW.event_type = 'turn' BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
    assert agent.respond("sergey", "Меня зовут Иван") == SAVE_FAILED
    assert store.facts("sergey") == before
    assert agent.history("sergey") == []
    assert store.connection.execute("SELECT count(*) FROM events WHERE event_type = 'memory.update'").fetchone()[0] == 0
    assert "Сергей" in agent.show_memory("sergey")


class CommitFailureProxy:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def commit(self):
        raise sqlite3.OperationalError("secret raw error should not escape")


def test_actual_commit_error_is_rolled_back(agent, llm, store):
    store.connection = CommitFailureProxy(store.connection)
    llm.operations = [operation()]
    assert agent.respond("sergey", "Меня зовут Сергей") == SAVE_FAILED
    assert store.facts("sergey") == []
    assert agent.history("sergey") == []
    assert store.connection.execute("SELECT count(*) FROM events").fetchone()[0] == 0


def test_forget_failure_has_no_false_confirmation_and_keeps_history(agent, store, seed):
    seed(store)
    agent.respond("sergey", "Привет")
    history = agent.history("sergey")
    store.connection.execute("""CREATE TRIGGER fail_forget BEFORE DELETE ON events
        BEGIN SELECT RAISE(ABORT, 'failure'); END""")
    with pytest.raises(StorageError):
        agent.forget("sergey")
    assert store.facts("sergey")[0].value == "Сергей"
    assert agent.history("sergey") == history


def test_best_effort_logging_does_not_break_local_commands(agent, store, seed):
    seed(store)
    agent.respond("sergey", "Привет")
    store.connection.execute("CREATE TRIGGER no_logs BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'failure'); END")
    assert "Сергей" in agent.show_memory("sergey")
    assert "очищена" in agent.reset("sergey")
    assert agent.history("sergey") == []


def test_stale_snapshot_cannot_overwrite_another_process(store, seed):
    from memory_agent.schemas import Extraction, prepare_memory
    from memory_agent.memory import MemoryStore
    seed(store)
    changes = Extraction.model_validate({"operations": [operation(value="Иван", evidence="Меня зовут Иван")]})
    plan = prepare_memory(store.facts("sergey"), changes, "Меня зовут Иван")
    with MemoryStore(store.path) as other:
        seed(other, value="Пётр")
    with pytest.raises(StorageError):
        store.commit_turn("sergey", plan, "Меня зовут Иван", "Запомнил")
    assert store.facts("sergey")[0].value == "Пётр"


def test_message_limits_are_local_and_do_not_call_llm(agent, llm):
    assert agent.respond("sergey", " " * 100) == ""
    assert "8000" in agent.respond("sergey", "a" * 8001)
    assert "управляющие" in agent.respond("sergey", "hello\x1b")
    assert llm.calls == []
