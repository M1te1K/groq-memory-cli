import copy

import pytest

from memory_agent.agent import Agent
from memory_agent.memory import MemoryStore
from memory_agent.schemas import Extraction, prepare_memory


def operation(kind="name", value="Сергей", evidence="Меня зовут Сергей", action="add", fact_id=None):
    return {"action": action, "fact_id": fact_id, "kind": kind, "value": value, "evidence": evidence}


class FakeLLM:
    def __init__(self):
        self.operations = []
        self.answer = "Готово"
        self.calls = []
        self.failure = None
        self.reply_failure = None
        self.validation_feedback = []

    def extract(self, message, facts, history, *, validation_error=None):
        self.validation_feedback.append(validation_error)
        self.calls.append(("extract", message, copy.deepcopy(facts), copy.deepcopy(history)))
        if self.failure:
            raise self.failure
        return Extraction.model_validate({"operations": self.operations})

    def reply(self, message, facts, history):
        self.calls.append(("reply", message, copy.deepcopy(facts), copy.deepcopy(history)))
        if self.reply_failure:
            raise self.reply_failure
        return self.answer


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as value:
        yield value


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def agent(store, llm):
    return Agent(store, llm)


@pytest.fixture
def seed():
    def insert(store, user="sergey", value="Сергей", kind="name", evidence=None):
        if evidence is None:
            evidence = {"name": "Меня зовут ", "city": "Я живу в ", "occupation": "Я ", "preference": "Я люблю "}[kind] + value
        changes = Extraction.model_validate({"operations": [operation(kind, value, evidence)]})
        plan = prepare_memory(store.facts(user), changes, evidence)
        store.commit_turn(user, plan, evidence, "Запомнил")
        return store.facts(user)[-1]
    return insert
