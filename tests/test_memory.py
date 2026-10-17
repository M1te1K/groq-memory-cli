import pytest
from pydantic import ValidationError

from conftest import operation
from memory_agent.agent import MEMORY_FAILED
from memory_agent.schemas import Extraction, Fact, InvalidMemory, MemoryFull, prepare_memory


def test_name_city_corrections_and_normalized_duplicates(agent, llm, store):
    for kind, first, second, phrase in [
        ("name", "Сергей", "Серёжа", "Меня зовут "),
        ("city", "Казань", "Москва", "Я живу в "),
    ]:
        for value in [first, first.lower() + ".", second]:
            llm.operations = [operation(kind, value, phrase + value)]
            assert agent.respond("sergey", phrase + value) == "Готово"
        facts = [f for f in store.facts("sergey") if f.kind == kind]
        assert len(facts) == 1
        assert facts[0].value == second


def test_preference_replacement_repeat_negative_and_delete(agent, llm, store):
    msg = "Я предпочитаю чай"
    llm.operations = [operation("preference", "Предпочитает чай", msg)]
    agent.respond("sergey", msg)
    agent.respond("sergey", msg)
    assert len(store.facts("sergey")) == 1
    fact = store.facts("sergey")[0]
    msg = "Теперь предпочитаю кофе вместо чая"
    llm.operations = [operation("preference", "Предпочитает кофе", msg, "update", fact.id)]
    assert agent.respond("sergey", msg) == "Готово"
    assert [f.value for f in store.facts("sergey")] == ["Предпочитает кофе"]
    assert store.facts("sergey")[0].id == fact.id
    msg = "Я не люблю кофе"
    llm.operations = [operation("preference", "Не любит кофе", msg, "update", fact.id)]
    agent.respond("sergey", msg)
    assert store.facts("sergey")[0].value == "Не любит кофе"
    msg = "Забудь предпочтение кофе"
    llm.operations = [operation("preference", None, msg, "delete", fact.id)]
    assert agent.respond("sergey", msg) == "Готово"
    assert store.facts("sergey") == []
    types = [r[0] for r in store.connection.execute("SELECT event_type FROM events")]
    assert "memory.update" in types and "memory.delete" in types


@pytest.mark.parametrize("message,evidence,kind,value", [
    ("Привет", "Меня зовут Сергей", "name", "Сергей"),
    ("Мой брат живёт в Омске", "Мой брат живёт в Омске", "city", "Омск"),
    ("Я сейчас в Москве", "Я сейчас в Москве", "city", "Москва"),
    ("Представь, что меня зовут Анна", "меня зовут Анна", "name", "Анна"),
    ('Он сказал: «Меня зовут Иван»', "Меня зовут Иван", "name", "Иван"),
    ('Переведи: Меня зовут Анна', "Меня зовут Анна", "name", "Анна"),
    ("Как меня зовут?", "меня зовут", "name", "Сергей"),
])
def test_bad_evidence_is_rejected(agent, llm, store, message, evidence, kind, value):
    llm.operations = [operation(kind, value, evidence)]
    assert agent.respond("sergey", message) == MEMORY_FAILED
    assert store.facts("sergey") == []
    assert agent.history("sergey") == []
    assert not any(c[0] == "reply" for c in llm.calls)


@pytest.mark.parametrize("patch", [
    {"action": "execute_sql"}, {"fact_id": 1}, {"value": None}, {"kind": "age"},
    {"user_id": "anna"}, {"sql": "DROP TABLE facts"}, {"value": "x" * 501},
    {"value": "\x1b[31m"}, {"value": " "}, {"evidence": ""}, {"fact_id": True},
])
def test_wire_schema_rejects_invalid_operations(patch):
    with pytest.raises(ValidationError):
        Extraction.model_validate({"operations": [{**operation(), **patch}]})


def test_application_checks_missing_id_kind_and_duplicate_targets(seed, store):
    fact = seed(store)
    for operations in [
        [operation(action="update", fact_id=fact.id + 100)],
        [operation(kind="city", value="Омск", evidence="Я живу в Омске", action="update", fact_id=fact.id)],
        [operation(action="update", fact_id=fact.id)] * 2,
        [operation(), operation(value="Иван", evidence="Меня зовут Иван")],
    ]:
        with pytest.raises(InvalidMemory):
            prepare_memory(store.facts("sergey"), Extraction.model_validate({"operations": operations}),
                           "Меня зовут Сергей. Я живу в Омске. Меня зовут Иван")


def test_fact_limit_never_evicts_existing_memory():
    facts = [Fact(i + 1, "preference", f"Любит {i}") for i in range(50)]
    changes = Extraction.model_validate({"operations": [operation("preference", "Любит чай", "Я люблю чай")]})
    with pytest.raises(MemoryFull):
        prepare_memory(facts, changes, "Я люблю чай")
    assert len(facts) == 50
