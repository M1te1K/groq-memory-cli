import pytest

from conftest import operation
from memory_agent.agent import Agent, MEMORY_FAILED
from memory_agent.memory import MemoryStore


def test_name_assertion_beside_return_question(agent, llm, store):
    llm.operations = [operation(evidence="Меня зовут Сергей")]
    assert agent.respond("you", "Меня зовут Сергей, а тебя?") == "Готово"
    assert [(f.kind, f.value) for f in store.facts("you")] == [("name", "Сергей")]
    assert [call[0] for call in llm.calls] == ["extract", "reply"]


@pytest.mark.parametrize("message", ["Пиши коротко", "Отвечай кратко", "Предпочитаю короткие ответы"])
def test_natural_reply_style_is_persisted(agent, llm, store, message):
    llm.operations = [operation("preference", "Предпочитает краткие ответы", message)]
    assert agent.respond("you", message) == "Готово"
    assert [f.value for f in store.facts("you")] == ["Предпочитает краткие ответы"]


def test_screenshot_scenario_reset_reopen_and_answer_context(tmp_path, llm):
    path = tmp_path / "screenshot.sqlite3"
    with MemoryStore(path) as store:
        agent = Agent(store, llm)
        llm.operations = [operation(value="Сергей", evidence="меня зовут сергей"),
                          operation("occupation", "Программист", "я программист")]
        assert agent.respond("you", "меня зовут сергей, я программист") == "Готово"
        facts = store.facts("you")
        assert [(f.kind, f.value) for f in facts] == [("name", "Сергей"), ("occupation", "Программист")]
        assert "Профессия/занятие: Программист" in agent.show_memory("you")
        agent.reset("you")
        assert agent.history("you") == [] and store.facts("you") == facts
    with MemoryStore(path) as store:
        agent = Agent(store, llm)
        llm.operations = []
        for question in ["кто я?", "чем я занимаюсь?"]:
            assert agent.respond("you", question) == "Готово"
            assert [(f.kind, f.value) for f in llm.calls[-1][2]] == [("name", "Сергей"), ("occupation", "Программист")]
        assert llm.calls[-3][3] == []  # first reply after reopening has no RAM history


@pytest.mark.parametrize("message", ["я программист", "я работаю программистом", "запомни что я программист"])
def test_occupation_explicit_request_correction_and_repeat(agent, llm, store, message):
    llm.operations = [operation("occupation", "Программист", message)]
    assert agent.respond("you", message) == "Готово"
    original = store.facts("you")[0]
    assert agent.respond("you", message) == "Готово"
    assert store.facts("you") == [original]
    llm.operations = [operation("occupation", "Дизайнер", "я дизайнер", "update", original.id)]
    assert agent.respond("you", "Теперь я дизайнер") == "Готово"
    assert [(f.id, f.value) for f in store.facts("you")] == [(original.id, "Дизайнер")]
    llm.operations = [operation("occupation", "Учитель", "я учитель")]
    assert agent.respond("you", "Теперь я учитель") == "Готово"
    assert [(f.id, f.value) for f in store.facts("you")] == [(original.id, "Учитель")]


@pytest.mark.parametrize("message,evidence", [
    ("Мой брат программист", "Мой брат программист"),
    ("Представь, что я программист", "я программист"),
    ("Если я программист, что дальше", "я программист"),
    ("Наверное, я программист", "я программист"),
    ("Предположим, я программист", "я программист"),
    ("Я программист?", "Я программист"),
    ('Он представился: «Я программист»', "Я программист"),
    ("Он представился: 'Я программист'", "Я программист"),
    ('Он представился: "Я программист"', 'Он представился: "Я программист"'),
    ("Если бы я программист, а ты?", "я программист"),
    ("Я думаю, мой брат программист", "Я думаю, мой брат программист"),
    ("Я хочу работать программистом", "Я хочу работать программистом"),
    ("Я не программист", "Я не программист"),
])
def test_occupation_rejects_non_personal_assertions(agent, llm, store, message, evidence):
    llm.operations = [operation("occupation", "Программист", evidence)]
    assert agent.respond("you", message) == MEMORY_FAILED
    assert store.facts("you") == [] and agent.history("you") == []
    assert [call[0] for call in llm.calls] == ["extract", "extract"]


@pytest.mark.parametrize("message,evidence", [
    ('Он сказал «Меня зовут Сергей». Меня зовут Сергей, а тебя?', "Меня зовут Сергей"),
    ("Меня зовут Сергей, а тебя?", "Меня зовут Сергей, а тебя?"),
])
def test_assertion_can_be_found_outside_quotes_and_before_question(agent, llm, store, message, evidence):
    llm.operations = [operation(evidence=evidence)]
    assert agent.respond("you", message) == "Готово"
    assert store.facts("you")[0].value == "Сергей"


def test_explicit_memory_request_cannot_silently_drop_fact(agent, llm, store):
    llm.answer = "Запомнил, ты программист"
    assert agent.respond("you", "запомни что я программист") == MEMORY_FAILED
    assert store.facts("you") == [] and agent.history("you") == []
    assert [call[0] for call in llm.calls] == ["extract", "extract"]


@pytest.mark.parametrize("bad_operation", [
    operation("occupation", "Программист", "нет в сообщении"),
    {**operation(), "kind": "unknown", "value": "private-key-in-validator-input"},
])
def test_memory_validation_error_is_separate_atomic_and_safe(agent, llm, store, seed, bad_operation):
    seed(store, "other", "Анна")
    agent.respond("other", "Привет")
    agent.respond("you", "Привет")
    other_facts, other_history = store.facts("other"), agent.history("other")
    other_events = store.connection.execute("SELECT * FROM events WHERE user_id='other'").fetchall()
    own_history = agent.history("you")
    llm.operations = [operation(), bad_operation]
    assert agent.respond("you", "Меня зовут Сергей") == MEMORY_FAILED
    assert store.facts("you") == [] and agent.history("you") == own_history
    assert store.facts("other") == other_facts and agent.history("other") == other_history
    assert store.connection.execute("SELECT * FROM events WHERE user_id='other'").fetchall() == other_events
    errors = store.connection.execute("SELECT payload_json FROM events WHERE event_type='memory.error'").fetchall()
    assert len(errors) == 1 and "invalid_memory" in errors[0][0]
    assert "private-key" not in errors[0][0]
    assert store.connection.execute("SELECT count(*) FROM events WHERE event_type='memory.add' AND user_id='you'").fetchone()[0] == 0
