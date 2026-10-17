import pytest

from conftest import operation
from memory_agent.agent import MEMORY_FAILED


def test_two_users_and_foreign_ids(agent, llm, store, seed):
    own = seed(store, "sergey")
    foreign = seed(store, "anna", "Анна")
    llm.operations = []
    agent.respond("anna", "Привет от Анны")
    agent.respond("sergey", "Привет от Сергея")
    assert agent.history("anna") == [("Привет от Анны", "Готово")]
    sergey_calls = llm.calls[-2:]
    assert all([f.value for f in c[2]] == ["Сергей"] and c[3] == [] for c in sergey_calls)
    llm.operations = [operation(value="Иван", evidence="Меня зовут Иван", action="update", fact_id=foreign.id)]
    assert agent.respond("sergey", "Меня зовут Иван") == MEMORY_FAILED
    llm.operations = [operation(value=None, evidence="Забудь имя", action="delete", fact_id=foreign.id)]
    assert agent.respond("sergey", "Забудь имя") == MEMORY_FAILED
    assert store.facts("sergey") == [own]
    assert store.facts("anna") == [foreign]


def test_reset_preserves_facts_and_other_history(agent, store, seed):
    seed(store)
    agent.respond("sergey", "Привет")
    agent.respond("anna", "Привет, я Анна")
    other = agent.history("anna")
    assert "Сохранённые факты остались" in agent.reset("sergey")
    assert agent.history("sergey") == []
    assert agent.history("anna") == other
    assert store.facts("sergey")[0].value == "Сергей"


def test_forget_removes_all_personal_rows_and_context(agent, store, seed, llm):
    seed(store)
    seed(store, "anna", "Анна")
    agent.respond("sergey", "Привет")
    agent.respond("anna", "Добрый день")
    other_facts = store.facts("anna")
    other_history = agent.history("anna")
    other_events = store.connection.execute("SELECT * FROM events WHERE user_id = ?", ("anna",)).fetchall()
    assert "удалены" in agent.forget("sergey")
    assert store.facts("sergey") == []
    assert agent.history("sergey") == []
    assert store.connection.execute("SELECT count(*) FROM events WHERE user_id = ?", ("sergey",)).fetchone()[0] == 0
    assert store.facts("anna") == other_facts
    assert agent.history("anna") == other_history
    assert store.connection.execute("SELECT * FROM events WHERE user_id = ?", ("anna",)).fetchall() == other_events
    agent.respond("sergey", "Как меня зовут?")
    assert all(c[2] == [] and c[3] == [] for c in llm.calls[-2:])


def test_prompt_history_is_bounded_in_complete_pairs(agent, llm):
    for i in range(15):
        agent.respond("sergey", str(i))
    assert len(agent.history("sergey")) == 10
    assert agent.history("sergey")[0][0] == "5"
    llm.answer = "a" * 7_000
    for _ in range(5):
        agent.respond("sergey", "u" * 7_000)
    history = agent.history("sergey")
    assert len(history) == 1
    assert sum(len(u) + len(a) for u, a in history) <= 24_000


def test_individual_deletion_does_not_return_fact_through_history(agent, llm, store, seed):
    fact = seed(store)
    agent.respond("sergey", "Я Сергей")
    agent.respond("anna", "Привет")
    other_history = agent.history("anna")
    llm.operations = [operation(value=None, evidence="Забудь имя Сергей", action="delete", fact_id=fact.id)]
    assert agent.respond("sergey", "Забудь имя Сергей") == "Готово"
    assert llm.calls[-1][2] == [] and llm.calls[-1][3] == []
    assert agent.history("sergey") == [] and agent.history("anna") == other_history
    llm.operations = []
    agent.respond("sergey", "Как меня зовут?")
    assert llm.calls[-1][2] == [] and llm.calls[-1][3] == []


@pytest.mark.parametrize("user", ["", "../anna", "anna\n", "a" * 65, "Сергей", "-x", "a b", "x';DROP"])
def test_invalid_ids_rejected_everywhere(agent, store, user):
    for call in [store.facts, store.forget, agent.reset, agent.show_memory, agent.history]:
        with pytest.raises(ValueError):
            call(user)
