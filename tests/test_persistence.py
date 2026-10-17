import json
import os
import subprocess
import sys

from memory_agent.agent import Agent
from memory_agent.memory import MemoryStore


def test_reopen_and_real_new_process(tmp_path, seed, llm):
    path = tmp_path / "persistent.sqlite3"
    with MemoryStore(path) as first:
        seed(first)
    with MemoryStore(path) as reopened:
        assert reopened.facts("sergey")[0].value == "Сергей"
        assert Agent(reopened, llm).history("sergey") == []
    script = """
import json, sys
from pathlib import Path
from memory_agent.memory import MemoryStore
from memory_agent.agent import Agent
with MemoryStore(Path(sys.argv[1])) as store:
    print(json.dumps([f.as_data() for f in store.facts('sergey')], ensure_ascii=False))
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)[0]["value"] == "Сергей"
    env = {**os.environ, "MEMORY_DB_PATH": str(path), "GROQ_API_KEY": ""}
    cli = subprocess.run([sys.executable, "-m", "memory_agent", "--user", "sergey"],
                         input="/memory\n/exit\n", env=env, capture_output=True, text=True, check=True)
    assert "Имя: Сергей" in cli.stdout


def test_turn_logs_are_atomic_and_include_operations(agent, store, llm):
    llm.operations = [{"action": "add", "fact_id": None, "kind": "name", "value": "Сергей", "evidence": "Меня зовут Сергей"}]
    assert agent.respond("sergey", "Меня зовут Сергей") == "Готово"
    rows = store.connection.execute("SELECT * FROM events WHERE user_id = ? ORDER BY id", ("sergey",)).fetchall()
    assert [r["event_type"] for r in rows] == ["memory.read", "memory.add", "turn"]
    assert all(r["timestamp"].endswith("+00:00") for r in rows)
    turn = json.loads(rows[-1]["payload_json"])
    assert turn == {"request": "Меня зовут Сергей", "response": "Готово"}
    assert json.loads(rows[1]["payload_json"])["fact_id"] == store.facts("sergey")[0].id
