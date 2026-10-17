"""Manual, bounded Groq check; synthetic data and a disposable DB only.

Run from the project root: .venv/bin/python tests/live_smoke.py
Never collected by pytest. Neither secrets nor raw provider errors are printed.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_agent.agent import Agent, MEMORY_FAILED, SAVE_FAILED, UNAVAILABLE
from memory_agent.config import load_settings
from memory_agent.llm import GroqLLM
from memory_agent.memory import MemoryStore


def phase(name, path):
    settings = load_settings().model_copy(update={"db_path": path})
    if not settings.api_key.get_secret_value():
        print(json.dumps({"ok": False, "reason": "missing_key"}))
        return 1
    llm = GroqLLM(settings)
    checks = []
    try:
        with MemoryStore(path) as store:
            agent = Agent(store, llm)

            def turn(message):
                answer = agent.respond("synthetic_check", message)
                if answer in (UNAVAILABLE, MEMORY_FAILED, SAVE_FAILED):
                    row = store.connection.execute("""SELECT event_type,
                        json_extract(payload_json, '$.category'), json_extract(payload_json, '$.status'),
                        json_extract(payload_json, '$.reason')
                        FROM events WHERE event_type IN ('turn.error','memory.error','storage.error')
                        ORDER BY id DESC LIMIT 1""").fetchone()
                    print(json.dumps({"ok": False, "phase": name, "completed": checks,
                                      "error": list(row) if row else ["unknown"]}, ensure_ascii=False))
                    return None
                return answer

            def occupation():
                facts = [f for f in store.facts("synthetic_check") if f.kind == "occupation"]
                assert len(facts) == 1
                return facts[0]

            if name == "seed":
                if turn("меня зовут сергей, я программист") is None:
                    return 1
                assert {f.kind for f in store.facts("synthetic_check")} == {"name", "occupation"}
                assert "программ" in occupation().value.casefold()
                checks.append("two_facts_in_one_message")
                agent.reset("synthetic_check")
                assert agent.history("synthetic_check") == []
                assert "программ" in occupation().value.casefold()
                checks.append("reset_keeps_occupation")
            else:
                assert agent.history("synthetic_check") == []
                original_id = occupation().id
                for message, expected in [("кто я?", "сергей"), ("чем я занимаюсь?", "программ")]:
                    answer = turn(message)
                    if answer is None:
                        return 1
                    assert expected in answer.casefold()
                checks.append("answers_from_profile_after_process_restart")
                if turn("запомни что я программист") is None:
                    return 1
                assert occupation().id == original_id
                checks.append("explicit_remember_without_duplicate")
                for message in ["Теперь я работаю дизайнером", "я работаю дизайнером"]:
                    if turn(message) is None:
                        return 1
                    assert occupation().id == original_id and "дизайн" in occupation().value.casefold()
                checks.append("correction_and_repeat_keep_id")
                if turn("Меня зовут Сергей, а тебя?") is None:
                    return 1
                assert any(f.kind == "name" and "сергей" in f.value.casefold() for f in store.facts("synthetic_check"))
                checks.append("name_beside_return_question")
                for message in ["Пиши коротко", "Отвечай кратко", "Предпочитаю короткие ответы"]:
                    if turn(message) is None:
                        return 1
                    assert any(f.kind == "preference" and any(w in f.value.casefold() for w in ("кратк", "коротк")) for f in store.facts("synthetic_check"))
                checks.append("three_natural_style_requests")
            print(json.dumps({"ok": True, "phase": name, "checks": checks}, ensure_ascii=False))
        return 0
    except Exception:
        # Do not print exception repr, traceback, prompts or provider bodies.
        print(json.dumps({"ok": False, "phase": name, "completed": checks, "reason": "check_failed"}))
        return 1
    finally:
        llm.close()


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--phase":
        return phase(sys.argv[2], Path(sys.argv[3]))
    # Parent owns the temporary directory; two child processes prove restart.
    with tempfile.TemporaryDirectory(prefix="memory-agent-groq-") as directory:
        path = Path(directory) / "synthetic.sqlite3"
        for name in ("seed", "reopen"):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", name, str(path)],
                                    text=True, capture_output=True, timeout=240)
            print(result.stdout.strip(), flush=True)
            if result.returncode:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
