import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory_agent.agent import Agent, UNAVAILABLE
from memory_agent.cli import run_cli
from memory_agent.config import ConfigurationError, load_settings
from memory_agent.llm import GroqLLM, ServiceUnavailable


def play(agent, lines, user="sergey"):
    iterator = iter(lines)
    prompts, output = [], []

    def read(prompt):
        prompts.append(prompt)
        value = next(iterator)
        if isinstance(value, BaseException):
            raise value
        return value

    assert run_cli(agent, user, read=read, write=output.append) == 0
    return prompts, output


def test_commands_no_key_empty_unknown_invalid_profiles(store, tmp_path):
    settings = load_settings(tmp_path, environ={})
    adapter = GroqLLM(settings)
    assert adapter.client is None
    agent = Agent(store, adapter)
    prompts, output = play(agent, ["", "/help", "Привет", "/memory", "/reset", "/user", "/user ../x",
                                  "/user anna", "/unknown", "/memory extra", "/forget", "/exit"])
    assert any(UNAVAILABLE in line for line in output)
    assert any("Сохранённых фактов нет" in line for line in output)
    assert "[anna] > " in prompts
    assert any("Неизвестная" in line for line in output)
    assert store.facts("sergey") == [] and store.facts("anna") == []
    assert store.connection.execute("SELECT count(*) FROM events WHERE user_id = ?", ("anna",)).fetchone()[0] == 0


@pytest.mark.parametrize("end", [EOFError(), KeyboardInterrupt()])
def test_eof_and_ctrl_c(agent, end):
    _, output = play(agent, [end])
    assert "До встречи" in output[-1] or "завершена" in output[-1]


def test_cli_continues_after_forget_storage_failure(agent, store, seed):
    seed(store)
    store.connection.execute("CREATE TRIGGER no_delete BEFORE DELETE ON facts BEGIN SELECT RAISE(ABORT, 'failure'); END")
    _, output = play(agent, ["/forget", "/memory", "/exit"])
    assert "не выполнена" in output[0]
    assert "Сергей" in output[1]
    assert not any("удалены" in line for line in output)


def test_config_uses_stable_root_preserves_env_and_environment_overrides(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    content = "GROQ_API_KEY=secret-test\nMEMORY_DB_PATH=./data/test.sqlite3\nGROQ_TIMEOUT_SECONDS=15\n"
    envfile = root / ".env"
    envfile.write_text(content)
    monkeypatch.chdir(tmp_path)
    settings = load_settings(root, environ={})
    assert settings.db_path == root / "data" / "test.sqlite3"
    assert settings.timeout == 15
    assert "secret-test" not in repr(settings)
    override = load_settings(root, environ={"GROQ_API_KEY": "", "GROQ_TIMEOUT_SECONDS": "2"})
    assert override.api_key.get_secret_value() == "" and override.timeout == 2
    assert envfile.read_text() == content


@pytest.mark.parametrize("value", ["NaN", "inf", "-1", "0", "121", "secret-input"])
def test_bad_config_does_not_leak_values(tmp_path, value):
    with pytest.raises(ConfigurationError) as error:
        load_settings(tmp_path, environ={"GROQ_TIMEOUT_SECONDS": value, "GROQ_API_KEY": "secret-key"})
    assert "secret" not in str(error.value)


def test_missing_key_never_constructs_sdk(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("SDK should not be created for missing key")
    monkeypatch.setattr("memory_agent.llm.groq.Groq", forbidden)
    adapter = GroqLLM(load_settings(tmp_path, environ={}))
    with pytest.raises(ServiceUnavailable, match=UNAVAILABLE):
        adapter.extract("Привет", [], [])


def test_real_process_eof_and_invalid_user(tmp_path):
    env = {**os.environ, "GROQ_API_KEY": "", "MEMORY_DB_PATH": str(tmp_path / "cli.sqlite3")}
    result = subprocess.run([sys.executable, "-m", "memory_agent", "--user", "sergey"],
                            input="", env=env, text=True, capture_output=True)
    assert result.returncode == 0 and "До встречи" in result.stdout
    invalid = subprocess.run([sys.executable, "-m", "memory_agent", "--user", "../anna"],
                             input="", env=env, text=True, capture_output=True)
    assert invalid.returncode == 2 and "user_id" in invalid.stderr


def test_source_launcher_from_another_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "GROQ_API_KEY": "", "MEMORY_DB_PATH": str(tmp_path / "other-cwd.sqlite3")}
    result = subprocess.run([sys.executable, str(root / "run.py"), "--user", "sergey"], cwd=tmp_path,
                            input="/memory\n/exit\n", env=env, text=True, capture_output=True, check=True)
    assert "Сохранённых фактов нет" in result.stdout
    assert str(tmp_path / "other-cwd.sqlite3") in result.stdout
