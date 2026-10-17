"""Local commands work without a configured Groq key or network access."""

import argparse
import sys
from collections.abc import Callable

from .agent import Agent
from .config import ConfigurationError, load_settings
from .llm import GroqLLM
from .memory import MemoryStore, StorageError
from .schemas import validate_user_id

HELP = """/help        — справка
/memory      — факты активного профиля из SQLite
/reset       — очистить текущую историю; сохранить факты
/forget      — удалить факты, логи и историю активного профиля
/user <id>   — переключить локальный профиль
/exit        — выйти"""


def run_cli(
    agent: Agent, user_id: str, *,
    read: Callable[[str], str] | None = None,
    write: Callable[[str], None] | None = None,
) -> int:
    read = read or input
    write = write or print
    validate_user_id(user_id)
    while True:
        try:
            line = read(f"[{user_id}] > ").strip()
            if not line:
                continue
            if not line.startswith("/"):
                write("Бот: " + agent.respond(user_id, line))
                continue
            parts = line.split()
            command = parts[0]
            if command == "/user":
                if len(parts) != 2:
                    write("Использование: /user <id>")
                    continue
                try:
                    new_user = validate_user_id(parts[1])
                except ValueError as exc:
                    write(str(exc))
                    continue
                user_id = new_user
                write(f"Активный профиль: {user_id}")
            elif command in {"/help", "/memory", "/reset", "/forget", "/exit"}:
                if len(parts) != 1:
                    write(f"Команда {command} не принимает аргументы.")
                    continue
                if command == "/exit":
                    write("До встречи!")
                    return 0
                if command == "/help":
                    write(HELP)
                elif command == "/memory":
                    write(agent.show_memory(user_id))
                elif command == "/reset":
                    write(agent.reset(user_id))
                elif command == "/forget":
                    write(agent.forget(user_id))
            else:
                write("Неизвестная команда. Список команд: /help")
        except EOFError:
            write("\nДо встречи!")
            return 0
        except KeyboardInterrupt:
            write("\nРабота завершена.")
            return 0
        except StorageError:
            write("Операция с БД не выполнена. Проверьте доступ к хранилищу и повторите.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLI-бот с постоянной памятью (Groq + SQLite)")
    parser.add_argument("--user", required=True, help="Локальный профиль, например sergey")
    args = parser.parse_args(argv)
    try:
        validate_user_id(args.user)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        settings = load_settings()
        with MemoryStore(settings.db_path) as store:
            llm = GroqLLM(settings)
            try:
                print(f"База памяти: {settings.db_path}")
                print(f"Модель Groq: {settings.model}")
                print("Команды: /help. Локальные профили доступны без аутентификации.")
                if not settings.api_key.get_secret_value():
                    print("GROQ_API_KEY не задан. Локальные команды доступны; для диалога заполните .env и перезапустите.")
                return run_cli(Agent(store, llm), args.user)
            finally:
                llm.close()
    except (ConfigurationError, StorageError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nРабота завершена.")
        return 0
