"""A turn is staged in RAM, persisted atomically, then added to history."""

from typing import Protocol

from pydantic import ValidationError

from .llm import ServiceUnavailable, bounded_history
from .memory import MemoryStore, StorageError
from .schemas import (
    Extraction, Fact, InvalidMemory, MemoryFull, MAX_MESSAGE_CHARS,
    MAX_REPLY_CHARS, prepare_memory, safe_text, validate_user_id,
)

UNAVAILABLE = "Сервис временно недоступен"
SAVE_FAILED = "Не удалось сохранить данные. Повторите сообщение."
MEMORY_FAILED = "Не удалось проверить изменения памяти. Ничего не сохранено. Повторите факт отдельным утверждением; проверить память можно через /memory."


class LLM(Protocol):
    def extract(self, message: str, facts: list[Fact], history: list[tuple[str, str]], *,
                validation_error: str | None = None) -> Extraction: ...
    def reply(self, message: str, facts: list[Fact], history: list[tuple[str, str]]) -> str: ...


class Agent:
    def __init__(self, store: MemoryStore, llm: LLM):
        self.store = store
        self.llm = llm
        self.histories: dict[str, list[tuple[str, str]]] = {}

    def history(self, user_id: str) -> list[tuple[str, str]]:
        validate_user_id(user_id)
        return list(self.histories.get(user_id, []))

    def respond(self, user_id: str, message: str) -> str:
        validate_user_id(user_id)
        if not message.strip():
            return ""
        if len(message) > MAX_MESSAGE_CHARS:
            return f"Сообщение слишком длинное. Лимит — {MAX_MESSAGE_CHARS} символов."
        if not safe_text(message):
            return "Сообщение содержит недопустимые управляющие символы."
        try:
            facts = self.store.facts(user_id)
            history = self.history(user_id)
            validation_error = None
            for attempt in range(2):
                try:
                    changes = self.llm.extract(message, facts, history, validation_error=validation_error)
                    plan = prepare_memory(facts, changes, message)
                    break
                except MemoryFull:
                    raise
                except (InvalidMemory, ValidationError) as exc:
                    if attempt:
                        raise
                    validation_error = exc.code if isinstance(exc, InvalidMemory) else "wire_schema"
            deleting = any(op["action"] == "delete" for op in plan.operations)
            answer = self.llm.reply(message, list(plan.after), [] if deleting else history)
            if not answer.strip() or len(answer) > MAX_REPLY_CHARS or not safe_text(answer):
                raise ServiceUnavailable("invalid_response")
        except MemoryFull:
            return "Достигнут лимит 50 фактов. Попросите забыть ненужные сведения или выполните /forget."
        except (InvalidMemory, ValidationError) as exc:
            self.store.log_best_effort(user_id, "memory.error", {
                "category": "invalid_memory", "stage": "extraction_validation",
                "reason": exc.code if isinstance(exc, InvalidMemory) else "wire_schema",
                "response": MEMORY_FAILED,
            })
            return MEMORY_FAILED
        except ServiceUnavailable as exc:
            self.store.log_best_effort(user_id, "turn.error", {
                "request": message, "response": UNAVAILABLE,
                "category": exc.category, "status": exc.status,
            })
            return UNAVAILABLE
        except StorageError:
            return "Не удалось прочитать память. Проверьте доступ к БД."
        try:
            self.store.commit_turn(user_id, plan, message, answer)
        except StorageError:
            # Logging is allowed to fail independently; never print an uncommitted answer.
            self.store.log_best_effort(user_id, "storage.error", {
                "request": message, "response": SAVE_FAILED,
            })
            return SAVE_FAILED
        # Even the deletion request may name the removed value. Keep it out of
        # future context as well as clearing the earlier conversation.
        self.histories[user_id] = [] if deleting else bounded_history([*history, (message, answer)])
        return answer

    def show_memory(self, user_id: str) -> str:
        facts = self.store.facts(user_id)
        labels = {"name": "Имя", "city": "Город", "occupation": "Профессия/занятие", "preference": "Предпочтение"}
        result = "\n".join(f"#{f.id} {labels[f.kind]}: {f.value}" for f in facts) or "Сохранённых фактов нет."
        self.store.log_best_effort(user_id, "memory.read", {
            "request": "/memory", "response": result, "fact_ids": [f.id for f in facts],
        })
        return result

    def reset(self, user_id: str) -> str:
        validate_user_id(user_id)
        self.histories.pop(user_id, None)
        result = "История диалога очищена. Сохранённые факты остались."
        self.store.log_best_effort(user_id, "history.reset", {"request": "/reset", "response": result})
        return result

    def forget(self, user_id: str) -> str:
        self.store.forget(user_id)
        self.histories.pop(user_id, None)
        return "Все локальные сведения этого профиля удалены."
