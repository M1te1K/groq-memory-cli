"""Wire schema plus application-level ownership, evidence and limit checks."""

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_MESSAGE_CHARS = 8_000
MAX_REPLY_CHARS = 8_000
MAX_FACT_CHARS = 500
MAX_FACTS = 50
MAX_HISTORY_PAIRS = 10
MAX_HISTORY_CHARS = 24_000
Kind = Literal["name", "city", "occupation", "preference"]
SINGLETON_KINDS = ("name", "city", "occupation")
QUOTES = re.compile(r"""«[^»]*»|"[^"\n]*"|“[^”]*”|`[^`]*`|‘[^’]*’|'[^'\n]*'""", re.S)


class InvalidMemory(ValueError):
    def __init__(self, code: str = "invalid_memory"):
        # Codes only, never interpolate model output or Pydantic error inputs.
        self.code = code if code in {
            "evidence_source", "evidence_missing", "personal_assertion", "value_missing",
            "occupation_subject", "occupation_not_current", "deletion_request",
            "empty_explicit_request", "conflicting_operations", "fact_ownership",
            "duplicate_target", "wire_schema",
        } else "invalid_memory"
        super().__init__("Не удалось проверить изменения памяти.")


class MemoryFull(InvalidMemory):
    pass


def validate_user_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("user_id: 1–64 латинских букв, цифр, _, . или -; начало — буква/цифра.")
    return value


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).rstrip(".! ")


def safe_text(value: str) -> bool:
    return all(not unicodedata.category(c).startswith("C") or c in "\n\t" for c in value)


@dataclass(frozen=True)
class Fact:
    id: int
    kind: Kind
    value: str
    created_at: str = ""
    updated_at: str = ""

    def as_data(self) -> dict:
        return {"id": self.id, "kind": self.kind, "value": self.value}


class Operation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    action: Literal["add", "update", "delete"]
    fact_id: int | None
    kind: Kind
    value: str | None = Field(max_length=MAX_FACT_CHARS)
    evidence: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("value")
    @classmethod
    def clean_value(cls, value: str | None) -> str | None:
        if value is not None:
            value = " ".join(value.split()) if safe_text(value) else ""
            if not value or not normalized(value):
                raise ValueError("Empty or unsafe fact")
        return value

    @model_validator(mode="after")
    def valid_action(self):
        if self.action == "add" and self.fact_id is not None:
            raise ValueError("add requires a null ID")
        if self.action != "add" and (self.fact_id is None or self.fact_id <= 0):
            raise ValueError("Existing positive ID required")
        if (self.action == "delete") != (self.value is None):
            raise ValueError("Only delete requires a null value")
        return self


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    operations: list[Operation] = Field(max_length=MAX_FACTS)


def extraction_schema() -> dict:
    """Keep portable strict JSON Schema constraints; enforce lengths locally."""
    schema = Extraction.model_json_schema()

    def simplify(node):
        if isinstance(node, dict):
            return {k: simplify(v) for k, v in node.items() if k not in {
                "title", "minLength", "maxLength", "minItems", "maxItems"
            }}
        if isinstance(node, list):
            return [simplify(x) for x in node]
        return node

    return simplify(schema)


def assertion_source(evidence: str, message: str) -> str:
    """Find an unquoted occurrence and retain its surrounding source checks."""
    quotes = list(QUOTES.finditer(message))
    for match in re.finditer(re.escape(evidence), message):
        start, end = match.span()
        if any(q.start() <= start < q.end() or q.start() < end <= q.end() for q in quotes):
            continue
        left = max(message.rfind(c, 0, start) for c in ".!?;\n") + 1
        ends = [p for c in ".!?;\n" if (p := message.find(c, start)) >= 0]
        right = min(ends) if ends else len(message)
        sentence = message[left:right]
        if re.search(r"представь|предположим|допустим|вообрази|считай|\bесли\b|наверное|возможно|может быть|например|вымышлен|гипотетич|переведи|цитат|imagine|suppose|fiction|translate", sentence, re.I):
            continue
        # A separate return question does not turn the preceding assertion into
        # a question. Keep the hypothetical checks on the WHOLE sentence above.
        question = re.search(r"[,—]\s*а\s+(?:тебя|ты|вы|вас|у тебя|у вас|как|где|чем|что)\b", sentence, re.I)
        assertion_end = left + question.start() if question else right
        if right < len(message) and message[right] == "?" and not question:
            continue
        source = message[start:min(end, assertion_end)]
        if not source.strip() or "?" in source:
            continue
        # Quoted subfragments must not supply the personal assertion marker.
        return QUOTES.sub("", source)
    raise InvalidMemory("evidence_source")


def check_evidence(op: Operation, message: str) -> None:
    """Evidence must come from this user's message, never assistant/history text.

    These conservative guards catch obvious bad sources, not arbitrary semantic
    hallucinations. The extractor remains responsible for meaning and reference
    resolution. In ambiguous cases it must return no operation.
    """
    if not op.evidence.strip() or op.evidence not in message:
        raise InvalidMemory("evidence_missing")
    evidence = assertion_source(op.evidence, message)
    if op.action == "delete":
        if not re.search(r"забудь|удали|не (?:надо )?(?:хранить|запоминай)|forget|delete|remove", evidence, re.I):
            raise InvalidMemory("deletion_request")
        return
    markers = {
        "name": r"меня зовут|мо[её] имя|зови меня|имя\s*[:—-]|my name|call me|\bя\s+(?:теперь\s+)?" + re.escape(op.value or "") + r"\b",
        "city": r"(?<![а-я])живу\b|проживаю|переехал[аи]?\b|мо[йё] город|город проживания|i live|i moved|my city",
        "occupation": r"\bя\s+(?:(?:теперь|сейчас|уже)\s+)?(?:(?:работаю|тружусь)\s+|(?:по профессии|по специальности)\s+)?[а-яёa-z]|\b(?:работаю|тружусь)\s+[а-яёa-z]|\bмо[яё]\s+(?:профессия|занятие)|\bi (?:am|work as)\b",
        "preference": r"люблю|нравится|предпочитаю|обожаю|ненавижу|любим[а-я]*\s|мне (?:нрав|важн)|прошу|отвечай|(?:пиши|пишите|отвечайте)\s+(?:коротко|кратко|подробно|лаконично)|не хочу|i (?:like|love|prefer|dislike)|my favou?rite|please (?:answer|reply)",
    }
    if not re.search(markers[op.kind], evidence, re.I):
        raise InvalidMemory("personal_assertion")
    if op.kind == "name" and normalized(op.value) not in normalized(evidence):
        raise InvalidMemory("value_missing")
    if op.kind == "occupation":
        # Require a lexical link even when Russian case endings differ:
        # программист / программистом, учитель / учителем. Meaning is checked
        # by the extractor; unrelated values cannot pass just because of «я».
        words = re.findall(r"[а-яёa-z]{3,}", normalized(op.value))
        if not words or not all(re.search(r"\b" + re.escape(w[:max(3, len(w) - 2)]) + r"[а-яёa-z]*\b", evidence, re.I) for w in words):
            raise InvalidMemory("value_missing")
        first_stem = re.escape(words[0][:max(3, len(words[0]) - 2)])
        subject = (r"\bя\s+(?:(?:теперь|сейчас|уже)\s+)?(?:(?:работаю|тружусь)\s+|"
                   r"(?:по профессии|по специальности)\s+)?|\b(?:работаю|тружусь)\s+|"
                   r"\bмо[яё]\s+(?:профессия|занятие)\s*[:—-]?\s*|\bi (?:am|work as)\s+")
        if not re.search(r"(?:" + subject + r")" + first_stem + r"[а-яёa-z]*\b", evidence, re.I):
            raise InvalidMemory("occupation_subject")
        if re.search(r"\b(?:не|хочу|буду|был|была|стану|мечтаю|могу|могла|мог|бы|ли)\b", evidence, re.I):
            raise InvalidMemory("occupation_not_current")


@dataclass(frozen=True)
class MemoryPlan:
    before: tuple[Fact, ...]
    after: tuple[Fact, ...]
    operations: tuple[dict, ...]


def prepare_memory(facts: list[Fact], extraction: Extraction, message: str) -> MemoryPlan:
    if not extraction.operations and re.search(r"\bзапомни(?:те)?\b|\bremember\b", message, re.I):
        raise InvalidMemory("empty_explicit_request")
    original = {f.id: f for f in facts}
    result = dict(original)
    touched: set[int] = set()
    slots: set[str] = set()
    applied: list[dict] = []
    next_id = -1
    for op in extraction.operations:
        check_evidence(op, message)
        if op.kind in SINGLETON_KINDS:
            if op.kind in slots:
                raise InvalidMemory("conflicting_operations")
            slots.add(op.kind)
        target = original.get(op.fact_id)
        if op.action != "add":
            if target is None or target.kind != op.kind or target.id in touched:
                raise InvalidMemory("fact_ownership")
            touched.add(target.id)
        if op.action == "delete":
            del result[target.id]
            applied.append(op.model_dump())
            continue
        if op.action == "add" and op.kind in SINGLETON_KINDS:
            target = next((f for f in result.values() if f.kind == op.kind), None)
        duplicates = [f for f in result.values() if f.kind == op.kind and normalized(f.value) == normalized(op.value)]
        if duplicates:
            if target and duplicates[0].id != target.id:
                raise InvalidMemory("duplicate_target")
            continue
        if target:
            result[target.id] = replace(target, value=op.value)
            applied.append({**op.model_dump(), "action": "update", "fact_id": target.id})
        else:
            result[next_id] = Fact(next_id, op.kind, op.value)
            applied.append({**op.model_dump(), "fact_id": next_id})
            next_id -= 1
    if len(result) > MAX_FACTS:
        raise MemoryFull
    return MemoryPlan(tuple(facts), tuple(result.values()), tuple(applied))
