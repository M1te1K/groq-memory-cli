"""Configuration is anchored to the checkout, never to the current directory."""

import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigurationError(ValueError):
    pass


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_key: SecretStr = Field(repr=False)
    model: str = Field(min_length=1, max_length=200)
    db_path: Path
    timeout: float = Field(ge=1, le=120, allow_inf_nan=False)


def load_settings(
    root: Path = PROJECT_ROOT, environ: Mapping[str, str] | None = None
) -> Settings:
    env = os.environ if environ is None else environ
    # Do not mutate os.environ or overwrite an existing .env.
    values = {**dotenv_values(root / ".env", interpolate=False), **env}
    try:
        raw_path = values.get("MEMORY_DB_PATH", "./data/memory.sqlite3")
        if not raw_path or not raw_path.strip():
            raise ValueError
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        return Settings(
            api_key=SecretStr((values.get("GROQ_API_KEY") or "").strip()),
            model=(values.get("GROQ_MODEL") or "openai/gpt-oss-20b").strip(),
            db_path=path.resolve(),
            timeout=values.get("GROQ_TIMEOUT_SECONDS", "20"),
        )
    except (ValueError, TypeError, OSError, ValidationError):
        # Never include Pydantic's input values (possibly secrets) in CLI errors.
        raise ConfigurationError(
            "Некорректная конфигурация. Проверьте путь БД, модель и таймаут (1–120 секунд)."
        ) from None
