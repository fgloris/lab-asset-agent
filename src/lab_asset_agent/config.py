from __future__ import annotations

import os
from pathlib import Path

import yaml

from .models import AppConfig, InstrumentSpec


def load_config(path: Path) -> AppConfig:
    path = path.expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    if data.get("project_root") in (None, "."):
        data["project_root"] = str(path.parent)
    return AppConfig.model_validate(data)


def load_spec(path: Path) -> InstrumentSpec:
    path = path.expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Instrument spec must be a mapping: {path}")
    return InstrumentSpec.model_validate(data)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value
