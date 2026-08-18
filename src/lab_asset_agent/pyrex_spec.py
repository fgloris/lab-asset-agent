from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import InstrumentSpec

DEFAULT_SOURCE = Path("desc_dataset/labware_dataset/assets_major.jsonl")

_REQUIRED_KEYS = (
    "variant_id",
    "family_id",
    "name",
    "description",
    "all_images",
    "table_columns",
    "specs",
    "geometry_conditioning_specs",
    "conditioning_text",
)


def load_pyrex_line(source: Path, line: int) -> dict[str, Any]:
    """Return the parsed JSON record at the 1-indexed ``line`` of a JSONL file."""
    source = source.expanduser().resolve()
    data_lines = 0
    with source.open(encoding="utf-8") as handle:
        for index, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            data_lines += 1
            if data_lines == line:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {line} of {source} is not valid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Line {line} of {source} is not a JSON object.")
                return record
    raise ValueError(
        f"Line {line} is out of range; {source} has {data_lines} data lines (1-indexed)."
    )


def sanitize_variant_id(variant_id: str) -> str:
    """Coerce a catalog variant id into the ``^[a-z0-9][a-z0-9_-]*$`` id pattern."""
    value = re.sub(r"[^a-z0-9_-]", "_", variant_id.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError(f"variant_id cannot be sanitized into a valid id: {variant_id!r}")
    if not value[0].isalnum():
        value = "v" + value
    return value


def pyrex_record_to_spec(record: dict[str, Any], base_dir: Path) -> InstrumentSpec:
    """Map one catalog JSONL record to a dataset-driven :class:`InstrumentSpec`.

    ``base_dir`` is the directory containing the JSONL file; the relative
    ``all_images`` paths are resolved against it.
    """
    name = str(record["name"])
    specs = record.get("geometry_conditioning_specs") or record.get("specs") or {}
    if not isinstance(specs, dict):
        raise ValueError(f"Record {name!r} has non-dict specs: {specs!r}")
    specs = {str(key): str(value) for key, value in specs.items()}

    base_dir = base_dir.expanduser().resolve()
    reference_images: list[Path] = []
    for relative in record.get("all_images") or []:
        candidate = (base_dir / str(relative)).resolve()
        if candidate.is_file():
            reference_images.append(candidate)

    return InstrumentSpec(
        id=sanitize_variant_id(str(record["variant_id"])),
        name=name,
        description=str(record.get("conditioning_text") or record.get("description") or ""),
        specs=specs,
        reference_images=reference_images,
    )
