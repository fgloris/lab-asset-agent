from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


def utc_run_id(spec_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{spec_id}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response does not contain a JSON object.")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Parsed JSON value is not an object.")
    return value


def tail_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return "...<truncated>...\n" + text[-max_chars:]


def image_data_url(path: Path, *, max_side: int, jpeg_quality: int) -> str:
    """Encode an image as a JPEG data URL, RGBA flattened onto white."""
    with Image.open(path) as source:
        source.thumbnail((max_side, max_side))
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            rgba = source.convert("RGBA")
            image = Image.new("RGB", rgba.size, (255, 255, 255))
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = source.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
