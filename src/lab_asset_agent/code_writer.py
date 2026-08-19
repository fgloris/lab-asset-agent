from __future__ import annotations

import json
import re
from pathlib import Path

from .models import AppConfig, InstrumentSpec, OpenAICompatibleModelConfig, TokenUsage
from .openai_compatible import OpenAICompatibleClient
from .prompts import (
    INITIAL_WRITER_SYSTEM_PROMPT,
    build_initial_prompt,
    build_reference_image_guidance,
    build_shared_context,
)
from .utils import extract_json_object, image_data_url


class CodeWriter:
    """Generate only the initial Blender script.

    The model can be DeepSeek or the same GPT endpoint used by the iteration
    agent. All post-render reasoning and revisions are intentionally handled by
    :class:`VisionCodingAgent` in one multimodal request.
    """

    def __init__(
        self,
        config: AppConfig,
        model_config: OpenAICompatibleModelConfig | None = None,
        *,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.config = config
        self.model_config = model_config or config.models.initial_model
        self.client = client or OpenAICompatibleClient(self.model_config)
        self.toolkit = ""
        self.reference = ""
        self.docs = ""
        self.rules = ""

    async def start(self) -> None:
        self.toolkit = self.config.paths.toolkit.read_text(encoding="utf-8")
        self.reference = self.config.paths.reference.read_text(encoding="utf-8")
        self.rules = self.config.paths.rules.read_text(encoding="utf-8")
        doc_parts: list[str] = []
        for path in sorted(self.config.paths.docs_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".py"}:
                doc_parts.append(
                    f"\n### {path.relative_to(self.config.paths.docs_dir)}\n"
                    + path.read_text(encoding="utf-8", errors="replace")
                )
        self.docs = "\n".join(doc_parts)

    async def close(self) -> None:
        return None

    async def create_initial(
        self,
        spec: InstrumentSpec,
        candidate_path: Path,
        reference_images: list[Path] = (),
    ) -> tuple[str, TokenUsage]:
        reference_images = list(reference_images)
        # DeepSeek has no vision route, so reference images only reach the GPT
        # initial generator (which is also the iteration agent's vision config).
        send_images = self.config.models.initial_generator == "gpt" and bool(reference_images)
        prompt = build_initial_prompt(
            spec_json=json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2),
            shared_context=build_shared_context(
                rules=self.rules,
                docs=self.docs,
                reference=self.reference,
                toolkit=self.toolkit,
            ),
            candidate_path=candidate_path,
            reference_guidance=(
                build_reference_image_guidance(reference_images) if send_images else ""
            ),
        )
        return await self._complete_and_write(
            prompt,
            candidate_path,
            reference_images=reference_images if send_images else (),
        )

    async def _complete_and_write(
        self,
        prompt: str,
        candidate_path: Path,
        reference_images: list[Path] = (),
    ) -> tuple[str, TokenUsage]:
        user_content: str | list[dict] = prompt
        if reference_images:
            user_content = [{"type": "text", "text": prompt}]
            for image_path in reference_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(
                                image_path,
                                max_side=self.model_config.max_image_side,
                                jpeg_quality=self.model_config.jpeg_quality,
                            )
                        },
                    }
                )
        partial_path = candidate_path.with_suffix(".initial_response.partial.txt")
        final_response_path = candidate_path.with_suffix(".initial_response.txt")
        completion = await self.client.chat(
            [
                {"role": "system", "content": INITIAL_WRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            stream_label=f"initial generator {self.model_config.model}",
            stream_output_path=partial_path,
        )
        text = completion.text
        # Persist the exact streamed response before parsing so malformed model
        # output remains inspectable and a successful request is never opaque.
        final_response_path.write_text(text, encoding="utf-8")
        partial_path.unlink(missing_ok=True)
        script, summary = self._parse_response(text)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(script.rstrip() + "\n", encoding="utf-8")
        return summary, completion.usage

    @classmethod
    def _parse_response(cls, text: str) -> tuple[str, str]:
        """Extract a complete script from tagged, JSON, fenced, or raw output."""

        tagged_script = re.search(
            r"<BLENDER_SCRIPT>\s*(.*?)\s*</BLENDER_SCRIPT>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if tagged_script:
            tagged_summary = re.search(
                r"<SUMMARY>\s*(.*?)\s*</SUMMARY>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            script = cls._strip_code_fence(tagged_script.group(1))
            summary = (
                tagged_summary.group(1).strip()
                if tagged_summary
                else "The initial writer generated the candidate script."
            )
            if not script:
                raise RuntimeError("Initial-writer response contained an empty <BLENDER_SCRIPT> section.")
            return script, summary

        try:
            payload = extract_json_object(text)
        except (ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            script = payload.get("script")
            if isinstance(script, str) and script.strip():
                return (
                    cls._strip_code_fence(script),
                    str(payload.get("summary") or "The initial writer generated the candidate script."),
                )

        fenced = re.search(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            script = cls._strip_code_fence(fenced.group(1))
            if script:
                return script, "The initial writer returned a complete fenced Python script."

        raw = text.strip()
        if cls._looks_like_python_script(raw):
            return raw, "The initial writer returned a complete Python script."

        raise RuntimeError(
            "Initial-writer response did not contain <BLENDER_SCRIPT>, a legacy JSON `script`, "
            "or a recognizable complete Python file."
        )

    @staticmethod
    def _looks_like_python_script(text: str) -> bool:
        if not text:
            return False
        head = "\n".join(text.splitlines()[:20])
        has_import = bool(re.search(r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+)", head))
        has_blender = "bpy" in text or "lab_blender_toolkit" in text
        return has_import and has_blender

    @staticmethod
    def _strip_code_fence(script: str) -> str:
        script = script.strip()
        if script.startswith("```python"):
            script = script[len("```python") :].lstrip()
        elif script.startswith("```"):
            script = script[3:].lstrip()
        if script.endswith("```"):
            script = script[:-3].rstrip()
        return script
