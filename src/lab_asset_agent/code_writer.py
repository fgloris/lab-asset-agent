from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .models import (
    AppConfig,
    HistoricalVisualIssue,
    InstrumentSpec,
    ModelConfig,
    TokenUsage,
    VLMReview,
)
from .openai_compatible import OpenAICompatibleClient
from .prompts import (
    CODE_REVISE_SYSTEM_PROMPT,
    INITIAL_WRITER_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    build_human_hint_context,
    build_initial_prompt,
    build_issue_history_context,
    build_reference_image_guidance,
    build_repair_context,
    build_repair_prompt,
    build_revision_context,
    build_revision_prompt,
    build_shared_context,
)
from .utils import extract_json_object, image_data_url


_PARSE_RETRIES = 3


def _print_parse_retry(label: str, attempt: int, exc: Exception) -> None:
    message = str(exc).replace("\n", " ")
    if len(message) > 240:
        message = message[:237] + "..."
    print(
        f"[lab-asset-agent] {label}: parse failed on attempt "
        f"{attempt}/{_PARSE_RETRIES}; retrying. Reason: {message}",
        file=sys.stderr,
        flush=True,
    )


@dataclass
class ScriptWriteResult:
    script: str
    raw_response: str
    usage: TokenUsage = field(default_factory=TokenUsage)


class CodeWriter:
    """Write Blender scripts from text context.

    The same class handles the initial script, iterative code revisions based
    on a visual review JSON, and render-error repairs. A caller may pass any
    configured text or vision model; images are sent only for the initial route
    when that selected model supports them.
    """

    def __init__(
        self,
        config: AppConfig,
        model_config: ModelConfig | None = None,
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
        # Reference images only reach a vision-capable initial generator.
        send_images = self.model_config.vision and bool(reference_images)
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

    async def revise_from_review(
        self,
        spec: InstrumentSpec,
        script_path: Path,
        iteration: int,
        review: VLMReview,
        issue_history: list[HistoricalVisualIssue] | None = None,
        human_hint: str | None = None,
    ) -> ScriptWriteResult:
        prompt = build_revision_prompt(
            iteration=iteration,
            review_verdict=review.verdict,
            spec_json=json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2),
            revision_context=build_revision_context(
                rules=self.rules, docs=self.docs, toolkit=self.toolkit
            ),
            issue_history_context=build_issue_history_context(issue_history or []),
            human_hint_context=build_human_hint_context(human_hint),
            review_json=json.dumps(review.model_dump(mode="json"), ensure_ascii=False, indent=2),
            current_script=script_path.read_text(encoding="utf-8"),
        )
        return await self._complete_script_with_retries(
            messages=[
                {"role": "system", "content": CODE_REVISE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            label=f"GPT revision iteration {iteration}",
            stream_label=f"GPT revision iteration {iteration} ({self.model_config.model})",
            partial_path=script_path.parent / "gpt_revision_response.partial.txt",
            final_response_path=script_path.parent / "gpt_revision_response.txt",
        )

    async def repair_render_failure(
        self,
        spec: InstrumentSpec,
        script_path: Path,
        iteration: int,
        error: str,
        issue_history: list[HistoricalVisualIssue] | None = None,
        human_hint: str | None = None,
        reference_images: list[Path] = (),
    ) -> ScriptWriteResult:
        reference_images = list(reference_images)
        send_images = self.model_config.vision and bool(reference_images)
        prompt = build_repair_prompt(
            iteration=iteration,
            spec_json=json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2),
            repair_context=build_repair_context(rules=self.rules, toolkit=self.toolkit),
            issue_history_context=build_issue_history_context(issue_history or []),
            human_hint_context=build_human_hint_context(human_hint),
            reference_guidance=(
                build_reference_image_guidance(reference_images) if send_images else ""
            ),
            current_script=script_path.read_text(encoding="utf-8"),
            error=error,
        )
        user_content: str | list[dict] = prompt
        if send_images:
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
        return await self._complete_script_with_retries(
            messages=[
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            label=f"GPT render repair iteration {iteration}",
            stream_label=f"GPT render repair iteration {iteration} ({self.model_config.model})",
            partial_path=script_path.parent / "repair_agent_response.partial.txt",
            final_response_path=script_path.parent / "repair_agent_response.txt",
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
        script = self._parse_response(text)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(script.rstrip() + "\n", encoding="utf-8")
        return "Initial coder generated the candidate script.", completion.usage

    async def _complete_script_with_retries(
        self,
        *,
        messages: list[dict],
        label: str,
        stream_label: str,
        partial_path: Path,
        final_response_path: Path,
    ) -> ScriptWriteResult:
        total_usage = TokenUsage()
        last_error: Exception | None = None
        for attempt in range(_PARSE_RETRIES):
            completion = await self.client.chat(
                messages,
                stream_label=stream_label,
                stream_output_path=partial_path,
            )
            text = completion.text
            total_usage.add(completion.usage)
            final_response_path.write_text(text, encoding="utf-8")
            partial_path.unlink(missing_ok=True)
            try:
                script = self._parse_response(text)
                if not self._looks_like_python_script(script):
                    raise RuntimeError(
                        "Model returned a <BLENDER_SCRIPT> that is not recognizable as Blender Python."
                    )
                return ScriptWriteResult(
                    script=script,
                    raw_response=text,
                    usage=total_usage,
                )
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                _print_parse_retry(label, attempt + 1, exc)
        assert last_error is not None
        raise last_error

    @classmethod
    def _parse_response(cls, text: str) -> str:
        """Extract a complete script from tagged, JSON, fenced, or raw output."""

        tagged_script = re.search(
            r"<BLENDER_SCRIPT>\s*(.*?)\s*</BLENDER_SCRIPT>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if tagged_script:
            script = cls._strip_code_fence(tagged_script.group(1))
            if not script:
                raise RuntimeError("Initial-writer response contained an empty <BLENDER_SCRIPT> section.")
            return script

        try:
            payload = extract_json_object(text)
        except (ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            script = payload.get("script")
            if isinstance(script, str) and script.strip():
                return cls._strip_code_fence(script)

        fenced = re.search(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            script = cls._strip_code_fence(fenced.group(1))
            if script:
                return script

        raw = text.strip()
        if cls._looks_like_python_script(raw):
            return raw

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
