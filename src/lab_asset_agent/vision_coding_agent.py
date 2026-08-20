from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .code_writer import CodeWriter
from .models import (
    AppConfig,
    HistoricalVisualIssue,
    InstrumentSpec,
    TokenUsage,
    VLMReview,
)
from .openai_compatible import OpenAICompatibleClient
from .prompts import (
    REPAIR_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    build_human_hint_context,
    build_issue_history_context,
    build_repair_context,
    build_repair_prompt,
    build_reference_image_guidance,
    build_reference_pairing_guidance,
    build_revision_context,
    build_review_prompt,
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
class VisionCodeDecision:
    review: VLMReview
    revised_script: str | None
    summary: str
    raw_response: str
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class ScriptRepair:
    script: str
    summary: str
    raw_response: str
    usage: TokenUsage = field(default_factory=TokenUsage)


class VisionCodingAgent:
    """One GPT call reviews renders and chooses pass, revision, or a view retake."""

    def __init__(
        self,
        config: AppConfig,
        *,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.config = config
        self.model_config = config.models.iterative_model
        self.client = client or OpenAICompatibleClient(self.model_config)
        self.toolkit = ""
        self.docs = ""
        self.rules = ""

    async def start(self) -> None:
        self.toolkit = self.config.paths.toolkit.read_text(encoding="utf-8")
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

    async def review_and_revise(
        self,
        spec: InstrumentSpec,
        script_path: Path,
        images: list[Path],
        iteration: int,
        issue_history: list[HistoricalVisualIssue] | None = None,
        human_hint: str | None = None,
        reference_images: list[Path] = (),
    ) -> VisionCodeDecision:
        if not images:
            raise ValueError("No render images were supplied to the iterative generator.")
        selected = list(images)
        reference_images = list(reference_images)
        pair_count = min(len(selected), len(reference_images))

        if pair_count:
            paired_renders = selected[:pair_count]
            paired_refs = reference_images[:pair_count]
            extra_renders = selected[pair_count:][: self.model_config.extra_images]
            ordered_images: list[Path] = []
            image_labels: list[str] = []
            for ref_path, render_path in zip(paired_refs, paired_renders):
                ordered_images.append(ref_path)
                image_labels.append(f"图{len(image_labels) + 1}: 目标参考图 {ref_path.name}")
                ordered_images.append(render_path)
                image_labels.append(f"图{len(image_labels) + 1}: 当前渲染视角 {render_path.name}")
            for render_path in extra_renders:
                ordered_images.append(render_path)
                image_labels.append(f"图{len(image_labels) + 1}: 辅助渲染视角 {render_path.name}")
            reference_guidance = build_reference_pairing_guidance(
                pair_count, len(extra_renders)
            )
        else:
            ordered_images = selected
            image_labels = [
                f"图{index}: 当前渲染视角 {path.name}"
                for index, path in enumerate(selected, start=1)
            ]
            reference_guidance = build_reference_image_guidance(reference_images)

        content: list[dict] = [
            {
                "type": "text",
                "text": build_review_prompt(
                    iteration=iteration,
                    image_labels=image_labels,
                    pass_score=self.config.loop.pass_score,
                    spec_json=json.dumps(
                        spec.model_dump(mode="json"), ensure_ascii=False, indent=2
                    ),
                    revision_context=build_revision_context(
                        rules=self.rules, docs=self.docs, toolkit=self.toolkit
                    ),
                    issue_history_context=build_issue_history_context(issue_history or []),
                    human_hint_context=build_human_hint_context(human_hint),
                    reference_guidance=reference_guidance,
                    script=script_path.read_text(encoding="utf-8"),
                ),
            }
        ]
        for image_path in ordered_images:
            content.append(
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

        # Deliberately send no response_format. The response contains JSON review
        # plus a long Python file, which is more reliable as tagged plain text.
        partial_path = script_path.parent / "gpt_review_and_code_response.partial.txt"
        final_response_path = script_path.parent / "gpt_review_and_code_response.txt"
        total_usage = TokenUsage()
        last_error: Exception | None = None
        for attempt in range(_PARSE_RETRIES):
            completion = await self.client.chat(
                [
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                stream_label=(
                    f"GPT review+coder iteration {iteration} "
                    f"({self.model_config.model})"
                ),
                stream_output_path=partial_path,
            )
            total_usage.add(completion.usage)
            text = completion.text
            final_response_path.write_text(text, encoding="utf-8")
            partial_path.unlink(missing_ok=True)
            try:
                decision = self._parse_decision(text)
                decision.usage = total_usage
                return decision
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                _print_parse_retry(
                    f"GPT review+coder iteration {iteration}",
                    attempt + 1,
                    exc,
                )
        assert last_error is not None
        raise last_error

    async def repair_render_failure(
        self,
        spec: InstrumentSpec,
        script_path: Path,
        iteration: int,
        error: str,
        issue_history: list[HistoricalVisualIssue] | None = None,
        human_hint: str | None = None,
        reference_images: list[Path] = (),
    ) -> ScriptRepair:
        reference_images = list(reference_images)
        prompt = build_repair_prompt(
            iteration=iteration,
            spec_json=json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2),
            repair_context=build_repair_context(rules=self.rules, toolkit=self.toolkit),
            issue_history_context=build_issue_history_context(issue_history or []),
            human_hint_context=build_human_hint_context(human_hint),
            reference_guidance=build_reference_image_guidance(reference_images),
            current_script=script_path.read_text(encoding="utf-8"),
            error=error,
        )
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
        partial_path = script_path.parent / "repair_agent_response.partial.txt"
        final_response_path = script_path.parent / "repair_agent_response.txt"
        total_usage = TokenUsage()
        last_error: Exception | None = None
        for attempt in range(_PARSE_RETRIES):
            completion = await self.client.chat(
                [
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                stream_label=(
                    f"GPT render repair iteration {iteration} "
                    f"({self.model_config.model})"
                ),
                stream_output_path=partial_path,
            )
            total_usage.add(completion.usage)
            text = completion.text
            final_response_path.write_text(text, encoding="utf-8")
            partial_path.unlink(missing_ok=True)
            try:
                script, summary = CodeWriter._parse_response(text)
                return ScriptRepair(
                    script=script,
                    summary=summary,
                    raw_response=text,
                    usage=total_usage,
                )
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                _print_parse_retry(
                    f"GPT render repair iteration {iteration}",
                    attempt + 1,
                    exc,
                )
        assert last_error is not None
        raise last_error

    def write_revision(self, decision: VisionCodeDecision, candidate_path: Path) -> None:
        if decision.revised_script is None:
            raise RuntimeError("The GPT decision did not contain a revised script.")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(decision.revised_script.rstrip() + "\n", encoding="utf-8")

    @classmethod
    def _parse_decision(cls, text: str) -> VisionCodeDecision:
        difference_match = re.search(
            r"<DIFFERENCE>\s*(.*?)\s*</DIFFERENCE>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not difference_match:
            raise RuntimeError("GPT response is missing <DIFFERENCE>.")
        difference = difference_match.group(1).strip()
        if not difference:
            raise RuntimeError("GPT <DIFFERENCE> must not be empty.")

        review_match = re.search(
            r"<REVIEW_JSON>\s*(.*?)\s*</REVIEW_JSON>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not review_match:
            raise RuntimeError("GPT response is missing <REVIEW_JSON>.")
        review_payload = extract_json_object(review_match.group(1))
        if not isinstance(review_payload, dict):
            raise RuntimeError("GPT <REVIEW_JSON> must contain a JSON object.")
        issues_payload = review_payload.get("issues", [])
        if not isinstance(issues_payload, list):
            raise RuntimeError("GPT review `issues` must be a JSON array.")

        actionable_issues: list[dict] = []
        for issue_index, issue in enumerate(issues_payload, start=1):
            if not isinstance(issue, dict):
                raise RuntimeError(f"GPT review issue {issue_index} must be a JSON object.")
            # Keep old manifests readable, but never expose/store new minor findings.
            # Minor observations are deliberately omitted from the actionable protocol.
            if issue.get("severity") == "minor":
                continue
            if "review_axis" not in issue:
                raise RuntimeError(
                    f"GPT review issue {issue_index} is missing required `review_axis`."
                )
            if issue["review_axis"] == "shape":
                raise RuntimeError(
                    "GPT review issue uses deprecated `shape`; use `overall_shape` "
                    "or `component_shape`."
                )
            actionable_issues.append(issue)

        review_payload = dict(review_payload)
        review_payload["issues"] = actionable_issues
        review = VLMReview.model_validate(review_payload)

        if review.verdict == "revise" and not review.issues:
            raise RuntimeError(
                "verdict=revise requires at least one moderate-or-higher actionable issue."
            )

        if review.verdict == "retake_views":
            if not review.issues:
                raise RuntimeError("verdict=retake_views requires at least one camera_coverage issue.")
            if any(issue.review_axis != "camera_coverage" for issue in review.issues):
                raise RuntimeError(
                    "verdict=retake_views may contain only camera_coverage issues; "
                    "uncertain geometry must not be diagnosed before retaking views."
                )

        script_match = re.search(
            r"<BLENDER_SCRIPT>\s*(.*?)\s*</BLENDER_SCRIPT>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        script = None
        if script_match:
            script = CodeWriter._strip_code_fence(script_match.group(1))
            if not script:
                script = None

        if review.verdict in {"revise", "retake_views"} and script is None:
            raise RuntimeError(
                f"GPT returned verdict={review.verdict} but omitted the complete <BLENDER_SCRIPT>."
            )
        if script is not None and not CodeWriter._looks_like_python_script(script):
            raise RuntimeError("GPT returned a <BLENDER_SCRIPT> that is not recognizable as Blender Python.")

        return VisionCodeDecision(
            review=review,
            revised_script=script,
            summary=difference,
            raw_response=text,
        )
