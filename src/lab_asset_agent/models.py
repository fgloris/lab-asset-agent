from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class InstrumentSpec(BaseModel):
    """Dataset-driven target description.

    Fields mirror what a catalog dataset row provides. Anything shared across
    assets (material system, structural fidelity, render standard) lives in
    the prompts rather than here.
    """

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str
    description: str = ""
    specs: dict[str, str] = Field(default_factory=dict)
    reference_images: list[Path] = Field(default_factory=list)


class BlenderConfig(BaseModel):
    executable: str = "blender"
    timeout_seconds: int = 900
    render_engine: Literal["BLENDER_EEVEE", "CYCLES"] = "BLENDER_EEVEE"
    resolution: int = Field(default=768, ge=256, le=4096)
    minimum_render_count: int = Field(default=3, ge=1, le=12)


class OpenAICompatibleModelConfig(BaseModel):
    """Configuration shared by any OpenAI-compatible chat-completions endpoint."""

    base_url: str
    api_key_env: str
    model: str
    max_tokens: int | None = Field(default=4000, ge=256)
    temperature: float | None = Field(default=0.1, ge=0.0, le=2.0)
    max_retries: int = Field(default=8, ge=0)
    connect_timeout_seconds: float = Field(default=60.0, ge=1.0)
    request_timeout_seconds: float = Field(default=900.0, ge=10.0)
    response_format_mode: Literal["auto", "json_schema", "json_object", "text"] = "auto"
    stream: bool = True
    stream_to_terminal: bool = True
    stream_reasoning: Literal["hidden", "progress", "full"] = "progress"


class VisionCodeAgentConfig(OpenAICompatibleModelConfig):
    max_images: int = Field(default=4, ge=1, le=12)
    max_image_side: int = Field(default=1280, ge=256, le=4096)
    jpeg_quality: int = Field(default=90, ge=50, le=100)


class ModelsConfig(BaseModel):
    """Model routing.

    ``initial_generator`` controls only the first script. Every render review,
    visual revision, and render-error repair is handled by ``iteration_agent``.

    Old v0.2 configuration keys (``code_writer`` and ``visual_reviewer``) are
    accepted and migrated automatically so existing config.yaml files still load.
    """

    initial_generator: Literal["deepseek", "gpt"] = "deepseek"
    initial_writer: OpenAICompatibleModelConfig | None = None
    iteration_agent: VisionCodeAgentConfig

    @model_validator(mode="before")
    @classmethod
    def migrate_v02_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "initial_writer" not in data and "code_writer" in data:
            data["initial_writer"] = data["code_writer"]
        if "iteration_agent" not in data and "visual_reviewer" in data:
            migrated = dict(data["visual_reviewer"])
            # v0.2 visual review needed only a small JSON response; v0.3 may
            # return a complete Python file, so preserve compatibility with a
            # safer output budget and timeout. No response_format is sent.
            current_tokens = migrated.get("max_tokens")
            if current_tokens is None or int(current_tokens) < 12000:
                migrated["max_tokens"] = 12000
            current_timeout = migrated.get("request_timeout_seconds")
            if current_timeout is None or int(current_timeout) < 600:
                migrated["request_timeout_seconds"] = 600
            migrated["response_format_mode"] = "text"
            data["iteration_agent"] = migrated
        return data

    @model_validator(mode="after")
    def validate_initial_route(self) -> "ModelsConfig":
        if self.initial_generator == "deepseek" and self.initial_writer is None:
            raise ValueError(
                "models.initial_writer is required when initial_generator=deepseek."
            )
        return self

    @property
    def initial_model(self) -> OpenAICompatibleModelConfig:
        if self.initial_generator == "gpt":
            return self.iteration_agent
        assert self.initial_writer is not None
        return self.initial_writer


class LoopConfig(BaseModel):
    max_iterations: int = Field(default=8, ge=1, le=50)
    pass_score: float = Field(default=8.5, ge=0, le=10)
    keep_all_iterations: bool = True
    stop_on_repeated_script: bool = True
    max_consecutive_render_failures: int = Field(default=3, ge=1, le=20)


class PathsConfig(BaseModel):
    toolkit: Path
    reference: Path
    docs_dir: Path
    rules: Path
    runs_dir: Path


class AppConfig(BaseModel):
    project_root: Path = Path(".")
    blender: BlenderConfig
    models: ModelsConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    paths: PathsConfig

    @model_validator(mode="after")
    def resolve_paths(self) -> "AppConfig":
        root = self.project_root.expanduser().resolve()
        self.project_root = root
        for field_name in (
            "toolkit",
            "reference",
            "docs_dir",
            "rules",
            "runs_dir",
        ):
            value = getattr(self.paths, field_name)
            if not value.is_absolute():
                setattr(self.paths, field_name, (root / value).resolve())
        return self


class VisualIssue(BaseModel):
    review_axis: Literal["camera_coverage", "shape_silhouette", "graduations"] = (
        "shape_silhouette"
    )
    severity: Literal["critical", "major", "moderate", "minor"]
    view_names: list[str] = Field(default_factory=list)
    observation: str
    likely_cause: str
    recommended_change: str


class HistoricalVisualIssue(VisualIssue):
    """A prior review issue retained as regression memory for later iterations."""

    iteration: int = Field(ge=1)
    issue_index: int = Field(ge=1)


class VLMReview(BaseModel):
    verdict: Literal["pass", "revise", "retake_views"]
    overall_score: float = Field(ge=0, le=10)
    issues: list[VisualIssue] = Field(default_factory=list)
    preserve: list[str] = Field(default_factory=list)
    summary: str


class RenderResult(BaseModel):
    success: bool
    return_code: int | None
    command: list[str]
    log_path: Path
    images: list[Path] = Field(default_factory=list)
    blend_files: list[Path] = Field(default_factory=list)
    error_summary: str | None = None
    elapsed_seconds: float


class IterationRecord(BaseModel):
    iteration: int
    script_path: Path
    script_sha256: str
    writer_summary: str
    render: RenderResult
    review: VLMReview | None = None


class RunManifest(BaseModel):
    run_id: str
    spec_id: str
    status: Literal["running", "passed", "failed", "max_iterations"] = "running"
    final_script: Path | None = None
    final_score: float | None = None
    iterations: list[IterationRecord] = Field(default_factory=list)
    failure_reason: str | None = None
    human_hint: str | None = None
    human_hint_from_iteration: int = Field(default=1, ge=1)
