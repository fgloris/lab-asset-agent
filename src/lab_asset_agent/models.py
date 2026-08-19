from __future__ import annotations

from pathlib import Path
from typing import Literal

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


class ModelConfig(BaseModel):
    """Configuration for a single OpenAI-compatible model endpoint.

    ``vision`` marks whether the model accepts image inputs. Only a
    vision-capable model may be selected as ``iterative_generator``.
    """

    base_url: str
    api_key_env: str
    model: str
    vision: bool = False
    max_tokens: int | None = Field(default=4000, ge=256)
    temperature: float | None = Field(default=0.1, ge=0.0, le=2.0)
    max_retries: int = Field(default=8, ge=0)
    connect_timeout_seconds: float = Field(default=60.0, ge=1.0)
    request_timeout_seconds: float = Field(default=900.0, ge=10.0)
    stream: bool = True
    stream_to_terminal: bool = True
    stream_reasoning: Literal["hidden", "progress", "full"] = "progress"
    extra_images: int = Field(default=2, ge=0, le=12)
    max_image_side: int = Field(default=1280, ge=256, le=4096)
    jpeg_quality: int = Field(default=90, ge=50, le=100)


_SHARED_MODEL_FIELDS = (
    "stream",
    "stream_to_terminal",
    "stream_reasoning",
    "max_retries",
    "connect_timeout_seconds",
    "request_timeout_seconds",
    "max_image_side",
    "extra_images",
    "jpeg_quality",
)


class ModelsConfig(BaseModel):
    """Model routing via named providers.

    Shared transport/streaming/image settings live here and are applied to
    every provider unless that provider overrides them. ``iterative_generator``
    must be vision-capable.
    """

    initial_generator: str
    iterative_generator: str
    providers: dict[str, ModelConfig]

    stream: bool = True
    stream_to_terminal: bool = True
    stream_reasoning: Literal["hidden", "progress", "full"] = "progress"
    max_retries: int = Field(default=8, ge=0)
    connect_timeout_seconds: float = Field(default=60.0, ge=1.0)
    request_timeout_seconds: float = Field(default=900.0, ge=10.0)
    max_image_side: int = Field(default=1280, ge=256, le=4096)
    extra_images: int = Field(default=2, ge=0, le=12)
    jpeg_quality: int = Field(default=90, ge=50, le=100)

    @model_validator(mode="before")
    @classmethod
    def apply_shared_settings(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        providers = data.get("providers")
        if not isinstance(providers, dict):
            return data
        for provider in providers.values():
            if isinstance(provider, dict):
                for key in _SHARED_MODEL_FIELDS:
                    if key in data:
                        provider.setdefault(key, data[key])
        return data

    @model_validator(mode="after")
    def validate_routes(self) -> "ModelsConfig":
        unknown = [
            name
            for name in (self.initial_generator, self.iterative_generator)
            if name not in self.providers
        ]
        if unknown:
            raise ValueError(f"Unknown model source(s): {', '.join(unknown)}")
        if not self.providers[self.iterative_generator].vision:
            raise ValueError(
                "models.iterative_generator must reference a model with vision: true."
            )
        return self

    @property
    def initial_model(self) -> ModelConfig:
        return self.providers[self.initial_generator]

    @property
    def iterative_model(self) -> ModelConfig:
        return self.providers[self.iterative_generator]


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
    review_axis: Literal["camera_coverage", "shape", "graduations"] = (
        "shape"
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
    similarity_scores: list[float] = Field(default_factory=list)
    similarity_score: float = Field(ge=0.0, le=10.0)
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


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


class RunManifest(BaseModel):
    run_id: str
    spec_id: str
    status: Literal["running", "passed", "failed", "max_iterations"] = "running"
    final_script: Path | None = None
    final_score: float | None = None
    iterations: list[IterationRecord] = Field(default_factory=list)
    failure_reason: str | None = None
    human_hint: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
