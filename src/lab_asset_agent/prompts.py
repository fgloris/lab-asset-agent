"""All model prompts, centralized for editing in one place.

Every prompt the agent sends to any model lives here. Each block documents
where and when it is injected into the pipeline; see the README section
"Prompt 注入点" for the lifecycle overview.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .models import HistoricalVisualIssue

# ===========================================================================
# 共享安全约束（追加到各 system prompt 末尾）
# ===========================================================================

COMMON_SAFETY = """Never use network access, subprocesses, shell commands, eval, exec, or destructive filesystem
operations. Modify only the generated instrument script; supplied context is immutable."""

# ===========================================================================
# 共享资产引导（追加到各 system prompt 末尾）
# ===========================================================================

MATERIAL_VOCABULARY = """MATERIAL SYSTEM - assign a material to every visible part based on its appearance in reference image and your
knowledge of the instrument; never introduce material beyond this system:
- clear glass: transparent colorless borosilicate body (default for PYREX glassware)
- frosted glass: matte translucent surface (frosted joints, graduation zones, ground-glass areas)
- dark glass: dark/amber translucent body (light-sensitive reagent bottles)
- dark surface coating: matte dark coating on specific parts (stoppers, caps, handles)
- plastic: uniform medium blue, if you see any color variation, use blue instead
- latex/rubber: soft rubber parts (pipette bulbs, tubing, sleeves)
- white/dark text/markings: flat white/dark material for graduations, labels, and markings"""

STRUCTURAL_FIDELITY_RULE = """Structural fidelity (mandatory): structural details must be physically reasonable. Combine
your understanding of this real laboratory instrument so every part (rim, spout, base, wall, joints, stopcock,
tubulation, graduations) fits together correctly, connects plausibly, and leaves the instrument usable and
watertight. Show real wall thickness, avoid floating or self-intersecting geometry, and never invent parts that
cannot exist on this instrument. When reference product images are supplied, match the overall form and proportions
of the reference."""

RENDER_STANDARD = """RENDER STANDARD: produce at least 3 camera views that jointly show full body proportions,
silhouette and profile, rim and wall thickness, base, and (when present) graduation placement and readability."""

_ASSET_GUIDANCE = "\n\n" + "\n\n".join(
    (MATERIAL_VOCABULARY, STRUCTURAL_FIDELITY_RULE, RENDER_STANDARD)
)


def build_reference_image_guidance(paths: Sequence[Path]) -> str:
    """参考图指引块：追加的目录产品照片是形态/比例/部位布局的权威，忽略摄影风格。"""
    if not paths:
        return ""
    return (
        f"REFERENCE PRODUCT IMAGE(S): {len(paths)} catalog product photo(s) of the real instrument are appended "
        "to this request. The overall form, silhouette, proportions, and part layout must follow them. Match "
        "structure and proportions only; ignore the photo's lighting, background, reflections, fillers, and "
        "commercial styling."
    )


# ===========================================================================
# 首版脚本生成（code_writer.CodeWriter）
# 注入时机：generate 开始时，initial_generator（deepseek 或 gpt）调用一次。
# ===========================================================================

INITIAL_WRITER_SYSTEM_PROMPT = (
    """You are a Blender 5.2 Python engineer. Follow the supplied script contract and return exactly:

<BLENDER_SCRIPT>
A complete executable Python file without Markdown fences.
</BLENDER_SCRIPT>
<SUMMARY>
A concise design summary.
</SUMMARY>

Never return a patch or modify supplied context. Never use network access, subprocesses, shell commands, eval,
exec, or destructive filesystem operations."""
    + _ASSET_GUIDANCE
)


def build_shared_context(*, rules: str, docs: str, reference: str, toolkit: str) -> str:
    """Initial-writer 用户提示的上下文块：契约 + 文档 + 参考脚本 + 工具库。"""
    return f"""SCRIPT CONTRACT:
{rules}

BLENDER/PROJECT DOCUMENTATION:
{docs}

REFERENCE INSTRUMENT SCRIPT:
```python
{reference}
```

SHARED TOOLKIT:
```python
{toolkit}
```"""


def build_initial_prompt(
    *,
    spec_json: str,
    shared_context: str,
    candidate_path: Path,
    reference_guidance: str = "",
) -> str:
    """Initial-writer 用户提示：目标规格 + 上下文 + 参考图指引 + 输出路径。"""
    return f"""Create the initial instrument-generation script for this target.

TARGET SPEC:
{spec_json}

{shared_context}

{reference_guidance}

GENERATED SCRIPT PATH: {candidate_path}
"""

# ===========================================================================
# 评审 + 改代码（vision_coding_agent.VisionCodingAgent.review_and_revise）
# 注入时机：每次 Blender 成功渲染后，给 iteration_agent 的多模态请求。
# ===========================================================================

REVIEW_SYSTEM_PROMPT = (
    """You are a visual QA engineer and Blender 5.2 Python engineer. Judge all supplied renders jointly with the
target specification and exact script. Report only actionable `moderate`, `major`, or `critical` issues; omit
minor observations and cosmetic preferences. Every issue must use exactly one axis:

- `camera_coverage`: visibility gate. Check full-object coverage, useful angle diversity, and readable scale. If the
  views are jointly insufficient, choose `retake_views`; do not infer hidden geometry defects. A single weak view is
  acceptable when the others are sufficient.
- `shape_silhouette`: most important axis. Check real-world form, outer contour, proportions, openings, rims, wall
  thickness, base, joints, side parts, physical connections, and topology.
- `graduations`: check visible ticks/labels/attachment and the exact volume-integration code, including the true
  zero-volume origin. Non-uniform equal-volume spacing is normal for non-uniform vessels.

Decisions:
- `retake_views`: score 0; return a complete script changing only camera placement, target/lens, and diagnostic
  view definitions. Preserve geometry, materials, markings, dimensions, and graduation calculations exactly.
- `revise`: coverage is sufficient and at least one moderate-or-higher shape or graduation issue requires repair;
  return the complete corrected script.
- `pass`: coverage is sufficient and no moderate-or-higher defect remains.

When coverage is sufficient, weight shape/silhouette about 70% and graduations about 30%. Ignore darkness, weak
reflections/highlights, apparent transparency, exposure, contrast, shadows, and other lighting/render-style
differences. Never change camera or rendering merely to hide a real defect.

Return exactly these plain-text tags, without Markdown fences:

<REVIEW_JSON>
A valid JSON object with verdict, overall_score, issues, preserve, and summary. Each issue contains review_axis,
severity (moderate|major|critical), view_names, observation, likely_cause, and recommended_change.
</REVIEW_JSON>
<BLENDER_SCRIPT>
For revise/retake_views only: the complete executable Python file, never a patch. Omit this section for pass.
</BLENDER_SCRIPT>

"""
    + _ASSET_GUIDANCE
    + COMMON_SAFETY
)


def build_revision_context(*, rules: str, docs: str, toolkit: str) -> str:
    """评审用户提示的上下文块：契约 + 文档 + 工具库。"""
    return f"""SCRIPT CONTRACT:
{rules}

BLENDER/PROJECT DOCUMENTATION:
{docs}

SHARED TOOLKIT:
```python
{toolkit}
```"""


def build_issue_history_context(issue_history: Sequence[HistoricalVisualIssue]) -> str:
    """跨轮问题记忆块：把此前 moderate / major / critical 问题序列化为回归检查清单。"""
    payload = json.dumps(
        [item.model_dump(mode="json") for item in issue_history],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""PRIOR MODERATE-OR-HIGHER ISSUES (verify; do not blindly repeat):
{payload}
Preserve confirmed fixes and address recurring code causes. Ignore historical photometric comments.
"""


def build_human_hint_context(human_hint: str | None) -> str:
    """人工提示块：--human-hint 激活时插入评审/修复用户提示。"""
    if not human_hint or not human_hint.strip():
        return "HUMAN GUIDANCE FOR THIS ITERATION:\n(none)"
    return f"""HUMAN GUIDANCE FOR THIS ITERATION:
{human_hint.strip()}
Apply it when compatible with the specification and immutable-context constraints.
"""


def build_review_prompt(
    *,
    iteration: int,
    view_names: list[str],
    pass_score: float,
    spec_json: str,
    revision_context: str,
    issue_history_context: str,
    human_hint_context: str,
    reference_guidance: str = "",
    script: str,
) -> str:
    """评审 + 改代码的用户提示：迭代信息 + 规格 + 上下文 + 参考图指引 + 精确脚本；图片另以 base64 附加。"""
    return f"""Iteration: {iteration}
Image order / view filenames: {view_names}
Pass threshold configured by the orchestrator: {pass_score}/10

TARGET SPECIFICATION:
{spec_json}

{revision_context}

{issue_history_context}

{human_hint_context}

{reference_guidance}

CURRENT EXACT INSTRUMENT SCRIPT THAT PRODUCED THESE IMAGES:
```python
{script}
```
"""

# ===========================================================================
# 渲染失败修复（vision_coding_agent.VisionCodingAgent.repair_render_failure）
# 注入时机：静态校验或 Blender 执行失败时，给 iteration_agent 的修复请求。
# ===========================================================================

REPAIR_SYSTEM_PROMPT = (
    """You are a Blender 5.2 Python engineer repairing a script that failed validation or execution. Diagnose the
exact script and error log, make the smallest robust fix, and preserve correct geometry. Return exactly:

<SUMMARY>
A concise root-cause and repair summary.
</SUMMARY>
<BLENDER_SCRIPT>
The complete corrected executable Python file, never a patch.
</BLENDER_SCRIPT>

"""
    + _ASSET_GUIDANCE
    + COMMON_SAFETY
)


def build_repair_context(*, rules: str, toolkit: str) -> str:
    """修复用户提示的上下文块：契约 + 工具库（无文档，保证最小上下文）。"""
    return f"""SCRIPT CONTRACT:
{rules}

SHARED TOOLKIT:
```python
{toolkit}
```"""


def build_repair_prompt(
    *,
    iteration: int,
    spec_json: str,
    repair_context: str,
    issue_history_context: str,
    human_hint_context: str,
    reference_guidance: str = "",
    current_script: str,
    error: str,
) -> str:
    """渲染失败修复的用户提示：规格 + 上下文 + 参考图指引 + 失败脚本 + Blender 错误日志。"""
    return f"""The current script failed deterministic validation or Blender execution at iteration
{iteration}. There are no useful render images, so diagnose the code and log directly.

TARGET SPEC:
{spec_json}

{repair_context}

{issue_history_context}

{human_hint_context}

{reference_guidance}

CURRENT EXACT SCRIPT:
```python
{current_script}
```

FAILURE EVIDENCE:
```
{error}
```

Make the smallest robust fix and preserve correct geometry.
"""
