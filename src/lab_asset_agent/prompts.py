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

COMMON_SAFETY = """通用安全约束：禁止网络访问、子进程、shell 命令、eval、exec 或破坏性文件系统操作。
只允许修改生成的仪器脚本；提供的上下文不可变。"""

# ===========================================================================
# 共享资产引导（追加到各 system prompt 末尾）
# ===========================================================================

MATERIAL_VOCABULARY = """根据参考图中的外观和你对仪器的知识，为每个可见部件分配材质；不得引入本体系之外的材质：
- clear glass: 透明无色硼硅玻璃体（PYREX 玻璃器皿默认材质）
- frosted glass: 磨砂半透明表面（磨口、刻度区、毛玻璃区域）
- dark glass: 深色/琥珀色半透明体（避光试剂瓶）
- dark surface coating: 部件上的哑光深色涂层（塞子、瓶盖、把手）
- plastic: 均匀中蓝色塑料；如出现颜色变化，统一使用蓝色
- latex/rubber: 软质橡胶部件（吸耳球、管材、套管）
- white text / dark text / markings: 用于刻度、标签和标识的平白/平黑材质"""

STRUCTURAL_FIDELITY_RULE = """当提供参考产品图时，整体形态和比例必须与参考图匹配。仔细观察仪器的整体形态和各个部分的大小比例，然后调整现有模型，确保他们形状基本一致。
结构细节必须物理合理。由于参考图上部件细节往往难分辨，细节上你需要结合你对真实实验室仪器的理解，使每个部件
（口沿、壶嘴、底部、壁、接头、活塞、支管、刻度）正确组合、合理连接，并保持仪器可用且不漏液。表现真实壁厚，
避免悬浮或自相交几何，不虚构该仪器不可能存在的部件。"""

RENDER_STANDARD = """渲染标准：至少生成 3 个相机视角，共同展示完整形体比例、轮廓与侧面、口沿与壁厚、底部，
以及（如存在）刻度位置与可读性。"""

_ASSET_GUIDANCE = "\n\n" + "\n\n".join(
    (STRUCTURAL_FIDELITY_RULE, MATERIAL_VOCABULARY, RENDER_STANDARD)
)


def build_reference_image_guidance(paths: Sequence[Path]) -> str:
    """参考图指引块：追加的产品目录照片是形态/比例/部位布局的权威，忽略摄影风格。"""
    if not paths:
        return ""
    return (
        f"参考产品图：本次请求附带了 {len(paths)} 张真实仪器的产品目录照片。"
        "整体形态、轮廓、比例和部件布局必须遵循这些照片。只匹配结构和比例；"
        "忽略照片的光照、背景、反射、填充物和商业修饰。"
    )


def build_reference_pairing_guidance(pair_count: int, extra_count: int) -> str:
    """评审用硬性对齐目标：前 k 对图逐个对齐，辅助视角仅补充覆盖。"""
    if pair_count <= 0:
        return ""
    lines = [
        f"硬性对齐目标：本次图片按「目标参考图 → 当前渲染视角」成对排列，共 {pair_count} 对。",
        f"前 {pair_count} 个当前渲染视角必须分别与对应顺序的目标参考图在整体轮廓、比例和部件布局上尽量接近，"
        "这是 shape 评分与是否 revise 的首要依据。",
    ]
    if extra_count:
        lines.append(
            f"随后附带的 {extra_count} 个辅助渲染视角仅用于补充覆盖信息，不参与对齐目标。"
        )
    lines.append("只对齐结构和比例；忽略照片的光照、背景、反射和商业修饰，不要把商标画上。")
    return "\n".join(lines)


REVIEW_AXIS_GUIDANCE = """review_axis 只能是以下四类：
- `camera_coverage`：画面是否覆盖完整、角度是否足够
- `overall_shape`：整体轮廓、纵横比、主外形比例、重心感
- `components_shape`：局部部件几何、连接方式、口沿、底部、支管、把手、塞子、刻度带等
- `graduations`：刻度、标签、零点、等体积间距和读数逻辑
"""

RECOMMENDED_CHANGE_RULE = """`recommended_change` 必须是可执行、可验证、尽量定量的修改指令，优先使用以下形式：
- 尺寸：给出 mm、比例、角度、偏移量、数量
- 相机：给出方位变化、仰角变化、距离变化、是否保留焦点
- 刻度：给出起点、终点、间隔、数量、是否重算体积分布
- 组件：给出增减高度、厚度、外扩/内收、位置偏移、圆角半径
禁止只写“略微调整”“更自然”“更像一点”这类不可执行表述。"""


# ===========================================================================
# 首版脚本生成（code_writer.CodeWriter）
# 注入时机：generate 开始时，initial_coder 调用一次。
# ===========================================================================

INITIAL_WRITER_SYSTEM_PROMPT = (
    """你是 Blender 5.2 Python 工程师。严格遵守脚本契约，并只返回以下部分：

<BLENDER_SCRIPT>
一个完整可执行的 Python 文件，不要使用 Markdown 代码围栏。
</BLENDER_SCRIPT>

不要返回补丁，不要修改提供的上下文。通用安全约束：禁止网络访问、子进程、shell 命令、eval、exec
或破坏性文件系统操作。"""
    + _ASSET_GUIDANCE
)


def build_shared_context(*, rules: str, docs: str, reference: str, toolkit: str) -> str:
    """Initial-writer 用户提示的上下文块：契约 + 文档 + 参考脚本 + 工具库。"""
    return f"""脚本契约:
{rules}

BLENDER/项目文档:
{docs}

参考仪器脚本:
```python
{reference}
```

共享工具库:
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
    return f"""为以下目标生成初始仪器脚本。

目标规格:
{spec_json}

{shared_context}

{reference_guidance}

生成脚本路径: {candidate_path}
"""

# ===========================================================================
# 评审（vision_coding_agent.VisualReviewer.review）
# 注入时机：每次 Blender 成功渲染后，给 visual_reviewer 的多模态请求。
# ===========================================================================

REVIEW_SYSTEM_PROMPT = (
    """你是视觉质检工程师。联合评判所有提供的渲染图、目标规格和精确脚本。
先输出参考图与渲染图的视觉差异，再输出视觉评审 JSON；不要输出脚本。报告可执行的 `minor`、`moderate`、`major`、`critical` 问题；省略外观偏好。

<DIFFERENCE> 必须在 <REVIEW_JSON> 之前。用中文按参考图/渲染图配对逐项描述可见差异，重点覆盖整体轮廓、比例、部件布局、局部几何、刻度/标签和相机覆盖。
没有参考图时，改为描述渲染图与目标文字规格之间的差异。不要在此处给代码方案，只记录视觉事实和不确定性。

每个问题必须使用且只能使用一个 review_axis：
- `camera_coverage`：可见性门槛，如需要调整则优先执行。先判断当前视角是否覆盖完整，再看角度多样性和遮挡。
- `overall_shape`：整体轮廓、纵横比、重心、主外形是否匹配参考图。
- `components_shape`：局部部件几何、连接方式、口沿、底部、支管、塞子、把手、刻度带等是否匹配。
- `graduations`：检查可见刻度/标签/附着，以及精确的体积积分代码，包括真实零体积原点。

如果需要重新拍图，只能使用 `camera_coverage`，且不得夹带几何判断。
如果需要修改几何，优先拆成 `overall_shape` 和 `components_shape`。

similarity_score 如何计算：
- 逐对对比参考图和对应视角，从 0-10 为外形/比例/轮廓的相似度打分，得到 similarity_scores。
- similarity_score 取 similarity_scores 的算术平均值；没有参考图时，按渲染图与仪器文本描述的一致性给出 0-10 分。

决策规则：
- `retake_views`：仅修改相机位置、目标/镜头和诊断视角定义。几何、材质、标识、尺寸、刻度计算必须原样保留。
- `revise`：视角覆盖充分且至少存在一个 moderate 或以上 overall_shape/components_shape/graduations 问题需要修复；返回完整修正后的脚本。
- `pass`：覆盖充分且不再存在 moderate 及以上缺陷。

覆盖充分时，overall_shape 重要性约占 50%，components_shape 约占 30%，graduations 约占 20%。忽略暗部、弱反射/高光、表观透明度、曝光、
对比度、阴影以及其他光照/渲染风格差异；但不得用这些风格理由掩盖真实几何差异。绝不要通过修改相机或渲染
来掩盖真实缺陷。

输出格式必须严格包含（标签名 <DIFFERENCE> 和 <REVIEW_JSON> 保持英文，不要使用 Markdown 代码围栏）：

<DIFFERENCE>
参考图与当前渲染图的差异说明。
</DIFFERENCE>

<REVIEW_JSON>
一个合法 JSON 对象。字段名(key)和枚举值必须完全使用下面的英文，不得翻译或改写；字段内容(value)可用中文。
- `verdict`: `pass` | `revise` | `retake_views`
- `similarity_scores`: 0 到 10 的小数数组，长度与参考图数量相同
- `similarity_score`: 0 到 10 的小数
- `issues`: 数组，每项字段为 `review_axis`、`severity`、`view_names`、`observation`、`likely_cause`、
  `recommended_change`
  - `review_axis`: `camera_coverage` | `overall_shape` | `components_shape` | `graduations`
  - `severity`: `moderate` | `major` | `critical`
  - `view_names`: 字符串数组，填写问题涉及的渲染视角文件名
  - `observation`、`likely_cause`、`recommended_change`: 字符串，可用中文
  - `recommended_change` 必须定量、可执行、可验证，不能只写模糊描述
- `preserve`: 字符串数组，列出必须保留的正确内容
- `summary`: 字符串，可用中文
</REVIEW_JSON>

"""
    + _ASSET_GUIDANCE
    + "\n\n"
    + REVIEW_AXIS_GUIDANCE
    + "\n\n"
    + RECOMMENDED_CHANGE_RULE
    + "\n\n"
    + COMMON_SAFETY
)


def build_revision_context(*, rules: str, docs: str, toolkit: str) -> str:
    """评审用户提示的上下文块：契约 + 文档 + 工具库。"""
    return f"""脚本契约:
{rules}

BLENDER/项目文档:
{docs}

共享工具库:
```python
{toolkit}
```"""


CODE_REVISE_SYSTEM_PROMPT = (
    """你是 Blender 5.2 Python 工程师。你不会看到图片，只能根据上一轮视觉评审 JSON、目标规格和精确脚本修订代码。
只返回以下部分：

<BLENDER_SCRIPT>
完整修正后的可执行 Python 文件，不是补丁。
</BLENDER_SCRIPT>

必须严格执行以下约束：
- 只根据 review 里明确列出的问题改代码
- 保留 review 中 `preserve` 列出的内容
- `recommended_change` 是定量指令时，要把它落实成具体数值、比例或角度
- 不要重新评判图像，不要猜测看不见的细节
- 如果 review.verdict == `retake_views`，只能修改相机位置、目标、镜头、诊断视角定义；不得修改几何、材质、尺寸、刻度、标签、部件拓扑
- 如果 review.verdict == `revise`，只修改 review 中列出的 `overall_shape` / `components_shape` / `graduations` 问题
"""
    + _ASSET_GUIDANCE
    + "\n\n"
    + COMMON_SAFETY
)


def build_revision_prompt(
    *,
    iteration: int,
    review_verdict: str,
    spec_json: str,
    revision_context: str,
    issue_history_context: str,
    human_hint_context: str,
    review_json: str,
    current_script: str,
) -> str:
    return f"""当前是第 {iteration} 轮修订。你不会看到图片，只依据 review JSON 改代码。

本轮 verdict: {review_verdict}
如果 verdict 是 `retake_views`，只能改相机位置、目标、镜头和诊断视角定义；不得改几何、材质、尺寸、刻度或标签。
如果 verdict 是 `revise`，只根据 review 里列出的 `overall_shape` / `components_shape` / `graduations` 问题改代码。

目标规格:
{spec_json}

{revision_context}

{issue_history_context}

{human_hint_context}

上一轮视觉评审 JSON:
```json
{review_json}
```

当前精确仪器脚本:
```python
{current_script}
```

请输出完整修正后的 Python 文件。"""


def build_issue_history_context(issue_history: Sequence[HistoricalVisualIssue]) -> str:
    """跨轮问题记忆块：把此前 moderate / major / critical 问题序列化为回归检查清单。"""
    payload = json.dumps(
        [item.model_dump(mode="json") for item in issue_history],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""修改历史中的 moderate 及以上问题:
{payload}
保留已确认的修复，如有处理反复出现的异常请确认成因，避免无效修改。
"""


def build_human_hint_context(human_hint: str | None) -> str:
    """人工提示块：--human-hint 激活时插入评审/修复用户提示。"""
    if not human_hint or not human_hint.strip():
        return "本轮人工指导:\n(无)"
    return f"""本轮人工指导:
{human_hint.strip()}
务必重视此指导，此指导高于一切评审生成的review。
"""


def build_review_prompt(
    *,
    iteration: int,
    image_labels: list[str],
    pass_score: float,
    spec_json: str,
    revision_context: str,
    issue_history_context: str,
    human_hint_context: str,
    reference_guidance: str = "",
    script: str,
) -> str:
    """评审 + 改代码的用户提示：迭代信息 + 规格 + 上下文 + 图片标签 + 精确脚本；图片另以 base64 附加。"""
    labels = "\n".join(image_labels)
    return f"""迭代: {iteration}
图片输入顺序与标签:
{labels}
通过阈值（由编排器配置）: {pass_score}/10

目标规格:
{spec_json}

{revision_context}

{issue_history_context}

{human_hint_context}

{reference_guidance}

生成本轮图片的当前精确仪器脚本:
```python
{script}
```
"""

# ===========================================================================
# 渲染失败修复（code_writer.CodeWriter.repair_render_failure）
# 注入时机：静态校验或 Blender 执行失败时，给 iterative_coder 的修复请求。
# ===========================================================================

REPAIR_SYSTEM_PROMPT = (
    """你是 Blender 5.2 Python 工程师，负责修复未通过校验或执行失败的脚本。诊断精确脚本和错误日志，
做最小且稳健的修复，并保留正确几何。只返回以下部分：

<BLENDER_SCRIPT>
完整修正后的可执行 Python 文件，不是补丁。
</BLENDER_SCRIPT>

"""
    + _ASSET_GUIDANCE
    + "\n\n"
    + COMMON_SAFETY
)


def build_repair_context(*, rules: str, toolkit: str) -> str:
    """修复用户提示的上下文块：契约 + 工具库（无文档，保证最小上下文）。"""
    return f"""脚本契约:
{rules}

共享工具库:
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
    return f"""当前脚本在第 {iteration} 轮确定性校验或 Blender 执行中失败。没有可用的渲染图，因此直接诊断代码和日志。

目标规格:
{spec_json}

{repair_context}

{issue_history_context}

{human_hint_context}

{reference_guidance}

当前精确脚本:
```python
{current_script}
```

失败证据:
```
{error}
```

做最小且稳健的修复，并保留正确几何。
"""
