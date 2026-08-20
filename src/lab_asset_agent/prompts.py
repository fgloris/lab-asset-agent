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
        "这是 overall_shape 评分与是否 revise 的首要依据。",
    ]
    if extra_count:
        lines.append(
            f"随后附带的 {extra_count} 个辅助渲染视角仅用于补充覆盖信息，不参与对齐目标。"
        )
    lines.append("只对齐结构和比例；忽略照片的光照、背景、反射和商业修饰，不要把商标画上。")
    return "\n".join(lines)


# ===========================================================================
# 首版脚本生成（code_writer.CodeWriter）
# 注入时机：generate 开始时，initial_generator 调用一次。
# ===========================================================================

INITIAL_WRITER_SYSTEM_PROMPT = (
    """你是 Blender 5.2 Python 工程师。严格遵守脚本契约，并只返回以下两部分：

<BLENDER_SCRIPT>
一个完整可执行的 Python 文件，不要使用 Markdown 代码围栏。
</BLENDER_SCRIPT>
<SUMMARY>
一段简洁的设计摘要（中文）。
</SUMMARY>

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
# 评审 + 改代码（vision_coding_agent.VisionCodingAgent.review_and_revise）
# 注入时机：每次 Blender 成功渲染后，给 iterative_generator 的多模态请求。
# ===========================================================================

REVIEW_SYSTEM_PROMPT = (
    """你是视觉质检工程师兼 Blender 5.2 Python 工程师。联合评判所有提供的渲染图、目标规格和精确脚本。
报告可执行的 `minor`、`moderate`、`major`、`critical` 问题；省略外观偏好。每个问题必须使用且只能使用一个 review_axis：
- `camera_coverage`：可见性门槛，如需要调整则会应优先执行。首先确保前面的渲染视角必须分别与对应顺序的目标参考图一致，然后检查整体覆盖、有效角度多样性和可读比例。
若视角存在不足，选择`retake_views`；不要因为物体没有拍全，就认为其有几何缺失。
- `overall_shape`：最重要的维度。只评估仪器整体体态、外轮廓、长宽高比例、颈/肩/腹/底等大尺度比例关系、主体姿态和整体部件布局。
  必须把每一张当前渲染图与对应/相关的参考产品图逐项对比，指出具体的整体比例、轮廓或体态偏差。
- `component_shape`：评估局部部件几何和连接关系。检查开口、口沿(向上还是向下)、壁厚、底部、接头、侧面部件、把手/塞子/阀门/管嘴等局部结构、
  物理连接和拓扑。不要把整体高矮胖瘦问题放到此轴；那应归入 `overall_shape`。
- `graduations`：检查可见刻度/标签/附着，以及精确的体积积分代码，包括真实零体积原点。非均匀容器中等体积刻度间距不均是正常现象。

similarity_score 如何计算：
- 逐对对比参考图和对应视角，从 0-5 为外形/比例/轮廓的相似度打分(可用小数)，得到 similarity_scores。
- 评分锚点：0-1：完全不是同一种仪器或主体不可识别，参考图和渲染图完全对不上；1-2：仪器类别正确但整体轮廓、主要部件或比例明显错误；
  2-3：总体比例/轮廓有偏差，部分关键部件缺失；3-4：仪器轮廓和部件比例大致一致，参考图和渲染图语义上相似，没有关键部件缺失；
  4-5：仪器轮廓和部件比例一致，参考图和渲染图的轮廓可以在像素级别对比，但关键部位的转折、包含关系等细节仍有缺陷；5：渲染图与参考图在形态、比例、轮廓和关键结构上完全一致。
- 不要因为材质、透明度、曝光、阴影或背景风格相似而提高分数；评分主要依据几何形态、比例、轮廓和关键功能结构。
- similarity_score 取 similarity_scores 的算术平均值；没有参考图时，按渲染图与仪器文本描述的一致性给出 0-10 分。

决策规则：
- `retake_views`：仅修改相机位置、目标/镜头和诊断视角定义。几何、材质、标识、尺寸、刻度计算必须原样保留。
- `revise`：视角覆盖充分且至少存在一个 moderate 或以上 overall_shape/component_shape/graduations 问题需要修复；返回完整修正后的脚本。
- `pass`：覆盖充分且不再存在 moderate 及以上缺陷。

覆盖充分时，overall_shape 重要性最高，其次是 component_shape，graduations 用于补充判定。忽略暗部、弱反射/高光、表观透明度、曝光、
对比度、阴影以及其他光照/渲染风格差异；但不得用这些风格理由掩盖真实几何差异。绝不要通过修改相机或渲染
来掩盖真实缺陷。

输出格式必须严格如下（标签名 <DIFFERENCE>、<REVIEW_JSON> 和 <BLENDER_SCRIPT> 保持英文，不要使用 Markdown 代码围栏）：

<DIFFERENCE>
先显式对比目标参考图片与当前渲染结果。必须点名主要相同点、主要差异、差异涉及的视角/参考图；不要写成泛泛总结。
</DIFFERENCE>
<REVIEW_JSON>
一个合法 JSON 对象。字段名(key)和枚举值必须完全使用下面的英文，不得翻译或改写；字段内容(value)可用中文。
- `verdict`: `pass` | `revise` | `retake_views`
- `similarity_scores`: 0 到 5 的小数数组，长度与参考图数量相同
- `similarity_score`: 0 到 5 的小数
- `issues`: 数组，每项字段为 `review_axis`、`severity`、`observation`、`likely_cause`、`recommended_change`
  - `review_axis`: `camera_coverage` | `overall_shape` | `component_shape` | `graduations`
  - `severity`: `moderate` | `major` | `critical`
  - `observation`、`likely_cause`、`recommended_change`: 字符串，可用中文
- `preserve`: 字符串数组，列出需要保留的正确内容
</REVIEW_JSON>
<BLENDER_SCRIPT>
当 verdict 为 revise 或 retake_views 时，本段为完整可执行的 Python 文件。pass 时省略本段。
</BLENDER_SCRIPT>

"""
    + _ASSET_GUIDANCE
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

人类指导（非常重要，请严格遵循）：
{human_hint_context}

{reference_guidance}

生成本轮图片的当前精确仪器脚本:
```python
{script}
```
"""

# ===========================================================================
# 渲染失败修复（vision_coding_agent.VisionCodingAgent.repair_render_failure）
# 注入时机：静态校验或 Blender 执行失败时，给 iterative_generator 的修复请求。
# ===========================================================================

REPAIR_SYSTEM_PROMPT = (
    """你是 Blender 5.2 Python 工程师，负责修复未通过校验或执行失败的脚本。诊断精确脚本和错误日志，
做最小且稳健的修复，并保留正确几何。只返回以下两部分：

<SUMMARY>
简洁的原因和修复摘要（可用中文）。
</SUMMARY>
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
