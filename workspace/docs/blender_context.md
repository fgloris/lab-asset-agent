# 面向生成 agent 的 Blender 5.2 API精简上下文

本项目在后台模式运行 Blender 并执行一个 Python 文件。仪器脚本负责场景创建、渲染路径以及保存 `.blend` 文件。

## 总体API情况

- 提供的工具库明确要求 Blender 5.2，请使用与之对应的API。
- 工具库几何以 Blender 米为单位；使用 `lab.mm(value)` 表示毫米。
- `create_hollow_revolved_mesh` 根据外部与内部纵向轮廓构建封闭空心容器，并支持诸如倒液嘴的角变形。
- `smooth_profile(profile, samples_per_segment=8, sharp_indices=...)` 执行保形 PCHIP 轮廓插值。`smooth_profile_from_mm(...)` 结合毫米解析与平滑。网格构建、容量和刻度必须使用同一个平滑后的内部轮廓；用 `sharp_indices` 保留刻意的底部/口沿/接头拐角。
- `add_volume_graduations` 根据内部容积轮廓计算刻度高度，然后把刻度和标签包裹到实际外表面。
- `add_volume_graduations` 接受正的整毫升值，包括整数浮点数如 `250.0`；当规格为整数时，生成脚本中优先使用整数字面量。
- `configure_scene`、`create_grid_floor`、`create_camera`、`setup_glass_product_lighting`、`render_views` 和 `save_blend` 组成标准渲染管线。
- `enable_freestyle_outline()` 可选地在渲染图中叠加可见轮廓/开放边界。它是几何可见性的诊断辅助，不是几何修复，也不是干净效果渲染的必需项。
- 参考烧杯演示了预期组织方式，但新仪器可能需要为瓶颈、支管、把手、塞子、接头或非轴对称部件编写局部辅助函数。

## blender headless 执行约束

- 不要依赖 UI 上下文、活动编辑器区域或手动模式切换。
- 优先使用 Blender 数据 API 和显式对象链接，而不是依赖上下文的操作符。
- 必须使用操作符时，确定性地设置选中对象和活动对象。
- 输出路径必须为绝对路径，或从 `Path(__file__)` / `LAB_ASSET_OUTPUT_DIR` 推导。
- 成功脚本必须在不开 Blender GUI 的情况下生成多张 PNG 视角和一个 `.blend` 文件。

可将额外的 Blender HTML/文本说明放在本目录。代码生成模型会把这些文件作为不可变上下文接收。

## 平滑轮廓示例

```python
OUTER_PROFILE = lab.smooth_profile_from_mm(
    [
        (34.0, 0.0),
        (35.0, 4.0),
        (43.0, 55.0),
        (18.0, 92.0),
        (18.0, 110.0),
    ],
    samples_per_segment=10,
    sharp_indices={0, 1, 4},
)
```

对曲线腹部、肩部和瓶颈过渡使用平滑。不要对需要尖锐的口沿、底角或磨砂接头进行平滑。

## 轮廓线渲染示例

此功能可以使渲染出的图像更加清晰。在 `configure_scene(...)` 之后、渲染之前调用：

```python
lab.enable_freestyle_outline(
    thickness_px=2.5,
    include_open_borders=True,
    include_creases=False,
)
```

## 文件路径相关

生成的文件写入运行目录（`runs/<run-id>/candidate.py`）。不可变工具库位于 `workspace/toolkit/`；通过 `LAB_TOOLKIT_DIR` 定位它，不要假设当前工作目录：

```python
import importlib
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_DIR = Path(os.getenv("LAB_TOOLKIT_DIR", str(SCRIPT_DIR.parent / "toolkit"))).resolve()
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

import lab_blender_toolkit as lab
importlib.reload(lab)

NAME = "Stable_Asset_Name"
OUTPUT_DIR = Path(
    os.getenv("LAB_ASSET_OUTPUT_DIR", str(SCRIPT_DIR / "output" / NAME))
).resolve()
RENDER_ENGINE = os.getenv("LAB_RENDER_ENGINE", "BLENDER_EEVEE")
RESOLUTION = int(os.getenv("LAB_RENDER_RESOLUTION", "768"))
```

把 `RENDER_ENGINE` 和 `RESOLUTION` 传给 `lab.configure_scene`，并把 `OUTPUT_DIR` 同时传给 `lab.render_views` 和 `lab.save_blend`。设置 `LAB_ASSET_OUTPUT_DIR` 时不要向其他位置写文件。
