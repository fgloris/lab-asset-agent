# Claude Code 项目说明

本仓库运行的是确定性的本地编排循环；Claude Code 仅用于维护代码、测试、提示词与 Blender 工具。

## 编排流程

1. 生成首个脚本：由 `initial_coder` 选定的模型源生成 `runs/<run_id>/candidate.py`。
2. 本地静态校验脚本，随后在后台/离线模式启动 Blender 5.2 执行。
3. Blender 输出多张 PNG 视角与一个 `.blend` 文件。
4. 视觉评审模型（`visual_reviewer`）接收目标规格、不可变工具库/文档、精确脚本快照和 JPEG 渲染图，只输出 review JSON。
5. 编排器按 review 选择：
   - `pass`：通过。
   - `revise`：交给 `iterative_coder` 返回完整修正脚本，进入下一轮。
   - `retake_views`：交给 `iterative_coder` 仅返回相机修改脚本，重拍视角。
   渲染失败时，`iterative_coder` 依据脚本与 Blender 日志修复。

## 维护约束

- 保持视觉评审和编码分离；视觉模型只产出 review JSON，写代码由 coder 完成。
- 不要为了通过资产而修改共享工具库或参考脚本。
- API key 放在环境变量。
- 人工提示保持简单明确，不添加交互提示、文件监听或后台控制通道。
