# Lab Asset Agent v0.4.1

一个在本地驱动 Blender 5.2、通过 OpenAI-compatible API 迭代生成实验室仪器 3D 资产的轻量 agent。

流程：先让GPT生成初始脚本，后续每一轮由同一个 GPT 迭代模型在同一次请求里同时读取：目标规格、共享工具库、项目文档、**产生本轮渲染图的精确脚本快照**，以及多个 Blender 渲染视角，然后完成视觉评审并直接输出下一版完整 Blender Python 脚本。

```text
仪器 YAML
   │
   ├── 初始脚本：DeepSeek（可选）或 GPT
   ▼
本地静态检查 → Blender 后台建模与多视角渲染
   ▼
GPT：规格 + 工具代码 + 当前精确代码 + 多视角图片
   │
   ├── pass：保存 final
   ├── revise：同一响应直接返回完整下一版脚本
   └── retake_views：只改摄像机/诊断视角，重新拍一组图片
                              │
                              └────────→ 下一轮 Blender
```

运行时不依赖 Claude Code、Claude Agent SDK、Qwen3-VL 或 vLLM。

## 命令一览

入口统一为 `lab-asset-agent`（`pip install -e .` 后可用），也可用 `python -m lab_asset_agent.cli` 代替。

| 命令 | 作用 | 最小示例 |
| --- | --- | --- |
| `check-config` | 不调用付费 API，校验配置、路径与密钥 | `lab-asset-agent check-config -c config.yaml` |
| `ping` | 发送一句 hello，验证指定模型源是否可联通（会消耗少量 token） | `lab-asset-agent ping vectorengine -c config.yaml` |
| `generate` | 生成单个仪器（初始模型 → Blender → GPT 迭代） | `lab-asset-agent generate desc_dataset/specs/erlenmeyer_250ml.yaml` |
| `resume` | 续跑中断的 run，不重复已完成的首版生成 | `lab-asset-agent resume` |
| `batch` | 顺序批量生成一个目录下所有 YAML 规格 | `lab-asset-agent batch desc_dataset/common_chemistry_instruments_yaml` |
| `gen_pyrex` | 读取目录数据集 JSONL 某一行，带产品参考图生成 | `lab-asset-agent gen_pyrex -s desc_dataset/labware_dataset/assets_major.jsonl -l 1` |

除 `check-config` 外，其余命令在结果未通过时以退出码 `2` 结束。

## 1. 安装

建议使用 Python 3.11 / 3.12：

```bash
cd lab_asset_agent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

设置密钥（复制 `.env.example`，或直接在 shell 中导出）：

```bash
export VECTOR_ENGINE_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."   # 仅当 initial_generator 指向 deepseek 时需要
```

## 2. 配置

`config.yaml` 的核心分段：

```yaml
blender:
  executable: "blender"
  timeout_seconds: 900
  render_engine: BLENDER_EEVEE
  resolution: 768
  minimum_render_count: 3

models:
  initial_generator: vectorengine   # 首版脚本由哪个模型源生成
  iterative_generator: vectorengine # 之后评审/改代码/修复由哪个模型源执行，必须 vision: true

  # 所有模型源共享的连接/流式/图片设置
  stream: true
  stream_to_terminal: true
  stream_reasoning: progress
  max_retries: 8
  connect_timeout_seconds: 120
  request_timeout_seconds: 900
  max_image_side: 1280
  extra_images: 2
  jpeg_quality: 90

  providers:
    vectorengine:
      base_url: "https://api.vectorengine.ai/v1"
      api_key_env: VECTOR_ENGINE_API_KEY
      model: gpt-5.6-luna
      vision: true
    deepseek:
      base_url: "https://api.deepseek.com"
      api_key_env: DEEPSEEK_API_KEY
      model: deepseek-reasoner
      vision: false

loop:
  max_iterations: 8
  pass_score: 8.5
  issue_history_window: 5
  keep_all_iterations: true
  stop_on_repeated_script: true
  max_consecutive_render_failures: 3

paths:
  toolkit: workspace/toolkit/lab_blender_toolkit.py
  reference: workspace/references/beaker_low_250ml_reference.py
  docs_dir: workspace/docs
  rules: AGENT_RULES.md
  runs_dir: runs
```

要点：

- **只有第一版脚本**受 `initial_generator` 控制。首轮渲染之后，评审、改代码、修复 Blender 报错全部由 `iterative_generator` 完成。
- 每个 `providers` 条目是一个模型源；`vision: true` 表示该模型支持图像输入，只有支持视觉的模型源才能作为 `iterative_generator`。
- `stream` / `stream_to_terminal` / `stream_reasoning` 以及连接、超时、图片尺寸等是共享设置，放在 `models` 外层，自动应用到所有模型源；如需单独覆盖，也可在某个 provider 下重写。
- 工具库和参考脚本是**受保护文件**，运行结束会自动校验并恢复，不会被模型改动。

## 3. 检查配置：check-config

不调用任何付费 API，打印项目路径、Blender、模型路由与所需环境变量是否存在：

```bash
lab-asset-agent check-config -c config.yaml
```

示例输出：

```text
| Item                | Value                    | Status |
| Project root        | /home/.../lab_asset_agent   | True   |
| Blender             | blender                  | not probed |
| Toolkit             | /home/.../lab_blender_toolkit.py | True |
| Model vectorengine (initial, iterative) | gpt-5.6-luna @ ... (vision=True) | env VECTOR_ENGINE_API_KEY: set |
| Model deepseek                           | deepseek-reasoner @ ... (vision=False) | env DEEPSEEK_API_KEY: set |
```

### 3.1 测试 API 连通性：ping

`ping` 会向指定模型源发送一句 `Say hello.`，并打印返回内容，用于验证密钥与网络是否可用。这会消耗少量 token。传入 `providers` 里的模型源名即可：

```bash
lab-asset-agent ping vectorengine -c config.yaml
lab-asset-agent ping deepseek -c config.yaml
```

省略模型名时，默认测试 `iterative_generator` 指向的模型源：

```bash
lab-asset-agent ping -c config.yaml
```

## 4. 生成单个仪器：generate

```bash
lab-asset-agent generate desc_dataset/specs/erlenmeyer_250ml.yaml -c config.yaml
```

或烧杯示例：

```bash
lab-asset-agent generate desc_dataset/specs/beaker_low_250ml.yaml
```

`-c config.yaml` 可省略（默认就是 `config.yaml`）。

运行过程中：

```text
Run created: runs/20260802T..._erlenmeyer_250ml
Calling initial generator: gpt-5.6-luna (route=vectorengine)...
[lab-asset-agent] --- response stream ---
<BLENDER_SCRIPT>...脚本逐步出现...</BLENDER_SCRIPT>
Initial script saved: runs/20260802T..._erlenmeyer_250ml/candidate.py
──────────────── Iteration 1 ────────────────
Starting Blender...
Blender finished in 42.1s; images=3, success=True
Calling review+coder: gpt-5.6-luna with exact code and 3 image(s)...
<REVIEW_JSON>...逐步出现...</REVIEW_JSON>
<BLENDER_SCRIPT>...下一版完整代码...</BLENDER_SCRIPT>
Visual score: 7.80/10, verdict=revise
```

### 4.1 注入人工提示

在生成过程中，你可以插入一条人工意见，agent 将在每一轮评审与修复时带着这条意见继续整个流程：

```bash
lab-asset-agent generate desc_dataset/specs/beaker_low_250ml.yaml \
  --human-hint "倒液嘴应更突出，但不要改变杯身直径"
```

## 5. 续跑：resume

任务中断后不要重新执行 `generate`。省略参数时自动续跑最新的 run：

```bash
lab-asset-agent resume -c config.yaml
```

指定任务目录：

```bash
lab-asset-agent resume runs/20260731T135845827504Z_burette_50ml
```

续跑时加入一条新的人工提示（覆盖原 run 记录的提示）：

```bash
lab-asset-agent resume \
  runs/20260731T183050365491Z_separatory_funnel_250ml \
  --human-hint "瓶颈看起来偏粗，优先核对真实容量瓶的颈身比例"
```

回退到指定迭代并丢弃其后的记录后继续（用该迭代的脚本快照恢复 `candidate.py`）：

```bash
lab-asset-agent resume \
  runs/20260731T183050365491Z_separatory_funnel_250ml \
  --from-iteration 4
```

恢复规则：

- 已生成首版、未渲染：直接运行 Blender；
- 已渲染但 GPT 调用失败：复用脚本和图片，重新调用 GPT；
- GPT 已写出下一版、尚未渲染：直接渲染，不重复 GPT 调用；
- 上轮已通过：直接整理 `final/`。

## 6. 批量生成：batch

顺序批量生成目录下所有 `*.yaml` / `*.yml`，每个 spec 独立创建 run，结束后打印汇总表：

```bash
lab-asset-agent batch desc_dataset/common_chemistry_instruments_yaml -c config.yaml
```

默认失败后继续下一个；改为遇到失败立即停止：

```bash
lab-asset-agent batch desc_dataset/common_chemistry_instruments_yaml --no-continue-on-error
```

给每个仪器注入同一条人工提示：

```bash
lab-asset-agent batch desc_dataset/common_chemistry_instruments_yaml \
  --human-hint "刻度数字必须清晰可读"
```

## 7. 规格文件格式

`generate` 和 `batch` 读入的 YAML 规格（示例：`desc_dataset/specs/beaker_low_250ml.yaml`）：

```yaml
id: beaker_low_form_250ml          # ^[a-z0-9][a-z0-9_-]*$，同时决定 run 目录名
name: 250 mL low-form glass beaker
description: |
  A standard low-form laboratory beaker with a cylindrical body, slightly flared rim,
  a single clean pouring spout, a flat thick base, and white volume graduations.
  Nominal capacity 250 mL.
specs:
  Approximate Capacity (mL): "250"
  Approximate Outside Diameter x Height (mm): "74 x 88"
  Graduation Increment (mL): "10"
```

说明：

- 容量、尺寸、刻度等异质规格直接放进 `specs`（键名自带单位），不强制固定字段。
- `reference_images`（可选）：目录产品照片路径列表，随 prompt 作为参考图传给模型（见第 10 节 `gen_pyrex`）。
- 材质系统、结构合理性、渲染标准等共享约束由提示词统一注入，不写进单个规格（见第 9 节）。

规格文件所在位置：

- `desc_dataset/specs/`：烧杯、锥形瓶
- `desc_dataset/common_chemistry_instruments_yaml/`：量筒、容量瓶、试管、圆底烧瓶、分液漏斗、碱式滴定管、移液管、广口试剂瓶、表面皿、漏斗等 10 个

## 8. 每轮与最终产物

```text
runs/<run-id>/
├── candidate.py                         # 当前工作脚本，每轮被下一版覆盖
├── candidate.initial_response.txt       # 首版模型的原始响应
├── spec.json
├── manifest.json
├── issue_history.json                  # 全部历史 moderate / major / critical 问题（跨轮归档）
├── iteration_01/
│   ├── instrument.py                     # 产生本轮图片的精确脚本快照
│   ├── render/
│   │   ├── *.png
│   │   ├── *.blend
│   │   └── blender.log
│   ├── review.json
│   ├── gpt_review_and_code_response.txt  # GPT 原始合并响应
│   ├── next_instrument.py                # revise / retake_views 的下一版
│   └── repair_agent_response.txt         # 渲染失败时才出现
├── iteration_02/
└── final/                                # 通过时：instrument.py + render/
```

评审响应协议：`<REVIEW_JSON>` 内为 `verdict`（`pass` / `revise` / `retake_views`）、`similarity_score`、`issues` 等；`revise` 和 `retake_views` 必须同时携带完整 `<BLENDER_SCRIPT>`。`retake_views` 只允许 `camera_coverage` 问题，且返回脚本只能修改摄像机、目标点、镜头与诊断视角。

## 9. Prompt 注入点

所有发给模型的 prompt 集中在一个文件，统一修改只动这一个文件，无需触碰 agent 代码：

```text
src/lab_asset_agent/prompts.py
```

各 prompt 的名称、作用与注入时机如下。

### 首版脚本生成（每个 run 只发生一次，generate 开始时）

| Prompt | 角色 | 注入时机 |
| --- | --- | --- |
| `INITIAL_WRITER_SYSTEM_PROMPT` | system | 第一版脚本调用（`CodeWriter`），由 `initial_generator` 指定的模型源执行 |
| `build_shared_context()` | user 片段 | 脚本契约 + 项目文档 + 参考脚本 + 共享工具库 |
| `build_initial_prompt()` | user | 目标规格 + 上述上下文 + 生成脚本路径 |

### 评审 + 改代码（每轮成功渲染后）

| Prompt | 角色 | 注入时机 |
| --- | --- | --- |
| `REVIEW_SYSTEM_PROMPT` | system | 每次成功渲染后的 GPT 评审 + 改写请求（`VisionCodingAgent.review_and_revise`），要求返回 `<REVIEW_JSON>` 与完整 `<BLENDER_SCRIPT>` |
| `build_revision_context()` | user 片段 | 脚本契约 + 项目文档 + 共享工具库 |
| `build_issue_history_context()` | user 片段 | 此前最近 `loop.issue_history_window` 条 moderate / major / critical 问题的回归清单（跨轮记忆） |
| `build_human_hint_context()` | user 片段 | 每一轮注入的 `--human-hint` |
| `build_review_prompt()` | user | 迭代号、视角文件名、通过阈值、规格 + 上述片段 + 产生本轮图片的精确脚本；多视角图片随后以 JPEG base64 追加 |

### 渲染失败修复（静态校验或 Blender 执行失败时）

| Prompt | 角色 | 注入时机 |
| --- | --- | --- |
| `REPAIR_SYSTEM_PROMPT` | system | 渲染失败后的修复请求（`VisionCodingAgent.repair_render_failure`），要求返回 `<SUMMARY>` 与完整 `<BLENDER_SCRIPT>` |
| `build_repair_context()` | user 片段 | 脚本契约 + 共享工具库（不携带文档，保持最小上下文） |
| `build_issue_history_context()` / `build_human_hint_context()` | user 片段 | 同评审流程 |
| `build_repair_prompt()` | user | 目标规格 + 失败脚本 + Blender 错误日志 |

### 共享

| Prompt | 角色 | 注入时机 |
| --- | --- | --- |
| `COMMON_SAFETY` | system 后缀 | 追加在 `REVIEW_SYSTEM_PROMPT` 与 `REPAIR_SYSTEM_PROMPT` 末尾的安全约束 |

## 10. 配套工具：PDF 目录数据集提取

`desc_dataset/extract_labware_dataset_v2.py` 把产品目录 PDF 拆成图片 + 文本数据集（家庭/变体/资产三级 JSONL），依赖 `pymupdf`：

```bash
pip install pymupdf
python desc_dataset/extract_labware_dataset_v2.py CLS-GL-001.pdf -o labware_dataset
```

常用参数：

```text
-o, --output    输出目录（默认 labware_dataset）
--start-page N  从第 N 页开始（默认跳过目录页）
--end-page N    处理到第 N 页（默认停在 Technical Information 之前）
--render-dpi N  裁剪图渲染 DPI（默认 240）
--max-pages N   调试用：最多处理 N 页
```

### 10.1 从数据集生成：gen_pyrex

`gen_pyrex` 读取 `assets_major.jsonl` 的某一行（1-indexed），把该行的 `conditioning_text` 转成描述、`geometry_conditioning_specs` 转成 `specs`、`all_images` 中磁盘上真实存在的产品图作为参考图，走完整管线：

```bash
lab-asset-agent gen_pyrex -s desc_dataset/labware_dataset/assets_major.jsonl -l 2
```

常用参数：

```text
-s, --source    数据集 JSONL 路径（默认 desc_dataset/labware_dataset/assets_major.jsonl）
-l, --line      要生成的行号（1-indexed，默认 1）
-c, --config    config.yaml 路径
--human-hint    注入一条人工提示（作用于每一轮）
```

参考图注入到 GPT 首版/评审/修复；DeepSeek 首版生成路由不发送图片（不支持视觉），参考图仍进入 GPT 评审与修复。

## 11. 安全边界

生成脚本在启动 Blender 前会做 AST 静态检查，拒绝：

- 导入 `subprocess`、`socket`、`requests`、`httpx` 等；
- 调用 `eval`、`exec`、`compile`、`__import__`；
- 调用 `os.system`、`os.remove`、`shutil.rmtree` 等破坏性操作；
- 不满足 `build_asset()`、`if __name__ == "__main__"` 主入口、输出目录与渲染契约的脚本。

Blender 使用 `--background --factory-startup --offline-mode --python-exit-code 1`。这属于防御性限制，不等同于操作系统级沙箱，大规模运行建议使用独立用户、虚拟机或容器。

## 12. 测试

```bash
pytest -q
```

单独验证参考脚本的 Blender 后台渲染：

```bash
project="$(pwd)"
blender \
  --background --factory-startup --offline-mode \
  --python-use-system-env --python-exit-code 1 \
  --python "$project/workspace/references/beaker_low_250ml_reference.py"
```
