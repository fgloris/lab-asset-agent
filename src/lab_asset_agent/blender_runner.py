from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .models import AppConfig, RenderResult
from .utils import tail_text
from .validator import validate_generated_script


class BlenderRunner:
    def __init__(self, config: AppConfig):
        self.config = config

    def render(self, script_path: Path, output_dir: Path) -> RenderResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "blender.log"
        validation = validate_generated_script(script_path)
        if not validation.ok:
            message = "Static validation failed:\n" + "\n".join(f"- {e}" for e in validation.errors)
            log_path.write_text(message, encoding="utf-8")
            return RenderResult(
                success=False,
                return_code=None,
                command=[],
                log_path=log_path,
                error_summary=message,
                elapsed_seconds=0.0,
            )

        command = [
            self.config.blender.executable,
            "--background",
            "--factory-startup",
            "--offline-mode",
            "--python-exit-code",
            "1",
            "--python",
            str(script_path),
        ]
        env = os.environ.copy()
        env.update(
            {
                "LAB_ASSET_OUTPUT_DIR": str(output_dir.resolve()),
                "LAB_RENDER_ENGINE": self.config.blender.render_engine,
                "LAB_RENDER_RESOLUTION": str(self.config.blender.resolution),
                "LAB_TOOLKIT_DIR": str(self.config.paths.toolkit.parent.resolve()),
                "PYTHONUTF8": "1",
            }
        )

        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(script_path.parent),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.blender.timeout_seconds,
                check=False,
            )
            output = completed.stdout or ""
            return_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + "\nBLENDER TIMEOUT"
            return_code = None
        except OSError as exc:
            output = f"Failed to start Blender: {exc}"
            return_code = None

        elapsed = time.monotonic() - start
        log_path.write_text(output, encoding="utf-8")
        images = sorted(output_dir.rglob("*.png"))
        blend_files = sorted(output_dir.rglob("*.blend"))
        success = (
            return_code == 0
            and len(images) >= self.config.blender.minimum_render_count
            and bool(blend_files)
        )
        error_summary = None if success else tail_text(output)
        if return_code == 0 and len(images) < self.config.blender.minimum_render_count:
            error_summary = (
                f"Blender exited successfully but produced only {len(images)} PNG image(s); "
                f"expected at least {self.config.blender.minimum_render_count}.\n" + (error_summary or "")
            )
        if return_code == 0 and not blend_files:
            error_summary = "Blender produced no .blend file.\n" + (error_summary or "")

        return RenderResult(
            success=success,
            return_code=return_code,
            command=command,
            log_path=log_path,
            images=images,
            blend_files=blend_files,
            error_summary=error_summary,
            elapsed_seconds=elapsed,
        )
