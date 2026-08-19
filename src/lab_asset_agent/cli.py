from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config, load_spec
from .pyrex_spec import load_pyrex_line, pyrex_record_to_spec

app = typer.Typer(no_args_is_help=True, help="Iterative Blender laboratory asset generation agent")
console = Console()


@app.command()
def generate(
    spec: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", exists=True, dir_okay=False),
    human_hint: str | None = typer.Option(
        None,
        "--human-hint",
        help="Human guidance injected into eligible GPT review/repair iterations.",
    ),
    human_hint_from_iteration: int = typer.Option(
        1,
        "--human-hint-from-iteration",
        min=1,
        help="First iteration that receives --human-hint.",
    ),
) -> None:
    """Generate one instrument through initial model -> Blender -> GPT review+code iterations."""
    cfg = load_config(config)
    instrument = load_spec(spec)
    from .orchestrator import AssetGenerationOrchestrator

    orchestrator = AssetGenerationOrchestrator(
        cfg,
        console,
        human_hint=human_hint,
        human_hint_from_iteration=human_hint_from_iteration,
    )
    manifest = asyncio.run(orchestrator.run(instrument))
    console.print_json(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False))
    if manifest.status != "passed":
        raise typer.Exit(code=2)


@app.command("gen_pyrex")
def gen_pyrex(
    source: Path = typer.Option(
        Path("desc_dataset/labware_dataset/assets_major.jsonl"),
        "--source",
        "-s",
        exists=True,
        dir_okay=False,
        help="JSONL catalog dataset containing the asset line.",
    ),
    line: int = typer.Option(
        1,
        "--line",
        "-l",
        min=1,
        help="1-indexed line in the JSONL source to generate.",
    ),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", exists=True, dir_okay=False),
    human_hint: str | None = typer.Option(
        None,
        "--human-hint",
        help="Human guidance injected into eligible GPT review/repair iterations.",
    ),
    human_hint_from_iteration: int = typer.Option(
        1,
        "--human-hint-from-iteration",
        min=1,
        help="First iteration that receives --human-hint.",
    ),
) -> None:
    """Generate one asset from a 1-indexed line of a JSONL catalog dataset."""
    cfg = load_config(config)
    record = load_pyrex_line(source, line)
    instrument = pyrex_record_to_spec(record, base_dir=source.expanduser().resolve().parent)
    console.print(f"[cyan]Catalog line {line}[/cyan]: {instrument.name} (id={instrument.id})")
    from .orchestrator import AssetGenerationOrchestrator

    orchestrator = AssetGenerationOrchestrator(
        cfg,
        console,
        human_hint=human_hint,
        human_hint_from_iteration=human_hint_from_iteration,
    )
    manifest = asyncio.run(orchestrator.run(instrument))
    console.print_json(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False))
    if manifest.status != "passed":
        raise typer.Exit(code=2)


@app.command()
def resume(
    run_dir: Path | None = typer.Argument(
        None,
        help="Run directory to resume. Omit it to resume the newest run.",
    ),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", exists=True, dir_okay=False),
    human_hint: str | None = typer.Option(
        None,
        "--human-hint",
        help="Override or add human guidance for the resumed run.",
    ),
    human_hint_from_iteration: int = typer.Option(
        1,
        "--human-hint-from-iteration",
        min=1,
        help="First iteration that receives a newly supplied --human-hint.",
    ),
) -> None:
    """Resume an interrupted run without repeating completed initial-generation work."""
    cfg = load_config(config)
    if run_dir is None:
        candidates = sorted(
            (path for path in cfg.paths.runs_dir.iterdir() if path.is_dir() and (path / "manifest.json").is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise typer.BadParameter(f"No resumable runs found in {cfg.paths.runs_dir}")
        run_dir = candidates[0]
        console.print(f"[cyan]Selected latest run[/cyan]: {run_dir}")
    else:
        run_dir = run_dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise typer.BadParameter(f"Run directory does not exist: {run_dir}")

    from .orchestrator import AssetGenerationOrchestrator

    orchestrator = AssetGenerationOrchestrator(
        cfg,
        console,
        human_hint=human_hint,
        human_hint_from_iteration=human_hint_from_iteration,
    )
    manifest = asyncio.run(orchestrator.resume(run_dir))
    console.print_json(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False))
    if manifest.status != "passed":
        raise typer.Exit(code=2)


@app.command()
def batch(
    specs_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", exists=True, dir_okay=False),
    continue_on_error: bool = typer.Option(True, help="Continue after a failed instrument."),
    human_hint: str | None = typer.Option(
        None,
        "--human-hint",
        help="Human guidance applied to every generated instrument.",
    ),
    human_hint_from_iteration: int = typer.Option(
        1,
        "--human-hint-from-iteration",
        min=1,
        help="First iteration that receives --human-hint.",
    ),
) -> None:
    """Generate every YAML specification in a directory, sequentially."""
    from .orchestrator import AssetGenerationOrchestrator

    cfg = load_config(config)
    spec_paths = sorted([*specs_dir.glob("*.yaml"), *specs_dir.glob("*.yml")])
    if not spec_paths:
        raise typer.BadParameter(f"No YAML specs found in {specs_dir}")

    table = Table("Spec", "Status", "Final score", "Reason")
    failed = False
    for spec_path in spec_paths:
        try:
            manifest = asyncio.run(
                AssetGenerationOrchestrator(
                    cfg,
                    console,
                    human_hint=human_hint,
                    human_hint_from_iteration=human_hint_from_iteration,
                ).run(load_spec(spec_path))
            )
            table.add_row(
                spec_path.name,
                manifest.status,
                "" if manifest.final_score is None else f"{manifest.final_score:.2f}",
                manifest.failure_reason or "",
            )
            failed = failed or manifest.status != "passed"
            if manifest.status != "passed" and not continue_on_error:
                break
        except Exception as exc:
            failed = True
            table.add_row(spec_path.name, "exception", "", str(exc))
            if not continue_on_error:
                break
    console.print(table)
    if failed:
        raise typer.Exit(code=2)


@app.command("ping")
def ping(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", exists=True, dir_okay=False),
    model: str = typer.Option(
        "iteration",
        "--model",
        "-m",
        help="Which configured endpoint to ping: 'iteration' (default) or 'initial'.",
    ),
) -> None:
    """Send a one-word hello to a configured API to verify connectivity."""
    cfg = load_config(config)
    if model == "iteration":
        model_cfg = cfg.models.iteration_agent
        label = "iteration_agent"
    elif model == "initial":
        model_cfg = cfg.models.initial_model
        label = "initial"
    else:
        raise typer.BadParameter("--model must be 'iteration' or 'initial'")

    from .openai_compatible import OpenAICompatibleClient

    async def _ping() -> str:
        completion = await OpenAICompatibleClient(model_cfg).chat(
            [{"role": "user", "content": "Say hello."}],
            stream_label=f"{label}: {model_cfg.model}",
        )
        return completion.text

    reply = asyncio.run(_ping())
    console.print(f"\n[green]Ping OK[/green]: {reply.strip()}")


@app.command("check-config")
def check_config(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Validate configuration and local paths without calling paid APIs."""
    cfg = load_config(config)
    table = Table("Item", "Value", "Status")
    table.add_row("Project root", str(cfg.project_root), str(cfg.project_root.exists()))
    table.add_row("Blender", cfg.blender.executable, "not probed")
    table.add_row("Toolkit", str(cfg.paths.toolkit), str(cfg.paths.toolkit.exists()))
    table.add_row("Reference", str(cfg.paths.reference), str(cfg.paths.reference.exists()))
    table.add_row("Docs", str(cfg.paths.docs_dir), str(cfg.paths.docs_dir.exists()))
    table.add_row("Rules", str(cfg.paths.rules), str(cfg.paths.rules.exists()))
    initial = cfg.models.initial_model
    table.add_row(
        "Initial generator",
        f"{cfg.models.initial_generator}: {initial.model} @ {initial.base_url}",
        f"env {initial.api_key_env}: "
        + ("set" if os.getenv(initial.api_key_env) else "missing"),
    )
    agent = cfg.models.iteration_agent
    table.add_row(
        "GPT iteration agent",
        f"{agent.model} @ {agent.base_url}",
        f"env {agent.api_key_env}: "
        + ("set" if os.getenv(agent.api_key_env) else "missing"),
    )
    console.print(table)


if __name__ == "__main__":
    app()
