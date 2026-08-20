from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from rich.console import Console

from .blender_runner import BlenderRunner
from .code_writer import CodeWriter
from .models import (
    AppConfig,
    HistoricalVisualIssue,
    InstrumentSpec,
    IterationRecord,
    RunManifest,
    TokenUsage,
)
from .utils import sha256_file, utc_run_id, write_json
from .vision_coding_agent import VisionCodeDecision, VisionCodingAgent


class AssetGenerationOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        console: Console | None = None,
        *,
        human_hint: str | None = None,
    ):
        self.config = config
        self.console = console or Console()
        self.blender = BlenderRunner(config)
        self.iterative_agent = VisionCodingAgent(config)
        self.human_hint = human_hint.strip() if human_hint and human_hint.strip() else None

    async def run(self, spec: InstrumentSpec) -> RunManifest:
        """Start a new run.

        Only the initial script is routed through the configurable initial
        generator. After the first render, GPT reviews images and code and writes
        the next candidate in the same multimodal request.
        """

        self._validate_inputs()
        run_id = utc_run_id(spec.id)
        run_dir = self.config.paths.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        candidate_path = run_dir / "candidate.py"
        manifest = RunManifest(
            run_id=run_id,
            spec_id=spec.id,
            human_hint=self.human_hint,
        )
        manifest_path = run_dir / "manifest.json"
        write_json(run_dir / "spec.json", spec)
        write_json(manifest_path, manifest)
        write_json(run_dir / "issue_history.json", [])
        self._copy_reference_images(spec, run_dir)

        self.console.print(f"[cyan]Run created[/cyan]: {run_dir}")
        self._print_human_hint()
        protected = self._snapshot_protected_files()
        initial_agent = CodeWriter(self.config, self.config.models.initial_model)
        initial_started = False
        agent_started = False
        try:
            await initial_agent.start()
            initial_started = True
            route = self.config.models.initial_generator
            model = self.config.models.initial_model.model
            self.console.print(
                f"[cyan]Calling initial generator[/cyan]: {model} (route={route})..."
            )
            writer_summary, initial_usage = await initial_agent.create_initial(
                spec, candidate_path, reference_images=spec.reference_images
            )
            manifest.token_usage.add(initial_usage)
            write_json(manifest_path, manifest)
            self.console.print(f"[green]Initial script saved[/green]: {candidate_path}")
            self._restore_protected_files(protected)

            await self.iterative_agent.start()
            agent_started = True
            return await self._iterate(
                spec=spec,
                run_dir=run_dir,
                manifest=manifest,
                candidate_path=candidate_path,
                writer_summary=writer_summary,
                start_iteration=1,
                repeated_hashes=set(),
                consecutive_render_failures=0,
                protected=protected,
            )
        except Exception as exc:
            self._mark_failed(manifest, manifest_path, exc)
            raise
        finally:
            if initial_started:
                await initial_agent.close()
            if agent_started:
                await self.iterative_agent.close()
            self._restore_protected_files(protected)

    async def resume(self, run_dir: Path, from_iteration: int | None = None) -> RunManifest:
        """Resume without repeating a completed initial-generation request.

        v0.2 manifests are supported. If an old run has renders and a separate
        review but no revised candidate, the iterative generator receives the
        exact script plus those renders and performs review+revision in one call.
        """

        self._validate_inputs()
        run_dir = run_dir.expanduser().resolve()
        manifest_path = run_dir / "manifest.json"
        spec_path = run_dir / "spec.json"
        if not manifest_path.is_file() or not spec_path.is_file():
            raise FileNotFoundError(
                f"Resume directory must contain manifest.json and spec.json: {run_dir}"
            )

        spec = InstrumentSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.spec_id != spec.id:
            raise RuntimeError(
                f"Manifest spec_id={manifest.spec_id!r} does not match spec id={spec.id!r}."
            )
        if manifest.status == "passed":
            self.console.print(f"[green]Run already passed[/green]: {run_dir}")
            return manifest

        # A newly supplied command-line hint overrides the stored hint. When no
        # new hint is supplied, resume keeps the original run guidance.
        if self.human_hint is None:
            self.human_hint = manifest.human_hint
        else:
            manifest.human_hint = self.human_hint

        previous_failure = manifest.failure_reason
        manifest.status = "running"
        manifest.failure_reason = None
        write_json(manifest_path, manifest)
        self._copy_reference_images(spec, run_dir)
        self.console.print(f"[cyan]Resuming run[/cyan]: {run_dir}")
        self._print_human_hint()
        if previous_failure:
            self.console.print(f"[yellow]Previous interruption[/yellow]: {previous_failure}")

        candidate_path = run_dir / "candidate.py"
        if from_iteration is not None:
            self._truncate_iterations(run_dir, manifest, candidate_path, from_iteration)
        self._restore_candidate_if_missing(run_dir, manifest, candidate_path)
        if not candidate_path.is_file():
            raise FileNotFoundError(
                "No generated candidate is available to resume. Expected either "
                f"{candidate_path} or an iteration snapshot under {run_dir}."
            )

        protected = self._snapshot_protected_files()
        agent_started = False
        try:
            await self.iterative_agent.start()
            agent_started = True

            repeated_hashes = {record.script_sha256 for record in manifest.iterations}
            consecutive_render_failures = self._trailing_render_failure_count(manifest)
            start_iteration = 1
            writer_summary = "Resumed from the previously generated candidate script."

            if manifest.iterations:
                last = manifest.iterations[-1]
                self._resolve_record_artifacts(run_dir, last)
                candidate_hash = sha256_file(candidate_path)
                start_iteration = last.iteration + 1

                # The combined GPT response is persisted before the workspace
                # candidate is overwritten. Recover it if interruption occurred
                # in that tiny window, avoiding a duplicate paid GPT call.
                stored_next = (
                    run_dir
                    / f"iteration_{last.iteration:02d}"
                    / "next_instrument.py"
                )
                if candidate_hash == last.script_sha256 and stored_next.is_file():
                    stored_next_hash = sha256_file(stored_next)
                    if stored_next_hash != last.script_sha256:
                        shutil.copy2(stored_next, candidate_path)
                        candidate_hash = stored_next_hash
                        self.console.print(
                            f"[yellow]Recovered pending GPT revision[/yellow]: {stored_next}"
                        )

                if last.render.success and last.review is not None and self._review_passes(last.review):
                    self.console.print(
                        "[green]The last completed review already passes; finalizing existing files.[/green]"
                    )
                    self._finalize_passed(run_dir, spec, manifest, last)
                    return manifest

                if candidate_hash != last.script_sha256:
                    writer_summary = (
                        "Resumed with an unrendered candidate already written before interruption."
                    )
                    self.console.print(
                        "[green]Found an unrendered revised/repaired candidate[/green]; "
                        "skipping the model call and continuing with Blender."
                    )
                elif last.render.success:
                    if not last.render.images:
                        raise FileNotFoundError(
                            f"Iteration {last.iteration} was marked rendered but no PNG files exist."
                        )
                    self.console.print(
                        f"[cyan]Calling review+coder[/cyan]: "
                        f"{self.config.models.iterative_model.model} with code and "
                        f"{len(last.render.images)} existing image(s)..."
                    )
                    decision = await self.iterative_agent.review_and_revise(
                        spec,
                        self._record_script_path(run_dir, last),
                        last.render.images,
                        last.iteration,
                        issue_history=self._collect_issue_history(manifest),
                        human_hint=self._human_hint_for(),
                    )
                    manifest.token_usage.add(decision.usage)
                    self._save_decision(run_dir, manifest, last, decision)
                    self._print_review(decision.review)
                    self._print_token_usage(manifest.token_usage)
                    if self._review_passes(decision.review):
                        self._finalize_passed(run_dir, spec, manifest, last)
                        return manifest
                    self.iterative_agent.write_revision(decision, candidate_path)
                    writer_summary = decision.summary
                    self._print_revision_saved(decision)
                    self._restore_protected_files(protected)
                else:
                    self.console.print(
                        f"[cyan]Calling repair agent[/cyan]: "
                        f"{self.config.models.iterative_model.model} with code and Blender log..."
                    )
                    repair = await self.iterative_agent.repair_render_failure(
                        spec,
                        self._record_script_path(run_dir, last),
                        last.iteration,
                        last.render.error_summary or "Unknown Blender failure",
                        issue_history=self._collect_issue_history(manifest),
                        human_hint=self._human_hint_for(),
                    )
                    manifest.token_usage.add(repair.usage)
                    write_json(manifest_path, manifest)
                    candidate_path.write_text(repair.script.rstrip() + "\n", encoding="utf-8")
                    writer_summary = repair.summary
                    iteration_dir = run_dir / f"iteration_{last.iteration:02d}"
                    (iteration_dir / "repair_agent_response.txt").write_text(
                        repair.raw_response, encoding="utf-8"
                    )
                    self.console.print("[green]GPT repaired script saved[/green].")
                    self._print_token_usage(manifest.token_usage)
                    self._restore_protected_files(protected)
            else:
                self.console.print(
                    "[green]Found the previously generated initial script[/green]; "
                    "starting Blender iteration 1 without another initial-model call."
                )

            if start_iteration > self.config.loop.max_iterations:
                manifest.status = "max_iterations"
                manifest.failure_reason = (
                    f"The run already reached iteration {start_iteration - 1}; increase "
                    "loop.max_iterations in config.yaml to continue."
                )
                write_json(manifest_path, manifest)
                return manifest

            return await self._iterate(
                spec=spec,
                run_dir=run_dir,
                manifest=manifest,
                candidate_path=candidate_path,
                writer_summary=writer_summary,
                start_iteration=start_iteration,
                repeated_hashes=repeated_hashes,
                consecutive_render_failures=consecutive_render_failures,
                protected=protected,
            )
        except Exception as exc:
            self._mark_failed(manifest, manifest_path, exc)
            raise
        finally:
            if agent_started:
                await self.iterative_agent.close()
            self._restore_protected_files(protected)

    async def _iterate(
        self,
        *,
        spec: InstrumentSpec,
        run_dir: Path,
        manifest: RunManifest,
        candidate_path: Path,
        writer_summary: str,
        start_iteration: int,
        repeated_hashes: set[str],
        consecutive_render_failures: int,
        protected: dict[Path, bytes],
    ) -> RunManifest:
        manifest_path = run_dir / "manifest.json"

        for iteration in range(start_iteration, self.config.loop.max_iterations + 1):
            self.console.rule(f"[bold]Iteration {iteration}: {spec.id}")
            if not candidate_path.exists():
                raise RuntimeError(f"No candidate script exists: {candidate_path}")

            script_hash = sha256_file(candidate_path)
            iteration_dir = run_dir / f"iteration_{iteration:02d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            script_snapshot = iteration_dir / "instrument.py"
            shutil.copy2(candidate_path, script_snapshot)

            if self.config.loop.stop_on_repeated_script and script_hash in repeated_hashes:
                manifest.status = "failed"
                manifest.failure_reason = "The model produced a previously tested script without changes."
                write_json(manifest_path, manifest)
                break
            repeated_hashes.add(script_hash)

            self.console.print(f"[cyan]Starting Blender[/cyan]: {candidate_path.name}")
            render = await asyncio.to_thread(
                self.blender.render,
                candidate_path,
                iteration_dir / "render",
            )
            self.console.print(
                f"[cyan]Blender finished[/cyan] in {render.elapsed_seconds:.1f}s; "
                f"images={len(render.images)}, success={render.success}"
            )
            record = IterationRecord(
                iteration=iteration,
                script_path=script_snapshot,
                script_sha256=script_hash,
                writer_summary=writer_summary,
                render=render,
            )
            manifest.iterations.append(record)
            write_json(manifest_path, manifest)

            if not render.success:
                consecutive_render_failures += 1
                self.console.print(f"[red]Render failed[/red]: {render.error_summary}")
                if consecutive_render_failures >= self.config.loop.max_consecutive_render_failures:
                    manifest.status = "failed"
                    manifest.failure_reason = (
                        f"Reached {consecutive_render_failures} consecutive render failures."
                    )
                    write_json(manifest_path, manifest)
                    break
                self.console.print(
                    f"[cyan]Calling repair agent[/cyan]: "
                    f"{self.config.models.iterative_model.model} with current code and Blender log..."
                )
                repair = await self.iterative_agent.repair_render_failure(
                    spec,
                    script_snapshot,
                    iteration,
                    render.error_summary or "Unknown Blender failure",
                    issue_history=self._collect_issue_history(manifest),
                    human_hint=self._human_hint_for(),
                    reference_images=spec.reference_images,
                )
                manifest.token_usage.add(repair.usage)
                write_json(manifest_path, manifest)
                candidate_path.write_text(repair.script.rstrip() + "\n", encoding="utf-8")
                writer_summary = repair.summary
                (iteration_dir / "repair_agent_response.txt").write_text(
                    repair.raw_response, encoding="utf-8"
                )
                self.console.print("[green]GPT repaired script saved[/green].")
                self._print_token_usage(manifest.token_usage)
                self._restore_protected_files(protected)
                continue

            consecutive_render_failures = 0
            self.console.print(
                f"[cyan]Calling review+coder[/cyan]: "
                f"{self.config.models.iterative_model.model} with exact code and "
                f"{len(render.images)} image(s)..."
            )
            decision = await self.iterative_agent.review_and_revise(
                spec,
                script_snapshot,
                render.images,
                iteration,
                issue_history=self._collect_issue_history(manifest),
                human_hint=self._human_hint_for(),
                reference_images=spec.reference_images,
            )
            manifest.token_usage.add(decision.usage)
            self._save_decision(run_dir, manifest, record, decision)
            self._print_review(decision.review)
            self._print_token_usage(manifest.token_usage)

            if self._review_passes(decision.review):
                self._finalize_passed(run_dir, spec, manifest, record)
                break

            self.iterative_agent.write_revision(decision, candidate_path)
            writer_summary = decision.summary
            self._print_revision_saved(decision)
            self._restore_protected_files(protected)
        else:
            manifest.status = "max_iterations"
            if manifest.iterations and manifest.iterations[-1].review:
                manifest.final_score = manifest.iterations[-1].review.similarity_score
            manifest.failure_reason = "Maximum iteration count reached before passing threshold."
            write_json(manifest_path, manifest)

        return manifest

    def _save_decision(
        self,
        run_dir: Path,
        manifest: RunManifest,
        record: IterationRecord,
        decision: VisionCodeDecision,
    ) -> None:
        iteration_dir = run_dir / f"iteration_{record.iteration:02d}"
        legacy_review = iteration_dir / "review.json"
        if legacy_review.is_file() and record.review is not None:
            backup = iteration_dir / "review_before_combined_agent.json"
            if not backup.exists():
                shutil.copy2(legacy_review, backup)
        record.review = decision.review
        write_json(legacy_review, decision.review)
        (iteration_dir / "gpt_review_and_code_response.txt").write_text(
            decision.raw_response, encoding="utf-8"
        )
        if decision.revised_script is not None:
            (iteration_dir / "next_instrument.py").write_text(
                decision.revised_script.rstrip() + "\n", encoding="utf-8"
            )
        write_json(run_dir / "manifest.json", manifest)
        write_json(
            run_dir / "issue_history.json",
            [item.model_dump(mode="json") for item in self._collect_all_issue_history(manifest)],
        )

    def _finalize_passed(
        self,
        run_dir: Path,
        spec: InstrumentSpec,
        manifest: RunManifest,
        record: IterationRecord,
    ) -> None:
        del spec  # Kept in the signature for compatibility with existing callers.
        self._resolve_record_artifacts(run_dir, record)

        iteration_dir = run_dir / f"iteration_{record.iteration:02d}"
        source_render_dir = iteration_dir / "render"
        final_dir = run_dir / "final"

        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.mkdir(parents=True)

        source_script = self._record_script_path(run_dir, record)
        final_script = final_dir / "instrument.py"
        shutil.copy2(source_script, final_script)

        final_render_dir = final_dir / "render"
        if source_render_dir.is_dir():
            shutil.copytree(source_render_dir, final_render_dir)
        else:
            final_render_dir.mkdir(parents=True)
            artifact_paths = [
                *record.render.images,
                *record.render.blend_files,
                Path(record.render.log_path),
            ]
            copied: set[Path] = set()
            for path in artifact_paths:
                path = Path(path)
                if path.is_file() and path not in copied:
                    shutil.copy2(path, final_render_dir / path.name)
                    copied.add(path)
        
        manifest.status = "passed"
        manifest.failure_reason = None
        manifest.final_script = final_script
        manifest.final_score = record.review.similarity_score if record.review else None
        write_json(run_dir / "manifest.json", manifest)
        self.console.print(f"[bold green]Passed[/bold green]: {final_dir}")

    def _resolve_record_artifacts(self, run_dir: Path, record: IterationRecord) -> None:
        iteration_dir = run_dir / f"iteration_{record.iteration:02d}"
        render_dir = iteration_dir / "render"
        images = [Path(path) for path in record.render.images if Path(path).is_file()]
        blend_files = [Path(path) for path in record.render.blend_files if Path(path).is_file()]
        if not images and render_dir.is_dir():
            images = sorted(render_dir.rglob("*.png"))
        if not blend_files and render_dir.is_dir():
            blend_files = sorted(render_dir.rglob("*.blend"))
        record.render.images = images
        record.render.blend_files = blend_files
        local_log = render_dir / "blender.log"
        if not Path(record.render.log_path).is_file() and local_log.is_file():
            record.render.log_path = local_log
        local_script = iteration_dir / "instrument.py"
        if not Path(record.script_path).is_file() and local_script.is_file():
            record.script_path = local_script

    @staticmethod
    def _record_script_path(run_dir: Path, record: IterationRecord) -> Path:
        stored = Path(record.script_path)
        if stored.is_file():
            return stored
        local = run_dir / f"iteration_{record.iteration:02d}" / "instrument.py"
        if local.is_file():
            return local
        raise FileNotFoundError(f"Missing script snapshot for iteration {record.iteration}.")

    def _truncate_iterations(
        self,
        run_dir: Path,
        manifest: RunManifest,
        candidate_path: Path,
        from_iteration: int,
    ) -> None:
        if from_iteration < 1:
            raise ValueError("from_iteration must be at least 1.")
        keep = [record for record in manifest.iterations if record.iteration <= from_iteration]
        snapshot = run_dir / f"iteration_{from_iteration:02d}" / "instrument.py"
        if not snapshot.is_file():
            raise FileNotFoundError(
                f"Cannot rewind to iteration {from_iteration}: missing {snapshot}"
            )

        dropped = [record for record in manifest.iterations if record.iteration > from_iteration]
        if dropped:
            manifest.iterations = keep
            write_json(run_dir / "manifest.json", manifest)
            for record in dropped:
                iteration_dir = run_dir / f"iteration_{record.iteration:02d}"
                if iteration_dir.is_dir():
                    shutil.rmtree(iteration_dir)

        shutil.copy2(snapshot, candidate_path)
        # Discard the stale revision generated after iteration n so resume will
        # re-review iteration n and produce a fresh next script instead of
        # recovering the old one.
        stale_next = run_dir / f"iteration_{from_iteration:02d}" / "next_instrument.py"
        stale_next.unlink(missing_ok=True)

        if dropped:
            self.console.print(
                f"[yellow]Rewound to iteration {from_iteration}[/yellow]: "
                f"restored candidate from {snapshot} and dropped "
                f"{len(dropped)} later record(s)."
            )
        else:
            self.console.print(
                f"[yellow]Restored candidate from iteration {from_iteration}[/yellow]: {snapshot}"
            )

    def _restore_candidate_if_missing(
        self,
        run_dir: Path,
        manifest: RunManifest,
        candidate_path: Path,
    ) -> None:
        if candidate_path.is_file():
            return
        if not manifest.iterations:
            return
        next_script = run_dir / f"iteration_{manifest.iterations[-1].iteration:02d}" / "next_instrument.py"
        source = next_script if next_script.is_file() else self._record_script_path(run_dir, manifest.iterations[-1])
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, candidate_path)
        self.console.print(f"[yellow]Restored candidate from snapshot[/yellow]: {source}")

    def _snapshot_protected_files(self) -> dict[Path, bytes]:
        return {
            self.config.paths.toolkit: self.config.paths.toolkit.read_bytes(),
            self.config.paths.reference: self.config.paths.reference.read_bytes(),
        }

    def _validate_inputs(self) -> None:
        for path in (
            self.config.paths.toolkit,
            self.config.paths.reference,
            self.config.paths.docs_dir,
            self.config.paths.rules,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        self.config.paths.runs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _copy_reference_images(spec: InstrumentSpec, run_dir: Path) -> None:
        ref_dir = run_dir / "ref_imgs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        for image_path in spec.reference_images:
            image_path = Path(image_path)
            if image_path.is_file():
                shutil.copy2(image_path, ref_dir / image_path.name)

    def _review_passes(self, review) -> bool:
        return review.verdict == "pass" and review.similarity_score >= self.config.loop.pass_score

    def _collect_issue_history(self, manifest: RunManifest) -> list[HistoricalVisualIssue]:
        """Return recent prior moderate-or-higher issues in chronological order.
        The rolling window is used only for prompt context; issue_history.json
        remains a full archival record.
        """
        history = self._collect_all_issue_history(manifest)
        window = self.config.loop.issue_history_window
        if window == 0:
            return []
        return history[-window:]

    @staticmethod
    def _collect_all_issue_history(manifest: RunManifest) -> list[HistoricalVisualIssue]:
        """Return all prior moderate-or-higher issues in chronological order.
        Legacy minor issues remain readable in old manifests but are not fed back
        to the model or written to the active regression checklist.
        """
        remembered_severities = {"moderate", "major", "critical"}
        history: list[HistoricalVisualIssue] = []
        for record in manifest.iterations:
            if record.review is None:
                continue
            for issue_index, issue in enumerate(record.review.issues, start=1):
                if issue.severity not in remembered_severities:
                    continue
                history.append(
                    HistoricalVisualIssue(
                        iteration=record.iteration,
                        issue_index=issue_index,
                        **issue.model_dump(),
                    )
                )
        return history

    def _print_review(self, review) -> None:
        self.console.print(
            f"[cyan]Similarity score[/cyan]: {review.similarity_score:.2f}/5.0, "
            f"verdict={review.verdict}"
        )

    def _print_token_usage(self, usage: TokenUsage) -> None:
        self.console.print(
            f"[cyan]Total token usage[/cyan]: {usage.total_tokens:,} "
            f"(prompt {usage.prompt_tokens:,}, completion {usage.completion_tokens:,})"
        )

    def _print_revision_saved(self, decision: VisionCodeDecision) -> None:
        if decision.review.verdict == "retake_views":
            self.console.print(
                "[green]GPT camera-only retake script saved[/green]; "
                "the next iteration will render a new diagnostic view set."
            )
        else:
            self.console.print("[green]GPT revised script saved[/green].")

    def _human_hint_for(self) -> str | None:
        return self.human_hint

    def _print_human_hint(self) -> None:
        if self.human_hint is None:
            return
        self.console.print(f"[yellow]Human hint active[/yellow]: {self.human_hint}")

    @staticmethod
    def _trailing_render_failure_count(manifest: RunManifest) -> int:
        count = 0
        for record in reversed(manifest.iterations):
            if record.render.success:
                break
            count += 1
        return count

    @staticmethod
    def _restore_protected_files(protected: dict[Path, bytes]) -> None:
        for path, original in protected.items():
            current = path.read_bytes() if path.exists() else None
            if current != original:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original)

    @staticmethod
    def _mark_failed(manifest: RunManifest, manifest_path: Path, exc: Exception) -> None:
        manifest.status = "failed"
        manifest.failure_reason = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
