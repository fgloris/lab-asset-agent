from __future__ import annotations

import httpx
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import require_env
from .models import ModelConfig, TokenUsage


@dataclass
class ChatCompletion:
    text: str
    usage: TokenUsage


_SCHEMA_ERROR_MARKERS = (
    "invalid schema",
    "response_format",
    "json_schema",
    "json schema",
    "structured output",
    "additionalproperties",
    "additional properties",
    "json_object",
    "json object",
)


def make_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an OpenAI-style strict JSON schema without mutating the input."""

    normalized = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        for value in list(node.values()):
            visit(value)

        properties = node.get("properties")
        if node.get("type") == "object" or isinstance(properties, dict):
            node["type"] = "object"
            node["additionalProperties"] = False
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())

    visit(normalized)
    return normalized


class _StreamTerminal:
    """Render model streaming progress without exposing reasoning by default."""

    def __init__(
        self,
        *,
        enabled: bool,
        label: str,
        reasoning_mode: str,
    ) -> None:
        self.enabled = enabled
        self.label = label
        self.reasoning_mode = reasoning_mode
        self.reasoning_chars = 0
        self.content_chars = 0
        self._last_reasoning_report = 0
        self._reasoning_line_active = False
        self._reasoning_header_printed = False
        self._content_header_printed = False

    def start(self) -> None:
        if self.enabled:
            print(
                f"[lab-asset-agent] Streaming model response: {self.label}",
                flush=True,
            )

    def reasoning(self, text: str) -> None:
        if not text:
            return
        self.reasoning_chars += len(text)
        if not self.enabled or self.reasoning_mode == "hidden":
            return

        if self.reasoning_mode == "full":
            if not self._reasoning_header_printed:
                print("[lab-asset-agent] --- reasoning stream ---", flush=True)
                self._reasoning_header_printed = True
            print(text, end="", flush=True)
            return

        # The default "progress" mode proves that the endpoint is still
        # producing tokens without flooding the terminal with chain-of-thought.
        if self.reasoning_chars - self._last_reasoning_report >= 256:
            print(
                f"\r[lab-asset-agent] {self.label}: reasoning received "
                f"{self.reasoning_chars:,} chars",
                end="",
                flush=True,
            )
            self._last_reasoning_report = self.reasoning_chars
            self._reasoning_line_active = True

    def content(self, text: str) -> None:
        if not text:
            return
        self.content_chars += len(text)
        if not self.enabled:
            return
        if self._reasoning_line_active:
            print(flush=True)
            self._reasoning_line_active = False
        if self.reasoning_mode == "full" and self._reasoning_header_printed:
            print("\n[lab-asset-agent] --- final response stream ---", flush=True)
            self._reasoning_header_printed = False
        elif not self._content_header_printed:
            print("[lab-asset-agent] --- response stream ---", flush=True)
        self._content_header_printed = True
        print(text, end="", flush=True)

    def finish(self) -> None:
        if not self.enabled:
            return
        if self._reasoning_line_active:
            print(flush=True)
            self._reasoning_line_active = False
        if self._content_header_printed:
            print(flush=True)
        print(
            f"[lab-asset-agent] Stream complete: {self.content_chars:,} response chars"
            + (
                f", {self.reasoning_chars:,} reasoning chars"
                if self.reasoning_chars
                else ""
            ),
            flush=True,
        )


class OpenAICompatibleClient:
    """Shared adapter for DeepSeek, OpenAI, and compatible gateways.

    Chat completions stream by default. Final answer deltas are printed as they
    arrive and are accumulated for the existing parsers. DeepSeek-style
    ``reasoning_content`` is hidden, summarized as a live character count, or
    printed in full according to ``stream_reasoning``.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._stream_usage_supported = True
        if client is not None:
            self.client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The `openai` package is required for real API calls. "
                    "Install project dependencies first."
                ) from exc
            self.client = AsyncOpenAI(
                api_key=require_env(config.api_key_env),
                base_url=config.base_url.rstrip("/") + "/",
                max_retries=config.max_retries,
                timeout=httpx.Timeout(
                    timeout=config.request_timeout_seconds,
                    connect=config.connect_timeout_seconds,
                    read=config.request_timeout_seconds,
                    write=30.0,
                    pool=10.0,
                ),
            )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        response_format_mode: str | None = None,
        stream_label: str | None = None,
        stream_output_path: str | Path | None = None,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        formats = self._response_formats(response_schema, schema_name, response_format_mode)
        last_format_error: Exception | None = None

        for attempt, response_format in enumerate(formats, start=1):
            call_kwargs = dict(kwargs)
            if response_format is not None:
                call_kwargs["response_format"] = response_format
            if self.config.stream:
                call_kwargs["stream"] = True
            try:
                response = await self._create_stream_completion(call_kwargs)
                if self.config.stream and self._is_stream_response(response):
                    return await self._consume_stream(
                        response,
                        label=stream_label or self.config.model,
                        output_path=Path(stream_output_path) if stream_output_path else None,
                    )
                return ChatCompletion(
                    text=self._message_text(response),
                    usage=self._extract_usage(response),
                )
            except Exception as exc:
                if not self._is_response_format_error(exc):
                    raise
                last_format_error = exc
                if attempt < len(formats):
                    next_format = formats[attempt]
                    self._print_fallback_warning(response_format, next_format, exc)
                    continue
                break

        if last_format_error is not None:
            raise last_format_error
        raise RuntimeError("OpenAI-compatible endpoint returned no response.")

    @staticmethod
    def _is_stream_response(response: Any) -> bool:
        if getattr(response, "choices", None) is not None:
            return False
        return hasattr(response, "__aiter__") or hasattr(response, "__iter__")

    async def _create_stream_completion(self, call_kwargs: dict[str, Any]) -> Any:
        if not call_kwargs.get("stream"):
            return await self.client.chat.completions.create(**call_kwargs)
        if self._stream_usage_supported:
            call_kwargs["stream_options"] = {"include_usage": True}
            try:
                return await self.client.chat.completions.create(**call_kwargs)
            except Exception as exc:
                if not self._is_stream_options_error(exc):
                    raise
                call_kwargs.pop("stream_options", None)
                self._stream_usage_supported = False
                self._print_stream_options_fallback(exc)
        return await self.client.chat.completions.create(**call_kwargs)

    @staticmethod
    def _is_stream_options_error(exc: Exception) -> bool:
        parts = [str(exc)]
        for attribute in ("message", "body", "response"):
            value = getattr(exc, attribute, None)
            if value is not None:
                parts.append(str(value))
        text = " ".join(parts).lower()
        return any(
            marker in text
            for marker in (
                "stream_options",
                "include_usage",
                "unexpected keyword",
                "unknown parameter",
            )
        )

    @staticmethod
    def _print_stream_options_fallback(exc: Exception) -> None:
        message = str(exc).replace("\n", " ")
        if len(message) > 240:
            message = message[:237] + "..."
        print(
            "[lab-asset-agent] Endpoint rejected stream usage stats; retrying "
            f"without token usage. Reason: {message}",
            file=sys.stderr,
            flush=True,
        )

    async def _consume_stream(
        self,
        stream: Any,
        *,
        label: str,
        output_path: Path | None,
    ) -> ChatCompletion:
        terminal = _StreamTerminal(
            enabled=self.config.stream_to_terminal,
            label=label,
            reasoning_mode=self.config.stream_reasoning,
        )
        terminal.start()
        content_parts: list[str] = []
        usage = TokenUsage()
        output_file = None
        try:
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_file = output_path.open("w", encoding="utf-8")

            async for chunk in stream:
                chunk_usage = self._extract_usage(chunk)
                if (
                    chunk_usage.prompt_tokens
                    or chunk_usage.completion_tokens
                    or chunk_usage.total_tokens
                ):
                    usage = chunk_usage
                content, reasoning = self._chunk_text(chunk)
                if reasoning:
                    terminal.reasoning(reasoning)
                if content:
                    content_parts.append(content)
                    terminal.content(content)
                    if output_file is not None:
                        output_file.write(content)
                        output_file.flush()
        finally:
            if output_file is not None:
                output_file.close()
            terminal.finish()

        text = "".join(content_parts)
        if not text.strip():
            raise RuntimeError(
                "The streaming endpoint completed without final textual content. "
                "Reasoning tokens may have been received, but no parseable answer was returned."
            )
        return ChatCompletion(text=text, usage=usage)

    @classmethod
    def _chunk_text(cls, chunk: Any) -> tuple[str, str]:
        choices = cls._get_value(chunk, "choices") or []
        if not choices:
            return "", ""
        delta = cls._get_value(choices[0], "delta")
        if delta is None:
            return "", ""
        content = cls._normalize_text(cls._get_value(delta, "content"))
        reasoning = cls._normalize_text(
            cls._get_value(delta, "reasoning_content")
            or cls._get_value(delta, "reasoning")
        )
        return content, reasoning

    @staticmethod
    def _get_value(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text = cls._get_value(item, "text")
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
        return str(value)

    @classmethod
    def _extract_usage(cls, value: Any) -> TokenUsage:
        usage = cls._get_value(value, "usage")
        if usage is None:
            return TokenUsage()
        prompt_tokens = int(cls._get_value(usage, "prompt_tokens") or 0)
        completion_tokens = int(cls._get_value(usage, "completion_tokens") or 0)
        total_tokens = int(cls._get_value(usage, "total_tokens") or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _response_formats(
        self,
        response_schema: dict[str, Any] | None,
        schema_name: str,
        response_format_mode: str | None = None,
    ) -> list[dict[str, Any] | None]:
        if response_schema is None:
            return [None]

        strict_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": make_strict_json_schema(response_schema),
                "strict": True,
            },
        }
        json_object = {"type": "json_object"}
        mode = response_format_mode or self.config.response_format_mode

        if mode == "json_schema":
            candidates: list[dict[str, Any] | None] = [strict_schema, json_object, None]
        elif mode == "json_object":
            candidates = [json_object, None]
        elif mode == "text":
            candidates = [None]
        else:
            candidates = [strict_schema, json_object, None]

        unique: list[dict[str, Any] | None] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

    @staticmethod
    def _is_response_format_error(exc: Exception) -> bool:
        parts = [str(exc)]
        for attribute in ("message", "body", "response"):
            value = getattr(exc, attribute, None)
            if value is not None:
                parts.append(str(value))
        text = " ".join(parts).lower()
        return any(marker in text for marker in _SCHEMA_ERROR_MARKERS)

    @staticmethod
    def _format_label(response_format: dict[str, Any] | None) -> str:
        if response_format is None:
            return "plain text"
        return str(response_format.get("type", "unknown"))

    @classmethod
    def _print_fallback_warning(
        cls,
        current: dict[str, Any] | None,
        following: dict[str, Any] | None,
        exc: Exception,
    ) -> None:
        message = str(exc).replace("\n", " ")
        if len(message) > 240:
            message = message[:237] + "..."
        print(
            "[lab-asset-agent] Endpoint rejected "
            f"{cls._format_label(current)} response formatting; retrying with "
            f"{cls._format_label(following)}. Reason: {message}",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _message_text(response: Any) -> str:
        if not getattr(response, "choices", None):
            raise RuntimeError("OpenAI-compatible endpoint returned no choices.")
        message = response.choices[0].message
        text = getattr(message, "content", None) or getattr(message, "reasoning_content", None)
        if not text:
            raise RuntimeError("OpenAI-compatible endpoint returned no textual content.")
        return str(text)
