"""Helpers for using the local Claude CLI as an auxiliary client backend."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple


def _find_claude_cli_path() -> str:
    cli_path = (os.getenv("CLAUDE_CLI_PATH") or "").strip()
    if cli_path:
        return cli_path
    return shutil.which("claude") or ""


def _extract_system_prompt(messages: Iterable[Any]) -> Tuple[str, list[Any]]:
    system_parts: list[str] = []
    non_system: list[Any] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = _content_to_text(msg.get("content", "")).strip()
            if content:
                system_parts.append(content)
            continue
        non_system.append(msg)
    return "\n\n".join(system_parts).strip(), non_system


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
                elif block.get("type") == "image_url":
                    image_url = block.get("image_url") or {}
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = str(image_url)
                    if url:
                        parts.append(f"[Image omitted for Claude CLI wrapper: {url}]")
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _messages_to_prompt(messages: Iterable[Any]) -> str:
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role") or "user").strip().upper()
            content = _content_to_text(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "TOOL":
                tool_call_id = str(msg.get("tool_call_id") or "").strip()
                prefix = f"{role} ({tool_call_id})" if tool_call_id else role
            else:
                prefix = role
            parts.append(f"{prefix}:\n{content}")
        else:
            text = _content_to_text(msg).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def _build_claude_cli_args(
    *,
    prompt_text: str,
    model: str,
    system_prompt: str = "",
    output_format: str = "json",
) -> list[str]:
    cmd_env = (os.getenv("CLAUDE_CLI_CMD") or "").strip()
    if cmd_env:
        args = shlex.split(cmd_env)
    else:
        cli_path = _find_claude_cli_path()
        if not cli_path:
            raise RuntimeError("Local Claude CLI not configured or not installed")
        args = [cli_path]

    if "-p" not in args and "--print" not in args:
        args.append("--print")
    if "--output-format" not in args and output_format:
        args.extend(["--output-format", output_format])
    if "--model" not in args and model:
        args.extend(["--model", model])
    if "--tools" not in args and "--allowedTools" not in args and "--allowed-tools" not in args:
        args.extend(["--tools", ""])
    if "--no-session-persistence" not in args:
        args.append("--no-session-persistence")
    if "--system-prompt" not in args and system_prompt:
        args.extend(["--system-prompt", system_prompt])
    args.append(prompt_text)
    return args


def _parse_claude_cli_output(raw_text: str, *, output_format: str = "json") -> Tuple[str, Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        raise RuntimeError("Claude CLI returned empty output")

    if output_format == "json":
        last_line = text.splitlines()[-1].strip()
        try:
            payload = json.loads(last_line)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Claude CLI returned invalid JSON: {exc}") from exc

        result_text = str(payload.get("result") or payload.get("content") or "").strip()
        is_error = bool(payload.get("is_error"))
        if is_error:
            message = result_text or str(payload)
            raise RuntimeError(message)
        if not result_text:
            raise RuntimeError(f"Claude CLI JSON output missing result: {payload}")
        return result_text, payload

    return text, {"type": "result", "result": text, "stop_reason": "stop"}


def _call_claude_cli_from_messages(messages: Iterable[Any], model: str, timeout: float):
    system_prompt, non_system_messages = _extract_system_prompt(messages)
    prompt_text = _messages_to_prompt(non_system_messages)
    if not prompt_text:
        prompt_text = "Please respond helpfully."
    return _call_claude_cli(prompt_text, model, timeout, system_prompt=system_prompt)


def _call_claude_cli(
    prompt_text: str,
    model: str,
    timeout: float,
    *,
    system_prompt: str = "",
    output_format: str = "json",
):
    """Invoke the local Claude CLI and return a chat-completions-like object."""
    args = _build_claude_cli_args(
        prompt_text=prompt_text,
        model=model,
        system_prompt=system_prompt,
        output_format=output_format,
    )
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (rc={proc.returncode}) stderr={stderr or stdout}")

    parsed_text, payload = _parse_claude_cli_output(stdout, output_format=output_format)
    finish_reason = str(payload.get("stop_reason") or "stop")
    message = SimpleNamespace(role="assistant", content=parsed_text)
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, raw=payload, stderr=stderr)
