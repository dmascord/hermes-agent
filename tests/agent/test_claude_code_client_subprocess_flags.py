"""Tests that claude-code-cli subprocess args include sandbox-bypass flags."""

import subprocess
from unittest.mock import patch, ANY

from agent import claude_code_client as ccc


def test_subprocess_extra_flags_constant():
    """_SUBPROCESS_EXTRA_FLAGS contains the sandbox-bypass flags."""
    assert ccc._SUBPROCESS_EXTRA_FLAGS == [
        "--dangerously-skip-permissions",
        "--add-dir", "/tmp", "/opt", "/home", "/root",
    ]


def test_create_chat_completion_includes_extra_flags():
    """_create_chat_completion passes _SUBPROCESS_EXTRA_FLAGS to the subprocess."""
    captured_args = []

    def fake_popen(*args, **kwargs):
        captured_args.append(args[0])
        proc = subprocess.Popen  # real Popen for the mock return
        return proc(["echo", "ok"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    client = ccc.ClaudeCodeClient(claude_cwd="/tmp")
    with patch.object(subprocess, "Popen", side_effect=fake_popen):
        try:
            client._create_chat_completion(
                messages=[{"role": "user", "content": "hello"}],
                model="sonnet",
            )
        except Exception:
            pass  # may fail after Popen due to pipe handling; we only need args

    assert captured_args, "Popen was never called"
    cmd = captured_args[0]
    for flag in ccc._SUBPROCESS_EXTRA_FLAGS:
        assert flag in cmd, f"Missing flag {flag} in cmd_args {cmd}"


def test_run_with_tool_bridge_includes_extra_flags():
    """run_with_tool_bridge passes _SUBPROCESS_EXTRA_FLAGS to the subprocess."""
    captured_args = []

    def fake_popen(*args, **kwargs):
        captured_args.append(args[0])
        proc = subprocess.Popen
        return proc(["echo", "ok"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    client = ccc.ClaudeCodeClient(claude_cwd="/tmp")
    with patch.object(subprocess, "Popen", side_effect=fake_popen):
        try:
            for _ in client.run_with_tool_bridge(
                model="sonnet",
                messages=[{"role": "user", "content": "hello"}],
            ):
                pass
        except Exception:
            pass

    assert captured_args, "Popen was never called"
    cmd = captured_args[0]
    for flag in ccc._SUBPROCESS_EXTRA_FLAGS:
        assert flag in cmd, f"Missing flag {flag} in cmd_args {cmd}"
