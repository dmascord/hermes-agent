# Helper for opt-in local Claude CLI fallback
# Provides two helpers used by auxiliary_client:
#  - _messages_to_prompt(messages) -> str
#  - _call_claude_cli(prompt_text, model, timeout) -> SimpleNamespace-like response
from types import SimpleNamespace
import os
import shlex
import subprocess


def _messages_to_prompt(messages: list) -> str:
    parts: list = []
    for msg in messages:
        content = msg.get('content') if isinstance(msg, dict) else msg
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get('text') or block.get('content') or ''
                    if text:
                        parts.append(str(text).strip())
                        continue
                    parts.append(str(block))
                else:
                    parts.append(str(block))
            continue
        parts.append(str(content))
    return "\n\n".join(p for p in parts if p)


def _call_claude_cli(prompt_text: str, model: str, timeout: float):
    """Invoke a local Claude Code CLI and return an object compatible with
    auxiliary_client._validate_llm_response (choices[0].message.content).

    Requires either CLAUDE_CLI_CMD or CLAUDE_CLI_PATH in the environment.
    """
    cmd_env = os.getenv('CLAUDE_CLI_CMD') or ''
    cli_path = os.getenv('CLAUDE_CLI_PATH') or ''
    if not cmd_env and not cli_path:
        raise RuntimeError('Local Claude CLI not configured')

    if cmd_env:
        cmd_str = cmd_env
    else:
        safe_model = shlex.quote(model or 'claude')
        cmd_str = f"{cli_path} code --model {safe_model} --no-color"

    args = shlex.split(cmd_str)
    proc = subprocess.run(args, input=prompt_text.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    out = proc.stdout.decode('utf-8', errors='replace').strip()
    if proc.returncode != 0 or not out:
        stderr = proc.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(f'Claude CLI failed (rc={proc.returncode}) stderr={stderr}')

    message = SimpleNamespace(role='assistant', content=out)
    choice = SimpleNamespace(index=0, message=message, finish_reason='stop')
    return SimpleNamespace(choices=[choice], model=model)
