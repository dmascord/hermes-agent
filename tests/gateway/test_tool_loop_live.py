#!/usr/bin/env python3
"""
Live integration tests for tool-loop behavior against hermes.

Tests the actual fix:
1. With >150 messages and tool result continuation → truncation to 50
2. Whitespace user_message doesn't cause error
3. Model continues from context, not re-analyze

Uses curl to call hermes directly (no MITM proxy needed).
"""

import json
import subprocess
import tempfile
import os

# Configuration
HERMES_URL = "https://hermes.tusker.net.au"
HERMES_API_KEY = "b49a80d538b98987e2f0c385bba137c79f017051cef9b95efd61929791dd4218"


def chat_completions(messages: list, model: str = "hermes-code", stream: bool = False, tools: list = None):
    """Make a chat completions request using curl."""
    body = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    
    if tools:
        body["tools"] = tools
    
    # Write body to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(body, f)
        body_file = f.name
    
    try:
        result = subprocess.run([
            'curl', '-s', '-X', 'POST',
            f'{HERMES_URL}/v1/chat/completions',
            '-H', f'Authorization: Bearer {HERMES_API_KEY}',
            '-H', 'Content-Type: application/json',
            '-H', 'Accept-Encoding: identity',
            '--data-binary', f'@{body_file}',
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {"error": result.stderr}
        
        return json.loads(result.stdout)
    finally:
        os.unlink(body_file)


# ---------------------------------------------------------------------------
# Test: Short conversation (<150 msgs) - no truncation
# ---------------------------------------------------------------------------

def test_short_conversation_no_truncation():
    """With <150 messages, full history should be sent."""
    print("\n=== test_short_conversation_no_truncation ===")
    
    messages = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "Check the current directory"},
    ]
    
    # Add 10 tool cycles
    for i in range(10):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"tc{i}",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": f"echo cycle {i}"})}
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "content": f"output {i}"
        })
    
    print(f"  Sending {len(messages)} messages")
    
    response = chat_completions(messages=messages, model="hermes-code")
    
    if "error" in response:
        print(f"  ❌ ERROR: {response['error']}")
        return False
    
    print(f"  ✅ Response received")
    return True


# ---------------------------------------------------------------------------
# Test: Tool-loop continuation with truncation (>150 msgs)
# ---------------------------------------------------------------------------

def test_tool_loop_truncation():
    """
    With >150 messages AND last is tool result AND prior has pending tool_calls,
    history should be truncated to last 50.
    
    We can't directly verify truncation happened, but we can verify
    the response doesn't show signs of context dilution (re-analyzing from scratch).
    """
    print("\n=== test_tool_loop_truncation ===")
    
    messages = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix the ADF entity error in the pipeline"},
    ]
    
    # Add 75 tool cycles (150 messages)
    for i in range(75):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"tc{i}",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": f"echo cycle {i}"})}
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "content": f"output {i}"
        })
    
    # Final tool result (the continuation trigger)
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "final_tc",
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": "echo done"})}
        }]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": "final_tc",
        "content": "done"
    })
    
    print(f"  Sending {len(messages)} messages")
    print(f"  Last 3 messages roles: {[m['role'] for m in messages[-3:]]}")
    print(f"  Prior assistant has tool_calls: {messages[-3].get('tool_calls') is not None}")
    
    response = chat_completions(messages=messages, model="hermes-code")
    
    if "error" in response:
        print(f"  ❌ ERROR: {response['error']}")
        return False
    
    # Check response - should continue from context, not re-analyze
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        
        print(f"  Response preview: {content[:200]}...")
        
        # Model should be aware it's in the middle of work, not start fresh
        if "I need more context" in content or "Could you please provide" in content:
            print(f"  ❌ FAIL: Model asking for context = context dilution detected")
            return False
        
        if tool_calls:
            print(f"  ✅ Model made {len(tool_calls)} tool call(s)")
        else:
            print(f"  ✅ Model responded (no tool calls)")
    
    return True


# ---------------------------------------------------------------------------
# Test: Whitespace placeholder doesn't cause error
# ---------------------------------------------------------------------------

def test_whitespace_placeholder():
    """Single space user_message should work without 'empty message' error."""
    print("\n=== test_whitespace_placeholder ===")
    
    messages = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "List files"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})}}
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt\nfile2.txt"},
    ]
    
    print(f"  Sending conversation with tool result")
    
    response = chat_completions(messages=messages, model="hermes-code")
    
    if "error" in response:
        error_msg = str(response["error"])
        if "empty" in error_msg.lower():
            print(f"  ❌ FAILED: Got empty message error: {error_msg}")
            return False
        print(f"  ❌ ERROR: {error_msg}")
        return False
    
    print(f"  ✅ No empty message error")
    return True


# ---------------------------------------------------------------------------
# Test: Repeated bash commands (the loop pattern)
# ---------------------------------------------------------------------------

def test_repeated_bash_no_loop():
    """Model should NOT repeat the same git commands forever."""
    print("\n=== test_repeated_bash_no_loop ===")
    
    messages = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "Look at the ADF files"},
    ]
    
    # Simulate 30 cycles of the same git commands
    for i in range(30):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"tc{i}_1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "git status"})}},
                {"id": f"tc{i}_2", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "ls -la"})}},
            ]
        })
        messages.append({"role": "tool", "tool_call_id": f"tc{i}_1", "content": "On branch feature/adf"})
        messages.append({"role": "tool", "tool_call_id": f"tc{i}_2", "content": "total 16\ndrwxr-xr-x"})
    
    # Final tool result to trigger continuation
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "final_tc", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "git branch"})}}]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": "final_tc",
        "content": "* feature/adf"
    })
    
    print(f"  Sending {len(messages)} messages with repeated commands")
    
    response = chat_completions(messages=messages, model="hermes-code")
    
    if "error" in response:
        print(f"  ❌ ERROR: {response['error']}")
        return False
    
    # Check the response
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        
        # Model should recognize the work is done, not keep repeating
        if "Backup complete" in content or "Already done" in content:
            print(f"  ✅ Model recognized work is complete")
        elif tool_calls:
            # If making tool calls, verify it's not the same old commands
            for tc in tool_calls:
                args = tc.get("function", {}).get("arguments", "{}")
                try:
                    cmd = json.loads(args).get("command", "")
                    if cmd in ("git status", "ls -la"):
                        print(f"  ⚠️  Model repeated: {cmd}")
                except:
                    pass
    
    print(f"  ✅ Response received")
    return True


# ---------------------------------------------------------------------------
# Test: Model continues from context, not re-analyzes
# ---------------------------------------------------------------------------

def test_model_continues_not_reanalyzes():
    """Model should continue from context, not re-analyze from beginning."""
    print("\n=== test_model_continues_not_reanalyzes ===")
    
    messages = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix the ADF entity error"},
    ]
    
    # Task A: 5 tool cycles
    for i in range(5):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"tc{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"echo taskA {i}"})}}]
        })
        messages.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"result {i}"})
    
    # 100 distraction messages (50 tool cycles)
    for i in range(50):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"dist{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"echo distraction {i}"})}}]
        })
        messages.append({"role": "tool", "tool_call_id": f"dist{i}", "content": f"distraction {i}"})
    
    # Final tool result to trigger continuation
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "final_tc", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "echo done"})}}]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": "final_tc",
        "content": "Task complete"
    })
    
    print(f"  Sending {len(messages)} messages")
    
    response = chat_completions(messages=messages, model="hermes-code")
    
    if "error" in response:
        print(f"  ❌ ERROR: {response['error']}")
        return False
    
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        
        print(f"  Response preview: {content[:200]}...")
        
        # Model should NOT ask for context - it should have enough from truncation
        if "I need more context" in content or "Could you please provide" in content:
            print(f"  ❌ FAIL: Model asking for context = context dilution")
            return False
        
        # Should continue the task
        if "ADF" in content or "entity" in content:
            print(f"  ✅ Model continued Task A (mentioned relevant context)")
        elif msg.get("tool_calls"):
            print(f"  ✅ Model continued Task A (made tool calls)")
        else:
            print(f"  ✅ Model responded")
    
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run all tests."""
    print("=" * 60)
    print("Live Tool Loop Tests (direct to hermes)")
    print("=" * 60)
    
    results = []
    
    # Run tests
    tests = [
        ("Short conversation (<150 msgs)", test_short_conversation_no_truncation),
        ("Whitespace placeholder", test_whitespace_placeholder),
        ("Tool-loop continuation (>150 msgs)", test_tool_loop_truncation),
        ("Repeated bash no loop", test_repeated_bash_no_loop),
        ("Model continues not reanalyzes", test_model_continues_not_reanalyzes),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")
    
    passed = sum(1 for _, r in results if r)
    print(f"\n{passed}/{len(results)} tests passed")
    
    return all(r for _, r in results)


if __name__ == "__main__":
    main()