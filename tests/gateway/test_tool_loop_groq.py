#!/usr/bin/env python3
"""
Test tool-loop behavior directly with groq to compare against hermes.

If groq handles long conversations correctly, the issue is in hermes.
If groq also has issues, it's a model behavior problem.
"""

import json
import subprocess
import tempfile
import os

# Configuration
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_KEY = os.getenv("GROQ_API_KEY", "")  # From hermes env

# Fallback to hermes if needed
HERMES_URL = "https://hermes.tusker.net.au"
HERMES_KEY = os.getenv("API_SERVER_KEY", "")  # From hermes env


def chat_completions(messages: list, model: str, api_url: str = GROQ_URL, api_key: str = GROQ_KEY):
    """Make a chat completions request using curl."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(body, f)
        body_file = f.name
    
    try:
        result = subprocess.run([
            'curl', '-s', '-X', 'POST',
            f'{api_url}/chat/completions',
            '-H', f'Authorization: Bearer {api_key}',
            '-H', 'Content-Type: application/json',
            '--data-binary', f'@{body_file}',
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {"error": result.stderr}
        
        return json.loads(result.stdout)
    finally:
        os.unlink(body_file)


def test_with_groq():
    """Test with groq directly - this bypasses hermes entirely."""
    print("\n" + "=" * 60)
    print("Testing with Groq (llama-3.3-70b) DIRECTLY - no hermes")
    print("=" * 60)
    
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
    
    # Final tool result
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
    
    print(f"  Sending {len(messages)} messages to groq")
    
    response = chat_completions(messages=messages, model="llama-3.3-70b-versatile")
    
    if "error" in response:
        print(f"  ❌ ERROR: {response['error']}")
        return False
    
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        
        print(f"  Response preview: {content[:300]}...")
        
        if "I need more context" in content or "Could you please provide" in content:
            print(f"  ❌ FAIL: Model asking for context = context dilution")
            return False
        
        if tool_calls:
            print(f"  ✅ Model made {len(tool_calls)} tool call(s)")
        else:
            print(f"  ✅ Model responded (no tool calls)")
    
    return True


def test_distraction_scenario():
    """Test with distraction messages - simulates the real loop scenario."""
    print("\n" + "=" * 60)
    print("Testing distraction scenario with Groq")
    print("=" * 60)
    
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
    
    # Final tool result
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
    
    response = chat_completions(messages=messages, model="llama-3.3-70b-versatile")
    
    if "error" in response:
        print(f"  ❌ ERROR: {response['error']}")
        return False
    
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        
        print(f"  Response preview: {content[:300]}...")
        
        if "I need more context" in content or "Could you please provide" in content:
            print(f"  ❌ FAIL: Model asking for context = context dilution")
            return False
        
        if "ADF" in content or "entity" in content:
            print(f"  ✅ Model continued Task A (mentioned relevant context)")
        elif msg.get("tool_calls"):
            print(f"  ✅ Model continued Task A (made tool calls)")
        else:
            print(f"  ✅ Model responded")
    
    return True


def test_short_conversation():
    """Test short conversation - should work fine."""
    print("\n" + "=" * 60)
    print("Testing short conversation with Groq")
    print("=" * 60)
    
    messages = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "Check the current directory"},
    ]
    
    for i in range(10):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"tc{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"echo cycle {i}"})}}]
        })
        messages.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"output {i}"})
    
    print(f"  Sending {len(messages)} messages")
    
    response = chat_completions(messages=messages, model="llama-3.3-70b-versatile")
    
    if "error" in response:
        print(f"  ❌ ERROR: {response['error']}")
        return False
    
    print(f"  ✅ Response received")
    return True


def test_hermes_for_comparison():
    """Test with hermes for direct comparison."""
    print("\n" + "=" * 60)
    print("Testing with HERMES for comparison")
    print("=" * 60)
    
    messages = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix the ADF entity error in the pipeline"},
    ]
    
    for i in range(75):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"tc{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"echo cycle {i}"})}}]
        })
        messages.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"output {i}"})
    
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "final_tc", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "echo done"})}}]
    })
    messages.append({"role": "tool", "tool_call_id": "final_tc", "content": "done"})
    
    print(f"  Sending {len(messages)} messages")
    
    response = chat_completions(
        messages=messages,
        model="hermes-code",
        api_url=f"{HERMES_URL}/v1",
        api_key=HERMES_KEY
    )
    
    if "error" in response:
        print(f"  ❌ ERROR: {response['error']}")
        return False
    
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        print(f"  Response preview: {content[:300]}...")
        
        if "I need more context" in content or "Could you please provide" in content:
            print(f"  ❌ FAIL: Model asking for context = context dilution")
            return False
        
        print(f"  ✅ Response received")
    
    return True


def main():
    print("=" * 60)
    print("Tool Loop Behavior Tests")
    print("=" * 60)
    print("Comparing Groq (direct) vs Hermes (via hermes gateway)")
    print("=" * 60)
    
    results = []
    
    # Test groq first (direct, no hermes)
    tests = [
        ("Groq: Short conversation", test_short_conversation),
        ("Groq: Long conversation (>150 msgs)", test_with_groq),
        ("Groq: Distraction scenario", test_distraction_scenario),
        ("Hermes: Long conversation for comparison", test_hermes_for_comparison),
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