Unreleased

- Fix: Copilot GPT-5-mini incorrectly routed to Responses API (/responses). Now uses Copilot-specific API-mode logic to keep gpt-5-mini on chat_completions.
- Tests: Add regression test to ensure copilot gpt-5-mini stays on chat_completions (TBD).