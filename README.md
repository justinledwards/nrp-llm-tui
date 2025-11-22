# NRP LLM TUI

Simple Textual TUI and CLI helpers to list managed NRP models and chat with them using a user-facing agent prompt.

## Prerequisites
- Python 3.12 (managed via `mise` from `.mise.toml`)
- `OPENAI_API_KEY` set in your environment (or in `.env` sourced by tasks)

## Setup
```bash
mise run setup
```

## Usage
- TUI (model list + chat pane):
  ```bash
  mise run run
  ```
  Select a model in the left table, then type in the chat input on the right. Input is disabled while the model is responding; replies are not streamed yet.

- CLI chat:
  ```bash
  mise run chat -- --model gemma3
  ```
  Replace `gemma3` with any available model id.

- List models:
  ```bash
  mise run models
  ```

## Agents
- `UserResponseAgent` (see `AGENTS.md` and `nrp_tui/agent_stub.py`) adds a concise support-oriented system prompt and keeps conversation history.

## Environment
The provided tasks automatically load `.env` if present:
```bash
OPENAI_API_KEY=your_key_here
```

## Logs
- Runtime logs are written to `logs/tui.log`. The `logs/` directory is ignored by git by default.
