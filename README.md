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
  mise run run               # launches session picker (resume or create)
  mise run run -- --session demo       # skip picker, use/create named session
  mise run run -- --session demo --new-session  # force a fresh session
  ```
  Select a model in the left table, then type in the chat input on the right. The session picker also lets you delete old sessions. Input is disabled while the model is responding; replies are not streamed yet.

- CLI chat:
  ```bash
  mise run chat -- --model gemma3           # session name defaults to cli
  mise run chat -- --model gemma3 --session demo   # reuse/create "demo" session
  mise run chat -- --new-session                    # force a fresh session
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
- Each chat session gets its own folder under `logs/` with metadata in `session.json`.
- Per-model transcripts live in `logs/<session-id>/<model>-<session>-<timestamp>.log` plus a structured `*.jsonl` alongside it so previous chats can be resumed across models.

## Testing
Install dev dependencies and run pytest:
```bash
# option 1: using mise tasks
mise run setup-dev   # create .venv and install app + dev deps
mise run test        # runs pytest

# option 2: manual
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
