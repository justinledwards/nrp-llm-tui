## Agents

This project currently ships a single end-user focused agent and a minimal base:

- `UserResponseAgent` (in `nrp_tui/agent_stub.py`) wraps a selected NRP model with a system prompt: “You are User Response, a concise, friendly support agent for the Nautilus Research Platform. Answer the user directly, avoid meta commentary, and only request clarification when necessary. Be brief and actionable.” It keeps conversation history so replies stay in context.
- `SimpleAgent` is a tiny helper that sends a list of `{role, content}` messages to a model.

### How to chat

- TUI: `mise run run` to launch. On start you’ll get a session picker—select an existing session to resume history, enter a new name to start fresh, or delete an old session. Then pick models in the left pane and chat on the right. The app uses `UserResponseAgent` with the selected model(s) and writes logs under `logs/<session-id>/`.
- CLI: `mise run chat -- --model gemma3` (or another model id) for a terminal-only chat loop.
- Programmatic example:

  ```python
  from nrp_tui.agent_stub import UserResponseAgent
  agent = UserResponseAgent(model="gemma3")
  reply = agent.send("How do I list available models?")
  print(reply)
  ```

### Requirements

- Env var `OPENAI_API_KEY` set (optionally via `.env`).
- Virtualenv installed via `mise run setup` (or `mise run setup-dev` if you also want tests) before running chat/TUI commands.

### Testing

- Install dev deps and run tests with `mise run setup-dev` then `mise run test` (uses pytest).
- Manual: `pip install -r requirements.txt -r requirements-dev.txt` then `pytest`.
