from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

# Ensure project root on path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from textual.widgets import Input  # noqa: E402

from nrp_tui.sessions import SessionStore  # noqa: E402
from nrp_tui.tui import ModelTableApp  # noqa: E402


@pytest.mark.asyncio
async def test_tui_session_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Provide API key for config loading
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock model list to a single entry
    monkeypatch.setattr(
        "nrp_tui.tui.NRPClient.list_models",
        lambda self: [
            {
                "id": "gemma3",
                "created": None,
                "status": None,
                "title": None,
                "parameters": None,
                "context_tokens": None,
                "features": None,
                "notes": None,
            }
        ],
    )

    # Mock agent send to avoid network
    def fake_send(self, msg: str) -> str:  # type: ignore[override]
        # Mimic basic logging behavior without network calls.
        self.history.append({"role": "user", "content": msg})
        reply = f"echo: {msg}"
        self.history.append({"role": "assistant", "content": reply})
        return reply

    monkeypatch.setattr("nrp_tui.agent_stub.UserResponseAgent.send", fake_send)

    store = SessionStore(base_dir=tmp_path)
    session = store.create("integration", created_at=datetime(2024, 1, 1, 12, 0, 0))
    app = ModelTableApp(session=session, resume=False, store=store)

    async with app.run_test() as pilot:
        await app._add_model("gemma3")
        assert "gemma3" in app.agents

        input_widget = app.chat_input
        assert input_widget is not None
        app.selected_models = {"gemma3"}

        # Simulate user submitting a message
        event = Input.Submitted(input_widget, "hello from test")
        await app.on_input_submitted(event)

        # Verify history captured the assistant reply
        agent = app.agents["gemma3"]
        assert agent.history[-1]["content"] == "echo: hello from test"

        # Verify the log widget shows both user and assistant messages
        log_widget = app.chat_logs["gemma3"]
        lines = "\n".join(getattr(log_widget, "lines", []))
        assert "You: hello from test" in lines
        assert "gemma3: echo: hello from test" in lines
