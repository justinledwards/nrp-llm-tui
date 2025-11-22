import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, DataTable, Static, Input, Log
from textual.reactive import reactive

from .client import NRPClient
from .agent_stub import UserResponseAgent

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_logger = logging.getLogger("nrp_tui")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "tui.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    _logger.addHandler(handler)


class ModelTableApp(App):
    CSS_PATH = None

    loading: reactive[bool] = reactive(True)
    selected_model: reactive[Optional[str]] = reactive(None)
    chat_pending: reactive[bool] = reactive(False)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._models: List[Dict[str, Any]] = []
        self.client = NRPClient()
        self.agent: Optional[UserResponseAgent] = None
        self.chat_log: Optional[Log] = None
        self.chat_hint: Optional[Static] = None
        self.chat_input: Optional[Input] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Static("NRP Managed LLMs", id="title"),
                DataTable(id="models_table"),
                id="model_panel",
            ),
            Vertical(
                Static("User Response Chat", id="chat_title"),
                Static("Select a model to start chatting.", id="chat_hint"),
                Log(
                    id="chat_log",
                    highlight=False,
                    max_lines=500,
                ),
                Input(placeholder="Type a message and press Enter", id="chat_input"),
                id="chat_panel",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        table: DataTable = self.query_one("#models_table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "ID",
            "Status",
            "Params",
            "Context",
            "Created",
            "Features",
        )

        self.load_models()
        self.chat_log = self.query_one("#chat_log", Log)
        self.chat_hint = self.query_one("#chat_hint", Static)
        self.chat_input = self.query_one("#chat_input", Input)
        self.set_focus(table)

    def load_models(self) -> None:
        self.loading = True
        models = self.client.list_models()
        self._models = models

        table: DataTable = self.query_one("#models_table", DataTable)
        table.clear()

        for m in models:
            created_str = m["created"].strftime("%Y-%m-%d") if m["created"] else ""
            context_str = str(m["context_tokens"]) if m["context_tokens"] else ""
            table.add_row(
                m["id"],
                m["status"] or "",
                m["parameters"] or "",
                context_str,
                created_str,
                m["features"] or "",
                key=m["id"],
            )

        self.loading = False

    def action_refresh(self) -> None:
        """Refresh model list."""
        self.load_models()

    @on(DataTable.RowSelected)
    def handle_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = event.row_key
        model_id = row_key.value if hasattr(row_key, "value") else row_key
        model_id = str(model_id)
        self.selected_model = model_id
        self.agent = UserResponseAgent(model=model_id)
        if self.chat_log:
            self.chat_log.clear()
            self.chat_log.write(f"[system] Now chatting with {model_id}\n")
        if self.chat_hint:
            self.chat_hint.update(f"Chatting with {model_id}.")
        if self.chat_input:
            self.chat_input.placeholder = f"Message for {model_id}"
            self.set_focus(self.chat_input)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat_input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if not self.agent or not self.selected_model:
            if self.chat_log:
                self.chat_log.write("[warn] Select a model first.\n")
            return

        if self.chat_log:
            self.chat_log.write(f"You: {text}\n")
            self.chat_log.write("[system] Waiting for response...\n")
        if self.chat_input:
            self.chat_input.disabled = True
            self.chat_pending = True

        try:
            reply = await asyncio.to_thread(self.agent.send, text)
        except Exception as exc:  # pragma: no cover - runtime safety
            _logger.exception("Chat request failed for model %s", self.selected_model)
            if self.chat_log:
                self.chat_log.write(f"[error] Chat request failed: {exc}\n")
            return

        if self.chat_log:
            if self.chat_pending:
                self.chat_log.write("[system] Response received.\n")
            self.chat_log.write(f"{self.selected_model}: {reply}\n")
        if self.chat_input:
            self.chat_input.disabled = False
            self.set_focus(self.chat_input)
        self.chat_pending = False


def run_tui() -> None:
    app = ModelTableApp()
    app.run()
