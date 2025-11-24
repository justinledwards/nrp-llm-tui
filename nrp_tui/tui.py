import asyncio
import inspect
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, DataTable, Static, Input, Log
from textual.reactive import reactive

from .client import NRPClient
from .agent_stub import UserResponseAgent
from .logging_utils import LOG_DIR

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

    MAX_RESPONSE_SECONDS = 30

    loading: reactive[bool] = reactive(True)
    chat_pending: reactive[bool] = reactive(False)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._models: List[Dict[str, Any]] = []
        self.client = NRPClient()
        self.agents: Dict[str, UserResponseAgent] = {}
        self.selected_models: set[str] = set()
        self.chat_logs: Dict[str, Log] = {}
        self.chat_log_panels: Dict[str, Vertical] = {}
        self.chat_log_container: Optional[Horizontal] = None
        self.chat_status: Dict[str, Static] = {}
        self.chat_hint: Optional[Static] = None
        self.chat_input: Optional[Input] = None
        self.selected_column_key: Any | None = None
        self.session_label = "tui"

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
                Static("Select one or more models to start chatting.", id="chat_hint"),
                Horizontal(id="chat_logs_container"),
                Input(placeholder="Type a message and press Enter", id="chat_input"),
                id="chat_panel",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        table: DataTable = self.query_one("#models_table", DataTable)
        table.cursor_type = "row"
        columns = table.add_columns(
            "Selected",
            "ID",
            "Status",
            "Params",
            "Context",
            "Created",
            "Features",
        )
        if columns:
            self.selected_column_key = columns[0]

        self.load_models()
        self.chat_hint = self.query_one("#chat_hint", Static)
        self.chat_input = self.query_one("#chat_input", Input)
        self.chat_log_container = self.query_one("#chat_logs_container", Horizontal)
        # Bias layout: left panel ~1/3, right panel ~2/3 for more chat space.
        try:
            self.query_one("#model_panel").styles.width = "1fr"
            self.query_one("#chat_panel").styles.width = "2fr"
        except Exception:
            pass
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
                self._selected_indicator(m["id"]),
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
    async def handle_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = event.row_key
        model_id = row_key.value if hasattr(row_key, "value") else row_key
        model_id = str(model_id)
        await self.toggle_model_selection(model_id, row_key=row_key)
        if self.chat_input:
            self.set_focus(self.chat_input)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat_input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if not self.selected_models:
            if self.chat_hint:
                self.chat_hint.update("Select one or more models first.")
            return

        if self.chat_input:
            self.chat_input.disabled = True
            self.chat_pending = True

        sends: List[tuple[str, asyncio.Task[Any]]] = []
        for model_id in list(self.selected_models):
            log = self.chat_logs.get(model_id)
            if log:
                log.write(f"You: {text}\n")
                log.write("[system] Waiting for response...\n")
            self._set_status(model_id, "waiting")
            agent = self.agents.get(model_id)
            if agent:
                send_task = asyncio.wait_for(
                    asyncio.to_thread(agent.send, text), timeout=self.MAX_RESPONSE_SECONDS
                )
                sends.append((model_id, send_task))

        try:
            results = await asyncio.gather(
                *[task for _, task in sends], return_exceptions=True
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            _logger.exception("Chat request failed for models %s", self.selected_models)
            results = [exc] * len(sends)

        for (model_id, _), result in zip(sends, results):
            log = self.chat_logs.get(model_id)
            if isinstance(result, Exception):
                _logger.exception("Chat request failed for model %s", model_id)
                if log:
                    msg = "timed out" if isinstance(result, asyncio.TimeoutError) else f"failed: {result}"
                    log.write(f"[error] Chat request {msg}\n")
                self._set_status(model_id, "error")
                continue
            if log:
                if self.chat_pending:
                    log.write("[system] Response received.\n")
                log.write(f"{model_id}: {result}\n")
            self._set_status(model_id, "ok")

        if self.chat_input:
            self.chat_input.disabled = False
            self.set_focus(self.chat_input)
        self.chat_pending = False

    async def toggle_model_selection(self, model_id: str, row_key: Any | None = None) -> None:
        table: DataTable = self.query_one("#models_table", DataTable)
        target_row_key = row_key if row_key is not None else model_id
        if model_id in self.selected_models:
            self.selected_models.remove(model_id)
            self.agents.pop(model_id, None)
            panel = self.chat_log_panels.pop(model_id, None)
            self.chat_logs.pop(model_id, None)
            if panel:
                removed = panel.remove()
                if inspect.isawaitable(removed):
                    await removed
            try:
                col_key = self.selected_column_key or "Selected"
                table.update_cell(
                    target_row_key,
                    col_key,
                    self._selected_indicator(model_id, selected=False),
                )
            except Exception:
                _logger.warning("Unable to update selection cell for %s", model_id)
            if self.chat_hint:
                hint = (
                    "Select one or more models to start chatting."
                    if not self.selected_models
                    else f"Chatting with: {', '.join(sorted(self.selected_models))}"
                )
                self.chat_hint.update(hint)
            if self.chat_input and not self.selected_models:
                self.chat_input.placeholder = "Type a message and press Enter"
            return

        agent = UserResponseAgent(model=model_id, session_name=self.session_label)
        self.agents[model_id] = agent
        self.selected_models.add(model_id)
        await self._add_chat_panel(model_id, agent)
        try:
            col_key = self.selected_column_key or "Selected"
            table.update_cell(
                target_row_key,
                col_key,
                self._selected_indicator(model_id, selected=True),
            )
        except Exception:
            _logger.warning("Unable to update selection cell for %s", model_id)
        if self.chat_hint:
            self.chat_hint.update(
                f"Chatting with: {', '.join(sorted(self.selected_models))}"
            )
        if self.chat_input:
            self.chat_input.placeholder = f"Message for {', '.join(sorted(self.selected_models))}"

    def _selected_indicator(
        self, model_id: str, selected: Optional[bool] = None
    ) -> str:
        state = (
            self.selected_models.__contains__(model_id)
            if selected is None
            else selected
        )
        # Use simple text markers to avoid markup incompatibility across Textual versions.
        return "☑" if state else "☐"

    async def _add_chat_panel(self, model_id: str, agent: UserResponseAgent) -> None:
        if not self.chat_log_container:
            return
        panel_id = f"chat_panel_{self._slug(model_id)}"
        log_widget = Log(
            id=f"chat_log_{self._slug(model_id)}",
            highlight=False,
            max_lines=500,
        )
        self.chat_logs[model_id] = log_widget
        try:
            log_display = agent.log_path.relative_to(Path.cwd())
        except ValueError:
            log_display = agent.log_path

        panel = Vertical(
            Static(f"{model_id}", id=f"chat_label_{self._slug(model_id)}"),
            Static(f"Log: {log_display}", id=f"chat_log_path_{self._slug(model_id)}"),
            self._status_widget(model_id),
            log_widget,
            id=panel_id,
        )
        self.chat_log_panels[model_id] = panel
        mounted = self.chat_log_container.mount(panel)
        if inspect.isawaitable(mounted):
            await mounted
        log_widget.write(f"[system] Now chatting with {model_id}\n")
        log_widget.write(f"[system] Log file: {log_display}\n")

    def _slug(self, label: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)

    def _status_widget(self, model_id: str) -> Static:
        status = Static("[idle]", id=f"chat_status_{self._slug(model_id)}")
        self.chat_status[model_id] = status
        return status

    def _set_status(self, model_id: str, state: str) -> None:
        status = self.chat_status.get(model_id)
        if not status:
            return
        if state == "waiting":
            status.update("[waiting...]")
        elif state == "ok":
            status.update("[ok]")
        elif state == "error":
            status.update("[error]")
        else:
            status.update(f"[{state}]")


def run_tui() -> None:
    app = ModelTableApp()
    app.run()
