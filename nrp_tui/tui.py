import asyncio
import inspect
import itertools
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Input,
    Log,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection
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
        self.model_list: Optional[SelectionList[str]] = None
        self.chat_status: Dict[str, Static] = {}
        self.chat_hint: Optional[Static] = None
        self.chat_input: Optional[Input] = None
        self.status_spinners: Dict[str, Any] = {}
        self.session_label = "tui"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Static("NRP Managed LLMs", id="title"),
                SelectionList[str](id="models_list"),
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
        # Note: older Textual versions error on subscripted generics in query_one.
        self.model_list = self.query_one("#models_list", SelectionList)
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
        if self.model_list:
            self.set_focus(self.model_list)

    def load_models(self) -> None:
        self.loading = True
        models = self.client.list_models()
        self._models = models
        if self.model_list:
            # Preserve existing selections
            current_selected = set(self.selected_models)
            current_selected.update(
                str(getattr(sel, "value", sel)) for sel in self.model_list.selected
            )
            self.model_list.clear_options()
            options: List[Selection[str]] = []
            for m in models:
                created_str = m["created"].strftime("%Y-%m-%d") if m["created"] else ""
                context_str = str(m["context_tokens"]) if m["context_tokens"] else ""
                label_parts = [m["id"]]
                if m["status"]:
                    label_parts.append(f"[{m['status']}]")
                if m["parameters"]:
                    label_parts.append(str(m["parameters"]))
                if context_str:
                    label_parts.append(f"ctx {context_str}")
                if created_str:
                    label_parts.append(created_str)
                label = " ".join(label_parts)
                options.append(
                    Selection(
                        label,
                        m["id"],
                        m["id"] in current_selected,
                    )
                )
            for opt in options:
                # add_option is most compatible across Textual versions
                self.model_list.add_option(opt)

        self.loading = False

    def action_refresh(self) -> None:
        """Refresh model list."""
        self.load_models()

    @on(SelectionList.SelectedChanged)
    async def handle_selection_changed(self, event: SelectionList.SelectedChanged) -> None:
        selection_list = getattr(event, "selection_list", None) or self.model_list
        if not selection_list:
            return
        selections = selection_list.selected
        new_selected = {
            str(getattr(sel, "value", sel)) for sel in selections  # type: ignore[attr-defined]
        }
        to_add = new_selected - self.selected_models
        to_remove = self.selected_models - new_selected

        for model_id in list(to_remove):
            await self._remove_model(model_id)
        for model_id in list(to_add):
            await self._add_model(model_id)

        self.selected_models = new_selected
        if self.chat_hint:
            hint = (
                "Select one or more models to start chatting."
                if not self.selected_models
                else f"Chatting with: {', '.join(sorted(self.selected_models))}"
            )
            self.chat_hint.update(hint)
        if self.chat_input:
            if self.selected_models:
                self.chat_input.placeholder = (
                    f"Message for {', '.join(sorted(self.selected_models))}"
                )
            else:
                self.chat_input.placeholder = "Type a message and press Enter"
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

    async def _remove_model(self, model_id: str) -> None:
        self.selected_models.discard(model_id)
        self.agents.pop(model_id, None)
        panel = self.chat_log_panels.pop(model_id, None)
        self.chat_logs.pop(model_id, None)
        self.chat_status.pop(model_id, None)
        spinner = self.status_spinners.pop(model_id, None)
        if spinner:
            try:
                spinner.stop()
            except Exception:
                pass
        if panel:
            removed = panel.remove()
            if inspect.isawaitable(removed):
                await removed

    async def _add_model(self, model_id: str) -> None:
        if model_id in self.agents:
            return
        agent = UserResponseAgent(model=model_id, session_name=self.session_label)
        self.agents[model_id] = agent
        await self._add_chat_panel(model_id, agent)

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
        # Simple text spinner via interval timer to keep dependencies minimal.
        if state == "waiting":
            if model_id in self.status_spinners:
                return
            spinner_iter = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

            def spin() -> None:
                status.update(f"[waiting {next(spinner_iter)}]")

            timer = self.set_interval(0.1, spin)
            self.status_spinners[model_id] = timer
        else:
            timer = self.status_spinners.pop(model_id, None)
            if timer:
                try:
                    timer.stop()
                except Exception:
                    pass
            if state == "ok":
                status.update("[ok]")
            elif state == "error":
                status.update("[error]")
            else:
                status.update(f"[{state}]")


def run_tui() -> None:
    app = ModelTableApp()
    app.run()
