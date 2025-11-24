import re
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
# Directory is created lazily when the first log is written to avoid empty session folders.


def _slugify(label: str, default: str) -> str:
    """
    Convert a label into a filesystem-safe slug.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return slug or default


class ConversationLogger:
    """
    Write per-conversation chat transcripts to disk.

    Each conversation gets its own session folder under logs/ named
    "<session>-<timestamp>", and the log file inside is named
    "<model>-<session>-<timestamp>.log" to make multi-model comparisons easy.
    Directories/files are created lazily on first write to avoid empty logs.
    """

    def __init__(self, model: str, session_name: str | None = None) -> None:
        self.model = model
        self.session_label = (
            _slugify(session_name, "session") if session_name else "session"
        )
        self.started_at = datetime.now()
        timestamp = self.started_at.strftime("%Y%m%d-%H%M%S")

        self.session_dir = LOG_DIR / f"{self.session_label}-{timestamp}"

        safe_model = _slugify(self.model, "model")
        self.log_path = (
            self.session_dir / f"{safe_model}-{self.session_label}-{timestamp}.log"
        )
        self._initialized = False
        self._system_message: str | None = None

    def set_system_message(self, message: str) -> None:
        self._system_message = message

    def _ensure_file(self) -> None:
        if self._initialized:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        started = self.started_at.isoformat(timespec="seconds")
        header_lines = [
            f"conversation started: {started}",
            f"model: {self.model}",
            f"session: {self.session_label}",
        ]
        if self._system_message:
            header_lines.append(f"system: {self._system_message}")
        header_lines.append("")
        self.log_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")
        self._initialized = True

    def log_message(self, role: str, content: str) -> None:
        self._ensure_file()
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{timestamp}] {role}: {content}\n")
