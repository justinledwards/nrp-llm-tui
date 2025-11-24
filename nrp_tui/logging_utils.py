import re
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


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
    """

    def __init__(self, model: str, session_name: str | None = None) -> None:
        self.model = model
        self.session_label = _slugify(session_name, "session") if session_name else "session"
        self.started_at = datetime.now()
        timestamp = self.started_at.strftime("%Y%m%d-%H%M%S")

        self.session_dir = LOG_DIR / f"{self.session_label}-{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        safe_model = _slugify(self.model, "model")
        self.log_path = self.session_dir / f"{safe_model}-{self.session_label}-{timestamp}.log"
        self._write_header()

    def _write_header(self) -> None:
        started = self.started_at.isoformat(timespec="seconds")
        header = [
            f"conversation started: {started}",
            f"model: {self.model}",
            f"session: {self.session_label}",
            "",
        ]
        self.log_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    def log_message(self, role: str, content: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{timestamp}] {role}: {content}\n")
