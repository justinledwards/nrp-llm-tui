from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    from .sessions import Session

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
# Directory is created lazily when the first log is written to avoid empty session folders.


def slugify(label: str, default: str) -> str:
    """
    Convert a label into a filesystem-safe slug.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return slug or default


class ConversationLogger:
    """
    Write per-conversation chat transcripts to disk. Each session is
    responsible for creating its own folder; the logger only writes files.
    """

    def __init__(self, model: str, session: "Session") -> None:
        self.model = model
        self.session = session
        self.started_at = session.created_at
        self.session_dir = session.path

        safe_model = slugify(self.model, "model")
        base_name = f"{safe_model}-{session.label}-{session.created_tag}"
        self.log_path = self.session_dir / f"{base_name}.log"
        self.jsonl_path = self.session_dir / f"{base_name}.jsonl"
        self._initialized = False
        self._system_message: str | None = None
        self._jsonl_initialized = False

    def set_system_message(self, message: str) -> None:
        self._system_message = message
        if self._jsonl_initialized:
            # Ensure the system message is captured in structured logs when resuming.
            self._ensure_system_json_entry()

    def _ensure_file(self) -> None:
        if self._initialized:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            started = self.started_at.isoformat(timespec="seconds")
            header_lines = [
                f"conversation started: {started}",
                f"model: {self.model}",
                f"session: {self.session.display_name} ({self.session.id})",
            ]
            if self._system_message:
                header_lines.append(f"system: {self._system_message}")
            header_lines.append("")
            self.log_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")
        self._ensure_jsonl()
        self._initialized = True

    def _ensure_jsonl(self) -> None:
        if self._jsonl_initialized:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.touch(exist_ok=True)
        self._ensure_system_json_entry()
        self._jsonl_initialized = True

    def _ensure_system_json_entry(self) -> None:
        if not self._system_message:
            return
        if self._system_logged():
            return
        self._append_jsonl("system", self._system_message, self.started_at)

    def _system_logged(self) -> bool:
        if not self.jsonl_path.exists():
            return False
        try:
            for line in self.jsonl_path.open("r", encoding="utf-8"):
                obj = json.loads(line)
                if obj.get("role") == "system":
                    return True
        except Exception:
            return False
        return False

    def _append_jsonl(self, role: str, content: str, timestamp: datetime | None = None) -> None:
        ts = timestamp or datetime.now()
        record: Dict[str, str] = {
            "role": role,
            "content": content,
            "timestamp": ts.isoformat(timespec="seconds"),
            "model": self.model,
            "session_id": self.session.id,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_message(self, role: str, content: str) -> None:
        self._ensure_file()
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{timestamp}] {role}: {content}\n")
        self._append_jsonl(role, content)

    def read_messages(self) -> List[Dict[str, str]]:
        """
        Read structured history for this model in the session, if present.
        """
        if not self.jsonl_path.exists():
            if self._system_message:
                return [{"role": "system", "content": self._system_message}]
            return []
        messages: List[Dict[str, str]] = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            role = obj.get("role")
            content = obj.get("content")
            if role and content is not None:
                messages.append({"role": role, "content": content})
        if self._system_message and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self._system_message})
        return messages


# Backwards-compat alias for legacy imports
_slugify = slugify
