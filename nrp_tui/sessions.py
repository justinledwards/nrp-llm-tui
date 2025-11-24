from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .logging_utils import LOG_DIR, slugify


@dataclass
class Session:
    """
    Represents a chat session that may span multiple models.
    """

    id: str
    label: str
    display_name: str
    created_at: datetime
    path: Path
    title: Optional[str] = None

    @property
    def metadata_path(self) -> Path:
        return self.path / "session.json"

    @property
    def created_tag(self) -> str:
        return self.created_at.strftime("%Y%m%d-%H%M%S")


class SessionStore:
    """
    Manages chat sessions on disk under the logs directory.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or LOG_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        label: str,
        *,
        title: Optional[str] = None,
        created_at: Optional[datetime] = None,
        display_name: Optional[str] = None,
    ) -> Session:
        created = created_at or datetime.now()
        slug = slugify(label, "session")
        session_id = f"{slug}-{created.strftime('%Y%m%d-%H%M%S')}"
        path = self.base_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        session = Session(
            id=session_id,
            label=slug,
            display_name=display_name or label,
            created_at=created,
            path=path,
            title=title,
        )
        self._write_metadata(session)
        return session

    def get_or_create(self, label: str, *, title: Optional[str] = None, resume: bool = True) -> Session:
        """
        Returns the latest session matching the label (if resume=True), or creates one.
        """
        if resume:
            existing = self.find_latest_by_label(label)
            if existing:
                return existing
        return self.create(label, title=title)

    def find_latest_by_label(self, label: str) -> Optional[Session]:
        slug = slugify(label, "session")
        sessions = [s for s in self.list_sessions() if s.label == slug]
        if not sessions:
            return None
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[0]

    def list_sessions(self) -> List[Session]:
        """
        Reads session metadata from disk. Sessions without metadata are ignored.
        """
        sessions: List[Session] = []
        for path in self.base_dir.iterdir():
            if not path.is_dir():
                continue
            meta_path = path / "session.json"
            if not meta_path.exists():
                continue
            try:
                session = self._read_metadata(meta_path)
            except Exception:
                continue
            sessions.append(session)
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def load(self, session_id: str) -> Session:
        meta_path = self.base_dir / session_id / "session.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Session metadata not found for id '{session_id}'")
        return self._read_metadata(meta_path)

    def delete(self, session_id: str) -> bool:
        """
        Delete a session directory by id. Returns True if removed, False if missing.
        """
        path = self.base_dir / session_id
        if not path.exists():
            return False
        try:
            shutil.rmtree(path)
            return True
        except Exception:
            return False

    def _write_metadata(self, session: Session) -> None:
        data = {
            "id": session.id,
            "label": session.label,
            "display_name": session.display_name,
            "created_at": session.created_at.isoformat(),
            "title": session.title,
            "version": 1,
        }
        session.metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _read_metadata(self, meta_path: Path) -> Session:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        created_at = datetime.fromisoformat(data["created_at"])
        path = meta_path.parent
        return Session(
            id=data["id"],
            label=data.get("label") or data["id"],
            display_name=data.get("display_name") or data.get("label") or data["id"],
            created_at=created_at,
            path=path,
            title=data.get("title"),
        )
