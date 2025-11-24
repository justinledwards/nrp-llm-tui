from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import os
import pytest

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

from nrp_tui.logging_utils import ConversationLogger
from nrp_tui.sessions import SessionStore
from nrp_tui.tui import ModelTableApp


def test_session_store_create_and_resume(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    created_one = datetime(2024, 1, 1, 0, 0, 0)
    created_two = datetime(2024, 1, 2, 0, 0, 0)

    sess1 = store.create("demo", created_at=created_one)
    sess2 = store.create("demo", created_at=created_two)

    assert sess1.id == "demo-20240101-000000"
    assert sess2.id == "demo-20240102-000000"
    assert sess1.metadata_path.exists()
    assert json.loads(sess1.metadata_path.read_text())["id"] == sess1.id

    latest = store.get_or_create("demo", resume=True)
    assert latest.id == sess2.id  # Should pick the newest matching label


def test_conversation_logger_structured_history(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = store.create("demo", created_at=datetime(2024, 1, 1, 12, 0, 0))

    logger = ConversationLogger(model="gemma3", session=session)
    logger.set_system_message("system prompt")
    logger.log_message("user", "hello")
    logger.log_message("assistant", "hi there")

    messages = logger.read_messages()
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert ("user" in roles) and ("assistant" in roles)
    assert logger.jsonl_path.exists()
    # Ensure JSONL contains session_id for resume safety
    data = json.loads(logger.jsonl_path.read_text().splitlines()[-1])
    assert data["session_id"] == session.id


def test_discover_session_models(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = store.create("demo", created_at=datetime(2024, 1, 1, 12, 0, 0))

    # Create fake JSONL logs for two models
    suffix = f"-{session.label}-{session.created_tag}.jsonl"
    for model in ["gemma3", "gpt-oss"]:
        (session.path / f"{model}{suffix}").write_text("{}", encoding="utf-8")
    (session.path / "ignore.txt").write_text("nope", encoding="utf-8")

    app = ModelTableApp(session=session, resume=True, store=store)
    models = app._discover_session_models(session)
    assert sorted(models) == ["gemma3", "gpt-oss"]


@pytest.mark.asyncio
async def test_restore_previous_models_calls_add(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = store.create("demo", created_at=datetime(2024, 1, 1, 12, 0, 0))

    app = ModelTableApp(session=session, resume=True, store=store)
    app._discover_session_models = lambda sess: ["gemma3", "gpt-oss"]  # type: ignore[assignment]

    added = []

    async def fake_add(model_id: str) -> None:
        added.append(model_id)

    app._add_model = AsyncMock(side_effect=fake_add)  # type: ignore[assignment]
    await app._restore_previous_models()

    assert added == ["gemma3", "gpt-oss"]
