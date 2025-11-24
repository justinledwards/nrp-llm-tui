from typing import List, Dict, Any

from openai import OpenAI

from pathlib import Path

from .config import NRPConfig
from .logging_utils import ConversationLogger


class SimpleAgent:
    """
    Very small wrapper around a single NRP model.
    Later you can add tool calls, planning, memory, and so on.
    """

    def __init__(self, model: str = "gemma3", cfg: NRPConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or NRPConfig.from_env()
        self.client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: list of dicts with {role, content}
        Note: Do not set max_tokens per NRP guidance.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content or ""


class UserResponseAgent(SimpleAgent):
    """
    Small helper that prepends a system prompt tuned for end-user responses.
    Keeps message history so the conversation has context.
    """

    SYSTEM_MESSAGE = (
        "You are User Response, a concise, friendly support agent for the "
        "Nautilus Research Platform. Answer the user directly, avoid meta "
        "commentary, and only request clarification when necessary. Be brief "
        "and actionable."
    )

    def __init__(
        self,
        model: str = "gemma3",
        cfg: NRPConfig | None = None,
        session_name: str | None = None,
    ) -> None:
        super().__init__(model=model, cfg=cfg)
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_MESSAGE}
        ]
        self.session_logger = ConversationLogger(model=model, session_name=session_name)
        self.session_logger.set_system_message(self.SYSTEM_MESSAGE)

    def send(self, user_message: str) -> str:
        """
        Appends the user message to history, sends to the model, and records the reply.
        """
        self.history.append({"role": "user", "content": user_message})
        self.session_logger.log_message("user", user_message)
        reply = self.chat(self.history)
        self.history.append({"role": "assistant", "content": reply})
        self.session_logger.log_message("assistant", reply)
        return reply

    @property
    def log_path(self) -> Path:
        return self.session_logger.log_path

    @property
    def session_dir(self) -> Path:
        return self.session_logger.session_dir
