from typing import List, Dict, Any

from openai import OpenAI

from .config import NRPConfig


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

    def __init__(self, model: str = "gemma3", cfg: NRPConfig | None = None) -> None:
        super().__init__(model=model, cfg=cfg)
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_MESSAGE}
        ]

    def send(self, user_message: str) -> str:
        """
        Appends the user message to history, sends to the model, and records the reply.
        """
        self.history.append({"role": "user", "content": user_message})
        reply = self.chat(self.history)
        self.history.append({"role": "assistant", "content": reply})
        return reply
