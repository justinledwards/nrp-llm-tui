import os
from dataclasses import dataclass


@dataclass
class NRPConfig:
    api_key: str
    base_url: str = "https://ellm.nrp-nautilus.io/v1"

    @classmethod
    def from_env(cls) -> "NRPConfig":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return cls(api_key=api_key)

