from datetime import datetime
from typing import List, Dict, Any

from openai import OpenAI

from .config import NRPConfig
from .metadata import MODEL_METADATA, ModelMeta


class NRPClient:
    def __init__(self, cfg: NRPConfig | None = None) -> None:
        self.cfg = cfg or NRPConfig.from_env()
        self.client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Returns a list of model info dicts combining OpenAI /models
        response with static metadata where available.
        """
        models_page = self.client.models.list()
        results: List[Dict[str, Any]] = []

        for model in models_page.data:
            model_id = model.id
            created = getattr(model, "created", None)
            created_dt = (
                datetime.fromtimestamp(created) if isinstance(created, int) else None
            )

            meta: ModelMeta | None = MODEL_METADATA.get(model_id)

            results.append(
                {
                    "id": model_id,
                    "created": created_dt,
                    "status": meta.status if meta else None,
                    "title": meta.title if meta else None,
                    "parameters": meta.parameters if meta else None,
                    "context_tokens": meta.context_tokens if meta else None,
                    "features": meta.features if meta else None,
                    "notes": meta.notes if meta else None,
                }
            )

        # Sort by status then id for a nicer view
        status_order = {"main": 0, "eval": 1, "dep": 2, None: 3}
        results.sort(key=lambda m: (status_order.get(m["status"]), m["id"]))
        return results

