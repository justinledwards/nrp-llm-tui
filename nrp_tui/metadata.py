from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ModelMeta:
    id: str
    status: str  # "main", "eval", "dep"
    title: str
    parameters: Optional[str]
    context_tokens: Optional[int]
    features: str
    notes: Optional[str] = None


# Manually lifted from NRP docs as of Nov 2025.
# Keep ids in sync with /v1/models (qwen3, gpt-oss, etc).
MODEL_METADATA: Dict[str, ModelMeta] = {
    "qwen3": ModelMeta(
        id="qwen3",
        status="main",
        title="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
        parameters="235B",
        context_tokens=262_144,
        features="Vision, video, tool calling",
        notes="Frontier multimodal performance"
    ),
    "gpt-oss": ModelMeta(
        id="gpt-oss",
        status="eval",
        title="openai/gpt-oss-120b",
        parameters="120B",
        context_tokens=131_072,
        features="Tool calling, agentic",
        notes="MXFP4, good for batched prompts"
    ),
    "kimi": ModelMeta(
        id="kimi",
        status="eval",
        title="moonshotai/Kimi-K2-Thinking",
        parameters=None,
        context_tokens=262_144,
        features="Tool calling, coding focused",
        notes="Frontier general and coding performance"
    ),
    "glm-4.6": ModelMeta(
        id="glm-4.6",
        status="eval",
        title="QuantTrio/GLM-4.6-GPTQ-Int4-Int8Mix",
        parameters=None,
        context_tokens=202_752,
        features="Tool calling, coding",
        notes="GPTQ int4/int8 mixed"
    ),
    "minimax-m2": ModelMeta(
        id="minimax-m2",
        status="eval",
        title="MiniMaxAI/MiniMax-M2",
        parameters=None,
        context_tokens=262_144,
        features="Tool calling, coding",
        notes="Native FP8"
    ),
    "glm-v": ModelMeta(
        id="glm-v",
        status="eval",
        title="zai-org/GLM-4.5V-FP8",
        parameters=None,
        context_tokens=65_536,
        features="Vision, video, tool calling",
        notes="GPT-4o level multimodal"
    ),
    "gemma3": ModelMeta(
        id="gemma3",
        status="main",
        title="google/gemma-3-27b-it",
        parameters="27B",
        context_tokens=131_072,
        features="Vision, tool calling",
        notes="Recommended for batched prompts"
    ),
    "embed-mistral": ModelMeta(
        id="embed-mistral",
        status="main",
        title="intfloat/e5-mistral-7b-instruct",
        parameters="7B",
        context_tokens=None,
        features="Embeddings"
    ),
    "gorilla": ModelMeta(
        id="gorilla",
        status="eval",
        title="gorilla-llm/gorilla-openfunctions-v2",
        parameters=None,
        context_tokens=None,
        features="Function calling"
    ),
    "olmo": ModelMeta(
        id="olmo",
        status="eval",
        title="allenai/OLMo-2-0325-32B-Instruct",
        parameters="32B",
        context_tokens=None,
        features="General instruction model"
    ),
    "llama3-sdsc": ModelMeta(
        id="llama3-sdsc",
        status="dep",
        title="meta-llama/Llama-3.3-70B-Instruct",
        parameters="70B",
        context_tokens=131_072,
        features="Multilingual, tool use"
    ),
}

