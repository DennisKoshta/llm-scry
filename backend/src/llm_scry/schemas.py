from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelLoadRequest(BaseModel):
    name: str = Field(..., description="TransformerLens model name, e.g. 'gpt2'")
    device: str | None = None


class ModelInfo(BaseModel):
    name: str
    n_layers: int
    n_heads: int
    d_model: int
    d_vocab: int
    device: str


class TopKAlternative(BaseModel):
    token_id: int
    token_str: str
    logprob: float


class TokenEvent(BaseModel):
    position: int
    token_id: int
    token_str: str
    logprob: float
    top_k: list[TopKAlternative]


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 32
    top_k: int = 10
    temperature: float = 0.0


class GenerateResponse(BaseModel):
    session_id: str


class SessionSummary(BaseModel):
    id: str
    model_name: str
    prompt: str
    n_generated: int
    created_at: datetime
    status: str  # "running" | "complete" | "error"


class AttentionResponse(BaseModel):
    layer: int
    head: int | None
    tokens: list[str]
    pattern: list[list[float]]  # [query, key]
