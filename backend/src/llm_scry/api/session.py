from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..engine import engine
from ..schemas import AttentionResponse, PromptToken, SessionDetail
from ..sessions import store

router = APIRouter(prefix="/session", tags=["session"])


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str) -> SessionDetail:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    prompt_tokens = [
        PromptToken(token_id=i, token_str=s)
        for i, s in zip(
            session.prompt_token_ids, session.prompt_token_strs, strict=True
        )
    ]
    return SessionDetail(
        id=session.id,
        model_name=session.model_name,
        prompt=session.prompt,
        prompt_tokens=prompt_tokens,
        generated_tokens=session.tokens,
        status=session.status,
        error=session.error,
        created_at=session.created_at,
    )


@router.get("/{session_id}/attention", response_model=AttentionResponse)
def attention(session_id: str, layer: int, head: int | None = None) -> AttentionResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    if session.cache is None:
        raise HTTPException(status_code=409, detail="session activations not captured yet")
    if engine.model is None or engine.name != session.model_name:
        raise HTTPException(
            status_code=409, detail="model for this session is not currently loaded"
        )

    key = f"blocks.{layer}.attn.hook_pattern"
    cache = session.cache
    try:
        pattern = cache[key]  # type: ignore[index]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"activation not in cache: {key}") from e

    # Shape is typically [batch=1, head, query, key]
    tensor = pattern[0]
    if head is not None:
        tensor = tensor[head]
        data = tensor.detach().to("cpu").float().tolist()
    else:
        # Mean across heads for a coarse "layer view"
        data = tensor.mean(dim=0).detach().to("cpu").float().tolist()

    # Token strings across the full captured sequence (prompt + generated)
    prompt_tokens = session.prompt_token_strs
    generated_tokens = [t.token_str for t in session.tokens]
    tokens = prompt_tokens + generated_tokens

    return AttentionResponse(layer=layer, head=head, tokens=tokens, pattern=data)
