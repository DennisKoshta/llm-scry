from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ..engine import engine
from ..schemas import GenerateRequest, GenerateResponse, SessionSummary
from ..sessions import Session, store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])

# Each session has a queue of SSE events that the streamer consumes.
_queues: dict[str, asyncio.Queue[str | None]] = {}


async def _run_generation(session: Session, req: GenerateRequest) -> None:
    queue = _queues[session.id]
    try:
        tokens = engine.prepare_prompt(session, req.prompt)
        prompt_payload = json.dumps(
            {
                "type": "prompt",
                "tokens": [
                    {"token_id": i, "token_str": s}
                    for i, s in zip(
                        session.prompt_token_ids, session.prompt_token_strs, strict=True
                    )
                ],
            }
        )
        await queue.put(prompt_payload)

        async for event in engine.stream_generate(session, req, tokens):
            payload = json.dumps({"type": "token", **event.model_dump()})
            await queue.put(payload)
        session.status = "complete"
        done = json.dumps(
            {"type": "done", "session_id": session.id, "total_tokens": len(session.tokens)}
        )
        await queue.put(done)
    except Exception as e:
        logger.exception("generation failed")
        session.status = "error"
        session.error = str(e)
        await queue.put(json.dumps({"type": "error", "message": str(e)}))
    finally:
        await queue.put(None)  # sentinel closes the stream


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if engine.model is None:
        raise HTTPException(status_code=400, detail="no model loaded")
    session = store.new_session(model_name=engine.name or "", prompt=req.prompt)
    _queues[session.id] = asyncio.Queue()
    asyncio.create_task(_run_generation(session, req))
    return GenerateResponse(session_id=session.id)


@router.get("/generate/{session_id}/stream")
async def stream(session_id: str) -> EventSourceResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    queue = _queues.get(session_id)
    if queue is None:
        raise HTTPException(status_code=410, detail="session stream no longer available")

    async def events() -> AsyncIterator[dict[str, str]]:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield {"data": item}

    return EventSourceResponse(events())


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return [s.summary() for s in store.all()]
