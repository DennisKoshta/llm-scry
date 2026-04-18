from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import generate as generate_api
from .api import model as model_api
from .api import session as session_api
from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(title="llm-scry", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(model_api.router)
    app.include_router(generate_api.router)
    app.include_router(session_api.router)

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"service": "llm-scry", "status": "ok"}

    return app


app = create_app()
