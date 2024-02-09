from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from configs import VERSION
from configs.server_config import (
    OPEN_CROSS_DOMAIN,
)
from utils.llm_util import chat


def create_app():
    app = FastAPI(
        title="ChatChat-Neo4j",
        version=VERSION
    )

    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.post(
        "/chat",
        tags=["chat"],
        summary="llm交互"
    )(chat)

    return app
