from fastapi import FastAPI
from langchain import SQLDatabase
from langchain.graphs import Neo4jGraph
from starlette.middleware.cors import CORSMiddleware

from configs import VERSION
from configs.model_config import EMBEDDING_MODEL
from configs.database_config import DEFAULT_NEO4J_CONFIG, DEFAULT_MYSQL_CONFIG
from configs.server_config import OPEN_CROSS_DOMAIN
from entity.request import ChatRequest
from utils.embedding_util import load_embedding
from utils.faiss_util import FaissManager
from utils.llm_util import LLMManager

graph = Neo4jGraph(
    DEFAULT_NEO4J_CONFIG["url"],
    DEFAULT_NEO4J_CONFIG["username"],
    DEFAULT_NEO4J_CONFIG["password"],
    DEFAULT_NEO4J_CONFIG["database"],
)
db = SQLDatabase.from_uri(f"mysql+pymysql://{DEFAULT_MYSQL_CONFIG['user']}:{DEFAULT_MYSQL_CONFIG['password']}@{DEFAULT_MYSQL_CONFIG['host']}:{DEFAULT_MYSQL_CONFIG['port']}/{DEFAULT_MYSQL_CONFIG['name']}?charset=utf8mb4")

print("Loading embedding model...")
embeddings = load_embedding(EMBEDDING_MODEL)
faiss_manager = FaissManager(embeddings)
llm_manager = LLMManager(graph, db)


def chat(msg: ChatRequest):
    return llm_manager.chat(msg.query, faiss_manager)


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
