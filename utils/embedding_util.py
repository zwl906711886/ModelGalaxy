from langchain.embeddings import (
    HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
)
from langchain.embeddings.base import Embeddings
from configs.model_config import embedding_model_dict


def load_embedding(
        model: str,
) -> Embeddings:
    if 'bge-' in model:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_dict[model],
            query_instruction="为这个句子生成表示以用于检索相关文章：",
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[model],
        )
    return embeddings
