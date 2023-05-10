import os

# from chromadb.api.types import Include, QueryResult
from functools import lru_cache
from pathlib import Path
from typing import Literal

import chromadb
import chromadb.utils.embedding_functions as eb
from chromadb.config import Settings


from talking_python.models import TranscriptPassage


@lru_cache
def get_client(chromadb_dir: Path) -> "chromadb.Client":
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(chromadb_dir),
        )
    )


def get_embedding_fn(
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    type_: Literal["sentence_transformers", "hugging_face"] = "sentence_transformers",
) -> eb.SentenceTransformerEmbeddingFunction | eb.HuggingFaceEmbeddingFunction:
    r"""Get the function (and model) to embed the texts.

    The embedding model is chosen for semantic search:
    https://www.sbert.net/docs/pretrained_models.html#semantic-search

    The prefect task has retry delay according to
    [hugging face](https://huggingface.co/blog/getting-started-with-embeddings),
    if its called for 'hugging_face'.

    Args:
        model_name (str, optional): _description_. Defaults to MODEL_EMBEDDER.

    Returns:
        eb.SentenceTransformerEmbeddingFunction: _description_

    Example:

        ```pythom
        >>> ds = get_dataset()
        >>> embedder = get_embedding_fn()
        >>> embedder(next(ds)["contents"])
        ```
    """
    if type_ == "sentence_transformers":
        return eb.SentenceTransformerEmbeddingFunction(model_name=model_name)
    elif type_ == "hugging_face":
        api_key = os.environ.get("HUGGINGFACE_APIKEY")
        embed_fn = eb.HuggingFaceEmbeddingFunction(
            api_key, model_name=f"sentence-transformers/{model_name}"
        )
        # Force downloading the model, to have it ready once it's called
        # to embed texts
        embed_fn(["sample text"])
        return embed_fn
    else:
        raise ValueError(f"Type of embedding not defined: {type_}")


class Chroma:
    """Wrapper around Chroma client."""

    def __init__(
        self,
        client: "chromadb.Client" = None,
        collection_name: str = "talking_python_transcripts",
        embedding_fn: eb.EmbeddingFunction = None,
    ) -> None:
        """
        Args:
            client (chromadb.Client): _description_
            collection_name (str, optional): _description_. Defaults to None.
            embedding_fn (eb.EmbeddingFunction, optional): _description_. Defaults to None.
        """
        self.client = client or get_client()
        self.embedding_fn = embedding_fn or get_embedding_fn()

        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )

    def add(self, passages: list[TranscriptPassage]) -> int:
        self.collection.add(
            ids=[p.id for p in passages],
            embeddings=[p.embedding for p in passages],
            metadatas=[{"title": p.title, "line": p.line} for p in passages],
        )
        self.client.persist()
        return len(passages)

