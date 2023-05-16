import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import chromadb
import chromadb.utils.embedding_functions as eb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Include, QueryResult
from chromadb.config import Settings

EmbeddinFunction = Literal["sentence_transformers", "hugging_face"]


@lru_cache
def get_client(chromadb_dir: Path) -> "chromadb.Client":
    """Get the chroma client, only expected to work locally on duckdb+parquet.

    Args:
        chromadb_dir (Path): Path where the results will be written
            when persisted.

    Returns:
        chromadb.Client
    """
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(chromadb_dir),
        )
    )


def get_embedding_fn(
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    type_: EmbeddinFunction = "sentence_transformers",
) -> eb.SentenceTransformerEmbeddingFunction | eb.HuggingFaceEmbeddingFunction:
    r"""Get the function to embed the texts.

    The embedding model is chosen for semantic search:
    https://www.sbert.net/docs/pretrained_models.html#semantic-search

    Args:
        model_name (str, optional):
            Model name, its assumed to be one from sentence-transformers.
        type_ (str, optional):
            Only tested to work with SentenceTransformerEmbeddingFunction
            and HuggingFaceEmbeddingFunction (as it works directly with the REST API,
            its better not to run a lot of embeddings through this function).

    Returns:
        eb.SentenceTransformerEmbeddingFunction:
            Callable function ready to embed text.

    Example:

        ```pythom
        >>> embedder = get_embedding_fn()
        >>> text = "That's excellent. That is Eliot Horowitz who is the CTO of MongoDB
        right?"
        >>> embedder(text)
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
        response = embed_fn(["sample text"])
        if "error" in response.keys():
            # Raise error if fails loading the embedding function.
            raise ValueError(
                f"Loading HuggingFaceEmbeddingFunction, response: {response}."
            )
        return embed_fn

    else:
        raise ValueError(f"Type of embedding not defined: {type_}")


class Chroma:
    """Wrapper around Chroma client.

    It just exposes the functionality we need. Its only expected to deal with a single
    collection.

    Example:
        >>> client = Chroma(chroma_dir=Path("../talking_python/chroma"))
    """

    def __init__(
        self,
        client: "chromadb.Client" = None,
        chroma_dir: Path = None,
        collection_name: str = "talking_python_embeddings",
        embedding_fn: eb.EmbeddingFunction = None,
    ) -> None:
        """
        Args:
            client (chromadb.Client):
                Chroma client instance.
            chroma_dir (Path):
                If client is not given, the path to a chroma directory must be 
                informed
            collection_name (str, optional):
                Name of the collection.
            embedding_fn (eb.EmbeddingFunction, optional):
                Function to embed the texts. Defaults to None, in which case
                is obtained internally.
        """
        self._chroma_dir = chroma_dir
        self.client = client or get_client(self._chroma_dir)
        self.embedding_fn = embedding_fn or get_embedding_fn()

        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )

    def __repr__(self) -> str:
        return type(self).__name__ + f"(chroma_dir={self._chroma_dir})"

    def query(
        self,
        query_embeddings: list[list[float]] = None,
        query_texts: list[str] = None,
        n_results: int = 10,
        where: dict = None,
        where_document: dict = None,
        include: "Include" = ["metadatas", "distances"],
    ) -> QueryResult:
        """The method just redirects to query:
        https://github.com/chroma-core/chroma/blob/main/chromadb/api/models/Collection.py#L160

        Args:
            query_embeddings:
                The embeddings to get the closes neighbors of. Optional.
            query_texts:
                The document texts to get the closes neighbors of. Optional.
            n_results:
                The number of neighbors to return for each query_embedding or
                query_text. Optional.
            where:
                A Where type dict used to filter results by.
                E.g. {"title" : "111.txt"}. Optional.
            where_document:
                A WhereDocument type dict used to filter by the documents.
                E.g. {$contains: {"text": "hello"}}. Optional.
            include:
                A list of what to include in the results.
                Can contain "embeddings", "metadatas", "documents", "distances".
                Ids are always included. Defaults to
                ["metadatas", "documents", "distances"].
                Optional.

        Returns:
            QueryResult: A QueryResult object containing the results.

        Raises:
            ValueError: If you don't provide either query_embeddings or query_texts
            ValueError: If you provide both query_embeddings and query_texts

        Examples:

            An example to find text present that is indeed in a given podcast
            (just to check the content stored). In this case,
            in the episode 111 (the name of the corresponding file is 111.txt),
            we can find the following text in the line 178 (the second to last):
            'Matt Harrison: Okay, we'll see you.' The smallest distance corresponds
            to the line where this same phrase occurs.

            ```python
            >>> client.query(query_texts=["Okay, we'll see you."],
            ... where={"title": "111.txt"})
            {'ids': [['8eb712b1-5f82-4bbd-9c84-c6ce42116b24',
            'fe264b60-d827-4c99-95c8-6ff7a8bc7b55',
            '6756ce17-e107-42c5-838d-dfe4a3036b67',
            'fdce08dc-fc21-417a-9b76-e51912584255',
            'ccc7272b-e9dc-45be-9cde-aa616ee4f288',
            'd7655312-e816-4713-8b4e-f9232415ffa8',
            '5c869a1c-2730-46c0-aa9b-b608c97371ba',
            '65b63c60-7008-4a4a-a654-815e18cf6ea7',
            '09199113-1b58-4311-b0c1-6f890bba1a17',
            '7e2ef25f-7842-4417-8fe9-0310e2aca1d7']],
            'metadatas': [[{'title': '111.txt', 'line': '177'},
            {'title': '111.txt', 'line': '170'},
            {'title': '111.txt', 'line': '1'},
            {'title': '111.txt', 'line': '171'},
            {'title': '111.txt', 'line': '174'},
            {'title': '111.txt', 'line': '175'},
            {'title': '111.txt', 'line': '159'},
            {'title': '111.txt', 'line': '2'},
            {'title': '111.txt', 'line': '156'},
            {'title': '111.txt', 'line': '160'}]],
            'distances': [[0.7800952792167664,
            1.166372537612915,
            1.2460498809814453,
            1.2463676929473877,
            1.270250916481018,
            1.295353651046753,
            1.2983410358428955,
            1.3043357133865356,
            1.3078633546829224,
            1.3153605461120605]]}
            ```
        """
        return self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )

    def count(self) -> int:
        return self.collection.count()
