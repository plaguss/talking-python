"""Prefect flow to embed the passages of the podcasts. 

There are problems when running in a local DaskTaskRunner in the
ephemeral mode:
https://github.com/PrefectHQ/prefect/issues/7277
Prior to running the flow, `prefect orion start`, set
the PREFECT_API_URL as informed.
At least locally, its better to limit the maximum number of concurrent processes,
or the computer may crash.

```console
prefect server start
```

Run the command that's shown in the console.

```console
python flows/embed.py
```

To see the evolution of the flow:
http://127.0.0.1:8787/status

"""

import os
import uuid
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Iterator, Literal, TypedDict

import chromadb
import chromadb.utils.embedding_functions as eb
from chromadb.api.models import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
from dotenv import dotenv_values
from prefect import flow, task
from prefect.logging import get_logger
from pydantic import BaseModel

config = dotenv_values(".env")

_root: Path = Path(__file__).parent.parent.resolve()
flow_results: Path = _root / "flow_results"
cleaned_transcripts: Path = flow_results / "cleaned_transcripts"
chromadb_dir: Path = _root / "chroma"
# flow_environ_local: bool = True if config.get("FLOW_ENVIRON") == "local" else False
flow_environ_local = False
checkpoint_path: Path = flow_results / ".embedded_files.txt"

log = get_logger("embed")


if not chromadb_dir.is_dir():
    chromadb_dir.mkdir()

if not checkpoint_path.exists():
    EMBEDDED_FILES = set()
else:
    with open(checkpoint_path, "r") as f:
        EMBEDDED_FILES = set([name.replace("\n", "") for name in f.readlines()])


if flow_environ_local:
    from prefect_dask.task_runners import DaskTaskRunner

    task_runner = DaskTaskRunner(adapt_kwargs={"maximum": 4})
else:
    from prefect.task_runners import SequentialTaskRunner

    task_runner = SequentialTaskRunner()


def read_transcript(filename: Path) -> list[str]:
    """Read a clean transcript from a .txt file.

    Args:
        filename (Path):

    Returns:
        list[str]: Contents of the transcript.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().splitlines()


class TranscriptSlice(TypedDict):
    filename: str
    contents: list[str]
    lines: list[int]


def get_dataset(
    directory: Path = cleaned_transcripts, batch_size: int = 64
) -> Iterator[TranscriptSlice]:
    """_summary_

    Args:
        directory (Path, optional): _description_. Defaults to cleaned_transcripts.
        batch_size (int, optional): _description_. Defaults to 32.

    Yields:
        _type_: _description_

    Example:
        >>> dataset = get_dataset()
        >>> next(dataset)
        {'filename': '000.txt',
        'contents': ("This very short episode is just a ..."),
        'ids': [0, 1, 2, 3]}
    """
    for filename in sorted(directory.iterdir()):
        transcript = read_transcript(filename)
        it = iter(transcript)
        filename = filename.name.replace("_clean", "")
        ctr = 0
        prev_bs = 0
        # See reference for batched: https://docs.python.org/3/library/itertools.html
        while batch := tuple(islice(it, batch_size)):
            bs = len(batch)
            from_ = prev_bs * ctr
            to_ = from_ + bs
            yield TranscriptSlice(
                filename=filename,
                contents=batch,
                lines=list(range(from_, to_)),
            )
            prev_bs = bs
            ctr += 1


@lru_cache
def get_client(persist_directory: Path = chromadb_dir) -> "chromadb.Client":
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_directory),
        )
    )


# TODO: Move the function to a task, to allow retrying the first time
# its called.
@task(retries=3, retry_delay_seconds=20)
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


class TranscriptPassage(BaseModel):
    """A piece from a transcript."""

    id: str
    line: str
    title: str
    embedding: list[float]


@task
def embed_transcript_slice(
    transcript_slice: TranscriptSlice, embedding_fn: EmbeddingFunction = None,
) -> list[TranscriptPassage]:
    try:
        embeddings = embedding_fn(transcript_slice["contents"])
    except Exception as e:
        # Yet to decide what to do here.
        log.warning(f"ERROR: {e}")

    title = transcript_slice["filename"]
    transcript_passages = [
        TranscriptPassage(
            title=title, id=str(uuid.uuid4()), embedding=emb, line=str(line)
        )
        for emb, line in zip(embeddings, transcript_slice["lines"])
    ]
    return transcript_passages


# NOTE: I can't find the error, but if this function is run as a task,
# the program crashes due to:
# 20:58:47.103 | ERROR   | Task run 'add_to_collection-0' 
# - Crash detected! Execution was interrupted by an unexpected exception:
#  TypeError: Collection.__init__() missing 1 required positional argument: 'client'
@task
def add_to_collection(
    passages: list[TranscriptPassage], collection: Collection = None,
) -> int:
    """Adds the collections to the chroma database.

    Args:
        passages (list[TranscriptPassage]):
            List of transcript passages to register and persist on Chroma.
        collection (Collection):
            Chromadb colletion, obtained from the client. Defaults to None.

    Returns:
        int: Number of documents added.
    """
    if isinstance(passages, TranscriptPassage):
        passages = [passages]
    collection.add(
        ids=[p.id for p in passages],
        embeddings=[p.embedding for p in passages],
        metadatas=[{"title": p.title, "line": p.line} for p in passages],
    )
    return len(passages)


@flow(task_runner=task_runner)
def embed_transcripts():
    """_summary_"""
    dataset: Iterator[TranscriptSlice] = get_dataset(directory=cleaned_transcripts)

    chroma_client: "chromadb.Client" = get_client()
    embedding_fn = get_embedding_fn(model_name="multi-qa-MiniLM-L6-cos-v1")
    collection = chroma_client.get_or_create_collection(
        name="talking_python_embeddings", embedding_function=embedding_fn
    )

    try:
        for passage in dataset:
            filename = passage["filename"]
            s, e = passage["lines"][0], passage["lines"][-1]

            if (filename + str(e)) in EMBEDDED_FILES:
                log.info(f"Already processed: {filename} ({s}-{e})")
                continue

            log.info(f"Embedding passage: ({passage['filename']}, lines: {s} to {e})")
            transcript_passages = embed_transcript_slice.submit(
                passage, embedding_fn=embedding_fn
            )
            # NOTE: To properly run in the executor this should be in a 
            # separate task, but I cannot make it work
            collection.add(
                ids=[p.id for p in transcript_passages.result()],
                embeddings=[p.embedding for p in transcript_passages.result()],
                metadatas=[{"title": p.title, "line": p.line} for p in transcript_passages.result()],
            )
            EMBEDDED_FILES.add(filename + str(e))
            # add_to_collection.submit(transcript_passages, collection=collection)

    finally:
        log.info("Persist to disk and add checkpoint")
        chroma_client.persist()
        with open(checkpoint_path, "w") as f:
            f.writelines("\n".join(list(EMBEDDED_FILES)))


if __name__ == "__main__":
    embed_transcripts()
