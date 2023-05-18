"""Prefect flow to embed the passages of the podcasts. 

To run the flow locally:
```console
prefect server start
```

Run the command that's shown in the console.

```console
python flows/embed.py
```

Note:
    The flow is run sequentially. Different errors appear trying with
    DaskTaskExectuor, for the moment it wasn't worth the effort.
"""

import os
import uuid
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Iterator, Literal, TypedDict

import chromadb
import chromadb.utils.embedding_functions as eb
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings

from dotenv import load_dotenv
from prefect import flow, task
from prefect.logging import get_logger
from pydantic import BaseModel
from prefect.task_runners import SequentialTaskRunner

load_dotenv()

_root: Path = Path(__file__).parent.parent.resolve()
flow_results: Path = _root / "flow_results"
cleaned_transcripts: Path = flow_results / "cleaned_transcripts"
chromadb_dir: Path = _root / "chroma"
# flow_environ_local: bool = True if config.get("FLOW_ENVIRON") == "local" else False
# flow_environ_local = False
checkpoint_path: Path = flow_results / ".embedded_files.txt"

COLLECTION_NAME = "talking_python_embeddings"

log = get_logger("embed")

# Couldn't make it work with other task runners, the task should be rewritten.
task_runner = SequentialTaskRunner()

if not chromadb_dir.is_dir():
    chromadb_dir.mkdir()

# NOTE: The following checkpoint file only works if the batch size
# for the dataset generator is 64
if not checkpoint_path.exists():
    EMBEDDED_FILES = set()
else:
    with open(checkpoint_path, "r") as f:
        EMBEDDED_FILES = set([name.replace("\n", "") for name in f.readlines()])


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
    if persist_directory.is_dir():
        # Warning just to know if the chroma directory was found
        # when running with github actions
        if not (persist_directory / "chroma-embeddings.parquet").is_file():
            log.warning(f"There is no previous chroma data collected at: {persist_directory / 'chroma-embeddings.parquet'}")
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_directory),
        )
    )


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
        api_key = os.environ.get("HF_ACCESS_TOKEN")
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


class TranscriptPassage(BaseModel):
    """A piece from a transcript."""

    id: str
    line: str
    title: str
    embedding: list[float]


@task
def embed_transcript_slice(
    transcript_slice: TranscriptSlice,
    embedding_fn: EmbeddingFunction = None,
    client: "chromadb.Client" = None,
) -> list[TranscriptPassage]:
    """Function to embed a transcript slice and adding the contents
    to a collection.

    Note:
        This task is longer than it should, but due to different errors
        when running the flow (inside the DaskTaskRunner or the ConcurrentRunner),
        it does different 'subtasks' inside: embedding the passages,
        creating the TranscriptPassage models, grabbing the collection
        from the client, and inserting the content. The problem actually happens
        when the collection is passed through a function.

    Args:
        transcript_slice (TranscriptSlice):
            Piece of the dataset to embed.
        embedding_fn (EmbeddingFunction, optional):
            Function to use to embed the text. Defaults to None.
        client (chromadb.Client, optional): _description_. Defaults to None.

    Returns:
        list[TranscriptPassage]: _description_
    """
    try:
        embeddings = embedding_fn(transcript_slice["contents"])
    except Exception as e:
        # Yet to decide what to do here.
        log.warning(f"There was an error running: {e}, try again with the default")
        embedding_fn = get_embedding_fn()
        embeddings = embedding_fn(transcript_slice["contents"])

    title = transcript_slice["filename"]
    transcript_passages = [
        TranscriptPassage(
            title=title, id=str(uuid.uuid4()), embedding=emb, line=str(line)
        )
        for emb, line in zip(embeddings, transcript_slice["lines"])
    ]
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    collection.add(
        ids=[p.id for p in transcript_passages],
        embeddings=[p.embedding for p in transcript_passages],
        metadatas=[{"title": p.title, "line": p.line} for p in transcript_passages],
    )

    return transcript_passages


@flow(task_runner=task_runner)
def embed_transcripts(
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    embedding_function: str = "sentence_transformers"
):
    """_summary_

    Args:
        model_name (str, optional): _description_. Defaults to "multi-qa-MiniLM-L6-cos-v1".
        embedding_function (str, optional): _description_. Defaults to "sentence_transformers".
    """
    dataset: Iterator[TranscriptSlice] = get_dataset(directory=cleaned_transcripts)

    chroma_client: "chromadb.Client" = get_client()
    # Check the state, in case there was some error with the api key
    # for Hugging face
    try:
        embedding_fn = get_embedding_fn(model_name=model_name, type_=embedding_function)
    except ValueError as e:
        log.warning("Getting the original embedding function failed, retrying with 'sentence_transformers'")
        embedding_fn = get_embedding_fn(model_name=model_name, type_="sentence_transformers")

    try:
        for passage in dataset:
            filename = passage["filename"]
            s, e = passage["lines"][0], passage["lines"][-1]

            if (filename + str(e)) in EMBEDDED_FILES:
                log.info(f"Already processed: {filename} ({s}-{e})")
                continue

            log.info(f"Embedding passage: ({passage['filename']}, lines: {s} to {e})")
            embed_transcript_slice.submit(
                passage, embedding_fn=embedding_fn, client=chroma_client
            )
            EMBEDDED_FILES.add(filename + str(e))

    finally:
        log.info("Persist to disk and add checkpoint")
        chroma_client.persist()
        # NOTE: Writing to a file like this can only be done
        # when the flow is running sequentially, otherwise it can write the names
        # of the files before actually finishing the job.
        with open(checkpoint_path, "w") as f:
            f.writelines("\n".join(sorted(EMBEDDED_FILES)))


if __name__ == "__main__":
    # A single argument can be passed to run with a different
    # embedding function:
    # python flows/embed.py hugging_face
    import sys
    if len(sys.argv) > 1:
        embedding_function = sys.argv[1]
    else:
        embedding_function = "sentence_transformers"
    embed_transcripts(embedding_function=embedding_function)
