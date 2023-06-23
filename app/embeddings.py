"""Functions related to  """

from talking_python import chroma
from functools import lru_cache
import streamlit as st
from typing import Callable
from collections import defaultdict


class AuthenticationError(Exception):
    pass


@lru_cache
def get_embedding_function(api_key: str | None = None) -> chroma.EmbeddingFunction:
    """Get the embedding function using the api key supplied
    by the user.

    Args:
        api_key (str | None):
            Hugging Face API token. If None (the default), will try to obtain it from
            the session state.

    Returns:
        chroma.EmbeddingFunction:
            Function to generate the embeddings, will be given to the
            chroma client.
    """
    # Function cached to avoid loading again when rerunning the script,
    # is necessary?
    # TOOD: Add retry:
    # https://github.com/mmz-001/knowledge_gpt/blob/main/knowledge_gpt/embeddings.py
    if api_key is None:
        api_key = st.session_state.get("HF_API_TOKEN")
        if api_key == "":
            raise AuthenticationError(
                "Enter your HuggingFace API token in the sidebar. "
                "You can get your API token from https://huggingface.co/settings/tokens."
            )
    return chroma.get_embedding_fn(type_="hugging_face", api_key=api_key)


def get_chroma(embedding_fn: chroma.EmbeddingFunction) -> chroma.Chroma:
    """Creates the chroma client using the preloaded embedding function
    to ensure its the same.

    Args:
        embedding_fn (chroma.EmbeddingFunction):
            Function to create the embedding vectors. Its expected
            to be a hugging face endpoint.

    Returns:
        chroma.Chroma: client to query the dataset.
    """
    return chroma.Chroma(embedding_fn=embedding_fn)


MetadataType = list[dict[str, str]]
DistanceType = list[float]


def _sort_episodes(episodes: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Sorts the episodes in descending order, using
    the inverse of the distance.

    Args:
        episodes (list[tuple[str, float]]): _description_

    Returns:
        list[str]: _description_
    """
    return sorted(episodes, key=lambda x: x[1], reverse=True)


def raw_distance(
    metadatas: MetadataType, distances: DistanceType
) -> dict[str, MetadataType | DistanceType]:
    """Returns the values as they come.

    Args:
        metadatas (MetadataType): _description_
        distances (DistanceType): _description_

    Returns:
        dict[str, MetadataType | DistanceType]: _description_
    """
    metric = []
    for i, mt in enumerate(metadatas):
        metric.append((mt["title"], 1 / distances[i]))

    # This comes sorted already
    return metric


def minimum_distance(
    metadatas: MetadataType, distances: DistanceType
) -> dict[str, MetadataType | DistanceType]:
    """_summary_

    Args:
        metadatas (MetadataType): metadata from querying chroma.
        distances (DistanceType): distances from each query.

    Returns:
        dict[str, MetadataType | DistanceType]: _description_
    """
    titles = set()
    metric = defaultdict(float)
    for i, mt in enumerate(metadatas):
        if mt["title"] not in titles:
            titles.add(mt["title"])
            metric[mt["title"]] += 1 / distances[i]

    # Don't need to sort these
    return list(metric.items())


def sum_weighted_distance(
    metadatas: MetadataType, distances: DistanceType
) -> list[tuple[str, float]]:
    """Groups the functions giving more weight depending
    the more times a title appears, with the weight being
    the inverse of the distance (so adding more occurrences actually
    weights more to the final decision).

    Args:
        metadatas (MetadataType): metadata from querying chroma.
        distances (DistanceType): distances from each query.

    Returns:
        list[tuple[str, float]]: _description_
    """
    metric = defaultdict(float)
    for i, mt in enumerate(metadatas):
        metric[mt["title"]] += 1 / distances[i]

    return _sort_episodes(list(metric.items()))


def _match_aggregating_function(aggregating_function: str) -> Callable:
    """Get the function to group the chroma results.

    Args:
        aggregating_function (str):
            The name of a function to group the results from querying chroma.

    Returns:
        Callable: Function to be passed to query_db.
    """
    match aggregating_function:
        case "minimum":
            return minimum_distance
        case "sum_weighted":
            return sum_weighted_distance
        case "raw":
            return raw_distance
        case _:
            return NotImplementedError(
                f"Unknown aggregating_function: {aggregating_function}"
            )


def query_db(
    chroma: chroma.Chroma,
    query: str,
    n_results: int = 20,
    aggregating_function: Callable | None = sum_weighted_distance,
) -> list[str]:
    """Queries the chroma database to extract the most similar podcasts.

    Example result from querying chroma:

        ```python
        >>> result = client.query(query_texts=["tell me about pandas"])

        metadatas = [[{'title': '402-polars.txt', 'line': '209'},
        {'title': '012.txt', 'line': '198'},
        {'title': '012.txt', 'line': '197'},
        {'title': '410-intersection-of-tabular-data-and-general-ai.txt',
            'line': '113'},
        {'title': '341-25-pandas-functions.txt', 'line': '55'},
        {'title': '414-startup-row.txt', 'line': '70'},
        {'title': '402-polars.txt', 'line': '210'},
        {'title': '338-cibuildwheel-scikit-hep.txt', 'line': '240'},
        {'title': '402-polars.txt', 'line': '218'},
        {'title': '402-polars.txt', 'line': '221'}]]
        distances = [[0.3780611753463745,
        0.40761131048202515,
        0.42179733514785767,
        0.4531496465206146,
        0.4663689434528351,
        0.47604429721832275,
        0.485452264547348,
        0.5037158727645874,
        0.5327827334403992,
        0.5412342548370361]]
        ```

    Args:
        chroma (chroma.Chroma): chroma db client instance.
        query (str): str to query against chro,a
        n_results (int, optional):
            Number of results to obtain. Defaults to 10.
        aggregating_function (Callable, optional):
            Used to sort the response and present unique titles.

    Returns:
        titles (list[str]): List of titles to show.
    """
    # Even though it may return 10 results (given the passages),
    # we must apply some function to keep only as much results as
    # podcasts.
    result = chroma.query(query_texts=[query], n_results=n_results)
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    if aggregating_function is None:
        return [m["title"] for m in metadatas]

    agg = aggregating_function(metadatas, distances)
    # print("AGG", agg)  # just while debugging
    return [m[0] for m in agg]
