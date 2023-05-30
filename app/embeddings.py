""" """

from talking_python import chroma
from functools import lru_cache
import streamlit as st
from typing import Callable
from collections import defaultdict


@lru_cache
def get_embedding_function(api_key: str) -> chroma.EmbeddingFunction:
    """Get the embedding function using the api key supplied
    by the user.

    Args:
        api_key (str):
            Hugging Face API token.

    Returns:
        chroma.EmbeddingFunction:
            Function to generate the embeddings, will be given to the
            chroma client.
    """
    # Function cached to avoid loading again when rerunning the script,
    # is necessary?
    # TOOD: Add retry:
    # https://github.com/mmz-001/knowledge_gpt/blob/main/knowledge_gpt/embeddings.py
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




# sample data, move this to tests
"""
result = client.query(query_texts=["tell me about pandas"])

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
"""


# Functions to aggregate the queried data.
# Given a sample query, it is possible that the first
# response pertains to a
def minimum_distance(metadatas, distances):
    # Removes repeated titles, keeps the first occurence only
    # The distances are sorted from smaller to bigger, so the first 
    titles = set()
    result = defaultdict(list)
    for i, mt in enumerate(metadatas):
        if mt["title"] not in titles:
            titles.add(mt["title"])
            result["metadatas"].append(mt)
            result["distances"].append(distances[i])

    return dict(result)


def average():
    # Sort the results but using the average of distances.
    pass


@st.cache(allow_output_mutation=True)
def query_db(
    chroma: chroma.Chroma,
    query: str,
    n_results: int = 10,
    grouping_function: Callable = minimum_distance,
):
    """_summary_

    Args:
        chroma (chroma.Chroma): _description_
        query (str): _description_
        n_results (int, optional): _description_. Defaults to 10.
        grouping_function (Callable, optional):
            Used to sort the response and present unique titles.
    """
    # TODO: Even though it may return 10 results (given the passages),
    #  we must apply some function to keep only as much results as
    # podcasts.
    result = chroma.query(query_texts=[query], n_results=n_results)
    # Aggregates the response
    
    return grouping_function(result["metadatas"][0], result["distances"][0])
