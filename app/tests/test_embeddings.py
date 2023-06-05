import pytest
import app.embeddings as emb
import numpy.testing as npt
import numpy as np


@pytest.fixture(scope="module")
def client_query_result():
    yield {
        "metadatas": [
                {"title": "402-polars.txt", "line": "209"},
                {"title": "012.txt", "line": "198"},
                {"title": "012.txt", "line": "197"},
                {
                    "title": "410-intersection-of-tabular-data-and-general-ai.txt",
                    "line": "113",
                },
                {"title": "341-25-pandas-functions.txt", "line": "55"},
                {"title": "414-startup-row.txt", "line": "70"},
                {"title": "402-polars.txt", "line": "210"},
                {"title": "338-cibuildwheel-scikit-hep.txt", "line": "240"},
                {"title": "402-polars.txt", "line": "218"},
                {"title": "402-polars.txt", "line": "221"},
        ],
        "distances": [
                0.3780611753463745,
                0.40761131048202515,
                0.42179733514785767,
                0.4531496465206146,
                0.4663689434528351,
                0.47604429721832275,
                0.485452264547348,
                0.5037158727645874,
                0.5327827334403992,
                0.5412342548370361,
        ],
    }


def test_raw_distance(client_query_result):
    grouped = emb.raw_distance(
        client_query_result["metadatas"], client_query_result["distances"]
    )
    assert isinstance(grouped, dict)
    assert len(grouped.keys()) == 2
    assert all([k in grouped.keys() for k in ["metadatas", "distances"]])
    assert len(grouped["metadatas"]) == len(grouped["distances"]) == 10
    npt.assert_allclose(grouped["distances"], client_query_result["distances"])


def test_minimum_distance(client_query_result):
    grouped = emb.minimum_distance(
        client_query_result["metadatas"], client_query_result["distances"]
    )
    assert isinstance(grouped, dict)
    assert len(grouped.keys()) == 2
    assert all([k in grouped.keys() for k in ["metadatas", "distances"]])
    assert len(grouped["metadatas"]) == len(grouped["distances"]) == 6
    npt.assert_allclose(
        grouped["distances"],
        [
            0.3780611753463745,
            0.40761131048202515,
            0.4531496465206146,
            0.4663689434528351,
            0.47604429721832275,
            0.5037158727645874,
        ],
    )



# def test_average_weighted_distance():
#     grouped = emb.minimum_distance(
#         client_query_result["metadatas"], client_query_result["distances"]
#     )
#     assert isinstance(grouped, dict)
#     assert len(grouped.keys()) == 2
#     assert all([k in grouped.keys() for k in ["metadatas", "distances"]])
