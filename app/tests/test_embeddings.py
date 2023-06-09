import pytest
import app.embeddings as emb
import numpy.testing as npt
from talking_python import chroma


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


@pytest.fixture(scope="module")
def client_query_result_2():
    yield {
        "metadatas": [
            [
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
            ]
        ],
        "distances": [
            [
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
            ]
        ],
    }


def test_raw_distance(client_query_result):
    grouped = emb.raw_distance(
        client_query_result["metadatas"], client_query_result["distances"]
    )
    assert isinstance(grouped, list)
    assert len(grouped) == 10
    titles = [
        "402-polars.txt",
        "012.txt",
        "012.txt",
        "410-intersection-of-tabular-data-and-general-ai.txt",
        "341-25-pandas-functions.txt",
        "414-startup-row.txt",
        "402-polars.txt",
        "338-cibuildwheel-scikit-hep.txt",
        "402-polars.txt",
        "402-polars.txt",
    ]
    assert all([a[0] == b for a, b in zip(grouped, titles)])
    npt.assert_allclose(
        [a[1] for a in grouped],
        [
            2.645075,
            2.453318,
            2.370807,
            2.206777,
            2.144225,
            2.100645,
            2.059935,
            1.985246,
            1.876938,
            1.847629,
        ],
        rtol=1e-5,
    )


def test_minimum_distance(client_query_result):
    grouped = emb.minimum_distance(
        client_query_result["metadatas"], client_query_result["distances"]
    )
    assert isinstance(grouped, list)
    assert len(grouped) == 6
    titles = [
        "402-polars.txt",
        "012.txt",
        "410-intersection-of-tabular-data-and-general-ai.txt",
        "341-25-pandas-functions.txt",
        "414-startup-row.txt",
        "338-cibuildwheel-scikit-hep.txt",
    ]
    assert all([a[0] == b for a, b in zip(grouped, titles)])
    npt.assert_allclose(
        [a[1] for a in grouped],
        [2.645075, 2.453318, 2.206777, 2.144225, 2.100645, 1.985246],
        rtol=1e-5,
    )


def test_average_weighted_distance(client_query_result):
    grouped = emb.average_weighted_distance(
        client_query_result["metadatas"], client_query_result["distances"]
    )
    assert isinstance(grouped, list)
    assert len(grouped) == 6
    titles = [
        "402-polars.txt",
        "012.txt",
        "410-intersection-of-tabular-data-and-general-ai.txt",
        "341-25-pandas-functions.txt",
        "414-startup-row.txt",
        "338-cibuildwheel-scikit-hep.txt",
    ]
    assert all([a[0] == b for a, b in zip(grouped, titles)])
    npt.assert_allclose(
        [a[1] for a in grouped],
        [8.429576, 4.824124, 2.206777, 2.144225, 2.100645, 1.985246],
        rtol=1e-5,
    )


class MockResponse:
    # mock json() method always returns a specific testing dictionary
    @staticmethod
    def query():
        return {"mock_key": "mock_response"}


@pytest.mark.parametrize(
    "agg_function, expected",
    [
        (
            emb.minimum_distance,
            [
                "402-polars.txt",
                "012.txt",
                "410-intersection-of-tabular-data-and-general-ai.txt",
                "341-25-pandas-functions.txt",
                "414-startup-row.txt",
                "338-cibuildwheel-scikit-hep.txt",
            ],
        ),
        (
            emb.raw_distance,
            [
                "402-polars.txt",
                "012.txt",
                "012.txt",
                "410-intersection-of-tabular-data-and-general-ai.txt",
                "341-25-pandas-functions.txt",
                "414-startup-row.txt",
                "402-polars.txt",
                "338-cibuildwheel-scikit-hep.txt",
                "402-polars.txt",
                "402-polars.txt",
            ],
        ),
        (
            emb.average_weighted_distance,
            [
                "402-polars.txt",
                "012.txt",
                "410-intersection-of-tabular-data-and-general-ai.txt",
                "341-25-pandas-functions.txt",
                "414-startup-row.txt",
                "338-cibuildwheel-scikit-hep.txt",
            ],
        ),
    ],
)
def test_query_db(mocker, client_query_result_2, agg_function, expected):
    mocker.patch(
        "talking_python.chroma.Chroma.query", return_value=client_query_result_2
    )
    # mocking the result to just test the result of the aggregating functions
    result = emb.query_db(
        chroma.Chroma,
        "show me podcasts about numpy",
        n_results=10,
        aggregating_function=agg_function,
    )
    print("RESULT", result)
    assert result == expected
