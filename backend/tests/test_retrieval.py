from unittest.mock import MagicMock, patch
from app.retrieval import retrieve, RetrievedChunk


def _mock_result(text: str, page: int, score: float) -> MagicMock:
    r = MagicMock()
    r.payload = {"text": text, "page": page, "chunk_type": "text", "source_file": "handbook.pdf"}
    r.score = score
    return r


def test_retrieve_maps_qdrant_results_to_dataclass():
    with patch("app.retrieval.get_client") as mock_gc:
        mock_client = MagicMock()
        mock_gc.return_value = mock_client
        mock_client.search.return_value = [_mock_result("vacation policy", 9, 0.85)]

        results = retrieve([0.1] * 768)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].source_file == "handbook.pdf"
    assert results[0].page == 9
    assert results[0].score == 0.85


def test_retrieve_returns_empty_list_on_no_matches():
    with patch("app.retrieval.get_client") as mock_gc:
        mock_client = MagicMock()
        mock_gc.return_value = mock_client
        mock_client.search.return_value = []

        results = retrieve([0.1] * 768)

    assert results == []


def test_retrieve_passes_threshold_to_qdrant():
    with patch("app.retrieval.get_client") as mock_gc:
        mock_client = MagicMock()
        mock_gc.return_value = mock_client
        mock_client.search.return_value = []

        retrieve([0.1] * 768)

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["score_threshold"] == 0.45
