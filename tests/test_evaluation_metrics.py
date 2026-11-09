import pytest

from app.evaluation.metrics import latency_summary, retrieval_metrics, validate_ks


def test_hand_calculated_multiple_relevant_chunks():
    assert retrieval_metrics(["x", "a", "b"], ["a", "b", "c"], 3) == {"recall": 2/3, "rr": .5}
    assert retrieval_metrics(["x", "a", "b"], ["a", "b", "c"], 1) == {"recall": 0, "rr": 0}
    assert retrieval_metrics([], ["a"], 10) == {"recall": 0, "rr": 0}
    assert retrieval_metrics(["a", "a", "b"], ["a", "b"], 2) == {"recall": .5, "rr": 1}
    assert retrieval_metrics(["x"], [], 3) == {"recall": None, "rr": None}


@pytest.mark.parametrize("ks", [[], [0], [-1], [True], [1.5]])
def test_invalid_k(ks):
    with pytest.raises(ValueError):
        validate_ks(ks)


def test_latency_count_and_percentile():
    assert latency_summary([])["mean_s"] is None
    assert latency_summary([1., 3.])["median_s"] == 2
    assert latency_summary([1.] * 19)["p95_s"] is None
    assert latency_summary(list(range(1, 21)))["p95_s"] == 19
    assert validate_ks([10, 1, 3, 1]) == [1, 3, 10]
    with pytest.raises(ValueError):
        latency_summary([float("nan")])
