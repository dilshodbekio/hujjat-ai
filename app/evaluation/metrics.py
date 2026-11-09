"""Binary relevance metrics and explicitly denominated summaries."""

import math
from statistics import mean, median


def validate_ks(ks):
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("K values must be positive integers")
    return sorted(set(ks))


def retrieval_metrics(ranked_ids: list[str], relevant_ids: list[str], k: int) -> dict:
    validate_ks([k])
    relevant = set(relevant_ids)
    if not relevant:
        return {"recall": None, "rr": None}
    top = ranked_ids[:k]  # duplicates consume rank positions; never inflate recall
    return {"recall": len(set(top) & relevant) / len(relevant),
            "rr": next((1 / rank for rank, item in enumerate(top, 1) if item in relevant), 0.)}


def latency_summary(values: list[float]) -> dict:
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latencies must be nonnegative finite seconds")
    result = {"count": len(values), "mean_s": mean(values) if values else None,
              "median_s": median(values) if values else None, "p95_s": None}
    if len(values) >= 20:
        result["p95_s"] = sorted(values)[math.ceil(.95 * len(values)) - 1]
    return result
