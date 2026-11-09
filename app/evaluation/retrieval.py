"""Same query, corpus, filters, and K cutoffs for all retrieval methods."""

from dataclasses import asdict

from .metrics import retrieval_metrics


def compare_retrieval(question, store, embedder, ks, candidate_limit, clock):
    rows = {}
    embedding_time = None
    vector = None
    embedding_error = None
    start = clock()
    try:
        vector = embedder.embed_query(question.question)
    except Exception as error:
        embedding_error = type(error).__name__  # never serialize provider messages or key
    embedding_time = clock() - start
    for method in ("semantic", "bm25", "hybrid"):
        start = clock()
        error = None
        hits = []
        if method != "bm25" and embedding_error:
            error = embedding_error
        else:
            try:
                if method == "semantic":
                    hits = store.search(vector, max(ks), filters=question.filters)
                elif method == "bm25":
                    hits = store.keyword_search(question.question, limit=max(ks), filters=question.filters)
                else:
                    hits = store.hybrid_search(vector, question.question, limit=max(ks),
                        candidate_limit=candidate_limit, filters=question.filters)
            except Exception as failure:
                error = type(failure).__name__
        database_time = clock() - start
        ids = [hit.chunk.chunk_id for hit in hits]
        rows[method] = {"error": error, "ranked_ids": ids,
            "duplicate_ids": len(ids) != len(set(ids)),
            "hits": [asdict(hit) for hit in hits], "database_s": database_time if not embedding_error or method == "bm25" else None,
            "inclusive_s": database_time + (embedding_time if method != "bm25" else 0),
            "metrics": {str(k): retrieval_metrics(ids, question.relevant_chunk_ids, k) for k in ks}}
    return {"query_embedding_s": embedding_time, "methods": rows}
