from app.schemas import RetrievalBatchEvaluationRequest, RetrievalEvaluationRequest


def evaluate_retrieval(request: RetrievalEvaluationRequest) -> dict:
    """Calculate set-based Recall@K and reciprocal rank of the first hit."""

    ranked_items = sorted(request.retrieved_items, key=lambda item: item.rank)
    default_k = max((item.rank for item in ranked_items), default=0)
    k = request.k if request.k is not None else max(default_k, 1)

    gold_ids = set(request.gold_faq_ids)
    top_k_items = [item for item in ranked_items if item.rank <= k]

    retrieved_ids: set[str] = set()
    first_relevant_rank = None

    for item in top_k_items:
        item_ids = set(item.faq_ids)
        retrieved_ids.update(item_ids)
        if first_relevant_rank is None and item_ids.intersection(gold_ids):
            first_relevant_rank = item.rank

    matched_ids = gold_ids.intersection(retrieved_ids)
    missed_ids = gold_ids.difference(retrieved_ids)

    recall = len(matched_ids) / len(gold_ids)
    mrr = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0

    evaluated_items = []
    for item in ranked_items:
        item_ids = set(item.faq_ids)
        matched_item_ids = sorted(item_ids.intersection(gold_ids))
        evaluated_items.append({
            **item.model_dump(mode="json"),
            "is_relevant": bool(matched_item_ids),
            "matched_faq_ids": matched_item_ids,
            "included_at_k": item.rank <= k,
        })

    return {
        "sample_id": request.sample_id,
        "query": request.query,
        "reference_answer": request.reference_answer,
        "llm_answer": request.llm_answer,
        "metadata": request.metadata,
        "k": k,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "hit_rate_at_k": float(bool(matched_ids)),
        "relevant_retrieved_count": len(matched_ids),
        "total_relevant_count": len(gold_ids),
        "first_relevant_rank": first_relevant_rank,
        "matched_faq_ids": sorted(matched_ids),
        "missed_faq_ids": sorted(missed_ids),
        "ranked_items": evaluated_items,
    }


def evaluate_retrieval_batch(request: RetrievalBatchEvaluationRequest) -> dict:
    """Evaluate multiple queries and macro-average Recall, MRR and hit rate."""

    case_results = []
    totals = {
        k: {"recall_at_k": 0.0, "mrr_at_k": 0.0, "hit_rate_at_k": 0.0}
        for k in request.k_values
    }

    for case in request.cases:
        metrics = []
        ranked_items = None
        for k in request.k_values:
            evaluation = evaluate_retrieval(case.model_copy(update={"k": k}))
            metric = {
                "k": k,
                "recall_at_k": evaluation["recall_at_k"],
                "mrr_at_k": evaluation["mrr_at_k"],
                "hit_rate_at_k": evaluation["hit_rate_at_k"],
            }
            metrics.append(metric)
            for name in totals[k]:
                totals[k][name] += metric[name]
            if ranked_items is None:
                ranked_items = evaluation["ranked_items"]

        case_results.append({
            "sample_id": case.sample_id,
            "query": case.query,
            "reference_answer": case.reference_answer,
            "llm_answer": case.llm_answer,
            "metadata": case.metadata,
            "gold_faq_ids": case.gold_faq_ids,
            "ranked_items": ranked_items or [],
            "metrics": metrics,
        })

    query_count = len(request.cases)
    aggregate_metrics = [
        {
            "k": k,
            **{name: value / query_count for name, value in totals[k].items()},
        }
        for k in request.k_values
    ]
    return {
        "ok": True,
        "query_count": query_count,
        "k_values": request.k_values,
        "aggregate_metrics": aggregate_metrics,
        "cases": case_results,
    }


def _clean_ids(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _gold_ids_from_rag_record(record: dict) -> list[str]:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    for candidate in (
        record.get("gold_faq_ids"),
        record.get("target_faq_ids"),
        record.get("target_faq_id"),
        record.get("target_chunk_id"),
        metrics.get("target_faq_ids"),
        metrics.get("target_faq_id"),
        metrics.get("target_chunk_id"),
    ):
        cleaned = _clean_ids(candidate)
        if cleaned:
            return cleaned
    return []


def _initial_items(record: dict) -> list[dict]:
    items = []
    for index, source in enumerate(record.get("initial_sources") or [], start=1):
        if not isinstance(source, dict):
            continue
        source_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        faq_ids = _clean_ids(
            source.get("faq_ids")
            or source.get("faq_id")
            or source_metadata.get("faq_id")
            or source.get("node_id")
        )
        if not faq_ids:
            continue
        extra_metadata = {
            **source_metadata,
            **{key: value for key, value in source.items() if key not in {"rank", "faq_ids", "faq_id", "score", "metadata"}},
        }
        items.append({
            "rank": source.get("rank", index),
            "faq_ids": faq_ids,
            "score": source.get("score"),
            "metadata": extra_metadata,
        })
    return items


def _grouped_items(record: dict) -> list[dict]:
    items = []
    for index, group in enumerate(record.get("final_groups") or [], start=1):
        if not isinstance(group, dict):
            continue
        faq_ids = _clean_ids(group.get("faq_ids") or group.get("faq_id") or group.get("group_id"))
        if not faq_ids:
            continue
        items.append({
            "rank": group.get("group_rank", group.get("rank", index)),
            "faq_ids": faq_ids,
            "score": group.get("best_score", group.get("score")),
            "metadata": {
                key: value for key, value in group.items()
                if key not in {"group_rank", "rank", "faq_ids", "faq_id", "best_score", "score"}
            },
        })
    return items


def evaluate_rag_output_records(records: list[dict], k_values: list[int]) -> dict:
    """Evaluate initial and grouped retrieval from the documented JSON schema."""

    if not records:
        raise ValueError("The RAG output file must contain at least one record.")

    normalized_k_values = sorted(set(k_values))
    if not normalized_k_values or any(k < 1 for k in normalized_k_values):
        raise ValueError("k_values must contain positive integers.")

    initial_totals = {k: {"recall_at_k": 0.0, "mrr_at_k": 0.0, "hit_rate_at_k": 0.0} for k in normalized_k_values}
    grouped_totals = {k: {"recall_at_k": 0.0, "mrr_at_k": 0.0, "hit_rate_at_k": 0.0} for k in normalized_k_values}
    cases = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object.")
        gold_ids = _gold_ids_from_rag_record(record)
        if not gold_ids:
            raise ValueError(
                f"Record {index} has no gold ID. Expected metrics.target_faq_id, "
                "target_faq_id, target_chunk_id, or gold_faq_ids."
            )

        initial_items = _initial_items(record)
        grouped_items = _grouped_items(record)
        common = {
            "sample_id": record.get("sample_id"),
            "query": record.get("query") or record.get("question"),
            "reference_answer": record.get("reference_answer") or record.get("canonical_answer"),
            "llm_answer": record.get("llm_answer") or record.get("generated_answer") or record.get("answer"),
            "metadata": {
                "original_metrics": record.get("metrics") or {},
                "retrieval_config": record.get("retrieval_config") or {},
                "timestamp": record.get("timestamp"),
            },
            "gold_faq_ids": gold_ids,
        }

        initial_metrics = []
        grouped_metrics = []
        initial_ranked = []
        grouped_ranked = []
        for k in normalized_k_values:
            initial_evaluation = evaluate_retrieval(RetrievalEvaluationRequest(
                **common, retrieved_items=initial_items, k=k
            ))
            grouped_evaluation = evaluate_retrieval(RetrievalEvaluationRequest(
                **common, retrieved_items=grouped_items, k=k
            ))
            initial_metric = {name: initial_evaluation[name] for name in ("recall_at_k", "mrr_at_k", "hit_rate_at_k")}
            grouped_metric = {name: grouped_evaluation[name] for name in ("recall_at_k", "mrr_at_k", "hit_rate_at_k")}
            initial_metrics.append({"k": k, **initial_metric})
            grouped_metrics.append({"k": k, **grouped_metric})
            for name, value in initial_metric.items():
                initial_totals[k][name] += value
            for name, value in grouped_metric.items():
                grouped_totals[k][name] += value
            if not initial_ranked:
                initial_ranked = initial_evaluation["ranked_items"]
            if not grouped_ranked:
                grouped_ranked = grouped_evaluation["ranked_items"]

        cases.append({
            **common,
            "initial_metrics": initial_metrics,
            "grouped_metrics": grouped_metrics,
            "initial_sources": initial_ranked,
            "final_groups": grouped_ranked,
        })

    query_count = len(cases)
    aggregate = lambda totals: [
        {"k": k, **{name: value / query_count for name, value in totals[k].items()}}
        for k in normalized_k_values
    ]
    return {
        "ok": True,
        "source_format": "api_llama_grouped_output",
        "query_count": query_count,
        "k_values": normalized_k_values,
        "initial_aggregate_metrics": aggregate(initial_totals),
        "grouped_aggregate_metrics": aggregate(grouped_totals),
        "cases": cases,
    }
