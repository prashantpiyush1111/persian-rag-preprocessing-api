# Retrieval evaluation

Evaluation uses **gold FAQ IDs** and **ranked retrieved IDs**, not answer text.

```bash
curl -X POST http://127.0.0.1:8000/v1/evaluate/retrieval \
  -H "Content-Type: application/json" \
  --data-binary @examples/retrieval-request.json
```

The synthetic example has one relevant FAQ at rank 2. At K=2:

| Metric | Value | Meaning |
|---|---|---|
| Recall@2 | 1.0 | The relevant FAQ was retrieved |
| MRR@2 | 0.5 | First relevant result has rank 2 |
| HitRate@2 | 1.0 | At least one relevant result was found |

For batch evaluation, send `{"cases":[...],"k_values":[1,3,5]}` to
`/v1/evaluate/retrieval/batch`. Each case follows `examples/retrieval-request.json`.
Aggregate metrics are macro averages across cases. Ranks must be unique positive integers.
Each retrieved item can contain several `faq_ids` to represent a grouped result.

For uploads, choose `examples/retrieval-cases.json` at `/v1/evaluate/retrieval/file`
and set form field `k_values` to `[1,2]` (comma-separated `1,2` is also supported).

## Grouped RAG adapter

Upload `examples/grouped-rag-output.json` to `/v1/evaluate/retrieval/rag-output/file`.
Each record provides:

- `metrics.target_faq_id`: relevant FAQ identifier.
- `initial_sources`: ranked entries with `rank` and `metadata.faq_id`.
- `final_groups`: entries with `group_rank` and `faq_ids`.
- Optional `question`, `reference_answer` and `generated_answer` for report context.

The adapter reports initial and grouped metrics separately. In the synthetic example,
the relevant FAQ moves from initial rank 2 to group rank 1. That is a fixture for testing
the evaluator, not measured evidence that grouping improves retrieval on real data.
