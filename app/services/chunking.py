from copy import deepcopy
import re
import uuid
from typing import Any

import numpy as np

from app.schemas import ChunkingOptions
from app.services.embeddings import get_embedding_function


def _validate(records: list[dict[str, Any]]) -> None:
    for index, item in enumerate(records):
        if not isinstance(item, dict) or not str(item.get("question_normalized", "")).strip() or not str(item.get("answer_normalized", "")).strip():
            raise ValueError(f"Record {index} requires question_normalized and answer_normalized.")


def _faq(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in records:
        question = str(item["question_normalized"]).strip()
        answer = str(item["answer_normalized"]).strip()
        metadata = deepcopy(item.get("metadata") or {})
        chunk_id = str(item.get("chunk_id") or metadata.get("faq_id") or uuid.uuid4())
        metadata.update({
            "faq_id": metadata.get("faq_id") or chunk_id,
            "chunking_method": "whole_qa_pair", "chunk_level": 0,
            "language": metadata.get("language", "fa"), "embedding_text_version": "normalized",
            "question_length_chars": len(question), "answer_length_chars": len(answer),
        })
        metadata["processing_steps"] = metadata.get("processing_steps", []) + ["faq_chunking"]
        text = f"سوال: {question}\nپاسخ: {answer}"
        metadata["qa_text_length_chars"] = len(text)
        result.append({"chunk_id": chunk_id, "question_normalized": question, "answer_normalized": answer, "embedding_text": text, "metadata": metadata})
    return result


def _split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("؟ ", start, end), text.rfind("! ", start, end), text.rfind(" ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        while next_start > start and next_start < len(text) and not text[next_start - 1].isspace():
            next_start -= 1
        start = next_start if next_start > start else end
    return chunks


def _question_answer(records: list[dict[str, Any]], options: ChunkingOptions) -> dict[str, list[dict[str, Any]]]:
    questions: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        base = deepcopy(item.get("metadata") or {})
        parent_id = str(item.get("parent_id") or item.get("chunk_id") or base.get("faq_id") or f"auto_faq_{index:06d}")
        question = re.sub(r"\s+", " ", str(item["question_normalized"])).strip()
        answer = re.sub(r"\s+", " ", str(item["answer_normalized"])).strip()
        questions.append({
            "record_id": str(uuid.uuid4()), "parent_id": parent_id,
            "question_normalized": question, "embedding_text": f"سؤال: {question}",
            "metadata": {**base, "parent_id": parent_id, "unit_type": "question_canonical", "chunking_strategy": "question_only", "chunk_level": 0, "is_query_candidate": True, "is_split": False},
        })
        chunks = _split_with_overlap(answer, options.max_chars, options.overlap_chars) or [""]
        for chunk_index, text in enumerate(chunks):
            answers.append({
                "chunk_id": str(uuid.uuid4()), "parent_id": parent_id, "chunk_index": chunk_index,
                "answer_normalized": text, "embedding_text": f"پاسخ: {text}",
                "metadata": {**base, "parent_id": parent_id, "unit_type": "answer_chunk", "chunking_strategy": "character_window", "chunk_level": 1, "chunk_index": chunk_index, "num_chunks_from_parent": len(chunks), "is_query_candidate": False, "is_empty": not bool(text), "max_chars": options.max_chars, "overlap_chars": options.overlap_chars},
            })
    return {"questions": questions, "answer_chunks": answers}


def _segments(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!؟?۔\n])\s+", text.strip()) if part.strip()]


def _cosine(first: list[float], second: list[float]) -> float:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _semantic(records: list[dict[str, Any]], options: ChunkingOptions) -> list[dict[str, Any]]:
    embed = get_embedding_function(options.embedding_provider.value, options.embedding_model)
    output: list[dict[str, Any]] = []
    for item in records:
        question, answer = str(item["question_normalized"]).strip(), str(item["answer_normalized"]).strip()
        candidates = _segments(answer)
        if len(answer) < options.answer_length_threshold or len(candidates) <= 1:
            merged = [answer]
            reason = "answer_below_threshold" if len(answer) < options.answer_length_threshold else "no_boundary_found"
        else:
            vectors = embed(candidates)
            merged, current = [], [candidates[0]]
            current_len = len(candidates[0])
            for index in range(1, len(candidates)):
                if _cosine(vectors[index - 1], vectors[index]) < options.similarity_threshold and current_len >= options.min_chunk_chars:
                    merged.append(" ".join(current))
                    current, current_len = [candidates[index]], len(candidates[index])
                else:
                    current.append(candidates[index])
                    current_len += len(candidates[index]) + 1
            merged.append(" ".join(current))
            reason = "semantic_topic_shift" if len(merged) > 1 else "topic_shift_not_detected"
        parent_metadata = deepcopy(item.get("metadata") or {})
        parent_id = str(parent_metadata.get("faq_id") or item.get("chunk_id") or uuid.uuid4())
        cursor = 0
        for chunk_index, text in enumerate(merged):
            start = answer.find(text, cursor)
            start = cursor if start < 0 else start
            end = start + len(text)
            cursor = end
            embedding_text = f"سوال: {question}\nپاسخ: {text}"
            metadata = {**parent_metadata, "faq_id": parent_id, "parent_id": parent_id, "chunking_method": "semantic_qa_answer_split", "chunk_level": 1, "chunk_index": chunk_index, "num_chunks_from_parent": len(merged), "is_split": len(merged) > 1, "split_reason": reason, "answer_chunk_start_char": start, "answer_chunk_end_char": end, "question_length_chars": len(question), "answer_length_chars": len(text), "qa_text_length_chars": len(embedding_text)}
            metadata["processing_steps"] = metadata.get("processing_steps", []) + ["semantic_chunking_normalized_only"]
            output.append({"chunk_id": str(uuid.uuid4()), "question_normalized": question, "answer_normalized": text, "embedding_text": embedding_text, "metadata": metadata})
    return output


def chunk_records(records: list[dict[str, Any]], options: ChunkingOptions):
    _validate(records)
    if options.method.value == "question_answer":
        return _question_answer(records, options)
    if options.method.value == "semantic":
        return _semantic(records, options)
    return _faq(records)


def output_count(data: Any) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and "questions" in data and "answer_chunks" in data:
        return len(data["questions"]) + len(data["answer_chunks"])
    return 0
