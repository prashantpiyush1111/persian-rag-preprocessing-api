from functools import lru_cache
import os
from typing import Callable

from app.config import settings

EmbedFunction = Callable[[list[str]], list[list[float]]]


@lru_cache(maxsize=4)
def _local_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings are not installed. Install sentence-transformers."
        ) from exc
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _openai_client(api_key: str, base_url: str | None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI provider is not installed.") from exc
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_embedding_function(provider: str = "local", model_name: str | None = None) -> EmbedFunction:
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI embedding provider.")
        model = model_name or "text-embedding-3-small"
        client = _openai_client(settings.openai_api_key, settings.openai_base_url)

        def embed(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]

        return embed

    model = _local_model(model_name or settings.embedding_model)

    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    return embed
