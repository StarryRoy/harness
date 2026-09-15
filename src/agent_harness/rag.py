"""Provider-neutral retrieval fusion and reranking."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class RAGResult(TypedDict):
    """Stable document shape returned by :class:`RAGRetriever`."""

    content: str
    score: float | None
    source: str | None
    metadata: dict[str, Any]


class _RAGInput(BaseModel):
    query: str = Field(description="Query to retrieve reference documents for.")


class RAGRetriever(BaseTool):
    """Fuse user-owned vector and BM25 retrievers, then rerank their results.

    Retrievers must expose ``invoke(query)`` or ``retrieve(query)`` (and may expose
    ``ainvoke`` or ``aretrieve``).
    Rerankers may expose ``rerank(query, documents)``, ``rank(query, documents)``,
    or the Runnable-style ``invoke({"query": ..., "documents": ...})``. Async
    counterparts are preferred by :meth:`ainvoke`; sync-only components are run in
    worker threads.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "rag_retriever"
    description: str = (
        "Retrieve reference documents by fusing vector and BM25 recall, removing "
        "duplicates, and reranking the candidates."
    )
    args_schema: type[BaseModel] = _RAGInput

    vector_retriever: Any = Field(exclude=True, repr=False)
    bm25_retriever: Any = Field(exclude=True, repr=False)
    reranker: Any = Field(exclude=True, repr=False)

    def __init__(
        self, vector_retriever: Any, bm25_retriever: Any, reranker: Any
    ) -> None:
        if vector_retriever is None or bm25_retriever is None or reranker is None:
            raise ValueError(
                "vector_retriever, bm25_retriever, and reranker are required"
            )
        super().__init__(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
        )

    def _run(self, query: str, run_manager: Any = None) -> list[RAGResult]:
        self._validate_query(query)
        vector_results = self._retrieve(self.vector_retriever, query, "vector")
        bm25_results = self._retrieve(self.bm25_retriever, query, "bm25")
        candidates = self._deduplicate([*vector_results, *bm25_results])
        return self._rerank(query, candidates)

    async def _arun(self, query: str, run_manager: Any = None) -> list[RAGResult]:
        self._validate_query(query)
        vector_results, bm25_results = await asyncio.gather(
            self._aretrieve(self.vector_retriever, query, "vector"),
            self._aretrieve(self.bm25_retriever, query, "bm25"),
        )
        candidates = self._deduplicate([*vector_results, *bm25_results])
        return await self._arerank(query, candidates)

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

    @classmethod
    def _retrieve(cls, retriever: Any, query: str, channel: str) -> list[RAGResult]:
        method = None
        for name in ("invoke", "retrieve", "get_relevant_documents"):
            candidate = getattr(retriever, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None and callable(retriever):
            method = retriever
        if method is None:
            raise TypeError(
                f"{channel}_retriever must define invoke(query) or retrieve(query)"
            )
        raw = method(query)
        if inspect.isawaitable(raw):
            close = getattr(raw, "close", None)
            if callable(close):
                close()
            raise TypeError(
                f"{channel}_retriever is async-only; use RAGRetriever.ainvoke()"
            )
        return [cls._normalize(item, channel) for item in cls._items(raw)]

    @classmethod
    async def _aretrieve(
        cls, retriever: Any, query: str, channel: str
    ) -> list[RAGResult]:
        for name in ("ainvoke", "aretrieve", "aget_relevant_documents"):
            method = getattr(retriever, name, None)
            if callable(method):
                raw = method(query)
                raw = await raw if inspect.isawaitable(raw) else raw
                return [cls._normalize(item, channel) for item in cls._items(raw)]
        if cls._is_async_callable(retriever):
            raw = retriever(query)
            raw = await raw if inspect.isawaitable(raw) else raw
            return [cls._normalize(item, channel) for item in cls._items(raw)]
        return await asyncio.to_thread(cls._retrieve, retriever, query, channel)

    def _rerank(self, query: str, candidates: list[RAGResult]) -> list[RAGResult]:
        if not candidates:
            return []
        raw = None
        found = False
        for name in ("rerank", "rank"):
            method = getattr(self.reranker, name, None)
            if callable(method):
                raw = method(query, candidates)
                found = True
                break
        if not found:
            method = getattr(self.reranker, "invoke", None)
            if callable(method):
                raw = method({"query": query, "documents": candidates})
            elif callable(self.reranker):
                raw = self.reranker(query, candidates)
            else:
                raise TypeError(
                    "reranker must define rerank, rank, invoke, or be callable"
                )
        if inspect.isawaitable(raw):
            close = getattr(raw, "close", None)
            if callable(close):
                close()
            raise TypeError("reranker is async-only; use RAGRetriever.ainvoke()")
        return self._normalize_reranked(raw, candidates)

    async def _arerank(
        self, query: str, candidates: list[RAGResult]
    ) -> list[RAGResult]:
        if not candidates:
            return []
        for name in ("arerank", "arank"):
            method = getattr(self.reranker, name, None)
            if callable(method):
                raw = method(query, candidates)
                raw = await raw if inspect.isawaitable(raw) else raw
                return self._normalize_reranked(raw, candidates)
        method = getattr(self.reranker, "ainvoke", None)
        if callable(method):
            raw = method({"query": query, "documents": candidates})
            raw = await raw if inspect.isawaitable(raw) else raw
            return self._normalize_reranked(raw, candidates)
        if self._is_async_callable(self.reranker):
            raw = self.reranker(query, candidates)
            raw = await raw if inspect.isawaitable(raw) else raw
            return self._normalize_reranked(raw, candidates)
        return await asyncio.to_thread(self._rerank, query, candidates)

    @staticmethod
    def _is_async_callable(value: Any) -> bool:
        if inspect.iscoroutinefunction(value):
            return True
        return callable(value) and inspect.iscoroutinefunction(type(value).__call__)

    @staticmethod
    def _items(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            for key in ("results", "documents", "matches", "nodes"):
                nested = value.get(key)
                if nested is not None and not isinstance(nested, (str, bytes, Mapping)):
                    return list(nested)
            return [value]
        for key in ("results", "documents", "matches", "nodes"):
            nested = getattr(value, key, None)
            if nested is not None and not isinstance(nested, (str, bytes, Mapping)):
                return list(nested)
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, Iterable):
            return list(value)
        return [value]

    @classmethod
    def _normalize(cls, item: Any, channel: str | None) -> RAGResult:
        tuple_score: float | None = None
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and cls._number(item[1]) is not None
        ):
            item, tuple_score = item[0], cls._number(item[1])

        if isinstance(item, Mapping):
            nested = item.get("document", item.get("node"))
            if nested is not None:
                result = cls._normalize(nested, channel)
                result["metadata"].update(cls._mapping_metadata(item))
                outer_score = cls._mapping_score(item)
                if outer_score is not None:
                    result["score"] = outer_score
                elif tuple_score is not None:
                    result["score"] = tuple_score
                if result["source"] is None and item.get("source") is not None:
                    result["source"] = str(item["source"])
                return result
            content = cls._mapping_content(item)
            metadata = cls._mapping_metadata(item)
            score = cls._mapping_score(item)
            source = item.get("source") or cls._source(metadata)
        elif isinstance(item, bytes):
            content, metadata, score, source = item.decode(), {}, None, None
        elif isinstance(item, str):
            content, metadata, score, source = item, {}, None, None
        else:
            nested = getattr(item, "node", None)
            if nested is not None:
                result = cls._normalize(nested, channel)
                outer_score = cls._number(getattr(item, "score", None))
                if outer_score is not None:
                    result["score"] = outer_score
                return result
            content = cls._object_content(item)
            raw_metadata = getattr(item, "metadata", {})
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            for key in (
                "id",
                "page",
                "chunk_id",
                "document_id",
                "node_id",
                "ref_doc_id",
                "title",
                "url",
                "source",
            ):
                value = getattr(item, key, None)
                if value is not None:
                    metadata.setdefault(key, value)
            score = cls._number(getattr(item, "score", None))
            source = getattr(item, "source", None) or cls._source(metadata)

        if channel is not None:
            channels = metadata.get("retrieval_channels", [])
            if not isinstance(channels, list):
                channels = [channels]
            if channel not in channels:
                channels.append(channel)
            metadata["retrieval_channels"] = channels
        return {
            "content": str(content),
            "score": tuple_score if tuple_score is not None else score,
            "source": str(source) if source is not None else None,
            "metadata": metadata,
        }

    @staticmethod
    def _mapping_content(item: Mapping[str, Any]) -> Any:
        for key in ("content", "page_content", "text"):
            if key in item:
                return item[key]
        return ""

    @classmethod
    def _object_content(cls, item: Any) -> Any:
        for key in ("page_content", "content", "text"):
            value = getattr(item, key, None)
            if value is not None:
                return value
        get_content = getattr(item, "get_content", None)
        if callable(get_content):
            return get_content()
        return str(item)

    @staticmethod
    def _mapping_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
        raw_metadata = item.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        excluded = {
            "content",
            "page_content",
            "text",
            "metadata",
            "score",
            "relevance_score",
            "similarity",
            "document",
            "node",
            "index",
            "corpus_id",
        }
        for key, value in item.items():
            if key not in excluded:
                metadata.setdefault(key, value)
        return metadata

    @classmethod
    def _mapping_score(cls, item: Mapping[str, Any]) -> float | None:
        for key in ("score", "relevance_score", "similarity"):
            if (score := cls._number(item.get(key))) is not None:
                return score
        return None

    @staticmethod
    def _number(value: Any) -> float | None:
        return (
            float(value)
            if isinstance(value, Real) and not isinstance(value, bool)
            else None
        )

    @staticmethod
    def _source(metadata: Mapping[str, Any]) -> Any:
        for key in ("source", "url", "document_id", "title"):
            if metadata.get(key) is not None:
                return metadata[key]
        return None

    @classmethod
    def _deduplicate(cls, candidates: list[RAGResult]) -> list[RAGResult]:
        unique: list[RAGResult] = []
        positions: dict[tuple[Any, ...], int] = {}
        for candidate in candidates:
            keys = cls._dedupe_keys(candidate)
            existing = next((positions[key] for key in keys if key in positions), None)
            if existing is None:
                position = len(unique)
                unique.append(cls._copy_result(candidate))
                for key in keys:
                    positions[key] = position
                continue
            current = unique[existing]
            for key in keys:
                positions[key] = existing
            incoming_metadata = candidate["metadata"]
            for name, value in incoming_metadata.items():
                if name == "retrieval_channels":
                    channels = current["metadata"].setdefault(name, [])
                    for channel in value:
                        if channel not in channels:
                            channels.append(channel)
                else:
                    current["metadata"].setdefault(name, value)
            if current["source"] is None:
                current["source"] = candidate["source"]
            if candidate["score"] is not None and (
                current["score"] is None or candidate["score"] > current["score"]
            ):
                current["score"] = candidate["score"]
        return unique

    @staticmethod
    def _dedupe_keys(candidate: RAGResult) -> list[tuple[Any, ...]]:
        metadata = candidate["metadata"]
        chunk_id = metadata.get("chunk_id") or metadata.get("node_id")
        document_id = metadata.get("document_id") or metadata.get("ref_doc_id")
        keys: list[tuple[Any, ...]] = []
        if chunk_id is not None:
            keys.append(("chunk", str(document_id), str(chunk_id)))
        if document_id is not None:
            keys.append(
                (
                    "document",
                    str(document_id),
                    str(metadata.get("page")),
                    candidate["content"].strip(),
                )
            )
        if metadata.get("id") is not None:
            keys.append(("id", str(metadata["id"])))
        content = candidate["content"].strip()
        if content:
            keys.append(("content", content))
        return keys or [("empty", candidate["source"], str(metadata))]

    @classmethod
    def _normalize_reranked(
        cls, value: Any, candidates: list[RAGResult]
    ) -> list[RAGResult]:
        if value is None:
            return [cls._copy_result(candidate) for candidate in candidates]
        items = cls._items(value)
        if len(items) == len(candidates) and all(
            cls._number(item) is not None for item in items
        ):
            scored = [
                {**cls._copy_result(candidate), "score": cls._number(score)}
                for candidate, score in zip(candidates, items, strict=True)
            ]
            return sorted(
                scored,
                key=lambda item: (
                    item["score"] if item["score"] is not None else float("-inf")
                ),
                reverse=True,
            )

        ranked: list[RAGResult] = []
        for item in items:
            index = cls._rerank_index(item)
            if index is not None:
                if not 0 <= index < len(candidates):
                    raise ValueError(
                        f"reranker returned invalid candidate index: {index}"
                    )
                result = cls._copy_result(candidates[index])
                if isinstance(item, Mapping):
                    score = cls._mapping_score(item)
                    if score is not None:
                        result["score"] = score
                    result["metadata"].update(cls._mapping_metadata(item))
                else:
                    score = cls._number(getattr(item, "score", None))
                    if score is None:
                        score = cls._number(getattr(item, "relevance_score", None))
                    if score is not None:
                        result["score"] = score
                ranked.append(result)
            else:
                ranked.append(cls._normalize(item, None))
        return cls._deduplicate(ranked)

    @staticmethod
    def _rerank_index(item: Any) -> int | None:
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, Mapping):
            value = item.get("index", item.get("corpus_id"))
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        else:
            value = getattr(item, "index", getattr(item, "corpus_id", None))
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    @staticmethod
    def _copy_result(result: RAGResult) -> RAGResult:
        return {
            "content": result["content"],
            "score": result["score"],
            "source": result["source"],
            "metadata": dict(result["metadata"]),
        }
