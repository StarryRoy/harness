import asyncio
import inspect

from langchain_core.documents import Document
from langchain_core.tools import BaseTool

from agent_harness import RAGRetriever


class SyncRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.results


class IndexReranker:
    def __init__(self):
        self.call = None

    def rerank(self, query, documents):
        self.call = (query, documents)
        return [
            {"index": 2, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.88},
        ]


def test_rag_fuses_deduplicates_reranks_and_preserves_metadata():
    vector = SyncRetriever(
        [
            Document(
                page_content="shared",
                metadata={
                    "source": "guide.pdf",
                    "page": 3,
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "title": "Guide",
                },
            ),
            (
                Document(
                    page_content="vector only",
                    metadata={"source": "vector.txt", "chunk_id": "chunk-2"},
                ),
                0.7,
            ),
        ]
    )
    bm25 = SyncRetriever(
        [
            {
                "page_content": "shared",
                "score": 4.2,
                "metadata": {
                    "source": "guide.pdf",
                    "page": 3,
                    "url": "https://example.test/guide",
                },
            },
            {
                "text": "bm25 only",
                "source": "lexical.txt",
                "metadata": {"document_id": "doc-3"},
            },
        ]
    )
    reranker = IndexReranker()
    rag = RAGRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        reranker=reranker,
    )

    results = rag.invoke("refund policy")

    assert isinstance(rag, BaseTool)
    assert list(inspect.signature(RAGRetriever).parameters) == [
        "vector_retriever",
        "bm25_retriever",
        "reranker",
    ]
    assert vector.queries == bm25.queries == ["refund policy"]
    assert reranker.call[0] == "refund policy"
    assert len(reranker.call[1]) == 3
    assert [item["content"] for item in results] == ["bm25 only", "shared"]
    assert [item["score"] for item in results] == [0.99, 0.88]
    shared = results[1]
    assert shared["source"] == "guide.pdf"
    assert shared["metadata"]["page"] == 3
    assert shared["metadata"]["chunk_id"] == "chunk-1"
    assert shared["metadata"]["document_id"] == "doc-1"
    assert shared["metadata"]["title"] == "Guide"
    assert shared["metadata"]["url"] == "https://example.test/guide"
    assert shared["metadata"]["retrieval_channels"] == ["vector", "bm25"]
    assert set(shared) == {"content", "score", "source", "metadata"}


class AsyncRetriever:
    def __init__(self, content):
        self.content = content
        self.queries = []

    async def aretrieve(self, query):
        await asyncio.sleep(0)
        self.queries.append(query)
        return [{"content": self.content, "metadata": {"source": self.content}}]


class AsyncReranker:
    def __init__(self):
        self.documents = None

    async def arerank(self, query, documents):
        await asyncio.sleep(0)
        self.documents = documents
        return [0.2, 0.9]


def test_rag_async_recall_and_reranking():
    vector = AsyncRetriever("vector")
    bm25 = AsyncRetriever("bm25")
    reranker = AsyncReranker()
    rag = RAGRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        reranker=reranker,
    )

    results = asyncio.run(rag.ainvoke("async query"))

    assert vector.queries == bm25.queries == ["async query"]
    assert len(reranker.documents) == 2
    assert [item["content"] for item in results] == ["bm25", "vector"]
    assert [item["score"] for item in results] == [0.9, 0.2]


class RunnableReranker:
    def __init__(self):
        self.payload = None

    def invoke(self, payload):
        self.payload = payload
        return list(reversed(payload["documents"]))


class RetrieveOnly(SyncRetriever):
    invoke = None

    def retrieve(self, query):
        self.queries.append(query)
        return self.results


def test_rag_supports_runnable_style_reranker_and_tool_input():
    reranker = RunnableReranker()
    rag = RAGRetriever(
        vector_retriever=SyncRetriever(["vector reference"]),
        bm25_retriever=RetrieveOnly(["bm25 reference"]),
        reranker=reranker,
    )

    results = rag.invoke({"query": "question"})

    assert reranker.payload["query"] == "question"
    assert [item["content"] for item in results] == [
        "bm25 reference",
        "vector reference",
    ]
