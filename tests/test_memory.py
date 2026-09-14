import asyncio
import sqlite3
import sys
import types

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.store.sqlite import SqliteStore

from agent_harness.memory import LongTermMemory, MemoryConfig


class Item:
    def __init__(self, value, score=1.0):
        self.value = value
        self.score = score


class Store:
    def __init__(self):
        self.values = {}
        self.searches = []

    def search(self, namespace, *, query, limit):
        self.searches.append((namespace, query, limit))
        return [Item(value) for value in self.values.get(namespace, [])][:limit]


class Manager:
    def __init__(self, namespace, store, query_limit):
        self.namespace = namespace
        self.store = store
        self.query_limit = query_limit
        self.calls = []

    def _namespace(self, config):
        configurable = config["configurable"]
        return tuple(
            configurable[part[1:-1]] if part.startswith("{") else part
            for part in self.namespace
        )

    def _write(self, value, config):
        self.calls.append((value, config))
        self.store.values.setdefault(self._namespace(config), []).append(
            {"preference": "tea"}
        )

    def search(self, *, query, limit, config):
        return self.store.search(
            self._namespace(config), query=query, limit=limit
        )

    def invoke(self, value, *, config):
        self._write(value, config)

    async def ainvoke(self, value, *, config):
        self._write(value, config)


def test_manager_uses_official_model_store_and_dynamic_namespace(monkeypatch):
    captured = {}

    def create(model, *, namespace, store, query_limit, **kwargs):
        captured.update(model=model, namespace=namespace, kwargs=kwargs)
        captured.update(store=store, query_limit=query_limit)
        return Manager(namespace, store, query_limit)

    monkeypatch.setitem(
        sys.modules,
        "langmem",
        types.SimpleNamespace(create_memory_store_manager=create),
    )
    store = Store()
    model = object()
    memory = LongTermMemory(
        store, MemoryConfig(instructions="extract", search_limit=2), model
    )

    assert captured == {
        "model": model,
        "namespace": ("memory", "{agent_name}", "{memory_id}"),
        "store": store,
        "query_limit": 2,
        "kwargs": {"instructions": "extract"},
    }

    memory.update("assistant", "user-1", ["likes tea"])
    # Session IDs never enter the namespace, so another session using the same
    # agent and memory ID sees the persisted preference.
    assert memory.load("assistant", "user-1", "preference") == [{"preference": "tea"}]
    assert store.searches[-1] == (
        ("memory", "assistant", "user-1"),
        "preference",
        2,
    )

    asyncio.run(memory.aupdate("assistant", "user-2", ["likes tea"]))
    assert memory.load("assistant", "user-2", "preference") == [{"preference": "tea"}]

    # The same memory identity remains isolated between agents.
    assert memory.load("other-assistant", "user-1", "preference") == []


def test_search_limit_must_be_positive_integer():
    for invalid in (0, -1):
        with pytest.raises(ValueError):
            MemoryConfig(search_limit=invalid)

    for invalid in (True, 1.5, "2"):
        with pytest.raises(TypeError):
            MemoryConfig(search_limit=invalid)


def test_load_uses_native_store_semantic_ranking_and_search_limit(tmp_path):
    def embeddings(texts):
        concepts = (
            ("pizza", "hungry", "eat"),
            ("python", "code", "programming"),
            ("concise", "report", "answer"),
        )
        return [
            [
                float(sum(text.lower().count(term) for term in terms))
                for terms in concepts
            ]
            for text in texts
        ]

    connection = sqlite3.connect(tmp_path / "semantic-memory.sqlite")
    store = SqliteStore(
        connection,
        index={"dims": 3, "embed": embeddings, "fields": ["$"]},
    )
    store.setup()
    connection.commit()
    namespace = ("memory", "assistant", "user-1")
    # The relevant item is deliberately older; recency-only retrieval would
    # choose the Python memory inserted afterwards.
    store.put(
        namespace,
        "food",
        {"kind": "Memory", "content": {"content": "User loves pizza"}},
    )
    store.put(
        namespace,
        "work",
        {"kind": "Memory", "content": {"content": "User writes Python code"}},
    )
    memory = LongTermMemory(
        store,
        MemoryConfig(search_limit=1),
        FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
    )

    try:
        assert memory.load("assistant", "user-1", "I am hungry") == [
            {"content": "User loves pizza"}
        ]
        assert memory.load("assistant", "user-2", "I am hungry") == []
    finally:
        connection.close()
