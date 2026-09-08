import asyncio
import sys
import types

from agent_harness.memory import LongTermMemory, MemoryConfig


class Item:
    def __init__(self, value):
        self.value = value


class Store:
    def __init__(self):
        self.values = {}

    def search(self, namespace, *, query, limit):
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
        self.store.values.setdefault(self._namespace(config), []).append({"preference": "tea"})

    def search(self, *, query, config):
        return self.store.search(self._namespace(config), query=query, limit=self.query_limit)

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
    memory = LongTermMemory(store, MemoryConfig(instructions="extract"), model)

    assert captured == {
        "model": model,
        "namespace": ("memory", "{agent_name}", "{memory_id}"),
        "store": store,
        "query_limit": 6,
        "kwargs": {"instructions": "extract"},
    }

    memory.update("assistant", "user-1", ["likes tea"])
    # Session IDs never enter the namespace, so another session using the same
    # agent and memory ID sees the persisted preference.
    assert memory.load("assistant", "user-1", "preference") == [{"preference": "tea"}]

    asyncio.run(memory.aupdate("assistant", "user-2", ["likes tea"]))
    assert memory.load("assistant", "user-2", "preference") == [{"preference": "tea"}]

    # The same memory identity remains isolated between agents.
    assert memory.load("other-assistant", "user-1", "preference") == []
