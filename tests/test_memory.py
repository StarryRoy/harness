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
    def __init__(self, namespace):
        self.namespace = namespace
        self.calls = []

    def _write(self, value, config, store):
        self.calls.append((value, config))
        configurable = config["configurable"]
        namespace = tuple(
            configurable[part[1:-1]] if part.startswith("{") else part
            for part in self.namespace
        )
        store.values.setdefault(namespace, []).append({"preference": "tea"})

    def invoke(self, value, *, config, store):
        self._write(value, config, store)

    async def ainvoke(self, value, *, config, store):
        self._write(value, config, store)


def test_manager_uses_official_model_store_and_dynamic_namespace(monkeypatch):
    captured = {}

    def create(model, *, namespace, **kwargs):
        captured.update(model=model, namespace=namespace, kwargs=kwargs)
        return Manager(namespace)

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
        "kwargs": {"instructions": "extract"},
    }

    memory.update("assistant", "user-1", ["likes tea"])
    # Session IDs never enter the namespace, so another session using the same
    # agent and memory ID sees the persisted preference.
    assert memory.load("assistant", "user-1", "preference") == [{"preference": "tea"}]

    asyncio.run(memory.aupdate("assistant", "user-2", ["likes tea"]))
    assert memory.load("assistant", "user-2", "preference") == [{"preference": "tea"}]
