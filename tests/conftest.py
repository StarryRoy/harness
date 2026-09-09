import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from agent_harness import configure_default_persistence


class PersistentSaverStub(BaseCheckpointSaver):
    """Interface-complete explicit test double; production rejects its backend type."""

    def __init__(self):
        super().__init__()
        self.backend = InMemorySaver()
        self.deleted = []

    def get_tuple(self, *args, **kwargs):
        return self.backend.get_tuple(*args, **kwargs)

    def list(self, *args, **kwargs):
        return self.backend.list(*args, **kwargs)

    def put(self, *args, **kwargs):
        return self.backend.put(*args, **kwargs)

    def put_writes(self, *args, **kwargs):
        return self.backend.put_writes(*args, **kwargs)

    def delete_thread(self, *args, **kwargs):
        self.deleted.append(args[0])
        return self.backend.delete_thread(*args, **kwargs)

    async def aget_tuple(self, *args, **kwargs):
        return await self.backend.aget_tuple(*args, **kwargs)

    async def alist(self, *args, **kwargs):
        async for item in self.backend.alist(*args, **kwargs):
            yield item

    async def aput(self, *args, **kwargs):
        return await self.backend.aput(*args, **kwargs)

    async def aput_writes(self, *args, **kwargs):
        return await self.backend.aput_writes(*args, **kwargs)

    async def adelete_thread(self, *args, **kwargs):
        self.deleted.append(args[0])
        return await self.backend.adelete_thread(*args, **kwargs)


class PersistentStoreStub(BaseStore):
    def __init__(self):
        self.backend = InMemoryStore()

    def batch(self, *args, **kwargs):
        return self.backend.batch(*args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await self.backend.abatch(*args, **kwargs)


@pytest.fixture
def persistent_defaults():
    saver = PersistentSaverStub()
    store = PersistentStoreStub()
    configure_default_persistence(checkpointer=saver, store=store)
    try:
        yield saver, store
    finally:
        configure_default_persistence(checkpointer=None, store=None)
