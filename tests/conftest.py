import asyncio
import sqlite3

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from agent_harness import configure_default_persistence


class PersistentSaverStub(BaseCheckpointSaver):
    """Sync/async adapter over a real temporary SQLite checkpoint database."""

    def __init__(self, path):
        super().__init__()
        self.path = str(path)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.backend = SqliteSaver(self.connection)
        self.deleted = []

    def close(self):
        self.connection.close()

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
        return await asyncio.to_thread(self.backend.get_tuple, *args, **kwargs)

    async def alist(self, *args, **kwargs):
        items = await asyncio.to_thread(
            lambda: list(self.backend.list(*args, **kwargs))
        )
        for item in items:
            yield item

    async def aput(self, *args, **kwargs):
        return await asyncio.to_thread(self.backend.put, *args, **kwargs)

    async def aput_writes(self, *args, **kwargs):
        return await asyncio.to_thread(self.backend.put_writes, *args, **kwargs)

    async def adelete_thread(self, *args, **kwargs):
        self.deleted.append(args[0])
        return await asyncio.to_thread(self.backend.delete_thread, *args, **kwargs)


class PersistentStoreStub(BaseStore):
    def __init__(self, path):
        self.path = str(path)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.backend = SqliteStore(self.connection)
        self.backend.setup()
        self.connection.commit()

    def close(self):
        self.connection.close()

    def batch(self, *args, **kwargs):
        return self.backend.batch(*args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await asyncio.to_thread(self.backend.batch, *args, **kwargs)


@pytest.fixture
def persistent_defaults(tmp_path):
    saver = PersistentSaverStub(tmp_path / "checkpoints.sqlite")
    store = PersistentStoreStub(tmp_path / "store.sqlite")
    configure_default_persistence(checkpointer=saver, store=store)
    try:
        yield saver, store
    finally:
        configure_default_persistence(checkpointer=None, store=None)
        saver.close()
        store.close()
