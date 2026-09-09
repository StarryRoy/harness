"""Persistent defaults and configuration for LangGraph state."""

from __future__ import annotations

import os
import pickle
import tempfile
import threading
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


class FileCheckpointSaver(InMemorySaver):
    """A small durable saver for local deployments.

    It preserves LangGraph's serializer-backed in-memory representation on disk and
    implements the normal ``BaseCheckpointSaver`` API. Distributed deployments should
    pass an official Postgres/SQLite saver instead.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        super().__init__()
        self._restore()

    def _restore(self) -> None:
        if not self.path.exists():
            return
        with self._lock, self.path.open("rb") as stream:
            data = pickle.load(stream)  # noqa: S301 - trusted application-owned file
            self.storage.update(data["storage"])
            self.writes.update(data["writes"])
            self.blobs.update(data["blobs"])

    def _persist(self) -> None:
        with self._lock:
            fd, temporary = tempfile.mkstemp(
                dir=self.path.parent, prefix=".checkpoint-"
            )
            try:
                with os.fdopen(fd, "wb") as stream:
                    pickle.dump(
                        {
                            "storage": dict(self.storage),
                            "writes": dict(self.writes),
                            "blobs": self.blobs,
                        },
                        stream,
                    )
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def put(
        self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any
    ) -> Any:
        with self._lock:
            result = super().put(config, checkpoint, metadata, new_versions)
            self._persist()
            return result

    def put_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        with self._lock:
            super().put_writes(config, writes, task_id, task_path)
            self._persist()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            self._persist()


def default_checkpoint_path(namespace: str = "default") -> Path:
    configured = os.getenv("AGENT_HARNESS_CHECKPOINT_PATH")
    if configured:
        return Path(configured)
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in namespace
    )
    return Path(".agent_harness/checkpoints") / f"{safe}.pkl"


class FileStore(InMemoryStore):
    """Durable local implementation of the LangGraph Store interface."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__()
        if self.path.exists():
            with self.path.open("rb") as stream:
                data, vectors = pickle.load(stream)  # noqa: S301
                self._data.update(data)
                self._vectors.update(vectors)

    def _persist_store(self) -> None:
        with self.path.open("wb") as stream:
            pickle.dump((dict(self._data), dict(self._vectors)), stream)

    def batch(self, ops: Any) -> Any:
        result = super().batch(ops)
        self._persist_store()
        return result

    async def abatch(self, ops: Any) -> Any:
        result = await super().abatch(ops)
        self._persist_store()
        return result


def default_store_path() -> Path:
    configured = os.getenv("AGENT_HARNESS_STORE_PATH")
    return Path(configured or ".agent_harness/store.pkl")
