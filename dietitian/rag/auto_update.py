"""Single-process background source watcher with staged snapshot publication."""
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

from rag.source_files import read_directory, source_fingerprint

log = logging.getLogger(__name__)


class SourceChanged(RuntimeError):
    pass


class KnowledgeAutoUpdater:
    def __init__(self, store, directory, sources=None, *, interval=5, embedder=None, retry_seconds=30):
        self.store = store
        self.directory = Path(directory).resolve()
        self.sources = Path(sources).resolve() if sources else self.directory.parent / "sources.json"
        self.interval = max(1, interval)
        self.embedder = embedder
        self._stop = Event()
        self._run_lock = Lock()
        self._thread = None
        self._pending = None
        self._retry_after = 0
        self.retry_seconds = max(0, retry_seconds)
        self.state = {"enabled": True, "phase": "starting", "last_checked": None,
                      "last_updated": None, "error": None, "interval_seconds": self.interval}

    def _expected_signature(self, metadata):
        if metadata.get("source_signature"):
            return metadata["source_signature"]
        if self.store.storage is not None and metadata.get("dataset_id"):
            manifest = self.store.storage.source_manifest(metadata["dataset_id"])
            return source_fingerprint({name: info["sha256"] for name, info in manifest["files"].items()},
                                      manifest.get("sources", {}))
        return None

    def run_once(self):
        if not self._run_lock.acquire(blocking=False):
            return
        try:
            if monotonic() < self._retry_after:
                return
            self.state["last_checked"] = datetime.now(timezone.utc).isoformat()
            self.state["error"] = None
            documents, signature = read_directory(self.directory, self.sources)
            with self.store._update_lock:
                _, previous = self.store._read()
                expected = self._expected_signature(previous)
                if signature == expected:
                    self._pending = None
                    self.state["phase"] = "up_to_date"
                    return
                # Two equal observations avoid publishing intermediate file saves.
                if self._pending != signature:
                    self._pending = signature
                    self.state["phase"] = "waiting_for_stable_files"
                    return
                if self._stop.is_set():
                    return
                self.state["phase"] = "building"
                def guard():
                    if self._stop.is_set() or read_directory(self.directory, self.sources)[1] != signature:
                        raise SourceChanged("Sources changed while building")
                    if self.store._read()[1].get("version") != previous.get("version"):
                        raise SourceChanged("Another publisher activated a version")
                    if self.store.storage is None and self.store._read()[1].get("built_at") != previous.get("built_at"):
                        raise SourceChanged("Another local publisher activated a version")
                if self.store.storage is not None:
                    staged = self.store.storage.stage_documents(documents)
                    candidate = self.store.rebuild(dataset_id=staged["dataset_id"],
                        activate=False, embedder=self.embedder, source_signature=signature)
                    self.state["phase"] = "syncing_milvus"
                    self.store.activate_version(candidate["version"], expected_hash=candidate["sha256"],
                        before_activate=guard, expected_current=previous.get("version"))
                else:
                    self.store.rebuild(documents=documents, embedder=self.embedder,
                                       source_signature=signature, before_activate=guard)
                self._pending = None
                self.state.update(phase="up_to_date", last_updated=datetime.now(timezone.utc).isoformat())
                log.info("Knowledge snapshot and vector index updated")
        except SourceChanged:
            self._pending = None
            self.state["phase"] = "waiting_for_stable_files"
        except Exception as error:
            self.state.update(phase="error", error=type(error).__name__)
            self._retry_after = monotonic() + self.retry_seconds
            log.warning("Knowledge auto-update failed (%s); previous version retained", type(error).__name__)
        finally:
            self._run_lock.release()

    def start(self):
        def work():
            while not self._stop.is_set():
                self.run_once()
                self._stop.wait(self.interval)
        self._thread = Thread(target=work, name="knowledge-auto-update", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def configured_auto_updater(store):
    if os.getenv("RAG_AUTO_UPDATE", "false").lower() not in {"true", "1", "yes"}:
        return None
    root = Path(__file__).resolve().parents[1]
    directory = Path(os.getenv("RAG_WATCH_DIRECTORY", "knowledge/documents"))
    if not directory.is_absolute():
        directory = root / directory
    sources = os.getenv("RAG_WATCH_SOURCES")
    if sources:
        sources = Path(sources)
        if not sources.is_absolute():
            sources = root / sources
    return KnowledgeAutoUpdater(store, directory, sources,
        interval=max(1, int(os.getenv("RAG_WATCH_INTERVAL", "5"))),
        retry_seconds=max(1, int(os.getenv("RAG_AUTO_UPDATE_RETRY_SECONDS", "30"))))
