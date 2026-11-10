"""Explicit snapshot import and read-only HNSW search through Milvus Server.

COS/SQLite remains the text/BM25 snapshot source. Milvus owns durable vectors,
payload copies and HNSW indexes. Search never inserts data or builds indexes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from threading import RLock
from time import perf_counter
from urllib.parse import urlsplit
import uuid

from rag.ann import parameter


class MilvusIndex:
    def __init__(self, *, client=None):
        self._client = client
        self._lock = RLock()
        self._rows = None
        self._config = None
        self._snapshot = None
        self._verified = None

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient
            uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
            if urlsplit(uri).scheme not in {"http", "https"}:
                raise ValueError("MILVUS_URI must point to Milvus Server, not a Lite database file")
            self._client = MilvusClient(
                uri=uri, token=os.getenv("MILVUS_TOKEN", ""),
                db_name=os.getenv("MILVUS_DB_NAME", "default"), timeout=self.timeout,
            )
        return self._client

    @property
    def timeout(self):
        return parameter("MILVUS_TIMEOUT", 30, 1, 600)

    def _prepare(self, rows, metadata):
        prefix = os.getenv("MILVUS_COLLECTION_PREFIX", "nutrition_hnsw")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", prefix):
            raise ValueError("Invalid Milvus collection prefix")
        m = parameter("RAG_HNSW_M", 16, 4, 64)
        construction = parameter("RAG_HNSW_EF_CONSTRUCTION", 200, 16, 2000)
        config = prefix, m, construction, metadata.get("version"), metadata.get("embedding_model")
        if self._rows is rows and self._config == config:
            return self._snapshot
        if not rows or not metadata.get("embedding_model"):
            raise ValueError("A nonempty vector snapshot is required")
        ids, dimensions = {}, set()
        digest = hashlib.sha256()
        digest.update(json.dumps({"metadata": metadata, "m": m, "construction": construction},
                                 sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
        for position, row in enumerate(rows):
            key = row["id"]
            if not isinstance(key, str) or not key or len(key.encode()) > 128 or key in ids:
                raise ValueError("Invalid or duplicate chunk ID")
            vector = row["vector"]
            norm = math.sqrt(sum(float(v) ** 2 for v in vector))
            if not vector or not math.isfinite(norm) or norm == 0:
                raise ValueError("Invalid document vector")
            dimensions.add(len(vector))
            ids[key] = position
        if len(dimensions) != 1:
            raise ValueError("Inconsistent vector dimensions")
        # Canonical chunk order means new processes and reordered reads share the alias.
        for row in sorted(rows, key=lambda row: row["id"]):
            digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode())
            digest.update(b"\n")
        fingerprint = digest.hexdigest()
        snapshot = {
            "alias": f"{prefix}_{fingerprint[:24]}", "fingerprint": fingerprint,
            "positions": ids, "dimension": dimensions.pop(), "count": len(rows),
            "m": m, "construction": construction,
        }
        self._rows, self._config, self._snapshot = rows, config, snapshot
        self._verified = None
        return snapshot

    def _verify(self, snapshot, name):
        description = self.client.describe_collection(collection_name=name, timeout=self.timeout)
        manifest = json.loads(description.get("description", "{}"))
        if manifest.get("fingerprint") != snapshot["fingerprint"]:
            raise ValueError("Milvus snapshot fingerprint mismatch")
        details = self.client.describe_index(collection_name=name, index_name="vector_hnsw", timeout=self.timeout)
        if details.get("index_type") != "HNSW" or details.get("metric_type") != "COSINE":
            raise ValueError("Expected Milvus HNSW COSINE index")
        self.client.load_collection(collection_name=name, timeout=self.timeout)
        counts = self.client.query(collection_name=name, output_fields=["count(*)"],
                                   consistency_level="Strong", timeout=self.timeout)
        if not counts or counts[0].get("count(*)") != snapshot["count"]:
            raise ValueError("Incomplete Milvus snapshot")

    def sync(self, rows, metadata):
        """Build an isolated collection and expose its alias only after validation.

        Failure leaves old snapshots untouched. Unpublished staging collections are
        retained for diagnosis; never drop an existing collection automatically.
        """
        from pymilvus import DataType
        with self._lock:
            snapshot = self._prepare(rows, metadata)
            alias = snapshot["alias"]
            if self.client.has_collection(collection_name=alias, timeout=self.timeout):
                self._verify(snapshot, alias)
                self._verified = alias
                return {"vector_backend": "milvus_hnsw", "collection": alias,
                        "chunks": len(rows), "reused": True}
            staging = f"{alias}_{uuid.uuid4().hex[:8]}"
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False,
                description=json.dumps({"fingerprint": snapshot["fingerprint"], "chunks": len(rows)}))
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=snapshot["dimension"])
            schema.add_field("payload", DataType.JSON)
            self.client.create_collection(collection_name=staging, schema=schema,
                                          consistency_level="Strong", timeout=self.timeout)
            batch_size = parameter("MILVUS_INSERT_BATCH", 100, 1, 1000)
            for start in range(0, len(rows), batch_size):
                batch = [{"chunk_id": row["id"], "vector": row["vector"],
                          "payload": {k: v for k, v in row.items() if k != "vector"}}
                         for row in rows[start:start + batch_size]]
                self.client.insert(collection_name=staging, data=batch, timeout=self.timeout)
            self.client.flush(collection_name=staging, timeout=self.timeout)
            indexes = self.client.prepare_index_params()
            indexes.add_index(field_name="vector", index_name="vector_hnsw", index_type="HNSW",
                              metric_type="COSINE", params={"M": snapshot["m"],
                              "efConstruction": snapshot["construction"]})
            self.client.create_index(collection_name=staging, index_params=indexes, timeout=self.timeout)
            self._verify(snapshot, staging)
            try:
                self.client.create_alias(collection_name=staging, alias=alias, timeout=self.timeout)
            except Exception:
                # Another importer may have published this identical snapshot first.
                if not self.client.has_collection(collection_name=alias, timeout=self.timeout):
                    raise
                self._verify(snapshot, alias)
            self._verified = alias
            return {"vector_backend": "milvus_hnsw", "collection": alias,
                    "chunks": len(rows), "reused": False}

    def search(self, rows, metadata, vectors, *, count, minimum):
        started = perf_counter()
        with self._lock:
            snapshot = self._prepare(rows, metadata)
            alias = snapshot["alias"]
            cached = self._verified == alias
            if not cached:
                if not self.client.has_collection(collection_name=alias, timeout=self.timeout):
                    raise RuntimeError("Milvus snapshot missing; run python -m rag sync-milvus")
                self._verify(snapshot, alias)
                self._verified = alias
            queries = []
            for values in vectors:
                if len(values) != snapshot["dimension"]:
                    raise ValueError("Embedding dimensions changed; rebuild the snapshot")
                norm = math.sqrt(sum(float(v) ** 2 for v in values))
                if not math.isfinite(norm) or norm == 0:
                    raise ValueError("Invalid query vector")
                queries.append([float(v) / norm for v in values])
            expansion = max(count, parameter("RAG_HNSW_EF_SEARCH", 256, 16, 4000))
            matches = self.client.search(
                collection_name=alias, data=queries, anns_field="vector",
                output_fields=["chunk_id"],
                limit=min(count, len(rows)), search_params={"metric_type": "COSINE", "params": {"ef": expansion}},
                consistency_level="Strong", timeout=self.timeout,
            )
            if len(matches) != len(queries):
                raise ValueError("Milvus omitted a query result")
            rankings = []
            for batch in matches:
                ranking, seen = [], set()
                for hit in batch:
                    # Recent SDKs name the top-level PK after the schema field;
                    # older clients use id. Explicit output_fields covers both.
                    key = hit.get("chunk_id", hit.get("id"))
                    if key is None:
                        key = hit.get("entity", {}).get("chunk_id")
                    score = float(hit["distance"])
                    # Milvus COSINE returns similarity, unlike USearch's 1-similarity.
                    if key not in snapshot["positions"] or key in seen or not math.isfinite(score):
                        raise ValueError("Invalid Milvus result")
                    seen.add(key)
                    score = max(-1., min(1., score))
                    if score >= minimum:
                        ranking.append((snapshot["positions"][key], score))
                rankings.append(sorted(ranking, key=lambda item: item[1], reverse=True))
            return rankings, {"vector_backend": "milvus_hnsw", "milvus_collection": alias,
                "ann_cache_hit": cached, "ann_vectors": len(rows), "ann_m": snapshot["m"],
                "ann_ef_construction": snapshot["construction"], "ann_ef_search": expansion,
                "ann_ms": round((perf_counter() - started) * 1000, 2)}
