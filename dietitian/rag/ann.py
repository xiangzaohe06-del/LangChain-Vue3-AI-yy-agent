"""Process-local HNSW cache for immutable knowledge snapshots."""
from __future__ import annotations

import os
from threading import RLock
from time import perf_counter


def parameter(name: str, default: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(os.getenv(name, str(default)))))


def configured_vector_index():
    backend = os.getenv("RAG_VECTOR_BACKEND", "usearch").lower()
    if backend == "milvus":
        from rag.milvus_index import MilvusIndex
        return MilvusIndex()
    if backend == "usearch":
        return AnnIndex()
    raise ValueError("RAG_VECTOR_BACKEND must be milvus or usearch")


class AnnIndex:
    """Keep one snapshot indexed; publish replacements only after a successful build.

    Callers must treat rows as immutable and supply a new list on snapshot changes.
    Integer HNSW keys are positions in that exact rows list, not persistent chunk IDs.
    """

    def __init__(self):
        self._lock = RLock()
        self._rows = None
        self._signature = None
        self._index = None

    def search(self, rows, metadata, vectors, *, count: int, minimum: float):
        import numpy as np
        from usearch.index import BatchMatches, Index

        connectivity = parameter("RAG_HNSW_M", 16, 4, 64)
        construction = parameter("RAG_HNSW_EF_CONSTRUCTION", 200, 16, 2000)
        expansion = max(count, parameter("RAG_HNSW_EF_SEARCH", 256, 16, 4000))
        signature = (metadata.get("version"), metadata.get("embedding_model"),
                     connectivity, construction)
        started = perf_counter()
        with self._lock:
            cached = self._rows is rows and self._signature == signature
            if not cached:
                matrix = np.asarray([row["vector"] for row in rows], dtype=np.float32)
                if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
                    raise ValueError("Empty or inconsistent document vectors")
                norms = np.linalg.norm(matrix, axis=1)
                if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms == 0).any():
                    raise ValueError("Invalid document vectors")
                matrix /= norms[:, None]
                index = Index(ndim=matrix.shape[1], metric="cos", dtype="f32",
                              connectivity=connectivity, expansion_add=construction,
                              expansion_search=expansion)
                index.add(np.arange(len(rows), dtype=np.uint64), matrix, threads=1)
                self._index, self._rows, self._signature = index, rows, signature

            index = self._index
            queries = np.asarray(vectors, dtype=np.float32)
            if queries.ndim != 2 or queries.shape[1] != index.ndim:
                raise ValueError("Embedding dimensions changed; rebuild the index")
            norms = np.linalg.norm(queries, axis=1)
            if not np.isfinite(queries).all() or not np.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("Invalid query vectors")
            queries /= norms[:, None]
            index.expansion_search = expansion
            matches = index.search(queries, count=min(count, len(rows)), exact=False, threads=1)
            rankings = []
            batches = [matches[position] for position in range(len(queries))] if isinstance(matches, BatchMatches) else [matches]
            for match in batches:
                ranking = []
                for key, distance in zip(match.keys, match.distances):
                    # USearch cosine distance = 1 - cosine similarity.
                    similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
                    if similarity >= minimum:
                        ranking.append((int(key), similarity))
                rankings.append(ranking)
            return rankings, {
                "vector_backend": "usearch_hnsw", "ann_cache_hit": cached,
                "ann_vectors": len(rows), "ann_m": connectivity,
                "ann_ef_construction": construction, "ann_ef_search": expansion,
                "ann_ms": round((perf_counter() - started) * 1000, 2),
            }
