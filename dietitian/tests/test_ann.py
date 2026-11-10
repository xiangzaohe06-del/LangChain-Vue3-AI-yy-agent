import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rag.ann import AnnIndex
from rag.pipeline import PipelineSettings
from rag.store import KnowledgeStore


class AnnTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            'RAG_VECTOR_BACKEND': 'usearch',
            'RAG_HNSW_M': '16', 'RAG_HNSW_EF_CONSTRUCTION': '200',
            'RAG_HNSW_EF_SEARCH': '256', 'RAG_EMBEDDING_MODEL': 'text-embedding-v4',
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_single_and_multi_query_cosine_threshold_and_top_k(self):
        rows = [{'vector': [3., 0.]}, {'vector': [0., 4.]}, {'vector': [-1., 0.]}]
        ann = AnnIndex()
        results, trace = ann.search(rows, {}, [[2., 0.]], count=20, minimum=.35)
        self.assertEqual([i for i, _ in results[0]], [0])
        self.assertAlmostEqual(results[0][0][1], 1., places=5)
        self.assertFalse(trace['ann_cache_hit'])
        results, trace = ann.search(rows, {}, [[0., 1.], [-1., 0.]], count=1, minimum=.35)
        self.assertEqual([[i for i, _ in ranking] for ranking in results], [[1], [2]])
        self.assertTrue(trace['ann_cache_hit'])

    def test_snapshot_reorder_and_version_change_rebuild(self):
        rows = [{'vector': [1., 0.]}, {'vector': [0., 1.]}]
        ann = AnnIndex()
        ann.search(rows, {'version': 'v1'}, [[1., 0.]], count=1, minimum=.35)
        replacement = list(reversed(rows))
        results, trace = ann.search(replacement, {'version': 'v2'}, [[1., 0.]], count=1, minimum=.35)
        self.assertEqual(results[0][0][0], 1)
        self.assertFalse(trace['ann_cache_hit'])
        _, trace = ann.search(replacement, {'version': 'v3'}, [[1., 0.]], count=1, minimum=.35)
        self.assertFalse(trace['ann_cache_hit'])

    def test_invalid_vectors_and_dimensions_are_rejected(self):
        for vector in ([0., 0.], [float('nan'), 1.], [float('inf'), 1.]):
            with self.assertRaises(ValueError):
                AnnIndex().search([{'vector': vector}], {}, [[1., 0.]], count=1, minimum=0)
        ann = AnnIndex()
        rows = [{'vector': [1., 0.]}]
        for query in ([1., 0., 0.], [0., 0.], [float('nan'), 1.]):
            with self.assertRaises(ValueError):
                ann.search(rows, {}, [query], count=1, minimum=0)

    def test_failed_replacement_preserves_previous_index(self):
        ann = AnnIndex()
        rows = [{'vector': [1., 0.]}]
        ann.search(rows, {}, [[1., 0.]], count=1, minimum=0)
        with self.assertRaises(ValueError):
            ann.search([{'vector': [0., 0.]}], {}, [[1., 0.]], count=1, minimum=0)
        _, trace = ann.search(rows, {}, [[1., 0.]], count=1, minimum=0)
        self.assertTrue(trace['ann_cache_hit'])

    def test_recall_against_exact_cosine_on_seeded_fixture(self):
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(1500, 48)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1)[:, None]
        queries = rng.normal(size=(20, 48)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1)[:, None]
        rows = [{'vector': vector.tolist()} for vector in matrix]
        rankings, _ = AnnIndex().search(rows, {}, queries, count=10, minimum=-1)
        exact = np.argsort(-(queries @ matrix.T), axis=1)[:, :10]
        recall = sum(len(set(i for i, _ in ranking) & set(expected))
                     for ranking, expected in zip(rankings, exact)) / 200
        self.assertGreaterEqual(recall, .95)

    def test_store_pipeline_uses_ann_and_reuses_local_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / 'documents'
            docs.mkdir()
            (docs / 'alpha.md').write_text('alpha vegetable', encoding='utf-8')
            (docs / 'beta.md').write_text('beta fruit', encoding='utf-8')
            class Embedder:
                def embed_documents(self, texts):
                    return [[1., 0.] if 'alpha' in text else [0., 1.] for text in texts]
                def embed_query(self, query):
                    return [0., 1.]
            store = KnowledgeStore(root / 'index.sqlite3', docs)
            embedder = Embedder()
            store.rebuild(embedder=embedder)
            settings = PipelineSettings(rewrite=False, rerank=False)
            first = store.search('unmatched', embedder=embedder, settings=settings)
            second = store.search('unmatched', embedder=embedder, settings=settings)
            self.assertEqual(first['mode'], 'hybrid')
            self.assertEqual(first['trace']['vector_backend'], 'usearch_hnsw')
            self.assertEqual(first['hits'][0]['filename'], 'beta.md')
            self.assertNotIn('vector', first['hits'][0])
            self.assertFalse(first['trace']['ann_cache_hit'])
            self.assertTrue(second['trace']['ann_cache_hit'])
            (docs / 'beta.md').unlink()
            store.rebuild(embedder=embedder)
            third = store.search('unmatched', embedder=embedder, settings=settings)
            self.assertFalse(third['trace']['ann_cache_hit'])
            self.assertEqual(third['hits'], [])

    def test_ann_failure_falls_back_to_bm25_with_warning(self):
        class Snapshot:
            def load(self):
                return [{'id': 'a', 'title': 'alpha', 'filename': 'alpha.md',
                         'page': None, 'part': 1, 'content': 'alpha', 'vector': [1., 0.]}], {
                    'embedding_model': 'text-embedding-v4'}
        class Embedder:
            def embed_query(self, query):
                return [1., 0.]
        store = KnowledgeStore(storage=Snapshot())
        with patch.object(store._vector_index, 'search', side_effect=RuntimeError('secret')):
            result = store.search('alpha', embedder=Embedder(),
                                  settings=PipelineSettings(rewrite=False, rerank=False))
        self.assertEqual(result['mode'], 'lexical')
        self.assertTrue(result['hits'])
        self.assertTrue(result['warning'])
        self.assertNotIn('secret', json.dumps(result))
