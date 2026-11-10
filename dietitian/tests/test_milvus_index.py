import copy
import json
import os
import unittest
from unittest.mock import patch

from rag.milvus_index import MilvusIndex
from rag.pipeline import PipelineSettings
from rag.store import KnowledgeStore


class FakeSchema:
    def add_field(self, *args, **kwargs):
        pass


class FakeIndexes:
    def add_index(self, **kwargs):
        self.params = kwargs


class FakeMilvus:
    def __init__(self):
        self.collections = {}
        self.aliases = {}
        self.fail_insert = False
        self.search_override = None
        self.search_request = None

    def name(self, name):
        return self.aliases.get(name, name)

    def has_collection(self, collection_name, **kwargs):
        return self.name(collection_name) in self.collections

    def create_schema(self, **kwargs):
        schema = FakeSchema()
        schema.description = kwargs['description']
        return schema

    def create_collection(self, collection_name, schema, **kwargs):
        self.collections[collection_name] = {'description': schema.description, 'data': [], 'index': None}

    def insert(self, collection_name, data, **kwargs):
        if self.fail_insert:
            raise RuntimeError('simulated insertion failure')
        self.collections[self.name(collection_name)]['data'].extend(copy.deepcopy(data))

    def flush(self, **kwargs):
        pass

    def prepare_index_params(self):
        return FakeIndexes()

    def create_index(self, collection_name, index_params, **kwargs):
        self.collections[self.name(collection_name)]['index'] = index_params.params

    def describe_collection(self, collection_name, **kwargs):
        return self.collections[self.name(collection_name)]

    def describe_index(self, collection_name, **kwargs):
        return self.collections[self.name(collection_name)]['index']

    def load_collection(self, **kwargs):
        pass

    def query(self, collection_name, **kwargs):
        return [{'count(*)': len(self.collections[self.name(collection_name)]['data'])}]

    def create_alias(self, collection_name, alias, **kwargs):
        if alias in self.aliases:
            raise RuntimeError('alias exists')
        self.aliases[alias] = collection_name

    def search(self, collection_name, data, limit, **kwargs):
        self.search_request = kwargs
        if self.search_override is not None:
            return self.search_override
        import math
        output = []
        for query in data:
            matches = []
            for row in self.collections[self.name(collection_name)]['data']:
                norm = math.sqrt(sum(v*v for v in row['vector']))
                score = sum(a*b for a, b in zip(query, row['vector'])) / norm
                matches.append({'id': row['chunk_id'], 'distance': score})
            output.append(sorted(matches, key=lambda hit: hit['distance'], reverse=True)[:limit])
        return output


class MilvusTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            'MILVUS_COLLECTION_PREFIX': 'test_hnsw', 'RAG_HNSW_M': '16',
            'RAG_HNSW_EF_CONSTRUCTION': '200', 'RAG_HNSW_EF_SEARCH': '256',
        })
        env.start()
        self.addCleanup(env.stop)
        self.client = FakeMilvus()
        self.index = MilvusIndex(client=self.client)
        self.rows = [
            {'id': 'a', 'filename': 'a.md', 'title': 'alpha', 'content': 'alpha',
             'part': 1, 'page': None, 'vector': [1., 0.]},
            {'id': 'b', 'filename': 'b.md', 'title': 'beta', 'content': 'beta',
             'part': 1, 'page': None, 'vector': [0., 1.]},
        ]
        self.metadata = {'embedding_model': 'text-embedding-v4', 'version': 'v1'}

    def test_import_builds_hnsw_and_persists_payload_then_exposes_alias(self):
        result = self.index.sync(self.rows, self.metadata)
        collection = self.client.collections[self.client.aliases[result['collection']]]
        self.assertEqual(collection['index']['index_type'], 'HNSW')
        self.assertEqual(collection['index']['metric_type'], 'COSINE')
        self.assertEqual(collection['index']['params'], {'M': 16, 'efConstruction': 200})
        self.assertEqual(collection['data'][0]['payload']['content'], 'alpha')
        self.assertNotIn('vector', collection['data'][0]['payload'])
        self.assertTrue(self.index.sync(self.rows, self.metadata)['reused'])
        self.assertEqual(len(self.client.collections), 1)

    def test_query_does_not_import_missing_snapshot(self):
        with self.assertRaisesRegex(RuntimeError, 'sync-milvus'):
            self.index.search(self.rows, self.metadata, [[1., 0.]], count=20, minimum=.35)
        self.assertFalse(self.client.collections)

    def test_cosine_scores_multiquery_threshold_and_ef(self):
        self.index.sync(self.rows, self.metadata)
        rankings, trace = self.index.search(self.rows, self.metadata, [[4., 3.], [0., 1.]], count=20, minimum=.7)
        self.assertEqual(rankings[0][0][0], 0)
        self.assertAlmostEqual(rankings[0][0][1], .8)
        self.assertEqual(rankings[1], [(1, 1.)])
        self.assertEqual(trace['vector_backend'], 'milvus_hnsw')
        self.assertEqual(self.client.search_request['search_params']['params']['ef'], 256)

    def test_new_process_reuses_persistent_collection_and_maps_reordered_ids(self):
        first = self.index.sync(self.rows, self.metadata)
        fresh = MilvusIndex(client=self.client)
        reordered = list(reversed(self.rows))
        rankings, _ = fresh.search(reordered, self.metadata, [[1., 0.]], count=1, minimum=.35)
        self.assertEqual(rankings[0][0][0], 1)
        self.assertEqual(fresh.sync(reordered, self.metadata)['collection'], first['collection'])

    def test_failed_new_import_keeps_previous_alias_and_data(self):
        old = self.index.sync(self.rows, self.metadata)
        self.client.fail_insert = True
        with self.assertRaises(RuntimeError):
            self.index.sync(self.rows, {**self.metadata, 'version': 'v2'})
        self.assertEqual(list(self.client.aliases), [old['collection']])
        rankings, _ = self.index.search(self.rows, self.metadata, [[1., 0.]], count=1, minimum=.35)
        self.assertEqual(rankings[0][0][0], 0)

    def test_incomplete_or_wrong_index_is_rejected(self):
        result = self.index.sync(self.rows, self.metadata)
        target = self.client.collections[self.client.aliases[result['collection']]]
        target['index']['index_type'] = 'FLAT'
        with self.assertRaises(ValueError):
            MilvusIndex(client=self.client).search(self.rows, self.metadata, [[1., 0.]], count=1, minimum=0)
        target['index']['index_type'] = 'HNSW'
        target['data'].pop()
        with self.assertRaises(ValueError):
            MilvusIndex(client=self.client).sync(self.rows, self.metadata)

    def test_unknown_duplicate_or_nonfinite_results_are_rejected(self):
        self.index.sync(self.rows, self.metadata)
        for hits in ([{'id': 'unknown', 'distance': .8}],
                     [{'id': 'a', 'distance': float('nan')}],
                     [{'id': 'a', 'distance': .8}, {'id': 'a', 'distance': .7}]):
            self.client.search_override = [hits]
            with self.assertRaises(ValueError):
                self.index.search(self.rows, self.metadata, [[1., 0.]], count=20, minimum=0)

    def test_sdk_primary_field_name_and_entity_fallback(self):
        self.index.sync(self.rows, self.metadata)
        for hit in ({'chunk_id': 'a', 'distance': .9},
                    {'entity': {'chunk_id': 'a'}, 'distance': .9}):
            self.client.search_override = [[hit]]
            result, _ = self.index.search(self.rows, self.metadata, [[1., 0.]], count=1, minimum=.35)
            self.assertEqual(result, [[(0, .9)]])

    def test_invalid_documents_and_query_dimensions_are_rejected(self):
        for rows in ([self.rows[0], self.rows[0]], [{**self.rows[0], 'vector': [0., 0.]}],
                     [self.rows[0], {**self.rows[1], 'vector': [1.]}]):
            with self.assertRaises(ValueError):
                self.index.sync(rows, self.metadata)
        self.index.sync(self.rows, self.metadata)
        with self.assertRaises(ValueError):
            self.index.search(self.rows, self.metadata, [[1., 0., 0.]], count=1, minimum=0)

    def test_full_pipeline_preserves_ann_failure_warning_and_bm25_fallback(self):
        class Snapshot:
            def load(inner):
                return self.rows, self.metadata
        class Embedder:
            def embed_query(inner, query):
                return [0., 1.]
        store = KnowledgeStore(storage=Snapshot())
        store._vector_index = self.index
        settings = PipelineSettings(rewrite=False, rerank=False)
        missing = store.search('alpha', embedder=Embedder(), settings=settings)
        self.assertEqual(missing['mode'], 'lexical')
        self.assertTrue(missing['warning'])
        self.index.sync(self.rows, self.metadata)
        result = store.search('unmatched', embedder=Embedder(), settings=settings)
        self.assertEqual(result['mode'], 'hybrid')
        self.assertEqual(result['hits'][0]['filename'], 'b.md')
        self.assertEqual(result['trace']['vector_backend'], 'milvus_hnsw')

    def test_local_lite_uri_is_not_silently_used(self):
        with patch.dict(os.environ, {'MILVUS_URI': 'local.db'}):
            with self.assertRaises(ValueError):
                _ = MilvusIndex().client
