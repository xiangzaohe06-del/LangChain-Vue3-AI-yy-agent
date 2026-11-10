import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag.cos_storage import CosKnowledgeStorage
from rag.milvus_index import MilvusIndex
from rag.store import KnowledgeStore
from test_cos_storage import FakeCos
from test_milvus_index import FakeMilvus
from test_rag import FakeEmbeddings
from rag.auto_update import KnowledgeAutoUpdater


class UpdateFixture:
    def setUp(self):
        env = patch.dict(os.environ, {'RAG_VECTOR_BACKEND': 'milvus',
            'MILVUS_COLLECTION_PREFIX': 'update_test', 'RAG_QUERY_REWRITE': 'false', 'RAG_RERANK': 'false'})
        env.start()
        self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.docs = self.root / 'documents'
        self.docs.mkdir()
        (self.docs / 'chicken.md').write_text('鸡肉彩椒菜谱。', encoding='utf-8')
        self.cos = FakeCos()
        self.storage = CosKnowledgeStorage(client=self.cos, bucket='test-123', prefix='kb/test')
        self.milvus = FakeMilvus()
        self.store = KnowledgeStore(storage=self.storage)
        self.store._vector_index = MilvusIndex(client=self.milvus)

    def publish(self, **kwargs):
        return self.store.publish_directory(self.docs, embedder=FakeEmbeddings(), **kwargs)


class PublicationTests(UpdateFixture, unittest.TestCase):

    def test_publication_automatically_builds_milvus_before_pointer_switch(self):
        first = self.publish()
        old = self.cos.objects['kb/test/current.json']
        original = self.milvus.insert
        def insert(**kwargs):
            self.assertEqual(self.cos.objects['kb/test/current.json'], old)
            return original(**kwargs)
        self.milvus.insert = insert
        (self.docs / 'salt.md').write_text('成年人食盐摄入建议。', encoding='utf-8')
        second = self.publish()
        self.assertNotEqual(first['version'], second['version'])
        self.assertEqual(second['documents'], 2)
        self.assertEqual(len(self.milvus.aliases), 2)
        # A fresh instance can use the new published data without manual sync.
        fresh = KnowledgeStore(storage=self.storage)
        fresh._vector_index = MilvusIndex(client=self.milvus)
        result = fresh.search('鸡肉', embedder=FakeEmbeddings())
        self.assertEqual(result['mode'], 'hybrid')
        self.assertEqual(result['trace']['vector_backend'], 'milvus_hnsw')
        self.assertIsNone(result['warning'])

    def test_milvus_failure_preserves_active_snapshot(self):
        first = self.publish()
        pointer = self.cos.objects['kb/test/current.json']
        self.milvus.fail_insert = True
        (self.docs / 'salt.md').write_text('食盐建议。', encoding='utf-8')
        with self.assertRaises(RuntimeError):
            self.publish()
        self.assertEqual(self.cos.objects['kb/test/current.json'], pointer)
        self.assertEqual(self.store.status()['version'], first['version'])
        self.assertEqual(self.store.status()['documents'], 1)

    def test_failed_pointer_switch_does_not_destroy_old_snapshot(self):
        first = self.publish()
        self.cos.fail_key = 'kb/test/current.json'
        (self.docs / 'salt.md').write_text('食盐建议。', encoding='utf-8')
        with self.assertRaises(RuntimeError):
            self.publish()
        self.assertEqual(self.store.status()['version'], first['version'])

    def test_stage_does_not_activate_or_import_until_explicit_activation(self):
        first = self.publish()
        (self.docs / 'salt.md').write_text('食盐建议。', encoding='utf-8')
        candidate = self.publish(activate=False)
        self.assertEqual(self.store.status()['version'], first['version'])
        self.assertEqual(len(self.milvus.aliases), 1)
        self.store.activate_version(candidate['version'], expected_hash=candidate['sha256'])
        self.assertEqual(self.store.status()['version'], candidate['version'])
        self.assertEqual(len(self.milvus.aliases), 2)

    def test_tampered_candidate_cannot_activate_or_sync(self):
        candidate = self.publish(activate=False)
        key = self.storage._index_key(candidate['version'])
        self.cos.objects[key] = b'corrupt'
        with self.assertRaises(ValueError):
            self.store.activate_version(candidate['version'], expected_hash=candidate['sha256'])
        self.assertFalse(self.milvus.aliases)

    def test_local_sqlite_update_syncs_before_replacing_data(self):
        store = KnowledgeStore(self.root / 'local.sqlite3', self.docs)
        store._vector_index = MilvusIndex(client=self.milvus)
        first = store.rebuild(embedder=FakeEmbeddings())
        self.assertEqual(len(self.milvus.aliases), 1)
        (self.docs / 'salt.md').write_text('食盐建议。', encoding='utf-8')
        self.milvus.fail_insert = True
        with self.assertRaises(RuntimeError):
            store.rebuild(embedder=FakeEmbeddings())
        self.assertEqual(store.status()['built_at'], first['built_at'])
        self.assertEqual(store.status()['documents'], 1)

    def test_lexical_only_update_does_not_require_milvus(self):
        with patch.object(self.store._vector_index, 'sync', side_effect=AssertionError('must not call')):
            status = self.publish(vectors=False)
        self.assertEqual(status['mode'], 'lexical')


class WatcherTests(UpdateFixture, unittest.TestCase):
    # Watcher cases share the same isolated COS/Milvus fixture.
    def setUp(self):
        super().setUp()
        self.first = self.publish()
        self.watcher = KnowledgeAutoUpdater(self.store, self.docs, interval=1, embedder=FakeEmbeddings(), retry_seconds=0)

    def test_unchanged_sources_do_not_rebuild_or_call_embeddings(self):
        with patch.object(self.store, 'rebuild', side_effect=AssertionError('must not rebuild')):
            self.watcher.run_once()
            self.watcher.run_once()
        self.assertEqual(self.watcher.state['phase'], 'up_to_date')
        self.assertEqual(self.store.status()['version'], self.first['version'])

    def test_add_modify_delete_automatically_publish_after_stability(self):
        path = self.docs / 'salt.md'
        path.write_text('食盐建议。', encoding='utf-8')
        self.watcher.run_once()
        self.assertEqual(self.watcher.state['phase'], 'waiting_for_stable_files')
        self.assertEqual(self.store.status()['version'], self.first['version'])
        self.watcher.run_once()
        second = self.store.status()
        self.assertEqual(second['documents'], 2)
        self.assertNotEqual(second['version'], self.first['version'])
        path.write_text('修改后的食盐建议。', encoding='utf-8')
        self.watcher.run_once()
        self.watcher.run_once()
        self.assertNotEqual(self.store.status()['version'], second['version'])
        path.unlink()
        self.watcher.run_once()
        self.watcher.run_once()
        self.assertEqual(self.store.status()['documents'], 1)
        self.assertEqual(self.watcher.state['phase'], 'up_to_date')

    def test_manifest_only_changes_update_source_metadata(self):
        source_file = self.docs.parent / 'sources.json'
        source_file.write_text(json.dumps({'chicken.md': {'title': 'New Title', 'source_url': 'https://example.org/new'}}), encoding='utf-8')
        self.watcher.run_once()
        self.watcher.run_once()
        rows, _ = self.storage.load()
        self.assertEqual(rows[0]['title'], 'New Title')
        self.assertEqual(rows[0]['source_url'], 'https://example.org/new')

    def test_milvus_outage_keeps_old_version_and_retries(self):
        (self.docs / 'salt.md').write_text('食盐建议。', encoding='utf-8')
        self.milvus.fail_insert = True
        self.watcher.run_once()
        self.watcher.run_once()
        self.assertEqual(self.watcher.state['phase'], 'error')
        self.assertEqual(self.store.status()['version'], self.first['version'])
        self.milvus.fail_insert = False
        self.watcher.run_once()
        self.assertEqual(self.watcher.state['phase'], 'up_to_date')
        self.assertEqual(self.store.status()['documents'], 2)

    def test_file_changes_during_build_are_not_activated(self):
        path = self.docs / 'salt.md'
        path.write_text('revision A', encoding='utf-8')
        self.watcher.run_once()
        original = self.milvus.insert
        def insert(**kwargs):
            result = original(**kwargs)
            path.write_text('revision B', encoding='utf-8')
            return result
        self.milvus.insert = insert
        self.watcher.run_once()
        self.assertEqual(self.store.status()['version'], self.first['version'])
        self.assertEqual(self.watcher.state['phase'], 'waiting_for_stable_files')
        self.milvus.insert = original
        self.watcher.run_once()
        self.watcher.run_once()
        rows, _ = self.storage.load()
        self.assertTrue(any('revision B' in row['content'] for row in rows))

    def test_empty_directory_and_invalid_manifest_keep_last_good_version(self):
        (self.docs / 'chicken.md').unlink()
        self.watcher.run_once()
        self.assertEqual(self.watcher.state['phase'], 'error')
        self.assertEqual(self.store.status()['version'], self.first['version'])
        (self.docs / 'chicken.md').write_text('鸡肉菜谱。', encoding='utf-8')
        (self.docs.parent / 'sources.json').write_text('{invalid', encoding='utf-8')
        self.watcher.run_once()
        self.assertEqual(self.store.status()['version'], self.first['version'])

    def test_restart_detects_changes_made_while_service_was_offline(self):
        (self.docs / 'salt.md').write_text('新资料。', encoding='utf-8')
        fresh = KnowledgeAutoUpdater(self.store, self.docs, embedder=FakeEmbeddings())
        fresh.run_once()
        fresh.run_once()
        self.assertEqual(self.store.status()['documents'], 2)

    def test_legacy_snapshot_compares_manifest_without_reembedding(self):
        rows, meta = self.storage.load()
        meta = {key: value for key, value in meta.items() if key != 'source_signature'}
        self.storage.publish(rows, meta, meta['dataset_id'])
        with patch.object(self.store, 'rebuild', side_effect=AssertionError('must not rebuild')):
            self.watcher.run_once()
        self.assertEqual(self.watcher.state['phase'], 'up_to_date')

    def test_shutdown_prevents_late_activation(self):
        (self.docs / 'salt.md').write_text('新资料。', encoding='utf-8')
        self.watcher.run_once()
        original = self.milvus.insert
        def insert(**kwargs):
            result = original(**kwargs)
            self.watcher.stop()
            return result
        self.milvus.insert = insert
        self.watcher.run_once()
        self.assertEqual(self.store.status()['version'], self.first['version'])
