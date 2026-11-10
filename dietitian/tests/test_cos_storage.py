from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from qcloud_cos.cos_exception import CosServiceError

from rag.cos_storage import CosKnowledgeStorage, checked_name
from rag.store import KnowledgeStore
from test_rag import FakeEmbeddings


class FakeCos:
    def __init__(self):
        self.objects = {}
        self.acls = {}
        self.reads = []
        self.fail_key = None

    def get_object(self, Bucket, Key):
        self.reads.append(Key)
        if Key not in self.objects:
            raise CosServiceError('GET', {'code': 'NoSuchKey', 'message': 'Not found'}, 404)
        data = self.objects[Key]
        class Body:
            def get_raw_stream(self):
                return BytesIO(data)
        return {'Body': Body()}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if Key == self.fail_key:
            raise RuntimeError('Simulated COS upload error')
        self.objects[Key] = Body
        self.acls[Key] = kwargs['ACL']

    def list_objects(self, Bucket, Prefix, Marker, MaxKeys):
        keys = sorted(k for k in self.objects if k.startswith(Prefix) and k > Marker)
        page = keys[:1]
        return {'Contents': [{'Key': key} for key in page],
                'IsTruncated': 'true' if len(keys) > 1 else 'false',
                'NextMarker': page[-1] if page else ''}


class CosStorageTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        offline = patch.dict('os.environ', {'RAG_QUERY_REWRITE': 'false', 'RAG_RERANK': 'false', 'RAG_VECTOR_BACKEND': 'usearch'})
        offline.start()
        self.addCleanup(offline.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.docs = Path(self.temp.name) / 'documents'
        self.docs.mkdir()
        (self.docs / 'chicken.md').write_text('鸡肉彩椒菜谱。', encoding='utf-8')
        self.client = FakeCos()
        self.storage = CosKnowledgeStorage(client=self.client, bucket='test-123', prefix='knowledge/test')
        self.store = KnowledgeStore(db_path=Path(self.temp.name)/'must-not-exist.sqlite3', storage=self.storage)

    def publish(self):
        dataset = self.storage.stage_directory(self.docs)
        return self.store.rebuild(embedder=FakeEmbeddings(), dataset_id=dataset['dataset_id'])

    def test_cloud_roundtrip_survives_missing_local_files(self):
        status = self.publish()
        (self.docs / 'chicken.md').unlink()
        fresh = CosKnowledgeStorage(client=self.client, bucket='test-123', prefix='knowledge/test')
        independent = KnowledgeStore(storage=fresh)
        result = independent.search('鸡肉', embedder=FakeEmbeddings())
        self.assertEqual(result['hits'][0]['filename'], 'chicken.md')
        self.assertEqual(status['storage'], 'cos')
        self.assertFalse(self.store.db_path.exists())
        self.assertTrue(all(acl == 'private' for acl in self.client.acls.values()))

    def test_failed_pointer_update_preserves_active_version(self):
        first = self.publish()
        self.client.fail_key = 'knowledge/test/current.json'
        with self.assertRaises(RuntimeError):
            self.publish()
        self.assertEqual(self.store.status()['version'], first['version'])

    def test_cross_instance_cache_refresh_and_rollback(self):
        first = self.publish()
        fresh = KnowledgeStore(storage=CosKnowledgeStorage(client=self.client, bucket='test-123', prefix='knowledge/test'))
        self.assertEqual(fresh.status()['version'], first['version'])
        (self.docs / 'salt.md').write_text('食盐摄入建议。', encoding='utf-8')
        second = self.publish()
        self.assertNotEqual(first['version'], second['version'])
        self.assertEqual(fresh.status()['documents'], 2)
        self.storage.activate(first['version'])
        self.assertEqual(fresh.status()['version'], first['version'])
        self.assertEqual(fresh.status()['documents'], 1)

    def test_corrupt_index_is_rejected(self):
        published = self.publish()
        self.client.objects[self.storage._index_key(published['version'])] = b'corrupt'
        fresh = CosKnowledgeStorage(client=self.client, bucket='test-123', prefix='knowledge/test')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            fresh.load()

    def test_unchanged_version_only_downloads_pointer(self):
        self.publish()
        self.store.status()
        self.client.reads.clear()
        self.store.status()
        self.assertEqual(self.client.reads, ['knowledge/test/current.json'])

    def test_cloud_build_reads_and_validates_sources(self):
        first = self.publish()
        key = next(key for key in self.client.objects if '/documents/' in key)
        self.client.objects[key] = b'corrupt'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.store.rebuild(embedder=FakeEmbeddings())
        self.assertEqual(self.store.status()['version'], first['version'])

    def test_missing_index_returns_empty_status(self):
        self.assertFalse(self.store.status()['ready'])

    def test_paths_cannot_escape_dataset(self):
        for name in ('../a.md', '/a.md', 'a/../../b.txt', 'a\\b.md'):
            with self.assertRaises(ValueError):
                checked_name(name)
        with self.assertRaises(ValueError):
            self.storage.activate('../current')

    def test_versions_handles_paginated_listing(self):
        first, second = self.publish(), self.publish()
        self.assertEqual({v['version'] for v in self.storage.versions()},
                         {first['version'], second['version']})

    def test_access_denied_is_not_treated_as_empty_knowledge(self):
        from unittest.mock import patch
        denied = CosServiceError('GET', {'code': 'AccessDenied', 'message': 'Denied'}, 403)
        with patch.object(self.client, 'get_object', side_effect=denied):
            with self.assertRaises(CosServiceError):
                self.store.status()

    def test_region_endpoint_url_is_rejected_before_network_access(self):
        from unittest.mock import patch
        with patch.dict('os.environ', {'COS_SECRET_ID': 'dummy', 'COS_SECRET_KEY': 'dummy',
                                      'COS_BUCKET': 'test-123', 'COS_REGION': 'test-123.cos.ap-nanjing.myqcloud.com'}):
            with self.assertRaisesRegex(ValueError, 'region code'):
                CosKnowledgeStorage().client

    def test_staging_does_not_switch_current_version(self):
        first = self.publish()
        (self.docs / 'salt.md').write_text('食盐摄入建议。', encoding='utf-8')
        dataset = self.storage.stage_directory(self.docs)
        candidate = self.store.rebuild(embedder=FakeEmbeddings(), dataset_id=dataset['dataset_id'], activate=False)
        self.assertFalse(candidate['activated'])
        self.assertEqual(candidate['documents'], 2)
        self.assertEqual(self.store.status()['version'], first['version'])
        self.storage.activate(candidate['version'], expected_hash=candidate['sha256'])
        self.assertEqual(self.store.status()['documents'], 2)

    def test_large_index_compressed_roundtrip(self):
        self.publish()
        rows, meta = self.storage.load()
        rows = [dict(row) for row in rows]
        rows[0]['content'] = 'test content ' * 100_000
        pointer = self.storage.publish(rows, meta, meta['dataset_id'])
        self.assertTrue(self.client.objects[pointer['index_key']].startswith(b'\x1f\x8b'))
        self.assertEqual(self.storage.load()[0][0]['content'], rows[0]['content'])


if __name__ == '__main__':
    unittest.main()
