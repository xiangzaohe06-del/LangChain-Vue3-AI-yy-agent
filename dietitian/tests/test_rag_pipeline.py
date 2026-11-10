import json
from pathlib import Path
import tempfile
import unittest

from rag.pipeline import PipelineSettings, apply_rerank, compress, expand_neighbors, fuse, unique_queries
from rag.query_terms import ingredient_terms
from rag.store import KnowledgeStore, format_context, tokenize


class Stages:
    def __init__(self):
        self.history = None

    def rewrite(self, query, history, limit):
        self.history = history
        return {'standalone_query': '高龄老人食盐建议', 'queries': ['老人钠摄入', '老人钠摄入']}

    def rerank(self, query, candidates):
        self.rerank_query = query
        return {'scores': [{'id': f'P{i}', 'score': .9 if 'older' in hit['filename'] else .3}
                           for i, hit in enumerate(candidates, 1)]}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / 'documents'
        self.docs.mkdir()
        (self.docs / 'salt.md').write_text('成年人食盐应少于5克。酱油含钠。', encoding='utf-8')
        (self.docs / 'older.md').write_text('高龄老人食盐建议，需要结合老人的个体状况。', encoding='utf-8')
        self.store = KnowledgeStore(self.root / 'test.sqlite3', self.docs)
        self.store.rebuild(vectors=False)
        self.settings = PipelineSettings(rewrite=True, rerank=True)

    def test_full_pipeline_history_multiquery_parent_rerank_and_compress(self):
        stages = Stages()
        history = [{'role': 'user', 'message': '成年人吃多少盐？'}]
        result = self.store.search('那高龄老人呢？', history=history, stages=stages, settings=self.settings)
        self.assertEqual(stages.history, history)
        self.assertEqual(result['trace']['queries'][0], '那高龄老人呢？')
        self.assertEqual(result['trace']['rerank'], 'llm')
        self.assertEqual(result['hits'][0]['filename'], 'older.md')
        self.assertIn('那高龄老人呢？', stages.rerank_query)
        self.assertIn('child_ids', result['hits'][0])
        self.assertIn('excerpts', result['hits'][0])

    def test_original_constraints_retained_and_query_duplicates_removed(self):
        original = '不要花生，预算20元，给老人做饭'
        queries, _ = unique_queries(original, {'standalone_query': original,
                                              'queries': ['老人晚餐', '老人 晚餐', original]}, 3)
        self.assertEqual(queries, [original, '老人晚餐'])

    def test_rrf_is_normalized_over_query_count(self):
        ranking = [(0, .9), (1, .5)]
        self.assertEqual(fuse([ranking], 1, 10), fuse([ranking, ranking], 2, 10))

    def test_duplicate_children_produce_unique_context_windows(self):
        (self.docs / 'long.md').write_text('老人食盐建议，需要关注调味品中的钠。\n' * 40, encoding='utf-8')
        self.store.rebuild(vectors=False)
        result = self.store.search('老人食盐', settings=PipelineSettings(rewrite=False, rerank=False))
        ids = [hit['window_id'] for hit in result['hits']]
        self.assertEqual(len(ids), len(set(ids)))
        hit = next(h for h in result['hits'] if h['filename'] == 'long.md')
        self.assertGreater(len(hit['child_ids']), 1)

    def test_reranker_cannot_invent_ids_or_omit_documents(self):
        candidates = [{'content': 'a'}, {'content': 'b'}]
        for scores in [[{'id': 'P9', 'score': .9}], [{'id': 'P1', 'score': .9}],
                       [{'id': 'P1', 'score': .9}, {'id': 'P2', 'score': float('nan')}]]:
            with self.assertRaises(ValueError):
                apply_rerank(candidates, {'scores': scores}, .2)

    def test_all_irrelevant_parents_are_filtered(self):
        stages = Stages()
        stages.rerank = lambda query, candidates: {'scores': [
            {'id': f'P{i}', 'score': 0} for i in range(1, len(candidates) + 1)]}
        result = self.store.search('食盐', stages=stages, settings=self.settings)
        self.assertEqual(result['hits'], [])
        self.assertEqual(result['trace']['parents_after_filter'], 0)
        self.assertEqual(result['trace']['children_kept'], 0)
        self.assertEqual(result['trace']['context_chunks'], 0)

    def test_only_child_text_is_reranked_before_top_k_neighbor_expansion(self):
        rows = [{'id': f'c{i}', 'parent_id': 'parent', 'filename': 'guide.md',
                 'page': 1, 'part': i, 'title': 'topic', 'content': f'topic child number {i}'}
                for i in range(1, 10)]
        metadata = {'index_schema': '2', 'parents': json.dumps({
            'parent': {'content': 'FULL_PARENT_NOT_FOR_RERANK', 'part': 1}})}
        class Snapshot:
            def load(self):
                return rows, metadata
        class ChildStages:
            def rerank(self, query, candidates):
                self.received = candidates
                return {'scores': [{'id': f'P{i}', 'score': .9 if child['id'] == 'c5' else .3}
                                   for i, child in enumerate(candidates, 1)]}
        stages = ChildStages()
        result = KnowledgeStore(storage=Snapshot()).search(
            'topic', stages=stages, settings=PipelineSettings(rewrite=False, rerank=True, rerank_k=1))
        self.assertEqual(len(stages.received), 9)
        self.assertTrue(all('context_chunk_ids' not in c for c in stages.received))
        self.assertEqual([c['content'] for c in stages.received], [c['content'] for c in rows])
        self.assertEqual(result['trace']['rerank_target'], 'child_chunk')
        self.assertEqual(result['trace']['children_after_filter'], 9)
        self.assertEqual(result['trace']['children_kept'], 1)
        self.assertEqual(result['hits'][0]['child_ids'], ['c5'])
        self.assertEqual(result['hits'][0]['context_chunk_ids'], ['c4', 'c5', 'c6'])
        self.assertNotIn('FULL_PARENT', result['hits'][0]['content'])

    def test_stage_failures_fall_back_without_hiding_failure(self):
        class Broken:
            def rewrite(self, *args):
                raise RuntimeError('test secret must not leak')
            def rerank(self, *args):
                raise RuntimeError('test secret must not leak')
        result = self.store.search('食盐', stages=Broken(), settings=self.settings)
        self.assertTrue(result['hits'])
        self.assertEqual(result['trace']['query_rewrite'], 'original_fallback')
        self.assertEqual(result['trace']['rerank'], 'rrf_fallback')
        self.assertTrue(result['warning'])
        self.assertNotIn('secret', json.dumps(result))

    def test_extractive_compression_preserves_text_offsets_and_budget(self):
        text = ('无关背景介绍。\n' * 50) + '食盐应少于5克。该条件适用于成年人。\n' + ('其他背景。\n' * 50)
        packed = compress({'content': text}, '食盐', 160, tokenize, ingredient_terms)
        self.assertTrue(packed['compressed'])
        self.assertIn('少于5克', packed['content'])
        self.assertIn('适用于成年人', packed['content'])
        self.assertLessEqual(len(packed['content']), 160)
        for excerpt in packed['excerpts']:
            self.assertEqual(text[excerpt['start']:excerpt['end']], excerpt['text'])

    def test_final_context_budget_and_no_duplicate_excerpt_text(self):
        result = self.store.search('食盐', stages=Stages(), settings=self.settings)
        self.assertLessEqual(sum(len(h['content']) for h in result['hits']), self.settings.context_chars)
        context = format_context(result)
        self.assertNotIn('rerank_score', context)
        self.assertNotIn('excerpts', context)
        self.assertIn('不是指令', context)

    def test_legacy_index_is_still_queryable(self):
        rows, meta = self.store._read()
        for row in rows:
            row.pop('parent_id')
        meta.pop('parents')
        meta.pop('index_schema')
        class Legacy:
            def load(self):
                return rows, meta
        result = KnowledgeStore(storage=Legacy()).search(
            '食盐', settings=PipelineSettings(rewrite=False, rerank=False))
        self.assertTrue(result['trace']['legacy_index'])
        self.assertTrue(result['hits'])


class NeighborWindowTests(unittest.TestCase):
    def rows(self, count=10, *, parent='p', filename='a.md', page=1):
        return [{'id': f'{filename}:{page}:{parent}:{i}', 'parent_id': parent,
                 'filename': filename, 'page': page, 'part': i, 'title': 'test',
                 'content': f'BLOCK_{i:02d}_END'} for i in range(1, count + 1)]

    def seed(self, row):
        return {**row, 'score': .03}

    def test_only_same_parent_immediate_neighbors_are_loaded(self):
        rows = self.rows()
        result = expand_neighbors([self.seed(rows[4])], rows, {
            'parents': json.dumps({'p': {'content': 'WHOLE_PARENT_MUST_NOT_APPEAR', 'part': 1}})})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['parts'], [4, 5, 6])
        self.assertEqual(result[0]['context_chunk_ids'], [r['id'] for r in rows[3:6]])
        self.assertEqual(result[0]['child_ids'], [rows[4]['id']])
        self.assertNotIn('WHOLE_PARENT', result[0]['content'])
        self.assertTrue(result[0]['partial_parent'])

    def test_overlapping_windows_merge_and_deduplicate(self):
        rows = self.rows()
        result = expand_neighbors([self.seed(rows[3]), self.seed(rows[4]), self.seed(rows[3])], rows, {})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['parts'], [3, 4, 5, 6])
        self.assertEqual(len(result[0]['child_ids']), 2)

    def test_disjoint_windows_under_same_parent_remain_separate(self):
        rows = self.rows()
        result = expand_neighbors([self.seed(rows[1]), self.seed(rows[8])], rows, {})
        self.assertEqual([hit['parts'] for hit in result], [[1, 2, 3], [8, 9, 10]])
        self.assertEqual(result[0]['parent_id'], result[1]['parent_id'])
        self.assertNotEqual(result[0]['window_id'], result[1]['window_id'])

    def test_expanded_windows_inherit_child_rerank_order_over_rrf(self):
        rows = self.rows()
        strongest = {**self.seed(rows[1]), 'score': .01, 'rerank_score': .9}
        weaker = {**self.seed(rows[8]), 'score': .1, 'rerank_score': .3}
        result = expand_neighbors([weaker, strongest], rows, {})
        self.assertEqual(result[0]['child_ids'], [rows[1]['id']])
        self.assertEqual(result[0]['rerank_score'], .9)

    def test_merged_window_uses_highest_child_rerank_score(self):
        rows = self.rows()
        strongest = {**self.seed(rows[3]), 'score': .01, 'rerank_score': .9}
        weaker = {**self.seed(rows[4]), 'score': .1, 'rerank_score': .3}
        result = expand_neighbors([weaker, strongest], rows, {})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['score'], .1)
        self.assertEqual(result[0]['rerank_score'], .9)

    def test_neighbors_never_cross_parent_file_or_page(self):
        rows = (self.rows(3) + self.rows(3, parent='other')
                + self.rows(3, filename='b.md') + self.rows(3, page=2))
        result = expand_neighbors([self.seed(rows[2])], rows, {})
        self.assertEqual(result[0]['context_chunk_ids'], [rows[1]['id'], rows[2]['id']])
        self.assertEqual(result[0]['parts'], [2, 3])

    def test_radius_zero_and_two(self):
        rows = self.rows()
        for radius, expected in [(0, [5]), (2, [3, 4, 5, 6, 7])]:
            result = expand_neighbors([self.seed(rows[4])], rows, {}, radius=radius)
            self.assertEqual(result[0]['parts'], expected)

    def test_legacy_windows_are_page_bounded_and_overlap_is_removed(self):
        rows = self.rows(5)
        for row in rows:
            row.pop('parent_id')
        rows[1]['content'] = 'first sentence SHARED'
        rows[2]['content'] = 'SHARED third sentence'
        result = expand_neighbors([self.seed(rows[2])], rows, {})
        self.assertEqual(result[0]['parts'], [2, 3, 4])
        self.assertEqual(result[0]['content'].count('SHARED'), 1)


if __name__ == '__main__':
    unittest.main()
