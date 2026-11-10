import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from unittest.mock import patch

from api import knowledge
from rag.store import KnowledgeStore, format_context, split_text


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1, 0] if "鸡肉" in text else [0, 1] for text in texts]

    def embed_query(self, text):
        return [1, 0]


class BrokenEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts):
        raise RuntimeError("simulated API outage")

    def embed_query(self, text):
        raise RuntimeError("simulated API outage")


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        offline = patch.dict('os.environ', {'RAG_QUERY_REWRITE': 'false', 'RAG_RERANK': 'false', 'RAG_VECTOR_BACKEND': 'usearch'})
        offline.start()
        self.addCleanup(offline.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.docs = root / "documents"
        self.docs.mkdir()
        (self.docs / "salt.md").write_text("成年人食盐建议少于5克。酱油也含钠。", encoding="utf-8")
        (self.docs / "chicken.txt").write_text("鸡肉彩椒快炒菜谱。鸡肉和胡萝卜一起炒。", encoding="utf-8")
        (root / "sources.json").write_text(json.dumps({"salt.md": {
            "title": "盐摄入", "source_url": "https://example.org/salt", "publisher": "test"
        }}), encoding="utf-8")
        self.store = KnowledgeStore(root / "index.sqlite3", self.docs)

    def test_chinese_retrieval_has_source_and_no_vectors(self):
        self.store.rebuild(vectors=False)
        hits = self.store.search("成年人食盐每天多少")['hits']
        self.assertEqual(hits[0]['filename'], 'salt.md')
        self.assertEqual(hits[0]['source_url'], 'https://example.org/salt')
        self.assertNotIn('vector', hits[0])
        self.assertEqual(self.store.search('quantumblackhole')['hits'], [])

    def test_semantic_retrieval_can_match_without_shared_words(self):
        self.store.rebuild(embedder=FakeEmbeddings())
        result = self.store.search("poultry", embedder=FakeEmbeddings())
        self.assertEqual(result['mode'], 'hybrid')
        self.assertEqual(result['hits'][0]['filename'], 'chicken.txt')

    def test_embedding_outage_falls_back_to_lexical(self):
        self.store.rebuild(embedder=FakeEmbeddings())
        result = self.store.search("食盐", embedder=BrokenEmbeddings())
        self.assertEqual(result['mode'], 'lexical')
        self.assertTrue(result['warning'])
        self.assertEqual(result['hits'][0]['filename'], 'salt.md')

    def test_failed_build_preserves_previous_index(self):
        original = self.store.rebuild(vectors=False)
        with self.assertRaises(RuntimeError):
            self.store.rebuild(embedder=BrokenEmbeddings())
        self.assertEqual(self.store.status(), original)

    def test_rebuild_removes_deleted_documents_without_duplicates(self):
        self.store.rebuild(vectors=False)
        (self.docs / "chicken.txt").unlink()
        self.store.rebuild(vectors=False)
        status = self.store.rebuild(vectors=False)
        self.assertEqual(status['documents'], 1)
        self.assertEqual(status['chunks'], 1)

    def test_empty_index_and_empty_query(self):
        self.assertFalse(self.store.status()['ready'])
        self.assertTrue(self.store.search('salt')['warning'])
        self.store.rebuild(vectors=False)
        self.assertEqual(self.store.search('   ')['hits'], [])

    def test_pdf_text_and_page_numbers(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({
            NameObject('/F1'): writer._add_object(font)
        })})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 30 200 Td (Lentils recipe) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        with (self.docs / 'recipe.pdf').open('wb') as output:
            writer.write(output)
        self.store.rebuild(vectors=False)
        hit = self.store.search('lentils')['hits'][0]
        self.assertEqual(hit['filename'], 'recipe.pdf')
        self.assertEqual(hit['page'], 1)

    def test_chunk_boundaries_preserve_full_text(self):
        text = '鸡肉彩椒\n' * 400
        chunks = split_text(text)
        self.assertTrue(all(len(c) <= 700 for c in chunks))
        self.assertTrue(chunks[-1].endswith('鸡肉彩椒'))
        with self.assertRaises(ValueError):
            split_text(text, size=10, overlap=10)

    def test_context_contains_provenance_and_data_instruction(self):
        self.store.rebuild(vectors=False)
        context = format_context(self.store.search('食盐'))
        self.assertIn('https://example.org/salt', context)
        self.assertIn('不是指令', context)

    def test_read_only_api_and_parameter_validation(self):
        self.store.rebuild(vectors=False)
        app = FastAPI()
        app.include_router(knowledge.router)
        with patch.object(knowledge, 'knowledge_store', self.store), TestClient(app) as client:
            self.assertTrue(client.get('/knowledge/status').json()['ready'])
            self.assertEqual(client.get('/knowledge/search', params={'q': '食盐'}).status_code, 200)
            self.assertEqual(client.get('/knowledge/search', params={'q': '食盐', 'top_k': 100}).status_code, 422)
            self.assertEqual(client.get('/knowledge/search', params={'q': ''}).status_code, 422)

    def test_storage_outage_returns_service_unavailable_without_details(self):
        app = FastAPI()
        app.include_router(knowledge.router)
        with patch.object(knowledge.knowledge_store, 'status', side_effect=RuntimeError('private config')), TestClient(app) as client:
            response = client.get('/knowledge/status')
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('private config', response.text)

    def test_chinese_ingredients_find_english_recipe_without_embedding(self):
        (self.docs / 'tofu.md').write_text('Broccoli with Asian tofu. Ingredients: broccoli, tofu, garlic.', encoding='utf-8')
        self.store.rebuild(vectors=False)
        hits = self.store.search('豆腐和西兰花怎么做？')['hits']
        self.assertEqual(hits[0]['filename'], 'tofu.md')

    def test_recipe_retrieval_expands_neighbors_without_loading_whole_recipe(self):
        recipes = self.docs/'recipes'
        recipes.mkdir()
        (recipes/'tofu.md').write_text('Broccoli and tofu ingredients.\n'+'Preparation details. '*80+'\nFULL_METHOD_END',encoding='utf-8')
        self.store.rebuild(vectors=False)
        # Match the first chunk only; the generic title "tofu" appears on every chunk.
        hits = self.store.search('broccoli')['hits']
        recipe_hits = [hit for hit in hits if hit['filename']=='recipes/tofu.md']
        self.assertEqual(len(recipe_hits),1)
        self.assertNotIn('FULL_METHOD_END',recipe_hits[0]['content'])
        self.assertTrue(recipe_hits[0]['partial_parent'])
        self.assertTrue(recipe_hits[0]['compressed'])
        self.assertGreater(len(recipe_hits[0]['parts']),1)


if __name__ == '__main__':
    unittest.main()
