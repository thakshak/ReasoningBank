import unittest
import os
import json
from reasoningbank.memory import ChromaMemoryBackend, JSONMemoryBackend

class TestJSONMemoryBackend(unittest.TestCase):

    def setUp(self):
        self.filepath = "test_memory.json"
        self.backend = JSONMemoryBackend(self.filepath)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_add_and_query(self):
        items = [
            {"embedding": [0.1, 0.2], "metadata": {"title": "t1", "description": "d1", "content": "c1"}, "document": "c1"},
            {"embedding": [0.3, 0.4], "metadata": {"title": "t2", "description": "d2", "content": "c2"}, "document": "c2"},
        ]
        self.backend.add(items)

        # Test that the data was saved to the file
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["metadata"]["title"], "t1")

        # Test the query method
        results = self.backend.query([0.5, 0.6], k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "t2")

class TestChromaMemoryBackend(unittest.TestCase):

    def setUp(self):
        self.backend = ChromaMemoryBackend(collection_name="test_collection")

    def tearDown(self):
        # Delete the test collection to ensure test isolation
        self.backend.client.delete_collection(name="test_collection")

    def test_add_and_query(self):
        items = [
            {"embedding": [0.1, 0.2, 0.3], "metadata": {"title": "t1", "description": "d1", "content": "c1"}, "document": "c1"},
            {"embedding": [0.7, 0.8, 0.9], "metadata": {"title": "t2", "description": "d2", "content": "c2"}, "document": "c2"},
        ]
        self.backend.add(items)

        # The query should return the item with the embedding closest to the query embedding.
        # In this case, [0.6, 0.7, 0.8] is closer to [0.7, 0.8, 0.9] than to [0.1, 0.2, 0.3].
        results = self.backend.query([0.6, 0.7, 0.8], k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "t2")

if __name__ == '__main__':
    unittest.main()
