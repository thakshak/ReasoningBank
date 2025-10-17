import unittest
from unittest.mock import MagicMock, patch
from reasoningbank.core.bank import ReasoningBank
import os
import numpy as np


class TestReasoningBank(unittest.TestCase):

    @patch("reasoningbank.core.bank.load_config")
    def setUp(self, mock_load_config):
        # Mock the configuration
        mock_load_config.return_value = {
            "memory": {"backend": "json", "json": {"filepath": "test_memory.json"}},
            "embedding_model": {
                "model_name": "sentence-transformers",
                "st_model_name": "all-MiniLM-L6-v2",
            },
            "llm": {"provider": "langchain.llms.Fake"},
        }

        # Mock dependencies
        self.mock_memory_backend = MagicMock()
        with patch(
            "reasoningbank.core.bank.JSONMemoryBackend",
            return_value=self.mock_memory_backend,
        ):
            self.bank = ReasoningBank(config_path="dummy_config.yaml")

        self.mock_embedding_model = MagicMock()

        def mock_encode(input_data):
            if isinstance(input_data, list):  # For test_add_experience_success
                return np.array([[0.1, 0.2]])
            elif isinstance(input_data, str):  # For test_retrieve_memories
                return np.array([0.3, 0.4])
            return None

        self.mock_embedding_model.encode.side_effect = mock_encode
        self.bank.embedding_model = self.mock_embedding_model

        self.mock_llm = MagicMock()
        self.bank.llm = self.mock_llm

    def tearDown(self):
        if os.path.exists("test_memory.json"):
            os.remove("test_memory.json")

    def test_add_experience_success(self):
        # Configure the mock LLM to simulate a successful trajectory
        self.mock_llm.invoke.side_effect = [
            "Success",  # First call for judging
            '[{"title": "t1", "description": "d1", "content": "c1"}]',  # Second call for distilling
        ]

        self.bank.add_experience("some trajectory", "some query")

        # Assert that the LLM was called twice
        self.assertEqual(self.mock_llm.invoke.call_count, 2)

        # Assert that the embedding model was called to embed the query
        self.mock_embedding_model.encode.assert_any_call("some query")

        # Assert that the memory backend's add method was called with the correct data
        self.mock_memory_backend.add.assert_called_once()
        args, _ = self.mock_memory_backend.add.call_args
        self.assertEqual(len(args[0]), 1)
        added_item = args[0][0]
        self.assertEqual(added_item["metadata"]["query"], "some query")
        self.assertIn("distilled_items", added_item["metadata"])

    def test_retrieve_memories(self):
        # Configure the mock memory backend to return some data
        self.mock_memory_backend.query.return_value = [{"title": "retrieved_title"}]

        results = self.bank.retrieve_memories("some query", k=1)

        # Assert that the embedding model was called to embed the query
        self.mock_embedding_model.encode.assert_any_call("some query")

        # Assert that the memory backend's query method was called
        self.mock_memory_backend.query.assert_called_once_with([0.3, 0.4], 1)

        # Assert that the results are correct
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "retrieved_title")


if __name__ == "__main__":
    unittest.main()
