import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock all external dependencies
external_deps = [
    "yaml",
    "sentence_transformers",
    "langchain_community",
    "langchain_community.llms",
    "langchain_core",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
    "langchain_core.runnables",
    "langchain_core.language_models",
    "langchain_core.language_models.base",
    "chromadb",
    "numpy",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "posthog"
]

for dep in external_deps:
    sys.modules[dep] = MagicMock()

from reasoningbank.core.bank import ReasoningBank

class TestReasoningBankConfig(unittest.TestCase):

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_memory_section(self, mock_load_config):
        mock_load_config.return_value = {}
        with self.assertRaisesRegex(ValueError, "Missing 'memory' section"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_memory_backend(self, mock_load_config):
        mock_load_config.return_value = {"memory": {}}
        with self.assertRaisesRegex(ValueError, "Missing 'backend' key"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_unknown_memory_backend(self, mock_load_config):
        mock_load_config.return_value = {"memory": {"backend": "invalid"}}
        with self.assertRaisesRegex(ValueError, "Unknown memory backend type"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_chroma_config(self, mock_load_config):
        mock_load_config.return_value = {"memory": {"backend": "chroma"}}
        with self.assertRaisesRegex(ValueError, "Missing 'collection_name' for 'chroma' backend"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_json_config(self, mock_load_config):
        mock_load_config.return_value = {"memory": {"backend": "json"}}
        with self.assertRaisesRegex(ValueError, "Missing 'filepath' for 'json' backend"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_embedding_model_section(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {"backend": "json", "json": {"filepath": "test.json"}}
        }
        with self.assertRaisesRegex(ValueError, "Missing 'embedding_model' section"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_embedding_model_name(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {"backend": "json", "json": {"filepath": "test.json"}},
            "embedding_model": {}
        }
        with self.assertRaisesRegex(ValueError, "Missing 'model_name' key"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_unknown_embedding_model(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {"backend": "json", "json": {"filepath": "test.json"}},
            "embedding_model": {"model_name": "invalid"}
        }
        with self.assertRaisesRegex(ValueError, "Unknown embedding model"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_llm_section(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {"backend": "json", "json": {"filepath": "test.json"}},
            "embedding_model": {"model_name": "sentence-transformers"}
        }
        with self.assertRaisesRegex(ValueError, "Missing 'llm' section"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_llm_provider(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {"backend": "json", "json": {"filepath": "test.json"}},
            "embedding_model": {"model_name": "sentence-transformers"},
            "llm": {}
        }
        with self.assertRaisesRegex(ValueError, "Missing 'provider' key"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_unknown_llm_provider(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {"backend": "json", "json": {"filepath": "test.json"}},
            "embedding_model": {"model_name": "sentence-transformers"},
            "llm": {"provider": "invalid"}
        }
        with self.assertRaisesRegex(ValueError, "Unknown LLM provider"):
            ReasoningBank()

    @patch("reasoningbank.core.bank.load_config")
    def test_missing_ollama_model(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {"backend": "json", "json": {"filepath": "test.json"}},
            "embedding_model": {"model_name": "sentence-transformers"},
            "llm": {"provider": "ollama"}
        }
        with self.assertRaisesRegex(ValueError, "Missing 'model' key for 'ollama' provider"):
            ReasoningBank()

if __name__ == "__main__":
    unittest.main()
