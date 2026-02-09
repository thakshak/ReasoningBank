import unittest
from unittest.mock import MagicMock
import sys

# Create mock modules
mock_prompts = MagicMock()
mock_output_parsers = MagicMock()

# Mocking dependencies
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.prompts"] = mock_prompts
sys.modules["langchain_core.output_parsers"] = mock_output_parsers
sys.modules["langchain_core.language_models"] = MagicMock()
sys.modules["langchain_core.language_models.base"] = MagicMock()
sys.modules["langchain_core.runnables"] = MagicMock()
sys.modules["langchain_community"] = MagicMock()
sys.modules["langchain_community.llms"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["yaml"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.metrics.pairwise"] = MagicMock()

# Mocking the pipe operator for chains
class MockChain:
    def __init__(self, items):
        self.items = items
    def __or__(self, other):
        return MockChain(self.items + [other])
    def invoke(self, inputs):
        return "mock_response"

mock_prompts.PromptTemplate.from_template.return_value = MockChain(["template"])
mock_output_parsers.StrOutputParser.return_value = "parser"

from reasoningbank.core.matts import parallel_scaling, sequential_scaling
from reasoningbank.core.bank import ReasoningBank

class TestPromptInjectionFix(unittest.TestCase):
    def setUp(self):
        self.mock_bank = MagicMock(spec=ReasoningBank)
        self.mock_bank.llm = MagicMock()
        self.mock_bank.retrieve_memories.return_value = []

        self.mock_agent = MagicMock()
        self.mock_agent.invoke.return_value = "mock_trajectory"

        # Reset the mock calls
        mock_prompts.PromptTemplate.from_template.reset_mock()

    def test_parallel_scaling_uses_template(self):
        parallel_scaling("test query", 1, self.mock_bank, self.mock_agent)
        mock_prompts.PromptTemplate.from_template.assert_called()

    def test_sequential_scaling_uses_template(self):
        sequential_scaling("test query", 1, self.mock_bank, self.mock_agent)
        mock_prompts.PromptTemplate.from_template.assert_called()

if __name__ == "__main__":
    unittest.main()
