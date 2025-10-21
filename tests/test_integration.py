import unittest
from unittest.mock import MagicMock, patch
from reasoningbank import ReasoningBank
from reasoningbank.integrations.langchain.memory import ReasoningBankMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.outputs import LLMResult, Generation
from langchain_core.output_parsers import StrOutputParser
import json
from reasoningbank.core.matts import parallel_scaling, sequential_scaling
from reasoningbank.core.agent import create_agent_executor


class TestReasoningBankIntegration(unittest.TestCase):

    @patch("reasoningbank.core.bank.load_config")
    def setUp(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {
                "backend": "chroma",
                "chroma": {"collection_name": "integration_test_collection"},
            },
            "embedding_model": {
                "model_name": "sentence-transformers",
                "st_model_name": "all-MiniLM-L6-v2",
            },
            "llm": {"provider": "langchain.llms.Fake"},
        }
        self.bank = ReasoningBank(config_path="dummy_config.yaml")

        # Overwrite the LLM with a mock for testing
        self.mock_llm = MagicMock()
        self.bank.llm = self.mock_llm

    def tearDown(self):
        self.bank.memory_backend.client.delete_collection(
            name="integration_test_collection"
        )

    def test_add_and_retrieve(self):
        # Configure the mock LLM
        self.mock_llm.invoke.side_effect = [
            "Success",
            json.dumps(
                [
                    {
                        "title": "Test Title",
                        "description": "Test Description",
                        "content": "This is a test.",
                    }
                ]
            ),
        ]

        # Add an experience
        self.bank.add_experience("trajectory", "query")

        # Retrieve memories
        retrieved_memories = self.bank.retrieve_memories(
            "a similar query", k=1
        )

        self.assertEqual(len(retrieved_memories), 1)
        self.assertEqual(retrieved_memories[0]["query"], "query")
        distilled_items = json.loads(retrieved_memories[0]["distilled_items"])
        self.assertEqual(distilled_items[0]["title"], "Test Title")


class TestLangChainIntegration(unittest.TestCase):

    @patch("reasoningbank.core.bank.load_config")
    def setUp(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {
                "backend": "chroma",
                "chroma": {"collection_name": "langchain_test_collection"},
            },
            "embedding_model": {
                "model_name": "sentence-transformers",
                "st_model_name": "all-MiniLM-L6-v2",
            },
            "llm": {"provider": "langchain.llms.Fake"},
        }
        self.bank = ReasoningBank(config_path="dummy_config.yaml")

        self.mock_llm = MagicMock(spec=BaseLanguageModel)
        self.bank.llm = self.mock_llm

        self.memory = ReasoningBankMemory(reasoning_bank=self.bank)

    def tearDown(self):
        self.bank.memory_backend.client.delete_collection(
            name="langchain_test_collection"
        )

    def test_chain_with_memory(self):
        self.mock_llm.invoke.side_effect = [
            "Success",
            json.dumps(
                [
                    {
                        "title": "Chain Title",
                        "description": "Chain Description",
                        "content": "Chain content.",
                    }
                ]
            ),
            "Final answer based on memory",
        ]

        self.memory.save_context(
            {"input": "langchain query"}, {"output": "langchain trajectory"}
        )

        template = (
            "Based on this memory: {history}\\nAnswer the question: {input}"
        )
        prompt = PromptTemplate(
            input_variables=["history", "input"], template=template
        )

        chain = prompt | self.mock_llm | StrOutputParser()

        result = chain.invoke({"input": "another langchain query", "history": self.memory.load_memory_variables({"input": "another langchain query"})["history"]})

        # Check that the memory was loaded and included in the prompt to the LLM
        self.assertIn("Chain Title", self.mock_llm.invoke.call_args[0][0].to_string())
        self.assertEqual(result, "Final answer based on memory")


class TestMaTTSIntegration(unittest.TestCase):

    @patch("reasoningbank.core.bank.load_config")
    def setUp(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {
                "backend": "chroma",
                "chroma": {"collection_name": "matts_test_collection"},
            },
            "embedding_model": {
                "model_name": "sentence-transformers",
                "st_model_name": "all-MiniLM-L6-v2",
            },
            "llm": {"provider": "langchain.llms.Fake"},
        }
        self.bank = ReasoningBank(config_path="dummy_config.yaml")
        self.mock_llm = MagicMock(spec=BaseLanguageModel)
        self.bank.llm = self.mock_llm
        self.agent_executor = create_agent_executor(self.mock_llm)

    def tearDown(self):
        self.bank.memory_backend.client.delete_collection(
            name="matts_test_collection"
        )

    def test_parallel_scaling(self):
        k = 2
        self.mock_llm.invoke.side_effect = [
            "trajectory 1",
            "Success",
            "[]",
            "trajectory 2",
            "Success",
            "[]",
            "synthesized answer",
        ]

        final_answer = parallel_scaling(
            "test query", k, self.bank, self.agent_executor
        )
        self.assertEqual(final_answer, "synthesized answer")

    def test_sequential_scaling(self):
        k = 2
        self.mock_llm.invoke.side_effect = [
            "refined trajectory 1",
            "refined trajectory 2",
            "Success",
            "[]",
        ]
        final_trajectory = sequential_scaling(
            "test query", k, self.bank, self.agent_executor
        )
        self.assertEqual(final_trajectory, "refined trajectory 2")


if __name__ == "__main__":
    unittest.main()
