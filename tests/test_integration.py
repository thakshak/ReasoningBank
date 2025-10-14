import unittest
from unittest.mock import MagicMock, patch
from reasoningbank import ReasoningBank
from reasoningbank.integrations.langchain.memory import ReasoningBankMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.outputs import LLMResult, Generation
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
            "embedding_model": {"model_name": "all-MiniLM-L6-v2"},
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
        self.assertEqual(retrieved_memories[0]["title"], "Test Title")
        self.assertEqual(retrieved_memories[0]["content"], "This is a test.")


class TestLangChainIntegration(unittest.TestCase):

    @patch("reasoningbank.core.bank.load_config")
    def setUp(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {
                "backend": "chroma",
                "chroma": {"collection_name": "langchain_test_collection"},
            },
            "embedding_model": {"model_name": "all-MiniLM-L6-v2"},
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
            "Success",
            json.dumps(
                [
                    {
                        "title": "Another Title",
                        "description": "Another Description",
                        "content": "Another content.",
                    }
                ]
            ),
        ]
        generation = Generation(text="Final answer based on memory")
        llm_result = LLMResult(generations=[[generation]])
        self.mock_llm.generate_prompt.return_value = llm_result

        self.memory.save_context(
            {"input": "langchain query"}, {"output": "langchain trajectory"}
        )

        template = (
            "Based on this memory: {history}\\nAnswer the question: {input}"
        )
        prompt = PromptTemplate(
            input_variables=["history", "input"], template=template
        )
        chain = LLMChain(llm=self.mock_llm, prompt=prompt, memory=self.memory)

        result = chain.invoke({"input": "another langchain query"})

        self.assertIn(
            "Chain Title",
            self.memory.load_memory_variables(
                {"input": "another langchain query"}
            )["history"],
        )
        self.assertEqual(result["text"], "Final answer based on memory")


class TestMaTTSIntegration(unittest.TestCase):

    @patch("reasoningbank.core.bank.load_config")
    def setUp(self, mock_load_config):
        mock_load_config.return_value = {
            "memory": {
                "backend": "chroma",
                "chroma": {"collection_name": "matts_test_collection"},
            },
            "embedding_model": {"model_name": "all-MiniLM-L6-v2"},
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
            "Success",
            "[]",
            "Success",
            "[]",
            "synthesized answer",
        ]
        self.mock_llm.generate_prompt.side_effect = [
            LLMResult(generations=[[Generation(text="trajectory 1")]]),
            LLMResult(generations=[[Generation(text="trajectory 2")]]),
        ]

        final_answer = parallel_scaling(
            "test query", k, self.bank, self.agent_executor
        )
        self.assertEqual(final_answer, "synthesized answer")

    def test_sequential_scaling(self):
        k = 2
        self.mock_llm.invoke.side_effect = ["Success", "[]"]
        self.mock_llm.generate_prompt.side_effect = [
            LLMResult(generations=[[Generation(text="refined trajectory 1")]]),
            LLMResult(generations=[[Generation(text="refined trajectory 2")]]),
        ]
        final_trajectory = sequential_scaling(
            "test query", k, self.bank, self.agent_executor
        )
        self.assertEqual(final_trajectory, "refined trajectory 2")


if __name__ == "__main__":
    unittest.main()
