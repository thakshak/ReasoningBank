import unittest
from unittest.mock import MagicMock
from reasoningbank.bank import ReasoningBank
from reasoningbank.memory import ChromaMemoryBackend
from sentence_transformers import SentenceTransformer
from reasoningbank.integrations.langchain.memory import ReasoningBankMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.outputs import LLMResult, Generation

class TestReasoningBankIntegration(unittest.TestCase):

    def setUp(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory_backend = ChromaMemoryBackend(collection_name="integration_test_collection")
        self.mock_llm = MagicMock()

        self.bank = ReasoningBank(
            memory_backend=self.memory_backend,
            embedding_model=self.embedding_model,
            llm=self.mock_llm
        )

    def tearDown(self):
        self.memory_backend.client.delete_collection(name="integration_test_collection")

    def test_add_and_retrieve(self):
        # Configure the mock LLM
        self.mock_llm.invoke.side_effect = [
            "Success",
            '[{"title": "Test Title", "description": "Test Description", "content": "This is a test."}]'
        ]

        # Add an experience
        self.bank.add_experience("trajectory", "query")

        # Retrieve memories
        retrieved_memories = self.bank.retrieve_memories("a similar query", k=1)

        self.assertEqual(len(retrieved_memories), 1)
        self.assertEqual(retrieved_memories[0]['title'], 'Test Title')
        self.assertEqual(retrieved_memories[0]['content'], 'This is a test.')

class TestLangChainIntegration(unittest.TestCase):

    def setUp(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory_backend = ChromaMemoryBackend(collection_name="langchain_test_collection")
        self.mock_llm = MagicMock(spec=BaseLanguageModel)

        self.bank = ReasoningBank(
            memory_backend=self.memory_backend,
            embedding_model=self.embedding_model,
            llm=self.mock_llm
        )

        self.memory = ReasoningBankMemory(reasoning_bank=self.bank)

    def tearDown(self):
        self.memory_backend.client.delete_collection(name="langchain_test_collection")

    def test_chain_with_memory(self):
        # Configure the mock LLM for the different calls
        # The `add_experience` method calls `invoke`
        self.mock_llm.invoke.side_effect = [
            # For the first save_context call
            "Success",
            '[{"title": "Chain Title", "description": "Chain Description", "content": "Chain content."}]',
            # For the second save_context call (inside the chain)
            "Success",
            '[{"title": "Another Title", "description": "Another Description", "content": "Another content."}]',
        ]
        # The `LLMChain` calls `generate_prompt` and expects an LLMResult
        generation = Generation(text="Final answer based on memory")
        llm_result = LLMResult(generations=[[generation]])
        self.mock_llm.generate_prompt.return_value = llm_result


        # First run to save something to memory
        self.memory.save_context({"input": "langchain query"}, {"output": "langchain trajectory"})

        # Now, create and run a chain that uses the memory
        template = "Based on this memory: {history}\\nAnswer the question: {input}"
        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        chain = LLMChain(llm=self.mock_llm, prompt=prompt, memory=self.memory)

        result = chain.invoke({"input": "another langchain query"})

        # The history should contain the memory we just added
        self.assertIn("Chain Title", self.memory.load_memory_variables({"input": "another langchain query"})['history'])

        # The final answer should be what we mocked
        self.assertEqual(result['text'], "Final answer based on memory")

if __name__ == '__main__':
    unittest.main()
