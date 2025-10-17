"""Core components for the ReasoningBank library."""
import json
from typing import Any, List, Dict
from ..memory.base import MemoryBackend
from ..distillation.distill import judge_trajectory, distill_trajectory
from ..utils.config import load_config
from ..memory.chroma import ChromaMemoryBackend
from ..memory.json import JSONMemoryBackend
from sentence_transformers import SentenceTransformer
from langchain_community.llms import FakeListLLM

# Placeholder for a generic Embedding Model interface.
# The user would provide a model with an `embed_documents` method.
EmbeddingModel = Any
LLM = Any


class ReasoningBank:
    """
    The core class for the ReasoningBank library.

    Orchestrates memory storage, distillation, and retrieval. It integrates a
    memory backend, an embedding model, and a language model to provide a
    comprehensive solution for managing and utilizing agent experiences.

    Attributes:
        memory_backend (MemoryBackend): The backend used for storing and
            retrieving memories.
        embedding_model (EmbeddingModel): The model used for generating
            embeddings for text.
        llm (LLM): The language model used for judging and distilling
            trajectories.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the ReasoningBank from a configuration file.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.config = load_config(config_path)

        self.memory_backend = self._init_memory_backend()
        self.embedding_model = self._init_embedding_model()
        self.llm = self._init_llm()

    def _init_memory_backend(self) -> MemoryBackend:
        """Initializes the memory backend based on the configuration."""
        backend_type = self.config["memory"]["backend"]
        if backend_type == "chroma":
            return ChromaMemoryBackend(
                collection_name=self.config["memory"]["chroma"]["collection_name"]
            )
        elif backend_type == "json":
            return JSONMemoryBackend(
                filepath=self.config["memory"]["json"]["filepath"]
            )
        else:
            raise ValueError(f"Unknown memory backend type: {backend_type}")

    def _init_embedding_model(self) -> EmbeddingModel:
        """Initializes the embedding model based on the configuration."""
        model_name = self.config["embedding_model"]["model_name"]
        return SentenceTransformer(model_name)

    def _init_llm(self) -> LLM:
        """Initializes the language model based on the configuration."""
        # This is a placeholder for a more complex LLM initialization.
        # In a real application, this would involve loading the specified LLM
        # from a library like LangChain.
        provider = self.config["llm"]["provider"]
        if provider == "langchain.llms.Fake":
            # For demonstration purposes, we use a fake LLM.
            responses = [
                "Success",
                json.dumps(
                    [
                        {
                            "title": "Fake Memory",
                            "description": "A fake memory",
                            "content": "This is a fake memory.",
                        }
                    ]
                ),
            ]
            return FakeListLLM(responses=responses)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def add_experience(self, trajectory: str, query: str):
        """
        Adds a new experience to the bank.

        This method takes a trajectory and a query, judges the trajectory's
        success, distills it into memory items, generates embeddings for these
        items, and stores them in the memory backend.

        Args:
            trajectory (str): The sequence of actions and observations that
                constitute the experience.
            query (str): The initial query or task that the agent was trying
                to solve.
        """
        is_success = judge_trajectory(trajectory, query, self.llm)
        distilled_items = distill_trajectory(
            trajectory, query, self.llm, is_success
        )

        if not distilled_items:
            return

        # Generate embeddings for the content of each distilled item.
        contents = [item["content"] for item in distilled_items]
        embeddings = self.embedding_model.encode(contents)

        # Prepare the items for storage in the memory backend.
        items_to_add = []
        for i, item in enumerate(distilled_items):
            items_to_add.append(
                {
                    "embedding": embeddings[i].tolist(),
                    "metadata": {
                        "title": item["title"],
                        "description": item["description"],
                        "content": item["content"],
                    },
                    "document": item["content"],
                }
            )

        self.memory_backend.add(items_to_add)

    def retrieve_memories(self, query: str, k: int = 1) -> List[Dict]:
        """
        Retrieves the top k most relevant memories for a given query.

        Args:
            query (str): The query to retrieve relevant memories for.
            k (int): The number of memories to retrieve.

        Returns:
            List[Dict]: A list of the top k most relevant memories.
        """
        query_embedding = self.embedding_model.encode(query)
        return self.memory_backend.query(query_embedding.tolist(), k)
