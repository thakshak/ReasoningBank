from typing import Any, List, Dict
from .memory import MemoryBackend
from .distill import judge_trajectory, distill_trajectory

# Placeholder for a generic Embedding Model interface.
# The user would provide a model with an `embed_documents` method.
EmbeddingModel = Any
LLM = Any

class ReasoningBank:
    """
    The core class for the ReasoningBank library.
    Orchestrates memory storage, distillation, and retrieval.
    """

    def __init__(self, memory_backend: MemoryBackend, embedding_model: EmbeddingModel, llm: LLM):
        self.memory_backend = memory_backend
        self.embedding_model = embedding_model
        self.llm = llm

    def add_experience(self, trajectory: str, query: str):
        """
        Adds a new experience to the bank.
        This involves judging the trajectory, distilling it into memory items,
        and storing them in the memory backend.
        """
        is_success = judge_trajectory(trajectory, query, self.llm)
        distilled_items = distill_trajectory(trajectory, query, self.llm, is_success)

        if not distilled_items:
            return

        # Generate embeddings for the content of each distilled item.
        contents = [item['content'] for item in distilled_items]
        embeddings = self.embedding_model.encode(contents)

        # Prepare the items for storage in the memory backend.
        items_to_add = []
        for i, item in enumerate(distilled_items):
            items_to_add.append({
                "embedding": embeddings[i],
                "metadata": {
                    "title": item['title'],
                    "description": item['description'],
                    "content": item['content']
                },
                "document": item['content']
            })

        self.memory_backend.add(items_to_add)

    def retrieve_memories(self, query: str, k: int = 1) -> List[Dict]:
        """
        Retrieves the top k most relevant memories for a given query.
        """
        query_embedding = self.embedding_model.encode(query)
        return self.memory_backend.query(query_embedding, k)
