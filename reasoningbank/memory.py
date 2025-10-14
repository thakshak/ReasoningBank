import abc
import chromadb
from typing import List, Dict, Optional
import uuid
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MemoryBackend(abc.ABC):
    """Abstract base class for memory backends."""

    @abc.abstractmethod
    def add(self, items: List[Dict]):
        """Adds a list of memory items to the backend."""
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """Queries the backend for the k most similar items."""
        raise NotImplementedError

class ChromaMemoryBackend(MemoryBackend):
    """A memory backend that uses ChromaDB for storage."""

    def __init__(self, collection_name: str = "reasoning_bank"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, items: List[Dict]):
        """
        Adds a list of memory items to the ChromaDB collection.

        Each item is a dictionary that should contain:
        - embedding: The embedding of the memory item.
        - metadata: A dictionary with title, description, and content.
        - document: The content of the memory item.
        """
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in items],
            embeddings=[item["embedding"] for item in items],
            metadatas=[item["metadata"] for item in items],
            documents=[item["document"] for item in items],
        )

    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Queries the ChromaDB collection for the k most similar items.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        # The query returns a list of lists for metadatas, one for each query embedding.
        # Since we only pass one embedding, we take the first list of results.
        return results["metadatas"][0] if results["metadatas"] else []

class JSONMemoryBackend(MemoryBackend):
    """A simple memory backend that stores memories in a JSON file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> List[Dict]:
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f)

    def add(self, items: List[Dict]):
        self.data.extend(items)
        self._save()

    def query(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Queries the JSON file for the k most similar items using cosine similarity.
        """
        if not self.data:
            return []

        embeddings = np.array([item['embedding'] for item in self.data])
        query_embedding = np.array(query_embedding).reshape(1, -1)

        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # Get the indices of the top k most similar items
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [self.data[i]['metadata'] for i in top_k_indices]
