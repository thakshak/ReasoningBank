# Design of the ReasoningBank Library

## 1. High-Level Architecture

The ReasoningBank library will be designed with a modular architecture, centered around the `ReasoningBank` class. This class will be responsible for orchestrating the storage, distillation, and retrieval of memories. It will interact with a `MemoryBackend` for persistence and an `EmbeddingModel` for generating embeddings.

The library will be structured as follows:

```
reasoningbank/
├── __init__.py
├── bank.py          # Core ReasoningBank class
├── memory.py        # Memory backend interfaces and implementations
├── distill.py       # Functions for distilling trajectories into memories
├── matts.py         # MaTTS implementations (parallel and sequential)
└── integrations/
    └── langchain/
        └── memory.py # LangChain memory integration
```

## 2. Core Components

### 2.1. `ReasoningBank` Class (`bank.py`)

This will be the main entry point for interacting with the library.

**Properties:**

- `memory_backend`: An instance of a class that implements the `MemoryBackend` interface.
- `embedding_model`: An instance of a class that can generate embeddings for text.
- `llm`: A language model to be used for distillation and judging.

**Methods:**

- `add_experience(trajectory: str, query: str)`: Adds a new experience to the bank. This method will use the LLM to judge the trajectory, distill it into memory items, and then store them in the memory backend.
- `retrieve_memories(query: str, k: int = 1)`: Retrieves the top `k` most relevant memories for a given query.
- `matts_parallel(query: str, k: int)`: Implements the parallel scaling version of MaTTS.
- `matts_sequential(query: str, k: int)`: Implements the sequential scaling version of MaTTS.

### 2.2. `MemoryBackend` (`memory.py`)

This module will define the `MemoryBackend` interface and provide concrete implementations.

**`MemoryBackend` Interface:**

- `add(items: List[Dict])`: Adds a list of memory items to the backend.
- `query(query_embedding: List[float], k: int)`: Queries the backend for the `k` most similar items to the given embedding.

**Implementations:**

- `ChromaMemoryBackend`: An implementation of the `MemoryBackend` interface using ChromaDB. This will be the default backend.
- `JSONMemoryBackend`: A simple implementation that stores memories in a JSON file. This will be useful for testing and simple use cases.

### 2.3. Trajectory Distillation (`distill.py`)

This module will contain functions for distilling raw trajectories into structured memory items.

- `distill_trajectory(trajectory: str, query: str, llm)`: This function will take a raw trajectory and a query, and use an LLM to generate a list of memory items (title, description, content).
- `judge_trajectory(trajectory: str, query: str, llm)`: This function will use an LLM to judge whether a trajectory was successful or not.

### 2.4. MaTTS (`matts.py`)

This module will contain the implementations of the Memory-aware Test-Time Scaling algorithms.

- `parallel_scaling(query: str, k: int, reasoning_bank: ReasoningBank)`: Generates `k` trajectories in parallel and uses the `ReasoningBank` to synthesize a final answer.
- `sequential_scaling(query: str, k: int, reasoning_bank: ReasoningBank)`: Iteratively refines a single trajectory `k` times.

## 3. LangChain Integration (`integrations/langchain/memory.py`)

To facilitate integration with LangChain, we will provide a custom LangChain memory class.

- `ReasoningBankMemory`: This class will inherit from LangChain's `BaseMemory` and will use the `ReasoningBank` library to store and retrieve memories.

## 4. Choice of Technology

- **Memory Backend:** We will start with **ChromaDB** as the primary memory backend. It is a popular and easy-to-use vector database that is well-suited for this project.
- **Embedding Model:** We will use a popular open-source embedding model from a library like `sentence-transformers`.
- **LLM:** The user of the library will be able to provide any LLM that is compatible with the LangChain interface.
