# Requirements for ReasoningBank Library

## 1. Core Functionality

- **Memory Storage:** The system must provide a mechanism to store agent experiences. These experiences include both successful and failed trajectories. The storage format should be a JSON file, where each entry consists of a task query, the original trajectory, and the corresponding memory items.
- **Memory Schema:** Each memory item must be structured with three components:
    - **Title:** A concise identifier summarizing the core strategy or reasoning pattern.
    - **Description:** A brief one-sentence summary of the memory item.
    - **Content:** The distilled reasoning steps, decision rationales, or operational insights.
- **Memory Distillation:** The system must be able to distill raw trajectories into the structured, reusable format defined by the memory schema. This process should be guided by an LLM.
- **Memory Retrieval:** The system must provide a mechanism to retrieve relevant memories based on a given query or context. Retrieval will be based on embedding-based semantic similarity search (cosine distance) on the task query embeddings.
- **Memory Consolidation:** New memory items generated from an agent's experience should be appended to the memory pool.
- **Self-Correction and Learning:** The system should learn from both successes and failures. An LLM-as-a-judge will be used to provide correctness signals (success/failure) for trajectories, which will then guide the memory distillation process.
- **Integration with Agentic Frameworks:** The library should be designed to be easily integrated with popular agentic frameworks like LangChain.

## 2. MaTTS (Memory-aware Test-Time Scaling)

- **Parallel Scaling:** The system must support the generation of multiple trajectories in parallel for a single query. This involves using self-contrast across trajectories to curate more reliable memories.
- **Sequential Scaling:** The system must support iterative self-refinement of a single trajectory to improve the quality of the solution and the extracted memory.

## 3. Technical Requirements

- **Python Implementation:** The core library must be implemented in Python.
- **Memory Backend:** The system should support a pluggable memory backend. The initial implementation will use ChromaDB for storing and querying memory embeddings, but it should be possible to extend it to other backends like FAISS or a simple JSON file with pre-computed embeddings.
- **Embeddings:** The system will use `gemini-embedding-001` for creating embeddings for task queries.
- **Dependencies:** The library should have a clear `requirements.txt` or `pyproject.toml` file to manage dependencies.

## 4. Documentation

- **README.md:** A comprehensive README file with installation instructions, usage examples, and a brief overview of the library.
- **API Documentation:** Clear and concise documentation for the public API of the library.

## 5. Testing

- **Unit Tests:** The library must have a suite of unit tests to ensure the correctness of individual components.
- **Integration Tests:** The library must have integration tests to ensure that the different components work together as expected, and to test the integration with LangChain.
