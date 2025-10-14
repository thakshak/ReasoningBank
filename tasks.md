# AWS Kiro Style Task List for ReasoningBank Library

## Phase 1: Core Library and Memory Backend

- **Task 1.1: Set up the project structure.**
  - **Description:** Create the directory structure as outlined in `design.md`. Initialize a `pyproject.toml` file with basic project metadata and dependencies (e.g., `chromadb`, `sentence-transformers`, `langchain`).
  - **Links to Requirements:** 3.1, 3.3

- **Task 1.2: Implement the `MemoryBackend` interface and `ChromaMemoryBackend`.**
  - **Description:** Create the `memory.py` module. Define the `MemoryBackend` abstract base class with `add` and `query` methods. Implement the `ChromaMemoryBackend` class that uses the `chromadb` library to store and retrieve memory items.
  - **Links to Requirements:** 1.1, 1.3, 3.2

- **Task 1.3: Implement the trajectory distillation functions.**
  - **Description:** Create the `distill.py` module. Implement the `judge_trajectory` and `distill_trajectory` functions. These functions will use a provided LLM to perform their tasks. For now, the prompts can be simple, and we can refine them later.
  - **Links to Requirements:** 1.2, 1.4

- **Task 1.4: Implement the core `ReasoningBank` class.**
  - **Description:** Create the `bank.py` module. Implement the `ReasoningBank` class with the `add_experience` and `retrieve_memories` methods. This class will wire together the `MemoryBackend` and the distillation functions.
  - **Links to Requirements:** 1.1, 1.2, 1.3, 1.4

## Phase 2: Testing

- **Task 2.1: Write unit tests for the `MemoryBackend`.**
  - **Description:** Create a `tests/` directory. Write unit tests for the `ChromaMemoryBackend` to ensure that it correctly adds and retrieves data. It would be beneficial to also create a `JSONMemoryBackend` for easier testing, and write tests for it as well.
  - **Links to Requirements:** 5.1

- **Task 2.2: Write unit tests for the `ReasoningBank` class.**
  - **Description:** Write unit tests for the `ReasoningBank` class. Mock the `MemoryBackend` and the LLM to test the logic of the `add_experience` and `retrieve_memories` methods in isolation.
  - **Links to Requirements:** 5.1

## Phase 3: MaTTS and LangChain Integration

- **Task 3.1: Implement the MaTTS functions.**
  - **Description:** Create the `matts.py` module. Implement the `parallel_scaling` and `sequential_scaling` functions as described in `design.md`.
  - **Links to Requirements:** 2.1, 2.2

- **Task 3.2: Implement the LangChain integration.**
  - **Description:** Create the `integrations/langchain/memory.py` module. Implement the `ReasoningBankMemory` class that inherits from `BaseMemory` and uses the `ReasoningBank` library.
  - **Links to Requirements:** 1.5

- **Task 3.3: Write integration tests.**
  - **Description:** Write integration tests to verify that the `ReasoningBank` works correctly with the `ChromaMemoryBackend` and a real (or mocked) LLM. Also, write an integration test for the `ReasoningBankMemory` class to ensure it works within the LangChain framework.
  - **Links to Requirements:** 5.2

## Phase 4: Documentation and Finalization

- **Task 4.1: Write the README.md file.**
  - **Description:** Create a comprehensive `README.md` file with installation instructions, usage examples for the standalone library and the LangChain integration, and a brief overview of the library.
  - **Links to Requirements:** 4.1

- **Task 4.2: Add API documentation.**
  - **Description:** Add docstrings to all public classes and methods to serve as API documentation.
  - **Links to Requirements:** 4.2
