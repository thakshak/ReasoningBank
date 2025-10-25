# ReasoningBank

ReasoningBank is a Python library that provides a memory framework for LLM-powered agents. It allows agents to learn from their past experiences, both successful and failed, to improve their performance on future tasks.

This library is an implementation of the concepts presented in the research paper [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/html/2509.25140v1).

## Features

-   **Distills Experiences:** Automatically distills raw agent trajectories into structured, reusable reasoning patterns.
-   **Learns from Success and Failure:** Captures both effective strategies from successful attempts and preventative lessons from failures.
-   **Pluggable Memory Backend:** Comes with a `ChromaMemoryBackend` for persistent, embedding-based memory, and a simple `JSONMemoryBackend` for testing.
-   **LangChain Integration:** Provides a `ReasoningBankMemory` class for seamless integration with the LangChain framework.
-   **Memory-aware Test-Time Scaling (MaTTS):** Includes implementations of parallel and sequential scaling to enhance agent learning.

## Getting Started

Follow these steps to set up the project for local development.

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/example/reasoningbank.git
cd reasoningbank
```

### 2. Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

For development, you may also want to install the tools for formatting and linting:

```bash
pip install -r requirements-dev.txt
```

### 3. Configure the Environment

Copy the example configuration file to create your own local configuration:

```bash
cp config.yaml.example config.yaml
```

Then, edit `config.yaml` to match your desired settings. For example, you may need to specify the correct Ollama model or ChromaDB collection name.

## Configuration

ReasoningBank can be configured using a `config.yaml` file in the root of your project. This allows you to easily switch between different models and backends. A utility function `reasoningbank.utils.config.load_config` is provided to load these settings.

Here is an example `config.yaml`:

```yaml
# Default configuration for the ReasoningBank library

# Memory backend settings
memory:
  # The type of memory backend to use. Options: "chroma", "json"
  backend: "chroma"

  # Settings for the ChromaDB backend
  chroma:
    collection_name: "reasoning_bank"

  # Settings for the JSON backend
  json:
    filepath: "memory.json"

# Embedding model settings
embedding_model:
  # The embedding model to use. Options: "gemini-embedding-001", "sentence-transformers"
  model_name: "embeddinggemma:300m"

  # The name of the sentence-transformer model to use, if model_name is "sentence-transformers"
  st_model_name: "all-MiniLM-L6-v2"

# LLM settings
llm:
  # The LLM provider to use. Options: "ollama", "langchain.llms.Fake"
  provider: "ollama"

  # The model to use, if the provider is "ollama"
  model: "gemma3:270m"
```

### Key Configuration Options:

-   **`memory.backend`**: Choose between `chroma` for a persistent, vector-based memory, or `json` for a simple file-based memory.
-   **`embedding_model.model_name`**: Specify the embedding model.
-   **`llm.provider`**: Define the LLM to be used for distillation and synthesis. Supports `ollama` for local models.

## Installation

There are two ways to install the library, depending on your use case.

### For Users

If you want to use the `reasoningbank` library in your own project, you can install it directly from this repository:

```bash
pip install git+https://github.com/example/reasoningbank.git
```

This will install the library and its dependencies.

### For Developers

If you want to contribute to the development of the library, you should clone the repository and install it in editable mode, as described in the "Getting Started" section.

## Usage

The following examples assume you have a `config.yaml` file in your project's root directory.

### Standalone ReasoningBank

Here's how to use the `ReasoningBank` class, initializing its components based on your `config.yaml`:

```python
from reasoningbank.bank import ReasoningBank
from reasoningbank.utils.config import load_config
from reasoningbank.memory import ChromaMemoryBackend
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

# 1. Load configuration
config = load_config()

# 2. Initialize components based on the configuration
#    (In a real application, you might use a factory pattern for this)
memory_backend = ChromaMemoryBackend(**config['memory']['chroma'])
embedding_model = SentenceTransformer(config['embedding_model']['st_model_name'])
llm = Ollama(**config['llm'])

# 3. Create the ReasoningBank
bank = ReasoningBank(
    memory_backend=memory_backend,
    embedding_model=embedding_model,
    llm=llm
)

# 4. Add an experience
trajectory = "Agent did this... and it worked."
query = "How to do the thing?"
bank.add_experience(trajectory, query)

# 5. Retrieve memories for a new query
retrieved_memories = bank.retrieve_memories("a similar query", k=1)
print(retrieved_memories)
```

### LangChain Integration

ReasoningBank can be used as a memory component within a LangChain chain:

```python
from reasoningbank.integrations.langchain.memory import ReasoningBankMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# (Set up bank as in the previous example)

# 1. Create the ReasoningBankMemory
memory = ReasoningBankMemory(reasoning_bank=bank)

# 2. Create a chain with the memory
template = "Based on this memory: {history}\\nAnswer the question: {input}"
prompt = PromptTemplate(input_variables=["history", "input"], template=template)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# 3. Run the chain
# This will save the input/output as an experience in the ReasoningBank
chain.invoke({"input": "What is the capital of France?"})

# A subsequent call with a similar query will load the relevant memories
chain.invoke({"input": "What is the main city in France?"})
```

### Memory-aware Test-Time Scaling (MaTTS)

You can use the MaTTS functions to generate multiple trajectories and enhance learning.

```python
from reasoningbank.matts import parallel_scaling, sequential_scaling
from reasoningbank.agent import create_agent_executor
from langchain_community.llms.fake import FakeListLLM

# (Set up bank as in the previous example)

# Use a mock LLM for the agent
agent_llm = FakeListLLM(responses=["trajectory 1", "trajectory 2", "synthesized answer"])
agent_executor = create_agent_executor(agent_llm)

# Use parallel scaling to generate 2 trajectories
final_answer = parallel_scaling(
    query="test query",
    k=2,
    reasoning_bank=bank,
    agent_executor=agent_executor
)
print(final_answer)
```

## Running the Examples

The `examples/` directory contains a simple script to demonstrate the basic functionality of the library.

To run the example:

1.  Make sure you have installed the dependencies as described in the "Getting Started" section.
2.  Run the `simple_usage.py` script from the root of the repository:

```bash
python -m examples.simple_usage
```

This will run a simple demonstration of adding an experience to the memory bank and then retrieving it.
