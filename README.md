# ReasoningBank

ReasoningBank is a Python library that provides a memory framework for LLM-powered agents. It allows agents to learn from their past experiences, both successful and failed, to improve their performance on future tasks.

This library is an implementation of the concepts presented in the research paper [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/html/2509.25140v1).

## Features

-   **Distills Experiences:** Automatically distills raw agent trajectories into structured, reusable reasoning patterns.
-   **Learns from Success and Failure:** Captures both effective strategies from successful attempts and preventative lessons from failures.
-   **Pluggable Memory Backend:** Comes with a `ChromaMemoryBackend` for persistent, embedding-based memory, and a simple `JSONMemoryBackend` for testing.
-   **LangChain Integration:** Provides a `ReasoningBankMemory` class for seamless integration with the LangChain framework.
-   **Memory-aware Test-Time Scaling (MaTTS):** Includes implementations of parallel and sequential scaling to enhance agent learning.

## Installation

You can install ReasoningBank and its dependencies using pip:

```bash
pip install chromadb sentence-transformers langchain numpy scikit-learn
```

## Usage

### Standalone ReasoningBank

Here's how to use the `ReasoningBank` class on its own:

```python
from reasoningbank.bank import ReasoningBank
from reasoningbank.memory import ChromaMemoryBackend
from sentence_transformers import SentenceTransformer
from langchain_community.llms.fake import FakeListLLM

# 1. Set up the components
memory_backend = ChromaMemoryBackend(collection_name="my_agent_memory")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Use a mock LLM for this example
llm = FakeListLLM(responses=[
    "Success",
    '[{"title": "Example Strategy", "description": "A good way to do things.", "content": "Do this, then that."}]'
])

# 2. Create the ReasoningBank
bank = ReasoningBank(
    memory_backend=memory_backend,
    embedding_model=embedding_model,
    llm=llm
)

# 3. Add an experience
trajectory = "Agent did this... and it worked."
query = "How to do the thing?"
bank.add_experience(trajectory, query)

# 4. Retrieve memories for a new query
retrieved_memories = bank.retrieve_memories("a similar query", k=1)
print(retrieved_memories)
```

### LangChain Integration

ReasoningBank can be used as a memory component within a LangChain chain:

```python
from reasoningbank.integrations.langchain.memory import ReasoningBankMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# (Set up bank, memory_backend, embedding_model, llm as in the previous example)

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

# (Set up bank, memory_backend, embedding_model as in the first example)

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
