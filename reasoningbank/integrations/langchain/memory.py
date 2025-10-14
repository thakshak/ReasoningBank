from langchain_core.memory import BaseMemory
from typing import Dict, Any, List
from reasoningbank.bank import ReasoningBank

class ReasoningBankMemory(BaseMemory):
    """A LangChain memory class that uses the ReasoningBank."""

    reasoning_bank: ReasoningBank
    memory_key: str = "history"  # The key for the memory variables.

    @property
    def memory_variables(self) -> List[str]:
        """The list of memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load the memory variables."""
        # We'll use the first key in the inputs dict as the query.
        # This is a simplification; a more robust implementation might
        # have a more explicit way of defining the query.
        query = next(iter(inputs.values()))

        retrieved_memories = self.reasoning_bank.retrieve_memories(query, k=1)

        # Format the memories into a string for the prompt.
        formatted_memories = "\n".join(
            f"Title: {m['title']}\nDescription: {m['description']}\nContent: {m['content']}"
            for m in retrieved_memories
        )

        return {self.memory_key: formatted_memories}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of a chain run to the ReasoningBank."""
        query = next(iter(inputs.values()))
        trajectory = next(iter(outputs.values()))

        self.reasoning_bank.add_experience(trajectory, query)

    def clear(self) -> None:
        """Clear the memory."""
        # This is not directly supported by the current ReasoningBank design,
        # as the bank is meant to be persistent.
        # A possible implementation would be to clear the memory backend.
        pass
