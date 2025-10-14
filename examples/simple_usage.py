from reasoningbank import ReasoningBank
import os


def run_example():
    """A simple example of how to use the ReasoningBank library."""

    # Initialize the ReasoningBank from the example config
    bank = ReasoningBank(config_path="examples/config.yaml")

    # Add an experience
    trajectory = (
        "1. Thought: I need to find the capital of France. "
        "2. Action: Search for 'capital of France'. "
        "3. Observation: The capital of France is Paris."
    )
    query = "What is the capital of France?"
    bank.add_experience(trajectory, query)

    print("Experience added to the memory bank.")

    # Retrieve memories
    retrieved_memories = bank.retrieve_memories(
        "What is the main city in France?", k=1
    )

    print("\nRetrieved memories:")
    for memory in retrieved_memories:
        print(f"  Title: {memory['title']}")
        print(f"  Description: {memory['description']}")
        print(f"  Content: {memory['content']}")

    # Clean up the example memory file
    if os.path.exists("example_memory.json"):
        os.remove("example_memory.json")


if __name__ == "__main__":
    run_example()
