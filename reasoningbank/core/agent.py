"""Agent-related functionalities for the ReasoningBank library."""

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from typing import List, Dict


def create_agent_executor(llm: BaseLanguageModel) -> LLMChain:
    """
    Creates a simple agent executor for generating trajectories.

    This function sets up a basic LLMChain that takes a set of memories and a query,
    and generates a response that represents the agent's thought process or trajectory.
    This is intended for demonstration and testing purposes.

    Args:
        llm (BaseLanguageModel): An instance of a LangChain compatible language model.

    Returns:
        LLMChain: A LangChain LLMChain configured to generate trajectories.
    """
    template = """
    You are a helpful assistant.
    Based on the following memories, answer the user's query.

    Memories:
    {memories}

    Query: {query}

    Your response is a trajectory of your thought process.
    Trajectory:
    """
    prompt = PromptTemplate(input_variables=["memories", "query"], template=template)

    # Note: LLMChain is deprecated, but we use it here for simplicity
    # to match the existing test structure. A production implementation
    # would use the LangChain Expression Language (LCEL).
    return LLMChain(llm=llm, prompt=prompt)


def format_memories_for_prompt(memories: List[Dict]) -> str:
    """
    Formats a list of memory dictionaries into a string suitable for a prompt.

    Args:
        memories (List[Dict]): A list of memory dictionaries, where each dictionary
            is expected to have 'title', 'description', and 'content' keys.

    Returns:
        str: A formatted string of memories, or a message indicating that no
             relevant memories were found.
    """
    if not memories:
        return "No relevant memories found."

    return "\n---\n".join(
        f"Title: {m.get('metadata', {}).get('title', 'N/A')}\nDescription: {m.get('metadata', {}).get('description', 'N/A')}\nContent: {m.get('document', 'N/A')}"
        for m in memories
    )
