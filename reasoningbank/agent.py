from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from typing import List, Dict

def create_agent_executor(llm: BaseLanguageModel) -> LLMChain:
    """
    Creates a simple agent executor for generating trajectories.
    This is a basic LLMChain that responds to a query based on provided memories.
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
    Formats a list of memory dictionaries into a string for the prompt.
    """
    if not memories:
        return "No relevant memories found."

    return "\n---\n".join(
        f"Title: {m.get('title', 'N/A')}\nDescription: {m.get('description', 'N/A')}\nContent: {m.get('content', 'N/A')}"
        for m in memories
    )
