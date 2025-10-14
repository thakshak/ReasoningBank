from typing import List, Dict, Any
import json

# A placeholder for a generic LLM interface.
# In a real implementation, this would be a more specific type,
# for example, a LangChain BaseLanguageModel.
LLM = Any

def judge_trajectory(trajectory: str, query: str, llm: LLM) -> bool:
    """
    Judges whether a trajectory was successful or not using an LLM.
    """
    prompt = f"""
    Given the following query and trajectory, determine if the trajectory successfully addresses the query.
    Respond with "Success" or "Failure".

    Query: {query}

    Trajectory:
    {trajectory}
    """
    response = llm.invoke(prompt)
    return "success" in response.lower()

def distill_trajectory(trajectory: str, query: str, llm: LLM, is_success: bool) -> List[Dict]:
    """
    Distills a raw trajectory into a list of structured memory items.
    """
    if is_success:
        prompt = f"""
        The following trajectory was successful in addressing the query.
        Distill the key reasoning steps and strategies into a few memory items.
        Each memory item should have a title, a short description, and content.
        Format the output as a JSON string representing a list of dictionaries. For example:
        [
            {{
                "title": "Example Title",
                "description": "A short description.",
                "content": "The detailed reasoning steps."
            }}
        ]

        Query: {query}

        Trajectory:
        {trajectory}
        """
    else:
        prompt = f"""
        The following trajectory failed to address the query.
        Analyze the failure and distill the lessons learned into a few memory items.
        Each memory item should have a title, a short description, and content describing the pitfall and how to avoid it.
        Format the output as a JSON string representing a list of dictionaries. For example:
        [
            {{
                "title": "Example Pitfall",
                "description": "A short description of the error.",
                "content": "A detailed explanation of the mistake and how to avoid it in the future."
            }}
        ]

        Query: {query}

        Trajectory:
        {trajectory}
        """

    response = llm.invoke(prompt)

    # Use json.loads for safe parsing of the LLM's JSON output.
    try:
        distilled_memories = json.loads(response)
        if isinstance(distilled_memories, list):
            return distilled_memories
    except json.JSONDecodeError:
        # If parsing fails, return an empty list.
        return []

    return []
