from typing import Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .bank import ReasoningBank
from .agent import format_memories_for_prompt

# A placeholder for a generic agent execution function.
# In a real implementation, this would be a proper agent class or function.
AgentExecutor = Any


def parallel_scaling(
    query: str, k: int, reasoning_bank: ReasoningBank, agent_executor: AgentExecutor
) -> str:
    """
    Implements parallel scaling MaTTS.
    Generates k trajectories in parallel, learns from them, and synthesizes a
    final answer.
    """
    # 1. Retrieve initial memories to guide the parallel generation.
    initial_memories = reasoning_bank.retrieve_memories(query, k=1)
    formatted_memories = format_memories_for_prompt(initial_memories)

    # 2. Generate k trajectories in parallel.
    # In a real implementation, this could be done with asyncio or threading.
    trajectories = []
    for _ in range(k):
        trajectory = agent_executor.invoke(
            {"memories": formatted_memories, "query": query}
        )
        trajectories.append(trajectory)

    # 3. Add the new experiences to the ReasoningBank to learn from them.
    for trajectory in trajectories:
        reasoning_bank.add_experience(trajectory, query)

    # 4. Synthesize a final answer from the generated trajectories.
    trajectories_str = "\n---\n".join(trajectories)
    synthesis_template = """
    Given the following query and {k} proposed trajectories, select the best
    one or synthesize a final answer.

    Query: {query}

    Trajectories:
    {trajectories_str}
    """
    synthesis_prompt = PromptTemplate.from_template(synthesis_template)
    synthesis_chain = synthesis_prompt | reasoning_bank.llm | StrOutputParser()
    final_answer = synthesis_chain.invoke(
        {"query": query, "trajectories_str": trajectories_str, "k": k}
    )
    return final_answer


def sequential_scaling(
    query: str, k: int, reasoning_bank: ReasoningBank, agent_executor: AgentExecutor
) -> str:
    """
    Implements sequential scaling MaTTS.
    Iteratively refines a single trajectory k times.
    """
    trajectory = ""
    for _ in range(k):
        # 1. Retrieve memories to guide the current refinement step.
        memories = reasoning_bank.retrieve_memories(query, k=1)
        formatted_memories = format_memories_for_prompt(memories)

        # 2. Run the agent for one step of refinement.
        # The agent is prompted to refine the existing trajectory.
        refinement_template = """
        Based on the following memories, refine the current trajectory to
        better answer the query.

        Memories:
        {memories}

        Query: {query}

        Current Trajectory:
        {trajectory}

        Refined Trajectory:
        """
        refinement_prompt = PromptTemplate.from_template(refinement_template)
        refinement_chain = refinement_prompt | reasoning_bank.llm | StrOutputParser()
        trajectory = refinement_chain.invoke(
            {
                "memories": formatted_memories,
                "query": query,
                "trajectory": trajectory,
            }
        )

    # 3. Add the final trajectory to the ReasoningBank.
    reasoning_bank.add_experience(trajectory, query)

    return trajectory
