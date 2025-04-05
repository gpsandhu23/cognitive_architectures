"""Define the reflection agent graph with generation and reflection nodes."""

from datetime import UTC, datetime
from typing import Dict, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from reflection_agent.configuration import Configuration
from reflection_agent.state import InputState, State
from reflection_agent.utils import load_chat_model


async def generation_node(state: State, config: RunnableConfig) -> Dict:
    """Generate an initial response to the user's question.

    Args:
        state: The current state of the conversation.
        config: Configuration for the model run.

    Returns:
        dict: Updated state with new response message and reflection flag.
    """
    configuration = Configuration.from_runnable_config(config)
    print(f"GENERATION START - Reflection count: {state.reflection_count}")

    model = load_chat_model(configuration)

    # Create dictionary to hold updated state values
    updated_state = {}

    # Store the original question if this is the first generation
    if not state.original_question and state.messages:
        first_msg = next(
            (msg.content for msg in state.messages if isinstance(msg, HumanMessage)),
            None,
        )
        if first_msg:
            updated_state["original_question"] = first_msg
            print(f"GENERATION - Setting original question: {first_msg[:50]}...")

    question = state.original_question or first_msg or "No question provided"

    # Format the system prompt
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat(),
        question=question,
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Always set needs_reflection to True after generation
    print("GENERATION COMPLETE - Moving to reflection")

    return {"needs_reflection": True, "messages": [response]}


async def reflection_node(state: State, config: RunnableConfig) -> Dict:
    """Evaluate the generated response and determine if improvements are needed.

    Args:
        state: The current state of the conversation.
        config: Configuration for the model run.

    Returns:
        dict: Updated state with new messages and updated reflection status.
    """
    configuration = Configuration.from_runnable_config(config)

    # Debug current state
    print(f"REFLECTION START - Current count: {state.reflection_count}")

    # Explicitly update the state
    updated_state = {
        "reflection_count": state.reflection_count + 1,
    }
    print(f"REFLECTION - Setting count to {state.reflection_count + 1}")

    # Hard limit on reflection cycles
    if state.reflection_count + 1 >= configuration.max_reflections:
        print(f"REFLECTION - Hit max reflections: {configuration.max_reflections}")
        return {
            "needs_reflection": False,
            "reflection_count": state.reflection_count + 1,
            "messages": [
                HumanMessage(content="Max reflections reached, ending cycle.")
            ],
        }

    model = load_chat_model(configuration)

    # Get the last generated response
    last_response = next(
        (msg.content for msg in reversed(state.messages) if isinstance(msg, AIMessage)),
        None,
    )

    if not last_response:
        print("REFLECTION - No response to evaluate")
        return {
            "needs_reflection": False,
            "reflection_count": state.reflection_count + 1,
            "messages": [HumanMessage(content="No response to evaluate.")],
        }

    # Get original question - if not set, use first message
    if not state.original_question and state.messages:
        human_msg = next(
            (msg.content for msg in state.messages if isinstance(msg, HumanMessage)),
            None,
        )
        updated_state["original_question"] = human_msg
        print(f"REFLECTION - Setting original question: {human_msg}")

    original_question = state.original_question or "No question provided"

    # Format the reflection prompt with explicit instructions
    reflection_message = (
        "You are evaluating an AI's response. You MUST end your evaluation with "
        "EXACTLY ONE of these two statements on a new line:\n"
        "NEEDS_IMPROVEMENT: <your specific critique>\n"
        "or\n"
        "NO_IMPROVEMENT_NEEDED\n\n"
        f"Original Question: {original_question}\n"
        f"Generated Response: {last_response}\n\n"
        "Evaluate based on:\n"
        "1. Accuracy\n"
        "2. Completeness\n"
        "3. Clarity\n"
        "4. Relevance\n"
        "5. Appropriateness\n"
    )

    # Get the reflection evaluation
    reflection = await model.ainvoke(
        [{"role": "system", "content": reflection_message}], config
    )

    response_text = reflection.content.strip()
    print(f"REFLECTION - Response: {response_text[:50]}...")

    # Check for explicit improvement needed marker
    if "NEEDS_IMPROVEMENT:" in response_text:
        critique = response_text.split("NEEDS_IMPROVEMENT:", 1)[1].strip()
        if critique.strip():
            print("REFLECTION - Improvements needed")
            return {
                "needs_reflection": True,
                "reflection_count": state.reflection_count + 1,
                "messages": [
                    HumanMessage(
                        content=f"Please improve your response based on this feedback: {critique}"
                    )
                ],
            }

    # If we get here, either:
    # 1. NO_IMPROVEMENT_NEEDED was found
    # 2. NEEDS_IMPROVEMENT had no actual critique
    # 3. The format wasn't followed
    print("REFLECTION - No improvements needed")
    return {
        "needs_reflection": False,
        "reflection_count": state.reflection_count + 1,
        "messages": [HumanMessage(content="No improvements needed.")],
    }


# Define the graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the nodes
builder.add_node("generation", generation_node)
builder.add_node("reflection", reflection_node)

# Set the entrypoint as generation
builder.add_edge("__start__", "generation")
builder.add_edge("generation", "reflection")


def should_reflect_again(state: State) -> Literal["__end__", "generation"]:
    """Decide whether to perform another reflection cycle.

    This is called after the reflection node to determine if we should:
    1. Generate another response based on the reflection, or
    2. End the process

    Args:
        state: The current state

    Returns:
        Either "__end__" or "generation"
    """
    # Print debug info
    print(
        f"DECISION POINT - Reflection count: {state.reflection_count}, Needs reflection: {state.needs_reflection}"
    )

    # Hard limit on reflection count
    if state.reflection_count >= 2:
        print("ENDING: Maximum reflection count reached")
        return "__end__"

    # Check if reflection node indicated no more improvements needed
    if not state.needs_reflection:
        print("ENDING: No more reflection needed")
        return "__end__"

    # Continue with another generation
    print("CONTINUING: Generating improved response")
    return "generation"


# Add a conditional edge from reflection
builder.add_conditional_edges("reflection", should_reflect_again)

# Compile the graph
graph = builder.compile()
graph.name = "Reflection Agent"
