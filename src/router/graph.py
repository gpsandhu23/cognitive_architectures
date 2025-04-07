"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from router.configuration import Configuration
from router.state import InputState, State
from router.utils import load_chat_model

# Define the function that calls the model


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model
    model = load_chat_model(configuration.model)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def vowel(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """Node that generates 5 random words starting with a vowel."""
    return {"messages": [AIMessage(content="AI response started with a vowel!")]}


async def consonant(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """Node that generates 5 random words starting with a consonant."""
    return {"messages": [AIMessage(content="AI response started with a consonant!")]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node(vowel)
builder.add_node(consonant)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["vowel", "consonant"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message starts with a vowel.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("vowel" or "consonant").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If the last message starts with a vowel, go to node1
    if last_message.content[0].lower() in "aeiou":
        return "vowel"
    # Otherwise go to node2
    else:
        return "consonant"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `nodes` to `end`
builder.add_edge("vowel", "__end__")
builder.add_edge("consonant", "__end__")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "Router"  # This customizes the name in LangSmith
