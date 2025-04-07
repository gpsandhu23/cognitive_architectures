"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from chain.configuration import Configuration
from chain.state import InputState, State
from chain.utils import load_chat_model

# Define the function that calls the model


async def joke_setup(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Generate a joke setup.

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
    system_message = configuration.joke_setup_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    joke_setup = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Return the model's response as a list to be added to existing messages
    return {"messages": [joke_setup]}


async def joke_punchline(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Generate a joke punchline.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model
    model = load_chat_model(configuration.model)

    # Get the setup from the previous message
    if not state.messages:
        return {
            "messages": [
                AIMessage(content="No setup found to generate a punchline for.")
            ]
        }

    setup_message = next(
        (msg for msg in reversed(state.messages) if isinstance(msg, AIMessage)), None
    )

    if not setup_message:
        return {
            "messages": [
                AIMessage(content="No setup found to generate a punchline for.")
            ]
        }

    # Format the system prompt with the setup
    system_message = configuration.joke_punchline_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Create a message that explicitly includes the setup
    setup_context = f"Here is the joke setup: {setup_message.content}\n\nNow generate just the punchline:"

    # Get the model's response
    try:
        joke_punchline = cast(
            AIMessage,
            await model.ainvoke(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": setup_context},
                ],
                config,
            ),
        )
    except Exception as e:
        error_msg = f"Failed to generate punchline: {str(e)}"

        joke_punchline = AIMessage(content=error_msg)

    # Return the model's response as a list to be added to existing messages
    return {"messages": [joke_punchline]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes that will be called one after the other
builder.add_node(joke_setup)
builder.add_node(joke_punchline)

# Set the entrypoint as `joke_setup`
# This means that this node is the first one called
builder.add_edge("__start__", "joke_setup")
builder.add_edge("joke_setup", "joke_punchline")
builder.add_edge("joke_punchline", "__end__")


# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "Chain"  # This customizes the name in LangSmith
