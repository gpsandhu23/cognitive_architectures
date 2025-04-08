from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Dict, List, Optional, Sequence, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.graph import StateGraph, add_messages
from langgraph.managed import IsLastStep
from langmem import create_manage_memory_tool, create_search_memory_tool

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """


model = ChatAnthropic(model="claude-3-sonnet-20240229")

namespace = "jarvis"

# Create memory tools

manage_memory_tool = create_manage_memory_tool(namespace)
search_memory_tool = create_search_memory_tool(namespace)


def get_memories(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """Get relevant memories for the current state."""
    # Extract the query from the last message
    query = state.messages[-1].content

    # Search for relevant memories using the search tool
    memories = search_memory_tool.invoke({"query": query})

    return {"messages": [AIMessage(content=memories)]}


def add_memory(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """Add a memory to the current state."""
    memory = manage_memory_tool.invoke({"query": state.messages[-1].content})
    return {"messages": [AIMessage(content=memory)]}


async def call_model(
    state: State, config: RunnableConfig, model: BaseChatModel = model
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.
        model (BaseChatModel): The model to use for the agent's main interactions.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Get the model's response

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                *state.messages,
            ],
            config,
        ),
    )

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node("get_memories", get_memories)
builder.add_node("add_memory", add_memory)
builder.add_node("call_model", call_model)

builder.add_edge("__start__", "get_memories")
builder.add_edge("get_memories", "call_model")
builder.add_edge("call_model", "add_memory")
builder.add_edge("add_memory", "__end__")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "Semantic Long Term Memory"  # This customizes the name in LangSmith
