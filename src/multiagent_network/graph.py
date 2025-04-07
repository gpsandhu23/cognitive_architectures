from typing import Dict, List, Literal, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Command


# Define the state for our graph
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str


# Define tools for our agents
@tool
def search(query: str) -> str:
    """Search for information on the web."""
    # This is a placeholder - in a real implementation, this would call a search API
    return f"Search results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


# Create a list of tools
tools = [search, calculator]

# Define the system prompts for our agents
researcher_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a researcher. Your job is to search for information and provide it to the writer."
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

writer_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a writer. Your job is to write content based on the information provided by the researcher."
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the LLM
llm = ChatOpenAI(model="gpt-4o")


# Define the agent nodes
def researcher_node(state: AgentState) -> Union[Dict, Command]:
    """Node for the researcher agent."""
    messages = state["messages"]

    # Run the agent
    response = llm.invoke(
        researcher_prompt.format_messages(
            messages=messages,
            agent_scratchpad=[],
        )
    )

    # Check if the agent wants to use a tool
    if "tool_calls" in response.additional_kwargs:
        tool_calls = response.additional_kwargs["tool_calls"]
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            # Find the tool by name
            tool_to_use = next((t for t in tools if t.__name__ == tool_name), None)
            if tool_to_use:
                # Execute the tool
                tool_result = tool_to_use(**eval(tool_args))
                tool_results.append(tool_result)

        # Add the tool results to the messages
        messages.append(
            AIMessage(
                content=response.content, additional_kwargs={"tool_calls": tool_calls}
            )
        )
        for tool_result in tool_results:
            messages.append(AIMessage(content=str(tool_result)))

        # Continue to the writer
        return {"messages": messages, "next": "writer"}

    # If no tool calls, add the response to the messages and continue to the writer
    messages.append(AIMessage(content=response.content))
    return {"messages": messages, "next": "writer"}


def writer_node(state: AgentState) -> Union[Dict, Command]:
    """Node for the writer agent."""
    messages = state["messages"]

    # Run the agent
    response = llm.invoke(
        writer_prompt.format_messages(
            messages=messages,
            agent_scratchpad=[],
        )
    )

    # Check if the agent wants to use a tool
    if "tool_calls" in response.additional_kwargs:
        tool_calls = response.additional_kwargs["tool_calls"]
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            # Find the tool by name
            tool_to_use = next((t for t in tools if t.__name__ == tool_name), None)
            if tool_to_use:
                # Execute the tool
                tool_result = tool_to_use(**eval(tool_args))
                tool_results.append(tool_result)

        # Add the tool results to the messages
        messages.append(
            AIMessage(
                content=response.content, additional_kwargs={"tool_calls": tool_calls}
            )
        )
        for tool_result in tool_results:
            messages.append(AIMessage(content=str(tool_result)))

        # Continue to the researcher
        return {"messages": messages, "next": "researcher"}

    # If no tool calls, check if the agent is done
    messages.append(AIMessage(content=response.content))

    # Check if the agent has indicated it's done
    if "FINAL ANSWER" in response.content:
        return Command(goto=END)

    # Otherwise, continue to the researcher
    return {"messages": messages, "next": "researcher"}


# Define the router function
def router(state: AgentState) -> Literal["researcher", "writer", END]:
    """Route to the next node based on the state."""
    return state["next"]


# Create the graph
def create_multi_agent_graph():
    """Create the multi-agent graph."""
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    # Add the edges
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "researcher")

    # Set the entry point
    workflow.set_entry_point("researcher")

    # Set the router
    workflow.add_conditional_edges(
        "researcher", router, {"researcher": "researcher", "writer": "writer", END: END}
    )

    workflow.add_conditional_edges(
        "writer", router, {"researcher": "researcher", "writer": "writer", END: END}
    )

    # Compile the graph
    return workflow.compile()


# Create the graph instance
graph = create_multi_agent_graph()
