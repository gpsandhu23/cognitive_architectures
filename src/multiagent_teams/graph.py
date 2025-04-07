from typing import Dict, List, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# Define the state for our graph
class State(TypedDict):
    messages: List[BaseMessage]


# Define tools for our agents
@tool
def search(query: str) -> str:
    """Search for information on the web."""
    # This is a placeholder - in a real implementation, this would call a search API
    return f"Search results for: {query}"


@tool
def scrape_webpage(url: str) -> str:
    """Scrape a webpage for information."""
    # This is a placeholder - in a real implementation, this would scrape the webpage
    return f"Scraped content from: {url}"


@tool
def create_outline(points: List[str]) -> str:
    """Create an outline from a list of points."""
    outline = "Outline:\n"
    for i, point in enumerate(points):
        outline += f"{i + 1}. {point}\n"
    return outline


@tool
def write_document(content: str) -> str:
    """Write a document with the given content."""
    # This is a placeholder - in a real implementation, this would write to a file
    return f"Document written with content: {content}"


# Create a list of tools for each team
research_tools = [search, scrape_webpage]
writing_tools = [create_outline, write_document]


# Define the system prompts for our agents
def make_system_prompt(role: str, tools: List) -> str:
    tool_descriptions = "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in tools]
    )
    return f"""You are a {role}. Your job is to help with the task at hand.
You have access to the following tools:
{tool_descriptions}

Use these tools to help with your task. If you need to use a tool, call it with the appropriate arguments.
If you have completed your task, respond with "TASK COMPLETE".
"""


# Create the LLM
llm = ChatOpenAI(model="gpt-4o")


# Helper function to create a worker agent
def make_worker_agent(role: str, tools: List):
    """Create a worker agent with the given role and tools."""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=make_system_prompt(role, tools)),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    def worker_node(state: State) -> Union[Dict, Command]:
        """Node for the worker agent."""
        messages = state["messages"]

        # Run the agent
        response = llm.invoke(
            prompt.format_messages(
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
                    content=response.content,
                    additional_kwargs={"tool_calls": tool_calls},
                )
            )
            for tool_result in tool_results:
                messages.append(AIMessage(content=str(tool_result)))

            # Continue to the supervisor
            return {"messages": messages}

        # If no tool calls, add the response to the messages
        messages.append(AIMessage(content=response.content))

        # Check if the agent has indicated it's done
        if "TASK COMPLETE" in response.content:
            return Command(goto="supervisor")

        # Otherwise, continue to the supervisor
        return {"messages": messages}

    return worker_node


# Helper function to create a supervisor agent
def make_supervisor_node(llm, worker_names: List[str]):
    """Create a supervisor agent that can delegate to the given workers."""
    worker_descriptions = "\n".join(
        [f"- {name}: A worker that can help with the task" for name in worker_names]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are a supervisor. Your job is to delegate tasks to the appropriate workers.
You have the following workers available:
{worker_descriptions}

To delegate a task to a worker, respond with "DELEGATE TO: <worker_name>".
If the task is complete, respond with "TASK COMPLETE".
"""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    def supervisor_node(state: State) -> Union[Dict, Command]:
        """Node for the supervisor agent."""
        messages = state["messages"]

        # Run the agent
        response = llm.invoke(
            prompt.format_messages(
                messages=messages,
            )
        )

        # Add the response to the messages
        messages.append(AIMessage(content=response.content))

        # Check if the agent wants to delegate to a worker
        if "DELEGATE TO:" in response.content:
            worker_name = response.content.split("DELEGATE TO:")[1].strip()
            if worker_name in worker_names:
                return Command(goto=worker_name)

        # Check if the agent has indicated it's done
        if "TASK COMPLETE" in response.content:
            return Command(goto=END)

        # Otherwise, continue to the supervisor
        return {"messages": messages}

    return supervisor_node


# Create the research team
def create_research_team():
    """Create the research team graph."""
    # Create the graph
    workflow = StateGraph(State)

    # Create the agents
    researcher = make_worker_agent("researcher", research_tools)

    # Add the nodes
    workflow.add_node("researcher", researcher)

    # Add the edges
    workflow.add_edge("researcher", END)

    # Set the entry point
    workflow.set_entry_point("researcher")

    # Compile the graph
    return workflow.compile()


# Create the writing team
def create_writing_team():
    """Create the writing team graph."""
    # Create the graph
    workflow = StateGraph(State)

    # Create the agents
    writer = make_worker_agent("writer", writing_tools)

    # Add the nodes
    workflow.add_node("writer", writer)

    # Add the edges
    workflow.add_edge("writer", END)

    # Set the entry point
    workflow.set_entry_point("writer")

    # Compile the graph
    return workflow.compile()


# Create the research and writing teams
research_graph = create_research_team()
writing_graph = create_writing_team()


# Function to call research team
def call_research_team(state: State) -> Union[Dict, Command]:
    """Node for calling the research team."""
    response = research_graph.invoke({"messages": state["messages"]})
    return {"messages": response["messages"], "next": "supervisor"}


# Function to call writing team
def call_writing_team(state: State) -> Union[Dict, Command]:
    """Node for calling the writing team."""
    response = writing_graph.invoke({"messages": state["messages"]})
    return {"messages": response["messages"], "next": "supervisor"}


# Create the top-level supervisor
def create_top_level_supervisor():
    """Create the top-level supervisor graph."""
    # Create the graph
    workflow = StateGraph(State)

    # Create the supervisor
    supervisor = make_supervisor_node(llm, ["research_team", "writing_team"])

    # Add the nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("research_team", call_research_team)
    workflow.add_node("writing_team", call_writing_team)

    # Add the edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "research_team")
    workflow.add_edge("supervisor", "writing_team")
    workflow.add_edge("research_team", "supervisor")
    workflow.add_edge("writing_team", "supervisor")

    # Add conditional edges for ending the graph
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: END if "TASK COMPLETE" in x["messages"][-1].content else x["next"],
        {"research_team": "research_team", "writing_team": "writing_team", END: END},
    )

    # Set the entry point
    workflow.set_entry_point("supervisor")

    # Compile the graph
    return workflow.compile()


# Create the top-level supervisor
super_graph = create_top_level_supervisor()
