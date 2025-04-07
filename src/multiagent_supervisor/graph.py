from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing_extensions import TypedDict

# Initialize tools
tavily_tool = TavilySearchResults(max_results=5)


# Define team members and options
members = ["joke_setup", "joke_punchline"]
options = members + ["FINISH"]

# System prompt for the supervisor
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


# Initialize the LLM
llm = ChatAnthropic(model="claude-3-sonnet-20240229")


class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto, update={"next": goto})


# Create the joke setup agent
joke_setup_agent = create_react_agent(
    llm,
    tools=[],
    prompt="Your job is to generate a joke setup, but no punchline ever.",
)


def joke_setup_node(state: State) -> Command[Literal["supervisor"]]:
    result = joke_setup_agent.invoke(state)
    return Command(
        update={
            "messages": [
                {
                    "role": "human",
                    "content": result["messages"][-1].content,
                    "name": "joke_setup",
                }
            ]
        },
        goto="supervisor",
    )


# Create the joke punchline agent
joke_punchline_agent = create_react_agent(
    llm,
    tools=[],
    prompt="Your job is to generate a joke punchline, but no setup ever.",
)


def joke_punchline_node(state: State) -> Command[Literal["supervisor"]]:
    result = joke_punchline_agent.invoke(state)
    return Command(
        update={
            "messages": [
                {
                    "role": "human",
                    "content": result["messages"][-1].content,
                    "name": "joke_punchline",
                }
            ]
        },
        goto="supervisor",
    )


# Build the graph
def build_graph():
    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("joke_setup", joke_setup_node)
    builder.add_node("joke_punchline", joke_punchline_node)
    builder.add_edge("supervisor", "joke_setup")
    builder.add_edge("supervisor", "joke_punchline")
    builder.add_edge("joke_setup", "supervisor")
    builder.add_edge("joke_punchline", "supervisor")
    builder.add_edge("supervisor", END)
    return builder.compile()


# Create the graph instance
graph = build_graph()
