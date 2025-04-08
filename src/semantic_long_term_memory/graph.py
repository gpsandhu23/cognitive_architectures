from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool

model = "anthropic:claude-3-7-sonnet-latest"

namespace = "jarvis"

# Create memory tools
memory_tools = [
    create_manage_memory_tool(namespace),
    create_search_memory_tool(namespace),
]


# Create and return the agent with memory capabilities
graph = create_react_agent(
    model,
    prompt="""
    You are a helpful AI assistant named JARVIS.
    Always use the create memory tool to create a new memory when you need to remember something about the user.
    Always use the search memory tool first before answering a question.
    """,
    tools=memory_tools,
)
