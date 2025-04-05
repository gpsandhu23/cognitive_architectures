"""Define the state structures for the reflection agent.

This agent uses a two-node system:
1. Generation node - produces initial responses
2. Reflection node - evaluates and potentially improves responses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the reflection agent.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the conversation state of the agent.

    The message sequence follows this pattern:
    1. HumanMessage - user's question
    2. AIMessage - initial response from generation node
    3. HumanMessage - reflection critique (if needed)
    4. AIMessage - improved response (if reflection suggested improvements)
    """


@dataclass
class State(InputState):
    """Represents the complete state of the reflection agent.

    Extends InputState with reflection-specific attributes and tracking.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.
    """

    needs_reflection: bool = field(default=True)
    """
    Indicates whether the current response needs reflection.
    Set to True after generation, False after reflection if no improvements needed.
    """

    original_question: Optional[str] = field(default=None)
    """
    Stores the original user question for reference during reflection.
    """

    reflection_count: int = field(default=0)
    """
    Tracks the number of reflection cycles to prevent infinite loops.
    """

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
