"""Configuration for the reflection agent."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from reflection_agent.prompts import GENERATION_PROMPT, REFLECTION_PROMPT


class Configuration(BaseModel):
    """Configuration for the reflection agent."""

    model: str = Field(
        default="anthropic/claude-3-5-sonnet-20240620",
        description="The model to use for generation and reflection.",
    )
    """The model to use for generation and reflection."""

    system_prompt: str = Field(
        default=GENERATION_PROMPT,
        description="The system prompt for the generation node.",
    )
    """The system prompt for the generation node."""

    reflection_prompt: str = Field(
        default=REFLECTION_PROMPT,
        description="The system prompt for the reflection node.",
    )
    """The system prompt for the reflection node."""

    max_reflections: int = Field(
        default=2,
        description="Maximum number of reflection cycles before ending.",
    )
    """Maximum number of reflection cycles before ending."""

    temperature: float = Field(
        default=0.7,
        description="Temperature for model sampling.",
    )
    """Temperature for model sampling."""

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> Configuration:
        """Create a Configuration from a RunnableConfig.

        Args:
            config: The RunnableConfig to create the Configuration from.

        Returns:
            A Configuration instance.
        """
        return cls(**config.get("configurable", {}))
