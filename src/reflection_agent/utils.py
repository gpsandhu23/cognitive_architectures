"""Utility functions for the reflection agent."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from reflection_agent.configuration import Configuration


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(config: Configuration) -> BaseChatModel:
    """Load a chat model based on the configuration.

    Args:
        config: The configuration containing model settings.

    Returns:
        A chat model instance.
    """
    # Extract provider and model name from the config.model string
    provider, model_name = config.model.split(":", 1)

    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=config.temperature,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")


def extract_last_response(messages: list[Any]) -> str | None:
    """Extract the last AI response from the message history.

    Args:
        messages: List of messages from the conversation.

    Returns:
        The content of the last AI message, or None if no AI message is found.
    """
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content")
    return None
